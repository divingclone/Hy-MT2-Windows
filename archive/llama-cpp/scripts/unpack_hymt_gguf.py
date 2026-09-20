"""Restore standard Hy-MT2 projection tensors without requantizing any weights.

The input must use this project's packed Hy-MT2-1.8B layout. QKV/QK and
gate/up are split on logical row boundaries; every output tensor is verified
against its exact input byte slice. All metadata except the private packed
layout marker is retained. Per-layer tensor names are restored in lexical
order, as in the source models used by this project. Tensor directory order
and padding are not recoverable in general, so whole-file identity is reported
separately from tensor-byte identity.

Requires numpy and the pinned upstream gguf-py directory. No CUDA is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKED_KEY = "hunyuan-dense.hymt_packed_projections"
EXPECTED_METADATA = {
    "general.architecture": "hunyuan-dense",
    "hunyuan-dense.embedding_length": 2048,
    "hunyuan-dense.block_count": 32,
    "hunyuan-dense.feed_forward_length": 6144,
    "hunyuan-dense.attention.head_count": 16,
    "hunyuan-dense.attention.head_count_kv": 4,
}


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def byte_sha256(data) -> str:
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def load_gguf(directory: Path):
    directory = directory.resolve()
    if not (directory / "gguf" / "__init__.py").is_file():
        raise FileNotFoundError(f"Expected gguf-py directory: {directory}")
    sys.path.insert(0, str(directory))
    gguf = importlib.import_module("gguf")
    if Path(gguf.__file__).resolve().parent != directory / "gguf":
        raise RuntimeError("A different gguf module is already loaded")
    return gguf


def unpack(source: Path, output: Path, gguf_python: Path,
           expected_sha256: str | None = None) -> dict:
    gguf = load_gguf(gguf_python)
    source, output = source.resolve(), output.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest_path = output.with_suffix(".manifest.json")
    for target in (output, temporary, manifest_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Expected SHA256 must contain 64 hexadecimal characters")
    reader = gguf.GGUFReader(source)
    fields = reader.fields
    for key, expected in EXPECTED_METADATA.items():
        if key not in fields or fields[key].contents() != expected:
            raise ValueError(f"Not the expected Hy-MT2-1.8B layout: {key}")
    if PACKED_KEY not in fields or fields[PACKED_KEY].contents() is not True:
        raise ValueError("Input does not declare packed Hy-MT projections")
    if "split.count" in fields and fields["split.count"].contents() != 1:
        raise ValueError("Merge GGUF shards before unpacking")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("Duplicate tensor names")

    # Each replacement describes a view of the original rows, never a cast.
    replacements = {}
    modes = Counter(qkv=0, qk=0, separate_qkv=0, gate_up=0, separate_ffn=0)

    def require_shape(name, rows, columns=2048):
        tensor = tensors.get(name)
        if tensor is None or tuple(map(int, tensor.shape)) != (columns, rows):
            raise ValueError(f"Unexpected or missing tensor: {name}; expected [{columns}, {rows}]")
        if len(tensor.data.shape) != 2 or tensor.data.shape[0] != rows:
            raise ValueError(f"Tensor does not expose contiguous rows: {name}")
        if not tensor.data.flags.c_contiguous or tensor.data.nbytes != tensor.n_bytes:
            raise ValueError(f"Invalid contiguous byte representation: {name}")
        return tensor

    for layer in range(32):
        prefix = f"blk.{layer}."
        q, k, v = (prefix + name + ".weight" for name in ("attn_q", "attn_k", "attn_v"))
        qkv = prefix + "attn_qkv.weight"
        gate, up = (prefix + name + ".weight" for name in ("ffn_gate", "ffn_up"))
        if any(prefix + name + suffix in tensors
               for name in ("attn_q", "attn_k", "attn_v", "attn_qkv", "ffn_gate", "ffn_up")
               for suffix in (".bias", ".scale", ".input_scale")):
            raise ValueError(f"Projection biases/external scales are unsupported: {prefix}")
        if qkv in tensors:
            if any(name in tensors for name in (q, k, v)):
                raise ValueError(f"Ambiguous packed and separate QKV: {prefix}")
            require_shape(qkv, 3072)
            replacements[qkv] = [(q, 0, 2048), (k, 2048, 512), (v, 2560, 512)]
            modes["qkv"] += 1
        elif q in tensors and tuple(map(int, tensors[q].shape)) == (2048, 2560):
            if k in tensors:
                raise ValueError(f"Ambiguous packed and separate QK: {prefix}")
            require_shape(q, 2560)
            require_shape(v, 512)
            replacements[q] = [(q, 0, 2048), (k, 2048, 512)]
            modes["qk"] += 1
        else:
            for name, rows in ((q, 2048), (k, 512), (v, 512)):
                require_shape(name, rows)
            modes["separate_qkv"] += 1
        if up in tensors and tuple(map(int, tensors[up].shape)) == (2048, 12288):
            if gate in tensors:
                raise ValueError(f"Ambiguous packed and separate FFN: {prefix}")
            require_shape(up, 12288)
            replacements[up] = [(gate, 0, 6144), (up, 6144, 6144)]
            modes["gate_up"] += 1
        else:
            require_shape(gate, 6144)
            require_shape(up, 6144)
            modes["separate_ffn"] += 1

    entries = []
    for tensor in reader.tensors:
        if tensor.name not in replacements:
            entries.append((tensor.name, tensor, tensor.data, 0, tuple(map(int, tensor.shape))))
            continue
        row_bytes = tensor.n_bytes // int(tensor.shape[1])
        for name, start, count in replacements[tensor.name]:
            data = tensor.data[start:start + count]
            if data.nbytes != count * row_bytes:
                raise ValueError(f"Incomplete byte slice: {name}")
            entries.append((name, tensor, data, start * row_bytes, (2048, count)))

    # Packing discards positions of later components. Restore the known source
    # models' per-layer lexical order; keep non-layer tensors in their order.
    layer_entries = {}
    for entry in entries:
        match = re.match(r"^blk\.(\d+)\.", entry[0])
        if match:
            layer_entries.setdefault(int(match[1]), []).append(entry)
    ordered, emitted_layers = [], set()
    for entry in entries:
        match = re.match(r"^blk\.(\d+)\.", entry[0])
        if not match:
            ordered.append(entry)
        elif int(match[1]) not in emitted_layers:
            index = int(match[1])
            ordered.extend(sorted(layer_entries[index], key=lambda item: item[0]))
            emitted_layers.add(index)
    if len({entry[0] for entry in ordered}) != len(ordered):
        raise ValueError("Unpacking would create duplicate tensor names")
    if sum(entry[2].nbytes for entry in ordered) != sum(tensor.n_bytes for tensor in reader.tensors):
        raise ValueError("Unpacking would lose or duplicate source bytes")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(temporary, "hunyuan-dense", endianess=reader.endianess)
    writer.data_alignment = reader.alignment
    for field in fields.values():
        if field.name in ("general.architecture", PACKED_KEY) or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=subtype)
    for name, tensor, data, _, _ in ordered:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for _, _, data, _, _ in ordered:
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    finally:
        writer.close()

    restored = gguf.GGUFReader(temporary)
    try:
        restored_tensors = {tensor.name: tensor for tensor in restored.tensors}
        if len(restored_tensors) != len(ordered):
            raise ValueError("Output tensor count differs")
        verification = []
        for name, tensor, data, offset, shape in ordered:
            actual = restored_tensors[name]
            if actual.tensor_type != tensor.tensor_type or tuple(map(int, actual.shape)) != shape:
                raise ValueError(f"Output tensor type/shape differs: {name}")
            expected_digest, actual_digest = byte_sha256(data), byte_sha256(actual.data)
            if actual.n_bytes != data.nbytes or actual_digest != expected_digest:
                raise ValueError(f"Output tensor bytes differ: {name}")
            verification.append({"source": tensor.name, "output": name, "source_offset_bytes": offset,
                                 "bytes": data.nbytes, "shape": list(shape),
                                 "tensor_type": tensor.tensor_type.name, "sha256": actual_digest})
        metadata = {name for name in fields if not name.startswith("GGUF.") and name != PACKED_KEY}
        if {name for name in restored.fields if not name.startswith("GGUF.")} != metadata:
            raise ValueError("Output metadata keys differ")
        for name in metadata:
            if (fields[name].types != restored.fields[name].types or
                    fields[name].contents() != restored.fields[name].contents()):
                raise ValueError(f"Output metadata changed: {name}")
    finally:
        # Closing the mapped output explicitly is necessary before Windows rename.
        restored.data._mmap.close()
    os.replace(temporary, output)
    output_digest = file_sha256(output)
    manifest = {
        "schema_version": 1, "input": str(source), "output": str(output),
        "input_sha256": file_sha256(source), "output_sha256": output_digest,
        "output_bytes": output.stat().st_size, "quantization_changed": False,
        "all_tensor_slices_verified": True, "metadata_verified_except_removed_key": PACKED_KEY,
        "tensor_order": "per-layer lexical; non-layer source order",
        "modes": dict(modes), "packed_tensors": len(reader.tensors), "output_tensors": len(ordered),
        "verified_tensor_bytes": sum(item["bytes"] for item in verification),
        "tensors": verification,
    }
    if expected_sha256 is not None:
        manifest["expected_output_sha256"] = expected_sha256.lower()
        manifest["whole_file_matches_expected"] = output_digest == expected_sha256.lower()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "tensors"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-python", type=Path, default=ROOT / "src/llama.cpp/gguf-py")
    parser.add_argument("--expected-sha256", help="Original whole-file SHA256; match/mismatch is recorded, not enforced")
    args = parser.parse_args()
    print(json.dumps(unpack(args.input, args.output, args.gguf_python, args.expected_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
