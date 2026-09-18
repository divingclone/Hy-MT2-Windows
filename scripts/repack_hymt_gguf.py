"""Pack Hy-MT2-1.8B projections without changing any quantized weight bytes.

The result requires this workspace's modified llama.cpp. Matching Q/K/V types
use one QKV projection; mixed V precision uses QK plus V. Gate precedes up in
the fused FFN. LLAMA_HYMT_FUSED_PROJ=0 runs separate matmuls on the same file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "llama.cpp" / "gguf-py"))

import numpy as np
import gguf


PACKED_KEY = "hunyuan-dense.hymt_packed_projections"


def digest(data: np.ndarray) -> str:
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def compatible(parts: list[gguf.ReaderTensor]) -> bool:
    return all(
        part.tensor_type == parts[0].tensor_type
        and len(part.shape) == 2
        and part.shape[0] == parts[0].shape[0]
        for part in parts
    )


def repack(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    reader = gguf.GGUFReader(source)
    fields = reader.fields
    arch = fields["general.architecture"].contents()
    expected = {
        "general.architecture": "hunyuan-dense",
        "hunyuan-dense.embedding_length": 2048,
        "hunyuan-dense.block_count": 32,
        "hunyuan-dense.feed_forward_length": 6144,
        "hunyuan-dense.attention.head_count": 16,
        "hunyuan-dense.attention.head_count_kv": 4,
    }
    for key, value in expected.items():
        if key not in fields or fields[key].contents() != value:
            raise ValueError(f"Not the expected Hy-MT2-1.8B layout: {key}")
    if PACKED_KEY in fields:
        raise ValueError("Input is already packed")
    if "split.count" in fields and fields["split.count"].contents() != 1:
        raise ValueError("Merge GGUF shards before packing")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    groups: dict[str, list[gguf.ReaderTensor]] = {}
    removed: set[str] = set()
    modes = {"qkv": 0, "qk": 0, "separate_qkv": 0, "gate_up": 0, "separate_ffn": 0}
    for layer in range(32):
        prefix = f"blk.{layer}."
        q, k, v = [tensors[prefix + name + ".weight"] for name in ("attn_q", "attn_k", "attn_v")]
        gate, up = [tensors[prefix + name + ".weight"] for name in ("ffn_gate", "ffn_up")]
        if any(prefix + name + suffix in tensors for name in ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up")
               for suffix in (".bias", ".scale", ".input_scale")):
            raise ValueError("Projection biases and external quantization scales are unsupported")
        if compatible([q, k, v]):
            groups[prefix + "attn_qkv.weight"] = [q, k, v]
            removed.update(part.name for part in (q, k, v))
            modes["qkv"] += 1
        elif compatible([q, k]):
            groups[q.name] = [q, k]
            removed.update(part.name for part in (q, k))
            modes["qk"] += 1
        else:
            modes["separate_qkv"] += 1
        if compatible([gate, up]):
            groups[up.name] = [gate, up]
            removed.update(part.name for part in (gate, up))
            modes["gate_up"] += 1
        else:
            modes["separate_ffn"] += 1

    # Emit each group at its first source tensor's position to keep layer order.
    first_source = {parts[0].name: (name, parts) for name, parts in groups.items()}
    entries: list[tuple[str, list[gguf.ReaderTensor]]] = []
    for tensor in reader.tensors:
        if tensor.name in first_source:
            entries.append(first_source[tensor.name])
        elif tensor.name not in removed:
            entries.append((tensor.name, [tensor]))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    writer = gguf.GGUFWriter(temporary, arch, endianess=reader.endianess)
    writer.data_alignment = reader.alignment
    for field in fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=subtype)
    writer.add_bool(PACKED_KEY, True)
    for name, parts in entries:
        shape = parts[0].data.shape
        if len(parts) > 1:
            shape = (sum(part.data.shape[0] for part in parts), *shape[1:])
        writer.add_tensor_info(name, shape, parts[0].data.dtype,
                               sum(part.n_bytes for part in parts), parts[0].tensor_type)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for name, parts in entries:
            data = parts[0].data if len(parts) == 1 else np.concatenate([part.data for part in parts], axis=0)
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        writer.close()

        packed = gguf.GGUFReader(temporary)
        packed_tensors = {tensor.name: tensor for tensor in packed.tensors}
        verification = []
        for name, parts in entries:
            result = packed_tensors[name]
            if result.tensor_type != parts[0].tensor_type:
                raise ValueError(f"Quantization changed: {name}")
            raw = result.data.reshape(-1).view(np.uint8)
            offset = 0
            for part in parts:
                actual = digest(raw[offset:offset + part.n_bytes])
                original = digest(part.data)
                if actual != original:
                    raise ValueError(f"Packed bytes differ: {part.name}")
                verification.append({"source": part.name, "packed": name, "offset": offset,
                                     "bytes": part.n_bytes, "sha256": original})
                offset += part.n_bytes
            if offset != result.n_bytes:
                raise ValueError(f"Packed byte count differs: {name}")
        for key, field in fields.items():
            if not key.startswith("GGUF.") and field.contents() != packed.fields[key].contents():
                raise ValueError(f"Metadata changed: {key}")
        # Windows requires closing all mmap views before renaming the file.
        packed.data._mmap.close()
        del packed_tensors, packed, result, raw
        os.replace(temporary, output)
    finally:
        writer.close()

    manifest = {"input": str(source), "output": str(output), "quantization_changed": False,
                "modes": modes, "source_tensors": len(reader.tensors), "packed_tensors": len(entries),
                "verified_source_tensor_bytes": sum(row["bytes"] for row in verification),
                "tensors": verification}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "tensors"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(repack(args.input, args.output), indent=2))
