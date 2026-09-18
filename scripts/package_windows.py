"""Build an allowlisted Windows NVIDIA runtime package; models download separately.

The --dry-run mode is read-only and tolerates artifacts still being built.
The actual build requires every planned file and a closed PE dependency graph.
Use --include-models only for a larger offline package, not a GitHub release asset.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXES = ("hy-batch.exe", "llama-server.exe")
NATIVE_DLLS = (
    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "llama.dll",
    "llama-common.dll", "llama-server-impl.dll", "mtmd.dll",
)
CUDA_DLLS = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")
MSVC_DLLS = (
    "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vccorlib140.dll",
    "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
)
SCRIPTS = (
    "translate_batch.py", "translate.py", "serve.ps1",
    "stop-server.ps1", "serve.py", "gpu_config.py", "run-python.cmd", "setup_model.py",
    "run_native_experiment.py", "benchmark_cases.json", "benchmark_upstream.py",
    "summarize_upstream_benchmark.py", "unpack_hymt_gguf.py",
)
LAUNCHERS = ("translate-batch.cmd", "start-server.cmd", "stop-server.cmd", "gpu-info.cmd", "setup-model.cmd")
MODELS = ("Hy-MT2-1.8B-Q4_K_M-fused.gguf", "Hy-MT2-1.8B-NVFP4-fused.gguf")
PYTHON_ROOT_FILES = (
    "python.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "LICENSE.txt", "BUILD",
)
PYTHON_EXTENSIONS = (
    "_asyncio.pyd", "_bz2.pyd", "_ctypes.pyd", "_decimal.pyd", "_elementtree.pyd",
    "_hashlib.pyd", "_lzma.pyd", "_msi.pyd", "_multiprocessing.pyd", "_overlapped.pyd",
    "_queue.pyd", "_socket.pyd", "_sqlite3.pyd", "_ssl.pyd", "_uuid.pyd", "_wmi.pyd",
    "_zoneinfo.pyd", "pyexpat.pyd", "select.pyd", "unicodedata.pyd", "winsound.pyd",
    "libcrypto-3-x64.dll", "libffi-8.dll", "libssl-3-x64.dll", "sqlite3.dll",
)
# Top-level standard-library packages are explicit. No site-packages, pip,
# venv, Tk, test suites, SDK import libraries, caches, or developer tools.
PYTHON_PACKAGES = frozenset((
    "asyncio", "collections", "concurrent", "ctypes", "curses", "dbm", "email", "encodings",
    "html", "http", "importlib", "json", "lib2to3", "logging", "msilib", "multiprocessing",
    "pydoc_data", "re", "sqlite3", "tomllib", "unittest", "urllib", "wsgiref", "xml",
    "xmlrpc", "zipfile", "zoneinfo",
))
LICENSE_FILES = (
    "SOURCES.json", "THIRD_PARTY_NOTICES.md", "model-Hy-MT2.txt", "MODEL_CHANGES.md",
    "openssl-LICENSE.txt", "libffi-LICENSE.txt", "msvc-runtime-license.docx",
    "msvc-runtime-license.txt", "msvc-redistribution.html", "python-3.12-license.html",
    "cccl-LICENSE.txt", "stb-LICENSE.txt", "subprocess-LICENSE.txt", "sha1-NOTICE.txt",
    "sqlite-PUBLIC-DOMAIN.html", "xz-COPYING.txt", "xz-0BSD.txt",
)
SYSTEM_DLLS = frozenset((
    "advapi32.dll", "authz.dll", "bcrypt.dll", "bcryptprimitives.dll", "cabinet.dll",
    "cfgmgr32.dll", "combase.dll", "comctl32.dll", "comdlg32.dll", "crypt32.dll",
    "cryptbase.dll", "cryptsp.dll", "dbghelp.dll", "dnsapi.dll", "dwmapi.dll", "gdi32.dll",
    "imagehlp.dll", "imm32.dll", "iphlpapi.dll", "kernel32.dll", "kernelbase.dll", "msi.dll", "msvcrt.dll",
    "ncrypt.dll", "netapi32.dll", "normaliz.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
    "powrprof.dll", "propsys.dll", "psapi.dll", "rpcrt4.dll", "secur32.dll", "setupapi.dll",
    "shell32.dll", "shcore.dll", "shlwapi.dll", "ucrtbase.dll", "user32.dll", "userenv.dll",
    "usp10.dll", "version.dll", "winhttp.dll", "wininet.dll", "winmm.dll", "winspool.drv",
    "wintrust.dll", "wldap32.dll", "ws2_32.dll", "wtsapi32.dll",
))
DRIVER_DLLS = frozenset(("nvcuda.dll", "nvml.dll"))


@dataclass(frozen=True)
class Entry:
    source: Path
    relative: str
    component: str


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") or ":" in p for p in path.parts):
        raise ValueError(f"Unsafe archive path: {value}")
    return path.as_posix()


def pe_imports(path: Path) -> list[str]:
    """Read normal and delay-load DLL names from an AMD64 PE, without executing it."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def u16(offset): return struct.unpack_from("<H", data, offset)[0]
        def u32(offset): return struct.unpack_from("<I", data, offset)[0]
        def u64(offset): return struct.unpack_from("<Q", data, offset)[0]
        if data[:2] != b"MZ": raise ValueError(f"Not a PE file: {path.name}")
        pe = u32(0x3C)
        if data[pe:pe + 4] != b"PE\0\0" or u16(pe + 4) != 0x8664:
            raise ValueError(f"Expected Windows x64 PE: {path.name}")
        optional = pe + 24
        if u16(optional) != 0x20B: raise ValueError(f"Expected PE32+: {path.name}")
        base = u64(optional + 24)
        directories = optional + 112
        section_start = optional + u16(pe + 20)
        sections = []
        for index in range(u16(pe + 6)):
            offset = section_start + 40*index
            sections.append((u32(offset + 12), u32(offset + 8), u32(offset + 20), u32(offset + 16)))

        def rva_offset(rva):
            if rva < u32(optional + 60): return rva
            for address, virtual_size, raw, raw_size in sections:
                if address <= rva < address + max(virtual_size, raw_size):
                    result = raw + rva - address
                    if result >= raw + raw_size or result >= len(data): break
                    return result
            raise ValueError(f"Invalid PE RVA in {path.name}: {rva}")

        def dll_name(rva):
            offset = rva_offset(rva)
            end = data.find(b"\0", offset, min(offset + 1024, len(data)))
            if end < 0: raise ValueError(f"Unterminated DLL name: {path.name}")
            name = data[offset:end].decode("ascii").lower()
            if "/" in name or "\\" in name or ":" in name:
                raise ValueError(f"Unexpected import path: {name}")
            return name

        names = set()
        for directory, size in ((1, 20), (13, 32)):
            if u32(optional + 108) <= directory: continue
            rva, total = struct.unpack_from("<II", data, directories + directory*8)
            if not rva or not total: continue
            offset = rva_offset(rva)
            for index in range(total // size + 1):
                pos = offset + index*size
                descriptor = data[pos:pos + size]
                if len(descriptor) != size: raise ValueError(f"Truncated import table: {path.name}")
                if not any(descriptor): break
                name_rva = u32(pos + (12 if directory == 1 else 4))
                if directory == 13 and not (u32(pos) & 1): name_rva -= base
                names.add(dll_name(name_rva))
            else:
                raise ValueError(f"Unterminated import table: {path.name}")
        return sorted(names)


def make_plan(root: Path, binary_dir: Path, python_dir: Path, include_models: bool = False) -> tuple[list[Entry], list[str], dict]:
    entries: dict[str, Entry] = {}
    problems = []
    def add(source: Path, relative: str, component: str):
        relative = safe_relative(relative)
        key = relative.casefold()
        if key in entries: raise ValueError(f"Duplicate package path: {relative}")
        # The chosen Python root may be a junction, but all descendants must be
        # ordinary files. This prevents unexpected links escaping an allowlist.
        if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
            raise ValueError(f"Unexpected linked file: {relative}")
        entries[key] = Entry(source, relative, component)
        if not source.is_file(): problems.append(f"Missing: {relative}")

    for name in LAUNCHERS: add(root/name, name, "launcher")
    for name in SCRIPTS: add(root/"scripts"/name, f"scripts/{name}", "script")
    add(root/"README.md", "README.md", "documentation")
    add(root/"LICENSE", "LICENSE", "project license")
    for directory, suffixes in (("docs", {".md"}), ("benchmarks", {".json", ".svg", ".md"}),
                                ("huggingface", {".md", ".json"}), ("patches", {".patch", ".json"})):
        for path in sorted((root/directory).glob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                if path.stat().st_size > 2 * 1024**2:
                    problems.append(f"Documentation/evidence file exceeds 2 MiB allowlist limit: {path.name}")
                    continue
                add(path, f"{directory}/{path.name}", "documentation" if directory == "docs" else "benchmark evidence")
    add(binary_dir/"build-info.json", "bin/build-info.json", "build provenance")
    add(root/"models/manifest.json", "models/manifest.json", "model download manifest")
    add(root/"models/README.md", "models/README.md", "model documentation")
    if include_models:
        for name in MODELS: add(root/"models"/name, f"models/{name}", "model")
    add(root/"examples/input.jsonl", "examples/input.jsonl", "example")
    for name in LICENSE_FILES: add(root/"licenses"/name, f"licenses/{name}", "license")
    for source, destination in (
        ("src/llama.cpp/LICENSE", "llama.cpp-LICENSE.txt"),
        ("src/llama.cpp/licenses/LICENSE-jsonhpp", "jsonhpp-LICENSE.txt"),
        ("src/llama.cpp/vendor/cpp-httplib/LICENSE", "cpp-httplib-LICENSE.txt"),
        ("src/llama.cpp/vendor/hash/xxhash/LICENSE", "xxhash-LICENSE.txt"),
        ("src/llama.cpp/vendor/hash/sha256/LICENSE", "sha256-LICENSE.txt"),
        ("src/llama.cpp/vendor/hash/rotate-bits/LICENSE.md", "rotate-bits-LICENSE.md"),
    ):
        local_license = root/"licenses"/destination
        add(local_license if local_license.is_file() else root/source, f"licenses/{destination}", "license")
    cuda_license = root/"licenses/NVIDIA-CUDA-EULA.txt"
    if not cuda_license.is_file(): cuda_license = root/"runtime/toolchain/cuda/EULA.txt"
    add(cuda_license, "licenses/NVIDIA-CUDA-EULA.txt", "license")
    for name in PYTHON_ROOT_FILES: add(python_dir/name, f"runtime/python/{name}", "Python")
    for name in PYTHON_EXTENSIONS: add(python_dir/"DLLs"/name, f"runtime/python/DLLs/{name}", "Python extension")
    lib = python_dir/"Lib"
    if not lib.is_dir(): problems.append("Missing Python Lib directory")
    else:
        for current, dirs, files in os.walk(lib, followlinks=False):
            directory = Path(current)
            rel_dir = directory.relative_to(lib)
            dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "test", "tests") and
                             (rel_dir.parts or d in PYTHON_PACKAGES))
            for d in dirs:
                child = directory/d
                if child.is_symlink() or child.is_junction(): raise ValueError(f"Linked stdlib directory: {child.name}")
            for name in sorted(files):
                if Path(name).suffix not in (".py", ".txt", ".pickle") and name != "EXTERNALLY-MANAGED": continue
                source = directory/name
                add(source, "runtime/python/Lib/" + source.relative_to(lib).as_posix(), "Python standard library")

    candidates: dict[str, tuple[Path, str, str]] = {}
    for names, directory, dest, component in (
        (NATIVE_DLLS, binary_dir, "bin", "llama.cpp runtime"),
        (CUDA_DLLS, root/"runtime/cuda", "runtime/cuda", "NVIDIA runtime"),
        (MSVC_DLLS, root/"runtime/msvc", "runtime/msvc", "MSVC runtime"),
    ):
        for name in names: candidates[name.lower()] = (directory/name, f"{dest}/{name}", component)
    # Backend libraries and the subprocess implementation are loaded dynamically;
    # PE imports alone cannot find them. All other native DLLs require a PE edge.
    for name in (*EXES, "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "llama-server-impl.dll"):
        add(binary_dir/name, f"bin/{name}", "llama.cpp runtime")
    # Include the small established runtime closure in a pre-build dry-run too.
    # Actual PE parsing below still verifies dependencies rather than trusting it.
    if not all((binary_dir/name).is_file() for name in EXES):
        for name in NATIVE_DLLS:
            if f"bin/{name}".casefold() not in entries: add(binary_dir/name, f"bin/{name}", "llama.cpp runtime")
        for name in CUDA_DLLS:
            source, dest, component = candidates[name.lower()]
            add(source, dest, component)
    imports = {}
    inspected = set()
    while True:
        pending = [e for e in entries.values() if e.relative not in inspected and e.source.suffix.lower() in (".exe", ".dll", ".pyd")]
        if not pending: break
        for entry in pending:
            inspected.add(entry.relative)
            if not entry.source.is_file(): continue
            try: names = pe_imports(entry.source)
            except (ValueError, struct.error) as exc:
                problems.append(f"PE audit failed for {entry.relative}: {exc}")
                continue
            imports[entry.relative] = names
            for name in names:
                if name in SYSTEM_DLLS or name in DRIVER_DLLS or name.startswith(("api-ms-win-", "ext-ms-win-")): continue
                # Python resolves its runtime DLLs beside python.exe or in DLLs;
                # native executables resolve package bin and runtime PATH dirs.
                compatible = [e for e in entries.values() if Path(e.relative).name.lower() == name and
                              (entry.relative.startswith("runtime/python/") or not e.relative.startswith("runtime/python/"))]
                if compatible: continue
                if name not in candidates:
                    problems.append(f"Unresolved non-system dependency: {entry.relative} -> {name}")
                    continue
                source, destination, component = candidates[name]
                if destination.casefold() not in entries: add(source, destination, component)
    return sorted(entries.values(), key=lambda e: e.relative.casefold()), sorted(set(problems)), imports


def verify_python(package: Path):
    """Run only the copied interpreter; assert imports resolve under its new root."""
    code = (
        "import sys,json,pathlib,ssl,ctypes,urllib.request,concurrent.futures,sqlite3,bz2,lzma; "
        "expected=pathlib.Path(sys.argv[1]).resolve()/'runtime/python'; "
        "root=pathlib.Path(sys.executable).resolve().parent; "
        "assert root==expected; "
        "assert pathlib.Path(sys.prefix).resolve()==root; "
        "assert pathlib.Path(sys.base_prefix).resolve()==root; "
        "assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in sys.path); "
        "modules=(ssl,ctypes,urllib.request,concurrent.futures,sqlite3,bz2,lzma); "
        "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in modules); "
        "print(json.dumps({'version':sys.version.split()[0],'isolated':bool(sys.flags.isolated),"
        "'stdlib_imports':'ok','executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'search_path':sys.path,'module_files':{m.__name__:m.__file__ for m in modules}}))"
    )
    env = dict(os.environ)
    for key in list(env):
        if key.upper().startswith("PYTHON"): env.pop(key)
    result = subprocess.run([str(package/"runtime/python/python.exe"), "-I", "-S", "-B", "-c", code, str(package.resolve())],
                            cwd=package, env=env, capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        detail = result.stderr[-2000:].replace(str(package.resolve()), "<package>")
        raise RuntimeError("Relocated Python failed: " + detail)
    return normalize_python_check(json.loads(result.stdout), package)


def normalize_python_check(check: dict, package: Path) -> dict:
    """Keep relocation evidence without publishing an original/staging path."""
    package = package.resolve()
    runtime = package/"runtime/python"

    def relative(value):
        path = Path(value).resolve()
        if not path.is_relative_to(runtime):
            raise ValueError("Relocated Python reported a path outside the packaged runtime")
        return safe_relative(path.relative_to(package).as_posix())

    # Select the allowed fields, rather than serializing arbitrary child-process
    # metadata. A future extra field cannot silently leak an absolute host path.
    result = {
        "version": check["version"], "isolated": check["isolated"],
        "stdlib_imports": check["stdlib_imports"], "path_base": "package root",
        "executable": relative(check["executable"]),
        "prefix": relative(check["prefix"]), "base_prefix": relative(check["base_prefix"]),
        "search_path": [relative(path) for path in check["search_path"]],
        "module_files": {name: relative(path) for name, path in check["module_files"].items()},
    }
    if result["executable"] != "runtime/python/python.exe" or result["prefix"] != "runtime/python" or result["base_prefix"] != "runtime/python":
        raise ValueError("Relocated Python did not use the expected interpreter root")
    return result


def report_path(path: Path, root: Path) -> str:
    """Console JSON can be retained as provenance without exposing a user home."""
    path, root = path.resolve(), root.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def create_package(entries: list[Entry], imports: dict, output: Path, root: Path):
    archive = output.with_name(output.name + ".zip")
    if output.exists() or archive.exists(): raise FileExistsError("Output directory or ZIP already exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)).resolve()
    package = stage/output.name
    staged_zip = stage/(output.name + ".zip")
    published_zip = False
    try:
        package.mkdir()
        records = []
        for entry in entries:
            target = package/entry.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = entry.source.stat()
            shutil.copyfile(entry.source, target)
            source_hash = sha256(entry.source)
            after = entry.source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(target) != source_hash:
                raise RuntimeError(f"Source changed during packaging: {entry.relative}")
            records.append({"path": entry.relative, "size": after.st_size, "sha256": source_hash, "component": entry.component})
        build_info = json.loads((package/"bin/build-info.json").read_text(encoding="utf-8-sig"))
        if str(root).replace("\\", "/").lower() in json.dumps(build_info).replace("\\\\", "/").lower():
            raise ValueError("build-info.json contains the private workspace path; use portable provenance")
        python_check = verify_python(package)
        model_manifest = json.loads((package/"models/manifest.json").read_text(encoding="utf-8-sig"))
        for record in records:
            if record["component"] != "model": continue
            expected = next((item for item in model_manifest.get("files", {}).values()
                             if item.get("filename") == Path(record["path"]).name), None)
            if not expected or expected.get("size_bytes") != record["size"] or expected.get("sha256", "").lower() != record["sha256"]:
                raise ValueError(f"Offline model does not match models/manifest.json: {record['path']}")
        manifest = {
            "format": 1, "package": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": records, "pe_imports": imports, "python_relocation_check": python_check,
            "includes_models": any(entry.component == "model" for entry in entries),
            "model_setup": "setup-model.cmd",
            "external_dependencies": ["Windows x64 system libraries", "NVIDIA driver supporting CUDA 13"],
            "excluded": ["SDK", "source checkout", "git", "venv", "site-packages", "results", "cache", "downloads", "PDB"],
        }
        (package/"MANIFEST.sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (package/"SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in records) +
                                            f"{sha256(package/'MANIFEST.sha256.json')}  MANIFEST.sha256.json\n", encoding="utf-8")
        # Store already-quantized model weights directly. Deflate small text and
        # executable files; ZIP64 permits packages larger than 4 GiB.
        with zipfile.ZipFile(staged_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zip_file:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    relative = safe_relative(output.name + "/" + path.relative_to(package).as_posix())
                    zip_file.write(path, relative, compress_type=zipfile.ZIP_STORED if path.suffix == ".gguf" else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(staged_zip) as zip_file:
            if zip_file.testzip() is not None: raise RuntimeError("ZIP CRC verification failed")
        zip_bytes = staged_zip.stat().st_size
        release_compatible = zip_bytes < 2 * 1024**3
        if not manifest["includes_models"] and not release_compatible:
            raise RuntimeError("Runtime ZIP exceeds GitHub's per-asset limit of less than 2 GiB")
        # Each rename publishes a finished artifact, never a partially copied
        # directory. Windows rename fails if a destination appeared meanwhile.
        if output.exists() or archive.exists(): raise FileExistsError("Output appeared while staging")
        staged_zip.rename(archive)
        published_zip = True
        package.rename(output)
        published_zip = False
        return {"directory": report_path(output, root), "zip": report_path(archive, root), "zip_sha256": sha256(archive),
                "files": len(records) + 2, "payload_bytes": sum(r["size"] for r in records), "python": python_check,
                "zip_bytes": zip_bytes, "includes_models": manifest["includes_models"],
                "github_release_compatible": release_compatible}
    except BaseException:
        if published_zip:
            # Only roll back the exact ZIP published by this invocation.
            archive.rename(staged_zip)
        raise
    finally:
        # The staging path is created here, resolved and confined to the explicit
        # output parent. No computed recursive operation touches a source tree.
        if stage.parent != output.parent.resolve() or not stage.name.startswith(f".{output.name}.stage-"):
            raise RuntimeError("Unsafe staging cleanup path")
        shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bin-dir", type=Path, default=Path("bin"))
    parser.add_argument("--python-dir", type=Path, help="Default: detect the bundled Python layout")
    parser.add_argument("--output", type=Path, help="Default: dist/HyMT-Windows-NVIDIA-runtime or -offline")
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument("--include-models", action="store_true", help="Include both GGUF files for offline transfer")
    model_options.add_argument("--runtime-only", action="store_true", help="Omit model weights; this is the default")
    parser.add_argument("--dry-run", action="store_true", help="Read-only inventory; report missing artifacts without failing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda path: (path if path.is_absolute() else root/path).resolve()
    python_dir = args.python_dir
    if python_dir is None:
        candidates = [Path("runtime/python"), Path("runtime/python/cpython-3.12-windows-x86_64-none")]
        python_dir = next((path for path in candidates if (root/path/"python.exe").is_file()), candidates[0])
    output = args.output or Path("dist/HyMT-Windows-NVIDIA-" + ("offline" if args.include_models else "runtime"))
    binary_dir, python_dir, output = map(resolve, (args.bin_dir, python_dir, output))
    if output == root or root.is_relative_to(output) or output.is_relative_to(binary_dir) or output.is_relative_to(python_dir):
        parser.error("output must be separate from source and runtime input directories")
    entries, problems, imports = make_plan(root, binary_dir, python_dir, args.include_models)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output": report_path(output, root), "includes_models": args.include_models,
                          "file_count": len(entries),
                          "known_bytes": sum(e.source.stat().st_size for e in entries if e.source.is_file()),
                          "missing_or_invalid": problems,
                          "files": [{"path": e.relative, "component": e.component,
                                     "size": e.source.stat().st_size if e.source.is_file() else None} for e in entries],
                          "pe_imports": imports}, ensure_ascii=False, indent=2))
        return 0
    if os.name != "nt": parser.error("Actual packaging and relocation validation require Windows")
    if problems: raise RuntimeError("Package inputs are incomplete:\n" + "\n".join(problems))
    print(json.dumps(create_package(entries, imports, output, root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Package error: {exc}", file=sys.stderr)
        raise SystemExit(1)
