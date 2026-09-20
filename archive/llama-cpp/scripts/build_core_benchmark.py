"""Build an isolated instrumented hy-batch against the existing inference DLLs.

Run in an x64 MSVC developer environment. Neither production sources nor bin/
are edited. Requires the source/build tree prepared by docs/BUILD.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/core-benchmark")
    parser.add_argument("--build", type=Path, default=ROOT / "build/production")
    args = parser.parse_args()
    dest, build = args.output.resolve(), args.build.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    source = ROOT / "src/llama.cpp"
    copied = dest / "tools/hy-batch/hy-batch.cpp"
    copied.parent.mkdir(parents=True)
    original = source / "tools/hy-batch/hy-batch.cpp"
    shutil.copy2(original, copied)
    patch = ROOT / "patches/hy-batch-core-benchmark.patch"
    # No git index or working-tree modifications: apply only to the isolated copy.
    subprocess.run(["git", "apply", "--no-index", str(patch)], cwd=dest, check=True)
    includes = [source / p for p in ("common", "vendor", "include", "ggml/include")]
    libs = [build / p for p in ("common/llama-common.lib", "src/llama.lib",
            "ggml/src/ggml.lib", "ggml/src/ggml-cpu.lib", "ggml/src/ggml-cuda/ggml-cuda.lib",
            "ggml/src/ggml-base.lib", "common/llama-common-base.lib")]
    command = ["cl.exe", "/nologo", "/EHsc", "/O2", "/Ob2", "/DNDEBUG", "/std:c++17",
               "/MD", "/utf-8", "/bigobj", "/DGGML_BACKEND_SHARED", "/DGGML_SHARED",
               "/DGGML_USE_CPU", "/DGGML_USE_CUDA", "/DLLAMA_SHARED", "/DLLAMA_SUBPROCESS",
               "/D_CRT_SECURE_NO_WARNINGS", *["/I" + str(p) for p in includes], str(copied),
               "/Fo" + str(dest / "hy-batch-core.obj"), "/Fe" + str(dest / "hy-batch-core.exe"),
               "/link", "/MACHINE:X64", "/INCREMENTAL:NO", *map(str, libs),
               "kernel32.lib", "user32.lib", "advapi32.lib", "shell32.lib", "ole32.lib"]
    subprocess.run(command, cwd=dest, check=True)
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    (dest / "build.json").write_text(json.dumps({
        "original_source_sha256": digest(original), "patch_sha256": digest(patch),
        "instrumented_source_sha256": digest(copied),
        "executable_sha256": digest(dest / "hy-batch-core.exe"), "command": command,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
