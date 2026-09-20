"""Fetch the pinned llama.cpp commit and apply the verified project patch.

Uses Python's standard library and Git. Existing destinations are never changed.
For an offline check, --repository may name a local Git checkout containing the
pinned commit; its working files and index are not used or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
COMMIT = "bdcbaaf6e7520b68c8c60ff724c67409970d70e1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(executable: str, directory: Path, *arguments: str) -> bytes:
    result = subprocess.run([executable, "-C", str(directory), *arguments], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed:\n" + result.stderr.decode("utf-8", "replace"))
    return result.stdout


def prepare(destination: Path, repository: str, executable: str) -> dict:
    if "://" not in repository and Path(repository).is_dir():
        repository = str(Path(repository).resolve())
    manifest = json.loads((ROOT / "patches/hy-mt2.manifest.json").read_text(encoding="utf-8"))
    patch = ROOT / "patches/hy-mt2.patch"
    patch_bytes = patch.read_bytes()
    if manifest["base_commit"] != COMMIT or sha256(patch_bytes) != manifest["patch_sha256"]:
        raise ValueError("Pinned commit or patch SHA256 does not match the manifest")
    records = {}
    for record in manifest["changed_files"]:
        path = PurePosixPath(record["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in record["path"] or ":" in record["path"]:
            raise ValueError("Invalid manifest path")
        if record["path"] in records:
            raise ValueError("Duplicate manifest path")
        records[record["path"]] = record
    if os.path.lexists(destination):
        raise FileExistsError("Destination exists; choose a new --destination. No files were changed.")
    destination.mkdir(parents=True)
    git(executable, destination, "init", "--template=")
    git(executable, destination, "config", "core.autocrlf", "false")
    git(executable, destination, "config", "core.eol", "lf")
    git(executable, destination, "remote", "add", "origin", UPSTREAM)
    git(executable, destination, "fetch", "--no-tags", "--depth=1", repository, COMMIT)
    git(executable, destination, "checkout", "--detach", COMMIT)
    actual_commit = git(executable, destination, "rev-parse", "HEAD").decode("ascii").strip()
    if actual_commit != COMMIT:
        raise ValueError("Fetched commit does not match the pinned revision")
    git(executable, destination, "apply", "--check", "--index", "--whitespace=nowarn", str(patch))
    git(executable, destination, "apply", "--index", "--whitespace=nowarn", str(patch))
    changed = {p.decode("utf-8") for p in git(executable, destination, "diff", "--name-only", "--no-renames", "-z", "HEAD").split(b"\0") if p}
    if changed != set(records):
        raise ValueError("Applied patch file list differs from the manifest")
    for name, record in records.items():
        path = destination / name
        if record["kind"] == "deleted":
            if path.exists():
                raise ValueError(f"Patch did not delete {name}")
        else:
            raw = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
            raw_matches = len(raw) == record["bytes"] and sha256(raw) == record["sha256"]
            lf_matches = record.get("sha256_lf") == sha256(raw)
            if not (raw_matches or lf_matches):
                raise ValueError(f"Prepared file differs from the manifest: {name}")
    # Index changes belong only to this newly created checkout. It provides an
    # exact tree ID without committing, modifying the source checkout, or hooks.
    result = {"base_commit": COMMIT, "patch_sha256": manifest["patch_sha256"],
              "changed_files_verified": len(records),
              "patched_tree": git(executable, destination, "write-tree").decode("ascii").strip()}
    (destination / ".git/hy-mt2-prepared.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "src/llama.cpp")
    parser.add_argument("--repository", default=UPSTREAM, help="Fetch source: official URL or existing local Git repository")
    parser.add_argument("--git", default=shutil.which("git"), help="Git executable; default: system PATH")
    args = parser.parse_args()
    if not args.git:
        parser.error("Git is required; install Git for Windows or pass --git")
    destination = args.destination.resolve()
    try:
        result = prepare(destination, args.repository, args.git)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Source preparation failed: {error}\nAny partial destination is retained for inspection.", file=sys.stderr)
        return 1
    print(json.dumps({"destination": str(destination), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
