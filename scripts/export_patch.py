"""Export the private llama.cpp changes without staging or changing either checkout.

The patch contains all tracked changes relative to HEAD, plus untracked sources
under the explicit private-source allowlist. Git binary patches preserve binary edits.
The separate baseline checkout is used only by `git apply --check`, never apply.

Usage: python scripts/export_patch.py
       python scripts/export_patch.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXTRA_DIRECTORIES = ("tools/hy-batch", "tools/hy-logits")
EXTRA_FILES = ("ggml/src/ggml-cuda/hymt-penalties.cu", "ggml/src/ggml-cuda/hymt-penalties.cuh",
               "tests/test-hymt-prefix.cpp")
GIT_EXECUTABLE: str | None = None


def find_git() -> str:
    candidate = GIT_EXECUTABLE or shutil.which("git")
    if candidate:
        return candidate
    bundled = ROOT / "runtime/git/cmd/git.exe"
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError("Git is not installed; add Git to PATH or pass --git")


def portable_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def git(repo: Path, *arguments: str, data: bytes | None = None, codes: tuple[int, ...] = (0,)) -> bytes:
    environment = os.environ.copy()
    # Even read-only Git commands must not refresh/write the user's index.
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    process = subprocess.run(
        [find_git(), "-C", str(repo), *arguments], input=data, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if process.returncode not in codes:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed ({process.returncode}): {message}")
    return process.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paths_from_z(data: bytes) -> list[str]:
    return [value.decode("utf-8") for value in data.split(b"\0") if value]


def checkout_state(repo: Path) -> dict:
    index_path = Path(git(repo, "rev-parse", "--git-path", "index").decode("utf-8").strip())
    if not index_path.is_absolute():
        index_path = repo / index_path
    return {
        "commit": git(repo, "rev-parse", "HEAD").decode("ascii").strip(),
        "status": git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        "index_sha256": sha256(index_path.read_bytes()) if index_path.exists() else None,
    }


def source_record(repo: Path, name: str, tracked: bool) -> dict:
    # Paths come from Git, but still reject anything outside the intended tree.
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid repository-relative path: {name}")
    path = repo / relative
    if path.is_symlink():
        raw = os.fsencode(os.readlink(path))
        kind = "symlink"
    elif path.is_file():
        raw = path.read_bytes()
        kind = "file"
    elif not path.exists():
        return {"path": name, "tracked": tracked, "kind": "deleted", "bytes": 0, "sha256": None}
    else:
        raise ValueError(f"cannot export this file type: {name}")
    record = {"path": name, "tracked": tracked, "kind": kind, "bytes": len(raw), "sha256": sha256(raw)}
    if kind == "file" and b"\0" not in raw:
        record["sha256_lf"] = sha256(raw.replace(b"\r\n", b"\n"))
    return record


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export(args: argparse.Namespace) -> dict:
    repo, baseline = resolve(args.repo), resolve(args.check_against)
    repo = Path(git(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    baseline = Path(git(baseline, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if repo == baseline or os.path.samefile(repo, baseline):
        raise ValueError("apply --check requires a separate baseline checkout, never the modified source tree")
    source_before, baseline_before = checkout_state(repo), checkout_state(baseline)
    if source_before["commit"] != baseline_before["commit"]:
        raise ValueError("baseline HEAD must match source HEAD")
    if baseline_before["status"]:
        raise ValueError("baseline checkout must be pristine, including untracked files")

    tracked = paths_from_z(git(repo, "diff", "--name-only", "--no-renames", "-z", "HEAD", "--"))
    added = paths_from_z(git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", *EXTRA_DIRECTORIES, *EXTRA_FILES))
    records = [source_record(repo, name, True) for name in tracked]
    records += [source_record(repo, name, False) for name in added]
    records.sort(key=lambda item: item["path"])
    diff_options = ("--no-ext-diff", "--no-textconv", "--binary", "--full-index", "--src-prefix=a/", "--dst-prefix=b/")
    patch = git(repo, "diff", *diff_options, "--no-renames", "HEAD", "--")
    for name in added:
        # Git recognizes NUL as /dev/null on Windows and produces normal a/b
        # headers directly. No index entries or temporary Git objects are made.
        patch += git(repo, "diff", "--no-index", *diff_options, "--",
                     "NUL" if os.name == "nt" else "/dev/null", name, codes=(0, 1))
    if not patch or not records:
        raise ValueError("no source changes to export")
    check_command = ["apply", "--check", "--whitespace=nowarn", "-"]
    git(baseline, *check_command, data=patch)

    # Refuse to certify a snapshot if another agent edited it during the export.
    if checkout_state(repo) != source_before:
        raise RuntimeError("source Git state changed during export; retry after edits finish")
    if [source_record(repo, item["path"], item["tracked"]) for item in records] != records:
        raise RuntimeError("source file contents changed during export; retry after edits finish")
    if checkout_state(baseline) != baseline_before:
        raise RuntimeError("baseline state changed during read-only validation")

    output = resolve(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    # Output artifacts must not overwrite source or baseline checkout files.
    for directory in (repo, baseline):
        if output.is_relative_to(directory) or manifest_path.is_relative_to(directory):
            raise ValueError("patch and manifest must be saved outside both source checkouts")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": portable_path(repo), "base_commit": source_before["commit"],
        "upstream_repository": "https://github.com/ggml-org/llama.cpp.git",
        "patch": portable_path(output), "patch_bytes": len(patch), "patch_sha256": sha256(patch),
        "binary_patch": True, "changed_files": records,
        "tracked_files": len(tracked), "added_untracked_files": len(added),
        "extra_source_directories": list(EXTRA_DIRECTORIES),
        "extra_source_files": list(EXTRA_FILES),
        "source_index_sha256": source_before["index_sha256"],
        "git_state_unchanged": True,
        "apply_check": {"passed": True, "checkout": portable_path(baseline),
                        "base_commit": baseline_before["commit"], "command": ["git", *check_command],
                        "patch_sent_via_stdin": True, "applied": False},
        "notes": [
            "Tracked changes are exported relative to HEAD, including staged and unstaged edits; nothing is staged or committed.",
            "Untracked files are restricted to the explicit private-source directories and file allowlist.",
            "Per-file SHA256 values describe raw working-tree bytes; Git text conversion can normalize line endings in the patch.",
            "sha256_lf records text with CRLF normalized to LF for clean cross-machine source preparation.",
            "Validation uses git apply --check on the distinct pristine baseline; the patch is never applied to either checkout.",
        ],
    }
    if not args.dry_run:
        atomic_write(output, patch)
        atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    global GIT_EXECUTABLE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="src/llama.cpp")
    parser.add_argument("--check-against", default="src/llama-baseline")
    parser.add_argument("--output", default="patches/hy-mt2.patch")
    parser.add_argument("--git", help="Git executable; default: PATH, then optional local runtime/git")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory; do not write artifacts")
    args = parser.parse_args()
    GIT_EXECUTABLE = args.git
    try:
        manifest = export(args)
        print(json.dumps({key: value for key, value in manifest.items() if key != "changed_files"},
                         ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
