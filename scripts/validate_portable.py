"""Extract and smoke-test the finished Windows ZIP in a relocated Unicode path.

This runs real GPU inference. It never builds, packages, downloads, overwrites an
extraction, or recursively removes anything. Logs and the report are retained.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LABEL = "portable-smoke"
PORT = 18081
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_parts(name: str) -> tuple[str, ...]:
    """Reject traversal, Windows aliases/ADS, and ambiguous ZIP names."""
    require(isinstance(name, str) and name and "\\" not in name, "Invalid ZIP/manifest path")
    parts = tuple(name.split("/"))
    require(not PurePosixPath(name).is_absolute(), "Absolute ZIP/manifest path")
    for part in parts:
        require(part not in ("", ".", "..") and not part.endswith((" ", ".")), "Ambiguous ZIP/manifest path")
        require(not any(ord(c) < 32 or c in '<>:"|?*' for c in part), "Invalid Windows path component")
        require(part.split(".")[0].upper() not in RESERVED, "Reserved Windows filename")
    return parts


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_and_verify(archive: Path, destination: Path) -> dict:
    require(not os.path.lexists(destination), "Extraction destination already exists; choose a new --destination")
    with zipfile.ZipFile(archive) as source:
        members, seen, files, roots = [], set(), set(), set()
        total = 0
        for item in source.infolist():
            parts = safe_parts(item.filename[:-1] if item.is_dir() else item.filename)
            key = "/".join(parts).casefold()
            require(key not in seen, "Duplicate or case-colliding ZIP member")
            seen.add(key)
            roots.add(parts[0])
            mode = stat.S_IFMT(item.external_attr >> 16)
            require(mode in (0, stat.S_IFREG, stat.S_IFDIR), "ZIP contains a link or special file")
            require(not item.external_attr & 0x400, "ZIP contains a Windows reparse point")
            require(not item.flag_bits & 1, "Encrypted ZIP entries are unsupported")
            require(item.is_dir() or len(parts) > 1, "ZIP payload must have one top-level directory")
            if not item.is_dir():
                files.add(key)
                total += item.file_size
            members.append((item, parts))
        require(len(roots) == 1 and files, "ZIP must contain one nonempty package directory")
        require(total <= 20 * 1024**3, "Unexpected ZIP expansion over 20 GiB")
        for _, parts in members:
            require(all("/".join(parts[:i]).casefold() not in files for i in range(1, len(parts))), "ZIP file/directory collision")
        archive_root = next(iter(roots))
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(destination.parent).free > total + 128 * 1024**2, "Insufficient free space for extraction")
        destination.mkdir()  # Never reuse an existing directory, even on failure.
        for item, parts in members:
            target = destination.joinpath(*parts[1:])
            require(target.resolve().is_relative_to(destination.resolve()), "ZIP member escapes destination")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as input_file, target.open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file, 4 * 1024 * 1024)
    manifest = load_json(destination / "MANIFEST.sha256.json")
    require(manifest.get("format") == 1 and manifest.get("package") == archive_root, "Unsupported/mismatched package manifest")
    records = {}
    for record in manifest["files"]:
        relative = "/".join(safe_parts(record["path"]))
        require(relative.casefold() not in records, "Duplicate manifest entry")
        require(type(record["size"]) is int and record["size"] >= 0, "Invalid manifest size")
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "Invalid manifest SHA256")
        records[relative.casefold()] = {**record, "path": relative}
    actual = {p.relative_to(destination).as_posix().casefold() for p in destination.rglob("*") if p.is_file()}
    require(actual == set(records) | {"manifest.sha256.json", "sha256sums.txt"}, "Manifest does not exactly cover ZIP payload")
    sums = {}
    for line in (destination / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        checksum, separator, relative = line.partition("  ")
        require(separator and re.fullmatch(r"[0-9a-f]{64}", checksum), "Invalid SHA256SUMS line")
        key = "/".join(safe_parts(relative)).casefold()
        require(key not in sums, "Duplicate SHA256SUMS entry")
        sums[key] = checksum
    require(set(sums) == set(records) | {"manifest.sha256.json"}, "SHA256SUMS coverage mismatch")
    require(digest(destination / "MANIFEST.sha256.json") == sums["manifest.sha256.json"], "Manifest hash mismatch")
    for key, record in records.items():
        path = destination / record["path"]
        require(path.stat().st_size == record["size"], f"Payload size mismatch: {record['path']}")
        require(digest(path) == record["sha256"] == sums[key], f"Payload hash mismatch: {record['path']}")
    return {"verified": True, "payload_files": len(records), "payload_bytes": total,
            "zip_sha256": digest(archive), "manifest_sha256": sums["manifest.sha256.json"]}


def isolated_environment(logs: Path) -> dict[str, str]:
    environment = dict(os.environ)
    # Erase inherited build settings and Python/site customizations. The two bad
    # Python variables below deliberately remain for testing run-python.cmd -E.
    for name in list(environment):
        if name.upper().startswith(("PYTHON", "CUDA", "GGML", "LLAMA", "HF_", "VIRTUAL_ENV", "CONDA", "HY_")):
            environment.pop(name)
        elif name.upper().endswith("_PROXY"):
            environment.pop(name)
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    environment.update(PATH=os.pathsep.join(map(str, (system, system / "WindowsPowerShell/v1.0", system / "Wbem"))),
                       PYTHONHOME=str(logs / "deliberately-missing-python-home"),
                       PYTHONPATH=str(logs / "deliberately-missing-python-path"),
                       NO_PROXY="127.0.0.1,localhost")
    return environment


def cmd_line(arguments: list[str]) -> str:
    # All arguments are controlled paths/options. cmd expands %, ! and shell
    # metacharacters even in surprising quoting contexts, so refuse them here.
    require(all(not any(c in str(arg) for c in '\r\n"%!?&|<>^') for arg in arguments), "Unsafe character in command argument")
    return '"' + " ".join('"' + str(arg) + '"' for arg in arguments) + '"'


class WindowsProcess:
    """Keep the queried handle open, so cleanup cannot target a recycled PID."""
    def __init__(self, pid: int):
        self.pid = pid
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.handle = self.kernel.OpenProcess(0x100000 | 0x400 | 0x10 | 0x1, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            buffer, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            if not self.kernel.QueryFullProcessImageNameW(self.handle, 0, buffer, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.executable = Path(buffer.value).resolve()
            times = [wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(value) for value in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.created_unix = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 10_000_000 - 11644473600
        except BaseException:
            self.close()
            raise

    def alive(self):
        return self.kernel.WaitForSingleObject(self.handle, 0) == 0x102

    def modules(self) -> list[Path]:
        api = ctypes.WinDLL("psapi", use_last_error=True)
        api.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        api.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        handles, needed = (wintypes.HMODULE * 4096)(), wintypes.DWORD()
        if not api.EnumProcessModulesEx(self.handle, handles, ctypes.sizeof(handles), ctypes.byref(needed), 3):
            raise ctypes.WinError(ctypes.get_last_error())
        require(needed.value <= ctypes.sizeof(handles), "Too many loaded modules to audit")
        result = []
        for handle in handles[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            buffer = ctypes.create_unicode_buffer(32768)
            if not api.GetModuleFileNameExW(self.handle, handle, buffer, len(buffer)):
                raise ctypes.WinError(ctypes.get_last_error())
            result.append(Path(buffer.value).resolve())
        return result

    def terminate_owned(self):
        if self.alive() and not self.kernel.TerminateProcess(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        require(self.kernel.WaitForSingleObject(self.handle, 15000) == 0, "Owned server did not terminate")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_cmd(name: str, arguments: list[str], *, package: Path, logs: Path, environment: dict, records: list, timeout=300):
    stdout, stderr = logs / f"{name}.stdout.log", logs / f"{name}.stderr.log"
    system = Path(next((value for key, value in environment.items() if key.upper() == 'SYSTEMROOT'), r'C:\Windows')) / 'System32'
    # Pass a raw Windows command line: list2cmdline's backslash quote escaping
    # targets the C runtime and is inappropriate for cmd.exe /s /c.
    command = f'"{system / "cmd.exe"}" /d /v:off /s /c {cmd_line(arguments)}'
    started = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        with subprocess.Popen(command, cwd=package, env=environment, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                              creationflags=subprocess.CREATE_NO_WINDOW) as process:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only this still-owned launcher and its descendants; never
                # taskkill by image name or stop another label/port.
                subprocess.run([str(system / "taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
                               stdout=err, stderr=err, env=environment, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                process.wait(timeout=30)
                raise
    records.append({"name": name, "returncode": returncode, "seconds": time.monotonic() - started,
                    "stdout": stdout.name, "stderr": stderr.name})
    require(returncode == 0, f"{name} failed ({returncode}); see {stderr.name}")
    return stdout


PYTHON_PROBE = '''import sys, json, pathlib, ssl, ctypes, sqlite3, bz2, lzma, urllib.request, concurrent.futures
package = pathlib.Path(sys.argv[1]).resolve()
runtime = package / 'runtime/python'
assert pathlib.Path(sys.executable).resolve() == runtime / 'python.exe'
assert pathlib.Path(sys.prefix).resolve() == runtime == pathlib.Path(sys.base_prefix).resolve()
assert sys.flags.ignore_environment and sys.flags.no_user_site and sys.flags.utf8_mode
# The test script itself lives in the external log directory; every remaining
# search path and every imported library must come from the relocated runtime.
assert pathlib.Path(sys.path[0]).resolve() == pathlib.Path(__file__).resolve().parent
assert all(pathlib.Path(p).resolve().is_relative_to(runtime) for p in sys.path[1:])
modules = {name: pathlib.Path(m.__file__).resolve() for name, m in sys.modules.copy().items()
           if getattr(m, '__file__', None) and name != '__main__'}
assert all(path.is_relative_to(runtime) for path in modules.values())
print(json.dumps({'version': sys.version.split()[0], 'ignore_environment': bool(sys.flags.ignore_environment),
 'no_user_site': bool(sys.flags.no_user_site), 'utf8_mode': bool(sys.flags.utf8_mode),
 'executable': pathlib.Path(sys.executable).resolve().relative_to(package).as_posix(),
 'prefix': pathlib.Path(sys.prefix).resolve().relative_to(package).as_posix(),
 'stdlib_modules_checked': len(modules),
 'search_path': [pathlib.Path(p).resolve().relative_to(package).as_posix() for p in sys.path[1:]]}))
'''


def check_config(config: dict, package: Path, mode: str, parallel: int | None = None):
    require(config.get("ok") is True, "GPU configuration failed")
    require(Path(config["root"]).resolve() == package, "Config root is not the relocated package")
    require(Path(config["binary_dir"]).resolve() == package / "bin", "Config selected an external binary directory")
    executable = package / "bin" / ("hy-batch.exe" if mode == "batch" else "llama-server.exe")
    require(Path(config["executable"]).resolve() == executable, "Config selected an external executable")
    require(Path(config["model"]).resolve().is_relative_to(package / "models"), "Config selected an external model")
    require(config["mode"] == mode, "Configuration mode mismatch")
    if parallel is not None:
        require(config["parallel"] == parallel, "Configured parallelism differs from request")
    if "command" in config:
        require(Path(config["command"][0]).resolve() == executable, "Server command selected an external executable")


def check_translations(path: Path, count: int):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require(len(records) == count, "Translation count mismatch")
    require(all(r.get("ok") and str(r.get("translation", "")).strip() and not r.get("truncated") and not r.get("error") for r in records),
            "Translation failed, was empty, or was truncated")
    return {"requests": count, "nonempty": True, "errors": 0, "truncated": 0}


def validate(args) -> int:
    archive, package, report_file = args.archive.resolve(), args.destination.resolve(), args.report.resolve()
    logs = report_file.with_suffix("").with_name(report_file.stem + "-logs")
    require(archive.is_file(), "Finished package ZIP is missing")
    require(not os.path.lexists(package), "Extraction destination already exists")
    require(not report_file.exists() and not os.path.lexists(logs), "Report or log directory already exists")
    require(not logs.is_relative_to(package) and not report_file.is_relative_to(package), "Validation logs/report must be outside the extracted package")
    logs.mkdir(parents=True)
    report = {"schema_version": 1, "ok": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "archive": str(archive), "relocated_root": str(package), "logs": str(logs), "commands": [],
              "notes": ["Smoke test on this machine, not certification of every compiled GPU architecture.",
                        "Example input has three requests; --parallel 4 is capped by the batch launcher to that count."]}
    environment = isolated_environment(logs)
    report["environment"] = {"PATH": environment["PATH"], "bad_python_environment_injected": True}
    owned = None
    server_started_at = None
    server_attempted = False
    pid_path = package / "results" / f"{LABEL}.pid"

    def run(name, arguments, timeout=300):
        return run_cmd(name, list(map(str, arguments)), package=package, logs=logs, environment=environment,
                       records=report["commands"], timeout=timeout)

    def own_server():
        if not pid_path.is_file():
            return None
        pid = int(pid_path.read_text(encoding="ascii").strip())
        process = WindowsProcess(pid)
        try:
            require(process.executable == package / "bin/llama-server.exe", "PID file does not identify the relocated server")
            require(server_started_at is not None and process.created_unix >= server_started_at - 1, "PID belongs to an older server")
            return process
        except BaseException:
            process.close()
            raise

    try:
        report["manifest"] = extract_and_verify(archive, package)
        probe = logs / "python-relocation-probe.py"
        probe.write_text(PYTHON_PROBE, encoding="utf-8")
        report["python"] = load_json(run("python-relocation", [package / "scripts/run-python.cmd", "-B", probe, package], 30))
        model_manifest = load_json(package / 'models/manifest.json')
        report['model_setup'] = []
        for profile, item in model_manifest['files'].items():
            target = package/'models'/item['filename']
            if not target.is_file():
                if args.model_directory:
                    source_model = args.model_directory.resolve()/item['filename']
                    require(source_model.stat().st_size == item['size_bytes'] and digest(source_model) == item['sha256'], 'Local model checksum mismatch')
                    shutil.copyfile(source_model, target)
                    method = 'verified local model copy (offline validation)'
                else:
                    run(f'setup-{profile}', [package/'setup-model.cmd', '--profile', profile], max(args.timeout, 1800))
                    method = 'Hugging Face download through setup-model.cmd'
                require(target.stat().st_size == item['size_bytes'] and digest(target) == item['sha256'], 'Installed model checksum mismatch')
                report['model_setup'].append({'profile': profile, 'method': method, 'sha256': item['sha256']})
        info = load_json(run("gpu-info", [package / "gpu-info.cmd", "--json", "--parallel", "4"], 30))
        check_config(info, package, "batch", 4)
        report["gpu_info"] = info
        source = package / "examples/input.jsonl"
        count = sum(bool(line.strip()) for line in source.read_text(encoding="utf-8-sig").splitlines())
        require(count > 0, "Example input is empty")
        report["batch"] = {}
        for profile in ("auto", "official"):
            output = logs / f"batch-{profile}.jsonl"
            run(f"batch-{profile}", [package / "translate-batch.cmd", source, output, "--profile", profile,
                                      "--parallel", "4", "--max-tokens", "128"], args.timeout)
            config = load_json(output.with_suffix(".config.json"))
            check_config(config, package, "batch", min(4, count))
            if profile == "official":
                require(config["profile"] == "official" and Path(config["model"]).name == "Hy-MT2-1.8B-Q4_K_M-fused.gguf", "Official profile did not select Q4_K_M")
            summary = load_json(output.with_suffix(".summary.json"))
            require(summary["requests"] == count and summary["empty_outputs"] == summary["truncated"] == 0, "Native batch summary failed")
            report["batch"][profile] = {**check_translations(output, count), "requested_parallel": 4,
                                          "actual_parallel": config["actual_parallel"], "config": config, "summary": summary}
        with socket.socket() as probe_socket:
            probe_socket.bind(("127.0.0.1", PORT))
        require(not pid_path.exists(), "Validation server label already exists")
        server_started_at = time.time()
        server_attempted = True
        run("server-start", [package / "start-server.cmd", "-Label", LABEL, "-Port", str(PORT), "-Parallel", "4"], 180)
        owned = own_server()
        require(owned is not None and owned.alive(), "Started server is not running")
        config = load_json(package / "results" / f"{LABEL}.config.json")
        check_config(config, package, "server", 4)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{PORT}/health", timeout=10) as response:
            health = json.load(response)
        require(health.get("status") == "ok", "Relocated API health failed")
        result = logs / "api-translation.json"
        run("api-translation", [package / "scripts/run-python.cmd", package / "scripts/translate.py",
                                "The next train arrives in ten minutes.", "--target", "Chinese", "--url", f"http://127.0.0.1:{PORT}",
                                "--max-tokens", "128", "--json", "--output", result], args.timeout)
        modules = owned.modules()
        forbidden = [ROOT / "runtime", ROOT / ".venv", ROOT / "build", ROOT / "bin"]
        require(not any(path.is_relative_to(root) for path in modules for root in forbidden), "Server loaded an original workspace runtime/binary")
        package_modules = [path.relative_to(package).as_posix() for path in modules if path.is_relative_to(package)]
        required_modules = {"ggml-cuda.dll", "llama.dll", "llama-server-impl.dll", "cublas64_13.dll", "cublaslt64_13.dll"}
        require(required_modules <= {Path(path).name.lower() for path in package_modules}, "GPU/server libraries did not load from the relocated package")
        report["api"] = {**check_translations(result, 1), "config": config, "health": health, "pid": owned.pid,
                         "running_executable": owned.executable.relative_to(package).as_posix(), "package_modules": package_modules,
                         "original_runtime_modules": [], "loaded_modules_checked": len(modules)}
        report["ok"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if server_attempted:
            try:
                if owned is None:
                    try:
                        owned = own_server()
                    except OSError as error:
                        if getattr(error, "winerror", None) != 87:  # Already exited PID.
                            raise
                if owned is not None and owned.alive():
                    require(int(pid_path.read_text(encoding="ascii").strip()) == owned.pid, "Managed PID changed; do not stop another service")
                    try:
                        run("server-stop", [package / "stop-server.cmd", "-Label", LABEL], 30)
                        require(owned.kernel.WaitForSingleObject(owned.handle, 15000) == 0, "Stop launcher left owned server running")
                        report["cleanup"] = {"owned_pid": owned.pid, "stopped": True, "method": "stop-server.cmd"}
                    except BaseException:
                        # A failed stop-wrapper must not leak this invocation's
                        # GPU process. This retained handle identifies only ours.
                        owned.terminate_owned()
                        report["cleanup"] = {"owned_pid": owned.pid, "stopped": True, "method": "owned-handle fallback"}
                        raise
                else:
                    report["cleanup"] = {"stopped": True, "method": "server absent or already exited"}
            except BaseException as error:
                report["ok"] = False
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
            finally:
                if owned is not None:
                    owned.close()
            for suffix in ("config.json", "stdout.log", "stderr.log"):
                source = package / "results" / f"{LABEL}.{suffix}"
                if source.is_file():
                    try:
                        with source.open("rb") as input_file, (logs / source.name).open("xb") as output_file:
                            shutil.copyfileobj(input_file, output_file)
                    except OSError as error:
                        report.setdefault("log_copy_errors", []).append(str(error))
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with report_file.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps({"ok": report["ok"], "report": str(report_file), "error": report.get("error"),
                      "cleanup_error": report.get("cleanup_error")}, ensure_ascii=False))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/HyMT-Windows-NVIDIA-runtime.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / ".local/便携 包迁移验证/HyMT-Windows-NVIDIA-runtime")
    parser.add_argument("--report", type=Path, default=ROOT / ".local/portable-validation.json")
    parser.add_argument('--model-directory', type=Path, help='Offline validation: use verified models from this directory instead of downloading')
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each batch/API translation")
    args = parser.parse_args()
    require(os.name == "nt", "Runtime relocation validation requires Windows")
    require(args.timeout > 0, "Timeout must be positive")
    return validate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Validation error: {error}", file=sys.stderr)
        raise SystemExit(1)
