"""Measure Windows WDDM memory peaks for graph coverage at 32 concurrency.

Uses the saved plan from a benchmark_graph_coverage matrix, keeping KV bytes
fixed. Results contain local process IDs and paths; publish a compact summary.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import hashlib
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from vllm_runtime import environment, server_command, stop_process_tree

class Value(C.Structure):
    _fields_ = [('status', W.DWORD), ('value', C.c_longlong)]

class Item(C.Structure):
    _fields_ = [('name', W.LPWSTR), ('value', Value)]

class Counters:
    def __init__(self):
        self.dll = C.WinDLL('pdh')
        self.query = W.HANDLE()
        self.dll.PdhOpenQueryW(None, 0, C.byref(self.query))
        self.handles = {}
        for label in ('Dedicated Usage', 'Shared Usage'):
            handle = W.HANDLE()
            path = '\\GPU Process Memory(*)\\' + label
            rc = self.dll.PdhAddEnglishCounterW(self.query, C.c_wchar_p(path), 0, C.byref(handle))
            if rc: raise RuntimeError(('add counter', rc))
            self.handles[label] = handle
    def sample(self):
        self.dll.PdhCollectQueryData(self.query)
        result = {}
        for label, handle in self.handles.items():
            size, count = W.DWORD(), W.DWORD()
            self.dll.PdhGetFormattedCounterArrayW(handle, 0x400, C.byref(size), C.byref(count), None)
            if not size.value: continue
            buf = C.create_string_buffer(size.value)
            rc = self.dll.PdhGetFormattedCounterArrayW(handle, 0x400, C.byref(size), C.byref(count), buf)
            if rc: raise RuntimeError(('read counter', rc))
            rows = C.cast(buf, C.POINTER(Item))
            values = {}
            for i in range(count.value):
                item = rows[i]
                if item.value.status not in (0, 1): continue
                pid = int(item.name.split('_')[1])
                values[pid] = values.get(pid, 0) + item.value.value
            result[label] = values
        return result

def main():
    counter = Counters()
    if '--probe' in sys.argv:
        print(json.dumps(counter.sample()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    plan = json.loads(args.matrix.read_text(encoding='utf-8'))['plan']
    if plan['parallel'] != 32:
        raise ValueError('Expected a 32-concurrency plan')
    env = environment(ROOT, gpu=plan['gpu']['uuid'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with socket.socket() as probe: probe.bind(('127.0.0.1', 18086))
    report = {'method': 'WDDM GPU Process Memory counters summed for server process tree; 100ms target polling interval; observed maxima, not guaranteed absolute peaks', 'concurrency': 32, 'kv_cache_mib': plan['budget']['kv_total_mib'], 'runs': []}
    dll = ROOT / 'runtime/vllm/Lib/site-packages/torch/lib/torch_cuda.dll'
    with dll.open('rb') as stream:
        report['torch_cuda_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    for index, cap in enumerate((64, 256, 256, 64), 1):
        directory = output / f'run{index}-cap{cap}'
        directory.mkdir()
        command = server_command(plan, port=18086)
        pos = command.index('--compilation-config') + 1
        command[pos] = json.dumps({'max_cudagraph_capture_size': cap})
        samples, errors = [], []
        done = threading.Event()
        phase = ['startup']
        with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            parent = psutil.Process(process.pid)
            started = time.perf_counter()
            def monitor():
                try:
                    while not done.is_set():
                        ids = {process.pid} | {p.pid for p in parent.children(recursive=True)}
                        values = counter.sample()
                        samples.append({'s': time.perf_counter()-started, 'phase': phase[0], 'pids': sorted(ids), 'dedicated': sum(v for p,v in values.get('Dedicated Usage', {}).items() if p in ids), 'shared': sum(v for p,v in values.get('Shared Usage', {}).items() if p in ids)})
                        done.wait(.1)
                except Exception as error: errors.append(repr(error))
            thread = threading.Thread(target=monitor)
            thread.start()
            try:
                while True:
                    if process.poll() is not None: raise RuntimeError('server exited')
                    try:
                        with opener.open('http://127.0.0.1:18086/health', timeout=2) as response:
                            if response.status == 200: break
                    except OSError: pass
                    if time.perf_counter()-started > 600: raise TimeoutError('startup')
                    time.sleep(.5)
                phase[0] = 'ready'
                time.sleep(1)
                phase[0] = 'requests'
                bench = [sys.executable, '-X', 'utf8', str(ROOT/'scripts/benchmark_backend.py'), '--backend', 'vllm', '--url', 'http://127.0.0.1:18086', '--concurrency', '32', '--client', 'aiohttp', '--repeats', '1', '--output', str(directory/'bench')]
                with (directory/'bench.log').open('wb') as log:
                    subprocess.run(bench, stdout=log, stderr=log, cwd=ROOT, check=True, timeout=900)
                phase[0] = 'after_requests'
                time.sleep(1)
            finally:
                done.set()
                thread.join()
                stop_process_tree(process)
                (directory/'samples.json').write_text(json.dumps(samples))
            if errors: raise RuntimeError(errors)
            bench_report = json.loads((directory/'bench/report.json').read_text())
            record = {'capture_size': cap, 'sample_count': len(samples), 'max_interval_s': max(b['s']-a['s'] for a,b in zip(samples,samples[1:])), 'phases': {}}
            for name in ('startup', 'ready', 'requests', 'after_requests', 'all'):
                rows = [r for r in samples if name == 'all' or r['phase'] == name]
                record['phases'][name] = {field+'_peak_mib': max(r[field] for r in rows)/2**20 for field in ('dedicated', 'shared')}
            if not all(row['metrics']['valid'] for row in bench_report['runs']):
                raise RuntimeError('Invalid request pass')
            record['benchmark_report'] = bench_report
            report['runs'].append(record)
            (output/'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({k:v for k,v in record.items() if k != 'benchmark_report'}), flush=True)
            time.sleep(2)

if __name__ == '__main__': main()
