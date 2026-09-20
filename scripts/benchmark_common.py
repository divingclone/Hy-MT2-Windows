import json,subprocess,threading,time
from contextlib import contextmanager
from benchmark_inputs import ROOT, build_requests

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

@contextmanager
def telemetry(path):
    fields = ["clocks.sm", "clocks.mem", "power.draw", "utilization.gpu",
              "memory.used", "temperature.gpu", "pstate"]
    proc = subprocess.Popen(["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    ready = threading.Event()
    def reader():
        with path.open("w", encoding="utf-8") as stream:
            for line in proc.stdout:
                row = dict(zip(fields, [x.strip() for x in line.split(",")]))
                row["received_unix_s"] = time.time()
                stream.write(json.dumps(row) + "\n")
                ready.set()
    worker = threading.Thread(target=reader)
    worker.start()
    try:
        if not ready.wait(15):
            raise RuntimeError("GPU telemetry did not start")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        worker.join(timeout=10)
