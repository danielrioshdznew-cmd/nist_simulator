"""NIST simulator lab: serves the three.js page (static/) and relays POST /api to worker.py, which runs the competition's own
released v0.2.0 simulator (cloned by setup into ./apex-competition-nist-assembly) with a frozen policy.

    python server.py [--port 8765] [--host 127.0.0.1] [--policy policy/c053] [--sandbox /path/to/netns-run]
then open http://127.0.0.1:8765"""
import argparse
import json
import platform
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from another machine")
    ap.add_argument("--referee", default=str(HERE / "apex-competition-nist-assembly"), help="repo root that contains referee/nist_sim")
    ap.add_argument("--policy", default=str(HERE / "policy/c053"))
    ap.add_argument("--bank", default=str(HERE / "data/failure_bank_c053.json.gz"))
    ap.add_argument("--sandbox", default="", help="optional wrapper to run the simulator in (e.g. a netns-run script)")
    a = ap.parse_args()
    a.referee, a.bank = str(Path(a.referee).resolve()), str(Path(a.bank).resolve())
    if not (Path(a.referee) / "referee/nist_sim/env.py").exists():
        sys.exit(f"simulator not found in {a.referee}: run setup first (see README)")
    cmd = [sys.executable, "-I", "-B", str(HERE / "worker.py"), str(HERE), a.referee, str(Path(a.policy).resolve()), a.bank]
    if a.sandbox:
        cmd = [a.sandbox] + cmd
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(HERE), text=True, bufsize=1)
    first = proc.stdout.readline()
    if '"ready"' not in first:
        sys.exit(f"worker failed to start: {first!r}")
    lock = threading.Lock()

    def call(req: bytes) -> bytes:
        with lock:
            proc.stdin.write(req.decode().replace("\n", " ") + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        if not line:
            return json.dumps({"error": "worker exited; restart server.py"}).encode()
        return line.encode()

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(HERE / "static"), **kw)

        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_POST(self):
            if self.path != "/api":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            out = call(body)
            try:
                if json.loads(body).get("cmd") == "export":   # also keep a copy in ./exports
                    d = json.loads(out)
                    ex = HERE / "exports"
                    ex.mkdir(exist_ok=True)
                    f = ex / f"simlab_{d['seed']}_{d['episode']}_step{d['step']}_{time.strftime('%H%M%S')}.json"
                    f.write_text(json.dumps(d, indent=1))
                    d["saved_to"] = str(f)
                    out = json.dumps(d).encode()
            except Exception:  # noqa: BLE001
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    print(f"NIST simulator lab ({platform.system()}, policy {Path(a.policy).name}) - open http://127.0.0.1:{a.port}", flush=True)
    if a.host == "0.0.0.0":
        try:
            s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s_.connect(("10.255.255.255", 1))          # no packet is sent; this only picks the outgoing interface
            print(f"   or from another machine: http://{s_.getsockname()[0]}:{a.port}", flush=True)
        except OSError:
            pass
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
