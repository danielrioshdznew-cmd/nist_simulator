"""Rebuild a policy (ONNX) from controller source + RF_* switches.

    python tools/build_policy.py --out policy/my_fix --set RF_MYIDEA=1 [--set RF_OTHER=0.5] [--src policy/c053/src]

Put every new idea in policy/c053/src/refport_fx.py (or a copy given with --src) as a DEFAULT-OFF switch:
    MYIDEA = E("RF_MYIDEA", "0") == "1"   ...   if MYIDEA: <new graph nodes>
Rule: with the new switch off, the build must reproduce c053 exactly: `python tools/build_policy.py --out policy/check` must print
sha256 20283cda64dc9473... Writes <out>/policy.onnx, <out>/src/ (the source + ENV.txt used)."""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "policy/c053/src"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="RF_NAME=value, repeatable")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    env = dict(kv.split("=", 1) for kv in (src / "ENV.txt").read_text().split())
    for kv in a.set:
        k, v = kv.split("=", 1)
        env[k] = v
    (out / "src").mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.py"):
        shutil.copy2(f, out / "src" / f.name)
    (out / "src/ENV.txt").write_text(" ".join(f"{k}={v}" for k, v in env.items()) + "\n")
    r = subprocess.run([sys.executable, "-I", "-B", str(out / "src/export_rf.py"), str(out / "policy.onnx")],
                       env={**os.environ, **env, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-2000:])
    if r.returncode:
        sys.exit("build failed")
    sha = hashlib.sha256((out / "policy.onnx").read_bytes()).hexdigest()
    print(f"{out / 'policy.onnx'}  sha256 {sha}" + ("   (= c053, unchanged)" if sha.startswith("20283cda") else ""))


if __name__ == "__main__":
    main()
