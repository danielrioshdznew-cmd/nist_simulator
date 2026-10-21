"""Build ONE self-contained HTML page of whole-episode replays (open it by double-clicking: no Python, no server, offline).

    python tools/build_video_page.py --policy c054_userfixes --cases 30 197 40 35 262 180 272 394 --out videos/c054_fixes.html

Needs the lab running (server.py) to simulate the episodes; the page itself embeds three.js and every frame (every step,
16-bit positions / rotations), so it plays anywhere. It shows the episode only: no result text, nothing on the last frame."""
import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEP = 1               # keep every simulation step (25 per second of simulated time): smooth slow motion
POS_Q = 0.00005        # 0.05 mm per unit (int16: +-1.6 m)


def api(url, cmd, **a):
    req = urllib.request.Request(url + "/api", data=json.dumps({"cmd": cmd, **a}).encode())
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def mat2quat(m):
    """Row-major 3x3 rotation matrices (n, 9) -> quaternions (n, 4) as x, y, z, w."""
    m = m.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i, r in enumerate(m):
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def pack(arr12):
    """(n, 12) float32 [pos 3, xmat 9] -> int16 bytes of [pos 3, quat 4] per row."""
    pos = np.round(arr12[:, :3] / POS_Q)
    quat = np.round(mat2quat(arr12[:, 3:]) * 32767)
    return np.concatenate([pos, quat], axis=1).clip(-32767, 32767).astype("<i2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="c054_userfixes")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="c054: your fixes, whole episodes")
    ap.add_argument("--api", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    notes = json.loads((ROOT / "exports" / "notes.json").read_text()) if (ROOT / "exports" / "notes.json").exists() else {}
    clips = api(a.api, "clips", ids=a.cases, before=75, after=75, full=True, policy=a.policy)["clips"]
    data = []
    for cl in clips:
        c = cl["case"]
        nd = len(cl["dyn"])
        fr = np.frombuffer(base64.b64decode(cl["frames"]), dtype=np.float32).reshape(-1, nd, 12)
        keep = list(range(0, len(fr), STEP))
        if keep[-1] != len(fr) - 1:
            keep.append(len(fr) - 1)                     # always end on the episode's real last frame
        st = np.frombuffer(base64.b64decode(cl["static"]), dtype=np.float32).reshape(-1, 12)
        geoms = [{k: g[k] for k in ("type", "size", "rgba", "body", "name")} for g in cl["scene"]["geoms"]]
        data.append({
            "id": c["id"], "board": f"{c['seed']}/{c['episode']}", "peg": c["peg"], "shape": c["shape"], "dia": c["dia_mm"],
            "start": c["start"], "c053": cl["label"], "c053_step": c["fail_step"],
            "note": (notes.get(str(c["id"])) or {}).get("text", ""), "board_xy": cl["scene"].get("board_xy"),
            "geoms": geoms, "dyn": cl["dyn"], "static": base64.b64encode(pack(st)).decode(),
            "steps": [cl["steps"][i] for i in keep], "frames": base64.b64encode(pack(fr[keep].reshape(-1, 12))).decode(),
            "peg_pos": [cl["pos"][i] for i in keep]})
        print(f"#{c['id']}: {len(cl['steps'])} steps -> {len(keep)} frames, {nd} moving geoms", flush=True)
    vend = ROOT / "static" / "vendor"
    three = base64.b64encode((vend / "three.module.js").read_bytes()).decode()
    orbit = base64.b64encode((vend / "OrbitControls.js").read_bytes()).decode()
    html = (ROOT / "tools" / "video_page_template.html").read_text()
    html = (html.replace("__TITLE__", a.title).replace("__POLICY__", a.policy)
            .replace("__THREE_B64__", three).replace("__ORBIT_B64__", orbit)
            .replace("__DATA__", json.dumps({"policy": a.policy, "pos_q": POS_Q, "step": STEP, "cases": data}, separators=(",", ":"))))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, {len(data)} episodes")


if __name__ == "__main__":
    sys.exit(main())
