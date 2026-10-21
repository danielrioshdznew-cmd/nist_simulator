"""Diagnose failure cases precisely: replay each case with the policy and, around the failure, record what actually happened.

    python tools/diagnose.py --cases 272 275 402 ... [--policy policy/c053] [--procs 8] [--out diag.json]
    python tools/diagnose.py --cat tip_pick --procs 300 --out diag_tip_pick.json

Per case it reports:
  - the FIRST gripper contact with the failing peg in the 120 steps before the failure: which body touched it (left pad,
    right pad, palm, fingers), the controller phase and target peg, the true hand-vs-peg offset (mm) across / along the jaws,
    the hand's yaw error against the peg (square pegs: modulo 90 deg), the jaw gap vs the peg's width;
  - whether the approach phase (P1) ended by arriving or by its 57-step time-out;
  - contacts between the failing peg and OTHER pegs (and which), and the carried peg if the hand was holding one;
  - the policy's estimate error of the failing peg's position at that moment;
  - the nearest other peg (distance, size, upright?) at the start of the descent."""
import argparse
import importlib
import json
import math
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cases_lib  # noqa: E402
import simlib  # noqa: E402


def _yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def diag_one(args):
    case, policy_dir = args
    simlib.setup()
    import mujoco
    src = Path(policy_dir) / "src"
    for kv in (src / "ENV.txt").read_text().split():
        k, v = kv.split("=", 1)
        os.environ[k] = v
    sys.path.insert(0, str(src))
    import fx
    R = importlib.import_module("refport_fx")
    act = simlib.policy(policy_dir)
    c, e, noise = simlib.board(case["seed"], case["episode"])
    m = e.model
    body = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ""  # noqa: E731
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""  # noqa: E731
    k = case["peg"]
    peg_body = e.peg_body[k]
    fail = case["fail_step"] if case["fail_step"] is not None else 10 ** 9
    lo = fail - 120
    p1_start, p1_len = None, {}
    hist, first_touch, peg_contacts = [], None, {}
    prev_ph = None
    while not e.done and e.steps <= min(fail + 2, c.max_steps):
        o = np.round(e.observe(noise), 6).astype(np.float32)
        a = act(o[None])[0]
        code = int(R._decode(fx.NP, np.array([[a[0]]], np.float32))[1][0, 0])
        ph, tgt = simlib.PH[code % 16], (code // 16) % 8
        if ph == "P1" and prev_ph != "P1":
            p1_start = e.steps
        if prev_ph == "P1" and ph != "P1" and p1_start is not None:
            p1_len[p1_start] = e.steps - p1_start
        prev_ph = ph
        if e.steps >= lo:
            d = e.data
            hand = d.qpos[0:4].copy()
            p, q = e.peg_pose(k)
            touch_geoms, other = [], set()
            for i in range(d.ncon):
                con = d.contact[i]
                b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
                if peg_body in (b1, b2):
                    ob = b2 if b1 == peg_body else b1
                    og = con.geom2 if b1 == peg_body else con.geom1
                    nm = body(ob)
                    if nm in ("hand", "fl", "fr"):
                        touch_geoms.append(gname(og))
                    elif nm.startswith("peg"):
                        other.add(nm)
            for nm in other:
                peg_contacts.setdefault(nm, e.steps)
            if touch_geoms and first_touch is None and tgt is not None:
                yaw_h = float(hand[3])
                dx, dy = p[0] - hand[0], p[1] - hand[1]
                across = -math.sin(yaw_h) * dx + math.cos(yaw_h) * dy          # along the jaws' closing direction (hand y)
                along = math.cos(yaw_h) * dx + math.sin(yaw_h) * dy
                sq = c.pegs[k].shape() == "square"
                yerr = _wrap(_yaw_of(q) - yaw_h, math.pi / 2 if sq else 2 * math.pi) if sq else 0.0
                f = o[21 + 17 * k: 38 + 17 * k]
                bx, by = c.board_xy
                est = np.array([(f[1] + f[13]) / 10 + bx, (f[2] + f[14]) / 10 + by])
                first_touch = {"step": int(e.steps), "phase": ph, "target": tgt, "own_target": tgt == k, "by": sorted(set(touch_geoms)),
                               "across_mm": round(across * 1000, 2), "along_mm": round(along * 1000, 2),
                               "hand_z_mm": round(float(hand[2]) * 1000, 1), "peg_top_mm": round((p[2] + 0.025) * 1000, 1),
                               "yaw_err_deg": round(math.degrees(yerr), 1), "gap_mm": round(float(d.qpos[4] + d.qpos[5]) * 1000, 2),
                               "peg_width_mm": round(c.pegs[k].diameter() * (math.sqrt(2) if sq else 1) * 1000, 1) if False else round(c.pegs[k].diameter() * 1000, 1),
                               "est_err_mm": round(float(np.linalg.norm(est - p[:2])) * 1000, 2),
                               "upright": round(abs(1 - 2 * (float(q[1]) ** 2 + float(q[2]) ** 2)), 3)}
        e.step(np.asarray(a, dtype=np.float64))
    # nearest other peg to the failing one at the window start
    return {"id": case["id"], "seed": case["seed"], "episode": case["episode"], "peg": k, "cat": case["cat"], "fail_step": case["fail_step"],
            "shape": case["shape"], "dia_mm": case["dia_mm"], "start": case["start"], "first_touch": first_touch,
            "p1_timeouts": [s for s, n in p1_len.items() if n >= 57 and s >= lo - 60],
            "peg_contacts": peg_contacts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="*", default=[])
    ap.add_argument("--cat")
    ap.add_argument("--policy", default=str(ROOT / "policy/c053"))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out")
    a = ap.parse_args()
    cases = cases_lib.build_cases(cases_lib.load_census(ROOT / "data/failure_bank_c053.json.gz"))
    sel = [cases[i] for i in a.cases] + ([c for c in cases if c["cat"] == a.cat] if a.cat else [])
    with get_context("spawn").Pool(a.procs) as pool:
        rows = pool.map(diag_one, [(c, a.policy) for c in sel], chunksize=1)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        t = r["first_touch"] or {}
        print(f"#{r['id']} {r['seed']}/{r['episode']} peg{r['peg']} {r['shape'][0]}{r['dia_mm']:g} fail@{r['fail_step']}: "
              + (f"first touch @{t['step']} {t['phase']}->peg{t['target']} by {t['by']} across {t['across_mm']} along {t['along_mm']} "
                 f"yawerr {t['yaw_err_deg']} gap {t['gap_mm']} est_err {t['est_err_mm']} hand_z {t['hand_z_mm']} top {t['peg_top_mm']}"
                 if t else "no gripper contact") + f" | P1 timeouts {r['p1_timeouts']} | peg contacts {r['peg_contacts']}")


if __name__ == "__main__":
    main()
