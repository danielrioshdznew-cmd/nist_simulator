"""Reproduce one failure case headless and show what happened around the failure; optionally write a per-step CSV trace.

    python tools/reproduce.py --case 123 [--policy policy/c053] [--window 40] [--csv trace_123.csv]
    python tools/reproduce.py --board 6804 60 [--csv trace.csv]          # any board

The CSV has one row per step: step, decoded policy phase + target peg, the 5 actions, hand x/y/z/yaw/jaw gap, wrist force/torque,
and for every peg its true x/y/z, tilt (|axis z|, 1 = upright), grip contact and seated flag. Compare a failing target's rows with
a successful peg of the same type to see what differs (the suggested method)."""
import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cases_lib  # noqa: E402
import simlib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int)
    ap.add_argument("--board", type=int, nargs=2, metavar=("SEED", "EPISODE"))
    ap.add_argument("--policy", default=str(ROOT / "policy/c053"))
    ap.add_argument("--window", type=int, default=40, help="steps before/after the failure to print")
    ap.add_argument("--csv")
    a = ap.parse_args()
    case = None
    if a.case is not None:
        case = cases_lib.build_cases(cases_lib.load_census(ROOT / "data/failure_bank_c053.json.gz"))[a.case]
        seed, epi = case["seed"], case["episode"]
        print(f"case {case['id']}: board {seed}/{epi}, peg {case['peg']} ({case['shape']} {case['dia_mm']} mm, {case['start']})")
        print(f"cause: {dict(cases_lib.CATS)[case['cat']]}\nwhy: {case['why']}\n")
    else:
        seed, epi = a.board
    simlib.setup()
    src = Path(a.policy) / "src"
    for kv in (src / "ENV.txt").read_text().split():
        k, v = kv.split("=", 1)
        os.environ[k] = v
    sys.path.insert(0, str(src))
    import fx
    R = importlib.import_module("refport_fx")
    rows = []

    def on_step(i, ep, obs, act):
        code = int(R._decode(fx.NP, np.array([[act[0]]], np.float32))[1][0, 0])
        row = [ep.steps, simlib.PH[code % 16], (code // 16) % 8, *[round(float(x), 4) for x in act],
               *[round(float(x), 5) for x in ep.data.qpos[0:4]], round(float(ep.data.qpos[4] + ep.data.qpos[5]) * 1000, 2),
               *[round(float(x), 3) for x in obs[9:16]]]
        for q in range(ep.n_pegs):
            p, qq = ep.peg_pose(q)
            row += [round(float(x), 5) for x in p] + [round(abs(1 - 2 * (float(qq[1]) ** 2 + float(qq[2]) ** 2)), 3),
                                                      int(ep._gripper_touching(ep.peg_body[q])), int(ep.seated_now[q])]
        rows.append(row)

    res = simlib.run_boards(a.policy, [(seed, epi)], on_step=on_step)[0]
    c, _, _ = simlib.board(seed, epi)
    cond = simlib.conditions(c)
    print("board conditions:", {k: v for k, v in cond.items() if k != "pegs"})
    for i, pg in enumerate(cond["pegs"]):
        print(f"  peg {i}: {pg['shape']} {pg['dia_mm']} mm, start {pg['start']}, slot {pg['slot']}, seated at end: {res['seated'][i]}")
    print(f"result: score {res['score']:.4f}, {res['terminal']} at step {res['steps']}\n")
    head = ["step", "phase", "target", "a_x", "a_y", "a_z", "a_yaw", "a_grip", "hand_x", "hand_y", "hand_z", "hand_yaw", "gap_mm",
            "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "pad_force"]
    for q in range(c.n_pegs):
        head += [f"p{q}_x", f"p{q}_y", f"p{q}_z", f"p{q}_upright", f"p{q}_grip", f"p{q}_seated"]
    if case and case["fail_step"] is not None:
        f, peg = case["fail_step"], case["peg"]
        print(f"steps {f - a.window}..{f + 10} around the failure (peg {peg}: z mm / upright / grip):")
        base = 20 + 6 * peg
        for r in rows:
            if f - a.window <= r[0] <= f + 10:
                print(f"  {r[0]:5d} {r[1]:>4} t{r[2]} act {r[3]:+.2f} {r[4]:+.2f} {r[5]:+.2f} {r[6]:+.2f} {r[7]:+.2f}  hand z {r[10] * 1000:6.1f} gap {r[12]:5.1f}"
                      f"  |  peg z {r[base + 2] * 1000:6.1f} upright {r[base + 3]:.3f} grip {r[base + 4]}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)
        print(f"\ntrace: {len(rows)} steps -> {a.csv}")


if __name__ == "__main__":
    main()
