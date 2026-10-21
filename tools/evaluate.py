"""Measure a policy change ONLY where it matters (the user's method): run c053 and your candidate on the failing boards of the
causes you target, plus a control sample of boards where c053 seats every peg (to catch regressions elsewhere).

    python tools/evaluate.py --policy policy/my_fix --cause tip_pick [--cause tip_insert ...] [--control 4] [--procs 8]
    python tools/evaluate.py --policy policy/my_fix --all            # every failing board (+ control)

Reports: failing pegs fixed / still failing per cause, pegs lost on control boards, and the estimated change of the full
score over all 8,192 boards = (sum of board deltas on the evaluated failing boards + control_rate x control deltas) / 8192,
with a standard error over the 128 seeds. Boards of causes you didn't select are assumed unchanged."""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cases_lib  # noqa: E402
import simlib  # noqa: E402


def _task(x):
    pol, chunk = x
    return pol, simlib.run_boards(pol, chunk)   # many boards share each batched policy call (much faster than one at a time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--base", default=str(ROOT / "policy/c053"))
    ap.add_argument("--cause", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--control", type=int, default=4, help="use every N-th all-seated board as control (0 = none)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save")
    a = ap.parse_args()
    census = cases_lib.load_census(ROOT / "data/failure_bank_c053.json.gz")
    cases = cases_lib.build_cases(census)
    sel = [c for c in cases if a.all or c["cat"] in a.cause]
    if not sel:
        sys.exit("choose --cause <key> (see cases/README or the dashboard) or --all")
    fail_boards = sorted({(c["seed"], c["episode"]) for c in sel})
    ctrl, n_good = cases_lib.control_episodes(census, a.control) if a.control else ([], 0)
    boards = sorted(set(fail_boards) | set(ctrl))
    import hashlib
    base_sha = hashlib.sha256((Path(a.base) / "policy.onnx").read_bytes()).hexdigest()[:16]
    cache_f = ROOT / "cache" / f"base_{base_sha}.json"     # the base policy's results never change: run each board once
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    shipped = ROOT / "data/base_c053_results.json.gz"          # c053's results for every bank board ship with the repo
    if not cache and base_sha == "20283cda64dc9473" and shipped.exists():
        import gzip
        cache = json.loads(gzip.open(shipped, "rt").read())
    todo_base = [b for b in boards if f"{b[0]}_{b[1]}" not in cache]
    jobs = []
    for pol, bl in ((a.base, todo_base), (a.policy, boards)):
        if bl:
            n_chunks = max(1, min(len(bl), a.procs * 2))
            jobs += [(pol, bl[i::n_chunks]) for i in range(n_chunks)]
    print(f"{len(fail_boards)} failing boards ({len(sel)} failing pegs) + {len(ctrl)} control boards; base cached for "
          f"{len(boards) - len(todo_base)}; {len(jobs)} jobs on {a.procs} processes ...",
          flush=True)
    with get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(_task, jobs, chunksize=1)
    R = {(p, r["seed"], r["episode"]): r for p, rs in res for r in rs}
    for p_, r in [(p_, r) for p_, rs in res for r in rs if p_ == a.base]:
        cache[f"{r['seed']}_{r['episode']}"] = r
    if todo_base:
        cache_f.parent.mkdir(exist_ok=True)
        cache_f.write_text(json.dumps(cache))
    for s_, e_ in boards:
        R[(a.base, s_, e_)] = cache[f"{s_}_{e_}"]
    base, cand = a.base, a.policy
    fixed, still = defaultdict(int), defaultdict(int)
    for c in sel:
        ok = R[(cand, c["seed"], c["episode"])]["seated"][c["peg"]]
        (fixed if ok else still)[c["cat"]] += 1
    lost_ctrl = sum(sum(x and not y for x, y in zip(R[(base, s, e)]["seated"], R[(cand, s, e)]["seated"])) for s, e in ctrl)
    new_fail = sum(sum(x and not y for x, y in zip(R[(base, s, e)]["seated"], R[(cand, s, e)]["seated"])) for s, e in fail_boards)
    ctrl_set, D = set(ctrl), defaultdict(float)
    for s, e in boards:
        w = a.control if (s, e) in ctrl_set else 1.0
        D[s] += w * (R[(cand, s, e)]["score"] - R[(base, s, e)]["score"])
    per = [D[s] / cases_lib.EPISODES_PER_SEED for s in cases_lib.SEEDS]
    m = sum(per) / len(per)
    se = math.sqrt(sum((x - m) ** 2 for x in per) / (len(per) - 1) / len(per))
    print("\nper cause: failing pegs now seated / still failing")
    for k, lab in cases_lib.CATS:
        if fixed[k] or still[k]:
            print(f"  {k:11s} {fixed[k]:4d} fixed / {still[k]:4d} still failing   ({lab})")
    print(f"\nother pegs on those failing boards newly lost: {new_fail}")
    print(f"control boards: pegs lost {lost_ctrl} on {len(ctrl)} boards (each stands for {a.control} all-seated boards)")
    print(f"\nESTIMATED FULL-SCORE CHANGE: {m:+.5f} +- {se:.5f} per episode ({m / se if se else 0:+.1f} SE)")
    if a.save:
        Path(a.save).write_text(json.dumps({"policy": cand, "base": base, "causes": a.cause or "all", "delta": m, "se": se,
                                            "fixed": fixed, "still": still, "lost_ctrl": lost_ctrl, "new_fail": new_fail}, indent=1))


if __name__ == "__main__":
    main()
