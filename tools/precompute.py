"""Precompute, for every board in the failure bank (all failing boards + the control sample), c053's own results and the board
conditions, so tools/evaluate.py never re-simulates the base and cases/ can list the exact conditions of each case.

    python tools/precompute.py --procs 300            # writes data/base_c053_results.json.gz and data/board_conditions.json.gz"""
import argparse
import gzip
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cases_lib  # noqa: E402
import simlib  # noqa: E402


def _task(chunk):
    res = simlib.run_boards(str(ROOT / "policy/c053"), chunk)
    conds = {}
    for s, e in chunk:
        c, _, _ = simlib.board(s, e)
        conds[f"{s}_{e}"] = simlib.conditions(c)
    return res, conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--control", type=int, default=4)
    a = ap.parse_args()
    census = cases_lib.load_census(ROOT / "data/failure_bank_c053.json.gz")
    fail = {(r["seed"], r["episode"]) for r in census if not r["seated"]}
    ctrl, _ = cases_lib.control_episodes(census, a.control)
    boards = sorted(fail | set(ctrl))
    n = max(1, min(len(boards), a.procs))
    chunks = [boards[i::n] for i in range(n)]
    print(f"{len(boards)} boards in {n} chunks", flush=True)
    with get_context("spawn").Pool(a.procs) as pool:
        out = pool.map(_task, chunks, chunksize=1)
    base = {f"{r['seed']}_{r['episode']}": r for res, _ in out for r in res}
    conds = {k: v for _, c in out for k, v in c.items()}
    with gzip.open(ROOT / "data/base_c053_results.json.gz", "wt") as f:
        json.dump(base, f)
    with gzip.open(ROOT / "data/board_conditions.json.gz", "wt") as f:
        json.dump(conds, f)
    print(f"wrote {len(base)} base results and {len(conds)} board conditions")


if __name__ == "__main__":
    main()
