"""Headless helpers around the competition's released simulator, used by tools/*.py.

A board is (seed, episode): the official conditions are nist_sim.conditions.sample_episodes(seed, 8, "live")[episode], and its two
random streams are SeedSequence([seed, 0x5245, episode]).spawn(2) (physics/episode rng, observation-noise rng). The loop is the
official one: observe (rounded to 6 decimals, float32) -> act -> step. So every result here matches the referee exactly."""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFEREE = Path(os.environ.get("NIST_REFEREE", HERE / "apex-competition-nist-assembly"))
PH = ["NEXT", "P1", "P2", "P3", "P4", "I0", "I1", "I2", "I3", "I4", "I5", "RT", "IDLE", "CAL", "KC", "KD"]


def setup(referee=None):
    ref = Path(referee or REFEREE)
    if not (ref / "referee/nist_sim/env.py").exists():
        sys.exit(f"simulator not found in {ref}: run setup first (or set NIST_REFEREE)")
    if str(ref / "referee") not in sys.path:
        sys.path.insert(0, str(ref / "referee"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def board(seed, episode):
    from nist_sim.conditions import sample_episodes
    from nist_sim.env import Episode
    _, conds = sample_episodes(int(seed), 8, "live")
    c = conds[int(episode)]
    r1, r2 = (np.random.default_rng(s) for s in np.random.SeedSequence([int(seed), 0x5245, int(episode)]).spawn(2))
    return c, Episode(c, r1), r2


def conditions(c):
    return {"k": c.n_pegs, "horizon": int(c.max_steps), "friction": round(float(c.friction), 4),
            "peg_mass_g": round(float(c.peg_mass) * 1000, 2), "clearance_mm": round(float(c.clearance) * 1000, 4),
            "obs_noise_mm": round(float(c.obs_noise) * 1000, 3), "act_delay": int(c.act_delay), "clutter": len(c.clutter),
            "pegs": [{"shape": p.shape(), "dia_mm": round(p.diameter() * 1000, 1), "start": p.start_type, "slot": int(p.slot)}
                     for p in c.pegs]}


def policy(policy_dir):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(Path(policy_dir) / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    return lambda obs: sess.run(None, {"obs": obs})[0]


def run_boards(policy_dir, boards, referee=None, on_step=None):
    """Run whole episodes for [(seed, episode), ...] (one seed per call is fastest); returns one result dict per board."""
    setup(referee)
    from nist_sim.scoring import episode_score
    act = policy(policy_dir)
    eps = [board(s, e) for s, e in boards]
    while True:
        ids = [i for i, (_, e, _) in enumerate(eps) if not e.done]
        if not ids:
            break
        obs = np.round(np.stack([eps[i][1].observe(eps[i][2]) for i in ids]), 6).astype(np.float32)
        a = act(obs)
        for k, (i, ai) in enumerate(zip(ids, a)):
            if on_step:
                on_step(i, eps[i][1], obs[k], ai)
            eps[i][1].step(np.asarray(ai, dtype=np.float64))
    out = []
    for (s, e), (c, ep, _) in zip(boards, eps):
        seated, first = ep.peg_seated_at_end(), ep.peg_first_seat_steps()
        out.append({"seed": s, "episode": e, "score": episode_score(seated, first, c.max_steps, ep.action_l2 / max(1, ep.steps)),
                    "seated": seated, "first_seat": first, "terminal": ep.terminal_reason, "steps": ep.steps})
    return out
