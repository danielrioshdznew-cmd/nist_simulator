"""NIST simulator lab worker: the REAL released v0.2.0 referee simulator (nist_sim.env.Episode, MuJoCo) + a frozen policy (ONNX),
driven by one JSON request per stdin line, one JSON reply per stdout line. server.py starts it under netns-run (no network).

The step order is the official one: observe -> act -> step -> observe -> ... Each observation is computed once and is the one
the next action sees, so a policy-only run reproduces the official episode exactly. Manual actions replace the policy's action
for that step; the policy's proposal is still computed (it has no hidden state: its memory is the action echo in the obs), and
after manual steps the policy reads YOUR last action as its memory, so expect it to be confused when you hand back.

    python -I -B worker.py <lab dir> <referee repo root (contains referee/nist_sim)> <policy dir> <failure bank .json[.gz]>"""
import gzip
import base64
import copy
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

NIST, EXPORT, POLICY, BANK = (Path(x) for x in sys.argv[1:5])
OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # anything else printed (MuJoCo warnings) goes to stderr, never into the protocol

for kv in (POLICY / "src/ENV.txt").read_text().split():
    k, v = kv.split("=", 1)
    os.environ[k] = v
os.environ["OMP_NUM_THREADS"] = "1"
sys.path[:0] = [str(EXPORT / "referee"), str(POLICY / "src")]
import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402
from nist_sim import env as E  # noqa: E402
from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.scoring import episode_score  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases_lib  # noqa: E402
import fx  # noqa: E402
import refport_fx as R  # noqa: E402

PH = ["NEXT", "P1 approach", "P2 descend", "P3 close", "P4 lift", "I0 rise", "I1 carry/align", "I2 insert", "I3 open",
      "I4 rise", "I5 hold", "RT retreat", "IDLE", "CAL calibrate", "KC park-carry", "KD park-drop"]
GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def axis(q):
    w, x, y, z = q
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]


class Lab:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(POLICY / "policy.onnx"), so, providers=["CPUExecutionProvider"])
        self.snaps, self.e, self.census, self.clip_cache = {}, None, None, {}

    # ---- episode control -------------------------------------------------------------------------------------------
    def reset(self, seed, episode):
        _, conds = sample_episodes(int(seed), 8, "live")
        c = conds[int(episode)]
        r1, r2 = (np.random.default_rng(s) for s in np.random.SeedSequence([int(seed), 0x5245, int(episode)]).spawn(2))
        self.e, self.noise, self.c = E.Episode(c, r1), r2, c
        self.seed, self.episode = int(seed), int(episode)
        self.actions = []
        self.obs = self._observe()
        self.pa = self._policy(self.obs)
        return self.scene()

    def _observe(self):
        return np.round(self.e.observe(self.noise), 6).astype(np.float32)

    def _policy(self, o):
        return self.sess.run(None, {"obs": o[None]})[0][0]

    def step(self, n=1, mode="policy", action=None):
        e = self.e
        for _ in range(int(n)):
            if e.done:
                break
            a = self.pa if mode == "policy" else np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            self.actions.append({"step": e.steps, "mode": mode, "action": [round(float(x), 6) for x in a]})
            e.step(np.asarray(a, dtype=np.float64))
            if not e.done:
                self.obs = self._observe()
                self.pa = self._policy(self.obs)
        return self.state()

    def goto(self, seed, episode, step):
        self.reset(seed, episode)
        while self.e.steps < int(step) and not self.e.done:
            self.step(1, "policy")
        return {**self.scene(), **self.state()}

    def snapshot(self, name):
        self.snaps[name] = copy.deepcopy((self.e, self.noise, self.obs, self.pa, self.actions, self.c, self.seed, self.episode))
        return {"snapshots": sorted(self.snaps), "saved": name, "step": self.e.steps}

    def restore(self, name):
        (self.e, self.noise, self.obs, self.pa, self.actions, self.c, self.seed, self.episode) = copy.deepcopy(self.snaps[name])
        return {**self.scene(), **self.state()}

    # ---- views ----------------------------------------------------------------------------------------------------
    def scene(self, e=None, c=None, seed=None, episode=None):
        e, c = e or self.e, c or self.c
        seed, episode = (self.seed, self.episode) if seed is None else (seed, episode)
        m = e.model
        geoms = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            geoms.append({"type": GEOM_TYPES.get(int(m.geom_type[g]), "other"), "size": [float(x) for x in m.geom_size[g]],
                          "rgba": [float(x) for x in m.geom_rgba[g]],
                          "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
                          "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""})
        pegs = [{"i": i, "shape": p.shape(), "dia_mm": round(p.diameter() * 1000, 1), "start": p.start_type, "slot": int(p.slot)}
                for i, p in enumerate(c.pegs)]
        return {"seed": seed, "episode": episode, "horizon": int(c.max_steps), "geoms": geoms, "pegs_meta": pegs,
                "board_xy": [float(x) for x in c.board_xy], "board_yaw": float(c.board_yaw),
                "conditions": {"friction": round(float(c.friction), 3), "peg_mass_g": round(float(c.peg_mass) * 1000, 1),
                               "clearance_mm": round(float(c.clearance) * 1000, 3), "obs_noise_mm": round(float(c.obs_noise) * 1000, 2),
                               "act_delay": int(c.act_delay), "clutter": len(c.clutter), "k": c.n_pegs}}

    def state(self):
        e, c, o = self.e, self.c, self.obs
        d = e.data
        frames = np.concatenate([d.geom_xpos, d.geom_xmat], axis=1).astype(np.float32)
        bx, by = c.board_xy
        code = int(R._decode(fx.NP, np.array([[self.pa[0]]], np.float32))[1][0, 0])
        pegs = []
        for i in range(e.n_pegs):
            p, q = e.peg_pose(i)
            f = o[E.PROPRIO_DIM + E.ACT_DIM + i * E.PEG_FEAT: E.PROPRIO_DIM + E.ACT_DIM + (i + 1) * E.PEG_FEAT]
            est = [float(f[1] + f[13]) / 10 + bx, float(f[2] + f[14]) / 10 + by, float(f[3]) / 10 + E.BLOCK_H]
            hole_est = [float(f[13]) / 10 + bx, float(f[14]) / 10 + by]
            ax = axis(q)
            pegs.append({"i": i, "pos": [float(x) for x in p], "az": round(abs(ax[2]), 3), "est": est, "hole": [float(x) for x in e.slot_xy[i]],
                         "hole_est": hole_est, "seated_now": bool(e.seated_now[i]), "first_seat": e.first_seat_step[i],
                         "touch": bool(e._gripper_touching(e.peg_body[i])), "obs_seated": bool(f[16] > 0.5)})
        hand = d.qpos[0:4]
        out = {"step": int(e.steps), "done": bool(e.done), "terminal": e.terminal_reason, "frames": base64.b64encode(frames.tobytes()).decode(),
               "hand": {"xyz": [float(x) for x in hand[:3]], "yaw": float(hand[3]), "gap_mm": round(float(d.qpos[4] + d.qpos[5]) * 1000, 2),
                        "target": [float(x) for x in e.target], "grip_cmd_mm": round(float(e.grip) * 1000, 2),
                        "force": [round(float(x), 3) for x in o[9:12]], "torque": [round(float(x), 3) for x in o[12:15]], "pad": round(float(o[15]), 3)},
               "policy": {"action": [round(float(x), 4) for x in self.pa], "phase": PH[code % 16], "target": (code // 16) % 8},
               "pegs": pegs, "n_actions": len(self.actions), "msa": round(e.action_l2 / max(1, e.steps), 3)}
        if e.done:
            seated, first = e.peg_seated_at_end(), e.peg_first_seat_steps()
            out["score"] = round(episode_score(seated, first, c.max_steps, e.action_l2 / max(1, e.steps)), 4)
            out["seated_at_end"] = seated
        return out

    # ---- failure bank (cases_lib.py) ------------------------------------------------------------------------------
    def _load_cases(self):
        if self.census is None:
            self.census = cases_lib.load_census(BANK)
            self.all_cases = cases_lib.build_cases(self.census)

    def categories(self):
        self._load_cases()
        n = {}
        for c in self.all_cases:
            n[c["cat"]] = n.get(c["cat"], 0) + 1
        return {"total": len(self.all_cases), "boards": len({(c["seed"], c["episode"]) for c in self.all_cases}),
                "episodes_total": len(cases_lib.SEEDS) * cases_lib.EPISODES_PER_SEED,
                "cats": [{"key": k, "label": lab, "n": n.get(k, 0)} for k, lab in cases_lib.CATS if n.get(k)]}

    def summary(self):
        self._load_cases()
        return cases_lib.summary(self.all_cases)

    def cases(self, cause="all", limit=0):
        self._load_cases()
        out = [c for c in self.all_cases if cause in ("all", c["cat"])]
        return {"cases": out[:limit] if limit else out, "n": len(out)}

    def _window(self, cs, before, after):
        f = cs["fail_step"]
        if f is not None:
            return max(0, f - before), f + after
        start = cs.get("first_targeted") or 0
        return start, start + before + after

    def clips(self, ids, before=75, after=75):
        """Movies of failure cases: each is simulated from step 0 with the policy (exactly as scored) and recorded from
        `before` steps ahead of its failure to `after` steps past it. Several cases run together, sharing each policy call."""
        self._load_cases()
        out, todo = {}, []
        for i in ids:
            key = (int(i), int(before), int(after))
            if key in self.clip_cache:
                out[int(i)] = self.clip_cache[key]
            else:
                todo.append(int(i))
        runs = []
        for i in todo:
            cs = self.all_cases[i]
            _, conds = sample_episodes(int(cs["seed"]), 8, "live")
            c = conds[int(cs["episode"])]
            r1, r2 = (np.random.default_rng(x) for x in np.random.SeedSequence([int(cs["seed"]), 0x5245, int(cs["episode"])]).spawn(2))
            e = E.Episode(c, r1)
            m = e.model
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "" for g in range(m.ngeom)]
            dyn = [g for g, b in enumerate(names) if b.startswith(("peg", "clutter")) or b in ("hand", "fl", "fr")]
            a, b = self._window(cs, before, after)
            runs.append({"i": i, "cs": cs, "c": c, "e": e, "noise": r2, "dyn": dyn, "a": a, "b": min(b, int(c.max_steps)),
                         "frames": [], "steps": [], "phases": [], "targets": [], "az": [], "touch": [], "pos": [], "static": None})
        while True:
            live = [r for r in runs if not r["e"].done and r["e"].steps <= r["b"]]
            if not live:
                break
            obs = np.round(np.stack([r["e"].observe(r["noise"]) for r in live]), 6).astype(np.float32)
            acts = self.sess.run(None, {"obs": obs})[0]
            for r, act in zip(live, acts):
                e = r["e"]
                if e.steps >= r["a"]:
                    d = e.data
                    if r["static"] is None:
                        r["static"] = np.concatenate([d.geom_xpos, d.geom_xmat], axis=1).astype(np.float32)
                    r["frames"].append(np.concatenate([d.geom_xpos[r["dyn"]], d.geom_xmat[r["dyn"]]], axis=1).astype(np.float32))
                    code = int(R._decode(fx.NP, np.array([[act[0]]], np.float32))[1][0, 0])
                    k = r["cs"]["peg"]
                    p, q = e.peg_pose(k)
                    r["steps"].append(int(e.steps))
                    r["phases"].append(PH[code % 16].split(" ")[0])
                    r["targets"].append((code // 16) % 8)
                    r["az"].append(round(abs(1 - 2 * (float(q[1]) ** 2 + float(q[2]) ** 2)), 3))
                    r["touch"].append(int(e._gripper_touching(e.peg_body[k])))
                    r["pos"].append([round(float(x), 5) for x in p])
                e.step(np.asarray(act, dtype=np.float64))
        for r in runs:
            cs = r["cs"]
            clip = {"case": {k: cs[k] for k in ("id", "seed", "episode", "peg", "shape", "dia_mm", "start", "cat", "why", "fail_step", "terminal")},
                    "label": dict(cases_lib.CATS)[cs["cat"]], "scene": self.scene(r["e"], r["c"], cs["seed"], cs["episode"]),
                    "static": base64.b64encode(r["static"].tobytes()).decode() if r["static"] is not None else None,
                    "dyn": r["dyn"], "frames": base64.b64encode(np.stack(r["frames"]).tobytes()).decode() if r["frames"] else None,
                    "steps": r["steps"], "phases": r["phases"], "targets": r["targets"], "az": r["az"], "touch": r["touch"],
                    "pos": r["pos"], "end": r["e"].terminal_reason if r["e"].done else "running"}
            self.clip_cache[(r["i"], int(before), int(after))] = clip
            out[r["i"]] = clip
        return {"clips": [out[int(i)] for i in ids]}

    def export(self):
        return {"seed": self.seed, "episode": self.episode, "actions": self.actions, "step": self.e.steps}


def main():
    lab = Lab()
    OUT.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        try:
            r = json.loads(line)
            cmd = r.pop("cmd")
            res = {"reset": lambda: {**lab.reset(**r), **lab.state()}, "step": lambda: lab.step(**r), "goto": lambda: lab.goto(**r),
                   "snapshot": lambda: lab.snapshot(**r), "restore": lambda: lab.restore(**r), "cases": lambda: lab.cases(**r),
                   "categories": lab.categories, "summary": lab.summary, "clips": lambda: lab.clips(**r),
                   "export": lab.export, "state": lab.state}[cmd]()
        except Exception as ex:  # noqa: BLE001
            res = {"error": repr(ex), "trace": traceback.format_exc()[-1500:]}
        OUT.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
