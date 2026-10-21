"""Failure-case catalogue shared by the lab (worker.py), the dashboard and the tools: classify every unseated peg of the
failure bank (per-peg census of c053 on 8,192 boards, seeds 6801-6928 x 64 episodes) into a cause, explain it in plain words,
and give the step where it failed. No simulator import here: this is pure data processing."""
import gzip
import json
from pathlib import Path

CATS = [("tip_pick", "Knocked over while the policy picked / carried it"),
        ("tip_insert", "Fell over during / after its own insertion, release or retreat"),
        ("tip_other", "Knocked over while the hand worked on another peg"),
        ("tip_free", "Fell over with no hand contact (by itself or an environment push)"),
        ("lost_self", "Fell off the table itself (that ends the episode)"),
        ("part_lost", "Never seated: the episode ended early because another peg fell off the table"),
        ("squatter", "Squatter: started on another hole's collar and is stuck there"),
        ("blocked", "Its own hole is covered by a squatter peg"),
        ("timeout", "Ran out of time: repeated attempts that never seated it"),
        ("never", "Never attempted before the episode ended"),
        ("lying", "Started lying on its side: the policy has no way to stand it up"),
        ("other", "Other")]
SEEDS = list(range(6801, 6929))
EPISODES_PER_SEED = 64



def _classify(r):
    tip, ph = r.get("tip") or {}, (r.get("tip") or {}).get("ph")
    if r.get("start_lying"):
        return "lying"
    if r.get("lost"):
        return "lost_self"
    if r.get("squat_slot") is not None:
        return "squatter"
    if r.get("blocked_by"):
        return "blocked"
    if tip:
        if tip.get("own"):
            return "tip_pick" if ph in ("P1", "P2", "P3", "P4", "CAL", "I0", "I1", "KC", "KD") else "tip_insert"
        return "tip_other" if tip.get("hand_dt", 99) <= 30 else "tip_free"
    if r.get("terminal") == "part_lost":
        return "part_lost"
    if r.get("first") is None:
        return "never"
    if r.get("terminal") == "time_limit":
        return "timeout"
    return "other"

def _explain(r, cat, lost_in_ep):
    if (r.get("tip") or {}).get("ph") in ("J14", "J15"):
        r = {**r, "tip": {**r["tip"], "ph": {"J14": "KC", "J15": "KD"}[r["tip"]["ph"]]}}
    name = f"peg {r['peg']} ({r['shape']} {r['dia_mm']:g} mm, start '{r['start']}')"
    ev, lines = [], []
    tip = r.get("tip") or {}
    if r.get("first") is not None:
        lines.append(f"The policy first targeted it at step {r['first']}; it spent {r.get('tgt', 0)} steps on it, with {r.get('picks', 0)} pick(s) "
                     f"and {r.get('i2', 0)} insertion steps. Its last phase on this peg: {r.get('last_ph')}.")
        ev.append((r["first"], "first targeted"))
    if cat == "lying":
        lines.insert(0, f"{name} started LYING on its side. c053 skips lying pegs (no reliable stand-up yet), so it can never be seated."
                     " Try: push / roll it against a ring wall and tip it up, then grasp it upright.")
    if r.get("squat_slot") is not None:
        lines.insert(0, f"{name} started on top of hole {r['squat_slot']}'s collar (a 'squatter') and is wedged there.")
    if r.get("blocked_by"):
        lines.insert(0, f"{name}'s own hole is covered by squatter peg(s) {r['blocked_by']}, so it cannot be inserted.")
    if tip:
        who = "its own pick/insert" if tip.get("own") else f"work on peg {tip.get('target')}"
        hd = tip.get("hand_dt", 99)
        touch = (f"the hand touched it {hd} steps before (hand phase {tip.get('hand_ph')})" if hd <= 30 else "no hand contact in the last 30 steps")
        pg = f"; another peg touched it {tip['peg_dt']} steps before" if tip.get("peg_dt", 9999) <= 30 else ""
        dist = "; an environment push was scheduled then" if tip.get("disturbance") else ""
        lines.insert(0, f"{name} TIPPED OVER at step {tip['step']} during phase {tip.get('ph')} ({who}): {touch}{pg}{dist}. "
                     f"Its centre was at {tip.get('z_mm')} mm height.")
        ev.append((tip["step"], f"tipped over ({tip.get('ph')})"))
    if r.get("lost"):
        lines.insert(0, f"{name} fell off the table at step {r.get('end_step')}, which ended the episode (part_lost).")
        ev.append((r.get("end_step"), "left the table"))
    elif r.get("terminal") == "part_lost" and lost_in_ep:
        j, st = lost_in_ep
        lines.insert(0, f"{name} was not seated yet when peg {j} fell off the table at step {st}, which ended the episode.")
        ev.append((st, f"peg {j} left the table"))
    if cat == "timeout":
        lines.insert(0, f"{name} was still unseated when time ran out at step {r.get('steps')} of {r.get('horizon')}.")
    if cat == "never":
        lines.insert(0, f"{name} was never attempted before the episode ended ({r.get('terminal')} at step {r.get('steps')}).")
    if r.get("dist_target") not in (None, "none"):
        lines.append(f"Environment disturbance '{r['dist_target']}' at step {r.get('dist_step')}.")
        ev.append((r.get("dist_step"), f"env push '{r['dist_target']}'"))
    if r.get("foreign_end") is not None:
        lines.append(f"At the end it was standing in hole {r['foreign_end']}, the WRONG hole.")
    lines.append(f"End state: {'upright' if r.get('end_upright') else 'not upright'}, centre at {r.get('end_z_mm')} mm; episode {r.get('terminal')} "
                 f"at step {r.get('steps')}/{r.get('horizon')}; tier {r.get('tier')}.")
    ev = sorted({(int(a), b) for a, b in ev if a is not None})
    fail = tip.get("step") if tip else (r.get("end_step") if r.get("lost") else (lost_in_ep[1] if (r.get("terminal") == "part_lost" and lost_in_ep) else None))
    return " ".join(lines), ev, fail

def load_census(bank_path):
    bank_path = Path(bank_path)
    if not bank_path.exists():
        return []
    raw = gzip.open(bank_path, "rt").read() if bank_path.suffix == ".gz" else bank_path.read_text()
    return json.loads(raw)["census"]


def build_cases(census):
    """All unseated pegs as cases: cause, plain explanation, failure step, events, and what's needed to reproduce the board."""
    lost = {}
    for r in census:
        if r.get("lost"):
            lost[(r["seed"], r["episode"])] = (r["peg"], r.get("end_step"))
    out = []
    for r in census:
        if r["seated"]:
            continue
        cat = _classify(r)
        why, ev, fail = _explain(r, cat, lost.get((r["seed"], r["episode"])))
        first = r.get("first")
        go = max(0, (fail if fail is not None else (first if first is not None else 40)) - 40)
        tip = dict(r.get("tip") or {})
        if tip.get("ph") in ("J14", "J15"):          # the census names the two park phases J14/J15
            tip["ph"] = {"J14": "KC", "J15": "KD"}[tip["ph"]]
        out.append({"id": len(out), "seed": r["seed"], "episode": r["episode"], "peg": r["peg"], "start": r["start"], "shape": r["shape"],
                    "dia_mm": r["dia_mm"], "cat": cat, "why": why, "events": ev, "fail_step": fail, "goto": go,
                    "tip_phase": tip.get("ph"), "k": r.get("k"), "terminal": r["terminal"], "tier": r.get("tier"),
                    "horizon": r.get("horizon"), "picks": r.get("picks"), "insert_steps": r.get("i2"), "first_targeted": first})
    return out


def control_episodes(census, rate=4):
    """Boards where c053 seated every peg; every rate-th one (deterministic) is the control sample for regression checks."""
    bad = {(r["seed"], r["episode"]) for r in census if not r["seated"]}
    good = sorted({(r["seed"], r["episode"]) for r in census} - bad)
    return good[::rate], len(good)


PHASES = {"NEXT": "choose the next peg", "P1": "approach above the peg", "P2": "descend to the grasp height", "P3": "close the jaws",
          "P4": "lift", "CAL": "calibrate the grip offset (first pick)", "I0": "rise before the carry", "I1": "carry over the hole and align",
          "I2": "insert (descend into the hole with a small search spiral)", "I3": "open the jaws (release)", "I4": "rise off the seated peg",
          "I5": "hold clear until the peg counts as seated", "RT": "retreat", "IDLE": "nothing left to do: park high",
          "KC": "park-carry a lying round peg between rings", "KD": "park-drop it"}


def summary(cases, episodes_total=len(SEEDS) * EPISODES_PER_SEED):
    """Aggregates for the dashboard. Score lost is approximate: an unseated peg costs its board about 0.93 / k
    (a seated peg is worth 0.75-1.0, ~0.93 on average for c053; the board score is the mean over its k pegs)."""
    lab = dict(CATS)
    by = {}
    for c in cases:
        d = by.setdefault(c["cat"], {"key": c["cat"], "label": lab[c["cat"]], "n": 0, "boards": set(), "lost": 0.0,
                                     "size": {}, "shape": {}, "start": {}, "phase": {}})
        d["n"] += 1
        d["boards"].add((c["seed"], c["episode"]))
        d["lost"] += 0.93 / max(1, c["k"] or 1) / episodes_total
        for f, v in (("size", str(int(round(c["dia_mm"])))), ("shape", c["shape"]), ("start", c["start"]), ("phase", c.get("tip_phase"))):
            if v:
                d[f][v] = d[f].get(v, 0) + 1
    cats = sorted(by.values(), key=lambda d: -d["lost"])
    for d in cats:
        d["boards"] = len(d["boards"])
        d["lost"] = round(d["lost"], 5)
    return {"failed_pegs": len(cases), "boards_with_failure": len({(c["seed"], c["episode"]) for c in cases}),
            "episodes_total": episodes_total, "score_lost": round(sum(d["lost"] for d in cats), 4), "cats": cats,
            "phases": PHASES, "seeds": [SEEDS[0], SEEDS[-1]]}
