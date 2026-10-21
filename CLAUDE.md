# NIST simulator lab: briefing for Claude

You are helping improve **c053**, a hand-built controller (exported as an ONNX policy) for the Apex SN1 competition
**NIST-Lite Robot Assembly** (#14, v0.2.0). A 2-finger gripper (moves in x/y/z and turns about the vertical; no tilt) must insert
3-6 round/square pegs (4, 8, 12, 16 mm) into their own holes on a board. c053 scores about **0.845**; the rival's trained
network scores **0.892**, so to take the crown we need ≥ ~0.901. This repo lets you study exactly where c053 fails and test fixes
**only on the failing boards**, which is fast.

Everything here uses the **competition's own released simulator** (`apex-competition-nist-assembly` at tag v0.2.0 = `62859c2`,
cloned by setup into `apex-competition-nist-assembly/`). Results are identical to the official scorer: a board is fixed by
`(seed, episode)` and replays exactly.

## Files

| path | what |
|---|---|
| `data/failure_bank_c053.json.gz` | per-peg census of c053 on 8,192 boards (seeds 6801-6928 × 64 episodes): seated or not, first seat, tip-over event (step, controller phase, target, last hand / peg contact), part_lost, squatter, end state |
| `data/base_c053_results.json.gz` | c053's own result on every failing board + the control sample, so evaluations only simulate your candidate |
| `data/board_conditions.json.gz` | each of those boards' conditions (k, friction, peg mass, clearance, observation noise, action delay, pegs) |
| `cases_lib.py` | turns the census into **cases** (one per unseated peg: cause, plain explanation, failure step, events); `summary()` for the dashboard |
| `simlib.py` | headless runner: `run_boards(policy_dir, [(seed, episode), ...])` with the exact official loop |
| `tools/reproduce.py` | replay one case, print the steps around the failure, write a per-step CSV trace |
| `tools/build_policy.py` | rebuild `policy.onnx` from controller source + `RF_*` switches |
| `tools/evaluate.py` | candidate vs c053 on the failing boards of chosen causes + control boards → fixed / lost / estimated score change |
| `policy/c053/` | `policy.onnx` (sha256 `20283cda...`) and `src/`: `refport_fx.py` (the controller, ~1,300 lines), `fx.py` (numpy/ONNX graph builder), `export_rf.py`, `ENV.txt` (the switches c053 is built with) |
| `server.py`, `worker.py`, `static/` | the browser lab (http://127.0.0.1:8765) and dashboard (`/dashboard.html`) |

## The failure causes (keys used by the tools)

| key | meaning | ≈ score lost / episode |
|---|---|---|
| `lying` | started lying on its side; c053 skips it (no stand-up method) | 0.040 |
| `part_lost` | never seated because ANOTHER peg fell off the table, which ends the episode | 0.0095 |
| `squatter` | started on another hole's collar, wedged (mostly immovable) | 0.0086 |
| `timeout` | repeated attempts, never seated before the time limit | 0.0053 |
| `blocked` | its own hole is covered by a squatter | 0.0053 |
| `tip_pick` | knocked over while c053 picked / carried it (phases P1-P4, CAL, I0, I1) | 0.0053 |
| `tip_free` | fell over with no hand contact in the last 30 steps (spawn instability, env push) | 0.0053 |
| `tip_insert` | fell over during / after its own insertion, release or retreat (I2-I5, RT) | 0.0042 |
| `lost_self` | fell off the table itself | 0.0025 |
| `tip_other` | knocked over while the hand worked on another peg | 0.0022 |
| `never` | never attempted before the episode ended | 0.0018 |

## How the controller works (refport_fx.py)

- It is a phase machine evaluated as one ONNX graph per step. It has no hidden state: its memory (phase, target peg, counters,
  estimates) is encoded in tiny offsets of its own previous action, which the observation echoes back (`_encode` / `_decode`).
- Phases: NEXT (choose a peg), P1 approach, P2 descend, P3 close, P4 lift, CAL calibrate (first pick), I0 rise, I1 carry +
  align, I2 insert (descend with a search spiral), I3 open, I4 rise, I5 hold clear, RT retreat, IDLE, KC/KD park a lying round
  peg. `nxt[PHASE]` holds the exit conditions, `acts[PHASE]` the 5 actions (dx, dy, dz, dyaw, grip) in [-1, 1].
- Inputs: the 184-dim observation (hand exact; peg / hole positions noisy and biased; wrist force/torque; pad force; last action).
- Every feature is behind an environment switch read at build time: `X = E("RF_X", "0") == "1"`; `ENV.txt` lists c053's.

## The method (please follow it)

1. **Pick one cause**, from the dashboard or the table above. Open its cases in the lab and watch several fail
   (**▶ Play to the failure**).
2. **Compare failing and successful histories.** `tools/reproduce.py --case <id> --csv` gives every step. Compare the failing
   target's last ~40 steps with a successful peg of the same shape, size and start, on the same or another board: speed, height,
   jaw gap, contacts, wrist force, estimate error. Find what the successful ones did differently.
3. **Write the fix as a DEFAULT-OFF switch** in `policy/c053/src/refport_fx.py` (or a copy). With the switch off, the build must
   reproduce c053 byte for byte: `python tools/build_policy.py --out policy/check` must print `20283cda... (= c053, unchanged)`.
   Keep each change small; don't touch the echo encoding unless you must.
4. **Build and measure on that cause only:**
   `python tools/build_policy.py --out policy/my_fix --set RF_MYFIX=1` then
   `python tools/evaluate.py --policy policy/my_fix --cause <key> --control 8`.
   Good = many failing pegs fixed, ~0 pegs lost on control boards, and a positive estimated change of at least 2 SE.
   Then re-run with `--control 4` (more control boards) before reporting.
5. **Report back** with the switch name and value, the diff of `refport_fx.py`, and the evaluate output. Those are what the team
   needs to merge it and confirm it on fresh boards.

Lessons from earlier attempts: removing futile work wins; retries and detours on failed pegs usually lose. Lighter grips, early
releases and faster grip closes cost more seats than they save time. Lying pegs can't be grasped and rotated: the jaws only turn
about the vertical, and a gripped peg slips in twist. The one stand-up that worked pushes the upper rim of a lying 12/16 mm peg
against a ring's outer wall with the jaws closed (≈ 12% of attempts).

## Rules of the world (enforced by the scorer)

25 control steps/s; per step the hand moves ≤ 4 mm per axis (action ±1) and ≤ 0.08 rad in yaw (target clipped to ±3.1 rad);
moves below ~0.05 mm are eaten by backlash; boards have a 0-3 step action delay. Grip target gap = (a + 1) × 12 mm. Seated =
in its own hole, upright within 10°, bottom on the hole floor, jaws ≥ 2 mm above its top and not touching it, for 5 steps; only
pegs still seated at the end score. A peg leaving |x| or |y| > 0.30 m ends the episode (part_lost). Peg score = 0.75 + 0.25 ×
(0.85 × time left at its first seat + 0.15 × (1 − mean Σa² / 5)).
