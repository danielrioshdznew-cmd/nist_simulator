# NIST simulator lab

A browser lab for the Apex SN1 **NIST-Lite Robot Assembly** competition (#14, v0.2.0). The physics is the **competition's own
released simulator** (MuJoCo, `nist_sim.env.Episode` from the public repo at tag v0.2.0 = `62859c2`). The gripper is driven by
the **c053 policy** (our controller, an ONNX file) or by you. three.js only draws what the simulator computes, so whatever works
here works the same way in the official scorer. A policy-only run reproduces the official episode exactly.

## Setup (once)

Needs Python 3.12 and git.

| OS | setup | start |
|---|---|---|
| Windows (PowerShell) | `.\setup.ps1` | `.\run.ps1` |
| Linux / macOS | `./setup.sh` | `./run.sh` |

Setup creates `.venv` (numpy 2.5.4, mujoco 3.16.0, onnxruntime 1.31.0) and clones the public competition repo at `62859c2` into
`apex-competition-nist-assembly/`. Then open **http://127.0.0.1:8765**. The status line at the top of the panel says
"● simulator connected" when everything works, or names the problem in red.

Options: `--port N`, `--host 0.0.0.0` (reach it from another machine), `--policy policy/<other>` (any folder with `policy.onnx` +
`src/`).

## Failure movies

**http://127.0.0.1:8765/movies.html** shows **four failure cases at once, playing like videos in sync**, from ~3 s before each
failure to ~1 s after it, looping. Each clip is labelled with its case number and cause, board, peg and failure step, and carries
the explanation, a live step / phase / uprightness readout, a red marker on the failing peg and a **FAILS NOW** flash. Choose a
cause, page with **Next 4 / Prev 4**, pause and scrub. Type your diagnosis in the box under a clip: notes are saved to
`exports/notes.json` (case id, board, peg, your text) so Claude can read them and turn them into fixes. "open in lab" opens that
exact case in the interactive lab.

## Dashboard

**http://127.0.0.1:8765/dashboard.html** (or the "Dashboard ↗" link in the lab). It shows:
- where the score goes: score lost per episode by failure cause; click a cause to open its cases in the lab;
- failed pegs by peg size and by start type;
- the controller phase at which pegs tip over;
- a glossary of every cause, controller phase and term.

## For Claude on your machine

`CLAUDE.md` briefs a Claude session on everything: the failure data, how the controller works, the method, and the tools.

```bash
python tools/reproduce.py --case 123 --csv trace_123.csv            # replay one failure, per-step trace
python tools/build_policy.py --out policy/my_fix --set RF_MYFIX=1   # rebuild the policy with a new switch
python tools/evaluate.py --policy policy/my_fix --cause tip_pick --control 8   # test only on that cause's failing boards
```
Run them with the setup's Python (`.venv\Scripts\python` on Windows, `.venv/bin/python` elsewhere).
`data/failure_cases.json.gz` lists all 3,604 cases with their cause, explanation, failure step and board conditions.

## What you can do

| panel | what it does |
|---|---|
| **Episode** | Any board: seed + episode (0-63). The conditions line shows friction, peg mass, hole clearance, observation noise, action delay. |
| **Failure cases** | ALL 3,604 pegs c053 failed to seat on 8,192 boards (2,316 boards with a failure; seeds 6801-6928, `data/failure_bank_c053.json.gz`), in 12 causes with counts: knocked over while picking / carrying, fell over at its own insertion or release, knocked over while the hand worked on another peg, fell over by itself, fell off the table, episode ended by another peg falling off, squatter, blocked hole, ran out of time, never attempted, lying start. Filter by size and shape. Choosing a case jumps there, marks the failing peg with a red ring and arrow, and explains **why** it failed (what happened, at which step and phase, contacts, pushes, end state); the event steps are links. **▶ Play to the failure** runs the policy and pauses just after it fails; **⟲ Just before the failure** goes back. The page opens on a failure. |
| **Run** | Play at real speed (25 steps/s) or faster, Step, +25. Driver = Policy or Manual. |
| **Manual control** | **W/S** +y/−y, **A/D** −x/+x, **R/F** up/down, **Q/E** yaw. **G** toggles the jaws; the slider sets the exact jaw gap target (0-24 mm). "move" = action size (1.0 = the 4 mm/step maximum). Hold keys while playing, or press Step. |
| **Snapshots** | Save the exact simulator state, try something, Restore, try again. Unlimited tries from the same moment. |
| **Export my actions** | Saves every action since the reset (policy and manual, labelled) in `exports/` and offers a download. |
| **View** | Top / side / follow-hand camera. **Estimates**: wire spheres = where the policy *thinks* each peg is; coloured rings = where it thinks each hole is; white rings = the true holes. **See-through hand**. |

## Suggested workflow for a blocker

1. Failure cases → pick a cause (e.g. "tipped over") → **List** → choose one → **Go to case**.
2. **Save** a snapshot. **Play** with the Policy driver to watch it fail.
3. **Restore**, switch to **Manual**, and try your idea from the same moment (e.g. slower descent, a different grasp height, a push).
   Restore and repeat until something works on that case.
4. Try the same idea on 2-3 other cases of the same cause. The interesting fixes work across cases, not just one.
5. **Export my actions** and send the file with a sentence on the idea.

## The rules of this world (what the scorer enforces)

- **Control:** 25 steps/s. Per step the hand target moves by at most 4 mm per axis (action ±1) and the yaw by 0.08 rad (yaw
  target clipped to ±3.1 rad). Tiny moves are eaten by backlash (about 0.05 mm, i.e. |action| < ~0.0125), and each board has a
  0-3 step **action delay**.
- **Jaws:** the 5th action sets the target gap = (a + 1) × 12 mm (0 = closed, 24 mm = fully open). The fingers are position
  servos (kp 400, ≤ 15 N), so a smaller target squeezes harder.
- **Pads:** 16 × 16 mm, their bottom 4 mm above the hand point; the palm is 40 × 60 mm. Parts slide and twist with friction
  (condim 4) but don't resist **rolling**: a lying round peg that starts moving rolls until something stops it.
- **Seated** = in its own hole, upright within 10°, bottom at the hole floor, the pads above its top + 2 mm and not touching
  it, for 5 steps in a row. A peg is scored only if it is still seated at the end, so the hand must end clear and high.
- **Episode ends** when all pegs are seated, at the time limit (280 steps × number of pegs), or at once if any peg leaves the
  table (|x| or |y| > 0.30 m), which is `part_lost`.
- **Score per seated peg** = 0.75 + 0.25 × (0.85 × time left at its first seat + 0.15 × effort term). Unseated = 0.
- The policy sees only **noisy, biased** estimates (spheres/rings in the view). The hand position is exact. The wrist force/torque
  and pad force are in the "State" panel.
- The policy keeps its memory inside its own last action, so after manual steps it reads *your* action as its memory. Hand
  back to the policy only to watch, not to judge a fix.

## Files

| path | what |
|---|---|
| `server.py`, `worker.py`, `static/` | the lab (three.js r160 bundled in `static/vendor/`) and `static/dashboard.html` |
| `cases_lib.py`, `simlib.py`, `tools/` | the failure catalogue, the headless runner, and the reproduce / build / evaluate / precompute tools |
| `policy/c053/` | `policy.onnx` (sha256 20283cda...) and `src/` (controller source, graph builder, exporter, switches) |
| `data/` | the failure bank census, c053's results on those boards, board conditions, and the case list |
| `CLAUDE.md` | briefing for a Claude session working on fixes |
