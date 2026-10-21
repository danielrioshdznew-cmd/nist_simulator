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

## What you can do

| panel | what it does |
|---|---|
| **Episode** | Any board: seed + episode (0-63). The conditions line shows friction, peg mass, hole clearance, observation noise, action delay. |
| **Failure cases** | The ~3,600 pegs c053 failed to seat on 8,192 boards (seeds 6801-6928, `data/failure_bank_c053.json.gz`), with their cause: lying start, tipped over (and in which phase), wedged/squatter, blocked, part_lost. **Go to case** runs the policy up to ~40 steps before the failure. |
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

`server.py` (web server + relay), `worker.py` (simulator + policy, JSON over stdin/stdout), `static/` (three.js page; three.js r160
bundled in `static/vendor/`), `policy/c053/` (policy.onnx sha256 20283cda..., `src/` = the controller source the phase names are
decoded with), `data/failure_bank_c053.json.gz` (per-peg failure census).
