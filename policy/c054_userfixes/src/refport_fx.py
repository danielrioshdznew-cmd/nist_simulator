"""4e (2026-10-19): c3's refport (the author's reference_solver ScriptedPeg, obs-only, BYAW+BIAS+TCLAMP+PAVG+SQHAND) as ONE
exportable graph for the 2ccb2d3 observation (184 = proprio 16 + last_action 5 + 8 pegs x 17 + 3 clutter x 9), written against
ctrl/fx.py. Upright pegs only in this version (lying skipped, as the author does on busy boards); no clutter push yet.

Frame: board-centre x/y (as observed; the hand is exact), absolute z. Every peg/slot estimate carries one fixed per-episode bias:
removed by a per-step Procrustes fit of the slot rows against the public ATB layout (yaw + translation, df 12:58); hole centres
are taken from the fitted layout R(yaw) L_k.

Memory rides in the action echo (env.last_action): every channel is a command quantized to G = 1/128 plus a payload in (0, G)
that encodes 10 bits. Payload moves are below the env's backlash deadzone (>= 0.05 mm = 0.0125 action) and its yaw deadzone, so
they never move the hand; obs rounding (1e-6) is far below the payload step (7.6e-6).
    ch0: phase (4 bits) + 16 * current peg row (3 bits)
    ch1: step counter in the phase (8 bits)
    ch2, ch3: smoothed x, y of the current peg (rest: absolute, +-0.25 m; held: offset vs the hand, +-16 mm)
    ch4: mask of peg rows already attempted (8 bits)
Phases follow ScriptedPeg: P1 above / P2 descend / P3 close 12 / P4 lift 75 mm / I0 raise / I1 align (<= 140) / I2 descend with
lip spiral (<= 200) / I3 open 8 / I4 retreat 80 mm / I5 hold 13 / RT up 12 / NEXT / IDLE.
"""
import math
import os

DXYZ, DYAW = 0.004, 0.08
PEG_LEN, BLOCK_H, PAD_CENTRE_Z, SEATED_HOLD = 0.050, 0.020, 0.012, 5
CLOSED_CAL = os.environ.get("RF_CLOSED_CAL", "0") == "1"
CLOSED_CAL_SQ = os.environ.get("RF_CLOSED_CAL_SQ", "0") == "1"
# user's failure-movie notes (2026-10-21), each default-off:
BF_PEN = float(os.environ.get("RF_BF_PEN", "0"))     # bigger first: cost penalty for the smaller of two close pegs
BF_GAP = float(os.environ.get("RF_BF_GAP", "0.008"))  #   "close" = centre distance < (dia_a + dia_b) / 2 + BF_GAP
TL_PEN = float(os.environ.get("RF_TL_PEN", "0"))      # standing before tilted: cost penalty for a peg that reads tilted
TL_AZ = float(os.environ.get("RF_TL_AZ", "0.97"))     #   tilted = observed |axis z| below this (and not lying)
WIDE = float(os.environ.get("RF_WIDE", "0"))          # open wider on small pegs: gap = dia + PREOPEN + WIDE * (16 mm - dia)
WIDE_PICK_ONLY = os.environ.get("RF_WIDE_PICK_ONLY", "0") == "1"  # isolate approach width; release stays at the baseline opening
P2GATE = float(os.environ.get("RF_P2GATE", "0"))      # P2: hold the pads this far above the peg top until yaw / xy have settled
P2G_YAW = float(os.environ.get("RF_P2G_YAW", "0.04"))
P2G_XY = float(os.environ.get("RF_P2G_XY", "0.0015"))
PARK_LAST = os.environ.get("RF_PARK_LAST", "0") == "1"  # user: standing pegs first; park lying round pegs only once none are left
PARK_CHASE = float(os.environ.get("RF_PARK_CHASE", "-1"))  # user: catch a rolling peg fast: aim at its latest position + this x
                                                         #   (latest - smoothed) as a lead in its rolling direction (-1 = off)
PARK_CATCH = float(os.environ.get("RF_PARK_CATCH", "0"))  # user: catch a rolling peg fast: start lowering within this xy distance
CATCH_ARR = float(os.environ.get("RF_CATCH_ARR", "0.005"))  #   and accept arrival within this xy / CATCH_YAW yaw (jaws wide open)
CATCH_YAW = float(os.environ.get("RF_CATCH_YAW", "0.2"))
PARK_SETDOWN = os.environ.get("RF_PARK_SETDOWN", "0") == "1"  # user: lying round pegs first, even when the episode starts with a
                                                         #   peg in the hand: set that peg down upright where it is, park, come back
TILT_GRIP = os.environ.get("RF_TILT_GRIP", "0") == "1"  # user (#262 #35 #40): a leaning peg: aim at its top, jaws closing ALONG the lean,
TG_AZ = float(os.environ.get("RF_TG_AZ", "0.97"))        #   so the two vertical pads push it upright as they close (pivot on its foot)
RELMIN = float(os.environ.get("RF_RELMIN", "0"))         # user (#30 #197): release by opening only to dia + RELMIN (not +PREOPEN):
                                                         #   the opening pads don't sweep into a neighbouring peg
MAX_OPEN = 0.012   # per finger (env.MAX_OPEN): gap = 2 * MAX_OPEN * (a4 + 1) / 2
G = 1.0 / 128.0
NLEV = 3000   # payload levels per channel (step 2.6e-6 >> obs rounding 5e-7)
STEP = G / NLEV
OBS_DIM, ACT0, PEG0, PEGF = 184, 16, 21, 17
NEXT, P1, P2, P3, P4, I0, I1, I2, I3, I4, I5, RT, IDLE, CAL = range(14)
NPH = 14

E = os.environ.get
KP = float(E("RF_KP", "0.8"))                 # servo gain on (goal - measured hand)
PAD = float(E("RF_PAD", "0.0002"))
PAD_TH = float(E("RF_PAD_TH", "0.00015"))     # pad only when |err| > this (df: down to ~0.05 mm so the hand never parks in the deadzone)            # backlash padding (m) when |err| > 0.15 mm
YPAD = float(E("RF_YPAD", "0.004"))           # yaw deadzone padding (rad)
TOL = float(E("RF_TOL", "0.0006"))            # goto arrival tolerance (m), author 0.6 mm
VTOL = float(E("RF_VTOL", "0.02"))            # arrival hand speed (m/s)
PAVG = float(E("RF_PAVG", "0.25"))            # EMA weight for the current peg (rest position / held offset)
REST_RESET = float(E("RF_RESET", "0.006"))    # rest EMA resets on a jump larger than this (m)
TOUCH_M = float(E("RF_TOUCH_M", "0.004"))     # pad centre within r + this of the peg axis = held
HELD_DZ = float(E("RF_HELD_DZ", "-0.005"))    # held peg centre z - hand z (pads centred 8 mm under the top)
VERT_TIP = E("RF_VERT_TIP", "0") == "1"       # 1: held peg assumed vertical (tip = centre); 0: tip from the noisy quat
PICKFIX = E("RF_PICKFIX", "0") == "1"
LIFT_Z = float(E("RF_LIFT_Z", "0.075"))
RELINS = E("RF_RELINS", "0") == "1"
EMPTY_ABORT = E("RF_EMPTY_ABORT", "0") == "1"
TMAX = float(E("RF_TMAX", "199"))
FLIM = float(E("RF_FLIM", "0"))               # c3 14:50: compliant insert, cap how far the target runs below a stalled hand
ENDRISE = int(E("RF_ENDRISE", "0"))            # >0: ch4 = episode step counter (replaces the tried mask); in the last
                                               # ENDRISE steps open and rise so every seated peg is "clear" at the end
STEPS_PER_PEG = int(E("RF_STEPS_PER_PEG", "280"))  # 2ccb2d3 conditions.STEPS_PER_PEG (max_steps = this x K)
HELDFIX = E("RF_HELDFIX", "1") == "1"
TGTEST = E("RF_TGTEST", "0") == "1"           # servo the ESTIMATED env target like the author (target ~ hand + last commanded
TGT_LAMBDA = float(E("RF_TGT_LAMBDA", "1.0"))  # move x lambda, clamped to +-TGT_CLAMP of the hand = refport TCLAMP), gain 1
TGT_CLAMP = float(E("RF_TGT_CLAMP", "0.0005"))          # held-start rule = author's (pz > 40 mm, not lying); 0 = the old buggy test
HELD = E("RF_HELD", "0") == "1"               # held-out shapes (hex / dflat, reserved split only; c3 RP_HELD): kept out of the
                                               # board fit, slot = nearer same-size round/square row, hex yaw mod 60 deg, dflat pick yaw = hand
CALRG = E("RF_CALRG", "0") == "1"             # c3 CALRG: first pick with no in-hole anchor: after the close, open, rise, turn the
                                               # hand +90 deg IN PLACE, regrip: the second grip's full 2D reading is the whole delta
DELTA = E("RF_DELTA", "0") == "1"
DIN = float(E("RF_DIN", "0.007"))
DGUARD = float(E("RF_DGUARD", "0"))           # c3: ignore held readings with |peg_est - t - hand| > this (|delta| <= 5 mm + drift); 0 = off              # in-hole radius on |prel_xy| (prel carries delta <= 5 mm + noise; near_hole starts >= 30 mm)             # 7646db6: peg estimates carry one extra shared bias delta (1.5-5 mm). Stateless
                                               # estimate: pegs IN their hole (|prel_xy| < 7 mm, centre z ~ 25 mm, upright, not held)
                                               # give delta = prel_xy; before any seat, the held peg's jaw-axis component
TPADTH = float(E("RF_TPADTH", "0"))           # 4e 0bbb77d: TGTEST pads only when |goal - est. target| > this (0 = author: always).
TDEAD = float(E("RF_TDEAD", "0"))             # and commands nothing below this. The hand + last-move estimate has a pole at -1:
                                               # with the pad always on it settles into a +-0.18 two-step chatter (easy: 0.3x backlash)
NBYAW = E("RF_NBYAW", "0") == "1"             # choose the pick yaw so no standing neighbour lies in the pads' closing sweep
NB_M = float(E("RF_NB_M", "0.001"))            # extra margin (m) on that test
NBSHIFT = E("RF_NBSHIFT", "0") == "1"          # with NBYAW: no free yaw -> neighbour along the pads + off-centre grip (<= NB_SMAX)
NB_SMAX = float(E("RF_NB_SMAX", "0.006"))
GTRANS = float(E("RF_GTRANS", "0"))           # >0: gripper command in I4/I5/RT/IDLE/end-rise (0.65 = 19.8 mm gap, above every
                                               # touch window (dia-2, dia+3) mm so no false "held" reading); effort term only
SPRMAX_NA = float(E("RF_SPRMAX_NA", "0")) / 1000.0   # >0 (mm): spiral radius while no peg is in a hole (have_d false)
SPIRAL_NA = float(E("RF_SPIRAL_NA", "0.35")) / 1000.0  # its pitch (mm)
SPMOVE = float(E("RF_SPMOVE", "0"))                    # >0 (m/s): no lateral search while the hand descends faster than this
SPIN_Z = float(E("RF_SPIN_Z", "0"))                    # >0 (m): no lateral search below this estimated bottom height
P4_EARLY = float(E("RF_P4_EARLY", "0"))          # 4e: >0 (m): P4 hands over to I1 this far below the lift height; I1 keeps rising
I1_DESC = float(E("RF_I1_DESC", "0"))            # 4e: >0 (m): I1 already descends once the tip is this close to the hole in xy
I1_DESC_FLOOR = float(E("RF_I1_DESC_FLOOR", "0.012"))  # 4e: ...but not below bottom = BLOCK_H + this
HOLDG = float(E("RF_HOLDG", "0"))                # 4e: >0 (m): while carrying (P4/I0/I1, +I2 if HOLDG_I2) command gap = dia - HOLDG, not -1
HOLDG_I2 = E("RF_HOLDG_I2", "0") == "1"          #     (effort term: msa = mean sum(a^2); a4 = -1 costs 1.0 per step)
DROPIN = float(E("RF_DROPIN", "0"))              # 4e: >0 (m): once the held bottom is this far below the lip and moving down, open (drop it in)
DROP_VZ = float(E("RF_DROP_VZ", "0.004"))          # 4e: ...and the hand descends faster than this (m/s)
BORE_AZ = float(E("RF_BORE_AZ", "0"))            # 4e: <0: descent command once the peg bottom is BORE_DZ below the lip (deep in the bore)
BORE_DZ = float(E("RF_BORE_DZ", "0.004"))
SP_HI = float(E("RF_SP_HI", "0.7"))             # 4e: I2 descent speed while the bottom is above the block (+3 mm)
SP_LO = float(E("RF_SP_LO", "0.35"))            # 4e: I2 descent speed below that, outside the lip band
BAND_AZ = float(E("RF_BAND_AZ", "-0.33"))       # 4e: I2 descent command inside the lip band (BANDCNT)
S_P2 = float(E("RF_S_P2", "0.4"))               # 4e: P2 (descend to grasp) servo speed, fraction of DXYZ/DYAW per step
S_P4 = float(E("RF_S_P4", "0.5"))               # 4e: P4 lift (and JUP) servo speed
P3_N = float(E("RF_P3_N", "10.5"))              # 4e: P3 close steps (cnt > P3_N)
I3_N = float(E("RF_I3_N", "6.5"))               # 4e: I3 open steps (cnt > I3_N)
HELDTOUCH = E("RF_HELDTOUCH", "0") == "1"      # held start / held-high bonus also need the jaws on the peg (touch)
SQUAT = E("RF_SQUAT", "0") == "1"              # c3: a peg that settled IN another present peg's hole at spawn (12.5% of episodes; a
                                               # 16 mm in the 12 mm hole is wedged) blocks that peg: move a squatter that fits first,
                                               # the blocked peg last (no 201-step futile I2 on the squatter's top, no fall)
SQ_R = float(E("RF_SQ_R", "0.007"))            # squatter: estimated centre within this of the other peg's hole (delta-uncorrected <= 5 mm)
SQ_DZ = float(E("RF_SQ_DZ", "0.006"))          # ... and centre z within this of 25 mm (standing on the board / in the hole)
SQ_PREF = float(E("RF_SQ_PREF", "0.1"))        # cost bonus for a squatter that fits its foreign hole (m); < CW_HELD so a held peg wins
SQ_FIT = float(E("RF_SQ_FIT", "0.001"))        # "fits" = its width (a square's diagonal in a round hole) + this < the hole's size
SQ_BLOCK = float(E("RF_SQ_BLOCK", "1.0"))      # cost penalty for a blocked peg (m): still tried when nothing else is open
RELPOS = E("RF_RELPOS", "0") == "1"            # c3 (unarbos PR #9: every slot estimate has its OWN bias): the obs frame is centred on the TRUE
                                               # board centre, so fit only the board yaw (about the origin), take holes from the layout
                                               # and peg positions as hole_i + prel_i (prel = peg - its own slot estimate: the slot bias
                                               # cancels, only the shared delta stays). The global fit's tx left bias_i - mean in px
RELROT = E("RF_RELROT", "0") == "1"            # with RELPOS: board yaw from a rotation-only fit (PR #9 only); default keeps the Procrustes yaw
PYAW2 = E("RF_PYAW2", "0") == "1"              # c3: the peg yaw from its quaternion is 2*atan2(qz, qw); the head computes atan2(qz, qw) =
                                               # HALF the yaw, so square picks aimed at yaw/2 (38% corner grips), ybias from seated
                                               # squares (a square at +-90 deg gave garbage) and SQTWIST's sign used half angles
G4 = E("RF_G4", "0") == "1"                    # c3: gripper command on a 1/16 grid (rounded UP: the jaws never open less); its 3 low 1/128
                                               # bits become payload (b, 0..7). Codec cost alone = the G4-only arm
BETAMEM = E("RF_BETAMEM", "0") == "1"          # c3: pose_bias_yaw memory in b (0 = unknown, 1..7 = (b - 4) x BM_STEP): read while a square
                                               # is held FLAT (gap ~ side) or the grasped start is in its spawn grip (hand on its yaw),
                                               # used for square picks before any square is seated (38% of square picks were corner-
                                               # gripped by beta: the pads never square a square)
BM_STEP = float(E("RF_BM_STEP", "0.08"))
BM_HYST = float(E("RF_BM_HYST", "0.75"))       # after the first reading, move one level only when a reading is this many levels away
BM_T0 = float(E("RF_BM_T0", "60"))              # grasped-start readings only in the first steps (before any pick could finish)
BM_FLAT = float(E("RF_BM_FLAT", "0.0003"))      # flat grip: gap within (side - 1 mm, side + this)
BM_SQTW = E("RF_BM_SQTW", "1") == "1"          # 1: SQTWIST treats a remembered beta as an anchor; 0: only a seated square does
BM_PY = E("RF_BM_PY", "1") == "1"              # 1: the held-peg yaw (touch false) also uses the remembered beta
SQTWIST = E("RF_SQTWIST", "0") == "1"          # c3: a square held off-square (corner grip, 14 deg median before any square anchor) has
                                               # gap = side (cos t + sin t): |t| from the obs gap (exact to 0.2 deg); sign from the obs
                                               # peg-minus-hand yaw (= t + pose_bias_yaw): minus ybias once a square is seated, else
                                               # the root whose bias lies in [SQTW_LO, SQTW_HI]; the insert yaw then aims the PEG
SQTW_G = float(E("RF_SQTW_G", "0.0003"))       # gap excess over the side (m) below which the grip counts as square
SQTW_LO = float(E("RF_SQTW_LO", "0.085"))      # pose_bias_yaw range (rad): hard tier 0.12-0.28 (+-2 deg margin)
SQTW_HI = float(E("RF_SQTW_HI", "0.314"))
SQTW_AMB = E("RF_SQTW_AMB", "1") == "1"        # both/neither root in range: 1 = take the obs sign, 0 = no correction
CALDIR = E("RF_CALDIR", "0") == "1"            # c3: CAL turns toward yaw 0 (-90 deg when the hand yaw > 0). The env clips its yaw target
                                               # at +-3.1 rad, and every CAL turned +90: after one CAL the next ones were cut short (64 deg)
                                               # or lost (hand parked at 177.6 deg), so the regrip left the pick's pad-axis error (= -delta)
CALWAIT = int(E("RF_CALWAIT", "0"))            # c3: extra steps between the CAL turn and the regrip close (the yaw lags ~14 deg there)
HELDGAP = E("RF_HELDGAP", "0") == "1"          # held start / held-high need jaws closed to the peg width and the hand within 15 mm
ZSNAP_D = float(E("RF_ZSNAP_D", "0"))          # ZSNAP only for pegs farther than this from their hole (stuck-in-hole pegs keep 25)
ZSNAP = E("RF_ZSNAP", "0") == "1"              # pick height of a resting peg: centre 45 mm if est. > 35 mm (on the block), else 25
LEAN_Z = float(E("RF_LEAN_Z", "0"))            # >0: pegs away from their hole with estimated centre z below this are not picked
FGX = float(E("RF_FGX", "0"))                  # force-guided search gains (m per obs force unit) along the pads (x) and the
FGY = float(E("RF_FGY", "0"))                  # jaw axis (y), hand frame; active in the lip band when |F_lat| > FTH (obs units)
FTH = float(E("RF_FTH", "0.03"))
LIFT_HI = float(E("RF_LIFT_HI", "0"))          # >0: P4 lifts to this when a tall upright peg is within LIFT_PATH_R of the path
LIFT_PATH_R = float(E("RF_LIFT_PATH_R", "0.025"))
LIFT_TALL = float(E("RF_LIFT_TALL", "0.06"))     # 4e: obstacle top (m) above which LIFT_HI is used (0.045 also counts board pegs, tops 50 mm)
CW_SLOT = float(E("RF_CW_SLOT", "0.35"))       # peg choice cost = |peg - hand| - CW_HELD (held & high) + CW_SLOT * |peg - hole|
CW_HELD = float(E("RF_CW_HELD", "0.20"))
CW_SQ = float(E("RF_CW_SQ", "0"))              # 4e: extra pick cost (m) for square pegs (later = shortest-job-first for the time term)
SPIRAL = float(E("RF_SPIRAL", "0")) / 1000.0    # >0: band search = Archimedean spiral with this pitch (mm; c3 RP_SPIRAL 0.5)
SPARC = float(E("RF_SPARC", "0.4")) / 1000.0     # arc step per band step (mm)
SPR0 = float(E("RF_SPR0", "0.3")) / 1000.0       # start radius (mm)
SPRMAX = float(E("RF_SPRMAX", "2.5")) / 1000.0   # wind back in past this radius (mm)
RESTZ = E("RF_RESTZ", "0") == "1"             # pick height from the observed peg height instead of a fixed 25 mm centre
I2MAX = float(E("RF_I2MAX", "199"))            # I2 search cap (steps from band entry with BANDCNT); hard time-limit eps spend
                                               # 56% of steps on attempts that never seat, mostly 85-180 steps of I2 search
PREOPEN = float(E("RF_PREOPEN", "0"))
PREOPEN_ALL = E("RF_PREOPEN_ALL", "0") == "1"  # apply PREOPEN from the first pick (c3: dia + 14 mm always, +0.0125 in refport)         # >0: jaw opening dia + this (m) on approach/release once a peg is in a hole
FASTREL = int(E("RF_FASTREL", "0"))           # after a seat: leave I4/I5 (1: and RT) as soon as the obs seated flag is set (hand > FASTREL_Z)
FASTREL_Z = float(E("RF_FASTREL_Z", "0.05"))
FASTREL2 = E("RF_FASTREL2", "0") == "1"        # c3: leave a seat at once: I3 ends when the jaws are clear of the peg (gap > width + FR2_G),
FR2_G = float(E("RF_FR2_G", "0.0015"))         # and I4/I5/RT end as soon as the peg looks in its hole and the pads are FR2_Z above its top,
FR2_Z = float(E("RF_FR2_Z", "0.006"))          # without waiting for the env's seated flag (5 clear steps while still rising to 80 mm)
P1E = E("RF_P1E", "0") == "1"                  # c3/Codex: P1 hands over to P2 once ROUGHLY above an anchored, upright floor peg
P1E_XY = float(E("RF_P1E_XY", "0.002"))        # (hand within this in x and y, P1E_Z of top + 10 mm, P1E_YAW of the pick yaw): P2 keeps
P1E_Z = float(E("RF_P1E_Z", "0.0015"))         # correcting xy/yaw on the way down and keeps its exact arrival gate. Not for PARK pegs,
P1E_YAW = float(E("RF_P1E_YAW", "0.06"))       # pegs estimated on a collar/block (> 30 mm) or (P1E_NB) a grasp shifted off a neighbour
P1E_NB = E("RF_P1E_NB", "1") == "1"
FR2_D = float(E("RF_FR2_D", "0.006"))          # "in its hole": estimate within this of the hole, centre z ~ 25 mm, upright
TGTV = float(E("RF_TGTV", "0"))              # >0: TGTEST lead = clip(TGTV x hand velocity, +-TGT_CLAMP) instead of the last move
BANDCNT = E("RF_BANDCNT", "0") == "1"        # I2 lip spiral counts steps IN the band like refport's `stuck` (cnt from I2 entry
                                               # made it ~1.3 mm wide at first contact); band az -0.33 marks "previous step in band"
HMEAN = float(E("RF_HMEAN", "0"))
HSTART_P4 = E("RF_HSTART_P4", "0") == "1"      # start the held mean after the grip closed (P4), EMA during P3             # c3 14:30: held offset = running mean, weight max(1/k, 1/HMEAN), k = steps held; 0 = EMA              # timeout of the goto phases P1/P2/P4/I0 (author 200; whole-peg budget ~280)  # c3 14:15: grip closed > 3 steps and gap < width - 1.5 mm -> open, rise, give up           # insert on the bias-free tip-to-hole vector (prel), smoothed as q = tip_rel - hand       # P4 lift height (m); c3 14:00: >= 0.095 keeps a held peg above standing tops         # square pick: hand parallel to the peg faces (author/refport: counter-rotate)
RF_JAM = int(E("RF_JAM", "0"))  # +30: c3 retry semantics; -30: transport-only control
if RF_JAM:
    assert abs(RF_JAM) == 30 and RELINS and ENDRISE > 0
    assert 0 < HMEAN <= 20 and 0 < FLIM <= 0.33 and STEPS_PER_PEG <= 280
    NPH = 16
JUP, JALIGN = 14, 15
PARK = E("RF_PARK", "0") == "1"                  # df: a lying ROUND peg creeps (MuJoCo cylinder contact) and ~16% roll off the board,
                                                 # which ends the episode (part_lost). Park it: pick it level, carry it high, drop it
                                                 # centred in the gap between two neighbouring rings of a row, axis across the row,
                                                 # so the two rings block the roll both ways (privileged: 7 of 9 losses saved)
PARK_PREF = float(E("RF_PARK_PREF", "0.15"))     # park cost = PARK_EDGE_W * (distance to the board edge) - PARK_PREF (< CW_HELD:
PARK_EDGE_W = float(E("RF_PARK_EDGE_W", "0.5"))  # a held start still goes first; any upright peg after)
PARK_FIT = float(E("RF_PARK_FIT", "0.008"))      # a gap fits if its free width - dia >= this
PARK_APPZ = float(E("RF_PARK_APPZ", "0.058"))    # P1 hand z above a lying peg (pads above standing tops)
PARK_LIFT = float(E("RF_PARK_LIFT", "0.070"))    # P4 / carry hand z (peg bottom ~ 74 mm)
PARK_DROPZ = float(E("RF_PARK_DROPZ", "0.0175"))  # release hand z: pad bottom 21.5 mm, above the ring tops
PARK_DROPZ_HI = float(E("RF_PARK_DROPZ_HI", "0.032"))  # ... palm bottom 53 mm when an upright peg stands under the palm
PARK_S2 = float(E("RF_PARK_S2", "1.0"))          # P2 descent speed for a lying peg
PARK_TXY = float(E("RF_PARK_TXY", "0.0025"))    # lying-peg P1/P2 arrival: xy (m) and yaw (rad) tolerances
PARK_TYAW = float(E("RF_PARK_TYAW", "0.08"))
PARK_OFF = float(E("RF_PARK_OFF", "0.012"))      # off-centre grasp options (m along the peg) when the centre's footprint is blocked
PARK_ON_X = float(E("RF_PARK_ON_X", "0.010"))    # "parked": within this of a gap centre across the row (lying spawns are >= 25 mm away)
PARK_ON_Y = float(E("RF_PARK_ON_Y", "0.016"))    # ... and along it
PARK_CREF = E("RF_PARK_CREF", "1") == "1"
PARK_NOHELD = E("RF_PARK_NOHELD", "1") == "1"
PARK_OCC2 = E("RF_PARK_OCC2", "1") == "1"
KD_LOW, KD_OPEN, KD_RISE = (int(E("RF_PARK_KD_" + k, d)) for k, d in (("LOW", "16"), ("OPEN", "6"), ("RISE", "12")))
KC, KD = 14, 15                                  # park phases: carry over the gap / lower, open, rise (counter)
if PARK:
    assert not RF_JAM
    NPH = 16
PARK_GAPS = ((-0.05, 0.0308), (0.0, 0.0268), (0.05, 0.0228))   # board-local gap x between neighbouring rings, free width at the
                                                               # row line for clearance 0.6 mm (rings 4/8, 8/12, 12/16 mm)
JBASE, JQ, JBUCKET, JPAIR, JMETA = 2100000, 214, 14, 5595, 2816

SLOT_LX = (-0.075, -0.025, 0.025, 0.075)
SLOT_LY_ROUND, SLOT_LY_SQUARE = 0.045, -0.045


PICK_FCAP = E("RF_PICK_FCAP", "0") == "1"


def _wrap(F, x):
    return x - 2 * math.pi * F.round(x / (2 * math.pi))


def _wrap90(F, x):
    return x - (math.pi / 2) * F.round(x / (math.pi / 2))


def _atan2(F, y, x):
    return 2 * F.atan(y / (F.sqrt(x * x + y * y) + x + 1e-12))


def _onehot_pick(F, sel, vals):
    out = vals[0] * 0.0
    for s, v in zip(sel, vals):
        out = out + F.where(s, v, 0.0)
    return out


def _decode(F, a):
    """Echo channel -> (command part, 10-bit code); code -1 at the all-zero reset echo."""
    q = F.floor(a / G) * G
    code = F.floor((a - q) / STEP)
    return q, F.clip(code, 0.0, NLEV - 1.0)


def _encode(F, cmd, code):
    q = F.round(F.clip(cmd, -1.0 + G, 1.0 - 2 * G) / G) * G
    return q + (code + 0.5) * STEP


def _encode_g4(F, cmd, code, b):
    j = F.clip(-F.floor(-((cmd + 1.0) / (8 * G)) + 1e-6), 0.0, 31.0)   # ceil: the 1/16 grid point at or above the command
    q = -1.0 + (8.0 * j + b) * G
    return q + (code + 0.5) * STEP


def _jam_qdecode(F, q):
    mid = (q - 16.0) * (0.032 / 181.0) - 0.016
    return F.where(q < 16.0, -0.04 + q * 0.0015,
                   F.where(q > 197.0, 0.016 + (q - 197.0) * 0.0015, mid))


def _jam_qencode(F, x):
    q = F.where(x < -0.016, (x + 0.04) / 0.0015,
                F.where(x > 0.016, 197.0 + (x - 0.016) / 0.0015,
                        16.0 + (x + 0.016) * (181.0 / 0.032)))
    return F.clip(F.round(q), 0.0, JQ - 1.0)


def _jam_decode(F, pa):
    z = pa[0] * 0.0
    ds = [_decode(F, a)[1] for a in pa]
    marked = (((pa[4] > -0.985) & (pa[4] < -0.979)) |
              ((pa[4] > 0.976) & (pa[4] < 0.982)))
    anchor = F.where(pa[4] < 0, -0.984375, 0.9765625)
    fine = F.floor((pa[4] - anchor) / (G / 6000.0))
    overflow = F.floor(fine / 1700.0)
    clock = fine - 1700.0 * overflow
    m2, m3 = F.floor(ds[2] / JQ), F.floor(ds[3] / JQ)
    q2, q3 = ds[2] - JQ * m2, ds[3] - JQ * m3
    carry = m2 + JBUCKET * m3 + (JBUCKET * JBUCKET) * overflow
    d0 = F.where(marked, ds[0], 0.0)
    w1 = F.where(marked, ds[1] + NLEV * carry, 0.0)
    i2 = w1 >= 700.0
    # Subtract JBASE in base3000, then divide by2816 without forming a >2^24 word.
    borrow = z
    a0 = F.where(i2, d0, 0.0)
    a1 = F.where(i2, w1 - 700.0, 0.0)
    ah = F.floor(a1 / JMETA)
    ar = a1 - JMETA * ah
    small = a0 + NLEV * ar
    cl = F.floor(small / JMETA)
    meta = small - JMETA * cl
    combo = cl + NLEV * ah
    stuck = F.floor(combo / JPAIR)
    pair = combo - JPAIR * stuck
    cnt = F.where(pair < 435.0, F.floor((F.sqrt(F.maximum(8.0 * pair + 1.0, 0.0)) - 1.0) / 2.0),
                  F.floor((pair + 435.0) / 30.0))
    age = F.where(pair < 435.0, pair - cnt * (cnt + 1.0) / 2.0, pair + 435.0 - 30.0 * cnt)
    pg = meta - 8.0 * F.floor(meta / 8.0)
    mm = F.floor(meta / 8.0)
    retry = mm - 4.0 * F.floor(mm / 4.0)
    mm = F.floor(mm / 4.0)
    best = mm - 44.0 * F.floor(mm / 44.0)
    stallbit = F.floor(mm / 44.0)
    low = d0 + NLEV * F.minimum(w1, 699.0)
    lpg = low - 8.0 * F.floor(low / 8.0)
    lm = F.floor(low / 8.0)
    lr = lm - 4.0 * F.floor(lm / 4.0)
    lm = F.floor(lm / 4.0)
    lp = lm - 2.0 * F.floor(lm / 2.0)
    lm = F.floor(lm / 2.0)
    lc = lm - 64.0 * F.floor(lm / 64.0)
    goal = F.floor(lm / 64.0)
    return dict(marked=marked, i2=i2, phase=F.where(i2, 7.0, 14.0 + lp),
                peg=F.where(i2, pg, lpg), retry=F.where(i2, retry, lr),
                cnt=F.where(i2, cnt, lc), best=(best + 1.0) * 0.0005,
                age=age, stuck=stuck, stallbit=stallbit, goal=goal * 0.0005 - 0.003,
                clock=clock, x=_jam_qdecode(F, q2), y=_jam_qdecode(F, q3))


def _jam_wire(F, action, pg, retry, best, stallbit, cnt, age, stuck, ni2, align, goal, clock, x, y):
    cnt = F.clip(cnt, 0.0, F.where(ni2, 200.0, 63.0))
    age = F.minimum(F.clip(age, 0.0, 29.0), cnt)
    pair = F.where(cnt < 29.0, cnt * (cnt + 1.0) / 2.0 + age, 30.0 * cnt - 435.0 + age)
    combo = pair + JPAIR * F.clip(stuck, 0.0, 200.0)
    ch = F.floor(combo / NLEV)
    cl = combo - NLEV * ch
    meta = pg + 8.0 * (retry + 4.0 * (best + 44.0 * stallbit))
    word = JBASE + meta + JMETA * cl  # <=10,547,999, exactly representable float32
    wh = F.floor(word / NLEV)
    d0 = word - NLEV * wh
    w1 = wh + JMETA * ch
    gcode = F.clip(F.round((goal + 0.003) / 0.0005), 0.0, 511.0)
    short = pg + 8.0 * (retry + 4.0 * (align + 2.0 * (cnt + 64.0 * gcode)))
    sh = F.floor(short / NLEV)
    d0, w1 = F.where(ni2, d0, short - NLEV * sh), F.where(ni2, w1, sh)
    carry = F.floor(w1 / NLEV)
    d1 = w1 - NLEV * carry
    overflow = F.floor(carry / (JBUCKET * JBUCKET))
    rem = carry - (JBUCKET * JBUCKET) * overflow
    m3 = F.floor(rem / JBUCKET)
    m2 = rem - JBUCKET * m3
    q2, q3 = _jam_qencode(F, x), _jam_qencode(F, y)
    anchor = F.where(action[4] < 0.0, -0.984375, 0.9765625)
    return [_encode(F, action[0], d0), _encode(F, action[1], d1),
            _encode(F, action[2], q2 + JQ * m2), _encode(F, action[3], q3 + JQ * m3),
            anchor + (1700.0 * overflow + F.minimum(clock, 1680.0) + 0.5) * (G / 6000.0)]


def policy(F, obs):
    if CLOSED_CAL:
        assert CALRG and CALDIR and DELTA and RELINS and HMEAN > 0 and not RF_JAM
    c = lambda i: F.col(obs, i)  # noqa: E731
    zero = c(0) * 0.0
    one = zero + 1.0
    hx, hy, hz = c(0) * 0.1, c(1) * 0.1, c(2) * 0.1
    hyaw = _atan2(F, c(3), c(4))
    vx, vy, vz = c(5), c(6), c(7)
    gap = c(8) * 0.02
    speed = F.sqrt(vx * vx + vy * vy + vz * vz)

    # ---- memory (echo)
    pa = [c(ACT0 + i) for i in range(5)]
    psum = F.abs(pa[0]) + F.abs(pa[1]) + F.abs(pa[2]) + F.abs(pa[3]) + F.abs(pa[4])
    reset = psum < 1e-9
    codes = [_decode(F, x)[1] for x in pa]
    phase = F.where(reset, float(NEXT), codes[0] - 16 * F.floor(codes[0] / 16))
    pg_ = F.floor(codes[0] / 16)
    peg_m = F.where(reset, 0.0, pg_ - 8 * F.floor(pg_ / 8))
    kheld = F.where(reset, 0.0, F.floor(codes[0] / 128))              # steps held since this grasp (capped 22)
    cnt = F.where(reset, 0.0, codes[1] - 256 * F.floor(codes[1] / 256))
    STALL_Q = (FLIM / 10.0) if FLIM > 0 else 0.05                      # FLIM memory: in [-FLIM, 0], 11 levels
    stall = F.where(reset, 0.0, -STALL_Q * F.floor(codes[1] / 256))
    if CLOSED_CAL:
        cc_mode = ~reset & (kheld < 0.5) & (phase > P4 - 0.5) & (phase < I2 + 0.5)
        cc_is_i2 = F.abs(phase - I2) < 0.5
        cc_h = F.floor(codes[1] / 256.0)
        cc_dir_bit = F.where(cc_is_i2, F.floor(cc_h / 2.0), cc_h)
        cc_decode_neg = cc_dir_bit > 0.5
        cc_stallbit = cc_h - 2.0 * F.floor(cc_h / 2.0)
        stall = F.where(cc_mode & cc_is_i2, -FLIM * cc_stallbit, stall)
    if BETAMEM:
        assert G4 and ENDRISE > 0
        m4 = F.round((F.floor(pa[4] / G) * G + 1.0) / G)
        b_prev = F.where(reset, 0.0, m4 - 8.0 * F.floor(m4 / 8.0))
    if ENDRISE > 0:
        tstep = F.where(reset, 0.0, codes[4])                          # control steps taken so far (this one excluded)
        mask = zero
    else:
        mask = F.where(reset, 0.0, codes[4])
    if RF_JAM:
        jd = _jam_decode(F, pa)
        marked = jd["marked"] & ~reset
        phase = F.where(marked, jd["phase"], phase)
        peg_m = F.where(marked, jd["peg"], peg_m)
        cnt = F.where(marked, jd["cnt"], cnt)
        kheld = F.where(marked, 19.0, kheld)
        stall = F.where(marked, F.where(jd["i2"], -FLIM * jd["stallbit"], 0.0), stall)
        tstep = F.where(marked, jd["clock"], tstep)
        codes[2] = F.where(marked, (jd["x"] + 0.04) * ((NLEV - 1.0) / 0.08), codes[2])
        codes[3] = F.where(marked, (jd["y"] + 0.04) * ((NLEV - 1.0) / 0.08), codes[3])
        jretry = F.where(marked, jd["retry"], 0.0)
    ph = lambda k: F.abs(phase - k) < 0.5  # noqa: E731

    # ---- per-row quantities and the board fit
    rows = []
    for k in range(8):
        b = PEG0 + PEGF * k
        pres = c(b) > 0.5
        sq = c(b + 9) > 0.5
        dia = c(b + 12) * 0.02
        dia_mm = dia * 1000.0
        lx = F.where(dia_mm < 6.0, SLOT_LX[0], F.where(dia_mm < 10.0, SLOT_LX[1], F.where(dia_mm < 14.0, SLOT_LX[2], SLOT_LX[3])))
        ly = F.where(sq, SLOT_LY_SQUARE, SLOT_LY_ROUND)
        hexs, dfl = c(b + 10) > 0.5, c(b + 11) > 0.5
        rows.append(dict(pres=pres, sq=sq, hexs=hexs, dfl=dfl, ho=hexs | dfl, dia=dia, lx=lx, ly=ly, ox=c(b + 13) * 0.1, oy=c(b + 14) * 0.1,
                         px=(c(b + 13) + c(b + 1)) * 0.1, py=(c(b + 14) + c(b + 2)) * 0.1, pz=c(b + 3) * 0.1 + BLOCK_H,
                         qw=c(b + 4), qx=c(b + 5), qy=c(b + 6), qz=c(b + 7), seated=c(b + 16) > 0.5,
                         rx=c(b + 1) * 0.1, ry=c(b + 2) * 0.1))
    w = [F.where(r["pres"] & ~r["ho"], 1.0, 0.0) for r in rows] if HELD else [F.where(r["pres"], 1.0, 0.0) for r in rows]
    n = sum(w) + 1e-9
    msx = sum(wi * r["lx"] for wi, r in zip(w, rows)) / n; msy = sum(wi * r["ly"] for wi, r in zip(w, rows)) / n
    mox = sum(wi * r["ox"] for wi, r in zip(w, rows)) / n; moy = sum(wi * r["oy"] for wi, r in zip(w, rows)) / n
    cr = sum(wi * ((r["lx"] - msx) * (r["oy"] - moy) - (r["ly"] - msy) * (r["ox"] - mox)) for wi, r in zip(w, rows))
    dt = sum(wi * ((r["lx"] - msx) * (r["ox"] - mox) + (r["ly"] - msy) * (r["oy"] - moy)) for wi, r in zip(w, rows))
    byaw = _atan2(F, cr, dt)
    if RELPOS and RELROT:         # rotation-only fit about the true centre: best under PR #9's independent slot biases, but a
        cr0 = sum(wi * (r["lx"] * r["oy"] - r["ly"] * r["ox"]) for wi, r in zip(w, rows))      # SHARED bias (today's referee)
        dt0 = sum(wi * (r["lx"] * r["ox"] + r["ly"] * r["oy"]) for wi, r in zip(w, rows))      # turns into a wrong yaw (-0.07)
        byaw = _atan2(F, cr0, dt0)
    cb, sb = F.cos(byaw), F.sin(byaw)
    tx, ty = mox - (cb * msx - sb * msy), moy - (sb * msx + cb * msy)
    if RELPOS:
        tx, ty = zero, zero
    # yaw bias of peg quats: seated squares sit at the board yaw (mod 90)
    ynum, yden = zero, zero + 1e-9
    for r in rows:
        r["pyaw"] = (4 if PYAW2 else 2) * F.atan(r["qz"] / (F.sqrt(r["qw"] * r["qw"] + r["qz"] * r["qz"]) + r["qw"] + 1e-12))
        on = r["pres"] & r["sq"] & r["seated"]
        ynum = ynum + F.where(on, _wrap90(F, r["pyaw"] - byaw), 0.0); yden = yden + F.where(on, 1.0, 0.0)
    ybias = ynum / yden
    for r in rows:
        if not RELPOS:
            r["px"], r["py"] = r["px"] - tx, r["py"] - ty                 # de-biased peg estimate
        r["hx"], r["hy"] = cb * r["lx"] - sb * r["ly"], sb * r["lx"] + cb * r["ly"]   # hole centre from the fitted layout
        if HELD:                  # held-out shape: its hole is the round or the square row of the same size, whichever is nearer
            rawx, rawy = r["ox"] - tx, r["oy"] - ty
            ax_, ay_ = cb * r["lx"] - sb * SLOT_LY_ROUND, sb * r["lx"] + cb * SLOT_LY_ROUND
            bx_, by_ = cb * r["lx"] - sb * SLOT_LY_SQUARE, sb * r["lx"] + cb * SLOT_LY_SQUARE
            da = (ax_ - rawx) * (ax_ - rawx) + (ay_ - rawy) * (ay_ - rawy)
            db = (bx_ - rawx) * (bx_ - rawx) + (by_ - rawy) * (by_ - rawy)
            r["hx"] = F.where(r["ho"], F.where(da < db, ax_, bx_), r["hx"])
            r["hy"] = F.where(r["ho"], F.where(da < db, ay_, by_), r["hy"])
        if RELPOS:                # peg = its own hole (fitted layout) + prel (peg - its own slot estimate; the slot bias cancels)
            r["px"], r["py"] = r["hx"] + r["rx"], r["hy"] + r["ry"]
        r["ax"] = 2 * (r["qx"] * r["qz"] + r["qw"] * r["qy"]); r["ay"] = 2 * (r["qy"] * r["qz"] - r["qw"] * r["qx"])
        r["az"] = 1 - 2 * (r["qx"] * r["qx"] + r["qy"] * r["qy"])
        r["lying"] = F.abs(r["az"]) < 0.5
        dx, dy, dz = hx - r["px"], hy - r["py"], hz + PAD_CENTRE_Z - r["pz"]
        sgn = F.where(r["az"] >= 0, 1.0, -1.0)
        ux, uy, uz = r["ax"] * sgn, r["ay"] * sgn, r["az"] * sgn
        t = F.clip(dx * ux + dy * uy + dz * uz, -PEG_LEN / 2, PEG_LEN / 2)
        ex, ey, ez = dx - t * ux, dy - t * uy, dz - t * uz
        near = F.sqrt(ex * ex + ey * ey + ez * ez) < r["dia"] / 2 + TOUCH_M
        r["touch"] = near & (gap > r["dia"] - 0.002) & (gap < r["dia"] + 0.003)
        r["ux"], r["uy"] = ux, uy
        ddx, ddy = r["px"] - hx, r["py"] - hy
        r["dhand"] = F.sqrt(ddx * ddx + ddy * ddy)
        dsx, dsy = r["px"] - r["hx"], r["py"] - r["hy"]
        r["dslot"] = F.sqrt(dsx * dsx + dsy * dsy)

    # ---- shared peg-estimate bias delta (7646db6)
    if DELTA:
        dn, dsx, dsy = zero, zero, zero
        for r in rows:
            inh = r["pres"] & (r["az"] > 0.95) & (F.abs(r["pz"] - 0.025) < 0.004) & ~r["touch"] & \
                (F.sqrt(r["rx"] * r["rx"] + r["ry"] * r["ry"]) < DIN)
            dn = dn + F.where(inh, 1.0, 0.0); dsx = dsx + F.where(inh, r["rx"], 0.0); dsy = dsy + F.where(inh, r["ry"], 0.0)
        have_d = dn > 0.5
        ddx, ddy = F.where(have_d, dsx / F.maximum(dn, 1.0), 0.0), F.where(have_d, dsy / F.maximum(dn, 1.0), 0.0)
        # before any seat: the held peg. peg_est - hand = delta + (true peg - hand). Across the jaws the peg is centred,
        # and a grasped START has the hand exactly on the peg in both axes (c3), so the full 2D reading is delta there;
        # for a picked peg its pad-axis component is ~0 (the pick itself was off by delta), i.e. no worse than jaw-only.
        djx, djy = zero, zero
        for r in rows:
            held_r = r["touch"] & (gap < r["dia"] + 0.0015)
            if DGUARD > 0:
                ex_, ey_ = r["px"] - hx, r["py"] - hy
                held_r = held_r & (F.sqrt(ex_ * ex_ + ey_ * ey_) < DGUARD)
            djx = djx + F.where(held_r, r["px"] - hx, 0.0); djy = djy + F.where(held_r, r["py"] - hy, 0.0)
        ddx = F.where(have_d, ddx, djx); ddy = F.where(have_d, ddy, djy)
        for r in rows:            # corrected estimates: absolute peg pose and the peg-vs-slot vector
            r["px"], r["py"] = r["px"] - ddx, r["py"] - ddy
            r["rx"], r["ry"] = r["rx"] - ddx, r["ry"] - ddy
            ddx2, ddy2 = r["px"] - hx, r["py"] - hy
            r["dhand"] = F.sqrt(ddx2 * ddx2 + ddy2 * ddy2)
            dsx2, dsy2 = r["px"] - r["hx"], r["py"] - r["hy"]
            r["dslot"] = F.sqrt(dsx2 * dsx2 + dsy2 * dsy2)

    if PARK:
        # board-local coordinates of the (de-biased) estimates; the frame is board-centred, rotated by byaw
        for k, r in enumerate(rows):
            r["bx"], r["by"] = cb * r["px"] + sb * r["py"], -sb * r["px"] + cb * r["py"]
            nl = F.sqrt(r["ax"] * r["ax"] + r["ay"] * r["ay"]) + 1e-9
            r["lux"], r["luy"] = r["ax"] / nl, r["ay"] / nl                   # lying axis in xy (sign fixed by the body, not by az)
            onb = zero > 1.0                                                  # parked: on a gap line, axis across the row
            for gy in (SLOT_LY_ROUND, SLOT_LY_SQUARE):
                for gx, _w in PARK_GAPS:
                    onb = onb | ((F.abs(r["bx"] - gx) < PARK_ON_X) & (F.abs(r["by"] - gy) < PARK_ON_Y))
            r["parked"] = r["lying"] & onb & (F.abs(cb * r["lux"] + sb * r["luy"]) < 0.4)
            r["pk_cand"] = r["pres"] & r["lying"] & ~r["sq"] & ~r["ho"] & ~r["seated"] & (r["pz"] < 0.014) & ~r["parked"]
            r["edge"] = F.minimum(0.22 - F.abs(r["bx"]), 0.18 - F.abs(r["by"]))
        # ONE focus row gets the costly tests: at NEXT the candidate nearest the board edge (not the one just tried), else the
        # current peg (the same row through P1..KD)
        at_next_ = ph(NEXT)
        fbest, fsel = zero + 1e6, []
        for k, r in enumerate(rows):
            e_ = F.where(r["pk_cand"] & ((F.abs(peg_m - float(k)) > 0.5) | reset), r["edge"], 1e6)
            bt = e_ < fbest
            fsel = [x & ~bt for x in fsel] + [bt]
            fbest = F.where(bt, e_, fbest)
        fsel = [(at_next_ & f_) | (~at_next_ & (F.abs(peg_m - float(k)) < 0.5)) for k, f_ in enumerate(fsel)]
        fg = lambda key: _onehot_pick(F, fsel, [r[key] for r in rows])  # noqa: E731
        f_px, f_py, f_dia, f_lux, f_luy = fg("px"), fg("py"), fg("dia"), fg("lux"), fg("luy")
        f_cand = _onehot_pick(F, fsel, [F.where(r["pk_cand"], 1.0, 0.0) for r in rows]) > 0.5
        # target gap: nearest fitting gap that no other peg occupies
        best_d, gwx, gwy, gok = zero + 1e6, zero, zero, zero > 1.0
        for gy in (SLOT_LY_ROUND, SLOT_LY_SQUARE):
            for gx, wg in PARK_GAPS:
                occ = zero > 1.0
                for j, rj in enumerate(rows):
                    if PARK_OCC2:     # the parked peg's own half-width + the other's + a margin (2542/95 dropped onto a tray peg)
                        occ = occ | (rj["pres"] & ~fsel[j] & (F.abs(rj["bx"] - gx) < f_dia / 2 + rj["dia"] / 2 + 0.006) & (F.abs(rj["by"] - gy) < 0.031 + rj["dia"] / 2))
                    else:
                        occ = occ | (rj["pres"] & ~fsel[j] & (F.abs(rj["bx"] - gx) < 0.010 + rj["dia"] / 2) & (F.abs(rj["by"] - gy) < 0.030 + rj["dia"] / 2))
                if PARK_OCC2:         # clutter blocks (12-25 mm tall) next to the gap: the peg lands on them (2542/95)
                    for jc in range(3):
                        bc = PEG0 + 8 * PEGF + 9 * jc
                        cxw, cyw = c(bc + 1) * 0.1 - tx, c(bc + 2) * 0.1 - ty
                        cbx, cby = cb * cxw + sb * cyw, -sb * cxw + cb * cyw
                        ce = c(bc + 8) * 0.02 * 1.42 + 0.003
                        occ = occ | ((c(bc) > 0.5) & (F.abs(cbx - gx) < f_dia / 2 + ce) & (F.abs(cby - gy) < PEG_LEN / 2 + ce))
                gxw, gyw = cb * gx - sb * gy, sb * gx + cb * gy
                d_ = F.sqrt((f_px - gxw) * (f_px - gxw) + (f_py - gyw) * (f_py - gyw))
                okg = ~occ & (wg - f_dia >= PARK_FIT) & (d_ < best_d)
                best_d, gwx, gwy, gok = F.where(okg, d_, best_d), F.where(okg, gxw, gwx), F.where(okg, gyw, gwy), gok | okg
        tall = zero > 1.0             # an upright peg (in a neighbouring ring or standing) under the palm at the release point
        for j, rj in enumerate(rows):
            dxg, dyg = rj["px"] - gwx, rj["py"] - gwy
            lxg, lyg = cb * dxg + sb * dyg, -sb * dxg + cb * dyg              # board axes: palm +-33 mm across the row (board x)
            tall = tall | (rj["pres"] & ~rj["lying"] & ~fsel[j] & (F.abs(lxg) < 0.034 + rj["dia"] / 2) & (F.abs(lyg) < 0.024 + rj["dia"] / 2))
        f_dz = F.where(tall, PARK_DROPZ_HI, PARK_DROPZ)
        # grasp plan: first (z, offset) whose open pads (+-10 mm along, r + 13 mm across) and palm (+-23 x +-33 mm) are clear
        obst = []                 # (x, y, radius, top) of everything the open pads or the palm could land on
        for (shape_, dia_s, lx_, ly_) in [("round", d_, x_, SLOT_LY_ROUND) for d_, x_ in zip((0.004, 0.008, 0.012, 0.016), SLOT_LX)] + \
                [("square", d_, x_, SLOT_LY_SQUARE) for d_, x_ in zip((0.004, 0.008, 0.012, 0.016), SLOT_LX)]:
            rr_ = (dia_s / 2 + 0.0008 + 0.006 + 0.001) * (1.42 if shape_ == "square" else 1.0)
            obst.append((cb * lx_ - sb * ly_, sb * lx_ + cb * ly_, rr_, 0.0205))
        for j, rj in enumerate(rows):
            rad = F.where(rj["lying"], PEG_LEN / 2 + rj["dia"] / 2, rj["dia"] * 0.75)
            top = F.where(rj["lying"], rj["pz"] + rj["dia"] / 2, rj["pz"] + PEG_LEN / 2)
            obst.append((rj["px"], rj["py"], rad, F.where(rj["pres"] & ~fsel[j], top, -1.0)))
        for j in range(3):
            b_ = PEG0 + 8 * PEGF + 9 * j
            obst.append((c(b_ + 1) * 0.1 - tx, c(b_ + 2) * 0.1 - ty, c(b_ + 8) * 0.02 * 1.42 + 0.002,
                         F.where(c(b_) > 0.5, 2.0 * c(b_ + 3) * 0.1 + 0.002, -1.0)))
        hw_pad = f_dia / 2 + 0.013
        plan_ok, f_off, f_z = zero > 1.0, zero, zero - 0.003
        for zz, need_big in ((-0.003, False), (-0.0004, True)):
            for off in (0.0, PARK_OFF, -PARK_OFF):
                gx_, gy_ = f_px + off * f_lux, f_py + off * f_luy
                clear = (f_dia > 0.0105) if need_big else (zero < 1.0)       # high grasp (palm over the ring tops): 12/16 mm only
                for (ox, oy, rr_, top) in obst:
                    dx_, dy_ = ox - gx_, oy - gy_
                    ax2 = F.abs(f_lux * dx_ + f_luy * dy_)
                    ay2 = F.abs(-f_luy * dx_ + f_lux * dy_)
                    qa, qb = F.maximum(ax2 - 0.010, 0.0), F.maximum(ay2 - hw_pad, 0.0)
                    qc, qd = F.maximum(ax2 - 0.023, 0.0), F.maximum(ay2 - 0.033, 0.0)
                    r2 = rr_ * rr_
                    hit = ((top > zz + 0.004) & (qa * qa + qb * qb < r2)) | ((top > zz + 0.021) & (qc * qc + qd * qd < r2))
                    clear = clear & ~hit
                take = clear & ~plan_ok
                f_off, f_z, plan_ok = F.where(take, off, f_off), F.where(take, zz, f_z), plan_ok | clear
        f_parkable = f_cand & gok & plan_ok
        if PARK_NOHELD:           # a peg held up at the start (centre > 40 mm, the NEXT held-high rule) goes in first: opening the
            held_any = zero > 1.0                   # jaws for a park dropped it (B 2525-2620: grasped+lying episodes net -3.2)
            for r in rows:
                held_any = held_any | (r["pres"] & ~r["seated"] & ~r["lying"] & (r["pz"] > 0.04))
            f_parkable = f_parkable & ~held_any
        for k, r in enumerate(rows):
            r["parkable"] = fsel[k] & f_parkable

    # ---- NEXT: choose a peg (held & high first, then nearest + 0.35 * distance to its hole); lying skipped
    best, sel_new = zero + 1e6, []
    tried = [F.floor(mask / (2 ** k)) - 2 * F.floor(mask / (2 ** (k + 1))) > 0.5 for k in range(8)]
    untried_open = zero
    for k, r in enumerate(rows):
        r["open"] = r["pres"] & ~r["seated"] & ~r["lying"]
        if LEAN_Z > 0:            # knocked-over pegs leaning on neighbours (centre ~17 mm vs 25): every grasp closes on air
            r["open"] = r["open"] & ~((r["pz"] < LEAN_Z) & (r["dslot"] > 0.010))   # (11/11 empty grasps); skip like lying
        untried_open = untried_open + F.where(r["open"] & ~tried[k], 1.0, 0.0)
    clear_mask = untried_open < 0.5                                   # every open peg tried once: try them all again
    if PARK_LAST:
        any_open_up = zero > 1.0
        for r in rows:
            any_open_up = any_open_up | r["open"]
    if SQUAT:
        for r in rows:            # squat: in another present peg's hole and small enough to lift out (a bigger one is wedged)
            r["squat"], r["blocked"] = zero > 1.0, zero > 1.0
        for j, rj in enumerate(rows):
            sj = rj["pres"] & ~rj["seated"] & (rj["az"] > 0.9) & (F.abs(rj["pz"] - 0.025) < SQ_DZ)
            for i, ri in enumerate(rows):
                if i == j:
                    continue
                dx_, dy_ = rj["px"] - ri["hx"], rj["py"] - ri["hy"]
                on = sj & ri["pres"] & ~ri["seated"] & (dx_ * dx_ + dy_ * dy_ < SQ_R * SQ_R)
                width = F.where(rj["sq"] & ~ri["sq"], rj["dia"] * math.sqrt(2.0), rj["dia"])
                rj["squat"] = rj["squat"] | (on & (width + SQ_FIT < ri["dia"]))
                ri["blocked"] = ri["blocked"] | on
    for k, r in enumerate(rows):
        held_high = ((r["pz"] > 0.04) & ~r["lying"]) if HELDFIX else (r["touch"] & (gap < 0.014) & (r["pz"] > 0.04))
        if HELDTOUCH:             # a peg standing ON the block (centre 45 mm) is not in the hand: require the jaws on it
            held_high = held_high & r["touch"]
        if HELDGAP:               # same, but without touch's axis test (delta up to 5 mm breaks it): jaws closed to ~this
            held_high = held_high & (gap < r["dia"] + 0.003) & (r["dhand"] < 0.015)   # peg's width and the hand near it
        cost = r["dhand"] - F.where(held_high, CW_HELD, 0.0) + CW_SLOT * r["dslot"]
        if PARK and DELTA and PARK_CREF:  # before any in-hole anchor, rank from the board centre (= the start pose), not from the
            dcen = F.sqrt(r["px"] * r["px"] + r["py"] * r["py"])               # park's release point: the first pick does the CAL
            cost = F.where(have_d, cost, cost - r["dhand"] + dcen)             # regrip, which fails on small pegs (2863/102)
        if CW_SQ != 0:            # never for a held peg: above CW_HELD it would drop a grasped-start square (c048: -0.07 at 0.25)
            cost = cost + F.where(r["sq"] & ~held_high, CW_SQ, 0.0)
        if SQUAT:
            cost = cost - F.where(r["squat"], SQ_PREF, 0.0) + F.where(r["blocked"], SQ_BLOCK, 0.0)
        if BF_PEN > 0:            # user: two pegs touching -> take the bigger one first (the small one is easier once alone)
            near_big = zero > 1.0
            for j, rj in enumerate(rows):
                if j == k:
                    continue
                dxb, dyb = rj["px"] - r["px"], rj["py"] - r["py"]
                lim = (r["dia"] + rj["dia"]) / 2 + BF_GAP
                near_big = near_big | (rj["pres"] & ~rj["seated"] & ~rj["lying"] & (rj["dia"] > r["dia"] + 0.001)
                                       & (dxb * dxb + dyb * dyb < lim * lim))
            cost = cost + F.where(near_big, BF_PEN, 0.0)
        if TL_PEN > 0:            # user: pick the standing peg before a tilted one
            cost = cost + F.where((F.abs(r["az"]) < TL_AZ) & ~r["lying"], TL_PEN, 0.0)
        if ENDRISE > 0:
            ok = r["open"] & ((F.abs(peg_m - float(k)) > 0.5) | (untried_open < 1.5) | reset)
        else:
            ok = r["open"] & (~tried[k] | clear_mask)
        cost = F.where(ok, cost, 1e6)
        if PARK:
            park_now = r["parkable"] & at_next_
            if PARK_LAST:
                park_now = park_now & ~any_open_up
            cost = F.where(park_now, PARK_EDGE_W * r["edge"] - PARK_PREF, cost)
        better = cost < best
        sel_new = [s & ~better for s in sel_new] + [better]
        best = F.where(better, cost, best)
    any_new = best < 1e5
    k_new = _onehot_pick(F, sel_new, [zero + float(k) for k in range(8)])
    mask_eff = F.where(clear_mask, 0.0, mask)

    # current peg row: from memory, or the new choice when entering from NEXT
    at_next = ph(NEXT)
    kcur = F.where(at_next, k_new, peg_m)
    sel = [F.abs(kcur - k) < 0.5 for k in range(8)]
    g = lambda key: _onehot_pick(F, sel, [r[key] for r in rows])  # noqa: E731
    gb = lambda key: _onehot_pick(F, sel, [F.where(r[key], 1.0, 0.0) for r in rows]) > 0.5  # noqa: E731
    px_raw, py_raw, pz_raw = g("px"), g("py"), g("pz")
    dia, sq, touch, seated = g("dia"), gb("sq"), gb("touch"), gb("seated")
    holx, holy = g("hx"), g("hy")
    ux, uy, pyaw = g("ux"), g("uy"), g("pyaw")
    # author's ScriptedPeg.policy: an upright peg whose centre is >= 40 mm up skips the pick and goes straight to insert
    # (refport start_type "grasped"); the old touch & gap < 14 mm test missed every 16 mm peg and dropped it (4e 15:3x)
    held_start = (pz_raw > 0.04) & ~gb("lying") if HELDFIX else touch & (gap < 0.014) & (pz_raw > 0.04)
    if HELDTOUCH:                 # c038 hard time-limit eps: 18 empty-handed "held" starts on block-standing pegs, re-tried in loops
        held_start = held_start & touch
    if HELDGAP:
        held_start = held_start & (gap < dia + 0.003) & (g("dhand") < 0.015)

    # ---- smoothed current-peg position (PAVG): rest = absolute EMA, held = offset-vs-hand EMA
    held_mode = (phase >= P3 - 0.5) & (phase <= I2 + 0.5)
    if RF_JAM:
        held_mode = held_mode | ph(JUP) | ph(JALIGN)
    ema_x_rest = (codes[2] / (NLEV - 1)) * 0.5 - 0.25
    ema_y_rest = (codes[3] / (NLEV - 1)) * 0.5 - 0.25
    # held field: RELINS stores q + (lx, ly) of the current peg (the exact public layout centre, no noise), so the range
    # is +-4 cm (board yaw <= 0.15 rad, |tip - hand| <= 16 mm) and the step 0.027 mm: a 1/20 running mean is not rounded away
    HR = 0.04 if RELINS else 0.016                                    # held-field half range (m)
    refx = _onehot_pick(F, [F.abs(peg_m - k) < 0.5 for k in range(8)], [r["lx"] for r in rows]) if RELINS else zero
    refy = _onehot_pick(F, [F.abs(peg_m - k) < 0.5 for k in range(8)], [r["ly"] for r in rows]) if RELINS else zero
    ema_x_off = (codes[2] / (NLEV - 1)) * 2 * HR - HR - refx
    ema_y_off = (codes[3] / (NLEV - 1)) * 2 * HR - HR - refy
    fresh = at_next | reset
    jx, jy = px_raw - ema_x_rest, py_raw - ema_y_rest
    jump = F.sqrt(jx * jx + jy * jy) > REST_RESET
    rx = F.where(fresh | jump, px_raw, ema_x_rest + PAVG * (px_raw - ema_x_rest))
    ry = F.where(fresh | jump, py_raw, ema_y_rest + PAVG * (py_raw - ema_y_rest))
    if RELINS:                                                         # q = (tip - hole) - hand: constant while held, bias-free
        if VERT_TIP:
            offx_raw, offy_raw = g("rx") - hx, g("ry") - hy
        else:
            offx_raw, offy_raw = g("rx") - g("ux") * PEG_LEN / 2 - hx, g("ry") - g("uy") * PEG_LEN / 2 - hy
    else:
        offx_raw, offy_raw = px_raw - hx, py_raw - hy
    enter_held = ph(P3) & (cnt < 0.5)
    restart = enter_held | fresh
    if CALRG:
        restart = restart | (ph(P4) & (cnt < 0.5))
    if HMEAN > 0:
        kk = kheld + 1.0
        wh = F.maximum(1.0 / kk, 1.0 / HMEAN)
        if HSTART_P4:             # closing (P3) shifts the peg in the jaws: EMA there, fresh mean from the first P4 step
            restart = restart | (ph(P4) & (cnt < 0.5))
            wh = F.where(ph(P3), PAVG, wh)
    else:
        wh = PAVG
    ox_ = F.where(restart, offx_raw, ema_x_off + wh * (offx_raw - ema_x_off))
    oy_ = F.where(restart, offy_raw, ema_y_off + wh * (offy_raw - ema_y_off))
    kheld_new = F.where(restart, 1.0, F.minimum(kheld + 1.0, 22.0))
    if CLOSED_CAL:
        if CLOSED_CAL_SQ:
            cc_mode = cc_mode & ~gb("ho")
        else:
            cc_mode = cc_mode & ~sq & ~gb("ho")
        cc_mark = F.floor(codes[1] / 256.0)
        cc_prep = ph(P3) & (cc_mark > 1.5)
        cc_cal = ph(CAL) & (cc_mark > 1.5)
        cc_neg_cal = ((cnt < 13.5) & (hyaw > 0.0)) | ((cnt >= 13.5) & (cc_mark > 2.5))
        cc_neg = (cc_cal & cc_neg_cal) | (cc_mode & cc_decode_neg)
        cc_sign = F.where(cc_neg, -1.0, 1.0)
        cc_bias = (codes[2] / (NLEV - 1.0)) * 0.032 - 0.016
        cc_offset = (codes[3] / (NLEV - 1.0)) * 0.032 - 0.016
        cc_c, cc_s = F.cos(hyaw), F.sin(hyaw)
        cc_zxw, cc_zyw = px_raw + ddx - hx, py_raw + ddy - hy
        cc_zx, cc_zy = cc_c * cc_zxw + cc_s * cc_zyw, -cc_s * cc_zxw + cc_c * cc_zyw
        cc_bias_new = F.where(cc_cal & (cnt < 13.5), cc_bias + (cc_zy - cc_bias) / F.maximum(cnt + 2.0, 1.0), cc_bias)
        cc_observed_offset = cc_zx - cc_sign * cc_bias_new
        cc_offset_new = F.where(cc_cal & (cnt > 33.5), cc_offset + (cc_observed_offset - cc_offset) / F.maximum(cnt - 33.0, 1.0),
                               F.where(cc_mode, cc_offset + (cc_observed_offset - cc_offset) / HMEAN, cc_offset))
        # Two physical constraints: centred between the jaws; longitudinal offset
        # estimated using the first jaw-bias component after the quarter-turn.
        cc_tipx = cc_c * cc_offset_new - ux * PEG_LEN / 2.0 - g("hx")
        cc_tipy = cc_s * cc_offset_new - uy * PEG_LEN / 2.0 - g("hy")
        ox_, oy_ = F.where(cc_mode, cc_tipx, ox_), F.where(cc_mode, cc_tipy, oy_)
    use_held = held_mode & touch
    if CLOSED_CAL:
        use_held = use_held | (cc_mode & (gap > dia - 0.002) & (gap < dia + 0.003))
    if RELINS:
        pex = F.where(held_mode, px_raw, rx)
        pey = F.where(held_mode, py_raw, ry)
    else:
        pex = F.where(use_held, hx + ox_, F.where(held_mode, px_raw, rx))
        pey = F.where(use_held, hy + oy_, F.where(held_mode, py_raw, ry))
    # RESTZ: tray pegs drop 5-20 mm and lean on neighbours; a fixed 25 mm centre made P3 close on air (node order kept so
    # the default build stays byte-identical)
    pez = F.where(use_held, hz + HELD_DZ, F.where(held_mode, pz_raw, F.clip(pz_raw, 0.012, 0.032) if RESTZ else (
        F.where((pz_raw > 0.035) & (g("dslot") > ZSNAP_D), 0.045, 0.025) if ZSNAP else 0.025 + zero)))   # ZSNAP: floor (25) / ON block (45)
    bottom = pez - PEG_LEN / 2

    # ---- yaw targets (round: keep the hand yaw; square: BYAW + SQHAND)
    if BETAMEM:
        bm_flat = sq & touch & held_mode & (gap > dia - 0.001) & (gap < dia + BM_FLAT)
        bm_spawn = (tstep < BM_T0) & touch & held_mode & (pz_raw > 0.035) & (ph(I0) | ph(I1) | ph(I2))
        bm_have = bm_flat | bm_spawn
        bm_r = _wrap90(F, pyaw - hyaw)                                  # = pose_bias_yaw (+ noise) when the twist is 0
        bm_known = b_prev > 0.5
        bm_v = (b_prev - 4.0) * BM_STEP
        bm_first = F.clip(F.round(bm_r / BM_STEP) + 4.0, 1.0, 7.0)
        bm_move = F.clip(b_prev + F.where(bm_r > bm_v, 1.0, -1.0), 1.0, 7.0)
        b_new = F.where(bm_have, F.where(bm_known, F.where(F.abs(bm_r - bm_v) > BM_HYST * BM_STEP, bm_move, b_prev), bm_first), b_prev)
        ybias_e = F.where(yden > 0.5, ybias, F.where(b_new > 0.5, (b_new - 4.0) * BM_STEP, 0.0))
    else:
        ybias_e = ybias
    py_true = F.where(touch, hyaw, pyaw - (ybias_e if BM_PY else ybias))
    yerr = _wrap90(F, byaw - py_true)
    if HELD:
        hexc, dflc = gb("hexs"), gb("dfl")
        y60 = lambda x: x - (math.pi / 3) * F.round(x / (math.pi / 3))  # noqa: E731
        yerr = F.where(hexc, y60(byaw - py_true), yerr)
        sq = sq | hexc | dflc
    if PICKFIX:
        pick_yaw = F.where(sq, hyaw + _wrap90(F, (pyaw - ybias_e) - hyaw), hyaw)
        if HELD:
            pick_yaw = F.where(hexc, hyaw + y60((pyaw - ybias_e) - hyaw), F.where(dflc, hyaw, pick_yaw))
    else:
        pick_yaw = F.where(sq, hyaw + yerr, hyaw)
    hole_yaw = F.where(sq, hyaw + yerr, hyaw)
    if SQTWIST:
        xg = F.clip(gap / (dia * math.sqrt(2.0) + 1e-9), 1.0 / math.sqrt(2.0), 0.999999)
        tw = F.atan(xg / F.sqrt(1.0 - xg * xg)) - math.pi / 4          # |twist| in the jaws
        orel = _wrap90(F, pyaw - hyaw)                                  # twist + pose_bias_yaw (+ obs noise)
        anch = ((yden > 0.5) | (b_new > 0.5)) if (BETAMEM and BM_SQTW) else yden > 0.5
        bp, bm = F.abs(_wrap90(F, orel - tw)), F.abs(_wrap90(F, orel + tw))
        inp = (bp >= SQTW_LO) & (bp <= SQTW_HI)
        inm = (bm >= SQTW_LO) & (bm <= SQTW_HI)
        one_root = (inp & ~inm) | (inm & ~inp)
        s_obs = F.where(orel >= 0.0, 1.0, -1.0)
        s_na = F.where(one_root, F.where(inp, 1.0, -1.0), s_obs)
        sgn = F.where(anch, F.where(_wrap90(F, orel - (ybias_e if BM_SQTW else ybias)) >= 0.0, 1.0, -1.0), s_na)
        ok_tw = anch | one_root if not SQTW_AMB else anch | (zero < 1.0)
        on_tw = sq & held_mode & (gap > dia + SQTW_G) & (gap < 1.42 * dia + 0.0005) & ok_tw
        hole_yaw = F.where(on_tw, hyaw + _wrap90(F, byaw - (hyaw + sgn * tw)), hole_yaw)
    if NBYAW:                     # turn the jaw axis (hand y) away from standing neighbours before the pick (1012/54: a peg 10 mm
        rt = dia / 2              # away sat in the closing pad's sweep and was squeezed off the table -> part_lost, episode lost)
        offs = [0.0, math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3, math.pi / 2]
        score = []
        for off in offs:
            th = pick_yaw + off; c_, s_ = F.cos(th), F.sin(th)
            n_c = zero
            for k, r in enumerate(rows):
                other = r["pres"] & ~r["lying"] & ~sel[k]
                dx_, dy_ = r["px"] - pex, r["py"] - pey
                xj, yj = c_ * dx_ + s_ * dy_, -s_ * dx_ + c_ * dy_
                rj = r["dia"] / 2
                hit = (F.abs(xj) < 0.008 + rj + NB_M) & (F.abs(yj) < 0.018 + rj + NB_M) & (F.abs(yj) > rt - rj - NB_M)
                n_c = n_c + F.where(other & hit, 1.0, 0.0)
            if off not in (0.0, math.pi / 2):
                n_c = F.where(sq, 99.0, n_c)          # squares: only face-aligned grips (0 / 90 deg)
            score.append(n_c)
        best = zero
        for off, n_c in reversed(list(zip(offs, score))):   # most preferred (smallest turn) conflict-free offset; none -> 0
            best = F.where(n_c < 0.5, off, best)
        free_any = zero > 1.0
        for n_c in score:
            free_any = free_any | (n_c < 0.5)
        pick_yaw = pick_yaw + F.where(score[0] < 0.5, 0.0, best)
        nb_sx, nb_sy = zero, zero
        if NBSHIFT:               # no conflict-free yaw: put the nearest neighbour along the pads (hand x) and grip the target
            bd, bdx, bdy = zero + 1.0, zero, zero          # off-centre, shifting the 16 mm pad span away from it (1013/54)
            for k, r in enumerate(rows):
                other = r["pres"] & ~r["lying"] & ~sel[k]
                dx_, dy_ = r["px"] - pex, r["py"] - pey
                d_ = F.sqrt(dx_ * dx_ + dy_ * dy_)
                closer = other & (d_ < bd)
                bd, bdx, bdy = F.where(closer, d_, bd), F.where(closer, dx_, bdx), F.where(closer, dy_, bdy)
            phi = _atan2(F, bdy, bdx)
            y_rnd = hyaw + (phi - hyaw) - math.pi * F.round((phi - hyaw) / math.pi)        # nearest yaw = phi mod pi
            c0, s0 = F.cos(pick_yaw), F.sin(pick_yaw)
            along0 = F.abs(c0 * bdx + s0 * bdy)
            along90 = F.abs(-s0 * bdx + c0 * bdy)
            y_sq = F.where(along90 > along0, pick_yaw + math.pi / 2, pick_yaw)
            stuck_ = ~free_any & (bd < 0.05)
            pick_yaw = F.where(stuck_, F.where(sq, y_sq, y_rnd), pick_yaw)
            c_, s_ = F.cos(pick_yaw), F.sin(pick_yaw)
            s_hi, s_lo = zero + 1.0, zero - 1.0
            for k, r in enumerate(rows):
                other = r["pres"] & ~r["lying"] & ~sel[k]
                dx_, dy_ = r["px"] - pex, r["py"] - pey
                xj, yj = c_ * dx_ + s_ * dy_, -s_ * dx_ + c_ * dy_
                rj = r["dia"] / 2
                near_y = F.abs(yj) < 0.018 + rj + NB_M
                s_hi = F.where(other & near_y & (xj > 0), F.minimum(s_hi, xj - rj - NB_M - 0.008), s_hi)
                s_lo = F.where(other & near_y & (xj <= 0), F.maximum(s_lo, xj + rj + NB_M + 0.008), s_lo)
            sh = F.minimum(F.maximum(zero, s_lo), s_hi)
            ok_ = stuck_ & (s_lo <= s_hi) & (F.abs(sh) <= NB_SMAX)
            nb_sx, nb_sy = F.where(ok_, sh * c_, 0.0), F.where(ok_, sh * s_, 0.0)

    # last commanded move per axis (command part of the echo; payloads are sub-deadzone) -> target estimate
    pq = [_decode(F, x)[0] for x in pa]
    if TGTV > 0:                  # critically damped joints: target - hand = (kv/kp) v; fit on 0bbb77d easy rmse 0.20 vs 0.28 mm
        lead = {id(hx): F.clip(vx * TGTV, -TGT_CLAMP, TGT_CLAMP), id(hy): F.clip(vy * TGTV, -TGT_CLAMP, TGT_CLAMP),
                id(hz): F.clip(vz * TGTV, -TGT_CLAMP, TGT_CLAMP)}
    else:
        lead = {id(hx): F.clip(pq[0] * DXYZ * TGT_LAMBDA, -TGT_CLAMP, TGT_CLAMP),
                id(hy): F.clip(pq[1] * DXYZ * TGT_LAMBDA, -TGT_CLAMP, TGT_CLAMP),
                id(hz): F.clip(pq[2] * DXYZ * TGT_LAMBDA, -TGT_CLAMP, TGT_CLAMP)}

    def servo(goal, cur, s):
        if TGTEST and id(cur) in lead:                 # author's _act: (goal - target) / DXYZ, gain 1, + pad
            e = goal - (cur + F.where(reset, 0.0, lead[id(cur)]))
            pad = F.where(F.abs(e) > max(TPADTH, 1e-9), F.where(e > 0, PAD, -PAD), 0.0)
            out = F.clip((e + pad) / DXYZ, -s, s)
            return F.where(F.abs(e) < TDEAD, 0.0, out) if TDEAD > 0 else out
        e = goal - cur
        pad = F.where(F.abs(e) > PAD_TH, F.where(e > 0, PAD, -PAD), 0.0)
        return F.clip((KP * e + pad) / DXYZ, -s, s)

    def yservo(goal, s):
        e = _wrap(F, goal - hyaw)
        pad = F.where(F.abs(e) > 0.002, F.where(e > 0, YPAD, -YPAD), 0.0)
        return F.clip((e + pad) / DYAW, -s, s)

    def arrived(gx, gy, gz, gyaw=None):
        e = F.maximum(F.maximum(F.abs(gx - hx), F.abs(gy - hy)), F.abs(gz - hz))
        if TGTEST:                                      # author's goto: target within tol, then hand within 3 tol
            lx_, ly_, lz_ = (F.where(reset, 0.0, lead[id(v)]) for v in (hx, hy, hz))
            et = F.maximum(F.maximum(F.abs(gx - hx - lx_), F.abs(gy - hy - ly_)), F.abs(gz - hz - lz_))
            ok = (et < TOL) & (e < 3 * TOL)
        else:
            ok = (e < TOL) & (speed < VTOL)
        if gyaw is not None:
            ok = ok & (F.abs(_wrap(F, gyaw - hyaw)) < 0.02)
        return ok

    # tip of the held peg (author: tip = p - axis * L/2; off = tip_xy - hand_xy)
    if VERT_TIP:
        tipx, tipy = pex, pey
    else:
        tipx, tipy = pex - ux * PEG_LEN / 2, pey - uy * PEG_LEN / 2
    offx, offy = tipx - hx, tipy - hy
    if RELINS:                    # hand goal = hole - off  <=>  -q ; tip-to-hole = hand + q
        holx, holy = zero, zero
        offx, offy = ox_, oy_
        tipx, tipy = hx + ox_, hy + oy_

    if PARK:                      # current peg is a lying round peg being parked
        park_cur = gb("lying") & ~sq
        carry_ph = ph(KC) | ph(KD)
        lpx, lpy = F.where(carry_ph, px_raw, pex), F.where(carry_ph, py_raw, pey)   # rest EMA (P1/P2); raw while carried (EMA lags)
        if PARK_CHASE >= 0:       # the rest EMA trails a rolling peg by ~6 steps: chase its latest position, led along its motion
            lpx = F.where(carry_ph, px_raw, px_raw + PARK_CHASE * (px_raw - pex))
            lpy = F.where(carry_ph, py_raw, py_raw + PARK_CHASE * (py_raw - pey))
        lux_c, luy_c = f_lux, f_luy
        off_c, zg_c = f_off, f_z
        ggx, ggy = lpx + off_c * lux_c, lpy + off_c * luy_c               # grasp point
        ay_ = _atan2(F, luy_c, lux_c)
        yaw_l = hyaw + (ay_ - hyaw) - math.pi * F.round((ay_ - hyaw) / math.pi)      # hand x along the peg, nearest turn
        yb_ = byaw + math.pi / 2
        yaw_b = hyaw + (yb_ - hyaw) - math.pi * F.round((yb_ - hyaw) / math.pi)      # hand x along board y (axis across the row)
        open_l = F.clip((dia + 0.012) / MAX_OPEN - 1.0, -1.0, 1.0)
        gpk_x, gpk_y, dz_c = gwx, gwy, f_dz
        cgx, cgy = hx + (gpk_x - lpx), hy + (gpk_y - lpy)                 # hand goal that puts the peg centre on the gap

        def arrived_l(gx_, gy_, gz_, gyaw_):   # lying-peg estimates jitter (pose noise 0.4-2 mm, quat 0.015-0.06 rad): looser arrival
            e_ = F.maximum(F.abs(gx_ - hx), F.abs(gy_ - hy))
            return (e_ < PARK_TXY) & (F.abs(gz_ - hz) < 0.0015) & (F.abs(_wrap(F, gyaw_ - hyaw)) < PARK_TYAW)

    # ---- per-phase actions [ax, ay, az, ayaw, grip] and exits
    acts, nxt = {}, {}
    # P1 above
    top = pez + PEG_LEN / 2
    g1z = top + 0.01
    if PREOPEN > 0:               # once a peg is in a hole (delta known, picks accurate) open only to dia + PREOPEN: the full
        known = have_d if (DELTA and not PREOPEN_ALL) else (zero > -1.0)   # 24 mm sweep squeezed neighbours (1012/54)
        want_gap = (dia + PREOPEN + WIDE * (0.016 - dia)) if WIDE > 0 and not WIDE_PICK_ONLY else (dia + PREOPEN)   # off = the original graph
        open_g = F.where(known, F.clip(want_gap / MAX_OPEN - 1.0, -1.0, 1.0), 1.0)
    else:
        open_g = one
    open_pick_g = open_g
    if PREOPEN > 0 and WIDE > 0 and WIDE_PICK_ONLY:
        pick_gap = dia + PREOPEN + WIDE * (0.016 - dia)
        open_pick_g = F.where(known, F.clip(pick_gap / MAX_OPEN - 1.0, -1.0, 1.0), 1.0)
    gpx, gpy = (pex + nb_sx, pey + nb_sy) if NBYAW else (pex, pey)
    if TILT_GRIP:                 # leaning (not lying) peg: pads at its top along the axis, jaws closing in the lean's direction
        tz_ = F.abs(g("az"))
        tilted_ = (tz_ < TG_AZ) & (tz_ > 0.5)
        tl_ = PEG_LEN / 2 - 0.008
        gpx = F.where(tilted_, pex + ux * tl_, gpx)
        gpy = F.where(tilted_, pey + uy * tl_, gpy)
        ty_ = _atan2(F, zero - ux, uy)                                   # hand y (the closing direction) along (ux, uy)
        dty_ = _wrap(F, ty_ - hyaw)
        pick_yaw = F.where(tilted_, hyaw + dty_ - math.pi * F.round(dty_ / math.pi), pick_yaw)   # nearest of the two jaw yaws
    acts[P1] = [servo(gpx, hx, 1.0), servo(gpy, hy, 1.0), servo(g1z, hz, 1.0), yservo(pick_yaw, 1.0), open_pick_g]
    nxt[P1] = arrived(gpx, gpy, g1z, pick_yaw) | (cnt > TMAX)
    # P2 descend to the grasp height (pads centred 8 mm under the top)
    zg = pez + (PEG_LEN / 2 - 0.008) - PAD_CENTRE_Z
    if TILT_GRIP:
        zg = F.where(tilted_, pez + (PEG_LEN / 2 - 0.008) * tz_ - PAD_CENTRE_Z, zg)
    acts[P2] = [servo(gpx, hx, S_P2), servo(gpy, hy, S_P2), servo(zg, hz, S_P2), yservo(pick_yaw, S_P2), open_pick_g]
    if P2GATE > 0:                # user: the wrist turns <= 4.6 deg per step; don't bring the pads down past the top while unsettled
        unsettled = (F.abs(_wrap(F, pick_yaw - hyaw)) > P2G_YAW) | (F.maximum(F.abs(gpx - hx), F.abs(gpy - hy)) > P2G_XY)
        hold_z = top + P2GATE - 0.004             # pad bottom = hand + 4 mm
        acts[P2] = [acts[P2][0], acts[P2][1], F.where(unsettled, servo(F.maximum(zg, hold_z), hz, S_P2), acts[P2][2]),
                    acts[P2][3], acts[P2][4]]
    nxt[P2] = arrived(gpx, gpy, zg, pick_yaw) | (cnt > TMAX)
    if PICK_FCAP:
        pick_load = ph(P2) & ~gb("lying") & (hz > 0.038) & (c(15) > 0.05) & (F.abs(c(11)) > 0.65 * c(15))
        relief = F.clip(F.abs(c(11)) / 0.4, 0.0, 1.0)
        acts[P2][2] = F.where(pick_load, F.maximum(acts[P2][2], relief), acts[P2][2])
    if P1E:
        p1e = (have_d if DELTA else zero > 1.0) & ~gb("lying") & (g("az") > 0.95) & (pez < 0.03) & \
            (F.abs(gpx - hx) < P1E_XY) & (F.abs(gpy - hy) < P1E_XY) & (F.abs(g1z - hz) < P1E_Z) & \
            (F.abs(_wrap(F, pick_yaw - hyaw)) < P1E_YAW)
        if NBYAW and P1E_NB:
            p1e = p1e & (F.abs(nb_sx) + F.abs(nb_sy) < 0.0002)
        nxt[P1] = nxt[P1] | p1e
    if CLOSED_CAL:
        cc_shift = F.abs(nb_sx) + F.abs(nb_sy) if NBYAW else zero
        if CLOSED_CAL_SQ:
            cc_eligible = ph(P2) & ~gb("ho") & ~gb("lying") & ~have_d & (g("az") > 0.985) & \
                (F.sqrt((gpx - hx) * (gpx - hx) + (gpy - hy) * (gpy - hy)) < 0.0015) & (F.abs(hz - zg) < 0.002) & (cc_shift < 0.0002)
        else:
            cc_eligible = ph(P2) & ~sq & ~gb("ho") & ~gb("lying") & ~have_d & (g("az") > 0.985) & \
                (F.sqrt((gpx - hx) * (gpx - hx) + (gpy - hy) * (gpy - hy)) < 0.0015) & (F.abs(hz - zg) < 0.002) & (cc_shift < 0.0002)
    if PARK:
        a1 = [servo(ggx, hx, 1.0), servo(ggy, hy, 1.0), servo(PARK_APPZ + zero, hz, 1.0), yservo(yaw_l, 1.0), open_l]
        if PARK_CATCH > 0:        # rolling peg: descend while closing in, jaws wide; hand over to P2 as soon as roughly there
            dxy_c = F.maximum(F.abs(ggx - hx), F.abs(ggy - hy))
            a1 = [a1[0], a1[1], servo(F.where(dxy_c < PARK_CATCH, zg_c, PARK_APPZ + zero), hz, 1.0), a1[3], a1[4]]
        acts[P1] = [F.where(park_cur, x1, x0) for x1, x0 in zip(a1, acts[P1])]
        if PARK_CATCH > 0:
            p1_ok = (dxy_c < CATCH_ARR) & (F.abs(_wrap(F, yaw_l - hyaw)) < CATCH_YAW)
            nxt[P1] = (park_cur & (p1_ok | (cnt > TMAX))) | (~park_cur & nxt[P1])
        else:
            nxt[P1] = (park_cur & (arrived_l(ggx, ggy, PARK_APPZ + zero, yaw_l) | (cnt > TMAX))) | (~park_cur & nxt[P1])
        a2 = [servo(ggx, hx, PARK_S2), servo(ggy, hy, PARK_S2), servo(zg_c, hz, PARK_S2), yservo(yaw_l, PARK_S2), open_l]
        acts[P2] = [F.where(park_cur, x1, x0) for x1, x0 in zip(a2, acts[P2])]
        if PARK_CATCH > 0:
            p2_ok = (F.maximum(F.abs(ggx - hx), F.abs(ggy - hy)) < CATCH_ARR) & (F.abs(zg_c - hz) < 0.0015) & \
                (F.abs(_wrap(F, yaw_l - hyaw)) < CATCH_YAW)
            nxt[P2] = (park_cur & (p2_ok | (cnt > TMAX))) | (~park_cur & nxt[P2])
        else:
            nxt[P2] = (park_cur & (arrived_l(ggx, ggy, zg_c, yaw_l) | (cnt > TMAX))) | (~park_cur & nxt[P2])
    # P3 close 12
    acts[P3] = [zero, zero, zero, zero, zero - 1.0]
    nxt[P3] = cnt > P3_N
    # CAL (counter-driven, no xy motion): 0-5 open, 6-13 rise to top + 10 mm, 14-33 turn +90 deg, 34-41 descend to the grasp
    # height, 42-47 close; then P4. The hand never moves in xy, so it keeps the true peg position along the first jaw axis.
    # 0-5 open, 6-13 rise, 14-33 turn +90 deg, 34-47 descend (full speed, then settle), 48-53 close; then P4
    cal_open, cal_rise = cnt < 5.5, (cnt > 5.5) & (cnt < 13.5)
    cal_turn, cal_down = (cnt > 13.5) & (cnt < 33.5), (cnt > 33.5) & (cnt < 47.5 + CALWAIT)
    if CALDIR:                    # direction bit: set from the hand yaw before the turn (cnt < 14), then kept in codes[1] // 256
        cal_neg = ((cnt < 13.5) & (hyaw > 0.0)) | ((cnt > 13.5) & (F.floor(codes[1] / 256) > 0.5))
    if CLOSED_CAL:
        cal_neg = (cc_cal & cc_neg) | (~cc_cal & cal_neg)
    acts[CAL] = [zero, zero,
                 F.where(cal_rise, servo(pez + PEG_LEN / 2 + 0.01, hz, 1.0), F.where(cal_down, servo(zg, hz, 1.0), 0.0)),
                 F.where(cal_turn, F.where(cal_neg, -0.98, 0.98) if CALDIR else 0.98 + zero, 0.0),
                 F.where(cal_open | cal_rise | cal_turn | cal_down, 1.0, -1.0)]
    nxt[CAL] = cnt > 53.5 + CALWAIT
    if CLOSED_CAL:
        cc_actions = [zero, zero, servo(zero + LIFT_Z, hz, 1.0),
                      F.where(cal_turn, F.where(cc_neg, -0.98, 0.98), 0.0), zero - 1.0]
        acts[CAL] = [F.where(cc_cal, ca, old) for ca, old in zip(cc_actions, acts[CAL])]
    gh = (lambda: F.clip((dia - HOLDG) / MAX_OPEN - 1.0, -1.0, 1.0)) if HOLDG > 0 else (lambda: zero - 1.0)   # fresh node per use (off = original graph)
    # P4 lift to 75 mm
    if LIFT_HI > 0:               # c3: carried pegs tip by clipping standing pegs (tops 50 mm, 70 mm on the block). Lift high
        hsx, hsy = g("hx"), g("hy")                      # only if a tall upright peg is near the straight path hand -> hole
        sdx, sdy = hsx - hx, hsy - hy; sl2 = sdx * sdx + sdy * sdy + 1e-9
        tall_hit = zero > 1.0
        for k, r in enumerate(rows):
            other = r["pres"] & ~r["lying"] & ~sel[k] & (r["pz"] + PEG_LEN / 2 > LIFT_TALL)
            tt = F.clip(((r["px"] - hx) * sdx + (r["py"] - hy) * sdy) / sl2, 0.0, 1.0)
            ex_, ey_ = r["px"] - (hx + tt * sdx), r["py"] - (hy + tt * sdy)
            tall_hit = tall_hit | (other & (F.sqrt(ex_ * ex_ + ey_ * ey_) < LIFT_PATH_R))
        lift_t = F.where(tall_hit, LIFT_HI, LIFT_Z)
        acts[P4] = [zero, zero, servo(lift_t, hz, S_P4), zero, gh()]
        nxt[P4] = arrived(hx, hy, lift_t) | (cnt > TMAX)
    else:                         # original two expressions (byte-identical default build)
        acts[P4] = [zero, zero, servo(LIFT_Z + zero, hz, S_P4), zero, gh()]
        nxt[P4] = arrived(hx, hy, LIFT_Z + zero) | (cnt > TMAX)
        if PARK:
            acts[P4] = [acts[P4][0], acts[P4][1], F.where(park_cur, servo(PARK_LIFT + zero, hz, S_P4), acts[P4][2]), acts[P4][3], acts[P4][4]]
            nxt[P4] = (park_cur & (arrived(hx, hy, PARK_LIFT + zero) | (cnt > TMAX))) | (~park_cur & nxt[P4])
        if P4_EARLY > 0:
            nxt[P4] = nxt[P4] | (hz > LIFT_Z - P4_EARLY)
    # I0 raise if the bottom is below 24 mm
    need_up = bottom < BLOCK_H + 0.004
    acts[I0] = [zero, zero, servo(hz + (BLOCK_H + 0.004 - bottom), hz, 1.0), zero, gh()]
    nxt[I0] = ~need_up | (cnt > TMAX)
    # I1 align the tip over the hole (<= 140)
    acts[I1] = [servo(holx - offx, hx, 1.0), servo(holy - offy, hy, 1.0), zero, yservo(hole_yaw, 1.0), gh()]
    if P4_EARLY > 0 or I1_DESC > 0:
        i1z = zero
        if P4_EARLY > 0:          # keep rising to the lift height while moving (P4 left early)
            i1z = F.where(hz < (lift_t if LIFT_HI > 0 else LIFT_Z + zero) - 0.0005, servo((lift_t if LIFT_HI > 0 else LIFT_Z + zero), hz, 1.0), i1z)
        if I1_DESC > 0:           # tip near the hole: start the descent, never below the block + floor
            exy = F.sqrt((tipx - holx) * (tipx - holx) + (tipy - holy) * (tipy - holy))
            i1z = F.where((exy < I1_DESC) & (bottom > BLOCK_H + I1_DESC_FLOOR), zero - SP_HI, i1z)
        acts[I1] = [acts[I1][0], acts[I1][1], i1z, acts[I1][3], acts[I1][4]]
    tdx, tdy = tipx - holx, tipy - holy
    tip_ok = F.sqrt(tdx * tdx + tdy * tdy) < 0.0006
    nxt[I1] = (tip_ok & (F.abs(_wrap(F, hole_yaw - hyaw)) < 0.03)) | (cnt > 139)
    # I2 descend with the lip spiral (<= 200)
    band = (bottom <= BLOCK_H + 0.0015) & (bottom > 0.001)
    if BANDCNT:                   # previous z command -0.35 / -0.7 = it was outside the band (in band: -0.33 or FLIM-limited)
        band_entry = ph(I2) & band & (pq[2] < -0.34)
        cnt_sp = F.where(band_entry, 1.0, cnt)
    else:
        band_entry, cnt_sp = zero < -1.0, cnt
    if SPIRAL > 0:                # c3: Archimedean spiral (pitch ~ clearance, constant arc step) covers the disc evenly;
        kk = SPIRAL / (2 * math.pi)                     # r = r0 + kk*th; arc s = r0*th + kk*th^2/2 = SPARC * n (closed form)
        thm = (SPRMAX - SPR0) / kk; smax = SPR0 * thm + kk * thm * thm / 2
        u = SPARC * cnt_sp; um = u - 2 * smax * F.floor(u / (2 * smax)); s_ = smax - F.abs(um - smax)   # out, back in, out
        th_ = (F.sqrt(SPR0 * SPR0 + 2 * kk * s_) - SPR0) / kk; rr = SPR0 + kk * th_
        wx = F.where(band, rr * F.cos(th_), 0.0)
        wy = F.where(band, rr * F.sin(th_), 0.0)
        if SPRMAX_NA > 0 and DELTA:   # no in-hole anchor yet (delta rough): aim errors p50 2.5 mm, 54% beyond 2.5 mm -> wider spiral
            kk2 = SPIRAL_NA / (2 * math.pi)
            thm2 = (SPRMAX_NA - SPR0) / kk2; smax2 = SPR0 * thm2 + kk2 * thm2 * thm2 / 2
            u2 = SPARC * cnt_sp; um2 = u2 - 2 * smax2 * F.floor(u2 / (2 * smax2)); s2_ = smax2 - F.abs(um2 - smax2)
            th2 = (F.sqrt(SPR0 * SPR0 + 2 * kk2 * s2_) - SPR0) / kk2; rr2 = SPR0 + kk2 * th2
            wx = F.where(have_d, wx, F.where(band, rr2 * F.cos(th2), 0.0))
            wy = F.where(have_d, wy, F.where(band, rr2 * F.sin(th2), 0.0))
    else:
        amp = F.minimum(0.0025, 0.0004 + 0.00008 * cnt_sp)
        wx = F.where(band, amp * F.sin(cnt_sp * 0.9), 0.0)
        wy = F.where(band, amp * F.cos(cnt_sp * 0.7), 0.0)
    if SPIN_Z > 0:                # tip inside the hole (est. bottom below SPIN_Z): stop the lateral search, let it slide (wedging)
        wx = F.where(bottom < SPIN_Z, 0.0, wx)
        wy = F.where(bottom < SPIN_Z, 0.0, wy)
    if SPMOVE > 0:                # the peg is sliding down (hand vz < -SPMOVE m/s): no lateral search while it moves; search only
        sliding = vz < -SPMOVE    # when stalled (SPIN_Z by depth helps easy +0.014 but hurts hard -0.05: depth is unreliable there)
        wx = F.where(sliding, 0.0, wx)
        wy = F.where(sliding, 0.0, wy)
    wyaw = F.where(band & sq, 0.04 * F.sin(cnt_sp * 0.5), 0.0)
    if RF_JAM:
        jam_mode = sq & ph(I2) & (kheld >= 19.0)
        old_stuck = F.where(marked & jd["i2"], jd["stuck"], 0.0)
        stuck_now = F.where(band, old_stuck + 1.0, 0.0)
        if RF_JAM > 0:
            ja = F.minimum(0.0025, 0.0004 + 0.00008 * stuck_now)
            wx = F.where(jam_mode, F.where(band, ja * F.sin(stuck_now * 0.9), 0.0), wx)
            wy = F.where(jam_mode, F.where(band, ja * F.cos(stuck_now * 0.7), 0.0), wy)
            wyaw = F.where(jam_mode, F.where(band, 0.04 * F.sin(stuck_now * 0.5), 0.0), wyaw)
    sp = F.where(bottom > BLOCK_H + 0.003, SP_HI, SP_LO)
    az_i2 = F.where(band, BAND_AZ, zero - sp) if BANDCNT else zero - sp
    if BORE_AZ < 0:             # deep in the bore the lip can no longer catch the tip: descend faster (FLIM still caps a jam)
        az_i2 = F.where(band & (bottom < BLOCK_H - BORE_DZ), BORE_AZ + zero, az_i2)
    if RF_JAM > 0:
        az_i2 = F.where(jam_mode, zero - sp, az_i2)
    if FLIM > 0:                  # hand not moving down: let the target run at most FLIM action units (x 4 mm) below it
        stalled = vz * 0.04 > -0.0002
        az_i2 = F.where(stalled, F.minimum(F.maximum(az_i2, -FLIM - stall), 0.0), az_i2)
        stall_new = F.where(stalled & ph(I2), F.maximum(stall + az_i2, -FLIM), 0.0)
    else:
        stall_new = zero
    if FGX != 0 or FGY != 0:      # force-guided lip search: on the lip the lateral wrist force (obs 9-10, world frame, N x 0.05)
        cy_, sy_ = F.cos(hyaw), F.sin(hyaw)                 # points at the hole along the jaw axis (90-100% sign agreement at
        fwx, fwy = c(9), c(10)                              # > 1 N) and away from it along the pads; nudge the aim with it
        fhx_, fhy_ = cy_ * fwx + sy_ * fwy, -sy_ * fwx + cy_ * fwy
        on_ = band & (F.sqrt(fwx * fwx + fwy * fwy) > FTH)
        chx_, chy_ = F.where(on_, FGX * fhx_, 0.0), F.where(on_, FGY * fhy_, 0.0)
        wx = wx + cy_ * chx_ - sy_ * chy_
        wy = wy + sy_ * chx_ + cy_ * chy_
    acts[I2] = [F.clip(servo(holx - offx + wx, hx, 1.0), -sp, sp), F.clip(servo(holy - offy + wy, hy, 1.0), -sp, sp),
                az_i2, F.clip(yservo(hole_yaw + wyaw, 1.0), -sp, sp), gh() if HOLDG_I2 else zero - 1.0]
    nxt[I2] = (bottom < 0.0006) | seated | (cnt > I2MAX)
    if DROPIN > 0:              # the tip is well inside the bore and still sliding: let gravity finish (saves the bore push + release)
        nxt[I2] = nxt[I2] | (band & (bottom < BLOCK_H - DROPIN) & (vz < -DROP_VZ))
    if RF_JAM > 0:
        nxt[I2] = (jam_mode & ((bottom < 0.0006) | seated | (cnt > 198.5))) | (~jam_mode & nxt[I2])
    if RF_JAM:
        first_descent = ~marked | ~jd["i2"] | (cnt < 0.5)
        ref_bottom = F.where(first_descent, bottom, jd["best"])
        progress = first_descent | (bottom < ref_bottom - 0.0005)
        next_best = F.where(progress, bottom, ref_bottom)
        # c3 updates progress on EVERY descent step, not just inside the band.
        next_age = F.where(progress, 0.0, F.where(band, jd["age"] + 1.0, jd["age"]))
        jam = jam_mode & band & (next_age >= 29.5) & (jretry < 2.5) & ~nxt[I2]
        if RF_JAM < 0:
            jam = zero < -1.0
        # Compute at the first lift observation, AFTER the last descent's physics.
        lift_goal = F.where(cnt < 0.5, hz + F.maximum(BLOCK_H + 0.004 - bottom, 0.004), jd["goal"])
        acts[JUP] = [zero, zero, servo(lift_goal, hz, 0.5), zero, zero - 1.0]
        nxt[JUP] = arrived(hx, hy, lift_goal) | (cnt > 28.5)
        acts[JALIGN] = [servo(holx - offx, hx, 1.0), servo(holy - offy, hy, 1.0), zero,
                       yservo(hole_yaw, 1.0), zero - 1.0]
        nxt[JALIGN] = (tip_ok & (F.abs(_wrap(F, hole_yaw - hyaw)) < 0.03)) | (cnt > 58.5)
    # I3 open 8
    acts[I3] = [zero, zero, zero, zero, open_g]
    nxt[I3] = cnt > I3_N
    # I4 retreat to 80 mm
    g_tr = (GTRANS + zero) if GTRANS > 0 else open_g     # effort term: msa = mean sum(a^2); a4 = +-1 every step costs ~0.0075/peg
    if RELMIN > 0:                # release: open just past the peg, so the pads don't sweep into a neighbour
        rel_g = F.clip((dia + RELMIN) / MAX_OPEN - 1.0, -1.0, 1.0)
        acts[I3] = [zero, zero, zero, zero, rel_g]
        g_tr = rel_g
    acts[I4] = [zero, zero, servo(PEG_LEN + 0.03 + zero, hz, 1.0), zero, g_tr]
    nxt[I4] = arrived(hx, hy, PEG_LEN + 0.03 + zero) | (cnt > 59)
    # I5 hold open 13
    acts[I5] = [zero, zero, zero, zero, g_tr]
    nxt[I5] = cnt > SEATED_HOLD + 6.5
    # RT 12 steps up and open
    acts[RT] = [zero, zero, one, zero, g_tr]
    nxt[RT] = cnt > 10.5
    if FASTREL:                   # the env's seated flag (first seat recorded, gripper clear) is in the obs: once it is set and the
        done_ = seated & (hz > FASTREL_Z)          # hand is clear, the hold and the rise only delay the next peg's seat time
        nxt[I4], nxt[I5] = nxt[I4] | done_, nxt[I5] | seated
        if FASTREL == 1:          # 2 keeps RT's 12-step rise: pegs standing on the block have tops at 70 mm (c3)
            nxt[RT] = nxt[RT] | seated
    if FASTREL2:
        fr2_go = (g("dslot") < FR2_D) & (F.abs(pz_raw - 0.025) < 0.004) & (g("az") > 0.95) & \
            (hz > PEG_LEN - 0.004 + FR2_Z)                 # pad bottom (hand + 4 mm) FR2_Z above the seated top (50 mm)
        nxt[I3] = nxt[I3] | ((gap > dia + FR2_G) & (cnt > 1.5))
        nxt[I4], nxt[I5], nxt[RT] = nxt[I4] | fr2_go, nxt[I5] | fr2_go, nxt[RT] | fr2_go
    if PARK:
        # KC: carry the peg centre over the gap, turn its axis across the row
        acts[KC] = [servo(cgx, hx, 1.0), servo(cgy, hy, 1.0), servo(PARK_LIFT + zero, hz, 1.0), yservo(yaw_b, 1.0), zero - 1.0]
        dgx, dgy = lpx - gpk_x, lpy - gpk_y
        nxt[KC] = ((F.sqrt(dgx * dgx + dgy * dgy) < 0.002) & (F.abs(_wrap(F, yaw_b - hyaw)) < 0.03)) | (cnt > 90)
        # KD (counter): lower to the release height (KD_LOW steps), open (KD_OPEN), rise (KD_RISE)
        kd_low, kd_open = cnt < KD_LOW - 0.5, (cnt > KD_LOW - 0.5) & (cnt < KD_LOW + KD_OPEN - 0.5)
        acts[KD] = [F.where(kd_low, servo(cgx, hx, 1.0), 0.0), F.where(kd_low, servo(cgy, hy, 1.0), 0.0),
                    F.where(kd_low, servo(dz_c, hz, 1.0), F.where(kd_open, 0.0, 1.0)), F.where(kd_low, yservo(yaw_b, 1.0), 0.0),
                    F.where(kd_low, -1.0, 1.0)]
        nxt[KD] = cnt > KD_LOW + KD_OPEN + KD_RISE - 2.5
        if PARK_SETDOWN:          # KD on an UPRIGHT held peg = set it down: lower in place until its foot is 1 mm over the board,
            sd_ = ~park_cur       # open, rise; then NEXT parks the lying round pegs (the hand is free) and picks this one later
            z_sd = hz + (BLOCK_H + 0.001 - bottom)
            a_sd = [zero, zero, F.where(kd_low, servo(z_sd, hz, 1.0), F.where(kd_open, 0.0, 1.0)), zero, F.where(kd_low, -1.0, 1.0)]
            acts[KD] = [F.where(sd_, x1, x0) for x1, x0 in zip(a_sd, acts[KD])]

    # IDLE: rise to 80 mm, then hold still (effort term)
    acts[IDLE] = [zero, zero, F.where(hz < 0.08, 0.2, 0.0), zero, (GTRANS + zero) if GTRANS > 0 else one]
    nxt[IDLE] = any_new                                               # a peg became available again

    # NEXT resolves within the same step: a held start goes to I0, an upright peg to P1, nothing left to IDLE
    start_phase = F.where(~any_new, float(IDLE), F.where(held_start, float(I0), float(P1)))
    if PARK and PARK_SETDOWN:     # held at the start while a lying round peg could roll off: set the held one down first
        lr_any = zero > 1.0
        for r in rows:
            lr_any = lr_any | r["pk_cand"]
        start_phase = F.where(~any_new, float(IDLE), F.where(held_start, F.where(lr_any, float(KD), float(I0)), float(P1)))
    phase_eff = F.where(at_next | (ph(IDLE) & any_new), start_phase, phase)
    cnt_eff = F.where(at_next | (ph(IDLE) & any_new), 0.0, cnt)
    if BANDCNT:
        cnt_eff = F.where(band_entry, 1.0, cnt_eff)
    if RF_JAM > 0:
        # Keep total descent steps independent of a baseline BANDCNT restart.
        cnt_eff = F.where(jam_mode, cnt, cnt_eff)
        # c3 checks the post-yield seated flag before entering a planned retry.
        cancel_jam = marked & seated & ((ph(I2) & (cnt > 0.5)) | (ph(JUP) & (cnt < 0.5)))
        phase_eff = F.where(cancel_jam, float(I3), phase_eff)
        cnt_eff = F.where(cancel_jam, 0.0, cnt_eff)
        nxt[I3] = nxt[I3] & ~cancel_jam
    pe = lambda k: F.abs(phase_eff - k) < 0.5  # noqa: E731

    action = [zero, zero, zero, zero, zero]
    for k in range(1, NPH):
        if k == NEXT:
            continue
        for i in range(5):
            action[i] = F.where(pe(k), acts[k][i], action[i])
    if EMPTY_ABORT:              # missed grasp: the fingers closed past the peg; leave P4/I0/I1/I2 for I3 (open) at once
        empty = (gap < dia - 0.0015) & (cnt > 3.5)
        if PARK:                  # a dropped peg while carrying: go straight to the release (it opens and rises)
            nxt[KC] = nxt[KC] | empty
        if RF_JAM > 0:
            empty = (gap < dia - 0.0015) & ((cnt > 3.5) | marked)
            jam = jam & ~empty
        for k in ((P4, I0, I1, I2, JUP, JALIGN) if RF_JAM else (P4, I0, I1, I2)):
            nxt[k] = nxt[k] | empty
    adv = zero < -1.0
    for k in range(1, NPH):
        if k in nxt:
            adv = adv | (pe(k) & nxt[k])
    if ENDRISE > 0:
        kpres = sum(F.where(r["pres"], 1.0, 0.0) for r in rows)
        late = tstep >= kpres * STEPS_PER_PEG - ENDRISE
        rise = [zero, zero, F.where(hz < 0.08, 1.0, 0.0), zero, (GTRANS + zero) if GTRANS > 0 else one]
        action = [F.where(late, rv, av) for rv, av in zip(rise, action)]
    # phase after this step
    succ = {P1: P2, P2: P3, P3: P4, P4: I0, I0: I1, I1: I2, I2: I3, I3: I4, I4: I5, I5: RT, RT: NEXT, IDLE: NEXT, CAL: P4}
    if RF_JAM:
        succ.update({JUP: JALIGN, JALIGN: I2})
    nph = phase_eff
    for k, v in succ.items():
        nph = F.where(pe(k) & adv, float(v), nph)
    if CALRG and DELTA:           # first pick with no in-hole anchor: P3 -> CAL instead of P4
        nph = F.where(pe(P3) & adv & ~have_d & ~park_cur, float(CAL), nph) if PARK else F.where(pe(P3) & adv & ~have_d, float(CAL), nph)
    if PARK:                      # a parked-peg pick: P4 -> carry -> release -> NEXT
        nph = F.where(pe(P4) & adv & park_cur & ~empty, float(KC), nph) if EMPTY_ABORT else F.where(pe(P4) & adv & park_cur, float(KC), nph)
        nph = F.where(pe(KC) & adv, float(KD), nph)
        nph = F.where(pe(KD) & adv, float(NEXT), nph)
    if RF_JAM:
        nph = F.where(jam, float(JUP), nph)
        if EMPTY_ABORT:
            nph = F.where((pe(JUP) | pe(JALIGN)) & empty, float(I3), nph)
    ncnt = F.where(adv, 0.0, F.minimum(cnt_eff + 1.0, 255.0))
    if RF_JAM:
        ncnt = F.where(jam, 0.0, ncnt)
    # attempted mask: set the current peg's bit when its attempt ends (RT -> NEXT)
    bitk = _onehot_pick(F, sel, [zero + float(2 ** k) for k in range(8)])
    cur_bit = F.floor(mask_eff / F.maximum(bitk, 1.0)) - 2 * F.floor(mask_eff / F.maximum(2 * bitk, 1.0))
    nmask = F.where(pe(RT) & adv & (cur_bit < 0.5), mask_eff + bitk, mask_eff)
    # memory payloads
    # the stored form follows the phase AFTER this step: held phases keep the offset EMA, the others the rest EMA
    nheld = (nph >= P3 - 0.5) & (nph <= I2 + 0.5)
    if RF_JAM:
        nheld = nheld | (F.abs(nph - JUP) < 0.5) | (F.abs(nph - JALIGN) < 0.5)
    nrefx = _onehot_pick(F, sel, [r["lx"] for r in rows]) if RELINS else zero
    nrefy = _onehot_pick(F, sel, [r["ly"] for r in rows]) if RELINS else zero
    ema_xc = F.where(nheld, (ox_ + nrefx + HR) / (2 * HR), (rx + 0.25) / 0.5)
    ema_yc = F.where(nheld, (oy_ + nrefy + HR) / (2 * HR), (ry + 0.25) / 0.5)
    c2 = F.round(F.clip(ema_xc, 0.0, 1.0) * (NLEV - 1))
    c3 = F.round(F.clip(ema_yc, 0.0, 1.0) * (NLEV - 1))
    c0 = nph + 16 * kcur + 128 * F.where(nheld & held_mode, kheld_new, F.where(nheld, 1.0, 0.0))
    sq_code = F.where(pe(I2) & (nph > I2 - 0.5) & (nph < I2 + 0.5), F.round(-stall_new / STALL_Q), 0.0)
    if CALDIR:                    # sq_code is 0 outside I2, so CAL may use the same field for its direction bit
        sq_code = sq_code + F.where(pe(CAL) & (F.abs(nph - CAL) < 0.5) & cal_neg, 1.0, 0.0)
    if CLOSED_CAL:
        cc_next_cal = F.abs(nph - CAL) < 0.5
        cc_next_p3 = F.abs(nph - P3) < 0.5
        cc_handoff = pe(CAL) & cc_cal & adv & nheld
        cc_keep = cc_mode & nheld
        cc_enter = pe(P3) & cc_prep & adv & cc_next_cal
        cc_fields = cc_keep | cc_handoff | (cc_cal & cc_next_cal) | cc_enter
        cc_store_bias = F.where(cc_enter, cc_zy, cc_bias_new)
        cc_store_off = F.where(cc_enter, 0.0, cc_offset_new)
        cc_bcode = F.round(F.clip((cc_store_bias + 0.016) / 0.032, 0.0, 1.0) * (NLEV - 1.0))
        cc_ocode = F.round(F.clip((cc_store_off + 0.016) / 0.032, 0.0, 1.0) * (NLEV - 1.0))
        c2, c3 = F.where(cc_fields, cc_bcode, c2), F.where(cc_fields, cc_ocode, c3)
        c0 = F.where(cc_keep | cc_handoff, nph + 16.0 * kcur, c0)
        c1 = F.clip(ncnt, 0.0, 255.0) + 256 * F.clip(sq_code, 0.0, 10.0)
        cc_prep_next = (pe(P2) & adv & cc_eligible & cc_next_p3) | (pe(P3) & cc_prep & (cc_next_p3 | cc_next_cal))
        c1 = F.where(cc_prep_next, ncnt + 512.0, c1)
        cc_dir_out = F.where(cc_neg, 1.0, 0.0)
        c1 = F.where(cc_cal & cc_next_cal, ncnt + 256.0 * (2.0 + cc_dir_out), c1)
        cc_stall_out = F.where(stall_new < -FLIM * 0.5, 1.0, 0.0)
        cc_next_i2 = F.abs(nph - I2) < 0.5
        c1 = F.where(cc_keep | cc_handoff, ncnt + 256.0 * F.where(cc_next_i2, cc_stall_out + 2.0 * cc_dir_out, cc_dir_out), c1)
        cc_event = cc_eligible | cc_prep | cc_cal | cc_mode
    out = [_encode(F, action[0], c0), _encode(F, action[1], c1 if CLOSED_CAL else F.clip(ncnt, 0.0, 255.0) + 256 * F.clip(sq_code, 0.0, 10.0)),
           _encode(F, action[2], c2),
           _encode(F, action[3], c3),
           (_encode_g4(F, action[4], F.minimum(tstep + 1.0, NLEV - 1.0) if ENDRISE > 0 else nmask, b_new if BETAMEM else zero)
            if G4 else _encode(F, action[4], F.minimum(tstep + 1.0, NLEV - 1.0) if ENDRISE > 0 else nmask))]
    if RF_JAM:
        ni2 = F.abs(nph - I2) < 0.5
        nup, nalign = F.abs(nph - JUP) < 0.5, F.abs(nph - JALIGN) < 0.5
        use_joint = (sq & ni2 & (kheld_new >= 19.0)) | nup | nalign
        fresh_i2 = ~ph(I2) & ni2
        nr = F.where(jam, jretry + 1.0, jretry)
        nb = F.clip(F.round(F.where(fresh_i2, bottom, next_best) / 0.0005) - 1.0, 0.0, 43.0)
        na = F.where(fresh_i2, 0.0, F.minimum(next_age, 29.0))
        ns = F.where(fresh_i2, 0.0, F.minimum(stuck_now, 200.0))
        ng = F.where(ph(JUP), lift_goal, 0.0)
        wire = _jam_wire(F, action, kcur, nr, nb, F.round(sq_code / 10.0), ncnt, na, ns,
                         ni2, F.where(nalign, 1.0, 0.0), ng, tstep + 1.0,
                         F.clip(ema_xc, 0.0, 1.0) * 0.08 - 0.04,
                         F.clip(ema_yc, 0.0, 1.0) * 0.08 - 0.04)
        out = [F.where(use_joint, w, a) for w, a in zip(wire, out)]
    return F.concat(out)
