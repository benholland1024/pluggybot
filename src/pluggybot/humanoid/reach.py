"""Reaching (issue #507, stage 2): a hand sent to a pose in the stance frame
while the standing policy balances (`policy.StandDriver`), as a
manipulator's reach is made -- the posture and the arm's last joints
PLANNED on the robot's model of itself (`plan`), the arm MOVED there in
joint space to a pre-grasp pose short of the target, then the hand
APPROACHING it along a straight line, its joints solved again every
command as the trunk sways (`arms.SelfModel`), and HOLDING there. A
retreat goes back out along that line, then home in joint space by a way
judged on the same model (`way_home`). The grippers are driven and read
beside it (`arms.Gripper`). Everything it reads is the robot's own senses.

⚠ From a hanging arm a big move solved in one jump falls into the wrong
basin (the shoulder thrown back, the elbow over): a hanging arm is nearly
singular for a hand moving up. So the plan's solve starts from several
arms (`SEEDS`) and the motion's from wherever the arm last was.

A hand that is OUT moves on CARRIED (#493): along a straight line to a
point, or through a path's points in turn (`carry`), from where the hand
IS, no faster than `HOLD_SPEED`, in the least-bent posture that reaches
(`posture_for`) or follows the path (`path_for`), stopping where the
arm's own solve misses rather than command the miss. Its turn is the one
it is asked to keep -- a path may turn it point by point -- and never
yields to what twists it: a drop handle in a hand that pivoted, or whose
turn followed a twist past 4 or 10 deg, swung round its pin under the
first push and buckled (SimNotes, "The humanoid opens the chest").

`Reach.step()` runs once a physics step, after the standing policy's and
before `mj_step`; the arms take new commands every `EVERY_S`.
"""

from dataclasses import dataclass
import itertools
import math
import os

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import (REACH_POS_TOL, REACH_ROT_TOL, Arm, Gripper, Motion,
                                     HandGoal, SelfModel, Waist, _rot_error, kept_value,
                                     value_back)
from pluggybot.humanoid.policy import HEIGHT_RANGE, PITCH_RANGE, StandDriver


#: The arms' commands are refreshed this often, s: 100 Hz, inside the RS
#: MIT mode's 1 kHz, and a trunk's sway between two is under a millimetre
#: at the hand.
EVERY_S = 0.01
#: A hand's fastest along its line, m/s, and its grip's fastest turn, rad/s
#: (a person's careful reach is 0.2-0.5 m/s).
HAND_SPEED, HAND_TURN = 0.2, 1.0
#: An arm moving in joint space does so no faster than this at its fastest
#: joint, rad/s, every joint arriving together.
JOINT_RATE = 1.0
#: The pre-grasp: this far short of the target, m, back along the way the
#: fingers point.
APPROACH_M = 0.08
#: A plan's slack: the arm must also reach this much further out, m, away
#: from its shoulder -- a goal planned at full stretch was 2.8 cm out of
#: reach of the posture the policy held, 2 cm under the one asked. It is
#: slack over the robot as it stands (`Lean`), never the lean itself: kept
#: over a statue balanced at the soles' middle, the live trunk stood 1.6-6.6
#: cm behind it and the hand arrived short (#627).
REACH_MARGIN_M = 0.015
#: A hand whose reach's posture, read as the robot stands with its arm out
#: at its pre-grasp, does not put the goal in reach takes another this many
#: times at most before it goes in (`Reach._look`, #627): the lean an arm
#: brings is the standing policy's own and no table predicts it -- over
#: random arms the CoM moved -2 to +5 cm, a fit to the arm's angles still
#: off 0.7-1.7 cm rms -- but the robot reads it, out, to 0.1-0.2 cm.
LOOKS = 2
#: A hand's force (`force`) is the WHOLE BODY's reading while the hand is at
#: work in one of these phases and the standing policy has the legs
#: (`standing`; `arms.SelfModel.felt`, #547), else its arm's alone
#: (`SelfModel.hand_force`). ⚠ ALONG A STRAIGHT ARM THE ARM IS BLIND: 20 N
#: at the grip read 5 pushing away and 2 away and down, where the whole
#: body read 19.7-20.2 every way of 14. ⚠ WALKING, A SOLE IS IN THE AIR,
#: which the balance has planted: a swinging leg's own torques read up to
#: 400 N (`humanoid.odometry`).
FELT_PHASES = ("approach", "hold", "pull", "blocked", "carry", "retreat")
#: WHAT THE ROBOT READS of a hand's force (`sensed_force`: `Body.hand_force`,
#: a procedure's `read`, its stop conditions, a verb's report) takes the
#: body's own motion in (`arms.Motion`, #631) while the hand goes its own way
#: -- swung out or home, on its line in or out -- never in these phases,
#: where it holds or is held and is read as still: pressing slowly on what it
#: holds, its motion is smaller than its estimate's noise. ⚠ THE BODY'S OWN
#: LOOPS NEVER READ IT (`force`: the reach's stop on a push, the ease, the
#: force move's loop): taken into `force` everywhere, the hands checks lost
#: 18 set-outs of 480 for 5 won on one pod (the cube 46 of 96 where 53, a
#: pull's give following the estimate's noise); kept out of these phases,
#: still 7 for 1, its approaches stopped on a push at other moments, and 3
#: drawers fell.
HELD_PHASES = ("hold", "pull", "blocked", "carry")
#: A hand on its way in STOPS where it is (`blocked`) once its own solve
#: misses its command by this, m -- the goal out of reach from the posture
#: the policy truly holds -- or once the force it reads at the hand
#: (`force`) passes this, N, pushing on something it was not sent to. ⚠
#: Unstopped, a crouch that sat further back than the plan's statue left a
#: chest's knob 10.7 cm out of reach, and the arm drove the gripper into
#: the counter's edge until it pushed the robot over backwards.
BLOCKED_M, BLOCKED_N = 0.03, 30.0
#: ...for this long, s: the force read off the drivers assumes the body is
#: still, and a deep bow's folding read 81.5 N at the hand for an instant
#: as its approach began. A push lasts; a transient does not.
BLOCKED_S = 0.15
#: A hand held whose force pushes on along its fingers past this, N, is
#: eased before its jaws shut (`ease`, #560): pressed on a cube's board at
#: 40-120 N, the jaws dragged across it and the feet swivelled 4-6 deg
#: (SimNotes, "Setting a table").
EASE_N = 10.0
#: The posture's own loop: the pelvis's height and the trunk's pitch as the
#: robot's model of itself has them (its legs' encoders, its IMU) held to
#: the plan's by trimming what the policy is asked -- this much of the
#: miss a second, at most this far, m and rad. Unheld, the policy stood 2
#: cm under the height it was asked.
TRIM_RATE, TRIM_MAX = 1.0, (0.05, 0.15)
#: A hand's approach starts once the posture the robot measures is this
#: near the plan's -- the pelvis's height, m, the trunk's pitch and the
#: waist's yaw, rad -- and its trunk is still -- its gyro under
#: `TRUNK_STILL_RATE`, rad/s -- or this long after its command arrived, s: a
#: full-stretch reach begun before the body had folded into a deep bow
#: stopped short. ⚠ The force read off the drivers assumes the body still
#: (`BLOCKED_N`): the floor's approach begun as its bow stopped turning
#: (0.59 rad/s) read 492 N at the hand for 0.2 s and stopped 6 cm short of
#: it; begun under 0.05, 62 N for an instant, and it held there (#520).
POSTURE_TOL, TRUNK_STILL_RATE, POSTURE_WAIT_S = (0.025, 0.06, 0.02), 0.05, 4.0
#: The waist turns to its posture's angle no faster than this, rad/s: the
#: trunk's pitch's ramp (`policy.PITCH_RATE`) -- and is set going and stopped
#: at this, rad/s^2 (`turned`), as training's scripted waist turns it
#: (`training/humanoid_train/stand.py`): the stand is served the turns it
#: trained on. (Set going at once, the stand that knew no waist turned its
#: soles no more, 60 deg with both arms out: 1.85 deg to 1.81.)
WAIST_RATE, WAIST_ACCEL = 0.6, 1.5
#: A hand that is out moves no faster than this when carried,
#: m/s: at the reach's own 0.2 a lid's pull outran the posture, which
#: comes a second or two behind its command, and the arm's solve missed
#: its command by 2-5 cm all the way up (#493).
HOLD_SPEED = 0.06
#: A path's slack (`path_for`): each of its points further from the
#: shoulder than its first is also reached this much further AHEAD, m: the
#: lean its load still brings. Lifting, the live trunk stood 1.5-2 cm behind
#: a statue balanced at the soles' middle all the way up a lid's arc and 2 cm
#: more at its top, where a path planned with no slack stopped at 57 deg
#: (#493); the statue stands as the robot reads itself standing now
#: (`Lean`, #627), and only this last 2 cm is to come.
LIVE_MARGIN_M = 0.02
#: ...and each is followed within this, m: a solve trades position for the
#: turn it holds (`arms.IK_ROT_WEIGHT`), and down a lid's arc, the wrist near
#: its stop, left points 1.1-1.2 cm off that the live hand passed -- which
#: stops only past `BLOCKED_M`.
PATH_POS_TOL = 0.02
#: A carried hand whose solve misses its command by this, m, or
#: its turn by this, rad, takes a posture that reaches it, planned at most
#: once this often, s (`posture_for` is a fraction of a second of the
#: processor's). ⚠ THE TURN TOO: a pull up a lid's arc standing held its
#: place to a millimetre and missed its turn by 0.19-0.2 rad, its wrist at
#: its stop pushing the hand 2 cm sideways, until it stopped (#493).
REPLAN_M, REPLAN_ROT, REPLAN_S = 0.015, 0.12, 1.0
#: A carried hand's grip may turn this far off its plan, rad: a held drop
#: handle swings on its own pin, so 10 deg of it moves the knob under a
#: centimetre, which the arm's give takes up -- held to a reach's 0.1, a
#: lid's arc reached within a millimetre failed on the wrist's last 7-14
#: deg (#493).
HOLD_ROT_TOL = 0.2
#: A carried hand has arrived once within this of its goal, m; one whose
#: command arrived and which is still short of it this long after, s, is
#: held back by something (`carry`'s report says how far short).
ARRIVE_M, HELD_BACK_S = 0.01, 1.0
#: A carried hand's solve that misses its command by this, m, has
#: diverged and is never commanded; and a move is given this long, s, to
#: begin before a miss can end it.
WILD_M, SETTLE_S = 0.06, 0.3
#: A PULL (`pull`, #525): a force move. The hardest it pushed and whether it
#: is held fast are read off the hand's reading along its way over this
#: long, s; a hand that has gone no further than this, m, in this long, s,
#: while it pushes with this share of its force -- or that is not going
#: (below) with its command as far ahead as it may go -- is held fast...
MEAN_S, FAST_M, FAST_S, FAST_SHARE = 0.1, 0.004, 0.6, 0.8
#: ...the force it is CLOSED on is that reading over this long, s, carried
#: this far ahead, s, at the rate it rises, read over this long, s: the move
#: slows before the force reaches its ask, not once it has. ⚠ Closed on
#: `MEAN_S`'s reading as it stood, it built over 100 N/s on a locked bar and
#: put 1.7 times its ask on it before it read the ask -- closed even on the
#: bar's true force (#547)...
ADMIT_SMOOTH_S, AHEAD_S, RISE_SMOOTH_S = 0.03, 0.1, 0.08
#: ...and it backs off past its force up to this many times its speed: no
#: faster than it walked in, the bar took 1.3 times a 10 N ask while a body
#: leaning on the held hand kept the force up, and 1.2 at twice (#547).
BACK_SPEED = 2.0
#: ...a hand GOES while it gets this far along its way, m, in this long, s
#: (and at that pace over `FAST_S`, or it is not held fast pinned). One that
#: does not -- HELD -- whose command is out of the arm's reach takes no other
#: posture where the arm reaches the hand itself: its command stays where
#: the arm last reached -- but backs off toward the hand past its force, its
#: arm's joints toward those that reach it (held still, a pull on a locked
#: bar stood at 21-24 N for 20 while its body leaned on it), and pinned
#: there pushing under `FAST_SHARE` of its force, it is out of reach, not
#: held fast (in free air, a pull past the arm's reach was "held fast" after
#: 1.1 cm, #547). ⚠ AGAINST A HAND HELD, THE COMMAND'S LEAD IS THE
#: SOLVE'S MISS: postures taken for it bowed, crouched and stood the body up
#: with the hand on a locked bar, 30-100 N on the bar for a pull asked
#: 10-20, and 7 ways in 14 never ended (#547). ⚠ BUT A HAND ITS ARM CANNOT
#: HOLD WHERE IT IS TAKES ONE: a lid's knob gripped 15-18 mm past the arm's
#: reach opened only from another posture, before the lid had moved; and
#: no lead tells the two apart -- moving lids took theirs with the command
#: up to 17 mm ahead, and on the bar a push took six with it under 15.
#: ⚠ ALONG ITS WAY, AND FASTER THAN A GRIP SLIDES: held on the bar, the
#: hand's sensed place moved 4 mm in a tenth of a second, and its grip slid
#: 3-7 mm a second, 10-12 as a push leaned the body in; but held fast at its
#: force off its way alone, a door swung by its hinge across the hand's way
#: was let go of...
GO_ON_M, GO_S = 0.015, 1.0
#: ...its command leads it by at most this along its way, m, whatever the
#: force asks, and moves no faster than this times its speed along its way
#: and across it: what runs away with the hand -- a lid falling toward the
#: robot across a way down -- the arm's spring holds back...
LEAD_MAX_M, ACROSS_SPEED = 0.08, 2.0
#: ...giving, its turn gives to what it holds this share a command (100 a
#: second: a tenth of a second's lag) -- held, a wrist twisted by a drop
#: handle's swing put 20-30 N across the pull at the grip and turned the
#: robot 22 deg on its feet...
GIVE_RATE = 0.1
#: ...the way it reports at its end is the hand's over this much of its last
#: travel, m, its sensed place read each time the hand has gone this far, m
#: (at 6 cm/s a hand goes 0.6 mm a command, about what its sensed place
#: scatters)...
TRAIL_M, STEP_M = 0.02, 0.006
#: ...and its arm holds at this share of its gains: the force is closed on
#: the hand's reading, whatever the stiffness.
PULL_STIFFNESS = 0.6
#: THE WAY HOME (`way_home`, #539; SimNotes, "The way home"): straight to
#: the rest in joint space -- unless that takes some part of the arm ahead
#: of the robot's front (`front`), where whatever the hand worked over is,
#: more than this much lower, m, than the lowest of it there now -- else
#: the SIDE WAY, where that dips less: rolled out to its side to at least
#: this much, rad, its other joints home with it held there, then lowered
#: at its side. ⚠ Straight, an arm came down onto a 0.72 m counter's edge
#: 0.28 m ahead from 25 of 30 places a hand works over it, and onto a 0.90
#: m one from 23 of 26; this way onto neither: held out this far its elbow
#: stays near its shoulder's height, and lowered at its side the arm
#: reaches 9 cm ahead.
HOME_DIP_M, HOME_ROLL = 0.01, 1.3
#: ...its roll in -- lowered at its side -- with its shoulder pitched this
#: much back, rad, then forward to its rest (#560). ⚠ Lowered in the hanging
#: arm's line, it dragged a jaw over a counter the robot stood turned to for
#: 10 s; on its model this far back clears one at the toes turned 10 deg.
HOME_BACK = 0.2
#: ...sampled every this much of its furthest joint's walk, rad.
HOME_SAMPLE_RAD = 0.05
#: ...and home is the rest commanded and its drivers reading every joint
#: within this of it, rad (they read to 0.0026).
HOME_RAD = 0.05
#: The way OUT is judged so from its other end (`way_out`, #560): straight, a
#: hanging hand at a table 13 cm out was driven into its top at 353-359 N.

#: The postures a reach may take, in the order they are tried -- the least
#: bent first: the pelvis's height off the stand's, m, and the trunk's
#: pitch, rad, inside what the standing policy trained on
#: (`policy.HEIGHT_RANGE`, `PITCH_RANGE`).
POSTURES = ((0.0, 0.0), (-0.08, 0.0), (-0.16, 0.0), (-0.16, 0.3), (-0.2, 0.55),
            (-0.22, 0.8), (-0.25, 0.95), (-0.25, 1.1))
#: The arms a plan's solve starts from, the LEFT side's (shoulder pitch,
#: roll, yaw, elbow, wrist pitch, wrist roll; the right mirrors roll and
#: yaw): hanging, held out level, out and down, up, down to the floor, and
#: hanging plumb from a trunk bowed forward.
SEEDS = ((0.0, 0.1, 0.0, 0.3, 0.0, 0.0), (-1.4, 0.1, 0.0, 0.3, 0.0, 0.0),
         (-0.9, 0.1, 0.0, 0.6, 0.6, 0.0), (-2.4, 0.1, 0.0, 0.3, 0.0, 0.0),
         (-0.5, 0.1, 0.0, 0.2, -0.3, 0.0), (0.8, 0.1, 0.0, 0.3, 0.3, 0.0),
         (1.1, 0.1, 0.0, 0.1, 0.0, 0.0))
_MIRRORED = (1, 2, 5)
#: ...and the postures a hand that is out may take while it is carried
#: (`carry`, `path_for`): a reach's, with a bow at full
#: height beside the stand (#493). Bowed 0.15-0.3 rad its shoulder comes
#: ~6 cm forward, and a lid's arc out of reach from the stand -- the knob
#: 0.5 m ahead and over the shoulder's height by 60 deg -- was in it.
HOLD_POSTURES = ((0.0, 0.0), (0.0, 0.15), (0.0, 0.3), (-0.08, 0.0), (-0.08, 0.15),
                 (-0.16, 0.0), (-0.16, 0.3), (-0.2, 0.55), (-0.22, 0.8), (-0.25, 0.95),
                 (-0.25, 1.1))
#: The waist's yaw a posture may take, rad, in the order tried at each of
#: the legs' (#560's stage (c3), as #620 measured them, `waist_spike.py`):
#: none, then either way in steps of 15 deg to its range's 60 -- and how bent
#: a turned one is: this much of a turn as bent as the legs' next posture
#: (#620's `bent`).
WAIST_YAWS = tuple(math.radians(a) for a in (0, 15, -15, 30, -30, 45, -45, 60, -60))
WAIST_BENT_RAD = math.radians(20)


def with_waist(legs, yaws=WAIST_YAWS) -> tuple[tuple[float, float, float], ...]:
  """Every posture of `legs` (height, pitch) at each of the waist's `yaws`,
  (height, pitch, yaw): every one with the waist square first, in `legs`'
  order, then the turned ones the least bent first -- a legs' posture's
  place in `legs` and its yaw over `WAIST_BENT_RAD`, ties in `legs`' order,
  then `yaws'`. ⚠ #620'S COST ALONE TURNED THE WAIST BEFORE IT BOWED
  DEEPER, and the served door and chest passed 23 and 14 of 32 draws from
  the tour where square first they passed 28 and 25 (today's body 31 and
  20; the stand that knew no waist, one pod): the waist reaches where no
  square posture does."""
  rows = [(w != 0, i + abs(w) / WAIST_BENT_RAD, i, k, (float(h), float(p), float(w)))
          for i, (h, p) in enumerate(legs) for k, w in enumerate(yaws)]
  return tuple(r[4] for r in sorted(rows, key=lambda r: r[:4]))


#: ...a reach's postures, and a carried hand's (`posture_for`, `path_for`).
REACH_POSTURES = with_waist(POSTURES)
CARRY_POSTURES = with_waist(HOLD_POSTURES)
#: However the robot reads itself standing, its shoulder stands within this
#: of the statue balanced over the soles' middle with its arms hanging, m: a
#: goal further out than that of its reach is skipped on that statue, kept,
#: before one is balanced as it stands (`Lean`; it stood them up to 6.6 cm
#: back and, an arm over the top, 4.4 cm of weight forward, #627).
LEAN_MOST_M = 0.10
#: A posture whose trunk above the waist comes this far into a thigh or a
#: shank stands nowhere, m: the robot's own drawing meets its own parts only
#: in the pairs it names, and none is the trunk's against a leg -- at the
#: hips' deepest bow a yaw of 15-30 deg put the torso's corner 1-2.5 cm into
#: a thigh (#620).
TRUNK_INTO_M = 0.01


def turned(at: np.ndarray, speed: np.ndarray, goal: np.ndarray, dt: float
           ) -> tuple[np.ndarray, np.ndarray]:
  """A command walked toward `goal` one step of `dt`: no faster than
  `WAIST_RATE`, its speed changed by no more than `WAIST_ACCEL` a second,
  slowing in time to stop there. (command, speed)."""
  to_go = goal - at
  # the fastest that can still stop there, a step's travel counted in
  brake = WAIST_ACCEL * dt * (np.sqrt(0.25 + 2.0 * np.abs(to_go) / (WAIST_ACCEL * dt * dt)) - 0.5)
  want = np.sign(to_go) * np.minimum(brake, WAIST_RATE)
  speed = speed + np.clip(want - speed, -WAIST_ACCEL * dt, WAIST_ACCEL * dt)
  step = speed * dt
  # ...there once a step toward it reaches it, or it is there and all but stopped
  # (never on a step away: a goal moved just behind a turning command would
  # stop it dead)
  there = (((np.sign(step) == np.sign(to_go)) & (np.abs(step) >= np.abs(to_go)))
           | ((np.abs(to_go) < 1e-9) & (np.abs(speed) <= WAIST_ACCEL * dt)))
  return np.where(there, goal, at + step), np.where(there, 0.0, speed)


def seed(side: str, s) -> np.ndarray:
  q = np.array(s, dtype=float)
  if side == "right":
    q[list(_MIRRORED)] *= -1
  return q


def _balance(sm: SelfModel, height: float, pitch: float, arms: dict | None, waist: float,
             com: float):
  """What a statue's legs zero (`static_legs`): of (the pelvis's place
  ahead, the hip's, the knee's and the ankle's pitch), the sole's bottom on
  the floor, the sole flat, and the centre of mass `com` ahead of it -- the
  pelvis `height` off the stand's, pitched `pitch`, its arms and waist so."""
  m, d = sm.m, sm.d
  root, sole = sm.root, sm.soles[0]
  adr = {n: sm.qadr[k] for k, n in enumerate(hm.JOINT_NAMES)}
  h = hm.stand_height() + height
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  stand = {**hm.stand_qpos(), **{j: float(waist) for j in hm.WAIST}, **(arms or {})}

  def resid(v):
    x, hip, knee, ankle = v
    d.qpos[:] = 0.0
    d.qpos[0], d.qpos[2] = x, h
    d.qpos[3:7] = quat
    for n, val in stand.items():
      d.qpos[adr[n]] = val
    for s in hm.SIDES:
      d.qpos[adr[f"{s}_hip_pitch"]] = hip
      d.qpos[adr[f"{s}_knee"]] = knee
      d.qpos[adr[f"{s}_ankle_pitch"]] = ankle
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    r = d.geom_xmat[sole].reshape(3, 3)
    bottom = d.geom_xpos[sole][2] - m.geom_size[sole][2]
    return [10 * bottom, 3 * r[0, 2], 10 * (d.subtree_com[root][0] - d.geom_xpos[sole][0] - com)]
  return resid


def static_legs(height: float, pitch: float, model: SelfModel | None = None,
                arms: dict | None = None, waist: float = 0.0, com: float = 0.0) -> dict:
  """The legs' sagittal joints holding the pelvis `height` off the stand's
  and the trunk pitched `pitch`, both soles flat and the centre of mass
  `com` ahead of their middle, m (over it, by default): a statue's answer
  -- with the arms where `arms` puts them (joint name -> angle), else
  hanging, and the waist turned `waist`. ⚠ An arm held out moves the centre
  of mass, and the policy moves its pelvis back for it: balanced with its
  arms hanging, the statue stood 3 cm in front of the robot reaching a lid's
  knob and 7 cm in front of it at the lid's top, where the plan's reach was
  out of the robot's (#493). ⚠ AND THE POLICY HOLDS ITS WEIGHT WHERE IT
  HOLDS IT: 1.2 cm behind the soles' middle at the stand, 2.9 with an arm
  out after a step (#627; `Lean`). Joint name -> angle; with its arms
  hanging, the model's own answer kept (`SelfModel.statues`)."""
  sm = model or SelfModel()
  # ...the one over the soles' middle kept: one at a lean read is that read's
  kept = arms is None and not com
  key = (float(height), float(pitch), float(waist))
  if kept and key in sm.statues:
    return dict(sm.statues[key][0])
  resid = _balance(sm, height, pitch, arms, waist, com)
  best = None
  guesses = ([0.0, -0.2 - pitch, 0.4, -0.2], [0.0, -0.8 - pitch, 1.6, -0.8])
  if not kept and key in sm.statues:
    # ...one weighed otherwise starts from the statue's own legs: a fit, not two
    legs = sm.statues[key][0]
    guesses = ([0.0, legs["left_hip_pitch"], legs["left_knee"], legs["left_ankle_pitch"]],)
  for guess in guesses:
    fit = least_squares(resid, guess, bounds=([-1.0, -3.0, 0.0, -0.9], [1.0, 2.2, 2.6, 0.6]))
    if best is None or fit.cost < best.cost:
      best = fit
  _, hip, knee, ankle = best.x
  legs = {f"{s}_{k}": v for s in hm.SIDES
          for k, v in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle))}
  if kept:
    sm.statues[key] = (dict(legs), None)
  return legs


def _posed(sm: SelfModel, height: float, pitch: float, q0: np.ndarray,
           arms: dict | None = None, waist: float = 0.0, com: float = 0.0,
           roll: float = 0.0) -> None:
  q = q0.copy()
  q[sm.waist] = waist
  for n, v in (arms or {}).items():
    q[hm.JOINT_NAMES.index(n)] = v
  for n, v in static_legs(height, pitch, sm, arms, waist, com).items():
    q[hm.JOINT_NAMES.index(n)] = v
  # gravity in the trunk's frame for a roll, then a pitch (`SelfModel.pose`)
  sm.pose(q, np.array([math.sin(pitch), -math.cos(pitch) * math.sin(roll),
                       -math.cos(pitch) * math.cos(roll)]))


@dataclass(frozen=True)
class Lean:
  """Where the robot reads itself standing (#627), off its model of itself
  posed off its senses (`Lean.read`): its centre of mass `com` ahead of its
  soles' middle, m, and its pelvis shifted `dx`, `dy`, m, and turned `turn`,
  rad, off the statue balanced there -- the stance frame's -- and its trunk
  rolled `roll`, rad, as its IMU reads it. A statue posed so (`statue`)
  stands where the live trunk stands. ⚠ THE STANDING POLICY
  HOLDS ITS WEIGHT WHERE IT HOLDS IT, AND NO TABLE PREDICTS IT: 1.2 cm behind
  the soles' middle at the stand and none in a deep bow, and an arm out
  moves it -2 to +5 cm by how the arm is bent, not by its weight alone (a
  natural arm and an arm over the top, the same weight, -0.8 and +4.4) --
  so a plan reads it, and reads it again with the arm out (`Reach._look`).
  The robot's own model read it to 0.05-0.2 cm of the truth at every
  posture, step and arm measured. The posture it was read in (`at`, as
  asked) is posed as the robot holds it (`held`): asked the deepest bow,
  the standing policy held 0.05-0.07 rad less of it, and the statue as
  asked stood a shoulder 1.1 cm ahead of the truth."""
  com: float = 0.0
  dx: float = 0.0
  dy: float = 0.0
  turn: float = 0.0
  roll: float = 0.0
  at: tuple | None = None
  held: tuple | None = None

  @staticmethod
  def read(live: SelfModel, posture, planner: SelfModel) -> "Lean":
    """The lean of `live`, posed off the robot's senses, asked `posture`
    (height, pitch, yaw): its centre of mass, the posture it holds as it
    measures it (its pelvis's height and pitch, its waist), and its pelvis
    off the statue of that posture balanced there with its arms as `live`
    has them (posed on `planner`)."""
    com = float(live.d.subtree_com[live.root][0])
    arms = {n: float(live.q[hm.JOINT_NAMES.index(n)])
            for s in hm.SIDES for n in (f"{s}_{j}" for j in hm.ARM)}
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    r = live.d.xmat[live.root].reshape(3, 3)
    held = (float(live.d.qpos[2]) - hm.stand_height(),
            math.asin(max(-1.0, min(1.0, -float(r[2, 0])))), float(live.q[live.waist][0]))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    _posed(planner, *held[:2], q0, arms, held[2], com, roll)
    (x0, y0), t0 = _pelvis(planner)
    (x1, y1), t1 = _pelvis(live)
    turn = math.remainder(t1 - t0, 2 * math.pi)
    c, s = math.cos(turn), math.sin(turn)
    return Lean(com, x1 - (c * x0 - s * y0), y1 - (s * x0 + c * y0), turn, roll,
                tuple(float(v) for v in posture), held)


def _pelvis(sm: SelfModel) -> tuple[tuple[float, float], float]:
  """The pelvis's place on the floor, stance frame, and its heading."""
  r = sm.d.xmat[sm.root].reshape(3, 3)
  return ((float(sm.d.xpos[sm.root][0]), float(sm.d.xpos[sm.root][1])),
          math.atan2(r[1, 0], r[0, 0]))


#: A statue balanced at a lean sooner (`statue(fast=True)`): its legs the
#: kept statue's moved by their own change as its weight's place moves,
#: worked out across this much of it, m, either way, and kept -- then this
#: many Newton steps on its balance from there.
SLOPE_M, FAST_STEPS = 0.02, 2


def _slope(sm: SelfModel, height: float, pitch: float, waist: float) -> np.ndarray:
  """How the statue's legs (hip, knee, ankle pitch) move as its centre of
  mass is balanced further ahead of the soles' middle, rad a metre; kept."""
  key = ("slope", height, pitch, waist)
  if key not in sm.statues:
    ends = [static_legs(height, pitch, sm, None, waist, c) for c in (SLOPE_M, -SLOPE_M)]
    sm.statues[key] = np.array([(ends[0][f"left_{j}"] - ends[1][f"left_{j}"]) / (2 * SLOPE_M)
                                for j in ("hip_pitch", "knee", "ankle_pitch")])
  return sm.statues[key]


def _posed_fast(sm: SelfModel, height: float, pitch: float, q0: np.ndarray,
                arms: dict | None, waist: float, com: float, roll: float) -> None:
  """`_posed`, its balance solved by Newton's method (`FAST_STEPS`) from the
  kept statue's legs moved by the lean, less what the arms' weight moves it
  ahead (`_slope`): the same balance, without a least squares' cost a
  posture -- a plan weighs dozens."""
  legs = static_legs(height, pitch, sm, None, waist)
  resid = _balance(sm, height, pitch, arms, waist, com)
  v = np.array([0.0, *(legs[f"left_{j}"] for j in ("hip_pitch", "knee", "ankle_pitch"))])
  # ...as far as the weight stands off `com` there, the arms set (the
  # residual's third row is ten times it)
  v[1:] -= resid(v)[2] / 10 * _slope(sm, height, pitch, waist)
  for _ in range(FAST_STEPS):
    r0 = np.array(resid(v))
    jac = np.empty((3, 3))
    for k in range(3):
      e = np.zeros(4)
      e[k + 1] = 1e-6
      jac[:, k] = (np.array(resid(v + e)) - r0) / 1e-6
    v[1:] -= np.linalg.solve(jac, r0)
  q = q0.copy()
  q[sm.waist] = waist
  for n, val in (arms or {}).items():
    q[hm.JOINT_NAMES.index(n)] = val
  for s in hm.SIDES:
    for k, j in enumerate(("hip_pitch", "knee", "ankle_pitch")):
      q[hm.JOINT_NAMES.index(f"{s}_{j}")] = v[k + 1]
  sm.pose(q, np.array([math.sin(pitch), -math.cos(pitch) * math.sin(roll),
                       -math.cos(pitch) * math.cos(roll)]))


def statue(sm: SelfModel, posture, q0: np.ndarray, arms: dict | None = None,
           lean: Lean | None = None, fast: bool = False) -> None:
  """`sm` posed as the robot would stand in `posture` (height, pitch, yaw):
  a statue's legs (`static_legs`), its arms where `arms` puts them (joint
  name -> angle; else hanging), balanced with its centre of mass where
  `lean` reads it and its pelvis shifted and turned as `lean` says (#627),
  the posture it was read in as it held it; with no lean, over the soles'
  middle. `fast`: its balance solved from the kept statue's (`_posed_fast`):
  within 0.01 mm of the least squares over random arms and leans of -3.5 to
  +5 cm at every carry posture, at 0.3 ms where it takes 12."""
  lean = lean or Lean()
  if lean.at is not None and np.allclose(posture, lean.at, atol=1e-9):
    # ...as it holds it, balanced afresh: a held posture's statue is a read's own
    posture, fast = lean.held, False
  height, pitch, waist = (float(v) for v in posture)
  if fast:
    _posed_fast(sm, height, pitch, q0, arms, waist, lean.com, lean.roll)
  else:
    _posed(sm, height, pitch, q0, arms or None, waist, lean.com, lean.roll)
  if lean.dx or lean.dy or lean.turn:
    d = sm.d
    c, s = math.cos(lean.turn), math.sin(lean.turn)
    x, y = float(d.qpos[0]), float(d.qpos[1])
    d.qpos[0], d.qpos[1] = c * x - s * y + lean.dx, s * x + c * y + lean.dy
    turn, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), lean.turn)
    mujoco.mju_mulQuat(out, turn, d.qpos[3:7].copy())
    d.qpos[3:7] = out
    mujoco.mj_kinematics(sm.m, d)
    mujoco.mj_comPos(sm.m, d)


def stands(sm: SelfModel, posture, q0: np.ndarray) -> bool:
  """Whether a posture (height, pitch, yaw) stands, its statue posed, its
  arms hanging (`_posed`): its trunk clear of its legs (`TRUNK_INTO_M`),
  kept with its statue (`SelfModel.statues`)."""
  height, pitch, waist = posture
  _posed(sm, height, pitch, q0, waist=waist)
  key = (float(height), float(pitch), float(waist))
  legs, ok = sm.statues[key]
  if ok is None:
    ok = trunk_clear(sm) > -TRUNK_INTO_M
    sm.statues[key] = (legs, ok)
  return bool(ok)


#: The trunk's parts above the waist, and the legs' they may fold into.
UPPER_GEOMS = ("torso",) + hm.HEAD_GEOMS
LEG_PARTS = tuple(f"{s}_{g}" for s in hm.SIDES for g in ("thigh", "shank"))


def trunk_clear(sm: SelfModel) -> float:
  """How far the trunk above the waist is from the thighs and the shanks as
  posed, m (negative: through them; `mj_geomDistance`, as #620 measured)."""
  m, d = sm.m, sm.d
  return min(float(mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, 0.1, None))
             for a in UPPER_GEOMS for b in LEG_PARTS)


def arm_reach(sm: SelfModel, side: str) -> float:
  """The furthest the grip gets from its shoulder's roll axis, m: every
  link of the arm hangs on the one line below it, so it is the arm hanging
  straight -- a bound no solve can beat, kept (`SelfModel.statues`), and
  measured on a pose of its own: ⚠ measured on the model as posed, the first
  in a process left it with straight legs, and the plan solved its first
  posture on that."""
  key = ("reach", side)
  if key not in sm.statues:
    d = mujoco.MjData(sm.m)
    mujoco.mj_kinematics(sm.m, d)
    shoulder = d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
    sm.statues[key] = float(np.linalg.norm(d.site_xpos[sm.grip[side]] - shoulder))
  return sm.statues[key]


def out_of_reach(sm: SelfModel, side: str, pos, further: float = 0.0) -> bool:
  """Whether `pos` is further from `side`'s shoulder than any solve could
  bring the grip within the reach's tolerance (`REACH_POS_TOL`), `further`
  on top, the model as posed: a plan skips the posture there, nothing lost."""
  shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
  return float(np.linalg.norm(np.asarray(pos) - shoulder)) + further >= (
    arm_reach(sm, side) + REACH_POS_TOL)


def _arm(side: str, q_arm) -> dict:
  """One arm's joints by name, for a statue to balance."""
  return {f"{side}_{j}": float(v) for j, v in zip(hm.ARM, q_arm)}


def solve_seeded(sm: SelfModel, side: str, pos, rot, start: np.ndarray | None = None
                 ) -> tuple[np.ndarray, float, float]:
  """`SelfModel.solve` from each of `SEEDS` (and `start`), the best kept:
  within tolerance and nearest `start` -- or, with none, nearest the arm
  hanging -- else the least off. ⚠ NEAREST, NEVER THE FIRST TO REACH: kept
  in its seeds' order, the hanging arm's own solve took the shoulder thrown
  back and the arm over the top where five other seeds reached the same
  point with the arm in front -- and an arm so bent leant the standing
  policy 4.4 cm forward where the other leant it 0.8 back (#627)."""
  tries = [seed(side, s) for s in SEEDS]
  if start is not None:
    tries.insert(0, np.asarray(start, dtype=float))
  near = tries[0]
  best, best_key = None, None
  for q0 in tries:
    sm.q[sm.arm[side]] = q0
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    ok = ep < REACH_POS_TOL and er < REACH_ROT_TOL
    key = (not ok, float(np.abs(q - near).max()) if ok else ep + 0.1 * er)
    if best_key is None or key < best_key:
      best, best_key = (q, ep, er), key
  return best


def _reaching(sm: SelfModel, side: str, pos, rot, start=None, seeded: bool = True):
  """The arm that puts the grip at `pos` (turned to `rot`) as `sm` is
  posed, within the reach's tolerance, nearest `start` (`solve_seeded`) --
  from `start` alone first where `seeded` is False -- or None."""
  if not seeded and start is not None:
    sm.q[sm.arm[side]] = np.asarray(start, dtype=float)
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    if ep < REACH_POS_TOL and er < REACH_ROT_TOL:
      return q
  q, ep, er = solve_seeded(sm, side, pos, rot, start)
  return q if ep < REACH_POS_TOL and er < REACH_ROT_TOL else None


def _goal_arm(sm: SelfModel, side: str, goal: HandGoal, slack: bool, start=None,
              seeded: bool = True):
  """The arm that puts the grip at `goal` as `sm` is posed -- and, with
  `slack`, at `REACH_MARGIN_M` further from its shoulder too -- nearest
  `start`; or None. ⚠ EACH POINT IS ALSO SOLVED FROM THE OTHER'S ARM: from
  the seeds alone, the two missed by 0.4-0.5 rad of turn at points 5 mm
  from points that reached, none of seven seeds converging, and the plan
  took a deeper bow there -- with no tag in view (#627, the table's cup)."""
  q_goal = _reaching(sm, side, goal.pos, goal.rot, start, seeded)
  if not slack:
    return q_goal
  shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
  out = np.asarray(goal.pos, dtype=float) - shoulder
  far = np.asarray(goal.pos, dtype=float) + REACH_MARGIN_M * out / np.linalg.norm(out)
  q_far = _reaching(sm, side, far, goal.rot, start if q_goal is None else q_goal,
                    seeded and q_goal is None)
  if q_far is not None and q_goal is None:
    q_goal = _reaching(sm, side, goal.pos, goal.rot, q_far, seeded=False)
  return q_goal if q_far is not None else None


def _reached(sm: SelfModel, side: str, posture, q0: np.ndarray, goal: HandGoal,
             pre: HandGoal, slack: bool, lean: "Lean | None", arms: dict | None, start):
  """(the arm at the pre-grasp, the arm at the goal) where `posture` puts
  `goal` in `side`'s reach -- with its slack -- on the robot as it would
  stand there (`statue`): with this arm where `arms` puts it (else hanging),
  then balanced with it where that solve put it and solved again from
  there, as an arm held out moves the trunk; or None."""
  others = dict(arms or {})
  statue(sm, posture, q0, others, lean, fast=True)
  if out_of_reach(sm, side, goal.pos, REACH_MARGIN_M if slack else 0.0):
    return None
  q = _goal_arm(sm, side, goal, slack, start)
  if q is None:
    return None
  statue(sm, posture, q0, {**others, **_arm(side, q)}, lean, fast=True)
  q = _goal_arm(sm, side, goal, slack, q, seeded=False)
  if q is None:
    return None
  q_pre = _reaching(sm, side, pre.pos, pre.rot, q, seeded=False)
  return None if q_pre is None else (q_pre, q)


def plan(side: str, goal: HandGoal, model: SelfModel | None = None,
         postures=REACH_POSTURES, approach: float = APPROACH_M, lean: "Lean | None" = None,
         arms: dict | None = None, start=None):
  """The least-bent posture from which `side`'s hand reaches `goal` -- and
  its pre-grasp, `approach` short of it -- WITH `REACH_MARGIN_M` to spare,
  else the least bent that reaches it at full stretch; and the arm's
  joints at both, nearest `start` (the arm as it is; else hanging). Tried
  on the robot's model of itself as it would stand in each posture
  (`statue`): a statue's legs and waist (`stands`), balanced where `lean`
  reads it (#627; with none, over the soles' middle), the other arm where
  `arms` puts it and this one where its solve does (`_reached`). Returns
  ((height, pitch, yaw), arm at the pre-grasp, arm at the goal, whether it
  has slack), or None if no posture reaches it."""
  sm = model or SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  pre = pregrasp(goal, approach)
  for slack in (True, False):
    for posture in postures:
      if not stands(sm, posture, q0) or out_of_reach(sm, side, goal.pos, -LEAN_MOST_M):
        continue
      got = _reached(sm, side, posture, q0, goal, pre, slack, lean, arms, start)
      if got is not None:
        return tuple(posture), got[0], got[1], slack
  return None


def pregrasp(goal: HandGoal, approach: float = APPROACH_M) -> HandGoal:
  """`approach` short of `goal`, back along the way its fingers point (the
  grip's -z); without an orientation, straight back toward the body."""
  if goal.rot is None:
    back = np.array([-1.0, 0.0, 0.0])
  else:
    back = np.asarray(goal.rot)[:, 2]
  return HandGoal(np.asarray(goal.pos) + approach * back, goal.rot, goal.stiffness)


#: An arm's solid parts, as a way home keeps them clear (`way_home`).
ARM_PARTS = hm.ARM_GEOMS + tuple(f"{j}_pad" for j in hm.JAWS)


def toes(sm: SelfModel) -> float:
  """How far ahead of the soles' middle its toes are, m: the soles' front as
  the robot's model of itself stands."""
  d, m = sm.d, sm.m
  return max(float(d.geom_xpos[g][0] + np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g])
             for g in sm.soles)


def outline(sm: SelfModel, side: str) -> np.ndarray:
  """The arm's solid parts as (how far ahead, how high), m, stance frame: a
  box's corners, a capsule's two ends with its radius ahead of them and
  under them."""
  m, d = sm.m, sm.d
  out = []
  for part in ARM_PARTS:
    g = m.geom(f"{side}_{part}").id
    pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      out += [(e[0] + size[0], e[2] - size[0])
              for e in (pos + size[1] * rot[:, 2], pos - size[1] * rot[:, 2])]
    else:
      corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * size
      out += [(p[0], p[2]) for p in pos + corners @ rot.T]
  return np.array(out)


def front(sm: SelfModel, side: str, rest) -> float:
  """How far ahead the robot's own front is, m: its toes, or its arm
  hanging at `rest` where that reaches further -- where every way home
  ends, so never a dip. ⚠ Posed as a reach's statue, the hanging grip is
  0.8 cm past the toes: judged from the toes alone, straight home tied the
  side way there and took it down onto the counter. Leaves the arm at
  rest."""
  sm._set_arm(side, rest)
  return max(toes(sm), float(outline(sm, side)[:, 0].max()))


def dip(sm: SelfModel, side: str, q0, ways, low: float, ahead: float) -> float:
  """How far under `low` any part of the arm goes further ahead than
  `ahead`, m, as it walks from `q0` through `ways` in joint space, every
  joint arriving together (`Reach._joints`)."""
  worst, prev = 0.0, np.asarray(q0, dtype=float)
  for w in ways:
    n = max(1, math.ceil(float(np.abs(w - prev).max()) / HOME_SAMPLE_RAD))
    for k in range(1, n + 1):
      sm._set_arm(side, prev + (w - prev) * (k / n))
      p = outline(sm, side)
      p = p[p[:, 0] > ahead]
      if len(p):
        worst = max(worst, low - float(p[:, 1].min()))
    prev = w
  return worst


def _side_way(side: str, q_out, q_home) -> list[np.ndarray]:
  """The side way from an arm at `q_out` to `q_home` in joint space: rolled
  out to its side (`HOME_ROLL`, or further where it is already), its other
  joints to `q_home`'s with its shoulder pitched `HOME_BACK` back, its roll
  then in, and its pitch forward -- the targets in turn."""
  k, p = hm.ARM.index("shoulder_roll"), hm.ARM.index("shoulder_pitch")
  out = seed(side, (0.0, HOME_ROLL, 0.0, 0.0, 0.0, 0.0))[k]
  raised, lowered, back = (np.asarray(q, dtype=float).copy() for q in (q_out, q_home, q_home))
  raised[k] = lowered[k] = raised[k] if abs(raised[k]) > HOME_ROLL else out
  lowered[p] = back[p] = back[p] + HOME_BACK
  return [raised, lowered, back, np.asarray(q_home, dtype=float).copy()]


#: ANOTHER ROBOT'S BODY, AS A HAND'S WAY KEEPS OFF IT (#678, a prototype):
#: a cylinder standing round the middle of its soles where `Body.others`
#: places it, this wide, m -- the standing body, its arms hanging, its
#: grippers 0.28 m out -- and this tall, its head's top...
OTHER_BODY_M, OTHER_TOP_M = 0.28, 1.55
#: The prototype's switch, for its measurement alone (`$PLUGGY_ARMS_KEEP_OFF`
#: 0: no body is kept off).
ARMS_KEEP_OFF = os.environ.get("PLUGGY_ARMS_KEEP_OFF", "1") != "0"
#: ...with this much room between it and every part of the arm, m.
OTHER_CLEAR_M = 0.05
#: The way drawn in: the arm's elbow bent this far, rad, before its shoulder
#: moves (its range ends at 2.10).
TUCK_ELBOW = 1.9


def arm_points(sm: SelfModel, side: str) -> tuple[np.ndarray, np.ndarray]:
  """The arm's solid parts as points (x, y, z), stance frame, each with the
  radius round it: a capsule's axis at five points, a box's corners and its
  middle."""
  m, d = sm.m, sm.d
  pts, rad = [], []
  for part in ARM_PARTS:
    g = m.geom(f"{side}_{part}").id
    pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      pts += [pos + f * size[1] * rot[:, 2] for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
      rad += [size[0]] * 5
    else:
      corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * size
      pts += list(pos + corners @ rot.T) + [pos.copy()]
      rad += [0.0] * 9
  return np.array(pts), np.array(rad)


def room(sm: SelfModel, side: str, q0, ways, keep_out) -> float:
  """The least room, m, between any part of the arm and any robot of
  `keep_out` ((x, y) of its soles' middle, stance frame: `OTHER_BODY_M`
  round, `OTHER_TOP_M` tall) as the arm walks from `q0` through `ways` in
  joint space (as `dip` walks it), its start included; inf with none."""
  if not len(keep_out):
    return math.inf
  centres = np.asarray(keep_out, dtype=float).reshape(-1, 2)
  worst, prev = math.inf, np.asarray(q0, dtype=float)
  for w in [prev] + list(ways):
    w = np.asarray(w, dtype=float)
    n = max(1, math.ceil(float(np.abs(w - prev).max()) / HOME_SAMPLE_RAD))
    for k in range(1, n + 1):
      sm._set_arm(side, prev + (w - prev) * (k / n))
      p, r = arm_points(sm, side)
      under = p[:, 2] - r < OTHER_TOP_M
      if under.any():
        gap = np.hypot(p[under, 0, None] - centres[:, 0], p[under, 1, None] - centres[:, 1])
        worst = min(worst, float((gap - r[under, None]).min()) - OTHER_BODY_M)
    prev = w
  return worst


def _same_way(a, b) -> bool:
  return len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))


def _tucked_way(side: str, q0, q1) -> list[np.ndarray]:
  """The way drawn in from `q0` to `q1` in joint space: its elbow bent to
  `TUCK_ELBOW` where it stands, its shoulder and wrist then to `q1`'s, the
  elbow held, then its elbow to `q1`'s."""
  e = hm.ARM.index("elbow")
  a, b = (np.asarray(q, dtype=float).copy() for q in (q0, q1))
  a[e] = b[e] = max(float(a[e]), TUCK_ELBOW)
  return [a, b, np.asarray(q1, dtype=float).copy()]


class Reach:
  #: (#678's prototype) none kept off where a test builds one round `__init__`
  keep_out: tuple = ()

  """The arms and grippers of one body beside its standing policy (module
  docstring). A hand's PHASE: `rest` (hanging), `move` (joint space to the
  pre-grasp, the posture walking to its plan), `approach` (a straight line
  in), `hold`, `pull` (the goal moved while holding), `blocked` (stopped on
  its way, `BLOCKED_M`), `carry` (a hand that is out carried on, `carry`),
  `joints` (an arm walked joint by joint, `set_joints`), `retreat` (a
  straight line out), `home` (joint space home, its own way: `way_home`)."""

  def __init__(self, model, data, stand: StandDriver, prefix: str = "",
               key: str = "humanoid"):
    self.m, self.d, self.stand = model, data, stand
    self.self_model = SelfModel()
    self.arms = {s: Arm(model, data, s, prefix) for s in hm.SIDES}
    self.grippers = {s: Gripper(model, data, s, prefix, key) for s in hm.SIDES}
    #: The waist, turned to the posture's angle (`posture`'s last), its
    #: command ramped there (`WAIST_RATE`).
    self.waist = Waist(model, data, prefix)
    self.waist_cmd = np.zeros(len(hm.WAIST))
    self.waist_speed = np.zeros(len(hm.WAIST))
    stand_q = hm.stand_qpos()
    self.rest = {s: np.array([stand_q.get(f"{s}_{j}", 0.0) for j in hm.ARM])
                 for s in hm.SIDES}
    self.phase = {s: "rest" for s in hm.SIDES}
    #: Each hand's goal, the pose its command is at on its line, the joint
    #: target its arm walks to in joint space, and what it last commanded.
    self.goal: dict[str, HandGoal | None] = {s: None for s in hm.SIDES}
    self.cmd: dict[str, HandGoal | None] = {s: None for s in hm.SIDES}
    self.q_to = {s: self.rest[s].copy() for s in hm.SIDES}
    self.q_cmd = {s: self.rest[s].copy() for s in hm.SIDES}
    #: Each arm's way home still to walk after `q_to` (`way_home`), and its
    #: way out (`way_out`); and every joint as its driver last read it
    #: (`home`), unread at first.
    self.homes: dict[str, list] = {s: [] for s in hm.SIDES}
    self.outs: dict[str, list] = {s: [] for s in hm.SIDES}
    self.sensed = np.full(len(hm.JOINT_NAMES), np.nan)
    #: How far the last solve left the model's grip off its command, m, and
    #: whether the reach's plan had slack (`plan`).
    self.miss = {s: 0.0 for s in hm.SIDES}
    self.slack = {s: True for s in hm.SIDES}
    #: How long each hand has been out of reach or pushing (`BLOCKED_S`).
    self._off = {s: 0.0 for s in hm.SIDES}
    self.every = max(1, round(EVERY_S / model.opt.timestep))
    self.steps = 0
    #: The posture the plan wants -- the pelvis's height off the stand's,
    #: the trunk's pitch, the waist's yaw -- and the trim on what the
    #: standing policy is asked (`TRIM_RATE`), its height and pitch.
    self.posture = np.zeros(2 + len(hm.WAIST))
    self.trim = np.zeros(2)
    self.tau = np.zeros(len(hm.JOINT_NAMES))
    self.qd = np.zeros(len(hm.JOINT_NAMES))
    #: The body's own motion as its senses have it, taken into its hands'
    #: force (`arms.Motion`, #631).
    self.motion = Motion(self.every, model.opt.timestep)
    #: The arm joints' and the waist's friction as calibrated and stored (`friction.FILE`),
    #: taken off the drivers' torques where a joint turns; and what every
    #: joint's may hold still (`friction.still`), which the whole body's
    #: reading weighs it by.
    self.friction = friction.load()
    sm = self.self_model
    self.still = friction.still(self.friction, sm.m.dof_frictionloss[sm.vadr])
    #: Whether the standing policy has the legs, both soles down: the body's
    #: owner says, every step (`HumanoidMission`); the whole body's reading
    #: assumes it, so until it is said the arm's alone is read.
    self.standing = False
    #: The force each hand pushes with, N, stance frame: the whole body's
    #: reading for a hand at work, standing, else its arm's (`FELT_PHASES`)
    #: -- and its arm's alone, whichever is read.
    self.force = {s: np.zeros(3) for s in hm.SIDES}
    self.arm_force = {s: np.zeros(3) for s in hm.SIDES}
    #: ...and what the robot reads of it (`HELD_PHASES`, #631).
    self.sensed_force = {s: np.zeros(3) for s in hm.SIDES}
    #: Where each grip was by its senses at the last command, stance frame
    #: (`_update`): what a stop condition reads (#554), so that reading it
    #: draws nothing more on the drivers' noise; None before the first.
    self.sensed_grip: dict[str, np.ndarray | None] = {s: None for s in hm.SIDES}
    #: How fast each hand's command goes along its line, m/s; a carry's
    #: state; and how its last carry ended (`report`).
    self.speed = {s: HAND_SPEED for s in hm.SIDES}
    self.pulls: dict[str, dict | None] = {s: None for s in hm.SIDES}
    self.report: dict[str, dict | None] = {s: None for s in hm.SIDES}
    #: Each reach's looks (`_look`, #627): the postures it may take, those it
    #: has left, how often it has looked, and whether its hand waits at its
    #: pre-grasp while its body takes the posture a look chose.
    self.allowed: dict[str, list] = {s: [] for s in hm.SIDES}
    self.left: dict[str, list] = {s: [] for s in hm.SIDES}
    self.looks = {s: 0 for s in hm.SIDES}
    self.looking = {s: False for s in hm.SIDES}
    self._planner: SelfModel | None = None
    #: The other robots' bodies a hand's way keeps off (#678, a prototype):
    #: each one's soles' middle (x, y), stance frame, set by the body before
    #: it asks a reach or a withdraw; and, by side, how the last way judged
    #: against them was refused (`kept_off`), else None.
    self.keep_out: list = []
    self.kept: dict[str, dict | None] = {s: None for s in hm.SIDES}

  # ---- what a caller asks ----

  def reach(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=REACH_POSTURES, flip: bool = True) -> bool:
    """A hand to `goal`: planned, then moved, approached and held. False
    (nothing moves) if no posture the policy holds puts it in reach. ⚠ A
    TWO-JAW GRIP IS THE SAME GRIP HALF A TURN ROUND ITS FINGERS (#493): where
    the one asked plans no reach, the other is planned -- the left hand
    reached a chest's knob with its jaws closing to its right in twice the
    places it did with them closing to its left, its wrist's roll at the
    end of its range. `postures` are the ones it may take, `flip` whether
    it may take the other grip."""
    plan_, goal = self._planned(side, goal, approach, postures, flip)
    self.__dict__.setdefault("kept", {s: None for s in hm.SIDES})[side] = None
    if plan_ is None:
      return False
    posture, q_pre, _, slack = plan_
    outs = self.way_out(side, q_pre, posture)
    if outs is None:
      # every way out crosses another robot's body (#678): nothing moves
      return False
    if not self._returns(side, q_pre, posture):
      # ...nor goes where every way home from would (#678): stranded there,
      # an arm's withdraw is refused and so is every walk's hand-over
      self.kept[side]["back"] = True
      return False
    self.slack[side], self.outs[side] = slack, outs
    self.hold_posture(*posture)
    self._off[side] = 0.0
    self.speed[side] = HAND_SPEED
    self.goal[side] = goal
    self.q_to[side] = self.outs[side].pop(0)
    self.allowed[side] = [tuple(float(v) for v in p_) for p_ in postures]
    self.left[side], self.looks[side], self.looking[side] = [], 0, False
    self.phase[side] = "move"
    return True

  def clear_way(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
                postures=REACH_POSTURES, flip: bool = True) -> bool | None:
    """Whether `reach` would plan `goal` for `side` and take a way out that
    keeps off every other robot's body (#678) -- None where it plans none;
    nothing moved, `kept` as it was."""
    plan_, _ = self._planned(side, goal, approach, postures, flip)
    if plan_ is None:
      return None
    kept = self.kept[side]
    clear = (self.way_out(side, plan_[1], plan_[0]) is not None
             and self._returns(side, plan_[1], plan_[0]))
    self.kept[side] = kept
    return clear

  def _returns(self, side: str, q_pre, posture) -> bool:
    """Whether some way home from the pre-grasp `q_pre` keeps off every
    other robot's body (#678), in `posture` and stood up; True with none
    by. False sets `kept`."""
    return not len(self.keep_out) or self._judged(
      side, np.asarray(q_pre, dtype=float).copy(), self.rest[side].copy(), out=False,
      poses=(posture, (0.0, 0.0, 0.0)), refuse=True) is not None

  def plans(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=REACH_POSTURES, flip: bool = True) -> bool:
    """Whether `reach` would plan `goal` for `side` -- nothing moved."""
    return self._planned(side, goal, approach, postures, flip)[0] is not None

  def _planned(self, side, goal, approach, postures, flip):
    # ...on the robot as it reads itself standing, the other arm where it is
    # and this one's answer nearest it as it is (#627)
    sm = self._planner_model()
    kw = {"lean": self._lean(), "arms": self._arms(but=side), "start": self.q_cmd[side]}
    plan_ = plan(side, goal, sm, approach=approach, postures=postures, **kw)
    if plan_ is None and goal.rot is not None and flip:
      flipped = np.asarray(goal.rot, dtype=float) @ np.diag([-1.0, -1.0, 1.0])
      goal = HandGoal(np.asarray(goal.pos, dtype=float), flipped, goal.stiffness)
      plan_ = plan(side, goal, sm, approach=approach, postures=postures, **kw)
    return plan_, goal

  def _lean(self) -> Lean | None:
    """How the robot reads itself standing now (`Lean.read`): its model of
    itself as its arms' loop last posed it -- never its joints read again,
    which would draw on their noise -- in the posture it was asked; None
    before its first reading."""
    if np.isnan(self.sensed).any():
      return None
    return Lean.read(self.self_model, self.posture, self._planner_model())

  def _arms(self, but: str | None = None) -> dict:
    """The arms where they are commanded, by joint name -- all but `but`'s."""
    return {k: v for s in hm.SIDES if s != but for k, v in _arm(s, self.q_cmd[s]).items()}

  def stop(self, side: str, why: str) -> None:
    """A carry ended from outside (its caller's patience), held where the
    hand IS, as its own stops are."""
    if self.phase[side] == "carry":
      self._end(side, why, *self.grip_now(side))

  def halt(self, side: str, why: str) -> bool:
    """A hand's move stopped now, held where it is (#554: a procedure's stop
    condition): a carry or a pull where the hand IS, as its own stops hold
    it (`stop`, its report saying `why`); a hand on its way in, or holding
    with its goal moved (`move_goal`), where its command is, as one stopped
    on its way is (`blocked`); an arm on its way out to its pre-grasp where
    its joints are commanded, and the body in the posture it has got to --
    out there, its goal and its command the grip they put it at, so it may
    be carried on or withdrawn as any hand that is out. ⚠ THE POSTURE TOO:
    held at its joints alone, an arm stopped 0.15 s into a reach rode the
    crouch its plan was still taking 17 cm down, and the next verb, starting
    from a command the body had left, stepped a joint 1.4 rad in one
    command. True where a move was under way; False, nothing changed, where
    none was."""
    phase = self.phase[side]
    if phase == "carry":
      self.stop(side, why)
      return True
    if phase in ("approach", "pull"):
      self.phase[side], self.looking[side] = "blocked", False
      return True
    if phase != "move":
      return False
    self.hold_posture(*self.posture_now())
    # the grip its commanded joints put it at, the body as last posed
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    sm._set_arm(side, self.q_cmd[side])
    pos, rot = sm.grip_pose(side)
    k = self.goal[side].stiffness
    self.goal[side], self.cmd[side] = HandGoal(pos.copy(), rot.copy(), k), HandGoal(pos, rot, k)
    self.q_to[side] = self.q_cmd[side].copy()
    self.outs[side] = []
    self.phase[side] = "blocked"
    return True

  def ease(self, side: str) -> bool:
    """A hand held where it is -- holding (`hold`) or stopped on its way
    (`blocked`, pressed hardest) -- whose force pushes on along its fingers
    past `EASE_N`: held where its senses put it, its turn kept, its command
    walked there along its line at `HAND_SPEED` (a setpoint is ramped), its
    arm's spring no longer pressing it on what it met. True where it was."""
    c, at = self.cmd[side], self.sensed_grip[side]
    if self.phase[side] not in ("hold", "blocked") or c is None or at is None:
      return False
    # ...its fingers' way as its loop last posed it: a fresh read of its
    # joints would draw on their noise
    rot = self.self_model.grip_pose(side)[1] if c.rot is None else np.asarray(c.rot, dtype=float)
    if float(self.force[side] @ -rot[:, 2]) <= EASE_N:
      return False
    self.goal[side] = HandGoal(at.copy(), None if c.rot is None else rot.copy(), c.stiffness)
    self.speed[side], self.phase[side] = HAND_SPEED, "hold"
    return True

  def out(self, side: str) -> bool:
    """Whether a hand is out at a pose of its own: holding it, moving on
    from it or stopped on its way (what `carry` moves on)."""
    return self.goal[side] is not None and self.phase[side] in (
      "hold", "pull", "blocked", "carry")

  def _planner_model(self) -> SelfModel:
    if self._planner is None:
      self._planner = SelfModel()
    return self._planner

  def posture_for(self, side: str, pos, rot=None, onward: bool = False,
                  past=()) -> tuple | None:
    """The least-bent posture (`CARRY_POSTURES`: the legs' and the waist's)
    from which `side`'s grip reaches `pos` -- turned to `rot`, or at any
    turn -- on the robot's model of itself as it would stand in each, its
    arms where they are (`statue`, as it reads itself standing: `Lean`), the
    arm starting where it is and from `SEEDS`; or None if none reaches it.
    `onward`: a held hand going on, whose arm missed `pos` in the posture it
    holds -- another posture, from the arm as it is alone, and none of
    `past`. ⚠ AS IT STANDS, NEVER A STATUE BALANCED AT ITS SOLES' MIDDLE: a
    carry re-planned on one took the posture it held again, and ended out of
    reach (#626, #627)."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    lean, arms = self._lean(), self._arms()
    starts = [self.q_cmd[side]] + ([] if onward else [seed(side, s_) for s_ in SEEDS])
    # the posture it holds first, where it still reaches: a change of
    # posture mid-move is a change of the arm's whole branch -- ⚠ and going
    # on, never the posture it holds, from a fresh seed: up a lid's arc the
    # arm stood at its elbow's stop 2 cm short of its way for 9 s, while
    # another branch of the same posture "reached" (#525)
    here = tuple(float(v) for v in self.posture)
    gone = {here} | {tuple(float(v) for v in p_) for p_ in past}
    order = ([p_ for p_ in CARRY_POSTURES if tuple(p_) not in gone] if onward
             else [here] + [p_ for p_ in CARRY_POSTURES if tuple(p_) != here])
    for posture in order:
      if not stands(sm, posture, q0) or out_of_reach(sm, side, pos, -LEAN_MOST_M):
        continue
      statue(sm, posture, q0, arms, lean, fast=True)
      if out_of_reach(sm, side, pos):
        continue
      for start in starts:
        sm._set_arm(side, start)
        _, ep, er = sm.solve(side, pos, rot, iters=80)
        if ep < REACH_POS_TOL and er < REACH_ROT_TOL:
          return tuple(posture)
    return None

  def path_for(self, side: str, poses, rot=None) -> tuple | None:
    """A posture (`CARRY_POSTURES`, the least bent first) from which `side`'s grip
    FOLLOWS `poses` (stance frame) in turn at the turn `rot` -- or at each
    pose its own, where `rot` is a list -- the first
    solved from the seeds, each next ON from the last, as a moving arm must
    -- or None: on the robot as it stands in the posture it holds (below),
    else as it would stand in another (`statue`, as it reads itself
    standing: `Lean`), its arms where they are and this one, point by point,
    where the last point put it. Each point further from its shoulder than
    the first is reached `LIVE_MARGIN_M` further ahead too: the lean its
    load still brings. ⚠ A point
    reached from a fresh seed is not a path: the arm went up a lid's arc in
    one branch, and where only another reached on, it swung between them
    and jerked the lid shut at 3.3 rad/s (#493)."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    poses = [np.asarray(p, dtype=float) for p in poses]
    rots = list(rot) if isinstance(rot, (list, tuple)) else [rot] * len(poses)
    ahead = np.array([LIVE_MARGIN_M, 0.0, 0.0])
    shoulder_id = sm.m.body(f"{side}_shoulder_roll_link").id
    # a hand that is out goes on from the arm as it IS, its posture first:
    # the branch its grip left it in is the branch its path must follow --
    # where the path starts at the hand (a reach planned while a hand is
    # out, stopped short, starts elsewhere)
    out = (self.out(side)
           and float(np.linalg.norm(poses[0] - self.grip_now(side)[0])) < PATH_POS_TOL)
    here = tuple(float(v) for v in self.posture)
    order = ([here] + [p_ for p_ in CARRY_POSTURES if tuple(p_) != here]) if out \
      else CARRY_POSTURES
    lean, arms = self._lean(), self._arms()
    for posture in order:
      # ⚠ ...AND THE ROBOT AS IT STANDS, its model posed off its senses, in
      # the posture it holds: up a lid's arc it stood 4 cm behind the
      # statue of that posture, which could not reach where its hand WAS
      live = out and tuple(posture) == here
      if live:
        # ...its arm as its encoders read it, the branch its grip is in,
        # never as commanded: a lid's weight held the hand 3.5 cm off its
        # command at the arc's top, from which the path down solved short
        sm.q = self.self_model.q.copy()
        sm.d.qpos[:] = self.self_model.d.qpos
        mujoco.mj_kinematics(sm.m, sm.d)
        q, ep, er = sm.q[sm.arm[side]].copy(), 0.0, 0.0
      else:
        # another posture than the one it holds: where it is held there,
        # from the arm as it is or any other way (the live solve carries the
        # arm there as the body moves)
        if not stands(sm, posture, q0) or out_of_reach(sm, side, poses[0], -LEAN_MOST_M):
          continue
        statue(sm, posture, q0, arms, lean, fast=True)
        if out_of_reach(sm, side, poses[0]):
          continue
        q, ep, er = solve_seeded(sm, side, poses[0], rots[0],
                                 self.q_cmd[side] if out else None)
      if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
        continue
      reach0 = float(np.linalg.norm(poses[0] - sm.d.xpos[shoulder_id]))
      for p, r_ in zip(poses[1:], rots[1:]):
        if not live:
          # ...balanced with its arm out where it is going (`static_legs`)
          statue(sm, posture, q0, {**arms, **_arm(side, q)}, lean, fast=True)
        further = float(np.linalg.norm(p - sm.d.xpos[shoulder_id])) > reach0
        target = p + ahead if further else p
        sm._set_arm(side, q)
        q, ep, er = sm.solve(side, target, r_, iters=80)
        if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
          break
      else:
        return tuple(posture)
    return None

  def carry(self, side: str, pos, stiffness: float | None = None,
            speed: float = HOLD_SPEED, path=None, turns=None) -> bool:
    """A hand that is out (`out`) carried from where it IS along a straight
    line to `pos`, stance frame, at `speed`, its turn kept, at `stiffness`
    (or its goal's), the posture the least bent that reaches the end --
    through `path`'s points first, from one posture that follows them all
    (`path_for`), turning to `turns`' at each (one a point of `path` and
    one at `pos`, None keeping the one before), where given. False (nothing moves) for a hand that is
    not out, or an end no posture reaches. How it ends is its `report`."""
    if not self.out(side):
      return False
    h, r = self.grip_now(side)
    # ⚠ THE TURN IT WAS ASKED TO KEEP, never the one it sagged to: planned
    # from a hand stretched up a lid's arc, the wrist's 0.1-0.15 rad of give
    # put the path back down out of every posture's reach (#493)
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r = np.asarray(self.cmd[side].rot, dtype=float)
    end = np.asarray(pos, dtype=float)
    ways = [np.asarray(p, dtype=float) for p in (path or ())]
    rots, last = [], r
    for t in (turns if turns is not None else [None] * (len(ways) + 1)):
      # a point with no turn keeps the one before: the hand's own, first
      last = last if t is None else np.asarray(t, dtype=float)
      rots.append(last)
    if len(rots) != len(ways) + 1:
      raise ValueError(f"a turn for each of the path's {len(ways)} points and its end, "
                       f"not {len(rots)}")
    if ways or turns is not None:
      # a PATH: one posture that follows it all, its points walked in turn
      posture = self.path_for(side, [h] + ways + [end], rot=[r] + rots)
    else:
      posture = self.posture_for(side, end, rot=r)
    if posture is None:
      self.report[side] = {"why": "out of reach", "shortM": float(np.linalg.norm(end - h)),
                           "moved": 0.0, "seconds": 0.0}
      return False
    k = self.goal[side].stiffness if stiffness is None else float(stiffness)
    moved_posture = tuple(posture) != tuple(float(v) for v in self.posture)
    self.hold_posture(*posture)
    self.cmd[side] = HandGoal(h.copy(), r.copy(), k)
    self.goal[side] = HandGoal(ways[0] if ways else end, rots[0].copy(), k)
    self.speed[side], self._off[side] = float(speed), 0.0
    self.pulls[side] = {"start": h.copy(), "t": 0.0,
                        "ways": list(zip(ways[1:] + [end], rots[1:])) if ways else [],
                        "fixed": bool(ways) or turns is not None}
    if moved_posture:
      # the hand holds still while the body takes the posture (`_held`)
      self.pulls[side]["postured"] = 0.0
    self.report[side] = None
    self.phase[side] = "carry"
    return True

  def pull(self, side: str, way, force: float, distance_m: float,
           speed: float = HOLD_SPEED, give: bool = True) -> bool:
    """A hand that is out PULLED along `way` (stance frame) -- or pushed: a
    FORCE MOVE (#525) -- with at most `force`, N, as the hand reads it,
    and at most `speed`, m/s. Across its way, GIVING, the hand yields to
    what pushes it, no faster than `ACROSS_SPEED` times its speed, and its
    turn gives to what it holds (`GIVE_RATE`) but about the jaws' own line;
    not giving, it holds its line and its turn -- hybrid force and
    position, the robot arm's old division. Until it has gone `distance_m` along its way ("distance"), it
    is held fast, a grip it began with is lost ("let go"), or it can reach
    no further. The posture is the one from which the hand reaches its
    way's end, taken first; a hand held takes no other (`GO_ON_M`).
    False (nothing moves) for a hand that is not out. How it ends is its
    `report`."""
    if force <= 0.0:
      raise ValueError(f"a pull pushes with some force, not {force} N")
    if not self.out(side):
      return False
    h, r = self.grip_now(side)
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r = np.asarray(self.cmd[side].rot, dtype=float)
    w = np.asarray(way, dtype=float)
    w = w / max(float(np.linalg.norm(w)), 1e-9)
    # the posture its way's end is reached from, the hand held still while
    # it comes: a posture changed partway jerks what the hand holds
    posture = self.path_for(side, [h, h + float(distance_m) * w], rot=r)
    moved = posture is not None and tuple(posture) != tuple(float(v) for v in self.posture)
    if moved:
      self.hold_posture(*posture)
    self.cmd[side] = HandGoal(h.copy(), r.copy(), PULL_STIFFNESS)
    self.goal[side] = HandGoal(h.copy(), r.copy(), PULL_STIFFNESS)
    self.speed[side], self._off[side] = float(speed), 0.0
    self.pulls[side] = {"start": h.copy(), "t": 0.0, "ways": [], "fixed": True, "peak": 0.0,
                        **({"postured": 0.0} if moved else {}),
                        "pull": {"way": w, "force": float(force), "distance": float(distance_m),
                                 "give": bool(give), "gripped": self.grippers[side].held(),
                                 "last": h.copy(), "went": 0.0, "trail": [h.copy()],
                                 "still": 0.0, "anchor": h.copy(), "pinned": 0.0,
                                 "past": [(0.0, 0.0)]}}
    self.report[side] = None
    self.phase[side] = "carry"
    return True

  def _look(self, side: str) -> bool:
    """Whether a hand at its pre-grasp goes in from the posture it holds
    (#627): as the robot reads itself standing now, its arm out (`Lean`),
    the least bent of the postures its reach may take that puts its goal in
    reach -- with its slack, else at full stretch, as `plan` chooses -- is
    the one it holds, or none is, or it has looked `LOOKS` times. Else that
    posture is taken, never one it has left this reach, and it looks again
    once there; its hand waits at its pre-grasp meanwhile (`looking`). ⚠
    EITHER WAY: the table's cup is reached by an arm over the top, which leant
    the policy 1.5 cm forward where a natural arm leans it 1.1 back, so its
    plan bows deeper than it need -- out of its tags' view."""
    if self.looks[side] >= LOOKS:
      return True
    self.looks[side] += 1
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    goal = self.goal[side]
    pre = pregrasp(goal)
    lean, arms = self._lean(), self._arms()
    here = tuple(float(v) for v in self.posture)
    gone = {tuple(float(v) for v in p_) for p_ in self.left[side]}
    chosen = None
    for slack in (True, False):
      for posture in (tuple(float(v) for v in p_) for p_ in self.allowed[side]):
        if (posture in gone or not stands(sm, posture, q0)
            or out_of_reach(sm, side, goal.pos, -LEAN_MOST_M)):
          continue
        if _reached(sm, side, posture, q0, goal, pre, slack, lean, arms,
                    self.q_cmd[side]) is not None:
          chosen = posture
          break
      if chosen is not None:
        break
    if chosen is None or chosen == here:
      return True
    self.left[side].append(here)
    self.hold_posture(*chosen)
    return False

  def _reaching(self, side: str, pos: np.ndarray, rot) -> np.ndarray | None:
    """The arm's joints that put the grip at `pos` and `rot` in the posture
    held -- on the robot's model of itself -- or None where none does."""
    q, ep, er = self.self_model.solve(side, pos, rot)
    return q if ep <= REPLAN_M and er <= REPLAN_ROT else None

  def hold_posture(self, height: float, pitch: float, waist: float = 0.0) -> None:
    """The posture to stand in, its legs' held by their own loop
    (`TRIM_RATE`), the waist turned to `waist`, rad, to the left positive."""
    self.posture = np.array([height, pitch, waist], dtype=float)
    self.stand.ask(*(self.posture[:2] + self.trim))

  def turn_waist(self, waist: float) -> None:
    """The posture's waist alone turned to `waist`, rad (its command walks
    there, `turned`): the friction's calibration (`friction`)."""
    self.posture[2] = float(waist)

  def waist_home(self) -> bool:
    """Whether the waist is home: its posture's angle zero, commanded there,
    and its driver reading it within `HOME_RAD` (a walk and a get-up hold it
    there: neither policy knows it) -- or, a fresh reach's before its first
    reading, commanded there alone: a body boots and stands up with it home."""
    sensed = self.sensed[self.self_model.waist]
    read = ~np.isnan(sensed)
    return bool(not self.posture[2:].any() and not self.waist_cmd.any()
                and (np.abs(sensed[read]) <= HOME_RAD).all())

  def hold_waist(self) -> None:
    """Once a physics step while the get-up has every other joint: the
    waist walked home to zero (`turned`) and held there stiff, nothing
    read -- no model of itself is posed while it tumbles."""
    if self.steps % self.every == 0:
      self.posture[2:] = 0.0
      self._turn_waist(self.every * self.m.opt.timestep, np.zeros(len(hm.WAIST)))
    self.steps += 1

  def _turn_waist(self, dt: float, feed: np.ndarray) -> None:
    """The waist's command one step toward the posture's angle (`turned`),
    sent."""
    self.waist_cmd, self.waist_speed = turned(self.waist_cmd, self.waist_speed,
                                              self.posture[2:], dt)
    self.waist.command(self.waist_cmd, feed)

  def held_pitch(self) -> float:
    """The trunk's pitch the posture is held at, rad, by the robot's own
    reckoning: what the standing policy is asked once it is shown it (what
    it is shown on its way there), less the trim that ask carries, which
    only takes up what the policy misses by -- none where the ask is held
    at the deepest bow the policy trained on (`PITCH_RANGE`): taken off
    there too, an 8 s stand's trim (1.3 deg) read the floor's held bow
    2.3 deg tipped, not 1.5. Exactly 0 at the stand. The fall watch reads
    a tip off it (`getup.FallWatch`, #520)."""
    shown = self.stand.goal if self.stand.settled else self.stand.posture
    plan = float(self.posture[1])
    carried = min(max(plan + float(self.trim[1]), PITCH_RANGE[0]), PITCH_RANGE[1]) - plan
    return float(shown[1]) - carried

  def set_joints(self, side: str, q) -> None:
    """The arm to joint angles `q` (its six, `hm.ARM` order) in joint space
    and held there: the motor level's (`Body.ramp_routine`), its hand's
    goal let go."""
    self.goal[side] = self.cmd[side] = None
    self.outs[side] = []
    self.q_to[side] = np.asarray(q, dtype=float).copy()
    self.looking[side] = False
    self.phase[side] = "joints"

  def stiffen(self, side: str, stiffness: float) -> None:
    """The stiffness the hand's commands carry from now (a share of
    `model.ARM_GAIN_SCALE`'s gains)."""
    if self.goal[side] is not None:
      self.goal[side].stiffness = stiffness

  def move_goal(self, side: str, pos) -> None:
    """While holding: the hand's goal moved (a pull is the goal drawn back
    with a low stiffness), its command walking there on its line -- a
    `pull`, which pushes on what it holds and so never stops for it."""
    self.goal[side].pos = np.asarray(pos, dtype=float)
    self.speed[side] = HAND_SPEED
    self.phase[side] = "pull"

  def retreat(self, side: str, approach: float = APPROACH_M) -> None:
    """Out along the line it came in on, then home its own way
    (`way_home`); the posture back to the stand. An arm on its way home
    already goes on."""
    if self.phase[side] in ("rest", "home", "retreat"):
      return
    if self.phase[side] in ("joints", "move") or self.cmd[side] is None:
      # an arm moved joint by joint (`set_joints`), or still walking out to
      # its pre-grasp, has no line to come out along: home from where its
      # command is (unhandled, a withdraw waited forever)
      self._go_home(side)
      return
    self.goal[side] = pregrasp(self.goal[side], approach)
    self.speed[side] = HAND_SPEED
    self.looking[side] = False
    self.phase[side] = "retreat"

  def way_home(self, side: str, refuse: bool = False) -> list[np.ndarray] | None:
    """The joint targets an arm walks home through from its command, judged
    on the robot's model of itself as it stands (`HOME_DIP_M`): its rest
    alone -- or, where that walk takes some part of the arm further ahead
    than the robot's front (`front`) and lower than the lowest of it there
    now, the side way: rolled out to its side (`HOME_ROLL`), its other
    joints home but its shoulder pitched back (`HOME_BACK`), its roll, then
    its pitch. The side way only where it dips less. Where another robot
    stands by (`keep_out`, #678), the way judged against its body as it
    stands and stood up (`_kept_off`): None where every way crosses it and
    `refuse`, else today's way."""
    return self._judged(side, self.q_cmd[side].copy(), self.rest[side].copy(), out=False,
                        poses=(None, (0.0, 0.0, 0.0)), refuse=refuse)

  def home_clear(self, side: str) -> bool:
    """Whether some way home from the arm's command keeps off every other
    robot's body (#678): False sets `kept`, nothing moved."""
    return self.way_home(side, refuse=True) is not None

  def way_out(self, side: str, q_to, posture=None) -> list[np.ndarray] | None:
    """The joint targets an arm walks out through from its command to `q_to`,
    a reach's pre-grasp: the way home's judgment from its other end --
    `q_to` alone, or, from an arm with none of it ahead of the robot's front
    whose walk there takes some part of it further ahead and lower than the
    lowest of it there at `q_to`, the side way home walked back. The side
    way only where it dips less. Judged as the robot stands, never in the
    posture the reach then takes: of 151 reaches planned in a crouch or a
    bow, 10 judged otherwise there, each straight where standing it was the
    side way (#560). Where another robot stands by (`keep_out`, #678),
    judged against its body as it stands and in `posture` (`_kept_off`):
    None where every way crosses it."""
    return self._judged(side, self.q_cmd[side].copy(), np.asarray(q_to, dtype=float).copy(),
                        out=True, poses=(None, posture), refuse=True)

  def _judged(self, side: str, q0, q1, out: bool, poses=(None,),
              refuse: bool = False) -> list[np.ndarray] | None:
    """The joint targets from `q0` to `q1` on the robot's model of itself
    as it stands (`HOME_DIP_M`): `q1` alone, or the side way (`_side_way`,
    walked back going `out`) where straight takes some part of the arm
    further ahead than the robot's front (`front`) and lower than the
    lowest of it there at the end over the work -- `q0` going home, `q1`
    going out -- and the side way dips less. With another robot by, the
    first way of today's, straight, the side way and the way drawn in that
    keeps off it (`_kept_off`)."""
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    pick, dips = self._picked(sm, side, q0, q1, out)
    if not len(self.keep_out):
      return pick
    return self._kept_off(sm, side, q0, q1, out, pick, dips, poses, refuse)

  def _picked(self, sm, side: str, q0, q1, out: bool):
    """`_judged`'s way with nobody by, and how far a way dips (m; 0 where
    nothing it works over is ahead)."""
    ahead = front(sm, side, self.rest[side])
    if out:
      sm._set_arm(side, q0)
      if (outline(sm, side)[:, 0] > ahead).any():
        # a hand already out sweeps nothing aside on its way to the next
        return [q1], lambda w: 0.0
    sm._set_arm(side, q1 if out else q0)
    p = outline(sm, side)
    there = p[p[:, 0] > ahead]
    # ⚠ none of it ahead worked over nothing there: judged as if all under
    # it were solid, a drawer pulled close took the side way, which swept a
    # jaw into its face and pushed it 4 cm back in
    if not len(there):
      return [q1], lambda w: 0.0
    low = float(there[:, 1].min())

    def dips(w):
      return dip(sm, side, q0, w, low, ahead)
    straight = dips([q1])
    if straight <= HOME_DIP_M:
      return [q1], dips
    aside = _side_way(side, q1, q0)[::-1][1:] + [q1] if out else _side_way(side, q0, q1)
    return (aside if dips(aside) < straight else [q1]), dips

  def _kept_off(self, sm, side, q0, q1, out, pick, dips, poses, refuse):
    """A HAND'S WAY KEEPS OFF ANOTHER ROBOT'S BODY (#678, a prototype):
    the first of today's way (`pick`), straight, the side way and the way
    drawn in (`_tucked_way`) whose arm keeps `OTHER_CLEAR_M` off every body
    of `keep_out` (`room`) -- an arm begun nearer only going no nearer -- in
    each of `poses` (None as the robot stands, else a posture's statue),
    and dips no deeper over the work than `pick`; else None where `refuse`,
    `kept` saying how, or `pick`."""
    straight = [np.asarray(q1, dtype=float).copy()]
    aside = _side_way(side, q1, q0)[::-1][1:] + [q1] if out else _side_way(side, q0, q1)
    ways = [pick] + [w for w in (straight, aside, _tucked_way(side, q0, q1))
                     if not _same_way(w, pick)]
    deepest = max(HOME_DIP_M, dips(pick))
    best = -math.inf
    for i, w in enumerate(ways):
      self._pose_planner(sm, None)
      if i and dips(w) > deepest + 1e-9:
        continue
      worst = math.inf
      for pose in poses:
        self._pose_planner(sm, pose)
        start = room(sm, side, q0, [], self.keep_out)
        worst = min(worst, room(sm, side, q0, w, self.keep_out) - min(OTHER_CLEAR_M, start))
      best = max(best, worst)
      if worst >= 0.0:
        self.kept[side] = None
        self.__dict__.setdefault("ways_log", []).append(
          (round(float(self.d.time), 2), side, "out" if out else "home", i, len(ways)))
        return w
    if not refuse:
      return pick
    c = np.asarray(self.keep_out, dtype=float).reshape(-1, 2)
    k = int(np.argmin(np.hypot(c[:, 0], c[:, 1])))
    self.__dict__.setdefault("ways_log", []).append(
      (round(float(self.d.time), 2), side, "out" if out else "home", -1, len(ways)))
    self.kept[side] = {"home": not out, "k": k, "m": round(float(np.hypot(*c[k])), 2),
                       "x": round(float(c[k, 0]), 2), "y": round(float(c[k, 1]), 2),
                       "short": round(-best, 3), "ways": len(ways)}
    return None

  def _pose_planner(self, sm, pose) -> None:
    """`sm` as the robot stands (None), or in posture `pose`'s statue."""
    if pose is None:
      sm.q = self.self_model.q.copy()
      sm.d.qpos[:] = self.self_model.d.qpos
      mujoco.mj_kinematics(sm.m, sm.d)
      return
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    statue(sm, tuple(float(v) for v in pose), q0, arms=self._arms(), lean=self._lean(),
           fast=True)

  def _go_home(self, side: str) -> None:
    """The arm on its way home (`way_home`), its goal let go; the posture
    back to the stand once no hand is out."""
    self.goal[side] = self.cmd[side] = None
    self.outs[side] = []
    self.looking[side] = False
    self.homes[side] = self.way_home(side)
    self.q_to[side] = self.homes[side].pop(0)
    self.phase[side] = "home"
    if all(p in ("rest", "home") for p in self.phase.values()):
      self.hold_posture(0.0, 0.0)

  def home(self, side: str) -> bool:
    """Whether an arm is home: its rest commanded, and every joint of it
    read there by its driver (`HOME_RAD`)."""
    return self.phase[side] == "rest" and bool(
      np.abs(self.sensed[self.self_model.arm[side]] - self.rest[side]).max() <= HOME_RAD)

  def done(self, side: str) -> bool:
    """Whether a hand has arrived where it was sent (or hangs), or stopped
    on its way (`blocked`), or an arm sent in joint space is there."""
    if self.phase[side] == "joints":
      return bool(np.allclose(self.q_cmd[side], self.q_to[side]))
    return self.phase[side] in ("hold", "rest", "blocked")

  # ---- the loop ----

  def _pose(self, arms: dict | None = None) -> None:
    q, qd, tau = self.stand.joints.read(self.d)
    self.tau, self.qd, self.sensed = tau, qd, q.copy()
    if arms is not None:
      for s in hm.SIDES:
        q[self.self_model.arm[s]] = arms[s]
    self.self_model.pose(q, self.stand.trunk.gravity())

  def grip_now(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Where the robot believes its grip is, stance frame: its model posed
    from its senses."""
    self._pose()
    return self.self_model.grip_pose(side)

  def _posture_there(self, dt: float) -> bool:
    """Whether the body has reached its plan's posture (`POSTURE_TOL`) and
    stopped there (`TRUNK_STILL_RATE`), or has been given `POSTURE_WAIT_S` to."""
    self._posture_wait += dt
    near = (np.abs(self.posture_now() - self.posture) <= POSTURE_TOL).all()
    still = float(np.linalg.norm(self.stand.trunk.rate)) < TRUNK_STILL_RATE
    if (near and still) or self._posture_wait >= POSTURE_WAIT_S:
      self._posture_wait = 0.0
      return True
    return False

  _posture_wait = 0.0

  def _posture_held(self) -> bool:
    """Whether the body stands still in the posture it holds (`POSTURE_TOL`,
    `TRUNK_STILL_RATE`)."""
    near = (np.abs(self.posture_now() - self.posture) <= POSTURE_TOL).all()
    return bool(near and float(np.linalg.norm(self.stand.trunk.rate)) < TRUNK_STILL_RATE)

  def _line(self, side: str, dt: float) -> bool:
    """The command one step along its line to the goal; True once there."""
    g, c = self.goal[side], self.cmd[side]
    step = g.pos - c.pos
    n = float(np.linalg.norm(step))
    v = self.speed[side]
    c.pos = g.pos.copy() if n <= v * dt else c.pos + step * (v * dt / n)
    if g.rot is not None:
      e = _rot_error(g.rot, c.rot)
      a = float(np.linalg.norm(e))
      if a <= HAND_TURN * dt:
        c.rot = np.array(g.rot, dtype=float)
      else:
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, e / a, HAND_TURN * dt)
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        c.rot = r.reshape(3, 3) @ c.rot
    c.stiffness = g.stiffness
    return n <= v * dt

  def _joints(self, side: str, dt: float) -> bool:
    """The arm one step toward `q_to` in joint space; True once there."""
    gap = self.q_to[side] - self.q_cmd[side]
    worst = float(np.abs(gap).max())
    if worst <= JOINT_RATE * dt:
      self.q_cmd[side] = self.q_to[side].copy()
      return True
    self.q_cmd[side] = self.q_cmd[side] + gap * (JOINT_RATE * dt / worst)
    return False

  def step(self) -> None:
    for g in self.grippers.values():
      g.sample()
    self.motion.sample(self.steps, self.stand, self.d)
    if self.steps % self.every == 0:
      self._update(self.every * self.m.opt.timestep)
    self.steps += 1

  #: What a restart carries of a reach (#559, `HumanoidMission.kept_state`):
  #: every hand's phase and commands, its filters and what it last read.
  KEPT = ("phase", "goal", "cmd", "q_to", "q_cmd", "homes", "outs", "sensed", "miss", "slack",
          "_off", "steps", "posture", "trim", "tau", "qd", "standing", "force",
          "arm_force", "sensed_grip", "speed", "pulls", "report", "waist_cmd", "waist_speed",
          "allowed", "left", "looks", "looking")

  def kept_state(self) -> dict:
    return {"reach": {k: kept_value(getattr(self, k)) for k in self.KEPT},
            "grippers": {s: {"shut": g.shut_asked, "force": g.force, "gap": g.gap,
                             "gapMean": g.gap_mean, "squeeze": g.squeeze,
                             "rng": g.sense.rng.bit_generator.state}
                         for s, g in self.grippers.items()},
            "motion": self.motion.kept_state(),
            # ...and its model of itself as its loop last posed it: what a
            # spot is placed through between two of the loop's ticks
            "selfModel": self.self_model.kept_state()}

  def restore_kept(self, state: dict) -> None:
    for k, v in state["reach"].items():
      setattr(self, k, value_back(v))
    self.motion.restore_kept(state.get("motion", {}))
    if "selfModel" in state:                  # a save from before it: posed at the next tick
      self.self_model.restore_kept(state["selfModel"])
    self.sensed_force = {s: np.array(f, dtype=float) for s, f in self.force.items()}
    for s, kept in state["grippers"].items():
      g = self.grippers[s]
      g.shut_asked, g.force = bool(kept["shut"]), float(kept["force"])
      g.gap, g.squeeze = float(kept["gap"]), float(kept["squeeze"])
      g.gap_mean = float(kept.get("gapMean", kept["gap"]))     # a save from before it
      g.sense.rng.bit_generator.state = kept["rng"]

  def posture_now(self) -> np.ndarray:
    """The posture as the robot's model of itself has it: the pelvis's
    height off the stand's (its legs' encoders), the trunk's pitch (its
    IMU) and the waist's yaw (its driver), as last posed."""
    sm = self.self_model
    return np.array([sm.d.qpos[2] - hm.stand_height(),
                     math.asin(max(-1.0, min(1.0, float(self.stand.trunk.gravity()[0])))),
                     *sm.q[sm.waist]])

  def _trim(self, dt: float) -> None:
    """The posture as the robot's model has it against the plan's; the
    policy asked that much more, once the posture it was shown has arrived
    (a ramp's lag is not a miss), never past what it trained on. ⚠ IT TRIMS
    ON WHILE THE WALKER HOLDS THE LEGS, off the walker's posture: a walk
    leaves the standing goal pitched forward, 0.008-0.017 rad after a short
    step and 0.020 after a walk of a second or more (the edge of the band
    `settled` reads), taken up again a second or two after the hand-back.
    Trimmed only while the standing policy steps, our table's reference
    laid 20 of 32 from the tour where it lays 27 (#668): every hand job is
    measured on it as it is."""
    have = self.posture_now()[:2]
    if self.stand.settled:
      trim = np.clip(self.trim + TRIM_RATE * (self.posture[:2] - have) * dt,
                     -np.array(TRIM_MAX), TRIM_MAX)
      asked = self.posture[:2] + trim
      inside = (asked >= [HEIGHT_RANGE[0], PITCH_RANGE[0]]) & (asked <= [HEIGHT_RANGE[1], PITCH_RANGE[1]])
      self.trim = np.where(inside, trim, self.trim)
      self.stand.ask(*(self.posture[:2] + self.trim))

  # ---- a hand that is out, carried (#493) ----

  def _end(self, side: str, why: str, h: np.ndarray, r: np.ndarray,
           at_goal: bool = False) -> None:
    """A carry over: held at its goal (one that arrived) or
    where the hand IS -- never at a command the arm could not follow -- and
    its `report`: why, how long, how far it went and where it ended up
    from where it began (stance frame, m), and the hardest it read along
    its way where it watched its force (N)."""
    st = self.pulls[side]
    c = self.cmd[side]
    short = float(np.linalg.norm(self.goal[side].pos - h))
    if not at_goal:
      # where the hand IS, at the turn it was ASKED (`carry`): held at the
      # turn it sagged to, a pull at a time crept the drop handle round
      turn = (np.asarray(c.rot, dtype=float) if c.rot is not None else r).copy()
      self.goal[side] = HandGoal(h.copy(), turn.copy(), c.stiffness)
      self.cmd[side] = HandGoal(h.copy(), turn, c.stiffness)
    disp = h - st["start"]
    rep = {"why": why, "seconds": round(st["t"], 3), "moved": round(float(np.linalg.norm(disp)), 4),
           "shortM": round(short, 4), "dx": round(float(disp[0]), 4),
           "dy": round(float(disp[1]), 4), "dz": round(float(disp[2]), 4)}
    if st.get("pull") is not None:
      pv = st["pull"]
      rep["force"] = round(st["peak"], 2)
      rep["went"] = round(pv["went"] + float((h - pv["last"]) @ pv["way"]), 4)
      # the way the hand was going at its end: its last `TRAIL_M` of travel
      went = h - pv["trail"][0]
      n = float(np.linalg.norm(went))
      rep["way"] = [round(float(v), 4) for v in (went / n if n > STEP_M else pv["way"])]
    self.report[side] = rep
    self.pulls[side] = None
    self.phase[side] = "hold"

  def _held(self, side: str, dt: float, h: np.ndarray, r: np.ndarray) -> float:
    """One command of a carried hand (`carry`), `h` and `r` where its grip
    IS by its senses; its stiffness. Its solve is
    commanded as the reach's always is, near as the arm comes -- ⚠ BUT NEVER
    ONE THAT DIVERGED (`WILD_M`: a miss of 10-15 cm commanded swung an arm
    round a chest's knob and out of it) -- and a miss past `BLOCKED_M` held
    `BLOCKED_S` ends the move where the hand is (`out of reach`), once the
    move has had `SETTLE_S` to begin: a carry stopped at the arm's stretch
    starts the next where its solve cannot quite put it."""
    st = self.pulls[side]
    st["t"] += dt
    # ⚠ A HAND HOLDS STILL WHILE ITS BODY TAKES A NEW POSTURE: carried on
    # meanwhile, a lid was jerked down at 1.7-2.3 rad/s and up to 81 deg
    settling = (st.get("postured") is not None and not self._posture_held()
                and st["t"] - st["postured"] < POSTURE_WAIT_S)
    there = False if settling else self._line(side, dt)
    c, sm = self.cmd[side], self.self_model
    q, ep, er = sm.solve(side, c.pos, c.rot)
    if ((ep > REPLAN_M or er > REPLAN_ROT) and not st.get("fixed")
        and st["t"] - st.get("planned", -REPLAN_S) >= REPLAN_S):
      # The arm runs out of reach from this posture: the least bent that
      # reaches the command now, while there is one (`posture_for`).
      st["planned"] = st["t"]
      posture = self.posture_for(side, c.pos, rot=c.rot)
      if posture is not None and tuple(posture) != tuple(self.posture):
        self.hold_posture(*posture)
        st["postured"] = st["t"]
    if ep < WILD_M:
      self.q_cmd[side], self.miss[side] = q, ep
    if ep > BLOCKED_M or er > HOLD_ROT_TOL:
      # ...given a posture it took the time to come before it is out of reach
      coming = (st["t"] < SETTLE_S
                or st["t"] - st.get("postured", -POSTURE_WAIT_S) < POSTURE_WAIT_S)
      self._off[side] = 0.0 if coming else self._off[side] + dt
      if self._off[side] >= BLOCKED_S:
        self._end(side, "out of reach", h, r)
      return c.stiffness
    self._off[side] = 0.0
    if there and self.phase[side] == "carry" and st.get("ways"):
      self.goal[side].pos, self.goal[side].rot = st["ways"].pop(0)
      return c.stiffness
    if there and self.phase[side] == "carry":
      # arrived once the HAND is there, never its command alone: on a soft
      # arm a lid held the hand 5 cm back while every command arrived (#493)
      short = float(np.linalg.norm(self.goal[side].pos - h))
      st["there"] = st.get("there", 0.0) + dt
      if short <= ARRIVE_M:
        self._end(side, "arrived", h, r, at_goal=True)
      elif st["there"] >= HELD_BACK_S:
        self._end(side, "held back", h, r, at_goal=True)
    return c.stiffness

  def _pulled(self, side: str, dt: float, h: np.ndarray, r: np.ndarray) -> float:
    """One command of a PULL (`pull`), `h` and `r` where its grip IS by its
    senses: the command walked along the way as the force allows, following
    the hand across it, no faster than its speed, its turn giving; then
    solved and commanded as a carry's is. Its stiffness."""
    st = self.pulls[side]
    pv = st["pull"]
    c, sm = self.cmd[side], self.self_model
    st["t"] += dt
    w, limit = pv["way"], pv["force"]
    if pv["gripped"] and not self.grippers[side].held():
      # A GRIP LOST ENDS THE PULL where the hand is: else it pulls on through
      # the air to its distance, and its report is that free motion's
      self._end(side, "let go", h, r)
      return c.stiffness
    dh = h - pv["last"]
    if float(np.linalg.norm(dh)) >= STEP_M:
      pv["went"] += float(dh @ w)
      pv["last"] = h.copy()
      pv["trail"] = (pv["trail"] + [h.copy()])[-1 - int(round(TRAIL_M / STEP_M)):]
    # held, or GOING along its way (`GO_ON_M` in `GO_S`, or at that pace over
    # `FAST_S`); off its place (`FAST_M`), not held fast at its force
    gone = float((h - st["start"]) @ w)
    pv["past"] = [p for p in pv["past"] if st["t"] - p[0] <= GO_S] + [(st["t"], gone)]
    going = gone - min(p[1] for p in pv["past"]) >= GO_ON_M
    pace = gone - min(p[1] for p in pv["past"] if st["t"] - p[0] <= FAST_S) >= (
      GO_ON_M * FAST_S / GO_S)
    moved = float(np.linalg.norm(h - pv["anchor"])) > FAST_M
    if moved:
      pv["anchor"], pv["still"] = h.copy(), 0.0
    settling = (st.get("postured") is not None and not self._posture_held()
                and st["t"] - st["postured"] < POSTURE_WAIT_S)
    # THE FORCE, CLOSED ON WHAT THE HAND FEELS (admittance): along its way the
    # command walks at its speed while the force the hand pushes with along
    # it, a moment ahead (`AHEAD_S`), is under half its limit, slows to a
    # stand as it reaches it, and backs off past it (`BACK_SPEED`) -- through
    # the arm's own springs, its model put 15 N on a drawer's stop for a 5 N
    # limit and 27 for 10, where the hand's reading was within 0.5 N of the
    # truth
    k, km, kr = (min(1.0, dt / s_) for s_ in (ADMIT_SMOOTH_S, MEAN_S, RISE_SMOOTH_S))
    before = st.get("f", self.force[side])
    f = st["f"] = (1.0 - k) * before + k * self.force[side]
    fm = st["fm"] = (1.0 - km) * st.get("fm", self.force[side]) + km * self.force[side]
    rise = st["rise"] = (1.0 - kr) * st.get("rise", 0.0) + kr * float((f - before) @ w) / dt
    along = float(fm @ w)
    st["peak"] = max(st["peak"], along)
    x = 2.0 * (limit - float(f @ w) - AHEAD_S * rise) / limit
    if going and x < 0.0:
      # ...but a hand GOING along its way is held by nothing: where its force
      # is going only slows it, and it backs off past its force as read, at
      # its speed -- the whole body's reading rides 4-5 N on a hand moving in
      # free air, and backed off on where that went, a 4 N pull up 0.3 m
      # stopped after 1.1 cm (#547). ⚠ One that meets a stop goes on reading
      # as going for up to `GO_S`.
      rate = self.speed[side] * min(0.0, max(-1.0, 2.0 * (limit - float(f @ w)) / limit))
    else:
      rate = self.speed[side] * (min(1.0, x) if x >= 0.0 else BACK_SPEED * max(-1.0, x))
    if settling:
      # ...standing while its body takes a posture, but backing off: stood
      # still, it held on reading 15 times its force as a posture stood the
      # body up with the hand on a locked bar (#547)
      rate = min(rate, 0.0)
    prev, prev_rot = c.pos.copy(), c.rot
    to = c.pos + rate * dt * w
    ahead = float((to - h) @ w)
    pinned = ahead > LEAD_MAX_M
    if pinned:
      to = to - (ahead - LEAD_MAX_M) * w
    # ACROSS its way, GIVING, the command follows the hand -- what it holds
    # takes it -- no faster than `ACROSS_SPEED` times its speed (yielding
    # only to a force read past a dead band, it lagged a drop handle's flip
    # and ran out of reach on every lid); HOLDING, it keeps its line (giving,
    # a cube set down slid 13 cm along what it touched)
    if pv["give"]:
      off = to - h
      to = to - (off - float(off @ w) * w)
    step = to - prev
    n = float(np.linalg.norm(step))
    most = ACROSS_SPEED * self.speed[side] * dt
    c.pos = prev + (step if n <= most else step * (most / n))
    # ...and its TURN gives to what it holds, giving -- ⚠ but never about
    # the jaws' own line, which what it holds turns about between the pads:
    # a lid's weight bent the wrist about that line, the give followed it
    # 70 deg down, and the arm ran out of reach at 46-54 deg of lid on 4
    # chests of 8; held there, 7 of 8 opened past 57 (#525)
    if pv["give"] and c.rot is not None:
      e = _rot_error(r, c.rot)
      jaws = c.rot[:, 0]
      e = e - float(e @ jaws) * jaws
      a_ = float(np.linalg.norm(e))
      if a_ > 1e-9:
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, e / a_, GIVE_RATE * a_)
        m9 = np.zeros(9)
        mujoco.mju_quat2Mat(m9, quat)
        c.rot = m9.reshape(3, 3) @ c.rot
    q, ep, er = sm.solve(side, c.pos, c.rot)
    pv["beyond"] = False
    if ep > REPLAN_M or er > REPLAN_ROT:
      there = None if going else self._reaching(side, h, c.rot)
      if there is not None:
        # HELD (`GO_ON_M`): the command stays where the arm last reached --
        # backing off, a share of the way to the hand, its arm's joints as
        # far toward those that reach it, none faster than `JOINT_RATE` (a
        # setpoint is ramped, never written across a gap: the solve that
        # reaches the hand begins where one that missed left the arm)
        lead = float((prev - h) @ w)
        back = min(1.0, -rate * dt / lead) if rate < 0.0 and lead > 0.0 else 0.0
        gap = float(np.abs(there - self.q_cmd[side]).max())
        back = min(back, JOINT_RATE * dt / gap) if gap > 0.0 else back
        c.pos, c.rot, pinned = prev + back * (h - prev), prev_rot, True
        q, ep, er = self.q_cmd[side] + back * (there - self.q_cmd[side]), self.miss[side], 0.0
        pv["beyond"] = True
      elif st["t"] - st.get("planned", -REPLAN_S) >= REPLAN_S:
        st["planned"] = st["t"]
        # ...never one it has left this move: free to, two postures could take
        # turns, each switch a fresh wait, and the move never end out of reach
        left_ = st.setdefault("left", [])
        posture = self.posture_for(side, c.pos, rot=c.rot, onward=True, past=left_)
        if posture is not None and tuple(posture) != tuple(self.posture):
          left_.append(tuple(float(v) for v in self.posture))
          self.hold_posture(*posture)
          st["postured"] = st["t"]
    self.goal[side].pos, self.goal[side].rot = c.pos.copy(), c.rot
    # held fast: the hand stays put while it pushes with its whole force, or
    # goes nowhere with its command as far ahead as it may go -- as far as its
    # arm reaches, only where it pushes its whole force: else it can reach no
    # further
    if not moved and along >= FAST_SHARE * limit and not settling:
      pv["still"] += dt
    pv["pinned"] = pv["pinned"] + dt if pinned and not pace and not settling else 0.0
    if ep < WILD_M:
      self.q_cmd[side], self.miss[side] = q, ep
    if ep > BLOCKED_M or er > HOLD_ROT_TOL:
      coming = (st["t"] < SETTLE_S
                or st["t"] - st.get("postured", -POSTURE_WAIT_S) < POSTURE_WAIT_S)
      self._off[side] = 0.0 if coming else self._off[side] + dt
      if self._off[side] >= BLOCKED_S:
        self._end(side, "out of reach", h, r)
      return c.stiffness
    self._off[side] = 0.0
    if pv["went"] >= pv["distance"]:
      self._end(side, "distance", h, r)
    elif pv["still"] >= FAST_S:
      self._end(side, "held fast", h, r)
    elif pv["pinned"] >= FAST_S:
      short = pv["beyond"] and along < FAST_SHARE * limit
      self._end(side, "out of reach" if short else "held fast", h, r)
    return c.stiffness

  def _feel(self) -> None:
    """Each hand's force off the drivers' torques as just read, the model
    posed off the senses (`_pose`), a turning joint's friction taken off
    (the arms' and the waist's): every hand's arm alone, then the whole
    body's for those at work, standing (`FELT_PHASES`) -- and what the robot
    reads of it (`sensed_force`), the same with the body's own motion in it
    as its senses have it for a hand going its own way (`arms.Motion`,
    `HELD_PHASES`, #631)."""
    sm = self.self_model
    motion = self.motion.read(sm, self.qd, self.stand)
    tau = self.tau.copy()
    tau[sm.waist] -= friction.taken_by(self.friction, hm.WAIST, self.qd[sm.waist])
    for s in hm.SIDES:
      arm = sm.arm[s]
      tau[arm] -= friction.taken(self.friction, s, self.qd[arm])
      self.arm_force[s] = sm.hand_force(s, tau[arm])
    self.force = dict(self.arm_force)
    working = [s for s in hm.SIDES if self.phase[s] in FELT_PHASES]
    whole = self.standing and bool(working)
    if whole:
      self.force.update(sm.felt(working, tau, self.still))
    self.sensed_force = dict(self.force)
    going = [s for s in hm.SIDES if self.phase[s] not in HELD_PHASES + ("rest",)]
    if motion is None or not going:
      return
    for s in going:
      if not (whole and s in working):
        self.sensed_force[s] = sm.hand_force(s, tau[sm.arm[s]], motion=motion)
    if whole and all(s in going for s in working):
      self.sensed_force.update(sm.felt(working, tau, self.still, motion))

  def _update(self, dt: float) -> None:
    sm = self.self_model
    self._pose()
    self._feel()
    # Where each hand IS, by its senses (#493): a carried hand's to steer by,
    # and every hand's for a stop condition to read (#554)
    seen = {s: sm.grip_pose(s) for s in hm.SIDES}
    self.sensed_grip = {s: seen[s][0].copy() for s in hm.SIDES}
    have = {s: seen[s] for s in hm.SIDES if self.phase[s] == "carry"}
    self._trim(dt)
    self._pose(self.q_cmd)
    for s in hm.SIDES:
      phase, stiffness = self.phase[s], 1.0
      if phase == "carry":
        pulled = self.pulls[s] is not None and self.pulls[s].get("pull") is not None
        stiffness = (self._pulled if pulled else self._held)(s, dt, *have[s])
        sm._set_arm(s, self.q_cmd[s])
      elif phase in ("rest", "move", "joints", "home"):
        there = self._joints(s, dt)
        sm._set_arm(s, self.q_cmd[s])
        self.miss[s] = 0.0
        if phase == "move" and there and self.outs[s]:
          self.q_to[s] = self.outs[s].pop(0)
        elif phase == "move" and there and self.stand.settled and self._posture_there(dt):
          pos, rot = sm.grip_pose(s)
          self.cmd[s] = HandGoal(pos, rot, self.goal[s].stiffness)
          self.phase[s] = "approach"
          # ...once it has looked where it stands (#627): else its hand waits
          # at its pre-grasp while its body takes the posture the look chose
          self.looking[s] = not self._look(s)
        elif phase == "home" and there:
          if self.homes[s]:
            self.q_to[s] = self.homes[s].pop(0)
          else:
            self.phase[s] = "rest"
      elif phase == "blocked":
        sm._set_arm(s, self.q_cmd[s])
        stiffness = self.cmd[s].stiffness
      else:
        # approach, pull, hold or retreat: a line (`_line`), solved again.
        held = HandGoal(self.cmd[s].pos.copy(), self.cmd[s].rot, self.cmd[s].stiffness)
        waiting = phase == "approach" and self.looking[s]
        if waiting:
          # ...but a hand waiting at its pre-grasp holds there while its body
          # takes the posture its look chose (`_look`), and looks again there
          there = False
          if self.stand.settled and self._posture_there(dt):
            self.looking[s] = not self._look(s)
        else:
          there = self._line(s, dt)
        c = self.cmd[s]
        q, ep, _ = sm.solve(s, c.pos, c.rot)
        # Out of reach on its way or while it holds; pushing only on its
        # way in (a hold may carry a weight, and a pull pushes on purpose) --
        # never while its body takes a posture with the hand waiting: the
        # force off the drivers assumes the body still (`BLOCKED_S`).
        off = phase != "retreat" and not waiting and (
          ep > BLOCKED_M or (phase == "approach" and np.linalg.norm(self.force[s]) > BLOCKED_N))
        self._off[s] = self._off[s] + dt if off else 0.0
        if self._off[s] >= BLOCKED_S:
          # Stopped where it was last commanded: the goal is out of its reach,
          # or the hand pushes on something.
          self.cmd[s] = held
          q, ep, _ = sm.solve(s, held.pos, held.rot)
          self.phase[s], there = "blocked", False
        self.q_cmd[s], self.miss[s], stiffness = q, ep, c.stiffness
        if there and phase in ("approach", "pull"):
          self.phase[s] = "hold"
        elif there and phase == "retreat":
          self._go_home(s)
      self.arms[s].command(self.q_cmd[s], sm.gravity(s), stiffness)
    # ...and the waist to the posture's angle, the trunk above it held as
    # the arms are commanded
    self._turn_waist(dt, sm.waist_gravity())
