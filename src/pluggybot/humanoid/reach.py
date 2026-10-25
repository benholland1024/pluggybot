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

import itertools
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import (REACH_POS_TOL, REACH_ROT_TOL, Arm, Gripper,
                                     HandGoal, SelfModel, _rot_error)
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
#: reach of the posture the policy held, 2 cm under the one asked.
REACH_MARGIN_M = 0.015
#: A hand on its way in STOPS where it is (`blocked`) once its own solve
#: misses its command by this, m -- the goal out of reach from the posture
#: the policy truly holds -- or once the force its arm's currents read at
#: the hand passes this, N, pushing on something it was not sent to. ⚠
#: Unstopped, a crouch that sat further back than the plan's statue left a
#: chest's knob 10.7 cm out of reach, and the arm drove the gripper into
#: the counter's edge until it pushed the robot over backwards.
BLOCKED_M, BLOCKED_N = 0.03, 30.0
#: ...for this long, s: the force read off the currents assumes the arm is
#: still, and a deep bow's folding read 81.5 N at the hand for an instant
#: as its approach began. A push lasts; a transient does not.
BLOCKED_S = 0.15
#: The posture's own loop: the pelvis's height and the trunk's pitch as the
#: robot's model of itself has them (its legs' encoders, its IMU) held to
#: the plan's by trimming what the policy is asked -- this much of the
#: miss a second, at most this far, m and rad. Unheld, the policy stood 2
#: cm under the height it was asked.
TRIM_RATE, TRIM_MAX = 1.0, (0.05, 0.15)
#: A hand's approach starts once the posture the robot measures is this
#: near the plan's, m and rad, and its trunk is still -- its gyro under
#: `TRUNK_STILL_RATE`, rad/s -- or this long after its command arrived, s: a
#: full-stretch reach begun before the body had folded into a deep bow
#: stopped short. ⚠ The force read off the currents assumes the arm still
#: (`BLOCKED_N`): the floor's approach begun as its bow stopped turning
#: (0.59 rad/s) read 492 N at the hand for 0.2 s and stopped 6 cm short of
#: it; begun under 0.05, 62 N for an instant, and it held there (#520).
POSTURE_TOL, TRUNK_STILL_RATE, POSTURE_WAIT_S = (0.025, 0.06), 0.05, 4.0
#: A hand that is out moves no faster than this when carried,
#: m/s: at the reach's own 0.2 a lid's pull outran the posture, which
#: comes a second or two behind its command, and the arm's solve missed
#: its command by 2-5 cm all the way up (#493).
HOLD_SPEED = 0.06
#: A path's slack (`path_for`): each of its points must also be reached
#: this much further AHEAD, m. Lifting, the live robot stands further back
#: than its statue: 1.5-2 cm all the way up a lid's arc, 2 cm more at its
#: top, where a path planned with none stopped at 57 deg (#493).
PATH_MARGIN_M = 0.035
#: ...planned on the robot as it stands, the lean it already has is in its
#: pose, and only its last 2 cm is to come: with the statue's whole margin
#: on top, a path planned on a statue was refused once the knob was held.
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
#: A PULL (`pull`, #525): a force move. The force it is closed on -- the
#: hand's own reading along its way -- is read over this long, s; a hand
#: that has gone no further than this, m, in this long, s, while it pushes
#: with this share of its force, is held fast...
ADMIT_SMOOTH_S, FAST_M, FAST_S, FAST_SHARE = 0.1, 0.004, 0.6, 0.8
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
#: ...sampled every this much of its furthest joint's walk, rad.
HOME_SAMPLE_RAD = 0.05
#: ...and home is the rest commanded and its drivers reading every joint
#: within this of it, rad (they read to 0.0026).
HOME_RAD = 0.05

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


def seed(side: str, s) -> np.ndarray:
  q = np.array(s, dtype=float)
  if side == "right":
    q[list(_MIRRORED)] *= -1
  return q


def static_legs(height: float, pitch: float, model: SelfModel | None = None,
                arms: dict | None = None) -> dict:
  """The legs' sagittal joints holding the pelvis `height` off the stand's
  and the trunk pitched `pitch`, both soles flat and the centre of mass
  over their middle: a statue's answer, which the policy's own stance is
  near -- with the arms where `arms` puts them (joint name -> angle), else
  hanging. ⚠ An arm held out moves the centre of mass, and the policy
  moves its pelvis back for it: balanced with its arms hanging, the statue
  stood 3 cm in front of the robot reaching a lid's knob and 7 cm in front
  of it at the lid's top, where the plan's reach was out of the robot's
  (#493). Joint name -> angle."""
  sm = model or SelfModel()
  m, d = sm.m, sm.d
  root, sole = sm.root, sm.soles[0]
  adr = {n: sm.qadr[k] for k, n in enumerate(hm.JOINT_NAMES)}
  h = hm.stand_height() + height
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  stand = {**hm.stand_qpos(), **(arms or {})}

  def put(v):
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

  def resid(v):
    put(v)
    r = d.geom_xmat[sole].reshape(3, 3)
    bottom = d.geom_xpos[sole][2] - m.geom_size[sole][2]
    return [10 * bottom, 3 * r[0, 2], 10 * (d.subtree_com[root][0] - d.geom_xpos[sole][0])]

  best = None
  for guess in ([0.0, -0.2 - pitch, 0.4, -0.2], [0.0, -0.8 - pitch, 1.6, -0.8]):
    fit = least_squares(resid, guess, bounds=([-1.0, -3.0, 0.0, -0.9], [1.0, 2.2, 2.6, 0.6]))
    if best is None or fit.cost < best.cost:
      best = fit
  _, hip, knee, ankle = best.x
  return {f"{s}_{k}": v for s in hm.SIDES
          for k, v in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle))}


def _posed(sm: SelfModel, height: float, pitch: float, q0: np.ndarray,
           arms: dict | None = None) -> None:
  q = q0.copy()
  for n, v in (arms or {}).items():
    q[hm.JOINT_NAMES.index(n)] = v
  for n, v in static_legs(height, pitch, sm, arms).items():
    q[hm.JOINT_NAMES.index(n)] = v
  sm.pose(q, np.array([math.sin(pitch), 0.0, -math.cos(pitch)]))


def _arm(side: str, q_arm) -> dict:
  """One arm's joints by name, for a statue to balance."""
  return {f"{side}_{j}": float(v) for j, v in zip(hm.ARM, q_arm)}


def solve_seeded(sm: SelfModel, side: str, pos, rot, start: np.ndarray | None = None
                 ) -> tuple[np.ndarray, float, float]:
  """`SelfModel.solve` from each of `SEEDS` (and `start`), the best kept:
  within tolerance and nearest `start`, else the least off."""
  tries = [seed(side, s) for s in SEEDS]
  if start is not None:
    tries.insert(0, np.asarray(start, dtype=float))
  best, best_key = None, None
  for q0 in tries:
    sm.q[sm.arm[side]] = q0
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    ok = ep < REACH_POS_TOL and er < REACH_ROT_TOL
    far = 0.0 if start is None else float(np.abs(q - start).max())
    key = (not ok, far if ok else ep + 0.1 * er)
    if best_key is None or key < best_key:
      best, best_key = (q, ep, er), key
  return best


def plan(side: str, goal: HandGoal, model: SelfModel | None = None,
         postures=POSTURES, approach: float = APPROACH_M):
  """The least-bent posture from which `side`'s hand reaches `goal` -- and
  its pre-grasp, `approach` short of it -- WITH `REACH_MARGIN_M` to spare,
  else the least bent that reaches it at full stretch; and the arm's
  joints at both. Tried on the robot's model of itself, a statue's legs at
  each posture. Returns ((height, pitch), arm at the pre-grasp, arm at the
  goal, whether it has slack), or None if no posture reaches it."""
  sm = model or SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  pre = pregrasp(goal, approach)
  shoulder = sm.m.body(f"{side}_shoulder_roll_link").id

  def reaches(pos, rot, start=None):
    q, ep, er = solve_seeded(sm, side, pos, rot, start)
    return q if ep < REACH_POS_TOL and er < REACH_ROT_TOL else None

  for slack in (True, False):
    for height, pitch in postures:
      _posed(sm, height, pitch, q0)
      if slack:
        out = np.asarray(goal.pos) - sm.d.xpos[shoulder]
        if reaches(np.asarray(goal.pos) + REACH_MARGIN_M * out / np.linalg.norm(out),
                   goal.rot) is None:
          continue
      q_goal = reaches(goal.pos, goal.rot)
      if q_goal is None:
        continue
      q_pre = reaches(pre.pos, pre.rot, start=q_goal)
      if q_pre is not None:
        return (height, pitch), q_pre, q_goal, slack
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


class Reach:
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
    #: Each arm's way home still to walk after `q_to` (`way_home`), and
    #: every joint as its driver last read it (`home`), unread at first.
    self.homes: dict[str, list] = {s: [] for s in hm.SIDES}
    self.sensed = np.full(len(hm.JOINT_NAMES), np.nan)
    #: How far the last solve left the model's grip off its command, m, and
    #: whether the reach's plan had slack (`plan`).
    self.miss = {s: 0.0 for s in hm.SIDES}
    self.slack = {s: True for s in hm.SIDES}
    #: How long each hand has been out of reach or pushing (`BLOCKED_S`).
    self._off = {s: 0.0 for s in hm.SIDES}
    self.every = max(1, round(EVERY_S / model.opt.timestep))
    self.steps = 0
    #: The posture the plan wants, and the trim on what the policy is
    #: asked (`TRIM_RATE`).
    self.posture = np.zeros(2)
    self.trim = np.zeros(2)
    self.tau = np.zeros(len(hm.JOINT_NAMES))
    self.qd = np.zeros(len(hm.JOINT_NAMES))
    #: The arm joints' friction as calibrated and stored (`friction.FILE`),
    #: taken off the drivers' torques where a joint turns.
    self.friction = friction.load()
    #: The force each hand pushes with, N, stance frame: its arm's
    #: currents through its model (`SelfModel.hand_force`).
    self.force = {s: np.zeros(3) for s in hm.SIDES}
    #: How fast each hand's command goes along its line, m/s; a carry's
    #: state; and how its last carry ended (`report`).
    self.speed = {s: HAND_SPEED for s in hm.SIDES}
    self.pulls: dict[str, dict | None] = {s: None for s in hm.SIDES}
    self.report: dict[str, dict | None] = {s: None for s in hm.SIDES}
    self._planner: SelfModel | None = None

  # ---- what a caller asks ----

  def reach(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=POSTURES, flip: bool = True) -> bool:
    """A hand to `goal`: planned, then moved, approached and held. False
    (nothing moves) if no posture the policy holds puts it in reach. ⚠ A
    TWO-JAW GRIP IS THE SAME GRIP HALF A TURN ROUND ITS FINGERS (#493): where
    the one asked plans no reach, the other is planned -- the left hand
    reached a chest's knob with its jaws closing to its right in twice the
    places it did with them closing to its left, its wrist's roll at the
    end of its range. `postures` are the ones it may take, `flip` whether
    it may take the other grip."""
    plan_, goal = self._planned(side, goal, approach, postures, flip)
    if plan_ is None:
      return False
    posture, q_pre, _, self.slack[side] = plan_
    self.hold_posture(*posture)
    self._off[side] = 0.0
    self.speed[side] = HAND_SPEED
    self.goal[side] = goal
    self.q_to[side] = q_pre
    self.phase[side] = "move"
    return True

  def plans(self, side: str, goal: HandGoal, approach: float = APPROACH_M,
            postures=POSTURES, flip: bool = True) -> bool:
    """Whether `reach` would plan `goal` for `side` -- nothing moved."""
    return self._planned(side, goal, approach, postures, flip)[0] is not None

  def _planned(self, side, goal, approach, postures, flip):
    plan_ = plan(side, goal, approach=approach, postures=postures)
    if plan_ is None and goal.rot is not None and flip:
      flipped = np.asarray(goal.rot, dtype=float) @ np.diag([-1.0, -1.0, 1.0])
      goal = HandGoal(np.asarray(goal.pos, dtype=float), flipped, goal.stiffness)
      plan_ = plan(side, goal, approach=approach, postures=postures)
    return plan_, goal

  def stop(self, side: str, why: str) -> None:
    """A carry ended from outside (its caller's patience), held where the
    hand IS, as its own stops are."""
    if self.phase[side] == "carry":
      self._end(side, why, *self.grip_now(side))

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
    """The least-bent posture (`HOLD_POSTURES`) from which `side`'s grip reaches
    `pos` -- turned to `rot`, or at any turn -- on the robot's model of
    itself, a statue's legs at each, the arm starting where it is and from
    `SEEDS`; or None if none reaches it. `onward`: a held hand going on,
    whose arm missed `pos` in the posture it holds -- another posture, from
    the arm as it is alone, and none of `past`."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    starts = [self.q_cmd[side]] + ([] if onward else [seed(side, s_) for s_ in SEEDS])
    # the posture it holds first, where it still reaches: a change of
    # posture mid-move is a change of the arm's whole branch -- ⚠ and going
    # on, never the posture it holds, from a fresh seed: up a lid's arc the
    # arm stood at its elbow's stop 2 cm short of its way for 9 s, while
    # another branch of the same posture "reached" (#525)
    here = tuple(float(v) for v in self.posture)
    gone = {here} | {tuple(float(v) for v in p_) for p_ in past}
    order = ([p_ for p_ in HOLD_POSTURES if tuple(p_) not in gone] if onward
             else [here] + [p_ for p_ in HOLD_POSTURES if tuple(p_) != here])
    for height, pitch in order:
      _posed(sm, height, pitch, q0)
      for start in starts:
        sm._set_arm(side, start)
        _, ep, er = sm.solve(side, pos, rot, iters=80)
        if ep < REACH_POS_TOL and er < REACH_ROT_TOL:
          return (height, pitch)
    return None

  def path_for(self, side: str, poses, rot=None) -> tuple | None:
    """A posture (`HOLD_POSTURES`, the least bent first) from which `side`'s grip
    FOLLOWS `poses` (stance frame) in turn at the turn `rot` -- or at each
    pose its own, where `rot` is a list -- the first
    solved from the seeds, each next ON from the last, as a moving arm must
    -- or None. Each point after the first is reached further ahead too: on
    a statue, all of them by `PATH_MARGIN_M`; on the robot as it stands
    (below), those further from its shoulder than the first, by
    `LIVE_MARGIN_M`. ⚠ A point
    reached from a fresh seed is not a path: the arm went up a lid's arc in
    one branch, and where only another reached on, it swung between them
    and jerked the lid shut at 3.3 rad/s (#493)."""
    sm = self._planner_model()
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    poses = [np.asarray(p, dtype=float) for p in poses]
    rots = list(rot) if isinstance(rot, (list, tuple)) else [rot] * len(poses)
    ahead = np.array([PATH_MARGIN_M, 0.0, 0.0])
    # a hand that is out goes on from the arm as it IS, its posture first:
    # the branch its grip left it in is the branch its path must follow --
    # where the path starts at the hand (a reach planned while a hand is
    # out, stopped short, starts elsewhere)
    out = (self.out(side)
           and float(np.linalg.norm(poses[0] - self.grip_now(side)[0])) < PATH_POS_TOL)
    here = tuple(float(v) for v in self.posture)
    order = ([here] + [p_ for p_ in HOLD_POSTURES if tuple(p_) != here]) if out \
      else HOLD_POSTURES
    for height, pitch in order:
      # ⚠ ...AND THE ROBOT AS IT STANDS, its model posed off its senses, in
      # the posture it holds: up a lid's arc it stood 4 cm behind the
      # statue of that posture, which could not reach where its hand WAS
      live = out and (height, pitch) == here
      if live:
        # ...its arm as its encoders read it, the branch its grip is in,
        # never as commanded: a lid's weight held the hand 3.5 cm off its
        # command at the arc's top, from which the path down solved short
        sm.q = self.self_model.q.copy()
        sm.d.qpos[:] = self.self_model.d.qpos
        mujoco.mj_kinematics(sm.m, sm.d)
        shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id].copy()
        reach0 = float(np.linalg.norm(poses[0] - shoulder))
        q, ep, er = sm.q[sm.arm[side]].copy(), 0.0, 0.0
      else:
        # another posture than the one it holds: where it is held there,
        # from the arm as it is or any other way (the live solve carries the
        # arm there as the body moves)
        _posed(sm, height, pitch, q0)
        q, ep, er = solve_seeded(sm, side, poses[0], rots[0],
                                 self.q_cmd[side] if out else None)
      if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
        continue
      for p, r_ in zip(poses[1:], rots[1:]):
        if live:
          further = float(np.linalg.norm(p - shoulder)) > reach0
          target = p + (LIVE_MARGIN_M, 0.0, 0.0) if further else p
        else:
          # the statue balanced with its arm out where it is going (`static_legs`)
          _posed(sm, height, pitch, q0, _arm(side, q))
          target = p + ahead
        sm._set_arm(side, q)
        q, ep, er = sm.solve(side, target, r_, iters=80)
        if ep >= PATH_POS_TOL or er >= HOLD_ROT_TOL:
          break
      else:
        return (height, pitch)
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
    no further. The posture is the one from
    which the hand reaches its way's end, taken first. False (nothing
    moves) for a hand that is not out. How it ends is its `report`."""
    if force <= 0.0:
      raise ValueError(f"a pull pushes with some force, not {force} N")
    if not self.out(side):
      return False
    h, r = self.grip_now(side)
    if self.cmd[side] is not None and self.cmd[side].rot is not None:
      r = np.asarray(self.cmd[side].rot, dtype=float)
    w = np.asarray(way, dtype=float)
    w = w / max(float(np.linalg.norm(w)), 1e-9)
    # the posture its way's end is reached from, taken first, the hand held
    # still meanwhile: a posture changed partway jerks what the hand holds
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
                                 "still": 0.0, "anchor": h.copy()}}
    self.report[side] = None
    self.phase[side] = "carry"
    return True

  def hold_posture(self, height: float, pitch: float) -> None:
    """The posture to stand in, held by its own loop (`TRIM_RATE`)."""
    self.posture = np.array([height, pitch], dtype=float)
    self.stand.ask(*(self.posture + self.trim))

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
    self.q_to[side] = np.asarray(q, dtype=float).copy()
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
    self.phase[side] = "retreat"

  def way_home(self, side: str) -> list[np.ndarray]:
    """The joint targets an arm walks home through from its command, judged
    on the robot's model of itself as it stands (`HOME_DIP_M`): its rest
    alone -- or, where that walk takes some part of the arm further ahead
    than the robot's front (`front`) and lower than the lowest of it there
    now, the side way: rolled out to its side (`HOME_ROLL`), its other
    joints home, then its roll. The side way only where it dips less."""
    sm = self._planner_model()
    sm.q = self.self_model.q.copy()
    sm.d.qpos[:] = self.self_model.d.qpos
    q0, rest = self.q_cmd[side].copy(), self.rest[side].copy()
    ahead = front(sm, side, rest)
    sm._set_arm(side, q0)
    p = outline(sm, side)
    there = p[p[:, 0] > ahead]
    # ⚠ none of it ahead worked over nothing there: judged as if all under
    # it were solid, a drawer pulled close took the side way, which swept a
    # jaw into its face and pushed it 4 cm back in
    if not len(there):
      return [rest]
    low = float(there[:, 1].min())
    straight = dip(sm, side, q0, [rest], low, ahead)
    if straight <= HOME_DIP_M:
      return [rest]
    k = hm.ARM.index("shoulder_roll")
    out = seed(side, (0.0, HOME_ROLL, 0.0, 0.0, 0.0, 0.0))[k]
    raised, lowered = q0.copy(), rest.copy()
    raised[k] = lowered[k] = q0[k] if abs(q0[k]) > HOME_ROLL else out
    aside = [raised, lowered, rest]
    return aside if dip(sm, side, q0, aside, low, ahead) < straight else [rest]

  def _go_home(self, side: str) -> None:
    """The arm on its way home (`way_home`), its goal let go; the posture
    back to the stand once no hand is out."""
    self.goal[side] = self.cmd[side] = None
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
    if self.steps % self.every == 0:
      self._update(self.every * self.m.opt.timestep)
    self.steps += 1

  def posture_now(self) -> np.ndarray:
    """The posture as the robot's model of itself has it: the pelvis's
    height off the stand's (its legs' encoders) and the trunk's pitch (its
    IMU)."""
    sm = self.self_model
    return np.array([sm.d.qpos[2] - hm.stand_height(),
                     math.asin(max(-1.0, min(1.0, float(self.stand.trunk.gravity()[0]))))])

  def _trim(self, dt: float) -> None:
    """The posture as the robot's model has it against the plan's; the
    policy asked that much more, once the posture it was shown has arrived
    (a ramp's lag is not a miss), never past what it trained on."""
    have = self.posture_now()
    if self.stand.settled:
      trim = np.clip(self.trim + TRIM_RATE * (self.posture - have) * dt,
                     -np.array(TRIM_MAX), TRIM_MAX)
      asked = self.posture + trim
      inside = (asked >= [HEIGHT_RANGE[0], PITCH_RANGE[0]]) & (asked <= [HEIGHT_RANGE[1], PITCH_RANGE[1]])
      self.trim = np.where(inside, trim, self.trim)
      self.stand.ask(*(self.posture + self.trim))

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
    settling = (st.get("postured") is not None and not self._posture_held()
                and st["t"] - st["postured"] < POSTURE_WAIT_S)
    # THE FORCE, CLOSED ON WHAT THE HAND FEELS (admittance): along its way the
    # command walks at its speed while the force the hand pushes with along
    # it is under half its limit, slows to a stand as it reaches it, and
    # backs off past it -- through the arm's own springs, its model put 15 N
    # on a drawer's stop for a 5 N limit and 27 for 10, where the hand's
    # reading was within 0.5 N of the truth
    k = min(1.0, dt / ADMIT_SMOOTH_S)
    f = st["f"] = (1.0 - k) * st.get("f", self.force[side]) + k * self.force[side]
    along = float(f @ w)
    st["peak"] = max(st["peak"], along)
    rate = 0.0 if settling else self.speed[side] * min(1.0, max(-1.0, 2.0 * (limit - along) / limit))
    prev = c.pos.copy()
    to = c.pos + rate * dt * w
    ahead = float((to - h) @ w)
    if ahead > LEAD_MAX_M:
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
    self.goal[side].pos, self.goal[side].rot = c.pos.copy(), c.rot
    # held fast: the hand stays put while it pushes with its whole force
    if float(np.linalg.norm(h - pv["anchor"])) > FAST_M:
      pv["anchor"], pv["still"] = h.copy(), 0.0
    elif along >= FAST_SHARE * limit and not settling:
      pv["still"] += dt
    q, ep, er = sm.solve(side, c.pos, c.rot)
    if ((ep > REPLAN_M or er > REPLAN_ROT)
        and st["t"] - st.get("planned", -REPLAN_S) >= REPLAN_S):
      st["planned"] = st["t"]
      # ...never one it has left this move: free to, two postures could take
      # turns, each switch a fresh wait, and the move never end out of reach
      left_ = st.setdefault("left", [])
      posture = self.posture_for(side, c.pos, rot=c.rot, onward=True, past=left_)
      if posture is not None and tuple(posture) != tuple(self.posture):
        left_.append(tuple(float(v) for v in self.posture))
        self.hold_posture(*posture)
        st["postured"] = st["t"]
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
    return c.stiffness

  def _update(self, dt: float) -> None:
    sm = self.self_model
    self._pose()
    for s in hm.SIDES:
      arm = sm.arm[s]
      self.force[s] = sm.hand_force(
        s, self.tau[arm] - friction.taken(self.friction, s, self.qd[arm]))
    # Where a carried hand IS, by its senses (#493).
    have = {s: sm.grip_pose(s) for s in hm.SIDES if self.phase[s] == "carry"}
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
        if phase == "move" and there and self.stand.settled and self._posture_there(dt):
          pos, rot = sm.grip_pose(s)
          self.cmd[s] = HandGoal(pos, rot, self.goal[s].stiffness)
          self.phase[s] = "approach"
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
        there = self._line(s, dt)
        c = self.cmd[s]
        q, ep, _ = sm.solve(s, c.pos, c.rot)
        # Out of reach on its way or while it holds; pushing only on its
        # way in (a hold may carry a weight, and a pull pushes on purpose).
        off = phase != "retreat" and (
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
