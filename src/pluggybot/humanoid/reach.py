"""Reaching (issue #507, stage 2): a hand sent to a pose in the stance frame
while the standing policy balances (`policy.StandDriver`), as a
manipulator's reach is made -- the posture and the arm's last joints
PLANNED on the robot's model of itself (`plan`), the arm MOVED there in
joint space to a pre-grasp pose short of the target, then the hand
APPROACHING it along a straight line, its joints solved again every
command as the trunk sways (`arms.SelfModel`), and HOLDING there. A
retreat goes back the same way. The grippers are driven and read beside it
(`arms.Gripper`). Everything it reads is the robot's own senses.

⚠ From a hanging arm a big move solved in one jump falls into the wrong
basin (the shoulder thrown back, the elbow over): a hanging arm is nearly
singular for a hand moving up. So the plan's solve starts from several
arms (`SEEDS`) and the motion's from wherever the arm last was.

`Reach.step()` runs once a physics step, after the standing policy's and
before `mj_step`; the arms take new commands every `EVERY_S`.
"""

import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

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
#: The posture's own loop: the pelvis's height and the trunk's pitch as the
#: robot's model of itself has them (its legs' encoders, its IMU) held to
#: the plan's by trimming what the policy is asked -- this much of the
#: miss a second, at most this far, m and rad. Unheld, the policy stood 2
#: cm under the height it was asked.
TRIM_RATE, TRIM_MAX = 1.0, (0.05, 0.15)

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


def seed(side: str, s) -> np.ndarray:
  q = np.array(s, dtype=float)
  if side == "right":
    q[list(_MIRRORED)] *= -1
  return q


def static_legs(height: float, pitch: float, model: SelfModel | None = None) -> dict:
  """The legs' sagittal joints holding the pelvis `height` off the stand's
  and the trunk pitched `pitch`, both soles flat and the centre of mass
  over their middle: a statue's answer, which the policy's own stance is
  near. Joint name -> angle."""
  sm = model or SelfModel()
  m, d = sm.m, sm.d
  root, sole = sm.root, sm.soles[0]
  adr = {n: sm.qadr[k] for k, n in enumerate(hm.JOINT_NAMES)}
  h = hm.stand_height() + height
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  stand = hm.stand_qpos()

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


def _posed(sm: SelfModel, height: float, pitch: float, q0: np.ndarray) -> None:
  q = q0.copy()
  for n, v in static_legs(height, pitch, sm).items():
    q[hm.JOINT_NAMES.index(n)] = v
  sm.pose(q, np.array([math.sin(pitch), 0.0, -math.cos(pitch)]))


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


class Reach:
  """The arms and grippers of one body beside its standing policy (module
  docstring). A hand's PHASE: `rest` (hanging), `move` (joint space to the
  pre-grasp, the posture walking to its plan), `approach` (a straight line
  in), `hold`, `retreat` (a straight line out, then joint space home)."""

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
    #: How far the last solve left the model's grip off its command, m, and
    #: whether the reach's plan had slack (`plan`).
    self.miss = {s: 0.0 for s in hm.SIDES}
    self.slack = {s: True for s in hm.SIDES}
    self.every = max(1, round(EVERY_S / model.opt.timestep))
    self.steps = 0
    #: The posture the plan wants, and the trim on what the policy is
    #: asked (`TRIM_RATE`).
    self.posture = np.zeros(2)
    self.trim = np.zeros(2)
    self.tau = np.zeros(len(hm.JOINT_NAMES))
    #: The force each hand pushes with, N, stance frame: its arm's
    #: currents through its model (`SelfModel.hand_force`).
    self.force = {s: np.zeros(3) for s in hm.SIDES}

  # ---- what a caller asks ----

  def reach(self, side: str, goal: HandGoal, approach: float = APPROACH_M) -> bool:
    """A hand to `goal`: planned, then moved, approached and held. False
    (nothing moves) if no posture the policy holds puts it in reach."""
    plan_ = plan(side, goal, approach=approach)
    if plan_ is None:
      return False
    posture, q_pre, _, self.slack[side] = plan_
    self.hold_posture(*posture)
    self.goal[side] = goal
    self.q_to[side] = q_pre
    self.phase[side] = "move"
    return True

  def hold_posture(self, height: float, pitch: float) -> None:
    """The posture to stand in, held by its own loop (`TRIM_RATE`)."""
    self.posture = np.array([height, pitch], dtype=float)
    self.stand.ask(*(self.posture + self.trim))

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
    with a low stiffness), its command walking there on its line."""
    self.goal[side].pos = np.asarray(pos, dtype=float)
    self.phase[side] = "approach"

  def retreat(self, side: str, approach: float = APPROACH_M) -> None:
    """Out along the line it came in on, then home in joint space; the
    posture back to the stand."""
    if self.goal[side] is None:
      return
    self.goal[side] = pregrasp(self.goal[side], approach)
    self.phase[side] = "retreat"

  def done(self, side: str) -> bool:
    """Whether a hand has arrived where it was sent (or hangs), or an arm
    sent in joint space is there."""
    if self.phase[side] == "joints":
      return bool(np.allclose(self.q_cmd[side], self.q_to[side]))
    return self.phase[side] in ("hold", "rest")

  # ---- the loop ----

  def _pose(self, arms: dict | None = None) -> None:
    q, _, tau = self.stand.joints.read(self.d)
    self.tau = tau
    if arms is not None:
      for s in hm.SIDES:
        q[self.self_model.arm[s]] = arms[s]
    self.self_model.pose(q, self.stand.trunk.gravity())

  def grip_now(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Where the robot believes its grip is, stance frame: its model posed
    from its senses."""
    self._pose()
    return self.self_model.grip_pose(side)

  def _line(self, side: str, dt: float) -> bool:
    """The command one step along its line to the goal; True once there."""
    g, c = self.goal[side], self.cmd[side]
    step = g.pos - c.pos
    n = float(np.linalg.norm(step))
    c.pos = g.pos.copy() if n <= HAND_SPEED * dt else c.pos + step * (HAND_SPEED * dt / n)
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
    return n <= HAND_SPEED * dt

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
    if self.steps % self.every == 0:
      self._update(self.every * self.m.opt.timestep)
    self.steps += 1

  def _trim(self, dt: float) -> None:
    """The posture as the robot's model has it against the plan's; the
    policy asked that much more, once the posture it was shown has arrived
    (a ramp's lag is not a miss), never past what it trained on."""
    sm = self.self_model
    have = np.array([sm.d.qpos[2] - hm.stand_height(),
                     math.asin(max(-1.0, min(1.0, float(self.stand.trunk.gravity()[0]))))])
    if self.stand.settled:
      trim = np.clip(self.trim + TRIM_RATE * (self.posture - have) * dt,
                     -np.array(TRIM_MAX), TRIM_MAX)
      asked = self.posture + trim
      inside = (asked >= [HEIGHT_RANGE[0], PITCH_RANGE[0]]) & (asked <= [HEIGHT_RANGE[1], PITCH_RANGE[1]])
      self.trim = np.where(inside, trim, self.trim)
      self.stand.ask(*(self.posture + self.trim))

  def _update(self, dt: float) -> None:
    sm = self.self_model
    self._pose()
    for s in hm.SIDES:
      self.force[s] = sm.hand_force(s, self.tau[sm.arm[s]])
    self._trim(dt)
    self._pose(self.q_cmd)
    for s in hm.SIDES:
      phase, stiffness = self.phase[s], 1.0
      if phase in ("rest", "move", "joints"):
        there = self._joints(s, dt)
        sm._set_arm(s, self.q_cmd[s])
        self.miss[s] = 0.0
        if phase == "move" and there and self.stand.settled:
          pos, rot = sm.grip_pose(s)
          self.cmd[s] = HandGoal(pos, rot, self.goal[s].stiffness)
          self.phase[s] = "approach"
      else:
        there = self._line(s, dt)
        c = self.cmd[s]
        q, ep, _ = sm.solve(s, c.pos, c.rot)
        self.q_cmd[s], self.miss[s], stiffness = q, ep, c.stiffness
        if there and phase == "approach":
          self.phase[s] = "hold"
        elif there and phase == "retreat":
          self.goal[s] = self.cmd[s] = None
          self.q_to[s] = self.rest[s].copy()
          self.phase[s] = "rest"
          if all(p == "rest" for p in self.phase.values()):
            self.hold_posture(0.0, 0.0)
      self.arms[s].command(self.q_cmd[s], sm.gravity(s), stiffness)
