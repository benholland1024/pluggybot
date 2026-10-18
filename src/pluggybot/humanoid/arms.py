"""The humanoid's arms and grippers (issue #507, stage 2): a hand sent to a
pose by inverse kinematics on the robot's model of itself, every command
carrying the stiffness its caller asks and the arm's own weight; the force
at a hand read off its arm's motor currents; and a grip judged by what the
gripper reports.

THE ROBOT'S MODEL OF ITSELF is its own drawing -- `model.body_xml()`
compiled alone (`SelfModel`), never the world's: what a real robot is built
from. It is posed from the robot's senses: every joint off its driver
(`senses.JointSense`) and the trunk's tilt off its IMU's filter
(`policy.TrunkSense`), the feet flat where the legs say they are. A hand's
target is given in the STANCE frame: on the floor midway between the soles,
x along their mean heading, z up -- fixed in the world while the feet stay
planted, so a target there stays put as the trunk sways, and the legs'
encoders say where the trunk is in it.

A command is the RobStride's MIT command: a target, the gains it rides on,
and a torque fed forward (here the target offset by it over the stiffness,
as `legs.drivers` carries a torque). The gains are the policy's
(`legs.actuator.driver_gains`) times `model.ARM_GAIN_SCALE`, times the
caller's `stiffness`, the damping by its square root (the loop's damping
ratio kept): a compliant pull is a low stiffness.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.senses import GripperSense
from pluggybot.legs.actuator import driver_gains
from pluggybot.legs.drivers import Drivers
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The inverse kinematics' damping (damped least squares, m), its largest
#: step a joint, rad, and an orientation error's weight against a
#: position's, m per rad: a hand 1 cm off weighs as 0.033 rad off.
IK_DAMPING, IK_STEP, IK_ROT_WEIGHT = 0.03, 0.2, 0.3
#: Solved when the grip is this close, m and rad (a handle sits in jaws
#: that open 90 mm).
IK_POS_TOL, IK_ROT_TOL = 0.002, 0.02
#: A target is reachable when the solve ends within this, m and rad.
REACH_POS_TOL, REACH_ROT_TOL = 0.01, 0.1
#: A joint's solution stays this far inside its range, rad.
RANGE_MARGIN = 0.05
#: The force at the hand is read only along the arm's directions whose
#: singular value is past this (`hand_force`, a truncated pseudo-inverse):
#: near a singular arm a few tenths of a newton-metre of friction and
#: counts in the torques read as a push of hundreds of newtons -- MEASURED
#: along a lid's pull (#493), 68 readings in 179 past 100 N and one of
#: 3,000 N; truncated here, 2 and 141 N at most. Damping instead biased a
#: still arm's reading: at 1e-3 a 20 N push read 17.8, where this reads it
#: as undamped (19.3; that arm's least is 0.085).
FORCE_SIGMA = 0.05


def frame(approach, closing) -> np.ndarray:
  """The grip's orientation (stance from hand) for fingers pointing along
  `approach` and jaws closing along `closing`: the hand's -z is where the
  fingers point and its x is the jaws' travel (`model._jaw`)."""
  a = np.asarray(approach, dtype=float)
  a = a / np.linalg.norm(a)
  x = np.asarray(closing, dtype=float)
  x = x - a * (x @ a)
  x = x / np.linalg.norm(x)
  z = -a
  y = np.cross(z, x)
  return np.column_stack([x, y, z])


#: The grips the scenes use: fingers forward with the jaws closing up and
#: down (a horizontal bar, a drawer's pull; the arm held out level puts the
#: hand so), and fingers down with the jaws closing fore and aft (an object
#: on the floor).
FORWARD = frame((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
DOWN = frame((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))


def _rot_error(want: np.ndarray, have: np.ndarray) -> np.ndarray:
  """The rotation taking `have` to `want`, as an axis times its angle."""
  r = want @ have.T
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, r.ravel())
  v = np.zeros(3)
  mujoco.mju_quat2Vel(v, q, 1.0)
  return v


class SelfModel:
  """The robot's model of itself, posed from its senses (module
  docstring): kinematics, Jacobians and the arm's weight off its own
  drawing."""

  def __init__(self, spec: hm.HumanoidSpec = hm.CHOSEN):
    self.m = mujoco.MjModel.from_xml_string(hm.body_xml(spec))
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.ranges = np.array([hm.joint_range(n) for n in hm.JOINT_NAMES])
    self.grip = {s: m.site(f"{s}_grip").id for s in hm.SIDES}
    self.soles = [m.geom(f"{s}_sole").id for s in hm.SIDES]
    self.root = m.body(ROBOT_ROOT).id
    #: Each arm's six joints' places in `hm.JOINT_NAMES`.
    self.arm = {s: np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for j in hm.ARM])
                for s in hm.SIDES}
    self.q = np.zeros(len(hm.JOINT_NAMES))

  def pose(self, q: np.ndarray, gravity_b: np.ndarray) -> None:
    """The model as the senses say: joints `q` (`hm.JOINT_NAMES` order) and
    the trunk tilted so gravity reads `gravity_b` in its frame; placed in
    the stance frame."""
    g = np.asarray(gravity_b, dtype=float)
    g = g / np.linalg.norm(g)
    # g_b = (sin p, -cos p sin r, -cos p cos r) for a roll r then a pitch p.
    pitch = math.asin(max(-1.0, min(1.0, g[0])))
    roll = math.atan2(-g[1], -g[2])
    self.q = np.array(q, dtype=float)
    d = self.d
    d.qpos[:] = 0.0
    d.qpos[self.qadr] = self.q
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), "XYZ")
    d.qpos[3:7] = quat
    mujoco.mj_kinematics(self.m, d)
    # The soles' mean heading and their middle on the floor: the frame the
    # model is moved into.
    heads = [math.atan2(d.geom_xmat[g_][3], d.geom_xmat[g_][0]) for g_ in self.soles]
    yaw = math.atan2(sum(math.sin(h) for h in heads), sum(math.cos(h) for h in heads))
    bottoms = [d.geom_xpos[g_] - d.geom_xmat[g_].reshape(3, 3)[:, 2] * self.m.geom_size[g_][2]
               for g_ in self.soles]
    mid = np.mean(bottoms, axis=0)
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), -yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    d.qpos[0:3] = rz @ (-mid)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, turn, quat)
    d.qpos[3:7] = out
    mujoco.mj_kinematics(self.m, d)
    mujoco.mj_comPos(self.m, d)

  def _set_arm(self, side: str, q_arm: np.ndarray) -> None:
    self.q[self.arm[side]] = q_arm
    self.d.qpos[self.qadr[self.arm[side]]] = q_arm
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)

  def grip_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Where the grip is in the stance frame, and its orientation."""
    sid = self.grip[side]
    return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

  def jacobian(self, side: str) -> np.ndarray:
    """The grip's 6 x 6 Jacobian over its arm's joints: position, then
    rotation."""
    jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
    mujoco.mj_jacSite(self.m, self.d, jp, jr, self.grip[side])
    cols = self.vadr[self.arm[side]]
    return np.vstack([jp[:, cols], jr[:, cols]])

  def solve(self, side: str, pos, rot=None, iters: int = 40) -> tuple[np.ndarray, float, float]:
    """The arm's joints that put its grip at `pos` (and `rot`, if given) in
    the stance frame, from where the arm is now: damped least squares,
    every joint held `RANGE_MARGIN` inside its range. Returns the joints and
    how far off they leave the grip, m and rad."""
    idx = self.arm[side]
    lo, hi = self.ranges[idx, 0] + RANGE_MARGIN, self.ranges[idx, 1] - RANGE_MARGIN
    q = np.clip(self.q[idx], lo, hi)
    self._set_arm(side, q)
    ep, er = np.inf, np.inf
    for _ in range(iters):
      have_p, have_r = self.grip_pose(side)
      e_p = np.asarray(pos, dtype=float) - have_p
      ep = float(np.linalg.norm(e_p))
      if rot is None:
        e, j, er = e_p, self.jacobian(side)[:3], 0.0
      else:
        e_r = _rot_error(np.asarray(rot), have_r)
        er = float(np.linalg.norm(e_r))
        e = np.concatenate([e_p, IK_ROT_WEIGHT * e_r])
        j = self.jacobian(side)
        j[3:] *= IK_ROT_WEIGHT
      if ep < IK_POS_TOL and er < IK_ROT_TOL:
        break
      dq = j.T @ np.linalg.solve(j @ j.T + IK_DAMPING ** 2 * np.eye(len(e)), e)
      dq = np.clip(dq, -IK_STEP, IK_STEP)
      q = np.clip(q + dq, lo, hi)
      self._set_arm(side, q)
    have_p, have_r = self.grip_pose(side)
    ep = float(np.linalg.norm(np.asarray(pos) - have_p))
    er = 0.0 if rot is None else float(np.linalg.norm(_rot_error(np.asarray(rot), have_r)))
    return q, ep, er

  def solve_pinch(self, side: str, pos, jaws, iters: int = 40
                  ) -> tuple[np.ndarray, float, float, np.ndarray]:
    """`solve` for a PINCH (#493): the grip at `pos` with its jaws closing
    along the line `jaws` (either way: a two-jaw grip closes along both),
    the turn about that line the arm's to take -- from where the arm is
    now, so it turns no more than it must. What a held handle that swings
    on its own pin takes: the jaws' line held, the hand pivoting round it.
    Returns the joints, the misses (m, rad) and the grip's turn reached."""
    idx = self.arm[side]
    lo, hi = self.ranges[idx, 0] + RANGE_MARGIN, self.ranges[idx, 1] - RANGE_MARGIN
    q = np.clip(self.q[idx], lo, hi)
    self._set_arm(side, q)
    want = np.asarray(jaws, dtype=float) / np.linalg.norm(jaws)

    def miss():
      have_p, have_r = self.grip_pose(side)
      x = have_r[:, 0]
      w = want if x @ want >= 0.0 else -want
      return np.asarray(pos, dtype=float) - have_p, np.cross(x, w), x
    for _ in range(iters):
      e_p, e_r, x = miss()
      if np.linalg.norm(e_p) < IK_POS_TOL and np.linalg.norm(e_r) < IK_ROT_TOL:
        break
      # the turn about the jaws' own line is free: out of the rotation's rows
      free = np.eye(3) - np.outer(x, x)
      j = self.jacobian(side)
      j = np.vstack([j[:3], IK_ROT_WEIGHT * (free @ j[3:])])
      e = np.concatenate([e_p, IK_ROT_WEIGHT * (free @ e_r)])
      dq = j.T @ np.linalg.solve(j @ j.T + IK_DAMPING ** 2 * np.eye(6), e)
      q = np.clip(q + np.clip(dq, -IK_STEP, IK_STEP), lo, hi)
      self._set_arm(side, q)
    e_p, e_r, _ = miss()
    return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r)), self.grip_pose(side)[1]

  def gravity(self, side: str) -> np.ndarray:
    """The torque each of the arm's joints holds its weight with, still, as
    the model is posed: the generalized gravity force (`mj_rne`, no
    velocity, no acceleration)."""
    d = self.d
    d.qvel[:] = 0.0
    mujoco.mj_comVel(self.m, d)
    out = np.zeros(self.m.nv)
    mujoco.mj_rne(self.m, d, 0, out)
    return out[self.vadr[self.arm[side]]]

  def hand_force(self, side: str, tau: np.ndarray, *, still: np.ndarray | None = None
                 ) -> np.ndarray:
    """The force the hand pushes on what it holds, N, in the stance frame:
    what the arm's motors push with, `tau` (their drivers' torque readings)
    less what holding the arm still takes (`gravity`, or `still`), through
    the grip's Jacobian -- along the directions the arm can tell apart
    (`FORCE_SIGMA`). A joint's friction is in it, unknown."""
    hold = self.gravity(side) if still is None else still
    # tau = hold - J^T f_on_hand: the world's push on the hand is what the
    # motors do not hold; the hand pushes back with its negative.
    u, s, vt = np.linalg.svd(self.jacobian(side))
    inv = np.where(s > FORCE_SIGMA, 1.0 / np.maximum(s, 1e-12), 0.0)
    f_on_hand = u @ (inv * (vt @ (hold - np.asarray(tau))))
    return -f_on_hand[:3]


class Arm:
  """One arm's six drivers, commanded as the robot commands them (module
  docstring)."""

  def __init__(self, model, data, side: str, prefix: str = ""):
    self.side = side
    self.joints = tuple(f"{side}_{j}" for j in hm.ARM)
    self.drivers = Drivers(model, data, prefix, joints=self.joints)
    gains = np.array([driver_gains(hm.CHOSEN.motor(j)) for j in self.joints])
    self.kp = gains[:, 0] * hm.ARM_GAIN_SCALE[0]
    self.kd = gains[:, 1] * hm.ARM_GAIN_SCALE[1]

  def command(self, target: np.ndarray, feed: np.ndarray, stiffness: float = 1.0) -> None:
    kp = self.kp * stiffness
    self.drivers.pd(np.asarray(target) + np.asarray(feed) / kp, kp,
                    self.kd * math.sqrt(stiffness))


#: A grip is HELD, by the gripper's own reading, when shut it stopped this
#: far short of closed, m, with the squeeze up, N: a hand shut on nothing
#: closes; one shut on something stalls against it, pressing.
HELD_GAP_M, HELD_SQUEEZE_N = 0.005, 0.5 * hp.GRIPPER.grip_force_n


class Gripper:
  """One hand's gripper as the robot drives it and reads it: opened to a
  width or shut (it then presses with its sheet's force, `parts.Gripper`),
  and a grip judged by its own encoder and torque estimate
  (`senses.GripperSense`) -- never by what the world says it touches."""

  def __init__(self, model, data, side: str, prefix: str = "", key: str = "humanoid"):
    self.d = data
    self.act = model.actuator(f"{prefix}{side}_gripper").id
    self.sense = GripperSense(model, side, prefix, key)
    self.shut_asked = False

  def open(self, width: float = hp.GRIPPER.stroke) -> None:
    self.d.ctrl[self.act] = min(max(width, 0.0), hp.GRIPPER.stroke) / 2
    self.shut_asked = False

  def shut(self) -> None:
    self.d.ctrl[self.act] = 0.0
    self.shut_asked = True

  def read(self) -> tuple[float, float]:
    return self.sense.read(self.d)

  def held(self) -> bool:
    gap, squeeze = self.read()
    return self.shut_asked and gap > HELD_GAP_M and squeeze > HELD_SQUEEZE_N


@dataclass
class HandGoal:
  """Where a hand is sent: a grip pose in the stance frame and the
  stiffness its commands carry."""
  pos: np.ndarray
  rot: np.ndarray | None = None
  stiffness: float = 1.0
