"""The humanoid's arms and grippers (issue #507, stage 2): a hand sent to a
pose by inverse kinematics on the robot's model of itself, every command
carrying the stiffness its caller asks and the arm's own weight; the force
at a hand read off its arm's motor currents -- or, standing, off the whole
body's (`SelfModel.felt`, #547); and a grip judged by what the gripper
reports.

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
as `humanoid.drivers` carries a torque). The gains are the policy's
(`humanoid.actuator.driver_gains`) times `model.ARM_GAIN_SCALE`, times the
caller's `stiffness`, the damping by its square root (the loop's damping
ratio kept): a compliant pull is a low stiffness.
"""

from dataclasses import dataclass
import hashlib
import math

import mujoco
import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.senses import GripperSense
from pluggybot.humanoid.actuator import driver_gains
from pluggybot.humanoid.drivers import Drivers
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
#: THE WHOLE BODY'S READING (`SelfModel.felt`, #547): each row of the
#: balance weighed by what it cannot know -- the trunk's six, N and N*m, the
#: motion a balance taken as still leaves out (the body's sway, a moving
#: arm); a joint's, the friction it may hold still (`friction.still`), at
#: least this, N*m (a driver's torque reading's counts and noise, 0.02-0.04)
#: -- and a hand's twist on what it holds taken as about this, N*m: on the
#: hands check's grips it was 0.7-1.7 on average and under 4 in 19 readings
#: of 20; left free, a pull on a locked bar read 180 N for an instant, and
#: taken as none, a grip on a drawer's knob read 15 N off in 1 reading of 20.
TRUNK_SD, TORQUE_SD, TWIST_SD = (2.0, 1.0), 0.05, 1.0
#: A joint its encoder reads this near an end of its range, rad, is on its
#: STOP, and its row is left out: the stop carries what its motor does not
#: -- pushed along, an arm straightened onto its elbow's stop, which held 3.1
#: N*m its motor never saw, and the hand read 9 N short.
STOP_RAD = 0.02
#: The balance CLOSES while what its rows miss by, weighed and shared over
#: its spare rows, is under this; past it the body is not standing still on
#: its soles, and the whole body reads nothing. Over the hands check, the
#: bench and the bar it was 0.6 at the median and 28 at the 99th percentile;
#: past 50 (0.7 % of readings) the arm read truer (10.6 N rms against 14.4),
#: and dragged onto a sole's edge by `--hold`'s hardest pull, it was 225-727
#: while the balance read 60 N the wrong way and the arm the truth.
CLOSES = 50.0


#: `SelfModel.statues`, by the drawing's own hash.
_STATUES: dict[str, dict] = {}


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
    xml = hm.body_xml(spec)
    self.m = mujoco.MjModel.from_xml_string(xml)
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.ranges = np.array([hm.joint_range(n) for n in hm.JOINT_NAMES])
    self.grip = {s: m.site(f"{s}_grip").id for s in hm.SIDES}
    self.soles = [m.geom(f"{s}_sole").id for s in hm.SIDES]
    self.root = m.body(ROBOT_ROOT).id
    #: Each arm's six joints' places in `hm.JOINT_NAMES`, each leg's, and
    #: the waist's.
    self.arm = {s: np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for j in hm.ARM])
                for s in hm.SIDES}
    self.leg = {s: np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for j in hm.LEG])
                for s in hm.SIDES}
    self.waist = np.array([hm.JOINT_NAMES.index(j) for j in hm.WAIST])
    self.q = np.zeros(len(hm.JOINT_NAMES))
    #: What a planner works out once off this drawing and keeps, shared by
    #: every model of it -- a fresh reach after each stand-up included: each
    #: posture's statue and whether it stands (`reach.static_legs`,
    #: `reach.stands`), each arm's reach (`reach.arm_reach`). A function of
    #: the drawing alone (36 ms a statue), so kept or worked out afresh it is
    #: the same answer.
    self.statues: dict = _STATUES.setdefault(hashlib.sha256(xml.encode()).hexdigest(), {})

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

  def gravity(self, side: str) -> np.ndarray:
    """The torque each of the arm's joints holds its weight with, still, as
    the model is posed: the generalized gravity force (`mj_rne`, no
    velocity, no acceleration)."""
    return self._held()[self.vadr[self.arm[side]]]

  def waist_gravity(self) -> np.ndarray:
    """...and the waist's joints, the trunk above them with the arms as
    posed: none upright, its axis plumb, but a bow tilts it."""
    return self._held()[self.vadr[self.waist]]

  def _held(self, motion: tuple | None = None) -> np.ndarray:
    """What each dof takes to hold the model as posed against its weight --
    and, given `motion` (`Motion.read`: its speeds and accelerations, the
    model's dofs), to move it so: the inverse dynamics, the mass matrix
    with each motor's rotor (`mj_crb`, `mj_mulM`)."""
    d = self.d
    d.qvel[:] = 0.0 if motion is None else motion[0]
    mujoco.mj_comVel(self.m, d)
    out = np.zeros(self.m.nv)
    mujoco.mj_rne(self.m, d, 0, out)
    if motion is not None:
      d.qvel[:] = 0.0
      mujoco.mj_crb(self.m, d)
      moved = np.zeros(self.m.nv)
      mujoco.mj_mulM(self.m, d, moved, motion[1])
      out += moved
    return out

  def hand_force(self, side: str, tau: np.ndarray, *, still: np.ndarray | None = None,
                 motion: tuple | None = None) -> np.ndarray:
    """The force the hand pushes on what it holds, N, in the stance frame:
    what the arm's motors push with, `tau` (their drivers' torque readings)
    less what holding the arm still takes (`gravity`, or `still`) -- or,
    given the body's `motion` (`Motion.read`), moving it as it moves --
    through the grip's Jacobian, along the directions the arm can tell apart
    (`FORCE_SIGMA`). A joint's friction is the caller's to take off `tau`
    (`humanoid.friction.taken`)."""
    hold = (self._held(motion)[self.vadr[self.arm[side]]] if still is None and motion is not None
            else self.gravity(side) if still is None else still)
    # tau = hold - J^T f_on_hand: the world's push on the hand is what the
    # motors do not hold; the hand pushes back with its negative.
    u, s, vt = np.linalg.svd(self.jacobian(side))
    inv = np.where(s > FORCE_SIGMA, 1.0 / np.maximum(s, 1e-12), 0.0)
    f_on_hand = u @ (inv * (vt @ (hold - np.asarray(tau))))
    return -f_on_hand[:3]

  def felt(self, hands, tau: np.ndarray, still: np.ndarray,
           motion: tuple | None = None) -> dict[str, np.ndarray]:
    """The force each of `hands` pushes with on what it holds, N, in the
    stance frame, off the WHOLE body's drivers as the model is posed
    (`pose`), standing: the balance of the robot, its weight -- and, given
    `motion` (`Motion.read`), its motion -- against what its motors push
    with (`tau`, every joint, a turning arm's friction taken off), M a + C
    + G = S^T tau + sum J^T w over each sole's wrench and each
    hand's -- its force, and its twist about its grip -- solved in least
    squares: the trunk's six rows (`TRUNK_SD`), each leg's joints, the
    waist's (it carries a hand's load to the legs) and each of `hands`'
    arm's, a joint weighed by the friction it may hold still
    (`still`, `TORQUE_SD`) and left out on its stop (`STOP_RAD`), and each
    twist about none (`TWIST_SD`) -- or nothing, `{}`, where the balance
    does not close (`CLOSES`). The legs feel what a straight arm carries
    through its structure (`hand_force` does not, #547); and everything the
    world pushes on but the soles is read as the hands'."""
    m, d = self.m, self.d
    bias = self._held(motion)
    pushed = np.zeros(m.nv)
    pushed[self.vadr] = tau
    jac = []
    for g in self.soles:
      jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jp, jr, d.geom_xpos[g], m.geom_bodyid[g])
      jac += [jp, jr]
    for s in hands:
      jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
      mujoco.mj_jacSite(m, d, jp, jr, self.grip[s])
      jac += [jp, jr]
    joints = np.concatenate([self.leg[s] for s in hm.SIDES] + [self.waist]
                            + [self.arm[s] for s in hands])
    q, lo, hi = self.q[joints], self.ranges[joints, 0], self.ranges[joints, 1]
    joints = joints[(q - lo >= STOP_RAD) & (hi - q >= STOP_RAD)]
    # the free joint's six dofs are the trunk's: no motor pushes there
    rows = np.concatenate([np.arange(6), self.vadr[joints]])
    w = 1.0 / np.concatenate([np.repeat(TRUNK_SD, 3), np.maximum(still[joints], TORQUE_SD)])
    a = np.vstack(jac).T[rows] * w[:, None]
    b = (bias - pushed)[rows] * w
    twist = np.zeros((3 * len(hands), a.shape[1]))
    for k in range(len(hands)):
      twist[3 * k:3 * k + 3, 15 + 6 * k:18 + 6 * k] = np.eye(3) / TWIST_SD
    a, b = np.vstack([a, twist]), np.concatenate([b, np.zeros(len(twist))])
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    miss = a @ x - b
    if float(miss @ miss) / max(1, len(b) - a.shape[1]) > CLOSES:
      return {}
    # ...each hand's force ON it: it pushes back with the negative
    return {s: -x[12 + 6 * k:15 + 6 * k] for k, s in enumerate(hands)}


#: THE BODY'S OWN MOTION IS IN WHAT THE ROBOT READS OF A HAND'S FORCE
#: (`Motion`, #631; `reach.Reach.sensed_force`), a hand going its own way's
#: (`reach.HELD_PHASES`): taken as still, a hand moving in free air read
#: 11-14 N as a reach swung its arm out -- the motors' torques that move the
#: arm and sway the trunk, read as a push -- and a stop condition on a force
#: stopped a reach before its hand left the robot's side. With the simulator's own accelerations the balance read 1-4 N, so
#: the rest is the estimate's. The joints whose motion it takes are the arms' and the
#: waist's (`Motion.MOVED`), each one's acceleration its driver's speeds
#: differenced over this many physics steps: over one, a speed field's count
#: (0.01-0.024 rad/s) is a few rad/s^2, and Luca's reach read 8 N; over a
#: control period, 5 ms late, 9.4 N on its way in; over two, 5.5. The legs'
#: are left still: taken in, they moved one move's worst of 18 by 2 N.
MOTION_STEPS = 2


class Motion:
  """The body's own motion as its senses have it, for its hands' force
  (`SelfModel.felt`, `hand_force`): the trunk's off its IMU -- its turn rate
  off the gyro, that rate's change since the last read, and its
  acceleration off the accelerometer, gravity taken off through the tilt the
  model is posed with -- and the arms' and the waist's joints' off their
  drivers' speeds (`MOVED`, `MOTION_STEPS`). Sampled where the reach is
  stepped, it reads nothing that draws on a part's noise."""

  #: The joints whose motion is taken (`hm.JOINT_NAMES` places).
  MOVED = np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for s in hm.SIDES for j in hm.ARM]
                   + [hm.JOINT_NAMES.index(j) for j in hm.WAIST])

  def __init__(self, every: int, timestep: float):
    self.every, self.h = every, timestep
    #: Every joint's speed `MOTION_STEPS` physics steps before the coming
    #: read, and the trunk's turn rate at the last read; None until then.
    self.speeds_was: np.ndarray | None = None
    self.rate_was: np.ndarray | None = None

  def sample(self, steps: int, stand, data) -> None:
    """Once a physics step, before a read on step `steps` of the reach's
    count (`every` steps a read): the joints' speeds where they will be
    differenced from, as `stand`'s drivers report them (`JointSense.speeds`)."""
    if (steps + MOTION_STEPS) % self.every == 0:
      self.speeds_was = stand.joints.speeds(data)

  def read(self, sm: SelfModel, qd: np.ndarray, stand) -> tuple | None:
    """(speeds, accelerations) in `sm`'s dofs as it is posed (`pose`), off
    the joints' speeds `qd` as just read and `stand`'s IMU (its `trunk`,
    `policy.TrunkSense`) -- None where a reading is missing (the first read,
    a restart's from before #631): the body then taken as still."""
    trunk = stand.trunk
    rate = np.asarray(trunk.rate, dtype=float)
    was, self.rate_was = self.rate_was, rate.copy()
    if self.speeds_was is None or was is None or trunk.accel is None:
      return None
    nv = sm.m.nv
    qvel, qacc = np.zeros(nv), np.zeros(nv)
    dofs = sm.vadr[self.MOVED]
    qvel[dofs] = qd[self.MOVED]
    qacc[dofs] = (qd[self.MOVED] - self.speeds_was[self.MOVED]) / (MOTION_STEPS * self.h)
    # the IMU sits at the trunk's origin, square to it: its rates are the
    # free joint's own, and its specific force less gravity is the trunk's
    # acceleration, turned into the stance frame the model is posed in
    qvel[3:6] = rate
    qacc[3:6] = (rate - was) / (self.every * self.h)
    turn = sm.d.xmat[sm.root].reshape(3, 3)
    qacc[0:3] = turn @ np.asarray(trunk.accel, dtype=float) + sm.m.opt.gravity
    return qvel, qacc

  def kept_state(self) -> dict:
    return {"speedsWas": None if self.speeds_was is None else self.speeds_was.tolist(),
            "rateWas": None if self.rate_was is None else self.rate_was.tolist()}

  def restore_kept(self, state: dict) -> None:
    got = state.get("speedsWas"), state.get("rateWas")
    self.speeds_was, self.rate_was = (None if v is None else np.array(v, dtype=float)
                                      for v in got)


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


class Waist:
  """The waist's drivers (`hm.WAIST`), commanded as an arm's are (`Arm`): a
  target, the policy's gains (`driver_gains`) times `model.WAIST_GAIN_SCALE`,
  and the torque holding the trunk above it fed forward. No policy drives
  it: the reach turns it to its posture's angle (`reach.Reach`)."""

  def __init__(self, model, data, prefix: str = ""):
    self.joints = hm.WAIST
    self.drivers = Drivers(model, data, prefix, joints=self.joints)
    gains = np.array([driver_gains(hm.CHOSEN.motor(j)) for j in self.joints])
    self.kp = gains[:, 0] * hm.WAIST_GAIN_SCALE[0]
    self.kd = gains[:, 1] * hm.WAIST_GAIN_SCALE[1]

  def command(self, target: np.ndarray, feed: np.ndarray) -> None:
    self.drivers.pd(np.asarray(target, dtype=float) + np.asarray(feed, dtype=float) / self.kp,
                    self.kp, self.kd)


#: A grip is HELD, by the gripper's own reading, when shut it stopped this
#: far short of closed, m, with the squeeze up to this share of what its
#: drive pushes there (`pushes`): a hand shut on nothing closes; one shut on
#: something stalls against it, pressing. ⚠ Not of the force asked: under 2
#: cm a hard squeeze's spring stalls short of it, and judged against it the
#: plate's 8 mm wall, its jaws stopped 1 cm apart pushing 56-59 N a pad,
#: read nothing held at 120 N (#646).
HELD_GAP_M, HELD_SHARE = 0.005, 0.5
#: The least a grip may be shut with, N a pad: the sheet's adjustable
#: range is 5-120 (`parts.GRIPPER`).
GRIP_MIN_N = 5.0
#: The squeeze a grip is judged by is its reading's running mean over this,
#: s, sampled every physics step (`Gripper.sample`): one reading's noise,
#: the current sense's 5.1 N, read a 15 N grip as nothing held 7 % of the
#: time and a 5 N one 31 % (#525).
SQUEEZE_S = 0.1


def pushes(gap: float, force: float) -> float:
  """What the gripper's drive squeezes with, N a pad, shut with `force` and
  its jaws stopped `gap` m apart, off its own command: `force`, its torque
  limit, or short of it its spring's (`parts.GRIP_KP`) pull toward shut --
  the actuated jaw `gap / 2` from it, and each pad half the motor's push."""
  return min(force, hp.GRIP_KP * (gap / 2) / 2)


def holds(gap: float, squeeze: float, force: float) -> bool:
  """A shut gripper's word off its own reading (`HELD_GAP_M`, `HELD_SHARE`)."""
  return gap > HELD_GAP_M and squeeze > HELD_SHARE * pushes(gap, force)


class Gripper:
  """One hand's gripper as the robot drives it and reads it: opened to a
  width or shut (it then presses with its sheet's force, `parts.Gripper`),
  and a grip judged by its own encoder and torque estimate
  (`senses.GripperSense`) -- never by what the world says it touches."""

  def __init__(self, model, data, side: str, prefix: str = "", key: str = "humanoid"):
    self.m, self.d = model, data
    self.act = model.actuator(f"{prefix}{side}_gripper").id
    self.sense = GripperSense(model, side, prefix, key)
    self.shut_asked = False
    self.force = hp.GRIPPER.grip_force_n
    self.gap, self.squeeze = self.sense.read(data)

  def _limit(self, force: float) -> None:
    """Each pad squeezes at most `force`, N: the motor's limit, twice it
    (`parts.Gripper.dcmotor`) -- the torque limit the Dex1-1 takes with
    every command."""
    self.force = float(force)
    self.m.actuator_forcerange[self.act] = (-2.0 * self.force, 2.0 * self.force)

  def open(self, width: float = hp.GRIPPER.stroke) -> None:
    self._limit(hp.GRIPPER.grip_force_n)
    self.d.ctrl[self.act] = min(max(width, 0.0), hp.GRIPPER.stroke) / 2
    self.shut_asked = False

  def shut(self, force: float | None = None) -> None:
    """Shut, squeezing with `force`, N a pad -- the sheet's grip force where
    none is given (#525: pinched at 120 N, a door's lever could not turn in
    the jaws, their pads' torsional friction 2.4 N*m against its hinge)."""
    g = hp.GRIPPER.grip_force_n
    self._limit(g if force is None else min(max(float(force), GRIP_MIN_N), g))
    self.d.ctrl[self.act] = 0.0
    self.shut_asked = True

  def sample(self) -> None:
    """One physics step's reading: the gap as it reads, the squeeze into
    its running mean (`SQUEEZE_S`)."""
    gap, squeeze = self.sense.read(self.d)
    self.gap = gap
    self.squeeze += min(1.0, self.m.opt.timestep / SQUEEZE_S) * (squeeze - self.squeeze)

  def read(self) -> tuple[float, float]:
    """(gap m, squeeze N) as last sampled, the squeeze its running mean."""
    return self.gap, self.squeeze

  def held(self) -> bool:
    return self.shut_asked and holds(*self.read(), self.force)


@dataclass
class HandGoal:
  """Where a hand is sent: a grip pose in the stance frame and the
  stiffness its commands carry."""
  pos: np.ndarray
  rot: np.ndarray | None = None
  stiffness: float = 1.0


def kept_value(v):
  """A controller's value as JSON, exactly: an array its dtype and its
  numbers (a float's repr round-trips), a goal its pose."""
  if isinstance(v, np.ndarray):
    return {"nd": v.tolist(), "dtype": str(v.dtype)}
  if isinstance(v, dict):
    return {"dict": {str(k): kept_value(x) for k, x in v.items()}}
  if isinstance(v, (list, tuple)):
    return {"list": [kept_value(x) for x in v]}
  if isinstance(v, HandGoal):
    return {"goal": [kept_value(v.pos), kept_value(v.rot), float(v.stiffness)]}
  if isinstance(v, np.generic):
    return v.item()
  return v


def value_back(v):
  """`kept_value`'s inverse."""
  if isinstance(v, dict):
    if "nd" in v:
      return np.array(v["nd"], dtype=v["dtype"])
    if "dict" in v:
      return {k: value_back(x) for k, x in v["dict"].items()}
    if "list" in v:
      return [value_back(x) for x in v["list"]]
    if "goal" in v:
      pos, rot, stiffness = v["goal"]
      return HandGoal(value_back(pos), value_back(rot), float(stiffness))
  return v
