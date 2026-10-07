"""Legged odometry (issue #377 item 8): where the body thinks it is, from its
legs and its IMU, with the noise the parts have.

A foot on the ground does not slide, so the torso's velocity is minus the
contact point's velocity relative to the torso: v = -(J(q) q' + w x r(q)),
corrected for the ball foot's roll, over the feet in stance that agree with
the last estimate; J and r are the leg's kinematics at the MEASURED joint angles
(a scratch copy of the model, torso at the origin). The orientation is the
IMU's own (`perception/imu.Attitude`): roll and pitch pulled to the
accelerometer's gravity, so they do not drift, and the heading integrated
off the gyro, so it does. The estimate is only as good as the parts:

  encoders      the output shaft is known to within the gearbox's backlash,
                15 arcmin (both Steadywin tables): uniform in +-half of it,
                redrawn each step; and the driver reports it over CAN in
                whole counts of its 16-bit field, the speed in its 12-bit
                one (`perception/encoders.py`, issue #386).
  IMU           an ICM-42688-P's noise, calibration residue and scale error,
                on all three axes of both parts (`perception/imu.py`); at
                rest, a zero-rate update (`imu.Standstill`, issue #425).
  contact       read off current, not a switch: ODRI measured a current-
                based estimate lagging the truth by ~31 ms (Grimminger 2020),
                so the sim's contact is delayed by `CONTACT_LAG_S`.

Nothing in the served sim reads it yet; #387 puts it there, corrected by
scan matching (`mapping/scan_match.py`, `scripts/drift_spike.py`).
"""

import math

import mujoco
import numpy as np

from pluggybot.legs.actuator import BACKLASH_RAD
from pluggybot.legs.model import JOINT_NAMES, LEGS
from pluggybot.perception.encoders import LEG_POSITION_LSB, LEG_VELOCITY_LSB, quantised
from pluggybot.perception.imu import Attitude, Imu, Standstill
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: How late a foot's contact is known, s (ODRI's current-based estimate).
CONTACT_LAG_S = 0.031
#: A flagged foot whose contact point rises faster than this, relative to the
#: body, is lifting off, m/s: in a trot with a moment of four feet down, the
#: lifting pair dragged the estimate from 0.5 to 0.03 m/s past the gate below.
LIFT_M_S = 0.25
#: A stance foot whose implied body velocity is this far from the last
#: estimate is lifting off, m/s: the lagged contact alone made the estimate
#: 27 % of distance wrong, and a median of the flagged feet 12 %.
CONSENSUS_M_S = 0.15


def _cross(a, b) -> np.ndarray:
  """`np.cross` of two 3-vectors, in its own arithmetic (bit-identical) at a
  tenth of its cost: it ran eight times a physics step per robot (#387)."""
  return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]])


class LegOdometry:
  """Integrates (x, y, z, yaw) in the world frame from the start pose; z is
  what a map of the stairs is built on (#388)."""

  def __init__(self, model, data, seed: int = 0, prefix: str = ""):
    self.prefix = prefix
    self.rng = np.random.default_rng(seed)
    #: The IMU (issue #386), its own stream off the robot's prefix and the
    #: seed; its accelerometer is the model's.
    self.imu = Imu(f"{prefix}quad:{seed}")
    self.rebind(model, data)
    self.history: list[np.ndarray] = []
    self.x, self.y, self.z = (float(v) for v in data.qpos[self.qroot:self.qroot + 3])
    #: The orientation as the IMU says it, from the one the body starts in.
    self.att = Attitude(data.xquat[self.root])
    #: The body's word that it lies at rest (`legs.body`: the posture
    #: machine's `lying`, its drivers limp), and the gyro's rate taken
    #: through the zero-rate update that word allows (issue #425).
    self.resting = False
    self.still = Standstill()
    self.yaw = self.att.yaw()
    self.distance = 0.0
    #: The body's velocity estimate, body frame, and the steps it has been
    #: held with no foot agreeing.
    self.v = np.zeros(3)
    self.held = 0

  def rebind(self, model, data) -> None:
    """The reckoning on a world (and a recompiled one, issue #407): its
    sensors, joints and feet by name, and a kinematic copy of the new
    model; where it believes it is, and its contact history, are state."""
    prefix = self.prefix
    self.m, self.d = model, data
    self.acc_adr = int(model.sensor(f"{prefix}imu_lin_acc").adr[0])
    self.root = model.body(f"{prefix}{ROBOT_ROOT}").id
    ids = [model.joint(f"{prefix}{n}").id for n in JOINT_NAMES]
    self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
    self.vadr = np.array([model.jnt_dofadr[j] for j in ids])
    self.feet_geom = [model.geom(f"{prefix}{leg}_foot").id for leg in LEGS]
    self.feet_site = [model.site(f"{prefix}{leg}_foot").id for leg in LEGS]
    # The kinematic copy: the same model, torso at the origin, legs at the
    # measured angles.
    self.kin = mujoco.MjData(model)
    self.jac = np.zeros((3, model.nv))
    self.jacr = np.zeros((3, model.nv))
    self.foot_r = float(model.geom_size[self.feet_geom[0]][0])
    self.fa = model.body_dofadr[self.root]
    #: The robot's free joint in qpos, by name: a world may carry jointed
    #: bodies before it (#378).
    self.qroot = int(model.jnt_qposadr[model.body_jntadr[self.root]])
    self.lag = max(1, round(CONTACT_LAG_S / model.opt.timestep))

  def _true_yaw(self) -> float:
    w, x, y, z = self.d.xquat[self.root]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

  def _contact(self) -> np.ndarray:
    """Which feet touch anything, as the sim knows it now."""
    present = np.zeros(self.m.ngeom, dtype=bool)
    present[self.d.contact.geom[:self.d.ncon].ravel()] = True
    return present[self.feet_geom]

  def step(self) -> None:
    """Call once per physics step, after it."""
    m, d, dt = self.m, self.d, self.m.opt.timestep
    self.history.append(self._contact())
    if len(self.history) > self.lag:
      self.history.pop(0)
    stance = self.history[0]
    # Measured joints: the output within the backlash band, in the
    # driver's counts.
    half = BACKLASH_RAD / 2
    q = quantised(d.qpos[self.qadr] + self.rng.uniform(-half, half, 12),
                  LEG_POSITION_LSB)
    qd = quantised(d.qvel[self.vadr], LEG_VELOCITY_LSB)
    # Measured body rate and specific force (both in the body frame), and
    # the orientation they say.
    w_meas = self.still.rate(self.imu.gyro(d.qvel[self.fa + 3:self.fa + 6], dt),
                             self.resting, dt)
    self.att.step(w_meas, self.imu.accel(d.sensordata[self.acc_adr:self.acc_adr + 3], dt), dt)
    # Leg kinematics at the measured angles, torso at the origin, level.
    k = self.kin
    k.qpos[:] = m.qpos0
    k.qpos[self.qroot:self.qroot + 7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    k.qpos[self.qadr] = q
    k.qvel[:] = 0.0
    k.qvel[self.vadr] = qd
    mujoco.mj_kinematics(m, k)
    mujoco.mj_comPos(m, k)
    rot = self.att.matrix()
    up = rot[2, :]                             # the floor's normal, body frame
    vs = []
    for i, sid in enumerate(self.feet_site):
      if not stance[i]:
        continue
      mujoco.mj_jacSite(m, k, self.jac, self.jacr, sid)
      r = k.site_xpos[sid]
      # The foot is a ball that ROLLS: its contact point is still, its centre
      # is not (w_shank x r_foot * up). Without this the estimate ran 6.6 %
      # long with perfect contact and no noise.
      w_shank = self.jacr[:, self.vadr] @ qd + w_meas
      v_center = self.jac[:, self.vadr] @ qd + _cross(w_meas, r)
      v_contact = v_center + _cross(w_shank, -self.foot_r * up)
      # A foot rising off the floor is not in stance, whatever the lagging
      # contact estimate says (a planted foot's point moves with the body's
      # bob, a few cm/s).
      if float(up @ v_contact) > LIFT_M_S:
        continue
      vs.append(-v_contact)
    # A foot the lagging contact still calls planted while it lifts off
    # implies a velocity far from the body's: gate each foot against the last
    # estimate, and carry that estimate through the steps when none passes
    # (in a trot, the 31 ms after each footfall, only the lifting pair is
    # flagged; a median of the flagged feet cannot see that).
    passed = [v for v in vs if np.linalg.norm(v - self.v) < CONSENSUS_M_S]
    if passed:
      self.v = np.mean(passed, axis=0)
      self.held = 0
    else:
      self.held += 1
      # Held longer than a lagging contact can explain: the estimate lost the
      # body (a push, a stumble), so it takes the planted feet's median again.
      if self.held > 2 * self.lag and vs:
        self.v = np.median(vs, axis=0)
        self.held = 0
    self.yaw = self.att.yaw()
    v = self.att.level() @ self.v             # the torso's tilt, heading removed
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    self.x += (c * v[0] - s * v[1]) * dt
    self.y += (s * v[0] + c * v[1]) * dt
    self.z += v[2] * dt
    true_v = d.qvel[self.fa:self.fa + 2]
    self.distance += float(np.hypot(*true_v)) * dt

  def height(self) -> float:
    """The torso's centre over its lowest foot, m, as its own senses put it:
    the feet where the last step's measured angles put them (the kinematic
    copy), levelled by the IMU's tilt."""
    feet = self.kin.site_xpos[self.feet_site] @ self.att.level().T
    return float(self.foot_r - feet[:, 2].min())

  def correct(self, x: float, y: float, yaw: float) -> None:
    """A pose from outside the legs (a scan match, the dock): the position
    and the heading, the tilt left to the IMU."""
    self.x, self.y = float(x), float(y)
    self.att.turn(float(yaw) - self.yaw)
    self.yaw = self.att.yaw()

  def error(self) -> tuple[float, float]:
    """(position error, m; heading error, rad) against the truth."""
    x, y = self.d.qpos[self.qroot:self.qroot + 2]
    e = math.hypot(self.x - x, self.y - y)
    dyaw = math.atan2(math.sin(self.yaw - self._true_yaw()),
                      math.cos(self.yaw - self._true_yaw()))
    return e, dyaw

  def height_error(self) -> float:
    """The height estimate's error against the truth, m."""
    return self.z - float(self.d.qpos[self.qroot + 2])
