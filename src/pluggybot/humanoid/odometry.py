"""Where the walking humanoid is, from its own senses (issue #507, stage 4):
leg odometry for a biped. The heading is its IMU's (`perception.imu.
Attitude`, the gyro integrated, the tilt held to gravity); the position
moves as the foot it stands on does not -- each tick, the planted sole's
place relative to the pelvis, in the pelvis's level frame, off the legs'
encoders through the robot's model of itself (`arms.SelfModel`); the pelvis
moved by what that place moved, the other way. Which foot is planted: the
lower, levelled, once it is `SWAP_M` lower than the other (a hysteresis).

The quadruped's `legs.odometry.LegOdometry` is its four feet's; nothing
here reads the simulator's truth: the scan matcher corrects what slips
(`correct`, the `Navigator`'s `_set_pose`).
"""

import math

import mujoco
import numpy as np

from pluggybot.humanoid.arms import SelfModel

#: The planted foot changes once the other sole is this much lower, m: a
#: swing foot lifts 5-10 cm, and both down differ by millimetres.
SWAP_M = 0.01


class StanceOdometry:
  """The pelvis's (x, y) and heading, believed, from the IMU and the
  planted foot (module docstring)."""

  def __init__(self, self_model: SelfModel, attitude, x: float = 0.0, y: float = 0.0):
    self.sm = self_model
    self.att = attitude
    self.x, self.y = float(x), float(y)
    #: Its heading's offset from the IMU's own: the IMU starts where the
    #: body boots, and a correction moves this, never the filter's tilt.
    self.offset = 0.0
    self.foot: int | None = None
    self.last: np.ndarray | None = None
    self.soles = self.sm.soles

  @property
  def yaw(self) -> float:
    return self.att.yaw() + self.offset

  def correct(self, x: float, y: float, yaw: float) -> None:
    """A scan match's or a fixture's word: here, facing so."""
    self.x, self.y = float(x), float(y)
    self.offset += math.remainder(yaw - self.yaw, 2 * math.pi)

  def _soles(self, q: np.ndarray, gravity_b: np.ndarray) -> np.ndarray:
    """Each sole's bottom middle relative to the pelvis, in the pelvis's
    level frame (x along its heading, z up), m."""
    sm = self.sm
    sm.pose(q, gravity_b)
    d = sm.d
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, d.qpos[3:7])
    r = rot.reshape(3, 3)
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    out = []
    for g in self.soles:
      bottom = d.geom_xpos[g] - d.geom_xmat[g].reshape(3, 3)[:, 2] * sm.m.geom_size[g][2]
      out.append(level @ (bottom - d.qpos[0:3]))
    return np.array(out)

  def step(self, q: np.ndarray, gravity_b: np.ndarray) -> None:
    """One tick: the joints as their drivers report them (`hm.JOINT_NAMES`
    order) and gravity in the trunk's frame (the IMU's filter)."""
    soles = self._soles(q, gravity_b)
    lower = int(np.argmin(soles[:, 2]))
    if self.foot is None or (lower != self.foot
                             and soles[self.foot, 2] - soles[lower, 2] > SWAP_M):
      self.foot, self.last = lower, soles[lower].copy()
      return
    now = soles[self.foot]
    moved = now - self.last
    self.last = now.copy()
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    # The planted foot moved back relative to the pelvis: the pelvis moved on.
    self.x -= c * moved[0] - s * moved[1]
    self.y -= s * moved[0] + c * moved[1]

  def kept_state(self) -> dict:
    return {"x": self.x, "y": self.y, "offset": self.offset}

  def restore_kept(self, state: dict) -> None:
    self.x, self.y, self.offset = state["x"], state["y"], state["offset"]
    self.foot, self.last = None, None

