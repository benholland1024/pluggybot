"""Where the walking humanoid is, from its own senses (issue #507, stage 4):
leg odometry for a biped. The heading is its IMU's (`perception.imu.
Attitude`, the gyro integrated, the tilt held to gravity); the position
moves as the foot it stands on does not -- each tick, the planted sole's
place relative to the pelvis, in the pelvis's level frame, off the legs'
encoders through the robot's model of itself (`arms.SelfModel`); the pelvis
moved by what that place moved, the other way. Which foot is planted: the
lower, levelled, once it is `SWAP_M` lower than the other (a hysteresis).

Off flat ground (stage 5: a flight) its height is kept too, and the planted
foot is read off the TERRAIN the depth camera mapped (`terrain`, the
height under a sole placed there, `humanoid.scan`): a planted sole rests on
the mapped ground and a swinging one is above it, going up a flight or
down, so the planted foot is the one nearer the ground under it, by
`CLEAR_M`. The height moves only as the planted sole does, and the camera's
frames keep it honest (`scan.HeadScan`). ⚠ NEVER LAID OFF THE MAP: laid
off the ground under a landing sole, a footprint believed 6 cm up the
flight read the next tread, the height jumped a riser, and the walker it
fed stumbled. ⚠ NOT THE LOWER FOOT there: on a stair the lower is the
TRAILING one, lifting off its tread while the leading one carries the body,
and read as planted until it rose a riser past the other it lowered the
height a riser a step, and moved the pelvis back as it swung forward; going
down, the leading foot is lower long before it lands. ⚠ NOR A LOAD off the
legs' torques (`SelfModel.hand_force`'s way, at a sole): standing it read
215 N against a true 215, but walking a swinging leg's own torques read up
to 400 N and a planted one's went negative. ⚠ NOR THE IMU's velocity
against each foot's (the two hypotheses): its height drifted a metre in 8 s
on flat ground.

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
#: ...or, where the ground under the soles differs by more than `STEP_M`
#: (a stair's riser, not a floor's unevenness), once the other is this much
#: nearer the ground under it, m. ⚠ NOT ON ONE FLOOR: there both soles
#: read about 0 in double support, and swapping on the map's centimetre of
#: noise read a walk 10 % short where the lower-foot rule read it 2.5 %.
CLEAR_M, STEP_M = 0.01, 0.05
#: A sole lies flat when its up is within this of the world's (cos of 3
#: deg): only then is its footprint the ground's (`felt`).
FLAT_COS = math.cos(math.radians(3.0))
#: A sole box's four bottom corners, in its half-extents. The planted sole
#: is followed by the one that was lowest -- the one on the ground -- since
#: its middle travels 2-3 cm a step as it rolls heel to toe: read off the
#: middle, a walk read 20 % short where the planted foot changed late.
CORNERS = np.array([[a, b, -1.0] for a in (-1.0, 1.0) for b in (-1.0, 1.0)])


class StanceOdometry:
  """The pelvis's (x, y) and heading, believed, from the IMU and the
  planted foot (module docstring)."""

  def __init__(self, self_model: SelfModel, attitude, x: float = 0.0, y: float = 0.0,
               z: float | None = None, terrain=None, felt=None):
    self.sm = self_model
    self.att = attitude
    self.x, self.y = float(x), float(y)
    #: The pelvis's height, m, over the floor it booted on: None until the
    #: first tick reads it off the soles (both on that floor).
    self.z = None if z is None else float(z)
    #: The ground under a sole placed at (x, y) facing `yaw`, m, as the robot
    #: mapped it (None where it has not seen it): `terrain(x, y, yaw)`; and
    #: where the planted sole is each tick, told to the map that keeps it
    #: (`felt(x, y, z, yaw)`, `scan.HeadScan.feel`).
    self.terrain, self.felt = terrain, felt
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
    level frame (x along its heading, z up), m; and, kept for `step`, its
    four bottom corners so, and whether it lies flat."""
    sm = self.sm
    sm.pose(q, gravity_b)
    d = sm.d
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, d.qpos[3:7])
    r = rot.reshape(3, 3)
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    out, corners, flat = [], [], []
    for g in self.soles:
      mat, size = d.geom_xmat[g].reshape(3, 3), sm.m.geom_size[g]
      bottom = d.geom_xpos[g] - mat[:, 2] * size[2]
      out.append(level @ (bottom - d.qpos[0:3]))
      pts = d.geom_xpos[g] + (CORNERS * size) @ mat.T
      corners.append((pts - d.qpos[0:3]) @ level.T)
      flat.append(float(mat[2, 2]) > FLAT_COS)
    self.corners, self.flat = np.array(corners), np.array(flat)
    self.low = self.corners[:, :, 2].min(axis=1)
    return np.array(out)

  def _ground(self, soles: np.ndarray) -> np.ndarray | None:
    """The mapped ground under each sole, m, or None where either is not
    known: each sole put in the world through the belief."""
    if self.terrain is None or self.z is None:
      return None
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    out = []
    for sole in soles:
      g = self.terrain(self.x + c * sole[0] - s * sole[1],
                       self.y + s * sole[0] + c * sole[1], self.yaw)
      if g is None:
        return None
      out.append(g)
    return np.array(out)

  def step(self, q: np.ndarray, gravity_b: np.ndarray) -> None:
    """One tick: the joints as their drivers report them (`hm.JOINT_NAMES`
    order) and gravity in the trunk's frame (the IMU's filter)."""
    soles = self._soles(q, gravity_b)
    if self.z is None:
      self.z = -float(self.low.min())
    ground = self._ground(soles)
    if ground is not None and abs(ground[0] - ground[1]) > STEP_M:
      # Each sole's clearance over the ground under it, its lowest corner's.
      clear = self.z + self.low - ground
      near = int(np.argmin(clear))
      swap = self.foot is None or (near != self.foot
                                   and clear[self.foot] - clear[near] > CLEAR_M)
    else:
      near = int(np.argmin(soles[:, 2]))
      swap = self.foot is None or (near != self.foot
                                   and soles[self.foot, 2] - soles[near, 2] > SWAP_M)
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    if swap:
      self.foot, self.last = near, self.corners[near].copy()
    else:
      # The planted foot's corner on the ground moved back relative to the
      # pelvis: the pelvis moved on.
      k = int(np.argmin(self.last[:, 2]))
      moved = self.corners[self.foot, k] - self.last[k]
      self.last = self.corners[self.foot].copy()
      self.x -= c * moved[0] - s * moved[1]
      self.y -= s * moved[0] + c * moved[1]
      self.z -= moved[2]
    if self.felt is not None and self.flat[self.foot]:
      sole = soles[self.foot]
      self.felt(self.x + c * sole[0] - s * sole[1], self.y + s * sole[0] + c * sole[1],
                self.z + self.low[self.foot], self.yaw)

  def kept_state(self) -> dict:
    return {"x": self.x, "y": self.y, "z": self.z, "offset": self.offset}

  def restore_kept(self, state: dict) -> None:
    self.x, self.y, self.offset = state["x"], state["y"], state["offset"]
    self.z = state.get("z")
    self.foot, self.last = None, None

