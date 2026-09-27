"""The perceptive policy's height scan as the robot would have it (#388).

`legs.policy.PolicyDriver` casts its scan as ideal rays by default: the
terrain itself, which is what mjlab trained on. The robot has no such thing.
It has the D435 on its nose (`perception/depth.py`, mount "body") and a map
that keeps what the camera saw (`perception/heightmap.py`), laid in the
world through the body's pose estimate: roll and pitch off gravity (they do
not drift), heading, position and height from the legs (`legs/odometry.py`)
-- or the truth, to tell the sensor's error from the reckoning's. The scan
is the map read at the scan's grid.

A scan point on a cell the camera has not seen reads as the NEAREST cell it
has. The camera sees the floor from ~0.5 m ahead of the body and ~0.45 m to
either side, so the scan's outer rows (0.5 m out) and, at a start, the ground
under and behind the robot are unseen; read as the ground under the feet,
they put a flight's upper treads at the floor and the scan was up to 0.4 m
wrong on a fifth of its points (`quad_spike.py --climb --scan map`). Read as
the nearest seen cell, they are the same tread a few centimetres in. Before
the camera has seen anything, a point reads as the ground under the feet.
"""

import math

import numpy as np
from scipy import ndimage

from pluggybot.legs.model import LEGS
from pluggybot.legs.policy import scan_offsets
from pluggybot.perception.depth import PERIOD, DepthCamera
from pluggybot.perception.heightmap import CELL_M, HeightMap
from pluggybot.robot import FIRST, RobotHandle

#: A point this far over the body is not ground the scan reaches: climbing a
#: flight the camera tilts ~3° up and would map the underside of a floor.
CEILING_M = 0.5
#: The map's window, m: the scan's farthest point is 0.94 m from the body.
WINDOW_M = 4.0


class MapScan:
  """The D435's map, read at the policy's scan grid. `step()` after every
  physics step (a frame every `period`); `scan()` whenever the policy asks.
  `odometry` is the pose the map is laid through; None is the truth."""

  def __init__(self, model, data, odometry=None, handle: RobotHandle = FIRST,
               period: float = PERIOD, seed: int = 0):
    self.m, self.d, self.odo = model, data, odometry
    self.cam = DepthCamera(model, handle, mount="body", seed=seed)
    self.map = HeightMap(WINDOW_M, CELL_M, z_band=(-math.inf, math.inf))
    self.root = model.body(handle.root).id
    self.feet = [model.site(handle.el(f"{leg}_foot")).id for leg in LEGS]
    self.foot_r = float(model.geom_size[model.geom(handle.el("FL_foot")).id][0])
    self.every = max(1, round(period / model.opt.timestep))
    self.xy = scan_offsets()
    self.steps = 0
    #: Per map cell, the (iy, ix) of the nearest cell the camera has seen;
    #: None until it has seen one.
    self.nearest: np.ndarray | None = None
    self.look()

  def pose(self) -> tuple[float, float, float, float, np.ndarray]:
    """(x, y, z, heading) as believed, and the torso's tilt with the heading
    taken out."""
    rot = self.d.xmat[self.root].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
    if self.odo is None:
      x, y, z = self.d.xpos[self.root]
      return float(x), float(y), float(z), yaw, level
    return self.odo.x, self.odo.y, self.odo.z, self.odo.yaw, level

  def look(self) -> None:
    """One frame into the map."""
    frame = self.cam.frame(self.d)
    x, y, z, yaw, level = self.pose()
    p = frame.points @ level.T
    p[:, 2] += z
    self.map.update((x, y, yaw), p[p[:, 2] < z + CEILING_M])
    unseen = np.isnan(self.map.height)
    if not unseen.all():
      self.nearest = ndimage.distance_transform_edt(
        unseen, return_distances=False, return_indices=True)

  def step(self) -> None:
    self.steps += 1
    if self.steps % self.every == 0:
      self.look()

  def ground(self) -> float:
    """The believed height of the ground under the lowest foot."""
    drop = min(self.d.site_xpos[f][2] for f in self.feet) - self.d.xpos[self.root][2]
    return self.pose()[2] + float(drop) - self.foot_r

  def cells(self) -> tuple[np.ndarray, np.ndarray]:
    """The map cell (iy, ix) under each scan point."""
    x, y, _, yaw, _ = self.pose()
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * self.xy[:, 0] - s * self.xy[:, 1]
    wy = y + s * self.xy[:, 0] + c * self.xy[:, 1]
    m = self.map
    ix = np.floor(wx / m.cell_m).astype(np.int64) - m.origin[0]
    iy = np.floor(wy / m.cell_m).astype(np.int64) - m.origin[1]
    return iy, ix

  def seen(self) -> np.ndarray:
    """Whether the camera has seen the cell under each scan point."""
    return ~np.isnan(self.map.height[self.cells()])

  def scan(self) -> np.ndarray:
    """The body's height over the terrain under each scan point, as
    `PolicyDriver.height_scan` casts it."""
    if self.nearest is None:
      return np.full(len(self.xy), self.pose()[2] - self.ground())
    iy, ix = self.cells()
    h = self.map.height[self.nearest[0][iy, ix], self.nearest[1][iy, ix]]
    return self.pose()[2] - h
