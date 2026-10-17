"""The seeing walker's height scan as the humanoid has it (#507, stage 5).

The policy was shown the terrain's height under a grid round its feet and
ahead (`training/humanoid_train/walk.py`, `SCAN`); the robot has its D435,
1.26 m up on its head and pitched 40 deg down (`perception/depth.py`, the
root body's mount: the pelvis), and a map that keeps what the camera saw
(`perception/heightmap.py`), laid through what the robot believes: its
heading, position and height off its leg odometry (`humanoid.odometry`),
its tilt off its IMU. The scan is that map read at the policy's grid, as
the quadruped's is (`legs/scan.py`, #388).

The camera sees the floor from 0.48 m ahead of the feet, so the ground
under and behind them is what it saw on the way -- and what its soles FEEL:
a planted sole is on the ground, so its footprint goes into the map at its
height (`feel`), the ground it stands on known before it has looked. A
point it has neither seen nor stood on reads as the nearest cell it has: read
as the ground under the feet instead, a flight's upper treads were the floor
(#388). ⚠ Without the soles' footprints, the floor behind a robot that had
just looked reached the nearest cell it had seen -- next to a wall, the wall,
0.67 m up.

The same map tells the odometry where the ground is under a sole (`under`),
which is how it knows which foot is planted on a stair -- and keeps its
height honest: each frame overlaps the ground already mapped, and the
median height between its points and the cells they land on is how far the
believed height drifted since the last; the belief is moved by it
(`correct`) before the frame goes in. Without it the legs' height ran away
a metre in a few seconds of flat walking, and the map with it.

What the walker is SHOWN is ground it could step on (`ground`): a wall, the
deck beside the house's flight, a counter are the planner's to go round
(the LIDAR's map, the depth camera's low layer), and a point on one reads
as the nearest ground that is not. Training had none (its ground is floors,
blocks and flights, all of a piece), and shown the house's flight between
its walls as they stand, the walker fell over backwards before the first
riser. Ground is what joins the ground under the walker through rises no
taller than `STEP_UP_M` -- a flight does, a counter's top or the deck's,
behind the jump at its edge, does not -- and is no wall's: a cell where one
frame's points ran taller than `WALL_SPAN_M` (a wall's face fills its cell
from the floor up, a riser's no taller than the riser). ⚠ NOT BY HOW HIGH
A POINT IS, near or far: half a metre to the side, a wall the camera had
seen only the foot of read 0.64-0.75 m, as tall as two treads at the
scan's edge, and a counter's top 1.1 m ahead is as high as a flight's
fourth tread.
"""

import math

import mujoco
import numpy as np
from scipy import ndimage

from pluggybot.humanoid import model as hm
from pluggybot.perception.depth import PERIOD, DepthCamera
from pluggybot.perception.heightmap import CELL_M, HeightMap
from pluggybot.robot import FIRST, RobotHandle

#: The policy's grid (`walk.SCAN`): 1.6 m along the heading and 1.0 m
#: across at 0.1 m, its middle `SCAN_AHEAD_M` ahead of the pelvis, ordered
#: as mjlab's (x fastest); a ray cast from `SCAN_RAISE_M` over the pelvis,
#: and a miss reading `SCAN_MAX_M`. The term is scaled by 1/`SCAN_MAX_M`
#: (the policy's own metadata carries it).
SCAN_SIZE_M, SCAN_RES_M, SCAN_AHEAD_M, SCAN_RAISE_M, SCAN_MAX_M = (1.6, 1.0), 0.1, 0.3, 1.0, 5.0
#: A point this far over the pelvis is not ground the scan reaches (a
#: wall, the half wall over a flight).
CEILING_M = 0.5
#: The map's window, m: the scan's farthest point is 1.2 m from the pelvis.
WINDOW_M = 5.0
#: A sole's footprint is read at this many points along and across it,
#: and it rests on the height under this share of them.
SOLE_POINTS, SOLE_CARRIES = (5, 3), 0.3
#: A frame's height is registered on at least this many mapped cells it
#: lands on, within `REGISTER_M` of them (a riser's face and a wall land on
#: cells of another height).
REGISTER_N, REGISTER_M = 100, 0.05
#: ...on cells whose points ran no taller than this, in the frame and in
#: the map, m: flat ground (the D435's noise at 2 m is 2.5 mm).
REGISTER_FLAT_M = 0.02
#: Ground a walker could step on joins the ground under it through rises
#: no taller than this, m (training's highest riser is 0.23); the ground
#: under it is what most of the cells within `UNDER_M` of the pelvis join.
STEP_UP_M, UNDER_M = 0.25, 0.3
#: A cell whose points one frame ran taller than this is a wall's, m.
WALL_SPAN_M = 0.3


def scan_offsets() -> np.ndarray:
  """The grid's points in the pelvis's heading frame, (n, 2), m."""
  xs = np.arange(-SCAN_SIZE_M[0] / 2, SCAN_SIZE_M[0] / 2 + SCAN_RES_M / 2, SCAN_RES_M)
  ys = np.arange(-SCAN_SIZE_M[1] / 2, SCAN_SIZE_M[1] / 2 + SCAN_RES_M / 2, SCAN_RES_M)
  gx, gy = np.meshgrid(xs + SCAN_AHEAD_M, ys, indexing="xy")
  return np.stack([gx.ravel(), gy.ravel()], axis=1)


def ideal_scan(model, data, root: int, xy: np.ndarray | None = None,
               exclude: int = -1) -> np.ndarray:
  """The scan as training cast it, off the world itself: the pelvis's height
  over the terrain under each point, the rays down from `SCAN_RAISE_M` over
  it, turned with the heading. OUR instrument (the truth), never the
  robot's."""
  xy = scan_offsets() if xy is None else xy
  body = data.xpos[root]
  rot = data.xmat[root].reshape(3, 3)
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  c, s = math.cos(yaw), math.sin(yaw)
  top = body[2] + SCAN_RAISE_M
  down = np.array([0.0, 0.0, -1.0])
  geomid = np.zeros(1, dtype=np.int32)
  # The world's geoms alone (group 0), as training cast them: a ray at the
  # pelvis hit the robot's own head and read the floor 0.68 m over it.
  groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
  out = np.full(len(xy), SCAN_MAX_M)
  for i, (x, y) in enumerate(xy):
    origin = np.array([body[0] + c * x - s * y, body[1] + s * x + c * y, top])
    dist = mujoco.mj_ray(model, data, origin, down, groups, 1, exclude, geomid)
    if 0 <= dist <= SCAN_MAX_M + SCAN_RAISE_M:
      out[i] = body[2] - (top - dist)
  return out


class HeadScan:
  """The D435's map, read at the policy's grid (module docstring).
  `step()` after every physics step (a frame every `period`); `scan()`
  whenever the policy asks. `belief()` is the pose the map is laid
  through, (x, y, z, yaw) -- the odometry's -- and `level()` the trunk's
  tilt with the heading taken out (level from body), its IMU's; None for
  either is the truth, to tell the sensor's error from the reckoning's."""

  def __init__(self, model, data, handle: RobotHandle = FIRST, belief=None, level=None,
               correct=None, period: float = PERIOD, seed: int = 0,
               spec: hm.HumanoidSpec = hm.CHOSEN):
    self.m, self.d = model, data
    self.belief, self.level = belief, level
    #: Moves the believed height by how far a frame says it drifted, m:
    #: `correct(dz)`; None leaves the belief alone.
    self.correct = correct
    self.cam = DepthCamera(model, handle, mount="body", seed=seed)
    self.map = HeightMap(WINDOW_M, CELL_M, z_band=(-math.inf, math.inf), spans=True)
    self.root = model.body(handle.root).id
    self.every = max(1, round(period / model.opt.timestep))
    self.xy = scan_offsets()
    self.steps = 0
    #: Per map cell, the (iy, ix) of the nearest cell the camera has seen;
    #: None until it has seen one.
    self.nearest: np.ndarray | None = None
    self.stale = False
    #: The camera's last frame, raw (the trunk's own axes).
    self.frame = None
    # A sole's footprint about its middle, along and across it.
    fl, fw = spec.foot_len, spec.foot_w
    a, b = np.meshgrid(np.linspace(-fl / 2, fl / 2, SOLE_POINTS[0]),
                       np.linspace(-fw / 2, fw / 2, SOLE_POINTS[1]), indexing="xy")
    self.sole = np.stack([a.ravel(), b.ravel()], axis=1)

  def pose(self) -> tuple[float, float, float, float, np.ndarray]:
    """(x, y, z, heading) as believed, and the trunk's tilt with the heading
    taken out."""
    rot = self.d.xmat[self.root].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    if self.level is None:
      c, s = math.cos(yaw), math.sin(yaw)
      level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
    else:
      level = self.level()
    if self.belief is None:
      x, y, z = self.d.xpos[self.root]
      return float(x), float(y), float(z), yaw, level
    x, y, z, yaw = self.belief()
    return x, y, z, yaw, level

  def look(self) -> None:
    """One frame into the map; the frame kept (`frame`) for whoever else
    reads the camera (the planner's low layer, `humanoid.mission`)."""
    frame = self.frame = self.cam.frame(self.d)
    x, y, z, yaw, level = self.pose()
    p = frame.points @ level.T
    p[:, 2] += z
    p = p[p[:, 2] < z + CEILING_M]
    if self.correct is not None and len(p):
      dz = self._drift(x, y, yaw, p)
      if dz:
        self.correct(dz)
        p[:, 2] += dz
    self.map.update((x, y, yaw), p)
    self._redraw()

  def _drift(self, x: float, y: float, yaw: float, p: np.ndarray) -> float:
    """How far the belief's height drifted since the map was laid, m: the
    median of the mapped cells' heights less the frame's points on them."""
    m = self.map
    if m.origin is None:
      return 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    ix = np.floor((x + c * p[:, 0] - s * p[:, 1]) / m.cell_m).astype(np.int64) - m.origin[0]
    iy = np.floor((y + s * p[:, 0] + c * p[:, 1]) / m.cell_m).astype(np.int64) - m.origin[1]
    inside = (ix >= 0) & (ix < m.n) & (iy >= 0) & (iy < m.n)
    # Like with like: the map keeps each cell's highest point, so the frame's
    # highest point in each cell is compared with it. Point by point, the
    # cells' maxima stood higher than the points, and the belief ratcheted up
    # 9 cm in 7 s of walking.
    flat = iy[inside] * m.n + ix[inside]
    order = np.argsort(flat, kind="stable")
    flat, z = flat[order], p[inside, 2][order]
    if not len(flat):
      return 0.0
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    top = np.maximum.reduceat(z, starts)
    cells = flat[starts]
    off = m.height.reshape(-1)[cells] - top
    # Flat ground alone, in the frame and in the map: a riser's face or a
    # tread's edge holds what its view of it did, and on a flight the
    # belief ratcheted up a riser in two.
    level = ((top - np.minimum.reduceat(z, starts) < REGISTER_FLAT_M)
             & (m.span.reshape(-1)[cells] < REGISTER_FLAT_M))
    off = off[level & (np.abs(off) < REGISTER_M)]
    return float(np.median(off)) if len(off) >= REGISTER_N else 0.0

  def ground(self) -> np.ndarray:
    """Which cells are ground the walker could step on (module docstring):
    seen or stood on, no wall's, and joined to the ground under it through
    rises no taller than `STEP_UP_M` -- across what it has not seen, each
    such cell as high as the nearest it has."""
    m = self.map
    seen = ~np.isnan(m.height) & ~(m.span > WALL_SPAN_M)
    if not seen.any():
      return seen
    iy, ix = ndimage.distance_transform_edt(~seen, return_distances=False, return_indices=True)
    filled = m.height[iy, ix]
    edge = np.zeros_like(seen)
    rise = np.abs(np.diff(filled, axis=0)) > STEP_UP_M
    edge[:-1] |= rise
    edge[1:] |= rise
    rise = np.abs(np.diff(filled, axis=1)) > STEP_UP_M
    edge[:, :-1] |= rise
    edge[:, 1:] |= rise
    labels, _ = ndimage.label(~edge)
    x, y = self.pose()[:2]
    cx, cy = m._index(x, y)
    r = int(round(UNDER_M / m.cell_m))
    cx, cy = cx - m.origin[0], cy - m.origin[1]
    near = labels[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
    near = near[near > 0]
    if not len(near):
      return seen
    return seen & (labels == np.bincount(near).argmax())

  def _redraw(self) -> None:
    """The nearest-ground table, for the map as it stands."""
    self.stale = False
    other = ~self.ground()
    if not other.all():
      self.nearest = ndimage.distance_transform_edt(
        other, return_distances=False, return_indices=True)

  def step(self) -> None:
    if self.steps % self.every == 0:
      self.look()
    self.steps += 1

  def forget(self) -> None:
    """The map gone (a fall: what the legs reckoned of the height is gone
    with it); the next frame lays a new one."""
    self.map = HeightMap(WINDOW_M, CELL_M, z_band=(-math.inf, math.inf), spans=True)
    self.nearest, self.stale = None, False

  def feel(self, x: float, y: float, z: float, yaw: float) -> None:
    """A planted sole's footprint into the map: its middle at (x, y), its
    bottom `z` up, facing `yaw`, as the robot believes."""
    m = self.map
    if m.origin is None:
      m.recentre(x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * self.sole[:, 0] - s * self.sole[:, 1]
    wy = y + s * self.sole[:, 0] + c * self.sole[:, 1]
    ix = np.floor(wx / m.cell_m).astype(np.int64) - m.origin[0]
    iy = np.floor(wy / m.cell_m).astype(np.int64) - m.origin[1]
    inside = (ix >= 0) & (ix < m.n) & (iy >= 0) & (iy < m.n)
    if inside.any():
      iy, ix = iy[inside], ix[inside]
      # The nearest-cell table is redrawn only when a cell is new to the map.
      self.stale |= bool((np.isnan(m.height[iy, ix]) | (m.span[iy, ix] > WALL_SPAN_M)).any())
      m.height[iy, ix] = z
      m.span[iy, ix] = 0.0
      m.count[iy, ix] += 1

  def _heights(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray | None:
    """The mapped height at world points, the nearest ground's where unseen
    or a wall's; None before any ground is seen."""
    if self.stale:
      self._redraw()
    if self.nearest is None:
      return None
    m = self.map
    ix = np.clip(np.floor(wx / m.cell_m).astype(np.int64) - m.origin[0], 0, m.n - 1)
    iy = np.clip(np.floor(wy / m.cell_m).astype(np.int64) - m.origin[1], 0, m.n - 1)
    return m.height[self.nearest[0][iy, ix], self.nearest[1][iy, ix]]

  def scan(self) -> np.ndarray:
    """The pelvis's height over the terrain under each scan point, as the
    policy was shown it (before its scale): ground it could step on."""
    x, y, z, yaw, _ = self.pose()
    c, s = math.cos(yaw), math.sin(yaw)
    h = self._heights(x + c * self.xy[:, 0] - s * self.xy[:, 1],
                      y + s * self.xy[:, 0] + c * self.xy[:, 1])
    if h is None:
      return np.full(len(self.xy), z)
    return z - h

  def under(self, x: float, y: float, yaw: float) -> float | None:
    """The ground under a sole whose middle is at (x, y), facing `yaw`, m:
    the height that carries `SOLE_CARRIES` of its footprint the robot has
    seen or stood on (a sole overhanging a tread's edge rests on the edge);
    None where it has done neither. ⚠ NOT ITS HIGHEST POINT: a toe reaching
    the next riser's edge read the tread above, and the height laid off it
    jumped a riser at the first step. ⚠ NEVER THE NEAREST CELL's, as the
    scan's fill is: at boot the nearest was a wall's, and the height it
    anchored was 0.67 m up."""
    m = self.map
    if m.origin is None:
      return None
    c, s = math.cos(yaw), math.sin(yaw)
    ix = np.floor((x + c * self.sole[:, 0] - s * self.sole[:, 1]) / m.cell_m).astype(np.int64)
    iy = np.floor((y + s * self.sole[:, 0] + c * self.sole[:, 1]) / m.cell_m).astype(np.int64)
    ix, iy = ix - m.origin[0], iy - m.origin[1]
    inside = (ix >= 0) & (ix < m.n) & (iy >= 0) & (iy < m.n)
    iy, ix = iy[inside], ix[inside]
    h = m.height[iy, ix]
    h = h[~np.isnan(h) & ~(m.span[iy, ix] > WALL_SPAN_M)]
    return float(np.quantile(h, 1.0 - SOLE_CARRIES)) if len(h) else None
