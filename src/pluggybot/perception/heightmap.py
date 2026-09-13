"""Robot-centric 2.5D height map: what is on the floor near the robot
(issue #34; the representation decision is docs/Parts.md "near-field depth
camera", the measurement behind it `scripts/nearfield_spike.py --cost`).

A window of `SIZE_M` square in `CELL_M` cells, WORLD-AXIS ALIGNED and
recentred on the robot by whole cells as it moves (the elevation-map
convention: content persists while it stays in the window, nothing is ever
rotated or resampled, and what leaves the window is forgotten). Each cell
holds the HIGHEST point the last frame to see it delivered, so a thing that
was carried away reads as floor again the next time the camera looks. The
map answers "is something there, where, and how tall" and cannot say what
is UNDER an overhang -- a table, the rack shelf, a carried module all
project to their top -- which is why points above `Z_MAX` are not floor
things and are dropped, and why a voxel map is the answer if overhangs ever
matter (measured 25-200x the cells for the same frame; the spike).

The update takes the BELIEVED pose (dead reckoning, the axle midpoint, as
`OccupancyGrid.update` does) and robot-frame points from `DepthCamera`;
nothing here reads the sim.
"""

import math

import numpy as np
from scipy import ndimage

SIZE_M = 4.0     # window edge, so +-2 m: the camera's floor band ends at
                 # ~2.1 m ahead, and the half behind is what it drove past
CELL_M = 0.02    # 200 x 200 = 40 000 cells; the 2D grid is 56 000 at 5 cm
Z_MAX = 0.50     # m above the floor: a point higher is not a floor thing (a
                 # table top, the rack shelf, the module on the fork) and a
                 # 2.5D map cannot represent what is under it
Z_MIN = -0.05    # m: below this is noise, never a hole
RAISED_M = 0.03  # m: a cell this far above the floor is "something", above
                 # the 2.5 mm floor noise at 3 m and the wheel's 5 mm creep


class HeightMap:
  """The height map: `height` is (n, n) metres above the floor, NaN where
  never seen; `count` how many frames have measured each cell; indexed
  `[iy, ix]` with `(ix, iy)` at the API, like the occupancy grid."""

  def __init__(self, size_m: float = SIZE_M, cell_m: float = CELL_M) -> None:
    self.cell_m = cell_m
    self.n = int(round(size_m / cell_m))
    self.height = np.full((self.n, self.n), np.nan)
    self.count = np.zeros((self.n, self.n), dtype=np.int32)
    self.origin: tuple[int, int] | None = None   # cell (0, 0)'s world index

  @property
  def cells(self) -> int:
    return self.n * self.n

  def _index(self, x: float, y: float) -> tuple[int, int]:
    return int(math.floor(x / self.cell_m)), int(math.floor(y / self.cell_m))

  def recentre(self, x: float, y: float) -> None:
    """Move the window so the robot's cell is the centre cell, shifting the
    content by whole cells and blanking what enters."""
    ix, iy = self._index(x, y)
    new = (ix - self.n // 2, iy - self.n // 2)
    if self.origin is None:
      self.origin = new
      return
    dx, dy = new[0] - self.origin[0], new[1] - self.origin[1]
    if dx == 0 and dy == 0:
      return
    for arr, blank in ((self.height, np.nan), (self.count, 0)):
      shifted = np.full_like(arr, blank)
      sy = slice(max(dy, 0), self.n + min(dy, 0))
      sx = slice(max(dx, 0), self.n + min(dx, 0))
      ty = slice(max(-dy, 0), self.n + min(-dy, 0))
      tx = slice(max(-dx, 0), self.n + min(-dx, 0))
      if abs(dx) < self.n and abs(dy) < self.n:
        shifted[ty, tx] = arr[sy, sx]
      arr[...] = shifted
    self.origin = new

  def update(self, pose, points: np.ndarray) -> int:
    """Fold one frame in. `pose` = (x, y, theta) at the axle midpoint,
    believed; `points` (k, 3) in the robot frame. Returns cells touched."""
    x, y, th = pose
    self.recentre(x, y)
    if len(points) == 0:
      return 0
    keep = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
    p = points[keep]
    c, s = math.cos(th), math.sin(th)
    wx = x + c * p[:, 0] - s * p[:, 1]
    wy = y + s * p[:, 0] + c * p[:, 1]
    ix = np.floor(wx / self.cell_m).astype(np.int64) - self.origin[0]
    iy = np.floor(wy / self.cell_m).astype(np.int64) - self.origin[1]
    inside = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
    if not inside.any():
      return 0
    flat = iy[inside] * self.n + ix[inside]
    # the highest point per cell: one sort, one segmented max
    order = np.argsort(flat, kind="stable")
    flat, z = flat[order], p[inside, 2][order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    top = np.maximum.reduceat(z, starts)
    cells = flat[starts]
    self.height.reshape(-1)[cells] = top
    self.count.reshape(-1)[cells] += 1
    return int(len(cells))

  def cell_world(self, ix: int, iy: int) -> tuple[float, float]:
    """The world centre of a cell."""
    return ((self.origin[0] + ix + 0.5) * self.cell_m,
            (self.origin[1] + iy + 0.5) * self.cell_m)

  def things(self, raised_m: float = RAISED_M) -> list[dict]:
    """What stands on the floor: connected runs of raised cells, each as
    its world centroid, its height (the highest cell) and its footprint in
    cells, tallest first. Runs are joined across ONE unmeasured cell: from
    a standstill the camera's rows land ~3 cm apart on the floor at 1 m
    (`nearfield_spike.py --find`), so a 5 cm cube arrives as two stripes
    with a 2 cm gap; a measured floor cell between them still splits."""
    seen = np.isfinite(self.height)
    raised = seen & (np.nan_to_num(self.height, nan=-1.0) >= raised_m)
    bridge = ndimage.binary_dilation(raised, np.ones((3, 3))) & ~seen
    labels, k = ndimage.label(raised | bridge, structure=np.ones((3, 3)))
    labels[~raised] = 0
    out = []
    for i in range(1, k + 1):
      iy, ix = np.nonzero(labels == i)
      cx, cy = self.cell_world(float(ix.mean()), float(iy.mean()))
      out.append({"x": cx, "y": cy,
                  "height": float(self.height[iy, ix].max()),
                  "cells": int(len(ix))})
    return sorted(out, key=lambda t: -t["height"])

  def seen_fraction(self) -> float:
    return float(np.isfinite(self.height).mean())
