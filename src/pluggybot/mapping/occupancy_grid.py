
import numpy as np
import math


L_FREE = -0.4   # Amount to subtract when a ray passes through a cell
L_OCC = 0.85    # amount to add when a ray collides a cell


#: The most cells a world's grid may ask for (issues #67, #558): FOUR TIMES
#: THE HOUSE'S GRID, the largest size measured. A tripwire, not a cliff -- an
#: accidentally 10x world is loud instead of quiet. MEASURED 2026-10-29 ON
#: THE DEPLOY BOX (`scripts/grid_spike.py`: the house's bounds grown evenly
#: round it, the bodies' own code, #215's synthetic grid for the plans; the
#: serving image on a core beside the live pair):
#:
#:   cells      a scan's  a plan across  the explore's  the quadruped  the pair's
#:              update    the grid       A* across (s)  pair walking   save
#:     469,200   1.2 ms     42 ms          0.80           1.37x          32 ms
#:     937,200   1.4        74             1.66           1.31x          61
#:   1,408,000   2.3       144             2.64           1.29x          82
#:   1,876,800   2.6       177             3.79           1.30x         138
#:
#: What grows is each scan's update (its evidence is a grid-sized `bincount`
#: added to every cell and clipped), each plan's inflation transform and
#: lattice weights (32 of a 3 m hop's 42 ms at 4x), the explore's frontier
#: leg (A* in pure Python), the wire's image and the save; what does not is
#: the scan matcher's field (a 15 m window) or the height map (4 m); since
#: #712's rule (`GRAZE_M`) a scan's update takes about 1.4 ms more on the
#: dev box on today's 856,800 cells, the slant 0.5 of it. ⚠ The
#: humanoid pays two of them oftener -- its camera's walls are its whole
#: grid dilated every depth frame, and its arrival replans every step (2.5
#: to 11 plans a second where the quadruped makes 0.5), each plan 43 ms at
#: 4x. SimNotes, "The map's size limit"; move it with a measurement, not by
#: feel.
MAX_CELLS = 1_876_800

#: A ray that ends on a surface frees a cell the map holds a wall in, over
#: its last `GRAZE_M` / tan(slant) metres -- where it runs within this times
#: cos(slant) of that surface, m, this far at a graze -- only at
#: `sin(slant) ** GRAZE_POWER` of `L_FREE` (`update`'s `graze_m`, #712): the
#: alley's rays, laid 7-10 cm off the research center's 4 cm south wall the
#: robot had laid sure from inside, erased it in 8.5 s -- none of those
#: clears at the true pose. SimNotes, "A thin wall met at a graze".
GRAZE_M = 0.15
#: ...the weight's power, never a veto: vetoed, every copy of a wall that a
#: belief a few cm off had laid was kept, and the belief held to it. 16 deg
#: off a wall (the alley's median) a clear weighs 0.006 of one, 45 deg 0.25,
#: 60 deg 0.56. Replayed closed-loop on eight flights' scans, every power
#: from 2 to 8 kept the map's walk from the mall at 43 m on all four of its
#: flights, and 4 laid the fewest walls twice (48 cells, 75 at 3, 338 at 2).
GRAZE_POWER = 4.0
#: ...over its last this many metres at most.
GRAZE_MOST_M = 2.0
#: The slant is read off the line from a ray's end to a neighbour's (`_slant`)
#: at least this far from it, m: well over the range noise (a 2 cm sigma,
#: `lidar.SIGMA`) -- off the next bearing's, 1.7 cm across at a metre, a wall
#: seen square on read as met at a graze, and at 0.10 m the replays laid four
#: times the base's walls twice.
SLANT_CHORD_M = 0.15
#: ...the nearest such neighbour each way round, within this many of the
#: scan's bearings and this far round, with no ray between that ended on
#: nothing.
SLANT_REACH = 12
SLANT_SPAN_RAD = math.radians(15.0)


class OccupancyGrid:
  def __init__(self, x_min, y_min, x_max, y_max, resolution=0.05):
    self.x_min = x_min
    self.y_min = y_min
    self.x_max = x_max
    self.y_max = y_max
    self.resolution = resolution

    row_count = int((y_max - y_min) / resolution)
    column_count = int((x_max - x_min) / resolution)
    self.grid = np.zeros((row_count, column_count))

  def world_to_cell(self, x, y) -> tuple[int, int]:
    """Given world coordinate, return the coordinate of the occupancy map"""
    cell_x = int((x - self.x_min) / self.resolution)
    cell_y = int((y - self.y_min) / self.resolution)
    return (cell_x, cell_y)

  def cell_to_world(self, x, y) -> tuple[float, float]:
    """Given a cell on the occupancy map, return the world coordinates"""
    world_x = ((x + 0.5) * self.resolution) + self.x_min
    world_y = ((y + 0.5) * self.resolution) + self.y_min
    return (world_x, world_y)
    
  def update(self, pose, angles, ranges, max_range, origin=(0.03, 0.03), free_from=None,
             graze_m=None):
    """Update the occupancy map. pose = (x, y, theta) at the AXLE midpoint.

    `origin` is where the sensor sits relative to that axle, in robot-frame
    (forward, left) metres. It defaults to the head camera's offset; the
    LIDAR sits elsewhere and passes its own (perception.lidar.LIDAR_ORIGIN),
    which is what the mission maps with. Getting this wrong slides
    the whole map by the difference, which is exactly the class of quiet
    error the rack-frame verdict bug was.

    `free_from`, m a ray, is where each ray may begin to clear a cell the
    map holds a wall in (`perception.lidar.Scan`): nearer, it frees only
    cells with no wall's evidence in them.

    `graze_m` (None: off), m: a ray that ends on a surface frees a cell the
    map holds a wall in, over its last `graze_m` / tan(slant) metres --
    where it runs within `graze_m` cos(slant) of that surface -- only at
    `sin(slant) ** GRAZE_POWER` of `L_FREE` (`GRAZE_M`): square on, nothing
    more than the one cell short of its end -- the slant read off the
    scan's own endpoints either side (`_slant`).
    """
    px, py, theta = pose
    fwd, left = origin
    ox = px + fwd * math.cos(theta) - left * math.sin(theta)
    oy = py + fwd * math.sin(theta) + left * math.cos(theta)

    rows, cols = self.grid.shape
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    if angles.size == 0:
      return

    a = theta + angles                         # ray directions in world frame
    cos_a, sin_a = np.cos(a), np.sin(a)
    hit = ranges < max_range - 1e-6            # max-range rays hit nothing!

    # -- free space: sample along every ray at once, each stopping one cell
    # short of its endpoint. Rays differ in length, so the (ray, sample) grid
    # is sized for the longest and shorter rays mask off their tail.
    free_len = np.where(hit, ranges - self.resolution, ranges)
    n = np.maximum(2, (free_len / (self.resolution / 2)).astype(np.int32))
    sample = np.arange(n.max(), dtype=np.int32)
    ts = sample[None, :] * (free_len / (n - 1))[:, None]
    ts[np.arange(len(n)), n - 1] = free_len    # pin each endpoint exactly
    live = sample[None, :] < n[:, None]
    ixs = ((ox + ts * cos_a[:, None] - self.x_min) / self.resolution).astype(np.int32)
    iys = ((oy + ts * sin_a[:, None] - self.y_min) / self.resolution).astype(np.int32)
    # One unsigned compare covers both bounds: a negative index wraps to a
    # huge uint32, so < cols rejects it too. (No negative wrap-arounds.)
    live &= (ixs.view(np.uint32) < cols) & (iys.view(np.uint32) < rows)

    # A cell counts once per ray no matter how many samples land in it, but
    # every ray crossing it counts separately. A straight ray never re-enters
    # a cell, so within a ray the repeats are consecutive: keep a sample only
    # where the cell differs from the previous live sample on the same ray.
    cells = iys * cols + ixs
    keep = live.copy()
    keep[:, 1:] &= (cells[:, 1:] != cells[:, :-1]) | ~live[:, :-1]

    # -- The hits themselves
    ex, ey = ox + ranges * cos_a, oy + ranges * sin_a
    hxs = ((ex - self.x_min) / self.resolution).astype(np.int32)
    hys = ((ey - self.y_min) / self.resolution).astype(np.int32)
    hin = hit & (hxs.view(np.uint32) < cols) & (hys.view(np.uint32) < rows)

    # One weighted bincount accumulates all the evidence in a single pass:
    # a clear of a cell the map holds a wall in (read before this scan) by a
    # sample `held` weighs its ray's `w_ray` of one (none without a graze)
    held = w_ray = None
    if graze_m is not None:
      # ...over a ray's last graze_m / tan(slant) metres, where it runs
      # alongside the surface it ends on, sin(slant) ** GRAZE_POWER
      sin_in = _slant(a, ex, ey, hit, cos_a, sin_a)
      along = graze_m * np.sqrt(np.maximum(1.0 - sin_in ** 2, 0.0)) / np.maximum(sin_in, 1e-9)
      stretch = np.minimum(np.minimum(along, GRAZE_MOST_M), ranges)
      held = keep & (ts > (ranges - stretch)[:, None])
      w_ray = sin_in ** GRAZE_POWER
    if free_from is not None:
      # ...and nearer than its `free_from`, nothing
      near = keep & (ts < np.asarray(free_from, dtype=float)[:, None])
      held = near if held is None else held | near
    if held is not None and held.any():
      held_cells = cells[held]
      walled = self.grid.reshape(-1)[held_cells] > 0.0
      w = (np.zeros(held_cells.size) if w_ray is None
           else np.broadcast_to(w_ray[:, None], held.shape)[held])
      if free_from is not None and w_ray is not None:
        w[near[held]] = 0.0
      weighed = walled & (w > 0.0)
      free_cells = np.concatenate([cells[keep & ~held], held_cells[~walled], held_cells[weighed]])
      free_weights = np.concatenate([np.full(free_cells.size - int(weighed.sum()), L_FREE),
                                     L_FREE * w[weighed]])
    else:
      free_cells = cells[keep]
      free_weights = np.full(free_cells.size, L_FREE)
    hit_cells = (hys * cols + hxs)[hin]
    weights = np.concatenate([free_weights, np.full(hit_cells.size, L_OCC)])
    evidence = np.bincount(np.concatenate([free_cells, hit_cells]),
                           weights=weights, minlength=rows * cols)
    self.grid += evidence.reshape(rows, cols)

    np.clip(self.grid, -5.0, 5.0, out=self.grid)


  def to_image(self) -> np.ndarray:             # uint8: 0 wall / 255 free / 127 unknown
    """Generate an image of the occupancy map"""
    img = np.full(self.grid.shape, 127, dtype=np.uint8)
    img[self.grid < -0.5] = 255    # confidently free
    img[self.grid > 0.5] = 0       # confidently occupied
    return img


def _slant(a: np.ndarray, ex: np.ndarray, ey: np.ndarray, hit: np.ndarray,
           cos_a: np.ndarray, sin_a: np.ndarray) -> np.ndarray:
  """Per ray (world bearings `a`, ends `ex`, `ey`), the sine of the angle at
  which it meets the surface it ends on, read off the line from its end to
  a neighbour's each way round -- the nearest that ended on something
  `SLANT_CHORD_M` or more from it, within `SLANT_REACH` bearings and
  `SLANT_SPAN_RAD`, no ray between it and this one ending on nothing: the
  larger of the two sines where both ways found one (at an occluding edge
  the way across the jump reads a graze), the one where one did, and 1
  (square on) where neither did."""
  n = a.size
  out = np.ones(n)
  if n < 3:
    return out
  order = np.argsort(np.mod(a, 2 * math.pi), kind="stable")
  s = np.mod(a[order], 2 * math.pi)
  h, sx, sy, ca, sa = hit[order], ex[order], ey[order], cos_a[order], sin_a[order]
  sides = []
  for step in (1, -1):
    sine = np.ones(n)
    found = np.zeros(n, dtype=bool)
    look = np.flatnonzero(h)                   # still looking that way
    for k in range(1, SLANT_REACH + 1):
      if look.size == 0:
        break
      j = (look + step * k) % n
      gap = np.abs(np.mod(s[j] - s[look] + math.pi, 2 * math.pi) - math.pi)
      stop = ~h[j] | (gap > SLANT_SPAN_RAD) | (j == look)
      dx, dy = sx[j] - sx[look], sy[j] - sy[look]
      length = np.hypot(dx, dy)
      ok = ~stop & (length >= SLANT_CHORD_M)
      at = look[ok]
      sine[at] = np.abs(ca[at] * dy[ok] - sa[at] * dx[ok]) / length[ok]
      found[at] = True
      look = look[~stop & ~ok]
    sides.append((sine, found))
  (s1, f1), (s2, f2) = sides
  # ...the larger where both ways found one, else the one found: a miss
  # beside it has no end to read a jump off (the far ray of a graze's)
  out[order] = np.where(f1 & f2, np.maximum(s1, s2), np.where(f1, s1, s2))
  return out
