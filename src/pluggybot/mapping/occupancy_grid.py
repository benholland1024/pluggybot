
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
#: the scan matcher's field (a 15 m window) or the height map (4 m). ⚠ The
#: humanoid pays two of them oftener -- its camera's walls are its whole
#: grid dilated every depth frame, and its arrival replans every step (2.5
#: to 11 plans a second where the quadruped makes 0.5), each plan 43 ms at
#: 4x. SimNotes, "The map's size limit"; move it with a measurement, not by
#: feel.
MAX_CELLS = 1_876_800


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
    
  def update(self, pose, angles, ranges, max_range, origin=(0.03, 0.03), free_from=None):
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
    hxs = ((ox + ranges * cos_a - self.x_min) / self.resolution).astype(np.int32)
    hys = ((oy + ranges * sin_a - self.y_min) / self.resolution).astype(np.int32)
    hin = hit & (hxs.view(np.uint32) < cols) & (hys.view(np.uint32) < rows)

    # One weighted bincount accumulates all the evidence in a single pass
    if free_from is not None:
      # ...nearer than its `free_from`, a ray frees no cell the map holds
      # any wall's evidence in (read before this scan)
      near = keep & (ts < np.asarray(free_from, dtype=float)[:, None])
      keep &= ~near
      walled = self.grid.reshape(-1)[cells[near]] > 0.0
      free_cells = np.concatenate([cells[keep], cells[near][~walled]])
    else:
      free_cells = cells[keep]
    hit_cells = (hys * cols + hxs)[hin]
    weights = np.concatenate([np.full(free_cells.size, L_FREE),
                              np.full(hit_cells.size, L_OCC)])
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