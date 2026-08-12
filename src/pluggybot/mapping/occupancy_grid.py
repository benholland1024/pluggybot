
import numpy as np
import math


L_FREE = -0.4   # Amount to subtract when a ray passes through a cell
L_OCC = 0.85    # amount to add when a ray collides a cell


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
    
  def update(self, pose, angles, ranges, max_range, origin=(0.03, 0.03)):
    """Update the occupancy map. pose = (x, y, theta) at the AXLE midpoint.

    `origin` is where the sensor sits relative to that axle, in robot-frame
    (forward, left) metres. It defaults to the head camera's offset, which is
    what the plug-era Scanner uses; the hub robot's LIDAR sits elsewhere and
    passes its own (perception.lidar.LIDAR_ORIGIN). Getting this wrong slides
    the whole map by the difference, which is exactly the class of quiet
    error the rack-frame verdict bug was.
    """
    px, py, theta = pose
    fwd, left = origin
    ox = px + fwd * math.cos(theta) - left * math.sin(theta)
    oy = py + fwd * math.sin(theta) + left * math.cos(theta)

    rows, cols = self.grid.shape
    for angle, r in zip(angles,ranges):        # zip: merge two arrays in pairs
      a = theta + angle                        # ray direction in world frame
      hit = r < max_range - 1e-6               # max-range rays hit nothing!

      # -- free space: sample along the ray, stopping one cell short of the endpoint
      free_len = r - self.resolution if hit else r
      n = max(2, int(free_len / (self.resolution / 2)))
      ts = np.linspace(0.0, free_len, n)
      ixs = ((ox + ts * np.cos(a) - self.x_min) / self.resolution).astype(int)
      iys = ((oy + ts * np.sin(a) - self.y_min) / self.resolution).astype(int)
      valid = (ixs >= 0) & (ixs < cols) & (iys >= 0) & (iys < rows)    # No negative wrap arounds
      self.grid[iys[valid], ixs[valid]] += L_FREE

      # -- The hit itself
      if hit:
        ix, iy = self.world_to_cell(ox + r * math.cos(a), oy + r * math.sin(a))
        if 0 <= ix < cols and 0 <= iy < rows:
          self.grid[iy, ix] += L_OCC

    np.clip(self.grid, -5.0, 5.0, out=self.grid)


  def to_image(self) -> np.ndarray:             # uint8: 0 wall / 255 free / 127 unknown
    """Generate an image of the occupancy map"""
    img = np.full(self.grid.shape, 127, dtype=np.uint8)
    img[self.grid < -0.5] = 255    # confidently free
    img[self.grid > 0.5] = 0       # confidently occupied
    return img