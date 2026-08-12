"""2D scanning LIDAR, modelled by ray casts (sensor-realism pass, Aug 2026).

Replaces the stereo pair as the mapper's range source. The measurement that
forced this (docs/SimNotes.md): running real SGBM on the sim's own stereo pair,
mid-room facing a painted wall, produced disparity for only **49.7 %** of the
scan row and was **593 mm** out at the median — against a 50 mm grid cell.
Flat painted walls are the classic no-disparity case, and this room is made of
them. Stereo cannot build this map.

The swap is cheap because the abstraction was already right: `Scanner.scan()`
has always returned `(angles, ranges)`, which is a laser scan's interface, and
`scanner.py` was never actually stereo — it read a depth image from ONE camera
and took the centre row. The mapping has been a 2D laser scan wearing a
camera's clothes since milestone 4.

Modelled on an RPLIDAR C1-class unit: 360°, ~10 Hz, 12 m, ±30 mm.

Three things here are deliberately honest rather than convenient:

  RAY CASTS, NOT A DEPTH IMAGE. `mj_ray` intersects real scene geometry, so
  there is no camera FOV ceiling and each ray has proper per-ray semantics.
  A depth-image row could never have produced 360° anyway.

  THE LIDAR SEES ITS OWN ROBOT. The mast stands in the scan plane, so rays
  that hit the robot are dropped rather than reported — which is exactly the
  self-filter a real unit needs, and it means those bearings return NOTHING
  rather than a false free-space reading. Dropping them (returning shorter
  arrays) is what keeps the grid honest: a self-hit is an absence of
  information, not a measurement of empty space.

  NO RETURN IS A REAL READING. A ray that hits nothing within range reports
  max_range, and the grid treats that as free space along its length. That is
  different from a self-hit and must not be confused with one.
"""

import math

import mujoco
import numpy as np

# Where the unit sits relative to the AXLE midpoint (which is what odometry
# tracks), in robot-frame metres: forward, left. The body origin is 8 cm ahead
# of the axle, and the site is 2 cm ahead of the body origin.
LIDAR_ORIGIN = (0.10, 0.0)
LIDAR_PERIOD = 0.1        # s between scans: 10 Hz, the part's real rate. The
                          # camera scanner ran at 50 Hz because a depth render
                          # is free in sim; a spinning mirror is not.


class Lidar:
  """360-degree planar LIDAR over MuJoCo ray casts."""

  def __init__(self, model, site_name: str = "lidar", n_rays: int = 360,
               max_range: float = 8.0, sigma_m: float = 0.010,
               sigma_frac: float = 0.01, dropout: float = 0.02,
               robot_body: str = "pluggybot", seed: int = 0) -> None:
    self.model = model
    self.site_id = model.site(site_name).id
    self.n_rays = n_rays
    self.max_range = max_range
    self.sigma_m = sigma_m          # fixed term: ±10 mm
    self.sigma_frac = sigma_frac    # proportional term: ±1 % of range
    self.dropout = dropout          # returns lost to dark/specular surfaces
    self.rng = np.random.default_rng(seed)
    self.ray_angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
    self._dirs = np.stack([np.cos(self.ray_angles),
                           np.sin(self.ray_angles),
                           np.zeros(n_rays)], axis=1)
    self._self_geoms = self._robot_geoms(model, robot_body)
    self._geomid = np.zeros(1, dtype=np.int32)

  @staticmethod
  def _robot_geoms(model, root_name: str) -> set:
    """Every geom belonging to the robot, root body and all descendants.

    Needed because MuJoCo's ray `bodyexclude` skips a single body, while the
    robot is a tree -- mast, head, carriage, arm and fork are separate bodies
    and all of them can stand in the scan plane.
    """
    try:
      root = model.body(root_name).id
    except KeyError:
      return set()
    bodies = set()
    for b in range(model.nbody):
      p = b
      while p > 0:
        if p == root:
          bodies.add(b)
          break
        p = int(model.body_parentid[p])
    bodies.add(root)
    return {g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in bodies}

  def scan(self, data) -> tuple[np.ndarray, np.ndarray]:
    """(angles, ranges) in the ROBOT's frame; self-occluded bearings absent.

    The returned arrays are shorter than n_rays whenever the robot blocks its
    own view. Consumers zip the two together, so a dropped bearing simply
    contributes nothing -- which is the correct treatment of "I cannot see
    that way", and is why this returns short arrays rather than padding with
    max_range.
    """
    pos = np.array(data.site_xpos[self.site_id], dtype=np.float64)
    mat = np.array(data.site_xmat[self.site_id],
                   dtype=np.float64).reshape(3, 3)
    world_dirs = self._dirs @ mat.T          # site frame -> world

    angles, ranges = [], []
    for i in range(self.n_rays):
      vec = np.ascontiguousarray(world_dirs[i])
      dist = mujoco.mj_ray(self.model, data, pos, vec, None, 1, -1,
                           self._geomid)
      if dist >= 0.0 and int(self._geomid[0]) in self._self_geoms:
        continue                              # self-filter: no information
      if dist < 0.0 or dist >= self.max_range:
        angles.append(self.ray_angles[i])     # nothing out there: free to max
        ranges.append(self.max_range)
        continue
      if self.rng.random() < self.dropout:
        continue                              # surface gave no return
      noisy = dist + self.rng.normal(
        0.0, self.sigma_m + self.sigma_frac * dist)
      angles.append(self.ray_angles[i])
      ranges.append(float(np.clip(noisy, 0.02, self.max_range)))
    return np.asarray(angles), np.asarray(ranges)

  def blind_fraction(self, data) -> float:
    """Share of bearings the robot's own body occludes -- a mounting metric."""
    angles, _ = self.scan(data)
    return 1.0 - len(angles) / self.n_rays
