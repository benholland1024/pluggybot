"""2D scanning LIDAR, modelled by ray casts.

The mapper's range source since the sensor-realism pass (Aug 2026), when it
replaced a stereo pair that measured unable to build the map (docs/Parts.md
"Vision & ranging" keeps the decision and the numbers). The camera scanner
it replaced already returned `(angles, ranges)`, a laser scan's interface, so
the swap changed the sensor and not the mapping.

Modelled on an RPLIDAR C1-class unit: 360°, ~10 Hz, 12 m, ±30 mm.

Three things here are deliberately honest rather than convenient:

  RAY CASTS, NOT A DEPTH IMAGE. `mj_multiRay` intersects real scene geometry, so
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


#: 8 m, under the part's 12 m, on purpose. `OccupancyGrid.update` samples
#: every ray to its full length at half a cell, so max_range sets the per-scan
#: cost (360 rays x 320 samples at 8 m, 480 at 12 m), and a no-return ray only
#: clears free space the robot will scan again as it drives. Under-ranging the
#: part is the conservative error -- a return past 8 m is dropped, never
#: invented -- and it has not been measured to cost the map anything. Raise
#: it if a world grows a sightline the map needs from a standstill.
MAX_RANGE = 8.0


def robot_geoms(model, root_name: str) -> set:
  """Every geom belonging to the robot, root body and all descendants.

  Needed because MuJoCo's ray `bodyexclude` skips a single body, while the
  robot is a tree -- mast, head, carriage, arm and fork are separate bodies
  and all of them can stand in the scan plane (or the depth camera's frame,
  `perception/depth.py`, which filters with the same set).
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


class Lidar:
  """360-degree planar LIDAR over MuJoCo ray casts."""

  def __init__(self, model, site_name: str = "lidar", n_rays: int = 360,
               max_range: float = MAX_RANGE, sigma_m: float = 0.010,
               sigma_frac: float = 0.01, dropout: float = 0.02,
               robot_body: str = "pluggybot", seed: int = 0) -> None:
    self.model = model
    self.site_name, self.robot_body = site_name, robot_body
    self.site_id = model.site(site_name).id
    self.n_rays = n_rays
    self.max_range = max_range
    self.sigma_m = sigma_m          # fixed term: ±10 mm
    self.sigma_frac = sigma_frac    # proportional term: ±1 % of range
    self.dropout = dropout          # returns lost to dark/specular surfaces
    self.rng = np.random.default_rng(seed)
    #: ...and a SECOND stream for the returns that come back off another
    #: robot (issue #316). A peer return is as noisy as any other -- it is
    #: a measurement -- but drawing its noise from the room's stream would
    #: have moved every reading the MAP takes in a pair world the moment
    #: peers stopped being thrown away, and then no pair flight could say
    #: whether it changed because the reflex now brakes or because the
    #: sensor rolled different numbers. MEASURED both ways: with one
    #: stream the map's 20-scan hash moved, with two it is byte-identical
    #: to the scan before this issue. Per-ray noise is IID either way, so
    #: nothing about the sensor's honesty rests on which stream it is
    #: drawn from.
    self.peer_rng = np.random.default_rng(seed + 1)
    self.ray_angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
    self._dirs = np.stack([np.cos(self.ray_angles),
                           np.sin(self.ray_angles),
                           np.zeros(n_rays)], axis=1)
    self._self_geoms = self._robot_geoms(model, robot_body)
    #: OTHER robots' geoms (issue #167), kept OUT OF THE MAP: a bearing
    #: that hit another robot is no information about the room. A fleet
    #: does this by subtracting each robot's broadcast pose and footprint
    #: from the scan; here the pose is exact and the footprint is the body
    #: itself. Without it a robot that drove past painted a wake of
    #: occupied cells into the map, inflated to 0.35 m, and the robot it
    #: passed was walled in by the ghost (measured: robot 1 planned None
    #: from its own cell for 12 s and gave up the bay).
    #: ⚠ OUT OF THE MAP IS NOT OUT OF THE SCAN (issue #316): those returns
    #: are real measurements and `scan_split` answers them separately, for
    #: the consumers that must see a robot -- the front-stop reflex above
    #: all. Dropping them outright left the one moving obstacle in the
    #: world invisible to every sensor on board.
    self._other_geoms: set = set()
    self._other_roots: list[str] = []
    self._dist = np.zeros(n_rays)
    self._hit = np.zeros(n_rays, dtype=np.int32)
    self._index_geoms()

  def _index_geoms(self) -> None:
    """The two geom sets as arrays, for the scan's `np.isin`."""
    self._self_ids = np.fromiter(self._self_geoms, dtype=np.int32)
    self._other_ids = np.fromiter(self._other_geoms, dtype=np.int32)

  def rebind(self, model) -> None:
    """A recompiled world: the site and the geom sets by name again
    (issue #168 slice C). The excluded OTHER robots are re-resolved too."""
    self.model = model
    self.site_id = model.site(self.site_name).id
    self._self_geoms = self._robot_geoms(model, self.robot_body)
    self._other_geoms = set()
    for root in list(self._other_roots):
      self._other_geoms |= self._robot_geoms(model, root)
    self._index_geoms()

  def exclude_robot(self, root_name: str) -> None:
    """Drop another robot's body from every scan (see `_other_geoms`)."""
    self._other_roots.append(root_name)
    self._other_geoms |= self._robot_geoms(self.model, root_name)
    self._index_geoms()

  @staticmethod
  def _robot_geoms(model, root_name: str) -> set:
    return robot_geoms(model, root_name)

  def scan(self, data) -> tuple[np.ndarray, np.ndarray]:
    """(angles, ranges) in the ROBOT's frame; self-occluded bearings absent.

    The returned arrays are shorter than n_rays whenever the robot blocks its
    own view. Consumers zip the two together, so a dropped bearing simply
    contributes nothing -- which is the correct treatment of "I cannot see
    that way", and is why this returns short arrays rather than padding with
    max_range.

    THE MAP's answer: another robot's returns are not in it. A consumer
    that must see a robot -- the front-stop reflex -- takes all four arrays
    from `scan_split` instead.
    """
    angles, ranges, _, _ = self.scan_split(data)
    return angles, ranges

  def scan_split(self, data) -> tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """One set of casts, two honest answers (issue #316):
    `(angles, ranges, peer_angles, peer_ranges)` -- the room, then the
    bearings that came back off ANOTHER ROBOT.

    The split is here and not at the consumers because the two needs are
    opposite and both are right: the map must not contain the other robot
    (issue #167, `_other_geoms`), and the reflex that stops this one short
    of a wall must. One scan fed both for as long as the exclusion existed,
    which made the only moving obstacle in the world the one thing no
    sensor on board could see. A peer return is measured exactly like any
    other -- same dropout, same noise -- because it IS one; what it is not
    is a fact about the room.
    """
    pos = np.array(data.site_xpos[self.site_id], dtype=np.float64)
    mat = np.array(data.site_xmat[self.site_id],
                   dtype=np.float64).reshape(3, 3)
    world_dirs = np.ascontiguousarray((self._dirs @ mat.T).reshape(-1))
    # ONE call for the whole scan (issue #385): every ray's distance and
    # geom are bit-identical to `mj_ray`'s one at a time (108 000 rays at
    # 300 poses in the home world), for a third of the cost. No cutoff, so
    # the call is the only thing that changed.
    dist, hit = self._dist, self._hit
    mujoco.mj_multiRay(self.model, data, pos, world_dirs, None, 1, -1,
                       hit, dist, None, self.n_rays, mujoco.mjMAXVAL)
    struck = dist >= 0.0
    mine = struck & np.isin(hit, self._self_ids)   # self-filter: no information
    far = ~mine & (~struck | (dist >= self.max_range))  # nothing: free to max
    peer = ~mine & ~far & np.isin(hit, self._other_ids)
    kept = far.copy()
    ranges = np.full(self.n_rays, self.max_range)
    # The noise is still drawn ray by ray, in bearing order, from each
    # stream in turn: a batched draw takes the same numbers in another
    # order, and no day flown before it would hash the same again.
    sd = self.sigma_m + self.sigma_frac * dist
    for i in np.flatnonzero(~mine & ~far):
      rng = self.peer_rng if peer[i] else self.rng
      if rng.random() < self.dropout:
        continue                              # surface gave no return
      ranges[i] = dist[i] + rng.normal(0.0, sd[i])
      kept[i] = True
    np.clip(ranges, 0.02, self.max_range, out=ranges)
    room, theirs = kept & ~peer, kept & peer
    return (self.ray_angles[room], ranges[room],
            self.ray_angles[theirs], ranges[theirs])

  def blind_fraction(self, data) -> float:
    """Share of bearings the robot's own body occludes -- a mounting metric."""
    angles, _ = self.scan(data)
    return 1.0 - len(angles) / self.n_rays
