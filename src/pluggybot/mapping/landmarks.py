"""Landmark memory: world positions of recognized outlets.

Landmarks are the sparse complement to the occupancy grid: a handful of
continuous (x, y, z) points rather than dense cells. Repeat sightings of the
same outlet (inevitable, since odometry drifts between visits) are merged by
nearest-neighbor gating, and each merge refines the stored position with a
running average. Sighting counts double as a confidence filter: a spurious
detection is seen once, a real outlet every time the robot looks at it.

Each landmark also averages WHERE the robot stood when it saw the outlet.
The mean seen-from point is necessarily out in free space on the outlet's
open side, so it hands the recharge behavior an approach direction without
any wall-normal geometry: see Landmark.standoff().
"""

import math

from pluggybot.mapping.frontier import FREE_THRESH

GATE_RADIUS = 0.4   # m: sightings closer than this (in 2D) are the same outlet.
                    # Odometry drift is <2% of path, outlets sit >1 m apart;
                    # z is excluded because it is the noisiest estimate and two
                    # outlets never share a wall spot at different heights.


class Landmark:
  def __init__(self, x: float, y: float, z: float,
               seen_from: tuple[float, float],
               recency: float | None = None) -> None:
    self.x = x
    self.y = y
    self.z = z
    self.n_sightings = 1     # a Landmark only exists because something was seen
    # Mean robot position across sightings (2D: the camera height is fixed,
    # so a seen-from z would be a constant, not information).
    self.seen_from_x, self.seen_from_y = seen_from
    #: Floor under the merge weight, or None for the pure running average.
    #: See `merge` -- this is what lets a landmark TRACK a drifting frame
    #: instead of remembering the mission-long mean of one (issue #42).
    self.recency = recency

  def merge(self, x: float, y: float, z: float,
            seen_from: tuple[float, float]) -> None:
    """Fold one new sighting into the estimates.

    Running average: after n sightings each stored value is the mean of all
    n, so a new observation moves it by 1/n of the residual — early (noisy,
    far-away) sightings get corrected, later ones only fine-tune.

    ⚠ With `recency` set, the weight FLOORS there instead of vanishing
    (issue #42). The pure average is the right estimator for a stationary
    world observed from a stationary frame — and this system's frame is dead
    reckoning, which drifts. Sightings are placed through the believed pose,
    so a landmark averaged over a whole mission remembers the MEAN historical
    frame; navigation happens in the CURRENT one, and the gap between the two
    is exactly the belief decoherence that dropped tools (issue #30 — a
    recovery spin's fresh sightings moved a long-run average by nothing,
    measured: the recovery regression test fails without this floor). The
    floor turns the estimate into an exponential moving average once enough
    sightings exist: recent looks dominate, and the belief follows the frame
    the robot is actually navigating in.
    """
    self.n_sightings += 1
    w = 1.0 / self.n_sightings
    if self.recency is not None:
      w = max(w, self.recency)
    self.x += (x - self.x) * w
    self.y += (y - self.y) * w
    self.z += (z - self.z) * w
    self.seen_from_x += (seen_from[0] - self.seen_from_x) * w
    self.seen_from_y += (seen_from[1] - self.seen_from_y) * w

  def standoff(self, distance: float = 0.6,
               direction: tuple[float, float] | None = None,
               ) -> tuple[float, float, float]:
    """The docking start pose: (x, y, heading) `distance` out from the
    outlet on its open side, facing it.

    `direction` is the outlet's outward normal. Prefer wall_normal() below,
    which reads it off the map. Without one, this falls back to the mean
    seen-from point — which only says where the robot happened to drive, and
    was measured 31 deg off the true normal after a drive-by. Good enough to
    tell which SIDE of the wall is open; not good enough to dock against.
    """
    if direction is None:
      dx, dy = self.seen_from_x - self.x, self.seen_from_y - self.y
    else:
      dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0    # degenerate only if seen from inside the wall
    sx = self.x + distance * dx / norm
    sy = self.y + distance * dy / norm
    return sx, sy, math.atan2(self.y - sy, self.x - sx)


def wall_normal(grid, x: float, y: float,
                fallback: tuple[float, float] = (1.0, 0.0),
                radii: tuple[float, ...] = (0.25, 0.35, 0.45),
                n_dirs: int = 48) -> tuple[float, float]:
  """Which way the wall under a landmark faces, read off the occupancy grid.

  Sum a unit vector toward every nearby cell the map calls confidently free.
  A wall blocks roughly half the circle, so the sum points out of it — no
  line fitting needed. Sampling several radii keeps one noisy ring dominating.

  `fallback` is the seen-from direction (outlet -> mean observer position). It
  does two jobs, and both survive the map being more precise:

    1. Fallback proper, when nothing nearby is known free — an outlet spotted
       from across the room may sit in territory never driven through.
    2. Sign check. A free-standing partition has open space on BOTH sides, so
       the sums nearly cancel and whichever direction survives may point at
       the wrong face. The robot physically saw this outlet, so the normal
       must lie in the hemisphere it was seen from; when the map disagrees,
       its axis is kept and the sign is flipped.
  """
  nx, ny, _ = wall_normal_conf(grid, x, y, fallback, radii, n_dirs)
  return nx, ny


def wall_normal_conf(grid, x: float, y: float,
                     fallback: tuple[float, float] = (1.0, 0.0),
                     radii: tuple[float, ...] = (0.25, 0.35, 0.45),
                     n_dirs: int = 48) -> tuple[float, float, float]:
  """`wall_normal`, plus HOW WELL CONDITIONED that answer is (0..1).

  The confidence is the free-space sum's magnitude divided by the number of
  free cells it summed: 1 means every free cell lies on one side, 0 means
  they cancel exactly. A wall blocks about half the circle, and the sum of a
  half-circle of unit vectors is 2/pi ~ 0.64 of its count -- so a real wall
  reads around 0.6 and a landmark with open space all round reads near 0.

  It exists because a nearly-cancelling sum is not a bad measurement, it is
  NOT A MEASUREMENT: the direction that survives is whatever noise was left
  over, and it can be tens of degrees from the wall it is supposed to
  describe. The docstring above has always said so about free-standing
  partitions; this is the number that lets a caller act on it.

  ⚠ It is not the same check as the sign flip. That one fixes a normal
  pointing at the wrong FACE of a correctly-found axis; this one says the
  axis itself is not there to be found. A caller that has a better answer
  already -- a remembered pose, a prior -- should keep it rather than adopt
  one of these (rack/localize.py, `RackFinder.estimate`).
  """
  fx = fy = 0.0
  free = 0
  rows, cols = grid.grid.shape
  for r in radii:
    for k in range(n_dirs):
      a = 2 * math.pi * k / n_dirs
      ix, iy = grid.world_to_cell(x + r * math.cos(a), y + r * math.sin(a))
      if 0 <= ix < cols and 0 <= iy < rows and grid.grid[iy, ix] < FREE_THRESH:
        fx += math.cos(a)
        fy += math.sin(a)
        free += 1
  norm = math.hypot(fx, fy)
  if norm <= 1e-6:
    return fallback[0], fallback[1], 0.0
  nx, ny = fx / norm, fy / norm
  if nx * fallback[0] + ny * fallback[1] < 0.0:
    nx, ny = -nx, -ny            # right wall, wrong face: flip to the seen side
  return nx, ny, norm / free if free else 0.0


class LandmarkStore:
  def __init__(self, gate_radius: float = GATE_RADIUS,
               recency: float | None = None) -> None:
    self.gate_radius = gate_radius
    #: Passed to every Landmark this store creates. None -- the default, and
    #: what the outlet map uses -- is the pure running average; the rack
    #: finder passes a floor so its one landmark tracks the current odometry
    #: frame (issue #42; the argument is on Landmark.merge).
    self.recency = recency
    self.landmarks: list[Landmark] = []

  def add_sighting(self, x: float, y: float, z: float,
                   seen_from: tuple[float, float]) -> Landmark:
    """Record one detection: merge into the nearest landmark within the gate
    (2D distance), or start a new landmark if none is close enough.
    seen_from: the robot's (x, y) at detection time."""
    nearest = None
    min_dist = self.gate_radius
    for landmark in self.landmarks:
      dist = math.hypot(landmark.x - x, landmark.y - y)
      if dist < min_dist:
        min_dist = dist
        nearest = landmark

    if nearest is None:
      nearest = Landmark(x, y, z, seen_from, recency=self.recency)
      self.landmarks.append(nearest)
    else:
      nearest.merge(x, y, z, seen_from)
    return nearest

  def confirmed(self, min_sightings: int = 3) -> list[Landmark]:
    """Landmarks seen at least min_sightings times: the trustworthy ones."""
    return [lm for lm in self.landmarks if lm.n_sightings >= min_sightings]

  def nearest_confirmed(self, x: float, y: float,
                        min_sightings: int = 3) -> Landmark | None:
    """The confirmed landmark closest (2D) to a world point — recharge mode's
    'which outlet do I go to'. None if nothing is confirmed yet."""
    candidates = self.confirmed(min_sightings)
    if not candidates:
      return None
    return min(candidates, key=lambda lm: math.hypot(lm.x - x, lm.y - y))
