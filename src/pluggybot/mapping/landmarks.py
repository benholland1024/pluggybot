"""Landmark memory: world positions of things the robot recognized -- today
the rack's AprilTag (rack/localize.py, `RackFinder`).

Landmarks are the sparse complement to the occupancy grid: a handful of
continuous (x, y, z) points rather than dense cells. Repeat sightings of the
same thing (inevitable, since odometry drifts between visits) are merged, and
each merge refines the stored position with a running average. Sighting
counts double as a confidence filter: a spurious detection is seen once, a
real landmark every time the robot looks at it.

Each landmark also averages WHERE the robot stood when it saw it. The mean
seen-from point is necessarily out in free space on the landmark's open
side, which is `wall_normal_conf`'s fallback and its sign check.
"""

import math

from pluggybot.mapping.frontier import FREE_THRESH


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


def wall_normal_conf(grid, x: float, y: float,
                     fallback: tuple[float, float] = (1.0, 0.0),
                     radii: tuple[float, ...] = (0.25, 0.35, 0.45),
                     n_dirs: int = 48) -> tuple[float, float, float]:
  """Which way the wall under a landmark faces, read off the occupancy grid,
  and HOW WELL CONDITIONED that answer is (0..1).

  Sum a unit vector toward every nearby cell the map calls confidently free.
  A wall blocks roughly half the circle, so the sum points out of it -- no
  line fitting needed. Sampling several radii keeps one noisy ring from
  dominating.

  `fallback` is the seen-from direction (landmark -> mean observer
  position). It does two jobs:

    1. Fallback proper, when nothing nearby is known free -- a landmark
       spotted from across the room may sit in territory never driven
       through.
    2. Sign check. A free-standing partition has open space on BOTH sides,
       so the sums nearly cancel and whichever direction survives may point
       at the wrong face. The robot physically saw this landmark, so the
       normal must lie in the hemisphere it was seen from; when the map
       disagrees, its axis is kept and the sign is flipped.

  The confidence is the free-space sum's magnitude divided by the number of
  free cells it summed: 1 means every free cell lies on one side, 0 means
  they cancel exactly. A wall blocks about half the circle, and the sum of a
  half-circle of unit vectors is 2/pi ~ 0.64 of its count -- so a real wall
  reads around 0.6 and a landmark with open space all round reads near 0.

  It exists because a nearly-cancelling sum is not a bad measurement, it is
  NOT A MEASUREMENT: the direction that survives is whatever noise was left
  over, and it can be tens of degrees from the wall it is supposed to
  describe.

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
  def __init__(self, recency: float | None = None) -> None:
    #: Passed to every Landmark this store creates. None -- the default -- is
    #: the pure running average; the rack finder passes a floor so its one
    #: landmark tracks the current odometry frame (issue #42; the argument
    #: is on Landmark.merge).
    self.recency = recency
    self.landmarks: list[Landmark] = []

  def add_sighting(self, x: float, y: float, z: float,
                   seen_from: tuple[float, float]) -> Landmark:
    """Start a landmark from its first sighting. Later sightings are merged
    into it by whoever knows what they are (the rack's finder merges by
    decoded tag id, never by distance). seen_from: the robot's (x, y)."""
    landmark = Landmark(x, y, z, seen_from, recency=self.recency)
    self.landmarks.append(landmark)
    return landmark
