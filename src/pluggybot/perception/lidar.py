"""The humanoid's LIDAR (issue #581): a Livox Mid-360S on its head, upside
down, modelled by ray casts. It replaced a 2D LIDAR spinning one plane 1.41
m up (the RPLIDAR C1), over the house's 0.9 m fences, which left a robot in
the street nothing to localize by (#572; SimNotes, "Where the humanoids get
lost").

Its sheet (livoxtech.com/mid-360s/specs): 360 deg round and -7 to +52 deg
up, 200,000 points a second in 10 Hz frames, a range's 1 sigma <= 2 cm at 10
m and <= 4 cm at 0.2 m, its angle's < 0.15 deg, a 0.1 m blind zone, 40 m at
10 % reflectivity. Upside down, as Unitree's G1 carries one, its field is
+7 down to -52 deg (`FOV_DEG`). A frame is modelled as rings `RING_DEG`
apart, `BEARINGS` bearings each -- its manual: "at 0.1 s, coverage is
similar to a 32-line mechanical LiDAR" -- 15 rings, 5,400 rays, where the
part spreads 20,000 points in a pattern that never repeats.

What it answers:

  band               the cloud laid flat: per bearing the nearest surface
                     `BAND_M` over the floor, levelled through the robot's
                     IMU -- the 2D scan the map is laid from and matched by
                     (#581's stage 2), with where on each bearing its free
                     run may begin (`Scan`)
  scan, scan_split   the LEVEL RING, its 360 bearings at 0 deg: the walls
                     a lost robot steps toward (`HumanoidBody.lost_routine`)
  cloud              every return, a point in the sensor's frame, the
                     room's and another robot's apart

Three things are deliberately honest rather than convenient:

  RAY CASTS, NOT A DEPTH IMAGE. `mj_multiRay` intersects real scene
  geometry, so there is no camera FOV ceiling and each ray has per-ray
  semantics.

  THE LIDAR SEES ITS OWN ROBOT. A ray that hits the robot first is DROPPED
  rather than reported, as is one inside the blind zone: no information,
  not a measurement of empty space, so its bearing returns nothing.

  NO RETURN IS A REAL READING. A ray that hits nothing within range reports
  max_range, and the grid treats that as free space along its length. That
  is different from a self-hit and must not be confused with one.
"""

import math
from typing import NamedTuple

import mujoco
import numpy as np

from pluggybot.perception.imu import seed_for

#: The Navigator's default for where a body's LIDAR rides from its pose
#: point, m forward and left (the rover's, `rover-final`); the humanoid
#: says its own, off its model (`HumanoidMission.LIDAR_ORIGIN`).
LIDAR_ORIGIN = (0.10, 0.0)
#: s between frames: the part's 10 Hz.
LIDAR_PERIOD = 0.1


#: 8 m, under the part's 40 m, on purpose. `OccupancyGrid.update` samples
#: every ray to its full length at half a cell, so max_range sets the per-scan
#: cost (360 rays x 320 samples at 8 m), and a no-return ray only clears free
#: space the robot will scan again as it walks. Under-ranging the part is the
#: conservative error -- a return past 8 m is dropped, never invented -- and
#: the matcher reads 6.5 m of it (`scan_match.MATCH_RANGE_M`). Raise it if a
#: world grows a sightline the map needs from a standstill.
MAX_RANGE = 8.0
#: Its field upside down, deg: the sheet's -7 to +52 turned over. A frame's
#: rings run from its lower edge, `RING_DEG` apart, `BEARINGS` round each:
#: 15 rings (Ben's call 3, #581: 30, 2 deg apart, ran the served pair at
#: 0.97-1.15x real time on the deploy box, 15 at 1.17-1.42x).
FOV_DEG = (-52.0, 7.0)
RING_DEG = 4.0
BEARINGS = 360
#: Nearer than this it returns nothing, m (the sheet; 0.1-0.2 m it detects,
#: its precision unpromised).
BLIND_M = 0.1
#: A range's 1 sigma, m: the sheet's two figures, <= 4 cm at 0.2 m and <= 2
#: cm at 10 m, joined by a line and held past either end.
SIGMA = ((0.2, 0.04), (10.0, 0.02))
#: Its angle's 1 sigma, on each of a ray's two angles: the sheet's < 0.15 deg.
ANGLE_SD = math.radians(0.15)
#: Returns lost to dark or shiny surfaces: the C1's model's 2 %, carried --
#: the sheet states only a false alarm rate (< 0.01 %), and nothing in the
#: house is past its 40 m at 10 % reflectivity.
DROPOUT = 0.02
#: The band laid flat, m over the floor (`band`): over the floor's points
#: by more than its noise, up to above the robot's head -- #572's probe's
#: band, which fixed the heading everywhere outdoors...
BAND_M = (0.15, 2.2)
#: ...its floor rising this much a metre out, for its levelling's error:
#: walking, the IMU's tilt is off by a median 0.8 deg, 1.8 at its 95th
#: percentile and 2.1-2.7 at its 99th (bowed, 0.6 at its 95th), and 1.8
#: deg lifts the floor 6.9 m out, where the -12 deg ring meets it, to 0.22
#: m. MEASURED over the loop's explore, a walk home and bows (#581, stage 2).
BAND_RISE = math.tan(math.radians(2.0))


def robot_geoms(model, root_name: str) -> set:
  """Every geom belonging to the robot, root body and all descendants.

  Needed because MuJoCo's ray `bodyexclude` skips a single body, while the
  robot is a tree -- its head, the LIDAR's mount, its arms and hands are
  separate bodies and all of them can stand in its rays (or the depth
  camera's frame, `perception/depth.py`, which filters with the same set).
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


def _sigma(dist: np.ndarray) -> np.ndarray:
  (r0, s0), (r1, s1) = SIGMA
  return np.interp(dist, (r0, r1), (s0, s1))


class Scan(NamedTuple):
  """One scan as the map and the front stop take it: the room's bearings
  and ranges, another robot's apart (issue #316), and how far out each of
  the room's bearings may first clear a cell the map holds a wall in, m
  from the scan's origin (`free_from`; None, from the sensor itself)."""
  angles: np.ndarray
  ranges: np.ndarray
  peer_angles: np.ndarray
  peer_ranges: np.ndarray
  free_from: np.ndarray | None = None


class Cloud(NamedTuple):
  """One frame's returns, in the sensor's frame (x ahead, z up the robot):
  the room's `points` and another robot's `peers`; `looked` is each ray (in
  `Lidar._dirs`' order) that went somewhere other than the robot itself,
  `seen` the bearings (`Lidar.ray_angles`) any of those was cast on, and
  the ray each of the room's points and the peers' came back along."""
  points: np.ndarray
  peers: np.ndarray
  seen: np.ndarray
  looked: np.ndarray
  rays: np.ndarray | None = None
  peer_rays: np.ndarray | None = None


class Lidar:
  """The Mid-360S over MuJoCo ray casts (the module docstring)."""

  def __init__(self, model, site_name: str = "lidar", robot_body: str = "pluggybot",
               seed: int | None = None, max_range: float = MAX_RANGE,
               noise: bool = True) -> None:
    self.model = model
    self.site_id = model.site(site_name).id
    self.max_range = max_range
    #: No range or angle noise and no dropouts: the geometry's truth, for a
    #: test or a probe.
    self.noise = noise
    self.dropout = DROPOUT if noise else 0.0
    #: Its noise is its own robot's (`imu.seed_for`, never `hash()`).
    seed = seed_for(f"lidar:{robot_body}") if seed is None else seed
    self.rng = np.random.default_rng(seed)
    #: ...and a SECOND stream for the returns that come back off another
    #: robot (issue #316): drawn from the room's stream, the peers' returns
    #: would move every reading the MAP takes in a pair world.
    self.peer_rng = np.random.default_rng(seed + 1)
    #: The level ring's bearings, the rings' elevations and every ray's
    #: direction in the sensor's frame (rings by bearings).
    self.ray_angles = np.linspace(-math.pi, math.pi, BEARINGS, endpoint=False)
    self.n_rays = BEARINGS
    self.elevations = np.radians(np.arange(FOV_DEG[0], FOV_DEG[1], RING_DEG))
    el, az = np.meshgrid(self.elevations, self.ray_angles, indexing="ij")
    self._el, self._az = el.reshape(-1), az.reshape(-1)
    self._dirs = _unit(self._el, self._az)
    self._ring = _unit(np.zeros(BEARINGS), self.ray_angles)
    self._bearing = np.tile(np.arange(BEARINGS), len(self.elevations))
    #: The sensor's place and turn in its robot's root frame: what `band`
    #: lays a point through.
    self.offset, self.turn = _site_in(model, self.site_id, robot_body)
    #: ...and the level ring's bearings in the robot's frame, its rays turned
    #: as the sensor sits on it (level as it sits: a tilted mount's ring is
    #: no level one, and is not modelled)
    turned = self._ring @ self.turn.T
    self._ring_angles = (self.ray_angles if np.array_equal(self.turn, np.eye(3))
                         else np.arctan2(turned[:, 1], turned[:, 0]))
    self._self_geoms = self._robot_geoms(model, robot_body)
    #: OTHER robots' geoms (issue #167), kept OUT OF THE MAP: a return off
    #: another robot is no information about the room, and painted in and
    #: inflated, a robot walking past walled in the one it passed.
    #: ⚠ OUT OF THE MAP IS NOT OUT OF THE SCAN (issue #316): those returns
    #: are real measurements and `scan_split` answers them separately, for
    #: the consumers that must see a robot -- the front stop above all.
    self._other_geoms: set = set()
    self._index_geoms()

  def _index_geoms(self) -> None:
    """The geom sets as arrays, for a scan's `np.isin`."""
    self._self_ids = np.fromiter(self._self_geoms, dtype=np.int32)
    self._other_ids = np.fromiter(self._other_geoms, dtype=np.int32)

  def exclude_robot(self, root_name: str) -> None:
    """Another robot's body out of the map's returns (see `_other_geoms`)."""
    self._other_geoms |= self._robot_geoms(self.model, root_name)
    self._index_geoms()

  @staticmethod
  def _robot_geoms(model, root_name: str) -> set:
    return robot_geoms(model, root_name)

  # ---- the casts -------------------------------------------------------------

  def _cast(self, data, dirs: np.ndarray):
    """Every ray of `dirs` (n, 3, the sensor's frame) in ONE `mj_multiRay`
    (issue #385): (distance, geom) a ray, -1 for nothing at all. ⚠ NO
    CUTOFF: MuJoCo culls a plane by its centre's distance, and at 8 m every
    ray that met the house's floor -- its centre 19 m off -- read nothing
    (`test_the_floor_is_struck_however_far_its_middle`); one saved no time."""
    pos = np.array(data.site_xpos[self.site_id], dtype=np.float64)
    mat = np.array(data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
    world = np.ascontiguousarray((dirs @ mat.T).reshape(-1))
    n = len(dirs)
    dist, hit = np.zeros(n), np.zeros(n, dtype=np.int32)
    mujoco.mj_multiRay(self.model, data, pos, world, None, 1, -1, hit, dist, None, n,
                       mujoco.mjMAXVAL)
    return dist, hit

  def _sort(self, dist: np.ndarray, hit: np.ndarray):
    """A cast's rays sorted: the robot's own and the blind zone's (no
    information), the far ones (nothing within reach), another robot's,
    and the room's returns."""
    struck = dist >= 0.0
    mine = struck & (np.isin(hit, self._self_ids) | (dist < BLIND_M))
    far = ~mine & (~struck | (dist >= self.max_range))
    peer = ~mine & ~far & np.isin(hit, self._other_ids)
    room = ~mine & ~far & ~peer
    return mine, far, peer, room

  def _ranges(self, dist: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """`dist`'s returns as the part reads them: which came back (its
    dropouts drawn first), and each one's range with its noise."""
    kept = rng.random(len(dist)) >= self.dropout if self.noise else np.ones(len(dist), bool)
    r = dist[kept]
    if self.noise:
      r = r + rng.normal(0.0, 1.0, len(r)) * _sigma(r)
    return kept, np.clip(r, BLIND_M, self.max_range)

  def scan(self, data) -> tuple[np.ndarray, np.ndarray]:
    """(angles, ranges) of the level ring in the ROBOT's frame; a bearing
    the robot hides from itself is absent. THE MAP's answer: another
    robot's returns are not in it (`scan_split` answers them)."""
    angles, ranges, _, _ = self.scan_split(data)
    return angles, ranges

  def scan_split(self, data) -> tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """The level ring, one set of casts and two honest answers (issue
    #316): `(angles, ranges, peer_angles, peer_ranges)` -- the room, then
    the bearings that came back off ANOTHER ROBOT. Short arrays where the
    robot hides a bearing from itself: "I cannot see that way" contributes
    nothing, never free space."""
    dist, hit = self._cast(data, self._ring)
    _, far, peer, room = self._sort(dist, hit)
    out = []
    for which, rng in ((room, self.rng), (peer, self.peer_rng)):
      idx = np.flatnonzero(which)
      kept, r = self._ranges(dist[idx], rng)
      idx = idx[kept]
      a = self._ring_angles[idx]
      if self.noise:
        a = a + rng.normal(0.0, ANGLE_SD, len(a))
      out.append((idx, a, r))
    (ri, ra, rr), (_, pa, pr) = out
    # ...and every bearing that returned nothing within reach: free to it
    fi = np.flatnonzero(far)
    order = np.argsort(np.concatenate((ri, fi)), kind="stable")
    angles = np.concatenate((ra, self._ring_angles[fi]))[order]
    ranges = np.concatenate((rr, np.full(len(fi), self.max_range)))[order]
    return angles, ranges, pa, pr

  def cloud(self, data) -> Cloud:
    """Every ring's returns as points in the sensor's frame, each at its
    noisy range along its noisy direction (`Cloud`)."""
    dist, hit = self._cast(data, self._dirs)
    mine, _, peer, room = self._sort(dist, hit)
    out, rays = [], []
    for which, rng in ((room, self.rng), (peer, self.peer_rng)):
      idx = np.flatnonzero(which)
      kept, r = self._ranges(dist[idx], rng)
      idx = idx[kept]
      el, az = self._el[idx], self._az[idx]
      if self.noise:
        jitter = rng.normal(0.0, ANGLE_SD, (2, len(idx)))
        el, az = el + jitter[0], az + jitter[1]
      out.append(_unit(el, az) * r[:, None])
      rays.append(idx)
    points, peers = out
    seen = np.zeros(BEARINGS, dtype=bool)
    seen[self._bearing[~mine]] = True
    return Cloud(points, peers, seen, ~mine, rays[0], rays[1])

  def band(self, data, level: np.ndarray, height: float,
           band_m: tuple[float, float] = BAND_M, rise: float = BAND_RISE) -> Scan:
    """The cloud laid flat into a 2D scan (`Scan`): each point levelled
    through `level` (the robot's tilt, level from its root body: its IMU's,
    `imu.Attitude.level`) and set `height` over the floor (its root's, off
    its own model), and per bearing -- the point's own, levelled, from the
    scan's origin (the sensor's place on the robot, unlevelled: what the map
    lays a scan from) -- the nearest whose height is within `band_m`, its
    floor `rise` higher a metre out, read as the median of its returns
    within 3 sigma of that nearest. A bearing with none there is free to
    max range where a ray, levelled, looked along it, staying in the band
    that far, and brought back nothing in it; else absent. ⚠ BINNED
    BY THE BEARING LEVELLED, never the one a ray was cast on: tilted, a
    steep ray levels a degree or more aside, and a free ray laid down a
    neighbour's return would clear the very cell it struck -- and NO RAY
    FREES A BEARING THAT BROUGHT BACK A POINT IN THE BAND, OR LEFT THE BAND
    SHORT OF MAX RANGE: behind the robot, where the bracket's post hides the
    rest of a bearing's rays, one whose point jittered a bearing over, or a
    steep one that saw the band empty for a metre and met the floor, laid
    its bearing free to 8 m through the wall behind, every bearing round as
    the robot turned. ⚠ THE MEDIAN,
    NEVER THE NEAREST: a dozen rings strike one wall, and the least of a
    dozen noisy ranges read every surface 1.7 sigma short, 5-7 cm, a cell
    into the room each side. ⚠ A BEARING CLEARS A WALL ONLY FROM WHERE ITS
    STEEPEST RAY COMES DOWN THROUGH THE BAND'S FLOOR (`free_from`, read by
    `OccupancyGrid.update`): nearer, every ray passes over whatever is lower
    than it, and a bearing freed from the sensor out cleared a low thing
    mapped from afar as the robot walked up to it."""
    c = self.cloud(data)
    rot = np.asarray(level, dtype=float) @ self.turn
    base = np.asarray(level, dtype=float) @ self.offset
    lo, hi = band_m

    def flat(points):
      q = points @ rot.T + base
      z = height + q[:, 2]
      v = q[:, :2] - self.offset[:2]
      r = np.hypot(v[:, 0], v[:, 1])
      keep = (z >= lo + rise * r) & (z <= hi) & (r < self.max_range)
      banded = keep
      a, r = np.arctan2(v[keep, 1], v[keep, 0]), r[keep]
      b = _bin(a)
      near = np.full(BEARINGS, np.inf)
      np.minimum.at(near, b, r)
      within = r <= near[b] + 3.0 * _sigma(near[b])
      b, r, a = b[within], r[within], a[within]
      order = np.lexsort((r, b))
      b, r, a = b[order], r[order], a[order]
      count = np.bincount(b, minlength=BEARINGS)
      have = count > 0
      first = np.cumsum(count) - count
      m0, m1 = (first + (count - 1) // 2)[have], (first + count // 2)[have]
      # ...the median point's own angle kept, and which points were in the band
      return have, a[m0], (r[m0] + r[m1]) / 2, banded

    have, angles, ranges, banded = flat(c.points)
    p_have, peer_angles, peer_ranges, p_banded = flat(c.peers)
    quiet = c.looked.copy()
    quiet[c.rays[banded]] = False
    quiet[c.peer_rays[p_banded]] = False
    d = self._dirs[c.looked] @ rot.T
    bins = _bin(np.arctan2(d[:, 1], d[:, 0]))
    # where each ray that looked leaves the band, from the scan's origin:
    # down through its floor, or up through its top
    flat_d = np.hypot(d[:, 0], d[:, 1])
    over = height + base[2]
    falls = -d[:, 2] - rise * flat_d
    down, up = falls > 1e-9, d[:, 2] > 1e-9
    t = np.full(len(d), np.inf)
    t[down] = max(over - lo, 0.0) / falls[down]
    t[up] = max(hi - over, 0.0) / d[up, 2]
    leaves = np.full(len(d), np.inf)
    went = np.isfinite(t)
    leaves[went] = np.hypot(base[0] + t[went] * d[went, 0] - self.offset[0],
                            base[1] + t[went] * d[went, 1] - self.offset[1])
    seen = np.zeros(BEARINGS, dtype=bool)
    seen[bins[quiet[c.looked] & (leaves >= self.max_range)]] = True
    # ...and where a bearing's steepest ray first comes down through the
    # band's floor -- one that never does vouches for none of it
    free_from = np.full(BEARINGS, np.inf)
    np.minimum.at(free_from, bins[down], leaves[down])
    free = seen & ~have & ~p_have
    fa = self.ray_angles[free]
    a = np.concatenate((angles, fa))
    idx = np.concatenate((np.flatnonzero(have), np.flatnonzero(free)))
    order = np.argsort(idx, kind="stable")
    r = np.concatenate((ranges, np.full(len(fa), self.max_range)))
    return Scan(a[order], r[order], peer_angles, peer_ranges, free_from[idx][order])

  def hidden(self, data) -> np.ndarray:
    """Which of a frame's rays the robot's own body hides, rings by
    bearings (`elevations` by `ray_angles`) -- a mounting metric
    (`lidar_spike.py --hidden`)."""
    dist, hit = self._cast(data, self._dirs)
    mine = (dist >= 0.0) & np.isin(hit, self._self_ids)
    return mine.reshape(len(self.elevations), BEARINGS)

  def blind_fraction(self, data) -> float:
    """Share of a frame's rays the robot's own body hides."""
    return float(self.hidden(data).mean())


def _bin(a: np.ndarray) -> np.ndarray:
  """Bearings, rad, to the nearest of `Lidar.ray_angles`' indices."""
  return np.round((np.asarray(a) + math.pi) / (2 * math.pi / BEARINGS)).astype(np.int64) % BEARINGS


def _unit(el: np.ndarray, az: np.ndarray) -> np.ndarray:
  """Directions at elevations `el` and bearings `az`, rad: (n, 3)."""
  c = np.cos(el)
  return np.stack([c * np.cos(az), c * np.sin(az), np.sin(el)], axis=1)


def _site_in(model, site: int, root_name: str) -> tuple[np.ndarray, np.ndarray]:
  """A site's place and turn in its robot's root body's frame, off the
  model: (3,) and (3, 3); the world's frame where there is no such body."""
  pos = np.array(model.site_pos[site], dtype=float)
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, model.site_quat[site])
  rot = mat.reshape(3, 3).copy()
  try:
    root = model.body(root_name).id
  except KeyError:
    root = 0
  body = int(model.site_bodyid[site])
  while body not in (root, 0):
    mujoco.mju_quat2Mat(mat, model.body_quat[body])
    m = mat.reshape(3, 3).copy()
    pos = model.body_pos[body] + m @ pos
    rot = m @ rot
    body = int(model.body_parentid[body])
  return pos, rot
