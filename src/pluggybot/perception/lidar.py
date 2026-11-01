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
similar to a 32-line mechanical LiDAR" -- 30 rings, 10,800 rays, where the
part spreads 20,000 points in a pattern that never repeats.

What it answers:

  scan, scan_split   the LEVEL RING, its 360 bearings at 0 deg: the 2D scan
                     the map is laid from -- the C1's plane, from this part
                     (#581's stage 1; its stage 2 lays the band instead)
  cloud              every return, a point in the sensor's frame, the
                     room's and another robot's apart
  band               the cloud laid flat: per bearing the nearest return
                     `BAND_M` over the floor, levelled through the robot's
                     IMU -- a 2D scan from every ring (flown, not served)

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
#: rings run from its lower edge, `RING_DEG` apart, `BEARINGS` round each.
FOV_DEG = (-52.0, 7.0)
RING_DEG = 2.0
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
#: by more than its noise and its levelling's error at range, up to above
#: the robot's head -- #572's probe's band, which fixed the heading
#: everywhere outdoors.
BAND_M = (0.15, 2.2)


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


class Cloud(NamedTuple):
  """One frame's returns, in the sensor's frame (x ahead, z up the robot):
  the room's `points` and another robot's `peers`, each with the bearing it
  was cast on (an index into `Lidar.ray_angles`); `seen` is the bearings
  any ray of went somewhere other than the robot itself."""
  points: np.ndarray
  bearings: np.ndarray
  peers: np.ndarray
  peer_bearings: np.ndarray
  seen: np.ndarray


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
    self._self_geoms = self._robot_geoms(model, robot_body)
    #: OTHER robots' geoms (issue #167), kept OUT OF THE MAP: a return off
    #: another robot is no information about the room, and painted in and
    #: inflated, a robot walking past walled in the one it passed.
    #: ⚠ OUT OF THE MAP IS NOT OUT OF THE SCAN (issue #316): those returns
    #: are real measurements and `scan_split` answers them separately, for
    #: the consumers that must see a robot -- the front stop above all.
    self._other_geoms: set = set()
    #: Something this robot carries: its own for the scan, as its body is.
    self._carried_geoms: set = set()
    self._index_geoms()

  def _index_geoms(self) -> None:
    """The geom sets as arrays, for a scan's `np.isin`."""
    self._self_ids = np.fromiter(self._self_geoms | self._carried_geoms, dtype=np.int32)
    self._other_ids = np.fromiter(self._other_geoms, dtype=np.int32)

  def carry(self, body: str | None) -> None:
    """A body this robot carries is its own to the scan, or None."""
    self._carried_geoms = set() if body is None else self._robot_geoms(self.model, body)
    self._index_geoms()

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
    (issue #385): (distance, geom) a ray, -1 for nothing within reach."""
    pos = np.array(data.site_xpos[self.site_id], dtype=np.float64)
    mat = np.array(data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
    world = np.ascontiguousarray((dirs @ mat.T).reshape(-1))
    n = len(dirs)
    dist, hit = np.zeros(n), np.zeros(n, dtype=np.int32)
    mujoco.mj_multiRay(self.model, data, pos, world, None, 1, -1, hit, dist, None, n,
                       self.max_range)
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
      a = self.ray_angles[idx]
      if self.noise:
        a = a + rng.normal(0.0, ANGLE_SD, len(a))
      out.append((idx, a, r))
    (ri, ra, rr), (_, pa, pr) = out
    # ...and every bearing that returned nothing within reach: free to it
    fi = np.flatnonzero(far)
    order = np.argsort(np.concatenate((ri, fi)), kind="stable")
    angles = np.concatenate((ra, self.ray_angles[fi]))[order]
    ranges = np.concatenate((rr, np.full(len(fi), self.max_range)))[order]
    return angles, ranges, pa, pr

  def cloud(self, data) -> Cloud:
    """Every ring's returns as points in the sensor's frame, each at its
    noisy range along its noisy direction (`Cloud`)."""
    dist, hit = self._cast(data, self._dirs)
    mine, _, peer, room = self._sort(dist, hit)
    out = []
    for which, rng in ((room, self.rng), (peer, self.peer_rng)):
      idx = np.flatnonzero(which)
      kept, r = self._ranges(dist[idx], rng)
      idx = idx[kept]
      el, az = self._el[idx], self._az[idx]
      if self.noise:
        jitter = rng.normal(0.0, ANGLE_SD, (2, len(idx)))
        el, az = el + jitter[0], az + jitter[1]
      out.append((_unit(el, az) * r[:, None], self._bearing[idx]))
    (points, bearings), (peers, peer_bearings) = out
    seen = np.zeros(BEARINGS, dtype=bool)
    seen[self._bearing[~mine]] = True
    return Cloud(points, bearings, peers, peer_bearings, seen)

  def band(self, data, level: np.ndarray, height: float,
           band_m: tuple[float, float] = BAND_M) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """The cloud laid flat into a 2D scan, `scan_split`'s four arrays: each
    point levelled through `level` (the robot's tilt, level from its root
    body: its IMU's, `imu.Attitude.level`) and set `height` over the floor
    (its root's, off its own model), and per bearing the nearest whose
    height is within `band_m`, its bearing and range from the scan's
    origin (the sensor's place on the robot, unlevelled: what the map lays
    a scan from). A bearing with none there that saw anything is free to
    max range; one the robot hid from itself, absent."""
    c = self.cloud(data)
    rot = np.asarray(level, dtype=float) @ self.turn
    base = np.asarray(level, dtype=float) @ self.offset
    lo, hi = band_m

    def flat(points, bearings):
      q = points @ rot.T + base
      z = height + q[:, 2]
      keep = (z >= lo) & (z <= hi)
      v = q[keep, :2] - self.offset[:2]
      r = np.hypot(v[:, 0], v[:, 1])
      keep_r = r < self.max_range
      v, r, b = v[keep_r], r[keep_r], bearings[keep][keep_r]
      a = np.arctan2(v[:, 1], v[:, 0])
      best = np.full(BEARINGS, np.inf)
      np.minimum.at(best, b, r)
      first = np.isfinite(best)
      # ...each bearing's nearest point, its own angle kept
      pick = np.zeros(BEARINGS, dtype=np.int64)
      on = np.flatnonzero(r == best[b])
      pick[b[on]] = on
      return first, a[pick[first]], best[first]

    have, angles, ranges = flat(c.points, c.bearings)
    p_have, peer_angles, peer_ranges = flat(c.peers, c.peer_bearings)
    free = c.seen & ~have & ~p_have
    fa = self.ray_angles[free]
    a = np.concatenate((angles, fa))
    order = np.argsort(np.concatenate((np.flatnonzero(have), np.flatnonzero(free))),
                       kind="stable")
    r = np.concatenate((ranges, np.full(len(fa), self.max_range)))
    return a[order], r[order], peer_angles, peer_ranges

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
  rot = mat.reshape(3, 3)
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
