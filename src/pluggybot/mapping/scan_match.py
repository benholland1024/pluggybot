"""Scan matching (issue #386): each LIDAR scan is aligned against the robot's
own map BEFORE it is fused, and the pose it aligns to is the one the map is
laid through, the planner plans from and the other robot is told.

Hector SLAM's method (Kohlbrecher et al. 2011): Gauss-Newton on a smooth
function of the map -- here the SIGNED DISTANCE to the nearest wall, a
chamfer transform of the occupied cells read between cell centres, so a
point pulls on the pose from up to `REACH_M` away and its residual is
metres. Odometry's prediction is a term of the same fit (a robust prior),
the scan may fix only the directions its walls face (`DEGENERATE`), a fit
the map disagrees with is past the field's reach and a bounded search
finds where to fit from (`_search`), a robot its map keeps refusing is
lost past that search and searches wider (`_relocate`, issue #422), and a
scan goes into the map only when a fit agreed and the robot has moved
(`Match.fuse`, `ScanMatcher.fuses`). docs/SimNotes.md, "The map stays true
under drift" and "Lost after a long explore, and found again", has why
each piece is there and what it measured.

Deterministic: no threads, and every sum is numpy's own loop in a fixed
order -- never a BLAS product, whose summation order is the library's.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from pluggybot.mapping.frontier import OCC_THRESH

#: How far a scan point is pulled from, m: past it the field is flat and a
#: point pulls on nothing -- new geometry, a door that opened, a moved bin.
REACH_M = 0.30
#: Unseen cells this close behind a wall are its inside, cells: a thin wall
#: is one cell, and its returns land on both sides of that cell's centre.
BEHIND_CELLS = 2.0
#: Returns at or past this range are left out, m: the map's own reach
#: (`lidar.MAX_RANGE`) since #560, the park's middle 8 m from its nearest
#: fence. ⚠ A WALL AT THE REACH IS SEEN SHORT where its long draws are
#: clipped to "no return": the C1 2D LIDAR's far wall pulled the pose 32 mm
#: a scan, so its cut was 6.5 m. The band's sigma there is 2.4 cm and it
#: reads a surface by its returns' median: matched from the true pose where
#: a scan has returns 7.5-8 m out, the pull was 1.0 mm (p95 2.1) at 8.0
#: against 0.8 (1.9) at 6.5, and the park's middle went from its heading
#: unpinned at 71 % of places to 26 % (`lidar_spike.py --probe`, the band);
#: flown, the belief's worst error was a median 0.13 m against 0.15 (the
#: drift spike, eight flights an arm on one pod; SimNotes, "The grown world").
MATCH_RANGE_M = 8.0
#: ...and returns nearer than this, m: what the hands work on -- the
#: station's box, the plug in the hand, a drawer, a door -- each low surface
#: struck by one ring or two at the part's noisiest (4 cm at 0.2 m), where a
#: far wall is struck by ten. MEASURED (#581's stage 2): with them, a dock's
#: matcher moved the belief 40-67 cm in 16-19 s, a seat step walked by the
#: belief stopped 3.6 cm inside the seat, and 2 of 3 wide starts never lined
#: up; without them, 26-41 cm, and 3 of 3 docked.
MATCH_MIN_M = 1.0
#: The field is recomputed after this much travel from where it was, m, or
#: this many fused scans: 5 ms a refresh; a fresher field did no better.
REFRESH_M = 1.0
REFRESH_SCANS = 20
#: The field's window, m either way: a point outside it pulls on nothing.
#: ⚠ SHORT OF MATCH_RANGE_M AND A REFRESH'S TRAVEL ON PURPOSE (#560): grown
#: to their 9 m, the static probe pinned the park's middle better (19 % of
#: places unpinned against 26) on a map laid from the true poses, but flown
#: on the robot's own map the drift spike's worst error was a median 0.21 m
#: against 0.13 at 7.5 (the same eight flights, five worse, one 0.13 to 0.47).
WINDOW_M = 7.5
#: The robust weight's scale, m (Cauchy): a point this far off its wall
#: counts half, one three times as far a tenth.
ROBUST_M = 0.05
#: A point this close to a mapped wall at the matched pose is an inlier, m,
#: and a match needs MIN_INLIERS of them: fewer is a map too thin to say
#: where the robot is (a day's first scans, a new room).
INLIER_M = 0.10
MIN_INLIERS = 40
#: Gauss-Newton iterations, at most, and the step under which it stops.
MAX_ITER = 8
CONVERGED_M = 5e-4
#: The most one scan may move the pose: past it the fit has found another
#: wall to sit on, and its correction is never taken -- the bounded search's
#: pose may be (DOUBT_*), else the scan is refused and not fused ("slid").
MAX_STEP_M = 0.15
MAX_STEP_RAD = math.radians(5.0)
#: ODOMETRY'S PRIOR, in points' worth (a unit-gradient point adds 1): how
#: hard the fit is held to the pose odometry predicted. Without it every
#: scan put the map's quantisation into the pose -- a wall's surface lies
#: anywhere in the 5 cm cell it marks -- and with perfect odometry the
#: house read 5-12 cm and 0.8 deg off (MEASURED, replayed flights).
PRIOR_XY = 300.0
PRIOR_TH = 300.0
#: ...and it is ROBUST (Cauchy): a disagreement past this is odometry
#: wrong, not the map, and the prior lets go. MEASURED: wheels spinning on
#: the lab's feed plate pumped ~10 mm of travel a scan for 12 s; a plain
#: prior held on to it, the fit left its basin and turned the pose 12 deg
#: to explain it, and the robot drove home 3.4 m out.
PRIOR_SCALE = 0.005
#: A direction whose information, read off the walls' SMOOTHED normals over
#: the inliers, is under this share of the points in the fit (and under
#: DEGENERATE points in all) is one the walls do not fix: there the pose
#: stays odometry's. MEASURED on synthetic scans: a straight corridor
#: 0.003 along it, a room 0.34 at its weakest. Off the raw gradient a
#: corridor read 0.12 -- 5 cm cells make a straight wall step -- and a
#: threshold that held it also held the lab's weak direction during the
#: plate's pump, where odometry was the thing that was wrong.
DEGENERATE_SHARE = 0.05
DEGENERATE = 12.0
#: The walls' normals are read off the field averaged over a box this many
#: cells wide (`refresh`).
NORMAL_BOX_CELLS = 5
#: Rotation against translation, m: one radian at LEVER_M is LEVER_M metres.
LEVER_M = 1.0
#: A fit whose inliers are under this share of the points on mapped cells
#: is "inconsistent": the pose is off by more than the fit reaches. The
#: search (`_search`) then looks this far round the prediction, in these
#: turns, and a pose it finds must put this share on walls to be taken.
LOST_SHARE = 0.5
SEARCH_M = 0.6
SEARCH_RAD = math.radians(6.0)
SEARCH_STEP_RAD = math.radians(1.0)
FOUND_SHARE = 0.75
#: ⚠ A FIT THE WALLS ONLY PARTLY AGREE WITH IS DOUBTED (issue #681): one
#: that held a direction with under DOUBT_HELD_SHARE of its points on walls,
#: or any under DOUBT_SHARE, is searched as an inconsistent one is -- and a
#: slid one too -- a pose found taken only at DOUBT_FOUND_SHARE; found
#: nothing, it is what it was ("doubted": the fit's correction stands and
#: its scan is laid in, as `ok`; a slid one stays refused). MEASURED: at the
#: storeroom's counter the side walls kept a fit `ok` 0.3-0.4 m off along
#: the counter at 40 of 40 poses, its scan laid in, and the copy kept the
#: error for hours (Luca, 11-08, 0.58 m); doubted, 40 of 40 taken back.
#: From the truth over the house, held fits under 0.95 were 15 of 2,306 and
#: any fit under 0.8 32 of 7,761; a pose found taken at FOUND_SHARE moved
#: the truth over 0.1 m at 11 places of 7,840 (the stairs), at 0.9 at 3, the
#: base 2. ⚠ NOT LAID IN, doubted scans starved the map of new ground: a
#: robot walked to the mall's station 0.43 m off, the station's fix kept
#: the map it had laid, and no walk out of there found a route.
DOUBT_HELD_SHARE = 0.95
DOUBT_SHARE = 0.8
DOUBT_FOUND_SHARE = 0.9
#: ⚠ ...AND ITS SEARCH TAKES ONLY THE ONE PLACE IN ITS WINDOW THAT EXPLAINS
#: THE SCAN: none UNIQUE_M or more from the best puts DOUBT_RIVAL_SHARE of
#: the best's points on walls (each place at its best turn). MEASURED on a
#: drift flight (seed 19 from the hall): along a sidewalk -- a wall and a
#: fence, parallel, nothing across -- every place along it scored 0.95-1.00
#: of the best, and the search took the best 25 times, each a few points
#: ahead and up to 0.6 m behind the robot, until it was 5.95 m off; at the
#: counter the rival is the alias 0.6 m east, 0.75 of the truth.
DOUBT_RIVAL_SHARE = 0.9
#: ⚠ A ROBOT ITS MAP KEEPS REFUSING IS LOST PAST THE SEARCH'S REACH (issue
#: #422): after LOST_RUN scans running "inconsistent" -- any other verdict
#: breaks the run -- and every LOST_RUN more, the search looks WIDE_M and
#: WIDE_RAD round the belief (`_relocate`): on a lattice WIDE_STRIDE cells
#: apart with the walls widened to cover it, then cell by cell round its
#: best. A pose it finds must clear FOUND_SHARE with all three directions
#: fixed; explain the scan best of the WIDE_TRIES best PLACES (lattice poses
#: UNIQUE_M apart at any heading: one spot at three headings is one place,
#: and would hide a second), none of the others found explaining it as well
#: (RIVAL_SHARE); and be found AGAIN by the next wide search, the same
#: correction within UNIQUE_M and UNIQUE_RAD. ⚠ ONCE IS A
#: COINCIDENCE: on a sidewalk whose thin wall the map had eroded (#401), a
#: pose 1.8 m along it explained a scan better than the truth did, and the
#: next search took a different one.
LOST_RUN = 10
WIDE_M = 2.0
WIDE_RAD = math.radians(15.0)
WIDE_STRIDE = 4
WIDE_TRIES = 3
UNIQUE_M = 0.5
UNIQUE_RAD = math.radians(5.0)
#: ...and a second place found stops a relocation only if it puts this
#: share of the best's inliers on walls: a twin room explains the scan as
#: well, to the noise. MEASURED: a pose 2 m off, most of the scan past the
#: map's edge and the rest on its walls, was "found" with 216 inliers where
#: the truth had 359 (0.60), and refused a relocation the truth had earned.
RIVAL_SHARE = 0.9
#: ⚠ AN ANCHOR RE-LAYS THE MAP ROUND IT (issue #422): once the dock has put
#: the belief in its frame to millimetres (`anchored`), the next
#: ANCHORED_SCANS scans that go into the map are laid at that belief,
#: unmatched. MEASURED: a robot come home lost had laid a copy of the living
#: room 0.9 m askew; anchored on the dock, matching against the copy pulled
#: it 0.62 m back into it as it backed off. 30 is the undock's own back-out
#: (0.9 m at 0.3 m/s, ten scans a second), and a wall clamped at +5 is
#: floor after 14 misses.
ANCHORED_SCANS = 30
#: A scan is fused only once the pose has moved this far or turned this
#: much since the last one fused, or FUSE_S has passed. MEASURED: docked for
#: 388 s, every scan fused at a pose jittering by a millimetre walked the
#: map and the pose together 0.15 m; fused every 5 s, 0.03 m.
FUSE_M = 0.01
FUSE_RAD = math.radians(0.5)
FUSE_S = 5.0

NO_STEP = (0.0, 0.0, 0.0)
#: The verdicts that move the pose, and those whose scan may be fused
#: (`Match.fuse`): matched, found by the search, doubted and found nothing
#: better (as `ok`), laid at an anchor, or too little map under it to say --
#: never a scan whose fit slid or whose map disagreed, nor the one the wide
#: search relocated by (the next, matched from there, is).
ACCEPTED = frozenset({"ok", "found", "relocated", "doubted"})
FUSED = frozenset({"ok", "found", "doubted", "anchored", "no map", "sparse", "unconstrained"})
#: ...and a fit taken as it came, no search having moved it: what was "ok"
#: before #681 doubted a share of it. An instrument counting matched scans
#: counts these (`lidar_spike.py --probe`, the drift spike's `--done`).
FIT_TAKEN = frozenset({"ok", "doubted"})


class Match(NamedTuple):
  """One scan's alignment. `pose` is where to lay the scan: the matched pose
  if `accepted`, else the one asked about."""
  pose: tuple[float, float, float]
  accepted: bool
  #: "ok", "found" (by the search, the fit having been inconsistent, slid or
  #: doubted), "doubted" (the walls only partly agreed, DOUBT_*, and the
  #: search found nothing better: as "ok"), "relocated" (by the wide
  #: search, the robot lost past the search's reach), "anchored" (not
  #: matched: laid at an anchor's belief, ANCHORED_SCANS), or why not: "no
  #: map" (nothing mapped near), "sparse" (fewer than MIN_INLIERS),
  #: "unconstrained" (no direction fixed), "slid" (past MAX_STEP, and the
  #: search found nothing), "inconsistent" (the map disagreed, and the
  #: search found nothing it agreed with).
  why: str
  inliers: int
  #: The inliers' RMS distance to the map at the matched pose, m.
  rms_m: float
  #: The fit's weakest direction: its information, and the unit direction
  #: (x, y, LEVER_M * theta) it points along.
  weakest: float
  weak_dir: tuple[float, float, float]
  #: How many of the three directions odometry kept (0 = a full fit).
  held: int
  #: The correction applied, (dx m, dy m, dtheta rad); zero if refused.
  step: tuple[float, float, float]

  @property
  def fuse(self) -> bool:
    return self.why in FUSED


def _solve3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """a x = b for a 3x3, by Cramer's rule in floats: a tenth of
  `np.linalg.solve`'s overhead, and the fit makes a few a scan."""
  (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a.tolist()
  b0, b1, b2 = b.tolist()
  c00 = a11 * a22 - a12 * a21
  c01 = a12 * a20 - a10 * a22
  c02 = a10 * a21 - a11 * a20
  det = a00 * c00 + a01 * c01 + a02 * c02
  x0 = (b0 * c00 + a01 * (a12 * b2 - b1 * a22) + a02 * (b1 * a21 - a11 * b2)) / det
  x1 = (a00 * (b1 * a22 - a12 * b2) + b0 * c01 + a02 * (a10 * b2 - b1 * a20)) / det
  x2 = (a00 * (a11 * b2 - b1 * a21) + a01 * (b1 * a20 - a10 * b2) + b0 * c02) / det
  return np.array([x0, x1, x2])


def _same(a, b) -> bool:
  """Two poses, or two corrections, (x, y, theta) the wide search found are
  one: within UNIQUE_M and UNIQUE_RAD of each other."""
  turned = abs(math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2])))
  return math.hypot(a[0] - b[0], a[1] - b[1]) < UNIQUE_M and turned < UNIQUE_RAD


def _rival(places: np.ndarray, shifts: np.ndarray, best_at) -> int:
  """The most points on walls at any place, of a search's `places` (each
  place's most at any turn, [dy, dx] over `shifts`, m), UNIQUE_M or more
  from the best's shift `best_at` (dx, dy, dtheta); 0 if there is none."""
  apart = np.hypot(shifts[:, None] - best_at[1], shifts[None, :] - best_at[0])
  far = apart >= UNIQUE_M - 1e-9                # whole cells: 0.5 m reads 0.49999...
  return int(places[far].max()) if far.any() else 0


def _normals(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The walls' unit normals over a field: its gradient once averaged over
  NORMAL_BOX_CELLS, unseen cells left out of the average."""
  known = ~np.isnan(field)
  weight = ndimage.uniform_filter(known.astype(float), NORMAL_BOX_CELLS)
  smooth = ndimage.uniform_filter(np.where(known, field, 0.0), NORMAL_BOX_CELLS)
  smooth /= np.maximum(weight, 1e-9)
  nx, ny = np.zeros_like(smooth), np.zeros_like(smooth)
  nx[:, 1:-1] = smooth[:, 2:] - smooth[:, :-2]
  ny[1:-1, :] = smooth[2:, :] - smooth[:-2, :]
  norm = np.hypot(nx, ny)
  norm[norm < 1e-12] = np.inf
  return nx / norm, ny / norm


class ScanMatcher:
  """Aligns scans against one `OccupancyGrid` (the robot's own). `origin`
  is the sensor's (forward, left) offset from the pose point, as
  `OccupancyGrid.update` takes it."""

  def __init__(self, grid, origin: tuple[float, float] = (0.0, 0.0),
               max_range: float = 8.0) -> None:
    self.grid = grid
    self.origin = origin
    self.max_range = max_range
    #: The signed distance over the window last computed, m (NaN unseen),
    #: its lower corner in grid cells (ix, iy), and the walls' normals.
    self.field: np.ndarray | None = None
    self.normals: tuple[np.ndarray, np.ndarray] | None = None
    self.corner = (0, 0)
    #: Where the field was computed (world x, y) and the scans fused since.
    self.at: tuple[float, float] | None = None
    self.fused_since = 0
    #: The pose and sim time of the last scan fused (`fuses`).
    self.fused_pose: tuple[float, float, float] | None = None
    self.fused_t = -math.inf
    #: Every verdict over a life, and the matches held along a direction.
    self.counts: dict[str, int] = {}
    self.last: Match | None = None
    #: The scans running the map has refused ("inconsistent"): at LOST_RUN,
    #: the wide search (`_relocate`); and the correction the last one found,
    #: (dx, dy, dtheta), until the next finds it again.
    self.lost_run = 0
    self.pending: tuple[float, float, float] | None = None
    #: Scans still to be laid at an anchor's belief, unmatched (`anchored`).
    self.anchoring = 0

  # ---- the field ------------------------------------------------------------

  def refresh(self, x: float, y: float) -> None:
    """Recompute the field over the window around (x, y)."""
    g = self.grid
    res = g.resolution
    rows, cols = g.grid.shape
    half = int(round(WINDOW_M / res))
    cx, cy = g.world_to_cell(x, y)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, cols), min(cy + half, rows)
    self.at, self.fused_since = (x, y), 0
    self.field = self.normals = None
    if x1 - x0 < 2 or y1 - y0 < 2:
      return
    window = g.grid[y0:y1, x0:x1]
    occupied = window > OCC_THRESH
    seen = window != 0.0
    # Seen and not a wall is free here, even with mixed evidence: a wall met
    # at a grazing angle is hit and passed in turn.
    free = seen & ~occupied
    if not occupied.any() or not free.any():
      return
    # SIGNED, zero on the centre of a wall's first cell (a return lands
    # anywhere in the cell it marks), + in front, - inside measured from the
    # FREE side: a wall seen through ranging noise is a band of cells deep,
    # and measured unsigned a point in it pulled on nothing, measured from
    # its back one past the middle was pushed out through the wall. The
    # chamfer transform, not the Euclidean: 4x cheaper, and across a
    # straight wall the same zero and the same normal.
    outside = ndimage.distance_transform_cdt(~occupied, metric="taxicab").astype(float)
    inside = ndimage.distance_transform_cdt(~free, metric="taxicab").astype(float)
    reach = REACH_M / res
    d = np.where(free, np.minimum(outside, reach), -np.minimum(inside - 1.0, reach))
    # A cell never seen is no information: pulled to the nearest known
    # cell, a return off new geometry would drag the pose to the edge of
    # what was mapped.
    d[~seen & (outside > BEHIND_CELLS)] = np.nan
    d *= res
    self.field, self.corner = d, (x0, y0)
    self.normals = _normals(d)

  def _stale(self, x: float, y: float) -> bool:
    return (self.at is None or self.fused_since >= REFRESH_SCANS
            or math.hypot(x - self.at[0], y - self.at[1]) > REFRESH_M)

  # ---- when a scan goes into the map ------------------------------------------

  def fuses(self, match: Match, t: float) -> bool:
    """Whether the scan just matched goes into the map: its verdict allows
    it, and the robot has moved since the last one fused (FUSE_*)."""
    if not match.fuse:
      return False
    if self.fused_pose is None or t - self.fused_t >= FUSE_S:
      return True
    x, y, th = match.pose
    fx, fy, fth = self.fused_pose
    turned = abs(math.atan2(math.sin(th - fth), math.cos(th - fth)))
    return math.hypot(x - fx, y - fy) >= FUSE_M or turned >= FUSE_RAD

  def anchored(self, on: bool = True) -> None:
    """The belief was just put where the world says it is to millimetres
    (the dock's board): the next ANCHORED_SCANS scans fused are laid at it,
    unmatched, the next of them at once. `on=False` closes the window: the
    belief was put somewhere else since."""
    self.anchoring = ANCHORED_SCANS if on else 0
    if on:
      self.lost_run, self.pending, self.fused_pose = 0, None, None

  def fuse_next(self) -> None:
    """The next scan the verdict allows is fused, moved or not: a robot back
    on its wheels after a fall maps again at once."""
    self.fused_pose = None

  def fused(self, pose, t: float) -> None:
    """The scan just matched went into the map at `pose`, at sim time `t`."""
    self.fused_since += 1
    self.fused_pose = tuple(float(v) for v in pose)
    self.fused_t = float(t)
    if self.anchoring:
      self.anchoring -= 1

  # ---- the fit ----------------------------------------------------------------

  def points(self, angles, ranges, origin=None) -> np.ndarray:
    """The scan's returns from MATCH_MIN_M to short of MATCH_RANGE_M as (N,
    2) points in the pose's frame, taken from `origin` (None: `origin`'s);
    a ray that hit nothing is not a point."""
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    hit = (ranges >= MATCH_MIN_M) & (ranges < min(self.max_range - 1e-6, MATCH_RANGE_M))
    a, r = angles[hit], ranges[hit]
    fwd, left = self.origin if origin is None else origin
    return np.stack([fwd + r * np.cos(a), left + r * np.sin(a)], axis=1)

  def _world(self, pose, pts: np.ndarray):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]

  def distance(self, pose, pts: np.ndarray):
    """Each point's signed distance to the map at `pose`, m, its gradient
    (d/dx, d/dy, d/dtheta), and `live`: the points on mapped cells."""
    x, y, _ = pose
    wx, wy = self._world(pose, pts)
    g, f = self.grid, self.field
    res = g.resolution
    # Cell centres at integer coordinates: the four cells round a point.
    u = (wx - g.x_min) / res - 0.5 - self.corner[0]
    v = (wy - g.y_min) / res - 0.5 - self.corner[1]
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    h, w = f.shape
    live = (i >= 0) & (i < w - 1) & (j >= 0) & (j < h - 1)
    k = np.where(live, j * w + i, 0)
    fu, fv = u - i, v - j
    flat = f.ravel()
    d00, d10 = flat[k], flat[k + 1]
    d01, d11 = flat[k + w], flat[k + w + 1]
    a, b = d10 - d00, d11 - d01
    top = d00 + fu * a
    bottom = d01 + fu * b
    d = top + fv * (bottom - top)
    gx = (a + fv * (b - a)) / res
    gy = (bottom - top) / res
    live &= ~np.isnan(d)
    d, gx, gy = (np.where(live, arr, 0.0) for arr in (d, gx, gy))
    # d(wx, wy)/dtheta is the point's offset from the pose, turned 90 deg.
    gth = gx * -(wy - y) + gy * (wx - x)
    return d, gx, gy, gth, live

  def match(self, pose, angles, ranges, origin=None) -> Match:
    """Align one scan, taken at the believed `pose` from `origin` (None:
    `origin`'s), against the map."""
    x0, y0, th0 = (float(v) for v in pose)
    if self.anchoring:
      return self._done(pose, "anchored")
    if self.field is None or self._stale(x0, y0):
      self.refresh(x0, y0)
    pts = self.points(angles, ranges, origin)
    if self.field is None or len(pts) < MIN_INLIERS:
      return self._done(pose, "no map" if self.field is None else "sparse")
    at, why, kw = self._judge((x0, y0, th0), pts, self._fit((x0, y0, th0), pts))
    if why not in ("inconsistent", "doubted", "slid"):
      return self._done(at, why, **kw)
    # Most of the scan on mapped cells and too little of it on walls: the
    # pose is off by more than the fit reaches -- a wheel pump while the
    # robot was tilted and unmatched left it 0.45 m out, the fit could not
    # pull it back, and every scan fused painted a second copy of the house.
    # Search round the prediction, fit from the best, and take it only if
    # the walls agree; else the scan is refused and NOT fused. A fit that
    # slid or that the walls only partly agree with is searched too, its
    # best taken only where no other place explains the scan as well, and
    # else it is what it was: slid refused, doubted laid in as ok.
    start = self._search((x0, y0, th0), pts, unique=why != "inconsistent")
    if start is not None:
      found_at, found, found_kw = self._judge(
        (x0, y0, th0), pts, self._fit(start, pts), found=True,
        share=FOUND_SHARE if why == "inconsistent" else DOUBT_FOUND_SHARE)
      if found == "found":
        return self._done(found_at, found, **found_kw)
    if why != "inconsistent":
      # ...doubted or slid, and nothing better near: as it was
      return self._done(at, why, **kw)
    # ...and refused for long enough, the robot is lost past the search's
    # reach: search wider (LOST_RUN), and take what it finds the second time
    self.lost_run += 1
    if self.lost_run % LOST_RUN == 0:
      wide = self._relocate((x0, y0, th0), pts)
      seen, self.pending = self.pending, None if wide is None else wide[1]["step"]
      if wide is not None and seen is not None and _same(seen, self.pending):
        return self._done(wide[0], "relocated", **wide[1])
    return self._done(at, why, **kw)

  def _fit(self, start, pts: np.ndarray):
    """Gauss-Newton from `start`, odometry's prior centred on it: the pose
    it converged to, and the normal matrix, distances and liveness there."""
    x0, y0, th0 = start
    x, y, th = start
    prior = np.array([PRIOR_XY, PRIOR_XY, PRIOR_TH])
    for _ in range(MAX_ITER):
      hess, grad, d, live = self._normal((x, y, th), pts)
      off = np.array([x - x0, y - y0, (th - th0) * LEVER_M])
      apart = np.array([math.hypot(off[0], off[1])] * 2 + [abs(off[2])])
      lam = prior / (1.0 + (apart / PRIOR_SCALE) ** 2)
      step = _solve3(hess + np.diag(lam), -(grad + lam * off))
      x, y, th = x + step[0], y + step[1], th + step[2] / LEVER_M
      if math.hypot(step[0], step[1]) < CONVERGED_M and abs(step[2]) < CONVERGED_M:
        break
    else:
      hess, grad, d, live = self._normal((x, y, th), pts)
    return (x, y, th), d, live

  def _judge(self, asked, pts, fit, found: bool = False, share: float = FOUND_SHARE):
    """The verdict on a fit: (the pose to lay the scan at, why, the rest of
    `Match`). Which directions the walls fix is read where the fit ENDED
    (offset, a room's points sit off its walls and it looks like a
    corridor), and the correction is kept only along those (Zhang, Kaess &
    Singh 2016, "solution remapping"). Converged, the last evaluation is the
    final pose's to within CONVERGED_M."""
    (x, y, th), d, live = fit
    x0, y0, th0 = asked
    info, vec = self._directions((x, y, th), pts, d, live)
    keep = info >= max(DEGENERATE, DEGENERATE_SHARE * float(live.sum()))
    total = np.array([x - x0, y - y0, (th - th0) * LEVER_M])
    kept = vec[:, keep] @ (vec[:, keep].T @ total)
    dx, dy, dth = float(kept[0]), float(kept[1]), float(kept[2]) / LEVER_M
    inl = live & (np.abs(d) < INLIER_M)
    n, n_live = int(inl.sum()), int(live.sum())
    kw = dict(inliers=n, rms_m=float(np.sqrt(np.mean(d[inl] ** 2))) if n else math.inf,
              weakest=float(info[0]), weak_dir=tuple(float(v) for v in vec[:, 0]),
              held=int(3 - keep.sum()))
    fit_pose = (x0 + dx, y0 + dy, th0 + dth)
    if found:
      if n >= MIN_INLIERS and n >= share * n_live and keep.all():
        return fit_pose, "found", dict(step=(dx, dy, dth), **kw)
      return asked, "inconsistent", kw
    if n_live >= MIN_INLIERS and n < LOST_SHARE * n_live:
      return asked, "inconsistent", kw
    if not keep.any():
      return asked, "unconstrained", kw
    if n < MIN_INLIERS:
      return asked, "sparse", kw
    if math.hypot(dx, dy) > MAX_STEP_M or abs(dth) > MAX_STEP_RAD:
      return asked, "slid", kw
    if n < (DOUBT_SHARE if keep.all() else DOUBT_HELD_SHARE) * n_live:
      return fit_pose, "doubted", dict(step=(dx, dy, dth), **kw)
    return fit_pose, "ok", dict(step=(dx, dy, dth), **kw)

  def _search(self, pose, pts: np.ndarray, reach_m: float = SEARCH_M,
              reach_rad: float = SEARCH_RAD, unique: bool = False):
    """The pose within `reach_m` and `reach_rad` of `pose` at which the most
    of the scan lands on a mapped wall (|d| < INLIER_M, the nearest cell):
    brute force over whole-cell shifts and SEARCH_STEP_RAD turns. None if
    no candidate beats `pose` itself -- or, `unique`, if a place UNIQUE_M
    from it puts DOUBT_RIVAL_SHARE as many on walls (`_rival`)."""
    g, f = self.grid, self.field
    res = g.resolution
    on_wall = np.abs(np.nan_to_num(f, nan=np.inf)) < INLIER_M
    h, w = f.shape
    n = int(round(reach_m / res))
    shifts = np.arange(-n, n + 1)
    x0, y0, th0 = pose
    best, best_at = -1, None
    # each place's most at any turn, [dy, dx]: one spot turned is one place
    places = np.zeros((len(shifts), len(shifts)), dtype=np.int64) if unique else None
    for dth in np.arange(-reach_rad, reach_rad + 1e-12, SEARCH_STEP_RAD):
      wx, wy = self._world((x0, y0, th0 + dth), pts)
      i = np.floor((wx - g.x_min) / res).astype(np.int64) - self.corner[0]
      j = np.floor((wy - g.y_min) / res).astype(np.int64) - self.corner[1]
      ii = i[None, None, :] + shifts[None, :, None]
      jj = j[None, None, :] + shifts[:, None, None]
      inside = (ii >= 0) & (ii < w) & (jj >= 0) & (jj < h)
      hits = on_wall[np.where(inside, jj, 0), np.where(inside, ii, 0)] & inside
      score = hits.sum(axis=2)
      if unique:
        np.maximum(places, score, out=places)
      k = int(np.argmax(score))
      if score.flat[k] > best:
        best, best_at = int(score.flat[k]), (float(shifts[k % len(shifts)]) * res,
                                             float(shifts[k // len(shifts)]) * res, float(dth))
    here = self._on_wall_count(pose, pts, on_wall)
    if best_at is None or best <= here:
      return None
    if unique and _rival(places, shifts * res, best_at) >= DOUBT_RIVAL_SHARE * best:
      return None
    return (x0 + best_at[0], y0 + best_at[1], th0 + best_at[2])

  def _relocate(self, pose, pts: np.ndarray):
    """The wide search (LOST_RUN): the one pose within WIDE_M and WIDE_RAD of
    `pose` the map agrees with, as (pose, `Match`'s fields), or None. Each of
    the best lattice places is searched cell by cell round it and fitted;
    the "found" one with the most inliers is taken, unless another found
    elsewhere explains the scan as well (RIVAL_SHARE)."""
    found = []
    for start in self._wide_candidates(pose, pts):
      fine = self._search(start, pts, reach_m=WIDE_STRIDE * self.grid.resolution,
                          reach_rad=SEARCH_STEP_RAD) or start
      at, why, kw = self._judge(pose, pts, self._fit(fine, pts), found=True)
      if why == "found":
        found.append((at, kw))
    if not found:
      return None
    best = max(found, key=lambda f: f[1]["inliers"])
    if any(not _same(at, best[0]) and kw["inliers"] >= RIVAL_SHARE * best[1]["inliers"]
           for at, kw in found):
      return None
    return best

  def _wide_candidates(self, pose, pts: np.ndarray) -> list[tuple[float, float, float]]:
    """The WIDE_TRIES best poses of a lattice WIDE_STRIDE cells and
    SEARCH_STEP_RAD apart within WIDE_M and WIDE_RAD of `pose`, each scored
    by its points within half a stride of a mapped wall, and each UNIQUE_M
    from those taken before it at ANY heading: one place a candidate."""
    g, f = self.grid, self.field
    res = g.resolution
    on_wall = (np.abs(np.nan_to_num(f, nan=np.inf)) < INLIER_M).astype(np.uint8)
    near = ndimage.maximum_filter(on_wall, size=WIDE_STRIDE + 1) > 0
    h, w = near.shape
    n = int(round(WIDE_M / res / WIDE_STRIDE))
    shifts = np.arange(-n, n + 1) * WIDE_STRIDE
    turns = np.arange(-WIDE_RAD, WIDE_RAD + 1e-12, SEARCH_STEP_RAD)
    x0, y0, th0 = pose
    scores = np.empty((len(turns), len(shifts), len(shifts)), dtype=np.int64)
    for k, dth in enumerate(turns):
      wx, wy = self._world((x0, y0, th0 + dth), pts)
      i = np.floor((wx - g.x_min) / res).astype(np.int64) - self.corner[0]
      j = np.floor((wy - g.y_min) / res).astype(np.int64) - self.corner[1]
      ii = i[None, None, :] + shifts[None, :, None]
      jj = j[None, None, :] + shifts[:, None, None]
      inside = (ii >= 0) & (ii < w) & (jj >= 0) & (jj < h)
      scores[k] = (near[np.where(inside, jj, 0), np.where(inside, ii, 0)] & inside).sum(axis=2)
    out: list[tuple[float, float, float]] = []
    for flat in np.argsort(-scores, axis=None, kind="stable"):
      k, a, b = np.unravel_index(int(flat), scores.shape)
      cand = (x0 + float(shifts[b]) * res, y0 + float(shifts[a]) * res, th0 + float(turns[k]))
      if any(math.hypot(cand[0] - c[0], cand[1] - c[1]) < UNIQUE_M for c in out):
        continue
      out.append(cand)
      if len(out) == WIDE_TRIES:
        break
    return out

  def _on_wall_count(self, pose, pts, on_wall) -> int:
    g = self.grid
    wx, wy = self._world(pose, pts)
    i = np.floor((wx - g.x_min) / g.resolution).astype(np.int64) - self.corner[0]
    j = np.floor((wy - g.y_min) / g.resolution).astype(np.int64) - self.corner[1]
    h, w = on_wall.shape
    inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    return int((on_wall[np.where(inside, j, 0), np.where(inside, i, 0)] & inside).sum())

  def _normal(self, pose, pts: np.ndarray):
    """The fit's normal matrix and gradient at `pose` (rotation scaled by
    LEVER_M, Cauchy-weighted), and the points' distances and liveness."""
    d, gx, gy, gth, live = self.distance(pose, pts)
    w = np.where(live, 1.0 / (1.0 + (d / ROBUST_M) ** 2), 0.0)
    jac = np.stack((gx, gy, gth / LEVER_M))
    hess = np.einsum("in,jn->ij", jac * w, jac)
    grad = np.einsum("in,n->i", jac, w * d)
    return hess, grad, d, live

  def _directions(self, pose, pts, d, live):
    """The information the inliers' SMOOTHED wall normals carry, as its
    eigenvalues (ascending) and eigenvectors, rotation scaled by LEVER_M."""
    x, y, _ = pose
    wx, wy = self._world(pose, pts)
    g = self.grid
    i = np.floor((wx - g.x_min) / g.resolution).astype(np.int64) - self.corner[0]
    j = np.floor((wy - g.y_min) / g.resolution).astype(np.int64) - self.corner[1]
    nx_map, ny_map = self.normals
    h, w = nx_map.shape
    use = live & (np.abs(d) < INLIER_M) & (i >= 0) & (i < w) & (j >= 0) & (j < h)
    i, j = np.where(use, i, 0), np.where(use, j, 0)
    nx = np.where(use, nx_map[j, i], 0.0)
    ny = np.where(use, ny_map[j, i], 0.0)
    jac = np.stack((nx, ny, (nx * -(wy - y) + ny * (wx - x)) / LEVER_M))
    return np.linalg.eigh(np.einsum("in,jn->ij", jac, jac))

  def _done(self, pose, why: str, inliers: int = 0, rms_m: float = math.inf,
            weakest: float = 0.0, weak_dir=(0.0, 0.0, 0.0), held: int = 3,
            step=NO_STEP) -> Match:
    m = Match(tuple(float(v) for v in pose), why in ACCEPTED, why, inliers,
              rms_m, weakest, tuple(weak_dir), held, tuple(step))
    if why != "inconsistent":
      # any other verdict breaks the run of refusals, and what the wide
      # search found in it (LOST_RUN)
      self.lost_run, self.pending = 0, None
    self.counts[why] = self.counts.get(why, 0) + 1
    if why in ACCEPTED and held:
      self.counts["held"] = self.counts.get("held", 0) + 1
    self.last = m
    return m

  # ---- a restart (issue #345) ---------------------------------------------------

  def forget(self) -> None:
    """The map it matched against is gone (issue #419, a true death): no
    field, nothing fused, and a new life's verdicts counted from zero."""
    self.field = self.normals = None
    self.corner, self.at, self.fused_since = (0, 0), None, 0
    self.fused_pose, self.fused_t = None, -math.inf
    self.counts, self.last = {}, None
    self.lost_run, self.pending, self.anchoring = 0, None, 0

  def kept_state(self) -> tuple[dict, dict]:
    """The field is the map as it was when last computed, not as it is now,
    so it is kept whole (its normals are its own function)."""
    state = {"corner": list(self.corner),
             "at": None if self.at is None else list(self.at),
             "fusedSince": self.fused_since,
             "fusedPose": None if self.fused_pose is None else list(self.fused_pose),
             "fusedT": None if self.fused_pose is None else self.fused_t,
             "counts": dict(self.counts), "lostRun": self.lost_run,
             "pending": None if self.pending is None else list(self.pending),
             "anchoring": self.anchoring}
    return state, ({} if self.field is None else {"matchField": self.field})

  def restore_kept(self, state: dict, arrays: dict) -> None:
    self.corner = tuple(int(v) for v in state["corner"])
    self.at = None if state["at"] is None else tuple(float(v) for v in state["at"])
    self.fused_since = int(state["fusedSince"])
    if state.get("fusedPose") is not None:
      self.fused_pose = tuple(float(v) for v in state["fusedPose"])
      self.fused_t = float(state["fusedT"])
    self.counts = dict(state.get("counts", {}))
    self.lost_run = int(state.get("lostRun", 0))
    pending = state.get("pending")
    self.pending = None if pending is None else tuple(float(v) for v in pending)
    self.anchoring = int(state.get("anchoring", 0))
    self.field = arrays.get("matchField")
    self.normals = None if self.field is None else _normals(self.field)
