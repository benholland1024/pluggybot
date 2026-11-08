"""What a depth cloud says of a box in front of the robot (issue #466, stage
2): the height of its top, which way it faces, and where its top's edges
are -- off the points alone, in the map frame the robot laid them in (z
over the floor). Nothing here knows what the box is or what opens it: a
guess like "the lid hinges at the back of its top" is the caller's (the
reference opener's foresight, `evaluation.opener.guess`).

THE TOP is the highest level a large share of the points stand at: a flat
face fills a narrow band of heights, where a wall or a front face spreads
its points over all of its own.

AN EDGE (`edge`). A face's points along a direction are dense up to its
edge, and blurred past it by the camera's noise. Of the points from `c`, a
little inside a rough edge, uniform on [c, b], the mean is (c + b) / 2
whatever the blur -- a symmetric blur moves no mean -- so the edge is b =
2 * mean - c. Unbiased where the points are even across the window; the
rows a stereo camera lays at a grazing angle, 4-5 mm apart from the probe's
stance, all but average out along the edge's length: the mean of 32 reads
within 0.3 mm at rows 4.6 mm apart under 1.5 mm of noise, 0.7 at 6 and 3
(`tests/test_box.py`).

AN EDGE WITH ITS FACE TOWARD THE CAMERA (`faced_edge`) breaks that: the
face's points stand AT the edge, and a window a quarter of whose points
stood there read a bracket's tip 4 mm past it. They stand at it and never
past it, so a low percentile of the points reads it -- where the noise is
small against what is read: 0.4 mm on a bracket 0.32 m off.

AN END SEEN PAST (`past_end`, #636). A top's points also end wherever the
camera stopped seeing: the picture's border, something taller in front, the
range. Nothing in the points tells these apart, so an end is judged in the
PICTURE: each of the end's points is walked outward, pixel by pixel, along
where the end's direction falls there, and told by the first pixel past the
top. A pixel lower than the top says the top stopped there, even a face
going down; a higher one, that something taller stands there and hides
what lies beyond (the end of the top's free surface). A pixel with no
reading is walked past: it may be the D435's shadow, a band of nothing
beside a near thing's image-left, and a walk that crosses one reads what
stands behind it. The robot's own body reads nothing either, and is walked
past as any other. A flat top leaves no such hole where it runs on, so a
walk that leaves the picture across only unread pixels, more than a dropout
could be, saw it stop (a far edge's floor beyond the camera's range);
one that leaves right from the top's own points was cut by the picture.

A FACE TOWARD THE ROBOT (`face`): a vertical face's points all stand at
one distance along the direction it faces, so its place is their median
there, its turn the slope of a line through them across it, and its middle
halfway between its two side edges (`edge`).

WHICH WAY IT FACES (`facing`): roughly, the turn of the least rectangle
the top's points fill (its sides read off trimmed extents); then square to
its far edge, laid slice by slice across it (`edge`) and a line drawn
through them. Trimmed, the least rectangle's area is flat within a degree
or two of its least, and alone it read a plain lid of 90 000 points 2.5 deg
off; the far edge is a long straight line.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: The depth frames a box is measured off (the `measure` verb, the
#: reference opener): the D435's own 848 x 480. From the stance the top's
#: back edge is laid within a millimetre; at a seventh of it (120 x 70, the
#: near-field's stream) the back 3 cm of a top went unsampled (SimNotes, "The
#: probe from the robot's own senses").
DEPTH_W, DEPTH_H = 848, 480
#: ...and that stream's nearest depth, m: Intel's tuning guide puts it at
#: the focal length (px) times the baseline over the 126 disparities the
#: ASIC searches, 17 cm at 848 x 480 (1280 x 720's is 0.28).
DEPTH_MIN_Z = 0.17
#: ...this many frames of it, this far apart (s), standing still.
FRAMES, APART_S = 3, 0.1

#: A point this near the floor is floor, m: the D435's noise lifts the
#: floor's own points nearly this high at its 3 m reach (#630: windows of
#: bare floor 1.5-3.0 m off held 12 000-61 000 points, none over 5.3 cm; at
#: 0.01 every one read as a top 1.2-2.6 cm up).
FLOOR_M = 0.06
#: The top's heights are binned this fine, m; the top is the highest bin
#: holding this share of the fullest; a point within `TOP_BAND_M` of its
#: height is on it -- the D435's noise is 1.4 mm at 0.6 m (`depth.NOISE_K`).
TOP_BIN_M = 0.002
TOP_SHARE = 0.25
TOP_BAND_M = 0.005
#: An edge is read off the points this far inside a rough one, m, and the
#: rough one is this percentile of them.
EDGE_WINDOW_M = 0.020
EDGE_ROUGH_PCT = 99.0
#: The fewest points a face is read off.
MIN_POINTS = 30
#: An edge with its face toward the camera is read at this percentile of
#: the points from its end (the module docstring).
FACED_PCT = 1.0
#: Which way a box faces: searched this far either side of a hint, rad, at
#: this step and then a finer one, over at most this many points; its sides
#: read off these percentiles.
FACING_SPAN = math.radians(45.0)
FACING_STEP = math.radians(0.5)
FACING_FINE = math.radians(0.02)
FACING_POINTS = 4000
FACING_TRIM_PCT = 2.0
#: ...and squared to its far edge: laid in slices this wide across the
#: middle of it (its points between these percentiles across), this many
#: times over.
EDGE_SLICE_M = 0.03
EDGE_ACROSS_PCT = (5.0, 95.0)
SQUARE_PASSES = 2
#: AN END SEEN PAST (`past_end`, #636): at most this many of the end's
#: points are walked outward in the picture, this many pixels each: past
#: the end's own 2 cm strip (51 px at the part's nearest, 0.17 m) and the
#: D435's shadow at its widest there (f B / z, 127 px).
PAST_WALKS, PAST_STEPS = 400, 200
#: ...the walk's direction taken off where the point this far along `way`
#: falls in the picture, m.
PAST_AHEAD_M = 0.05
#: ...a pixel is on the top within `TOP_BAND_M` or this many sigmas of the
#: part's noise IN HEIGHT, the noise along its ray times the ray's rise
#: (`DepthPicture.rises`): along the ray alone, 2.4 cm at 1.5 m, a face 5
#: cm behind a far edge passed for the top, 1.7 cm under it.
PAST_SIGMAS = 3.0
#: ...a walk that leaves the picture fewer than this many pixels past the
#: last point on the top went off it with the top: only a pixel that read
#: nothing stood between (a dropout, 1 % a pixel, three running 1e-6).
PAST_CUT_PX = 3
#: ...an end whose points' median depth stands within this of the camera's
#: reach is the reach's, m: 3 of its 3 m sigmas (`depth.NOISE_K`) -- and
#: within this of its nearest reading, the nearest's: a pixel nearer reads
#: nothing, and a walk would take that for the top stopping.
PAST_REACH_M, PAST_NEAR_M = 0.1, 0.02
#: ...and a walk whose own top runs this far past where the end reads, m, was
#: stopped short of it: the top goes on there.
PAST_OVER_M = 0.03


def top_height(points: np.ndarray) -> float | None:
  """The height of the highest flat face among `points` (k, 3), m: None
  where too few stand off the floor to read one."""
  z = np.asarray(points, dtype=float)[:, 2]
  z = z[z > FLOOR_M]
  if len(z) < MIN_POINTS:
    return None
  edges = np.arange(FLOOR_M, float(z.max()) + 2 * TOP_BIN_M, TOP_BIN_M)
  counts, _ = np.histogram(z, edges)
  full = np.flatnonzero(counts >= TOP_SHARE * counts.max())
  centre = float(edges[full.max()]) + TOP_BIN_M / 2
  band = z[np.abs(z - centre) < TOP_BAND_M]
  return float(np.median(band))


def on_top(points: np.ndarray, height: float) -> np.ndarray:
  """Which of `points` stand on a face at `height`."""
  return np.abs(np.asarray(points)[:, 2] - height) < TOP_BAND_M


def edge(along: np.ndarray, far: bool = True, window: float = EDGE_WINDOW_M) -> float | None:
  """Where a face ends along a direction, m (the module docstring): `along`
  is its points' distances along it, and `far` asks for the end the
  direction points to, else the other. None with too few points."""
  got = edge_read(along, far, window)
  return None if got is None else got[0]


def edge_read(along: np.ndarray, far: bool = True,
              window: float = EDGE_WINDOW_M) -> tuple[float, np.ndarray] | None:
  """`edge`, and which of the points it was read off (a mask): for a caller
  that asks whether they end where the face does or where the look at it
  did (`procedure.steps._measure`, #630)."""
  x = np.asarray(along, dtype=float) * (1.0 if far else -1.0)
  if len(x) < MIN_POINTS:
    return None
  cut = float(np.percentile(x, EDGE_ROUGH_PCT)) - window
  sel = x >= cut
  if int(sel.sum()) < MIN_POINTS // 3:
    return None
  b = 2.0 * float(x[sel].mean()) - cut
  return (b if far else -b), sel


@dataclass
class Look:
  """Frames of the depth camera (`Body.depth_pictures`) as one cloud, each
  point's picture and pixel kept: `points` (k, 3), `frame` (k) which of
  `pictures` it came from, `rows` and `cols` (k) its pixel there."""
  pictures: list
  points: np.ndarray
  frame: np.ndarray
  rows: np.ndarray
  cols: np.ndarray

  @classmethod
  def of(cls, pictures: list) -> "Look":
    if not pictures:
      none = np.zeros(0, dtype=int)
      return cls([], np.zeros((0, 3)), none, none, none)
    at = [p.pixels() for p in pictures]
    return cls(list(pictures), np.concatenate([p.points for p in pictures]),
               np.concatenate([np.full(len(p.points), i) for i, p in enumerate(pictures)]),
               np.concatenate([a[0] for a in at]), np.concatenate([a[1] for a in at]))


def past_end(look: Look, picks: np.ndarray, way: np.ndarray, top: float, end: float) -> dict:
  """What the camera saw just past an end of a flat top (the module
  docstring, "AN END SEEN PAST"): `look`'s points `picks` are the end's
  (`edge_read`'s strip), `way` the end's outward direction in the believed
  world frame, `top` the top's height and `end` where the end reads, along
  `way`. Each picked point is walked outward across the picture and told
  by the first pixel past the top: `lower` (the top stopped there),
  `taller` (something higher stands there), `cut` (the picture ended
  first) -- or `runs`, the top's own pixels going on past where the end
  reads: the look stopped short of it, nothing the camera saw. What the
  shares of those are, and `reach` and `near`, whether the end's points
  stand at the camera's farthest or nearest reading."""
  way = np.asarray(way, dtype=float)
  walks = np.asarray(picks)
  if len(walks) > PAST_WALKS:
    walks = walks[np.linspace(0, len(walks) - 1, PAST_WALKS).astype(int)]
  counts = {"lower": 0, "taller": 0, "cut": 0, "runs": 0}
  depths = []
  for i, picture in enumerate(look.pictures):
    these = walks[look.frame[walks] == i]
    if not len(these):
      continue
    start = np.column_stack([look.rows[these], look.cols[these]]).astype(float)
    depths.append(picture.z[look.rows[these], look.cols[these]])
    for key, n in _walked(picture, look.points[these], start, way, top, end).items():
      counts[key] += n
  total = max(sum(counts.values()), 1)
  deep = float(np.median(np.concatenate(depths))) if depths else None
  first = look.pictures[0] if look.pictures else None
  return {**{key: n / total for key, n in counts.items()},
          "reach": deep is not None and deep > first.reach - PAST_REACH_M,
          "near": deep is not None and deep < first.near + PAST_NEAR_M}


def _agreed(mask: np.ndarray) -> np.ndarray:
  """Where a walk's pixel (the axis 1 along it) is true and one of the next
  two along the walk is: a lone pixel past `PAST_SIGMAS` is the sensor's
  noise, a walk crossing a hundred of them meets some (at 3 sigma, a
  quarter of such walks read `lower`), and a surface is several pixels
  wide."""
  next_one, next_two = np.zeros_like(mask), np.zeros_like(mask)
  next_one[:, :-1], next_two[:, :-2] = mask[:, 1:], mask[:, 2:]
  return mask & (next_one | next_two)


def _walked(picture, points: np.ndarray, start: np.ndarray, way: np.ndarray,
            top: float, end: float) -> dict[str, int]:
  """`past_end`'s walks in one picture, told by kind: from each pixel
  `start` (its point `points`) outward along where `way` falls in the
  picture, a pixel of its longer axis a step, so that no pixel is walked
  twice (`_agreed`: a pixel read twice would agree with itself)."""
  ahead = picture.project(points + PAST_AHEAD_M * way) - start
  ahead /= np.maximum(np.abs(ahead).max(axis=1, keepdims=True), 1e-9)
  steps = np.arange(1, PAST_STEPS + 1)
  rows = np.rint(start[:, 0, None] + ahead[:, 0, None] * steps).astype(int)
  cols = np.rint(start[:, 1, None] + ahead[:, 1, None] * steps).astype(int)
  height, width = picture.z.shape
  inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
  rows, cols = np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)
  at = picture.world[rows, cols]
  heights = at[..., 2]
  sigma = picture.noise_k * np.nan_to_num(picture.z[rows, cols]) ** 2
  # NaN compares false either way: a pixel with no reading is neither
  slack = np.maximum(TOP_BAND_M, PAST_SIGMAS * sigma * np.abs(picture.rises(rows, cols)))
  lower = _agreed(inside & (heights < top - slack))
  taller = _agreed(inside & (heights > top + slack))
  on = inside & (np.abs(heights - top) <= slack)

  def first(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(axis=1), mask.argmax(axis=1), PAST_STEPS)

  border = first(~inside)
  hit = first(lower | taller)
  # how far past the reading, beyond what an end's own blur may, its own top ran
  over = np.where(on, at @ way - end - PAST_OVER_M, -np.inf)
  got = {"lower": 0, "taller": 0, "cut": 0, "runs": 0}
  for k in range(len(start)):
    if over[k, :min(hit[k], border[k])].max(initial=-np.inf) > 0.0:
      got["runs"] += 1
    elif hit[k] < border[k]:
      got["lower" if lower[k, hit[k]] else "taller"] += 1
    elif border[k] >= PAST_STEPS:
      got["lower"] += 1        # the whole walk read nothing: looked past
    else:
      tops = np.flatnonzero(on[k, :border[k]])
      last = int(tops[-1]) if len(tops) else -1
      got["cut" if border[k] - last - 1 < PAST_CUT_PX else "lower"] += 1
  return got


def faced_edge(along: np.ndarray, far: bool = True) -> float | None:
  """Where a face ends along a direction when the face that stands at the
  end looks toward the camera (the module docstring): `FACED_PCT` of its
  points from the end. None with too few points."""
  x = np.asarray(along, dtype=float)
  if len(x) < MIN_POINTS:
    return None
  return float(np.percentile(x, 100.0 - FACED_PCT if far else FACED_PCT))


def _area(xy: np.ndarray, yaw: float) -> float:
  c, s = math.cos(yaw), math.sin(yaw)
  u, v = xy @ np.array([c, s]), xy @ np.array([-s, c])
  lo, hi = FACING_TRIM_PCT, 100.0 - FACING_TRIM_PCT
  return float(np.ptp(np.percentile(u, (lo, hi))) * np.ptp(np.percentile(v, (lo, hi))))


def square_to_far_edge(xy: np.ndarray, yaw: float) -> float | None:
  """`yaw` turned square to the far edge of `xy` along it: the edge laid
  slice by slice across (`edge`), a line through the slices' edges. None
  where fewer than three slices read one."""
  u = np.array([math.cos(yaw), math.sin(yaw)])
  v = np.array([-u[1], u[0]])
  pu, pv = xy @ u, xy @ v
  lo, hi = np.percentile(pv, EDGE_ACROSS_PCT)
  at, far = [], []
  for a in np.arange(lo, hi - EDGE_SLICE_M / 2, EDGE_SLICE_M):
    sel = (pv >= a) & (pv < a + EDGE_SLICE_M)
    e = edge(pu[sel], far=True)
    if e is not None:
      at.append(a + EDGE_SLICE_M / 2)
      far.append(e)
  if len(at) < 3:
    return None
  slope = float(np.polyfit(at, far, 1)[0])
  return yaw - math.atan(slope)


def facing(xy: np.ndarray, hint: float) -> float | None:
  """The yaw of a rectangular face's side nearest `hint` (rad), off its
  points' (x, y), squared to the far edge along it (the module
  docstring). None with too few points."""
  xy = np.asarray(xy, dtype=float)
  if len(xy) < MIN_POINTS:
    return None
  few = xy[np.linspace(0, len(xy) - 1, FACING_POINTS).astype(int)] if len(xy) > FACING_POINTS else xy
  few = few - few.mean(axis=0)
  coarse = hint + np.arange(-FACING_SPAN, FACING_SPAN + 1e-9, FACING_STEP)
  best = float(coarse[int(np.argmin([_area(few, a) for a in coarse]))])
  fine = best + np.arange(-FACING_STEP, FACING_STEP + 1e-9, FACING_FINE)
  best = float(fine[int(np.argmin([_area(few, a) for a in fine]))])
  # a rectangle's sides are a quarter turn apart: the one nearest the hint
  k = round((hint - best) / (math.pi / 2))
  best += k * math.pi / 2
  for _ in range(SQUARE_PASSES):
    squared = square_to_far_edge(xy - xy.mean(axis=0), best)
    if squared is None:
      break
    best = squared
  return math.atan2(math.sin(best), math.cos(best))


def face(along: np.ndarray, across: np.ndarray) -> tuple[float, float, float] | None:
  """A vertical face off its points' distances `along` the direction it
  faces and `across` it (the module docstring): its place along, its middle
  across, and the slope of its line (along per across) -- or None with too
  few points or no edge either side."""
  along, across = np.asarray(along, dtype=float), np.asarray(across, dtype=float)
  if len(along) < MIN_POINTS:
    return None
  lo, hi = edge(across, far=False), edge(across, far=True)
  if lo is None or hi is None:
    return None
  slope, at = np.polyfit(across, along, 1)
  middle = (lo + hi) / 2
  return float(at + slope * middle), float(middle), float(slope)
