"""What a depth cloud says of a box in front of the robot (issue #466, stage
2): the height of its top, which way it faces, and where its top's edges
are -- off the points alone, in the map frame the robot laid them in (z
over the floor). Nothing here knows what the box is or what opens it: a
guess like "the lid hinges at the back of its top" is the caller's
(`legs/probe.py`'s foresight, which stage 3's model of the box replaces).

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

import math

import numpy as np

#: A point this near the floor is floor, m.
FLOOR_M = 0.01
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
  x = np.asarray(along, dtype=float) * (1.0 if far else -1.0)
  if len(x) < MIN_POINTS:
    return None
  cut = float(np.percentile(x, EDGE_ROUGH_PCT)) - window
  sel = x[x >= cut]
  if len(sel) < MIN_POINTS // 3:
    return None
  b = 2.0 * float(sel.mean()) - cut
  return b if far else -b


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
