"""Counting what is in a zone, out of the map the robot built (issue #13).

The census is the first task in this repo with **hidden ground truth**: the
sim knows how many plants stand in the garden, and the robot has to go and
find out. Every earlier criterion was ground truth the robot could not get
wrong by being lazy -- a contact conducts or it does not, ink lands or it
does not. This one is failable in the interesting way: drive half the zone
and you report half the plants, with no error anywhere and a confident number
on the screen.

Which is exactly why the count comes off the OCCUPANCY GRID rather than out
of the model. The grid is the robot's belief, built from LIDAR returns it
actually collected from places it actually stood; `true_count` below reads
the model and is the evaluator's alone. Keeping those two apart is the whole
task -- a census that counted bodies would be a lookup with a drive attached.

The counting itself is deliberately dumb: threshold, label connected
components, throw away anything the wrong size. Three filters carry it:

  MARGIN. The zone's own boundary is a fence, and a fence is a long line of
  occupied cells. Everything within `margin` of the rectangle's edge is
  discarded, which also drops the gate and the fence posts with it.

  SPAN. A plant is a 80 mm cylinder -- two or three cells across at the
  50 mm grid. A component wider than `max_span` is scenery, not an object.

  COVERAGE, which is not a filter but a confession: the fraction of the zone
  the robot has actually SEEN. A count of 2 with 45 % coverage is not the
  same claim as a count of 2 with 95 %, and a scoring rule that ignores the
  difference rewards a robot for stopping early.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH

#: m of the zone's edge treated as boundary rather than contents. The garden
#: fence is exactly on the rectangle, and its posts are the only other
#: plant-sized things in there.
MARGIN = 0.40
#: m: the widest a counted object may be. Garden plants are 80 mm; the next
#: thing up in the home world is a 1 m couch.
MAX_SPAN = 0.35
#: cells: below this a component is a stray return, not an object.
MIN_CELLS = 2

#: 8-connectivity -- a diagonal pair of cells is one plant seen from two
#: bearings, not two plants.
_NEIGHBOURHOOD = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class Zone:
  """A named rectangle of the world, in the same shape the scene ships."""

  name: str
  min: tuple[float, float]
  max: tuple[float, float]

  @classmethod
  def from_meta(cls, spec: dict) -> "Zone":
    return cls(name=spec["name"], min=tuple(spec["min"]), max=tuple(spec["max"]))

  @property
  def centre(self) -> tuple[float, float]:
    return ((self.min[0] + self.max[0]) / 2, (self.min[1] + self.max[1]) / 2)

  def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
    return (self.min[0] + margin <= x <= self.max[0] - margin
            and self.min[1] + margin <= y <= self.max[1] - margin)


def survey_route(zone: Zone, inset: float = 1.25,
                 entry: tuple[float, float] | None = None) -> list[tuple[float, float]]:
  """Four standing points that between them see the whole rectangle.

  A lawnmower would be thorough and slow; the LIDAR reaches 8 m and turns
  360 degrees, so what a survey actually needs is a handful of vantage points
  far enough from the walls to stand at. Ordered nearest-first from `entry`
  (the doorway the robot comes in by), so the route does not open with a
  drive across the zone and back.
  """
  x0, x1 = zone.min[0] + inset, zone.max[0] - inset
  y0, y1 = zone.min[1] + inset, zone.max[1] - inset
  if x1 < x0:
    x0 = x1 = (x0 + x1) / 2
  if y1 < y0:
    y0 = y1 = (y0 + y1) / 2
  points = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
  if entry is None:
    return points
  # Rotate the ring so it starts at the corner nearest the door, keeping the
  # cycle order -- the points are a loop, and a loop driven from anywhere is
  # still a loop.
  k = min(range(len(points)),
          key=lambda i: math.dist(points[i], entry))
  return points[k:] + points[:k]


def count_objects(grid, zone: Zone, margin: float = MARGIN,
                  max_span: float = MAX_SPAN,
                  min_cells: int = MIN_CELLS) -> dict:
  """What the robot BELIEVES is standing in the zone, from its own map.

  Returns the count, each object's world centre and cell size, and the
  fraction of the zone the map has any opinion about at all.
  """
  logodds = np.asarray(grid.grid)
  rows, cols = logodds.shape
  res = grid.resolution
  ix0, iy0 = grid.world_to_cell(zone.min[0] + margin, zone.min[1] + margin)
  ix1, iy1 = grid.world_to_cell(zone.max[0] - margin, zone.max[1] - margin)
  ix0, iy0 = max(ix0, 0), max(iy0, 0)
  ix1, iy1 = min(ix1 + 1, cols), min(iy1 + 1, rows)
  if ix1 <= ix0 or iy1 <= iy0:
    return {"count": 0, "objects": [], "coverage": 0.0, "cells": 0}

  window = logodds[iy0:iy1, ix0:ix1]
  known = (window > OCC_THRESH) | (window < FREE_THRESH)
  coverage = float(known.mean()) if known.size else 0.0

  labels, n = label(window > OCC_THRESH, structure=_NEIGHBOURHOOD)
  objects = []
  for i in range(1, n + 1):
    ys, xs = np.nonzero(labels == i)
    if len(xs) < min_cells:
      continue
    span = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * res)
    if span > max_span:
      continue                       # a fence run, a wall, a piece of furniture
    wx, wy = grid.cell_to_world(float(xs.mean()) + ix0, float(ys.mean()) + iy0)
    # float()/int() rather than the numpy scalars they arrive as: these end up
    # in a result dict that gets json.dumps'd, and numpy scalars are exactly
    # what it refuses (the same trap dynamic_flags documents).
    objects.append({"x": round(float(wx), 3), "y": round(float(wy), 3),
                    "cells": int(len(xs)), "span": round(span, 3)})
  objects.sort(key=lambda o: (o["x"], o["y"]))
  return {"count": len(objects), "objects": objects,
          "coverage": round(coverage, 3), "cells": int(known.size)}


def true_count(model, zone: Zone, prefix: str = "plant",
               margin: float = MARGIN) -> int:
  """The answer, read out of the world. THE EVALUATOR'S, NEVER THE ROBOT'S.

  Derived from the compiled model rather than written down as a number, so a
  world that grows a fifth plant re-scores itself and cannot quietly disagree
  with the garden the robot is standing in. `margin` matches the counter's,
  or an object sitting in the excluded border would be scored as missed when
  it was never countable.
  """
  n = 0
  for b in range(model.nbody):
    name = model.body(b).name or ""
    if not name.startswith(prefix):
      continue
    pos = model.body_pos[b]
    if zone.contains(float(pos[0]), float(pos[1]), margin=margin):
      n += 1
  return n


def score(counted: int, truth: int, coverage: float) -> dict:
  """The deterministic evaluator: code scores the task, never the robot.

  `correct` is the only thing that awards a point. `coverage` rides along
  because a right answer off a third of the zone is a lucky guess, and the
  overseer's ledger should be able to see the difference (design doc,
  "Evaluation -- four tiers").
  """
  return {"counted": counted, "truth": truth, "correct": counted == truth,
          "error": counted - truth, "coverage": round(float(coverage), 3)}
