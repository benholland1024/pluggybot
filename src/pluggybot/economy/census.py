"""Counting what stands in an area, out of what the robot saw (issue #13; on
legs, #407).

The census is the first task in this repo with **hidden ground truth**: the
sim knows how many plants stand in the garden, and the robot has to go and
find out. Every earlier criterion was ground truth the robot could not get
wrong by being lazy -- a contact conducts or it does not, ink lands or it
does not. This one is failable in the interesting way: survey half the
garden and you report half the plants, with no error anywhere and a
confident number on the screen.

Which is why the count comes off what the robot SAW rather than out of the
model -- and on legs, off the depth camera: the plants are 0.30 m tall and
the quadruped's LIDAR scans at 0.51 m, over them, so they are the depth
camera's low layer (`legs.body.QuadMission.low`: what it saw 0.08-0.60 m up,
under the LIDAR's plane, kept off the LIDAR's walls). `true_count` reads
the model and is the evaluator's alone. Keeping those two apart is the
whole task -- a census that counted bodies would be a lookup with a walk
attached. ⚠ NOT THE PLANNER'S LAYER ITSELF: it takes a cell back for every
floor point seen in it, and a stalk's cell seen from one side was erased
from another -- a survey counted 2 of 4 plants at 91 % coverage. A survey
keeps its own record (`legs.survey.CensusLayer`: frames that found
something in a cell, never taken back), counted at `HITS` frames.

THE AREA IS FOUND, NOT HANDED OVER (#419, #407): the floor the LIDAR's
walls and fence enclose round where the robot stands, a doorway closed --
any gap narrower than `DOOR_M` (`area_from_walls`). The counting itself is
deliberately dumb: label the low layer's obstacle cells inside the area and
throw away anything the wrong size (`MAX_SPAN`: a plant is an 80 mm stalk,
two or three cells across). COVERAGE is not a filter but a confession: the
share of the area's floor the depth camera has seen at all. A count of 2
with 45 % coverage is not the same claim as a count of 2 with 95 %, and a
scoring rule that ignores the difference rewards a robot for stopping early.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: The widest gap in a wall that is a doorway, m: a doorway closes the area
#: (every one in the house is 1.0 m), a gap wider than this does not.
DOOR_M = 1.2
#: m: the widest a counted object may be. Garden plants are 80 mm; the next
#: thing up in the home world is a 1 m couch.
MAX_SPAN = 0.35
#: A cell is something standing once this many of the survey's frames
#: found it there (`legs.survey.CensusLayer`): one is a stray point.
HITS = 3

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

  def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
    return (self.min[0] + margin <= x <= self.max[0] - margin
            and self.min[1] + margin <= y <= self.max[1] - margin)


def area_from_walls(occupied: np.ndarray, free: np.ndarray, seed: tuple[int, int],
                    resolution: float, door_m: float = DOOR_M) -> np.ndarray:
  """The floor round `seed` (a cell, (ix, iy)) that the walls enclose: the
  known-free cells reachable from it without passing a gap narrower than
  `door_m` -- an OPENING of the free floor by half a door (its middle kept
  where it is half a door from every wall, the one piece holding the seed
  grown back over the free floor, a cell a step, ⚠ NEVER THROUGH A WALL: by
  distance alone it took in the floor behind a wall a cell thick). Empty
  where the seed is no such floor."""
  r = door_m / 2.0
  clear = ndimage.distance_transform_edt(~occupied) * resolution
  core = free & (clear >= r)
  labels, _ = ndimage.label(core, structure=_NEIGHBOURHOOD)
  ix, iy = seed
  rows, cols = free.shape
  if not (0 <= iy < rows and 0 <= ix < cols) or labels[iy, ix] == 0:
    return np.zeros_like(free, dtype=bool)
  return ndimage.binary_dilation(labels == labels[iy, ix], structure=_NEIGHBOURHOOD,
                                 iterations=int(math.ceil(r / resolution)) + 1,
                                 mask=free & ~occupied)


def count_low(obstacle: np.ndarray, area: np.ndarray, resolution: float,
              max_span: float = MAX_SPAN) -> list[tuple[float, float, int]]:
  """Each object the low layer saw inside the area: (ix, iy) of its middle
  and its cells, one per 8-connected patch no wider than `max_span`."""
  labels, n = ndimage.label(obstacle & area, structure=_NEIGHBOURHOOD)
  out = []
  for i in range(1, n + 1):
    ys, xs = np.nonzero(labels == i)
    span = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * resolution)
    if span > max_span:
      continue                     # a hedge, a wall's foot, a piece of furniture
    out.append((float(xs.mean()), float(ys.mean()), int(len(xs))))
  return out


def seen_share(seen: np.ndarray, area: np.ndarray) -> float:
  """The share of the area's floor the depth camera saw, either way."""
  cells = int(area.sum())
  if not cells:
    return 0.0
  return float((seen & area).sum()) / cells


def true_count(model, zones, prefix: str = "plant") -> int:
  """The answer, read out of the world. THE EVALUATOR'S, NEVER THE ROBOT'S.

  Derived from the compiled model rather than written down as a number, so a
  world that grows a fifth plant re-scores itself and cannot quietly disagree
  with the garden the robot is standing in. `zones` are the rectangles the
  area is (the garden's two, `home.world.ZONES`)."""
  n = 0
  for b in range(model.nbody):
    name = model.body(b).name or ""
    if not name.startswith(prefix):
      continue
    x, y = (float(v) for v in model.body_pos[b][:2])
    if any(z.contains(x, y) for z in zones):
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
