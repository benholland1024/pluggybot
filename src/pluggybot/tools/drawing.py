"""The whiteboards' geometry: where a board is and where on it a figure may go.

What drew on them was the rover's pen on its lift (milestone 8), deleted with
the rover (#376; `rover-final` has the plotter). #406 puts the pen on the
quadruped's arm and re-derives the reach; until then `Envelope.for_board` is
the rover's reach, stated as numbers, because the board state's cells and fill
(`tools/boards.py`) are laid on it and a served board must not move.
"""

import math
from dataclasses import dataclass

PEN_MODULE = "module_pen"


@dataclass(frozen=True)
class Board:
  """A drawing surface (issue #6): where it is, which geom is its face (ink
  is a contact fact, so the geom name matters), and `heading` -- the heading
  a robot squares up with, facing INTO the board's outward normal. `half` is
  (depth, width, height) in the board's own frame, whatever world axes the
  slab happens to lie along."""
  geom: str
  x: float
  y: float
  z: float
  half: tuple[float, float, float]
  heading: float

  @classmethod
  def from_meta(cls, spec: dict) -> "Board":
    """A board from the home generator's meta sidecar entry."""
    return cls(spec["geom"], *spec["pos"], tuple(spec["half"]),
               spec["heading"])


#: The rover pen's reach (`rover-final`'s `drawing.py`), which the board
#: state is laid on until #406: the carriage's +-55 mm along the board, and
#: the lift's 0.02-0.30 m, its zero `LIFT_ZERO_BELOW` under the board's
#: centre (the pen tip hung 72 mm below the peg, the peg 172 mm over the lift).
PEN_TRAVEL = 0.055
LIFT_MIN, LIFT_MAX = 0.02, 0.30
LIFT_ZERO_BELOW = 0.100
#: m of the board's face discounted from its edge: a figure is centred on
#: where the pen actually is, and its home sat up to 23 mm off centre.
HOME_ALLOWANCE = 0.030


@dataclass(frozen=True)
class Envelope:
  """Where on the board a figure may go, in board-local metres relative to
  the pen's home (issue #11): the INTERSECTION of the pen's lateral travel,
  its vertical range, and the board's own face less `HOME_ALLOWANCE` -- ink
  off the edge of the slab is not ink."""
  lat_min: float
  lat_max: float
  z_min: float
  z_max: float

  @classmethod
  def for_board(cls, board: "Board") -> "Envelope":
    lift0 = board.z - LIFT_ZERO_BELOW
    face_lat = board.half[1] - HOME_ALLOWANCE
    face_z = board.half[2] - HOME_ALLOWANCE
    return cls(
      lat_min=max(-PEN_TRAVEL, -face_lat),
      lat_max=min(PEN_TRAVEL, face_lat),
      z_min=max(LIFT_MIN - lift0, -face_z),
      z_max=min(LIFT_MAX - lift0, face_z),
    )

  def contains(self, lat_min: float, lat_max: float,
               z_min: float, z_max: float) -> bool:
    return (lat_min >= self.lat_min and lat_max <= self.lat_max
            and z_min >= self.z_min and z_max <= self.z_max)

  @property
  def size(self) -> tuple[float, float]:
    return self.lat_max - self.lat_min, self.z_max - self.z_min


def square_path(size: float = 0.075, n: int = 240) -> list[tuple[float, float]]:
  """A closed square in board coordinates, centred on the origin: each edge
  moves one axis alone, so an error can be put on one axis."""
  h = size / 2
  corners = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]
  pts = []
  for a, b in zip(corners, corners[1:]):
    for k in range(n // 4):
      f = k / (n // 4)
      pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
  pts.append(corners[-1])
  return pts


def circle_path(size: float = 0.075, n: int = 240) -> list[tuple[float, float]]:
  """A closed circle: every sample moves both axes."""
  r = size / 2
  return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n))
          for k in range(n + 1)]
