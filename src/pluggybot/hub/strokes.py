"""Stroke programs: what to draw, separated from how to draw it (issue #11).

`hub/drawing.py` knows how to get a pen tip to a board coordinate. This module
knows what those coordinates should be, and knows nothing else -- no MuJoCo, no
model, no data. That split is the whole point: every later drawing feature (a
daily greeting, a tic-tac-toe move, a figure the robot picked from a menu)
becomes a pure function returning polylines, unit-testable in milliseconds, and
choosing one becomes a MENU LOOKUP rather than path planning.

A **stroke program** is a named list of polylines. Between polylines the pen
lifts; within one it stays down. That is the only concept, and it is enough for
figures, for text, and for the grid-plus-glyph games that come later.

## Coordinates

Board-local metres, `(lat, height)`, centred on the figure's own origin --
exactly what `PenPlotter.draw_program` consumes, which re-centres the whole
program on where the pen actually is.

`lat` is measured LEFT of the robot's approach heading (`PenPlotter.pen_board`),
so **+lat is the LEFT of the board as a viewer facing it sees it** and text
advances toward -lat. Figures that are not left/right symmetric are therefore
authored in the READING frame -- x right, y up, in a unit box -- and flipped
once, in `_from_unit`. One convention, one place it is applied; the font module
does the same thing for JHF's y-down coordinates.

## Size

Programs are sized to the pen's ENVELOPE, not to the board. The home world's
boards are 320 x 260 mm, but the carriage has 110 mm of travel and the base is
parked for the whole drawing, so 110 mm is the widest mark the robot can make
without repositioning. `drawing.Envelope.for_board` intersects the two, and
`StrokeProgram.fits` is what every generator's default parameters are chosen
against.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from pluggybot.hub import hershey
from pluggybot.hub.drawing import Envelope, circle_path, square_path

Point = tuple[float, float]
Polyline = tuple[Point, ...]

FIGURE_SIZE = 0.075       # m, the default figure box: the size the square and
                          # circle diagnostics have always been drawn at, and
                          # comfortably inside the 110 mm carriage stroke.
TEXT_CAP = 0.018          # m of cap height, and MEASURED rather than chosen.
                          # A multi-stroke figure's form error is ~1.1 mm --
                          # nearly twice the 0.6 mm of a single-stroke square,
                          # because every stroke re-presses and each press
                          # seats the module slightly differently. At the 12 mm
                          # caps that first looked reasonable that is 9 % of a
                          # letter's height and the parts of a letter visibly
                          # miss each other; at 18 mm it is 6 % and the word
                          # reads. Legibility, not width, is what sets this --
                          # 18 mm caps only fit ~5 characters per line, so
                          # anything longer wraps or shrinks (see `fit_text`).
TEXT_WIDTH = 0.100        # m of usable line width (110 mm of travel, less a
                          # margin for the loaded calibration's re-zero).
MIN_FEATURE = 0.010       # m. The shortest stroke worth authoring into a
                          # figure, and it is a MACHINE limit, not a taste
                          # one: strokes land with ~4 mm of scatter because
                          # each one re-presses (drawing.draw_program), so a
                          # detail shorter than about 10 mm arrives visibly
                          # detached from what it belongs to. The house's
                          # first window bars were 8 mm and drew as a line
                          # floating outside the window frame; the robot's
                          # first eyes were 4 mm. Text is exempt and sized by
                          # cap height instead -- its short strokes are parts
                          # of a letter the reader reassembles.


@dataclass(frozen=True)
class StrokeProgram:
  """A named figure: polylines in board-local metres, pen up between them."""
  name: str
  strokes: tuple[Polyline, ...]

  def __post_init__(self) -> None:
    if not self.strokes:
      raise ValueError(f"stroke program {self.name!r} draws nothing")
    for s in self.strokes:
      if len(s) < 2:
        raise ValueError(
          f"stroke program {self.name!r} has a stroke with {len(s)} point(s) -- "
          f"a single point is a dot the plotter cannot draw, since ink needs a "
          f"press and a press needs somewhere to go")

  # ---- measurement ---------------------------------------------------------

  def bounds(self) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, z_min, z_max)."""
    ys = [y for s in self.strokes for y, _ in s]
    zs = [z for s in self.strokes for _, z in s]
    return min(ys), max(ys), min(zs), max(zs)

  @property
  def size(self) -> tuple[float, float]:
    y0, y1, z0, z1 = self.bounds()
    return y1 - y0, z1 - z0

  @property
  def points(self) -> int:
    return sum(len(s) for s in self.strokes)

  @property
  def ink_length(self) -> float:
    """Metres of pen-down travel. The honest cost estimate for a drawing --
    at DRAW_SPEED this is most of the errand's clock, and a battery-driven
    mission has to be able to ask before it commits."""
    return sum(math.dist(a, b) for s in self.strokes for a, b in zip(s, s[1:]))

  def fits(self, envelope: Envelope) -> bool:
    return envelope.contains(*self.bounds())

  # ---- transforms ----------------------------------------------------------

  def scaled(self, factor: float) -> "StrokeProgram":
    return self._map(lambda y, z: (y * factor, z * factor))

  def translated(self, dy: float, dz: float) -> "StrokeProgram":
    return self._map(lambda y, z: (y + dy, z + dz))

  def centered(self) -> "StrokeProgram":
    y0, y1, z0, z1 = self.bounds()
    return self.translated(-(y0 + y1) / 2, -(z0 + z1) / 2)

  def fitted(self, envelope: Envelope, margin: float = 0.0) -> "StrokeProgram":
    """Shrink (never grow) until the program fits, then centre it.

    Shrinking rather than clipping, because a clipped figure is a DIFFERENT
    figure drawn silently -- the plotter would happily trace it and the error
    stats would call it a success."""
    prog = self.centered()
    w, h = prog.size
    lat = min(envelope.lat_max, -envelope.lat_min) * 2 - 2 * margin
    tall = min(envelope.z_max, -envelope.z_min) * 2 - 2 * margin
    factor = min(lat / w if w else 1.0, tall / h if h else 1.0, 1.0)
    return prog.scaled(factor) if factor < 1.0 else prog

  def _map(self, fn: Callable[[float, float], Point]) -> "StrokeProgram":
    return StrokeProgram(self.name,
                         tuple(tuple(fn(y, z) for y, z in s)
                               for s in self.strokes))


# ---- the registry ----------------------------------------------------------

PROGRAMS: dict[str, Callable[..., StrokeProgram]] = {}


def register(name: str) -> Callable[[Callable[..., StrokeProgram]],
                                    Callable[..., StrokeProgram]]:
  def wrap(fn):
    if name in PROGRAMS:
      raise ValueError(f"stroke program {name!r} is already registered")
    PROGRAMS[name] = fn
    return fn
  return wrap


def program(name: str, **params) -> StrokeProgram:
  """Build a registered program. This is the LLM's whole interface to drawing
  (issue #15): a name off a menu and a few keyword parameters."""
  if name not in PROGRAMS:
    raise KeyError(f"unknown stroke program {name!r}; "
                   f"have {', '.join(sorted(PROGRAMS))}")
  return PROGRAMS[name](**params)


def from_cli(name: str, size: float | None = None,
             text: str | None = None) -> StrokeProgram:
  """The demo scripts' three knobs -> a program. Shared rather than copied,
  because `--size` means cap height for text and figure box for everything
  else, and two scripts disagreeing about that is a bug waiting to happen."""
  params: dict = {}
  if name == "text":
    if text:
      params["text"] = text
    if size:
      params["cap_height"] = size
  elif size:
    params["size"] = size
  return program(name, **params)


# ---- authoring helpers -----------------------------------------------------


def _from_unit(name: str, strokes: Iterable[Sequence[Point]],
               size: float) -> StrokeProgram:
  """Author in the READING frame (x right, y up, unit box) -> board coords.

  The flip to board lateral (+lat is the viewer's LEFT) happens exactly here,
  and the figure is scaled so its LARGER dimension is `size` -- so `size` means
  the same thing for a tall tree and a wide house."""
  prog = StrokeProgram(name, tuple(tuple((-x, y) for x, y in s)
                                   for s in strokes)).centered()
  w, h = prog.size
  span = max(w, h)
  return prog.scaled(size / span) if span else prog


def _circle(cx: float, cy: float, r: float, n: int = 28) -> list[Point]:
  return [(cx + r * math.cos(2 * math.pi * k / n),
           cy + r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
  return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


# ---- the diagnostics -------------------------------------------------------
# Ported, not rewritten: `square_path` and `circle_path` still produce exactly
# the point sequences they always have. They are the plotter's calibration
# figures and the numbers in PluggyPlan.md (0.57 mm form error on the square)
# were measured against those exact samples -- regenerating them "more cleanly"
# would quietly move the baseline the refactor is supposed to hold.


@register("square")
def square(size: float = FIGURE_SIZE) -> StrokeProgram:
  """One closed square. The FIRST diagnostic, because each edge moves one axis
  alone: an error is attributable to the carriage or to the lift."""
  return StrokeProgram("square", (tuple(square_path(size)),))


@register("circle")
def circle(size: float = FIGURE_SIZE) -> StrokeProgram:
  """One closed circle: every sample moves both axes, so scale mismatch and
  following lag show up as shape distortion."""
  return StrokeProgram("circle", (tuple(circle_path(size)),))


# ---- text ------------------------------------------------------------------


def fit_text(text: str, cap_height: float = TEXT_CAP,
             width: float = TEXT_WIDTH, tracking: float = 0.0,
             shrink: bool = True) -> tuple[float, list[str]]:
  """The cap height and wrapped lines `text` will actually be drawn at.

  Shrinking until the LONGEST WORD fits is what makes this safe to hand to an
  LLM (issue #15): otherwise "GOOD MORNING" at a 12 mm cap silently becomes
  "GOOD MORNI / NG", which is a legal drawing, passes every bounds check, and
  is not what anyone asked for. Split out from `text` so the choice can be
  asserted directly -- reading a cap height back out of a bag of polylines is
  guesswork.
  """
  cap = cap_height
  words = [w for line in text.split("\n") for w in line.split()]
  if shrink and words:
    widest = max(hershey.text_width(w, cap_height, tracking) for w in words)
    if widest > width:
      cap = cap_height * width / widest
  return cap, hershey.wrap(text, cap, width, tracking)


@register("text")
def hershey_text(text: str = "HELLO", cap_height: float = TEXT_CAP,
                 width: float = TEXT_WIDTH, line_gap: float | None = None,
                 tracking: float = 0.0, align: str = "center",
                 shrink: bool = True) -> StrokeProgram:
  """Hershey single-stroke text, wrapped to `width` metres and shrunk to fit
  (see `fit_text`)."""
  if not text.strip():
    raise ValueError("text program needs something to write")
  cap, _ = fit_text(text, cap_height, width, tracking, shrink)
  polys = hershey.layout(text, cap, max_width=width, line_gap=line_gap,
                         tracking=tracking, align=align)
  # Reading frame -> board: +lat is the viewer's LEFT, so text advances -lat.
  return StrokeProgram("text", tuple(tuple((-x, y) for x, y in s)
                                     for s in polys))


# ---- the curated figure library --------------------------------------------
# Hand-authored, and deliberately so: this is the safe content source for
# "draw something" until there is a vision pipeline to derive figures from
# what the robot can actually see. Four figures a person recognises from
# across a room beats a hundred that need explaining.
#
# Every stroke here is at least MIN_FEATURE long at the default size, which is
# a real constraint on the drawing rather than a style rule -- see the comment
# on that constant.

LIBRARY = ("house", "tree", "sun", "robot")


@register("house")
def house(size: float = FIGURE_SIZE) -> StrokeProgram:
  return _from_unit("house", [
    _rect(-0.34, -0.42, 0.34, 0.14),                  # walls
    [(-0.44, 0.14), (0.0, 0.50), (0.44, 0.14)],       # roof
    [(-0.12, -0.42), (-0.12, -0.10), (0.12, -0.10), (0.12, -0.42)],   # door
    _rect(0.10, -0.06, 0.30, 0.10),                   # window
    [(0.20, -0.06), (0.20, 0.10)],                    # window bars
    [(0.10, 0.02), (0.30, 0.02)],
    # The chimney SITS ON the roof line rather than crossing it: its two
    # downstrokes stop where the slope is, which is 0.5 - x/0.44 * 0.36.
    [(0.22, 0.32), (0.22, 0.48), (0.30, 0.48), (0.30, 0.25)],
  ], size)


@register("tree")
def tree(size: float = FIGURE_SIZE) -> StrokeProgram:
  return _from_unit("tree", [
    [(-0.07, -0.50), (-0.07, -0.06)],                 # trunk, two strokes so
    [(0.07, -0.50), (0.07, -0.06)],                   # the pen never doubles back
    [(-0.20, -0.50), (0.20, -0.50)],                  # ground line
    _circle(0.0, 0.16, 0.30, 24),                     # canopy
    [(-0.07, -0.20), (-0.18, -0.30)],                 # roots
    [(0.07, -0.20), (0.18, -0.30)],
  ], size)


@register("sun")
def sun(size: float = FIGURE_SIZE) -> StrokeProgram:
  rays = [[(0.30 * math.cos(a), 0.30 * math.sin(a)),
           (0.50 * math.cos(a), 0.50 * math.sin(a))]
          for a in (math.pi * k / 4 for k in range(8))]
  return _from_unit("sun", [_circle(0.0, 0.0, 0.22, 32), *rays], size)


@register("robot")
def robot(size: float = FIGURE_SIZE) -> StrokeProgram:
  """Pluggy's own portrait: head, face, body, wheels, an arm and the antenna.
  The one figure in the library that has to be asymmetric to read as a robot
  rather than as a box -- hence the single arm."""
  return _from_unit("robot", [
    _rect(-0.24, 0.12, 0.24, 0.42),                   # head
    [(-0.20, 0.30), (-0.04, 0.30)],                   # eyes
    [(0.04, 0.30), (0.20, 0.30)],
    [(-0.11, 0.20), (0.11, 0.20)],                    # mouth
    [(0.0, 0.42), (0.0, 0.58)],                       # antenna
    _circle(0.0, 0.62, 0.05, 12),
    _rect(-0.32, -0.26, 0.32, 0.06),                  # body
    [(-0.18, -0.10), (0.18, -0.10)],                  # coupling face
    [(0.32, -0.02), (0.48, -0.02), (0.48, -0.22)],    # the fork arm
    _circle(-0.18, -0.38, 0.11, 14),                  # wheels
    _circle(0.18, -0.38, 0.11, 14),
  ], size)
