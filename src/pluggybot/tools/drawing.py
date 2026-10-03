"""The whiteboards and the pen on the arm (issue #406): where a board is, the
tags it is found by, where on it a figure may go, and the plotter that draws
with the quadruped's arm.

THE PLOTTER is two axes the robot has and one it acquires:

  lateral   the pen module's own carriage (`legs.rack`: a lead screw, +-50
            mm) -- the arm moves in its own plane and has none
  height    the arm's fork, up and down that plane (`legs.arm`)
  pressure  the fork driven into the board onto the pen's sprung quill: the
            spring sets the force, the arm only how far in

The body LIES in front of the board while it draws (`legs/draw.py`): with a
pen held pressed for 30 s its torso drifted 0.25 mm, against 5 mm standing
on the walking policy, and a square came out at 0.24 mm of form error
against 0.86 (SimNotes, "Drawing on legs").

CALIBRATION READS NO GROUND TRUTH, which the rover's did (`rover-final`: it
read its pen's tip off the sim, PluggyPlan's "Road to hardware"). What the
plotter steers by is the robot's own: the arm's joint encoders through its
kinematics, the carriage's position, the quill's Hall sensor, and the
board's tags through the nose camera (where the board is, roughly, before
it lies down). The board's plane is found by TOUCH: the pen driven in at
four points until the quill reads contact (`calibrate_routine`). Scale needs
no calibrating: the carriage is a lead screw with a position sense, and the
height is the arm's own kinematics.

What the plotter RECORDS -- where the tip went, and whether it touched -- is
the world's: the trace, the ink the board book keeps, the error stats the
evaluator reads. Nothing that steers reads it.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from pluggybot.tick import Routine


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

  @property
  def normal(self) -> tuple[float, float]:
    """The face's outward direction on the floor."""
    return -math.cos(self.heading), -math.sin(self.heading)

  @property
  def left(self) -> tuple[float, float]:
    """+lat: left of the approach heading, the viewer's LEFT."""
    return -math.sin(self.heading), math.cos(self.heading)

  def local(self, p) -> tuple[float, float, float]:
    """A world point as (lat, height, out of the face) about the board's
    middle."""
    dx, dy = float(p[0]) - self.x, float(p[1]) - self.y
    lx, ly = self.left
    nx, ny = self.normal
    return (dx * lx + dy * ly, float(p[2]) - self.z,
            dx * nx + dy * ny - self.half[0])


# ---- the boards' tags (#406) ------------------------------------------------------

#: A board's two tags hang on the wall either side of it, level with its
#: middle, this far out from it (m, centre to centre): the board's half-width
#: (0.16), 30 mm, and the tag's plate (`tags.plate_half_extent`, 75 mm). The
#: board's drawing, which a robot knows before it sees one: its middle is
#: halfway between them, and its face `BOARD_PROUD_M` in front of them.
BOARD_TAG_LAT = 0.265
BOARD_PROUD_M = 0.019


def board_tags(name: str) -> tuple[int, int]:
  """A board's tags, (on its left, on its right) as one faces it."""
  from pluggybot.rack.tags import BOARD_TAG_IDS
  return BOARD_TAG_IDS[name]


def board_fixture(name: str):
  """A board's two tags as one fixture (`mapping.places.Fixture`): in the
  board's frame (x along +lat, y out of the face), their faces along +y."""
  from pluggybot.mapping.places import Fixture
  left, right = board_tags(name)
  return Fixture(layout={left: (BOARD_TAG_LAT, 0.0), right: (-BOARD_TAG_LAT, 0.0)},
                 facing=math.pi / 2)


def board_tags_xml(name: str, board: Board) -> str:
  """A board's two tags on the wall either side of it, as worldbody MJCF:
  each a thin plate facing out of the wall, its texture the tag's
  (`rack.tags.asset_xml`). Put in at load by the quadruped's world, after
  its robots (`legs/world.py`): the house's body order is the day's hash."""
  from pluggybot.rack.tags import BOARD_TAG_SIZE, plate_half_extent
  half = plate_half_extent(BOARD_TAG_SIZE)
  nx, ny = board.normal
  lx, ly = board.left
  # the wall's face: the board's back, `half[0]` behind its centre
  wx, wy = board.x - nx * board.half[0], board.y - ny * board.half[0]
  yaw = math.atan2(ny, nx)
  out = []
  for tag, lat in zip(board_tags(name), (BOARD_TAG_LAT, -BOARD_TAG_LAT)):
    x, y = wx + nx * 0.001 + lx * lat, wy + ny * 0.001 + ly * lat
    out.append(f'<body name="{name}_tag{tag}" pos="{x:.4f} {y:.4f} {board.z:.4f}" '
               f'quat="{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}">'
               f'<geom name="{name}_tag{tag}" type="box" size="0.001 {half:.4f} {half:.4f}" '
               f'contype="0" conaffinity="0" material="tagmat{tag}"/></body>')
  return "".join(out)


# ---- the pen, as the arm holds it ------------------------------------------------

def tip_from_vertex() -> tuple[float, float]:
  """The pen's point off the fork's V vertex, (ahead, up) in the torso
  frame, the module hanging plumb: its drawing (`legs.rack`), never measured
  off the sim. A tool faces the robot, so its -x is the robot's ahead. ⚠
  THE POINT, `PEN_TIP_R` past the tip site: with the site, every face the
  probes found was 2 mm short (`test_the_board_is_found_by_touch_...`)."""
  from pluggybot.legs import rack as rk
  from pluggybot.legs.arm import ArmSpec
  from pluggybot.rack.coupling import PEG_ABOVE_BODY
  ahead = -(rk.PEN_MOUNT_X - 0.008 - rk.PEN_LEN) + rk.PEN_TIP_R
  up = ArmSpec().fork.seat_rise() - PEG_ABOVE_BODY + rk.PEN_RAIL_Z + rk.PEN_LINE_DZ
  return ahead, up


#: m of the board's face discounted from its edge: a figure is centred on
#: where the pen is, and the body lies a few cm off the board's middle.
HOME_ALLOWANCE = 0.030


@dataclass(frozen=True)
class Envelope:
  """Where on the board a figure may go, in board-local metres relative to
  the pen's home (issue #11): the INTERSECTION of the carriage's travel and
  the board's own face less `HOME_ALLOWANCE` -- ink off the edge of the
  slab is not ink. The arm reaches the face's whole height from where the
  body lies (`legs.draw`; `tests/test_drawing.py` reads it)."""
  lat_min: float
  lat_max: float
  z_min: float
  z_max: float

  @classmethod
  def for_board(cls, board: "Board") -> "Envelope":
    from pluggybot.legs.rack import PEN_TRAVEL
    face_lat = board.half[1] - HOME_ALLOWANCE
    face_z = board.half[2] - HOME_ALLOWANCE
    lat = min(PEN_TRAVEL, face_lat)
    return cls(lat_min=-lat, lat_max=lat, z_min=-face_z, z_max=face_z)

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


def pen_on_board(model, data, board_geom: str) -> bool:
  """Is the pen's shaft touching the board? Ink is a contact fact -- the
  WORLD's, what the board book and the evaluator read, never the plotter's
  steering (module docstring)."""
  from pluggybot.legs.rack import PEN_SHAFT
  from pluggybot.rack.coupling import geom_id, touching
  shaft, board = geom_id(model, PEN_SHAFT), geom_id(model, board_geom)
  return shaft is not None and board is not None and touching(data, shaft, (board,))


# ---- the plotter ----------------------------------------------------------------

#: m/s along a drawn path: the rover's. The carriage's share of it is under
#: the slide's 25 mm/s.
DRAW_SPEED = 0.020
#: m/s of the fork between strokes, the pen off the board, and into it.
TRAVEL_V = 0.05
PRESS_V = 0.02
#: The probe: the fork walks in at PROBE_V until the quill reads QUILL_TOUCH
#: (its Hall sensor, `quill`), from PROBE_CLEAR short of where the tags put
#: the face, at most PROBE_PAST beyond it (m/s, m).
PROBE_V = 0.005
QUILL_TOUCH = 0.0005
PROBE_CLEAR = 0.030
PROBE_PAST = 0.040
#: Where it probes, off the figure's middle: the carriage either way, the
#: fork up and down (m). Four points for three unknowns, so one bad touch
#: shows in the residual (`MAX_PLANE_RMS`).
PROBE_LAT, PROBE_Z = 0.035, 0.060
MAX_PLANE_RMS = 0.002
#: The fork stands this far past the touch plane while it draws (m): the
#: module swung onto its lean-pad (the first ~5 mm past a touch) and the
#: quill in by the rest.
PRESS_EXTRA = 0.010
#: ...and this far short of it between strokes, the pen clear of the board.
LIFT = 0.012
#: A press holds this long before the stroke (s): the module settles onto the
#: pad.
PRESS_SETTLE_S = 0.3
#: The Hall sensor on the quill: what it reads is the quill's travel with
#: this much noise (m), drawn per physics step and robot (`axes.noise`'s
#: crc32 seeding, never `hash()`).
QUILL_NOISE = 5e-5


def _quill_noise(prefix: str, step: int) -> float:
  seed = zlib.crc32(f"quill:{prefix}:{step}".encode())
  return float(np.random.default_rng(seed).normal(0.0, QUILL_NOISE))


@dataclass(frozen=True)
class BoardEstimate:
  """Where the robot believes a board is, in its TORSO frame: the face's
  distance ahead (x), its middle's lateral (y, + left) and height (z)."""
  x_face: float
  y_mid: float
  z_mid: float


class PenPlotter:
  """Draws with the pen on the arm, the body lying (the module docstring).
  `mission` is the quadruped's (`legs.body.QuadMission`): its arm, its
  steps, its encoders. `board` is the world's board, read only for what the
  plotter RECORDS (`trace`), and `on_stroke(i, points, program)` hears each
  stroke's ink as it finishes."""

  def __init__(self, mission, board: Board, on_stroke=None) -> None:
    from pluggybot.legs import rack as rk
    self.mission, self.board = mission, board
    self.model, self.data = mission.model, mission.data
    self.arm, self.spec = mission.arm, mission.arm_spec
    self.on_stroke = on_stroke
    #: `should_stop()`, asked before each stroke with the pen up (issue
    #: #116): the one place in a drawing where stopping is legal.
    self.should_stop = None
    m = self.model
    self.pen_act = m.actuator(rk.PEN_ACTUATOR).id
    self._quill = int(m.jnt_qposadr[m.joint(rk.PEN_QUILL_JOINT).id])
    self._tip = m.site(rk.PEN_TIP).id
    self.tip_ahead, self.tip_up = tip_from_vertex()
    self.cal: dict = {}
    #: The figure's home across the board, torso y (+ left): the board's
    #: middle as far as the carriage's travel lets the figure go there.
    self.lat0 = 0.0
    #: (t, lat, height, commanded lat, commanded height, touching, stroke),
    #: board-local and the WORLD's (`_trace`); `stroke` is -1 travelling.
    self.trace: list[tuple[float, float, float, float, float, bool, int]] = []
    self.commanded: tuple = ()
    self._home: np.ndarray | None = None
    #: The posture it draws in (lying, or the premise's standing) and the
    #: body's falls so far: a body that leaves it, falls, or holds the pen no
    #: more fell, and its fall folded the arm -- the plotter aims it no more
    #: (`_aim`). ⚠ NOT THE POSTURE ALONE: a fall on the way in was up and
    #: lying again by the stance, and the pen it threw lay on the floor.
    self.posture = mission.posture
    self._falls = mission.falls

  # ---- what the robot senses ---------------------------------------------------

  def quill(self) -> float:
    """The quill's travel off its rest, m: its Hall sensor."""
    step = int(round(float(self.data.time) / float(self.model.opt.timestep)))
    return float(self.data.qpos[self._quill]) + _quill_noise(self.mission.handle.prefix, step)

  def vertex(self) -> tuple[float, float]:
    """The fork's V vertex in the torso frame, off the arm's encoders and
    its kinematics (`legs.arm`)."""
    from pluggybot.legs import arm as am
    qs, qf = self.arm.q()
    wx, wz = am.wrist_xz(self.spec, qs, qf - qs)
    return wx + self.spec.fork.vertex_x, wz + self.spec.fork.vertex_z

  def carriage(self) -> float:
    """Where the carriage is sent: the slide's own position loop holds it."""
    return float(self.data.ctrl[self.pen_act])

  # ---- what the world records ----------------------------------------------------

  def pen_board(self) -> tuple[float, float]:
    """The tip on the board's plane, (lat, height) about its middle: the
    WORLD's, for the trace."""
    lat, h, _ = self.board.local(self.data.site_xpos[self._tip])
    return lat, h

  def _trace(self, stroke: int, cmd=None) -> None:
    """One sample of the tip; `cmd`, the figure's point about its home, put
    where the first press landed (`_mark_home`)."""
    lat, h = self.pen_board()
    c = (math.nan, math.nan) if cmd is None else (cmd[0] + self._home[0], cmd[1] + self._home[1])
    self.trace.append((float(self.data.time), lat, h, float(c[0]), float(c[1]),
                       pen_on_board(self.model, self.data, self.board.geom), stroke))

  def _mark_home(self, start) -> None:
    """THE INSTRUMENT'S ALIGNMENT, never the plotter's: the figure the stats
    score against sits where the first press landed (the WORLD says where),
    so `shape` is the drawing's own error after that."""
    if self._home is None:
      self._home = np.subtract(self.pen_board(), start)

  def _drawn(self, program, i: int) -> None:
    """Stroke `i`'s ink to whoever records it (`on_stroke`)."""
    if self.on_stroke is not None:
      self.on_stroke(i, self.inked_polyline(i), getattr(program, "name", None))

  def _report(self, n: int, drawn: int, stopped: str) -> dict:
    """What a figure came to, off the WORLD's trace: `drew` is ink landed
    (a stroke of two touching samples), never a press made."""
    inked = sum(len(self.inked_polyline(i)) >= 2 for i in range(n))
    return {"drew": inked > 0, "strokes": n, "strokes_drawn": drawn,
            **({"stopped": stopped} if stopped else {}), **self.error_stats()}

  # ---- the motions --------------------------------------------------------------

  def _still_routine(self, seconds: float = 0.0, record: int | None = None) -> Routine:
    n = max(1, round(seconds / self.model.opt.timestep)) if seconds else 1
    for _ in range(n):
      yield from self.mission._twist_routine(0.0, 0.0, 0.0)
      if record is not None:
        self._trace(record)

  def fell(self) -> bool:
    """Did the body fall since it took its stance, or lose the pen?"""
    m = self.mission
    return (m.posture != self.posture or m.falls != self._falls
            or m.carrying != "module_pen")

  def _aim(self, x: float, z: float) -> bool:
    """The fork's vertex aimed at (x, z): False out of reach, or once the
    body has fallen (`fell`)."""
    from pluggybot.legs import arm as am
    if self.fell():
      return False
    g = self.arm.goal
    q = am.solve_vertex(self.spec, x, z, near=(float(g[0]), float(g[1] - g[0])))
    if q is None:
      return False
    self.arm.aim(*q)
    return True

  def _goal_vertex(self) -> tuple[float, float]:
    return self.mission._vertex_goal()

  def move_routine(self, x: float, z: float, speed: float = TRAVEL_V,
                   carriage: float | None = None, record: int | None = None) -> Routine:
    """The fork's vertex along a straight line to (x, z), torso frame, and
    the carriage with it, at most `speed` and the slide's own: False if a
    point on the way is out of the arm's reach."""
    from pluggybot.legs.rack import PEN_SPEED
    x0, z0 = self._goal_vertex()
    c0 = self.carriage()
    c1 = c0 if carriage is None else carriage
    dt = self.model.opt.timestep
    n = max(1, int(max(math.hypot(x - x0, z - z0) / speed, abs(c1 - c0) / PEN_SPEED) / dt))
    for k in range(1, n + 1):
      f = k / n
      if not self._aim(x0 + (x - x0) * f, z0 + (z - z0) * f):
        return False
      self.data.ctrl[self.pen_act] = c0 + (c1 - c0) * f
      yield from self._still_routine(record=record)
    return True

  # ---- calibration ----------------------------------------------------------------

  def _probe_routine(self, carriage: float, z: float, x_from: float,
                     x_to: float) -> Routine:
    """The pen in from `x_from` to `x_to` at PROBE_V, the carriage at
    `carriage` and the fork at `z`, until the quill reads a touch: the
    fork's vertex x then, off the encoders, or None; and back out."""
    if not (yield from self.move_routine(x_from, z, carriage=carriage)):
      return None
    yield from self._still_routine(0.2)
    touched = None
    dt = self.model.opt.timestep
    n = max(1, int((x_to - x_from) / PROBE_V / dt))
    for k in range(1, n + 1):
      if not self._aim(x_from + (x_to - x_from) * k / n, z):
        break
      yield from self._still_routine()
      if self.quill() >= QUILL_TOUCH:
        touched = self.vertex()[0]
        break
    yield from self.move_routine(x_from, z)
    return touched

  def calibrate_routine(self, est: BoardEstimate) -> Routine:
    """The board's plane in the arm's coordinates, by touch: four probes
    round the figure's middle, and a least-squares plane `x = a + b * c +
    k * z` through where the quill first read the board (c the carriage,
    z the fork's height, x its reach). The figure's home: the carriage's
    middle, and the fork where the tip meets the board's middle as the tags
    placed it. Nothing here reads the sim's tip."""
    zm = est.z_mid - self.tip_up
    x_face = est.x_face - self.tip_ahead
    rows = []
    for c, dz in ((PROBE_LAT, -PROBE_Z), (-PROBE_LAT, -PROBE_Z),
                  (-PROBE_LAT, PROBE_Z), (PROBE_LAT, PROBE_Z)):
      x = yield from self._probe_routine(c, zm + dz, x_face - PROBE_CLEAR,
                                         x_face + PROBE_PAST)
      if x is not None:
        rows.append((c, zm + dz, x))
    self.cal = {"probes": len(rows), "zHome": zm, "yMid": est.y_mid,
                "touches": [list(r) for r in rows]}
    if len(rows) < 3:
      self.cal["ok"] = False
      return self.cal
    a = np.array([[1.0, c, z] for c, z, _ in rows])
    b = np.array([x for _, _, x in rows])
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    rms = float(np.sqrt(np.mean((a @ coef - b) ** 2)))
    self.cal.update(ok=rms <= MAX_PLANE_RMS, plane=[float(v) for v in coef],
                    rmsMm=round(rms * 1000, 3),
                    yawDeg=round(math.degrees(math.atan(float(coef[1]))), 2))
    return self.cal

  def _x_at(self, carriage: float, z: float, past: float) -> float:
    a, b, k = self.cal["plane"]
    return a + b * carriage + k * z + past

  # ---- drawing --------------------------------------------------------------------

  def _targets(self, lat: float, h: float) -> tuple[float, float]:
    """A board-local point of the figure -> (carriage, fork z). The tool
    faces the robot, so the carriage's + is the board's -lat."""
    return -(self.lat0 + lat), self.cal["zHome"] + h

  def _home_across(self, strokes) -> float:
    """Where across the board the figure goes, torso y: on the board's
    middle (`cal["yMid"]`), moved as little as the carriage needs to reach
    all of the figure."""
    from pluggybot.legs.rack import PEN_TRAVEL
    lats = [p[0] for s in strokes for p in s]
    lo, hi = -PEN_TRAVEL - min(lats), PEN_TRAVEL - max(lats)
    return min(max(float(self.cal.get("yMid", 0.0)), lo), hi) if lo <= hi else 0.0

  def draw_program_routine(self, program) -> Routine:
    """Draw a stroke program: polylines in board coordinates about the
    figure's home, the pen UP between them (issue #11). Each stroke: the
    pen lifted, moved to its start, pressed onto the plane the probes
    found, and traced. Returns what it drew and the trace's error stats."""
    strokes = [list(s) for s in getattr(program, "strokes", program)]
    if not strokes:
      return {"drew": False, "reason": "the program draws nothing"}
    if not self.cal.get("ok"):
      return {"drew": False, "reason": "never found the board's face by touch",
              "strokes": len(strokes), "strokes_drawn": 0}
    self.commanded = tuple(tuple(s) for s in strokes)
    self.lat0 = self._home_across(strokes)
    ts = self.model.opt.timestep
    drawn, stopped = 0, ""
    for i, path in enumerate(strokes):
      if self.fell():
        stopped = "fell"
        break
      if self.should_stop is not None and self.should_stop():
        stopped = "interrupted"
        break
      c0, z0 = self._targets(*path[0])
      # Up, across, and in: the pen never drags to a start. Only the move
      # ACROSS is a travel row (-1): ink there is a line nothing asked for,
      # where the lift off and the press on cross no board. The first
      # stroke's approach is no travel at all.
      z_now = self._goal_vertex()[1]
      if not ((yield from self.move_routine(self._x_at(self.carriage(), z_now, -LIFT),
                                            z_now))
              and (yield from self.move_routine(self._x_at(c0, z0, -LIFT), z0, carriage=c0,
                                                record=None if i == 0 else -1))
              and (yield from self.move_routine(self._x_at(c0, z0, PRESS_EXTRA), z0,
                                                speed=PRESS_V))):
        if self.fell():
          stopped = "fell"
          break
        continue                       # out of reach: a gap, not the figure's end
      yield from self._still_routine(PRESS_SETTLE_S)
      self._mark_home(path[0])
      drawn += 1
      for (ay, az), (by, bz) in zip(path, path[1:]):
        steps = max(int(math.hypot(by - ay, bz - az) / DRAW_SPEED / ts), 1)
        for k in range(steps):
          f = (k + 1) / steps
          ly, lz = ay + (by - ay) * f, az + (bz - az) * f
          c, z = self._targets(ly, lz)
          self.data.ctrl[self.pen_act] = c
          if not self._aim(self._x_at(c, z, PRESS_EXTRA), z):
            break
          yield from self.mission._twist_routine(0.0, 0.0, 0.0)
          self._trace(i, (ly, lz))
        if self.fell():
          break
      self._drawn(program, i)
      if self.fell():
        stopped = "fell"
        break
    z_now = self._goal_vertex()[1]
    yield from self.move_routine(self._x_at(self.carriage(), z_now, -LIFT), z_now)
    return self._report(len(strokes), drawn, stopped)

  def inked_polyline(self, stroke: int) -> list[tuple[float, float]]:
    """Where the tip was, board-local, for the samples of `stroke` that
    were TOUCHING: the mark it left, not the one it was asked to leave."""
    return [(t[1], t[2]) for t in self.trace if t[6] == stroke and t[5]]

  def error_stats(self) -> dict:
    """`track`, the tip against where it was meant to be AT THAT MOMENT;
    `shape`, each inked point against the NEAREST point of the figure (lag
    along the path is no error); and `form`, the shape after the best
    translation -- the wrong SHAPE apart from the wrong PLACE (the rover's
    plotter, `rover-final`)."""
    drawing = [t for t in self.trace if t[6] >= 0]
    travel = [t for t in self.trace if t[6] < 0]
    inked = [t for t in drawing if t[5]]
    if not inked or not self.commanded or self._home is None:
      return {"inked_fraction": 0.0, "track_rms_mm": None, "shape_rms_mm": None}
    track = [math.hypot(py - cy, pz - cz) for _, py, pz, cy, cz, _, _ in inked]
    pts = np.array([[t[1], t[2]] for t in inked])
    poly = [np.asarray(s, dtype=float) + self._home for s in self.commanded]
    shape, _ = _nearest(pts, poly)
    offset = np.zeros(2)
    for _ in range(6):                       # translation-only ICP
      _, proj = _nearest(pts + offset, poly)
      step = (proj - (pts + offset)).mean(axis=0)
      offset = offset + step
      if np.linalg.norm(step) < 1e-6:
        break
    form, _ = _nearest(pts + offset, poly)
    return {
      "inked_fraction": len(inked) / len(drawing),
      # ink laid while TRAVELLING between strokes: a line nothing commanded
      "travel_ink_fraction": (sum(t[5] for t in travel) / len(travel)
                              if travel else 0.0),
      "track_rms_mm": math.sqrt(sum(e * e for e in track) / len(track)) * 1000,
      "track_max_mm": max(track) * 1000,
      "shape_rms_mm": float(np.sqrt((shape ** 2).mean()) * 1000),
      "shape_max_mm": float(shape.max() * 1000),
      "offset_mm": float(np.linalg.norm(offset) * 1000),
      "form_rms_mm": float(np.sqrt((form ** 2).mean()) * 1000),
      "form_max_mm": float(form.max() * 1000),
      "samples": len(self.trace),
    }


def _nearest(pts: np.ndarray, polys):
  """(distance, closest point on the figure) for each point. Segments are
  built per polyline: joined end to start, a line the pen was never asked
  to draw would score ink as a good drawing."""
  a = np.concatenate([np.asarray(p, dtype=float)[:-1] for p in polys])
  b = np.concatenate([np.asarray(p, dtype=float)[1:] for p in polys])
  seg = b - a
  denom = np.maximum((seg * seg).sum(1), 1e-12)
  rel = pts[:, None, :] - a[None, :, :]
  t = np.clip((rel * seg[None, :, :]).sum(2) / denom[None, :], 0.0, 1.0)
  proj = a[None, :, :] + t[:, :, None] * seg[None, :, :]
  d = np.sqrt(((pts[:, None, :] - proj) ** 2).sum(2))
  k = d.argmin(1)
  return d.min(1), proj[np.arange(len(pts)), k]
