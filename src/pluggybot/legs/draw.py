"""Drawing on a whiteboard, on legs (issue #406): the walk to a board the
robot has found (#419's places: its two tags), lying down in front of it, the
pen on the arm (`tools/drawing.py`'s plotter), and back up and out. A mixin:
`QuadMission` is the rest of the body.

THE APPROACH. To `LOOK_M` out of the board's face, facing it, where both its
tags are in the nose camera's view: a look fits the board's facing to their
baseline (#88's rule, `Places.facing`) and reads their height above the
floor (the IMU's level, the legs' kinematics). Then the walk in, steered by
the tags as a plate's press is (`rack.walk_in_twist`), to `STAND_M`; the tags
leave the view in its last ~0.15 m, and that is walked on the belief.

THE STANCE IS LYING. Standing on the walking policy the body keeps turning
(the rack's `SETTLE_DRIFT`), and a pen held to the board wandered 3.6 mm in
30 s; lying, 0.08 mm, and lying costs 39 W less. The lie-down moves the body
`dock.LIE_SHIFT_M` back, so it stands that much nearer. From the floor the
arm reaches the board's whole height (`tests/test_drawing.py`).

THE BOARD BY TOUCH. Lying, the nose camera is under the tags' view, so the
board's plane is found with the pen (`PenPlotter.calibrate_routine`), from
where the tags put it.
"""

from __future__ import annotations

import contextlib
import math
from collections import deque

import numpy as np

from pluggybot.legs import arm as am
from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.tick import Routine

#: The look's distance out of the board's face, m: both tags (0.53 m apart,
#: 120 mm) in the nose camera's 67 deg from 0.73 m on, and decoded square-on
#: past 5 m. ⚠ The walk in from it is what lines the body up across the
#: board: from 1.0 m (0.42 m of walk) one of six lay 5 cm off its axis and
#: 5.5 deg askew; the rack's walk in is a metre.
LOOK_M = 1.4
#: Lying, the torso's centre this far out of the face, m: the pen 0.50 ahead
#: of it, inside the arm's reach at every row of the board, the front feet
#: 0.30 m off the wall...
LIE_M = 0.55
#: ...so it stands this far out before it lies down (the lie-down's shift).
STAND_M = LIE_M + dk.LIE_SHIFT_M
#: The walk in: its budget, s, and how often it looks, s.
WALK_IN_S = 20.0
LOOK_EVERY_S = 0.25
#: Stopped, how far off the stance it may believe it is and still lie down,
#: m and rad; else it backs out to the look point and walks in again, at
#: most `WALK_IN_TRIES` times. Across is what a figure has little room for:
#: the carriage's 100 mm hold a two-digit answer's 81 with 9.5 mm a side.
LINEUP_ACROSS, LINEUP_YAW = 0.02, math.radians(4.0)
WALK_IN_TRIES = 3
#: Stopped, it stands this long before it lies down, s: the policy's coast.
STOPPED_S = 0.8
#: Walking back out from the board when it is done, m, s.
BACK_OUT_M, BACK_OUT_S = 0.6, 6.0
#: A walk to the look point gives up after this long, s.
WALK_PATIENCE_S = 90.0
#: Not both tags in the first look: turns from there, deg, a look after each.
SEARCH_DEG = (12.0, -24.0, 36.0, -48.0)
#: The fork's way off the board when the drawing is done: back this far
#: (torso frame, m) before it rises to the carry pose.
CLEAR_BACK_M = 0.06
#: A board's height is the median of its tags' last this many readings.
HEIGHT_LOOKS = 40


class BoardWork:
  """The whiteboards' routines (the module docstring), on `QuadMission`."""

  def _init_draw(self) -> None:
    #: The boards' heights above the floor, off the tags' last looks (m).
    self._board_z: dict[str, deque] = {}
    #: The last drawing's record, and its plotter (the trace, the stats).
    self.last_draw: dict | None = None
    self.last_plotter = None

  # ---- where a board is ---------------------------------------------------------

  def board_frame(self, name: str) -> tuple[float, float, float] | None:
    """A board's middle on the floor and its face's direction (x, y,
    facing), in the map, off the places its tags are -- halfway between
    them (the board's drawing) -- or None for a board it has not found."""
    from pluggybot.tools.drawing import board_tags
    left, right = board_tags(name)
    a, b = self.places.expected(left), self.places.expected(right)
    if a is None or b is None:
      return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, a[2]

  def board_pose(self, name: str, dist: float) -> tuple[float, float, float] | None:
    """`dist` out of the board's face, facing it, in the map."""
    f = self.board_frame(name)
    if f is None:
      return None
    x, y, facing = f
    return (x + dist * math.cos(facing), y + dist * math.sin(facing),
            dk._wrap(facing + math.pi))

  def _look_at_board(self, name: str) -> list[int]:
    """One look (`detect_board`, which lays the places in); the board's tags
    in it, and their heights above the floor kept, off the camera's mount
    levelled by the IMU and the legs' height."""
    from pluggybot.legs.body import NAV_EYE
    from pluggybot.tools.drawing import board_tags
    self._place_look = (float(self.data.time), self.pose)
    dets = self.detect_board()
    tags = [t for t in board_tags(name) if t in dets]
    if tags:
      level, r_mount, p_mount = dk.camera_mount(self.model, self.data,
                                                self.handle.el(NAV_EYE), self.root)
      h = self._height()
      for t in tags:
        tx, ty, tz = dets[t]["t"]
        p = level @ (r_mount @ np.array([tx, -ty, -tz]) + p_mount)
        self._board_z.setdefault(name, deque(maxlen=HEIGHT_LOOKS)).append(float(p[2]) + h)
    return tags

  def board_height(self, name: str) -> float | None:
    """The board's middle above the floor, off the looks so far."""
    zs = self._board_z.get(name)
    return float(np.median(zs)) if zs else None

  # ---- the drawing ----------------------------------------------------------------

  @contextlib.contextmanager
  def _at_the_board(self):
    """The part of a drawing AT the board, as a swap's at its bay: the rest
    reflex and a restart's save wait it out (`working`)."""
    was, self.working = self.working, True
    try:
      yield
    finally:
      self.working = was

  def draw_routine(self, name: str, board, program, patience: float,
                   stop=None, on_stroke=None, stance: str = "lie") -> Routine:
    """Draw `program` on the board `name` the robot has found, the pen on
    its fork (the module docstring): `board` is the world's (`drawing.Board`,
    for the record of the ink, never the steering). `stop` is the robot's
    own interrupt, asked on the walk and between strokes; `on_stroke(i,
    points, program)` hears each stroke's ink. `stance` "stand" draws on
    its feet, the walking policy holding it: the measurement's premise
    (`scripts/draw_spike.py --stance stand`). Returns its record: `drew`,
    `why` ("drew", "no pen", "not found", "gave up", "lost", "never
    touched", "out of time", "interrupted", "fell", "not drawn"), each walk in's
    stop against the stance (`lineups`), the plotter's stats and its
    calibration (`cal`)."""
    from pluggybot.navigator import DRIVE_STOPPED
    from pluggybot.tools.drawing import PenPlotter
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec: dict = {"board": name}
    self.last_draw = rec

    def left() -> float:
      return until - float(self.data.time)

    def done(why: str, **more) -> dict:
      rec.update(more)
      rec.update(drew=why == "drew", why=why,
                 seconds=round(float(self.data.time) - t0, 1))
      return rec

    if self.carrying != "module_pen":
      return done("no pen")
    if self.board_pose(name, LOOK_M) is None:
      return done("not found")
    yield from self.stand_routine()
    sx, sy, sh = self.board_pose(name, LOOK_M)
    arrived = yield from self.drive_to_routine(
      sx, sy, timeout=min(WALK_PATIENCE_S, max(0.0, left())),
      **({"stop": stop} if stop is not None else {}))
    if stop is not None and (self.last_drive or {}).get("why") == DRIVE_STOPPED:
      return done("interrupted")
    if not arrived and math.hypot(sx - self.pose[0], sy - self.pose[1]) > 0.5:
      rec["walk"] = self.last_drive
      return done("gave up")
    if not (yield from self._face_board_routine(name)):
      return done("lost")
    with self._at_the_board():
      rec["lineups"] = []
      for attempt in range(WALK_IN_TRIES):
        rec["walkIn"] = yield from self._board_walk_in_routine(name)
        yield from self._drive_routine(STOPPED_S, 0.0, 0.0)
        ex, ey, eth = dk.relative(self.pose, self.board_pose(name, STAND_M))
        rec["lineups"].append([round(ex, 3), round(ey, 3), round(math.degrees(eth), 1)])
        if (abs(ey) <= LINEUP_ACROSS and abs(eth) <= LINEUP_YAW) or \
            attempt + 1 == WALK_IN_TRIES or left() <= 0.0:
          break
        # ...not lined up: back to the look point, and in again
        yield from self._back_out_by_routine(LOOK_M - STAND_M, WALK_IN_S, rk.APPROACH_V,
                                             dk.BACK_OUT_SETTLE_S)
        if not (yield from self._face_board_routine(name)):
          break
      if stance == "lie":
        yield from self.rest_routine()
      est = self._board_estimate(name)
      rec["est"] = [round(v, 4) for v in (est.x_face, est.y_mid, est.z_mid)]
      plotter = PenPlotter(self, board, on_stroke=on_stroke)
      plotter.should_stop = stop
      self.last_plotter = plotter
      cal = yield from plotter.calibrate_routine(est)
      rec["cal"] = cal
      used: dict = {}
      if not cal.get("ok"):
        why = "never touched"
      elif left() <= 0.0:
        why = "out of time"
      else:
        used = yield from plotter.draw_program_routine(program)
        why = ("fell" if used.get("stopped") == "fell"
               else "drew" if used.get("drew")
               else "interrupted" if used.get("stopped") == "interrupted" else "not drawn")
      yield from self._off_the_board_routine(plotter)
      yield from self.stand_routine()
    yield from self._back_out_by_routine(BACK_OUT_M, BACK_OUT_S, rk.APPROACH_V,
                                         dk.BACK_OUT_SETTLE_S)
    return done(why, **used)

  def _face_board_routine(self, name: str) -> Routine:
    """Face the board's MIDDLE and look; then turn by `SEARCH_DEG` either
    way, a look after each, until both its tags are in one look: True if
    they were. ⚠ THE MIDDLE, NOT THE FACING: a board found by one tag faces
    where that tag was seen from, up to 6 deg out, and squared to that from
    0.1 m off its axis one tag was past the camera's edge."""
    from pluggybot.tools.drawing import board_tags
    want = set(board_tags(name))
    for turn in (0.0, *SEARCH_DEG):
      if turn:
        yield from self._turn_by_routine(turn)
      else:
        cx, cy, _ = self.board_frame(name)
        x, y, _ = self.pose
        yield from self.face_routine(math.atan2(cy - y, cx - x))
      if want <= set(self._look_at_board(name)):
        return True
    return False

  def _board_estimate(self, name: str):
    """The board as the arm needs it, in the torso frame: off the places
    (the map, through the belief) and the tags' height, the torso's own
    height off the legs (lying, the belly's)."""
    from pluggybot.tools.drawing import BOARD_PROUD_M, BoardEstimate
    cx, cy, facing = self.board_frame(name)
    ex, ey, _ = dk.relative((cx, cy, 0.0), self.pose)
    return BoardEstimate(x_face=ex - BOARD_PROUD_M, y_mid=ey,
                         z_mid=(self.board_height(name) or 0.0) - self._height())

  def _board_walk_in_routine(self, name: str) -> Routine:
    """From the look point to `STAND_M` out of the face, steered by the
    board's tags while they are in view (`rack.walk_in_twist`): "stopped"
    or "budget"."""
    from pluggybot.legs.policy import Twist
    t0 = last = float(self.data.time)
    while self.data.time - t0 < WALK_IN_S:
      pose = self.board_pose(name, STAND_M)
      tw = rk.walk_in_twist(*dk.relative(self.pose, pose))
      if tw == Twist():
        return "stopped"
      yield from self._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
      if self.data.time - last >= LOOK_EVERY_S:
        self._look_at_board(name)
        last = float(self.data.time)
    return "budget"

  def _off_the_board_routine(self, plotter) -> Routine:
    """The pen off the board and the tool back to its carry pose: the
    carriage to its middle, the fork straight back, then up over the
    nose. Nothing, after a fall: it folded the arm and let go of the pen."""
    if plotter.fell():
      return
    x, z = plotter._goal_vertex()
    yield from plotter.move_routine(x - CLEAR_BACK_M, z, carriage=0.0)
    yield from plotter.move_routine(*am.CARRY)
    self.arm.aim(*am.CARRY_Q)
    yield from self._drive_routine(0.5, 0.0, 0.0)
