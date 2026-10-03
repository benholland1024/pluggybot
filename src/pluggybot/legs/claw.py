"""Taking a cube and setting one down, on legs (issue #407): the walk to a
cube the robot has found, the stance, and the claw at the arm's reach
(`tools/claw.py`'s hand). A mixin: `QuadMission` is the rest of the body.

FINDING A CUBE. A cube is 26 mm, its tag 21 mm: the nose camera, level at
0.36 m, sees no floor nearer than 1.1 m and decodes such a tag no further
than 0.8 m, so a cube is the D435 colour imager's (`color_eye`, pitched 30
deg down): standing it reads one 0.5-0.9 m ahead, lying 0.38-0.68.
Where to look, best first: from where the robot stands; where it saw that
cube last (`cubes`, its memory, laid in its map as the places are); and in
front of its area's tags -- the tower's blocks and the bench's masses are
set out on the floor in front of a pair of tags (`home/places.json` says
so in words, and `cube_areas` which), so the search stops along them
`SEARCH_OUT_M` out, facing them, and looks either way at each stop.

THE STANCE IS LYING, as the pen's (`legs/draw.py`): standing, the walking
policy never quite stops (the rack's `SETTLE_DRIFT`), and at the claw's
0.55 m that is a centimetre. The walk in is steered by the cube's tag
(`rack.walk_in_twist`, re-read every `LOOK_EVERY_S`) to `STAND_AT_M` off
it, then the body lies down -- `dock.LIE_SHIFT_M` back -- and looks again:
inside the claw's reach (`claw.REACH_X`, `REACH_Y`) the hand works; else it
stands, backs out and walks in again, `TRIES` times.

WHAT IT DOES AT THE CUBE is `working` (a restart's save waits it out, as a
swap's at its bay), and every cube a look decodes is remembered where the
belief puts it -- kept with the map across a restart, forgotten with it.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.tick import Routine

#: The colour imager's frame: a D435 colour mode (its 1920 x 1080 decodes no
#: further on a 21 mm tag than the window the stance needs).
COLOR_W, COLOR_H = 1280, 720
#: Lying, the cube this far ahead of the torso's centre: the middle of the
#: claw's reach (`claw.REACH_X`)...
LIE_AT_M = 0.54
#: ...so the walk in stops this far off it: lying moves the body back.
STAND_AT_M = LIE_AT_M - dk.LIE_SHIFT_M
#: The walk in starts from this far off the cube, facing it: inside the
#: standing window (a 21 mm tag decodes 0.5-0.9 m ahead), and long enough
#: for the steering to line the body up (the rack's walk in is a metre).
LOOK_M = 0.85
#: The walk in: its budget, its looks, its tries; the stand before lying
#: down; the walk back out of a stance that missed, s / m.
WALK_IN_S = 20.0
LOOK_EVERY_S = 0.4
TRIES = 3
STOPPED_S = 0.8
BACK_OUT_M, BACK_OUT_S = 0.45, 6.0
#: The search in front of a cube's area: look points this far out of its
#: tags' face and this far along it from their middle, facing them, a look
#: and a look either way (deg). MEASURED: from 1.2-1.5 m out a row set out
#: 0.7 m in front of the tags decodes whole, with the tags' places true; the
#: bench's, first seen from the street, were remembered 0.2 m off and 6 deg
#: askew, and from one point on their middle cube 24 decoded at 0.78 m and
#: cube 23 stood 0.96 m off, past the imager's 0.9.
SEARCH_OUT_M = 1.4
SEARCH_SIDE_M = (0.0, 0.35, -0.35)
SWEEP_DEG = (30.0, -60.0)
#: A walk to a stance gives up after this long, s -- a look point's walk
#: has the verb's whole patience: from the rack to the bench's is the
#: street (~100 s), and cut at 90 the search looked from the wrong side of it.
WALK_PATIENCE_S = 90.0
#: `put` sets a cube down `LIE_AT_M` ahead, or this far to either side where
#: a cube a look finds is nearer than `PUT_CLEAR_M` to the spot, m.
PUT_SIDE_M = 0.040
PUT_CLEAR_M = 0.045


def cube_areas() -> dict[int, tuple[int, ...]]:
  """Which area's tags each cube is set out in front of: the tower's
  blocks the workshop corner's pair, the bench's masses the bench's."""
  from pluggybot.rack.tags import (BENCH_TAG_IDS, BLOCK_TAG_IDS, MASS_TAG_IDS,
                                   TOWER_TAG_IDS)
  return {**{t: TOWER_TAG_IDS for t in BLOCK_TAG_IDS},
          **{t: BENCH_TAG_IDS for t in MASS_TAG_IDS}}


class CubeWork:
  """The cubes' routines (the module docstring), on `QuadMission`."""

  def _init_claw(self, model) -> None:
    from pluggybot.tools.claw import cube_bodies
    self._color = None
    #: Each cube it has seen, by tag, in its map: (x, y, layer, when).
    self.cubes: dict[int, tuple[float, float, int, float]] = {}
    #: The cubes this world has (its bodies' names, by tag).
    self.cube_tags = tuple(t for t, b in cube_bodies().items()
                           if _has_body(model, b))
    #: The last pick's or place's record.
    self.last_cube: dict | None = None

  # ---- looking ------------------------------------------------------------------

  def color_detector(self):
    if self._color is None:
      from pluggybot.rack.tags import BLOCK_TAG_SIZE, TagDetector
      self._color = TagDetector(self.model, self.handle.el("color_eye"),
                                width=COLOR_W, height=COLOR_H, tag_size=BLOCK_TAG_SIZE)
    return self._color

  def floor_below(self) -> float:
    """The torso's centre over the floor under it: lying, its belly's depth;
    standing, its legs' (`_height`, the encoders)."""
    from pluggybot.legs.body import LYING
    from pluggybot.legs.model import CHOSEN
    return CHOSEN.belly_depth if self.posture == LYING else self._height()

  def look_cubes(self, ignore=(), held=()) -> dict:
    """One decode of the colour imager: every cube in it, in the HEADING
    frame (`tools.claw.Cube`), each remembered in the map where the belief
    puts it -- but the one in the jaws (`held`), which is no cube's place."""
    from pluggybot.tools.claw import cubes_seen
    det = self.color_detector()
    level, r_mount, p_mount = dk.camera_mount(self.model, self.data,
                                              self.handle.el("color_eye"), self.root)
    seen = cubes_seen(det.detect_all(self.data), level, r_mount, p_mount,
                      det.camera_params, self.floor_below(), ignore=ignore, held=held)
    x, y, th = self.pose
    c, s = math.cos(th), math.sin(th)
    t = float(self.data.time)
    for tag, cube in seen.items():
      if tag not in held:
        self.cubes[tag] = (x + c * cube.x - s * cube.y, y + s * cube.x + c * cube.y,
                           cube.layer, t)
    return seen

  def kept_cubes(self) -> dict:
    """Where it saw each cube, for a restart (#345): a pick walks there."""
    return {str(t): [float(v) for v in c] for t, c in self.cubes.items()}

  def restore_cubes(self, kept: dict | None) -> None:
    self.cubes = {int(t): (float(c[0]), float(c[1]), int(c[2]), float(c[3]))
                  for t, c in (kept or {}).items()}

  def forget_cubes(self) -> None:
    self.cubes = {}

  # ---- where to look ---------------------------------------------------------------

  def _area_front(self, tag: int) -> tuple[float, float, float] | None:
    """The middle of a cube's area's tags and the direction out of their
    face, in the map -- or None where it has found none of them."""
    tags = cube_areas().get(int(tag), ())
    found = [self.places.expected(t) for t in tags]
    found = [p for p in found if p is not None]
    if not found:
      return None
    x = sum(p[0] for p in found) / len(found)
    y = sum(p[1] for p in found) / len(found)
    return x, y, found[0][2]

  def _look_points(self, tag: int) -> list[tuple[float, float, float]]:
    """Where to stand to look for a cube, best first: `LOOK_M` off where it
    was last seen, on the line from here; then along its area's tags."""
    out = []
    px, py, _ = self.pose
    if tag in self.cubes:
      cx, cy = self.cubes[tag][:2]
      d = math.hypot(cx - px, cy - py)
      ux, uy = ((cx - px) / d, (cy - py) / d) if d > 1e-6 else (1.0, 0.0)
      out.append((cx - LOOK_M * ux, cy - LOOK_M * uy, math.atan2(uy, ux)))
    front = self._area_front(tag)
    if front is not None:
      fx, fy, facing = front
      c, s = math.cos(facing), math.sin(facing)
      for side in SEARCH_SIDE_M:
        out.append((fx + SEARCH_OUT_M * c - side * s, fy + SEARCH_OUT_M * s + side * c,
                    dk._wrap(facing + math.pi)))
    return out

  def _sight_routine(self, tag: int, ignore, until: float, stop) -> Routine:
    """Look for cube `tag` (the module docstring): True once a look has it.
    ⚠ EACH LOOK POINT IS PLANNED AS IT IS WALKED TO, off the tags as they
    are remembered then: they are remembered better as the robot nears them
    (`SEARCH_OUT_M`'s bench, planned once from the street)."""
    if tag in self.look_cubes(ignore):
      return True
    k = 0
    while k < len(points := self._look_points(tag)):
      x, y, heading = points[k]
      k += 1
      left = until - float(self.data.time)
      if left <= 0.0 or (stop is not None and stop()):
        return False
      kw = {"stop": stop} if stop is not None else {}
      yield from self.drive_to_routine(x, y, timeout=left, **kw)
      yield from self.face_routine(heading)
      if tag in self.look_cubes(ignore):
        return True
      for turn in SWEEP_DEG:
        yield from self._turn_by_routine(turn)
        if tag in self.look_cubes(ignore):
          return True
    return False

  # ---- the approach -----------------------------------------------------------------

  @contextlib.contextmanager
  def _at_the_cube(self):
    """The part of a pick or a place AT the cube, as a swap's at its bay:
    the rest reflex and a restart's save wait it out (`working`)."""
    was, self.working = self.working, True
    try:
      yield
    finally:
      self.working = was

  def _stance(self, tag: int, toward: tuple[float, float]) -> tuple[float, float, float]:
    """`STAND_AT_M` off the cube on the line from `toward`, facing it."""
    cx, cy = self.cubes[tag][:2]
    d = math.hypot(cx - toward[0], cy - toward[1])
    ux, uy = ((cx - toward[0]) / d, (cy - toward[1]) / d) if d > 1e-6 else (1.0, 0.0)
    return cx - STAND_AT_M * ux, cy - STAND_AT_M * uy, math.atan2(uy, ux)

  def _cube_walk_in_routine(self, tag: int, ignore, line: tuple[float, float]) -> Routine:
    """To `STAND_AT_M` off the cube along `line` (the approach's direction,
    fixed at its start), steered by the cube's tag while it is in view:
    "stopped" or "budget"."""
    from pluggybot.legs.policy import Twist
    ux, uy = line
    t0 = last = float(self.data.time)
    while self.data.time - t0 < WALK_IN_S:
      cx, cy = self.cubes[tag][:2]
      pose = (cx - STAND_AT_M * ux, cy - STAND_AT_M * uy, math.atan2(uy, ux))
      tw = rk.walk_in_twist(*dk.relative(self.pose, pose))
      if tw == Twist():
        return "stopped"
      yield from self._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
      if self.data.time - last >= LOOK_EVERY_S:
        self.look_cubes(ignore)
        last = float(self.data.time)
    return "budget"

  def _approach_routine(self, tag: int, rec: dict, ignore, until: float, stop) -> Routine:
    """Find cube `tag`, walk in to it and lie down where the claw reaches
    it: the cube as the lying look saw it (`tools.claw.Cube`, heading
    frame), or None with `rec["why"]`."""
    from pluggybot.navigator import DRIVE_STOPPED
    from pluggybot.tools.claw import REACH_X, REACH_Y
    yield from self.stand_routine()
    if not (yield from self._sight_routine(tag, ignore, until, stop)):
      rec["why"] = ("interrupted" if stop is not None and stop()
                    else "out of time" if float(self.data.time) >= until else "not found")
      return None
    rec["tries"] = []
    for _ in range(TRIES):
      att = {}
      rec["tries"].append(att)
      if float(self.data.time) >= until:
        rec["why"] = "out of time"
        return None
      px, py, _ = self.pose
      sx, sy, heading = self._stance(tag, (px, py))
      ux, uy = math.cos(heading), math.sin(heading)
      lx, ly = sx - (LOOK_M - STAND_AT_M) * ux, sy - (LOOK_M - STAND_AT_M) * uy
      if math.hypot(lx - px, ly - py) > 0.15:
        kw = {"stop": stop} if stop is not None else {}
        yield from self.drive_to_routine(
          lx, ly, timeout=min(WALK_PATIENCE_S, max(1.0, until - float(self.data.time))), **kw)
        if stop is not None and (self.last_drive or {}).get("why") == DRIVE_STOPPED:
          rec["why"] = "interrupted"
          return None
      cx, cy = self.cubes[tag][:2]
      yield from self.face_routine(math.atan2(cy - self.pose[1], cx - self.pose[0]))
      self.look_cubes(ignore)
      with self._at_the_cube():
        att["walkIn"] = yield from self._cube_walk_in_routine(tag, ignore, (ux, uy))
        yield from self._drive_routine(STOPPED_S, 0.0, 0.0)
        if self.carrying != "module_claw":
          rec["why"] = "fell"
          return None
        if stop is not None and stop():
          rec["why"] = "interrupted"
          return None
        yield from self.rest_routine()
        cube = self.look_cubes(ignore).get(tag)
      if cube is not None:
        att["at"] = [round(cube.x, 3), round(cube.y, 3), cube.layer]
        if REACH_X[0] <= cube.x <= REACH_X[1] and abs(cube.y) <= REACH_Y:
          return cube
        att["why"] = "out of reach"
      else:
        att["why"] = "not seen lying"
      yield from self.stand_routine()
      yield from self._back_out_by_routine(BACK_OUT_M, BACK_OUT_S, rk.APPROACH_V,
                                           dk.BACK_OUT_SETTLE_S)
      self.look_cubes(ignore)
    rec["why"] = "never lined up"
    return None

  # ---- the two verbs ------------------------------------------------------------------

  def _hand(self):
    from pluggybot.tools.claw import ClawHand
    return ClawHand(self)

  def _jaws_middle(self, hand, level) -> np.ndarray:
    """The jaws' middle in the HEADING frame, off the arm's encoders and
    the slide's command: straight under the seated peg."""
    vx, vz = hand.vertex()
    seat = level @ np.array([vx, 0.0, vz + hand.rise])
    slide = float(self.data.ctrl[hand.slide])
    return seat + level @ np.array([0.0, -slide, 0.0]) - np.array([0.0, 0.0, rk.CLAW_JAW_DROP])

  def pick_cube_routine(self, tag: int, patience: float, stop=None) -> Routine:
    """Take cube `tag` in the claw (the module docstring): its record --
    `picked` (the jaws hold it, off the world) and `why` ("picked", "no
    claw", "holding", "not found", "never lined up", "missed", "fell",
    "out of time", "interrupted")."""
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec: dict = {"op": "pick", "tag": int(tag)}
    self.last_cube = rec

    def done(why: str) -> dict:
      rec.update(picked=why == "picked", why=why,
                 seconds=round(float(self.data.time) - t0, 1))
      return rec

    if self.carrying != "module_claw":
      return done("no claw")
    hand = self._hand()
    if hand.held() is not None:
      rec["holding"] = hand.held()
      return done("holding")
    cube = yield from self._approach_routine(int(tag), rec, (), until, stop)
    if cube is None:
      yield from self._leave_routine(hand)
      return done(rec.get("why", "not found"))
    with self._at_the_cube():
      hand = self._hand()
      level, _, _ = dk.camera_mount(self.model, self.data, self.handle.el("color_eye"),
                                    self.root)
      got = yield from hand.pick_routine(cube, level)
      rec["held"] = hand.held()
      why = ("fell" if hand.fell() else "picked" if got and rec["held"] else "missed")
      yield from self._leave_routine(hand)
    return done(why)

  def place_cube_routine(self, tag: int, patience: float, stop=None) -> Routine:
    """Set the cube in the claw down on cube `tag`: its record -- `placed`
    (let go, and resting on it: one edge above, its middle within half an
    edge across, `challenge.stack`'s "rests on", off the world) and `why`
    ("placed", "no claw", "nothing held", "not found", "never lined up",
    "out of reach", "not on it", "fell", "out of time", "interrupted")."""
    from pluggybot.tools.claw import CUBE_HALF, GRIP_Z
    t0 = float(self.data.time)
    until = t0 + float(patience)
    rec: dict = {"op": "place", "tag": int(tag)}
    self.last_cube = rec

    def done(why: str) -> dict:
      rec.update(placed=why == "placed", why=why,
                 seconds=round(float(self.data.time) - t0, 1))
      return rec

    if self.carrying != "module_claw":
      return done("no claw")
    hand = self._hand()
    held = hand.held_tag()
    if held is None:
      return done("nothing held")
    if held == int(tag):
      return done("that cube is in the claw")
    rec["held"] = held
    target = yield from self._approach_routine(int(tag), rec, (held,), until, stop)
    if target is None:
      yield from self._leave_routine(hand)
      return done(rec.get("why", "not found"))
    with self._at_the_cube():
      hand = self._hand()
      level, _, _ = dk.camera_mount(self.model, self.data, self.handle.el("color_eye"),
                                    self.root)
      # ...shown to the imager first: where the cube hangs in the jaws is
      # measured, not assumed -- it creeps down the pads (`legs.rack`)
      if not (yield from hand.show_routine(target, level)):
        yield from self._leave_routine(hand)
        return done("fell" if hand.fell() else "out of reach")
      yield from self._drive_routine(0.3, 0.0, 0.0)
      seen = self.look_cubes((), held=(held,))
      jaws = self._jaws_middle(hand, level)
      mine = seen.get(held)
      hang = ((mine.x - jaws[0], mine.y - jaws[1], mine.z - jaws[2]) if mine is not None
              else (0.0, 0.0, CUBE_HALF - GRIP_Z))
      rec["hang"] = [round(v * 1000, 1) for v in hang]
      rec["hangSeen"] = mine is not None
      target = seen.get(int(tag)) or target
      let_go = yield from hand.place_routine(target, level, hang)
      rests = self._rests_on(held, int(tag))
      why = ("fell" if hand.fell() else "placed" if let_go and rests
             else "not on it")
      yield from self._leave_routine(hand)
    return done(why)

  def put_cube_routine(self, patience: float, stop=None) -> Routine:
    """Set the cube in the claw down on the floor in front, where it lies:
    its record -- `put` (let go, and resting on the floor, off the world)
    and `why` ("put", "no claw", "nothing held", "fell", "not down")."""
    from pluggybot.tools.claw import CUBE_HALF, Cube
    t0 = float(self.data.time)
    rec: dict = {"op": "put"}
    self.last_cube = rec

    def done(why: str) -> dict:
      rec.update(put=why == "put", why=why, seconds=round(float(self.data.time) - t0, 1))
      return rec

    if self.carrying != "module_claw":
      return done("no claw")
    hand = self._hand()
    held = hand.held_tag()
    if held is None:
      return done("nothing held")
    rec["held"] = held
    with self._at_the_cube():
      yield from self.rest_routine()
      hand = self._hand()
      level, _, _ = dk.camera_mount(self.model, self.data, self.handle.el("color_eye"),
                                    self.root)
      # ...on the floor in front, clear of any cube a look finds there: the
      # slide's stroke either way, else where it lies
      seen = self.look_cubes((held,))
      h = self.floor_below()
      spot = None
      for y in (0.0, PUT_SIDE_M, -PUT_SIDE_M):
        if all(math.hypot(c.x - LIE_AT_M, c.y - y) > PUT_CLEAR_M for c in seen.values()):
          spot = y
          break
      if spot is None:
        yield from self._leave_routine(hand)
        return done("no room")
      # the floor as a cube's "layer below": a phantom one an edge under it
      floor = Cube(0, LIE_AT_M, spot, -h - CUBE_HALF, -1, "floor")
      if not (yield from hand.show_routine(floor, level)):
        yield from self._leave_routine(hand)
        return done("fell" if hand.fell() else "out of reach")
      yield from self._drive_routine(0.3, 0.0, 0.0)
      jaws = self._jaws_middle(hand, level)
      mine = self.look_cubes((), held=(held,)).get(held)
      from pluggybot.tools.claw import GRIP_Z
      hang = ((mine.x - jaws[0], mine.y - jaws[1], mine.z - jaws[2]) if mine is not None
              else (0.0, 0.0, CUBE_HALF - GRIP_Z))
      let_go = yield from hand.place_routine(floor, level, hang)
      z = float(self.data.xpos[self.model.body(_cube_body(held)).id][2])
      why = ("fell" if hand.fell() else "put" if let_go and abs(z - CUBE_HALF) < 0.004
             else "not down")
      yield from self._leave_routine(hand)
    return done(why)

  def _rests_on(self, upper: int, lower: int) -> bool:
    """Does cube `upper` rest on cube `lower` (`challenge.stack`'s rule),
    off the world -- a step's verdict, never a belief."""
    from pluggybot.challenge.stack import PITCH_M, PITCH_TOL_M, REST_OFFSET_M
    from pluggybot.tools.claw import cube_bodies
    bodies = cube_bodies()
    a = self.data.xpos[self.model.body(bodies[upper]).id]
    b = self.data.xpos[self.model.body(bodies[lower]).id]
    return (abs(float(a[2] - b[2]) - PITCH_M) <= PITCH_TOL_M
            and math.hypot(float(a[0] - b[0]), float(a[1] - b[1])) <= REST_OFFSET_M)

  def _leave_routine(self, hand) -> Routine:
    """Up from the cube: the slide to its middle, the fork to the carry
    pose, then onto its feet -- nothing, after a fall."""
    from pluggybot.legs import arm as am
    if hand.fell():
      return
    yield from hand.centre_routine()
    yield from hand.move_routine(*am.CARRY, speed=0.15)
    if hand.fell():
      return
    self.arm.aim(*am.CARRY_Q)
    yield from self._drive_routine(0.3, 0.0, 0.0)
    yield from self.stand_routine()

  def claw_routine(self, closed: bool) -> Routine:
    """The jaws shut or open where the claw is: True if anything is held
    once shut, or nothing once open."""
    if self.carrying != "module_claw":
      return False
    hand = self._hand()
    yield from hand.jaws_routine(closed=closed, settle=0.6 if closed else 0.4)
    return (hand.held() is not None) == closed


def _cube_body(tag: int) -> str:
  from pluggybot.tools.claw import cube_bodies
  return cube_bodies()[int(tag)]


def _has_body(model, name: str) -> bool:
  import mujoco
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
