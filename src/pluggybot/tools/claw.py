"""The claw on the arm (issue #407): a cube found by its tags, taken off the
floor, carried, and set down on another -- the hand of it. The body's half,
the walk to a cube and the stance, is `legs/claw.py`.

THE HAND. The arm moves the claw in the robot's forward-and-up plane, the
module's own slide moves it across (`legs.rack.claw_face`, the pen's
L12-100 again: the body cannot sidestep a few millimetres), and one servo
closes the two jaws. A module hangs plumb from its peg, so the jaws' middle
is `CLAW_JAW_DROP` straight under the seated peg's axis, the slide's offset
across; the arm aims the fork's V vertex through its own kinematics
(`legs.arm`), never a measurement of the jaws.

THE EYE is the D435's colour imager (`legs.model`: `color_eye`, pitched 30
deg down with the depth camera). ⚠ A CUBE CARRIES ITS TAG ON EVERY FACE
(`challenge.stack.block_xml`): one render decodes its top and its sides
alike, so each decode is classed by its normal and the one seen most nearly
square-on is read. Where it is comes off the tag's centre pixel cut at the
face's known height (`at_height`: the floor under the robot, through its
legs or its belly, and a layer is a cube's edge), not off PnP's range --
the rover's lesson (`rover-final`, #264): a small tag's range is the noisy
part. MEASURED from lying 0.58 m off: both faces put the cube within 0.1 mm.

THE GRIP is judged off the world: both pads touching one body that is
neither the robot's nor the claw's and can move (`held`), never off the
jaws' command -- lowered onto the floor the pads touch it, and a graze on
the way past reads as one pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pluggybot.challenge.stack import BLOCK_HALF
from pluggybot.legs import rack as rk
from pluggybot.tick import Routine

#: The cubes the claw knows by their tags (the tower's blocks and the
#: bench's masses, `rack.tags`), and their body names.
CUBE_HALF = BLOCK_HALF
#: The jaws close with the pads' bottoms this far over the surface the cube
#: stands on: the rover's 2 mm -- at 0 they pressed the floor and the module
#: rode up its fork (`rover-final`).
FLOOR_CLEARANCE = 0.002
#: ...so the jaws' middle over a cube's bottom as they close on it.
GRIP_Z = rk.CLAW_PAD_HALF[2] + FLOOR_CLEARANCE
#: Over a cube before going down onto it, and over a stack before setting
#: one down: the pads' bottoms (a held cube's bottom) this far over its top, m.
HOVER_M = 0.030
#: A cube set down on another is let go this far over it, m: a cube let go
#: under the other's top presses it down and shoves it; one dropped from
#: higher turns as it falls.
PLACE_GAP = 0.003
#: Before setting a cube down the claw shows it to the colour imager: this
#: far nearer the robot than the target, its middle this high over the
#: floor -- from lying the imager sees a cube's whole tag only under ~40 mm
#: at that reach, and a held cube creeps down the pads (`legs.rack`).
SHOW_BACK_M = 0.06
SHOW_Z = 0.030
#: The fork's speed along its straight lines at a cube, m/s, and the jaws'
#: and the slide's setpoint speeds (m/s; the slide's is the L12's 25 mm/s).
WORK_V = 0.08
JAW_SPEED = 0.03
SLIDE_SPEED = rk.PEN_SPEED
#: The jaws close for this long before the grip is judged, s.
GRIP_SETTLE_S = 0.6
#: Held is both pads within this of one body, m -- a held cube's are a few
#: microns in -- and only a body whose middle is this near the jaws' is
#: looked at.
HELD_GAP_M = 0.0005
HELD_NEAR_M = 0.06
#: A fork move is given this long past its line to arrive, s, within this.
ARRIVE_S, ARRIVE_TOL = 1.5, 0.003
#: The claw's working place in front of the robot, the torso frame from
#: lying: a cube within this much of the jaws' middle across (the slide's
#: stroke and a pad's room) and this band ahead -- the colour imager sees a
#: cube's whole tag from lying between 0.38 and 0.68 m ahead, and the arm
#: reaches the floor from 0.35 to 0.82 (SimNotes, "The claw on legs").
REACH_X = (0.40, 0.68)
REACH_Y = rk.CLAW_TRAVEL - 0.006
#: The cubes, by tag (the tower's blocks, the bench's masses).
def cube_bodies() -> dict[int, str]:
  from pluggybot.challenge.bench import MASSES
  from pluggybot.challenge.stack import BLOCKS
  from pluggybot.rack.tags import BLOCK_TAG_IDS, MASS_TAG_IDS
  return {**dict(zip(BLOCK_TAG_IDS, BLOCKS)), **dict(zip(MASS_TAG_IDS, MASSES))}


@dataclass(frozen=True)
class Cube:
  """A cube as one look saw it: its middle in the robot's HEADING frame
  (origin the torso's centre, x ahead, levelled by the IMU, z up), which
  layer of a stack it is, and the face that was read."""

  tag: int
  x: float
  y: float
  z: float
  layer: int
  face: str

  @property
  def top(self) -> float:
    """Its top face's height over its middle, heading frame (+z)."""
    return self.z + CUBE_HALF


def cubes_seen(dets, level, r_mount, p_mount, camera_params, h: float,
               ignore=(), held=()) -> dict[int, Cube]:
  """Every cube in one decode (`TagDetector.detect_all`), by tag: each face
  read off its centre pixel at its known height, the face seen most nearly
  square-on kept. `h` is the torso's centre over the floor; the camera's
  mount and the levelling are `legs.dock.camera_mount`'s; `ignore` is
  tags not to read. ⚠ A cube in the jaws (`held`) stands on no layer: cut
  at one, its ray put it 18 mm long, so its middle is PnP's, from close."""
  f, _, cx, cy = camera_params
  o = level @ p_mount
  bodies = cube_bodies()
  best: dict[int, tuple[float, Cube]] = {}
  for d in dets:
    tag = int(d["id"])
    if tag not in bodies or tag in ignore:
      continue
    nx, ny, nz = d["normal"]
    n = level @ (r_mount @ np.array([nx, -ny, -nz]))       # into the face
    u, v = d["center"]
    r = level @ (r_mount @ np.array([(u - cx) / f, -(v - cy) / f, -1.0]))
    r = r / np.linalg.norm(r)
    tx, ty, tz = d["t"]
    pt = level @ (r_mount @ np.array([tx, -ty, -tz]) + p_mount)
    pz = float(pt[2]) + h
    square = abs(float(n @ r))
    if tag in held:
      c = pt + CUBE_HALF * n
      cube = Cube(tag, float(c[0]), float(c[1]), float(c[2]), 0, "held")
    elif abs(float(n[2])) > 0.7:                            # a top face
      layer = max(0, round(pz / (2 * CUBE_HALF)) - 1)
      at = 2 * CUBE_HALF * (layer + 1) - h
      if abs(r[2]) < 1e-6:
        continue
      p = o + (at - o[2]) / r[2] * r
      cube = Cube(tag, float(p[0]), float(p[1]), at - CUBE_HALF, layer, "top")
    else:                                                   # a side
      layer = max(0, int(pz // (2 * CUBE_HALF)))
      at = CUBE_HALF + 2 * CUBE_HALF * layer - h
      if abs(r[2]) < 1e-6:
        continue
      p = o + (at - o[2]) / r[2] * r
      nh = n[:2] / max(float(np.linalg.norm(n[:2])), 1e-9)
      cube = Cube(tag, float(p[0] + CUBE_HALF * nh[0]), float(p[1] + CUBE_HALF * nh[1]),
                  at, layer, "side")
    if tag not in best or square > best[tag][0]:
      best[tag] = (square, cube)
  return {tag: c for tag, (_, c) in best.items()}


class ClawHand:
  """The arm, the slide and the jaws as one hand (the module docstring).
  `mission` is the quadruped's (`legs.body.QuadMission`)."""

  def __init__(self, mission) -> None:
    self.mission = mission
    self.model, self.data = mission.model, mission.data
    self.arm, self.spec = mission.arm, mission.arm_spec
    m = self.model
    self.slide = m.actuator(rk.CLAW_SLIDE).id
    self.jaws = tuple(m.actuator(j).id for j in rk.CLAW_JAWS)
    self.pads = tuple(m.geom(p).id for p in rk.CLAW_PADS)
    self._pads = frozenset(self.pads)
    self._own = int(m.body_rootid[m.geom_bodyid[self.pads[0]]])
    self.rise = self.spec.fork.seat_rise()
    #: The posture it works in and the body's falls so far: a body that
    #: leaves it, falls or loses the claw has its arm folded by the fall.
    self.posture = mission.posture
    self._falls = mission.falls

  # ---- what the hand senses ------------------------------------------------------

  def fell(self) -> bool:
    m = self.mission
    return (m.posture != self.posture or m.falls != self._falls
            or m.carrying != "module_claw")

  def held(self) -> str | None:
    """The body both pads touch -- within `HELD_GAP_M` of each -- where it
    is free to move and no robot's and no tool's: a cube -- or None.
    ⚠ BY DISTANCE, NOT THE CONTACT LIST: on the stiff pads' microns of
    penetration a held cube's contacts come and go from step to step
    (`legs.rack.CLAW_PAD_SOLREF`), and read at one instant a cube in the
    jaws was "nothing held". ⚠ Only something that can MOVE is held:
    lowered onto the floor, both pads touch it (`rover-final`, #264:
    "already holding floor")."""
    import mujoco
    model, data = self.model, self.data
    grip = data.site_xpos[model.site(rk.CLAW_GRIP).id]
    for name, root in self._loose():
      if float(np.linalg.norm(data.xpos[root] - grip)) > HELD_NEAR_M:
        continue
      gids = [g for g in range(int(model.body_geomadr[root]),
                               int(model.body_geomadr[root] + model.body_geomnum[root]))]
      if all(any(mujoco.mj_geomDistance(model, data, pad, g, 2 * HELD_GAP_M, None) < HELD_GAP_M
                 for g in gids) for pad in self.pads):
        return name
    return None

  def _loose(self) -> list[tuple[str, int]]:
    """The free bodies a claw may hold, by name and id: no robot's, no
    tool's (`module_*`), kept per world."""
    loose = getattr(self.mission, "_loose_bodies", None)
    if loose is None or loose[0] is not self.model:
      from pluggybot.telemetry.protocol import robot_roots
      model = self.model
      robots = set(robot_roots(model))
      out = []
      for b in range(1, model.nbody):
        name = model.body(b).name or ""
        if (int(model.body_rootid[b]) == b and model.body_dofnum[b] > 0
            and name not in robots and not name.startswith("module_")
            and model.body_geomnum[b] > 0):
          out.append((name, b))
      loose = (model, out)
      self.mission._loose_bodies = loose
    return loose[1]

  def held_tag(self) -> int | None:
    """The tag of the cube in the jaws, or None."""
    name = self.held()
    return next((t for t, b in cube_bodies().items() if b == name), None)

  def vertex(self) -> tuple[float, float]:
    """The fork's V vertex in the torso frame, off the arm's encoders."""
    from pluggybot.legs import arm as am
    qs, qf = self.arm.q()
    wx, wz = am.wrist_xz(self.spec, qs, qf - qs)
    return wx + self.spec.fork.vertex_x, wz + self.spec.fork.vertex_z

  # ---- the arm -------------------------------------------------------------------

  def _aim(self, x: float, z: float) -> bool:
    from pluggybot.legs import arm as am
    if self.fell():
      return False
    g = self.arm.goal
    q = am.solve_vertex(self.spec, x, z, near=(float(g[0]), float(g[1] - g[0])))
    if q is None:
      return False
    self.arm.aim(*q)
    return True

  def move_routine(self, x: float, z: float, speed: float = WORK_V) -> Routine:
    """The fork's vertex along a straight line to (x, z), torso frame: False
    if a point on the way is out of reach or the body fell."""
    x0, z0 = self.mission._vertex_goal()
    n = max(1, int(math.hypot(x - x0, z - z0) / speed / self.model.opt.timestep))
    for k in range(1, n + 1):
      if not self._aim(x0 + (x - x0) * k / n, z0 + (z - z0) * k / n):
        return False
      yield from self.mission._twist_routine(0.0, 0.0, 0.0)
    t0 = float(self.data.time)
    while not self.arm.arrived(ARRIVE_TOL) and self.data.time - t0 < ARRIVE_S:
      yield from self.mission._twist_routine(0.0, 0.0, 0.0)
    return not self.fell()

  def vertex_for(self, x: float, y: float, z: float, level) -> tuple[float, float, float]:
    """Where to send the fork's vertex (torso x, z) and the slide (its
    command) to put the jaws' middle at (x, y, z) in the HEADING frame:
    straight under the seated peg (a module hangs plumb), the slide's
    offset across. The tool faces the robot, so the slide's + is the
    robot's -y."""
    seat = level.T @ np.array([x, y, z + rk.CLAW_JAW_DROP])
    return float(seat[0]), float(seat[2] - self.rise), -float(seat[1])

  def to_routine(self, x: float, y: float, z: float, level,
                 speed: float = WORK_V) -> Routine:
    """The jaws' middle to (x, y, z), heading frame: the slide first, then
    the fork. False out of reach, past the slide's stroke, or fallen."""
    from pluggybot.tools import servo
    vx, vz, slide = self.vertex_for(x, y, z, level)
    if abs(slide) > rk.CLAW_TRAVEL + 1e-9:
      return False
    yield from servo.ramp_routine(self.mission, self.slide, slide, SLIDE_SPEED)
    return (yield from self.move_routine(vx, vz, speed))

  def jaws_routine(self, closed: bool, settle: float = 0.3) -> Routine:
    """The jaws shut (on whatever is between them) or open, ramped."""
    from pluggybot.tools import servo
    full = rk.CLAW_JAW_OPEN - rk.CLAW_JAW_CLOSED
    yield from servo.ramp_many_routine(self.mission, self.jaws, 0.0 if closed else full,
                                       JAW_SPEED, settle)

  def centre_routine(self) -> Routine:
    """The slide back to its middle."""
    from pluggybot.tools import servo
    yield from servo.ramp_routine(self.mission, self.slide, 0.0, SLIDE_SPEED)

  # ---- taking a cube, and setting one down ---------------------------------------

  def pick_routine(self, cube: Cube, level) -> Routine:
    """Open, over the cube, down astride it, close, up: True if the jaws
    hold it -- `held`, off the world."""
    yield from self.jaws_routine(closed=False)
    above = cube.z - CUBE_HALF + 2 * CUBE_HALF + HOVER_M + GRIP_Z
    if not (yield from self.to_routine(cube.x, cube.y, above, level)):
      return False
    if not (yield from self.to_routine(cube.x, cube.y, cube.z - CUBE_HALF + GRIP_Z, level,
                                       WORK_V / 2)):
      return False
    yield from self.jaws_routine(closed=True, settle=GRIP_SETTLE_S)
    if self.held() is None:
      yield from self.jaws_routine(closed=False)
      yield from self.to_routine(cube.x, cube.y, above, level)
      return False
    yield from self.to_routine(cube.x, cube.y, above, level)
    return self.held() is not None

  def show_routine(self, target: Cube, level) -> Routine:
    """The cube in the jaws low in front of `target`, where the colour
    imager sees it from lying (`SHOW_BACK_M`, `SHOW_Z`): up first, clear of
    whatever stands there, then across and down. False out of reach."""
    x0, z0 = self.mission._vertex_goal()
    up = target.top + HOVER_M + 2 * CUBE_HALF + GRIP_Z
    seat = level @ np.array([x0, 0.0, z0 + self.rise])
    jaws_z = float(seat[2]) - rk.CLAW_JAW_DROP
    if jaws_z < up:
      _, y_now = self.slide_y()
      if not (yield from self.to_routine(float(seat[0]), y_now, up, level)):
        return False
    floor = target.z - CUBE_HALF - 2 * CUBE_HALF * target.layer
    x = target.x - SHOW_BACK_M
    if not (yield from self.to_routine(x, target.y, up, level)):
      return False
    return (yield from self.to_routine(x, target.y, floor + SHOW_Z + CUBE_HALF - GRIP_Z
                                       + CUBE_HALF, level, WORK_V / 2))

  def slide_y(self) -> tuple[float, float]:
    """The slide's command, and the robot's y it puts the jaws at."""
    c = float(self.data.ctrl[self.slide])
    return c, -c

  def place_routine(self, target: Cube, level, hang: tuple[float, float, float]) -> Routine:
    """The cube in the jaws set down on `target`: up from the show pose,
    over the target, down until the held cube's bottom is `PLACE_GAP` over
    the target's top, open, and up. `hang` is where the held cube's middle
    sits off the jaws' middle (heading frame), as a look measured it. True
    once let go."""
    hx, hy, hz = hang
    jaw_z = target.top + PLACE_GAP + CUBE_HALF - hz
    x, y = target.x - hx, target.y - hy
    over = jaw_z + HOVER_M
    xn, yn = target.x - SHOW_BACK_M - hx, target.y - hy
    if not (yield from self.to_routine(xn, yn, over, level)):
      return False
    if not (yield from self.to_routine(x, y, over, level)):
      return False
    if not (yield from self.to_routine(x, y, jaw_z, level, WORK_V / 3)):
      return False
    yield from self.jaws_routine(closed=False, settle=0.5)
    yield from self.to_routine(x, y, over + 2 * CUBE_HALF, level)
    return self.held() is None
