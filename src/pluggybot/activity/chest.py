"""The drop-handle chest (issue #466): demo 1's mechanism, a toy chest whose
lid is lifted by a drop handle, as an ACTIVITY (docs/ActivityPattern.md). It
owns its geometry, its hidden parameters and its state, and reports what is
sensed of it.

#469's spike chose it (SimNotes, "Opening a box from lying"): a chest 0.22 x
0.30 x 0.14 m of 12 mm board, its lid hinged along the back of its top. The
knob is the 26 mm cube the claw is proven on, hung on an L from its own pin
at the lid's front edge, parallel to the hinge, so the claw's grip does not
turn as the lid does. It opened from every stance at every gain, and swept
at Kp 8 it keeps the claw seated along a hinge guessed up to 4 cm off.

THE HIDDEN PARAMETERS (`Lid`, drawn by `draw`): the board's mass, a hidden
weight in it, the hinge's spring, damping and Coulomb friction, and a
magnetic catch. ⚠ THE TRUTH IS KEPT AS `Task.secret` IS: never in the
robot's context or on the wire, and the flags carry none of it.

WHAT THE SCENE LANGUAGE CANNOT SAY (`imagination.scene`), so the robot's
model of the chest is always an approximation, and `reference_document` is
the closest the language comes:
  - the catch's pull falls with the gap as a magnet's does, where the
    language's catch is one release force;
  - the chest's own contact settings (MuJoCo's defaults, as the spike flew
    it, with the handle's pairs named), where the imagination compiles its
    own (`imagination.compile.CONTACT`);
  - the hinges' armature.
Stiction and a non-linear spring are left out: SimNotes, "The imagination's
first world", says why.

The frame is the CHEST's: origin on the floor under the middle of its front
face (the robot's side), +x into the chest, z up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from pluggybot.activity.base import Activity, Threshold
from pluggybot.challenge.stack import BLOCK_HALF
from pluggybot.rack.tags import CHEST_TAG_IDS

# ---- the geometry (#469's; `scripts/mechanism_spike.py` imports it) --------------

#: The chest: a toy chest's size, its walls and lid 12 mm board.
BOX_D, BOX_W, BOX_H = 0.22, 0.30, 0.14
WALL_T = LID_T = 0.012
#: The knob the claw grips is the cube the claw is proven on (26 mm), on a
#: stem out of the handle: the stem leaves the jaws out of their back, under
#: the crossbar that joins them.
KNOB = BLOCK_HALF
STEM, STEM_HALF_W = 0.030, 0.004
#: The drop handle: its pin this far in front of the lid's front edge, at
#: the lid's mid-thickness, and its arm down to the stem. The claw holds the
#: handle as it hangs while the lid turns under it, so at 69 deg its 40 mm
#: arm is nearly along the lid: 20 mm out, it lay in the lid's front edge,
#: unseen -- a body and its parent never collide, so the handle's pairs
#: with the board are named.
PIN_AHEAD, HANDLE_DROP = 0.050, 0.040
PIN_Z = LID_T / 2
#: The handle's parts, kg: its arm, its stem and the knob; and the bracket
#: the pin hangs from, out of the lid's front edge.
HANDLE_KG = (0.010, 0.005, 0.020)
BRACKET_KG = 0.005
#: The hidden weight: a lump this size inside the board, which nothing
#: touches.
LUMP_HALF = (0.010, 0.010, 0.004)
#: The lid's travel off its stop, rad: shut on the walls (the stop is the
#: hinge's lower limit) to 109 deg.
HINGE_RANGE = (0.0, 1.9)
#: What a joint carries that the scene language has no field for: the
#: hinges' armature, kg*m^2 (a drawn lid's own inertia about its hinge is
#: 0.008-0.015). The pin's damping and friction are a joint's, which it has.
HINGE_ARMATURE = 0.0005
PIN_ARMATURE = 0.0001
PIN_DAMPING = 0.002
PIN_FRICTION = 0.002

#: The hinge's line (y) in the chest's frame, the pin in the lid's, and the
#: knob's middle off the pin with the handle hanging plumb.
HINGE = np.array([BOX_D, 0.0, BOX_H])
PIN = np.array([-BOX_D - PIN_AHEAD, 0.0, PIN_Z])
KNOB_OFF = np.array([-STEM - KNOB, 0.0, -HANDLE_DROP])
#: The pin's distance off the hinge's line, m: the catch's lever.
PIN_R = float(np.hypot(PIN[0], PIN[2]))

#: The knob's tag, on every face as the cubes' are, so the claw's walk-in
#: can steer by it.
KNOB_TAG_ID = CHEST_TAG_IDS[0]

WOOD = 'rgba="0.55 0.42 0.30 1"'
LID_WOOD = 'rgba="0.70 0.55 0.38 1"'
METAL = 'rgba="0.20 0.45 0.75 1"'

# ---- the hidden parameters ------------------------------------------------------


@dataclass(frozen=True)
class Lid:
  """One chest's hidden parameters, about the hinge (kg, m, N*m, rad, N)."""

  #: The board, and a hidden weight in it this fraction of the way from the
  #: hinge to the front edge, this high over the hinge's line.
  lid_kg: float = 0.30
  lump_kg: float = 0.0
  lump_at: float = 0.5
  lump_z: float = LID_T / 2
  #: The hinge: a spring (and where it is slack), damping, Coulomb friction.
  stiffness: float = 0.0
  springref: float = 0.0
  damping: float = 0.04
  friction: float = 0.04
  #: A magnetic catch holding it shut, N at the knob (0: none).
  catch_n: float = 0.0


#: The ranges `draw` draws from (#469's): the board, the hidden weight and
#: where it sits, the spring, damping and friction. In this order: a draw
#: is the spike's set-out of the same number.
RANGES = {"lid_kg": (0.15, 0.45), "lump_kg": (0.0, 0.15), "lump_at": (0.2, 0.9),
          "stiffness": (0.0, 0.15), "damping": (0.0, 0.20), "friction": (0.0, 0.12)}
#: The catch, N at the knob: 4 is the most #469 flew, and the grip held its
#: 5.9 N pull at release with 0.5 mm of slip.
CATCH_RANGE = (1.0, 4.0)
#: Where a spring is slack, rad: past the range, so it eases the lift.
SPRING_SLACK = 2.5
#: ⚠ A DROP HANDLE ONLY PULLS: the claw hangs on its peg and swings off any
#: push away from the robot, so a lid must shut harder than its spring,
#: friction and damping hold it open -- by this much, N*m at the hinge (0.2 N
#: at the knob), at every angle up to `SWEPT_TO` swept down at `SWEPT_RATE`
#: (#469's sweeps). A lid whose spring beat its weight near the top by 0.15
#: N*m pushed the handle back at the claw: it swung 9-14 deg on its peg,
#: where no shut lid swung it past 8.
CLOSING_MARGIN_NM = 0.05
SWEPT_TO = 1.2
SWEPT_RATE = 0.45
#: `draw(k)`'s seed is this plus k: the spike's, so a draw is its set-out.
DRAW_SEED = 4690
G = 9.81


def closing_torque(lid: Lid, th, rate: float) -> np.ndarray:
  """What shuts the lid at `th` (rad) going down at `rate` (rad/s), N*m: its
  gravity -- the board, the hidden weight and the bracket, out along the lid
  and up off the hinge's line (the height alone is up to 0.03 N*m at 69
  deg) -- less its spring, friction and damping."""
  th = np.asarray(th, dtype=float)
  out = (lid.lid_kg * BOX_D / 2 + lid.lump_kg * BOX_D * lid.lump_at
         + BRACKET_KG * (BOX_D + PIN_AHEAD / 2 - 0.003))
  up = (lid.lid_kg + BRACKET_KG) * LID_T / 2 + lid.lump_kg * lid.lump_z
  return (G * (out * np.cos(th) - up * np.sin(th)) - lid.stiffness * (lid.springref - th)
          - lid.friction - lid.damping * rate)


def draw(seed: int) -> Lid:
  """Chest `seed`'s hidden parameters (seeded, so demo 2 can draw held-out
  ones later): drawn from `RANGES` together until the lid shuts harder than
  `CLOSING_MARGIN_NM` all the way up -- every other one with no spring --
  and then its catch. The six are the spike's set-out `seed`."""
  if seed < 0:
    raise ValueError(f"a chest's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng(DRAW_SEED + seed)
  th = np.linspace(0.0, SWEPT_TO, 25)
  while True:
    kw = {n: float(rng.uniform(*r)) for n, r in RANGES.items()}
    if seed % 2 == 0:
      kw["stiffness"] = 0.0
    lid = Lid(springref=SPRING_SLACK, **kw)
    if closing_torque(lid, th, SWEPT_RATE).min() >= CLOSING_MARGIN_NM:
      return replace(lid, catch_n=float(rng.uniform(*CATCH_RANGE)))


# ---- the catch ----------------------------------------------------------------

#: The magnet's pull falls as (1 + gap/G0)^-2 with the gap at the knob, and
#: is cut to nothing at CATCH_CUT (ten G0), less its pull there so it lets
#: go to nothing (#469's `--latch`).
CATCH_G0 = 0.0005
CATCH_CUT = 10 * CATCH_G0


def catch_torque(lid: Lid, angle: float) -> float:
  """The catch's pull on the lid at `angle` (rad), N*m about the hinge, -
  holding it shut: `catch_n` at the knob shut, falling with the gap."""
  if lid.catch_n <= 0.0:
    return 0.0
  gap = angle * PIN_R
  if gap > CATCH_CUT:
    return 0.0
  pull = lid.catch_n * ((1.0 + max(gap, 0.0) / CATCH_G0) ** -2
                        - (1.0 + CATCH_CUT / CATCH_G0) ** -2)
  return -pull * PIN_R


# ---- the MJCF -----------------------------------------------------------------

def _f(v: float) -> str:
  """Every digit: the world's chest is its `Lid` exactly (a mass cut to six
  figures left the reference's moments 6e-7 off the world's)."""
  return repr(float(v))


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def chest_xml(lid: Lid, name: str = "chest", tags: bool = True) -> str:
  """The chest as a `<mujoco>` to attach at its pose: a static base, the lid
  on its hinge, the handle on its pin, and the hinge's angle sensor. The
  lid's rest on the walls is its hinge's lower limit, so the lid and its
  base do not collide. `tags` puts the knob's tag on its faces, whose
  material (`tagmat<KNOB_TAG_ID>`) the world declares (`attach_chest`)."""
  t, d, w, h = WALL_T, BOX_D, BOX_W, BOX_H
  g = [f'<geom name="{name}_floor" type="box" size="{_v(d / 2, w / 2, t / 2)}" '
       f'pos="{_v(d / 2, 0, t / 2)}" {WOOD}/>',
       f'<geom name="{name}_back" type="box" size="{_v(t / 2, w / 2, h / 2)}" '
       f'pos="{_v(d - t / 2, 0, h / 2)}" {WOOD}/>',
       f'<geom name="{name}_front" type="box" size="{_v(t / 2, w / 2, h / 2)}" '
       f'pos="{_v(t / 2, 0, h / 2)}" {WOOD}/>']
  for s, lbl in ((-1, "r"), (1, "l")):
    g.append(f'<geom name="{name}_side{lbl}" type="box" size="{_v(d / 2, t / 2, h / 2)}" '
             f'pos="{_v(d / 2, s * (w / 2 - t / 2), h / 2)}" {WOOD}/>')
  skin = f'material="tagmat{KNOB_TAG_ID}"' if tags else METAL
  knob = (f'<body name="{name}_knob_body" pos="{_v(*KNOB_OFF)}">'
          f'<geom name="{name}_knob" type="box" size="{_v(KNOB, KNOB, KNOB)}" '
          f'mass="{_f(HANDLE_KG[2])}" {skin}/></body>')
  lid_parts = [
    f'<joint name="{name}_hinge" type="hinge" axis="0 1 0" range="{_v(*HINGE_RANGE)}" '
    f'armature="{_f(HINGE_ARMATURE)}" stiffness="{_f(lid.stiffness)}" '
    f'springref="{_f(lid.springref)}" damping="{_f(lid.damping)}" '
    f'frictionloss="{_f(lid.friction)}"/>',
    f'<geom name="{name}_lid" type="box" size="{_v(d / 2, w / 2, LID_T / 2)}" '
    f'pos="{_v(-d / 2, 0, LID_T / 2)}" mass="{_f(lid.lid_kg)}" {LID_WOOD}/>',
    f'<geom name="{name}_bracket" type="box" size="{_v(PIN_AHEAD / 2 + 0.003, 0.006, 0.004)}" '
    f'pos="{_v(-d - PIN_AHEAD / 2 + 0.003, 0, PIN_Z)}" mass="{_f(BRACKET_KG)}" {LID_WOOD}/>',
    f'<body name="{name}_handle" pos="{_v(*PIN)}">'
    f'<joint name="{name}_pin" type="hinge" axis="0 1 0" damping="{_f(PIN_DAMPING)}" '
    f'frictionloss="{_f(PIN_FRICTION)}" armature="{_f(PIN_ARMATURE)}"/>'
    f'<geom name="{name}_drop" type="box" size="{_v(0.003, STEM_HALF_W, HANDLE_DROP / 2)}" '
    f'pos="{_v(0, 0, -HANDLE_DROP / 2)}" mass="{_f(HANDLE_KG[0])}" {METAL}/>'
    f'<geom name="{name}_stem" type="box" size="{_v(STEM / 2 + 0.003, STEM_HALF_W, 0.004)}" '
    f'pos="{_v(-STEM / 2 + 0.003, 0, -HANDLE_DROP)}" mass="{_f(HANDLE_KG[1])}" {METAL}/>'
    + knob + '</body>']
  if lid.lump_kg > 0.0:
    lid_parts.append(
      f'<geom name="{name}_lump" type="box" size="{_v(*LUMP_HALF)}" '
      f'pos="{_v(-d * lid.lump_at, 0, lid.lump_z)}" mass="{_f(lid.lump_kg)}" '
      f'contype="0" conaffinity="0" group="3" rgba="0.8 0.2 0.2 1"/>')
  moving = (f'<body name="{name}_lid_body" pos="{_v(*HINGE)}">' + "".join(lid_parts)
            + '</body>')
  contact = (f'<exclude body1="{name}" body2="{name}_lid_body"/>'
             + "".join(f'<pair geom1="{name}_{part}" geom2="{name}_lid"/>'
                       for part in ("drop", "stem", "knob")))
  return (f'<mujoco><compiler angle="radian"/><worldbody>'
          f'<body name="{name}">' + "".join(g) + moving + '</body>'
          f'</worldbody><contact>{contact}</contact>'
          f'<sensor><jointpos name="{name}_lid" joint="{name}_hinge"/></sensor></mujoco>')


def attach_chest(spec, lid: Lid, pos, yaw: float = 0.0, name: str = "chest",
                 tags: bool = True) -> None:
  """The chest into a world's `spec`, its frame at `pos` (x, y on the floor)
  turned `yaw` (rad) about z; with `tags`, the knob's tag texture too, off
  the generator's `tags/` beside the world's file (`legs.world`'s)."""
  import mujoco
  if tags:
    spec.add_texture(name=f"tagtex{KNOB_TAG_ID}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{KNOB_TAG_ID}.png")
    mat = spec.add_material(name=f"tagmat{KNOB_TAG_ID}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{KNOB_TAG_ID}"
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), 0.0]
  frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  spec.attach(mujoco.MjSpec.from_string(chest_xml(lid, name, tags)), prefix="", frame=frame)


# ---- the activity ---------------------------------------------------------------

#: The lid reads OPEN past 10 deg, and shut again under 5: the knob 50 mm up
#: against 25 (the gap is the hysteresis).
OPEN_ON, OPEN_OFF = math.radians(10.0), math.radians(5.0)
#: ...and OPENED, latched, once past 57 deg: what #469 counted as open, of
#: the 69 a sweep asks (an arm at Kp 8 leaves it 4 short).
OPENED_RAD = 1.0
#: The catch has LET GO once the knob is past its reach (`CATCH_CUT`), and
#: holds again within half of it.
CAUGHT_ON, CAUGHT_OFF = CATCH_CUT / 2 / PIN_R, CATCH_CUT / PIN_R


class Chest(Activity):
  """The chest's state machine, off the hinge's angle SENSOR (a real chest
  has an encoder or a reed switch there, never `qpos`):

    lid     LIVE: "shut", or "open" past `OPEN_ON` until under `OPEN_OFF`
    opened  LATCHED: the lid has been lifted past `OPENED_RAD`
    caught  LIVE: the knob within the catch's reach

  and the catch's pull, set here for the next step: the one piece of the
  chest's physics MuJoCo has no element for (`catch_torque`). ⚠ `lid` is the
  truth (`draw`): nothing of it is a flag."""

  def __init__(self, model, data, lid: Lid, name: str = "chest") -> None:
    super().__init__(name)
    self.lid = lid
    self.rebind(model, data)
    self.open = Threshold(on=OPEN_ON, off=OPEN_OFF)
    self.past = Threshold(on=OPENED_RAD, latch=True)
    self.held = Threshold(on=CAUGHT_ON, off=CAUGHT_OFF)
    self.held.value = True
    self.set(lid="shut", opened=False, caught=True)

  def rebind(self, model, data) -> None:
    self.sensor_adr = int(model.sensor(f"{self.name}_lid").adr[0])
    self.dof = int(model.jnt_dofadr[model.joint(f"{self.name}_hinge").id])

  def angle(self, data) -> float:
    """The lid's angle off its stop, rad, as its sensor reads it."""
    return float(data.sensordata[self.sensor_adr])

  def sense(self, model, data) -> None:
    a = self.angle(data)
    data.qfrc_applied[self.dof] = catch_torque(self.lid, a)
    self.update(a)

  def update(self, angle: float) -> None:
    """The flags off one reading of the lid's angle, rad."""
    is_open = self.open.update(angle)
    self.set(lid="open" if is_open else "shut", opened=self.past.update(angle),
             caught=self.held.update(angle))


# ---- the closest the scene language comes ------------------------------------

#: A static part's mass is not the world's to give (nothing moves it): the
#: base's board at the density MuJoCo gives a geom without one, kg/m^3.
BASE_DENSITY = 1000.0


def _euler_xyz(rot: np.ndarray) -> list[float]:
  """MuJoCo's x-y-z euler angles of a rotation, degrees."""
  b = math.asin(max(-1.0, min(1.0, float(rot[0, 2]))))
  a = math.atan2(-float(rot[1, 2]), float(rot[2, 2]))
  c = math.atan2(-float(rot[0, 1]), float(rot[0, 0]))
  return [round(math.degrees(v), 9) for v in (a, b, c)]


def reference_document(lid: Lid, pos, yaw: float = 0.0, pin: float = 0.0) -> dict:
  """The chest written in the scene language (`imagination.scene`) as
  closely as it allows, by code that knows the truth: the BEST-EXPRESSIBLE
  REFERENCE. Its frame is the map's (the world's, for a robot whose belief
  is exact), the chest's at `pos` turned `yaw`, in the language's units (mm,
  degrees, kg, N, N*m, rad for a joint's spring and damping).

  The handle hangs `pin` rad round its pin, where it rests (as a robot would
  see it). What it cannot carry is the module docstring's list: the catch
  here is one release force, `catch_n` at the pin (the magnet's own lever,
  `PIN_R`); the contact settings are the imagination's; the armature is
  gone."""
  c, s = math.cos(yaw), math.sin(yaw)
  x0, y0 = float(pos[0]), float(pos[1])

  def at(p) -> list[float]:
    """A chest-frame point in the map, mm."""
    px, py, pz = (float(v) for v in p)
    return [round(1000 * (x0 + c * px - s * py), 6), round(1000 * (y0 + s * px + c * py), 6),
            round(1000 * pz, 6)]

  def mm(half) -> list[float]:
    return [round(2000 * float(v), 6) for v in half]

  euler = [0.0, 0.0, round(math.degrees(yaw), 9)]
  cp, sp = math.cos(pin), math.sin(pin)
  swing = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
  hung = _euler_xyz(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ swing)
  t, d, w, h = WALL_T, BOX_D, BOX_W, BOX_H

  def base(pid, half, centre, on=None):
    vol = 8 * half[0] * half[1] * half[2]
    part = {"id": pid, "shape": "box", "size": mm(half), "pos": at(centre), "euler": euler,
            "mass": round(BASE_DENSITY * vol, 6)}
    return part | ({"on": on} if on else {})

  # the walls ride the floor: one rigid base, as the world's is one body
  parts = [base("floor", (d / 2, w / 2, t / 2), (d / 2, 0, t / 2)),
           base("back", (t / 2, w / 2, h / 2), (d - t / 2, 0, h / 2), "floor"),
           base("front", (t / 2, w / 2, h / 2), (t / 2, 0, h / 2), "floor"),
           base("side_r", (d / 2, t / 2, h / 2), (d / 2, -(w / 2 - t / 2), h / 2), "floor"),
           base("side_l", (d / 2, t / 2, h / 2), (d / 2, w / 2 - t / 2, h / 2), "floor")]
  lid_frame = HINGE                              # the lid's body, shut
  pin_at = lid_frame + PIN
  parts += [
    {"id": "lid", "shape": "slab", "size": mm((d / 2, w / 2, LID_T / 2)),
     "pos": at(lid_frame + [-d / 2, 0, LID_T / 2]), "euler": euler, "on": "back",
     "mass": lid.lid_kg},
    {"id": "bracket", "shape": "box", "size": mm((PIN_AHEAD / 2 + 0.003, 0.006, 0.004)),
     "pos": at(lid_frame + [-d - PIN_AHEAD / 2 + 0.003, 0, PIN_Z]), "euler": euler,
     "on": "lid", "mass": BRACKET_KG},
    {"id": "handle", "shape": "box", "size": mm((0.003, STEM_HALF_W, HANDLE_DROP / 2)),
     "pos": at(pin_at + swing @ [0, 0, -HANDLE_DROP / 2]), "euler": hung, "on": "bracket",
     "mass": HANDLE_KG[0]},
    {"id": "stem", "shape": "box", "size": mm((STEM / 2 + 0.003, STEM_HALF_W, 0.004)),
     "pos": at(pin_at + swing @ [-STEM / 2 + 0.003, 0, -HANDLE_DROP]), "euler": hung,
     "on": "handle", "mass": HANDLE_KG[1]},
    {"id": "knob", "shape": "box", "size": mm((KNOB, KNOB, KNOB)),
     "pos": at(pin_at + swing @ KNOB_OFF), "euler": hung, "on": "stem",
     "mass": HANDLE_KG[2]}]
  if lid.lump_kg > 0.0:
    parts.append({"id": "lump", "shape": "box", "size": mm(LUMP_HALF),
                  "pos": at(lid_frame + [-d * lid.lump_at, 0, lid.lump_z]), "euler": euler,
                  "on": "lid", "mass": lid.lump_kg})
  axis = [round(-s, 9), round(c, 9), 0.0]       # the chest's y in the map
  joints = [
    {"id": "hinge", "type": "hinge", "part": "lid", "at": at(HINGE), "axis": axis,
     "range": [math.degrees(v) for v in HINGE_RANGE], "stiffness": lid.stiffness,
     "slack": math.degrees(lid.springref), "damping": lid.damping, "friction": lid.friction},
    {"id": "pin", "type": "hinge", "part": "handle", "at": at(pin_at), "axis": axis,
     "damping": PIN_DAMPING, "friction": PIN_FRICTION}]
  doc = {"parts": parts, "joints": joints}
  if lid.catch_n > 0.0:
    doc["catches"] = [{"joint": "hinge", "at": at(pin_at), "release": lid.catch_n}]
  return doc
