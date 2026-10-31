"""A cupboard door with a lever handle (issue #525): one of the hands'
capability checks (docs/Hands.md), as an ACTIVITY (docs/ActivityPattern.md)
-- its geometry, its hidden parameters and its latch. On the humanoids'
storeroom counter (#559, `home.storeroom`), never the quadrupeds' world.

A cupboard 0.24 m wide, 0.24 deep and 0.30 high on the counter, its door
hung on a hinge down its left edge (the robot's left) so it swings open
toward the robot, and a LEVER HANDLE: an arm 9 cm long from a pivot on the
door's face near the hinge, pointing toward the door's free edge, sprung
back up and stopped at `LEVER_TRAVEL`, its tip the chest's 26 mm cube
wearing its own tag on every face (`rack.tags.DOOR_TAG_IDS`), in the door's
middle. ⚠ HINGED ON THE LEFT: hung on its right, the door swung into the
left arm that held its lever and stopped there, 36-45 deg open; ⚠ AND THE
LEVER POINTING AWAY FROM IT: pointing toward the hinge, pressing it down
rolled the left wrist toward its stop, and 6 presses of 8 went 1-3 cm
(#525). A LATCH holds the door
shut until the lever is pressed past `UNLATCH_RAD`; it catches again when
the door is shut with the lever up.

THE HIDDEN PARAMETERS (`Hinge`, drawn by `draw`): the door's hinge friction
and damping, what the door weighs, and the lever's spring. ⚠ THE TRUTH IS
KEPT AS `Task.secret` IS: never in the robot's context.

The frame is the CUPBOARD's: origin on its floor's level under the middle
of its front face (the robot's side), +x into it, y to the robot's left, z
up -- the chest's.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity.base import Activity, Threshold
from pluggybot.challenge.stack import BLOCK_HALF
from pluggybot.rack.tags import DOOR_TAG_IDS

#: The carcass: depth, width, height and board, m.
BOX_D, BOX_W, BOX_H, WALL_T = 0.24, 0.24, 0.30, 0.012
#: The door: its thickness, and the gap round it, m; it covers the mouth.
DOOR_T, GAP = 0.015, 0.002
DOOR_W, DOOR_H = BOX_W - 2 * GAP, BOX_H - 2 * GAP
#: How far the door opens on its hinge, rad (100 deg: a cupboard's stop).
DOOR_TRAVEL = math.radians(100.0)
#: The lever: its pivot this far in from the door's hinged edge, at the
#: door's middle height, standing this far proud of its face on a boss --
#: a lever handle's 5-6 cm: at 18 mm the jaws round its tip met the door's
#: face and held it shut against a 40 N pull (#525) -- its arm this long
#: toward the free edge; pressed down to this, rad, against its spring, and
#: stopped there.
PIVOT_IN, BOSS, LEVER_L = 0.035, 0.045, 0.090
LEVER_TRAVEL = math.radians(45.0)
#: The latch TRIPS -- frees the door -- once the lever is pressed past this
#: much of its travel, rad, and stays tripped until the door has been open
#: past half `KICK_RAD` and is shut again -- its hinge within this, rad -- with the
#: lever up: a trip latch, so a door held shut by the hand that pressed it
#: is free once that hand lets go...
UNLATCH_RAD, SHUT_RAD = math.radians(32.0), math.radians(0.3)
#: ...and while tripped its kick spring pushes the door ajar to this, rad,
#: at this stiffness, N*m/rad, damped by this, N*m*s/rad: 0.25 N*m shut,
#: past every drawn hinge's friction, and soft-closed onto its end, so a
#: freed door stands a few degrees open (#525: one grip could not hold the
#: lever down and leave the door's hinge free in the jaws, and the hand
#: gripping the lever held the door shut; an undamped 0.25 N*m swung a
#: low-friction door 33-48 deg, out of the hand's reach).
KICK_RAD, KICK_K, KICK_D = math.radians(6.0), 2.4, 0.4
#: The lever's tip: the chest's 26 mm cube.
KNOB = BLOCK_HALF
KNOB_TAG_ID = DOOR_TAG_IDS[0]
#: The hinge's line (its foot), and the lever's pivot, the cupboard's frame.
HINGE_AT = np.array([-DOOR_T, DOOR_W / 2, 0.0])
PIVOT_AT = np.array([-DOOR_T - BOSS, DOOR_W / 2 - PIVOT_IN, BOX_H / 2])
#: The tip's middle with the lever up, the cupboard's frame: its back face
#: flush with the arm's.
TIP_OFF = np.array([-KNOB + 0.006, -LEVER_L, 0.0])
KNOB_AT = PIVOT_AT + TIP_OFF

WOOD = 'rgba="0.47 0.38 0.30 1"'
DOOR_WOOD = 'rgba="0.68 0.56 0.40 1"'
METAL = 'rgba="0.70 0.70 0.74 1"'


@dataclass(frozen=True)
class Hinge:
  """One door's hidden parameters: its hinge's Coulomb friction (N*m) and
  damping (N*m*s/rad), what the door weighs (kg), and its lever's spring
  (N*m/rad, slack up)."""

  friction: float = 0.06
  damping: float = 0.08
  door_kg: float = 0.7
  lever_spring: float = 0.8


#: The ranges `draw` draws from: a lever whose spring alone asks 2.5-7.4 N
#: at its tip to trip the latch and 3.5-10.5 N at its stop.
RANGES = {"friction": (0.02, 0.15), "damping": (0.02, 0.20), "door_kg": (0.4, 1.0),
          "lever_spring": (0.4, 1.2)}
#: `draw(k)`'s stream is this and k (`counter._drawn`).
DRAW_SEED = 5252


def draw(seed: int) -> Hinge:
  """Door `seed`'s hidden parameters (seeded): drawn from `RANGES`."""
  if seed < 0:
    raise ValueError(f"a door's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng((DRAW_SEED, seed))
  return Hinge(**{n: float(rng.uniform(*r)) for n, r in RANGES.items()})


def _f(v: float) -> str:
  return repr(float(v))


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def cupboard_xml(hinge: Hinge, name: str = "door", tags: bool = True) -> str:
  """The cupboard as a `<mujoco>` to attach at its pose: a static carcass,
  the door on its hinge, the lever on its pivot, and the two sensors. The
  door's limit is the latch's (`Door.sense`); `tags` puts the tip's tag
  on its faces (`tagmat<KNOB_TAG_ID>`, which the world declares)."""
  t, d, w, h = WALL_T, BOX_D, BOX_W, BOX_H
  carcass = [
    f'<geom name="{name}_floor" type="box" size="{_v(d / 2, w / 2, t / 2)}" '
    f'pos="{_v(d / 2, 0, t / 2)}" {WOOD}/>',
    f'<geom name="{name}_top" type="box" size="{_v(d / 2, w / 2, t / 2)}" '
    f'pos="{_v(d / 2, 0, h - t / 2)}" {WOOD}/>',
    f'<geom name="{name}_back" type="box" size="{_v(t / 2, w / 2, h / 2)}" '
    f'pos="{_v(d - t / 2, 0, h / 2)}" {WOOD}/>']
  for s, lbl in ((-1, "r"), (1, "l")):
    carcass.append(f'<geom name="{name}_side{lbl}" type="box" size="{_v(d / 2, t / 2, h / 2)}" '
                   f'pos="{_v(d / 2, s * (w / 2 - t / 2), h / 2)}" {WOOD}/>')
  skin = f'material="tagmat{KNOB_TAG_ID}"' if tags else METAL
  # the lever, in its pivot's frame: its arm along -y, its tip at its end
  lever = (f'<body name="{name}_lever" pos="{_v(*(PIVOT_AT - HINGE_AT))}">'
           f'<joint name="{name}_lever_pivot" type="hinge" axis="1 0 0" '
           f'range="0 {_f(LEVER_TRAVEL)}" stiffness="{_f(hinge.lever_spring)}" springref="0" '
           f'damping="0.01" armature="0.0002"/>'
           f'<geom name="{name}_boss" type="cylinder" size="0.008 {_f(BOSS / 2)}" '
           f'pos="{_v(BOSS / 2, 0, 0)}" quat="0.7071068 0 0.7071068 0" mass="0.02" {METAL}/>'
           f'<geom name="{name}_arm" type="box" size="{_v(0.006, LEVER_L / 2, 0.005)}" '
           f'pos="{_v(0, -LEVER_L / 2, 0)}" mass="0.03" {METAL}/>'
           f'<geom name="{name}_tip" type="box" size="{_v(KNOB, KNOB, KNOB)}" '
           f'pos="{_v(*TIP_OFF)}" mass="0.02" {skin}/></body>')
  door = (f'<body name="{name}_door" pos="{_v(*HINGE_AT)}">'
          f'<joint name="{name}_hinge" type="hinge" axis="0 0 -1" range="0 {_f(DOOR_TRAVEL)}" '
          f'damping="{_f(hinge.damping)}" frictionloss="{_f(hinge.friction)}" armature="0.001"/>'
          f'<geom name="{name}_panel" type="box" size="{_v(DOOR_T / 2, DOOR_W / 2, DOOR_H / 2)}" '
          f'pos="{_v(DOOR_T / 2, -DOOR_W / 2, h / 2)}" mass="{_f(hinge.door_kg)}" {DOOR_WOOD}/>'
          + lever + '</body>')
  return (f'<mujoco><compiler angle="radian"/><worldbody>'
          f'<body name="{name}">' + "".join(carcass) + door + '</body>'
          f'</worldbody><contact><exclude body1="{name}" body2="{name}_door"/>'
          f'<exclude body1="{name}_door" body2="{name}_lever"/></contact>'
          f'<sensor><jointpos name="{name}_open" joint="{name}_hinge"/>'
          f'<jointpos name="{name}_pressed" joint="{name}_lever_pivot"/></sensor></mujoco>')


def attach_cupboard(spec, hinge: Hinge, pos, yaw: float = 0.0, name: str = "door",
                    tags: bool = True, z: float = 0.0) -> None:
  """The cupboard into a world's `spec`, its frame at `pos` (x, y), `z` up,
  turned `yaw` (rad) about z; with `tags`, the tip's tag texture, once."""
  import mujoco
  if tags and spec.texture(f"tagtex{KNOB_TAG_ID}") is None:
    spec.add_texture(name=f"tagtex{KNOB_TAG_ID}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{KNOB_TAG_ID}.png")
    mat = spec.add_material(name=f"tagmat{KNOB_TAG_ID}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{KNOB_TAG_ID}"
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), float(z)]
  frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  spec.attach(mujoco.MjSpec.from_string(cupboard_xml(hinge, name, tags)), prefix="",
              frame=frame)


# ---- the activity ---------------------------------------------------------------

#: The door reads OPEN past 3 deg and shut again under 1, and OPENED,
#: latched, once past the job's (`challenge.door.OPEN_RAD`'s 50 deg).
OPEN_ON, OPEN_OFF, OPENED_RAD = math.radians(3.0), math.radians(1.0), math.radians(50.0)


class Door(Activity):
  """The door's state machine, off its two angle SENSORS (a real door's
  reed switch and a lever's microswitch, never `qpos`):

    door      LIVE: "shut", or "open" past `OPEN_ON` until under `OPEN_OFF`
    opened    LATCHED: it has been open past `OPENED_RAD`
    latched   LIVE: the latch holds it shut

  and the LATCH, set here for the next step: the door's hinge limited to
  shut -- a bolt -- until the lever is pressed past `UNLATCH_RAD`, which
  TRIPS it: free, its KICK spring pushing the door ajar to `KICK_RAD`,
  until the door has been open past half of it and is shut
  again with the lever up. ⚠ The latch is PHYSICS, off the joints as they
  are (`qpos`), as the chest's catch is. ⚠ `hinge` is the truth (`draw`):
  nothing of it is a flag."""

  def __init__(self, model, data, hinge: Hinge, name: str = "door") -> None:
    super().__init__(name)
    self.hinge = hinge
    self._model = model
    self.open_adr = int(model.sensor(f"{name}_open").adr[0])
    self.lever_adr = int(model.sensor(f"{name}_pressed").adr[0])
    door = model.joint(f"{name}_hinge").id
    lever = model.joint(f"{name}_lever_pivot").id
    self.door_jnt = door
    self.door_q = int(model.jnt_qposadr[door])
    self.door_dof = int(model.jnt_dofadr[door])
    self.lever_q = int(model.jnt_qposadr[lever])
    self.open = Threshold(on=OPEN_ON, off=OPEN_OFF)
    self.past = Threshold(on=OPENED_RAD, latch=True)
    self.bolted = True
    #: ...since its trip, whether the door has been open past half `KICK_RAD`
    self.ajar = False
    self._limit(model)
    self.set(door="shut", opened=False, latched=True)

  def _limit(self, model) -> None:
    model.jnt_range[self.door_jnt] = (0.0, SHUT_RAD if self.bolted else DOOR_TRAVEL)

  def kept_state(self) -> dict:
    """The flags, the latch and the thresholds behind them (#559: the
    served counter's door rides a restart): restored without them, an open
    door came back bolted shut round it."""
    return {**super().kept_state(), "bolted": self.bolted, "ajar": self.ajar,
            "open": self.open.value, "past": self.past.value}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    self.bolted = bool(state.get("bolted", True))
    self.ajar = bool(state.get("ajar", False))
    self.open.value = bool(state.get("open", False))
    self.past.value = bool(state.get("past", False))
    self._limit(self._model)

  def angle(self, data) -> float:
    """The door's angle off shut, rad, as its sensor reads it."""
    return float(data.sensordata[self.open_adr])

  def pressed(self, data) -> float:
    """The lever's angle off up, rad, as its sensor reads it."""
    return float(data.sensordata[self.lever_adr])

  def sense(self, model, data) -> None:
    door, lever = float(data.qpos[self.door_q]), float(data.qpos[self.lever_q])
    bolted = self.bolted
    if self.bolted and lever >= UNLATCH_RAD:
      bolted, self.ajar = False, False
    elif not self.bolted:
      self.ajar = self.ajar or door > KICK_RAD / 2
      bolted = self.ajar and door <= SHUT_RAD and lever < UNLATCH_RAD
    if bolted != self.bolted:
      self.bolted = bolted
      self._limit(model)
    kick = 0.0
    if not bolted and door < KICK_RAD:
      kick = KICK_K * (KICK_RAD - door) - KICK_D * float(data.qvel[self.door_dof])
    data.qfrc_applied[self.door_dof] = kick
    a = self.angle(data)
    self.set(door="open" if self.open.update(a) else "shut", opened=self.past.update(a),
             latched=self.bolted)
