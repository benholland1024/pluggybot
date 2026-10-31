"""A drawer in a cabinet on the counter (issue #525): one of the hands'
capability checks (docs/Hands.md), as an ACTIVITY (docs/ActivityPattern.md)
-- it owns its geometry, its hidden parameters and its state, and reports
what is sensed of it. Never in the served world.

A cabinet 0.34 m wide, 0.30 deep and 0.16 high of 12 mm board, open at the
front, and a drawer in it that slides out toward the robot on its runners
-- a slide joint, `TRAVEL` long, that holds where it is left. Its handle is
a KNOB, the 26 mm cube the chest's is, on a stem from the middle of its
front, wearing its own tag on every face (`rack.tags.DRAWER_TAG_IDS`).

THE HIDDEN PARAMETERS (`Runners`, drawn by `draw`): the runners' Coulomb
friction and damping, and what the drawer weighs with what is in it. ⚠
THE TRUTH IS KEPT AS `Task.secret` IS: never in the robot's context.

The frame is the CABINET's: origin on its floor's level under the middle
of its front face (the robot's side), +x into it, y to the robot's left, z
up -- the chest's.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity.base import Activity, Threshold
from pluggybot.challenge.stack import BLOCK_HALF
from pluggybot.rack.tags import DRAWER_TAG_IDS

#: The cabinet: its depth, width and height, and its board.
BOX_D, BOX_W, BOX_H, WALL_T = 0.30, 0.34, 0.16, 0.012
#: The drawer's front panel's thickness, and the gap round it in the
#: cabinet's mouth, m.
FRONT_T, GAP = 0.015, 0.002
#: How far it slides out, m: the stop at the end of its runners.
TRAVEL = 0.24
#: The knob: the chest's 26 mm cube, on a stem this long out of the front's
#: middle, m.
KNOB, STEM = BLOCK_HALF, 0.020
KNOB_TAG_ID = DRAWER_TAG_IDS[0]
#: Where the knob's middle is, shut, in the cabinet's frame.
KNOB_AT = np.array([-STEM - KNOB, 0.0, BOX_H / 2])

WOOD = 'rgba="0.50 0.40 0.30 1"'
FRONT_WOOD = 'rgba="0.66 0.52 0.36 1"'


@dataclass(frozen=True)
class Runners:
  """One drawer's hidden parameters: its runners' Coulomb friction (N) and
  damping (N*s/m), and what it weighs with what is in it (kg)."""

  friction: float = 3.0
  damping: float = 3.0
  drawer_kg: float = 0.8


#: The ranges `draw` draws from: a light drawer's to a heavy one's that
#: still slides on a soft pull.
RANGES = {"friction": (1.5, 6.0), "damping": (1.0, 6.0), "drawer_kg": (0.4, 1.6)}
#: `draw(k)`'s stream is this and k (`counter._drawn`).
DRAW_SEED = 5250


def draw(seed: int) -> Runners:
  """Drawer `seed`'s hidden parameters (seeded): drawn from `RANGES`."""
  if seed < 0:
    raise ValueError(f"a drawer's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng((DRAW_SEED, seed))
  return Runners(**{n: float(rng.uniform(*r)) for n, r in RANGES.items()})


def _f(v: float) -> str:
  return repr(float(v))


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def cabinet_xml(runners: Runners, name: str = "drawer", tags: bool = True) -> str:
  """The cabinet as a `<mujoco>` to attach at its pose: a static carcass,
  the drawer on its slide, the knob on its stem, and the slide's sensor.
  The drawer never touches the carcass (its runners are the joint); `tags`
  puts the knob's tag on its faces (`tagmat<KNOB_TAG_ID>`, which the world
  declares: `attach_cabinet`)."""
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
  # the drawer: its front flush with the cabinet's mouth when shut, its tray
  # behind it inside the carcass
  fw, fh = w / 2 - t - GAP, h / 2 - t - GAP
  tray_d = d - t - FRONT_T - 0.01
  skin = f'material="tagmat{KNOB_TAG_ID}"' if tags else 'rgba="0.20 0.45 0.75 1"'
  tray = [
    f'<joint name="{name}_slide" type="slide" axis="-1 0 0" range="0 {_f(TRAVEL)}" '
    f'damping="{_f(runners.damping)}" frictionloss="{_f(runners.friction)}"/>',
    f'<geom name="{name}_front" type="box" size="{_v(FRONT_T / 2, fw, fh)}" '
    f'pos="{_v(FRONT_T / 2, 0, h / 2)}" mass="{_f(0.25 * runners.drawer_kg)}" {FRONT_WOOD}/>',
    f'<geom name="{name}_tray" type="box" size="{_v(tray_d / 2, fw - 0.004, 0.004)}" '
    f'pos="{_v(FRONT_T + tray_d / 2, 0, t + GAP + 0.004)}" mass="{_f(0.75 * runners.drawer_kg)}" '
    f'{WOOD}/>',
    f'<geom name="{name}_stem" type="box" size="{_v(STEM / 2, 0.004, 0.004)}" '
    f'pos="{_v(-STEM / 2, 0, h / 2)}" mass="0.005" rgba="0.20 0.45 0.75 1"/>',
    f'<geom name="{name}_knob" type="box" size="{_v(KNOB, KNOB, KNOB)}" '
    f'pos="{_v(*KNOB_AT)}" mass="0.02" {skin}/>']
  moving = f'<body name="{name}_drawer">' + "".join(tray) + '</body>'
  return (f'<mujoco><compiler angle="radian"/><worldbody>'
          f'<body name="{name}">' + "".join(carcass) + moving + '</body>'
          f'</worldbody><contact><exclude body1="{name}" body2="{name}_drawer"/></contact>'
          f'<sensor><jointpos name="{name}_out" joint="{name}_slide"/></sensor></mujoco>')


def attach_cabinet(spec, runners: Runners, pos, yaw: float = 0.0, name: str = "drawer",
                   tags: bool = True, z: float = 0.0) -> None:
  """The cabinet into a world's `spec`, its frame at `pos` (x, y), `z` up
  (on a counter), turned `yaw` (rad) about z; with `tags`, the knob's tag
  texture too (`models/tags/`), once a world."""
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
  spec.attach(mujoco.MjSpec.from_string(cabinet_xml(runners, name, tags)), prefix="",
              frame=frame)


# ---- the activity ---------------------------------------------------------------

#: The drawer reads OPEN past 2 cm and shut again under 1 cm, and OPENED,
#: latched, once past the job's 15 cm (`challenge.drawer.OPEN_M`'s).
OPEN_ON, OPEN_OFF, OPENED_M = 0.02, 0.01, 0.15


class Drawer(Activity):
  """The drawer's state machine, off its slide's position SENSOR (a real
  drawer has a reed switch or a draw-wire there, never `qpos`):

    drawer  LIVE: "shut", or "open" past `OPEN_ON` until under `OPEN_OFF`
    opened  LATCHED: it has been out past `OPENED_M`

  ⚠ `runners` is the truth (`draw`): nothing of it is a flag."""

  def __init__(self, model, data, runners: Runners, name: str = "drawer") -> None:
    super().__init__(name)
    self.runners = runners
    self.sensor_adr = int(model.sensor(f"{name}_out").adr[0])
    self.open = Threshold(on=OPEN_ON, off=OPEN_OFF)
    self.past = Threshold(on=OPENED_M, latch=True)
    self.set(drawer="shut", opened=False)

  def kept_state(self) -> dict:
    """The flags and the thresholds behind them (#559: the served counter's
    drawer rides a restart)."""
    return {**super().kept_state(), "open": self.open.value, "past": self.past.value}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    self.open.value = bool(state.get("open", False))
    self.past.value = bool(state.get("past", False))

  def out(self, data) -> float:
    """How far the drawer is out, m, as its sensor reads it."""
    return float(data.sensordata[self.sensor_adr])

  def sense(self, model, data) -> None:
    out = self.out(data)
    self.set(drawer="open" if self.open.update(out) else "shut", opened=self.past.update(out))
