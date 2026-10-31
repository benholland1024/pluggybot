"""A cube to be picked up and placed on a mat (issue #525): one of the
hands' capability checks (docs/Hands.md), as an ACTIVITY
(docs/ActivityPattern.md) -- its geometry, its hidden parameters and what
is sensed of it. On the humanoids' storeroom counter (#559,
`home.storeroom`), never the quadrupeds' world.

The cube is the chest's knob's 26 mm cube, loose on the counter, wearing
its own tag on every face (`rack.tags.CUBE_TAG_IDS`). The mat is a plate
80 mm square and 4 mm thick on the counter beside it, its top wearing its
own tag across it (`MAT_TAG_IDS`).

THE HIDDEN PARAMETERS (`Load`, drawn by `draw`): what the cube weighs and
how it slides on what it rests on. ⚠ THE TRUTH IS KEPT AS `Task.secret`
IS: never in the robot's context.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity.base import Activity, Threshold
from pluggybot.challenge.stack import BLOCK_HALF
from pluggybot.rack.tags import CUBE_TAG_IDS, MAT_TAG_IDS

#: The cube's half side, m: the chest's knob's.
CUBE = BLOCK_HALF
#: The mat's half side and half thickness, m.
MAT, MAT_T = 0.040, 0.002
CUBE_TAG_ID, MAT_TAG_ID = CUBE_TAG_IDS[0], MAT_TAG_IDS[0]


@dataclass(frozen=True)
class Load:
  """One cube's hidden parameters: what it weighs (kg) and its sliding
  friction on what it rests on."""

  cube_kg: float = 0.10
  friction: float = 0.8


#: The ranges `draw` draws from: a die's weight to twice a steel cube's.
RANGES = {"cube_kg": (0.03, 0.30), "friction": (0.5, 1.0)}
#: `draw(k)`'s stream is this and k (`counter._drawn`).
DRAW_SEED = 5254


def draw(seed: int) -> Load:
  """Cube `seed`'s hidden parameters (seeded): drawn from `RANGES`."""
  if seed < 0:
    raise ValueError(f"a cube's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng((DRAW_SEED, seed))
  return Load(**{n: float(rng.uniform(*r)) for n, r in RANGES.items()})


def _f(v: float) -> str:
  return repr(float(v))


def _material(spec, tag_id: int) -> None:
  import mujoco
  if spec.texture(f"tagtex{tag_id}") is None:
    spec.add_texture(name=f"tagtex{tag_id}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{tag_id}.png")
    mat = spec.add_material(name=f"tagmat{tag_id}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{tag_id}"


def attach_props(spec, load: Load, cube_at, mat_at, yaw: float = 0.0, z: float = 0.0,
                 name: str = "cube", tags: bool = True) -> None:
  """The cube resting at `cube_at` (x, y) and the mat at `mat_at` (x, y),
  both on the surface `z` up (a counter), the cube turned `yaw` (rad); with
  `tags`, their tags' textures, once a world. The mat's tag is the texture
  across its top (a cube texture on a thin plate: its sides are slivers)."""
  import mujoco
  if tags:
    _material(spec, CUBE_TAG_ID)
    _material(spec, MAT_TAG_ID)
  mat = spec.worldbody.add_body(name=f"{name}_mat", pos=[float(mat_at[0]), float(mat_at[1]),
                                                        float(z) + MAT_T])
  mat.add_geom(name=f"{name}_mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[MAT, MAT, MAT_T],
               **({"material": f"tagmat{MAT_TAG_ID}"} if tags else {"rgba": [0.3, 0.3, 0.3, 1]}))
  body = spec.worldbody.add_body(name=name, pos=[float(cube_at[0]), float(cube_at[1]),
                                                 float(z) + CUBE],
                                 quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  body.add_freejoint(name=f"{name}_free")
  body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE, CUBE, CUBE],
                mass=float(load.cube_kg), friction=[float(load.friction), 0.005, 0.0001],
                **({"material": f"tagmat{CUBE_TAG_ID}"} if tags else {"rgba": [0.2, 0.45, 0.75, 1]}))


# ---- the activity ---------------------------------------------------------------

#: The cube reads LIFTED with its bottom this far over the counter, m, and
#: down again under half of it. On the mat or not is the grader's
#: (`challenge/place.py`).
LIFT_ON = 0.010


class Cube(Activity):
  """The cube's state machine, off its pose and the mat's (a scale and a
  camera over the counter, never the robot's senses):

    lifted  LIVE: its bottom more than `LIFT_ON` over the surface it was on

  ⚠ `load` is the truth (`draw`): nothing of it is a flag."""

  def __init__(self, model, data, load: Load, surface_z: float, name: str = "cube") -> None:
    super().__init__(name)
    self.body = int(model.body(name).id)
    self.surface_z = float(surface_z)
    self.renew(model, load)

  def renew(self, model, load: Load) -> None:
    """Begun again over `load`, a cube set out afresh (#559's counter): on
    its board, its threshold clear."""
    self.load = load
    self.lifted = Threshold(on=LIFT_ON, off=LIFT_ON / 2)
    self.set(lifted=False)

  def kept_state(self) -> dict:
    """The flag and the threshold behind it (#559: the served counter's
    cube rides a restart)."""
    return {**super().kept_state(), "lifted": self.lifted.value}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    self.lifted.value = bool(state.get("lifted", False))

  def sense(self, model, data) -> None:
    bottom = float(data.xpos[self.body][2]) - CUBE
    self.set(lifted=self.lifted.update(bottom - self.surface_z))
