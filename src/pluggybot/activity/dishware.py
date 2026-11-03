"""Dishware the hands can take (issue #560, a table set): a plate, a cup and
a fork, and the place card a place is laid at -- as an ACTIVITY
(docs/ActivityPattern.md): their geometry, their hidden parameters and
what is sensed of them. Free bodies only while a task sets them out: every
free body costs the physics, so no world carries them otherwise (#560).

Each is shaped for the Dex1-1's two jaws, which open 90 mm (`humanoid.parts`):

  plate  a disc with a wall round it, pinched by the wall -- one pad in the
         well, one outside -- its tag on the well's floor
  cup    a lidded cylinder taken across its body, its tag on the lid
  fork   a thick handle (an adaptive fork's) and a flat head, taken by the
         handle's sides, its tag on the handle's top
  card   a place card standing at a place's far edge, facing the diner,
         wearing the place's tag: what a place is laid by

THE HIDDEN PARAMETERS (`Load`, drawn by `draw`): what each piece weighs and
how it slides. ⚠ THE TRUTH IS KEPT AS `Task.secret` IS: never in the
robot's context.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity.base import Activity, Threshold
from pluggybot.perception.tags import (DISH_TAG_IDS, DISH_TAG_SIZES, PLACE_TAG_IDS,
                                       plate_half_extent)

#: The pieces, in the order a place is laid and their tags are numbered.
PIECES = ("plate", "cup", "fork")
TAG_OF = dict(zip(PIECES, DISH_TAG_IDS))
CARD_TAG_ID = PLACE_TAG_IDS[0]

#: THE PLATE: its outer radius, its wall's thickness and height, and its
#: floor's thickness, m. The wall is `PLATE_WALLS` boxes round it: a pad 30
#: mm wide meets a flat face. ⚠ 8 mm: pinched at 90 N, a 6 mm wall read as
#: jaws shut on nothing (`HumanoidBody.grip_routine`'s gap).
PLATE_R, PLATE_WALL_T, PLATE_H, PLATE_FLOOR_T = 0.090, 0.008, 0.028, 0.005
PLATE_WALLS = 16
#: THE CUP: its radius and height, m: taken across its body, 70 mm of the
#: jaws' 90.
CUP_R, CUP_H = 0.035, 0.090
#: THE FORK: its handle's half-extents -- along it, across it, up -- and its
#: head's, m; the head lies at the handle's far end, flat on the table. The
#: fork's frame is the handle's middle on the table, x along it toward the
#: head.
HANDLE = (0.055, 0.012, 0.011)
HEAD = (0.035, 0.014, 0.0025)
#: THE PLACE CARD: its half-extents -- across its face, its thickness, up --
#: m, and its tag's black square, m.
CARD = (0.06, 0.003, 0.055)
#: The tags' black squares, m: the plate's on its floor, the cup's on its
#: lid, the fork's on its handle, the card's on its face (the registry's).
TAG_SIZE = {**{p: DISH_TAG_SIZES[TAG_OF[p]] for p in PIECES}, "card": DISH_TAG_SIZES[CARD_TAG_ID]}
#: A tag's film over the face it is printed on, m: a visual geom, no contact.
FILM = 0.0004


@dataclass(frozen=True)
class Load:
  """One place's hidden parameters: what each piece weighs (kg) and its
  sliding friction on what it rests on."""

  plate_kg: float = 0.35
  cup_kg: float = 0.25
  fork_kg: float = 0.08
  friction: float = 0.7


#: The ranges `draw` draws from: melamine to stoneware, an empty cup to a
#: full one, a light fork to a heavy one.
RANGES = {"plate_kg": (0.25, 0.50), "cup_kg": (0.15, 0.35), "fork_kg": (0.04, 0.12),
          "friction": (0.5, 0.9)}
#: `draw(k)`'s stream is this and k: a stream of its own, never the set-out's
#: (`evaluation.counter._drawn`).
DRAW_SEED = 5601


def draw(seed: int) -> Load:
  """Place `seed`'s hidden parameters (seeded): drawn from `RANGES`."""
  if seed < 0:
    raise ValueError(f"a place's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng((DRAW_SEED, seed))
  return Load(**{n: float(rng.uniform(*r)) for n, r in RANGES.items()})


def _material(spec, tag_id: int) -> str:
  import mujoco
  name = f"tagmat{tag_id}"
  if spec.texture(f"tagtex{tag_id}") is None:
    spec.add_texture(name=f"tagtex{tag_id}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{tag_id}.png")
    mat = spec.add_material(name=name, specular=0.05, shininess=0.05, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{tag_id}"
  return name


def _film(body, name: str, tag_id: int, size: float, z: float, spec, tags: bool) -> None:
  """A tag printed flat on a face `z` up the body: a visual film, no contact."""
  import mujoco
  if not tags:
    return
  half = plate_half_extent(size)
  body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, FILM / 2],
                pos=[0.0, 0.0, z + FILM / 2], contype=0, conaffinity=0, mass=0.0,
                material=_material(spec, tag_id))


def _free(spec, name: str, at, yaw: float, z: float):
  body = spec.worldbody.add_body(name=name, pos=[float(at[0]), float(at[1]), float(z)],
                                 quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  body.add_freejoint(name=f"{name}_free")
  return body


def attach_plate(spec, at, yaw: float, z: float, kg: float, friction: float,
                 tags: bool = True, name: str = "plate") -> None:
  """The plate at `at` (x, y) on the surface `z` up, turned `yaw`: its floor,
  its wall in `PLATE_WALLS` boxes, its tag on the floor's middle."""
  import mujoco
  body = _free(spec, name, at, yaw, z)
  fr = [float(friction), 0.005, 0.0001]
  floor_r = PLATE_R - PLATE_WALL_T
  area = math.pi * floor_r ** 2 * PLATE_FLOOR_T
  wall = 2 * math.pi * (PLATE_R - PLATE_WALL_T / 2) * PLATE_WALL_T * PLATE_H
  rgba = [0.93, 0.92, 0.88, 1]
  body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[floor_r, PLATE_FLOOR_T / 2, 0], pos=[0, 0, PLATE_FLOOR_T / 2],
                mass=float(kg) * area / (area + wall), friction=fr, rgba=rgba)
  r = PLATE_R - PLATE_WALL_T / 2
  half_t = r * math.tan(math.pi / PLATE_WALLS) + PLATE_WALL_T / 2
  for i in range(PLATE_WALLS):
    a = 2 * math.pi * i / PLATE_WALLS
    body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[PLATE_WALL_T / 2, half_t, PLATE_H / 2],
                  pos=[r * math.cos(a), r * math.sin(a), PLATE_H / 2],
                  quat=[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)],
                  mass=float(kg) * wall / (area + wall) / PLATE_WALLS, friction=fr, rgba=rgba)
  _film(body, f"{name}_tag", TAG_OF["plate"], TAG_SIZE["plate"], PLATE_FLOOR_T, spec, tags)


def attach_cup(spec, at, yaw: float, z: float, kg: float, friction: float,
               tags: bool = True, name: str = "cup") -> None:
  """The cup standing at `at` (x, y) on the surface `z` up, its tag on its lid."""
  import mujoco
  body = _free(spec, name, at, yaw, z)
  body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_R, CUP_H / 2, 0],
                pos=[0, 0, CUP_H / 2], mass=float(kg), friction=[float(friction), 0.005, 0.0001],
                rgba=[0.80, 0.33, 0.25, 1])
  _film(body, f"{name}_tag", TAG_OF["cup"], TAG_SIZE["cup"], CUP_H, spec, tags)


def attach_fork(spec, at, yaw: float, z: float, kg: float, friction: float,
                tags: bool = True, name: str = "fork") -> None:
  """The fork lying at `at` (x, y) -- its handle's middle -- on the surface
  `z` up, its head toward `yaw`, its tag on the handle's top."""
  import mujoco
  body = _free(spec, name, at, yaw, z)
  fr = [float(friction), 0.005, 0.0001]
  hv = 8 * HANDLE[0] * HANDLE[1] * HANDLE[2]
  dv = 8 * HEAD[0] * HEAD[1] * HEAD[2]
  rgba = [0.75, 0.76, 0.78, 1]
  body.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HANDLE),
                pos=[0, 0, HANDLE[2]], mass=float(kg) * hv / (hv + dv), friction=fr, rgba=rgba)
  body.add_geom(name=f"{name}_head", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HEAD),
                pos=[HANDLE[0] + HEAD[0], 0, HEAD[2]], mass=float(kg) * dv / (hv + dv),
                friction=fr, rgba=rgba)
  _film(body, f"{name}_tag", TAG_OF["fork"], TAG_SIZE["fork"], 2 * HANDLE[2], spec, tags)


ATTACH = {"plate": attach_plate, "cup": attach_cup, "fork": attach_fork}


def attach_card(spec, at, yaw: float, z: float, tags: bool = True, name: str = "place_card"):
  """The place card standing at `at` (x, y) on the surface `z` up, its face
  -- and its tag -- toward `yaw` (the diner): fixed scenery."""
  import mujoco
  c, s = math.cos(yaw), math.sin(yaw)
  # the card's frame: its x across its face, its y out of it toward the diner
  q = [math.cos((yaw - math.pi / 2) / 2), 0.0, 0.0, math.sin((yaw - math.pi / 2) / 2)]
  body = spec.worldbody.add_body(name=name, pos=[float(at[0]), float(at[1]), float(z)], quat=q)
  body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(CARD),
                pos=[0, 0, CARD[2]], rgba=[0.96, 0.95, 0.90, 1])
  if tags:
    half = plate_half_extent(TAG_SIZE["card"])
    body.add_geom(name=f"{name}_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half, FILM / 2, half], pos=[0, CARD[1] + FILM / 2, CARD[2]],
                  contype=0, conaffinity=0, mass=0.0, material=_material(spec, CARD_TAG_ID))
  del c, s
  return body


# ---- the activity ---------------------------------------------------------------

#: A piece reads LIFTED with its bottom this far over where it rested, m,
#: and down again under half of it. Where it belongs is the grader's
#: (`challenge/table.py`).
LIFT_ON = 0.010


class Dishware(Activity):
  """The pieces' state machine, off their poses (a camera over the table,
  never the robot's senses):

    lifted  LIVE, each piece: its bottom more than `LIFT_ON` over the
            surface it was set out on

  ⚠ `load` is the truth (`draw`): nothing of it is a flag."""

  def __init__(self, model, data, load: Load, surface_z: float, name: str = "dishware") -> None:
    super().__init__(name)
    self.bodies = {p: int(model.body(p).id) for p in PIECES}
    self.surface_z = float(surface_z)
    self.renew(model, load)

  def renew(self, model, load: Load) -> None:
    """Begun again over `load`, the pieces set out afresh: their
    thresholds clear."""
    self.load = load
    self.lifted = {p: Threshold(on=LIFT_ON, off=LIFT_ON / 2) for p in PIECES}
    self.set(**{f"{p}Lifted": False for p in PIECES})

  def kept_state(self) -> dict:
    return {**super().kept_state(), "lifted": {p: t.value for p, t in self.lifted.items()}}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    for p, v in (state.get("lifted") or {}).items():
      if p in self.lifted:
        self.lifted[p].value = bool(v)

  def sense(self, model, data) -> None:
    self.set(**{f"{p}Lifted": self.lifted[p].update(float(data.xpos[b][2]) - self.surface_z)
                for p, b in self.bodies.items()})
