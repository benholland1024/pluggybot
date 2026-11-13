"""Dishware the hands can take (issue #560, a table set): a plate, a cup and
a fork, and the place card a place is laid at -- as an ACTIVITY
(docs/ActivityPattern.md): their geometry, their hidden parameters and
what is sensed of them. Live only while a task sets them out: every free
body costs the physics and every geom a ray, so the served world compiles
them PARKED (`park_spec`) and an offer sets them out (`place`), as
pre-allocated toggles do -- out of sight under the floor, their contacts
off, their weight held, in a geom group no sensor casts against and no
camera draws (`perception.cull.PARKED_GROUP`).

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
#: mm wide meets a flat face. ⚠ 8 mm: a margin over `HELD_GAP_M` (a test),
#: jaws stopped on it reading clear of closed, and every measure of the
#: table flew it.
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


def card_quat(yaw: float) -> list[float]:
  """The card's frame facing `yaw` (the diner): its x across its face, its y
  out of it toward the diner."""
  return [math.cos((yaw - math.pi / 2) / 2), 0.0, 0.0, math.sin((yaw - math.pi / 2) / 2)]


def attach_card(spec, at, yaw: float, z: float, tags: bool = True, name: str = "place_card",
                mocap: bool = False):
  """The place card standing at `at` (x, y) on the surface `z` up, its face
  -- and its tag -- toward `yaw` (the diner): fixed scenery, or, `mocap`, a
  body a job sets out and puts away (`place`, `park`)."""
  import mujoco
  body = spec.worldbody.add_body(name=name, pos=[float(at[0]), float(at[1]), float(z)],
                                 quat=card_quat(yaw), mocap=mocap)
  body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(CARD),
                pos=[0, 0, CARD[2]], rgba=[0.96, 0.95, 0.90, 1])
  if tags:
    half = plate_half_extent(TAG_SIZE["card"])
    body.add_geom(name=f"{name}_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half, FILM / 2, half], pos=[0, CARD[1] + FILM / 2, CARD[2]],
                  contype=0, conaffinity=0, mass=0.0, material=_material(spec, CARD_TAG_ID))
  return body


# ---- put away and set out (#560, the table set served) ------------------------

#: Where a piece waits while it is put away, m: this far under the floor
#: (under the table it is set out on), where a ray from over the floor never
#: comes and a visitor's page never looks -- belt and braces with its geom
#: group, which every cast leaves out.
PARK_Z = -1.0


def _is_film(model, g: int) -> bool:
  """A tag printed on a piece or the card (`_film`'s): it never collides."""
  return model.geom(g).name.endswith("_tag")


def _geoms(model, body: int) -> range:
  return range(int(model.body_geomadr[body]), int(model.body_geomadr[body] + model.body_geomnum[body]))


def park_spec(spec, names, at: dict) -> None:
  """The bodies `names` (the pieces and the card, attached) compiled PUT
  AWAY: each `at[name]` (x, y) at `PARK_Z`, its geoms' contacts off and in
  the parked group, a piece's weight held (`gravcomp`). Its live masks are
  its kind's (`_film`), so nothing of them need be kept."""
  from pluggybot.perception.cull import PARKED_GROUP
  for name in names:
    body = spec.body(name)
    body.pos = [float(at[name][0]), float(at[name][1]), PARK_Z]
    if not body.mocap:
      body.gravcomp = 1.0
    for geom in body.geoms:
      geom.contype, geom.conaffinity, geom.group = 0, 0, PARKED_GROUP


def _switch(model, body: int, live: bool) -> None:
  """A body's geoms and its broadphase masks live (a solid geom meets the
  world, a film nothing; group 0) or put away (nothing; the parked group),
  and a free piece's weight its own or held."""
  from pluggybot.perception.cull import PARKED_GROUP
  for g in _geoms(model, body):
    solid = live and not _is_film(model, g)
    model.geom_contype[g] = model.geom_conaffinity[g] = 1 if solid else 0
    model.geom_group[g] = 0 if live else PARKED_GROUP
  model.body_contype[body] = model.body_conaffinity[body] = 1 if live else 0
  if model.body_mocapid[body] < 0:
    model.body_gravcomp[body] = 0.0 if live else 1.0


def _pose(model, data, body: int, xyz, quat=None) -> None:
  """A free piece or the card at `xyz`, turned `quat` (as it was, None), and
  still."""
  mocap = int(model.body_mocapid[body])
  if mocap >= 0:
    data.mocap_pos[mocap] = xyz
    if quat is not None:
      data.mocap_quat[mocap] = quat
    return
  j = int(model.body_jntadr[body])
  q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
  data.qpos[q:q + 3] = xyz
  if quat is not None:
    data.qpos[q + 3:q + 7] = quat
  data.qvel[v:v + 6] = 0.0
  data.qacc_warmstart[v:v + 6] = 0.0
  data.qfrc_applied[v:v + 6] = 0.0


def _yaw_quat(yaw: float) -> list[float]:
  return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def place(model, data, pieces: dict, card: tuple, z: float, card_name: str = "place_card",
          piece_z: float | None = None) -> None:
  """The dishware SET OUT on a surface `z` up: each piece of `pieces`
  ({name: (x, y, yaw)}) resting where it is set out, as `attach_<piece>` puts
  it -- on the surface `piece_z` up where it is another (the pass, #682) --
  and the card standing at `card` (x, y, facing) -- all live and still."""
  for name, (x, y, yaw) in pieces.items():
    b = model.body(name).id
    _switch(model, b, True)
    _pose(model, data, b, (x, y, z if piece_z is None else piece_z), _yaw_quat(yaw))
  b = model.body(card_name).id
  _switch(model, b, True)
  _pose(model, data, b, (card[0], card[1], z), card_quat(card[2]))


def park(model, data, at: dict) -> None:
  """The dishware PUT AWAY (`park_spec`'s, at run time): each body of `at`
  ({name: (x, y)}) under the floor there, its contacts off, its weight held,
  in the parked group and still."""
  for name, (x, y) in at.items():
    b = model.body(name).id
    _switch(model, b, False)
    _pose(model, data, b, (x, y, PARK_Z))


def parked(model, data, at: dict) -> bool:
  """Whether every body of `at` is put away where `park` puts it: its geoms
  in the parked group and its place within a millimetre of its own."""
  from pluggybot.perception.cull import PARKED_GROUP
  for name, (x, y) in at.items():
    b = model.body(name).id
    if model.geom_group[int(model.body_geomadr[b])] != PARKED_GROUP:
      return False
    if np.linalg.norm(_placed_at(model, data, b) - np.array([x, y, PARK_Z])) > 1e-3:
      return False
  return True


def _placed_at(model, data, body: int) -> np.ndarray:
  """Where a body is placed in `data` -- its free joint's or its mocap's --
  which a forward pass has yet to carry into `xpos`."""
  mocap = int(model.body_mocapid[body])
  if mocap >= 0:
    return np.array(data.mocap_pos[mocap])
  q = int(model.jnt_qposadr[int(model.body_jntadr[body])])
  return np.array(data.qpos[q:q + 3])


# ---- the activity ---------------------------------------------------------------

#: A piece reads LIFTED with its bottom this far over where it rested, m,
#: and down again under half of it. Where it belongs is the grader's
#: (`challenge/table.py`).
LIFT_ON = 0.010


class Dishware(Activity):
  """The pieces' state machine, off their poses (a camera over the table,
  never the robot's senses):

    setOut  LIVE: the dishware out where a job set it, never put away (its
            plate's geoms out of the parked group, `place` / `park`): what
            a page that draws the world hides the pieces by
    lifted  LIVE, each piece: its bottom more than `LIFT_ON` over the
            surface it was set out on, while it is out -- the height it
            rested at when first sensed out, the table's or the pass's
            (#682's stage C), never the table's alone

  ⚠ `load` is the truth (`draw`): nothing of it is a flag."""

  def __init__(self, model, data, load: Load, surface_z: float, name: str = "dishware") -> None:
    super().__init__(name)
    from pluggybot.perception.cull import PARKED_GROUP
    self.bodies = {p: int(model.body(p).id) for p in PIECES}
    self._plate_geom = int(model.body_geomadr[self.bodies["plate"]])
    self._parked = PARKED_GROUP
    #: ...where a save from before `rest` was kept puts them: the table's
    self.surface_z = float(surface_z)
    self.renew(model, load)

  def renew(self, model, load: Load) -> None:
    """Begun again over `load`, the pieces set out afresh: their
    thresholds clear, and where each rests read again once it is out."""
    self.load = load
    self.lifted = {p: Threshold(on=LIFT_ON, off=LIFT_ON / 2) for p in PIECES}
    #: The height each piece rested at, set out: read at the first sense
    #: that finds it out, off where `place` put it (`_placed_at`) -- ⚠ never
    #: `xpos`: an offer sets out in the physics hook, the activities' hook
    #: senses in the same pass, and `xpos` is still the parked pose there.
    self.rest: dict[str, float] | None = None
    self.set(setOut=bool(model.geom_group[self._plate_geom] != self._parked),
             **{f"{p}Lifted": False for p in PIECES})

  def kept_state(self) -> dict:
    return {**super().kept_state(), "lifted": {p: t.value for p, t in self.lifted.items()},
            "rest": None if self.rest is None else dict(self.rest)}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    for p, v in (state.get("lifted") or {}).items():
      if p in self.lifted:
        self.lifted[p].value = bool(v)
    if "rest" in state:
      self.rest = None if state["rest"] is None else {p: float(z) for p, z in state["rest"].items()}
    elif state.get("lifted"):
      # ...a save from before `rest` was kept: its pieces were the table's
      self.rest = {p: self.surface_z for p in PIECES}

  def sense(self, model, data) -> None:
    if model.geom_group[self._plate_geom] == self._parked:
      if self.flags.get("setOut") is not False:
        self.set(setOut=False, **{f"{p}Lifted": False for p in PIECES})
      self.rest = None
      return
    if self.rest is None:
      self.rest = {p: float(_placed_at(model, data, b)[2]) for p, b in self.bodies.items()}
    self.set(setOut=True,
             **{f"{p}Lifted": self.lifted[p].update(float(data.xpos[b][2]) - self.rest[p])
                for p, b in self.bodies.items()})
