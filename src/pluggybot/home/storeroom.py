"""The storeroom's counter (issue #559, stage 3): the humanoids' first jobs,
the hands' four (docs/Hands.md), set into the house. A counter against the
east wall of the research center's storeroom, the empty room its address
lands in, and on it, from left to right as a robot faces them, the chest,
a cabinet of drawers, a cupboard and a cube with its mat -- each a hidden
mechanism's activity (`activity/`), as the practice's counter carries one
(`evaluation/counter.py`). Over each, on the wall, a 120 mm tag: a ROW the
robot finds the counter by (`mapping.places.Fixture`), never a position
handed over (#419).

Each prop stands on the counter as its practice set it out, square to the
counter's edge: its handle where the middle of its drawn set-outs put it
from a robot standing at its WORK SPOT (`work_spot`), the soles' middle
facing the wall. A robot walks to its own idea of that spot; nothing here
is the robot's.

Put in at load by the humanoids' world (`humanoid/world.py`) and never the
quadrupeds' (their served geometry, and so their trajectories, stay).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.evaluation import counter as ct
from pluggybot.home import world as home
from pluggybot.rack.tags import COUNTER_TAG_IDS, plate_half_extent, AREA_TAG_SIZE

#: The counter: its top's height and depth, the practice's (`evaluation.
#: counter`), its back on the storeroom's east wall.
COUNTER_H, COUNTER_D = ct.COUNTER_H, ct.COUNTER_D
BACK_X = home.LAB_X[1] - home.WALL_HALF_T
EDGE_X = BACK_X - COUNTER_D
#: The props, left to right as a robot faces the wall (north to south), and
#: the tag over each.
PROPS = ("chest", "drawer", "door", "cube")
TAGS = dict(zip(PROPS, COUNTER_TAG_IDS))
#: Each prop's handle this far along the wall, m: a metre and a fifth apart,
#: so a robot at one prop, its arms out, is clear of the next and of a door
#: swung open toward it, about the storeroom's middle.
SPACING_M = 1.2
HANDLE_Y = {p: (home.STORE_Y[0] + home.STORE_Y[1]) / 2 + SPACING_M * (1.5 - i)
            for i, p in enumerate(PROPS)}
#: The counter runs this far past the props at either end, m.
END_M = 0.6
#: The row's tags: centred this high, m, over the counter's tallest prop (the
#: cupboard's 1.02 m), at the navigation camera's look -- "the tags on the
#: walls 0.4-1.2 m up, read from a metre on" (`humanoid.model`).
TAG_Z = 1.15
#: The handle's place from the work spot, (ahead, left) m: the middle of
#: each prop's practice set-outs (`evaluation.counter`): the chest's knob
#: plumb and the drawer's knob alike, the door's lever's tip, the cube.
HANDLE_FROM_SPOT = {
  "chest": (sum(ct.KNOB_AHEAD) / 2, sum(ct.KNOB_LEFT) / 2),
  "drawer": (sum(ct.KNOB_AHEAD) / 2, sum(ct.KNOB_LEFT) / 2),
  "door": (sum(ct.DOOR_AHEAD) / 2, sum(ct.DOOR_LEFT) / 2),
  "cube": (sum(ct.CUBE_AHEAD) / 2, sum(ct.CUBE_LEFT) / 2),
}


def _handle_in(prop: str) -> float:
  """How far behind the counter's edge a prop's handle stands, m: where
  its practice set-out put the prop's front off the counter's edge
  (`EDGE_BACK`; the cube's board `CUBE_EDGE` plus its margin)."""
  from pluggybot.activity import chest as ch, door as dd, drawer as dr
  if prop == "chest":
    return ct.EDGE_BACK + float((ch.HINGE + ch.PIN + ch.KNOB_OFF)[0])
  if prop == "drawer":
    return ct.EDGE_BACK + float(dr.KNOB_AT[0])
  if prop == "door":
    return ct.EDGE_BACK + float(dd.KNOB_AT[0])
  if prop == "cube":
    return ct.CUBE_EDGE + ct.BOARD_MARGIN
  raise KeyError(prop)


def handle(prop: str) -> tuple[float, float]:
  """Where a prop's handle stands in the world (x, y): the chest's knob's
  plumb place, the drawer's knob, the door's lever's tip, the cube."""
  return EDGE_X + _handle_in(prop), HANDLE_Y[prop]


def work_spot(prop: str) -> tuple[float, float, float]:
  """Where the soles' middle stands to work at a prop, facing the wall: its
  handle where its practice set-outs' middle put it (`HANDLE_FROM_SPOT`).
  OUR staging and the reference's measure, never the robot's."""
  hx, hy = handle(prop)
  ahead, left = HANDLE_FROM_SPOT[prop]
  return hx - ahead, hy - left, 0.0


def tag_at(prop: str) -> tuple[float, float, float, float]:
  """The prop's tag on the wall: (x, y, z) of its centre and the way its
  face points (west, into the room)."""
  return BACK_X, HANDLE_Y[prop], TAG_Z, math.pi


def fixture():
  """The row as one fixture (`mapping.places.Fixture`): x along the robot's
  left as it faces the wall, y out of the tags' faces."""
  from pluggybot.mapping.places import Fixture
  mid = HANDLE_Y["drawer"]
  return Fixture(layout={TAGS[p]: (HANDLE_Y[p] - mid, 0.0) for p in PROPS},
                 facing=math.pi / 2)


def _material(spec, tag_id: int) -> None:
  import mujoco
  if spec.texture(f"tagtex{tag_id}") is None:
    spec.add_texture(name=f"tagtex{tag_id}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{tag_id}.png")
    mat = spec.add_material(name=f"tagmat{tag_id}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{tag_id}"


def attach(spec, draws: dict | None = None) -> None:
  """The counter, its props and its row of tags into a world's `spec`: each
  prop's hidden parameters its activity's draw `draws[prop]` (k; 0 where
  none is named) -- what an offer draws afresh into the compiled world
  (`redraw`)."""
  import mujoco
  from pluggybot.activity import chest as ch, cube as cb, door as dd, drawer as dr
  draws = draws or {}
  ys = [HANDLE_Y[p] for p in PROPS]
  y0, y1 = min(ys) - END_M, max(ys) + END_M
  body = spec.worldbody.add_body(name="store_counter",
                                 pos=[EDGE_X + COUNTER_D / 2, (y0 + y1) / 2, 0.0])
  body.add_geom(name="store_counter", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COUNTER_D / 2, (y1 - y0) / 2, COUNTER_H / 2], pos=[0, 0, COUNTER_H / 2],
                rgba=[0.86, 0.85, 0.82, 1])
  hx, hy = handle("chest")
  off = ch.HINGE + ch.PIN + ch.KNOB_OFF
  ch.attach_chest(spec, ch.draw(draws.get("chest", 0)), (hx - off[0], hy - off[1]), 0.0,
                  z=COUNTER_H)
  hx, hy = handle("drawer")
  dr.attach_cabinet(spec, dr.draw(draws.get("drawer", 0)),
                    (hx - dr.KNOB_AT[0], hy - dr.KNOB_AT[1]), 0.0, z=COUNTER_H)
  hx, hy = handle("door")
  dd.attach_cupboard(spec, dd.draw(draws.get("door", 0)),
                     (hx - dd.KNOB_AT[0], hy - dd.KNOB_AT[1]), 0.0, z=COUNTER_H)
  cx, cy = handle("cube")
  mat = (cx + sum(ct.MAT_AHEAD) / 2, cy + sum(ct.MAT_LEFT) / 2)
  x0, x1 = min(cx, mat[0]) - ct.BOARD_MARGIN, max(cx, mat[0]) + ct.BOARD_MARGIN
  b0, b1 = min(cy, mat[1]) - ct.BOARD_MARGIN, max(cy, mat[1]) + ct.BOARD_MARGIN
  board = spec.worldbody.add_body(name="cube_board", pos=[(x0 + x1) / 2, (b0 + b1) / 2,
                                                          COUNTER_H])
  board.add_geom(name="cube_board", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[(x1 - x0) / 2, (b1 - b0) / 2, ct.BOARD_H / 2],
                 pos=[0, 0, ct.BOARD_H / 2], rgba=[0.55, 0.45, 0.36, 1])
  cb.attach_props(spec, cb.draw(draws.get("cube", 0)), (cx, cy), mat, 0.0,
                  z=COUNTER_H + ct.BOARD_H)
  half = plate_half_extent(AREA_TAG_SIZE)
  for prop in PROPS:
    tag = TAGS[prop]
    _material(spec, tag)
    x, y, z, facing = tag_at(prop)
    px = x + 0.002 * math.cos(facing)
    t = spec.worldbody.add_body(name=f"store_tag{tag}", pos=[px, y, z],
                                quat=[math.cos(facing / 2), 0.0, 0.0, math.sin(facing / 2)])
    t.add_geom(name=f"store_tag{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.001, half, half], contype=0, conaffinity=0, material=f"tagmat{tag}")


def counter_box() -> tuple[np.ndarray, np.ndarray]:
  """The counter's footprint, (x0, y0) and (x1, y1), m: what a staging
  keeps a robot off."""
  ys = [HANDLE_Y[p] for p in PROPS]
  return (np.array([EDGE_X, min(ys) - END_M]), np.array([BACK_X, max(ys) + END_M]))


# ---- in the compiled world: the props' activities, drawn and set out -----------------

#: The cube rests on its board, this high, m.
CUBE_SURFACE_Z = COUNTER_H + ct.BOARD_H


def drawn(prop: str, k: int):
  """A prop's hidden parameters at draw `k` (`<its activity's module>.draw`):
  the truth an offer sets into the world, kept as `Task.secret` is -- the
  grade's to read (the chest's catch, secret on its row), never a flag."""
  from pluggybot.activity import chest as ch, cube as cb, door as dd, drawer as dr
  return {"chest": ch.draw, "drawer": dr.draw, "door": dd.draw, "cube": cb.draw}[prop](k)


def activity_for(model, data, prop: str, k: int):
  """A prop's activity over its draw `k` (`<prop's module>.draw(k)`): the
  truth it keeps (never a flag) and its state machine, fresh -- the chest's
  catch held, the door's latch bolted."""
  from pluggybot.activity import chest as ch, cube as cb, door as dd, drawer as dr
  if prop == "chest":
    return ch.Chest(model, data, ch.draw(k))
  if prop == "drawer":
    return dr.Drawer(model, data, dr.draw(k))
  if prop == "door":
    return dd.Door(model, data, dd.draw(k))
  if prop == "cube":
    return cb.Cube(model, data, cb.draw(k), CUBE_SURFACE_Z)
  raise KeyError(prop)


def activities(model, data) -> list:
  """The four props' activities where the world has the counter (`attach`'s
  draws, 0 each), or none."""
  try:
    model.body("store_counter")
  except KeyError:
    return []
  return [activity_for(model, data, p, 0) for p in PROPS]


def _subtree(model, prop: str) -> list[int]:
  root = model.body(prop).id
  return [b for b in range(model.nbody)
          if b == root or _under(model, b, root)]


def _under(model, b: int, root: int) -> bool:
  while b > 0:
    b = int(model.body_parentid[b])
    if b == root:
      return True
  return False


def _drawn_model(prop: str, k: int):
  """The prop alone, compiled over its draw `k` (no tags): where its drawn
  dynamics are read off by name."""
  import mujoco
  from pluggybot.activity import chest as ch, door as dd, drawer as dr
  xml = {"chest": lambda: ch.chest_xml(ch.draw(k), tags=False),
         "drawer": lambda: dr.cabinet_xml(dr.draw(k), tags=False),
         "door": lambda: dd.cupboard_xml(dd.draw(k), tags=False)}[prop]()
  return mujoco.MjModel.from_xml_string(xml)


def redraw(model, data, prop: str, k: int) -> None:
  """A prop's hidden parameters drawn afresh into the compiled world: its
  moving parts' masses and inertias, its joints' damping, friction,
  armature and springs, its geoms' friction -- copied by name off the prop
  compiled alone over draw `k` (the cube's by its own arithmetic, as the
  bench's unknown mass is set, `bench.set_unknown_mass`), the derived
  constants recomputed. ⚠ THE WORLD'S PINNED STATISTIC SURVIVES IT, as the
  bench's must: `mj_setConst` re-derives the extent every camera's near
  plane is scaled by. ⚠ A JOINT'S RANGE IS NOT COPIED: the door's latch
  holds it (`activity.door.Door`)."""
  import mujoco
  if prop == "cube":
    from pluggybot.activity import cube as cb
    load = cb.draw(k)
    bid = model.body("cube").id
    model.body_inertia[bid] *= float(load.cube_kg) / float(model.body_mass[bid])
    model.body_mass[bid] = float(load.cube_kg)
    model.geom_friction[model.geom("cube").id][0] = float(load.friction)
  else:
    alone = _drawn_model(prop, k)
    for b in range(1, alone.nbody):
      bid = model.body(alone.body(b).name).id
      model.body_mass[bid] = alone.body_mass[b]
      model.body_ipos[bid] = alone.body_ipos[b]
      model.body_iquat[bid] = alone.body_iquat[b]
      model.body_inertia[bid] = alone.body_inertia[b]
    for j in range(alone.njnt):
      jid = model.joint(alone.joint(j).name).id
      dof, adof = int(model.jnt_dofadr[jid]), int(alone.jnt_dofadr[j])
      model.dof_damping[dof] = alone.dof_damping[adof]
      model.dof_frictionloss[dof] = alone.dof_frictionloss[adof]
      model.dof_armature[dof] = alone.dof_armature[adof]
      model.jnt_stiffness[jid] = alone.jnt_stiffness[j]
      model.qpos_spring[int(model.jnt_qposadr[jid])] = alone.qpos_spring[int(alone.jnt_qposadr[j])]
    for g in range(alone.ngeom):
      name = alone.geom(g).name
      if name:
        model.geom_friction[model.geom(name).id] = alone.geom_friction[g]
  extent, center = float(model.stat.extent), model.stat.center.copy()
  spring = model.qpos_spring.copy()
  mujoco.mj_setConst(model, mujoco.MjData(model))
  model.stat.extent = extent
  model.stat.center[:] = center
  model.qpos_spring[:] = spring


def set_out(model, data, prop: str) -> None:
  """A prop back as the world compiled it: every joint of it at its rest
  (`qpos0`) and still -- the drawer shut, the door shut with its lever up,
  the lid down, the cube on its board. Its activity is the caller's to
  build afresh after (`activity_for`): the door's latch bolts on a door
  already shut."""
  for b in _subtree(model, prop):
    for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b])):
      q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
      nq = {0: 7, 1: 4}.get(int(model.jnt_type[j]), 1)
      nv = {0: 6, 1: 3}.get(int(model.jnt_type[j]), 1)
      data.qpos[q:q + nq] = model.qpos0[q:q + nq]
      data.qvel[v:v + nv] = 0.0
      data.qfrc_applied[v:v + nv] = 0.0


#: Each prop's activity's truth, by the attribute it keeps it in.
TRUTH = {"chest": "lid", "drawer": "runners", "door": "hinge", "cube": "load"}


def draw_into(act, fresh) -> None:
  """A prop's activity given a fresh draw's truth, its state kept: a prop a
  robot is working, or one a restart brought back as it was."""
  name = TRUTH[act.name]
  setattr(act, name, getattr(fresh, name))


def touched_by(model, data, prop: str, geoms) -> bool:
  """Whether anything of `geoms` (a robot's) touches the prop: one being
  worked is left as it is."""
  from pluggybot.rack.coupling import contact_pairs
  mine = set(_subtree(model, prop))
  own = np.isin(model.geom_bodyid, list(mine))
  g = contact_pairs(data)
  if not g.shape[0]:
    return False
  other = np.isin(np.arange(model.ngeom), list(geoms))
  a, b = g[:, 0], g[:, 1]
  return bool(((own[a] & other[b]) | (own[b] & other[a])).any())
