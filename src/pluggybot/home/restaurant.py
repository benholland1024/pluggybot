"""The restaurant's table (issue #560, the table set served): the place a job
lays at the dining room's table 0 (`home.world.SET_TABLE`), its south side,
and its dishware -- a plate, a cup, a fork and the place card
(`activity/dishware.py`) -- compiled into the humanoids' house at load, as
the storeroom's counter is (`home.storeroom`), and PUT AWAY until an offer
sets it out: every free body costs the physics and every geom a ray, and a
world is compiled once.

The table and the place are the offline set-out's (`evaluation.counter.
table_set_out`): a job's draw k sets the pieces out where set-out k does,
each drawn off its own stream, their hidden weights and friction
`dishware.draw(k)`'s -- so the practice's and the job's place are one.
Nothing here is the robot's: it finds the place by the card's tag (93), and
the pieces by theirs (90-92).

The storeroom's members, so the lifecycle sets either out, puts it back
after a restart and keeps an offer off it while a robot touches it
(`home.jobs`); and one the counter has no need of: `put_away`, the dishware
parked again once no job holds the table.
"""

from __future__ import annotations

import numpy as np

from pluggybot.activity import dishware as dw
from pluggybot.evaluation import counter as ct
from pluggybot.home import world as home

#: The table the place is laid at: the job's target, its own body in the
#: house (so a page's marker for the job stands over it), and a `table`.
TABLE = f"dining_table_{home.SET_TABLE}"
TARGETS = {TABLE: "table"}
#: The dishware's bodies: the pieces and the card.
BODIES = (*dw.PIECES, "place_card")
#: The pieces rest on the table's top, m.
SURFACE_Z = ct.TABLE_H


def _middles() -> dict:
  """Where each body waits put away, (x, y): under where it is set out --
  a piece's set-outs' middle, the card's place."""
  out = {p: ct.place_to_world(*ct.TABLE_SOURCES[p][0]) for p in dw.PIECES}
  out["place_card"] = ct.place_to_world(ct.CARD_IN, 0.0)
  return out


PARK_AT = _middles()


def attach(spec, draw: int = 0) -> None:
  """The dishware into a world's `spec`, put away (`dishware.park_spec`):
  the pieces at draw `draw`'s weights and friction -- what an offer draws
  afresh into the compiled world (`redraw`) -- the card a mocap body the
  job moves."""
  so = ct.table_set_out(draw)
  dw.attach_card(spec, so.card[:2], so.card[2], SURFACE_Z, mocap=True)
  for name, (x, y, yaw) in so.pieces.items():
    dw.ATTACH[name](spec, (x, y), yaw, SURFACE_Z, getattr(so.load, f"{name}_kg"),
                    so.load.friction)
  dw.park_spec(spec, BODIES, PARK_AT)


def present(model) -> bool:
  """Whether this world has the place's dishware, put away to be set out
  (the set-out's own world has its pieces fixed in place, no card to move)."""
  try:
    return int(model.body("place_card").mocapid[0]) >= 0
  except KeyError:
    return False


def drawn(target: str, k: int) -> dw.Load:
  """The place's hidden parameters at draw `k` (`dishware.draw`): the truth
  an offer sets into the world, kept as `Task.secret` is, never a flag."""
  _ours(target)
  return dw.draw(k)


def _ours(target: str) -> None:
  if target not in TARGETS:
    raise KeyError(target)


def redraw(model, data, target: str, k: int) -> None:
  """The pieces' hidden weights and friction at draw `k` set into the
  compiled world -- each piece's mass and its inertia (its shape's, scaled),
  every solid geom of it the drawn friction (a tag's film meets nothing) --
  the derived constants recomputed,
  the world's pinned statistic kept (`storeroom.redraw`'s trap)."""
  import mujoco
  _ours(target)
  load = dw.draw(k)
  for p in dw.PIECES:
    b = model.body(p).id
    kg = float(getattr(load, f"{p}_kg"))
    model.body_inertia[b] *= kg / float(model.body_mass[b])
    model.body_mass[b] = kg
    for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b] + model.body_geomnum[b])):
      if not dw._is_film(model, g):
        model.geom_friction[g][0] = float(load.friction)
  extent, center = float(model.stat.extent), model.stat.center.copy()
  spring = model.qpos_spring.copy()
  mujoco.mj_setConst(model, mujoco.MjData(model))
  model.stat.extent = extent
  model.stat.center[:] = center
  model.qpos_spring[:] = spring


#: Each kind's set-out of the place: its pieces beside it on the table, or on
#: the pass (#682's stage C, easy mode; `counter.pass_set_out`), the card at
#: the table either way.
SET_OUTS = {"set_table": ct.table_set_out, "set_table_pass": ct.pass_set_out}


def set_out(model, data, target: str, k: int, kind: str = "set_table") -> None:
  """The place set out for an offer of `kind` drawing `k`: every piece where
  that kind's set-out k puts it (`SET_OUTS`) and the card at its place, live
  and still, their hidden parameters drawn (`redraw`)."""
  _ours(target)
  so = SET_OUTS[kind](k)
  redraw(model, data, target, k)
  dw.place(model, data, so.pieces, so.card, SURFACE_Z, piece_z=so.surface)


def put_away(model, data, target: str) -> None:
  """The dishware parked again: no job holds the table."""
  _ours(target)
  dw.park(model, data, PARK_AT)


def away(model, data, target: str) -> bool:
  """Whether the dishware is put away where `put_away` puts it."""
  _ours(target)
  return dw.parked(model, data, PARK_AT)


def activity_for(model, data, target: str, k: int):
  """The place's activity over draw `k`, named by its table: its pieces'
  state machine (`dishware.Dishware`)."""
  _ours(target)
  return dw.Dishware(model, data, dw.draw(k), SURFACE_Z, name=target)


def activities(model, data) -> list:
  """The place's activity where the world has its dishware, or none."""
  return [activity_for(model, data, TABLE, 0)] if present(model) else []


def touched_by(model, data, target: str, geoms) -> bool:
  """Whether anything of `geoms` (a robot's) touches a piece or the card:
  dishware a hand is on is no target for an offer, which would set it out
  from under the hand, and is not put away from under it."""
  from pluggybot.contacts import contact_pairs
  _ours(target)
  own = np.isin(model.geom_bodyid, [model.body(n).id for n in BODIES])
  g = contact_pairs(data)
  if not g.shape[0]:
    return False
  other = np.isin(np.arange(model.ngeom), list(geoms))
  a, b = g[:, 0], g[:, 1]
  return bool(((own[a] & other[b]) | (own[b] & other[a])).any())
