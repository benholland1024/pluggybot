"""The contact list is read as an array, once (rooftop-media-2026 #296).

MEASURED: the electrical criteria and the two chassis scans walked
`data.contact[i]` -- a pybind struct per contact -- every physics step per
robot, and a py-spy profile of the served pair put 49 % of the physics
thread there against 13 % in `mj_step`. Each reader now answers off
`data.contact.geom`; what is pinned is that the answers are IDENTICAL to
the struct-by-struct loop on a live world with dozens of contacts, and
that a geom id cached by model does not survive a recompile.
"""

import mujoco
import numpy as np

from pluggybot.rack import coupling
from pluggybot.rack.coupling import (
  FORK_POLE_GEOMS, HUB_STATION_YS, contact_pairs, geom_id,
  module_power_state, rack_charge_contact,
)
from pluggybot.rack.swap import HubSwap
from pluggybot.robot import FIRST


def _loop_power(model, data, name, prefix=""):
  """The reader as it was: one struct at a time."""
  out = {}
  for side, plates in FORK_POLE_GEOMS.items():
    try:
      peg = model.geom(f"{name}_peg_{side}").id
      plate_ids = {model.geom(prefix + g).id for g in plates}
    except KeyError:
      out[side] = False
      continue
    hit = False
    for i in range(data.ncon):
      pair = {data.contact[i].geom1, data.contact[i].geom2}
      if peg in pair and plate_ids & pair:
        hit = True
        break
    out[side] = hit
  return {"left": out["l"], "right": out["r"], "powered": out["l"] and out["r"]}


def _loop_charge(model, data, prefix=""):
  pins = {model.geom("rack_pin_l").id, model.geom("rack_pin_r").id}
  chassis = model.geom(prefix + "chassis").id
  seen = set()
  for i in range(data.ncon):
    c = data.contact[i]
    pair = {c.geom1, c.geom2}
    if chassis in pair:
      seen |= pins & pair
  return len(seen) == 2


def _picked_world():
  """The hub world with the LCD module picked onto the fork (the swap's
  own routine, as `test_hub_swap` does it): the electrical criterion true,
  beside dozens of unrelated contacts."""
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data, handle=FIRST)
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap._run(1.0, 0.0)
  swap.pick()
  swap._run(1.0, 0.0)
  return model, data, swap


def test_the_array_readers_agree_with_the_struct_loops_on_a_live_world():
  model, data, swap = _picked_world()
  assert data.ncon >= 8, "a world with real contacts"
  assert contact_pairs(data).shape == (data.ncon, 2)
  for name in ("module_lcd", "module_pen", "no_such_module"):
    assert module_power_state(model, data, name) == _loop_power(model, data, name)
  assert module_power_state(model, data, "module_lcd")["powered"], "the seated one"
  assert rack_charge_contact(model, data) == _loop_charge(model, data)
  # Every contact geom, asked "touching what?", answers as the loop does.
  g = contact_pairs(data)
  for a in np.unique(g):
    others = set(g[g[:, 0] == a, 1]) | set(g[g[:, 1] == a, 0])
    assert coupling.touching(data, int(a), others)
    assert not coupling.touching(data, int(a), [model.ngeom + 1])
  assert not coupling.touching(mujoco.MjData(model), 0, [1])


def test_a_geom_id_is_cached_per_model_object_and_a_new_model_gets_its_own():
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  assert geom_id(model, "chassis") == model.geom("chassis").id
  assert geom_id(model, "not_a_geom") is None
  other = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  assert geom_id(other, "chassis") == other.geom("chassis").id
  # The cache keys on identity and holds the model: a hit for `model` is
  # never handed to `other`, whatever `id()` does.
  assert coupling._GEOM_IDS[(id(model), "chassis")][0] is model
  assert coupling._GEOM_IDS[(id(other), "chassis")][0] is other
