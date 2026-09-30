"""The contact list is read as an array, once (rooftop-media-2026 #296).

MEASURED: the electrical criteria and the body's contact scans walked
`data.contact[i]` -- a pybind struct per contact -- every physics step per
robot, and a py-spy profile of the served pair put 49 % of the physics
thread there against 13 % in `mj_step`. Each reader now answers off
`data.contact.geom`; what is pinned is that the answers are IDENTICAL to
the struct-by-struct loop on a live world with dozens of contacts, and
that a geom id cached by model does not survive a recompile.
"""

import math

import mujoco
import numpy as np

from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.legs import world as lw
from pluggybot.legs.model import CHOSEN
from pluggybot.rack import coupling
from pluggybot.rack.coupling import (PEG_ABOVE_BODY, bay_prefix, bay_switches,
                                     contact_pairs, geom_id)


def _pairs_loop(data):
  return [{data.contact[i].geom1, data.contact[i].geom2} for i in range(data.ncon)]


def _loop_power(model, data, name, prefix=""):
  """The fork's criterion as it was read: one struct at a time."""
  out = {}
  for side, plates in rk.FORK_POLES.items():
    try:
      peg = model.geom(f"{name}_peg_{side}").id
      plate_ids = {model.geom(prefix + g).id for g in plates}
    except KeyError:
      out[side] = False
      continue
    out[side] = any(peg in pair and plate_ids & pair for pair in _pairs_loop(data))
  return {"left": out["l"], "right": out["r"], "powered": out["l"] and out["r"]}


def _loop_hung(model, data, name, bay):
  """`rk.on_bay`'s contact half, struct by struct: a peg on every flank."""
  pegs = [model.geom(f"{name}_peg_{s}").id for s in ("l", "r")]
  flanks = [model.geom(f"{bay_prefix(bay)}tray_{lbl}_{ab}").id
            for lbl in ("l", "r") for ab in ("a", "b")]
  pairs = _pairs_loop(data)
  return all(any(f in pair and set(pegs) & pair for pair in pairs) for f in flanks)


def _loop_switches(model, data):
  out = []
  for i in range(len(coupling.STATION_YS)):
    ids = [geom_id(model, bay_prefix(i) + p) for p in coupling.BAY_SWITCH_PLATES]
    if None in ids:
      out.append(None)
      continue
    out.append(any(set(ids) & pair for pair in _pairs_loop(data)))
  return tuple(out)


def _seated_world():
  """The house with the quadruped standing in the living room, the claw
  dropped onto its fork at the carry pose (as a pick leaves it) and the
  lcd and the pen hung on the rack: a tenth of a second on, the claw
  conducts, beside dozens of unrelated contacts."""
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, "", 1.5, 0.5, 0.0)
  for joint, q in zip(("shoulder", "elbow", "wrist"), (*am.CARRY_Q, -sum(am.CARRY_Q))):
    data.qpos[model.jnt_qposadr[model.joint(f"arm_{joint}").id]] = q
  mujoco.mj_forward(model, data)
  root = model.body("pluggybot").id
  rot = data.xmat[root].reshape(3, 3)
  peg = data.site_xpos[model.site("arm_seat").id] + rot @ np.array(
    [0.0, 0.0, CHOSEN.arm.fork.seat_rise() + 0.0003])
  yaw = math.atan2(rot[1, 0], rot[0, 0]) + math.pi
  q = model.jnt_qposadr[model.body("module_claw").jntadr[0]]
  data.qpos[q:q + 3] = peg - np.array([0.0, 0.0, PEG_ABOVE_BODY])
  data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(model, data)
  for _ in range(50):
    mujoco.mj_step(model, data)
  return model, data


def test_the_array_readers_agree_with_the_struct_loops_on_a_live_world():
  model, data = _seated_world()
  assert data.ncon >= 20, "a world with real contacts"
  assert contact_pairs(data).shape == (data.ncon, 2)
  for name in ("module_claw", "module_lcd", "no_such_module"):
    assert rk.tool_power(model, data, name) == _loop_power(model, data, name)
  assert rk.tool_power(model, data, "module_claw")["powered"], "the seated one"
  for name, bay in rk.TOOL_BAYS.items():
    assert rk.on_bay(model, data, name, rk.DEFAULT, bay) == _loop_hung(model, data, name, bay)
  assert [rk.on_bay(model, data, n, rk.DEFAULT, b) for n, b in rk.TOOL_BAYS.items()] \
    == [True, True, False], "two hung, the claw's bay empty"
  assert bay_switches(model, data) == _loop_switches(model, data)
  # Every contact geom, asked "touching what?", answers as the loop does.
  g = contact_pairs(data)
  for a in np.unique(g):
    others = set(g[g[:, 0] == a, 1]) | set(g[g[:, 1] == a, 0])
    assert coupling.touching(data, int(a), others)
    assert not coupling.touching(data, int(a), [model.ngeom + 1])
    # ...and not what is in contact with something else only.
    elsewhere = set(g.ravel()) - others - {a}
    assert not coupling.touching(data, int(a), elsewhere)
  assert not coupling.touching(mujoco.MjData(model), 0, [1])


def test_a_geom_id_is_cached_per_model_object_and_a_new_model_gets_its_own():
  spec = lw.home_spec()
  model = spec.compile()
  assert geom_id(model, "belly_pad_l") == model.geom("belly_pad_l").id
  assert geom_id(model, "not_a_geom") is None
  other = spec.compile()
  assert geom_id(other, "belly_pad_l") == other.geom("belly_pad_l").id
  # The cache keys on identity and holds the model: a hit for `model` is
  # never handed to `other`, whatever `id()` does.
  assert coupling._GEOM_IDS[(id(model), "belly_pad_l")][0] is model
  assert coupling._GEOM_IDS[(id(other), "belly_pad_l")][0] is other
