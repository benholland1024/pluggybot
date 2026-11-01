"""The contact list is read as an array, once (rooftop-media-2026 #296).

MEASURED: the electrical criteria and the body's contact scans walked
`data.contact[i]` -- a pybind struct per contact -- every physics step per
robot, and a py-spy profile of the served pair put 49 % of the physics
thread there against 13 % in `mj_step`. A per-step reader answers off
`data.contact.geom` (`contacts.contact_pairs`, the hand jobs' graders' and
the cage's); what is pinned is that its answer is IDENTICAL to the
struct-by-struct loop on a live world with dozens of contacts.
"""

import mujoco

from pluggybot.contacts import contact_pairs
from pluggybot.humanoid import world as hw


def _pairs_loop(data):
  return [(int(data.contact[i].geom1), int(data.contact[i].geom2))
          for i in range(data.ncon)]


def _live_world():
  """The house with the humanoid in the living room a tenth of a second
  on: its soles on the floor, beside dozens of unrelated contacts -- the
  props on their floors and the counter's on their counter."""
  model = hw.home_spec().compile()
  data = mujoco.MjData(model)
  for _ in range(50):
    mujoco.mj_step(model, data)
  return model, data


def test_the_array_reader_agrees_with_the_struct_loop_on_a_live_world():
  model, data = _live_world()
  assert data.ncon >= 20, "a world with real contacts"
  g = contact_pairs(data)
  assert g.shape == (data.ncon, 2)
  assert [(int(a), int(b)) for a, b in g] == _pairs_loop(data)
  soles = {model.geom(f"{side}_sole").id for side in ("left", "right")}
  assert soles <= set(g.ravel().tolist()), "the robot's own contacts among them"
  # ...and a world with none answers none, never a stale row
  assert contact_pairs(mujoco.MjData(model)).shape == (0, 2)
