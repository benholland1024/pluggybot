"""Guards for the admin tool reset (issue #30, fix 2).

Prevention is the measured bay standoff (test_swap_approach.py); this is the
RECOVERY: a module that is on the floor anyway -- a collision, an unlucky
jam, anything measurement cannot promise away -- is invisible to the whole
swap stack and litters the rack's approach lane, and on hardware a person
would pick it up. `reset_tool` is that hand, reaching in through the admin
page: an inbound message handled by CODE on the physics thread, never shown
to the overseer, that puts the module back on its own bay.
"""

import math

import mujoco

from pluggybot import lifecycle as lc
from pluggybot.mind.inbox import Inbox
from pluggybot.lifecycle import HubLifecycle
from pluggybot.telemetry.protocol import CODE_HANDLED_TYPES, INBOUND_TYPES

MODULE = "module_lcd"


def lifecycle_with_inbox():
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  life = HubLifecycle(model, data, viewer=None, realtime=False,
                      errand=False, inbox=Inbox())
  return life


def module_qpos(life):
  jid = int(life.model.body(MODULE).jntadr[0])
  adr = int(life.model.jnt_qposadr[jid])
  return adr, life.data.qpos[adr:adr + 7]


def test_the_inbox_parses_a_reset_and_refuses_one_naming_nothing():
  box = Inbox()
  msg = box.offer({"type": "reset_tool", "id": "a_01", "module": MODULE,
                   "from": "ben"})
  assert msg is not None and msg.kind == "reset_tool"
  assert msg.module == MODULE
  assert msg.as_dict()["module"] == MODULE
  assert box.offer({"type": "reset_tool", "id": "a_02"}) is None


def test_a_reset_puts_a_floored_module_back_on_its_bay():
  life = lifecycle_with_inbox()
  try:
    adr, q = module_qpos(life)
    home_pose = list(life.model.qpos0[adr:adr + 7])
    # knock it to the floor a metre from the rack, spinning
    life.data.qpos[adr:adr + 3] = (1.0, 1.0, 0.03)
    jid = int(life.model.body(MODULE).jntadr[0])
    dadr = int(life.model.jnt_dofadr[jid])
    life.data.qvel[dadr:dadr + 6] = 0.5
    mujoco.mj_forward(life.model, life.data)
    life.inbox.offer({"type": "reset_tool", "id": "a_01", "module": MODULE,
                      "from": "ben"})
    life._visitor_step()
    _, q = module_qpos(life)
    assert all(math.isclose(a, b, abs_tol=1e-9)
               for a, b in zip(q, home_pose)), \
        "the reset pose is the model's own qpos0 -- hung at its bay"
    assert all(v == 0.0 for v in life.data.qvel[dadr:dadr + 6])
    assert "back on its bay" in life.log[-1]
  finally:
    life.mission.close()


def test_a_tool_in_use_is_not_lost(monkeypatch):
  """Yanking a seated module out of the coupling mid-errand would MAKE the
  mess the reset exists to clean up -- refused, with a narration."""
  life = lifecycle_with_inbox()
  try:
    adr, _ = module_qpos(life)
    before = list(life.data.qpos[adr:adr + 7])
    monkeypatch.setattr(lc, "module_power_contact", lambda *a, **k: True)
    life.inbox.offer({"type": "reset_tool", "id": "a_01", "module": MODULE})
    life._visitor_step()
    assert "refused" in life.log[-1] and "seated on the fork" in life.log[-1]
    assert list(life.data.qpos[adr:adr + 7]) == before
  finally:
    life.mission.close()


def test_a_reset_cannot_teleport_things_that_are_not_modules():
  """The admin vocabulary is modules, not arbitrary free bodies -- a reset
  of the claw's practice block (or a garden seed) is refused by name."""
  life = lifecycle_with_inbox()
  try:
    for name in ("chassis", "no_such_module", "module_ghost"):
      life.inbox.offer({"type": "reset_tool", "id": f"a_{name}",
                        "module": name})
      life._visitor_step()
      assert "refused" in life.log[-1], name
  finally:
    life.mission.close()


def test_the_code_handled_kinds_are_a_subset_of_the_inbound_vocabulary():
  assert set(CODE_HANDLED_TYPES) <= set(INBOUND_TYPES)
  assert "reset_tool" in CODE_HANDLED_TYPES
  assert "message" not in CODE_HANDLED_TYPES
