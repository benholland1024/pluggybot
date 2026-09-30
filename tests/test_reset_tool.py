"""Guards for the admin tool reset (issue #30, fix 2).

The RECOVERY: a module that is on the floor anyway -- a collision, an
unlucky drop, anything measurement cannot promise away -- is invisible to
the whole swap stack and litters the rack's approach lane, and on hardware
a person would pick it up. `reset_tool` is that hand, reaching in through
the admin page: an inbound message handled by CODE on the physics thread,
never shown to the overseer, that puts the module back on its own bay. On
the quadruped's house, whose rack the tools hang on.
"""

import math

import mujoco

from pluggybot.legs import rack as legs_rack
from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, world_config
from pluggybot.mind.inbox import Inbox
from pluggybot.robot import SECOND, world_spec
from pluggybot.telemetry.protocol import CODE_HANDLED_TYPES, INBOUND_TYPES

MODULE = "module_lcd"


def lifecycle_with_inbox(second_at=None):
  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], second_at=second_at).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return HubLifecycle(model, data, viewer=None, realtime=False, world=QUAD_HOME,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      inbox=Inbox())


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
    life.body.close()


def test_a_tool_in_use_is_not_lost(monkeypatch):
  """Yanking a seated module out of the coupling mid-errand would MAKE the
  mess the reset exists to clean up -- refused, with a narration."""
  life = lifecycle_with_inbox()
  try:
    adr, _ = module_qpos(life)
    before = list(life.data.qpos[adr:adr + 7])
    monkeypatch.setattr(legs_rack, "tool_power", lambda *a, **k: {"powered": True})
    life.inbox.offer({"type": "reset_tool", "id": "a_01", "module": MODULE})
    life._visitor_step()
    assert "refused" in life.log[-1] and "seated on the fork" in life.log[-1]
    assert list(life.data.qpos[adr:adr + 7]) == before
  finally:
    life.body.close()


def test_a_reset_cannot_teleport_things_that_are_not_modules():
  """The admin vocabulary is modules, not arbitrary free bodies -- a reset
  of a tower block (or a garden seed) is refused by name."""
  life = lifecycle_with_inbox()
  try:
    for name in ("block_1", "no_such_module", "module_ghost"):
      life.inbox.offer({"type": "reset_tool", "id": f"a_{name}",
                        "module": name})
      life._visitor_step()
      assert "refused" in life.log[-1], name
  finally:
    life.body.close()


def test_the_code_handled_kinds_are_a_subset_of_the_inbound_vocabulary():
  assert set(CODE_HANDLED_TYPES) <= set(INBOUND_TYPES)
  assert "reset_tool" in CODE_HANDLED_TYPES
  assert "message" not in CODE_HANDLED_TYPES


def test_a_tool_the_other_robot_is_holding_is_not_lost_either(monkeypatch):
  """⚠ EVERY robot's fork, not the one that took the message
  (rooftop-media-2026 #337). A reach-in lands in whichever inbox the
  website addressed -- and until the site named one, always the primary's
  -- so reading the primary's fork alone meant a module the SECOND robot
  was carrying read as lost, and the reset yanked it out of the coupling:
  exactly the mess the refusal exists to prevent."""
  life = lifecycle_with_inbox(second_at=world_config(QUAD_HOME)["start2"][:2])
  try:
    adr, _ = module_qpos(life)
    before = list(life.data.qpos[adr:adr + 7])
    #  Seated on the SECOND robot's fork and nobody else's: the first
    #  robot's own fork reads empty, which is what the old check read.
    monkeypatch.setattr(legs_rack, "tool_power",
                        lambda m, d, name, prefix="": {"powered": prefix == SECOND.prefix})
    life.inbox.offer({"type": "reset_tool", "id": "a_01", "module": MODULE})
    life._visitor_step()
    assert f"seated on {SECOND.root}'s fork" in life.log[-1]
    assert list(life.data.qpos[adr:adr + 7]) == before
  finally:
    life.body.close()
