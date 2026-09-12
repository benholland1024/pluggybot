"""Death, the survival clock, and the admin's reset (issue #107).

`reset_tool`'s shape exactly (tests/test_reset_tool.py): an inbound kind,
code-handled on the physics thread, never shown to the overseer, refused
while a module is seated on the fork. What is new is that the robot can now
DIE -- `flat` when the pack reaches zero, `stuck` when it is knocked over
or cannot reach the rack -- and that dying is recorded three ways at once:
a `death` event on the wire, a line in the one file the robot cannot edit,
and a survival clock the next decision is shown.
"""

import math

import mujoco
import pytest

from pluggybot import tick
from pluggybot import lifecycle as lc
from pluggybot.lifecycle import HubLifecycle, world_config, zone_centre
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.thoughts import HISTORY
from pluggybot.telemetry.protocol import (
  CODE_HANDLED_TYPES, DEATH_CAUSES, INBOUND_TYPES, PROTOCOL_VERSION,
)


def _life(world: str = "room_hub", inbox=None, mortal=True,
          **kw) -> HubLifecycle:
  """A lifecycle that can die. ⚠ `mortal=True` is explicit here and the
  DEFAULT IS OFF (it follows the inbox): on a demo cell the pack reaches
  zero mid-errand as documented behaviour, so mortality is opt-in exactly
  as job offers and hunger are, and every recording and mission test reads
  as it did."""
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world=world,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      inbox=inbox, mortal=mortal, **kw)
  life.mission.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  life.survival_since = float(data.time)
  return life


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


def _topple(life) -> None:
  """Lay the robot on its side where it stands."""
  d = life.data
  d.qpos[2] += 0.15
  d.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
  d.qvel[:6] = 0.0
  mujoco.mj_forward(life.model, d)


# ---- the vocabulary ------------------------------------------------------


def test_the_reset_is_an_admin_kind_code_handles_and_the_wire_bumped():
  assert "reset_robot" in INBOUND_TYPES
  assert "reset_robot" in CODE_HANDLED_TYPES
  assert set(CODE_HANDLED_TYPES) <= set(INBOUND_TYPES)
  # ...and `unpaid` since issue #136: upkeep came due and could not be paid.
  # ...and `unminded` since issue #127: the agent's own event map stopped
  # consulting its mind. FOUR causes, never summed -- a decision failure, a
  # physics failure, an economic one, and a robot that configured itself out
  # of ever being asked anything.
  assert DEATH_CAUSES == ("flat", "stuck", "unpaid", "unminded")
  assert PROTOCOL_VERSION == "0.19.0"
  box = Inbox()
  msg = box.offer({"type": "reset_robot", "id": "rr_01", "from": "ben"})
  assert msg is not None and msg.kind == "reset_robot" and msg.who == "ben"


# ---- dying ---------------------------------------------------------------


def test_a_pack_reaching_zero_is_a_flat_death_recorded_three_ways():
  """The moment is caught on the physics seam, not between errands: the
  event carries the clock, History carries the line, and the frame's
  robot record says `dead: flat`."""
  life = _life()
  seen = _events(life)
  life.mission._drive(1.0, 0.0, 0.0)
  assert life.dead is None and life.survival_s == pytest.approx(1.0, abs=0.05)
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is not None and life.dead["cause"] == "flat"
  assert life.deaths == [life.dead]
  assert seen[-1]["type"] == "death" and seen[-1]["cause"] == "flat"
  assert seen[-1]["survivalS"] == pytest.approx(1.0, abs=0.15)
  assert life.telemetry_status()["survival"]["dead"] == "flat"
  assert any("died" in line and "flat" in line
             for line in life.thoughts.volatile()[HISTORY])
  # ...and idempotent: a body that also topples afterwards is still one flat
  _topple(life)
  life.mission._drive(3.0, 0.0, 0.0)
  assert len(life.deaths) == 1 and life.dead["cause"] == "flat"


def test_a_robot_on_its_side_is_a_stuck_death_and_a_wobble_is_not():
  life = _life()
  seen = _events(life)
  # a brief lean recovers: nothing recorded
  d = life.data
  d.qpos[3:7] = [math.cos(0.3), math.sin(0.3), 0.0, 0.0]
  mujoco.mj_forward(life.model, d)
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is None
  _topple(life)
  life.mission._drive(lc.TOPPLE_HOLD_S / 2, 0.0, 0.0)
  assert life.dead is None, "held for less than TOPPLE_HOLD_S: not yet"
  life.mission._drive(lc.TOPPLE_HOLD_S, 0.0, 0.0)
  assert life.dead is not None and life.dead["cause"] == "stuck"
  assert "knocked over" in life.dead["why"]
  assert seen[-1]["type"] == "death" and seen[-1]["cause"] == "stuck"


def test_a_failed_dock_is_a_stuck_death_and_ends_a_day_nobody_can_reset(
    monkeypatch):
  """`stranded` used to end the day as a sentence of its own; it is §3's
  third `stuck` cause, and with no inbox -- nobody to reset it -- the day
  still ends, as it always did."""
  life = _life()
  life.battery.energy_wh = life.low_battery_wh * 0.5
  monkeypatch.setattr(life, "go_charge_routine", lambda: tick.result(False))
  r = life.run(world_config("room_hub")["start"], max_sim_time=30.0,
               explore_budget=5.0)
  assert r["stranded"] and r["dead"] == "stuck"
  assert r["deaths"][0]["cause"] == "stuck" and "charger" in r["deaths"][0]["why"]
  assert r["state"] == "DONE" and r["sim_time"] < 30.0


# ---- the reset -----------------------------------------------------------


def test_a_reset_rights_a_robot_on_its_side_in_the_garden():
  """The acceptance line: on its side in the home world's garden, dead,
  then an admin's `reset_robot` -- back at the start pose, upright, pack
  full, clock restarted, and both the event and History say so."""
  life = _life("home", inbox=Inbox())
  seen = _events(life)
  gx, gy = zone_centre("home", "garden")
  life.mission.start_at(gx, gy, 0.0)
  life.mission._drive(2.0, 0.0, 0.0)
  _topple(life)
  life.mission._drive(lc.TOPPLE_HOLD_S + 0.5, 0.0, 0.0)
  assert life.dead is not None and life.dead["cause"] == "stuck"
  life.battery.energy_wh = 0.3 * life.battery.capacity_wh
  life.inbox.offer({"type": "reset_robot", "id": "rr_01", "from": "ben"})
  life._visitor_step()
  assert life.dead is None and not life.stranded
  assert life.battery.fraction == pytest.approx(1.0)
  assert life._chassis_tilt() < math.radians(2.0), "still on its side"
  x, y, _ = life.home_pose
  assert math.hypot(life.data.qpos[0] - x, life.data.qpos[1] - y) < 0.15
  assert life.survival_s < 0.01, "the survival clock restarts at a reset"
  assert len(life.deaths) == 1 and len(life.resets) == 1
  reset = seen[-1]
  assert reset["type"] == "reset" and reset["by"] == "ben"
  assert reset["wasDead"] == "stuck" and reset["intervention"] is False
  assert reset["deadS"] > 0
  history = life.thoughts.volatile()[HISTORY]
  assert any("reset by ben" in line for line in history)
  assert life.telemetry_status()["survival"] == {"s": pytest.approx(0.0, abs=0.01),
                                                 "deaths": 1, "dead": None}


def test_a_reset_is_refused_mid_swap(monkeypatch):
  """A module seated on the fork is a tool in use; warping the robot out
  from under it would MAKE the mess `reset_tool` exists to clean up."""
  life = _life(inbox=Inbox())
  monkeypatch.setattr(lc, "module_power_contact", lambda *a, **k: True)
  life.mission._drive(0.2, 0.0, 0.0)
  assert life.tool_powered, "the seam did not see the seated module"
  pose = life.data.qpos[:3].copy()
  life.inbox.offer({"type": "reset_robot", "id": "rr_02", "from": "ben"})
  life._visitor_step()
  assert "refused" in life.status and "seated on the fork" in life.status
  assert list(life.data.qpos[:3]) == pytest.approx(list(pose))
  assert life.resets == []


def test_a_reset_of_a_living_robot_is_marked_as_an_intervention():
  life = _life(inbox=Inbox())
  seen = _events(life)
  life.battery.energy_wh *= 0.5
  life.inbox.offer({"type": "reset_robot", "id": "rr_03", "from": "ben"})
  life._visitor_step()
  #  Found by TYPE rather than by position: since issue #119 a reset of a
  #  living robot is followed by an `intervention` event, and an assertion
  #  on the last message would be about the emit order rather than about
  #  the reset.
  reset = next(e for e in seen if e["type"] == "reset")
  assert reset["intervention"] is True
  assert reset["wasDead"] is None and life.battery.fraction == pytest.approx(1.0)
  #  ...and that second event is what a rollup counts (issue #119).
  assert [e["what"] for e in seen if e["type"] == "intervention"] \
    == ["reset_robot"]
  assert life.interventions and life.interventions[0]["by"] == "ben"


def test_a_dead_robot_with_an_inbox_waits_and_a_reset_resumes_the_day():
  """End to end through `run()`: dead on the first step, the loop stands in
  DEAD streaming frames rather than ending, the admin's reset arrives on
  the inbox, and the day carries on."""
  life = _life(inbox=Inbox())
  life.battery.energy_wh = 0.0
  states: list[str] = []
  sent = [False]

  def hook():
    if life.state != (states[-1] if states else None):
      states.append(life.state)
    if life.dead is not None and life.data.time > 8.0 and not sent[0]:
      sent[0] = True
      life.inbox.offer({"type": "reset_robot", "id": "rr_04", "from": "ben"})

  life.mission.step_hooks.append(hook)
  life.stop_when(lambda: bool(life.resets) and life.state not in ("DEAD",))
  r = life.run(world_config("room_hub")["start"], max_sim_time=60.0,
               explore_budget=5.0)
  assert "DEAD" in states, states
  assert r["deaths"][0]["cause"] == "flat" and len(r["resets"]) == 1
  assert r["dead"] is None and r["battery"] > 0.9


# ---- what the mind is shown --------------------------------------------


def test_the_next_decision_is_shown_the_death_and_the_clock():
  life = _life()
  life.mission._drive(1.5, 0.0, 0.0)
  life.battery.energy_wh = 0.0
  life.mission._drive(0.3, 0.0, 0.0)
  ctx = ov.context_for(life, thoughts=life.thoughts)
  assert ctx["survival"]["deaths"] == 1
  assert ctx["survival"]["aliveS"] == pytest.approx(1.8, abs=0.2)
  assert any("died" in line for line in ctx["thoughts"][HISTORY])


def test_the_robot_is_told_it_can_die_only_where_it_can():
  """⚠ A RULE THE ARM CONTRADICTS IS A FALSE STATEMENT THE MODEL ACTS ON.

  That is the lesson M14 drew from the charging rule (Evaluation.md §2):
  the shipped prompt says "charging is not your decision", which the
  `autonomous` arm makes untrue, so the arm has to correct the prompt in
  the same change. Death has the same shape one rule over -- mortality is
  opt-in, so a world whose robot cannot die must not be told that it can.
  """
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.economy.scoring import default_table

  def rules(mortal: bool) -> str:
    return ov.system_prompt(ThoughtFiles(), Menu.for_world("home", None),
                            default_table(), mortal=mortal)[0]["text"]

  assert "YOU CAN DIE" in rules(True)
  assert "survival.aliveS" in rules(True)
  assert "YOU CAN DIE" not in rules(False)
  # ...and the flag ADDS a block and changes nothing else, so an immortal
  # world's cached prefix is what it was before issue #107 -- which is what
  # keeps every existing run's prompt cache warm.
  assert rules(True).replace(ov.MORTAL_RULE, "").strip() == rules(False).strip()


def test_mortality_is_opt_in_and_an_immortal_day_ends_as_it_always_did():
  """⚠ THE DEFAULT IS OFF, and not out of caution. On a demo cell the pack
  reaches ZERO mid-errand as documented behaviour and the robot then limps
  to the rack and carries on -- the committed home recording finishes a
  census at frac 0.000. Made mortal by default, room_hub's own recording
  died at t=184 and ended with the pack back at 87 %, which is a fixture
  describing a robot that is not there.

  So: an immortal lifecycle records no death, and its loop ends on an empty
  pack exactly where the pre-#107 one did.
  """
  life = _life(mortal=False)
  assert life.mortal is False
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is None and life.deaths == []
  r = life.run(world_config("room_hub")["start"], max_sim_time=30.0,
               explore_budget=5.0)
  assert r["dead"] is None and r["deaths"] == []
  assert "BATTERY DEAD" in life.log[-1]
  # ...and the default follows the inbox: a served world can be reset, a
  # test, a spike and a filmstrip cannot.
  assert _life(mortal=None).mortal is False
  assert _life(inbox=Inbox(), mortal=None).mortal is True
