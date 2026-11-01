"""Death, the survival clock, and the admin's reset (issue #107).

An inbound kind, code-handled on the physics thread, never shown to the
overseer. What is new is that the robot can DIE -- `flat` when the pack
reaches zero, `stuck` when it is knocked over or cannot reach its charger
-- and that dying is recorded three ways at once: a `death` event on the
wire, a line in the one file the robot cannot edit, and a survival clock
the next decision is shown.

The bookkeeping is pinned on the stub body (`tests/test_body.py`); what is
the body's -- its map while it lies on its side, where a reset puts it --
on the served humanoid.
"""

import math

import mujoco
import pytest

from pluggybot import tick
from pluggybot.body import StubBody
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, world_config, zone_centre
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.overseer import Menu
from pluggybot.mind.thoughts import HISTORY
from pluggybot.robot import world_spec
from pluggybot.telemetry.protocol import (
  CODE_HANDLED_TYPES, DEATH_CAUSES, INBOUND_TYPES, PROTOCOL_VERSION,
)
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _born(life) -> HubLifecycle:
  start = world_config(HUMANOID_HOME)["start"]
  life.body.start_at(*start)
  life.home_pose = tuple(start)
  life.survival_since = float(life.data.time)
  return life


def _life(inbox=None, mortal=True, **kw) -> HubLifecycle:
  """A lifecycle that can die, on the stub. ⚠ `mortal=True` is explicit
  here and the DEFAULT IS OFF (it follows the inbox): mortality is opt-in
  exactly as job offers and hunger are."""
  return _born(stub_life(inbox=inbox, mortal=mortal, **kw))


def _humanoid(inbox=None, mortal=True, **kw) -> HubLifecycle:
  """...and on the served humanoid, in its house."""
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  return _born(HubLifecycle(model, mujoco.MjData(model), realtime=False,
                            world=HUMANOID_HOME,
                            battery_wh=cfg["battery_wh"],
                            grid_bounds=cfg["grid_bounds"],
                            low_battery_wh=cfg["low_battery_wh"],
                            inbox=inbox, mortal=mortal, **kw))


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


def _topple(life) -> None:
  """The stub's IMU reads it on its side."""
  life.body.attitude = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


def _topple_humanoid(life) -> None:
  """Lay the humanoid on its side where it stands."""
  d = life.data
  q = life.body.handle.qpos_adr(life.model)
  d.qpos[q + 2] = 0.2
  d.qpos[q + 3:q + 7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
  d.qvel[:] = 0.0
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
  assert PROTOCOL_VERSION == "0.21.0"
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
  _hold(life, 1.0)
  assert life.dead is None and life.survival_s == pytest.approx(1.0, abs=0.05)
  life.battery.energy_wh = 0.0
  _hold(life, 0.5)
  assert life.dead is not None and life.dead["cause"] == "flat"
  assert life.deaths == [life.dead]
  assert seen[-1]["type"] == "death" and seen[-1]["cause"] == "flat"
  assert seen[-1]["survivalS"] == pytest.approx(1.0, abs=0.15)
  assert life.telemetry_status()["survival"]["dead"] == "flat"
  assert any("died" in line and "flat" in line
             for line in life.thoughts.volatile()[HISTORY])
  # ...and idempotent: a body that also topples afterwards is still one flat
  _topple(life)
  _hold(life, 3.0)
  assert len(life.deaths) == 1 and life.dead["cause"] == "flat"


def test_a_robot_on_its_side_is_a_stuck_death_and_a_wobble_is_not():
  life = _life()
  seen = _events(life)
  # a lean short of the topple is no fall, however long it is held
  life.body.attitude = (math.cos(0.3), math.sin(0.3), 0.0, 0.0)
  _hold(life, life.body.stuck_after_s + 0.5)
  assert life.dead is None
  _topple(life)
  _hold(life, life.body.stuck_after_s / 2)
  assert life.dead is None, "held for less than stuck_after_s: not yet"
  _hold(life, life.body.stuck_after_s)
  assert life.dead is not None and life.dead["cause"] == "stuck"
  assert "knocked over" in life.dead["why"]
  assert seen[-1]["type"] == "death" and seen[-1]["cause"] == "stuck"


def test_a_robot_on_its_side_writes_nothing_into_its_map():
  """Issue #339, off the deployed world: a robot lay on its side scanning,
  half its rays saw the sky, and they went into the map as free space 8 m
  through the walls -- a map its every later life inherited. The height
  map had the depth camera's frames the same way. The grid takes the
  LIDAR's points laid flat only while the trunk is on its feet in the
  posture its legs hold (`HumanoidMission.maps`, #581), the height map the
  camera's only while level, each as its own IMU reads it; standing again,
  each takes its sensor again. The scan matcher is off: on, it refuses a
  toppled scan anyway (#386), and the rule would be pinned by nothing."""
  life = _humanoid(near_field=True)
  life.body.mission.matcher = None
  grid = life.body.grid
  folds = []
  fold = life.near_field.update
  life.near_field.update = lambda pose, points: (folds.append(1), fold(pose, points))
  _hold(life, 0.3)
  assert grid.grid.any() and folds, "the premise: standing, both are written"
  _topple_humanoid(life)
  # ...laid down between two steps, which its gyro never sees: its IMU's
  # filter reads the tilt off its accelerometer as the body drops, past the
  # camera's level in 0.15 s and past the map's tip in 0.4 (a fall turns
  # it, and the gyro reads that as it turns)
  _hold(life, 0.5)
  assert not life.body.level() and not life.body.mission.maps()
  before, seen = grid.grid.copy(), len(folds)
  _hold(life, 1.0)                                   # ten scans on its side
  assert not life.body.level() and not life.body.mission.maps()
  assert (grid.grid == before).all(), "a scan taken on its side went into the map"
  assert len(folds) == seen, "a toppled camera's frames went into the height map"
  life.body.start_at(*world_config(HUMANOID_HOME)["start"])
  _hold(life, 0.3)
  assert (grid.grid != before).any(), "standing again, the map takes scans"
  assert len(folds) > seen


def test_a_failed_dock_is_a_stuck_death_and_ends_a_day_nobody_can_reset(
    monkeypatch):
  """A robot that cannot reach its charger is `stuck`, and with no inbox
  -- nobody to reset it -- the day ends."""
  life = _life()
  life.battery.energy_wh = life.low_battery_wh * 0.5
  monkeypatch.setattr(life, "go_charge_routine", lambda: tick.result(False))
  r = life.run(world_config(HUMANOID_HOME)["start"], max_sim_time=30.0,
               explore_budget=5.0)
  assert r["stranded"] and r["dead"] == "stuck"
  assert r["deaths"][0]["cause"] == "stuck" and "charger" in r["deaths"][0]["why"]
  assert r["state"] == "DONE" and r["sim_time"] < 30.0


# ---- the reset -----------------------------------------------------------


def test_a_reset_rights_a_robot_on_its_side_in_the_garden():
  """The acceptance line: on its side in the garden, dead, then an admin's
  `reset_robot` -- back at the start pose, standing, pack full, clock
  restarted, and both the event and History say so. (That lying there is a
  `stuck` death is the stub's rule, above.)"""
  life = _humanoid(inbox=Inbox())
  seen = _events(life)
  gx, gy = zone_centre(HUMANOID_HOME, "garden")
  life.body.start_at(gx, gy, 0.0)
  _topple_humanoid(life)
  _hold(life, 0.5)
  life._die("stuck", "knocked over")
  life.battery.energy_wh = 0.3 * life.battery.capacity_wh
  life.inbox.offer({"type": "reset_robot", "id": "rr_01", "from": "ben"})
  life._visitor_step()
  assert life.dead is None and not life.stranded
  assert life.battery.fraction == pytest.approx(1.0)
  assert life._chassis_tilt() < math.radians(2.0), "still on its side"
  x, y, _ = life.home_pose
  tx, ty, _ = life.body.true_pose()
  assert math.hypot(tx - x, ty - y) < 0.15
  assert life.survival_s < 0.01, "the survival clock restarts at a reset"
  assert len(life.deaths) == 1 and len(life.resets) == 1
  reset = seen[-1]
  assert reset["type"] == "reset" and reset["by"] == "ben"
  assert reset["wasDead"] == "stuck" and reset["intervention"] is False
  assert reset["deadS"] >= 0
  history = life.thoughts.volatile()[HISTORY]
  assert any("reset by ben" in line for line in history)
  assert life.telemetry_status()["survival"] == {"s": pytest.approx(0.0, abs=0.01),
                                                 "deaths": 1, "dead": None}


def test_a_reset_of_a_living_robot_is_marked_as_an_intervention():
  life = _life(inbox=Inbox())
  seen = _events(life)
  life.battery.energy_wh *= 0.5
  life.inbox.offer({"type": "reset_robot", "id": "rr_03", "from": "ben"})
  life._visitor_step()
  #  Found by TYPE rather than by position: a reset of a living robot is
  #  followed by an `intervention` event (issue #119).
  reset = next(e for e in seen if e["type"] == "reset")
  assert reset["intervention"] is True
  assert reset["wasDead"] is None and life.battery.fraction == pytest.approx(1.0)
  assert [e["what"] for e in seen if e["type"] == "intervention"] \
    == ["reset_robot"]
  assert life.interventions and life.interventions[0]["by"] == "ben"


def test_a_dead_robot_with_an_inbox_waits_and_a_reset_resumes_the_day():
  """End to end through `run()`: dead as the day starts, the loop stands in
  DEAD streaming frames rather than ending, the admin's reset arrives on
  the inbox, and the day carries on."""
  life = _life(inbox=Inbox())
  life.battery.energy_wh = 0.0
  _hold(life, 0.2)
  assert life.dead is not None
  states: list[str] = []
  sent = [False]

  def hook():
    if life.state != (states[-1] if states else None):
      states.append(life.state)
    if life.dead is not None and life.data.time > 8.0 and not sent[0]:
      sent[0] = True
      life.inbox.offer({"type": "reset_robot", "id": "rr_04", "from": "ben"})

  life.body.step_hooks.append(hook)
  life.stop_when(lambda: bool(life.resets) and life.state not in ("DEAD",))
  r = life.run(world_config(HUMANOID_HOME)["start"], max_sim_time=60.0,
               explore_budget=5.0)
  assert "DEAD" in states, states
  assert r["deaths"][0]["cause"] == "flat" and len(r["resets"]) == 1
  assert r["dead"] is None and r["battery"] > 0.9


# ---- what the mind is shown --------------------------------------------


def test_the_next_decision_is_shown_the_death_and_the_clock():
  life = _life()
  _hold(life, 1.5)
  life.battery.energy_wh = 0.0
  _hold(life, 0.3)
  ctx = ov.context_for(life, thoughts=life.thoughts)
  assert ctx["survival"]["deaths"] == 1
  assert ctx["survival"]["aliveS"] == pytest.approx(1.8, abs=0.2)
  assert any("died" in line for line in ctx["thoughts"][HISTORY])


def test_the_robot_is_told_it_can_die_only_where_it_can():
  """⚠ A RULE THE ARM CONTRADICTS IS A FALSE STATEMENT THE MODEL ACTS ON.
  Mortality is opt-in, so a world whose robot cannot die must not be told
  that it can (Evaluation.md §2)."""
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.economy.scoring import default_table

  def rules(mortal: bool) -> str:
    return ov.system_prompt(ThoughtFiles(), Menu.for_world(HUMANOID_HOME, None),
                            default_table(), mortal=mortal)[0]["text"]

  assert "YOU CAN DIE" in rules(True)
  assert "survival.aliveS" in rules(True)
  assert "YOU CAN DIE" not in rules(False)
  # ...and the flag ADDS a block and changes nothing else, so an immortal
  # world's cached prefix is what it was before issue #107. (A world with
  # no appetite is told the rule without its upkeep clauses, issue #387.)
  assert rules(True).replace("\n\n" + ov.mortal_rule(False), "") \
      == rules(False)


def test_mortality_is_opt_in_and_an_immortal_day_ends_as_it_always_did():
  """⚠ THE DEFAULT IS OFF, and not out of caution: on a demo cell the pack
  reaches ZERO mid-errand as documented behaviour and the robot carries on.
  An immortal lifecycle records no death, and its loop ends on an empty
  pack."""
  life = _life(mortal=False)
  assert life.mortal is False
  life.battery.energy_wh = 0.0
  _hold(life, 0.5)
  assert life.dead is None and life.deaths == []
  r = life.run(world_config(HUMANOID_HOME)["start"], max_sim_time=30.0,
               explore_budget=5.0)
  assert r["dead"] is None and r["deaths"] == []
  assert "BATTERY DEAD" in life.log[-1]
  # ...and the default follows the inbox: a served world can be reset, a
  # test, a spike and a filmstrip cannot.
  assert _life(mortal=None).mortal is False
  assert _life(inbox=Inbox(), mortal=None).mortal is True


# ---- the `stuck` death, a body's own budget (issue #387) ---------------------


def test_a_body_that_rights_itself_is_stuck_only_past_its_budget():
  """The `stuck` death for a body that gets back up: not at the default
  `TOPPLE_HOLD_S`, but once its get-up's budget has run out, and in words
  that say it fell and could not get up."""
  body = StubBody()
  body.rights_itself, body.stuck_after_s = True, 4.0
  life = stub_life(body=body, mortal=True)
  body.attitude = (math.cos(0.7), math.sin(0.7), 0.0, 0.0)   # 80 deg over
  body.run(body.hold_routine(3.0))
  assert life.dead is None, "past TOPPLE_HOLD_S, inside the get-up's budget"
  body.run(body.hold_routine(1.5))
  assert life.dead is not None and life.dead["cause"] == "stuck"
  assert "fell and could not get up" in life.dead["why"]


def test_a_body_that_does_not_right_itself_is_knocked_over_at_its_own_hold():
  """...and one that cannot get back up is `stuck` at its own hold, in
  words that say it was knocked over."""
  body = StubBody()
  life = stub_life(body=body, mortal=True)
  body.attitude = (math.cos(0.7), math.sin(0.7), 0.0, 0.0)
  body.run(body.hold_routine(2.5))
  assert life.dead is not None and "knocked over" in life.dead["why"]


# ---- upkeep off, a configuration (issue #387) --------------------------------


def test_upkeep_is_said_only_where_there_is_one():
  """A world with no appetite has no upkeep to fail and no hours of paid
  work to price a heart in; a world WITH one reads the rule byte for byte."""
  assert ov.mortal_rule(True) == ov.MORTAL_RULE
  text = ov.mortal_rule(False).lower()
  assert "upkeep" not in text and "hours of work" not in text
  assert "heartprice" in text and "buy_heart" in text
  assert "upkeep you cannot pay" in ov.mortal_rule(True)


def test_upkeep_off_is_a_configuration_no_unpaid_death_and_no_metabolism_on_the_wire():
  """Mortal, broke and with no appetite: a day of nothing costs no heart,
  and the header advertises no hunger for a robot that has none."""
  from pluggybot.humanoid import world as hw
  from pluggybot.telemetry.recorder import FrameBuilder
  body = StubBody()
  life = stub_life(body=body, mortal=True)
  assert life.metabolism is None and life.ledger is None or life.ledger.balance() == 0
  body.run(body.hold_routine(30.0))
  assert life.dead is None and not life.deaths
  house = hw.home_spec().compile()
  data = mujoco.MjData(house)
  mujoco.mj_forward(house, data)
  builder = FrameBuilder(house, data, model_name=HUMANOID_HOME)
  assert builder.header()["hungerStates"] == []
