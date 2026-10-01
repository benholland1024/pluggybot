"""An admin reaching into world state, and the record of every one (#119).

`reset_tool` (issue #30) and `reset_robot` (issue #107) established the
shape and this follows it exactly: an inbound kind on the existing
bidirectional socket, admin-only at the WEBSITE's door, code-handled on the
physics thread, never shown to the overseer, refused while a module is
seated on the fork.

What is new is the RECORD. docs/Evaluation.md §5: an admin intervention
contaminates every survival number in its run, so every one is written
down. These are the two kinds that fill `interventions`, and the four
traces each one leaves.

⚠ WHO IS AN ADMIN IS NOT A QUESTION THE SIM CAN ANSWER, here or for either
reset. `from` is a label the website puts on the message, and the gate is
`role !== 'admin'` at the API boundary one repo over. What the sim owes is
that the kinds exist, that they are code's rather than the model's, and
that every use is written down where it cannot be edited.
"""

import json

import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, world_config
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.overseer import ACTIONS, Menu, model_state
from pluggybot.mind.thoughts import HISTORY
from pluggybot.telemetry.protocol import (
  CODE_HANDLED_TYPES, INBOUND_TYPES, INTERVENTION_KINDS, PROTOCOL_VERSION,
)

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _life(world: str = QUAD_HOME, inbox=None, ledger=None,
          **kw) -> HubLifecycle:
  """An admin's reach-in is the lifecycle's bookkeeping: a stub body
  carries it (issue #380)."""
  cfg = world_config(world)
  life = stub_life(world, inbox=inbox, ledger=ledger, **kw)
  life.body.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  return life


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


# ---- the vocabulary ----------------------------------------------------------


def test_the_two_kinds_are_admin_kinds_code_handles_and_the_wire_bumped():
  for kind in ("set_battery", "set_points"):
    assert kind in INBOUND_TYPES
    assert kind in CODE_HANDLED_TYPES, \
      f"{kind} would need an overseer to be acted on"
  assert set(CODE_HANDLED_TYPES) <= set(INBOUND_TYPES)
  #  ⚠ `reset_robot` IS in this list even though it has an event of its own,
  #  because it is the one that is SOMETIMES not an intervention -- see the
  #  vocabulary's own note. Counting interventions must be one query.
  assert INTERVENTION_KINDS == ("reset_robot", "set_battery", "set_points")
  assert PROTOCOL_VERSION == "0.21.0", \
    "new inbound kinds and a new event type are a two-repo event"


# ---- what the inbox will accept ----------------------------------------------


@pytest.mark.parametrize("raw, why", [
  ({"type": "set_battery"}, "neither a fraction nor a watt-hour figure"),
  ({"type": "set_battery", "frac": 0.5, "wh": 4.0}, "two requests in one"),
  ({"type": "set_battery", "frac": 1.5}, "a fraction over one"),
  ({"type": "set_battery", "frac": -0.1}, "a negative fraction"),
  ({"type": "set_battery", "frac": "half"}, "not a number"),
  ({"type": "set_battery", "frac": float("nan")}, "NaN"),
  ({"type": "set_battery", "wh": -1.0}, "a negative pack"),
  ({"type": "set_points"}, "no balance at all"),
  ({"type": "set_points", "points": -5}, "a debt"),
  ({"type": "set_points", "points": "lots"}, "not a number"),
])
def test_the_inbox_refuses_an_intervention_it_cannot_act_on(raw, why):
  box = Inbox()
  assert box.offer({**raw, "id": "x", "from": "ben"}) is None, why
  assert box.dropped_invalid == 1


def test_the_inbox_takes_a_fraction_or_watt_hours_and_keeps_them_apart():
  #  A fraction is what an operator thinks in; watt-hours are what the pack
  #  is measured in. Neither is derived from the other HERE, because the
  #  queue does not know the capacity.
  box = Inbox()
  by_frac = box.offer({"type": "set_battery", "id": "a", "frac": 0.25})
  by_wh = box.offer({"type": "set_battery", "id": "b", "wh": 2.5})
  assert (by_frac.frac, by_frac.wh) == (0.25, None)
  assert (by_wh.frac, by_wh.wh) == (None, 2.5)
  points = box.offer({"type": "set_points", "id": "c", "points": 40})
  assert points.points == 40
  assert points.as_dict()["points"] == 40


# ---- the pack ----------------------------------------------------------------


def test_setting_the_battery_moves_the_pack_and_leaves_four_traces(tmp_path):
  """The change is the easy half. The RECORD is what the issue is about, and
  each of the four answers a question the others cannot."""
  life = _life(inbox=Inbox(), thoughts=lc.ThoughtFiles.open(tmp_path))
  seen = _events(life)
  life.battery.energy_wh = life.battery.capacity_wh * 0.2
  life.inbox.offer({"type": "set_battery", "id": "sb_1", "from": "ben",
                    "frac": 1.0})
  life._visitor_step()

  assert life.battery.fraction == pytest.approx(1.0)
  #  1. the lifecycle's own list, which says a run is not a survival data point
  assert len(life.interventions) == 1
  entry = life.interventions[0]
  assert entry["what"] == "set_battery" and entry["by"] == "ben"
  assert entry["before"]["frac"] == pytest.approx(0.2, abs=1e-3)
  assert entry["after"]["frac"] == pytest.approx(1.0)
  #  2. a structured event on the wire, for the operator log
  assert [e["type"] for e in seen] == ["intervention"]
  #  3. a narration line, for whoever is watching
  assert "ADMIN ben" in life.status and "battery" in life.status
  #  4. and the robot's own unrevisable record, which the next decision reads
  history = (tmp_path / HISTORY).read_text()
  assert "ben reached in" in history


def test_a_pack_is_clamped_to_its_capacity_and_the_line_says_what_landed():
  life = _life(inbox=Inbox())
  life.inbox.offer({"type": "set_battery", "id": "sb_2", "from": "ben",
                    "wh": 999.0})
  life._visitor_step()
  assert life.battery.energy_wh == pytest.approx(life.battery.capacity_wh)
  assert "100%" in life.status


def test_a_full_pack_does_not_stand_a_dead_robot_up(monkeypatch):
  """⚠ `reset_robot` is the revival and this is not. Conflating them would
  give an operator two ways to do one thing and no way to do the other -- a
  pack refilled under a robot lying on its side is exactly as stuck as it
  was. The narration says so rather than leaving a full gauge next to a
  corpse unexplained."""
  life = _life(inbox=Inbox(), mortal=True)
  life.battery.energy_wh = 0.0
  #  The seam is throttled to DEATH_CHECK_S, so a bare call at t=0 on a
  #  lifecycle whose clock has already moved is a no-op.
  life._next_death_check = 0.0
  life._death_step()
  assert life.dead and life.dead["cause"] == "flat"

  life.inbox.offer({"type": "set_battery", "id": "sb_3", "from": "ben",
                    "frac": 1.0})
  life._visitor_step()
  assert life.battery.fraction == pytest.approx(1.0)
  assert life.dead is not None, "a top-up quietly revived a dead robot"
  assert "still dead" in life.status


# ---- the balance -------------------------------------------------------------


def test_setting_the_points_breaks_the_identity_and_says_so(tmp_path):
  """⚠ `earned - consumed - spent == balance` is SUPPOSED to break here.

  Papering the difference into `earned` would hide a reach-in inside the one
  number the reward system exists to make un-fakeable (issue #14). The
  identity failing is how an intervention shows up in the ECONOMY column and
  not only in the survival one.
  """
  from pluggybot.economy.ledger import Ledger

  ledger = Ledger(path=tmp_path / "ledger.json")
  life = _life(inbox=Inbox(), ledger=ledger)
  life.inbox.offer({"type": "set_points", "id": "sp_1", "from": "ben",
                    "points": 250})
  life._visitor_step()

  assert ledger.balance() == 250
  #  `earned` is untouched -- that is the whole point.
  assert ledger.robots["pluggybot"]["earned"] == 0
  #  ...and the receipt says how big the reach-in was, without ever being a
  #  term in the identity.
  assert ledger.intervened() == 250
  assert ledger.snapshot()["pluggybot"]["intervened"] == 250
  #  It survives a restart, like every other thing in this file.
  assert Ledger(path=tmp_path / "ledger.json").intervened() == 250


def test_a_balance_has_no_debt_and_a_world_with_no_ledger_says_so():
  life = _life(inbox=Inbox(), ledger=None)
  life.inbox.offer({"type": "set_points", "id": "sp_2", "from": "ben",
                    "points": 10})
  life._visitor_step()
  assert "keeps no ledger" in life.status
  assert life.interventions == []


# ---- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("raw", [
  {"type": "set_battery", "frac": 1.0},
  {"type": "set_points", "points": 99},
])
def test_an_intervention_is_refused_while_a_module_is_on_the_fork(raw):
  """⚠ NOT symmetry with `reset_tool` for its own sake. The energy gate
  prices the next errand against the pack BETWEEN errands and never inside
  one, so a pack that moves while a module is seated changes the arithmetic
  of a decision already taken. The window is the swap, not the errand, so a
  rescue waits seconds rather than minutes."""
  from pluggybot.economy.ledger import Ledger

  life = _life(inbox=Inbox(), ledger=Ledger())
  life.body.holding = life.module                 # seated on its coupling
  life.body.run(life.body.hold_routine(0.2))
  assert life.tool_powered, "the seam did not see the seated module"
  before = (life.battery.energy_wh, life.ledger.balance())

  life.inbox.offer({**raw, "id": "r_1", "from": "ben"})
  life._visitor_step()
  assert "refused" in life.status and "seated on the fork" in life.status
  assert (life.battery.energy_wh, life.ledger.balance()) == before
  assert life.interventions == []


# ---- the model is never told -------------------------------------------------


def test_neither_kind_reaches_the_overseer():
  """An admin command is code's to apply, not the robot's to weigh -- so it
  is in no action menu, no context, and not among the messages a decision is
  shown."""
  assert not set(ACTIONS) & {"set_battery", "set_points", "reset_robot",
                             "reset_tool"}
  schema = json.dumps(Menu.for_world(QUAD_HOME).schema())
  assert "set_battery" not in schema and "set_points" not in schema

  #  ...and nothing about one can ride in as a visitor message: the physics
  #  seam drains these kinds before any decision is made, and `as_context`
  #  carries no field they could hide in even if one were still queued.
  box = Inbox()
  box.offer({"type": "set_battery", "id": "sb_9", "from": "ben", "frac": 1.0})
  box.offer({"type": "message", "id": "m_1", "from": "ada", "text": "hello"})
  from pluggybot.mind import overseer as ov

  boss = ov.build(QUAD_HOME, enabled=True)
  life = _life(inbox=box, overseer=boss)
  life._visitor_step()
  assert [m.kind for m in box.peek()] == ["message"]
  assert set(box.peek()[0].as_context()) == {"id", "from", "text"}

  state = lc.overseer_context(life)
  assert "set_battery" not in json.dumps(state)
  assert "set_battery" not in json.dumps(model_state(state))
