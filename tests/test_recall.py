"""Recall (issue #221): the one action that reads memory, and what it costs.

Each rule pinned without a model: the lookup's key forms and its caps on
the ThoughtFiles alone; the chain, its cap and its clearing through a real
lifecycle with a fake client (three ten-second stands, no errand); the
enums a recall may not appear in; `askedBy` off the arbitration seam.
"""

import pytest

from pluggybot.lifecycle import world_config
from pluggybot.mind import events as ev
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mind.thoughts import (
  HISTORY_RECALLED, RECALLED_CHAIN_CHARS, RECALLED_CHARS, ThoughtFiles,
)

from test_overseer import FakeClient, _lifecycle, full


# ---- the lookup ----------------------------------------------------------------


@pytest.fixture()
def files():
  f = ThoughtFiles()
  f.remember("draw on whiteboard_b failed: no route", t=10.0)
  f.note({"topic": "tasks/draw", "title": "far board", "text": "fails from the explore's end"}, t=20.0)
  f.note({"topic": "tasks/carry", "title": "pay", "text": "worth little"}, t=21.0)
  f.record({"quantity": "block mass", "value": 0.4, "unit": "kg", "method": "lift",
            "topic": "mass"}, t=30.0)
  f.pin("bay C sticks", t=40.0)
  f.unpin("bay C", t=50.0)
  return f


def test_read_takes_a_number_a_note_a_topic_a_family_or_history(files):
  one = files.recall(read="tasks/draw/far board")
  assert one["hits"] == 1 and one["lines"] == ["#2 [tasks/draw/far board] fails from the explore's end"]
  topic = files.recall(read="tasks")                    # the family tasks/*
  assert [ln.split()[0] for ln in topic["lines"]] == ["#2", "#3"]
  assert files.recall(read="tasks/carry")["hits"] == 1
  assert files.recall(read="findings")["lines"] == ["#4 [findings/mass] block mass = 0.4 kg -- lift"]
  by_id = files.recall(read="#1")
  assert by_id["lines"] == ["#1 [history] [t=10s] draw on whiteboard_b failed: no route"]
  assert files.recall(read="1") == by_id | {"read": "1"}
  assert files.recall(read="history")["hits"] == 1
  assert files.recall(read="nothing/like/this") == {"read": "nothing/like/this", "find": "",
                                                    "hits": 0, "lines": []}
  # A retired line is a hit, and says so.
  assert files.recall(read="#5")["lines"] == ["#5 [Top_of_mind.md t=40s retired] bay C sticks"]


def test_find_searches_everything_this_life_and_never_the_record_of_a_recall(files):
  hit = files.recall(find="whiteboard route")
  assert hit["hits"] == 1 and hit["lines"][0].startswith("#1 [history]")
  assert files.recall(find="bay C")["lines"] == ["#5 [Top_of_mind.md t=40s retired] bay C sticks"]
  both = files.recall(read="tasks/carry/pay", find="explore's end")
  assert [ln.split()[0] for ln in both["lines"]] == ["#3", "#2"]
  files.remember("chose recall (find 'whiteboard'): because", t=60.0)
  files.remember("recalled 1 line -- find 'whiteboard'", t=61.0)
  assert files.recall(find="whiteboard")["hits"] == 1, "a recall found itself"
  assert files.recall(read="history")["hits"] == 1, "...or the record of one"
  assert files.recall(find="") == {"read": "", "find": "", "hits": 0, "lines": []}


def test_a_block_is_capped_in_characters_and_says_what_it_cut():
  files = ThoughtFiles()
  for i in range(60):
    files.remember(f"line {i} " + "x" * 200, t=float(i))
  block = files.recall(read="history")
  assert block["hits"] == HISTORY_RECALLED
  assert sum(len(ln) + 1 for ln in block["lines"]) <= RECALLED_CHARS + 1
  assert block["cut"] == HISTORY_RECALLED - len(block["lines"]) > 0
  assert RECALLED_CHAIN_CHARS == 2 * RECALLED_CHARS


# ---- the enums -------------------------------------------------------------------


def test_recall_is_an_action_and_never_an_order():
  menu = Menu.for_world("home", None)
  assert "recall" in menu.available()
  schema = menu.schema(standing_orders=True, event_map=True)
  assert "recall" in schema["properties"]["action"]["enum"]
  assert "recall" not in schema["properties"]["standing_order"]["enum"]
  rows = schema["properties"]["event_map"]["items"]["properties"]["action"]["enum"]
  assert "recall" not in rows and ev.ASK in rows
  with pytest.raises(ValueError):
    ov.standing_order("recall", menu)
  with pytest.raises(ValueError):
    ev.row({"event": "every", "action": "recall", "value": 60, "kind": ""}, menu)
  assert not ov.order_runnable(menu, "recall", {"possibleActions": ["carry"]})
  # ...and at the cap it leaves the action enum for that call.
  assert "recall" not in menu.schema(recall=False)["properties"]["action"]["enum"]
  assert ov._recall_allowed({}) and ov._recall_allowed({"recallsLeft": 1})
  assert not ov._recall_allowed({"recallsLeft": 0})


def test_a_recall_names_what_it_looks_up_or_is_malformed():
  menu = Menu.for_world("room_hub", None)
  d = menu.validate(full(action="recall", read="tasks", find="pen"))
  assert (d.read, d.find) == ("tasks", "pen") and d.summary().startswith("recall (read tasks find 'pen')")
  with pytest.raises(ValueError):
    menu.validate(full(action="recall"))
  with pytest.raises(ValueError):
    menu.validate(full(action="recall", find="pen"), recall=False)
  # The parameters belong to `recall` alone.
  assert menu.validate(full(action="idle", read="tasks")).read == ""


# ---- the chain, through the lifecycle ----------------------------------------------


def _chain(*answers):
  boss = Overseer(Menu.for_world("room_hub", None), client=FakeClient(*answers))
  life = _lifecycle("room_hub", overseer=boss, errand=False)
  life.thoughts.remember("draw on whiteboard_b failed", t=1.0)
  life.mission.start_at(*world_config("room_hub")["start"])
  return boss, life


def test_the_chain_accumulates_and_an_external_action_clears_it():
  boss, life = _chain(full(action="recall", find="whiteboard"),
                      full(action="recall", read="history"),
                      full(action="idle"),
                      full(action="idle"))
  seen = []
  life.on_event.append(seen.append)
  try:
    life._decide()
    assert life.state == "RECALL" and life._recall_run == 1
    t_after_first = float(life.data.time)
    life._decide()
    assert life._recall_run == 2 and [b["hits"] for b in life._recalled] == [1, 1]
    assert float(life.data.time) - t_after_first == pytest.approx(ov.RECALL_S, abs=0.2)
    # The context the third call would see carries the whole chain...
    from pluggybot.lifecycle import overseer_context
    state = overseer_context(life)
    assert [b["read"] or b["find"] for b in state["recalled"]] == ["whiteboard", "history"]
    assert state["recallsLeft"] == ov.MAX_RECALL_RUN - 2
    assert state["askedBy"] == {"event": "loop"}
    life._decide()                            # idle: an external action
    assert life._recalled == [] and life._recall_run == 0
    assert overseer_context(life)["recalled"] == []
  finally:
    life.mission.close()
  assert [d["action"] for d in life.decisions][:3] == ["recall", "recall", "idle"]
  assert [(r["run"], r["hits"], r["shown"]) for r in life.recalls] == [(1, 1, 1), (2, 1, 1)]
  wire = [m for m in seen if m["type"] == "recall"]
  assert [(m["read"], m["find"], m["hits"]) for m in wire] == [("", "whiteboard", 1), ("history", "", 1)]
  assert "recalled 1 line -- read history" in life.thoughts.read("History.md")
  # ...and the rows reach the result dict `end()` builds for the record.
  import inspect
  from pluggybot.lifecycle import HubLifecycle
  assert '"recalls": list(self.recalls)' in inspect.getsource(HubLifecycle.end)


def test_the_fourth_recall_in_a_row_is_refused_and_the_chain_ends():
  boss, life = _chain(*([full(action="recall", read="history")] * 4),
                      full(action="idle"))
  try:
    for _ in range(3):
      life._decide()
    assert life._recall_run == ov.MAX_RECALL_RUN == 3
    from pluggybot.lifecycle import overseer_context
    assert overseer_context(life)["recallsLeft"] == 0
    life._decide()                            # the fourth: malformed
  finally:
    life.mission.close()
  assert boss.decisions[-1].source == "fallback:garbled"
  assert life._recall_run == 0 and life._recalled == []


def test_a_map_row_that_asks_says_what_asked(monkeypatch):
  """`askedBy` is the row that fired -- until #221 an ask at 30 % and an
  ask with nothing to do were the same prompt."""
  boss, life = _chain(full(action="idle"), full(action="idle"))
  from pluggybot.lifecycle import overseer_context
  seen = []
  real = overseer_context

  def spy(life_):
    state = real(life_)
    seen.append(state.get("askedBy"))
    return state

  monkeypatch.setattr("pluggybot.lifecycle.overseer_context", spy)
  try:
    life.mission.run(life._decide_routine({"event": "battery_below", "kind": "", "value": 0.3}))
    life._decide()
  finally:
    life.mission.close()
  assert seen[0] == {"event": "battery_below", "kind": "", "value": 0.3}
  assert seen[1] == {"event": "loop"}
