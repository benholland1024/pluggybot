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
from pluggybot.mind.inbox import MAX_EARLIER
from pluggybot.mind.thoughts import (
  CHAT_ANSWERED, CHAT_HEARD, CHAT_RECALLED_CHARS, CHAT_SAID, HISTORY_RECALLED,
  RECALLED_CHAIN_CHARS, RECALLED_CHARS, ThoughtFiles,
)

from test_body import stub_life
from test_overseer import FakeClient, full


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


def test_read_chat_is_the_chat_in_history_and_nothing_else():
  """`chat` (issue #485) reads the robot's chat as History holds it -- what
  people said, what it answered, what it said unasked -- and nothing else
  History holds, oldest first, each line marked as the chat's. Shown to
  fail by reading `chat` as a note topic: no hits."""
  f = ThoughtFiles()
  f.remember("docked", t=1.0)
  f.remember("ada said: hello", t=2.0, chat=CHAT_HEARD)
  f.remember("replied to ada: hi ada", t=3.0, chat=CHAT_ANSWERED)
  f.remember("charged to 90%", t=4.0)
  f.remember("said in my chat: the sun on b is done", t=5.0, chat=CHAT_SAID)
  block = f.recall(read="chat")
  assert (block["hits"], "cut" in block) == (3, False)
  assert block["lines"] == [
    "#2 [chat] [t=2s] ada said: hello",
    "#3 [chat] [t=3s] replied to ada: hi ada",
    "#5 [chat] [t=5s] said in my chat: the sun on b is done"]
  assert f.recall(read="CHAT")["lines"] == block["lines"]
  assert f.last_said() == 5.0


def test_read_chat_is_wider_than_a_message_and_ends_at_the_newest_line():
  """The window `chat` opens is wider than the lines a message carries
  (`MAX_EARLIER`) -- the chain's whole room, where one block's would hold
  seven full-length messages -- and it is the NEWEST lines that fit, with
  no hole: a window onto a conversation ends at its latest line, and what
  it cuts it counts. Shown to fail with `RECALLED_CHARS` as its room, or
  with the oldest lines kept."""
  f = ThoughtFiles()
  words = "w" * 470
  for i in range(40):
    f.remember(f"ada said: {i:02d} {words}", t=float(i), chat=CHAT_HEARD,
               room=ov.MAX_REPLY)
  f.remember(f"ada said: 40 {'x' * 9}", t=40.0, chat=CHAT_HEARD)
  block = f.recall(read="chat")
  said = [int(line.split("ada said: ")[1][:2]) for line in block["lines"]]
  assert len(said) > 2 * MAX_EARLIER
  assert said == list(range(41 - len(said), 41))
  assert sum(len(line) + 1 for line in block["lines"]) <= CHAT_RECALLED_CHARS + 1
  assert (block["hits"], block["cut"]) == (41, 41 - len(said))


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
  menu = Menu.for_world("home_quad", None)
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
  menu = Menu.for_world("home_quad", None)
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
  boss = Overseer(Menu.for_world("home_quad", None), client=FakeClient(*answers))
  life = stub_life(overseer=boss)
  life.thoughts.remember("draw on whiteboard_b failed", t=1.0)
  life.body.start_at(*world_config("home_quad")["start"])
  life.max_sim_time = 0.0            # a fallback's explore ends where it starts
  return boss, life


def test_the_chain_accumulates_and_an_external_action_clears_it():
  boss, life = _chain(full(action="recall", find="whiteboard"),
                      full(action="recall", read="history"),
                      full(action="idle"),
                      full(action="idle"))
  seen = []
  life.on_event.append(seen.append)
  # ...and how many think slices each call in flight was stood out in: as
  # many as its worker thread took, which is the box's, not the recall's
  thinks = []
  real = life.body.hold_routine
  life.body.hold_routine = lambda s: (thinks.append(s == ov.THINK_SLICE_S), real(s))[1]
  try:
    life._decide()
    assert life.state == "RECALL" and life._recall_run == 1
    t_after_first = float(life.data.time)
    thinks.clear()
    life._decide()
    assert life._recall_run == 2 and [b["hits"] for b in life._recalled] == [1, 1]
    stood = float(life.data.time) - t_after_first - sum(thinks) * ov.THINK_SLICE_S
    assert stood == pytest.approx(ov.RECALL_S, abs=0.01)
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
    life.body.close()
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
    life.body.close()
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
    life.body.run(life._decide_routine({"event": "battery_below", "kind": "", "value": 0.3}))
    life._decide()
  finally:
    life.body.close()
  assert seen[0] == {"event": "battery_below", "kind": "", "value": 0.3}
  assert seen[1] == {"event": "loop"}
