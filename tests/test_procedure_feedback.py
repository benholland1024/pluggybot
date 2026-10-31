"""The development loop a robot needs to finish a challenge (issue #264):
the one History line a procedure leaves, the one-answer replacement the
prompt promises, a procedure run on the answer that defines it, what a
failed walk says, and a failed step's trace -- each rule pinned in
milliseconds. (The sentence a failed pick was said in, and the fork checked
before a fetch, went with the quadruped's tools: #559.)
"""

import math
import re
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import navigator as nav
from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, errand_from
from pluggybot.mind.overseer import Decision, FIELD_INDEX, HANDS_PROCEDURE_HEAD
from pluggybot.procedure import library as lib
from pluggybot.procedure import steps as st
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

WORLD = "home_humanoid"


# ---- 1. one History line per procedure the robot wrote ---------------------


def test_a_procedure_cut_short_tells_the_robot_the_line_and_the_reason():
  """A walk that never arrives, on a procedure's first line: the robot is
  told where and why -- on the History it reads, and as `failedReason` on
  the wire."""
  life = stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  life.drive_why = lambda x, y, record=None: "no route there on my map"
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts(WORLD))
  L.define("walk", 'def walk():\n  drive_to(1, 1)\n  wait(1)\n')
  life.run_errand(errand_from(Decision(action="procedure:walk"), WORLD, library=L))
  history = life.thoughts.read("History.md")
  assert ("the procedure walk did not finish -- it stopped at line 2, drive_to: "
          "did not arrive at (1, 1): no route there on my map, at (0.0, 0.0) "
          "(0 steps had run)") in history
  cut = next(e for e in events if e.get("type") == "procedure"
             and e.get("outcome") == "aborted")
  assert cut["failedLine"] == 2
  assert cut["failedReason"].startswith("did not arrive at (1, 1)")


def test_a_call_refused_at_run_time_is_named_once():
  """A computed tag is checked when it is computed, and History names the
  failed step's call before its reason -- a reason that named it too read
  `line 3, pick: pick(tag=99) ...` (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("big", "def big():\n  t = 99\n  find(t, 1, 1)\n")
  life.run_errand(errand_from(Decision(action="procedure:big"), WORLD, library=L))
  assert ("it stopped at line 3, find: tag 99 names no place this world has "
          "(places: 35 to 46, 60 to 63) (0 steps had run)") in life.thoughts.read("History.md")


def test_an_argument_the_run_cannot_compute_is_named_by_its_call():
  """A fault computing a call's argument said `it stopped at an expression:
  division by zero`, no line, no call -- though the step knew both (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("u", "def u():\n  t = 0\n  wait(1 / t)\n")
  life.run_errand(errand_from(Decision(action="procedure:u"), WORLD, library=L))
  assert "it stopped at line 3, wait: division by zero" in life.thoughts.read("History.md")


def test_a_local_is_said_with_every_digit():
  """`:g` kept six digits: a readout of 1234567 reached History as
  `1.23457e+06`, beside a refusal that quoted its tag whole (#463)."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("big", "def big():\n  t = 1234567\n  u = 2.5\n  wait(0.1)\n")
  life.run_errand(errand_from(Decision(action="procedure:big"), WORLD, library=L))
  assert "it ended with t = 1234567, u = 2.5" in life.thoughts.read("History.md")


def test_a_procedure_that_finished_says_so():
  """A robot must know its procedure ENDED to say `done` after it: round 2
  of #264's probes set `done` before the run, and the grade found one block."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("tidy", 'def tidy():\n  wait(1)\n  face(0.5)\n')
  life.run_errand(errand_from(Decision(action="procedure:tidy"), WORLD, library=L))
  history = life.thoughts.read("History.md")
  assert re.search(r"ran the procedure tidy to its end \(2 steps\)$", history, re.M)


def test_a_procedure_out_of_budget_says_first_that_it_did_not_finish():
  """MEASURED (ladder B, 2026-09-24): the line said "(4/4 steps) -- it ran
  out of the time its budget gave it", the model read "all four steps
  landed", said `done`, and the grade found two blocks of three. The count
  is of calls MADE, so a stopped run never shows it as a fraction."""
  run = {"ok": False, "completed": 4, "total": 4, "stopped": "budget", "seconds": 324.2,
         "steps": [{"i": 3, "verb": "pick", "line": 6, "ok": True}]}
  [line] = lc.procedure_outcome("stack_tower", run)
  assert line == ("the procedure stack_tower did not finish -- it ran out of the time "
                  "its budget gave it (324 s) after line 6 (4 steps had run)")
  assert "/" not in line


def test_a_long_reason_never_costs_a_run_its_readout():
  """History cuts a line at 400 characters without a mark, its own
  "[t=NNNNs] " stamp INSIDE the 400, and a failed pick's reason in front of
  a weighing's locals cut off `mass =` -- the one number the bench grade
  needed recorded. Written through History at a clock with a long stamp,
  across every length where the old split kept "mass = 0" of
  "mass = 0.207812", the number arrives whole or not at all. A fault inside
  the run is named, never quoted (the exception is the log's), and keeps no
  false count."""
  life = stub_life()
  life.data.time = 123456.0
  for k in range(120, 420):
    run = {"ok": False, "completed": 7, "failedAt": 6, "stopped": None,
           "steps": [{"i": 6, "verb": "pick", "line": 12, "ok": False, "reason": "x" * k}],
           "locals": {"f23": 7.36057, "f0": 6.40608, "mass": 0.207812}}
    for line in lc.procedure_outcome("weigh", run):
      life._remember(line)
    last = life.thoughts.read("History.md").splitlines()[-1]
    assert "mass = 0.207812" in last, (k, last[-60:])
  # twelve locals with long names: whole values only, and it says so
  names = {f"reading_of_the_lift_force_number_{i:02d}": 1.2345 + i for i in range(12)}
  run = {"ok": True, "completed": 3, "locals": names}
  lines = lc.procedure_outcome("weigh", run)
  assert len(lines) == 2 and lines[1].endswith(" more)")
  for item in lines[1].split(" ended with ")[1].rsplit(" (+", 1)[0].split(", "):
    key, value = item.split(" = ")
    assert names[key] == pytest.approx(float(value), abs=1e-4)
  [line] = lc.procedure_outcome("weigh", {"ok": False, "completed": 0,
                                          "error": "KeyError: 'axis'"})
  assert line == "the procedure weigh did not finish -- it stopped on a fault inside the run"

# ---- 2. a replacement is one answer, as the prompt says --------------------


def test_one_answer_replaces_a_procedure_and_the_prompt_says_it_can():
  """The lifecycle has always applied `undefine` before `define`; the robot
  was told "there is no replace -- undefine, then define" and read it as two
  turns. The claim and the behaviour are pinned together."""
  assert "no replace" not in HANDS_PROCEDURE_HEAD
  assert "on\nthe same answer" in HANDS_PROCEDURE_HEAD or "same answer" in HANDS_PROCEDURE_HEAD
  assert any(name == "undefine" and "same answer" in text
             for name, _, _, text in FIELD_INDEX)
  life = SimpleNamespace(data=SimpleNamespace(time=0.0), world=WORLD,
                         root="pluggybot",
                         _say=lambda *a, **k: None, _remember=lambda *a, **k: None,
                         _emit=lambda e: None)
  L = lib.Library(lc.world_facts(WORLD), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("b", "def b():\n  wait(1)\n")          # full
  life.overseer = SimpleNamespace(library=L)
  HubLifecycle._define(life, Decision(action="idle", undefine="a",
                                      define={"name": "a", "source": "def a():\n  wait(2)\n"}))
  assert "wait(2)" in L.entries["a"].source and len(L.entries) == 2


def test_a_refused_define_says_the_undefine_goes_on_the_same_answer():
  L = lib.Library(lc.world_facts(WORLD), cap=1)
  L.define("a", "def a():\n  wait(1)\n")
  with pytest.raises(lib.LibraryRefused) as e:
    L.define("a", "def a():\n  wait(2)\n")
  said = " ".join(e.value.reasons)
  assert "same answer" in said and "there is no replace" not in said
  assert "(it holds 1)" in said                     # never "2 of 1"


# ---- 3. a procedure is run on the answer that defines it -------------------


def _menu():
  from dataclasses import replace
  from pluggybot.mind.overseer import Menu
  from pluggybot.lifecycle import board_book
  return replace(Menu.for_world(WORLD, board_book(WORLD)), procedures=True)


def test_the_answer_that_defines_a_procedure_can_name_it_even_in_an_empty_library():
  """The enum is built from the library BEFORE the answer: a procedure
  defined in it could never be named in it, and the decoder substituted
  one that could -- Luca's weighing ran `count_blocks` and it blamed
  itself. `procedure:new` is the answer's own define, and never an order."""
  from pluggybot.mind.overseer import PROCEDURE_NEW, standing_order
  menu = _menu()
  schema = menu.schema(procedures=())
  assert PROCEDURE_NEW in schema["properties"]["action"]["enum"]
  assert PROCEDURE_NEW not in menu.orderable(())
  with pytest.raises(ValueError, match="an order has no answer"):
    standing_order(PROCEDURE_NEW, menu)


def test_procedure_new_needs_a_define_on_the_same_answer():
  """...and the refusal says how to run one that is already there: ladder B
  on the bench lost two turns in a row to `procedure:new` with no define,
  meaning "run my weigh again"."""
  from pluggybot.mind.overseer import PROCEDURE_NEW
  menu = _menu()
  with pytest.raises(ValueError, match="defines none"):
    menu.validate({"action": PROCEDURE_NEW, "reason": "x"}, procedures=())
  with pytest.raises(ValueError, match=r"name it \(procedure:weigh\)"):
    menu.validate({"action": PROCEDURE_NEW, "reason": "x"}, procedures=("weigh",))
  menu.validate({"action": PROCEDURE_NEW, "reason": "x",
                 "define": {"name": "weigh", "source": "def weigh():\n  wait(1)\n"}},
                procedures=())


def _deciding_life(library):
  life = stub_life()
  life.overseer = SimpleNamespace(library=library, event_map=None)
  events = []
  life.on_event.append(events.append)
  return life, events


def test_procedure_new_runs_what_the_same_answer_defined_and_not_another():
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts(WORLD))
  L.define("old", "def old():\n  wait(1)\n  face(0.5)\n")
  life, events = _deciding_life(L)
  decision = Decision(action=PROCEDURE_NEW, reason="weigh it now",
                      define={"name": "weigh", "source": "def weigh():\n  wait(1)\n"})
  life.body.run(life._after_decision_routine(decision))
  queued = [e.program.name for e in life.errands if e.program is not None]
  assert queued == ["weigh"]                 # queued for the loop, and not `old`


def test_procedure_new_runs_nothing_when_the_define_was_refused():
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts(WORLD))
  L.define("old", "def old():\n  wait(1)\n")
  life, events = _deciding_life(L)
  decision = Decision(action=PROCEDURE_NEW, reason="try",
                      define={"name": "bad", "source": "def bad():\n  import os\n"})
  life.body.run(life._after_decision_routine(decision))
  assert not [e for e in life.errands if e.program is not None]
  assert "ran nothing: `procedure:new`" in life.thoughts.read("History.md")


def test_a_refused_define_is_written_where_the_robot_reads():
  """Narrated and put on the wire, a refusal reached everyone but the robot
  that made it -- and both deployed libraries were full, so "an `undefine`
  on the same answer makes room" was said to nobody who could act on it."""
  life, _ = _deciding_life(lib.Library(lc.world_facts(WORLD)))
  life._define(Decision(action="idle", reason="x",
                        define={"name": "bad", "source": "def bad():\n  import os\n"}))
  assert "could not write the procedure bad:" in life.thoughts.read("History.md")


def test_a_one_answer_rewrite_that_is_refused_keeps_the_procedure_it_had():
  """Undefine and define of one name is a swap: applied in order, a refused
  define had already deleted the procedure it was meant to improve."""
  L = lib.Library(lc.world_facts(WORLD))
  L.define("weigh", "def weigh():\n  wait(1)\n")
  life, _ = _deciding_life(L)
  life._define(Decision(action="idle", reason="x", undefine="weigh",
                        define={"name": "weigh", "source": "def weigh():\n  import os\n"}))
  assert "weigh" in L.runnable() and "wait(1)" in L.entries["weigh"].source
  history = life.thoughts.read("History.md")
  assert "could not rewrite the procedure weigh -- the one I had is kept:" in history
  assert "forgot the procedure weigh" not in history
  life._define(Decision(action="idle", reason="x", undefine="weigh",
                        define={"name": "weigh", "source": "def weigh():\n  wait(2)\n"}))
  assert "wait(2)" in L.entries["weigh"].source and life._defined_now == "weigh"


def test_new_is_not_a_name_a_procedure_may_take():
  L = lib.Library(lc.world_facts(WORLD))
  with pytest.raises(lib.LibraryRefused, match="procedure:new"):
    L.define("new", "def new():\n  wait(1)\n")


# ---- 4. a failed drive says what it read ------------------------------------


def test_a_drive_that_did_not_arrive_says_where_it_stopped_and_why():
  """...and WHY (issue #350): "stopped 9.1 m short of (22, 3)" was a route
  the planner could not make, read by Rowan as the pack running short."""
  asked = []
  life = SimpleNamespace(
    body=SimpleNamespace(go_to_routine=lambda x, y, timeout: tick.result(False),
                         in_sight=lambda x, y: True, pose=(1.0, 2.0, 0.0)),
    drive_why=lambda x, y: (asked.append((x, y)), "the drive gave up (why)")[1])
  verdict = tick.run(SimpleNamespace(step=lambda *a: None),
                     st._drive_to(life, {"x": 4.0, "y": 6.0}))
  assert verdict["reason"] == "did not arrive at (4, 6): the drive gave up (why), at (1.0, 2.0)"
  assert verdict["why"] == "the drive gave up (why)" and asked == [(4.0, 6.0)]


# ---- 5. a failed step leaves its trace in the log --------------------------


def test_the_trace_measures_the_belief_against_the_true_pose():
  """A walk's evidence says how far the belief had drifted from the truth
  when it gave up (a find's trace, a drift row): the belief minus the true
  pose, mm and degrees, the heading's difference wrapped."""
  me = SimpleNamespace(true_pose=lambda: (1.0, 1.0, 3.1), pose=(1.012, 0.995, -3.1))
  dx, dy, dyaw = nav.Navigator.truth_error(me)
  assert (dx, dy) == pytest.approx((12.0, -5.0))
  assert dyaw == pytest.approx(round(math.degrees(2 * math.pi - 6.2), 2))   # not -355


def test_a_failed_step_puts_its_trace_in_the_log_and_never_the_status():
  """`_say`'s message is the robot's status line on the site -- a sentence
  it could say; a step's trace is evidence, `detail`, the log's alone."""
  life = stub_life()
  life.body.find_tag_routine = lambda tag, near, patience, stop=None: tick.result({
    "tag": tag, "found": False, "why": "out of time", "seconds": 12.0,
    "trail": [{"to": "address", "goal": (1.0, 1.0), "why": "gave up", "shortM": 0.4,
               "seconds": 9.0}], "err": (3.0, -12.0, 0.4)})
  said = []
  life.say_hooks.append(lambda t, msg, *a: said.append(msg))
  L = lib.Library(lc.world_facts(WORLD))
  L.define("look", "def look():\n  find(35, 1, 1)\n")
  life.run_errand(errand_from(Decision(action="procedure:look"), WORLD, library=L))
  assert any("belief off +3,-12 mm" in line for line in life.log)
  assert any("PROCEDURE look failed at step 1" in msg for msg in said)
  assert not any("belief off" in msg for msg in said)


# ---- review of #336: the library's edges -----------------------------------


def test_undefine_can_name_an_entry_that_cannot_run(tmp_path):
  """An entry marked not runnable (after a restart, or a retired tool) was
  outside the `undefine` enum -- the one thing that could free its slot --
  while the refusal told the robot to undefine it. And a procedure saved as
  `new` before the token existed loads as one that cannot run, so
  `procedure:new` keeps its one meaning and the slot can still be freed."""
  (tmp_path / f"new{lib.SUFFIX}").write_text("def new():\n  wait(1)\n")
  (tmp_path / f"broken{lib.SUFFIX}").write_text("def broken():\n  fetch('module_gone')\n")
  (tmp_path / f"ok{lib.SUFFIX}").write_text("def ok():\n  wait(1)\n")
  L = lib.Library(lc.world_facts(WORLD), root=tmp_path)
  assert set(L.names()) == {"new", "broken", "ok"} and L.runnable() == ("ok",)
  schema = _menu().schema(procedures=L.runnable(), keeps=L.names())
  props = schema["properties"]
  assert {"new", "broken", "ok"} <= set(props["undefine"]["enum"])
  assert "procedure:broken" not in props["action"]["enum"]
  assert props["action"]["enum"].count("procedure:new") == 1


def test_making_room_for_a_refused_procedure_keeps_the_one_it_would_have_replaced():
  """The refusal for a full library advises an `undefine` on the same answer
  -- and both deployed libraries were full. Applied in order, a refused
  define then cost the working procedure it was making room for (second
  review of #336); the undefine now waits for a define that goes through.
  An answer with no undefine is a plain refusal, never "kept"."""
  L = lib.Library(lc.world_facts(WORLD), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("c", "def c():\n  wait(1)\n")
  life, _ = _deciding_life(L)
  life._define(Decision(action="idle", reason="x", undefine="a",
                        define={"name": "b", "source": "def b():\n  import os\n"}))
  history = life.thoughts.read("History.md")
  assert set(L.names()) == {"a", "c"}
  assert "could not write the procedure b, so a is kept too" in history
  assert "forgot the procedure a" not in history
  life._define(Decision(action="idle", reason="x", undefine="a",
                        define={"name": "b", "source": "def b():\n  wait(2)\n"}))
  assert set(L.names()) == {"b", "c"}                    # the room was made
  life._define(Decision(action="idle", reason="x",
                        define={"name": "", "source": "def x():\n  wait(1)\n"}))
  assert "could not write the procedure :" in life.thoughts.read("History.md")
  assert "is kept" not in life.thoughts.read("History.md").splitlines()[-1]
