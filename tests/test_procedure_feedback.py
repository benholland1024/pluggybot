"""The development loop a robot needs to finish a challenge (issue #264):
the sentence a failed pick is said in, the fork checked before a fetch
drives anywhere, the one History line a procedure leaves, and the
one-answer replacement the prompt promises -- each rule pinned in
milliseconds.
"""

import re
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, errand_from
from pluggybot.mind.overseer import Decision, FIELD_INDEX, PROCEDURE_HEAD
from pluggybot.procedure import library as lib
from pluggybot.procedure import steps as st
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_procedure import _stub_swaps

WORLD = "home_quad"


# ---- 1. one sentence for a failed pick ---------------------------------------


def _self(on_forks=(), blocked=None, hung=True):
  """A stand-in `self` for `HubLifecycle.pick_failure`: its peers (each
  holding what `on_forks` names), the peer-at-bay verdict, the bay."""
  peers = [SimpleNamespace(robot_name=name, root=f"r_{name}", module=held,
                           body=SimpleNamespace(
                             module_state=lambda t, held=held: {"on_fork": t == held}))
           for name, held in on_forks]
  me = SimpleNamespace(
    peers=peers, peer_at_the_bay=lambda station_y: blocked, last_bay_wait=None,
    body=SimpleNamespace(module_state=lambda t: {"on_fork": False, "hung": hung}))
  me.held_for = lambda b: HubLifecycle.held_for(me, b)
  return me


def test_a_failed_pick_names_who_holds_the_tool_before_anything_else():
  """Carrying is the others' PUBLIC surface, and "it was not on its bay"
  sent Luca guessing ("Suspect Rowan took the pen") at what code knew."""
  me = _self(on_forks=[("Rowan", "module_claw")], blocked=("Rowan", 0.2))
  assert (HubLifecycle.pick_failure(me, "module_claw", 0.1, "arrived")
          == "module_claw is on Rowan's fork")


def test_a_failed_pick_tells_a_miss_from_an_approach_that_never_got_there():
  miss = HubLifecycle.pick_failure(_self(), "module_pen", 0.1, "arrived")
  assert miss == ("the pick missed and it is still on its bay "
                  "(the fork went in and came out without it)")
  stall = HubLifecycle.pick_failure(_self(), "module_pen", 0.1, "stalled")
  assert "stalled" in stall and stall.startswith("the pick missed")
  never = HubLifecycle.pick_failure(_self(), "module_pen", 0.1, "no-route")
  assert "no pick was tried" in never and "missed" not in never
  # ...and a lane the refine could not get back into (#339's budget)
  held_off = HubLifecycle.pick_failure(_self(), "module_pen", 0.1, "blocked")
  assert "blocked, so no pick was tried" in held_off and "missed" not in held_off
  blocked = HubLifecycle.pick_failure(_self(blocked=("Rowan", 0.14)),
                                      "module_pen", 0.1, "peer-at-bay")
  assert blocked.startswith("Rowan was standing 0.14 m from the bay")
  lost = HubLifecycle.pick_failure(_self(hung=False), "module_pen", 0.1, "arrived")
  assert lost == "it was not on its bay, and no robot is carrying it"


def test_a_tool_that_came_onto_this_fork_unseated_is_not_called_lost():
  """A half-seated pick: on this robot's own fork, no power contact. It fell
  through every case to "not on its bay, and no robot is carrying it"."""
  me = SimpleNamespace(peers=[], peer_at_the_bay=lambda station_y: None,
                       body=SimpleNamespace(
                         module_state=lambda t: {"on_fork": True, "hung": False}))
  said = HubLifecycle.pick_failure(me, "module_claw", 0.1, "arrived")
  assert said.startswith("module_claw came onto the fork but did not seat")


# ---- 2. the fork, before a fetch drives anywhere ---------------------------


def _fetching_life(carrying: str | None, seated: bool = False):
  drove = []

  def fetch(station, module):
    drove.append(("pick", module))
    return tick.result("arrived")

  def state(tool):
    return {"on_fork": tool == carrying, "hung": tool != carrying}
  life = SimpleNamespace(
    rack_inventory=dict(st.TOOL_BAYS), module="", swaps_done=0,
    model=None, data=None,
    body=SimpleNamespace(fetch_tool_routine=fetch, module_state=state,
                         tool_powered=lambda tool: seated,
                         swap_trace=lambda: "no swap recorded"),
    pick_failure=lambda tool, station, why: "the pick missed and it is still on its bay")
  return life, drove


def _fetch(life, tool):
  return tick.run(SimpleNamespace(step=lambda *a: None),
                  st._fetch(life, {"tool": tool}))


def test_a_fetch_with_another_tool_on_the_fork_drives_nowhere_and_says_so():
  life, drove = _fetching_life(carrying="module_pen")
  verdict = _fetch(life, "module_claw")
  assert not verdict["ok"] and drove == []
  assert verdict["reason"] == "the fork already holds module_pen; stow it first"


def test_a_fetch_of_the_tool_already_on_the_fork_has_it():
  life, drove = _fetching_life(carrying="module_claw", seated=True)
  verdict = _fetch(life, "module_claw")
  assert verdict["ok"] and drove == [] and life.module == "module_claw"


def test_a_failed_fetch_carries_the_reason_the_errand_would_have_said():
  life, drove = _fetching_life(carrying=None)
  verdict = _fetch(life, "module_claw")
  assert drove == [("pick", "module_claw")] and not verdict["ok"]
  assert verdict["reason"] == ("could not pick up module_claw: the pick missed "
                               "and it is still on its bay")


# ---- 3. one History line per procedure the robot wrote ---------------------


def test_a_procedure_cut_short_tells_the_robot_the_line_and_the_reason():
  """The deployed pair's `stack3`, cut to its first two lines: the fetch
  fails, and the robot is told where and why -- on the History it reads,
  and as `failedReason` on the wire."""
  life = stub_life()
  _stub_swaps(life, fetch_ok=False, hung=True)
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts(WORLD))
  L.define("stack3", 'def stack3():\n  fetch("module_claw")\n  wait(1)\n')
  life.run_errand(errand_from(Decision(action="procedure:stack3"), WORLD, library=L))
  history = life.thoughts.read("History.md")
  assert ("the procedure stack3 did not finish -- it stopped at line 2, fetch: "
          "could not pick up module_claw: the pick missed and it is still on its "
          "bay (the fork went in and came out without it) (0 steps had run)") in history
  cut = next(e for e in events if e.get("type") == "procedure"
             and e.get("outcome") == "aborted")
  assert cut["failedLine"] == 2
  assert cut["failedReason"].startswith("could not pick up module_claw")


def test_a_procedure_that_finished_says_so():
  """A robot must know its procedure ENDED to say `done` after it: round 2
  of #264's probes set `done` before the run, and the grade found one block."""
  life = stub_life()
  L = lib.Library(lc.world_facts(WORLD))
  L.define("tidy", 'def tidy():\n  fetch("module_claw")\n  stow()\n')
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

# ---- 4. a replacement is one answer, as the prompt says --------------------


def test_one_answer_replaces_a_procedure_and_the_prompt_says_it_can():
  """The lifecycle has always applied `undefine` before `define`; the robot
  was told "there is no replace -- undefine, then define" and read it as two
  turns. The claim and the behaviour are pinned together."""
  assert "no replace" not in PROCEDURE_HEAD
  assert "on\nthe same answer" in PROCEDURE_HEAD or "same answer" in PROCEDURE_HEAD
  assert any(name == "undefine" and "same answer" in text
             for name, _, _, text in FIELD_INDEX)
  life = SimpleNamespace(data=SimpleNamespace(time=0.0), world=WORLD,
                         rack_inventory=None, root="pluggybot",
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


# ---- 5. a procedure is run on the answer that defines it -------------------


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
  L.define("old", "def old():\n  fetch(\"module_claw\")\n  stow()\n")
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


# ---- 6. a failed grade and a failed drive say what they read ---------------


def test_a_bench_grade_with_no_finding_says_what_it_reads():
  """Luca wrote its 207.8 g as a NOTE under findings/tasks; the grade said
  "nothing since the claim" and Luca learned a timing lesson instead."""
  from pluggybot.challenge import bench
  ok, _, reason = bench.eval_mass({"truth": 0.2, "reported": None})
  assert not ok
  assert "`record`" in reason and "mass_bench" in reason and "notes are not read" in reason


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


# ---- 7. a stow puts the tool back as a pick left it, first -----------------


def test_a_stow_restores_the_carry_configuration_before_the_return():
  """A return computes its approach from the pose it STARTS at, and a
  procedure may have moved the arm anywhere: the arm goes back to its
  carrying pose first. This is the order of the calls."""
  calls = []

  def rec(name):
    def make(*a, **kw):
      calls.append(name)
      return tick.result("arrived")
    return make
  body = SimpleNamespace(module_state=lambda t: {"on_fork": t == "module_claw",
                                                 "hung": t != "module_claw"},
                         retract_arm_routine=rec("retract_arm"),
                         stow_tool_routine=rec("return"),
                         swap_trace=lambda: "no swap recorded", held_cube=lambda: None)
  life = SimpleNamespace(rack_inventory=dict(st.TOOL_BAYS), swaps_done=0,
                         model=None, world=WORLD, body=body)
  tick.run(SimpleNamespace(step=lambda *a: None), st._stow(life, {}))
  assert calls == ["retract_arm", "return"]


def test_a_cube_in_the_claw_is_set_down_before_the_claw_goes_back():
  """A claw hung back holding a cube is a cube hung on the rack (#407): a
  stow puts it down first, where the robot stands. Shown to fail by dropping
  the check from `carry_configuration_routine`: the claw went back full."""
  calls, held = [], [23]

  def rec(name):
    def make(*a, **kw):
      calls.append(name)
      if name == "put":
        held.clear()
        return tick.result({"held": 23})
      return tick.result("arrived")
    return make
  body = SimpleNamespace(module_state=lambda t: {"on_fork": t == "module_claw",
                                                 "hung": t != "module_claw"},
                         put_cube_routine=rec("put"), retract_arm_routine=rec("retract_arm"),
                         stow_tool_routine=rec("return"), swap_trace=lambda: "",
                         held_cube=lambda: held[0] if held else None)
  life = SimpleNamespace(rack_inventory=dict(st.TOOL_BAYS), swaps_done=0,
                         model=None, world=WORLD, body=body)
  tick.run(SimpleNamespace(step=lambda *a: None), st._stow(life, {}))
  assert calls == ["put", "retract_arm", "return"]


def test_a_refused_build_says_what_is_in_the_way(monkeypatch):
  """A deployed robot holding a claw it could not stow was told only "never
  mid-errand", and specified its tool twice more: the refusal names the
  module on the fork, and whose."""
  from pluggybot.procedure import steps
  me = SimpleNamespace(peers=[], state="DECIDE", robot_name="Luca", root="pluggybot",
                       MID_ERRAND=HubLifecycle.MID_ERRAND)
  monkeypatch.setattr(steps, "_carried", lambda life: "module_claw")
  assert HubLifecycle.seam_busy(me).startswith("your fork holds module_claw: stow it first")
  me.peers = [SimpleNamespace(state="SWAP_PICK", robot_name="Rowan", root="r2_pluggybot")]
  monkeypatch.setattr(steps, "_carried", lambda life: None)
  assert HubLifecycle.seam_busy(me).startswith("Rowan is busy (it is mid-errand)")


# ---- 8. a failed swap leaves its trace in the log --------------------------


def test_the_trace_measures_the_belief_against_the_true_pose():
  """Every live pick missed for a reason no local reproduction showed; the
  trace says how far the belief had drifted from the truth when it tried --
  on the served body, its reckoning against its torso."""
  import mujoco
  from pluggybot.legs import body as qb
  from pluggybot.legs.world import home_spec
  model = home_spec().compile()
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=lc.world_config(WORLD)["grid_bounds"])
  m = body.mission
  m.start_at(1.0, 1.0, 0.5)
  start = m.truth_error()
  assert start == pytest.approx([0.0, 0.0, 0.0], abs=5.0)
  m.odo.x += 0.012
  assert m.truth_error()[0] - start[0] == pytest.approx(12.0, abs=0.2)


def test_a_failed_pick_puts_its_trace_in_the_log_and_never_the_status():
  """`_say`'s message is the robot's status line on the site -- a sentence
  it could say; the trace is evidence, `detail`, the log's alone."""
  from pluggybot.mission.errand import Errand
  from pluggybot.rack.coupling import STATION_YS
  life = stub_life()
  _stub_swaps(life, fetch_ok=False)
  life.body.swap_trace = lambda: "route ok; #1 fix plane:2, belief off +3,-12 mm"
  said = []
  life.say_hooks.append(lambda t, msg, *a: said.append(msg))
  life.run_errand(Errand(name="carry:test", module="module_lcd",
                         station_y=STATION_YS[0], use_at=(1.0, 1.0),
                         use=lambda _l: {}, needs_use_pose=False))
  assert any("belief off +3,-12 mm" in line for line in life.log)
  assert any("SWAP_PICK FAILED" in msg for msg in said)
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


def test_a_retired_tool_on_a_peers_record_is_carried_by_nobody():
  """`carrying` reads a peer's last module; a built tool since retired is
  no body in the world, and the lookup raised where `_carried` has always
  guarded it -- now reached from every failed pick's sentence, too."""
  def gone(name):
    raise KeyError(name)
  peer = SimpleNamespace(module="probe", body=SimpleNamespace(module_state=gone))
  assert lc.carrying(peer) == ""
  # ...and it is the FORK that is read, not the module a robot was last sent
  # for: a failed stow of the claw, then an errand for the pen
  riding = SimpleNamespace(module="module_pen", body=SimpleNamespace(
    module_state=lambda t: {"on_fork": t == "module_claw"}))
  assert lc.carrying(riding) == "module_claw"


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
