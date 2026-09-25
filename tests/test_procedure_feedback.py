"""The development loop a robot needs to finish a challenge (issue #264).

Read off the deployed pair: the robots wrote the tower's procedure right --
`fetch("module_claw"); pick(22); place(21); pick(20); place(22); stow()` --
and it died on its first line, five times over, because nothing told them
why; a fix could not be written either, because "undefine it first, there
is no replace" read as a turn of its own and the library filled with
`stack3b` and `weigh_cube2`. Each rule here is pinned in milliseconds: the
sentence a failed pick is said in, the fork checked before a fetch drives
anywhere, the one History line a procedure leaves, and the one-answer
replacement the prompt now promises.
"""

import re
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, errand_from, world_config
from pluggybot.mind.overseer import Decision, FIELD_INDEX, PROCEDURE_HEAD
from pluggybot.procedure import library as lib
from pluggybot.procedure import steps as st
from test_procedure import _stub_swaps


def _room_hub_life():
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  return HubLifecycle(model, mujoco.MjData(model), realtime=False, world="room_hub",
                      errand=False, battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"])


# ---- 1. one sentence for a failed pick ---------------------------------------


def _self(on_forks=(), blocked=None, hung=True):
  """A stand-in `self` for `HubLifecycle.pick_failure`: its peers (each
  holding what `on_forks` names), the peer-at-bay verdict, the bay."""
  peers = [SimpleNamespace(robot_name=name, root=f"r_{name}", module=held,
                           mission=SimpleNamespace(swap=SimpleNamespace(
                             module_state=lambda t, held=held: {"on_fork": t == held})))
           for name, held in on_forks]
  me = SimpleNamespace(
    peers=peers, peer_at_the_bay=lambda station_y: blocked, last_bay_wait=None,
    mission=SimpleNamespace(swap=SimpleNamespace(
      module_state=lambda t: {"on_fork": False, "hung": hung})))
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
                       mission=SimpleNamespace(swap=SimpleNamespace(
                         module_state=lambda t: {"on_fork": True, "hung": False})))
  said = HubLifecycle.pick_failure(me, "module_claw", 0.1, "arrived")
  assert said.startswith("module_claw came onto the fork but did not seat")


# ---- 2. the fork, before a fetch drives anywhere ---------------------------


def _fetching_life(carrying: str | None):
  drove = []

  def swap_at_bay(station, verb, module=None, tries=2):
    drove.append((verb, module))
    return tick.result("arrived")

  def state(tool):
    return {"on_fork": tool == carrying, "hung": tool != carrying}
  life = SimpleNamespace(
    rack_inventory=dict(st.TOOL_BAYS), module="", swaps_done=0,
    model=None, data=None,
    mission=SimpleNamespace(swap_at_bay_routine=swap_at_bay,
                            swap=SimpleNamespace(module_state=state,
                                                 handle=SimpleNamespace(prefix=""))),
    pick_failure=lambda tool, station, why: "the pick missed and it is still on its bay")
  return life, drove


def _fetch(life, tool):
  return tick.run(SimpleNamespace(_step_once=lambda *a: None),
                  st._fetch(life, {"tool": tool}))


def test_a_fetch_with_another_tool_on_the_fork_drives_nowhere_and_says_so():
  life, drove = _fetching_life(carrying="module_pen")
  verdict = _fetch(life, "module_claw")
  assert not verdict["ok"] and drove == []
  assert verdict["reason"] == "the fork already holds module_pen; stow it first"


def test_a_fetch_of_the_tool_already_on_the_fork_has_it(monkeypatch):
  monkeypatch.setattr(st, "module_power_contact", lambda *a, **k: True)
  life, drove = _fetching_life(carrying="module_claw")
  verdict = _fetch(life, "module_claw")
  assert verdict["ok"] and drove == [] and life.module == "module_claw"


def test_a_failed_fetch_carries_the_reason_the_errand_would_have_said():
  life, drove = _fetching_life(carrying=None)
  verdict = _fetch(life, "module_claw")
  assert drove == [("pick", "module_claw")] and not verdict["ok"]
  assert verdict["reason"] == ("could not pick up module_claw: the pick missed "
                               "and it is still on its bay")


# ---- 3. one History line per procedure the robot wrote ---------------------


def test_a_procedure_cut_short_tells_the_robot_the_line_and_the_reason(monkeypatch):
  """The deployed pair's `stack3`, cut to its first two lines: the fetch
  fails, and the robot is told where and why -- on the History it reads,
  and as `failedReason` on the wire."""
  life = _room_hub_life()
  _stub_swaps(life, monkeypatch, fetch_ok=False, hung=True)
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("stack3", 'def stack3():\n  fetch("module_claw")\n  wait(1)\n')
  life.run_errand(errand_from(Decision(action="procedure:stack3"), "room_hub", library=L))
  history = life.thoughts.read("History.md")
  assert ("the procedure stack3 did not finish -- it stopped at line 2, fetch: "
          "could not pick up module_claw: the pick missed and it is still on its "
          "bay (the fork went in and came out without it) (0 steps had run)") in history
  cut = next(e for e in events if e.get("type") == "procedure"
             and e.get("outcome") == "aborted")
  assert cut["failedLine"] == 2
  assert cut["failedReason"].startswith("could not pick up module_claw")


def test_a_procedure_that_finished_says_so(monkeypatch):
  """A robot must know its procedure ENDED to say `done` after it: round 2
  of #264's probes set `done` before the run, and the grade found one block."""
  life = _room_hub_life()
  _stub_swaps(life, monkeypatch)
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("tidy", 'def tidy():\n  fetch("module_claw")\n  stow()\n')
  life.run_errand(errand_from(Decision(action="procedure:tidy"), "room_hub", library=L))
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
  life = _room_hub_life()
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
  life = SimpleNamespace(data=SimpleNamespace(time=0.0), world="room_hub",
                         rack_inventory=None, root="pluggybot",
                         _say=lambda *a, **k: None, _remember=lambda *a, **k: None,
                         _emit=lambda e: None)
  L = lib.Library(lc.world_facts("room_hub"), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("b", "def b():\n  wait(1)\n")          # full
  life.overseer = SimpleNamespace(library=L)
  HubLifecycle._define(life, Decision(action="idle", undefine="a",
                                      define={"name": "a", "source": "def a():\n  wait(2)\n"}))
  assert "wait(2)" in L.entries["a"].source and len(L.entries) == 2


def test_a_refused_define_says_the_undefine_goes_on_the_same_answer():
  L = lib.Library(lc.world_facts("room_hub"), cap=1)
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
  return replace(Menu.for_world("home", board_book("home")), procedures=True)


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


def _deciding_life(monkeypatch, library):
  life = _room_hub_life()
  _stub_swaps(life, monkeypatch)
  life.overseer = SimpleNamespace(library=library)
  events = []
  life.on_event.append(events.append)
  return life, events


def test_procedure_new_runs_what_the_same_answer_defined_and_not_another(monkeypatch):
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("old", "def old():\n  fetch(\"module_claw\")\n  stow()\n")
  life, events = _deciding_life(monkeypatch, L)
  decision = Decision(action=PROCEDURE_NEW, reason="weigh it now",
                      define={"name": "weigh", "source": "def weigh():\n  wait(1)\n"})
  tick.run(life.mission.swap, life._after_decision_routine(decision))
  queued = [e.program.name for e in life.errands if e.program is not None]
  assert queued == ["weigh"]                 # queued for the loop, and not `old`


def test_procedure_new_runs_nothing_when_the_define_was_refused(monkeypatch):
  from pluggybot.mind.overseer import PROCEDURE_NEW
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("old", "def old():\n  wait(1)\n")
  life, events = _deciding_life(monkeypatch, L)
  life.mission._drive_routine = lambda *a, **kw: tick.result(None)
  decision = Decision(action=PROCEDURE_NEW, reason="try",
                      define={"name": "bad", "source": "def bad():\n  import os\n"})
  tick.run(life.mission.swap, life._after_decision_routine(decision))
  assert not [e for e in life.errands if e.program is not None]
  assert "ran nothing: `procedure:new`" in life.thoughts.read("History.md")


def test_a_refused_define_is_written_where_the_robot_reads(monkeypatch):
  """Narrated and put on the wire, a refusal reached everyone but the robot
  that made it -- and both deployed libraries were full, so "an `undefine`
  on the same answer makes room" was said to nobody who could act on it."""
  life, _ = _deciding_life(monkeypatch, lib.Library(lc.world_facts("room_hub")))
  life._define(Decision(action="idle", reason="x",
                        define={"name": "bad", "source": "def bad():\n  import os\n"}))
  assert "could not write the procedure bad:" in life.thoughts.read("History.md")


def test_a_one_answer_rewrite_that_is_refused_keeps_the_procedure_it_had(monkeypatch):
  """Undefine and define of one name is a swap: applied in order, a refused
  define had already deleted the procedure it was meant to improve."""
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("weigh", "def weigh():\n  wait(1)\n")
  life, _ = _deciding_life(monkeypatch, L)
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
  L = lib.Library(lc.world_facts("room_hub"))
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


def test_a_drive_that_did_not_arrive_says_where_it_stopped():
  life = SimpleNamespace(mission=SimpleNamespace(
    drive_to_routine=lambda x, y, timeout: tick.result(False), pose=(1.0, 2.0, 0.0)))
  verdict = tick.run(SimpleNamespace(_step_once=lambda *a: None),
                     st._drive_to(life, {"x": 4.0, "y": 6.0}))
  assert verdict["reason"] == "did not arrive: it stopped 5.0 m short of (4, 6), at (1.0, 2.0)"


# ---- 7. a stow puts the tool back as a pick left it, first -----------------


def test_a_stow_restores_the_carry_configuration_before_the_return(monkeypatch):
  """A return computes its release heights from the lift it STARTS at, and
  a procedure may have moved it. MEASURED: a claw stowed from 0.03 m -- where
  Luca's weighing had lowered it -- was driven into the rack and knocked to
  the floor; from the pick's height it hung. The flight is behind
  --endurance (tests/test_solutions.py); this is the order of the calls."""
  from pluggybot.tools.gripper import CLAW_MODULE, MODULE_DRIVE_LIFT
  calls = []

  def rec(name, *args):
    def make(*a, **kw):
      calls.append((name, a[0] if a else None))
      return tick.result("arrived")
    return make
  state = {"on_fork": True, "hung": False}
  swap = SimpleNamespace(module_state=lambda t: dict(state) if t == CLAW_MODULE
                         else {"on_fork": False, "hung": True},
                         set_lift_routine=rec("set_lift"), handle=SimpleNamespace(prefix=""))
  life = SimpleNamespace(rack_inventory=dict(st.TOOL_BAYS), swaps_done=0,
                         model=None, data=None, world="room_hub",   # no routes home
                         mission=SimpleNamespace(swap=swap, set_arm_routine=rec("set_arm"),
                                                 swap_at_bay_routine=rec("return"),
                                                 pose_xy=lambda: (0.0, 0.0)))
  held = SimpleNamespace(held=lambda: "block_1", set_down_routine=rec("set_down"))
  monkeypatch.setattr(st, "_claw", lambda _life: held)
  tick.run(SimpleNamespace(_step_once=lambda *a: None), st._stow(life, {}))
  assert [c[0] for c in calls] == ["set_down", "set_arm", "set_lift", "return"]
  assert ("set_arm", 0.0) in calls and ("set_lift", MODULE_DRIVE_LIFT) in calls


def test_a_pick_that_cannot_see_its_cube_says_whether_it_ever_got_there(monkeypatch):
  """Live, `pick(22)` failed "not a cube this robot can see from here" where
  the same six lines stacked the tower locally, and the line could not say
  whether the route to the cube never arrived or arrived and the tag did not
  decode -- two different things to fix."""
  claw = SimpleNamespace(calibrate_from_body=lambda: None,
                         tuck_routine=lambda: tick.result(None))
  life = SimpleNamespace(world="home", mission=SimpleNamespace(
    pose_xy=lambda: (-6.0, 1.0), face_routine=lambda h: tick.result(True)))
  monkeypatch.setattr(st, "_spot_routine", lambda life, tag: tick.result(None))
  stepper = SimpleNamespace(_step_once=lambda *a: None)
  life.mission.drive_to_routine = lambda x, y, timeout: tick.result(False)
  seen, _, unseen = tick.run(stepper, st._approach_routine(life, claw, 22, carrying=False))
  assert seen is None
  assert unseen.endswith("the route to where the house set it out stopped at (-6.0, 1.0)")
  life.mission.drive_to_routine = lambda x, y, timeout: tick.result(True)
  seen, _, unseen = tick.run(stepper, st._approach_routine(life, claw, 22, carrying=False))
  assert seen is None and unseen.startswith(
    "tag 22 did not decode even from where the house set it out, by (-11.00, -4.50)")


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
  trace says how far the belief had drifted from the truth when it tried."""
  life = _room_hub_life()
  m = life.mission
  m.start_at(1.0, 1.0, 0.5)
  assert m.truth_error() == pytest.approx([0.0, 0.0, 0.0], abs=0.2)
  m.swap.reckoner.x += 0.012
  assert m.truth_error()[0] == pytest.approx(12.0, abs=0.2)


def test_a_missed_pick_is_measured_where_the_approach_ended():
  """The trace's offset is the one a miss is ABOUT: the module against the
  fork before the lift, in the robot's frame. A fork 30 mm to the right of
  the peg -- outside the +/-11 mm capture window -- reads the module 30 mm
  LEFT and at the fork, and it is read before the lift's first step; read
  after the retreat (as it first was), a miss put the fork 0.35 m away."""
  from pluggybot.rack.coupling import HUB_STATION_YS
  from pluggybot.rack.swap import HubSwap
  from pluggybot.control import wheel_targets
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  swap = HubSwap(model, mujoco.MjData(model))
  swap.place_at_standoff(HUB_STATION_YS[0], dy=0.03)
  for cmd in swap.pick_routine(module="module_lcd"):
    swap._step_once(*wheel_targets(*cmd))
    if swap.approach_end is not None:
      break                                   # the claim is settled here
  else:
    pytest.fail("the pick returned without measuring its approach")
  ahead, left = swap.approach_end
  assert 0.022 < left < 0.038, left
  assert abs(ahead) < 0.05, ahead


def test_a_failed_pick_puts_its_trace_in_the_log_and_never_the_status():
  """`_say`'s message is the robot's status line on the site -- a sentence
  it could say; the trace is evidence, `detail`, the log's alone."""
  from pluggybot.mission.errand import Errand
  from pluggybot.mission.mission import swap_trace
  from pluggybot.rack.coupling import HUB_STATION_YS
  rec = {"verb": "pick", "route": "ok", "attempts": [
    {"fix": "plane:2", "err": [3.0, -12.4, -1.4], "travel": 0.187, "why": "arrived",
     "moduleFromForkMm": [2.0, 15.1]}]}
  assert swap_trace(rec) == ("route ok; #1 fix plane:2, belief off +3,-12 mm -1.4 deg, "
                             "travel 0.187 m -> arrived, module from fork +2 ahead +15 left mm")
  life = _room_hub_life()

  def failed(*a, **kw):
    life.mission.last_swap = rec
    return tick.result("arrived")
  life.mission.swap_at_bay_routine = failed
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": True}
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(True)
  said = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  life.run_errand(Errand(name="carry:test", module="module_lcd",
                         station_y=HUB_STATION_YS[0], use_at=(1.0, 1.0),
                         use=lambda _l: {}, needs_use_pose=False))
  assert any("belief off +3,-12 mm" in line for line in life.log)
  assert not any("belief off" in msg for msg in said)


# ---- a stow from across the street comes home by the street ----------------


def test_home_route_is_a_zones_route_reversed_from_where_the_robot_stands():
  """Ladder B on the bench (2026-09-24): a weighing failed in the lab, the
  stow's single drive home across the street failed twice, and the claw
  stayed on the fork and was lost at the garden door. The way home starts at
  the door the robot's ZONE is behind (`HOME_FROM`): by straight line, the
  lobby was sent to the lab's door, the ring outside the facility into it,
  and the south street nowhere at all (both reviews of #336)."""
  lab = lc.lab_route("home")
  assert lc.home_route("home", (27.0, 1.5)) == lab[::-1]              # the lab
  assert lc.home_route("home", (20.5, 0.0)) == lab[3::-1]             # the lobby
  assert lc.home_route("home", (25.0, -3.0)) == lab[3::-1]            # the store
  assert lc.home_route("home", (17.5, 2.0)) == lab[2::-1]             # garden_2
  for outside in ((13.0, 3.1), (28.8, 0.0), (22.0, 6.8), (8.0, -9.0)):
    assert lc.home_route("home", outside) == lab[1::-1], outside      # the gate first
  assert lc.home_route("home", (7.5, 1.3)) == [lab[0]]                # the garden
  rack = world_config("home")["start"][:2]
  for inside in (rack, (-8.5, 4.0), (-8.5, -3.0)):                     # the house
    assert lc.home_route("home", inside) == [], inside
  assert lc.home_route("room_hub", (1.0, 1.0)) == []

def test_both_stows_drive_the_route_home_before_the_swap(monkeypatch):
  """`stow()` and the stow after a procedure both come home by the route
  first: the legs, in order, and then the bay."""
  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = HubLifecycle(model, mujoco.MjData(model), realtime=False, world="home",
                      errand=False, battery_wh=8.0, rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"], low_battery_wh=cfg["low_battery_wh"])
  _stub_swaps(life, monkeypatch)
  trips = []
  real_swap = life.mission.swap_at_bay_routine

  def drive(x, y, timeout=None):
    trips.append(("drive", round(x, 2), round(y, 2)))
    return tick.result(True)

  def swap(station, verb, module=None, tries=2):
    trips.append((verb,))
    return real_swap(station, verb, module=module, tries=tries)
  life.mission.drive_to_routine = drive
  life.mission.swap_at_bay_routine = swap
  legs = [("drive", round(x, 2), round(y, 2)) for x, y in lc.lab_route("home")[::-1]]
  L = lib.Library(lc.world_facts("home"))
  L.define("fetch_only", 'def fetch_only():\n  fetch("module_claw")\n')
  L.define("fetch_stow", 'def fetch_stow():\n  fetch("module_claw")\n  stow()\n')
  for name in ("fetch_stow", "fetch_only"):
    trips.clear()
    life.mission.start_at(27.0, 1.5, 0.0)
    life.run_errand(errand_from(Decision(action=f"procedure:{name}"), "home", library=L))
    back = trips[trips.index(("pick",)) + 1:]
    assert back[:len(legs)] == legs and back[len(legs)] == ("return",), (name, back)


def test_a_drive_that_stalled_short_of_the_stand_is_not_a_look_from_it():
  """Review of #336: a failed drive to the final stand still reported a look
  "from where the house set it out" when it had stopped metres short. It
  still faces the cube and LOOKS from where it stopped -- the cube may be in
  view, as it always was (second review) -- and a failed look says where it
  was taken from. A drive that stagnates centimetres out is there."""
  _, _, stand, heading = st.prop_stand("home", 21)

  def travel(dx):
    faced = []
    life = SimpleNamespace(world="home", mission=SimpleNamespace(
      pose_xy=lambda: (stand[0] + dx, stand[1]),
      drive_to_routine=lambda x, y, timeout: tick.result((x, y) != stand),
      face_routine=lambda h: (faced.append(h), tick.result(True))[1]))
    went, why = tick.run(SimpleNamespace(_step_once=lambda *a: None), st._travel_routine(life, 21))
    return went, why, faced
  went, why, faced = travel(2.0)
  assert went and faced == [heading]
  assert why.startswith("and the drive to where the house set it out stopped 2.0 m short of it")
  assert travel(0.3) == (True, "", [heading])

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
  L = lib.Library(lc.world_facts("room_hub"), root=tmp_path)
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
  peer = SimpleNamespace(module="probe", mission=SimpleNamespace(
    swap=SimpleNamespace(module_state=gone)))
  assert lc.carrying(peer) == ""
  # ...and it is the FORK that is read, not the module a robot was last sent
  # for: a failed stow of the claw, then an errand for the pen
  riding = SimpleNamespace(module="module_pen", mission=SimpleNamespace(swap=SimpleNamespace(
    module_state=lambda t: {"on_fork": t == "module_claw"})))
  assert lc.carrying(riding) == "module_claw"


def test_making_room_for_a_refused_procedure_keeps_the_one_it_would_have_replaced(monkeypatch):
  """The refusal for a full library advises an `undefine` on the same answer
  -- and both deployed libraries were full. Applied in order, a refused
  define then cost the working procedure it was making room for (second
  review of #336); the undefine now waits for a define that goes through.
  An answer with no undefine is a plain refusal, never "kept"."""
  L = lib.Library(lc.world_facts("room_hub"), cap=2)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("c", "def c():\n  wait(1)\n")
  life, _ = _deciding_life(monkeypatch, L)
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
