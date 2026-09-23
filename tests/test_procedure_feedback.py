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
  return SimpleNamespace(
    peers=peers, peer_at_the_bay=lambda station_y: blocked,
    mission=SimpleNamespace(swap=SimpleNamespace(
      module_state=lambda t: {"on_fork": False, "hung": hung})))


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
  blocked = HubLifecycle.pick_failure(_self(blocked=("Rowan", 0.14)),
                                      "module_pen", 0.1, "peer-at-bay")
  assert blocked.startswith("Rowan was standing 0.14 m from the bay")
  lost = HubLifecycle.pick_failure(_self(hung=False), "module_pen", 0.1, "arrived")
  assert lost == "it was not on its bay, and no robot is carrying it"


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
  assert ("ran the procedure stack3 (0/1 steps) -- it stopped at line 2, fetch: "
          "could not pick up module_claw: the pick missed and it is still on its "
          "bay (the fork went in and came out without it)") in history
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
  assert re.search(r"ran the procedure tidy \(2/2 steps\)$", history, re.M)


def test_a_procedure_out_of_budget_says_so_after_its_last_line():
  run = {"ok": False, "completed": 3, "total": 3, "stopped": "budget",
         "steps": [{"i": 2, "verb": "pick", "line": 5, "ok": True}]}
  assert (lc.procedure_outcome("stack", run)
          == "ran the procedure stack (3/3 steps) -- it ran out of the time its "
             "budget gave it after line 5")


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
  from pluggybot.mind.overseer import PROCEDURE_NEW
  menu = _menu()
  with pytest.raises(ValueError, match="defines none"):
    menu.validate({"action": PROCEDURE_NEW, "reason": "x"}, procedures=())
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
                         model=None, data=None,
                         mission=SimpleNamespace(swap=swap, set_arm_routine=rec("set_arm"),
                                                 swap_at_bay_routine=rec("return")))
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
