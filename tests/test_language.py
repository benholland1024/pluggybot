"""The procedure language (procedure/lang.py, axes.py, library.py; issue
#166): parsed never executed, total by construction, the robot's own
library, and the `autonomous` arm's vocabulary for it.
"""

import ast
import inspect
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.lifecycle import HubLifecycle, errand_from, world_config, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Decision, Menu, Overseer
from pluggybot.mission.mission import HubMission
from pluggybot.procedure import axes, lang, library as lib, steps as st
from pluggybot.procedure.steps import Refused

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
HOME = world_facts("home")
HUB = world_facts("room_hub")

LOOK_AROUND = '''def look_around():
    budget(steps=40, seconds=240)
    n = 0
    fetch("module_lcd")
    for i in range(2):
        drive(0.0, 0.8, 1.5)
        look()
        if read("look.tag") >= 0 and read("look.range") < 1.5:
            wait(2)
        elif read("look.tag") < 0:
            pass
        else:
            wait(1)
    while read("arm") < 0.04 and n < 8:
        move("arm", read("arm") + 0.01)
        n += 1
    stow()
'''


def compile_ok(src, facts=HOME):
  return lang.compile_procedure(src, facts)


def refusals(src, facts=HOME):
  with pytest.raises(Refused) as e:
    lang.compile_procedure(src, facts)
  return e.value.reasons


# ---- the parser: a construct outside the grammar is refused, with its line ---


def test_the_worked_example_compiles():
  p = compile_ok(LOOK_AROUND)
  assert p.name == "look_around" and p.verbs == 7
  assert p.steps_budget == 40 and p.budget_s == 240.0
  assert p.first("fetch", "tool") == "module_lcd"


@pytest.mark.parametrize("src, needle", [
  ("look()\n", "exactly one `def name():`"),
  ("def a():\n  look()\ndef b():\n  look()\n", "exactly one `def"),
  ("def A():\n  look()\n", "not a name this language allows"),
  ("def x(y):\n  look()\n", "takes no arguments"),
  ("@dec\ndef x():\n  look()\n", "takes no arguments and no decorators"),
  ("def x():\n  pass\n", "calls no verb"),
  ("def x():\n  \"\"\"only a docstring\"\"\"\n", "has no steps"),
  ("def x():\n  import os\n", "line 2: Import is not part"),
  ("def x():\n  print(1)\n", "unknown verb 'print'"),
  ("def x():\n  fly()\n", "unknown verb 'fly'"),
  ("def x():\n  read(\"arm\")\n", "read(...) is an expression"),
  ("def x():\n  wait(look())\n", "only read(...) may be used inside an expression"),
  ("def x():\n  a = \"s\"\n", "a str is not a number"),
  ("def x():\n  a = [1]\n", "List is not part"),
  ("def x():\n  a = b.c\n", "Attribute is not part"),
  ("def x():\n  a = 1 < 2 < 3\n", "compare two things with one of"),
  ("def x():\n  for i in [1, 2]:\n    look()\n", "`for name in range(N)`"),
  ("def x():\n  for i in range(n):\n    look()\n", "literal N"),
  ("def x():\n  for i in range(1000):\n    look()\n", "outside 0..100"),
  ("def x():\n  while 1:\n    look()\n  else:\n    look()\n", "while takes no else"),
  ("def x():\n  return 1\n", "return takes no value"),
  ("def x():\n  wait(1, 2)\n", "wait takes 1 argument"),
  ("def x():\n  wait(seconds=1, z=2)\n", "wait takes no z"),
  ("def x():\n  fetch()\n", "fetch needs tool"),
  ("def x():\n  fetch(3)\n", "fetch(tool=...) takes a name in quotes"),
  ("def x():\n  wait(\"long\")\n", "wait(seconds=...) takes a number"),
  ("def x():\n  wait(999)\n", "seconds=999.0 is above 60"),
  ("def x():\n  move(\"lift\", 5)\n", "lift target 5.0 is outside"),
  ("def x():\n  move(\"warp\", 1)\n", "axis='warp' is not one of"),
  ("def x():\n  fetch(\"module_x\")\n", "tool='module_x' is not one of"),
  ("def x():\n  a = read(\"nope\")\n", "unknown sensor 'nope'"),
  ("def x():\n  drive_to(99, 0)\n", "outside the map"),
  ("def x():\n  budget(steps=999)\n  look()\n", "steps=999 is outside 1..200"),
  ("def x():\n  budget(seconds=0)\n  look()\n", "seconds=0.0 is outside"),
  ("def x():\n  budget(40)\n  look()\n", "budget takes keywords"),
  ("def x():\n  budget(steps=\"many\")\n  look()\n", "not a literal steps or seconds"),
  ("def x():\n  a = 1 +\n", "invalid syntax"),
  ("def x():\n  if 1:\n    if 1:\n      if 1:\n        if 1:\n          look()\n",
   "nested deeper than"),
  ("def x():\n  read = 3\n  look()\n", "reserved"),
  ("def x():\n  wait = 3\n  look()\n", "reserved"),
])
def test_each_construct_outside_the_grammar_is_refused(src, needle):
  reasons = refusals(src)
  assert any(needle in r for r in reasons), reasons


def test_the_source_cap_is_a_refusal():
  src = "def x():\n" + "  look()\n" * 400
  assert any("characters; the cap is" in r for r in refusals(src))


def test_a_draw_in_a_world_without_boards_is_refused_by_the_world():
  reasons = refusals('def x():\n  draw("sun", "whiteboard_a")\n', HUB)
  assert any("nothing this world has" in r for r in reasons)


def test_every_reason_is_reported_at_once():
  reasons = refusals("def x():\n  fly()\n  wait(999)\n  a = [1]\n")
  assert len(reasons) == 3


def test_the_language_never_executes_its_input():
  """Parsed, never run: no exec/eval/compile anywhere in the module."""
  src = inspect.getsource(lang)
  tree = ast.parse(src)
  names = {n.func.id for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
  assert not names & {"exec", "eval", "compile", "__import__"}


# ---- the interpreter, on a stubbed life ---------------------------------------


def _stub_life():
  calls: list = []

  def routine(name, value=True):
    def make(*a, **kw):
      calls.append((name, a))
      return tick.result(value)
    return make
  swap = SimpleNamespace(module_state=lambda tool: {"on_fork": False, "hung": True},
                         set_lift_routine=routine("set_lift"),
                         ramp_routine=routine("ramp"), pressing=False)
  model = SimpleNamespace(actuator=lambda name: SimpleNamespace(id=0))
  mission = SimpleNamespace(
    swap=swap, swap_at_bay_routine=routine("swap", "arrived"),
    drive_to_routine=routine("drive_to", True), face_routine=routine("face", True),
    _drive_routine=routine("drive"), pose=(0.0, 0.0, 0.0),
    tags=SimpleNamespace(detect=lambda d: {}))
  return SimpleNamespace(mission=mission, data=SimpleNamespace(time=0.0),
                         module="", swaps_done=0, interrupted=lambda: False,
                         _say=lambda *a, **k: None, calls=calls, model=model,
                         world="home", boards=None, ledger=None,
                         battery=SimpleNamespace(fraction=0.5, energy_wh=4.0))


def run(src, life=None, facts=HOME):
  life = life or _stub_life()
  proc = lang.compile_procedure(src, facts)
  r = tick.run(SimpleNamespace(_step_once=lambda *a: None),
               lang.run_procedure_routine(life, proc, facts))
  return r, life


def test_locals_arithmetic_and_branches_run_as_written():
  src = '''def x():
    a = 2
    b = a * 3 + 1
    if b > 10:
        wait(3)
    elif b == 7:
        wait(1)
    else:
        wait(2)
    a -= 1
    if not a == 1 or b < 0:
        wait(9)
    for i in range(2):
        wait(i + 0.5)
    return
    wait(5)
'''
  r, life = run(src)
  assert r["ok"] and r["completed"] == 3
  assert [c[1][0] for c in life.calls] == [1.0, 0.5, 1.5]


def test_a_while_reads_the_world_each_time_round():
  life = _stub_life()
  life.battery.fraction = 0.0

  def wait(seconds, *a, **kw):
    life.battery.fraction += 0.25
    return tick.result(None)
  life.mission._drive_routine = wait
  r, _ = run('def x():\n  while read("battery.frac") < 1:\n    wait(1)\n', life)
  assert r["ok"] and r["completed"] == 4


def test_a_computed_argument_is_checked_when_it_is_computed():
  r, _ = run('def x():\n  a = 30\n  wait(a * 3)\n')
  assert not r["ok"] and r["failedAt"] == 0
  assert "seconds=90.0 is above 60" in r["steps"][0]["reason"]


def test_a_failed_step_ends_the_procedure_honestly():
  life = _stub_life()
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  r, _ = run('def x():\n  wait(1)\n  drive_to(1, 1)\n  wait(1)\n', life)
  assert not r["ok"] and r["completed"] == 1 and r["failedAt"] == 1
  assert r["total"] == 2 and len(r["steps"]) == 2


@pytest.mark.parametrize("src, why", [
  ('def x():\n  a = 1 / 0\n  wait(1)\n', "division by zero"),
  ('def x():\n  wait(b)\n', "'b' was read before it was set"),
  ('def x():\n  n = 0\n  while 1:\n    n += 1\n  look()\n', "loop-cap"),
  ('def x():\n  while 1:\n    wait(0.1)\n', "steps"),
])
def test_an_evaluation_fault_and_a_runaway_loop_stop_the_procedure(src, why):
  r, _ = run(src)
  assert not r["ok"] and r.get("stopped", r["steps"][-1].get("reason")) == why \
    or (not r["ok"] and any(why in s.get("reason", "") for s in r["steps"]))


def test_the_step_budget_stops_at_a_step_boundary():
  r, life = run('def x():\n  budget(steps=3)\n  for i in range(10):\n    wait(1)\n')
  assert r["stopped"] == "steps" and r["completed"] == 3 and not r["ok"]
  assert len(life.calls) == 3


def test_the_time_budget_stops_at_a_step_boundary():
  life = _stub_life()

  def slow(seconds, *a, **kw):
    life.data.time += 100.0
    return tick.result(None)
  life.mission._drive_routine = slow
  r, _ = run('def x():\n  budget(seconds=150)\n  for i in range(5):\n    wait(1)\n',
             life)
  assert r["stopped"] == "budget" and r["completed"] == 2


def test_an_interrupt_stops_between_verbs():
  life = _stub_life()
  life.interrupted = lambda: True
  r, _ = run('def x():\n  look()\n  look()\n', life)
  assert r["stopped"] == "interrupted" and r["completed"] == 1


def test_move_goes_through_the_swaps_ramp_and_refuses_a_missing_tool():
  r, life = run('def x():\n  move("lift", 0.2)\n  move("pen.carriage", 0.01)\n')
  assert r["steps"][0]["ok"] and [c[0] for c in life.calls] == ["ramp"]
  assert not r["steps"][1]["ok"]
  assert r["steps"][1]["reason"] == "axis 'pen.carriage' needs module_pen on the fork"


def test_a_sensor_that_needs_a_tool_fails_without_it():
  r, _ = run('def x():\n  a = read("pen.contact")\n  wait(1)\n')
  assert not r["ok"] and "needs module_pen on the fork" in r["steps"][-1]["reason"]


# ---- the fence, extended to loops ------------------------------------------------


def test_the_language_modules_write_no_control_register():
  from test_procedure import CTRL_WRITERS, _writes_ctrl
  for name in ("lang.py", "axes.py", "library.py", "steps.py"):
    path = SRC / "procedure" / name
    assert not _writes_ctrl(ast.parse(path.read_text()))
    assert f"procedure/{name}" not in CTRL_WRITERS


def test_the_axes_ramp_at_the_speeds_the_tools_measured():
  from pluggybot.tools.drawing import CARRIAGE_SPEED
  from pluggybot.tools.gripper import LIFT_SPEED
  assert axes.AXES["lift"].speed == LIFT_SPEED
  assert axes.AXES["pen.carriage"].speed == CARRIAGE_SPEED
  assert all(a.lo < a.hi and a.speed > 0 for a in axes.AXES.values())
  assert set(HOME.axes) == set(axes.AXES) and set(HOME.sensors) == set(axes.SENSORS)


# ---- determinism: the same procedure on the same world is one trajectory -------


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def _hash(data):
  return np.concatenate([data.qpos, data.qvel, data.ctrl]).tobytes()


def test_the_same_procedure_twice_is_one_trajectory(hub_model):
  src = '''def wiggle():
    move("lift", 0.2)
    drive(0.1, 0.4, 0.8)
    look()
    if read("look.tag") < 0:
        move("lift", 0.15)
'''
  proc = lang.compile_procedure(src, HUB)

  def fly():
    data = mujoco.MjData(hub_model)
    mission = HubMission(hub_model, data, viewer=None, realtime=False)
    mission.start_at(0.5, 3.0, 0.0)
    life = SimpleNamespace(mission=mission, model=hub_model, data=data,
                           module="", swaps_done=0, interrupted=lambda: False,
                           _say=lambda *a, **k: None, world="room_hub",
                           boards=None, ledger=None,
                           battery=SimpleNamespace(fraction=0.5, energy_wh=1.0))
    r = mission.run(lang.run_procedure_routine(life, proc, HUB))
    assert r["ok"], r
    return _hash(data), r["completed"]
  a, b = fly(), fly()
  assert a == b and a[1] == 4


# ---- the library ------------------------------------------------------------------


def test_define_undefine_and_the_two_refusals(tmp_path):
  L = lib.Library(HOME, root=tmp_path / "procedures", cap=2)
  L.define("look_around", LOOK_AROUND)
  assert L.names() == ("look_around",) and L.get("look_around").verbs == 7
  with pytest.raises(lib.LibraryRefused, match="already defined"):
    L.define("look_around", LOOK_AROUND)
  with pytest.raises(lib.LibraryRefused, match="def is named 'other'"):
    L.define("mine", "def other():\n  look()\n")
  with pytest.raises(lib.LibraryRefused, match="unknown verb 'fly'"):
    L.define("bad", "def bad():\n  fly()\n")
  L.define("two", "def two():\n  look()\n")
  with pytest.raises(lib.LibraryRefused, match="full"):
    L.define("three", "def three():\n  look()\n")
  with pytest.raises(lib.LibraryRefused, match="no procedure named"):
    L.undefine("nope")
  L.undefine("two")
  assert L.names() == ("look_around",)
  assert L.stats() == {"count": 1, "runnable": 1, "cap": 2, "defined": 2,
                       "undefined": 1, "refused": 5}
  assert len(L.refusals) == 5


def test_the_library_survives_a_restart_and_reads_back_the_same(tmp_path):
  root = tmp_path / "procedures"
  lib.Library(HOME, root=root).define("look_around", LOOK_AROUND)
  again = lib.Library(HOME, root=root)
  assert again.names() == ("look_around",)
  assert again.get("look_around").source == LOOK_AROUND
  assert (root / "look_around.procedure").read_text() == LOOK_AROUND


def test_a_procedure_the_world_moved_under_is_kept_and_marked(tmp_path):
  root = tmp_path / "procedures"
  lib.Library(HOME, root=root).define(
    "sun", 'def sun():\n  fetch("module_pen")\n  draw("sun", "whiteboard_a")\n  stow()\n')
  moved = lib.Library(HUB, root=root)          # a world with no boards
  assert moved.names() == ("sun",) and moved.runnable() == ()
  ctx = moved.as_context()[0]
  assert ctx["runnable"] is False and any("nothing this world has" in r
                                           for r in ctx["reasons"])


def test_there_is_no_replace_verb():
  assert not [m for m in dir(lib.Library) if "replace" in m or "update" in m]
  assert not hasattr(Decision, "redefine")


# ---- the arm: menu, schema, validation, orders, the prompt -----------------------


@pytest.fixture(scope="module")
def menu():
  from pluggybot.lifecycle import board_book
  return replace(Menu.for_world("home", board_book("home")), procedures=True)


def test_the_family_is_on_the_menu_and_the_tokens_in_the_schema(menu):
  assert "procedure" in menu.available()
  assert "procedure" not in Menu.for_world("home").available()
  schema = menu.schema(standing_orders=True, event_map=True,
                       procedures=("look_around", "sun"))
  actions = schema["properties"]["action"]["enum"]
  assert "procedure:look_around" in actions and "procedure" not in actions
  assert "procedure:sun" in schema["properties"]["standing_order"]["enum"]
  rows = schema["properties"]["event_map"]["items"]["properties"]["action"]["enum"]
  assert "procedure:sun" in rows
  assert schema["properties"]["define"]["required"] == ["name", "source"]
  assert schema["properties"]["undefine"]["enum"] == ["look_around", "sun", ""]
  # an empty library: no token, and no family left behind as a bare word
  empty = menu.schema(procedures=())["properties"]["action"]["enum"]
  assert not any(a.startswith("procedure") for a in empty)


def test_validate_accepts_a_library_name_and_refuses_the_rest(menu):
  d = menu.validate({"action": "procedure:sun", "reason": "r",
                     "define": {"name": "x", "source": "def x():\n  look()\n"},
                     "undefine": "sun"}, procedures=("sun",))
  assert d.action == "procedure:sun" and d.undefine == "sun"
  assert d.define == {"name": "x", "source": "def x():\n  look()\n"}
  assert d.as_dict()["define"]["name"] == "x" and d.as_dict()["undefine"] == "sun"
  with pytest.raises(ValueError, match="no runnable procedure named 'nope'"):
    menu.validate({"action": "procedure:nope"}, procedures=("sun",))
  with pytest.raises(ValueError, match="unknown action 'procedure'"):
    menu.validate({"action": "procedure"}, procedures=("sun",))
  # no library offered: the fields are dropped, the action refused
  plain = Menu.for_world("home")
  with pytest.raises(ValueError, match="unknown action"):
    plain.validate({"action": "procedure:sun"})
  d = plain.validate({"action": "idle", "define": {"name": "x", "source": "y"},
                      "undefine": "z"})
  assert d.define is None and d.undefine == ""


def test_a_procedure_may_be_a_standing_order_and_a_row(menu):
  assert ov.standing_order("procedure:sun", menu) == "procedure:sun"
  with pytest.raises(ValueError):
    ov.standing_order("procedure:sun", Menu.for_world("home"))
  with pytest.raises(ValueError, match="names a procedure"):
    ov.standing_order("procedure:", menu)
  assert ov.order_runnable(menu, "procedure:sun", {"procedures": ["sun"]})
  assert not ov.order_runnable(menu, "procedure:sun", {"procedures": []})
  assert not ov.order_runnable(menu, "procedure:sun", {})
  from pluggybot.mind import events as ev
  row = ev.row({"event": "every", "action": "procedure:sun", "value": 30}, menu)
  assert row.action == "procedure:sun"
  d = ov.order_decision(menu, "procedure:sun", {"procedures": ["sun"]}, "timeout")
  assert d.action == "procedure:sun" and d.source == "fallback:timeout"


def test_the_rule_is_on_the_autonomous_prompt_and_not_the_guarded_one(menu):
  L = lib.Library(HOME)
  boss = Overseer(menu, autonomous=True, library=L)
  text = boss.system[0]["text"]
  assert "PROCEDURES YOU MAY WRITE" in text
  assert boss._procedures() == () and boss.stats()["library"]["count"] == 0
  guarded = Overseer(Menu.for_world("home"))
  assert "PROCEDURES" not in guarded.system[0]["text"]
  assert guarded._procedures() is None and "library" not in guarded.stats()


def test_the_rules_worked_example_hands_over_no_survival_policy():
  """EVENT_MAP_RULE's rule, for the same reason: the example must show a
  capability, not the charging policy the arm is measured on."""
  text = ov.procedure_rule()
  example = text[text.index("def look_around"):text.index("Statements:")]
  for word in ("charge", "battery", "rack"):
    assert word not in example, word
  # ...and every verb, axis and sensor is listed, so nothing is a secret
  for v in st.VERBS:
    assert f"  {v}(" in text
  for a in axes.AXES:
    assert f"  {a}:" in text
  for s in axes.SENSORS:
    assert f"  {s} --" in text


def test_the_guarded_rules_have_not_moved():
  from test_autonomous import GUARDED_RULES_SHA
  import hashlib
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA


# ---- the lifecycle: define, run by name, stow ------------------------------------


def _life(world="room_hub"):
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  return HubLifecycle(model, mujoco.MjData(model), realtime=False, world=world,
                      errand=False, battery_wh=cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"])


def test_a_decision_defines_and_undefines_and_every_refusal_is_narrated(tmp_path):
  life = _life()
  L = lib.Library(HUB, root=tmp_path / "p")
  life.overseer = SimpleNamespace(library=L)
  events: list = []
  life.on_event.append(events.append)
  life._define(Decision(action="idle",
                        define={"name": "two", "source": "def two():\n  look()\n  look()\n"}))
  assert L.names() == ("two",) and events[-1]["outcome"] == "defined"
  assert events[-1]["program"]["source"].startswith("def two")
  life._define(Decision(action="idle", define={"name": "two", "source": "def two():\n  look()\n"}))
  assert events[-1]["outcome"] == "refused" and "already defined" in events[-1]["reasons"][0]
  assert any("refused" in line for line in life.log[-2:])
  life._define(Decision(action="idle", undefine="two",
                        define={"name": "two", "source": "def two():\n  look()\n"}))
  assert L.names() == ("two",) and L.get("two").verbs == 1
  assert [e["outcome"] for e in events[-2:]] == ["undefined", "defined"]


def test_a_procedure_runs_by_name_as_a_composed_errand(monkeypatch):
  from test_procedure import _stub_swaps
  life = _life()
  _stub_swaps(life, monkeypatch)
  L = lib.Library(HUB)
  L.define("carry", 'def carry():\n  fetch("module_lcd")\n  n = 0\n'
                    '  while n < 2:\n    drive_to(1, 1)\n    n += 1\n  stow()\n')
  errand = errand_from(Decision(action="procedure:carry"), "room_hub", library=L)
  assert errand is not None and errand.name == "procedure"
  assert errand.program is L.get("carry") and errand.module == "module_lcd"
  assert errand_from(Decision(action="procedure:nope"), "room_hub", library=L) is None
  result = life.run_errand(errand)
  run = result["procedure"]
  assert run["ok"] and run["completed"] == 4 and result["stowed"]
  assert [s["verb"] for s in run["steps"]] == ["fetch", "drive_to", "drive_to", "stow"]
  assert result["verdict"]["ok"] and result["verdict"]["task"] == "program"


def test_a_procedure_cut_short_is_stowed(monkeypatch):
  from test_procedure import _stub_swaps
  life = _life()
  on_fork = _stub_swaps(life, monkeypatch)
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  L = lib.Library(HUB)
  L.define("short", 'def short():\n  fetch("module_lcd")\n  drive_to(1, 1)\n  stow()\n')
  result = life.run_errand(errand_from(Decision(action="procedure:short"),
                                       "room_hub", library=L))
  assert result["procedure"]["failedAt"] == 1 and on_fork == {}
  assert result["stowed"] and not result["verdict"]["ok"]


# ---- the flown one: written by the agent, invoked by its own row ------------------


@pytest.mark.endurance
def test_a_procedure_the_agent_wrote_is_invoked_from_its_own_row(tmp_path):
  """The integration the issue asks for: one flown run in which the model
  defines a procedure and writes a row that runs it, and the row fires and
  the procedure completes -- with no scripted rotation anywhere (the arm is
  `autonomous`). 84 s, behind --endurance: every rule it exercises (the row
  token, the define, the errand by name, the stow) is pinned above."""
  from test_event_map import attach
  from test_overseer import FakeClient, full
  from pluggybot.lifecycle import run_demo
  src = "def look_twice():\n  look()\n  wait(1)\n  look()\n"
  first = full(action="idle", reason="setting up",
               define={"name": "look_twice", "source": src},
               event_map=[{"event": "every", "action": "procedure:look_twice",
                           "value": 20, "kind": ""},
                          {"event": "nothing_to_do", "action": "ask", "value": 0,
                           "kind": ""}])
  client = FakeClient(first, full(action="idle", reason="waiting"))
  out = run_demo(view=False, realtime=False, world="room_hub", errand="none",
                 max_sim_time=75.0, overseer=True, standing_orders=True,
                 autonomous=True, origin="seeded",
                 thoughts_root=str(tmp_path / "t"),
                 ledger_state=str(tmp_path / "ledger.json"),
                 on_ready=attach(client))
  fired = [d for d in out["decisions"] if d["action"] == "procedure:look_twice"]
  assert fired and fired[0]["source"] == "event:every"
  runs = [e for e in out["errands"] if e.get("procedure")]
  assert runs and runs[0]["procedure"]["ok"] and runs[0]["procedure"]["completed"] == 3
  assert out["overseer"]["library"] == {"count": 1, "runnable": 1, "cap": 8,
                                        "defined": 1, "undefined": 0, "refused": 0}
  assert (tmp_path / "t" / "procedures" / "look_twice.procedure").read_text() == src
  assert not [d for d in out["decisions"] if ov.fallback_class(d["source"]) == "failure"]
