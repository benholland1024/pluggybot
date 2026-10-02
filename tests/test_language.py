"""The procedure language (procedure/lang.py, axes.py, library.py; issue
#166): parsed never executed, total by construction, the robot's own
library, and a mind's vocabulary for it.
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
from pluggybot.lifecycle import errand_from, world_config, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Decision, Menu, Overseer
from pluggybot.procedure import axes, lang, library as lib, steps as st
from pluggybot.procedure.steps import Refused
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
WORLD = "home_quad"
HOME = world_facts(WORLD)

LOOK_AROUND = '''def look_around():
    budget(steps=40, seconds=240)
    n = 0
    fetch("module_lcd")
    for i in range(2):
        drive(0.0, 0.8, 1.5)
        if read("bumper") >= 1 and read("time") < 100:
            wait(2)
        elif read("bumper") < 0:
            pass
        else:
            wait(1)
    while read("shoulder") < 1.0 and n < 8:
        move("shoulder", read("shoulder") + 0.1)
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


def test_a_procedure_in_every_construct_compiles():
  p = compile_ok(LOOK_AROUND)
  assert p.name == "look_around" and p.verbs == 6
  assert p.steps_budget == 40 and p.budget_s == 240.0
  assert p.first("fetch", "tool") == "module_lcd"


def test_the_served_example_compiles_on_its_world():
  """The rule's own example, as the robot reads it (#434): it walked at
  0.3 m/s against `drive`'s 0.25, so a robot that copied it was taught a
  refusal, while the copy above passed. With the served world's rack and
  without."""
  menu = Menu.for_world(WORLD)
  assert menu.swaps, "the served rule is the one that fetches"
  for swaps in (True, False):
    text = ov.procedure_rule(swaps, menu.places, menu.plates, menu.draws)
    example = text[text.index("def look_around"):text.index("Statements:")]
    compile_ok(example.split("\n\n")[0] + "\n")


@pytest.mark.parametrize("src, needle", [
  ("stow()\n", "exactly one `def name():`"),
  ("def a():\n  stow()\ndef b():\n  stow()\n", "exactly one `def"),
  ("def A():\n  stow()\n", "not a name this language allows"),
  ("def x(y):\n  stow()\n", "takes no arguments"),
  ("@dec\ndef x():\n  stow()\n", "takes no arguments and no decorators"),
  ("def x():\n  pass\n", "calls no verb"),
  ("def x():\n  \"\"\"only a docstring\"\"\"\n", "has no steps"),
  ("def x():\n  import os\n", "line 2: Import is not part"),
  ("def x():\n  print(1)\n", "unknown verb 'print'"),
  ("def x():\n  fly()\n", "unknown verb 'fly'"),
  ("def x():\n  read(\"shoulder\")\n", "read(...) is an expression"),
  ("def x():\n  wait(stow())\n", "only read(...) may be used inside an expression"),
  ("def x():\n  a = \"s\"\n", "a str is not a number"),
  ("def x():\n  a = [1]\n", "List is not part"),
  ("def x():\n  a = b.c\n", "Attribute is not part"),
  ("def x():\n  a = 1 < 2 < 3\n", "compare two things with one of"),
  ("def x():\n  for i in [1, 2]:\n    stow()\n", "`for name in range(N)`"),
  ("def x():\n  for i in range(n):\n    stow()\n", "literal N"),
  ("def x():\n  for i in range(1000):\n    stow()\n", "outside 0..100"),
  ("def x():\n  while 1:\n    stow()\n  else:\n    stow()\n", "while takes no else"),
  ("def x():\n  return 1\n", "return takes no value"),
  ("def x():\n  wait(1, 2)\n", "wait takes 1 argument"),
  ("def x():\n  wait(seconds=1, z=2)\n", "wait takes no z"),
  ("def x():\n  fetch()\n", "fetch needs tool"),
  ("def x():\n  fetch(3)\n", "fetch(tool=...) takes a name in quotes"),
  ("def x():\n  wait(\"long\")\n", "wait(seconds=...) takes a number"),
  ("def x():\n  wait(999)\n", "seconds=999.0 is above 60"),
  ("def x():\n  move(\"shoulder\", 5)\n", "shoulder target 5.0 is outside"),
  ("def x():\n  move(\"warp\", 1)\n", "axis='warp' is not one of"),
  ("def x():\n  fetch(\"module_x\")\n", "tool='module_x' is not one of"),
  ("def x():\n  a = read(\"nope\")\n", "unknown sensor 'nope'"),
  ("def x():\n  drive_to(99, 0)\n", "outside the map"),
  ("def x():\n  budget(steps=999)\n  stow()\n", "steps=999 is outside 1..200"),
  ("def x():\n  budget(seconds=0)\n  stow()\n", "seconds=0.0 is outside"),
  ("def x():\n  budget(40)\n  stow()\n", "budget takes keywords"),
  ("def x():\n  budget(steps=\"many\")\n  stow()\n", "not a literal steps or seconds"),
  ("def x():\n  a = 1 +\n", "invalid syntax"),
  ("def x():\n  if 1:\n    if 1:\n      if 1:\n        if 1:\n          stow()\n",
   "nested deeper than"),
  ("def x():\n  read = 3\n  stow()\n", "reserved"),
  ("def x():\n  wait = 3\n  stow()\n", "reserved"),
])
def test_each_construct_outside_the_grammar_is_refused(src, needle):
  reasons = refusals(src)
  assert any(needle in r for r in reasons), reasons


def test_the_source_cap_is_a_refusal():
  src = "def x():\n" + "  stow()\n" * 400
  assert any("characters; the cap is" in r for r in refusals(src))


def test_a_fetch_in_a_world_with_an_empty_rack_is_refused_by_the_world():
  reasons = refusals('def x():\n  fetch("module_pen")\n', replace(HOME, tools=()))
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


def _no_actuator(name):
  raise KeyError(name)


def _stub_life():
  """A life whose every primitive records itself and steps nothing, with
  no arm for a drive to pose: what is under test is the interpreter."""
  calls: list = []

  def routine(name, value=True):
    def make(*a, **kw):
      calls.append((name, a))
      return tick.result(value)
    return make
  from pluggybot.robot import FIRST
  model = SimpleNamespace(actuator=lambda name: SimpleNamespace(id=0))
  body = SimpleNamespace(
    module_state=lambda tool: {"on_fork": False, "hung": True},
    ramp_routine=routine("ramp"), pressing=False, handle=FIRST,
    actuator=_no_actuator, setpoint=lambda act: 0.0,
    fetch_tool_routine=routine("swap", "arrived"),
    stow_tool_routine=routine("swap", "arrived"),
    go_to_routine=routine("drive_to", True), face_routine=routine("face", True),
    in_sight=lambda x, y: True, hold_routine=routine("wait"),
    velocity_routine=routine("drive"), pose=(0.0, 0.0, 0.0),
    )
  return SimpleNamespace(body=body, data=SimpleNamespace(time=0.0, ctrl=[0.0]),
                         module="", swaps_done=0, interrupted=lambda: False,
                         _say=lambda *a, **k: None, calls=calls, model=model,
                         world=WORLD, boards=None, ledger=None,
                         battery=SimpleNamespace(fraction=0.5, energy_wh=4.0),
                         drive_why=lambda x, y: "the drive gave up (why)")


def run(src, life=None, facts=HOME):
  life = life or _stub_life()
  proc = lang.compile_procedure(src, facts)
  r = tick.run(SimpleNamespace(step=lambda *a: None),
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
  life.body.hold_routine = wait
  r, _ = run('def x():\n  while read("battery.frac") < 1:\n    wait(1)\n', life)
  assert r["ok"] and r["completed"] == 4


def test_a_computed_argument_is_checked_when_it_is_computed():
  r, _ = run('def x():\n  a = 30\n  wait(a * 3)\n')
  assert not r["ok"] and r["failedAt"] == 0
  assert "seconds=90.0 is above 60" in r["steps"][0]["reason"]


def test_a_failed_step_ends_the_procedure_honestly():
  life = _stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  r, _ = run('def x():\n  wait(1)\n  drive_to(1, 1)\n  wait(1)\n', life)
  assert not r["ok"] and r["completed"] == 1 and r["failedAt"] == 1
  assert r["total"] == 2 and len(r["steps"]) == 2


@pytest.mark.parametrize("src, why", [
  ('def x():\n  a = 1 / 0\n  wait(1)\n', "division by zero"),
  ('def x():\n  wait(b)\n', "'b' was read before it was set"),
  ('def x():\n  n = 0\n  while 1:\n    n += 1\n  stow()\n', "loop-cap"),
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
  life.body.hold_routine = slow
  r, _ = run('def x():\n  budget(seconds=150)\n  for i in range(5):\n    wait(1)\n',
             life)
  assert r["stopped"] == "budget" and r["completed"] == 2


def test_an_interrupt_stops_between_verbs():
  life = _stub_life()
  life.interrupted = lambda: True
  r, _ = run('def x():\n  wait(1)\n  wait(1)\n', life)
  assert r["stopped"] == "interrupted" and r["completed"] == 1


@pytest.fixture
def scoop_facts():
  """A built tool's axis and sensor registered as the workshop registers
  them (`workshop.build.register`, issue #168), each needing its module on
  the fork, and the facts that name them; the registries put back after."""
  from pluggybot.workshop import build, validate
  from test_workshop import SCOOP
  before_a, before_s = set(axes.AXES), set(axes.SENSORS)
  names = build.register(validate.check(SCOOP))
  yield replace(HOME, axes=HOME.axes + tuple(names), sensors=HOME.sensors + tuple(names))
  for k in set(axes.AXES) - before_a:
    del axes.AXES[k]
  for k in set(axes.SENSORS) - before_s:
    del axes.SENSORS[k]


def test_move_goes_through_the_bodys_ramp_and_refuses_a_missing_tool(scoop_facts):
  r, life = run('def x():\n  move("shoulder", 1.0)\n  move("scoop.tilt", 0.5)\n',
                facts=scoop_facts)
  assert r["steps"][0]["ok"] and [c[0] for c in life.calls] == ["ramp"]
  assert not r["steps"][1]["ok"]
  assert r["steps"][1]["reason"] == "axis 'scoop.tilt' needs module_scoop on the fork"


def test_a_sensor_that_needs_a_tool_fails_without_it(scoop_facts):
  r, _ = run('def x():\n  a = read("scoop.tilt")\n  wait(1)\n', facts=scoop_facts)
  assert not r["ok"] and "needs module_scoop on the fork" in r["steps"][-1]["reason"]


# ---- the fence, extended to loops ------------------------------------------------


def test_the_language_modules_write_no_control_register():
  from test_procedure import CTRL_WRITERS, _writes_ctrl
  for name in ("lang.py", "axes.py", "library.py", "steps.py"):
    path = SRC / "procedure" / name
    assert not _writes_ctrl(ast.parse(path.read_text()))
    assert f"procedure/{name}" not in CTRL_WRITERS


def test_the_arms_axes_ramp_at_its_slew_and_the_world_names_them_all():
  from pluggybot.legs.arm import ARM_SLEW
  assert {n: a.speed for n, a in axes.AXES.items()} == {"shoulder": ARM_SLEW,
                                                        "elbow": ARM_SLEW}
  assert all(a.lo < a.hi and a.speed > 0 for a in axes.AXES.values())
  # the quadruped's world names every axis and every sensor there is
  assert set(HOME.axes) == set(axes.AXES) == set(axes.ARM_JOINTS)
  assert set(HOME.sensors) == set(axes.SENSORS)


# ---- determinism: the same procedure on the same world is one trajectory -------


def _hash(data):
  return np.concatenate([data.qpos, data.qvel, data.ctrl]).tobytes()


def test_the_same_procedure_twice_is_one_trajectory():
  """On the served body: a walk, a sensor read and a move computed from it,
  flown twice from one start, leave the world byte for byte the same."""
  from pluggybot.legs import body as qb
  from pluggybot.legs.world import home_spec
  src = '''def wiggle():
    drive(0.1, 0.4, 0.8)
    if read("bumper") < 1:
        move("shoulder", read("shoulder") - 0.1)
'''
  proc = lang.compile_procedure(src, HOME)
  model = home_spec().compile()

  def fly():
    data = mujoco.MjData(model)
    body = qb.QuadBody(model, data, realtime=False,
                       grid_bounds=world_config(WORLD)["grid_bounds"])
    body.start_at(1.5, 0.5, 0.0)
    life = SimpleNamespace(body=body, model=model, data=data,
                           module="", swaps_done=0, interrupted=lambda: False,
                           _say=lambda *a, **k: None, world=WORLD,
                           boards=None, ledger=None,
                           battery=SimpleNamespace(fraction=0.5, energy_wh=1.0))
    r = body.run(lang.run_procedure_routine(life, proc, HOME))
    assert r["ok"], r
    return _hash(data), r["completed"]
  a, b = fly(), fly()
  assert a == b and a[1] == 2


# ---- the library ------------------------------------------------------------------


def test_define_undefine_and_the_two_refusals(tmp_path):
  L = lib.Library(HOME, root=tmp_path / "procedures", cap=2)
  L.define("look_around", LOOK_AROUND)
  assert L.names() == ("look_around",) and L.get("look_around").verbs == 6
  with pytest.raises(lib.LibraryRefused, match="already defined"):
    L.define("look_around", LOOK_AROUND)
  with pytest.raises(lib.LibraryRefused, match="def is named 'other'"):
    L.define("mine", "def other():\n  stow()\n")
  with pytest.raises(lib.LibraryRefused, match="unknown verb 'fly'"):
    L.define("bad", "def bad():\n  fly()\n")
  L.define("two", "def two():\n  stow()\n")
  with pytest.raises(lib.LibraryRefused, match="full"):
    L.define("three", "def three():\n  stow()\n")
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
    "sun", 'def sun():\n  fetch("module_pen")\n  stow()\n')
  moved = lib.Library(replace(HOME, tools=()), root=root)     # a world with no rack
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
  return replace(Menu.for_world(WORLD, board_book(WORLD)), procedures=True)


def test_the_family_is_on_the_menu_and_the_tokens_in_the_schema(menu):
  assert "procedure" in menu.available()
  assert "procedure" not in Menu.for_world(WORLD).available()
  schema = menu.schema(standing_orders=True, event_map=True,
                       procedures=("look_around", "sun"))
  actions = schema["properties"]["action"]["enum"]
  assert "procedure:look_around" in actions and "procedure" not in actions
  assert "procedure:sun" in schema["properties"]["standing_order"]["enum"]
  rows = schema["properties"]["event_map"]["items"]["properties"]["action"]["enum"]
  assert "procedure:sun" in rows
  assert schema["properties"]["define"]["required"] == ["name", "source"]
  assert schema["properties"]["undefine"]["enum"] == ["look_around", "sun", ""]
  # an empty library: no family left behind as a bare word, and one token --
  # `procedure:new`, the procedure the answer itself defines (issue #264),
  # which no order and no map row may name
  empty = menu.schema(procedures=())["properties"]["action"]["enum"]
  assert [a for a in empty if a.startswith("procedure")] == ["procedure:new"]
  assert "procedure:new" not in schema["properties"]["standing_order"]["enum"]
  assert "procedure:new" not in rows


def test_validate_accepts_a_library_name_and_refuses_the_rest(menu):
  d = menu.validate({"action": "procedure:sun", "reason": "r",
                     "define": {"name": "x", "source": "def x():\n  stow()\n"},
                     "undefine": "sun"}, procedures=("sun",))
  assert d.action == "procedure:sun" and d.undefine == "sun"
  assert d.define == {"name": "x", "source": "def x():\n  stow()\n"}
  assert d.as_dict()["define"]["name"] == "x" and d.as_dict()["undefine"] == "sun"
  with pytest.raises(ValueError, match="no runnable procedure named 'nope'"):
    menu.validate({"action": "procedure:nope"}, procedures=("sun",))
  with pytest.raises(ValueError, match="unknown action 'procedure'"):
    menu.validate({"action": "procedure"}, procedures=("sun",))
  # no library offered: the fields are dropped, the action refused
  plain = Menu.for_world(WORLD)
  with pytest.raises(ValueError, match="unknown action"):
    plain.validate({"action": "procedure:sun"})
  d = plain.validate({"action": "idle", "define": {"name": "x", "source": "y"},
                      "undefine": "z"})
  assert d.define is None and d.undefine == ""


def test_a_procedure_may_be_a_standing_order_and_a_row(menu):
  assert ov.standing_order("procedure:sun", menu) == "procedure:sun"
  with pytest.raises(ValueError):
    ov.standing_order("procedure:sun", Menu.for_world(WORLD))
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


def test_the_rule_is_on_a_prompt_with_a_library_and_not_without_one(menu):
  L = lib.Library(HOME)
  boss = Overseer(menu, library=L)
  text = boss.system[0]["text"]
  assert "PROCEDURES YOU MAY WRITE" in text
  assert boss._procedures() == () and boss.stats()["library"]["count"] == 0
  bare = Overseer(Menu.for_world(WORLD))
  assert "PROCEDURES" not in bare.system[0]["text"]
  assert bare._procedures() is None and "library" not in bare.stats()


def test_the_rules_worked_example_hands_over_no_survival_policy():
  """EVENT_MAP_RULE's rule, for the same reason: the example must show a
  capability, not the charging policy the arm is measured on."""
  menu = Menu.for_world(WORLD)
  text = ov.procedure_rule(menu.swaps, menu.places, menu.plates, menu.draws)
  example = text[text.index("def look_around"):text.index("Statements:")]
  for word in ("charge", "battery", "rack"):
    assert word not in example, word
  # ...and every verb, axis and sensor is listed, so nothing is a secret --
  # the fork's verbs where the world has a rack (#405), the places' where
  # it keeps places and the plates' where they are (#419)
  for v in st.VERBS:
    assert f"  {v}(" in text, v
  for a in axes.AXES:
    assert f"  {a}:" in text and f"  {a} --" in text, a
  for s in axes.SENSORS:
    assert f"  {s} --" in text, s
  swaps, places = st.SWAP_VERBS, st.PLACE_VERBS + st.PLATE_VERBS
  bare = ov.procedure_rule()
  assert not any(f"  {v}(" in bare for v in swaps + places)
  assert all(f"  {v}(" in ov.procedure_rule(swaps=True) for v in swaps)
  finding = ov.procedure_rule(places=True)
  assert "  find(" in finding and "  press(" not in finding
  assert "  press(" in ov.procedure_rule(places=True, plates=True)
  # ...and `draw` where it draws on the boards it finds (#406)
  assert "  draw(" not in ov.procedure_rule(swaps=True, places=True)
  assert "  draw(" in ov.procedure_rule(swaps=True, places=True, draws=True)


# ---- the lifecycle: define, run by name, stow ------------------------------------


def _life(world=WORLD):
  return stub_life(world)


def test_a_decision_defines_and_undefines_and_every_refusal_is_narrated(tmp_path):
  life = _life()
  L = lib.Library(HOME, root=tmp_path / "p")
  life.overseer = SimpleNamespace(library=L)
  events: list = []
  life.on_event.append(events.append)
  life._define(Decision(action="idle",
                        define={"name": "two", "source": "def two():\n  stow()\n  stow()\n"}))
  assert L.names() == ("two",) and events[-1]["outcome"] == "defined"
  assert events[-1]["program"]["source"].startswith("def two")
  life._define(Decision(action="idle", define={"name": "two", "source": "def two():\n  stow()\n"}))
  assert events[-1]["outcome"] == "refused" and "already defined" in events[-1]["reasons"][0]
  assert any("refused" in line for line in life.log[-2:])
  life._define(Decision(action="idle", undefine="two",
                        define={"name": "two", "source": "def two():\n  stow()\n"}))
  assert L.names() == ("two",) and L.get("two").verbs == 1
  assert [e["outcome"] for e in events[-2:]] == ["undefined", "defined"]


def test_every_library_event_carries_the_library_as_it_stands_and_its_cap(tmp_path):
  """A consumer folding `defined`/`undefined` rows (the website's panel,
  rooftop-media-2026 #342) re-anchors on the names each library event
  carries, so a row it missed cannot leave a procedure on its page that the
  robot no longer keeps -- and the cap is read off the wire, never typed.
  Each event carries the library AFTER its own change: an undefine and a
  define in one decision are two different libraries."""
  life = _life()
  L = lib.Library(HOME, root=tmp_path / "p", cap=2)
  life.overseer = SimpleNamespace(library=L)
  events: list = []
  life.on_event.append(events.append)
  life._define(Decision(action="idle", define={"name": "one", "source": "def one():\n  stow()\n"}))
  assert events[-1]["library"] == {"names": ["one"], "cap": 2}
  life._define(Decision(action="idle", define={"name": "one", "source": "def one():\n  wait(1)\n"}))
  assert events[-1]["outcome"] == "refused" and events[-1]["library"] == {"names": ["one"], "cap": 2}
  life._define(Decision(action="idle", undefine="nope"))
  assert events[-1]["verb"] == "undefine" and events[-1]["library"]["names"] == ["one"]
  life._define(Decision(action="idle", undefine="one",
                        define={"name": "two", "source": "def two():\n  stow()\n"}))
  assert [(e["outcome"], e["library"]["names"]) for e in events[-2:]] == [
    ("undefined", []), ("defined", ["two"])]


def test_a_procedure_runs_by_name_as_a_composed_errand():
  life = _life()
  L = lib.Library(HOME)
  L.define("carry", 'def carry():\n  fetch("module_lcd")\n  n = 0\n'
                    '  while n < 2:\n    drive_to(1, 1)\n    n += 1\n  stow()\n')
  errand = errand_from(Decision(action="procedure:carry"), WORLD, library=L)
  assert errand is not None and errand.name == "procedure"
  assert errand.program is L.get("carry") and errand.module == "module_lcd"
  assert errand_from(Decision(action="procedure:nope"), WORLD, library=L) is None
  result = life.run_errand(errand)
  run = result["procedure"]
  assert run["ok"] and run["completed"] == 4 and result["stowed"]
  assert [s["verb"] for s in run["steps"]] == ["fetch", "drive_to", "drive_to", "stow"]
  assert result["verdict"]["ok"] and result["verdict"]["task"] == "program"


def test_a_procedure_cut_short_is_stowed():
  from test_procedure import _stub_swaps
  life = _life()
  on_fork = _stub_swaps(life)
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  L = lib.Library(HOME)
  L.define("short", 'def short():\n  fetch("module_lcd")\n  drive_to(1, 1)\n  stow()\n')
  result = life.run_errand(errand_from(Decision(action="procedure:short"),
                                       WORLD, library=L))
  assert result["procedure"]["failedAt"] == 1 and on_fork == {}
  assert result["stowed"] and not result["verdict"]["ok"]


def test_a_failed_run_names_the_line_that_failed_and_why():
  """`failedAt` counts verb calls EXECUTED, so inside a loop it names no
  line of the source (here the third call is on line 4, not the third
  verb in the text). The `aborted` event says the line and the step's own
  reason beside the count (rooftop-media-2026 #342), which is what lets a
  reader mark the failing line on the source rather than guess it; a run
  that went through carries neither."""
  life = _life()
  events: list = []
  life.on_event.append(events.append)
  L = lib.Library(HOME)
  L.define("loop", 'def loop():\n  for i in range(2):\n    wait(0.1)\n  stow()\n')
  L.define("fine", 'def fine():\n  for i in range(2):\n    wait(0.1)\n')
  for name in ("loop", "fine"):
    life.run_errand(errand_from(Decision(action=f"procedure:{name}"), WORLD, library=L))
  aborted, ran = [e for e in events if e.get("type") == "procedure"
                  and e["outcome"] in ("aborted", "ran")]
  assert aborted["name"] == "loop" and aborted["outcome"] == "aborted"
  assert (aborted["failedAt"], aborted["failedLine"]) == (2, 4)
  assert aborted["failedReason"] == "nothing on the fork to stow"
  assert ran["outcome"] == "ran" and "failedLine" not in ran and "failedReason" not in ran


# ---- the day: written by the agent, invoked by its own row -----------------------


def test_a_procedure_the_agent_wrote_is_invoked_from_its_own_row(tmp_path):
  """The integration the issue asks for, as a day on the stub (issue #380):
  the model defines a procedure and writes a row that runs it, the row
  fires, and the procedure completes -- with no rotation anywhere: the
  fallback is the agent's own. The row, queued while the robot idles, runs
  when it is next free. Shown to fail by dropping `self._define(decision)`
  from `_after_decision_routine`.
  """
  from test_overseer import FakeClient, full
  from pluggybot.mind.thoughts import ThoughtFiles
  src = "def pace():\n  wait(1)\n  drive(0.1, 0.0, 1.0)\n  wait(1)\n"
  first = full(action="idle", reason="setting up",
               define={"name": "pace", "source": src},
               event_map=[{"event": "every", "action": "procedure:pace",
                           "value": 20, "kind": ""},
                          {"event": "nothing_to_do", "action": "ask", "value": 0,
                           "kind": ""}])
  client = FakeClient(first, full(action="idle", reason="waiting"))
  memory = ThoughtFiles.open(str(tmp_path / "t"))
  boss = ov.build(WORLD, None, enabled=True, client=client, thoughts=memory,
                  origin="seeded")
  life = stub_life(WORLD, overseer=boss, thoughts=memory)
  life.stop_when(lambda: any(e.get("procedure") for e in life.errand_results))
  # The claim ends the day; the budget has room for late answers, which on
  # the stub are SIM time.
  out = life.run(start=world_config(WORLD)["start"], max_sim_time=600.0)
  fired = [d for d in out["decisions"] if d["action"] == "procedure:pace"]
  assert fired and fired[0]["source"] == "event:every"
  runs = [e for e in out["errands"] if e.get("procedure")]
  assert runs and runs[0]["procedure"]["ok"] and runs[0]["procedure"]["completed"] == 3
  assert out["overseer"]["library"] == {"count": 1, "runnable": 1, "cap": 8,
                                        "defined": 1, "undefined": 0, "refused": 0}
  assert (tmp_path / "t" / "procedures" / "pace.procedure").read_text() == src
  assert not [d for d in out["decisions"] if ov.fallback_class(d["source"]) == "failure"]
