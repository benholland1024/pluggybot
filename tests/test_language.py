"""The procedure language (procedure/lang.py, axes.py, library.py; issue
#166): parsed never executed, total by construction, the robot's own
library, and a mind's vocabulary for it.
"""

import ast
import math
import inspect
import pathlib
import textwrap
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
WORLD = "home_humanoid"
HOME = world_facts(WORLD)

LOOK_AROUND = '''def look_around():
    budget(steps=40, seconds=240)
    n = 0
    reach("right", 0.3, -0.2, 1.0)
    for i in range(2):
        face(0.8)
        if read("right.z") >= 1 and read("time") < 100:
            wait(2)
        elif read("right.z") < 0:
            pass
        else:
            wait(1)
    while read("right.x") < 0.5 and n < 8:
        move_hand("right", read("right.x") + 0.05, -0.2, 1.0)
        n += 1
    withdraw("right")
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
  assert p.first("reach", "hand") == "right"


def test_the_served_example_compiles_on_its_world():
  """The rule's own example, as the robot reads it (#434): it walked at
  0.3 m/s against `drive`'s 0.25, so a robot that copied it was taught a
  refusal, while the copy above passed. The served world's rule, the
  humanoid's (#559)."""
  menu = Menu.for_world(WORLD)
  text = ov.hands_procedure_rule(menu.place_tags)
  example = text[text.index("  def wave"):text.index("Statements:")]
  proc = compile_ok(textwrap.dedent(example).split("\n\n")[0] + "\n")
  assert proc.name == "wave" and proc.verbs == 4


@pytest.mark.parametrize("src, needle", [
  ("wait(1)\n", "exactly one `def name():`"),
  ("def a():\n  wait(1)\ndef b():\n  wait(1)\n", "exactly one `def"),
  ("def A():\n  wait(1)\n", "not a name this language allows"),
  ("def x(y):\n  wait(1)\n", "takes no arguments"),
  ("@dec\ndef x():\n  wait(1)\n", "takes no arguments and no decorators"),
  ("def x():\n  pass\n", "calls no verb"),
  ("def x():\n  \"\"\"only a docstring\"\"\"\n", "has no steps"),
  ("def x():\n  import os\n", "line 2: Import is not part"),
  ("def x():\n  print(1)\n", "unknown verb 'print'"),
  ("def x():\n  fly()\n", "unknown verb 'fly'"),
  ("def x():\n  read(\"time\")\n", "read(...) is an expression"),
  ("def x():\n  wait(face(1))\n",
   "line 2, wait: only read(...), sin, cos, atan2 and sqrt may be used inside an expression"),
  ("def x():\n  a = \"s\"\n", "a str is not a number"),
  ("def x():\n  a = [1]\n", "List is not part"),
  ("def x():\n  a = b.c\n", "b.c: a name with a dot is not part of this language"),
  # a sensor written as Python is told its read (#493)
  ("def x():\n  a = right.fx\n",
   'right.fx is a sensor: read it with read("right.fx")'),
  ("def x():\n  wait(battery.frac)\n",
   'line 2, wait: battery.frac is a sensor: read it with read("battery.frac")'),
  ("def x():\n  a = 5 % 2\n", "% is not part of this language: numbers combine with"),
  ("def x():\n  a = 1 < 2 < 3\n", "compare two things with one of"),
  ("def x():\n  for i in [1, 2]:\n    wait(1)\n", "`for name in range(N)`"),
  ("def x():\n  for i in range(n):\n    wait(1)\n", "literal N"),
  ("def x():\n  for i in range(1000):\n    wait(1)\n", "outside 0..100"),
  ("def x():\n  while 1:\n    wait(1)\n  else:\n    wait(1)\n", "while takes no else"),
  ("def x():\n  return 1\n", "return takes no value"),
  ("def x():\n  wait(1, 2)\n", "wait: takes 1 argument"),
  ("def x():\n  wait(seconds=1, z=2)\n", "wait: takes no z"),
  ("def x():\n  reach()\n", "reach: needs hand"),
  ("def x():\n  open_hand(3)\n", "open_hand: hand takes a name in quotes"),
  ("def x():\n  wait(\"long\")\n", "wait: seconds takes a number"),
  ("def x():\n  wait(999)\n", "seconds=999.0 is above 60"),
  ("def x():\n  move(\"left_elbow\", 5)\n", "left_elbow target 5.0 is outside"),
  ("def x():\n  move(\"warp\", 1)\n", "axis='warp' is not one of"),
  ("def x():\n  open_hand(\"middle\")\n", "hand='middle' is not one of left, right"),
  ("def x():\n  a = read(\"nope\")\n", "unknown sensor 'nope'"),
  ("def x():\n  drive_to(99, 0)\n", "outside the map"),
  ("def x():\n  budget(steps=999)\n  wait(1)\n", "steps=999 is outside 1..200"),
  ("def x():\n  budget(seconds=0)\n  wait(1)\n", "seconds=0.0 is outside"),
  ("def x():\n  budget(40)\n  wait(1)\n", "line 2, budget: takes keywords"),
  ("def x():\n  budget(steps=\"many\")\n  wait(1)\n", "not a literal steps or seconds"),
  ("def x():\n  a = 1 +\n", "invalid syntax"),
  ("def x():\n  if 1:\n    if 1:\n      if 1:\n        if 1:\n          wait(1)\n",
   "nested deeper than"),
  ("def x():\n  read = 3\n  wait(1)\n", "reserved"),
  ("def x():\n  wait = 3\n  wait(1)\n", "reserved"),
])
def test_each_construct_outside_the_grammar_is_refused(src, needle):
  reasons = refusals(src)
  assert any(needle in r for r in reasons), reasons


def test_the_source_cap_is_a_refusal():
  src = "def x():\n" + "  wait(1)\n" * 400
  assert any("characters; the cap is" in r for r in refusals(src))


@pytest.mark.parametrize("src, said", [
  ("def x():\n  wait(" + "9" * 309 + ")\n", "line 2, wait: "),
  ("def x():\n  wait(1e400)\n", "line 2, wait: "),
  ("def x():\n  t = 0x" + "f" * 260 + "\n  wait(1)\n", "line 2: "),
  ("def x():\n  budget(seconds=" + "9" * 309 + ")\n  wait(1)\n", "line 2, budget: "),
], ids=["an argument", "a float", "a hex assignment", "a budget"])
def test_a_number_too_large_for_a_float_is_refused_never_raised(src, said):
  """`float()` raised OverflowError on an int past a float's range, out
  through the define, and nothing above it caught that: one answer of 330
  characters ended the served process, both robots with it (#500). 1e400,
  inf, is the same mistake and reads the same."""
  [why] = refusals(src)
  assert why == said + "a number is too large to hold", why


def test_a_number_too_large_is_one_reason_among_every_other():
  """Refused before the parse, it was the only reason said: the define's
  other faults cost the robot another turn each (#500)."""
  reasons = refusals("def x():\n  wait(" + "9" * 309 + ")\n  import os\n  fly()\n")
  assert len(reasons) == 3 and reasons[0] == "line 2, wait: a number is too large to hold", reasons


@pytest.mark.parametrize("arg", ["-" * 1000 + "1", "+".join(["1"] * 60),
                                 " and ".join(["1"] * 200)],
                         ids=["unary minuses", "a long sum", "an and-chain"])
def test_an_expression_nested_too_deep_is_refused_never_raised(arg):
  """The parser recursed once a level: ~980 unary minuses, a source of
  1,000 characters, raised RecursionError out of the define, and the served
  process ended (#500). A chain of `and`s nests as deep in what it builds,
  and in `_eval` running it."""
  [why] = refusals(f"def x():\n  wait({arg})\n")
  assert why == f"line 2, wait: an expression nests deeper than {lang.MAX_EXPR_DEPTH}", why
  assert compile_ok("def x():\n  wait(" + "+".join(["1"] * 40) + ")\n")


@pytest.mark.parametrize("src, said", [
  ("def x():\n  wait(1)  # \ud800\n", "the source holds a character that is not text"),
  ("def x():\n  wait(" + "(" * 199 + "-" * 385 + "1" + ")" * 199 + ")\n",
   "the source nests too deeply to read"),
], ids=["a lone surrogate", "the parser's own stack"])
def test_a_source_python_cannot_read_is_refused_never_raised(src, said):
  """`ast.parse` raised UnicodeEncodeError on a lone surrogate and
  MemoryError past its own stack, and only SyntaxError was caught (#500)."""
  assert refusals(src) == [said]


def test_a_hand_in_a_world_with_no_hands_is_refused_by_the_world():
  assert compile_ok('def x():\n  open_hand("left")\n')
  reasons = refusals('def x():\n  open_hand("left")\n', replace(HOME, hands=()))
  assert any("hand='left' is not one of nothing this world has" in r for r in reasons)


def test_a_refused_find_beside_a_tag_that_passed_names_the_find():
  """#463's procedure: `find` asked for a cube and `place` given one. The
  refusal names the `find` on its line, so it reads as no refusal of the
  verb beside it that takes the same tag (here the hands' `spot`)."""
  [why] = refusals("def x():\n  spot(21)\n  find(20, 25.2, -2.4, 300)\n  spot(20)\n")
  assert why.startswith("line 3, find: tag 20 names no place this world has"), why


@pytest.mark.parametrize("call, said", [
  ("find(20, 25.2, -2.4)", "find: tag 20 names no place"),
  ("spot(1000)", "spot: tag=1000.0 is above 586.0"),
  ("open_hand(\"middle\")", "open_hand: hand='middle' is not one of left, right"),
  ("pull(\"left\", 1, 0, 0, 99, 0.1)", "pull: force=99.0 is above 20.0"),
  ("wait(99)", "wait: seconds=99.0 is above 60"),
  ("open_hand(3)", "open_hand: hand takes a name in quotes"),
  ("move(\"left_elbow\", read(\"nope\"))", "move: unknown sensor 'nope'"),
])
def test_a_refusal_the_world_gives_names_its_call(call, said):
  """The robot never sees a refused define's source again -- History keeps
  the reasons, the library only what it kept -- so `line 2` alone points
  at nothing it can read: each refusal names its call, as History says a
  step that failed (`it stopped at line 2, wait: ...`; #463)."""
  [why] = refusals(f"def x():\n  {call}\n")
  assert why.startswith(f"line 2, {said}"), why


@pytest.mark.parametrize("stmt", [
  'a = read("nope")', 'a += read("nope")', 'if read("nope") > 1:\n    wait(1)',
  'while 1 < read("nope"):\n    wait(1)'])
def test_an_unknown_sensor_outside_a_call_names_its_line(stmt):
  """A read in an assignment or a test is in no call to name, so the
  refusal names the read's own line, as every other refusal names one: a
  page's editor marks the line it names (rooftop-media-2026 #441), and
  marked none."""
  [why] = refusals(f"def x():\n  wait(1)\n  {stmt}\n")
  assert why.startswith("line 3: unknown sensor 'nope' (have: "), why


@pytest.mark.parametrize("verb", sorted(st.VERBS))
def test_every_refusal_inside_a_call_names_it_once(verb):
  """The parser's (an argument it cannot read, one too many, one it does
  not take, one it needs), and on a body that has none of the verbs the
  world's (not on this body), for every verb: each starts with its call,
  and none names it twice."""
  for facts in (HOME, replace(HOME, verbs=())):
    reasons = refusals(f"def x():\n  {verb}(None, zzz=1)\n", facts)
    assert reasons and all(r.startswith(f"line 2, {verb}: ") for r in reasons), reasons
    assert not any(f"{verb}: {verb}" in r for r in reasons), reasons
  assert any("not on this body yet" in r for r in reasons), reasons


@pytest.mark.parametrize("arg, said", [
  ("None", "a NoneType is not a number"),
  ("read(1)", "read takes one sensor name"),
  ("int(21)", "only read(...), sin, cos, atan2 and sqrt may be used inside an expression"),
  ("1 < 2 < 3", "compare two things with one of"),
  ("+21", "UnaryOp is not part of this language"),
])
def test_an_argument_the_parser_refused_is_not_refused_again_as_a_call(arg, said):
  """The parser stood a refused argument in as 0, and the world's check
  refused that 0 as well: `pick(int(t))` also read `tag 0 is no cube`, a
  tag the source never wrote (#463). One case for each place the parser
  refuses an expression (`find`'s tag now, 0 being no place either)."""
  [why] = refusals(f"def x():\n  find({arg}, 1, 1)\n")
  assert why.startswith(f"line 2, find: {said}"), why


@pytest.mark.parametrize("call", ["open_hand(None)", "open_hand(int(3))",
                                  "move(read(\"time\"), 1)"])
def test_a_name_argument_that_is_no_name_is_refused_once(call):
  """`fetch(None)` read `a NoneType is not a number` and `tool takes a name
  in quotes`: the parser refused the argument, then `validate` refused what
  it left (#463)."""
  [why] = refusals(f"def x():\n  {call}\n")
  assert why.endswith("takes a name in quotes"), why


def test_an_argument_given_by_position_and_by_keyword_is_refused():
  """Python refuses it; the language let the keyword overwrite the written
  one, and `find(35, 1, 2, x=5)` searched round x=5, a call never written."""
  [why] = refusals("def x():\n  find(35, 1, 2, x=5)\n")
  assert why == "line 2, find: x is given twice", why


def test_a_call_written_across_lines_is_named_at_its_own_line():
  """Each refusal inside a call names the call's line, as History names a
  step's: a part's own line read as a second call of one verb."""
  reasons = refusals("def x():\n  drive_to(\n    None,\n    99)\n")
  assert len(reasons) == 2 and all(r.startswith("line 2, drive_to: ") for r in reasons), reasons


def test_the_sensors_are_listed_once_a_define():
  """A refused define is one History line of 400 characters: a misspelt
  read in three calls listed every sensor three times, and cut #463's own
  refusal after it short."""
  reasons = refusals("def x():\n  move(\"left_elbow\", read(\"shoulderr\"))\n"
                     "  move(\"right_elbow\", read(\"shoulderr\"))\n"
                     "  t = read(\"shoulderr\")\n  find(20, 25.2, -2.4)\n")
  assert sum("(have:" in r for r in reasons) == 1, reasons
  assert reasons[-1].startswith("line 5, find: tag 20 names no place"), reasons


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
    ramp_routine=routine("ramp"), handle=FIRST,
    actuator=_no_actuator, setpoint=lambda act: 0.0,
    go_to_routine=routine("drive_to", True), face_routine=routine("face", True),
    in_sight=lambda x, y: True, hold_routine=routine("wait"),
    pose=(0.0, 0.0, 0.0), watch=lambda tag: None,
    )
  return SimpleNamespace(body=body, data=SimpleNamespace(time=0.0, ctrl=[0.0]),
                         interrupted=lambda: False,
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
  ('def x():\n  n = 0\n  while 1:\n    n += 1\n  wait(1)\n', "loop-cap"),
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


def test_move_goes_through_the_bodys_ramp():
  r, life = run('def x():\n  move("left_elbow", 1.0)\n')
  assert r["ok"] and [c[0] for c in life.calls] == ["ramp"]


# ---- the fence, extended to loops ------------------------------------------------


def test_the_language_modules_write_no_control_register():
  from test_procedure import CTRL_WRITERS, _writes_ctrl
  for name in ("lang.py", "axes.py", "library.py", "steps.py"):
    path = SRC / "procedure" / name
    assert not _writes_ctrl(ast.parse(path.read_text()))
    assert f"procedure/{name}" not in CTRL_WRITERS


def test_the_arms_axes_ramp_at_its_slew_and_the_world_names_them_all():
  from pluggybot.humanoid.reach import JOINT_RATE
  assert {axes.AXES[n].speed for n in axes.HUMANOID_JOINTS} == {JOINT_RATE}
  assert all(a.lo < a.hi and a.speed > 0 for a in axes.AXES.values())
  # the served world names every axis and every sensor its body has, and
  # every one registered is its body's: the humanoid is the one body (#559)
  assert set(HOME.axes) == set(axes.BODY_AXES["humanoid"]) == set(axes.AXES)
  assert set(HOME.sensors) == set(axes.SENSORS)
  assert len(HOME.sensors) == len(set(HOME.sensors)), "a sensor named twice"


# ---- determinism: the same procedure on the same world is one trajectory -------


def _hash(data):
  return np.concatenate([data.qpos, data.qvel, data.ctrl]).tobytes()


def test_the_same_procedure_twice_is_one_trajectory():
  """On the served body: a sensor read off its encoders and a move computed
  from it, flown twice from one start, leave the world byte for byte the
  same."""
  from pluggybot.body import body_for
  from pluggybot.humanoid import model as hm
  src = '''def wiggle():
    wait(0.2)
    if read("left.z") < 2:
        move("left_elbow", read("left.z"))
'''
  proc = lang.compile_procedure(src, HOME)

  def fly():
    # ⚠ a world each: the drivers' gains live in the MODEL (a restart keeps
    # them, `HumanoidMission.kept_state`), so a model one flight has stepped
    # is not the world the next starts from
    model = mujoco.MjModel.from_xml_string(hm.body_xml())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    body = body_for(model, data)
    life = SimpleNamespace(body=body, model=model, data=data, interrupted=lambda: False,
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
    L.define("mine", "def other():\n  wait(1)\n")
  with pytest.raises(lib.LibraryRefused, match="unknown verb 'fly'"):
    L.define("bad", "def bad():\n  fly()\n")
  L.define("two", "def two():\n  wait(1)\n")
  with pytest.raises(lib.LibraryRefused, match="full"):
    L.define("three", "def three():\n  wait(1)\n")
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
    "sun", 'def sun():\n  open_hand("left")\n  wait(1)\n')
  moved = lib.Library(replace(HOME, hands=()), root=root)     # a world with no hands
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
                     "define": {"name": "x", "source": "def x():\n  wait(1)\n"},
                     "undefine": "sun"}, procedures=("sun",))
  assert d.action == "procedure:sun" and d.undefine == "sun"
  assert d.define == {"name": "x", "source": "def x():\n  wait(1)\n"}
  assert d.as_dict()["define"]["name"] == "x" and d.as_dict()["undefine"] == "sun"
  with pytest.raises(ValueError, match="no runnable procedure named 'nope'"):
    menu.validate({"action": "procedure:nope"}, procedures=("sun",))
  with pytest.raises(ValueError, match="unknown action 'procedure'"):
    menu.validate({"action": "procedure"}, procedures=("sun",))
  # no library offered: the fields are dropped, the action refused
  plain = Menu.for_world(WORLD)
  with pytest.raises(ValueError, match="unknown action"):
    plain.validate({"action": "procedure:sun"})
  d = plain.validate({"action": "idle",
                      "define": {"name": "x", "source": "def x():\n  wait(1)\n"},
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
  text = ov.hands_procedure_rule(menu.place_tags)
  example = text[text.index("  def wave"):text.index("Statements:")]
  for word in ("charge", "battery", "rack"):
    assert word not in example, word
  # ...and every verb, axis and sensor is listed, so nothing is a secret --
  # the places' `find` where it keeps places (#419)
  for v in HOME.verbs:
    assert f"  {v}(" in text, v
  for a in axes.BODY_AXES["humanoid"]:
    assert f"  {a}: " in text, a
  for s in HOME.sensors:
    assert f"  {s} --" in text, s
  # ...and none of the quadruped's verbs, another body's (#559)
  for v in ("fetch", "stow", "drive", "press", "draw", "pick", "place", "survey"):
    assert f"  {v}(" not in text, v
  assert "  find(" not in ov.hands_procedure_rule(()), "no places, no find"


# ---- the lifecycle: define, run by name ------------------------------------------


def _life(world=WORLD):
  return stub_life(world)


def test_a_decision_defines_and_undefines_and_every_refusal_is_narrated(tmp_path):
  life = _life()
  L = lib.Library(HOME, root=tmp_path / "p")
  life.overseer = SimpleNamespace(library=L)
  events: list = []
  life.on_event.append(events.append)
  life._define(Decision(action="idle",
                        define={"name": "two", "source": "def two():\n  wait(1)\n  wait(1)\n"}))
  assert L.names() == ("two",) and events[-1]["outcome"] == "defined"
  assert events[-1]["program"]["source"].startswith("def two")
  life._define(Decision(action="idle", define={"name": "two", "source": "def two():\n  wait(1)\n"}))
  assert events[-1]["outcome"] == "refused" and "already defined" in events[-1]["reasons"][0]
  assert any("refused" in line for line in life.log[-2:])
  life._define(Decision(action="idle", undefine="two",
                        define={"name": "two", "source": "def two():\n  wait(1)\n"}))
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
  life._define(Decision(action="idle", define={"name": "one", "source": "def one():\n  face(1)\n"}))
  assert events[-1]["library"] == {"names": ["one"], "cap": 2}
  life._define(Decision(action="idle", define={"name": "one", "source": "def one():\n  wait(1)\n"}))
  assert events[-1]["outcome"] == "refused" and events[-1]["library"] == {"names": ["one"], "cap": 2}
  life._define(Decision(action="idle", undefine="nope"))
  assert events[-1]["verb"] == "undefine" and events[-1]["library"]["names"] == ["one"]
  life._define(Decision(action="idle", undefine="one",
                        define={"name": "two", "source": "def two():\n  face(1)\n"}))
  assert [(e["outcome"], e["library"]["names"]) for e in events[-2:]] == [
    ("undefined", []), ("defined", ["two"])]


def test_a_procedure_runs_by_name_as_a_composed_errand():
  life = _life()
  L = lib.Library(HOME)
  L.define("pace", 'def pace():\n  n = 0\n'
                   '  while n < 2:\n    drive_to(1, 1)\n    n += 1\n  face(0.5)\n')
  errand = errand_from(Decision(action="procedure:pace"), WORLD, library=L)
  assert errand is not None and errand.name == "procedure"
  assert errand.program is L.get("pace")
  assert errand_from(Decision(action="procedure:nope"), WORLD, library=L) is None
  result = life.run_errand(errand)
  run = result["procedure"]
  assert run["ok"] and run["completed"] == 3 and result["stowed"]
  assert [s["verb"] for s in run["steps"]] == ["drive_to", "drive_to", "face"]
  assert result["verdict"]["ok"] and result["verdict"]["task"] == "program"


def test_a_procedure_cut_short_fails_its_verdict():
  life = _life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  L = lib.Library(HOME)
  L.define("short", 'def short():\n  wait(1)\n  drive_to(1, 1)\n  face(0.5)\n')
  result = life.run_errand(errand_from(Decision(action="procedure:short"),
                                       WORLD, library=L))
  assert result["procedure"]["failedAt"] == 1 and result["procedure"]["completed"] == 1
  assert result["stowed"] and not result["verdict"]["ok"]


def test_a_failed_run_names_the_line_that_failed_and_why():
  """`failedAt` counts verb calls EXECUTED, so inside a loop it names no
  line of the source (here the third call is on line 4, not the third
  verb in the text). The `aborted` event says the line and the step's own
  reason beside the count (rooftop-media-2026 #342), which is what lets a
  reader mark the failing line on the source rather than guess it; a run
  that went through carries neither."""
  life = _life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  life.drive_why = lambda x, y, record=None: "no route there on my map"
  events: list = []
  life.on_event.append(events.append)
  L = lib.Library(HOME)
  L.define("loop", 'def loop():\n  for i in range(2):\n    wait(0.1)\n  drive_to(1, 1)\n')
  L.define("fine", 'def fine():\n  for i in range(2):\n    wait(0.1)\n')
  for name in ("loop", "fine"):
    life.run_errand(errand_from(Decision(action=f"procedure:{name}"), WORLD, library=L))
  aborted, ran = [e for e in events if e.get("type") == "procedure"
                  and e["outcome"] in ("aborted", "ran")]
  assert aborted["name"] == "loop" and aborted["outcome"] == "aborted"
  assert (aborted["failedAt"], aborted["failedLine"]) == (2, 4)
  assert aborted["failedReason"] == ("did not arrive at (1, 1): no route there on my "
                                     "map, at (0.0, 0.0)")
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
  src = "def pace():\n  wait(1)\n  face(0.5)\n  wait(1)\n"
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


def test_a_verb_is_an_ifs_test_only_where_the_world_allows_it():
  """Ben, 2026-10-19: a move held back ended a lid's lowering half way with
  the jaws on its knob. Where `verb_tests` is on -- the humanoid's served
  world since #559 -- a verb may stand alone as an `if`'s or an `elif`'s
  test, `not` or not, and nowhere else in an expression; where it is off,
  nowhere."""
  src = ("def a():\n  if not wait(1):\n    wait(2)\n  elif wait(3):\n    wait(4)\n"
         "  else:\n    wait(5)\n")
  with pytest.raises(Refused, match="only read\\(\\.\\.\\.\\) may be used inside an "
                                    "expression; a verb is a statement of its own$"):
    compile_ok(src, replace(HOME, verb_tests=False, math=False))
  facts = HOME
  assert facts.verb_tests
  proc = compile_ok(src, facts)
  assert proc.verbs == 5
  arm = proc.body[0][1][0][0]
  assert arm[0] == "try" and arm[1] is True and arm[2][1] == "wait"
  for bad in ("def a():\n  if wait(1) and wait(2):\n    pass\n",
              "def a():\n  x = wait(1)\n",
              "def a():\n  while wait(1):\n    pass\n"):
    with pytest.raises(Refused, match="or alone as an `if`'s test"):
      compile_ok(bad, facts)
  # ...a tested verb is checked as any call is, and what it reads is named
  with pytest.raises(Refused, match="line 2, wait: seconds=999.0 is above 60"):
    compile_ok("def a():\n  if wait(999):\n    pass\n", facts)
  proc = compile_ok('def a():\n  if move("left_elbow", read("left.z")):\n    pass\n', facts)
  assert proc.first("move", "axis") == "left_elbow"
  with pytest.raises(Refused, match="line 2, move: unknown sensor 'elbow'"):
    compile_ok('def a():\n  if move("left_elbow", read("elbow")):\n    pass\n', facts)


def test_sin_cos_atan2_and_sqrt_only_where_the_world_gives_the_maths():
  """#493: an arc round a hinge, a pivot's angle and a distance need them --
  the humanoid's served world has them since #559; a world without the
  maths does not."""
  src = ("def a():\n  x = sin(0.5) + cos(0.5) * 2\n  y = atan2(1, 1)\n  z = sqrt(16)\n"
         "  wait(1)\n")
  with pytest.raises(Refused, match="only read\\(\\.\\.\\.\\) may be used inside an expression"):
    compile_ok(src, replace(HOME, math=False))
  facts = HOME
  assert facts.math
  proc = compile_ok(src, facts)
  life = SimpleNamespace(data=SimpleNamespace(time=0.0), interrupted=lambda: False,
                         _say=lambda *a, **k: None, reports={})
  env = {}
  assert lang._eval(proc.body[0][2], env, life, facts) == pytest.approx(
    math.sin(0.5) + 2 * math.cos(0.5))
  assert lang._eval(proc.body[1][2], env, life, facts) == pytest.approx(math.pi / 4)
  assert lang._eval(proc.body[2][2], env, life, facts) == pytest.approx(4.0)
  with pytest.raises(Refused, match="atan2 takes 2 number"):
    compile_ok("def a():\n  x = atan2(1)\n  wait(1)\n", facts)
  with pytest.raises(lang._Stop, match="sqrt of a negative number"):
    lang._eval(compile_ok("def a():\n  x = sqrt(0 - 4)\n  wait(1)\n", facts).body[0][2],
               env, life, facts)
