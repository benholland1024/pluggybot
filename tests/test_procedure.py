"""The step vocabulary (procedure/steps.py, issue #58): an errand as
validated steps over the guarded primitives, ticked from the loop, one
verdict per step -- and the fence that keeps the vocabulary the only surface.
"""

import ast
import json
import math
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pluggybot import tick
from pluggybot.economy import scoring
from pluggybot.economy.tasks import Task
from pluggybot.lifecycle import errand_for_task, world_facts
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import steps as st
from pluggybot.procedure.steps import Program, Refused, Step
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
WORLD = "home_humanoid"
HOME = world_facts(WORLD)
TABLE = scoring.challenge_table()
#: A step any program may take, where what it is does not matter.
WAIT = {"verb": "wait", "args": {"seconds": 1.0}}


def two_places(name="two-places"):
  """The issue's acceptance shape: two places, one verdict (its tools went
  with the quadruped, #559)."""
  return Program.single(name, [
    Step("drive_to", {"x": 1.5, "y": 1.8}),
    Step("move", {"axis": "left_elbow", "target": 1.0}),
    Step("face", {"heading": 1.0}),
    Step("drive_to", {"x": 3.0, "y": 0.5}),
    Step("wait", {"seconds": 2.0}),
  ])


# ---- the program is data ------------------------------------------------------


def test_a_program_round_trips_through_dict_and_json():
  p = two_places()
  assert Program.from_dict(p.as_dict()) == p
  assert Program.from_json(p.to_json()) == p
  assert json.loads(p.to_json())["roles"] == {"robot": [s.as_dict() for s in p.steps()]}


def test_the_single_role_shorthand_is_stored_in_the_full_shape():
  """`roles` is the slot M12 fills; a person may type `steps` and gets the
  default role, and the stored shape is always the full one."""
  p = Program.from_dict({"name": "s", "steps": [WAIT]})
  assert p.roles == {"robot": (Step("wait", {"seconds": 1.0}),)}
  assert set(p.as_dict()) == {"name", "budgetS", "roles"}


def test_the_vocabulary_is_the_issues_verbs():
  assert set(st.VERBS) == {"drive_to", "face", "wait",
                           # the motor level, issue #166
                           "move",
                           # places, issue #419
                           "find"}
  assert all(d["doc"] for d in st.describe_vocabulary())


# ---- validation: total, one test per rule, nothing partial ----------------


def _refusals(spec, facts=HOME):
  return st.validate(Program.from_dict(spec), facts)


@pytest.mark.parametrize("spec, needle", [
  ({"name": "", "steps": [WAIT]}, "needs a name"),
  ({"name": "x", "roles": {}}, "at least one role"),
  ({"name": "x", "roles": {"robot": []}}, "has no steps"),
  ({"name": "x", "steps": [{"verb": "fly"}]}, "unknown verb 'fly'"),
  ({"name": "x", "steps": [{"verb": "wait"}]}, "robot[0], wait: needs seconds"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": 1.0, "tool": "module_pen"}}]},
   "robot[0], wait: takes no tool"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": "long"}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": float("nan")}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": True}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": 999}}]},
   "is above 60"),
  ({"name": "x", "steps": [{"verb": "face", "args": {"heading": -7.0}}]},
   "heading=-7.0 is below -3.14"),
  ({"name": "x", "steps": [{"verb": "face", "args": {"heading": 7.0}}]},
   "is above 3.14"),
  ({"name": "x", "steps": [{"verb": "drive_to", "args": {"x": 99.0, "y": 0.0}}]},
   "outside the map"),
  ({"name": "x", "steps": [{"verb": "move", "args": {"axis": 3, "target": 0.1}}]},
   "axis must be a name"),
  ({"name": "x", "steps": [{"verb": "move", "args": {"axis": "lift", "target": 0.1}}]},
   "axis='lift' is not one of left_shoulder_pitch, left_shoulder_roll"),
  ({"name": "x", "steps": [{"verb": "move", "args": {"axis": "left_elbow",
                                                       "target": 9.0}}]},
   "left_elbow target 9.0 is outside"),
  ({"name": "x", "steps": [{"verb": "find", "args": {"tag": 12, "x": 0.0, "y": 0.0}}]},
   "robot[0], find: tag 12 names no place this world has"),
  ({"name": "x", "steps": [WAIT] * (st.MAX_STEPS + 1)},
   f"the cap is {st.MAX_STEPS}"),
  ({"name": "x", "budgetS": st.MAX_BUDGET_S + 1, "steps": [WAIT]},
   "budgetS"),
  ({"name": "x", "budgetS": 0, "steps": [WAIT]}, "budgetS"),
])
def test_each_rule_refuses_with_a_readable_reason(spec, needle):
  reasons = _refusals(spec)
  assert any(needle in r for r in reasons), reasons


@pytest.mark.parametrize("verb, extra, other_kind", [
  ("find", {"x": 25.2, "y": -2.4}, 20)])
@pytest.mark.parametrize("tag", [None, 1000, float("inf"), float("nan")])
def test_a_tag_is_refused_once(verb, extra, other_kind, tag):
  """A bound said `tag=1000.0 is above 999` before the list's refusal, and a
  tag that is no number was refused twice -- where `int(tag)` had not raised
  on it first (#463). One reason: what the tag is not on, or that it is no
  number."""
  tag = other_kind if tag is None else tag
  [why] = st.check_step(st.VERBS[verb], {**extra, "tag": tag}, HOME)
  noun = st.VERBS[verb].args["tag"].choices[:-1]
  said = ("tag must be a finite number" if not math.isfinite(tag)
          else f"tag {tag} names no {noun} this world has")
  assert why.startswith(said), why


def test_a_reason_never_names_its_call():
  """Whoever knows where a step is names its call, once -- `validate`,
  `lang.validate`, History's `procedure_outcome` -- so a reason that named
  it too was said twice: `it stopped at line 3, pick: pick(tag=99) ...`."""
  bare = replace(HOME, verbs=())
  for name, verb in {**st.VERBS, **st.HANDS_VERBS, **st.GUARDED_VERBS}.items():
    for facts in (HOME, bare):
      for why in st.check_step(verb, {"zzz": 1}, facts):
        assert not why.startswith(name) and f"{name}(" not in why, why


def test_every_tag_is_checked_against_its_list():
  """A tag `Arg` carries no bound, so its list is its only check: a verb
  that took a tag with no list would take any number. (The hands' `spot`
  and `watch` look for any tag the camera can see, bounded instead.)"""
  tagged = [name for name, verb in st.VERBS.items() if "tag" in verb.args]
  assert tagged == ["find"]
  for name in tagged:
    assert st.VERBS[name].args["tag"].choices == "places", name


def test_a_tag_refusal_quotes_the_tag_exactly():
  """`:g` kept six digits: 23.0000001 was quoted `pick(tag=23)`, a call that
  passes, beside a list holding 23, and 1234567 as `1.23457e+06`; and a
  program's tag is an int, never rounded through a float. (`find` says it
  now, beside its places' list.)"""
  def find(tag):
    return st.check_step(st.VERBS["find"], {"tag": tag, "x": 25.2, "y": -2.4}, HOME)
  assert 35 in HOME.places
  [why] = find(35.0000001)
  assert why.startswith("tag 35.0000001 names no place"), why
  [why] = find(1234567)
  assert why.startswith("tag 1234567 names no place"), why
  [why] = find(2**53 + 1)
  assert why.startswith(f"tag {2**53 + 1} names no place"), why


def test_a_tag_refusal_lists_a_run_of_tags_as_lo_to_hi():
  """A refused define reaches the robot as ONE History line, cut at 400
  characters, and #463's own refused rewrite -- two `find`s, each listing
  the twelve places in full -- reached 401 once the call led them. A run of
  three or more is `lo to hi`, never `lo..hi`, which reads as leaving `hi` out."""
  [why] = st.check_step(st.VERBS["find"], {"tag": 20, "x": 25.2, "y": -2.4}, HOME)
  assert "(places: 35 to 46, 60 to 63)" in why, why
  odd = replace(HOME, places=(5, 7, 8, 9, 12, 23, 24))
  [why] = st.check_step(st.VERBS["find"], {"tag": 99, "x": 25.2, "y": -2.4}, odd)
  assert why.endswith("(places: 5, 7 to 9, 12, 23, 24)"), why


def test_a_coordinate_written_as_a_number_is_checked_beside_one_that_is_not():
  """A coordinate the parser refused stands in as nothing now, never as 0
  (#463), and one computed is checked when it is computed -- so a written
  one beside either is checked on its own, or a y off the map was said only
  after the robot had mended the x and defined again."""
  [why] = st.check_step(st.VERBS["drive_to"], {"y": 99999.0}, HOME, partial=True)
  assert why.startswith("y=99999.0 is outside the map's ["), why
  assert st.check_step(st.VERBS["find"], {"tag": 35, "x": 1.0}, HOME, partial=True) == []


def test_each_axis_is_checked_once_in_one_wording():
  """Both written, one off the map read `(1.0, 99999.0) is outside the map
  [...] x [...]`, beside one computed `y=99999.0 is outside the map's [...]`
  -- one fault, two wordings -- and an inf `check_arg` had refused was
  refused again as off the map (#463)."""
  [why] = st.check_step(st.VERBS["drive_to"], {"x": 1.0, "y": 99999.0}, HOME)
  assert why.startswith("y=99999.0 is outside the map's ["), why
  [why] = st.check_step(st.VERBS["drive_to"], {"x": float("inf")}, HOME, partial=True)
  assert why == "x must be a finite number, not inf", why
  [why] = st.check_step(st.VERBS["wait"], {"seconds": 1, "x": 10**400}, HOME)
  assert why == "takes no x", "a verb with no coordinates converts none"


def test_a_world_with_no_axes_refuses_a_move_and_says_so():
  """The WORLD decides a name argument's choices, and none is said as none."""
  step = {"verb": "move", "args": {"axis": "left_elbow", "target": 1.0}}
  assert _refusals({"name": "x", "steps": [step]}) == []
  reasons = _refusals({"name": "x", "steps": [step]}, replace(HOME, axes=()))
  assert any("axis='left_elbow' is not one of nothing this world has" in r
             for r in reasons), reasons


def test_every_reason_is_reported_at_once():
  reasons = _refusals({"name": "x", "steps": [{"verb": "fly"},
                                              {"verb": "wait"}]})
  assert len(reasons) == 2


def test_a_valid_program_has_no_reasons():
  assert st.validate(two_places(), HOME) == []
  assert st.compile_program(two_places(), HOME) == two_places()


def _no_actuator(name):
  raise KeyError(name)


def _stub_life(clock=None):
  """A life whose every primitive records itself and steps nothing: what is
  under test is the runner, not a mission. It has no arm to pose."""
  calls: list = []

  def routine(name, value=True):
    def make(*a, **kw):
      calls.append((name, a))
      return tick.result(value)
    return make
  body = SimpleNamespace(
    actuator=_no_actuator, ramp_routine=routine("ramp"),
    setpoint=lambda act: 0.0,
    go_to_routine=routine("drive_to", True), in_sight=lambda x, y: True,
    face_routine=routine("face", True), hold_routine=routine("wait"),
    pose=(0.0, 0.0, 0.0))
  life = SimpleNamespace(body=body, data=SimpleNamespace(time=0.0, ctrl=[0.0]),
                         interrupted=lambda: False,
                         _say=lambda *a, **k: None, calls=calls,
                         model=None, world=WORLD, boards=None,
                         drive_why=lambda x, y: "the drive gave up (why)")
  return life


def test_a_refused_program_never_runs_a_step():
  """Invalid means NOTHING runs -- not "runs until the bad step"."""
  life = _stub_life()
  bad = Program.single("bad", [Step("wait", {"seconds": 1.0}), Step("fly")])
  with pytest.raises(Refused) as e:
    tick.run(SimpleNamespace(step=lambda *a: None),
             st.run_program_routine(life, bad, HOME))
  assert "unknown verb 'fly'" in str(e.value)
  assert life.calls == []


def test_two_roles_validate_but_one_robot_refuses_to_run_them():
  """The slot exists for M12 and is not implemented here -- refused with
  the reason, never silently run as one."""
  wait = Step("wait", {"seconds": 1.0})
  p = Program(name="hide", roles={"hider": (wait,), "seeker": (wait,)})
  assert st.validate(p, HOME) == []
  with pytest.raises(Refused, match="2 roles: this robot must be told"):
    tick.run(SimpleNamespace(step=lambda *a: None),
             st.run_program_routine(_stub_life(), p, HOME))


# ---- per-step verdicts, on a stubbed life --------------------------------------


def _run(life, program):
  return tick.run(SimpleNamespace(step=lambda *a: None),
                  st.run_program_routine(life, program, HOME))


def test_every_step_gets_a_verdict_and_the_order_is_the_programs():
  life = _stub_life()
  p = Program.single("p", [Step("wait", {"seconds": 1.0}),
                           Step("face", {"heading": 1.0}),
                           Step("drive_to", {"x": 1.0, "y": 1.0})])
  r = _run(life, p)
  assert r["ok"] and r["completed"] == 3 and r["total"] == 3
  assert [s["verb"] for s in r["steps"]] == ["wait", "face", "drive_to"]
  assert all(s["ok"] for s in r["steps"])
  assert [c[0] for c in life.calls] == ["wait", "face", "drive_to"]


def test_a_failed_step_stops_the_program_with_an_honest_partial_result():
  life = _stub_life()

  def short(*a, **kw):
    life.calls.append(("drive_to", a))
    return tick.result(False)
  life.body.go_to_routine = short
  p = Program.single("p", [Step("wait", {"seconds": 1.0}),
                           Step("drive_to", {"x": 1.0, "y": 1.0}),
                           Step("face", {"heading": 0.0})])
  r = _run(life, p)
  assert not r["ok"] and r["completed"] == 1 and r["failedAt"] == 1
  assert len(r["steps"]) == 2 and not r["steps"][1]["ok"]
  assert [c[0] for c in life.calls] == ["wait", "drive_to"], "ran past the failure"


def test_the_budget_stops_a_program_at_a_step_boundary():
  life = _stub_life()

  def slow_wait(seconds, *a, **kw):
    life.data.time += 100.0
    return tick.result(None)
  life.body.hold_routine = slow_wait
  p = Program.single("p", [Step("wait", {"seconds": 1.0})] * 3, budget_s=150.0)
  r = _run(life, p)
  assert r["stopped"] == "budget" and r["completed"] == 2 and not r["ok"]


def test_an_interrupt_stops_a_program_between_steps():
  life = _stub_life()
  life.interrupted = lambda: True
  p = Program.single("p", [Step("wait", {"seconds": 1.0})] * 2)
  r = _run(life, p)
  assert r["stopped"] == "interrupted" and r["completed"] == 1


# ---- two places, one verdict (on the stub) -------------------------------------


def test_a_task_spanning_two_places_resolves_to_one_verdict():
  life = stub_life()
  p = Program.single("errand", [
    Step("drive_to", {"x": 1.0, "y": 1.0}), Step("wait", {"seconds": 1.0}),
    Step("face", {"heading": 0.5}), Step("drive_to", {"x": 2.0, "y": 2.0})])
  events: list = []
  life.on_event.append(events.append)
  errand = programmed_errand(p)
  assert errand.task == "program"
  result = life.run_errand(errand)
  run = result["procedure"]
  assert run["ok"] and run["completed"] == 4 and result["stowed"]
  verdict = scoring.evaluate("program", scoring.sample_program(life, errand, result, {}),
                             table=TABLE)
  assert verdict.ok and verdict.points == TABLE["program"].base
  assert verdict.reason == "errand: 4/4 steps, tools hung"
  assert [e["outcome"] for e in events if e["type"] == "procedure"] == \
    ["validated", "ran"]


def test_a_program_cut_short_is_scored_as_it_stands():
  """The drive fails at step 2: the verdict is a failure that says how far
  it got, and the run's event says where it stopped."""
  life = stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  p = Program.single("short", [Step("wait", {"seconds": 1.0}),
                               Step("drive_to", {"x": 1.0, "y": 1.0}),
                               Step("face", {"heading": 0.5})])
  events: list = []
  life.on_event.append(events.append)
  result = life.run_errand(programmed_errand(p))
  run = result["procedure"]
  assert run["failedAt"] == 1 and run["completed"] == 1
  assert result["stowed"] and run["toolsHung"]          # the quadruped's words, kept
  v = scoring.evaluate("program", scoring.sample_program(life, programmed_errand(p),
                                                          result, {}), table=TABLE)
  assert not v.ok and v.points == 0 and "1/3 steps, step 2 failed" in v.reason
  assert events[-1]["outcome"] == "aborted" and events[-1]["failedAt"] == 1


def test_a_refused_program_is_an_errand_that_did_nothing():
  life = stub_life()
  events: list = []
  life.on_event.append(events.append)
  bad = Program.single("bad", [Step("wait", {"seconds": 1.0}), Step("fly")])
  result = life.run_errand(programmed_errand(bad))
  assert result["error"].startswith("refused:") and not result["picked"]
  assert events[-1]["outcome"] == "refused" and events[-1]["reasons"]
  v = scoring.evaluate("program", scoring.sample_program(
    life, programmed_errand(bad), result, {}), table=TABLE)
  assert not v.ok and "refused" in v.reason


def test_a_run_nothing_measured_is_never_a_pass():
  """A missing measurement is not a passing one: an errand whose result
  carries no run is sampled as nothing, and graded as never measured."""
  m = scoring.sample_program(SimpleNamespace(), None, {}, {})
  assert m["total"] is None and m["toolsHung"] is None
  ok, _, reason = scoring.eval_program(m)
  assert not ok and reason == "the procedure was never measured"
  ok, _, reason = scoring.eval_program({})
  assert not ok and "never measured" in reason


def test_a_run_its_own_budget_stopped_fails_whole_and_says_so_without_a_fraction():
  """Issue #264: a procedure's `total` is the calls it MADE, so a budget stop
  after every call succeeded read "2/2 steps, tools hung" -- graded ok, and
  banked into History beside the line saying the run did not finish."""
  m = {"program": "p", "total": 2, "completed": 2, "failedAt": None,
       "stopped": "budget", "toolsHung": True}
  ok, _, reason = scoring.eval_program(m)
  assert not ok and reason == "p: did not finish -- stopped (budget) after 2 steps"


# ---- serialisation: task state, the wire, a recording --------------------------


@pytest.fixture
def errand_kind(monkeypatch):
  """A kind a task's own procedure discharges (#58), graded by the generic
  `program` evaluator: no served kind is one now (the quadruped's errand
  jobs went with it, #559), and `errand_for_task` still builds one."""
  from pluggybot.economy import tasks
  kind = replace(tasks.KINDS["open_drawer"], name="walk_there", task="program",
                 discharge="errand")
  monkeypatch.setitem(tasks.KINDS, kind.name, kind)
  return kind.name


def test_a_program_rides_a_task_into_its_state_and_back(errand_kind):
  p = two_places()
  task = Task.create(errand_kind, "drawer", "t_1",
                     params={"procedure": p.as_dict(), "minutes": 5},
                     description="a job carrying the procedure that does it")
  state = json.loads(json.dumps(task.as_state()))
  assert Program.from_dict(state["params"]["procedure"]) == p
  back = Task.from_json(state)
  assert Program.from_dict(back.params["procedure"]) == p
  # ...and on the wire, the tasks block carries it whole
  assert Program.from_dict(task.as_dict()["params"]["procedure"]) == p
  # ...and the task builds the composed errand, graded by its own kind
  errand = errand_for_task(back, WORLD, book=None)
  assert errand is not None and errand.program == p and errand.task == "program"
  assert errand.task_id == "t_1"


def test_a_task_carrying_an_invalid_program_builds_nothing(errand_kind):
  task = Task.create(errand_kind, "drawer", "t_2",
                     params={"procedure": {"name": "x", "steps": [{"verb": "fly"}]},
                             "minutes": 5})
  assert errand_for_task(task, WORLD, book=None) is None


#: The least a recorder records: a floor and a robot's root.
RECORDED_XML = """<mujoco><worldbody><geom type="plane" size="3 3 0.1"/>
  <body name="pluggybot" pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1"/>
  </body></worldbody></mujoco>"""


def test_the_procedure_event_replays_from_a_recording(tmp_path):
  import mujoco
  from pluggybot.telemetry.recorder import TelemetryRecorder
  model = mujoco.MjModel.from_xml_string(RECORDED_XML)
  data = mujoco.MjData(model)
  path = str(tmp_path / "rec.jsonl")
  rec = TelemetryRecorder(model, data, path, model_name=WORLD)
  p = two_places()
  rec.emit({"type": "procedure", "t": 1.0, "robot": "pluggybot", "name": p.name,
            "outcome": "validated", "program": p.as_dict()})
  rec.close()
  events = [json.loads(line) for line in open(path) if '"procedure"' in line]
  assert len(events) == 1
  assert Program.from_dict(events[0]["program"]) == p


# ---- the fence: the vocabulary is the only surface -----------------------------


#: The modules that may write the control register. Adding one is a design
#: decision made here, in the test, on purpose.
CTRL_WRITERS = {
  # issue #345: a restart PUTS BACK the controls the saved world held, by
  # actuator name -- the state the controllers below wrote, never a choice.
  "continuation.py",
  # issue #385, the quadruped's, the humanoid's since #559: its joints'
  # drivers -- a policy's joint targets and gains, a routine's torque, or
  # nothing held -- the one path every joint is commanded through. A
  # policy commands them; it does not write `ctrl`.
  "humanoid/drivers.py",
  # issue #507: the humanoid's grippers -- a Dex1-1's jaws opened to a width
  # or shut (`humanoid.arms.Gripper`), the one path its jaws are commanded
  # through; its joints are `humanoid/drivers.py`'s.
  "humanoid/arms.py",
}


def _writes_ctrl(tree) -> bool:
  for node in ast.walk(tree):
    targets = []
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for t in targets:
      if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
          and t.value.attr == "ctrl":
        return True
      if isinstance(t, ast.Attribute) and t.attr == "ctrl":
        return True
  return False


def test_no_new_path_writes_the_control_register():
  """⚠ THE FENCE. `data.ctrl` is written by the guarded primitives and by
  nothing else -- not the vocabulary, not the runner, not the lifecycle. A
  program can only reach the motors through a verb, which is what makes the
  ramping rule unbreakable however the steps are assembled."""
  writers = set()
  for path in sorted(SRC.rglob("*.py")):
    if _writes_ctrl(ast.parse(path.read_text(), filename=str(path))):
      writers.add(str(path.relative_to(SRC)))
  assert writers == CTRL_WRITERS, writers ^ CTRL_WRITERS
  assert not any(w.startswith("procedure/") for w in writers)


def test_the_fence_sees_a_write():
  assert _writes_ctrl(ast.parse("d.ctrl[3] = 1.0"))
  assert _writes_ctrl(ast.parse("self.data.ctrl[a] += 0.1"))
  assert not _writes_ctrl(ast.parse("x = d.ctrl[3]"))
