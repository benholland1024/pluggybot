"""The step vocabulary (procedure/steps.py, issue #58): an errand as
validated steps over the guarded primitives, ticked from the loop, one
verdict per step -- and the fence that keeps the vocabulary the only surface.
"""

import ast
import json
import pathlib
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import tick
from pluggybot.economy import scoring
from pluggybot.economy.tasks import Task
from pluggybot.lifecycle import (
  HubLifecycle, errand_for_task, world_config, world_facts,
)
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import steps as st
from pluggybot.procedure.steps import Program, Refused, Step

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
HOME = world_facts("home")
TABLE = scoring.challenge_table()


def two_tools(name="two-tools"):
  """The issue's acceptance shape: two tools, two places, one verdict."""
  return Program.single(name, [
    Step("fetch", {"tool": "module_pen"}),
    Step("drive_to", {"x": 1.5, "y": 1.8}),
    Step("draw", {"figure": "sun", "board": "whiteboard_a"}),
    Step("stow"),
    Step("fetch", {"tool": "module_lcd"}),
    Step("drive_to", {"x": 3.0, "y": 0.5}),
    Step("wait", {"seconds": 2.0}),
    Step("stow"),
  ])


# ---- the program is data ------------------------------------------------------


def test_a_program_round_trips_through_dict_and_json():
  p = two_tools()
  assert Program.from_dict(p.as_dict()) == p
  assert Program.from_json(p.to_json()) == p
  assert json.loads(p.to_json())["roles"] == {"robot": [s.as_dict() for s in p.steps()]}


def test_the_single_role_shorthand_is_stored_in_the_full_shape():
  """`roles` is the slot M12 fills; a person may type `steps` and gets the
  default role, and the stored shape is always the full one."""
  p = Program.from_dict({"name": "s", "steps": [{"verb": "look"}]})
  assert p.roles == {"robot": (Step("look"),)}
  assert set(p.as_dict()) == {"name", "budgetS", "roles"}


def test_the_vocabulary_is_the_issues_verbs():
  assert set(st.VERBS) == {"fetch", "stow", "drive_to", "face", "set_lift",
                           "grip", "release", "draw", "look", "wait",
                           # the motor level, issue #166
                           "move", "drive"}
  assert all(d["doc"] for d in st.describe_vocabulary())


# ---- validation: total, one test per rule, nothing partial ----------------


def _refusals(spec, facts=HOME):
  return st.validate(Program.from_dict(spec), facts)


@pytest.mark.parametrize("spec, needle", [
  ({"name": "", "steps": [{"verb": "look"}]}, "needs a name"),
  ({"name": "x", "roles": {}}, "at least one role"),
  ({"name": "x", "roles": {"robot": []}}, "has no steps"),
  ({"name": "x", "steps": [{"verb": "fly"}]}, "unknown verb 'fly'"),
  ({"name": "x", "steps": [{"verb": "wait"}]}, "wait needs seconds"),
  ({"name": "x", "steps": [{"verb": "stow", "args": {"tool": "module_pen"}}]},
   "stow takes no tool"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": "long"}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": float("nan")}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": True}}]},
   "must be a finite number"),
  ({"name": "x", "steps": [{"verb": "wait", "args": {"seconds": 999}}]},
   "is above 60"),
  ({"name": "x", "steps": [{"verb": "set_lift", "args": {"height": -0.1}}]},
   "is below 0.02"),
  ({"name": "x", "steps": [{"verb": "face", "args": {"heading": 7.0}}]},
   "is above 3.14"),
  ({"name": "x", "steps": [{"verb": "drive_to", "args": {"x": 99.0, "y": 0.0}}]},
   "outside the map"),
  ({"name": "x", "steps": [{"verb": "fetch", "args": {"tool": "module_x"}}]},
   "tool='module_x' is not one of"),
  ({"name": "x", "steps": [{"verb": "fetch", "args": {"tool": 3}}]},
   "must be a name"),
  ({"name": "x", "steps": [{"verb": "draw", "args": {"figure": "sun",
                                                       "board": "whiteboard_z"}}]},
   "board='whiteboard_z' is not one of"),
  ({"name": "x", "steps": [{"verb": "draw", "args": {"figure": "answer",
                                                       "board": "whiteboard_a"}}]},
   "figure='answer' is not one of"),
  ({"name": "x", "steps": [{"verb": "look"}] * (st.MAX_STEPS + 1)},
   f"the cap is {st.MAX_STEPS}"),
  ({"name": "x", "budgetS": st.MAX_BUDGET_S + 1, "steps": [{"verb": "look"}]},
   "budgetS"),
  ({"name": "x", "budgetS": 0, "steps": [{"verb": "look"}]}, "budgetS"),
])
def test_each_rule_refuses_with_a_readable_reason(spec, needle):
  reasons = _refusals(spec)
  assert any(needle in r for r in reasons), reasons


def test_a_world_without_boards_refuses_a_draw():
  reasons = _refusals({"name": "x", "steps": [
    {"verb": "draw", "args": {"figure": "sun", "board": "whiteboard_a"}}]},
    world_facts("room_hub"))
  assert any("nothing this world has" in r for r in reasons), reasons


def test_every_reason_is_reported_at_once():
  reasons = _refusals({"name": "x", "steps": [{"verb": "fly"},
                                              {"verb": "wait"}]})
  assert len(reasons) == 2


def test_a_valid_program_has_no_reasons():
  assert st.validate(two_tools(), HOME) == []
  assert st.compile_program(two_tools(), HOME) == two_tools()


def _stub_life(clock=None):
  """A life whose every primitive records itself and steps nothing: what is
  under test is the runner, not a mission."""
  calls: list = []

  def routine(name, value=True):
    def make(*a, **kw):
      calls.append((name, a))
      return tick.result(value)
    return make
  swap = SimpleNamespace(
    module_state=lambda tool: {"on_fork": False, "hung": True},
    set_lift_routine=routine("set_lift"))
  mission = SimpleNamespace(
    swap=swap, swap_at_bay_routine=routine("swap", "arrived"),
    drive_to_routine=routine("drive_to", True),
    face_routine=routine("face", True), _drive_routine=routine("wait"),
    pose=(0.0, 0.0, 0.0), tags=SimpleNamespace(detect=lambda d: {}))
  life = SimpleNamespace(mission=mission, data=SimpleNamespace(time=0.0),
                         module="", swaps_done=0, interrupted=lambda: False,
                         _say=lambda *a, **k: None, calls=calls,
                         model=None, world="home", boards=None)
  return life


def test_a_refused_program_never_runs_a_step():
  """Invalid means NOTHING runs -- not "runs until the bad step"."""
  life = _stub_life()
  bad = Program.single("bad", [Step("wait", {"seconds": 1.0}), Step("fly")])
  with pytest.raises(Refused) as e:
    tick.run(SimpleNamespace(_step_once=lambda *a: None),
             st.run_program_routine(life, bad, HOME))
  assert "unknown verb 'fly'" in str(e.value)
  assert life.calls == []


def test_two_roles_validate_but_one_robot_refuses_to_run_them():
  """The slot exists for M12 and is not implemented here -- refused with
  the reason, never silently run as one."""
  p = Program(name="hide", roles={"hider": (Step("look"),),
                                  "seeker": (Step("look"),)})
  assert st.validate(p, HOME) == []
  with pytest.raises(Refused, match="2 roles need 2 robots"):
    tick.run(SimpleNamespace(_step_once=lambda *a: None),
             st.run_program_routine(_stub_life(), p, HOME))


# ---- per-step verdicts, on a stubbed life --------------------------------------


def _run(life, program):
  return tick.run(SimpleNamespace(_step_once=lambda *a: None),
                  st.run_program_routine(life, program, HOME))


def test_every_step_gets_a_verdict_and_the_order_is_the_programs():
  life = _stub_life()
  p = Program.single("p", [Step("wait", {"seconds": 1.0}),
                           Step("face", {"heading": 1.0}),
                           Step("drive_to", {"x": 1.0, "y": 1.0}),
                           Step("look")])
  r = _run(life, p)
  assert r["ok"] and r["completed"] == 4 and r["total"] == 4
  assert [s["verb"] for s in r["steps"]] == ["wait", "face", "drive_to", "look"]
  assert all(s["ok"] for s in r["steps"])
  assert [c[0] for c in life.calls] == ["wait", "face", "drive_to"]


def test_a_failed_step_stops_the_program_with_an_honest_partial_result():
  life = _stub_life()

  def short(*a, **kw):
    life.calls.append(("drive_to", a))
    return tick.result(False)
  life.mission.drive_to_routine = short
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
  life.mission._drive_routine = slow_wait
  p = Program.single("p", [Step("wait", {"seconds": 1.0})] * 3, budget_s=150.0)
  r = _run(life, p)
  assert r["stopped"] == "budget" and r["completed"] == 2 and not r["ok"]


def test_an_interrupt_stops_a_program_between_steps():
  life = _stub_life()
  life.interrupted = lambda: True
  p = Program.single("p", [Step("look"), Step("look")])
  r = _run(life, p)
  assert r["stopped"] == "interrupted" and r["completed"] == 1


def test_a_tool_verb_without_its_tool_fails_rather_than_pretending():
  life = _stub_life()
  r = _run(life, Program.single("p", [Step("grip")]))
  assert not r["ok"] and r["steps"][0]["reason"] == "the claw is not on the fork"
  r = _run(life, Program.single("p", [Step("stow")]))
  assert r["steps"][0]["reason"] == "nothing on the fork to stow"


# ---- two tools, two places, one verdict (stubbed drives, no mission) ----------


def _life(world="room_hub"):
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  return HubLifecycle(model, mujoco.MjData(model), realtime=False, world=world,
                      errand=False, battery_wh=cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"])


def _stub_swaps(life, monkeypatch, fetch_ok=True, hung=True):
  """The swap stack as stubs: `swap_at_bay_routine` steps nothing, the
  coupling reports whatever the test says. `module_power_contact` is the
  seating criterion the fetch verb reads, so it is stubbed at the verb."""
  on_fork: dict = {}

  def swap_at_bay(station, verb, module=None, tries=2):
    on_fork.clear()
    if verb == "pick" and fetch_ok:
      on_fork[module] = True
    return tick.result("arrived")
  life.mission.swap_at_bay_routine = swap_at_bay
  life.mission.swap.module_state = lambda tool: {
    "on_fork": on_fork.get(tool, False),
    "hung": hung and not on_fork.get(tool, False)}
  monkeypatch.setattr(st, "module_power_contact", lambda *a, **k: True)
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(True)
  life.mission._drive_routine = lambda *a, **kw: tick.result(None)
  return on_fork


def test_a_task_spanning_two_tools_and_two_places_resolves_to_one_verdict(monkeypatch):
  life = _life()
  _stub_swaps(life, monkeypatch)
  # room_hub has no boards, so the two-tool program here carries the LCD
  # somewhere and the claw somewhere else; the draw variant flies under
  # --endurance.
  p = Program.single("errand", [
    Step("fetch", {"tool": "module_lcd"}), Step("drive_to", {"x": 1.0, "y": 1.0}),
    Step("wait", {"seconds": 1.0}), Step("stow"),
    Step("fetch", {"tool": "module_claw"}), Step("drive_to", {"x": 2.0, "y": 2.0}),
    Step("stow")])
  events: list = []
  life.on_event.append(events.append)
  errand = programmed_errand(p)
  assert errand.task == "program" and errand.module == "module_lcd"
  result = life.run_errand(errand)
  run = result["procedure"]
  assert run["ok"] and run["completed"] == 7 and result["stowed"]
  verdict = scoring.evaluate("program", scoring.sample_program(life, errand, result, {}),
                             table=TABLE)
  assert verdict.ok and verdict.points == TABLE["program"].base
  assert verdict.reason == "errand: 7/7 steps, tools hung"
  assert [e["outcome"] for e in events if e["type"] == "procedure"] == \
    ["validated", "ran"]


def test_a_program_cut_short_is_stowed_and_scored_as_it_stands(monkeypatch):
  """Abort means stow: the drive fails at step 2 with the LCD on the fork,
  the loop hangs it back before the verdict, and the verdict is a failure
  that says how far it got."""
  life = _life()
  on_fork = _stub_swaps(life, monkeypatch)
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  p = Program.single("short", [Step("fetch", {"tool": "module_lcd"}),
                               Step("drive_to", {"x": 1.0, "y": 1.0}),
                               Step("stow")])
  events: list = []
  life.on_event.append(events.append)
  result = life.run_errand(programmed_errand(p))
  run = result["procedure"]
  assert run["failedAt"] == 1 and run["completed"] == 1
  assert on_fork == {}, "the LCD was left on the fork"
  assert result["stowed"] and run["toolsHung"]
  v = scoring.evaluate("program", scoring.sample_program(life, programmed_errand(p),
                                                          result, {}), table=TABLE)
  assert not v.ok and v.points == 0 and "1/3 steps, step 2 failed" in v.reason
  assert events[-1]["outcome"] == "aborted" and events[-1]["failedAt"] == 1


def test_a_refused_program_is_an_errand_that_did_nothing(monkeypatch):
  life = _life()
  _stub_swaps(life, monkeypatch)
  events: list = []
  life.on_event.append(events.append)
  bad = Program.single("bad", [Step("fetch", {"tool": "module_lcd"}), Step("fly")])
  result = life.run_errand(programmed_errand(bad))
  assert result["error"].startswith("refused:") and not result["picked"]
  assert events[-1]["outcome"] == "refused" and events[-1]["reasons"]
  v = scoring.evaluate("program", scoring.sample_program(
    life, programmed_errand(bad), result, {}), table=TABLE)
  assert not v.ok and "refused" in v.reason


def test_the_generic_verdict_reads_the_rack_not_the_runner():
  """A runner that says every step passed is still failed by a tool that is
  not hung: `toolsHung` is read off the swap in the sampler."""
  life = SimpleNamespace(mission=SimpleNamespace(swap=SimpleNamespace(
    module_state=lambda t: {"hung": False})))
  result = {"procedure": {"program": "p", "total": 2, "completed": 2, "ok": True,
                          "steps": [{"verb": "fetch", "ok": True, "tool": "module_lcd"},
                                    {"verb": "stow", "ok": True}]}}
  m = scoring.sample_program(life, None, result, {})
  ok, _, reason = scoring.eval_program(m)
  assert not ok and "not back on its bay" in reason
  ok, _, reason = scoring.eval_program({})
  assert not ok and "never measured" in reason


# ---- serialisation: task state, the wire, a recording --------------------------


def test_a_program_rides_a_task_into_its_state_and_back(tmp_path):
  p = two_tools()
  task = Task.create("draw_figure", "whiteboard_a", "t_1",
                     params={"procedure": p.as_dict(), "program": "sun"},
                     description="draw a sun the composed way")
  state = json.loads(json.dumps(task.as_state()))
  assert Program.from_dict(state["params"]["procedure"]) == p
  back = Task.from_json(state)
  assert Program.from_dict(back.params["procedure"]) == p
  # ...and on the wire, the tasks block carries it whole
  assert Program.from_dict(task.as_dict()["params"]["procedure"]) == p
  # ...and the task builds the composed errand, graded by its own kind
  errand = errand_for_task(back, "home", book=None)
  assert errand is not None and errand.program == p and errand.task == "draw"
  assert errand.detail["board"] == "whiteboard_a" and errand.task_id == "t_1"


def test_a_task_carrying_an_invalid_program_builds_nothing():
  task = Task.create("draw_figure", "whiteboard_a", "t_2",
                     params={"procedure": {"name": "x", "steps": [{"verb": "fly"}]}})
  assert errand_for_task(task, "home", book=None) is None


def test_the_procedure_event_replays_from_a_recording(tmp_path):
  from pluggybot.telemetry.recorder import TelemetryRecorder
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  path = str(tmp_path / "rec.jsonl")
  rec = TelemetryRecorder(model, data, path, model_name="hub_world")
  p = two_tools()
  rec.emit({"type": "procedure", "t": 1.0, "robot": "pluggybot", "name": p.name,
            "outcome": "validated", "program": p.as_dict()})
  rec.close()
  events = [json.loads(line) for line in open(path) if '"procedure"' in line]
  assert len(events) == 1
  assert Program.from_dict(events[0]["program"]) == p


def test_the_example_programs_validate_against_home():
  """`scripts/programs/*.json` are what `hub_lifecycle.py --program` flies;
  an example that no longer validates is a demo that refuses to start."""
  from pluggybot.lifecycle import load_program
  for path in sorted((SRC.parents[1] / "scripts" / "programs").iterdir()):
    assert load_program(str(path), "home").name, path.name


# ---- the fence: the vocabulary is the only surface -----------------------------


#: The modules that may write the control register. Adding one is a design
#: decision made here, in the test, on purpose.
CTRL_WRITERS = {
  "rack/swap.py", "rack/coupling.py", "mission/mission.py",
  "tools/drawing.py", "tools/gripper.py", "tools/dispenser.py",
  "envs/dock_env.py", "docking/schuko.py",
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


# ---- the integration, on demand -------------------------------------------------


@pytest.mark.endurance
def test_a_composed_draw_produces_the_native_drawings_board_result():
  """The vocabulary is complete enough when a drawing composed from it lands
  the same ink as the native errand: same world, same start, same steps in
  the same order -- fetch, the standoff drive, the board, stow -- so on a
  deterministic world the board book should carry the same strokes."""
  from pluggybot.lifecycle import board_book, draw_errand_for
  from pluggybot.tools.drawing import Board, board_standoff

  def fly(build):
    life = _life("home")
    life.boards = board_book("home")
    cfg = world_config("home")
    life.mission.start_at(*cfg["start"])
    life.mission.start_discovery()
    life.mission._spin()
    life.run_errand(build(life))
    rec = life.boards["whiteboard_a"]
    return rec.strokes, [ln["points"] for ln in rec.lines]

  native = fly(lambda life: draw_errand_for("home", life.boards, "whiteboard_a",
                                            program_name="sun"))
  meta = json.loads(pathlib.Path(world_config("home")["meta"]).read_text())
  sx, sy = board_standoff(Board.from_meta(meta["boards"]["whiteboard_a"]))
  composed = fly(lambda life: programmed_errand(Program.single("sun", [
    Step("fetch", {"tool": "module_pen"}),
    Step("drive_to", {"x": sx, "y": sy}),
    Step("draw", {"figure": "sun", "board": "whiteboard_a"}),
    Step("stow")]), task="draw"))
  assert composed[0] == native[0] > 0
  assert composed[1] == native[1]
