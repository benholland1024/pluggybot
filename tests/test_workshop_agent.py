"""The workshop as the agent's (issue #168, slice D): `build_tool` and
`retire_tool` as decision fields on the `autonomous` arm, a library of
what it built, the fabrication cost, the `tool` events.

What these hold down:

  1. `guarded` IS UNCHANGED. The fields, the rule and the grammar exist
     only where a workshop does; `guarded`'s prefix hash is
     `test_autonomous.py`'s and does not move.
  2. A BUILD PAYS, WAITS, HANGS -- in that order, each on the wire. Cheap:
     the decision is handed to the routine directly, no mind, no mission.
  3. A REFUSAL SPENDS NOTHING: the envelope, the price and the seam are
     checked before a point moves, and the reasons ride the event.
  4. WHAT THE ROBOT BUILT SURVIVES A RESTART: the records re-validate and
     re-hang at the start of the next day; the points are not paid twice.
  5. THE PROMPT'S EXAMPLE IS A CAPABILITY, NOT A POLICY.
"""

import copy
import json

import mujoco
import pytest

from pluggybot import tick
from pluggybot.economy.ledger import Ledger
from pluggybot.lifecycle import HubLifecycle, overseer_context, world_config
from pluggybot.mind import overseer as ov
from pluggybot.procedure import axes
from pluggybot.robot import world_spec
from pluggybot.workshop import cost
from pluggybot.workshop.library import BAYS, Workshop, WorkshopRefused
from test_workshop import SCOOP


@pytest.fixture(autouse=True)
def _clean_registries():
  before_a, before_s = set(axes.AXES), set(axes.SENSORS)
  yield
  for k in set(axes.AXES) - before_a:
    del axes.AXES[k]
  for k in set(axes.SENSORS) - before_s:
    del axes.SENSORS[k]


class _Mind:
  """What the lifecycle reads off an overseer on this path: the workshop,
  the library, and the map the physics seam consults every step."""
  event_map = None
  library = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None

  def __init__(self, workshop):
    self.workshop = workshop
    self.decisions = []
    self.menu = ov.Menu.for_world("room_hub")


def _life(tmp_path, world="room_hub", points=100, workshop=None):
  cfg = world_config(world)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  data = mujoco.MjData(model)
  ledger = Ledger(path=str(tmp_path / "ledger.json"))
  if points:
    ledger.intervene(points, t=0.0)
  life = HubLifecycle(model, data, realtime=False, world=world,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      spec=spec, errand=False, ledger=ledger,
                      overseer=_Mind(workshop if workshop is not None
                                     else Workshop(tmp_path / "tools")))
  life.state = "DECIDE"
  for _ in range(200):
    mujoco.mj_step(model, data)
  # The print and assembly wait is ~15 sim-minutes of physics; the fast
  # tests record it rather than step it (the seconds are pinned once in
  # `test_a_build_pays_waits_and_hangs`), on the rule that a stubbed
  # drive stubs the ROUTINE.
  life.waited: list = []

  def fabricate(seconds):
    life.waited.append(seconds)
    yield from life.mission._drive_routine(0.2, 0.0, 0.0)
  life._fabricate_routine = fabricate
  return life


def _decision(**fields):
  return ov.Decision(action="idle", reason="", **fields)


def _run(life, decision):
  events = []
  life.on_event.append(events.append)
  tick.run(life.mission.swap, life._workshop_routine(decision))
  return events


def _outcomes(events):
  return [e["outcome"] for e in events if e.get("type") == "tool"]


# ---- 1. guarded is unchanged -------------------------------------------------

def test_the_fields_exist_only_with_a_workshop():
  menu = ov.Menu.for_world("room_hub")
  plain = menu.schema()
  assert "build_tool" not in plain["properties"] and "retire_tool" not in plain["properties"]
  shop = menu.schema(tools=("scoop",))
  assert shop["properties"]["retire_tool"]["enum"] == ["scoop", ""]
  assert shop["properties"]["build_tool"]["properties"]["bay"]["enum"] == [*BAYS, ""]
  # a guarded parse DROPS the fields rather than raising on them
  raw = {"action": "idle", "reason": "r", "build_tool": {"name": "x", "bay": "A",
                                                          "spec": SCOOP}}
  d = menu.validate(raw)
  assert d.build_tool is None
  d = menu.validate(raw, tools=())
  assert d.build_tool == {"name": "x", "bay": "A", "spec": SCOOP}
  assert d.as_dict()["buildTool"]["bay"] == "A"


def test_guarded_prefix_does_not_move():
  from test_autonomous import GUARDED_RULES_SHA
  import hashlib
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  from pluggybot.economy.scoring import default_table
  from pluggybot.mind.thoughts import ThoughtFiles
  menu = ov.Menu.for_world("room_hub")
  args = (ThoughtFiles(), menu, default_table())
  assert "TOOLS YOU MAY BUILD" not in "".join(p["text"] for p in ov.system_prompt(*args))
  assert "TOOLS YOU MAY BUILD" in "".join(
    p["text"] for p in ov.system_prompt(*args, workshop=True))


# ---- 2. a build pays, waits, hangs ------------------------------------------

def test_a_build_pays_waits_and_hangs(tmp_path):
  life = _life(tmp_path, points=100)
  from pluggybot.workshop import validate
  bill = cost.price(validate.check(SCOOP))
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "built", "hung"]
  hung = next(e for e in events if e.get("outcome") == "hung")
  assert hung["module"] == "module_scoop" and hung["retired"] == "module_pen"
  assert hung["verbs"] == ["scoop.tilt"] and hung["cost"]["points"] == bill["points"]
  # paid: the catalog's price in points, off the balance, counted as spent
  assert life.ledger.balance() == 100 - bill["points"]
  assert bill["points"] == 3          # the servo's €2.90 and 9 g of PLA
  # waited: the print and assembly time, issued to the wait routine (the
  # one place the number is pinned; the routine itself is a stand-still)
  assert life.waited == [bill["waitS"]] and bill["waitS"] > 600
  # hung: the world has it, the workshop records it, the wire saw the scene
  assert life.rack_inventory["module_scoop"] == 2
  assert life.overseer.workshop.names() == ("scoop",)
  assert any(e.get("type") == "scene_changed" for e in events)
  # ...and the record survives on disk, whole
  rec = json.loads((tmp_path / "tools" / "scoop.tool.json").read_text())
  assert rec["spec"] == SCOOP and rec["bay"] == 2
  # the model is shown the rack and its tools
  state = overseer_context(life)
  assert state["rack"]["C"] == "module_scoop" and state["tools"][0]["hung"]


# ---- 3. a refusal spends nothing --------------------------------------------

def test_an_envelope_refusal_spends_nothing(tmp_path):
  life = _life(tmp_path, points=100)
  bad = copy.deepcopy(SCOOP)
  bad["parts"][1]["size"] = [260, 2, 2]
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": bad}))
  assert _outcomes(events) == ["specified", "refused"]
  refused = events[-1]
  assert refused["verb"] == "build_tool" and any(r.startswith("bed:") for r in refused["reasons"])
  assert life.ledger.balance() == 100
  assert "module_pen" in life.rack_inventory and life.overseer.workshop.names() == ()


def test_an_unaffordable_tool_is_refused_before_anything_prints(tmp_path):
  life = _life(tmp_path, points=1)
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "refused"]
  assert any("cannot afford it: 3 points" in r for r in events[-1]["reasons"])
  assert life.ledger.balance() == 1
  assert life.waited == []                          # no print time waited


def test_a_bad_bay_or_name_or_a_taken_one_is_refused(tmp_path):
  life = _life(tmp_path)
  events = _run(life, _decision(build_tool={"name": "Scoop!", "bay": "Z", "spec": SCOOP}))
  reasons = events[-1]["reasons"]
  assert any("not a name" in r for r in reasons) and any("bay 'Z'" in r for r in reasons)
  assert any("named 'scoop', not 'Scoop!'" in r for r in reasons)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "D", "spec": SCOOP}))
  assert any("already built" in r for r in events[-1]["reasons"])


def test_a_retire_empties_the_bay_and_a_build_may_take_it(tmp_path):
  life = _life(tmp_path)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  events = _run(life, _decision(retire_tool="scoop"))
  assert _outcomes(events) == ["retired"]
  assert "module_scoop" not in life.rack_inventory and "scoop.tilt" not in axes.AXES
  assert not (tmp_path / "tools" / "scoop.tool.json").exists()
  assert overseer_context(life)["rack"] == {"A": "module_lcd", "B": "module_plug",
                                           "D": "module_claw", "E": "module_seed"}
  events = _run(life, _decision(retire_tool="scoop"))
  assert _outcomes(events) == ["refused"]
  # an empty bay takes a build with nothing retired
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  assert next(e for e in events if e.get("outcome") == "hung")["retired"] is None


# ---- 4. survives a restart ---------------------------------------------------

def test_what_the_robot_built_hangs_again_next_day(tmp_path):
  life = _life(tmp_path, points=100)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  spent = 100 - life.ledger.balance()
  # a new process: the world from its file, the workshop from its records
  shop = Workshop(tmp_path / "tools")
  assert shop.names() == ("scoop",) and shop.entries["scoop"].valid
  again = _life(tmp_path, points=0, workshop=shop)
  again.ledger = life.ledger
  hung = again.restore_tools()
  assert hung == ["scoop"] and again.rack_inventory["module_scoop"] == 2
  assert "scoop.tilt" in axes.AXES
  assert life.ledger.balance() == 100 - spent      # paid once


def test_a_record_the_catalog_no_longer_validates_is_kept_and_marked(tmp_path):
  (tmp_path / "tools").mkdir()
  bad = copy.deepcopy(SCOOP)
  bad["parts"][0]["part"] = "module_servo"           # a candidate: refused today
  (tmp_path / "tools" / "scoop.tool.json").write_text(json.dumps(
    {"name": "scoop", "spec": bad, "bay": 2, "cost": {"points": 3}, "t": 1.0}))
  shop = Workshop(tmp_path / "tools")
  entry = shop.entries["scoop"]
  assert not entry.valid and any("not chosen" in r for r in entry.reasons)
  assert shop.as_context()[0]["hung"] is False
  with pytest.raises(WorkshopRefused):
    shop.check("scoop", SCOOP, "C")      # the name is taken by the record


# ---- 5. the cost and the prompt ---------------------------------------------

def test_the_cost_is_the_catalogs_price():
  from pluggybot.workshop import validate
  bill = cost.price(validate.check(SCOOP))
  assert bill["eur"] == pytest.approx(2.90 + 0.008928 * cost.FILAMENT_EUR_PER_KG, abs=0.01)
  assert bill["points"] == 3
  assert bill["printedG"] == pytest.approx(8.9, abs=0.1)
  assert bill["waitS"] == pytest.approx(8.928 * cost.PRINT_S_PER_G + 3 * cost.ASSEMBLE_S_PER_PART, abs=1)


def test_the_prompt_lists_only_what_can_be_built_from_and_says_why_not():
  rule = ov.workshop_rule()
  usable = [line for line in rule.splitlines()
            if line.startswith("  ") and " -- " in line and "CANNOT" not in line]
  assert [line.split(":")[0].strip() for line in usable] == ["servo_fs90", "scaffold_pla_box"]
  assert "pi_camera_3" in rule and "draw unknown" in rule
  # the envelope the robot is told is the one the code refuses against
  from pluggybot.rack import coupling
  assert f"under {coupling.MODULE_MASS_CEILING * 1000:.0f} g" in rule
  assert f"under {coupling.LATCH_MOMENT_NM:.2f} N·m" in rule


def test_the_example_is_a_capability_not_a_policy():
  head = ov.WORKSHOP_HEAD.lower()
  example = head[head.index('{"name": "scoop"'):head.index("the envelope")]
  for word in ("charge", "battery", "survive", "die"):
    assert word not in example


# ---- the integration, on demand -------------------------------------------------


@pytest.mark.endurance
def test_a_tool_the_agent_specified_is_built_fetched_used_and_stowed(tmp_path, monkeypatch):
  """The flown proof #168 asks for: the model specifies a tool, the world
  builds it mid-mission, the model writes a procedure that fetches it,
  moves the axis it named, and stows it -- and the procedure runs, with no
  scripted rotation anywhere (the arm is `autonomous`). Behind --endurance:
  every rule it exercises (the spec, the envelope, the price, the seam, the
  registries, the fetch by name, the stow) is pinned in milliseconds above
  and in tests/test_recompile.py. The print wait is shortened -- its length
  is pinned fast, and fifteen sim-minutes of standing still proves nothing
  about the integration."""
  from test_event_map import attach
  from test_overseer import FakeClient, full
  from pluggybot.lifecycle import run_demo
  monkeypatch.setattr(cost, "PRINT_S_PER_G", 1.0)
  monkeypatch.setattr(cost, "ASSEMBLE_S_PER_PART", 5.0)
  ledger = Ledger(path=str(tmp_path / "ledger.json"))
  ledger.intervene(50, t=0.0)
  src = ("def scoop_up():\n  fetch(\"module_scoop\")\n  move(\"scoop.tilt\", 1.2)\n"
         "  wait(1)\n  move(\"scoop.tilt\", 0)\n  stow()\n")
  answers = [
    full(action="idle", reason="building a scoop",
         build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}),
    full(action="idle", reason="writing how to use it",
         define={"name": "scoop_up", "source": src}),
    full(action="procedure:scoop_up", reason="using it"),
    full(action="idle", reason="done"),
  ]
  client = FakeClient(*answers)
  out = run_demo(view=False, realtime=False, world="room_hub", errand="none",
                 max_sim_time=300.0, battery_wh=3.0, overseer=True, standing_orders=True,
                 autonomous=True, origin="seeded",
                 thoughts_root=str(tmp_path / "t"),
                 ledger_state=str(tmp_path / "ledger.json"),
                 on_ready=attach(client))
  shop = out["overseer"]["workshop"]
  assert shop["built"] == 1 and shop["hung"] == 1 and shop["refused"] == 0
  assert shop["spentPoints"] == 3
  runs = [e for e in out["errands"] if e.get("procedure")]
  assert runs, out["errands"]
  proc = runs[0]["procedure"]
  assert proc["ok"] and proc["completed"] == 5, proc
  assert (tmp_path / "t" / "tools" / "scoop.tool.json").exists()
  assert not [d for d in out["decisions"] if ov.fallback_class(d["source"]) == "failure"]
