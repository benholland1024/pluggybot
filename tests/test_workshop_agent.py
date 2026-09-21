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
  6. THE ORIGINALS ARE PERMANENT AND A BUILT TOOL HANGS ON ITS OWN RAIL
     (issue #277): `build_tool` names a bay of the built-tool rail and
     nothing else, `retire_tool` refuses the five hand-built modules with
     the reason, the grammar and the context say which rack is whose, and
     a world without the rail has no workshop at all.
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
from pluggybot.procedure.steps import TOOL_BAYS
from pluggybot.rack.coupling import built_bay_index
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


def test_the_idle_build_tool_a_decoder_emits_is_not_a_build(monkeypatch):
  """A constrained decoder fills every property, so an answer that is not
  building still carries `build_tool: {"name": "", "bay": "", "spec":
  {"name": "", "parts": []}}` (or an all-empty part). MEASURED (issue
  #264, ladder B): sent to the workshop, that was refused on 13 of 24
  answers in one day, each a `tool` row for a build the robot never
  described. Idle means "not this time"; a name, a bay or a named part
  means a build, and THAT is the workshop's to refuse."""
  menu = ov.Menu.for_world("room_hub")
  idle = {"name": "", "bay": "", "spec": {"name": "", "parts": []}}
  idle_part = {"name": "", "bay": "", "spec": {"name": "", "parts": [
    {"id": "", "part": "", "pos": [0, 0, 0], "euler": [0, 0, 0], "on": "",
     "size": [0, 0, 0], "axis": {"verb": "", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}}]}}
  # ...and a bay alone is not a build either: the decoder fills an enum
  # with its first member (`"bay": "A"`, measured on the second round)
  for shop in (idle, idle_part, {**idle, "bay": "A"}):
    assert ov.idle_build(shop)
    d = menu.validate({"action": "idle", "reason": "r", "build_tool": shop}, tools=())
    assert d.build_tool is None
  for shop in ({**idle, "name": "scoop"}, {**idle, "spec": {"name": "scoop", "parts": []}},
               {**idle, "spec": {"name": "", "parts": [{"id": "", "part": "servo_fs90"}]}}):
    assert not ov.idle_build(shop)
    assert menu.validate({"action": "idle", "reason": "r", "build_tool": shop},
                         tools=()).build_tool is not None


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
  assert hung["module"] == "module_scoop" and hung["retired"] is None
  assert hung["verbs"] == ["scoop.tilt"] and hung["cost"]["points"] == bill["points"]
  # paid: the catalog's price in points, off the balance, counted as spent
  assert life.ledger.balance() == 100 - bill["points"]
  assert bill["points"] == 3          # the servo's €2.90 and 9 g of PLA
  # waited: the print and assembly time, issued to the wait routine (the
  # one place the number is pinned; the routine itself is a stand-still)
  assert life.waited == [bill["waitS"]] and bill["waitS"] > 600
  # hung: the world has it -- in the RAIL's bay C, the originals untouched
  # -- the workshop records it, the wire saw the scene
  assert life.rack_inventory["module_scoop"] == built_bay_index(2)
  assert all(m in life.rack_inventory for m in TOOL_BAYS)
  assert life.overseer.workshop.names() == ("scoop",)
  assert any(e.get("type") == "scene_changed" for e in events)
  # ...and the record survives on disk, whole
  rec = json.loads((tmp_path / "tools" / "scoop.tool.json").read_text())
  assert rec["spec"] == SCOOP and rec["bay"] == 2
  # the model is shown both racks and its tools: the originals as a list
  # no field can name, its own rail by the letters `build_tool` takes
  state = overseer_context(life)
  assert state["rack"] == {"original": list(TOOL_BAYS),
                           "built": {"A": None, "B": None, "C": "module_scoop"}}
  assert state["tools"][0]["hung"] and state["tools"][0]["bay"] == "C"


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
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "A", "spec": SCOOP}))
  assert any("already built" in r for r in events[-1]["reasons"])


# ---- 6. the originals are permanent, and a built tool has its own rail ------

def test_a_hand_built_bay_cannot_be_named_and_the_refusal_says_whose_it_is(tmp_path):
  """`build_tool.bay` is the rail's A-C. The first rack's D and E were a
  build's to take until #277; an answer that still names one is refused
  with whose bay it is, and nothing is spent or retired."""
  life = _life(tmp_path, points=100)
  for letter, module in (("D", "claw"), ("E", "seed")):
    events = _run(life, _decision(build_tool={"name": "scoop", "bay": letter, "spec": SCOOP}))
    assert _outcomes(events) == ["specified", "refused"]
    reasons = events[-1]["reasons"]
    assert any(f"bay '{letter}' is not one of A, B, C" in r and module in r
               and "permanent" in r for r in reasons), reasons
  assert life.ledger.balance() == 100 and life.waited == []
  assert set(TOOL_BAYS) <= set(life.rack_inventory)
  # ...and the grammar never offered them: the enum is the rail's
  assert ov.BAY_LETTERS == BAYS == ("A", "B", "C")
  assert ov.Menu.for_world("room_hub").schema(tools=())["properties"]["build_tool"][
    "properties"]["bay"]["enum"] == ["A", "B", "C", ""]


@pytest.mark.parametrize("name", ["lcd", "plug", "pen", "claw", "seed"])
def test_retiring_an_original_is_refused_with_the_reason(tmp_path, name):
  life = _life(tmp_path)
  events = _run(life, _decision(retire_tool=name))
  assert _outcomes(events) == ["refused"]
  assert events[-1]["verb"] == "retire_tool"
  assert any("original modules" in r for r in events[-1]["reasons"]), events[-1]
  assert f"module_{name}" in life.rack_inventory
  assert life.overseer.workshop.refusals[-1]["name"] == name


def test_a_world_without_the_rail_has_no_workshop(monkeypatch):
  """The tower's shape (issue #207): where the world cannot hang a built
  tool, `build_tool` is not in the grammar and the prompt says nothing
  about building. Both served worlds carry the rail; a world that did not
  is made here by telling `build()` so through `world_config`."""
  from test_overseer import FakeClient
  from pluggybot import lifecycle
  def grammar(boss):
    return boss.menu.schema(tools=boss._tools(), procedures=boss._procedures())["properties"]
  with_rail = ov.build("room_hub", enabled=True, client=FakeClient(), autonomous=True)
  assert with_rail.workshop is not None and with_rail.menu.workshop
  assert "build_tool" in grammar(with_rail)
  assert "TOOLS YOU MAY BUILD" in "".join(t for _, t in with_rail.sections)
  real = lifecycle.world_config
  monkeypatch.setattr(lifecycle, "world_config",
                      lambda world: {**real(world), "built_bays": 0})
  without = ov.build("room_hub", enabled=True, client=FakeClient(), autonomous=True)
  assert without.workshop is None and not without.menu.workshop
  assert "build_tool" not in grammar(without)
  assert "TOOLS YOU MAY BUILD" not in "".join(t for _, t in without.sections)
  # ...and the count the grammar keys off is the count the world carries
  assert real("room_hub")["built_bays"] == len(BAYS) == real("home")["built_bays"]


def test_a_procedure_that_fetches_a_built_tool_goes_to_the_rail(tmp_path):
  """The errand a `procedure:` decision builds fetches the tool from the
  bay the inventory says -- on the rail, past the five. MEASURED on the
  flown proof: with `errand.py` still indexing the first rack's five, the
  rail's index raised, `errand_from` answered None and the day said
  "nothing to build" for a procedure the robot had just written."""
  from pluggybot.lifecycle import errand_from
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.procedure import lang
  from pluggybot.procedure.library import Library
  from pluggybot.lifecycle import world_facts
  from pluggybot.rack.coupling import BUILT_STATION_YS
  life = _life(tmp_path, points=100)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "B", "spec": SCOOP}))
  src = "def scoop_up():\n  fetch(\"module_scoop\")\n  stow()\n"
  proc = lang.compile_procedure(src, world_facts("room_hub", rack=life.rack_inventory))
  errand = programmed_errand(proc, rack=life.rack_inventory)
  assert errand.module == "module_scoop"
  assert errand.station_y == BUILT_STATION_YS[1]
  library = Library(world_facts("room_hub", rack=life.rack_inventory))
  library.define("scoop_up", src)
  built = errand_from(ov.Decision(action="procedure:scoop_up", reason=""), "room_hub", None,
                      library=library, rack=life.rack_inventory)
  assert built is not None and built.station_y == BUILT_STATION_YS[1]


def test_the_prompt_says_which_rack_is_whose():
  rule = ov.workshop_rule()
  assert '"bay": "<A-C>"' in rule and "3 bays, A-C" in rule
  assert "five original modules hang on the first and are permanent" in rule
  assert "none of them can be retired" in rule


def test_a_retire_empties_the_bay_and_a_build_may_take_it(tmp_path):
  life = _life(tmp_path)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  events = _run(life, _decision(retire_tool="scoop"))
  assert _outcomes(events) == ["retired"]
  assert "module_scoop" not in life.rack_inventory and "scoop.tilt" not in axes.AXES
  assert not (tmp_path / "tools" / "scoop.tool.json").exists()
  assert overseer_context(life)["rack"] == {"original": list(TOOL_BAYS),
                                           "built": {"A": None, "B": None, "C": None}}
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
  assert hung == ["scoop"] and again.rack_inventory["module_scoop"] == built_bay_index(2)
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
  assert [line.split(":")[0].strip() for line in usable] == [
    "bumper_switch", "servo_fs90", "servo_fs90mg", "slide_l12_100", "esp32_cam",
    "scaffold_pla_box"]
  assert "pi_camera_3" in rule and "draw unknown" in rule
  # a switch is told as a sense, and the prompt says what a sense is for
  assert "bumper_switch" in rule and "sense contact" in rule
  assert "<name>.<id>.contact" in rule
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


def _bare_objects(node, path=""):
  """Every `type: object` with neither `properties` nor `anyOf`."""
  found = []
  if isinstance(node, dict):
    if (node.get("type") == "object" and "properties" not in node
        and "anyOf" not in node):
      found.append(path)
    for key, value in node.items():
      found += _bare_objects(value, f"{path}/{key}")
  elif isinstance(node, list):
    for i, value in enumerate(node):
      found += _bare_objects(value, f"{path}[{i}]")
  return found


def test_the_spec_is_described_so_a_strict_provider_decodes_it():
  """`build_tool.spec` was a bare `{"type": "object"}` and the stricter
  providers behind the router refuse that before decoding a token ("Object
  fields require at least one of: 'properties' or 'anyOf'", issue #225) --
  every candidate on them fell to the prose retry and `constrained: false`.
  The described spec names exactly the fields `workshop/spec.py` accepts, so
  a field added there without a grammar entry fails here."""
  from pluggybot.workshop import spec as spec_mod
  schema = ov.Menu.for_world("room_hub").schema(tools=("scoop",))
  assert _bare_objects(schema) == []
  described = schema["properties"]["build_tool"]["properties"]["spec"]
  part = described["properties"]["parts"]["items"]
  assert set(part["properties"]) == spec_mod.PART_FIELDS
  assert set(part["properties"]["axis"]["properties"]) == spec_mod.AXIS_FIELDS
  assert set(described["properties"]) == spec_mod.SPEC_FIELDS
  # ...and the example the prompt shows fits the grammar: every field of
  # every part is one the grammar names (no jsonschema in the tree; the
  # shape is small enough to walk by hand).
  for part_ in SCOOP["parts"]:
    assert set(part_) <= set(part["properties"]), part_
    assert set(part_) >= set(part["required"]), part_
