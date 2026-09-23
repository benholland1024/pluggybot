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


def _life(tmp_path, world="room_hub", points=100, workshop=None, step=True):
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
  # ⚠ `step=False` is the REAL path's state at `begin()`: a fresh `MjData`
  # that nothing has stepped, `xpos` all zeros (issue #315).
  for _ in range(200 if step else 0):
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


#: THE ROW THE DEPLOYED PAIR SENT 433 TIMES IN SEVEN DAYS (issue #315),
#: verbatim off the observatory: `"scoop"` is the prompt's own worked
#: example and `parts` is empty. Every one of the 433 was refused, and
#: `idle_build` did not drop it because the spec name was set.
DEPLOYED_IDLE = {"name": "", "bay": "A",
                 "spec": {"name": "scoop", "parts": []}}


def test_the_idle_build_tool_a_decoder_emits_is_not_a_build(monkeypatch):
  """A constrained decoder fills every property, so an answer that is not
  building still carries a whole `build_tool`. MEASURED (issue #264,
  ladder B): sent to the workshop, that was refused on 13 of 24 answers
  in one day, each a `tool` row for a build the robot never described.

  ⚠ THE PART LIST IS THE WHOLE TEST (issue #315). `spec.name` is a
  required string, so the decoder fills it with the nearest string the
  prefix offers -- the example's `"scoop"` -- and `bay` is an enum it
  fills with the first member. `parts` is the one required field whose
  zero the prompt cannot supply. A spec naming no part describes nothing
  to build, whatever it is called."""
  menu = ov.Menu.for_world("room_hub")
  idle = {"name": "", "bay": "", "spec": {"name": "", "parts": []}}
  idle_part = {"name": "", "bay": "", "spec": {"name": "", "parts": [
    {"id": "", "part": "", "pos": [0, 0, 0], "euler": [0, 0, 0], "on": "",
     "size": [0, 0, 0], "axis": {"verb": "", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}}]}}
  for shop in (idle, idle_part, {**idle, "bay": "A"}, DEPLOYED_IDLE,
               {**idle, "name": "scoop"},
               {**idle, "name": "scoop", "spec": {"name": "scoop", "parts": []}}):
    assert ov.idle_build(shop), shop
    d = menu.validate({"action": "idle", "reason": "r", "build_tool": shop}, tools=())
    assert d.build_tool is None
  # ...and ONE NAMED PART is a build, however wrong: an id the catalog
  # does not have is a real attempt and the workshop refuses it out loud,
  # which is the teaching path. A drop teaches nothing.
  for shop in ({**idle, "spec": {"name": "", "parts": [{"id": "", "part": "servo_fs90"}]}},
               {**idle, "spec": {"name": "", "parts": [{"id": "blade", "part": ""}]}},
               {**idle, "spec": {"name": "scoop", "parts": [{"id": "x", "part": "nope"}]}}):
    assert not ov.idle_build(shop), shop
    assert menu.validate({"action": "idle", "reason": "r", "build_tool": shop},
                         tools=()).build_tool is not None


def test_a_refusal_names_what_a_tool_may_be_built_from(tmp_path):
  """"If it IS an attempt, the refusal is useless" (issue #315): the part
  list is the field 433 specs never filled in, and `no catalog part
  'blade'` never said what would have worked. Both refusals now carry the
  buildable catalog -- one predicate, `spec.unbuildable`, so the list the
  prompt shows and the list a refusal names cannot drift apart."""
  from pluggybot.workshop import spec as wspec
  life = _life(tmp_path, points=100)
  bad = {"name": "scoop", "parts": [{"id": f"p{i}", "part": f"widget{i}",
                                     "pos": [-30, 0, -8]} for i in range(4)]}
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": bad}))
  assert _outcomes(events) == ["specified", "refused"]
  reasons = events[-1]["reasons"]
  catalog_lines = [r for r in reasons if "may be built from" in r]
  assert len(catalog_lines) == 1, reasons
  for part in wspec.buildable():
    assert part in catalog_lines[0], catalog_lines
  assert "servo_fs90" in catalog_lines[0] and "scaffold_pla_box" in catalog_lines[0]
  # ⚠ ONCE, and last: repeated beside each bad part it ran 598 chars on
  # this spec and a History line is 400 -- the robot read the catalog three
  # times and never saw that `widget3` was one of the parts it got wrong.
  joined = "; ".join(reasons)
  assert len(joined) < 400, len(joined)
  for i in range(4):
    assert f"no catalog part 'widget{i}'" in joined, joined
  assert life.ledger.balance() == 100 and life.waited == []
  # ...and the same list when the spec names no part at all -- reachable
  # from a hand-written or a restored spec, never from a decision now
  with pytest.raises(wspec.Refused) as e:
    wspec.parse({"name": "scoop", "parts": []})
  assert any("a spec IS its parts" in r and "servo_fs90" in r for r in e.value.reasons)
  # every part the PROMPT offers is a part a refusal names, and no other
  from pluggybot.rack import catalog
  assert [p.id for p in catalog.PARTS
          if wspec.unbuildable(p) is None] == wspec.buildable()


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
  assert state["rack"] == {
    "original": list(TOOL_BAYS),
    # ⚠ A BUILT BAY SAYS WHOSE IT IS (issue #324): the rail is the world's
    # and a pair shares it, so "the scoop is in C" was never the whole fact.
    "built": {"A": None, "B": None,
              "C": {"module": "module_scoop", "by": "you"}}}
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


def test_the_seam_is_checked_before_a_point_moves(tmp_path):
  """THE ORDER (issue #315): spec -> seam -> already-hung -> price -> PAY
  -> print -> hang. A build the seam would refuse is refused before
  `Ledger.spend` runs, so a paid, refused build is not a thing the robot
  can be handed. The one refusal that can arrive after the money is the
  seam breaking DURING the print (a death mid-fabrication), which says so
  with its `cost` on the event."""
  life = _life(tmp_path, points=100)
  life.state = "USE_TOOL"                       # what `can_reshape` refuses
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "refused"]
  assert any("between errands" in r for r in events[-1]["reasons"])
  assert "cost" not in events[-1]
  assert life.ledger.balance() == 100 and life.waited == []
  assert life.overseer.workshop.names() == ()


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


def test_a_built_tool_comes_back_after_a_restart_on_the_real_path(tmp_path):
  """`test_what_the_robot_built_hangs_again_next_day` calls `restore_tools`
  on a world that has been STEPPED. The real path does not: `begin()` runs
  it before anything has stepped, and a fresh `MjData` has `xpos` all
  zeros -- so `module_state` read every module as on the fork, `_carried`
  said `module_lcd`, `can_reshape` refused, and every built tool was lost
  with "could not be hung again" (issue #315). On a served world a restart
  is roughly hourly, so nothing built would ever have survived one."""
  shop = Workshop(tmp_path / "tools")
  life = _life(tmp_path, points=100, workshop=shop)
  _run(life, _decision(build_tool={"name": "scoop", "bay": "A", "spec": SCOOP}))
  assert "module_scoop" in life.rack_inventory
  # ...a new day, a new world, the records the only thing carried over
  again = _life(tmp_path, points=100, workshop=Workshop(tmp_path / "tools"),
                step=False)
  entry = again.overseer.workshop.entries["scoop"]
  assert entry.valid and entry.reasons == []
  assert "module_scoop" not in again.rack_inventory        # not yet
  again.begin((0.0, 0.0, 0.0), max_sim_time=10.0)
  assert again.rack_inventory.get("module_scoop") == built_bay_index(0)
  assert again.overseer.workshop.entries["scoop"].reasons == []
  assert again.ledger.balance() == 100                     # paid once


# ---- 7. a build is never paid for and lost (issue #315) ---------------------

def test_a_finished_build_waits_for_room_on_the_rack(tmp_path, monkeypatch):
  """⚠ A PRINT IS LONGER THAN AN ERRAND. The scoop's print and assembly is
  896 sim s; an errand runs 200-500. So on a pair the OTHER robot is
  usually mid-errand by the time the parts are ready, and `can_reshape`
  checking every robot -- which it must, because a peer's tool controller
  is never rebound -- turned that into a build PAID FOR AND LOST: no tool,
  no record, the points gone. That is the "paid, refused build" the issue
  set out to prevent, arriving through a door only a pair has.

  The assembly is done, so the tool hangs when there is room: the robot
  stands still (where it already was) and polls until the seam frees."""
  life = _life(tmp_path, points=100)
  # free when the build is priced and paid for -- or it is refused up front,
  # which is the OTHER half of the rule and is already pinned -- then busy
  # the moment it starts printing, as a peer taking an errand would.
  busy = {"why": ""}
  monkeypatch.setattr(HubLifecycle, "seam_busy", lambda self: busy["why"])
  real = life.mission._drive_routine

  def fabricate(seconds):
    life.waited.append(seconds)
    busy["why"] = "Rowan is busy: a tool is hung between errands"
    yield from real(0.2, 0.0, 0.0)
  life._fabricate_routine = fabricate

  # ...and the peer lets go while the finished tool stands waiting
  def freeing(sec, v, w):
    if life.waited and float(life.data.time) > 1.0:
      busy["why"] = ""
    yield from real(sec, v, w)
  life.mission._drive_routine = freeing

  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "A",
                                            "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "built", "hung"]
  assert "module_scoop" in life.rack_inventory
  assert life.ledger.balance() == 97                 # paid once, and it hung
  # ...and the HUNG row says how long it stood: a build that waited and
  # then hung is what a contended rack looks like, and an uncontended one
  # reads 0. Without it the observatory sees only the give-ups.
  assert events[-1]["waitedS"] > 0
  assert life.overseer.workshop.names() == ("scoop",)
  # ...and it did NOT wait for ever: the bound is a constant with an argument
  from pluggybot.lifecycle import HANG_WAIT_S, SEAM_POLL_S
  assert HANG_WAIT_S == 600.0 and SEAM_POLL_S == 5.0


def test_a_build_that_cannot_hang_is_kept_and_hangs_next_run(tmp_path, monkeypatch):
  """And when the rack never frees, THE POINTS STILL BUY SOMETHING. The
  tool is recorded, the robot is told, and `restore_tools` hangs it at the
  next mission start without paying again -- the same path a tool built
  yesterday takes. Losing the points AND the tool is what this forbids."""
  import pluggybot.lifecycle as lc
  monkeypatch.setattr(lc, "HANG_WAIT_S", 20.0)       # keep the test in ms
  life = _life(tmp_path, points=100)
  busy = {"why": ""}                                 # free until it prints
  monkeypatch.setattr(HubLifecycle, "seam_busy", lambda self: busy["why"])
  real = life.mission._drive_routine

  def fabricate(seconds):
    life.waited.append(seconds)
    busy["why"] = "Rowan is busy: mid-errand"
    yield from real(0.2, 0.0, 0.0)
  life._fabricate_routine = fabricate

  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "A",
                                            "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "built", "refused"]
  last = events[-1]
  assert last["verb"] == "hang" and last["waitedS"] >= 20.0
  assert any("hangs when the rack is free" in r for r in last["reasons"])
  assert any("Rowan is busy" in r for r in last["reasons"])
  assert "module_scoop" not in life.rack_inventory    # not on the rack
  assert life.ledger.balance() == 97                  # but paid, once
  assert life.overseer.workshop.names() == ("scoop",)  # ...and RECORDED
  # ⚠ AND THE ROBOT IS NOT TOLD IT HAS IT. `hung` in the context means ON
  # THE RACK; the entry still VALIDATES (that is what re-hangs it next
  # run), and telling the robot those are the same thing would leave it
  # believing it had a tool it does not.
  shown = life.overseer.workshop.as_context()[0]
  assert shown["hung"] is False
  assert any("hangs when the rack is free" in r for r in shown["reasons"])
  # ⚠ THE NEXT RUN HANGS IT, and does not pay again
  monkeypatch.undo()
  again = _life(tmp_path, points=100, workshop=Workshop(tmp_path / "tools"),
                step=False)
  again.begin((0.0, 0.0, 0.0), max_sim_time=10.0)
  assert again.rack_inventory.get("module_scoop") == built_bay_index(0)
  assert again.ledger.balance() == 100
  # ...and now it IS hung, so why it was not is stale and gone
  shown = again.overseer.workshop.as_context()[0]
  assert shown["hung"] is True and "reasons" not in shown


def test_the_wait_stops_rather_than_stranding_the_robot(tmp_path, monkeypatch):
  """Standing still is 10.5 W: the print alone is 2.6 Wh, a third of
  home's hosting pack, and a full 600 s wait on top would take it to 55 %
  against a 2.05 Wh reserve. The ROBOT chose to build; the WAITING is
  code's, so code stops spending the pack once what is left is the return
  trip's. Not a rail — it is on every arm, because on no arm should the
  loop's own retry be what strands the robot."""
  life = _life(tmp_path, points=100)
  busy = {"why": ""}
  monkeypatch.setattr(HubLifecycle, "seam_busy", lambda self: busy["why"])
  real = life.mission._drive_routine

  def fabricate(seconds):
    life.waited.append(seconds)
    busy["why"] = "Rowan is busy: mid-errand"      # ...and never frees
    yield from real(0.2, 0.0, 0.0)
  life._fabricate_routine = fabricate
  # a pack already down to its reserve: the clock has 600 s left to run
  life.battery.energy_wh = life.low_battery_wh
  t0 = float(life.data.time)

  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "A",
                                            "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "built", "refused"]
  assert events[-1]["waitedS"] < 5.0                # gave up at once
  assert float(life.data.time) - t0 < 5.0
  # ...and gave up is not lost: recorded, paid once, hangs next run
  assert life.overseer.workshop.names() == ("scoop",)
  assert life.ledger.balance() == 97


# ---- 8. whose tool is it, and which tool does a procedure need (issue #324) --

def test_a_built_bay_says_whose_tool_it_is():
  """One rail, two robots: a bay may hold the other robot's tool, which this
  robot may not take and may not retire. Until this it could only find out
  by trying. ⚠ The TAG cannot carry it — a built module's tag is `15 + bay`
  and belongs to the BAY, reused by whatever hangs there next — so the
  context is the only place ownership can be said."""
  from pluggybot.lifecycle import rack_context
  from pluggybot.pair import build_pair
  from pluggybot.workshop import validate
  a, b = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                    realtime=False)
  for _ in range(100):
    mujoco.mj_step(a.model, a.data)
  a.state = b.state = "DECIDE"
  a.hang_tool(validate.check(SCOOP), 0)
  b.hang_tool(validate.check({**SCOOP, "name": "grabber"}), 1)

  mine = rack_context(a.rack_inventory, a.built_by())["built"]
  theirs = rack_context(b.rack_inventory, b.built_by())["built"]
  assert mine["A"] == {"module": "module_scoop", "by": "you"}
  assert mine["B"] == {"module": "module_grabber", "by": b.robot_name}
  assert theirs["A"] == {"module": "module_scoop", "by": a.robot_name}
  assert theirs["B"] == {"module": "module_grabber", "by": "you"}
  assert mine["C"] is theirs["C"] is None
  # ...and it agrees with what the seam will actually refuse
  with pytest.raises(Exception, match="not yours to retire"):
    b.retire_tool("module_scoop")


def test_a_procedure_says_which_tool_it_needs(tmp_path):
  """`build.register` puts `requires=module_<name>` on a built tool's axes
  and sensors, so `move("scoop.tilt", ...)` without the scoop on the fork
  fails with a reason that names the module — but only at the point of
  failure, after an errand has been committed to it. The source was always
  shown, so the association was inferable; this states it."""
  from pluggybot.lifecycle import world_facts
  from pluggybot.procedure.library import Library
  from pluggybot.workshop import validate
  life = _life(tmp_path, points=100)
  life.hang_tool(validate.check(SCOOP), 0)
  lib = Library(world_facts("room_hub", rack=life.rack_inventory))
  lib.define("dig", 'def dig():\n  fetch("module_scoop")\n'
                    '  move("scoop.tilt", 1.0)\n  stow()')
  lib.define("look", 'def look():\n  wait(1.0)\n')
  rows = {r["name"]: r for r in lib.as_context()}
  assert rows["dig"]["needs"] == ["module_scoop"]
  assert "needs" not in rows["look"]        # absent, not empty: a learnable slot
  # the walk reaches into loops, branches and read() inside a condition
  lib.define("probe", 'def probe():\n  for i in range(2):\n'
                      '    if read("lift.force") > 0:\n'
                      '      move("claw.jaws", 0.02)\n')
  rows = {r["name"]: r for r in lib.as_context()}
  assert rows["probe"]["needs"] == ["module_claw"]


def test_retiring_a_tool_takes_its_procedures_out_of_the_enum(tmp_path):
  """An unknown axis is refused at `define` and again when the library
  LOADS, so a restart always marked a procedure whose tool was gone —
  ⚠ but nothing did it WITHIN a run, and the workshop can retire a tool
  mid-day. The procedure stayed in `runnable()` and in the action enum, and
  the robot found out by committing an errand to it and watching `move`
  fail. Kept and MARKED, never dropped: rebuilding the tool makes it
  runnable again, because this recompiles rather than remembering."""
  from pluggybot.lifecycle import world_facts
  from pluggybot.procedure.library import Library
  from pluggybot.workshop import validate
  life = _life(tmp_path, points=100)
  life.hang_tool(validate.check(SCOOP), 0)
  lib = Library(world_facts("room_hub", rack=life.rack_inventory))
  lib.define("dig", 'def dig():\n  fetch("module_scoop")\n'
                    '  move("scoop.tilt", 1.0)\n  stow()')
  life.overseer.library = lib
  assert lib.runnable() == ("dig",)

  life.retire_tool("module_scoop")
  assert lib.runnable() == ()                        # out of the enum
  row = lib.as_context()[0]
  assert row["runnable"] is False and "needs" not in row
  assert any("scoop.tilt" in r for r in row["reasons"])
  assert lib.entries["dig"].source                   # kept, not deleted

  life.hang_tool(validate.check(SCOOP), 0)
  assert lib.runnable() == ("dig",)                  # and it runs again
  assert lib.as_context()[0]["needs"] == ["module_scoop"]
