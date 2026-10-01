"""The workshop as the agent's (issue #168, slice D): `build_tool` and
`retire_tool` as decision fields of a mind with a workshop, a library of
what it built, the fabrication cost, the `tool` events.

No served world carries the built-tool rail since #376 (#407 re-homes the
workshop on legs), so the lifecycle's half runs on the stub: where a claim
is about what happens past the rail's own check, the rail is stood in
(`_life(rail=True)`); nothing here hangs a tool.

What these hold down:

  1. THE FIELDS, THE RULE AND THE GRAMMAR exist only where a workshop
     does.
  2. A REFUSAL SPENDS NOTHING: the envelope, the seam and the price are
     checked before a point moves, and the reasons ride the event.
  3. A PAID BUILD IS NEVER LOST: a print the rack has no room for when it
     is done, or a stand-up in the middle of, is recorded and said.
  4. THE PROMPT'S EXAMPLE IS A CAPABILITY, NOT A POLICY.
  5. THE ORIGINALS ARE PERMANENT AND A BUILT TOOL HANGS ON ITS OWN RAIL
     (issue #277): `build_tool` names a bay of the built-tool rail and
     nothing else, `retire_tool` refuses the hand-built modules with the
     reason, and a world without the rail has no workshop at all.
  6. WHOSE TOOL, AND WHICH TOOL A PROCEDURE NEEDS (issue #324).
"""

import copy
import json
from pathlib import Path

import mujoco
import pytest

from pluggybot.body import STUB_WORLD, StubBody
from pluggybot.economy.ledger import Ledger
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.mind import overseer as ov
from pluggybot.procedure import axes
from pluggybot.rack.coupling import BUILT_STATION_YS, built_bay_index
from pluggybot.robot import SECOND
from pluggybot.workshop import cost, seam
from pluggybot.workshop.library import BAYS, Workshop, WorkshopRefused
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_workshop import SCOOP

WORLD = "home_quad"


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
    self.menu = ov.Menu.for_world(WORLD)


def _life(tmp_path, points=100, workshop=None, rail=False):
  """A stub lifecycle whose world was compiled from a spec, with a wallet
  and a workshop; `rail` stands the built-tool rail in, so the seam's
  checks past it run. The print and assembly wait is recorded, not stood
  through (the seconds are pinned once, `test_the_cost_is_the_catalogs_price`)."""
  cfg = world_config(WORLD)
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.modelfiledir = str(Path(cfg["model"]).parent.resolve())
  model = spec.compile()
  ledger = Ledger(path=str(tmp_path / "ledger.json"))
  if points:
    ledger.intervene(points, t=0.0)
  life = stub_life(body=StubBody(model, mujoco.MjData(model), rack=cfg["rack"],
                                 grid_bounds=cfg["grid_bounds"]),
                   spec=spec, ledger=ledger,
                   overseer=_Mind(workshop if workshop is not None
                                  else Workshop(tmp_path / "tools")))
  life.state = "DECIDE"
  life.has_built_rack = rail
  life.waited: list = []

  def fabricate(seconds):
    life.waited.append(seconds)
    yield from life.body.hold_routine(0.2)
  life._fabricate_routine = fabricate
  return life


def _decision(**fields):
  return ov.Decision(action="idle", reason="", **fields)


def _run(life, decision):
  events = []
  life.on_event.append(events.append)
  life.body.run(life._workshop_routine(decision))
  return events


def _outcomes(events):
  return [e["outcome"] for e in events if e.get("type") == "tool"]


# ---- 1. only where a workshop is ---------------------------------------------

def test_the_fields_exist_only_with_a_workshop():
  menu = ov.Menu.for_world(WORLD)
  plain = menu.schema()
  assert "build_tool" not in plain["properties"] and "retire_tool" not in plain["properties"]
  shop = menu.schema(tools=("scoop",))
  assert shop["properties"]["retire_tool"]["enum"] == ["scoop", ""]
  assert shop["properties"]["build_tool"]["properties"]["bay"]["enum"] == [*BAYS, ""]
  # a parse with no workshop DROPS the fields rather than raising on them
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


def test_the_idle_build_tool_a_decoder_emits_is_not_a_build():
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
  menu = ov.Menu.for_world(WORLD)
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


def test_the_rule_rides_only_where_the_workshop_does():
  from pluggybot.economy.scoring import default_table
  from pluggybot.mind.thoughts import ThoughtFiles
  menu = ov.Menu.for_world(WORLD)
  args = (ThoughtFiles(), menu, default_table())
  assert "TOOLS YOU MAY BUILD" not in "".join(p["text"] for p in ov.system_prompt(*args))
  assert "TOOLS YOU MAY BUILD" in "".join(
    p["text"] for p in ov.system_prompt(*args, workshop=True))


# ---- 2. a refusal spends nothing --------------------------------------------

def test_an_envelope_refusal_spends_nothing(tmp_path):
  life = _life(tmp_path, points=100, rail=True)
  bad = copy.deepcopy(SCOOP)
  bad["parts"][1]["size"] = [260, 2, 2]
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": bad}))
  assert _outcomes(events) == ["specified", "refused"]
  refused = events[-1]
  assert refused["verb"] == "build_tool" and any(r.startswith("bed:") for r in refused["reasons"])
  assert life.ledger.balance() == 100
  assert "module_pen" in life.rack_inventory and life.overseer.workshop.names() == ()


def test_an_unaffordable_tool_is_refused_before_anything_prints(tmp_path):
  life = _life(tmp_path, points=1, rail=True)
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
  assert _outcomes(events) == ["specified", "refused"]
  assert any("cannot afford it: 3 points" in r for r in events[-1]["reasons"])
  assert life.ledger.balance() == 1
  assert life.waited == []                          # no print time waited


def test_the_seam_is_checked_before_a_point_moves(tmp_path):
  """THE ORDER (issue #315): spec -> seam -> already-hung -> price -> PAY
  -> print -> hang. A build the seam would refuse is refused before
  `Ledger.spend` runs, so a paid, refused build is not a thing the robot
  can be handed -- the served world's refusal (no rail) and a robot
  mid-errand where a rail stands alike."""
  for rail, state, why in ((False, "DECIDE", "no built-tool rack"),
                           (True, "USE_TOOL", "between errands")):
    life = _life(tmp_path, points=100, rail=rail)
    life.state = state
    events = _run(life, _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP}))
    assert _outcomes(events) == ["specified", "refused"]
    assert any(why in r for r in events[-1]["reasons"]), events[-1]["reasons"]
    assert "cost" not in events[-1]
    assert life.ledger.balance() == 100 and life.waited == []
    assert life.overseer.workshop.names() == ()


def test_a_bad_bay_or_name_or_a_taken_one_is_refused(tmp_path):
  life = _life(tmp_path)
  events = _run(life, _decision(build_tool={"name": "Scoop!", "bay": "Z", "spec": SCOOP}))
  reasons = events[-1]["reasons"]
  assert any("not a name" in r for r in reasons) and any("bay 'Z'" in r for r in reasons)
  assert any("named 'scoop', not 'Scoop!'" in r for r in reasons)
  # a tool the workshop has a record of is built, hung or not
  shop = life.overseer.workshop
  tool, bay = shop.check("scoop", SCOOP, "C")
  shop.record(tool, SCOOP, bay, cost.price(tool), 0.0)
  events = _run(life, _decision(build_tool={"name": "scoop", "bay": "A", "spec": SCOOP}))
  assert any("already built" in r for r in events[-1]["reasons"])


# ---- 3. a paid build is never lost (issue #315) ------------------------------

def test_a_build_that_cannot_hang_is_kept_and_said(tmp_path, monkeypatch):
  """When the rack never frees, THE POINTS STILL BUY SOMETHING. The tool
  is recorded and the robot is told, and `restore_tools` hangs it at the
  next mission start without paying again -- the same path a tool built
  yesterday takes. Losing the points AND the tool is what this forbids."""
  import pluggybot.lifecycle as lc
  monkeypatch.setattr(lc, "HANG_WAIT_S", 20.0)       # keep the test in ms
  life = _life(tmp_path, points=100, rail=True)
  busy = {"why": ""}                                 # free until it prints
  monkeypatch.setattr(HubLifecycle, "seam_busy", lambda self: busy["why"])

  def fabricate(seconds):
    life.waited.append(seconds)
    busy["why"] = "Rowan is busy: mid-errand"
    yield from life.body.hold_routine(0.2)
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
  rec = json.loads((tmp_path / "tools" / "scoop.tool.json").read_text())
  assert rec["spec"] == SCOOP and rec["bay"] == 0
  # ...and it did NOT wait for ever: the bound is a constant with an argument
  assert (lc.HANG_WAIT_S, lc.SEAM_POLL_S) == (20.0, 5.0)
  monkeypatch.undo()
  assert (lc.HANG_WAIT_S, lc.SEAM_POLL_S) == (600.0, 5.0)


def test_a_stand_up_during_the_print_keeps_the_paid_tool(tmp_path):
  """⚠ ISSUE #348: a stand-up closes the decision's action, and a build is
  one. The points are spent before the ~15-minute print and the tool was
  recorded only after it, so a robot that died `unpaid` two minutes into
  the print -- the balance spent on the build -- paid and got nothing,
  #315's "paid, refused build" by another door. Recorded like a rack that
  never freed. Shown to fail without the `GeneratorExit` branch."""
  from pluggybot.lifecycle import STOOD_UP
  life = _life(tmp_path, points=100, rail=True)
  life.home_pose = tuple(world_config(WORLD)["start"])

  def fabricate(seconds):
    yield from life.body.hold_routine(0.2)
    life.stand_up("ben", auto=False)              # lands mid-print
    while True:
      yield from life.body.hold_routine(0.2)
  life._fabricate_routine = fabricate

  events: list = []
  life.on_event.append(events.append)
  build = _decision(build_tool={"name": "scoop", "bay": "A", "spec": SCOOP})
  out = life.body.run(life._until_stood_up_routine(life._workshop_routine(build)))
  assert out is STOOD_UP
  assert life.ledger.balance() == 97                  # paid, once
  assert life.overseer.workshop.names() == ("scoop",)  # ...and RECORDED
  assert "module_scoop" not in life.rack_inventory
  assert _outcomes(events) == ["specified", "built", "refused"]
  assert any("a stand-up ended the wait" in r for r in events[-1]["reasons"])


def test_the_wait_stops_rather_than_stranding_the_robot(tmp_path, monkeypatch):
  """Standing still draws power: the ROBOT chose to build; the WAITING is
  code's, so code stops spending the pack once what is left is the return
  trip's. Not a rail -- the waiting is code's, and the loop's own retry
  must never be what strands the robot."""
  life = _life(tmp_path, points=100, rail=True)
  busy = {"why": ""}
  monkeypatch.setattr(HubLifecycle, "seam_busy", lambda self: busy["why"])

  def fabricate(seconds):
    life.waited.append(seconds)
    busy["why"] = "Rowan is busy: mid-errand"      # ...and never frees
    yield from life.body.hold_routine(0.2)
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


# ---- 5. the originals are permanent, and a built tool has its own rail ------

def test_a_bay_off_the_rail_cannot_be_named_and_nothing_is_spent(tmp_path):
  """`build_tool.bay` is the rail's A-C; an answer naming another letter is
  refused, and nothing is spent or retired."""
  life = _life(tmp_path, points=100, rail=True)
  for letter in ("D", "E"):
    events = _run(life, _decision(build_tool={"name": "scoop", "bay": letter, "spec": SCOOP}))
    assert _outcomes(events) == ["specified", "refused"]
    reasons = events[-1]["reasons"]
    assert any(f"bay '{letter}' is not one of A, B, C" in r for r in reasons), reasons
  assert life.ledger.balance() == 100 and life.waited == []
  # ...and the grammar never offered them: the enum is the rail's
  assert ov.BAY_LETTERS == BAYS == ("A", "B", "C")
  assert ov.Menu.for_world(WORLD).schema(tools=())["properties"]["build_tool"][
    "properties"]["bay"]["enum"] == ["A", "B", "C", ""]


@pytest.mark.parametrize("module", seam.HAND_BUILT)
def test_retiring_an_original_is_refused_with_the_reason(tmp_path, module):
  life = _life(tmp_path)
  had = module in life.rack_inventory
  name = module.removeprefix("module_")
  events = _run(life, _decision(retire_tool=name))
  assert _outcomes(events) == ["refused"]
  assert events[-1]["verb"] == "retire_tool"
  assert any("original modules" in r for r in events[-1]["reasons"]), events[-1]
  assert (module in life.rack_inventory) is had
  assert life.overseer.workshop.refusals[-1]["name"] == name


def test_a_world_without_the_rail_has_no_workshop(monkeypatch):
  """The tower's shape (issue #207): where the world cannot hang a built
  tool, `build_tool` is not in the grammar and the prompt says nothing
  about building. The served world has no rail; a world that did is made
  here by telling `build()` so through `world_config`."""
  from test_overseer import FakeClient
  from pluggybot import lifecycle

  def grammar(boss):
    return boss.menu.schema(tools=boss._tools(), procedures=boss._procedures())["properties"]
  real = lifecycle.world_config
  # the count the grammar keys off is the count the world carries
  assert real(WORLD)["built_bays"] == 0
  without = ov.build(WORLD, enabled=True, client=FakeClient())
  assert without.workshop is None and not without.menu.workshop
  assert "build_tool" not in grammar(without)
  assert "TOOLS YOU MAY BUILD" not in "".join(t for _, t in without.sections)
  monkeypatch.setattr(lifecycle, "world_config",
                      lambda world: {**real(world), "built_bays": len(BAYS)})
  with_rail = ov.build(WORLD, enabled=True, client=FakeClient())
  assert with_rail.workshop is not None and with_rail.menu.workshop
  assert "build_tool" in grammar(with_rail)
  assert "TOOLS YOU MAY BUILD" in "".join(t for _, t in with_rail.sections)


def test_a_procedure_that_fetches_a_built_tool_goes_to_the_rail():
  """The errand a `procedure:` decision builds fetches the tool from the
  bay the inventory says -- on the rail, past the first rack. MEASURED on
  the flown proof: with `errand.py` indexing the first rack alone, the
  rail's index raised, `errand_from` answered None and the day said
  "nothing to build" for a procedure the robot had just written."""
  from pluggybot.lifecycle import errand_from, world_facts
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.procedure import lang
  from pluggybot.procedure.library import Library
  from pluggybot.procedure.steps import TOOL_BAYS
  rack = {**TOOL_BAYS, "module_scoop": built_bay_index(1)}
  src = "def scoop_up():\n  fetch(\"module_scoop\")\n  stow()\n"
  proc = lang.compile_procedure(src, world_facts(WORLD, rack=rack))
  errand = programmed_errand(proc, rack=rack)
  assert errand.module == "module_scoop"
  assert errand.station_y == BUILT_STATION_YS[1]
  library = Library(world_facts(WORLD, rack=rack))
  library.define("scoop_up", src)
  built = errand_from(ov.Decision(action="procedure:scoop_up", reason=""), WORLD, None,
                      library=library, rack=rack)
  assert built is not None and built.station_y == BUILT_STATION_YS[1]


def test_the_prompt_says_which_rack_is_whose():
  rule = ov.workshop_rule()
  assert '"bay": "<A-C>"' in rule and "3 bays, A-C" in rule
  assert "none of them can be retired" in rule


# ---- 4. the cost and the prompt ---------------------------------------------

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


def test_a_decisions_build_reaches_the_workshop_through_the_loops_own_door(tmp_path):
  """A `build_tool` on an answer is applied by the decision loop's own door
  (`_after_decision_routine`), as a `define` is. Shown to fail by dropping
  `yield from self._workshop_routine(decision)` from it."""
  life = _life(tmp_path, points=100)
  events: list = []
  life.on_event.append(events.append)
  life.body.run(life._after_decision_routine(
    _decision(build_tool={"name": "scoop", "bay": "C", "spec": SCOOP})))
  assert _outcomes(events) == ["specified", "refused"]
  assert any("no built-tool rack" in r for r in events[-1]["reasons"])


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
  schema = ov.Menu.for_world(WORLD).schema(tools=("scoop",))
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


# ---- 6. whose tool is it, and which tool does a procedure need (issue #324) --

def test_a_built_bay_says_whose_tool_it_is():
  """One rail, two robots: a bay may hold the other robot's tool, which this
  robot may not take and may not retire. ⚠ The TAG cannot carry it -- a
  built module's tag is `15 + bay` and belongs to the BAY, reused by
  whatever hangs there next -- so the context is the only place ownership
  can be said. `built` is what a lifecycle hung; the rail is shared."""
  from pluggybot.lifecycle import rack_context
  from pluggybot.workshop import validate
  a, b = stub_life(), stub_life(body=StubBody(handle=SECOND), handle=SECOND,
                                robot_name="Rowan")
  b.rack_inventory = a.rack_inventory
  a.peers, b.peers = [b], [a]
  for life, raw, bay in ((a, SCOOP, 0), (b, {**SCOOP, "name": "grabber"}, 1)):
    tool = validate.check(raw)
    life.rack_inventory[tool.body] = built_bay_index(bay)       # as `hang_tool`
    life.built[tool.body] = tool                                # leaves them
  places = {m: "on its bay" for m in a.rack_inventory}
  mine = rack_context(a.rack_inventory, places, a.built_by())["built"]
  theirs = rack_context(b.rack_inventory, places, b.built_by())["built"]
  here = {"where": "on its bay"}
  assert mine["A"] == {"module": "module_scoop", "by": "you", **here}
  assert mine["B"] == {"module": "module_grabber", "by": "Rowan", **here}
  assert theirs["A"] == {"module": "module_scoop", "by": a.robot_name, **here}
  assert theirs["B"] == {"module": "module_grabber", "by": "you", **here}
  assert mine["C"] is theirs["C"] is None


def test_a_procedure_says_which_tool_it_needs():
  """`build.register` puts `requires=module_<name>` on a built tool's axes
  and sensors, so `move("scoop.tilt", ...)` without the scoop on the fork
  fails with a reason that names the module -- but only at the point of
  failure, after an errand has been committed to it. This states it; the
  facts name the tool's axes as a world whose rack holds it would."""
  from dataclasses import replace
  from pluggybot.lifecycle import world_facts
  from pluggybot.procedure.library import Library
  from pluggybot.workshop import build, validate
  names = build.register(validate.check(SCOOP))
  rack = {**world_config(WORLD)["tool_bays"], "module_scoop": built_bay_index(0)}
  facts = world_facts(WORLD, rack=rack)
  lib = Library(replace(facts, axes=facts.axes + tuple(names),
                        sensors=facts.sensors + tuple(names)))
  lib.define("dig", 'def dig():\n  fetch("module_scoop")\n'
                    '  move("scoop.tilt", 1.0)\n  stow()')
  lib.define("look", 'def look():\n  wait(1.0)\n')
  rows = {r["name"]: r for r in lib.as_context()}
  assert rows["dig"]["needs"] == ["module_scoop"]
  assert "needs" not in rows["look"]        # absent, not empty: a learnable slot
  # the walk reaches into loops, branches and read() inside a condition
  lib.define("probe", 'def probe():\n  for i in range(2):\n'
                      '    if read("scoop.tilt") > 0:\n'
                      '      wait(1)\n')
  # ⚠ ...AND WHAT A `fetch` NAMES, or it lies by omission: a procedure that
  # fetches a tool and uses no axis of it names no axis at all, and
  # reported NO needs, which reads as "needs no tool".
  lib.define("hold", 'def hold():\n  fetch("module_claw")\n  wait(1)\n  stow()')
  rows = {r["name"]: r for r in lib.as_context()}
  assert rows["probe"]["needs"] == ["module_scoop"]
  assert rows["hold"]["needs"] == ["module_claw"], rows["hold"]
  # ...and this is WHY it was missed: not one axis, not one sensor
  assert lib.get("hold").references()["axes"] == ()
  assert lib.get("hold").references()["sensors"] == ()
