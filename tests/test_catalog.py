"""The parts catalog as data (issue #185): `protocol/parts.json`.

What these hold down:

  1. THE FIXTURE CANNOT DRIFT FROM THE SIM. Every `feeds` value is read off
     the compiled world or the live constant when the fixture is built, so
     the committed file is stale the moment a literal moves -- the website's
     parts page shows what the sim uses, or the suite says otherwise.
  2. THE PART AND THE SIM ARE ONE FACT WHERE THEY CLAIM TO BE. A feed with
     `expect` says the datasheet's number IS the sim's; the test moves a
     constant and shows the check catches it.
  3. A NUMBER THE DOC DOES NOT KNOW IS NULL WITH A REASON, NEVER A GUESS,
     and a reason for a number that is known is a stale excuse.
  4. NO FEED IS TYPED: the syntax tree is walked for a `Feed(...)` whose
     value is a literal.
  5. THE BILL IS THE DATA (issue #379): Parts.md's bill of materials is
     rendered from `LINES`, every line is a price off its part or an
     allowance with its basis, and the total is summed, never typed.
"""

import ast
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import pytest

from pluggybot.rack import catalog

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "protocol" / "parts.json"
MODULE = ROOT / "src" / "pluggybot" / "rack" / "catalog.py"

#: The body's parts Parts.md chose (the quadruped, its sensors, its dock and
#: arm), by the number you would order with. The doc gains a pointer to the
#: fixture and stops being the only copy of these.
PARTS_MD_CHOSEN = ("SW-GIM8108-8-S68", "INR21700-P45B", "RPLIDAR C1", "D435",
                   "Camera Module 3", "EV_ICM-42688-P", "0858-0-15-20-82-14-11-0",
                   "CFT-WF-20-18-1")


@pytest.fixture(scope="module")
def fixture() -> dict:
  return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def parts(fixture) -> dict:
  return {p["id"]: p for p in fixture["parts"]}


def test_the_fixture_is_not_stale(fixture):
  """Regenerating must reproduce the committed file exactly -- claim 1.

  Shown to fail by changing any literal a feed reads (a joint's armature
  in `models/quadruped.xml`, `perception.depth.MAX_Z`) without regenerating.
  """
  assert fixture == catalog.build(), \
      "stale: uv run python -m pluggybot.rack.catalog"


def test_every_entry_is_honest(fixture):
  for p in fixture["parts"]:
    assert catalog.validate(p) == [], p["id"]
  assert catalog.mismatches(fixture) == []
  ids = [p["id"] for p in fixture["parts"]]
  assert len(ids) == len(set(ids))


def test_the_validator_rejects_what_the_rules_forbid(parts):
  """Claim 3, each rule shown to bite on an entry that is honest today."""
  good = parts["gim8108_8"]
  assert catalog.validate(good) == []
  missing = {k: v for k, v in good.items() if k != "massG"}
  assert catalog.validate(missing) == ["missing massG"]
  unexplained = {**good, "priceEur": None}
  assert "priceEur is null with no why" in catalog.validate(unexplained)
  stale = {**good, "why": {"priceEur": "quote-only"}}
  assert any("stale excuse" in r for r in catalog.validate(stale))
  wrong_key = {**good, "why": {"name": "unknown"}}
  assert any("cannot be null" in r for r in catalog.validate(wrong_key))
  not_a_url = {**good, "source": "Eckstein"}
  assert "source must be a URL" in catalog.validate(not_a_url)
  unknown_kind = {**good, "kind": "widget"}
  assert any("kind" in r for r in catalog.validate(unknown_kind))
  no_shelf = {**good, "shelves": []}
  assert any("shelves" in r for r in catalog.validate(no_shelf))
  # A guess is a number where the doc has none; the rule cannot see a
  # guess, but it can see the excuse for one that was later filled in.
  filled_in = {**parts["amass_xt30_22_pair"], "massG": 5}
  assert any("stale excuse" in r for r in catalog.validate(filled_in))


def test_a_moved_constant_is_caught(monkeypatch):
  """Claim 2: `expect` pins the datasheet to the sim, shown to fail.

  The GIM8108-8's 22 N·m peak is read off the motor the legs are built
  from, through an attribute path; move it and the mismatch is named.
  Cheap: `build()` is a spec parse, not a compile.
  """
  import dataclasses
  from pluggybot.legs import actuator
  monkeypatch.setattr(actuator, "GIM8108_8",
                      dataclasses.replace(actuator.GIM8108_8, peak_torque=18.0))
  assert catalog.mismatches(catalog.build()) == [
    "gim8108_8: legs.actuator.GIM8108_8.peak_torque is 18, the part says 22 N·m"]


def test_twelve_drivers_that_come_apart_are_refused():
  """`legs()` reads the twelve leg drivers as one number and refuses if the
  design has quietly come apart -- rather than reporting the first."""
  from pluggybot.legs.model import JOINT_NAMES
  spec = mujoco.MjSpec.from_file(catalog.WORLDS["quadruped"])
  reader = catalog.legs(catalog.actuator, "forcerange")
  assert reader(spec) == 22
  next(a for a in spec.actuators if a.name == JOINT_NAMES[-1]).forcerange = [-18, 18]
  with pytest.raises(ValueError, match="asymmetric"):
    reader(spec)


def test_no_feed_is_typed():
  """Claim 4, structurally: a `Feed(...)` whose second argument is a
  literal would be a hand-copied number the stale check could not see."""
  tree = ast.parse(MODULE.read_text())
  typed = [
    node.lineno for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    and node.func.id == "Feed" and len(node.args) > 1
    and isinstance(node.args[1], ast.Constant)
  ]
  assert typed == []


def test_parts_md_chosen_list_is_in_the_catalog_and_the_doc_points_here(parts):
  by_number = {p["partNumber"]: p for p in parts.values()
               if p["partNumber"] and "build" in p["shelves"]}
  for number in PARTS_MD_CHOSEN:
    p = by_number.get(number)
    assert p, f"Parts.md chose {number} and the bill has no such part"
    assert p["status"] == "chosen", number
    assert p["feeds"], f"{number} is chosen and feeds nothing"
  doc = (ROOT / "docs" / "Parts.md").read_text()
  assert "protocol/parts.json" in doc and "pluggybot.rack.catalog" in doc


def test_the_scaffold_primitive_has_a_density_and_a_print_bed(parts):
  scaffolds = [p for p in parts.values() if p["kind"] == "scaffold"]
  assert len(scaffolds) == 1, "ONE scaffold primitive"
  cap = scaffolds[0]["capabilities"]
  assert cap["densityKgM3"] > 0
  assert set(cap["printBedMm"]) == {"x", "y", "z"}
  assert scaffolds[0]["shelves"] == ["catalog"]


def test_every_used_by_names_a_body_in_its_robots_world(fixture):
  """`usedBy` is checked against the compiled world of the robot the part
  is for, so a module that is retired takes its parts' claims with it."""
  bodies = {robot: {b.name for b in mujoco.MjSpec.from_file(path).bodies}
            for robot, path in catalog.WORLDS.items()}
  for p in fixture["parts"]:
    for user in p["usedBy"]:
      assert user in bodies[p["robot"]], \
          f"{p['id']} is used by {user!r}, not in the {p['robot']}'s world"


# ---- the build's bill of materials (#379) ------------------------------------

def test_the_bill_in_parts_md_is_the_data(fixture):
  """Parts.md's bill is rendered from `LINES` between two markers: a price
  edited in the doc, or a line added to the data without re-rendering,
  fails here."""
  doc = (ROOT / "docs" / "Parts.md").read_text()
  assert catalog.with_bill(doc, fixture) == doc, \
      "stale bill: uv run python -m pluggybot.rack.catalog"


def test_every_line_of_the_bill_is_honest(fixture):
  parts = {p["id"]: p for p in fixture["parts"]}
  assert fixture["build"]["lines"], "the bill is empty"
  for ln in fixture["build"]["lines"]:
    assert catalog.validate_line(ln, parts) == [], ln["part"]


def test_the_line_validator_rejects_what_the_rules_forbid(fixture):
  """Claim 5, each rule shown to bite on a line that is honest today: an
  allowance is never beside a price, a part with no price is never
  silent, a lead time is the seller's words or a stated unknown."""
  parts = {p["id"]: p for p in fixture["parts"]}
  lines = fixture["build"]["lines"]
  priced = next(ln for ln in lines if not ln["allowance"])
  allowed = next(ln for ln in lines if ln["allowance"])
  assert catalog.validate_line(priced, parts) == []
  assert catalog.validate_line(allowed, parts) == []
  v = catalog.validate_line
  assert "a priced part carries its price, not an allowance" in \
      v({**priced, "basis": "about that much"}, parts)
  assert "a part with no price needs an allowance and its basis" in \
      v({**allowed, "lineEur": None}, parts)
  assert "a part with no price needs an allowance and its basis" in \
      v({**allowed, "basis": ""}, parts)
  assert "a lead time is the seller's words, or null with a why" in \
      v({**priced, "leadTime": None}, parts)
  assert any("group" in r for r in v({**priced, "group": "misc"}, parts))
  assert "servo_fs90 is not on the build shelf" in \
      v({**priced, "part": "servo_fs90"}, parts)


def test_the_bill_fits_its_budget_and_is_summed_from_its_lines(fixture):
  b = fixture["build"]
  t = b["totals"]
  assert t["sourcedEur"] == pytest.approx(
    sum(ln["lineEur"] for ln in b["lines"] if not ln["allowance"]))
  assert t["allowanceEur"] == pytest.approx(
    sum(ln["lineEur"] for ln in b["lines"] if ln["allowance"]))
  assert t["totalEur"] == pytest.approx(
    (t["sourcedEur"] + t["allowanceEur"]) * (1 + b["contingency"]), abs=0.02)
  assert t["totalUsd"] <= b["budgetUsd"]


def test_the_bill_buys_what_the_design_uses(fixture):
  """A build part's `quantity` is what the design uses (on the robot, the
  rack, the dock, the rigs); the bill's lines must buy at least that, or a
  design count raised without a purchase goes unnoticed. The workshop
  catalog's parts keep their own meaning of `quantity`."""
  bought = defaultdict(int)
  for ln in fixture["build"]["lines"]:
    bought[ln["part"]] += ln["quantity"]
  for p in fixture["parts"]:
    if p["shelves"] == ["build"]:
      assert p["quantity"] > 0, p["id"]
      assert bought[p["id"]] >= p["quantity"], (p["id"], bought[p["id"]])


def test_an_allowance_beside_a_price_is_refused_where_it_is_written(monkeypatch):
  """The fixture cannot show an allowance written on a priced part (the
  price wins), so the bill refuses it as it is built."""
  line = next(ln for ln in catalog.LINES if ln.allowanceEur is None)
  monkeypatch.setattr(catalog, "LINES", (
    catalog.Line(line.part, 1, line.group, "now", allowanceEur=9.0,
                 basis="about that"),))
  with pytest.raises(ValueError, match="carries its price, not an allowance"):
    catalog.build()


def test_one_part_number_is_one_set_of_facts(fixture):
  """A part with two uses is an entry per use -- the Omron switch in a bay
  (`bay_switch`) and on a tool (`bumper_switch`) -- and one price and one
  mass between them."""
  by_number = defaultdict(list)
  for p in fixture["parts"]:
    if p["partNumber"]:
      by_number[p["partNumber"]].append(p)
  for number, ps in by_number.items():
    for key in ("priceEur", "massG"):
      assert len({json.dumps(p[key]) for p in ps}) == 1, (number, key)


def test_the_generator_reads_its_module_mass_off_the_named_constants(parts):
  """The refactor half of the issue's constraint: `module_xml` and the spike
  scene are emitted at `MODULE_MASS`, whose peg share is `PEG_MASS`, and
  the fixture reads both -- one number, named, rather than a `0.12` default
  and a `- 0.02` in two f-strings."""
  from pluggybot.rack import coupling
  import inspect
  assert inspect.signature(coupling.module_xml).parameters["mass"].default \
      == coupling.MODULE_MASS
  assert inspect.signature(coupling.scene_xml).parameters["tool_mass"].default \
      == coupling.MODULE_MASS
  frame = parts["module_frame"]
  assert {f["constant"]: f["value"] for f in frame["feeds"]}[
    "rack.coupling.MODULE_MASS"] == coupling.MODULE_MASS
  assert 'mass="0.100"' in coupling.module_xml("m", 0, 0, 0.3, "0 0 0 1")


def test_nothing_in_the_economy_reads_the_catalog():
  """The scoring path never reads the catalog (issue #185): a tool's
  usefulness is graded by predicates on the world, never by what it is
  made of, and nothing awards itself points for a part list. (#168 slice D
  decided how a MIND sees it -- through `overseer.workshop_rule()`, on the
  `autonomous` arm alone -- so `mind/` left this fence then.) A grep would
  pass on a comment; the syntax tree does not."""
  for path in (ROOT / "src/pluggybot/economy").glob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      names = []
      if isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module] + [a.name for a in node.names]
      elif isinstance(node, ast.Import):
        names = [a.name for a in node.names]
      assert not any("catalog" in n for n in names), path


def test_the_fixture_says_what_the_workshop_may_build_from(fixture):
  """`workshop.usable` per part is the validator's own predicate (issue
  #168), so the parts page marks exactly what a spec may name. Today: the
  micro servo and the scaffold, and every other part says why not."""
  from pluggybot.workshop.spec import unbuildable
  by_id = catalog.by_id()
  usable = sorted(p["id"] for p in fixture["parts"] if p["workshop"]["usable"])
  assert usable == sorted(p.id for p in catalog.PARTS if unbuildable(p) is None)
  # Widened on purpose by issue #199: a slide, a switch, an eye, a second
  # servo -- each sourced with a datasheet, none by guessing a number.
  assert usable == ["bumper_switch", "esp32_cam", "scaffold_pla_box",
                    "servo_fs90", "servo_fs90mg", "slide_l12_100"]
  for p in fixture["parts"]:
    if not p["workshop"]["usable"]:
      assert p["workshop"]["why"] == unbuildable(by_id[p["id"]])
