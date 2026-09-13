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
"""

import ast
import json
from pathlib import Path

import mujoco
import pytest

from pluggybot import control
from pluggybot.rack import catalog

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "protocol" / "parts.json"
MODULE = ROOT / "src" / "pluggybot" / "rack" / "catalog.py"

#: Parts.md's ✅ CHOSEN list, by the number you would order with. The doc
#: gains a pointer to the fixture and stops being the only copy of these.
PARTS_MD_CHOSEN = ("4753", "1435-1439 (by colour)", "1999", "955",
                   "RPLIDAR C1", "Camera Module 3", "DLE-LA-0001")


@pytest.fixture(scope="module")
def fixture() -> dict:
  return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def parts(fixture) -> dict:
  return {p["id"]: p for p in fixture["parts"]}


def test_the_fixture_is_not_stale(fixture):
  """Regenerating must reproduce the committed file exactly -- claim 1.

  Shown to fail by changing any literal a feed reads (a mass in
  `pluggybot_fork.xml`, `control.WHEEL_RADIUS`) without regenerating.
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
  good = parts["gearmotor_37d_50"]
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
  filled_in = {**parts["wheel_90x10"], "massG": 60}
  assert any("stale excuse" in r for r in catalog.validate(filled_in))


def test_a_moved_constant_is_caught(monkeypatch):
  """Claim 2: `expect` pins the datasheet to the sim, shown to fail.

  The wheel radius is `control.WHEEL_RADIUS` and the 90 mm wheel says it is
  0.045; move the constant and the mismatch is named. Cheap: `build()` is a
  spec parse, not a compile.
  """
  assert catalog.mismatches(catalog.build()) == []
  monkeypatch.setattr(control, "WHEEL_RADIUS", 0.05)
  off = catalog.mismatches(catalog.build())
  assert off == ["wheel_90x10: control.WHEEL_RADIUS is 0.05, the part says "
                 "0.045 m"]


def test_a_symmetric_pair_that_comes_apart_is_refused():
  """`same()` reads the two wheels as one number and refuses if the design
  has quietly become asymmetric -- rather than reporting the left."""
  spec = mujoco.MjSpec.from_file(catalog.WORLD)
  next(g for g in spec.geoms if g.name == "right_tire").size[0] = 0.05
  reader = catalog.same(catalog.geom("left_tire", "size", 0),
                        catalog.geom("right_tire", "size", 0))
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
  by_number = {p["partNumber"]: p for p in parts.values() if p["partNumber"]}
  for number in PARTS_MD_CHOSEN:
    p = by_number.get(number)
    assert p, f"Parts.md chose {number} and the catalog has no such part"
    assert p["status"] == "chosen" and "body" in p["shelves"], number
    # The mounting hub is an adapter: it sets nothing the sim models.
    assert p["feeds"] or number == "1999", f"{number} is chosen and feeds nothing"
  doc = (ROOT / "docs" / "Parts.md").read_text()
  assert "protocol/parts.json" in doc and "pluggybot.rack.catalog" in doc


def test_the_scaffold_primitive_has_a_density_and_a_print_bed(parts):
  scaffolds = [p for p in parts.values() if p["kind"] == "scaffold"]
  assert len(scaffolds) == 1, "ONE scaffold primitive"
  cap = scaffolds[0]["capabilities"]
  assert cap["densityKgM3"] > 0
  assert set(cap["printBedMm"]) == {"x", "y", "z"}
  assert scaffolds[0]["shelves"] == ["catalog"]


def test_every_used_by_names_a_body_in_the_world(fixture):
  """`usedBy` is checked against the compiled world, so a module that is
  retired takes its parts' claims with it."""
  spec = mujoco.MjSpec.from_file(catalog.WORLD)
  bodies = {b.name for b in spec.bodies}
  for p in fixture["parts"]:
    for user in p["usedBy"]:
      assert user in bodies, f"{p['id']} is used by {user!r}, not in the world"


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
  assert usable == ["scaffold_pla_box", "servo_fs90"]
  for p in fixture["parts"]:
    if not p["workshop"]["usable"]:
      assert p["workshop"]["why"] == unbuildable(by_id[p["id"]])
