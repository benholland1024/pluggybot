"""The parts catalog as data (issue #185): `protocol/parts.json`, the
humanoid's bill (`humanoid/bill.py`, #379) over the catalog's machinery and
its bought parts' facts (`humanoid/catalog.py`).

What these hold down:

  1. THE FIXTURE CANNOT DRIFT FROM THE SIM. Every `feeds` value is read off
     the compiled model or the live constant when the fixture is built, so
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
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import mujoco
import pytest

from pluggybot.humanoid import bill, catalog

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "protocol" / "parts.json"
MODULES = (ROOT / "src" / "pluggybot" / "humanoid" / "catalog.py",
           ROOT / "src" / "pluggybot" / "humanoid" / "bill.py")

#: The body's parts Parts.md chose (its actuators, gripper, cells and
#: sensors), by the number you would order with. The doc gains a pointer to
#: the fixture and stops being the only copy of these.
PARTS_MD_CHOSEN = ("RS04", "RS03", "RS02", "RS00", "RS05", "Dex1-1", "INR21700-P45B",
                   "Mid-360S", "D435", "Camera Module 3", "EV_ICM-42688-P")


@pytest.fixture(scope="module")
def fixture() -> dict:
  return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def parts(fixture) -> dict:
  return {p["id"]: p for p in fixture["parts"]}


def test_the_fixture_is_not_stale(fixture):
  """Regenerating must reproduce the committed file exactly -- claim 1.

  Shown to fail by changing any literal a feed reads (a joint's armature
  in `models/humanoid.xml`, `activity.station.BREAK_N`) without
  regenerating.
  """
  assert fixture == bill.build(), "stale: uv run python -m pluggybot.humanoid.bill"


def test_it_is_the_humanoids_build(fixture):
  """The parts page shows the served robot's build (#185): the humanoid's,
  every part read against its model, since the quadruped's went (#559)."""
  assert fixture["worlds"] == {"humanoid": "models/humanoid.xml"}
  assert {p["robot"] for p in fixture["parts"]} == {"humanoid"}


def test_every_entry_is_honest(fixture):
  for p in fixture["parts"]:
    assert catalog.validate(p, worlds=bill.WORLDS) == [], p["id"]
  assert catalog.mismatches(fixture) == []
  ids = [p["id"] for p in fixture["parts"]]
  assert len(ids) == len(set(ids))


def test_the_validator_rejects_what_the_rules_forbid(parts):
  """Claim 3, each rule shown to bite on an entry that is honest today."""
  good = parts["rs04"]

  def v(entry):
    return catalog.validate(entry, worlds=bill.WORLDS)
  assert v(good) == []
  missing = {k: x for k, x in good.items() if k != "massG"}
  assert v(missing) == ["missing massG"]
  unexplained = {**good, "priceEur": None}
  assert "priceEur is null with no why" in v(unexplained)
  stale = {**good, "why": {"priceEur": "quote-only"}}
  assert any("stale excuse" in r for r in v(stale))
  wrong_key = {**good, "why": {"name": "unknown"}}
  assert any("cannot be null" in r for r in v(wrong_key))
  not_a_url = {**good, "source": "OpenELAB"}
  assert "source must be a URL" in v(not_a_url)
  unknown_kind = {**good, "kind": "widget"}
  assert any("kind" in r for r in v(unknown_kind))
  no_shelf = {**good, "shelves": []}
  assert any("shelves" in r for r in v(no_shelf))
  elsewhere = {**good, "robot": "quadruped"}
  assert any("robot" in r for r in v(elsewhere))
  # A guess is a number where the doc has none; the rule cannot see a
  # guess, but it can see the excuse for one that was later filled in.
  filled_in = {**parts["amass_xt30_22_pair"], "massG": 5}
  assert any("stale excuse" in r for r in v(filled_in))


def test_a_moved_constant_is_caught(monkeypatch):
  """Claim 2: `expect` pins the datasheet to the sim, shown to fail.

  The RoPD plug breaks away at the sheets' 20 N, the station's catch's
  `BREAK_N`, read through an attribute path; move it and the mismatch is
  named. Cheap: `build()` is a spec parse, not a compile.
  """
  from pluggybot.activity import station
  monkeypatch.setattr(station, "BREAK_N", 25.0)
  assert catalog.mismatches(bill.build()) == [
    "ropd_panel_plug: activity.station.BREAK_N is 25, the part says 20 N"]


def test_drivers_the_design_keeps_alike_that_come_apart_are_refused():
  """`same()` reads the elements one part drives as one number and refuses
  if the design has quietly come apart -- rather than reporting the first."""
  from pluggybot.humanoid import model as hm
  spec = mujoco.MjSpec.from_file(bill.WORLDS[bill.ROBOT])
  reader = catalog.same(*(catalog.actuator(f"{s}_gripper", "forcerange") for s in hm.SIDES))
  assert reader(spec) == 240
  next(a for a in spec.actuators if a.name == f"{hm.SIDES[-1]}_gripper").forcerange = [-200, 200]
  with pytest.raises(ValueError, match="asymmetric"):
    reader(spec)


def test_no_feed_is_typed():
  """Claim 4, structurally: a `Feed(...)` whose second argument is a
  literal would be a hand-copied number the stale check could not see."""
  for module in MODULES:
    tree = ast.parse(module.read_text())
    typed = [
      node.lineno for node in ast.walk(tree)
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
      and node.func.id == "Feed" and len(node.args) > 1
      and isinstance(node.args[1], ast.Constant)
    ]
    assert typed == [], module.name


def test_a_bought_parts_mass_is_one_fact():
  """The catalog keeps a bought part's facts once: the body's model weighs
  the head's sensors and the torso's electronics off the catalog's masses
  (`humanoid.parts._catalog_kg`), never a second typed copy -- which an
  equal value cannot show, so the source is read for the call."""
  from pluggybot.humanoid import parts as hp
  kg = {pid: catalog.by_id()[pid].massG / 1000 for pid in
        ("realsense_d435", "pi_camera_3", "livox_mid360s", "waveshare_can_hat_plus")}
  assert hp.D435_KG == kg["realsense_d435"]
  assert hp.CAMERA_KG == kg["pi_camera_3"]
  assert hp.LIDAR.kg == kg["livox_mid360s"]
  assert hp.ELECTRONICS["waveshare_can_hat_plus"] == kg["waveshare_can_hat_plus"]
  tree = ast.parse((ROOT / "src" / "pluggybot" / "humanoid" / "parts.py").read_text())
  read = {node.args[0].value for node in ast.walk(tree)
          if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_catalog_kg"
          and node.args and isinstance(node.args[0], ast.Constant)}
  assert set(kg) <= read, set(kg) - read


def test_parts_md_chosen_list_is_in_the_catalog_and_the_doc_points_here(parts):
  by_number = {p["partNumber"]: p for p in parts.values()
               if p["partNumber"] and "build" in p["shelves"]}
  for number in PARTS_MD_CHOSEN:
    p = by_number.get(number)
    assert p, f"Parts.md chose {number} and the bill has no such part"
    assert p["status"] == "chosen", number
    assert p["feeds"], f"{number} is chosen and feeds nothing"
  doc = (ROOT / "docs" / "Parts.md").read_text()
  assert "protocol/parts.json" in doc and "pluggybot.humanoid.bill" in doc


def test_every_used_by_names_a_body_in_its_robots_world(fixture):
  """`usedBy` is checked against the compiled model of the robot the part
  is for, so a body part that is redrawn takes its parts' claims with it.
  ⚠ The spec is HELD while its bodies are read: iterated off a temporary,
  it was freed part way and its names read as garbage (a UnicodeDecodeError,
  or rarely a wrong name)."""
  specs = {robot: mujoco.MjSpec.from_file(path) for robot, path in bill.WORLDS.items()}
  bodies = {robot: {b.name for b in spec.bodies} for robot, spec in specs.items()}
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
  assert bill.with_bill(doc, fixture) == doc, \
      "stale bill: uv run python -m pluggybot.humanoid.bill"


def test_every_line_of_the_bill_is_honest(fixture):
  parts = {p["id"]: p for p in fixture["parts"]}
  assert fixture["build"]["lines"], "the bill is empty"
  for ln in fixture["build"]["lines"]:
    assert catalog.validate_line(ln, parts, groups=bill.GROUPS) == [], ln["part"]


def test_the_line_validator_rejects_what_the_rules_forbid(fixture):
  """Claim 5, each rule shown to bite on a line that is honest today: an
  allowance is never beside a price, a part with no price is never
  silent, a lead time is the seller's words or a stated unknown."""
  parts = {p["id"]: p for p in fixture["parts"]}
  lines = fixture["build"]["lines"]
  priced = next(ln for ln in lines if not ln["allowance"] and ln["leadTime"])
  allowed = next(ln for ln in lines if ln["allowance"])

  def v(line, ps=parts):
    return catalog.validate_line(line, ps, groups=bill.GROUPS)
  assert v(priced) == []
  assert v(allowed) == []
  assert "a priced part carries its price, not an allowance" in \
      v({**priced, "basis": "about that much"})
  assert "a part with no price needs an allowance and its basis" in \
      v({**allowed, "lineEur": None})
  assert "a part with no price needs an allowance and its basis" in \
      v({**allowed, "basis": ""})
  assert "a lead time is the seller's words, or null with a why" in \
      v({**priced, "leadTime": None})
  assert any("group" in r for r in v({**priced, "group": "misc"}))
  off_shelf = {**parts[priced["part"]], "id": "off_shelf", "shelves": []}
  assert "off_shelf is not on the build shelf" in \
      v({**priced, "part": "off_shelf"}, {**parts, "off_shelf": off_shelf})


def test_the_bill_is_summed_from_its_lines_and_claims_no_budget(fixture):
  """The totals are summed, never typed; and the bill names no budget --
  #379's $20,000 was set for the quadruped's build -- so it says none
  (`null`) rather than carrying that one."""
  b = fixture["build"]
  t = b["totals"]
  assert t["sourcedEur"] == pytest.approx(
    sum(ln["lineEur"] for ln in b["lines"] if not ln["allowance"]))
  assert t["allowanceEur"] == pytest.approx(
    sum(ln["lineEur"] for ln in b["lines"] if ln["allowance"]))
  assert t["totalEur"] == pytest.approx(
    (t["sourcedEur"] + t["allowanceEur"]) * (1 + b["contingency"]), abs=0.02)
  assert t["byGroup"] == {g: pytest.approx(sum(ln["lineEur"] or 0 for ln in b["lines"]
                                               if ln["group"] == g)) for g in b["groups"]}
  assert b["budgetUsd"] is None and t["budgetEur"] is None


def test_the_bill_says_its_days_as_days(fixture):
  """The site prints `checked` and `rateDate` as calendar days, and a page
  that cannot read one renders nothing (the humanoid's prose did): a bill
  read on several days carries the oldest, which the page says as "on or
  after", and only Parts.md says them all."""
  b = fixture["build"]
  for day in (b["checked"], b["rateDate"]):
    assert date.fromisoformat(day).isoformat() == day, day
  assert b["checked"] == min(bill.READ_ON)
  with pytest.raises(ValueError, match="calendar day"):
    catalog.bom({}, lines=(), groups=(), checked=bill.CHECKED)


def test_the_bill_buys_what_the_design_uses(fixture):
  """A build part's `quantity` is what the design uses (on the robot, in
  the station, in the shop); a priced part's lines must buy at least that,
  or a design count raised without a purchase goes unnoticed. A part made
  to a drawing is bought by one ALLOWANCE line, a sum for all of it (its
  basis says how many), and a part printed in the shop is not bought at
  all: its material is the filament's line."""
  bought = defaultdict(int)
  for ln in fixture["build"]["lines"]:
    bought[ln["part"]] += ln["quantity"]
  for p in fixture["parts"]:
    assert p["quantity"] > 0, p["id"]
    if p["why"].get("partNumber") == catalog.PRINTED:
      continue
    if p["priceEur"] is None:
      assert bought[p["id"]] >= 1, p["id"]
    else:
      assert bought[p["id"]] >= p["quantity"], (p["id"], bought[p["id"]])


def test_an_allowance_beside_a_price_is_refused_where_it_is_written(monkeypatch):
  """The fixture cannot show an allowance written on a priced part (the
  price wins), so the bill refuses it as it is built."""
  priced = {p.id for p in bill.PARTS if p.priceEur is not None}
  line = next(ln for ln in bill.LINES if ln.part in priced)
  monkeypatch.setattr(bill, "LINES", (
    catalog.Line(line.part, 1, line.group, "now", allowanceEur=9.0, basis="about that"),))
  with pytest.raises(ValueError, match="carries its price, not an allowance"):
    bill.build()


def test_one_part_number_is_one_set_of_facts(fixture):
  """A part number is one price and one mass, however many entries carry
  it."""
  by_number = defaultdict(list)
  for p in fixture["parts"]:
    if p["partNumber"]:
      by_number[p["partNumber"]].append(p)
  for number, ps in by_number.items():
    for key in ("priceEur", "massG"):
      assert len({json.dumps(p[key]) for p in ps}) == 1, (number, key)


def test_nothing_in_the_economy_or_the_mind_reads_the_catalog():
  """The scoring path never reads the catalog or the bill (issue #185): a
  job is graded by predicates on the world, never by what the robot is made
  of, and nothing awards itself points for a part list. A grep would pass
  on a comment; the syntax tree does not."""
  for path in [*(ROOT / "src/pluggybot/economy").glob("*.py"),
               *(ROOT / "src/pluggybot/mind").glob("*.py")]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      names = []
      if isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module] + [a.name for a in node.names]
      elif isinstance(node, ast.Import):
        names = [a.name for a in node.names]
      assert not any("catalog" in n or n.endswith(".bill") or n == "bill" for n in names), path
