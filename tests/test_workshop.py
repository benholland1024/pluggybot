"""The workshop's spec and validator (issue #168, slice A): ToolPattern §2
as code, one test per rule, each shown to bite on a spec that is otherwise
fine. `SCOOP` is the reference: a servo on the plate's front, a printed
blade on the servo -- 142 g, 0.007 N·m, and every number the catalog knows.
"""

import copy
import math

import pytest

from pluggybot.power import MODULE_IDLE_W
from pluggybot.rack import catalog
from pluggybot.rack.coupling import (
  LATCH_MOMENT_NM, MODULE_MASS, MODULE_MASS_CEILING, PEG_POWER_W,
)
from pluggybot.workshop import spec, validate
from pluggybot.workshop.spec import Refused, parse, poses

SCOOP = {
  "name": "scoop",
  "parts": [
    {"id": "hinge", "part": "servo_fs90", "pos": [-20, 0, -45],
     "axis": {"verb": "tilt", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
    {"id": "blade", "part": "scaffold_pla_box", "size": [60, 30, 4],
     "pos": [-30, 0, -8], "on": "hinge"},
  ],
}


def scoop(**over) -> dict:
  raw = copy.deepcopy(SCOOP)
  raw.update(over)
  return raw


def with_part(*parts: dict) -> dict:
  raw = scoop()
  raw["parts"].extend(parts)
  return raw


def refused(raw) -> list[str]:
  with pytest.raises(Refused) as info:
    validate.check(raw)
  return info.value.reasons


def test_the_reference_tool_validates():
  tool = validate.check(SCOOP)
  assert tool.body == "module_scoop"
  assert 0.14 < MODULE_MASS + tool.mass < 0.15
  assert validate.worst_moment(tool) < 0.01
  assert [p.axis.verb for p in tool.axes] == ["tilt"]


def test_units_are_converted_once_at_the_door():
  """mm and degrees in, metres and radians out -- a pos of 20 mm is 0.02 m
  and a 90° range is π/2. Shown to fail by the bug this pins: the first
  draft placed a servo 20 012 mm out the front."""
  tool = parse(SCOOP)
  hinge = tool.by_id["hinge"]
  assert hinge.pos == pytest.approx((-0.020, 0.0, -0.045))
  assert hinge.axis.hi == pytest.approx(math.pi / 2)
  assert hinge.axis.speed == pytest.approx(math.radians(600))
  assert tool.by_id["blade"].half == pytest.approx((0.030, 0.015, 0.002))


def test_a_part_on_an_actuator_rides_its_axis():
  """The blade on the hinge: at 90° about +y a point at (-30, 0, -8) from
  the hinge lands at (-8, 0, +30) -- checked by hand, so the validator's
  moment and clearance at the range ends are of the pose the tool will
  actually take."""
  tool = parse(SCOOP)
  stow = poses(tool)["blade"][0]
  up = poses(tool, {"tilt": math.pi / 2})["blade"][0]
  assert stow == pytest.approx((-0.050, 0.0, -0.053))
  assert up == pytest.approx((-0.028, 0.0, -0.015))


# ---- structure -------------------------------------------------------------

def test_a_spec_cannot_carry_a_solver_option():
  """The rule that does not move: no field for a solver option, a contact
  parameter, a mass or a friction exists, and an unknown one is REFUSED,
  not dropped -- dropped, `noslip: 1` would read as a spec that set it."""
  for word in ("noslip", "noslip_iterations", "solref", "solimp", "friction",
               "mass", "priority", "condim", "timestep"):
    assert word not in spec.SPEC_FIELDS | spec.PART_FIELDS | spec.AXIS_FIELDS
  reasons = refused(scoop(noslip=1))
  assert any("unknown field 'noslip'" in r for r in reasons)
  raw = scoop()
  raw["parts"][0]["friction"] = 0.1
  assert any("unknown field 'friction'" in r for r in refused(raw))


def test_every_reason_at_once():
  """Structure first (a spec that does not hang together has no envelope
  to check), and within a stage every reason together."""
  raw = scoop(noslip=1, name="Bad Name")
  raw["parts"][0]["axis"]["stow"] = 100
  reasons = refused(raw)
  assert len(reasons) == 3
  raw = with_part({"id": "sheet", "part": "scaffold_pla_box", "size": [260, 2, 2],
                   "pos": [0, 30, -20]})
  reasons = refused(raw)
  assert {r.split(":")[0] for r in reasons} >= {"bed", "bracket"}


def test_a_name_the_rack_already_has_is_refused():
  assert any("already has a module_pen" in r for r in refused(scoop(name="pen")))


def test_only_catalog_shelf_chosen_parts_with_known_numbers():
  # not a part
  assert any("no catalog part 'unobtainium'" in r for r in refused(
    with_part({"id": "x", "part": "unobtainium", "pos": [0, 0, -40]})))
  # a body-shelf part (the gearmotor) is not something a module is built from
  assert any("body shelf" in r for r in refused(
    with_part({"id": "x", "part": "gearmotor_37d_50", "pos": [0, 0, -40]})))
  # a candidate with no part behind it
  reasons = refused(with_part({"id": "x", "part": "module_servo", "pos": [0, 0, -40],
                               "axis": {"verb": "v", "dir": [0, 1, 0],
                                        "range": [0, 10], "stow": 0}}))
  assert any("candidate, not chosen (no part chosen)" in r for r in reasons)
  # chosen, on the catalog, and the catalog does not weigh it: the rod
  reasons = refused(with_part({"id": "x", "part": "peg_rod_6mm", "pos": [0, 0, -40]}))
  assert any("does not know peg_rod_6mm's mass" in r for r in reasons)


def test_the_servo_the_workshop_leans_on_has_every_number():
  """`servo_fs90` is the one chosen actuator the catalog knows everything
  about; a catalog edit that nulls one of these fails HERE, not in an
  agent's refusal at run time."""
  part = catalog.by_id()["servo_fs90"]
  assert part.status == "chosen" and "catalog" in part.shelves
  assert part.massG and part.dimensionsMm
  for key in ("motion", "angleDeg", "speedDegS", "torqueNm", "powerW"):
    assert part.capabilities.get(key) is not None, key


def test_an_axis_must_fit_its_part():
  raw = scoop()
  raw["parts"][0]["axis"]["range"] = [0, 200]
  assert any("range spans 200°, more than the part's 120°" in r for r in refused(raw))
  raw = scoop()
  raw["parts"][0]["axis"]["stow"] = 100
  assert any("stow 100 is outside" in r for r in refused(raw))
  raw = scoop()
  del raw["parts"][0]["axis"]
  assert any("an actuator needs an axis" in r for r in refused(raw))
  raw = scoop()
  raw["parts"][1]["axis"] = {"verb": "v", "dir": [1, 0, 0], "range": [0, 1], "stow": 0}
  assert any("is a scaffold, not an actuator" in r for r in refused(raw))


def test_the_assembly_graph_is_a_tree_on_the_frame():
  raw = scoop()
  raw["parts"][1]["on"] = "nothing"
  assert any("mounted on 'nothing', which is not a part" in r for r in refused(raw))
  raw = scoop()
  raw["parts"][0]["on"] = "blade"
  assert any("form a cycle" in r for r in refused(raw))
  raw = with_part({"id": "blade", "part": "scaffold_pla_box", "size": [10, 10, 10],
                   "pos": [0, 0, -50]})
  assert any("id used twice" in r for r in refused(raw))
  raw = scoop()
  raw["parts"] = raw["parts"] + [
    {"id": f"b{i}", "part": "scaffold_pla_box", "size": [5, 5, 5], "pos": [0, 0, -50]}
    for i in range(spec.MAX_PARTS)]
  assert any(f"the most a tool may have is {spec.MAX_PARTS}" in r for r in refused(raw))


# ---- the envelope, one rule each -------------------------------------------

def test_mass_class():
  """A 100 × 100 × 15 mm printed block is 186 g of PLA: over the class with
  the plate. Shown to fail by raising MODULE_MASS_CEILING to 0.5."""
  raw = with_part({"id": "block", "part": "scaffold_pla_box", "size": [100, 100, 15],
                   "pos": [0, 0, -50]})
  reasons = refused(raw)
  assert any(r.startswith("mass:") and f"over the {MODULE_MASS_CEILING * 1000:.0f} g" in r
             for r in reasons)


def test_moment_about_the_peg_and_reach_costs_more_than_mass():
  """FINDING, from writing the rule: inside the mass class and the wall
  clearance the moment rule cannot bite FORWARD -- 250 g at 90 mm is
  0.22 N·m, half the latch's budget -- so what it guards is reach toward
  the robot (+x, past the fork) and, later, a payload. A 99 g plate 480 mm
  toward the robot is inside the class and over the moment; the reason
  says which way to move it."""
  near = validate.check(SCOOP)
  assert validate.worst_moment(near) < 0.05 * LATCH_MOMENT_NM
  raw = with_part({"id": "plate", "part": "scaffold_pla_box", "size": [100, 100, 8],
                   "pos": [480, 0, -60]})
  reasons = refused(raw)
  assert any(r.startswith("moment:") and "reach costs more than mass" in r
             for r in reasons)
  assert not any(r.startswith("mass:") for r in reasons)


def test_moment_is_judged_at_the_axis_ends():
  """A blade hanging straight DOWN from the hinge has the hinge's own reach
  at stow (20 mm) and swings out to 55 mm at ±60°. The worst moment is
  taken over the ends, not read at the stow pose."""
  raw = scoop()
  raw["parts"][0]["axis"]["range"] = [-60, 60]
  raw["parts"][1] = {"id": "blade", "part": "scaffold_pla_box", "size": [4, 30, 60],
                     "pos": [0, 0, -40], "on": "hinge"}
  tool = validate.check(raw)
  at_stow = sum(p.mass * validate.G * abs(float(poses(tool)[p.id][0][0]))
                for p in tool.parts)
  assert validate.worst_moment(tool) > at_stow


def test_the_fork_volume_is_the_forks():
  """A part where the prongs hold the peg: |y| 58 mm, at the peg line, on
  the robot's side."""
  raw = with_part({"id": "tab", "part": "scaffold_pla_box", "size": [20, 10, 10],
                   "pos": [20, 58, 15]})
  assert any(r.startswith("fork:") and "'tab'" in r for r in refused(raw))
  # ...and the mirror side
  raw = with_part({"id": "tab", "part": "scaffold_pla_box", "size": [20, 10, 10],
                   "pos": [20, -58, 15]})
  assert any(r.startswith("fork:") for r in refused(raw))


def test_the_tray_volume_is_the_racks():
  raw = with_part({"id": "tab", "part": "scaffold_pla_box", "size": [10, 10, 10],
                   "pos": [0, 40, 22]})
  assert any(r.startswith("trays:") and "'tab'" in r for r in refused(raw))


def test_the_bracket_band_outboard_of_the_plate():
  """The pen's lesson: a part at |y| > 20 mm in z -30..-9 arrives under the
  tray brackets on a set-down. The same part within the plate's width, or
  below the band, is fine."""
  raw = with_part({"id": "wing", "part": "scaffold_pla_box", "size": [10, 30, 10],
                   "pos": [-25, 30, -20]})
  reasons = refused(raw)
  assert any(r.startswith("bracket:") and "'wing'" in r for r in reasons)
  ok = with_part({"id": "wing", "part": "scaffold_pla_box", "size": [10, 30, 10],
                  "pos": [-25, 0, -20]})
  validate.check(ok)
  low = with_part({"id": "wing", "part": "scaffold_pla_box", "size": [10, 30, 10],
                   "pos": [-25, 30, -60]})
  validate.check(low)


def test_the_wall_is_90_mm_out_the_front():
  raw = with_part({"id": "probe", "part": "scaffold_pla_box", "size": [60, 10, 10],
                   "pos": [-70, 0, -60]})
  assert any(r.startswith("wall:") and "'probe'" in r for r in refused(raw))
  ok = with_part({"id": "probe", "part": "scaffold_pla_box", "size": [60, 10, 10],
                  "pos": [-55, 0, -60]})
  validate.check(ok)


def test_the_power_budget_through_the_peg():
  """Three FS90s at stall plus the ESP32 is 15 W against the peg's 12 (two
  were over the 6 W it had until 2026-09-15; the budget is a design
  decision argued at the constant). And a sensor whose draw is unknown
  (the Pi camera: the maker does not publish it) cannot be budgeted, so a
  tool cannot carry one -- refused with the gap named, not waved through
  at 0 W."""
  raw = with_part({"id": "hinge2", "part": "servo_fs90", "pos": [-20, 0, -70],
                   "axis": {"verb": "tilt2", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
                  {"id": "hinge3", "part": "servo_fs90", "pos": [-20, 20, -70],
                   "axis": {"verb": "tilt3", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}})
  reasons = refused(raw)
  draw = MODULE_IDLE_W + 3 * 4.8
  assert draw > PEG_POWER_W > MODULE_IDLE_W + 2 * 4.8, "the pin straddles the budget"
  assert any(r.startswith("power:") and f"{draw:.1f} W" in r
             and f"over its {PEG_POWER_W:g} W" in r for r in reasons)
  raw = with_part({"id": "eye", "part": "pi_camera_3", "pos": [-25, 0, -70]})
  reasons = refused(raw)
  assert any("does not know what pi_camera_3 draws" in r for r in reasons)


def test_the_print_bed():
  raw = with_part({"id": "sheet", "part": "scaffold_pla_box", "size": [260, 2, 2],
                   "pos": [0, 0, -50]})
  reasons = refused(raw)
  assert any(r.startswith("bed:") and "'sheet'" in r for r in reasons)
  # 230 mm fits the 250 mm z standing up: orientation is the printer's,
  # so a sheet longer than the bed's x is not refused for that
  raw = with_part({"id": "sheet", "part": "scaffold_pla_box", "size": [230, 2, 2],
                   "pos": [0, 0, -50]})
  try:
    validate.check(raw)
  except Refused as e:
    assert not any(r.startswith("bed:") for r in e.reasons)


def test_a_scaffold_is_priced_by_its_volume_at_pla_density():
  tool = parse(SCOOP)
  blade = tool.by_id["blade"]
  density = catalog.by_id()["scaffold_pla_box"].capabilities["densityKgM3"]
  assert blade.mass == pytest.approx(density * 0.060 * 0.030 * 0.004)
