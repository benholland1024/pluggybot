"""The workshop's spec and validator (issue #168, slice A; on legs, #407):
the arm's envelope as code, one test per rule, each shown to bite on a spec
that is otherwise fine. `SCOOP` is the reference: a servo under the plate,
a printed blade hanging plumb from it -- 152 g, its centre of mass under
the peg, and every number the catalog knows.
"""

import copy
import math

import pytest

from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.power import MODULE_IDLE_W
from pluggybot.rack import catalog
from pluggybot.rack.coupling import PEG_POWER_W
from pluggybot.workshop import spec, validate
from pluggybot.workshop.spec import Refused, parse, poses

SCOOP = {
  "name": "scoop",
  "parts": [
    {"id": "hinge", "part": "servo_fs90", "pos": [0, 0, -50],
     "axis": {"verb": "tilt", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
    {"id": "blade", "part": "scaffold_pla_box", "size": [4, 30, 70],
     "pos": [0, 0, -45], "on": "hinge"},
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
  assert 0.15 < validate.MODULE_MASS + tool.mass < 0.16
  assert validate.worst_moment(tool) < 0.01
  assert validate.hang_tilt_deg(tool) < 0.1
  assert [p.axis.verb for p in tool.axes] == ["tilt"]


def test_units_are_converted_once_at_the_door():
  """mm and degrees in, metres and radians out -- a pos of 50 mm is 0.05 m
  and a 90° range is π/2. Shown to fail by the bug this pins: the first
  draft placed a servo 20 012 mm out the front."""
  tool = parse(SCOOP)
  hinge = tool.by_id["hinge"]
  assert hinge.pos == pytest.approx((0.0, 0.0, -0.050))
  assert hinge.axis.hi == pytest.approx(math.pi / 2)
  assert hinge.axis.speed == pytest.approx(math.radians(600))
  assert tool.by_id["blade"].half == pytest.approx((0.002, 0.015, 0.035))


def test_a_part_on_an_actuator_rides_its_axis():
  """The blade on the hinge: at 90° about +y a point 45 mm under the hinge
  swings 45 mm out ahead of it (-x) -- checked by hand, so the validator's
  checks at the range ends are of the pose the tool will actually take."""
  tool = parse(SCOOP)
  stow = poses(tool)["blade"][0]
  up = poses(tool, {"tilt": math.pi / 2})["blade"][0]
  assert stow == pytest.approx((0.0, 0.0, -0.095))
  assert up == pytest.approx((-0.045, 0.0, -0.050))


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
  assert {r.split(":")[0] for r in reasons} >= {"bed", "board", "behind"}


def test_a_name_the_rack_already_has_is_refused():
  assert any("already has a module_pen" in r for r in refused(scoop(name="pen")))


def test_only_catalog_shelf_chosen_parts_with_known_numbers():
  # not a part
  assert any("no catalog part 'unobtainium'" in r for r in refused(
    with_part({"id": "x", "part": "unobtainium", "pos": [0, 0, -40]})))
  # a build-shelf part (the leg's motor) is not something a module is built from
  assert any("build shelf" in r for r in refused(
    with_part({"id": "x", "part": "gim8108_8", "pos": [0, 0, -40]})))
  # a candidate with no part behind it
  reasons = refused(with_part({"id": "x", "part": "module_servo", "pos": [0, 0, -40],
                               "axis": {"verb": "v", "dir": [0, 1, 0],
                                        "range": [0, 10], "stow": 0}}))
  assert any("candidate, not chosen (no part chosen)" in r for r in reasons)
  # chosen, on the catalog, and the catalog does not weigh it: the frame
  reasons = refused(with_part({"id": "x", "part": "module_frame", "pos": [0, 0, -40]}))
  assert any("does not know module_frame's mass" in r for r in reasons)
  # ...or does not size it: the peg
  reasons = refused(with_part({"id": "x", "part": "quad_tool_peg", "pos": [0, 0, -40]}))
  assert any("does not know its size" in r for r in reasons)


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
  """A 100 x 100 x 25 mm printed block is 310 g of PLA: over the arm's
  400 g with the plate, peg and scoop. Shown to fail by raising
  `arm.TOOL_MAX_KG` to 0.5."""
  raw = with_part({"id": "block", "part": "scaffold_pla_box", "size": [100, 100, 25],
                   "pos": [0, 0, -60]})
  assert any(r.startswith("mass:") and f"over the arm's {am.TOOL_MAX_KG * 1000:.0f} g" in r
             for r in refused(raw))


def test_a_tool_must_hang_plumb():
  """Hung on the trays nothing holds a tool level, and one hung off plumb
  is not hung (`rack.on_bay`, 2 deg; the rover's scoop, carried over, hung
  5 deg off in the rig). A 20 mm cube 60 mm ahead of the peg tilts it; the
  same cube under the peg does not. Shown to fail by dropping the rule."""
  raw = with_part({"id": "weight", "part": "scaffold_pla_box", "size": [20, 20, 20],
                   "pos": [-60, 0, -40]})
  reasons = refused(raw)
  assert any(r.startswith("hangs:") and f"past the {rk.HUNG_TILT_DEG:g}" in r for r in reasons)
  under = with_part({"id": "weight", "part": "scaffold_pla_box", "size": [20, 20, 20],
                     "pos": [0, 0, -40]})
  assert validate.hang_tilt_deg(validate.check(under)) < 0.1


def test_ahead_and_behind_are_judged_at_the_axis_ends():
  """The scoop's blade hangs plumb at stow and swings out ahead at 90°:
  its centre of mass is taken at every axis end, not read at the stow
  pose. Swung the other way, a 30 g block on the hinge puts it BEHIND the
  peg, on the lean-pad's post, and that is refused at the end that does."""
  tool = validate.check(SCOOP)
  assert validate.ahead_of_peg(tool, {"tilt": math.pi / 2}) > validate.ahead_of_peg(tool) + 0.002
  raw = scoop()
  raw["parts"][0]["axis"]["range"] = [-90, 0]
  raw["parts"][1] = {"id": "block", "part": "scaffold_pla_box", "size": [20, 30, 40],
                     "pos": [0, 0, -45], "on": "hinge"}
  reasons = refused(raw)
  assert any(r.startswith("ahead:") and "BEHIND the peg" in r and "low end" in r
             for r in reasons)


def test_the_moment_rule_cannot_bite_inside_the_mass_and_the_lever():
  """FINDING, from writing the rules: at most 400 g at most 60 mm ahead is
  0.235 N*m, under the arm's 0.35, so the moment rule is a backstop that a
  spec inside the other two never reaches."""
  assert validate.G * am.TOOL_MAX_KG * am.TOOL_MAX_AHEAD_M < am.TOOL_MAX_MOMENT_NM


def test_the_fork_volume_is_the_forks():
  """A part where the fork's V's hold the peg's ends: |y| 100 mm at the peg
  line -- and its mirror."""
  for y in (100, -100):
    raw = with_part({"id": "tab", "part": "scaffold_pla_box", "size": [10, 10, 10],
                     "pos": [0, y, 20]})
    assert any(r.startswith("fork:") and "'tab'" in r for r in refused(raw)), y


def test_the_tray_volume_is_the_racks():
  raw = with_part({"id": "tab", "part": "scaffold_pla_box", "size": [10, 10, 10],
                   "pos": [0, 45, 20]})
  assert any(r.startswith("trays:") and "'tab'" in r for r in refused(raw))


def test_a_hung_tool_leaves_its_bays_tags_in_view():
  """The claw's crossbar hid its bay's tags from the working pose, and no
  fetch of it fitted its bay (#407): a part in front of the plate beside
  the bay's middle, at the tags' height, is refused; the same part on the
  middle is not."""
  raw = with_part({"id": "wing", "part": "scaffold_pla_box", "size": [10, 10, 10],
                   "pos": [-40, 60, -80]})
  assert any(r.startswith("tags:") and "'wing'" in r for r in refused(raw))
  validate.check(with_part({"id": "wing", "part": "scaffold_pla_box", "size": [10, 10, 10],
                            "pos": [-40, 0, -80]}))
  # ...and the plate's own slab is no shelter: the camera looks past the
  # plate at the tags behind it, and a crossbar there 76 mm wide, as the
  # claw's first was, hid both (#407 review: it validated, and the working
  # pose decoded no tag of its bay)
  bar = with_part({"id": "cross", "part": "scaffold_pla_box", "size": [16, 76, 8],
                   "pos": [-2, 0, -104]})
  assert any(r.startswith("tags:") and "'cross'" in r for r in refused(bar))


def test_nothing_rises_into_the_racks_rail():
  """The rail the trays hang from runs 0.11-0.13 m over the peg and out to
  60 mm from the board, and a pick lifts a tool 56 mm (`arm.LIFT`): a part
  that reaches it, hung or lifted, jams there. A thin handle 50 mm ahead of
  the peg up to 150 mm validated and jammed on the rail at 15 deg (#407
  review); the same handle stopping 60 mm over the peg does not reach it."""
  raw = with_part({"id": "handle", "part": "scaffold_pla_box", "size": [4, 4, 140],
                   "pos": [-50, 0, 80]})
  assert any(r.startswith("rail:") and "'handle'" in r for r in refused(raw))
  validate.check(with_part({"id": "handle", "part": "scaffold_pla_box", "size": [4, 4, 50],
                            "pos": [-50, 0, 35]}))


def test_a_tool_hangs_centred_between_the_trays():
  """The trays hold the peg at +-45 mm: MEASURED in the rig, a centre of
  mass 30 mm to one side hangs, is taken and hangs back with the fork 15 mm
  either way (the walk-in's gate), at 35 mm a hang-back with the fork 15 mm
  toward it does not seat, and at 45 it does not hang (#407 review: 70 mm
  validated and tipped 47 deg off the trays). The rule is 25 mm."""
  def weight(y_mm):
    return {"name": "side", "parts": [{"id": "w", "part": "scaffold_pla_box",
                                       "size": [20, 40, 40], "pos": [0, y_mm, -150]}]}
  far = 35.0 * (validate.MODULE_MASS + 0.03968) / 0.03968        # its CoM at 35 mm
  assert any(r.startswith("side:") for r in refused(weight(far)))
  near = 20.0 * (validate.MODULE_MASS + 0.03968) / 0.03968
  assert abs(validate.side_of_peg(validate.check(weight(near))) - 0.020) < 0.001


def test_the_rack_board_is_a_hand_out_the_front():
  """Hung, the rack's back board is 84 mm ahead of the peg: a part may
  reach to 10 mm short of it."""
  raw = with_part({"id": "probe", "part": "scaffold_pla_box", "size": [60, 5, 5],
                   "pos": [-70, 0, -60]})
  assert any(r.startswith("board:") and "'probe'" in r for r in refused(raw))
  validate.check(with_part({"id": "probe", "part": "scaffold_pla_box", "size": [60, 5, 5],
                            "pos": [-40, 0, -60]}))


def test_carried_nothing_crosses_the_lidars_plane_or_the_forks_side():
  """Under the peg at most 200 mm at every pose (carried, the LIDAR's plane
  is under it), and nothing behind the plate's back face, where the fork's
  prongs and lean-pad are."""
  deep = with_part({"id": "keel", "part": "scaffold_pla_box", "size": [5, 5, 40],
                    "pos": [0, 0, -190]})
  assert any(r.startswith("drop:") and "'keel'" in r for r in refused(deep))
  back = with_part({"id": "boss", "part": "scaffold_pla_box", "size": [10, 10, 10],
                    "pos": [20, 0, -60]})
  assert any(r.startswith("behind:") and "'boss'" in r for r in refused(back))


def test_the_power_budget_through_the_peg():
  """Three FS90s at stall plus the ESP32 is 15 W against the peg's 12 (two
  were over the 6 W it had until 2026-09-15; the budget is a design
  decision argued at the constant). And a sensor whose draw is unknown
  (the Pi camera: the maker does not publish it) cannot be budgeted, so a
  tool cannot carry one -- refused with the gap named, not waved through
  at 0 W."""
  raw = with_part({"id": "hinge2", "part": "servo_fs90", "pos": [0, 0, -80],
                   "axis": {"verb": "tilt2", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
                  {"id": "hinge3", "part": "servo_fs90", "pos": [0, 20, -80],
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
  assert blade.mass == pytest.approx(density * 0.004 * 0.030 * 0.070)
