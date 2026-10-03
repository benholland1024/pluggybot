"""Slice B of issue #168, on legs (#407): a validated spec becomes a module,
and the rig -- the legs rack, the arm's own fork on a carrier -- answers the
questions the build sequence asks, off the world.

The rig is a spike run (~0.3 s); five of them here, deliberately: the
reference tool, a tool the validator passes and the rig refuses, and the
fork off the bay's middle where it still takes a tool and where it no
longer does. Everything else is the emitted XML and the registries, in
milliseconds.
"""

import copy
import math

import mujoco
import pytest

from pluggybot.legs import rack as rk
from pluggybot.procedure import axes
from pluggybot.rack import coupling
from pluggybot.workshop import build, validate
from test_workshop import SCOOP


@pytest.fixture(scope="module")
def scoop():
  return validate.check(SCOOP)


def test_the_reference_tool_passes_the_rig(scoop):
  """Hangs, picked, conducts, works, held through use, stowed."""
  res, _ = build.rig(scoop)
  assert res["ok"], res
  assert res["poles"] == {"left": True, "right": True, "powered": True}


def test_the_rig_refuses_what_the_validator_cannot_see():
  """The validator is static; the rig has gravity. A 390 g tool whose
  FS90 must hold 0.18 N*m at its axis's end, against the servo's 0.147,
  is inside every envelope rule and fails `works` on the rig. Shown to
  pass the rig -- wrongly -- while an actuator's load carried a stand-in
  inertia, which replaced its parts' masses: every moving part of a built
  tool was massless (#407)."""
  heavy = copy.deepcopy(SCOOP)
  heavy["parts"][1] = {"id": "block", "part": "scaffold_pla_box", "size": [20, 100, 100],
                       "pos": [0, 0, -75], "on": "hinge"}
  tool = validate.check(heavy)      # the validator lets it through
  assert tool.by_id["block"].mass * validate.G * 0.075 > tool.by_id["hinge"].axis.force
  res, _ = build.rig(tool)
  assert res["hangs"] and res["picked"] and not res["works"]["tilt"] and not res["ok"]


def test_the_fork_takes_a_built_module_off_the_bays_middle_as_far_as_a_walk_in_stops(scoop):
  """A walk-in stops within 15 mm across (`rack.LINEUP_ACROSS`), and a
  built module keeps the fork's capture there; 30 mm off, the fork does
  not take it."""
  assert rk.LINEUP_ACROSS == pytest.approx(0.015)
  for dy in (0.015, -0.015):
    ok, _ = build.rig(scoop, dy=dy)
    assert ok["ok"], (dy, ok)
  far, _ = build.rig(scoop, dy=0.030)
  assert not far["picked"] and not far["ok"]


def test_a_hinges_range_and_its_ctrlrange_are_radians_where_it_compiles_in_radians(scoop):
  """MEASURED: a compiler in degrees converts a hinge's `range` and never a
  position actuator's `ctrlrange`, and every legs emitter compiles in
  radians (the rig, the seam's child spec). Written in degrees, a 0-90 deg
  hinge had a 0-90 RADIAN limit -- none -- and a heavy blade swung over
  the top (#407)."""
  model = mujoco.MjModel.from_xml_string(build.rig_xml(scoop))
  j = model.joint("tool_tilt_joint")
  a = model.actuator("tool_tilt")
  assert model.jnt_range[j.id][1] == pytest.approx(math.pi / 2)
  assert model.actuator_ctrlrange[a.id][1] == pytest.approx(math.pi / 2)
  assert model.actuator_forcerange[a.id][1] == pytest.approx(0.147)
  assert 'range="0.000000 1.570796"' in build.face_xml(scoop, "tool")


def test_the_face_carries_no_contact_parameters(scoop):
  """A built tool inherits the generator's contact rules and sets none of
  its own: no friction, priority, solimp, condim in what it emits -- the
  spec has no field for them and the emitter adds none."""
  face = build.face_xml(scoop, "tool") + build.actuator_xml(scoop, "tool")
  for word in ("friction", "priority", "solimp", "solref", "condim"):
    assert word not in face
  # the load is a CHILD body of the module, so its contact with the servo
  # it rides is filtered by MuJoCo's parent-child rule
  assert '<body name="tool_hinge_load"' in face
  # and the module carries the real split peg, so the rig can read poles
  xml = build.rig_xml(scoop)
  assert "tool_peg_l" in xml and "tool_peg_insul" in xml


def test_the_spike_did_not_move_without_a_face():
  """The tolerance sweep's scene, compiled: same wall, same push, same
  peg, same camera, same rest height -- the geometry the new parameters
  default to."""
  model = mujoco.MjModel.from_xml_string(coupling.scene_xml())
  back = model.geom("hub_back")
  assert model.geom_pos[back.id] == pytest.approx((-0.018, 0.0, 0.10))
  assert model.geom_size[back.id] == pytest.approx((0.004, 0.10, 0.10))
  assert model.actuator_ctrlrange[model.actuator("push").id][1] == 10.0
  assert model.geom("tool_peg").id >= 0
  assert model.cam_pos[model.camera("side").id] == pytest.approx((0.30, -0.30, 0.28))
  assert model.body_pos[model.body("rail").id][2] == pytest.approx(coupling.PEG_Z)


def test_module_for_goes_through_the_racks_own_tool_xml(scoop):
  """A built module is `legs.rack.tool_xml`'s, as the hand-built ones are:
  its default class, its body, the 220 mm peg and the plate's mass."""
  default, xml = build.module_for(scoop, (0.09, 0.125, 0.30))
  assert default == rk.tool_default("module_scoop")
  assert '<body name="module_scoop"' in xml and 'childclass="module_scoop_tool"' in xml
  assert 'name="module_scoop_body"' in xml and 'name="module_scoop_peg_l"' in xml
  assert 'name="module_scoop_hinge"' in xml and 'name="module_scoop_blade"' in xml
  assert f'mass="{rk._f(rk.MODULE_MASS - rk.peg_kg())}"' in xml   # the plate less the peg


def test_register_puts_the_verb_in_the_registries(scoop):
  """`move("scoop.tilt", ...)` and `read("scoop.tilt")` exist once the tool
  is registered, gated on the module being on the fork, in the sim's
  units. Cleaned up after: the registries are the process's."""
  before_a, before_s = set(axes.AXES), set(axes.SENSORS)
  try:
    names = build.register(scoop)
    assert names == ["scoop.tilt"]
    axis = axes.AXES["scoop.tilt"]
    assert axis.requires == "module_scoop" and axis.actuator == "module_scoop_tilt"
    assert axis.lo == 0.0 and axis.hi == pytest.approx(math.pi / 2)
    assert axis.speed == pytest.approx(math.radians(600)) and axis.unit == "rad"
    assert axis.run is not None
    assert axes.SENSORS["scoop.tilt"].requires == "module_scoop"
    assert "scoop.tilt" in {a["name"] for a in axes.describe()["axes"]}
  finally:
    for k in set(axes.AXES) - before_a:
      del axes.AXES[k]
    for k in set(axes.SENSORS) - before_s:
      del axes.SENSORS[k]


#: The scoop with a microswitch on the blade's underside (issue #199): the
#: first tool that can FEEL. Mounted on the hinge's load so it rides the
#: tilt, 40 mm out along the blade, 6 mm below it.
FEELER = {
  "name": "feeler",
  "parts": [
    *copy.deepcopy(SCOOP["parts"]),
    {"id": "switch", "part": "bumper_switch", "pos": [-50, 0, -14], "on": "hinge"},
  ],
}


def test_a_switch_on_a_tool_is_a_contact_sense_and_reads_the_world():
  """A catalog sensor whose `sense` is `contact` becomes `<tool>.<id>.contact`
  when the tool is registered, after the axes; it reads 0 while the switch
  touches nothing but the module it is part of, and 1 once it is pressed
  into something else -- the bumper's own criterion, off the contact list.
  Pinned on the spike scene with a fake life: the module is a free body,
  so pushing it down until the switch meets the floor is one qpos write.
  A retire takes the sense out with the verbs."""
  tool = validate.check(FEELER)
  before_a, before_s = set(axes.AXES), set(axes.SENSORS)
  try:
    names = build.register(tool)
    assert names == ["feeler.tilt", "feeler.switch.contact"]
    sense = axes.SENSORS["feeler.switch.contact"]
    assert sense.requires == "module_feeler"

    model = mujoco.MjModel.from_xml_string(build.rig_xml(tool, body="module_feeler"))
    data = mujoco.MjData(model)

    class Life:
      pass
    life = Life()
    life.model, life.data = model, data
    mujoco.mj_forward(model, data)
    # hanging on the trays: the switch touches nothing outside the module
    # (the blade it is mounted beside is the module's own root)
    assert sense.read(life) == 0.0
    # ...pressed into the floor: the free body dropped until the switch,
    # the lowest thing on the module, is inside the plane
    q = model.joint("module_feeler_free").qposadr[0]
    gid = model.geom("module_feeler_switch").id
    lowest = float(data.geom_xpos[gid][2]) - float(model.geom_size[gid][2])
    data.qpos[q + 2] -= lowest + 0.002
    mujoco.mj_forward(model, data)
    assert sense.read(life) == 1.0
    assert build.unregister("module_feeler") == ["feeler.tilt", "feeler.switch.contact"]
    assert "feeler.switch.contact" not in axes.SENSORS
  finally:
    for k in set(axes.AXES) - before_a:
      del axes.AXES[k]
    for k in set(axes.SENSORS) - before_s:
      del axes.SENSORS[k]


def test_a_camera_part_is_a_sensor_with_no_contact_sense():
  """`esp32_cam` is a sensor the catalog fully knows (issue #199) and it
  validates onto a tool, but it is not something to `read()` a touch off:
  no `sense`, no `.contact` name."""
  eyed = {"name": "eye", "parts": [
    {"id": "mast", "part": "scaffold_pla_box", "size": [10, 10, 30], "pos": [0, 0, -45]},
    {"id": "cam", "part": "esp32_cam", "pos": [0, 0, -40], "euler": [0, 90, 0],
     "on": "mast"}]}
  tool = validate.check(eyed)
  assert build.contact_sensors(tool) == []
  assert build.register(tool) == []


def test_the_peg_budget_fits_a_servo_and_an_eye_and_refuses_three_servos():
  """`PEG_POWER_W` is a design decision (coupling.py), and this pins what it
  buys. At 6 W the FS90 at stall (4.8) plus the ESP32-CAM with its flash
  (1.55) plus the module's 0.6 W was 6.95 W and a tool that looked AND
  moved was refused -- found by #199, one part earlier than the "two
  actuators" its acceptance expected. Raised to 12 W (1 A at 12 V,
  2026-09-15): a servo and an eye fit, and three servos at stall (15 W)
  still do not, since the validator sums every ceiling as if simultaneous."""
  from pluggybot.workshop.spec import Refused
  eyed = copy.deepcopy(SCOOP)
  eyed["parts"].append({"id": "eye", "part": "esp32_cam", "pos": [0, 0, -110],
                        "euler": [0, 90, 0]})
  validate.check(eyed)                 # 6.95 W under 12
  three = copy.deepcopy(SCOOP)
  for i, y in enumerate((-15, 15)):
    three["parts"].append({"id": f"servo{i}", "part": "servo_fs90", "pos": [-20, y, -20],
                           "axis": {"verb": f"turn{i}", "dir": [0, 1, 0], "range": [0, 90],
                                    "stow": 0}})
  with pytest.raises(Refused, match=r"power: 15\.0 W .* over its 12 W"):
    validate.check(three)
