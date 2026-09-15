"""Slice B of issue #168: a validated spec becomes a module, and the rig
answers the four questions the build sequence asks -- off the world.

The rig is a spike run (~1.1 s); three of them here, deliberately: the
reference tool, a tool the validator passes and the rig refuses, and one
lateral offset each way of the envelope. Everything else is the emitted
XML and the registries, in milliseconds.
"""

import copy
import math

import mujoco
import pytest

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
  """The validator is static; the rig has gravity and a floor. A pendant
  that hangs below the rack's shelf is inside every envelope rule and
  falls off its trays on the rig -- `hangs` False, before any pick."""
  deep = copy.deepcopy(SCOOP)
  deep["parts"].append({"id": "pendant", "part": "scaffold_pla_box",
                        "size": [10, 10, 240], "pos": [0, 0, -180]})
  tool = validate.check(deep)      # the validator lets it through
  res, _ = build.rig(tool)
  assert not res["hangs"] and not res["ok"]


def test_the_lateral_envelope_holds_for_a_built_module(scoop):
  """±4 mm is the coupling's measured window; a built module keeps it on
  the rig (+4 mm passes) and loses it where the spike does (8 mm fails
  the return)."""
  ok, _ = build.rig(scoop, dy=0.004)
  assert ok["ok"]
  far, _ = build.rig(scoop, dy=0.008)
  assert far["picked"] and not far["stowed"]


def test_joint_range_is_degrees_and_ctrlrange_is_radians(scoop):
  """MEASURED: the compiler converts a hinge's `range` and not a position
  actuator's `ctrlrange`. Both must come out as π/2 in the compiled model
  or the servo would be asked for 90 radians."""
  xml = coupling.scene_xml(face=build.face_xml(scoop, "tool"),
                           actuators=build.actuator_xml(scoop, "tool"))
  model = mujoco.MjModel.from_xml_string(xml)
  j = model.joint("tool_tilt_joint")
  a = model.actuator("tool_tilt")
  assert model.jnt_range[j.id][1] == pytest.approx(math.pi / 2)
  assert model.actuator_ctrlrange[a.id][1] == pytest.approx(math.pi / 2)
  assert model.actuator_forcerange[a.id][1] == pytest.approx(0.147)
  assert 'range="0.000 90.000"' in build.face_xml(scoop, "tool")


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
  xml = coupling.scene_xml(face=face)
  assert "tool_peg_l" in xml and "tool_peg_insul" in xml


def test_the_spike_did_not_move_without_a_face():
  """The tolerance sweep's scene, compiled: same wall, same push, same
  peg, same camera, same rest height. (`test_hub_coupling.py` is the
  behaviour; this is the geometry the new parameters default to.)"""
  model = mujoco.MjModel.from_xml_string(coupling.scene_xml())
  back = model.geom("hub_back")
  assert model.geom_pos[back.id] == pytest.approx((-0.018, 0.0, 0.10))
  assert model.geom_size[back.id] == pytest.approx((0.004, 0.10, 0.10))
  assert model.actuator_ctrlrange[model.actuator("push").id][1] == 10.0
  assert model.geom("tool_peg").id >= 0
  assert model.cam_pos[model.camera("side").id] == pytest.approx((0.30, -0.30, 0.28))
  assert model.body_pos[model.body("rail").id][2] == pytest.approx(coupling.PEG_Z)


def test_module_for_goes_through_module_xml(scoop):
  xml = build.module_for(scoop, 0.09, 0.125, 0.30)
  assert '<body name="module_scoop"' in xml
  assert 'name="module_scoop_body"' in xml and 'name="module_scoop_peg_l"' in xml
  assert 'name="module_scoop_hinge"' in xml and 'name="module_scoop_blade"' in xml
  assert 'mass="0.100"' in xml           # the plate: MODULE_MASS less the peg


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

    xml = coupling.scene_xml(face=build.face_xml(tool, "module_feeler"),
                             actuators=build.actuator_xml(tool, "module_feeler"))
    model = mujoco.MjModel.from_xml_string(xml)
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
    q = model.joint("tool").qposadr[0] if "tool" in [model.joint(i).name for i in range(model.njnt)] else 0
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
    {"id": "mast", "part": "scaffold_pla_box", "size": [10, 10, 30], "pos": [-10, 0, -20]},
    {"id": "cam", "part": "esp32_cam", "pos": [-20, 0, -30], "on": "mast"}]}
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
  eyed["parts"].append({"id": "eye", "part": "esp32_cam", "pos": [10, 0, -20]})
  validate.check(eyed)                 # 6.95 W under 12
  three = copy.deepcopy(SCOOP)
  for i, y in enumerate((-15, 15)):
    three["parts"].append({"id": f"servo{i}", "part": "servo_fs90", "pos": [-20, y, -20],
                           "axis": {"verb": f"turn{i}", "dir": [0, 1, 0], "range": [0, 90],
                                    "stow": 0}})
  with pytest.raises(Refused, match=r"power: 15\.0 W .* over its 12 W"):
    validate.check(three)
