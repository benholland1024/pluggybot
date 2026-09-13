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
