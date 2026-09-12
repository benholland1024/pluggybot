"""Two robots in one world (issue #167, M12), slice A: the namespacing.

A second robot is the same model file attached with a prefix; a
`RobotHandle` is that prefix, and every class that resolves a robot element
does so through it. The first robot keeps the bare names, so a single-robot
world is byte-for-byte what it was -- the parity flight is the proof at
mission scale, and the first test here is it at the scale of a spin and a
drive.
"""

import pathlib
import re

import mujoco
import numpy as np
import pytest

from pluggybot.mission.mission import HubMission
from pluggybot.rack.coupling import (
  HUB_STATION_YS, module_power_contact, rack_charge_contact,
)
from pluggybot.rack.swap import HubSwap
from pluggybot.robot import FIRST, SECOND, RobotHandle, world_with_robots
from pluggybot.tools.gripper import CLAW_MODULE

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"
PARK = (5.5, 5.5)          # room_hub: a far corner, off every route


@pytest.mark.parametrize("world", ["models/room_hub.xml", "models/home_world.xml"])
def test_two_namespaced_robots_compile(world):
  model = world_with_robots(world, second_at=(9.0, -5.0) if "home" in world else PARK)
  names = {model.body(i).name for i in range(model.nbody)}
  assert FIRST.root in names and SECOND.root in names
  for el in ("lift", "arm", "left_motor", "right_motor"):
    model.actuator(SECOND.el(el))
  for el in ("dock_eye", "left_eye"):
    model.camera(SECOND.el(el))
  model.sensor(SECOND.el("imu_gyro"))
  for el in ("lidar", "fork_vertex"):
    model.site(SECOND.el(el))
  assert FIRST.qpos_adr(model) == 0 and SECOND.qpos_adr(model) > 7
  # ...and the rack, the bays and the modules are the WORLD's, not prefixed
  assert "rack" in names and "module_pen" in names
  assert not any(n.startswith("r2_module") or n.startswith("r2_rack") for n in names)


def _subtree_hash(model, data, handle: RobotHandle):
  root = model.body(handle.root).id
  ids = [i for i in range(model.nbody) if model.body_rootid[i] == root]
  return np.concatenate([data.xpos[ids].ravel(), data.xquat[ids].ravel()]).tobytes()


def test_a_parked_second_robot_leaves_the_first_robots_trajectory_byte_identical():
  """The parity claim at test scale. MuJoCo 3.10's solver is island-
  separable, so an extra robot resting on the floor changes nothing in the
  first robot's numbers -- measured near and far before this was built."""
  def fly(model):
    data = mujoco.MjData(model)
    m = HubMission(model, data, viewer=None, realtime=False)
    m.start_at(0.5, 3.0, 1.5708)
    m._spin()
    m.drive_to(1.5, 2.0, timeout=12.0)
    return _subtree_hash(model, data, FIRST), m.pose, float(data.time)
  alone = fly(mujoco.MjModel.from_xml_path("models/room_hub.xml"))
  with_r2 = fly(world_with_robots("models/room_hub.xml", second_at=PARK))
  assert alone == with_r2


def test_the_second_robot_drives_through_its_handle_and_the_first_stays_put():
  model = world_with_robots("models/room_hub.xml", second_at=PARK)
  data = mujoco.MjData(model)
  first = HubMission(model, data, viewer=None, realtime=False, handle=FIRST)
  second = HubMission(model, data, viewer=None, realtime=False, handle=SECOND)
  first.start_at(0.5, 3.0, 1.5708)
  second.start_at(3.0, 3.0, 0.0)
  before = _subtree_hash(model, data, FIRST)
  q = SECOND.qpos_adr(model)
  x0 = float(data.qpos[q])
  second._drive(2.0, 0.15, 0.0)
  assert second.face(1.2)
  assert float(data.qpos[q]) - x0 > 0.2, "the second robot did not move"
  assert abs(second.pose[2] - 1.2) < 0.05
  assert abs(first.pose[0] - 0.5) < 1e-6 and abs(first.pose[1] - 3.0) < 1e-6
  # the first robot's bodies did not move while the second drove (it was
  # held by its own brake; the hash is over its whole subtree)
  after = _subtree_hash(model, data, FIRST)
  assert np.frombuffer(before, float).round(3).tolist() == \
    np.frombuffer(after, float).round(3).tolist()


def test_the_second_robot_picks_a_module_and_only_its_own_fork_powers_it():
  """The coupling through the handle: the second robot's fork plates are
  `r2_fork_v*`, so `module_power_contact` with its prefix says powered while
  the first robot's says not -- a module on the other robot's fork is not
  this robot's."""
  model = world_with_robots("models/hub_world.xml", second_at=(1.5, 1.5))
  data = mujoco.MjData(model)
  first = HubSwap(model, data, handle=FIRST)
  first.place_at_standoff(HUB_STATION_YS[0])          # out of the way, bay A
  second = HubSwap(model, data, handle=SECOND)
  second.place_at_standoff(HUB_STATION_YS[3])         # the claw's bay
  second.pick()
  assert second.module_state(CLAW_MODULE)["on_fork"]
  assert module_power_contact(model, data, CLAW_MODULE, SECOND.prefix)
  assert not module_power_contact(model, data, CLAW_MODULE, FIRST.prefix)
  assert not first.module_state(CLAW_MODULE)["on_fork"]
  assert not rack_charge_contact(model, data, SECOND.prefix)


# ---- the fence: no bare robot name in mission code -----------------------------

#: Element names that belong to a robot, which mission code may only reach
#: through a handle. Module, rack, bay and board names are the WORLD's.
ROBOT_ELEMENTS = ("chassis", "lift", "arm", "left_motor", "right_motor",
                  "left_wheel_joint", "right_wheel_joint", "lift_joint",
                  "arm_joint", "imu_gyro", "dock_eye", "left_eye", "lidar",
                  "fork_vertex", "pluggybot")
#: The mission stack: what a second robot runs a copy of.
MISSION_CODE = ("lifecycle.py", "power.py", "rack/swap.py", "mission/mission.py",
                "mission/errand.py", "tools/drawing.py", "tools/gripper.py",
                "tools/dispenser.py", "tools/screen.py", "procedure/steps.py",
                "procedure/axes.py", "procedure/lang.py", "perception/lidar.py",
                "rack/localize.py", "rack/tags.py", "economy/census.py")


def test_mission_code_resolves_every_robot_element_through_the_handle():
  bare = re.compile(r'\.(body|geom|joint|actuator|site|camera|sensor)\("('
                    + "|".join(ROBOT_ELEMENTS) + r')"\)')
  root_index = re.compile(r"qpos\[(0|1|2|3:7|:7|:3)\]")
  bad = []
  for rel in MISSION_CODE:
    text = (SRC / rel).read_text()
    for n, line in enumerate(text.splitlines(), 1):
      if bare.search(line) or root_index.search(line):
        bad.append(f"{rel}:{n}: {line.strip()}")
  assert not bad, "bare robot names in mission code:\n  " + "\n  ".join(bad)
