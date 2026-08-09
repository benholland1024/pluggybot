"""Guards for hub-in-room navigation (hub/mission.py, milestone 8)."""

import math

import mujoco
import pytest

from pluggybot.hub.coupling import HUB_STATION_YS, RACK_HANG_X, rack_frame_to_world
from pluggybot.hub.mission import bay_standoff, rack_heading


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path("models/room_hub.xml")


def test_room_hub_shares_room_1_scenery(room_model):
  """room_hub must be room_1's floor plan plus the rack -- not a fork of it.
  If these drift apart, mapping/navigation results stop being comparable."""
  room_1 = mujoco.MjModel.from_xml_path("models/room_1.xml")
  for name in ("wall-north", "wall-east", "corner-box", "floor-box",
               "L-box-north", "outlet_a", "decoy_switch_w"):
    a = room_1.body(name).pos
    b = room_model.body(name).pos
    assert all(abs(float(a[k]) - float(b[k])) < 1e-9 for k in range(3)), \
      f"{name} moved between room_1 and room_hub"


def test_bay_standoff_faces_the_rack(room_model):
  """The hand-off pose must sit in front of its bay, facing the rack, at the
  swap standoff -- the geometry the whole terminal approach assumes."""
  hd = rack_heading()
  for station_y in HUB_STATION_YS:
    sx, sy, h = bay_standoff(station_y)
    assert abs(h - hd) < 1e-9
    hx, hy = rack_frame_to_world(RACK_HANG_X, station_y)
    # distance along the approach axis, and lateral offset (plug offset only)
    along = (hx - sx) * math.cos(h) + (hy - sy) * math.sin(h)
    lateral = -(hx - sx) * math.sin(h) + (hy - sy) * math.cos(h)
    assert 0.4 < along < 0.6, f"standoff {along:.3f} m from the bay"
    assert abs(abs(lateral) - 0.05) < 1e-6, "fork line should be plug-offset"
    assert -2.0 < sx < 2.0 and 0.0 < sy < 6.0, "standoff pose is outside room 1"


def test_module_state_is_rack_frame_relative(room_model):
  """The stow verdict must be computed in the RACK's frame. Comparing world
  coordinates against rack-local constants silently reported every correct
  placement in room_hub as a failure (the rack sits at (-0.9, 5.99) yaw -90
  there, and at the origin in the bare hub world)."""
  from pluggybot.hub.swap import HubSwap
  data = mujoco.MjData(room_model)
  mujoco.mj_forward(room_model, data)
  swap = HubSwap(room_model, data)
  st = swap.module_state("module_lcd")
  assert st["hung"], "a module untouched on its rack must read as hung"
  assert st["bay_err_mm"] < 30.0


@pytest.mark.slow
def test_full_hub_mission():
  """The milestone-8 claim, end to end: map the room, navigate to the rack,
  fine-align on the fiducials, pick the LCD module, carry it across the
  room, come back, and hang it up -- collision-free."""
  from pluggybot.hub.mission import run_demo
  result = run_demo(start=(0.5, 3.0, math.pi / 2))
  assert result["picked"], "never got the module off the rack"
  assert result["returned"], "never hung the module back up"
  assert result["collision_steps"] == 0, "the robot hit something"
