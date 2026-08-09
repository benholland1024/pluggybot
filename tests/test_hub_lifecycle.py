"""Guards for the hub-era mission loop (hub/lifecycle.py, milestone 8)."""

import math

import mujoco
import pytest

from pluggybot.hub.coupling import CHARGE_BAY_Y, HUB_STATION_YS
from pluggybot.hub.lifecycle import HubLifecycle, LOW_BATTERY_WH
from pluggybot.hub.mission import CHARGE_PIN_X, bay_standoff, charge_standoff
from pluggybot.hub.swap import PLUG_LATERAL


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path("models/room_hub.xml")


def test_charge_standoff_lines_up_the_chassis_not_the_fork():
  """Charging presses the BUMPER onto the pogo pins, so the chassis
  centreline must line up with the charge bay -- unlike a tool bay, where
  the FORK line is what matters. Getting this wrong aims the robot half a
  chassis-width off the pins."""
  sx, sy, hd = charge_standoff()
  from pluggybot.hub.localize import RackPose
  rack = RackPose.prior()
  px, py = rack.to_world(CHARGE_PIN_X, CHARGE_BAY_Y)
  lateral = -(px - sx) * math.sin(hd) + (py - sy) * math.cos(hd)
  assert abs(lateral) < 1e-9, "charge approach must not carry a plug offset"
  # ...whereas a tool bay must
  bx, by, bhd = bay_standoff(HUB_STATION_YS[0])
  hx, hy = rack.to_world(0.09, HUB_STATION_YS[0])
  bay_lat = -(hx - bx) * math.sin(bhd) + (hy - by) * math.cos(bhd)
  assert abs(abs(bay_lat) - PLUG_LATERAL) < 1e-6


def test_carried_module_survives_a_turn(room_model):
  """The fork's V-notches hold a peg vertically and fore-aft but NOT along
  its own axis. Without end-stops the first carried module walked sideways
  off the fork during a turn and landed on the floor 2.5 m away."""
  from pluggybot.hub.mission import HubMission
  data = mujoco.MjData(room_model)
  mission = HubMission(room_model, data)
  try:
    mission.start_at(-0.9, 4.6, math.pi / 2)
    mission._spin()
    mission.swap_at_bay(HUB_STATION_YS[0], "pick")
    assert mission.swap.module_state("module_lcd")["on_fork"], "pick failed"
    mission.drive_to(-1.2, 2.5, timeout=60.0)      # a turn and a haul
    st = mission.swap.module_state("module_lcd")
  finally:
    mission.close()
  assert st["on_fork"], f"the tool fell off while driving (at {st['pos']})"
  assert st["pos"][2] > 0.15, "the module ended up on the floor"


def test_arm_is_stowed_after_a_swap(room_model):
  """An extended fork rides at module height and sweeps a rack clean --
  measured, by the robot knocking down the module it had just stowed while
  driving past to the charge bay. Driving configuration is arm retracted."""
  from pluggybot.hub.mission import HubMission
  data = mujoco.MjData(room_model)
  mission = HubMission(room_model, data)
  try:
    mission.start_at(-0.9, 4.6, math.pi / 2)
    mission._spin()
    mission.swap_at_bay(HUB_STATION_YS[0], "pick")
    ext = float(data.ctrl[room_model.actuator("arm").id])
  finally:
    mission.close()
  assert ext == pytest.approx(0.0, abs=1e-6), "the arm was left deployed"


def test_low_battery_is_an_absolute_reserve(room_model):
  """Milestone 7's lesson, inherited: the cost of getting home is set by
  the room, not by the pack, so the go-charge threshold is absolute energy.
  A big pack at 30 % must NOT be considered low."""
  data = mujoco.MjData(room_model)
  life = HubLifecycle(room_model, data, battery_wh=4.0)
  life.battery.energy_wh = 4.0 * 0.30           # 30 % of a large pack
  assert not life.needs_charge
  life.battery.energy_wh = LOW_BATTERY_WH * 0.9
  assert life.needs_charge


@pytest.mark.slow
def test_full_hub_lifecycle():
  """The milestone-8 claim end to end: find the hub by looking, fetch a
  tool, use it elsewhere, stow it, then notice the battery and charge at
  the hub -- collision-free."""
  from pluggybot.hub.lifecycle import run_demo
  r = run_demo(start=(0.5, 3.0, math.pi / 2))
  assert r["rack_discovered"], "never localized the rack from its tag"
  assert r["swaps_done"] == 2, "the tool errand did not complete"
  assert r["module_stowed"], "the module was not put back"
  assert r["charge_cycles"] >= 1, "never charged at the hub"
  assert r["battery"] > 0.5, "ended flat"
  assert r["collision_steps"] == 0, "the robot hit something"
