"""Guards for the hub-era mission loop (pluggybot/lifecycle.py, milestone 8)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.rack.coupling import CHARGE_BAY_Y, HUB_STATION_YS
from pluggybot.lifecycle import HubLifecycle, LOW_BATTERY_WH
from pluggybot.mission.mission import CHARGE_PIN_X, bay_standoff, charge_standoff
from pluggybot.rack.swap import PLUG_LATERAL


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path("models/room_hub.xml")


def test_charge_standoff_lines_up_the_chassis_not_the_fork():
  """Charging presses the BUMPER onto the pogo pins, so the chassis
  centreline must line up with the charge bay -- unlike a tool bay, where
  the FORK line is what matters. Getting this wrong aims the robot half a
  chassis-width off the pins."""
  sx, sy, hd = charge_standoff()
  from pluggybot.rack.localize import RackPose
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
  from pluggybot.mission.mission import HubMission
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
  from pluggybot.mission.mission import HubMission
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
@pytest.mark.parametrize("world", ["room_hub", "home"])
def test_full_hub_lifecycle(world):
  """The milestone-8 claim end to end: find the hub by looking, fetch a
  tool, use it elsewhere, stow it, then notice the battery and charge at
  the hub -- collision-free.

  The home arm (issue #9) is the one the website serves, and it is a
  harder room than room_hub: the mission has to get through a door gap to
  reach its use_at, so a bad world constant shows up here as a collision
  or a dropped module rather than as an exception. Every constant comes
  from world_config, so this is also the guard on that table.
  """
  from pluggybot.lifecycle import run_demo

  # ⚠ STOP ON THE CLAIM, NOT THE BUDGET (issue #54). The whole milestone-8
  # claim is settled once the tool has gone out and come back and the robot
  # has charged; `run_demo`'s default 600 s budget then buys a SECOND charge
  # cycle in home (219.7 -> 353.6 sim-seconds since the garden plate landed)
  # that repeats the first. Measured 369.6 s / 236.7 s of wall clock.
  #
  # The predicate is the SUCCESS condition, so a world constant that breaks
  # the mission never satisfies it: the run goes the whole distance and fails
  # on the same assertion it always did.
  def settled(life):
    return (life.swaps_done >= 2 and life.charge_cycles >= 1
            and life.mission.swap.module_state(life.module)["hung"])

  # ⚠ THE HOME ARM STARTS HALF-CHARGED (issue #84). On the old 1.1 Wh cell
  # one carry errand demanded a charge; the grown 3.0 Wh cell funds the same
  # day with 41 % to spare, and the loop -- correctly -- ends a done mission
  # without visiting the hub, which fails this test's whole point. Starting
  # at 45 % makes the SAME day need the hub again, and in a better order:
  # the errand is refused up front (CHARGE_FIRST: 0.69 + 0.90 reserve
  # against 1.35 held), the robot charges, then runs it -- the
  # defer-then-charge-then-resume path on real physics, a strictly harder
  # claim than the old "ran out mid-day", in LESS wall clock than the old
  # test. (A two-errand day was tried first and measured 430-500 s of wall
  # against the old 184; it also rolled #88's tag-yaw dice four times per
  # run instead of two, and flaked. Draining the start is the cheap lever.)
  #
  # charge_scale=3 ~ the cells' ratio (3.0/1.1), so the PRESS lasts about
  # what it always did on the old cell -- the pinned-odometry mechanics are
  # exercised for similar sim-seconds -- while the deployed rate stays
  # honest (test_battery pins the served default at 1.0).
  if world == "home":
    import os
    os.environ["PLUGGY_CHARGE_SCALE"] = "3"
    try:
      r = run_demo(world=world, stop_when=settled, battery_fraction=0.45)
    finally:
      os.environ.pop("PLUGGY_CHARGE_SCALE", None)
  else:
    r = run_demo(world=world, stop_when=settled)   # start pose from world config
  assert r["rack_discovered"], "never localized the rack from its tag"
  assert r["swaps_done"] == 2, "the tool errand did not complete"
  assert r["module_stowed"], "the module was not put back"
  assert r["charge_cycles"] >= 1, "never charged at the hub"
  assert r["battery"] > 0.5, "ended flat"
  # ⚠ NO LOW-WATER ASSERTION HERE, and that is deliberate. A first pass at
  # this added `min(fraction) > 0` as a "strictly stronger" replacement for
  # the line above, and room_hub failed it at 0.0 % -- correctly. The pack
  # reaching empty inside an errand is documented, honest behaviour on a demo
  # cell: `needs_charge` is checked BETWEEN errands and never inside one, so
  # an errand bigger than what is left cannot be survived by any charging
  # policy, and the committed home recording finishes a census at frac 0.000.
  # An assertion the design explicitly does not promise is a plausibility
  # guard rejecting the truth (SimNotes), not extra rigour.
  assert r["collision_steps"] == 0, "the robot hit something"
  # ...and it was PAID for the work, by code (issue #14). Both tasks a bare
  # mission performs are scoreable, and the verdicts come from evaluators
  # that measured the sim -- the lifecycle awards nothing itself.
  tasks = {v["task"] for v in r["verdicts"]}
  assert {"carry", "charge"} <= tasks, f"unscored mission: {tasks}"
  assert all(v["ok"] for v in r["verdicts"]), r["verdicts"]
  assert r["points"] == r["earned"] > 0


@pytest.mark.parametrize("world", ["room_hub", "home"])
def test_world_config_use_at_is_inside_its_own_world(world):
  """The errand destination must be free floor in the world it belongs to.

  room_hub's (-1.2, 2.5) sits inside wall_divider_0 in the home world
  (DIV_Y = 2.5, and the door gap is only x in [1.0, 2.0]) -- inheriting it
  there aims a 60 s drive at a wall, and nothing raises. This is a cheap
  geometric check of the same thing the slow home arm proves by driving.
  """
  from pluggybot.lifecycle import world_config
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  # Look straight down at the destination from above: the first thing the
  # ray meets must be the floor. A wall there stops it 1.2 m up.
  x, y = cfg["use_at"]
  hit = np.zeros(1, dtype=np.int32)
  dist = mujoco.mj_ray(model, data, np.array([x, y, 3.0]),
                       np.array([0.0, 0.0, -1.0]), None, 1, -1, hit)
  assert dist >= 0, f"use_at {cfg['use_at']} for {world} has no floor under it"
  top = 3.0 - float(dist)
  assert top < 0.05, (
    f"use_at {cfg['use_at']} for {world} is blocked by "
    f"{model.geom(int(hit[0])).name} standing {top:.2f} m tall")
