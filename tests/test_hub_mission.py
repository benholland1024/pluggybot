"""Guards for hub-in-room navigation (mission/mission.py, milestone 8)."""

import math

import mujoco
import pytest

from pluggybot.rack.coupling import HUB_STATION_YS, RACK_HANG_X, rack_frame_to_world
from pluggybot.mission.mission import bay_standoff, rack_heading


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
  from pluggybot.rack.swap import HubSwap
  data = mujoco.MjData(room_model)
  mujoco.mj_forward(room_model, data)
  swap = HubSwap(room_model, data)
  st = swap.module_state("module_lcd")
  assert st["hung"], "a module untouched on its rack must read as hung"
  assert st["bay_err_mm"] < 30.0


def test_a_second_fetch_works_after_a_stow(room_model):
  """Issue #10: an errand you can only run ONCE is not a loop.

  Every mission before this one fetched exactly once, so nothing ever
  picked a module up after putting one down -- and the lift preset was
  established once, at `start_at`. A stow ends with the lift at RELEASE
  height, 50 mm below where a pick must enter, so the second fetch slid the
  fork under the peg, missed it, and came away empty.

  It fails SILENTLY, which is why it needs a test rather than a glance at a
  log: the travel is ranged correctly, the approach reports "arrived", and
  the only sign is a module still hanging in its bay. Bay A and the LCD,
  because nothing here is about the pen -- any tool inherits it.
  """
  from pluggybot.rack.coupling import module_power_contact
  from pluggybot.mission.mission import HubMission
  data = mujoco.MjData(room_model)
  mission = HubMission(room_model, data)
  bay = HUB_STATION_YS[0]
  try:
    mission.start_at(-0.9, 4.6, math.pi / 2)
    mission._spin()
    mission.swap_at_bay(bay, "pick", module="module_lcd")
    assert mission.swap.module_state("module_lcd")["on_fork"], "first pick failed"
    mission.swap_at_bay(bay, "return", module="module_lcd")
    assert mission.swap.module_state("module_lcd")["hung"], "stow failed"

    mission.swap_at_bay(bay, "pick", module="module_lcd")
    st = mission.swap.module_state("module_lcd")
    powered = module_power_contact(room_model, data, "module_lcd")
  finally:
    mission.close()
  assert st["on_fork"], (
    "the second fetch came away empty -- the fork almost certainly entered "
    "at the lift the STOW left it at, not the align preset")
  assert powered, "the second fetch seated the module but did not power it"


def test_a_drive_ends_with_a_terminal_approach_and_sweeps_before_it(room_model, monkeypatch):
  """The last leg of `drive_to` -- its final waypoint and the goal -- is a
  terminal approach (`slow_radius`), every waypoint before it the sweeping
  law (issue #277; `ARRIVAL_SLOW_RADIUS`). MEASURED why: a stow begun 13 cm
  from the bay's standoff handed the plain law a goal closer than its
  overshoot and orbited it for 690 deg, and whether the stagnation cut then
  fell inside the 15 cm 'close enough' was the map's coin -- the rail
  beside bay E turned it over. Kinematic, no physics: the plan is stubbed
  to three waypoints, the step to a teleport, and `drive_toward` is spied
  for the `slow_radius` each call carries. Fails without the wiring: every
  call carries None."""
  from pluggybot.mission import mission as mm
  from pluggybot.mission.mission import HubMission
  from pluggybot import tick
  data = mujoco.MjData(room_model)
  mission = HubMission(room_model, data, realtime=False)
  try:
    mission.start_at(0.0, 0.0, 0.0)
    goal = (0.9, 0.0)
    legs = [(0.3, 0.0), (0.6, 0.0), (0.9, 0.0)]
    mission._plan_to = lambda wx, wy: list(legs)
    calls = []
    real = mm.drive_toward

    def spy(pose, waypoint, slow_radius=None):
      calls.append((tuple(round(v, 3) for v in waypoint), slow_radius))
      return real(pose, waypoint, slow_radius=slow_radius)
    monkeypatch.setattr(mm, "drive_toward", spy)

    def teleport(v, w):
      # the step: 5 cm along the bow per call, into the reckoner directly --
      # no physics, no wheels
      r = mission.swap.reckoner
      r.x += 0.05 * math.cos(r.theta)
      r.y += 0.05 * math.sin(r.theta)
      data.time += 0.1
      yield 0.0, 0.0
    mission._nav_routine = teleport
    arrived = tick.run(mission.swap, mission.drive_to_routine(*goal, timeout=30.0))
  finally:
    mission.close()
  assert arrived
  by_target = {}
  for target, radius in calls:
    by_target.setdefault(target, set()).add(radius)
  assert by_target[(0.3, 0.0)] == {None} and by_target[(0.6, 0.0)] == {None}, by_target
  assert by_target[(0.9, 0.0)] == {mm.ARRIVAL_SLOW_RADIUS}, by_target


def test_the_drive_back_to_a_bay_standoff_gives_up_at_its_budget(room_model, monkeypatch):
  """Issue #339, off the deployed pair: Rowan was knocked over mid-pick and
  `refine_standoff`'s drive back to the standoff had no budget. On its side
  it could not get there, so the loop outlived the death; the timer stood it
  up at the far end of the house, and the same loop drove it straight at the
  rack's standoff, into a wall at full torque until the pack was flat --
  thirteen lives. Kinematic, no physics: a robot that cannot move (the step
  only advances the clock) must still get out, within three passes of
  `REFINE_BUDGET_S`. Fails without the budget: the step cap fires."""
  from pluggybot import tick
  from pluggybot.mission import mission as mm
  data = mujoco.MjData(room_model)
  mission = mm.HubMission(room_model, data, realtime=False)
  try:
    mission.start_at(0.0, 0.0, 0.0)
    cap = int(4 * mm.REFINE_BUDGET_S / room_model.opt.timestep)

    def cannot_move(v, w):
      if (data.time - t0) / room_model.opt.timestep > cap:
        raise RuntimeError("refine_standoff is unbounded again (issue #339)")
      data.time += room_model.opt.timestep
      yield v, w
    monkeypatch.setattr(mission, "_nav_routine", cannot_move)
    monkeypatch.setattr(mission, "_drive_routine", lambda s, v, w: tick.result(None))
    monkeypatch.setattr(mission, "face_routine", lambda hd: tick.result(True))
    t0 = float(data.time)
    routine = mission.refine_standoff_routine(0.0, 0.3, 0.0)
    try:
      while True:
        next(routine)                           # no physics: nobody moves
    except StopIteration as done:
      lateral = done.value
    assert lateral == pytest.approx(0.3), "the robot never moved, and says so"
    assert data.time - t0 == pytest.approx(3 * mm.REFINE_BUDGET_S, abs=0.01)
  finally:
    mission.close()


@pytest.mark.slow
def test_full_hub_mission():
  """The milestone-8 claim, end to end: map the room, navigate to the rack,
  fine-align on the fiducials, pick the LCD module, carry it across the
  room, come back, and hang it up -- collision-free."""
  from pluggybot.mission.mission import run_demo
  result = run_demo(start=(0.5, 3.0, math.pi / 2))
  assert result["picked"], "never got the module off the rack"
  assert result["returned"], "never hung the module back up"
  assert result["collision_steps"] == 0, "the robot hit something"
