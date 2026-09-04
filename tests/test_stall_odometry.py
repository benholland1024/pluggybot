"""Guards for the bumper rule on dead reckoning (issue #94).

Odometry integrates wheel rotation, and a wheel slipping against something
immovable rotates just the same. `HubSwap.pinned` guarded the one press the
code DECLARED (the charge cycle, 828 mm pumped before it); an ordinary drive
that stalls -- an obstacle, a wall corner, a wedged approach -- had no guard
at all and is unbounded: measured, 30 s against the garden fence pumped
4.28 m of travel that never happened, and `drive_to` then reported arrival
at a point the robot never reached.

`HubSwap.pressing` is the same rule sensed rather than declared: a chassis
contact on the side the wheels are driving toward holds the reckoner's
travel. The premise test pins the defect with the rule bypassed (the
test_navigation doctrine); sweep tables in scripts/stall_spike.py.
"""

import math

import mujoco
import pytest

from pluggybot.behavior.navigation import drive_toward
from pluggybot.home import world as home
from pluggybot.lifecycle import CHARGE_APPROACH_MAX, CHARGE_CREEP, CHARGE_PRESS
from pluggybot.mission.mission import HubMission, charge_standoff
from pluggybot.rack.coupling import HUB_STATION_YS, rack_charge_contact
from pluggybot.rack.localize import RackPose
from pluggybot.rack.swap import press_opposes_drive

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))
#: Garden, facing the east fence (x = 10) on a line clear of the gate.
FENCE_START, BEYOND_FENCE = (9.0, 0.0, 0.0), (14.0, 0.0)
#: A box the scan plane (0.223 m) looks straight over and the bumper
#: (chassis 0.06-0.12 m) cannot: knee height to this robot.
LOW_BOX_POS, LOW_BOX_SIZE = (9.0, 0.0, 0.075), (0.15, 0.4, 0.075)


def true_axle(data):
  q = data.qpos
  th = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]),
                  1 - 2 * (q[5] * q[5] + q[6] * q[6]))
  return q[0] - 0.08 * math.cos(th), q[1] - 0.08 * math.sin(th), th


def home_mission(low_box=False):
  if low_box:
    spec = mujoco.MjSpec.from_file("models/home_world.xml")
    g = spec.worldbody.add_geom()
    g.name, g.type = "low_box", mujoco.mjtGeom.mjGEOM_BOX
    g.size, g.pos = list(LOW_BOX_SIZE), list(LOW_BOX_POS)
    model = spec.compile()
  else:
    model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  return HubMission(model, data, viewer=None, realtime=False,
                    rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)


def press_the_fence(mission, seconds=30.0) -> tuple[float, float]:
  """The issue's measurement: the navigation law driven at a point beyond
  the fence, no mission loop around it. Returns (true, believed) travel."""
  mission.start_at(*FENCE_START)
  tx0, ty0, _ = true_axle(mission.data)
  bx0, by0, _ = mission.pose
  t0 = mission.data.time
  while mission.data.time - t0 < seconds:
    v, w = drive_toward(mission.pose, BEYOND_FENCE)
    mission._drive(mission.model.opt.timestep, v, w)
  tx, ty, _ = true_axle(mission.data)
  bx, by, _ = mission.pose
  return math.hypot(tx - tx0, ty - ty0), math.hypot(bx - bx0, by - by0)


def test_a_press_is_a_contact_on_the_side_the_wheels_drive_toward():
  """The rule, as a truth table: ahead + forward and behind + reverse are
  presses; a contact on the far side is a scrape the robot is leaving, and
  a pure spin has no travel to hold."""
  assert press_opposes_drive(+0.12, +5.0)
  assert press_opposes_drive(-0.12, -5.0)
  assert not press_opposes_drive(+0.12, -5.0)
  assert not press_opposes_drive(-0.12, +5.0)
  assert not press_opposes_drive(+0.12, 0.0)


def test_a_stalled_drive_does_not_pump_dead_reckoning():
  """30 s into the fence: the belief must not walk away from the robot.
  Blind it walked 4.28 m (scripts/stall_spike.py --blind)."""
  mission = home_mission()
  try:
    true_m, believed_m = press_the_fence(mission)
    # the fixture: a metre from the fence, aimed 5 m past it -- the robot
    # must have stopped AT the fence (measured 0.79 m), or nothing pressed
    assert 0.5 < true_m < 1.0, true_m
    # measured -0.004 m; blind, +4.28
    assert abs(believed_m - true_m) < 0.05, (true_m, believed_m)
    assert mission.swap.press_steps > 1000
  finally:
    mission.close()


#: SLOW because it cannot catch a regression in the fix -- it BYPASSES the
#: rule and asserts the old defect still reproduces (issue #54's premise
#: rule). If the pump ever stops without the rule, find out what else is
#: holding the reckoner.
@pytest.mark.slow
def test_a_stalled_drive_still_pumps_without_the_bumper_rule(monkeypatch):
  from pluggybot.rack.swap import HubSwap
  monkeypatch.setattr(HubSwap, "_pressing", lambda self: False)
  mission = home_mission()
  try:
    true_m, believed_m = press_the_fence(mission)
    assert believed_m - true_m > 2.0, (true_m, believed_m)
  finally:
    mission.close()


def test_drive_to_does_not_arrive_at_a_point_it_never_reached():
  """A point behind a knee-high box the lidar looks over. Blind, the belief
  pumps past the box and `drive_to` returns True with the robot a metre
  short -- the lie every downstream frame then inherits. With the rule the
  belief stays with the robot, and either the drive fails honestly or the
  robot is truly there."""
  mission = home_mission(low_box=True)
  try:
    mission.start_at(8.0, 0.0, 0.0)
    mission._spin()
    arrived = mission.drive_to(10.0, 0.0, timeout=60.0)
    tx, ty, _ = true_axle(mission.data)
    bx, by, _ = mission.pose
    # Measured 0.07 m: twelve seconds of bumping, backing off and bumping
    # again skids the chassis sideways a little, and wheels cannot see
    # lateral motion. Blind, the error was 1.30 m AND the drive said True.
    assert math.hypot(bx - tx, by - ty) < 0.15, (tx, ty, bx, by)
    if arrived:
      assert math.hypot(tx - 10.0, ty) < 0.15
    assert mission.swap.press_steps > 0, "the fixture: the box was hit"
  finally:
    mission.close()


def test_the_bumper_rule_is_quiet_through_a_swap_cycle():
  """A rule that fired on a contact that is not a press would freeze REAL
  travel. Measured zero chassis contacts through pick, carry and return; this
  keeps it so -- a module on the fork, the lean-pad, a tray never touch the
  chassis, and if one starts to, this is where it shows."""
  mission = home_mission()
  try:
    st = HUB_STATION_YS[0]
    sx, sy, hd = TRUE_RACK.bay_standoff(st, 1.2, 0.05)
    mission.start_at(sx, sy, hd)
    mission.swap_at_bay(st, "pick", module="module_lcd")
    assert mission.swap.module_state("module_lcd")["on_fork"]
    mission.drive_to(sx - 1.5, sy + 1.0, timeout=40.0)
    mission.face(hd)
    mission.swap_at_bay(st, "return", module="module_lcd")
    assert mission.swap.module_state("module_lcd")["hung"]
    assert mission.swap.press_steps == 0
  finally:
    mission.close()


def test_the_charge_press_pins_itself():
  """The 828 mm lesson (issue #22) caught by the sensed rule alone: nobody
  sets `pinned`, the robot holds CHARGE_PRESS against the pins, and the
  belief stays put. The explicit flag in `charge()` is now belt and braces."""
  mission = home_mission()
  try:
    sx, sy, hd = charge_standoff(TRUE_RACK)
    mission.start_at(sx, sy, hd)
    mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    assert rack_charge_contact(mission.model, mission.data)
    assert not mission.swap.pinned
    bx0, by0, _ = mission.pose
    mission._drive(10.0, CHARGE_PRESS, 0.0)
    bx, by, _ = mission.pose
    assert math.hypot(bx - bx0, by - by0) < 0.01
    assert mission.swap.press_steps > 0
  finally:
    mission.close()
