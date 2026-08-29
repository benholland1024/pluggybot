"""Guards for the bay approach under belief error (issue #30).

The long-run tool drop: swaps run flawlessly until accumulated drift
decoheres the believed frame from the world by ~4 cm (or a few degrees of
heading), and then a return's peg misses the tray V's +/-8 mm window and the
retreat drags the module off the trays and on to the floor at the rack's
foot -- where every later pick finds an empty bay and the approach lane is
littered (one run ended `GO_CHARGE: no route to the charge bay` because of
it). Reproduced in two 4+ sim-hour runs, one with ZERO model calls, so it is
physics and not policy.

The fix is `HubMission.bay_fix` -- the #32 measured-standoff medicine one
bay over: measure the standoff off the bay's own tag before lining up,
instead of trusting a believed pose that inherits the whole shift's drift.
The defect tests pin the PREMISE (the test_navigation doctrine): if the
blind approach ever starts forgiving these poses, the measurement is
redundant and this file should say so.

Sweep tables: scripts/swap_spike.py (`--blind` for the before-fix rows).
"""

import math

import pytest
import mujoco

from pluggybot.home import world as home
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.localize import RackPose
from pluggybot.mission.mission import HubMission

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))
MODULE = "module_lcd"
STATION = HUB_STATION_YS[0]
#: The measured cliff: 4 cm decoheres a blind return into a floor drop
#: (2 cm still hangs; 10 cm misses the trays and keeps the module on the
#: fork, which is the less destructive failure).
DROP_ACROSS = 0.06


def carrying_mission():
  """A robot that has cleanly picked the LCD and driven off a little."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  sx, sy, hd = TRUE_RACK.bay_standoff(STATION, 1.2, 0.05)
  mission.start_at(sx, sy, hd)
  mission.swap_at_bay(STATION, "pick", module=MODULE)
  assert mission.swap.module_state(MODULE)["on_fork"], \
      "the clean pick is the fixture, not the test"
  mission.swap._drive_until(0.6, -0.12, stall_stop=False)
  return mission


def decohere(mission, across, dyaw_deg=0.0):
  r = mission.swap.reckoner
  r.x += -math.sin(TRUE_RACK.yaw) * across
  r.y += math.cos(TRUE_RACK.yaw) * across
  r.theta += math.radians(dyaw_deg)


def module_z(mission) -> float:
  return float(mission.data.xpos[int(mission.model.body(MODULE).id)][2])


#: SLOW because it cannot catch a regression in the fix -- it BYPASSES the
#: fix and asserts the old defect still reproduces. That premise needs
#: re-checking when the geometry moves, which is a merge, not every iterate
#: loop. Cost is why it was looked at; this is why it qualifies (issue #54).
@pytest.mark.slow
def test_a_blind_return_at_the_cliff_drops_the_module_on_the_floor():
  """The defect, pinned: with the measurement bypassed, six centimetres of
  belief decoherence puts the module on the floor at the rack's foot. If
  this starts passing, bay_fix has lost its premise -- find out why."""
  mission = carrying_mission()
  try:
    mission.bay_fix = lambda *a, **k: None      # what swap_at_bay did before
    decohere(mission, DROP_ACROSS)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    st = mission.swap.module_state(MODULE)
    assert not st["hung"]
    assert module_z(mission) < 0.10, \
        "expected the retreat to drag the module on to the floor"
  finally:
    mission.close()


def test_the_measured_return_hangs_the_module_where_the_blind_one_dropped_it():
  mission = carrying_mission()
  try:
    decohere(mission, DROP_ACROSS)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    assert mission.swap.module_state(MODULE)["hung"]
  finally:
    mission.close()


def test_the_measured_return_survives_a_heading_decoherence_too():
  """-3 deg of heading error alone dropped a module blind (swap_spike) --
  the smallest measured kill, so the row most worth pinning green."""
  mission = carrying_mission()
  try:
    decohere(mission, 0.0, dyaw_deg=-3.0)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    assert mission.swap.module_state(MODULE)["hung"]
  finally:
    mission.close()


def test_a_measured_pick_lifts_a_module_a_blind_one_knocked_to_the_floor():
  """Pick-side cliff: blind at 8 cm the fork never captures the peg and the
  attempt knocks the module off its bracket. Measured, the whole cycle --
  pick, carry, return -- completes from the same decohered belief."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  try:
    sx, sy, hd = TRUE_RACK.bay_standoff(STATION, 1.2, 0.05)
    mission.start_at(sx, sy, hd)
    decohere(mission, 0.08)
    mission.swap_at_bay(STATION, "pick", module=MODULE)
    assert mission.swap.module_state(MODULE)["on_fork"]
    mission.swap._drive_until(0.6, -0.12, stall_stop=False)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    assert mission.swap.module_state(MODULE)["hung"]
  finally:
    mission.close()


def test_bay_fix_measures_the_standoff_the_bay_is_actually_at():
  """The lateral half of the arithmetic, guarded on its own: the fork line
  rides PLUG_LATERAL right of the chassis, and a sign slip there would park
  every approach 10 cm off -- plausible-looking and always wrong. The
  measured standoff must land where the TRUE standoff sits relative to the
  robot, re-expressed in the decohered believed frame."""
  mission = carrying_mission()
  try:
    across = 0.05
    decohere(mission, across)
    fix = mission.bay_fix(STATION)
    assert fix is not None, "the bay tag must decode from this pose"
    fx, fy, fhd = fix
    from pluggybot.rack.swap import PLUG_LATERAL, STANDOFF
    tsx, tsy, tshd = TRUE_RACK.bay_standoff(STATION, STANDOFF, PLUG_LATERAL)
    # the believed frame is the true frame shifted by `across` along the
    # rack's local +y, so the expected fix is the true standoff plus that
    # same shift
    ex = tsx + -math.sin(TRUE_RACK.yaw) * across
    ey = tsy + math.cos(TRUE_RACK.yaw) * across
    assert math.hypot(fx - ex, fy - ey) < 0.04
    assert abs(math.degrees(math.atan2(math.sin(fhd - tshd),
                                       math.cos(fhd - tshd)))) < 5.0
  finally:
    mission.close()


def test_half_a_metre_of_drift_blinds_the_first_look():
  """The premise of the recovery branch (issue #30's validation run): at
  0.5 m of decoherence the bay tag sits outside the dock camera's whole
  field, so `bay_fix` answers None -- and the pre-recovery code then trusted
  the believed standoff, which is the old cliff through the gap. If this
  starts decoding, the recovery has lost its premise."""
  mission = carrying_mission()
  try:
    from pluggybot.mission.mission import bay_standoff
    decohere(mission, 0.5)
    sx, sy, hd = bay_standoff(STATION, mission.rack)
    mission.drive_to(sx, sy, timeout=30.0)
    mission.face(hd)
    assert mission.bay_fix(STATION) is None
  finally:
    mission.close()


def test_the_recovery_finds_a_bay_the_first_look_lost():
  """The whole chain, on physics: a robot that mapped the rack and sighted
  its tag, decohered by half a metre mid-carry, must still hang the module.
  The first look answers None (above); the recovery spins, re-sights the
  rack tag from the decohered frame, and the refreshed belief pulls the
  standoff close enough for the second look to decode. The charge bay had
  exactly this recovery and docked 6/6 in the run that lost the pen."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  try:
    sx, sy, hd = TRUE_RACK.bay_standoff(STATION, 1.2, 0.05)
    mission.start_at(sx, sy, hd)
    mission.start_discovery()
    mission._spin()                       # seed the map and the tag sightings
    mission.swap_at_bay(STATION, "pick", module=MODULE)
    assert mission.swap.module_state(MODULE)["on_fork"]
    mission.swap._drive_until(0.6, -0.12, stall_stop=False)
    decohere(mission, 0.5)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    assert mission.swap.module_state(MODULE)["hung"]
  finally:
    mission.close()
