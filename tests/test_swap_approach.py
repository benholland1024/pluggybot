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


def true_axle(mission) -> tuple[float, float, float]:
  """The robot's TRUE axle pose off qpos (the chassis origin rides 8 cm
  ahead of it)."""
  q = mission.data.qpos
  w, x, y, z = q[3], q[4], q[5], q[6]
  th = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return q[0] - 0.08 * math.cos(th), q[1] - 0.08 * math.sin(th), th


def fix_error(mission, fix, true_standoff) -> tuple[float, float]:
  """(m, deg) between a believed-frame fix and the TRUE standoff, the fix
  re-based from the believed pose on to the true one -- so this is the
  MEASUREMENT's error with none of the belief's in it (the same arithmetic
  as scripts/swap_spike.py --yaw)."""
  fx, fy, fhd = fix
  bx, by, bth = mission.pose
  tx, ty, tth = true_axle(mission)
  dx, dy = fx - bx, fy - by
  rx = dx * math.cos(bth) + dy * math.sin(bth)
  ry = -dx * math.sin(bth) + dy * math.cos(bth)
  px = tx + rx * math.cos(tth) - ry * math.sin(tth)
  py = ty + rx * math.sin(tth) + ry * math.cos(tth)
  tsx, tsy, tshd = true_standoff
  dh = (tth + fhd - bth) - tshd
  return (math.hypot(px - tsx, py - tsy),
          abs(math.degrees(math.atan2(math.sin(dh), math.cos(dh)))))


def nudge_truth(mission, q0, v0, metres: float) -> None:
  """Move the TRUE robot -- and the module on its fork -- across the
  approach by `metres`, from the saved state, belief untouched."""
  model, data = mission.model, mission.data
  data.qpos[:] = q0
  data.qvel[:] = v0
  for body in (model.geom("chassis").bodyid[0], model.body(MODULE).id):
    adr = model.jnt_qposadr[model.body(body).jntadr[0]]
    data.qpos[adr] += -math.sin(TRUE_RACK.yaw) * metres
    data.qpos[adr + 1] += math.cos(TRUE_RACK.yaw) * metres
  mujoco.mj_forward(model, data)


#: The issue-88 sweep: thirteen true poses 2 mm apart across the approach.
SWEEP_MM = range(0, 26, 2)
#: What the measured standoff must hold at EVERY one of them. Measured
#: (scripts/swap_spike.py --yaw): 0.0075 m / 0.71 deg worst with the
#: facing fitted over the rack's tags; 0.016-0.066 m / 2.0-8.4 deg with it
#: read off the bay tag's own PnP yaw, over a bar at 12 of the 13. So these
#: fail without the fit, and they still catch the ~0.10 m PLUG_LATERAL sign
#: slip the single-pose test below was written for.
FIX_BAR_M, FIX_BAR_DEG = 0.02, 2.0


#: SLOW because it cannot catch a regression in the fix -- it BYPASSES the
#: fix and asserts the old defect still reproduces. That premise needs
#: re-checking when the geometry moves, which is a merge, not every iterate
#: loop. Cost is why it was looked at; this is why it qualifies (issue #54).
@pytest.mark.slow
def test_a_blind_return_at_the_cliff_fails_to_hang_the_module():
  """The defect, pinned: with the measurement bypassed, six centimetres of
  belief decoherence loses the module. If this starts passing, bay_fix has
  lost its premise -- find out why.

  ⚠ WHICH FAILURE is deliberately NOT pinned any more (issue #88). It used
  to also assert the module ended up on the FLOOR, which was true and useful
  right up until it wasn't: the tag's decoded yaw is ambiguous square-on to
  the bay and flips sign under 2 mm of pose difference, so at this
  decoherence the blind return lands on either side of a coin toss --
  dragged to the floor, or left on the fork. This file's own docstring
  already records fork-retention as what 10 cm does, so both are the same
  defect wearing different clothes.

  What must not change is that the blind return FAILS. Pinning the mode as
  well made this a test of the coin rather than of the premise -- it went red
  when issue #68 moved a wall on the other side of the house, which changed
  the pick trajectory by a fraction of a millimetre.
  """
  mission = carrying_mission()
  try:
    mission.bay_fix = lambda *a, **k: None      # what swap_at_bay did before
    decohere(mission, DROP_ACROSS)
    mission.swap_at_bay(STATION, "return", module=MODULE)
    st = mission.swap.module_state(MODULE)
    assert not st["hung"], \
        "the blind return hung the module: bay_fix's premise is gone"
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
  robot, re-expressed in the decohered believed frame.

  Bars: 0.02 m and 2 deg. They were 0.04 / 5 until issue #68 moved a wall
  on the far side of the house and this went red at 0.0561 -- and 0.07 / 10
  from then until issue #88, which found why: the bay tag's PnP yaw is a
  coin flip square-on, and the fixture had been landing on the winning
  side. With the facing fitted over the rack's tags instead, the residual
  here is ~0.009 m at every one of 13 poses 2 mm apart (the sweep test
  below), so the bars are set from that with headroom.
  """
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
    assert math.hypot(fx - ex, fy - ey) < FIX_BAR_M
    assert abs(math.degrees(math.atan2(math.sin(fhd - tshd),
                                       math.cos(fhd - tshd)))) < FIX_BAR_DEG
    assert mission.fix_source.startswith("plane:"), mission.fix_source
  finally:
    mission.close()


def test_the_fix_holds_across_two_millimetre_nudges_of_the_robot():
  """Issue #88, the regression test: the measured standoff must be a
  MEASUREMENT, not a toss. The robot is moved 2 mm at a time across the
  approach -- truth only, belief untouched, one render per pose -- and the
  fix's error against the true standoff must stay under the bar at every
  pose. A single tag's PnP yaw fails this at 12 of 13 (0.016-0.066 m),
  bimodally: the solver picks the mirrored branch on pixel noise, and 2 mm
  of pose is enough to re-roll it."""
  from pluggybot.rack.swap import PLUG_LATERAL, STANDOFF
  mission = carrying_mission()
  try:
    q0, v0 = mission.data.qpos.copy(), mission.data.qvel.copy()
    truth = TRUE_RACK.bay_standoff(STATION, STANDOFF, PLUG_LATERAL)
    worst = (0.0, 0.0)
    for mm in SWEEP_MM:
      nudge_truth(mission, q0, v0, mm / 1000)
      fix = mission.bay_fix(STATION)
      assert fix is not None, f"the bay tag must decode at +{mm} mm"
      assert mission.fix_source.startswith("plane:"), mission.fix_source
      err_m, err_deg = fix_error(mission, fix, truth)
      worst = max(worst, (err_m, err_deg))
      assert err_m < FIX_BAR_M and err_deg < FIX_BAR_DEG, \
          f"+{mm} mm: {err_m:.4f} m / {err_deg:.2f} deg ({mission.fix_source})"
  finally:
    mission.close()


#: SLOW because it cannot catch a regression in the fix -- it reads the
#: detector's raw yaw and asserts the OLD defect still reproduces (the
#: premise-pinning rule, issue #54). If this starts failing, the solver has
#: stopped flipping and the fit may be redundant -- find out why.
@pytest.mark.slow
def test_one_tags_yaw_is_still_a_coin_flip_square_on():
  """The premise, pinned: over the same 2 mm sweep, with the true heading
  never changing, the bay tag's own decoded yaw must still swing by more
  than the +/-3 deg that drops a module blind. Measured -7.5..+7.0 deg."""
  from pluggybot.rack.coupling import bay_tag_id
  mission = carrying_mission()
  try:
    q0, v0 = mission.data.qpos.copy(), mission.data.qvel.copy()
    yaws = []
    for mm in SWEEP_MM:
      nudge_truth(mission, q0, v0, mm / 1000)
      det = mission.tags.detect(mission.data).get(bay_tag_id(STATION))
      assert det is not None
      yaws.append(math.degrees(det["yaw"]))
    assert max(yaws) - min(yaws) > 6.0, yaws
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
