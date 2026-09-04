"""Guards for the charge-bay approach after a long shift (issue #32).

On a hosting-sized pack the robot works ~1000 sim-seconds between charges,
and some way into a run `go_charge` drove to where it believed the standoff
was, crept, touched nothing, and gave up -- ending the mission with the pack
at 7 % and the words "mission complete". The tool bays kept working in the
same runs, because their terminal creep is steered and ranged off the bay's
own tag; the charge bay had its own tag (the generator comment has always
said "the terminal servo needs a mark here like anywhere else") and nothing
reading it.

Measured envelope of the BLIND creep (scripts/charge_spike.py): ~6 cm of
lateral belief error, ~10 deg of heading -- and a 10 deg rack-yaw belief
error, which one badly conditioned free-space look delivers (the issue-78
lesson measured ~20 deg), swings the standoff sideways and tilts the
approach at once, killing it on both axes. The fix measures the standoff off
the charge tag's own PnP pose and verifies by the electrical criterion with
a backed-off retry, exactly as `swap_at_bay` always has.

The defect tests here pin the PREMISE: if the blind creep ever starts
forgiving these poses, the measured approach is redundant and this file
should say so (the test_navigation doctrine).
"""

import math

import pytest
import mujoco

from pluggybot.home import world as home
from pluggybot.rack.coupling import rack_charge_contact
from pluggybot.lifecycle import (
  CHARGE_APPROACH_MAX, CHARGE_CREEP, HubLifecycle,
)
from pluggybot.rack.localize import RackPose
from pluggybot.mission.mission import HubMission, charge_standoff
from pluggybot.rack.swap import align_lift

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))


def mission_with_belief_error(across, dyaw_deg):
  """Robot TRULY at the charge standoff + (across, dyaw), reckoner believing
  it is exactly at the standoff -- the state a long shift's drift delivers."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  sx, sy, hd = charge_standoff(TRUE_RACK)
  lx, ly = -math.sin(TRUE_RACK.yaw), math.cos(TRUE_RACK.yaw)
  tx, ty = sx + lx * across, sy + ly * across
  tyaw = hd + math.radians(dyaw_deg)
  d = data
  d.qpos[0] = tx + 0.08 * math.cos(tyaw)
  d.qpos[1] = ty + 0.08 * math.sin(tyaw)
  d.qpos[2] = 0.045
  d.qpos[3:7] = [math.cos(tyaw / 2), 0, 0, math.sin(tyaw / 2)]
  lift0 = align_lift()
  d.qpos[model.joint("lift_joint").qposadr[0]] = lift0
  d.ctrl[model.actuator("lift").id] = lift0
  d.ctrl[model.actuator("arm").id] = 0.0
  d.qpos[model.joint("arm_joint").qposadr[0]] = 0.0
  mujoco.mj_forward(model, d)
  r = mission.swap.reckoner
  r.x, r.y, r.theta = sx, sy, hd
  r.update(float(d.qpos[mission.swap.left_adr]),
           float(d.qpos[mission.swap.right_adr]))
  mission._drive(1.0, 0.0, 0.0)
  return mission, (sx, sy, hd)


#: The failure shape a rack-yaw belief error produces: the standoff swings
#: sideways by 0.42*sin(err) AND the approach heading tilts by err.
RACK_YAW_ERR_DEG = 10
RACK_YAW_ACROSS = 0.42 * math.sin(math.radians(RACK_YAW_ERR_DEG))


def blind_creep(mission, sx, sy, hd) -> bool:
  """What go_charge did before issue #32."""
  mission.face(hd)
  mission.refine_standoff(sx, sy, hd)
  mission.swap._drive_until(
    CHARGE_APPROACH_MAX, CHARGE_CREEP, stall_stop=True,
    stop_fn=lambda: rack_charge_contact(mission.model, mission.data))
  return rack_charge_contact(mission.model, mission.data)


#: SLOW because it cannot catch a regression in the fix -- it BYPASSES the
#: fix and asserts the old defect still reproduces. That premise needs
#: re-checking when the geometry moves, which is a merge, not every iterate
#: loop. Cost is why it was looked at; this is why it qualifies (issue #54).
@pytest.mark.slow
def test_the_blind_creep_misses_at_a_ten_degree_rack_yaw_error():
  """The defect, pinned: face + refine against a belief that reads perfect
  are no-ops, and the creep sails past the pins. If this starts passing,
  the measured approach has lost its premise -- find out why."""
  mission, (sx, sy, hd) = mission_with_belief_error(RACK_YAW_ACROSS,
                                                    RACK_YAW_ERR_DEG)
  try:
    assert not blind_creep(mission, sx, sy, hd)
  finally:
    mission.close()


def test_the_measured_approach_connects_where_the_blind_creep_missed():
  mission, _ = mission_with_belief_error(RACK_YAW_ACROSS, RACK_YAW_ERR_DEG)
  try:
    mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    assert rack_charge_contact(mission.model, mission.data)
  finally:
    mission.close()


def test_the_measured_approach_survives_the_issue_78_magnitude():
  """~20 deg is what one badly conditioned free-space look actually
  delivered (tests/test_rack_belief.py) -- the deep end of the envelope."""
  mission, _ = mission_with_belief_error(0.42 * math.sin(math.radians(20)),
                                         20)
  try:
    mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    assert rack_charge_contact(mission.model, mission.data)
  finally:
    mission.close()


def test_the_aligned_approach_still_connects():
  """Zero belief error -- every existing mission's case -- must be untouched
  by the measuring and the servo target."""
  mission, _ = mission_with_belief_error(0.0, 0.0)
  try:
    mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    assert rack_charge_contact(mission.model, mission.data)
  finally:
    mission.close()


def test_charge_bay_fix_measures_where_the_bay_actually_is():
  """The corrected standoff, expressed in the believed frame, must land
  where the TRUE standoff sits relative to the robot -- a few cm and a few
  degrees, from one look at a 30 mm tag at ~0.4 m. The pose keeps the tag
  in the dock camera's view -- a pose that loses it is the retry loop's
  territory, tested above."""
  across, dyaw = 0.06, -6
  mission, (sx, sy, hd) = mission_with_belief_error(across, dyaw)
  try:
    # what charge_approach does before its first look: bring the camera down
    # to tag height (from the align preset the tag is out of view entirely)
    from pluggybot.mission.mission import CHARGE_LOOK_LIFT
    mission.swap._run(1.5, 0.0, lift_target=CHARGE_LOOK_LIFT)
    fix = mission.charge_bay_fix()
    assert fix is not None, "the charge tag must decode from this pose"
    fx, fy, fhd = fix
    # where the true standoff sits in the believed frame: believed pose
    # composed with the robot's TRUE offset from the true standoff
    bx, by, bth = mission.pose
    tsx, tsy, tshd = charge_standoff(TRUE_RACK)
    # The TRUE pose off qpos, not the commanded one: the placement's settle
    # drive turns the robot ~2.6 deg (the reckoner follows, so the belief
    # error stays what was asked), and an expectation built on `hd + dyaw`
    # carried that as a 2.5 deg "measurement error" the old 5 deg bar hid.
    q = mission.data.qpos
    tyaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]),
                      1 - 2 * (q[5] * q[5] + q[6] * q[6]))
    tx = float(q[0]) - 0.08 * math.cos(tyaw)
    ty = float(q[1]) - 0.08 * math.sin(tyaw)
    # relative transform true->standoff, re-based onto the believed pose
    dxr = (tsx - tx) * math.cos(tyaw) + (tsy - ty) * math.sin(tyaw)
    dyr = -(tsx - tx) * math.sin(tyaw) + (tsy - ty) * math.cos(tyaw)
    ex = bx + dxr * math.cos(bth) - dyr * math.sin(bth)
    ey = by + dxr * math.sin(bth) + dyr * math.cos(bth)
    ehd = bth + (tshd - tyaw)
    # 0.02 m / 2 deg, from 0.04 / 5: measured 0.0016 m / 0.09 deg here with
    # the facing fitted over the two rack tags in view (issue #88), and the
    # one-tag arithmetic read the same to 0.2 mm -- at -6 deg the robot is
    # NOT square to the tag, so its PnP yaw was never ambiguous at this
    # pose. The coin flip lives at the bay standoffs, tests/test_swap_
    # approach.py; this pins that the charge bay's fit is as good.
    assert math.hypot(fx - ex, fy - ey) < 0.02
    assert abs(math.degrees(math.atan2(math.sin(fhd - ehd),
                                       math.cos(fhd - ehd)))) < 2.0
    assert mission.fix_source.startswith("plane:"), mission.fix_source
  finally:
    mission.close()


def test_one_rack_tag_in_view_falls_back_to_its_yaw_and_says_so():
  """The fit needs two rack tags (issue #88); with one, the fix reads that
  tag's own PnP yaw -- which is fine OFF square, where the solution is
  unambiguous, and is what a robot that arrived 10 deg off actually sees
  (measured: only the charge tag decodes from there). The fallback must
  still measure, and must LABEL itself, because a mission log that cannot
  tell a fitted facing from a coin toss cannot explain a drop."""
  across, dyaw = 0.06, -6
  mission, _ = mission_with_belief_error(across, dyaw)
  try:
    from pluggybot.mission.mission import CHARGE_LOOK_LIFT, CHARGE_TAG_ID
    mission.swap._run(1.5, 0.0, lift_target=CHARGE_LOOK_LIFT)
    real = mission.tags.detect
    mission.tags.detect = lambda data: {
      k: v for k, v in real(data).items() if k == CHARGE_TAG_ID}
    fix = mission.charge_bay_fix()
    assert fix is not None
    assert mission.fix_source == "yaw"
    both = real(mission.data)
    assert sum(1 for k in both if k != CHARGE_TAG_ID and k < 10) >= 1, \
        "the premise: a second rack tag IS in view for the fit to use"
    mission.tags.detect = real
    fitted = mission.charge_bay_fix()
    assert mission.fix_source.startswith("plane:")
    # off square the two agree to millimetres; square-on they would not
    assert math.hypot(fix[0] - fitted[0], fix[1] - fitted[1]) < 0.01
  finally:
    mission.close()


def test_a_failed_dock_is_not_narrated_as_mission_complete():
  """The other half of issue #32: `run()` used to break out of the loop on a
  failed go_charge and, because the battery was not empty, close the day
  with "mission complete". A robot that could not reach its charger has not
  completed anything."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, viewer=None, realtime=False,
                      battery_wh=0.7, errand=False,
                      low_battery_wh=10.0)     # needs_charge from step one
  life.go_charge = lambda: False               # ...and the dock unreachable
  result = life.run((0.5, 3.0, math.pi / 2), max_sim_time=60.0)
  assert result["stranded"] is True
  assert "mission complete" not in life.log[-1]
  assert "stranded" in life.log[-1]
