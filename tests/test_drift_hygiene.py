"""Guards for drift hygiene (issue #42).

Dead reckoning is never otherwise corrected, and the measured record of what
unbounded drift does is long: 0.69 m over three cycles (pre-#32 diagnostics),
the charge approach walking out of its ~6 cm envelope (#32), modules dragged
on to the floor at 4 cm of decoherence (#30), and the pen lost at t=7372 of
the issue-30 validation run when even the MEASURED bay approach could not
find a tag the drift had put outside the camera's whole field.

Three fixes, one mechanism -- keep the believed frame coherent with the one
the robot navigates in:

  ANCHOR AT THE DOCK. With both pogo pins conducting the robot holds the one
  pose it ever occupies to millimetres by construction, so every charge
  zeroes the shift's accumulated drift (`HubMission.anchor_at_dock`).

  MERGE THE RACK BY IDENTITY. Its sightings carry a decoded ID, so gating
  them by the outlet store's 0.4 m distance gate let a decohered spin spawn
  a second landmark the stale one outvoted.

  WEIGHT THE RACK BELIEF FOR RECENCY. A mission-long running average
  remembers the MEAN historical frame; an EMA (`RACK_RECENCY`) follows the
  frame the robot is actually navigating in, which is what makes a recovery
  spin able to move the belief at all.

The recovery chain the three add up to is pinned end-to-end by
tests/test_swap_approach.py::test_the_recovery_finds_a_bay_the_first_look_lost.
"""

import math

import mujoco

from pluggybot.home import world as home
from pluggybot.rack.coupling import CHARGE_BAY_Y, rack_charge_contact
from pluggybot.lifecycle import CHARGE_APPROACH_MAX, CHARGE_CREEP
from pluggybot.rack.localize import RACK_RECENCY, RackFinder, RackPose
from pluggybot.mission.mission import (
  CHARGE_PIN_X, DOCK_ALONG, HubMission, charge_standoff,
)
from pluggybot.mapping.landmarks import LandmarkStore

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))


def true_axle(model, data):
  qw, qx, qy, qz = data.qpos[3:7]
  yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  return (float(data.qpos[0]) - 0.08 * math.cos(yaw),
          float(data.qpos[1]) - 0.08 * math.sin(yaw), yaw)


def test_the_anchor_is_the_dock_pose_of_the_commissioned_rack():
  """The arithmetic, bare: wherever dead reckoning AND the rack belief had
  wandered, the anchor puts the robot at DOCK_ALONG out from the PRIOR's pin
  line, facing in, and the belief back on the prior with it. Measured why
  (issue #42's acceptance run): anchored to the belief instead, the pose
  tracked a belief that drifted 0.003 -> 0.344 m over four sim-hours."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  try:
    rec = mission.swap.reckoner
    rec.x, rec.y, rec.theta = 4.2, -3.7, 1.234    # anywhere at all
    mission.rack = RackPose(TRUE_RACK.x + 0.3, TRUE_RACK.y - 0.2,
                            TRUE_RACK.yaw + 0.1)   # ...and a drifted belief
    mission.anchor_at_dock()
    assert mission.rack == TRUE_RACK
    bx, by = TRUE_RACK.to_world(CHARGE_PIN_X, CHARGE_BAY_Y)
    nx, ny = math.cos(TRUE_RACK.yaw), math.sin(TRUE_RACK.yaw)
    assert rec.x == bx + nx * DOCK_ALONG
    assert rec.y == by + ny * DOCK_ALONG
    assert rec.theta == TRUE_RACK.heading
  finally:
    mission.close()


def test_a_charge_kills_the_shifts_accumulated_drift():
  """The acceptance criterion, on physics: 0.3 m of decoherence rides the
  robot all the way ONTO the pins (the measured approach absorbs it), the
  believed pose is still 0.3 m of fiction at contact -- the premise, pinned
  -- and the anchor collapses it to the millimetres the dock is good for."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  try:
    sx, sy, hd = charge_standoff(TRUE_RACK)
    mission.start_at(sx, sy, hd)
    rec = mission.swap.reckoner
    lx, ly = -math.sin(TRUE_RACK.yaw), math.cos(TRUE_RACK.yaw)
    rec.x += lx * 0.15
    rec.y += ly * 0.15
    assert mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    assert rack_charge_contact(model, data)
    tx, ty, _ = true_axle(model, data)
    before = math.hypot(rec.x - tx, rec.y - ty)
    assert before > 0.10, \
        "the premise: contact does not correct the reckoner by itself"
    mission.anchor_at_dock()
    after = math.hypot(rec.x - tx, rec.y - ty)
    assert after < 0.01, f"anchored belief still {after * 1000:.1f} mm out"
  finally:
    mission.close()


class OneSpotter:
  """A spotter that reports the tag wherever the test says it is."""

  def __init__(self):
    self.at = (0.5, -1.98)

  def spot(self, data, pose):
    return [(self.at[0], self.at[1], 0.4, 1)]

  def close(self):
    pass


def test_the_rack_merges_by_identity_however_far_the_frame_drifted():
  """The defect, pinned from the other side: sightings half a metre apart
  are OUTSIDE the outlet store's 0.4 m gate, and used to spawn a second
  landmark the stale one outvoted -- the recovery spin that existed to fix
  the belief could not touch it. A decoded ID is identity; there is exactly
  one rack landmark, wherever the frame puts its sightings."""
  finder = RackFinder.__new__(RackFinder)
  finder.spotter = OneSpotter()
  finder.landmarks = LandmarkStore(recency=RACK_RECENCY)
  finder.facing = None
  for _ in range(50):
    finder.look(None, (0.5, -1.0, 0.0))
  finder.spotter.at = (1.0, -1.98)          # the frame drifted half a metre
  for _ in range(10):
    finder.look(None, (1.0, -1.0, 0.0))
  assert len(finder.landmarks.landmarks) == 1
  lm = finder.landmarks.landmarks[0]
  # ...and the EMA has carried the belief most of the way to the new frame:
  # ten sightings at RACK_RECENCY leave (1 - w)^10 ~ 6 % of the residual.
  assert abs(lm.x - 1.0) < 0.5 * (1 - RACK_RECENCY) ** 10 + 0.01


def test_the_outlet_landmarks_keep_the_pure_average():
  """The outlet map is surveyed once on a short clock and its estimator is
  the running mean -- the recency floor must not leak into it."""
  store = LandmarkStore()
  for _ in range(99):
    store.add_sighting(1.0, 0.0, 0.4, seen_from=(0.0, 0.0))
  store.add_sighting(1.3, 0.0, 0.4, seen_from=(0.0, 0.0))
  assert len(store.landmarks) == 1
  assert store.landmarks[0].x == (99 * 1.0 + 1.3) / 100
