"""Guards for rack discovery (rack/localize.py, milestone 8)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.rack.coupling import RACK_HANG_X
from pluggybot.rack.localize import RackPose, RackSpotter

TAG_WORLD = (-0.9, 5.906)      # the rack tag's true world position


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path("models/room_hub.xml")


def _place(model, data, x, y, yaw):
  data.qpos[0] = x + 0.08 * math.cos(yaw)
  data.qpos[1] = y + 0.08 * math.sin(yaw)
  data.qpos[2] = 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  data.qvel[:] = 0
  mujoco.mj_forward(model, data)
  for _ in range(200):
    mujoco.mj_step(model, data)


def test_every_hub_marker_decodes_with_the_right_id(room_model):
  """Identity is the whole reason for real tags. From a vantage that sees
  the rack, every marker on it must decode as ITS id -- rack, both bays,
  the charge bay, and both modules. The colour stand-in could only guess
  which fiducial was which, and once guessed wrong badly enough to drag a
  module toward the wrong bay."""
  from pluggybot.rack.tags import (
    BAY_TAG_IDS, CHARGE_TAG_ID, MODULE_TAG_IDS, RACK_TAG_ID, RACK_TAG_SIZE,
    TagDetector,
  )
  data = mujoco.MjData(room_model)
  _place(room_model, data, -0.9, 4.0, math.pi / 2)
  det = TagDetector(room_model, "left_eye", tag_size=RACK_TAG_SIZE)
  try:
    found = det.detect(data)
  finally:
    det.close()
  expected = {RACK_TAG_ID, CHARGE_TAG_ID, *BAY_TAG_IDS, *MODULE_TAG_IDS.values()}
  assert expected <= set(found), f"missing tags: {sorted(expected - set(found))}"


def test_tag_pnp_range_is_accurate(room_model):
  """Range comes from the marker's own pose, not a depth buffer -- so it
  has to be right. This is what the terminal approach trusts instead of
  odometry, which had drifted 20 mm by the return leg."""
  from pluggybot.rack.tags import RACK_TAG_ID, RACK_TAG_SIZE, TagDetector
  data = mujoco.MjData(room_model)
  det = TagDetector(room_model, "left_eye", tag_size=RACK_TAG_SIZE)
  try:
    for standoff in (1.5, 2.5):
      _place(room_model, data, TAG_WORLD[0], TAG_WORLD[1] - standoff,
             math.pi / 2)
      found = det.detect(data)
      assert RACK_TAG_ID in found, f"rack tag not decoded at {standoff} m"
      cam = room_model.camera("left_eye").id
      cp = data.cam_xpos[cam]
      cm = data.cam_xmat[cam].reshape(3, 3)
      gp = data.geom_xpos[room_model.geom("rack_tag").id]
      true = float(-((gp - cp) @ cm)[2])
      assert abs(found[RACK_TAG_ID]["t"][2] - true) < 0.03, \
        f"PnP range off by {(found[RACK_TAG_ID]['t'][2] - true) * 1000:.0f} mm"
  finally:
    det.close()


def test_rack_tag_is_seen_and_projected(room_model):
  """From a clear vantage the tag must project to its true world position."""
  data = mujoco.MjData(room_model)
  x, y = -0.9, 3.4
  yaw = math.atan2(TAG_WORLD[1] - y, TAG_WORLD[0] - x)
  _place(room_model, data, x, y, yaw)
  spotter = RackSpotter(room_model)
  try:
    hits = spotter.spot(data, (x, y, yaw))
  finally:
    spotter.close()
  assert hits, "the rack tag was not detected from a clear 2.5 m vantage"
  hx, hy, hz, _px = hits[0]
  assert math.hypot(hx - TAG_WORLD[0], hy - TAG_WORLD[1]) < 0.10
  assert hz > 0.40, "the rack tag sits high on the mast"


def test_no_false_fiducials_from_room_furniture(room_model):
  """The room is full of pale plates (outlets, a light switch, a blank
  cover) and five other AprilTags. None may be reported as the RACK's
  marker -- a phantom rack is a robot that drives into a wall to swap
  tools."""
  data = mujoco.MjData(room_model)
  spotter = RackSpotter(room_model)
  rng = np.random.default_rng(0)
  false_hits = []
  try:
    for _ in range(12):
      x, y = rng.uniform(-1.6, 1.6), rng.uniform(0.0, 5.0)
      yaw = rng.uniform(-math.pi, math.pi)
      _place(room_model, data, x, y, yaw)
      for hx, hy, _hz, _px in spotter.spot(data, (x, y, yaw)):
        if math.hypot(hx - TAG_WORLD[0], hy - TAG_WORLD[1]) > 0.30:
          false_hits.append((round(hx, 2), round(hy, 2)))
  finally:
    spotter.close()
  assert not false_hits, f"non-rack fiducial detections: {false_hits}"


def test_rack_pose_round_trips_through_its_frame():
  prior = RackPose.prior()
  wx, wy = prior.to_world(RACK_HANG_X, 0.125)
  # a point on the hang plane must sit in front of the rack origin
  assert math.hypot(wx - prior.x, wy - prior.y) == pytest.approx(
    math.hypot(RACK_HANG_X, 0.125), abs=1e-9)
  # and from_tag must invert the tag's own offset
  tag_x, tag_y = prior.to_world(0.084, 0.0)
  back = RackPose.from_tag(tag_x, tag_y, prior.yaw)
  pos_err, yaw_err = back.error_against(prior)
  assert pos_err < 1e-9 and yaw_err < 1e-9


def test_discovery_corrects_a_wrong_prior(room_model):
  """The point of looking: a robot whose stored dock pose has gone stale
  (odometry drift, or someone nudged the rack) must fix it by observation
  rather than driving confidently to the wrong place."""
  from pluggybot.mission.mission import HubMission
  data = mujoco.MjData(room_model)
  stale = RackPose(RACK_ROOM_WRONG[0], RACK_ROOM_WRONG[1],
                   math.radians(-90.0))
  mission = HubMission(room_model, data, rack=stale)
  try:
    mission.start_at(-0.9, 4.2, math.pi / 2)
    mission.start_discovery()
    mission._spin()
    found = mission.refresh_rack()
  finally:
    mission.close()
  assert found is not None, "never confirmed the tag from a clear vantage"
  pos_err, yaw_err = found.error_against(RackPose.prior())
  assert pos_err < 0.10, f"discovered pose {pos_err * 1000:.0f} mm off truth"
  assert math.degrees(yaw_err) < 5.0
  stale_err, _ = stale.error_against(RackPose.prior())
  assert pos_err < stale_err, "discovery did not improve on the stale prior"


RACK_ROOM_WRONG = (-0.60, 5.99)   # 30 cm off: a prior that has gone stale


# ---- the facing fit (issue #88) --------------------------------------------

def _rotated_layout(yaw_deg, dx, dy, ids=None, jitter=None):
  """The rack's tag layout as an observer at (dx, dy, yaw) would decode it."""
  from pluggybot.rack.coupling import RACK_TAG_FACES
  c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
  rng = np.random.default_rng(88)
  out = {}
  for i in ids or RACK_TAG_FACES:
    lx, ly = RACK_TAG_FACES[i]
    x, y = dx + lx * c - ly * s, dy + lx * s + ly * c
    if jitter:
      x, y = x + rng.normal(0, jitter), y + rng.normal(0, jitter)
    out[i] = (x, y)
  return out


def test_the_facing_fit_recovers_the_rack_yaw_and_its_sign():
  """A rigid layout seen at +7 deg fits to +7 deg, not -7 -- the SIGN is
  the whole reason the fit exists: a single tag's PnP yaw is a coin flip
  between the two, square-on (issue #88)."""
  from pluggybot.rack.localize import fit_rack_facing
  for yaw in (7.0, -7.0, 0.0, 173.0):
    fit = fit_rack_facing(_rotated_layout(yaw, 1.3, -0.2))
    assert fit is not None
    err = math.degrees(math.atan2(math.sin(fit.yaw - math.radians(yaw)),
                                  math.cos(fit.yaw - math.radians(yaw))))
    assert abs(err) < 1e-9, (yaw, err)
    assert fit.rms < 1e-12


def test_the_facing_fit_needs_two_rack_tags_and_ignores_the_rest():
  """One point fixes no direction, and a MODULE's tag is not on the rack's
  plane (it may be on the fork), so a module tag beside one rack tag is
  still one rack tag."""
  from pluggybot.rack.coupling import RACK_TAG_FACES
  from pluggybot.rack.localize import fit_rack_facing
  from pluggybot.rack.tags import BAY_TAG_IDS, CHARGE_TAG_ID, MODULE_TAG_IDS
  seen = _rotated_layout(5.0, 1.0, 0.0, ids=[CHARGE_TAG_ID])
  assert fit_rack_facing(seen) is None
  seen[MODULE_TAG_IDS["module_lcd"]] = (0.9, 0.3)
  assert fit_rack_facing(seen) is None
  assert MODULE_TAG_IDS["module_lcd"] not in RACK_TAG_FACES
  # two rack tags at the bay pitch -- the charge standoff's whole view -- is
  # enough, and the extra id is ignored rather than fitted
  seen.update(_rotated_layout(5.0, 1.0, 0.0, ids=[BAY_TAG_IDS[1]]))
  fit = fit_rack_facing(seen)
  assert fit is not None and fit.n == 2
  assert abs(math.degrees(fit.yaw) - 5.0) < 1e-9


def test_a_millimetre_of_translation_noise_is_a_fraction_of_a_degree():
  """The fit's accuracy budget, stated: 1 mm of scatter on each decoded
  translation (what the dock camera measures) over the bay pitch and the
  rail must stay well inside the +/-3 deg that drops a module blind."""
  from pluggybot.rack.localize import fit_rack_facing
  from pluggybot.rack.tags import BAY_TAG_IDS, CHARGE_TAG_ID
  two = fit_rack_facing(_rotated_layout(0.0, 1.2, 0.0, jitter=0.001,
                                        ids=[CHARGE_TAG_ID, BAY_TAG_IDS[1]]))
  assert abs(math.degrees(two.yaw)) < 1.0
  assert two.rms < 0.003
  every = fit_rack_facing(_rotated_layout(0.0, 1.2, 0.0, jitter=0.001))
  assert abs(math.degrees(every.yaw)) < 0.3
