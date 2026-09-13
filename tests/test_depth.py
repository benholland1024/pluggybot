"""Guards for the near-field depth camera (perception/depth.py, issue #34).

Each pins one honest failure of a D435-class unit and fails without it:
the self-filter, axial depth through the nominal mount, z² noise, the
image-left occlusion shadow, out-of-range as UNKNOWN, and the mount's
floor band. No mission is flown; a settled robot and one frame each.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.perception import depth as depthmod
from pluggybot.perception.depth import DepthCamera
from pluggybot.perception.heightmap import HeightMap
from pluggybot.robot import FIRST, SECOND, world_with_robots


def _world(props=()):
  """room_hub, optionally with boxes `(x, y, size_xyz)` standing on the
  floor ahead of the robot (which starts at the origin facing +x)."""
  spec = mujoco.MjSpec.from_file("models/room_hub.xml")
  for i, (x, y, size) in enumerate(props):
    g = spec.worldbody.add_geom()
    g.name = f"prop_{i}"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [s / 2 for s in size]
    g.pos = [x, y, size[2] / 2]
  model = spec.compile()
  data = mujoco.MjData(model)
  for _ in range(300):
    mujoco.mj_step(model, data)
  return model, data


def _clean(model, **kw):
  """The geometric truth of a frame: no noise, no dropout."""
  return DepthCamera(model, noise_k=0.0, dropout=0.0, **kw)


@pytest.fixture(scope="module")
def room():
  return _world()


def test_the_frame_is_the_floor_ahead_and_the_deck_is_dropped(room):
  """The mount metric. From the mast top at 40°, the centre column sees the
  floor from just past the bumper (the deck shadows nearer) to ~2 m, and
  the deck itself -- chassis, LIDAR body, fork -- is dropped rather than
  reported as a 7-17 cm thing standing where the robot is. Without the
  self-filter the frame carries points inside the chassis footprint."""
  model, data = room
  cam = _clean(model)
  frame = cam.frame(data)
  assert 0.01 < frame.self_fraction < 0.15, frame.self_fraction
  near, far = cam.floor_band(data)
  assert 0.20 < near < 0.35, f"floor starts {near:.2f} m ahead of the axle"
  assert far > 1.8, f"floor ends {far:.2f} m ahead"
  deck = frame.points[(frame.points[:, 0] < 0.20) & (frame.points[:, 2] > 0.02)]
  assert len(deck) == 0, f"{len(deck)} points on the robot's own deck"


def test_the_floor_reconstructs_flat_through_the_nominal_mount(room):
  """The point comes back through the pixel's direction and the mount as
  the model states it -- the camera's rest pose relative to the axle on the
  floor. Get the mount wrong (the wheel radius, the pitch, the live pose)
  and a flat floor tilts, curves or sinks. Clean, every floor point within
  2 m sits on z = 0 to a millimetre."""
  model, data = room
  frame = _clean(model).frame(data)
  p = frame.points[(frame.points[:, 0] < 2.0) & (np.abs(frame.points[:, 1]) < 0.6)]
  floor = p                       # nothing stands there but floor
  assert len(floor) > 3000
  assert np.abs(floor[:, 2]).max() < 0.002, np.abs(floor[:, 2]).max()


def test_a_thing_on_the_floor_measures_its_height_and_place():
  """A 5 cm cube 0.9 m ahead: one clean frame into the height map finds
  ONE thing there, 50 mm tall to 3 mm, 30 mm in place."""
  model, data = _world(props=(((0.9 + 0.025, 0.0, (0.05, 0.05, 0.05)),)))
  frame = _clean(model).frame(data)
  hm = HeightMap()
  hm.update((-0.08, 0.0, 0.0), frame.points)   # the axle sits 8 cm behind
  things = [t for t in hm.things() if abs(t["y"]) < 0.5]   # not the rack
  assert len(things) == 1, things
  t = things[0]
  assert abs(t["height"] - 0.05) < 0.003, t
  assert abs(t["x"] - 0.925) < 0.03 and abs(t["y"]) < 0.03, t


def test_noise_grows_with_the_square_of_depth(room):
  """Stereo error is quadratic in z: a fixed sub-pixel disparity error.
  Against the clean frame, the residual at 2 m is ~4x the residual at 1 m
  (16x the variance), and at 1 m it is `NOISE_K` = 3.6 mm."""
  model, data = room
  truth = _clean(model).frame(data).z.reshape(-1)
  noisy = DepthCamera(model, dropout=0.0).frame(data).z.reshape(-1)
  ok = np.isfinite(truth) & np.isfinite(noisy)
  res = (noisy - truth)[ok]
  z = truth[ok]
  s1 = res[(z > 0.9) & (z < 1.1)].std()
  s2 = res[(z > 1.9) & (z < 2.1)].std()
  assert 0.0025 < s1 < 0.005, f"σ at 1 m {s1 * 1e3:.2f} mm"
  assert 3.0 < s2 / s1 < 5.5, f"σ(2 m)/σ(1 m) = {s2 / s1:.2f}"


def test_a_near_edge_shadows_the_background_on_its_image_left():
  """The right imager cannot see what a nearer surface hides. A 30 cm post
  0.6 m ahead against the floor beyond: the pixels just LEFT of it in the
  image are invalid, the pixels just right of it are not, and the band is
  f·B·(1/z_near - 1/z_far) wide. Without `_shadow` both sides are valid."""
  model, data = _world(props=(((0.7, 0.0, (0.1, 0.1, 0.3)),)))
  cam = _clean(model)
  z = cam.frame(data).z
  h, w = z.shape
  # the row through the post's TOP, where the background is the floor far
  # behind it (a post against floor at nearly its own depth casts a
  # sub-pixel shadow, correctly: the step has to exceed 1/(f·B) = 0.32/m)
  for r in range(h):
    row = z[r]
    near = np.flatnonzero(np.nan_to_num(row, nan=9) < 0.9)
    if len(near) and np.nanmedian(row) > 1.2:
      break
  else:
    pytest.fail("the post's top is not in the frame")
  a, b = near.min(), near.max()
  left = row[a - 4:a]
  right = row[b + 1:b + 5]
  assert np.isnan(left[-1]), "no shadow on the image-left of the post"
  assert not np.isnan(right).any(), f"shadow on the image-right: {right}"
  expected = cam.focal_px * cam.baseline * (1 / row[a] - 1 / np.nanmean(right))
  assert np.isnan(left).sum() >= math.floor(expected), (
    f"shadow {np.isnan(left).sum()} px, expected ~{expected:.1f}")


def test_out_of_range_is_unknown_not_free_and_the_limit_is_axial(room):
  """The LIDAR's rule inverted: a ray past `max_z` is a NaN pixel, never a
  point at max_z (which the map would take for floor at 3 m). The limit is
  on z ALONG THE AXIS, as the part's is: a corner pixel at z = 1 m is 1.6 m
  away by range and still valid. And MIN_Z does not bind on this mount: the
  nearest thing that is not the robot is past 0.5 m."""
  model, data = room
  cam = _clean(model, max_z=1.0)
  frame = cam.frame(data)
  z = frame.z[np.isfinite(frame.z)]
  assert z.max() <= 1.0
  assert not np.any(np.isclose(z, 1.0, atol=1e-6)), "a pixel clipped to max"
  assert len(frame.points) < 0.7 * frame.z.size, "far pixels became points"
  ranges = np.linalg.norm(frame.points - cam.origin_robot, axis=1)
  assert ranges.max() > 1.2, f"the cut is on range ({ranges.max():.2f} m)"
  full = _clean(model).frame(data)
  assert np.nanmin(full.z) > 0.45, np.nanmin(full.z)
  assert depthmod.MIN_Z < np.nanmin(full.z)


def test_a_second_robot_rides_the_handle_and_is_filtered_like_the_first():
  """The second robot's camera is `r2_depth_eye` on `r2_pluggybot`, and
  `exclude_robot` drops the FIRST robot from its frame as the LIDAR does --
  another robot is no information about the floor."""
  model = world_with_robots("models/room_hub.xml", second_at=(-1.0, 0.0))
  data = mujoco.MjData(model)
  for _ in range(300):
    mujoco.mj_step(model, data)
  cam = _clean(model, handle=SECOND)
  assert cam.camera_name == "r2_depth_eye"
  seen = cam.frame(data)
  tall = seen.points[(seen.points[:, 2] > 0.05) & (seen.points[:, 0] < 1.5)]
  assert len(tall) > 20, "the first robot should stand in the second's frame"
  cam.exclude_robot(FIRST.root)
  hidden = cam.frame(data)
  tall = hidden.points[(hidden.points[:, 2] > 0.05) & (hidden.points[:, 0] < 1.5)]
  assert len(tall) == 0, f"{len(tall)} points of the other robot remain"
