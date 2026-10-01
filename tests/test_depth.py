"""Guards for the near-field depth camera (perception/depth.py, issue #34),
on the quadruped's nose.

Each pins one honest failure of a D435-class unit and fails without it:
a thing's height and place through the mount, z² noise, the image-left
occlusion shadow, out-of-range as UNKNOWN, and the other robot filtered.
No mission is flown; a standing robot and one frame each. (That the floor
reconstructs through the body's own attitude is `test_map_scan.py`.)
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.legs import model as qm
from pluggybot.perception import depth as depthmod
from pluggybot.perception.depth import DepthCamera
from pluggybot.perception.heightmap import HeightMap
from pluggybot.robot import FIRST, SECOND


def _world(props=()):
  """The quadruped standing at the origin facing +x on a bare floor,
  optionally with boxes `(x, y, size_xyz)` standing ahead of it."""
  scenery = "".join(
    f'<geom name="prop_{i}" type="box" size="{s[0] / 2} {s[1] / 2} {s[2] / 2}" '
    f'pos="{x} {y} {s[2] / 2}"/>' for i, (x, y, s) in enumerate(props))
  model = mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN, scenery=scenery))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def _in_world(model, data, points, root="pluggybot"):
  """A frame's points (the torso's own axes) in the world."""
  r = model.body(root).id
  return data.xpos[r] + points @ data.xmat[r].reshape(3, 3).T


def _clean(model, **kw):
  """The geometric truth of a frame: no noise, no dropout."""
  return DepthCamera(model, mount="body", noise_k=0.0, dropout=0.0, **kw)


@pytest.fixture(scope="module")
def room():
  return _world()


def test_a_thing_on_the_floor_measures_its_height_and_place():
  """A 5 cm cube 0.9 m ahead: one clean frame into the height map finds
  ONE thing there, 50 mm tall to 3 mm, 30 mm in place."""
  model, data = _world(props=(((0.9 + 0.025, 0.0, (0.05, 0.05, 0.05)),)))
  frame = _clean(model).frame(data)
  hm = HeightMap()
  hm.update((0.0, 0.0, 0.0), _in_world(model, data, frame.points))
  things = [t for t in hm.things() if abs(t["y"]) < 0.5]
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
  noisy = DepthCamera(model, mount="body", dropout=0.0).frame(data).z.reshape(-1)
  ok = np.isfinite(truth) & np.isfinite(noisy)
  res = (noisy - truth)[ok]
  z = truth[ok]
  s1 = res[(z > 0.9) & (z < 1.1)].std()
  s2 = res[(z > 1.9) & (z < 2.1)].std()
  assert 0.0025 < s1 < 0.005, f"σ at 1 m {s1 * 1e3:.2f} mm"
  assert 3.0 < s2 / s1 < 5.5, f"σ(2 m)/σ(1 m) = {s2 / s1:.2f}"


def test_a_near_edge_shadows_the_background_on_its_image_left():
  """The right imager cannot see what a nearer surface hides. A 20 cm post
  0.8 m ahead against the floor beyond: the pixels just LEFT of it in the
  image are invalid, the pixels just right of it are not, and the band is
  f·B·(1/z_near - 1/z_far) wide. Without `_shadow` both sides are valid."""
  model, data = _world(props=(((0.8, 0.0, (0.1, 0.1, 0.2)),)))
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
  point at max_z (which the map would take for floor at 3 m) -- exactly
  the pixels past it, and no others. The limit is on z ALONG THE AXIS, as
  the part's is: a corner pixel at z = 1 m is further by range and still
  valid. And MIN_Z does not bind on this mount: the nearest floor the nose
  sees is past it."""
  model, data = room
  cam = _clean(model, max_z=1.0)
  frame = cam.frame(data)
  full = _clean(model).frame(data)
  z = frame.z[np.isfinite(frame.z)]
  assert z.max() <= 1.0
  assert not np.any(np.isclose(z, 1.0, atol=1e-6)), "a pixel clipped to max"
  assert np.array_equal(np.isnan(frame.z), np.isnan(full.z) | (full.z > 1.0)), \
      "far pixels became points, or near ones were lost"
  ranges = np.linalg.norm(frame.points - cam.origin_robot, axis=1)
  assert ranges.max() > 1.2, f"the cut is on range ({ranges.max():.2f} m)"
  assert depthmod.MIN_Z < np.nanmin(full.z)


def test_a_second_robot_rides_the_handle_and_is_filtered_like_the_first():
  """The second robot's camera is `r2_depth_eye` on `r2_pluggybot`, and
  `exclude_robot` drops the FIRST robot from its frame as the LIDAR does --
  another robot is no information about the floor. The pair stand in the
  living room, the second a metre behind the first."""
  from pluggybot.legs import world as lw
  model = lw.home_spec(first_at=(1.0, -0.5), second_at=(0.0, -0.5)).compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, FIRST.prefix, 1.0, -0.5, 0.0)
  lw.stand(model, data, SECOND.prefix, 0.0, -0.5, 0.0)
  cam = _clean(model, handle=SECOND)
  assert cam.camera_name == "r2_depth_eye"

  def tall(frame):
    p = frame.points
    return p[(_in_world(model, data, p, SECOND.root)[:, 2] > 0.05) & (p[:, 0] < 1.5)]
  assert len(tall(cam.frame(data))) > 20, "the first robot should stand in the second's frame"
  cam.exclude_robot(FIRST.root)
  left = tall(cam.frame(data))
  assert len(left) == 0, f"{len(left)} points of the other robot remain"
