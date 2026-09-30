"""Guards for the LIDAR (perception/lidar.py), on the quadruped's rear mast."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.legs import model as qm
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.perception.lidar import Lidar

#: A 4 x 3 m box room round the robot, which stands at the origin facing +x:
#: walls' inner faces at x -0.98 and 2.98, y -1.48 and 1.48.
ROOM = "".join(
  f'<geom name="wall_{i}" type="box" size="{sx} {sy} 0.6" pos="{x} {y} 0.6"/>'
  for i, (sx, sy, x, y) in enumerate([(0.02, 1.5, 3.0, 0.0), (0.02, 1.5, -1.0, 0.0),
                                      (2.0, 0.02, 1.0, 1.5), (2.0, 0.02, 1.0, -1.5)]))


def _arm(model, data, shoulder: float, elbow: float) -> None:
  for name, q in (("arm_shoulder", shoulder), ("arm_elbow", elbow)):
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
  mujoco.mj_forward(model, data)


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_string(qm.body_xml(qm.CHOSEN, scenery=ROOM))


@pytest.fixture
def settled(room_model):
  """Standing, the arm stowed (the stand keyframe)."""
  data = mujoco.MjData(room_model)
  mujoco.mj_resetDataKeyframe(room_model, data, 0)
  mujoco.mj_forward(room_model, data)
  return data


def _clean(model):
  """A noiseless, dropout-free unit: the geometric truth of the scan."""
  return Lidar(model, sigma_m=0.0, sigma_frac=0.0, dropout=0.0)


def test_scan_covers_the_full_circle(room_model, settled):
  """360 deg, every bearing a return inside the unit's range."""
  angles, ranges = _clean(room_model).scan(settled)
  assert angles.min() < math.radians(-150)
  assert angles.max() > math.radians(150)
  assert len(angles) > 300
  assert np.all(ranges > 0.0)
  assert np.all(ranges <= 8.0 + 1e-9)


def test_the_robot_occludes_itself_and_says_so(room_model, settled):
  """The stowed arm is under the scan plane and hides nothing; unfolded, it
  stands in it. Those bearings must be ABSENT, not reported as free space
  -- a self-hit is an absence of information, and reporting it as
  max_range would carve a phantom corridor through the map in exactly the
  direction the robot cannot see."""
  lidar = _clean(room_model)
  assert lidar.blind_fraction(settled) == 0.0, "the stowed arm is in the plane"
  _arm(room_model, settled, math.pi, -1.5)
  angles, _ = lidar.scan(settled)
  blind = lidar.blind_fraction(settled)
  assert 0.01 < blind < 0.20, (
    f"blind fraction {blind:.1%} -- either the self-filter is not running, "
    f"or the mount has been moved somewhere that occludes far too much")
  # The gap is ONE contiguous sector (the arm), not scattered holes...
  missing = sorted(set(np.round(np.degrees(lidar.ray_angles)).astype(int))
                   - set(np.round(np.degrees(angles)).astype(int)))
  assert missing == list(range(missing[0], missing[-1] + 1)), missing
  # ...and it points at the arm, dead ahead of the rear mast
  assert abs(np.mean(missing)) < 3, f"blind sector centred at {np.mean(missing):.0f} deg"


def test_noise_is_present_and_bounded(room_model, settled):
  """A sensor that returns the same number twice is not being modelled."""
  clean = _clean(room_model)
  noisy = Lidar(room_model, seed=1)
  a_c, r_c = clean.scan(settled)
  a_n, r_n = noisy.scan(settled)
  truth = dict(zip(np.round(a_c, 6), r_c))
  paired = [(truth[k], v) for k, v in zip(np.round(a_n, 6), r_n) if k in truth]
  assert len(paired) > 250
  err = np.array([n - t for t, n in paired])
  assert np.any(err != 0.0), "the LIDAR is noiseless -- that is not a sensor"
  # Each ray against ITS OWN sigma: the model is +/-10 mm + 1 % of range,
  # so a flat threshold fails on the long rays for no good reason.
  t = np.array([t for t, _ in paired])
  hits = t < 8.0 - 1e-9
  sigma = 0.010 + 0.01 * t
  assert np.all(np.abs(err[hits]) < 4.0 * sigma[hits]), "noise exceeds 4 sigma"
  assert (np.abs(err[hits]) / sigma[hits]).mean() < 1.5


def test_ranges_match_geometry(room_model, settled):
  """Ranging accuracy against the model's own geometry: forward and
  backward through the mast span the room wall to wall, 3.96 m."""
  angles, ranges = _clean(room_model).scan(settled)
  fwd = ranges[np.argmin(np.abs(angles))]
  back = ranges[np.argmin(np.abs(np.abs(angles) - math.pi))]
  assert fwd + back == pytest.approx(3.96, abs=0.002)
  assert fwd > back, "the rear mast is nearer the back wall"


def test_grid_origin_must_match_the_sensor(room_model, settled):
  """occupancy_grid.update takes the sensor's offset from the robot's
  centre. The LIDAR sits on the rear mast, and passing the default slides
  the entire map by the difference."""
  angles, ranges = _clean(room_model).scan(settled)
  root = room_model.body("pluggybot").id
  mast = settled.site_xpos[room_model.site("lidar").id] - settled.xpos[root]
  pose = (0.0, 0.0, 0.0)
  a = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  b = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  a.update(pose, angles, ranges, 8.0, origin=(float(mast[0]), float(mast[1])))
  b.update(pose, angles, ranges, 8.0)          # the default origin
  assert not np.allclose(a.grid, b.grid), \
    "the origin argument does nothing -- the map cannot be sensor-agnostic"


def _one_ray_at_a_time(lidar, data):
  """The scan as it was cast before issue #385: `mj_ray` per bearing, the
  noise drawn inside the loop. Kept here as the reference the batch must
  equal bit for bit."""
  pos = np.array(data.site_xpos[lidar.site_id])
  mat = np.array(data.site_xmat[lidar.site_id]).reshape(3, 3)
  dirs = lidar._dirs @ mat.T
  geomid = np.zeros(1, dtype=np.int32)
  out = ([], [], [], [])
  for i in range(lidar.n_rays):
    dist = mujoco.mj_ray(lidar.model, data, pos, np.ascontiguousarray(dirs[i]),
                         None, 1, -1, geomid)
    hit = int(geomid[0])
    if dist >= 0.0 and hit in lidar._self_geoms:
      continue
    peer = dist >= 0.0 and hit in lidar._other_geoms
    if dist < 0.0 or dist >= lidar.max_range:
      out[0].append(lidar.ray_angles[i])
      out[1].append(lidar.max_range)
      continue
    rng = lidar.peer_rng if peer else lidar.rng
    if rng.random() < lidar.dropout:
      continue
    noisy = dist + rng.normal(0.0, lidar.sigma_m + lidar.sigma_frac * dist)
    a, r = (out[2], out[3]) if peer else (out[0], out[1])
    a.append(lidar.ray_angles[i])
    r.append(float(np.clip(noisy, 0.02, lidar.max_range)))
  return tuple(np.asarray(x, dtype=float) for x in out)


def test_the_batched_scan_is_the_scan_it_replaced():
  """Issue #385 casts the 360 rays in ONE `mj_multiRay` and keeps the noise
  loop only for the draws. A day must hash as it did (the parity rule), so
  every ray is pinned against the per-ray scan it replaced: the self-hits,
  another robot's returns on their own stream, the dropouts, the bearings
  that return nothing -- over several poses of the pair in the house, so
  both noise streams run long enough to drift apart if a draw moved. The
  other robot stands under a level scan's plane, so the first is pitched
  as on a flight, and its scan cuts the other's torso."""
  from pluggybot.legs import world as lw
  from pluggybot.robot import SECOND
  model = lw.home_spec(second_at=(2.5, 0.5)).compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, SECOND.prefix, 2.5, 0.5, 0.0)
  batched, reference = Lidar(model, seed=3), Lidar(model, seed=3)
  for lidar in (batched, reference):
    lidar.exclude_robot(SECOND.root)
  own = _clean(model)                   # counts self-hits off the two streams
  q = model.jnt_qposadr[model.body("pluggybot").jntadr[0]]
  saw_peer = saw_far = saw_self = False
  for k, (x, y, yaw, pitch) in enumerate([(1.5, 0.5, 0.0, 0.2), (1.0, 0.0, 0.4, 0.0),
                                          (2.0, -0.5, -0.3, 0.15), (0.5, 1.5, 2.5, 0.0),
                                          (1.6, 0.6, 0.1, 0.25)] * 3):
    lw.stand(model, data, "", x, y, yaw)
    _arm(model, data, math.pi, -1.5 if k % 2 else -math.pi + 0.175)
    tilt = np.zeros(4)
    mujoco.mju_mulQuat(tilt, data.qpos[q + 3:q + 7].copy(),
                       np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    data.qpos[q + 3:q + 7] = tilt
    mujoco.mj_forward(model, data)
    got, want = batched.scan_split(data), _one_ray_at_a_time(reference, data)
    for g, w, name in zip(got, want, ("angles", "ranges", "peer angles",
                                      "peer ranges")):
      assert g.dtype == w.dtype and np.array_equal(g, w), \
        f"pose {k}: the batched scan's {name} differ from the per-ray scan's"
    saw_peer |= got[2].size > 0
    saw_far |= bool((got[1] == batched.max_range).any())
    saw_self |= own.blind_fraction(data) > 0.05
  assert saw_peer and saw_far and saw_self, \
    "the poses never exercised a peer, a far ray or a self-hit"
