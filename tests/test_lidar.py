"""Guards for the LIDAR that replaced the stereo pair (perception/lidar.py)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.perception.lidar import LIDAR_ORIGIN, Lidar


@pytest.fixture(scope="module")
def room_model():
  return mujoco.MjModel.from_xml_path("models/room_hub.xml")


@pytest.fixture
def settled(room_model):
  data = mujoco.MjData(room_model)
  yaw = math.pi / 2
  data.qpos[:3] = [0.5, 3.0, 0.045]
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(room_model, data)
  for _ in range(300):
    mujoco.mj_step(room_model, data)
  return data


def _clean(model):
  """A noiseless, dropout-free unit: the geometric truth of the scan."""
  return Lidar(model, sigma_m=0.0, sigma_frac=0.0, dropout=0.0)


def test_scan_covers_the_full_circle(room_model, settled):
  """The whole point of the swap: 360 deg, not a 67 deg camera cone. The
  forward reflex fundamentally could not protect a spin before this."""
  angles, ranges = _clean(room_model).scan(settled)
  assert angles.min() < math.radians(-150)
  assert angles.max() > math.radians(150)
  assert len(angles) > 300
  assert np.all(ranges > 0.0)
  assert np.all(ranges <= 8.0 + 1e-9)


def test_the_robot_occludes_itself_and_says_so(room_model, settled):
  """The mast stands in the scan plane. Those bearings must be ABSENT, not
  reported as free space -- a self-hit is an absence of information, and
  reporting it as max_range would carve a phantom corridor through the map
  in exactly the direction the robot cannot see."""
  lidar = _clean(room_model)
  angles, _ = lidar.scan(settled)
  blind = lidar.blind_fraction(settled)
  assert 0.01 < blind < 0.20, (
    f"blind fraction {blind:.1%} -- either the self-filter is not running, "
    f"or the mount has been moved somewhere that occludes far too much")
  # The gap should be ONE contiguous sector (the mast), not scattered holes.
  missing = sorted(set(np.round(np.degrees(lidar.ray_angles)).astype(int))
                   - set(np.round(np.degrees(angles)).astype(int)))
  runs, run = [], [missing[0]]
  for a in missing[1:]:
    (run.append(a) if a == run[-1] + 1 else (runs.append(run), run.clear(),
                                             run.append(a)))
  runs.append(run)
  biggest = max(len(r) for r in runs)
  assert biggest >= 10, "the mast's shadow should be a contiguous sector"
  # ...and it should point at the mast, which sits at body-local (-0.10,-0.05)
  centre = np.mean([r for r in runs if len(r) == biggest][0])
  assert abs(centre - math.degrees(math.atan2(-0.05, -0.10))) < 12, (
    f"blind sector centred at {centre:.0f} deg is not the mast")


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
  # Bound each ray against ITS OWN sigma: the model is +/-10 mm + 1 % of
  # range, so a 6 m return is allowed 73 mm of 1-sigma noise while a 0.5 m
  # return is allowed 15 mm. A single flat threshold fails on the long rays
  # for no good reason (it did: 196 mm at 6.3 m is 2.7 sigma, ordinary across
  # 300 rays).
  t = np.array([t for t, _ in paired])
  hits = t < 8.0 - 1e-9
  sigma = 0.010 + 0.01 * t
  assert np.all(np.abs(err[hits]) < 4.0 * sigma[hits]), "noise exceeds 4 sigma"
  assert (np.abs(err[hits]) / sigma[hits]).mean() < 1.5


def test_ranges_match_geometry(room_model, settled):
  """Ranging accuracy against the model's own geometry: the robot sits at
  (0.5, 3.0) facing +y in room_hub, so the forward ray must reach the far
  wall, and opposite bearings must sum to the room's width."""
  angles, ranges = _clean(room_model).scan(settled)
  fwd = ranges[np.argmin(np.abs(angles))]
  back = ranges[np.argmin(np.abs(np.abs(angles) - math.pi))]
  assert 0.3 < fwd < 8.0
  # Forward + backward through the robot spans the room along y.
  assert 2.0 < fwd + back < 12.0


def test_grid_origin_must_match_the_sensor(room_model, settled):
  """occupancy_grid.update takes the sensor's offset from the axle. The
  default is the plug-era head camera; the LIDAR sits elsewhere, and passing
  the wrong one slides the entire map by the difference."""
  angles, ranges = _clean(room_model).scan(settled)
  pose = (0.5, 3.0, math.pi / 2)
  a = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  b = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  a.update(pose, angles, ranges, 8.0, origin=LIDAR_ORIGIN)
  b.update(pose, angles, ranges, 8.0)          # wrong (camera) origin
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
  that return nothing -- over several poses, so both noise streams run
  long enough to drift apart if a draw moved."""
  from pluggybot.robot import SECOND, world_with_robots
  model = world_with_robots("models/room_hub.xml", second_at=(2.0, 3.0))
  data = mujoco.MjData(model)
  batched, reference = Lidar(model, seed=3), Lidar(model, seed=3)
  for lidar in (batched, reference):
    lidar.exclude_robot(SECOND.root)
  saw_peer = saw_far = False
  for k, (x, y, yaw) in enumerate([(0.5, 3.0, 0.0), (1.0, 2.5, 0.4),
                                   (0.8, 3.4, -0.3), (-1.0, 1.0, 2.5),
                                   (1.2, 3.0, 0.1)] * 3):
    data.qpos[:3] = [x, y, 0.045]
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    got, want = batched.scan_split(data), _one_ray_at_a_time(reference, data)
    for g, w, name in zip(got, want, ("angles", "ranges", "peer angles",
                                      "peer ranges")):
      assert g.dtype == w.dtype and np.array_equal(g, w), \
        f"pose {k}: the batched scan's {name} differ from the per-ray scan's"
    saw_peer |= got[2].size > 0
    saw_far |= bool((got[1] == batched.max_range).any())
  assert saw_peer and saw_far, "the poses never exercised a peer or a far ray"
