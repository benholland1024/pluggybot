"""Guards for the LIDAR (perception/lidar.py): the Mid-360S upside down on
the humanoid's head (#581)."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.perception import lidar as lidar_mod
from pluggybot.perception.lidar import Lidar

#: A 4 x 3 m box room round the robot, which stands at the origin facing +x:
#: walls' inner faces at x -0.98 and 2.98, y -1.48 and 1.48, and 2.4 m tall,
#: a room's.
ROOM = "".join(
  f'<geom name="wall_{i}" type="box" size="{sx} {sy} 1.2" pos="{x} {y} 1.2"/>'
  for i, (sx, sy, x, y) in enumerate([(0.02, 1.5, 3.0, 0.0), (0.02, 1.5, -1.0, 0.0),
                                      (2.0, 0.02, 1.0, 1.5), (2.0, 0.02, 1.0, -1.5)]))
#: ...and a 0.9 m fence across it 2 m ahead: under the level ring, in the band.
FENCE = '<geom name="fence" type="box" size="0.02 1.2 0.45" pos="2.0 0 0.45"/>'
#: The rays' origin over the pelvis, the robot standing (its model).
SITE_Z = hm.CHOSEN.head_z + hm.CHOSEN.head_half[2] + hm.CHOSEN.lidar_rise


def _post(spec, z: float = SITE_Z) -> None:
  """A part of the robot in its own level ring: a post on its pelvis, 0.3 m
  ahead of the LIDAR, collidable as its own geoms are."""
  spec.body("pluggybot").add_geom(name="plane_post", type=mujoco.mjtGeom.mjGEOM_BOX,
                                  pos=[0.3, 0.0, z], size=[0.02, 0.06, 0.05],
                                  contype=1, conaffinity=0)


def _world(scenery: str = ROOM, post: bool = False, pitch: float = 0.0):
  """The robot standing (the stand keyframe, its arms hanging) in
  `scenery`, its trunk pitched `pitch` rad forward: (model, data)."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=scenery))
  if post:
    _post(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  if pitch:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, data.qpos[3:7].copy(),
                       np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    data.qpos[3:7] = out
  mujoco.mj_forward(model, data)
  return model, data


@pytest.fixture(scope="module")
def room():
  return _world()


def _clean(model):
  """No noise and no dropouts: the geometric truth of the scan."""
  return Lidar(model, noise=False)


def _truth(model, data):
  """The robot's true tilt, level from its root body, and its root's height
  over the floor: what its IMU and its legs' encoders would say."""
  root = model.body("pluggybot").id
  rot = data.xmat[root].reshape(3, 3)
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot, float(data.xpos[root][2])


def _behind(angles, half_deg: float = 2.5) -> np.ndarray:
  """Bearings within `half_deg` of straight back: the bracket's post."""
  return np.abs(np.abs(np.degrees(angles)) - 180.0) <= half_deg


def test_the_level_ring_covers_the_full_circle(room):
  """360 deg, every bearing a return inside the unit's range but the few
  its bracket's post hides behind it, from where the model puts the rays'
  origin."""
  model, data = room
  assert data.site_xpos[model.site("lidar").id][2] == pytest.approx(
    hm.stand_height() + SITE_Z, abs=0.02)
  lidar = _clean(model)
  angles, ranges = lidar.scan(data)
  hidden = np.setdiff1d(np.round(lidar.ray_angles, 9), np.round(angles, 9))
  assert 1 <= len(hidden) <= 5 and _behind(hidden).all(), np.degrees(hidden)
  assert angles.min() < math.radians(-175) and angles.max() > math.radians(175)
  assert np.all(ranges > 0.0) and np.all(ranges <= lidar_mod.MAX_RANGE + 1e-9)


def test_the_level_ring_ranges_match_geometry(room):
  """Forward and backward the ring spans the room wall to wall, 3.96 m."""
  model, data = room
  angles, ranges = _clean(model).scan(data)
  fwd = ranges[np.argmin(np.abs(angles))]
  back = ranges[np.argmin(np.abs(np.abs(angles) - math.pi))]
  assert fwd + back == pytest.approx(3.96, abs=0.002)
  assert fwd > back, "the robot stands nearer the back wall"


def test_grid_origin_must_match_the_sensor(room):
  """occupancy_grid.update takes the sensor's offset from the robot's
  centre (`Navigator.LIDAR_ORIGIN`). The humanoid's LIDAR is over its
  pelvis, but a sensor 0.3 m off it laid with the default would slide the
  entire map by the difference."""
  model, data = room
  angles, ranges = _clean(model).scan(data)
  root = model.body("pluggybot").id
  head = data.site_xpos[model.site("lidar").id] - data.xpos[root]
  assert np.allclose(head[:2], 0.0), "the premise: the humanoid's is over its pelvis"
  pose = (0.0, 0.0, 0.0)
  a = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  b = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
  a.update(pose, angles, ranges, 8.0, origin=(0.3, 0.0))
  b.update(pose, angles, ranges, 8.0)          # the default origin
  assert not np.allclose(a.grid, b.grid), \
    "the origin argument does nothing -- the map cannot be sensor-agnostic"


def test_the_robot_occludes_itself_and_says_so(room):
  """Standing, nothing of the robot is in its level ring; a part of it
  that is (`_post`) hides those bearings, which must be ABSENT, not
  reported as free space -- a self-hit is an absence of information, and
  reporting it as max_range would carve a phantom corridor through the map
  in exactly the direction the robot cannot see."""
  model, data = room
  angles, _ = _clean(model).scan(data)
  assert len(angles) >= lidar_mod.BEARINGS - 5, "its own body is in the ring"
  model, data = _world(post=True)
  lidar = _clean(model)
  angles, _ = lidar.scan(data)
  angles = angles[~_behind(angles, 5.0)]
  missing = sorted(set(range(-170, 171)) - set(np.round(np.degrees(angles)).astype(int)))
  assert 2 < len(missing) < 40, missing
  # The gap is ONE contiguous sector (the post), not scattered holes,
  # pointing at the post, dead ahead of the head
  assert missing == list(range(missing[0], missing[-1] + 1)), missing
  assert abs(np.mean(missing)) < 3, f"blind sector centred at {np.mean(missing):.0f} deg"


def test_its_mount_hides_little_and_nothing_ahead_or_aside(room):
  """What the robot's own body hides of a frame, standing with its arms
  hanging (#581, `HumanoidSpec.lidar_rise`): nothing within 10 deg of
  ahead or of either side, down to the field's steepest ray -- the head's
  top clears it there -- and behind only the bracket's post, a sliver;
  the head's corners hide the steepest rays between."""
  model, data = room
  lidar = _clean(model)
  mine = lidar.hidden(data)
  deg = np.degrees(lidar.ray_angles)
  for name, centre in (("ahead", 0.0), ("left", 90.0), ("right", -90.0)):
    cols = np.abs((deg - centre + 180) % 360 - 180) <= 10
    assert not mine[:, cols].any(), f"its body hides rays {name}"
  post = _behind(lidar.ray_angles)
  assert mine[:, np.abs(np.abs(deg) - 180) <= 1.0].all(), "the premise: the post is behind it"
  steep = np.degrees(lidar.elevations) < -45
  assert not mine[~steep][:, ~post].any(), "something but the post above -45 deg"
  assert lidar.blind_fraction(data) < 0.05


def test_a_return_inside_its_blind_zone_is_no_reading():
  """A thing nearer than 0.1 m returns nothing: no range, and no free
  space through it either (the sheet's blind zone)."""
  z = hm.stand_height() + SITE_Z
  near = f'<geom name="near" type="box" size="0.005 0.008 0.01" pos="0.06 0 {z}"/>'
  model, data = _world(ROOM + near)
  angles, _ = _clean(model).scan(data)
  ahead = np.abs(angles) < math.radians(5)
  assert not ahead.any(), "a bearing through the blind zone reported something"
  assert len(angles) > lidar_mod.BEARINGS - 25


def test_noise_is_present_and_bounded(room):
  """A sensor that returns the same number twice is not being modelled:
  each range against its own sigma, the sheet's line from 4 cm at 0.2 m to
  2 cm at 10 m -- and each bearing its own 0.15 deg."""
  model, data = room
  a_c, r_c = _clean(model).scan(data)
  a_n, r_n = Lidar(model, seed=1).scan(data)
  idx = np.abs(a_n[:, None] - a_c[None, :]).argmin(axis=1)
  t = r_c[idx]
  hits = t < lidar_mod.MAX_RANGE - 1e-9
  err = r_n[hits] - t[hits]
  assert hits.sum() > 300 and np.any(err != 0.0), "the LIDAR is noiseless"
  sigma = lidar_mod._sigma(t[hits])
  # ...a bearing's own jitter moves a range along a wall seen askew
  assert np.all(np.abs(err) < 5.0 * sigma + 0.01), "noise past 5 sigma"
  assert 0.5 < np.sqrt(np.mean((err / sigma) ** 2)) < 2.0
  bearing = a_n - a_c[idx]
  assert 0.5 < np.std(bearing) / lidar_mod.ANGLE_SD < 1.5
  assert len(a_n) < len(a_c), "no return dropped out"


def test_the_cloud_lies_on_the_room():
  """Every return of a noiseless frame is a point on the room's walls or
  floor, where the rays truly struck: the sensor's frame is the trunk's,
  and its origin the site's."""
  model, data = _world()
  c = _clean(model).cloud(data)
  site = data.site_xpos[model.site("lidar").id]
  world = c.points + site                      # standing level, facing +x
  on_floor = np.abs(world[:, 2]) < 1e-6
  on_wall = ((np.abs(world[:, 0] - 2.98) < 1e-6) | (np.abs(world[:, 0] + 0.98) < 1e-6)
             | (np.abs(np.abs(world[:, 1]) - 1.48) < 1e-6))
  assert (on_floor | on_wall).all()
  assert on_floor.sum() > 1000 and on_wall.sum() > 1000
  assert len(c.peers) == 0
  # ...and every bearing saw something but the few its bracket's post hides
  lidar = _clean(model)
  assert (c.seen | _behind(lidar.ray_angles)).all() and c.seen.sum() >= lidar_mod.BEARINGS - 5


def test_the_floor_is_struck_however_far_its_middle():
  """A plane is infinite, and the frame's steep rays meet the floor 1.15-8 m
  out wherever the floor's own middle is: the house's is 19 m off at its
  edge, where a cast cut off at 8 m read nothing off it."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN))
  floor = next(g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE)
  floor.pos = [20.0, 0.0, 0.0]
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  c = _clean(model).cloud(data)
  site = data.site_xpos[model.site("lidar").id]
  on_floor = np.abs((c.points + site)[:, 2]) < 1e-6
  assert on_floor.sum() > 3000, f"{on_floor.sum()} returns off the floor"


def test_the_band_lays_a_low_fence_the_level_ring_passes_over():
  """#581's point: the 0.9 m fence 2 m ahead is under the level ring,
  which reads the wall behind it -- and in the band, its near face's 1.98
  m ahead; the floor never is."""
  model, data = _world(ROOM + FENCE)
  lidar = _clean(model)
  angles, ranges = lidar.scan(data)
  assert ranges[np.argmin(np.abs(angles))] == pytest.approx(2.98, abs=0.01), \
    "the premise: the ring passes over the fence"
  level, height = _truth(model, data)
  b_angles, b_ranges, _, _ = lidar.band(data, level, height)
  ahead = np.abs(b_angles) < math.radians(20)
  assert np.allclose(b_ranges[ahead], 1.98 / np.cos(b_angles[ahead]), atol=0.02)
  assert b_ranges.min() > 0.95, "the floor, or the robot, in the band"


def test_the_band_is_laid_through_the_tilt():
  """Pitched 10 deg forward, its points through the robot's tilt still put
  the fence's face 1.98 m ahead of its pelvis, the floor out of the band.
  Laid as if level, the floor ahead rises into it and walls the robot in."""
  model, data = _world(ROOM + FENCE, pitch=math.radians(10))
  lidar = _clean(model)
  level, height = _truth(model, data)
  angles, ranges, _, _ = lidar.band(data, level, height)
  ahead = np.abs(angles) < math.radians(15)
  assert np.allclose(ranges[ahead], 1.98 / np.cos(angles[ahead]), atol=0.03)
  angles, ranges, _, _ = lidar.band(data, np.eye(3), height)
  ahead = np.abs(angles) < math.radians(15)
  assert ranges[ahead].min() < 1.5, "the premise: unlevelled, the floor reads as a wall"


def test_the_band_lays_no_free_ray_down_a_return():
  """Tilted, a steep ray's point levels degrees off the bearing it was cast
  on. Binned by that bearing, a low box 1.4 m ahead -- which only the
  steep rays reach -- came back on its neighbours' bearings, while its own
  bearing's rays passed over it and read free: a free ray laid 0.12 deg
  from its return, clearing the cell it struck. So every return is half a
  bearing or more from every free ray, the robot rolled 10 deg, in the
  open."""
  low = '<geom name="low" type="box" size="0.03 0.03 0.15" pos="1.4 0 0.15"/>'
  model, data = _world(low)
  roll = math.radians(10)
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, data.qpos[3:7].copy(),
                     np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]))
  data.qpos[3:7] = out
  mujoco.mj_forward(model, data)
  lidar = _clean(model)
  level, height = _truth(model, data)
  angles, ranges, _, _ = lidar.band(data, level, height)
  hit = ranges < lidar.max_range - 1e-9
  assert hit.sum() >= 2, "the premise: the band strikes the box"
  gap = np.abs((angles[hit][:, None] - angles[~hit][None, :] + math.pi) % (2 * math.pi) - math.pi)
  assert gap.min() >= math.radians(0.5) - 1e-9, \
    f"a free ray {math.degrees(gap.min()):.2f} deg from a return"


def test_the_sensors_turn_is_read_through_its_site_and_its_mount():
  """The band lays points through the sensor's place and turn on the root
  (`Lidar.offset`, `Lidar.turn`), read off the model: a site turned on a
  mount turned is both turns, as the compiled world has them."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN))
  spec.body("lidar_mount").quat = [math.cos(0.15), 0.0, 0.0, math.sin(0.15)]
  spec.site("lidar").quat = [math.cos(0.2), math.sin(0.2), 0.0, 0.0]
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  lidar = _clean(model)
  root = model.body("pluggybot").id
  rot = data.xmat[root].reshape(3, 3)
  site = model.site("lidar").id
  assert np.allclose(lidar.turn, rot.T @ data.site_xmat[site].reshape(3, 3), atol=1e-9)
  assert np.allclose(lidar.offset, rot.T @ (data.site_xpos[site] - data.xpos[root]), atol=1e-9)


def test_its_noise_is_its_robots_own():
  """Two robots' LIDARs draw apart, one robot's the same twice: the seed
  is off its name (`imu.seed_for`), never `hash()` or a shared 0."""
  model, data = _world()
  first = Lidar(model, robot_body="pluggybot").scan(data)
  again = Lidar(model, robot_body="pluggybot").scan(data)
  other = Lidar(model, robot_body="r2_pluggybot").scan(data)
  assert all(np.array_equal(a, b) for a, b in zip(first, again))
  assert not np.array_equal(first[1], other[1])
