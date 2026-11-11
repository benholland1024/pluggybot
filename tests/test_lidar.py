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

#: The waist's reading at its zero, where these tests' bodies stand (#560's stage (c3):
#: the head's sensors ride the waist, and a frame is laid through its reading).
W0 = {"waist_yaw": 0.0}

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
SITE_Z = hm.CHOSEN.lidar_z
#: The fin's shadow, deg either side of straight back (`model.FIN`: its
#: width at its inner face, behind the sensor).
FIN_DEG = math.degrees(math.atan2(hm.FIN[0] / 2, hm.FIN[1]))
#: ...and how many of the level ring's bearings that is, at most.
FIN_BEARINGS = 2 * math.ceil(FIN_DEG) + 1


def _post(spec, z: float = SITE_Z, half_h: float = 0.05) -> None:
  """A part of the robot in its own level ring: a post on its pelvis, 0.3 m
  ahead of the LIDAR, `half_h` m above and below it, collidable as its own
  geoms are."""
  spec.body("pluggybot").add_geom(name="plane_post", type=mujoco.mjtGeom.mjGEOM_BOX,
                                  pos=[0.3, 0.0, z], size=[0.02, 0.06, half_h],
                                  contype=1, conaffinity=0)


def _world(scenery: str = ROOM, post: bool = False, pitch: float = 0.0,
           post_half_h: float = 0.05):
  """The robot standing (the stand keyframe, its arms hanging) in
  `scenery`, its trunk pitched `pitch` rad forward: (model, data)."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=scenery))
  if post:
    _post(spec, half_h=post_half_h)
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


def _behind(angles, half_deg: float = FIN_DEG + 0.6) -> np.ndarray:
  """Bearings within `half_deg` of straight back: the fin's shadow."""
  return np.abs(np.abs(np.degrees(angles)) - 180.0) <= half_deg


def test_the_level_ring_covers_the_full_circle(room):
  """360 deg, every bearing a return inside the unit's range but the few
  the fin that carries the cap hides behind it, from where the model puts
  the rays' origin."""
  model, data = room
  assert data.site_xpos[model.site("lidar").id][2] == pytest.approx(
    hm.stand_height() + SITE_Z, abs=0.02)
  lidar = _clean(model)
  angles, ranges = lidar.scan(data, W0)
  hidden = np.setdiff1d(np.round(lidar.ray_angles, 9), np.round(angles, 9))
  assert 1 <= len(hidden) <= FIN_BEARINGS and _behind(hidden).all(), np.degrees(hidden)
  edge = math.radians(180 - FIN_DEG - 1.5)
  assert angles.min() < -edge and angles.max() > edge
  assert np.all(ranges > 0.0) and np.all(ranges <= lidar_mod.MAX_RANGE + 1e-9)


def test_the_level_ring_ranges_match_geometry(room):
  """Every bearing reads the room's walls where they stand, 2.98 m ahead
  and 0.98 behind."""
  model, data = room
  angles, ranges = _clean(model).scan(data, W0)
  assert np.allclose(ranges, _room_range(angles), atol=0.002)
  fwd = ranges[np.argmin(np.abs(angles))]
  back = ranges[np.argmin(np.abs(np.abs(angles) - math.pi))]
  assert fwd > back, "the robot stands nearer the back wall"


def test_grid_origin_must_match_the_sensor(room):
  """occupancy_grid.update takes the sensor's offset from the robot's
  centre (`Navigator.LIDAR_ORIGIN`). The humanoid's LIDAR is over its
  pelvis, but a sensor 0.3 m off it laid with the default would slide the
  entire map by the difference."""
  model, data = room
  angles, ranges = _clean(model).scan(data, W0)
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
  angles, _ = _clean(model).scan(data, W0)
  assert len(angles) >= lidar_mod.BEARINGS - FIN_BEARINGS, "its own body is in the ring"
  model, data = _world(post=True)
  lidar = _clean(model)
  angles, _ = lidar.scan(data, W0)
  angles = angles[~_behind(angles, 5.0)]
  missing = sorted(set(range(-170, 171)) - set(np.round(np.degrees(angles)).astype(int)))
  assert 2 < len(missing) < 40, missing
  # The gap is ONE contiguous sector (the post), not scattered holes,
  # pointing at the post, dead ahead of the head
  assert missing == list(range(missing[0], missing[-1] + 1)), missing
  assert abs(np.mean(missing)) < 3, f"blind sector centred at {np.mean(missing):.0f} deg"


def test_the_head_hides_nothing_ahead_and_behind_only_the_floors_rays(room):
  """What the robot's own body hides of a frame, standing with its arms
  hanging (#581's stage 3, `HumanoidSpec.lidar_z`): nothing within 60 deg of
  ahead, down to the field's steepest ray -- it clears the depth camera's
  corners, the visor and the navigation camera; the rings the band's walls
  and fences come off, -24 deg and up, hidden nowhere but on the fin's few
  bearings behind; and the steeper rings, the floor's within 2.6 m, hidden
  round the back half by its panels (`model.BACK`) and aside by the
  shoulders' motors -- a quarter of a frame, nearly all of it floor."""
  model, data = room
  lidar = _clean(model)
  mine = lidar.hidden(data)
  deg = np.degrees(lidar.ray_angles)
  assert not mine[:, np.abs(deg) <= 60].any(), "its body hides rays ahead"
  fin = _behind(lidar.ray_angles)
  assert mine[:, np.abs(np.abs(deg) - 180) <= 1.0].all(), "the premise: the fin is behind it"
  band = np.degrees(lidar.elevations) > -25    # its walls' and its fences' rings, -24 and up
  assert not mine[band][:, ~fin].any(), "a ring the band is laid from, hidden off the fin"
  assert mine[~band][:, np.abs(deg) >= 180 - hm.BACK[0] + 2].all(), \
    "the premise: the back's panels close the steeper rings round the back half"
  _, hit = lidar._cast(data, lidar._dirs)
  by = {model.geom(int(g)).name for g in hit.reshape(mine.shape)[mine]}
  assert {n.removesuffix("_inside") for n in by} <= (
    {f"head_back{k}" for k in range(hm.BACK[1])}
    | {"head_fin", "left_shoulder_roll_motor", "right_shoulder_roll_motor"}), by
  assert lidar.blind_fraction(data) < 0.27


def test_the_lidar_and_the_head_round_it_collide():
  """#581's stage 3: no part of the head is the sim's alone -- the face,
  its visor, the dome, the cap and its rim, the fin, the back's panels and
  the LIDAR in them each meet the world as the head did (stage 1's unit,
  post and plate touched nothing, and a fall passed through them), and a
  box laid on the cap rests on it. Only the panels' painted inner faces
  are drawn and nothing else."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml(hm.CHOSEN))
  mount = model.body("lidar_mount").id
  parts = set(hm.HEAD_GEOMS) | {model.geom(g).name for g in range(model.ngeom)
                                if model.geom_bodyid[g] == mount}
  for name in sorted(parts):
    assert model.geom_contype[model.geom(name).id] & 1, f"{name} touches nothing"
  drawn = {model.geom(g).name for g in range(model.ngeom)
           if model.geom(g).name.startswith("head") and not model.geom_contype[g]}
  assert drawn == {f"head_back{k}_inside" for k in range(hm.BACK[1])}, drawn
  model, data = _world()
  cap = model.geom("head_cap").id
  top = data.geom_xpos[cap][2] + model.geom_size[cap][1]
  lid = f'<geom name="lid" type="box" size="0.3 0.3 0.01" pos="0 0 {top + 0.008}"/>'
  model, data = _world(lid)
  touching = {frozenset((model.geom(c.geom1).name, model.geom(c.geom2).name))
              for c in data.contact[:data.ncon]}
  assert frozenset(("head_cap", "lid")) in touching, touching


def _surface(model, data, g: int, n: int = 48) -> np.ndarray:
  """Points over geom `g`'s surface, the world's frame: a box's corners and
  its edges' points, a cylinder's two rims, a sphere's every way."""
  size, t = model.geom_size[g], model.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    k = np.linspace(-1, 1, 9)
    a, b, c = np.meshgrid(k, k, k, indexing="ij")
    on = (np.abs(a) == 1).astype(int) + (np.abs(b) == 1) + (np.abs(c) == 1) >= 2
    local = np.stack([a[on], b[on], c[on]], axis=1) * size
  elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([size[0] * np.cos(th), size[0] * np.sin(th)], axis=1)
    local = np.vstack([np.column_stack([ring, np.full(n, z)]) for z in (-size[1], size[1])])
  elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
    u, v = np.meshgrid(np.linspace(0, math.pi, n), np.linspace(0, 2 * math.pi, n))
    local = size[0] * np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)],
                               axis=-1).reshape(-1, 3)
  else:
    raise ValueError(t)
  return data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T


def test_no_part_of_the_head_is_in_the_lidars_field_but_the_fin_and_the_back(room):
  """The rings are a sim's sample of the field, and the part's is whole: +7
  down to -52 deg upside down (`model.FIELD_DEG`, its sheet's). Every part of
  the head keeps out of it -- under its steepest ray or over its highest, by
  `FIELD_CLEAR` -- but the fin that carries the cap and the back's panels,
  which hide what they are drawn to hide (the test above): a face, a rim
  or a cap that grazed a field the rings step over would hide rays the
  real unit casts."""
  model, data = room
  site = model.site("lidar").id
  origin, rot = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
  lo, hi = (math.tan(math.radians(a)) for a in hm.FIELD_DEG)
  ours = [g for g in range(model.ngeom) if model.geom(g).name.startswith("head")
          and not model.geom(g).name.startswith(("head_fin", "head_back"))]
  assert len(ours) == 5, [model.geom(g).name for g in ours]
  for g in ours:
    p = (_surface(model, data, g) - origin) @ rot
    p[:, 2] *= -1                          # upside down: the unit's up is the world's down
    r, z = np.hypot(p[:, 0], p[:, 1]), -p[:, 2]
    clear = np.minimum(np.abs(z - hi * r), np.abs(z - lo * r))
    inside = (z < hi * r) & (z > lo * r)
    name = model.geom(g).name
    assert not inside.any(), f"{name} is in the field"
    assert clear.min() > hm.FIELD_CLEAR / 2, f"{name} grazes the field ({clear.min():.4f} m)"


def _reach(model, data, g: int, u: np.ndarray) -> np.ndarray:
  """How far geom `g` reaches along each unit direction in `u` (k, 3), m:
  its support function."""
  c, R, s, t = (data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g],
                model.geom_type[g])
  local = u @ R
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    return u @ c + np.abs(local) @ s
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    return u @ c + s[0]
  if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
    a = np.abs(local[:, 2])
    return u @ c + s[1] * a + s[0] * np.sqrt(np.clip(1 - a * a, 0, 1))
  raise ValueError(t)


def test_no_plane_meets_the_lidar_before_its_head(room):
  """#581's stage 3: the unit lies inside its head's other parts by a
  margin every way -- the floor in a fall, a wall, a door frame, a shelf
  over it -- so whatever comes at the head meets the rim, the cap, the face
  or the back first (the head's support every way past the unit's, read
  off the shapes). At least 15 mm (18: the cap's top over the unit's base),
  and 20 within 60 deg of level, the ways a fall meets a head; stage 3's
  first head held 6 over its unit, and its hard falls put the unit on the
  floor behind the crown."""
  model, data = room
  k = np.arange(20000) + 0.5
  polar, az = np.arccos(1 - 2 * k / len(k)), math.pi * (1 + 5 ** 0.5) * k
  local = np.stack([np.cos(az) * np.sin(polar), np.sin(az) * np.sin(polar), np.cos(polar)], 1)
  u = local @ data.xmat[model.body("pluggybot").id].reshape(3, 3).T
  head = [g for g in range(model.ngeom) if model.geom(g).name.startswith("head")
          and model.geom_contype[g]]
  margin = (np.max([_reach(model, data, g, u) for g in head], axis=0)
            - _reach(model, data, model.geom("lidar").id, u))
  assert margin.min() > 0.015, margin.min()
  level = np.abs(local[:, 2]) < math.sin(math.radians(60))
  assert margin[level].min() > 0.020, margin[level].min()


def test_its_rays_leave_through_its_window_never_its_housing(room):
  """The unit is a body of its own round its rays' origin (`model._lidar`),
  and a cast skips it (`Lidar._housing`): its rays leave through its
  window. Cast from inside its housing, every ray met the housing and the
  robot saw nothing (#386)."""
  model, data = room
  lidar = _clean(model)
  assert len(lidar.scan(data, W0)[0]) >= lidar_mod.BEARINGS - FIN_BEARINGS
  lidar._housing = -1
  assert not len(lidar.scan(data, W0)[0]), "the premise: its housing is round its rays' origin"


def _frame_hits(model, data, cam: str, aspect: float = 16 / 9, n: tuple = (64, 36)) -> set:
  """The robot's own geoms within 2 m a camera's frame meets: rays over it,
  `aspect` wide."""
  ours = lidar_mod.robot_geoms(model, "pluggybot")
  c = model.camera(cam).id
  ty = math.tan(math.radians(model.cam_fovy[c]) / 2)
  u, v = np.meshgrid(np.linspace(-ty * aspect, ty * aspect, n[0]), np.linspace(-ty, ty, n[1]))
  local = np.stack([u.ravel(), v.ravel(), -np.ones(u.size)], axis=1)
  dirs = local @ data.cam_xmat[c].reshape(3, 3).T
  dirs /= np.linalg.norm(dirs, axis=1)[:, None]
  dist, hit = np.zeros(len(dirs)), np.zeros(len(dirs), dtype=np.int32)
  mujoco.mj_multiRay(model, data, data.cam_xpos[c].copy(), dirs.reshape(-1), None, 1, -1, hit,
                     dist, None, len(dirs), mujoco.mjMAXVAL)
  return {model.geom(int(g)).name for g, r in zip(hit, dist) if g in ours and r < 2.0}


def test_the_cameras_see_nothing_of_the_head_round_the_lidar(room):
  """The head was built round the LIDAR with its cameras where they rode
  (#581's stage 3): the depth camera, its colour imager and the
  navigation camera each see nothing of the robot standing -- the cap,
  the fin and the unit are behind and over them. A hand held up in front
  of the face is seen, the premise."""
  model, data = room
  for cam in ("depth_eye", "color_eye", "nav_eye"):
    assert not _frame_hits(model, data, cam), cam
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=ROOM))
  spec.body("pluggybot").add_geom(name="raised_hand", type=mujoco.mjtGeom.mjGEOM_BOX,
                                  pos=[0.4, 0.0, hm.CHOSEN.nav_cam_z], size=[0.04, 0.04, 0.04],
                                  contype=1, conaffinity=0)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  assert "raised_hand" in _frame_hits(model, data, "nav_eye"), "the premise"


def test_a_return_inside_its_blind_zone_is_no_reading():
  """A thing nearer than 0.1 m returns nothing: no range, and no free
  space through it either (the sheet's blind zone)."""
  z = hm.stand_height() + SITE_Z
  near = f'<geom name="near" type="box" size="0.005 0.008 0.01" pos="0.06 0 {z}"/>'
  model, data = _world(ROOM + near)
  angles, _ = _clean(model).scan(data, W0)
  ahead = np.abs(angles) < math.radians(5)
  assert not ahead.any(), "a bearing through the blind zone reported something"
  # ...and every bearing but the box's own 17 and the fin's behind did
  assert len(angles) >= lidar_mod.BEARINGS - 19 - FIN_BEARINGS


def test_noise_is_present_and_bounded(room):
  """A sensor that returns the same number twice is not being modelled:
  each range against its own sigma, the sheet's line from 4 cm at 0.2 m to
  2 cm at 10 m -- and each bearing its own 0.15 deg."""
  model, data = room
  a_c, r_c = _clean(model).scan(data, W0)
  a_n, r_n = Lidar(model, seed=1).scan(data, W0)
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
  rays = len(c.looked)
  assert on_floor.sum() > 0.15 * rays and on_wall.sum() > 0.15 * rays
  assert len(c.peers) == 0
  # ...and every bearing saw something but the few the fin hides
  lidar = _clean(model)
  assert (c.seen | _behind(lidar.ray_angles)).all()
  assert c.seen.sum() >= lidar_mod.BEARINGS - FIN_BEARINGS


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
  assert on_floor.sum() > 0.3 * len(c.looked), f"{on_floor.sum()} returns off the floor"


def _room_range(angles, x0: float = 0.0, fence: bool = False) -> np.ndarray:
  """How far the room's walls are along each bearing from `x0` ahead of the
  pelvis (and the fence's near face, 1.98 m ahead of it and 1.2 m either
  side, where there is one)."""
  c, s = np.cos(angles), np.sin(angles)
  with np.errstate(divide="ignore"):
    r = np.minimum(np.where(c > 0, (2.98 - x0) / c, (0.98 + x0) / -c), 1.48 / np.abs(s))
    if fence:
      ahead = 1.98 - x0
      r = np.where((c > 0) & (np.abs(ahead * s / c) <= 1.2), np.minimum(r, ahead / c), r)
  return r


def test_the_band_lays_a_low_fence_the_level_ring_passes_over():
  """#581's point: the 0.9 m fence 2 m ahead is under the level ring,
  which reads the wall behind it -- and in the band, its near face's 1.98
  m ahead. Every other bearing is the room's walls, never the floor, and
  the fin's shadow behind is no reading, never free space."""
  model, data = _world(ROOM + FENCE)
  lidar = _clean(model)
  angles, ranges = lidar.scan(data, W0)
  assert ranges[np.argmin(np.abs(angles))] == pytest.approx(2.98, abs=0.01), \
    "the premise: the ring passes over the fence"
  level, height = _truth(model, data)
  b_angles, b_ranges = lidar.band(data, level, height, joints=W0)[:2]
  assert np.allclose(b_ranges, _room_range(b_angles, fence=True), atol=0.02)
  assert lidar_mod.BEARINGS - FIN_BEARINGS <= len(b_angles) < lidar_mod.BEARINGS


def test_the_band_is_laid_through_the_tilt():
  """Pitched 10 deg forward, its points through the robot's tilt still put
  the fence's face 1.98 m ahead of its pelvis, seen from where its sensor
  is (`Scan.origin`), the floor out of the band. Laid as if level, the
  floor ahead rises into it and walls the robot in."""
  model, data = _world(ROOM + FENCE, pitch=math.radians(10))
  lidar = _clean(model)
  level, height = _truth(model, data)
  scan = lidar.band(data, level, height, joints=W0)
  ahead = np.abs(scan.angles) < math.radians(15)
  assert np.allclose(scan.ranges[ahead], (1.98 - scan.origin[0]) / np.cos(scan.angles[ahead]),
                     atol=0.03)
  angles, ranges = lidar.band(data, np.eye(3), height, joints=W0)[:2]
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
  angles, ranges = lidar.band(data, level, height, joints=W0)[:2]
  hit = ranges < lidar.max_range - 1e-9
  assert hit.sum() >= 2, "the premise: the band strikes the box"
  gap = np.abs((angles[hit][:, None] - angles[~hit][None, :] + math.pi) % (2 * math.pi) - math.pi)
  assert gap.min() >= math.radians(0.5) - 1e-9, \
    f"a free ray {math.degrees(gap.min()):.2f} deg from a return"


def test_the_sensors_turn_is_read_through_its_site_its_mount_and_the_waist():
  """The band lays points through the sensor's place and turn on the root
  (`Lidar.mount`), read off the model: a site turned on a mount turned is
  both turns -- and the waist under them turned as its reading is handed in
  (#560's stage (c3)), as the compiled world has them at every angle; a
  scan handed no reading is refused, never laid at zero."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN))
  spec.body("lidar_mount").quat = [math.cos(0.15), 0.0, 0.0, math.sin(0.15)]
  spec.body("lidar_mount").pos = [0.03, 0.01, hm.CHOSEN.lidar_z]
  spec.site("lidar").quat = [math.cos(0.2), math.sin(0.2), 0.0, 0.0]
  model = spec.compile()
  data = mujoco.MjData(model)
  lidar = _clean(model)
  root = model.body("pluggybot").id
  site = model.site("lidar").id
  waist = model.jnt_qposadr[model.joint("waist_yaw").id]
  for turn in (0.0, 0.4, -1.0):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[waist] = turn
    mujoco.mj_forward(model, data)
    rot = data.xmat[root].reshape(3, 3)
    offset, mat = lidar.mount.pose({"waist_yaw": turn})
    assert np.allclose(mat, rot.T @ data.site_xmat[site].reshape(3, 3), atol=1e-9), turn
    assert np.allclose(offset, rot.T @ (data.site_xpos[site] - data.xpos[root]), atol=1e-9), turn
  with pytest.raises(ValueError, match="waist_yaw"):
    lidar.band(data, np.eye(3), hm.stand_height())


def test_the_band_is_laid_through_the_waists_turn_its_tilt_the_pelvis_imus():
  """#560's stage (c3): the LIDAR rides the waist and the IMU the pelvis. The
  waist turned 0.6 rad, a band laid through its reading -- and the level
  ring -- reads the room from the pelvis's own heading, as the map and the
  matcher take it; laid at the waist's zero -- the turn left out -- the same
  rays read it 0.6 rad round, and the asymmetric room says so."""
  model, data = _world()
  w = model.jnt_qposadr[model.joint("waist_yaw").id]
  data.qpos[w] = 0.6
  mujoco.mj_forward(model, data)
  level, height = _truth(model, data)
  lidar = _clean(model)
  for name, scan in (("band", lambda j: lidar.band(data, level, height, joints=j)[:2]),
                     ("ring", lambda j: lidar.scan(data, j))):
    angles, ranges = scan({"waist_yaw": 0.6})
    assert len(angles) >= lidar_mod.BEARINGS - FIN_BEARINGS, name
    assert np.allclose(ranges, _room_range(angles), atol=0.03), name
    angles, ranges = scan(W0)
    assert np.abs(ranges - _room_range(angles)).max() > 0.5, f"{name}: the premise"


def test_the_band_reads_a_wall_where_it_stands():
  """A dozen rings or more strike a wall on each bearing, each range with
  its own noise: the band reads the median of the nearest surface's, never
  the least of them -- the least of a dozen read every wall 1.7 sigma
  short, 6-7 cm, a map cell into the room."""
  model, data = _world()
  level, height = _truth(model, data)
  err = []
  for seed in range(4):
    angles, ranges = Lidar(model, seed=seed).band(data, level, height, joints=W0)[:2]
    err.append(ranges - _room_range(angles))
  err = np.concatenate(err)
  assert len(err) > 4 * (lidar_mod.BEARINGS - 10)
  assert abs(np.mean(err)) < 0.01 and abs(np.median(err)) < 0.01, \
    f"the band reads the walls {100 * np.median(err):+.1f} cm off"


def _off(angles, deg: float) -> np.ndarray:
  """How far each bearing is from `deg`, rad, round the circle."""
  return np.abs((angles - math.radians(deg) + math.pi) % (2 * math.pi) - math.pi)


def test_a_turned_mount_lays_its_rays_where_they_went():
  """A sensor 5 cm ahead of the pelvis and turned 10 deg on it: the level
  ring and the band read the front wall straight ahead at 2.93 m from it
  (unturned, the ray read there went 10 deg aside, 2.975 m), and the fin's
  shadow -- the fin is the trunk's, straight behind the sensor -- is 180
  deg round the robot, where the fin is, not 170, where the sensor's own
  bearings would put the rays that met it: no reading there, in either,
  and nothing free anywhere in a closed room."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=ROOM))
  mount = spec.body("lidar_mount")
  mount.pos = np.asarray(mount.pos) + [0.05, 0.0, 0.0]
  mount.quat = [math.cos(math.radians(5)), 0.0, 0.0, math.sin(math.radians(5))]
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  lidar = _clean(model)
  assert lidar.mount.pose(W0)[0][:2] == pytest.approx((0.05, 0.0), abs=1e-6), "the premise"
  level, height = _truth(model, data)
  for name, (angles, ranges) in (("ring", lidar.scan(data, W0)),
                                 ("band", lidar.band(data, level, height, joints=W0)[:2])):
    ahead = np.argmin(np.abs(angles))
    assert ranges[ahead] == pytest.approx(2.93 / math.cos(angles[ahead]), abs=0.005), name
    assert np.allclose(ranges, _room_range(angles, x0=0.05), atol=0.02), name
    assert _off(angles, 180.0).min() > math.radians(1.2), f"{name}: a reading in the fin's shadow"
    assert _off(angles, 170.0).min() < math.radians(0.6), f"{name}: no reading 170 deg round"
    assert len(angles) >= lidar_mod.BEARINGS - FIN_BEARINGS, name


def test_its_noise_is_its_robots_own():
  """Two robots' LIDARs draw apart, one robot's the same twice: the seed
  is off its name (`imu.seed_for`), never `hash()` or a shared 0."""
  model, data = _world()
  first = Lidar(model, robot_body="pluggybot").scan(data, W0)
  again = Lidar(model, robot_body="pluggybot").scan(data, W0)
  other = Lidar(model, robot_body="r2_pluggybot").scan(data, W0)
  assert all(np.array_equal(a, b) for a, b in zip(first, again))
  assert not np.array_equal(first[1], other[1])


def test_a_bearing_clears_no_wall_nearer_than_its_steepest_ray_comes_down():
  """A box 0.3 m tall 0.75 m ahead is under every ray there (the steepest
  meets the band's floor 1 m out), so the band reads the wall behind it:
  laid from the sensor out, that bearing cleared the box a map had from
  afar as the robot walked up to it. Nearer than where its steepest ray
  comes down through the band's floor, a bearing clears no cell the map
  holds a wall in (#581's stage 2), so the box stays as the map had it --
  and lays floor where nothing was seen, as a robot standing amid cells it
  never saw from afar, at its boot, needs to find its way off them."""
  box = '<geom name="low" type="box" size="0.05 0.3 0.15" pos="0.8 0 0.15"/>'
  model, data = _world(ROOM + box)
  lidar = _clean(model)
  level, height = _truth(model, data)
  scan = lidar.band(data, level, height, joints=W0)
  ahead = np.argmin(np.abs(scan.angles))
  assert scan.ranges[ahead] == pytest.approx(2.98, abs=0.03), \
    "the premise: every ray passes over the box"
  assert 0.9 < scan.free_from[ahead] < 1.1, scan.free_from[ahead]
  assert np.all(scan.free_from > 0.9), "a bearing frees floor nearer than its rays come down"

  def laid(free_from):
    g = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
    box, unseen = g.world_to_cell(0.75, 0.0), g.world_to_cell(0.5, 0.0)
    g.grid[box[1], box[0]] = 2.0                    # the box, mapped from afar
    for _ in range(10):
      g.update((0.0, 0.0, 0.0), scan.angles, scan.ranges, lidar.max_range,
               origin=scan.origin, free_from=free_from)
    return g.grid[box[1], box[0]], g.grid[unseen[1], unseen[0]]

  box, unseen = laid(scan.free_from)
  assert box == 2.0, "the band cleared a box it cannot see"
  assert unseen < 0.0, "the band laid no floor where nothing had been seen"
  assert laid(None)[0] < 0.0, "the premise: freed from the sensor out, the box is cleared"


def test_the_bands_floor_rises_with_range():
  """Walking, the IMU's tilt is off by up to 1.8 deg (its 95th percentile,
  `BAND_RISE`): levelled 1.8 deg wrong, the floor 4-8 m out reads 0.13-0.25
  m up, over the band's 0.15 m floor, and walls in a robot in the open. The
  floor rises with range, so none of it is in the band; with no rise, the
  premise, the far floor is."""
  model, data = _world("")                          # no walls: the floor alone
  lidar = _clean(model)
  level, height = _truth(model, data)
  err = math.radians(1.8)
  tilted = np.array([[math.cos(err), 0.0, -math.sin(err)], [0.0, 1.0, 0.0],
                     [math.sin(err), 0.0, math.cos(err)]]) @ level
  scan = lidar.band(data, tilted, height, joints=W0)
  assert np.all(scan.ranges >= lidar.max_range - 1e-9), \
    f"the floor read {scan.ranges.min():.2f} m off in the band"
  scan = lidar.band(data, tilted, height, rise=0.0, joints=W0)
  assert (scan.ranges < lidar.max_range - 1e-9).any(), \
    "the premise: with no rise, the far floor tilted 1.8 deg is in the band"


def test_a_closed_room_has_no_free_bearing_behind_the_fin():
  """Behind the robot a part of it (stage 1's bracket's post, the fin that
  carries the cap now) hides most of a bearing's rays, and the one that
  peeks past it decided the bearing: its point, jittered,
  landing a bearing over, or a steep ray seeing the band empty for a metre
  and meeting the floor, laid its bearing free to 8 m through the wall it
  never saw past -- every bearing round, as the robot turned (#581's stage
  2: 78 frames in 240 here). A bearing is free only off a ray that stayed in
  the band to max range and brought nothing back: in a closed room, none,
  whatever the noise or a walking trunk's tilt."""
  leaks = []
  for roll, pitch in ((0.0, 0.0), (0.01, 0.0), (0.0, 0.015), (-0.012, 0.01), (0.02, -0.01)):
    model, data = _world(ROOM)
    turn = np.zeros(4)
    mujoco.mju_mulQuat(turn, np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]),
                       np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, data.qpos[3:7].copy(), turn)
    data.qpos[3:7] = out
    mujoco.mj_forward(model, data)
    level, height = _truth(model, data)
    for seed in range(20):
      scan = Lidar(model, seed=seed).band(data, level, height, joints=W0)
      free = scan.ranges >= lidar_mod.MAX_RANGE - 1e-9
      leaks += [(roll, pitch, seed, float(np.degrees(a))) for a in scan.angles[free]]
  assert not leaks, f"{len(leaks)} free bearings in a closed room: {leaks[:4]}"


def test_a_dropped_return_frees_nothing():
  """A return the part drops is no reading, as the level ring's
  (`scan_split`): taken for nothing back, a near-level ray whose return off
  the wall was dropped laid its bearing free to 8 m through it -- half the
  returns dropped, in a closed room, 25 bearings in 50 frames; all of them,
  every bearing round."""
  model, data = _world(ROOM)
  level, height = _truth(model, data)
  lidar = Lidar(model, seed=3)
  for dropout in (0.5, 1.0):
    lidar.dropout = dropout
    for _ in range(10):
      scan = lidar.band(data, level, height, joints=W0)
      assert not (scan.ranges >= lidar_mod.MAX_RANGE - 1e-9).any(), \
        f"{dropout:.0%} of the returns dropped laid a bearing free"
  assert not len(lidar.scan_split(data, W0)[0]), \
    "the premise: every return dropped, the ring says nothing"


def test_a_ray_the_rising_floor_meets_short_of_max_range_frees_nothing():
  """The band's floor RISES to meet a ray under level: the -8 deg ring
  comes down through it 7.5 m out. Taken for a floor that fell, it stayed
  in the band to 12.5 m and freed its bearing to 8 m. So, the level rings
  hidden ahead by a post on the robot and nothing within reach, no bearing
  behind the post is free; without the post, the level ring frees them."""
  def ahead(a):
    return np.abs(a) < math.radians(8)
  model, data = _world("", post=True, post_half_h=0.035)
  level, height = _truth(model, data)
  scan = _clean(model).band(data, level, height, joints=W0)
  assert not ahead(scan.angles).any(), \
    f"{int(ahead(scan.angles).sum())} bearings behind the post laid free"
  model, data = _world("")
  scan = _clean(model).band(data, level, height, joints=W0)
  assert ahead(scan.angles).sum() >= 10, "the premise: in the open, the level ring frees them"


def _square(half: float) -> str:
  """A closed square room `half` m from the robot each way, 2.4 m tall."""
  return "".join(
    f'<geom name="side_{i}" type="box" size="{sx} {sy} 1.2" pos="{x} {y} 1.2"/>'
    for i, (sx, sy, x, y) in enumerate([(0.02, half + 0.04, half + 0.02, 0.0),
                                        (0.02, half + 0.04, -half - 0.02, 0.0),
                                        (half + 0.04, 0.02, 0.0, half + 0.02),
                                        (half + 0.04, 0.02, 0.0, -half - 0.02)]))


def test_a_bowed_robots_scan_is_laid_from_where_its_sensor_is():
  """Bowed 52 deg, the sensor is 0.56 m ahead of where it sits on the robot
  upright. A free bearing laid from there to 8 m ran up to 0.5 m past what
  its ray reached: in a room 7 m round, 16 bearings through the wall
  behind. Laid from the scan's origin, where the sensor is, none."""
  half = 7.0
  model, data = _world(_square(half), pitch=math.radians(52))
  level, height = _truth(model, data)
  for lidar in (_clean(model), Lidar(model, seed=1), Lidar(model, seed=2)):
    scan = lidar.band(data, level, height, joints=W0)
    assert scan.origin[0] == pytest.approx(SITE_Z * math.sin(math.radians(52)), abs=0.01), \
      "the premise: bowed, the sensor is half a metre ahead"
    ox, oy = scan.origin
    through = []
    for a in scan.angles[scan.ranges >= lidar.max_range - 1e-9]:
      c, s = math.cos(a), math.sin(a)
      wall = min((w - o) / u for w, o, u in ((half, ox, c), (-half, ox, c), (half, oy, s),
                                             (-half, oy, s)) if abs(u) > 1e-9 and (w - o) / u > 0)
      if wall < lidar.max_range - 0.05:
        through.append((round(math.degrees(a), 1), round(wall, 2)))
    assert not through, f"free bearings through the wall: {through[:6]}"


def _cleared(scenery, cell, pitch: float = 0.0, peer: str = "", scans: int = 10):
  """A wall's cell (`cell`, x, y) the map holds at +2, and what `scans` of
  the band, laid from where it stands, leave of it; and the last scan."""
  model, data = _world(scenery, pitch=pitch)
  lidar = Lidar(model, seed=5)
  if peer:
    lidar.exclude_robot(peer)
  level, height = _truth(model, data)
  g = OccupancyGrid(x_min=-7, y_min=-7, x_max=7, y_max=7, resolution=0.05)
  cx, cy = g.world_to_cell(*cell)
  g.grid[cy, cx] = 2.0
  for _ in range(scans):
    scan = lidar.band(data, level, height, joints=W0)
    g.update((0.0, 0.0, 0.0), scan.angles, scan.ranges, lidar.max_range,
             origin=scan.origin, free_from=scan.free_from)
  return g.grid[cy, cx], scan


def test_a_bowed_robot_clears_no_wall(monkeypatch):
  """Bowed, the rings no longer fan down along every bearing -- behind,
  they point up -- and the steepest, turned past straight down, land
  behind it a few cm out: taken for those bearings' floor, a low box 2 m
  behind, under every ray, was cleared through the wall's return 4 m
  behind, 8 bearings of 9 bowed 52 deg. Tilted past `CLEARS_TILT_RAD` a
  scan clears no wall. (Since the head's back closed its steep rings round
  the back half, #581's stage 3, the ones that turn past straight down
  bowed are a side's, and land beside it: a box 105 or 255 deg round is the
  one they clear.)"""
  def behind(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return ((f'<geom name="far" type="box" size="0.02 3 1.2" pos="{4 * c} {4 * s} 1.2" '
             f'euler="0 0 {deg}"/><geom name="low" type="box" size="0.15 0.3 0.25" '
             f'pos="{2 * c} {2 * s} 0.25" euler="0 0 {deg}"/>'), (1.84 * c, 1.84 * s))
  for deg in (105, 150, 180, 210, 255):
    kept, scan = _cleared(*behind(deg), pitch=math.radians(52), scans=20)
    assert kept >= 2.0, f"bowed, the box {deg} deg round was cleared ({kept:+.1f})"
    assert np.isinf(scan.free_from).all()
  monkeypatch.setattr(lidar_mod, "CLEARS_TILT_RAD", math.pi / 2)
  for deg in (105, 255):
    assert _cleared(*behind(deg), pitch=math.radians(52), scans=20)[0] < 0.0, \
      f"the premise: clearing walls bowed, the box {deg} deg round is cleared"


def test_nothing_low_behind_another_robot_is_cleared():
  """Another robot 2 m ahead hides a 0.5 m box 3.7 m ahead, and the rings
  over its head read the wall 6 m ahead: laid out to it, that bearing
  cleared the box (+5 to -5 in 3 s). Behind another robot nearer than the
  room, a bearing clears no wall."""
  wall = '<geom name="far" type="box" size="0.02 3 1.2" pos="6.02 0 1.2"/>'
  low = '<geom name="low" type="box" size="0.2 0.4 0.25" pos="3.7 0 0.25"/>'
  peer = ('<body name="peer" pos="2.0 0 0"><geom name="peer_body" type="box" '
          'size="0.12 0.25 0.72" pos="0 0 0.72"/></body>')
  kept, scan = _cleared(wall + low + peer, (3.52, 0.0), peer="peer")
  ahead = np.argmin(np.abs(scan.angles))
  assert scan.ranges[ahead] == pytest.approx(6.0, abs=0.1), \
    "the premise: the room ahead reads the wall over its head"
  assert kept == 2.0, f"the box behind the other robot was cleared ({kept:+.1f})"


def test_the_instruments_scan_the_waists_body_at_its_reading(room):
  """#560's stage (c3): a mount the waist turns is never posed without its
  reading, and the instruments that cast the LIDAR from static poses hand it
  in (the waist home): `lidar_spike.py --probe` crashed on the new body
  before they did."""
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  import lidar_spike as ls
  model, data = room
  lidar = _clean(model)
  with pytest.raises(ValueError, match="waist_yaw turns this mount"):
    lidar.scan(data)
  for variant in ("ring", "band"):
    angles, ranges, _ = ls._scan(variant, model, data, lidar)
    assert len(angles) and np.isfinite(ranges).any(), variant


def test_the_static_probe_counts_a_doubted_fit_as_the_fit_it_was(capsys):
  """#681: a fit the walls only partly agree with is "doubted" and pins its
  directions as the "ok" one it was; read as no fit, every place on a
  sidewalk it fired at came out "NOT heading" in `lidar_spike.py --probe`."""
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  import lidar_spike as ls
  ls.probe_summary([{"variant": "band", "zone": "alley", "x": 20.0, "y": -7.0, "why": "doubted",
                     "held": 0, "weak": [1.0, 0.0, 0.0], "returns": 300}])
  line = next(ln for ln in capsys.readouterr().out.splitlines() if "block sidewalks" in ln)
  assert line.split("|")[1].split()[0] == "100%", line
