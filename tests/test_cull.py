"""A ray cast leaves out what it cannot reach (issues #560 and #775;
`perception.cull`): every geom in the model costs every ray of an
`mj_multiRay`, so a geom past a sensor's reach is left out of its cast,
welded to the world or not -- and the reading is the same reading, bit for
bit, noise and all. Pinned on the served world, both sensors that cast, cull
off against on; on the pair in the house, the place put away and set out,
the other robot near and far, the cull against the cull as it was (welded
alone) and none, noise streams and all; and on small worlds, the shapes, a
long wall whose middle is far, a far geom an activity moves or grows into
reach, a far body that moves into reach, and a geom that moves kept where it
could hide a robot the depth camera reads past its gate. And it leaves out
nothing it can reach (#657): MuJoCo's own cull of a whole body, in every
world the robots fly."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from pluggybot.evaluation import counter as ct
from pluggybot.home import restaurant, storeroom
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.world import home_spec
from pluggybot.perception.cull import CULL_GROUP
from pluggybot.perception.depth import UNCULLED, DepthCamera
from pluggybot.perception.lidar import Lidar, robot_geoms
from pluggybot.robot import FIRST, SECOND

#: The waist's reading at its zero, where these tests' bodies stand (#560's stage (c3):
#: the head's sensors ride the waist, and a frame is laid through its reading).
W0 = {"waist_yaw": 0.0}

#: From the living room to the park's west sidewalk, by way of the street,
#: the research center, the mall and the park: (x, y, heading).
PLACES = [(1.5, 0.5, 0.3), (4.0, -4.0, -1.0), (12.5, 1.0, 1.5), (20.5, 1.0, 1.0),
          (21.0, -13.0, 2.5), (18.0, -22.0, -2.0), (-1.0, -17.0, 0.7), (-11.0, -20.0, 0.2)]


def _place(model, data, handle, x: float, y: float, yaw: float) -> None:
  """`handle`'s robot standing at (x, y) facing `yaw`, placed, never stepped."""
  q = handle.qpos_adr(model)
  data.qpos[q:q + 7] = [x, y, hm.stand_height(), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  for name, v in hm.stand_qpos().items():
    data.qpos[model.jnt_qposadr[model.joint(handle.el(name)).id]] = v


def _stand(model, data, x: float, y: float, yaw: float) -> None:
  """The first robot standing at (x, y) facing `yaw`, posed, never stepped."""
  _place(model, data, FIRST, x, y, yaw)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)


def _same(a, b) -> bool:
  """Two readings field by field, bit for bit (a NaN equal to a NaN)."""
  if dataclasses.is_dataclass(a):
    a, b = ([getattr(r, f.name) for f in dataclasses.fields(r)] for r in (a, b))
  def equal(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f")
  return all(equal(x, y) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def world():
  model = home_spec(second_at=(2.5, 0.5)).compile()
  return model, mujoco.MjData(model)


def test_a_cast_leaves_out_nothing_it_could_reach(world):
  model, data = world
  assert not np.any(model.geom_group == CULL_GROUP), "a world put something in the cull's group"
  lidars = Lidar(model), Lidar(model)
  depths = (DepthCamera(model, mount="body", reach=UNCULLED),
            DepthCamera(model, mount="body", reach=UNCULLED))
  lidars[0]._cull = None
  depths[0]._cull.on = False
  groups = model.geom_group.copy()
  left_out = []
  for x, y, yaw in PLACES:
    _stand(model, data, x, y, yaw)
    band = [lid.band(data, np.eye(3), hm.stand_height(), joints=W0) for lid in lidars]
    assert _same(*band), f"the band at ({x}, {y})"
    assert _same(*[lid.scan_split(data, W0) for lid in lidars]), f"the level ring at ({x}, {y})"
    assert _same(*[cam.frame(data, W0) for cam in depths]), f"the depth frame at ({x}, {y})"
    assert np.array_equal(model.geom_group, groups), "a cast left a geom in the cull's group"
    site = data.site_xpos[lidars[1].site_id]
    left_out.append(len(lidars[1]._cull.far(data, site, lidars[1].max_range)))
  assert min(left_out) > 100, f"left out per place: {left_out}"


#: An open floor round the robot: a wall 3 m ahead; 12-15 m off, a box, a
#: cylinder, a capsule and a sphere; and a wall 30 m long whose middle is
#: 20 m off and whose near end is 6 m off.
OPEN = ('<geom name="near" type="box" size="0.05 2 1.2" pos="3 0 1.2"/>'
        '<geom name="far_box" type="box" size="0.5 0.5 1" pos="-12 0 1"/>'
        '<geom name="far_cyl" type="cylinder" size="0.3 1" pos="0 13 1"/>'
        '<geom name="far_cap" type="capsule" size="0.3 0.8" pos="0 -14 1.2"/>'
        '<geom name="far_ball" type="sphere" size="0.6" pos="-10 -10 1"/>'
        '<geom name="long" type="box" size="15 0.05 1.2" pos="6 -20 1.2" euler="0 0 90"/>')


#: ...and a body that moves, 12 m off (scenery with a joint goes after the
#: robot's, `body_xml`).
FREE = '<body name="far_free" pos="0 -12 0.5"><freejoint/><geom name="far_free" type="box" size="0.3 0.3 0.3"/></body>'


def test_what_is_left_out_is_judged_where_and_how_big_it_is_now():
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=OPEN, after=FREE))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  free = model.jnt_qposadr[model.body("far_free").jntadr[0]]
  data.qpos[free:free + 7] = [0.0, -12.0, 0.5, 1.0, 0.0, 0.0, 0.0]
  mujoco.mj_forward(model, data)
  lidars = Lidar(model), Lidar(model)
  lidars[0]._cull = None
  site = data.site_xpos[lidars[1].site_id]
  out = {model.geom(g).name for g in lidars[1]._cull.far(data, site, lidars[1].max_range)}
  assert {"far_box", "far_cyl", "far_cap", "far_ball", "far_free"} <= out, out
  assert not {"near", "long"} & out, out

  def same_band():
    return _same(*[lid.band(data, np.eye(3), hm.stand_height(), joints=W0) for lid in lidars])
  assert same_band()
  # ...an activity moves a geom into reach (`activity.base` writes
  # `geom_pos`), and grows another (`geom_size`), and a body moves into reach
  # (#775: where its joint puts it as the cast meets it): each is cast where
  # it is now
  model.geom_pos[model.geom("far_box").id] = [-4.0, 0.0, 1.0]
  model.geom_size[model.geom("far_ball").id] = [9.0, 0.0, 0.0]
  data.qpos[free:free + 3] = [0.0, -4.0, 0.5]
  mujoco.mj_kinematics(model, data)
  out = {model.geom(g).name for g in lidars[1]._cull.far(data, site, lidars[1].max_range)}
  assert not {"far_box", "far_ball", "far_free"} & out, out
  assert same_band()


# ---- what moves (#775) ------------------------------------------------------

#: The pair's poses in the served house, (x, y, heading) each robot: near each
#: other in the living room; the second in the bedroom behind the first's
#: wall; the second across the street in the research center's lobby; the
#: first in the kitchen and the second in the dining room by the place's
#: table; both in the mall, the first at that table and the second by the
#: pass; the first in the concourse and the second in the park.
PAIR_POSES = [((1.5, 0.5, 0.3), (3.0, 1.2, 2.5)),
              ((0.5, 0.5, 1.4), (1.0, 4.5, -1.0)),
              ((3.0, -1.0, 0.0), (20.5, 1.0, 3.1)),
              ((-8.5, 4.0, -0.4), (21.0, -13.0, 2.5)),
              ((21.2, -12.4, -1.6), (26.0, -12.0, 3.0)),
              ((18.0, -22.0, 1.2), (-1.0, -17.0, 0.7))]


def _pair_sensors(model, me, other) -> tuple[list, list]:
  """`me`'s LIDAR and depth camera as the pair builds them, each excluding the
  other robot (`HumanoidBody.know_peer`), three of each: this cull, the cull
  as it was (welded to the world alone) and none."""
  lidars = [Lidar(model, site_name=me.el("lidar"), robot_body=me.root, seed=11) for _ in range(3)]
  depths = [DepthCamera(model, me, mount="body", reach=UNCULLED, seed=12) for _ in range(3)]
  for sensor in lidars + depths:
    sensor.exclude_robot(other.root)
  lidars[1]._cull.moving = depths[1]._cull.moving = False
  lidars[2]._cull = None
  depths[2]._cull.on = False
  return lidars, depths


def _streams(sensor) -> tuple:
  return sensor.rng.bit_generator.state, sensor.peer_rng.bit_generator.state


def test_a_cast_leaves_out_what_moves_past_reach_and_every_reading_is_as_it_was():
  # ...a world of its own: an offer's set-out redraws the place's hidden weights and friction
  model = home_spec(second_at=(2.5, 0.5)).compile()
  data = mujoco.MjData(model)
  sensors = {me: _pair_sensors(model, me, other) for me, other in ((FIRST, SECOND), (SECOND, FIRST))}
  for me, (lidars, depths) in sensors.items():
    assert not set(lidars[0]._cull.geoms) & robot_geoms(model, me.root), "a LIDAR left out its own robot"
    assert not set(depths[0]._cull.geoms) & set(depths[0]._filtered), "a camera left out a robot it reads"
  dish = np.isin(model.geom_bodyid, [model.body(n).id for n in restaurant.BODIES])
  moves = model.body_weldid[model.geom_bodyid] != 0
  theirs = {FIRST: np.isin(np.arange(model.ngeom), list(robot_geoms(model, SECOND.root))),
            SECOND: np.isin(np.arange(model.ngeom), list(robot_geoms(model, FIRST.root)))}
  left_out = {"the dishware": 0, "the props": 0, "the other robot": 0, "the dishware, by a camera": 0}
  for place in ("away", "out"):
    if place == "out":
      restaurant.set_out(model, data, restaurant.TABLE, 0)
    for first, second in PAIR_POSES:
      _place(model, data, FIRST, *first)
      _place(model, data, SECOND, *second)
      mujoco.mj_kinematics(model, data)
      mujoco.mj_camlight(model, data)
      for me, (lidars, depths) in sensors.items():
        at = f"{me.root} at {first if me is FIRST else second}, the place {place}"
        for read in (lambda lid: lid.band(data, np.eye(3), hm.stand_height(), joints=W0),
                     lambda lid: lid.scan_split(data, W0), lambda lid: lid.cloud(data)):
          got = [read(lid) for lid in lidars]
          assert _same(got[0], got[1]) and _same(got[0], got[2]), f"a LIDAR reading, {at}"
        assert _streams(lidars[0]) == _streams(lidars[1]) == _streams(lidars[2]), at
        frames = [cam.frame(data, W0) for cam in depths]
        assert _same(frames[0], frames[1]) and _streams(depths[0]) == _streams(depths[1]), \
          f"the depth frame against the cull as it was, {at}"
        # ...and against none: the room's pixels always; the rest where the
        # cull as it was is none's (#775: a far wall culled may hide a robot
        # the camera reads past its gate, SimNotes)
        assert _same((frames[0].z, frames[0].points), (frames[2].z, frames[2].points)), at
        if _same(frames[1], frames[2]) and _streams(depths[1]) == _streams(depths[2]):
          assert _same(frames[0], frames[2]), f"the depth frame against none, {at}"
        depths[2].rng.bit_generator.state, depths[2].peer_rng.bit_generator.state = _streams(depths[0])
        site = data.site_xpos[lidars[0].site_id]
        more = np.setdiff1d(lidars[0]._cull.far(data, site, lidars[0].max_range),
                            lidars[1]._cull.far(data, site, lidars[1].max_range))
        left_out["the dishware"] += int(np.count_nonzero(dish[more]))
        left_out["the props"] += int(np.count_nonzero(moves[more] & ~dish[more] & ~theirs[me][more]))
        left_out["the other robot"] += int(np.count_nonzero(theirs[me][more]))
        cam, was = depths[0], depths[1]
        eye, turn = data.cam_xpos[cam.cam_id], data.cam_xmat[cam.cam_id].reshape(3, 3)
        seen = cam._cull.farthest(data, eye, -turn[:, 2], math.acos(cam._widest))
        reach = cam.max_z / cam._widest
        more = np.setdiff1d(cam._cull.far(data, eye, reach, max(cam.max_z, seen) / cam._widest),
                            was._cull.far(data, eye, reach))
        left_out["the dishware, by a camera"] += int(np.count_nonzero(dish[more]))
  assert all(left_out.values()), f"what this cull left out that the cull as it was kept: {left_out}"


#: A robot the depth camera reads (`exclude_robot`) 6 m ahead on an open
#: floor, and a lump that moves 5 m ahead in front of its foot: past the
#: gate's reach (4.4 m at the image's corner), nearer than the robot.
HIDDEN = ('<body name="other" pos="6.0 0 0.3"><freejoint/><geom type="box" size="0.2 0.2 0.3"/></body>'
          '<body name="lump" pos="5.0 0 0.15"><freejoint/><geom type="box" size="0.1 0.3 0.15"/></body>')


def test_a_geom_that_moves_stays_in_a_depth_frame_where_it_could_hide_a_robot_it_reads():
  model = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, after=HIDDEN)).compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)
  _stand(model, data, 0.0, 0.0, 0.0)
  cams = [DepthCamera(model, mount="body", reach=UNCULLED, seed=5) for _ in range(3)]
  for cam in cams:
    cam.exclude_robot("other")
  cams[1]._cull.moving = False
  cams[2]._cull.farthest = lambda *a: 0.0         # left out at the gate's reach alone
  frames = [cam.frame(data, W0) for cam in cams]
  assert _same(frames[0], frames[1]) and _streams(cams[0]) == _streams(cams[1])
  # ...where left out at the gate's reach, the rays through it read the robot
  # behind it, past the gate: its share of the frame, and its peer stream drawn
  assert frames[2].self_fraction > frames[0].self_fraction == 0.0
  assert _streams(cams[2])[1] != _streams(cams[0])[1]


# ---- nothing a ray meets is left out (#657) ----------------------------------

def _misplaced(model) -> dict:
  """How far `mj_multiRay` stands each body welded to the world's cull sphere
  off the body, m, where it does at all: `|ximat @ c - c|`, `c` the offset of
  the body's box from its centre of mass, which MuJoCo 3.10 adds unturned
  (`mju_singleRay`)."""
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  out = {}
  for b in range(1, model.nbody):
    if model.body_bvhadr[b] < 0 or model.body_weldid[b] or model.body_mocapid[b] >= 0:
      continue
    c = model.bvh_aabb[model.body_bvhadr[b], :3]
    off = float(np.linalg.norm(data.ximat[b].reshape(3, 3) @ c - c))
    if off > 1e-5:
      out[model.body(b).name] = round(off, 3)
  return out


def test_no_body_welded_to_the_world_has_its_cull_sphere_misplaced():
  worlds = {"the house's file": mujoco.MjModel.from_xml_path("models/home_world.xml"),
            "the pair's": home_spec(second_at=(2.5, 0.5)).compile()}
  for so in (ct.set_out(0), ct.drawer_set_out(0), ct.door_set_out(0), ct.cube_set_out(0),
             ct.table_set_out(0)):
    worlds[type(so).__name__] = ct.model_of(so)
  assert {name: off for name, off in ((n, _misplaced(m)) for n, m in worlds.items()) if off} == {}


def _both_casts(model, data, eye, dirs):
  """`dirs` from `eye` as one `mj_multiRay` and as one `mj_ray` a ray: each
  one's (geoms, distances)."""
  n = len(dirs)
  multi, multi_at = np.zeros(n, dtype=np.int32), np.zeros(n)
  mujoco.mj_multiRay(model, data, eye, dirs.ravel(), None, 1, -1, multi, multi_at, None, n,
                     mujoco.mjMAXVAL)
  one, hit, at = np.zeros(1, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n)
  for i, v in enumerate(dirs):
    at[i] = mujoco.mj_ray(model, data, eye, v, None, 1, -1, one)
    hit[i] = one[0]
  return (multi, multi_at), (hit, at)


def test_a_cast_meets_every_dining_table_where_one_ray_does():
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  # the ray that went through table 0's top to the floor (#657)
  eye = np.array([20.80, -11.927, 1.2566])
  ray = np.array([21.656, -10.876, 0.0]) - eye
  (multi, multi_at), (one, one_at) = _both_casts(model, data, eye, ray[None] / np.linalg.norm(ray))
  assert model.geom(multi[0]).name == model.geom(one[0]).name == "dining_table_0_geom"
  assert multi_at[0] == pytest.approx(one_at[0])
  # ...and a fan from a head's height at each table's top, from all round
  on_a_top = 0
  for i in range(4):
    top = model.geom(f"dining_table_{i}_geom").id
    hx, hy, hz = model.geom_size[top]
    middle = data.geom_xpos[top]
    tx, ty = np.meshgrid(np.linspace(-1.2 * hx, 1.2 * hx, 31), np.linspace(-1.2 * hy, 1.2 * hy, 21))
    aims = np.column_stack((middle[0] + tx.ravel(), middle[1] + ty.ravel(),
                            np.full(tx.size, middle[2] + hz)))
    for a in np.arange(8) * math.pi / 4:
      eye = middle + np.array([1.5 * math.cos(a), 1.5 * math.sin(a), 0.5])
      dirs = aims - eye
      dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
      (multi, multi_at), (one, one_at) = _both_casts(model, data, eye, dirs)
      assert np.array_equal(multi, one), f"table {i} from {a:.2f} rad"
      assert np.allclose(multi_at, one_at), f"table {i} from {a:.2f} rad"
      on_a_top += int(np.count_nonzero(one == top))
  assert on_a_top > 8000, "the fan missed the tables"


def test_a_redraw_leaves_a_frame_welded_to_the_world_where_it_is():
  model = home_spec().compile()
  data = mujoco.MjData(model)
  still = model.body_weldid == 0
  ipos, iquat = model.body_ipos[still].copy(), model.body_iquat[still].copy()
  for prop in storeroom.PROPS:
    storeroom.redraw(model, data, prop, 7)
  assert np.array_equal(model.body_ipos[still], ipos)
  assert np.array_equal(model.body_iquat[still], iquat)
