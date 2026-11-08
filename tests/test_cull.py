"""A ray cast leaves out what it cannot reach (issue #560; `perception.cull`):
every geom in the model costs every ray of an `mj_multiRay`, so a static
geom past a sensor's reach is left out of its cast -- and the reading is the
same reading, bit for bit, noise and all. Pinned on the served world, both
sensors that cast, cull off against on; and on a small open world, the
shapes, a long wall whose middle is far, and a far geom an activity moves or
grows into reach. And it leaves out nothing it can reach (#657): MuJoCo's own
cull of a whole body, in every world the robots fly."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from pluggybot.evaluation import counter as ct
from pluggybot.home import storeroom
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.world import home_spec
from pluggybot.perception.cull import CULL_GROUP
from pluggybot.perception.depth import UNCULLED, DepthCamera
from pluggybot.perception.lidar import Lidar
from pluggybot.robot import FIRST

#: The waist's reading at its zero, where these tests' bodies stand (#560's stage (c3):
#: the head's sensors ride the waist, and a frame is laid through its reading).
W0 = {"waist_yaw": 0.0}

#: From the living room to the park's west sidewalk, by way of the street,
#: the research center, the mall and the park: (x, y, heading).
PLACES = [(1.5, 0.5, 0.3), (4.0, -4.0, -1.0), (12.5, 1.0, 1.5), (20.5, 1.0, 1.0),
          (21.0, -13.0, 2.5), (18.0, -22.0, -2.0), (-1.0, -17.0, 0.7), (-11.0, -20.0, 0.2)]


def _stand(model, data, x: float, y: float, yaw: float) -> None:
  """The first robot standing at (x, y) facing `yaw`, placed, never stepped."""
  q = FIRST.qpos_adr(model)
  data.qpos[q:q + 7] = [x, y, hm.stand_height(), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  for name, v in hm.stand_qpos().items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = v
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


def test_what_is_left_out_is_judged_where_and_how_big_it_is_now():
  spec = mujoco.MjSpec.from_string(hm.body_xml(hm.CHOSEN, scenery=OPEN))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  lidars = Lidar(model), Lidar(model)
  lidars[0]._cull = None
  site = data.site_xpos[lidars[1].site_id]
  out = {model.geom(g).name for g in lidars[1]._cull.far(data, site, lidars[1].max_range)}
  assert {"far_box", "far_cyl", "far_cap", "far_ball"} <= out and not {"near", "long"} & out, out

  def same_band():
    return _same(*[lid.band(data, np.eye(3), hm.stand_height(), joints=W0) for lid in lidars])
  assert same_band()
  # ...an activity moves a geom into reach (`activity.base` writes
  # `geom_pos`), and grows another (`geom_size`): each is cast where it is now
  model.geom_pos[model.geom("far_box").id] = [-4.0, 0.0, 1.0]
  model.geom_size[model.geom("far_ball").id] = [9.0, 0.0, 0.0]
  mujoco.mj_kinematics(model, data)
  out = {model.geom(g).name for g in lidars[1]._cull.far(data, site, lidars[1].max_range)}
  assert not {"far_box", "far_ball"} & out, out
  assert same_band()


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
