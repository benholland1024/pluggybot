"""A ray cast leaves out what it cannot reach (issue #560; `perception.cull`):
every geom in the model costs every ray of an `mj_multiRay`, so a static
geom past a sensor's reach is left out of its cast -- and the reading is the
same reading, bit for bit, noise and all. Pinned on the served world, both
sensors that cast, cull off against on; and on a small open world, the
shapes, a long wall whose middle is far, and a far geom an activity moves or
grows into reach."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid.world import home_spec
from pluggybot.perception.cull import CULL_GROUP
from pluggybot.perception.depth import UNCULLED, DepthCamera
from pluggybot.perception.lidar import Lidar
from pluggybot.robot import FIRST

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
    band = [lid.band(data, np.eye(3), hm.stand_height()) for lid in lidars]
    assert _same(*band), f"the band at ({x}, {y})"
    assert _same(*[lid.scan_split(data) for lid in lidars]), f"the level ring at ({x}, {y})"
    assert _same(*[cam.frame(data) for cam in depths]), f"the depth frame at ({x}, {y})"
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
    return _same(*[lid.band(data, np.eye(3), hm.stand_height()) for lid in lidars])
  assert same_band()
  # ...an activity moves a geom into reach (`activity.base` writes
  # `geom_pos`), and grows another (`geom_size`): each is cast where it is now
  model.geom_pos[model.geom("far_box").id] = [-4.0, 0.0, 1.0]
  model.geom_size[model.geom("far_ball").id] = [9.0, 0.0, 0.0]
  mujoco.mj_kinematics(model, data)
  out = {model.geom(g).name for g in lidars[1]._cull.far(data, site, lidars[1].max_range)}
  assert not {"far_box", "far_ball"} & out, out
  assert same_band()
