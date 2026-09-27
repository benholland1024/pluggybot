"""The perceptive policy's scan as the robot would have it (#388,
`legs/scan.py`): the D435 on the quadruped's nose, a map that keeps what it
saw, read at the scan's grid. `quad_spike.py --climb --scan map` flies it."""

import math

import mujoco
import numpy as np

from pluggybot.legs import model as qm
from pluggybot.legs.actuator import JointLimits
from pluggybot.legs.odometry import LegOdometry
from pluggybot.legs.policy import scan_offsets
from pluggybot.legs.scan import MapScan
from pluggybot.legs.scripted import Command, VirtualModel
from pluggybot.perception.depth import DepthCamera
from pluggybot.perception.heightmap import HeightMap


def _compiled(scenery: str = "", pitch: float = 0.0, drive: str = "position"):
  model = mujoco.MjModel.from_xml_string(
    qm.body_xml(qm.CHOSEN, scenery=scenery, drive=drive))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
  mujoco.mj_forward(model, data)
  return model, data


def test_a_legged_bodys_camera_leaves_the_attitude_to_the_caller():
  # The rover's reconstruction assumes a level chassis; a quadruped pitches
  # 33 degrees on a flight. Pitched 20 degrees nose-down, the floor's points
  # land on the floor through the body's own attitude, and 0.1 m off it
  # through a level one.
  model, data = _compiled(pitch=math.radians(20))
  cam = DepthCamera(model, mount="body", noise_k=0.0, dropout=0.0)
  points = cam.frame(data).points
  root = model.body("pluggybot").id
  body, rot = data.xpos[root], data.xmat[root].reshape(3, 3)
  assert len(points) > 1000
  assert np.abs((body + points @ rot.T)[:, 2]).max() < 1e-6
  assert np.abs((body + points)[:, 2]).max() > 0.1


def test_a_legged_bodys_map_keeps_a_landing_the_rovers_band_drops():
  # The rover's map drops a point 0.5 m over its floor (a table top); a
  # landing up the stairs is 1.8 m over the hall.
  landing = np.array([[1.0, 0.0, 1.8], [1.1, 0.0, 1.8]])
  rover = HeightMap()
  rover.update((0.0, 0.0, 0.0), landing)
  legs = HeightMap(z_band=(-math.inf, math.inf))
  legs.update((0.0, 0.0, 0.0), landing)
  assert np.isnan(rover.height).all()
  assert np.nanmax(legs.height) == 1.8


def test_the_map_scan_reads_the_floor_under_the_robot_at_a_start():
  # The camera sees the floor from ~0.5 m ahead; at a start most of the grid
  # is under and behind the robot, unseen, and it is the floor.
  model, data = _compiled()
  eye = MapScan(model, data)
  assert (~eye.seen()).sum() > 100
  assert np.allclose(eye.scan(), data.qpos[2], atol=0.02)


def test_an_unseen_point_on_a_step_reads_the_step_not_the_floor():
  # The scan's outer rows are outside the D435's field. Read as the ground
  # under the feet, a flight's upper treads were the floor: 0.1-0.4 m wrong
  # on up to a fifth of the points, all the way up a flight of ten. Read as
  # the nearest cell the camera saw, they are the step a few cm inward.
  step = ('\n    <geom name="step" type="box" size="1.0 2.0 0.09" '
          'pos="1.6 0 0.09"/>')                  # 0.18 m, from x = 0.6
  model, data = _compiled(scenery=step)
  eye = MapScan(model, data)
  xy = scan_offsets()
  hidden = (xy[:, 0] > 0.65) & ~eye.seen()
  assert hidden.sum() >= 2
  assert np.allclose(eye.scan()[hidden], data.qpos[2] - 0.18, atol=0.03)


def test_odometry_reckons_the_height_the_legs_lower_the_body():
  # A map of the stairs is laid at the body's reckoned height: a crouch
  # from the stand to 0.15 m at 0.1 m/s, feet planted, is followed to a
  # centimetre. (Faster than `LIFT_M_S`, every planted foot reads as lifting
  # and the height stops following; a flight stepped down at 0.4 m/s
  # lowers the body ~0.26 m/s, `quad_spike.py --climb` measures it.)
  model, data = _compiled(drive="torque")      # the scripted gait: torque
  vm = VirtualModel(model, data, qm.CHOSEN)
  lim = JointLimits.of(qm.CHOSEN.motor)
  odo = LegOdometry(model, data)
  start = data.qpos[2]
  for _ in range(int(2.5 / model.opt.timestep)):
    height = max(0.15, start - 0.1 * data.time)
    data.ctrl[:] = lim.clip(vm.torque(Command(height=height)), data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
    odo.step()
  assert start - data.qpos[2] > 0.15
  assert abs(odo.height_error()) < 0.01
