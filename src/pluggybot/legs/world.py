"""The home world with legs in it (issue #387): the house the rover lived in,
the rover taken out, the quadruped (#377) put in -- one, or a pair -- and
#378's dock on the living room's south wall.

Built at load from `models/home_world.xml` (the generator's file) rather
than generated beside it, so there is one house: a layout change there is a
change here. What is taken out is the rover's and nothing else -- its body,
and the actuators, sensor and exclude that name its joints, site and
bodies; the tools' actuators and the plates' sensors are the world's and
stay. The rack stays too: its tools wait for the arm (#378, step 4 of #375).

⚠ THE ROBOTS ARE ATTACHED AFTER THE HOUSE, so their free joints are not at
`qpos[0]`: everything reads a robot's joints by name (`posture.Joints`,
`LegOdometry.qroot`, `RobotHandle.qpos_adr`).
"""

import math

import mujoco

from pluggybot.legs import dock as dk
from pluggybot.legs.model import CHOSEN, attachable, pose_qpos
from pluggybot.rack.tags import DOCK_TAG_IDS
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The house the quadruped moves into.
HOME_XML = "models/home_world.xml"
#: The rover's root body: `_remove_rover` takes it out, with every
#: actuator, sensor and exclude that names its joints, site or bodies.
ROVER_ROOT = ROBOT_ROOT


#: The quadruped's packs in this world, Wh. A test's: small enough that a
#: day's test charges. The served pack is the real one (`model.PACK_WH`).
DEMO_WH = 20.0
#: The return-trip reserve, Wh: MEASURED on the quadruped over the rover's
#: worst-case route (`home.HOME_WORST_RETURN_PATH`: the loop's south-west
#: corner, the sidewalk band, the gate and the garden door) to a REAL dock,
#: lying on the pins (`scripts/energy_spike.py --world home_quad --reserve`):
#:
#:     travel   2.639 Wh over 44.62 m of route  (59.1 mWh/m, twice the rover's)
#:     dock     0.432 Wh  (to the standoff, the board, the walk in, the lie-down)
#:     floor    3.071 Wh
#:
#: ...plus the constant's own definition, one failed docking: another dock
#: leg and the stand-up and lie-down it costs (93 mWh, #377) -- 3.596,
#: carried as 3.6. A property of the floor plan, not of the pack, as the
#: rover's `HOME_LOW_BATTERY_WH` is.
RESERVE_WH = 3.6


def dock_pose() -> tuple[float, float, float]:
  """The dock's frame in the world, (x, y, yaw rad): its seat on the living
  room's floor, the board against the south wall (+x toward it). Clear of
  the rack (x -1.25..1.45) and of the couch's south face (y 0.45) by the
  approach's standoff: the robot comes in from 1.0 m behind the seat.
  The COMMISSIONED pose, which a robot lying on it is anchored to."""
  from pluggybot.home import world as home
  wall = home.HOUSE_Y[0] + home.WALL_HALF_T          # the wall's inner face
  back = dk.DEFAULT.board_x + 0.06                    # the board's post, behind it
  return 3.5, wall + back, -math.pi / 2


def _remove_rover(spec: mujoco.MjSpec) -> None:
  """Take the rover out of a world spec: what names its joints, site or
  bodies, then its body."""
  body = spec.body(ROVER_ROOT)
  joints = {j.name for j in body.find_all(mujoco.mjtObj.mjOBJ_JOINT) if j.name}
  sites = {s.name for s in body.find_all(mujoco.mjtObj.mjOBJ_SITE) if s.name}
  bodies = {b.name for b in body.find_all(mujoco.mjtObj.mjOBJ_BODY)} | {ROVER_ROOT}
  for act in list(spec.actuators):
    if act.target in joints:
      spec.delete(act)
  for sensor in list(spec.sensors):
    if sensor.objname in joints | sites | bodies:
      spec.delete(sensor)
  for ex in list(spec.excludes):
    if ex.bodyname1 in bodies or ex.bodyname2 in bodies:
      spec.delete(ex)
  spec.delete(body)


def _attach_quad(spec: mujoco.MjSpec, pos, prefix: str, rgba) -> None:
  robot = attachable(CHOSEN)
  if rgba is not None and tuple(rgba) != CHASSIS_RGBA:
    paint(robot, rgba)
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), 0.0]
  spec.attach(robot, prefix=prefix, frame=frame)


def _attach_dock(spec: mujoco.MjSpec, pose) -> None:
  """The cradle and its board, and the board's tag textures (the generator's
  `tags/` beside the world's file)."""
  for i in DOCK_TAG_IDS:
    tex = spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                           file=f"tags/tag{i}.png")
    del tex
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  x, y, yaw = pose
  child = mujoco.MjSpec.from_string(
    "<mujoco><compiler angle=\"radian\"/><worldbody>"
    + dk.dock_xml(pos=(x, y), yaw_deg=math.degrees(yaw)) + "</worldbody></mujoco>")
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


def home_spec(first_at=(1.5, 0.5), second_at=None, second_prefix: str = "r2_",
              second_rgba=SECOND_CHASSIS_RGBA, path: str = HOME_XML) -> mujoco.MjSpec:
  """The home world's SPEC with the quadruped at `first_at` and, optionally,
  a second at `second_at` in its own livery, and the dock. Kept by the
  lifecycle as the rover's is (`robot.world_spec`)."""
  spec = mujoco.MjSpec.from_file(path)
  _remove_rover(spec)
  _attach_quad(spec, first_at, "", None)
  if second_at is not None:
    _attach_quad(spec, second_at, second_prefix, second_rgba)
  _attach_dock(spec, dock_pose())
  return spec


def stand(model, data, prefix: str = "", x: float | None = None,
          y: float | None = None, yaw: float = 0.0) -> None:
  """Put one quadruped standing, joints in the stand pose, at rest; x and y
  default to where it is. Forwards the data."""
  root = model.body(f"{prefix}{ROBOT_ROOT}").id
  q = int(model.jnt_qposadr[model.body_jntadr[root]])
  v = int(model.jnt_dofadr[model.body_jntadr[root]])
  if x is not None:
    data.qpos[q:q + 2] = (x, y)
  data.qpos[q + 2] = CHOSEN.stand_height
  data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  j = int(model.jnt_qposadr[model.joint(f"{prefix}FL_hip_abd").id])
  data.qpos[j:j + 12] = pose_qpos(CHOSEN, CHOSEN.stand_height)
  jv = int(model.jnt_dofadr[model.joint(f"{prefix}FL_hip_abd").id])
  data.qvel[v:v + 6] = 0.0
  data.qvel[jv:jv + 12] = 0.0
  mujoco.mj_forward(model, data)
