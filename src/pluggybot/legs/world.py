"""The home world with legs in it (issue #387): the house the rover lived in,
the rover taken out, the quadruped (#377) put in -- one, or a pair --
#378's dock on the living room's south wall, and beside it the rack its arm
takes tools from (#405).

Built at load from `models/home_world.xml` (the generator's file) rather
than generated beside it, so there is one house: a layout change there is a
change here. What is taken out is the rover's and nothing else: its body,
and the actuators, sensor and exclude that name its joints, site and
bodies; and its rack -- the rail, the five modules on their 150 mm pegs, the
dispenser's seeds, and everything that names them (#405: the arm's fork
takes a 220 mm peg). The plates' sensors are the world's and stay.

⚠ THE ROBOTS ARE ATTACHED AFTER THE HOUSE, so their free joints are not at
`qpos[0]`: everything reads a robot's joints by name (`posture.Joints`,
`LegOdometry.qroot`, `RobotHandle.qpos_adr`).
"""

import math

import mujoco

from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
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
#: lying on the pins (`scripts/energy_spike.py --world home_quad --reserve`),
#: with the arm aboard (#405):
#:
#:     travel   2.699 Wh over 44.64 m of route  (60.5 mWh/m; 59.1 without it)
#:     dock     0.270 Wh  (to the standoff, the board, the walk in, the lie-down)
#:
#: The dock leg read 0.432 Wh in #387's walk, and the dearer is carried:
#: 3.131 Wh, plus the constant's own definition, one failed docking --
#: another dock leg and the stand-up and lie-down it costs (93 mWh, #377) --
#: 3.656, carried as 3.7. A property of the floor plan, not of the pack, as
#: the rover's `HOME_LOW_BATTERY_WH` is.
RESERVE_WH = 3.7


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


#: The tool rack's middle bay along the south wall, m: between where the
#: rover's rack stood (x -1.25..1.45) and the dock (x 3.5), its board
#: spanning x 1.6..2.6 -- bay A, the east one, works 1.1 m from the dock's
#: axis, over the peer disc of a robot lying there -- and north of it the
#: floor is clear to the couch's south face (y 0.3).
RACK_X = 2.1


def rack_pose() -> tuple[float, float, float]:
  """The tool rack's frame in the world, (x, y, yaw rad): its origin on the
  floor under the middle bay's peg, its board's back face against the
  living room's south wall, +x north into the room (`legs.rack`'s frame).
  The COMMISSIONED pose, which the robot finds by its tags."""
  from pluggybot.home import world as home
  wall = home.HOUSE_Y[0] + home.WALL_HALF_T          # the wall's inner face
  back = -(rk.DEFAULT.back_x - 0.006)                 # the board's back face
  return RACK_X, wall + back, math.pi / 2


#: The cameras' near plane in a world with legs, m: the nose camera's (a
#: Camera Module 3 focuses from 10 cm). MuJoCo's is `visual.map.znear`
#: times the world's extent, which the house pins at 37.2 m
#: (`home.CAMERA_EXTENT_M`) -- 0.37 m, and a bay's tags are 0.31 m from the
#: nose at its working pose: clipped, the robot found the rack from its
#: approach a metre off and never a bay once it stood at one (#405).
NEAR_M = 0.10

#: What the rover's rack is, by name: taken out of a world with legs (#405).
ROVER_RACK = ("rack", "rack_built")
ROVER_TOOL_PREFIXES = ("module_", "seed_")


def _remove(spec: mujoco.MjSpec, bodies: list) -> None:
  """Take bodies out of a world spec, and everything that names their
  joints, sites, geoms or bodies: actuators, sensors, tendons, equalities,
  excludes, contact pairs."""
  joints, sites, geoms, names = set(), set(), set(), set()
  for body in bodies:
    for b in [body, *body.find_all(mujoco.mjtObj.mjOBJ_BODY)]:
      names.add(b.name)
    joints |= {j.name for j in body.find_all(mujoco.mjtObj.mjOBJ_JOINT) if j.name}
    sites |= {s.name for s in body.find_all(mujoco.mjtObj.mjOBJ_SITE) if s.name}
    geoms |= {g.name for g in body.find_all(mujoco.mjtObj.mjOBJ_GEOM) if g.name}
  every = joints | sites | geoms | names
  tendons = {t.name for t in spec.tendons
             if any(w.target in joints for w in getattr(t, "wraps", []))}
  for act in list(spec.actuators):
    if act.target in joints | sites | tendons | names:
      spec.delete(act)
  for sensor in list(spec.sensors):
    if sensor.objname in every | tendons:
      spec.delete(sensor)
  for tendon in list(spec.tendons):
    if tendon.name in tendons:
      spec.delete(tendon)
  for eq in list(spec.equalities):
    if eq.name1 in every or eq.name2 in every:
      spec.delete(eq)
  for ex in list(spec.excludes):
    if ex.bodyname1 in names or ex.bodyname2 in names:
      spec.delete(ex)
  for pair in list(spec.pairs):
    if pair.geomname1 in geoms or pair.geomname2 in geoms:
      spec.delete(pair)
  for body in bodies:
    spec.delete(body)


def _remove_rover_rack(spec: mujoco.MjSpec) -> None:
  """The rover's rack out of a world spec: its two bodies, its five modules
  and the dispenser's seeds, with everything that names them."""
  top = [b for b in spec.worldbody.bodies
         if b.name in ROVER_RACK or b.name.startswith(ROVER_TOOL_PREFIXES)]
  _remove(spec, top)


def _attach_rack(spec: mujoco.MjSpec, pose) -> None:
  """The tool rack, its tags' textures (the generator's `tags/`) and the
  tools hung on its bays."""
  for i in rk.RACK_TAG_IDS:
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  x, y, yaw = pose
  defaults, tools = rk.tools_xml(pos=(x, y), yaw=yaw)
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><default>{defaults}</default><worldbody>'
    + rk.rack_xml(pos=(x, y), yaw=yaw, name=rk.RACK_BODY) + tools
    + "</worldbody></mujoco>")
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


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
  _remove_rover_rack(spec)
  _attach_quad(spec, first_at, "", None)
  if second_at is not None:
    _attach_quad(spec, second_at, second_prefix, second_rgba)
  _attach_dock(spec, dock_pose())
  _attach_rack(spec, rack_pose())
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec


def stand(model, data, prefix: str = "", x: float | None = None,
          y: float | None = None, yaw: float = 0.0) -> None:
  """Put one quadruped standing, joints in the stand pose and its arm
  stowed, at rest; x and y default to where it is. Forwards the data."""
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
  qs, qe = CHOSEN.arm.stow
  for name, q in (("arm_shoulder", qs), ("arm_elbow", qe), ("arm_wrist", -(qs + qe))):
    jid = model.joint(f"{prefix}{name}").id
    data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[model.jnt_dofadr[jid]] = 0.0
  mujoco.mj_forward(model, data)
