"""The home world with legs in it (issue #387): the house, the quadruped
(#377) put in -- one, or a pair -- #378's dock on the living room's south
wall, and beside it the rack its arm takes tools from (#405).

Built at load from `models/home_world.xml` (the generator's file, a house
with no robot in it) rather than generated beside it, so there is one house:
a layout change there is a change here.

⚠ THE ROBOTS ARE ATTACHED AFTER THE HOUSE, so their free joints are not at
`qpos[0]`: everything reads a robot's joints by name (`posture.Joints`,
`LegOdometry.qroot`, `RobotHandle.qpos_adr`).
"""

import math

import mujoco

from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.legs.model import CHOSEN, attachable, pose_qpos
from pluggybot.rack.tags import BOARD_TAG_IDS, DOCK_TAG_IDS, PLATE_TAG_IDS
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The house the quadruped moves into.
HOME_XML = "models/home_world.xml"


#: The quadruped's packs in this world, Wh. A test's: small enough that a
#: day's test charges. The served pack is the real one (`model.PACK_WH`).
DEMO_WH = 20.0
#: The return-trip reserve, Wh: MEASURED on the quadruped over the house's
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
#: 3.656, carried as 3.7. A property of the floor plan, not of the pack.
RESERVE_WH = 3.7


def dock_pose() -> tuple[float, float, float]:
  """The dock's frame in the world, (x, y, yaw rad): its seat on the living
  room's floor, the board against the south wall (+x toward it). Clear of
  the couch's south face (y 0.45) by the approach's standoff: the robot
  comes in from 1.0 m behind the seat.
  The COMMISSIONED pose, which a robot lying on it is anchored to."""
  from pluggybot.home import world as home
  wall = home.HOUSE_Y[0] + home.WALL_HALF_T          # the wall's inner face
  back = dk.DEFAULT.board_x + 0.06                    # the board's post, behind it
  return 3.5, wall + back, -math.pi / 2


#: The tool rack's middle bay along the south wall, m: west of the dock
#: (x 3.5), its board spanning x 1.6..2.6 -- bay A, the east one, works
#: 1.1 m from the dock's axis, over the peer disc of a robot lying there --
#: and north of it the floor is clear to the couch's south face (y 0.3).
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

def _attach_rack(spec: mujoco.MjSpec, pose) -> None:
  """The tool rack, its tags' textures (the generator's `tags/`) and the
  tools hung on its bays -- and beside them on the same board the built-tool
  rail (`rack.BUILT`, issue #407), empty until the workshop hangs a tool."""
  from pluggybot.rack.coupling import BUILT_RACK_BODY
  for i in (*rk.RACK_TAG_IDS, *rk.BUILT.tag_ids):
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  x, y, yaw = pose
  defaults, tools = rk.tools_xml(pos=(x, y), yaw=yaw)
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><default>{defaults}</default><worldbody>'
    + rk.rack_xml(pos=(x, y), yaw=yaw, name=rk.RACK_BODY)
    + rk.rack_xml(rk.BUILT, pos=(x, y), yaw=yaw, name=BUILT_RACK_BODY) + tools
    + f"</worldbody><actuator>{rk.tool_actuators_xml()}</actuator></mujoco>")
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


def _attach_signs(spec: mujoco.MjSpec) -> None:
  """The lab's plate signs (issue #419; `activity.cage.plate_signs_xml`)
  and their tags' textures, where the house has a lab."""
  from pluggybot.activity import cage
  from pluggybot.home import world as home
  if not any(b.name == "lab_cage" for b in spec.worldbody.bodies):
    return
  for i in PLATE_TAG_IDS:
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  child = mujoco.MjSpec.from_string(
    "<mujoco><compiler angle=\"radian\"/><worldbody>"
    + cage.plate_signs_xml(home.LAB_CAGE_XY) + "</worldbody></mujoco>")
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


def _attach_board_tags(spec: mujoco.MjSpec) -> None:
  """The whiteboards' tags (issue #406; `tools.drawing.board_tags_xml`), a
  pair on the wall either side of each board the house has, and their
  textures."""
  from pluggybot.home import world as home
  from pluggybot.tools.drawing import Board, board_tags_xml
  names = [n for n in home.BOARDS if n in BOARD_TAG_IDS]
  if not names:
    return
  for i in (t for n in names for t in BOARD_TAG_IDS[n]):
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  body = "".join(board_tags_xml(n, Board.from_meta(home.BOARDS[n])) for n in names)
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>')
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


def _attach_area_tags(spec: mujoco.MjSpec) -> None:
  """The claw's and the census's areas' tags (issue #407; `home.areas`),
  and their textures."""
  from pluggybot.home import areas
  for i in areas.area_ids():
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><worldbody>{areas.area_tags_xml()}'
    '</worldbody></mujoco>')
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


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
  a second at `second_at` in its own livery, the dock, the rack, the lab's
  plate signs, the whiteboards' tags and the claw's and the census's areas'
  (#407). Kept by the lifecycle (`robot.world_spec`)."""
  spec = mujoco.MjSpec.from_file(path)
  _attach_quad(spec, first_at, "", None)
  if second_at is not None:
    _attach_quad(spec, second_at, second_prefix, second_rgba)
  _attach_dock(spec, dock_pose())
  _attach_rack(spec, rack_pose())
  _attach_signs(spec)
  _attach_board_tags(spec)
  _attach_area_tags(spec)
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
