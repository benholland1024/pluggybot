"""The home world with the humanoids in it (issue #559, stage 1): the house,
one humanoid or a pair, and the house's own fixtures -- the lab's plate
signs, the whiteboards' tags and the task areas' tags -- built at load from
`models/home_world.xml`, so there is one house; its charging station
(stage 2) and the storeroom's counter (stage 3).

⚠ THE ROBOTS ARE ATTACHED AFTER THE HOUSE, and the fixtures after them in
one order: the house's body order is the day's hash. Their free joints are
not at `qpos[0]`: everything reads a robot's joints by name (`RobotHandle`).
"""

import math

import mujoco

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.perception.tags import BOARD_TAG_IDS, PLATE_TAG_IDS
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The house the humanoids move into.
HOME_XML = "models/home_world.xml"
#: The pack in this world, Wh, a test's (`--pack demo`): small enough that a
#: day's test charges -- about a quarter of an hour on its feet at the 80 W
#: it draws standing. The served pack is the real one (`parts.PACK`).
DEMO_WH = 20.0
PACK_WH = hp.PACK.wh
#: The return-trip reserve, Wh: the walk to its station from the house's
#: worst place and its plug-in, with one failed, on the humanoid (`scripts/energy_spike.py --world
#: home_humanoid --reserve`, 30 Oct 2026):
#:
#:     travel   13.722 Wh over 44.62 m to the garden doorway (307.6 mWh/m:
#:              the quadruped's 60.5 five times over)
#:     dock      1.85 Wh for its last 6 m at that rate, and a plug-in -- 17-36 s
#:              walking in and reaching (`station_spike.py --approach`) at
#:              the 86.9 W it draws standing -- twice: 1.74 Wh
#:
#: ⚠ The spike's own dock leg gave up 6 m short ("no route over the floor
#: mapped so far": set down at the worst place off the truth, on the map
#: its explore left), so the leg is the travel's rate and the plug-in's time.
#: 17.3, carried as 17.5. A property of the floor plan, not of the pack.
RESERVE_WH = 17.5
#: THE PAIR COLLIDES (Ben, 2026-10-14, #455): each robot's colliding geoms
#: (`contype 1, conaffinity 0` -- they meet only the pairs its model names,
#: `model.LEG_GEOMS`/`ARM_GEOMS`) get a bit of their own and the other's as
#: their affinity, so each meets the other robot's body, never its own, and
#: the house as before. Bits the house never sets (its own are 1 and 2), so
#: no geom of the house's can change what it meets.
PAIR_BITS = ((8, 16), (16, 8))
#: The cameras' near plane, m, as in the counter's world (`evaluation.
#: counter.NEAR_M`): MuJoCo's own, scaled from the house's pinned extent, is
#: 0.37 m, and cut a hand near the head.
NEAR_M = 0.10


def _attach_humanoid(spec: mujoco.MjSpec, pos, prefix: str, rgba, bits=None) -> None:
  robot = hm.attachable()
  if rgba is not None and tuple(rgba) != CHASSIS_RGBA:
    paint(robot, rgba)
  if bits is not None:                       # `PAIR_BITS`
    own, other = bits
    for geom in robot.geoms:
      if geom.contype or geom.conaffinity:
        geom.contype, geom.conaffinity = geom.contype | own, geom.conaffinity | other
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), 0.0]
  spec.attach(robot, prefix=prefix, frame=frame)


def _attach_tags(spec: mujoco.MjSpec, ids, body_xml: str) -> None:
  """Tags' textures and materials (`tags/tag<i>.png`), then the fixture
  that wears them, attached to the world."""
  for i in ids:
    spec.add_texture(name=f"tagtex{i}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                     file=f"tags/tag{i}.png")
    mat = spec.add_material(name=f"tagmat{i}", specular=0.05, shininess=0.05,
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tagtex{i}"
  child = mujoco.MjSpec.from_string(
    f'<mujoco><compiler angle="radian"/><worldbody>{body_xml}</worldbody></mujoco>')
  spec.attach(child, prefix="", frame=spec.worldbody.add_frame())


def _attach_signs(spec: mujoco.MjSpec) -> None:
  """The lab's plate signs (issue #419; `activity.cage.plate_signs_xml`)
  and their tags' textures, where the house has a lab."""
  from pluggybot.activity import cage
  from pluggybot.home import world as home
  if not any(b.name == "lab_cage" for b in spec.worldbody.bodies):
    return
  _attach_tags(spec, PLATE_TAG_IDS, cage.plate_signs_xml(home.LAB_CAGE_XY))


def _attach_board_tags(spec: mujoco.MjSpec) -> None:
  """The whiteboards' tags (issue #406; `home.boards.board_tags_xml`), a
  pair on the wall either side of each board the house has, and their
  textures."""
  from pluggybot.home import world as home
  from pluggybot.home.boards import Board, board_tags_xml
  names = [n for n in home.BOARDS if n in BOARD_TAG_IDS]
  if not names:
    return
  _attach_tags(spec, (t for n in names for t in BOARD_TAG_IDS[n]),
               "".join(board_tags_xml(n, Board.from_meta(home.BOARDS[n])) for n in names))


def _attach_area_tags(spec: mujoco.MjSpec) -> None:
  """The task areas' tags (issue #407; `home.areas`), and their
  textures."""
  from pluggybot.home import areas
  _attach_tags(spec, areas.area_ids(), areas.area_tags_xml())


def station_pose() -> tuple[float, float, float]:
  """The charging station's frame in the world, (x, y, yaw rad): its face on
  the living room's south wall where the quadrupeds' dock lay (x 3.5), +x
  into the wall (`activity.station`'s frame). The COMMISSIONED pose, which
  a robot at it fixes its belief to."""
  from pluggybot.activity import station as st
  from pluggybot.home import world as home
  wall = home.HOUSE_Y[0] + home.WALL_HALF_T          # the wall's inner face
  return 3.5, wall + st.BOX_DEPTH_M, -math.pi / 2


def home_spec(first_at=(1.5, 0.5), second_at=None, second_prefix: str = "r2_",
              second_rgba=SECOND_CHASSIS_RGBA, path: str = HOME_XML) -> mujoco.MjSpec:
  """The home world's SPEC with a humanoid at `first_at` and, optionally, a
  second at `second_at` in its own livery, the two colliding with each other
  (`PAIR_BITS`), the lab's plate signs, the whiteboards' tags and the task
  areas' -- the house's fixtures, which every ray the robots cast meets. Kept
  by the lifecycle (`robot.world_spec`); a robot is stood up by its body
  (`HumanoidBody.start_at`)."""
  spec = mujoco.MjSpec.from_file(path)
  pair = second_at is not None
  _attach_humanoid(spec, first_at, "", None, PAIR_BITS[0] if pair else None)
  if pair:
    _attach_humanoid(spec, second_at, second_prefix, second_rgba, PAIR_BITS[1])
  _attach_signs(spec)
  _attach_board_tags(spec)
  _attach_area_tags(spec)
  from pluggybot.activity.station import attach_station
  attach_station(spec, station_pose(), roots=(ROBOT_ROOT,) + ((second_prefix + ROBOT_ROOT,)
                                                               if pair else ()))
  # ...and the storeroom's counter, its four jobs' props and its row of tags
  # (stage 3, `home.storeroom`)
  from pluggybot.home import storeroom
  storeroom.attach(spec)
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec
