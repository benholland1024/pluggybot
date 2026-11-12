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
from pluggybot.perception.cull import fix_static_inertials
from pluggybot.perception.tags import BOARD_TAG_IDS, PLATE_TAG_IDS
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The house the humanoids move into.
HOME_XML = "models/home_world.xml"
#: The pack in this world, Wh, a test's (`--pack demo`), never a part: small
#: enough that a day's test charges -- about 21 minutes on its feet at the
#: 93 W it draws standing -- and big enough that charged (`CHARGED`) it funds
#: the dearest job and the reserve both, so the job gate keeps the reserve
#: as on the served pack (`parts.PACK`): 29.7 of 29.4 Wh (#560: the cube
#: from the mall's station, 9.86, and the reserve, 19.5; 30 Wh was 27 of
#: 26.6 before them). At 20 it kept none, a branch the served pack never
#: takes (#580).
DEMO_WH = 33.0
PACK_WH = hp.PACK.wh
#: The return-trip reserve, Wh: the walk to the NEAREST of its stations
#: from the world's worst place, and its plug-in with one failed, on the
#: humanoid (`scripts/energy_spike.py --world home_humanoid --reserve`, a
#: pod, 3 Nov 2026; #560): from the park's west sidewalk through the park
#: to the mall's front door, 14.20 Wh in 301 s -- an upper bound, its walk
#: 54.8 m where the waypoints are 42.4 -- and from there `go_charge`, down
#: the concourse to the mall's station and plugged in, 4.38 Wh: 18.57; and a
#: plug-in tried again, 0.93 at the 93.1 W it draws standing: 19.5. A
#: property of the floor plan, not of the pack (17.86 to the one station on
#: #581's head).
RESERVE_WH = 19.5
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
  """The house's charging station's frame in the world, (x, y, yaw rad): its
  face on the living room's south wall where the quadrupeds' dock lay (x
  3.5), +x into the wall (`activity.station`'s frame). The COMMISSIONED
  pose, which a robot at it fixes its belief to."""
  return stations()["station"][0]


def stations() -> dict[str, tuple[tuple[float, float, float], tuple[int, int]]]:
  """Every charging station by its body's name: its frame in the world (as
  `station_pose`'s) and its tags. The house's, and #560's two -- the
  research center's at its lobby's south end, the mall's at its
  concourse's: each on a south wall, a robot facing it south, and each
  COMMISSIONED, as the house's is. Which one a robot charges at is its body's
  to choose (`Body.choose_station`)."""
  from pluggybot.activity import station as st
  from pluggybot.home import world as home
  from pluggybot.perception.tags import STATION_2_TAG_IDS, STATION_3_TAG_IDS, STATION_TAG_IDS

  tags = {"station": STATION_TAG_IDS, "station_facility": STATION_2_TAG_IDS,
          "station_mall": STATION_3_TAG_IDS}
  return {name: ((x, wall_y + home.WALL_HALF_T + st.BOX_DEPTH_M, -math.pi / 2), tags[name])
          for name, (x, wall_y) in home.STATION_WALLS.items()}


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
  roots = (ROBOT_ROOT,) + ((second_prefix + ROBOT_ROOT,) if pair else ())
  for name, (pose, tags) in stations().items():
    attach_station(spec, pose, name=name, roots=roots, tags=tags)
  # ...and the storeroom's counter, its four jobs' props and its row of tags
  # (stage 3, `home.storeroom`), and the restaurant's place's dishware, put
  # away until a job sets it out (#560, `home.restaurant`)
  from pluggybot.home import restaurant, storeroom
  storeroom.attach(spec)
  restaurant.attach(spec)
  fix_static_inertials(spec)
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec
