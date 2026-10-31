"""The home world with the humanoids in it (issue #559, stage 1): the house,
one humanoid or a pair, and the house's own fixtures -- the lab's plate
signs, the whiteboards' tags and the task areas' tags -- built at load from
`models/home_world.xml` as `legs/world.py` builds the quadrupeds' house, so
there is one house.

NOT the quadruped's dock, rack or tools: they go with the quadruped (#559),
and a humanoid charges by a cable it plugs into itself (stage 2). Until then
a humanoid has nowhere to charge, and a served one flies the real pack
(`parts.PACK`, 778 Wh: hours on its feet) or an admin's `set_battery`.

⚠ THE ROBOTS ARE ATTACHED AFTER THE HOUSE, so their free joints are not at
`qpos[0]`: everything reads a robot's joints by name (`RobotHandle`).
"""

import math

import mujoco

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.legs import world as lw
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The house the humanoids move into.
HOME_XML = lw.HOME_XML
#: The pack in this world, Wh, a test's (`--pack demo`): the quadruped's
#: demo cell, about a quarter of an hour on its feet at the 80 W it draws
#: standing. The served pack is the real one (`parts.PACK`).
DEMO_WH = lw.DEMO_WH
PACK_WH = hp.PACK.wh
#: The return-trip reserve, Wh: the walk to its station from the house's
#: worst place and its plug-in, with one failed (`legs.world.RESERVE_WH`'s
#: rule), on the humanoid (`scripts/energy_spike.py --world home_humanoid
#: --reserve`, 30 Oct 2026):
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
#: THE PAIR COLLIDES, as the quadrupeds do (`legs.world.PAIR_BITS`): each
#: robot's colliding geoms (`contype 1, conaffinity 0` -- they meet only the
#: pairs its model names, `model.LEG_GEOMS`/`ARM_GEOMS`) get a bit of their
#: own and the other's as their affinity, so each meets the other robot's
#: body, never its own, and the house as before.
PAIR_BITS = lw.PAIR_BITS
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


def station_pose() -> tuple[float, float, float]:
  """The charging station's frame in the world, (x, y, yaw rad): its face on
  the living room's south wall where the quadrupeds' dock lies (x 3.5), +x
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
  lw._attach_signs(spec)
  lw._attach_board_tags(spec)
  lw._attach_area_tags(spec)
  from pluggybot.activity.station import attach_station
  attach_station(spec, station_pose(), roots=(ROBOT_ROOT,) + ((second_prefix + ROBOT_ROOT,)
                                                               if pair else ()))
  # ...and the storeroom's counter, its four jobs' props and its row of tags
  # (stage 3, `home.storeroom`)
  from pluggybot.home import storeroom
  storeroom.attach(spec)
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec
