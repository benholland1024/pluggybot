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

import mujoco

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.legs import world as lw
from pluggybot.robot import CHASSIS_RGBA, SECOND_CHASSIS_RGBA, paint

#: The house the humanoids move into.
HOME_XML = lw.HOME_XML
#: The pack in this world, Wh, a test's (`--pack demo`): the quadruped's
#: demo cell, about a quarter of an hour on its feet at the 80 W it draws
#: standing. The served pack is the real one (`parts.PACK`).
DEMO_WH = lw.DEMO_WH
PACK_WH = hp.PACK.wh
#: The return-trip reserve, Wh: nothing yet. It is the walk to the nearest
#: charging station, measured once there is one (stage 2); with none, the
#: pack running out is the `flat` death.
RESERVE_WH = 0.0
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
  spec.visual.map.znear = NEAR_M / spec.stat.extent
  return spec
