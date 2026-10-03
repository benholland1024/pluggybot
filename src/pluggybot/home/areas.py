"""The task areas' tags the world puts up for the claw and the census
(issue #407): where each hangs, and each area's fixture for the places
memory (`mapping.places`). The words a job gives are `places.json`'s.

  the tower   a pair on the workshop corner's west wall, either side of the
              row its blocks are set out in (`home.TOWER_XY`), facing the room
  the bench   a pair on the bench's front, either side, facing the lab; its
              masses are set out in front of it (`challenge.bench`)
  the garden  one on its east fence, facing the house straight across the
              grass from the living room's doorway -- seen through the
              doorway from inside, and from anywhere in the garden; on the
              house's wall outside, a robot in the living room never saw it.
              The census finds the garden by it

Each is a 120 mm tag at the nose camera's height, read from across a room
(the boards' and the plates'). Put in at load by the quadruped's world,
after its robots (`legs/world.py`): the house's body order is the day's
hash.
"""

from __future__ import annotations

import math

from pluggybot.rack.tags import (AREA_TAG_SIZE, BENCH_TAG_IDS, GARDEN_TAG_IDS,
                                 TOWER_TAG_IDS, plate_half_extent)

#: The tags' centres over the floor, m: the nose camera's height standing.
TAG_Z = 0.37
#: A pair's tags this far either side of its area's middle, m: outside the
#: row of props in front of it, and both in the nose camera's view from
#: where the search looks (`legs.claw.SEARCH_OUT_M`).
PAIR_HALF_M = 0.45


def area_tags() -> dict[str, list[tuple[int, float, float, float]]]:
  """Every area's tags as (tag, x, y, facing): each on its wall's face,
  facing out of it, a pair's left tag first as the robot faces them."""
  from pluggybot.challenge.bench import BENCH_HALF
  from pluggybot.home import world as home
  row = home.TOWER_XY[1]
  wall = home.WING_X[0] + home.WALL_HALF_T                 # the workshop's west wall
  bx, by = home.LAB_BENCH_XY
  front = bx - BENCH_HALF[0]                                # the bench's front face
  fence = home.GARDEN_X[1] - home.WALL_HALF_T               # the east fence, inside
  door = sum(home.DOOR_GARDEN_Y) / 2.0                      # across from the doorway
  return {
    # facing the wall (west), the robot's left is south
    "tower": [(TOWER_TAG_IDS[0], wall, row[1] - PAIR_HALF_M, 0.0),
              (TOWER_TAG_IDS[1], wall, row[1] + PAIR_HALF_M, 0.0)],
    # facing the bench (east), its left is north
    "bench": [(BENCH_TAG_IDS[0], front, by + PAIR_HALF_M, math.pi),
              (BENCH_TAG_IDS[1], front, by - PAIR_HALF_M, math.pi)],
    "garden": [(GARDEN_TAG_IDS[0], fence, door, math.pi)],
  }


def area_tags_xml() -> str:
  """The areas' tags as worldbody MJCF: each a thin plate a millimetre off
  its wall, its texture the tag's (`rack.tags.asset_xml`)."""
  half = plate_half_extent(AREA_TAG_SIZE)
  out = []
  for area, tags in area_tags().items():
    for tag, x, y, facing in tags:
      px, py = x + 0.002 * math.cos(facing), y + 0.002 * math.sin(facing)
      out.append(f'<body name="{area}_tag{tag}" pos="{px:.4f} {py:.4f} {TAG_Z:.4f}" '
                 f'quat="{math.cos(facing / 2):.6f} 0 0 {math.sin(facing / 2):.6f}">'
                 f'<geom name="{area}_tag{tag}" type="box" size="0.001 {half:.4f} {half:.4f}" '
                 f'contype="0" conaffinity="0" material="tagmat{tag}"/></body>')
  return "".join(out)


def area_fixtures():
  """Each pair as one fixture (`mapping.places.Fixture`): in its frame x
  along the robot's left as it faces them, y out of their face."""
  from pluggybot.mapping.places import Fixture
  out = []
  for area, tags in area_tags().items():
    if len(tags) == 2:
      (left, *_), (right, *_) = tags
      out.append(Fixture(layout={left: (PAIR_HALF_M, 0.0), right: (-PAIR_HALF_M, 0.0)},
                         facing=math.pi / 2))
  return tuple(out)


def area_ids() -> tuple[int, ...]:
  return tuple(t for tags in area_tags().values() for t, *_ in tags)
