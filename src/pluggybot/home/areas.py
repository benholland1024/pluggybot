"""The house's task areas' tags (issue #407): where each hangs, and each
area's fixture for the places memory (`mapping.places`). The words a job
gives are `places.json`'s.

  the tower   a pair on the workshop corner's west wall, either side of the
              row its blocks are set out in (`home.TOWER_XY`), facing the room
  the bench   a pair on the bench's front, either side, facing the lab; its
              masses lie in front of it (`home.props`)
  the garden  one on its east fence, facing the house straight across the
              grass from the living room's doorway -- seen through the
              doorway from inside, and from anywhere in the garden; on the
              house's wall outside, a robot in the living room never saw it

...and the grown world's places (issue #560), hung at the navigation
camera's look, as the storeroom counter's row is (`storeroom.TAG_Z`):

  the restaurant   a pair on the dining room's east wall either side of the
                   pass, the expo line's counter in it
  the shop         a pair on its north wall, over the room
  the park         one on a sign in its middle, facing north-east -- toward
                   both its gates, a sign being read only from in front of it
  each robot's room and the study room
                   one on the wall facing its door

Each is a 120 mm tag, read from across a room (the boards' and the
plates'). Put in at load by the humanoids' world, after its robots
(`humanoid/world.py`): the house's body order is the day's hash.
"""

from __future__ import annotations

import math

from pluggybot.perception.tags import (AREA_TAG_SIZE, BENCH_TAG_IDS, GARDEN_TAG_IDS,
                                 PARK_TAG_IDS, RESTAURANT_TAG_IDS, ROOM_TAG_IDS,
                                 SHOP_TAG_IDS, STUDY_TAG_IDS, TOWER_TAG_IDS,
                                 plate_half_extent)

#: The tags' centres over the floor, m: the quadruped's nose camera's
#: height, where they were hung (#407) and stay.
TAG_Z = 0.37
#: ...and the grown world's (#560): the storeroom counter's row's, the
#: humanoid's navigation camera's look.
NEW_TAG_Z = 1.15
#: A pair's tags this far either side of its area's middle, m: outside the
#: row of props in front of it.
PAIR_HALF_M = 0.45
#: The park's sign (#560): where it stands, which way its face points, and
#: its post's and board's half-sizes, m.
PARK_SIGN_XY = (-1.0, -17.0)
PARK_SIGN_FACING = math.pi / 4
SIGN_POST_HALF, SIGN_BOARD_HALF_T = 0.03, 0.01


def area_tags() -> dict[str, list[tuple[int, float, float, float]]]:
  """Every area's tags as (tag, x, y, facing): each on its wall's face,
  facing out of it, a pair's left tag first as the robot faces them."""
  from pluggybot.home.props import BENCH_HALF
  from pluggybot.home import world as home
  row = home.TOWER_XY[1]
  wall = home.WING_X[0] + home.WALL_HALF_T                 # the workshop's west wall
  bx, by = home.LAB_BENCH_XY
  front = bx - BENCH_HALF[0]                                # the bench's front face
  fence = home.GARDEN_X[1] - home.WALL_HALF_T               # the east fence, inside
  door = sum(home.DOOR_GARDEN_Y) / 2.0                      # across from the doorway
  t = home.WALL_HALF_T
  pass_y = sum(home.EXPO_Y) / 2.0
  shop_x = (home.CONCOURSE_X[1] + home.BACK_X) / 2.0 - 0.45
  sx, sy = PARK_SIGN_XY
  board = SIGN_POST_HALF + 2 * SIGN_BOARD_HALF_T            # the board's face off the post's axis
  return {
    # facing the wall (west), the robot's left is south
    "tower": [(TOWER_TAG_IDS[0], wall, row[1] - PAIR_HALF_M, 0.0),
              (TOWER_TAG_IDS[1], wall, row[1] + PAIR_HALF_M, 0.0)],
    # facing the bench (east), its left is north
    "bench": [(BENCH_TAG_IDS[0], front, by + PAIR_HALF_M, math.pi),
              (BENCH_TAG_IDS[1], front, by - PAIR_HALF_M, math.pi)],
    "garden": [(GARDEN_TAG_IDS[0], fence, door, math.pi)],
    # ---- the grown world (#560) ---------------------------------------------
    # facing the pass (east), its left is north: 2 m either side of it, on
    # the wall north of it and between it and the kitchen's staff door
    "restaurant": [(RESTAURANT_TAG_IDS[0], home.BACK_X - t, pass_y + 2.0, math.pi),
                   (RESTAURANT_TAG_IDS[1], home.BACK_X - t, pass_y - 2.0, math.pi)],
    # facing the shop's north wall, its left is west
    "shop": [(SHOP_TAG_IDS[0], shop_x, home.SHOP_Y[1] - t, -math.pi / 2),
             (SHOP_TAG_IDS[1], shop_x + 0.9, home.SHOP_Y[1] - t, -math.pi / 2)],
    "park": [(PARK_TAG_IDS[0], sx + board * math.cos(PARK_SIGN_FACING),
              sy + board * math.sin(PARK_SIGN_FACING), PARK_SIGN_FACING)],
    "room_west": [(ROOM_TAG_IDS[0], (home.ROOMS_X[0] + home.ROOM_DIV_X) / 2,
                   home.ROOMS_Y[0] + t, math.pi / 2)],
    "room_east": [(ROOM_TAG_IDS[1], (home.ROOM_DIV_X + home.ROOMS_X[1]) / 2,
                   home.ROOMS_Y[0] + t, math.pi / 2)],
    "study_room": [(STUDY_TAG_IDS[0], home.STUDY_X[0] + t, home.STUDY_TABLE[1], 0.0)],
  }


def tag_z(area: str) -> float:
  """An area's tags' height, m (`TAG_Z`; the grown world's `NEW_TAG_Z`)."""
  return TAG_Z if area in ("tower", "bench", "garden") else NEW_TAG_Z


def park_sign_xml() -> str:
  """The park's sign (#560): a post, and a board on it facing
  `PARK_SIGN_FACING`, its tag on the board's face (`area_tags_xml`)."""
  half = plate_half_extent(AREA_TAG_SIZE)
  b = half + 0.01
  top = NEW_TAG_Z + b
  p, t = SIGN_POST_HALF, SIGN_BOARD_HALF_T
  x, y = PARK_SIGN_XY
  f = PARK_SIGN_FACING
  return (f'<body name="park_sign" pos="{x:.4f} {y:.4f} 0" '
          f'quat="{math.cos(f / 2):.6f} 0 0 {math.sin(f / 2):.6f}">'
          f'<geom name="park_sign_post" type="box" size="{p:.4f} {p:.4f} {top / 2:.4f}" '
          f'pos="0 0 {top / 2:.4f}" rgba="0.32 0.33 0.36 1"/>'
          f'<geom name="park_sign_board" type="box" size="{t:.4f} {b:.4f} {b:.4f}" '
          f'pos="{p + t:.4f} 0 {NEW_TAG_Z:.4f}" rgba="0.95 0.95 0.95 1"/></body>')


def area_tags_xml() -> str:
  """The areas' tags as worldbody MJCF: each a thin plate a millimetre off
  its wall, its texture the tag's (`perception.tags.asset_xml`) -- and the
  park's sign it hangs on."""
  half = plate_half_extent(AREA_TAG_SIZE)
  out = [park_sign_xml()]
  for area, tags in area_tags().items():
    z = tag_z(area)
    for tag, x, y, facing in tags:
      px, py = x + 0.002 * math.cos(facing), y + 0.002 * math.sin(facing)
      out.append(f'<body name="{area}_tag{tag}" pos="{px:.4f} {py:.4f} {z:.4f}" '
                 f'quat="{math.cos(facing / 2):.6f} 0 0 {math.sin(facing / 2):.6f}">'
                 f'<geom name="{area}_tag{tag}" type="box" size="0.001 {half:.4f} {half:.4f}" '
                 f'contype="0" conaffinity="0" material="tagmat{tag}"/></body>')
  return "".join(out)


def area_fixtures():
  """Each pair as one fixture (`mapping.places.Fixture`): in its frame x
  along the robot's left as it faces them, y out of their face -- each tag
  half the pair's own spacing from its middle."""
  from pluggybot.mapping.places import Fixture
  out = []
  for area, tags in area_tags().items():
    if len(tags) == 2:
      (left, lx, ly, _), (right, rx, ry, _) = tags
      half = math.hypot(lx - rx, ly - ry) / 2.0
      out.append(Fixture(layout={left: (half, 0.0), right: (-half, 0.0)},
                         facing=math.pi / 2))
  return tuple(out)


def area_ids() -> tuple[int, ...]:
  return tuple(t for tags in area_tags().values() for t, *_ in tags)
