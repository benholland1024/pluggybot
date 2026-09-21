"""Home-world generator (issue #6): a house with a garden, from one source.

The park doctrine from the PluggyWorld design doc, applied at house scale:
GENERATE the world, never hand-write it. One run of this module emits

  models/home_world.xml        the physics world MuJoCo loads
  models/home_world.meta.json  everything the physics does NOT need:
                               per-body visual hints for the website's
                               parametric renderer, zone rectangles, spawn
                               poses, board specs -- machine-readable, so
                               the scene transpiler and the scripts read
                               the same source this file wrote

and the two can never drift apart, because neither is hand-edited.

Layout (meters; y+ north, x+ east). Two houses, a loop street and a fence
(issue #215) around the property issue #68 authored -- these diagrams are
the plan, not a picture of it. The whole plot:

        x=-16.5 -13.5 -12       -5  -2        5      10 11.5 14.5 16  19  22       28 29.5 32.5
  y=10.5 ╔══════════════════════════════ fence ═══════════════════════════════════════════╗
         ║               street_north (the loop)                                          ║
  y=7.5  ║   ┌────────── sidewalk_north ───────────┐   ┌───── sidewalk_2_north ──────┐   ║
  y=6    ║   │ ┌─────────┬───┬─────────┬─fence─┐ s │ s │ s ┌─fence─┬───┬───────────┐ │   ║
         ║ s │ │ kitchen │   │ bedroom │       │ i │ t │ i │       │ l │    lab    │ │ s ║
  y=3.1  ║ t │ │         │ H │         │garden ╪ d │ r │ d ╪garden ╪ o ╪  [cage]   │ │ t ║
  y=2.5  ║ r │ ├─────────┤ A ├──wall───┤       │ e │ e │ e │  _2   │ b │ [bench]   │ │ r ║
  y=0.7  ║ e │ │workshop │ L │  living ╪ [plt] │ w │ e │ w │       │ b ├───────────┤ │ e ║
         ║ e │ │         │ L │  [rack] │       │ a │ t │ a │       │ y │           │ │ e ║
  y=-2   ║ t │ │         │ L ├─────────┴───────┤ l │   │ l │       │   │           │ │ t ║
  y=-3   ║   │ │         │ L │ garden_south    │ k │   │ k │       │   ╪   store   │ │   ║
  y=-6   ║   │ │         │▓▓▓│                 │   │   │   │       │   │           │ │   ║
         ║   │ └─────────┴───┴─────────────────┘   │   │   └───────┴───┴───────────┘ │   ║
  y=-7.5 ║   └────────── sidewalk_south ───────────┘   └───── sidewalk_2_south ──────┘   ║
         ║               street_south (the loop)                                          ║
  y=-10.5╚═════════════════════════════════════════════════════════════════════════════════╝

  The middle street (x 11.5..14.5) is the one #68 drew, with its invisible
  ends now joined to a RING: the loop runs round both houses and the middle
  street runs on through the sidewalk to meet it, so each property has its
  own ring of sidewalk, and an unbroken fence closes the world. The two front gates face each other across the street
  at y=3.1. The second house is a lobby running north-south and two rooms:
  the LAB, the experiment zone (#226 the mouse, #227 the bench: its props
  are generated here, scenery until their activities land), and a STORE.

The first house in detail, as #68 authored it:

        x=-12        -5  -2          5        10  11.5  14.5
  y=6    ┌────────────┬───┬───────────┬──fence──┐  ┃     ┃
         │  kitchen   │   │  bedroom  │         │  ┃     ┃
         │   7 x 4    d S │  [wb_b]   │         │  ┃     ┃
  y=2.5  ├────────────┤ T ├───wall────┤ garden  │  ┃  s  ┃
         │            │ A │           │         │  ┃  i  ┃
         │            │ I │  living   ╪ GATE    ╪  ┃  d  ┃  street
  y=1    │            d R │  [wb_a]   │  y=3.1  │  ┃  e  ┃
         │  workshop  │ H │           │ [plate] │  ┃  w  ┃
         │   7 x 8    │ A │ ┌──rack──┐│         │  ┃  a  ┃
  y=-2   │            │ L ├─┴────────┴┴─────────┤  ┃  l  ┃
         │            │ L │                     │  ┃  k  ┃
         │            │   │   garden, wrapping  │  ┃     ┃
  y=-4   │            │   │   south of the house│  ┃     ┃
         │            ├───┤                     │  ┃     ┃
         │            │▓▓▓│                     │  ┃     ┃
  y=-6   └────────────┴───┴───────fence─────────┘  ┃     ┃

   d = doorway   ╪ = the OPEN street doorway (issue #93: the gate is
   gone; the plate turns on a light)   ▓ = staircase (3 x 3, solid box)

⚠ The kitchen/workshop divider is at y=2.0 (the ZONES table's number), not
the y=2.5 the diagram's left edge sits near -- 2.5 is the living/bedroom
divider on the other side of the hall, and the two are unrelated.

⚠ The living<->hall doorway is at y -1.0..0.0 FOR A REASON: `whiteboard_a`
is mounted on the living room's west wall at y=1.0, facing east into the
room. A doorway anywhere near y=1.0 deletes a drawing surface.

⚠ The south fence meets the HALL at (-2,-6), as drawn. Issue #68 flags the
alternative -- swapping hall and workshop in the wing's southern half so the
fence meets the workshop -- as a question rather than a decision, so this
builds what was drawn.

The staircase is ONE SOLID BOX, 0.9 m tall, and that is a sensing decision
rather than a modelling shortcut: the LIDAR rides at 0.223 m, so real risers
at ~0.18 m would be solid geometry the robot cannot see -- a wall it maps as
open floor. The browser draws the flight (hint `stairs`, frozen in #66).

Visual hints ride in the SIDECAR, never in geom colors: the robot's cameras
render rgba, so color-as-encoding would leak into the tag detector's frames
and couple perception to the website's art direction (the design doc's
explicit warning). Wall/floor colors here are chosen for the ROBOT's world
-- muted, house-like -- and the website is free to ignore them entirely.

Plants and the floor/ground overlays are contype=0 conaffinity=0: they are
VISUAL bodies (carrying hints for the renderer, honest color for the
cameras) that the physics never collides with, so the mission's navigation
is exactly as hard as the walls make it and not an inch harder.
"""

import json
import math

from pluggybot.rack.coupling import (
  BUILT_RACK_BODY, CHARGE_BAY_Y, HUB_STATION_YS, rack_and_modules_xml,
  rack_frame_to_world,
  claw_actuator_xml, dispenser_actuator_xml, pen_actuator_xml,
)
from pluggybot.activity.plate import (
  plate_light_xml,
)
from pluggybot.activity.cage import PLATE_NAMES, cage_xml
from pluggybot.challenge.bench import bench_xml
from pluggybot.rack.tags import BLOCK_TAG_IDS, asset_xml, write_tag_pngs

# ---- layout constants (the one source) --------------------------------------
# The ORIGIN sits inside the living room, like room_1's: the robot model
# spawns at (0,0) before any script places it, and a world whose origin is
# inside a wall greets every bare `MjData` with a chassis wedged in masonry
# (the first cut of this file did exactly that -- 78 contacts at settle).
# ⚠ HOUSE_X/HOUSE_Y are the ORIGINAL two-room block (living + bedroom) and
# are deliberately unchanged by issue #68 -- the rack, both whiteboards, both
# spawns and every measured energy figure are anchored to them, so moving them
# would re-price the world for no layout reason. The property grew AROUND
# them: a wing to the west, a garden wrapping south, a street to the east.
HOUSE_X = (-2.0, 5.0)         # indoor extent of the original block
HOUSE_Y = (-2.0, 6.0)
GARDEN_X = (5.0, 10.0)        # the EAST garden, beside the house
DIV_Y = 2.5                   # living/bedroom divider
DOOR_DIV_X = (1.0, 2.0)       # doorway in the divider
DOOR_GARDEN_Y = (0.2, 1.2)    # doorway from the living room to the garden

# ---- the property, issue #68 ------------------------------------------------
# The whole plot is x -12..14.5, y -6..6 -- 26.5 x 12 m, against the two-room
# 7 x 8 m it replaced. `ZONES` below is the source of truth for what is where;
# these are the edges the geometry is built from.
PROPERTY_Y = (-6.0, 6.0)      # every wing and the street share this extent
WING_X = (-12.0, -5.0)        # kitchen (north) + workshop (south)
HALL_X = (-5.0, -2.0)         # the spine joining the wing to the house
KITCHEN_Y = (2.0, 6.0)        # 7 x 4
WORKSHOP_Y = (-6.0, 2.0)      # 7 x 8
GARDEN_SOUTH_Y = (-6.0, -2.0) # the garden wrapping south, x HOUSE_X[0]..GARDEN_X[1]
SIDEWALK_X = (10.0, 11.5)
STREET_X = (11.5, 14.5)


# The three new doorways. All 1.0 m, which is `_wall_run`'s gap and comfortably
# past the 0.6 m `test_doorways_are_wide_enough_to_drive_through` demands: a
# doorway the inflated planning mask refuses is a wall, and the mask inflates
# by 0.35 m either side.
DOOR_HALL_Y = (-1.0, 0.0)     # living <-> hall, in the house's west wall
DOOR_KITCHEN_Y = (3.5, 4.5)   # hall <-> kitchen, in the wing's east wall
DOOR_WORKSHOP_Y = (0.5, 1.5)  # hall <-> workshop, same wall

# The staircase: ONE SOLID BOX filling the hall's south end. See the module
# docstring -- 0.9 m so the LIDAR sees it, and not modelled as steps because
# real risers are below the beam and would map as open floor.
STAIRS_X = (-5.0, -2.0)
STAIRS_Y = (-6.0, -3.0)
STAIRS_HALF_H = 0.45

WALL_HALF_T = 0.02
WALL_HALF_H = 0.60            # 1.2 m walls, like room_1
FENCE_HALF_H = 0.45           # 0.9 m garden fence: lidar (0.223 m) sees it
#: How far the drawn ground reaches past the property on every side. A metre,
#: so the fence line is not the literal edge of the visible world.
GROUND_MARGIN = 1.0

# The rack, against the living room's south wall, facing north into the room.
HOME_RACK_POS = (0.5, HOUSE_Y[0] + WALL_HALF_T)
HOME_RACK_YAW = 90.0

# Whiteboards: wall-mounted drawing surfaces (the milestone-8 board port).
# `heading` is the robot's heading when squared up to the board; `half` is
# (depth, width, height) in the board's own frame. Geom names are what
# `pen_on_board` checks contact against.
BOARDS = {
  "whiteboard_a": {                       # west wall of the living room
    "geom": "board",
    "pos": (HOUSE_X[0] + WALL_HALF_T + 0.010, 1.0, 0.30),
    "half": (0.010, 0.16, 0.13),
    "heading": math.pi,                   # board faces +x; robot heads -x
  },
  "whiteboard_b": {                       # north wall of the bedroom
    "geom": "board_b",
    "pos": (0.5, HOUSE_Y[1] - WALL_HALF_T - 0.010, 0.30),
    "half": (0.010, 0.16, 0.13),
    "heading": math.pi / 2,               # board faces -y; robot heads +y
  },
}

PLANTS = ((6.5, -0.8), (8.8, 1.0), (7.6, 4.5), (9.2, 5.2))

# The reference ACTIVITY (issue #8): a pressure plate just inside the garden
# doorway, turning on a garden light beside it (issue #93 -- it used to latch
# a GATE in the east fence, until #68 put the street behind that gate and
# broke the gate's own design rule, "blocks no route the robot needs"). The
# light keeps the activity's whole pattern -- sensed criterion, hysteresis,
# latch, rgba toggle, telemetry -- while blocking nothing, ever, by
# construction.
PLATE_XY = (5.7, 0.7)
#: Centre of the 1 m OPEN doorway in the east fence -- the way to the
#: sidewalk and street. It was the gate's position (issue #93 removed the
#: panel), and it keeps the exact same geometry so `HOME_WORST_RETURN_PATH`,
#: the committed recordings' fence segmentation and every measured route
#: survive the change: what left is the panel, not the gap.
STREET_DOOR_Y = 3.1
STREET_DOOR_HALF = 0.50

# ---- the second house, the loop and the fence (issue #215) -------------------
# Across the middle street, a second property of the same height, and around
# BOTH a sidewalk band and a street ring inside one fence. Every edge here is
# derived from the two above so the ring follows the block if it grows.
SIDEWALK_2_X = (14.5, 16.0)    # the middle street's east pavement
PLOT_2_X = (16.0, 28.0)        # the second property, edge to edge
GARDEN_2_X = (16.0, 19.0)      # its front garden, the whole height
HOUSE_2_X = (19.0, 28.0)       # the second house
LOBBY_X = (19.0, 22.0)         # its hall, running north-south like ours
LAB_X = (22.0, 28.0)           # the experiment zone, north half...
LAB_Y = (0.0, 6.0)
STORE_Y = (-6.0, 0.0)          # ...and a storeroom, empty, south half
#: The second house's doorways, all 1.0 m like the first's. The front gate
#: sits on the SAME line as ours (`STREET_DOOR_Y`), so the two face each
#: other across the street; the lab's and the store's are off the lobby.
DOOR_GARDEN_2_Y = (STREET_DOOR_Y - 0.5, STREET_DOOR_Y + 0.5)   # street -> garden_2
DOOR_LOBBY_Y = (STREET_DOOR_Y - 0.5, STREET_DOOR_Y + 0.5)      # garden_2 -> lobby
DOOR_LAB_Y = (2.5, 3.5)        # lobby -> lab
DOOR_STORE_Y = (-3.5, -2.5)    # lobby -> store
#: The block both properties make, the sidewalk band around it, and the
#: street ring around that. The ring is 3 m like the middle street.
BLOCK_X = (WING_X[0], PLOT_2_X[1])
BLOCK_Y = PROPERTY_Y
PAVEMENT_W = 1.5
LOOP_W = 3.0
PAVEMENT_X = (BLOCK_X[0] - PAVEMENT_W, BLOCK_X[1] + PAVEMENT_W)
PAVEMENT_Y = (BLOCK_Y[0] - PAVEMENT_W, BLOCK_Y[1] + PAVEMENT_W)
LOOP_X = (PAVEMENT_X[0] - LOOP_W, PAVEMENT_X[1] + LOOP_W)
LOOP_Y = (PAVEMENT_Y[0] - LOOP_W, PAVEMENT_Y[1] + LOOP_W)

#: The plot, tiled. Every zone is ONE RECTANGLE and together they cover the
#: property exactly once -- `tests/test_home_world.py` checks both halves of
#: that. The garden is the reason `garden_south` exists: issue #68 draws it
#: L-shaped, and an L is not a rect, so it is two zones sharing a `kind`
#: rather than one zone with a hole in the middle of its bounding box.
#:
#: ⚠ `garden` is still the EAST rectangle alone, and the census still surveys
#: it. All four plants are inside it and `true_count` counts by prefix WITHIN
#: the zone rect, so the census's hidden answer is unchanged at 4 -- which is
#: what keeps the committed recording and `test_the_home_fixture_shows_the_
#: census_answer` describing the same world they always did.
#:
#: The loop (issue #215) is four `street_*` legs, the long ones spanning the
#: corners. The MIDDLE street runs on through the sidewalk band to meet it
#: at both ends, so the band is broken there and each property has its own
#: ring of sidewalk: its front (`sidewalk`, `sidewalk_2`), a north and a
#: south strip, and its far side (`sidewalk_west`, `sidewalk_east`). The
#: first nine zones keep #68's names and order; `street` is longer.
#:
#: ⚠ A ROOM names its BUILDING (`protocol.BUILDINGS`): which house it is in,
#: a fact the site paints each building's walls by. Everything outdoors
#: names none.
ZONES = (
  {"name": "kitchen", "kind": "room", "building": "house",
   "min": [WING_X[0], KITCHEN_Y[0]], "max": [WING_X[1], KITCHEN_Y[1]]},
  {"name": "workshop", "kind": "room", "building": "house",
   "min": [WING_X[0], WORKSHOP_Y[0]], "max": [WING_X[1], WORKSHOP_Y[1]]},
  {"name": "hall", "kind": "room", "building": "house",
   "min": [HALL_X[0], PROPERTY_Y[0]], "max": [HALL_X[1], PROPERTY_Y[1]]},
  {"name": "living", "kind": "room", "building": "house",
   "min": [HOUSE_X[0], HOUSE_Y[0]], "max": [HOUSE_X[1], DIV_Y]},
  {"name": "bedroom", "kind": "room", "building": "house",
   "min": [HOUSE_X[0], DIV_Y], "max": [HOUSE_X[1], HOUSE_Y[1]]},
  {"name": "garden", "kind": "garden",
   "min": [GARDEN_X[0], HOUSE_Y[0]], "max": [GARDEN_X[1], HOUSE_Y[1]]},
  {"name": "garden_south", "kind": "garden",
   "min": [HOUSE_X[0], GARDEN_SOUTH_Y[0]], "max": [GARDEN_X[1], GARDEN_SOUTH_Y[1]]},
  {"name": "sidewalk", "kind": "outdoor",
   "min": [SIDEWALK_X[0], PROPERTY_Y[0]], "max": [SIDEWALK_X[1], PROPERTY_Y[1]]},
  {"name": "street", "kind": "outdoor",
   "min": [STREET_X[0], PAVEMENT_Y[0]], "max": [STREET_X[1], PAVEMENT_Y[1]]},
  # ---- the second house (issue #215) ----------------------------------------
  {"name": "sidewalk_2", "kind": "outdoor",
   "min": [SIDEWALK_2_X[0], PROPERTY_Y[0]], "max": [SIDEWALK_2_X[1], PROPERTY_Y[1]]},
  {"name": "garden_2", "kind": "garden",
   "min": [GARDEN_2_X[0], PROPERTY_Y[0]], "max": [GARDEN_2_X[1], PROPERTY_Y[1]]},
  {"name": "lobby", "kind": "room", "building": "facility",
   "min": [LOBBY_X[0], PROPERTY_Y[0]], "max": [LOBBY_X[1], PROPERTY_Y[1]]},
  {"name": "lab", "kind": "room", "building": "facility",
   "min": [LAB_X[0], LAB_Y[0]], "max": [LAB_X[1], LAB_Y[1]]},
  {"name": "store", "kind": "room", "building": "facility",
   "min": [LAB_X[0], STORE_Y[0]], "max": [LAB_X[1], STORE_Y[1]]},
  # ---- each property's ring of sidewalk, and the loop -----------------------
  {"name": "sidewalk_north", "kind": "outdoor",
   "min": [PAVEMENT_X[0], BLOCK_Y[1]], "max": [STREET_X[0], PAVEMENT_Y[1]]},
  {"name": "sidewalk_2_north", "kind": "outdoor",
   "min": [STREET_X[1], BLOCK_Y[1]], "max": [PAVEMENT_X[1], PAVEMENT_Y[1]]},
  {"name": "sidewalk_south", "kind": "outdoor",
   "min": [PAVEMENT_X[0], PAVEMENT_Y[0]], "max": [STREET_X[0], BLOCK_Y[0]]},
  {"name": "sidewalk_2_south", "kind": "outdoor",
   "min": [STREET_X[1], PAVEMENT_Y[0]], "max": [PAVEMENT_X[1], BLOCK_Y[0]]},
  {"name": "sidewalk_west", "kind": "outdoor",
   "min": [PAVEMENT_X[0], BLOCK_Y[0]], "max": [BLOCK_X[0], BLOCK_Y[1]]},
  {"name": "sidewalk_east", "kind": "outdoor",
   "min": [BLOCK_X[1], BLOCK_Y[0]], "max": [PAVEMENT_X[1], BLOCK_Y[1]]},
  {"name": "street_north", "kind": "outdoor",
   "min": [LOOP_X[0], PAVEMENT_Y[1]], "max": [LOOP_X[1], LOOP_Y[1]]},
  {"name": "street_south", "kind": "outdoor",
   "min": [LOOP_X[0], LOOP_Y[0]], "max": [LOOP_X[1], PAVEMENT_Y[0]]},
  {"name": "street_west", "kind": "outdoor",
   "min": [LOOP_X[0], PAVEMENT_Y[0]], "max": [PAVEMENT_X[0], PAVEMENT_Y[1]]},
  {"name": "street_east", "kind": "outdoor",
   "min": [PAVEMENT_X[1], PAVEMENT_Y[0]], "max": [LOOP_X[1], PAVEMENT_Y[1]]},
)

#: The loop's four legs, in order round it -- what `tests/test_home_world.py`
#: walks to check the ring is closed.
LOOP_ZONES = ("street_north", "street_east", "street_south", "street_west")

#: THE TOWER'S BLOCKS (issue #207; challenge/stack.py): three 26 mm cubes
#: on the workshop floor, a block's throw apart, in the room's south-west
#: corner -- 1 m off both walls, 3 m from the workshop spawn and 6 m from
#: its door, on no route the robot needs. The lidar plane is 22 cm up and
#: does not see them; the depth camera does. The challenge is graded
#: wherever the tower ends up standing, so this is where it STARTS, not
#: where it has to be built.
TOWER_XY = ((-11.0, -5.0), (-11.0, -4.75), (-11.0, -4.5))

#: THE LAB'S PROPS (issue #215; activity/cage.py, challenge/bench.py). The
#: cage against the lab's north wall, centred, its three plates a row in
#: front of it; the bench against the east wall with its two masses on the
#: floor a metre in front. The lab's doorway is on its west wall, so the
#: robot enters facing the bench and turns north for the cage: nothing
#: stands on the line from the door to either.
LAB_CAGE_XY = ((LAB_X[0] + LAB_X[1]) / 2, LAB_Y[1] - WALL_HALF_T - 0.40)
LAB_BENCH_XY = (LAB_X[1] - WALL_HALF_T - 0.30, 1.5)

SPAWNS = {
  "start": [1.5, 0.5, math.pi / 2],       # living room, facing the bedroom
  "garden": [7.5, 2.0, 0.0],
  # The new wing, for scripts that want to start somewhere other than home.
  # `start` is deliberately still the living room: it is where the rack is,
  # and every mission test measures from it.
  "hall": [-3.5, 1.0, 0.0],               # facing the living-room doorway
  "kitchen": [-8.5, 4.0, 0.0],
  "workshop": [-8.5, -2.0, 0.0],
  # The second house (issue #215): its hall, and the lab facing its cage.
  "lobby": [20.5, 0.0, math.pi / 2],
  "lab": [25.0, 2.0, math.pi / 2],
}

# Occupancy-grid bounds for this world (HubMission takes them as a
# parameter; room_hub keeps its historical defaults).
#
# ⚠ It used to reach a metre PAST the drawn floor (x_max 11.0 against a
# shared 10.0 slab), which issue #67 documented as deliberate and safe:
# unknown space is never traversable, so a cell over nothing can never be
# planned into. That is still true and still pinned
# (`tests/test_world_budget.py::test_unknown_space_is_never_driveable`), but
# the discrepancy itself is GONE -- issue #68 generates the floor from the
# same layout and the same margin these bounds use, so the map and the ground
# a visitor sees now coincide to the millimetre.
#: ⚠ GROWN FOR THE LOOP (issue #215): 51 x 23 m at 5 cm is 1020 x 460 =
#: 469,200 cells, against `occupancy_grid.MAX_CELLS` of 750,000 (the
#: measured table is at the constant; the cap moved with it). It was 159,600
#: for #68's property and 56,000 before that. The margin past the fence on
#: every side is what lets a scan that overshoots a boundary land somewhere
#: rather than being silently clipped.
GRID_BOUNDS = (LOOP_X[0] - GROUND_MARGIN, LOOP_Y[0] - GROUND_MARGIN,
               LOOP_X[1] + GROUND_MARGIN, LOOP_Y[1] + GROUND_MARGIN)

# Battery tuning for the bigger floor plan (issue #6 "rides along"). The
# reserve is absolute energy, per the milestone-7 lesson: it must cover the
# WORST return trip to the rack plus one failed press-and-retry. Measured
# empirically (see docs/PluggyPlan.md): a full living-room crossing plus
# charge approach costs ~0.3 Wh at cruise draw, so the room_hub reserve of
# 0.35 is too thin here; the demo cell grows with it so one explore + one
# errand still runs the pack down and the loop still has to charge.
#: MEASURED on the plan with the loop (issue #215; before it #70 / #84 /
#: #34 -- `scripts/energy_spike.py --reserve`, runnable again whenever a
#: wall moves). The worst-case return -- the loop's south-west corner,
#: along the sidewalk band, up our sidewalk, through the gate and the
#: garden door, to a REAL dock with the pins conducting:
#:
#:     travel   1.354 Wh over 44.63 m of route  (30.3 mWh/m)
#:     dock     0.316 Wh  (drive to standoff + tag creep + press)
#:     floor    1.669 Wh
#:
#: ...plus one failed press-and-retry, which is the constant's own definition
#: and is priced as one more dock leg: 1.669 + 0.316 = 1.985, carried as
#: 2.05. (Before the loop: 0.314 over 10.49 m + 0.297, 0.908 with the retry,
#: carried as 0.95; before the depth camera, 0.90.)
#:
#: ⚠ THE DISTANCE IS THE RESERVE NOW. #68's finding -- the dock dominates,
#: fifteen metres of house cost less than one docking attempt -- held while
#: the far corner was 15 m away; the loop put it 45 m away at the same
#: 30 mWh/m, and travel is four fifths of the floor. The reserve is still a
#: property of the FLOOR PLAN and not of the pack: it is what a hosting
#: pack keeps in hand (8 Wh funds it and every errand), and what sizes the
#: demo cell below. Do not scale this with the pack, and re-measure it when
#: the PLAN changes, not when the battery does.
HOME_LOW_BATTERY_WH = 2.05

#: The point the reserve should be measured from: one robot-length inside the
#: loop's south-west corner, which is the farthest the robot can legally
#: stand from the rack BY ROUTE (issue #215) -- not by straight line: the
#: loop's east legs are further as the crow flies and closer as the robot
#: drives, because the rack is reached only through the middle street's
#: gate at y=3.1. Named rather than left implicit so `tests/test_world_
#: budget.py` can check the claim against the compiled world's own routes --
#: a comment saying "the worst case is X" rots the moment somebody moves a
#: wall, and #68 and #215 between them moved most of them.
HOME_WORST_RETURN = (LOOP_X[0] + 0.4, LOOP_Y[0] + 0.4)

#: ...and the route home from there, as the waypoints a return actually
#: threads: onto the sidewalk band, east along its south leg to the first
#: house's own sidewalk, north up that to the gate, through the gate,
#: through the garden doorway, to the rack. Kept here rather than in the
#: test because it is a fact about the floor plan, and the plan lives in
#: this file; the test checks it against the compiled world's own routes.
HOME_WORST_RETURN_PATH = (
  HOME_WORST_RETURN,
  (PAVEMENT_X[0] + PAVEMENT_W / 2, PAVEMENT_Y[0] + PAVEMENT_W / 2),  # the band's SW corner
  (sum(SIDEWALK_X) / 2, PAVEMENT_Y[0] + PAVEMENT_W / 2),  # east along it, to our sidewalk
  (sum(SIDEWALK_X) / 2, STREET_DOOR_Y),      # north up the sidewalk
  (SIDEWALK_X[0], STREET_DOOR_Y),            # the street doorway
  (GARDEN_X[0], sum(DOOR_GARDEN_Y) / 2.0),   # the garden doorway
  HOME_RACK_POS,
)
#: Its length. Measured 2026-09-19 against the plan above; 15.59 m before
#: #215 and 11.96 m before #68.
HOME_WORST_RETURN_M = 49.64

#: SIZED FROM THE MEASURED RESERVE (issue #84; re-sized for the loop, #215),
#: not guessed, and the arithmetic is short enough to carry here. Off one
#: charge (`CHARGED` = 0.9) the cell must hold the reserve above plus the
#: dearest errand (the census, 1.304 Wh): the floor is (2.05 + 1.304) / 0.9
#: = 3.73 Wh. Carried at 4.5 -- ~20 % of headroom, the same margin 3.0 gave
#: the 2.41 floor before the loop.
#:
#: ⚠ 4.5 IS ALSO A DAY WITH A CHARGE IN IT, verified on the mission the
#: suite flies: the milestone-8 home arm starts at 55 % (2.48 Wh -- above the
#: 2.05 floor, so the FLOOR does not fire) and its carry errand (0.914) is
#: refused up front against that reserve, so the loop defers, charges, and
#: resumes -- the defer-then-charge-then-resume path on real physics. A full pack funds a whole preset day with room to
#: spare, which is why that test does not start full: a demo mission that
#: never needs the hub proves nothing about charging.
#:
#: ⚠ THE MARGIN IS NON-ZERO ON THIS CELL, deliberately (issue #84). The old
#: 1.1 Wh cell ran its errands on `overspend` and finished the recorded
#: census at frac 0.000. The charged pack funds reserve + dearest, so
#: `margin_wh` charges the full reserve and the mid-errand death stops
#: being reachable on the world the tests fly most. room_hub's 1.0 Wh cell
#: is untouched and still zero-margin.
HOME_DEMO_CAPACITY_WH = 4.5
#: ...and the pack a WATCHED world runs on (issue #15). The demo cell flattens
#: in minutes by design, which is right for a test and reads as a robot that
#: only ever charges; a hosting-sized one gives the hours-long work/charge
#: rhythm the site wants. `--pack hosting`, `$PLUGGY_PACK=hosting`, or
#: `--battery-wh` for anything else -- and it is what the deployment has
#: actually been running since rooftop-media-2026 #20.
#:
#: ⚠ THE RESERVE IS NOT SCALED WITH IT, deliberately. It is the absolute
#: energy needed to reach the dock -- a property of the FLOOR PLAN, not a
#: fraction of the pack (the milestone-7 lesson) -- so it is the same
#: HOME_LOW_BATTERY_WH on either cell. What changes on a hosting pack is that
#: the reserve becomes a margin the robot can afford to KEEP: economy/energy.py then requires every
#: errand to finish with it intact, which is what stops a mid-errand death.
HOME_HOSTING_CAPACITY_WH = 8.0

#: ⚠ THE CAMERAS' NEAR PLANE IS PINNED HERE, and the loop is why (issue
#: #215). MuJoCo derives a model `statistic.extent` from the geometry's
#: bounding box and scales every camera's near clipping plane by it
#: (`visual.map.znear` x extent, 0.01 x extent by default): #68's property
#: auto-derived 37.2 m, a 0.37 m near plane, and the tag pipeline was
#: proven against that -- the dock camera decodes a bay tag from the 0.34 m
#: standoff. The loop doubled the box to 70 m and the near plane to 0.70 m,
#: and the dock camera at a bay CLIPPED THE WHOLE RACK OUT OF ITS OWN IMAGE:
#: every bay fix returned None, every pick and stow ran blind, and the pen
#: went on the floor at the first stow after the change (measured: mean
#: pixel brightness 94.6 -> 27.6 at the pen bay, identical again with the
#: extent pinned). So the extent is a property of the ROBOT'S CAMERAS and
#: is written down, not derived from however big the world happens to be;
#: `tests/test_home_world.py` decodes a bay tag from the standoff to hold
#: it. The centre rides with it so the shadow map lands where it always
#: did; the second house is beyond it and simply casts no shadow.
CAMERA_EXTENT_M = 37.2
STATISTIC_CENTER = (0.577, -1.233, 0.153)

WALL_RGBA = "0.78 0.75 0.70 1"
FENCE_RGBA = "0.55 0.45 0.35 1"
FLOOR_RGBA = "0.62 0.55 0.45 1"
GROUND_RGBA = "0.45 0.52 0.40 1"
SIDEWALK_RGBA = "0.68 0.67 0.64 1"
STREET_RGBA = "0.30 0.30 0.32 1"


def _box_body(name: str, cx: float, cy: float, cz: float,
              hx: float, hy: float, hz: float, rgba: str,
              extra: str = "", geom_name: str | None = None) -> str:
  gname = geom_name or f"{name}_geom"
  return (f'    <body name="{name}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">\n'
          f'      <geom name="{gname}" type="box" '
          f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{rgba}"{extra}/>\n'
          f'    </body>')


def _wall_run(prefix: str, x0: float, x1: float, y0: float, y1: float,
              half_h: float, rgba: str,
              gaps: tuple[tuple[float, float], ...] = ()) -> list[str]:
  """A straight wall from (x0,y0) to (x1,y1) -- axis-aligned -- split into
  segments around doorway gaps. Each segment is its own BODY, because
  visual hints are per body: a wall that was one geom of `world` could
  never be hinted (learned designing the protocol -- the fixture's room
  walls are world geoms and are stuck as raw primitives forever)."""
  along_x = abs(x1 - x0) > abs(y1 - y0)
  lo, hi = (min(x0, x1), max(x0, x1)) if along_x else (min(y0, y1), max(y0, y1))
  cuts = [lo] + [v for g in sorted(gaps) for v in g] + [hi]
  bodies = []
  for i in range(0, len(cuts), 2):
    a, b = cuts[i], cuts[i + 1]
    if b - a < 0.05:
      continue
    mid, half = (a + b) / 2, (b - a) / 2
    n = f"{prefix}_{i // 2}" if len(cuts) > 2 else prefix
    if along_x:
      bodies.append(_box_body(n, mid, y0, half_h, half, WALL_HALF_T, half_h, rgba))
    else:
      bodies.append(_box_body(n, x0, mid, half_h, WALL_HALF_T, half, half_h, rgba))
  return bodies


def _board_xml(name: str, spec: dict) -> str:
  """A wall-mounted whiteboard. friction+priority on the drawing face is the
  board CONTRACT (hub_world's board comment): pair friction combines as the
  elementwise MAX, so without priority the pen would drag at the world's
  default 1.0 and the carriage servo would fight it all figure long."""
  x, y, z = spec["pos"]
  depth, width, height = spec["half"]
  h = spec["heading"]
  # The board's thin axis lies along the robot's heading; width is lateral.
  if abs(math.sin(h)) < 0.5:              # heading along +-x: thin in x
    hx, hy = depth, width
  else:                                   # heading along +-y: thin in y
    hx, hy = width, depth
  return _box_body(name, x, y, z, hx, hy, height, "0.95 0.95 0.93 1",
                   extra=' friction="0.25" priority="1"',
                   geom_name=spec["geom"])


def build_home_world() -> tuple[str, dict]:
  """(MJCF text, meta dict). Pure -- writing happens in write_home_world."""
  hints: dict[str, str] = {}
  bodies: list[str] = []

  def add(xml: str, name: str, hint: str | None) -> None:
    bodies.append(xml)
    if hint:
      hints[name] = hint

  hx0, hx1 = HOUSE_X
  hy0, hy1 = HOUSE_Y
  gx0, gx1 = GARDEN_X

  wgx0, wgx1 = WING_X
  hlx0, hlx1 = HALL_X
  py0, py1 = PROPERTY_Y

  # Perimeter + divider walls (indoor), fences (garden). ⚠ The ORIGINAL five
  # keep their names: `wall_west` is still the house's west wall, now with the
  # hall doorway cut into it, and the wing's outer wall is `wall_wing_west`.
  # Renaming would churn every visual hint and every body in the committed
  # scene for no reason.
  walls = (
    ("wall_west", _wall_run("wall_west", hx0, hx0, hy0, hy1,
                            WALL_HALF_H, WALL_RGBA, gaps=(DOOR_HALL_Y,))),
    ("wall_south", _wall_run("wall_south", hx0, hx1, hy0, hy0,
                             WALL_HALF_H, WALL_RGBA)),
    ("wall_north", _wall_run("wall_north", hx0, hx1, hy1, hy1,
                             WALL_HALF_H, WALL_RGBA)),
    ("wall_east", _wall_run("wall_east", hx1, hx1, hy0, hy1, WALL_HALF_H,
                            WALL_RGBA, gaps=(DOOR_GARDEN_Y,))),
    ("wall_divider", _wall_run("wall_divider", hx0, hx1, DIV_Y, DIV_Y,
                               WALL_HALF_H, WALL_RGBA, gaps=(DOOR_DIV_X,))),
    # ---- the west wing and the hall (issue #68) -----------------------------
    ("wall_wing_west", _wall_run("wall_wing_west", wgx0, wgx0, py0, py1,
                                 WALL_HALF_H, WALL_RGBA)),
    ("wall_wing_north", _wall_run("wall_wing_north", wgx0, wgx1, py1, py1,
                                  WALL_HALF_H, WALL_RGBA)),
    ("wall_wing_south", _wall_run("wall_wing_south", wgx0, wgx1, py0, py0,
                                  WALL_HALF_H, WALL_RGBA)),
    # The wing's east wall carries BOTH inner doorways; the kitchen and the
    # workshop reach each other only through the hall, which is what makes the
    # hall a route rather than a corridor-shaped room.
    ("wall_wing_east", _wall_run("wall_wing_east", wgx1, wgx1, py0, py1,
                                 WALL_HALF_H, WALL_RGBA,
                                 gaps=(DOOR_WORKSHOP_Y, DOOR_KITCHEN_Y))),
    ("wall_kitchen_div", _wall_run("wall_kitchen_div", wgx0, wgx1,
                                   KITCHEN_Y[0], KITCHEN_Y[0],
                                   WALL_HALF_H, WALL_RGBA)),
    ("wall_hall_north", _wall_run("wall_hall_north", hlx0, hlx1, py1, py1,
                                  WALL_HALF_H, WALL_RGBA)),
    ("wall_hall_south", _wall_run("wall_hall_south", hlx0, hlx1, py0, py0,
                                  WALL_HALF_H, WALL_RGBA)),
    # ...and the stretch of x=-2 SOUTH of the house, where the hall's east side
    # faces the garden rather than the living room. `wall_west` covers y -2..6;
    # this covers -6..-2, and without it the garden and the hall are one room.
    ("wall_hall_east", _wall_run("wall_hall_east", hlx1, hlx1,
                                 py0, hy0, WALL_HALF_H, WALL_RGBA)),
    # ---- the garden's fences ------------------------------------------------
    # ⚠ `fence_south` moved from y=-2 (x 5..10) to the PROPERTY's south edge,
    # and `fence_east` now runs the full height: the garden wraps south of the
    # house, so the old south fence is interior lawn now.
    ("fence_south", _wall_run("fence_south", hx0, gx1, py0, py0,
                              FENCE_HALF_H, FENCE_RGBA)),
    ("fence_north", _wall_run("fence_north", gx0, gx1, hy1, hy1,
                              FENCE_HALF_H, FENCE_RGBA)),
    ("fence_east", _wall_run("fence_east", gx1, gx1, py0, py1,
                             FENCE_HALF_H, FENCE_RGBA,
                             gaps=((STREET_DOOR_Y - STREET_DOOR_HALF,
                                    STREET_DOOR_Y + STREET_DOOR_HALF),))),
    # ---- the second house (issue #215) ----------------------------------------
    # Its front garden's fence faces ours across the street, gate to gate;
    # the house's west wall carries the front door, the lobby's east wall
    # both inner doorways, and the lab and the store are divided at y=0.
    ("fence_2_west", _wall_run("fence_2_west", GARDEN_2_X[0], GARDEN_2_X[0],
                               py0, py1, FENCE_HALF_H, FENCE_RGBA,
                               gaps=(DOOR_GARDEN_2_Y,))),
    ("fence_2_north", _wall_run("fence_2_north", GARDEN_2_X[0], GARDEN_2_X[1],
                                py1, py1, FENCE_HALF_H, FENCE_RGBA)),
    ("fence_2_south", _wall_run("fence_2_south", GARDEN_2_X[0], GARDEN_2_X[1],
                                py0, py0, FENCE_HALF_H, FENCE_RGBA)),
    ("wall_2_west", _wall_run("wall_2_west", HOUSE_2_X[0], HOUSE_2_X[0],
                              py0, py1, WALL_HALF_H, WALL_RGBA,
                              gaps=(DOOR_LOBBY_Y,))),
    ("wall_2_north", _wall_run("wall_2_north", HOUSE_2_X[0], HOUSE_2_X[1],
                               py1, py1, WALL_HALF_H, WALL_RGBA)),
    ("wall_2_south", _wall_run("wall_2_south", HOUSE_2_X[0], HOUSE_2_X[1],
                               py0, py0, WALL_HALF_H, WALL_RGBA)),
    ("wall_2_east", _wall_run("wall_2_east", HOUSE_2_X[1], HOUSE_2_X[1],
                              py0, py1, WALL_HALF_H, WALL_RGBA)),
    ("wall_lobby_east", _wall_run("wall_lobby_east", LOBBY_X[1], LOBBY_X[1],
                                  py0, py1, WALL_HALF_H, WALL_RGBA,
                                  gaps=(DOOR_STORE_Y, DOOR_LAB_Y))),
    ("wall_lab_div", _wall_run("wall_lab_div", LAB_X[0], LAB_X[1],
                               LAB_Y[0], LAB_Y[0], WALL_HALF_H, WALL_RGBA)),
    # ---- the fence around the street ----------------------------------------
    # Unbroken: the loop is the whole outdoors now, and a world with no way
    # out has no invisible ends. The horizon is the website's.
    ("fence_loop_west", _wall_run("fence_loop_west", LOOP_X[0], LOOP_X[0],
                                  LOOP_Y[0], LOOP_Y[1], FENCE_HALF_H, FENCE_RGBA)),
    ("fence_loop_east", _wall_run("fence_loop_east", LOOP_X[1], LOOP_X[1],
                                  LOOP_Y[0], LOOP_Y[1], FENCE_HALF_H, FENCE_RGBA)),
    ("fence_loop_north", _wall_run("fence_loop_north", LOOP_X[0], LOOP_X[1],
                                   LOOP_Y[1], LOOP_Y[1], FENCE_HALF_H, FENCE_RGBA)),
    ("fence_loop_south", _wall_run("fence_loop_south", LOOP_X[0], LOOP_X[1],
                                   LOOP_Y[0], LOOP_Y[0], FENCE_HALF_H, FENCE_RGBA)),
  )
  for prefix, segments in walls:
    hint = "fence" if prefix.startswith("fence") else "wall"
    for xml in segments:
      name = xml.split('name="', 1)[1].split('"', 1)[0]
      add(xml, name, hint)

  # Floor/ground overlays: visual-only 1 mm slabs over the grey plane, so
  # indoors reads as floorboards and the garden as grass -- to robot cameras
  # and website alike -- without adding a single contact.
  visual = ' contype="0" conaffinity="0"'
  add(_box_body("home_floor", (hx0 + hx1) / 2, (hy0 + hy1) / 2, 0.001,
                (hx1 - hx0) / 2, (hy1 - hy0) / 2, 0.001, FLOOR_RGBA,
                extra=visual),
      "home_floor", "floor")
  add(_box_body("garden_ground", (gx0 + gx1) / 2, (hy0 + hy1) / 2, 0.001,
                (gx1 - gx0) / 2, (hy1 - hy0) / 2, 0.001, GROUND_RGBA,
                extra=visual),
      "garden_ground", "ground")

  # ...and the surfaces issue #68 added. One body per room rather than one
  # slab for the building, because a hint is per BODY: a single floor could
  # only ever be one material, and the kitchen, the workshop and the hall are
  # three rooms a visitor should be able to tell apart.
  for name, (fx0, fx1), (fy0, fy1) in (
    ("kitchen_floor", WING_X, KITCHEN_Y),
    ("workshop_floor", WING_X, WORKSHOP_Y),
    ("hall_floor", HALL_X, PROPERTY_Y),
    # The second house (issue #215), one slab per room for the same reason.
    ("lobby_floor", LOBBY_X, PROPERTY_Y),
    ("lab_floor", LAB_X, LAB_Y),
    ("store_floor", LAB_X, STORE_Y),
  ):
    add(_box_body(name, (fx0 + fx1) / 2, (fy0 + fy1) / 2, 0.001,
                  (fx1 - fx0) / 2, (fy1 - fy0) / 2, 0.001, FLOOR_RGBA,
                  extra=visual), name, "floor")
  for name, (ox0, ox1), (oy0, oy1), rgba, hint in (
    ("garden_south_ground", (hx0, gx1), GARDEN_SOUTH_Y, GROUND_RGBA, "ground"),
    ("sidewalk_ground", SIDEWALK_X, PROPERTY_Y, SIDEWALK_RGBA, "sidewalk"),
    # The middle street runs on through the sidewalk band to the loop.
    ("street_ground", STREET_X, PAVEMENT_Y, STREET_RGBA, "street"),
    # The second house's front, and the loop (issue #215): one slab per
    # zone, so the surfaces are the zones a visitor sees.
    ("sidewalk_2_ground", SIDEWALK_2_X, PROPERTY_Y, SIDEWALK_RGBA, "sidewalk"),
    ("garden_2_ground", GARDEN_2_X, PROPERTY_Y, GROUND_RGBA, "ground"),
    ("sidewalk_north_ground", (PAVEMENT_X[0], STREET_X[0]),
     (BLOCK_Y[1], PAVEMENT_Y[1]), SIDEWALK_RGBA, "sidewalk"),
    ("sidewalk_2_north_ground", (STREET_X[1], PAVEMENT_X[1]),
     (BLOCK_Y[1], PAVEMENT_Y[1]), SIDEWALK_RGBA, "sidewalk"),
    ("sidewalk_south_ground", (PAVEMENT_X[0], STREET_X[0]),
     (PAVEMENT_Y[0], BLOCK_Y[0]), SIDEWALK_RGBA, "sidewalk"),
    ("sidewalk_2_south_ground", (STREET_X[1], PAVEMENT_X[1]),
     (PAVEMENT_Y[0], BLOCK_Y[0]), SIDEWALK_RGBA, "sidewalk"),
    ("sidewalk_west_ground", (PAVEMENT_X[0], BLOCK_X[0]), BLOCK_Y,
     SIDEWALK_RGBA, "sidewalk"),
    ("sidewalk_east_ground", (BLOCK_X[1], PAVEMENT_X[1]), BLOCK_Y,
     SIDEWALK_RGBA, "sidewalk"),
    ("street_north_ground", LOOP_X, (PAVEMENT_Y[1], LOOP_Y[1]), STREET_RGBA, "street"),
    ("street_south_ground", LOOP_X, (LOOP_Y[0], PAVEMENT_Y[0]), STREET_RGBA, "street"),
    ("street_west_ground", (LOOP_X[0], PAVEMENT_X[0]), PAVEMENT_Y, STREET_RGBA, "street"),
    ("street_east_ground", (PAVEMENT_X[1], LOOP_X[1]), PAVEMENT_Y, STREET_RGBA, "street"),
  ):
    add(_box_body(name, (ox0 + ox1) / 2, (oy0 + oy1) / 2, 0.001,
                  (ox1 - ox0) / 2, (oy1 - oy0) / 2, 0.001, rgba,
                  extra=visual), name, hint)

  # Whiteboards (the milestone-8 drawing-surface port).
  for name, spec in BOARDS.items():
    add(_board_xml(name, spec), name, "whiteboard")

  # Plants: visual markers the parametric renderer grows into greenery.
  for i, (px, py) in enumerate(PLANTS):
    add(f'    <body name="plant_{i}" pos="{px:.4f} {py:.4f} 0.15">\n'
        f'      <geom name="plant_{i}_geom" type="cylinder" '
        f'size="0.04 0.15" rgba="0.35 0.50 0.30 1"{visual}/>\n'
        f'    </body>',
        f"plant_{i}", "plant")

  # The reference activity. Its module owns BOTH the geometry and the state
  # machine (activity/plate.py), so a world adds one by calling one function
  # -- the same shape as rack/coupling.py owning the tool modules' faces.
  # The PAD is hinted `plate` and says what it is for (`plates` below, the
  # site's glyph); the light itself has no hint, because there is no `light`
  # in the vocabulary and inventing one ahead of a consumer is how a shared
  # vocabulary rots.
  act_body, act_sensor = plate_light_xml(PLATE_XY)
  bodies.append(act_body)
  hints["garden_plate"] = "plate"
  plates = {"garden_plate": {"purpose": "light"}}

  # The lab's props (issue #215): the cage, what lives in it, its three
  # plates, and the bench with its two masses. Each module owns its own
  # geometry (ActivityPattern.md section 2; challenge/bench.py); their
  # activities and grader are #226's and #227's and touch nothing here.
  cage_body, cage_sensors = cage_xml(LAB_CAGE_XY)
  bodies.append(cage_body)
  hints["lab_cage"] = "cage"
  hints["lab_mouse"] = "mouse"
  # The lab's three plates are what the robot ACTS on the cage with (#226);
  # each pad says which act, so a visitor can tell them apart. The robot is
  # told nothing here: the glyph is the site's and its cameras never see it.
  for name in PLATE_NAMES:
    hints[f"lab_{name}_plate"] = "plate"
    plates[f"lab_{name}_plate"] = {"purpose": name}
  bodies.append(bench_xml(LAB_BENCH_XY))
  hints["lab_bench"] = "table"

  # Furniture: real obstacles, for exploration to have something to map --
  # and, since issue #66 froze the names, HINTED. The couch and the bed were
  # unhinted only because there was no word for them; there is now, and all
  # of these sit in `UNBUILT` on the website, so nothing renders differently
  # until rooftop#129 draws them.
  #
  # ⚠ EXACTLY ONE BOX EACH. That is the hint vocabulary's rule for furniture
  # (`hints.ONE_BOX_HINTS`), and it is what lets a builder key on the name
  # rather than on proportions -- `largestBox` picks by volume, so a couch
  # modelled as a frame plus cushions would hand the builder the wrong one.
  add(_box_body("furniture_couch", 3.2, 0.8, 0.25, 0.50, 0.35, 0.25,
                "0.45 0.40 0.50 1"), "furniture_couch", "couch")
  add(_box_body("furniture_bed", -0.5, 4.8, 0.20, 0.90, 0.60, 0.20,
                "0.55 0.50 0.55 1"), "furniture_bed", "bed")
  add(_box_body("furniture_counter", -8.5, 5.2, 0.45, 2.00, 0.30, 0.45,
                "0.60 0.56 0.50 1"), "furniture_counter", "counter")
  add(_box_body("furniture_table", -8.5, -2.0, 0.37, 0.60, 0.40, 0.37,
                "0.50 0.42 0.34 1"), "furniture_table", "table")

  # The staircase. ONE SOLID BOX filling the hall's south end, 0.9 m tall so
  # the LIDAR (0.223 m) sees it -- see the module docstring. It is the only
  # piece of the new plan that BLOCKS a route rather than decorating one, and
  # it blocks a dead end: the hall runs y -6..6 and nothing lies south of it.
  add(_box_body("stairs", (STAIRS_X[0] + STAIRS_X[1]) / 2,
                (STAIRS_Y[0] + STAIRS_Y[1]) / 2, STAIRS_HALF_H,
                (STAIRS_X[1] - STAIRS_X[0]) / 2,
                (STAIRS_Y[1] - STAIRS_Y[0]) / 2, STAIRS_HALF_H,
                "0.58 0.54 0.50 1"), "stairs", "stairs")

  hints["rack"] = "rack"
  # ...and the built-tool rail beside it (issue #277), the same silhouette
  # for the site to reskin.
  hints[BUILT_RACK_BODY] = "rack"

  # ---- the ground the website draws (issue #68) -----------------------------
  # GENERATED from the layout rather than written down, which is the point:
  # the plot grew from 7 x 8 m to 26.5 x 12 m and a literal `size="10 10"` in
  # a shared include is exactly the thing that then gets forgotten. It moved
  # out of models/world_fork.xml so growing it here cannot wrap room_hub's
  # 8 x 8 m room in a 32 m apron of grey.
  #
  # ⚠ A plane's `size` is RENDERING ONLY -- collision is with the infinite
  # half-space (verified: a box 15 m outside a 1x1 plane rests at the same
  # height as one on it). So this is what a VISITOR sees the world standing
  # on, and a body outside it is drawn floating over nothing.
  # `tests/test_world_budget.py` is what holds the two together.
  px0, px1 = LOOP_X[0] - GROUND_MARGIN, LOOP_X[1] + GROUND_MARGIN
  py_lo, py_hi = LOOP_Y[0] - GROUND_MARGIN, LOOP_Y[1] + GROUND_MARGIN
  floor_xml = (f'<geom name="floor" type="plane" '
               f'pos="{(px0 + px1) / 2:.4f} {(py_lo + py_hi) / 2:.4f} 0" '
               f'size="{(px1 - px0) / 2:.4f} {(py_hi - py_lo) / 2:.4f} 0.1" '
               f'rgba="0.5 0.5 0.5 1"/>')

  # One light per new room, so the wing is not a cave on camera.
  lights = "\n".join(
    f'    <light pos="{lx:.4f} {ly:.4f} 3" dir="0 0 -1"/>'
    for lx, ly in (
      ((WING_X[0] + WING_X[1]) / 2, (KITCHEN_Y[0] + KITCHEN_Y[1]) / 2),
      ((WING_X[0] + WING_X[1]) / 2, (WORKSHOP_Y[0] + WORKSHOP_Y[1]) / 2),
      ((HALL_X[0] + HALL_X[1]) / 2, 0.0),
      ((hx0 + gx1) / 2, (GARDEN_SOUTH_Y[0] + GARDEN_SOUTH_Y[1]) / 2),
      ((STREET_X[0] + STREET_X[1]) / 2, 0.0),
      # The second house and the loop (issue #215).
      ((LOBBY_X[0] + LOBBY_X[1]) / 2, 0.0),
      ((LAB_X[0] + LAB_X[1]) / 2, (LAB_Y[0] + LAB_Y[1]) / 2),
      ((LAB_X[0] + LAB_X[1]) / 2, (STORE_Y[0] + STORE_Y[1]) / 2),
      ((LOOP_X[0] + LOOP_X[1]) / 2, (PAVEMENT_Y[1] + LOOP_Y[1]) / 2),
      ((LOOP_X[0] + LOOP_X[1]) / 2, (LOOP_Y[0] + PAVEMENT_Y[0]) / 2),
      ((LOOP_X[0] + PAVEMENT_X[0]) / 2, 0.0),
      ((PAVEMENT_X[1] + LOOP_X[1]) / 2, 0.0),
    ))

  # The tower's blocks (issue #207): the challenge's own props, in the
  # generator now that the challenge is offered -- Challenges.md §3 item 4.
  from pluggybot.challenge.stack import BLOCKS, block_xml
  for name, (bx, by), tag_id in zip(BLOCKS, TOWER_XY, BLOCK_TAG_IDS):
    add(block_xml(name, bx, by, tag_id), name, None)

  tag_ids = write_tag_pngs()
  xml = f"""<!-- GENERATED by pluggybot.home.world.write_home_world().
     Regenerate: uv run python -m pluggybot.home.world
     The home world (issue #6, expanded by #68, joined by #215): kitchen,
     workshop, hall, living room, bedroom, a garden wrapping south and
     east, the sidewalk and street through the open fence doorway -- and
     across the street a second house (a lobby, the lab with the cage and
     the bench, a store), a sidewalk band and a loop street round both,
     inside one fence. Whiteboard drawing surfaces; the tool rack on the
     living room's south wall.
     Layout constants + visual hints + zones live in home/world.py; the
     sidecar models/home_world.meta.json is emitted alongside. -->
<mujoco model="home_world">
  <include file="world_fork.xml"/>
  <statistic center="{STATISTIC_CENTER[0]} {STATISTIC_CENTER[1]} {STATISTIC_CENTER[2]}"
             extent="{CAMERA_EXTENT_M}"/>
  <asset>
    {asset_xml(tag_ids)}
  </asset>
  <worldbody>
    {floor_xml}
    <light pos="1.5 0.5 3" dir="0 0 -1"/>
    <light pos="1.5 4.5 3" dir="0 0 -1"/>
    <light pos="7.5 2.0 3" dir="0 0 -1"/>
{lights}

{chr(10).join(bodies)}

    {rack_and_modules_xml(HOME_RACK_POS, HOME_RACK_YAW)}
  </worldbody>
  <sensor>
    {act_sensor}
    {cage_sensors}
  </sensor>
  <actuator>
    {pen_actuator_xml()}
    {claw_actuator_xml()}
    {dispenser_actuator_xml()}
  </actuator>
</mujoco>
"""

  meta = {
    "model": "home_world",
    "visualHints": hints,
    "zones": list(ZONES),
    "spawns": {k: list(v) for k, v in SPAWNS.items()},
    "boards": {name: {"geom": s["geom"], "pos": list(s["pos"]),
                      "half": list(s["half"]), "heading": s["heading"]}
               for name, s in BOARDS.items()},
    "rack": {"pos": list(HOME_RACK_POS), "yaw_deg": HOME_RACK_YAW},
    "tower": {"name": "workshop", "blocks": [list(xy) for xy in TOWER_XY]},
    # The experiment zone (issue #215): where its props stand, by the names
    # #226 and #227 read them by. The room is the `lab` zone above.
    "lab": {"name": "lab", "cage": list(LAB_CAGE_XY), "bench": list(LAB_BENCH_XY)},
    # Every pressure plate and what it is FOR (`protocol.PLATE_PURPOSES`),
    # by body name: the site draws a glyph on the pad for visitors.
    "plates": plates,
    "gridBounds": list(GRID_BOUNDS),
    "battery": {"lowWh": HOME_LOW_BATTERY_WH,
                "demoCapacityWh": HOME_DEMO_CAPACITY_WH,
                # What a WATCHED run uses (issue #15). In the sidecar because
                # the website reads this file to know what world it is
                # drawing, and "the robot charges twice an hour" and "the
                # robot charges twice a minute" are different worlds.
                "hostingCapacityWh": HOME_HOSTING_CAPACITY_WH},
  }
  return xml, meta


def charge_bay_world() -> tuple[float, float]:
  """World position of the home rack's charge bay (for scripts/tests)."""
  return rack_frame_to_world(0.114, CHARGE_BAY_Y, HOME_RACK_POS, HOME_RACK_YAW)


def bay_world(i: int) -> tuple[float, float]:
  """World position of tool bay i's hang point."""
  return rack_frame_to_world(0.09, HUB_STATION_YS[i], HOME_RACK_POS,
                             HOME_RACK_YAW)


def write_home_world(xml_path: str = "models/home_world.xml",
                     meta_path: str = "models/home_world.meta.json") -> None:
  xml, meta = build_home_world()
  with open(xml_path, "w") as fh:
    fh.write(xml)
  with open(meta_path, "w") as fh:
    json.dump(meta, fh, indent=1)
    fh.write("\n")


if __name__ == "__main__":
  write_home_world()
  print("wrote models/home_world.xml and models/home_world.meta.json")
