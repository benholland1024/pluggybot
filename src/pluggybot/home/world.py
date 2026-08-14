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

Layout (meters; y+ north, x+ east; the robot lives indoors and may wander
the garden):

    y=8 ┌───────wall──────┬─────fence──────┐
        │   bedroom       │                │
        │   [whiteboard_b]d     garden     │
        │   furniture_bed │    (plants)    │
    y=4.5├──wall──d──wall──┤                │
        │   living        d                │
     [whiteboard_a]       │                │
        │  furniture_couch│                │
        │   ┌─rack─┐      │                │
    y=0 └───┴──────┴─wall─┴─────fence──────┘
        x=0              x=7             x=12

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

from pluggybot.hub.coupling import (
  CHARGE_BAY_Y, HUB_STATION_YS, rack_and_modules_xml, rack_frame_to_world,
  claw_actuator_xml, pen_actuator_xml,
)
from pluggybot.hub.tags import asset_xml, write_tag_pngs

# ---- layout constants (the one source) --------------------------------------
# The ORIGIN sits inside the living room, like room_1's: the robot model
# spawns at (0,0) before any script places it, and a world whose origin is
# inside a wall greets every bare `MjData` with a chassis wedged in masonry
# (the first cut of this file did exactly that -- 78 contacts at settle).
HOUSE_X = (-2.0, 5.0)         # indoor extent
HOUSE_Y = (-2.0, 6.0)
GARDEN_X = (5.0, 10.0)        # garden shares the house's y extent
DIV_Y = 2.5                   # living/bedroom divider
DOOR_DIV_X = (1.0, 2.0)       # doorway in the divider
DOOR_GARDEN_Y = (0.2, 1.2)    # doorway from the living room to the garden

WALL_HALF_T = 0.02
WALL_HALF_H = 0.60            # 1.2 m walls, like room_1
FENCE_HALF_H = 0.45           # 0.9 m garden fence: lidar (0.223 m) sees it

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

ZONES = (
  {"name": "living", "kind": "room",
   "min": [HOUSE_X[0], HOUSE_Y[0]], "max": [HOUSE_X[1], DIV_Y]},
  {"name": "bedroom", "kind": "room",
   "min": [HOUSE_X[0], DIV_Y], "max": [HOUSE_X[1], HOUSE_Y[1]]},
  {"name": "garden", "kind": "garden",
   "min": [GARDEN_X[0], HOUSE_Y[0]], "max": [GARDEN_X[1], HOUSE_Y[1]]},
)

SPAWNS = {
  "start": [1.5, 0.5, math.pi / 2],       # living room, facing the bedroom
  "garden": [7.5, 2.0, 0.0],
}

# Occupancy-grid bounds for this world (HubMission takes them as a
# parameter; room_hub keeps its historical defaults).
GRID_BOUNDS = (-3.0, -3.0, 11.0, 7.0)

# Battery tuning for the bigger floor plan (issue #6 "rides along"). The
# reserve is absolute energy, per the milestone-7 lesson: it must cover the
# WORST return trip to the rack plus one failed press-and-retry. Measured
# empirically (see docs/PluggyPlan.md): a full living-room crossing plus
# charge approach costs ~0.3 Wh at cruise draw, so the room_hub reserve of
# 0.35 is too thin here; the demo cell grows with it so one explore + one
# errand still runs the pack down and the loop still has to charge.
HOME_LOW_BATTERY_WH = 0.55
HOME_DEMO_CAPACITY_WH = 1.1

WALL_RGBA = "0.78 0.75 0.70 1"
FENCE_RGBA = "0.55 0.45 0.35 1"
FLOOR_RGBA = "0.62 0.55 0.45 1"
GROUND_RGBA = "0.45 0.52 0.40 1"


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

  # Perimeter + divider walls (indoor), fences (garden).
  walls = (
    ("wall_west", _wall_run("wall_west", hx0, hx0, hy0, hy1,
                            WALL_HALF_H, WALL_RGBA)),
    ("wall_south", _wall_run("wall_south", hx0, hx1, hy0, hy0,
                             WALL_HALF_H, WALL_RGBA)),
    ("wall_north", _wall_run("wall_north", hx0, hx1, hy1, hy1,
                             WALL_HALF_H, WALL_RGBA)),
    ("wall_east", _wall_run("wall_east", hx1, hx1, hy0, hy1, WALL_HALF_H,
                            WALL_RGBA, gaps=(DOOR_GARDEN_Y,))),
    ("wall_divider", _wall_run("wall_divider", hx0, hx1, DIV_Y, DIV_Y,
                               WALL_HALF_H, WALL_RGBA, gaps=(DOOR_DIV_X,))),
    ("fence_south", _wall_run("fence_south", gx0, gx1, hy0, hy0,
                              FENCE_HALF_H, FENCE_RGBA)),
    ("fence_north", _wall_run("fence_north", gx0, gx1, hy1, hy1,
                              FENCE_HALF_H, FENCE_RGBA)),
    ("fence_east", _wall_run("fence_east", gx1, gx1, hy0, hy1,
                             FENCE_HALF_H, FENCE_RGBA)),
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

  # Furniture: real obstacles, for exploration to have something to map.
  add(_box_body("furniture_couch", 3.2, 0.8, 0.25, 0.50, 0.35, 0.25,
                "0.45 0.40 0.50 1"), "furniture_couch", None)
  add(_box_body("furniture_bed", -0.5, 4.8, 0.20, 0.90, 0.60, 0.20,
                "0.55 0.50 0.55 1"), "furniture_bed", None)

  hints["rack"] = "rack"

  tag_ids = write_tag_pngs()
  xml = f"""<!-- GENERATED by pluggybot.home.world.write_home_world().
     Regenerate: uv run python -m pluggybot.home.world
     The home world (issue #6): living room + bedroom + garden, whiteboard
     drawing surfaces, the tool rack on the living room's south wall.
     Layout constants + visual hints + zones live in home/world.py; the
     sidecar models/home_world.meta.json is emitted alongside. -->
<mujoco model="home_world">
  <include file="world_fork.xml"/>
  <asset>
    {asset_xml(tag_ids)}
  </asset>
  <worldbody>
    <light pos="1.5 0.5 3" dir="0 0 -1"/>
    <light pos="1.5 4.5 3" dir="0 0 -1"/>
    <light pos="7.5 2.0 3" dir="0 0 -1"/>

{chr(10).join(bodies)}

    {rack_and_modules_xml(HOME_RACK_POS, HOME_RACK_YAW)}
  </worldbody>
  <actuator>
    {pen_actuator_xml()}
    {claw_actuator_xml()}
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
    "gridBounds": list(GRID_BOUNDS),
    "battery": {"lowWh": HOME_LOW_BATTERY_WH,
                "demoCapacityWh": HOME_DEMO_CAPACITY_WH},
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
