"""Guards for the generated home world (issue #6) and its hint sidecar.

The generator is the single source for four artifacts that must agree: the
MJCF, the meta sidecar, the scene JSON the website renders, and the
constants the scripts navigate by. Most of these tests exist because those
four can drift silently -- a wall moved in Python while models/ still holds
last week's XML looks fine until a mission drives into it.
"""

import itertools
import json
import math
from pathlib import Path

import mujoco
import pytest

from pluggybot.tools.drawing import Board, PenPlotter
from pluggybot.home import world as home
from pluggybot.telemetry.protocol import VISUAL_HINTS
from pluggybot.telemetry.scene import scene_dict

REPO = Path(__file__).parent.parent
WORLD = REPO / "models" / "home_world.xml"
META = REPO / "models" / "home_world.meta.json"


@pytest.fixture(scope="module")
def home_model():
  return mujoco.MjModel.from_xml_path(str(WORLD))


@pytest.fixture(scope="module")
def meta():
  return json.loads(META.read_text())


# ---- the generated artifacts agree with the generator -----------------------

def test_committed_world_matches_the_generator():
  """The committed XML must be what the current generator emits -- editing
  models/home_world.xml by hand, or changing world.py without rerunning it,
  is exactly the drift this file exists to catch."""
  xml, _ = home.build_home_world()
  assert xml == WORLD.read_text(), \
    "stale world: uv run python -m pluggybot.home.world"


def test_committed_meta_matches_the_generator(meta):
  _, fresh = home.build_home_world()
  assert fresh == meta, "stale sidecar: uv run python -m pluggybot.home.world"


def test_generator_is_deterministic():
  a, _ = home.build_home_world()
  b, _ = home.build_home_world()
  assert a == b


# ---- the hint vocabulary (the cross-repo contract) --------------------------

def test_hints_stay_inside_the_shared_vocabulary(meta):
  """The website renders a parametric component per hint and falls back to
  primitives otherwise, so an invented hint is a silently unstyled body --
  and renaming one is a two-repo breaking change."""
  assert set(meta["visualHints"].values()) <= set(VISUAL_HINTS)


def test_every_hinted_body_exists_in_the_world(home_model, meta):
  names = {home_model.body(i).name for i in range(home_model.nbody)}
  missing = set(meta["visualHints"]) - names
  assert not missing, f"hints for bodies that do not exist: {sorted(missing)}"


def test_walls_fences_and_surfaces_are_all_hinted(home_model, meta):
  """Every body a visitor sees as architecture must carry a hint: an
  unhinted wall renders as a bare grey slab next to its styled neighbours,
  which looks like a bug and is one."""
  hints = meta["visualHints"]
  for i in range(home_model.nbody):
    name = home_model.body(i).name
    if name.startswith(("wall_", "fence_", "plant_", "whiteboard_")):
      assert name in hints, f"{name} has no visual hint"


def test_hints_are_not_encoded_as_geom_colors(home_model, meta):
  """The design doc's explicit warning: the robot's cameras render rgba, so
  colour-as-encoding would couple the tag detector to the website's art
  direction. Walls and fences must be distinguishable by NAME/sidecar, and
  the sidecar must not be reconstructible from colour -- assert the two
  hint classes are free to share a colour by checking the sidecar is the
  only thing that separates them."""
  hints = meta["visualHints"]
  # Pick the segments by hint rather than by name: a wall run is SPLIT into
  # numbered segments around every doorway, so `fence_east` became
  # `fence_east_0`/`_1` the day the garden gained a gate (issue #8). The
  # claim under test is about hints vs colour, not about segment counts.
  wall = next(n for n, h in hints.items() if h == "wall" and n.startswith("wall_"))
  fence = next(n for n, h in hints.items() if h == "fence")
  assert hints[wall] == "wall" and hints[fence] == "fence"
  # both are plain boxes; nothing in the geom itself says which is which
  wall_g = home_model.geom(f"{wall}_geom")
  fence_g = home_model.geom(f"{fence}_geom")
  assert wall_g.type == fence_g.type


def test_scene_json_carries_hints_zones_and_spawns(home_model, meta):
  scene = scene_dict(home_model, "home_world", meta=meta)
  hinted = {b["name"]: b["visual"] for b in scene["bodies"] if b["visual"]}
  assert hinted == meta["visualHints"]
  # Against the generator, not a literal list: this used to name the three
  # zones outright, which made adding a room a two-file edit and the second
  # file easy to forget. What the scene must carry is whatever `home.ZONES`
  # says, in order.
  assert [z["name"] for z in scene["zones"]] == [z["name"] for z in home.ZONES]
  assert set(scene["spawns"]) == set(home.SPAWNS)
  # ...and the two the rest of the suite and every script depend on by name.
  assert {"start", "garden"} <= set(scene["spawns"])


def test_scene_transpiler_rejects_an_unknown_hint(home_model):
  """A typo'd hint must fail the transpile rather than ship a body the
  website silently renders as a grey box."""
  with pytest.raises(ValueError, match="unknown visual hints"):
    scene_dict(home_model, "home_world",
               meta={"visualHints": {"wall_west": "wallpaper"}})


def test_room_hub_still_transpiles_without_a_sidecar():
  """Worlds with no generator meta keep working -- hints are optional."""
  model = mujoco.MjModel.from_xml_path(str(REPO / "models" / "room_hub.xml"))
  scene = scene_dict(model, "room_hub")
  assert all(b["visual"] is None for b in scene["bodies"])
  assert "zones" not in scene


# ---- the world is actually navigable ----------------------------------------

def test_robot_spawns_clear_of_the_geometry(home_model):
  """A bare MjData puts the robot at the origin, so the origin must be
  INSIDE a room: the first cut of this layout had the house starting at
  (0,0), which greeted every model load with a chassis wedged in the south
  wall (78 contacts at settle)."""
  data = mujoco.MjData(home_model)
  for _ in range(1500):
    mujoco.mj_step(home_model, data)
  chassis = home_model.geom("chassis").id
  touching = [i for i in range(data.ncon)
              if chassis in (data.contact[i].geom1, data.contact[i].geom2)]
  assert not touching, "the robot settles inside the world's geometry"


def test_doorways_are_wide_enough_to_drive_through(home_model):
  """EVERY doorway must clear the robot's 0.21 m track with margin for the
  inflated planning mask -- a doorway the planner refuses is a wall.

  All five, not the two that existed before issue #68: the wing is reachable
  through exactly one of the new three, so a doorway too narrow to plan
  through would quietly cut a third of the house off the map."""
  doors = {"divider": home.DOOR_DIV_X, "garden": home.DOOR_GARDEN_Y,
           "hall": home.DOOR_HALL_Y, "kitchen": home.DOOR_KITCHEN_Y,
           "workshop": home.DOOR_WORKSHOP_Y}
  for name, (lo, hi) in doors.items():
    assert hi - lo >= 0.6, f"{name} doorway too narrow for planning + control"


def test_zones_tile_the_world_without_overlapping():
  """EVERY pair, not the adjacent ones. The old version checked ZONES[0] vs
  [1] and [1] vs [2], which was every pair when there were three; with nine it
  would have missed the garden overlapping the street."""
  for a, b in itertools.combinations(home.ZONES, 2):
    overlap_x = (min(a["max"][0], b["max"][0]) - max(a["min"][0], b["min"][0]))
    overlap_y = (min(a["max"][1], b["max"][1]) - max(a["min"][1], b["min"][1]))
    assert overlap_x <= 0 or overlap_y <= 0, f"{a['name']} overlaps {b['name']}"


def test_the_zones_tile_the_whole_plot_with_no_gaps():
  """...and TILE it: the areas sum to the bounding rectangle exactly.

  Overlap-freedom alone is satisfied by a plan with a hole in it, and a hole
  is a room the robot can be sent to that belongs to no zone -- which is how
  `zone_centre` returns something nobody meant."""
  area = sum((z["max"][0] - z["min"][0]) * (z["max"][1] - z["min"][1])
             for z in home.ZONES)
  plot = ((home.STREET_X[1] - home.WING_X[0])
          * (home.PROPERTY_Y[1] - home.PROPERTY_Y[0]))
  assert area == pytest.approx(plot), \
      f"zones cover {area:.1f} m2 of a {plot:.1f} m2 plot"


def test_the_new_rooms_are_reachable_from_the_living_room():
  """A room with no doorway into it is scenery. Checked as a GRAPH over the
  zones rather than by eye, because the wing hangs off the hall and the hall
  hangs off one 1 m gap in the living room's west wall -- three rooms behind a
  single door, which is exactly the shape that goes wrong quietly."""
  edges = {("living", "hall"), ("hall", "kitchen"), ("hall", "workshop"),
           ("living", "bedroom"), ("living", "garden"),
           ("garden", "garden_south"), ("garden", "sidewalk"),
           ("sidewalk", "street")}
  reached, frontier = {"living"}, ["living"]
  while frontier:
    here = frontier.pop()
    for a, b in edges:
      for src, dst in ((a, b), (b, a)):
        if src == here and dst not in reached:
          reached.add(dst)
          frontier.append(dst)
  names = {z["name"] for z in home.ZONES}
  assert reached == names, f"unreachable from the living room: {names - reached}"


def test_spawns_and_rack_sit_inside_the_house(home_model):
  x, y, _ = home.SPAWNS["start"]
  assert home.HOUSE_X[0] < x < home.HOUSE_X[1]
  assert home.HOUSE_Y[0] < y < home.HOUSE_Y[1]
  gx, gy, _ = home.SPAWNS["garden"]
  assert home.GARDEN_X[0] < gx < home.GARDEN_X[1]


def test_grid_bounds_contain_the_whole_world():
  """A grid sized for one room truncates every scan past its edge, and the
  frontier planner then believes the world ends there."""
  x0, y0, x1, y1 = home.GRID_BOUNDS
  assert x0 < home.HOUSE_X[0] and y0 < home.HOUSE_Y[0]
  assert x1 > home.GARDEN_X[1] and y1 > home.HOUSE_Y[1]


# ---- the drawing surfaces ---------------------------------------------------

def test_both_whiteboards_are_real_drawing_geoms(home_model, meta):
  """`pen_on_board` checks contact against a NAMED geom, so a board whose
  geom name drifted from the sidecar is a drawing that can never register
  ink."""
  for spec in meta["boards"].values():
    geom = home_model.geom(spec["geom"])
    assert geom.priority == 1, "board friction needs priority (the MAX rule)"
    assert geom.friction[0] == pytest.approx(0.25)


def test_board_standoff_puts_the_robot_in_front_of_any_wall(meta):
  """The plotter was born against one hardcoded board facing -x. The home
  world hangs boards on two different walls, so the standoff must come out
  of the board's own heading -- offset back along it, and the fork line (not
  the chassis centreline) landing on the board's centre."""
  for name, spec in meta["boards"].items():
    board = Board.from_meta(spec)
    x, y = PenPlotter.board_standoff(
      type("P", (), {"board": board})(), standoff=0.34)
    back = math.hypot(x - board.x, y - board.y)
    assert 0.3 < back < 0.42, f"{name}: standoff {back:.2f} m off the board"
    # the robot must end up INSIDE the house, not through the wall
    assert home.HOUSE_X[0] < x < home.HOUSE_X[1]
    assert home.HOUSE_Y[0] < y < home.HOUSE_Y[1]


def test_hub_board_spec_is_unchanged_by_the_port():
  """hub_world's board is the plotter's historical contract; every existing
  drawing demo and test still runs against it."""
  board = Board.hub()
  assert board.geom == "board" and board.heading == 0.0
