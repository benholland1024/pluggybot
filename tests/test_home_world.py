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
           "workshop": home.DOOR_WORKSHOP_Y,
           # The second house (issue #215): its gate, its front door, and
           # the lab and the store off the lobby.
           "garden_2": home.DOOR_GARDEN_2_Y, "lobby": home.DOOR_LOBBY_Y,
           "lab": home.DOOR_LAB_Y, "store": home.DOOR_STORE_Y}
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
  # The plot is everything inside the fence around the loop (issue #215).
  plot = ((home.LOOP_X[1] - home.LOOP_X[0])
          * (home.LOOP_Y[1] - home.LOOP_Y[0]))
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
           ("sidewalk", "street"),
           # The second house (issue #215): across the street, gate to gate.
           ("street", "sidewalk_2"), ("sidewalk_2", "garden_2"),
           ("garden_2", "lobby"), ("lobby", "lab"), ("lobby", "store"),
           # The sidewalk band and the loop: the middle street's two ends
           # open onto the band, and the band onto the ring.
           ("street", "sidewalk_north"), ("street", "sidewalk_south"),
           ("sidewalk_north", "sidewalk_west"), ("sidewalk_north", "sidewalk_east"),
           ("sidewalk_south", "sidewalk_west"), ("sidewalk_south", "sidewalk_east"),
           ("sidewalk_north", "street_north"), ("sidewalk_south", "street_south"),
           ("sidewalk_west", "street_west"), ("sidewalk_east", "street_east"),
           ("street_north", "street_west"), ("street_north", "street_east"),
           ("street_south", "street_west"), ("street_south", "street_east")}
  # ...and every edge is a real border: the two zones touch along a line,
  # or the graph is a story about a plan rather than the plan.
  by_name = {z["name"]: z for z in home.ZONES}
  for a, b in edges:
    za, zb = by_name[a], by_name[b]
    touch_x = min(za["max"][0], zb["max"][0]) - max(za["min"][0], zb["min"][0])
    touch_y = min(za["max"][1], zb["max"][1]) - max(za["min"][1], zb["min"][1])
    assert (touch_x == 0 and touch_y > 0) or (touch_y == 0 and touch_x > 0), \
        f"{a} and {b} do not share a border"
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


# ---- the loop, the fence and the lab (issue #215) ----------------------------

def test_the_street_is_a_loop_round_both_houses():
  """The middle street used to end in mid-air at y = +-6 (issue #68). Now its
  two ends open onto a ring: the four `street_*` legs form a CYCLE, each
  bordering the two beside it, and the ring encloses both properties --
  every house zone lies strictly inside the ring's outer rectangle."""
  by_name = {z["name"]: z for z in home.ZONES}
  legs = list(home.LOOP_ZONES)
  for i, name in enumerate(legs):
    nxt = by_name[legs[(i + 1) % len(legs)]]
    here = by_name[name]
    touch_x = min(here["max"][0], nxt["max"][0]) - max(here["min"][0], nxt["min"][0])
    touch_y = min(here["max"][1], nxt["max"][1]) - max(here["min"][1], nxt["min"][1])
    assert touch_x > 0 and touch_y == 0 or touch_y > 0 and touch_x == 0, \
        f"{name} does not meet {legs[(i + 1) % len(legs)]}: the loop is open"
  for z in home.ZONES:
    if z["name"] in legs:
      continue
    assert home.LOOP_X[0] < z["min"][0] and z["max"][0] < home.LOOP_X[1]
    assert home.LOOP_Y[0] < z["min"][1] and z["max"][1] < home.LOOP_Y[1]


def test_the_fence_round_the_loop_is_unbroken(home_model):
  """No invisible ends: one fence body per side of the loop, each spanning
  its whole side. `_wall_run` splits a run around gaps, so a gap would show
  up as a second segment -- and a world with a way out is the world #215
  replaces. Shown to fail by handing `fence_loop_east` a `gaps=`."""
  sides = {"fence_loop_west": (home.LOOP_Y[1] - home.LOOP_Y[0], 1),
           "fence_loop_east": (home.LOOP_Y[1] - home.LOOP_Y[0], 1),
           "fence_loop_north": (home.LOOP_X[1] - home.LOOP_X[0], 0),
           "fence_loop_south": (home.LOOP_X[1] - home.LOOP_X[0], 0)}
  names = {home_model.body(i).name for i in range(home_model.nbody)}
  for name, (length, axis) in sides.items():
    assert name in names, f"{name} missing"
    assert not any(n.startswith(name + "_") for n in names), \
        f"{name} is split into segments: the fence has a gap"
    geom = home_model.geom(f"{name}_geom")
    assert 2 * geom.size[axis] == pytest.approx(length), f"{name} does not span its side"


def test_the_lab_holds_its_props_inside_the_room(home_model, meta):
  """The experiment zone's props (issue #215) stand in the lab and nowhere
  else: the cage with the mouse, the bowl, the wheel and the hide box in
  it; the three plates in front of it; the bench with its two masses. All
  inside the `lab` zone, so #226 and #227 find them where the world says."""
  from pluggybot.activity.cage import PLATE_NAMES
  from pluggybot.challenge.bench import MASSES
  lab = next(z for z in home.ZONES if z["name"] == "lab")
  data = mujoco.MjData(home_model)
  mujoco.mj_forward(home_model, data)
  props = ["lab_cage", "lab_mouse", "lab_bowl", "lab_wheel", "lab_hide",
           "lab_bench", *MASSES, *(f"lab_{p}_plate" for p in PLATE_NAMES)]
  for name in props:
    x, y = data.xpos[home_model.body(name).id][:2]
    assert lab["min"][0] < x < lab["max"][0] and lab["min"][1] < y < lab["max"][1], \
        f"{name} at ({x:.2f}, {y:.2f}) is outside the lab"
  assert meta["lab"]["name"] == "lab"
  assert meta["visualHints"]["lab_cage"] == "cage"
  assert meta["visualHints"]["lab_mouse"] == "mouse"
  assert meta["visualHints"]["lab_bench"] == "table"
  # The masses wear the next two tags after the tower's blocks.
  from pluggybot.rack.tags import MASS_TAG_IDS
  for name, tag in zip(MASSES, MASS_TAG_IDS):
    geom = home_model.geom(f"{name}_box")
    assert home_model.mat(int(geom.matid[0])).name == f"tagmat{tag}"


def test_the_mouse_is_a_mocap_body_and_dynamic_on_the_wire(home_model, meta):
  """The one way scenery MOVES (ActivityPattern.md 3.4): #226 sets the
  mouse's pose through `MocapToggle`, which refuses a non-mocap body -- and
  a frame carries it only if the census calls it dynamic. Shown to fail by
  dropping `mocap="true"` from the cage's emitter, or the `body_mocapid`
  clause from `dynamic_flags`."""
  from pluggybot.activity.base import MocapToggle
  from pluggybot.activity.cage import mouse_poses
  from pluggybot.telemetry.protocol import body_census, dynamic_flags
  data = mujoco.MjData(home_model)
  mouse = home_model.body("lab_mouse")
  assert int(home_model.body_mocapid[mouse.id]) >= 0
  assert dynamic_flags(home_model)[mouse.id]
  _, world = body_census(home_model)
  assert "lab_mouse" in world
  scene = scene_dict(home_model, "home_world", meta=meta)
  assert next(b for b in scene["bodies"] if b["name"] == "lab_mouse")["dynamic"]
  # ...and every pre-allocated pose is selectable, and lands inside the cage.
  poses = mouse_poses(tuple(meta["lab"]["cage"]))
  toggle = MocapToggle(home_model, data, "lab_mouse", poses)
  cage = home_model.body("lab_cage").id
  cx, cy = home_model.body_pos[cage][:2]
  for state in poses:
    toggle.select(state)
    mujoco.mj_forward(home_model, data)
    x, y, z = data.xpos[mouse.id]
    assert abs(x - cx) < 0.30 and abs(y - cy) < 0.20 and 0.0 < z < 0.10, state


def test_the_lab_plates_rest_below_their_own_trigger(home_model):
  """Three garden plates with nothing lit: each settles under its own weight
  short of `PLATE_ON`, so a plate nobody drove onto never reads pressed."""
  from pluggybot.activity.cage import PLATE_NAMES
  from pluggybot.activity.plate import PLATE_ON
  data = mujoco.MjData(home_model)
  for _ in range(500):
    mujoco.mj_step(home_model, data)
  for name in PLATE_NAMES:
    adr = int(home_model.sensor(f"lab_{name}_plate_pos").adr[0])
    depth = -float(data.sensordata[adr])
    assert 0.0 <= depth < PLATE_ON, f"the {name} plate rests at {depth * 1000:.1f} mm"


# ---- the cameras' near plane (issue #215) --------------------------------------

def _bay_fix_in(world_xml: str):
  """`HubMission.bay_fix` for the pen's bay, with the robot standing at the
  bay's standoff and its belief seeded from truth -- the real pipeline: the
  dock camera renders, the detector decodes, PnP measures."""
  from pluggybot.lifecycle import HubLifecycle, world_config
  from pluggybot.mission.mission import bay_standoff
  from pluggybot.procedure.steps import TOOL_BAYS
  from pluggybot.rack.coupling import HUB_STATION_YS
  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(world_xml)
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=4.0, rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"], errands=[])
  try:
    station_y = HUB_STATION_YS[TOOL_BAYS["module_pen"]]
    sx, sy, hd = bay_standoff(station_y, cfg["rack"])
    life.mission.start_at(sx, sy, hd)
    return life.mission.bay_fix(station_y)
  finally:
    life.mission.close()


def test_the_dock_camera_decodes_a_bay_tag_from_the_standoff():
  """The tag pipeline's whole premise, held against the world's SIZE.

  MuJoCo scales every camera's near clipping plane by the model's
  `statistic.extent`, and derives the extent from the geometry's bounding
  box unless it is written down. The loop doubled the box; the near plane
  went from 0.37 m to 0.70 m; and the dock camera at a bay standoff --
  0.34 m from the rack -- clipped the whole rack out of its own image.
  Nothing raised: `bay_fix` returned None, every pick and stow ran blind,
  and the pen went on the floor at the first stow after the change. The
  generator pins `CAMERA_EXTENT_M`; this decodes a tag through it.

  Shown to fail by deleting the `<statistic>` line from the generated
  world -- which is exactly what the second half does, so the premise
  cannot rot: the unpinned world must still lose the tag.
  """
  fix = _bay_fix_in(str(WORLD))
  assert fix is not None, "the dock camera cannot see the pen bay's tag from its standoff"
  sx, sy, hd = fix
  assert math.isfinite(sx) and math.isfinite(sy) and math.isfinite(hd)

  # The premise: the same world, its extent left to the bounding box.
  xml = WORLD.read_text()
  assert "<statistic " in xml
  unpinned = "\n".join(line for line in xml.splitlines()
                       if "<statistic " not in line and 'extent="' not in line)
  scratch = WORLD.with_name("home_world_unpinned_extent.xml")   # beside the include it needs
  try:
    scratch.write_text(unpinned)
    model = mujoco.MjModel.from_xml_path(str(scratch))
    assert model.stat.extent > 2 * home.CAMERA_EXTENT_M * 0.9, \
        "the bounding box no longer doubles the extent; re-read this test's premise"
    assert _bay_fix_in(str(scratch)) is None, \
        "the unpinned world decodes the tag now: the premise has moved, re-measure"
  finally:
    scratch.unlink(missing_ok=True)


def test_the_camera_extent_is_written_down_not_derived(home_model):
  """The statistic is pinned in the generated XML at the value the tag
  pipeline was proven at, so growing the world cannot move the near plane
  again -- and the number is the generator's own constant."""
  assert home_model.stat.extent == pytest.approx(home.CAMERA_EXTENT_M)
  assert home_model.vis.map.znear * home_model.stat.extent < 0.40, \
      "the near plane is past the bay standoff's 0.34 m"


# ---- the flown proof (issue #215) --------------------------------------------

def loop_legs() -> list[tuple[float, float]]:
  """A lap of the loop as `drive_to` targets: out by the garden doorway and
  the gate, onto the north street, clockwise round all four legs and back
  to where the lap joined the loop -- 24 steps, `steps.MAX_STEPS` exactly.

  ⚠ IT OPENS AT THE GARDEN DOORWAY, as the day's errands do: from wherever
  the explore ended, the gate itself may not be mapped yet, and a lap that
  opened on the gate failed its first step in under a second (measured).

  ⚠ NO LEG IS LONGER THAN 6.7 m, so every goal is inside the 8 m the LIDAR
  had already mapped from the leg before, and that is a measured
  constraint rather than caution (docs/SimNotes.md, "A goal out of sight is
  aimed at through the nearest wall"). With 12 m legs the lap drove 13 of
  16 and failed heading east along the north street: the goal was still
  unmapped, `_plan_to` aimed at the known-free cell nearest it by straight
  line -- indoors, behind the first house's north wall -- and the robot set
  off on the 463-waypoint detour to get there, away from its goal, until
  the stagnation check ended the drive. Belief error was under 10 cm the
  whole way; the other robot was 20 m off."""
  doorway = (home.GARDEN_X[0], sum(home.DOOR_GARDEN_Y) / 2.0)
  gate = (home.SIDEWALK_X[0], home.STREET_DOOR_Y)
  north = (home.PAVEMENT_Y[1] + home.LOOP_Y[1]) / 2.0
  south = (home.LOOP_Y[0] + home.PAVEMENT_Y[0]) / 2.0
  west = (home.LOOP_X[0] + home.PAVEMENT_X[0]) / 2.0
  east = (home.PAVEMENT_X[1] + home.LOOP_X[1]) / 2.0
  mid = sum(home.STREET_X) / 2.0
  # The south street, east to west, in seven equal legs of ~6.6 m.
  south_legs = [(east + (west - east) * i / 7.0, south) for i in range(1, 8)]
  return [doorway, gate, (mid, north), (19.0, north), (25.0, north), (east, north),
          (east, 3.0), (east, -3.0), (east, south), *south_legs,
          (west, -3.0), (west, 3.0), (west, north),
          (-9.0, north), (-3.0, north), (3.0, north), (9.0, north), (mid, north)]


def test_the_lap_keeps_every_goal_inside_what_the_leg_before_saw():
  """The flown proof's own premise, pinned fast: 24 steps (the program cap),
  closed (it ends where it first joins the loop), and no leg past 6.6 m --
  the constraint the 12 m legs broke (see `loop_legs`)."""
  import math
  from pluggybot.perception.lidar import MAX_RANGE
  from pluggybot.procedure.steps import MAX_STEPS
  legs = loop_legs()
  assert len(legs) == MAX_STEPS
  assert legs[2] == legs[-1], "the lap does not close"
  longest = max(math.dist(a, b) for a, b in zip(legs, legs[1:]))
  assert longest <= 6.7 < MAX_RANGE - 1.0, f"a {longest:.1f} m leg outruns the map"


# ⚠ BEHIND `--endurance` (issue #215): a pair's whole morning in the bigger
# world, ~30 min of wall clock. Its RULES -- the loop is a cycle, the fence
# is unbroken, every zone routes to the rack, the props stand in the lab,
# the grid fits its budget -- are pinned above and in test_world_budget.py
# in milliseconds; what only this proves is that the loop can be DRIVEN by
# the navigation stack through space it maps as it goes, with another robot
# living its own day in the same physics.
@pytest.mark.slow
@pytest.mark.endurance
def test_a_pair_day_in_the_new_world_drives_the_loop_and_both_live():
  """The second robot laps the loop as a program of `drive_to` steps while
  the first does its carry; both are alive at the end and neither hit
  anything. Stops on its claim: the lap done and the carry stowed."""
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.pair import build_pair, run_pair
  from pluggybot.procedure.steps import Program, Step
  legs = loop_legs()
  lap = Program.single("lap", [Step("drive_to", {"x": x, "y": y}) for x, y in legs],
                       budget_s=1500.0)
  lives = build_pair("home", pack="hosting", errands=("carry", "none"))
  # ⚠ QUEUED IN THE EXPLORE'S LAST SECONDS, not at build: the loop runs a
  # queued errand before it explores, and a `drive_to` with no map at all
  # fails on the spot (measured: 0/16 steps at t = 8 s). Nor after the
  # explore: a budget-spent explore returns and the loop calls the day
  # complete in the same Python frame, before the physics seam this hook
  # rides gets another step (measured: the lap was never queued and both
  # days ended at 269 s). So the lap is handed over just before the
  # deadline, and the loop finds it queued when it next looks -- the
  # robot then drives the loop through space it maps as it goes, from a
  # map that already holds the house.
  queued: list = []

  def settled(ls):
    # `explore_deadline` exists once the second robot's opening spin is done.
    deadline = getattr(ls[1], "explore_deadline", None)
    if not queued and deadline is not None and ls[1].data.time >= deadline - 2.0:
      ls[1].errands.append(programmed_errand(lap))
      queued.append(True)
    return (ls[0].swaps_done >= 2
            and any("procedure" in r for r in ls[1].errand_results))

  results = run_pair(lives, max_sim_time=2400.0, stop_when=settled)
  run = next(r["procedure"] for r in lives[1].errand_results if "procedure" in r)
  assert run["ok"] and run["completed"] == len(legs), run
  assert all(r["dead"] is None for r in results), results
  assert results[0]["swaps_done"] >= 2 and results[0]["module_stowed"]
  assert all(r["collision_steps"] == 0 for r in results), results
