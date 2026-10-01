"""Guards for the generated home world (issue #6) and its hint sidecar.

The generator is the single source for four artifacts that must agree: the
MJCF, the meta sidecar, the scene JSON the website renders, and the
constants the scripts navigate by. Most of these tests exist because those
four can drift silently -- a wall moved in Python while models/ still holds
last week's XML looks fine until a mission drives into it.
"""

import itertools
import json
from pathlib import Path

import mujoco
import pytest

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


def test_a_world_still_transpiles_without_a_sidecar(home_model):
  """A world with no generator meta keeps working -- hints are optional."""
  scene = scene_dict(home_model, "home_world")
  assert all(b["visual"] is None for b in scene["bodies"])
  assert "zones" not in scene


# ---- the world is actually navigable ----------------------------------------

def test_doorways_are_wide_enough_to_drive_through(home_model):
  """EVERY doorway is wider than the quadruped's inflated planning mask
  closes from both jambs, with a cell to spare -- a doorway the planner
  refuses is a wall, and the wing hangs off one of them."""
  from pluggybot.legs.body import QuadMission
  shut = 2 * QuadMission.INFLATION_CELLS * 0.05 + 0.05
  doors = {"divider": home.DOOR_DIV_X, "garden": home.DOOR_GARDEN_Y,
           "hall": home.DOOR_HALL_Y, "kitchen": home.DOOR_KITCHEN_Y,
           "workshop": home.DOOR_WORKSHOP_Y,
           # The second house (issue #215): its gate, its front door, and
           # the lab and the store off the lobby.
           "garden_2": home.DOOR_GARDEN_2_Y, "lobby": home.DOOR_LOBBY_Y,
           "lab": home.DOOR_LAB_Y, "store": home.DOOR_STORE_Y}
  for name, (lo, hi) in doors.items():
    assert hi - lo > shut, f"{name} doorway: its {hi - lo:.2f} m is shut by the inflation"


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
           # The middle street runs on to the loop at both ends, and each
           # property's ring of sidewalk meets the loop and the street.
           ("street", "street_north"), ("street", "street_south"),
           ("street", "sidewalk_north"), ("street", "sidewalk_2_north"),
           ("street", "sidewalk_south"), ("street", "sidewalk_2_south"),
           ("sidewalk", "sidewalk_north"), ("sidewalk", "sidewalk_south"),
           ("sidewalk_north", "sidewalk_west"), ("sidewalk_south", "sidewalk_west"),
           ("sidewalk_2", "sidewalk_2_north"), ("sidewalk_2", "sidewalk_2_south"),
           ("sidewalk_2_north", "sidewalk_east"), ("sidewalk_2_south", "sidewalk_east"),
           ("sidewalk_north", "street_north"), ("sidewalk_2_north", "street_north"),
           ("sidewalk_south", "street_south"), ("sidewalk_2_south", "street_south"),
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



def _touch(a: dict, b: dict) -> tuple[float, float]:
  """How far two zone rectangles overlap along x and along y: a shared
  border is one of them zero and the other positive."""
  return (min(a["max"][0], b["max"][0]) - max(a["min"][0], b["min"][0]),
          min(a["max"][1], b["max"][1]) - max(a["min"][1], b["min"][1]))


def test_the_middle_street_runs_through_the_sidewalk_to_the_loop():
  """Ben, after #215 deployed: the road between the two houses was cut off
  at both ends by the sidewalk band. The street now meets the loop's north
  and south legs directly -- a border along its whole width -- and no
  sidewalk lies across it, so the band is two rings, one per property, that
  never touch. Shown to fail by putting `street` back to PROPERTY_Y."""
  z = {zone["name"]: zone for zone in home.ZONES}
  street = z["street"]
  width = home.STREET_X[1] - home.STREET_X[0]
  for leg in ("street_north", "street_south"):
    across, along = _touch(street, z[leg])
    assert along == 0 and across == pytest.approx(width), \
        f"the middle street does not meet {leg} along its whole width"
  for side in ("north", "south"):
    west, east = z[f"sidewalk_{side}"], z[f"sidewalk_2_{side}"]
    assert min(_touch(west, east)) < 0, \
        f"sidewalk_{side} and sidewalk_2_{side} meet: the band still crosses the street"
  for zone in home.ZONES:
    if zone["name"].startswith("sidewalk"):
      across, along = _touch(street, zone)
      assert across <= 0 or along <= 0, f"{zone['name']} overlaps the street"


def test_every_room_names_its_building_and_nothing_outdoors_does(home_model):
  """The fact the site paints walls by (`protocol.BUILDINGS`): each room says
  which house it is in -- the house with the rack, the facility with the
  lab -- and nothing outdoors claims a building. The sim's own wall colour
  does not move: that is what the robot's cameras render."""
  from pluggybot.telemetry.protocol import BUILDINGS
  data = mujoco.MjData(home_model)
  mujoco.mj_forward(home_model, data)
  rooms = [zone for zone in home.ZONES if zone["kind"] == "room"]
  assert rooms and all(zone.get("building") in BUILDINGS for zone in rooms)
  assert not [zone["name"] for zone in home.ZONES
              if zone["kind"] != "room" and "building" in zone]

  def building_at(x, y):
    return next(zone["building"] for zone in rooms
                if zone["min"][0] <= x <= zone["max"][0]
                and zone["min"][1] <= y <= zone["max"][1])
  from pluggybot.legs.world import rack_pose
  rack = rack_pose()
  assert building_at(rack[0], rack[1] + 0.5) == "house"      # the rack stands on the living room's wall
  cage = data.xpos[home_model.body("lab_cage").id]
  assert building_at(cage[0], cage[1]) == "facility"


def test_the_scene_transpiler_rejects_an_unknown_building(home_model):
  """A typo'd building is a wall the site cannot paint: refused at the
  transpile, on the hints' terms."""
  meta = {"zones": [{"name": "kitchen", "kind": "room", "building": "castle",
                     "min": [0, 0], "max": [1, 1]}]}
  with pytest.raises(ValueError, match="unknown buildings"):
    scene_dict(home_model, "home_world", meta=meta)


def test_every_plate_is_hinted_and_says_what_it_is_for(home_model, meta):
  """The site draws a glyph on each pressure plate so a visitor can tell the
  shock plate from the feed plate (`protocol.PLATE_PURPOSES`). Every plate
  body in the world -- the garden's and the lab's three -- is hinted
  `plate`, names a purpose the vocabulary knows, and rides `scene.plates`
  by its body name. The glyph is the site's: the sim's plate rgba, what the
  robot's cameras render, is one colour for all four."""
  from pluggybot.telemetry.protocol import PLATE_PURPOSES
  pads = {home_model.body(i).name for i in range(home_model.nbody)
          if home_model.body(i).name.endswith("_plate")}
  assert pads == {"garden_plate", "lab_shock_plate", "lab_feed_plate", "lab_toy_plate"}
  assert set(meta["plates"]) == pads
  for name in pads:
    assert meta["visualHints"][name] == "plate", name
    assert meta["plates"][name]["purpose"] in PLATE_PURPOSES, name
  assert {meta["plates"][n]["purpose"] for n in pads} == set(PLATE_PURPOSES), \
      "a purpose in the vocabulary has no plate, or two plates share one"
  scene = scene_dict(home_model, "home_world", meta=meta)
  assert scene["plates"] == meta["plates"]
  rgba = {tuple(home_model.geom(f"{n}_pad").rgba) for n in pads}
  assert len(rgba) == 1, "the plates' colour is telling the cameras which is which"


def test_the_scene_transpiler_rejects_an_unknown_plate_purpose(home_model):
  meta = {"plates": {"garden_plate": {"purpose": "tickle"}}}
  with pytest.raises(ValueError, match="unknown plate purposes"):
    scene_dict(home_model, "home_world", meta=meta)

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
  from pluggybot.legs.world import home_spec
  _, world = body_census(home_spec().compile())      # the census needs a robot
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

def test_the_camera_extent_is_written_down_not_derived(home_model):
  """MuJoCo scales every camera's near plane by `statistic.extent`, derived
  from the bounding box unless written down: the house pins it
  (`CAMERA_EXTENT_M`) so growing the world cannot move a camera's near
  plane, and the quadruped's world sets its own off it in metres
  (`legs.world.NEAR_M`; that a bay's tags are read through it is
  `test_quad_rack.py::test_the_nose_camera_reads_a_bays_tags_from_its_working_pose`)."""
  from pluggybot.legs import world as lw
  assert home_model.stat.extent == pytest.approx(home.CAMERA_EXTENT_M)
  quad = lw.home_spec().compile()
  assert quad.stat.extent == pytest.approx(home.CAMERA_EXTENT_M)
  assert quad.vis.map.znear * quad.stat.extent == pytest.approx(lw.NEAR_M)
