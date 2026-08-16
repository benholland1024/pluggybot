"""Guards for the LCD module's display: faces, text/count, census (issue #13).

Every assertion here is one a build actually paid for. The two that would
have cost the most, and that fail loudly without their fix:

  - a census that counts the FENCE (the zone's own boundary is a long line of
    occupied cells, and a counter with no margin reports a garden full of
    objects), and
  - a face that is drawn on a module hanging unpowered on the rack, because
    "am I carrying it" was asked instead of "is the coupling conducting".
"""

import json

import mujoco
import numpy as np
import pytest

from pluggybot.hub.census import (
  MARGIN, Zone, count_objects, score, survey_route, true_count,
)
from pluggybot.hub.errand import DANCE_ROUTINE, census_errand, dance_errand
from pluggybot.hub.lifecycle import errands_for, world_screens
from pluggybot.hub.screen import ANXIOUS_FRAC, Screen, ScreenSet, face_for
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.telemetry.protocol import FACE_STATES, SCREEN_HINTS, SCREEN_MODES
from pluggybot.telemetry.recorder import FrameBuilder
from pluggybot.telemetry.scene import scene_dict, screen_map

META = json.load(open("models/home_world.meta.json"))
GARDEN = Zone.from_meta(next(z for z in META["zones"] if z["kind"] == "garden"))


@pytest.fixture(scope="module")
def home_model():
  return mujoco.MjModel.from_xml_path("models/home_world.xml")


@pytest.fixture
def home_data(home_model):
  data = mujoco.MjData(home_model)
  mujoco.mj_forward(home_model, data)
  return data


@pytest.fixture
def screen(home_model, home_data):
  s = Screen(home_model, home_data)
  s.sense(home_model, home_data, powered=True)   # as if seated on the fork
  return s


# ---- the display ------------------------------------------------------------


def test_an_unpowered_screen_shows_nothing_whatever_it_was_told(home_model,
                                                                home_data):
  """A module on the rack is DARK, and the criterion is electrical.

  The whole reason this is not "am I carrying it": a half-seated coupling --
  one pole conducting -- is a real failure mode of the two-point latch, and
  it is not a lit screen. The content is remembered, though: what the module
  would show survives the power cut, so a display picked back up is already
  showing its number rather than starting from idle.
  """
  s = Screen(home_model, home_data)
  s.show_count(4, "plants")
  assert s.flags == {"mode": "off", "powered": False}
  s.sense(home_model, home_data, powered=True)
  assert s.flags["mode"] == "count" and s.flags["count"] == 4


def test_the_module_hanging_in_its_bay_is_not_powered(home_model, home_data):
  """The real world, not a stub: at qpos0 every module hangs on the rack."""
  s = Screen(home_model, home_data)
  s.sense(home_model, home_data)
  assert s.powered is False
  assert s.flags["mode"] == "off"


def test_faces_and_hints_stay_inside_the_two_repo_vocabulary(screen):
  """The website renders a component per face name. An unknown string is a
  blank screen over there and no error over here, so it fails HERE."""
  for state in ("EXPLORE", "GO_CHARGE", "CHARGE", "SWAP_PICK", "USE_TOOL",
                "SWAP_RETURN", "DONE", "WHATEVER"):
    face, hint = face_for(state)
    assert face in FACE_STATES and hint in SCREEN_HINTS
  with pytest.raises(ValueError):
    screen.face("smug")
  with pytest.raises(ValueError):
    screen.face("happy", hint="moonwalk")


def test_every_mode_the_vocabulary_names_is_reachable(screen):
  seen = set()
  screen.face("idle")
  seen.add(screen.flags["mode"])
  screen.show_text("HI")
  seen.add(screen.flags["mode"])
  screen.show_count(3, "plants")
  seen.add(screen.flags["mode"])
  screen.blank()
  seen.add(screen.flags["mode"])
  assert seen == set(SCREEN_MODES)


def test_a_flat_battery_looks_worried(screen):
  assert face_for("GO_CHARGE", battery_frac=0.9)[0] == "determined"
  assert face_for("GO_CHARGE", battery_frac=ANXIOUS_FRAC - 0.01)[0] == "worried"
  assert face_for("CHARGE")[0] == "sleepy"


def test_content_claims_the_screen_and_release_hands_it_back(screen):
  """An errand's number must not be overwritten by the resting face two
  milliseconds later -- `held` is what stops the lifecycle's per-step update
  from stomping on it, and `release()` is the hand-back at a state change."""
  assert screen.held is False
  screen.show_count(4, "plants")
  assert screen.held is True
  screen.release()
  assert screen.held is False
  screen.face("curious", hold=True)
  assert screen.held is True


def test_only_real_changes_are_published(screen):
  screen.face("happy", "none")
  before = screen.changes
  screen.face("happy", "none")
  assert screen.changes == before, "an identical face was republished"
  screen.face("happy", "blink")
  assert screen.changes == before + 1


# ---- the wire ---------------------------------------------------------------


def test_screens_ride_in_frames_sparsely_and_re_ship_on_a_keyframe(home_model,
                                                                   home_data):
  """The same rule activities and boards follow, and for the same reason: a
  face is not a pose, so this block is the only record of it in the stream."""
  screens = world_screens(home_model, home_data)
  s = next(iter(screens))
  s.sense(home_model, home_data, powered=True)
  builder = FrameBuilder(home_model, home_data, keyframe_s=5.0, screens=screens)
  assert builder.header()["screens"] == ["module_lcd"]

  first = builder.build()
  assert first["key"] is True and "screens" in first

  home_data.time = 0.1
  assert "screens" not in builder.build(), "unchanged screen was re-sent"

  s.show_count(4, "plants")
  home_data.time = 0.2
  assert builder.build()["screens"]["module_lcd"]["count"] == 4

  home_data.time = 5.5                     # the next keyframe
  frame = builder.build()
  assert frame.get("key") is True and "screens" in frame


def test_the_scene_says_which_geom_carries_the_face(home_model):
  """Without this mapping a client has a face and nowhere to paint it: the
  telemetry key is `module_lcd` and the geom is `module_lcd_screen`."""
  scene = scene_dict(home_model, "home_world", META)
  panel = scene["screens"]["module_lcd"]
  assert panel["geom"] == "module_lcd_screen"
  geoms = {g["name"] for b in scene["bodies"] if b["name"] == "module_lcd"
           for g in b["geoms"]}
  assert panel["geom"] in geoms, "the scene points at a geom it never shipped"


def test_the_panel_normal_points_out_of_the_module(home_model):
  """A screen painted on the wrong face is a screen nobody ever sees. The
  panel sits at -x on the module, so its outward normal is -x."""
  panel = screen_map(home_model)["module_lcd"]
  assert panel["normal"] == [-1.0, 0.0, 0.0]
  # ...and it is the THIN axis that is normal, not merely the first one.
  assert panel["size"].index(min(panel["size"])) == 0


def test_the_panel_is_big_enough_to_read(home_model):
  """Issue #28's acceptance is legibility at visitor camera distance. The
  panel the swap was built around was 28 x 40 mm; anything that quietly
  shrinks it back is a regression in the feature, not in the geometry."""
  panel = screen_map(home_model)["module_lcd"]
  _, width, height = panel["size"]
  assert width >= 0.05 and height >= 0.07


# ---- the census -------------------------------------------------------------


def plant_grid(zone: Zone, plants, res=0.05, fence=True,
               dropout=0.0, seed=0) -> OccupancyGrid:
  """A believable garden map: a fence around the rectangle, plants inside.

  `dropout` is the honest part. A real scan does not paint a solid line along
  a fence -- the LIDAR loses returns to dark and specular surfaces, and its
  rays fan out with range -- so the boundary arrives as a dotted line, and
  the dots are exactly plant-sized.
  """
  grid = OccupancyGrid(zone.min[0] - 1, zone.min[1] - 1,
                       zone.max[0] + 1, zone.max[1] + 1, resolution=res)
  grid.grid[:] = -1.0                       # everything seen, and free
  rng = np.random.default_rng(seed)
  if fence:
    for t in np.arange(zone.min[0], zone.max[0], res / 2):
      for y in (zone.min[1], zone.max[1]):
        if rng.random() < dropout:
          continue
        ix, iy = grid.world_to_cell(t, y)
        grid.grid[iy, ix] = 2.0
    for t in np.arange(zone.min[1], zone.max[1], res / 2):
      for x in (zone.min[0], zone.max[0]):
        if rng.random() < dropout:
          continue
        ix, iy = grid.world_to_cell(x, t)
        grid.grid[iy, ix] = 2.0
  for px, py in plants:
    ix, iy = grid.world_to_cell(px, py)
    grid.grid[iy:iy + 2, ix:ix + 2] = 2.0   # ~2 cells: an 80 mm stem
  return grid


PLANTS = [(6.5, -0.8), (8.8, 1.0), (7.6, 4.5), (9.2, 5.2)]


@pytest.mark.parametrize("dropout,phantoms", [(0.3, 12), (0.5, 45)])
def test_the_census_does_not_count_the_fence(dropout, phantoms):
  """THE failure this task is one line away from.

  A SOLID fence is rejected by the span filter -- it is one enormous
  component -- which is why this test uses a scanned one. Drop 30 % of the
  returns, as the real LIDAR does, and the boundary becomes a dotted line of
  fragments that are plant-sized in every dimension the counter measures.
  Nothing but the margin rejects those.
  """
  grid = plant_grid(GARDEN, PLANTS, dropout=dropout)
  assert count_objects(grid, GARDEN)["count"] == 4
  # Pin the defect, so the fix's premise cannot rot: without the margin the
  # garden reports dozens of plants and does it with total confidence.
  assert count_objects(grid, GARDEN, margin=0.0)["count"] >= phantoms


def test_the_census_ignores_things_that_are_not_object_sized():
  """A couch is not a plant. Span, not cell count, is the discriminator --
  a long thin wall run has few cells per row and is still not an object."""
  grid = plant_grid(GARDEN, [(6.5, 0.0)], fence=False)
  ix, iy = grid.world_to_cell(8.0, 3.0)
  grid.grid[iy:iy + 12, ix:ix + 12] = 2.0        # a 600 mm blob
  tally = count_objects(grid, GARDEN)
  assert tally["count"] == 1
  assert tally["objects"][0]["x"] == pytest.approx(6.5, abs=0.1)


def test_coverage_reports_how_much_was_actually_seen():
  """A right answer off a third of the zone is a lucky guess, and the score
  has to be able to tell the difference."""
  plants = [(6.5, -0.8), (8.8, 1.0)]
  grid = plant_grid(GARDEN, plants)
  assert count_objects(grid, GARDEN)["coverage"] > 0.95
  half = plant_grid(GARDEN, plants)
  ix, _ = half.world_to_cell(7.5, 0.0)
  half.grid[:, ix:] = 0.0                        # unknown: never scanned
  tally = count_objects(half, GARDEN)
  assert tally["coverage"] < 0.6
  assert tally["count"] == 1, "counted an object in unscanned space"


def test_ground_truth_comes_out_of_the_world(home_model):
  """Hidden ground truth, and hidden from the ROBOT: the evaluator reads the
  model. Written as a query rather than as the number 4, so a fifth plant
  re-scores the task instead of quietly failing every run."""
  assert true_count(home_model, GARDEN) == 4
  living = Zone.from_meta(next(z for z in META["zones"] if z["name"] == "living"))
  assert true_count(home_model, living) == 0


def test_ground_truth_and_the_counter_exclude_the_same_border(home_model):
  """A plant inside the counter's discarded margin would be scored as MISSED
  when it was never countable -- the two margins have to be one number."""
  narrow = Zone(name="edge", min=(6.0, -1.2), max=(7.0, -0.4))
  assert true_count(home_model, narrow, margin=MARGIN) == 0
  assert true_count(home_model, narrow, margin=0.0) == 1


def test_the_score_is_code_and_says_when_it_is_wrong():
  assert score(4, 4, 1.0)["correct"] is True
  wrong = score(3, 4, 0.5)
  assert wrong["correct"] is False and wrong["error"] == -1


def test_the_survey_route_starts_at_the_door():
  """A route that opens with a drive across the zone and back is a route
  that spends the battery on travel instead of on looking."""
  entry = (5.4, 0.7)
  route = survey_route(GARDEN, entry=entry)
  assert len(route) == 4
  first = min(route, key=lambda p: (p[0] - entry[0]) ** 2 + (p[1] - entry[1]) ** 2)
  assert route[0] == first
  for x, y in route:
    assert GARDEN.contains(x, y, margin=1.0), "a vantage point inside a wall"


# ---- the errands ------------------------------------------------------------


def test_the_lcd_errands_are_on_the_menu():
  census = errands_for("census", "home")
  assert len(census) == 1 and census[0].module == "module_lcd"
  dance = errands_for("dance", "home")
  assert len(dance) == 1 and dance[0].module == "module_lcd"
  with pytest.raises(ValueError):
    errands_for("census", "room_hub")        # nothing countable in there


def test_the_showcase_queue_leaves_both_surfaces_marked(home_model):
  """What the site serves: one errand that puts ink on a board and one that
  puts a face on the screen, so a single recording exercises both."""
  from pluggybot.hub.lifecycle import board_book
  book = board_book("home")
  queue = errands_for("showcase", "home", book)
  assert [e.module for e in queue] == ["module_pen", "module_lcd"]


def test_the_dance_is_a_routine_not_a_random_walk():
  """Same sequence every time, or "did it dance" has no answer. And it must
  come back to where it started: a dance that ends three metres away has
  driven off, whatever its expressions said."""
  net_turn = sum(w * s for _, _, _, _, w, s in DANCE_ROUTINE)
  net_travel = sum(v * s for _, _, _, v, _, s in DANCE_ROUTINE)
  assert abs(net_travel) < 0.05, "the routine walks away from its spot"
  assert abs(net_turn) < 2.0, "the routine ends facing somewhere else"
  for _, face, hint, _, _, _ in DANCE_ROUTINE:
    assert face in FACE_STATES and hint in SCREEN_HINTS


def test_a_screen_set_presents_the_activity_duck_type(home_model, home_data):
  """One code path in FrameBuilder diffs activities, boards AND screens. The
  duck type is the contract that lets it."""
  screens = world_screens(home_model, home_data)
  assert isinstance(screens, ScreenSet)
  assert screens.names == list(screens.snapshot())
  assert len(screens) == 1


def test_errand_construction_needs_no_physics():
  """An overseer picks errands from a menu (issue #15); building one must not
  require a compiled world."""
  assert census_errand(GARDEN).use_at == survey_route(GARDEN)[0]
  assert dance_errand((1.0, 2.0)).use_at == (1.0, 2.0)
