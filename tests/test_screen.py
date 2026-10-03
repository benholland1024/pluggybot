"""Guards for the LCD module's display: faces, text/count (issue #13; the
census's counting is `test_census.py`'s).

Every assertion here is one a build actually paid for. The one that would
have cost the most, and that fails loudly without its fix: a face drawn on
a module hanging unpowered on the rack, because "am I carrying it" was
asked instead of "is the coupling conducting".
"""

import json

import mujoco
import pytest

from pluggybot.legs import world as lw
from pluggybot.lifecycle import world_screens
from pluggybot.tools.screen import ANXIOUS_FRAC, Screen, ScreenSet, face_for
from pluggybot.telemetry.protocol import FACE_STATES, SCREEN_HINTS, SCREEN_MODES
from pluggybot.telemetry.recorder import FrameBuilder
from pluggybot.telemetry.scene import scene_dict, screen_map

META = json.load(open("models/home_world.meta.json"))


@pytest.fixture(scope="module")
def home_model():
  """The served house, `home_quad`: its rack carries the LCD."""
  return lw.home_spec().compile()


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
  scene = scene_dict(home_model, "home_quad", META)
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


# ---- the set ---------------------------------------------------------------


def test_a_screen_set_presents_the_activity_duck_type(home_model, home_data):
  """One code path in FrameBuilder diffs activities, boards AND screens. The
  duck type is the contract that lets it."""
  screens = world_screens(home_model, home_data)
  assert isinstance(screens, ScreenSet)
  assert screens.names == list(screens.snapshot())
  assert len(screens) == 1
