"""The visual-hint freeze (issue #66, M13's freeze point).

`VISUAL_HINTS` is a tuple of strings and a builder needs more than that, so
`protocol/hints.json` carries -- per hint -- one conformance body in exactly
`scene_dict`'s shape plus a machine-readable rule. Two lanes of M13 build
against it in parallel, which is the whole reason it exists: an artist-shaped
session in the website and a generator-shaped session here cannot be allowed
to guess differently about what a `couch` marker is.

What these hold down:

  1. A NAME CANNOT BE FROZEN WITHOUT BEING DESCRIBED. Adding to `VISUAL_HINTS`
     and forgetting the fixture is the failure mode the freeze exists to
     prevent -- the art lane would build against a name nobody defined.
  2. THE FIXTURE CANNOT DRIFT FROM `scene_dict`. It is emitted through the
     real transpiler, and a hand-written example that has silently diverged
     from the real body shape is worse than no example, because it looks
     authoritative.
  3. ADDING A HINT IS NOT A VERSION BUMP. The browser falls back to raw
     primitives for a hint it has no builder for, which is the asymmetry that
     lets the sim ship a name before the art exists.
"""

import json
from pathlib import Path

import pytest

from pluggybot.telemetry import hints as hintgen
from pluggybot.telemetry.protocol import PROTOCOL_VERSION, VISUAL_HINTS

FIXTURE = Path(__file__).parent.parent / "protocol" / "hints.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
  return json.loads(FIXTURE.read_text())


def test_every_hint_has_a_conformance_body(fixture):
  """THE acceptance criterion, and the one that has to be able to fail.

  Shown to fail by appending a name to `VISUAL_HINTS` without adding a
  `Marker` for it: `missing` is that name and the fixture cannot be
  regenerated to hide it, because `build()` iterates MARKERS.
  """
  described = set(fixture["hints"])
  missing = [h for h in VISUAL_HINTS if h not in described]
  assert not missing, (
    f"frozen in VISUAL_HINTS but described nowhere: {missing}. Add a Marker "
    f"to telemetry/hints.py and regenerate: "
    f"uv run python -m pluggybot.telemetry.hints")
  # ...and the other direction: a rule for a hint nobody may emit is a
  # builder built against a name the sim will refuse to ship.
  assert described == set(VISUAL_HINTS)
  assert fixture["vocabulary"] == list(VISUAL_HINTS), \
      "the fixture's own copy of the vocabulary has drifted"


def test_the_fixture_is_not_stale(fixture):
  """Regenerating must reproduce the committed file byte for byte.

  This is what makes claim 2 true: the bodies come out of `scene_dict`, so if
  its body record ever grows a field this fails until somebody regenerates --
  rather than the fixture quietly describing last month's shape.
  """
  assert fixture == hintgen.build(), \
      "stale: uv run python -m pluggybot.telemetry.hints"


def test_adding_a_hint_is_not_a_version_bump(fixture):
  """The vocabulary grew by six names in M13 and the version did not move.

  Deliberate, and it is the asymmetry the whole two-lane plan rests on:
  `scene.py` RAISES on a hint outside the vocabulary (the sim refuses to
  describe what it has no word for) while `sceneGraph.ts` FALLS BACK to raw
  primitives for a hint it has no builder for (the browser degrades). So the
  sim may ship a hint before the art exists and the world renders plainly
  rather than breaking.
  """
  assert fixture["protocolVersion"] == PROTOCOL_VERSION == "0.14.0"


def test_the_conformance_bodies_are_the_real_body_shape(fixture):
  """Every conformance body has exactly the keys a scene's bodies have --
  checked against a REAL scene rather than a list written here, or this
  would pin the shape twice and drift in the same direction."""
  scene = json.loads((FIXTURE.parent / "scene.home_world.json").read_text())
  want = set(scene["bodies"][0])
  for hint, spec in fixture["hints"].items():
    assert set(spec["body"]) == want, hint
    assert spec["body"]["visual"] == hint, \
        f"{hint}'s conformance body is not hinted as {hint}"
    for geom in spec["body"]["geoms"]:
      assert set(geom) == set(scene["bodies"][1]["geoms"][0]), hint


@pytest.mark.parametrize("hint", VISUAL_HINTS)
def test_every_rule_is_machine_readable(hint, fixture):
  """A builder switches on these, so they are closed sets rather than prose.
  The `note` is for a person; nothing may need to parse it."""
  spec = fixture["hints"][hint]
  assert spec["build"] in hintgen.BUILD_MODES
  assert isinstance(spec["collides"], bool)
  assert spec["markers"] >= 1
  assert set(spec["axes"].values()) <= set(hintgen.AXIS_SLOTS)
  if spec["markers"] == 1:
    assert spec["marker"] == spec["body"]["geoms"][0]["type"]
    # A single-marker hint has to say what its axes MEAN, or a builder has
    # three numbers and no way to tell height from depth.
    assert spec["axes"], f"{hint} is one primitive and describes no axis"
  else:
    # `rack` and `robot`: the primitives ARE the shape, so there is nothing
    # to name axes on -- the builder is told to keep them instead.
    assert spec["marker"] == "many" and spec["build"] == "reskin"
    assert spec["axes"] == {}
  assert spec["note"].strip()


def test_furniture_is_exactly_one_box(fixture):
  """Issue #66's second recorded decision, pinned rather than intended.

  Three names (`couch`/`bed`/`table`) beat one `furniture` precisely because a
  builder can key on the NAME instead of on proportions -- and the fallback of
  keying on proportions only works while there is one box to take proportions
  of. `largestBox` selects by VOLUME, so a couch modelled as a frame plus
  cushions could hand a builder the wrong one. If the expanded house wants
  furniture this cannot describe, that is a bug on #66 rather than something
  to work around in a builder.
  """
  assert set(hintgen.ONE_BOX_HINTS) <= set(VISUAL_HINTS)
  for hint in hintgen.ONE_BOX_HINTS:
    spec = fixture["hints"][hint]
    assert spec["markers"] == 1 and spec["marker"] == "box", hint
    assert len(spec["body"]["geoms"]) == 1, hint
    assert {"height"} <= set(spec["axes"]), hint


def test_the_robot_hint_rides_the_real_root_body(fixture):
  """`robot` is the only hint whose conformance body is a REAL body name, and
  the only dynamic one. Both matter to a builder: it anchors on `pluggybot`
  and its pose is overwritten by every telemetry frame, unlike every piece of
  scenery, whose scene pose is final."""
  body = fixture["hints"]["robot"]["body"]
  assert body["name"] == "pluggybot" and body["robot"] == "pluggybot"
  assert body["dynamic"] is True
  assert all(spec["body"]["dynamic"] is False
             for h, spec in fixture["hints"].items() if h != "robot")


def test_the_sim_still_refuses_a_word_it_does_not_have():
  """The other half of the asymmetry, and the reason the freeze is worth
  anything: the generator cannot ship a hint that is not in the vocabulary,
  so a typo in a sidecar is a failed build rather than a body that silently
  renders as a grey box."""
  import mujoco

  from pluggybot.telemetry.scene import scene_dict

  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body name="pluggybot"><freejoint/>'
    '<geom type="box" size="0.1 0.1 0.1"/></body>'
    '<body name="sofa"><geom type="box" size="1 1 1"/></body>'
    "</worldbody></mujoco>")
  with pytest.raises(ValueError, match="unknown visual hints"):
    scene_dict(model, "typo", meta={"visualHints": {"sofa": "couch_bed"}})
  # ...and the frozen name goes through, which is what says the guard is
  # checking the vocabulary rather than refusing everything.
  ok = scene_dict(model, "fine", meta={"visualHints": {"sofa": "couch"}})
  assert [b["visual"] for b in ok["bodies"] if b["name"] == "sofa"] == ["couch"]
