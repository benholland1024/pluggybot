"""The humanoid finds a place (issue #559, stage 3; `humanoid/places.py`):
the quadruped's search with the humanoid's look -- its head's navigation
camera, the place tags in it laid through its model of itself and its
stance into its map -- and the readings a procedure walks up to them by
(`find.*`). Each rule pinned with the robot set down where the rule asks, a
look or two rendered, nothing flown; the search across the house is
`scripts/solve.py --feature drawer`'s.
"""

import math

import mujoco
import pytest

from pluggybot.home import storeroom
from pluggybot.procedure import steps as st


@pytest.fixture(scope="module")
def counter():
  """The humanoids' house, one robot in the storeroom's doorway facing its
  counter -- turned a little off square, so its stance's turn is in every
  look -- its map begun: (model, data, body)."""
  from pluggybot.body import body_for
  from pluggybot.humanoid import world as hw
  model = hw.home_spec(first_at=(22.8, -3.0)).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  body.start_at(22.8, -3.0, 0.15)
  body.run(body.hold_routine(0.5))
  yield model, data, body
  body.close()


def test_a_look_lays_the_counters_tags_where_they_are_and_the_row_its_facing(counter):
  """From the storeroom's doorway, 5 m off, the navigation camera reads the
  row's 120 mm tags; each goes into the map where it stands (its belief
  here the truth), and the row -- a fixture -- gives the wall's facing."""
  model, data, body = counter
  seen = body.mission.look_for_places()
  assert set(storeroom.TAGS.values()) <= set(seen)
  for prop, tag in storeroom.TAGS.items():
    p = body.places.get(tag)
    x, y, _, _ = storeroom.tag_at(prop)
    assert math.hypot(p.x - x, p.y - y) < 0.15, (prop, p.x, p.y)
  facing, how = body.places.facing(storeroom.TAGS["drawer"])
  assert how == "fixture" and abs(math.remainder(facing - math.pi, 2 * math.pi)) < math.radians(4)


def test_its_places_are_the_tags_its_world_puts_up(counter):
  """What a look records (`place_tags`, off the world it is in) is what its
  procedures may name in a `find` (`world_config`'s `places`): a tag on one
  list and not the other is a find that can never succeed."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  model, data, body = counter
  assert set(body.places.ids) == set(world_config(HUMANOID_HOME)["places"])


def test_find_says_where_it_found_its_place_in_the_map(counter):
  """`find` of a place in view: ok at once, and `find.x`, `find.y` and
  `find.facing` read where the robot remembers it -- what a procedure walks
  up to it by; a find that failed leaves no reading."""
  model, data, body = counter

  class Life:
    pass
  life = Life()
  life.body, life.data, life.model = body, data, model
  out = body.run(st._find(life, {"tag": storeroom.TAGS["door"], "x": 25.2, "y": -2.4,
                                 "patience": 30.0}))
  assert out["ok"], out
  p = body.places.get(storeroom.TAGS["door"])
  rep = life.reports["find"]
  assert (rep["x"], rep["y"]) == (p.x, p.y)
  assert rep["facing"] == body.places.facing(storeroom.TAGS["door"])[0]
  out = body.run(st._find(life, {"tag": 999, "x": 25.2, "y": -2.4, "patience": 1.0}))
  assert not out["ok"] and "find" not in life.reports


def test_places_ride_a_restart_with_the_map_and_go_at_a_true_death(counter):
  """Its places are kept with the map they are laid in (#419's rule): back
  with the map, gone with it, and forgotten at a true death."""
  model, data, body = counter
  body.mission.look_for_places()
  assert len(body.places)
  state, arrays = body.kept_state()
  kept = {p.tag: (p.x, p.y) for p in body.places}
  body.places.forget()
  body.restore_kept(state, arrays)
  assert {p.tag: (p.x, p.y) for p in body.places} == kept
  body.places.forget()
  body.restore_kept(state, {k: v for k, v in arrays.items() if k != "grid"})
  assert not len(body.places), "no map back, no places"
  body.restore_kept(state, arrays)
  body.forget_world()
  assert not len(body.places)


def test_the_humanoids_world_finds_what_it_puts_up():
  """Its world's places are the tags its world carries -- the counter's row
  among them -- and its procedures may name each in a `find`, reading
  where it found it."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config, world_facts
  from pluggybot.procedure import axes
  facts = world_facts(HUMANOID_HOME)
  assert set(storeroom.TAGS.values()) <= set(facts.places)
  assert set(facts.places) == set(world_config(HUMANOID_HOME)["places"])
  assert "find" in facts.verbs and set(axes.FIND_SENSORS) <= set(facts.sensors)
