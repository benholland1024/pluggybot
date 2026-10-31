"""A humanoid is told its own body and world (issue #559, stage 3): the
constitutions' body paragraph and every rule in its words -- it charges
standing at its station by a cable it plugs into itself, it has no rack,
no tool and no question jobs -- its reward table its world's rows alone,
and its procedures in its hands' vocabulary as its practice states it.

⚠ THE QUADRUPED'S TEXT IS THE TEXT: the files and the rules are what the
served pair reads, byte for byte, and a humanoid's are them swapped at load
(`constitution.for_body`, `overseer.for_body`) until the quadruped goes.
Every rule here is pinned on the prompt as it is built, nothing flown.
"""

import itertools
import re
import textwrap
from dataclasses import replace

import pytest

from pluggybot.lifecycle import HUMANOID_HOME, QUAD_HOME, board_book, world_facts
from pluggybot.mind import acts
from pluggybot.mind import constitution as c
from pluggybot.mind import overseer as ov
from pluggybot.mind import practice
from pluggybot.mind.overseer import Menu
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st
from pluggybot.economy.scoring import default_table

#: What no humanoid is told of: the quadruped's body, its dock and its rack,
#: the tools on it, the question jobs written on a board.
QUADRUPED_WORDS = re.compile(
  r"\b(four-legged|small dog|dock|lie down|lying|lie on|rack|bay [A-E]|bays|fork|bumper|"
  r"tools?|modules?|module_\w+|stow|pen|claw|LCD|needsAnswer|whole number of at most)\b",
  re.I)


def _sections(world, **flags):
  menu = Menu.for_world(world, board_book(world))
  menu = replace(menu, procedures=flags.pop("procedures", False),
                                wiki=flags.get("wiki", False),
                                tickets=flags.get("tickets", False),
                                look=flags.get("look", False))
  body = "humanoid" if world == HUMANOID_HOME else "quadruped"
  thoughts = ThoughtFiles(None, constitution=c.resolve(body=body))
  return ov.system_sections(thoughts, menu, default_table(), name="Luca",
                            procedures=menu.procedures, **flags)


#: The prompt's switches, each one way and the other: what a section is in
#: or out on.
FLAGS = ("appetite", "mortal", "hearts", "event_map", "seeded", "procedures", "acts",
         "wiki", "tickets", "look")


def _configurations():
  yield {}
  for on in FLAGS:
    yield {on: True}
  yield {f: True for f in FLAGS} | {"others": ("Rowan",)}
  # ...the deployed arm's: an event map, unseeded (`UNSEEDED_RULE`)
  yield {f: True for f in FLAGS} | {"seeded": False, "others": ("Rowan",)}
  yield {"event_map": True, "mortal": True, "others": ("Rowan",), "acts": True}


def test_the_humanoid_reads_its_own_body_and_the_rest_of_its_constitution_unchanged():
  """Every library file opens with the humanoid's paragraph and keeps its
  disposition; it hands over no answer, as the quadruped's never does
  (`test_constitution`'s regexes), and names no dock, rack or tool."""
  from test_constitution import _ACT, _TACTIC
  for name in c.names():
    quad, hum = c.load(name), c.for_body(c.load(name), "humanoid")
    assert c.for_body(quad, "quadruped") is quad, "the quadruped's is the file's"
    assert hum.name == name and hum.sha == c.sha_of(hum.text) != quad.sha
    assert hum.text.startswith("You are a humanoid robot")
    assert not QUADRUPED_WORDS.search(hum.text), QUADRUPED_WORDS.search(hum.text)
    assert not re.search(r"\d", hum.text) and "->" not in hum.text and "%" not in hum.text
    assert not _TACTIC.search(hum.text) and not _ACT.search(hum.text)
    # ...the rest of it, sentence for sentence
    tail = quad.text.split("\n\n", 1)[1]
    for old, new in c.BODY_SWAPS["humanoid"][1:]:
      tail = tail.replace(old, new)
    assert hum.text.split("\n\n", 1)[1] == tail
  assert c.resolve("curious", body="humanoid") == c.for_body(c.load("curious"), "humanoid")


def test_a_file_whose_sentence_moved_is_refused_rather_than_read_half_swapped():
  moved = c.Constitution.of("default", c.load("default").text.replace("small dog", "dog"))
  with pytest.raises(ValueError, match="no longer says"):
    c.for_body(moved, "humanoid")


@pytest.mark.parametrize("flags", list(_configurations()), ids=lambda f: ",".join(f) or "bare")
def test_no_humanoid_prompt_names_the_quadrupeds_world(flags):
  text = "\n\n".join(t for _, t in _sections(HUMANOID_HOME, **flags))
  hit = QUADRUPED_WORDS.search(text)
  assert hit is None, text[max(0, hit.start() - 120):hit.end() + 60]


def test_every_swap_is_read_in_some_humanoid_prompt_and_the_quadrupeds_is_its_own():
  """A swap whose sentence moved would leave the quadruped's words in a
  humanoid's prompt silently: each is found in some configuration's
  quadruped text, and none of a quadruped's prompt is swapped."""
  quad = "\n\n".join("\n\n".join(t for _, t in _sections(QUAD_HOME, **f))
                     for f in _configurations())
  for old, new in ov.BODY_SWAPS["humanoid"]:
    assert old in quad, f"BODY_SWAPS' sentence moved: {old[:60]!r}"
  assert ov.for_body(quad, "quadruped") == quad


def test_a_humanoid_is_shown_only_the_reward_rows_its_world_can_pay():
  """None of the quadruped's jobs is offered in its house: the table is
  the charge's, a procedure's and a ticket's (`overseer.ALWAYS_SHOWN`) and
  its own jobs', the hands' on the storeroom's counter (stage 3); and the
  quadruped's is every row it always was."""
  from pluggybot.economy.tasks import HAND_JOBS
  rows = {r["task"] for r in default_table().as_context(
    challenges=True, kinds=Menu.for_world(HUMANOID_HOME).kinds)}
  assert rows == set(ov.ALWAYS_SHOWN) | set(HAND_JOBS)
  assert Menu.for_world(QUAD_HOME).kinds is None
  menu = Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME))
  assert menu.body == "humanoid" and menu.programs == () and menu.boards


def test_a_humanoids_procedures_are_its_hands_and_its_walks_every_switch_on():
  """`world_facts` of its house is its practice's hands -- `measure`, a
  guarded move, a watched tag, a verb as an `if`'s test, the maths -- and
  its walks over its map; its pack and wallet, never the quadruped's
  bumper, arm or tools."""
  facts = world_facts(HUMANOID_HOME)
  hands = st.hand_facts()
  assert facts.verbs == st.SERVED_WALK_VERBS + st.PLACE_VERBS + hands.verbs
  assert set(hands.sensors) <= set(facts.sensors) and "bumper" not in facts.sensors
  assert {"battery.wh", "battery.frac", "points"} <= set(facts.sensors)
  assert facts.axes == axes.BODY_AXES["humanoid"] and facts.hands == axes.HANDS
  assert facts.verb_tests and facts.math and facts.guarded and facts.tools == ()
  assert world_facts(QUAD_HOME).verbs[:5] == st.BODY_VERBS, "the quadruped's are its own"


def test_a_humanoids_procedure_rule_is_its_practices_language_and_body():
  """One vocabulary, told one way: the served rule carries the practice's
  sections byte for byte, less an attempt's set-out; its worked example
  compiles on the house as served (#434) and hands over no survival
  policy; every verb, sensor and axis it may name is listed, and none of the
  quadruped's."""
  places = world_facts(HUMANOID_HOME).places
  rule = ov.hands_procedure_rule(places)
  sections = dict(practice.system_sections(st.served_hand_facts(places=places)))
  assert sections["the language"] in rule
  assert sections["your body"].replace("Both grippers start shut. ", "") in rule
  example = rule[rule.index("  def wave"):rule.index("Statements:")]
  lang.compile_procedure(textwrap.dedent(example).strip("\n") + "\n", world_facts(HUMANOID_HOME))
  for word in ("charge", "battery", "station", "dock"):
    assert word not in example, word
  facts = world_facts(HUMANOID_HOME)
  for v in facts.verbs:
    assert f"  {v}(" in rule, v
  for s in facts.sensors:
    assert f"  {s} --" in rule, s
  for a in facts.axes:
    assert f"  {a}:" in rule, a
  assert "  shoulder:" not in rule and "fetch(" not in rule
  text = "\n\n".join(t for _, t in _sections(HUMANOID_HOME, procedures=True))
  assert rule in text, "the humanoid's prompt carries it"


def test_a_humanoids_claim_about_its_station_is_checked():
  """`tell`'s worked claim in a humanoid's words is one code can check."""
  assert acts.check_claim("the charging station is free", charging=False) == (
    "the charging station is free", True)
  assert acts.check_claim("the charge bay is taken", charging=True)[1]
  text = "\n\n".join(t for _, t in _sections(HUMANOID_HOME, others=("Rowan",), acts=True))
  assert '("the charging station is free")' in text


def test_a_world_with_no_boards_offers_no_drawing_to_rate():
  """`rate` judges a drawing on a board (#208): where the world has no
  boards the field, its index line and its bullet go, together -- `schema`
  and the index off one gate, as `test_powers` pins them."""
  bare = Menu.for_world(HUMANOID_HOME)
  boarded = Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME))
  for menu, rated in ((bare, False), (boarded, True)):
    schema = menu.schema(others=("Rowan",))
    assert ("rate" in schema["properties"]) is rated
    assert ("rate" in menu.fields(others=True)) is rated
  assert "- `rate`" not in ov.acts_rule(rate=False) and "Five paperwork" in ov.acts_rule(False)
  assert ov.acts_rule(rate=True) == ov.ACTS_RULE
  for name, text in itertools.chain(_sections(HUMANOID_HOME, others=("Rowan",), acts=True)):
    if name == "WHAT YOU CAN DO ABOUT THE OTHER ROBOT":
      assert "- `rate`" in text, "the humanoid's house has its whiteboards"
