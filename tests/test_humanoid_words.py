"""A humanoid is told its own body and world (issue #559, stages 3 and 5):
the constitutions' body paragraph and every rule in its words -- it charges
standing at its station by a cable it plugs into itself, it has no rack,
no tool and no question jobs -- its reward table its world's rows alone,
and its procedures in its hands' vocabulary as its practice states it.

⚠ THE HUMANOID'S WORDS ARE THE TEXT (stage 5): the quadruped's rules and
constitutions were swapped into them at load until it went, and are gone
with it; what the served pair reads is the rules and the library files as
written, byte for byte. So none of the quadruped's words -- nor the
rover's before it (#376) -- may be in anything a robot reads. Every rule
here is pinned on the prompt as it is built, nothing flown.
"""

import itertools
import json
import re
import textwrap
from dataclasses import replace

import pytest

from pluggybot.lifecycle import HUMANOID_HOME, board_book, overseer_context, world_facts
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
#: the tools on it and what it carried, the drawing and question jobs
#: written on a board (#579: the last of them were in its form, its turns
#: and its context).
QUADRUPED_WORDS = re.compile(
  r"\b(four-legged|small dog|dock|lie down|lying|lie on|rack|bay [A-E]|bays|fork|bumper|"
  r"tools?|a_tool|modules?|module_\w+|stow\w*|bracket|what it is carrying|pen|claw|LCD|draw|"
  r"figures|"
  r"needsAnswer|whole number of at most)\b",
  re.I)
#: ...nor the rover's (#376 deleted it), which the quadruped's rules were
#: written over in #427.
ROVER_WORDS = ("two-wheeled", "wheel", "chassis", "the mast", "set_lift", "lift",
               "hub's charge bay", "your \"hub\"", "LCD face", "drive onto")


def _foreign(text: str) -> str:
  """The first word of another body's world in `text`, with its context,
  or "" where there is none."""
  hit = QUADRUPED_WORDS.search(text)
  if hit is not None:
    return text[max(0, hit.start() - 120):hit.end() + 60]
  return next((w for w in ROVER_WORDS if w in text), "")


def _sections(**flags):
  menu = Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME))
  menu = replace(menu, procedures=flags.pop("procedures", False),
                 wiki=flags.get("wiki", False),
                 tickets=flags.get("tickets", False),
                 look=flags.get("look", False))
  thoughts = ThoughtFiles(None, constitution=c.resolve())
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


def test_every_constitution_is_the_humanoids_in_its_own_words():
  """Every library file opens with the humanoid's paragraph and keeps its
  disposition; it hands over no answer (`test_constitution`'s regexes) and
  names none of the quadruped's world -- its dock, its rack, its tools --
  nor the rover's. Resolved, a name is its file as written."""
  from test_constitution import _ACT, _TACTIC
  for name in c.names():
    text = c.load(name).text
    assert text.startswith("You are a humanoid robot"), name
    assert not _foreign(text), (name, _foreign(text))
    assert not re.search(r"\d", text) and "->" not in text and "%" not in text
    assert not _TACTIC.search(text) and not _ACT.search(text)
    assert c.resolve(name) == c.load(name)


def test_the_served_pair_reads_the_rules_and_its_constitution_as_written(monkeypatch):
  """The deployed menu (#206, Evaluation.md §2): the pair as `serve.py`
  builds it, through `pair.build_pair` -- a mind each, the other named,
  mortal, unseeded, jobs on offer -- and each mind's whole prefix in its own
  words; each reads its constitution's file byte for byte, swapped into
  nothing at load."""
  from pluggybot.pair import build_pair
  for env in ("PLUGGY_LOOK", c.NAME_ENV, c.SECOND_NAME_ENV):
    monkeypatch.delenv(env, raising=False)
  lives = build_pair(HUMANOID_HOME, overseer=True, origin="unseeded", mortal=True,
                     tasks=True, names=("Luca", "Rowan"),
                     constitutions_named=("default", "curious"))
  for life, other in zip(lives, ("Rowan", "Luca")):
    sections = dict(life.overseer.sections)
    assert "THE OTHER ROBOT" in sections and other in sections["THE OTHER ROBOT"]
    text = life.overseer.system[0]["text"]
    assert not _foreign(text), (life.robot_name, _foreign(text))
    charter = life.thoughts.constitution
    assert charter == c.load(charter.name) and charter.text in text
  assert [life.thoughts.constitution.name for life in lives] == ["default", "curious"]


def test_the_probe_measures_the_served_prompt_in_the_humanoids_words(monkeypatch, capsys):
  """`overseer_probe.py --deployed --prompt` (#559's stage 4): the prefix
  the served pair sends, in the humanoid's words, named a body or not --
  the humanoid is the one there is."""
  import importlib.util
  import sys
  from pathlib import Path
  path = Path(__file__).resolve().parents[1] / "scripts" / "overseer_probe.py"
  spec = importlib.util.spec_from_file_location("overseer_probe", path)
  probe = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(probe)
  monkeypatch.delenv("PLUGGY_BODY", raising=False)
  for body in ("humanoid", None):
    monkeypatch.setattr(sys, "argv", ["overseer_probe.py", "--deployed", "--prompt",
                                      "--backend", "local"]
                        + (["--body", body] if body else []))
    probe.main()
    out = capsys.readouterr().out
    assert "You are a humanoid robot" in out and not _foreign(out), (body, _foreign(out))


@pytest.mark.parametrize("flags", list(_configurations()), ids=lambda f: ",".join(f) or "bare")
def test_no_humanoid_prompt_names_another_bodys_world(flags):
  text = "\n\n".join(t for _, t in _sections(**flags))
  assert not _foreign(text), _foreign(text)


def test_what_a_humanoid_is_shown_each_turn_names_no_other_bodys_world():
  """#579: the turns and the context carried the quadruped's leftovers after
  its rules had gone -- an offer's `needsAnswer`, the context's `rack`, the
  other robot's `carrying`, a turn that told the robot to put its tool back
  on its bracket -- and the form its `board`, `program` and `answer`."""
  from test_two_minds import stub_pair
  a, _ = stub_pair(tasks=True)
  a.tasks.offer("open_drawer", "drawer", t=0.0)
  state = overseer_context(a)
  assert state["offeredTasks"] and state["others"]
  shown = json.dumps(state)
  for key in ("needsAnswer", "rack", "carrying"):
    assert f'"{key}"' not in shown, key
  turns = (ov._user_turn(ov.model_state(dict(state))),
           ov._interrupt_turn(ov.model_state(dict(state)), "procedure:open",
                              "your pack is at 28%"))
  for turn in turns:
    assert not _foreign(turn), _foreign(turn)
  assert not {"board", "program", "answer"} & set(a.overseer.menu.schema()["properties"])


#: The quadruped's station in narration's words: its rack and its bays, its
#: dock, its board and its pins, the tools it fetched and stowed.
STATION_WORDS = re.compile(r"\b(rack|dock|bays?|pins|fork|tools?|stow\w*|bracket|modules?|"
                           r"charge bay|lie down|lying down|lying across)\b", re.I)


#: Where narration, History and the mind's text are written: every module
#: whose strings a robot or a watcher reads, the parts catalog's and the hint
#: vocabularies' left out (their racks, modules and pins are real parts).
NARRATING = ("lifecycle.py", "navigator.py", "continuation.py", "pair.py",
             "humanoid/body.py", "humanoid/charge.py", "humanoid/way.py",
             "humanoid/mission.py", "humanoid/places.py", "procedure/steps.py",
             "procedure/lang.py", "mind/overseer.py", "mind/events.py",
             "mind/thoughts.py")


def test_no_narration_or_history_line_names_the_quadrupeds_station():
  """#579: narration -- a frame's `status`, the event stream, the other
  robot's `doing` -- and History called the humanoids' charging station the
  quadruped's: the rack, the charge bay, the dock and its board, its pins.
  Every string those are built from is read (never a docstring); the one
  left is `dock`, a key and a token, never said."""
  import ast
  from pathlib import Path
  src = Path(__file__).resolve().parents[1] / "src" / "pluggybot"
  said = []
  for module in NARRATING:
    tree = ast.parse((src / module).read_text())
    docs = {id(node.body[0].value) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)}
    said += [(module, node.lineno, node.value) for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and id(node) not in docs and node.value != "dock"
             and STATION_WORDS.search(node.value)]
  assert not said, said


def test_a_humanoid_is_shown_only_the_reward_rows_its_world_can_pay():
  """None of the quadruped's jobs is offered in its house: the table is
  the charge's, a procedure's and a ticket's (`overseer.ALWAYS_SHOWN`) and
  its own jobs', the hands' on the storeroom's counter (stage 3)."""
  from pluggybot.economy.cadence import default_cadence
  from pluggybot.economy.tasks import KINDS
  rows = {r["task"] for r in default_table().as_context(
    challenges=True, kinds=Menu.for_world(HUMANOID_HOME).kinds)}
  jobs = {KINDS[k].task for k in default_cadence(HUMANOID_HOME).kinds}
  assert jobs and rows == set(ov.ALWAYS_SHOWN) | jobs
  assert Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME)).boards, \
    "its house has its whiteboards"


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
  assert facts.verb_tests and facts.math and facts.guarded


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
  text = "\n\n".join(t for _, t in _sections(procedures=True))
  assert rule in text, "the humanoid's prompt carries it"


def test_a_humanoids_claim_about_its_station_is_checked():
  """`tell`'s worked claim in a humanoid's words is one code can check."""
  assert acts.check_claim("the charging station is free", charging=False) == (
    "the charging station is free", True)
  assert acts.check_claim("the charge bay is taken", charging=True)[1]
  text = "\n\n".join(t for _, t in _sections(others=("Rowan",), acts=True))
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
  for name, text in itertools.chain(_sections(others=("Rowan",), acts=True)):
    if name == "WHAT YOU CAN DO ABOUT THE OTHER ROBOT":
      assert "- `rate`" in text, "the humanoid's house has its whiteboards"
