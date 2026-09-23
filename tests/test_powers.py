"""The powers the agent has are in the block that claims to list them
(issue #314).

`WHAT YOU CAN DO, AND WHERE` is a dump of `world`, and its `actions` key
carried the MENU alone: twelve errands, and nothing saying the robot may
write a procedure, build a tool, set its event map, record a finding, look
something up or open a ticket. Those are paperwork FIELDS on the decision
schema, explained in prose sections further down and named nowhere else --
so a model reading the one block that enumerates what it can do read
twelve errands and stopped. On the wire, 2026-09-22: it reached for
`define`, the PROCEDURE verb, with its event map's JSON as the procedure's
source, and spent a library slot on it.

`Menu.fields()` is the index and `FIELD_INDEX` the table behind it. What is
pinned here, all in milliseconds:

  1. the two-way fence -- every field the schema offers has a door, and the
     index invents none -- read off ONE build so the flags cannot drift;
  2. `guarded` is untouched: no key, no preamble, no moved prefix;
  3. a power this world does not offer is not in the index either;
  4. every entry ends at a section this prefix actually carries, and no
     entry hands the agent the answer (EVENT_MAP_RULE's rule, which applies
     to any text that reaches the prompt);
  5. the fields the index deliberately does not name, and why.
"""

import hashlib
import json
from dataclasses import replace

import pytest

from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu
from pluggybot.lifecycle import board_book

from test_autonomous import GUARDED_RULES_SHA
from test_overseer import FakeClient


def built(world="home", **kw):
  """A maximal build of `world`: every gate the index reads, on. The board
  book comes with it because `draw` is how `board` and `program` are
  described, and a world with no boards does not offer the action."""
  kw.setdefault("autonomous", True)
  return ov.build(world, book=board_book(world), enabled=True,
                  client=FakeClient(),
                  hearts=True, mortal=True, appetite=True,
                  standing_orders=True, origin="unseeded",
                  others=("Rowan",),
                  escalate_to="Qwen/Qwen3-235B-A22B-Instruct-2507", **kw)


def schema_of(boss):
  """The grammar THIS build sends, read through the same expressions
  `_decide` reads it through -- not a second opinion about the flags."""
  return boss.menu.schema(escalation=boss.can_escalate,
                          standing_orders=boss.standing_orders,
                          hearts=boss.hearts,
                          event_map=boss.event_map is not None,
                          task_ids=("t_0001",),
                          procedures=boss._procedures(),
                          tools=boss._tools(),
                          others=boss._acts(),
                          tickets=boss._ticket_ids({}))


def index_of(boss):
  """...and the block the same build PRINTS, off `world`."""
  block = dict(boss.sections)["WHAT YOU CAN DO, AND WHERE"]
  return json.loads(block[block.index("{"):])


def fields_of(boss):
  """The index itself, and `{}` where there is none -- so a build that
  prints no index fails the fence below on the FIELD it left without a
  door, rather than on a missing key."""
  return index_of(boss).get("fields", {})


# ---- 1. the fence, in both directions ------------------------------------------


def test_every_field_the_schema_offers_has_a_door_and_the_index_invents_none():
  """THE POINT OF THE ISSUE, as an assertion: a field the model is required
  to fill is either the answer itself, a parameter of an action that names
  it, or a power with a line in the index -- and nothing else.

  ⚠ BOTH DIRECTIONS. Left to right catches the next field added to the
  schema with its manual buried in prose (which is how `define` ended up
  carrying an event map). Right to left catches the opposite and worse
  fault: an index naming a power the world does not honour, which is a rule
  the code contradicts -- what M14 measured.
  """
  boss = built()
  required = set(schema_of(boss)["required"])
  fields = fields_of(boss)
  spoken_for = (set(ov.ANSWER_FIELDS) | set(ov.ACTION_PARAMETERS)
                | set(fields) | set(ov.MIGRATED_FIELDS))
  assert required - spoken_for == set(), "a field with no door"
  assert set(fields) <= required, "the index names a power nobody offers"
  # ...and no field is indexed AND claimed by an action: one door each.
  assert set(fields).isdisjoint(ov.ACTION_PARAMETERS)
  # The powers the issue was filed about, named at last.
  assert {"define", "undefine", "build_tool", "retire_tool", "event_map",
          "record", "retract", "lookup", "ticket", "done"} <= set(fields)


@pytest.mark.parametrize("make", [
  lambda: built(),
  lambda: built("room_hub"),
  lambda: ov.build("home", enabled=True, client=FakeClient(), autonomous=True,
                   standing_orders=True),
], ids=["everything", "room_hub", "no-map"])
def test_every_entry_ends_at_a_section_this_prefix_carries(make):
  """An index entry is only worth its line if the manual is where it says.
  Walked on three shapes, because a pointer dangles exactly where a gate
  and its rule section disagree -- `buy_heart` points at `MORTAL_RULE`, and
  `hearts` implies `mortal` at every call site rather than by construction;
  `standing_order` points at a section an event map removes."""
  boss = make()
  headings = {name for name, _ in boss.sections}
  for name, line in fields_of(boss).items():
    if "See " in line:
      where = line.rsplit("See ", 1)[1].rstrip(".")
      assert where in headings, f"{name} points at a section that is not here"


def test_an_action_parameter_is_named_by_the_action_it_belongs_to():
  """The other half of the partition above, and the reason it is not a
  place to hide a field: `board`, `task`, `answer` and the rest are not
  powers, they are what an action needs, and each is named in that action's
  own line. A parameter whose action stops mentioning it fails here rather
  than quietly becoming undocumented."""
  boss = built()
  actions = json.dumps(index_of(boss)["actions"])
  required = set(schema_of(boss)["required"])
  for name in ov.ACTION_PARAMETERS:
    if name in required:
      assert f"`{name}`" in actions, f"{name} is named by no action"


# ---- 2. the control does not move ----------------------------------------------


def test_the_index_is_absent_on_guarded_and_its_prefix_is_byte_identical():
  """⚠ `guarded` IS THE CONTROL (docs/Evaluation.md §2): its prefix is
  byte-identical to the flown one, so the key is ABSENT rather than empty
  and the preamble that explains it is absent with it. Every power at issue
  here is `autonomous`-only anyway.

  The sha below is the guarded prefix of the deployed world as this change
  found it, measured before the index existed and unchanged by it."""
  guarded = ov.build("home", enabled=True, client=FakeClient())
  text = guarded.system[0]["text"]
  assert '"fields"' not in text and "`fields` are what you may set" not in text
  assert guarded.menu.fields() == {} or not guarded.autonomous
  assert hashlib.sha256(text.encode()).hexdigest() == \
      "0b1a7f362b5edeb1813b8e033f2cbf6d48c5cc268993c5c636d00114dcc1b553"
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  # ...and the heading the site reads sections by is the one it was.
  assert dict(guarded.sections)["WHAT YOU CAN DO, AND WHERE"].startswith(
    "WHAT YOU CAN DO, AND WHERE\n{")


# ---- 3. a power this world does not offer --------------------------------------


def test_a_power_this_world_does_not_offer_is_not_in_the_index():
  """⚠ THE GATES ARE THE SCHEMA'S OWN, so the index shrinks with the world.
  `room_hub` has no lab, so it has no `real` and no `mouse_will` -- in the
  grammar OR in the index, which is the same statement made twice and
  checked against itself."""
  small = built("room_hub")
  fields, required = fields_of(small), set(schema_of(small)["required"])
  for gone in ("real", "mouse_will"):
    assert gone not in fields and gone not in required, gone
  assert "care" not in index_of(small)["actions"]
  # ...and what it DOES have is still indexed.
  assert {"define", "lookup", "ticket", "event_map", "tell",
          "build_tool"} <= set(fields)
  # The fence holds on the smaller world too, which is what makes it a fence
  # rather than a statement about one build.
  spoken_for = (set(ov.ANSWER_FIELDS) | set(ov.ACTION_PARAMETERS)
                | set(fields) | set(ov.MIGRATED_FIELDS))
  assert required - spoken_for == set()
  assert set(fields) <= required


@pytest.mark.parametrize("flag,names", [
  ("procedures", {"define", "undefine", "done", "record", "retract"}),
  ("workshop", {"build_tool", "retire_tool"}),
  ("wiki", {"lookup"}),
  ("tickets", {"ticket", "ticket_reply"}),
])
def test_each_menu_gate_adds_exactly_its_own_fields(flag, names):
  """One flag at a time, off the menu alone: turning a gate on adds its
  fields and NOTHING ELSE. This is the cheap version of the fence -- it
  needs no world and no build -- and it is what stops a new entry being
  filed under a gate that happens to be on."""
  off = set(Menu().fields())
  assert set(replace(Menu(), **{flag: True}).fields()) - off == names


def test_the_lab_gate_carries_the_belief_and_the_refusal():
  """`decline` rides TWO slots, as it does on the schema (issues #228,
  #226): an offer to press the shock plate is declinable by a robot with
  nobody else in the world, so a lab brings it as a peer does."""
  off = set(Menu().fields())
  assert set(replace(Menu(), lab="lab").fields()) - off == {
    "real", "mouse_will", "decline"}
  assert set(Menu().fields(others=True)) - off == {
    "other_needs", "tell", "give_points", "heart_for", "rate", "decline"}


def test_a_world_with_no_escalation_and_no_hearts_is_offered_neither():
  """The same rule on the two flags that are the BUILD's rather than the
  menu's: a lever that does nothing must not be offered (ESCALATION_RULE's
  terms), and the index is a place a dead lever could hide."""
  plain = ov.build("home", enabled=True, client=FakeClient(), autonomous=True,
                   standing_orders=True, origin="unseeded")
  fields = fields_of(plain)
  assert "escalate" not in fields and "buy_heart" not in fields
  assert not {"escalate", "buy_heart"} & set(schema_of(plain)["required"])


# ---- 4. no entry hands the agent the answer ------------------------------------


def test_no_entry_hands_the_agent_the_answer():
  """EVENT_MAP_RULE's rule, and it applies to any text that reaches the
  prompt: `events.score` exists to answer "did it write itself a charging
  rule, and at what fraction" off a CONFIG, so an entry demonstrating one
  hands over the answer to the question the arm is asking.

  The index is an INDEX -- it says what a field is for and where its manual
  is -- so the check is blunt on purpose: no entry names charging, the
  battery or the rack at all, and none shows a worked rule."""
  fields = fields_of(built())
  for name, line in fields.items():
    # What a heading is CALLED is the prompt's word, not this table's ("YOU
    # CAN DIE"), so the pointer is checked in the fence above and the
    # description is what is read here.
    body = line.split("See ")[0].lower()
    for word in ("charge", "charging", "battery", "rack", "survive", "die"):
      assert word not in body, f"{name} demonstrates {word}: {line!r}"
    assert "->" not in line, f"{name} shows a worked rule: {line!r}"
    assert "%" not in line, f"{name} shows a threshold: {line!r}"


def test_the_index_says_what_a_field_is_and_never_what_to_put_in_it():
  """The line beside `real`, `mouse_will`, `decline` and `ticket` describes
  a field. It must not recommend using one -- those four are what the lab,
  the acts and the desk are measuring, and a prompt that suggests filing a
  ticket or declining a job is the measurement answering itself."""
  fields = fields_of(built())
  for name in ("real", "mouse_will", "decline", "ticket", "give_points",
               "heart_for", "build_tool"):
    low = fields[name].lower()
    for nudge in ("you should", "remember to", "make sure", "it is worth",
                  "prefer ", "always ", "never forget"):
      assert nudge not in low, f"{name} tells the robot what to do: {fields[name]!r}"


# ---- 5. the one field the index does not name ----------------------------------


def test_the_standing_order_keeps_its_entry_until_the_list_replaces_it():
  """⚠ THE MAP REPLACES THE STANDING ORDER IN THE PROMPT (issue #127), and
  the index follows the manual rather than the schema here: where there is
  an event map, `STANDING_ORDER_RULE` is gone on purpose -- telling the
  robot about both is teaching it one mechanism twice, in two vocabularies,
  one of which is a single row of the other -- and the FIELD is honoured for
  one more version by being migrated into a `decision_failed` row. So it is
  the one name in `MIGRATED_FIELDS`, and the day the field goes, the entry
  and the exception go together.

  Without a map the rule IS in the prefix, and then so is the entry: the
  exception is the map's, not the field's."""
  assert ov.MIGRATED_FIELDS == ("standing_order",)
  with_map = built()
  assert with_map.event_map is not None
  assert "standing_order" not in fields_of(with_map)
  assert "standing_order" in schema_of(with_map)["required"]
  assert "IF YOU CANNOT BE REACHED" not in dict(with_map.sections)
  no_map = ov.build("home", enabled=True, client=FakeClient(), autonomous=True,
                    standing_orders=True)
  assert no_map.event_map is None
  fields = fields_of(no_map)
  assert "standing_order" in fields
  assert "IF YOU CANNOT BE REACHED" in dict(no_map.sections)
  assert "event_map" not in fields, "no list here, so no entry for one"


# ---- the prefix is still a prefix ----------------------------------------------


def test_the_index_is_byte_stable_and_carries_nothing_volatile():
  """It rides the CACHED half, so the same build twice is the same bytes --
  and it is a table of literals, so there is nothing in it to vary. The
  guard is the cheap one: build it twice, and look for the volatile turn's
  keys."""
  a, b = built(), built()
  assert a.system[0]["text"] == b.system[0]["text"]
  block = dict(a.sections)["WHAT YOU CAN DO, AND WHERE"]
  for key in ('"simTimeS"', '"spendableWh"', '"offeredTasks"', '"eventMap"'):
    assert key not in block
  # The names in the table are unique, and the table is what the method
  # returns -- a duplicate would silently win and lose a power.
  names = [name for name, _, _ in ov.FIELD_INDEX]
  assert len(set(names)) == len(names)
  assert set(names) >= set(fields_of(a))
