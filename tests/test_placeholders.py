"""A placeholder is empty, and an answer full of them acts on nothing it
cannot show (issue #462; docs/Overseer.md "Placeholders").

Every field of the decision schema is required, so a field the model means
nothing by is still written -- and the deployed model sometimes wrote `n`,
`none` or `:` where `""` was meant, and `validate` took it as meant: a
message of "n", a decline because "none", a goal called `intend`, and three
hearts bought for the other robot. The answers below are the served pair's,
as the observatory recorded what they did.
"""

import dataclasses

import pytest

from pluggybot.economy.ledger import HEARTS
from pluggybot.lifecycle import board_book
from pluggybot.mind import overseer as ov

WORLD = "home_quad"


def _menu():
  """A mind's menu on legs with every power the served pair has."""
  return dataclasses.replace(ov.Menu.for_world(WORLD, board_book(WORLD)),
                             procedures=True, wiki=True, tickets=True,
                             lab="lab", look=True)


def _validate(raw, **kw):
  kw = {"offered": ("t_0758", "t_0759"), "procedures": ("workshop_hunt",),
        "others": ("Rowan",), "tickets": ("tk_0024",), "event_map": True,
        "standing_orders": True, **kw}
  return _menu().validate(raw, **kw)


def _quiet(**fields) -> dict:
  """An answer that says nothing but its action, as the schema has it:
  every field present, `""` / `false` / an empty object where unused."""
  raw = {"think": "", "action": "idle", "reason": "", "board": "",
         "program": "", "zone": "", "read": "", "find": "", "respond_to": "",
         "outcome": "", "reply": "", "task": "", "answer": "", "pin": "",
         "unpin": "", "note": {"topic": "", "title": "", "text": ""},
         "unnote": "", "cites": "", "intend": "", "drop_goal": "", "serves": "",
         "standing_order": "", "buy_heart": False, "event_map": [],
         "define": {"name": "", "source": ""}, "undefine": "", "done": "",
         "record": {"quantity": "", "value": 0, "unit": "", "method": "",
                    "topic": ""},
         "retract": "", "other_needs": "", "tell": {"to": "", "text": ""},
         "give_points": {"to": "", "amount": 0}, "heart_for": "",
         "rate": {"board": "", "quality": 0}, "decline": {"task": "", "reason": ""},
         "care": "", "real": "", "mouse_will": "", "lookup": "",
         "ticket": {"kind": "", "title": "", "text": ""},
         "ticket_reply": {"ticket": "", "text": ""}}
  return {**raw, **fields}


#: Luca, t=391502.947 (2026-10-04, build 5ba8a0c): a procedure it meant to
#: run, and "n" everywhere else -- it bought Rowan a heart for 200 points.
LUCA_391503 = _quiet(
  think="Rowan's message is just a friendly confirmation of what I knew.",
  action="procedure:workshop_hunt",
  reason="Rowan's message just confirms the game I already logged.",
  respond_to="n", reply="n", pin="n", unpin="n", unnote="n", cites="n",
  intend="n", drop_goal="n", serves="n", retract="n", lookup="n",
  note={"topic": "n", "title": "n", "text": "n"},
  define={"name": "n", "source": "n"}, done="t_0758",
  record={"quantity": "n", "value": 0, "unit": "n", "method": "n", "topic": "n"},
  buy_heart=True, heart_for="Rowan", other_needs="nothing",
  tell={"to": "Rowan", "text": "n"}, give_points={"to": "Rowan", "amount": 0},
  rate={"board": "whiteboard_a", "quality": 0.5},
  decline={"task": "t_0759", "reason": "n"},
  ticket={"kind": "bug", "title": "n", "text": "n"},
  ticket_reply={"ticket": "tk_0024", "text": "n"})


def test_a_placeholder_is_punctuation_a_letter_a_null_word_or_the_fields_own_name():
  for value in ("n", "X", ":", "},", "  n  ", 'a"}', "none", "None.", ": null",
                "n/a", "N/A", "nil", "placeholder"):
    assert ov.placeholder(value), value
  assert ov.placeholder("intend", ("intend",))
  assert ov.placeholder("drop goal", ("drop_goal",))
  assert ov.placeholder("Text", ("tell", "text"))
  # ⚠ a real short value stands: two letters are a word, a number is a
  # number, and one character outside ASCII may be a whole word
  for value in ("", "ok", "hi", "no", "Na", "keep", "8", "5", "👍", "好",
                "bay C is empty"):
    assert not ov.placeholder(value), value
  assert not ov.placeholder("intend", ("pin",)), "another field's name is a word"


def test_an_answer_full_of_placeholders_acts_on_nothing_and_its_action_stands():
  d = _validate(LUCA_391503)
  assert d.action == "procedure:workshop_hunt", "the action stands"
  assert not d.buy_heart and d.heart_for == "", "it bought Rowan a heart"
  assert d.tell is None and d.decline is None and d.lookup == ""
  assert d.ticket is None and d.ticket_reply is None and d.reply == ""
  assert d.rate is None and d.other_needs == "" and d.give_points is None
  assert d.done == "", "an id copied off the board shows nothing"
  for write in ("pin", "unpin", "unnote", "intend", "drop_goal", "retract", "serves"):
    assert getattr(d, write) == "", write
  assert d.note is None and d.record is None and d.define is None
  assert d.left_out["why"] == "filled" and d.left_out["words"] == ["n"]
  assert d.left_out["acts"] == ["buy_heart", "heart_for", "rate", "other_needs", "done"]
  assert len(d.left_out["fields"]) == 15 and "tell" in d.left_out["fields"]
  assert d.as_dict()["leftOut"] == d.left_out, "the decision's row carries it"


def test_one_placeholder_leaves_out_its_own_field_and_nothing_else():
  raw = _quiet(reason="the feed job is mine", action="take_task", task="t_0759",
               tell={"to": "Rowan", "text": "n"},
               pin="Feed presses register again since the restart.",
               buy_heart=True, heart_for="Rowan",
               rate={"board": "whiteboard_a", "quality": 0.7})
  d = _validate(raw)
  assert d.tell is None, "a message of 'n' went to the other robot"
  assert d.pin == raw["pin"] and d.buy_heart and d.heart_for == "Rowan"
  assert d.rate == {"board": "whiteboard_a", "quality": 0.7}
  assert d.left_out == {"why": "placeholder", "fields": ["tell"], "words": ["n"]}


def test_a_real_short_value_stands():
  for said in ("ok", "hi", "no"):
    d = _validate(_quiet(tell={"to": "Rowan", "text": said}))
    assert d.tell == {"to": "Rowan", "text": said} and d.left_out is None
  # ...the answer to a question job, a digit
  d = _validate(_quiet(action="take_task", task="t_0759", answer="8"),
                answering=("t_0759",))
  assert d.answer == "8" and d.left_out is None
  # ...a finding's one-letter unit, and record ids
  finding = {"quantity": "cube_mass", "value": 0.42, "unit": "g",
             "method": "read the claw's load at rest", "topic": "mass_bench"}
  d = _validate(_quiet(record=finding, cites="7"))
  assert d.record["unit"] == "g" and d.cites == "7" and d.left_out is None
  # ...a note the robot titled after a bay, a goal by its number, a smile
  d = _validate(_quiet(note={"topic": "bays", "title": "C", "text": "holds the claw"},
                       serves="1", tell={"to": "Rowan", "text": "👍"}))
  assert d.note["title"] == "C" and d.serves == "1" and d.tell["text"] == "👍"
  assert d.left_out is None
  # ...a sum answered to a visitor (a reply rides an id only a waiting
  # message has, and is never judged)
  d = _validate(_quiet(respond_to="m_7", outcome="replied", reply="5"),
                waiting=("m_7",))
  assert (d.respond_to, d.reply, d.left_out) == ("m_7", "5", None)
  # ...a finding whose optional method said nothing: graded all the same
  weighed = {"quantity": "unknown mass", "value": 0.31, "unit": "kg",
             "method": "none", "topic": "mass_bench"}
  d = _validate(_quiet(record=weighed))
  assert d.record == {**weighed, "method": ""}
  assert d.left_out == {"why": "placeholder", "fields": ["record"], "words": ["none"]}
  # ...and a ticket titled with a placeholder is the desk's to title
  d = _validate(_quiet(ticket={"kind": "bug", "title": "x",
                               "text": "The feed plate never registers."}))
  assert d.ticket == {"kind": "bug", "title": "",
                      "text": "The feed plate never registers."}
  assert d.left_out["fields"] == ["ticket"]


def test_one_word_in_three_fields_is_that_answers_placeholder():
  """Rowan, t=240869.76 (build a85772d): `keep` as a goal, a goal to drop and
  a finding to retract, beside real writing. None of the three is a
  placeholder alone; together they are this answer's."""
  raw = _quiet(action="take_task", task="t_0759", reason="the feed job is back",
               drop_goal="keep", intend="keep", retract="keep",
               pin="Feed job taken on the fifth restart cycle.",
               tell={"to": "Rowan", "text": "Feed job is up and I've taken it."},
               other_needs="unknown", give_points={"to": "Rowan", "amount": 1})
  d = _validate(raw)
  assert d.intend == d.drop_goal == d.retract == ""
  assert d.pin == raw["pin"] and d.tell["text"] == raw["tell"]["text"], \
    "free text that is no placeholder stands"
  assert d.other_needs == "" and d.give_points is None
  assert d.left_out == {"why": "filled", "words": ["keep"],
                        "fields": ["intend", "drop_goal", "retract"],
                        "acts": ["give_points", "other_needs"]}


def test_a_quote_names_a_line_and_a_placeholder_quote_only_itself():
  """Outside an answer full of them a placeholder QUOTE stands: Luca's `n`
  goal, written before #462, can be picked out by nothing else. It takes out
  only a line that is exactly it -- as a substring, Rowan's `,` took out its
  goal and this pin (t=308982.446, build 8a61ada)."""
  from pluggybot.mind.thoughts import TOP_OF_MIND, ThoughtFiles, ThoughtRefused
  memory = ThoughtFiles(None)
  memory.pin("The task id, not the kind, is what take_task reads")
  with pytest.raises(ThoughtRefused, match="nothing on the page matches"):
    memory.unpin(",")
  memory.note({"topic": "tower", "title": "Block hunt plan",
               "text": "workshop mapped; next, locate the blocks"})
  with pytest.raises(ThoughtRefused):
    memory.unnote("n")
  memory.pin("n")                                   # written before #462
  assert memory.unpin("n") == "n"
  assert memory.read(TOP_OF_MIND).strip() == "The task id, not the kind, is what take_task reads"
  # ...and two letters are a word, too short to be part of a line
  memory.intend("Feed the mouse at noon")
  with pytest.raises(ThoughtRefused):
    memory.drop_goal("no")
  d = _validate(_quiet(drop_goal="n"))
  assert d.drop_goal == "n" and d.left_out is None


def test_a_word_filing_a_robots_records_together_is_no_placeholder():
  """One word in three fields is an answer's filler, but not in a topic: a
  lookup, a note and a finding filed under `mouse` are a robot keeping its
  records together (found in review)."""
  raw = _quiet(lookup="Mouse",
               note={"topic": "mouse", "title": "after a feed", "text": "it rests"},
               record={"quantity": "rest after feed", "value": 196, "unit": "s",
                       "method": "watched from the lab", "topic": "mouse"},
               give_points={"to": "Rowan", "amount": 10}, buy_heart=True,
               heart_for="Rowan")
  d = _validate(raw)
  assert d.left_out is None and d.lookup == "Mouse" and d.note is not None
  assert d.give_points == {"to": "Rowan", "amount": 10} and d.buy_heart


def test_an_undefine_beside_a_define_left_out_waits():
  """Luca's answer, with the procedure it runs in `undefine` -- the decoder
  fills an enum with the job in hand, as it did `done`. With the define left
  out, the undefine alone took out `workshop_hunt`, and the same answer's
  action had nothing to run (found in review). It waits, as it does beside a
  define the library refuses (#264)."""
  d = _validate({**LUCA_391503, "undefine": "workshop_hunt"})
  assert d.undefine == "" and d.left_out["undefine"] == "workshop_hunt"
  assert d.action == "procedure:workshop_hunt"
  assert ("the undefine of workshop_hunt, which waits for a define that goes "
          "through") in ov.left_out_said(d.left_out)
  # ...and beside a define that stands it is a replacement, as ever
  source = "def workshop_hunt():\n    wait(1)\n"
  d = _validate(_quiet(undefine="workshop_hunt",
                       define={"name": "workshop_hunt", "source": source}))
  assert d.undefine == "workshop_hunt" and d.define["source"] == source


def test_procedure_new_beside_a_placeholder_define_runs_nothing_and_the_rest_stands():
  """A `procedure:new` whose define the library refuses runs nothing and the
  answer stands; one whose define was a placeholder must not cost the whole
  answer -- its rules, its message -- as a garbled one (found in review)."""
  from test_body import stub_life
  rule = {"event": "every", "action": "ask", "value": 1500, "kind": ""}
  raw = _quiet(action=ov.PROCEDURE_NEW, reason="r", event_map=[rule],
               define={"name": "n", "source": "n"},
               tell={"to": "Rowan", "text": "Back in a minute."})
  d = _validate(raw)
  assert d.action == ov.PROCEDURE_NEW and d.define is None
  assert [r.action for r in d.event_map] == ["ask"] and d.tell is not None
  assert d.left_out == {"why": "placeholder", "fields": ["define"], "words": ["n"]}
  life = stub_life()
  life._after_decision(d)
  assert life.thoughts.read("History.md").rstrip().endswith(
    "runs the procedure the same answer defines, and that define was left out")


def test_a_decline_of_the_job_the_same_answer_takes_is_left_out():
  raw = _quiet(action="take_task", task="t_0759", reason="mine",
               decline={"task": "t_0759", "reason": "not declining -- taking it"})
  d = _validate(raw)
  assert d.task == "t_0759" and d.decline is None
  assert d.left_out == {"why": "decline", "decline": "t_0759"}
  # ...and one of another offer is a decline
  d = _validate({**raw, "decline": {"task": "t_0758", "reason": "too far"}})
  assert d.decline == {"task": "t_0758", "reason": "too far"} and d.left_out is None


def test_a_filled_answer_keeps_what_it_configures_and_the_procedure_it_defines():
  """Luca, t=247102.424 (build a85772d): back from a death it was not asked
  for, its answer wrote a rule asking it every 1500 s -- and `test` in its
  record, decline and ticket, beside a heart for Rowan."""
  rule = {"event": "every", "action": "ask", "value": 1500, "kind": ""}
  source = "def wait_pen():\n    wait(1)\n"
  raw = _quiet(action=ov.PROCEDURE_NEW, reason="a heartbeat rule first",
               event_map=[rule], standing_order="idle",
               define={"name": "wait_pen", "source": source},
               record={"quantity": "test", "value": 1, "unit": "n", "method": "test",
                       "topic": ""},
               decline={"task": "t_0759", "reason": "test"},
               ticket={"kind": "bug", "title": "test", "text": ""},
               buy_heart=True, heart_for="Rowan")
  d = _validate(raw)
  assert not d.buy_heart and d.heart_for == ""
  assert d.record is None and d.decline is None
  assert d.left_out["why"] == "filled" and d.left_out["words"] == ["test"]
  assert d.action == ov.PROCEDURE_NEW and d.define["source"] == source
  assert [r.action for r in d.event_map] == ["ask"] and d.standing_order == "idle"


def test_every_power_is_judged_left_out_or_kept():
  powers = {name for name, *_ in ov.FIELD_INDEX} | set(ov.MIGRATED_FIELDS)
  judged = set(ov.PLACEHOLDER_TEXT) | set(ov.PLACEHOLDER_OBJECTS)
  assert powers == judged | set(ov.UNSHOWN_PAPERWORK) | set(ov.PLACEHOLDER_KEPT)
  assert not set(ov.PLACEHOLDER_KEPT) & (judged | set(ov.UNSHOWN_PAPERWORK))
  assert not set(ov.ANSWER_FIELDS + ov.ACTION_PARAMETERS) & judged, "the action stands"
  # ...and what a filled answer leaves out goes back to the field's default
  assert set(ov.UNSHOWN_PAPERWORK) <= {f.name for f in dataclasses.fields(ov.Decision)}
  # ...and every sub-field judged is one the schema has
  props = _menu().schema(standing_orders=True, hearts=True, event_map=True,
                         procedures=("workshop_hunt",), tools=(),
                         others=("Rowan",), tickets=("tk_0024",))["properties"]
  for name, (content, labels) in ov.PLACEHOLDER_OBJECTS.items():
    assert set(content + labels) <= set(props[name]["properties"]), name


def test_what_was_left_out_is_one_history_line_an_event_and_the_decisions_row():
  from pluggybot.mind.inbox import Inbox
  from test_two_minds import stub_pair
  a, b = stub_pair(inboxes=(Inbox(), Inbox()))
  a.ledger.intervene(500, by="test", t=0.0)
  b.ledger.lose_heart()
  events = []
  a.on_event.append(events.append)
  raw = _quiet(action="idle", reason="resting", intend="none", pin="none",
               tell={"to": "Rowan", "text": "none"}, buy_heart=True,
               heart_for="Rowan", rate={"board": "whiteboard_a", "quality": 0.5})
  a._after_decision(a.overseer.menu.validate(raw, others=("Rowan",)))
  assert b.ledger.hearts() == HEARTS - 1 and a.ledger.balance() == 500, \
    "a heart was bought for the other robot"
  assert len(b.inbox) == 0 and "none" not in a.thoughts.read("Goals.md")
  [gone] = [e for e in events if e["type"] == "left_out"]
  assert gone["why"] == "filled" and gone["acts"] == ["buy_heart", "heart_for", "rate"]
  history = [ln for ln in a.thoughts.read("History.md").splitlines() if ln.strip()]
  [line] = [ln for ln in history if "left out of that answer" in ln]
  assert ("3 fields holding only 'none' read as empty -- pin, intend, tell; and "
          "with 3 or more, not acted on either: buy_heart, heart_for, rate") in line
  assert any("  LEFT OUT 3 fields holding only 'none'" in ln for ln in a.log)
  assert a.decisions[-1]["leftOut"] == {k: v for k, v in gone.items()
                                        if k not in ("type", "t", "robot")}


def test_the_line_says_how_many_and_which():
  assert ov.left_out_said({"why": "placeholder", "fields": ["tell"],
                           "words": ["n"]}) == \
    "1 field holding only 'n' read as empty -- tell"
  assert ov.left_out_said({"why": "decline", "decline": "t_0305"}) == \
    "the decline of t_0305, the job it took"
