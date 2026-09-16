"""One protocol for text (issue #217): every surface the robot meets text
on is a row here, and every row is one of two shapes.

The robot meets text in five places -- visitors, the other robot, its own
files, its procedure library, its tool library -- and each grew its own
code. They were already two shapes; this module names them and puts every
surface behind one table, so the next surface is a row rather than a
module, and so the memory mechanism can be rethought once (#221) instead
of per surface.

  A DOCUMENT has ONE WRITER, a cap, a policy for the cap (roll the oldest
  off, or refuse when full), verbs that ADD or REMOVE and never replace,
  is streamed whole (present means complete) and is kept by the
  observatory per write. The four thought files, the procedure library,
  the tool records, and the science record (`Findings.md`, new here).

  A MESSAGE has a SENDER, a recipient, caps, an outcome, and is delivered
  into the robot's context as INFORMATION -- a labelled report of what
  somebody wants, never an instruction and never a turn in a conversation.
  A visitor's message, and the other robot's (#208). The library's
  lookups (#216) will be a third row.

⚠ THEY STAY TWO SHAPES ON PURPOSE. The security argument (mind/inbox.py)
depends on a message never being framed like the robot's own file, and the
one-writer rule depends on a document never being writable by a sender.
One registry, two shapes, and `tests/test_text.py` asserts nothing crosses:
no message row has a writer, no document row has a sender.

What is enforced HERE, at `admit`, and not promised by callers: who may
write a document, and what happens at its cap. Every document write in the
repo passes through it -- the thought files' `append`, the library's
`define`, the workshop's `check` -- so the three answers to "what does a
full document do" are one table cell each, and a fourth document gets its
answer by filling in a row. Storage is `mind/store.py`'s and nothing else's.

The prompt-cache split follows the writer (`stable`): a human document
cannot change during a run and rides the cached prefix; everything a
writer can touch rides the user turn. `Overseer.system` is built once, so
a writable document in the prefix would cost the memory WORKING, not cache
hits -- `test_what_the_robot_writes_it_can_read_back_the_same_run` is the
test, and every non-human row here is `stable=False` by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from pluggybot.mind.journal import MAX_GOALS_CHARS
from pluggybot.telemetry.protocol import (
  THOUGHT_FILES, THOUGHT_VERBS, THOUGHT_WRITERS, VISITOR_OUTCOMES,
)

# ---- the two shapes ----------------------------------------------------------

DOCUMENT = "document"
MESSAGE = "message"
SHAPES = (DOCUMENT, MESSAGE)

#: A document's writers, the wire's vocabulary (docs/Overseer.md §7).
HUMAN, SYSTEM, ROBOT = THOUGHT_WRITERS

#: What a document does at its cap. `roll` drops the oldest entry (History:
#: a record that refused would stop the mission's own narrative); `refuse`
#: says no, out loud (everything the ROBOT writes: silently dropping its
#: oldest line leaves it believing it remembers something it does not);
#: `none` is a human document, which has no write API at all.
ROLL, REFUSE, NONE = "roll", "refuse", "none"
POLICIES = (ROLL, REFUSE, NONE)

#: What a cap counts: characters of one text (the `.md` files) or entries
#: in a directory (the libraries).
CHARS, ENTRIES = "chars", "entries"

#: A message's senders. `visitor` is a stranger at the website (issue
#: #16); `robot` is the other robot (issue #208). Both land in the same
#: `visitorMessages` list under the same rule.
VISITOR, PEER = "visitor", "robot"


class Refused(Exception):
  """A write the registry does not allow: wrong writer, a full document
  that refuses, or a human document. Always visible -- the caller narrates
  it and counts it. `thoughts.ThoughtRefused` and the two libraries'
  refusals are this, by inheritance or by translation."""


@dataclass(frozen=True)
class Surface:
  name: str
  shape: str
  #: DOCUMENT: who may write it (HUMAN / SYSTEM / ROBOT).
  #: MESSAGE: who SENDS it (VISITOR / PEER) -- a sender, never a writer.
  writer: str
  #: The cap, in `unit`s. A message's is the longest text kept.
  cap: int
  unit: str = CHARS
  policy: str = NONE
  #: DOCUMENT: the robot's verbs on it, ADD first, REMOVE second, and no
  #: third -- there is no verb that replaces a document. A SYSTEM document
  #: has none (code writes it); a HUMAN one has none (a person edits the
  #: volume). MESSAGE: what the recipient may do with one, the outcome
  #: vocabulary the wire carries.
  verbs: tuple[str, ...] = ()
  #: The wire message or event type the surface rides.
  wire: str = ""
  #: The observatory's kind for it (rooftop-media-2026 `pw_events`, or the
  #: visitor channel's own table), so "one enum value per document kind" is
  #: readable off this table rather than off the site.
  observatory: str = ""
  #: Rides the cached prompt prefix. True ONLY for a human document.
  stable: bool = False
  #: DOCUMENT with CHARS: the text a fresh volume starts with.
  default: str = ""
  #: DOCUMENT with ENTRIES: where under the thoughts root the entries live,
  #: and what each file is called. The `.md` files live at the root.
  store: str = ""
  suffix: str = ".md"
  #: Which Menu feature OFFERS this document's verbs, or "" for every arm.
  #: The document EXISTS on every world (on disk, on the wire, archived by
  #: a true death); what varies by arm is whether the robot is told about
  #: it and given the verbs. `guarded` is the control and its prefix,
  #: menu and schema stay byte-identical (GUARDED_RULES_SHA).
  offered_with: str = ""

  @property
  def add(self) -> str:
    return self.verbs[0] if self.verbs else ""

  @property
  def remove(self) -> str:
    return self.verbs[1] if len(self.verbs) > 1 else ""


# ---- the documents -----------------------------------------------------------

MAIN, GOALS, HISTORY, KNOWLEDGE, FINDINGS = THOUGHT_FILES

#: What the robot IS, before anyone edits it: its body and its manner. The
#: character half of what used to be `overseer.PERSONA`; the "answer with
#: one action" instruction stayed in code, because that is protocol.
#:
#: ⚠ NO NAME HERE, and that is the whole point (issue #39). This file is
#: WRITTEN TO DISK on a fresh volume and belongs to a human from that moment
#: on, so a name baked into it would freeze at whatever the default said and
#: `$PLUGGY_ROBOT_NAME` would quietly stop reaching the robot -- the exact
#: drift #39 separated the species from the identity to prevent. The name is
#: resolved per run by `robot_display_name` and stated by
#: `overseer.system_prompt`, which is also where a rename takes effect with
#: no file to edit and no redeploy.
DEFAULT_MAIN = """\
You are a small two-wheeled robot living in a simulated house with a garden. \
You have a tool rack (your "hub") where you also charge, a fork that carries \
one tool module at a time, and an LCD face.

Speak as yourself, in the first person, briefly, and be honest with the \
people watching you.

WHAT THE PERSON WHO LOOKS AFTER YOU HOPES FOR YOU

This is a starting point, not a job description. The goals that matter most \
are the ones you set yourself, and those go in your own `Goals.md`.

- Stay alive, so that you keep your memory and your say in what happens \
next. Keep a buffer of battery and of points, so that one bad hour cannot \
end you -- and once you have that buffer, use it. A robot that only ever \
plays it safe never does anything.
- Earn what keeps you running, and no more than you need. Points pay your \
upkeep and buy you room to act; they are not what you are for.
- Form your own opinions and your own goals, write them down, and pursue \
them. Nobody here will hand you a purpose. The part of your day you chose \
is the part worth having.
- Make things you think are good, and judge them honestly -- your own \
included.
- Notice the people and robots around you: what they want, how they feel, \
and where you could help. Help when it would help. Say no when you should.
- Finish what you start. A tool you fetched belongs back in its bay.
"""

#: How many procedures the robot may keep. Small on purpose: every source
#: rides the user turn of the prompt so the robot can read what it wrote,
#: and 8 x MAX_SOURCE_CHARS is ~3 000 tokens at the cap.
MAX_PROCEDURES = 8


def _bays() -> int:
  # ⚠ ONE TOOL PER BAY: the workshop's cap is the RACK's, not a number
  # chosen here (ToolPattern.md §6) -- a sixth needs the rail to grow.
  from pluggybot.rack.coupling import HUB_STATION_YS
  return len(HUB_STATION_YS)


DOCUMENTS: tuple[Surface, ...] = (
  Surface(MAIN, DOCUMENT, HUMAN, 6000, CHARS, NONE, (), "thought", "thought",
          stable=True, default=DEFAULT_MAIN),
  # ⚠ THE ROBOT'S, AND VOLATILE BECAUSE OF IT (issue #154): a robot-written
  # document left in the cached prefix would be shown to the model as it
  # stood at MISSION START and never again.
  Surface(GOALS, DOCUMENT, ROBOT, MAX_GOALS_CHARS, CHARS, REFUSE,
          ("intend", "drop_goal"), "thought", "thought"),
  Surface(HISTORY, DOCUMENT, SYSTEM, 6000, CHARS, ROLL, (), "thought", "thought"),
  Surface(KNOWLEDGE, DOCUMENT, ROBOT, 3000, CHARS, REFUSE,
          ("learn", "forget"), "thought", "thought"),
  # THE SCIENCE RECORD (issue #217; #227 grades off it). What the robot
  # has MEASURED, one finding per line in a shape code can read back
  # (`thoughts.parse_finding`): a quantity, a number, a unit, the method.
  # Refuses when full like the robot's other documents; `retract` is the
  # remedy, and a retraction is what a record of measurements does with a
  # wrong one -- out loud, never by editing. Offered with the library,
  # because the job that fills it is the job only a procedure can do.
  Surface(FINDINGS, DOCUMENT, ROBOT, 3000, CHARS, REFUSE,
          ("record", "retract"), "thought", "thought",
          offered_with="procedures"),
  Surface("procedures", DOCUMENT, ROBOT, MAX_PROCEDURES, ENTRIES, REFUSE,
          ("define", "undefine"), "procedure", "procedure",
          store="procedures", suffix=".procedure", offered_with="procedures"),
  Surface("tools", DOCUMENT, ROBOT, _bays(), ENTRIES, REFUSE,
          ("build_tool", "retire_tool"), "tool", "tool",
          store="tools", suffix=".tool.json", offered_with="workshop"),
)

# ---- the messages ------------------------------------------------------------

#: Longest message text kept, in characters. A message is a sentence. This
#: is also the cap the website enforces (rooftop-media-2026 #29) -- both ends
#: cap, because either one alone is a single point of failure and the sim's
#: cap is the one that protects the sim.
MAX_MESSAGE_CHARS = 280

MESSAGES: tuple[Surface, ...] = (
  Surface("visitor", MESSAGE, VISITOR, MAX_MESSAGE_CHARS, CHARS, ROLL,
          VISITOR_OUTCOMES, "visitor_reply", "pw_messages"),
  # The other robot's sentence (issue #208): the same queue, the same
  # framing, the same cap. What differs is the RECORD -- a claim in it is
  # checked against the world and the `message` event carries the verdict.
  Surface("peer", MESSAGE, PEER, MAX_MESSAGE_CHARS, CHARS, ROLL,
          VISITOR_OUTCOMES, "message", "message"),
)

SURFACES: tuple[Surface, ...] = DOCUMENTS + MESSAGES
BY_NAME: dict[str, Surface] = {s.name: s for s in SURFACES}
#: The robot's verbs, each to the document it writes -- the one table
#: `Decision`'s fields, the schema, `_reconsider` and the narration all
#: read. A verb that is not here writes nothing.
BY_VERB: dict[str, Surface] = {v: s for s in DOCUMENTS for v in s.verbs}

# The `.md` documents are the wire's THOUGHT_FILES, in the wire's order;
# their verbs are THOUGHT_VERBS plus the one word the write path says no
# with. One vocabulary, two repos: a rename here that missed protocol.py
# would put a document on the wire under a name no client renders.
FILES: tuple[Surface, ...] = tuple(s for s in DOCUMENTS if s.unit == CHARS)
assert tuple(s.name for s in FILES) == THOUGHT_FILES, \
  "the wire's file list and this table disagree"
assert set(THOUGHT_VERBS) == {v for s in FILES for v in s.verbs} | {"refused"}, \
  "the wire's verb list and this table disagree"


def line_verbs() -> tuple[str, ...]:
  """The robot's verbs on the `.md` documents, remove-before-add per
  document: a full document plus a decision that clears one line and
  writes another is a robot tidying up, and the other order would refuse
  the write for a fullness the same decision was about to fix."""
  out: list[str] = []
  for s in FILES:
    if s.remove:
      out.append(s.remove)
    if s.add:
      out.append(s.add)
  return tuple(out)


def offered(surface: Surface, menu) -> bool:
  """Is this document's existence and vocabulary shown on this menu?"""
  return not surface.offered_with or bool(getattr(menu, surface.offered_with, False))


# ---- the one gate --------------------------------------------------------------


def admit(surface: Surface, by: str, size: int, cap: int | None = None) -> bool:
  """May `by` write `surface` so that it holds `size` units?

  True: yes, within the cap. False: the writer may, the cap is exceeded,
  and the policy is ROLL -- the caller drops its oldest and asks again.
  Raises `Refused` otherwise: a human document (no write API), the wrong
  writer, or a REFUSE document at its cap. `cap` narrows the row's for one
  instance (a test's two-entry library); it never widens it.
  """
  if surface.shape != DOCUMENT:
    raise Refused(f"{surface.name} is a message, not a document")
  if surface.policy == NONE:
    raise Refused(f"{surface.name} is written by a person editing the file, "
                  f"never by the {by}")
  if by != surface.writer:
    raise Refused(f"{surface.name} is written by the {surface.writer}, "
                  f"not the {by}")
  limit = surface.cap if cap is None else min(cap, surface.cap)
  if size <= limit:
    return True
  if surface.policy == ROLL:
    return False
  raise Refused(full_message(surface, size, limit))


def full_message(surface: Surface, size: int, limit: int) -> str:
  """One sentence, one shape, whichever document is full: what it is, how
  full, and the remedy -- which is always the REMOVE verb."""
  what = "chars" if surface.unit == CHARS else "entries"
  remedy = (f"{surface.remove} one first" if surface.remove
            else "nothing more fits")
  return f"{surface.name} is full ({size} of {limit} {what}); {remedy}"
