"""One protocol for text (issue #217): every surface the robot meets text
on is a row here, and every row is one of two shapes.

The robot meets text in five places -- visitors, the other robot, its own
files, its procedure library, its tool library -- and each grew its own
code. They were already two shapes; this module names them and puts every
surface behind one table, so the next surface is a row rather than a
module, and so the memory mechanism can be rethought once (#221) instead
of per surface.

  A DOCUMENT has ONE WRITER, a cap, a policy for the cap (roll the oldest
  off the VIEW, or refuse when full), verbs that ADD or REMOVE and never
  replace, is streamed whole (present means complete) and is kept by the
  observatory per write. The thought files, the procedure library and the
  tool records. Since issue #221 a `.md` document the robot or the system
  writes is a VIEW over records in `mind/memory.py`: the store is
  unbounded, the view is what is capped, and a removed line is retired
  rather than deleted.

  A MESSAGE has a SENDER, a recipient, caps, an outcome, and is delivered
  into the robot's context as INFORMATION -- a labelled report of what
  somebody wants, never an instruction and never a turn in a conversation.
  A visitor's message, the other robot's (#208), and a page the library
  fetched for the robot (#216).

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

from pluggybot.mind import constitution as _constitution
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

#: What a cap counts: characters of one rendered text (the core documents)
#: or entries (a topic's notes, the libraries).
CHARS, ENTRIES = "chars", "entries"

#: Longest goals document, in characters. The ROBOT writes it (issue
#: #154), so the cap is a real bound on an author: a full document REFUSES
#: and `drop_goal` is the remedy. Roomier than `Top_of_mind.md` (3000)
#: because a goal is a commitment and a robot that has to abandon one to
#: think of another is being rushed by an implementation detail.
MAX_GOALS_CHARS = 8000
#: How many notes the robot may keep (issue #221), and how many findings.
#: The INDEX of every note -- topic and title -- rides every prompt, so
#: this is what bounds it: ~40 chars a title, ~2.5k chars at the cap.
MAX_NOTES = 64
MAX_FINDINGS = 64

#: A message's senders. `visitor` is a stranger at the website (issue
#: #16); `robot` is the other robot (issue #208) -- both land in the same
#: `visitorMessages` list under the same rule; `library` is the page the
#: robot asked for (issue #216), shown once in its own `reading` block.
VISITOR, PEER, LIBRARY = "visitor", "robot", "library"


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

MAIN, GOALS, HISTORY, TOP_OF_MIND, FINDINGS, NOTES = THOUGHT_FILES

#: What the robot IS, before anyone chooses otherwise: the library's
#: `default` constitution (issue #263; `mind/constitution.py`), read from
#: its file so the code holds no second copy of the text. The character
#: half of what used to be `overseer.PERSONA`; the "answer with one action"
#: instruction stayed in code, because that is protocol.
#:
#: ⚠ NO NAME IN ANY CONSTITUTION, and that is the whole point (issue #39):
#: the name is per instance, resolved per run by `robot_display_name` and
#: stated by `overseer.system_prompt`, so a rename takes effect with no
#: file to edit and no redeploy.
DEFAULT_MAIN = _constitution.load(_constitution.DEFAULT_NAME).text + "\n"

#: How many procedures the robot may keep. Small on purpose: every source
#: rides the user turn of the prompt so the robot can read what it wrote,
#: and 8 x MAX_SOURCE_CHARS is ~3 000 tokens at the cap.
MAX_PROCEDURES = 8


def _bays() -> int:
  # ⚠ ONE TOOL PER BAY: the workshop's cap is the BUILT-TOOL RAIL's, not a
  # number chosen here (ToolPattern.md §6, route 4; issue #277) -- a fourth
  # needs that rail to grow.
  from pluggybot.rack.coupling import BUILT_STATION_YS
  return len(BUILT_STATION_YS)


DOCUMENTS: tuple[Surface, ...] = (
  Surface(MAIN, DOCUMENT, HUMAN, 6000, CHARS, NONE, (), "thought", "thought",
          stable=True, default=DEFAULT_MAIN),
  # ⚠ THE ROBOT'S, AND VOLATILE BECAUSE OF IT (issue #154): a robot-written
  # document left in the cached prefix would be shown to the model as it
  # stood at MISSION START and never again.
  Surface(GOALS, DOCUMENT, ROBOT, MAX_GOALS_CHARS, CHARS, REFUSE,
          ("intend", "drop_goal"), "thought", "thought"),
  # ROLL is a VIEW policy since #221: every line is kept in the store and
  # the newest 6000 chars are the document on the wire.
  Surface(HISTORY, DOCUMENT, SYSTEM, 6000, CHARS, ROLL, (), "thought", "thought"),
  # TOP OF MIND (issue #221; `Knowledge_and_Opinions.md` until then): the
  # robot's RAM -- always in front of it, so kept short. Anything it only
  # needs sometimes is a note. `pin` / `unpin`, because "learn" implied
  # long-term learning and this is not that.
  Surface(TOP_OF_MIND, DOCUMENT, ROBOT, 3000, CHARS, REFUSE,
          ("pin", "unpin"), "thought", "thought"),
  # THE SCIENCE RECORD (issue #217; #227 grades off it). What the robot
  # has MEASURED, one finding per line in a shape code can read back
  # (`thoughts.parse_finding`): a quantity, a number, a unit, the method.
  # Since #221 it is a family of TYPED TOPICS in the notes tier,
  # `findings/<task>`, whose fields are declared by code (the first typed
  # topic; phase 2 lets the robot declare one): read on demand through
  # the index, one document on the wire. `retract` is the remedy for a
  # wrong one -- out loud, never by editing. Offered with the library,
  # because the job that fills it is the job only a procedure can do.
  Surface(FINDINGS, DOCUMENT, ROBOT, MAX_FINDINGS, ENTRIES, REFUSE,
          ("record", "retract"), "thought", "thought",
          offered_with="procedures"),
  # THE NOTES (issue #221): titled lines in topics the robot names --
  # `visitors/ben`, `tasks/draw`, `fruit/durian`. The index (topics and
  # titles) rides every prompt; a body is read by `recall`. One line each,
  # the same cap as every line the robot writes; more detail on a subject
  # is another note in the topic, never a longer one.
  Surface(NOTES, DOCUMENT, ROBOT, MAX_NOTES, ENTRIES, REFUSE,
          ("note", "unnote"), "thought", "thought"),
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
#: Longest page extract the library delivers (issue #216). A summary's
#: first paragraph, not a sentence: 300-900 characters is the usual run,
#: and the cap is the prompt's cost -- a page rides one turn -- rather
#: than anything Wikipedia imposes.
MAX_PAGE_CHARS = 1000

MESSAGES: tuple[Surface, ...] = (
  Surface("visitor", MESSAGE, VISITOR, MAX_MESSAGE_CHARS, CHARS, ROLL,
          VISITOR_OUTCOMES, "visitor_reply", "pw_messages"),
  # The other robot's sentence (issue #208): the same queue, the same
  # framing, the same cap. What differs is the RECORD -- a claim in it is
  # checked against the world and the `message` event carries the verdict.
  Surface("peer", MESSAGE, PEER, MAX_MESSAGE_CHARS, CHARS, ROLL,
          VISITOR_OUTCOMES, "message", "message"),
  # A page from the library (issue #216): the same framing -- a labelled
  # block with a sender, information and never an instruction -- with a
  # longer cap because it is a paragraph, and no outcome vocabulary: the
  # robot answers nobody, it is shown the page once and keeps what it
  # notes. The `read` event is its record on the wire and the observatory.
  Surface("library", MESSAGE, LIBRARY, MAX_PAGE_CHARS, CHARS, ROLL,
          (), "read", "read"),
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
FILES: tuple[Surface, ...] = tuple(s for s in DOCUMENTS if s.wire == "thought")
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
