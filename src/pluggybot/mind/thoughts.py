"""The thought files: the robot's memory as named documents, each with an
owner (issue #38), read off the registry in `mind/text.py` (issue #217),
and since issue #221 VIEWS over one record store (`mind/memory.py`).

  Main.md          HUMAN   the CONSTITUTION: body, manner, and what the person
                           who looks after it hopes for it. A file, edited on
                           the volume; a robot that can rewrite who it is
                           defeats the point.
  Goals.md         ROBOT   what IT has decided to do (issue #154). Core:
                           always shown.
  Top_of_mind.md   ROBOT   its RAM: what it wants in front of it every turn,
                           kept short. Core: always shown.
  History.md       SYSTEM  what happened, append-only. A robot that can edit
                           its own history breaks the same principle that
                           stops it awarding itself points. The tail is
                           shown; the rest is there to `recall`.
  Findings.md      ROBOT   what it has MEASURED (issue #217): typed topics
                           `findings/<task>` in the notes tier, one finding
                           per line in a shape code reads back, so #227 is
                           graded off the record and never off a report.
  Notes.md         ROBOT   titled lines in topics it names. The INDEX is
                           always shown; a body is read by `recall`.

⚠ THE OWNERSHIP SPLIT IS THE POINT (issue #154): a human writes the
constitution and the robot writes its goals, so quality 5 of the mission --
goal creation and follow-through (docs/PluggyPlan.md) -- is read off a
document nobody else wrote.

"Written by" is enforced at the ONE gate (`text.admit`), never promised by
callers: a write by anyone but the document's owner raises `ThoughtRefused`
and is recorded in `refusals`, so a refusal is a visible event rather than
a silent no-op. Human documents have no write API at all.

⚠ THE SPLIT IS BY WRITER (docs/Overseer.md §7): `stable()` and `volatile()`
are one flag read twice, and the ONLY way a document reaches the prompt is
through one of them. A writable document in the cached prefix would not
cost cache hits -- `Overseer.system` is built ONCE -- it would cost the
memory working at all: the model shown its documents as they stood at
mission start.

⚠ THE STORE IS UNBOUNDED; THE VIEW IS CAPPED (issue #221). A removed line
is RETIRED, never deleted, and `recall` can still find it. History's ROLL
is the newest lines that fit its cap; the robot's documents REFUSE, loudly,
rather than silently truncating -- the remedy is the document's remove
verb. Append or remove, never rewrite: there is no verb that replaces a
document, because a full-rewrite verb lets one bad generation erase
everything the robot knows.

The views are written to the volume beside the store so an operator can
read them, and are streamed as `thought` messages (protocol 0.11.0)
whenever one changes, so the site's tab shows the same bytes the model is
shown. The site renders them read-only; nothing it can send changes one.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.mind import text as registry
from pluggybot.mind.memory import RecordStore
from pluggybot.mind.store import FileStore, MemoryStore, Store
from pluggybot.mind.text import (
  DEFAULT_MAIN, FINDINGS, GOALS, HISTORY, HUMAN, MAIN, NOTES, ROBOT, SYSTEM,
  TOP_OF_MIND, Surface,
)
from pluggybot.telemetry.protocol import ROBOT_ROOT

__all__ = ["DEFAULT_MAIN", "FINDINGS", "GOALS", "HISTORY", "HUMAN",
           "TOP_OF_MIND", "NOTES", "MAIN", "ROBOT", "SYSTEM", "Spec", "FILES",
           "SPECS", "NAMES", "ThoughtFiles", "ThoughtRefused"]

#: Longest single line written through the API, in characters -- a goal, a
#: pin, a finding, a NOTE. An LLM handed an unbounded text field will
#: eventually write an essay into a document that is replayed into every
#: one of its future prompts; more on a subject is another note in the
#: topic.
MAX_LINE_CHARS = 400
#: How much of History the model is shown. The whole document is on the
#: wire and every line is in the store; the prompt gets the tail, because
#: the last dozen things that happened are context and the last hundred
#: are tokens. Each line carries its record id so a `pin` or a `note` can
#: cite it.
HISTORY_SHOWN = 12
#: A topic or a title: lowercase, `a-z 0-9 _ - /`, at most two levels,
#: this long. The robot names topics; the index is what it sees them by.
MAX_TOPIC_CHARS = 40
MAX_TITLE_CHARS = 60
#: The science record's family of topics. `record` writes here and `note`
#: may not: the record is read by code, and code reads one shape.
FINDINGS_PREFIX = "findings/"
DEFAULT_FINDINGS_TOPIC = FINDINGS_PREFIX + "general"
#: The store's file, beside the views. `:memory:` without a root.
STORE_FILE = "memory.sqlite"
#: What a `recall` may put in front of the model (issue #221): one block's
#: lines, in characters, and the whole chain's -- paid for as input tokens
#: on every turn until an external action clears it. The oldest block goes
#: first when the chain is over; a block is cut at its cap and says so.
RECALLED_CHARS = 4000
RECALLED_CHAIN_CHARS = 8000
#: How many History lines `read history` returns: the tail beyond the
#: dozen the prompt always carries.
HISTORY_RECALLED = 40

#: Env knob, resolved the way every other deploy setting is: a directory for
#: the documents and the store (`/var/lib/pluggybot` in the image).
ROOT_ENV = "PLUGGY_THOUGHTS"

#: The rows, under the names the rest of the repo reads them by.
Spec = Surface
FILES: tuple[Surface, ...] = registry.FILES
SPECS: dict[str, Surface] = {s.name: s for s in FILES}
NAMES: tuple[str, ...] = tuple(s.name for s in FILES)
#: The core documents: rendered from `core` records, one per line.
CORE = (GOALS, TOP_OF_MIND)


class ThoughtRefused(registry.Refused):
  """A write the table does not allow: wrong writer, full document, or a
  remove that quotes nothing on the page. Always visible -- the caller
  narrates it, and `ThoughtFiles.refusals` keeps it."""


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _line(text) -> str:
  """One line, whitespace collapsed, capped. Empty in -> empty out."""
  return " ".join(str(text or "").split())[:MAX_LINE_CHARS]


def topic_name(text, default: str = "") -> str:
  """A topic the robot named, made safe to key by: lowercase, spaces to
  `_`, anything else dropped, at most two `/` levels, capped. "" -> the
  default."""
  raw = str(text or "").strip().lower().replace(" ", "_")
  parts = [re.sub(r"[^a-z0-9_-]", "", p) for p in raw.split("/")]
  parts = [p.strip("_-") for p in parts if p.strip("_-")]
  name = "/".join(parts[:2])[:MAX_TOPIC_CHARS].strip("/_-")
  return name or default


def title_name(text) -> str:
  return " ".join(str(text or "").split())[:MAX_TITLE_CHARS]


# ---- the science record's line ----------------------------------------------

#: `<quantity> = <value> <unit> -- <method>`: what was measured, the number,
#: its unit, how. The unit and the method are optional; the number is not,
#: because a finding without one is an opinion and belongs in
#: `Top_of_mind.md` or a note. `parse_finding` reads the same shape back,
#: which is what "code-checkable" means: a grader (#227) reads a quantity's
#: latest value off the record with no model in the loop.
_FINDING = re.compile(
  r"^(?P<quantity>.+?) = (?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
  r"(?: (?P<unit>[^\s-][^\s]*))?(?: -- (?P<method>.+))?$")


def format_finding(finding) -> str:
  """A `record` field -> one line, or "" where it is not a finding (no
  quantity, or a value that is not a number)."""
  if not isinstance(finding, dict):
    return ""
  quantity = _line(finding.get("quantity")).replace(" = ", " ").replace(" -- ", " - ")
  try:
    value = float(finding.get("value"))
  except (TypeError, ValueError):
    return ""
  if not quantity or value != value or value in (float("inf"), float("-inf")):
    return ""
  unit = _line(finding.get("unit")).split(" ")[0] if finding.get("unit") else ""
  method = _line(finding.get("method")).replace(" -- ", " - ")
  line = f"{quantity} = {value:g}" + (f" {unit}" if unit else "")
  # The method is what gives at the line cap, never the shape: a line cut
  # inside `<quantity> = <value>` would not read back as a finding.
  room = MAX_LINE_CHARS - len(line) - len(" -- ")
  return line + (f" -- {method[:room]}" if method and room > 0 else "")


def parse_finding(line: str) -> dict | None:
  """One recorded line -> `{quantity, value, unit, method}`, or None where
  the line is not in the record's shape."""
  m = _FINDING.match(line.strip())
  if m is None:
    return None
  return {"quantity": m.group("quantity"), "value": float(m.group("value")),
          "unit": m.group("unit") or "", "method": m.group("method") or ""}


def findings_topic(text) -> str:
  """`mass_bench` or `findings/mass_bench` -> `findings/mass_bench`."""
  name = topic_name(text)
  if not name:
    return DEFAULT_FINDINGS_TOPIC
  if name.startswith(FINDINGS_PREFIX):
    return name
  return FINDINGS_PREFIX + name.split("/")[0]


class ThoughtFiles:
  """The `.md` documents, their permissions, their store and their views.

  `root=None` is in-memory: the defaults, a `:memory:` store, nothing on
  disk, which is what a unit test and a demo without a state directory
  want. With a root, `Main.md` lives at `root/Main.md` (written out with
  its default so there is something on the volume to edit), the store at
  `root/memory.sqlite`, and every other document is rendered there as a
  file an operator can read.

  ⚠ AN OLD VOLUME STARTS BLANK (issue #221). A root with the pre-#221
  files and no store puts the robot's and the system's `.md` files aside
  (`Store.archive`, the true-death path: `History.1.md` beside the fresh
  one) and imports nothing, so an observation period holds only what was
  written through the new verbs. `Main.md` is untouched.
  """

  def __init__(self, root: str | os.PathLike | None = None,
               texts: dict | None = None,
               clock: Callable[[], str] = _now,
               robot: str = ROBOT_ROOT,
               records: RecordStore | None = None) -> None:
    #: WHOSE documents these are on the wire (issue #167): the `thought`
    #: event's `robot`, and the store's key. The first robot's is the
    #: species name, as it always was.
    self.robot = robot
    self.root = Path(root) if root is not None else None
    self.clock = clock
    # The one path to the disk for FILES (issue #217): the constitution and
    # the rendered views.
    self.store: Store = FileStore(self.root) if self.root is not None else MemoryStore()
    fresh = self.root is None or not (self.root / STORE_FILE).exists()
    self.records: RecordStore = records or RecordStore(
      str(self.root / STORE_FILE) if self.root is not None else ":memory:",
      clock=clock)
    self.texts: dict[str, str] = {}
    self.on_event: list[Callable[[dict], None]] = []
    self.refusals: list[str] = []
    self.writes: dict[str, int] = {n: 0 for n in NAMES}
    self.dropped: dict[str, int] = {n: 0 for n in NAMES}
    self.archived_at_start: list[str] = []
    if self.root is not None and fresh and records is None:
      self.archived_at_start = self._start_blank()
    self.texts[MAIN] = self._load_main(texts)
    for name, text in (texts or {}).items():
      if name != MAIN:
        self._seed(name, text)
    for name in NAMES:
      if name != MAIN:
        self._render(name)
    if self.root is not None and self.store.read(MAIN) is None:
      # Materialised so a person finds a file to EDIT, carrying the same
      # text the robot is already living by. A bootstrap, not a write: the
      # counters and the hooks do not see it. ⚠ THE ONLY DOCUMENT LAID OUT
      # UP FRONT: an empty `Goals.md` on the volume would invite exactly
      # the human edit the ownership split exists to stop (issue #154). The
      # views appear when there is something in them.
      self._write(MAIN)

  @classmethod
  def open(cls, root: str | os.PathLike | None = None,
           robot: str = ROBOT_ROOT) -> "ThoughtFiles":
    """The deploy shape: an explicit root, else the environment, else memory."""
    root = root or os.environ.get(ROOT_ENV, "").strip() or None
    return cls(root, robot=robot)

  # ---- starting ---------------------------------------------------------------

  def _start_blank(self) -> list[str]:
    """Put a pre-#221 volume's robot- and system-written files aside."""
    aside = []
    for name in [n for n in NAMES if n != MAIN] + ["Knowledge_and_Opinions.md"]:
      if self.store.read(name) is not None and self.store.archive(name):
        aside.append(name)
    return aside

  def _load_main(self, texts: dict | None) -> str:
    spec = SPECS[MAIN]
    given = (texts or {}).get(MAIN)
    if given is not None:
      return given.strip()[:spec.cap]
    if self.root is not None:
      text = self.store.read(MAIN)
      if text is not None:
        # Capped on read: a guard against a mounted file being something
        # nobody intended (a log), not against the author.
        text = text[:spec.cap].strip()
        if text:
          return text
    return spec.default.strip()

  def _seed(self, name: str, text: str) -> None:
    """A document given as text (a test's `texts=`), line by line."""
    for line in (text or "").splitlines():
      line = _line(line)
      if not line:
        continue
      if name in CORE:
        self.records.add(self.robot, "core", ROBOT, line, topic=name)
      elif name == HISTORY:
        self.records.add(self.robot, "history", SYSTEM, line)
      elif name == FINDINGS:
        parsed = parse_finding(line)
        if parsed:
          self.records.add(self.robot, "note", ROBOT, line,
                           topic=DEFAULT_FINDINGS_TOPIC,
                           title=parsed["quantity"], fields=parsed)

  # ---- reading --------------------------------------------------------------

  def read(self, name: str) -> str:
    return self.texts[name]

  def spec(self, name: str) -> Surface:
    return SPECS[name]

  def lines(self, name: str) -> list[str]:
    return [ln for ln in self.texts[name].splitlines() if ln.strip()]

  def _core(self, name: str) -> list:
    return self.records.active(self.robot, "core", topic=name)

  def _findings(self) -> list:
    return self.records.active(self.robot, "note", prefix=FINDINGS_PREFIX)

  def _notes(self) -> list:
    return [r for r in self.records.active(self.robot, "note")
            if not r.topic.startswith(FINDINGS_PREFIX)]

  def findings(self) -> list[dict]:
    """The science record, parsed: every finding, oldest first, each with
    its `topic`. Read off the records, never off the rendered text."""
    return [dict(r.fields, topic=r.topic) for r in self._findings()]

  def index(self, name: str) -> dict[str, list[str]]:
    """What the model is shown of a notes-tier document: every topic with
    its titles -- for findings, `quantity = value unit` (the method waits
    for a `recall`)."""
    out: dict[str, list[str]] = {}
    rows = self._findings() if name == FINDINGS else self._notes()
    for r in rows:
      label = (r.text.split(" -- ")[0] if name == FINDINGS else r.title)
      out.setdefault(r.topic, []).append(label)
    return out

  def history_tail(self, n: int = HISTORY_SHOWN) -> list[str]:
    """The newest `n` History lines, each prefixed with its record id so
    it can be cited."""
    return [f"#{r.id} {r.text}" for r in self.records.tail(self.robot, "history", n)]

  def stable(self) -> dict[str, str]:
    """The documents that ride the cached prefix: the human-only ones."""
    return {s.name: self.texts[s.name] for s in FILES if s.stable}

  def volatile(self, menu=None) -> dict:
    """The documents that ride the user turn, keyed BY DOCUMENT NAME.

    ⚠ Derived from the SAME `stable` flag `stable()` reads, and inverted
    rather than listed: two hand-written lists can disagree, and the way
    they disagree is a document that reaches the model through NEITHER
    half -- which reads, from the outside, exactly like a robot that never
    learns anything.

    `menu` narrows it to the documents this arm is TOLD ABOUT
    (`text.offered`): a document whose verbs are not on the menu is not
    shown either. None shows every writable document, which is what the
    halves-agree test reads.

    Core documents are shown WHOLE; `History.md` is TAILED (with record
    ids); the notes-tier documents are shown as their INDEX -- the topics
    and the titles, which is what bounds the prompt whatever the robot
    has written, and a body is a `recall` away.
    """
    out = {}
    for s in FILES:
      if s.stable or (menu is not None and not registry.offered(s, menu)):
        continue
      if s.name == HISTORY:
        out[s.name] = self.history_tail()
      elif s.name in (FINDINGS, NOTES):
        out[s.name] = self.index(s.name)
      else:
        out[s.name] = self.texts[s.name]
    return out

  # ---- writing --------------------------------------------------------------

  def _refuse(self, why: str) -> None:
    self.refusals.append(why)
    raise ThoughtRefused(why)

  def _admit(self, name: str, by: str, size: int) -> bool:
    """`text.admit` with the refusal counted here. True within the cap;
    False where History has to roll; raises otherwise."""
    spec = SPECS.get(name)
    if spec is None:
      self._refuse(f"{name}: no such thought file")
    try:
      return registry.admit(spec, by, size)
    except registry.Refused as e:
      self._refuse(str(e))

  def append(self, name: str, text: str, by: str, t: float = 0.0,
             cites=()) -> str:
    """Add one line to a core document or to History. Returns the line
    written, "" if there was nothing to.

    History rolls (the view keeps the newest lines that fit); the robot's
    documents refuse when full -- the row says which.
    """
    line = _line(text)
    size = (len("\n".join([r.text for r in self._core(name)] + [line]))
            if name in CORE and line else 0)
    self._admit(name, by, size)            # the writer is checked regardless
    if not line:
      return ""
    if name == HISTORY:
      self.records.add(self.robot, "history", by, line, t=t)
    elif name in CORE:
      self.records.add(self.robot, "core", by, line, t=t, topic=name,
                       cites=_cites(cites))
    else:
      self._refuse(f"{name}: not a line document -- its verbs are "
                   f"{' / '.join(SPECS[name].verbs)}")
    self._commit(name, t)
    return line

  def forget(self, name: str, text: str, by: str, t: float = 0.0) -> str:
    """Retire one line of a core document. The robot QUOTES what it wants
    gone; an exact line wins, else the one line containing the quote, else
    a refusal -- because a remove that matched loosely could take out a
    line the robot meant to keep, and a silent miss would leave it
    believing something it had decided not to."""
    self._admit(name, by, 0)
    quote = _line(text)
    if not quote:
      return ""
    if name not in CORE:
      self._refuse(f"{name}: not a line document")
    hit = _match(self._core(name), quote, name, self._refuse)
    self.records.retire(hit.id, t=t)
    self._commit(name, t)
    return hit.text

  # The robot's verbs, by name, and the one the SYSTEM has -- each names its
  # document in code, so a caller cannot get the writer wrong and no
  # parameter names a document.

  def pin(self, text: str, t: float = 0.0, cites=()) -> str:
    """Put one line in front of the robot every turn (issue #221)."""
    return self.append(TOP_OF_MIND, text, by=ROBOT, t=t, cites=cites)

  def unpin(self, text: str, t: float = 0.0) -> str:
    return self.forget(TOP_OF_MIND, text, by=ROBOT, t=t)

  def intend(self, text: str, t: float = 0.0) -> str:
    """Add one goal the robot set itself (issue #154)."""
    return self.append(GOALS, text, by=ROBOT, t=t)

  def drop_goal(self, text: str, t: float = 0.0) -> str:
    """Drop one goal the robot can quote -- finished, or thought better of.

    Deliberately the same verb shape as `unpin` and NOT a separate
    "completed" state: a goals document that distinguished done from
    abandoned would need the robot to be honest about which, and the
    reason it gives is already on the wire. What is measurable either way
    is that the line left the page and when.
    """
    return self.forget(GOALS, text, by=ROBOT, t=t)

  def record(self, finding, t: float = 0.0) -> str:
    """Add one finding to the science record (issue #217): `{quantity,
    value, unit?, method?, topic?}`, written in the one shape
    `parse_finding` reads into the topic `findings/<task>` it names
    (`findings/general` when it names none). Refused, not reshaped, when
    it is not a finding: a record that accepted prose would be a second
    opinions document."""
    line = format_finding(finding)
    if not line:
      if finding:
        self._refuse(f"{FINDINGS}: a finding is a quantity, a number and a "
                     f"unit, not {str(finding)[:60]!r}")
      return ""
    self._admit(FINDINGS, ROBOT, len(self._findings()) + 1)
    parsed = parse_finding(line)
    self.records.add(self.robot, "note", ROBOT, line, t=t,
                     topic=findings_topic(finding.get("topic")),
                     title=title_name(parsed["quantity"]), fields=parsed)
    self._commit(FINDINGS, t)
    return line

  def retract(self, text: str, t: float = 0.0) -> str:
    """Take one finding off the record, quoted -- a retraction, out loud,
    which is what a record of measurements does with a wrong one."""
    self._admit(FINDINGS, ROBOT, 0)
    quote = _line(text)
    if not quote:
      return ""
    hit = _match(self._findings(), quote, FINDINGS, self._refuse)
    self.records.retire(hit.id, t=t)
    self._commit(FINDINGS, t)
    return hit.text

  def note(self, payload, t: float = 0.0, cites=()) -> str:
    """Add one note (issue #221): `{topic, title, text}`. The topic is the
    robot's to name; `findings/...` is not (that is `record`'s). A title
    already in the topic is refused -- `unnote` it first -- because there
    is no verb that replaces."""
    if not isinstance(payload, dict):
      self._refuse(f"{NOTES}: a note is a topic, a title and a text, not "
                   f"{str(payload)[:60]!r}")
    topic = topic_name(payload.get("topic"))
    title = title_name(payload.get("title"))
    text = _line(payload.get("text"))
    if not (topic and title and text):
      self._refuse(f"{NOTES}: a note needs a topic, a title and a text")
    if topic.split("/")[0] == FINDINGS_PREFIX.rstrip("/"):
      self._refuse(f"{NOTES}: {topic!r} is the science record; write a "
                   f"finding with `record`")
    notes = self._notes()
    if any(r.topic == topic and r.title == title for r in notes):
      self._refuse(f"{NOTES}: {topic}/{title} is already written; unnote it first")
    self._admit(NOTES, ROBOT, len(notes) + 1)
    self.records.add(self.robot, "note", ROBOT, text, t=t, topic=topic,
                     title=title, cites=_cites(cites))
    self._commit(NOTES, t)
    return f"{topic}/{title}: {text}"

  def unnote(self, text, t: float = 0.0) -> str:
    """Retire one note: `topic/title`, a title, or a quote of its text --
    one match, or a refusal."""
    self._admit(NOTES, ROBOT, 0)
    quote = _line(text)
    if not quote:
      return ""
    notes = self._notes()
    hits = [r for r in notes if f"{r.topic}/{r.title}" == quote]
    if not hits:
      hits = [r for r in notes if r.title == quote]
    if not hits:
      hits = [r for r in notes if r.text == quote]
    if not hits:
      hits = [r for r in notes if quote in r.text or quote in r.title]
    if len(hits) != 1:
      self._refuse(f"{NOTES}: {'nothing' if not hits else f'{len(hits)} notes'}"
                   f" match {quote[:60]!r}")
    hit = hits[0]
    self.records.retire(hit.id, t=t)
    self._commit(NOTES, t)
    return f"{hit.topic}/{hit.title}: {hit.text}"

  def remember(self, text: str, t: float = 0.0) -> str:
    """The narrative record, the SYSTEM's. Prefixed with the sim clock,
    because a line of history with no "when" is an anecdote."""
    return self.append(HISTORY, f"[t={float(t):.0f}s] {text}", by=SYSTEM,
                       t=t)

  def think(self, text: str, t: float = 0.0, why: str = "") -> str:
    """The scratch the model wrote before a decision (issue #221): a
    `think` record, kept apart from History so its tail stays what
    happened rather than what was mused. Not a document -- nothing is
    rendered -- and it rides the wire as the `journal` message (0.7.0:
    `text`, and `why` is the decision it preceded), which the site kept
    for the `journal` action this replaced."""
    line = " ".join(str(text or "").split())
    if not line:
      return ""
    self.records.add(self.robot, "think", ROBOT, line, t=t)
    msg = {"type": "journal", "t": round(float(t), 3), "at": self.clock(),
           "robot": self.robot, "text": line}
    if why:
      msg["why"] = " ".join(str(why).split())[:MAX_LINE_CHARS]
    for hook in self.on_event:
      hook(dict(msg))
    return line

  def last_thoughts(self, n: int) -> list[str]:
    return [r.text for r in self.records.tail(self.robot, "think", n)]

  # ---- recall (issue #221) --------------------------------------------------

  def recall(self, read: str = "", find: str = "") -> dict:
    """Look something up. `read` is a KEY: `#123` a record by its number
    (as the History tail and a recalled line show it); `topic/title` one
    note; `topic` every note in it (`findings` reads every finding, a
    family by its first level); `history` the tail beyond what the prompt
    carries. `find` is WORDS: full-text over everything this robot has
    written or been told this life, retired lines included. Both may be
    set. Returns the block the next turn shows: `{read, find, hits,
    lines}`, each line `#id [where] text`, capped at `RECALLED_CHARS`."""
    rows: list = []
    seen: set[int] = set()
    key = " ".join(str(read or "").split())
    if key:
      for r in self._read(key):
        if r.id not in seen and not _about_recall(r):
          rows.append(r)
          seen.add(r.id)
    words = " ".join(str(find or "").split())
    if words:
      for r in self.records.find(self.robot, words):
        if r.id not in seen and not _about_recall(r):
          rows.append(r)
          seen.add(r.id)
    lines: list[str] = []
    size = 0
    cut = 0
    for r in rows:
      line = self._recalled_line(r)
      if size + len(line) > RECALLED_CHARS:
        cut += 1
        continue
      lines.append(line)
      size += len(line) + 1
    block = {"read": key, "find": words, "hits": len(rows), "lines": lines}
    if cut:
      block["cut"] = cut
    return block

  def _read(self, key: str) -> list:
    """The rows a key names, or none."""
    if re.fullmatch(r"#?\d+", key):
      r = self.records.get(int(key.lstrip("#")))
      return [r] if r is not None and r.robot == self.robot \
        and r.generation == self.records.generation(self.robot) else []
    if key.lower() in ("history", HISTORY.lower()):
      return self.records.tail(self.robot, "history", HISTORY_RECALLED)
    notes = self.records.active(self.robot, "note")
    exact = [r for r in notes if f"{r.topic}/{r.title}" == key]
    if exact:
      return exact
    topic = topic_name(key)
    hit = [r for r in notes if r.topic == topic]
    if hit:
      return hit
    return [r for r in notes if r.topic.startswith(topic + "/")]

  def _recalled_line(self, r) -> str:
    if r.kind == "note":
      where = f"{r.topic}/{r.title}" if not r.topic.startswith(FINDINGS_PREFIX) else r.topic
    elif r.kind == "history":
      where = "history"                 # its text carries `[t=..s]` already
    else:
      where = f"{r.topic or r.kind} t={r.t:.0f}s"
    if not r.active:
      where += " retired"
    return f"#{r.id} [{where}] {r.text}"

  def apply(self, verb: str, payload, t: float = 0.0, cites=()) -> str:
    """The robot's verbs, BY NAME (issue #217): the registry says which
    document a verb is on and whether it adds or removes; this dispatches.
    `_reconsider` iterates `text.line_verbs()` over it, so a new document's
    verbs reach the mission by adding a row, not a branch. Returns the line
    written or removed, "" for nothing to do; raises `ThoughtRefused`."""
    surface = registry.BY_VERB.get(verb)
    if surface is None or surface.wire != "thought":
      self._refuse(f"{verb!r} is not a verb on any thought file")
    if surface.name == FINDINGS:
      return self.retract(payload, t=t) if verb == surface.remove else self.record(payload, t=t)
    if surface.name == NOTES:
      return self.unnote(payload, t=t) if verb == surface.remove else self.note(payload, t=t, cites=cites)
    if verb == surface.remove:
      return self.forget(surface.name, payload, by=ROBOT, t=t)
    return self.append(surface.name, payload, by=ROBOT, t=t, cites=cites)

  # ---- the views --------------------------------------------------------------

  def _render(self, name: str) -> None:
    if name in CORE:
      self.texts[name] = "\n".join(r.text for r in self._core(name))
    elif name == HISTORY:
      # The ROLL, as a view: the newest lines that fit the cap.
      rows = self.records.active(self.robot, "history")
      kept: list[str] = []
      size = 0
      for r in reversed(rows):
        size += len(r.text) + (1 if kept else 0)
        if size > SPECS[HISTORY].cap:
          break
        kept.append(r.text)
      self.dropped[HISTORY] = len(rows) - len(kept)
      self.texts[name] = "\n".join(reversed(kept))
    elif name == FINDINGS:
      self.texts[name] = _grouped(self._findings(), lambda r: r.text)
    elif name == NOTES:
      self.texts[name] = _grouped(self._notes(), lambda r: f"- {r.title}: {r.text}")

  def _commit(self, name: str, t: float) -> None:
    self._render(name)
    self.writes[name] += 1
    if self.root is not None:
      self._write(name)
    msg = self.message(name, t)
    for hook in self.on_event:
      hook(dict(msg))

  def _write(self, name: str) -> None:
    self.store.write(name, self.texts[name] + "\n")

  # ---- true death (issue #136) ----------------------------------------------

  def archive(self, t: float = 0.0) -> dict:
    """Put this robot's memory away and start the next one empty.

    Called only by `HubLifecycle._true_death`, when the hearts run out. What
    goes is what the ROBOT and the SYSTEM wrote -- History, its own
    unrevisable record of what happened to it; Top_of_mind, Notes and
    Findings, everything it worked out and measured; and since issue #154
    `Goals.md`, everything it meant to do. Per Evaluation.md section 6 that
    is the cheapest real cost there is, and it is the whole of what makes a
    true death different from an ordinary one.

    ⚠ THE GOALS GO WITH IT, and that follows from the writer table rather
    than from a list here: they are the ROBOT's, and the next robot is a NEW
    ROBOT. Inheriting a dead predecessor's goals would be the one thing a
    true death is supposed to cost -- carrying on its work -- handed back for
    free.

    ⚠ THE CONSTITUTION SURVIVES, and this is not softness. `Main.md` says who
    a robot here is and what the person who looks after it hopes for it; a
    person put it on the volume by hand and there is no write API for it.

    ⚠ NOTHING IS DELETED: the store moves the robot on to its next
    GENERATION and keeps the rows (`RecordStore.archive`), and the rendered
    views are kept beside the new ones (`History.1.md`, `Store.archive`). A
    stake whose evidence is unlinked is a stake nobody can audit afterwards,
    and the volume is where the operator looks.
    """
    gone = [n for n in NAMES if SPECS[n].writer in (SYSTEM, ROBOT)]
    kept = {n: self.texts[n] for n in gone}
    rows = self.records.archive(self.robot, t=t)
    for name in gone:
      if self.root is not None:
        self.store.archive(name)
      self._render(name)
      if self.root is not None:
        self._write(name)
    return {"cleared": gone, "records": rows,
            "chars": {n: len(v) for n, v in kept.items()}}

  # ---- the wire -------------------------------------------------------------

  def message(self, name: str, t: float = 0.0) -> dict:
    """One document as a `thought` message (0.11.0): the whole text, not a
    delta, because a document is small and "present means complete" is the
    rule that keeps a late joiner and a scrubbed recording honest."""
    spec = SPECS[name]
    return {"type": spec.wire, "t": round(float(t), 3), "robot": self.robot,
            "name": name, "text": self.texts[name], "writer": spec.writer,
            "cap": spec.cap}

  def messages(self, t: float = 0.0) -> list[dict]:
    return [self.message(n, t) for n in NAMES]

  def stats(self) -> dict:
    """Counters, plus the ROBOT'S OWN GOALS in full (issue #154).

    ⚠ ONE TEXT, AND ONLY THIS ONE. Quality 5 of the mission is read off what
    the robot was still holding at the end of a run, and a count of writes
    cannot answer it: three goals written and three abandoned looks identical
    to three written and kept. The other documents stay counters here --
    they are on the wire in full as `thought` messages, and a run record
    that embedded every document would carry the constitution in every row.
    """
    return {"root": str(self.root) if self.root is not None else "",
            "chars": {n: len(self.texts[n]) for n in NAMES},
            "writes": dict(self.writes), "dropped": dict(self.dropped),
            "records": {"active": self.records.count(self.robot, status="active"),
                        "retired": self.records.count(self.robot, status="retired"),
                        "generation": self.records.generation(self.robot)},
            "goals": self.texts[GOALS],
            "refusals": list(self.refusals[-5:])}


def _about_recall(r) -> bool:
  """A recall never finds the record of a recall: the decision's own
  `chose recall (find ...)` line matches its own words every time, and a
  `read history` full of `recalled 3 lines` would be a robot reading
  about reading."""
  return r.kind == "history" and ("] chose recall" in r.text
                                  or "] recalled " in r.text)


def _cites(cites) -> tuple[int, ...]:
  out = []
  for c in cites or ():
    try:
      out.append(int(str(c).lstrip("#")))
    except ValueError:
      continue
  return tuple(out)


def _match(rows, quote: str, name: str, refuse):
  """The one row a quote picks out: an exact text wins, else the one row
  containing it, else a refusal."""
  hits = [r for r in rows if r.text == quote]
  if not hits:
    hits = [r for r in rows if quote in r.text]
  if len(hits) != 1:
    refuse(f"{name}: {'nothing' if not hits else f'{len(hits)} lines'}"
           f" on the page match {quote[:60]!r}")
  return hits[0]


def _grouped(rows, line) -> str:
  """A notes-tier document as text: `## topic` then its lines."""
  out: list[str] = []
  topic = None
  for r in rows:
    if r.topic != topic:
      if out:
        out.append("")
      out.append(f"## {r.topic}")
      topic = r.topic
    out.append(line(r))
  return "\n".join(out)
