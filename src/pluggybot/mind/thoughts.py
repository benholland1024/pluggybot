"""The thought files: the robot's memory as named documents, each with an
owner (issue #38), read off the registry in `mind/text.py` (issue #217).

Issue #15 gave the overseer two files -- `goals.md`, read and never written,
and `journal.json`, written and never edited -- and the asymmetry between
them was the design. Issue #38 made it a TABLE of Markdown documents, each
saying who may write it; #217 moved that table into the registry every
text surface shares, so this module is now the `.md` DOCUMENTS' owner and
nothing else: it reads the rows, formats lines, and asks `text.admit` and
`store.Store` for everything a row decides.

  Main.md                    HUMAN   the CONSTITUTION: body, manner, and what
                                     the person who looks after it hopes for
                                     it. A robot that can rewrite who it is
                                     defeats the point.
  Goals.md                   ROBOT   what IT has decided to do (issue #154).
  History.md                 SYSTEM  what happened, append-only. A robot that
                                     can edit its own history breaks the same
                                     principle that stops it awarding itself
                                     points (economy/scoring.py).
  Knowledge_and_Opinions.md  ROBOT   what it has learned and what it thinks.
  Findings.md                ROBOT   what it has MEASURED (issue #217): one
                                     finding per line, in a shape code can
                                     read back, so a job like #227's is
                                     graded off the record and never off a
                                     report.

⚠ THE OWNERSHIP SPLIT IS THE POINT (issue #154). A human writes the
constitution and the robot writes its goals, so quality 5 of the mission --
goal creation and follow-through (docs/PluggyPlan.md) -- is read off a file
nobody else wrote.

"Written by" is enforced at the ONE gate (`text.admit`), never promised by
callers: a write by anyone but the file's owner raises `ThoughtRefused` and
is recorded in `refusals`, so a refusal is a visible event rather than a
silent no-op. Human files have no write API at all -- a person edits the
file on the volume, and the next run reads it.

⚠ THE SPLIT IS BY WRITER (docs/Overseer.md §7): `stable()` and `volatile()`
are one flag read twice, and the ONLY way a file reaches the prompt is
through one of them. A writable file in the cached prefix would not cost
cache hits -- `Overseer.system` is built ONCE -- it would cost the memory
working at all: the model shown its files as they stood at mission start.

Caps are the rows'. History ROLLS (the oldest lines fall off the front);
the robot's files REFUSE, loudly, rather than silently truncating -- the
remedy is the file's remove verb, and that is the point: an unbounded file
the robot appends to eventually eats the context window and then the
budget. Append or remove, never rewrite: there is no verb that replaces a
file, because a full-rewrite verb lets one bad generation erase everything
the robot knows.

Persisted through a `Store` with the rest of the world state (the volume
the boards and the ledger live in) and streamed as `thought` messages
(protocol 0.11.0) whenever one changes, so the site's tab shows the same
bytes the model is shown. The site renders them read-only; nothing it can
send changes a file.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.mind import text as registry
from pluggybot.mind.store import FileStore, MemoryStore, Store
from pluggybot.mind.text import (
  DEFAULT_MAIN, FINDINGS, GOALS, HISTORY, HUMAN, KNOWLEDGE, MAIN, ROBOT,
  SYSTEM, Surface,
)
from pluggybot.telemetry.protocol import ROBOT_ROOT

__all__ = ["DEFAULT_MAIN", "FINDINGS", "GOALS", "HISTORY", "HUMAN", "KNOWLEDGE",
           "MAIN", "ROBOT", "SYSTEM", "Spec", "FILES", "SPECS", "NAMES",
           "ThoughtFiles", "ThoughtRefused"]

#: Longest single line written through the API, in characters. The same
#: figure as a journal note, for the same reason: an LLM handed an unbounded
#: text field will eventually write an essay into a file that is replayed
#: into every one of its future prompts.
MAX_LINE_CHARS = 400
#: How much of History the model is shown. The whole file is on the wire and
#: on disk; the prompt gets the tail, because the last dozen things that
#: happened are context and the last hundred are tokens.
HISTORY_SHOWN = 12

#: Env knobs, resolved the way every other deploy setting is: a directory
#: for the files (`/var/lib/pluggybot` in the image) and the pre-#38 goals
#: path, which keeps meaning "this file is Goals.md" so an existing volume's
#: hand-edited goals.md is not silently replaced by the defaults.
ROOT_ENV = "PLUGGY_THOUGHTS"
GOALS_ENV = "PLUGGY_GOALS"

#: The rows, under the names the rest of the repo reads them by.
Spec = Surface
FILES: tuple[Surface, ...] = registry.FILES
SPECS: dict[str, Surface] = {s.name: s for s in FILES}
NAMES: tuple[str, ...] = tuple(s.name for s in FILES)


class ThoughtRefused(registry.Refused):
  """A write the table does not allow: wrong writer, full file, or a
  `forget` that quotes nothing on the page. Always visible -- the caller
  narrates it, and `ThoughtFiles.refusals` keeps it."""


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _line(text) -> str:
  """One line, whitespace collapsed, capped. Empty in -> empty out."""
  return " ".join(str(text or "").split())[:MAX_LINE_CHARS]


# ---- the science record's line ----------------------------------------------

#: `<quantity> = <value> <unit> -- <method>`: what was measured, the number,
#: its unit, how. The unit and the method are optional; the number is not,
#: because a finding without one is an opinion and belongs in
#: `Knowledge_and_Opinions.md`. `parse_finding` reads the same shape back,
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


class ThoughtFiles:
  """The `.md` documents, their permissions, and their persistence.

  `root=None` is in-memory: the defaults, nothing on disk, which is what a
  unit test and a demo without a state directory want. With a root, every
  file lives at `root/<Name>`; missing HUMAN files are written out with
  their defaults so there is something on the volume to edit, and the
  others are created on first write.

  `goals_path` overrides where Goals.md is read from (`--goals`,
  `$PLUGGY_GOALS`) -- the pre-#38 file, still honoured so a deploy that has
  been editing `goals.md` keeps its goals.
  """

  def __init__(self, root: str | os.PathLike | None = None,
               goals_path: str | os.PathLike | None = None,
               texts: dict | None = None,
               clock: Callable[[], str] = _now,
               robot: str = ROBOT_ROOT) -> None:
    #: WHOSE files these are on the wire (issue #167): the `thought` event's
    #: `robot`. The first robot's is the species name, as it always was.
    self.robot = robot
    self.root = Path(root) if root is not None else None
    self.goals_path = Path(goals_path) if goals_path is not None else None
    self.clock = clock
    # The one path to the disk (issue #217). A goals alias without a root
    # READS the pre-#38 file and writes nothing, because reading a file
    # should not create one -- `_commit` persists only with a root.
    self.store: Store = (FileStore(self.root, aliases={GOALS: self.goals_path}
                                   if self.goals_path is not None else None)
                         if (self.root is not None or self.goals_path is not None)
                         else MemoryStore())
    self.texts: dict[str, str] = {}
    self.on_event: list[Callable[[dict], None]] = []
    self.refusals: list[str] = []
    self.writes: dict[str, int] = {n: 0 for n in NAMES}
    self.dropped: dict[str, int] = {n: 0 for n in NAMES}
    for spec in FILES:
      self.texts[spec.name] = self._load(spec, (texts or {}).get(spec.name))
    if self.root is not None:
      for spec in FILES:
        if spec.writer == HUMAN and self.store.read(spec.name) is None:
          # Materialised so a person finds a file to EDIT, carrying the same
          # text the robot is already living by. A bootstrap, not a write:
          # the counters and the hooks do not see it, and an existing file is
          # never touched.
          self._write(spec.name)

  @classmethod
  def open(cls, root: str | os.PathLike | None = None,
           goals_path: str | os.PathLike | None = None,
           robot: str = ROBOT_ROOT) -> "ThoughtFiles":
    """The deploy shape: explicit paths, else the environment, else memory."""
    root = root or os.environ.get(ROOT_ENV, "").strip() or None
    goals_path = goals_path or os.environ.get(GOALS_ENV, "").strip() or None
    return cls(root, goals_path=goals_path, robot=robot)

  # ---- reading --------------------------------------------------------------

  def _load(self, spec: Surface, given: str | None) -> str:
    if given is not None:
      return given.strip()[:spec.cap]
    if self.root is not None or spec.name == GOALS:
      text = self.store.read(spec.name)
      if text is not None:
        # Capped on read as well as on write: a human file is a guard against
        # a mounted file being something nobody intended (a log), not against
        # the author -- the same rule `read_goals` applies.
        text = text[:spec.cap].strip()
        if text or spec.writer != HUMAN:
          return text
    return spec.default.strip()

  def read(self, name: str) -> str:
    return self.texts[name]

  def spec(self, name: str) -> Surface:
    return SPECS[name]

  def lines(self, name: str) -> list[str]:
    return [ln for ln in self.texts[name].splitlines() if ln.strip()]

  def findings(self) -> list[dict]:
    """The science record, parsed: every line in the record's shape, oldest
    first. A line the robot wrote in some other shape is not a finding and
    is not here -- the record is read by code, and code reads one shape."""
    out = []
    for ln in self.lines(FINDINGS):
      parsed = parse_finding(ln)
      if parsed is not None:
        out.append(parsed)
    return out

  def stable(self) -> dict[str, str]:
    """The files that ride the cached prefix: the human-only ones."""
    return {s.name: self.texts[s.name] for s in FILES if s.stable}

  def volatile(self, menu=None) -> dict:
    """The files that ride the user turn, keyed BY FILE NAME.

    ⚠ Derived from the SAME `stable` flag `stable()` reads, and inverted
    rather than listed: two hand-written lists can disagree, and the way
    they disagree is a file that reaches the model through NEITHER half --
    which reads, from the outside, exactly like a robot that never learns
    anything.

    `menu` narrows it to the documents this arm is TOLD ABOUT
    (`text.offered`): a document whose verbs are not on the menu is not
    shown either, so `guarded` sees the files it always saw. None shows
    every writable document, which is what the halves-agree test reads.

    The names rather than tidier keys, because the rules block in the prompt
    names these files and a model shown `Knowledge_and_Opinions.md` in one
    breath and `knowledge` in the next has to guess they are the same page.
    `History.md` is TAILED: the whole file is on disk and on the wire, and
    the last dozen lines are what is worth paying input tokens for.
    """
    return {s.name: (self.lines(s.name)[-HISTORY_SHOWN:] if s.name == HISTORY
                     else self.texts[s.name])
            for s in FILES if not s.stable
            and (menu is None or registry.offered(s, menu))}

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

  def append(self, name: str, text: str, by: str, t: float = 0.0) -> str:
    """Add one line. Returns the line written, "" if there was nothing to.

    History rolls (oldest lines off the front); the robot's files refuse
    when full -- the row says which.
    """
    line = _line(text)
    if not line:
      self._admit(name, by, 0)             # the writer is checked regardless
      return ""
    lines = self.lines(name)
    lines.append(line)
    while not self._admit(name, by, len("\n".join(lines))):
      if len(lines) == 1:
        self._refuse(registry.full_message(SPECS[name], len(line), SPECS[name].cap))
      lines.pop(0)
      self.dropped[name] += 1
    self._commit(name, lines, t)
    return line

  def forget(self, name: str, text: str, by: str, t: float = 0.0) -> str:
    """Remove one line. The robot QUOTES what it wants gone; an exact line
    wins, else the one line containing the quote, else a refusal -- because
    a `forget` that matched loosely could take out a line the robot meant to
    keep, and a silent miss would leave it believing something it had
    decided not to."""
    self._admit(name, by, 0)
    quote = _line(text)
    if not quote:
      return ""
    lines = self.lines(name)
    hits = [i for i, ln in enumerate(lines) if ln == quote]
    if not hits:
      hits = [i for i, ln in enumerate(lines) if quote in ln]
    if len(hits) != 1:
      self._refuse(f"{name}: {'nothing' if not hits else f'{len(hits)} lines'}"
                   f" on the page match {quote[:60]!r}")
    gone = lines.pop(hits[0])
    self._commit(name, lines, t)
    return gone

  # The robot's verbs, by name, and the one the SYSTEM has -- each names its
  # file in code, so a caller cannot get the writer wrong and no parameter
  # names a file.

  def learn(self, text: str, t: float = 0.0) -> str:
    return self.append(KNOWLEDGE, text, by=ROBOT, t=t)

  def unlearn(self, text: str, t: float = 0.0) -> str:
    return self.forget(KNOWLEDGE, text, by=ROBOT, t=t)

  def intend(self, text: str, t: float = 0.0) -> str:
    """Add one goal the robot set itself (issue #154)."""
    return self.append(GOALS, text, by=ROBOT, t=t)

  def drop_goal(self, text: str, t: float = 0.0) -> str:
    """Drop one goal the robot can quote -- finished, or thought better of.

    Deliberately the same verb shape as `unlearn` and NOT a separate
    "completed" state: a goals file that distinguished done from abandoned
    would need the robot to be honest about which, and the reason it gives
    is already on the wire. What is measurable either way is that the line
    left the page and when.
    """
    return self.forget(GOALS, text, by=ROBOT, t=t)

  def record(self, finding, t: float = 0.0) -> str:
    """Add one finding to the science record (issue #217): `{quantity,
    value, unit?, method?}`, written in the one shape `parse_finding`
    reads. Refused, not reshaped, when it is not a finding: a record that
    accepted prose would be a second opinions file."""
    line = format_finding(finding)
    if not line:
      if finding:
        self._refuse(f"{FINDINGS}: a finding is a quantity, a number and a "
                     f"unit, not {str(finding)[:60]!r}")
      return ""
    return self.append(FINDINGS, line, by=ROBOT, t=t)

  def retract(self, text: str, t: float = 0.0) -> str:
    """Take one finding off the record, quoted -- a retraction, out loud,
    which is what a record of measurements does with a wrong one."""
    return self.forget(FINDINGS, text, by=ROBOT, t=t)

  def remember(self, text: str, t: float = 0.0) -> str:
    """The narrative record, the SYSTEM's. Prefixed with the sim clock,
    because a line of history with no "when" is an anecdote."""
    return self.append(HISTORY, f"[t={float(t):.0f}s] {text}", by=SYSTEM,
                       t=t)

  def apply(self, verb: str, payload, t: float = 0.0) -> str:
    """The robot's verbs, BY NAME (issue #217): the registry says which
    document a verb is on and whether it adds or removes; this dispatches.
    `_reconsider` iterates `text.line_verbs()` over it, so a new document's
    verbs reach the mission by adding a row, not a branch. Returns the line
    written or removed, "" for nothing to do; raises `ThoughtRefused`."""
    surface = registry.BY_VERB.get(verb)
    if surface is None or surface.unit != registry.CHARS:
      self._refuse(f"{verb!r} is not a verb on any thought file")
    if verb == surface.remove:
      return self.forget(surface.name, payload, by=ROBOT, t=t)
    if surface.name == FINDINGS:
      return self.record(payload, t=t)
    return self.append(surface.name, payload, by=ROBOT, t=t)

  def _commit(self, name: str, lines: list[str], t: float) -> None:
    self.texts[name] = "\n".join(lines)
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
    goes is what the ROBOT and the SYSTEM wrote -- `History.md`, its own
    unrevisable record of what happened to it, `Knowledge_and_Opinions.md`,
    everything it worked out, `Findings.md`, everything it measured, and
    since issue #154 `Goals.md`, everything it meant to do. Per
    Evaluation.md section 6 that is the cheapest real cost there is, and it
    is the whole of what makes a true death different from an ordinary one.

    ⚠ THE GOALS GO WITH IT, and that follows from the writer table rather
    than from a list here: they are the ROBOT's, and the next robot is a NEW
    ROBOT. Inheriting a dead predecessor's goals would be the one thing a
    true death is supposed to cost -- carrying on its work -- handed back for
    free.

    ⚠ THE CONSTITUTION SURVIVES, and this is not softness. `Main.md` says who
    a robot here is and what the person who looks after it hopes for it; a
    person put it on the volume by hand and there is no write API for it. A
    world that wiped it would need somebody to type it back in before it
    could run again -- and a new robot is a new robot, not a new species.

    ⚠ THE FILES ARE KEPT, not deleted: `History.1.md` beside the new empty
    one (`Store.archive`). A stake whose evidence is unlinked is a stake
    nobody can audit afterwards, and the volume is where the operator looks.
    """
    gone = [n for n in NAMES if SPECS[n].writer in (SYSTEM, ROBOT)]
    kept: dict[str, str] = {}
    for name in gone:
      kept[name] = self.texts[name]
      self.texts[name] = ""
      if self.root is not None:
        self.store.archive(name)
        self._write(name)
    return {"cleared": gone,
            "chars": {n: len(v) for n, v in kept.items()}}

  # ---- the wire -------------------------------------------------------------

  def message(self, name: str, t: float = 0.0) -> dict:
    """One document as a `thought` message (0.11.0): the whole text, not a
    delta, because a file is small and "present means complete" is the rule
    that keeps a late joiner and a scrubbed recording honest."""
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
    to three written and kept. The other files stay counters here -- they
    are on the wire in full as `thought` messages, and a run record that
    embedded every document would carry the constitution in every row.
    """
    return {"root": str(self.root) if self.root is not None else "",
            "chars": {n: len(self.texts[n]) for n in NAMES},
            "writes": dict(self.writes), "dropped": dict(self.dropped),
            "goals": self.texts[GOALS],
            "refusals": list(self.refusals[-5:])}
