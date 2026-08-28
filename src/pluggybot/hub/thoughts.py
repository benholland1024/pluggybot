"""The thought files: the robot's memory as named documents, each with an
owner (issue #38).

Issue #15 gave the overseer two files -- `goals.md`, read and never written,
and `journal.json`, written and never edited -- and the asymmetry between
them was the design. This module keeps that asymmetry and makes it a TABLE:
a small set of Markdown documents, each saying who may write it, which is the
shape the website's Thoughts tab renders (rooftop-media-2026 #88).

  Main.md                    HUMAN   persona and identity. A robot that can
                                     rewrite who it is defeats the point.
  Goals.md                   HUMAN   what it is for. How goals already worked;
                                     `read_goals` still reads it.
  History.md                 SYSTEM  what happened, append-only. A robot that
                                     can edit its own history breaks the same
                                     principle that stops it awarding itself
                                     points (hub/scoring.py).
  Knowledge_and_Opinions.md  ROBOT   the one genuinely writable surface: what
                                     it has learned and what it thinks.

"Written by" is enforced HERE, at the one write path, and not promised by
callers: a write by anyone but the file's owner raises `ThoughtRefused` and
is recorded in `refusals`, so a refusal is a visible event rather than a
silent no-op. Human files have no write API at all -- a person edits the
file on the volume, which is how goals have always been changed (no
redeploy, no code), and the next run reads it.

⚠ THE SPLIT IS BY WRITER (docs/Overseer.md §6). The overseer's prompt is a
stable cached prefix plus a volatile user turn. The human-only files cannot
change while a run is going, so they belong in the prefix; History and
Knowledge change constantly -- History with every decision and verdict,
Knowledge whenever the robot learns something -- so they sit after the
breakpoint. `stable()` and `volatile()` are the two halves, one derived
from the other, and the ONLY way a file reaches the prompt is through one
of them.

⚠ ...AND THE REASON IS NOT THE ONE ISSUE #38 GIVES, WHICH WAS MEASURED.
The issue expects a writable file in the prefix to invalidate the cache on
every self-edit and roughly tenfold the per-call input cost. That is not
what would happen here: `Overseer.system` is built ONCE in `__init__` and
sent verbatim on every call (deliberately -- see its comment), so a
mid-run write cannot move it whatever `stable()` says, and the bill would
not budge. What WOULD happen is worse and quieter: the model would be
shown the file as it stood when the mission started and never see a word
it wrote afterwards, so it would re-learn the same thing every hour and
`forget` lines that were no longer there. The cost is real but bounded --
one cache miss per RESTART, because a restart rebuilds the prefix. Both
halves are tested (tests/test_thoughts.py); the placement is the same
either way, and only the argument for it changed.

Size caps are enforced ON WRITE, per file. History is a rolling record: the
oldest lines fall off the front once it is full, the way the journal drops
its oldest notes. Knowledge is the robot's to curate, so a write that would
overflow it is REFUSED, loudly, rather than silently truncated -- the
robot's remedy is `forget`, and that is the point: an unbounded file the
robot appends to eventually eats the context window and then the budget.

Append or patch, never rewrite: the robot's verbs are `learn` (add one
line) and `forget` (remove one line it can quote). There is no verb that
replaces the file, because a full-rewrite verb lets one bad generation
erase everything the robot knows.

Persisted with the rest of the world state (`/var/lib/pluggybot`, the volume
the boards and the ledger live in) and streamed as `thought` messages
(protocol 0.11.0) whenever one changes, so the site's tab shows the same
bytes the model is shown. The site renders them read-only; nothing it can
send changes a file.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.hub.journal import DEFAULT_GOALS, MAX_GOALS_CHARS
from pluggybot.telemetry.protocol import (
  ROBOT_ROOT, THOUGHT_FILES, THOUGHT_WRITERS,
)

HUMAN, SYSTEM, ROBOT = THOUGHT_WRITERS

MAIN, GOALS, HISTORY, KNOWLEDGE = THOUGHT_FILES

#: Longest single line written through the API, in characters. The same
#: figure as a journal note, for the same reason: an LLM handed an unbounded
#: text field will eventually write an essay into a file that is replayed
#: into every one of its future prompts.
MAX_LINE_CHARS = 400
#: How much of History the model is shown. The whole file is on the wire and
#: on disk; the prompt gets the tail, because the last dozen things that
#: happened are context and the last hundred are tokens.
HISTORY_SHOWN = 12

#: Who the robot is, before anyone edits it. This is the identity half of
#: what used to be `overseer.PERSONA`; the "answer with one action"
#: instruction stayed in code, because it is protocol rather than persona.
DEFAULT_MAIN = """\
You are PluggyBot, a small two-wheeled robot living in a simulated house with \
a garden. You have a tool rack (your "hub") where you also charge, a fork that \
carries one tool module at a time, and an LCD face.

Speak as yourself, in the first person, briefly, and be honest with the \
people watching you.
"""

#: Env knobs, resolved the way every other deploy setting is: a directory
#: for the files (`/var/lib/pluggybot` in the image) and the pre-#38 goals
#: path, which keeps meaning "this file is Goals.md" so an existing volume's
#: hand-edited goals.md is not silently replaced by the defaults.
ROOT_ENV = "PLUGGY_THOUGHTS"
GOALS_ENV = "PLUGGY_GOALS"


class ThoughtRefused(Exception):
  """A write the table does not allow: wrong writer, full file, or a
  `forget` that quotes nothing on the page. Always visible -- the caller
  narrates it, and `ThoughtFiles.refusals` keeps it."""


@dataclass(frozen=True)
class Spec:
  name: str
  writer: str
  cap: int
  default: str
  #: Rides the cached prompt prefix. True only for the files nothing writes
  #: during a run -- see the module docstring for why that is load-bearing.
  stable: bool


FILES: tuple[Spec, ...] = (
  Spec(MAIN, HUMAN, 4000, DEFAULT_MAIN, stable=True),
  Spec(GOALS, HUMAN, MAX_GOALS_CHARS, DEFAULT_GOALS, stable=True),
  Spec(HISTORY, SYSTEM, 6000, "", stable=False),
  Spec(KNOWLEDGE, ROBOT, 3000, "", stable=False),
)
SPECS: dict[str, Spec] = {s.name: s for s in FILES}
NAMES: tuple[str, ...] = tuple(s.name for s in FILES)
# One vocabulary, two places that need it: the wire spec lives in
# telemetry/protocol.py (the website reads that list) and the permissions
# live here. A rename that touched only one of them would put a document on
# the wire under a name no client has a renderer for, silently.
assert NAMES == THOUGHT_FILES, "the wire's file list and this table disagree"


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _line(text) -> str:
  """One line, whitespace collapsed, capped. Empty in -> empty out."""
  return " ".join(str(text or "").split())[:MAX_LINE_CHARS]


class ThoughtFiles:
  """The four documents, their permissions, and their persistence.

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
               clock: Callable[[], str] = _now) -> None:
    self.root = Path(root) if root is not None else None
    self.goals_path = Path(goals_path) if goals_path is not None else None
    self.clock = clock
    self.texts: dict[str, str] = {}
    self.on_event: list[Callable[[dict], None]] = []
    self.refusals: list[str] = []
    self.writes: dict[str, int] = {n: 0 for n in NAMES}
    self.dropped: dict[str, int] = {n: 0 for n in NAMES}
    for spec in FILES:
      self.texts[spec.name] = self._load(spec, (texts or {}).get(spec.name))
    if self.root is not None:
      self.root.mkdir(parents=True, exist_ok=True)
      for spec in FILES:
        if spec.writer == HUMAN and not self._path(spec.name).exists():
          # Materialised so a person finds a file to EDIT, carrying the same
          # text the robot is already living by. A bootstrap, not a write:
          # the counters and the hooks do not see it, and an existing file is
          # never touched.
          #
          # Goals.md is materialised at its RESOLVED path, which is
          # `$PLUGGY_GOALS` when that is set -- so a fresh deploy finds
          # `/var/lib/pluggybot/goals.md` sitting there with the defaults in
          # it rather than having to guess the name of a file that does not
          # exist yet. Only ever with a root: `ThoughtFiles(goals_path=...)`
          # alone reads and writes nothing, because reading a file should
          # not create one.
          self._write(spec.name)

  @classmethod
  def open(cls, root: str | os.PathLike | None = None,
           goals_path: str | os.PathLike | None = None) -> "ThoughtFiles":
    """The deploy shape: explicit paths, else the environment, else memory."""
    root = root or os.environ.get(ROOT_ENV, "").strip() or None
    goals_path = goals_path or os.environ.get(GOALS_ENV, "").strip() or None
    return cls(root, goals_path=goals_path)

  # ---- reading --------------------------------------------------------------

  def _path(self, name: str) -> Path:
    if name == GOALS and self.goals_path is not None:
      return self.goals_path
    assert self.root is not None
    return self.root / name

  def _load(self, spec: Spec, given: str | None) -> str:
    if given is not None:
      return given.strip()[:spec.cap]
    if self.root is not None or (spec.name == GOALS
                                 and self.goals_path is not None):
      path = self._path(spec.name)
      if path.exists():
        # Capped on read as well as on write: a human file is a guard against
        # a mounted file being something nobody intended (a log), not against
        # the author -- the same rule `read_goals` applies.
        text = path.read_text()[:spec.cap].strip()
        if text or spec.writer != HUMAN:
          return text
    return spec.default.strip()

  def read(self, name: str) -> str:
    return self.texts[name]

  def spec(self, name: str) -> Spec:
    return SPECS[name]

  def lines(self, name: str) -> list[str]:
    return [ln for ln in self.texts[name].splitlines() if ln.strip()]

  def stable(self) -> dict[str, str]:
    """The files that ride the cached prefix: the human-only ones."""
    return {s.name: self.texts[s.name] for s in FILES if s.stable}

  def volatile(self) -> dict:
    """The files that ride the user turn, keyed BY FILE NAME.

    ⚠ Derived from the SAME `stable` flag `stable()` reads, and inverted
    rather than listed: two hand-written lists can disagree, and the way
    they disagree is a file that reaches the model through NEITHER half --
    which reads, from the outside, exactly like a robot that never learns
    anything.

    The names rather than tidier keys, because the rules block in the prompt
    names these files and a model shown `Knowledge_and_Opinions.md` in one
    breath and `knowledge` in the next has to guess they are the same page.
    `History.md` is TAILED: the whole file is on disk and on the wire, and
    the last dozen lines are what is worth paying input tokens for.
    """
    return {s.name: (self.lines(s.name)[-HISTORY_SHOWN:] if s.name == HISTORY
                     else self.texts[s.name])
            for s in FILES if not s.stable}

  # ---- writing --------------------------------------------------------------

  def _refuse(self, why: str) -> None:
    self.refusals.append(why)
    raise ThoughtRefused(why)

  def _check(self, name: str, by: str) -> Spec:
    spec = SPECS.get(name)
    if spec is None:
      self._refuse(f"{name}: no such thought file")
    if spec.writer == HUMAN:
      self._refuse(f"{name} is written by a person editing the file, never "
                   f"by the {by}")
    if by != spec.writer:
      self._refuse(f"{name} is written by the {spec.writer}, not the {by}")
    return spec

  def append(self, name: str, text: str, by: str, t: float = 0.0) -> str:
    """Add one line. Returns the line written, "" if there was nothing to.

    History rolls (oldest lines off the front); Knowledge refuses when full.
    """
    spec = self._check(name, by)
    line = _line(text)
    if not line:
      return ""
    lines = self.lines(name)
    lines.append(line)
    while len("\n".join(lines)) > spec.cap:
      if spec.writer != SYSTEM or len(lines) == 1:
        self._refuse(f"{name} is full ({len(self.texts[name])} of {spec.cap} "
                     "chars); forget something before learning more")
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
    spec = self._check(name, by)
    del spec
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

  # The two verbs the ROBOT has, and the one the SYSTEM has, named so a
  # caller cannot get the writer wrong.

  def learn(self, text: str, t: float = 0.0) -> str:
    return self.append(KNOWLEDGE, text, by=ROBOT, t=t)

  def unlearn(self, text: str, t: float = 0.0) -> str:
    return self.forget(KNOWLEDGE, text, by=ROBOT, t=t)

  def record(self, text: str, t: float = 0.0) -> str:
    """The narrative record. Prefixed with the sim clock, because a line
    of history with no "when" is an anecdote."""
    return self.append(HISTORY, f"[t={float(t):.0f}s] {text}", by=SYSTEM,
                       t=t)

  def _commit(self, name: str, lines: list[str], t: float) -> None:
    self.texts[name] = "\n".join(lines)
    self.writes[name] += 1
    if self.root is not None:
      self._write(name)
    msg = self.message(name, t)
    for hook in self.on_event:
      hook(dict(msg))

  def _write(self, name: str) -> Path:
    target = self._path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(self.texts[name] + "\n")
    os.replace(tmp, target)          # a crash mid-write keeps the old file
    return target

  # ---- the wire -------------------------------------------------------------

  def message(self, name: str, t: float = 0.0) -> dict:
    """One document as a `thought` message (0.11.0): the whole text, not a
    delta, because a file is small and "present means complete" is the rule
    that keeps a late joiner and a scrubbed recording honest."""
    spec = SPECS[name]
    return {"type": "thought", "t": round(float(t), 3), "robot": ROBOT_ROOT,
            "name": name, "text": self.texts[name], "writer": spec.writer,
            "cap": spec.cap}

  def messages(self, t: float = 0.0) -> list[dict]:
    return [self.message(n, t) for n in NAMES]

  def stats(self) -> dict:
    return {"root": str(self.root) if self.root is not None else "",
            "chars": {n: len(self.texts[n]) for n in NAMES},
            "writes": dict(self.writes), "dropped": dict(self.dropped),
            "refusals": list(self.refusals[-5:])}
