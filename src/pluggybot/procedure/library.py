"""The robot's library of procedures (issue #166): a directory beside the
thought files, one Python-shaped source per procedure, owned by the robot
on `Goals.md`'s terms -- a DOCUMENT in `mind/text.py`'s registry (issue
#217), whose row says the writer, the cap and the two verbs.

Two verbs, `define` and `undefine`, both decision FIELDS so writing one
costs no turn (as `pin` / `intend` do), and deliberately no verb that
REPLACES a procedure or the library: one bad generation must not be able to
rewrite everything the robot knows how to do. Redefining a name is refused;
the robot undefines it first, on purpose, in a decision of its own.

Every rule fails OUT LOUD -- a full library refuses (`text.admit`, the one
gate every document write passes), a source that does not compile refuses
with the parser's reasons, a name that does not match its `def` refuses --
because a library that silently dropped a procedure would leave the robot
believing it had one (the argument every ROBOT-written document shares).

What survives a restart: the sources. Each is recompiled against TODAY's
world when the library loads, and one that no longer validates is kept,
listed, and marked invalid rather than deleted -- the robot wrote it, and a
world that moved under it is a fact it should be shown.
"""

import os
import re
from dataclasses import dataclass

from pluggybot.mind import text as registry
from pluggybot.mind.store import FileStore, MemoryStore, Store
from pluggybot.mind.text import MAX_PROCEDURES
from pluggybot.procedure import lang
from pluggybot.procedure.steps import Refused, WorldFacts

__all__ = ["MAX_PROCEDURES", "SUFFIX", "Entry", "Library", "LibraryRefused"]

ROW = registry.BY_NAME["procedures"]
SUFFIX = ROW.suffix   # Python-SHAPED, not Python: not a name a linter or a person should run
#: The one refusal `procedure:new` needs (issue #264): the answer's own
#: define, never an entry -- at `define`, at `check` and at load.
NEW_REFUSED = ("'new' is not a name this library allows: "
               "`procedure:new` runs whatever the answer defines")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class LibraryRefused(ValueError):
  def __init__(self, reasons: list[str]) -> None:
    super().__init__("; ".join(reasons))
    self.reasons = list(reasons)


@dataclass
class Entry:
  name: str
  source: str
  procedure: lang.Procedure | None      # None when it no longer validates
  reasons: list[str]                    # why, when it does not

  @property
  def valid(self) -> bool:
    return self.procedure is not None


class Library:
  def __init__(self, facts: WorldFacts, root: str | os.PathLike | None = None,
               cap: int = MAX_PROCEDURES) -> None:
    self.facts = facts
    #: The one path to the disk (issue #217): a `Store`, the volume's or
    #: memory's. `cap` narrows the row's for one instance (a test's
    #: two-entry library) and never widens it -- `text.admit` reads both.
    self.store: Store = FileStore(root) if root is not None else MemoryStore()
    self.cap = min(cap, ROW.cap)
    self.entries: dict[str, Entry] = {}
    self.refusals: list[dict] = []
    self.defined = 0
    self.undefined = 0
    for key in self.store.keys(suffix=SUFFIX):
      name = key[:-len(SUFFIX)]
      if _NAME.match(name):
        self.entries[name] = self._entry(name, self.store.read(key) or "")

  @property
  def root(self):
    return self.store.root if isinstance(self.store, FileStore) else None

  def _entry(self, name: str, source: str) -> Entry:
    if name == "new":
      # saved before `procedure:new` existed (issue #264): kept and shown, not
      # runnable, so the token keeps its one meaning and the slot can be freed
      return Entry(name, source, None, [NEW_REFUSED + " -- undefine it and write "
                                        "it again under another name"])
    try:
      proc = lang.compile_procedure(source, self.facts)
    except Refused as e:
      return Entry(name, source, None, list(e.reasons))
    if proc.name != name:
      return Entry(name, source, None,
                   [f"the def is named {proc.name!r}, the entry {name!r}"])
    return Entry(name, source, proc, [])

  # ---- reading -------------------------------------------------------------

  def names(self) -> tuple[str, ...]:
    return tuple(self.entries)

  def runnable(self) -> tuple[str, ...]:
    return tuple(n for n, e in self.entries.items() if e.valid)

  def get(self, name: str) -> lang.Procedure | None:
    entry = self.entries.get(name)
    return entry.procedure if entry is not None else None

  def as_context(self) -> list[dict]:
    """What the robot is shown of its own library: every source, and why an
    entry is not runnable if it is not. (Which tools a procedure needed was
    said too while the quadruped carried its rack's, #324.)"""
    out = []
    for e in self.entries.values():
      row = {"name": e.name, "source": e.source, "runnable": e.valid}
      if e.reasons:
        row["reasons"] = e.reasons
      out.append(row)
    return out

  def stats(self) -> dict:
    return {"count": len(self.entries), "runnable": len(self.runnable()),
            "cap": self.cap, "defined": self.defined,
            "undefined": self.undefined, "refused": len(self.refusals)}

  # ---- the two verbs -------------------------------------------------------

  def define(self, name: str, source: str, t: float = 0.0) -> lang.Procedure:
    """Add a procedure, or raise LibraryRefused with every reason."""
    reasons: list[str] = []
    name = str(name or "").strip()
    if not _NAME.match(name):
      reasons.append(f"{name!r} is not a name this library allows")
    elif name == "new":
      # `procedure:new` is the answer's own define (issue #264), never an entry
      reasons.append(NEW_REFUSED)
    if name in self.entries:
      # ⚠ ON THE SAME ANSWER (issue #264): the lifecycle applies `undefine`
      # before `define`, so a replacement is one answer -- and "undefine it
      # first" read as a turn of its own to every deployed robot, which
      # wrote `stack3b` and `weigh_cube2` beside the originals instead.
      reasons.append(f"{name!r} is already defined -- to replace it, put "
                     f"{name!r} in `undefine` on the same answer as this "
                     "define: the undefine is done first")
    try:
      registry.admit(ROW, registry.ROBOT, len(self.entries) + 1, cap=self.cap)
    except registry.Refused:
      reasons.append(f"the library is full (it holds {self.cap}) -- an "
                     "`undefine` on the same answer as this define makes room "
                     "first")
    proc = None
    if not reasons:
      try:
        proc = lang.compile_procedure(source, self.facts)
      except Refused as e:
        reasons.extend(e.reasons)
      else:
        if proc.name != name:
          reasons.append(f"the def is named {proc.name!r}, not {name!r}")
    if reasons or proc is None:
      self.refusals.append({"t": t, "verb": "define", "name": name,
                            "reasons": reasons})
      raise LibraryRefused(reasons)
    self.entries[name] = Entry(name, source, proc, [])
    self.defined += 1
    self.store.write(f"{name}{SUFFIX}", source)
    return proc

  def admit(self, name: str, source: str, verdict, t: float = 0.0) -> lang.Procedure:
    """VERIFIED ADMISSION (issue #493, #487's library): `define`, but only
    on a verdict its grader passed -- a `scoring.Verdict`, which only
    `scoring.evaluate` can build, `ok` -- the attempt that ran `source`
    (`attempts.attempt(sealed=)`). Else LibraryRefused, said and kept as
    `define`'s refusals are."""
    from pluggybot.economy.scoring import Verdict
    if not isinstance(verdict, Verdict) or not verdict.ok:
      why = ("its grader did not pass it" if isinstance(verdict, Verdict)
             else "no grader's verdict came with it")
      reasons = [f"{name!r} is admitted only once its grader has passed it: {why}"]
      self.refusals.append({"t": t, "verb": "admit", "name": name, "reasons": reasons})
      raise LibraryRefused(reasons)
    return self.define(name, source, t)

  def undefine(self, name: str, t: float = 0.0) -> None:
    name = str(name or "").strip()
    if name not in self.entries:
      self.refusals.append({"t": t, "verb": "undefine", "name": name,
                            "reasons": [f"no procedure named {name!r}"]})
      raise LibraryRefused([f"no procedure named {name!r}"])
    del self.entries[name]
    self.undefined += 1
    self.store.remove(f"{name}{SUFFIX}")

  def check(self, name: str, source: str, freeing: str = "") -> list[str]:
    """What `define` would refuse this source for once `freeing` -- an entry
    the same answer undefines -- is gone: every reason `define` gives, in its
    words (issue #264). Changes nothing: the lifecycle asks BEFORE that
    undefine, because in order a refused define had already deleted the
    procedure it was meant to improve or to make room for. There is still no
    replace: `define` and `undefine` are the library's only two changes."""
    name = str(name or "").strip()
    freeing = str(freeing or "").strip()
    held = [e for e in self.entries if e != freeing]
    reasons: list[str] = []
    if not _NAME.match(name):
      reasons.append(f"{name!r} is not a name this library allows")
    elif name == "new":
      reasons.append(NEW_REFUSED)
    if name in held:
      reasons.append(f"{name!r} is already defined -- to replace it, put "
                     f"{name!r} in `undefine` on the same answer as this "
                     "define: the undefine is done first")
    try:
      registry.admit(ROW, registry.ROBOT, len(held) + 1, cap=self.cap)
    except registry.Refused:
      reasons.append(f"the library is full (it holds {self.cap}) -- an "
                     "`undefine` on the same answer as this define makes room "
                     "first")
    if reasons:
      return reasons
    try:
      proc = lang.compile_procedure(source, self.facts)
    except Refused as e:
      return list(e.reasons)
    return [] if proc.name == name else [f"the def is named {proc.name!r}, not {name!r}"]
