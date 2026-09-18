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
    entry is not runnable if it is not."""
    return [{"name": e.name, "source": e.source, "runnable": e.valid,
             **({"reasons": e.reasons} if e.reasons else {})}
            for e in self.entries.values()]

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
    if name in self.entries:
      reasons.append(f"{name!r} is already defined -- undefine it first, "
                     "there is no replace")
    try:
      registry.admit(ROW, registry.ROBOT, len(self.entries) + 1, cap=self.cap)
    except registry.Refused as e:
      reasons.append(f"the library {str(e).removeprefix(ROW.name + ' ')}")
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

  def undefine(self, name: str, t: float = 0.0) -> None:
    name = str(name or "").strip()
    if name not in self.entries:
      self.refusals.append({"t": t, "verb": "undefine", "name": name,
                            "reasons": [f"no procedure named {name!r}"]})
      raise LibraryRefused([f"no procedure named {name!r}"])
    del self.entries[name]
    self.undefined += 1
    self.store.remove(f"{name}{SUFFIX}")
