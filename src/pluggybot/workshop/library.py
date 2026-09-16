"""The robot's tools (issue #168, slice D): what it built, hanging where,
on `procedure/library.py`'s terms -- a DOCUMENT in `mind/text.py`'s
registry (issue #217), whose row says the writer, the cap and the verbs.

A directory beside the thought files (`$PLUGGY_THOUGHTS/tools/`), one
JSON record per built tool: the spec as the robot wrote it, the bay it
hangs in, what it cost, when. Two verbs, both decision FIELDS so writing
one costs no turn -- `build_tool` and `retire_tool` -- and no verb that
replaces: a bay is taken by naming it, and a built tool in it is retired
first, on purpose, in the same decision (`bay` names the bay, and the
module there goes, hand-built or not).

Every rule fails OUT LOUD: a spec that does not validate, a price the
balance cannot cover, a bay that does not exist, a name already hung, a
full record (`text.admit`, the one gate every document write passes) --
narrated, counted, on the wire as a `tool` event with its reasons. A
workshop that silently dropped a build would leave the robot believing
it had a tool it does not (the argument every ROBOT-written document
shares).

What survives a restart: the records. Each is re-validated against
today's catalog and world when the workshop loads and, if it still
validates, HUNG AGAIN at its bay at the start of the day -- a world file
knows nothing of built tools, so a restart recompiles them in. One that no
longer validates is kept, marked, and shown; the robot wrote it, and a
catalog that moved under it is a fact it should be shown. The points were
paid once and are not paid again.

  ⚠ ONE TOOL PER BAY, FIVE BAYS. The cap is the rack's, not a number
  chosen here (ToolPattern.md §6; the registry row reads it off
  `HUB_STATION_YS`): a sixth needs the rail to grow.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from pluggybot.mind import text as registry
from pluggybot.mind.store import FileStore, MemoryStore, Store
from pluggybot.workshop import validate
from pluggybot.workshop.spec import Refused, Tool

ROW = registry.BY_NAME["tools"]
SUFFIX = ROW.suffix
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
BAYS = tuple(chr(ord("A") + i) for i in range(ROW.cap))


class WorkshopRefused(ValueError):
  def __init__(self, reasons: list[str]) -> None:
    super().__init__("; ".join(reasons))
    self.reasons = list(reasons)


def bay_index(letter: str) -> int:
  """'A'..'E' -> 0..4, or WorkshopRefused."""
  letter = str(letter or "").strip().upper()
  if letter not in BAYS:
    raise WorkshopRefused([f"bay {letter!r} is not one of {', '.join(BAYS)}"])
  return BAYS.index(letter)


@dataclass
class Entry:
  name: str
  spec: dict
  bay: int
  cost: dict
  t: float
  tool: Tool | None                 # None when it no longer validates
  reasons: list[str] = field(default_factory=list)

  @property
  def valid(self) -> bool:
    return self.tool is not None

  def as_record(self) -> dict:
    return {"name": self.name, "spec": self.spec, "bay": self.bay,
            "cost": self.cost, "t": self.t}


class Workshop:
  def __init__(self, root: str | os.PathLike | None = None) -> None:
    #: The one path to the disk (issue #217): a `Store`, the volume's or
    #: memory's.
    self.store: Store = FileStore(root) if root is not None else MemoryStore()
    self.entries: dict[str, Entry] = {}
    self.built = 0
    self.retired = 0
    self.refusals: list[dict] = []
    self.spent = 0
    self._load()

  @property
  def root(self):
    return self.store.root if isinstance(self.store, FileStore) else None

  def _load(self) -> None:
    for key in self.store.keys(suffix=SUFFIX):
      try:
        rec = json.loads(self.store.read(key) or "")
      except ValueError:
        continue
      name = str(rec.get("name", ""))
      tool, reasons = None, []
      try:
        tool = validate.check(rec.get("spec"))
        if tool.name != name:
          tool, reasons = None, [f"the spec is named {tool.name!r}, not {name!r}"]
      except Refused as e:
        reasons = list(e.reasons)
      self.entries[name] = Entry(name, rec.get("spec"), int(rec.get("bay", 0)),
                                 dict(rec.get("cost", {})), float(rec.get("t", 0.0)),
                                 tool, reasons)

  # ---- what the robot is shown ---------------------------------------------

  def names(self) -> tuple[str, ...]:
    return tuple(self.entries)

  def hung(self) -> list[Entry]:
    return [e for e in self.entries.values() if e.valid]

  def as_context(self) -> list[dict]:
    """Every tool the robot built: its spec, its bay, what it cost, and why
    it is not on the rack if it is not."""
    return [{"name": e.name, "bay": BAYS[e.bay], "spec": e.spec,
             "costPoints": e.cost.get("points"), "hung": e.valid,
             **({"reasons": e.reasons} if e.reasons else {})}
            for e in self.entries.values()]

  def stats(self) -> dict:
    return {"count": len(self.entries), "hung": len(self.hung()),
            "built": self.built, "retired": self.retired,
            "refused": len(self.refusals), "spentPoints": self.spent}

  # ---- the two verbs -------------------------------------------------------

  def check(self, name: str, raw_spec, bay: str) -> tuple[Tool, int]:
    """Everything that can be refused BEFORE any points move: the name, the
    bay, the spec against the envelope. Returns the tool and the bay index."""
    reasons: list[str] = []
    name = str(name or "").strip()
    if not _NAME.match(name):
      reasons.append(f"{name!r} is not a name the workshop allows")
    if name in self.entries:
      reasons.append(f"{name!r} is already built -- retire it first, there is no replace")
    try:
      registry.admit(ROW, registry.ROBOT, len(self.entries) + 1)
    except registry.Refused as e:
      reasons.append(f"the workshop {str(e).removeprefix(ROW.name + ' ')}")
    idx = -1
    try:
      idx = bay_index(bay)
    except WorkshopRefused as e:
      reasons.extend(e.reasons)
    tool = None
    try:
      tool = validate.check(raw_spec)
    except Refused as e:
      reasons.extend(e.reasons)
    if tool is not None and tool.name != name:
      reasons.append(f"the spec is named {tool.name!r}, not {name!r}")
    if reasons or tool is None:
      raise WorkshopRefused(reasons)
    return tool, idx

  def record(self, tool: Tool, raw_spec, bay: int, cost: dict, t: float) -> Entry:
    """A built, hung tool into the records (after the points are paid and
    the module hangs)."""
    entry = Entry(tool.name, raw_spec, bay, dict(cost), t, tool, [])
    self.entries[tool.name] = entry
    self.built += 1
    self.spent += int(cost.get("points", 0))
    self.store.write(f"{tool.name}{SUFFIX}",
                     json.dumps(entry.as_record(), indent=1) + "\n")
    return entry

  def refuse(self, name: str, reasons: list[str], t: float) -> None:
    self.refusals.append({"t": t, "name": name, "reasons": list(reasons)})

  def retire(self, name: str) -> Entry:
    name = str(name or "").strip()
    entry = self.entries.get(name)
    if entry is None:
      raise WorkshopRefused([f"no built tool {name!r} to retire"])
    del self.entries[name]
    self.retired += 1
    self.store.remove(f"{name}{SUFFIX}")
    return entry
