"""The points ledger: what a robot has earned, and how (issue #14).

A balance per robot and an APPEND-ONLY log of how it got there -- one entry
per finished task, carrying the verdict, the points, and the moment. It is
world state, not run state: it lives in a JSON file next to the boards
(`--ledger state.json`, `$PLUGGY_LEDGER` in the deploy) and survives the
restart that ends every mission, because a scoreboard that resets whenever the
container cycles is not a scoreboard.

ONLY AN EVALUATOR CAN PAY (design doc, "Evaluation -- four tiers, and only
code awards points"). `award` takes a `scoring.Verdict` and nothing else:

  - a Verdict can only be built by `scoring.evaluate` (it carries a
    construction token that is cleared on the way out, so a `replace()` of a
    real one does not open the door either);
  - a dict, a namespace, or any other verdict-shaped object is a TypeError,
    not a duck;
  - and the points are RE-DERIVED here from the reward table and the
    evaluator's own metrics. A verdict whose numbers do not survive that
    recomputation is refused outright rather than trimmed to fit -- if the two
    disagree, something has gone wrong that a silent correction would hide.

That is three locks on one door, and the door is the reason the whole reward
system is worth building: an agent that can score its own work will learn to
declare victory rather than to do the task.

SPENDING is designed and deliberately not implemented (issue #14, "design now,
implement later"). When it lands it may buy COSMETIC and CAPABILITY unlocks
only -- LCD face styles, figures in the drawing library, access to a zone or a
tool -- and never anything the survival loop depends on. A robot that can spend
itself out of a charge is a robot that eventually will, and the resulting brick
is not an interesting failure. The `spent` counter and the balance arithmetic
are here so that ledgers written today stay readable when it does.

The telemetry surface is the same duck type an `ActivitySet`, a `BoardBook` and
a `ScreenSet` present (`names` + `snapshot()`), so the frame builder diffs it
with the one code path it already has -- and for the same reason those exist:
points are not a pose, so this block and the `earned` events are the only
record of them anywhere in the stream.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.hub.scoring import RewardTable, Verdict, default_table
from pluggybot.telemetry.protocol import ROBOT_ROOT

STATE_VERSION = 1

#: Entries kept per robot in the state file. The balance is exact forever --
#: it is a running total, not a re-sum of the log -- and this only bounds the
#: HISTORY, which is a display feature. `dropped` counts what aged out, so a
#: truncated log says so instead of pretending to be complete.
MAX_ENTRIES = 200
#: ...and how many of them ride in the telemetry block, where they are what
#: catches a late-joining browser up on the last few earnings without needing
#: a snapshot message of their own.
RECENT = 5


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Ledger:
  """Per-robot balances and their earnings log.

  `on_event` receives complete `earned` protocol messages as they happen --
  wire the live publisher and the recorder into it, exactly as for boards.
  """

  def __init__(self, robots=(ROBOT_ROOT,),
               path: str | os.PathLike | None = None,
               table: RewardTable | None = None,
               clock: Callable[[], str] = _now) -> None:
    self.table = table if table is not None else default_table()
    self.path = Path(path) if path is not None else None
    self.clock = clock
    self.on_event: list[Callable[[dict], None]] = []
    self.robots: dict[str, dict] = {
      name: {"balance": 0, "earned": 0, "spent": 0, "seq": 0, "dropped": 0,
             "entries": []}
      for name in robots
    }
    if self.path is not None and self.path.exists():
      self.load()

  # ---- reading -------------------------------------------------------------

  def _acct(self, robot: str) -> dict:
    if robot not in self.robots:
      # A robot the ledger was not opened for. Created rather than refused --
      # the shared world grows a second robot before this file learns its
      # name -- but it starts at zero, which is the only honest opening
      # balance.
      self.robots[robot] = {"balance": 0, "earned": 0, "spent": 0, "seq": 0,
                            "dropped": 0, "entries": []}
    return self.robots[robot]

  def balance(self, robot: str = ROBOT_ROOT) -> int:
    return self._acct(robot)["balance"]

  def entries(self, robot: str = ROBOT_ROOT) -> list[dict]:
    return list(self._acct(robot)["entries"])

  def pending(self, robot: str = ROBOT_ROOT) -> list[dict]:
    """Verdicts waiting on a visitor rating (issue #16)."""
    return [e for e in self._acct(robot)["entries"] if e.get("pending")]

  # ---- the telemetry surface (matches ActivitySet / BoardBook / ScreenSet) --

  @property
  def names(self) -> list[str]:
    return list(self.robots)

  def snapshot(self) -> dict:
    out = {}
    for name, acct in self.robots.items():
      recent = acct["entries"][-RECENT:]
      out[name] = {
        "balance": acct["balance"],
        "earned": acct["earned"],
        "spent": acct["spent"],
        "tasks": acct["seq"],
        "pending": sum(1 for e in acct["entries"] if e.get("pending")),
        # Compact on purpose: this rides in every keyframe, and the full
        # verdict is in the `earned` message that went out when it happened.
        "recent": [{"seq": e["seq"], "task": e["task"], "points": e["points"],
                    "ok": e["ok"], "t": e["t"]} for e in recent],
      }
    return out

  # ---- the one way in ------------------------------------------------------

  def award(self, verdict: Verdict, t: float = 0.0,
            robot: str = ROBOT_ROOT) -> dict:
    """Bank an evaluator's verdict. The ONLY method that moves a balance up.

    Returns the ledger entry (which is also the body of the `earned` message).
    A failed verdict is still recorded, at zero points: "it tried and did not
    manage it" is the most interesting line in a robot's log, and dropping it
    would make the ledger read as if the robot only ever succeeded.
    """
    if not isinstance(verdict, Verdict):
      raise TypeError(
        f"award() takes a scoring.Verdict, got {type(verdict).__name__} -- "
        "points are awarded by a deterministic evaluator, never by a task "
        "(or an LLM) reporting on itself (issue #14)")
    reward = self.table[verdict.task]
    if verdict.tier != reward.tier:
      raise ValueError(f"{verdict.task}: verdict tier {verdict.tier!r} is not "
                       f"the table's {reward.tier!r}")
    # Re-derive rather than trust: the table is the authority on what a task
    # pays, and the evaluator's own metrics are the authority on how well it
    # was done. A verdict that does not survive this was not produced by the
    # pair of them.
    quality = reward.quality(verdict.metrics)
    expected = 0 if verdict.pending else reward.points(verdict.ok, quality)
    if quality != verdict.quality or expected != verdict.points:
      raise ValueError(
        f"{verdict.task}: verdict claims {verdict.points} points at quality "
        f"{verdict.quality}, the reward table makes it {expected} at "
        f"{quality} -- refusing to bank a verdict that does not re-derive")
    return self._post(robot, verdict.task, verdict.tier, verdict.ok,
                      quality, verdict.points, verdict.reason,
                      verdict.public_metrics(), t, pending=verdict.pending)

  def settle(self, seq: int, quality: float, robot: str = ROBOT_ROOT,
             by: str = "visitor", t: float = 0.0) -> dict:
    """Pay a deferred (visitor-judged) verdict once its rating arrives.

    The slot the design doc asks for: an aesthetic call cannot be made by
    code, so the evaluator confirms the work HAPPENED and the rating settles
    what it was worth. Still code that pays: the rating supplies a 0..1
    quality, and the reward table -- not the rater, and not the robot --
    turns it into points.
    """
    acct = self._acct(robot)
    entry = next((e for e in acct["entries"] if e["seq"] == seq), None)
    if entry is None:
      raise KeyError(f"{robot}: no ledger entry {seq}")
    if not entry.get("pending"):
      raise ValueError(f"{robot}: entry {seq} ({entry['task']}) is not pending")
    if not 0.0 <= float(quality) <= 1.0:
      raise ValueError(f"a rating is a 0..1 quality, got {quality!r}")
    reward = self.table[entry["task"]]
    points = reward.points(entry["ok"], float(quality))
    entry.update({"pending": False, "settledBy": by, "quality": round(float(quality), 4),
                  "points": points, "settledAt": self.clock()})
    acct["balance"] += points
    acct["earned"] += points
    entry["balance"] = acct["balance"]
    # `t` is when the RATING landed, not when the work was done -- the entry
    # keeps the latter, and `seq` is what ties the two messages together. Note
    # the spread comes first here: with `**entry` last, the stored award time
    # would silently overwrite the settle time.
    self._emit({**entry, "type": "earned", "robot": robot,
                "t": round(float(t), 3), "settled": True})
    return entry

  def _post(self, robot: str, task: str, tier: str, ok: bool,
            quality: float | None, points: int, reason: str, metrics: dict,
            t: float, pending: bool = False) -> dict:
    acct = self._acct(robot)
    acct["seq"] += 1
    acct["balance"] += points
    acct["earned"] += points
    entry = {"seq": acct["seq"], "t": round(float(t), 3), "at": self.clock(),
             "task": task, "tier": tier, "ok": bool(ok), "points": int(points),
             "quality": quality, "reason": reason, "metrics": metrics,
             "pending": bool(pending), "balance": acct["balance"]}
    acct["entries"].append(entry)
    if len(acct["entries"]) > MAX_ENTRIES:
      acct["dropped"] += len(acct["entries"]) - MAX_ENTRIES
      del acct["entries"][:len(acct["entries"]) - MAX_ENTRIES]
    self._emit({"type": "earned", "robot": robot, **entry})
    return entry

  def _emit(self, msg: dict) -> None:
    for hook in self.on_event:
      hook(dict(msg))
    # Saved on every entry, like the boards and for the same reason: a
    # balance that survives only a CLEAN shutdown does not survive the thing
    # restarts are usually about. Entries arrive a handful of times per
    # mission, so the write is free.
    self.save()

  # ---- persistence ---------------------------------------------------------

  def save(self, path: str | os.PathLike | None = None) -> Path | None:
    target = Path(path) if path is not None else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": STATE_VERSION,
                               "robots": self.robots}, indent=1) + "\n")
    # Rename over the target: a crash mid-write leaves the previous ledger
    # intact rather than a truncated file that loads as a wiped balance.
    os.replace(tmp, target)
    return target

  def load(self, path: str | os.PathLike | None = None) -> "Ledger":
    """Restore balances and history. Older state versions load; newer ones
    refuse -- the same asymmetry as the board state file, and for the same
    reason: /var/lib/pluggybot outlives the image, so an upgrade must read
    what the previous version wrote, while a downgrade silently dropping
    fields it does not know is how a balance quietly loses points."""
    target = Path(path) if path is not None else self.path
    if target is None or not target.exists():
      return self
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if not 1 <= version <= STATE_VERSION:
      raise ValueError(
        f"{target}: ledger state version {doc.get('version')!r}, expected "
        f"1..{STATE_VERSION} -- delete the file to start the robots at zero")
    for name, acct in doc.get("robots", {}).items():
      entries = list(acct.get("entries", []))
      self.robots[name] = {
        "balance": int(acct.get("balance", 0)),
        "earned": int(acct.get("earned", 0)),
        "spent": int(acct.get("spent", 0)),
        # A log whose entries were trimmed still knows how many tasks there
        # were: `seq` is the counter, not the length.
        "seq": int(acct.get("seq", len(entries))),
        "dropped": int(acct.get("dropped", 0)),
        "entries": entries,
      }
    return self
