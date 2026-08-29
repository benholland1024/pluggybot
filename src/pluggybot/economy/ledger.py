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

CONSUMPTION is not spending, and the two counters are separate for that reason
(issue #36). `consume` is METABOLISM -- points eaten by the passage of sim
time, driven by `economy/metabolism.py`, chosen by nobody -- while `spent` is
still reserved for a purchase the robot decides on. Rolling them together
would make "what has this robot bought" unanswerable the day the first
purchase lands. Both come off the balance; only one of them will ever be the
robot's idea.

THE CAP is the other half of the same issue, and it lives here because this is
the only code that banks anything. Above `cap` an award is not banked -- and
`spilled` is what says so, on the entry and on the account. Silently paying
less than `economy/rewards.json` promised would undo the reward table's whole
design: a robot cannot check its own arithmetic, so a number that quietly
disagrees with the published one is indistinguishable to it from a bug. `cap`
is None everywhere the mechanic is off, which is every existing mission,
recording and test.

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

from pluggybot.economy.scoring import RewardTable, Verdict, default_table
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: Bumped to 2 by issue #36: an account gained `consumed`, `spilled` and
#: `owed`. A v1 file loads (the fields default to zero, which is what a
#: ledger written before hunger existed honestly means); a v1 BUILD refuses a
#: v2 file, which is the asymmetry `load` explains and the reason this is not
#: just three more `.get` calls -- an old image reading a new file would drop
#: the fraction of a point owed and re-round it in the robot's favour on
#: every restart.
STATE_VERSION = 2

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


def _account() -> dict:
  """A robot's opening account. One factory so the constructor, the
  grown-a-second-robot path and `load` cannot disagree about the fields --
  they did not, and then issue #36 added three at once."""
  return {"balance": 0, "earned": 0, "spent": 0, "seq": 0, "dropped": 0,
          # Points eaten by living (issue #36), kept apart from `spent` for
          # the reason the module docstring gives: metabolism is not a
          # purchase. `spilled` is what the cap refused, and `owed` the
          # fraction of a point the appetite is carrying -- persisted so a
          # restart resumes mid-point instead of rounding a free meal.
          "consumed": 0, "spilled": 0, "owed": 0.0,
          "entries": []}


class Ledger:
  """Per-robot balances and their earnings log.

  `on_event` receives complete `earned` protocol messages as they happen --
  wire the live publisher and the recorder into it, exactly as for boards.
  """

  def __init__(self, robots=(ROBOT_ROOT,),
               path: str | os.PathLike | None = None,
               table: RewardTable | None = None,
               cap: int | None = None,
               clock: Callable[[], str] = _now) -> None:
    self.table = table if table is not None else default_table()
    self.path = Path(path) if path is not None else None
    #: The most a balance may hold (issue #36), or None for the unbounded
    #: accumulation every run before the metabolism had. Supplied by whoever
    #: read `economy/metabolism.json`, so this class stays the thing that
    #: BANKS and never the thing that decides the policy -- the same split as
    #: `table`.
    self.cap = int(cap) if cap is not None else None
    self.clock = clock
    self.on_event: list[Callable[[dict], None]] = []
    self.robots: dict[str, dict] = {name: _account() for name in robots}
    if self.path is not None and self.path.exists():
      self.load()

  # ---- reading -------------------------------------------------------------

  def _acct(self, robot: str) -> dict:
    if robot not in self.robots:
      # A robot the ledger was not opened for. Created rather than refused --
      # the shared world grows a second robot before this file learns its
      # name -- but it starts at zero, which is the only honest opening
      # balance.
      self.robots[robot] = _account()
    return self.robots[robot]

  def balance(self, robot: str = ROBOT_ROOT) -> int:
    return self._acct(robot)["balance"]

  def consumed(self, robot: str = ROBOT_ROOT) -> int:
    """Points eaten by living, since this file was opened (issue #36)."""
    return self._acct(robot)["consumed"]

  def spilled(self, robot: str = ROBOT_ROOT) -> int:
    """Points the CAP refused. The receipt for every award that came in over
    the ceiling -- without it the robot would simply see the reward table
    paying less than it says, which is the one thing a scoreboard may not
    do quietly."""
    return self._acct(robot)["spilled"]

  def owed(self, robot: str = ROBOT_ROOT) -> float:
    """The fraction of a point the appetite is carrying. Stored here rather
    than in `Metabolism` because it has to survive a restart, and this is
    the file that already does."""
    return float(self._acct(robot)["owed"])

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
        # What living has cost and what the cap refused (0.13.0, issue #36).
        # Here as well as in the `metabolism` block because they are the
        # BALANCE's arithmetic -- earned - consumed - spent is what is left,
        # and a site showing the balance without them cannot explain it.
        # Zero on every world with no appetite attached, which is honest:
        # nothing has been eaten and nothing refused.
        "consumed": acct["consumed"],
        "spilled": acct["spilled"],
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
    # Through the cap like any other award (issue #36). A rating that arrives
    # while the robot is full is worth exactly what the work was worth and
    # exactly as much of it fits -- a deferred payout is not a way around a
    # ceiling.
    banked, over = self._to_balance(acct, points)
    if self.cap is not None:
      entry.update({"banked": banked, "spilled": over})
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
    banked, over = self._to_balance(acct, int(points))
    entry = {"seq": acct["seq"], "t": round(float(t), 3), "at": self.clock(),
             "task": task, "tier": tier, "ok": bool(ok),
             # ⚠ WHAT THE JOB PAID, NOT WHAT FIT. `points` stays the reward
             # table's answer whatever the cap did, because that is the
             # number `award` above re-derived and the number the robot was
             # promised. `banked` and `spilled` are how much of it reached
             # the balance -- present only where a cap can actually refuse
             # something, so every capless run's entries are byte-identical
             # to the ones it wrote before this existed.
             "points": int(points),
             **({"banked": banked, "spilled": over}
                if self.cap is not None else {}),
             "quality": quality, "reason": reason, "metrics": metrics,
             "pending": bool(pending), "balance": acct["balance"]}
    acct["entries"].append(entry)
    if len(acct["entries"]) > MAX_ENTRIES:
      acct["dropped"] += len(acct["entries"]) - MAX_ENTRIES
      del acct["entries"][:len(acct["entries"]) - MAX_ENTRIES]
    self._emit({"type": "earned", "robot": robot, **entry})
    return entry

  def _to_balance(self, acct: dict, points: int) -> tuple[int, int]:
    """Bank `points`, refusing whatever the cap will not hold.

    `earned` follows the BALANCE rather than the reward table, so
    earned - consumed - spent == balance stays an identity a reader can
    check. What the table paid over the ceiling is in `spilled`, and the two
    together reconstruct the gross.
    """
    banked = (points if self.cap is None
              else max(0, min(points, self.cap - acct["balance"])))
    over = points - banked
    acct["balance"] += banked
    acct["earned"] += banked
    acct["spilled"] += over
    return banked, over

  # ---- the one way down (issue #36) ----------------------------------------

  def carry(self, owed: float, robot: str = ROBOT_ROOT) -> None:
    """Record the fraction of a point the appetite has not charged yet.

    IN MEMORY ONLY, and deliberately: this moves every tick, and saving a
    file a second on the physics thread to persist a hundredth of a point
    would be paying for the accuracy in the wrong currency. Any later
    `save()` -- an award, a consume -- writes whatever the carry is by then,
    so the most a crash can round in the robot's favour is the fraction
    accumulated since the last write, which is under one point by
    construction.
    """
    self._acct(robot)["owed"] = round(float(owed), 6)

  def consume(self, points: int, t: float = 0.0, robot: str = ROBOT_ROOT,
              owed: float | None = None) -> int:
    """Eat `points`. The ONLY method that moves a balance down.

    Called by `economy/metabolism.py` off the physics seam, never by anything
    the robot can influence -- which is `award`'s rule inverted, and the same
    reason: a robot that could decline to be hungry would.

    ⚠ THE FLOOR IS ZERO, AND THERE IS NO DEBT. A starving robot that earns
    five points has five points, not five minus however long it went without
    -- arrears would make the first job after a bad night pay nothing, which
    is the discouragement gradient at exactly the wrong moment. Hunger is a
    state to be shown, never a hole to be climbed out of (issue #36, "zero is
    narrative, never a capability lock").

    No event and no `_emit`: consumption is CONTINUOUS, and one message per
    point would bury the awards it sits between. The `metabolism` block in
    every frame is where a site reads it. The save is here because the write
    is what makes hunger survive a restart, and it lands at the appetite's
    rate -- a handful of times a sim-hour, exactly like an award.
    """
    acct = self._acct(robot)
    eaten = max(0, min(int(points), acct["balance"]))
    acct["balance"] -= eaten
    acct["consumed"] += eaten
    if owed is not None:
      self.carry(owed, robot)
    self.save()
    return eaten

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
        # Absent in a v1 file, and zero is what one honestly means: it was
        # written by a build in which nothing was ever eaten.
        "consumed": int(acct.get("consumed", 0)),
        "spilled": int(acct.get("spilled", 0)),
        "owed": float(acct.get("owed", 0.0)),
        "entries": entries,
      }
    return self
