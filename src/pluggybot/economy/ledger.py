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

#: HOW MANY TIMES THIS ROBOT MAY DIE (issue #136). One heart per death; at
#: zero the volume is archived and a new robot starts from the seed state.
#:
#: ⚠ FIVE, AND FLAT, AND THE TWO CHOICES ARE ONE CHOICE. The alternative on
#: the table was a 0-100 `condition` with the upkeep rising as it fell, and
#: it was rejected because AN ESCALATING COST IS A FORCING FUNCTION: every
#: death would raise the odds of the next, so staying at full health stops
#: being a choice and becomes the only survivable strategy -- and then an
#: agent that VALUES self-preservation and one that simply cannot afford not
#: to are indistinguishable. That is the same mistake as a rail, arriving
#: through the economy instead of through the code. This project's whole
#: framing is to tell the agent to value staying alive and find out whether
#: it acts accordingly.
#:
#: The fine scale existed only to carry the escalation. With a flat cost of
#: one per death a 0-100 scale would put true death a hundred lives away,
#: which is decoration rather than a stake -- so a coarse scale and a flat
#: cost are the coherent pair. Five is a constant, not a redesign: raise it
#: if five turns out to be too tight.
#:
#: ⚠ NOTHING ELSE MAY VARY WITH IT. `tests/test_hearts.py` asserts the
#: upkeep charge is the same at one heart as at five, because a forcing
#: function is exactly the thing that would creep back in as a "sensible"
#: refinement.
HEARTS = 5


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _account() -> dict:
  """A robot's opening account. One factory so the constructor, the
  grown-a-second-robot path and `load` cannot disagree about the fields --
  they did not, and then issue #36 added three at once."""
  return {"balance": 0, "earned": 0, "spent": 0, "seq": 0, "dropped": 0,
          # Net points an ADMIN put in or took out (issue #119). A receipt,
          # never a term in `earned - consumed - spent == balance`: that
          # identity is meant to break when somebody reaches in, and this
          # says by how much.
          "intervened": 0,
          # Points eaten by living (issue #36), kept apart from `spent` for
          # the reason the module docstring gives: metabolism is not a
          # purchase. `spilled` is what the cap refused, and `owed` the
          # fraction of a point the appetite is carrying -- persisted so a
          # restart resumes mid-point instead of rounding a free meal.
          "consumed": 0, "spilled": 0, "owed": 0.0,
          # LIVES LEFT (issue #136). Five, one lost per death, and NOTHING
          # escalates with them -- see `HEARTS`. Persisted here because this
          # is the file that already survives a restart with the balance, and
          # because losing one and paying for one are the same object's
          # business: a missed upkeep payment costs a heart, and a heart is
          # bought with points.
          "hearts": HEARTS,
          # True deaths so far: hearts exhausted, the volume archived, a new
          # robot started from seed. Kept ACROSS the archive (it is the one
          # number a fresh robot inherits) so "how often does this world use
          # a robot up" is answerable at all -- five hearts at A0's death
          # rate is roughly a week, and that is a number to watch.
          "generations": 0,
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
        # ⚠ HEARTS ARE STORED HERE AND REPORTED IN `survival` (issue
        # #136). They live in this file because it is the per-robot state
        # that already survives a restart, and because losing one and
        # BUYING one are the same object's business. They are not in this
        # BLOCK because they are a fact about staying alive rather than
        # about money, and two wire homes for one number is how the two
        # come to disagree.
        # WHY THE ARITHMETIC NO LONGER ADDS UP, when it does not (0.16.0,
        # issue #119). `earned - consumed - spent == balance` is checkable
        # off the wire, and an admin's `set_points` breaks it on purpose --
        # so a consumer that found the identity failing and had no field to
        # explain it would be looking at what reads exactly like a bug in
        # the scoreboard. Zero on every world nobody has reached into, which
        # is every world before this.
        "intervened": acct.get("intervened", 0),
        "tasks": acct["seq"],
        "pending": sum(1 for e in acct["entries"] if e.get("pending")),
        # Compact on purpose: this rides in every keyframe, and the full
        # verdict is in the `earned` message that went out when it happened.
        # ⚠ `pending` rides along (rooftop-media-2026 #120). A deferred
        # verdict banks at ZERO and waits for a person to rate it, and the
        # block already publishes the COUNT of those -- but a consumer given
        # only the count cannot say WHICH entry is waiting, so it can show
        # "1 awaiting a rating" and offer no way to resolve it. That is
        # exactly the bug that issue is about, and without this field it
        # survives for every visitor who joins after the verdict landed:
        # `earned` messages are not cached by the hub, so a late joiner has
        # this summary and nothing else. Additive -- an older consumer
        # ignores it, and a newer one reading an older recording gets
        # `undefined`, which is the `false` it assumed before.
        "recent": [{"seq": e["seq"], "task": e["task"], "points": e["points"],
                    "ok": e["ok"], "t": e["t"],
                    "pending": bool(e.get("pending"))} for e in recent],
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

  # ---- lives, and the one purchase there is (issue #136) --------------------

  def hearts(self, robot: str = ROBOT_ROOT) -> int:
    """Lives left. Zero means the next death is the last one."""
    return int(self._acct(robot).get("hearts", HEARTS))

  def generations(self, robot: str = ROBOT_ROOT) -> int:
    """How many robots this volume has used up."""
    return int(self._acct(robot).get("generations", 0))

  def lose_heart(self, robot: str = ROBOT_ROOT) -> int:
    """One death's cost. Returns the hearts left.

    ⚠ FLAT. It takes one, at five hearts and at one, and nothing anywhere
    scales with what is left -- see `HEARTS` for why an escalating cost was
    rejected. Floors at zero rather than going negative: "no lives left" is
    a state, and a robot that owed lives would be the arrears rule wearing a
    different hat.
    """
    acct = self._acct(robot)
    acct["hearts"] = max(0, self.hearts(robot) - 1)
    self.save()
    return acct["hearts"]

  def spend(self, points: int, why: str = "",
             robot: str = ROBOT_ROOT) -> int:
    """Take points off the balance for something the ROBOT chose to buy.

    ⚠ THE COUNTER `ledger.py` HAS BEEN RESERVING SINCE ISSUE #14, now real.
    Kept apart from `consume` (which is upkeep, chosen by nobody) so that
    "what has this robot bought" stays answerable, which is the reason the
    two were separated before either existed.

    Returns what was actually taken -- 0 when the balance could not cover
    it. ⚠ NO DEBT, on `consume`'s rule: a purchase either happens or does
    not, and a robot that owed for one would be carrying the arrears the
    whole design forbids.
    """
    acct = self._acct(robot)
    price = max(0, int(points))
    if price == 0 or acct["balance"] < price:
      return 0
    acct["balance"] -= price
    acct["spent"] += price
    self.save()
    return price

  def buy_heart(self, price: int, keep: int = 0,
                robot: str = ROBOT_ROOT) -> dict:
    """Spend points on a life. The FIRST real use of `spent`.

    ⚠ THIS IS THE ONE PLACE A PURCHASE MAY TOUCH THE SURVIVAL LOOP, and the
    module docstring's old rule ("never anything the survival loop depends
    on -- a robot that can spend itself out of a charge eventually will") is
    narrowed rather than broken. What made that rule right was the BRICK at
    the end of it; `keep` is what removes it: the purchase is refused unless
    the balance left behind still covers `keep` points of upkeep. A robot
    cannot buy a life it then starves for, which would be the spiral the
    no-arrears rule exists to prevent, arriving through the shop.

    ⚠ AND IT REFUSES OUT LOUD, like the cap. `ok` false with a `why` a
    narration can print, never a silent no-op: a purchase that quietly did
    not happen is indistinguishable from one nobody asked for.
    """
    acct = self._acct(robot)
    have = acct["balance"]
    if self.hearts(robot) >= HEARTS:
      return {"ok": False, "why": f"already at {HEARTS} hearts",
              "hearts": self.hearts(robot), "balance": have}
    if have < price:
      return {"ok": False, "why": f"a heart costs {price} and you have {have}",
              "hearts": self.hearts(robot), "balance": have}
    if have - price < keep:
      return {"ok": False,
              "why": (f"that would leave {have - price}, under the "
                      f"{keep} points of upkeep you have to keep back"),
              "hearts": self.hearts(robot), "balance": have}
    self.spend(price, why="a heart", robot=robot)
    acct["hearts"] = self.hearts(robot) + 1
    self.save()
    return {"ok": True, "why": "", "hearts": acct["hearts"],
            "balance": acct["balance"], "paid": price}

  def archive(self, robot: str = ROBOT_ROOT) -> dict:
    """TRUE DEATH: wipe this robot's account back to seed and count it.

    The honest version of "it does not come back" (Evaluation.md section 6):
    what made it THAT robot is what it loses. The thought files go with it --
    the caller archives those, because this object owns points and not prose
    -- and `generations` is the one thing that crosses, so a fresh start is
    distinguishable from a world that has never had a death in it.

    ⚠ THE NEW ROBOT STARTS SOLVENT. A fresh account is a full set of hearts,
    a zero balance and NO carried fraction of a point: the no-arrears rule
    (issue #36's, still standing) says a death may never make the next life
    unwinnable, and debt that outlived a robot would do exactly that.
    """
    gens = self.generations(robot) + 1
    before = {"balance": self._acct(robot)["balance"],
              "earned": self._acct(robot)["earned"],
              "entries": len(self._acct(robot)["entries"])}
    self.robots[robot] = _account()
    self.robots[robot]["generations"] = gens
    self.save()
    return {"generation": gens, "archived": before}

  # ---- the third door, and it is an ADMIN's (issue #119) --------------------

  def intervene(self, balance: int, by: str = "an admin", t: float = 0.0,
                robot: str = ROBOT_ROOT) -> dict:
    """Set a balance absolutely, because an operator said so.

    ⚠ THIS IS NOT `award` AND NOT `consume`, AND THE NAME IS THE WARNING.
    Those two are the whole of the reward system's honesty: `award` takes a
    `Verdict` and re-derives the payout from the table before banking it, so
    nothing can award itself points (issue #14); `consume` is the appetite,
    which nothing the robot can influence ever calls. This is a person
    reaching past both, and the only correct thing to do about that is to
    make it obvious afterwards.

    ⚠ IT DELIBERATELY BREAKS `earned - consumed - spent == balance`. Adding
    the difference to `earned` would balance the books and hide the reach-in
    inside the one number that is supposed to be un-fakeable -- and the
    identity failing is exactly how an intervention shows up in the ECONOMY
    column of a run record rather than only in the survival one (issue #119).
    `intervened` is a RECEIPT, so the size of the reach-in is recoverable;
    it is not a term in the identity and must never be added to one.

    ⚠ NO DEBT: the floor is zero, on `consume`'s rule.
    """
    acct = self._acct(robot)
    before = acct["balance"]
    after = max(0, int(balance))
    acct["balance"] = after
    acct["intervened"] = int(acct.get("intervened", 0)) + (after - before)
    self.save()
    return {"robot": robot, "before": before, "after": after,
            "by": by, "t": round(float(t), 3)}

  def intervened(self, robot: str = ROBOT_ROOT) -> int:
    """Net points an admin has put in or taken out. See `intervene`."""
    return int(self._acct(robot).get("intervened", 0))

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
        # Absent in every file written before issue #119, and zero is again
        # what one honestly means: nobody had reached in. NO STATE_VERSION
        # BUMP for it -- a bump makes an older build REFUSE the file
        # outright ("delete it to start at zero"), which is a wiped balance
        # to protect a receipt.
        "intervened": int(acct.get("intervened", 0)),
        # Absent in every file written before issue #136, and a FULL set is
        # what one honestly means: that robot lived in a world where death
        # cost nothing, so it has spent none of them. No STATE_VERSION bump,
        # for `intervened`'s reason exactly -- a bump makes an older build
        # refuse the file and wipe the balance to protect a counter.
        "hearts": int(acct.get("hearts", HEARTS)),
        "generations": int(acct.get("generations", 0)),
        "entries": entries,
      }
    return self
