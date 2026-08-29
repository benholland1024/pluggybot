"""The weekly USD allowance, and what it has bought (issue #37).

Money enters this repo for the first time here, and the design is the points
ledger's, one layer up: the robot can SEE what it has spent and has no verb
that moves it. Nothing in `mind/overseer.py` writes this file -- the spend is
recorded by the code that made the call, from the response's own usage block
and the backend's own rates, after the fact. A model cannot decline to be
billed.

THREE CEILINGS, AND ONLY THE MIDDLE ONE IS CODE:

  the provider balance   Ben tops the HuggingFace account up and never more.
                         Outside the sim entirely, which is the point: it is
                         the one limit no bug in this loop can raise.
  the weekly allowance   HERE. A soft budget enforced where the calls are
                         actually made, so a website outage cannot stop the
                         robot thinking and no HTTP round trip sits in front
                         of a decision.
  the hourly call cap    `Overseer.calls_per_hour`, unchanged since #15. It
                         bounds a LOOP BUG; this bounds a MONTH.

⚠ THE HOURLY WINDOW COULD NOT BE REUSED, AND THE REASON IS THE RESTART.
`Overseer._calls` is a deque of `time.monotonic` stamps, which is right for
an hour inside one process and useless across a week: every mission end is a
restart (`restart: unless-stopped`, and PLUGGY_MAX_SIM_TIME bounds a mission
at an hour), so a week's spend that lives in a process is a week's spend that
resets several times a day. This is wall-clock stamps in a file on the state
volume, beside the boards and the ledger, for the same reason those are.

⚠ A WEEK IS ROLLING, NOT CALENDAR. A calendar week hands the robot a full
allowance at midnight on Sunday and none at 23:00 on Saturday; a rolling
seven days spends at a steady rate no matter when it started. It also means
"the budget resets" never happens as an event -- entries age out one at a
time, which is what `left` reports.

The numbers are USD, and they are ESTIMATES: they come off the token counts
the endpoint reported and the rates its catalogue published, so they track
the invoice without being it. `priced=False` spending (an endpoint with no
published rates) is recorded as zero dollars AND counted, because a call
whose cost is unknown must not read as a call that was free -- `unpriced`
in the snapshot is what says so.
"""

import json
import os
import time
from pathlib import Path

STATE_VERSION = 1

#: The state file, on the volume the boards and the ledger already live on.
SPEND_ENV = "PLUGGY_SPEND"
#: ...and the allowance itself, so a deployment re-tunes it with an
#: environment edit rather than a rebuild. Dollars per rolling week.
BUDGET_ENV = "PLUGGY_WEEKLY_USD"

#: The soft weekly budget, in USD. Issue #37's number: ~$10/week against a
#: ~$50/month top-up, which leaves the provider balance as the backstop
#: rather than the thing that gets hit.
WEEKLY_USD = 10.0
#: A rolling week, in seconds.
WEEK_S = 7 * 24 * 3600.0
#: Entries kept in the file. Bounds the file, not the total: `spent` is a
#: sum over the entries INSIDE the window, so an aged-out entry is one that
#: no longer counts anyway.
MAX_ENTRIES = 500


class SpendBook:
  """What the thinking has cost this week, and whether there is any left.

  `clock` is injected for the same reason `Overseer.clock` is: a test that
  has to wait seven days to check a window is a test nobody runs.
  """

  def __init__(self, path: str | os.PathLike | None = None,
               weekly_usd: float = WEEKLY_USD, clock=time.time) -> None:
    self.path = Path(path) if path else None
    self.weekly_usd = float(weekly_usd)
    self.clock = clock
    self.entries: list[dict] = []
    self.dropped = 0
    if self.path is not None and self.path.exists():
      self.load()

  # ---- the window ----------------------------------------------------------

  def _live(self) -> list[dict]:
    """Entries inside the rolling week. Ages the list as a side effect."""
    cutoff = self.clock() - WEEK_S
    if self.entries and self.entries[0]["t"] < cutoff:
      self.entries = [e for e in self.entries if e["t"] >= cutoff]
    return self.entries

  @property
  def spent(self) -> float:
    """USD spent in the last seven days."""
    return round(sum(e["usd"] for e in self._live()), 6)

  @property
  def left(self) -> float:
    """USD of allowance remaining. Never negative: an overshoot is a spend
    that already happened, and reporting -0.30 invites arithmetic that
    treats it as credit somewhere."""
    return max(0.0, round(self.weekly_usd - self.spent, 6))

  def can_spend(self, usd: float) -> bool:
    """Is there room for a call estimated at `usd`?

    ⚠ Checked BEFORE the call, against an ESTIMATE, because the true cost is
    only known from the response. The allowance is soft by construction: the
    last permitted call can overshoot by whatever the estimate was wrong by,
    which is cents, and the alternative (refusing everything near the line)
    spends less than the budget on purpose.
    """
    return self.left >= max(0.0, float(usd))

  # ---- recording -----------------------------------------------------------

  def record(self, usd: float, model: str = "", kind: str = "escalation",
             priced: bool = True, tokens: int = 0) -> dict:
    """Bank one call's cost. Returns the entry.

    Called AFTER the response, by the code that made it -- never by anything
    the model can influence. An unpriced call books $0 and is flagged, so
    "we do not know what this cost" cannot be read as "this was free".
    """
    entry = {"t": round(float(self.clock()), 3),
             "usd": round(max(0.0, float(usd)), 6),
             "model": str(model)[:80], "kind": str(kind)[:32],
             "tokens": int(tokens), "priced": bool(priced)}
    self.entries.append(entry)
    self._live()
    if len(self.entries) > MAX_ENTRIES:
      self.dropped += len(self.entries) - MAX_ENTRIES
      self.entries = self.entries[-MAX_ENTRIES:]
    self.save()
    return entry

  # ---- what the site is shown ----------------------------------------------

  def snapshot(self) -> dict:
    """The wire block (protocol 0.12.0). Money is DISPLAY, like the ledger's
    balance: there is no inbound message that can move any of it."""
    live = self._live()
    return {
      "weeklyUsd": round(self.weekly_usd, 4),
      "spentUsd": self.spent,
      "leftUsd": self.left,
      "calls": len(live),
      "escalations": sum(1 for e in live if e["kind"] == "escalation"),
      # A call whose price nobody published. Reported rather than folded in,
      # because the sum above is understated by exactly this many calls.
      "unpriced": sum(1 for e in live if not e["priced"]),
      "recent": [{"t": e["t"], "usd": e["usd"], "model": e["model"],
                  "kind": e["kind"]} for e in live[-5:]],
    }

  # ---- persistence ---------------------------------------------------------

  def save(self, path: str | os.PathLike | None = None) -> Path | None:
    target = Path(path) if path is not None else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": STATE_VERSION,
                               "entries": self.entries,
                               "dropped": self.dropped}, indent=1) + "\n")
    # Rename over the target: a crash mid-write leaves last week's record
    # intact rather than a truncated file that loads as a fresh allowance --
    # which is the failure that spends the month's money in an afternoon.
    os.replace(tmp, target)
    return target

  def load(self, path: str | os.PathLike | None = None) -> "SpendBook":
    target = Path(path) if path is not None else self.path
    if target is None or not target.exists():
      return self
    try:
      raw = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
      # ⚠ A CORRUPT FILE MUST NOT READ AS AN EMPTY ONE. Loading nothing here
      # would hand the robot a full allowance every time the file was
      # damaged, which is the one direction this class must never fail in.
      raise ValueError(f"unreadable spend file {target} -- refusing to treat "
                       "a damaged record as an unspent week")
    self.entries = [e for e in raw.get("entries", []) if isinstance(e, dict)]
    self.dropped = int(raw.get("dropped", 0))
    self._live()
    return self


def open_book(path: str | os.PathLike | None = None,
              weekly_usd: float | None = None, clock=time.time) -> SpendBook:
  """The deployment's spend book: `$PLUGGY_SPEND` and `$PLUGGY_WEEKLY_USD`
  when nothing is passed, like every other data file here."""
  budget = weekly_usd
  if budget is None:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    budget = float(raw) if raw else WEEKLY_USD
  return SpendBook(path or os.environ.get(SPEND_ENV) or None,
                   weekly_usd=budget, clock=clock)
