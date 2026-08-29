"""Points as metabolism: hunger, satisfaction, and a cap (issue #36).

Before this, points only ever went up, and the robot only ever wanted more of
them. Both halves are a problem. An unbounded score is not a motivation --
1 400 points and 1 420 points are the same day -- and a robot that always
wants more has no reason to ever do anything that does not pay.

So points are FOOD. They are consumed at a steady rate whatever the robot is
doing, they stop accumulating at a cap, and once the balance is high enough
the robot is SATISFIED: it has eaten, and the hours it did not have to spend
earning are the ones it spends on `Goals.md`. The free time is the whole
mechanic. Everything here exists to make sure there is some.

FOUR RULES, AND EACH ONE IS A FAILURE MODE THIS MODULE IS SHAPED AROUND.

  ZERO IS NARRATIVE, NEVER A CAPABILITY LOCK. A starving robot shows it -- a
  worried face, a line in `History.md`, a prompt that says go and earn
  something -- and is not prevented from doing one single thing. Nothing in
  the charge branch, the navigator or a stow reads a balance, and
  `tests/test_metabolism.py` flies a mission at zero points to keep it that
  way. This is `ledger.py`'s "a robot that can spend itself out of a charge
  eventually will", arriving from the other direction: hunger that could
  brick a world overnight is a worse mechanic than no hunger at all.

  DECAY RUNS ON SIM TIME AND MUST NOT DOUBLE-CHARGE. It ticks on the physics
  seam, like `TaskProducer` and for the same reason -- an appetite that only
  moved between errands would not charge the robot for the twenty minutes it
  spent on one. Elapsed SIM seconds, so a fast-forwarded world gets hungry
  faster and a paused one does not get hungry at all, which is what a viewer
  would expect of both.

  ...AND A RESTART IS NOT A MEAL, NOR A MISSED ONE. Every mission end is a
  restart and sim time starts again at 0, so the anchor is re-taken on the
  first tick of a run and NOTHING is charged for the gap. What survives is
  the BALANCE -- the ledger's file, which is where hunger actually lives --
  plus the fraction of a point owed at the last carry. A robot that came back
  from a restart with a full stomach, or with an hour of appetite charged in
  one step, would be the same bug wearing opposite signs.

  THE CAP REFUSES EARNINGS OUT LOUD. Points over the cap are not banked, and
  `Ledger` records how many were refused rather than silently trimming the
  award -- a scoreboard that quietly pays less than the reward table says is
  the reward table's whole design undone. See `Ledger._post`.

TWO CURRENCIES, AND THEY DO NOT CONVERT. Points are this: an in-game
metabolism, earned by working and consumed by living. USD (`mind/spend.py`,
issue #37) is the real thinking budget. There is no path from one to the
other in either direction, deliberately -- a robot that could buy thinking
with points would have a reason to grind, and one that could buy points with
money would have a reward table denominated in Ben's invoice.

The numbers are DATA (`economy/metabolism.json`, `$PLUGGY_METABOLISM`), the
fifth such file after rewards, cadence, questions and energy, and they are
MEASURED: two unattended 1-sim-hour `home` runs bank 102 points/hour on the
hosting pack, so the shipped 45/hour is ~44 % of the world's income and the
robot's own time is the other half. The file's own note carries the runs, the
rhythm that falls out of them, and what to re-run before re-tuning.

⚠ TUNE ON `--pack hosting`, NEVER ON THE DEMO CELL. The demo run banked a
comparable 80 points/hour and completed ZERO JOBS: a charged demo pack holds
0.990 Wh and every target but one costs more, so every point came from
CHARGING. A rate calibrated there would make charging the food and work
optional, which is this module upside down.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

#: What the robot's appetite may be called on the wire. A two-repo vocabulary
#: on `FACE_STATES`' terms, so it lives in telemetry/protocol.py with the rest
#: of them; imported here because this is the module that DECIDES which one is
#: true, and a consumer of this one should not have to know that.
from pluggybot.telemetry.protocol import HUNGER_STATES, ROBOT_ROOT  # noqa: F401

#: The shipped appetite. Overridable per deploy with $PLUGGY_METABOLISM, the
#: same door economy/rewards.json, cadence.json, questions.json and
#: energy.json are opened by.
METABOLISM_PATH = Path(__file__).with_name("metabolism.json")
METABOLISM_ENV = "PLUGGY_METABOLISM"
METABOLISM_VERSION = 1

#: Sim seconds between appetite ticks. A POLLING interval rather than a rate
#: -- the rate is in the data file -- and it exists for `cadence.CHECK_S`'s
#: reason: this hangs off the ~500 Hz physics seam, and recomputing an
#: appetite that moves a whole point every eighty seconds a thousand times a
#: second is Python spent to learn nothing.
TICK_S = 1.0

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Appetite:
  """One world's hunger policy, as read off economy/metabolism.json.

  Frozen, and every field is a number: an Appetite is the ANSWER to "how fast
  does this robot get hungry", not the thing that acts on one. `Metabolism`
  below is that.
  """

  world: str = ""
  points_per_hour: float = 45.0
  cap: int = 90
  satisfied_at: int = 45
  hungry_at: int = 20

  def __post_init__(self) -> None:
    # Checked at LOAD, not at use. This is a small hand-edited file on a
    # mounted volume, and the two orderings below fail in ways that are
    # invisible at runtime: `satisfied_at > cap` is a robot that can never be
    # satisfied (it starves forever at a full balance), and
    # `hungry_at > satisfied_at` inverts the hysteresis into a latch that
    # flips on every point. Neither raises anything downstream; both just
    # quietly delete the mechanic.
    if self.points_per_hour < 0:
      raise ValueError(f"pointsPerHour must be >= 0, got "
                       f"{self.points_per_hour}")
    if self.cap <= 0:
      raise ValueError(f"cap must be > 0, got {self.cap}")
    if not 0 < self.hungry_at <= self.satisfied_at <= self.cap:
      raise ValueError(
        f"expected 0 < hungryAt <= satisfiedAt <= cap, got "
        f"hungryAt={self.hungry_at}, satisfiedAt={self.satisfied_at}, "
        f"cap={self.cap}")

  @classmethod
  def load(cls, world: str = "",
           path: str | os.PathLike | None = None) -> "Appetite":
    """Read one world's block. `path`, else $PLUGGY_METABOLISM, else the
    shipped file; the world's block overrides the default block key by key."""
    target = Path(path or os.environ.get(METABOLISM_ENV) or METABOLISM_PATH)
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if version != METABOLISM_VERSION:
      # Refused rather than read defensively, exactly as for the reward
      # table, the cadence and the energy model: running a world's appetite
      # out of a shape this build does not understand is worse than refusing
      # to start.
      raise ValueError(f"{target}: metabolism version {version}, "
                       f"expected {METABOLISM_VERSION}")
    spec = dict(doc.get("default") or {})
    spec.update(dict((doc.get("worlds") or {}).get(world) or {}))
    return cls(world=world,
               points_per_hour=float(spec.get("pointsPerHour", 45.0)),
               cap=int(spec.get("cap", 90)),
               satisfied_at=int(spec.get("satisfiedAt", 45)),
               hungry_at=int(spec.get("hungryAt", 20)))


class Metabolism:
  """The robot's appetite: what it eats, and whether it has had enough.

  Given a `Ledger` and an `Appetite`, and driven by `tick(sim_time)` off the
  physics seam. It is the only thing in the repo that moves a balance DOWN,
  and it has no verb the robot can reach -- the ledger's rule ("nothing
  awards itself points") from the opposite end: nothing declines to be
  hungry either.
  """

  def __init__(self, ledger, appetite: Appetite | None = None,
               robot: str = ROBOT_ROOT) -> None:
    self.ledger = ledger
    self.appetite = appetite if appetite is not None else Appetite()
    self.robot = robot
    #: Sim time of the last tick. `None` until the first one, which ANCHORS
    #: and charges nothing -- see the module docstring: a restart must not
    #: back-charge the gap, and sim time begins again at 0 on every run.
    self._last_t: float | None = None
    #: Fraction of a point owed but not yet charged, carried between ticks.
    #: Read back from the ledger's file so a restart resumes mid-point
    #: rather than rounding the robot a free meal every time it wakes up.
    self.owed = float(ledger.owed(robot))
    #: The hysteresis latch. Seeded from the balance rather than persisted,
    #: and deliberately on the HUNGRY side of the ambiguous band: a robot
    #: that came back from a restart still coasting on a satisfaction it
    #: could no longer justify would idle through the first stretch of every
    #: mission. Going back to work is the recoverable direction.
    self.satisfied = ledger.balance(robot) >= self.appetite.satisfied_at
    #: What the last tick changed the state to, for callers that narrate a
    #: transition rather than a level (`HubLifecycle._metabolism_step`).
    self._last_state = self.state

  # ---- the appetite --------------------------------------------------------

  @property
  def points(self) -> int:
    return self.ledger.balance(self.robot)

  @property
  def state(self) -> str:
    """One of `HUNGER_STATES`, with the latch already applied.

    ⚠ `starving` OUTRANKS the latch and is checked first, but cannot
    contradict it: `hungry_at` is > 0 by construction, so a balance of zero
    has already dropped the latch on the tick that got there. The order is
    for readers, not for correctness.
    """
    if self.points <= 0:
      return "starving"
    if self.satisfied:
      return "satisfied"
    if self.points < self.appetite.hungry_at:
      return "hungry"
    # Above the hungry line and climbing, but not yet enough to stop for.
    return "fed"

  def _latch(self) -> None:
    """Move the satisfaction latch. A Schmitt trigger, and the gap between
    the two thresholds is what stops the state flapping once a point per
    eighty seconds around a single line -- the hysteresis rule
    docs/ActivityPattern.md states for a sensed criterion, applied to a
    sensed BALANCE."""
    if self.satisfied:
      if self.points < self.appetite.hungry_at:
        self.satisfied = False
    elif self.points >= self.appetite.satisfied_at:
      self.satisfied = True

  # ---- the clock -----------------------------------------------------------

  def tick(self, t: float) -> int:
    """Charge whatever sim time has passed. Returns the points consumed.

    Cheap enough for the physics seam: an early return on the interval, and
    the ledger is only touched when a WHOLE point is due.
    """
    t = float(t)
    if self._last_t is None or t < self._last_t:
      # First tick of a run, or sim time went backwards -- which on this seam
      # means a new mission against a persisted ledger. Re-anchor and charge
      # nothing: the hunger that survived is the balance in the file, and
      # billing the robot for the wall-clock hours the container was down is
      # the failure the module docstring names.
      self._last_t = t
      self._latch()
      return 0
    dt = t - self._last_t
    if dt < TICK_S:
      return 0
    self._last_t = t
    self.owed += dt * self.appetite.points_per_hour / SECONDS_PER_HOUR
    whole = int(self.owed)
    eaten = 0
    if whole:
      self.owed -= whole
      # The carry rides WITH the write, so the file is never a fraction of a
      # point out of step with the balance it sits beside.
      eaten = self.ledger.consume(whole, t=t, robot=self.robot,
                                  owed=self.owed)
    else:
      # ...and it is kept current in memory even when no point is due, so
      # the next save from ANY source (an award, the next consume) carries
      # it. Without this a mission that ended mid-point discarded the
      # fraction every time, which is a systematic rounding in the robot's
      # favour -- small on an hour-long mission and total on a short one.
      self.ledger.carry(self.owed, self.robot)
    self._latch()
    return eaten

  def changed(self) -> str:
    """The new state if it has moved since this was last asked, else "".

    A level is a gauge and a transition is an event; this is what turns the
    first into the second so a caller can narrate "got hungry again" without
    saying it once a second.
    """
    now = self.state
    if now == self._last_state:
      return ""
    self._last_state = now
    return now

  # ---- what the site and the model are shown -------------------------------

  def snapshot(self) -> dict:
    """The wire block (protocol 0.13.0) and the overseer's context, in one
    shape. DISPLAY, like the ledger's balance and the week's spend: there is
    no inbound message and no decision field that moves any of it."""
    consumed = self.ledger.consumed(self.robot)
    return {
      "state": self.state,
      "satisfied": self.satisfied,
      "points": self.points,
      "cap": self.appetite.cap,
      "pointsPerHour": round(self.appetite.points_per_hour, 3),
      "satisfiedAt": self.appetite.satisfied_at,
      "hungryAt": self.appetite.hungry_at,
      # Totals since the ledger was opened, which is the state volume's
      # lifetime rather than this mission's. `spilled` is the cap's receipt:
      # it is what the reward table paid and the balance refused, and it
      # would be invisible without this number.
      "consumed": consumed,
      "spilled": self.ledger.spilled(self.robot),
    }


def load(world: str = "", path: str | os.PathLike | None = None) -> Appetite:
  """Module-level convenience, matching `cadence.default_cadence` and
  `energy.load`."""
  return Appetite.load(world, path)
