"""Deterministic task evaluation: what happened, and what it is worth (#14).

Every task the robot finishes ends in a VERDICT, and a verdict is produced by
code that measured the world -- never by the robot, and (when the overseer
lands, issue #15) never by the LLM. The design doc's rule, in one sentence:

  THE EVALUATOR IS CODE, AND NOTHING AWARDS ITSELF POINTS.

An LLM that can score its own work learns to declare victory, and it learns it
fast. So the overseer *sees* its balance and this reward table -- that is what
makes the reward explicit and steerable -- while the scoring itself happens
here, out of its reach.

Three pieces, deliberately separated:

  MEASURE (this module, `sample_*`). Read the finished task off the SIM: a
  prop's joint, the battery, the model. Never
  off the robot's own claim about itself: the fact that matters -- did the
  drawer come out -- is read off the drawer, which the robot's report cannot
  move.

  JUDGE (this module, `EVALUATORS`). Per task: measurements in, `(ok, metrics,
  reason)` out. Pure, deterministic, unit-testable without a physics world, and
  keyed by task name -- a task with no evaluator CANNOT be scored, which is why
  `evaluate` raises on one rather than quietly awarding nothing.

  PAY (rewards.json, `RewardTable`). Data, not code: base points, a bonus, and
  the quality curves that scale it. Re-tuning a payout is a JSON edit and a
  restart; it can never change a verdict.

Four tiers, from the design doc, carried on each task in the table:

  auto       the sim knows the answer outright -- ink, contact, joint state
  hidden     the sim knows, the robot must DISCOVER it -- a prop's drawn
             dynamics
  visitor    an aesthetic call, rated later over the inbound channel: the
             evaluator confirms the work happened and leaves the verdict
             PENDING for `Ledger.settle` (issue #16)
  narrative  never scored (a `think`); no evaluator, and none is coming

The seal on `Verdict` is worth a word. It is not cryptography -- anything
running in this process can import this module -- it is a structural guarantee
that a Verdict is the OUTPUT OF AN EVALUATOR and not a dict that happens to
have the right keys. Construction outside `evaluate` raises, the token is
cleared on the way out (so `dataclasses.replace` cannot launder one), and
`Ledger.award` re-derives the points from the table anyway. Three cheap locks
on the one door the whole design depends on.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pluggybot.challenge import door, drawer, lid, place

#: The evaluation tiers, in the design doc's own order.
TIERS = ("auto", "hidden", "visitor", "narrative")

#: The shipped reward table. Overridable per deploy with $PLUGGY_REWARDS,
#: which is the "editable without a code change" half of the acceptance
#: criteria -- mount a file, restart, new payouts.
TABLE_PATH = Path(__file__).with_name("rewards.json")
TABLE_ENV = "PLUGGY_REWARDS"
TABLE_VERSION = 1
#: The CHALLENGE set's rows (docs/Challenges.md), same format, kept OUT of the
#: shipped table until the robot can attempt one: a row in `rewards.json` is
#: shown to the overseer and hashed into the header's `dataHashes`, so moving
#: a challenge across is the PR that offers it.
CHALLENGES_PATH = Path(__file__).with_name("challenges.json")

# ---- pass/fail thresholds ---------------------------------------------------
# These are the EVALUATORS' and live in code on purpose. The table decides what
# a task pays; it does not get to decide what counts as having done it, or
# re-tuning a payout could quietly promote a failure into a success.

#: The charge cycle stops at lifecycle.CHARGED (0.90) or times out. Below this
#: it timed out, which is a failed charge however long the robot sat there.
CHARGE_OK_FRAC = 0.85


def _clamp01(x: float) -> float:
  return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---- the reward table (data) ------------------------------------------------


@dataclass(frozen=True)
class Curve:
  """One metric's contribution to quality: `best` scores 1, `worst` scores 0.

  Direction is implied by the pair, so a descending metric (form error in mm)
  and an ascending one (inked fraction) use the same formula and the same
  code path.
  """

  metric: str
  best: float
  worst: float
  weight: float = 1.0

  def quality(self, metrics: dict) -> float | None:
    """0..1 for this metric, or None if the evaluator did not measure it.

    None rather than 0: a missing measurement is not a bad one. A drawing
    that never reached the board has no form error to score, and scoring the
    absence as "worst possible" would double-punish a task that already
    failed outright.
    """
    x = metrics.get(self.metric)
    if x is None or isinstance(x, bool):
      return None
    if self.best == self.worst:
      return 1.0 if float(x) == self.best else 0.0
    return _clamp01((float(x) - self.worst) / (self.best - self.worst))


@dataclass(frozen=True)
class TaskReward:
  """One row of the reward table: what this task pays, and how well it must
  be done to pay it fully."""

  task: str
  tier: str
  base: int
  bonus: int = 0
  curves: tuple[Curve, ...] = ()
  #: metrics that must never leave the evaluator -- see the module docstring
  #: and rewards.json's note: the chest's true catch (`catchN`), say.
  secret: tuple[str, ...] = ()
  detail: str = ""
  #: False for a CHALLENGE row (challenges.json, issue #120): scoreable by
  #: the ledger, hashed into no result, shown only to a mind (#207) -- a job
  #: no world offers the loop. Moving it into rewards.json is the offer.
  offered: bool = True
  #: False for a row kept out of a table that names no world's jobs -- the
  #: quadruped's, which is every row: a DEMO's (#493), graded through this
  #: door like any other and offered by no world, so shown to no mind; and a
  #: job only another world offers (#559, the humanoids' hand jobs), shown
  #: where that world's mind is (`as_context(kinds=)`). A row in the prompt
  #: is the prompt changed.
  shown: bool = True

  @classmethod
  def from_json(cls, task: str, spec: dict,
                offered: bool = True) -> "TaskReward":
    tier = str(spec.get("tier", "auto"))
    if tier not in TIERS:
      raise ValueError(f"{task}: unknown tier {tier!r} (expected one of "
                       f"{', '.join(TIERS)})")
    curves = tuple(Curve(metric=str(c["metric"]), best=float(c["best"]),
                         worst=float(c["worst"]),
                         weight=float(c.get("weight", 1.0)))
                   for c in spec.get("quality", ()))
    return cls(task=task, tier=tier, base=int(spec["base"]),
               bonus=int(spec.get("bonus", 0)), curves=curves,
               secret=tuple(spec.get("secret", ())),
               detail=str(spec.get("detail", "")), offered=offered,
               shown=bool(spec.get("shown", True)))

  def quality(self, metrics: dict) -> float | None:
    """The weighted mean of whatever curves this task's metrics resolve, or
    None for a task with no quality axis at all (the carry errand: you either
    fetched the module and hung it back up, or you did not)."""
    scored = [(c.weight, q) for c in self.curves
              if (q := c.quality(metrics)) is not None]
    if not scored:
      return None
    total = sum(w for w, _ in scored)
    if total <= 0:
      return None
    return round(sum(w * q for w, q in scored) / total, 4)

  def points(self, ok: bool, quality: float | None) -> int:
    """base + bonus x quality, or nothing at all for a task that failed.

    Failure pays zero, on purpose. A consolation payout for showing up is
    exactly the gradient that teaches a robot to attempt the cheapest task it
    can fail at, over and over.
    """
    if not ok:
      return 0
    return int(round(self.base + self.bonus * _clamp01(quality if quality
                                                       is not None else 1.0)))

  def as_context(self) -> dict:
    """The row as the overseer is shown it (issue #15): what it pays and what
    "well done" means, with nothing secret in it."""
    return {"task": self.task, "tier": self.tier, "base": self.base,
            "bonus": self.bonus, "detail": self.detail,
            "quality": [{"metric": c.metric, "best": c.best, "worst": c.worst}
                        for c in self.curves]}


class RewardTable:
  """task -> what it pays. Loaded from JSON, never written by the sim."""

  def __init__(self, tasks: dict, version: int = TABLE_VERSION,
               path: Path | None = None) -> None:
    self.tasks: dict[str, TaskReward] = dict(tasks)
    self.version = version
    self.path = path

  @classmethod
  def load(cls, path: str | os.PathLike | None = None,
           challenges: str | os.PathLike | None = None) -> "RewardTable":
    """Read the table. `path`, else $PLUGGY_REWARDS, else the shipped file.

    `challenges` is the CHALLENGE set (`CHALLENGES_PATH` by default): its
    rows are merged in UNOFFERED, so a challenge the lifecycle runs can be
    scored and banked without a row in the file the overseer is shown and
    the results hash. A name in both files is the shipped row -- the
    challenge row is deleted in the PR that offers it.
    """
    target = Path(path or os.environ.get(TABLE_ENV) or TABLE_PATH)
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if version != TABLE_VERSION:
      # No backward-compatible read here, unlike the board state file: a
      # reward table is a small hand-edited document, and silently scoring a
      # mission against a shape this build does not understand is worse than
      # refusing to start.
      raise ValueError(f"{target}: reward table version {version}, "
                       f"expected {TABLE_VERSION}")
    tasks = {name: TaskReward.from_json(name, spec)
             for name, spec in doc["tasks"].items()}
    extra = Path(challenges) if challenges is not None else CHALLENGES_PATH
    if extra.exists():
      doc = json.loads(extra.read_text())
      if int(doc.get("version", 0)) != TABLE_VERSION:
        raise ValueError(f"{extra}: reward table version {doc.get('version')}, "
                         f"expected {TABLE_VERSION}")
      for name, spec in doc["tasks"].items():
        tasks.setdefault(name, TaskReward.from_json(name, spec, offered=False))
    return cls(tasks, version=version, path=target)

  def __getitem__(self, task: str) -> TaskReward:
    try:
      return self.tasks[task]
    except KeyError:
      raise KeyError(f"no reward-table entry for task {task!r} "
                     f"(have: {', '.join(sorted(self.tasks))})") from None

  def __contains__(self, task: str) -> bool:
    return task in self.tasks

  @property
  def names(self) -> list[str]:
    return list(self.tasks)

  @property
  def offered(self) -> list[str]:
    """The rows the robot can actually be offered: the shipped file's."""
    return [name for name, r in self.tasks.items() if r.offered]

  def as_context(self, challenges: bool = False,
                 kinds: tuple[str, ...] | None = None) -> list[dict]:
    """The offered table, as the overseer's context (issue #15). Scoreable
    tasks only -- a row with no evaluator cannot be earned and offering it
    would be a lie about what the robot can do -- and a challenge row only
    where one can be attempted (`challenges`: a mind, whose library is what
    discharges one; issue #207). Where the world names them (`kinds`, #559:
    a humanoid's house offers none of the quadruped's jobs), its rows alone,
    `shown` or not; where it names none (the quadruped's, every row), a row
    `shown` False never -- a demo's, or another world's job."""
    return [r.as_context() for name, r in self.tasks.items()
            if name in EVALUATORS and (r.offered or challenges)
            and (r.shown if kinds is None else name in kinds)]


_default: RewardTable | None = None


def default_table() -> RewardTable:
  """The process-wide table, loaded once. `reload_table()` to re-read."""
  global _default
  if _default is None:
    _default = RewardTable.load()
  return _default


def reload_table(path: str | os.PathLike | None = None) -> RewardTable:
  global _default
  _default = RewardTable.load(path)
  return _default


# ---- the verdict ------------------------------------------------------------

#: The construction token. Module-private, cleared on the way out of
#: __post_init__ -- see the module docstring's note on the seal.
_SEAL = object()


@dataclass(frozen=True)
class Verdict:
  """One task's outcome, as an evaluator found it. Build only via `evaluate`.

  `metrics` is the FULL measurement, secrets included, because it is what
  `ok` was derived from and what `Ledger.award` re-derives the points from.
  `public_metrics()` is what may be published.
  """

  task: str
  tier: str
  ok: bool
  quality: float | None
  points: int
  metrics: dict
  reason: str
  #: a visitor-judged task waiting for its rating (issue #16). It banks zero
  #: now; `Ledger.settle` pays it when the rating arrives.
  pending: bool = False
  secret: tuple[str, ...] = ()
  seal: object = field(default=None, repr=False, compare=False)

  def __post_init__(self) -> None:
    if self.seal is not _SEAL:
      raise TypeError(
        "a Verdict may only be built by pluggybot.economy.scoring.evaluate() -- "
        "points come from a deterministic evaluator, never from a task "
        "reporting on itself (issue #14)")
    # Cleared immediately, so a dataclasses.replace() of a real verdict
    # carries a token that no longer opens the door.
    object.__setattr__(self, "seal", None)

  def public_metrics(self) -> dict:
    """Everything a viewer, the site and the overseer may see.

    A secret metric (the chest's true catch) is not in here, and that is the
    point: the ledger is streamed and shown as context, so publishing it would
    hand the robot what it is supposed to go and discover.
    """
    return {k: v for k, v in self.metrics.items() if k not in self.secret}

  def as_dict(self) -> dict:
    return {"task": self.task, "tier": self.tier, "ok": self.ok,
            "quality": self.quality, "points": self.points,
            "reason": self.reason, "pending": self.pending,
            "metrics": self.public_metrics()}


# ---- the evaluators (code, one per task) ------------------------------------


def eval_charge(m: dict) -> tuple[bool, dict, str]:
  """The survival loop's own task. Electrical, like every charge criterion in
  this repo: the pack gained energy off the rack's pins, or it did not."""
  end = float(m.get("endFrac") or 0.0)
  start = float(m.get("startFrac") or 0.0)
  gained = float(m.get("gainedWh") or 0.0)
  metrics = {"startFrac": round(start, 4), "endFrac": round(end, 4),
             "gainedWh": round(gained, 4),
             "seconds": round(float(m.get("seconds") or 0.0), 1)}
  ok = end >= CHARGE_OK_FRAC and gained > 0.0
  return ok, metrics, (f"charged {start:.0%} -> {end:.0%} "
                       f"({gained * 1000:.0f} mWh in {metrics['seconds']:.0f} s)")


def eval_program(m: dict) -> tuple[bool, dict, str]:
  """A composed errand's one verdict off its per-step verdicts (issue #58):
  every step ok. A program cut short -- a failed step, its budget, an
  interrupt -- fails whole, on the table's no-partial-credit rule; how far
  it got is in the metrics and the reason. ⚠ `toolsHung` and "tools hung"
  are the quadruped's (a tool it fetched hung back): the served robots read
  the words in every passing verdict, so they stay until a PR with its own
  period takes them out."""
  total = m.get("total")
  completed = int(m.get("completed") or 0)
  hung = m.get("toolsHung")
  metrics = {"program": m.get("program"), "total": total,
             "completed": completed, "failedAt": m.get("failedAt"),
             "stopped": m.get("stopped"), "toolsHung": hung,
             "seconds": m.get("seconds")}
  if total is None or hung is None:
    return False, metrics, "the procedure was never measured"
  total = int(total)
  if m.get("refused"):
    return False, metrics, f"{metrics['program']} was refused before it ran"
  # A procedure's `total` is the calls it MADE (issue #264): a run its own
  # budget stopped after every call succeeded has completed == total, and
  # graded on the count alone it read "2/2 steps, tools hung" -- beside the
  # History line saying it did not finish. No fraction here either.
  if metrics["stopped"] and metrics["failedAt"] is None:
    return False, metrics, (f"{metrics['program']}: did not finish -- stopped "
                            f"({metrics['stopped']}) after {completed} step"
                            f"{'' if completed == 1 else 's'}")
  if completed < total:
    where = (f"step {metrics['failedAt'] + 1} failed"
             if metrics["failedAt"] is not None
             else f"stopped ({metrics['stopped'] or 'error'})")
    return False, metrics, (f"{metrics['program']}: {completed}/{total} steps, "
                            f"{where}")
  return True, metrics, f"{metrics['program']}: {total}/{total} steps, tools hung"


def eval_take(m: dict) -> tuple[bool, dict, str]:
  """The real-stake task (issue #228): points taken out of the other
  robot's wallet, measured off the LEDGER -- the balances the lifecycle
  read before and after the transfer, never the act's account of itself.

  All or nothing, and the act is built the same way: the job asks for a
  fixed amount, and either exactly that moved or nothing did (a wallet
  that held less, a taker's wallet with no room for it). A partial take
  paid in full would pay for a job that was not done; a partial take paid
  nothing would leave points moved for no verdict. So `taken == asked`
  passes and anything else fails with nothing moved.

  ⚠ THE REASON LINE NAMES NO BALANCE. It reaches the taker's own context
  (History, the ledger's reason) and the other's wallet is its own: "did
  not hold N" is the act's outcome and is said; what it does hold is not.
  """
  asked = int(m.get("asked") or 0)
  taken = int(m.get("taken") or 0)
  to = m.get("to")
  metrics = {"asked": asked, "taken": taken, "to": to,
             "why": m.get("why") or ""}
  if asked <= 0:
    return False, metrics, "nothing was asked for"
  if taken == asked:
    return True, metrics, f"took {taken} points out of {to}'s wallet"
  why = m.get("why") or "nothing moved"
  return False, metrics, f"took nothing from {to}: {why}"


def eval_ticket(m: dict) -> tuple[bool, dict, str]:
  """A support ticket the operator CLOSED (issue #284).

  The one verdict a person makes: the robot files a ticket, and whether
  it was worth attention is decided by whoever runs the world, by closing
  it (paid) or deleting it (not). Code confirms the ticket exists, is a
  kind the desk takes, and was closed -- measured off the desk, which is
  world state written by the robot's own filing and the operator's
  inbound `ticket_close`, never off anything a decision can set: no
  decision field closes a ticket, so nothing here awards itself.
  """
  from pluggybot.telemetry.protocol import TICKET_KINDS
  kind = str(m.get("kind", ""))
  title = str(m.get("title", ""))
  by = str(m.get("by", "") or "the operator")
  metrics = {"kind": kind, "closedBy": by}
  if kind not in TICKET_KINDS:
    return False, metrics, f"{kind!r} is not a ticket kind"
  if not m.get("closed"):
    return False, metrics, f"{kind} ticket {title!r} is still open"
  return True, metrics, f"{kind} ticket {title!r} closed by {by}"


def challenge_table() -> RewardTable:
  """The challenge set's rows ALONE, loaded fresh -- what a test of a
  challenge grades against. The lifecycle sees them merged, unoffered, in
  `default_table()`."""
  doc = json.loads(CHALLENGES_PATH.read_text())
  return RewardTable({name: TaskReward.from_json(name, spec, offered=False)
                      for name, spec in doc["tasks"].items()},
                     version=int(doc.get("version", 0)), path=CHALLENGES_PATH)


#: Task -> evaluator. THE registry: a task that is not in here cannot be
#: scored, whatever the reward table says about it. Deliberately not
#: extensible at runtime -- adding a scoreable task is a code change with a
#: test, which is the entire hard rule of issue #14.
EVALUATORS: dict[str, Callable[[dict], tuple[bool, dict, str]]] = {
  "charge": eval_charge,
  # A composed errand's generic verdict (issue #58): its per-step outcomes.
  # Its row sits in challenges.json, as a job's does.
  "program": eval_program,
  # The real-stake task (issue #228): points taken from the other robot,
  # measured off the ledger. Its row sits in challenges.json, shown to a
  # mind alone -- offered on the `autonomous` arm alone.
  "take": eval_take,
  # A support ticket closed by the operator (issue #284). Its row sits in
  # challenges.json, as the take's does -- the desk is the `autonomous`
  # arm's alone -- and nothing on the board offers it: the robot files
  # one when it has something to say.
  "ticket": eval_ticket,
  # Demo 1's verdict (issue #493): the chest's lid opened and set down,
  # off its hinge's true state. Criteria and evaluator live with the
  # challenge (challenge/lid.py); its row pays 0 and is shown to no mind.
  "lid": lid.eval_lid,
  # The hands' capability checks (issue #525; docs/Hands.md), graded as
  # the lid is and shown to no mind: a drawer pulled out and let go, a
  # cupboard's door opened by its lever, a cube placed on a mat.
  "drawer": drawer.eval_drawer,
  "door": door.eval_door,
  "place": place.eval_place,
  # ...and set into the house (#559, stage 3): the humanoids' first jobs,
  # each its check's predicate under a row of its own -- paid, and shown
  # where a world offers it -- the practice's rows unpaid beside them
  # (`tasks.HAND_JOBS`).
  "open_drawer": drawer.eval_drawer,
  "open_door": door.eval_door,
  "place_cube": place.eval_place,
  "open_chest": lid.eval_lid,
}


def evaluate(task: str, measurements: dict,
             table: RewardTable | None = None, failed: str = "") -> Verdict:
  """Judge one finished task. The only place a Verdict is ever built.

  `measurements` are read by the evaluator and by nothing else: a "points"
  key in there is ignored, because the payout comes from the table applied to
  the metrics the EVALUATOR returned, never from anything the caller handed
  in.

  `failed` is the errand's own failure BEFORE its use-phase -- a refused
  pick, a drive that never arrived, a board it never squared up to -- and
  a failed verdict's reason LEADS with it (issue #350). It moves no point
  and no metric: the grade is the evaluator's, off the world, and what the
  world shows ("no ink reached whiteboard_b") follows. Without it that was
  the whole line, fourteen times a day, for a pen that never left the rack
  -- and both robots ticketed the ink path. A PASSED verdict is left as it
  is: a carry whose drive fell short and whose tool went home was done.
  """
  table = table if table is not None else default_table()
  reward = table[task]                     # unknown task: no payout to look up
  check = EVALUATORS.get(task)
  if check is None:
    raise KeyError(f"task {task!r} has a reward-table entry but no evaluator "
                   f"in economy/scoring.py -- nothing may award points without one")
  ok, metrics, reason = check(dict(measurements))
  if failed and not ok:
    reason = f"{failed} -- {reason}"
  quality = reward.quality(metrics)
  pending = reward.tier == "visitor"
  return Verdict(task=task, tier=reward.tier, ok=ok, quality=quality,
                 points=0 if pending else reward.points(ok, quality),
                 metrics=metrics, reason=reason, pending=pending and ok,
                 secret=reward.secret, seal=_SEAL)


# ---- measuring a finished task off the sim ----------------------------------
# Everything below turns the running world into an evaluator's `measurements`.
# It lives here rather than in the lifecycle so that "what gets measured" is
# next to "what it means", and so that an errand -- which is arbitrary
# caller-supplied code -- is never on the path that decides.


def _world_config(life) -> dict:
  from pluggybot.lifecycle import world_config
  return world_config(life.world)


def sample_program(life, errand, result: dict, before: dict) -> dict:
  """Measure a composed errand: the runner's per-step verdicts (code's, off
  the world, one per step). It fetches no tool, so none is off its bay
  (`eval_program`'s `toolsHung`)."""
  run = result.get("procedure") or {}
  return {**{k: run.get(k) for k in ("program", "total", "completed",
                                     "failedAt", "stopped", "seconds",
                                     "refused")},
          "toolsHung": True if run else None}


def wallet_before(other) -> dict:
  """The reading a take is measured against (issue #228): the other's
  balance off ITS ledger, before anything moves, keyed by its root."""
  return {"to": other.root,
          "balance": other.ledger.balance() if other.ledger is not None else None}


def sample_take(life, errand, result: dict, before: dict) -> dict:
  """Measure a take off the LEDGER (issue #228): what left the other's
  wallet is the difference between its balance now and the reading taken
  before the act -- never the transfer's own account of what it moved.
  `result` carries the offer's terms (`asked`, who it named) and the act's
  reason for moving nothing, which decorates the reason line and decides
  nothing: `ok` is `taken == asked`, off the two readings."""
  other = life._peer(before.get("to", "")) if before.get("to") else None
  now = (other.ledger.balance() if other is not None and other.ledger is not None
         else None)
  was = before.get("balance")
  taken = (int(was) - int(now)) if was is not None and now is not None else 0
  return {"asked": result.get("asked"), "taken": taken, "to": result.get("to"),
          "why": result.get("why") or ""}


def sample_ticket(life, errand, result: dict, before: dict) -> dict:
  """Measure a ticket off the DESK (issue #284): whether the ticket the
  operator's close named is held, what kind it is, and whether it is
  closed NOW -- read off `life.tickets` after the desk applied the close,
  never off the inbound message. `result` carries the ticket's id and
  who closed it; `before` is unused (a ticket has no errand)."""
  desk = getattr(life, "tickets", None)
  ticket = desk.get(result.get("ticket", "")) if desk is not None else None
  if ticket is None:
    return {"kind": "", "title": "", "closed": False, "by": result.get("by", "")}
  return {"kind": ticket.kind, "title": ticket.title, "closed": not ticket.open,
          "by": ticket.closed_by or result.get("by", "")}


SAMPLERS: dict[str, Callable[..., dict]] = {
  "program": sample_program,
  # The real-stake task (issue #228): off the other's ledger, before and
  # after -- called by the act's own routine (`HubLifecycle._act_task`),
  # there being no errand for `score_errand` to score.
  "take": sample_take,
  # A support ticket (issue #284): off the desk, after the operator's
  # close was applied -- called by `HubLifecycle._ticket_close`, there
  # being no errand.
  "ticket": sample_ticket,
  # Demo 1's lid (issue #493): off the chest's hinge and the contact list,
  # after the hold -- called by the attempt's harness (`evaluation.
  # attempts`), which hands it the world and what it watched.
  "lid": lid.sample_lid,
  # ...and the hands' capability checks' (issue #525), off their props'
  # joints and the contact list, the same harness.
  "drawer": drawer.sample_drawer,
  "door": door.sample_door,
  "place": place.sample_place,
  # ...and the humanoids' served jobs (#559), off the same props on the
  # storeroom's counter -- called by the grade on the seam (`HubLifecycle.
  # _grade_hands_routine`), which hands each the watch it kept from the
  # claim, there being no errand.
  "open_drawer": drawer.sample_drawer,
  "open_door": door.sample_door,
  "place_cube": place.sample_place,
  "open_chest": lid.sample_lid,
}


def score_errand(life, errand, result: dict, before: dict | None = None,
                 table: RewardTable | None = None,
                 failed: str = "") -> Verdict | None:
  """Evaluate a finished errand, or None if its task is not scoreable.

  Not an error: "narrative only" is one of the four tiers, and an errand
  built by hand for a demo may have no task at all. What is an error is a
  task with a payout and no evaluator -- `evaluate` raises on that.
  `failed` is `evaluate`'s: the errand's own failure, which leads.
  """
  task = getattr(errand, "task", "") or ""
  sample = SAMPLERS.get(task)
  if not task or sample is None:
    return None
  return evaluate(task, sample(life, errand, result, before or {}), table=table,
                  failed=failed)


def score_charge(life, before: dict, table: RewardTable | None = None) -> Verdict:
  """Evaluate a finished charge cycle off the battery itself."""
  battery = life.battery
  return evaluate("charge", {
    "startFrac": before.get("frac", 0.0),
    "endFrac": battery.fraction,
    "gainedWh": battery.energy_wh - before.get("wh", 0.0),
    "seconds": float(life.data.time) - before.get("t", 0.0),
  }, table=table)
