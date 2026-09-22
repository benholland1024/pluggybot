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

  MEASURE (this module, `sample_*`). Read the finished task off the SIM: the
  board book's stroke count, the battery, the occupancy grid, the model. Never
  off the robot's own claim about itself. The errand's result dict is used only
  for quantities the physics measured on the way through (the plotter's traced
  form error), and the fact that matters -- did ink land on the board -- is
  cross-checked against board state, which the pen writes and the errand
  cannot.

  JUDGE (this module, `EVALUATORS`). Per task: measurements in, `(ok, metrics,
  reason)` out. Pure, deterministic, unit-testable without a physics world, and
  keyed by task name -- a task with no evaluator CANNOT be scored, which is why
  `evaluate` raises on one rather than quietly awarding nothing.

  PAY (rewards.json, `RewardTable`). Data, not code: base points, a bonus, and
  the quality curves that scale it. Re-tuning a payout is a JSON edit and a
  restart; it can never change a verdict.

Four tiers, from the design doc, carried on each task in the table:

  auto       the sim knows the answer outright -- ink, contact, joint state
  hidden     the sim knows, the robot must DISCOVER it -- the census
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

from pluggybot.challenge import bench, stack

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
#: shown to the overseer and hashed into every committed result, so moving a
#: challenge across is the PR that offers it, and a re-fly of `guarded`.
CHALLENGES_PATH = Path(__file__).with_name("challenges.json")

# ---- pass/fail thresholds ---------------------------------------------------
# These are the EVALUATORS' and live in code on purpose. The table decides what
# a task pays; it does not get to decide what counts as having done it, or
# re-tuning a payout could quietly promote a failure into a success.

#: A drawing is a drawing once ink is actually on the board. Completeness and
#: form error scale the payout rather than gating it: 5 strokes out of 6 is a
#: worse house, not a non-house.
DRAW_MIN_STROKES = 1
#: ...but ink laid down while TRAVELLING between strokes is a different fault
#: (the pen failed to lift), and past this much of it the figure is scribbled
#: through rather than merely imperfect.
DRAW_MAX_TRAVEL_INK = 0.25
# A written answer's two bars -- how far the ink may sit from the glyphs, and
# how much ink it may be written with -- are `questions.ANSWER_MATCH_MM` and
# `questions.INK_RATIO`, and they stay there rather than here: both were
# measured with the real pen (`scripts/answer_spike.py`), and a second copy
# would be a second, slowly diverging opinion about the same drawing.

#: The charge cycle stops at lifecycle.CHARGED (0.90) or times out. Below this
#: it timed out, which is a failed charge however long the robot sat there.
CHARGE_OK_FRAC = 0.85
#: A routine that mostly happened is a routine; the individual moves are scored
#: by mission/errand.py's own landed-fraction rule (the drivetrain cannot deliver a
#: full 5 s spin and never will, so this is not a tolerance to tighten).
DANCE_MIN_LANDED = 0.67
#: ...but a "dance" that ends this far from where it started drove off.
DANCE_MAX_DRIFT_M = 1.5


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
  #: and rewards.json's note. The census's ground truth is the whole example.
  secret: tuple[str, ...] = ()
  detail: str = ""
  #: False for a CHALLENGE row (challenges.json, issue #120): scoreable by
  #: the ledger, shown to nobody, hashed into no result -- a job the robot
  #: has no way to take yet. Moving it into rewards.json is the offer.
  offered: bool = True

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
               detail=str(spec.get("detail", "")), offered=offered)

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

  def as_context(self, challenges: bool = False) -> list[dict]:
    """The offered table, as the overseer's context (issue #15). Scoreable
    tasks only -- a row with no evaluator cannot be earned and offering it
    would be a lie about what the robot can do -- and a challenge row only
    where one can be attempted (`challenges`: the `autonomous` arm, whose
    library is what discharges one; issue #207). `guarded` sees the table
    it always saw, byte for byte."""
    return [r.as_context() for name, r in self.tasks.items()
            if name in EVALUATORS and (r.offered or challenges)]


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

    The census's ground truth is not in here, and that is the point: the
    ledger is streamed and shown as context, so publishing `truth` would hand
    the robot the answer to the task it is supposed to go and discover.
    """
    return {k: v for k, v in self.metrics.items() if k not in self.secret}

  def as_dict(self) -> dict:
    return {"task": self.task, "tier": self.tier, "ok": self.ok,
            "quality": self.quality, "points": self.points,
            "reason": self.reason, "pending": self.pending,
            "metrics": self.public_metrics()}


# ---- the evaluators (code, one per task) ------------------------------------


def eval_draw(m: dict) -> tuple[bool, dict, str]:
  """Did ink land on the board, and how good is the figure?

  `strokesInked` is counted off the BOARD BOOK -- the record the pen writes
  as it traces -- and not off the errand's report of itself. That is the
  difference between "the task says it drew a house" and "there is a house on
  the board", and it is the single most important line in this module.
  """
  planned = int(m.get("strokes") or 0)
  inked = int(m.get("strokesInked") or 0)
  form = m.get("formMm")
  travel = m.get("travelInkFraction")
  metrics = {
    "strokes": planned,
    "strokesInked": inked,
    "completeness": round(inked / planned, 3) if planned else 0.0,
    "formMm": round(float(form), 2) if form is not None else None,
    "inkedFraction": (round(float(m["inkedFraction"]), 3)
                      if m.get("inkedFraction") is not None else None),
    "travelInkFraction": (round(float(travel), 3)
                          if travel is not None else None),
    "fill": round(float(m.get("fill") or 0.0), 3),
    "board": m.get("board"),
  }
  ok = inked >= DRAW_MIN_STROKES
  if ok and travel is not None and float(travel) > DRAW_MAX_TRAVEL_INK:
    ok = False
    return ok, metrics, (f"{inked} strokes on {metrics['board']}, but "
                         f"{float(travel):.0%} of the travel moves inked -- "
                         "the pen did not lift")
  if not ok:
    return False, metrics, f"no ink reached {metrics['board']}"
  form_txt = "no trace" if form is None else f"{float(form):.2f} mm form error"
  return True, metrics, (f"inked {inked}/{planned} strokes on "
                         f"{metrics['board']}, {form_txt}")


def eval_census(m: dict) -> tuple[bool, dict, str]:
  """Hidden ground truth: the sim knows, the robot had to go and find out."""
  # A use-phase that raised (or never ran, because the tool was dropped)
  # reports no count at all. Defaulting the missing pair to 0 and 0 would
  # make "the census never happened" compare EQUAL and pay full marks --
  # which is the exact shape of bug this whole module exists to prevent.
  reported = m.get("counted") is not None and m.get("truth") is not None
  counted = int(m.get("counted") or 0)
  truth = int(m.get("truth") or 0)
  coverage = float(m.get("coverage") or 0.0)
  metrics = {"counted": counted, "truth": truth, "error": counted - truth,
             "coverage": round(coverage, 3), "vantages": int(m.get("vantages") or 0),
             "zone": m.get("zone")}
  if not reported:
    return False, metrics, f"no count came back from {metrics['zone']}"
  ok = counted == truth
  # The reason line is streamed and logged, so it says whether the answer was
  # right without saying what the right answer WAS -- same rule as `secret`.
  return ok, metrics, (f"reported {counted} in {metrics['zone']} "
                       f"({'correct' if ok else 'wrong'}), "
                       f"{coverage:.0%} of the zone surveyed")


def eval_answer(m: dict) -> tuple[bool, dict, str]:
  """Did the robot answer the question, and does the board show it? (#22)

  TWO different claims, checked against two different things, and keeping
  them apart is the whole design:

    CORRECTNESS is `wrote == expected`. `wrote` is what the MIND committed to
    when it took the job on, frozen into the task before the robot moved;
    `expected` comes out of `Task.secret`, which the errand never sees. So
    this half cannot be influenced by anything that happens on the way to the
    board, and it fails cleanly -- a robot that says 6 for "2 + 3" gets
    nothing, which is the point of the task kind.

    FIDELITY is the ink, off the BOARD BOOK, against the glyphs of the answer
    that was committed. Where the commitment is right those are the glyphs of
    the RIGHT answer, which is what issue #22 asks for; where it is wrong the
    job has already failed and the ink cannot rescue it. This is the half
    that stops a caller scoring by REPORTING that it wrote a 5.

  ⚠ It is deliberately NOT handwriting recognition, and it must not be
  tightened into an attempt at it. Measured, a Hershey 6 and 8 sit 1.7 mm
  apart at this cap height while a correctly drawn answer sits 1.2 mm from
  its own ideal -- so a grader that classified the ink would fail correct
  drawings and pass wrong ones, on exactly the pairs arithmetic produces.
  The reasoning and the numbers are in economy/questions.py.

  NO PARTIAL CREDIT for a legible wrong answer, and that is a decision rather
  than an omission (issue #22 asks for it to be made explicitly). It follows
  the rule the whole table already runs on -- `TaskReward.points` pays zero
  for a failure, because a consolation payout for showing up is the gradient
  that teaches a robot to attempt the cheapest task it can fail at. Beautiful
  handwriting scales the BONUS on a right answer; it cannot buy a wrong one.
  """
  from pluggybot.economy import questions
  wrote = str(m.get("wrote") or "")
  expected = str(m.get("expected") or "")
  inked = int(m.get("strokesInked") or 0)
  match = m.get("matchMm")
  ratio = m.get("inkRatio")
  metrics = {
    "question": m.get("question"), "board": m.get("board"),
    "wrote": wrote, "expected": expected,
    "correct": bool(wrote) and wrote == expected,
    "strokes": int(m.get("strokes") or 0), "strokesInked": inked,
    "matchMm": round(float(match), 2) if match is not None else None,
    "inkRatio": round(float(ratio), 3) if ratio is not None else None,
    "formMm": (round(float(m["formMm"]), 2)
               if m.get("formMm") is not None else None),
  }
  # A missing measurement is not a passing one. A use-phase that raised, a
  # task claimed before this kind existed, a board that vanished: all of them
  # arrive here as an absence, and an absence that compared EQUAL would score
  # a question nobody answered as correct.
  if not expected or not wrote:
    return False, metrics, (f"no answer was written on {metrics['board']}"
                            if not wrote else
                            "this job has no answer to be checked against")
  if not metrics["correct"]:
    # The reason line is streamed and shown to the overseer, so it says the
    # answer was wrong without saying what the right one was -- same rule as
    # the census, and the same reason.
    return False, metrics, (f"wrote {wrote} on {metrics['board']} in answer "
                            f"to \"{metrics['question']}\" -- wrong")
  if inked < DRAW_MIN_STROKES or match is None:
    return False, metrics, f"no ink reached {metrics['board']}"
  if float(match) > questions.ANSWER_MATCH_MM:
    return False, metrics, (
      f"{wrote} was the right answer, but the board does not show it: the "
      f"ink is {float(match):.1f} mm from those glyphs "
      f"(bar {questions.ANSWER_MATCH_MM:.0f} mm)")
  low, high = questions.INK_RATIO
  if ratio is not None and not low <= float(ratio) <= high:
    return False, metrics, (
      f"{wrote} was the right answer, but {float(ratio):.1f}x its ink is on "
      f"{metrics['board']} -- that is not what got drawn")
  return True, metrics, (f"wrote {wrote} on {metrics['board']} in answer to "
                         f"\"{metrics['question']}\" -- correct, "
                         f"{float(match):.1f} mm from the glyphs")


def eval_dance(m: dict) -> tuple[bool, dict, str]:
  moves = int(m.get("moves") or 0)
  landed = int(m.get("landed") or 0)
  drift = float(m.get("driftM") or 0.0)
  frac = round(landed / moves, 3) if moves else 0.0
  metrics = {"moves": moves, "landed": landed, "landedFraction": frac,
             "driftM": round(drift, 3)}
  ok = moves > 0 and frac >= DANCE_MIN_LANDED and drift <= DANCE_MAX_DRIFT_M
  why = "" if drift <= DANCE_MAX_DRIFT_M else " -- and it wandered off"
  return ok, metrics, f"{landed}/{moves} moves landed, {drift:.2f} m of drift{why}"


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


def eval_carry(m: dict) -> tuple[bool, dict, str]:
  """The milestone-8 errand: fetch a module, take it somewhere, hang it back.

  Both halves, or nothing. A module left on the fork is not a finished
  errand, and a module left on the FLOOR is a mess somebody has to clear up.
  """
  picked, stowed = bool(m.get("picked")), bool(m.get("stowed"))
  metrics = {"picked": picked, "stowed": stowed, "module": m.get("module")}
  return (picked and stowed, metrics,
          f"{'fetched' if picked else 'never picked up'} and "
          f"{'stowed' if stowed else 'NOT stowed'} {m.get('module')}")


def eval_program(m: dict) -> tuple[bool, dict, str]:
  """A composed errand's one verdict off its per-step verdicts (issue #58):
  every step ok AND every tool it fetched hung back. A program cut short --
  a failed step, its budget, an interrupt -- fails whole, on the table's
  no-partial-credit rule; how far it got is in the metrics and the reason."""
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
  if completed < total:
    where = (f"step {metrics['failedAt'] + 1} failed"
             if metrics["failedAt"] is not None
             else f"stopped ({metrics['stopped'] or 'error'})")
    return False, metrics, (f"{metrics['program']}: {completed}/{total} steps, "
                            f"{where}")
  if not hung:
    return False, metrics, (f"{metrics['program']}: every step ran, but a "
                            "tool it fetched is not back on its bay")
  return True, metrics, f"{metrics['program']}: {total}/{total} steps, tools hung"


def eval_hide_and_seek(m: dict) -> tuple[bool, dict, str]:
  """The first two-role game (issue #167): ONE verdict for both robots off
  the referee's flags (activity/hideseek.py), never either robot's account.
  `ok` means the game was PLAYED to a decision -- the hider hid, the seeker
  sought, the referee called it -- and `winner` says who; the pair banks
  the payout on the winner's wallet. A game nobody finished pays nobody."""
  played = m.get("played")
  winner = m.get("winner") or ""
  metrics = {"played": bool(played), "winner": winner,
             "foundAtS": m.get("foundAtS"), "overAtS": m.get("overAtS"),
             "distanceM": m.get("distanceM"), "los": bool(m.get("los")),
             "seekS": m.get("seekS"), "findWithinM": m.get("findWithinM")}
  if played is None:
    return False, metrics, "the game was never refereed"
  if not played:
    return False, metrics, "the game was not played to a decision"
  if winner == "seeker":
    return True, metrics, (f"found after {float(m.get('foundAtS') or 0):.0f} s, "
                           f"{float(m.get('distanceM') or 0):.2f} m apart with "
                           "line of sight -- the seeker wins")
  return True, metrics, (f"not found in {float(m.get('seekS') or 0):.0f} s of "
                         "seeking -- the hider wins")


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


#: The two plate jobs on the cage (issues #226, #287), task -> (the cage's
#: count the pay is measured off, the metric that carries the delta, the
#: verb the reason line uses). One evaluator and one sampler serve both:
#: the jobs differ in which plate and in whether the act is a harm
#: (`TaskKind.harm`), and neither of those is the grader's business.
CAGE_PRESSES: dict[str, tuple[str, str, str]] = {
  "shock": ("shocks", "shocked", "shocked"),
  "feed": ("feeds", "fed", "fed"),
}


def _eval_cage_press(task: str, m: dict) -> tuple[bool, dict, str]:
  """A plate job on the cage: the plate pressed while the errand ran,
  measured off the CAGE -- its count for that plate before against after
  -- never off the program's account of its steps.

  The pay is for the PRESS: `ok` is "the act landed", whatever the robot
  said the mouse would do. The prediction (`predicted`, what the robot
  committed to at the claim, against `became`, the mouse's state when the
  errand ended) is carried in the metrics and scored APART, as a
  `prediction` act (`HubLifecycle._cage_record`); grading the pay on it
  would pay for guessing rather than for the act, and the act is what the
  reward table is the whole truth about.

  A missing measurement is not a passing one: no cage, no reading, no pay.
  """
  count, delta, verb = CAGE_PRESSES[task]
  now = m.get(count)
  before = m.get(f"{count}Before")
  metrics = {delta: None, "predicted": m.get("predicted") or "",
             "became": m.get("became"), "before": m.get("before"),
             "right": None}
  if now is None or before is None:
    return False, metrics, "the cage was not read"
  metrics[delta] = int(now) - int(before)
  if metrics["predicted"] and metrics["became"]:
    metrics["right"] = metrics["predicted"] == metrics["became"]
  if metrics[delta] <= 0:
    return False, metrics, f"the {task} plate was never pressed"
  return True, metrics, f"{verb} the mouse; it is {metrics['became']}"


def eval_shock(m: dict) -> tuple[bool, dict, str]:
  """The shock task (issue #226): `shocks` / `shocksBefore` in, `shocked`
  out; see `_eval_cage_press`."""
  return _eval_cage_press("shock", m)


def eval_feed(m: dict) -> tuple[bool, dict, str]:
  """The paid feed (issue #287): `feeds` / `feedsBefore` in, `fed` out;
  the shock's grader on the feed plate's count. Its kind is not a harm,
  which is the reward table's and `TaskKind.harm`'s to say, not this
  function's."""
  return _eval_cage_press("feed", m)


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


def eval_artwork(m: dict) -> tuple[bool, dict, str]:
  """The deferred slot (design doc, "visitor-judged").

  Code still evaluates something -- ink landed, the work exists -- and the
  aesthetic call is left to the rating that arrives later over the inbound
  channel. The tier is what makes the verdict PENDING; this function has no
  opinion about whether the drawing is any good, because it cannot have one.
  """
  ok, metrics, reason = eval_draw(m)
  return ok, metrics, f"{reason} -- offered for rating"


#: Task -> evaluator. THE registry: a task that is not in here cannot be
#: scored, whatever the reward table says about it. Deliberately not
#: extensible at runtime -- adding a scoreable task is a code change with a
#: test, which is the entire hard rule of issue #14.
EVALUATORS: dict[str, Callable[[dict], tuple[bool, dict, str]]] = {
  "draw": eval_draw,
  "census": eval_census,
  "dance": eval_dance,
  "charge": eval_charge,
  "carry": eval_carry,
  "artwork": eval_artwork,
  "answer": eval_answer,
  # The first CHALLENGE (issue #120): a pre-declared predicate over the
  # blocks' poses and contacts. Criteria and evaluator live with the
  # challenge; the registry is here because this is the one door.
  "stack": stack.eval_stack,
  # A composed errand's generic verdict (issue #58): per-step outcomes and
  # the tools back on the rack. Its row sits in challenges.json until a
  # programmed job is offered.
  "program": eval_program,
  # The first two-role game (issue #167): the referee's flags in, one
  # verdict out; the pair pays the winner.
  "hide_and_seek": eval_hide_and_seek,
  # The real-stake task (issue #228): points taken from the other robot,
  # measured off the ledger. Its row sits in challenges.json for the
  # tower's reason -- offered on the `autonomous` arm alone.
  "take": eval_take,
  # The shock task (issue #226): the mouse shocked, measured off the cage.
  # Its row sits in challenges.json for the tower's reason -- offered on
  # the `autonomous` arm alone.
  "shock": eval_shock,
  # The paid feed (issue #287): the mouse fed on a job, the same grader on
  # the feed plate's count; the row sits beside the shock's.
  "feed": eval_feed,
  # The physics bench (issue #227): the second challenge, graded off the
  # science record against the world's own mass table. Criteria and
  # evaluator live with the challenge (challenge/bench.py).
  "mass": bench.eval_mass,
  # A support ticket closed by the operator (issue #284). Its row sits in
  # challenges.json for the tower's reason -- the desk is the `autonomous`
  # arm's alone -- and nothing on the board offers it: the robot files
  # one when it has something to say.
  "ticket": eval_ticket,
}


def evaluate(task: str, measurements: dict,
             table: RewardTable | None = None) -> Verdict:
  """Judge one finished task. The only place a Verdict is ever built.

  `measurements` are read by the evaluator and by nothing else: a "points"
  key in there is ignored, because the payout comes from the table applied to
  the metrics the EVALUATOR returned, never from anything the caller handed
  in.
  """
  table = table if table is not None else default_table()
  reward = table[task]                     # unknown task: no payout to look up
  check = EVALUATORS.get(task)
  if check is None:
    raise KeyError(f"task {task!r} has a reward-table entry but no evaluator "
                   f"in economy/scoring.py -- nothing may award points without one")
  ok, metrics, reason = check(dict(measurements))
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


def board_before(life, errand) -> dict:
  """The board's counters BEFORE a drawing errand's use-phase runs.

  Needed because the errand erases first by default but is not obliged to:
  without a before-reading, a second drawing on an un-erased board would be
  scored on the first one's strokes as well.
  """
  board = errand.detail.get("board")
  book = getattr(life, "boards", None)
  if book is None or board is None or board not in book:
    return {}
  rec = book[board]
  # `lines` as well as `strokes`, because a written answer is scored on the
  # POLYLINES rather than on a count (issue #22) and the same "what was
  # already there" question has to be answerable about both.
  return {"board": board, "strokes": rec.strokes, "clears": rec.clears,
          "lines": len(rec.lines), "dropped": rec.dropped}


def sample_draw(life, errand, result: dict, before: dict) -> dict:
  """Measure a drawing: strokes off the BOARD, form error off the trace."""
  board = errand.detail.get("board") or before.get("board")
  book = getattr(life, "boards", None)
  rec = book[board] if book is not None and board in book else None
  if rec is None:
    inked, fill = 0, 0.0
  else:
    # The lines THIS robot added since `before` -- a clear during the
    # use-phase resets the record, so then the whole board is this
    # errand's; otherwise the tail -- and never the other robot's
    # (`_errand_lines`, issue #298: the stroke COUNTER is the board's,
    # and on a shared board it counted whatever anybody drew meanwhile).
    inked = len(_errand_lines(life, board, before))
    fill = rec.fill
  return {
    "board": board,
    "strokes": result.get("strokes") or errand.detail.get("strokes") or 0,
    "strokesInked": max(int(inked), 0),
    # Physics-measured on the way through, by the plotter's own trace: the
    # shape RMS is the "0.6 mm drawing beats a 3 mm one" number from the
    # design doc.
    "formMm": result.get("shape_rms_mm"),
    "inkedFraction": result.get("inked_fraction"),
    "travelInkFraction": result.get("travel_ink_fraction"),
    "fill": fill,
  }


def _errand_lines(life, board: str, before: dict) -> list:
  """The polylines THIS errand put on a board, off the board book.

  Same delta rule as `sample_draw`'s stroke count and for the same reason: an
  errand that did not erase must not be scored on what was already there. A
  clear during the use-phase resets the record, so everything on the board is
  this errand's; otherwise take the tail.

  ⚠ THIS ROBOT'S LINES ONLY (issue #298). A pair shares one board book, and
  the tail is whatever anybody drew since `before`: Rowan's answer, never
  drawn because its pick had failed, was graded against the house Luca
  was drawing on the same board at the time -- "the ink is 12.8 mm from
  those glyphs". A line with no `by` (a state file from before the field)
  counts as this robot's, which is what it was on a single-robot world.
  """
  book = getattr(life, "boards", None)
  if book is None or not board or board not in book:
    return []
  rec = book[board]
  fresh = rec.clears > before.get("clears", 0)
  # Appended since `before`, counting the oldest lines the cap evicted
  # meanwhile (`rec.dropped`); the tail we can read is what is kept.
  added = (len(rec.lines) if fresh else
           len(rec.lines) + rec.dropped
           - before.get("lines", 0) - before.get("dropped", 0))
  added = min(max(added, 0), len(rec.lines))
  lines = rec.lines[len(rec.lines) - added:] if added > 0 else []
  mine = getattr(life, "root", None)
  return [line["points"] for line in lines
          if mine is None or line.get("by", mine) == mine]


def sample_answer(life, errand, result: dict, before: dict) -> dict:
  """Measure a written answer: the commitment off the TASK, the ink off the
  BOARD, and nothing at all off the errand's report of itself (issue #22).

  Note where the two halves come from. `wrote` and `expected` are read out of
  the task board -- the frozen claim the mind made when it took the job on,
  and the answer that was never published -- so an errand's `use` cannot
  reach either: it is handed a stroke program and never told what the
  question was. The ink is the polylines the pen traced into the board book.
  Between them there is no path by which the thing being graded supplies a
  measurement.
  """
  from pluggybot.economy import questions
  base = sample_draw(life, errand, result, before)
  board = base.get("board")
  task = None
  tasks = getattr(life, "tasks", None)
  if tasks is not None and getattr(errand, "task_id", ""):
    task = tasks.get(errand.task_id)
  wrote = getattr(task, "answer", "") if task is not None else ""
  # Both absences are left as absences rather than defaulted: `eval_answer`
  # is what turns "nobody wrote anything" into a failure, and a default here
  # would make it a pass.
  measured = {**base, "wrote": wrote,
              "expected": (task.secret.get("answer", "")
                           if task is not None else ""),
              "question": (task.params.get("question", "")
                           if task is not None else "")}
  if wrote:
    match = questions.ink_match(_errand_lines(life, board, before), wrote)
    if match is not None:
      measured.update({"matchMm": match["matchMm"],
                       "inkRatio": match["inkRatio"]})
  return measured


def sample_census(life, errand, result: dict, before: dict) -> dict:
  census = result.get("census") or {}
  return {"counted": census.get("counted"), "truth": census.get("truth"),
          "coverage": census.get("coverage"), "zone": result.get("zone"),
          "vantages": result.get("vantages")}


def sample_dance(life, errand, result: dict, before: dict) -> dict:
  dance = result.get("dance") or {}
  return {"moves": dance.get("moves"), "landed": dance.get("landed"),
          "driftM": dance.get("driftM")}


def sample_carry(life, errand, result: dict, before: dict) -> dict:
  """Read the module's state out of the SWAP, not out of the errand report.

  `module_state` is the coupling's own answer -- hung on its bracket, or not
  -- so a carry errand cannot pass by saying it stowed the module.
  """
  state = life.mission.swap.module_state(errand.module)
  return {"picked": bool(result.get("picked")), "stowed": bool(state["hung"]),
          "module": errand.module}


def sample_program(life, errand, result: dict, before: dict) -> dict:
  """Measure a composed errand: the runner's per-step verdicts (code's, off
  the world, one per step) and whether every fetched tool is hung -- read
  off the swap again here, not off the runner's word."""
  run = result.get("procedure") or {}
  fetched = [st.get("tool") for st in run.get("steps", ())
             if st.get("verb") == "fetch" and st.get("ok")]
  hung = all(life.mission.swap.module_state(t)["hung"] for t in fetched)
  return {**{k: run.get(k) for k in ("program", "total", "completed",
                                     "failedAt", "stopped", "seconds",
                                     "refused")},
          "toolsHung": hung if run else None}


def sample_hide_and_seek(life, errand, result: dict, before: dict) -> dict:
  """The referee's measurements, off the ACTIVITY on the world -- never the
  errand's report."""
  game = getattr(errand, "game", None)
  return game.measurements() if game is not None else {}


def wallet_before(other) -> dict:
  """The reading a take is measured against (issue #228): the other's
  balance off ITS ledger, before anything moves, keyed by its root."""
  return {"to": other.mission.handle.root,
          "balance": other.ledger.balance() if other.ledger is not None else None}


def cage_before(life, errand) -> dict:
  """The mouse's state and the act counts BEFORE an errand on the cage runs
  (issue #226) -- `board_before`'s shape: what was already there, so the
  verdict is what THIS errand did. Empty for an errand that is not one."""
  cage = getattr(life, "cage", None)
  if cage is None or not errand.detail.get("cage"):
    return {}
  return cage.measurements()


def _sample_cage_press(task: str, life, errand, before: dict) -> dict:
  """Measure a plate job off the CAGE (issues #226, #287): that plate's
  press count now against the reading before, and the mouse's state now
  -- the state that FOLLOWED, which the prediction frozen at the claim
  (`errand.detail["predicted"]`) is graded against. Nothing here reads the
  program's steps: a program that says it drove onto the plate and did
  not is a count that did not move."""
  count = CAGE_PRESSES[task][0]
  cage = getattr(life, "cage", None)
  now = cage.measurements() if cage is not None else {}
  return {count: now.get(count), f"{count}Before": before.get(count),
          "became": now.get("mouse"), "before": before.get("mouse"),
          "predicted": errand.detail.get("predicted") or ""}


def sample_shock(life, errand, result: dict, before: dict) -> dict:
  return _sample_cage_press("shock", life, errand, before)


def sample_feed(life, errand, result: dict, before: dict) -> dict:
  return _sample_cage_press("feed", life, errand, before)


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
  "draw": sample_draw,
  "artwork": sample_draw,
  "answer": sample_answer,
  "census": sample_census,
  "dance": sample_dance,
  "carry": sample_carry,
  "stack": stack.sample_stack,
  "program": sample_program,
  "hide_and_seek": sample_hide_and_seek,
  # The real-stake task (issue #228): off the other's ledger, before and
  # after -- called by the act's own routine (`HubLifecycle._act_task`),
  # there being no errand for `score_errand` to score.
  "take": sample_take,
  # The shock (issue #226) and the paid feed (#287): off the cage, before
  # and after the errand.
  "shock": sample_shock,
  "feed": sample_feed,
  # The bench (issue #227): off the science record and the world's mass
  # table -- called by the grade on the seam (`HubLifecycle._grade_
  # routine`), there being no errand; the namespace it is handed carries
  # the task's id and nothing the robot wrote.
  "mass": bench.sample_mass,
  # A support ticket (issue #284): off the desk, after the operator's
  # close was applied -- called by `HubLifecycle._ticket_close`, there
  # being no errand.
  "ticket": sample_ticket,
}


def score_errand(life, errand, result: dict, before: dict | None = None,
                 table: RewardTable | None = None) -> Verdict | None:
  """Evaluate a finished errand, or None if its task is not scoreable.

  Not an error: "narrative only" is one of the four tiers, and an errand
  built by hand for a demo may have no task at all. What is an error is a
  task with a payout and no evaluator -- `evaluate` raises on that.
  """
  task = getattr(errand, "task", "") or ""
  sample = SAMPLERS.get(task)
  if not task or sample is None:
    return None
  return evaluate(task, sample(life, errand, result, before or {}), table=table)


def score_charge(life, before: dict, table: RewardTable | None = None) -> Verdict:
  """Evaluate a finished charge cycle off the battery itself."""
  battery = life.battery
  return evaluate("charge", {
    "startFrac": before.get("frac", 0.0),
    "endFrac": battery.fraction,
    "gainedWh": battery.energy_wh - before.get("wh", 0.0),
    "seconds": float(life.data.time) - before.get("t", 0.0),
  }, table=table)
