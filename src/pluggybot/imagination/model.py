"""The robot's model of what is in front of it (issue #466, stage 3): an
author's structure (`author.py`), placed where depth put the object, its
unknowns fitted to the probe's record (`fit.py`), judged, and sent back
for another structure where the fit left too much. Code runs the rounds;
the mind decides nothing (when to imagine is demo 4's).

  rounds  `MAX_ROUNDS` at most: the first answer and four revisions, each
          answer repaired up to `author.MAX_REPAIRS` times where it did not
          parse. ⚠ THE AUTHOR IS NEVER TOLD THE CAP (a test reads every
          text it is sent), so a run capped at five asks as one capped at
          three until its fourth round, and a batch reads its passes by
          round three and by round five off the same runs (#481)
  poor    a phase whose force left over passes its bar (`Phase.bar`, the
          caller's: what the language's best usually leaves there)
  kept    of the rounds that passed every bar, the one whose fit left the
          least over what was fitted (`kept_of`): a revision can be worse
          than what it revised
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pluggybot.imagination.author import Author, Turn, in_map
from pluggybot.imagination.fit import Fitted, Unfittable, fit, rms
from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import Readings
from pluggybot.imagination.scene import Refused, Template

#: The rounds a model takes at most: at three, every pass of #480's batch
#: came in a revision, and none in the first answer.
MAX_ROUNDS = 5
#: A poor phase's trace is shown in means this long, s.
TRACE_S = 0.5
#: The judge's means, s: the residual's own (`fit.BIN_S`).
MEAN_S = 0.1


@dataclass(frozen=True)
class Phase:
  """A stretch of the record as an author is told it: its `name`, its
  `label`, its rows [first, past the last) and the force a fit may leave
  there before it is poor (`bar`, N at the tool, RMS of `MEAN_S` means;
  None: reported, never judged). A `whole` spans others, and its trace is
  theirs."""
  name: str
  label: str
  rows: tuple[int, int]
  bar: float | None
  whole: bool = False


def _means(x: np.ndarray, per: int) -> np.ndarray:
  n = len(x) // per
  return x[:n * per].reshape(n, per, *x.shape[1:]).mean(axis=1)


def judge(record: Record, readings: Readings, phases: list[Phase],
          felt: np.ndarray | None = None) -> list[dict]:
  """Each phase: the force the readings put on the tool apart from the
  record's (RMS of `MEAN_S` means), whether it is past its bar, and its
  trace -- `TRACE_S` means of the force the REAL object put on the jaws
  minus the imagined one's (toward the object, up), and, given what the
  real one put on them (`felt`, rows of the tool's force), that and the
  imagined one's side by side (`felt`)."""
  from pluggybot.legs.imagined import force_apart
  apart = force_apart(record.sensed, readings.sensed)
  per, per_trace = max(1, round(MEAN_S / record.dt)), max(1, round(TRACE_S / record.dt))
  out = []
  for p in phases:
    a, b = p.rows
    f = apart[a:b]
    got = rms(_means(f, per))
    trace = [(round(i * TRACE_S, 2), (round(float(-m[0]), 3), round(float(-m[1]), 3)))
             for i, m in enumerate(_means(f, per_trace))]
    row = {"phase": p.name, "label": p.label, "rmsN": got, "barN": p.bar,
           "poor": p.bar is not None and got > p.bar, "whole": p.whole, "trace": trace}
    if felt is not None:
      real = _means(np.asarray(felt)[a:b], per_trace)
      imagined = real + _means(f, per_trace)
      row["felt"] = [(round(i * TRACE_S, 2), (round(float(r[0]), 3), round(float(r[1]), 3)),
                      (round(float(m[0]), 3), round(float(m[1]), 3)))
                     for i, (r, m) in enumerate(zip(real, imagined))]
    out.append(row)
  return out


def felt_by_phase(record: Record, phases: list[Phase], felt: np.ndarray) -> list[dict]:
  """What the first turn is shown of `felt` (rows of the force the real
  object put on the jaws, toward it and up): each phase's `TRACE_S` means,
  by its label; a `whole` is the stretches it spans, never shown twice."""
  per = max(1, round(TRACE_S / record.dt))
  out = []
  for p in phases:
    if p.whole:
      continue
    a, b = p.rows
    out.append({"label": p.label,
                "felt": [(round(i * TRACE_S, 2), (round(float(m[0]), 3), round(float(m[1]), 3)))
                         for i, m in enumerate(_means(np.asarray(felt)[a:b], per))]})
  return out


def report(fitted: Fitted | None, judged: list[dict], failed: str = "") -> dict:
  """What a revision is shown: the values found, which ended at an end of
  their range (and which end) or were never seen, and the phases judged
  (or why no fit could be read)."""
  return {"values": {} if fitted is None else dict(fitted.values),
          "atEnd": {} if fitted is None else dict(fitted.at_end),
          "unseen": [] if fitted is None else list(fitted.unseen),
          "phases": judged, "failed": failed}


@dataclass
class Round:
  """One structure: the author's settled `turn`, its template in the map
  (`template`), its fit, and the phases judged -- or why there is none."""
  turn: Turn
  template: Template | None = None
  fitted: Fitted | None = None
  judged: list = field(default_factory=list)
  error: str = ""

  @property
  def poor(self) -> bool:
    return self.fitted is None or any(p["poor"] for p in self.judged)

  def as_dict(self) -> dict:
    return {"turn": self.turn.as_dict(), "error": self.error,
            "fitted": None if self.fitted is None else self.fitted.as_dict(),
            "document": None if self.fitted is None else self.fitted.document,
            "judged": [{k: v for k, v in p.items() if k != "trace"} for p in self.judged],
            "poor": self.poor}


@dataclass
class Model:
  """The rounds a model took, the one `kept` (its index, None where no
  round was fitted), and every turn of the conversation, repairs too."""
  rounds: list[Round]
  kept: int | None
  turns: list[Turn]

  @property
  def round(self) -> Round | None:
    return None if self.kept is None else self.rounds[self.kept]

  @property
  def document(self) -> dict | None:
    """The model: the kept round's fitted document, in the map."""
    r = self.round
    return None if r is None else r.fitted.document

  @property
  def tokens(self) -> tuple[int, int]:
    return (sum(t.tokens[0] for t in self.turns), sum(t.tokens[1] for t in self.turns))

  @property
  def revisions(self) -> int:
    return sum(1 for t in self.turns if t.kind == "revise")

  def as_dict(self) -> dict:
    return {"kept": self.kept, "revisions": self.revisions,
            "repairs": sum(1 for t in self.turns if t.kind == "repair"),
            "retries": sum(1 for t in self.turns if t.kind == "retry"),
            "tokensIn": self.tokens[0], "tokensOut": self.tokens[1],
            "rounds": [r.as_dict() for r in self.rounds],
            "turns": [t.as_dict() for t in self.turns]}


def imagine(author: Author, record: Record, phases: list[Phase], fit_rows: slice,
            sizes: dict, origin_mm, yaw_deg: float, did: str, picture: bytes | None,
            pool, rounds: int = MAX_ROUNDS, felt: np.ndarray | None = None,
            **fit_kw) -> Model:
  """The robot's model of the object its `record` probed (the module
  docstring): `sizes` are what its author is shown of the object's frame,
  which sits at `origin_mm` turned `yaw_deg` in the map; `fit_rows` are the
  rows fitted, `pool` the workers they are rolled out in; `felt` what the
  real object put on the jaws, shown phase by phase in the first turn
  (#481: the weight and a catch letting go are in it) and beside a poor
  fit's own."""
  out: list[Round] = []
  shown = None if felt is None else felt_by_phase(record, phases, felt)
  turn = author.first(sizes, did, picture, felt=shown)
  for r in range(rounds):
    if r:
      last = out[-1]
      turn = author.revise(report(last.fitted, last.judged, last.error))
    rnd = Round(turn=turn)
    out.append(rnd)
    if turn.template is None:
      rnd.error = turn.error or "its document was refused: " + "; ".join(turn.refused)
      break
    try:
      rnd.template = in_map(turn.template, origin_mm, yaw_deg)
      rnd.fitted = fit(rnd.template, record, fit_rows, pool, **fit_kw)
    except (Refused, Unfittable) as e:
      rnd.error = f"every simulation of the document went unstable or was refused: {e}"
      continue
    rnd.judged = judge(record, rnd.fitted.readings, phases, felt)
    if not rnd.poor:
      break
  return Model(rounds=out, kept=kept_of(out), turns=list(author.turns))


def kept_of(rounds: list[Round]) -> int | None:
  """The round kept: of those that passed every bar, the one that left
  least over what was fitted; where none passed, the least of all. ⚠ The
  least over the fitted rows alone kept a round its take had failed (the
  review)."""
  fitted = [i for i, r in enumerate(rounds) if r.fitted is not None]
  passed = [i for i in fitted if not rounds[i].poor] or fitted
  return min(passed, key=lambda i: rounds[i].fitted.rms) if passed else None
