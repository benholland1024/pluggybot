"""Questions the house asks, and how an answer is checked (issue #22).

The first task kind with a RIGHT ANSWER. A `whiteboard_answer` job poses a
short question -- "2 + 3" -- and the robot discharges it by fetching the pen
and writing the answer on a named board. Everything about getting there
already existed: the drawing errand, Hershey lettering, the evaluators, the
ledger. What is new is the question, the answer, and the two different things
that have to be true for the job to have been done.

  THE ARITHMETIC IS THE MIND'S JOB. The answer comes from whatever is
  deciding for the robot (`hub/overseer.py`), travels with the CLAIM, and is
  frozen into the task before a wheel turns. Code never computes it -- a sim
  that answered its own question would be marking its own homework, and the
  whole point of the kind is that a weaker backend (issue #19) gets it wrong
  where a better one does not, visibly, on a whiteboard.

  ...WHICH MEANS A ROBOT WITH NO MIND CANNOT TAKE THE JOB. The scripted
  rotation leaves a question standing and it lapses, honestly, as `expired`.
  That is not a gap: "nobody who could answer it came past" is a true thing
  about a robot's day, and the wire has always been able to say it.

  THE ANSWER IS NEVER PUBLISHED. It lives in `Task.secret`, which reaches
  neither the telemetry stream nor the model's context by any path -- the
  same treatment the census's ground truth gets, and for the same reason:
  this stream is read by the site AND fed back to the overseer.

## Why the ink is a FIDELITY check and not handwriting recognition

The obvious design is to read the board: render the correct answer's glyphs,
compare them to the polylines the pen actually inked, and call it right or
wrong. MEASURED, THAT DOES NOT WORK, and it fails quietly enough to be worth
writing down. Symmetric mean nearest-neighbour distance between Hershey
`futural` digits, rendered at the cap heights this robot can draw:

    cap      6 vs 8     6 vs 5     the robot's own form error
    18 mm    0.55 mm    0.86 mm    1.10 mm  (multi-stroke, measured #11)
    55 mm    1.70 mm    2.57 mm    1.10 mm
    75 mm    2.31 mm    3.44 mm    1.10 mm

A 6 and an 8 are closer to each other than a correctly drawn 6 is to its own
ideal, at every size the 110 mm carriage can reach. Tolerance-based coverage
is no better -- at a 2 mm tolerance a drawn 6 covers 97 % of an 8. So a
grader that classified the ink would fail correct drawings and pass wrong
ones, roughly at random, on exactly the pairs a question is most likely to
produce.

So the two questions are asked separately, of the two things that can
actually answer them:

  CORRECTNESS is decided against the committed answer, which the mind gave
  and cannot revise. Exact, unambiguous, and impossible to fake -- the
  errand's `use` never sees it.

  FIDELITY is decided off the BOARD BOOK: the ink has to match the glyphs of
  the answer that was committed. That is the check the acceptance criteria
  are really about -- a caller that draws a house, or nothing, or scribbles,
  cannot score by reporting that it wrote a 5 -- and where the commitment is
  correct it is literally "the inked polylines against the expected glyphs
  for the correct answer".

`ANSWER_MATCH_MM` is therefore set to catch WRONG WORK, not wrong glyphs, and
`test_questions.py` pins both halves of that claim.

## The bank is data

`questions.json` is a flat list of questions and their answers, and there is
deliberately NO expression evaluator here: a data file that can compute is a
data file that can be made to compute something else, and the bank is the one
thing in this module a deploy is expected to edit. Adding a question is a
JSON edit and a restart ($PLUGGY_QUESTIONS to point somewhere else), exactly
as re-pricing a task is (`hub/rewards.json`).

Answers are numeric and at most `MAX_ANSWER` characters, and that is a
MACHINE limit rather than a taste one: the pen's usable line is 100 mm wide,
two digits at a legible 50 mm cap measure 95 mm, and three do not fit at all.
`QuestionBank.load` refuses a question whose answer the robot could not write.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pluggybot.hub import hershey

BANK_PATH = Path(__file__).with_name("questions.json")
BANK_ENV = "PLUGGY_QUESTIONS"
BANK_VERSION = 1

#: Cap height an answer is written at, m. Chosen against the pen's own line
#: width rather than for looks: `hershey.text_width` puts two digits at this
#: cap at 95.2 mm, and `strokes.TEXT_WIDTH` -- the carriage travel less the
#: re-zero margin -- is 100 mm. At 55 mm they measure 104.8 and `fit_text`
#: would silently shrink the whole answer instead.
ANSWER_CAP = 0.050
#: ...and the widest line, which is `strokes.TEXT_WIDTH`. Duplicated rather
#: than imported because `hub/strokes.py` reaches MuJoCo through
#: `hub/drawing.py`, and this module is on the evaluator's path, which is
#: pure. `test_questions.py` asserts the two agree.
ANSWER_WIDTH = 0.100
#: Characters an answer may have. See the module docstring -- three do not
#: fit, so a question with a three-digit answer is one the robot cannot be
#: asked.
MAX_ANSWER = 2
#: What an answer may be MADE of. Digits only, for the same reason: the
#: alphabet is what fits, not what is expressible. A yes/no question would
#: need "YES" (three glyphs) or a convention nobody watching would read.
ANSWER_ALPHABET = "0123456789"
#: Longest question text. A question is a sentence somebody reads off a
#: marker on a website; `hub/inbox.py` caps a visitor's suggestion at 280 for
#: the same reason, and a question may come from one (issue #23).
MAX_QUESTION = 200

#: How far the ink may sit from the glyphs of the answer that was committed,
#: mm, before the board is not showing that answer.
#:
#: ⚠ A FIDELITY BAR, NOT A LEGIBILITY ONE -- see the module docstring. MEASURED
#: with the real pen on a real board (`scripts/answer_spike.py`), which is the
#: only way it could have been set: the first guess was 8.0 mm from synthetic
#: renderings, and the robot figure below goes straight through it.
#:
#:     a correctly written answer  ("5", "12")          0.8 - 1.2 mm
#:     the `robot` figure, drawn instead of a "5"       5.1 mm
#:
#: 5.1 mm rather than the 20 mm a glance at the two figures suggests, because
#: a busy figure covers the glyph it is standing in for -- the one-way
#: glyph-to-ink distance is 1.5 mm, and only the ink-to-glyph direction
#: notices. Hence a symmetric figure AND `INK_RATIO`, which the same drawing
#: fails four times over.
ANSWER_MATCH_MM = 4.0
#: ...and how much ink an answer may be written with, as a multiple of what
#: those glyphs need. The other half of the fidelity check and the cheaper
#: half: measured, the `robot` figure is 2.76x the ink of the "5" it was
#: drawn instead of, while a correctly written answer is 0.92x (a decimated
#: polyline is slightly shorter than the arc it stands for). Loose in both
#: directions, because how much of each stroke found the board varies.
INK_RATIO = (0.6, 1.8)
#: Metres between resampled points when two figures are compared. Fine
#: relative to a 50 mm glyph and coarse relative to the plotter's own ~0.04 mm
#: sample spacing: this is a shape comparison, not a trace.
MATCH_STEP = 0.0005


def clean_answer(text) -> str:
  """Whatever came back from the mind -> something the pen can write.

  Sanitising, on the inbox's terms and for the inbox's reason: this is the
  ONE string a model chooses that ends up drawn a metre wide on a wall a
  stranger is watching. `hub/overseer.py` keeps `text` off the figure menu
  precisely because it takes arbitrary caller text -- an answer is where
  that lands safely, and it is safe because of this function rather than
  because of anything the model was asked to do.

  Everything outside `ANSWER_ALPHABET` is dropped rather than escaped, and
  the result is truncated to `MAX_ANSWER`. An answer that survives as "" is
  no answer, and the claim is refused.
  """
  raw = "" if text is None else str(text)
  kept = [c for c in raw.strip().upper() if c in ANSWER_ALPHABET]
  return "".join(kept[:MAX_ANSWER])


def answer_strokes(text: str) -> tuple[tuple[tuple[float, float], ...], ...]:
  """The polylines that WRITE an answer, in board-local metres.

  The single implementation, used twice: `hub/strokes.py` registers it as the
  `answer` program so the plotter can draw one, and the evaluator renders the
  same call to compare against the ink. Two renderings that could disagree
  would be a grader marking against a figure the robot was never asked to
  draw.

  `-x` is the board's lateral flip (`hub/strokes.py`: +lat is the viewer's
  LEFT, so text advances toward -lat). It is applied here rather than by the
  caller so that the reference and the drawing cannot end up mirrored
  relative to each other -- a fault no symmetric diagnostic can see.
  """
  clean = clean_answer(text)
  if not clean:
    raise ValueError(f"{text!r} is not an answer this pen can write "
                     f"(allowed: {ANSWER_ALPHABET}, at most "
                     f"{MAX_ANSWER} characters)")
  polys = hershey.layout(clean, ANSWER_CAP, max_width=ANSWER_WIDTH,
                         align="center")
  return tuple(tuple((-x, y) for x, y in s) for s in polys)


# ---- the bank (data) ---------------------------------------------------------


@dataclass(frozen=True)
class Question:
  """One thing the house can ask, and what the answer is.

  `answer` is the half that never leaves: it goes into `Task.secret`, and
  from there into nothing at all. The task carries `ask` in its `params` and
  in its description, because a question is a work order -- exactly what the
  honesty rule says a network may carry (docs/TaskPattern.md, issue #24).
  """

  id: str
  ask: str
  answer: str
  note: str = ""

  @classmethod
  def from_json(cls, spec: dict) -> "Question":
    qid = str(spec["id"]).strip()
    ask = " ".join(str(spec["ask"]).split())[:MAX_QUESTION]
    raw = str(spec["answer"]).strip()
    answer = clean_answer(raw)
    if not qid or not ask:
      raise ValueError(f"question {qid!r} has no id or nothing to ask")
    if answer != raw.upper():
      # Refused rather than silently trimmed. A bank whose answer is "12.5"
      # or "SEVEN" would offer a job the robot is structurally unable to get
      # right, and it would look like a robot that is bad at arithmetic.
      raise ValueError(
        f"question {qid!r} answers {raw!r}, which this pen cannot write -- "
        f"answers are at most {MAX_ANSWER} characters from "
        f"{ANSWER_ALPHABET!r} (hub/questions.py: it is the carriage's "
        f"100 mm line, not a style rule)")
    return cls(id=qid, ask=ask, answer=answer,
               note=str(spec.get("note", "")).strip())


class QuestionBank:
  """Every question this world can ask. Loaded from JSON, never written."""

  def __init__(self, questions, version: int = BANK_VERSION,
               path: Path | None = None) -> None:
    self.questions: tuple[Question, ...] = tuple(questions)
    self.version = version
    self.path = path
    self.by_id = {q.id: q for q in self.questions}

  @classmethod
  def load(cls, path: str | os.PathLike | None = None) -> "QuestionBank":
    """Read the bank. `path`, else $PLUGGY_QUESTIONS, else the shipped file."""
    target = Path(path or os.environ.get(BANK_ENV) or BANK_PATH)
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if version != BANK_VERSION:
      # No backward-compatible read, exactly as for the reward table: this is
      # a small hand-edited document, and asking a mission's worth of
      # questions out of a shape this build does not understand is worse than
      # refusing to start.
      raise ValueError(f"{target}: question bank version {version}, "
                       f"expected {BANK_VERSION}")
    questions = [Question.from_json(spec) for spec in doc["questions"]]
    if not questions:
      raise ValueError(f"{target}: a question bank with no questions in it")
    seen = [q.id for q in questions]
    if len(set(seen)) != len(seen):
      raise ValueError(f"{target}: duplicate question ids "
                       f"({', '.join(sorted({q for q in seen if seen.count(q) > 1}))})")
    return cls(questions, version=version, path=target)

  def __len__(self) -> int:
    return len(self.questions)

  def __getitem__(self, qid: str) -> Question:
    try:
      return self.by_id[qid]
    except KeyError:
      raise KeyError(f"no question {qid!r} in the bank "
                     f"(have: {', '.join(sorted(self.by_id))})") from None

  def pick(self, n: int) -> Question:
    """The nth question, wrapping. DETERMINISTIC, and it has to be.

    Rotation on a counter rather than a random draw, for the reason
    `overseer._fill` rotates: a world that asked a different question every
    run is a mission test that is a different test every run. The counter the
    caller passes is the task board's own sequence number, which survives a
    restart -- so a deployed robot works through the bank instead of being
    asked the same thing every morning.
    """
    return self.questions[int(n) % len(self.questions)]


_default: QuestionBank | None = None


def default_bank() -> QuestionBank:
  """The process-wide bank, loaded once. `reload_bank()` to re-read."""
  global _default
  if _default is None:
    _default = QuestionBank.load()
  return _default


def reload_bank(path: str | os.PathLike | None = None) -> QuestionBank:
  global _default
  _default = QuestionBank.load(path)
  return _default


# ---- comparing ink to glyphs -------------------------------------------------


def _resample(polys, step: float = MATCH_STEP) -> np.ndarray:
  """A list of polylines -> points spaced `step` apart along the ink.

  By ARC LENGTH, not by index. The pen slows into corners and the board book
  decimates by distance, so index sampling would weight two figures'
  corners and straights differently and the comparison would depend on how
  fast the robot was going.
  """
  pts: list[tuple[float, float]] = []
  for s in polys:
    poly = [(float(y), float(z)) for y, z in s]
    if len(poly) < 2:
      continue
    for a, b in zip(poly, poly[1:]):
      d = math.dist(a, b)
      n = max(int(d / step), 1)
      for k in range(n):
        f = k / n
        pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    pts.append(poly[-1])
  return np.array(pts, dtype=float) if pts else np.empty((0, 2))


def _nearest(pts: np.ndarray, polys) -> tuple[np.ndarray, np.ndarray]:
  """(distance, closest point) from each point to the nearest SEGMENT.

  Per polyline, never over a concatenation: joining the end of one stroke to
  the start of the next invents a segment nobody drew, and ink measured
  against it scores as a good drawing. Same rule and same reason as
  `PenPlotter._nearest`, which this is deliberately a copy of -- that one
  lives next to MuJoCo and the evaluator's path may not import it.
  """
  a = np.concatenate([np.asarray(p, dtype=float)[:-1] for p in polys
                      if len(p) >= 2])
  b = np.concatenate([np.asarray(p, dtype=float)[1:] for p in polys
                      if len(p) >= 2])
  seg = b - a
  denom = np.maximum((seg * seg).sum(1), 1e-12)
  rel = pts[:, None, :] - a[None, :, :]
  t = np.clip((rel * seg[None]).sum(2) / denom[None], 0.0, 1.0)
  proj = a[None] + t[..., None] * seg[None]
  d = np.linalg.norm(pts[:, None, :] - proj, axis=2)
  i = d.argmin(1)
  rows = np.arange(len(pts))
  return d[rows, i], proj[rows, i]


def _ink_length(polys) -> float:
  """Metres of ink in a set of polylines. `StrokeProgram.ink_length` for
  something that is not a program -- the board book stores plain lists."""
  return sum(math.dist(a, b) for s in polys for a, b in zip(s, s[1:]))


def ink_match(drawn, answer: str) -> dict | None:
  """How closely the ink on a board shows `answer`. None if nothing is there.

  `drawn` is the polylines the pen actually inked, off the board book;
  `answer` is what was committed. Returns `matchMm` -- the symmetric mean
  nearest-neighbour distance after a translation-only fit -- plus the two
  one-way numbers it is the worse of.

  Three deliberate choices:

  SYMMETRIC. One-way distance from the ink to the glyphs would pass a robot
  that drew the top half of a 5 and stopped; from the glyphs to the ink would
  pass one that scribbled over the whole board. The reported figure is the
  worse of the two, so missing ink and extra ink both count.

  TRANSLATION ONLY. The plotter centres a figure on where the pen actually
  is, and the base parks to a few centimetres, so WHERE on the board the
  answer landed is not something to mark it down for. Rotation and scale are
  NOT fitted -- an answer drawn sideways or half-size is a different mark,
  and fitting them out would be the grader helping.

  NONE, NOT ZERO, for a board with no ink on it. A missing measurement is not
  a passing one (hub/scoring.py), and the evaluator is what turns the absence
  into a failure.
  """
  ink = [tuple(tuple(float(v) for v in p) for p in line)
         for line in (drawn or ()) if len(line) >= 2]
  if not ink:
    return None
  ref = answer_strokes(answer)
  ink_pts, ref_pts = _resample(ink), _resample(ref)
  if not len(ink_pts) or not len(ref_pts):
    return None
  # Translation-only ICP, exactly as `PenPlotter.error_stats` decomposes a
  # figure's offset from its shape: six passes is where that one converges
  # and this is the same problem at the same scale.
  offset = np.zeros(2)
  for _ in range(6):
    _, proj = _nearest(ink_pts + offset, ref)
    step = (proj - (ink_pts + offset)).mean(axis=0)
    offset = offset + step
    if np.linalg.norm(step) < 1e-6:
      break
  to_ref, _ = _nearest(ink_pts + offset, ref)
  to_ink, _ = _nearest(ref_pts - offset, ink)
  glyph_m = _ink_length(ref)
  return {
    "matchMm": round(float(max(to_ref.mean(), to_ink.mean())) * 1000, 3),
    "inkToGlyphMm": round(float(to_ref.mean()) * 1000, 3),
    "glyphToInkMm": round(float(to_ink.mean()) * 1000, 3),
    "offsetMm": round(float(np.linalg.norm(offset)) * 1000, 3),
    # How MUCH was written, against how much the answer takes. Cheap, and it
    # catches the two faults the distance is weakest on: a figure that is
    # merely busy enough to cover the glyph, and an answer the pen only got
    # half of.
    "inkRatio": round(_ink_length(ink) / glyph_m, 3) if glyph_m else None,
  }
