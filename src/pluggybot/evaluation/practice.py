"""Graded practice (issue #493, stages 2 and 3): a mind writes a procedure
for the lid (`mind.practice`), the humanoid runs it from the set-out
(`ProcedureLife`), the world grades the try through stage 1's door
(`evaluation.attempts`), the mind is shown how it went and writes again --
attempts 1 to `ATTEMPTS`, each graded and unpaid, a fresh set-out each.

  ProcedureLife    what a procedure runs in inside an attempt: the body,
                   its model and data, the reports its verbs leave, no
                   interrupt, and what it said
  procedure_skill  a compiled procedure as an attempt's skill
  practise         one chest: the conversation and its attempts, until one
                   passes or the attempts run out; a pass ADMITS the
                   procedure to the robot's library (`Library.admit`)
  JOB              the job as its grader measures it, never how

⚠ A FRESH ROBOT A CHEST: nothing carries from one chest to the next -- no
library, no conversation -- so a chest's curve is its own robot's.
"""

from __future__ import annotations

import math
import time

from pluggybot.challenge import lid
from pluggybot.evaluation import attempts as at
from pluggybot.procedure import lang
from pluggybot.procedure.steps import WorldFacts

#: Attempts a chest, the most: stage 1's curve, S@1 to S@5.
ATTEMPTS = 5
#: The tag the knob carries (`rack.tags.CHEST_TAG_IDS`): the stated subsidy.
KNOB_TAG = 53
#: The head camera's picture the mind is shown, pixels.
PICTURE_W, PICTURE_H = 640, 480

#: The job, as its grader measures it (`challenge/lid.py`), and nothing of
#: how: what is in front of the robot, the stated subsidy, and the criteria.
JOB = f"""In front of you, on a counter, is a box with a lid, shut. A knob hangs
from its front, and the knob carries tag {KNOB_TAG} on every face.

Open the lid -- past {math.degrees(lid.OPEN_RAD):.0f} degrees from shut -- then set
it back down shut, and let go of it. While it is open more than
{math.degrees(lid.SHUT_RAD):.0f} degrees the lid must never come down faster than
{lid.LOWER_RAD_S:.1f} rad/s: let fall, it is not set down. The world grades
the job when your procedure ends and again {lid.HOLD_S:.0f} s later: the lid
must be shut by then, and nothing of yours may touch the lid, its knob or its
handle. A try may take {{limit:.0f}} s at most; a fall loses you the time it
takes to get up."""


class ProcedureLife:
  """What a procedure's verbs reach in an attempt (`procedure.lang`'s
  `life`): the body, its model and data, the reports its verbs leave for
  `read` (`axes.reported`), no interrupt, and the lines it said."""

  def __init__(self, body) -> None:
    self.body = body
    self.model, self.data = body.model, body.data
    self.reports: dict = {}
    self.said: list[str] = []
    self.step_now = None
    self.step_until = None
    self.aborting = False

  def interrupted(self) -> bool:
    return False

  def _say(self, text: str, detail: str = "", **_) -> None:
    self.said.append(text)


def procedure_skill(proc: lang.Procedure | None, facts: WorldFacts, out: dict):
  """`proc` as an attempt's skill (`attempts.attempt`): run from the
  set-out, its result -- every verb and what it said, its locals -- left in
  `out["ran"]` and its verb calls in `out["steps"]`. None stands still: a
  try with no procedure that compiled."""
  def skill(body):
    if proc is None:
      out["steps"] = 0
      return {"ok": False, "why": "no procedure"}
      yield
    life = ProcedureLife(body)
    ran = yield from lang.run_procedure_routine(life, proc, facts)
    out["ran"], out["steps"] = ran, int(ran.get("total", 0))
    return {"ok": bool(ran.get("ok")),
            "why": ran.get("stopped") or ("ran to its end" if ran.get("ok") else "a verb failed")}
  return skill


def head_picture(task: at.Task) -> bytes:
  """The head camera's picture at `task`'s set-out, a JPEG: what the robot
  sees as it stands before its first try."""
  import io
  import mujoco
  from PIL import Image
  world = at.set_out(task)
  try:
    cam = world.body.head_camera
    renderer = mujoco.Renderer(world.model, PICTURE_H, PICTURE_W)
    try:
      renderer.update_scene(world.data, camera=cam)
      pixels = renderer.render()
    finally:
      renderer.close()
  finally:
    world.close()
  buf = io.BytesIO()
  Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
  return buf.getvalue()


def practise(chest: int, practice, attempts: int = ATTEMPTS, *, library=None,
             picture: bool = True, budget_s: float = at.BUDGET_S,
             info: dict | None = None, build=at.set_out) -> list[dict]:
  """One chest, a fresh robot: the mind's procedure tried from the set-out,
  graded, and shown back (`mind.practice.report_text`), up to `attempts`
  times, ending at the first pass -- which `library` ADMITS on its sealed
  verdict (`Library.admit`). Every attempt's row (`attempts.attempt`'s)
  carries the procedure it ran, the answers it took and what they cost.
  `build` makes a set-out (`attempts.attempt`'s: a test's own world)."""
  from pluggybot.mind.practice import report_text
  task = at.Task(chest=chest, body="humanoid", mount="counter")
  rows: list[dict] = []
  shown = head_picture(task) if picture else None
  since = 0
  turn = practice.first(shown)
  for n in range(1, attempts + 1):
    spent = practice.spent(since)
    since = len(practice.turns)
    out: dict = {}
    sealed: list = []
    t0 = time.time()
    row = at.attempt(task, procedure_skill(turn.procedure, practice.facts, out), n,
                     budget_s=budget_s, sealed=sealed, build=build,
                     info={**(info or {}), "process": "practice",
                           "skill": turn.procedure.name if turn.procedure else None,
                           "tokens": spent["tokensIn"] + spent["tokensOut"], **out})
    row["steps"] = out.get("steps")
    row["practice"] = {**spent, "turns": [t.as_dict() for t in practice.turns[-spent["calls"]:]]
                       if spent["calls"] else [], "ran": _trimmed(out.get("ran")),
                       "flightWallS": round(time.time() - t0, 1)}
    if row["verdict"]["ok"] and library is not None and turn.procedure is not None:
      name = turn.procedure.name
      if name in library.entries:
        library.undefine(name)
      library.admit(name, turn.procedure.source, sealed[0])
      row["admitted"] = name
    rows.append(row)
    if row["verdict"]["ok"] or n == attempts:
      break
    turn = practice.after(report_text(n, out.get("ran"), row["verdict"], row["simS"], row["cut"],
                                      attempts - n))
  return rows


def _trimmed(ran: dict | None) -> dict | None:
  """A run's record for the row: every step but its trace."""
  if ran is None:
    return None
  return {**ran, "steps": [{k: v for k, v in s.items() if k != "trace"}
                           for s in ran.get("steps", [])]}
