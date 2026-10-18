"""Graded practice (issue #493, stages 2 and 3): a mind writes a procedure
for the lid (`mind.practice`), the humanoid runs it from the set-out
(`ProcedureLife`), the world grades the try through stage 1's door
(`evaluation.attempts`), the mind is shown how it went and writes again --
attempts 1 to `ATTEMPTS`, each graded and unpaid, a fresh set-out each.

  ProcedureLife    what a procedure runs in inside an attempt: the body,
                   its model and data, the reports its verbs leave, no
                   interrupt, and what it said
  procedure_skill  a compiled procedure as an attempt's skill
  practice_on      one chest: the conversation and its attempts, until one
                   passes or the attempts run out; a pass ADMITS the
                   procedure to the robot's library (`Library.admit`)
  keep             the procedures a robot carries to its next chest
  out_of_reach     whether the bar can still be met (the flight's early stop)
  JOB              the job as its grader measures it, never how

Two protocols (Evaluation.md §4). FRESH: nothing carries from one chest to
the next -- no library, no conversation -- so a chest's curve is its own
robot's. KEPT (#493, Ben, 2026-10-18): one robot ACQUIRES on the public
chests in turn, carrying only what passed (`keep`); then each held-out chest
gets a copy of it, frozen, and the bar is read there. ⚠ A conversation
never carries: a chest starts from the job, the picture and what was kept.
"""

from __future__ import annotations

import math
import time

from pluggybot.activity import chest as ch
from pluggybot.challenge import lid
from pluggybot.evaluation import attempts as at
from pluggybot.procedure import lang
from pluggybot.procedure.library import MAX_PROCEDURES
from pluggybot.procedure.steps import WorldFacts
from pluggybot.rack.tags import CHEST_TAG_IDS

#: Attempts a chest, the most: stage 1's curve, S@1 to S@5.
ATTEMPTS = 5
#: The tag the knob carries (`rack.tags.CHEST_TAG_IDS`): the stated subsidy;
#: and the knob's size, mm (`activity.chest.KNOB`, a half side), what anyone
#: looking at it would see -- untold, a mind gripped the tag's face and
#: shut its jaws on nothing.
KNOB_TAG = CHEST_TAG_IDS[0]
KNOB_MM = 2000 * ch.KNOB
#: The head camera's picture the mind is shown, pixels.
PICTURE_W, PICTURE_H = 640, 480
#: The bar (call 3, Ben, 2026-10-18): this share of the held-out chests
#: opened and set down within this many tries.
BAR, BAR_TRIES = 0.8, 3
#: KEPT's practice ends once this many chests running pass at their first
#: try, or this many are practiced; and nothing more is flown where nothing
#: is kept after this many (Ben, 2026-10-18: stop once it is plain).
UNTIL, MAX_CHESTS, GIVE_UP_AFTER = 3, 16, 8

#: The job, as its grader measures it (`challenge/lid.py`), and nothing of
#: how: what is in front of the robot, the stated subsidy, and the criteria.
JOB = f"""In front of you, on a counter, is a box with a lid, shut. A knob -- a cube
{KNOB_MM:.0f} mm on a side -- hangs from its front, and it carries tag {KNOB_TAG} on
every face.

Open the lid -- past {math.degrees(lid.OPEN_RAD):.0f} degrees from shut -- then set
it back down shut, and let go of it. While it is open more than
{math.degrees(lid.SHUT_RAD):.0f} degrees the lid must never come down faster than
{lid.LOWER_RAD_S:.1f} rad/s: let fall, it is not set down. The world grades
the job when your procedure ends and again {lid.HOLD_S:.0f} s later: the lid
must be shut by then, and nothing of yours may touch the lid, its knob or its
handle. A try may take {{limit:.0f}} s at most; a fall loses you the time it
takes to get up."""


def job_for(task: at.Task) -> str:
  """The job as the mind is told it at `task`'s set-out: its time limit the
  task's own (`Task.limit_s`)."""
  return JOB.format(limit=task.limit_s)


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


def practice_on(chest: int, practice, attempts: int = ATTEMPTS, *, library=None,
                kept=(), picture: bool = True, budget_s: float | None = None,
                info: dict | None = None, build=at.set_out) -> list[dict]:
  """One chest: the mind's procedure tried from the set-out, graded, and
  shown back (`mind.practice.report_text`), up to `attempts` times, ending
  at the first pass -- which `library` ADMITS on its sealed verdict
  (`Library.admit`). `kept`, the procedures the robot carries (`keep`), are
  shown with the job. Every attempt's row (`attempts.attempt`'s) carries
  the procedure it ran, the answers it took and what they cost. `build`
  makes a set-out (`attempts.attempt`'s: a test's own world)."""
  from pluggybot.mind.practice import report_text
  task = at.Task(chest=chest, body="humanoid", mount="counter")
  limit = task.limit_s if budget_s is None else float(budget_s)
  # ⚠ THE MIND IS TOLD THE LIMIT IT IS CUT AT: told 300 s and cut at 67,
  # a flight was flown and thrown away
  if f"{limit:.0f} s at most" not in practice.job:
    raise ValueError(f"the job tells another time limit than the try's {limit:.0f} s")
  rows: list[dict] = []
  shown = head_picture(task) if picture else None
  since = 0
  turn = practice.first(shown, kept=kept)
  for n in range(1, attempts + 1):
    spent = practice.spent(since)
    since = len(practice.turns)
    out: dict = {}
    sealed: list = []
    t0 = time.time()
    row = at.attempt(task, procedure_skill(turn.procedure, practice.facts, out), n,
                     budget_s=budget_s, sealed=sealed, build=build,
                     info={"process": "practice", "skill": practice.model_id, **(info or {}),
                           "tokens": spent["tokensIn"] + spent["tokensOut"]})
    row["steps"] = out.get("steps")
    row["kept"] = [k["name"] for k in kept]
    row["practice"] = {**spent, "procedure": turn.procedure.name if turn.procedure else None,
                       "source": turn.procedure.source if turn.procedure else None,
                       "turns": [t.as_dict() for t in practice.turns[-spent["calls"]:]]
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


def first_pass(rows: list[dict]) -> int | None:
  """The try a chest's rows first passed at, or None."""
  return next((r["attempt"] for r in rows if (r.get("verdict") or {}).get("ok")), None)


def keep(kept: list[dict], name: str, source: str, chest: int, attempt: int,
         cap: int = MAX_PROCEDURES) -> tuple[list[dict], str | None]:
  """`kept` with a procedure the world passed (`name`, its `source`, the
  chest and try), the newest -- replacing one of its name, else the OLDEST
  going where the library is full. Returns the new list and the name that
  went to make room, or None."""
  out = [k for k in kept if k["name"] != name]
  dropped = None
  if len(out) == len(kept) and len(out) >= cap:
    dropped = out.pop(0)["name"]
  return out + [{"name": name, "source": source, "chest": chest, "attempt": attempt}], dropped


def out_of_reach(firsts: list[int | None], n: int, bar: float = BAR,
                 within: int = BAR_TRIES) -> bool:
  """Whether the bar can no longer be met: more of the `n` chests have failed
  within `within` tries (`firsts`, each chest's first pass, or None) than
  `bar` leaves room for -- the flight's early stop (Ben, 2026-10-18)."""
  failed = sum(1 for f in firsts if f is None or f > within)
  return failed > n - math.ceil(bar * n - 1e-9)


def chest_first(rows: list[dict]) -> int | None:
  """A chest's first pass; None where it passed none or its practice raised
  (an `error` row: no try, never dropped from the denominator)."""
  return None if any("error" in r for r in rows) else first_pass(rows)


def acquire(run_chest, chests, until: int = UNTIL, max_chests: int = MAX_CHESTS,
            give_up_after: int = GIVE_UP_AFTER, write=lambda rows: None) -> tuple[list, dict]:
  """KEPT's practice: `chests` in turn, ONE robot carrying what passed
  (`keep`) -- `run_chest(k, kept)` flies chest k and returns its rows --
  until `until` running pass at their first try or `max_chests` are
  practiced, or nothing is kept after `give_up_after`. Returns what it kept
  and how the practice went."""
  kept: list[dict] = []
  firsts: list = []
  streak, ended = 0, "the chests ran out"
  for k in list(chests)[:max_chests]:
    rows = run_chest(k, kept)
    for r in rows:
      if r.get("admitted"):
        kept, r["dropped"] = keep(kept, r["admitted"], r["practice"]["source"], k, r["attempt"])
    write(rows)
    firsts.append([k, chest_first(rows)])
    streak = streak + 1 if firsts[-1][1] == 1 else 0
    if streak >= until:
      ended = f"{until} chests running passed at their first try"
      break
    if not kept and len(firsts) >= give_up_after:
      ended = f"nothing kept after {give_up_after} chests: given up"
      break
    if len(firsts) == max_chests:
      ended = f"{max_chests} chests practiced"
  return kept, {"firsts": firsts, "ended": ended, "until": until, "maxChests": max_chests}


def flown(results, n: int, early_stop: bool, write=lambda rows: None) -> dict | None:
  """`results`, the `n` chests' rows as each chest ends, written as they
  come; with `early_stop`, ended once the bar is out of reach
  (`out_of_reach`). Returns where it stopped, or None."""
  firsts: list = []
  for rows in results:
    write(rows)
    firsts.append(chest_first(rows))
    if early_stop and out_of_reach(firsts, n):
      return {"after": len(firsts), "of": n, "bar": BAR, "within": BAR_TRIES,
              "failedWithin": sum(1 for f in firsts if f is None or f > BAR_TRIES)}
  return None


def _trimmed(ran: dict | None) -> dict | None:
  """A run's record for the row: every step but its trace."""
  if ran is None:
    return None
  return {**ran, "steps": [{k: v for k, v in s.items() if k != "trace"}
                           for s in ran.get("steps", [])]}
