"""The robot's practice (issue #493): a mind writes a PROCEDURE for a job, the
procedure is tried from the job's start, the world grades the try, and the
mind is shown how its procedure ran and the grade, and writes it again --
trying and reacting, its own skill or none.

THE CONVERSATION (`Practice`): the first turn is the job and the head
camera's picture; each later one what the last procedure did -- every verb
it ran and what that verb said, where it stopped, its variables at the end
-- and the world's grade (`after`). A source that does not compile goes
back with every reason (`MAX_REPAIRS`), and costs tokens but no try.

⚠ IT IS TOLD THE JOB AS ITS GRADER MEASURES IT, AND NOTHING OF HOW: its
rules prescribe no way to open anything (a test reads them), the worked
example waves a hand, and the vocabulary is the body's (`procedure.steps`'
hand verbs, Ben's call 2: none of them knows the chest). What it learns of
the chest it learns from its tries -- graded practice (Ben, 2026-10-10).
The grade is the world's (`challenge/lid.py`), shown as its reason, never
its secret metrics.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import time

from pluggybot.procedure import lang
from pluggybot.procedure.steps import Refused, WorldFacts, describe_vocabulary, signature

#: The tokens an answer may take: the deployed model reasons first (#480:
#: 32,000 left 70 answers in 180 unwritten; every provider took 64,000).
MAX_TOKENS = 64000
#: A refused source goes back with its reasons this many times a try, and
#: an answer that never came is asked again this many times, this long
#: apart (s); an ask the router refused for its rate waits this long times
#: the waits so far, this many times (`imagination.author`'s).
MAX_REPAIRS, MAX_RETRIES, RETRY_WAIT_S = 2, 2, 20.0
RATE_WAIT_S, RATE_WAITS = 60.0, 5

WHAT_YOU_DO = """WHAT YOU ARE DOING

You are the mind of a humanoid robot, 1.4 m tall, standing where it was
put. You do a job by writing a PROCEDURE -- a small program in the language
below -- which your body then runs. Each try starts from the same place:
you standing as you stand now, everything in front of you as it is now.
When a try ends, the world grades the job, and you are shown how your
procedure ran -- every verb it called and what that verb said, where it
stopped, and its variables at the end -- and the grade. Then you write the
procedure again for the next try: the same, changed, or new. You have a
few tries. The grade is the only judge of the job; what your procedure
says of itself is not graded."""

LANGUAGE = """THE LANGUAGE

A procedure is Python-SHAPED text, read and never executed, and it may use
only what is listed here. It is checked whole before anything moves and
refused, with the line at fault, if it uses anything else.

  def wave():                         # one def, no arguments, a lowercase name
      budget(steps=40, seconds=60)    # optional, first; both capped
      n = 0
      reach("right", 0.25, -0.30, 1.30)
      for i in range(3):              # a literal count
          move_hand("right", 0.25, -0.20, 1.30)
          move_hand("right", 0.25, -0.40, 1.30)
          n += 1
      if read("right.z") > 1.0:
          withdraw("right")

Statements: a verb call, `name = expr`, `name += expr`, `if/elif/else`,
`for name in range(N)` (N at most 100), `while` (stopped at 100 rounds),
`pass`, `return` (alone). Expressions: numbers, your variables,
`read("sensor")`, + - * /, comparisons, `and`/`or`/`not`. Nothing else: no
other calls, no strings but a verb's or a read's argument, no imports, no
trigonometry. A procedure runs until it ends, a verb fails, or its budget
runs out; a verb fails when the world says it did not do what it said --
and then the procedure stops there. A verb's numbers come back only
through `read`, and your variables are your readout."""

YOUR_BODY = """YOUR BODY

Every place is in your STANCE frame, in metres: from the floor between your
feet, x straight ahead, y to your left, z up. Your shoulders are 1.15 m up,
4 cm behind your feet's middle and 0.18 m either side of it; an arm held
straight puts its grip 0.58 m from its shoulder, and it reaches less far
with its hand turned some ways. Your hands are "left" and "right": each ends in a two-jaw gripper
whose jaws close across the point between them (its grip), its fingers 8 cm
long. Both grippers start shut. A hand that is OUT is one you have reached
with, until you withdraw it.

VERBS (a statement each; arguments in this order, or by keyword)
%(verbs)s

SENSORS for `read("<sensor>")` -- one number each
%(sensors)s

AXES for `move("<axis>", target)` -- one arm joint's angle, rad, walked to
it at %(slew)g rad/s
%(axes)s"""

ANSWER = """YOUR ANSWER

One JSON object: "think" first -- your reasoning, as long as you need --
then "source": the whole procedure, as text."""


def system_sections(facts: WorldFacts) -> list[tuple[str, str]]:
  """The practice's system prompt, a section at a time: the body's verbs,
  sensors and axes off the registries and what `facts` lets it call."""
  from pluggybot.procedure import axes
  verbs = "\n".join(f"  {signature(v)}  -- {v['doc']}"
                    for v in describe_vocabulary(facts.verbs))
  docs = {s["name"]: s["doc"] for s in axes.describe()["sensors"]}
  sensors = "\n".join(f"  {n} -- {docs[n]}" for n in facts.sensors)
  ax = "\n".join(f"  {a.name}: {a.lo:g}..{a.hi:g} {a.unit} -- {a.doc}"
                 for a in (axes.AXES[n] for n in facts.axes))
  return [("what you do", WHAT_YOU_DO), ("the language", LANGUAGE),
          ("your body", YOUR_BODY % {"verbs": verbs, "sensors": sensors, "axes": ax,
                                     "slew": axes.HUMANOID_SLEW}),
          ("your answer", ANSWER)]


def system_prompt(facts: WorldFacts) -> str:
  return "\n\n".join(text for _, text in system_sections(facts))


def answer_schema() -> dict:
  return {"type": "object",
          "properties": {"think": {"type": "string"}, "source": {"type": "string"}},
          "required": ["think", "source"], "additionalProperties": False}


def extract(response) -> dict:
  """A response's first text block as a JSON object; ValueError otherwise."""
  text = ""
  for block in getattr(response, "content", ()) or ():
    if getattr(block, "type", None) == "text":
      text = block.text
      break
  text = text.strip()
  if text.startswith("```"):
    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
  try:
    raw = json.loads(text)
  except json.JSONDecodeError as e:
    raise ValueError(f"the answer was not JSON: {text[:120]!r}") from e
  if not isinstance(raw, dict) or not isinstance(raw.get("source"), str):
    raise ValueError("the answer had no source")
  return raw


@dataclass
class Turn:
  """One answer: what was asked (`kind`: first, after, repair, retry), the
  answer, the procedure it compiled to (or why it was refused), its tokens,
  how long it took and who answered."""
  kind: str
  answer: dict | None = None
  procedure: lang.Procedure | None = None
  refused: list[str] = field(default_factory=list)
  error: str = ""
  tokens: tuple[int, int] = (0, 0)
  wall_s: float = 0.0
  provider: str | None = None

  def as_dict(self) -> dict:
    return {"kind": self.kind, "think": (self.answer or {}).get("think"),
            "source": (self.answer or {}).get("source"), "refused": list(self.refused),
            "error": self.error, "tokensIn": self.tokens[0], "tokensOut": self.tokens[1],
            "wallS": round(self.wall_s, 1), "provider": self.provider}


class Practice:
  """The conversation (the module docstring). `client` is the overseer's
  seam: anything with `.messages.create(**kwargs)` (`mind.llm`)."""

  def __init__(self, client, model: str, facts: WorldFacts, job: str,
               backend: str = "huggingface", max_tokens: int = MAX_TOKENS,
               retry_wait_s: float = RETRY_WAIT_S, rate_wait_s: float = RATE_WAIT_S) -> None:
    self.client, self.model_id, self.facts, self.job = client, model, facts, job
    self.backend, self.max_tokens = backend, max_tokens
    self.retry_wait_s, self.rate_wait_s = retry_wait_s, rate_wait_s
    self.messages: list[dict] = []
    self.turns: list[Turn] = []
    self._last = None

  def _ask(self, kind: str, content) -> Turn:
    self._last = content
    self.messages.append({"role": "user", "content": content})
    turn = Turn(kind=kind)
    t0 = time.monotonic()
    try:
      response = self.client.messages.create(
        model=self.model_id, max_tokens=self.max_tokens,
        system=[{"type": "text", "text": system_prompt(self.facts)}],
        output_config={"format": {"type": "json_schema", "schema": answer_schema()}},
        messages=list(self.messages))
      usage = getattr(response, "usage", None)
      turn.tokens = (int(getattr(usage, "input_tokens", 0) or 0),
                     int(getattr(usage, "output_tokens", 0) or 0))
      turn.provider = getattr(response, "provider", None)
      turn.wall_s = time.monotonic() - t0
      turn.answer = extract(response)
    except Exception as e:                                # noqa: BLE001 -- said why
      turn.error = f"{type(e).__name__}: {e}"
      turn.wall_s = time.monotonic() - t0
      self.messages.pop()
      self.turns.append(turn)
      return turn
    self.messages.append({"role": "assistant", "content": json.dumps(turn.answer)})
    try:
      turn.procedure = lang.compile_procedure(turn.answer["source"], self.facts)
    except Refused as e:
      turn.refused = list(e.reasons)
    self.turns.append(turn)
    return turn

  def _settled(self, turn: Turn) -> Turn:
    """`turn`, or the repairs and retries it took (`MAX_REPAIRS`,
    `MAX_RETRIES`, `RATE_WAITS`)."""
    repairs = retries = waits = 0
    while True:
      if turn.refused and repairs < MAX_REPAIRS:
        repairs += 1
        turn = self._ask("repair", "Your procedure was refused:\n"
                         + "\n".join(f"- {r}" for r in turn.refused)
                         + "\n\nWrite it again, the whole procedure, every reason answered.")
      elif " 429" in turn.error and waits < RATE_WAITS and self._last is not None:
        waits += 1
        time.sleep(self.rate_wait_s * waits)
        turn = self._ask("retry", self._last)
      elif turn.error and retries < MAX_RETRIES and self._last is not None:
        retries += 1
        time.sleep(self.retry_wait_s)
        turn = self._ask("retry", self._last)
      else:
        return turn

  def first(self, picture: bytes | None = None) -> Turn:
    """The first procedure: the job, and the head camera's picture."""
    text = (f"THE JOB\n\n{self.job}\n\nThe picture is your head camera's, as you stand "
            "now. Write the procedure for your first try.") if picture is not None else \
      f"THE JOB\n\n{self.job}\n\nWrite the procedure for your first try."
    if picture is None:
      content = text
    else:
      from pluggybot.mind.llm import image_part
      content = [image_part(self.backend, base64.b64encode(picture).decode("ascii")),
                 {"type": "text", "text": text}]
    return self._settled(self._ask("first", content))

  def after(self, report: str) -> Turn:
    """The next procedure, off how the last try went (`report_text`)."""
    return self._settled(self._ask("after", report))

  def spent(self, since: int = 0) -> dict:
    """The tokens, calls and wall time of the turns from `since` on."""
    turns = self.turns[since:]
    return {"tokensIn": sum(t.tokens[0] for t in turns),
            "tokensOut": sum(t.tokens[1] for t in turns), "calls": len(turns),
            "llmWallS": round(sum(t.wall_s for t in turns), 1)}


def report_text(n: int, ran: dict | None, verdict: dict, sim_s: float, cut: bool,
                remaining: int) -> str:
  """What the mind is shown of try `n`: how its procedure ran (`ran`, the
  language's result, or None where it had none to run) and the world's
  grade -- its reason, never its secret metrics."""
  lines = [f"TRY {n}", ""]
  if ran is None:
    lines.append("You had no procedure that compiled, so your body stood still.")
  else:
    head = (f"Your procedure `{ran.get('program')}` called {ran.get('total', 0)} verbs "
            f"in {ran.get('seconds', sim_s):.1f} s")
    stopped = ran.get("stopped")
    if "failedAt" in ran and ran["steps"]:
      bad = ran["steps"][-1]
      head += (f" and stopped at line {bad.get('line', '?')}, {bad.get('verb')}: "
               f"{bad.get('reason', 'it failed')}")
    elif stopped:
      head += f" and stopped: {stopped}"
    else:
      head += " and ended"
    lines.append(head + ".")
    for step in ran.get("steps", [])[-40:]:
      word = "ok" if step.get("ok") else "FAILED"
      said = {k: v for k, v in step.items()
              if k not in ("i", "verb", "line", "ok", "reason", "trace")}
      lines.append(f"  line {step.get('line', '?')} {step.get('verb')}: {word}"
                   + (f" -- {step['reason']}" if step.get("reason") else "")
                   + (f" {json.dumps(said, default=str)}" if said else ""))
    if ran.get("locals"):
      lines.append("Its variables at the end: " + ", ".join(
        f"{k} = {v:g}" for k, v in ran["locals"].items()))
  if cut:
    lines.append(f"The try's time ran out after {sim_s:.0f} s.")
  lines += ["", f"THE GRADE: {'PASSED' if verdict['ok'] else 'failed'} -- {verdict['reason']}", ""]
  lines.append(f"Write the procedure for try {n + 1} ({remaining} left)." if remaining
               else "That was your last try.")
  return "\n".join(lines)
