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

#: The tokens an answer may take, its reasoning first: at 64,000, 37 of
#: #493's 537 answers ran out still thinking, and answered ones went to
#: 63,064 (the providers take 200,000). The mind is told to finish well
#: inside `AIM_TOKENS`, and an answer cut off is asked again, saying so.
MAX_TOKENS, AIM_TOKENS = 128000, 64000
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
procedure ran -- every verb it called and what that verb said, what your
hand felt while each of its verbs ran (where its grip was, the force it
pushed with, whether it held), where it stopped, and its variables at the
end -- and the grade. Then you write the
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
#: ...where the world gives the maths (`WorldFacts.math`), what replaces
#: its "no trigonometry".
NO_TRIG = """no imports, no
trigonometry."""
MATH_TEXT = """no imports -- but
`sin(a)`, `cos(a)` (a in radians), `atan2(y, x)` and `sqrt(v)` may be used
in an expression."""
#: ...and where a verb may be an `if`'s test (`WorldFacts.verb_tests`).
VERB_TESTS = """
A verb may also stand alone as the test of an `if` or an `elif` -- `if
move_hand("left", 0.3, 0.2, 0.9):` or `if not reach(...):` -- and then it
runs as the test: true when it did what it said, false when it did not,
and its failing does not stop the procedure."""

YOUR_BODY = """YOUR BODY

Every place is in your STANCE frame, in metres: from the floor between your
feet, x straight ahead, y to your left, z up. The frame moves with your
feet, and taking a posture to reach can move them a little: a place you
read before a reach may lie elsewhere in the frame after it. Your shoulders are 1.15 m up,
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

One JSON object: "think" first -- your reasoning -- then "source": the
whole procedure, as text. Your reply, your thinking included, is cut off
at %(cap)s tokens, and a reply cut off is lost: take what thinking you
need, and finish well inside %(aim)s."""
#: What an answer asked again after it was cut off is told first.
CUT_NOTE = ("(Your last reply ran out of its {tokens:,} tokens, your thinking "
            "included, before you wrote your answer, and it was lost. Think less "
            "this time, and answer.)")


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
  return [("what you do", WHAT_YOU_DO),
          ("the language", (LANGUAGE.replace(NO_TRIG, MATH_TEXT) if facts.math else LANGUAGE)
           + (VERB_TESTS if facts.verb_tests else "")),
          ("your body", YOUR_BODY % {"verbs": verbs, "sensors": sensors, "axes": ax,
                                     "slew": axes.HUMANOID_SLEW}),
          ("your answer", ANSWER % {"cap": f"{MAX_TOKENS:,}", "aim": f"{AIM_TOKENS:,}"})]


def system_prompt(facts: WorldFacts) -> str:
  return "\n\n".join(text for _, text in system_sections(facts))


#: What the robot carries to its next chest (#493, Ben, 2026-10-18: what a
#: person takes from a failed attempt): its notes, at most this long --
#: asked again once past it, then cut and said so -- and a draft procedure.
NOTES_CHARS = 1500
NOTES_ASK = """THIS BOX IS DONE: {how}. Your next box is another like it, and
you will start it as you started this one, with nothing of this
conversation but what you carry. Write what you will carry: "notes" -- what
worked, what did not, what you would try next, at most {cap} characters,
keeping what still holds from any notes you were given -- and "draft": the
procedure you would start the next box from, the whole of it, or "" for
none. A procedure that passed is in your library already."""


NOTES_LONG = """Your notes are {n} characters, past the {cap} you may carry.
Write them again, at most {cap}, the most useful first; the draft as
before."""


def notes_schema() -> dict:
  return {"type": "object",
          "properties": {"think": {"type": "string"}, "notes": {"type": "string"},
                         "draft": {"type": "string"}},
          "required": ["think", "notes", "draft"], "additionalProperties": False}


def carried_text(carried: dict) -> str:
  """What the robot is shown of what it carried from its last chest: its
  own notes and its draft, both its own words."""
  out = []
  if carried.get("notes"):
    out += ["YOUR NOTES, written after your last box:", "", carried["notes"].rstrip()]
  if carried.get("draft"):
    out += (["", ""] if out else []) + [
      "YOUR DRAFT, the procedure you meant to start this box from (it never passed):",
      "", carried["draft"].rstrip()]
  return "\n".join(out)


def answer_schema() -> dict:
  return {"type": "object",
          "properties": {"think": {"type": "string"}, "source": {"type": "string"}},
          "required": ["think", "source"], "additionalProperties": False}


def extract(response, key: str = "source") -> dict:
  """A response's first text block as a JSON object with `key` a string;
  ValueError otherwise."""
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
  if not isinstance(raw, dict) or not isinstance(raw.get(key), str):
    raise ValueError(f"the answer had no {key}")
  return raw


def library_text(kept) -> str:
  """What the robot is shown of the procedures it KEPT (#493): each, the
  box and the try whose pass admitted it, and its source -- a pass the
  world graded, never a word of how it did."""
  lines = ["YOUR LIBRARY", "",
           "Procedures you kept, each because the world graded a try of it a pass on "
           "a box like this one:"]
  for k in kept:
    lines += ["", f"{k['name']} -- passed box {k['chest']} on its try {k['attempt']}:",
              k["source"].rstrip()]
  return "\n".join(lines)


def _noted(content, note: str):
  """`content` -- a user turn's text, or its parts -- with `note` first."""
  if not note:
    return content
  if isinstance(content, str):
    return f"{note}\n\n{content}"
  return [{"type": "text", "text": note}, *content]


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
  #: its tokens ran out before it was written
  cut: bool = False

  def as_dict(self) -> dict:
    return {"kind": self.kind, "think": (self.answer or {}).get("think"),
            "source": (self.answer or {}).get("source"), "refused": list(self.refused),
            "error": self.error, "tokensIn": self.tokens[0], "tokensOut": self.tokens[1],
            "wallS": round(self.wall_s, 1), "provider": self.provider, "cut": self.cut}


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
    self._last, self._notes = None, False

  def _ask(self, kind: str, content, note: str = "", notes: bool = False) -> Turn:
    """One answer to `content`, sent with `note` first where there is one
    (the content is what a retry asks again): a procedure, or with `notes`
    what the robot carries to its next chest (`notes_schema`)."""
    self._last, self._notes = content, notes
    self.messages.append({"role": "user", "content": _noted(content, note)})
    turn = Turn(kind=kind)
    t0 = time.monotonic()
    try:
      response = self.client.messages.create(
        model=self.model_id, max_tokens=self.max_tokens,
        system=[{"type": "text", "text": system_prompt(self.facts)}],
        output_config={"format": {"type": "json_schema",
                                  "schema": notes_schema() if notes else answer_schema()}},
        messages=list(self.messages))
      usage = getattr(response, "usage", None)
      turn.tokens = (int(getattr(usage, "input_tokens", 0) or 0),
                     int(getattr(usage, "output_tokens", 0) or 0))
      turn.provider = getattr(response, "provider", None)
      turn.wall_s = time.monotonic() - t0
      turn.cut = getattr(response, "stop_reason", None) == "max_tokens"
      turn.answer = extract(response, "notes" if notes else "source")
    except Exception as e:                                # noqa: BLE001 -- said why
      turn.error = (f"cut off: its {self.max_tokens:,} tokens ran out before the answer "
                    "was written" if turn.cut else f"{type(e).__name__}: {e}")
      turn.wall_s = time.monotonic() - t0
      self.messages.pop()
      self.turns.append(turn)
      return turn
    self.messages.append({"role": "assistant", "content": json.dumps(turn.answer)})
    if not notes:
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
      elif turn.cut and turn.error and retries < MAX_RETRIES and self._last is not None:
        retries += 1
        turn = self._ask("retry", self._last, note=CUT_NOTE.format(tokens=self.max_tokens),
                         notes=self._notes)
      elif " 429" in turn.error and waits < RATE_WAITS and self._last is not None:
        waits += 1
        time.sleep(self.rate_wait_s * waits)
        turn = self._ask("retry", self._last, notes=self._notes)
      elif turn.error and retries < MAX_RETRIES and self._last is not None:
        retries += 1
        time.sleep(self.retry_wait_s)
        turn = self._ask("retry", self._last, notes=self._notes)
      else:
        return turn

  def first(self, picture: bytes | None = None, kept=(), carried: dict | None = None) -> Turn:
    """The first procedure: the job, the head camera's picture, the
    procedures the robot KEPT (`library_text`), and the notes and draft it
    CARRIED from its last chest (`carried_text`), where it has any."""
    text = (f"THE JOB\n\n{self.job}\n\nThe picture is your head camera's, as you stand "
            "now. Write the procedure for your first try.") if picture is not None else \
      f"THE JOB\n\n{self.job}\n\nWrite the procedure for your first try."
    if carried and (carried.get("notes") or carried.get("draft")):
      text = f"{carried_text(carried)}\n\n{text}"
    if kept:
      text = f"{library_text(kept)}\n\n{text}"
    if picture is None:
      content = text
    else:
      from pluggybot.mind.llm import image_part
      content = [image_part(self.backend, base64.b64encode(picture).decode("ascii")),
                 {"type": "text", "text": text}]
    return self._settled(self._ask("first", content))

  def carry(self, passed_at: int | None) -> tuple[dict, Turn]:
    """What the robot carries to its next chest, written as this one ends
    (`NOTES_ASK`): {"notes", "draft"} -- the notes cut at `NOTES_CHARS`, and
    said so -- and the answer's turn. Nothing where no answer came."""
    how = (f"it passed on your try {passed_at}" if passed_at
           else "none of your tries passed")
    turn = self._settled(self._ask("notes", NOTES_ASK.format(how=how, cap=NOTES_CHARS),
                                   notes=True))
    if turn.answer is not None and len(str(turn.answer.get("notes") or "").strip()) > NOTES_CHARS:
      # asked again once, as a refused source is: cut, a smoke run's notes
      # lost their end -- what it meant to try next
      again = self._settled(self._ask("notes", NOTES_LONG.format(
        n=len(str(turn.answer["notes"]).strip()), cap=NOTES_CHARS), notes=True))
      turn = again if again.answer is not None else turn
    if turn.answer is None:
      return {"notes": "", "draft": ""}, turn
    notes = str(turn.answer.get("notes") or "").strip()
    if len(notes) > NOTES_CHARS:
      notes = notes[:NOTES_CHARS] + f" [cut at {NOTES_CHARS} characters]"
    return {"notes": notes, "draft": str(turn.answer.get("draft") or "").strip()}, turn

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
      word = ("ok" if step.get("ok") else
              "FAILED, read by its `if`, and the procedure went on" if step.get("tested")
              else "FAILED")
      said = {k: v for k, v in step.items()
              if k not in ("i", "verb", "line", "ok", "reason", "trace", "felt", "tested")}
      lines.append(f"  line {step.get('line', '?')} {step.get('verb')}: {word}"
                   + (f" -- {step['reason']}" if step.get("reason") else "")
                   + (f" {json.dumps(said, default=str)}" if said else ""))
      if step.get("felt"):
        lines.append("    felt -- s: grip x y z, m | force it pushed with x y z, N | holding")
        lines += [f"      {f['t']:.2f}: {f['at'][0]:.3f} {f['at'][1]:.3f} {f['at'][2]:.3f} | "
                  f"{f['force'][0]:+.1f} {f['force'][1]:+.1f} {f['force'][2]:+.1f} | "
                  f"{'yes' if f['held'] else 'no'}" for f in step["felt"]]
    if ran.get("locals"):
      lines.append("Its variables at the end: " + ", ".join(
        f"{k} = {v:g}" for k, v in ran["locals"].items()))
  if cut:
    lines.append(f"The try's time ran out after {sim_s:.0f} s.")
  lines += ["", f"THE GRADE: {'PASSED' if verdict['ok'] else 'failed'} -- {verdict['reason']}", ""]
  lines.append(f"Write the procedure for try {n + 1} ({remaining} left)." if remaining
               else "That was your last try.")
  return "\n".join(lines)
