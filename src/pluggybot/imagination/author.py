"""The model's author (issue #466, stage 3): a language model writes the
STRUCTURE of the robot's model of what is in front of it -- what the parts
are, which joint joins them and where, whether something holds one -- as a
template in the scene language, every number it cannot see left as a range
(`scene.parse_template`). Code measured the geometry it is shown, and code
fits the numbers it leaves (`fit.py`); it decides nothing else.

THE CONVERSATION (`Author`): the first turn is the colour camera's picture,
the sizes depth and the tags measured, and what the arm did (`first`); a
document that does not parse goes back with every reason (`repair`); a
fitted one that left too much goes back with the fit's report (`revise`,
`model.py` judges when).

⚠ IT IS TOLD NOTHING OF THE WORLD'S MODEL: what its turns carry is the
robot's own -- the picture, the sizes, the arm's own path and what the fit
of its own document left -- and nothing in this package can read a
mechanism's activity (the fence, `tests/test_imagination.py`). ⚠ ITS RULES
PRESCRIBE NOTHING about what the object is, and the worked example is a
cupboard's door: an example of the answer would hand it over.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json

from pluggybot.imagination.scene import Refused, Template, parse_template, placed

#: The tokens an answer may take: the deployed model reasons first, and
#: shown the picture it reasoned 22,351 tokens for one document (8,192, the
#: overseer's, left it no answer at all).
MAX_TOKENS = 32000
#: A refused document goes back with every reason this many times a round
#: at most, and an answer that never came (cut off, or no JSON) is asked
#: again this many, apart -- shared, the retries left a refusal no repair --
#: after this long (s): a router that refused a request at once refuses it
#: again.
MAX_REPAIRS = 2
MAX_RETRIES = 2
RETRY_WAIT_S = 20.0
#: What a retry adds to the ask: the deployed model reasoned past 32,000
#: tokens and wrote nothing, two answers in five.
RETRY_NOTE = ("\n\n(Your last answer ran out of room before its document was written. "
              "Think as briefly as you can, then write the document.)")

WHAT_YOU_DO = """WHAT YOU ARE DOING

You write a robot's model of an object in front of it: a small world of
simple solid parts that a physics simulation can run. The robot took hold
of the object with its claw and moved it, recording everything its arm was
sent and everything it felt. Your document is simulated along exactly what
the arm was sent, and code then finds each number you leave unknown -- a
mass, a spring, a damping, a friction, the pull a catch lets go at -- so
that the simulated arm feels what the real one felt.

So write what the object IS: its parts, which of them move, on what joint
and where, and whether anything holds a part until it is pulled. Write
every number you can see or were given. Leave the ones nobody can see as a
range they surely fall in.

Before you write it, think of more than one way the object could be built,
and keep the one that explains everything: what the picture shows, the
sizes, and where the robot's jaws were able to go."""

LANGUAGE = """THE SCENE LANGUAGE

A document is a JSON object of "parts", "joints" and "catches". Lengths are
mm, angles degrees, masses kg, all in the BOX'S FRAME: its origin on the
floor under the middle of the object's front face (the face toward the
robot), x into the object, y to its left as the robot sees it, z up. The
robot is in front, at negative x.

A PART is one solid primitive with its own mass:
  "shape"  "box" ("size" its x, y and z extents), "slab" (a board: "size"
           its x and y and its thickness as z, at most a quarter of the
           other two) or "cylinder" ("size" its diameter and its length,
           along its own z)
  "pos"    its middle; "euler" its turn ([x, y, z] degrees, applied x then
           y then z; [0, 0, 0] square to the frame)
  "on"     the id of the part it rides, or null: fixed to the floor. The
           parts on nothing are one rigid whole.
  "mass"   kg

A JOINT lets its "part", and everything on that part, move against what the
part is on:
  "type"   "hinge" (turns about "axis" through the point "at") or "slide"
           (runs along "axis"; "at" only says where)
  "range"  [lo, hi], how far it can go from where you drew the part (degrees
           for a hinge, mm for a slide): it includes 0, the pose you drew.
           null is unlimited
  "stiffness"  a spring's, N*m/rad (a slide's N/m), slack at "slack"
           (degrees, or mm, off the pose you drew)
  "damping"    N*m*s/rad (N*s/m), and "friction", N*m (N): the joint's own
A hinge's turn is right-handed about its axis.

A CATCH holds its "joint" at the end of its range where you drew the part
(a range [0, hi] is caught at 0), with one pull: "release" newtons at the
point "at". Pulled harder there, it lets go; it takes the part again when
the part comes back.

Parts touch each other, but for a moving part and the parts it is hinged
to. A part may be on a part that moves: a door's knob rides its door.

An UNKNOWN: a number nobody can see -- a mass, a stiffness, a slack, a
damping, a friction, a release -- may be written {"between": [lo, hi]}, a
range it surely falls in, and code finds it. Sizes, places, axes and ranges
are never unknown: they are measured, or they are yours to draw.

An example, a cupboard with a door hinged on its right edge and a pull:
  {"parts": [
     {"id": "cupboard", "shape": "box", "size": [300, 400, 500],
      "pos": [150, 0, 250], "euler": [0, 0, 0], "on": null, "mass": 8.0},
     {"id": "door", "shape": "box", "size": [18, 380, 300],
      "pos": [-9, 0, 330], "euler": [0, 0, 0], "on": "cupboard",
      "mass": {"between": [0.3, 3.0]}},
     {"id": "pull", "shape": "cylinder", "size": [20, 30],
      "pos": [-33, 160, 330], "euler": [0, 90, 0], "on": "door",
      "mass": 0.04}],
   "joints": [
     {"id": "door_hinge", "type": "hinge", "part": "door",
      "at": [-9, -190, 330], "axis": [0, 0, 1], "range": [0, 110],
      "stiffness": 0, "slack": 0, "damping": {"between": [0, 0.5]},
      "friction": {"between": [0, 0.3]}}],
   "catches": []}"""

ANSWER = """YOUR ANSWER

One JSON object: "think" first (your reasoning, as long as you need), then
"what" (what the object is, in a sentence), then "parts", "joints" and
"catches" as the language has them. Write every field: "on" and "range"
may be null, "euler" is [0, 0, 0] when a part is square, and a list may be
empty."""


def system_sections() -> list[tuple[str, str]]:
  """The author's system prompt, a section at a time."""
  return [("what you do", WHAT_YOU_DO), ("the scene language", LANGUAGE),
          ("your answer", ANSWER)]


def system_prompt() -> str:
  return "\n\n".join(text for _, text in system_sections())


# ---- the answer's shape -----------------------------------------------------------------

def _numbers(lo: int, hi: int) -> dict:
  return {"type": "array", "items": {"type": "number"}, "minItems": lo, "maxItems": hi}


_BETWEEN = {"type": "object", "properties": {"between": _numbers(2, 2)},
            "required": ["between"], "additionalProperties": False}
_MAYBE = {"anyOf": [{"type": "number"}, _BETWEEN]}


def _object(props: dict) -> dict:
  return {"type": "object", "properties": props, "required": list(props),
          "additionalProperties": False}


def answer_schema() -> dict:
  """The answer as a JSON schema (`response_format`): the language's own
  fields, each written, a number or an unknown where one may be."""
  part = _object({"id": {"type": "string"},
                  "shape": {"type": "string", "enum": ["box", "slab", "cylinder"]},
                  "size": _numbers(2, 3), "pos": _numbers(3, 3), "euler": _numbers(3, 3),
                  "on": {"type": ["string", "null"]}, "mass": _MAYBE})
  joint = _object({"id": {"type": "string"},
                   "type": {"type": "string", "enum": ["hinge", "slide"]},
                   "part": {"type": "string"}, "at": _numbers(3, 3), "axis": _numbers(3, 3),
                   "range": {"anyOf": [_numbers(2, 2), {"type": "null"}]},
                   "stiffness": _MAYBE, "slack": _MAYBE, "damping": _MAYBE,
                   "friction": _MAYBE})
  catch = _object({"joint": {"type": "string"}, "at": _numbers(3, 3), "release": _MAYBE})
  return _object({"think": {"type": "string"}, "what": {"type": "string"},
                  "parts": {"type": "array", "items": part},
                  "joints": {"type": "array", "items": joint},
                  "catches": {"type": "array", "items": catch}})


def template_of(answer: dict) -> dict:
  """An answer's document, as the language reads it: its own three lists,
  a null `on` or `range` left out."""
  doc = {}
  for key in ("parts", "joints", "catches"):
    items = answer.get(key, [])
    if not isinstance(items, list):
      doc[key] = items
      continue
    doc[key] = [{k: v for k, v in item.items() if not (k in ("on", "range") and v is None)}
                if isinstance(item, dict) else item for item in items]
  return doc


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
  if not isinstance(raw, dict):
    raise ValueError(f"the answer was not an object: {type(raw).__name__}")
  return raw


# ---- the turns ----------------------------------------------------------------------

def measures_text(sizes: dict, did: str) -> str:
  """The first turn's words: what depth and the tags measured, in the box's
  frame (`legs.probe.Sizes`), and what the arm did."""
  s = sizes
  k = s["knob"]
  return "\n".join([
    "WHAT THE ROBOT MEASURED (its depth camera and the tag; mm, the box's frame)",
    "",
    f"- The object's top is flat at z = {s['top']:.1f}, from its front face at x = 0 to "
    f"its back edge at x = {s['depth']:.1f}, and {s['width']:.1f} wide (y from "
    f"{-s['width'] / 2:.1f} to {s['width'] / 2:.1f}). Its front face stands from the "
    f"floor to the top.",
    f"- A thin flat bracket comes out of the top's front edge toward the robot: "
    f"{s['bracket_width']:.1f} wide, its tip at x = {s['tip']:.1f}, its upper face at "
    f"z = {s['bracket_top']:.1f}.",
    f"- A cube {s['knob_size']:.0f} mm on a side, a tag printed on each face, hangs "
    f"below and in front of the bracket's tip: its middle at ({k[0]:.1f}, {k[1]:.1f}, "
    f"{k[2]:.1f}).",
    "",
    "WHAT THE ROBOT DID",
    "",
    did,
    "",
    "The picture is its colour camera's, before it lay down. Write the object's model."])


def report_text(report: dict) -> str:
  """A revision turn's words off `model.report`: what the fit found, where
  the simulated arm felt what the real one did not, and the question."""
  if report.get("failed"):
    return "\n".join(["THE FIT", "", f"No fit could be read: {report['failed']}.", "",
                      "Write the document again so that it can be simulated: parts that "
                      "do not start inside each other, joints a part can turn or slide "
                      "on. Answer as before: the whole document."])
  lines = ["THE FIT", "", "Code fitted your unknowns:"]
  for name, v in report["values"].items():
    note = ("  (at an end of your range)" if name in report["atEnd"]
            else "  (nothing the arm felt moves it: held at the middle of your range)"
            if name in report["unseen"] else "")
    lines.append(f"  {name} = {v:.4g}{note}")
  if not report["values"]:
    lines.append("  (you left none unknown)")
  lines += ["", "What the simulated arm felt apart from the real one, N at the jaws (RMS "
            "of 0.1 s means); past the bar is more than the scene language's best "
            "model of an object usually leaves:"]
  for p in report["phases"]:
    flag = "  PAST THE BAR" if p["poor"] else ""
    bar = "" if p["barN"] is None else f" (bar {p['barN']:.3f})"
    lines.append(f"  {p['label']}: {p['rmsN']:.3f}{bar}{flag}")
  spanned = any(p["poor"] and p.get("whole") for p in report["phases"])
  for p in report["phases"]:
    # the stretches that were poor, or that a poor whole spans
    if p.get("whole") or not (p["poor"] or (spanned and p["barN"] is None)):
      continue
    if p.get("felt"):
      lines += ["", f"{p['label'].capitalize()}, 0.5 s at a time: the force the real "
                f"object put on the jaws, and the simulated one, N (toward the object, "
                f"up):"]
      for t0, (rx, rz), (mx, mz) in p["felt"]:
        lines.append(f"  {t0:4.1f} s  real ({rx:+.2f}, {rz:+.2f})   simulated ({mx:+.2f}, "
                     f"{mz:+.2f})")
      continue
    lines += ["", f"{p['label'].capitalize()}, 0.5 s at a time: the force the real "
              f"object put on the jaws minus the simulated one's, N (toward the "
              f"object, up):"]
    for t0, (fx, fz) in p["trace"]:
      lines.append(f"  {t0:4.1f} s  ({fx:+.2f}, {fz:+.2f})")
  lines += ["", "If your document's structure is wrong -- a part, a joint or a catch "
            "missing, extra or misplaced -- write it again. You may change any range. "
            "Answer as before: the whole document."]
  return "\n".join(lines)


@dataclass
class Turn:
  """One answer: what was asked (`kind`: first, repair or revise), the
  answer as written, its template in the box's frame (or the reasons it
  was refused), and the tokens it cost."""
  kind: str
  answer: dict | None = None
  template: Template | None = None
  refused: list[str] = field(default_factory=list)
  error: str = ""
  tokens: tuple[int, int] = (0, 0)

  def as_dict(self) -> dict:
    return {"kind": self.kind, "what": (self.answer or {}).get("what"),
            "think": (self.answer or {}).get("think"),
            "document": None if self.answer is None else template_of(self.answer),
            "refused": list(self.refused), "error": self.error,
            "tokensIn": self.tokens[0], "tokensOut": self.tokens[1]}


def _noted(content):
  """An ask with `RETRY_NOTE` at its end, once: a text, or the text part of
  a picture and its words."""
  if isinstance(content, str):
    return content.split(RETRY_NOTE)[0] + RETRY_NOTE
  out = [dict(part) for part in content]
  for part in reversed(out):
    if part.get("type") == "text":
      part["text"] = part["text"].split(RETRY_NOTE)[0] + RETRY_NOTE
      break
  return out


class Author:
  """The conversation with the model's author (the module docstring).
  `client` is the injection seam the overseer's is: anything with
  `.messages.create(**kwargs)` returning `.content` and `.usage`
  (`mind.llm.build_client`); `backend` says how a picture is attached."""

  def __init__(self, client, model: str, backend: str = "huggingface",
               max_tokens: int = MAX_TOKENS, retry_wait_s: float = RETRY_WAIT_S) -> None:
    self.client, self.model_id, self.backend = client, model, backend
    self.max_tokens = max_tokens
    self.retry_wait_s = retry_wait_s
    self.messages: list[dict] = []
    self.turns: list[Turn] = []
    self._last_asked = None

  def _ask(self, kind: str, content) -> Turn:
    self._last_asked = content
    self.messages.append({"role": "user", "content": content})
    turn = Turn(kind=kind)
    try:
      response = self.client.messages.create(
        model=self.model_id, max_tokens=self.max_tokens,
        system=[{"type": "text", "text": system_prompt()}],
        output_config={"format": {"type": "json_schema", "schema": answer_schema()}},
        messages=list(self.messages))
      usage = getattr(response, "usage", None)
      turn.tokens = (int(getattr(usage, "input_tokens", 0) or 0),
                     int(getattr(usage, "output_tokens", 0) or 0))
      turn.answer = extract(response)
    except Exception as e:                                # noqa: BLE001 -- an answer said why
      turn.error = f"{type(e).__name__}: {e}"
      self.messages.pop()
      self.turns.append(turn)
      return turn
    self.messages.append({"role": "assistant", "content": json.dumps(turn.answer)})
    try:
      turn.template = parse_template(template_of(turn.answer))
    except Refused as e:
      turn.refused = list(e.reasons)
    self.turns.append(turn)
    return turn

  def _settled(self, turn: Turn) -> Turn:
    """`turn`, or the repairs it took: a refused document goes back with
    every reason (`MAX_REPAIRS`), and an answer that never came or was no
    JSON is asked again, saying so (`MAX_RETRIES`; its turn not kept)."""
    repairs = retries = 0
    while True:
      if turn.refused and repairs < MAX_REPAIRS:
        repairs += 1
        turn = self._ask("repair", "Your document was refused:\n" +
                         "\n".join(f"- {r}" for r in turn.refused) +
                         "\n\nAnswer again, the whole document, every reason answered.")
      elif turn.error and retries < MAX_RETRIES and self._last_asked is not None:
        retries += 1
        import time
        time.sleep(self.retry_wait_s)
        turn = self._ask("retry", _noted(self._last_asked))
      else:
        return turn

  def first(self, sizes: dict, did: str, picture: bytes | None) -> Turn:
    """The first answer: the picture, the sizes and what the arm did."""
    text = measures_text(sizes, did)
    if picture is None:
      content = text
    else:
      from pluggybot.mind.llm import image_part
      content = [image_part(self.backend, base64.b64encode(picture).decode("ascii")),
                 {"type": "text", "text": text}]
    return self._settled(self._ask("first", content))

  def revise(self, report: dict) -> Turn:
    """A revised answer, off the fit's report (`model.report`)."""
    return self._settled(self._ask("revise", report_text(report)))


def in_map(template: Template, origin_mm, yaw_deg: float) -> Template:
  """A template drawn in the box's frame, written in the map's at the box's
  `origin_mm` (x, y on the floor) turned `yaw_deg` (`legs.probe.Sizes`)."""
  return parse_template(placed(template.raw, (origin_mm[0], origin_mm[1], 0.0), yaw_deg))
