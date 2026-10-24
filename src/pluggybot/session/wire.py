"""What a session's stream carries beyond the world's frames, and what the
host and the site say to each other (issue #530; protocol/README.md, "A
world of one's own"). Pure: a message is built and read here, and a test
pins each shape without flying anything.

THE STREAM, a child's, to the session's own ingest path (`attempt`), after
the header:
  scene     the set-out's world, whole, as the site draws it: a person's
            set-out is drawn afresh, so no vendored fixture can know it
  step      a verb of the procedure began: its count, its line, its verb and
            what it was asked
  event     a narration line, the served world's message (the procedure's
            own words: each verb as it begins, a failure and why)
  readouts  the robot's senses as its procedure reads them (`read`), each
            `READOUT_S`; None where a sense has no reading yet
  depth     what its depth camera sees, a picture each `DEPTH_S`, on the
            grid's PNG path
  attempt   the end: the verdict, the run as the language recorded it, and
            the report the robot's mind is shown of a try -- never a secret
            metric

THE HOST, to the site's `/api/pluggyworld/sessions/host` (`host`), and the
site's two words back (`run`, `cancel`):
  hello     on every connect: the build, the task, the job as the robot is
            told it, and the language and the body a procedure is written
            in -- the same text the robot's mind is given (`mind.practice`)
            -- and its `index`, the names that text lists, as data
  refused   a run's procedure did not compile: every reason at once, and
            nothing built
  started   a run's world is being built, its stream about to open
  ended     a run's process ended, and why: a graded run's verdict with
            it, the stream's backstop (the child's word, kept before the
            stream's last message went)
  busy      a run asked while one was in flight
and the site's: `run` (one attempt, checked by `parse_run`) and `cancel`,
which names the run's session AND attempt -- a late cancel for the run
before stops nothing.
"""

from __future__ import annotations

import math
import re

import numpy as np

from pluggybot.perception.depth import HEIGHT, MAX_Z, MIN_Z, WIDTH
from pluggybot.telemetry.protocol import PROTOCOL_VERSION, ROBOT_ROOT
from pluggybot.telemetry.recorder import GridSampler

SCENE, STEP, READOUTS, DEPTH, ATTEMPT = "scene", "step", "readouts", "depth", "attempt"
HELLO, REFUSED, STARTED, ENDED, BUSY = "hello", "refused", "started", "ended", "busy"
RUN, CANCEL = "run", "cancel"
#: How a run's process ended (`ended.why`): graded (an attempt made, its
#: verdict streamed), cancelled (the site asked), out of time (the host's
#: own bound on a child, `host.CHILD_S`), or failed (it raised, or its world
#: was unstable: no attempt, no verdict).
ENDINGS = ("graded", "cancelled", "out of time", "failed")
#: The robot's senses each this often, sim s, and its depth camera's
#: picture each this often -- a person reads both, and neither is worth a
#: frame's rate.
READOUT_S, DEPTH_S = 0.5, 1.0
#: The depth picture: the sensor model's own size and range (`perception.
#: depth`). 0 is no return; 1-255 runs near to far.
DEPTH_W, DEPTH_H = WIDTH, HEIGHT
DEPTH_RANGE = (MIN_Z, MAX_Z)
#: A session id as the site mints it, what may stand in a path, and a
#: session's token, what may stand in a child's environment -- read WHOLE
#: (`fullmatch`: `$` let a trailing newline through, and a NUL in a token
#: made `Popen` raise in the host).
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")
INGEST_PREFIX = "/api/pluggyworld/sessions/"


def _number(v):
  if isinstance(v, bool):
    return v
  if isinstance(v, (int, float, np.floating, np.integer)):
    v = float(v)
    return round(v, 4) if math.isfinite(v) else None
  return v if isinstance(v, str) else str(v)


# ---- the stream -------------------------------------------------------------------


def scene_message(scene: dict) -> dict:
  return {"type": SCENE, "scene": scene}


def step_message(t: float, n: int, line: int, verb: str, args: dict,
                 robot: str = ROBOT_ROOT) -> dict:
  return {"type": STEP, "t": round(float(t), 3), "robot": robot, "n": int(n),
          "line": int(line), "verb": verb,
          "args": {k: _number(v) for k, v in args.items()}}


def readouts_message(t: float, values: dict, robot: str = ROBOT_ROOT) -> dict:
  return {"type": READOUTS, "t": round(float(t), 3), "robot": robot,
          "values": {k: None if v is None else _number(v) for k, v in values.items()}}


def depth_image(z: np.ndarray, near: float = DEPTH_RANGE[0],
                far: float = DEPTH_RANGE[1]) -> np.ndarray:
  """A depth frame as a picture: 0 where nothing returned, else 1 (near)
  to 255 (far) over [near, far]."""
  z = np.asarray(z, dtype=float)
  seen = np.isfinite(z)
  scaled = 1.0 + 254.0 * (np.clip(np.where(seen, z, near), near, far) - near) / (far - near)
  return np.where(seen, np.rint(scaled), 0).astype(np.uint8)


class DepthPicture:
  """The `grid` a `DepthSampler` draws: the body's watched depth camera
  (`Body.depth_watched`) as a picture."""

  def __init__(self, body, width: int = DEPTH_W, height: int = DEPTH_H) -> None:
    self.body, self.width, self.height = body, width, height

  def to_image(self) -> np.ndarray:
    return depth_image(self.body.depth_watched(self.width, self.height))


class DepthSampler(GridSampler):
  """What the depth camera sees, on the grid's path: the sink samples it on
  the frames' clock and encodes the PNG off the physics thread."""

  def __init__(self, body, hz: float = 1.0 / DEPTH_S, dedupe: bool = False) -> None:
    super().__init__(DepthPicture(body), hz=hz, dedupe=dedupe)

  def message(self, t: float) -> dict:
    return {"type": DEPTH, "t": round(float(t), 3), "robot": self.root,
            "width": self.grid.width, "height": self.grid.height,
            "near": DEPTH_RANGE[0], "far": DEPTH_RANGE[1]}


def public_run(ran: dict | None) -> dict | None:
  """The language's record of a run for the wire: every verb, what it said
  and what the hand felt, where it stopped and its variables -- not the
  source (the person has it) nor a verb's trace."""
  if ran is None:
    return None
  out = {k: v for k, v in ran.items() if k not in ("source", "steps")}
  out["steps"] = [{k: v for k, v in s.items() if k != "trace"} for s in ran.get("steps", [])]
  return out


def attempt_message(row: dict, ran: dict | None, report: str, t: float,
                    robot: str = ROBOT_ROOT) -> dict:
  """An attempt's end, off its record row (`attempts.attempt_in`): the
  verdict as the grader sealed it -- ⚠ its PUBLIC metrics, never the row's
  `secret` -- the run, and the report the robot's mind is shown of a try."""
  return {"type": ATTEMPT, "t": round(float(t), 3), "robot": robot, "n": row["attempt"],
          "task": row["task"], "verdict": row["verdict"], "own": row["own"],
          "simS": row["simS"], "cut": row["cut"], "steps": row.get("steps"),
          "limitS": row["run"]["budgetS"], "holdS": row["run"]["holdS"],
          "run": public_run(ran), "report": report}


# ---- the host ---------------------------------------------------------------------


def hello_message(build: dict, task: dict, job: str, sections: list[tuple[str, str]],
                  index: dict, source_chars: int) -> dict:
  """`index` is what `sections` list, as data
  (`mind.practice.vocabulary_index`): what a page's guide for people is
  checked against."""
  return {"type": HELLO, "protocolVersion": PROTOCOL_VERSION, "build": build, "task": task,
          "job": job, "vocabulary": [{"heading": h, "text": t} for h, t in sections],
          "index": index, "sourceChars": source_chars}


def refused_message(session: str, attempt: int, reasons: list[str]) -> dict:
  return {"type": REFUSED, "session": session, "attempt": attempt, "reasons": list(reasons)}


def started_message(session: str, attempt: int, chest: int) -> dict:
  return {"type": STARTED, "session": session, "attempt": attempt, "chest": chest}


def ended_message(session: str, attempt: int, why: str, detail: str = "",
                  verdict: dict | None = None) -> dict:
  """How a run's process ended; `verdict` ({ok, reason}) is a graded run's,
  and only a graded run's."""
  if why not in ENDINGS:
    raise ValueError(f"no ending {why!r}: {', '.join(ENDINGS)}")
  if (verdict is None) == (why == "graded"):
    raise ValueError("a run's verdict comes with its ending if, and only if, it was graded")
  msg = {"type": ENDED, "session": session, "attempt": attempt, "why": why, "detail": detail}
  if verdict is not None:
    msg["verdict"] = {"ok": bool(verdict["ok"]), "reason": str(verdict["reason"])}
  return msg


def busy_message(session: str, attempt: int) -> dict:
  return {"type": BUSY, "session": session, "attempt": attempt}


class BadRun(ValueError):
  """A `run` the host cannot take as sent: not a person's procedure refused,
  but the site's message malformed -- said back as an `ended` that failed."""


def parse_run(msg: dict, chests: range) -> dict:
  """The site's `run`, checked: the session it is for, the attempt's
  number, a public chest, the procedure's text, and the session's own
  ingest path and token -- or `BadRun`, every reason at once."""
  bad = []
  session = msg.get("session")
  if not isinstance(session, str) or not SESSION_RE.fullmatch(session):
    bad.append("no session id")
  attempt = msg.get("attempt")
  if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
    bad.append("no attempt number")
  chest = msg.get("chest")
  if not isinstance(chest, int) or isinstance(chest, bool) or chest not in chests:
    bad.append(f"no chest from {chests.start} to {chests.stop - 1}")
  source = msg.get("source")
  if not isinstance(source, str):
    bad.append("no procedure")
  ingest = msg.get("ingest")
  if not (isinstance(ingest, str) and isinstance(session, str)
          and ingest == f"{INGEST_PREFIX}{session}/ingest"):
    bad.append("no ingest path of the session's own")
  token = msg.get("token")
  if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
    bad.append("no session token")
  if bad:
    raise BadRun("; ".join(bad))
  return {"session": session, "attempt": attempt, "chest": chest, "source": source,
          "ingest": ingest, "token": token}
