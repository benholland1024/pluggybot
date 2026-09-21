"""The eye: the robot looks at the world as the site draws it (issue #275).

The robot's cameras are METRIC, not appearance: the tag detector reads
MuJoCo renders for bay standoffs and docking, the depth camera is 8400
raycasts for the height map. None of them says what anything LOOKS like,
and a visitor who asks "what do you think of the landscape?" is looking at
the TresJS world -- the fence with trees behind it -- which MuJoCo does not
draw. This is the split, and the one rule that keeps it honest:

  MUJOCO IS GEOMETRY, TRESJS IS APPEARANCE. What the robot can touch, drive
  on and measure is the sim's; what the world looks like, to a visitor and
  to the robot's eyes, is the site's. The dressing may never contradict the
  geometry where the robot can reach -- the site's half pins it
  (rooftop-media-2026 #321) -- because a tree drawn on free floor is a lie
  in the dangerous direction: the robot would describe it and then drive
  through it.

  AN IMAGE RENDERED FROM THE CAMERA POSE IS THE SENSOR (TaskPattern.md's
  honesty rule, as written). The wire carries the picture, never a
  caption: a code-written "a fence with pines behind it" would be the wire
  discovering what a sensor should. The MIND looks at the picture -- the
  deployed model takes an image on the request (MEASURED, 2026-09-21: on
  the router's `:cheapest` providers, with the deployed schema, ~400 input
  tokens for a 640 x 480 JPEG) -- and which model looked rides the build
  identity as `eyes`, because what the robot "saw" then depends on it.

  THE PATTERN IS THE LIBRARY'S (mind/wiki.py), WITH AN ACTION. `look` is an
  action beside `recall`: stand still, and the picture arrives on the NEXT
  turn as `seen` -- a labelled block on the visitor channel's terms, never
  a message role. The round trip is the ingest socket's, bidirectional
  since issue #16: the sim emits a `look` request with the head camera's
  pose, the website renders from that pose and answers with an `image`
  inbound kind, and a request nobody answers inside `LOOK_S` is `seen:
  none`, said so. The sim never waits on the renderer: standing still IS
  the wait, and the deadline is sim time.

  A LOOK IS SHOWN ONCE and rationed by its run: at most `MAX_LOOK_RUN` in a
  row (a second picture from the same spot is the same picture), then
  `look` leaves the menu for a turn, as `recall` does.

The physics thread owns this object. The image itself arrives on the
socket thread, into the bounded inbox like any other inbound kind, and is
drained here between physics steps.
"""

from __future__ import annotations

import base64
import math
from collections import Counter

import numpy as np

from pluggybot.mind.inbox import VisitorMessage
from pluggybot.telemetry.protocol import LOOK_OUTCOMES

#: Who the picture arrives from, in the block the model is shown.
SENDER = "your head camera"
#: The head camera's MJCF name, reached through the robot's handle
#: (`handle.el(CAMERA)`): the one navigation camera on the head, looking
#: along the body's +x (models/pluggybot_fork.xml).
CAMERA = "left_eye"
#: How long the robot stands still for its picture, in sim seconds, and
#: the deadline: a request the website has not answered by then is `none`.
#: The site's renderer is a headless browser already following the stream,
#: so an answer is a screenshot's worth of time (~1-3 s on the deploy box's
#: software GL); ten is the same figure `RECALL_S` uses for a pause that is
#: honest about taking real seconds, and the round trip rides a socket that
#: is paced to real time on the served world.
LOOK_S = 10.0
#: How many looks may run in a row before `look` leaves the menu for a turn
#: (`looksLeft` in the state). Two: the world may have moved between them,
#: a third from the same spot is the robot staring.
MAX_LOOK_RUN = 2
#: The picture asked for, in pixels. The head camera's fovy is 41 degrees
#: (vertical); 4:3 at 640 x 480 is what the real part would deliver and
#: costs ~400 input tokens as a JPEG, so a look is cheaper than a page.
WIDTH = 640
HEIGHT = 480
#: The bytes a picture may be, and what every JPEG starts with, are the
#: inbox's (`MAX_IMAGE_BYTES`, `JPEG_MAGIC`): checked at the door, because
#: the door is where every other inbound kind is checked.
MEDIA_TYPE = "image/jpeg"


def wrap_degrees(deg: float) -> float:
  """A heading in (-180, 180], and never -0."""
  wrapped = -((-deg + 180.0) % 360.0 - 180.0)
  return 0.0 if wrapped == 0 else wrapped


def camera_pose(model, data, name: str) -> dict:
  """Where the camera IS, in the world, for a renderer to stand in.

  Off `data.cam_xpos` / `cam_xmat` -- the camera's world pose after
  `mj_forward`, which is the pose the tag detector renders from, so the
  picture is taken from exactly where the metric camera is. MuJoCo's camera
  frame looks along its own -z with +y up (the columns of `cam_xmat` are
  the camera's axes in world coordinates), which is why `forward` and `up`
  are sent as world unit vectors: a three.js consumer does `position`,
  `up`, `lookAt(position + forward)` and never has to know that convention.
  `fovy` is the camera's own (degrees, vertical), and the size is what is
  asked for.
  """
  cid = model.camera(name).id
  pos = np.asarray(data.cam_xpos[cid], dtype=float)
  xmat = np.asarray(data.cam_xmat[cid], dtype=float).reshape(3, 3)
  forward = -xmat[:, 2]
  up = xmat[:, 1]
  return {"pos": [round(float(v), 4) for v in pos],
          "forward": [round(float(v), 5) for v in forward],
          "up": [round(float(v), 5) for v in up],
          "fovy": round(float(model.cam_fovy[cid]), 2),
          "width": WIDTH, "height": HEIGHT}


class Eye:
  """One robot's looks: the open request, the record of every look, and
  the picture waiting for the next turn.

  ONE REQUEST AT A TIME, by construction: a look stands still until it is
  answered or times out, so a second `ask` before the first resolved is a
  programming error and is raised on, not queued.
  """

  def __init__(self, root: str, wait_s: float = LOOK_S) -> None:
    self.root = root
    self.wait_s = float(wait_s)
    self.pending: dict | None = None
    #: Every look asked for, in order, as the row the wire and the run
    #: record carry -- `asked` when the request goes out, then `seen` or
    #: `none` when it resolves (the same row, updated in place).
    self.looks: list[dict] = []
    self.dropped: Counter = Counter()
    self._count = 0

  # ---- the request ------------------------------------------------------------

  def ask(self, camera: dict, t: float, x: float, y: float,
          heading: float) -> dict:
    """Open a request. Returns the row the `look` event carries."""
    if self.pending is not None:
      raise RuntimeError(f"a look is already open ({self.pending['ref']})")
    self._count += 1
    row = {"ref": f"look:{self.root}:{self._count}", "robot": self.root,
           "t": round(float(t), 3), "outcome": "asked",
           "camera": dict(camera),
           "at": {"x": round(float(x), 2), "y": round(float(y), 2),
                  # Wrapped to (-180, 180]: the reckoner's heading
                  # accumulates turns, and "facing 450 deg" is nobody's.
                  "headingDeg": round(wrap_degrees(math.degrees(float(heading))), 1)},
           "bytes": 0, "waitS": 0.0, "why": ""}
    self.pending = row
    self.looks.append(row)
    return row

  # ---- the answer (physics thread, off the inbox) -----------------------------

  def offer(self, msg: VisitorMessage, t: float) -> dict | None:
    """An `image` message drained from the inbox: the open request's
    answer, or nothing.

    A picture for a request that is not open -- one that timed out, one
    already answered, or one this robot never made -- is dropped and
    counted by why, because a late renderer must not be able to hand the
    robot a picture of where it used to be.
    """
    if msg.kind != "image" or not msg.image:
      self.dropped["not-an-image"] += 1
      return None
    if self.pending is None or msg.ref != self.pending["ref"]:
      self.dropped["stale"] += 1
      return None
    row = self.pending
    self.pending = None
    row.update(outcome="seen", bytes=len(msg.image),
               waitS=round(float(t) - row["t"], 3))
    row["_jpeg"] = bytes(msg.image)
    return row

  def give_up(self, t: float, why: str = "unanswered") -> dict | None:
    """The deadline passed with no picture: `none`, said so."""
    if self.pending is None:
      return None
    row = self.pending
    self.pending = None
    row.update(outcome="none", why=why,
               waitS=round(float(t) - row["t"], 3))
    return row

  def overdue(self, t: float) -> bool:
    return self.pending is not None and float(t) - self.pending["t"] >= self.wait_s

  # ---- the record -------------------------------------------------------------

  def stats(self) -> dict:
    return {"asked": len(self.looks),
            "seen": sum(1 for r in self.looks if r["outcome"] == "seen"),
            "none": sum(1 for r in self.looks if r["outcome"] == "none"),
            "dropped": dict(self.dropped)}


def wire_row(row: dict) -> dict:
  """The row as the wire and the record carry it: never the bytes."""
  out = {k: v for k, v in row.items() if not k.startswith("_")}
  assert out["outcome"] in LOOK_OUTCOMES
  return out


def as_context(row: dict) -> dict:
  """How the model is shown a look on its next turn: the visitor channel's
  block -- an id, a `from`, a `text` -- plus where the picture was taken
  from and whether one came. The JPEG rides beside the block as `jpeg`
  (base64), which `overseer.model_state` STRIPS from the text the model
  reads and `Overseer._call` attaches as an IMAGE part of the same user
  turn: the picture is a sensor reading, never a string in the JSON.
  Built through `VisitorMessage` so the framing IS the visitor channel's
  rather than resembling it.
  """
  seen = row["outcome"] == "seen"
  text = ("the picture you took, attached to this message as an image"
          if seen else
          f"no picture came back inside {row.get('waitS', 0.0):.0f} s")
  msg = VisitorMessage(id=str(row["ref"]), kind="message", who=SENDER,
                       text=text)
  out = {**msg.as_context(), "at": dict(row["at"]),
         "image": "attached" if seen else "none"}
  if not seen:
    out["why"] = row.get("why", "")
  if seen:
    out["jpeg"] = base64.b64encode(row["_jpeg"]).decode("ascii")
  return out
