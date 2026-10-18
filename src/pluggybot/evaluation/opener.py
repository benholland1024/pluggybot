"""The humanoid's reference opener (issue #493, stage 2): our reactive opener
as code, on the humanoid's hands -- the cheapest other way to open the
chest, as `legs/probe.py` is the quadruped's -- built on `Body`'s hand
members alone (`tests/test_body.py` holds it to them), so the vocabulary a
procedure is given (`procedure.steps`' hand verbs) can say everything it
does.

CODE'S FORESIGHT, stated -- what a robot learning the job would have to
find for itself:
  - the knob wears tag 53 (`rack.tags.CHEST_TAG_IDS`) on every face, and is
    a cube `KNOB_M` on a side: its middle half a cube in from the face read;
  - a knob that hangs at a box's front is gripped from the front, the
    fingers tilted down onto it (`TILTS_DEG`) and the jaws closing across
    it: the turn #507's demo found (`humanoid_spike.demo_scene`);
  - a lid lifts UP, and comes down the way it went up;
  - how hard and how far to pull (`PULLS`), and how fast to lower (`DOWN_M_S`).

THE STEPS (`opener`), from the robot's senses alone:
  look    standing still, the knob's tag read `LOOKS` times
  grip    the gripper opened, the hand on the side the knob is to reached
          to it, shut, and the grip judged by the gripper's own word -- a
          grip that did not hold opened, withdrawn and tried at the next tilt
  pull    the compliant pull (`Body.pull_routine`), straight up, its way
          turning to whatever the lid gives -- and again, harder, where the
          lid held fast before it rose `RISE_M`
  lower   the hand carried straight back down to where it gripped
          (`Body.move_hand_routine`) at `DOWN_M_S`, pivoting round its
          jaws as the handle swings
  let go  the gripper opened and the hand withdrawn along its way in

Its own word (what an attempt records as `own`): ok once the pull rose
`RISE_M` and the hand came back down to where it gripped, the knob still
held -- never the lid's angle, which it does not see.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.rack.tags import CHEST_TAG_IDS
from pluggybot.tick import Routine

#: The knob's tag, and the knob's size, m: the cube the claw is proven on.
TAG, KNOB_M = CHEST_TAG_IDS[0], 0.026
#: Standing still before it looks, s, and its looks, `APART_S` apart.
STILL_S, LOOKS, APART_S = 1.0, 3, 0.1
#: The fingers' tilt down onto the knob, tried in this order, deg.
TILTS_DEG = (25.0, 15.0, 35.0, 45.0)
#: Grips tried before it gives up.
GRIPS = 2
#: The pulls, in turn while the lid holds fast: the force limit, N, and the
#: distance, m. Straight up.
PULLS = ((20.0, 0.30), (35.0, 0.30))
#: A pull that raised the hand this far, m, opened the lid: the knob's
#: drop handle hangs 0.27 m from the lid's hinge, a pull that rose less
#: lifted it less than 50 deg.
RISE_M = 0.20
#: The lowering's speed, m/s, and patience, s.
DOWN_M_S, DOWN_PATIENCE_S = 0.06, 30.0
#: The pull's speed, m/s.
UP_M_S = 0.06


def _frame(approach, closing) -> np.ndarray:
  """A grip's turn: the fingers along `approach`, the jaws closing along
  `closing` (`Body.reach_routine`: fingers along its -z, jaws along its x)."""
  a = np.asarray(approach, dtype=float)
  a = a / np.linalg.norm(a)
  x = np.asarray(closing, dtype=float)
  x = x - a * (x @ a)
  x = x / np.linalg.norm(x)
  z = -a
  return np.column_stack([x, np.cross(z, x), z])


def _rz(yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def knob_seen(body) -> np.ndarray | None:
  """The knob's middle, the believed world frame: each look's face half a
  cube in along its normal, averaged; None if no look read the tag."""
  seen = []
  for _ in range(LOOKS):
    yield from body.hold_routine(APART_S)
    got = body.spot(TAG)
    if got is not None:
      seen.append(np.array([got["x"], got["y"], got["z"]])
                  + KNOB_M / 2 * np.asarray(got["normal"], dtype=float))
  return np.mean(seen, axis=0) if seen else None


def grip_routine(body, knob: np.ndarray, side: str, log: dict) -> Routine:
  """The knob gripped (the module docstring's `grip`): True once the
  gripper says it holds it."""
  sx, sy, yaw = body.stance
  for tilt in TILTS_DEG[:GRIPS * 2]:
    t = math.radians(tilt)
    rot = _rz(yaw) @ _frame((math.cos(t), 0.0, -math.sin(t)), (0.0, 1.0, 0.0))
    yield from body.grip_routine(side, False)
    reached = yield from body.reach_routine(side, knob, rot, stiffness=1.0, patience=20.0)
    log.setdefault("reaches", []).append({"tiltDeg": tilt, "reached": bool(reached)})
    if not reached:
      continue
    held = yield from body.grip_routine(side, True)
    log["reaches"][-1]["held"] = bool(held)
    if held:
      return True
    yield from body.grip_routine(side, False)
    yield from body.release_routine(side)
    if len([r for r in log["reaches"] if r["reached"]]) >= GRIPS:
      break
  return False


def opener(body) -> Routine:
  """The reference (the module docstring): its own word, `{ok, why, ...}`."""
  log: dict = {}
  yield from body.hold_routine(STILL_S)
  knob = yield from knob_seen(body)
  if knob is None:
    return {"ok": False, "why": "the knob's tag not seen", **log}
  sx, sy, yaw = body.stance
  across = float(_rz(yaw).T[1] @ (knob - [sx, sy, 0.0]))
  side = "left" if across > 0.0 else "right"
  log.update(side=side, knob=[round(float(v), 4) for v in knob])
  if not (yield from grip_routine(body, knob, side, log)):
    return {"ok": False, "why": "never held the knob", **log}
  start = np.asarray(body.hand_at(side), dtype=float)
  rise = 0.0
  for limit, distance in PULLS:
    got = yield from body.pull_routine(side, (0.0, 0.0, 1.0), limit, distance,
                                       stiffness=1.0, speed=UP_M_S)
    log.setdefault("pulls", []).append(got)
    rise = float(np.asarray(body.hand_at(side))[2] - start[2])
    if rise >= RISE_M or got["why"] in ("let go", "fell"):
      break
  down = yield from body.move_hand_routine(side, start, stiffness=1.0, speed=DOWN_M_S,
                                           patience=DOWN_PATIENCE_S)
  log["down"] = down
  still_held = body.gripped(side)
  yield from body.grip_routine(side, False)
  yield from body.release_routine(side)
  ok = rise >= RISE_M and down["why"] == "arrived" and still_held
  why = (f"lifted the knob {rise:.2f} m and lowered it back" if ok
         else f"the pull rose {rise:.2f} m ({log['pulls'][-1]['why']})" if rise < RISE_M
         else f"lowering ended {down['why']}" if down["why"] != "arrived"
         else "the knob left the jaws on the way down")
  return {"ok": ok, "why": why, "rise": round(rise, 4), **log}
