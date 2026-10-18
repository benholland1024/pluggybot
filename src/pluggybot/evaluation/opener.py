"""The humanoid's reference opener (issue #493, stage 2): our opener as code
on the humanoid's hands -- the cheapest other way to open the chest, and the
quadruped's probe's (`legs/probe.py`) plan on another body: the box
measured, the lid's arc planned, the hand carried along it and back. Built
on `Body`'s members alone: the body is all a skill is handed.

CODE'S FORESIGHT, stated -- what a robot learning the job would have to
find for itself, as the probe's docstring states its own:
  - the knob wears tag 53 (`rack.tags.CHEST_TAG_IDS`) on every face, a cube
    `KNOB_M` on a side: its middle half a cube in from the face read;
  - the box in front is lidded, its lid hinged along the back edge of its
    top, and the knob hangs on a drop handle from a pin at the tip of the
    bracket out of the lid's front edge: gripped, the knob moves as that pin
    does about the hinge (`legs.probe.pin_path`). The top, its back edge and
    the bracket's tip are MEASURED off the D435's depth at the part's 848 x
    480 (`legs.probe.guess`; the hinge a board's thickness high, as there);
  - it is gripped from the front, the fingers tilted and the jaws closing
    along the hinge's line -- at the first tilt (`TILTS_DEG`) from which
    the robot's model of itself FOLLOWS the whole arc (`Body.hand_follows`);
  - held at its turn, the hand keeps the handle as it hung while the lid
    turns under it, until the handle's arm meets the lid's front edge: so
    past `FREE_RAD` the hand turns as the lid does;
  - up to `TOP` and back down to `BOTTOM`, at `SPEED`: a hinge guessed a
    centimetre or two off is taken up by the arm's give, as the probe's Kp
    8 takes it.

THE STEPS (`opener`), from the robot's senses alone:
  look    standing still, the knob's tag read `LOOKS` times, and the box off
          `FRAMES` depth frames (`Body.depth_cloud`), looked at `LOOK_TRIES`
          times at most
  plan    the arc's waypoints, `STEP` apart, and the tilt that reaches them
  grip    the gripper opened, the hand on the side the knob is to reached
          to it at that tilt, shut, and the grip judged by the gripper's own
          word -- a grip that did not hold opened, withdrawn and tried again
  lift    the hand carried along the arc to `TOP` (`Body.move_hand_routine`)
  lower   ...and back down it to `BOTTOM`, from as far as it went
  let go  the gripper opened and the hand withdrawn along its way in

Its own word (what an attempt records as `own`): ok once the hand went up
the arc to `TOP` and back down it, the knob still held -- never the lid's
angle, which it does not see.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.legs.probe import DEPTH_H, DEPTH_MIN_Z, DEPTH_W, Guess, guess, rot_about
from pluggybot.rack.tags import CHEST_TAG_IDS
from pluggybot.tick import Routine

#: The knob's tag, and the knob's size, m: the cube the claw is proven on.
TAG, KNOB_M = CHEST_TAG_IDS[0], 0.026
#: Standing still before it looks, s, its looks, `APART_S` apart, its depth
#: frames, and how many times it looks for the box.
STILL_S, LOOKS, APART_S, FRAMES, LOOK_TRIES = 1.0, 3, 0.1, 3, 2
#: Where the knob is not in its colour camera's view standing -- the bottom
#: of its view from a step nearer -- it bows this far to look, rad.
GAZE_DOWN = 0.25
#: The fingers' tilt down onto the knob, deg (up, -), tried in this order
#: for the first that follows the whole arc: the hand turns down with the
#: lid past `FREE_RAD`, and the arm reaches the arc's top with the fingers
#: about level (#493).
TILTS_DEG = (0.0, -10.0, 10.0, -20.0, 20.0, 30.0)
#: The lid's angle past which the hand turns with it, rad: held at its turn,
#: the hand keeps the handle as it hung -- its pin 33 deg round from the
#: lid's own frame, shut -- and at 57 deg of lid the handle's arm lies on
#: the lid's front edge (`activity.chest.PIN_AHEAD`); pulled on past it, the
#: wrist gave 7-11 deg and the lid went on to 74 (#493).
FREE_RAD = math.radians(50.0)
#: Grips tried before it gives up.
GRIPS = 2
#: The arc: up to this, rad of lid (the probe's 1.2 a little short of the
#: arm's stretch), back down to this, rad -- under the grader's 5 deg -- in
#: steps of this, rad, at this speed, m/s, on an arm this soft.
TOP, BOTTOM, STEP = math.radians(64.0), -0.03, math.radians(4.0)
#: The tops tried, the highest first, rad: the arm's give leaves the lid a
#: few degrees under the plan's top (a plan to 64 deg opened lids 61-63, to
#: 60 deg 58).
TOPS = (TOP, math.radians(60.0))
SPEED, STIFFNESS = 0.05, 1.0
#: How long one step of the arc may take, s; how many times the hand is
#: carried back down the arc while still up, and up by less than this, m,
#: it is down.
STEP_PATIENCE_S, DOWNS, HOME_M = 6.0, 3, 0.03


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


def knob_seen(body) -> Routine:
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


def measure(body, knob: np.ndarray) -> Routine:
  """The box off the depth camera (`legs.probe.guess`): a `Guess`, or why
  there is none."""
  cloud = yield from body.depth_cloud(DEPTH_W, DEPTH_H, DEPTH_MIN_Z, frames=FRAMES,
                                      apart=APART_S)
  # ⚠ ONLY WHAT STANDS OVER THE KNOB: a lid's top is above its own knob, and
  # the counter's top -- lower, wider and in full view -- was read for it
  cloud = cloud[cloud[:, 2] > knob[2]]
  x, y, _ = body.stance
  return guess(cloud, knob, math.atan2(knob[1] - y, knob[0] - x))


def grip_turn(g: Guess, tilt_deg: float) -> np.ndarray:
  """The grip's turn, world frame: the fingers along the box's facing tilted
  `tilt_deg` down, the jaws closing along the hinge's line."""
  t = math.radians(tilt_deg)
  face = np.array([math.cos(g.yaw), math.sin(g.yaw), 0.0])
  return _frame(math.cos(t) * face + np.array([0.0, 0.0, -math.sin(t)]), g.axis)


def arc(g: Guess, start: np.ndarray, rot: np.ndarray, top: float = TOP,
        bottom: float = BOTTOM, step: float = STEP) -> tuple[list, list]:
  """The hand's waypoints up the lid's arc from `start` (the knob as it was
  gripped, at the turn `rot`) to `top`, and back down it to `bottom` (rad
  of lid), each (point, turn): the knob moves as the pin does, and past
  `FREE_RAD` turns round it as the lid does."""
  pin0, hinge = np.asarray(g.pin, dtype=float), np.asarray(g.hinge, dtype=float)

  def at(s: float) -> tuple:
    pin = hinge + rot_about(g.axis, s) @ (pin0 - hinge)
    turn = rot_about(g.axis, max(0.0, s - FREE_RAD))
    return pin + turn @ (np.asarray(start, dtype=float) - pin0), turn @ rot
  # ...ending AT `top`: stepped alone, an arc stopped on the last step under
  # it, and a plan said 66 deg that went to 64
  rises = list(np.arange(step, top - 1e-9, step)) + [top]
  ups = [at(s) for s in rises]
  downs = [at(s) for s in np.arange(top - step, bottom - 1e-9, -step)]
  return ups, downs


def follows(body, side: str, start: np.ndarray, rot: np.ndarray, points: list) -> bool:
  """Whether the robot's model of itself follows `points` ((point, turn)
  each) from `start` at `rot` (`Body.hand_follows`)."""
  return body.hand_follows(side, [start] + [p for p, _ in points],
                           [rot] + [r for _, r in points])


def plans(body, side: str, g: Guess, start: np.ndarray) -> list:
  """Every (tilt, top) -- the highest top first (`TOPS`), at each the tilts
  in turn (`TILTS_DEG`) -- at which the robot's model of itself FOLLOWS the
  arc from the knob up."""
  out = []
  for top in TOPS:
    for tilt in TILTS_DEG:
      rot = grip_turn(g, tilt)
      if follows(body, side, start, rot, arc(g, start, rot, top=top)[0]):
        out.append((tilt, top))
  return out


def grip_routine(body, knob: np.ndarray, side: str, rot: np.ndarray, log: dict,
                 ups: list) -> Routine:
  """The knob gripped at the turn `rot`, in a posture from which the hand
  then follows `ups` (`Body.reach_routine`'s path): True once the gripper
  says it holds it, a grip that did not hold opened, withdrawn and tried
  again."""
  for _ in range(GRIPS):
    yield from body.grip_routine(side, False)
    reached = yield from body.reach_routine(side, knob, rot, stiffness=1.0, patience=20.0,
                                            path=[p for p, _ in ups],
                                            turns=[r for _, r in ups])
    log.setdefault("reaches", []).append({"reached": bool(reached)})
    if not reached:
      continue
    held = yield from body.grip_routine(side, True)
    log["reaches"][-1]["held"] = bool(held)
    if held:
      return True
    yield from body.grip_routine(side, False)
    yield from body.release_routine(side)
  return False


def carry_routine(body, side: str, points: list, log: list) -> Routine:
  """The hand carried through `points` ((point, turn) each) as one motion
  (`Body.move_hand_routine`'s path and turns): True once it arrived at the
  last, its end said in `log`."""
  got = yield from body.move_hand_routine(side, points[-1][0], stiffness=STIFFNESS,
                                          speed=SPEED, patience=STEP_PATIENCE_S * len(points),
                                          path=[p for p, _ in points[:-1]],
                                          turns=[r for _, r in points])
  log.append(got)
  return got["why"] in ("arrived", "held back")


def opener(body) -> Routine:
  """The reference (the module docstring): its own word, `{ok, why, ...}`."""
  log: dict = {"looks": []}
  yield from body.hold_routine(STILL_S)
  g, knob = "the knob's tag not seen", None
  for i in range(LOOK_TRIES):
    knob = yield from knob_seen(body)
    if knob is None and i == 0:
      yield from body.gaze_routine(GAZE_DOWN)
      log["gazed"] = GAZE_DOWN
      knob = yield from knob_seen(body)
    if knob is not None:
      g = yield from measure(body, knob)
    log["looks"].append(g.as_dict() if isinstance(g, Guess) else g)
    if isinstance(g, Guess):
      break
  if log.get("gazed"):
    yield from body.gaze_routine(0.0)
  if not isinstance(g, Guess):
    return {"ok": False, "why": f"the box not made out: {g}", **log}
  sx, sy, yaw = body.stance
  across = -math.sin(yaw) * (knob[0] - sx) + math.cos(yaw) * (knob[1] - sy)
  side = "left" if across > 0.0 else "right"
  log["side"] = side
  options = plans(body, side, g, knob)
  log["plans"] = [list(o) for o in options]
  if not options:
    return {"ok": False, "why": "no tilt puts the lid's arc in reach", **log}
  ups = None
  for tilt, top in options[:GRIPS + 1]:
    rot = grip_turn(g, tilt)
    if not (yield from grip_routine(body, knob, side, rot, log,
                                    arc(g, knob, rot, top=top)[0])):
      continue
    start = np.asarray(body.hand_at(side), dtype=float)
    ups, downs = arc(g, start, grip_turn(g, tilt), top=top)
    # ...and from the arm as its grip left it: another branch is no path
    if follows(body, side, start, grip_turn(g, tilt), ups):
      log["tiltDeg"], log["top"] = tilt, top
      break
    log.setdefault("regrips", []).append([tilt, top])
    yield from body.grip_routine(side, False)
    yield from body.release_routine(side)
    ups = None
  if ups is None:
    return {"ok": False, "why": "never held the knob where its arm could follow the arc",
            **log}
  log["up"], log["down"] = [], []
  went = yield from carry_routine(body, side, ups, log["up"])
  # back down from wherever it got to: a little down the arc from its
  # nearest point, and again while the hand is still up -- the jaws never
  # open over an open lid (let go there, it fell at 7 rad/s)
  came = False
  for _ in range(DOWNS):
    here = np.asarray(body.hand_at(side), dtype=float)
    i = int(np.argmin([np.linalg.norm(here - p) for p, _ in downs]))
    came = yield from carry_routine(body, side, downs[min(i + 1, len(downs) - 1):],
                                    log["down"])
    if came or here[2] - start[2] < HOME_M:
      break
  if not came and float(np.asarray(body.hand_at(side))[2] - start[2]) > HOME_M:
    return {"ok": False, "why": f"down the arc: {log['down'][-1]['why']}, the knob held up",
            **log}
  still_held = body.gripped(side)
  yield from body.grip_routine(side, False)
  yield from body.release_routine(side)
  ok = went and came and still_held
  why = (f"carried the knob up the arc to {math.degrees(top):.0f} deg and back" if ok
         else f"up the arc: {log['up'][-1]['why']}" if not went
         else f"down the arc: {log['down'][-1]['why']}" if not came
         else "the knob left the jaws on the way")
  return {"ok": ok, "why": why, **log}
