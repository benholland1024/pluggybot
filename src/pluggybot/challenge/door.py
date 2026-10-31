"""Open the cupboard's door by its lever and let go of it (issue #525): one
of the hands' capability checks (docs/Hands.md), a predicate written before
any procedure flies it (docs/Challenges.md).

The door is `activity/door.py`'s: its lever must be pressed past most of
its travel before the latch frees the door, and the door's hinge friction
and damping, its weight and the lever's spring are hidden and drawn. Code
measures the door and the robot off the true state, and the verdict goes
through `scoring.evaluate` like every other. Graded and unpaid -- its row
in `challenges.json` is base 0 -- and shown to no mind.

THE CRITERIA, written first. An attempt passes when:

  1. OPENED: the door is open at least `OPEN_RAD` at the call, and still at
     least that far `HOLD_S` later (its hinge holds it where it is left);
  2. LET GO: nothing but the cupboard itself touches the door, its lever or
     the lever's tip at any moment of the hold;
  3. IN TIME: the call came within the task's time limit of the attempt's
     start (`limitS`; `lid.TIME_LIMIT_S` where none is given);
  4. NEVER FELL: the body's own watch took no fall from the attempt's start
     to the end of the hold, the lid's criterion 6 (`lid.judged`): a hard
     pull that swung the door swung the body over with it, every try of
     one learned skill (#535).

Said beside, and gating nothing: the most it opened, how far the lever was
ever pressed, the fastest the door swung and the body's worst tilt. How it
presses the lever and which way it pulls are the robot's business.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.challenge import lid
from pluggybot.contacts import contact_pairs

#: 1: open this far, rad (50 deg): an arm reaches in past the door's edge.
OPEN_RAD = math.radians(50.0)
#: The hold and the default time, the lid's.
HOLD_S, TIME_LIMIT_S = lid.HOLD_S, lid.TIME_LIMIT_S


def parts(model, name: str = "door") -> tuple[np.ndarray, np.ndarray]:
  """Two masks over the geoms: the cupboard's own and the MOVING parts --
  the door, its lever and the lever's tip."""
  own = lid._subtree(model, model.body(name).id)[model.geom_bodyid]
  moving = lid._subtree(model, model.body(f"{name}_door").id)[model.geom_bodyid]
  return own, moving


class Watch:
  """OUR reading of an attempt, every physics step (a step hook): the most
  the door opened, the lever's deepest press, the fastest the door swung,
  the robot's worst tilt and its falls, and -- once `hold` -- what else
  touched the moving parts. Off the true state. The robot never sees it."""

  def __init__(self, model, data, body=None, name: str = "door") -> None:
    self.model, self.data, self.body, self.name = model, data, body, name
    hinge, lever = model.joint(f"{name}_hinge").id, model.joint(f"{name}_lever_pivot").id
    self.qadr, self.dof = int(model.jnt_qposadr[hinge]), int(model.jnt_dofadr[hinge])
    self.lever = int(model.jnt_qposadr[lever])
    self.own, self.moving = parts(model, name)
    self.most = -math.inf
    self.pressed = 0.0
    self.fastest = 0.0
    self.worst_tilt = None if body is None else 0.0
    self.falls0 = None if body is None else int(body.falls)
    self.holding = False
    self.touched: set[str] = set()

  def hold(self) -> None:
    self.holding = True

  def falls(self) -> int | None:
    return None if self.body is None else int(self.body.falls) - self.falls0

  def kept_state(self) -> dict:
    """What a restart carries (#559: a served job is watched from its
    claim): the attempt's readings and its falls so far, never the hold's --
    a job a restart cut is called again."""
    return {"most": None if self.most == -math.inf else self.most, "pressed": self.pressed,
            "fastest": self.fastest, "worstTilt": self.worst_tilt, "falls": self.falls()}

  def restore_kept(self, state: dict) -> None:
    self.most = -math.inf if state.get("most") is None else float(state["most"])
    self.pressed = float(state.get("pressed", 0.0))
    self.fastest = float(state.get("fastest", 0.0))
    self.worst_tilt = state.get("worstTilt", self.worst_tilt)
    lid.kept_falls(self, state)

  def __call__(self) -> None:
    a = float(self.data.qpos[self.qadr])
    self.most = max(self.most, a)
    self.pressed = max(self.pressed, float(self.data.qpos[self.lever]))
    self.fastest = max(self.fastest, abs(float(self.data.qvel[self.dof])))
    if self.body is not None:
      self.worst_tilt = max(self.worst_tilt, lid.tilt(self.body.orientation()))
    if self.holding:
      g = contact_pairs(self.data)
      if g.shape[0]:
        a_, b_ = g[:, 0], g[:, 1]
        hit = (self.moving[a_] & ~self.own[b_]) | (self.moving[b_] & ~self.own[a_])
        for x, y in g[hit].tolist():
          other = y if self.moving[x] else x
          self.touched.add(self.model.geom(other).name or f"geom {other}")


def measure(model, data, name: str = "door") -> dict:
  """One snapshot of the door, as the grader sees it: its angle off shut,
  deg, off its hinge."""
  hinge = model.joint(f"{name}_hinge").id
  return {"t": float(data.time), "angleDeg": math.degrees(float(data.qpos[model.jnt_qposadr[hinge]]))}


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 started: float | None = None, limit_s: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED. Absences stay absent."""
  call, end, w = at_call or {}, settled or {}, watch
  return {
    "mostDeg": math.degrees(w.most) if w is not None and w.most > -math.inf else None,
    "pressedDeg": math.degrees(w.pressed) if w is not None else None,
    "fastestRadS": w.fastest if w is not None else None,
    "tiltDeg": (math.degrees(w.worst_tilt)
                if w is not None and w.worst_tilt is not None else None),
    "falls": w.falls() if w is not None else None,
    "touchedDuringHold": sorted(w.touched) if w is not None else [],
    "angleAtCallDeg": call.get("angleDeg"),
    "angleDeg": end.get("angleDeg"),
    "holdS": (end["t"] - call["t"]) if "t" in call and "t" in end else None,
    "elapsedS": (call["t"] - started) if "t" in call and started is not None else None,
    "limitS": None if limit_s is None else float(limit_s),
  }


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"mostDeg": 2, "pressedDeg": 1, "fastestRadS": 3, "tiltDeg": 1, "angleAtCallDeg": 2,
         "angleDeg": 2, "holdS": 2, "elapsedS": 2, "limitS": 1}


def eval_door(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate: pure, judged on the numbers as measured
  and shown rounded, a fall first."""
  raw = {k: m.get(k) for k in ("mostDeg", "pressedDeg", "fastestRadS", "tiltDeg", "falls",
                               "angleAtCallDeg", "angleDeg", "holdS", "elapsedS", "limitS")}
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items()}
  metrics["touchedDuringHold"] = list(m.get("touchedDuringHold") or ())
  for key in ("mostDeg", "angleAtCallDeg", "angleDeg", "holdS", "elapsedS"):
    if raw[key] is None:
      return False, metrics, "the door was never measured"
  ok, reason = lid.judged(raw["falls"], *_door(raw, metrics))
  return ok, metrics, reason


def _door(raw: dict, metrics: dict) -> tuple[bool, str]:
  """Criteria 1 to 3 on measurements all there: the verdict and its
  reason."""
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    return False, (f"measured {float(raw['holdS']):.1f} s after the call, "
                   f"not the {HOLD_S:.0f} s the job asks for")
  limit = TIME_LIMIT_S if raw["limitS"] is None else float(raw["limitS"])
  if float(raw["elapsedS"]) > limit + 1e-6:
    return False, (f"called {float(raw['elapsedS']):.0f} s after the start, past "
                   f"the {limit:.0f} s the job allows")
  most, at_call, end = (float(raw[k]) for k in ("mostDeg", "angleAtCallDeg", "angleDeg"))
  want = math.degrees(OPEN_RAD)
  if at_call < want:
    if most < 1.0:
      return False, f"the door never opened: it must open past {want:.0f} deg"
    if most < want:
      return False, f"the door opened {most:.0f} deg, short of the {want:.0f} it must"
    return False, (f"the door opened {most:.0f} deg but was {at_call:.0f} deg open "
                   f"when you finished, short of the {want:.0f} it must")
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, (f"the door {at_call:.0f} deg open, but {touched} still touched it "
                   "after the call -- let go")
  if end < want:
    return False, (f"the door {at_call:.0f} deg open at the call, but {end:.0f} deg "
                   f"{HOLD_S:.0f} s later")
  return True, f"the door {end:.0f} deg open in {float(raw['elapsedS']):.0f} s, and let go"


def sample_door(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`), as the drawer's:
  the snapshot at the call, the watch, and the world measured again NOW,
  after the hold. `result`, the skill's own word, is unread."""
  watch = before.get("watch")
  name = watch.name if watch is not None else "door"
  return measurements(before.get("at_call"), measure(life.model, life.data, name), watch,
                      started=before.get("started"), limit_s=before.get("limitS"))
