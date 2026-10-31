"""Open the drawer and let go of it (issue #525): one of the hands'
capability checks (docs/Hands.md), a predicate written before any procedure
flies it (docs/Challenges.md).

The drawer is `activity/drawer.py`'s, its runners' friction and damping
and its load hidden and drawn. Code measures the drawer and the robot off
the true state, and the verdict goes through `scoring.evaluate` like every
other. It is graded and unpaid -- its row in `challenges.json` is base 0 --
and shown to no mind; the served humanoids' job on the storeroom's counter
is graded by this predicate under a row of its own (`open_drawer`, #559).

THE CRITERIA, written first. An attempt passes when:

  1. OPENED: the drawer is out at least `OPEN_M` at the call, and still
     out at least that far `HOLD_S` later (its runners hold it where it is
     left);
  2. LET GO: nothing but the cabinet itself touches the drawer, its front
     or its knob at any moment of the hold;
  3. IN TIME: the call came within the task's time limit of the attempt's
     start (`limitS`; `lid.TIME_LIMIT_S` where none is given);
  4. NEVER FELL: the body's own watch took no fall from the attempt's start
     to the end of the hold, the lid's criterion 6 (`lid.judged`).

Said beside, and gating nothing: the most it was out, the fastest it moved
and the body's worst tilt. Everything else is the robot's business: how it
finds the knob, which way it pulls, whether it pushes the drawer past its
end.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.challenge import lid
from pluggybot.rack.coupling import contact_pairs

#: 1: out this far, m: a drawer a hand can reach into (its tray 0.26 m
#: deep), and 0.09 short of its runners' end.
OPEN_M = 0.15
#: The hold and the default time, the lid's.
HOLD_S, TIME_LIMIT_S = lid.HOLD_S, lid.TIME_LIMIT_S


def parts(model, name: str = "drawer") -> tuple[np.ndarray, np.ndarray]:
  """Two masks over the geoms: the cabinet's own and the MOVING parts --
  the drawer, its front, its stem and its knob."""
  own = lid._subtree(model, model.body(name).id)[model.geom_bodyid]
  moving = lid._subtree(model, model.body(f"{name}_drawer").id)[model.geom_bodyid]
  return own, moving


class Watch:
  """OUR reading of an attempt, every physics step (a step hook): the most
  the drawer was out, the fastest it moved, the robot's worst tilt and its
  falls, and -- once `hold` -- what else touched the moving parts. Off the
  true state: the slide's `qpos` and `qvel`, the contact list, the body's
  attitude and its fall count. The robot never sees it."""

  def __init__(self, model, data, body=None, name: str = "drawer") -> None:
    self.model, self.data, self.body, self.name = model, data, body, name
    slide = model.joint(f"{name}_slide").id
    self.qadr = int(model.jnt_qposadr[slide])
    self.dof = int(model.jnt_dofadr[slide])
    self.own, self.moving = parts(model, name)
    self.most = -math.inf
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
    return {"most": None if self.most == -math.inf else self.most, "fastest": self.fastest,
            "worstTilt": self.worst_tilt, "falls": self.falls()}

  def restore_kept(self, state: dict) -> None:
    self.most = -math.inf if state.get("most") is None else float(state["most"])
    self.fastest = float(state.get("fastest", 0.0))
    self.worst_tilt = state.get("worstTilt", self.worst_tilt)
    lid.kept_falls(self, state)

  def __call__(self) -> None:
    out, v = float(self.data.qpos[self.qadr]), abs(float(self.data.qvel[self.dof]))
    self.most = max(self.most, out)
    self.fastest = max(self.fastest, v)
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


def measure(model, data, name: str = "drawer") -> dict:
  """One snapshot of the drawer, as the grader sees it: how far it is out,
  m, off its slide."""
  slide = model.joint(f"{name}_slide").id
  return {"t": float(data.time), "outM": float(data.qpos[model.jnt_qposadr[slide]])}


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 started: float | None = None, limit_s: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED: the snapshot at the call, the one
  after the hold, and the attempt's per-step reading. Absences stay absent."""
  call, end, w = at_call or {}, settled or {}, watch
  return {
    "mostM": w.most if w is not None and w.most > -math.inf else None,
    "fastestMS": w.fastest if w is not None else None,
    "tiltDeg": (math.degrees(w.worst_tilt)
                if w is not None and w.worst_tilt is not None else None),
    "falls": w.falls() if w is not None else None,
    "touchedDuringHold": sorted(w.touched) if w is not None else [],
    "outAtCallM": call.get("outM"),
    "outM": end.get("outM"),
    "holdS": (end["t"] - call["t"]) if "t" in call and "t" in end else None,
    "elapsedS": (call["t"] - started) if "t" in call and started is not None else None,
    "limitS": None if limit_s is None else float(limit_s),
  }


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"mostM": 3, "fastestMS": 3, "tiltDeg": 1, "outAtCallM": 3, "outM": 3, "holdS": 2,
         "elapsedS": 2, "limitS": 1}


def eval_drawer(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate: pure, judged on the numbers as measured
  and shown rounded, a fall first."""
  raw = {k: m.get(k) for k in ("mostM", "fastestMS", "tiltDeg", "falls", "outAtCallM", "outM",
                               "holdS", "elapsedS", "limitS")}
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items()}
  metrics["touchedDuringHold"] = list(m.get("touchedDuringHold") or ())
  for key in ("mostM", "outAtCallM", "outM", "holdS", "elapsedS"):
    if raw[key] is None:
      return False, metrics, "the drawer was never measured"
  ok, reason = lid.judged(raw["falls"], *_drawer(raw, metrics))
  return ok, metrics, reason


def _drawer(raw: dict, metrics: dict) -> tuple[bool, str]:
  """Criteria 1 to 3 on measurements all there: the verdict and its
  reason."""
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    return False, (f"measured {float(raw['holdS']):.1f} s after the call, "
                   f"not the {HOLD_S:.0f} s the job asks for")
  limit = TIME_LIMIT_S if raw["limitS"] is None else float(raw["limitS"])
  if float(raw["elapsedS"]) > limit + 1e-6:
    return False, (f"called {float(raw['elapsedS']):.0f} s after the start, past "
                   f"the {limit:.0f} s the job allows")
  most, at_call, end = (100 * float(raw[k]) for k in ("mostM", "outAtCallM", "outM"))
  want = 100 * OPEN_M
  if at_call < want:
    if most < want:
      return False, f"the drawer came out {most:.0f} cm, short of the {want:.0f} it must"
    return False, (f"the drawer came out {most:.0f} cm but was {at_call:.0f} cm out "
                   f"when you finished, short of the {want:.0f} it must")
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, (f"the drawer {at_call:.0f} cm out, but {touched} still touched it "
                   "after the call -- let go")
  if end < want:
    return False, (f"the drawer {at_call:.0f} cm out at the call, but {end:.0f} cm "
                   f"{HOLD_S:.0f} s later")
  return True, f"the drawer {end:.0f} cm out in {float(raw['elapsedS']):.0f} s, and let go"


def sample_drawer(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`): `life` is the
  attempt's world and `before` what the harness took at the call (the
  snapshot, the watch, the start, the limit); the world is measured again
  NOW, after the hold. `result`, the skill's own word, is unread."""
  watch = before.get("watch")
  name = watch.name if watch is not None else "drawer"
  return measurements(before.get("at_call"), measure(life.model, life.data, name), watch,
                      started=before.get("started"), limit_s=before.get("limitS"))
