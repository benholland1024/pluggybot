"""Pick the cube up and place it on the mat (issue #525): one of the hands'
capability checks (docs/Hands.md), a predicate written before any procedure
flies it (docs/Challenges.md).

The cube and the mat are `activity/cube.py`'s, the cube's weight and its
sliding friction hidden and drawn. Code measures the cube and the robot off
the true state, and the verdict goes through `scoring.evaluate` like every
other. Graded and unpaid -- its row in `challenges.json` is base 0 -- and
shown to no mind.

THE CRITERIA, written first. An attempt passes when:

  1. PLACED: `HOLD_S` after the call the cube rests on the mat: its middle
     within `PLACE_M` of the mat's across the counter, its bottom within
     `RESTING_M` of the mat's top, and tilted no more than `UPRIGHT_RAD`;
  2. SET DOWN, NEVER DROPPED: at no moment, through the attempt and the
     hold, did the cube come down faster than `DROP_M_S`;
  3. LET GO: nothing of the robot touches the cube at any moment of the
     hold;
  4. IN TIME: the call came within the task's time limit of the attempt's
     start (`limitS`; `lid.TIME_LIMIT_S` where none is given).

Said beside, and gating nothing: whether it was ever lifted, the furthest
it was from the mat at the call, the falls the body's own watch took and
its worst tilt. How it picks the cube up and carries it is the robot's
business -- sliding it there passes as well as carrying it.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.challenge import lid
from pluggybot.rack.coupling import contact_pairs

#: 1: within this of the mat's middle, m (the mat's half side is 0.04: the
#: cube's whole bottom on it), resting within this of its top, m, and
#: upright within this, rad.
PLACE_M, RESTING_M, UPRIGHT_RAD = 0.025, 0.004, math.radians(10.0)
#: 2: coming down no faster than this, m/s: let go 1.3 cm over the mat, a
#: cube lands at 0.5.
DROP_M_S = 0.5
#: The hold and the default time, the lid's.
HOLD_S, TIME_LIMIT_S = lid.HOLD_S, lid.TIME_LIMIT_S


class Watch:
  """OUR reading of an attempt, every physics step (a step hook): the
  fastest the cube came down, how high it ever was, the robot's worst tilt
  and its falls, and -- once `hold` -- what of the robot touched the cube.
  Off the true state. The robot never sees it."""

  def __init__(self, model, data, body=None, name: str = "cube") -> None:
    self.model, self.data, self.body, self.name = model, data, body, name
    self.cube = int(model.body(name).id)
    self.geom = int(model.geom(name).id)
    free = model.joint(f"{name}_free").id
    self.dof = int(model.jnt_dofadr[free])
    self.z0 = None
    self.highest = 0.0
    self.fastest_down = 0.0
    self.mine = None if body is None else np.zeros(model.ngeom, dtype=bool)
    if body is not None:
      self.mine[list(body.geom_ids)] = True
    self.worst_tilt = None if body is None else 0.0
    self.falls0 = None if body is None else int(body.falls)
    self.holding = False
    self.touched: set[str] = set()

  def hold(self) -> None:
    self.holding = True

  def falls(self) -> int | None:
    return None if self.body is None else int(self.body.falls) - self.falls0

  def __call__(self) -> None:
    z = float(self.data.xpos[self.cube][2])
    if self.z0 is None:
      self.z0 = z
    self.highest = max(self.highest, z - self.z0)
    # the free joint's linear velocity is the world's, at the body's origin
    self.fastest_down = max(self.fastest_down, -float(self.data.qvel[self.dof + 2]))
    if self.body is not None:
      self.worst_tilt = max(self.worst_tilt, lid.tilt(self.body.orientation()))
    if self.holding and self.mine is not None:
      g = contact_pairs(self.data)
      if g.shape[0]:
        a_, b_ = g[:, 0], g[:, 1]
        hit = ((a_ == self.geom) & self.mine[b_]) | ((b_ == self.geom) & self.mine[a_])
        for x, y in g[hit].tolist():
          other = y if x == self.geom else x
          self.touched.add(self.model.geom(other).name or f"geom {other}")


def measure(model, data, name: str = "cube") -> dict:
  """One snapshot of the cube against the mat, as the grader sees it: how
  far its middle is from the mat's across the counter, m, how high its
  bottom stands over the mat's top, m, and its tilt, deg."""
  cube, mat = model.body(name).id, model.body(f"{name}_mat").id
  half = float(model.geom_size[model.geom(name).id][2])
  top = float(data.xpos[mat][2]) + float(model.geom_size[model.geom(f"{name}_mat").id][2])
  p, m = data.xpos[cube], data.xpos[mat]
  return {"t": float(data.time), "offM": float(math.hypot(p[0] - m[0], p[1] - m[1])),
          "aboveM": float(p[2]) - half - top, "tiltDeg": math.degrees(lid.tilt(data.xquat[cube]))}


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 started: float | None = None, limit_s: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED. Absences stay absent."""
  call, end, w = at_call or {}, settled or {}, watch
  return {
    "liftedM": w.highest if w is not None else None,
    "fastestDownMS": w.fastest_down if w is not None else None,
    "robotTiltDeg": (math.degrees(w.worst_tilt)
                     if w is not None and w.worst_tilt is not None else None),
    "falls": w.falls() if w is not None else None,
    "touchedDuringHold": sorted(w.touched) if w is not None else [],
    "offAtCallM": call.get("offM"),
    "offM": end.get("offM"),
    "aboveM": end.get("aboveM"),
    "tiltDeg": end.get("tiltDeg"),
    "holdS": (end["t"] - call["t"]) if "t" in call and "t" in end else None,
    "elapsedS": (call["t"] - started) if "t" in call and started is not None else None,
    "limitS": None if limit_s is None else float(limit_s),
  }


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"liftedM": 3, "fastestDownMS": 3, "robotTiltDeg": 1, "offAtCallM": 3, "offM": 3,
         "aboveM": 4, "tiltDeg": 1, "holdS": 2, "elapsedS": 2, "limitS": 1}


def eval_place(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate: pure, judged on the numbers as measured
  and shown rounded."""
  raw = {k: m.get(k) for k in ("liftedM", "fastestDownMS", "robotTiltDeg", "falls", "offAtCallM",
                               "offM", "aboveM", "tiltDeg", "holdS", "elapsedS", "limitS")}
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items()}
  metrics["touchedDuringHold"] = list(m.get("touchedDuringHold") or ())
  for key in ("fastestDownMS", "offM", "aboveM", "tiltDeg", "holdS", "elapsedS"):
    if raw[key] is None:
      return False, metrics, "the cube was never measured"
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    return False, metrics, (f"measured {float(raw['holdS']):.1f} s after the call, "
                            f"not the {HOLD_S:.0f} s the job asks for")
  limit = TIME_LIMIT_S if raw["limitS"] is None else float(raw["limitS"])
  if float(raw["elapsedS"]) > limit + 1e-6:
    return False, metrics, (f"called {float(raw['elapsedS']):.0f} s after the start, past "
                            f"the {limit:.0f} s the job allows")
  off, above, tilt = 100 * float(raw["offM"]), 100 * float(raw["aboveM"]), float(raw["tiltDeg"])
  if float(raw["fastestDownMS"]) > DROP_M_S:
    return False, metrics, (f"the cube came down at {float(raw['fastestDownMS']):.1f} m/s -- "
                            f"dropped, not set down (no faster than {DROP_M_S:.1f})")
  if off > 100 * PLACE_M or abs(above) > 100 * RESTING_M:
    where = f"{off:.0f} cm from the mat's middle" + (f", {above:.0f} cm over it" if above > 1 else "")
    return False, metrics, f"the cube ended {where}: it must rest on the mat, within {100 * PLACE_M:.1f} cm of its middle"
  if tilt > math.degrees(UPRIGHT_RAD):
    return False, metrics, f"the cube rests on the mat tipped {tilt:.0f} deg"
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, metrics, f"the cube on the mat, but {touched} still touched it after the call -- let go"
  return True, metrics, (f"the cube set down on the mat {off:.1f} cm from its middle in "
                         f"{float(raw['elapsedS']):.0f} s, and let go")


def sample_place(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`), as the drawer's.
  `result`, the skill's own word, is unread."""
  watch = before.get("watch")
  name = watch.name if watch is not None else "cube"
  return measurements(before.get("at_call"), measure(life.model, life.data, name), watch,
                      started=before.get("started"), limit_s=before.get("limitS"))
