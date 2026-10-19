"""Open the chest and set its lid down (issue #493): demo 1's verdict, a
predicate written before any robot learns the job (docs/Challenges.md §9).

The chest is #466's drop-handle chest (`activity/chest.py`), its dynamics
hidden and drawn. The job is the world's, never the robot's account of it:
code measures the lid and the robot off the true state, and the verdict goes
through `scoring.evaluate` like every other (economy/scoring.py). It is
graded and unpaid -- its row in `challenges.json` is base 0 -- and shown to
no mind: the chest is in no served world.

THE CRITERIA, written first. An attempt passes when:

  1. OPENED: at some moment the lid's true angle reached `OPEN_RAD`;
  2. LOWERED, NEVER LET FALL: at no moment while it was open -- above
     `SHUT_RAD` -- did it close faster than `LOWER_RAD_S`, through the
     attempt and the hold;
  3. SHUT: `HOLD_S` after the call, the lid is under `SHUT_RAD`. Whether
     the catch holds it again (`caught`) is said beside, and gates nothing;
  4. LET GO: nothing but the chest itself touches the lid, its bracket, its
     handle or its knob at any moment of the hold;
  5. IN TIME: the call came within `TIME_LIMIT_S` of the attempt's start
     -- the harness calls it there at the latest, and reads the world as
     it stands.

Said beside, and gating nothing (Ben, 2026-10-17: the job is the lid, and
a fall is lost time): the falls the body's own watch took (`Body.falls`),
its worst tilt (`Body.orientation`) and whether the tool its set-out gave
it is still seated (`Body.tool_powered`). ⚠ What a fall is, and which part
is the tool, is THAT BODY'S TO SAY, and nothing here names either: the
quadruped's watch is its torso past the topple line, the humanoid's its
trunk 35 deg past the posture its legs hold or its pelvis sunk
(`humanoid.getup`).

Everything else is the robot's business: how it finds the knob, which way
it pulls, whether it lies down, whether it falls on the way. `sample_lid`
reads nothing off the skill's report, and the call is wherever the attempt
ended.

What it cannot grade is Challenges.md §6's: the method, and a lid's
ELEGANCE -- a slam stopped a hair over the shut line passes, and so does a
lid lowered at 0.9 rad/s.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.rack.coupling import contact_pairs

# ---- the criteria (see the module docstring; numbered the same) -----------
#: 1: past 1.0 rad (57 deg), #469's "open": a lid there leaves 0.18 m
#: over the chest's front wall. The probe lifts each drawn lid to 62-65
#: deg (#466, its 128 set-outs). The activity's `opened` latch sits on the
#: same line; it is a sensed FLAG, and this is the verdict's own.
OPEN_RAD = 1.0
#: 2 and 3: under 5 deg the lid is shut -- its knob 24 mm over where it
#: rests -- the activity's own hysteresis line (`chest.OPEN_OFF`). ⚠ This
#: module imports nothing of the chest's: everything loads `scoring`, the
#: probe included (`tests/test_probe.py`), and the chest's module holds
#: its truth. Its parts are read by their names.
SHUT_RAD = math.radians(5.0)
#: 2: lowered is no faster than this while open, rad/s. MEASURED over the
#: 128 drawn lids (`scripts/attempts.py --drops`): let go from rest 1 deg
#: over the shut line each crosses it at 1.03-1.31 rad/s, from 2 deg over
#: at 1.45-1.85, from the activity's 57 deg at 2.9-8.3; the probe lowers
#: each at 0.49-0.57 at its fastest while open (its 128 records).
LOWER_RAD_S = 1.0
#: 3 and 4: the hold, s. MEASURED, the handle hanging: every drawn lid let
#: go at the shut line is caught inside 0.07 s, and let go from 57 deg
#: inside 2.0 -- but from the top of the sweep (`chest.SWEPT_TO`) 57 of
#: 128 never shut, propped 8.5 deg open on their own handle (its knob
#: wedged against the front wall): a drop can leave the chest open.
HOLD_S = 5.0
#: 5: the job's time, s, from the attempt's start to the call (Ben,
#: 2026-10-17): a fall costs time, and is judged by nothing else. The probe
#: takes 84-100 s (its 32 public chests); read again before #493's stage
#: 3 flies, as three times the humanoid's reference.
TIME_LIMIT_S = 300.0


# ---- the chest's parts --------------------------------------------------------

def _subtree(model, root: int) -> np.ndarray:
  """Every body under `root`, it included, as a mask over the bodies."""
  under = np.zeros(model.nbody, dtype=bool)
  for b in range(model.nbody):
    p = b
    while p > 0 and p != root:
      p = int(model.body_parentid[p])
    under[b] = p == root
  return under


def parts(model, name: str = "chest") -> tuple[np.ndarray, np.ndarray]:
  """Two masks over the geoms: the chest's own (its base, its lid and all
  that hangs off it) and the MOVING parts -- the lid, its bracket, its
  handle and its knob."""
  own = _subtree(model, model.body(name).id)[model.geom_bodyid]
  moving = _subtree(model, model.body(f"{name}_lid_body").id)[model.geom_bodyid]
  return own, moving


def tilt(quat) -> float:
  """Radians between a root's up axis and the world's, off its attitude
  (w, x, y, z): the topple death's own reading (`HubLifecycle._chassis_
  tilt`)."""
  _, x, y, _ = (float(v) for v in quat)
  return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


class Watch:
  """OUR reading of an attempt, every physics step (a step hook): the
  lid's highest, the fastest it closed while open, the robot's worst tilt
  and the falls its own watch took, and -- once `hold` -- what else touched
  the moving parts. Off the true state: the hinge's `qpos` and `qvel`, the
  contact list, the body's attitude and its fall count. The robot never
  sees it."""

  def __init__(self, model, data, body=None, name: str = "chest") -> None:
    self.model, self.data, self.body, self.name = model, data, body, name
    hinge = model.joint(f"{name}_hinge").id
    self.qadr = int(model.jnt_qposadr[hinge])
    self.dof = int(model.jnt_dofadr[hinge])
    self.own, self.moving = parts(model, name)
    self.top = -math.inf
    self.fastest = 0.0
    self.worst_tilt = None if body is None else 0.0
    self.falls0 = None if body is None else int(body.falls)
    self.holding = False
    self.touched: set[str] = set()

  def hold(self) -> None:
    """The call: from here every step reads what touches the lid's parts."""
    self.holding = True

  def falls(self) -> int | None:
    """The falls the body's own watch has taken since this began."""
    return None if self.body is None else int(self.body.falls) - self.falls0

  def __call__(self) -> None:
    a, w = float(self.data.qpos[self.qadr]), float(self.data.qvel[self.dof])
    if a > self.top:
      self.top = a
    if a > SHUT_RAD and -w > self.fastest:
      self.fastest = -w
    if self.body is not None:
      self.worst_tilt = max(self.worst_tilt, tilt(self.body.orientation()))
    if self.holding:
      g = contact_pairs(self.data)
      if g.shape[0]:
        a_, b_ = g[:, 0], g[:, 1]
        hit = (self.moving[a_] & ~self.own[b_]) | (self.moving[b_] & ~self.own[a_])
        for x, y in g[hit].tolist():
          other = y if self.moving[x] else x
          self.touched.add(self.model.geom(other).name or f"geom {other}")


def measure(model, data, name: str = "chest") -> dict:
  """One snapshot of the lid, as the grader sees it: its true angle (deg)
  and whether its catch holds it -- the magnet's pull on the hinge, which
  the chest's activity applies as physics and which falls to nothing at
  its reach (`chest.catch_torque`)."""
  hinge = model.joint(f"{name}_hinge").id
  a = float(data.qpos[model.jnt_qposadr[hinge]])
  return {"t": float(data.time), "angleDeg": math.degrees(a),
          "caught": bool(data.qfrc_applied[model.jnt_dofadr[hinge]] < 0.0)}


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 tool: str | None = None, seated: bool | None = None,
                 catch_n: float | None = None, started: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED (`eval_lid` judges the numbers and
  rounds only what it shows): the snapshot at the call, the one after the
  hold, the attempt's per-step reading, and the tool the set-out gave with
  the body's word on whether it is still seated. `started` is the
  attempt's start, sim s. `catch_n` is the chest's true catch (`chest.Lid`),
  OURS: secret on the row. Absences stay absent -- `eval_lid` turns them
  into a failure."""
  call, end = at_call or {}, settled or {}
  w = watch
  return {
    "topDeg": math.degrees(w.top) if w is not None and w.top > -math.inf else None,
    "downRadS": w.fastest if w is not None else None,
    "tiltDeg": (math.degrees(w.worst_tilt)
                if w is not None and w.worst_tilt is not None else None),
    "falls": w.falls() if w is not None else None,
    "touchedDuringHold": sorted(w.touched) if w is not None else [],
    "angleAtCallDeg": call.get("angleDeg"),
    "angleDeg": end.get("angleDeg"),
    "caught": end.get("caught"),
    "holdS": (end["t"] - call["t"]) if "t" in call and "t" in end else None,
    "elapsedS": (call["t"] - started) if "t" in call and started is not None else None,
    "tool": tool,
    "toolSeated": seated,
    "catchN": None if catch_n is None else float(catch_n),
  }


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"topDeg": 2, "downRadS": 3, "tiltDeg": 1, "angleAtCallDeg": 2, "angleDeg": 2,
         "holdS": 2, "elapsedS": 2, "catchN": 3}


def eval_lid(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate. Pure -- a dict of measurements in, a
  verdict out, judged on the numbers as measured and shown rounded. The
  reason says what the world shows and never the catch."""
  raw = {k: m.get(k) for k in ("topDeg", "downRadS", "tiltDeg", "falls", "angleAtCallDeg",
                               "angleDeg", "caught", "holdS", "elapsedS", "tool", "toolSeated",
                               "catchN")}
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items()}
  metrics["touchedDuringHold"] = list(m.get("touchedDuringHold") or ())
  # A missing measurement is not a passing one (scoring.py's rule).
  for key in ("topDeg", "downRadS", "angleDeg", "holdS", "elapsedS"):
    if raw[key] is None:
      return False, metrics, "the lid was never measured"
  top, down, end = (float(raw[k]) for k in ("topDeg", "downRadS", "angleDeg"))
  # A short hold is the HARNESS's fault, and it still fails.
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    return False, metrics, (f"measured {float(raw['holdS']):.1f} s after the call, "
                            f"not the {HOLD_S:.0f} s the job asks for")
  if float(raw["elapsedS"]) > TIME_LIMIT_S + 1e-6:
    return False, metrics, (f"called {float(raw['elapsedS']):.0f} s after the start, past "
                            f"the {TIME_LIMIT_S:.0f} s the job allows")
  if top < math.degrees(OPEN_RAD):
    if top < math.degrees(SHUT_RAD):
      return False, metrics, (f"the lid never opened: it must open past "
                              f"{math.degrees(OPEN_RAD):.0f} deg")
    return False, metrics, (f"the lid rose to {top:.0f} deg, short of the "
                            f"{math.degrees(OPEN_RAD):.0f} it must open past")
  if down > LOWER_RAD_S:
    return False, metrics, (f"opened to {top:.0f} deg, but it came down at {down:.1f} "
                            f"rad/s while open -- let fall, not lowered (no faster than "
                            f"{LOWER_RAD_S:.1f})")
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, metrics, (f"opened to {top:.0f} deg and lowered, but {touched} still "
                            "touched the lid or its handle after the call -- let go")
  if end >= math.degrees(SHUT_RAD):
    return False, metrics, (f"opened to {top:.0f} deg and lowered, but the lid was "
                            f"{end:.0f} deg open {HOLD_S:.0f} s after the call")
  return True, metrics, (f"opened to {top:.0f} deg and set down in "
                         f"{float(raw['elapsedS']):.0f} s: lowered at most {down:.2f} rad/s "
                         f"while open, shut {'and caught ' if raw['caught'] else ''}"
                         f"{HOLD_S:.0f} s after the call, let go")


def sample_lid(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`): `life` is the
  attempt's world (its `model`, `data` and `body`) and `before` what the
  harness took at the call -- the snapshot (`at_call`), the attempt's
  `Watch`, its start (`started`), the tool its set-out gave (`tool`) and
  the chest's true catch (`catchN`, ours) -- and the world is measured
  again NOW, after the hold.
  `result`, the skill's own word, is deliberately unread."""
  watch = before.get("watch")
  tool = before.get("tool")
  seated = bool(life.body.tool_powered(tool)) if tool else None
  name = watch.name if watch is not None else "chest"
  return measurements(before.get("at_call"), measure(life.model, life.data, name), watch,
                      tool=tool, seated=seated, catch_n=before.get("catchN"),
                      started=before.get("started"))
