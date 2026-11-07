"""Set a place at a table (issue #560, its first task, easy mode): a plate, a
fork and a cup laid neatly at a place card, a predicate written before any
procedure flies it (docs/Challenges.md).

The dishware and the card are `activity/dishware.py`'s, each piece's weight
and its sliding friction hidden and drawn. Code measures the pieces and the
robot off the true state, and the verdict goes through `scoring.evaluate`
like every other. Graded and unpaid -- its row in `challenges.json` is
base 0 -- and shown to no mind.

Where each piece belongs is in the PLACE's frame: from the foot of the
card's face, `a` toward the diner and `b` to the diner's left (the diner
faces the card). THE CRITERIA, written first. An attempt passes when:

  1. LAID: `HOLD_S` after the call each piece rests at its spot (`SPOTS`)
     -- its middle within `PLACE_M` of it across the table, its bottom within
     `RESTING_M` of the table's top (on the table, never on another piece),
     lying or standing as it is used within `UPRIGHT_RAD` (`resting_tilt`:
     the plate and the fork flat, the cup upright) -- and the fork along the
     place, its head toward the card, within `FORK_RAD`;
  2. SET DOWN, NEVER DROPPED: at no moment, through the attempt and the
     hold, did a piece nothing of the robot touched come down faster than
     `DROP_M_S` -- let go over the table, or knocked over;
  3. LET GO: nothing of the robot touches any piece at any moment of the
     hold;
  4. IN TIME: the call came within the task's time limit of the attempt's
     start (`limitS`; `lid.TIME_LIMIT_S` where none is given);
  5. NEVER FELL: the body's own watch took no fall from the attempt's start
     to the end of the hold, the lid's criterion 6 (`lid.judged`).

Said beside, and gating nothing: whether each piece was ever lifted, how
far each was from its spot at the call, the fork's turn and the body's
worst tilt. How the pieces get there is the robot's business -- slid there
passes as well as carried.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity.dishware import PIECES
from pluggybot.challenge import lid
from pluggybot.contacts import contact_pairs

#: Where each piece belongs, (a, b) m in the place's frame: the plate's
#: middle on the place's line, the fork's handle's middle to the plate's
#: left, the cup's middle to its right and further in -- a place as it is
#: laid, sized to what the hands reach from the place (SimNotes, "Setting a
#: table").
SPOTS = {"plate": (0.19, 0.0), "fork": (0.19, 0.16), "cup": (0.07, -0.24)}
#: 1: within this of its spot, m; resting within this of the table's top,
#: m; flat (the plate, the fork) or upright (the cup) within this, rad; the
#: fork along the place within this, rad.
PLACE_M, RESTING_M = 0.025, 0.004
UPRIGHT_RAD, FORK_RAD = math.radians(10.0), math.radians(15.0)
#: 2: a free piece coming down no faster than this, m/s: let go 1.3 cm over
#: the table, a piece lands at 0.5 -- the cube's line (`place.DROP_M_S`).
DROP_M_S = 0.5
#: The hold and the default time, the lid's.
HOLD_S, TIME_LIMIT_S = lid.HOLD_S, lid.TIME_LIMIT_S
#: The card's body (`dishware.attach_card`): the one place's, at whichever
#: table a job names (the restaurant's table 0 today, `home.restaurant`).
CARD = "place_card"


def place_frame(model, data, card: str = CARD) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """The place's frame off the card's true pose: its origin (the foot of
  its face, on the table), the way toward the diner, the diner's left, and
  the table's top, m."""
  b = model.body(card).id
  rot = data.xmat[b].reshape(3, 3)
  toward = rot[:, 1].copy()                # the card's face looks along its y
  toward[2] = 0.0
  toward /= np.linalg.norm(toward)
  left = np.array([toward[1], -toward[0], 0.0])
  origin = data.xpos[b].copy()
  return origin, toward, left, float(origin[2])


def resting_tilt(quat) -> float:
  """Radians a piece's own up is off the world's, either way up (a plate
  turned over is no plate laid, and reads past any line)."""
  w, x, y, z = (float(v) for v in quat)
  up_z = 1.0 - 2.0 * (x * x + y * y)
  return math.acos(min(1.0, max(-1.0, up_z)))


def _bottom(model, data, name: str) -> float:
  """The lowest point of a piece's geoms that touch, m: its bottom."""
  lo = math.inf
  b = model.body(name).id
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != b or model.geom_contype[g] == 0:
      continue
    rot = data.geom_xmat[g].reshape(3, 3)
    size = model.geom_size[g]
    if model.geom_type[g] == 5:          # a cylinder: its radius and half height
      half = abs(rot[2, 2]) * size[1] + math.hypot(rot[2, 0], rot[2, 1]) * size[0]
    else:                                 # a box
      half = float(np.abs(rot[2]) @ size)
    lo = min(lo, float(data.geom_xpos[g][2]) - half)
  return lo


def measure(model, data, table: str | None = None) -> dict:
  """One snapshot of the pieces against the place, as the grader sees it:
  each piece's offset from its spot, m, its bottom over the table's top, m,
  its tilt, deg, and the fork's turn off the place's line, deg. `table`, a
  job's target, names where the place is laid; its card is `CARD`."""
  origin, toward, left, top = place_frame(model, data, CARD)
  out: dict = {"t": float(data.time)}
  for p in PIECES:
    b = model.body(p).id
    rel = data.xpos[b] - origin
    a, s = float(rel @ toward), float(rel @ left)
    spot = SPOTS[p]
    out[p] = {"offM": math.hypot(a - spot[0], s - spot[1]),
              "aboveM": _bottom(model, data, p) - top,
              "tiltDeg": math.degrees(resting_tilt(data.xquat[b]))}
  # the fork's head is along its own x: laid, it points at the card
  head = data.xmat[model.body("fork").id].reshape(3, 3)[:, 0]
  way = -toward
  out["fork"]["turnDeg"] = math.degrees(math.atan2(float(way[0] * head[1] - way[1] * head[0]),
                                                   float(way[:2] @ head[:2])))
  return out


class Watch:
  """OUR reading of an attempt, every physics step (a step hook): each
  piece's fastest fall while nothing of the robot touched it and how high
  it was ever lifted, the robot's worst tilt and its falls, and -- once
  `hold` -- what of the robot touched a piece. Off the true state. The robot
  never sees it."""

  def __init__(self, model, data, body=None, table: str | None = None) -> None:
    self.model, self.data, self.body = model, data, body
    self.ids = {p: int(model.body(p).id) for p in PIECES}
    self.dofs = {p: int(model.jnt_dofadr[model.joint(f"{p}_free").id]) for p in PIECES}
    self.piece_of = np.full(model.ngeom, -1, dtype=int)
    for k, p in enumerate(PIECES):
      self.piece_of[model.geom_bodyid == self.ids[p]] = k
    self.mine = None if body is None else np.zeros(model.ngeom, dtype=bool)
    if body is not None:
      self.mine[list(body.geom_ids)] = True
    self.z0: dict = {p: None for p in PIECES}
    self.highest = {p: 0.0 for p in PIECES}
    self.fastest_free = {p: 0.0 for p in PIECES}
    self.worst_tilt = None if body is None else 0.0
    self.falls0 = None if body is None else int(body.falls)
    self.holding = False
    self.touched: set[str] = set()

  def hold(self) -> None:
    self.holding = True

  def falls(self) -> int | None:
    return None if self.body is None else int(self.body.falls) - self.falls0

  def kept_state(self) -> dict:
    """What a restart carries: the attempt's readings and its falls so far,
    never the hold's."""
    return {"z0": dict(self.z0), "highest": dict(self.highest),
            "fastestFree": dict(self.fastest_free), "worstTilt": self.worst_tilt,
            "falls": self.falls()}

  def restore_kept(self, state: dict) -> None:
    self.z0.update(state.get("z0") or {})
    self.highest.update(state.get("highest") or {})
    self.fastest_free.update(state.get("fastestFree") or {})
    self.worst_tilt = state.get("worstTilt", self.worst_tilt)
    lid.kept_falls(self, state)

  def __call__(self) -> None:
    held = np.zeros(len(PIECES), dtype=bool)
    if self.mine is not None:
      g = contact_pairs(self.data)
      if g.shape[0]:
        a_, b_ = g[:, 0], g[:, 1]
        for x, y in g[(self.piece_of[a_] >= 0) & self.mine[b_]].tolist():
          held[self.piece_of[x]] = True
        for x, y in g[(self.piece_of[b_] >= 0) & self.mine[a_]].tolist():
          held[self.piece_of[y]] = True
        if self.holding:
          for x, y in g[((self.piece_of[a_] >= 0) & self.mine[b_])
                        | ((self.piece_of[b_] >= 0) & self.mine[a_])].tolist():
            mine = y if self.mine[y] else x
            self.touched.add(self.model.geom(mine).name or f"geom {mine}")
    for k, p in enumerate(PIECES):
      z = float(self.data.xpos[self.ids[p]][2])
      if self.z0[p] is None:
        self.z0[p] = z
      self.highest[p] = max(self.highest[p], z - self.z0[p])
      if not held[k]:
        # the free joint's linear velocity is the world's, at the body's origin
        self.fastest_free[p] = max(self.fastest_free[p], -float(self.data.qvel[self.dofs[p] + 2]))
    if self.body is not None:
      self.worst_tilt = max(self.worst_tilt, lid.tilt(self.body.orientation()))


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 started: float | None = None, limit_s: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED. Absences stay absent."""
  call, end, w = at_call or {}, settled or {}, watch
  out = {
    "robotTiltDeg": (math.degrees(w.worst_tilt)
                     if w is not None and w.worst_tilt is not None else None),
    "falls": w.falls() if w is not None else None,
    "touchedDuringHold": sorted(w.touched) if w is not None else [],
    "holdS": (end["t"] - call["t"]) if "t" in call and "t" in end else None,
    "elapsedS": (call["t"] - started) if "t" in call and started is not None else None,
    "limitS": None if limit_s is None else float(limit_s),
  }
  for p in PIECES:
    c, e = call.get(p) or {}, end.get(p) or {}
    out[p] = {"liftedM": w.highest[p] if w is not None else None,
              "fastestFreeMS": w.fastest_free[p] if w is not None else None,
              "offAtCallM": c.get("offM"), "offM": e.get("offM"), "aboveM": e.get("aboveM"),
              "tiltDeg": e.get("tiltDeg")}
  out["fork"]["turnDeg"] = (end.get("fork") or {}).get("turnDeg")
  return out


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"robotTiltDeg": 1, "holdS": 2, "elapsedS": 2, "limitS": 1}
PIECE_SHOWN = {"liftedM": 3, "fastestFreeMS": 3, "offAtCallM": 3, "offM": 3, "aboveM": 4,
               "tiltDeg": 1, "turnDeg": 1}


def _shown(raw: dict) -> dict:
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items() if k not in PIECES}
  for p in PIECES:
    metrics[p] = {k: (round(float(v), PIECE_SHOWN[k]) if v is not None else None)
                  for k, v in raw[p].items()}
  metrics["touchedDuringHold"] = list(raw.get("touchedDuringHold") or ())
  return metrics


def eval_table(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate: pure, judged on the numbers as measured
  and shown rounded, a fall first."""
  raw = {k: m.get(k) for k in ("robotTiltDeg", "falls", "touchedDuringHold", "holdS",
                               "elapsedS", "limitS")}
  for p in PIECES:
    raw[p] = dict(m.get(p) or {})
  metrics = _shown(raw)
  missing = [k for k in ("holdS", "elapsedS") if raw[k] is None] + [
    p for p in PIECES for k in ("fastestFreeMS", "offM", "aboveM", "tiltDeg")
    if raw[p].get(k) is None]
  if raw["fork"].get("turnDeg") is None or missing:
    return False, metrics, "the place was never measured"
  ok, reason = lid.judged(raw["falls"], *_laid(raw, metrics))
  return ok, metrics, reason


def _laid(raw: dict, metrics: dict) -> tuple[bool, str]:
  """Criteria 1 to 4 on measurements all there: the verdict and its
  reason, the first piece at fault named."""
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    return False, (f"measured {float(raw['holdS']):.1f} s after the call, "
                   f"not the {HOLD_S:.0f} s the job asks for")
  limit = TIME_LIMIT_S if raw["limitS"] is None else float(raw["limitS"])
  if float(raw["elapsedS"]) > limit + 1e-6:
    return False, (f"called {float(raw['elapsedS']):.0f} s after the start, past "
                   f"the {limit:.0f} s the job allows")
  for p in PIECES:
    fast = float(raw[p]["fastestFreeMS"])
    if fast > DROP_M_S:
      return False, (f"the {p} came down at {fast:.1f} m/s with nothing holding it -- "
                     f"dropped or knocked over, not set down (no faster than {DROP_M_S:.1f})")
  for p in PIECES:
    off, above, tilt = 100 * float(raw[p]["offM"]), 100 * float(raw[p]["aboveM"]), float(raw[p]["tiltDeg"])
    if off > 100 * PLACE_M or abs(above) > 100 * RESTING_M:
      where = f"{off:.0f} cm from its place" + (f", {above:.0f} cm over the table" if above > 1 else "")
      return False, (f"the {p} ended {where}: it must rest on the table within "
                     f"{100 * PLACE_M:.1f} cm of where it goes")
    if tilt > math.degrees(UPRIGHT_RAD):
      lies = "upright" if p == "cup" else "flat"
      return False, f"the {p} rests at its place tipped {tilt:.0f} deg: it must stand {lies}"
  turn = float(raw["fork"]["turnDeg"])
  if abs(turn) > math.degrees(FORK_RAD):
    return False, (f"the fork lies {abs(turn):.0f} deg off the place's line: it must lie along "
                   f"it, its head toward the card, within {math.degrees(FORK_RAD):.0f} deg")
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, f"the place laid, but {touched} still touched it after the call -- let go"
  offs = ", ".join(f"the {p} {100 * float(raw[p]['offM']):.1f} cm" for p in PIECES)
  return True, (f"the place laid in {float(raw['elapsedS']):.0f} s -- {offs} from where each "
                f"goes -- and let go")


def sample_table(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`), as the cube's.
  `result`, the skill's own word, is unread."""
  return measurements(before.get("at_call"), measure(life.model, life.data), before.get("watch"),
                      started=before.get("started"), limit_s=before.get("limitS"))
