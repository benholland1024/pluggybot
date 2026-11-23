"""Stock a shelf (issue #736, #560's first store task, easy mode): a candle
and a desk frame taken off a cart at the shop's back shelf and each set
down over its label, a predicate written before any procedure flies it
(docs/Challenges.md §13).

The products, the cart and the labels are `activity/stock.py`'s, each
product's weight and its sliding friction hidden and drawn; which label
each goes over is the job's word (`slots`, {product: label id}). Code
measures the products and the robot off the true state, and the verdict
goes through `scoring.evaluate` like every other. Graded and unpaid -- its
row in `challenges.json` is base 0 -- and shown to no mind.

Where each product belongs is in the SHELF's frame, read off its boards'
true poses: along the shelf (to the right as one faces it), out of its
front edge toward the aisle, and up; its slot is its label's middle along
the board the label is on. THE CRITERIA, written first (#736's calls 1-5,
Ben's of 2026-11-22). An attempt passes when:

  1. STOCKED: `HOLD_S` after the call each product rests on its label's
     board -- its bottom within `RESTING_M` of the board's top -- its
     middle within `ALONG_M` of its label's along the board, its front no
     more than `FRONT_IN_M` behind the board's front edge and not over it,
     upright within `UPRIGHT_RAD` (the frame as its strut stands it), and
     the frame's face toward the aisle within `FACING_RAD`;
  2. SET DOWN, NEVER DROPPED: at no moment, through the attempt and the
     hold, did a product nothing of the robot touched come down faster
     than `DROP_M_S` -- let go over the board, or knocked over (the
     table's line, `table.Watch`);
  3. LET GO: nothing of the robot touches a product at any moment of the
     hold;
  4. IN TIME: the call came within the task's time limit of the attempt's
     start (`limitS`; `lid.TIME_LIMIT_S` where none is given);
  5. NEVER FELL: the body's own watch took no fall from the attempt's start
     to the end of the hold, the lid's criterion 6 (`lid.judged`).

Said beside, gating nothing: how high each product was ever lifted, how
far each was from its label at the call, each one's tilt and the frame's
turn, and the body's worst tilt. How a product gets there is the robot's
business. What it cannot grade: whether the shelf looks right, and
anything but its products (stock already there, and "nothing else moved",
are facing's, #560's next task).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity.stock import BOARD_OF, LABEL_IDS, PRODUCTS
from pluggybot.challenge import lid, table

#: 1: its middle within this of its label's along the board, m; its bottom
#: within this of the board's top, m; its front no further behind the
#: board's edge than this, m (and never over it); upright within this, rad;
#: the frame's face within this of looking out of the shelf, rad.
ALONG_M, RESTING_M, FRONT_IN_M = 0.04, 0.004, 0.10
UPRIGHT_RAD, FACING_RAD = math.radians(10.0), math.radians(20.0)
#: 2: a free product coming down no faster than this, m/s: the table's and
#: the cube's line (`table.DROP_M_S`).
DROP_M_S = table.DROP_M_S
#: The hold and the default time, the lid's.
HOLD_S, TIME_LIMIT_S = lid.HOLD_S, lid.TIME_LIMIT_S
#: The shelf a job stocks: the shop's, along its back wall (`home.world.
#: SHELVES`), its boards geoms `<shelf>_board_<i>`.
SHELF = "shop_shelf"
#: The boards a label may be on (#736's call 1: the second and third).
BOARDS = tuple(sorted(set(BOARD_OF.values())))


def shelf_frame(model, data, shelf: str = SHELF) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
  """The shelf's frame off its boards' true poses: its origin (the middle of
  its boards' front edge, on the floor), the way out of its front toward
  the aisle, the way to the right as one faces it, and each board's top
  over the floor, m ({board index: z})."""
  g = model.geom(f"{shelf}_board_{BOARDS[0]}").id
  rot = data.geom_xmat[g].reshape(3, 3)
  out = rot[:, 1].copy()                   # a board's y looks out of the shelf's front
  out[2] = 0.0
  out /= np.linalg.norm(out)
  right = np.array([-out[1], out[0], 0.0])  # facing in (-out), the right hand's way
  origin = data.geom_xpos[g] + float(model.geom_size[g][1]) * out
  origin[2] = 0.0
  tops = {}
  for i in range(8):
    try:
      b = model.geom(f"{shelf}_board_{i}").id
    except KeyError:
      break
    tops[i] = float(data.geom_xpos[b][2] + model.geom_size[b][2])
  return origin, out, right, tops


def _extent(model, data, g: int, n) -> float:
  """How far a geom reaches from its centre along the unit `n`, m: a box's
  or a cylinder's."""
  rot = data.geom_xmat[g].reshape(3, 3)
  size = model.geom_size[g]
  if model.geom_type[g] == 5:              # a cylinder: its radius and half height
    a = min(1.0, abs(float(rot[:, 2] @ n)))
    return a * float(size[1]) + math.sqrt(1.0 - a * a) * float(size[0])
  return float(np.abs(rot.T @ n) @ size[:3])


def _solid(model, body: int) -> list[int]:
  """A product's geoms that touch: its films are tags."""
  return [g for g in range(int(model.body_geomadr[body]),
                           int(model.body_geomadr[body] + model.body_geomnum[body]))
          if model.geom_contype[g] != 0]


def measure(model, data, shelf: str = SHELF) -> dict:
  """One snapshot of the products against the shelf, as the grader sees
  it: where each label's middle is along the shelf and its board's top, m,
  and for each product its middle along the shelf, how far its front is
  behind the boards' front edge (negative, over it), its bottom over the
  floor, m, its tilt, deg, and the frame's turn off facing the aisle, deg.
  Which label each product goes over is not the world's: `measurements`
  pairs them."""
  origin, out, right, tops = shelf_frame(model, data, shelf)
  up = np.array([0.0, 0.0, 1.0])
  snap: dict = {"t": float(data.time), "labels": {}}
  for i in LABEL_IDS:
    try:
      g = model.geom(f"label_{i}_tag").id
    except KeyError:
      continue
    snap["labels"][str(i)] = {"s": float((data.geom_xpos[g] - origin) @ right),
                              "topZ": tops.get(BOARD_OF[i])}
  for p in PRODUCTS:
    b = model.body(p).id
    geoms = _solid(model, b)
    if not geoms:
      continue
    snap[p] = {"s": float((data.xpos[b] - origin) @ right),
               "backM": -max(float((data.geom_xpos[g] - origin) @ out) + _extent(model, data, g, out)
                             for g in geoms),
               "bottomZ": min(float(data.geom_xpos[g][2]) - _extent(model, data, g, up)
                              for g in geoms),
               "tiltDeg": math.degrees(table.resting_tilt(data.xquat[b]))}
  if "frame" in snap:
    # the frame's face looks along its body's -x
    face = -data.xmat[model.body("frame").id].reshape(3, 3)[:, 0]
    snap["frame"]["turnDeg"] = math.degrees(math.atan2(float(out[0] * face[1] - out[1] * face[0]),
                                                       float(out[:2] @ face[:2])))
  return snap


class Watch(table.Watch):
  """OUR reading of an attempt, every physics step: the table's
  (`table.Watch`) over the products -- each one's fastest fall while
  nothing of the robot touched it and how high it was ever lifted, the
  robot's worst tilt and its falls, and once `hold` what of the robot
  touched a product. Off the true state; the robot never sees it."""

  pieces = PRODUCTS


def measurements(at_call: dict | None, settled: dict | None, watch: Watch | None,
                 slots: dict | None = None, started: float | None = None,
                 limit_s: float | None = None) -> dict:
  """The evaluator's input, UNROUNDED: each product against its label
  (`slots`, {product: label id}). Absences stay absent."""
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
  for p in PRODUCTS:
    c, e = call.get(p) or {}, end.get(p) or {}
    slot = (slots or {}).get(p)
    lc = (call.get("labels") or {}).get(str(slot)) if slot is not None else None
    le = (end.get("labels") or {}).get(str(slot)) if slot is not None else None
    out[p] = {"slot": None if slot is None else int(slot),
              "liftedM": w.highest[p] if w is not None else None,
              "fastestFreeMS": w.fastest_free[p] if w is not None else None,
              "alongAtCallM": c["s"] - lc["s"] if lc and "s" in c else None,
              "alongM": e["s"] - le["s"] if le and "s" in e else None,
              "aboveM": (e["bottomZ"] - le["topZ"]
                         if le and le.get("topZ") is not None and "bottomZ" in e else None),
              "backM": e.get("backM"), "tiltDeg": e.get("tiltDeg")}
  out["frame"]["turnDeg"] = (end.get("frame") or {}).get("turnDeg")
  return out


#: What a verdict shows of each number, decimal places: judged unrounded.
SHOWN = {"robotTiltDeg": 1, "holdS": 2, "elapsedS": 2, "limitS": 1}
PRODUCT_SHOWN = {"liftedM": 3, "fastestFreeMS": 3, "alongAtCallM": 3, "alongM": 3, "aboveM": 4,
                 "backM": 3, "tiltDeg": 1, "turnDeg": 1}


def _shown(raw: dict) -> dict:
  metrics = {k: (round(float(v), SHOWN[k]) if k in SHOWN and v is not None else v)
             for k, v in raw.items() if k not in PRODUCTS}
  for p in PRODUCTS:
    metrics[p] = {k: (round(float(v), PRODUCT_SHOWN[k]) if k in PRODUCT_SHOWN and v is not None
                      else v)
                  for k, v in raw[p].items()}
  metrics["touchedDuringHold"] = list(raw.get("touchedDuringHold") or ())
  return metrics


def eval_shelf(m: dict) -> tuple[bool, dict, str]:
  """The pre-declared predicate: pure, judged on the numbers as measured
  and shown rounded, a fall first."""
  raw = {k: m.get(k) for k in ("robotTiltDeg", "falls", "touchedDuringHold", "holdS",
                               "elapsedS", "limitS")}
  for p in PRODUCTS:
    raw[p] = dict(m.get(p) or {})
  metrics = _shown(raw)
  missing = [k for k in ("holdS", "elapsedS") if raw[k] is None] + [
    p for p in PRODUCTS for k in ("slot", "fastestFreeMS", "alongM", "aboveM", "backM", "tiltDeg")
    if raw[p].get(k) is None]
  if raw["frame"].get("turnDeg") is None or missing:
    return False, metrics, "the shelf was never measured"
  ok, reason = lid.judged(raw["falls"], *_stocked(raw, metrics))
  return ok, metrics, reason


_listed = table._listed


def _stocked(raw: dict, metrics: dict) -> tuple[bool, str]:
  """Criteria 1 to 4 on measurements all there: the verdict and its
  reason. EVERY PRODUCT AT FAULT IS NAMED, ONCE, BY ITS FIRST FAULT IN THE
  CRITERIA'S ORDER -- its drop, else where it ended, its front, how it
  stands or the frame's turn -- each rule said once over the products it
  names, as the table's reason is (#687). What still touched a product is
  said of a shelf otherwise stocked."""
  if float(raw["holdS"]) + 1e-6 < HOLD_S:
    held, asked = lid.against(float(raw["holdS"]), HOLD_S, 1, 0)
    return False, f"measured {held} s after the call, not the {asked} s the job asks for"
  said = []
  limit = TIME_LIMIT_S if raw["limitS"] is None else float(raw["limitS"])
  if float(raw["elapsedS"]) > limit + 1e-6:
    took, allows = lid.against(float(raw["elapsedS"]), limit)
    said.append(f"called {took} s after the start, past the {allows} s the job allows")
  # ...each reading against its line as `lid.against` prints it (#737), the
  # lines said once over both products exact at the places they are said at
  dropped = [p for p in PRODUCTS if float(raw[p]["fastestFreeMS"]) > DROP_M_S]
  if dropped:
    said.append(_listed([f"the {p} came down at "
                         f"{lid.against(float(raw[p]['fastestFreeMS']), DROP_M_S, 1)[0]} m/s"
                         for p in dropped])
                + (" with" if len(dropped) == 1 else ", each with")
                + " nothing holding it -- dropped or knocked over, not set down (no faster "
                f"than {DROP_M_S:.1f})")
  away, fronts, askew = [], [], []
  for p in (p for p in PRODUCTS if p not in dropped):
    q = raw[p]
    along, above, back = float(q["alongM"]), float(q["aboveM"]), float(q["backM"])
    tilt, label = float(q["tiltDeg"]), int(q["slot"])
    if abs(along) > ALONG_M or abs(above) > RESTING_M:
      # ...a passing distance beside a failing height too: never "4 cm" for
      # 3.96 beside "within 4 cm" (review round 2, #737's class)
      cm = lid.against(100 * abs(along), 100 * ALONG_M)[0]
      words = f"the {p} ended {cm} cm {'right' if along > 0 else 'left'} of label {label}"
      if abs(above) > RESTING_M:
        words += (f", {lid.against(100 * abs(above), 100 * RESTING_M, 1)[0]} cm "
                  f"{'over' if above > 0 else 'under'} its board's top")
      away.append(words)
    elif back > FRONT_IN_M:
      fronts.append(f"the {p}'s front ended {lid.against(100 * back, 100 * FRONT_IN_M)[0]} cm "
                    "behind the board's edge")
    elif back < 0.0:
      fronts.append(f"the {p} juts {lid.against(-100 * back, 0.0, 1)[0]} cm out over the "
                    "board's edge")
    elif tilt > math.degrees(UPRIGHT_RAD):
      tipped, within = lid.against(tilt, math.degrees(UPRIGHT_RAD))
      askew.append(f"the {p} stands tipped {tipped} deg: it must stand upright, within "
                   f"{within} deg")
    elif p == "frame" and abs(float(q["turnDeg"])) > math.degrees(FACING_RAD):
      faces, within = lid.against(abs(float(q["turnDeg"])), math.degrees(FACING_RAD))
      askew.append(f"the frame faces {faces} deg off the aisle: its face must look out of the "
                   f"shelf within {within} deg")
  if away:
    said.append(_listed(away) + (": it" if len(away) == 1 else ": each")
                + f" must rest on its label's board, its middle within {100 * ALONG_M:.0f} cm "
                "of the label's along it")
  if fronts:
    said.append(_listed(fronts) + (": its" if len(fronts) == 1 else ": each one's")
                + f" front must be within {100 * FRONT_IN_M:.0f} cm of the board's edge and "
                "not over it")
  said += askew
  if said:
    return False, "; ".join(said)
  if metrics["touchedDuringHold"]:
    touched = ", ".join(metrics["touchedDuringHold"])
    return False, f"the shelf stocked, but {touched} still touched it after the call -- let go"
  offs = _listed([f"the {p} {100 * abs(float(raw[p]['alongM'])):.1f} cm" for p in PRODUCTS])
  return True, (f"the shelf stocked in {float(raw['elapsedS']):.0f} s -- {offs} from its label "
                "-- and let go")


def sample_shelf(life, errand, result: dict | None, before: dict) -> dict:
  """MEASURE, in the registry's shape (`scoring.SAMPLERS`), as the table's.
  `result`, the skill's own word, is unread; `before["slots"]` is the job's
  word, which label each product goes over."""
  return measurements(before.get("at_call"), measure(life.model, life.data), before.get("watch"),
                      slots=before.get("slots"), started=before.get("started"),
                      limit_s=before.get("limitS"))
