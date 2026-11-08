#!/usr/bin/env python3
"""What `measure` reads of each end of a top, against the truth (#636;
docs/SimNotes.md, "Where `measure` looks"): the verb's own code on the
robot's own depth pictures, at a check's set-outs, from where the references
stand and from other viewpoints.

Each end it reads is checked against the world by a ray straight down 2 cm
past it (OURS, never the robot's), at five places along the end's own
points: is the top truly gone there (`lower`), still there (`on`), or does
something taller stand there. An end it does not read says why it did not
(`window`, `picture`, `reach`, `near`, `runs`).

  python scripts/measure_spike.py --job drawer --set-outs 0,3    # the windows of #636's probe
  python scripts/measure_spike.py --job door --set-outs 0-3 --poses all    # viewpoints as well
  python scripts/measure_spike.py --out rows.jsonl

MUJOCO_GL=egl/osmesa as any headless run. A flight is about a minute.
"""
import argparse
import json
import math
import types

import mujoco
import numpy as np

from pluggybot.evaluation import counter as ct
from pluggybot.perception import box
from pluggybot.procedure import steps as st

SET_OUTS = {"drawer": ct.drawer_set_out, "door": ct.door_set_out, "cube": ct.cube_set_out,
            "lid": ct.set_out, "table": ct.table_set_out}
#: The windows #630's and #636's probes looked at, from the reference's
#: stance, m in its stance frame.
PROBE_WINDOWS = ((0.9, 0.0), (0.6, 0.0), (0.4, -0.45), (0.5, -0.3), (0.3, -0.2), (0.6, -0.6),
                 (0.9, -0.6), (0.7, 0.6), (0.45, 0.65), (0.95, 0.9), (0.36, 0.2), (0.3, -0.3))
#: `--poses all`: a grid of windows, and viewpoints off the set-out's start
#: (how far back, how far to its left, m; how far it turned, deg).
GRID = tuple((x, y) for x in (0.3, 0.45, 0.6, 0.75, 0.9, 1.1)
             for y in (-0.7, -0.45, -0.2, 0.05, 0.3, 0.55, 0.8))
VIEWS = ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.2, 0.0), (0.5, -0.2, 0.0), (0.0, 0.3, 25.0),
         (0.0, -0.3, -25.0), (1.0, 0.0, 0.0), (1.6, 0.0, 0.0))
#: Why an end is unread, off its words (`steps._measure`).
WHY = (("looked within", "window"), ("off your depth camera", "picture"), ("reach", "reach"),
       ("nearer than", "near"), ("where it read its end", "runs"))


class Pictures:
  """A body the verb looks through: the pictures already taken."""

  def __init__(self, pictures, stance):
    self.pictures, self.stance = pictures, stance

  def depth_pictures(self, *_a, **_k):
    return self.pictures
    yield


def measured(pictures, stance, x, y) -> dict:
  """`measure(x, y)` on `pictures`: its verdict and its report."""
  life = types.SimpleNamespace(body=Pictures(pictures, stance), reports=None)
  gen = st._measure(life, {"x": x, "y": y})
  try:
    while True:
      next(gen)
  except StopIteration as stop:
    return stop.value, life.reports["measure"]


def world_truth(model, data, robot_bodies, wx, wy):
  """What stands at (wx, wy) under a ray straight down: its height, m, or
  None where the robot's own body is in the way."""
  geom = np.zeros(1, dtype=np.int32)
  dist = mujoco.mj_ray(model, data, np.array([wx, wy, 2.5]), np.array([0.0, 0.0, -1.0]), None, 1,
                       -1, geom)
  if dist < 0 or model.geom_bodyid[geom[0]] in robot_bodies:
    return None
  return 2.5 - dist


def verdict_of(kinds: list[str]) -> str:
  if not kinds:
    return "-"
  votes = {k: kinds.count(k) for k in ("on", "taller", "lower")}
  best = max(votes.values())
  if not best:
    return "robot"
  return next(k for k in ("on", "taller", "lower") if votes[k] == best)


def robot_tree(model, root):
  ids = set()
  for b in range(model.nbody):
    at = b
    while at not in (0, root):
      at = model.body_parentid[at]
    if at == root and b != 0:
      ids.add(b)
  return ids


def ends_at(model, data, tree, pictures, stance, windows, out, tag):
  """Every end `measure` reads at `windows`, against the world."""
  bx, by, yaw = stance
  c, s = math.cos(yaw), math.sin(yaw)
  look = box.Look.of(pictures)
  dx, dy = look.points[:, 0] - bx, look.points[:, 1] - by
  points = np.column_stack([c * dx + s * dy, -s * dx + c * dy, look.points[:, 2]])
  counts = {}
  for (mx, my) in windows:
    done, report = measured(pictures, stance, mx, my)
    if not done["ok"]:
      continue
    top = report["top"]
    inside = points[np.hypot(points[:, 0] - mx, points[:, 1] - my) < st.MEASURE_NEAR_M]
    on_top = inside[box.on_top(inside, top)]
    for key, axis, far, _ in st.MEASURE_ENDS:
      kinds, status, at = [], "?", None
      if key in report:
        at, status = report[key], "taller" if key in report.get("taller", ()) else "read"
        _, strip = box.edge_read(on_top[:, axis], far=bool(far))
        for across in np.percentile(on_top[strip, 1 - axis], (5, 27, 50, 73, 95)):
          sv = [0.0, 0.0]
          sv[axis], sv[1 - axis] = at + (0.02 if far else -0.02), float(across)
          h = world_truth(model, data, tree, bx + c * sv[0] - s * sv[1], by + s * sv[0] + c * sv[1])
          kinds.append("robot" if h is None else "lower" if h < top - 0.015
                       else "taller" if h > top + 0.015 else "on")
      else:
        why = report["unread"].get(key, "")
        status = next((word for text, word in WHY if text in why), "?")
      row = {**tag, "at": [mx, my], "top": round(top, 3), "end": key, "status": status,
             "read": None if at is None else round(at, 3), "truth": kinds}
      if out is not None:
        out.write(json.dumps(row) + "\n")
      key_ = (status, verdict_of(kinds))
      counts[key_] = counts.get(key_, 0) + 1
  return counts


def main(args) -> None:
  set_outs = _ints(args.set_outs)
  out = open(args.out, "w") if args.out else None
  total = {}
  for k in set_outs:
    so = SET_OUTS[args.job](k)
    model, data, body = ct.world(so, None, noise=f"spike.{args.job}.{k}")
    tree = robot_tree(model, model.body(body.handle.root).id)
    sx, sy, syaw = so.start
    body.run(body.hold_routine(2.0))
    views, windows = ((VIEWS, GRID) if args.poses == "all" else (((0.0, 0.0, 0.0),), PROBE_WINDOWS))
    c0, s0 = math.cos(syaw), math.sin(syaw)
    for back, left, turn in views:
      body.start_at(sx - c0 * back - s0 * left, sy - s0 * back + c0 * left,
                    syaw + math.radians(turn))
      body.run(body.hold_routine(1.5))
      pictures = body.run(body.depth_pictures(box.DEPTH_W, box.DEPTH_H, box.DEPTH_MIN_Z,
                                              frames=box.FRAMES, apart=box.APART_S))
      got = ends_at(model, data, tree, pictures, tuple(float(v) for v in body.stance), windows, out,
                    {"job": args.job, "k": k, "view": [back, left, turn]})
      for key, n in got.items():
        total[key] = total.get(key, 0) + n
  if out is not None:
    out.close()
  print("status/truth -> ends:")
  for key in sorted(total):
    print(f"  {key[0]:8s} {key[1]:7s} {total[key]}")


def _ints(text: str) -> list[int]:
  out = []
  for part in text.split(","):
    a, _, b = part.partition("-")
    out += list(range(int(a), int(b or a) + 1))
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--job", choices=sorted(SET_OUTS), default="drawer")
  ap.add_argument("--set-outs", default="0,3", help="A or A-B, comma separated")
  ap.add_argument("--poses", choices=("start", "all"), default="start",
                  help="the reference's stance and #636's windows, or viewpoints and a grid")
  ap.add_argument("--out", help="a row an end, JSON lines")
  main(ap.parse_args())
