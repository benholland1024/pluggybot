#!/usr/bin/env python
"""Does the feed press get onto its plate with the other robot lying by it?
(issue #439)

Live, 95 of 96 failed presses had the other robot within 2 m (the
observatory's `encounter` rows, three builds): one backed out of its own
press lies 0.25 m short of the feed plate's standoff, which is where the
other walks next. This flies that: the quadruped pair in `home_quad`, both
maps laid off the true floor, the depth camera on as the served world flies
it, one robot lying down to rest where a scene puts it, and the other
running the feed program's two verbs -- `find` the feed plate round the
lab's address, then `press` it. How the press ended and in how many tries,
how long its walk and its way in took, the resting robot's steps aside,
every plate the walker's feet came down on, touches and falls. One process
a scene.

    MUJOCO_GL=egl uv run python scripts/press_spike.py --parallel 3
    ... --scene backout,standoff_mid
    ... --drift-walker 1.24,-2.88   # the walker's map and belief laid that far
                                    # off the world together, as Luca's lab
                                    # was on 8a61ada; --drift the resting one's
    ... --out rows.json             # the records; --compare A.json B.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

H = math.pi / 2
#: Where the walker starts: just inside the lab's door (x 22, y 2.5-3.5),
#: the feed sign 3 m off and 53 deg off its face, in view; south of the row
#: facing it; and at the east wall, turned to the sign.
DOOR, SOUTH, EAST = (22.6, 3.0, 0.0), (25.0, 1.0, H), (27.2, 3.0, 2.44)
#: name -> (where the other robot lies down to rest (x, y, yaw), or None,
#: and where the walker starts). The feed sign is at (25.0, 4.83) facing
#: south: its standoff (25.0, 3.03), and a robot backed out of its own
#: press stands at (25.0, 3.28).
SCENES = {
  "alone": (None, DOOR),
  "backout": ((25.0, 3.28, H), DOOR),
  "standoff": ((25.0, 3.03, H), DOOR),
  "beside": ((24.4, 3.05, 0.0), DOOR),          # across the way in from the door
  "shock_backout": ((24.0, 3.28, H), DOOR),     # backed out of the shock plate
  "toy_backout": ((26.0, 3.28, H), DOOR),
  "backout_south": ((25.0, 3.28, H), SOUTH),
  "standoff_south": ((25.0, 3.03, H), SOUTH),
  "standoff_sw": ((25.0, 3.03, H), (24.4, 1.0, H)),
  "standoff_se": ((25.0, 3.03, H), (25.6, 1.0, H)),
  "standoff_near": ((25.0, 3.03, H), (25.0, 1.8, H)),
  "standoff_east": ((25.0, 3.03, H), EAST),
  "between_south": ((25.0, 3.15, H), SOUTH),
  "backout_east": ((25.0, 3.28, H), EAST),
}
FEED = 36
ADDRESS = (25.2, -2.4)


def drift(life, dx: float, dy: float) -> None:
  """`make_way_spike.drift`: this robot's map and belief laid (dx, dy) off
  the world together, as a scan-matched map and its belief agree."""
  import numpy as np
  g = life.body.mission.grid
  floor, out = g.grid.copy(), np.zeros_like(g.grid)
  cx, cy = round(dx / g.resolution), round(dy / g.resolution)
  rows, cols = floor.shape
  out[max(0, cy):rows + min(0, cy), max(0, cx):cols + min(0, cx)] = \
    floor[max(0, -cy):rows + min(0, -cy), max(0, -cx):cols + min(0, -cx)]
  g.grid[:] = out
  x, y, th = life.body.true_pose()
  life.body.mission._set_pose(x + dx, y + dy, th)


def fly_one(name: str, drifted=(0.0, 0.0), walker_drifted=(0.0, 0.0),
            patience: float | None = None) -> dict:
  from pluggybot import tick
  from pluggybot.activity import cage
  from pluggybot.lifecycle import QUAD_HOME
  from pluggybot.pair import build_pair
  from pluggybot.procedure.steps import PRESS_PATIENCE_S
  from unknown_spike import true_floor
  rest, start = SCENES[name]
  me, peer = build_pair(QUAD_HOME, errands=("none", "none"), near_field=True)
  for life in (me, peer):
    g = life.body.mission.grid
    g.grid[:] = true_floor(life.model, g)
  peer.body.start_at(*(rest or (8.5, 0.7, math.pi)))   # ...or far off in the garden
  me.body.start_at(*start)
  if any(drifted):
    drift(peer, *drifted)
  if any(walker_drifted):
    drift(me, *walker_drifted)
  m = me.body.mission
  pads = {n: m._pads[cage.PLATE_TAGS[n]] for n in cage.PLATE_NAMES}
  on, plates = dict.fromkeys(pads, False), dict.fromkeys(pads, 0)
  falls0 = (m.falls, peer.body.mission.falls)
  out: dict = {"scene": name, "drift": list(drifted), "walkerDrift": list(walker_drifted)}
  t_press, phases, asides, closest, seen = [0.0], [], [], [math.inf], [None]

  def feel():
    for n, pad in pads.items():
      now = m._feet_on(pad)
      plates[n] += int(now and not on[n])
      on[n] = now
    if rest is not None:
      ax, ay, _ = me.body.true_pose()
      bx, by, _ = peer.body.true_pose()
      closest[0] = min(closest[0], math.hypot(ax - bx, ay - by))
    a = peer.body.mission.last_aside
    if a is not None and a is not seen[0]:
      seen[0] = a
      asides.append({**a, "t": round(float(me.data.time) - t_press[0], 1)})

  def watched(gen):
    try:
      cmd = next(gen)
      while True:
        feel()
        cmd = gen.send((yield cmd))
    except StopIteration as e:
      return e.value

  def timed(kind, fn):
    def wrapped(*a, **kw):
      t0 = float(me.data.time)
      res = yield from fn(*a, **kw)
      d = dict(m.last_drive or {}) if kind == "walk" else {}
      phases.append({"kind": kind, "t0": round(t0 - t_press[0], 1),
                     "s": round(float(me.data.time) - t0, 1),
                     **({"arrived": bool(res), "why": d.get("why"), "asked": d.get("askedWay")}
                        if kind == "walk" else {"across": getattr(res, "root", None)})})
      return res
    return wrapped

  def resting():
    if rest is not None:
      yield from peer.body.rest_routine()
    while out.get("press") is None and out.get("find", {}).get("ok", True):
      yield from peer.body.hold_routine(0.5)

  def walking():
    while rest is not None and peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    near = (ADDRESS[0] + walker_drifted[0], ADDRESS[1] + walker_drifted[1])
    f = yield from watched(me.body.find_tag_routine(FEED, near=near, patience=300.0))
    out["find"] = {"ok": bool(f.get("found")), "why": f.get("why"), "s": f.get("seconds")}
    if not f.get("found"):
      return
    t_press[0] = float(me.data.time)
    m.drive_to_routine = timed("walk", m.drive_to_routine)
    m.clear_way_routine = timed("clear", m.clear_way_routine)
    rec = yield from watched(me.body.press_plate_routine(
      FEED, patience=PRESS_PATIENCE_S if patience is None else patience))
    out["press"] = {"pressed": bool(rec.get("pressed")), "why": rec.get("why"),
                    "s": round(float(me.data.time) - t_press[0], 1),
                    "tries": [a.get("why") for a in rec.get("attempts") or ()],
                    **({"leftS": rec["leftS"]} if "leftS" in rec else {})}

  wall = time.perf_counter()
  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, walking())],
                name="press_spike")
  rx, ry, _ = peer.body.true_pose()
  out.update({
    "phases": phases, "asides": asides, "plates": plates,
    "closestM": None if rest is None else round(closest[0], 2),
    "restMovedM": None if rest is None else round(math.dist(rest[:2], (rx, ry)), 2),
    "bumps": me.encounters.bumps if me.encounters is not None else None,
    "falls": [m.falls - falls0[0], peer.body.mission.falls - falls0[1]],
    "wallS": round(time.perf_counter() - wall, 1)})
  for life in (me, peer):
    life.body.close()
  return out


def report(rows: list[dict]) -> None:
  print(f"{'scene':15s} {'pressed':8s} {'why':12s} {'s':>6s} {'tries':24s} {'walk s':>7s} "
        f"{'asked':6s} {'asides':>6s} {'moved':>6s} {'other plates':>12s} {'bumps':>6s} falls")
  for r in rows:
    p = r.get("press") or {}
    walks = [ph for ph in r.get("phases", []) if ph["kind"] == "walk"]
    other = sum(n for plate, n in r.get("plates", {}).items() if plate != "feed")
    print(f"{r['scene']:15s} {str(p.get('pressed', False)):8s} "
          f"{p.get('why') or (r.get('find') or {}).get('why', '-'):12s} {p.get('s', 0.0):6.1f} "
          f"{','.join(p.get('tries', [])):24s} {walks[0]['s'] if walks else 0.0:7.1f} "
          f"{str(any(w.get('asked') for w in walks)):6s} {len(r.get('asides', [])):6d} "
          f"{r['restMovedM'] if r.get('restMovedM') is not None else '-':>6} {other:12d} "
          f"{str(r.get('bumps')):>6s} {r.get('falls')}")
  n = sum(bool((r.get("press") or {}).get("pressed")) for r in rows)
  print(f"pressed {n} of {len(rows)}")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--scene", default=None, help="scenes, comma separated (default all)")
  ap.add_argument("--drift", default="0,0", help="DX,DY: the resting robot's map and "
                  "belief laid that far off the world")
  ap.add_argument("--drift-walker", default="0,0", help="DX,DY: the walker's")
  ap.add_argument("--patience", type=float, default=None,
                  help="the press's patience, s (default `PRESS_PATIENCE_S`)")
  ap.add_argument("--parallel", type=int, default=1)
  ap.add_argument("--out", default=None)
  ap.add_argument("--compare", nargs="+", default=None, metavar="JSON")
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.compare:
    for path in args.compare:
      print(f"== {path}")
      report(json.loads(Path(path).read_text()))
    return 0
  drifted = tuple(float(v) for v in args.drift.split(","))
  walker_drifted = tuple(float(v) for v in args.drift_walker.split(","))
  if args.one:
    print(json.dumps(fly_one(args.one, drifted, walker_drifted, args.patience)))
    return 0
  names = args.scene.split(",") if args.scene else list(SCENES)

  def run(name: str) -> dict:
    cmd = [sys.executable, __file__, "--one", name, "--drift", args.drift,
           "--drift-walker", args.drift_walker]
    cmd += ["--patience", str(args.patience)] if args.patience is not None else []
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])
  with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
    rows = list(pool.map(run, names))
  report(rows)
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
