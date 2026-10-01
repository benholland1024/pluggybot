#!/usr/bin/env python
"""Does a robot resting across the other's way make way? (issue #415)

The quadruped pair in the house, both maps laid from the true floor (no
explore), one robot lying down to rest in a doorway -- or wherever a scene
puts it -- and the other walking from one side of it to the other. How the
walk ends (arrived, or why it gave up), what the resting robot did (asked,
stood, stepped how far aside, how long it took, why it stopped), whether the
two touched, and anybody falling. One process a scene.

    MUJOCO_GL=egl uv run python scripts/make_way_spike.py --parallel 4
    ... --before            # nobody asks: the drive before #415
    ... --finished          # the resting robot's day is over: nothing it
                            # runs holds it, as a scripted fixture day's
    ... --scene hall_door,garden_door
    ... --out scenes.json   # the records; --compare A.json B.json
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
#: name -> (where the robot rests, where the other starts, where it walks
#: to); poses as (x, y, yaw). The doorways are `home.world`'s.
SCENES = {
  # living <-> hall, the house's west wall at x -2, door y -1..0 (#415's)
  "hall_door": ((-2.0, -0.5, 0.0), (-3.5, 3.0, -H), (1.5, -0.5)),
  "hall_door_back": ((-2.0, -0.5, 0.0), (1.0, -0.5, math.pi), (-3.5, 3.0)),
  # living <-> bedroom, the divider at y 2.5, door x 1..2
  "bedroom_door": ((1.5, 2.5, H), (1.5, 0.0, H), (1.5, 4.5)),
  # hall <-> kitchen and hall <-> workshop, the wing's east wall at x -5
  "kitchen_door": ((-5.0, 4.0, math.pi), (-3.5, 0.0, H), (-8.5, 4.0)),
  "workshop_door": ((-5.0, 1.0, math.pi), (-3.5, 4.0, -H), (-8.5, -4.5)),
  # living <-> the east garden, the house's east wall at x 5, door y 0.2..1.2
  "garden_door": ((5.0, 0.7, 0.0), (2.0, 0.5, 0.0), (7.5, 0.7)),
  # the middle of the 3 m hall: the way round it is open, nobody is asked
  "hall_middle": ((-3.5, 1.0, H), (-3.5, 4.5, -H), (-3.5, -2.5)),
}


def fly_one(name: str, before: bool, finished: bool, patience: float) -> dict:
  from pluggybot import tick
  from pluggybot.lifecycle import QUAD_HOME
  from pluggybot.pair import build_pair
  from unknown_spike import true_floor
  rest, start, goal = SCENES[name]
  me, peer = build_pair(QUAD_HOME, errands=("none", "none"))
  for life in (me, peer):
    g = life.body.mission.grid
    g.grid[:] = true_floor(life.model, g)
    if before:
      life.body.ask_way = None
  peer.body.start_at(*rest)
  me.body.start_at(*start)
  falls0 = (me.body.mission.falls, peer.body.mission.falls)
  out: dict = {"scene": name, "before": before, "finished": finished}

  def resting():
    yield from peer.body.rest_routine()
    if finished:
      return
    t0 = float(peer.data.time)
    while peer.data.time - t0 < patience + 30.0 and out.get("walk") is None:
      yield from peer.body.hold_routine(0.5)
    yield from peer.body.hold_routine(2.0)

  def walking():
    while peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    t0 = float(me.data.time)
    ok = yield from me.body.go_to_routine(goal[0], goal[1], timeout=patience)
    rec = dict(me.body.mission.last_drive or {})
    out["walk"] = {"arrived": bool(ok), "why": rec.get("why", ""),
                   "seconds": round(float(me.data.time) - t0, 1),
                   "shortM": rec.get("shortM"), "askedWay": rec.get("askedWay", []),
                   "said": me.drive_why(*goal) if not ok else ""}

  wall = time.perf_counter()
  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, walking())],
                name="make_way")
  rx, ry, _ = peer.body.true_pose()
  out.update({
    "aside": peer.body.last_aside, "asides": peer.body.asides,
    "restEnd": [round(rx, 2), round(ry, 2)],
    "restMovedM": round(math.dist(rest[:2], (rx, ry)), 2),
    "bumps": me.encounters.bumps if me.encounters is not None else None,
    "falls": [me.body.mission.falls - falls0[0], peer.body.mission.falls - falls0[1]],
    "wallS": round(time.perf_counter() - wall, 1)})
  for life in (me, peer):
    life.body.close()
  return out


def report(rows: list[dict]) -> None:
  print(f"{'scene':16s} {'arrived':8s} {'why':6s} {'s':>6s} {'asked':6s} "
        f"{'aside':12s} {'aside s':>8s} {'moved':>6s} {'bumps':>6s} falls")
  for r in rows:
    w, a = r["walk"], r.get("aside") or {}
    print(f"{r['scene']:16s} {str(w['arrived']):8s} {w['why'] or '-':6s} "
          f"{w['seconds']:6.1f} {str(bool(w['askedWay'])):6s} "
          f"{a.get('why', '-'):12s} {a.get('seconds', 0.0):8.1f} {r['restMovedM']:6.2f} "
          f"{str(r['bumps']):>6s} {r['falls']}")
    if w["said"]:
      print(f"    {w['said']}")
  n = sum(r["walk"]["arrived"] for r in rows)
  print(f"arrived {n} of {len(rows)}")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--scene", default=None, help="scenes, comma separated (default all)")
  ap.add_argument("--before", action="store_true", help="nobody asks (the drive before #415)")
  ap.add_argument("--finished", action="store_true",
                  help="the resting robot's day is over once it lies down")
  ap.add_argument("--patience", type=float, default=90.0)
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
  if args.one:
    print(json.dumps(fly_one(args.one, args.before, args.finished, args.patience)))
    return 0
  names = args.scene.split(",") if args.scene else list(SCENES)

  def run(name: str) -> dict:
    cmd = [sys.executable, __file__, "--one", name, "--patience", str(args.patience)]
    cmd += ["--before"] if args.before else []
    cmd += ["--finished"] if args.finished else []
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
