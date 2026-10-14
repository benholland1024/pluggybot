#!/usr/bin/env python
"""The pair at the dock and the rack (issue #455's closing reading).

The quadruped pair in the house, both maps laid from the true floor (no
explore), the depth camera on as the served world flies it. One process a
scene:

  overlap   both lying on the dock's standoff 0.13 m apart, as afb473a's
            jams left them (their bodies do not touch); one walks to the hall
  double    one charging on the dock; the other comes to charge
  wait      one lying down to rest on the dock's standoff; the other comes
            to charge (its wait bounded at 60 s)
  swap      one lying down to rest 0.42 m from the claw bay's working pose,
            where afb473a's lay at t=1073381; the other fetches the claw

    MUJOCO_GL=egl uv run python scripts/dock_pair_spike.py --scene overlap,wait
    ... --live world.npz    # `overlap` on each robot's OWN saved map, at
                            # afb473a's three jams' true poses
                            # (`--jam 954017,1051301,1126640`)

Before #455's closing change the scenes fly on staging's code (`ef4b9b5`):
SimNotes, "Two bodies on one dock".
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENES = ("overlap", "double", "wait", "swap")
#: afb473a's jams, true poses off the `drift` rows: (Luca, Rowan), x, y, yaw deg
JAMS = {"954017": ((3.451, -0.496, 27.0), (3.409, -0.376, -83.4)),
        "1051301": ((3.428, -0.359, -84.0), (3.424, -0.412, -172.6)),
        "1126640": ((3.39, -0.423, -155.6), (3.562, -0.414, 172.6))}
HALL = (-3.5, 1.0)


def _pair(live: str | None = None):
  from pluggybot.lifecycle import QUAD_HOME
  from pluggybot.pair import build_pair
  from unknown_spike import true_floor
  me, peer = build_pair(QUAD_HOME, errands=("none", "none"), near_field=True)
  if live is None:
    for life in (me, peer):
      g = life.body.mission.grid
      g.grid[:] = true_floor(life.model, g)
    return me, peer
  from drift_spike import saved_robots
  saved = saved_robots(live)
  for life in (me, peer):                  # the save's clocks first (#476)
    life.data.time = saved[life.root]["t"]
  for life in (me, peer):
    rec = saved[life.root]
    life.body.mission.restore_kept(rec["state"], rec["arrays"])
  return me, peer


def _said(life) -> list[str]:
  keep = ("MAKE WAY", "MADE WAY", "WAIT", "GO_CHARGE")
  return [line for line in life.log if any(k in line for k in keep)]


def _aside(peer) -> dict:
  return {"asides": peer.body.asides, "aside": peer.body.last_aside,
          "refused": peer.body.way_refused}


def overlap(jam: str = "954017", live: str | None = None) -> dict:
  from pluggybot import tick
  me, peer = _pair(live)
  (lx, ly, lyaw), (rx, ry, ryaw) = JAMS[jam]
  me.body.start_at(lx, ly, math.radians(lyaw))
  peer.body.start_at(rx, ry, math.radians(ryaw))
  out: dict = {}

  def resting():
    yield from peer.body.rest_routine()
    t0 = float(peer.data.time)
    while peer.data.time - t0 < 150.0 and "walk" not in out:
      yield from peer.body.hold_routine(0.5)

  def walking():
    yield from me.body.rest_routine()
    while peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    mx, my = me.body.pose_xy()
    keep = me.body.others[0]()
    out["apartM"] = round(math.hypot(keep.x - mx, keep.y - my), 3)
    t0 = float(me.data.time)
    ok = yield from me.body.go_to_routine(*HALL, timeout=90.0)
    out["walk"] = {"arrived": bool(ok), "seconds": round(float(me.data.time) - t0, 1),
                   "said": "" if ok else me.drive_why(*HALL)}

  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, walking())],
                name="overlap")
  return out | _aside(peer) | {"peerSaid": _said(peer)}


def double() -> dict:
  from pluggybot import tick
  me, peer = _pair()
  sx, sy, syaw = peer.body.charge_standoff()
  peer.body.start_at(sx, sy, syaw)
  me.body.start_at(sx - 1.8, sy + 1.2, 0.0)
  me.wait_bound = lambda kind, x, y: (300.0, "charge")
  peer.battery.energy_wh = peer.battery.capacity_wh * 0.5
  me.battery.energy_wh = me.battery.capacity_wh * 0.3
  out: dict = {"bothChargingS": 0.0}

  def charges_then_leaves():
    out["firstDocked"] = bool((yield from peer.go_charge_routine()))
    t0 = float(peer.data.time)
    while peer.data.time - t0 < 60.0:
      yield from peer.body.dock_hold_routine(0.25)
      if me.body.charging():
        out["bothChargingS"] += 0.25
    yield from peer.body.undock_routine()
    out["firstLeftAt"] = round(float(peer.data.time), 1)
    yield from peer.body.go_to_routine(0.5, 1.5, timeout=60.0)
    while "second" not in out and peer.data.time < 900.0:
      yield from peer.body.hold_routine(0.5)

  def comes_to_charge():
    while not peer.body.charging():
      yield from me.body.hold_routine(0.5)
    ok = yield from me.go_charge_routine()
    mx, my, _ = me.body.true_pose()
    px, py, _ = peer.body.true_pose()
    out["second"] = {"docked": bool(ok), "at": round(float(me.data.time), 1),
                     "apartM": round(math.hypot(mx - px, my - py), 3),
                     "failure": me.charge_failure}

  tick.run_many([(peer.body.stepper, charges_then_leaves()),
                 (me.body.stepper, comes_to_charge())], name="double")
  return out | {"said": _said(me)}


def wait() -> dict:
  from pluggybot import tick
  me, peer = _pair()
  sx, sy, syaw = me.body.charge_standoff()
  peer.body.start_at(sx - 0.02, sy + 0.02, syaw)
  me.body.start_at(sx - 1.8, sy + 1.2, 0.0)
  me.wait_bound = lambda kind, x, y: (60.0, "charge")
  me.battery.energy_wh = me.battery.capacity_wh * 0.3
  out: dict = {}

  def resting():
    yield from peer.body.rest_routine()
    t0 = float(peer.data.time)
    while peer.data.time - t0 < 180.0 and "charge" not in out:
      yield from peer.body.hold_routine(0.5)

  def charging():
    while peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    t0 = float(me.data.time)
    ok = yield from me.go_charge_routine()
    out["charge"] = {"docked": bool(ok), "seconds": round(float(me.data.time) - t0, 1),
                     "failure": me.charge_failure}

  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, charging())], name="wait")
  return out | _aside(peer) | {"said": _said(me) + _said(peer)}


def swap(module: str = "module_claw") -> dict:
  from pluggybot import tick
  from pluggybot.rack.coupling import STATION_YS
  me, peer = _pair()
  bay = me.rack_inventory[module]
  peer.body.start_at(2.182, -1.21, math.radians(-29.7))
  me.body.start_at(0.5, 0.5, math.radians(-60.0))
  out: dict = {}

  def resting():
    yield from peer.body.rest_routine()
    t0 = float(peer.data.time)
    while peer.data.time - t0 < 200.0 and "fetch" not in out:
      yield from peer.body.hold_routine(0.5)

  def fetching():
    while peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    t0 = float(me.data.time)
    got = yield from me.body.fetch_tool_routine(STATION_YS[bay], module)
    out["fetch"] = {"answer": got, "seconds": round(float(me.data.time) - t0, 1),
                    "onFork": me.body.module_state(module)["on_fork"]}

  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, fetching())], name="swap")
  return out | _aside(peer) | {"said": _said(me) + _said(peer)}


def fly_one(scene: str, jam: str, live: str | None) -> dict:
  out = overlap(jam, live) if scene == "overlap" else globals()[scene]()
  return {"scene": scene, **({"jam": jam, "live": bool(live)} if scene == "overlap" else {}),
          **out}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--scene", default=",".join(SCENES), help="scenes, comma separated")
  ap.add_argument("--live", default=None, metavar="WORLD_NPZ",
                  help="`overlap` on each robot's own map out of a saved world")
  ap.add_argument("--jam", default="954017", help=f"`overlap`'s poses: {', '.join(JAMS)}")
  ap.add_argument("--out", default=None)
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one:
    print(json.dumps(fly_one(args.one, args.jam, args.live), default=str))
    return 0
  rows = []
  for scene in args.scene.split(","):
    for jam in (args.jam.split(",") if scene == "overlap" else [args.jam]):
      cmd = [sys.executable, __file__, "--one", scene, "--jam", jam]
      cmd += ["--live", args.live] if args.live else []
      done = subprocess.run(cmd, capture_output=True, text=True, check=True)
      rows.append(json.loads(done.stdout.strip().splitlines()[-1]))
      print(json.dumps(rows[-1], indent=1, default=str), flush=True)
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
