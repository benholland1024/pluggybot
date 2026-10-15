#!/usr/bin/env python
"""The pair at the dock and the rack (issue #455's closing reading).

The quadruped pair in the house, both maps laid from the true floor (no
explore), the depth camera on as the served world flies it. One process a
scene:

  overlap   both lying on the dock's standoff 0.13 m apart, as afb473a's
            jams left them before the pair collided -- with contact, put
            apart as a restore puts them (`HubLifecycle.unclip`); one walks
            to the hall (with contact, the living room: the other is put in
            the hall)
  double    one charging on the dock; the other comes to charge
  bump      the same, the dock's seat held by nobody (`Body.charge_seat` put
            far off): the walk in meets the robot on the dock
  wait      one lying down to rest on the dock's standoff; the other comes
            to charge (its wait bounded at 60 s)
  lane      one lying down to rest across the dock's way in, between the
            standoff and the seat and 0.2 m off its line; the other comes to
            charge (its wait bounded at 60 s)
  backout   one charging on the dock, the other lying down to rest on the
            standoff; the first backs off the dock
  swap      one lying down to rest 0.42 m from the claw bay's working pose,
            where afb473a's lay at t=1073381; the other fetches the claw
  rack      one lying down to rest ACROSS the claw bay's working pose, and
            nobody waits for it or asks it (`_bay_free_routine` and
            `make_way` put out): the rack's walk in meets it

Every scene says how deep the two robots' geoms went into each other
(`deepestM`, a contact's distance, negative inside), for how long they
touched, the fastest either root moved, and their falls; and every blind
walk either robot flew -- the dock's and the rack's walk in, every walk
back out (`headway`) -- as its own check saw it (`legs.body.Headway`): the
lowest share of its command the leg odometry made once the grace was
over, when the check fired (or, `--blind`, would have), and when the two
bodies first touched (the grader's truth, never the robot's).

    MUJOCO_GL=egl uv run python scripts/dock_pair_spike.py --scene overlap,wait
    ... --live world.npz    # `overlap` on each robot's OWN saved map, at
                            # afb473a's three jams' true poses
                            # (`--jam 954017,1051301,1126640`)
    ... --no-contact        # the pair passes through each other, as it did
                            # before #455's closing change (`PAIR_BITS`)
    ... --blind             # the blind walks in and back out do not stop
                            # when they get nowhere (`legs.body.Headway`)

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

SCENES = ("overlap", "double", "bump", "wait", "lane", "backout", "swap", "rack")
#: afb473a's jams, true poses off the `drift` rows: (Luca, Rowan), x, y, yaw deg
JAMS = {"954017": ((3.451, -0.496, 27.0), (3.409, -0.376, -83.4)),
        "1051301": ((3.428, -0.359, -84.0), (3.424, -0.412, -172.6)),
        "1126640": ((3.39, -0.423, -155.6), (3.562, -0.414, 172.6))}
HALL = (-3.5, 1.0)
#: ...and with contact, where the second robot is put is the hall: the
#: living room instead
LIVING = (0.5, 1.5)
#: `--no-contact` and `--blind`, for the child that flies a scene
NO_CONTACT = BLIND = False
#: ...and the check `--blind` put out, which the readout still replays
STALLED = None


def _pair(live: str | None = None):
  from pluggybot.legs import world as lw
  from pluggybot.lifecycle import QUAD_HOME
  from pluggybot.pair import build_pair
  from unknown_spike import true_floor
  if NO_CONTACT:
    lw.PAIR_BITS = ((0, 0), (0, 0))
  if BLIND:
    global STALLED
    from pluggybot.legs.body import Headway
    STALLED = STALLED or Headway.stalled
    Headway.stalled = lambda self, asked, made: False
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


def _contacts(me, peer):
  """The two bodies' contacts, as the grader reads them: a function giving
  this step's distances between a geom of one and a geom of the other."""
  import numpy as np
  m = me.model

  def owned(root):
    out = np.zeros(m.ngeom, dtype=bool)
    for g in range(m.ngeom):
      b = int(m.geom_bodyid[g])
      while int(m.body_parentid[b]) != 0:
        b = int(m.body_parentid[b])
      out[g] = m.body(b).name == root
    return out
  a, b = owned(me.root), owned(peer.root)

  def dists():
    d = me.data
    n = int(d.ncon)
    if not n:
      return d.contact.dist[:0]
    g = d.contact.geom[:n]
    return d.contact.dist[:n][(a[g[:, 0]] & b[g[:, 1]]) | (b[g[:, 0]] & a[g[:, 1]])]
  return dists


def _bodies(me, peer):
  """Watch the two bodies every physics step (`step_hooks`): how deep their
  geoms went into each other, m, how long they touched, s, the fastest
  either root moved, m/s, and every blind walk's headway (`_headway`).
  Returns what reads it out, with their falls."""
  import numpy as np
  m = me.model
  dists = _contacts(me, peer)
  dofs = [int(m.jnt_dofadr[m.joint(f"{life.root}_root").id]) for life in (me, peer)]
  falls0 = (me.body.mission.falls, peer.body.mission.falls)
  out = {"deepestM": 0.0, "touchingS": 0.0, "peakV": 0.0}

  def watch():
    d = me.data
    hit = dists()
    if len(hit):
      out["deepestM"] = min(out["deepestM"], float(hit.min()))
      out["touchingS"] += float(m.opt.timestep)
    out["peakV"] = max(out["peakV"], *(float(np.linalg.norm(d.qvel[k:k + 3])) for k in dofs))
  me.body.mission.step_hooks.append(watch)
  walks = _headway(me, peer, lambda: bool(len(dists())))

  def read() -> dict:
    return {**{k: round(v, 4) for k, v in out.items()},
            "falls": [me.body.mission.falls - falls0[0], peer.body.mission.falls - falls0[1]],
            "headway": walks()}
  return read


def _headway(me, peer, touching):
  """Every blind walk either robot flies, fed step by step to a `Headway` of
  its own, as the walk feeds its own: (when, asked, made, touching)."""
  import types
  from pluggybot.legs import body as qb
  walks: list = []
  for life in (me, peer):
    mis, now = life.body.mission, {"walk": None}
    for name in ("_walk_in_routine", "_rack_walk_in_routine", "_back_out_by_routine"):
      def walk(self, *a, _orig=getattr(type(mis), name), _name=name, _now=now, _root=life.root,
               **kw):
        rec = {"robot": _root, "walk": _name.strip("_").replace("_routine", "").replace("_", " "),
               "t0": float(self.data.time), "steps": []}
        walks.append(rec)
        _now["walk"] = rec
        try:
          got = yield from _orig(self, *a, **kw)
        finally:
          _now["walk"] = None
        rec["answer"] = got
        return got
      setattr(mis, name, types.MethodType(walk, mis))

    def twist(self, vx, vy, w, _orig=type(mis)._twist_routine, _now=now):
      yield from _orig(self, vx, vy, w)
      if _now["walk"] is not None:
        _now["walk"]["steps"].append((float(self.data.time), vx, float(self.odo.v[0]), touching()))
    mis._twist_routine = types.MethodType(twist, mis)

  def read() -> list:
    rows, stalled = [], STALLED or qb.Headway.stalled
    for w in walks:
      h, fired, lowest = qb.Headway(me.model.opt.timestep), None, None
      for t, asked, made, _ in w["steps"]:
        if stalled(h, asked, made):
          fired = round(t - w["t0"], 2)
          break
        if h.steps >= h.grace:
          share = h.sums[1] / h.sums[0]
          lowest = share if lowest is None else min(lowest, share)
      touched = next((round(t - w["t0"], 2) for t, *_, hit in w["steps"] if hit), None)
      rows.append({"robot": w["robot"], "walk": w["walk"], "answer": w.get("answer"),
                   "s": round(w["steps"][-1][0] - w["t0"], 2) if w["steps"] else 0.0,
                   "lowest": None if lowest is None else round(lowest, 2),
                   "firedS": fired, "touchedS": touched})
    return rows
  return read


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
  read = _bodies(me, peer)
  out: dict = {}
  if not NO_CONTACT:                       # only a restore lays them so now,
    import mujoco                          # and it puts the later one clear
    from pluggybot.lifecycle import QUAD_HOME, world_config
    peer.home_pose = tuple(world_config(QUAD_HOME)["start2"])
    mujoco.mj_forward(peer.model, peer.data)
    out["unclipped"] = peer.unclip()

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
    goal = HALL if NO_CONTACT else LIVING
    ok = yield from me.body.go_to_routine(*goal, timeout=90.0)
    out["walk"] = {"to": list(goal), "arrived": bool(ok),
                   "seconds": round(float(me.data.time) - t0, 1),
                   "said": "" if ok else me.drive_why(*goal)}

  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, walking())],
                name="overlap")
  return out | _aside(peer) | read() | {"peerSaid": _said(peer)}


def double(bump: bool = False) -> dict:
  from pluggybot import tick
  me, peer = _pair()
  sx, sy, syaw = peer.body.charge_standoff()
  peer.body.start_at(sx, sy, syaw)
  me.body.start_at(sx - 1.8, sy + 1.2, 0.0)
  if bump:                                 # nobody holds the seat: walk in
    me.body.charge_seat = lambda: (99.0, 99.0)
  read = _bodies(me, peer)
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
      if me.data.time > 120.0:              # the first never docked
        out["second"] = {"docked": False, "failure": "the first never docked"}
        return
    ok = yield from me.go_charge_routine()
    mx, my, _ = me.body.true_pose()
    px, py, _ = peer.body.true_pose()
    out["second"] = {"docked": bool(ok), "at": round(float(me.data.time), 1),
                     "apartM": round(math.hypot(mx - px, my - py), 3),
                     "failure": me.charge_failure}

  tick.run_many([(peer.body.stepper, charges_then_leaves()),
                 (me.body.stepper, comes_to_charge())], name="double")
  return out | read() | {"said": _said(me)}


def wait(lane: bool = False) -> dict:
  from pluggybot import tick
  me, peer = _pair()
  sx, sy, syaw = me.body.charge_standoff()
  if lane:                                 # half way in, 0.2 m off the line
    cx, cy = me.body.charge_seat()
    mx, my, d = (sx + cx) / 2, (sy + cy) / 2, math.dist((sx, sy), (cx, cy))
    peer.body.start_at(mx - 0.2 * (cy - sy) / d, my + 0.2 * (cx - sx) / d, syaw + math.pi / 2)
  else:
    peer.body.start_at(sx - 0.02, sy + 0.02, syaw)
  me.body.start_at(sx - 1.8, sy + 1.2, 0.0)
  me.wait_bound = lambda kind, x, y: (60.0, "charge")
  me.battery.energy_wh = me.battery.capacity_wh * 0.3
  read = _bodies(me, peer)
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
  return out | _aside(peer) | read() | {"said": _said(me) + _said(peer)}


def backout() -> dict:
  from pluggybot import tick
  me, peer = _pair()
  sx, sy, syaw = me.body.charge_standoff()
  me.body.start_at(sx, sy, syaw)
  peer.body.start_at(sx - 2.0, sy + 1.5, 0.0)
  me.battery.energy_wh = me.battery.capacity_wh * 0.5
  read = _bodies(me, peer)
  out: dict = {}

  def docks_then_backs_out():
    out["docked"] = bool((yield from me.go_charge_routine()))
    while peer.body.posture != "lying":
      yield from me.body.dock_hold_routine(0.25)
    x0, y0 = me.body.pose_xy()
    t0 = float(me.data.time)
    yield from me.body.undock_routine()
    x1, y1 = me.body.pose_xy()
    out["backedM"] = round(math.hypot(x1 - x0, y1 - y0), 3)
    out["backedS"] = round(float(me.data.time) - t0, 1)

  def comes_to_lie_on_the_standoff():
    while not me.body.charging():
      yield from peer.body.hold_routine(0.5)
      if peer.data.time > 120.0:
        return
    peer.body.start_at(sx, sy, syaw + math.pi / 2)  # across the standoff
    yield from peer.body.rest_routine()
    while "backedM" not in out and peer.data.time < 300.0:
      yield from peer.body.hold_routine(0.5)

  tick.run_many([(me.body.stepper, docks_then_backs_out()),
                 (peer.body.stepper, comes_to_lie_on_the_standoff())], name="backout")
  return out | read()


def swap(module: str = "module_claw") -> dict:
  from pluggybot import tick
  from pluggybot.rack.coupling import STATION_YS
  me, peer = _pair()
  bay = me.rack_inventory[module]
  peer.body.start_at(2.182, -1.21, math.radians(-29.7))
  me.body.start_at(0.5, 0.5, math.radians(-60.0))
  read = _bodies(me, peer)
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
  return out | _aside(peer) | read() | {"said": _said(me) + _said(peer)}


def rack(module: str = "module_claw") -> dict:
  from pluggybot import tick
  me, peer = _pair()
  mis = me.body.mission
  bay = me.rack_inventory[module]
  sx, sy, syaw = mis.rack_standoff(bay)
  wx, wy, _ = mis.work_pose(bay)
  me.body.start_at(sx, sy, syaw)
  peer.body.start_at(wx, wy, syaw + math.pi / 2)    # across the bay's working pose

  def free(*a, **kw):                                # nobody waits for it ...
    yield from ()
    return True
  mis._bay_free_routine = free
  peer.body.mission.make_way = lambda *a, **kw: False   # ... or asks it
  read = _bodies(me, peer)
  out: dict = {}

  def resting():
    yield from peer.body.rest_routine()
    while "fetch" not in out and peer.data.time < 200.0:
      yield from peer.body.hold_routine(0.5)

  def fetching():
    while peer.body.posture != "lying":
      yield from me.body.hold_routine(0.1)
    t0 = float(me.data.time)
    got = yield from mis.fetch_routine(bay, module)
    out["fetch"] = {"answer": got, "seconds": round(float(me.data.time) - t0, 1),
                    "walks": [a.get("walk") for a in mis.last_swap["attempts"]],
                    "onFork": me.body.module_state(module)["on_fork"]}

  tick.run_many([(peer.body.stepper, resting()), (me.body.stepper, fetching())], name="rack")
  return out | read()


def fly_one(scene: str, jam: str, live: str | None) -> dict:
  out = (overlap(jam, live) if scene == "overlap" else double(bump=True)
         if scene == "bump" else wait(lane=True) if scene == "lane" else globals()[scene]())
  return {"scene": scene, **({"jam": jam, "live": bool(live)} if scene == "overlap" else {}),
          **out}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--scene", default=",".join(SCENES), help="scenes, comma separated")
  ap.add_argument("--live", default=None, metavar="WORLD_NPZ",
                  help="`overlap` on each robot's own map out of a saved world")
  ap.add_argument("--jam", default="954017", help=f"`overlap`'s poses: {', '.join(JAMS)}")
  ap.add_argument("--no-contact", action="store_true",
                  help="the pair passes through each other, as before #455's closing change")
  ap.add_argument("--blind", action="store_true",
                  help="the blind walks do not stop on the other robot")
  ap.add_argument("--out", default=None)
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one:
    global NO_CONTACT, BLIND
    NO_CONTACT, BLIND = args.no_contact, args.blind
    print(json.dumps(fly_one(args.one, args.jam, args.live), default=str))
    return 0
  rows = []
  for scene in args.scene.split(","):
    for jam in (args.jam.split(",") if scene == "overlap" else [args.jam]):
      cmd = [sys.executable, __file__, "--one", scene, "--jam", jam]
      cmd += ["--live", args.live] if args.live else []
      cmd += ["--no-contact"] if args.no_contact else []
      cmd += ["--blind"] if args.blind else []
      done = subprocess.run(cmd, capture_output=True, text=True, check=True)
      rows.append(json.loads(done.stdout.strip().splitlines()[-1]))
      print(json.dumps(rows[-1], indent=1, default=str), flush=True)
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
