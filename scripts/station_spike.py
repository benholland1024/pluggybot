"""The humanoid's charging station (issue #559, stage 2; SimNotes, "The
humanoid's charging station"), measured as #378's dock was: the humanoid in
the house, its station on the living room's wall.

    MUJOCO_GL=egl uv run python scripts/station_spike.py            # filmstrip
    ... --approach [--n N] [--parallel P]   # from round the standoff: docked
                                # and unplugged or why not, tries, steps, time
    ... --hold [--seconds S]    # plugged in: the pack's gain, W, and the plug held
    ... --fall                  # shoved over plugged in: the breakaway, the
                                # plug home, the robot up again
    ... --seats [--parallel P]  # started at seats over what the seat check
                                # accepts (3 cm in to 6 cm back, 6 cm across,
                                # 6 deg): plugged in and unplugged or why not
    ... --walk-in [--explore 120,240] [--parallel P]   # explored, walked out
                                # across the house on its own map, then the
                                # loop's own charge path: docked or why not
    ... --lost [--n N] [--parallel P]   # a fresh robot 2 m out, its belief set
                                # 0.4-1 m wrong: the loop's charge path, its
                                # lost search and the fix off the station
    ... --lost --far [--from near,house,street] [--n N] [--explore S]
                                # from further off (#572): its belief 1-5 m
                                # and 5-20 deg wrong, 2 m out, in the house's
                                # rooms or across the street; how each ended
    ... --lost-table rows.json  # --far's table again, from its --out
    ... --reach                 # how far its cameras read the station's tags:
                                # each camera's tags decoded, out of the face and
                                # off its axis, and the head camera's fit (#572)
    ... --view [plug|fall|lost] # a scene in the viewer, looped (MUJOCO_GL unset)
    ... --strip [plug|lost]     # a scene's filmstrip
    ... --lost-from P --lost-off X,Y,DEG   # the lost scene from elsewhere
    ... --out rows.json         # the rows

Every trial is its own process and flies the served body off its own senses
(the station off its tags, its port off its model of itself); the truth
(where the plug is, whether it fell) is ours, read to grade it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

def _child(*args: str) -> list[str]:
  """A trial's process: this script with `args`."""
  return [sys.executable, __file__, *args]


#: A filmstrip's tile, px, and its row.
TILE_W, TILE_H, ROW = 480, 360, 4
#: The viewer's camera a scene, fixed from the robot's right (`side`): the
#: point it looks at in the station's frame (m) and how far off it stands --
#: the fall's wide enough for where the shove throws the robot, 1.8 m out
#: and 0.7 m across, and the plug's flight home. Then its elevation, deg,
#: and the physics steps between the viewer's frames.
VIEW_CAM = {"plug": ((-0.45, 0.075, 0.75), 2.0), "fall": ((-0.95, 0.35, 0.6), 3.2),
            "lost": ((-1.2, 0.3, 0.7), 3.8)}
VIEW_ELEVATION_DEG, SYNC_EVERY = -15.0, 10


def side(station) -> float:
  """A camera's azimuth, deg, on the robot's right as it faces the station,
  where its hand works. ⚠ The station is on the wall the robot faces: at
  the viewer's default azimuth, in front of the robot, the camera stood
  outside the house and the view was the back of that wall."""
  return math.degrees(station.yaw) + 90.0


def world(off=(0.0, 0.0, 0.0), at_seat: bool = False):
  """The house, one humanoid at the station's standoff -- or seat -- off by
  `off` (m along and across, deg) in the station's frame, its activities
  hooked; (model, data, body, station activity)."""
  from pluggybot.activity import station as st
  from pluggybot.body import body_for
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  body = body_for(model, data)
  s = body._station
  x0 = st.SEAT_X if at_seat else st.STANDOFF_X
  x, y = s.to_world(x0 - off[0], st.SEAT_Y + off[1])
  body.start_at(x, y, s.yaw + math.radians(off[2]))
  acts = cfg["activities"](model, data)
  station = next(a for a in acts if a.name == "station")
  body.step_hooks.append(acts.step_hook(model, data))
  body.run(body.hold_routine(1.0))
  return model, data, body, station


def trial(off) -> dict:
  """From `off` round the standoff: docked, a few seconds charging,
  unplugged."""
  model, data, body, station = world(off)
  t0 = float(data.time)
  why = body.run(body.dock_routine())
  out = {"off": list(off), "dock": why, "dockS": round(float(data.time) - t0, 1),
         "charging": body.charging(), "falls": body.falls}
  trace = list(body._charge_trace)
  out["seatSteps"] = sum(1 for t in trace if "seat" in t) - 1
  out["pushes"] = sum(1 for t in trace if "try" in t)
  if body.charging():
    body.run(body.dock_hold_routine(5.0))
    t1 = float(data.time)
    body.run(body.undock_routine())
    body.run(body.hold_routine(4.0))
    out.update(undock=next((t["undock"] for t in body._charge_trace if "undock" in t), "unplugged"),
               undockS=round(float(data.time) - t1, 1), unplugged=not body.charging(),
               parked=bool(station.flags["parked"]), breaks=station.breaks)
  out["fallsAfter"] = body.falls
  return out


def approach(n: int, parallel: int, seed: int = 559) -> list[dict]:
  """The starts: a grid of three (head on, and 0.2 m and 20 deg off each
  way), then `n` drawn round the standoff (+-0.2 m along, +-0.25 m across,
  +-25 deg) -- one process a trial."""
  rng = np.random.default_rng(seed)
  starts = [(0.0, 0.0, 0.0), (0.0, 0.2, 20.0), (0.0, -0.2, -20.0)]
  starts += [tuple(float(round(v, 3)) for v in (rng.uniform(-0.2, 0.2), rng.uniform(-0.25, 0.25),
                                                rng.uniform(-25, 25))) for _ in range(n)]

  def run(off):
    cmd = _child("--one", json.dumps(off))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
      return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
      return {"off": list(off), "dock": "crashed", "stderr": r.stderr[-400:]}

  with ThreadPoolExecutor(parallel) as pool:
    rows = list(pool.map(run, starts))
  print(f"{'start (m, m, deg)':22s} {'dock':40s} {'s':>5s} {'steps':>5s} {'push':>4s}  {'unplug':12s} {'s':>5s} falls")
  for r in rows:
    print(f"{str(tuple(r['off'])):22s} {r['dock'][:40]:40s} {r.get('dockS', 0):5.1f} "
          f"{r.get('seatSteps', 0):5d} {r.get('pushes', 0):4d}  {str(r.get('undock', '-'))[:12]:12s} "
          f"{r.get('undockS', 0):5.1f} {r.get('fallsAfter', '-')}")
  docked = [r for r in rows if r["dock"] == "docked"]
  unplugged = [r for r in docked if r.get("unplugged") and r.get("parked")]
  print(f"docked {len(docked)} of {len(rows)}; unplugged and parked {len(unplugged)} of {len(docked)}; "
        f"falls {sum(r.get('fallsAfter', 0) for r in rows)}")
  return rows


#: `--seats`' grid, in the station's frame: how far behind the seat (m, +
#: away from the wall), to its left (m) and turned to its left (deg).
SEAT_BACKS, SEAT_LEFTS, SEAT_TURNS = ((-0.03, 0.0, 0.03, 0.06), (-0.06, -0.04, -0.02, 0.0, 0.02,
                                                                0.04, 0.06), (-6, -3, 0, 3, 6))


def seat_trial(off) -> dict:
  """From the seat `off` (as `--seats` draws it): the station fitted from
  the standoff first and again at the seat, the standoff's fit kept where
  the seat's sees no pair -- as `dock_routine` keeps it -- then the plug-in
  and, plugged in, the unplug and the plug home."""
  from pluggybot.activity import station as st
  from pluggybot.humanoid import charge
  model, data, body, station = world()
  first = charge.look(body)
  s = body._station
  x, y = s.to_world(st.SEAT_X - off[0], st.SEAT_Y + off[1])
  body.start_at(x, y, s.yaw + math.radians(off[2]))
  body.run(body.hold_routine(1.0))
  seen = charge.look(body) or first
  out = {"off": list(off), "lookedAtSeat": charge.look(body) is not None}
  if seen is None:
    return {**out, "why": "no look"}
  trace = []
  t0 = float(data.time)
  why = body.run(charge.plug_in_routine(body, seen, trace))
  out.update(why=why, s=round(float(data.time) - t0, 1), tries=len(trace),
             swing=trace[-1]["swing"] if trace else None, falls=body.falls)
  if why == "plugged in":
    t1 = float(data.time)
    out["unplug"] = body.run(charge.unplug_routine(body, seen, []))
    body.run(body.hold_routine(4.0))
    out.update(unplugS=round(float(data.time) - t1, 1), parked=bool(station.flags["parked"]),
               fallsAfter=body.falls)
  return out


def seats(parallel: int) -> list[dict]:
  """Every seat of `SEAT_*`'s grid, one process a trial."""
  import itertools
  offs = list(itertools.product(SEAT_BACKS, SEAT_LEFTS, SEAT_TURNS))

  def run(off):
    cmd = _child("--one-seat", json.dumps(off))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
      return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
      return {"off": list(off), "why": "crashed", "stderr": r.stderr[-400:]}

  with ThreadPoolExecutor(parallel) as pool:
    rows = list(pool.map(run, offs))
  for r in rows:
    if r.get("why") != "plugged in" or r.get("unplug") != "unplugged" or not r.get("parked"):
      print(f"{str(tuple(r['off'])):22s} {r.get('why', '')[:50]:50s} {r.get('unplug', '-')}")
  docked = [r for r in rows if r.get("why") == "plugged in"]
  home = [r for r in docked if r.get("unplug") == "unplugged" and r.get("parked")]
  print(f"plugged in {len(docked)} of {len(rows)} (swung {sum(r['swing'] == 20 for r in docked)}, "
        f"straight across {sum(r['swing'] == 0 for r in docked)}); unplugged and the plug home "
        f"{len(home)} of {len(docked)}; falls {sum(r.get('fallsAfter', r.get('falls', 0)) for r in rows)}")
  return rows


#: Where `--walk-in` sends the robot before the loop sends it to charge: the
#: house's spawns and rooms across it from the station
#: (`models/home_world.meta.json`), and the facility's across the street.
WALK_IN_FROM = {"bedroom": (1.5, 4.5), "hall": (-3.5, 1.0), "kitchen": (-8.5, 4.0),
                "workshop": (-8.5, -2.0), "garden": (7.5, 2.0), "garden_south": (3.0, -4.0),
                "lobby": (20.5, 0.0), "lab": (25.0, 2.0)}


def walk_in(place: str, explore_s: float) -> dict:
  """The loop's own charge path (`HubLifecycle.go_charge`) from `place`: a
  fresh humanoid explores for `explore_s`, walks there on its own map, and
  is sent to charge -- the walk to the standoff on its belief, the station
  off its tags, the plug-in. Docked or why not, how far its belief was off
  (m, deg) there and at the end, the times."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  cfg = world_config(HUMANOID_HOME)
  life, body, station = _life()
  data = life.data
  life.blacklist, life.floor_explored = set(), False

  def off():
    (bx, by, bh), (tx, ty, th) = body.pose, body.true_pose()
    turn = abs((bh - th + math.pi) % (2 * math.pi) - math.pi)
    return [round(math.hypot(bx - tx, by - ty), 3), round(math.degrees(turn), 1)]

  out = {"place": place, "exploreS": explore_s}
  try:
    body.start_at(*cfg["start"])
    body.start_discovery()
    body.run(body.look_around_routine())
    life.explore(budget=explore_s, mark_done=False)
    x, y = WALK_IN_FROM[place]
    t0 = float(data.time)
    out["there"] = bool(body.run(body.go_to_routine(x, y, timeout=300.0)))
    out["outWhy"] = "" if out["there"] else life.drive_why(x, y)
    tx, ty, _ = body.true_pose()
    out.update(outS=round(float(data.time) - t0, 1), fromM=round(math.hypot(tx - x, ty - y), 2),
               offThere=off(), parked=bool(station.flags["parked"]))
    t1 = float(data.time)
    out["docked"] = bool(life.go_charge())
    out.update(why=life.charge_failure, inS=round(float(data.time) - t1, 1), offEnd=off(),
               tries=sum(1 for t in body._charge_trace if "try" in t), falls=body.falls,
               trace=body.charge_trace())
  finally:
    body.close()
  return out


def walk_ins(explores: tuple, parallel: int) -> list[dict]:
  """Every place in `WALK_IN_FROM` after each explore -- one process a
  trial."""
  trials = [(p, e) for e in explores for p in WALK_IN_FROM]

  def run(t):
    cmd = _child("--one-walk-in", json.dumps(t))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    try:
      return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
      return {"place": t[0], "exploreS": t[1], "docked": False, "why": "crashed",
              "stderr": r.stderr[-400:]}

  with ThreadPoolExecutor(parallel) as pool:
    rows = list(pool.map(run, trials))
  print(f"{'from':13s} {'explore':>7s} {'there':>5s} {'out s':>6s} {'off there':>12s} "
        f"{'docked':>6s} {'in s':>6s} {'off end':>12s} {'tries':>5s} falls  why not")
  for r in rows:
    print(f"{r['place']:13s} {r['exploreS']:7.0f} {str(r.get('there', '-')):>5s} "
          f"{r.get('outS', 0):6.1f} {str(r.get('offThere', '-')):>12s} {str(r['docked']):>6s} "
          f"{r.get('inS', 0):6.1f} {str(r.get('offEnd', '-')):>12s} {r.get('tries', 0):5d} "
          f"{r.get('falls', '-')!s:>5s}  {r.get('why', '')}")
  print(f"docked {sum(r['docked'] for r in rows)} of {len(rows)}")
  return rows


def _life():
  """The humanoid's house and a lifecycle on it, as the loop runs one: the
  station's activity hooked, no deadline. (life, body, station)."""
  from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, points_ledger, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=cfg["hosting_battery_wh"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], ledger=points_ledger(None),
                      world=HUMANOID_HOME, errands=[], near_field=True)
  acts = cfg["activities"](model, data)
  station = next(a for a in acts if a.name == "station")
  life.body.step_hooks.append(acts.step_hook(model, data))
  life.max_sim_time = life.explore_deadline = 1e9
  return life, life.body, station


#: `--lost`'s start, in the station's frame (m): 2 m out on the seat's line,
#: facing the station -- past 1.07 m, so the tags are not in view.
LOST_FROM_X = -2.0
#: ...and `--view lost`'s and `--strip lost`'s belief, set this wrong: 0.7 m
#: nearer the wall than it is, so the walk to the standoff ends 1.45 m off
#: the station's face, where its tags are above the colour camera's view.
LOST_SCENE_OFF = (0.7, 0.0, 0.0)
#: `--lost --far`'s starts (#572): 2 m out, the house's places and the
#: facility's across the street (`WALK_IN_FROM`), and the storeroom whose
#: counter the jobs are at (#559's stage 3) -- each facing the station.
LOST_FROM = {"station": None, **WALK_IN_FROM, "store": (25.0, -3.0)}
#: ...in the far table's three groups (`--from` takes a group's name too).
LOST_GROUPS = {"near": ("station",),
               "house": ("bedroom", "hall", "kitchen", "workshop", "garden", "garden_south"),
               "street": ("lobby", "lab", "store")}
#: ...its belief set this far wrong, m, and turned this much, deg, either
#: way: what the live pair's had drifted to (#572). 2 m out, a grid of each
#: (`FAR_GRID_M` by `FAR_GRID_DEG`, a bearing and a side drawn); elsewhere,
#: drawn.
FAR_M, FAR_DEG = (1.0, 5.0), (5.0, 20.0)
FAR_GRID_M, FAR_GRID_DEG = (1.0, 2.0, 3.0, 4.0, 5.0), (5.0, 12.0, 20.0)


def zone_of(x: float, y: float) -> str:
  """The house's zone a point is in (`home.ZONES`), or "?"."""
  from pluggybot.home import world as home
  for z in home.ZONES:
    (x0, y0), (x1, y1) = z["min"], z["max"]
    if x0 <= x <= x1 and y0 <= y <= y1:
      return z["name"]
  return "?"


def lost_start(body, off, place: str = "station") -> None:
  """`body` at `place` -- `LOST_FROM_X` facing the station, or one of
  `LOST_FROM` facing it -- its belief put `off` wrong (m along the
  station's +x, m across it, deg)."""
  from pluggybot.activity import station as st
  s = body._station
  if LOST_FROM[place] is None:
    (x, y), yaw = s.to_world(LOST_FROM_X, st.SEAT_Y), s.yaw
  else:
    x, y = LOST_FROM[place]
    cx, cy = s.to_world(0.0, 0.0)
    yaw = math.atan2(cy - y, cx - x)
  body.start_at(x, y, yaw)
  dx, dy = s.to_world(off[0], off[1])
  body.mission._set_pose(x + dx - s.x, y + dy - s.y, yaw + math.radians(off[2]))


def _off(belief, truth) -> list:
  """How far a belief is from the truth: m, and deg (+ turned left of it)."""
  (bx, by, bh), (tx, ty, th) = belief, truth
  return [round(math.hypot(bx - tx, by - ty), 3),
          round(math.degrees(math.remainder(bh - th, 2 * math.pi)), 1)]


def _wire(p) -> tuple:
  """A belief event's pose (`navigator.wire_pose`) back as (x, y, rad)."""
  return p["x"], p["y"], math.radians(p["yawDeg"])


def lost_trial(off, place: str = "station", explore_s: float = 0.0) -> dict:
  """A fresh humanoid at `place` (`lost_start`), its belief put `off`
  wrong (m along the station's +x, m across it, deg) on an empty map --
  explored `explore_s` first, if any, laying its map through that belief
  as the live pair's were -- and the loop's own charge path
  (`HubLifecycle.go_charge`): docked or why not, the belief's error at the
  start, as the charge began, where the lost search began (and where the
  robot truly was) and at the end (m, deg), how the walk to the standoff
  last ended, and the belief's records (`lost`, `fixed`, `searched`)."""
  life, body, station = _life()

  def err():
    return _off(body.pose, body.true_pose())

  try:
    lost_start(body, off, place)
    out = {"off": list(off), "from": place, "offStart": err()}
    if explore_s:
      life.blacklist, life.floor_explored = set(), False
      body.start_discovery()
      body.run(body.look_around_routine())
      life.explore(budget=explore_s, mark_done=False)
    out.update(exploreS=explore_s, offCharge=err())
    t0 = float(life.data.time)
    out["docked"] = bool(life.go_charge())
    walk = body.last_drive or {}
    out.update(why=life.charge_failure, s=round(float(life.data.time) - t0, 1), offEnd=err(),
               zoneEnd=zone_of(*body.true_pose()[:2]), falls=body.falls,
               walk={"why": walk.get("why"), "shortM": walk.get("shortM")})
    events = []
    for e in body.belief_events:
      rec = {k: v for k, v in e.items() if k in ("why", "fixture", "found", "ended", "seconds",
                                                  "viewpoints", "arounds", "dropped", "atM")}
      truth = _wire(e["truth"])
      rec.update(off=_off(_wire(e["before"]), truth), zone=zone_of(*truth[:2]))
      events.append(rec)
    out["events"] = events
    out["outcome"] = outcome(out)
  finally:
    body.close()
  return out


def outcome(r: dict) -> str:
  """How a lost trial ended: docked without a search (the station in sight
  where its walk ended), found by the search and docked, found and not
  docked, not found and docked all the same (a retried walk ended in
  sight of it), not found, or never searched -- its walk gave up further
  from the standoff than the board decides from."""
  if r.get("why") == "crashed":
    return "crashed"
  searched = next((e for e in r.get("events", []) if e["why"] == "searched"), None)
  if searched is None:
    return "docked" if r["docked"] else "never searched"
  if searched["found"]:
    return "found, docked" if r["docked"] else "found, not docked"
  return "missed, docked" if r["docked"] else "not found"


def losts(n: int, parallel: int, seed: int = 476) -> list[dict]:
  """Four set wrong each way -- 0.7 m out from the wall, in toward it, and
  either way along it -- and `n` drawn 0.4-1 m off in any direction and up
  to 5 deg turned, one process a trial."""
  rng = np.random.default_rng(seed)
  offs = [(-0.7, 0.0, 0.0), (0.7, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, -0.7, 0.0)]
  for _ in range(n):
    r, a = rng.uniform(0.4, 1.0), rng.uniform(-math.pi, math.pi)
    offs.append(tuple(float(round(v, 3)) for v in (r * math.cos(a), r * math.sin(a),
                                                     rng.uniform(-5.0, 5.0))))
  rows = _fly_losts([(off, "station", 0.0) for off in offs], parallel)
  print(f"{'belief set off (m, m, deg)':28s} {'docked':>6s} {'s':>6s} {'off end':>14s}  "
        f"searched  why not")
  for r in rows:
    found = next((e for e in r.get("events", []) if e["why"] == "searched"), None)
    print(f"{str(tuple(r['off'])):28s} {str(r['docked']):>6s} {r.get('s', 0):6.1f} "
          f"{str(r.get('offEnd', '-')):>14s}  "
          f"{'-' if found is None else (found['ended'] + ' ' + str(found['seconds']) + ' s'):9s} "
          f"{r.get('why', '')}")
  print(f"docked {sum(r['docked'] for r in rows)} of {len(rows)}")
  return rows


def far_losts(places: list[str], n: int, parallel: int, explore_s: float = 0.0,
              seed: int = 572) -> list[dict]:
  """`--lost --far` (#572): from each of `places` (`LOST_FROM`), the belief
  set `FAR_M` and `FAR_DEG` wrong -- 2 m out, every cell of the grid; from
  anywhere else, `n` drawn -- a bearing and a side drawn for each, one
  process a trial."""
  rng = np.random.default_rng(seed)
  trials = []
  for place in places:
    cells = ([(r, deg) for r in FAR_GRID_M for deg in FAR_GRID_DEG] if LOST_FROM[place] is None
             else [(rng.uniform(*FAR_M), rng.uniform(*FAR_DEG)) for _ in range(n)])
    for r, deg in cells:
      a, side = rng.uniform(-math.pi, math.pi), rng.choice((-1.0, 1.0))
      off = tuple(float(round(v, 3)) for v in (r * math.cos(a), r * math.sin(a), side * deg))
      trials.append((off, place, explore_s))
  rows = _fly_losts(trials, parallel)
  lost_table(rows)
  return rows


def _fly_losts(trials: list, parallel: int) -> list[dict]:
  """Each `(off, place, explore_s)` a `lost_trial` in a process of its own."""
  def run(t):
    off, place, explore_s = t
    cmd = _child("--one-lost", json.dumps(
      {"off": list(off), "from": place, "explore": explore_s}))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    try:
      return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
      return {"off": list(off), "from": place, "docked": False, "why": "crashed",
              "outcome": "crashed", "stderr": r.stderr[-400:]}

  with ThreadPoolExecutor(parallel) as pool:
    return list(pool.map(run, trials))


def _set_off(r: dict) -> tuple[float, float]:
  """A trial's belief as it was set wrong: m, and deg."""
  return math.hypot(r["off"][0], r["off"][1]), r["off"][2]


def lost_table(rows: list[dict]) -> None:
  """A far lost sweep: a line a trial -- where from, how wrong its belief
  was set, as its charge began and where the search began (and where it
  truly was), how it ended -- then how each group and each band of the set
  error ended."""
  for r in rows:
    r["outcome"] = outcome(r)
  print(f"{'from':12s} {'set m/deg':>11s} {'at charge':>13s} {'at search':>13s} "
        f"{'truly in':12s} {'outcome':17s} {'search s':>8s} {'s':>6s} {'end':>13s}  why not")
  for r in rows:
    m, deg = _set_off(r)
    lost = next((e for e in r.get("events", []) if e["why"] == "lost"), None)
    found = next((e for e in r.get("events", []) if e["why"] == "searched"), None)
    print(f"{r.get('from', 'station'):12s} {m:5.2f}/{deg:+5.1f} "
          f"{str(r.get('offCharge', '-')):>13s} {str(lost['off'] if lost else '-'):>13s} "
          f"{lost['zone'] if lost else r.get('zoneEnd', '-'):12s} {r['outcome']:17s} "
          f"{found['seconds'] if found else '-':>8} {r.get('s', 0):6.1f} "
          f"{str(r.get('offEnd', '-')):>13s}  {'' if r['docked'] else r.get('why', '')}")
  kinds = ("docked", "found, docked", "found, not docked", "missed, docked", "not found",
           "never searched", "crashed")

  def line(label, rs):
    if not rs:
      return
    mix = ", ".join(f"{k} {sum(r.get('outcome') == k for r in rs)}" for k in kinds
                    if any(r.get("outcome") == k for r in rs))
    print(f"  {label:26s} docked {sum(r['docked'] for r in rs):2d} of {len(rs):2d}  ({mix})")

  print("== how they ended")
  for group, places in LOST_GROUPS.items():
    line(group, [r for r in rows if r.get("from", "station") in places])
  for lo, hi in ((1.0, 2.0), (2.0, 3.0), (3.0, 5.01)):
    line(f"set {lo:g}-{min(hi, 5):g} m off", [r for r in rows if lo <= _set_off(r)[0] < hi])
  for lo, hi in ((5.0, 10.0), (10.0, 20.01)):
    line(f"set {lo:g}-{min(hi, 20):g} deg off",
         [r for r in rows if lo <= abs(_set_off(r)[1]) < hi])
  line("all", rows)


#: `--reach`: the robot this far out of the station's face, m, along lines
#: this far off its axis, deg (to its left), facing the station and turned
#: this far off it, deg -- each a decode of the head's camera and of the
#: colour imager.
REACH_M = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
REACH_DEG = (0.0, 15.0, 30.0, 50.0, 65.0)
REACH_TURNS_DEG = (-4.0, 4.0)


def reach() -> list[dict]:
  """How far its cameras read the station's two tags (#572): at each place
  of `REACH_M` by `REACH_DEG`, facing the station `REACH_TURNS_DEG` off, held
  0.6 s, one decode of the head's camera and one of the colour imager -- the
  tags each read -- and, off the head's, each tag placed as `see_station`
  places it and the pair fitted (`charge.fit`), against where they truly
  are. The belief is the truth here: what is measured is the look."""
  from pluggybot.activity import station as st
  from pluggybot.humanoid import charge
  from pluggybot.humanoid.body import TAG_CAMERA
  from pluggybot.humanoid.places import NAV_CAMERA
  from pluggybot.perception.tags import STATION_TAG_IDS, TagDetector
  model, data, body, _ = world()
  s, m = body._station, body.mission
  cams = {name: TagDetector(model, body.handle.el(cam))
          for name, cam in (("head", NAV_CAMERA), ("colour", TAG_CAMERA))}
  truth = {t: s.to_world(0.0, y) for t, (_, y, _) in st.tag_layout().items()}
  rows = []
  try:
    for dist in REACH_M:
      for off in REACH_DEG:
        for turn in REACH_TURNS_DEG:
          a = math.radians(off)
          x, y = s.to_world(-dist * math.cos(a), dist * math.sin(a))
          cx, cy = s.to_world(0.0, 0.0)
          body.start_at(x, y, math.atan2(cy - y, cx - x) + math.radians(turn))
          body.run(body.hold_routine(0.6))
          r = {"m": dist, "offDeg": off, "turnDeg": turn}
          for name, det in cams.items():
            r[name] = sorted(t for t in det.detect(data) if t in STATION_TAG_IDS)
          dets = cams["head"].detect(data)
          ids = [t for t in STATION_TAG_IDS if t in dets]
          if ids:
            _, placed = m._placed(dets, ids)
            r["tagM"] = max(round(math.hypot(p[0] - truth[t][0], p[1] - truth[t][1]), 3)
                            for t, (p, _) in placed.items())
            fit = charge.fit({t: p for t, (p, _) in placed.items()})
            if fit is not None:
              r["fitM"] = round(math.hypot(fit.x - s.x, fit.y - s.y), 3)
              r["fitDeg"] = round(math.degrees(math.remainder(fit.yaw - s.yaw, 2 * math.pi)), 1)
          rows.append(r)
  finally:
    for det in cams.values():
      det.close()
    body.close()
  print(f"{'out, m':>6s} {'off':>4s}  {'tags read: head, colour':24s}"
        f"{'head: worst tag':>16s} {'the pair fitted':>16s}")
  for dist in REACH_M:
    for off in REACH_DEG:
      rs = [r for r in rows if r["m"] == dist and r["offDeg"] == off]
      read = "  ".join(f"{len(r['head'])},{len(r['colour'])}" for r in rs)
      tag = ", ".join(f"{r['tagM']:.3f}" for r in rs if "tagM" in r) or "-"
      fit = ", ".join(f"{r['fitM']:.3f} m {r['fitDeg']:+.1f}" for r in rs if "fitM" in r) or "-"
      print(f"{dist:6.2f} {off:4.0f}  {read:24s}{tag:>16s} {fit:>32s}")
  return rows


def hold(seconds: float) -> dict:
  """Plugged in at the seat, held `seconds`: the pack's gain, W, net of its
  draw standing, and the plug held."""
  model, data, body, station = world(at_seat=True)
  why = body.run(body.dock_routine())
  if why != "docked":
    return {"dock": why}
  pack = body.pack(800.0, 1.0)
  draws, held = [], []
  body.step_hooks.append(lambda: (draws.append(pack.power_draw(data)),
                                  held.append(body.charging())))
  t0 = float(data.time)
  body.run(body.dock_hold_routine(seconds))
  dt = float(data.time) - t0
  out = {"dock": why, "seconds": round(dt, 1), "chargerW": round(pack.charge_w, 1),
         "drawW": round(float(np.mean(draws)), 1),
         "netW": round(pack.charge_w - float(np.mean(draws)), 1),
         "heldShare": round(float(np.mean(held)), 4), "falls": body.falls}
  print(json.dumps(out))
  return out


def shove(body, model, data) -> None:
  """Shoved backward off its feet, away from the station: 300 N for 0.3 s."""
  away = -np.array([math.cos(body._station.yaw), math.sin(body._station.yaw), 0.0])
  for _ in range(int(0.3 / model.opt.timestep)):
    data.xfrc_applied[body.root, :3] = away * 300.0
    body.run(body.hold_routine(model.opt.timestep))
  data.xfrc_applied[body.root] = 0.0


def fall() -> dict:
  """Plugged in at the seat, then shoved backward off its feet: whether the
  plug let go at the rope's stop, whether the balancer brought it home, and
  the robot up again."""
  model, data, body, station = world(at_seat=True)
  why = body.run(body.dock_routine())
  if why != "docked":
    return {"dock": why}
  breaks0 = station.breaks
  shove(body, model, data)
  t0 = float(data.time)
  home_at = None
  while float(data.time) - t0 < 30.0:
    body.run(body.hold_routine(0.25))
    if home_at is None and station.flags["parked"]:
      home_at = round(float(data.time) - t0, 1)
  out = {"dock": why, "falls": body.falls, "broke": station.breaks > breaks0,
         "plugHomeS": home_at, "stillPlugged": body.charging(),
         "up": body.mission.legs != "getup"}
  print(json.dumps(out))
  return out


#: The lost scene's camera from a start of its own or a belief set wrong
#: its own way: over the robot wherever it walks, this far off, m, and this
#: far down, deg -- near plumb, or a 2.4 m wall beside the robot hides it.
FOLLOW_DIST, FOLLOW_ELEVATION_DEG = 7.0, -75.0
#: ...and its filmstrip's tile this often, s: a walk and a search from
#: further off run 100-250 s.
FOLLOW_EVERY_S = 8.0


def _follows(lost) -> bool:
  """Whether the lost scene's camera follows the robot: from anywhere but
  its own start and belief (`LOST_SCENE_OFF`), whose fixed view beside the
  station the robot never leaves."""
  return LOST_FROM[lost[0]] is not None or tuple(lost[1]) != LOST_SCENE_OFF


def strip(scene: str = "plug", out: str = "station_spike.png",
          lost=("station", LOST_SCENE_OFF)) -> None:
  """The scene headless, a tile every 1.5 s from beside the station -- or,
  `fall`, every 0.5 s from the viewer's wider camera, from just before the
  shove (its walk in is `plug`'s) until it is up and the plug home -- or,
  `lost`, every
  4 s from further back: the loop's charge path from `lost`'s start (2 m
  out), its belief set `lost`'s offset wrong (`LOST_SCENE_OFF`); given a
  start or an offset of its own, every 8 s from over the robot
  (`_follows`)."""
  from PIL import Image, ImageDraw
  follow = scene == "lost" and _follows(lost)
  if scene == "lost":
    life, body, station = _life()
    model, data = life.model, life.data
    lost_start(body, lost[1], lost[0])
  else:
    model, data, body, station = world()
  r = mujoco.Renderer(model, TILE_H, TILE_W)
  tiles, due = [], [0.0]
  s = body._station
  every, (lx0, ly0, lz0), dist = ((4.0, *VIEW_CAM["lost"]) if scene == "lost"
                                  else (0.5, *VIEW_CAM["fall"]) if scene == "fall"
                                  else (1.5, (-0.25, 0.0, 0.85), 1.4))
  if follow:
    every = FOLLOW_EVERY_S
  # ...the fall's tiles begin once it is plugged in
  filming = [scene != "fall"]

  def hook():
    if not filming[0] or data.time < due[0]:
      return
    due[0] = float(data.time) + every
    cam = mujoco.MjvCamera()
    if follow:
      cam.lookat[:] = (*data.xpos[body.root][:2], 0.5)
      cam.distance, cam.azimuth, cam.elevation = FOLLOW_DIST, 90.0, FOLLOW_ELEVATION_DEG
    else:
      lx, ly = s.to_world(lx0, ly0)
      cam.lookat[:] = (lx, ly, lz0)
      cam.distance, cam.azimuth, cam.elevation = dist, side(s), -12.0
    light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.5, 0.6
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
    f = station.flags
    caption = f"{data.time:.1f} s  plugged in: {bool(f['mated'])}  parked: {f['parked']}"
    if scene == "fall" and body.mission.legs == "getup":
      caption += "  getting up"
    if follow:
      m, deg = _off(body.pose, body.true_pose())
      caption = (f"{data.time:.1f} s  belief {m:.2f} m {deg:+.0f} deg off  "
                 f"plugged in: {bool(f['mated'])}")
    draw = ImageDraw.Draw(img)
    draw.rectangle(draw.textbbox((6, 6), caption), fill=(0, 0, 0))
    draw.text((6, 6), caption, fill=(255, 255, 255))
    tiles.append(img)

  body.step_hooks.append(hook)
  if scene == "lost":
    why = "docked" if life.go_charge() else life.charge_failure
  else:
    why = body.run(body.dock_routine())
  if scene == "fall" and why == "docked":
    filming[0], due[0] = True, float(data.time)
  body.run(body.dock_hold_routine(2.0))
  if scene == "plug" and why == "docked":
    body.run(body.undock_routine())
    body.run(body.hold_routine(3.0))
  elif scene == "fall" and why == "docked":
    shove(body, model, data)
    t0 = float(data.time)
    # ...until it is up and the plug home (`fall`'s 20 s at most), and 2 s on
    while float(data.time) - t0 < 20.0 and not (
        float(data.time) - t0 > 2.0 and body.mission.legs != "getup"
        and station.flags["parked"]):
      body.run(body.hold_routine(0.25))
    body.run(body.hold_routine(2.0))
  if not tiles:
    raise SystemExit(f"{why}: nothing to draw")
  r.close()
  body.close()
  w, h = tiles[0].size
  img = Image.new("RGB", (w * min(ROW, len(tiles)), h * ((len(tiles) + ROW - 1) // ROW)))
  for k, t in enumerate(tiles):
    img.paste(t, (w * (k % ROW), h * (k // ROW)))
  img.save(out)
  print(f"{why}; wrote {out} ({len(tiles)} tiles)")


#: What each viewer scene shows, said in the terminal as it starts.
SCENES = {
  "lost": "the `lost` scene: a humanoid 2 m out whose belief is 0.7 m wrong walks to "
          "where it believes the standoff is, sees no station, steps in by its LIDAR, "
          "finds the station's tags, puts its belief right off them, and walks in again "
          "and plugs itself in.",
  "plug": "the `plug` scene: the humanoid walks in from its standoff, fits the station "
          "off its tags, plugs itself in, charges 3 s and unplugs.",
  "fall": "the `fall` scene: the humanoid plugs itself in as in `plug`, then is shoved "
          "over backward; the plug breaks away, its balancer takes it home, and the "
          "robot gets up.",
}


class _closing:
  """`body` closed as its scene ends, its window closing too: its cameras'
  contexts left to the interpreter's exit raised EGL errors there."""

  def __init__(self, body) -> None:
    self.body = body

  def __enter__(self):
    return self.body

  def __exit__(self, *exc) -> bool:
    self.body.close()
    return False


def view(scene: str, lost=("station", LOST_SCENE_OFF)) -> None:
  """A scene in the viewer at real time, again and again until the window
  closes, each from the standoff: `plug` docks, charges and unplugs; `fall`
  docks and is shoved over plugged in; `lost` is the loop's own charge path
  from `lost`'s start (2 m out), its belief set `lost`'s offset wrong
  (`LOST_SCENE_OFF`) -- given a start or an offset of its own, the camera
  over the robot (`_follows`). (From the seat itself, `fall()`'s start,
  the plug-in took four tries and 73 s: SimNotes.)"""
  import time
  import mujoco.viewer
  follow = scene == "lost" and _follows(lost)
  print((SCENES[scene] if not follow else
         f"the `lost` scene from the {lost[0]}, its belief set {lost[1]} wrong: the loop's "
         "charge path, and how it ends.") + " Close the window to quit.")
  while True:
    life = None
    if scene == "lost":
      life, body, station = _life()
      model, data = life.model, life.data
      lost_start(body, lost[1], lost[0])
    else:
      model, data, body, station = world()
    with mujoco.viewer.launch_passive(model, data) as v, _closing(body):
      if follow:
        v.cam.type, v.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, body.root
        v.cam.distance, v.cam.azimuth = FOLLOW_DIST, 90.0
        v.cam.elevation = FOLLOW_ELEVATION_DEG
      else:
        (lx, ly, lz), v.cam.distance = VIEW_CAM[scene]
        v.cam.lookat[:] = (*body._station.to_world(lx, ly), lz)
        v.cam.azimuth, v.cam.elevation = side(body._station), VIEW_ELEVATION_DEG
      wall = time.perf_counter() - float(data.time)

      def pace():
        if not v.is_running():
          raise SystemExit
        if int(round(data.time / model.opt.timestep)) % SYNC_EVERY == 0:
          v.sync()
          ahead = float(data.time) - (time.perf_counter() - wall)
          if ahead > 0:
            time.sleep(ahead)

      body.step_hooks.append(pace)
      t0 = float(data.time)
      if life is not None:
        why = "docked" if life.go_charge() else life.charge_failure
        for e in body.belief_events:
          print(f"  -> {e['why']}" + (f": found {e['found']} in {e['seconds']} s"
                                       if e["why"] == "searched" else ""))
      else:
        why = body.run(body.dock_routine())
      tries = sum(1 for t in body._charge_trace if "try" in t)
      print(f"  -> {why} in {float(data.time) - t0:.1f} s, {tries} "
            f"{'try' if tries == 1 else 'tries'}")
      if why != "docked":
        body.run(body.hold_routine(3.0))
      elif scene == "fall":
        breaks = station.breaks
        shove(body, model, data)
        body.run(body.hold_routine(20.0))
        print(f"  -> falls {body.falls}; broke away {station.breaks > breaks}; plug home "
              f"{bool(station.flags['parked'])}; up {body.mission.legs != 'getup'}")
      else:
        body.run(body.dock_hold_routine(3.0))
        body.run(body.undock_routine())
        print("  -> " + next((t["undock"] for t in reversed(body._charge_trace)
                              if "undock" in t), "not plugged in"))
        body.run(body.hold_routine(3.0))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--approach", action="store_true")
  ap.add_argument("--n", type=int, default=12)
  ap.add_argument("--parallel", type=int, default=3)
  ap.add_argument("--hold", action="store_true")
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--fall", action="store_true")
  ap.add_argument("--walk-in", action="store_true")
  ap.add_argument("--explore", default=None,
                  help="--walk-in's explores before it walks out, s, comma-separated "
                       "(120,240); --lost's one, after its belief is set wrong (none)")
  ap.add_argument("--seats", action="store_true")
  ap.add_argument("--lost", action="store_true")
  ap.add_argument("--far", action="store_true",
                  help="--lost from its belief 1-5 m and 5-20 deg wrong, from --from")
  ap.add_argument("--from", dest="places", default="near,house,street",
                  help="--far's starts: places of LOST_FROM or the groups near, house, "
                       "street, comma-separated")
  ap.add_argument("--lost-table", default=None, metavar="JSON",
                  help="--far's table again, from the rows its --out wrote")
  ap.add_argument("--reach", action="store_true",
                  help="how far its cameras read the station's tags, and the head's fit")
  ap.add_argument("--lost-from", default="station", choices=tuple(LOST_FROM),
                  help="the lost scene's start (--view lost, --strip lost)")
  ap.add_argument("--lost-off", default=None, metavar="X,Y,DEG",
                  help="the lost scene's belief set wrong, m along the station's +x, m "
                       "across it, deg (0.7,0,0)")
  ap.add_argument("--view", nargs="?", const="plug", choices=("plug", "fall", "lost"))
  ap.add_argument("--strip", nargs="?", const="plug", choices=("plug", "fall", "lost"),
                  help="the scene's filmstrip, station_spike.png (fall: station_spike_fall.png, "
                       "lost: station_spike_lost.png)")
  ap.add_argument("--out", default=None)
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--one-walk-in", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--one-seat", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--one-lost", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one:
    print(json.dumps(trial(tuple(json.loads(args.one)))))
    return 0
  if args.one_walk_in:
    print(json.dumps(walk_in(*json.loads(args.one_walk_in))))
    return 0
  if args.one_seat:
    print(json.dumps(seat_trial(tuple(json.loads(args.one_seat)))))
    return 0
  if args.one_lost:
    t = json.loads(args.one_lost)
    print(json.dumps(lost_trial(tuple(t["off"]), t["from"], t["explore"])))
    return 0
  lost = (args.lost_from, LOST_SCENE_OFF if args.lost_off is None
          else tuple(float(v) for v in args.lost_off.split(",")))
  if args.view:
    view(args.view, lost)
    return 0
  if args.lost_table:
    lost_table(json.loads(Path(args.lost_table).read_text()))
    return 0
  if args.lost and args.far:
    places = [p for name in args.places.split(",") for p in LOST_GROUPS.get(name, (name,))]
    unknown = [p for p in places if p not in LOST_FROM]
    if unknown:
      ap.error(f"--from: no such place {', '.join(unknown)} (of {', '.join(LOST_FROM)})")
    rows = far_losts(places, args.n, args.parallel, float(args.explore or 0.0))
  else:
    rows = (approach(args.n, args.parallel) if args.approach else
            reach() if args.reach else
            seats(args.parallel) if args.seats else
            losts(args.n, args.parallel) if args.lost else
            walk_ins(tuple(float(e) for e in (args.explore or "120,240").split(",")),
                     args.parallel)
            if args.walk_in else
            [hold(args.seconds)] if args.hold else [fall()] if args.fall else None)
  if rows is None:
    strip(args.strip or "plug", {"lost": "station_spike_lost.png",
                                 "fall": "station_spike_fall.png"}.get(args.strip,
                                                                       "station_spike.png"),
          lost)
    return 0
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
