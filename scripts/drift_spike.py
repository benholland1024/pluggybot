#!/usr/bin/env python
"""Does the map stay true under drift? (issue #386)

Believed pose against true pose over lab round trips -- the lab 25 m from
the living room, down the street -- and whether the lab's door is still open
in the robot's own map after them. The sim knows the truth, so the drift is
read off it as it happens; nothing here changes the walk.

    MUJOCO_GL=egl uv run python scripts/drift_spike.py --trips 3
    ... --out drift.json    # the trace, for --compare
    ... --compare A.json B.json

The walking policy along a route to the lab and back, steered by the TRUE
pose -- what is measured is the robot's estimate, so its steering must not
depend on it -- in the served house (its dock, rack and signs), with two
estimates riding one walk: legged odometry alone, and corrected by scan
matching against the robot's own map off the LIDAR.

Reports per leg the worst and the last position error and the heading
error; after each trip whether the lab door is open in the robot's own map
(a plan from where it believed the lobby was to where it believed the lab
was); the matcher's answers and cost. With `--out`, each map is saved
beside the trace, and a picture of it round the lobby and the lab.

A LONG EXPLORE (issue #422), steered by the BELIEF this time -- the loop's
own explore, then its walk home and the dock -- for one IMU seed a process:

    MUJOCO_GL=egl uv run python scripts/drift_spike.py --explore 480 --seed 3
    ... --rest 120          # lie first: the gyro's offset learned (#425)
    ... --out run3.json     # then --explore-table run*.json

Reports belief against truth every 10 s with the zone the robot is truly
in, the matcher's verdicts, the error the explore ended with, and whether
`go_charge` docked, or why not.

LOST ON ITS OWN MAP (issue #476), three measurements off the commissioned
fixtures and the deployed robots' own saved state (a `world.npz`, read
only):

    ... --looks                     # one look at the dock's board or the
                                    # rack, from round each and at the
                                    # bays: where it puts the robot, against
                                    # the truth
    ... --lived world.npz           # LAB_WALK steered by the truth, matched
                                    # estimates starting AT THE TRUTH on the
                                    # true floor, an empty map, and each
                                    # robot's own map out of the save
    ... --lost world.npz --robot r2_pluggybot [--fetch module_pen]
                                    # the served body given that robot's own
                                    # state, standing where it truly was,
                                    # sent to charge (or to fetch a tool)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Where the belief is compared with the truth, sim s.
EVERY_S = 0.5


def wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


# ---- the lab door in a map ----------------------------------------------------


#: The places either side of the lab's door (the lobby's east wall, x = 22,
#: y 2.5-3.5), on the route through it.
LOBBY_XY, LAB_XY = (20.5, 3.0), (23.5, 3.55)


class Places:
  """Where the robot BELIEVED it was the first time it truly stood at
  LOBBY_XY and at LAB_XY: the lab door is asked about in the robot's own
  map, in its own frame -- a map that drifted 2 m has its door 2 m off, and
  read at the true coordinates it says nothing."""

  def __init__(self):
    self.at: dict[str, tuple[float, float]] = {}

  def see(self, true_xy, believed_xy) -> None:
    for name, xy in (("lobby", LOBBY_XY), ("lab", LAB_XY)):
      if name not in self.at and math.dist(true_xy, xy) < 0.3:
        self.at[name] = (float(believed_xy[0]), float(believed_xy[1]))


def door_state(grid, places: Places) -> dict:
  """The lab's door as the robot's own planner has it: whether it can plan
  from where it believed the lobby was to where it believed the lab was,
  over its own map. (Whether the WALL round that door is still there is
  the map's picture, `--png`: a map smeared by drift rubs walls out as well
  as doubling them, and a door in a wall that is gone plans open.)"""
  from pluggybot.mapping.astar import astar, nearest_traversable
  from pluggybot.mapping.frontier import traversable_mask
  if not {"lobby", "lab"} <= set(places.at):
    return {"open": None, "mapped": False}
  trav = traversable_mask(grid.grid)
  a = nearest_traversable(trav, grid.world_to_cell(*places.at["lobby"]))
  b = nearest_traversable(trav, grid.world_to_cell(*places.at["lab"]))
  path = astar(trav, a, b) if a is not None and b is not None else None
  return {"open": bool(path), "mapped": True}


def map_png(grid, path: Path, box=(14.0, -3.0, 28.0, 9.0), scale: int = 3) -> None:
  """The map round the lobby and the lab as a PNG (walls black, free white,
  unseen grey), north up."""
  from PIL import Image
  ax, ay = grid.world_to_cell(box[0], box[1])
  bx, by = grid.world_to_cell(box[2], box[3])
  crop = grid.grid[ay:by, ax:bx]
  img = np.full(crop.shape, 127, np.uint8)
  img[crop < -0.5] = 255
  img[crop > 0.5] = 0
  Image.fromarray(img[::-1]).resize((img.shape[1] * scale, img.shape[0] * scale),
                                    Image.NEAREST).save(path)


# ---- the quadruped ------------------------------------------------------------

#: A way to the lab from the living room, thinned to 0.75 m: known to
#: clear every doorway, so the walk tests the estimate and never the route.
LAB_WALK = ((1.50, 0.50), (2.26, 0.45), (2.85, -0.01), (3.60, 0.06), (4.11, 0.62),
            (4.84, 0.89), (5.58, 1.13), (6.17, 1.62), (6.13, 2.37), (6.13, 3.12),
            (6.91, 3.12), (7.66, 3.12), (8.41, 3.12), (9.17, 3.12), (9.92, 3.12),
            (10.68, 3.12), (11.43, 3.12), (12.19, 3.12), (12.94, 3.12), (13.69, 3.12),
            (14.45, 3.12), (15.20, 3.12), (15.95, 3.12), (16.74, 3.12), (17.51, 3.12),
            (18.28, 3.12), (19.03, 3.04), (19.80, 3.03), (20.56, 3.02), (21.33, 3.02),
            (22.07, 3.21), (22.70, 3.62), (23.45, 3.58), (24.22, 3.57), (24.36, 3.57))
#: How the walk is steered (by the truth): pure pursuit at this speed and
#: lookahead, turning on the spot past this heading error.
WALK_V, LOOKAHEAD_M, TURN_ON_SPOT_RAD, TURN_RATE = 0.5, 0.4, math.radians(50), 0.6


def quad_home_world(x: float, y: float, yaw: float):
  """The served house (`legs.world.home_spec`: its dock, rack and signs)
  with one quadruped standing at (x, y) facing `yaw`; (model, data)."""
  import mujoco

  from pluggybot.legs import world as lw
  model = lw.home_spec(first_at=(x, y)).compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, x=x, y=y, yaw=yaw)
  return model, data


class Estimate:
  """One way of knowing where the body is: legged odometry, corrected by
  scan matching or not, and the map it lays the scans through."""

  def __init__(self, model, data, grid_bounds, origin, match: bool, seed: int = 0):
    from pluggybot.legs.odometry import LegOdometry
    from pluggybot.mapping.occupancy_grid import OccupancyGrid
    from pluggybot.mapping.scan_match import ScanMatcher
    self.odo = LegOdometry(model, data, seed=seed)
    self.grid = OccupancyGrid(*grid_bounds, resolution=0.05)
    self.origin = origin
    self.matcher = ScanMatcher(self.grid, origin=origin) if match else None
    self.ms = []
    self.places = Places()

  def scan(self, angles, ranges, max_range) -> None:
    from pluggybot.legs.body import QuadMission
    if self.odo.att.tilt() > QuadMission.LEVEL_TILT:
      return
    pose = (self.odo.x, self.odo.y, self.odo.yaw)
    if self.matcher is not None:
      t = time.perf_counter()
      m = self.matcher.match(pose, angles, ranges)
      self.ms.append(time.perf_counter() - t)
      if m.accepted:
        self.odo.correct(*m.pose)
        pose = m.pose
      if not self.matcher.fuses(m, self.odo.d.time):
        return
    self.grid.update(pose, angles, ranges, max_range, origin=self.origin)
    if self.matcher is not None:
      self.matcher.fused(pose, self.odo.d.time)


def fly_quadruped(trips: int, out: dict, seed: int = 0) -> None:
  """The walk out along LAB_WALK and back, `trips` times, steered by the
  truth; two estimates ride it -- legged odometry alone, and corrected by
  scan matching -- off the same IMU draws, so the only difference between
  them is the matcher."""
  from pluggybot.legs.policy import POLICY_NPZ, PolicyDriver, Twist, WalkingPolicy
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.perception.lidar import LIDAR_PERIOD, Lidar
  from pluggybot.legs.actuator import BUS_V_NOMINAL

  x0, y0 = LAB_WALK[0]
  yaw0 = math.atan2(LAB_WALK[1][1] - y0, LAB_WALK[1][0] - x0)
  model, data = quad_home_world(x0, y0, yaw0)
  drv = PolicyDriver(model, data, WalkingPolicy(POLICY_NPZ))
  drv.set_bus(BUS_V_NOMINAL)
  lidar = Lidar(model, site_name="lidar", robot_body="pluggybot")
  root = model.body("pluggybot").id
  # the LIDAR in the torso's frame, level: the pose point is the torso's origin
  rot = data.xmat[root].reshape(3, 3)
  rel = rot.T @ (data.site_xpos[lidar.site_id] - data.xpos[root])
  origin = (float(rel[0]), float(rel[1]))
  bounds = world_config(QUAD_HOME)["grid_bounds"]
  ests = {"odometry": Estimate(model, data, bounds, origin, match=False, seed=seed),
          "matched": Estimate(model, data, bounds, origin, match=True, seed=seed)}
  route = list(LAB_WALK)
  plan = []
  for _ in range(trips):
    plan += [("out", route), ("back", route[::-1])]
  next_scan, next_row, fell, stuck = 0.0, 0.0, False, False
  t0 = time.time()
  for leg, path in plan:
    i, leg_t0 = 0, float(data.time)
    budget = sum(math.dist(a, b) for a, b in zip(path, path[1:])) / WALK_V * 2.0 + 30.0
    while i < len(path) - 1 and not fell:
      if data.time - leg_t0 > budget:
        stuck = True
        print(f"t={data.time:7.1f} the walk {leg} is stuck at waypoint {i} of {len(path)}, "
              f"({float(data.xpos[root][0]):.2f}, {float(data.xpos[root][1]):.2f})", flush=True)
        break
      tx, ty = float(data.xpos[root][0]), float(data.xpos[root][1])
      r = data.xmat[root].reshape(3, 3)
      tyaw = math.atan2(r[1, 0], r[0, 0])
      # the lookahead point: the first waypoint past LOOKAHEAD_M
      while i < len(path) - 1 and math.hypot(path[i][0] - tx, path[i][1] - ty) < LOOKAHEAD_M:
        i += 1
      gx, gy = path[i]
      err = wrap(math.atan2(gy - ty, gx - tx) - tyaw)
      if abs(err) > TURN_ON_SPOT_RAD:
        twist = Twist(yaw_rate=math.copysign(TURN_RATE, err))
      else:
        twist = Twist(vx=WALK_V, yaw_rate=max(-1.0, min(1.0, 2.0 * err)))
      drv.step(twist)
      for e in ests.values():
        e.odo.step()
      if data.time >= next_scan:
        next_scan = data.time + LIDAR_PERIOD
        angles, ranges, _, _ = lidar.scan_split(data)
        for e in ests.values():
          e.scan(angles, ranges, lidar.max_range)
      if data.time >= next_row:
        next_row = (math.floor(data.time / EVERY_S) + 1) * EVERY_S
        row = {"t": round(float(data.time), 2), "leg": leg, "true": [round(tx, 3), round(ty, 3)]}
        for name, e in ests.items():
          row[name] = {"err": round(math.hypot(e.odo.x - tx, e.odo.y - ty), 4),
                       "dyaw": round(math.degrees(wrap(e.odo.yaw - tyaw)), 3)}
          e.places.see((tx, ty), (e.odo.x, e.odo.y))
        out["trace"].append(row)
      fell = float(data.xpos[root][2]) < 0.15
      if int(data.time / 0.002) % 5000 == 0:
        print(f"t={data.time:7.1f} {leg} waypoint {i}/{len(path)} at ({tx:.2f}, {ty:.2f}), "
              f"{time.time() - t0:.0f} s wall", flush=True)
    if stuck or fell:
      break
    if leg == "out":
      for name, e in ests.items():
        out["doors"].append({"t": round(float(data.time), 1), "after": f"{leg} ({name})",
                             **door_state(e.grid, e.places)})
  for name, e in ests.items():
    out["doors"].append({"t": round(float(data.time), 1), "after": f"day ({name})",
                         **door_state(e.grid, e.places)})
  m = ests["matched"]
  out["maps"] = {name: (e.grid, e.places) for name, e in ests.items()}
  out["summary"] = {"simS": round(float(data.time), 1), "wallS": round(time.time() - t0, 1),
                    "fell": fell, "stuck": stuck, "matcher": dict(m.matcher.counts),
                    "msPerMatch": round(1e3 * float(np.mean(m.ms)), 3) if m.ms else None,
                    "matches": len(m.ms),
                    "walkedM": round(ests["odometry"].odo.distance, 1)}


# ---- a long explore (issue #422) --------------------------------------------------

#: How often the explore's belief is compared with the truth, sim s.
EXPLORE_EVERY_S = 10.0


def zone_of(x: float, y: float) -> str:
  from pluggybot.home import world as home
  for z in home.ZONES:
    (x0, y0), (x1, y1) = z["min"], z["max"]
    if x0 <= x <= x1 and y0 <= y <= y1:
      return z["name"]
  return "?"


def fly_explore(budget: float, seed: int, rest: float, start, charge: bool) -> dict:
  """One quadruped in the served house explores for `budget` sim s from
  `start` (the commissioned one if None) -- after lying `rest` s, if any --
  with its IMU and encoders drawn off `seed`; then, if `charge`, walks home
  and docks. Belief against truth throughout."""
  import mujoco

  from pluggybot.legs.odometry import LegOdometry
  from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, points_ledger, world_config
  from pluggybot.robot import world_spec

  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=40.0, rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"], low_battery_wh=cfg["low_battery_wh"],
                      ledger=points_ledger(None), world=QUAD_HOME, errands=[], near_field=True)
  life.max_sim_time = life.explore_deadline = 1e9
  life.blacklist, life.floor_explored = set(), False
  start = tuple(cfg["start"]) if start is None else start
  life.body.start_at(*start)
  m = life.body.mission
  m.odo = LegOdometry(model, data, seed=seed, prefix=m.handle.prefix)
  m.odo.correct(*start)
  out = {"seed": seed, "start": list(start), "budget": budget, "rest": rest,
         "gyroBiasZ": math.degrees(float(m.odo.imu.gyro_bias[2])), "rows": []}
  due = [0.0]

  def row() -> dict:
    b, t = m.pose, m.true_pose()
    return {"t": round(float(data.time), 1), "err": round(math.hypot(b[0] - t[0], b[1] - t[1]), 3),
            "dyaw": round(math.degrees(wrap(b[2] - t[2])), 2), "zone": zone_of(t[0], t[1]),
            "true": [round(t[0], 2), round(t[1], 2)], "counts": dict(m.matcher.counts)}

  def hook() -> None:
    if data.time >= due[0]:
      due[0] = float(data.time) + EXPLORE_EVERY_S
      out["rows"].append(row())

  m.step_hooks.append(hook)
  t0 = time.time()
  if rest > 0:
    life.body.run(life.body.rest_routine())
    life.body.run(life.body.hold_routine(rest))
  life.body.run(life.body.look_around_routine())
  out["ended"] = life.explore(budget=budget, mark_done=False)
  out["after"] = row()
  if charge:
    out["docked"] = bool(life.go_charge())
    out["failure"] = life.charge_failure
    out["trace"] = life.body.charge_trace()
    out["home"] = row()
  out["wallS"] = round(time.time() - t0, 1)
  life.body.close()
  return out


def explore_table(runs: list[dict]) -> None:
  """One line a run: the seed, its gyro's offset, the error the explore
  ended with and the worst along the explore, the dock, and the verdicts
  after the walk home."""
  print(f"{'seed':>4s} {'rest':>4s} {'offset':>8s} {'ended':>6s} {'worst':>6s} {'where worst':16s}"
        f" docked  verdicts after the walk home")
  for r in runs:
    worst = max((q for q in r["rows"] if q["t"] <= r["after"]["t"]), key=lambda q: q["err"])
    home = r.get("home", r["after"])
    docked = r.get("docked")
    print(f"{r['seed']:4d} {r['rest']:4.0f} {r['gyroBiasZ']:+8.4f} {r['after']['err']:6.2f}"
          f" {worst['err']:6.2f} {worst['zone']:16s} {'-' if docked is None else str(docked):6s}"
          f"  {home['counts']}" + ("" if docked in (None, True) else f"  ({r['failure']})"))


# ---- lost on its own map (issue #476) ----------------------------------------------


def saved_robots(path) -> dict:
  """Each robot in a saved world (`continuation.write`'s file): its root ->
  `{"true": (x, y, yaw), "belief": (x, y, yaw), "state": its body's kept
  state, "arrays": its arrays, bare-named}`. True poses by joint NAME, the
  belief off the reckoning's own quaternion."""
  z = np.load(path)
  meta = json.loads(str(z["meta"]))
  adr, at = 0, {}
  for key, n, _ in meta["physics"]["joints"]:
    at[key] = (adr, n)
    adr += n
  out = {}
  for root, rec in meta["robots"].items():
    mission = rec["mission"]
    key = min((k for k in at if k.startswith(root) and at[k][1] == 7), key=len)
    a = at[key][0]
    x, y, _, qw, qx, qy, qz = (float(v) for v in z["qpos"][a:a + 7])
    w, bx, by, bz = mission["odometry"]["quat"]
    out[root] = {"true": (x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))),
                 "belief": (mission["odometry"]["x"], mission["odometry"]["y"],
                            math.atan2(2 * (w * bz + bx * by), 1 - 2 * (by * by + bz * bz))),
                 "state": mission, "t": float(meta["physics"]["t"]),
                 "arrays": {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(root + "/")}}
  return out


def fly_looks() -> list[dict]:
  """One decode at each of 392 poses round each commissioned fixture --
  0.8-4 m from its tags, 50 deg either side of its face, the robot facing it
  to 30 deg either way -- and 45 at each of the rack's six bays' working
  poses, over the line-up's spread (+-15 mm across, +-30 mm along, +-4 deg);
  nothing stepped: what each fixture's fit puts the robot at, against the
  truth, with how far its tags were and how far off the camera's axis."""
  import mujoco

  from pluggybot.legs import dock as dk
  from pluggybot.legs import rack as rk
  from pluggybot.legs import world as lw
  from pluggybot.legs.body import NAV_EYE, QuadMission
  from pluggybot.legs.fixtures import FIXTURE_TAGS
  from pluggybot.lifecycle import QUAD_HOME, world_config
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  m = QuadMission(model, data, realtime=False, grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  m.start_at(1.5, 0.5, 0.0)
  faces = {"dock": (*m.fixture_face("dock"), m.dock_prior[2] + math.pi),
           "rack": (*m.fixture_face("rack"), m.tool_rack_prior[2])}
  poses = []
  for fx_, fy, fyaw in faces.values():
    for r in (0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0):
      for b in (-50, -30, -15, 0, 15, 30, 50):
        for j in (-30, -22, -15, 0, 15, 22, 30):
          a = fyaw + math.radians(b)
          x, y = fx_ + r * math.cos(a), fy + r * math.sin(a)
          poses.append(("round", x, y, math.atan2(fy - y, fx_ - x) + math.radians(j)))
  for spec in rk.SPECS:
    for st in spec.stations:
      wx, wy, wyaw = m.work_pose(st)
      c, s_ = math.cos(wyaw), math.sin(wyaw)
      for da in (-0.015, 0.0, 0.015):
        for dx in (-0.03, 0.0, 0.03):
          for dth in (-4, -2, 0, 2, 4):
            poses.append(("bay", wx + c * dx - s_ * da, wy + s_ * dx + c * da,
                          wyaw + math.radians(dth)))
  rows = []
  for where, x, y, yaw in poses:
    lw.stand(model, data, "", x, y, yaw)
    seen = dk.seen_from(model, data, m.detect_board(), m.handle.el(NAV_EYE), m.root)
    tx, ty, tyaw = m.true_pose()
    for fixture in ("dock", "rack"):
      got = m.fixture_fit(fixture, seen)
      if got is None:
        continue
      fit, range_m = got
      wx, wy, wyaw = m.fixture_pose(fixture, fit)
      tags = [i for i in seen if i in FIXTURE_TAGS[fixture]]
      rows.append({"at": where, "fixture": fixture, "range": round(range_m, 2), "tags": fit.n,
                   "offAxis": round(float(np.mean([abs(math.degrees(math.atan2(
                     seen[i][1], seen[i][0]))) for i in tags])), 1),
                   "err": round(math.hypot(wx - tx, wy - ty), 4),
                   "dyaw": round(math.degrees(wrap(wyaw - tyaw)), 3)})
  m.close()
  return rows


def looks_table(rows: list[dict]) -> None:
  bay = [r for r in rows if r["at"] == "bay" and r["fixture"] == "rack"]
  e = np.array([r["err"] for r in bay])
  d = np.abs([r["dyaw"] for r in bay])
  print(f"== the rack from its bays' working poses: {len(bay)} looks, {min(r['tags'] for r in bay)}-"
        f"{max(r['tags'] for r in bay)} tags at {min(r['range'] for r in bay):.2f}-"
        f"{max(r['range'] for r in bay):.2f} m, position worst {100 * e.max():.1f} cm, "
        f"heading worst {d.max():.2f} deg")
  for fixture in ("dock", "rack"):
    for tags in (3, 4):
      print(f"== {fixture} from round it, {tags} tags or more: one look against the truth")
      for lo, hi in ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)):
        rs = [r for r in rows if r["at"] == "round" and r["fixture"] == fixture
              and r["tags"] >= tags and lo <= r["range"] < hi]
        if not rs:
          continue
        e = np.array([r["err"] for r in rs])
        d = np.abs([r["dyaw"] for r in rs])
        print(f"  {lo:.1f}-{hi:.1f} m: {len(rs):4d} looks, position worst {100 * e.max():6.1f} cm "
              f"(p90 {100 * np.percentile(e, 90):5.1f}), heading worst {d.max():5.2f} deg "
              f"(p90 {np.percentile(d, 90):4.2f})")


def fly_lived(save: str, trips: int = 1, seed: int = 0) -> dict:
  """LAB_WALK out and back `trips` times, steered by the truth, with
  matched estimates off one IMU draw, each starting AT THE TRUTH -- as a
  fixture would put it -- on the true floor, an empty map, and each robot's
  own map out of `save`: does a belief put right stay right on the map the
  robot laid?"""
  from unknown_spike import true_floor

  from pluggybot.legs.actuator import BUS_V_NOMINAL
  from pluggybot.legs.policy import POLICY_NPZ, PolicyDriver, Twist, WalkingPolicy
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.perception.lidar import LIDAR_PERIOD, Lidar
  saved = saved_robots(save)
  x0, y0 = LAB_WALK[0]
  model, data = quad_home_world(x0, y0, math.atan2(LAB_WALK[1][1] - y0, LAB_WALK[1][0] - x0))
  drv = PolicyDriver(model, data, WalkingPolicy(POLICY_NPZ))
  drv.set_bus(BUS_V_NOMINAL)
  lidar = Lidar(model, site_name="lidar", robot_body="pluggybot")
  root = model.body("pluggybot").id
  rel = data.xmat[root].reshape(3, 3).T @ (data.site_xpos[lidar.site_id] - data.xpos[root])
  origin = (float(rel[0]), float(rel[1]))
  bounds = world_config(QUAD_HOME)["grid_bounds"]
  ests = {}
  for name in ["true floor", "empty map"] + [f"{r}'s own map" for r in saved]:
    e = Estimate(model, data, bounds, origin, match=True, seed=seed)
    if name == "true floor":
      e.grid.grid[:] = true_floor(model, e.grid)
    elif name != "empty map":
      e.grid.grid[:] = saved[name.split("'", 1)[0]]["arrays"]["grid"]
    tx, ty = float(data.xpos[root][0]), float(data.xpos[root][1])
    r_ = data.xmat[root].reshape(3, 3)
    e.odo.correct(tx, ty, math.atan2(r_[1, 0], r_[0, 0]))
    ests[name] = e
  trace = []
  next_scan = next_row = 0.0
  for path in [p for _ in range(trips) for p in (LAB_WALK, LAB_WALK[::-1])]:
    i, leg_t0 = 0, float(data.time)
    budget = sum(math.dist(a, b) for a, b in zip(path, path[1:])) / WALK_V * 2.0 + 30.0
    while i < len(path) - 1 and data.time - leg_t0 < budget:
      tx, ty = float(data.xpos[root][0]), float(data.xpos[root][1])
      r_ = data.xmat[root].reshape(3, 3)
      tyaw = math.atan2(r_[1, 0], r_[0, 0])
      while i < len(path) - 1 and math.hypot(path[i][0] - tx, path[i][1] - ty) < LOOKAHEAD_M:
        i += 1
      err = wrap(math.atan2(path[i][1] - ty, path[i][0] - tx) - tyaw)
      drv.step(Twist(yaw_rate=math.copysign(TURN_RATE, err)) if abs(err) > TURN_ON_SPOT_RAD
               else Twist(vx=WALK_V, yaw_rate=max(-1.0, min(1.0, 2.0 * err))))
      for e in ests.values():
        e.odo.step()
      if data.time >= next_scan:
        next_scan = data.time + LIDAR_PERIOD
        angles, ranges, _, _ = lidar.scan_split(data)
        for e in ests.values():
          e.scan(angles, ranges, lidar.max_range)
      if data.time >= next_row:
        next_row = data.time + 1.0
        trace.append({"t": round(float(data.time), 1), **{
          n: [round(math.hypot(e.odo.x - tx, e.odo.y - ty), 3),
              round(math.degrees(wrap(e.odo.yaw - tyaw)), 2)] for n, e in ests.items()}})
  return {"trace": trace, "counts": {n: dict(e.matcher.counts) for n, e in ests.items()},
          "walkedM": round(ests["empty map"].odo.distance, 1)}


def lived_table(out: dict) -> None:
  print(f"walked {out['walkedM']} m from the truth, steered by it:")
  for name, counts in out["counts"].items():
    errs = [row[name][0] for row in out["trace"]]
    at20 = next((row[name][0] for row in out["trace"] if row["t"] >= 20.0), errs[-1])
    print(f"  {name:28s} worst {max(errs):5.2f} m, 20 s in {at20:5.2f} m, end {errs[-1]:5.2f} m, "
          f"worst heading {max(abs(row[name][1]) for row in out['trace']):5.1f} deg  {counts}")


def fly_lost(save: str, robot: str, fetch: str | None = None) -> dict:
  """The served body given `robot`'s own state out of `save` -- its map,
  belief, matcher and places -- standing where it truly was (the save's
  clocks with it), then the loop's own `go_charge`, or a fetch of `fetch`
  (every tool hangs on its bay here). Belief against truth every 5 s, what
  moved or lost the belief, and how it ended."""
  import mujoco

  from pluggybot.legs import world as lw
  from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, points_ledger, world_config
  from pluggybot.robot import world_spec
  rec = saved_robots(save)[robot]
  (tx, ty, tyaw), cfg = rec["true"], world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=200.0, rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"], low_battery_wh=cfg["low_battery_wh"],
                      ledger=points_ledger(None), world=QUAD_HOME, errands=[], near_field=True)
  life.max_sim_time = life.explore_deadline = 1e9
  m = life.body.mission
  data.time = rec["t"]                           # the save's clocks are absolute
  m.start_at(tx, ty, tyaw)
  m.restore_kept(rec["state"], rec["arrays"])
  lw.stand(model, data, "", tx, ty, tyaw)
  m.posture = "standing"
  if fetch:
    m.carry(None)
  out = {"robot": robot, "start": {"true": list(rec["true"]), "belief": list(m.pose)}, "rows": []}
  due = [0.0]

  def row() -> None:
    if data.time >= due[0]:
      due[0] = float(data.time) + 5.0
      b, t = m.pose, m.true_pose()
      out["rows"].append([round(float(data.time) - rec["t"], 1),
                          round(math.hypot(b[0] - t[0], b[1] - t[1]), 3),
                          round(math.degrees(wrap(b[2] - t[2])), 2)])

  m.step_hooks.append(row)
  t0 = time.time()
  if fetch:
    from pluggybot.rack.coupling import STATION_YS
    out["fetched"] = life.body.run(life.body.fetch_tool_routine(
      STATION_YS[life.rack_inventory[fetch]], fetch))
    out["onFork"] = life.body.module_state(fetch)["on_fork"]
    out["trace"] = life.body.swap_trace()
  else:
    out["docked"] = bool(life.go_charge())
    out["failure"] = life.charge_failure
    out["trace"] = life.body.charge_trace()
  out["events"] = list(m.belief_events)
  out["simS"] = round(float(data.time) - rec["t"], 1)
  out["wallS"] = round(time.time() - t0, 1)
  life.body.close()
  return out


# ---- the report ------------------------------------------------------------------


def legs(trace: list[dict]) -> list[dict]:
  """Consecutive samples with the same leg name, in order."""
  out: list[dict] = []
  for row in trace:
    if not out or out[-1]["leg"] != row["leg"]:
      out.append({"leg": row["leg"], "t0": row["t"], "rows": []})
    out[-1]["rows"].append(row)
  return out


def door_words(d: dict) -> str:
  if not d["mapped"]:
    return "not yet reached"
  return ("open: a plan from the lobby to the lab in the robot's own map" if d["open"]
          else "SHUT: no plan from the lobby to the lab in the robot's own map")


def report(out: dict) -> None:
  names = ("odometry", "matched")
  print(f"{'leg':6s} {'t0 s':>7s} {'dur s':>6s} " + " ".join(
    f"{n + ' worst/last m':>22s} {'yaw deg':>7s}" for n in names))
  for leg in legs(out["trace"]):
    rows = leg["rows"]
    cells = []
    for n in names:
      worst = max(r[n]["err"] for r in rows)
      yaw = max(abs(r[n]["dyaw"]) for r in rows)
      cells.append(f"{worst:10.3f} / {rows[-1][n]['err']:9.3f} {yaw:7.2f}")
    print(f"{leg['leg']:6s} {leg['t0']:7.1f} {rows[-1]['t'] - rows[0]['t']:6.1f} " + " ".join(cells))
  for n in names:
    errs = [r[n]["err"] for r in out["trace"]]
    print(f"{n:9s} over the walk: worst {max(errs):.3f} m, median {float(np.median(errs)):.3f} m, "
          f"last {errs[-1]:.3f} m, worst heading {max(abs(r[n]['dyaw']) for r in out['trace']):.2f} deg")
  for d in out["doors"]:
    print(f"lab door after {d['after']:20s} t={d['t']:7.1f}: " + door_words(d))
  print(json.dumps(out["summary"]))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--trips", type=int, default=None, help="out and back: 3, or 1 with --lived")
  ap.add_argument("--out", default=None)
  ap.add_argument("--compare", nargs="+", default=None, metavar="JSON")
  ap.add_argument("--explore", type=float, default=None, metavar="SECONDS",
                  help="a long explore steered by the belief, then home to the dock")
  ap.add_argument("--seed", type=int, default=0, help="the IMU's and encoders' draws")
  ap.add_argument("--rest", type=float, default=0.0, help="lie this long first, s")
  ap.add_argument("--start", default=None,
                  help="x,y,yaw-deg (--start=-3.5,1,0 when x is negative); the commissioned start if unset")
  ap.add_argument("--no-charge", action="store_true", help="stop when the explore does")
  ap.add_argument("--explore-table", nargs="+", default=None, metavar="JSON")
  ap.add_argument("--looks", action="store_true",
                  help="one look at a fixture from round it and at the bays, against the truth")
  ap.add_argument("--lived", default=None, metavar="WORLD_NPZ",
                  help="a walk from the truth on each robot's own map out of a save")
  ap.add_argument("--lost", default=None, metavar="WORLD_NPZ",
                  help="a robot's own state out of a save, sent to charge")
  ap.add_argument("--robot", default="pluggybot", help="whose state, with --lost")
  ap.add_argument("--fetch", default=None, metavar="MODULE", help="with --lost: fetch it instead")
  args = ap.parse_args(argv)
  if args.looks:
    rows = fly_looks()
    looks_table(rows)
    if args.out:
      Path(args.out).write_text(json.dumps(rows))
    return 0
  if args.lived:
    out = fly_lived(args.lived, trips=args.trips or 1, seed=args.seed)
    lived_table(out)
    if args.out:
      Path(args.out).write_text(json.dumps(out))
    return 0
  if args.lost:
    out = fly_lost(args.lost, args.robot, args.fetch)
    b, t = out["start"]["belief"], out["start"]["true"]
    print(f"{args.robot} began {math.hypot(b[0] - t[0], b[1] - t[1]):.2f} m and "
          f"{math.degrees(wrap(b[2] - t[2])):+.1f} deg off")
    for e in out["events"]:
      print(f"  t+{e['t'] - out['events'][0]['t']:6.1f} {e['why']:9s} "
            + " ".join(f"{k}={e[k]}" for k in ("fixture", "moved", "dropped", "found", "ended", "seconds")
                       if k in e))
    print(json.dumps({k: out[k] for k in ("docked", "failure", "fetched", "onFork", "trace", "simS",
                                          "wallS") if k in out}))
    print(f"  ended {out['rows'][-1][1]:.2f} m and {out['rows'][-1][2]:+.1f} deg off")
    if args.out:
      Path(args.out).write_text(json.dumps(out))
    return 0
  if args.explore_table:
    explore_table([json.loads(Path(p).read_text()) for p in args.explore_table])
    return 0
  if args.explore is not None:
    start = None
    if args.start:
      x, y, yaw = (float(v) for v in args.start.split(","))
      start = (x, y, math.radians(yaw))
    out = fly_explore(args.explore, args.seed, args.rest, start, not args.no_charge)
    for r in out["rows"]:
      print(f"t={r['t']:6.1f} {r['zone']:16s} off {r['err']:.2f} m {r['dyaw']:+6.2f} deg")
    explore_table([out])
    if args.out:
      Path(args.out).write_text(json.dumps(out))
    return 0
  if args.compare:
    for path in args.compare:
      print(f"== {path}")
      report(json.loads(Path(path).read_text()))
    return 0
  trips = args.trips or 3
  out = {"body": "quadruped", "trips": trips, "trace": [], "doors": []}
  fly_quadruped(trips, out)
  maps = out.pop("maps")
  report(out)
  if args.out:
    Path(args.out).write_text(json.dumps(out))
    # ...and each map with where the robot believed the door's two sides
    # were, so the door can be asked again without flying (`--compare`)
    np.savez_compressed(Path(args.out).with_suffix(".maps.npz"), **{
      name: grid.grid for name, (grid, _) in maps.items()})
    for name, (grid, _) in maps.items():
      map_png(grid, Path(args.out).with_suffix(f".{name}.png"))
    Path(args.out).with_suffix(".places.json").write_text(json.dumps(
      {name: {"bounds": [grid.x_min, grid.y_min, grid.x_max, grid.y_max, grid.resolution],
              "at": places.at} for name, (grid, places) in maps.items()}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
