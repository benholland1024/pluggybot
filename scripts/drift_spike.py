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
  ap.add_argument("--trips", type=int, default=3)
  ap.add_argument("--out", default=None)
  ap.add_argument("--compare", nargs="+", default=None, metavar="JSON")
  args = ap.parse_args(argv)
  if args.compare:
    for path in args.compare:
      print(f"== {path}")
      report(json.loads(Path(path).read_text()))
    return 0
  out = {"body": "quadruped", "trips": args.trips, "trace": [], "doors": []}
  fly_quadruped(args.trips, out)
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
