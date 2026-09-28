#!/usr/bin/env python
"""Can the quadruped walk into the unknown? (issue #381, the walking stage)

A fresh robot -- at its start pose, an empty map, one look round -- sent
once to each zone of the home world with legs, and how each walk ends:
arrived, or why it gave up and how far short; how long it took and what it
drew, the walk against the shortest route over the true floor, and what
the planning cost. Each target flies in its own process, so every walk
starts from nothing.

    MUJOCO_GL=egl uv run python scripts/unknown_spike.py --parallel 4
    ... --before              # the planner before: mapped floor only, stand-ins
    ... --to lab,kitchen      # some zones (default: every one)
    ... --patience 60         # the walk's budget, s (default 300)
    ... --again               # then back to the start and there again, on
                              # the map the first walk laid
    ... --unknown-cost 3      # `Navigator.UNKNOWN_COST` for the flight
    ... --out walks.json      # the records; --maps DIR saves each map
    ... --compare A.json B.json

A target is its zone's centre moved to the nearest floor the body can stand
on (`CLEAR_M` from every wall and piece of furniture, read off the compiled
world): a centre inside the workshop's table would measure the stand-in,
not the walk. The true route is the same planner over that true floor, the
body's inflation and all, with nothing left unknown. Every walk ends with
the robot standing still `STILL_S`, and how many cells its map grew by
meanwhile -- the noise `navigator.MAP_GROWTH_CELLS` must sit above.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: How far a target stands from anything it could walk into, m: the
#: planner's inflation (0.35 m) and a margin.
CLEAR_M = 0.45
#: What the true floor counts as in the way: every static geom crossing the
#: LIDAR's plane or the depth layer's band (`legs.body.LOW_MIN_Z`..`LOW_MAX_Z`).
BAND_Z = (0.08, 0.60)
#: How long each walk ends standing still, s: the map's growth then is noise.
STILL_S = 20.0


def true_floor(model, grid):
  """The compiled world's static obstacles on `grid`'s cells, as log-odds:
  +5 in the way, -5 floor. Dynamic bodies and the robots are left out."""
  import mujoco

  from pluggybot.telemetry.protocol import dynamic_flags, robot_body_ids
  G = mujoco.mjtGeom
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  rows, cols = grid.grid.shape
  occ = np.zeros((rows, cols), dtype=bool)
  robots, dyn = robot_body_ids(model), dynamic_flags(model)
  for k in range(model.ngeom):
    b = int(model.geom_bodyid[k])
    t = int(model.geom_type[k])
    if (b in robots or dyn[b] or t == G.mjGEOM_PLANE
        or (model.geom_contype[k] == 0 and model.geom_conaffinity[k] == 0)):
      continue
    s = model.geom_size[k]
    half = {G.mjGEOM_BOX: s, G.mjGEOM_ELLIPSOID: s,
            G.mjGEOM_SPHERE: [s[0]] * 3, G.mjGEOM_CYLINDER: [s[0], s[0], s[1]],
            G.mjGEOM_CAPSULE: [s[0], s[0], s[1] + s[0]]}.get(t)
    if half is None:
      continue
    ext = np.abs(data.geom_xmat[k].reshape(3, 3)) @ np.asarray(half, dtype=float)
    lo, hi = data.geom_xpos[k] - ext, data.geom_xpos[k] + ext
    if hi[2] < BAND_Z[0] or lo[2] > BAND_Z[1]:
      continue
    (ix0, iy0), (ix1, iy1) = grid.world_to_cell(lo[0], lo[1]), grid.world_to_cell(hi[0], hi[1])
    occ[max(iy0, 0):min(iy1, rows - 1) + 1, max(ix0, 0):min(ix1, cols - 1) + 1] = True
  return np.where(occ, 5.0, -5.0)


def targets(truth, grid, zones) -> dict[str, tuple[float, float]]:
  """Each zone's centre, moved to the nearest cell `CLEAR_M` from anything."""
  from scipy.ndimage import distance_transform_edt
  clear = distance_transform_edt(truth < 0) * grid.resolution >= CLEAR_M
  ys, xs = np.nonzero(clear)
  out = {}
  for z in zones:
    cx = (z["min"][0] + z["max"][0]) / 2.0
    cy = (z["min"][1] + z["max"][1]) / 2.0
    ix, iy = grid.world_to_cell(cx, cy)
    k = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
    out[z["name"]] = tuple(round(v, 2) for v in grid.cell_to_world(int(xs[k]), int(ys[k])))
  return out


def true_route_m(truth, grid, start, goal, inflation) -> float | None:
  """The shortest route the body could walk over the true floor, m."""
  from pluggybot.mapping import optimistic as op
  b = op.BLOCK
  cost = op.coarsen(op.map_costs(truth, inflation), b)
  lat = op.lattice_for(cost.shape, grid.resolution * b)
  s, g = grid.world_to_cell(*start), grid.world_to_cell(*goal)
  cells, stand_in = op.route(lat, cost, (s[0] // b, s[1] // b), (g[0] // b, g[1] // b))
  if cells is None or stand_in:
    return None
  pts = np.array(cells, dtype=float)
  return float(np.hypot(*np.diff(pts, axis=0).T).sum() * grid.resolution * b)


def world():
  """The home world with legs, one quadruped, its config and a true floor."""
  import mujoco  # noqa: F401

  from pluggybot.legs import world as lw
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.mapping.occupancy_grid import OccupancyGrid
  cfg = world_config(QUAD_HOME)
  model = lw.home_spec().compile()
  grid = OccupancyGrid(*cfg["grid_bounds"], resolution=0.05)
  return cfg, model, grid, true_floor(model, grid)


def fly_one(name: str, x: float, y: float, patience: float, before: bool,
            maps: str | None, again: bool = False,
            unknown_cost: float | None = None) -> dict:
  """One fresh robot, one walk (and with `again`, back and there again):
  the record."""
  import mujoco

  from pluggybot import navigator as nav
  from pluggybot.legs import body as qb
  cfg, model, grid, truth = world()
  if before:
    qb.QuadMission.OPTIMISTIC = False
    qb.QuadMission.PROGRESS_MAP_GROWTH = False
  if unknown_cost is not None:
    qb.QuadMission.UNKNOWN_COST = unknown_cost
  plans: list[float] = []
  real = nav.Navigator._plan_to

  def timed(self, wx, wy):
    t = time.perf_counter()
    try:
      return real(self, wx, wy)
    finally:
      plans.append(time.perf_counter() - t)
  nav.Navigator._plan_to = timed
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=cfg["grid_bounds"])
  m = body.mission
  pack = body.pack(1000.0, 1.0)
  sx, sy, syaw = cfg["start"]
  m.start_at(sx, sy, syaw)
  trail, walked, wh = [], [0.0], [0.0]
  last = [m.true_pose()[:2]]

  def watch():
    here = m.true_pose()[:2]
    walked[0] += math.dist(here, last[0])
    last[0] = here
    wh[0] += pack.power_draw(data) * model.opt.timestep / 3600.0
    if m.step_count % 250 == 0:
      trail.append((round(here[0], 2), round(here[1], 2)))
  m.step_hooks.append(watch)
  body.run(body.look_around_routine())
  walked[0], wh[0] = 0.0, 0.0
  t0, w0 = float(data.time), time.perf_counter()
  arrived = body.run(body.go_to_routine(x, y, timeout=patience))
  wall = time.perf_counter() - w0
  tx, ty, _ = m.true_pose()
  rec = dict(m.last_drive or {})
  rec.pop("goal", None)
  true_m = true_route_m(truth, grid, (sx, sy), (x, y), m.INFLATION_CELLS)
  first = {"walkedM": round(walked[0], 1), "wh": round(wh[0], 3),
           "seconds": round(float(data.time) - t0, 1)}
  second = None
  if again:
    body.run(body.go_to_routine(sx, sy, timeout=patience))
    walked[0], wh[0] = 0.0, 0.0
    t1 = float(data.time)
    home = body.run(body.go_to_routine(x, y, timeout=patience))
    second = {"arrived": bool(home), "why": (m.last_drive or {}).get("why", ""),
              "seconds": round(float(data.time) - t1, 1),
              "walkedM": round(walked[0], 1), "wh": round(wh[0], 3)}
  known = m._known_cells()
  body.run(body.hold_routine(STILL_S))
  still = m._known_cells() - known
  out = {"target": name, "goal": [x, y], "before": before, "patience": patience,
         "unknownCost": qb.QuadMission.UNKNOWN_COST,
         "arrived": bool(arrived), "why": rec.get("why", ""),
         "seconds": first["seconds"],
         "trueShortM": round(math.hypot(x - tx, y - ty), 2),
         "end": [round(tx, 2), round(ty, 2)],
         "walkedM": first["walkedM"], "trueRouteM": None if true_m is None else round(true_m, 1),
         "wh": first["wh"], "falls": m.falls, "again": second,
         "stillGrowth": int(still),
         "plans": len(plans), "planMsMedian": round(1000 * float(np.median(plans)), 1) if plans else 0.0,
         "planMsMax": round(1000 * max(plans, default=0.0), 1),
         "planMsSum": round(1000 * sum(plans), 1), "wallS": round(wall, 1),
         "trail": trail}
  if maps:
    map_png(m, truth, trail, (x, y), Path(maps) / f"{name}{'.before' if before else ''}.png")
  body.close()
  return out


def map_png(m, truth, trail, goal, path: Path, scale: int = 1) -> None:
  """The robot's map at the walk's end: free white, wall black, unseen grey,
  the true walls faint where it never saw them, the trail red, the goal
  blue."""
  from PIL import Image
  g = m.grid.grid
  img = np.full(g.shape + (3,), 150, dtype=np.uint8)
  img[truth > 0] = (120, 120, 120)
  img[g < -0.5] = 255
  img[g > 0.5] = 0
  for px, py in trail:
    cx, cy = m.grid.world_to_cell(px, py)
    img[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = (220, 30, 30)
  cx, cy = m.grid.world_to_cell(*goal)
  img[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4] = (30, 60, 220)
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(img[::-1]).resize((g.shape[1] * scale, g.shape[0] * scale)).save(path)


def report(walks: list[dict]) -> None:
  print(f"{'target':16s} {'arrived':8s} {'why':9s} {'s':>6s} {'short':>6s} "
        f"{'walked':>7s} {'route':>6s} {'Wh':>6s} {'plans':>6s} {'ms med':>7s} {'ms max':>7s}"
        f" {'still+':>6s}  again")
  for w in walks:
    a = w.get("again")
    print(f"{w['target']:16s} {str(w['arrived']):8s} {w['why'] or '-':9s} {w['seconds']:6.1f} "
          f"{w['trueShortM']:6.2f} {w['walkedM']:7.1f} {str(w['trueRouteM']):>6s} {w['wh']:6.2f} "
          f"{w['plans']:6d} {w['planMsMedian']:7.1f} {w['planMsMax']:7.1f}"
          f" {w.get('stillGrowth', 0):6d}  "
          + ("" if not a else f"{'arrived' if a['arrived'] else a['why']} {a['seconds']:.0f} s "
                              f"{a['walkedM']:.1f} m"))
  n = sum(w["arrived"] for w in walks)
  print(f"arrived {n} of {len(walks)}")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--to", default=None, help="zones, comma separated (default all)")
  ap.add_argument("--patience", type=float, default=300.0)
  ap.add_argument("--before", action="store_true",
                  help="the planner before: mapped floor only, stand-ins")
  ap.add_argument("--again", action="store_true",
                  help="then back to the start, and there again on the map it laid")
  ap.add_argument("--unknown-cost", type=float, default=None)
  ap.add_argument("--parallel", type=int, default=1)
  ap.add_argument("--out", default=None)
  ap.add_argument("--maps", default=None, metavar="DIR")
  ap.add_argument("--compare", nargs="+", default=None, metavar="JSON")
  ap.add_argument("--one", nargs=3, default=None, metavar=("NAME", "X", "Y"),
                  help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.compare:
    for path in args.compare:
      print(f"== {path}")
      report(json.loads(Path(path).read_text()))
    return 0
  if args.one:
    name, x, y = args.one
    print(json.dumps(fly_one(name, float(x), float(y), args.patience, args.before,
                             args.maps, args.again, args.unknown_cost)))
    return 0
  cfg, _, grid, truth = world()
  places = targets(truth, grid, cfg["zones"])
  if args.to:
    places = {k: places[k] for k in args.to.split(",")}

  def fly(item):
    name, (x, y) = item
    cmd = [sys.executable, __file__, "--one", name, str(x), str(y),
           "--patience", str(args.patience)]
    cmd += ["--before"] if args.before else []
    cmd += ["--again"] if args.again else []
    cmd += [] if args.unknown_cost is None else ["--unknown-cost", str(args.unknown_cost)]
    cmd += ["--maps", args.maps] if args.maps else []
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not line:
      print(f"{name}: no record\n{out.stderr[-2000:]}", file=sys.stderr)
      return None
    return json.loads(line[-1])
  with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
    walks = [w for w in pool.map(fly, places.items()) if w is not None]
  report(walks)
  if args.out:
    Path(args.out).write_text(json.dumps(walks))
  return 0


if __name__ == "__main__":
  sys.exit(main())
