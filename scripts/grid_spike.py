#!/usr/bin/env python
"""What a robot's map costs as the world grows (issue #558).

A robot's map is a grid of 5 cm cells over the world's rectangle, and
`occupancy_grid.MAX_CELLS` caps how many there may be. This times the work
that grows with the cells -- at today's size (`home.world.GRID_BOUNDS`,
469,200 cells) and at 2x, 3x and 4x as many, the bounds grown evenly round
the house by whole half metres, so every cell of today's map, and of the
planner's lattice, is where it was -- through the bodies' own code, each
scale in a process of its own:

  per scan, per depth frame and per second: the map's update, the camera's
    layer (`_fold_low`) and the walls it is kept out of (`_walls`), the
    map's growth (`_known_cells`), the grid's image for the wire;
  per plan: a walk's plan on the compiled lattice (`_plan_optimistic`)
    across the whole grid and a 3 m hop, piece by piece, and the lattice;
  per explore leg: the mask and the frontiers, the robot's own component,
    and A* in pure Python across the whole grid (#215's two columns);
  the scan matcher's field (`ScanMatcher.refresh`) and one match;
  on the wire: the grid's image and its PNG (`GridSampler`);
  `world.npz`: the pair's maps written (`continuation.write`);
  memory: each robot's arrays, a plan's peak, the lattice a pair shares.

The scan work, the masks and the plans run on #215's synthetic grid --
known floor with one wall across three quarters of it -- planned corner to
corner. The matcher runs on the house's true floor where it is. The PNG and
the save run on the house as a robot that walked everywhere would map it
(the floor it can reach and the walls round that, nothing else), stacked k
times along y: a world k times the house, no copy inside DEFLATE's 32 kB
window of another. Both bodies plan with one planner at one inflation (7
cells), so the plans are timed once; the scan work is each body's own.

    MUJOCO_GL=egl uv run python scripts/grid_spike.py            # the table
    ... --scales 1,2       # multiples of today's cells (default 1,2,3,4)
    ... --repeat 9         # calls a median is taken over (default 9)
    ... --walks [SIM_S]    # and the pairs walking #507's routes (`humanoid_
                           # spike.py --speed`) at each scale, quadrupeds and
                           # humanoids, SIM_S after the warm-up (default 60)
    ... --across           # and one fresh quadruped from the living room to
                           # the west street, the house's longest walk
    ... --out FILE         # every row as JSON
"""

from __future__ import annotations

import os

# One BLAS thread before numpy loads, as the other spikes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                  # the spikes imported below
sys.path.insert(0, str(HERE.parent / "src"))

#: The map's cell, m.
RES = 0.05
#: The bounds grow by whole steps of this many cells a side: half a metre,
#: a whole number of the lattice's 10 cm cells and a length a float holds
#: exactly, so no cell of today's map moves.
STEP_CELLS = 10
#: Past the grid's far edges, m: `OccupancyGrid` truncates its division.
EDGE_M = 1e-6
#: The corners a plan and A* go between, as shares of the grid's sides.
CORNER = 0.03
#: The house a robot that walked everywhere holds is laid by its own senses
#: from a spot every `LAY_EVERY_M` of the floor it can reach, each spot
#: `LAY_CLEAR_M` clear of anything it could walk into (`unknown_spike.py`'s
#: targets' margin).
LAY_EVERY_M, LAY_CLEAR_M = 1.0, 0.45
#: The across walk's goal (`unknown_spike.py`'s zones): the house's longest
#: walk from the living room, 43.6 m by the true route.
ACROSS_TO = "street_west"
#: What each body's map work is counted at a walked sim second: the scan
#: and the frames at their parts' rates (`lidar.LIDAR_PERIOD`, each body's
#: `DEPTH_PERIOD`), the growth at `navigator.MAP_LOOK_S`, the image at the
#: live `GRID_HZ`, a plan every 2 s (`Navigator.drive_to_routine`).
SCAN_HZ, GROWTH_HZ, IMAGE_HZ, PLAN_HZ = 10.0, 1.0, 1.0, 0.5
FRAME_HZ = {"quadruped": 10.0, "humanoid": 5.0}
#: The methods timed in flight: (module, class, method, label). `walls` is
#: called inside `layer` and `field` inside `match`: `TOTAL` is the rest.
TOTAL = ("update", "layer", "growth", "match", "plans")
WRAPS = (
  ("pluggybot.mapping.occupancy_grid", "OccupancyGrid", "update", "update"),
  ("pluggybot.legs.body", "QuadMission", "_walls", "walls"),
  ("pluggybot.humanoid.mission", "HumanoidMission", "_walls", "walls"),
  ("pluggybot.legs.body", "QuadMission", "_fold_low", "layer"),
  ("pluggybot.humanoid.mission", "HumanoidMission", "_fold_low", "layer"),
  ("pluggybot.navigator", "Navigator", "_known_cells", "growth"),
  ("pluggybot.mapping.scan_match", "ScanMatcher", "match", "match"),
  ("pluggybot.mapping.scan_match", "ScanMatcher", "refresh", "field"),
  ("pluggybot.navigator", "Navigator", "_plan_to", "plans"),
)


# ---- the grids ------------------------------------------------------------------


def today() -> tuple[float, float, float, float]:
  from pluggybot.lifecycle import QUAD_HOME, world_config
  return tuple(float(v) for v in world_config(QUAD_HOME)["grid_bounds"])


def bounds_at(scale: float) -> tuple[float, float, float, float]:
  """Today's bounds grown on every side, by whole `STEP_CELLS`, to the
  nearest `scale` times its cells with its shape kept."""
  x0, y0, x1, y1 = today()
  if scale == 1:
    return (x0, y0, x1, y1)
  cols, rows = round((x1 - x0) / RES), round((y1 - y0) / RES)
  k = math.sqrt(scale)
  best = None
  for px in _steps_near(cols * (k - 1) / 2):
    for py in _steps_near(rows * (k - 1) / 2):
      err = abs((cols + 2 * px) * (rows + 2 * py) - scale * cols * rows)
      if best is None or err < best[0]:
        best = (err, px, py)
  _, px, py = best
  return (x0 - px * RES, y0 - py * RES, x1 + px * RES + EDGE_M, y1 + py * RES + EDGE_M)


def _steps_near(cells: float) -> list[int]:
  base = STEP_CELLS * round(cells / STEP_CELLS)
  return [max(0, base + d * STEP_CELLS) for d in (-1, 0, 1)]


def synthetic(shape) -> np.ndarray:
  """#215's grid: known floor, and one wall two cells thick up the middle
  from the south edge across three quarters of it, so a plan from corner
  to corner goes round its end."""
  rows, cols = shape
  g = np.full(shape, -5.0)
  g[: 3 * rows // 4, cols // 2 - 1: cols // 2 + 1] = 5.0
  return g


def lay_house(path: Path) -> dict:
  """The house as a robot that walked everywhere holds it, laid by the
  quadruped's own senses at today's size: from a spot every `LAY_EVERY_M`
  of the floor it can reach from its start, a LIDAR scan into its map and a
  depth frame into its camera's layer, a quarter turn further round at each
  spot, at the true pose. The map and the layer go to `path`."""
  import mujoco
  from scipy import ndimage

  from humanoid_spike import speed_world
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from unknown_spike import true_floor
  model, data, (body,) = speed_world("quadruped", 1)
  m = body.mission
  body.run(body.hold_routine(0.5))
  g = m.grid
  sx, sy = g.world_to_cell(*world_config(QUAD_HOME)["start"][:2])
  labels, _ = ndimage.label(true_floor(model, g) < 0)
  floor = labels == labels[sy, sx]
  clear = ndimage.distance_transform_edt(floor) * RES >= LAY_CLEAR_M
  step = round(LAY_EVERY_M / RES)
  ys, xs = np.nonzero(clear[::step, ::step])
  adr = m.handle.qpos_adr(model)
  g.grid[...] = 0.0
  m.low[...] = 0.0
  for k, (iy, ix) in enumerate(zip(ys * step, xs * step)):
    yaw = (k % 4) * math.pi / 2
    data.qpos[adr:adr + 2] = g.cell_to_world(int(ix), int(iy))
    data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    mujoco.mj_forward(model, data)
    pose = m.true_pose()
    angles, ranges, _, _ = m.lidar.scan_split(data)
    g.update(pose, angles, ranges, m.lidar.max_range, origin=m.LIDAR_ORIGIN)
    m._set_pose(*pose)
    m._walls_t = None
    m._fold_low(m._levelled(m.depth.frame(data)).points)
  np.savez(path, grid=g.grid, low=m.low)
  known = (g.grid < -0.5) | (g.grid > 0.5)
  body.close()
  return {"spots": int(len(xs)), "knownShare": round(float(known.mean()), 3)}


def stacked(one: np.ndarray, scale: float) -> np.ndarray:
  """`one` repeated along y to `scale` times its cells."""
  reps = math.ceil(scale)
  return np.tile(one, (reps, 1))[: round(scale * one.shape[0])].copy()


# ---- timing -----------------------------------------------------------------------


def timed(fn, n: int, warm: bool = True) -> float:
  """The median of `n` calls of `fn`, ms, after one unmeasured (`warm`),
  with the collector off while it runs (a pass landing in a call is not the
  call's)."""
  if warm:
    fn()
  out = []
  gc.disable()
  try:
    for _ in range(n):
      t = time.perf_counter()
      fn()
      out.append(time.perf_counter() - t)
  finally:
    gc.enable()
  return 1000.0 * float(np.median(out))


class Timers:
  """Wall time spent in `WRAPS`' methods since `reset`, by label."""

  def __init__(self) -> None:
    import importlib
    self.s: dict[str, float] = {}
    self.n: dict[str, int] = {}
    for module, cls, name, label in WRAPS:
      klass = getattr(importlib.import_module(module), cls)
      setattr(klass, name, self._wrap(getattr(klass, name), label))

  def _wrap(self, real, label):
    def timed_call(*a, **kw):
      t = time.perf_counter()
      try:
        return real(*a, **kw)
      finally:
        self.s[label] = self.s.get(label, 0.0) + time.perf_counter() - t
        self.n[label] = self.n.get(label, 0) + 1
    return timed_call

  def reset(self) -> None:
    self.s.clear()
    self.n.clear()

  def per_sim_s(self, sim_s: float, robots: int = 1) -> dict:
    """ms of each label a sim second, per robot, and `total`: the walls are
    inside the layer, the field inside the match."""
    out = {k: round(1000.0 * v / sim_s / robots, 2) for k, v in self.s.items()}
    out["total"] = round(sum(out.get(k, 0.0) for k in TOTAL), 2)
    return out


def peak_rss_mb() -> float:
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def save_pair(arrays: dict, into: Path) -> tuple[float, float]:
  """`arrays` written as a world's save is (`continuation.write`, deflated
  and fsynced): (MB on disk, ms)."""
  from pluggybot import continuation
  snap = continuation.Snapshot({"format": continuation.FORMAT}, arrays)
  path = into / "world.npz"
  t = time.perf_counter()
  continuation.write(snap, path)
  ms = 1000.0 * (time.perf_counter() - t)
  mb = path.stat().st_size / 1e6
  path.unlink()
  return mb, ms


# ---- one scale's table ----------------------------------------------------------


def table_one(scale: float, repeat: int, house_path: str) -> dict:
  from scipy import ndimage

  from humanoid_spike import speed_world
  from pluggybot.mapping import optimistic
  from pluggybot.mapping.astar import astar
  from pluggybot.mapping.frontier import find_frontiers, traversable_mask
  from pluggybot.mapping.occupancy_grid import OccupancyGrid
  from pluggybot.perception.heightmap import HeightMap
  from pluggybot.telemetry.recorder import encode_grid_png

  bounds = bounds_at(scale)
  row: dict = {"scale": scale, "bounds": [round(v, 3) for v in bounds]}
  with np.load(house_path) as laid:
    house, house_low = laid["grid"], laid["low"]
  ox = round((today()[0] - bounds[0]) / RES)
  oy = round((today()[1] - bounds[1]) / RES)
  for kind in ("quadruped", "humanoid"):
    model, data, (body,) = speed_world(kind, 1, bounds=bounds)
    m = body.mission
    body.run(body.hold_routine(0.5))          # a scan and a depth frame
    shape = m.grid.grid.shape
    row["cells"] = int(shape[0] * shape[1])
    row["shape"] = [int(shape[0]), int(shape[1])]
    syn = synthetic(shape)
    m.grid.grid[:] = syn
    angles, ranges, _, _ = m.lidar.scan_split(data)
    pose = m.pose
    frame = m._frame if m._frame is not None else m._levelled(m.depth.frame(data))
    points = frame.points

    def walls():
      m._walls_t = None
      return m._walls()

    r = {"update": timed(lambda: m.grid.update(pose, angles, ranges, m.lidar.max_range,
                                               origin=m.LIDAR_ORIGIN), repeat),
         "walls": timed(walls, repeat)}
    m.grid.grid[:] = syn
    walls()
    r["layer"] = timed(lambda: m._fold_low(points), repeat)
    r["growth"] = timed(m._known_cells, repeat)
    r["image"] = timed(m.grid.to_image, repeat)
    fps = FRAME_HZ[kind]
    r["perSimS"] = (SCAN_HZ * r["update"] + min(SCAN_HZ, fps) * r["walls"] + fps * r["layer"]
                    + GROWTH_HZ * r["growth"] + IMAGE_HZ * r["image"])
    if kind == "quadruped":
      r.update(_plans(m, syn, repeat, optimistic, astar, traversable_mask, find_frontiers,
                      ndimage))
      r["perSimS"] += PLAN_HZ * r["planHop"]
      # the matcher, on the house where it is
      m.grid.grid[...] = 0.0
      m.grid.grid[oy:oy + house.shape[0], ox:ox + house.shape[1]] = house
      x, y, th = m.true_pose()
      r["field"] = timed(lambda: m.matcher.refresh(x, y), repeat)
      m.matcher.refresh(x, y)
      r["match"] = timed(lambda: m.matcher.match((x, y, th), angles, ranges), repeat)
      r["matcherMb"] = m.matcher.field.nbytes / 1e6
      r["robotMb"] = r["matcherMb"] + sum(a.nbytes for a in (m.grid.grid, m.low, m._walls_mask,
                                                             m._floor)) / 1e6
      # the wire and the save: the house a robot walked everywhere, k times
      one = OccupancyGrid(*today(), resolution=RES)
      big, low = stacked(house, scale), stacked(house_low, scale)
      g = OccupancyGrid(one.x_min, one.y_min, one.x_max,
                        one.y_min + big.shape[0] * RES + EDGE_M, resolution=RES)
      g.grid = big
      r["wireImage"] = timed(g.to_image, repeat)
      img = g.to_image()
      r["wirePng"] = timed(lambda: encode_grid_png(img), repeat)
      r["wirePngKb"] = len(encode_grid_png(img)) / 1000.0
      hm = HeightMap()
      arrays = {}
      for root in ("pluggybot", "r2_pluggybot"):
        arrays.update({f"{root}/grid": big, f"{root}/low": low,
                       f"{root}/matchField": m.matcher.field,
                       f"{root}/heightmap": hm.height, f"{root}/heightcount": hm.count})
      with tempfile.TemporaryDirectory(dir=str(HERE.parent)) as d:
        sizes = [save_pair(arrays, Path(d)) for _ in range(3)]
      r["saveMb"] = sizes[0][0]
      r["saveMs"] = float(np.median([s[1] for s in sizes]))
      r["saveRawMb"] = sum(a.nbytes for a in arrays.values()) / 1e6
    else:
      r["perSimS"] += PLAN_HZ * row["quadruped"]["planHop"]
      x, y, _ = m.pose
      m._plan_optimistic(x + 3.0, y)
      r["robotMb"] = sum(a.nbytes for a in (m.grid.grid, m.low, m._walls_mask, m._floor)) / 1e6
      r["robotMb"] += row["quadruped"]["matcherMb"]
    row[kind] = r
    body.close()
  row["peakRssMb"] = peak_rss_mb()
  return row


def _plans(m, syn, repeat, optimistic, astar, traversable_mask, find_frontiers, ndimage) -> dict:
  """A walk's plans on the lattice, piece by piece, and an explore leg's
  mask, frontiers and A*, on the synthetic grid."""
  g = m.grid
  rows, cols = syn.shape
  m.grid.grid[:] = syn
  m.low[...] = 0.0
  m.low[: 3, : 3] = 3.0                   # something seen low: the burn-in copies
  sx, sy = g.cell_to_world(int(CORNER * cols), int(CORNER * rows))
  gx, gy = g.cell_to_world(int((1 - CORNER) * cols), int((1 - CORNER) * rows))
  m._set_pose(sx, sy, 0.0)
  b = optimistic.BLOCK
  shape = (-(-rows // b), -(-cols // b))
  optimistic._LATTICES.clear()
  t = time.perf_counter()
  graph = optimistic.lattice_for(shape, RES * b)
  r = {"latticeBuildMs": 1000.0 * (time.perf_counter() - t),
       "latticeMb": sum(a.nbytes for a in (graph.graph.data, graph.graph.indices,
                                           graph.graph.indptr, graph._pad, graph._steps)) / 1e6}
  r["planAcross"] = timed(lambda: m._plan_optimistic(gx, gy), repeat)
  r["planHop"] = timed(lambda: m._plan_optimistic(sx + 3.0, sy), repeat)
  planning = m._planning_grid()
  r["planGrid"] = timed(m._planning_grid, repeat)
  r["planCosts"] = timed(lambda: optimistic.map_costs(planning, m.INFLATION_CELLS,
                                                      m.UNKNOWN_COST), repeat)
  cost = optimistic.map_costs(planning, m.INFLATION_CELLS, m.UNKNOWN_COST)
  r["planCoarsen"] = timed(lambda: optimistic.coarsen(cost, b), repeat)
  lattice = optimistic.coarsen(cost, b)
  r["planWeigh"] = timed(lambda: graph.weigh(lattice), repeat)
  start, goal = (int(CORNER * cols) // b, int(CORNER * rows) // b), \
                (int((1 - CORNER) * cols) // b, int((1 - CORNER) * rows) // b)
  r["planRoute"] = timed(lambda: optimistic.route(graph, lattice, start, goal), repeat)
  tracemalloc.start()
  before = tracemalloc.get_traced_memory()[0]
  m._plan_optimistic(gx, gy)
  r["planPeakMb"] = (tracemalloc.get_traced_memory()[1] - before) / 1e6
  tracemalloc.stop()
  # the explore's leg: #215's columns
  trav = traversable_mask(syn, m.INFLATION_CELLS)
  r["mask"] = timed(lambda: traversable_mask(syn, m.INFLATION_CELLS), repeat)
  r["frontiers"] = timed(lambda: find_frontiers(syn, traversable=trav), repeat)
  r["label"] = timed(lambda: ndimage.label(trav), repeat)
  a, z = (int(CORNER * cols), int(CORNER * rows)), (int((1 - CORNER) * cols), int((1 - CORNER) * rows))
  r["astarS"] = timed(lambda: astar(trav, a, z), max(1, repeat // 3), warm=False) / 1000.0
  m._plan_optimistic(gx, gy)            # leaves `_floor` as a walk does
  return r


# ---- in flight ------------------------------------------------------------------


def walks_one(kind: str, scale: float, sim_s: float) -> dict:
  """The pair of `kind` walking #507's routes (`humanoid_spike.py
  --speed`) for `sim_s` after its warm-up, the maps over `scale`'s bounds:
  real time, and the map's work a sim second for each robot."""
  from humanoid_spike import SPEED_ROUTES, SPEED_WARM_S, _paces, speed_world
  from pluggybot import tick
  timers = Timers()
  model, data, bodies = speed_world(kind, 2, bounds=bounds_at(scale))
  tick.run_many([(b.stepper, _paces(b, r, SPEED_WARM_S, data))
                 for b, r in zip(bodies, SPEED_ROUTES)])
  timers.reset()
  t_sim0 = float(data.time)
  t0 = time.perf_counter()
  tick.run_many([(b.stepper, _paces(b, r, t_sim0 + sim_s, data))
                 for b, r in zip(bodies, SPEED_ROUTES)])
  wall = time.perf_counter() - t0
  sim = float(data.time) - t_sim0
  arrays = {}
  for b in bodies:
    m = b.mission
    arrays.update({f"{m.handle.root}/grid": m.grid.grid, f"{m.handle.root}/low": m.low})
    if m.matcher is not None and m.matcher.field is not None:
      arrays[f"{m.handle.root}/matchField"] = m.matcher.field
  with tempfile.TemporaryDirectory(dir=str(HERE.parent)) as d:
    save_mb, save_ms = save_pair(arrays, Path(d))
  out = {"kind": kind, "scale": scale, "cells": int(bodies[0].mission.grid.grid.size),
         "simS": round(sim, 2), "wallS": round(wall, 2), "rtf": round(sim / wall, 3),
         "msPerSimS": timers.per_sim_s(sim, robots=len(bodies)),
         "plans": timers.n.get("plans", 0),
         "ends": [[round(v, 3) for v in b.mission.true_pose()] for b in bodies],
         "saveMb": round(save_mb, 3), "saveMs": round(save_ms, 1),
         "peakRssMb": round(peak_rss_mb(), 1)}
  for b in bodies:
    b.close()
  return out


def across_one(scale: float, patience: float) -> dict:
  """One fresh quadruped from the living room to `ACROSS_TO`, its map over
  `scale`'s bounds: the walk, its plans, and the map's work a sim second."""
  import mujoco

  from pluggybot.legs import body as qb
  from unknown_spike import targets, world
  cfg, model, grid, truth = world()
  zone = [z for z in cfg["zones"] if z["name"] == ACROSS_TO]
  goal = targets(truth, grid, zone)[ACROSS_TO]
  timers = Timers()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=bounds_at(scale))
  m = body.mission
  body.pack(1000.0, 1.0)
  m.start_at(*cfg["start"])
  body.run(body.look_around_routine())
  timers.reset()
  t_sim0 = float(data.time)
  t0 = time.perf_counter()
  arrived = body.run(body.go_to_routine(*goal, timeout=patience))
  wall = time.perf_counter() - t0
  sim = float(data.time) - t_sim0
  out = {"scale": scale, "cells": int(m.grid.grid.size), "goal": list(goal),
         "arrived": bool(arrived), "why": (m.last_drive or {}).get("why", ""),
         "simS": round(sim, 1), "wallS": round(wall, 1), "rtf": round(sim / wall, 3),
         "plans": timers.n.get("plans", 0),
         "planMsMean": round(1000.0 * timers.s.get("plans", 0.0) / max(1, timers.n.get("plans", 0)), 1),
         "msPerSimS": timers.per_sim_s(sim),
         "end": [round(v, 3) for v in m.true_pose()],
         "peakRssMb": round(peak_rss_mb(), 1)}
  body.close()
  return out


# ---- the report -------------------------------------------------------------------


def _one(args: list[str]) -> dict | None:
  out = subprocess.run([sys.executable, __file__, "--one", *args], capture_output=True, text=True)
  line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
  if not line:
    print(f"{' '.join(args)}: no record\n{out.stderr[-3000:]}", file=sys.stderr)
    return None
  return json.loads(line[-1])


def report_table(rows: list[dict]) -> None:
  def line(label: str, cells: list, fmt: str = "{:10.2f}") -> None:
    print(f"  {label:44s}" + "".join(fmt.format(c) if c is not None else f"{'-':>10s}"
                                      for c in cells))
  q = [r["quadruped"] for r in rows]
  h = [r["humanoid"] for r in rows]
  print(f"  {'cells':44s}" + "".join(f"{r['cells']:>10,d}" for r in rows))
  print(f"  {'x today':44s}" + "".join(f"{r['cells'] / rows[0]['cells']:10.2f}" for r in rows))
  print("PER SCAN, FRAME OR SECOND, ms a call")
  line("the map's update (a scan, 10 Hz)", [r["update"] for r in q])
  line("the walls, quadruped (its window, a scan)", [r["walls"] for r in q])
  line("the walls, humanoid (the grid, a frame)", [r["walls"] for r in h])
  line("the camera's layer, quadruped (10 Hz)", [r["layer"] for r in q])
  line("the camera's layer, humanoid (5 Hz)", [r["layer"] for r in h])
  line("the map's growth (1 Hz walking)", [r["growth"] for r in q])
  line("the grid's image for the wire (1 Hz live)", [r["image"] for r in q])
  line("=> map work a walked sim second, quadruped", [r["perSimS"] for r in q], "{:10.1f}")
  line("=> map work a walked sim second, humanoid", [r["perSimS"] for r in h], "{:10.1f}")
  print("A WALK'S PLAN ON THE LATTICE (every 2 s walking), ms")
  line("across the grid, corner to corner", [r["planAcross"] for r in q], "{:10.1f}")
  line("  the planning grid (the camera's burn-in)", [r["planGrid"] for r in q], "{:10.1f}")
  line("  the costs (the inflation's transform)", [r["planCosts"] for r in q], "{:10.1f}")
  line("  the lattice's cells (coarsen)", [r["planCoarsen"] for r in q], "{:10.1f}")
  line("  the route: weights, Dijkstra, the path", [r["planRoute"] for r in q], "{:10.1f}")
  line("    of which the weights", [r["planWeigh"] for r in q], "{:10.1f}")
  line("a 3 m hop", [r["planHop"] for r in q], "{:10.1f}")
  line("the lattice's build, once a process", [r["latticeBuildMs"] for r in q], "{:10.0f}")
  print("AN EXPLORE'S LEG (`navigation.plan`)")
  line("the mask, ms", [r["mask"] for r in q], "{:10.1f}")
  line("the frontiers, ms", [r["frontiers"] for r in q], "{:10.1f}")
  line("the robot's own component (label), ms", [r["label"] for r in q], "{:10.1f}")
  line("A* across the grid (pure Python), s", [r["astarS"] for r in q])
  print("THE SCAN MATCHER (its field every 1 m walked or 20 fused scans), ms")
  line("the field", [r["field"] for r in q])
  line("one match", [r["match"] for r in q])
  print("ON THE WIRE, the house mapped and stacked")
  line("the image, ms (physics thread)", [r["wireImage"] for r in q])
  line("its PNG, ms (the sender's thread)", [r["wirePng"] for r in q], "{:10.1f}")
  line("its PNG, kB in base64", [r["wirePngKb"] for r in q], "{:10.1f}")
  print("WORLD.NPZ, the pair's maps (every 60 sim s, physics thread)")
  line("raw, MB", [r["saveRawMb"] for r in q], "{:10.1f}")
  line("on disk, MB", [r["saveMb"] for r in q])
  line("the write, ms", [r["saveMs"] for r in q], "{:10.0f}")
  print("MEMORY, MB")
  line("a quadruped's map arrays", [r["robotMb"] for r in q], "{:10.1f}")
  line("a humanoid's map arrays", [r["robotMb"] for r in h], "{:10.1f}")
  line("a plan's peak, over them", [r["planPeakMb"] for r in q], "{:10.1f}")
  line("the lattice, once a process", [r["latticeMb"] for r in q], "{:10.1f}")
  line("the process's peak RSS", [r["peakRssMb"] for r in rows], "{:10.0f}")


def _breakdown(ms: dict) -> str:
  names = ("update", "layer", "walls", "growth", "match", "field", "plans")
  return "  ".join(f"{k} {ms.get(k, 0.0):.1f}" for k in names)


def report_walks(rows: list[dict]) -> None:
  print(f"  {'pair':10s} {'cells':>10s} {'sim s':>6s} {'wall s':>7s} {'real time':>9s} "
        f"{'vs 1x':>6s} {'wall ms':>8s} {'map ms':>7s} {'plans':>6s}")
  print(f"  {'':10s} {'':>10s} {'':>6s} {'':>7s} {'':>9s} {'':>6s} {'a sim s':>8s} "
        f"{'a robot':>7s}")
  base = {}
  for r in rows:
    base.setdefault(r["kind"], r["rtf"])
    print(f"  {r['kind']:10s} {r['cells']:>10,d} {r['simS']:6.1f} {r['wallS']:7.1f} "
          f"{r['rtf']:8.2f}x {r['rtf'] / base[r['kind']]:6.3f} {1000.0 / r['rtf']:8.0f} "
          f"{r['msPerSimS']['total']:7.1f} {r['plans']:6d}")
    print(f"  {'':10s} {_breakdown(r['msPerSimS'])}")
    print(f"  {'':10s} ends {r['ends']}; save {r['saveMb']:.2f} MB in {r['saveMs']:.0f} ms; "
          f"peak RSS {r['peakRssMb']:.0f} MB")


def report_across(rows: list[dict]) -> None:
  for r in rows:
    print(f"  {r['cells']:>10,d} cells: {'arrived' if r['arrived'] else r['why']} "
          f"in {r['simS']:.1f} sim s, {r['wallS']:.1f} wall s ({r['rtf']:.2f}x real time, "
          f"{1000.0 / r['rtf']:.0f} wall ms a sim s), {r['plans']} plans at "
          f"{r['planMsMean']:.1f} ms; end {r['end']}")
    print(f"  {'':10s}  map ms a sim second {r['msPerSimS']['total']:.1f}: "
          f"{_breakdown(r['msPerSimS'])}; peak RSS {r['peakRssMb']:.0f} MB")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--scales", default="1,2,3,4",
                  help="multiples of today's cells, comma separated (default 1,2,3,4)")
  ap.add_argument("--repeat", type=int, default=9, help="calls a median is taken over (9)")
  ap.add_argument("--no-table", action="store_true", help="skip the table")
  ap.add_argument("--walks", nargs="?", type=float, const=60.0, default=None, metavar="SIM_S",
                  help="the pairs' walks at each scale, SIM_S after the warm-up (60)")
  ap.add_argument("--across", action="store_true",
                  help=f"one fresh quadruped from the living room to {ACROSS_TO} at each scale")
  ap.add_argument("--patience", type=float, default=300.0, help="--across's walk, s (300)")
  ap.add_argument("--out", default=None, help="every row as JSON")
  ap.add_argument("--one", nargs="+", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one:
    what, *rest = args.one
    if what == "lay":
      out = lay_house(Path(rest[0]))
    elif what == "table":
      out = table_one(float(rest[0]), int(rest[1]), rest[2])
    elif what == "walks":
      out = walks_one(rest[0], float(rest[1]), float(rest[2]))
    else:
      out = across_one(float(rest[0]), float(rest[1]))
    print(json.dumps(out))
    return 0
  scales = [float(s) for s in args.scales.split(",")]
  result: dict = {}
  if not args.no_table:
    with tempfile.TemporaryDirectory(dir=str(HERE.parent)) as d:
      house = str(Path(d) / "house.npz")
      result["house"] = _one(["lay", house])
      rows = [r for s in scales
              if (r := _one(["table", str(s), str(args.repeat), house])) is not None]
    result["table"] = rows
    print("THE TABLE (#215's synthetic grid; medians of", args.repeat, "calls)")
    print(f"  the house laid from {result['house']['spots']} spots, "
          f"{result['house']['knownShare']:.0%} of its grid known")
    report_table(rows)
  if args.walks is not None:
    rows = [r for kind in ("quadruped", "humanoid") for s in scales
            if (r := _one(["walks", kind, str(s), str(args.walks)])) is not None]
    result["walks"] = rows
    print(f"THE PAIRS' WALKS (#507's routes, {args.walks:g} sim s after the warm-up)")
    report_walks(rows)
  if args.across:
    rows = [r for s in scales if (r := _one(["across", str(s), str(args.patience)])) is not None]
    result["across"] = rows
    print(f"ACROSS THE HOUSE (a fresh quadruped, the living room to {ACROSS_TO})")
    report_across(rows)
  if args.out:
    Path(args.out).write_text(json.dumps(result, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
