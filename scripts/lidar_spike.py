#!/usr/bin/env python
"""The humanoid's LIDAR (issue #581; SimNotes, "The 3D LIDAR on the head" and
"The band served"): the Livox Mid-360S upside down on its head -- what the
robot's own body hides of a frame, what a frame costs, and what the scan
matcher can fix outdoors with its points laid flat (`Lidar.band`, the scan
the map is laid from since #581's stage 2) against its level ring
(`Lidar.scan`, the C1's plane's, which it was laid from before).

    MUJOCO_GL=egl uv run python scripts/lidar_spike.py --hidden
                                  # what its body hides of a frame: standing,
                                  # and a hand held at each of the reach
                                  # scene's heights (`humanoid_spike.py`)
    ... --cost [--poses N]        # ms a frame in the house: the level ring,
                                  # the cloud, the band, the depth camera's
    ... --probe [--out rows.json] # #572's static probe on its model: at every
                                  # outdoor point 0.5 m apart, from the true
                                  # pose, does a scan pin the heading -- the
                                  # level ring's against the band's
    ... --view                    # a walk from the living room to the
                                  # storeroom across the street, its map
                                  # laid from its points laid flat, the
                                  # band's returns drawn red and the level
                                  # ring's blue (MUJOCO_GL unset)
    ... --strip [--out PNG]       # its filmstrip: lidar_spike.png

The truth -- where the robot is, what a ray struck -- is ours, read to
place it and to grade it; nothing here steers off it but the walk's own
legs and map.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pluggybot.humanoid import model as hm  # noqa: E402
from pluggybot.perception import lidar as ld  # noqa: E402

HOME_META = "models/home_world.meta.json"
#: The probe's groups of the house's zones, as #572's probe read them.
GROUPS = {
  "middle street": ("street",),
  "sidewalks by it": ("sidewalk", "sidewalk_2"),
  "gardens": ("garden", "garden_south", "garden_2"),
  "block sidewalks": ("sidewalk_north", "sidewalk_south", "sidewalk_2_north",
                      "sidewalk_2_south", "sidewalk_west", "sidewalk_east"),
  "the loop": ("street_north", "street_south", "street_west", "street_east"),
  "rooms": ("kitchen", "workshop", "hall", "living", "bedroom", "lobby", "lab", "store"),
}
#: Sectors of a frame a report names, deg from straight ahead, each
#: `SECTOR_DEG` either side.
SECTORS = {"ahead": 0.0, "left": 90.0, "behind": 180.0, "right": -90.0}
SECTOR_DEG = 20.0
#: The walk the scene flies: from the living room's start to the storeroom
#: across the street (#572's drift flights' far end).
WALK_TO = (25.0, -3.0)


def zones() -> list[dict]:
  return json.loads(Path(HOME_META).read_text())["zones"]


def lattice(zone: dict, step: float = 0.5, inset: float = 0.25) -> list[tuple[float, float]]:
  (x0, y0), (x1, y1) = zone["min"], zone["max"]
  xs = np.arange(x0 + inset, x1 - inset + 1e-9, step)
  ys = np.arange(y0 + inset, y1 - inset + 1e-9, step)
  return [(float(x), float(y)) for x in xs for y in ys]


# ---- the robot placed: standing, by its joints, nothing stepped -------------------

def house():
  """The served house (`humanoid.world.home_spec`) with one humanoid in it:
  (model, data, its LIDAR as the Navigator builds it)."""
  from pluggybot.humanoid.world import home_spec
  model = home_spec().compile()
  data = mujoco.MjData(model)
  return model, data, ld.Lidar(model)


def stand_at(model, data, x: float, y: float, yaw: float) -> None:
  """The robot standing at (x, y) facing `yaw`, its arms hanging -- placed,
  never stepped (`mj_kinematics`, and `mj_camlight`: the cameras ride on
  it too)."""
  from pluggybot.robot import FIRST
  q = FIRST.qpos_adr(model)
  data.qpos[q:q + 7] = [x, y, hm.stand_height(), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  for name, v in hm.stand_qpos().items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = v
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)


def level_of(yaw: float) -> np.ndarray:
  """Standing upright: its tilt is none, so level from its root is the
  identity (the band's `level` is the heading taken out)."""
  return np.eye(3)


# ---- --hidden ------------------------------------------------------------------

def hidden_report(model, data, lidar, label: str) -> dict:
  """What the robot's own body hides of one frame: its share; in each
  sector, how many bearings it hides whole (the bracket's post's), and
  down to which ring every other bearing there is clear, and whose geoms
  hide anything there."""
  mine = lidar.hidden(data)
  _, hit = lidar._cast(data, lidar._dirs)
  hit = hit.reshape(mine.shape)
  deg = np.degrees(lidar.ray_angles)
  el = np.degrees(lidar.elevations)
  out = {"pose": label, "hiddenPct": round(100 * float(mine.mean()), 2), "sectors": {}}
  whole = mine.all(axis=0)
  for name, centre in SECTORS.items():
    cols = np.abs((deg - centre + 180) % 360 - 180) <= SECTOR_DEG
    part = cols & ~whole
    clear = ~mine[:, part].any(axis=1)
    # the steepest ring from which, up, every bearing here but those hidden whole is clear
    k = len(el)
    while k > 0 and clear[k - 1]:
      k -= 1
    steep = float(el[k]) if k < len(el) else None
    names = sorted({model.geom(int(g)).name for g in np.unique(hit[:, cols][mine[:, cols]])})
    out["sectors"][name] = {"clearFromDeg": steep, "wholeBearings": int((cols & whole).sum()),
                            "hiddenRaysPct": round(100 * float(mine[:, cols].mean()), 1),
                            "by": names}
  return out


def hidden_table() -> list[dict]:
  """Standing with its arms hanging, then the reach scene's four holds
  (`humanoid_spike.reach_scene`: the left hand at a counter, a high shelf,
  a low shelf and the floor, each with the posture its reach took)."""
  import humanoid_spike as hs
  rows = []
  model, data, body = hs.reach_body_world()
  lidar = ld.Lidar(model)
  mujoco.mj_forward(model, data)
  rows.append(hidden_report(model, data, lidar, "standing, arms hanging"))
  holding = None
  for line in hs.reach_scene(body, model, data):
    if line and line.startswith("holding at"):
      holding = line[len("holding at "):]
    elif holding is not None and line is None:
      rows.append(hidden_report(model, data, lidar, f"its left hand held at {holding}"))
      holding = None
  body.close()
  print(f"WHAT ITS OWN BODY HIDES OF A FRAME (30 rings, -52 to +6 deg, 360 bearings; "
        f"each sector {SECTOR_DEG:g} deg either side):")
  for r in rows:
    print(f"  {r['pose']}: {r['hiddenPct']:.1f} % of the rays")
    for name, s in r["sectors"].items():
      clear = "no ring" if s["clearFromDeg"] is None else f"{s['clearFromDeg']:+.0f} deg"
      whole = f", {s['wholeBearings']} bearings hidden whole" if s["wholeBearings"] else ""
      print(f"    {name:6s}: {s['hiddenRaysPct']:5.1f} % hidden{whole}; every other bearing "
            f"clear down to {clear}; by {', '.join(s['by']) or 'nothing'}")
  return rows


# ---- --cost --------------------------------------------------------------------

def cost_table(n: int = 120, repeat: int = 3) -> dict:
  """ms a frame in the served house, at `n` places drawn over its zones:
  the level ring (served today), the whole frame's cloud, the band laid
  flat from it, and the depth camera's frame beside them (`perception.
  depth`, the planner's layer's sensor). Each the best of `repeat` passes
  over the places, one after another in this process."""
  import gc
  from pluggybot.perception.depth import UNCULLED, DepthCamera
  from pluggybot.robot import FIRST
  model, data, lidar = house()
  rng = np.random.default_rng(581)
  places = [p for z in zones() for p in lattice(z, step=1.0, inset=0.5)]
  pick = rng.choice(len(places), size=min(n, len(places)), replace=False)
  poses = [(*places[k], float(rng.uniform(-math.pi, math.pi))) for k in pick]
  depth = DepthCamera(model, FIRST, mount="body", reach=UNCULLED)
  calls = {
    "level ring (360 rays)": lambda: lidar.scan_split(data),
    "cloud (10,800 rays)": lambda: lidar.cloud(data),
    "band, laid flat": lambda: lidar.band(data, np.eye(3), hm.stand_height()),
    "depth camera frame (8,400 rays)": lambda: depth.frame(data),
  }
  out = {}
  gc.disable()
  try:
    for name, call in calls.items():
      best = math.inf
      for _ in range(repeat):
        t = 0.0
        for (x, y, yaw) in poses:
          stand_at(model, data, x, y, yaw)
          t0 = time.perf_counter()
          call()
          t += time.perf_counter() - t0
        best = min(best, t)
      out[name] = 1000 * best / len(poses)
  finally:
    gc.enable()
  print(f"A FRAME'S COST in the served house, ms, the best of {repeat} passes over "
        f"{len(poses)} places:")
  for name, ms in out.items():
    print(f"  {name:32s} {ms:7.2f}")
  return out


# ---- --probe -------------------------------------------------------------------

def _scan(variant: str, model, data, lidar):
  """(angles, ranges, where each ray's free run begins): the level ring's,
  or the band's, the map's since #581's stage 2."""
  if variant == "ring":
    return (*lidar.scan(data), None)
  s = lidar.band(data, np.eye(3), hm.stand_height())
  return s.angles, s.ranges, s.free_from


def lay(variant: str, model, data, lidar, bounds, step: float = 1.0, seed: int = 7):
  """A map laid by the robot's own rules (`OccupancyGrid.update`) from an
  even explore: one scan a `step` m over every zone, each at a heading
  drawn, from the true pose -- the best map its sensor lays."""
  from pluggybot.mapping.occupancy_grid import OccupancyGrid
  g = OccupancyGrid(x_min=bounds[0], y_min=bounds[1], x_max=bounds[2], y_max=bounds[3],
                    resolution=0.05)
  rng = np.random.default_rng(seed)
  for z in zones():
    for (x, y) in lattice(z, step=step, inset=step / 2):
      th = float(rng.uniform(-math.pi, math.pi))
      stand_at(model, data, x, y, th)
      a, r, free_from = _scan(variant, model, data, lidar)
      g.update((x, y, th), a, r, lidar.max_range, origin=(0.0, 0.0), free_from=free_from)
  return g


def probe(variant: str, seed: int = 581) -> list[dict]:
  """At every point of every zone 0.5 m apart, a fresh scan matched from the
  true pose against the map `lay` laid (`ScanMatcher`): its verdict, how many
  directions it held, and which."""
  from pluggybot.mapping.scan_match import ScanMatcher
  model, data, lidar = house()
  meta = json.loads(Path(HOME_META).read_text())
  g = lay(variant, model, data, lidar, meta["gridBounds"])
  rng = np.random.default_rng(seed)
  rows = []
  for z in zones():
    for (x, y) in lattice(z):
      th = float(rng.uniform(-math.pi, math.pi))
      stand_at(model, data, x, y, th)
      a, r, _ = _scan(variant, model, data, lidar)
      m = ScanMatcher(g, origin=(0.0, 0.0), max_range=lidar.max_range).match((x, y, th), a, r)
      rows.append({"variant": variant, "zone": z["name"], "x": x, "y": y, "why": m.why,
                   "held": m.held, "weak": [float(v) for v in m.weak_dir],
                   "returns": int(np.sum(r < lidar.max_range - 1e-9))})
  return rows


def probe_summary(rows: list[dict]) -> None:
  """By group: the share of places a scan pinned in all three directions,
  the heading and one (held along a wall), and not the heading."""
  print(f"\n=== {rows[0]['variant']}: places whose scan pins ...")
  print(f"  {'':16s} {'n':>4s} | {'all 3':>6s} {'heading':>8s} {'NOT heading':>11s} | returns")
  for grp, names in GROUPS.items():
    rs = [r for r in rows if r["zone"] in names]
    if not rs:
      continue
    n = len(rs)
    acc = [r for r in rs if r["why"] == "ok"]
    full = sum(1 for r in acc if r["held"] == 0)
    heading = sum(1 for r in acc if r["held"] == 1 and abs(r["weak"][2]) < 0.5)
    pc = lambda k: f"{100 * k / n:5.0f}%"  # noqa: E731
    print(f"  {grp:16s} {n:4d} | {pc(full):>6s} {pc(heading):>8s} {pc(n - full - heading):>11s} "
          f"| {np.mean([r['returns'] for r in rs]):6.0f}")


# ---- --view and --strip -----------------------------------------------------------

#: The drawn returns: the band's (what its points laid flat would put in the
#: map) and the level ring's (what its map is laid from today).
BAND_RGBA = np.array([0.95, 0.25, 0.15, 0.9], dtype=np.float32)
RING_RGBA = np.array([0.15, 0.45, 0.95, 0.9], dtype=np.float32)
DOT_M = 0.035


def walk_world():
  """The served body in its house at the living room's start, as the loop
  boots it: (model, data, body)."""
  from pluggybot.body import body_for
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  body = body_for(model, data, grid_bounds=cfg["grid_bounds"])
  body.start_at(*cfg["start"])
  return model, data, body


def watcher(model, body) -> ld.Lidar:
  """The LIDAR as a watcher draws it: the robot's own part, cast on a noise
  stream of its own -- drawn off the robot's, watching moved the flight."""
  return ld.Lidar(model, site_name=body.handle.el("lidar"), robot_body=body.handle.root,
                  seed=581)


def dots(model, data, lidar: ld.Lidar) -> list[tuple[np.ndarray, np.ndarray]]:
  """Where this frame's returns truly are, to draw (`lidar`, a `watcher`):
  per bearing the nearest point in the band (red) and the level ring's
  returns (blue), off the sensor's true pose -- what it sampled, not where
  its map put it."""
  pos = data.site_xpos[lidar.site_id].copy()
  mat = data.site_xmat[lidar.site_id].reshape(3, 3).copy()
  c = lidar.cloud(data)
  world = c.points @ mat.T + pos
  out = []
  band = (world[:, 2] >= ld.BAND_M[0]) & (world[:, 2] <= ld.BAND_M[1])
  pts = world[band]
  if len(pts):
    dx, dy = pts[:, 0] - pos[0], pts[:, 1] - pos[1]
    r = np.hypot(dx, dy)
    b = ld._bin(np.arctan2(dy, dx))
    best = np.full(ld.BEARINGS, np.inf)
    np.minimum.at(best, b, r)
    keep = r == best[b]
    out += [(p, BAND_RGBA) for p in pts[keep]]
  a, rr = lidar.scan(data)
  hit = rr < lidar.max_range - 1e-9
  dirs = np.stack([np.cos(a[hit]), np.sin(a[hit]), np.zeros(hit.sum())], axis=1) @ mat.T
  out += [(p, RING_RGBA) for p in pos + dirs * rr[hit, None]]
  return out


def draw(scene, marks) -> None:
  """Each mark a small cube in `scene`, while it has room."""
  for p, rgba in marks:
    if scene.ngeom >= scene.maxgeom:
      return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([DOT_M / 2] * 3), np.asarray(p, dtype=float),
                        np.eye(3).reshape(-1), rgba)
    scene.ngeom += 1


def walk_routine(body):
  """The walk: across the house and the street to the storeroom, on its
  own map, laying it as it goes (its level ring's, as served)."""
  yield from body.go_to_routine(*WALK_TO, timeout=240.0)


def view() -> None:
  from mujoco import viewer as mj_viewer
  from pluggybot import tick
  model, data, body = walk_world()
  eye = watcher(model, body)
  print("the walk: the living room to the storeroom across the street, its map laid from "
        "its points laid flat; its LIDAR's returns drawn every 0.1 s -- red, the nearest "
        "0.15-2.2 m over the floor each bearing; blue, the level ring its map is laid from "
        "as served. Close the window to quit.")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body.root
    # looking east, the way it walks, from behind it
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7.0, 0.0, -35.0
    marks: list = []
    wall0 = time.time() - data.time

    def frame():
      nonlocal marks
      if int(round(data.time / model.opt.timestep)) % 50 == 0:
        marks = dots(model, data, eye)
      if int(round(data.time / model.opt.timestep)) % 10 == 0:
        with viewer.lock():
          viewer.user_scn.ngeom = 0
          draw(viewer.user_scn, marks)
        viewer.sync()
        ahead = data.time - (time.time() - wall0)
        if ahead > 0:
          time.sleep(ahead)
      if not viewer.is_running():
        raise tick.MissionAborted("viewer closed")

    body.step_hooks.append(frame)
    try:
      while viewer.is_running():
        body.run(walk_routine(body))
        body.start_at(*walk_world_start())
    except tick.MissionAborted:
      pass
  body.close()


def walk_world_start():
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  return world_config(HUMANOID_HOME)["start"]


def strip(out: str = "lidar_spike.png", tiles: int = 9, w: int = 480, h: int = 360) -> None:
  """The walk's filmstrip: a tile every 15 s from 5 s, and one where the
  walk ends if its row has room, the returns drawn, each labelled with
  where the robot truly is."""
  from PIL import Image, ImageDraw
  from station_spike import zone_of
  model, data, body = walk_world()
  eye = watcher(model, body)
  r = mujoco.Renderer(model, h, w)
  shots: list = []
  cols = 3
  due = [5.0 + k * 15.0 for k in range(tiles)]

  def shot():
    if due and data.time >= due[0]:
      due.pop(0)
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[body.root] + [0.0, 0.0, 0.2]
      _, _, yaw = body.true_pose()
      # from behind it, looking the way it faces
      cam.distance, cam.azimuth, cam.elevation = 6.5, math.degrees(yaw), -32.0
      r.update_scene(data, cam)
      draw(r.scene, dots(model, data, eye))
      img = Image.fromarray(r.render())
      x, y, _ = body.true_pose()
      ImageDraw.Draw(img).text((8, 8), f"t={data.time:5.1f} s, in the {zone_of(x, y)}",
                               fill=(255, 255, 255))
      shots.append(img)

  body.step_hooks.append(shot)
  body.run(walk_routine(body))
  if len(shots) % cols:
    due[:] = [data.time]
    shot()
  body.close()
  r.close()
  rows = math.ceil(len(shots) / cols)
  sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
  for k, img in enumerate(shots):
    sheet.paste(img, ((k % cols) * w, (k // cols) * h))
  sheet.save(out)
  print(f"wrote {out}: {len(shots)} tiles")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--hidden", action="store_true", help="what its own body hides of a frame")
  ap.add_argument("--cost", action="store_true", help="ms a frame in the served house")
  ap.add_argument("--poses", type=int, default=120, help="--cost's places")
  ap.add_argument("--probe", action="store_true",
                  help="#572's static probe: the level ring against the band")
  ap.add_argument("--variants", default="ring,band", help="--probe's: ring, band or both")
  ap.add_argument("--view", action="store_true", help="the walk in the viewer (MUJOCO_GL unset)")
  ap.add_argument("--strip", action="store_true", help="the walk's filmstrip")
  ap.add_argument("--out", default=None, help="rows as JSON (--hidden, --cost, --probe), "
                  "or the filmstrip's PNG")
  args = ap.parse_args(argv)
  if args.view:
    view()
    return 0
  if args.strip:
    strip(args.out or "lidar_spike.png")
    return 0
  result = None
  if args.hidden:
    result = hidden_table()
  elif args.cost:
    result = cost_table(args.poses)
  elif args.probe:
    result = {}
    for v in args.variants.split(","):
      t0 = time.time()
      result[v] = probe(v)
      print(f"[{v}: {time.time() - t0:.0f} s]")
      probe_summary(result[v])
  else:
    ap.print_help()
    return 2
  if args.out:
    Path(args.out).write_text(json.dumps(result))
  return 0


if __name__ == "__main__":
  sys.exit(main())
