"""Near-field 3D sensing spike (issue #34): the numbers behind the sensor
decision, the mount, and the height-map-versus-voxel choice.

The robot's LIDAR looks along one plane at 0.223 m and never at the floor.
`perception/depth.py` is a D435-class depth camera on the mast top,
`perception/heightmap.py` the robot-centric 2.5D map it feeds; docs/Parts.md
"near-field depth camera" keeps the decision and this script keeps it measured.

Four tables, each one question:

  --mount      pitch sweep on the real mast-top mount: how much of the frame
               is the robot's own body, and which band of floor the centre
               column sees. The deck shadows the floor to ~7 cm past the
               bumper at ANY pitch >= 35°; pitch only sets the FAR edge.
  --cost       per-frame ms at three resolutions on both worlds, the height
               map's update, and what a voxel map of the same volume costs
               per update and in bytes -- the representation decision.
  --find       a cube of side s at distance d, robot stationary, one frame:
               found or not, and the height error. The floor sampling grows
               with distance (one row is ~26 mm of floor at 0.8 m), so the
               smallest thing seen standing still is a function of range.
  (default)    a filmstrip: the depth image, the height map after a frame,
               and what the camera sees, `nearfield_spike.png`.

Usage:
  MUJOCO_GL=egl uv run python scripts/nearfield_spike.py [--mount|--cost|--find]
"""

import argparse
import math
import sys
import time

import mujoco
import numpy as np

from pluggybot.perception import depth as depthmod
from pluggybot.perception.depth import DepthCamera
from pluggybot.perception.heightmap import CELL_M, SIZE_M, Z_MAX, HeightMap

WORLDS = ("models/room_hub.xml", "models/home_world.xml")


def true_axle(data) -> tuple[float, float, float]:
  q = data.qpos
  th = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]),
                  1 - 2 * (q[5] * q[5] + q[6] * q[6]))
  return q[0] - 0.08 * math.cos(th), q[1] - 0.08 * math.sin(th), th


def settled(path: str, pitch_deg: float | None = None, props=()):
  """A world, settled, optionally with the depth camera re-pitched and
  boxes (`(x, y, side)`, on the floor) added."""
  spec = mujoco.MjSpec.from_file(path)
  if pitch_deg is not None:
    th = math.radians(pitch_deg)
    # the model orients the camera with xyaxes; the spec keeps that as its
    # `alt` orientation, which is what has to change
    spec.camera("depth_eye").alt.xyaxes = [0.0, -1.0, 0.0,
                                            math.sin(th), 0.0, math.cos(th)]
  for i, (x, y, side) in enumerate(props):
    g = spec.worldbody.add_geom()
    g.name = f"prop_{i}"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [side / 2] * 3
    g.pos = [x, y, side / 2]
    g.rgba = [0.9, 0.6, 0.2, 1.0]
  model = spec.compile()
  data = mujoco.MjData(model)
  for _ in range(300):
    mujoco.mj_step(model, data)
  return model, data


def mount() -> None:
  print("pitch  self   floor  centre column floor band (m ahead of the axle)")
  for pitch in (30, 35, 40, 45, 50, 55):
    model, data = settled(WORLDS[0], pitch_deg=pitch)
    cam = DepthCamera(model, dropout=0.0)
    f = cam.frame(data)
    near, far = cam.floor_band(data)
    floor = (f.points[:, 2] < depthmod.FLOOR_TOL).sum() / f.z.size
    print(f"  {pitch:2d}°  {f.self_fraction:5.1%}  {floor:5.1%}  "
          f"{near:.2f} .. {far:.2f}")
  print("  bumper is 0.20 m ahead of the axle; the model pitches 40°")


def _ms(fn, k: int = 20) -> float:
  fn()
  t0 = time.perf_counter()
  for _ in range(k):
    fn()
  return (time.perf_counter() - t0) / k * 1e3


def cost() -> None:
  print("depth frame (rays + self-filter + shadow + noise), ms:")
  for path in WORLDS:
    model, data = settled(path)
    row = []
    for w, h in ((60, 35), (120, 70), (240, 140)):
      cam = DepthCamera(model, width=w, height=h)
      row.append(f"{w}x{h}: {_ms(lambda: cam.frame(data)):5.2f}")
    print(f"  {path:24s} " + "   ".join(row))
  model, data = settled(WORLDS[0])
  cam = DepthCamera(model)
  frame = cam.frame(data)
  pose = true_axle(data)
  hm = HeightMap()
  print(f"\nheight map {hm.n}x{hm.n} = {hm.cells} cells at {CELL_M} m, "
        f"{hm.height.nbytes + hm.count.nbytes} bytes: "
        f"update {_ms(lambda: hm.update(pose, frame.points)):.2f} ms "
        f"(the 2D grid's LIDAR update is ~1.3 ms, 56 000 cells at 5 cm)")
  print("voxel map of the same window, height Z_MAX, per update:")
  for cell in (0.02, 0.01):
    n = int(SIZE_M / cell)
    nz = int(Z_MAX / cell)
    grid = np.zeros((nz, n, n), dtype=np.int8)
    print(f"  {cell} m: {n}x{n}x{nz} = {grid.size:,} cells, "
          f"{grid.nbytes / 1e6:.1f} MB int8  "
          f"occupied-only {_ms(lambda: _voxel_hits(grid, pose, frame.points, cell)):6.2f} ms  "
          f"with free-space carving "
          f"{_ms(lambda: _voxel_carve(grid, pose, frame.points, cam.origin_robot, cell), 5):7.2f} ms")
  print("  (carving samples every ray at half a cell to its hit, as the 2D "
        "grid does; a voxel map that never carves cannot forget a moved "
        "object)")


def _world_points(pose, points):
  x, y, th = pose
  c, s = math.cos(th), math.sin(th)
  return np.stack([x + c * points[:, 0] - s * points[:, 1],
                   y + s * points[:, 0] + c * points[:, 1],
                   points[:, 2]], axis=1)


def _voxel_hits(grid, pose, points, cell):
  nz, n, _ = grid.shape
  p = _world_points(pose, points)
  x, y, _ = pose
  ix = np.floor((p[:, 0] - x) / cell).astype(np.int64) + n // 2
  iy = np.floor((p[:, 1] - y) / cell).astype(np.int64) + n // 2
  iz = np.floor(p[:, 2] / cell).astype(np.int64)
  ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < nz)
  flat = (iz[ok] * n + iy[ok]) * n + ix[ok]
  grid.reshape(-1)[flat] = 1


def _voxel_carve(grid, pose, points, origin, cell):
  """Every ray sampled at half a cell from the camera to just short of its
  hit, the 2D grid's method one dimension up."""
  nz, n, _ = grid.shape
  x, y, _ = pose
  p = _world_points(pose, points)
  o = _world_points(pose, origin[None, :])[0]
  vec = p - o
  length = np.linalg.norm(vec, axis=1)
  step = cell / 2
  t = np.arange(int(np.ceil(length.max() / step))) * step
  s = o + (vec / length[:, None])[:, None, :] * t[None, :, None]
  ok = (t[None, :] < (length - cell)[:, None]).reshape(-1)
  ix = np.floor((s[..., 0].reshape(-1) - x) / cell).astype(np.int64) + n // 2
  iy = np.floor((s[..., 1].reshape(-1) - y) / cell).astype(np.int64) + n // 2
  iz = np.floor(s[..., 2].reshape(-1) / cell).astype(np.int64)
  ok &= (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < nz)
  grid.reshape(-1)[((iz[ok] * n + iy[ok]) * n + ix[ok])] = 0
  _voxel_hits(grid, pose, points, cell)


def find() -> None:
  print("cube side -> found at distance d (one frame, stationary), height "
        "error mm; '-' is not found")
  sides = (0.02, 0.03, 0.05, 0.10)
  dists = (0.4, 0.6, 0.8, 1.2, 1.6)
  print("  side  " + "".join(f"{d:>9.1f} m" for d in dists))
  for side in sides:
    cells = []
    for dist in dists:
      model, data = settled(WORLDS[0], props=((0.08 + dist, 0.0, side),))
      cam = DepthCamera(model, dropout=0.0)
      hm = HeightMap()
      hm.update(true_axle(data), cam.frame(data).points)
      hit = [t for t in hm.things()
             if abs(t["x"] - (0.08 + dist)) < 0.05 and abs(t["y"]) < 0.05]
      cells.append(f"{(hit[0]['height'] - side) * 1e3:+8.0f} mm"
                   if hit else f"{'-':>11}")
    print(f"  {side * 100:3.0f} cm " + "".join(cells))
  print("  (the cube's near face is at d ahead of the body origin, 0.08 m "
        "ahead of the axle; a thing under the map's RAISED_M is floor)")


def filmstrip(out: str) -> None:
  from PIL import Image, ImageDraw
  model, data = settled(WORLDS[0], props=((0.9, 0.0, 0.05), (1.3, 0.3, 0.10),
                                          (0.7, -0.35, 0.03)))
  cam = DepthCamera(model)
  frame = cam.frame(data)
  hm = HeightMap()
  hm.update(true_axle(data), frame.points)
  h, w = frame.z.shape
  # the depth image: near bright, invalid red
  z = frame.z
  img = np.zeros((h, w, 3), dtype=np.uint8)
  ok = np.isfinite(z)
  shade = np.clip(255 * (1 - (np.nan_to_num(z, nan=0) - 0.4) / 2.6), 0, 255)
  img[ok] = shade[ok][:, None]
  img[~ok] = (200, 40, 40)
  depth_png = Image.fromarray(img).resize((w * 4, h * 4), Image.NEAREST)
  # the height map: floor grey, raised warm, unseen dark; row 0 = y_min
  hmi = np.zeros((hm.n, hm.n, 3), dtype=np.uint8) + 30
  seen = np.isfinite(hm.height)
  hmi[seen] = (120, 120, 120)
  raised = seen & (hm.height >= 0.02)
  hmi[raised, 0] = np.clip(120 + hm.height[raised] / Z_MAX * 800, 0, 255)
  hmi[raised, 1] = 80
  hmi[raised, 2] = 40
  hm_png = Image.fromarray(hmi[::-1]).resize((hm.n * 3, hm.n * 3), Image.NEAREST)
  # what the camera sees (RGB), for the reader
  renderer = mujoco.Renderer(model, height=h * 4, width=w * 4)
  renderer.update_scene(data, camera="depth_eye")
  rgb = Image.fromarray(renderer.render())
  renderer.close()
  strip = Image.new("RGB", (w * 4 * 2 + hm.n * 3 + 40, max(h * 4, hm.n * 3) + 30),
                    (255, 255, 255))
  strip.paste(rgb, (10, 30))
  strip.paste(depth_png, (w * 4 + 20, 30))
  strip.paste(hm_png, (w * 8 + 30, 30))
  draw = ImageDraw.Draw(strip)
  draw.text((10, 8), "depth_eye (RGB)", fill=(0, 0, 0))
  draw.text((w * 4 + 20, 8), "depth: near bright, invalid red", fill=(0, 0, 0))
  draw.text((w * 8 + 30, 8),
            f"height map {SIZE_M} m @ {CELL_M} m, +x right, +y up; things: "
            + ", ".join(f"{t['height'] * 100:.0f} cm" for t in hm.things()),
            fill=(0, 0, 0))
  strip.save(out)
  print(f"wrote {out}; things: {hm.things()}")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--mount", action="store_true")
  ap.add_argument("--cost", action="store_true")
  ap.add_argument("--find", action="store_true")
  ap.add_argument("--out", default="nearfield_spike.png")
  args = ap.parse_args(argv)
  if args.mount:
    mount()
  elif args.cost:
    cost()
  elif args.find:
    find()
  else:
    filmstrip(args.out)


if __name__ == "__main__":
  sys.exit(main())
