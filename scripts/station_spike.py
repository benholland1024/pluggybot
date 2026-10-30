"""The humanoid's charging station (issue #559, stage 2; SimNotes, "The
humanoid's charging station"), measured as #378's dock was: the humanoid in
the house, its station on the living room's wall.

    MUJOCO_GL=egl uv run python scripts/station_spike.py            # filmstrip
    ... --approach [--n N] [--parallel P]   # from round the standoff: docked
                                # and unplugged or why not, tries, steps, time
    ... --hold [--seconds S]    # plugged in: the pack's gain, W, and the plug held
    ... --fall                  # shoved over plugged in: the breakaway, the
                                # plug home, the robot up again
    ... --view [plug|fall]      # a scene in the viewer, looped (MUJOCO_GL unset)
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

#: A filmstrip's tile, px, and its row.
TILE_W, TILE_H, ROW = 480, 360, 4
#: The viewer's camera a scene, fixed from the robot's right (`side`): the
#: point it looks at in the station's frame (m) and how far off it stands --
#: the fall's wide enough for where the shove throws the robot, 1.8 m out
#: and 0.7 m across, and the plug's flight home. Then its elevation, deg,
#: and the physics steps between the viewer's frames.
VIEW_CAM = {"plug": ((-0.45, 0.075, 0.75), 2.0), "fall": ((-0.95, 0.35, 0.6), 3.2)}
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
  from pluggybot.lifecycle import HUMANOID_HOME, home_activities, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], body="humanoid").compile()
  data = mujoco.MjData(model)
  body = body_for(model, data)
  s = body._station
  x0 = st.SEAT_X if at_seat else st.STANDOFF_X
  x, y = s.to_world(x0 - off[0], st.SEAT_Y + off[1])
  body.start_at(x, y, s.yaw + math.radians(off[2]))
  acts = home_activities(model, data)
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
    cmd = [sys.executable, __file__, "--one", json.dumps(off)]
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


def strip(scene: str = "plug", out: str = "station_spike.png") -> None:
  """The scene headless, a tile every 1.5 s from beside the station."""
  from PIL import Image, ImageDraw
  model, data, body, station = world()
  r = mujoco.Renderer(model, TILE_H, TILE_W)
  tiles, due = [], [0.0]
  s = body._station

  def hook():
    if data.time < due[0]:
      return
    due[0] = float(data.time) + 1.5
    cam = mujoco.MjvCamera()
    lx, ly = s.to_world(-0.25, 0.0)
    cam.lookat[:] = (lx, ly, 0.85)
    cam.distance, cam.azimuth, cam.elevation = 1.4, side(s), -12.0
    light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.5, 0.6
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
    f = station.flags
    ImageDraw.Draw(img).text((6, 6), f"{data.time:.1f} s  plugged in: {bool(f['mated'])}  "
                             f"parked: {f['parked']}", fill=(255, 255, 255))
    tiles.append(img)

  body.step_hooks.append(hook)
  why = body.run(body.dock_routine())
  body.run(body.dock_hold_routine(2.0))
  if scene == "plug" and why == "docked":
    body.run(body.undock_routine())
    body.run(body.hold_routine(3.0))
  r.close()
  w, h = tiles[0].size
  img = Image.new("RGB", (w * min(ROW, len(tiles)), h * ((len(tiles) + ROW - 1) // ROW)))
  for k, t in enumerate(tiles):
    img.paste(t, (w * (k % ROW), h * (k // ROW)))
  img.save(out)
  print(f"{why}; wrote {out} ({len(tiles)} tiles)")


#: What each viewer scene shows, said in the terminal as it starts.
SCENES = {
  "plug": "the `plug` scene: the humanoid walks in from its standoff, fits the station "
          "off its tags, plugs itself in, charges 3 s and unplugs.",
  "fall": "the `fall` scene: the humanoid plugs itself in as in `plug`, then is shoved "
          "over backward; the plug breaks away, its balancer takes it home, and the "
          "robot gets up.",
}


def view(scene: str) -> None:
  """A scene in the viewer at real time, again and again until the window
  closes, each from the standoff: `plug` docks, charges and unplugs; `fall`
  docks and is shoved over plugged in. (From the seat itself, `fall()`'s
  start, the plug-in took four tries and 73 s: SimNotes.)"""
  import time
  import mujoco.viewer
  print(SCENES[scene] + " Close the window to quit.")
  while True:
    model, data, body, station = world()
    with mujoco.viewer.launch_passive(model, data) as v:
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
  ap.add_argument("--view", nargs="?", const="plug", choices=("plug", "fall"))
  ap.add_argument("--out", default=None)
  ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one:
    print(json.dumps(trial(tuple(json.loads(args.one)))))
    return 0
  if args.view:
    view(args.view)
    return 0
  rows = (approach(args.n, args.parallel) if args.approach else
          [hold(args.seconds)] if args.hold else [fall()] if args.fall else None)
  if rows is None:
    strip()
    return 0
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
