#!/usr/bin/env python
"""Two robots in one world, from one physics loop (issue #167, M12).

The quadruped pair in the house (`home_quad`, issue #387): each robot runs
its own day -- explores, docks when its pack runs low, runs its queue --
against one model, one dock, one rack and one task board, ticked in turn
from `tick.run_many`. `--view` watches it live. `--world home_humanoid` is
the humanoid pair in the same house (issue #559): no dock, rack or tools,
nowhere to charge yet, a day of mapping the house on its own legs.

    MUJOCO_GL=egl uv run python scripts/two_robots.py --view
    ... --errands feed,none        # the first walks to the lab and feeds
    ... --overseer --tasks --thoughts /tmp/two   # two minds, one job board
    ... --world home_humanoid --view --pack hosting   # the humanoid pair's day
    ... --world home_humanoid --fast --pack hosting --strip two_robots.png
                                   # ...headless, a tile of each every 20 s
"""

import argparse
import json
import math

from pluggybot.lifecycle import HUMANOID_HOME, QUAD_HOME, world_for
from pluggybot.pair import run_demo_pair

#: A filmstrip's tile, px, and its row.
TILE_W, TILE_H, ROW = 400, 300, 4


def strip_hook(lives, every_s: float, tiles: list):
  """A step hook that renders each robot from above and behind every
  `every_s` of sim time, labelled with its name and what it is doing --
  brightened for its own tiles alone, as the robots' cameras render in the
  house's light. Returns the hook and the renderer to close."""
  import mujoco
  from PIL import Image, ImageDraw
  model, data = lives[0].model, lives[0].data
  r = mujoco.Renderer(model, TILE_H, TILE_W)
  due = [0.0]

  def hook() -> None:
    if data.time < due[0]:
      return
    due[0] += every_s
    light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.4, 0.6
    for life in lives:
      x, y, yaw = life.body.true_pose()
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[model.body(life.root).id]
      cam.distance, cam.azimuth, cam.elevation = 6.0, math.degrees(yaw) + 180, -65
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      words = (f"{data.time:.0f} s {life.robot_name}: {life.state.lower()}"
               f"{', dead' if life.dead else ''}, falls {life.body.falls}")
      ImageDraw.Draw(img).text((8, 8), words[:60], fill=(255, 255, 255))
      tiles.append(img)
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
  return hook, r


def save_strip(tiles: list, out: str) -> None:
  from PIL import Image
  strip = Image.new("RGB", (TILE_W * min(ROW, len(tiles)),
                            TILE_H * ((len(tiles) + ROW - 1) // ROW)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (TILE_W * (k % ROW), TILE_H * (k // ROW)))
  strip.save(out)
  print(f"wrote {out} ({len(tiles)} tiles)")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", choices=(QUAD_HOME, HUMANOID_HOME, "home"), default=QUAD_HOME,
                  help="the house with the quadruped pair, its dock and rack "
                       "(issue #387; `home` names it too), or with the humanoid "
                       "pair (#559)")
  ap.add_argument("--strip", default=None, metavar="PNG",
                  help="a filmstrip of the day: each robot from above and behind "
                       "every --strip-every sim seconds (use with --fast)")
  ap.add_argument("--strip-every", type=float, default=20.0, metavar="S")
  ap.add_argument("--view", action="store_true", help="open the viewer")
  ap.add_argument("--fast", action="store_true", help="no real-time pacing")
  ap.add_argument("--max-sim-time", type=float, default=300.0)
  ap.add_argument("--pack", choices=("demo", "hosting"), default="demo")
  ap.add_argument("--errands", default="none,none", metavar="A,B",
                  help="each robot's preset queue (`lifecycle.errands_for`): "
                       "none, care, care:<act>, feed or shock")
  ap.add_argument("--boards", default=None, metavar="PATH")
  ap.add_argument("--overseer", action="store_true",
                  help="two MINDS (issue #167 slice C): each robot its own "
                       "overseer, thought files, wallet and library, with the "
                       "rails off and its own standing orders")
  ap.add_argument("--tasks", action="store_true", help="one shared task board")
  ap.add_argument("--metabolism", action="store_true",
                  help="points are food, for both robots (each its own "
                       "appetite over its own account)")
  ap.add_argument("--near-field", action="store_true",
                  help="each robot's depth camera builds its own height "
                       "map, streamed beside its grid (issue #34); the other "
                       "robot is dropped from each frame")
  ap.add_argument("--thoughts", default=None, metavar="DIR",
                  help="thought-file root; the second robot's live under "
                       "<DIR>/r2_pluggybot/")
  ap.add_argument("--record", default=None, metavar="PATH",
                  help="record both robots' stream (protocol fixtures: "
                       "protocol/telemetry.<world>_pair.jsonl.gz)")
  ap.add_argument("--battery", default=None, metavar="A,B",
                  help="each robot's pack at the start, as a fraction (the "
                       "quadruped fixture starts the first low, so it docks)")
  ap.add_argument("--names", default=None, metavar="A,B",
                  help="the two display names (default Pluggy,Rowan; "
                       "$PLUGGY_ROBOT_NAME_2 sets the second)")
  args = ap.parse_args()
  errands = tuple(args.errands.split(","))
  if len(errands) != 2:
    ap.error("--errands takes two names, one per robot")
  names = tuple(args.names.split(",")) if args.names else None

  tiles, renderer = [], []

  def on_ready(lives):
    if args.battery:
      for life, frac in zip(lives, (float(f) for f in args.battery.split(","))):
        life.battery.energy_wh = life.battery.capacity_wh * frac
    if args.strip:
      hook, r = strip_hook(lives, args.strip_every, tiles)
      lives[0].body.step_hooks.append(hook)
      renderer.append(r)
  world = world_for(args.world)
  results = run_demo_pair(world=world, max_sim_time=args.max_sim_time,
                          view=args.view, realtime=not args.fast, pack=args.pack,
                          errands=errands, board_state=args.boards,
                          overseer=args.overseer or None, tasks=args.tasks,
                          metabolism=args.metabolism,
                          near_field=args.near_field,
                          thoughts_root=args.thoughts, names=names,
                          record=args.record, on_ready=on_ready)
  if args.strip:
    renderer[0].close()
    save_strip(tiles, args.strip)
  for i, r in enumerate(results, 1):
    print(f"robot {i}: {r['state']} at {r['battery']:.0%}, "
          f"{r['charge_cycles']} charge(s), {r['swaps_done']} swaps, "
          f"{len(r['errands'])} errand(s), dead={r['dead']}")
  print(json.dumps([{k: r[k] for k in ("state", "battery", "swaps_done",
                                        "sim_time", "dead")} for r in results]))


if __name__ == "__main__":
  main()
