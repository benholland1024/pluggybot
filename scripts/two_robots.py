#!/usr/bin/env python
"""Two robots in one world, from one physics loop (issue #167, M12).

The quadruped pair in the house (`home_quad`, issue #387): each robot runs
its own day -- explores, docks when its pack runs low, runs its queue --
against one model, one dock, one rack and one task board, ticked in turn
from `tick.run_many`. `--view` watches it live.

    MUJOCO_GL=egl uv run python scripts/two_robots.py --view
    ... --errands feed,none        # the first walks to the lab and feeds
    ... --overseer --tasks --thoughts /tmp/two   # two minds, one job board
"""

import argparse
import json

from pluggybot.lifecycle import QUAD_HOME, world_for
from pluggybot.pair import run_demo_pair


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", choices=(QUAD_HOME, "home"), default=QUAD_HOME,
                  help="the house with the quadruped pair, its dock and rack "
                       "(issue #387); `home` names it too")
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
                       "overseer, thought files, wallet and library")
  ap.add_argument("--autonomous", action="store_true",
                  help="both minds on the autonomous arm (rails off, standing "
                       "orders, procedures)")
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

  def on_ready(lives):
    if args.battery:
      for life, frac in zip(lives, (float(f) for f in args.battery.split(","))):
        life.battery.energy_wh = life.battery.capacity_wh * frac
  results = run_demo_pair(world=world_for(args.world), max_sim_time=args.max_sim_time,
                          view=args.view, realtime=not args.fast, pack=args.pack,
                          errands=errands, board_state=args.boards,
                          overseer=args.overseer or None,
                          autonomous=args.autonomous,
                          standing_orders=args.autonomous, tasks=args.tasks,
                          metabolism=args.metabolism,
                          near_field=args.near_field,
                          thoughts_root=args.thoughts, names=names,
                          record=args.record, on_ready=on_ready)
  for i, r in enumerate(results, 1):
    print(f"robot {i}: {r['state']} at {r['battery']:.0%}, "
          f"{r['charge_cycles']} charge(s), {r['swaps_done']} swaps, "
          f"{len(r['errands'])} errand(s), dead={r['dead']}")
  print(json.dumps([{k: r[k] for k in ("state", "battery", "swaps_done",
                                        "sim_time", "dead")} for r in results]))


if __name__ == "__main__":
  main()
