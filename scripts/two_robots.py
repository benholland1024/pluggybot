#!/usr/bin/env python
"""Two robots in one world, from one physics loop (issue #167, M12).

Both robots run their own day -- the first fetches a tool and carries it,
the second explores -- against one model, one rack and one whiteboard book,
ticked in turn from `tick.run_many`. `--view` watches it live.

    MUJOCO_GL=egl uv run python scripts/two_robots.py --world room_hub --view
    ... --errands carry,carry      # both fetch the LCD: the first contested bay
"""

import argparse
import json

from pluggybot.pair import run_demo_pair


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", choices=("room_hub", "home"), default="room_hub")
  ap.add_argument("--view", action="store_true", help="open the viewer")
  ap.add_argument("--fast", action="store_true", help="no real-time pacing")
  ap.add_argument("--max-sim-time", type=float, default=300.0)
  ap.add_argument("--pack", choices=("demo", "hosting"), default="demo")
  ap.add_argument("--errands", default="carry,none", metavar="A,B",
                  help="each robot's preset queue: carry, draw, census, "
                       "dance, none")
  ap.add_argument("--boards", default=None, metavar="PATH")
  args = ap.parse_args()
  errands = tuple(args.errands.split(","))
  if len(errands) != 2:
    ap.error("--errands takes two names, one per robot")
  results = run_demo_pair(world=args.world, max_sim_time=args.max_sim_time,
                          view=args.view, realtime=not args.fast, pack=args.pack,
                          errands=errands, board_state=args.boards)
  for i, r in enumerate(results, 1):
    print(f"robot {i}: {r['state']} at {r['battery']:.0%}, "
          f"{r['swaps_done']} swaps, {len(r['errands'])} errand(s), "
          f"{r['collision_steps']} collision steps, dead={r['dead']}")
  print(json.dumps([{k: r[k] for k in ("state", "battery", "swaps_done",
                                        "sim_time", "dead")} for r in results]))


if __name__ == "__main__":
  main()
