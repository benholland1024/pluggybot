"""The hub-era mission (milestone 8): explore, charge at the hub, swap a tool.

The battery-driven loop from milestone 7 with the hub as its destination:
PluggyBot explores room_hub (learning where its rack is from the fiducial
along the way), runs low, noses into the charge bay until the pogo pins
connect, charges, then runs a tool errand -- fetch the LCD module, carry it
across the room, bring it back and stow it.

Usage:
  uv run python scripts/hub_lifecycle.py --view           # watch it live
  MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py    # headless
  MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --battery-wh 0.8
"""

import argparse
import math

from pluggybot.hub.lifecycle import DEMO_CAPACITY_WH, run_demo


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--view", action="store_true", help="open the viewer")
  parser.add_argument("--fast", action="store_true",
                      help="with --view: no real-time pacing")
  parser.add_argument("--battery-wh", type=float, default=DEMO_CAPACITY_WH,
                      help="battery capacity (demo cell by default)")
  parser.add_argument("--max-sim-time", type=float, default=600.0)
  args = parser.parse_args()

  r = run_demo(start=(0.5, 3.0, math.pi / 2), view=args.view,
               realtime=not args.fast, battery_wh=args.battery_wh,
               max_sim_time=args.max_sim_time)
  if r["aborted"]:
    print("mission aborted (viewer closed)")
    return
  print()
  print(f"rack discovered by tag : {r['rack_discovered']}")
  print(f"charge cycles          : {r['charge_cycles']}")
  print(f"tool swaps             : {r['swaps_done']}")
  print(f"module stowed at end   : {r['module_stowed']}")
  print(f"battery at end         : {r['battery']:.0%}")
  print(f"chassis-contact steps  : {r['collision_steps']} (should be 0)")
  print(f"sim time               : {r['sim_time']:.1f} s")
  ok = (r["charge_cycles"] >= 1 and r["swaps_done"] == 2
        and r["module_stowed"] and r["collision_steps"] == 0)
  print("LIFECYCLE:", "OK" if ok else "INCOMPLETE")


if __name__ == "__main__":
  main()
