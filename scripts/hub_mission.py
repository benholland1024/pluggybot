"""Hub-in-room mission demo (milestone 8): map, navigate, swap, return.

The fork robot starts across room 1, spins to seed its map, A*-navigates to
the rack (whose pose is prior knowledge -- you tell a robot where its dock
lives), fine-aligns with the fiducial servo, picks the LCD module, carries
it away, brings it back, and hangs it up.

Usage:
  uv run python scripts/hub_mission.py --view          # watch it live
  MUJOCO_GL=egl uv run python scripts/hub_mission.py   # headless, full speed
  uv run python scripts/hub_mission.py --view --fast   # viewer, no pacing
"""

import argparse
import math

from pluggybot.hub.mission import run_demo


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--view", action="store_true",
                      help="open the MuJoCo viewer and watch the mission")
  parser.add_argument("--fast", action="store_true",
                      help="with --view: run flat out instead of real time")
  parser.add_argument("--no-discover", action="store_true",
                      help="trust the boot-time prior instead of looking for "
                           "the rack tag")
  args = parser.parse_args()

  result = run_demo(start=(0.5, 3.0, math.pi / 2), view=args.view,
                    realtime=not args.fast, discover=not args.no_discover)
  if result["aborted"]:
    print("mission aborted (viewer closed)")
    return
  if result["discovered"]:
    print(f"rack     : found by its tag, {result['rack_pos_err_m'] * 1000:.0f} mm "
          f"/ {result['rack_yaw_err_deg']:.2f} deg from truth")
  else:
    print("rack     : not seen — navigated on the boot-time prior")
  print(f"picked   : {result['picked']}")
  print(f"returned : {result['returned']}")
  print(f"chassis-contact steps (should be 0): {result['collision_steps']}")
  print(f"sim time : {result['sim_time']:.1f} s")
  print("MISSION:", "OK" if result["picked"] and result["returned"] else "FAILED")


if __name__ == "__main__":
  main()
