"""Bay-swap tolerance sweep under BELIEF error (issue #30; charge_spike's
sibling).

The long-run tool drop, reproduced deterministically: the robot picks a
module cleanly, its dead-reckoned belief is then corrupted by `across`
metres and `dyaw` degrees -- the state a long shift's drift decoheres it
into -- and it attempts the return. Where the module ENDS UP is the verdict:
hung on its bay, still on the fork, or on the floor at the rack's foot,
where every later pick finds an empty bay and the approach lane is littered.

Measured on the blind rows (what swap_at_bay did before #30): 0-2 cm hangs,
**4-8 cm drops the module on the FLOOR**, 10 cm misses the trays entirely
and keeps it on the fork. The tray V forgives +/-8 mm, `refine_standoff`
kills BELIEVED lateral only, and the tag servo's authority over the 0.22 m
creep is ~1-2 cm -- so a few cm of real error sails through all three. The
measured bay standoff (`HubMission.bay_fix`, the #32 medicine one bay over)
re-anchors the approach to the bay's own tag and hangs the module at every
row of the sweep.

Usage:
  MUJOCO_GL=egl uv run python scripts/swap_spike.py           # the fix
  MUJOCO_GL=egl uv run python scripts/swap_spike.py --blind   # before it
  MUJOCO_GL=egl uv run python scripts/swap_spike.py --pick    # corrupt the
                        # belief BEFORE the pick instead: an empty-fork
                        # failure rather than a drop, same fix
"""

import argparse
import math

import mujoco

from pluggybot.home import world as home
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.localize import RackPose
from pluggybot.mission.mission import HubMission

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))
MODULE = "module_lcd"


def decohere(mission, across: float, dyaw_deg: float) -> None:
  """Shift the reckoner relative to truth -- what a long shift does."""
  r = mission.swap.reckoner
  lx, ly = -math.sin(TRUE_RACK.yaw), math.cos(TRUE_RACK.yaw)
  r.x += lx * across
  r.y += ly * across
  r.theta += math.radians(dyaw_deg)


def module_fate(mission) -> str:
  st = mission.swap.module_state(MODULE)
  z = float(mission.data.xpos[int(mission.model.body(MODULE).id)][2])
  return ("hung" if st["hung"] else
          "on_fork" if st["on_fork"] else
          "FLOOR" if z < 0.10 else f"loose@z={z:.3f}")


def run_one(across: float, dyaw_deg: float, blind: bool,
            corrupt_pick: bool) -> dict:
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  if blind:
    # the pre-#30 approach: trust the believed standoff, measure nothing
    mission.bay_fix = lambda *a, **k: None
  station = HUB_STATION_YS[0]
  sx, sy, hd = TRUE_RACK.bay_standoff(station, 1.2, 0.05)
  mission.start_at(sx, sy, hd)
  if corrupt_pick:
    decohere(mission, across, dyaw_deg)
  mission.swap_at_bay(station, "pick", module=MODULE)
  picked = mission.swap.module_state(MODULE)["on_fork"]
  if not picked:
    fate = module_fate(mission)
    mission.close()
    return {"picked": False, "fate": fate}
  # a mini errand away from the rack, then the decoherence (for the return
  # case: the drift accrues while the robot is out working)
  mission.swap._drive_until(0.6, -0.12, stall_stop=False)
  if not corrupt_pick:
    decohere(mission, across, dyaw_deg)
  mission.swap_at_bay(station, "return", module=MODULE)
  fate = module_fate(mission)
  mission.close()
  return {"picked": True, "fate": fate}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--blind", action="store_true",
                      help="reproduce the pre-fix believed-standoff rows")
  parser.add_argument("--pick", action="store_true",
                      help="corrupt the belief before the PICK instead of "
                           "before the return")
  args = parser.parse_args()
  label = ("blind" if args.blind else "fixed") + \
          (", pick corrupted" if args.pick else ", return corrupted")

  print(f"-- across sweep, dyaw=0 ({label}) --")
  for across in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
    r = run_one(across, 0.0, args.blind, args.pick)
    print(f"  across={across:+.3f}: picked={str(r['picked']):5s} "
          f"module={r['fate']}", flush=True)

  print(f"-- dyaw sweep, across=0 ({label}) --")
  for dyaw in (3, -3, 6, -6, 10, -10):
    r = run_one(0.0, dyaw, args.blind, args.pick)
    print(f"  dyaw={dyaw:+3d}deg: picked={str(r['picked']):5s} "
          f"module={r['fate']}", flush=True)


if __name__ == "__main__":
  main()
