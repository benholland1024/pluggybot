"""Charge-approach tolerance sweep (issue #32; schuko/hub_spike's sibling).

Measures what the charge-bay approach forgives of BELIEF error -- the state
a long shift delivers, where the robot's dead-reckoned pose and its rack
belief have drifted apart from the world. The robot is placed TRULY at the
charge standoff plus a controlled error while its reckoner believes it is
exactly at the standoff, then the terminal approach runs and the verdict is
the electrical criterion, `rack_charge_contact`.

Three sweeps: lateral offset, heading offset, and the issue-78 shape (a
rack-yaw belief error, which swings the standoff sideways AND tilts the
approach heading at once -- one bad free-space look delivers ~20 deg).

Usage:
  MUJOCO_GL=egl uv run python scripts/charge_spike.py           # the fix
  MUJOCO_GL=egl uv run python scripts/charge_spike.py --blind   # before it

Measured (home_world): the blind creep -- what go_charge did before #32 --
forgives ~6 cm across, ~10 deg of heading, and dies at a 10 deg rack-yaw
belief error. The tag-measured approach (mission.charge_approach) passes
every row of all three sweeps, to 20 cm and 20 deg.
"""

import argparse
import math

import mujoco

from pluggybot.home import world as home
from pluggybot.hub.coupling import rack_charge_contact
from pluggybot.hub.lifecycle import CHARGE_APPROACH_MAX, CHARGE_CREEP
from pluggybot.hub.localize import RackPose
from pluggybot.hub.mission import HubMission, charge_standoff
from pluggybot.hub.swap import align_lift

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))


def run_one(across: float, dyaw_deg: float, blind: bool) -> dict:
  """One approach from a standoff whose belief is wrong by (across, dyaw)."""
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  sx, sy, hd = charge_standoff(TRUE_RACK)
  # true pose: the standoff shifted ACROSS the pin line (rack local +y),
  # heading rotated by dyaw; belief: exactly at the standoff
  lx, ly = -math.sin(TRUE_RACK.yaw), math.cos(TRUE_RACK.yaw)
  tx, ty = sx + lx * across, sy + ly * across
  tyaw = hd + math.radians(dyaw_deg)
  d = data
  d.qpos[0] = tx + 0.08 * math.cos(tyaw)
  d.qpos[1] = ty + 0.08 * math.sin(tyaw)
  d.qpos[2] = 0.045
  d.qpos[3:7] = [math.cos(tyaw / 2), 0, 0, math.sin(tyaw / 2)]
  lift0 = align_lift()
  d.qpos[model.joint("lift_joint").qposadr[0]] = lift0
  d.ctrl[model.actuator("lift").id] = lift0
  d.ctrl[model.actuator("arm").id] = 0.0
  d.qpos[model.joint("arm_joint").qposadr[0]] = 0.0
  mujoco.mj_forward(model, d)
  r = mission.swap.reckoner
  r.x, r.y, r.theta = sx, sy, hd
  r.update(float(d.qpos[mission.swap.left_adr]),
           float(d.qpos[mission.swap.right_adr]))
  mission._drive(1.0, 0.0, 0.0)           # settle
  if blind:
    # what go_charge did before issue #32: face + refine in the believed
    # frame (no-ops against a belief that reads perfect), then a blind creep
    mission.face(hd)
    mission.refine_standoff(sx, sy, hd)
    why = mission.swap._drive_until(
      CHARGE_APPROACH_MAX, CHARGE_CREEP, stall_stop=True,
      stop_fn=lambda: rack_charge_contact(model, data))
  else:
    why = mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
  ok = rack_charge_contact(model, data)
  mission.close()
  return {"ok": ok, "why": why}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--blind", action="store_true",
                      help="reproduce the pre-fix blind creep rows")
  args = parser.parse_args()
  label = "blind" if args.blind else "fixed"

  print(f"-- across sweep, dyaw=0 ({label}) --")
  for across in (0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.20, -0.20):
    r = run_one(across, 0.0, args.blind)
    print(f"  across={across:+.3f}: {'ok  ' if r['ok'] else 'FAIL'}"
          f" ({r['why']})", flush=True)

  print(f"-- dyaw sweep, across=0 ({label}) --")
  for dyaw in (5, -5, 10, -10, 15, -15, 20, -20):
    r = run_one(0.0, dyaw, args.blind)
    print(f"  dyaw={dyaw:+3d}deg: {'ok  ' if r['ok'] else 'FAIL'}"
          f" ({r['why']})", flush=True)

  print(f"-- rack-yaw belief error rotates the standoff ({label}) --")
  for yaw_err in (5, 10, 15, 20):
    across = 0.42 * math.sin(math.radians(yaw_err))
    r = run_one(across, yaw_err, args.blind)
    print(f"  rack_yaw_err={yaw_err:2d}deg (across={across:+.3f}): "
          f"{'ok  ' if r['ok'] else 'FAIL'} ({r['why']})", flush=True)


if __name__ == "__main__":
  main()
