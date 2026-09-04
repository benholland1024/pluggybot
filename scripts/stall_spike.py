"""Stalled-drive odometry sweep (issue #94; the charge press's sibling).

Dead reckoning integrates wheel rotation, and a wheel that is slipping
against something immovable rotates just the same: an ordinary drive
stalled against the garden fence pumped 4.28 m of travel that never happened
into the pose in 30 s, and `drive_to` then "arrived" at a point the robot
never reached. `HubSwap.pinned` guarded the ONE press that was declared (the
charge cycle); `HubSwap.pressing` is the same rule sensed from the bumper --
a chassis contact on the side the wheels are driving toward -- and it holds
the reckoner's travel for as long as the press lasts.

Three tables:

  * the PUMP: robot placed in the garden, driven into the east fence for 30 s
    at each of the mission's speeds -- believed travel minus true travel,
    with the bumper rule and (`--blind`) without it;
  * the EFFORT, which is why the signal is the bumper and not the motor: a
    velocity servo pressed against a fence sits at cruise torque (the tyres
    slip long before a 1.8 kg robot's motors saturate);
  * the OUTCOME: `drive_to` a point behind a knee-high box the lidar looks
    straight over. Blind, it reports success from a metre short.

Usage:
  MUJOCO_GL=egl uv run python scripts/stall_spike.py           # the fix
  MUJOCO_GL=egl uv run python scripts/stall_spike.py --blind   # before it
"""

import argparse
import math

import mujoco
import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.home import world as home
from pluggybot.lifecycle import CHARGE_PRESS
from pluggybot.mission.mission import HubMission
from pluggybot.rack.localize import RackPose
from pluggybot.rack.swap import APPROACH_V, HubSwap

TRUE_RACK = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                     math.radians(home.HOME_RACK_YAW))
#: The garden's east fence stands at x = 10; this line is clear of the gate.
FENCE_RUN = ((9.0, 0.0, 0.0), (14.0, 0.0))
LOW_BOX = dict(pos=(9.0, 0.0, 0.075), size=(0.15, 0.4, 0.075))


def true_axle(data) -> tuple[float, float, float]:
  q = data.qpos
  th = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]),
                  1 - 2 * (q[5] * q[5] + q[6] * q[6]))
  return q[0] - 0.08 * math.cos(th), q[1] - 0.08 * math.sin(th), th


def home_model(low_box: bool = False):
  """The home world, optionally with a box the scan plane looks over."""
  if not low_box:
    return mujoco.MjModel.from_xml_path("models/home_world.xml")
  spec = mujoco.MjSpec.from_file("models/home_world.xml")
  g = spec.worldbody.add_geom()
  g.name, g.type = "low_box", mujoco.mjtGeom.mjGEOM_BOX
  g.size, g.pos = list(LOW_BOX["size"]), list(LOW_BOX["pos"])
  return spec.compile()


def new_mission(low_box: bool = False) -> HubMission:
  model = home_model(low_box)
  data = mujoco.MjData(model)
  return HubMission(model, data, viewer=None, realtime=False,
                    rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)


def press_run(v: float | None, seconds: float = 30.0) -> dict:
  """Drive into the fence for `seconds`: `v` fixed, or None for the
  navigation law's own speed toward a point beyond it."""
  mission = new_mission()
  model, data = mission.model, mission.data
  la = model.actuator("left_motor").id
  ra = model.actuator("right_motor").id
  effort_free, effort_pressed = [], []
  mission.step_hooks.append(lambda: (
    effort_pressed if mission.swap.press_steps else effort_free).append(
      max(abs(data.actuator_force[la]), abs(data.actuator_force[ra]))))
  start, target = FENCE_RUN
  mission.start_at(*start)
  tx0, ty0, _ = true_axle(data)
  bx0, by0, _ = mission.pose
  t0 = data.time
  contact_at = None
  while data.time - t0 < seconds:
    vv, w = (drive_toward(mission.pose, target) if v is None else (v, 0.0))
    mission._drive(model.opt.timestep, vv, w)
    if contact_at is None and mission.collision_steps:
      contact_at = data.time - t0
  tx, ty, _ = true_axle(data)
  bx, by, _ = mission.pose
  out = {
    "true_m": math.hypot(tx - tx0, ty - ty0),
    "believed_m": math.hypot(bx - bx0, by - by0),
    "contact_at": contact_at,
    "press_steps": mission.swap.press_steps,
    "effort_free": np.percentile(effort_free, 50) if effort_free else 0.0,
    "effort_pressed": (np.percentile(effort_pressed, 50)
                       if effort_pressed else float("nan")),
  }
  mission.close()
  return out


def low_box_run() -> dict:
  """`drive_to` a point behind the knee-high box."""
  mission = new_mission(low_box=True)
  mission.start_at(8.0, 0.0, 0.0)
  mission._spin()
  t0 = mission.data.time
  arrived = mission.drive_to(10.0, 0.0, timeout=60.0)
  tx, ty, _ = true_axle(mission.data)
  bx, by, _ = mission.pose
  out = {"arrived": arrived, "seconds": mission.data.time - t0,
         "true": (tx, ty), "believed": (bx, by),
         "belief_error_m": math.hypot(bx - tx, by - ty),
         "press_steps": mission.swap.press_steps}
  mission.close()
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--blind", action="store_true",
                      help="reproduce the pre-fix rows: no bumper rule")
  args = parser.parse_args()
  if args.blind:
    HubSwap._pressing = lambda self: False
  label = "blind" if args.blind else "bumper rule"

  print(f"-- pump: 30 s into the fence ({label}) --")
  for name, v in (("drive_toward (V_MAX)", None), ("0.30 m/s", 0.30),
                  ("0.12 m/s creep", 0.12), ("APPROACH_V", APPROACH_V),
                  ("CHARGE_PRESS", CHARGE_PRESS)):
    r = press_run(v)
    contact = ("no contact" if r["contact_at"] is None
               else f"contact at {r['contact_at']:5.1f} s")
    print(f"  {name:22s} true {r['true_m']:.3f} m  believed "
          f"{r['believed_m']:.3f} m  PUMP {r['believed_m'] - r['true_m']:+.3f} m"
          f"  ({contact}, press_steps {r['press_steps']})", flush=True)
    if v is None:
      print(f"  {'':22s} motor |torque| p50: free {r['effort_free']:.2f} N m,"
            f" pressed {r['effort_pressed']:.2f} N m (limit 2.06)")

  print(f"-- outcome: drive_to behind a 0.15 m box ({label}) --")
  r = low_box_run()
  print(f"  arrived={r['arrived']} after {r['seconds']:.1f} s; true "
        f"({r['true'][0]:.2f}, {r['true'][1]:.2f}) believed "
        f"({r['believed'][0]:.2f}, {r['believed'][1]:.2f}); belief error "
        f"{r['belief_error_m']:.3f} m; press_steps {r['press_steps']}")


if __name__ == "__main__":
  main()
