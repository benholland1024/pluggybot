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
  MUJOCO_GL=egl uv run python scripts/swap_spike.py --yaw     # issue #88:
                        # the measured standoff's error at thirteen TRUE
                        # poses 2 mm apart, bay and charge, with the facing
                        # from one tag's PnP yaw and from the fit over
                        # every rack tag in view. No physics, just looks.
"""

import argparse
import math

import mujoco

from pluggybot.home import world as home
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.localize import RackPose
from pluggybot.rack.swap import PLUG_LATERAL, STANDOFF, align_lift
from pluggybot.mission.mission import (
  CHARGE_LOOK_LIFT, HubMission, charge_standoff,
)

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


def true_axle(mission) -> tuple[float, float, float]:
  """The robot's TRUE axle pose off qpos (the chassis origin rides 8 cm
  ahead of it)."""
  q = mission.data.qpos
  th = math.atan2(2 * q[3] * q[6], 1 - 2 * q[6] * q[6])
  return q[0] - 0.08 * math.cos(th), q[1] - 0.08 * math.sin(th), th


def fix_error(mission, fix, true_standoff) -> tuple[float, float]:
  """(m, deg) between a believed-frame fix and the TRUE standoff, with the
  fix re-based from the believed pose on to the true one -- so the number is
  the MEASUREMENT's error, with none of the belief's in it."""
  fx, fy, fhd = fix
  bx, by, bth = mission.pose
  tx, ty, tth = true_axle(mission)
  dx, dy = fx - bx, fy - by
  rx = dx * math.cos(bth) + dy * math.sin(bth)
  ry = -dx * math.sin(bth) + dy * math.cos(bth)
  px = tx + rx * math.cos(tth) - ry * math.sin(tth)
  py = ty + rx * math.sin(tth) + ry * math.cos(tth)
  tsx, tsy, tshd = true_standoff
  dh = (tth + fhd - bth) - tshd
  return (math.hypot(px - tsx, py - tsy),
          math.degrees(math.atan2(math.sin(dh), math.cos(dh))))


def one_tag_only(mission, tag_id: int):
  """Hide every other tag from the fix -- what it saw before issue #88,
  when the facing came off this one tag's PnP yaw."""
  real = mission.tags.detect
  return lambda data: {k: v for k, v in real(data).items() if k == tag_id}


def yaw_sweep(mission, label: str, fix_fn, tag_id: int, true_standoff,
              qpos_adrs) -> None:
  """Thirteen TRUE poses 2 mm apart across the approach, belief untouched:
  the standoff error from one tag's yaw and from the multi-tag fit."""
  model, data = mission.model, mission.data
  q0, v0 = data.qpos.copy(), data.qvel.copy()
  lx, ly = -math.sin(TRUE_RACK.yaw), math.cos(TRUE_RACK.yaw)
  real_detect = mission.tags.detect
  print(f"-- {label}: nudge | one tag: yaw  err_m  err_deg | fit: source "
        f"err_m  err_deg --")
  worst_one = worst_fit = (0.0, 0.0)
  for k in range(13):
    data.qpos[:] = q0
    data.qvel[:] = v0
    for adr in qpos_adrs:
      data.qpos[adr] += lx * 0.002 * k
      data.qpos[adr + 1] += ly * 0.002 * k
    mujoco.mj_forward(model, data)
    yaw = math.degrees(real_detect(data)[tag_id]["yaw"])
    mission.tags.detect = one_tag_only(mission, tag_id)
    try:
      one = fix_fn()
    finally:
      mission.tags.detect = real_detect
    fit = fix_fn()
    if one is None or fit is None:
      print(f"  {2 * k:+3d} mm: no decode")
      continue
    e1, e2 = fix_error(mission, one, true_standoff), \
        fix_error(mission, fit, true_standoff)
    worst_one = max(worst_one, e1)
    worst_fit = max(worst_fit, e2)
    print(f"  {2 * k:+3d} mm | {yaw:+6.2f} {e1[0]:.4f} {e1[1]:+6.2f} | "
          f"{mission.fix_source:8s} {e2[0]:.4f} {e2[1]:+6.2f}", flush=True)
  print(f"  worst: one tag {worst_one[0]:.4f} m / {worst_one[1]:+.2f} deg, "
        f"fit {worst_fit[0]:.4f} m / {worst_fit[1]:+.2f} deg")
  data.qpos[:] = q0
  data.qvel[:] = v0
  mujoco.mj_forward(model, data)


def yaw_main() -> None:
  from pluggybot.rack.coupling import CHARGE_TAG_ID, bay_tag_id
  station = HUB_STATION_YS[0]
  # the bay: the carrying fixture tests/test_swap_approach.py uses
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  sx, sy, hd = TRUE_RACK.bay_standoff(station, 1.2, 0.05)
  mission.start_at(sx, sy, hd)
  mission.swap_at_bay(station, "pick", module=MODULE)
  assert mission.swap.module_state(MODULE)["on_fork"]
  mission.swap._drive_until(0.6, -0.12, stall_stop=False)
  robot_adr = model.jnt_qposadr[
    model.body(model.geom("chassis").bodyid[0]).jntadr[0]]
  module_adr = model.jnt_qposadr[model.body(MODULE).jntadr[0]]
  yaw_sweep(mission, "bay A return standoff",
            lambda: mission.bay_fix(station), bay_tag_id(station),
            TRUE_RACK.bay_standoff(station, STANDOFF, PLUG_LATERAL),
            (robot_adr, module_adr))
  mission.close()
  # the charge bay: placed at its standoff, camera down at tag height
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False,
                       rack=TRUE_RACK, grid_bounds=home.GRID_BOUNDS)
  sx, sy, hd = charge_standoff(TRUE_RACK)
  data.qpos[0] = sx + 0.08 * math.cos(hd)
  data.qpos[1] = sy + 0.08 * math.sin(hd)
  data.qpos[2] = 0.045
  data.qpos[3:7] = [math.cos(hd / 2), 0, 0, math.sin(hd / 2)]
  lift0 = align_lift()
  data.qpos[model.joint("lift_joint").qposadr[0]] = lift0
  data.ctrl[model.actuator("lift").id] = lift0
  mujoco.mj_forward(model, data)
  r = mission.swap.reckoner
  r.x, r.y, r.theta = sx, sy, hd
  r.update(float(data.qpos[mission.swap.left_adr]),
           float(data.qpos[mission.swap.right_adr]))
  mission._drive(1.0, 0.0, 0.0)
  mission.swap._run(1.5, 0.0, lift_target=CHARGE_LOOK_LIFT)
  yaw_sweep(mission, "charge standoff", mission.charge_bay_fix,
            CHARGE_TAG_ID, charge_standoff(TRUE_RACK), (robot_adr,))
  mission.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--yaw", action="store_true",
                      help="issue #88: standoff error across 2 mm true-pose "
                           "nudges, one-tag yaw vs the multi-tag fit")
  parser.add_argument("--blind", action="store_true",
                      help="reproduce the pre-fix believed-standoff rows")
  parser.add_argument("--pick", action="store_true",
                      help="corrupt the belief before the PICK instead of "
                           "before the return")
  args = parser.parse_args()
  if args.yaw:
    yaw_main()
    return
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
