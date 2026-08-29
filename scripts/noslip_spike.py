"""Single always-on noslip policy sweep (issue #3; hub_spike's sibling).

PluggyWorld puts two robots in ONE world, so solver settings can no longer be
phase-scoped per robot: whatever `noslip_iterations` is, it is for everyone,
all the time. This spike measures every behavior with a stake in that number
under each candidate policy, in one table:

  coupling   pick-and-return seat rate on the bare rig, across the measured
             misalignment envelope -- the peg seats by SLIDING into its V,
             which is exactly what the noslip pass exists to suppress
  schuko     plug insertion on the bare rig -- the other slide-to-seat
             interface in the repo, and the one the issue list forgot
  robot      full robot-driven pick (powered?) and return (hung? bay error)
             in hub_world, with hand-off jitter
  grip       block creep in the claw's jaws over a timed hold (the failure
             the noslip pass was originally brought in for, since cured at
             the source by GRIP_SOLIMP)
  pen        square traced on the board: ink %, form error (the behavior the
             per-phase toggle existed for -- its creep turned out to be the
             parked base ROLLING on unbraked wheels, see VERDICT below)
  cost       ms per mj_step in room_hub (the shared-world floor plan)

The plotter's own per-phase toggle is neutralized during the pen stage, so
each row measures the GLOBAL policy and nothing else.

VERDICT (recorded in docs/SimNotes.md): noslip_iterations = 0, always,
everywhere. Always-on noslip measured worse than useless -- at >=1 iteration
the jittered robot-level coupling half-seats (on the fork but not powered).
And the pen's loss turned out not to be solver drift at all: the parked base
was ROLLING under pen drag, because a wheel velocity servo commanded 0
resists speed, not force. The fix is the parking brake the real gearbox has
anyway -- `frictionloss` on the wheel joints (pluggybot_fork.xml); square
63 % -> 99 % inked at noslip 0. `--no-brake` zeroes it to reproduce the
before-fix rows.

Usage:
  MUJOCO_GL=egl uv run python scripts/noslip_spike.py
  MUJOCO_GL=egl uv run python scripts/noslip_spike.py --policies 0 3 --quick
  MUJOCO_GL=egl uv run python scripts/noslip_spike.py --no-brake
"""

import argparse
import time

import mujoco

from pluggybot.docking.schuko import run_trial
from pluggybot.rack.coupling import (
  HUB_STATION_YS, module_power_contact, run_cycle,
)
from pluggybot.tools.drawing import PenPlotter, PEN_MODULE, square_path
from pluggybot.tools.gripper import CARRY_LIFT, ClawTool, CLAW_MODULE
from pluggybot.rack.swap import HubSwap

HUB_WORLD = "models/hub_world.xml"
ROOM_HUB = "models/room_hub.xml"
NO_BRAKE = False          # set by --no-brake; see docstring


def _release_brake(model) -> None:
  """Zero the wheel joints' frictionloss (the before-fix state)."""
  for name in ("left_wheel_joint", "right_wheel_joint"):
    model.dof_frictionloss[model.joint(name).dofadr[0]] = 0.0


def load_hub(noslip: int):
  model = mujoco.MjModel.from_xml_path(HUB_WORLD)
  model.opt.noslip_iterations = noslip
  if NO_BRAKE:
    _release_brake(model)
  return model, mujoco.MjData(model)


# ---- bare rigs --------------------------------------------------------------

def rig_coupling(noslip: int, quick: bool) -> dict:
  """Pick-and-return cycles across the measured envelope, on the spike rig."""
  offsets = ([{}, {"dy": 0.004}, {"yaw_deg": 2.0}] if quick else
             [{}, {"dy": -0.004}, {"dy": 0.004}, {"dy": -0.002},
              {"dy": 0.002}, {"yaw_deg": -2.0}, {"yaw_deg": 2.0}])
  ok = 0
  worst_force = 0.0
  for kw in offsets:
    res, _ = run_cycle(noslip=noslip, **kw)
    ok += res["picked"] and res["returned"]
    worst_force = max(worst_force, res["max_force_n"])
  return {"seat": f"{ok}/{len(offsets)}", "worst_force_n": worst_force}


def rig_schuko(noslip: int, quick: bool) -> dict:
  """Plug insertions at aligned + edge-of-envelope, on the spike rig."""
  offsets = ([{}, {"yaw_deg": 2.0}] if quick else
             [{}, {"y_off": -0.002}, {"y_off": 0.002},
              {"yaw_deg": -2.0}, {"yaw_deg": 2.0}])
  ok = 0
  worst_gap = 0.0
  for kw in offsets:
    res, _ = run_trial(noslip=noslip, **kw)
    ok += res["success"]
    worst_gap = max(worst_gap, res["gap_mm"])
  return {"seat": f"{ok}/{len(offsets)}", "worst_gap_mm": worst_gap}


# ---- robot-level, hub_world -------------------------------------------------

def robot_swap(noslip: int, quick: bool) -> dict:
  """Full robot-driven pick + return with hand-off jitter: the coupling
  criterion that historically broke under noslip was ELECTRICAL seating
  (on the fork but not powered), so that is the one reported."""
  jitters = [(0.0, 0.0)] if quick else [(0.0, 0.0), (0.003, 1.0), (-0.003, -1.0)]
  picked = powered = hung = 0
  worst_bay_mm = 0.0
  for dy, dyaw in jitters:
    model, data = load_hub(noslip)
    swap = HubSwap(model, data)
    swap.place_at_standoff(HUB_STATION_YS[0], dy=dy, dyaw_deg=dyaw)
    swap.pick()
    st = swap.module_state("module_lcd")
    picked += st["on_fork"]
    powered += module_power_contact(model, data, "module_lcd")
    swap.put_back()
    st = swap.module_state("module_lcd")
    hung += st["hung"]
    worst_bay_mm = max(worst_bay_mm, st["bay_err_mm"])
  n = len(jitters)
  return {"picked": f"{picked}/{n}", "powered": f"{powered}/{n}",
          "hung": f"{hung}/{n}", "worst_bay_mm": worst_bay_mm}


def grip_hold(noslip: int, hold_s: float = 6.0) -> dict:
  """Fetch the claw, grip the block, and HOLD at carry height: creep in the
  jaws over a timed hold is the signature of regularized-friction drift
  (measured ~8 mm/s before GRIP_SOLIMP; slower lifts lost the block sooner,
  which no real friction failure does)."""
  model, data = load_hub(noslip)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  claw = ClawTool(model, data, swap)
  claw.calibrate()
  obj = data.xpos[model.body("pickup").id]
  if not claw.drive_over((float(obj[0]), float(obj[1])), 0.0):
    return {"held": False, "creep_mm_per_s": None, "note": "never reached block"}
  r = claw.pick_up()
  if not r["holding"]:
    return {"held": False, "creep_mm_per_s": None, "note": "grip failed"}
  claw.set_lift(CARRY_LIFT, settle=0.5)
  rel0 = float(data.xpos[model.body("pickup").id][2]) - float(claw.grip_world()[2])
  swap._run(hold_s, 0.0)
  rel1 = float(data.xpos[model.body("pickup").id][2]) - float(claw.grip_world()[2])
  return {"held": claw.holding(),
          "creep_mm_per_s": (rel1 - rel0) / hold_s * 1000.0,
          "module_powered": module_power_contact(model, data, CLAW_MODULE)}


def pen_square(noslip: int) -> dict:
  """Fetch the pen, drive to the board, trace the square -- the diagnostic
  figure, because it holds an extreme carriage offset for whole edges, which
  is exactly the sustained tangential load that drifts without noslip."""
  model, data = load_hub(noslip)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  plotter = PenPlotter(model, data, swap)
  if not plotter.drive_to_board():
    return {"note": "never reached the board"}
  # Neutralize the plotter's own per-phase toggle: this stage measures the
  # GLOBAL policy, and draw() would otherwise overwrite it in both directions.
  plotter.contact_physics = lambda on=True: None
  t0 = time.perf_counter()
  r = plotter.draw(square_path())
  wall = time.perf_counter() - t0
  if not r.get("drew"):
    return {"note": str(r)}
  return {"ink": r["inked_fraction"], "form_rms_mm": r["form_rms_mm"],
          "offset_mm": r["offset_mm"], "draw_wall_s": wall,
          "still_powered": module_power_contact(model, data, PEN_MODULE)}


def step_cost(noslip: int, n_steps: int = 4000) -> dict:
  """ms per mj_step in room_hub, robot settled on the floor -- the resting
  contact state the 24/7 shared world will spend most of its life in."""
  model = mujoco.MjModel.from_xml_path(ROOM_HUB)
  model.opt.noslip_iterations = noslip
  data = mujoco.MjData(model)
  for _ in range(1000):                        # settle out of the model keyframe
    mujoco.mj_step(model, data)
  t0 = time.perf_counter()
  for _ in range(n_steps):
    mujoco.mj_step(model, data)
  return {"ms_per_step": (time.perf_counter() - t0) / n_steps * 1000.0}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policies", type=int, nargs="+", default=[0, 1, 3],
                      help="noslip_iterations values to sweep")
  parser.add_argument("--quick", action="store_true",
                      help="fewer offsets/jitters per stage")
  parser.add_argument("--no-brake", action="store_true",
                      help="zero the wheel joints' frictionloss "
                           "(reproduces the before-fix rows)")
  args = parser.parse_args()
  global NO_BRAKE
  NO_BRAKE = args.no_brake

  rows = {}
  for n in args.policies:
    print(f"\n== policy: noslip_iterations = {n} ==")
    row = {}
    for name, fn in (("coupling", lambda: rig_coupling(n, args.quick)),
                     ("schuko", lambda: rig_schuko(n, args.quick)),
                     ("robot", lambda: robot_swap(n, args.quick)),
                     ("grip", lambda: grip_hold(n)),
                     ("pen", lambda: pen_square(n)),
                     ("cost", lambda: step_cost(n))):
      t0 = time.perf_counter()
      row[name] = fn()
      print(f"  {name:9s} {row[name]}   [{time.perf_counter() - t0:.0f} s]")
    rows[n] = row

  print("\n== summary ==")
  print(f"{'noslip':>6} | {'coupling':>8} | {'schuko':>6} | {'powered':>7} | "
        f"{'hung':>5} | {'creep mm/s':>10} | {'ink':>5} | {'form mm':>7} | "
        f"{'ms/step':>7}")
  for n, row in rows.items():
    creep = row["grip"].get("creep_mm_per_s")
    ink = row["pen"].get("ink")
    print(f"{n:>6} | {row['coupling']['seat']:>8} | {row['schuko']['seat']:>6} | "
          f"{row['robot']['powered']:>7} | {row['robot']['hung']:>5} | "
          f"{creep if creep is None else f'{creep:+.2f}':>10} | "
          f"{ink if ink is None else f'{ink:.0%}':>5} | "
          f"{row['pen'].get('form_rms_mm', float('nan')):>7.2f} | "
          f"{row['cost']['ms_per_step']:>7.3f}")


if __name__ == "__main__":
  main()
