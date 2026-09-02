#!/usr/bin/env python
"""What an errand COSTS, measured (issue #15).

`economy/energy.json` says how much energy each errand takes in each world, and
the mission loop refuses to start one it cannot pay for. Those numbers have
to be MEASURED -- a guessed energy model is how issue #21 shipped a fixture
in which the robot claimed a 0.93 Wh drawing at 88 %% of a 1.1 Wh cell, drew
it perfectly, and died on the way back. This script is where they come from,
and re-running it is how they are re-derived after anything that changes what
an errand does: a world's layout, the swap stack, the drivetrain, a routine.

    MUJOCO_GL=egl uv run python scripts/energy_spike.py --world home
    MUJOCO_GL=egl uv run python scripts/energy_spike.py --world home --write

The measurement is deliberately taken on an OVERSIZED pack (`--battery-wh`,
40 Wh by default). Not to flatter the numbers -- energy per errand does not
depend on capacity -- but because the demo cells are smaller than one errand,
so a measurement taken on one would be measuring where the robot died rather
than what the job costs. The census in particular breaks off early when
`needs_charge` fires, and an errand that stopped halfway is not a cost.

What comes out, per action:

    wh        SWAP_PICK -> end of SWAP_RETURN, the whole fetch-use-stow job.
              This is the number the gate compares against the pack, because
              it is the span the loop cannot interrupt.
    s         sim seconds it took, for reading the wh against
    w         mean load over the errand, as a sanity check against
              power.ELECTRONICS_W (8.5 W) plus a tool's MODULE_IDLE_W

...plus two figures that are not errands and are measured for the same file:
`exploreWhPerS` (so a bounded explore slice can be priced) and `chargeW` (the
NET rate into the pack, which is what `CHARGE_TIMEOUT` has to be sized
against -- 55 W of charger minus whatever the held press is drawing).
"""

import argparse
import json
import math
import time
from pathlib import Path

import mujoco

from pluggybot.lifecycle import (
  HubLifecycle, board_book, errands_for, points_ledger, world_config,
  world_screens,
)

#: Actions worth pricing: every errand a world can build, plus the two the
#: `showcase` queue is made of. `artwork` and `answer` are drawing errands
#: with a different tier and a different figure, so they are priced as
#: `draw` in economy/energy.json rather than flown twice for the same number.
ACTIONS = ("carry", "draw", "census", "dance")

#: A pack far bigger than any errand, so nothing being measured is cut short.
#: See the module docstring: this is about not measuring a death.
BIG_PACK_WH = 40.0


def measure(world: str, actions, battery_wh: float, explore_s: float,
            charge_s: float) -> dict:
  """Fly every errand once and report what each one took out of the pack."""
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  book = board_book(world)
  screens = world_screens(model, data)
  life = HubLifecycle(model, data, realtime=False, battery_wh=battery_wh,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], boards=book,
                      screen=next(iter(screens), None),
                      ledger=points_ledger(None), world=world, errands=[])
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))

  # `explore` reads these two off the running mission (`run()` sets them);
  # this script drives the phases directly, so it sets them itself.
  life.max_sim_time = 1e9
  life.blacklist = set()
  life.map_done = False
  life.explore_deadline = 1e9

  out: dict = {"world": world, "batteryWh": battery_wh, "actions": {}}
  try:
    life.mission.start_at(*cfg["start"])
    life.mission.start_discovery()
    life.mission._spin()

    # ---- explore, which is also how the rack gets found -------------------
    t0, e0 = float(data.time), life.battery.energy_wh
    life.explore(budget=explore_s, mark_done=False)
    dt = max(1e-6, float(data.time) - t0)
    out["exploreWhPerS"] = (e0 - life.battery.energy_wh) / dt
    out["exploreS"] = dt
    print(f"  explore   {dt:6.1f}s  {e0 - life.battery.energy_wh:.4f} Wh  "
          f"({out['exploreWhPerS'] * 3600:.1f} W)")

    # ---- one errand at a time ---------------------------------------------
    for action in actions:
      try:
        queue = errands_for(action, world, book)
      except ValueError as e:
        print(f"  {action:9s} skipped: {e}")
        continue
      for errand in queue:
        t0, e0 = float(data.time), life.battery.energy_wh
        # Topped up first, so every errand is measured from the same place in
        # the pack and none of them is measured against a battery that ran
        # out halfway. Written rather than charged: a real charge cycle is
        # what `chargeW` below measures, and paying for one between every
        # errand would triple the run for nothing.
        life.battery.energy_wh = battery_wh
        result = life.run_errand(errand)
        used = battery_wh - life.battery.energy_wh
        dt = max(1e-6, float(data.time) - t0)
        out["actions"][action] = {
          "wh": used, "s": dt, "w": used * 3600.0 / dt,
          "errand": errand.name, "stowed": bool(result.get("stowed")),
        }
        print(f"  {action:9s} {dt:6.1f}s  {used:.4f} Wh  "
              f"({used * 3600.0 / dt:5.1f} W)  "
              f"{'stowed' if result.get('stowed') else 'NOT STOWED'}")
        life.battery.energy_wh = battery_wh
        _ = e0, t0

    # ---- and the charger, for CHARGE_TIMEOUT ------------------------------
    # Half-empty, so the press has something to put back and the measurement
    # is not taken against a pack that fills in one step.
    life.battery.energy_wh = battery_wh * 0.5
    if life.go_charge():
      # Deliberately NOT life.charge(): that stops at CHARGED and times out
      # at CHARGE_TIMEOUT, and both are the things being sized here.
      #
      # ⚠ CLOCKED FROM AFTER THE DRIVE, and pressed BEFORE the contact is
      # believed -- exactly as `charge()` does it. Timing from before
      # `go_charge` charged the approach to the charger and read -18.7 W;
      # checking `charging_now` before the first press ends the measurement
      # on the step where the suspension has not settled yet.
      t0, e0 = float(data.time), life.battery.energy_wh
      end = t0 + charge_s
      while float(data.time) < end:
        life.mission._drive(0.25, 0.012, 0.0)
        if not life.charging_now:
          life.mission._drive(1.0, 0.04, 0.0)
          if not life.charging_now:
            print("  charge    lost the pins")
            break
      dt = max(1e-6, float(data.time) - t0)
      gained = life.battery.energy_wh - e0
      out["chargeW"] = gained * 3600.0 / dt
      print(f"  charge    {dt:6.1f}s  {gained:+.4f} Wh  "
            f"({out['chargeW']:5.1f} W net into the pack)")
    else:
      print("  charge    could not reach the rack -- no chargeW measured")
  finally:
    life.mission.close()
  return out


def measure_reserve(world: str, battery_wh: float, explore_s: float) -> dict:
  """What it COSTS to get home from the worst place to be (issues #70, #84).

  `HOME_LOW_BATTERY_WH` is the one energy number that is not about an errand:
  it is the absolute cost of reaching the dock from the worst point in the
  floor plan, which is why it is a property of the PLAN and deliberately not
  scaled with the pack. It was written down as ~0.3 Wh -- "a full living-room
  crossing plus charge approach", a 2.89 m route -- and issue #68 grew the
  plot to 26.5 x 12 m, which moves the worst point to the street's far
  corner, 15.6 m of route away.

  Two parts, measured separately because they fail differently:

    travelWh   driving `home.HOME_WORST_RETURN_PATH` from the worst point to
               the garden doorway. The waypoints are FOLLOWED, not planned --
               the route is a fact about the floor plan and lives in
               home/world.py -- so this is the physical cost of the distance,
               not of the planner's mood that day.
    dockWh     `go_charge()` from the garden doorway: the drive to the
               standoff, the tag-servo creep and the press, which is the part
               a plain distance model cannot predict.

  The sum is a FLOOR, not the constant: the constant also has to cover a
  failed press-and-retry, so the caller adds margin and says so at the
  constant. (An earlier draft drove the route OUTBOUND first, to open the
  gate the street used to be behind. Issue #93 removed the gate, so the
  street is plain floor now, and the outbound leg went with it.)
  """
  from pluggybot.behavior.navigation import drive_toward
  from pluggybot.home import world as home

  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=battery_wh,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=points_ledger(None), world=world, errands=[])
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))
  life.max_sim_time = 1e9
  life.blacklist = set()
  life.map_done = False
  life.explore_deadline = 1e9

  out: dict = {"world": world}
  try:
    life.mission.start_at(*cfg["start"])
    life.mission.start_discovery()
    life.mission._spin()
    life.explore(budget=explore_s, mark_done=False)

    path = list(home.HOME_WORST_RETURN_PATH)
    # Start AT the worst point, odometry seeded from truth: this measures the
    # route's energy, not the robot's confusion about where it is.
    hd = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    life.mission.start_at(path[0][0], path[0][1], hd)
    life.battery.energy_wh = battery_wh

    t0, e0 = float(data.time), life.battery.energy_wh
    metres = 0.0
    # ⚠ Every waypoint EXCEPT the last. The path ends at the rack itself, and
    # driving to within 15 cm of a rack centre is driving into the rack --
    # the final leg belongs to `go_charge`, which navigates to the charge
    # STANDOFF and then creeps on the tag. That split is also the honest one:
    # travel is what a distance model can predict, docking is what it cannot.
    for wx, wy in path[1:-1]:
      metres += math.hypot(wx - life.mission.pose[0],
                           wy - life.mission.pose[1])
      deadline = float(data.time) + 120.0
      while float(data.time) < deadline:
        px, py, _ = life.mission.pose
        if math.hypot(wx - px, wy - py) < 0.15:
          break
        v, w = drive_toward(life.mission.pose, (wx, wy), slow_radius=0.5)
        life.mission._drive(0.05, v, w)
    travel = e0 - life.battery.energy_wh
    out["travelWh"] = travel
    out["travelS"] = float(data.time) - t0
    out["routeM"] = metres
    print(f"  return    {out['travelS']:6.1f}s  {travel:.4f} Wh over "
          f"{metres:.2f} m  ({travel / max(metres, 1e-6) * 1000:.1f} mWh/m)"
          f"  [to the garden doorway; the rest is the dock]")

    e1 = life.battery.energy_wh
    docked = life.go_charge()
    out["dockWh"] = e1 - life.battery.energy_wh
    out["docked"] = bool(docked)
    print(f"  dock      {'reached' if docked else 'FAILED'}  "
          f"{out['dockWh']:.4f} Wh")
    out["reserveWh"] = out["travelWh"] + out["dockWh"]
    print(f"  RESERVE   {out['reserveWh']:.4f} Wh floor "
          f"(travel + dock, before the retry margin)")
  finally:
    life.mission.close()
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", default="home", choices=("home", "room_hub"))
  ap.add_argument("--battery-wh", type=float, default=BIG_PACK_WH,
                  help="an oversized pack, so nothing measured is cut short")
  ap.add_argument("--explore-s", type=float, default=None,
                  help="sim seconds of exploring before the errands "
                       "(default: the world's explore budget)")
  ap.add_argument("--charge-s", type=float, default=90.0,
                  help="sim seconds of held press to measure the charge rate")
  ap.add_argument("--actions", default=",".join(ACTIONS))
  ap.add_argument("--reserve", action="store_true",
                  help="measure the worst-case return trip instead of the "
                       "errands (issues #70/#84): what HOME_LOW_BATTERY_WH "
                       "has to cover on the current floor plan")
  ap.add_argument("--json", default=None, help="write the raw measurement here")
  ap.add_argument("--write", action="store_true",
                  help="fold the result into src/pluggybot/economy/energy.json")
  args = ap.parse_args()

  cfg = world_config(args.world)
  explore_s = args.explore_s if args.explore_s is not None \
      else float(cfg["explore_budget"])
  actions = tuple(a for a in args.actions.split(",") if a)
  wall = time.time()
  if args.reserve:
    print(f"== {args.world}: measuring the worst-case return trip on a "
          f"{args.battery_wh:g} Wh pack")
    out = measure_reserve(args.world, args.battery_wh, explore_s)
    out["wallS"] = round(time.time() - wall, 1)
    print(f"-- {out['wallS']:.0f} s of wall clock")
    if args.json:
      Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
      print(f"wrote {args.json}")
    # Deliberately never --write: the reserve is a constant in home/world.py
    # with a paragraph of reasoning attached, not a row in a data file, and
    # it needs a human to add the retry margin.
    return
  print(f"== {args.world}: pricing {', '.join(actions)} on a "
        f"{args.battery_wh:g} Wh pack")
  out = measure(args.world, actions, args.battery_wh, explore_s, args.charge_s)
  out["wallS"] = round(time.time() - wall, 1)
  print(f"-- {out['wallS']:.0f} s of wall clock")

  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.json}")
  if args.write:
    from pluggybot.economy import energy
    path = energy.ENERGY_PATH
    doc = json.loads(path.read_text())
    block = doc.setdefault("worlds", {}).setdefault(args.world, {})
    costs = block.setdefault("errandWh", {})
    for action, row in out["actions"].items():
      costs[action] = round(row["wh"], 3)
    # The two drawing tiers are the drawing errand: same tool, same board,
    # same figure size. Priced together rather than flown three times.
    if "draw" in costs:
      costs["artwork"] = costs["draw"]
      costs["answer"] = costs["draw"]
    if "exploreWhPerS" in out:
      block["exploreWhPerS"] = round(out["exploreWhPerS"], 6)
    if "chargeW" in out:
      block["chargeW"] = round(out["chargeW"], 1)
    block["measuredOn"] = f"{args.battery_wh:g} Wh pack, {math.floor(explore_s)} s explore"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"folded into {path}")


if __name__ == "__main__":
  main()
