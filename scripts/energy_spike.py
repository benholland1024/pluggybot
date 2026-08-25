#!/usr/bin/env python
"""What an errand COSTS, measured (issue #15).

`hub/energy.json` says how much energy each errand takes in each world, and
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

from pluggybot.hub.lifecycle import (
  HubLifecycle, board_book, errands_for, points_ledger, world_config,
  world_screens,
)

#: Actions worth pricing: every errand a world can build, plus the two the
#: `showcase` queue is made of. `artwork` and `answer` are drawing errands
#: with a different tier and a different figure, so they are priced as
#: `draw` in hub/energy.json rather than flown twice for the same number.
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
  ap.add_argument("--json", default=None, help="write the raw measurement here")
  ap.add_argument("--write", action="store_true",
                  help="fold the result into src/pluggybot/hub/energy.json")
  args = ap.parse_args()

  cfg = world_config(args.world)
  explore_s = args.explore_s if args.explore_s is not None \
      else float(cfg["explore_budget"])
  actions = tuple(a for a in args.actions.split(",") if a)
  wall = time.time()
  print(f"== {args.world}: pricing {', '.join(actions)} on a "
        f"{args.battery_wh:g} Wh pack")
  out = measure(args.world, actions, args.battery_wh, explore_s, args.charge_s)
  out["wallS"] = round(time.time() - wall, 1)
  print(f"-- {out['wallS']:.0f} s of wall clock")

  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.json}")
  if args.write:
    from pluggybot.hub import energy
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
