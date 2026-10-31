#!/usr/bin/env python
"""What the robot's work COSTS, measured (issue #15).

`economy/energy.json` says how much energy each errand takes, and the mission
loop refuses to start one it cannot pay for. Those numbers have to be
MEASURED -- a guessed energy model is how issue #21 shipped a fixture in
which the robot claimed a job at 88 % of its cell, did it perfectly, and died
on the way back. This script is where they come from, and re-running it is
how they are re-derived after anything that changes what an errand does: the
house's layout, the body, a routine.

    MUJOCO_GL=egl uv run python scripts/energy_spike.py
    MUJOCO_GL=egl uv run python scripts/energy_spike.py --actions open_drawer,open_door,place_cube,open_chest
    MUJOCO_GL=egl uv run python scripts/energy_spike.py --reserve
    MUJOCO_GL=egl uv run python scripts/energy_spike.py --write

The measurement is deliberately taken on an OVERSIZED pack (`--battery-wh`,
40 Wh by default). Not to flatter the numbers -- energy per errand does not
depend on capacity -- but so nothing measured is cut short by `needs_charge`:
an errand that stopped halfway is not a cost.

What comes out, on the humanoid in its house (`home_humanoid`, #559):
`exploreWhPerS` (so a bounded explore slice can be priced), `chargeW` (the
NET rate into the pack plugged in at its station, which the charge timeout
is sized against), and per hand job (`--actions`, each from the station
with its area remembered, its reference's two procedures flown):

    wh        the whole errand, the span the loop cannot interrupt: what the
              gate compares against the pack
    s         sim seconds it took, for reading the wh against
    w         mean load over the errand, as a sanity check

`--reserve` measures the return trip from the worst place in the house
instead (`humanoid.world.RESERVE_WH`).
"""

import argparse
import json
import math
import time
from pathlib import Path

import mujoco

from pluggybot.economy.tasks import TaskBoard
from pluggybot.robot import world_spec  # noqa: E402

from pluggybot.lifecycle import (
  HUMANOID_HOME, HubLifecycle, board_book, points_ledger, world_config, world_for,
)

#: The humanoids' hand jobs (#559, stage 3), priced on request, by kind: each
#: its prop on the storeroom's counter, its reference's two procedures
#: (`challenge.solutions`) flown from the station with its place remembered.
HAND_JOBS = {"open_drawer": "drawer", "open_door": "door", "place_cube": "cube",
             "open_chest": "chest"}


class _Writer:
  """What the lifecycle reads off a mind on a hand job's path: a library,
  the mark of something that can write a procedure, and nothing that
  decides -- the spike prices the hand-written solution, never a model."""
  event_map = pending = interrupt_pending = spend = None
  can_escalate = False
  library = object()
  decisions: list = []

  def __init__(self, world: str) -> None:
    from pluggybot.mind import overseer as ov
    self.menu = ov.Menu.for_world(world)


#: A pack far bigger than any errand, so nothing being measured is cut short.
#: See the module docstring: this is about not measuring a death.
BIG_PACK_WH = 40.0


def _from_the_station(life, prop: str, tag: int) -> None:
  """To the station, the prop's area (its tag `tag`) found first if it is
  not remembered: where a job is taken from (the energy table's rule)."""
  from pluggybot.home.places import area
  if life.body.places.get(tag) is None:
    at = area(prop)["address"]
    rec = life.body.run(life.body.find_tag_routine(tag, near=(at["x"], at["y"]),
                                                   patience=600.0))
    print(f"  (found {prop} first: {rec.get('why')} in {rec.get('seconds')} s, unpriced)")
  if life.go_charge():
    life.charge()


def measure(world: str, actions, battery_wh: float, explore_s: float,
            charge_s: float) -> dict:
  """Fly every errand once and report what each one took out of the pack."""
  cfg = world_config(world)
  # ...the world with its body in it
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  book = board_book(world)
  life = HubLifecycle(model, data, realtime=False, battery_wh=battery_wh,
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], boards=book,
                      ledger=points_ledger(None), world=world, errands=[],
                      # ...and a board for the hand jobs' offers, in memory:
                      # one is offered and claimed as the loop would
                      tasks=TaskBoard(),
                      # The served configuration (issue #34): the depth
                      # camera streams all day there, so the table prices
                      # the dearer case and a test day without it is safe.
                      near_field=True)
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.body.step_hooks.append(activities.step_hook(model, data))
    life.activities = activities

  # `explore` reads these two off the running mission (`run()` sets them);
  # this script drives the phases directly, so it sets them itself.
  life.max_sim_time = 1e9
  life.blacklist = set()
  life.floor_explored = False
  life.explore_deadline = 1e9

  out: dict = {"world": world, "batteryWh": battery_wh, "actions": {}}
  try:
    life.body.start_at(*cfg["start"])
    life.body.start_discovery()
    life.body.run(life.body.look_around_routine())

    # ---- explore ------------------------------------------------------------
    t0, e0 = float(data.time), life.battery.energy_wh
    life.explore(budget=explore_s, mark_done=False)
    dt = max(1e-6, float(data.time) - t0)
    out["exploreWhPerS"] = (e0 - life.battery.energy_wh) / dt
    out["exploreS"] = dt
    print(f"  explore   {dt:6.1f}s  {e0 - life.battery.energy_wh:.4f} Wh  "
          f"({out['exploreWhPerS'] * 3600:.1f} W)")

    # ---- one hand job at a time --------------------------------------------
    from pluggybot.challenge import solutions
    from pluggybot.home import storeroom
    for action in actions:
      # ...from the station, its area's tag found first if the explore missed it
      prop = HAND_JOBS[action]
      _from_the_station(life, prop, storeroom.TAGS[prop])
      t0 = float(data.time)
      # Topped up first, so every job is measured from the same place in the
      # pack and none of them against a battery that ran out halfway.
      # Written rather than charged: a real charge cycle is what `chargeW`
      # below measures.
      life.battery.energy_wh = battery_wh
      life.overseer = _Writer(world)
      try:
        run = life.body.run(solutions.job_routine(life, action, world))
      finally:
        life.overseer = None
      used = battery_wh - life.battery.energy_wh
      dt = max(1e-6, float(data.time) - t0)
      out["actions"][action] = {"wh": used, "s": dt, "w": used * 3600.0 / dt,
                                "errand": action, "ok": run["grade"]["ok"]}
      print(f"  {action:18s} {dt:6.1f}s  {used:.4f} Wh  ({used * 3600.0 / dt:5.1f} W)  "
            f"{'PASSED' if run['grade']['ok'] else 'FAILED'} -- {run['grade']['reason']}")
      life.battery.energy_wh = battery_wh

    # ---- and the charger, for CHARGE_TIMEOUT ------------------------------
    # Half-empty, so the charger has something to put back and the
    # measurement is not taken against a pack that fills in one step.
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
      # ...held as `charge()` holds it: plugged in at its station
      life.body.docked = True
      while float(data.time) < end:
        life.body.run(life.body.dock_hold_routine(0.25))
        if not life.charging_now:
          life.body.run(life.body.redock_routine())
          if not life.charging_now:
            print("  charge    lost the plug")
            break
      life.body.docked = False
      dt = max(1e-6, float(data.time) - t0)
      gained = life.battery.energy_wh - e0
      out["chargeW"] = gained * 3600.0 / dt
      print(f"  charge    {dt:6.1f}s  {gained:+.4f} Wh  "
            f"({out['chargeW']:5.1f} W net into the pack)")
    elif not life.charger:
      print("  charge    nowhere to charge in this world -- no chargeW measured")
    else:
      print("  charge    could not reach the station -- no chargeW measured")
  finally:
    life.body.close()
  return out


def measure_reserve(world: str, battery_wh: float, explore_s: float) -> dict:
  """What it COSTS to get home from the worst place to be (issues #70, #84).

  The reserve (`humanoid.world.RESERVE_WH`) is the one energy number that is
  not about an errand: it is the absolute cost of reaching the station from
  the worst point in the floor plan, which is why it is a property of the
  PLAN and deliberately not scaled with the pack.

  Two parts, measured separately because they fail differently:

    travelWh   walking `home.HOME_WORST_RETURN_PATH` from the worst point to
               the garden doorway. The waypoints are FOLLOWED, not planned --
               the route is a fact about the floor plan and lives in
               home/world.py -- so this is the physical cost of the distance,
               not of the planner's mood that day.
    dockWh     `go_charge()` from the garden doorway: the walk to the
               standoff, the station's tags, the walk in and the plug-in,
               which is the part a plain distance model cannot predict.

  The sum is a FLOOR, not the constant: the constant also has to cover a
  failed docking, so the caller adds margin and says so at the constant.
  """
  from pluggybot.behavior.navigation import drive_toward
  from pluggybot.home import world as home

  cfg = world_config(world)
  # ...the world with its body in it
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, battery_wh=battery_wh,
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=points_ledger(None), world=world, errands=[],
                      tasks=TaskBoard(),
                      near_field=True)
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.body.step_hooks.append(activities.step_hook(model, data))
  life.max_sim_time = 1e9
  life.blacklist = set()
  life.floor_explored = False
  life.explore_deadline = 1e9

  out: dict = {"world": world}
  try:
    life.body.start_at(*cfg["start"])
    life.body.start_discovery()
    life.body.run(life.body.look_around_routine())
    life.explore(budget=explore_s, mark_done=False)

    path = list(home.HOME_WORST_RETURN_PATH)
    # Start AT the worst point, odometry seeded from truth: this measures the
    # route's energy, not the robot's confusion about where it is.
    hd = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    life.body.start_at(path[0][0], path[0][1], hd)
    life.battery.energy_wh = battery_wh

    t0, e0 = float(data.time), life.battery.energy_wh
    metres = 0.0
    # ⚠ Every waypoint EXCEPT the last: the final leg belongs to `go_charge`,
    # which walks to the station's STANDOFF and in off its tags. That split
    # is also the honest one: travel is what a distance model can predict,
    # docking is what it cannot.
    for wx, wy in path[1:-1]:
      metres += math.hypot(wx - life.body.pose[0],
                           wy - life.body.pose[1])
      deadline = float(data.time) + 120.0
      while float(data.time) < deadline:
        px, py, _ = life.body.pose
        if math.hypot(wx - px, wy - py) < 0.15:
          break
        v, w = drive_toward(life.body.pose, (wx, wy), slow_radius=0.5)
        life.body.run(life.body.velocity_routine(0.05, v, w))
    travel = e0 - life.battery.energy_wh
    out["travelWh"] = travel
    out["travelS"] = float(data.time) - t0
    out["routeM"] = metres
    print(f"  return    {out['travelS']:6.1f}s  {travel:.4f} Wh over "
          f"{metres:.2f} m  ({travel / max(metres, 1e-6) * 1000:.1f} mWh/m)"
          f"  [to the garden doorway; the rest is the station]")

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
    life.body.close()
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--world", default=HUMANOID_HOME, choices=(HUMANOID_HOME, "home"),
                  help="the house with the humanoid in it (#559; `home` names it too): "
                       "its explore, its station and its hand jobs")
  ap.add_argument("--battery-wh", type=float, default=BIG_PACK_WH,
                  help="an oversized pack, so nothing measured is cut short")
  ap.add_argument("--explore-s", type=float, default=None,
                  help="sim seconds of exploring before the errands "
                       "(default: the world's explore budget)")
  ap.add_argument("--charge-s", type=float, default=90.0,
                  help="sim seconds plugged in to measure the charge rate")
  ap.add_argument("--actions", default="",
                  help="hand jobs to price besides the explore and the charger, "
                       f"comma separated: any of {','.join(HAND_JOBS)}")
  ap.add_argument("--reserve", action="store_true",
                  help="measure the worst-case return trip instead of the "
                       "errands (issues #70/#84): what humanoid.world.RESERVE_WH "
                       "has to cover on the current floor plan")
  ap.add_argument("--json", default=None, help="write the raw measurement here")
  ap.add_argument("--write", action="store_true",
                  help="fold the result into src/pluggybot/economy/energy.json")
  args = ap.parse_args()
  args.world = world_for(args.world)

  cfg = world_config(args.world)
  explore_s = args.explore_s if args.explore_s is not None \
      else float(cfg["explore_budget"])
  actions = tuple(a for a in args.actions.split(",") if a)
  unknown = [a for a in actions if a not in HAND_JOBS]
  if unknown:
    ap.error(f"--actions: {', '.join(unknown)} is no hand job ({', '.join(HAND_JOBS)})")
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
    # Deliberately never --write: the reserve is a constant in
    # humanoid/world.py with a paragraph of reasoning attached, not a row in
    # a data file, and it needs a human to add the retry margin.
    return
  print(f"== {args.world}: pricing the explore, the charger"
        f"{''.join(', ' + a for a in actions)} on a {args.battery_wh:g} Wh pack")
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
    for key, row in out["actions"].items():
      costs[key] = round(row["wh"], 3)
    if "exploreWhPerS" in out:
      block["exploreWhPerS"] = round(out["exploreWhPerS"], 6)
    if "chargeW" in out:
      block["chargeW"] = round(out["chargeW"], 1)
    block["measuredOn"] = f"{args.battery_wh:g} Wh pack, {math.floor(explore_s)} s explore"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"folded into {path}")


if __name__ == "__main__":
  main()
