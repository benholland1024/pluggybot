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
  MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --record out.jsonl.gz
    # PluggyWorld telemetry recording (protocol/README.md); .gz compresses
"""

import argparse

from pluggybot.hub.lifecycle import run_demo


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--view", action="store_true", help="open the viewer")
  parser.add_argument("--fast", action="store_true",
                      help="with --view: no real-time pacing")
  parser.add_argument("--battery-wh", type=float, default=None,
                      help="battery capacity (per-world demo cell by default)")
  parser.add_argument("--world", choices=("room_hub", "home"),
                      default="room_hub",
                      help="which world to run: room_hub (default) or the "
                           "generated home world (issue #6)")
  parser.add_argument("--pack", choices=("demo", "hosting"), default="demo",
                      help="which cell to fly on (issue #15). `demo` is the "
                           "minutes-long cell every mission test and both "
                           "recordings use; `hosting` is the hours-long one "
                           "a watched world runs on, where the return-trip "
                           "margin becomes real. --battery-wh overrides "
                           "either")
  parser.add_argument("--reserve-wh", type=float, default=None,
                      help="override the world's go-charge reserve, in Wh. "
                           "Absolute energy, not a fraction of the pack -- "
                           "the cost of getting home is set by the ROOM")
  parser.add_argument("--max-sim-time", type=float, default=600.0)
  parser.add_argument("--record", default=None, metavar="PATH",
                      help="write a PluggyWorld telemetry JSONL recording "
                           "(.gz to compress; see protocol/README.md)")
  parser.add_argument("--errand", choices=("carry", "draw", "draw2", "census",
                                          "dance", "artwork", "showcase",
                                          "none"),
                      default="carry",
                      help="what the robot is FOR this run (issue #12): carry "
                           "(the milestone-8 LCD errand), draw (pen -> erase a "
                           "whiteboard -> draw), draw2 (two boards, charging "
                           "in between), none")
  parser.add_argument("--boards", default=None, metavar="PATH",
                      help="JSON file the whiteboards' contents live in "
                           "between runs (default: blank boards every start)")
  parser.add_argument("--ledger", default=None, metavar="PATH",
                      help="JSON file the points ledger lives in between runs "
                           "(issue #14; default: the robot starts at zero)")
  parser.add_argument("--tasks", action="store_true",
                      help="offer the robot JOBS this run (issue #21): each "
                           "one has a description, a target, a reward off "
                           "hub/rewards.json and a deadline. More arrive on "
                           "the cadence in hub/cadence.json as the run goes "
                           "on (issue #23; $PLUGGY_CADENCE re-points it). "
                           "Off by default")
  parser.add_argument("--task-state", default=None, metavar="PATH",
                      help="JSON file the task board lives in between runs "
                           "($PLUGGY_TASKS; implies --tasks)")
  parser.add_argument("--overseer", action="store_true",
                      help="let an LLM choose the errands once --errand's "
                           "queue is empty (issue #15). Needs $ANTHROPIC_API_"
                           "KEY; without one it runs the scripted fallback "
                           "and says so")
  parser.add_argument("--goals", default=None, metavar="PATH",
                      help="the overseer's long-term goals, as prose "
                           "(human-editable; $PLUGGY_GOALS)")
  parser.add_argument("--journal", default=None, metavar="PATH",
                      help="JSON file the overseer's notes-to-self live in "
                           "between runs ($PLUGGY_JOURNAL)")
  args = parser.parse_args()

  r = run_demo(view=args.view,
               realtime=not args.fast, battery_wh=args.battery_wh,
               max_sim_time=args.max_sim_time, record=args.record,
               world=args.world, errand=args.errand,
               board_state=args.boards, ledger_state=args.ledger,
               overseer=args.overseer or None, goals=args.goals,
               journal_state=args.journal,
               tasks=args.tasks, tasks_state=args.task_state,
               pack=args.pack, reserve_wh=args.reserve_wh)
  if args.record:
    print(f"telemetry recorded -> {args.record}")
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
  for e in r["errands"]:
    extra = (f"  {e['figure']}, {e.get('inked_fraction', 0):.0%} inked,"
             f" form {e.get('form_rms_mm') or float('nan'):.2f} mm"
             if e.get("board") else "")
    print(f"errand {e['errand']:<16s}: picked={e['picked']}"
          f" stowed={e['stowed']}{extra}")
  for name, b in r["boards"].items():
    print(f"board {name:<17s}: {b['strokes']} strokes, {b['fill']:.0%} full, "
          f"{b['clears']} clear(s), programs {b['programs'] or '-'}")
  # What the robot EARNED, and why (issue #14). Every line here came out of a
  # deterministic evaluator in hub/scoring.py -- the mission awards nothing.
  for v in r["verdicts"]:
    print(f"score  {v['task']:<16s}: {v['points']:+d}"
          f"{' PENDING' if v['pending'] else ''}"
          f"  {'ok ' if v['ok'] else 'FAIL'}  {v['reason']}")
  print(f"points balance         : {r['points']} ({r['earned']} this mission)")
  # What the overseer chose, and what the choosing cost (issue #15). `source`
  # is on every line because "the robot chose to explore" and "the API was
  # down so the robot explored" look identical from outside.
  for d in r.get("decisions", ()):
    print(f"decide {d['action']:<16s}: {d['reason']} [{d['source']}]")
  for n in r.get("journal", ()):
    print(f"journal t={n['t']:<12.1f}: {n['text']}")
  if r.get("overseer"):
    o = r["overseer"]
    per_hour = o["usd"] / (r["sim_time"] / 3600.0) if r["sim_time"] else 0.0
    print(f"overseer ({o['model']}): {o['llmCalls']} call(s), "
          f"{o['fallbacks']} fallback(s), budget {o['budgetLeft']}/"
          f"{o['callsPerHour']} left")
    print(f"overseer cost          : ${o['usd']:.5f} this run "
          f"(${per_hour:.4f}/sim-hour), cache hit rate "
          f"{o['cacheHitRate']:.0%}")
    for err in o["errors"]:
      print(f"overseer note          : {err}")
  # Every errand's OWN verdict, not just the last module's: a queue where the
  # first errand silently failed and the second stowed cleanly would report a
  # perfect mission off `module_stowed` alone.
  errands_ok = all(e["picked"] and e["stowed"] and not e.get("error")
                   for e in r["errands"])
  ok = (r["charge_cycles"] >= 1 and errands_ok and not r["errands_left"]
        and r["collision_steps"] == 0)
  print("LIFECYCLE:", "OK" if ok else "INCOMPLETE")


if __name__ == "__main__":
  main()
