"""Build an agent-described tool and try it as the workshop does (issue #168;
on legs, #407).

  MUJOCO_GL=egl uv run python scripts/workshop.py --example
  MUJOCO_GL=egl uv run python scripts/workshop.py --spec my_tool.json
  MUJOCO_GL=egl uv run python scripts/workshop.py --example --served --bay B

Validates the spec against the arm's envelope (every reason at once), then
runs the rig the workshop gates a build on (`workshop.build.trial`): hung on
a bay, taken by the fork, conducting, every axis to its far end and back,
hung back -- with the fork on the bay's middle and at the walk-in's line-up
gate either side. `--served` then hangs it on the served pair's built-tool
rail (`HubLifecycle.hang_tool`: the recompile, every holder rebound) and has
the first robot's arm fetch it and hang it back: SimNotes, "The workshop on
legs". The example is the scoop the robot's prompt shows.
"""

import argparse
import json
import math
import sys
import time

from pluggybot.workshop import build, validate
from pluggybot.workshop.spec import Refused


def example() -> dict:
  """The scoop the robot's prompt shows (`overseer.WORKSHOP_HEAD`), as written."""
  from pluggybot.mind.overseer import WORKSHOP_HEAD
  start = WORKSHOP_HEAD.index('{"name": "scoop"')
  return json.loads(WORKSHOP_HEAD[start:WORKSHOP_HEAD.index("THE ENVELOPE")])


def try_on_the_rig(tool) -> bool:
  """The workshop's gate (`build.trial`): the rig at the bay's middle and
  the line-up gate either side, each record printed; True when it passed."""
  records: list = []
  t0 = time.time()
  reasons = build.trial(tool, records)
  for rec in records:
    print(f"  rig, fork {rec['dy'] * 1000:+.0f} mm: " + ", ".join(
      f"{k}={v}" for k, v in rec.items() if k not in ("dy", "poles")))
  print(f"trial ({time.time() - t0:.2f} s): " + ("passed" if not reasons else
                                              f"REFUSED -- {reasons[0]}"))
  return not reasons


def served(tool, bay: int) -> None:
  """Hang the tool on the served pair's rail bay `bay`, then fetch it with
  the first robot's arm and hang it back."""
  from pluggybot import tick
  from pluggybot.legs import rack as rk
  from pluggybot.pair import build_pair
  from pluggybot.rack.coupling import STATION_YS, built_bay_index
  lives = build_pair("home_quad", errands=("none", "none"))
  try:
    for life in lives:
      life.state = "DECIDE"
      life.body.start_at(*life.body.pose)
    a, b = lives
    tick.run_many([(life.body.stepper, life.body.hold_routine(2.0)) for life in lives])
    before = [life.body.true_pose() for life in lives]
    rec = a.hang_tool(tool, bay)
    tick.run_many([(life.body.stepper, life.body.hold_routine(3.0)) for life in lives])
    moved = [math.hypot(life.body.true_pose()[0] - p[0], life.body.true_pose()[1] - p[1])
             for life, p in zip(lives, before)]
    print(f"hung in rail bay {chr(ord('A') + bay)}: recompile {rec['recompileMs']} ms; "
          f"the robots moved {moved[0]:.4f} / {moved[1]:.4f} m; plumb on its bay "
          f"{bool(rk.on_bay(a.model, a.data, tool.body, rk.BUILT, bay))}")
    station = STATION_YS[built_bay_index(bay)]
    t0 = float(a.data.time)
    why = tick.run_many([(a.body.stepper, a.body.fetch_tool_routine(station, tool.body)),
                         (b.body.stepper, b.body.hold_routine(0.0))])
    print(f"fetch: {why[0]}, powered {a.body.tool_powered(tool.body)} -- {a.body.swap_trace()}")
    why = tick.run_many([(a.body.stepper, a.body.stow_tool_routine(station, tool.body)),
                         (b.body.stepper, b.body.hold_routine(0.0))])
    print(f"stow: {why[0]}, hung {bool(a.body.module_state(tool.body)['hung'])} -- "
          f"{a.body.swap_trace()}")
    print(f"falls {[life.body.mission.falls for life in lives]}, "
          f"{float(a.data.time) - t0:.0f} sim s for the two swaps")
  finally:
    for life in lives:
      life.body.close()


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--spec", help="a tool spec, JSON")
  parser.add_argument("--example", action="store_true",
                      help="the scoop the robot's prompt shows (the default)")
  parser.add_argument("--served", action="store_true",
                      help="then hang it on the served pair's rail, fetch it and hang it back")
  parser.add_argument("--bay", default="A", choices=("A", "B", "C"),
                      help="the rail's bay for --served")
  args = parser.parse_args(argv)
  raw = example() if args.example or not args.spec else json.load(open(args.spec))
  try:
    tool = validate.check(raw)
  except Refused as e:
    print("REFUSED:")
    for r in e.reasons:
      print("  -", r)
    return 1
  mass = (validate.MODULE_MASS + tool.mass) * 1000
  print(f"{tool.body}: {len(tool.parts)} parts, {mass:.0f} g with the plate and peg, "
        f"{validate.worst_moment(tool):.3f} N*m worst about the peg, "
        f"hangs {validate.hang_tilt_deg(tool):.1f} deg off plumb and "
        f"{validate.side_of_peg(tool) * 1000:+.0f} mm to the side; "
        f"verbs {[p.axis.verb for p in tool.axes]}")
  ok = try_on_the_rig(tool)
  if args.served and ok:
    served(tool, ord(args.bay) - ord("A"))
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
