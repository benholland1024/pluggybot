"""Live PluggyWorld publisher (webserver v1): run the mission, stream it.

Runs the battery-driven hub lifecycle headless, paced to real time, while
publishing protocol frames + occupancy-grid images + event lines to a
WebSocket endpoint (an outbound CLIENT -- point it at the website's ingest
socket, or at scripts/ws_sink.py to watch the wire locally). The sim never
blocks on the network: if the endpoint is down, frames drop and the
mission carries on. Docs: docs/Webserver.md; wire format: protocol/README.md.

Usage:
  MUJOCO_GL=egl    uv run python scripts/serve.py --endpoint ws://localhost:8765
  MUJOCO_GL=osmesa uv run python scripts/serve.py --endpoint ws://localhost:8765
    # CPU-only rendering, the deploy-server configuration
  ... --world home      # the generated house + garden (issue #6); room_hub
                        # is the default. Everything the world implies --
                        # model, scene name, rack pose, grid extent, battery,
                        # start pose, errand destination, explore budget --
                        # comes from hub.lifecycle.world_config(), so this
                        # flag can never half-apply.
  ... --errand draw     # a real drawing errand (issue #12): fetch the pen,
                        # navigate to a whiteboard, erase it, draw, stow. The
                        # strokes stream as `draw` events for the browser to
                        # paint -- they are never MuJoCo geometry.
  ... --boards state.json     # whiteboard contents that survive a restart
  ... --ledger points.json    # the points ledger, likewise (issue #14): the
                              # balance and the earnings log the site shows
  ... --rate 2.0        # sim seconds per wall second (default 1.0)
  ... --free-run        # no pacing: measure this machine's real-time multiple
  ... --record out.jsonl.gz   # also keep a v0 recording of the same run
  ... --token SECRET    # ingest shared secret (default: $PLUGGYWORLD_TOKEN)

The website's ingest path is authenticated; point --endpoint at
ws://host/api/pluggyworld/ingest and give it the same secret the server
holds. Prefer the environment variable to the flag -- an argument is
visible in `ps` to every user on the box.
"""

import argparse
import os
import time

import mujoco

from pluggybot.hub import overseer
from pluggybot.hub.inbox import Inbox
from pluggybot.hub.lifecycle import (
  SEED_STANDING_TTL_S, SEED_TTL_S, HubLifecycle, board_book, errands_for, points_ledger, seed_tasks,
  task_board, world_config, world_screens,
)
from pluggybot.telemetry.pacer import RealTimePacer
from pluggybot.telemetry.protocol import INBOUND_TYPES
from pluggybot.telemetry.publisher import WsPublisher
from pluggybot.telemetry.recorder import KEYFRAME_S, TelemetryRecorder


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--endpoint", default="ws://localhost:8765",
                      help="WebSocket endpoint to publish to")
  parser.add_argument("--world", choices=("room_hub", "home"),
                      default="room_hub",
                      help="which world to serve: room_hub (default) or the "
                           "generated home world (issue #6)")
  parser.add_argument("--rate", type=float, default=1.0,
                      help="pacing: sim seconds per wall second")
  parser.add_argument("--free-run", action="store_true",
                      help="disable pacing (real-time-multiple measurement)")
  parser.add_argument("--battery-wh", type=float, default=None,
                      help="battery capacity (per-world demo cell by default)")
  parser.add_argument("--max-sim-time", type=float, default=600.0)
  parser.add_argument("--record", default=None, metavar="PATH",
                      help="also write a v0 JSONL recording of this run")
  parser.add_argument("--token", default=os.environ.get("PLUGGYWORLD_TOKEN"),
                      help="ingest shared secret (default $PLUGGYWORLD_TOKEN)")
  parser.add_argument("--keyframe-s", type=float, default=KEYFRAME_S,
                      metavar="S", help="sim seconds between full keyframes"
                                        " (0 disables; late joiners then wait"
                                        " forever)")
  parser.add_argument("--errand", choices=("carry", "draw", "draw2", "census",
                                          "dance", "artwork", "showcase",
                                          "none"),
                      default="carry",
                      help="what the robot is FOR this run (issue #12): carry "
                           "(the milestone-8 LCD errand), draw (fetch the pen, "
                           "erase a whiteboard and draw on it), draw2 (two "
                           "boards, charging in between), none")
  parser.add_argument("--boards", default=None, metavar="PATH",
                      help="JSON file the whiteboards' contents live in "
                           "between runs (default: blank boards every start)")
  parser.add_argument("--ledger", default=None, metavar="PATH",
                      help="JSON file the points ledger lives in between runs "
                           "(issue #14; default: the robot starts at zero)")
  parser.add_argument("--overseer", action="store_true",
                      help="let an LLM choose what to do next once --errand's "
                           "queue is empty (issue #15; $PLUGGY_OVERSEER). "
                           "Needs $ANTHROPIC_API_KEY -- without one the robot "
                           "runs the scripted fallback and the run reports it")
  parser.add_argument("--tasks", action="store_true",
                      help="offer the robot JOBS this run (issue #21): each "
                           "one has a description, a target, a reward off "
                           "hub/rewards.json and a deadline. Off by default")
  parser.add_argument("--task-state", default=None, metavar="PATH",
                      help="JSON file the task board lives in between runs "
                           "($PLUGGY_TASKS; implies --tasks)")
  parser.add_argument("--goals", default=None, metavar="PATH",
                      help="the overseer's long-term goals, as prose. Mount it "
                           "and edit it to change what the robot is for, no "
                           "redeploy ($PLUGGY_GOALS)")
  parser.add_argument("--journal", default=None, metavar="PATH",
                      help="JSON file the overseer's notes-to-self live in "
                           "between runs ($PLUGGY_JOURNAL)")
  parser.add_argument("--overseer-budget", type=int, default=None,
                      metavar="N", help="hard cap on LLM calls per rolling "
                                        "hour (default 60)")
  args = parser.parse_args()

  cfg = world_config(args.world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  # Board state is the world's, not the run's: loaded before the mission and
  # written back on every stroke, so a restart walks into the house it left.
  book = board_book(args.world, state=args.boards)
  # The world's displays (issue #13): the lifecycle drives the LCD's face off
  # its own state, and every screen's content rides in the frames.
  screens = world_screens(model, data)
  # Points are world state too (issue #14), and the ROBOT's rather than the
  # room's: a balance that resets whenever the container cycles is not a
  # scoreboard. Every mission end is a restart, so the site's rivalry only
  # exists if this file does.
  ledger = points_ledger(args.ledger)
  # Job offers (issue #21), and world state on exactly the terms the boards
  # and the ledger are: a task that vanished because the container cycled is
  # a job somebody asked for and nobody ever declined. Off unless asked for.
  tasks = (task_board(args.task_state)
           if (args.tasks or args.task_state) else None)
  # The overseer decides what to do next once the preset queue is empty
  # (issue #15). Off unless asked for, and its memory is two more files in
  # the same volume the boards and the ledger live in: goals are read (and
  # human-edited between runs), the journal is written.
  overseer_kw = ({"calls_per_hour": args.overseer_budget}
                 if args.overseer_budget else {})
  boss, journal = overseer.build(args.world, book,
                                 enabled=args.overseer or None,
                                 goals_path=args.goals,
                                 journal_path=args.journal, **overseer_kw)
  # The goals file is read on every run, overseer or not: the site's goals
  # panel (rooftop-media-2026 #30) shows what the robot is FOR, and that is
  # as true of a scripted rotation as of a chosen errand. What is NOT the
  # same is whether anything is reading them, which is what `steering` says.
  goals_prose = overseer.goals_text(args.goals)
  # The visitor channel (issue #16). Only where there is somebody to hear it:
  # without an overseer nothing reads a suggestion, so accepting one would be
  # a promise the robot has no way to keep.
  inbox = Inbox() if boss is not None else None
  life = HubLifecycle(model, data, inbox=inbox,
                      battery_wh=args.battery_wh or cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"],
                      errands=errands_for(args.errand, args.world, book),
                      screen=next(iter(screens), None),
                      overseer=boss, journal=journal, world=args.world,
                      boards=book, ledger=ledger, tasks=tasks)
  # The world's task state machines, polled on the same per-step seam
  # everything else hangs off (issue #8). Their flags ride in the frames.
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))
  publisher = WsPublisher(model, data, args.endpoint,
                          model_name=cfg["model_name"],
                          status_fn=life.telemetry_status,
                          grid=life.mission.grid, token=args.token,
                          keyframe_s=args.keyframe_s,
                          activities=activities, boards=book,
                          screens=screens, ledger=ledger, tasks=tasks,
                          # What this run can actually HEAR (issue #16). Empty
                          # without an overseer, and the website reads it: a
                          # suggestion is only "delivered" if somebody who can
                          # act on it got it.
                          accepts=INBOUND_TYPES if inbox is not None else (),
                          goals=goals_prose, steering=boss is not None)
  life.mission.step_hooks.append(publisher.step_hook)
  life.say_hooks.append(publisher.event)
  if book is not None:
    # Strokes reach the browser as `draw` messages, never as geometry: the
    # website paints them into the board's canvas texture.
    book.on_event.append(publisher.message)
  # An award is an event on the same terms (issue #14): the balance rides in
  # the frames, and each `earned` message carries the verdict behind it.
  ledger.on_event.append(publisher.message)
  # ...and so is a job being offered, taken or resolved (issue #21).
  if tasks is not None:
    tasks.on_event.append(publisher.message)
  # ...and so are the robot's notes and its answers to visitors (#15, #16).
  if journal is not None:
    journal.on_event.append(publisher.message)
  life.visitor_hooks.append(publisher.message)
  if inbox is not None:
    # THE OTHER DIRECTION. `offer` runs on the publisher's socket thread and
    # does nothing but validate and enqueue -- see hub/inbox.py for why that
    # is the whole of what it is allowed to do.
    publisher.on_inbound.append(inbox.offer)
  pacer = None
  if not args.free_run:
    pacer = RealTimePacer(data, rate=args.rate)
    life.mission.step_hooks.append(pacer.step_hook)
  recorder = None
  if args.record is not None:
    recorder = TelemetryRecorder(model, data, args.record,
                                 model_name=cfg["model_name"],
                                 status_fn=life.telemetry_status,
                                 keyframe_s=args.keyframe_s,
                                 activities=activities, boards=book,
                                 screens=screens, ledger=ledger, tasks=tasks,
                                 goals=goals_prose,
                                 steering=boss is not None)
    life.mission.step_hooks.append(recorder.step_hook)
    if book is not None:
      book.on_event.append(recorder.emit)
    ledger.on_event.append(recorder.emit)
    if tasks is not None:
      tasks.on_event.append(recorder.emit)
    if journal is not None:
      journal.on_event.append(recorder.emit)
    life.visitor_hooks.append(recorder.emit)

  # ⚠ SEEDED LAST, after every hook above is attached: `offer` emits its
  # `task_offered` immediately, so seeding earlier drops those lines on the
  # floor -- see the note in `lifecycle.run_demo`. Only when nothing is
  # outstanding, so a restart resumes the jobs the last mission left.
  if tasks is not None and not tasks.open_tasks():
    seed_tasks(tasks, args.world, book, ttl=SEED_TTL_S,
               standing_ttl=SEED_STANDING_TTL_S)

  wall0 = time.monotonic()
  try:
    r = life.run(cfg["start"], use_at=cfg["use_at"],
                 max_sim_time=args.max_sim_time,
                 explore_budget=cfg["explore_budget"])
  finally:
    publisher.close()
    if recorder is not None:
      recorder.close()
  wall = time.monotonic() - wall0

  print()
  print(f"mission state          : {r['state']}"
        f" (swaps={r['swaps_done']}, charges={r['charge_cycles']},"
        f" stowed={r['module_stowed']})")
  for e in r["errands"]:
    extra = (f"  {e['figure']} on {e['board']}, board {e['fill']:.0%} full"
             if e.get("board") else "")
    print(f"errand {e['errand']:<20s}: picked={e['picked']}"
          f" stowed={e['stowed']}{extra}")
  print(f"points balance         : {r['points']}"
        f" ({r['earned']} earned over {len(r['verdicts'])} scored task(s))")
  # `.get`, not `[...]`: this is a SUMMARY, and a summary line must never be
  # the thing that fails a run. tests/test_webserver.py drives this path with
  # a stubbed result dict, which is exactly the shape a caller on an older
  # lifecycle would hand it.
  if r.get("overseer"):
    o = r["overseer"]
    # Cost per SIM-hour, which is the number that matters for a box that runs
    # paced to real time: at --rate 1.0 a sim-hour is an hour of electricity.
    per_hour = o["usd"] / (r["sim_time"] / 3600.0) if r["sim_time"] else 0.0
    print(f"overseer               : {o['llmCalls']} LLM call(s), "
          f"{o['fallbacks']} scripted, budget {o['budgetLeft']}/"
          f"{o['callsPerHour']} left, cache hit {o['cacheHitRate']:.0%}")
    print(f"overseer cost          : ${o['usd']:.5f}"
          f"  (${per_hour:.4f} per sim-hour, {o['model']})")
    for err in o["errors"]:
      # A run that was scripted all along looks identical to a thoughtful one
      # from the outside unless this is printed.
      print(f"overseer fell back on  : {err}")
  if r.get("visitors"):
    v = r["visitors"]
    print(f"visitors               : {v['received']} message(s), "
          f"{v['delivered']} answered, {v['queued']} still waiting")
    if v["droppedInvalid"] or v["droppedFull"]:
      # Both are normal in small numbers and a story in large ones: garbage
      # is somebody probing, a full queue is the channel outrunning the robot.
      print(f"visitors dropped       : {v['droppedInvalid']} malformed, "
            f"{v['droppedFull']} overflowed the queue")
    for reply in r.get("replies", ()):
      print(f"visitor {reply['outcome']:<14s}: {reply['reply']}")
  print(f"sim / wall             : {r['sim_time']:.1f} s / {wall:.1f} s"
        f"  ({r['sim_time'] / wall:.2f}x real time)")
  if pacer is not None:
    s = pacer.stats()
    print(f"pacing (target {s['rate']:.2f}x)  : drift {s['drift_s']:+.3f} s"
          f" at close, worst transient lag {s['max_lag_s']:.3f} s")
  print(f"frames sent / dropped  : {publisher.frames_sent}"
        f" / {publisher.frames_dropped}"
        f"  (connections: {publisher.connections})")
  if publisher.connections == 0 and publisher.last_error is not None:
    # A rejected token retries exactly like a down server; say which.
    print(f"never connected        : {publisher.last_error}")


if __name__ == "__main__":
  main()
