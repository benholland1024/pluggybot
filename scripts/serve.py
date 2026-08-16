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

from pluggybot.hub.lifecycle import (
  HubLifecycle, board_book, errands_for, world_config, world_screens,
)
from pluggybot.telemetry.pacer import RealTimePacer
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
                                          "dance", "showcase", "none"),
                      default="carry",
                      help="what the robot is FOR this run (issue #12): carry "
                           "(the milestone-8 LCD errand), draw (fetch the pen, "
                           "erase a whiteboard and draw on it), draw2 (two "
                           "boards, charging in between), none")
  parser.add_argument("--boards", default=None, metavar="PATH",
                      help="JSON file the whiteboards' contents live in "
                           "between runs (default: blank boards every start)")
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
  life = HubLifecycle(model, data,
                      battery_wh=args.battery_wh or cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"],
                      errands=errands_for(args.errand, args.world, book),
                      screen=next(iter(screens), None),
                      boards=book)
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
                          screens=screens)
  life.mission.step_hooks.append(publisher.step_hook)
  life.say_hooks.append(publisher.event)
  if book is not None:
    # Strokes reach the browser as `draw` messages, never as geometry: the
    # website paints them into the board's canvas texture.
    book.on_event.append(publisher.message)
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
                                 screens=screens)
    life.mission.step_hooks.append(recorder.step_hook)
    if book is not None:
      book.on_event.append(recorder.emit)

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
