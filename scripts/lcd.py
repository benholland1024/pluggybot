"""The LCD module's demo (issue #13): a face, a census, and a dance.

The last module to get a job, and the first whose OUTPUT is not physical. The
robot fetches `module_lcd` from bay A and then either

  --errand census   surveys the garden, counts the plants it can see, and
                    puts the number on the screen. Scored against hidden
                    ground truth read out of the model -- being lazy is a
                    real way to be wrong here, which no earlier task in this
                    repo could manage.
  --errand dance    performs a fixed routine with an expression per move,
                    scored on whether the moves happened and whether it
                    stayed where it started.

WHAT THIS DEMO CAN AND CANNOT SHOW. The face is drawn in the BROWSER (layer 3
of the three-layer model): what the sim owns is an enum plus an animation
hint, and MuJoCo renders a dark panel whatever the robot is feeling. So the
filmstrip shows the module and the timeline shows the state -- and the state
is the whole artifact, because it is exactly what goes on the wire for the
website to paint (rooftop-media-2026 #28).

Usage:
  MUJOCO_GL=egl uv run python scripts/lcd.py                 # census + lcd.png
  MUJOCO_GL=egl uv run python scripts/lcd.py --errand dance
  uv run python scripts/lcd.py --view                        # watch it live
"""

import argparse
import json
import math

import mujoco
from PIL import Image, ImageDraw

from pluggybot.hub.lifecycle import (
  HubLifecycle, board_book, errands_for, world_config, world_screens,
)

FRAME_W, FRAME_H = 420, 315
OUT = "lcd.png"
MAX_SHOTS = 8
BG, PANEL, INK, DIM = (24, 26, 30), (38, 41, 47), (232, 234, 238), (120, 126, 136)
GREEN, RED, BLUE = (61, 220, 132), (232, 92, 84), (108, 168, 255)


def run(errand: str, view: bool, realtime: bool, world: str,
        battery_wh: float | None, max_sim_time: float) -> tuple[dict, list]:
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)

  book = board_book(world)
  screens = world_screens(model, data)
  screen = next(iter(screens))
  life = HubLifecycle(model, data, viewer=viewer, realtime=realtime,
                      battery_wh=battery_wh or cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], boards=book,
                      screen=screen,
                      errands=errands_for(errand, world, book))
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))

  # The timeline IS the deliverable: every state the screen took, with the
  # sim time it took it at. Recorded off the same `on_change` hook telemetry
  # would use, so what this plots and what the website receives are one thing.
  timeline: list[tuple[float, dict]] = []
  renderer = None if view else mujoco.Renderer(model, FRAME_H, FRAME_W)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  shots: list[tuple[str, object]] = []
  lcd_bid = model.body("module_lcd").id

  def grab(label: str) -> None:
    if renderer is None or len(shots) >= MAX_SHOTS:
      return
    # Frame the MODULE, not the robot: on the rack or on the fork, the panel
    # is what this demo is about, and a whole-room shot renders it as three
    # dark pixels.
    #
    # The azimuth is COMPUTED from where the panel is actually pointing, not
    # picked. A fixed angle framed the rack bays from behind the south wall
    # and rendered two black rectangles: the module turns 90 degrees between
    # hanging in its bay and riding on the fork, so no constant can see the
    # face in both. The panel is the module's -x face (scene `screens` says
    # so), and MuJoCo's azimuth is where the CAMERA stands relative to the
    # subject -- so standing along the outward normal looks back at the face.
    at = data.xpos[lcd_bid]
    rot = data.xmat[lcd_bid].reshape(3, 3)
    out = -rot[:, 0]                       # body -x in world coordinates
    cam.lookat[:] = [at[0], at[1], at[2] + 0.04]
    cam.distance = 0.9
    cam.azimuth = math.degrees(math.atan2(out[1], out[0]))
    cam.elevation = -10
    renderer.update_scene(data, cam)
    shots.append((label, renderer.render().copy()))

  def on_change(name: str, flags: dict) -> None:
    timeline.append((float(data.time), dict(flags)))
    if flags["mode"] != "off":
      grab(f"t={data.time:5.1f}s  {flags.get('face')}"
           f"{'' if flags['mode'] == 'face' else '  ' + str(flags.get('count', flags.get('text')))}")

  screen.on_change.append(on_change)

  try:
    result = life.run(cfg["start"], use_at=cfg["use_at"],
                      max_sim_time=max_sim_time,
                      explore_budget=cfg["explore_budget"])
  finally:
    if viewer is not None:
      viewer.close()
    if renderer is not None:
      renderer.close()

  result["screen_timeline"] = [(round(t, 2), f) for t, f in timeline]
  result["faces_seen"] = sorted({f.get("face") for _, f in timeline
                                 if f.get("face")})
  result["lit_s"] = round(sum(
    (timeline[i + 1][0] if i + 1 < len(timeline) else result["sim_time"]) - t
    for i, (t, f) in enumerate(timeline) if f["mode"] != "off"), 1)
  return result, shots


def render(result: dict, shots: list, path: str = OUT) -> str:
  """Filmstrip of the module + the screen's state timeline underneath."""
  cols = max(len(shots), 1)
  strip_h = FRAME_H + 26 if shots else 0
  rows = [line for line in _report(result)]
  card_h = 22 * len(rows) + 24
  img = Image.new("RGB", (max(cols * FRAME_W, 720), strip_h + card_h), BG)
  draw = ImageDraw.Draw(img)
  for i, (label, frame) in enumerate(shots):
    img.paste(Image.fromarray(frame), (i * FRAME_W, 0))
    draw.text((i * FRAME_W + 8, FRAME_H + 6), label, fill=INK)
  y = strip_h + 12
  for text, colour in rows:
    draw.text((14, y), text, fill=colour)
    y += 22
  img.save(path)
  return path


def _report(result: dict):
  census = next((e.get("census") for e in result.get("errands", [])
                 if e.get("census")), None)
  dance = next((e.get("dance") for e in result.get("errands", [])
                if e.get("dance")), None)
  yield (f"mission {result['state']}  |  {result['sim_time']:.0f} s  |  "
         f"battery {result['battery']:.0%}  |  swaps {result['swaps_done']}",
         INK)
  yield (f"screen: {len(result['screen_timeline'])} changes, lit "
         f"{result['lit_s']} s, faces {', '.join(result['faces_seen']) or '-'}",
         BLUE)
  if census:
    yield (f"census: reported {census['counted']}, truth {census['truth']}, "
           f"{census['coverage']:.0%} of the zone surveyed  -- "
           f"{'CORRECT' if census['correct'] else 'WRONG'}",
           GREEN if census["correct"] else RED)
  if dance:
    yield (f"dance: {dance['landed']}/{dance['moves']} moves landed, "
           f"{dance['driftM']:.2f} m of drift  -- "
           f"{'COMPLETE' if dance['complete'] else 'INCOMPLETE'}",
           GREEN if dance["complete"] else RED)
  yield (f"module stowed: {result['module_stowed']}",
         GREEN if result["module_stowed"] else RED)
  for t, flags in result["screen_timeline"][:12]:
    extra = ""
    if flags["mode"] == "count":
      extra = f"  {flags['count']} {flags['label']}"
    elif flags["mode"] == "text":
      extra = f"  {flags['text']!r}"
    yield (f"  t={t:6.1f}  {flags['mode']:<5} "
           f"{flags.get('face', '-'):<11} {flags.get('hint', '')}{extra}",
           DIM)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--errand", choices=("census", "dance", "carry"),
                  default="census")
  ap.add_argument("--world", choices=("home", "room_hub"), default="home")
  ap.add_argument("--view", action="store_true",
                  help="watch it live in the MuJoCo viewer; skips the film")
  ap.add_argument("--fast", action="store_true",
                  help="with --view, run as fast as the machine allows")
  ap.add_argument("--battery-wh", type=float, default=None)
  ap.add_argument("--max-sim-time", type=float, default=900.0)
  ap.add_argument("--out", default=OUT)
  args = ap.parse_args()

  result, shots = run(args.errand, args.view, not args.fast, args.world,
                      args.battery_wh, args.max_sim_time)
  print(json.dumps({k: v for k, v in result.items()
                    if k not in ("screen_timeline",)}, indent=1, default=str))
  if shots or not args.view:
    print("wrote", render(result, shots, args.out))


if __name__ == "__main__":
  main()
