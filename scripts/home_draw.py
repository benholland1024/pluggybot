"""The home-world drawing errand (issue #6): fetch the pen, draw, stow.

The milestone-8 gap this closes: the pen only ever drew in the bare
hub_world, because no room world had a drawing surface. The generated home
world hangs whiteboards on real walls, so the full errand now runs in a
navigated world: pick the pen module from bay C, A* across the living room,
square up to a wall-mounted board, plot a figure through the sprung quill,
restore the carry configuration, drive back and hang the tool up.

Writes home_draw.png: a photo of the board with the robot at work, plus the
commanded-vs-inked trace in board coordinates.

Usage:
  MUJOCO_GL=egl uv run python scripts/home_draw.py             # headless
  uv run python scripts/home_draw.py --view                    # watch live
  MUJOCO_GL=egl uv run python scripts/home_draw.py --program circle
  MUJOCO_GL=egl uv run python scripts/home_draw.py --program text \
      --text "GOOD MORNING"
  MUJOCO_GL=egl uv run python scripts/home_draw.py --board whiteboard_b
"""

import argparse
import json
import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.hub.drawing import Board, Envelope, PenPlotter
from pluggybot.hub.strokes import PROGRAMS, from_cli
from pluggybot.hub.localize import RackPose
from pluggybot.hub.mission import HubMission, MissionAborted
from pluggybot.home import world as home

OUT = "home_draw.png"
PEN_BAY = HUB_STATION_YS[2]     # bay C, same as draw.py

BG = (24, 26, 30)
INK = (232, 234, 238)
DIM = (120, 126, 136)
RED = (226, 96, 82)
BLUE = (110, 168, 254)


def board_photo(model, data, board: Board, w: int = 480, h: int = 640):
  """A camera looking at the board along its outward normal."""
  cam = mujoco.MjvCamera()
  cam.lookat[:] = [board.x, board.y, board.z]
  cam.distance = 1.1
  cam.elevation = -10
  cam.azimuth = math.degrees(board.heading)
  renderer = mujoco.Renderer(model, h, w)
  renderer.update_scene(data, camera=cam)
  frame = renderer.render().copy()
  renderer.close()
  return frame


def trace_panel(plotter, size: int = 640) -> Image.Image:
  """Commanded figure vs inked path, board coordinates.

  Rendered the way SOMEBODY STANDING AT THE BOARD sees it: board `lat` is
  measured to the left of the robot's approach, so the panel's x axis is
  flipped. That was invisible while every figure was a square or a circle --
  both are mirror-symmetric -- and would have shown the first line of Hershey
  text back to front next to a photo of it reading correctly, which is a
  spectacular way to spend an afternoon debugging a font.
  """
  img = Image.new("RGB", (size, size), BG)
  d = ImageDraw.Draw(img)
  cy0, cz0 = plotter.cal["y0"], plotter.cal["z0"]
  half = max(max(abs(y - cy0), abs(z - cz0))
             for stroke in plotter.commanded for y, z in stroke) + 0.02

  def to_px(by, bz):
    return (size / 2 - (by - cy0) / half * size / 2.4,
            size / 2 - (bz - cz0) / half * size / 2.4)

  for stroke in plotter.commanded:
    d.line([to_px(y, z) for y, z in stroke], fill=DIM, width=2)
  for t in plotter.trace:
    if not t[5]:
      continue
    px, py = to_px(t[1], t[2])
    # Ink laid down between strokes is a fault, not a drawing: colour it.
    d.ellipse([px - 1, py - 1, px + 1, py + 1],
              fill=RED if t[6] >= 0 else BLUE)
  d.text((12, 10), "commanded (grey) vs inked (red)", fill=INK)
  return img


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--view", action="store_true", help="open the viewer")
  parser.add_argument("--fast", action="store_true",
                      help="with --view: no real-time pacing")
  parser.add_argument("--program", "--shape", dest="program",
                      choices=sorted(PROGRAMS), default="square",
                      help="what to draw (issue #11): square and circle are "
                           "the diagnostics, the rest are content")
  parser.add_argument("--text", default=None,
                      help="what --program text writes")
  parser.add_argument("--size", type=float, default=None,
                      help="figure box in metres -- cap height, for text")
  parser.add_argument("--board", default="whiteboard_a",
                      choices=sorted(home.BOARDS))
  parser.add_argument("--cycles", type=int, default=1,
                      help="repeat fetch -> draw -> stow this many times. "
                           "The point of >1 is the HAND-OFF: a stow that "
                           "leaves the rack's believed state wrong shows up "
                           "on the next fetch, not on the stow itself")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  viewer = None
  if args.view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)

  rack = RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                  math.radians(home.HOME_RACK_YAW))
  mission = HubMission(model, data, viewer=viewer, realtime=not args.fast,
                       rack=rack, grid_bounds=home.GRID_BOUNDS)
  meta = json.load(open("models/home_world.meta.json"))
  board = Board.from_meta(meta["boards"][args.board])
  # Size the figure against what the PEN can reach, which is nothing like the
  # board: 320 mm of whiteboard, 110 mm of carriage travel, base parked.
  envelope = Envelope.for_board(board)
  figure = from_cli(args.program, args.size, args.text)
  if not figure.fits(envelope):
    print(f"{figure.name} is {figure.size[0] * 1000:.0f} x "
          f"{figure.size[1] * 1000:.0f} mm; the pen can reach "
          f"{envelope.size[0] * 1000:.0f} x {envelope.size[1] * 1000:.0f} mm "
          f"-- shrinking to fit")
    figure = figure.fitted(envelope)
  print(f"{figure.name}: {len(figure.strokes)} strokes, "
        f"{figure.size[0] * 1000:.0f} x {figure.size[1] * 1000:.0f} mm, "
        f"{figure.ink_length:.2f} m of ink")
  aborted = False
  cycles = []
  try:
    mission.start_at(*home.SPAWNS["start"])
    mission.start_discovery()
    mission._spin()

    for n in range(max(args.cycles, 1)):
      if args.cycles > 1:
        print(f"\n--- cycle {n + 1}/{args.cycles} ---")
      why = mission.swap_at_bay(PEN_BAY, "pick", module="module_pen")
      picked = mission.swap.module_state("module_pen")["on_fork"]
      powered = module_power_contact(model, data, "module_pen")
      print(f"pick ({why}): on fork {picked}, powered {powered}")

      plotter = PenPlotter(model, data, mission.swap, board=board)
      tx, ty = plotter.board_standoff()
      mission.drive_to(tx, ty, timeout=90.0)
      squared = plotter.drive_to_board()
      print(f"at the board ({args.board}): {squared}")
      result = plotter.draw_program(figure)
      photo = board_photo(model, data, board)

      plotter.carry_config()
      why2 = mission.swap_at_bay(PEN_BAY, "return", module="module_pen")
      stowed = mission.swap.module_state("module_pen")["hung"]
      print(f"stow ({why2}): hung {stowed}")
      cycles.append({"picked": picked, "powered": powered, "stowed": stowed,
                     "inked": result.get("inked_fraction"),
                     "drew": result.get("drew")})
  except MissionAborted:
    aborted = True
  finally:
    mission.close()
    if viewer is not None:
      viewer.close()
  if aborted:
    print("aborted (viewer closed)")
    return

  panel = trace_panel(plotter)
  photo_img = Image.fromarray(np.asarray(photo))
  out = Image.new("RGB", (photo_img.width + panel.width,
                          max(photo_img.height, panel.height)), BG)
  out.paste(photo_img, (0, 0))
  out.paste(panel, (photo_img.width, 0))
  out.save(OUT)

  print()
  for key in ("inked_fraction", "travel_ink_fraction", "form_rms_mm",
              "form_max_mm", "shape_rms_mm", "offset_mm"):
    if result.get(key) is not None:
      print(f"{key:16s} {result[key]:.2f}")
  print(f"sim time {data.time:.1f}s  collisions {mission.collision_steps}")
  # A stow verdict only means anything if something was FETCHED first: a
  # module that never left the rack is still hanging in it, and `hung` says
  # True. The two-cycle run caught exactly that -- a failed second fetch
  # followed by a "stow OK" for a tool the robot never had.
  if len(cycles) > 1:
    for n, c in enumerate(cycles):
      stow = ("n/a" if not c["picked"] else "OK" if c["stowed"] else "FAILED")
      print(f"cycle {n + 1}: fetch {'OK' if c['picked'] and c['powered'] else 'FAILED'}"
            f"  draw {'OK' if c['drew'] and (c['inked'] or 0) > 0.9 else 'FAILED'}"
            f"  stow {stow}")
  # Fetch, draw and stow are still reported SEPARATELY, though the stow is
  # no longer a known failure (issue #10). They fail for unrelated reasons
  # and a single verdict would say which of them broke only by accident.
  drew_ok = all(c["picked"] and c["drew"] and (c["inked"] or 0) > 0.9
                for c in cycles)
  exercised = [c for c in cycles if c["picked"]]
  print("FETCH + DRAW:", "OK" if drew_ok else "FAILED")
  print("STOW:", "n/a (nothing was fetched)" if not exercised
        else "OK" if all(c["stowed"] for c in exercised) else "FAILED")
  print(f"-> {OUT}")


if __name__ == "__main__":
  main()
