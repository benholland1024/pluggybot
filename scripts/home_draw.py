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
  MUJOCO_GL=egl uv run python scripts/home_draw.py --shape circle
  MUJOCO_GL=egl uv run python scripts/home_draw.py --board whiteboard_b
"""

import argparse
import json
import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.hub.drawing import PATHS, Board, PenPlotter
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
  """Commanded figure vs inked path, board coordinates."""
  img = Image.new("RGB", (size, size), BG)
  d = ImageDraw.Draw(img)
  pts = plotter.commanded
  half = max(max(abs(y), abs(z)) for y, z in
             ((p[0] - plotter.cal["y0"], p[1] - plotter.cal["z0"])
              for p in pts)) + 0.02
  cy0, cz0 = plotter.cal["y0"], plotter.cal["z0"]

  def to_px(by, bz):
    return (size / 2 + (by - cy0) / half * size / 2.4,
            size / 2 - (bz - cz0) / half * size / 2.4)

  d.line([to_px(y, z) for y, z in pts], fill=DIM, width=2)
  inked = [(t[1], t[2]) for t in plotter.trace if t[5]]
  for by, bz in inked:
    px, py = to_px(by, bz)
    d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=RED)
  d.text((12, 10), "commanded (grey) vs inked (red)", fill=INK)
  return img


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--view", action="store_true", help="open the viewer")
  parser.add_argument("--fast", action="store_true",
                      help="with --view: no real-time pacing")
  parser.add_argument("--shape", choices=sorted(PATHS), default="square")
  parser.add_argument("--size", type=float, default=0.075)
  parser.add_argument("--board", default="whiteboard_a",
                      choices=sorted(home.BOARDS))
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
  aborted = False
  try:
    mission.start_at(*home.SPAWNS["start"])
    mission.start_discovery()
    mission._spin()

    why = mission.swap_at_bay(PEN_BAY, "pick", module="module_pen")
    picked = mission.swap.module_state("module_pen")["on_fork"]
    powered = module_power_contact(model, data, "module_pen")
    print(f"pick ({why}): on fork {picked}, powered {powered}")

    plotter = PenPlotter(model, data, mission.swap, board=board)
    tx, ty = plotter.board_standoff()
    mission.drive_to(tx, ty, timeout=90.0)
    squared = plotter.drive_to_board()
    print(f"at the board ({args.board}): {squared}")
    result = plotter.draw(PATHS[args.shape](args.size))
    photo = board_photo(model, data, board)

    plotter.carry_config()
    why2 = mission.swap_at_bay(PEN_BAY, "return", module="module_pen")
    stowed = mission.swap.module_state("module_pen")["hung"]
    print(f"stow ({why2}): hung {stowed}")
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
  for key in ("inked_fraction", "form_rms_mm", "form_max_mm",
              "shape_rms_mm", "offset_mm"):
    if result.get(key) is not None:
      print(f"{key:16s} {result[key]:.2f}")
  drew_ok = (picked and result.get("drew")
             and result["inked_fraction"] > 0.9)
  print(f"sim time {data.time:.1f}s  collisions {mission.collision_steps}")
  print("FETCH + DRAW:", "OK" if drew_ok else "FAILED")
  # Reported separately, and deliberately: the pen's stow after a navigated
  # errand is a KNOWN pre-existing failure -- it fails the same way in
  # room_hub (SimNotes, "The home world ... and a stow gap"). Rolling it
  # into one verdict would either hide a working drawing behind a known
  # bug or, worse, quietly redefine success to exclude it.
  print("STOW:", "OK" if stowed else "FAILED (known open item -- SimNotes)")
  print(f"-> {OUT}")


if __name__ == "__main__":
  main()
