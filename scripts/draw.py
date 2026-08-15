"""Drawing tool demo (milestone 8): the robot picks up an axis it doesn't have.

The fork robot fetches the PEN module from the rack's third bay, carries it to
a board, and plots a figure using an X-Y gantry it only owns while holding the
tool:

  Y   the module's own carriage, along the peg axis -- a differential drive
      cannot translate sideways, so this axis exists nowhere else on the robot
  Z   the robot's lift
  pen the arm's reach, through a sprung quill that sets pressure

Writes draw.png: what was commanded, what the pen actually traced, and the
error between them. MuJoCo cannot paint, so the "ink" is the recorded pen-tip
path filtered to the steps where the tip was genuinely in contact -- a
commanded path is a belief, contact is a fact.

Usage:
  uv run python scripts/draw.py --view                  # watch it live
  MUJOCO_GL=egl uv run python scripts/draw.py           # headless + draw.png
  MUJOCO_GL=egl uv run python scripts/draw.py --shape circle
  MUJOCO_GL=egl uv run python scripts/draw.py --record draw.mp4 --record-speed 3
"""

import argparse
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.hub.drawing import PATHS, PEN_MODULE, PenPlotter
from pluggybot.hub.swap import HubSwap
from pluggybot.viz import Recorder

FRAME_W, FRAME_H = 400, 300
PLOT = 900
OUT = "draw.png"
BG = (24, 26, 30)
PANEL = (38, 41, 47)
INK = (232, 234, 238)
DIM = (120, 126, 136)
GREEN = (61, 220, 132)
RED = (232, 92, 84)
BLUE = (99, 164, 255)


def run(shape: str, view: bool, realtime: bool, record: str | None = None,
        record_fps: int = 30, record_speed: float = 1.0):
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)

  viewer = None
  frames: list[tuple[str, np.ndarray]] = []
  renderer = cam = None
  hooks = []
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
    state = {"next": 0.0, "wall": time.time()}

    def hook():
      if data.time < state["next"]:
        return
      state["next"] = data.time + 0.02
      if not viewer.is_running():
        raise KeyboardInterrupt("viewer closed")
      viewer.sync()
      if realtime:
        ahead = data.time - (time.time() - state["wall"])
        if ahead > 0:
          time.sleep(min(ahead, 0.05))

    hooks.append(hook)
  else:
    renderer = mujoco.Renderer(model, FRAME_H, FRAME_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(PEN_MODULE).id
    cam.distance, cam.azimuth, cam.elevation = 0.85, 150, -12

  # Recording is independent of --view: the video comes from its own offscreen
  # renderer, so it is identical whether or not a window is open.
  #
  # The camera MOVES with the story, because no single azimuth covers this
  # demo. az=150 (the filmstrip's angle) is right at the rack but sits BEHIND
  # the board -- the board is a thin slab at x=1.30 and the pen draws on its
  # -x face, so the entire drawing phase renders as a grey rectangle. az=60
  # sees the face being written on but puts a wall through the lens back at
  # the rack. Both angles were picked by sweeping azimuth at the moment the
  # pen was confirmed in contact, not by reasoning about the geometry.
  recorder = None
  if record:
    recorder = Recorder(model, record, track_body=PEN_MODULE, fps=record_fps,
                        speed=record_speed, distance=0.85, azimuth=150,
                        elevation=-12)
    hooks.append(lambda: recorder.maybe_grab(data))

  if hooks:
    swap.on_step = lambda: [h() for h in hooks]

  def grab(label):
    if renderer is None:
      return
    renderer.update_scene(data, cam)
    frames.append((label, renderer.render().copy()))

  def stage(text, **camera):
    """Video caption for the phase about to run, plus an optional camera move.

    The filmstrip's labels name the state just reached, which reads a beat
    late over moving footage, so the video gets its own forward-looking ones.
    """
    if recorder is not None:
      recorder.set_label(text)
      if camera:
        recorder.set_camera(**camera)

  stage("approaching the rack")
  swap.place_at_standoff(HUB_STATION_YS[2])
  grab("at the rack")
  stage("picking up the pen module")
  swap.pick()
  powered = module_power_contact(model, data, PEN_MODULE)
  grab("tool picked")

  plotter = PenPlotter(model, data, swap)
  # Pan to the board-facing angle while the robot drives there -- the one
  # moment in the demo when a camera move reads as intentional.
  stage("carrying the pen to the board",
        azimuth=60, elevation=-18, distance=1.0)
  arrived = plotter.drive_to_board()
  grab("at the board")
  stage(f"drawing a {shape}")
  result = plotter.draw(PATHS[shape]())
  grab("drawn")

  result.update({
    "arrived": arrived,
    "powered_at_pick": powered,
    "seated_after": module_power_contact(model, data, PEN_MODULE),
    "sim_time": float(data.time),
  })
  # AFTER the result dict: a hold on the finished figure so the clip does not
  # cut on the last frame of motion. Recording must not move the numbers.
  if recorder is not None:
    stage(f"{shape} complete")
    swap._run(1.5, 0.0)

  if viewer is not None:
    viewer.close()
  if renderer is not None:
    renderer.close()
  if recorder is not None:
    print(f"wrote {recorder.close()} "
          f"({recorder.frames_written} frames, {record_fps} fps)")
  return plotter, result, frames


def axis_errors(plotter) -> tuple[float, float]:
  inked = [t for t in plotter.trace if t[5]]
  if not inked:
    return 0.0, 0.0
  dy = np.array([t[1] - t[3] for t in inked])
  dz = np.array([t[2] - t[4] for t in inked])
  return float(np.sqrt((dy ** 2).mean()) * 1000), \
      float(np.sqrt((dz ** 2).mean()) * 1000)


def draw_plot(img: Image.Image, plotter, x0: int, y0: int, size: int) -> None:
  """Commanded figure vs the ink the pen actually laid down."""
  d = ImageDraw.Draw(img)
  d.rectangle([x0, y0, x0 + size, y0 + size], fill=PANEL)
  cmd = np.array(plotter.commanded)
  inked = [t for t in plotter.trace if t[5]]
  lifted = [t for t in plotter.trace if not t[5]]
  cy, cz = cmd[:, 0].mean(), cmd[:, 1].mean()
  span = max(np.ptp(cmd[:, 0]), np.ptp(cmd[:, 1])) * 1.45 or 0.1
  scale = size / span

  def px(by, bz):
    # +y is to the viewer's LEFT when facing the board, so flip x to render
    # the figure the way somebody standing at the board would see it.
    return (x0 + size / 2 - (by - cy) * scale,
            y0 + size / 2 - (bz - cz) * scale)

  for k in range(-2, 3):                       # 25 mm grid
    g = k * 0.025
    d.line([px(g, -span), px(g, span)], fill=(52, 56, 62))
    d.line([px(-span, g), px(span, g)], fill=(52, 56, 62))
  d.line([px(p[0], p[1]) for p in cmd], fill=DIM, width=3)
  if lifted:
    for t in lifted[::4]:
      x, y = px(t[1], t[2])
      d.point((x, y), fill=RED)
  if inked:
    d.line([px(t[1], t[2]) for t in inked], fill=GREEN, width=2)
  d.text((x0 + 12, y0 + 10), "commanded", fill=DIM)
  d.text((x0 + 12, y0 + 26), "traced (pen down)", fill=GREEN)
  d.text((x0 + 12, y0 + 42), "pen lifted off", fill=RED)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--shape", choices=sorted(PATHS), default="circle",
                  help="circle exercises both axes continuously; square holds "
                       "each axis at an extreme, which is more diagnostic")
  ap.add_argument("--view", action="store_true",
                  help="watch it live in the MuJoCo viewer; skips the filmstrip")
  ap.add_argument("--fast", action="store_true",
                  help="with --view: run flat out instead of real time")
  ap.add_argument("--record", metavar="PATH",
                  help="record the tracking camera to PATH (.mp4 or .gif); "
                       "720p, independent of --view")
  ap.add_argument("--record-fps", type=int, default=30,
                  help="video frame rate (default 30)")
  ap.add_argument("--record-speed", type=float, default=1.0,
                  help="sim seconds per played second: 1.0 real time, "
                       "3.0 timelapse, 0.25 slow motion (default 1.0)")
  args = ap.parse_args()

  try:
    plotter, r, frames = run(args.shape, args.view, not args.fast,
                             args.record, args.record_fps, args.record_speed)
  except KeyboardInterrupt:
    print("aborted (viewer closed)")
    return

  ey, ez = axis_errors(plotter)
  print(f"\nshape               {args.shape}")
  print(f"tool powered        pick={r['powered_at_pick']} "
        f"after={r['seated_after']}")
  print(f"inked               {r['inked_fraction']:.0%} of the path")
  print(f"shape error         {r['shape_rms_mm']:.2f} mm rms, "
        f"{r['shape_max_mm']:.2f} mm max   <- absolute, where it landed")
  print(f"  rigid offset      {r['offset_mm']:.2f} mm "
        f"(y {r['offset_y_mm']:+.2f}, z {r['offset_z_mm']:+.2f})"
        f"   <- a calibration constant")
  print(f"  FORM error        {r['form_rms_mm']:.2f} mm rms, "
        f"{r['form_max_mm']:.2f} mm max   <- is it the right SHAPE")
  print(f"track error         {r['track_rms_mm']:.2f} mm rms, "
        f"{r['track_max_mm']:.2f} mm max   <- includes following lag")
  print(f"per axis (rms)      carriage {ey:.2f} mm, lift {ez:.2f} mm")
  print(f"sim time            {r['sim_time']:.1f} s")

  cols = max(len(frames), 1)
  top = FRAME_H if frames else 0
  sheet = Image.new("RGB", (max(FRAME_W * cols, PLOT + 520), top + PLOT + 40), BG)
  for i, (label, img) in enumerate(frames):
    sheet.paste(Image.fromarray(img), (i * FRAME_W, 0))
  d = ImageDraw.Draw(sheet)
  for i, (label, _) in enumerate(frames):
    d.rectangle([i * FRAME_W, 0, (i + 1) * FRAME_W - 1, FRAME_H - 1],
                outline=PANEL)
    d.text((i * FRAME_W + 12, 10), label, fill=INK)
  draw_plot(sheet, plotter, 20, top + 20, PLOT)

  tx = PLOT + 60
  ty = top + 40
  for line, colour in (
      (f"{args.shape}, drawn by module_pen", INK),
      ("", INK),
      (f"FORM error    {r['form_rms_mm']:6.2f} mm rms", GREEN),
      (f"              {r['form_max_mm']:6.2f} mm max", GREEN),
      ("  is it the right SHAPE -- rigid", DIM),
      ("  offset removed first", DIM),
      ("", INK),
      (f"rigid offset  {r['offset_mm']:6.2f} mm", BLUE),
      (f"   y {r['offset_y_mm']:+6.2f}  z {r['offset_z_mm']:+6.2f}", BLUE),
      ("  where it landed. a calibration", DIM),
      ("  constant, not a mechanics problem", DIM),
      ("", INK),
      (f"shape error   {r['shape_rms_mm']:6.2f} mm rms", INK),
      ("  absolute: form + offset together", DIM),
      (f"track error   {r['track_rms_mm']:6.2f} mm rms", INK),
      ("  same-instant error, includes lag", DIM),
      ("", INK),
      (f"carriage axis {ey:6.2f} mm rms", INK),
      (f"lift axis     {ez:6.2f} mm rms", INK),
      ("", INK),
      (f"inked         {r['inked_fraction']:6.0%} of path", INK),
      (f"tool seated   {r['seated_after']}", INK),
  ):
    d.text((tx, ty), line, fill=colour)
    ty += 22
  sheet.save(OUT)
  print(f"\nwrote {OUT}")


if __name__ == "__main__":
  main()
