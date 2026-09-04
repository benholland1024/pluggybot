"""Activity demo (issue #8, reworked by #93): a pressure plate turns on a light.

The reference consumer of the activity pattern (`docs/ActivityPattern.md`).
The robot drives onto a sprung plate in the garden; a joint sensor crossing a
threshold flips a live `pressed` flag, and that latches an irreversible
`state: on` which lights the bulb on the pole beside the plate. Then the
robot drives off, `pressed` goes false again, and the light STAYS on --
which is the whole point of the latch: one visit leaves a visible mark.

What each layer owns, and it is worth watching for it in the output:

  MUJOCO      the plate's sprung slide joint, and the wheel that pushes it.
              That is all. The bulb does not glow -- rgba changes, light
              does not radiate.
  PYTHON      "is it pressed" (threshold + hysteresis) and "is it on"
              (a latch). Discrete world state, remembered here because the
              physics has no restoring force to remember it with.
  THE BROWSER everything continuous -- in a real lamp, the glow, the moths
              -- keyed to the streamed flags, never simulated.

Note what the sim CAN do and the pose stream cannot: the bulb sits on a
STATIC body, so it ships once in the scene description and never again. Its
colour flips by `geom_rgba` toggle, which the robot's own cameras see and
the telemetry poses do not. The activity flag is not a convenience
duplicating the poses -- for a change like this it is the only channel
there is.

Usage:
  uv run python scripts/plate.py --view          # watch it live
  MUJOCO_GL=egl uv run python scripts/plate.py   # headless + plate.png
"""

import argparse
import math
import time

import mujoco
from PIL import Image, ImageDraw

from pluggybot.activity.base import ActivitySet
from pluggybot.activity.plate import PlateLight, plate_center
from pluggybot.control import slew, wheel_targets

FRAME_W, FRAME_H = 400, 300
OUT = "plate.png"
APPROACH = 1.3            # m of straight runway before the plate
DRIVE_V = 0.25
BG, PANEL, INK, DIM = (24, 26, 30), (38, 41, 47), (232, 234, 238), (120, 126, 136)
GREEN, RED = (61, 220, 132), (232, 92, 84)


def run(view: bool, realtime: bool):
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  acts = ActivitySet([PlateLight(model, data)])
  light = acts.activities[0]
  poll = acts.step_hook(model, data)

  px, py = plate_center(model)
  lid = model.body("garden_light").id
  lx, ly = float(model.body_pos[lid][0]), float(model.body_pos[lid][1])

  yaw = math.pi                       # approach the plate from the east
  data.qpos[0], data.qpos[1], data.qpos[2] = px + APPROACH, py, 0.045
  data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(model, data)

  frames, renderer, viewer = [], None, None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  else:
    renderer = mujoco.Renderer(model, FRAME_H, FRAME_W)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE

  def grab(label, target, distance, az, el):
    """Aim a free camera at a named point. One nice thing #93 bought this
    demo: the light stands 0.8 m from the plate, so cause and effect fit in
    ONE frame where the gate used to be 4.4 m away and the filmstrip had to
    alternate subjects."""
    if renderer is None:
      return
    cam.lookat[:] = [target[0], target[1], target[2]]
    cam.distance, cam.azimuth, cam.elevation = distance, az, el
    renderer.update_scene(data, cam)
    frames.append((label, renderer.render().copy()))

  def scene_view(label):
    # Frames the plate AND the pole together, near-level: a bulb's rgba flip
    # reads best against the fence, not from above (the filmstrip-angle
    # lesson from draw.py -- sweep, do not reason).
    grab(label, ((px + lx) / 2, (py + ly) / 2, 0.45), 2.8, 150, -12)

  L, R = model.actuator("left_motor").id, model.actuator("right_motor").id
  wall0 = time.monotonic()

  def step(seconds, v):
    tl, tr = wheel_targets(v, 0.0)
    for _ in range(round(seconds / model.opt.timestep)):
      data.ctrl[L] = slew(data.ctrl[L], tl, model.opt.timestep)
      data.ctrl[R] = slew(data.ctrl[R], tr, model.opt.timestep)
      mujoco.mj_step(model, data)
      poll()
      if viewer is not None:
        if not viewer.is_running():
          raise KeyboardInterrupt("viewer closed")
        viewer.sync()
        if realtime:
          ahead = data.time - (time.monotonic() - wall0)
          if ahead > 0:
            time.sleep(min(ahead, 0.02))

  step(1.0, 0.0)                                    # settle
  rest = dict(light.flags)
  scene_view("light off, approaching the plate")

  # Drive on, recording the deepest press and whether the flag ever tripped.
  deepest, pressed_ever = 0.0, False
  tl, tr = wheel_targets(DRIVE_V, 0.0)
  t0 = data.time
  while data.time - t0 < 12.0:
    data.ctrl[L] = slew(data.ctrl[L], tl, model.opt.timestep)
    data.ctrl[R] = slew(data.ctrl[R], tr, model.opt.timestep)
    mujoco.mj_step(model, data)
    poll()
    deepest = max(deepest, light.depth(data))
    if light.flags["pressed"] and not pressed_ever:
      pressed_ever = True
      # No settle step needed before the photo: an rgba toggle lands in the
      # model immediately, unlike the mocap pose the gate needed a forward
      # pass for. One less trap, by construction.
      scene_view("on the plate: pressed, light ON")
    if viewer is not None:
      viewer.sync()
    if float(data.qpos[0]) < px - 0.9:
      break
  step(2.0, 0.0)

  scene_view("driven off: released, light STAYS on")

  result = {
    "rest": rest,
    "deepest_mm": deepest * 1000,
    "pressed_ever": pressed_ever,
    "final": dict(light.flags),
    "changes": light.changes,
    "snapshot": acts.snapshot(),
    "bulb_rgba": [float(v) for v in
                  model.geom_rgba[model.geom("garden_light_bulb").id]],
    "sim_time": float(data.time),
  }
  if viewer is not None:
    viewer.close()
  if renderer is not None:
    renderer.close()
  return result, frames


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--view", action="store_true",
                  help="watch it live in the MuJoCo viewer; skips the filmstrip")
  ap.add_argument("--fast", action="store_true",
                  help="with --view: run flat out instead of real time")
  args = ap.parse_args()
  try:
    r, frames = run(args.view, not args.fast)
  except KeyboardInterrupt:
    print("aborted (viewer closed)")
    return

  from pluggybot.activity.plate import LAMP_LIT
  latched = (r["final"]["state"] == "on" and not r["final"]["pressed"])
  # Within float32: MuJoCo stores rgba as f32, so 0.85 comes back 0.8500000238
  # and an == against the Python literal calls a lit bulb dark.
  lit = all(abs(a - b) < 1e-5 for a, b in zip(r["bulb_rgba"], LAMP_LIT))
  ok = r["pressed_ever"] and latched and lit
  print(f"\nat rest                 {r['rest']}")
  print(f"deepest press           {r['deepest_mm']:.1f} mm "
        f"(trigger 6.0, travel 10.0)")
  print(f"pressed while driven on {r['pressed_ever']}")
  print(f"after driving off       {r['final']}")
  print(f"bulb rgba               {[round(v, 2) for v in r['bulb_rgba']]} "
        f"(rgba toggle: dim unlit vs warm lit)")
  print(f"flag changes            {r['changes']}")
  print(f"telemetry snapshot      {r['snapshot']}")
  print(f"sim time                {r['sim_time']:.1f} s")
  print(f"\nACTIVITY: {'OK' if ok else 'FAILED'}"
        f"  (pressed, latched, and the bulb actually changed)")

  if not frames:
    return
  cols = min(len(frames), 3)
  rows = (len(frames) + cols - 1) // cols
  panel_h = 210
  sheet = Image.new("RGB", (FRAME_W * cols, FRAME_H * rows + panel_h), BG)
  for i, (label, img) in enumerate(frames):
    sheet.paste(Image.fromarray(img),
                ((i % cols) * FRAME_W, (i // cols) * FRAME_H))
  dr = ImageDraw.Draw(sheet)
  for i, (label, _) in enumerate(frames):
    x, y = (i % cols) * FRAME_W, (i // cols) * FRAME_H
    dr.rectangle([x, y, x + FRAME_W - 1, y + FRAME_H - 1], outline=PANEL)
    dr.text((x + 12, y + 10), label, fill=INK)
  ty = FRAME_H * rows + 16
  for line, colour in (
      (f"ACTIVITY: {'OK' if ok else 'FAILED'}", GREEN if ok else RED),
      ("", INK),
      (f"deepest press        {r['deepest_mm']:5.1f} mm   "
       f"(trigger 6.0 / release 3.0 / travel 10.0)", INK),
      (f"pressed (live)       {r['final']['pressed']}   "
       f"-- false again once the robot drives off", INK),
      (f"state (latched)      {r['final']['state']}   "
       f"-- stays on; a world fact with no restoring force", INK),
      (f"bulb rgba            {[round(v, 2) for v in r['bulb_rgba']]}   "
       f"geom_rgba toggle, not light simulation", INK),
      (f"telemetry            {r['snapshot']}", INK),
      ("", INK),
      ("MuJoCo owns the sprung joint and the wheel on it. Python owns", DIM),
      ("'pressed' (threshold + hysteresis) and 'on' (a latch). The", DIM),
      ("browser owns the glow, keyed to these flags, never simulated.", DIM),
      ("The bulb is on a STATIC body: it ships once in the scene and", DIM),
      ("never again, so the flag is the only record of it on the wire.", DIM),
  ):
    dr.text((20, ty), line, fill=colour)
    ty += 15
  sheet.save(OUT)
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
