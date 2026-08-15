"""Activity demo (issue #8): a pressure plate latches a garden gate open.

The reference consumer of the activity pattern (`docs/ActivityPattern.md`).
The robot drives onto a sprung plate in the garden; a joint sensor crossing a
threshold flips a live `pressed` flag, and that latches an irreversible
`state: open` which drops the gate panel and turns its lamp from red to
green. Then the robot drives off, `pressed` goes false again, and the gate
STAYS open -- which is the whole point of the latch.

What each layer owns, and it is worth watching for it in the output:

  MUJOCO      the plate's sprung slide joint, and the wheel that pushes it.
              That is all. The gate does not swing, the lamp does not glow.
  PYTHON      "is it pressed" (threshold + hysteresis) and "is it open"
              (a latch). Discrete world state, remembered here because the
              physics has no restoring force to remember it with.
  THE BROWSER everything continuous -- in a real gate, the swing and the
              creak -- keyed to the streamed flags, never simulated.

Note what the sim CAN do and the pose stream cannot: the gate panel sits on
a STATIC body, so it ships once in the scene description and never again. It
moves by `geom_pos` toggle, which the robot's own cameras see and the
telemetry poses do not. The activity flag is not a convenience duplicating
the poses -- for a change like this it is the only channel there is.

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
from pluggybot.activity.plate import PlateGate, plate_center
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
  acts = ActivitySet([PlateGate(model, data)])
  gate = acts.activities[0]
  poll = acts.step_hook(model, data)

  px, py = plate_center(model)
  gate_bid = model.body("garden_gate").id
  gx, gy = float(model.body_pos[gate_bid][0]), float(model.body_pos[gate_bid][1])

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
    """Aim a free camera at a named point. Two subjects here -- the plate and
    the gate 4.4 m away -- so the filmstrip alternates between them rather
    than trying to frame both and showing neither."""
    if renderer is None:
      return
    cam.lookat[:] = [target[0], target[1], target[2]]
    cam.distance, cam.azimuth, cam.elevation = distance, az, el
    renderer.update_scene(data, cam)
    frames.append((label, renderer.render().copy()))

  def gate_view(label):
    # az/distance/elevation swept in both states, not reasoned: the first
    # guess framed the fence as an undifferentiated brown wall in which the
    # gate was invisible either way. Near-level and close is what shows a
    # panel present vs a gap.
    grab(label, (gx, gy, 0.45), 3.0, 200, -8)

  def plate_view(label):
    grab(label, (px, py, 0.2), 2.6, 140, -18)

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
  rest = dict(gate.flags)
  gate_view("gate closed, lamp red")
  plate_view("approaching the plate")

  # Drive on, recording the deepest press and whether the flag ever tripped.
  deepest, pressed_ever = 0.0, False
  tl, tr = wheel_targets(DRIVE_V, 0.0)
  t0 = data.time
  while data.time - t0 < 12.0:
    data.ctrl[L] = slew(data.ctrl[L], tl, model.opt.timestep)
    data.ctrl[R] = slew(data.ctrl[R], tr, model.opt.timestep)
    mujoco.mj_step(model, data)
    poll()
    deepest = max(deepest, gate.depth(data))
    if gate.flags["pressed"] and not pressed_ever:
      pressed_ever = True
      plate_view("on the plate: pressed")
      # One more step before looking at the gate. `sense()` selects the
      # toggle, but a mocap pose only reaches `geom_xpos` on the NEXT
      # forward pass -- grabbing immediately photographed the closed gate
      # under a caption saying it was open.
      mujoco.mj_step(model, data)
      poll()
      gate_view("gate open, lamp green")
    if viewer is not None:
      viewer.sync()
    if float(data.qpos[0]) < px - 0.9:
      break
  step(2.0, 0.0)

  plate_view("driven off: plate released")
  gate_view("gate stays open (latched)")

  result = {
    "rest": rest,
    "deepest_mm": deepest * 1000,
    "pressed_ever": pressed_ever,
    "final": dict(gate.flags),
    "changes": gate.changes,
    "snapshot": acts.snapshot(),
    # Read the WORLD pose, not the model field: the gate is a mocap body and
    # its pose lives in data. Reading model.geom_pos here reported +0.00 and
    # a cheerful FAILED -- a verdict checking the wrong side of the very
    # trap this activity exists to document.
    "gate_geom_z": float(data.geom_xpos[model.geom("garden_gate_panel").id][2]),
    "lamp_rgba": [float(v) for v in
                  model.geom_rgba[model.geom("garden_lamp_bulb").id]],
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

  latched = (r["final"]["state"] == "open" and not r["final"]["pressed"])
  ok = r["pressed_ever"] and latched and r["gate_geom_z"] < -0.5
  print(f"\nat rest                 {r['rest']}")
  print(f"deepest press           {r['deepest_mm']:.1f} mm "
        f"(trigger 6.0, travel 10.0)")
  print(f"pressed while driven on {r['pressed_ever']}")
  print(f"after driving off       {r['final']}")
  print(f"gate panel world z      {r['gate_geom_z']:+.2f} m "
        f"(mocap toggle; 0.45 closed, -0.75 open)")
  print(f"lamp rgba               {[round(v, 2) for v in r['lamp_rgba']]}")
  print(f"flag changes            {r['changes']}")
  print(f"telemetry snapshot      {r['snapshot']}")
  print(f"sim time                {r['sim_time']:.1f} s")
  print(f"\nACTIVITY: {'OK' if ok else 'FAILED'}"
        f"  (pressed, latched, and the gate actually moved)")

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
       f"-- stays open; a world fact with no restoring force", INK),
      (f"gate panel world z  {r['gate_geom_z']:+.2f} m   "
       f"mocap toggle (0.45 closed / -0.75 open), not physics", INK),
      (f"telemetry            {r['snapshot']}", INK),
      ("", INK),
      ("MuJoCo owns the sprung joint and the wheel on it. Python owns", DIM),
      ("'pressed' (threshold + hysteresis) and 'open' (a latch). The", DIM),
      ("browser owns the swing and the creak, keyed to these flags.", DIM),
      ("The gate is a STATIC body: it ships once in the scene and never", DIM),
      ("again, so the flag is the only record of it moving on the wire.", DIM),
  ):
    dr.text((20, ty), line, fill=colour)
    ty += 15
  sheet.save(OUT)
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
