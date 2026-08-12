"""Claw module demo (milestone 8): the robot picks something off the floor.

The fourth tool. The robot fetches the CLAW module from the rack's fourth bay,
drives over a block on the floor, grips it, carries it, and sets it down.

The arm drops from the coupling and angles 55 mm FORWARD, which buys the
tool's own camera a clear sightline to its grip point without spending the
coupling's moment budget: the gravity latch takes ~0.45 N*m, and 60 g at 55 mm
is 0.03 N*m of it. Reach is only expensive when it is long (400 g at 150 mm
was 0.59 N*m and unseated the module). The chassis -- the obvious worry --
was never close to the limit.

`claw_eye` is the first camera mounted on a TOOL rather than on the chassis.
It costs no CSI port on the Pi, because module data already crosses the
coupling wirelessly, and only the claw pays for it.

Writes pickup.png: a filmstrip plus what actually happened, judged on contact
rather than on command. Grasping runs with MuJoCo's `noslip` solver pass --
without it a gripped object creeps out of the jaws at ~8 mm/s no matter how
hard they squeeze (see hub/gripper.py). Note the robot is BLIND to the floor inside 0.48 m
(nav camera 180 mm up, 41 deg fovy) while the grip point is 244 mm ahead of
the axle, so the grasp is necessarily open-loop from a memorised pose --
finding the object is a separate, unsolved problem.

Usage:
  uv run python scripts/pickup.py --view          # watch it live
  MUJOCO_GL=egl uv run python scripts/pickup.py   # headless + pickup.png
"""

import argparse
import math
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.hub.gripper import CLAW_MODULE, ClawTool
from pluggybot.hub.swap import HubSwap

FRAME_W, FRAME_H = 400, 300
OUT = "pickup.png"
CARRY_TO = (0.95, -0.35)
BG, PANEL, INK, DIM = (24, 26, 30), (38, 41, 47), (232, 234, 238), (120, 126, 136)
GREEN, RED = (61, 220, 132), (232, 92, 84)


def run(view: bool, realtime: bool):
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  frames, renderer, cam, viewer = [], None, None, None

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

    swap.on_step = hook
  else:
    renderer = mujoco.Renderer(model, FRAME_H, FRAME_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pickup").id
    cam.distance, cam.azimuth, cam.elevation = 0.95, 135, -18

  eye = mujoco.Renderer(model, FRAME_H, FRAME_W) if not view else None

  def grab(label):
    if renderer is not None:
      renderer.update_scene(data, cam)
      frames.append((label, renderer.render().copy()))

  def grab_eye(label):
    """What the CLAW'S OWN camera sees -- the first camera on a tool rather
    than on the chassis. This is the view the nav camera cannot get: it goes
    blind to the floor inside 0.48 m and the grip point is ~285 mm ahead."""
    if eye is not None:
      eye.update_scene(data, camera="claw_eye")
      frames.append((label, eye.render().copy()))

  obj_bid = model.body("pickup").id

  def obj_z():
    return float(data.xpos[obj_bid][2])

  swap.place_at_standoff(HUB_STATION_YS[3])
  grab("claw in bay D")
  swap.pick()
  claw = ClawTool(model, data, swap)
  offset = claw.calibrate()
  powered = module_power_contact(model, data, CLAW_MODULE)
  grab("claw picked")

  obj0 = np.array(data.xpos[obj_bid], dtype=float)
  arrived = claw.drive_over((obj0[0], obj0[1]), 0.0)
  g = claw.grip_world()
  aim = (float(g[1] - obj0[1]), float(g[0] - obj0[0]))
  grab("over the block")

  grab_eye("claw eye: over it")
  picked = claw.pick_up()
  lifted_z = obj_z()
  grab_eye("claw eye: gripped")
  grab("gripped and lifted")

  # Carry it somewhere -- including a turn, which is where a 172 mm pendant
  # swings and where the fork's axial hold was first found wanting.
  from pluggybot.behavior.navigation import drive_toward
  from pluggybot.control import wheel_targets
  t0 = data.time
  held_steps = dropped = 0
  while data.time - t0 < 40.0:
    pose = (swap.reckoner.x, swap.reckoner.y, swap.reckoner.theta)
    if math.hypot(CARRY_TO[0] - pose[0], CARRY_TO[1] - pose[1]) < 0.05:
      break
    v, w = drive_toward(pose, CARRY_TO)
    tl, tr = wheel_targets(v, w)
    swap._step_once(tl, tr)
    held_steps += 1
    if obj_z() < 0.03:
      dropped += 1
  swap._run(1.5, 0.0)
  carried = claw.holding() and obj_z() > 0.03
  grab("carried")

  placed = claw.set_down()
  swap._run(1.0, 0.0)
  grab("set down")

  result = {
    "offset_fwd_mm": offset[0] * 1000, "offset_lat_mm": offset[1] * 1000,
    "powered": powered, "arrived": arrived,
    "aim_lat_mm": aim[0] * 1000, "aim_fwd_mm": aim[1] * 1000,
    "lifted_mm": (lifted_z - obj0[2]) * 1000,
    "carried": carried, "drop_steps": dropped,
    "released": placed["released"], "final_z_mm": obj_z() * 1000,
    "seated_after": module_power_contact(model, data, CLAW_MODULE),
    "sim_time": float(data.time), **picked,
  }
  if viewer is not None:
    viewer.close()
  if renderer is not None:
    renderer.close()
  if eye is not None:
    eye.close()
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

  grasp_ok = r["holding"] and r["lifted_mm"] > 40 and r["seated_after"]
  print(f"\ngrip offset (measured) {r['offset_fwd_mm']:.1f} mm fwd, "
        f"{r['offset_lat_mm']:.1f} mm lat")
  print(f"aim over the block     {r['aim_lat_mm']:+.1f} mm lat, "
        f"{r['aim_fwd_mm']:+.1f} mm fwd")
  print(f"tool powered           {r['powered']}")
  print(f"gripped                {r['holding']}   (both pads in contact)")
  print(f"lifted                 {r['lifted_mm']:.1f} mm off the floor")
  print(f"carried through a turn {r['carried']}   "
        f"({r['drop_steps']} steps below carry height)")
  print(f"released               {r['released']}, "
        f"block back at {r['final_z_mm']:.1f} mm")
  print(f"module still seated    {r['seated_after']}")
  print(f"sim time               {r['sim_time']:.1f} s")
  print(f"\nGRASP + LIFT: {'OK' if grasp_ok else 'FAILED'}")
  print(f"CARRY:        {'OK' if r['carried'] else 'FAILED'}")

  if not frames:
    return
  cols = min(len(frames), 3)
  rows = (len(frames) + cols - 1) // cols
  panel_h = 190
  sheet = Image.new("RGB", (FRAME_W * cols, FRAME_H * rows + panel_h), BG)
  for i, (label, img) in enumerate(frames):
    sheet.paste(Image.fromarray(img),
                ((i % cols) * FRAME_W, (i // cols) * FRAME_H))
  d = ImageDraw.Draw(sheet)
  for i, (label, _) in enumerate(frames):
    x, y = (i % cols) * FRAME_W, (i // cols) * FRAME_H
    d.rectangle([x, y, x + FRAME_W - 1, y + FRAME_H - 1], outline=PANEL)
    d.text((x + 12, y + 10), label, fill=INK)
  ty = FRAME_H * rows + 16
  for line, colour in (
      (f"GRASP + LIFT: {'OK' if grasp_ok else 'FAILED'}",
       GREEN if grasp_ok else RED),
      (f"CARRY: {'OK' if r['carried'] else 'FAILED'}",
       GREEN if r["carried"] else RED),
      ("", INK),
      (f"grip offset (measured)  {r['offset_fwd_mm']:6.1f} mm fwd  "
       f"{r['offset_lat_mm']:+6.1f} mm lat", INK),
      (f"aim over the block      {r['aim_lat_mm']:+6.1f} mm lat  "
       f"{r['aim_fwd_mm']:+6.1f} mm fwd", INK),
      (f"lifted                  {r['lifted_mm']:6.1f} mm", INK),
      (f"carried through a turn  {r['carried']}   "
       f"({r['drop_steps']} steps low)", INK),
      (f"module seated after     {r['seated_after']}", INK),
      ("", INK),
      ("arm angles forward 55 mm: enough for the tool's OWN camera to see", DIM),
      ("past it, and only 0.03 N.m of the coupling's 0.45 N.m budget.", DIM),
      ("nav camera is blind to the floor inside 0.48 m; claw_eye is not.", DIM),
  ):
    d.text((20, ty), line, fill=colour)
    ty += 15
  sheet.save(OUT)
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
