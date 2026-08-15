"""Seed dispenser demo: the robot sows a row.

The FIFTH tool, and the first one built against `docs/ToolPattern.md` rather
than mined for it. The robot fetches the dispenser module from the rack's
fifth bay, drives to each point of a row in turn, lowers the outlet to 50 mm
and meters out exactly one seed at each.

What the tool brings is a DISCRETE RELEASE -- the one thing the robot could
not do before, since it can carry and it can grip but it cannot let go of
exactly one of something. What it borrows is the lift (drop height) and the
base (placement). There is no new axis and no new interface, which is why it
cost a fraction of what the pen and claw did.

The escapement meters by GEOMETRY, not by timing: a pocket carries one seed
past the end of a shelf while a blanking slab arrives under the magazine
mouth. "Hold the gate open for 200 ms" would dispense a different number of
seeds on a differently loaded run.

Two numbers worth watching in the output. Placement error is CENTIMETRE
class and that is the design target, not a shortfall -- a sown seed lands
where it lands. And it was 220 mm until the seeds got rolling friction:
a condim=3 sphere rolls forever, and no amount of sliding friction touches
it (measured: identical travel to the millimetre at mu 0.7 and 1.0).

Writes dispense.png: a filmstrip plus where each seed actually ended up,
judged on contact with the ground rather than on the gate command.

Usage:
  uv run python scripts/dispense.py --view          # watch it live
  MUJOCO_GL=egl uv run python scripts/dispense.py   # headless + dispense.png
  MUJOCO_GL=egl uv run python scripts/dispense.py --record dispense.mp4
"""

import argparse
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import (
  HUB_STATION_YS, SEED_COUNT, module_power_contact,
)
from pluggybot.hub.dispenser import (
  APPROACH_LIFT, SEED_MODULE, SOW_OUTLET_Z, SeedDispenser,
)
from pluggybot.hub.swap import HubSwap
from pluggybot.viz import Recorder

FRAME_W, FRAME_H = 400, 300
OUT = "dispense.png"
# A row across open floor in front of the rack, on the fork line so the tool
# is not sowing off to one side of everything the robot can see.
ROW = [(0.95, 0.55), (0.95, 0.75), (0.95, 0.95)]
CAM_AZ, CAM_EL, CAM_LOOK_DZ = 120, -22, 0.09
CAM_NEAR, CAM_WIDE = 0.55, 1.15
BG, PANEL, INK, DIM = (24, 26, 30), (38, 41, 47), (232, 234, 238), (120, 126, 136)
GREEN, RED = (61, 220, 132), (232, 92, 84)


def run(view: bool, realtime: bool, record: str | None = None,
        record_fps: int = 30, record_speed: float = 1.0):
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  frames, renderer, cam, viewer = [], None, None, None
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
    # A FREE camera aimed at the OUTLET, not a tracking camera on the module
    # body -- and the angle comes from a sweep at the moment of release, not
    # from reasoning about the geometry (the rule the pen filmstrip's grey
    # rectangle bought). Tracking the module body put the tube behind the
    # chassis and the seed, a 16 mm sphere, was invisible in all eight
    # azimuths tried. Aiming 90 mm above the outlet at az=120 puts the
    # magazine, the ground under it and the sown seed all in frame.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = CAM_AZ, CAM_EL

  recorder = None
  if record:
    recorder = Recorder(model, record, track_body=SEED_MODULE, fps=record_fps,
                        speed=record_speed, distance=0.80,
                        azimuth=CAM_AZ, elevation=CAM_EL)
    hooks.append(lambda: recorder.maybe_grab(data))

  if hooks:
    swap.on_step = lambda: [h() for h in hooks]

  def grab(label, distance: float = CAM_NEAR):
    if renderer is not None:
      o = data.site_xpos[model.site("seed_outlet").id]
      cam.lookat[:] = [float(o[0]), float(o[1]), float(o[2]) + CAM_LOOK_DZ]
      cam.distance = distance
      renderer.update_scene(data, cam)
      frames.append((label, renderer.render().copy()))

  def stage(text):
    if recorder is not None:
      recorder.set_label(text)

  stage("approaching bay E")
  swap.place_at_standoff(HUB_STATION_YS[4])
  grab("dispenser in bay E", CAM_WIDE)

  stage("picking up the dispenser")
  swap.pick()
  disp = SeedDispenser(model, data, swap)
  offset = disp.calibrate()
  powered = module_power_contact(model, data, SEED_MODULE)
  loaded = disp.remaining()
  grab(f"picked, {loaded} seeds loaded", CAM_WIDE)

  # Sow ALONG the row, not across it. With the robot square to the row every
  # hop is a pure sideways translation -- the one motion a differential drive
  # cannot make -- and costs a 90 deg pivot out and another back, per seed.
  # A real seed drill drives down the row, and so does this.
  row_heading = SeedDispenser.row_heading(ROW)

  sown = []
  for i, xy in enumerate(ROW):
    stage(f"driving to seed point {i + 1}")
    residual = disp.drive_over(xy, row_heading)
    stage("lowering the outlet")
    disp.lower_outlet_to(SOW_OUTLET_Z)
    if i == 0:
      grab("outlet at sow height")
    stage("metering one seed")
    r = disp.dispense()
    placed = None
    if r["dropped"]:
      p = data.xpos[disp.seed_bids[r["dropped"][0]]]
      placed = (float(p[0]), float(p[1]))
    sown.append({
      "target": xy, "residual_mm": residual * 1000, "count": r["count"],
      "seed": r["dropped"][0] if r["dropped"] else None, "placed": placed,
      "error_mm": (float(np.hypot(placed[0] - xy[0], placed[1] - xy[1])) * 1000
                   if placed else None),
    })
    disp.set_lift(APPROACH_LIFT, settle=1.0)
    grab(f"seed {i + 1} sown")

  # Let the row settle before judging where anything is: a seed that has
  # stopped is a planting, a seed still moving is a claim.
  swap._run(2.0, 0.0)
  landed = sum(1 for k in range(SEED_COUNT) if disp.landed(k))

  result = {
    "offset_fwd_mm": offset[0] * 1000, "offset_lat_mm": offset[1] * 1000,
    "powered": powered, "loaded": loaded, "sown": sown, "landed": landed,
    "remaining": disp.remaining(),
    "metered_one_each": all(s["count"] == 1 for s in sown),
    "seated_after": module_power_contact(model, data, SEED_MODULE),
    "sim_time": float(data.time),
  }
  # AFTER the result dict, so the hold cannot shift the numbers it reports.
  if recorder is not None:
    stage("row sown")
    swap._run(2.0, 0.0)

  if viewer is not None:
    viewer.close()
  if renderer is not None:
    renderer.close()
  if recorder is not None:
    print(f"wrote {recorder.close()} "
          f"({recorder.frames_written} frames, {record_fps} fps)")
  return result, frames


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--view", action="store_true",
                  help="watch it live in the MuJoCo viewer; skips the filmstrip")
  ap.add_argument("--fast", action="store_true",
                  help="with --view: run flat out instead of real time")
  ap.add_argument("--record", metavar="PATH",
                  help="record the tracking camera to PATH (.mp4 or .gif); "
                       "720p, independent of --view")
  ap.add_argument("--record-fps", type=int, default=30)
  ap.add_argument("--record-speed", type=float, default=1.0,
                  help="sim seconds per played second (default 1.0)")
  args = ap.parse_args()
  try:
    r, frames = run(args.view, not args.fast,
                    args.record, args.record_fps, args.record_speed)
  except KeyboardInterrupt:
    print("aborted (viewer closed)")
    return

  errs = [s["error_mm"] for s in r["sown"] if s["error_mm"] is not None]
  sow_ok = (r["metered_one_each"] and r["landed"] == len(ROW)
            and r["seated_after"])
  print(f"\noutlet offset (measured) {r['offset_fwd_mm']:.1f} mm fwd, "
        f"{r['offset_lat_mm']:.1f} mm lat")
  print(f"tool powered            {r['powered']}")
  print(f"magazine loaded         {r['loaded']} seeds")
  for i, s in enumerate(r["sown"]):
    print(f"  point {i + 1} {s['target']}: metered {s['count']}, "
          f"drive residual {s['residual_mm']:.1f} mm, "
          f"placement error {s['error_mm']:.1f} mm")
  print(f"metered exactly one     {r['metered_one_each']}")
  print(f"seeds on the ground     {r['landed']} of {len(ROW)}")
  print(f"module still seated     {r['seated_after']}")
  print(f"sim time                {r['sim_time']:.1f} s")
  print(f"\nSOW: {'OK' if sow_ok else 'FAILED'}")

  if not frames:
    return
  cols = min(len(frames), 3)
  rows = (len(frames) + cols - 1) // cols
  panel_h = 235
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
  lines = [
    (f"SOW: {'OK' if sow_ok else 'FAILED'}", GREEN if sow_ok else RED),
    ("", INK),
    (f"outlet offset (measured) {r['offset_fwd_mm']:6.1f} mm fwd  "
     f"{r['offset_lat_mm']:+6.1f} mm lat", INK),
    (f"metered exactly one seed per cycle   {r['metered_one_each']}", INK),
    (f"seeds on the ground                  {r['landed']} of {len(ROW)}", INK),
    (f"placement error   min {min(errs):.0f} / mean {sum(errs) / len(errs):.0f} "
     f"/ max {max(errs):.0f} mm" if errs else "placement error   n/a", INK),
    (f"module seated after                  {r['seated_after']}", INK),
    ("", INK),
    ("the escapement meters by GEOMETRY, not by timing: one pocket-width", DIM),
    ("of travel carries exactly one seed past the end of its shelf.", DIM),
    ("centimetre placement is the design target -- but it was 220 mm until", DIM),
    ("the seeds got ROLLING friction (condim=6). Sliding friction, at any", DIM),
    ("value, changed the roll by 0.0 mm: a rolling ball is not sliding.", DIM),
  ]
  for line, colour in lines:
    dr.text((20, ty), line, fill=colour)
    ty += 15
  sheet.save(OUT)
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
