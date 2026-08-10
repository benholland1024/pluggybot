"""Module electrical interface demo (milestone 8): watch a tool wake up.

The coupling's power contacts are the PEG itself -- split into two conductors
around an insulated centre, so the fork's left and right V-notch pairs are the
two poles. That choice was made by measurement, not taste: the peg already
sits in those four plates carrying 0.43-0.49 N each of gravity preload, which
is 4-9x anything a lean-pad could supply (see docs/SimNotes.md).

This runs the real errand in room_hub -- navigate to the rack, pick the LCD
module, carry it across the room, bring it back, hang it up -- and records the
electrical criterion on EVERY physics step. The module's body is painted live
cyan while it is conducting and dead blue when it is not, so the filmstrip
shows the tool waking up and going to sleep. Underneath, a continuity timeline
plots each pole separately for the whole run, which is the part that would
expose a brown-out mid-haul.

Usage:
  uv run python scripts/module_power.py --view          # watch it live
  MUJOCO_GL=egl uv run python scripts/module_power.py   # headless + filmstrip
  MUJOCO_GL=egl uv run python scripts/module_power.py --bare   # faster
  uv run python scripts/module_power.py --view --fast   # viewer, no pacing

--view paints the module in the live viewer instead of rendering a filmstrip:
watching it IS the filmstrip, and the offscreen renderer would otherwise be
competing with the viewer for a GL context. The continuity timeline is still
written either way -- it costs nothing and it is the part you cannot see by
looking.
"""

import argparse
import math
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot.hub.coupling import HUB_STATION_YS, module_power_state
from pluggybot.hub.mission import HubMission, MissionAborted
from pluggybot.hub.swap import HubSwap
from pluggybot.power import MODULE_IDLE_W

MODULE = "module_lcd"
LIVE = (0.15, 0.95, 0.85, 1.0)      # module body while conducting
DEAD = (0.20, 0.45, 0.75, 1.0)      # ...and while it is just cargo
FRAME_W, FRAME_H = 420, 300
FRAME_PERIOD = 1.0                  # s of sim time between captured frames
N_TILES = 6
OUT = "module_power.png"

GREEN = (61, 220, 132)
RED = (232, 92, 84)
INK = (232, 234, 238)
DIM = (120, 126, 136)
BG = (24, 26, 30)
PANEL = (38, 41, 47)


class PowerRecorder:
  """Logs the electrical criterion every step and grabs frames on a cadence."""

  def __init__(self, model, data, render: bool = True) -> None:
    self.model, self.data = model, data
    self.body_gid = model.geom(f"{MODULE}_body").id
    self.renderer = mujoco.Renderer(model, FRAME_H, FRAME_W) if render else None
    self.cam = None
    if render:
      # Track the MODULE, not the robot: it is the thing whose colour is the
      # measurement, and it stays framed whether it is on the rack or on the
      # fork (which also keeps the approach in shot).
      self.cam = mujoco.MjvCamera()
      self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      self.cam.trackbodyid = model.body(MODULE).id
      self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 125, -14
    self.log: list[tuple[float, bool, bool]] = []
    self.frames: list[tuple[float, bool, np.ndarray]] = []
    self.marks: list[tuple[float, str]] = []
    self._next = 0.0
    self._painted: bool | None = None

  def mark(self, label: str) -> None:
    self.marks.append((float(self.data.time), label))

  def step(self) -> None:
    st = module_power_state(self.model, self.data, MODULE)
    self.log.append((float(self.data.time), st["left"], st["right"]))
    # Repaint on CHANGE, not every step: geom_rgba is read straight out of
    # the model by both the offscreen renderer and the passive viewer, so
    # this is what makes the tool visibly wake up in either one.
    if st["powered"] != self._painted:
      self._painted = st["powered"]
      self.model.geom_rgba[self.body_gid] = LIVE if st["powered"] else DEAD
    if self.renderer is not None and self.data.time >= self._next:
      self._next = self.data.time + FRAME_PERIOD
      self.renderer.update_scene(self.data, self.cam)
      self.frames.append((float(self.data.time), st["powered"],
                          self.renderer.render().copy()))

  def close(self) -> None:
    if self.renderer is not None:
      self.renderer.close()

  # ---- reporting -----------------------------------------------------------

  @property
  def powered_seconds(self) -> float:
    if len(self.log) < 2:
      return 0.0
    dt = (self.log[-1][0] - self.log[0][0]) / (len(self.log) - 1)
    return sum(dt for _, ls, rs in self.log if ls and rs)

  def pole_dropout_fraction(self) -> tuple[float, float]:
    """While the tool is HELD, how often is each pole open? This is the
    number that matters: a pole flickering during a haul is a tool that
    browns out mid-stroke."""
    held = [(ls, rs) for _, ls, rs in self.log if ls or rs]
    if not held:
      return 0.0, 0.0
    return (sum(1 for ls, _ in held if not ls) / len(held),
            sum(1 for _, rs in held if not rs) / len(held))

  def brownouts(self, t0: float = -1e9, t1: float = 1e9) -> list:
    """Spans where the tool is HELD but a pole is open, within a window.

    Duration is the number a hardware design actually consumes -- it sizes
    the module's holding capacitor -- and a percentage is not: 0.4 % spread
    as hundreds of microsecond blips and 0.4 % as one 200 ms outage are very
    different parts.

    Windowing matters just as much. Mating and releasing NECESSARILY break
    one pole before the other, so a run-wide figure is dominated by events
    that are not faults at all. What a motorised tool cares about is
    brown-outs during the HAUL.
    """
    spans, start, prev = [], None, None
    for t, ls, rs in self.log:
      if not (t0 <= t <= t1):
        continue
      if (ls or rs) and not (ls and rs):
        if start is None:
          start = t
      elif start is not None:
        spans.append((start, (prev if prev is not None else t) - start))
        start = None
      prev = t
    if start is not None and prev is not None:
      spans.append((start, prev - start))
    return spans

  def phase_windows(self) -> list[tuple[str, float, float]]:
    """(label, t0, t1) for each marked phase of the run."""
    out = []
    for i, (t, label) in enumerate(self.marks):
      end = self.marks[i + 1][0] if i + 1 < len(self.marks) else self.log[-1][0]
      out.append((label, t, end))
    return out


def select_tiles(frames, n=N_TILES):
  """Frames that tell the story: always the pair straddling each power
  transition (that IS the event), then evenly spaced filler."""
  want = set()
  for i in range(1, len(frames)):
    if frames[i][1] != frames[i - 1][1]:
      want |= {i - 1, i}
  for k in range(n):
    want.add(round(k * (len(frames) - 1) / max(n - 1, 1)))
  chosen = sorted(want)
  while len(chosen) > n:                    # drop filler, never a transition
    edges = {i for i in range(1, len(frames))
             if frames[i][1] != frames[i - 1][1]}
    edges |= {i - 1 for i in edges}
    droppable = [i for i in chosen if i not in edges]
    chosen.remove(droppable[len(droppable) // 2] if droppable else chosen[-1])
  return [frames[i] for i in chosen]


def draw_timeline(draw, log, marks, x0, y0, w, h):
  """Two pole strips plus the combined verdict, over the whole run."""
  if not log:
    return
  t0, t1 = log[0][0], log[-1][0]
  span = max(t1 - t0, 1e-6)
  rows = (("left pole", lambda e: e[1], 16),
          ("right pole", lambda e: e[2], 16),
          ("POWERED", lambda e: e[1] and e[2], 26))
  y = y0
  for label, pick, rh in rows:
    draw.text((x0, y - 1), label, fill=DIM)
    bx = x0 + 86
    bw = w - 86
    draw.rectangle([bx, y, bx + bw, y + rh], fill=PANEL)
    # One pixel column per x: a dropout of a few ms still shows as a notch.
    per = max(len(log) // bw, 1)
    for px in range(bw):
      chunk = log[px * per:(px + 1) * per]
      if chunk and all(pick(e) for e in chunk):
        draw.line([bx + px, y, bx + px, y + rh], fill=GREEN)
      elif chunk and any(pick(e) for e in chunk):
        draw.line([bx + px, y, bx + px, y + rh], fill=RED)
    y += rh + 8

  # phase markers + time axis
  bx, bw = x0 + 86, w - 86
  for t, label in marks:
    px = bx + int(bw * (t - t0) / span)
    draw.line([px, y0 - 6, px, y - 4], fill=INK)
    draw.text((px + 3, y - 2), label, fill=INK)
  for k in range(6):
    t = t0 + span * k / 5
    px = bx + int(bw * k / 5)
    draw.text((px, y + 14), f"{t:.0f}s", fill=DIM)


def _launch_viewer(model, data):
  # `from mujoco import viewer as ...`, NOT `import mujoco.viewer`: the
  # latter binds `mujoco` as a function-local and shadows the module-level
  # import for the whole function (the same trap hub/mission.py documents).
  from mujoco import viewer as mj_viewer
  return mj_viewer.launch_passive(model, data)


def run_bare(view: bool = False, realtime: bool = True):
  """The coupling on its own, in the bare hub world: pick, haul, stow."""
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  rec = PowerRecorder(model, data, render=not view)
  viewer = _launch_viewer(model, data) if view else None
  if viewer is None:
    swap.on_step = rec.step
  else:
    # HubSwap has no viewer support of its own (HubMission owns that), so
    # pace and sync from the same per-step hook the recorder uses.
    state = {"next": 0.0, "wall": time.time()}

    def hooked():
      rec.step()
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

    swap.on_step = hooked
  swap.place_at_standoff(HUB_STATION_YS[0])
  swap._run(1.0, 0.0)
  rec.mark("pick")
  swap.pick()
  rec.mark("haul")
  # Translation only. Turning is the interesting disturbance, but put_back's
  # default travel assumes the pose pick's retreat left it in, and 360 deg of
  # spinning drifts odometry well past the coupling's +/-11 mm capture window
  # (measured: the stow missed the trays entirely). The room errand does the
  # real turns -- that is what it is for.
  from pluggybot.control import wheel_targets
  for seconds, v, w in ((2.0, -0.25, 0.0), (1.0, 0.0, 0.0), (2.0, 0.25, 0.0),
                        (1.5, 0.0, 0.0)):
    tl, tr = wheel_targets(v, w)
    for _ in range(round(seconds / model.opt.timestep)):
      swap._step_once(tl, tr)
  rec.mark("stow")
  swap.put_back()
  swap._run(1.5, 0.0)
  state = swap.module_state(MODULE)
  if viewer is not None:
    viewer.close()
  return model, data, rec, state


def run_room(view: bool = False, realtime: bool = True):
  """The real errand in room_hub: navigate, pick, carry across, stow."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  viewer = _launch_viewer(model, data) if view else None
  mission = HubMission(model, data, viewer=viewer,
                       realtime=realtime if view else False)
  rec = PowerRecorder(model, data, render=not view)
  mission.step_hooks.append(rec.step)
  try:
    mission.start_at(0.5, 3.0, math.pi / 2)
    mission.start_discovery()
    mission._spin()
    rec.mark("pick")
    mission.swap_at_bay(HUB_STATION_YS[0], "pick")
    rec.mark("carry")
    mission.drive_to(-1.2, 2.5)
    rec.mark("stow")
    mission.swap_at_bay(HUB_STATION_YS[0], "return")
    state = mission.swap.module_state(MODULE)
  finally:
    mission.close()
    if viewer is not None:
      viewer.close()
  return model, data, rec, state


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--bare", action="store_true",
                  help="bare hub_world instead of the full room errand")
  ap.add_argument("--view", action="store_true",
                  help="watch it live in the MuJoCo viewer (the module is "
                       "painted live/dead there); skips the filmstrip")
  ap.add_argument("--fast", action="store_true",
                  help="with --view: run flat out instead of real time")
  args = ap.parse_args()

  run = run_bare if args.bare else run_room
  try:
    model, data, rec, state = run(view=args.view, realtime=not args.fast)
  except (KeyboardInterrupt, MissionAborted):
    print("aborted (viewer closed)")
    return

  left_drop, right_drop = rec.pole_dropout_fraction()
  tool_wh = rec.powered_seconds * MODULE_IDLE_W / 3600.0
  print(f"\nsim time            {float(data.time):.1f} s")
  print(f"tool powered for    {rec.powered_seconds:.1f} s "
        f"({rec.powered_seconds / max(float(data.time), 1e-9):.0%} of the run)")
  print(f"tool energy         {tool_wh * 1000:.1f} mWh at "
        f"{MODULE_IDLE_W} W")
  print(f"pole dropouts       left {left_drop:.2%}, right {right_drop:.2%} "
        f"(of steps while held)")
  print(f"module stowed       {state['hung']}")
  print("\nbrown-outs by phase (held, but a pole open):")
  for label, t0, t1 in rec.phase_windows():
    spans = rec.brownouts(t0, t1)
    worst = max((d for _, d in spans), default=0.0)
    note = ""
    if label in ("pick", "stow"):
      note = "   <- mating/release breaks one pole first, expected"
    print(f"  {label:8s} {t0:5.1f}-{t1:5.1f}s  {len(spans):3d} spans, "
          f"worst {worst * 1000:5.0f} ms{note}")

  tiles = select_tiles(rec.frames) if rec.frames else []
  cols = 3
  rows = (len(tiles) + cols - 1) // cols
  tl_h = 150
  sheet = Image.new("RGB", (FRAME_W * cols, FRAME_H * rows + tl_h), BG)
  for i, (t, powered, img) in enumerate(tiles):
    sheet.paste(Image.fromarray(img),
                ((i % cols) * FRAME_W, (i // cols) * FRAME_H))
  draw = ImageDraw.Draw(sheet)
  for i, (t, powered, _) in enumerate(tiles):
    x, y = (i % cols) * FRAME_W, (i // cols) * FRAME_H
    draw.rectangle([x, y, x + FRAME_W - 1, y + FRAME_H - 1], outline=PANEL)
    draw.rectangle([x + 6, y + 6, x + 128, y + 26],
                   fill=GREEN if powered else PANEL)
    draw.text((x + 12, y + 12), f"{'LIVE' if powered else 'dead'}  t={t:.0f}s",
              fill=(20, 22, 24) if powered else INK)
  draw_timeline(draw, rec.log, rec.marks, 14, FRAME_H * rows + 22,
                FRAME_W * cols - 28, tl_h - 40)
  sheet.save(OUT)
  rec.close()
  print(f"\nwrote {OUT}  ({len(rec.log)} samples, {len(rec.frames)} frames)")


if __name__ == "__main__":
  main()
