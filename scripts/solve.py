"""Ladder A of issue #264: the hand-written solution to each challenge,
run the way the robot's own attempt runs and graded by the feature's own
grader -- so a passing verdict here is the verdict a robot would be paid
for, and a robot that never earns it is a finding about the robot.

  --feature tower   `solutions.TOWER`: fetch the claw, drive to the
                    workshop, pick and place twice (`pick`/`place`),
                    bring the claw home, stow; `done`; the grade on the
                    lifecycle seam with its 10 s hold
  --feature bench   `solutions.WEIGH`: the claw to the lab, a tare, the
                    unknown cube lifted and `read("lift.force")` averaged,
                    set down, home; the finding recorded off the
                    procedure's `mass` as a mind would; `done`; graded
                    against the hidden mass
  --feature mouse   `lifecycle.cage_errand("home", "feed")`: the care
                    program to the lab and onto the feed plate, judged off
                    the cage's own count (a `care` event, `landed`)

Each starts at the rack in the living room. `challenge/solutions.py`
holds the procedures and the numbers they measured. Writes solve.png: a
filmstrip from a camera on the work.

Usage:
  MUJOCO_GL=egl uv run python scripts/solve.py --feature tower
  uv run python scripts/solve.py --feature bench --view
  MUJOCO_GL=egl uv run python scripts/solve.py --feature tower --at-the-row
      # skip the drive: start in the workshop with the claw on the fork
"""

import argparse
import math
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pluggybot import tick
from pluggybot.challenge import solutions, stack
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import HubLifecycle, errand_from, world_config, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.mission.mission import MissionAborted, bay_standoff
from pluggybot.procedure import library as lib
from pluggybot.rack.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.rack.swap import ARM_EXT, align_lift
from pluggybot.robot import world_spec

OUT = "solve.png"
FRAME_W, FRAME_H = 360, 270
FRAME_EVERY_S = 20.0
BG, INK = (24, 26, 30), (232, 234, 238)


class _Mind:
  """What the lifecycle reads off an overseer on the tower's path: a
  library (the mark of the autonomous arm, where the tower is offered) and
  nothing that decides. The procedure is the script's, not a model's."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  workshop = None
  library = object()
  decisions: list = []
  menu = ov.Menu.for_world("home")


def build_life(view: bool, state_dir: str):
  cfg = world_config("home")
  spec = world_spec(cfg["model"])
  model = spec.compile()
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=view, world="home",
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"], spec=spec,
                      errand=False, battery_wh=cfg["hosting_battery_wh"],
                      ledger=Ledger(path=str(Path(state_dir) / "ledger.json")),
                      tasks=TaskBoard(path=str(Path(state_dir) / "tasks.json")),
                      autonomous=True, overseer=_Mind())
  return life, viewer


def claw_in_hand_at_the_row(life, stand=(-10.2, -4.75, math.pi)) -> None:
  """Pick the claw at its bay, then carry robot AND module to a stand in
  the workshop by one rigid transform (docs/Testing.md, lever 4)."""
  m, swap, model, data = life.mission, life.mission.swap, life.model, life.data
  sx, sy, hd = bay_standoff(HUB_STATION_YS[3], m.rack)
  m.start_at(sx, sy, hd)
  lift0 = align_lift()
  data.qpos[swap.lift_qadr] = lift0
  data.ctrl[swap.lift_act] = lift0
  data.ctrl[swap.arm_act] = ARM_EXT
  data.qpos[swap.arm_qadr] = ARM_EXT
  mujoco.mj_forward(model, data)
  swap._run(1.0, 0.0)
  swap.pick()
  assert module_power_contact(model, data, "module_claw", swap.handle.prefix)
  life.module = "module_claw"
  x, y, yaw = stand
  q = swap.root_qadr
  yaw0 = 2 * math.atan2(float(data.qpos[q + 6]), float(data.qpos[q + 3]))
  ax0 = float(data.qpos[q]) - 0.08 * math.cos(yaw0)
  ay0 = float(data.qpos[q + 1]) - 0.08 * math.sin(yaw0)
  dth = yaw - yaw0
  c, s = math.cos(dth), math.sin(dth)
  dq = np.array([math.cos(dth / 2), 0.0, 0.0, math.sin(dth / 2)])

  def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

  for body in (swap.handle.el("pluggybot"), "module_claw"):
    jid = model.body_jntadr[model.body(body).id]
    adr, dof = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    rx, ry = float(data.qpos[adr]) - ax0, float(data.qpos[adr + 1]) - ay0
    data.qpos[adr], data.qpos[adr + 1] = x + c * rx - s * ry, y + s * rx + c * ry
    data.qpos[adr + 3:adr + 7] = qmul(dq, data.qpos[adr + 3:adr + 7])
    data.qvel[dof:dof + 6] = 0.0
  mujoco.mj_forward(model, data)
  r = swap.reckoner
  r.x, r.y, r.theta = x, y, yaw
  r.update(float(data.qpos[swap.left_adr]), float(data.qpos[swap.right_adr]))
  swap._run(1.0, 0.0)
  m.start_discovery()
  m._spin()


def _camera(life, frames: list, track: str):
  renderer = mujoco.Renderer(life.model, FRAME_H, FRAME_W)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = life.model.body(track).id
  cam.distance, cam.azimuth, cam.elevation = 0.7, 120, -25
  state = {"next": 0.0}
  data = life.data

  def grab(label: str | None = None, force: bool = False):
    if data.time < state["next"] and not force:
      return
    state["next"] = data.time + FRAME_EVERY_S
    renderer.update_scene(data, cam)
    frames.append((label or f"t={data.time:.0f}s {life.state}", renderer.render().copy()))
  life.mission.step_hooks.append(grab)
  return grab


def run(life, feature: str, source: str | None = None, frames: list | None = None) -> dict:
  """A feature's path exactly as a robot's: the offer claimed, the
  procedure defined and run as an errand (or the act's errand run),
  `done`, the grade on the seam."""
  from pluggybot import lifecycle as lc
  m, data = life.mission, life.data
  track = {"tower": stack.BLOCKS[0], "bench": "mass_unknown", "mouse": "lab_mouse"}[feature]
  grab = _camera(life, frames, track) if frames is not None else None
  events: list = []
  life.on_event.append(events.append)
  if feature == "mouse":
    result = life.run_errand(lc.cage_errand("home", "feed", from_xy=m.pose_xy()))
    care = [e for e in events if e["type"] == "care"]
    if grab:
      grab("the act", force=True)
    return {"errand": result, "grade": None, "care": care[-1] if care else None}
  if feature == "tower":
    task = life.tasks.offer("stack_tower", "workshop", t=float(data.time))
  else:
    task = life.tasks.offer("find_mass", "lab", ttl=3000.0, t=float(data.time),
                            params={"known_g": 100, "known_tag": 23, "unknown_tag": 24})
  assert task is not None and life._claim_task(task.id)
  library = lib.Library(world_facts("home"))
  name = "tower" if feature == "tower" else "weigh"
  library.define(name, source)
  errand = errand_from(ov.Decision(action=f"procedure:{name}"), "home", library=library)
  result = life.run_errand(errand)
  if feature == "bench":
    # the finding, as a mind writes it off the locals History shows it
    mass = (result["procedure"].get("locals") or {}).get("mass")
    if mass is not None:
      life._reconsider(ov.Decision(action="idle", record={
        "quantity": "unknown mass", "value": round(float(mass), 3), "unit": "kg",
        "method": "the lift, tared", "topic": "mass_bench"}))
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  tick.run(m.swap, life._grade_routine())
  if grab:
    grab(f"graded at t={data.time:.0f}s", force=True)
  return {"errand": result, "grade": life.grades[-1] if life.grades else None}


def filmstrip(frames, path: str) -> None:
  if not frames:
    return
  cols = min(len(frames), 6)
  rows = (len(frames) + cols - 1) // cols
  sheet = Image.new("RGB", (cols * FRAME_W, rows * FRAME_H), BG)
  for i, (label, rgb) in enumerate(frames):
    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).text((6, 6), label, fill=INK)
    sheet.paste(im, ((i % cols) * FRAME_W, (i // cols) * FRAME_H))
  sheet.save(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--feature", choices=("tower", "bench", "mouse"), default="tower")
  parser.add_argument("--view", action="store_true", help="open the viewer")
  parser.add_argument("--at-the-row", action="store_true",
                      help="tower only: start in the workshop with the claw on the fork")
  parser.add_argument("--out", default=OUT)
  args = parser.parse_args()

  frames: list = []
  with tempfile.TemporaryDirectory() as state_dir:
    life, viewer = build_life(args.view, state_dir)
    if args.feature == "mouse":
      from pluggybot import lifecycle as lc
      acts = lc.home_activities(life.model, life.data)
      life.mission.step_hooks.append(acts.step_hook(life.model, life.data))
      life.activities = acts
    m = life.mission
    t0 = time.time()
    try:
      if args.at_the_row and args.feature == "tower":
        claw_in_hand_at_the_row(life)
        out = run(life, "tower", solutions.TOWER_AT_THE_ROW, frames)
      else:
        m.start_at(*world_config("home")["start"])
        m.start_discovery()
        m._spin()
        source = {"tower": solutions.TOWER, "bench": solutions.WEIGH, "mouse": None}[args.feature]
        out = run(life, args.feature, source, frames)
    except MissionAborted:
      print("aborted (viewer closed)")
      return
    finally:
      m.close()
      if viewer is not None:
        viewer.close()
  proc = out["errand"]["procedure"]
  print()
  for step in proc["steps"]:
    extra = {k: v for k, v in step.items() if k not in ("i", "verb", "line", "ok")}
    print(f"  {'OK ' if step['ok'] else 'BAD'} {step['verb']:9s} {extra}")
  print(f"procedure: {proc['completed']}/{proc['total']} steps, "
        f"{'complete' if proc['ok'] else 'cut short'}, {proc['seconds']:.0f} sim s"
        + (f"; locals {proc['locals']}" if proc.get("locals") else "")
        + (f"; tools hung: {proc['toolsHung']}" if "toolsHung" in proc else ""))
  if args.feature == "mouse":
    care = out["care"]
    print(f"ACT: {'LANDED' if care and care['landed'] else 'FAILED'} -- "
          + (f"feed x{care['landed']}, the mouse {care['before']} -> {care['after']}, "
             f"{care['energyWh']:.2f} Wh" if care else "no care event"))
  else:
    grade = out["grade"]
    print(f"GRADE: {'PASSED' if grade and grade['ok'] else 'FAILED'} -- "
          f"{grade['reason'] if grade else 'never graded'}"
          f"{f' (+{grade['points']} points)' if grade else ''}")
  st = life.mission.swap.module_state("module_claw")
  print(f"claw: {'hung in its bay' if st['hung'] else 'NOT on the rack'}")
  print(f"sim {life.data.time:.0f} s, wall {time.time() - t0:.0f} s, "
        f"pack {life.battery.fraction:.0%}, "
        f"{(life.battery.capacity_wh - life.battery.energy_wh):.2f} Wh spent")
  filmstrip(frames, args.out)
  print(f"-> {args.out}")


if __name__ == "__main__":
  main()
