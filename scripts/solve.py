"""Ladder A of issue #264 on the humanoid's hands (#559, stage 3): each
offered job flown the way the robot's own attempt runs and graded by the
job's own grader -- so a passing verdict here is the verdict a robot would
be paid for, and a robot that never earns it is a finding about the robot.

  --feature drawer | door | cube | chest
                    a hand job: `open_drawer`, `open_door`, `place_cube` or
                    `open_chest` offered on the storeroom's counter as the
                    cadence offers it -- the area's address and directions,
                    the prop's hidden dynamics drawn afresh (`--draw K`, the
                    board's own sequence where none is given) and the prop
                    set out -- claimed, the hand-written reference's two
                    procedures (`challenge/solutions.py`: the walk there,
                    then the hands' check's own) defined and run as the
                    robot's, `done`, and the grade on the seam; one fresh
                    humanoid a flight on a fresh map, from in front of its
                    station (`--from station`) or inside the storeroom's
                    door (`--from store`); `--n` draws from K, a process
                    each (`--jobs`), a row each (`--into DIR`); `--view`
                    watches one, `--strip` its filmstrip, `solve.png`

Usage:
  MUJOCO_GL=egl uv run python scripts/solve.py --feature drawer --n 8 --jobs 4 --into DIR
  uv run python scripts/solve.py --feature door --from store --view
  MUJOCO_GL=egl uv run python scripts/solve.py --feature chest --from store --strip
"""

import argparse
import math
import tempfile
import time
from pathlib import Path

import mujoco

from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, board_book, world_config
from pluggybot.mind import overseer as ov
from pluggybot.robot import world_spec


class _Mind:
  """What the lifecycle reads off an overseer on a job's path: a library
  (the mark of a mind, where a procedure's job is offered) and nothing that
  decides. Its presence is also what takes the rails off
  (`HubLifecycle.autonomous`)."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  library = object()
  decisions: list = []
  menu = ov.Menu.for_world(HUMANOID_HOME)


#: The hand jobs by the feature that flies each: the kind and its prop.
HAND_JOBS = {"drawer": ("open_drawer", "drawer"), "door": ("open_door", "door"),
             "cube": ("place_cube", "cube"), "chest": ("open_chest", "chest")}
#: Where a flight starts, (x, y, heading): in front of its station, the place
#: a job is taken from, or inside the storeroom's door facing its counter --
#: or, `tour`, each flight of `--n` at the next of eight places across the
#: house, so a batch varies how the robot arrives as well as its draw: the
#: station, the storeroom, the lobby and the lab across the street, the
#: living room, the hall, the kitchen and the garden (`home.SPAWNS`).
STORE_START = (22.8, -3.0, 0.0)
TOUR = ("station", "store", "lobby", "lab", "start", "hall", "kitchen", "garden")


def hand_life(state_dir: str, start: str, viewer=None):
  """One humanoid in its house, as a served robot is built -- the hosting
  pack, near-field on, the world's activities on its seam, a board and a
  ledger, a mind with a library -- at `start`, a fresh map begun."""
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  if viewer == "open":
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=viewer is not None,
                      world=HUMANOID_HOME, grid_bounds=cfg["grid_bounds"],
                      battery_wh=cfg["hosting_battery_wh"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=Ledger(path=str(Path(state_dir) / "ledger.json")),
                      tasks=TaskBoard(path=str(Path(state_dir) / "tasks.json")),
                      boards=board_book(HUMANOID_HOME), overseer=_Mind(),
                      near_field=True)
  acts = cfg["activities"](model, data)
  life.body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  from pluggybot.home import world as home
  at = (life.body.charge_standoff() if start == "station" else STORE_START if start == "store"
        else tuple(home.SPAWNS[start]))
  life.body.start_at(*at)
  life.body.start_discovery()
  return life, viewer


def hand_job(feature: str, start: str, draw: int | None, view: bool = False,
             strip: str | None = None) -> dict:
  """One hand job flown (the module docstring): its row -- the grade, the
  sim and wall seconds, the find, the falls, the belief against the truth
  and the procedures' verbs."""
  from pluggybot.challenge import solutions
  from pluggybot.home import storeroom
  kind, prop = HAND_JOBS[feature]
  wall = time.time()
  with tempfile.TemporaryDirectory() as state_dir:
    life, viewer = hand_life(state_dir, start, "open" if view else None)
    data, body = life.data, life.body
    if draw is not None:
      life.tasks.seq = int(draw)
    falls = []
    seen = [body.falls]

    def watch_falls():
      if body.falls != seen[0]:
        seen[0] = body.falls
        falls.append(round(float(data.time), 1))
    body.step_hooks.append(watch_falls)
    if viewer is not None:
      hx, hy = storeroom.handle(prop)
      viewer.cam.lookat[:] = [hx - 0.3, hy - 0.2, storeroom.COUNTER_H + 0.1]
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, -35.0, -22.0
    shots = _HandStrip(life, prop) if strip else None

    def fly():
      yield from body.look_around_routine()
      return (yield from solutions.job_routine(life, kind, HUMANOID_HOME))
    run = None
    try:
      run = body.run(fly())
    finally:
      if shots is not None:
        shots.done(run["grade"] if run is not None else None, strip)
      body.close()
      if viewer is not None and not view:
        viewer.close()
    x, y, _ = body.pose
    tx, ty, _ = body.true_pose()
    find = body.mission.last_find or {}
    row = {"feature": feature, "kind": kind, "from": start, "draw": run["secret"].get("draw"),
           "grade": run["grade"], "simS": round(float(data.time), 1),
           "wallS": round(time.time() - wall), "falls": falls,
           "find": {k: find.get(k) for k in ("found", "why", "seconds", "viewpoints",
                                             "arounds")},
           "beliefM": round(math.hypot(x - tx, y - ty), 3), "steps": run["steps"]}
  if view and viewer is not None:
    print(f"{feature}: {'PASSED' if row['grade']['ok'] else 'FAILED'} -- "
          f"{row['grade']['reason']}. Close the window to quit.")
    while viewer.is_running():
      viewer.sync()
      time.sleep(0.05)
  return row


class _HandStrip:
  """A hand job's filmstrip: a tile from behind the robot as it sets off
  and once a minute on its way, then -- once it stands within `NEAR_M` of
  the prop, about where its walk there ends -- a tile every 4 s from beside
  the counter, the grade's last. (At 1.8 m a search's viewpoint near the
  counter drew it with the robot out of the frame.)"""

  TILES, WALK_EVERY_S, WORK_EVERY_S, NEAR_M = 12, 60.0, 4.0, 1.2

  def __init__(self, life, prop: str) -> None:
    from pluggybot.home import storeroom
    self.life, self.model, self.data = life, life.model, life.data
    self.r = mujoco.Renderer(self.model, 300, 400)
    self.tiles: list = []
    self.prop = prop
    self.hx, self.hy = storeroom.handle(prop)
    self.due = 0.0
    self.near = None
    life.body.step_hooks.append(self.hook)

  def _shoot(self, label: str, near: bool) -> None:
    from PIL import Image, ImageDraw
    from pluggybot.home import storeroom
    m, d = self.model, self.data
    cam = mujoco.MjvCamera()
    if near:
      # from the robot's left, the side its working hand and the prop are on
      cam.lookat[:] = [self.hx - 0.25, self.hy - 0.15, storeroom.COUNTER_H + 0.1]
      cam.distance, cam.azimuth, cam.elevation = 2.0, -70.0, -22.0
    else:
      # from behind, or turned round it where a wall stands in between (a
      # walk just through a doorway was drawn as the wall behind it, from
      # above the 2.4 m walls too)
      x, y, yaw = self.life.body.true_pose()
      cam.lookat[:] = [x, y, 0.8]
      cam.distance, cam.elevation = 3.5, -20.0
      for turn in (0.0, 60.0, -60.0, 120.0, -120.0, 180.0):
        cam.azimuth = math.degrees(yaw) + turn
        if not self._walled(cam):
          break
    light = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.4, 0.55
    self.r.update_scene(d, cam)
    img = Image.fromarray(self.r.render())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = light
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    self.tiles.append(img)

  def _walled(self, cam) -> bool:
    """Whether something fixed to the world stands between the camera and
    what it looks at."""
    import numpy as np
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    ahead = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    hit = np.zeros(1, dtype=np.int32)
    far = mujoco.mj_ray(self.model, self.data, np.array(cam.lookat) - cam.distance * ahead,
                        ahead, None, 1, -1, hit)
    return (hit[0] >= 0 and far < cam.distance - 0.5
            and self.model.body_weldid[self.model.geom_bodyid[hit[0]]] == 0)

  def hook(self) -> None:
    t = float(self.data.time)
    if len(self.tiles) >= self.TILES - 1:
      return
    x, y, _ = self.life.body.true_pose()
    near = math.hypot(x - self.hx, y - self.hy) < self.NEAR_M
    # ...a tile as it comes to the prop, whenever the walk's next was due
    if t < self.due and near == self.near:
      return
    self.near = near
    self.due = t + (self.WORK_EVERY_S if near else self.WALK_EVERY_S)
    self._shoot(f"{t:.0f} s" + ("" if near else ": on its way"), near)

  def done(self, grade, out: str) -> None:
    from PIL import Image
    if grade is not None:
      self._shoot(f"{self.data.time:.0f} s: {'PASSED' if grade['ok'] else 'failed'}", True)
    self.r.close()
    img = Image.new("RGB", (400 * min(4, len(self.tiles)), 300 * math.ceil(len(self.tiles) / 4)),
                    (255, 255, 255))
    for i, tile in enumerate(self.tiles):
      img.paste(tile, (400 * (i % 4), 300 * (i // 4)))
    img.save(out)
    print(f"wrote {out}")


def _hand_job_row(spec) -> dict:
  feature, start, draw = spec
  try:
    return hand_job(feature, start, draw)
  except Exception as e:  # noqa: BLE001 -- a row says what raised
    return {"feature": feature, "from": start, "draw": draw, "error": repr(e)}


def hands_main(args) -> None:
  """`--feature drawer|door|cube|chest`: one flight, or `--n` draws from
  `--draw` (0 where none is given), a process each, a row each."""
  import json
  import multiprocessing as mp
  if args.view or args.strip:
    row = hand_job(args.feature, args.start, args.draw, view=args.view,
                   strip=args.out_strip if args.strip else None)
    print(json.dumps({k: v for k, v in row.items() if k != "steps"}))
    return
  first = 0 if args.draw is None else args.draw
  specs = [(args.feature, TOUR[i % len(TOUR)] if args.start == "tour" else args.start, first + i)
           for i in range(args.n)]
  rows = []
  with mp.get_context("spawn").Pool(max(1, args.jobs), maxtasksperchild=1) as pool:
    for row in pool.imap(_hand_job_row, specs):
      rows.append(row)
      g = row.get("grade") or {}
      print(f"  draw {row['draw']}: " + (f"ERROR {row['error']}" if "error" in row else
            f"{'PASSED' if g.get('ok') else 'FAILED'} {row['simS']:5.0f} s, falls {row['falls']}, "
            f"belief {row['beliefM']:.2f} m off -- {g.get('reason')}"), flush=True)
      if args.into:
        Path(args.into).mkdir(parents=True, exist_ok=True)
        with open(Path(args.into) / "rows.jsonl", "a") as f:
          f.write(json.dumps(row) + "\n")
  passed = sum(1 for r in rows if (r.get("grade") or {}).get("ok"))
  print(f"\n{args.feature.upper()} from the {args.start}: {passed}/{len(rows)} passed")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--feature", choices=tuple(HAND_JOBS), default="drawer",
                      help="the hand job: the drawer, the door, the cube or the chest")
  parser.add_argument("--view", action="store_true", help="watch one flight in the viewer")
  parser.add_argument("--n", type=int, default=1,
                      help="how many draws, from --draw, a process each")
  parser.add_argument("--from", dest="start", choices=("station", "store", "tour"),
                      default="station",
                      help="each flight from in front of the station, inside the "
                           "storeroom's door, or each of `--n` from the next of eight "
                           "places across the house (`tour`)")
  parser.add_argument("--draw", type=int, default=None,
                      help="the draw of its prop's hidden dynamics (default: the "
                           "board's own sequence, 0; with --n, the first)")
  parser.add_argument("--jobs", type=int, default=1, help="flights at once, a process each")
  parser.add_argument("--into", default=None, help="the rows, DIR/rows.jsonl")
  parser.add_argument("--strip", action="store_true", help="one flight's filmstrip")
  parser.add_argument("--strip-out", dest="out_strip", default="solve.png",
                      help="--strip's file")
  args = parser.parse_args()
  if args.start == "tour" and (args.view or args.strip):
    parser.error("--view and --strip fly one: --from station or store")
  hands_main(args)


if __name__ == "__main__":
  main()
