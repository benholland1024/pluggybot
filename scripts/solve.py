"""Ladder A of issue #264 on the humanoid's hands (#559, stage 3): each
offered job flown the way the robot's own attempt runs and graded by the
job's own grader -- so a passing verdict here is the verdict a robot would
be paid for, and a robot that never earns it is a finding about the robot.

  --feature drawer | door | cube | chest | table | pass
                    a hand job: `open_drawer`, `open_door`, `place_cube` or
                    `open_chest` offered on the storeroom's counter, or
                    `set_table` at the restaurant's table (#560) -- or
                    `set_table_pass`, its place laid from the pass (#682's
                    stage C, never offered) -- as the cadence offers it -- the area's address and directions,
                    the prop's hidden dynamics drawn afresh (`--draw K`, the
                    board's own sequence where none is given) and the prop
                    set out -- claimed, the hand-written reference's two
                    procedures (`challenge/solutions.py`: the walk there,
                    then the hands' check's own) defined and run as the
                    robot's, `done`, and the grade on the seam; one fresh
                    humanoid a flight on a fresh map, from in front of its
                    station (`--from station`) or inside the storeroom's
                    door (`--from store`), or in front of the restaurant's
                    pass (`--from pass`); `--n` draws from K, a process
                    each (`--jobs`), a row each (`--into DIR`); `--view`
                    watches one, `--strip` its filmstrip, `solve.png`.
                    Each flight draws noise of its own, keyed by its job,
                    its draw and where it starts (`noise_of`, #560), or
                    `--noise KEY`'s: a draw flown again (a row's `noise`),
                    `robot` the robot's own, as every flight drew before;
                    `--walker NPZ` another walking policy (#682's stage B's
                    `models/humanoid_walk_carry.npz`), walk2 unset;
                    `--held-out N` N draws nobody has flown, sealed by the
                    frozen build's commit as the attempt kit seals a held-out
                    chest (`attempts.seed_of`, `held_out`; a dirty tree is
                    refused), each its own noise

Usage:
  MUJOCO_GL=egl uv run python scripts/solve.py --feature drawer --n 8 --jobs 4 --into DIR
  uv run python scripts/solve.py --feature door --from store --view
  MUJOCO_GL=egl uv run python scripts/solve.py --feature chest --from store --strip
"""

import argparse
import math
import os
import tempfile
import time
from pathlib import Path

import mujoco

from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, board_book, world_config
from pluggybot.mind import overseer as ov
from pluggybot.robot import world_spec

#: `--walker NPZ`: the walking policy flown (the served one, walk2, unset) --
#: through the environment, so a pool's spawned flights fly it too.
WALKER = os.environ.get("SOLVE_WALKER")
if WALKER:
  from pluggybot.humanoid import mission as _mission
  _mission.WALK_NPZ = Path(WALKER)


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


#: The hand jobs by the feature that flies each: the kind and its target --
#: a counter's prop, or the restaurant's table (#560).
HAND_JOBS = {"drawer": ("open_drawer", "drawer"), "door": ("open_door", "door"),
             "cube": ("place_cube", "cube"), "chest": ("open_chest", "chest"),
             "table": ("set_table", "dining_table_0"),
             "pass": ("set_table_pass", "dining_table_0")}


def work_view(target: str) -> tuple[tuple[float, float], list[float], float, float]:
  """Where a job's work is, (x, y), and a camera on it -- (lookat, distance,
  azimuth deg): a counter's prop's handle from the robot's left, the side its
  working hand and the prop are on; the restaurant's place from across its
  table, where the robot does not stand between the camera and the place
  (`scripts/hands_check.py`'s strip)."""
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import storeroom
  if target in storeroom.PROPS:
    hx, hy = storeroom.handle(target)
    return (hx, hy), [hx - 0.25, hy - 0.15, storeroom.COUNTER_H + 0.1], 2.0, -70.0
  cx, cy, facing = ct.table_set_out(0).card
  sx, sy = ct.place_to_world(-ct.TABLE_GAP, 0.0)
  return ((cx, cy), [0.6 * cx + 0.4 * sx, 0.6 * cy + 0.4 * sy, ct.TABLE_H + 0.15], 1.8,
          math.degrees(facing) + 20.0)


def pass_view() -> tuple[tuple[float, float], list[float], float, float]:
  """The pass's work, as `work_view` gives a job's: the middle of the expo
  counter's dining edge, and a camera from across the counter onto the
  robot's hands at it (`--feature pass`'s pick-ups)."""
  from pluggybot.evaluation import counter as ct
  px, py = ct.pass_to_world(0.0, 0.0)
  return (px, py), [px - 0.2, py, ct.PASS_H + 0.1], 2.2, 140.0
#: Where a flight starts, (x, y, heading): in front of its station, the place
#: a job is taken from, or inside the storeroom's door facing its counter --
#: or, `tour`, each flight of `--n` at the next of eight places across the
#: house, so a batch varies how the robot arrives as well as its draw: the
#: station, the storeroom, the lobby and the lab across the street, the
#: living room, the hall, the kitchen and the garden (`home.SPAWNS`).
STORE_START = (22.8, -3.0, 0.0)
TOUR = ("station", "store", "lobby", "lab", "start", "hall", "kitchen", "garden")


def pass_start() -> tuple[float, float, float]:
  """In front of the restaurant's pass, facing it (`--from pass`): where an
  offline flight of the place laid from it starts (`counter.pass_set_out`)."""
  from pluggybot.evaluation import counter as ct
  return ct.pass_set_out(0).start


def noise_of(kind: str, start: str, draw: int | None) -> str:
  """The noise a flight draws (`humanoid.senses.Noise`): its job's, its
  draw's and its start's -- every flight its own, as it draws its own
  prop (#560)."""
  return f"{kind}:{0 if draw is None else int(draw)}:{start}"


def hand_life(state_dir: str, start: str, viewer=None, noise: str | None = None):
  """One humanoid in its house, as a served robot is built -- the hosting
  pack, near-field on, the world's activities on its seam, a board and a
  ledger, a mind with a library -- at `start`, a fresh map begun, drawing
  `noise`'s noise (None: the robot's own)."""
  from pluggybot.body import body_for
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  data = mujoco.MjData(model)
  if viewer == "open":
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  body = body_for(model, data, viewer=viewer, realtime=viewer is not None,
                  grid_bounds=cfg["grid_bounds"], noise=noise)
  life = HubLifecycle(model, data, body=body,
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
        else pass_start() if start == "pass" else tuple(home.SPAWNS[start]))
  life.body.start_at(*at)
  life.body.start_discovery()
  return life, viewer


def hand_job(feature: str, start: str, draw: int | None, view: bool = False,
             strip: str | None = None, noise: str | None = None) -> dict:
  """One hand job flown (the module docstring): its row -- the grade, the
  sim and wall seconds, the find, the falls, the belief against the truth,
  the noise it drew and the procedures' verbs."""
  from pluggybot.challenge import solutions
  kind, prop = HAND_JOBS[feature]
  noise = noise_of(kind, start, draw) if noise is None else noise
  wall = time.time()
  with tempfile.TemporaryDirectory() as state_dir:
    life, viewer = hand_life(state_dir, start, "open" if view else None, noise)
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
      _, lookat, distance, azimuth = work_view(prop)
      viewer.cam.lookat[:] = lookat
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = distance + 0.2, azimuth, -22.0
    shots = None if not strip else (
      _HandStrip(life, prop, also=(pass_view(),), tiles=20, work_every=15.0) if feature == "pass"
      else _HandStrip(life, prop))

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
           "noise": noise, "grade": run["grade"], "simS": round(float(data.time), 1),
           "wallS": round(time.time() - wall), "falls": falls,
           "find": {k: find.get(k) for k in ("found", "why", "seconds", "viewpoints",
                                             "arounds")},
           "beliefM": round(math.hypot(x - tx, y - ty), 3), "steps": run["steps"]}
    if WALKER:
      row["walker"] = os.path.basename(WALKER)
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
  counter drew it with the robot out of the frame.) A job whose work is in
  two places -- the place laid from the pass: its pick-ups at the pass, its
  put-downs at the place -- names the other (`also`), a camera on each, and
  takes more tiles further apart."""

  TILES, WALK_EVERY_S, WORK_EVERY_S, NEAR_M = 12, 60.0, 4.0, 1.2

  def __init__(self, life, prop: str, also: tuple = (), tiles: int | None = None,
               work_every: float | None = None) -> None:
    self.life, self.model, self.data = life, life.model, life.data
    self.r = mujoco.Renderer(self.model, 300, 400)
    self.tiles: list = []
    self.prop = prop
    #: each place the work is, a camera on it: the prop's first
    self.works = (work_view(prop), *also)
    self.TILES = tiles or self.TILES
    self.WORK_EVERY_S = work_every or self.WORK_EVERY_S
    self.due = 0.0
    self.near = None
    life.body.step_hooks.append(self.hook)

  def _shoot(self, label: str, near: int | None) -> None:
    from PIL import Image, ImageDraw
    m, d = self.model, self.data
    cam = mujoco.MjvCamera()
    if near is not None:
      # on the work it stands at (`work_view`, `pass_view`)
      _, lookat, distance, azimuth = self.works[near]
      cam.lookat[:] = lookat
      cam.distance, cam.azimuth, cam.elevation = distance, azimuth, -22.0
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
    near = next((i for i, ((hx, hy), *_) in enumerate(self.works)
                 if math.hypot(x - hx, y - hy) < self.NEAR_M), None)
    # ...a tile as it comes to the work, whenever the walk's next was due
    if t < self.due and near == self.near:
      return
    self.near = near
    self.due = t + (self.WORK_EVERY_S if near is not None else self.WALK_EVERY_S)
    self._shoot(f"{t:.0f} s" + ("" if near is not None else ": on its way"), near)

  def done(self, grade, out: str) -> None:
    from PIL import Image
    if grade is not None:
      self._shoot(f"{self.data.time:.0f} s: {'PASSED' if grade['ok'] else 'failed'}", 0)
    self.r.close()
    img = Image.new("RGB", (400 * min(4, len(self.tiles)), 300 * math.ceil(len(self.tiles) / 4)),
                    (255, 255, 255))
    for i, tile in enumerate(self.tiles):
      img.paste(tile, (400 * (i % 4), 300 * (i // 4)))
    img.save(out)
    print(f"wrote {out}")


def _drafts(args) -> list[int]:
  """The draws `--n` flies from `--draw`, or `--held-out`'s, sealed by the
  frozen build (the attempt kit's `build`, `seed_of` and `held_out`)."""
  if args.held_out is None:
    first = 0 if args.draw is None else args.draw
    return [first + i for i in range(args.n)]
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from attempts import build
  from pluggybot.evaluation import attempts as at
  info = build()
  if info["dirty"] and args.seed is None:
    raise SystemExit("a held-out draw is a frozen build's: commit the tree first (or name --seed)")
  seed = args.seed if args.seed is not None else at.seed_of(info["commit"])
  ks = at.held_out(seed, args.held_out)
  print(f"held out: {args.held_out} draws by seed {seed} (commit {info['commit']}): {ks}",
        flush=True)
  return ks


def _hand_job_row(spec) -> dict:
  feature, start, draw, noise = spec
  try:
    return hand_job(feature, start, draw, noise=noise)
  except Exception as e:  # noqa: BLE001 -- a row says what raised
    return {"feature": feature, "from": start, "draw": draw, "error": repr(e)}


def hands_main(args) -> None:
  """`--feature drawer|door|cube|chest`: one flight, or `--n` draws from
  `--draw` (0 where none is given), a process each, a row each."""
  import json
  import multiprocessing as mp
  if args.view or args.strip:
    row = hand_job(args.feature, args.start, args.draw, view=args.view,
                   strip=args.out_strip if args.strip else None, noise=args.noise)
    print(json.dumps({k: v for k, v in row.items() if k != "steps"}))
    return
  specs = [(args.feature, TOUR[i % len(TOUR)] if args.start == "tour" else args.start, k,
            args.noise) for i, k in enumerate(_drafts(args))]
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
  from pluggybot.batch import watched
  watched()
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--feature", choices=tuple(HAND_JOBS), default="drawer",
                      help="the hand job: the drawer, the door, the cube, the chest or the table")
  parser.add_argument("--view", action="store_true", help="watch one flight in the viewer")
  parser.add_argument("--n", type=int, default=1,
                      help="how many draws, from --draw, a process each")
  parser.add_argument("--from", dest="start", choices=("station", "store", "pass", "tour"),
                      default="station",
                      help="each flight from in front of the station, inside the "
                           "storeroom's door, in front of the restaurant's pass, or each "
                           "of `--n` from the next of eight places across the house (`tour`)")
  parser.add_argument("--draw", type=int, default=None,
                      help="the draw of its prop's hidden dynamics (default: the "
                           "board's own sequence, 0; with --n, the first)")
  parser.add_argument("--noise", default=None, metavar="KEY",
                      help="every flight draws KEY's noise, not its own (a row's `noise`; "
                           "`robot` the robot's own)")
  parser.add_argument("--jobs", type=int, default=1, help="flights at once, a process each")
  parser.add_argument("--into", default=None, help="the rows, DIR/rows.jsonl")
  parser.add_argument("--strip", action="store_true", help="one flight's filmstrip")
  parser.add_argument("--held-out", dest="held_out", type=int, default=None, metavar="N",
                      help="N draws nobody has flown, sealed by the frozen build's commit")
  parser.add_argument("--seed", type=int, default=None,
                      help="--held-out's seed, not the build's")
  parser.add_argument("--walker", default=None, metavar="NPZ",
                      help="another walking policy's .npz (#682's stage B), walk2 unset")
  parser.add_argument("--strip-out", dest="out_strip", default="solve.png",
                      help="--strip's file")
  args = parser.parse_args()
  if args.held_out is not None:
    # ...as the attempt kit refuses them: a held-out draw is sealed with its
    # noise, and its draws are the build's, never `--draw`'s or `--n`'s
    if args.noise is not None:
      parser.error("a held-out draw's noise is sealed with it: --noise flies a draw again "
                   "at a named --draw")
    if args.view or args.strip:
      parser.error("--held-out flies a batch: --view and --strip fly one named --draw")
    if args.draw is not None or args.n != 1:
      parser.error("--held-out N names its own draws: no --draw, no --n")
  if args.walker:
    # ...through the environment, so a pool's spawned flights fly it too
    os.environ["SOLVE_WALKER"] = str(Path(args.walker).expanduser().resolve())
    global WALKER
    WALKER = os.environ["SOLVE_WALKER"]
    from pluggybot.humanoid import mission
    mission.WALK_NPZ = Path(WALKER)
  if args.start == "tour" and (args.view or args.strip):
    parser.error("--view and --strip fly one: --from station or store")
  hands_main(args)


if __name__ == "__main__":
  main()
