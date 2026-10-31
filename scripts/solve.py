"""Ladder A of issue #264 on legs: the mouse's paid feed and the whiteboards'
jobs, flown the way the robot's own attempt runs and graded by the job's own
grader -- so a passing verdict here is the verdict a robot would be paid
for, and a robot that never earns it is a finding about the robot.

  --feature mouse   the paid JOB (issue #403): `feed_mouse` offered,
                    claimed with a prediction, its errand walked (the plate
                    found by its sign and pressed) and graded by
                    `eval_feed` -- `--n` times, each from the dock or
                    (`--from lab`) from where the last ended, after one
                    walk in

  --feature answer | draw | artwork
                    a whiteboard's JOB (issue #406): `whiteboard_answer`
                    (claimed with the right answer to its question),
                    `draw_figure` (a house) or `rate_artwork` (the robot)
                    offered on a board -- `--board`, or the two in turn --
                    claimed, its errand run (the board found, the pen
                    fetched, drawn lying down, the pen hung back) and graded
                    by the job's own evaluator off the board's ink; `--n`
                    times, each from the dock

  --feature tower | bench
                    a CHALLENGE (issue #407): `stack_tower` or `find_mass`
                    offered as the cadence offers it -- the area's address
                    and directions, the bench's unknown drawn and set out --
                    claimed, the hand-written solution (`challenge/
                    solutions.py`: TOWER, WEIGH) defined in the robot's
                    library and run as its procedure errand, the bench's
                    finding recorded off the procedure's `mass` as a mind
                    would, `done`, and the grade on the seam; `--n` times,
                    the props set back out between, each from the dock

  --feature hide_and_seek
                    the pair's GAME (issue #404): offered as the cadence
                    offers it to a pair with minds, both roles claimed -- the
                    hider first -- each robot's role run from the queue
                    the referee fills, the referee's verdict banked on the
                    winner; one game a scene (`--scene`, both ways round
                    with `--swap`), both maps laid from the true floor, as
                    a robot that has explored its house knows it

  --feature drawer | door | cube | chest
                    a HUMANOID's hand job (issue #559, stage 3): `open_drawer`,
                    `open_door`, `place_cube` or `open_chest` offered on the
                    storeroom's counter as the cadence offers it -- the
                    area's address and directions, the prop's hidden
                    dynamics drawn afresh (`--draw K`, the board's own
                    sequence where none is given) and the prop set out --
                    claimed, the hand-written reference's two procedures
                    (`challenge/solutions.py`: the walk there, then the
                    hands' check's own) defined and run as the robot's,
                    `done`, and the grade on the seam; one fresh humanoid a
                    flight on a fresh map, from in front of its station
                    (`--from station`) or inside the storeroom's door
                    (`--from store`); `--n` draws from K, a process each
                    (`--jobs`), a row each (`--into DIR`); `--view` watches
                    one, `--strip` its filmstrip, `solve.png`

One quadruped in the house from its start, or with `--pair` the PAIR as
deployed (issue #353): hosting packs, near-field on, Luca at the dock and
Rowan in the hall; `--robot` flies while the other stands where it
started, one loop for both. The game is always the pair's.

Usage:
  MUJOCO_GL=egl uv run python scripts/solve.py --feature mouse
  uv run python scripts/solve.py --feature mouse --view
  MUJOCO_GL=egl uv run python scripts/solve.py --feature mouse --pair --n 10 --from lab
  MUJOCO_GL=egl uv run python scripts/solve.py --feature answer --pair --n 4
  MUJOCO_GL=egl uv run python scripts/solve.py --feature tower --pair
  MUJOCO_GL=egl uv run python scripts/solve.py --feature bench --pair --n 3
  MUJOCO_GL=egl uv run python scripts/solve.py --feature hide_and_seek --swap
  ... --feature hide_and_seek --scene served,kitchen --out games.json
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

from pluggybot import tick
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import (HUMANOID_HOME, QUAD_HOME, HubLifecycle, board_book,
                                 home_activities, world_config)
from pluggybot.mind import overseer as ov
from pluggybot.robot import world_spec
from pluggybot.tick import MissionAborted


class _Mind:
  """What the lifecycle reads off an overseer on a job's path: a library
  (the mark of a mind, where the lab's jobs are offered) and nothing that
  decides. The prediction is the script's, not a model's. Its presence is
  also what takes the rails off (`HubLifecycle.autonomous`)."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  library = object()
  decisions: list = []
  menu = ov.Menu.for_world(QUAD_HOME)


def build_life(view: bool, state_dir: str):
  """One quadruped in the house, as a served robot is built: the hosting
  pack, near-field on, the world's activities on its seam."""
  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=view, world=QUAD_HOME,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      battery_wh=cfg["hosting_battery_wh"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=Ledger(path=str(Path(state_dir) / "ledger.json")),
                      tasks=TaskBoard(path=str(Path(state_dir) / "tasks.json")),
                      boards=board_book(QUAD_HOME), overseer=_Mind(), near_field=True)
  acts = home_activities(model, data)
  life.body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  return life, viewer


def build_pair_lives(robot: int, state_dir: str, world: str = QUAD_HOME):
  """The home PAIR as deployed (issue #353): hosting packs, near-field on,
  and robot `robot` (1 or 2) given the job's mind and a task board."""
  from pluggybot.pair import build_pair
  lives = build_pair(world, pack="hosting", errands=("none", "none"),
                     overseer=False, near_field=True)
  life = lives[robot - 1]
  life.overseer = _Mind()
  life.tasks = TaskBoard(path=str(Path(state_dir) / "tasks.json"))
  # ...hooked as `HubLifecycle.__init__` hooks a board it is handed, or an
  # offer never sets the bench's mass or the props out (review of #353)
  life.tasks.on_event.append(life._bench_offered)
  return lives, life


def start(lives: list) -> None:
  """Each robot at its start, its map begun, every look-around in ONE
  loop: a quadruped whose stepper is not stepped lets go of its arm's hold
  (#405), and MEASURED, Luca stood 0.58 m from its belief while Rowan
  looked round alone."""
  cfg = world_config(lives[0].world)
  for each, at in zip(lives, (cfg["start"], cfg["start2"])):
    each.body.start_at(*at)
    each.body.start_discovery()
  tick.run_many([(each.body.stepper, each.body.look_around_routine())
                 for each in lives], name="look around")


def fly_beside(lives: list, life, routine):
  """`routine` flown by `life` while every other robot stands where it
  is, one physics loop for all (`tick.run_many`); its result."""
  done = [False]

  def flying():
    try:
      return (yield from routine)
    finally:
      done[0] = True

  def standing(other):
    while not done[0]:
      yield other.body.STILL
  return tick.run_many([(life.body.stepper, flying())]
                       + [(other.body.stepper, standing(other)) for other in lives
                          if other is not life], name="solve")[0]


#: What the hand-written feed predicts: a fed mouse eats.
FEED_PREDICTS = "eating"


def feed_job_routine(life, events: list):
  """The paid feed as a robot on legs does it (issue #403): the offer on
  the board, claimed with a prediction, its errand walked, the job graded
  by `eval_feed` through `scoring.evaluate` -- the door the robot's own
  claim goes through. What the cage counted on EVERY plate comes back, so a
  press of another is seen."""
  cage, data = life.cage, life.data
  counts = dict(cage.counts)
  task = life.tasks.offer("feed_mouse", "lab", t=float(data.time))
  assert task is not None and life._claim_task(task.id, answer=FEED_PREDICTS)
  errand = life.errands.pop(0)
  result = yield from life.run_errand_routine(errand)
  care = [e for e in events if e["type"] == "care"]
  done = life.tasks.get(task.id)
  verdict = dict(done.verdict or {}) if done is not None else {}
  return {"errand": result, "care": care[-1] if care else None,
          "grade": {"ok": bool(verdict.get("ok")), "reason": verdict.get("reason", ""),
                    "points": verdict.get("points", 0)},
          "presses": {k: cage.counts[k] - counts[k] for k in ("shock", "feed", "toy")},
          "presses_off_job": [e for e in events if e["type"] == "press"]}


#: The board jobs ladder A flies (issue #406): the kind, its params, the
#: secret its offer carries, and the claim's answer -- the right one.
BOARD_JOBS = {
  "answer": ("whiteboard_answer", {"question": "What is six times seven?"},
             {"answer": "42"}, "42"),
  "draw": ("draw_figure", {"program": "house"}, None, ""),
  "artwork": ("rate_artwork", {"program": "robot"}, None, ""),
}


def board_job_routine(life, feature: str, board: str, events: list):
  """A whiteboard's job as a robot on legs does it (issue #406): offered
  on `board`, claimed (an answer with its answer), its errand run and the
  job graded by its own evaluator through `scoring.evaluate`, off the ink
  the board book holds."""
  kind, params, secret, said = BOARD_JOBS[feature]
  task = life.tasks.offer(kind, board, params=params, secret=secret, t=float(life.data.time))
  assert task is not None and life._claim_task(task.id, answer=said)
  errand = life.errands.pop(0)
  result = yield from life.run_errand_routine(errand)
  done = life.tasks.get(task.id)
  verdict = dict(done.verdict or {}) if done is not None else {}
  return {"errand": result, "board": board,
          "grade": {"ok": bool(verdict.get("ok")), "reason": verdict.get("reason", ""),
                    "points": verdict.get("points", 0)},
          "presses": {"shock": 0, "feed": 0, "toy": 0}, "presses_off_job": []}


#: The challenges ladder A flies (issue #407): the kind and its target.
CHALLENGES = {"tower": ("stack_tower", "workshop"), "bench": ("find_mass", "lab_bench")}


def challenge_job_routine(life, feature: str, events: list):
  """A challenge's job, flown as `challenge.solutions.job_routine` flies it
  (issue #407), with the presses this report prints beside every job."""
  from pluggybot.challenge import solutions
  run = yield from solutions.job_routine(life, CHALLENGES[feature][0], QUAD_HOME)
  return {**run, "presses": {"shock": 0, "feed": 0, "toy": 0}, "presses_off_job": []}


def feed_trials_routine(life, n: int, start: str, events: list, feature: str = "mouse",
                        boards: tuple = ()):
  """`n` paid jobs on legs -- the feed, or a board's (`feature`, on
  `boards` in turn) -- each from the DOCK (walked back to it and lain on
  between) or, `start == "lab"`, each from where the last ended, after one
  walk in (issue #403: the success rate the issue asks for)."""
  out = []

  def job():
    if feature == "mouse":
      return feed_job_routine(life, events)
    if feature in CHALLENGES:
      return challenge_job_routine(life, feature, events)
    return board_job_routine(life, feature, boards[len(out) % len(boards)], events)

  if start == "lab":
    first = yield from job()
    print(f"  walk in: {'PASSED' if first['grade']['ok'] else 'FAILED'} "
          f"-- {first['grade']['reason']}")
  for i in range(n):
    if start == "dock":
      yield from life.go_charge_routine()
      yield from life.body.undock_routine()
    t0 = float(life.data.time)
    run = yield from job()
    x, y, _ = life.body.pose
    tx, ty, _ = life.body.true_pose()
    run["seconds"] = float(life.data.time) - t0
    run["drift"] = math.hypot(x - tx, y - ty)
    out.append(run)
    print(f"  {i + 1:2d}/{n}: {'PASSED' if run['grade']['ok'] else 'FAILED'} "
          f"{run['seconds']:5.0f} s, "
          + (f"{run['board']}, " if "board" in run else f"presses {run['presses']}, ")
          + f"belief {run['drift']:.2f} m off -- {run['grade']['reason']}")
    if "steps" in run:
      print(f"        steps {run['steps']}")
      print(f"        locals {run.get('locals')} secret {run.get('secret')}")
  return out


H = math.pi / 2
#: Where each game's two robots start, (hider, seeker), each (x, y, yaw):
#: the served pair's own starts first (Rowan's hall, Luca's living room),
#: then pairs about the house, each clear of the furniture.
GAME_SCENES = {
  "served": ((-3.5, 1.0, 0.0), (1.5, 0.5, H)),
  "living": ((0.0, -1.0, 0.0), (2.5, 1.8, math.pi)),
  "bedroom": ((1.5, 4.5, -H), (2.0, -1.0, H)),
  "kitchen": ((-8.5, 4.0, 0.0), (-3.5, -1.0, H)),
  "workshop": ((-8.5, -4.0, 0.0), (-3.5, 2.0, -H)),
  "garden": ((7.5, 2.0, math.pi), (2.0, 0.0, 0.0)),
  "hall": ((-3.5, -1.5, H), (-3.5, 3.5, -H)),
}


def build_game_pair(state_dir: str):
  """The home pair as served, each robot with the job's mind (`_Mind`:
  the game is offered where there are minds, and a mind takes the rails
  off) on a shared task board, and nothing offering work: the games are
  put up one at a time here, each refereed by the pair's referee
  (`arrange_game`). Both maps laid from the true floor
  (`unknown_spike.true_floor`)."""
  from unknown_spike import true_floor

  from pluggybot.pair import build_pair, referee_games
  lives = build_pair(QUAD_HOME, pack="hosting", errands=("none", "none"),
                     overseer=False, near_field=True,
                     tasks=True, task_state=str(Path(state_dir) / "tasks.json"),
                     ledger_state=str(Path(state_dir) / "ledger.json"),
                     thoughts_root=str(Path(state_dir) / "thoughts"))
  lives[0].producer = None
  for life in lives:
    life.overseer = _Mind()
    grid = life.body.mission.grid
    grid.grid[:] = true_floor(life.model, grid)
  # ...and the pair's referee, wired as a board that offers the game wires it
  referee_games(lives)
  return lives


def game_once(lives: list, hider, seeker, scene) -> dict:
  """One game from `scene`'s starts, played to the referee's call: what the
  referee said, who was paid, what each role cost (Wh), the hider's spot
  and the seeker's search, and how near the two came while it sought."""
  from pluggybot.pair import arrange_game
  for life, (x, y, yaw) in ((hider, scene[0]), (seeker, scene[1])):
    life.body.start_at(x, y, yaw)
  wh = {life.root: life.battery.energy_wh for life in lives}
  paid = {life.root: life.ledger.balance() for life in lives}
  t0 = float(hider.data.time)
  task, state = arrange_game(lives, t=t0)
  assert hider._claim_task(task.id) and seeker._claim_task(task.id)
  game = state["game"]
  assert game is not None and game.task_id == task.id
  near = [math.inf]

  def watch() -> None:
    if game.phase == "seeking" and game.flags.get("distanceM") is not None:
      near[0] = min(near[0], float(game.flags["distanceM"]))
  hider.body.step_hooks.append(watch)

  def play(life):
    return (yield from life.run_errand_routine(life.errands.pop(0)))
  try:
    runs = tick.run_many([(hider.body.stepper, play(hider)),
                          (seeker.body.stepper, play(seeker))], name="game")
  finally:
    hider.body.step_hooks.remove(watch)
  done = hider.tasks.get(task.id)
  verdict = dict(done.verdict or {})
  m = verdict.get("metrics") or {}
  by = {life.root: life for life in lives}
  winner = by.get(done.claims.get(m.get("winner") or "", ""))
  hide = dict(hider.body.mission.last_hide or {})
  seek = dict(seeker.body.mission.last_seek or {})
  return {"phase": game.phase, "winner": m.get("winner", ""),
          "paidTo": winner.robot_name if winner is not None else "",
          "paid": {life.robot_name: round(life.ledger.balance() - paid[life.root], 1)
                   for life in lives},
          "foundAtS": m.get("foundAtS"), "overAtS": m.get("overAtS"),
          "nearM": None if math.isinf(near[0]) else round(near[0], 2),
          "wh": {"hider": round(wh[hider.root] - hider.battery.energy_wh, 3),
                 "seeker": round(wh[seeker.root] - seeker.battery.energy_wh, 3)},
          "hide": {k: hide.get(k) for k in ("hid", "why", "at", "hidden", "walkM",
                                            "seekerWalkM", "reachM", "seconds")},
          "seek": {k: seek.get(k) for k in ("why", "targets", "gaveUp", "deferred",
                                             "seconds")},
          "steps": [r.get("procedure", {}).get("steps") for r in runs],
          "seconds": round(float(hider.data.time) - t0, 1)}


def game_main(args) -> None:
  import json

  from pluggybot.activity import hideseek as hs
  names = args.scene.split(",") if args.scene else list(GAME_SCENES)
  rows = []
  # the referee's clocks and reach, for a sweep: its own, and the program's
  # budget (`hide_and_seek_program` reads the module's)
  for flag, attr, const in ((args.head_s, "head_start_s", "SEEK_HEAD_START_S"),
                            (args.seek_s, "seek_s", "SEEK_S"),
                            (args.find, "find_within_m", "FIND_WITHIN_M")):
    if flag is not None:
      setattr(hs, const, flag)
  if args.sweep_step is not None:
    from pluggybot.legs import game as gm
    gm.SWEEP_STEP_M = args.sweep_step
  with tempfile.TemporaryDirectory() as state_dir:
    lives = build_game_pair(state_dir)
    game = lives[0].game
    game.head_start_s, game.seek_s = hs.SEEK_HEAD_START_S, hs.SEEK_S
    game.find_within_m = hs.FIND_WITHIN_M
    t0 = time.time()
    try:
      for name in names:
        hider_at, seeker_at = GAME_SCENES[name]
        for swap in ((False, True) if args.swap else (False,)):
          # ...both ways round: each robot starts where it did, roles swapped
          a, b = (lives[0], lives[1]) if swap else (lives[1], lives[0])
          scene = (seeker_at, hider_at) if swap else (hider_at, seeker_at)
          row = {"scene": name + ("/swapped" if swap else ""), "hider": a.robot_name,
                 **game_once(lives, a, b, scene)}
          rows.append(row)
          h, k = row["hide"], row["seek"]
          print(f"  {row['scene']:17s} {a.robot_name} hides: {row['phase']:5s} -> "
                f"{row['winner'] or 'nobody'} ({row['paidTo'] or '-'}) "
                f"found {row['foundAtS']} over {row['overAtS']}, nearest {row['nearM']} m; "
                f"hid {h['hid']} {h['why']} at {h['at']} hidden {h['hidden']} "
                f"(walk {h['walkM']}, seeker's {h['seekerWalkM']}); "
                f"seek {k['why']} {k['targets']} viewpoints ({k['gaveUp']} given up, "
                f"{k['deferred']} put off); "
                f"Wh {row['wh']['hider']:.2f}/{row['wh']['seeker']:.2f}", flush=True)
    finally:
      for life in lives:
        life.body.close()
  found = sum(1 for r in rows if r["winner"] == "seeker")
  over = sum(1 for r in rows if r["winner"] == "hider")
  print(f"\nHIDE AND SEEK on legs (head start {hs.SEEK_HEAD_START_S:g} s, seeking "
        f"{hs.SEEK_S:g} s, a find within {hs.FIND_WITHIN_M:g} m): "
        f"{len(rows)} games, the seeker found {found}, "
        f"the hider won {over}, uncalled {len(rows) - found - over}; dearest role "
        f"{max((max(r['wh'].values()) for r in rows), default=0.0):.2f} Wh; "
        f"wall {time.time() - t0:.0f} s")
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))


# ---- the humanoids' hand jobs (issue #559, stage 3) -------------------------------

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


class _HandsMind(_Mind):
  menu = ov.Menu.for_world(HUMANOID_HOME)


def hand_life(state_dir: str, start: str, viewer=None):
  """One humanoid in its house, as a served robot is built -- the hosting
  pack, near-field on, the world's activities on its seam, a board and a
  ledger, a mind with a library -- at `start`, a fresh map begun."""
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  data = mujoco.MjData(model)
  if viewer == "open":
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=viewer is not None,
                      world=HUMANOID_HOME, rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      battery_wh=cfg["hosting_battery_wh"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=Ledger(path=str(Path(state_dir) / "ledger.json")),
                      tasks=TaskBoard(path=str(Path(state_dir) / "tasks.json")),
                      boards=board_book(HUMANOID_HOME), overseer=_HandsMind(),
                      near_field=True)
  acts = home_activities(model, data)
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
  parser.add_argument("--feature", choices=("mouse", *BOARD_JOBS, *CHALLENGES,
                                             "hide_and_seek", *HAND_JOBS),
                      default="mouse",
                      help="the challenge: the mouse's paid feed, a whiteboard's job, "
                           "the tower or the bench, or the pair's game")
  parser.add_argument("--board", default=None,
                      help="a whiteboard's job on this board (default: the two in turn)")
  parser.add_argument("--view", action="store_true", help="open the viewer (one robot)")
  parser.add_argument("--pair", action="store_true",
                      help="the home pair as deployed; --robot flies, the other stands")
  parser.add_argument("--robot", type=int, choices=(1, 2), default=1,
                      help="with --pair: 1 flies from the dock, 2 from the hall")
  parser.add_argument("--n", type=int, default=1,
                      help="how many paid feeds, one after another")
  parser.add_argument("--from", dest="start",
                      choices=("dock", "lab", "station", "store", "tour"), default=None,
                      help="each feed from the dock, or from inside the lab; a hand job "
                           "from in front of the station (its default), inside the "
                           "storeroom's door, or each of `--n` from the next of eight "
                           "places across the house (`tour`)")
  parser.add_argument("--draw", type=int, default=None,
                      help="a hand job's draw of its prop's hidden dynamics (default: "
                           "the board's own sequence, 0; with --n, the first)")
  parser.add_argument("--jobs", type=int, default=1,
                      help="a hand job's flights at once, a process each")
  parser.add_argument("--into", default=None, help="a hand job's rows, DIR/rows.jsonl")
  parser.add_argument("--strip", action="store_true", help="a hand job's filmstrip")
  parser.add_argument("--strip-out", dest="out_strip", default="solve.png",
                      help="--strip's file")
  parser.add_argument("--scene", default=None, metavar="A,B",
                      help=f"the game's scenes (default all: {', '.join(GAME_SCENES)})")
  parser.add_argument("--swap", action="store_true",
                      help="each game's scene played both ways round")
  parser.add_argument("--out", default=None, help="the games' records, as JSON")
  parser.add_argument("--head-s", type=float, default=None,
                      help="the game's head start, s (default the referee's)")
  parser.add_argument("--seek-s", type=float, default=None,
                      help="the game's seeking, s (default the referee's)")
  parser.add_argument("--find", type=float, default=None,
                      help="the game's find, m (default the referee's)")
  parser.add_argument("--sweep-step", type=float, default=None,
                      help="the seeker's viewpoints' spacing, m (default legs.game's)")
  args = parser.parse_args()
  if args.feature == "hide_and_seek":
    game_main(args)
    return
  if args.feature in HAND_JOBS:
    args.start = args.start or "station"
    if args.start not in ("station", "store", "tour"):
      parser.error("a hand job starts from the station, the store or a tour")
    if args.start == "tour" and (args.view or args.strip):
      parser.error("--view and --strip fly one: --from station or store")
    hands_main(args)
    return
  args.start = args.start or "dock"
  if args.start not in ("dock", "lab"):
    parser.error("a quadruped's job starts from the dock or the lab")
  if args.pair and args.view:
    parser.error("--pair takes no --view")

  with tempfile.TemporaryDirectory() as state_dir:
    viewer = None
    if args.pair:
      lives, life = build_pair_lives(args.robot, state_dir)
    else:
      life, viewer = build_life(args.view, state_dir)
      lives = [life]
    others = [other for other in lives if other is not life]
    t0 = time.time()
    events: list = []
    life.on_event.append(events.append)
    try:
      start(lives)
      boards = (args.board,) if args.board else ("whiteboard_a", "whiteboard_b")
      trials = fly_beside(lives, life, feed_trials_routine(life, args.n, args.start, events,
                                                           args.feature, boards))
    except MissionAborted:
      print("aborted (viewer closed)")
      return
    finally:
      for each in lives:
        each.body.close()
      if viewer is not None:
        viewer.close()
  passed = sum(1 for run in trials if run["grade"]["ok"])
  other = {k: sum(run["presses"][k] for run in trials) for k in ("shock", "toy")}
  off = [e for e in events if e["type"] == "press"]
  print(f"\n{args.feature.upper()} on legs, from the {args.start}: {passed}/{len(trials)} paid; "
        f"presses of another plate: {other}; `press` events: {len(off)}; "
        f"mean {sum(r['seconds'] for r in trials) / max(len(trials), 1):.0f} sim s, "
        f"belief {max((r['drift'] for r in trials), default=0.0):.2f} m off at worst")
  for each in others:
    x, y, _ = each.body.true_pose()
    print(f"{each.robot_name}: stood at ({x:.2f}, {y:.2f})"
          + (f", DEAD ({each.dead['cause']})" if each.dead else ""))
  print(f"sim {life.data.time:.0f} s, wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
  main()
