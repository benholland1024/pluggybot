"""Ladder A of issue #264 on legs: the mouse's paid feed, flown the way the
robot's own attempt runs and graded by the job's own grader -- so a passing
verdict here is the verdict a robot would be paid for, and a robot that
never earns it is a finding about the robot.

  --feature mouse   the paid JOB (issue #403): `feed_mouse` offered,
                    claimed with a prediction, its errand walked (the plate
                    found by its sign and pressed) and graded by
                    `eval_feed` -- `--n` times, each from the dock or
                    (`--from lab`) from where the last ended, after one
                    walk in

One quadruped in the house from its start, or with `--pair` the PAIR as
deployed (issue #353): hosting packs, near-field on, Luca at the dock and
Rowan in the hall; `--robot` flies while the other stands where it
started, one loop for both.

Usage:
  MUJOCO_GL=egl uv run python scripts/solve.py --feature mouse
  uv run python scripts/solve.py --feature mouse --view
  MUJOCO_GL=egl uv run python scripts/solve.py --feature mouse --pair --n 10 --from lab
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
from pluggybot.lifecycle import QUAD_HOME, HubLifecycle, home_activities, world_config
from pluggybot.mind import overseer as ov
from pluggybot.robot import world_spec
from pluggybot.tick import MissionAborted


class _Mind:
  """What the lifecycle reads off an overseer on a job's path: a library
  (the mark of the autonomous arm, where the lab's jobs are offered) and
  nothing that decides. The prediction is the script's, not a model's."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  workshop = None
  library = object()
  decisions: list = []
  menu = ov.Menu.for_world(QUAD_HOME)


def build_life(view: bool, state_dir: str):
  """One quadruped in the house, as a served robot is built: the hosting
  pack, near-field on, the world's activities on its seam."""
  cfg = world_config(QUAD_HOME)
  spec = world_spec(cfg["model"], body=cfg["body"])
  model = spec.compile()
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=view, world=QUAD_HOME,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"], spec=spec,
                      battery_wh=cfg["hosting_battery_wh"],
                      low_battery_wh=cfg["low_battery_wh"],
                      ledger=Ledger(path=str(Path(state_dir) / "ledger.json")),
                      tasks=TaskBoard(path=str(Path(state_dir) / "tasks.json")),
                      autonomous=True, overseer=_Mind(), near_field=True)
  acts = home_activities(model, data)
  life.body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  return life, viewer


def build_pair_lives(robot: int, state_dir: str, world: str = QUAD_HOME):
  """The home PAIR as deployed (issue #353): hosting packs, near-field on,
  and robot `robot` (1 or 2) given the job's mind and a task board."""
  from pluggybot.pair import build_pair
  lives = build_pair(world, pack="hosting", errands=("none", "none"),
                     autonomous=True, overseer=False, near_field=True)
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


def feed_trials_routine(life, n: int, start: str, events: list):
  """`n` paid feeds on legs, each from the DOCK (walked back to it and
  lain on between) or, `start == "lab"`, each from where the last ended,
  after one walk in (issue #403: the success rate the issue asks for)."""
  out = []
  if start == "lab":
    first = yield from feed_job_routine(life, events)
    print(f"  walk in: {'PASSED' if first['grade']['ok'] else 'FAILED'} "
          f"-- {first['grade']['reason']}")
  for i in range(n):
    if start == "dock":
      yield from life.go_charge_routine()
      yield from life.body.undock_routine()
    t0 = float(life.data.time)
    run = yield from feed_job_routine(life, events)
    x, y, _ = life.body.pose
    tx, ty, _ = life.body.true_pose()
    run["seconds"] = float(life.data.time) - t0
    run["drift"] = math.hypot(x - tx, y - ty)
    out.append(run)
    print(f"  {i + 1:2d}/{n}: {'PASSED' if run['grade']['ok'] else 'FAILED'} "
          f"{run['seconds']:5.0f} s, presses {run['presses']}, "
          f"belief {run['drift']:.2f} m off -- {run['grade']['reason']}")
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--feature", choices=("mouse",), default="mouse",
                      help="the challenge: the mouse's paid feed")
  parser.add_argument("--view", action="store_true", help="open the viewer (one robot)")
  parser.add_argument("--pair", action="store_true",
                      help="the home pair as deployed; --robot flies, the other stands")
  parser.add_argument("--robot", type=int, choices=(1, 2), default=1,
                      help="with --pair: 1 flies from the dock, 2 from the hall")
  parser.add_argument("--n", type=int, default=1,
                      help="how many paid feeds, one after another")
  parser.add_argument("--from", dest="start", choices=("dock", "lab"), default="dock",
                      help="each feed from the dock, or from inside the lab")
  args = parser.parse_args()
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
      trials = fly_beside(lives, life, feed_trials_routine(life, args.n, args.start, events))
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
  print(f"\nFEED on legs, from the {args.start}: {passed}/{len(trials)} paid; "
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
