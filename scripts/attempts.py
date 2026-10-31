"""Attempts at a task, graded by the world (issue #493, stages 1 and 2): the
attempt protocol (`evaluation.attempts`), the lid's grader
(`challenge/lid.py`) and the experience record, with our opener as code as
the skill -- the humanoid's reference at its counter (`evaluation.opener`).

Each attempt is one try from its task's set-out, built afresh (the world's
reset, counted): drawn chest k on a counter and the humanoid in front of it
(`evaluation.counter`). The skill runs inside its budget, the grader
watches every step, and 5 s after the call one verdict goes through
`scoring.evaluate`. A row a try.

  --body B             humanoid (the reference, a chest on a counter), the
                       one there is
  --chests A-B|A,B,..  the chests (default 0-31: public)
  --held-out N         N chests nobody has flown instead, drawn by the
                       frozen build's seed (`attempts.seed_of`), or `--seed`
  --attempts N         tries a chest (default 1: code that learns nothing
                       gets the same verdict at every one)
  --jobs J             chests in parallel, each in a process of its own
  --into DIR           the record: rows.jsonl, one row an attempt
  --read DIR           a kept record, read as the per-task record
                       (`qualities.per_task_record`), its failures listed
  --drops              what the grader's constants were read off, no robot:
                       the 128 drawn lids let fall from rest at a range of
                       angles -- the fastest each closes while open, the
                       speed it lands at -- and how long each takes to shut
                       and be caught, from the shut line and from the top
  --view [K]           an attempt at chest K (0) in the MuJoCo viewer, at
                       real time, the verdict printed; MUJOCO_GL left off
  --strip [K]          its filmstrip: set out, opened, the call, the hold's
                       end -- attempts.png, or `--out`
  --record PATH        the humanoid's attempt at chest `--chest` (0) as the
                       site replays it (#530): a recording in a world of its
                       own (`session.attempt`'s stream: the set-out's scene,
                       the robot's senses, the verdict), and the scene and
                       its textures beside it

Usage:
  MUJOCO_GL=egl uv run python scripts/attempts.py --chests 0-3 --jobs 2
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
    uv run python scripts/attempts.py --chests 0-31 --jobs 3 --into DIR
  uv run python scripts/attempts.py --read DIR
  uv run python scripts/attempts.py --drops
  uv run python scripts/attempts.py --view 0
  MUJOCO_GL=egl uv run python scripts/attempts.py \\
    --record protocol/telemetry.home_humanoid_counter.jsonl.gz
"""

import os

# One BLAS thread before numpy loads (a policy's small products doubled a
# step on six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import subprocess  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.activity import chest as ch  # noqa: E402
from pluggybot.challenge import lid  # noqa: E402
from pluggybot.evaluation import attempts as at  # noqa: E402
from pluggybot.evaluation import qualities as q  # noqa: E402
from pluggybot.evaluation.identity import COMMIT_ENV, REPO  # noqa: E402

#: The reference's own word on itself: code, no LLM, no imagination, no
#: procedure to count steps of.
REFERENCE_INFO = {"process": "code", "skill": "opener", "tokens": 0, "imaginationCpuS": 0.0,
                  "steps": None}


def opener_skill(body):
  """The humanoid's reference (`evaluation.opener`), through `Body`: its own
  word is `ok` once its hand went up the arc and back down it."""
  from pluggybot.evaluation.opener import opener
  return opener(body)


#: Each body's reference: its skill, what it says of itself on the row, and
#: its task's set-out.
SKILLS = {"humanoid": (opener_skill, REFERENCE_INFO, {"body": "humanoid", "mount": "counter"})}


def build() -> dict:
  """Which build flies, for the record and the held-out seed: `$PLUGGY_COMMIT`,
  else the `COMMIT` file a pod's tree carries (`training/pod.sh setup-sim`:
  a `git archive`, no `.git`), else git's HEAD -- and whether the tree has
  changes HEAD does not (MuJoCo's own log aside)."""
  baked = os.environ.get(COMMIT_ENV, "").strip()
  if baked:
    return {"commit": at.commit_of(baked), "dirty": False}
  kept = REPO / "COMMIT"
  if kept.exists():
    return {"commit": at.commit_of(kept.read_text()), "dirty": False}
  head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True, cwd=REPO).stdout
  changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True, cwd=REPO).stdout
  dirty = [line for line in changed.splitlines() if not line.endswith("MUJOCO_LOG.TXT")]
  return {"commit": at.commit_of(head), "dirty": bool(dirty)}


def chests(spec: str) -> list[int]:
  out = []
  for part in spec.split(","):
    a, _, b = part.partition("-")
    out += list(range(int(a), int(b) + 1)) if b else [int(a)]
  return out


def fly(job) -> list[dict]:
  """One chest's attempts, each from a fresh set-out. An attempt that
  raises is a row saying what (`error`), its chest's earlier rows kept and
  its series ended there: `Pool.map` raises the first error a worker meets,
  and the batch's rows would go with it."""
  k, n, info, who = job
  skill, _, where = SKILLS[who]
  task = at.Task(chest=k, **where)
  rows = []
  for i in range(1, n + 1):
    try:
      rows.append(at.attempt(task, skill, i, info=dict(info)))
    except Exception as e:
      traceback.print_exc()
      rows.append({"task": task.as_dict(), "attempt": i, "heldOut": task.held_out,
                   "process": info.get("process"), "skill": info.get("skill"),
                   "run": {k_: info[k_] for k_ in ("commit", "dirty", "seed") if k_ in info},
                   "error": f"{type(e).__name__}: {e}"})
      break
  return rows


def run_pool(fn, jobs, n_jobs: int) -> list:
  """`fn` over `jobs`, each in a process of its own: a world dropped in a
  reused one is held until the cycle collector runs."""
  with Pool(max(1, n_jobs), maxtasksperchild=1) as pool:
    return pool.map(fn, jobs, chunksize=1)


def show(row: dict) -> str:
  if "error" in row:
    return f"  chest {row['task']['chest']} try {row['attempt']}: ERROR {row['error']}"
  v, own = row["verdict"], row["own"]
  word = "--" if own["ok"] is None else ("ok" if own["ok"] else "NO")
  return (f"  chest {row['task']['chest']:>3} try {row['attempt']}: "
          f"{'PASSED' if v['ok'] else 'failed'} in {row['simS']:.1f} sim s"
          f"{' (cut)' if row['cut'] else ''}, its own word {word} ({own['why']}) -- {v['reason']}")


def read(into: str) -> None:
  """A kept record, as the per-task record, one regime at a time."""
  rows = [json.loads(line) for line in open(Path(into) / "rows.jsonl")]
  kept = q.from_attempts(rows)
  regimes = sorted({r.run for r in kept})
  for regime in regimes:
    mine = [r for r in kept if r.run == regime]
    rec = q.per_task_record(mine)
    print(f"THE PER-TASK RECORD, {regime}: {rec['tasks']} chests, {rec['n']} attempts")
    for n, w in rec["within"].items():
      if w.get("n"):
        print(f"  passed within {n}: {w['k']} of {w['n']} = {w['share']:.2f} "
              f"[{w['lo']:.2f}, {w['hi']:.2f}]" + (f", {w['unflown']} unflown" if w["unflown"] else ""))
    print(f"  experience: {rec['experience']}")
    print(f"  help: {rec['help']}")
    print(f"  its own test against the verdict: {rec['ownTest']}")
    if rec["errors"]:
      print(f"  {rec['errors']} attempts with no verdict (they raised): no try, their "
            "tasks unflown from there")
    if any(r.data.get("run", {}).get("dirty") for r in mine):
      print("  ⚠ flown from a tree with changes its commit does not have")
    if any(r.data["heldOut"] for r in mine):
      seeds = sorted({str((r.data.get("run") or {}).get("seed")) for r in mine})
      print(f"  held out, by seed {', '.join(seeds)}: weaker than a sealed pool -- whoever "
            "built the generator knows its family")
    for r in mine:
      if r.subject != "passed":
        print(show(r.data))
    print("  what it flatters (#493):")
    for line in at.flatters(mine[0].data.get("task", {}).get("grader", "lid")):
      print(f"    - {line}")
    metrics = [(r.data.get("verdict") or {}).get("metrics") or {} for r in mine]
    tops = [m.get("topDeg") for m in metrics]
    downs = [m.get("downRadS") for m in metrics]
    if all(v is not None for v in tops + downs):
      print(f"  the lid's top {min(tops):.1f}-{max(tops):.1f} deg; fastest down while "
            f"open {min(downs):.3f}-{max(downs):.3f} rad/s")


# ---- --drops: what the constants were read off ----------------------------------

BARE = """<mujoco model="bare"><option integrator="implicitfast"/>
<worldbody><geom name="floor" type="plane" size="3 3 0.1"/></worldbody></mujoco>"""
DROP_DEG = (5.5, 6.0, 7.0, 10.0, 30.0, 57.3, 68.8)


def _chest_world(k: int):
  spec = mujoco.MjSpec.from_string(BARE)
  ch.attach_chest(spec, ch.draw(k), (0.0, 0.0), 0.0, tags=False)
  m = spec.compile()
  return m, mujoco.MjData(m)


def let_fall(k: int, deg: float, seconds: float = 3.0) -> dict:
  """Chest k's lid let go from rest at `deg`, its handle hanging: the
  fastest it closes while open (above `lid.SHUT_RAD`), the speed it lands
  at, when it is shut and when caught."""
  m, d = _chest_world(k)
  lid_ = ch.draw(k)
  h = int(m.jnt_qposadr[m.joint("chest_hinge").id])
  v = int(m.jnt_dofadr[m.joint("chest_hinge").id])
  p = int(m.jnt_qposadr[m.joint("chest_pin").id])
  chest = ch.Chest(m, d, lid_)
  for _ in range(1500):                         # the handle's rest, shut
    chest.sense(m, d)
    mujoco.mj_step(m, d)
  rest = float(d.qpos[p])
  mujoco.mj_resetData(m, d)
  d.qpos[h], d.qpos[p] = math.radians(deg), rest - math.radians(deg)
  mujoco.mj_forward(m, d)
  chest = ch.Chest(m, d, lid_)
  out = {"fastest": 0.0, "lands": None, "shut": None, "caught": None}
  for _ in range(int(seconds / m.opt.timestep)):
    chest.sense(m, d)
    mujoco.mj_step(m, d)
    a, w = float(d.qpos[h]), float(d.qvel[v])
    if a > lid.SHUT_RAD:
      out["fastest"] = max(out["fastest"], -w)
    if out["lands"] is None and a <= 0.002:
      out["lands"] = -w
    if out["shut"] is None and a < lid.SHUT_RAD:
      out["shut"] = float(d.time)
    if out["caught"] is None and a < ch.CAUGHT_ON:
      out["caught"] = float(d.time)
  return out


def drops(n: int = 128) -> None:
  print(f"THE {n} DRAWN LIDS LET FALL FROM REST, their handles hanging (the grader's "
        f"line: no faster than {lid.LOWER_RAD_S} rad/s while above "
        f"{math.degrees(lid.SHUT_RAD):.0f} deg):")
  for deg in DROP_DEG:
    got = [let_fall(k, deg) for k in range(n)]
    f = np.array([g["fastest"] for g in got])
    shut = [g["caught"] for g in got if g["caught"] is not None]
    caught = (f"caught {min(shut):.3f}-{max(shut):.3f} s after ({len(shut)} of {n})"
              if shut else "none caught")
    propped = sum(g["shut"] is None for g in got)
    print(f"  from {deg:5.1f} deg: fastest while open {f.min():.2f}-{f.max():.2f} rad/s "
          f"(median {np.median(f):.2f}); {caught}"
          + (f"; {propped} never under {math.degrees(lid.SHUT_RAD):.0f} deg, propped on "
             "their handles" if propped else ""))


# ---- --view and --strip ------------------------------------------------------------

class ViewerClosed(Exception):
  pass


def _pace(viewer, data, model):
  """A step hook: the viewer handed a frame every 10 steps, the sim held to
  real time; the window closed ends the attempt."""
  start = []

  def hook():
    if not viewer.is_running():
      raise ViewerClosed
    if round(data.time / model.opt.timestep) % 10:
      return
    viewer.sync()
    now = time.time()
    if not start:
      start.append((now, data.time))
    ahead = (data.time - start[0][1]) - (now - start[0][0])
    if ahead > 0:
      time.sleep(ahead)
  return hook


def view(k: int, who: str = "humanoid") -> None:
  from mujoco import viewer as mj_viewer
  skill, info, where = SKILLS[who]
  world = at.set_out(at.Task(chest=k, **where))
  print(f"an attempt at chest {k} (at its counter: the reference opens the lid and sets it "
        "down); the grader's verdict is printed. Close the window to quit.")
  with mj_viewer.launch_passive(world.model, world.data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = world.model.body("pluggybot").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 120.0, -20.0
    world.body.step_hooks.append(_pace(viewer, world.data, world.model))
    try:
      row = at.attempt_in(world, skill, info=info | build())
      print(show(row))
      while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
    except ViewerClosed:
      pass
  world.close()


def strip(k: int, out: str = "attempts.png", who: str = "humanoid") -> None:
  from PIL import Image, ImageDraw
  skill_, info, where = SKILLS[who]
  world = at.set_out(at.Task(chest=k, **where))
  m, d = world.model, world.data
  r = mujoco.Renderer(m, 300, 400)
  tiles = []
  x, y, yaw = (float(v) for v in at_chest(k))
  hinge = int(m.jnt_qposadr[m.joint("chest_hinge").id])
  # the chest stands on its counter, and the robot tall beside it
  low, far = 0.95, 2.4

  def shoot(label):
    # From the chest's side, where the lid's angle reads, between the chest
    # and the robot; the headlight brighter for this tile alone -- the
    # robot's own cameras render in the house's light.
    sx, sy, _ = world.start
    cx, cy = x + math.cos(yaw) * ch.BOX_D / 2, y + math.sin(yaw) * ch.BOX_D / 2
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55 * cx + 0.45 * sx, 0.55 * cy + 0.45 * sy, low]
    cam.distance, cam.azimuth, cam.elevation = far, math.degrees(yaw) + 90.0, -22.0
    light = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.35, 0.55
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = light
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(img)

  shoot("set out: the lid shut")
  opened = []

  def watch_open():
    if not opened and float(d.qpos[hinge]) >= lid.OPEN_RAD:
      opened.append(True)
      shoot(f"opened past {math.degrees(lid.OPEN_RAD):.0f} deg")
  world.body.step_hooks.append(watch_open)

  def skill(body):
    def routine():
      got = yield from skill_(body)
      shoot("the call: the skill has ended")
      return got
    return routine()
  row = at.attempt_in(world, skill, info=info)
  shoot(f"{lid.HOLD_S:.0f} s on: {'PASSED' if row['verdict']['ok'] else 'failed'}")
  print(show(row))
  r.close()
  world.close()
  img = Image.new("RGB", (400 * len(tiles), 300), (255, 255, 255))
  for i, tile in enumerate(tiles):
    img.paste(tile, (400 * i, 0))
  img.save(out)
  print(f"wrote {out}")


def record(k: int, out: str, who: str = "humanoid") -> None:
  """The reference's attempt at chest `k` recorded as a session streams one
  (`session.attempt.fly`), the scene and its textures beside it: a world of
  its own the site replays (#530)."""
  from pluggybot.session import attempt as sa
  from pluggybot.telemetry import scene as ts
  from pluggybot.telemetry.recorder import TelemetryRecorder
  skill, info, _ = SKILLS[who]
  flight = sa.fly(sa.task_of(k), lambda m, d, **kw: TelemetryRecorder(m, d, out, **kw),
                  skill=skill, info=info | build())
  print(show(flight.row))
  where = Path(out).parent
  (where / f"scene.{sa.WORLD}.json").write_text(json.dumps(flight.scene, indent=1) + "\n")
  ts.export_textures(flight.model, where / "textures")
  print(f"wrote {out}, and the scene of {sa.WORLD} and its textures beside it")


def at_chest(k: int):
  from pluggybot.evaluation import counter as ct
  return ct.set_out(k).chest


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--body", choices=sorted(SKILLS), default="humanoid")
  ap.add_argument("--chests", default="0-31")
  ap.add_argument("--held-out", type=int, default=None, metavar="N")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--attempts", type=int, default=1)
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--into", default=None)
  ap.add_argument("--read", default=None, metavar="DIR")
  ap.add_argument("--drops", action="store_true")
  ap.add_argument("--view", nargs="?", type=int, const=0, default=None, metavar="K")
  ap.add_argument("--strip", nargs="?", type=int, const=0, default=None, metavar="K")
  ap.add_argument("--out", default="attempts.png", help="--strip's file")
  ap.add_argument("--record", default=None, metavar="PATH")
  ap.add_argument("--chest", type=int, default=0, help="--record's chest")
  args = ap.parse_args(argv)
  if args.record:
    record(args.chest, args.record, args.body)
    return
  if args.read:
    read(args.read)
    return
  if args.drops:
    drops()
    return
  if args.view is not None:
    view(args.view, args.body)
    return
  if args.strip is not None:
    strip(args.strip, args.out, args.body)
    return
  info = SKILLS[args.body][1] | build()
  if args.held_out is not None:
    if info["dirty"] and args.seed is None:
      ap.error("a held-out draw is a frozen build's: commit the tree first (or name --seed)")
    seed = args.seed if args.seed is not None else at.seed_of(info["commit"])
    ks = at.held_out(seed, args.held_out)
    info["seed"] = seed
    print(f"held out: {args.held_out} chests by seed {seed}: {ks}")
  else:
    ks = chests(args.chests)
  if args.into:
    Path(args.into).mkdir(parents=True, exist_ok=True)
  done = run_pool(fly, [(k, args.attempts, info, args.body) for k in ks], args.jobs)
  rows = [r for got in done for r in got]
  for r in rows:
    print(show(r))
  if args.into:
    with open(Path(args.into) / "rows.jsonl", "w") as f:
      for r in rows:
        f.write(json.dumps(r) + "\n")
    read(args.into)


if __name__ == "__main__":
  main()
