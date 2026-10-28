"""The hands' capability check (issue #525; docs/Hands.md, "The capability
check"): for each generic job -- the chest's lid, a drawer, a cupboard's
door by its lever, a cube picked and placed on a mat -- the procedure we
wrote in the hands' vocabulary (`evaluation.hands_check`), flown at the
job's set-outs (`evaluation.counter`) and graded by the job's own grader
(`challenge/<job>.py`); the bar (`BAR` of `SET_OUTS` a job) was written
before it flew. Each flight is one attempt (`evaluation.attempts`): the
world built afresh, the procedure run inside the job's time, the call, the
hold, one verdict.

  --job J[,J,..]       lid, drawer, door, place (default: all four)
  --set-outs A-B|A,..  the set-outs (default: the check's, 0-7)
  --held-out N         N set-outs nobody has flown instead, by the frozen
                       build's seed (`attempts.seed_of`; a dirty tree is
                       refused), or `--seed`: a reference over a practice's
                       held-out set-outs (#535)
  --jobs N             flights in parallel, each in a process of its own
  --into DIR           the record: rows.jsonl, one row a flight
  --read DIR           a kept record: each job's passes against the bar, and
                       every failure's verdict and the verbs that failed
  --source JOB         the job's procedure, compiled: its source, its
                       length against the cap and its verb calls
  --view JOB [K]       a flight at set-out K (0) in the MuJoCo viewer, at
                       real time, the verdict printed; MUJOCO_GL left off.
                       JOB may be a scene (`hands_check.SCENES`): `until`,
                       the lid's lift stopped on its hand's height (#554);
                       `watch`, on where its watch puts the knob (stage 2)
  --strip JOB [K]      its filmstrip, a tile every few seconds, the call and
                       the hold's end: hands_check.png, or `--out`

Usage:
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
    env MUJOCO_GL=egl uv run python scripts/hands_check.py --jobs 3 --into DIR
  uv run python scripts/hands_check.py --read DIR
  uv run python scripts/hands_check.py --view door 3
  uv run python scripts/hands_check.py --view until
  uv run python scripts/hands_check.py --view watch 1
  MUJOCO_GL=egl uv run python scripts/hands_check.py --strip lid 0
"""

import os

# One BLAS thread before numpy loads (quad_spike.py: a policy's small
# products doubled a step on six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import mujoco  # noqa: E402

from pluggybot.evaluation import attempts as at  # noqa: E402
from pluggybot.evaluation import counter as ct  # noqa: E402
from pluggybot.evaluation import hands_check as hc  # noqa: E402
from pluggybot.evaluation import practice as ep  # noqa: E402
from pluggybot.evaluation import qualities as q  # noqa: E402

from attempts import ViewerClosed, _pace, build, chests, run_pool  # noqa: E402


def fly(job) -> dict:
  """One flight: the row, or a row saying what it raised (`error`)."""
  name, k, info = job
  try:
    return hc.fly(name, k, info=info)
  except Exception as e:
    traceback.print_exc()
    return {"task": {"chest": k, "grader": name}, "run": info,
            "error": f"{type(e).__name__}: {e}"}


def show(row: dict) -> str:
  t = row["task"]
  if "error" in row:
    return f"  {t['grader']:>6} {t['chest']:>2}: ERROR {row['error']}"
  v = row["verdict"]
  failed = [f"{s['verb']} ({s.get('why') or s.get('reason', '')})"
            for s in row.get("ran", {}).get("steps", []) if not s.get("ok")]
  return (f"  {t['grader']:>6} {t['chest']:>2}: {'PASSED' if v['ok'] else 'failed'} in "
          f"{row['simS']:.1f} sim s{' (cut)' if row['cut'] else ''} -- {v['reason']}"
          + (f"\n{'':14}verbs that failed: {'; '.join(failed)}" if failed and not v["ok"] else ""))


def report(rows: list[dict]) -> None:
  tally = hc.tally(rows)
  print(f"THE CHECK (#525): a job passes at {hc.BAR} set-outs of {len(hc.SET_OUTS)}")
  for job, t in tally.items():
    if not t["flown"]:
      continue
    word = {True: "MET", False: "NOT MET", None: "not every set-out flown"}[t["met"]]
    print(f"  {job:>6}: passed {len(t['passed'])} of {len(t['flown'])} "
          f"({', '.join(map(str, t['passed'])) or 'none'}) -- {word}")
  for job in hc.JOBS:
    mine = [r for r in rows if r["task"].get("grader") == job]
    got = q.share_interval([r["verdict"]["ok"] if "verdict" in r else None for r in mine])
    if got is None:
      continue
    took = sorted(r["simS"] for r in mine if "verdict" in r and r["verdict"]["ok"])
    seeds = sorted({str((r.get("run") or {}).get("seed")) for r in mine if r["task"]["chest"]
                    >= at.HELD_OUT_FROM})
    print(f"  {job:>6}, every set-out flown: passed {got['k']} of {got['n']} "
          f"[{got['lo']:.2f}, {got['hi']:.2f}]"
          + (f", held out by seed {', '.join(seeds)}" if seeds else "")
          + (f"; its passes took {took[0]:.2f}-{took[-1]:.2f} sim s" if took else ""))
  errors = [r for r in rows if "error" in r]
  if errors:
    print(f"  {len(errors)} flights raised: no verdict, no try")
  for r in rows:
    if "error" in r or not r["verdict"]["ok"]:
      print(show(r))
  if any((r.get("run") or {}).get("dirty") for r in rows):
    print("  ⚠ flown from a tree with changes its commit does not have")


def read(into: str) -> None:
  report([json.loads(line) for line in open(Path(into) / "rows.jsonl")])


def source(job: str) -> None:
  from pluggybot.procedure.lang import MAX_SOURCE_CHARS
  proc = hc.compiled(job)
  print(hc.JOBS[job])
  print(f"{len(hc.JOBS[job])} characters of {MAX_SOURCE_CHARS}; {proc.verbs} verb calls "
        f"written; budget {proc.steps_budget} steps, {proc.budget_s:.0f} s")


# ---- --view and --strip ------------------------------------------------------------

#: Where each job's prop stands at set-out k (x, y, its turn): what a
#: camera looks at -- and how far a filmstrip's camera stands back, m (the
#: cube is 26 mm).
PROPS = {"lid": lambda k: ct.set_out(k).chest, "drawer": lambda k: ct.drawer_set_out(k).cabinet,
         "door": lambda k: ct.door_set_out(k).cupboard, "place": lambda k: ct.cube_set_out(k).cube}
BACK = {"lid": 1.5, "drawer": 1.5, "door": 1.5, "place": 0.9}


def _task(job: str, k: int) -> at.Task:
  """`job`'s task at set-out `k` -- a scene's, its job's (`hc.SCENES`)."""
  return at.Task(chest=k, body="humanoid", mount="counter",
                 grader=hc.SCENES[job][0] if job in hc.SCENES else job)


def _prop(job: str) -> str:
  return hc.SCENES[job][0] if job in hc.SCENES else job


def _skill(job: str, out: dict):
  return ep.procedure_skill(hc.compiled(job), hc.facts(), out)


def view(job: str, k: int) -> None:
  from mujoco import viewer as mj_viewer
  world = at.set_out(_task(job, k))
  x, y, _ = PROPS[_prop(job)](k)
  print(f"the {job} at set-out {k}, our procedure in the hands' vocabulary; the grader's "
        "verdict is printed. Close the window to quit.")
  with mj_viewer.launch_passive(world.model, world.data) as viewer:
    viewer.cam.lookat[:] = [x, y, ct.COUNTER_H + 0.2]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 135.0, -20.0
    world.body.step_hooks.append(_pace(viewer, world.data, world.model))
    try:
      out: dict = {}
      row = at.attempt_in(world, _skill(job, out))
      print(show(row | {"ran": {}}))
      for s_ in (out.get("ran") or {}).get("steps", []):
        if s_.get("until"):
          print(f"  line {s_['line']} {s_['verb']} "
                + ("stopped on its reading" if s_.get("why") == "until" else "its stop condition")
                + f": {s_['until']}")
      while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
    except ViewerClosed:
      pass
  world.close()


#: A filmstrip's tiles: one at the set-out, then one this often, s, while
#: the procedure runs, the call and the hold's end -- at most this many.
EVERY_S, TILES = 5.0, 8


def strip(job: str, k: int, out: str = "hands_check.png") -> None:
  from PIL import Image, ImageDraw
  world = at.set_out(_task(job, k))
  m, d = world.model, world.data
  r = mujoco.Renderer(m, 300, 400)
  tiles = []
  x, y, yaw = (float(v) for v in PROPS[_prop(job)](k))
  sx, sy, _ = world.start

  def shoot(label):
    # From the side, between the prop and the robot; the headlight brighter
    # for this tile alone -- the robot's own cameras render in the house's light.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.6 * x + 0.4 * sx, 0.6 * y + 0.4 * sy, ct.COUNTER_H + 0.15]
    cam.distance, cam.azimuth, cam.elevation = BACK[_prop(job)], math.degrees(yaw) + 115.0, -25.0
    light = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.35, 0.55
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = light
    # ...and what a watch reads, where one runs (#554's stage 2)
    w = world.body.sighting()
    if w is not None:
      label += (f"\nwatch: tag {w['tag']} {'in view' if w['seen'] else 'not in view'}"
                + (f", last seen {d.time - w['last']['at']:.1f} s ago" if w["last"] else ""))
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(img)

  shoot(f"the {job}, set-out {k}")
  t0 = float(d.time)
  # a hand's move stopped on its reading (#554) is a tile of its own, and so
  # is the end of its hold: a scene's point
  stops: list[float] = []

  def every():
    rep = world.body.reach.report.get("left") or {}
    if rep.get("why") == "until" and not stops:
      stops.append(float(d.time))
      shoot(f"stopped on its reading, {d.time - t0:.1f} s")
    elif len(stops) == 1 and d.time - stops[0] >= 1.9:
      stops.append(float(d.time))
      shoot(f"held there, {d.time - t0:.1f} s")
    elif len(tiles) < TILES - 2 and d.time - t0 >= EVERY_S * (len(tiles) - len(stops)):
      shoot(f"{d.time - t0:.0f} s")
  world.body.step_hooks.append(every)

  def skill(body):
    def routine():
      got = yield from _skill(job, {})(body)
      world.body.step_hooks.remove(every)
      shoot(f"the call, {d.time - t0:.0f} s")
      return got
    return routine()
  row = at.attempt_in(world, skill)
  shoot(f"{row['holdS']:.0f} s on: {'PASSED' if row['verdict']['ok'] else 'failed'}")
  print(show(row | {"ran": {}}))
  r.close()
  world.close()
  img = Image.new("RGB", (400 * min(4, len(tiles)), 300 * math.ceil(len(tiles) / 4)),
                  (255, 255, 255))
  for i, tile in enumerate(tiles):
    img.paste(tile, (400 * (i % 4), 300 * (i // 4)))
  img.save(out)
  print(f"wrote {out}")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--job", default=",".join(hc.JOBS))
  ap.add_argument("--set-outs", default=f"{hc.SET_OUTS[0]}-{hc.SET_OUTS[-1]}")
  ap.add_argument("--held-out", type=int, default=None, metavar="N")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--into", default=None)
  ap.add_argument("--read", default=None, metavar="DIR")
  ap.add_argument("--source", choices=sorted(hc.JOBS), default=None)
  ap.add_argument("--view", nargs="+", default=None, metavar="JOB [K]")
  ap.add_argument("--strip", nargs="+", default=None, metavar="JOB [K]")
  ap.add_argument("--out", default="hands_check.png", help="--strip's file")
  args = ap.parse_args(argv)
  if args.read:
    read(args.read)
    return
  if args.source:
    source(args.source)
    return
  for flag in ("view", "strip"):
    got = getattr(args, flag)
    if got is not None:
      if got[0] not in (*hc.JOBS, *hc.SCENES) or len(got) > 2:
        ap.error(f"--{flag} JOB [K]: JOB one of {', '.join((*hc.JOBS, *hc.SCENES))}")
      k = int(got[1]) if len(got) > 1 else 0
      view(got[0], k) if flag == "view" else strip(got[0], k, args.out)
      return
  names = [j for j in args.job.split(",") if j]
  bad = [j for j in names if j not in hc.JOBS]
  if bad:
    ap.error(f"no job {', '.join(bad)}: {', '.join(hc.JOBS)}")
  info = build()
  ks = chests(args.set_outs)
  if args.held_out is not None:
    if info["dirty"] and args.seed is None:
      ap.error("a held-out draw is a frozen build's: commit the tree first (or name --seed)")
    info["seed"] = args.seed if args.seed is not None else at.seed_of(info["commit"])
    ks = at.held_out(info["seed"], args.held_out)
    print(f"held out: {args.held_out} set-outs by seed {info['seed']}: {ks}")
  todo = [(j, k, info) for j in names for k in ks]
  if args.into:
    Path(args.into).mkdir(parents=True, exist_ok=True)
  rows = run_pool(fly, todo, args.jobs)
  if args.into:
    with open(Path(args.into) / "rows.jsonl", "w") as f:
      for row in rows:
        f.write(json.dumps(row) + "\n")
  report(rows)


if __name__ == "__main__":
  main()
