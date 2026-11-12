#!/usr/bin/env python3
"""What `step` does with what it is asked (issue #649; SimNotes, "Stepping
toward the work"): the humanoid alone on a bare floor, stood from the boot,
after a 12 cm step, after a reach withdrawn or after a turn, the verb's word
and its report (`step.*`) beside what its soles truly did -- the sim's own
record (`_true_stance`), OUR reading, never a sense it has.

  scripts/step_spike.py --walks            each ask ahead, back, left and
                                           right, 3-30 cm, a few oblique
  scripts/step_spike.py --turns            each turn 1-90 deg, both ways
  scripts/step_spike.py --view [walk|turn] its scene in the viewer, looped
                                           (MUJOCO_GL unset)
  scripts/step_spike.py --strip [walk|turn]  its filmstrip, step_spike.png
  --noises N   keyed noise draws an ask (3)
  --jobs J     processes (4; each flight a process's, a floor world is small)
  --out PATH   the filmstrip's path
"""

import argparse
import json
import math
import multiprocessing as mp
import time

import mujoco

from pluggybot.body import body_for
from pluggybot.humanoid import model as hm

#: The stances a step is asked from: as it boots, after a step of its own,
#: after a reach came home and after a turn on the spot (#649: the same ask
#: from each goes differently, and after a big turn the most).
STARTS = ("boot", "step", "reach", "turn")
WALKS = tuple((v, 0.0) for v in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30))
WALKS = WALKS + tuple((-a, 0.0) for a, _ in WALKS) + tuple((0.0, a) for a, _ in WALKS) \
  + tuple((0.0, -a) for a, _ in WALKS) + ((0.05, 0.05), (0.05, -0.05), (0.15, 0.05))
TURNS_DEG = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 90)
#: The scenes: what each asks, in turn (ahead, left, turn rad), every step
#: from where the last one left it.
SCENES = {
  "walk": ((0.03, 0.0, 0.0), (0.06, 0.0, 0.0), (0.10, 0.0, 0.0),
           (0.0, 0.03, 0.0), (0.0, 0.06, 0.0), (-0.03, 0.0, 0.0), (-0.10, 0.0, 0.0)),
  "turn": ((0.0, 0.0, math.radians(3.0)), (0.0, 0.0, math.radians(-3.0)),
           (0.0, 0.0, math.radians(7.0)), (0.0, 0.0, math.radians(-15.0)),
           (0.0, 0.0, math.radians(45.0))),
}
#: The viewer's camera: from the robot's front-left, a little above.
CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION = 3.2, 140.0, -12.0


def floor_body(noise: str = "step-spike"):
  """The robot on a bare floor, standing as its keyframe poses it."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data, body_for(model, data, noise=noise)


def start_from(body, name: str) -> None:
  """Stood still, then as `STARTS` says."""
  body.run(body.hold_routine(1.0))
  if name == "step":
    body.run(body.step_routine(0.12, 0.0, 0.0))
    body.run(body.hold_routine(0.5))
  elif name == "reach":
    body.run(body.reach_routine("left", body.to_world((0.35, 0.2, 0.9)), None, patience=15.0))
    body.run(body.release_routine("left"))
    body.run(body.hold_routine(0.5))
  elif name == "turn":
    body.run(body.step_routine(0.0, 0.0, 0.5))
    body.run(body.hold_routine(0.5))


def moved(a, b) -> tuple[float, float, float]:
  """How the stance went from `a` to `b`, in the frame of `a` (m, m, deg)."""
  c, s = math.cos(a[2]), math.sin(a[2])
  dx, dy = b[0] - a[0], b[1] - a[1]
  return (c * dx + s * dy, -s * dx + c * dy,
          math.degrees(math.remainder(b[2] - a[2], 2 * math.pi)))


def fly(job: tuple) -> dict:
  """One step asked of a body stood as `start` says: its word, its report
  and what its soles truly did."""
  ahead, left, turn, start, noise = job
  _, _, body = floor_body(noise)
  start_from(body, start)
  a = body._true_stance()
  rep = body.run(body.step_routine(ahead, left, turn))
  fwd, side, tturn = moved(a, body._true_stance())
  body.close()
  return {"ask": [ahead, left, math.degrees(turn)], "start": start, "noise": noise,
          "why": rep["why"], "s": rep["seconds"], "fwd": fwd, "side": side, "tturn": tturn,
          "rdx": rep["dx"], "rdy": rep["dy"], "rturn": math.degrees(rep["turn"]),
          "fell": body.mission.falls}


def flights(asks, noises: int, jobs: int) -> list[dict]:
  todo = [(*ask, s, f"step-{k}") for ask in asks for s in STARTS for k in range(noises)]
  with mp.get_context("spawn").Pool(max(1, jobs), maxtasksperchild=1) as pool:
    return pool.map(fly, todo, chunksize=1)


def _along(r, xk, yk) -> float:
  ahead, left = r["ask"][:2]
  return (r[xk] * ahead + r[yk] * left) / math.hypot(ahead, left)


def read_walks(rows: list[dict]) -> None:
  """A row an ask: the truth along it (cm), what its legs reckoned, its
  word, and how many fell."""
  asks = sorted({tuple(r["ask"][:2]) for r in rows},
                key=lambda k: (abs(k[0]) + abs(k[1]), k[1] != 0, k))
  print(f"{'ask cm':>9} {'true mean [min,max]':>24} {'reckoned [min,max]':>24} "
        f"{'stepped/too short/other':>24} {'falls':>6}")
  for k in asks:
    rs = [r for r in rows if tuple(r["ask"][:2]) == k]
    t = [100 * _along(r, "fwd", "side") for r in rs]
    r_ = [100 * _along(r, "rdx", "rdy") for r in rs]
    why = [r["why"] for r in rs]
    stepped, short = why.count("stepped"), why.count("too short")
    print(f"{100 * k[0]:5.0f},{100 * k[1]:3.0f} {sum(t) / len(t):7.1f} [{min(t):5.1f},{max(t):5.1f}]   "
          f"{sum(r_) / len(r_):7.1f} [{min(r_):5.1f},{max(r_):5.1f}]   "
          f"{stepped:6}/{short}/{len(why) - stepped - short:<6} {sum(r['fell'] for r in rs):6}")
  lies = [r for r in rows if r["why"] == "stepped" and _along(r, "fwd", "side") < 0.01]
  print(f"read `stepped` having truly moved under 1 cm: {len(lies)} of {len(rows)}")


def read_turns(rows: list[dict]) -> None:
  """A row a turn, each way: the stance's turn, its report's, the time."""
  print(f"{'asked':>6} {'left: true (min..max), err max':>31} {'right':>31} {'secs':>5}")
  for d in TURNS_DEG:
    cells = []
    for sign in (1, -1):
      rs = [r for r in rows if round(r["ask"][2]) == sign * d]
      t = [r["tturn"] * sign for r in rs]
      cells.append(f"{sum(t) / len(t):6.1f} [{min(t):6.1f},{max(t):6.1f}] e<={max(abs(x - d) for x in t):4.1f}")
    rs = [r for r in rows if abs(round(r["ask"][2])) == d]
    print(f"{d:6} {cells[0]:>31} {cells[1]:>31} {sum(r['s'] for r in rs) / len(rs):5.2f}")
  err = [abs(r["rturn"] - r["tturn"]) for r in rows]
  print(f"its report against the truth: at most {max(err):.2f} deg; falls {sum(r['fell'] for r in rows)}")


def drive(body, routine):
  """A routine stepped a physics step at a time, yielding between (its
  stepper, as `tick.run` steps it): what it returned."""
  try:
    command = next(routine)
    while True:
      body.stepper.step(command)
      yield None
      command = routine.send(None)
  except StopIteration as done:
    return done.value


def scene(name: str, body):
  """Scene `name` once: yields a line for each step's word and truth."""
  yield from drive(body, body.hold_routine(1.0))
  for ask in SCENES[name]:
    a = body._true_stance()
    rep = yield from drive(body, body.step_routine(*ask))
    fwd, side, tturn = moved(a, body._true_stance())
    label = f"ahead {100 * ask[0]:.0f} left {100 * ask[1]:.0f} cm, turn {math.degrees(ask[2]):.0f} deg"
    yield (f"{label}: {rep['why']}; moved {100 * fwd:.1f} / {100 * side:.1f} cm, turned "
           f"{tturn:.1f} deg (it reckoned {100 * rep['dx']:.1f} / {100 * rep['dy']:.1f}, "
           f"{math.degrees(rep['turn']):.1f})")
    yield from drive(body, body.hold_routine(0.5))


def view(name: str) -> None:
  from mujoco import viewer as mj_viewer
  print(f"the `{name}` scene: the humanoid on a bare floor, each ask's word and what its feet "
        "truly did. Close the window to quit.")
  model, data, body = floor_body()
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(hm.ROBOT_ROOT).id
    viewer.cam.distance, viewer.cam.azimuth = CAM_DISTANCE, CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION
    while viewer.is_running():
      # again and again, a fresh robot each pass
      mujoco.mj_resetDataKeyframe(model, data, 0)
      mujoco.mj_forward(model, data)
      body = body_for(model, data, noise="step-spike")
      wall = time.time() - data.time
      for line in scene(name, body):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)


def strip(name: str, out: str) -> None:
  """A tile at each step's end, its word and truth on it."""
  from PIL import Image, ImageDraw
  model, data, body = floor_body()
  r = mujoco.Renderer(model, 300, 400)
  cam = mujoco.MjvCamera()
  tiles = []
  for line in scene(name, body):
    if line:
      print(f"  -> {line}")
      cam.lookat[:] = data.xpos[model.body(hm.ROBOT_ROOT).id] - [0.0, 0.0, 0.1]
      cam.distance, cam.azimuth, cam.elevation = CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      ImageDraw.Draw(img).text((8, 8), line[:64], fill=(255, 255, 255))
      tiles.append(img)
  r.close()
  body.close()
  sheet = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    sheet.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  sheet.save(out)
  print(f"wrote {out}")


def main() -> None:
  from pluggybot.batch import watched
  watched()
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--walks", action="store_true")
  ap.add_argument("--turns", action="store_true")
  ap.add_argument("--view", nargs="?", const="walk", choices=tuple(SCENES), default=None)
  ap.add_argument("--strip", nargs="?", const="walk", choices=tuple(SCENES), default=None)
  ap.add_argument("--noises", type=int, default=3)
  ap.add_argument("--jobs", type=int, default=4)
  ap.add_argument("--out", default="step_spike.png")
  ap.add_argument("--save", default=None, help="the flights, as JSON")
  args = ap.parse_args()
  if args.view:
    view(args.view)
  elif args.strip:
    strip(args.strip, args.out)
  elif args.walks or args.turns:
    rows = []
    if args.walks:
      rows += flights([(*ask, 0.0) for ask in WALKS], args.noises, args.jobs)
      read_walks([r for r in rows if r["ask"][:2] != [0.0, 0.0]])
    if args.turns:
      turns = [(0.0, 0.0, s * math.radians(d)) for d in TURNS_DEG for s in (1, -1)]
      got = flights(turns, args.noises, args.jobs)
      read_turns(got)
      rows += got
    if args.save:
      json.dump(rows, open(args.save, "w"))
  else:
    ap.print_help()


if __name__ == "__main__":
  main()
