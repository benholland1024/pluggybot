#!/usr/bin/env python
"""Is the world the same world twice? (issue #110)

Evaluation.md §1 rests on "nothing in the world is random", and the first
committed `scripted` series broke it: five identical days gave three
trajectories. This spike finds WHERE they part. It flies the scripted
mission N times in separate processes and, in each, hashes:

  step    qpos + qvel + ctrl every TRACE_EVERY_S of sim time
  tag     each TagDetector.detect: the rendered camera image, and the decode
  lidar   each Lidar.scan: the ranges
  say     every narration line

then aligns the traces and reports the first step whose state differs and
which perception input changed FIRST in the window before it -- a camera
image that differs is the GPU, a decode that differs on an identical image
is the detector, a scan that differs is the raycast, and nothing differing
is the physics itself.

    MUJOCO_GL=egl uv run python scripts/determinism_spike.py --runs 3 --sim-s 1500
    ... --sequential            # one at a time, the contention hypothesis
    ... --nthreads 1            # the AprilTag detector single-threaded
    ... --compare DIR           # re-read traces without flying
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TRACE_EVERY_S = 0.5


def _h(a) -> str:
  import numpy as np
  return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---- the child ---------------------------------------------------------------


def child(cfg: dict, trace_path: Path) -> None:
  import numpy as np

  from pluggybot.lifecycle import run_demo
  from pluggybot.perception import lidar as lidar_mod
  from pluggybot.rack import tags

  out = open(trace_path, "w")

  def log(**row):
    out.write(json.dumps(row) + "\n")
    out.flush()

  if cfg.get("nthreads"):
    from pupil_apriltags import Detector
    tags._DETECTOR = Detector(families=tags.FAMILY, nthreads=int(cfg["nthreads"]),
                              quad_decimate=1.0)

  real_detect = tags.TagDetector.detect

  def detect(self, data):
    # the same render the real method does, hashed before the decode
    self.renderer.update_scene(data, camera=self.camera_name)
    rgb = self.renderer.render()
    img = _h(rgb)
    found = real_detect(self, data)      # renders again: identical scene
    log(k="tag", t=round(float(data.time), 4), cam=self.camera_name, img=img,
        det=hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()[:16],
        n=len(found))
    return found

  tags.TagDetector.detect = detect

  real_scan = lidar_mod.Lidar.scan

  def scan(self, data):
    angles, ranges = real_scan(self, data)
    log(k="lidar", t=round(float(data.time), 4), h=_h(ranges), n=int(len(ranges)))
    return angles, ranges

  lidar_mod.Lidar.scan = scan

  state = {"next": 0.0}

  def on_ready(life):
    d = life.data

    def step():
      if d.time >= state["next"]:
        state["next"] = d.time + TRACE_EVERY_S
        log(k="step", t=round(float(d.time), 4),
            q=_h(np.concatenate([d.qpos, d.qvel])), c=_h(d.ctrl),
            wh=round(life.battery.energy_wh, 6), s=life.state)
    life.mission.step_hooks.append(step)
    life.say_hooks.append(lambda t, msg: log(k="say", t=round(float(t), 3), msg=msg))

  st = Path(tempfile.mkdtemp(prefix="pluggy-det-"))
  t0 = time.time()
  r = run_demo(view=False, realtime=False, world=cfg["world"], pack=cfg["pack"],
               errand=cfg.get("errand", "draw"), max_sim_time=float(cfg["simS"]),
               tasks=True, metabolism=True, overseer=False,
               thoughts_root=str(st / "thoughts"), ledger_state=str(st / "ledger.json"),
               board_state=str(st / "boards.json"), tasks_state=str(st / "tasks.json"),
               journal_state=str(st / "journal.json"), spend_state=str(st / "spend.json"),
               on_ready=on_ready)
  log(k="end", t=round(float(r["sim_time"]), 3), wall=round(time.time() - t0, 1),
      battery=r["battery"], charge_cycles=r["charge_cycles"])
  out.close()


# ---- the comparison ----------------------------------------------------------


def load(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines()
          if line.strip()]


def first_divergence(a: list[dict], b: list[dict]) -> dict:
  """Where two traces part, and what moved first."""
  sa = [r for r in a if r["k"] == "step"]
  sb = [r for r in b if r["k"] == "step"]
  n = min(len(sa), len(sb))
  div = next((i for i in range(n) if (sa[i]["t"], sa[i]["q"], sa[i]["c"])
              != (sb[i]["t"], sb[i]["q"], sb[i]["c"])), None)
  if div is None:
    return {"diverged": False, "stepsCompared": n, "endA": a[-1], "endB": b[-1]}
  t_ok = sa[div - 1]["t"] if div else 0.0
  t_bad = sa[div]["t"]
  # every perception event in the window, paired in order per kind
  def window(tr, kind):
    return [r for r in tr if r["k"] == kind and t_ok < r["t"] <= t_bad + 1e-9]
  report = {"diverged": True, "lastSameT": t_ok, "firstDiffT": t_bad,
            "stateA": sa[div]["s"], "stateB": sb[div]["s"],
            "ctrlDiffers": sa[div]["c"] != sb[div]["c"], "window": {}}
  for kind in ("tag", "lidar"):
    wa, wb = window(a, kind), window(b, kind)
    diffs = []
    for i, (x, y) in enumerate(zip(wa, wb)):
      if kind == "tag":
        same_img, same_det = x["img"] == y["img"], x["det"] == y["det"]
        if not (same_img and same_det):
          diffs.append({"t": x["t"], "cam": x["cam"], "imageSame": same_img,
                        "decodeSame": same_det, "nA": x["n"], "nB": y["n"]})
      elif x["h"] != y["h"]:
        diffs.append({"t": x["t"], "nA": x["n"], "nB": y["n"]})
    report["window"][kind] = {"events": len(wa), "eventsB": len(wb), "differ": diffs}
  # the narration just before, from both, for a human
  report["saysA"] = [r["msg"][:90] for r in a if r["k"] == "say" and t_ok - 30 < r["t"] <= t_bad]
  report["saysB"] = [r["msg"][:90] for r in b if r["k"] == "say" and t_ok - 30 < r["t"] <= t_bad]
  return report


def first_perception_difference(a: list[dict], b: list[dict]) -> dict | None:
  """The very first tag/lidar event that differs, anywhere -- earlier than
  the state divergence when perception drifts before it changes a command."""
  for kind in ("tag", "lidar"):
    ea = [r for r in a if r["k"] == kind]
    eb = [r for r in b if r["k"] == kind]
    for x, y in zip(ea, eb):
      key = ("img", "det") if kind == "tag" else ("h",)
      if any(x[k] != y[k] for k in key):
        return {"kind": kind, "t": x["t"], **({"imageSame": x["img"] == y["img"],
                                                "decodeSame": x["det"] == y["det"],
                                                "cam": x["cam"]} if kind == "tag" else {})}
  return None


def compare(traces: list[Path]) -> None:
  runs = [load(p) for p in traces]
  for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
      rep = first_divergence(runs[i], runs[j])
      print(f"\n{traces[i].name} vs {traces[j].name}:")
      if not rep["diverged"]:
        print(f"  IDENTICAL over {rep['stepsCompared']} state samples; "
              f"ends {rep['endA']['t']} / {rep['endB']['t']} s")
        continue
      print(f"  state diverges between t={rep['lastSameT']} and t={rep['firstDiffT']} "
            f"(states {rep['stateA']} / {rep['stateB']}, ctrl differs: {rep['ctrlDiffers']})")
      for kind, w in rep["window"].items():
        print(f"  {kind}: {w['events']} event(s) in the window, {len(w['differ'])} differ"
              + (f" -> {w['differ'][:3]}" if w["differ"] else ""))
      first = first_perception_difference(runs[i], runs[j])
      print(f"  first perception difference anywhere: {first}")
      for tag, says in (("A", rep["saysA"]), ("B", rep["saysB"])):
        for m in says[-4:]:
          print(f"    {tag}: {m}")


# ---- the parent --------------------------------------------------------------


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--child", nargs=2, metavar=("CFG", "TRACE"))
  ap.add_argument("--runs", type=int, default=2)
  ap.add_argument("--sim-s", type=float, default=1500.0)
  ap.add_argument("--world", default="home")
  ap.add_argument("--pack", default="hosting")
  ap.add_argument("--errand", default="draw", help="carry for room_hub")
  ap.add_argument("--nthreads", type=int, default=None,
                  help="AprilTag detector threads (the shipped detector uses 2)")
  ap.add_argument("--sequential", action="store_true")
  ap.add_argument("--out", default=None)
  ap.add_argument("--compare", default=None, metavar="DIR")
  args = ap.parse_args()
  if args.child:
    child(json.loads(Path(args.child[0]).read_text()), Path(args.child[1]))
    return 0
  if args.compare:
    compare(sorted(Path(args.compare).glob("*.trace.jsonl")))
    return 0
  out = Path(args.out or tempfile.mkdtemp(prefix="pluggy-det-spike-"))
  out.mkdir(parents=True, exist_ok=True)
  cfg = {"world": args.world, "pack": args.pack, "simS": args.sim_s,
         "errand": args.errand, "nthreads": args.nthreads}
  cfg_path = out / "config.json"
  cfg_path.write_text(json.dumps(cfg))
  env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
  procs = []
  for i in range(args.runs):
    trace = out / f"run{i}.trace.jsonl"
    p = subprocess.Popen([sys.executable, __file__, "--child", str(cfg_path), str(trace)],
                         env=env, stdout=open(out / f"run{i}.log", "w"),
                         stderr=subprocess.STDOUT)
    print(f"started run{i} (pid {p.pid})", flush=True)
    if args.sequential:
      p.wait()
    else:
      procs.append(p)
  for p in procs:
    p.wait()
  print(f"traces in {out}")
  compare(sorted(out.glob("*.trace.jsonl")))
  return 0


if __name__ == "__main__":
  sys.exit(main())
