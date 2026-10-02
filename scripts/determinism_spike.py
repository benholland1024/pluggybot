#!/usr/bin/env python
"""Is the world the same world twice? (issue #110)

Evaluation.md §1 rests on "nothing in the world is random", and the first
committed `scripted` series broke it: five identical days gave three
trajectories. This spike finds WHERE they part. It flies the scripted
mission N times in separate processes and, in each, hashes:

  step    qpos + qvel + ctrl every TRACE_EVERY_S of sim time
  tag     each TagDetector.detect: the rendered camera image, and the decode
  lidar   each Lidar.scan_split: the bearings and ranges, room and peers
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
    ... --pair --pack demo --battery-fraction 0.4 --sim-s 420
                                # issue #387: the PAIR's day, with a fall and
                                # a death arranged (`--fall-at`,
                                # `--drain-at`), the whole world hashed
    ... --resume-at 400         # issue #345: the day flown straight through,
                                # against the same day saved at the first
                                # idle moment past t=400 and carried on from
                                # the save in a new process -- identical
                                # after the restore point, or a restart
                                # changes the world
    ... --errand none --no-economy --on-fork module_claw --resume-at 20
                                # issue #420: the same, saved between two
                                # failed returns of a tool on the fork
"""

import argparse
import hashlib
import json
import math
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

  # `scan_split`, which `scan` calls too: the mission casts through it
  # (issue #316), so hooking `scan` logged no lidar rows at all.
  real_scan = lidar_mod.Lidar.scan_split

  def scan_split(self, data):
    out = real_scan(self, data)
    log(k="lidar", t=round(float(data.time), 4),
        h=_h(np.concatenate([out[0], out[1], out[2], out[3]])),
        n=int(len(out[1])), peers=int(len(out[3])))
    return out

  lidar_mod.Lidar.scan_split = scan_split

  state = {"next": 0.0}

  def on_ready(life):
    d = life.data

    def step():
      if d.time >= state["next"]:
        # on a FIXED grid of sim time, so a run that started from a save
        # samples the same instants as one that flew straight through
        state["next"] = (math.floor(d.time / TRACE_EVERY_S) + 1) * TRACE_EVERY_S
        log(k="step", t=round(float(d.time), 4),
            q=_h(np.concatenate([d.qpos, d.qvel])), c=_h(d.ctrl),
            wh=round(life.battery.energy_wh, 6), s=life.state)
    life.body.step_hooks.append(step)
    life.say_hooks.append(lambda t, msg: log(k="say", t=round(float(t), 3), msg=msg))
    if cfg.get("onFork") and not cfg.get("resume"):
      life.at_loop_top.append(_seat_on_fork(life, cfg["onFork"], log))
    if cfg.get("saveAt") is not None:
      # the SAVING arm of --resume-at: the world written at the first pass
      # of the day loop past the mark -- where nothing is in flight -- and
      # the day ended there
      from pluggybot import continuation
      from pluggybot.tick import MissionAborted

      # ...with where its last step began, as the keeper's saves carry it
      # (issue #420): hooked last
      last = continuation.LastStep(life)
      life.body.step_hooks.append(last.hook)

      def save_here():
        # ...never mid-move, as the served keeper waits it out
        # (`continuation.Keeper.busy`, issue #387)
        if (d.time >= float(cfg["saveAt"])
            and life.body.posture not in continuation.MOVING_POSTURES):
          continuation.write(continuation.capture([life], life.world_fingerprint,
                                                  last_step=last),
                             cfg["worldState"])
          log(k="saved", t=round(float(d.time), 4))
          raise MissionAborted("saved")
      life.at_loop_top.append(save_here)

  st = Path(cfg.get("stateDir") or tempfile.mkdtemp(prefix="pluggy-det-"))
  if cfg.get("pair"):
    return _pair_child(cfg, st, log, out)
  sim_s = float(cfg["simS"])
  if cfg.get("resume"):
    # the RESUMING arm: carries on from the save, to the same end
    from pluggybot import continuation
    sim_s -= continuation.read(cfg["worldState"]).t
  t0 = time.time()
  economy = cfg.get("economy", True)
  r = run_demo(view=False, realtime=False, world=cfg["world"], pack=cfg["pack"],
               errand=cfg.get("errand", "none"), max_sim_time=sim_s,
               tasks=economy, metabolism=economy, overseer=False,
               battery_fraction=float(cfg.get("batteryFraction", 1.0)),
               thoughts_root=str(st / "thoughts"), ledger_state=str(st / "ledger.json"),
               board_state=str(st / "boards.json"),
               tasks_state=str(st / "tasks.json") if economy else None,
               spend_state=str(st / "spend.json"), on_ready=on_ready,
               world_state=cfg["worldState"] if cfg.get("resume") else None)
  log(k="end", t=round(float(r["sim_time"]), 3), wall=round(time.time() - t0, 1),
      battery=r["battery"], charge_cycles=r["charge_cycles"])
  out.close()


def _seat_on_fork(life, module: str, log):
  """`--on-fork` (issue #420), an at-loop-top hook: at the first pass,
  `module` seated on the fork at the carry pose as a pick leaves it, and
  another tool hung in its bay, so every return of it fails. The economy
  that once put a tool there offers legs no jobs that fetch one."""
  import mujoco
  import numpy as np

  from pluggybot.legs import arm as am
  from pluggybot.legs import rack as rk
  from pluggybot.rack.coupling import PEG_ABOVE_BODY
  done = []

  def seat() -> None:
    if done:
      return
    done.append(True)
    mis, m, d = life.body.mission, life.model, life.data

    def free(name):
      j = m.body(name).jntadr[0]
      return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    (q, v), other = free(module), next(t for t in rk.TOOL_BAYS if t != module)
    (oq, ov) = free(other)
    d.qpos[oq:oq + 7] = m.qpos0[q:q + 7]           # hung where `module` hangs
    d.qvel[ov:ov + 6] = 0.0
    mis.arm.hold_at(*am.CARRY_Q)
    arm = {n: m.jnt_qposadr[m.joint(mis.handle.el(f"arm_{n}")).id]
           for n in ("shoulder", "elbow", "wrist")}
    d.qpos[arm["shoulder"]], d.qpos[arm["elbow"]] = am.CARRY_Q
    d.qpos[arm["wrist"]] = -sum(am.CARRY_Q)
    mujoco.mj_forward(m, d)
    rot = d.xmat[mis.root].reshape(3, 3)
    peg = (d.site_xpos[m.site(mis.handle.el("arm_seat")).id]
           + rot @ np.array([0.0, 0.0, mis.arm_spec.fork.seat_rise() + 0.0003]))
    yaw = math.atan2(rot[1, 0], rot[0, 0]) + math.pi
    d.qpos[q:q + 3] = peg - np.array([0.0, 0.0, PEG_ABOVE_BODY])
    d.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    d.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(m, d)
    mis.carry(module)
    log(k="onFork", t=round(float(d.time), 4), module=module, bayTakenBy=other)
  return seat


def _pair_child(cfg: dict, st: Path, log, out) -> None:
  """The PAIR's scripted home day (issue #387), the whole world hashed:
  the first robot starts low and walks to the dock, the second is
  knocked over and later emptied (`pair.arrange_hazards`), dies `flat` and
  is stood up -- every posture and both halves of a death, twice alike."""
  import numpy as np

  from pluggybot import pair
  from pluggybot.mind.inbox import Inbox

  # an inbox each, as `serve.py` builds them: a dead robot with somebody
  # who could reach in waits for its stand-up rather than ending its day
  lives = pair.build_pair(cfg["world"], pack=cfg["pack"], errands=("none", "none"),
                          inboxes=(Inbox(), Inbox()),
                          mortal=True, restart_after_s=float(cfg["restartAfterS"]),
                          thoughts_root=str(st / "thoughts"),
                          ledger_state=str(st / "ledger.json"))
  for life, frac in zip(lives, cfg["batteryFractions"]):
    life.battery.energy_wh = life.battery.capacity_wh * frac
  done = pair.arrange_hazards(lives, fall_at=float(cfg["fallAt"]),
                              drain_at=float(cfg["drainAt"]))
  d = lives[0].data
  state = {"next": 0.0}

  def step():
    if d.time >= state["next"]:
      state["next"] = (math.floor(d.time / TRACE_EVERY_S) + 1) * TRACE_EVERY_S
      log(k="step", t=round(float(d.time), 4), q=_h(np.concatenate([d.qpos, d.qvel])),
          c=_h(d.ctrl), wh=[round(life.battery.energy_wh, 6) for life in lives],
          s=[life.state for life in lives], p=[life.body.posture for life in lives])
  lives[0].body.step_hooks.append(step)
  for life in lives:
    life.say_hooks.append(lambda t, msg, root=life.root:
                          log(k="say", t=round(float(t), 3), msg=f"{root}: {msg}"))
  t0 = time.time()
  pair.run_pair(lives, max_sim_time=float(cfg["simS"]))
  log(k="end", t=round(float(d.time), 3), wall=round(time.time() - t0, 1),
      battery=[life.battery.fraction for life in lives], fell=done["fell"],
      drained=done["drained"], falls=[life.body.mission.falls for life in lives],
      deaths=[[x["cause"] for x in life.deaths] for life in lives])
  out.close()


# ---- the comparison ----------------------------------------------------------


def load(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines()
          if line.strip()]


def first_divergence(a: list[dict], b: list[dict]) -> dict:
  """Where two traces part, and what moved first. Compared over the sim
  times BOTH sampled: a run carried on from a save starts part way."""
  common = ({r["t"] for r in a if r["k"] == "step"}
            & {r["t"] for r in b if r["k"] == "step"})
  sa = [r for r in a if r["k"] == "step" and r["t"] in common]
  sb = [r for r in b if r["k"] == "step" and r["t"] in common]
  n = min(len(sa), len(sb))
  div = next((i for i in range(n) if (sa[i]["t"], sa[i]["q"], sa[i]["c"])
              != (sb[i]["t"], sb[i]["q"], sb[i]["c"])), None)
  if div is None:
    return {"diverged": False, "stepsCompared": n,
            "endA": a[-1] if a else None, "endB": b[-1] if b else None}
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


def resume_check(out: Path, cfg: dict, at: float, env: dict) -> int:
  """Issue #345's parity check: straight through, against saved at `at` and
  carried on from the save. Three processes, one after another -- the
  resuming one needs the saving one's files."""
  state = out / "resumed-state"
  arms = {"straight": {}, "saving": {"saveAt": at}, "resumed": {"resume": True}}
  for name, extra in arms.items():
    arm = {**cfg, **extra, "stateDir": str(out / f"{name}-state"),
           "worldState": str(state / "world.npz")}
    if name != "straight":
      arm["stateDir"] = str(state)
    path = out / f"{name}.config.json"
    path.write_text(json.dumps(arm))
    print(f"flying {name}", flush=True)
    subprocess.run([sys.executable, __file__, "--child", str(path),
                    str(out / f"{name}.trace.jsonl")],
                   env=env, stdout=open(out / f"{name}.log", "w"),
                   stderr=subprocess.STDOUT, check=True)
  saved = next(r["t"] for r in load(out / "saving.trace.jsonl") if r["k"] == "saved")
  a = [r for r in load(out / "straight.trace.jsonl") if r["t"] > saved]
  b = [r for r in load(out / "resumed.trace.jsonl") if r["t"] > saved]
  rep = first_divergence(a, b)
  print(f"saved at the day loop's pass at t={saved}; traces in {out}")
  if not rep["diverged"]:
    print(f"IDENTICAL after the restore over {rep['stepsCompared']} state samples")
    return 0
  print(f"DIVERGED: same until t={rep['lastSameT']}, differs at {rep['firstDiffT']} "
        f"(states {rep['stateA']} / {rep['stateB']})")
  return 1


def main() -> int:
  from pluggybot.lifecycle import QUAD_HOME, world_for
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--child", nargs=2, metavar=("CFG", "TRACE"))
  ap.add_argument("--runs", type=int, default=2)
  ap.add_argument("--sim-s", type=float, default=1500.0)
  ap.add_argument("--world", choices=(QUAD_HOME, "home"), default=QUAD_HOME,
                  help="the house with the quadruped in it; `home` names it too")
  ap.add_argument("--pack", default="hosting")
  ap.add_argument("--errand", default="none",
                  help="the queue at the start (`lifecycle.errands_for`): "
                       "none, care, care:<act>, feed or shock")
  ap.add_argument("--battery-fraction", type=float, default=1.0,
                  help="the pack at the start: low, and the day charges")
  ap.add_argument("--no-economy", dest="economy", action="store_false",
                  help="no job offers and no upkeep: the first quadruped "
                       "period's shape (issue #387)")
  ap.add_argument("--nthreads", type=int, default=None,
                  help="AprilTag detector threads (the shipped detector uses 2)")
  ap.add_argument("--sequential", action="store_true")
  ap.add_argument("--pair", action="store_true",
                  help="issue #387: the pair's home day, the whole "
                       "world hashed -- the first robot starts low and "
                       "charges, the second is knocked over at --fall-at and "
                       "emptied at --drain-at, dies and is stood up")
  ap.add_argument("--fall-at", type=float, default=40.0)
  ap.add_argument("--drain-at", type=float, default=120.0)
  ap.add_argument("--restart-after", type=float, default=30.0,
                  help="--pair: sim s from a death to the stand-up")
  ap.add_argument("--out", default=None)
  ap.add_argument("--compare", default=None, metavar="DIR")
  ap.add_argument("--resume-at", type=float, default=None, metavar="T",
                  help="issue #345: fly the day straight through AND saved at "
                       "the first idle moment past T then carried on from the "
                       "save in a new process; the two must be identical after")
  ap.add_argument("--on-fork", default=None, metavar="MODULE",
                  help="issue #420: MODULE seated on the fork at the loop's "
                       "first pass, its bay taken by another tool, so every "
                       "return fails and a save lands between two")
  args = ap.parse_args()
  if args.child:
    child(json.loads(Path(args.child[0]).read_text()), Path(args.child[1]))
    return 0
  if args.compare:
    compare(sorted(Path(args.compare).glob("*.trace.jsonl")))
    return 0
  if args.pair and args.resume_at is not None:
    # the pair's child has no save hook: three flights, then no `saved` row
    ap.error("--resume-at flies one robot's day, not --pair")
  out = Path(args.out or tempfile.mkdtemp(prefix="pluggy-det-spike-"))
  out.mkdir(parents=True, exist_ok=True)
  cfg = {"world": world_for(args.world), "pack": args.pack, "simS": args.sim_s,
         "errand": args.errand, "nthreads": args.nthreads, "economy": args.economy,
         "batteryFraction": args.battery_fraction, "onFork": args.on_fork}
  if args.pair:
    cfg.update(pair=True, fallAt=args.fall_at,
               drainAt=args.drain_at, restartAfterS=args.restart_after,
               batteryFractions=[args.battery_fraction, 1.0])
  cfg_path = out / "config.json"
  cfg_path.write_text(json.dumps(cfg))
  env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
  if args.resume_at is not None:
    return resume_check(out, cfg, args.resume_at, env)
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
