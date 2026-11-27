#!/usr/bin/env python
"""Is the world the same world twice? (issue #110)

Evaluation.md §1 rests on "nothing in the world is random", and the first
committed `scripted` series broke it: five identical days gave three
trajectories. This spike finds WHERE they part. It flies the scripted
mission N times in separate processes and, in each, hashes:

  step    qpos + qvel + ctrl every TRACE_EVERY_S of sim time
  tag     each TagDetector.detect: the rendered camera image, and the decode
  lidar   each Lidar.scan_split and Lidar.band: the bearings and ranges,
          room and peers
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
                                # the PAIR's day, with a fall and a death
                                # arranged (`--fall-at`, `--drain-at`), the
                                # whole world hashed
    ... --resume-at 400         # issue #345: the day flown straight through,
                                # against the same day saved at the first
                                # idle moment past t=400 and carried on from
                                # the save in a new process -- identical
                                # after the restore point, or a restart
                                # changes the world
    ... --resume-at 400 --in-step
                                # ...saved inside the hooks of the first step
                                # past t=400 whose scan is still to come, as
                                # the keeper saves (#749), and with nothing in
                                # flight that a restart ends (`in_flight`,
                                # #780): the steps it passed over, said

The world is the humanoids' house (`home_humanoid`, #559).
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


#: The day loop's routine that drives the one it wraps by hand, in its local
#: `routine` (`HubLifecycle._until_stood_up_routine`): what `in_flight` walks
#: through where a stack has no `gi_yieldfrom`.
HAND_DRIVEN = "_until_stood_up_routine"


def in_flight(life, day) -> list[str]:
  """WHAT A RESTART ENDS (#780), at a step: the routines the day loop `day`
  (`HubLifecycle.begin`'s) is inside, outermost first, where one is the
  loop's own -- its explore, a charge trip, an errand, a decision, a walk off
  the station -- with the body's routine it is in after them
  (`explore_routine > drive_to_routine`); else none, the loop in its
  stand-by, which a restored loop holds again but from a fresh 5 s (SimNotes,
  "A save in the middle of a walk"). ⚠ Before the loop's first pass the day's
  start is in flight, which this cannot tell (`Flight` can)."""
  names, owned = [], False
  g = getattr(day, "gi_yieldfrom", None)
  while g is not None and getattr(g, "gi_frame", None) is not None:
    mine = g.gi_frame.f_locals.get("self") is life
    if owned and not mine:
      names.append(g.gi_code.co_name)            # ...and the body's routine it is in
      break
    owned = owned or mine
    names.append(g.gi_code.co_name)
    # ...through the one routine that drives another by hand
    g = (g.gi_frame.f_locals.get("routine")
         if g.gi_yieldfrom is None and g.gi_code.co_name == HAND_DRIVEN else g.gi_yieldfrom)
  if not owned:
    return []
  # ...the hand driver's own name only where what it drives could not be read: never none
  return [n for n in names if n != HAND_DRIVEN] or names


class Flight:
  """What a restart would end, read at any step of one robot's day: hooked
  on its lifecycle before `run`, it keeps the day routine `begin` makes,
  whether the day loop has passed its top yet, every hold its body makes
  (`holds`), and the steps `clean` was asked at with something in flight
  (`passed`, the first of them `first`)."""

  def __init__(self, life) -> None:
    import weakref
    self.life, self.day, self.looped = life, None, False
    self.passed, self.first = 0, None
    self.holds = weakref.WeakSet()
    real_begin, real_hold = life.begin, life.body.hold_routine

    def begin(*a, **kw):
      self.day = real_begin(*a, **kw)
      return self.day

    def hold_routine(*a, **kw):
      held = real_hold(*a, **kw)
      self.holds.add(held)
      return held
    life.begin, life.body.hold_routine = begin, hold_routine
    life.at_loop_top.append(self._top)

  def _top(self) -> None:
    self.looped = True

  def ending(self) -> list[str]:
    """`in_flight` -- or, before the loop's first pass, the day's start,
    which a restored day skips (its look round, #345) -- and none only where
    the day loop is in a hold its body made (`holds`), its stand-by: any
    other routine of the body's it is in, by its name."""
    if not self.looped:
      return ["the day's start"]
    names = in_flight(self.life, self.day)
    if names:
      return names
    g = getattr(self.day, "gi_yieldfrom", None)
    return [] if g is not None and g in self.holds else [getattr(g, "__name__", "_day_routine")]

  def clean(self, t: float) -> bool:
    """Whether a save at this step, at sim time `t`, leaves nothing in flight
    that a restart ends (`ending`); one that does not is counted, the first
    kept with what was in flight."""
    ending = self.ending()
    if ending:
      self.passed += 1
      if self.first is None:
        self.first = {"t": round(float(t), 4), "inFlight": ending}
    return not ending


def saved_words(row: dict, in_step: bool, at: float) -> str:
  """Where the saving arm saved past `at` (its `saved` row), and the steps it
  passed over with something in flight (`Flight.clean`) -- or that it saved
  nowhere (its `nosave` row)."""
  first = row.get("firstInFlight")
  over = ("" if not first else
          f" from t={first['t']} it passed over {row['passedOver']} steps the keeper "
          f"would have saved in, the first with {' > '.join(first['inFlight'])} in "
          "flight, which a restart ends by design (the errand in flight ends, #780)")
  if row["k"] == "nosave" and not in_step:
    return f"NOT SAVED: the day loop never passed its top from t={at} to the day's end"
  if row["k"] == "nosave" and not first:
    return (f"NOT SAVED: no step from t={at} to the day's end was one the keeper saves "
            "in (owed, its scan due, nothing mid-move)")
  if row["k"] == "nosave":
    return f"NOT SAVED: no step from t={at} to the day's end had nothing in flight;{over}"
  where = "inside the hooks of the step" if in_step else "at the day loop's pass"
  return f"saved {where} at t={row['t']}" + (f", the first with nothing in flight:{over}"
                                             if over else "")


def unsaved_words(row: dict | None, in_step: bool, at: float, out) -> str:
  """The line a check whose saving arm saved nowhere ends with: why (its
  `nosave` row, `saved_words`), or that it said nothing, and where the
  traces are."""
  return (("NOT SAVED: the saving arm said neither where it saved nor why not"
           if row is None else saved_words(row, in_step, at)) + f"; traces in {out}")


def branched_apart(rep: dict, a: list[dict], b: list[dict]) -> tuple[str, float] | None:
  """Where two days part as #780's limit parts them, the branch and how far
  apart they took it, s; else None: at their first difference one day is
  still in the state both were in and the other has left it, and the first
  takes the same branch within one stand-by slice after (`WAIT_FOR_WORK_S`,
  and a sample's `TRACE_EVERY_S`)."""
  from pluggybot.lifecycle import WAIT_FOR_WORK_S
  sa = {r["t"]: r["s"] for r in a if r["k"] == "step"}
  sb = {r["t"]: r["s"] for r in b if r["k"] == "step"}
  t, t0 = rep["firstDiffT"], rep["lastSameT"]
  was = sa.get(t0)
  if was is None or sb.get(t0) != was:
    return None
  for still, gone in ((sa, sb), (sb, sa)):
    if still.get(t) == was and gone.get(t) not in (None, was):
      branch = gone[t]
      later = [u for u, s in sorted(still.items())
               if t < u <= t + WAIT_FOR_WORK_S + TRACE_EVERY_S and s == branch]
      return (branch, round(later[0] - t, 3)) if later else None
  return None


def verdict(rep: dict, sensed: dict | None, in_step: bool,
            a: list[dict] = (), b: list[dict] = ()) -> tuple[str, int]:
  """The check's last line and its exit code, off `first_divergence` and
  `first_perception_difference` over the two days' traces `a`, `b`. ⚠ A
  save in a step lands in the stand-by, which a restored loop holds again
  from a fresh 5 s, so the loop's next branch may come at another instant:
  where the two days part so (`branched_apart`) the line says it is #780's
  limit; a save at the pass keeps its passes' times."""
  if not rep["diverged"] and sensed is None:
    return (f"IDENTICAL after the restore over {rep['stepsCompared']} state samples "
            "and every look and scan", 0)
  if not rep["diverged"]:
    return (f"SENSED OTHERWISE: the state IDENTICAL over {rep['stepsCompared']} samples, "
            f"but the first look or scan that differs is {sensed}", 1)
  line = (f"DIVERGED: same until t={rep['lastSameT']}, differs at {rep['firstDiffT']} "
          f"(states {rep['stateA']} / {rep['stateB']})")
  apart = branched_apart(rep, a, b) if in_step else None
  if apart is not None:
    line += (f" -- both days took the branch to {apart[0]}, {apart[1]} s apart, inside one "
             "stand-by slice: a restored stand-by holds a fresh 5 s (#780's limit)")
  return line, 1


# ---- the child ---------------------------------------------------------------


def child(cfg: dict, trace_path: Path) -> None:
  import numpy as np

  from pluggybot.lifecycle import run_demo
  from pluggybot.perception import lidar as lidar_mod
  from pluggybot.perception import tags

  out = open(trace_path, "w")

  def log(**row):
    out.write(json.dumps(row) + "\n")
    out.flush()

  if cfg.get("nthreads"):
    tags._DETECTOR = tags.kept_detector(families=tags.FAMILY, nthreads=int(cfg["nthreads"]),
                                        quad_decimate=1.0)

  real_detect = tags.TagDetector.detect

  def detect(self, data):
    # the same render the real method does, hashed before the decode
    img = _h(self.frame(data))
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
  # ...and `band`, which the humanoid's map is laid from (#581)
  real_band = lidar_mod.Lidar.band

  def band(self, data, *a, **kw):
    s = real_band(self, data, *a, **kw)
    log(k="lidar", t=round(float(data.time), 4),
        h=_h(np.concatenate([s.angles, s.ranges, s.peer_angles, s.peer_ranges])),
        n=int(len(s.ranges)), peers=int(len(s.peer_ranges)))
    return s

  lidar_mod.Lidar.band = band

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
    if cfg.get("saveAt") is not None:
      # the SAVING arm of --resume-at: the world written at the first pass
      # of the day loop past the mark -- where nothing is in flight -- and
      # the day ended there; or, `--in-step`, inside the hooks of the first
      # step past it whose scan is still to come, the keeper's own place
      # (#749: the step's bookkeeping after the hooks is owed there)
      from pluggybot import continuation
      from pluggybot.tick import MissionAborted

      # ...with where its last step began, as the keeper's saves carry it
      # (issue #420): after every step hook, as the keeper's own
      last = continuation.LastStep(life)
      life.body.last_hooks.append(last.hook)

      seen: list = [None]

      def landed() -> bool:
        # ...nor on a step a death or a stand-up landed in, as the keeper
        # (`Keeper.landed`): read every step
        now = (None if life.dead is None else id(life.dead), len(life.resets))
        changed, seen[0] = seen[0] is not None and now != seen[0], now
        return changed

      # ...and in a step, only where nothing a restart ends is in flight
      # (`Flight`, #780)
      flight = state["flight"] = Flight(life)

      def save_here(in_step: bool = False):
        # ...never mid-move, as the served keeper waits it out
        # (`continuation.Keeper.busy`, issue #387)
        # ...and in a step, only one a routine still owes its bookkeeping,
        # its scan due: what the restore must book (#749)
        fresh = in_step and landed()
        mi = life.body.mission
        if (d.time >= float(cfg["saveAt"])
            and life.body.posture not in continuation.MOVING_POSTURES
            and not (in_step and (fresh or not mi._owed or d.time < mi._next_scan
                                  or continuation._mid_move(life)))):
          if in_step and not flight.clean(float(d.time)):
            return
          continuation.write(continuation.capture([life], life.world_fingerprint,
                                                  last_step=last),
                             cfg["worldState"])
          log(k="saved", t=round(float(d.time), 4), inStep=in_step,
              owed=bool(mi._owed) if in_step else None,
              passedOver=flight.passed, firstInFlight=flight.first)
          state["saved"] = True
          raise MissionAborted("saved")
      if cfg.get("inStep"):
        life.body.last_hooks.append(lambda: save_here(in_step=True))
      else:
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
  if cfg.get("saveAt") is not None and not state.get("saved"):
    # ...no step to the day's end had nothing in flight (`in_flight`)
    log(k="nosave", passedOver=state["flight"].passed, firstInFlight=state["flight"].first)
  log(k="end", t=round(float(r["sim_time"]), 3), wall=round(time.time() - t0, 1),
      battery=r["battery"], charge_cycles=r["charge_cycles"])
  out.close()


def _pair_child(cfg: dict, st: Path, log, out) -> None:
  """The PAIR's scripted home day (issue #387), the whole world hashed:
  the first robot starts low and walks to its station, the second is
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
      # ...and when: a look one day took and the other did not shifts the rest
      key = ("t", "img", "det") if kind == "tag" else ("t", "h")
      if any(x[k] != y[k] for k in key):
        return {"kind": kind, "t": x["t"], **({"imageSame": x["img"] == y["img"],
                                                "decodeSame": x["det"] == y["det"],
                                                "cam": x["cam"]} if kind == "tag" else {})}
    if len(ea) != len(eb) and a and b:
      longer = ea if len(ea) > len(eb) else eb
      x = longer[min(len(ea), len(eb))]
      if x["t"] <= min(a[-1]["t"], b[-1]["t"]):
        return {"kind": kind, "t": x["t"], "onlyIn": "A" if longer is ea else "B"}
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


def resume_check(out: Path, cfg: dict, at: float, env: dict, in_step: bool = False) -> int:
  """Issue #345's parity check: straight through, against saved at `at` and
  carried on from the save. Three processes, one after another -- the
  resuming one needs the saving one's files. `in_step`: saved inside the
  hooks of a step whose scan is still to come, as the keeper saves (#749),
  and with nothing in flight that a restart ends (`in_flight`, #780)."""
  state = out / "resumed-state"
  arms = {"straight": {}, "saving": {"saveAt": at, "inStep": in_step},
          "resumed": {"resume": True}}
  row = None
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
    if name == "saving":
      row = next((r for r in load(out / "saving.trace.jsonl")
                  if r["k"] in ("saved", "nosave")), None)
      if row is None or row["k"] == "nosave":
        print(unsaved_words(row, in_step, at, out))
        return 1
  saved = row["t"]
  a = [r for r in load(out / "straight.trace.jsonl") if r["t"] > saved]
  b = [r for r in load(out / "resumed.trace.jsonl") if r["t"] > saved]
  rep = first_divergence(a, b)
  print(f"{saved_words(row, in_step, at)}; traces in {out}")
  # ...and what it sensed: a look the straight day never took moves no
  # joint, and every restore took one until the last look was kept
  line, code = verdict(rep, first_perception_difference(a, b), in_step, a, b)
  print(line)
  return code


def main() -> int:
  from pluggybot.batch import watched
  watched()
  from pluggybot.lifecycle import HUMANOID_HOME, world_for
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--child", nargs=2, metavar=("CFG", "TRACE"))
  ap.add_argument("--runs", type=int, default=2)
  ap.add_argument("--sim-s", type=float, default=1500.0)
  ap.add_argument("--world", choices=(HUMANOID_HOME, "home"), default=HUMANOID_HOME,
                  help="the house with the humanoid in it (#559; `home` names it too)")
  ap.add_argument("--pack", default="hosting")
  ap.add_argument("--errand", default="none",
                  help="the queue at the start (`lifecycle.errands_for`): none")
  ap.add_argument("--battery-fraction", type=float, default=1.0,
                  help="the pack at the start: low, and the day charges")
  ap.add_argument("--no-economy", dest="economy", action="store_false",
                  help="no job offers and no upkeep")
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
  ap.add_argument("--in-step", action="store_true",
                  help="--resume-at saves inside the hooks of the first step past T "
                       "whose scan is still to come, as the keeper does (#749), with "
                       "nothing in flight that a restart ends -- a walk, an errand, "
                       "the explore -- and says what it passed over (#780): the "
                       "stand-by, after the explore, in a world that expects work")
  args = ap.parse_args()
  if args.child:
    child(json.loads(Path(args.child[0]).read_text()), Path(args.child[1]))
    return 0
  if args.compare:
    compare(sorted(Path(args.compare).glob("*.trace.jsonl")))
    return 0
  if args.in_step and args.resume_at is None:
    ap.error("--in-step places --resume-at's save")
  if args.pair and args.resume_at is not None:
    # the pair's child has no save hook: three flights, then no `saved` row
    ap.error("--resume-at flies one robot's day, not --pair")
  if args.in_step and not args.economy:
    # ...a save in a step only in the stand-by (`Flight`, #780), and a day
    # with no work to come never stands by: its loop ends after its explore
    ap.error("--in-step saves in the stand-by, which a day with --no-economy never reaches")
  out = Path(args.out or tempfile.mkdtemp(prefix="pluggy-det-spike-"))
  out.mkdir(parents=True, exist_ok=True)
  world = world_for(args.world)
  cfg = {"world": world, "pack": args.pack, "simS": args.sim_s,
         "errand": args.errand, "nthreads": args.nthreads, "economy": args.economy,
         "batteryFraction": args.battery_fraction}
  if args.pair:
    cfg.update(pair=True, fallAt=args.fall_at,
               drainAt=args.drain_at, restartAfterS=args.restart_after,
               batteryFractions=[args.battery_fraction, 1.0])
  cfg_path = out / "config.json"
  cfg_path.write_text(json.dumps(cfg))
  env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
  if args.resume_at is not None:
    return resume_check(out, cfg, args.resume_at, env, in_step=args.in_step)
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
