"""The scene language's own gap, and what a rollout costs (issue #466's stage
1; SimNotes, "The imagination's first world").

No robot flies. For each drawn chest (`activity.chest.draw`), code that
knows the truth writes it in the scene language as closely as the language
allows (`chest.reference_document`), and one probe -- the oracle's, planned
off the true hinge (`evaluation.imagined.oracle_record`) -- is replayed
through the world's chest and through the reference, each beside the
robot's own body: how far apart they behave is the language's own limit,
before any robot senses anything. Each chest is flown with its catch and
without, so the gap splits into the catch's and the rest's.

  --n N         drawn chests 0..N-1 (default 16)
  --jobs J      chests in parallel, each in a process of its own
  --into FILE   the rows as JSON lines too
  --cost        what a rollout costs, ms a sim-second: in this process and
                in the worker (`imagination.worker`), the whole probe,
                `--reps` times

Usage:
  uv run python scripts/imagination_gap.py [--n 16] [--jobs 3]
  uv run python scripts/imagination_gap.py --cost
"""

import os

# One BLAS thread before numpy loads (quad_spike.py: a policy's small
# products doubled a step on six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from pluggybot.activity import chest as ch  # noqa: E402
from pluggybot.evaluation import imagined as im  # noqa: E402


def one(k: int) -> dict:
  """Chest `k`, with its catch and without: each phase's gap."""
  lid = ch.draw(k)
  th = np.array([0.0])
  out = {"k": k, "catchN": round(lid.catch_n, 2),
         "gravityNm": round(float(ch.closing_torque(replace(lid, stiffness=0.0, friction=0.0,
                                                            damping=0.0), th, 0.0)[0]), 3),
         "springNmRad": round(lid.stiffness, 3), "frictionNm": round(lid.friction, 3),
         "dampingNmsRad": round(lid.damping, 3)}
  for label, variant in (("catch", lid), ("none", replace(lid, catch_n=0.0))):
    gap = im.language_gap(variant)
    out[label] = {g.phase: asdict(g) for g in gap["gap"]}
    out[f"{label}Top"] = [round(v, 2) for v in gap["gap"][1].top]
  return out


def table(rows: list[dict]) -> None:
  phases = list(rows[0]["catch"])
  print("chest  catch N  gravity N*m  " + "  ".join(f"{p:>13s}" for p in phases)
        + "   (force at the tool apart, N: RMS of 0.1 s means / largest)")
  for label in ("catch", "none"):
    print(f"-- {'with its catch' if label == 'catch' else 'with no catch'}")
    for r in rows:
      cells = "  ".join(f"{r[label][p]['force_rms']:6.3f}/{r[label][p]['force_max']:5.2f}"
                        for p in phases)
      print(f"k{r['k']:<4d}  {r['catchN']:7.2f}  {r['gravityNm']:11.3f}  {cells}")
    for stat, fn in (("median", np.median), ("largest", np.max)):
      cells = "  ".join(f"{fn([r[label][p]['force_rms'] for r in rows]):6.3f}/"
                        f"{fn([r[label][p]['force_max'] for r in rows]):5.2f}" for p in phases)
      print(f"{stat:7s} {'':7s}  {'':11s}  {cells}")
    lid = "  ".join(f"{np.max([r[label][p]['lid_max'] for r in rows]):13.2f}" for p in phases)
    print(f"{'lid apart, largest (deg)':29s}{lid}")


def cost(reps: int) -> None:
  """A rollout's cost, ms a sim-second, on this machine."""
  from pluggybot.imagination.compile import compile_scene
  from pluggybot.imagination.rollout import SETTLE_S, rollout
  from pluggybot.imagination.scene import parse
  from pluggybot.imagination.worker import Imagination
  lid = ch.draw(0)
  setting, pin = im.place(lid)
  record, _ = im.oracle_record(lid, setting)
  doc = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  sim_s = record.n * record.dt + SETTLE_S
  print(f"{platform.processor() or platform.machine()}, {os.cpu_count()} cpus; "
        f"the oracle's probe: {record.n} rows, {sim_s:.1f} sim s with the settle")
  t0 = time.perf_counter()
  world = compile_scene(parse(doc), carrying="module_claw")
  print(f"parse + compile: {1000 * (time.perf_counter() - t0):.1f} ms")
  runs = {}
  for label, build in (("reference, in this process", lambda: world),
                       ("the world's chest, in this process", lambda: im.truth_world(lid, setting))):
    w = build()
    times = []
    for _ in range(reps):
      t0 = time.perf_counter()
      rollout(w, record)
      times.append(time.perf_counter() - t0)
    runs[label] = times
  t0 = time.perf_counter()
  with Imagination(seed=0) as worker:
    worker.rollout(doc, record)
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
      t0 = time.perf_counter()
      worker.rollout(doc, record)
      times.append(time.perf_counter() - t0)
  runs["reference, in the worker (compiled each time)"] = times
  print(f"the worker's first answer, its start and imports included: {first:.2f} s")
  for label, times in runs.items():
    ms = 1000 * float(np.median(times)) / sim_s
    print(f"{label}: {ms:.1f} ms a sim-second (median of {reps}; "
          f"{1000 * min(times) / sim_s:.1f}-{1000 * max(times) / sim_s:.1f})")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--n", type=int, default=16)
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--into", default=None)
  ap.add_argument("--cost", action="store_true")
  ap.add_argument("--reps", type=int, default=5)
  args = ap.parse_args(argv)
  if args.cost:
    return cost(args.reps)
  # each chest in a process of its own (`mechanism_spike.run_pool`'s rule)
  with Pool(max(1, args.jobs), maxtasksperchild=1) as pool:
    rows = pool.map(one, range(args.n), chunksize=1)
  table(rows)
  if args.into:
    with open(args.into, "w") as f:
      for r in rows:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
  main()
