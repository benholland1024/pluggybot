"""The probe from the robot's own senses, flown over seeded set-outs (issue
#466's stage 2; SimNotes, "The probe from the robot's own senses").

Each set-out (`evaluation.probe.set_out(k)`) is drawn chest k in the house's
empty storeroom and the robot standing in front of it, the claw on its fork
-- a demo's world, never the served one. The robot probes it from its own
senses (`legs.probe`), and OUR grading reads the truth: whether it got
through and why not, the geometry it measured against the chest's own, the
lid's angle as its arm tells it, the friction it calibrated, and its record
replayed through the world's chest and through the best-expressible
reference (`evaluation.probe.replay`).

  --n N          set-outs 0..N-1 (default 16); --from K starts at K
  --jobs J       set-outs in parallel, each in a process of its own
  --into DIR     each set-out's row (rows.jsonl) and its probe
                 (probe_K.npz: `Probed.to_wire` in `imagination.worker.pack`'s
                 form, `unpack` reads it back)
  --no-replay    skip the replay

Usage:
  MUJOCO_GL=egl uv run python scripts/probe_chest.py --n 4 --jobs 2
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
    uv run python scripts/probe_chest.py --n 16 --jobs 3 --into DIR
"""

import os

# One BLAS thread before numpy loads (quad_spike.py: a policy's small
# products doubled a step on six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.activity import chest as ch  # noqa: E402
from pluggybot.evaluation import probe as ep  # noqa: E402
from pluggybot.legs import arm as am  # noqa: E402
from pluggybot.legs import probe as pr  # noqa: E402
from pluggybot.legs import rack as rk  # noqa: E402
from pluggybot.rack.coupling import PEG_ABOVE_BODY  # noqa: E402

#: The coupling's holding capacitor, s: a seat open longer loses the tool's
#: power (#469's criterion).
SEAT_HOLD_S = 0.2
#: Held is both pads within this of the knob, m (`tools.claw.HELD_GAP_M`).
HELD_GAP_M = 0.0005


class TruthRows:
  """OUR record of the chest and the claw each physics step, by step: the
  lid's and the handle's angles, the claw seated, both pads on the knob."""

  def __init__(self, m, d) -> None:
    self.m, self.d = m, d
    self.hinge = int(m.jnt_qposadr[m.joint("chest_hinge").id])
    self.pin = int(m.jnt_qposadr[m.joint("chest_pin").id])
    self.knob = m.geom("chest_knob").id
    self.pads = [m.geom(p).id for p in rk.CLAW_PADS]
    self.rows: dict[int, tuple] = {}

  def __call__(self) -> None:
    m, d = self.m, self.d
    step = int(round(float(d.time) / float(m.opt.timestep)))
    gripped = all(mujoco.mj_geomDistance(m, d, p, self.knob, 0.002, None) < HELD_GAP_M
                  for p in self.pads)
    self.rows[step] = (float(d.qpos[self.hinge]), float(d.qpos[self.pin]),
                       bool(rk.tool_power(m, d, "module_claw")["powered"]), gripped)

  def over(self, t0: float, n: int) -> np.ndarray:
    """The rows of a record that began at sim time `t0`, n of them."""
    first = int(round(t0 / float(self.m.opt.timestep))) + 1
    return np.array([self.rows[first + i] for i in range(n)], dtype=float)


class Watched(pr.Probe):
  """The probe, and at each of its measures OUR snapshot of the truth: the
  robot's true pose and its belief, and where the knob truly hung."""

  def __init__(self, mis) -> None:
    super().__init__(mis)
    self.snaps: list[dict] = []

  def measure_routine(self, keep: bool = False):
    got = yield from super().measure_routine(keep)
    m, d, mis = self.mis.model, self.mis.data, self.mis
    self.snaps.append({"keep": keep, "true": mis.true_pose(), "belief": mis.pose,
                       "knob": d.geom_xpos[m.geom("chest_knob").id].copy(),
                       "guess": got[0] if isinstance(got[0], pr.Guess) else None})
    return got

  def _probe_routine(self, hand, path, level, out):
    # the record begins as this does: where the robot truly lay then, and
    # where the knob truly hung
    m, d = self.mis.model, self.mis.data
    self.start_true = self.mis.true_pose()
    self.start_knob = d.geom_xpos[m.geom("chest_knob").id].copy()
    return (yield from super()._probe_routine(hand, path, level, out))


def mount(body, x: float, y: float, heading: float) -> None:
  """Standing at (x, y, heading), the claw seated on the fork at the carry
  pose: a placement, not a swap (`scripts/mechanism_spike.py`'s)."""
  mis, m, d = body.mission, body.model, body.data
  body.start_at(x, y, heading)
  mis.arm.aim(*am.CARRY_Q)
  body.run(body.hold_routine(2.0))
  seat = m.site(mis.handle.el("arm_seat")).id
  adr = m.jnt_qposadr[m.joint("module_claw_free").id]
  dof = m.jnt_dofadr[m.joint("module_claw_free").id]
  rot = d.xmat[mis.root].reshape(3, 3)
  _, _, yaw = mis.true_pose()
  peg = d.site_xpos[seat] + rot @ np.array([0, 0, mis.arm_spec.fork.seat_rise() + 0.0003])
  d.qpos[adr:adr + 3] = peg - [0, 0, PEG_ABOVE_BODY]
  d.qpos[adr + 3:adr + 7] = [math.cos((yaw + math.pi) / 2), 0, 0, math.sin((yaw + math.pi) / 2)]
  d.qvel[dof:dof + 6] = 0.0
  mujoco.mj_forward(m, d)
  mis.carry("module_claw")
  body.run(body.hold_routine(1.0))


def _longest(flags: np.ndarray, dt: float) -> float:
  """The longest run of False, s."""
  run = best = 0
  for f in flags:
    run = 0 if f else run + 1
    best = max(best, run)
  return best * dt


def fly(args) -> dict:
  """One set-out: flown, graded, and its probe kept (`into`)."""
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  k, into, do_replay = args
  so = ep.set_out(k)
  spec = lw.home_spec(first_at=so.start[:2])
  ch.attach_chest(spec, so.lid, so.chest[:2], so.chest[2])
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  body = qb.QuadBody(m, d, realtime=False, grid_bounds=home.GRID_BOUNDS)
  mis = body.mission
  chest = ch.Chest(m, d, so.lid)
  truth = TruthRows(m, d)
  mis.step_hooks += [lambda: chest.sense(m, d), truth]
  mount(body, *so.start)
  probe = Watched(mis)
  t0 = time.time()
  out = body.run(probe.routine())
  row = {"k": k, "ok": out.ok, "why": out.why, "simS": round(float(d.time), 1),
         "wallS": round(time.time() - t0, 1), "catchN": round(so.lid.catch_n, 2),
         "friction": None if out.friction is None else [round(v, 4) for v in out.friction],
         "tries": len(out.log.get("tries", [])), "heldDownN": out.log.get("heldDownN"),
         "opened": bool(chest.flags["opened"])}
  stance = [s for s in probe.snaps if s["keep"] and s["guess"] is not None]
  if stance:
    s = stance[-1]
    row["geometry"] = ep.geometry(s["guess"], s["belief"], s["true"], ep.truth(so.chest),
                                  s["knob"])
  if out.record is not None and out.frame is not None:
    rec = out.record
    rows = truth.over(out.t0, rec.n)
    lid, seated, gripped = rows[:, 0], rows[:, 2].astype(bool), rows[:, 3].astype(bool)
    sweeps = slice(out.phases["take"][1], rec.n)
    b, t = rec.start.pose, probe.start_true
    row["beliefMm"] = [round(1000 * (b[0] - t[0]), 1), round(1000 * (b[1] - t[1]), 1),
                       round(math.degrees(ep._wrap(b[2] - t[2])), 2)]
    # what it planned off: its measures moved by its look at the box lying
    row["planned"] = ep.geometry(out.guess, b, t, ep.truth(so.chest), probe.start_knob)
    row["refound"] = out.log["tries"][-1].get("refound")
    row.update({
      "rows": rec.n, "lidTopDeg": round(math.degrees(float(lid.max())), 1),
      "released": bool((lid > ch.CAUGHT_OFF).any()),
      "seatOpenS": round(_longest(seated, rec.dt), 3),
      "grippedPct": round(100.0 * float(gripped[sweeps].mean()), 1) if rec.n > sweeps.start else 0.0,
      "clouds": [len(c) for c in rec.depth], "detections": len(rec.detections)})
    row["gotThrough"] = bool(row["opened"] and row["released"] and row["seatOpenS"] < SEAT_HOLD_S
                             and row["grippedPct"] >= 95.0)
    est = pr.lid_from_arm(out, mis.arm_spec)
    err = np.degrees(est[sweeps] - lid[sweeps])
    row["lidFromArmDeg"] = {"rms": round(float(np.sqrt((err ** 2).mean())), 2),
                            "worst": round(float(np.abs(err).max()), 2)}
    if do_replay and row["gotThrough"]:
      pin = float(rows[0, 1])
      # where the chest stands in the robot's map at the record's start:
      # the truth through the true pose there and the belief
      chest_map = ep.in_map(so.chest, probe.start_true, rec.start.pose)
      gaps = ep.replay(rec, out.phases, so.lid, chest_map, pin, lid)
      row["replay"] = {name: {g.phase: [None if g.force_rms is None else round(g.force_rms, 3),
                                        None if g.force_max is None else round(g.force_max, 3)]
                              for g in gs} for name, gs in gaps.items()}
  else:
    row["gotThrough"] = False
  if into is not None:
    from pluggybot.imagination.worker import pack
    head, arrays = out.to_wire()
    (Path(into) / f"probe_{k}.npz").write_bytes(pack(head, arrays))
    if out.record is not None:
      # OURS, beside it: the truth along the record, row by row (the lid's
      # and the handle's angles, seated, gripped) and where it all stood
      np.savez(Path(into) / f"truth_{k}.npz", rows=truth.over(out.t0, out.record.n),
               chest=np.array(so.chest), start_true=np.array(probe.start_true),
               knob=np.array(probe.start_knob))
  return row


def run_pool(fn, jobs, n_jobs: int) -> list:
  """`fn` over `jobs`, each in a process of its own: a world dropped in a
  reused one is held until the cycle collector runs, and 300 flights through
  six reused workers took the dev box into swap (`mechanism_spike.run_pool`,
  2026-10-06)."""
  with Pool(max(1, n_jobs), maxtasksperchild=1) as pool:
    return pool.map(fn, jobs, chunksize=1)


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--n", type=int, default=16)
  ap.add_argument("--from", dest="first", type=int, default=0)
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--into", default=None)
  ap.add_argument("--no-replay", action="store_true")
  args = ap.parse_args(argv)
  if args.into:
    Path(args.into).mkdir(parents=True, exist_ok=True)
  jobs = [(k, args.into, not args.no_replay) for k in range(args.first, args.first + args.n)]
  rows = run_pool(fly, jobs, args.jobs)
  for r in rows:
    print(json.dumps(r))
  if args.into:
    with open(Path(args.into) / "rows.jsonl", "w") as f:
      for r in rows:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
  main()
