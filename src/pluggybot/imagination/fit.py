"""The fitter (issue #466, stage 3): a template's unknowns found by rolling
the world it describes out along a record, and comparing what the arm's
drivers would have read with what they did read. Nothing in it is specific
to a lid: it knows a template (`scene.Template`), a record, which of its
rows to fit, and how two sets of readings come apart at the tool (the
body's `legs.imagined.force_apart`).

THE RESIDUAL is the force at the tool the record's readings and a rollout's
put apart, in `BIN_S` means, both components: what the torques resolve
while both joints turn is 0.04-0.06 N in 0.1 s means (#469).

THE SEARCH, deterministic given the record and the template, in each
unknown's share of its range (0 to 1):
  start   the middle of every range and `SOBOL_POINTS` more spread over them
          (a scrambled Sobol sequence of a fixed seed), the best of them:
          from the middles alone a fit stalled with its lid 2.5 times too
          heavy and a spring it did not have (set-out 0)
  steps   Levenberg-Marquardt: the Jacobian off a rollout an unknown, `STEP`
          of its range away, and each step tried at `DAMPINGS` at once, the
          best kept
  robust  first on a soft-L1 loss (`ROBUST_N`), then on squares from where
          it ended: a catch's release puts a spike in the residual that a
          squared loss chases while the rest of the record fits nothing (the
          same set-out: 0.87 N left on the first sweep up, 0.10 after)
  stop    when `MAX_STALLS` steps running each gain less than `STALL` of the
          cost, or after `MAX_STEPS`
An unknown that moves the residual less than `UNSEEN_N` across its whole
range is UNSEEN: held at its range's middle, and said so. An unknown that
ends at an end of its range is said so too, and which end: the range, or
the structure, may be what is wrong. ⚠ A probe whose world was refused (unstable, or
unbuildable) is tried the other way; refused both ways, its unknown is
held that step and named (`held`), never dropped unsaid (the review).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import Readings
from pluggybot.imagination.scene import Template

#: The residual's bins, s.
BIN_S = 0.1
#: The start: this many points of a Sobol sequence of this seed, beside the
#: middle of every range.
SOBOL_POINTS, SOBOL_SEED = 16, 466
#: The Jacobian's step, a share of an unknown's range.
STEP = 0.01
#: Each step is tried at the damping it had times each of these.
DAMPINGS = (0.1, 1.0, 10.0, 100.0)
#: The damping a fit starts at, and what a step's gets after it is kept.
START_DAMPING, KEEP_DAMPING = 0.01, 0.5
#: The soft-L1 loss's scale, N at the tool in a bin: about the most the
#: torques' own scatter leaves in one (#469: 0.10-0.15 N a reading, 0.04-0.06
#: in 0.1 s means).
ROBUST_N = 0.1
#: A step gaining less than this share of the cost stalls; this many
#: running end a stage; and a fit takes this many steps at most.
STALL, MAX_STALLS, MAX_STEPS = 2e-3, 2, 40
#: An unknown moving the residual less than this across its whole range is
#: unseen, N (RMS over the bins).
UNSEEN_N = 1e-3
#: A share this near 0 or 1 is at its range's end.
AT_END = 1e-3


def residual(sensed: np.ndarray, readings: Readings, rows: slice, dt: float) -> np.ndarray:
  """(bins, 2): what `readings` put on the tool apart from `sensed` (rows of
  `record.SENSED`) over `rows`, in `BIN_S` means."""
  from pluggybot.legs.imagined import force_apart
  f = force_apart(sensed[rows], readings.sensed[rows])
  per = max(1, round(BIN_S / dt))
  n = len(f) // per
  return f[:n * per].reshape(n, per, 2).mean(axis=1)


def rms(bins: np.ndarray) -> float:
  """The RMS of the bins' magnitudes, N."""
  return float(np.sqrt((np.asarray(bins) ** 2).sum(axis=1).mean())) if len(bins) else 0.0


def _cost(bins, robust: bool) -> float:
  if bins is None:
    return math.inf
  m2 = (bins ** 2).sum(axis=1)
  if not robust:
    return float(m2.sum())
  return float((2.0 * ROBUST_N ** 2 * (np.sqrt(1.0 + m2 / ROBUST_N ** 2) - 1.0)).sum())


def _weights(bins) -> np.ndarray:
  """The robust loss as reweighted squares: each bin's square root of its
  soft-L1 weight, for both its components."""
  w = (1.0 + (bins ** 2).sum(axis=1) / ROBUST_N ** 2) ** -0.25
  return np.repeat(w, 2)


@dataclass(frozen=True)
class Fitted:
  """A template fitted to a record: its unknowns' `values` (author's units,
  by name), the fitted world's `readings` and `bins` (`residual`), and how
  it went: the `steps` and `rollouts`, the unknowns `at_end` of a range
  and which (`low` or `high`, by name), those `unseen`, those `held` at the
  last step (no probe of theirs could be read), and each step's (stage,
  cost) in `log`."""
  template: Template
  values: dict[str, float]
  readings: Readings
  bins: np.ndarray
  steps: int
  rollouts: int
  at_end: dict[str, str] = field(default_factory=dict)
  unseen: tuple[str, ...] = ()
  held: tuple[str, ...] = ()
  log: tuple = field(default=())

  @property
  def document(self) -> dict:
    return self.template.fill([self.values[n] for n in self.template.names])

  @property
  def rms(self) -> float:
    return rms(self.bins)

  def as_dict(self) -> dict:
    return {"values": dict(self.values), "rmsN": round(self.rms, 4), "steps": self.steps,
            "rollouts": self.rollouts, "atEnd": dict(self.at_end), "unseen": list(self.unseen),
            "held": list(self.held)}


def _step(share: float, first: bool) -> float | None:
  """A probe's step off `share`: up where there is room, else down; the
  second try the other way, where both had room (None: none left)."""
  up, down = share + STEP <= 1.0, share - STEP >= 0.0
  if first:
    return STEP if up else -STEP
  return -STEP if (up and down) else None


class Unfittable(RuntimeError):
  """A template no rollout of which could be read: every start refused."""


def fit(template: Template, record: Record, rows: slice, pool, *,
        sobol: int = SOBOL_POINTS, max_steps: int = MAX_STEPS) -> Fitted:
  """`template`'s unknowns fitted to `record` over `rows` (the module
  docstring), rolled out in `pool` (`worker.Imaginations`)."""
  if record.sensed is None:
    raise ValueError("a record to fit is one that was flown: it has no readings")
  # what a rollout and its residual read: no depth, no detections
  lean = Record(start=record.start, dt=record.dt, commands=record.commands,
                sensed=record.sensed)
  us = template.unknowns
  n = len(us)
  count = [0]

  def run(xs) -> list:
    """Each point's residual bins, or None where its world was refused."""
    docs = [template.fill([u.at(v) for u, v in zip(us, x)]) for x in xs]
    count[0] += len(docs)
    return [None if isinstance(r, Exception) else r for r in pool.residuals(docs, lean, rows)]

  # the start
  starts = [np.full(n, 0.5)]
  if n and sobol:
    from scipy.stats import qmc
    starts += list(qmc.Sobol(n, scramble=True, seed=SOBOL_SEED).random(sobol))
  got = run(starts)
  robust = True
  costs = [_cost(g, robust) for g in got]
  best = int(np.argmin(costs))
  if not math.isfinite(costs[best]):
    raise Unfittable("no rollout of the template could be read: every start went "
                     "unstable or was refused")
  x, bins, cost = np.array(starts[best], dtype=float), got[best], costs[best]
  log = [("start", cost)]
  active = list(range(n))
  unseen: list[int] = []
  lam, stalls, steps = START_DAMPING, 0, 0
  per_bin = math.sqrt(max(1, bins.size // 2))
  held: set[int] = set()
  while active and steps < max_steps:
    steps += 1
    # the Jacobian, an unknown a rollout: a probe whose world was refused is
    # tried the other way, and one refused both ways holds its unknown this
    # step -- never drops it for the rest of the fit, unsaid
    cols = {}
    for first in (True, False):
      want = [i for i in active if i not in cols and _step(x[i], first) is not None]
      probes = []
      for i in want:
        xi = x.copy()
        xi[i] += _step(x[i], first)
        probes.append(xi)
      for i, xi, g in zip(want, probes, run(probes) if probes else ()):
        if g is not None:
          cols[i] = ((g - bins) / (xi[i] - x[i])).ravel()
    held = {i for i in active if i not in cols}
    for i in [i for i in active if i in cols]:
      if np.linalg.norm(cols[i]) / per_bin < UNSEEN_N:
        unseen.append(i)
    active = [i for i in active if i not in unseen]
    moving = [i for i in active if i in cols]
    if not moving:
      break
    J = np.stack([cols[i] for i in moving], axis=1)
    r = bins.ravel()
    if robust:
      w = _weights(bins)
      J, r = J * w[:, None], r * w
    A, g = J.T @ J, J.T @ r
    tries, steps_at = [], []
    for d in DAMPINGS:
      try:
        dx = -np.linalg.solve(A + lam * d * np.diag(np.diag(A) + 1e-12), g)
      except np.linalg.LinAlgError:
        continue
      xt = x.copy()
      xt[moving] = np.clip(x[moving] + dx, 0.0, 1.0)
      tries.append(xt)
      steps_at.append(lam * d)
    got = run(tries)
    trial = [_cost(t, robust) for t in got]
    j = int(np.argmin(trial)) if trial else -1
    if j >= 0 and trial[j] < cost:
      gain = (cost - trial[j]) / cost
      x, bins, cost = tries[j], got[j], trial[j]
      lam = steps_at[j] * KEEP_DAMPING
      stalls = stalls + 1 if gain < STALL else 0
    else:
      lam *= DAMPINGS[-1]
      stalls += 1
    log.append(("robust" if robust else "squares", cost))
    if stalls >= MAX_STALLS:
      if not robust:
        break
      robust, stalls, lam = False, 0, START_DAMPING
      cost = _cost(bins, robust)
  for i in unseen:
    x[i] = 0.5
  # the fitted world's readings, whole: what it is judged and drawn by
  final = template.fill([u.at(v) for u, v in zip(us, x)])
  readings = pool.rollouts([final], lean)[0]
  count[0] += 1
  if isinstance(readings, Exception):
    raise Unfittable(f"the fitted document's world was refused: {readings}")
  bins = residual(record.sensed, readings, rows, record.dt)
  values = {u.name: u.at(v) for u, v in zip(us, x)}
  at_end = {us[i].name: "low" if x[i] < AT_END else "high" for i in range(n)
            if i not in unseen and (x[i] < AT_END or x[i] > 1.0 - AT_END)}
  return Fitted(template=template, values=values, readings=readings, bins=bins, steps=steps,
                rollouts=count[0], at_end=at_end, unseen=tuple(us[i].name for i in unseen),
                held=tuple(us[i].name for i in sorted(held)), log=tuple(log))
