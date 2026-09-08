"""Many records -> one rollup, per series (issue #106; Evaluation.md §4).

A SERIES is one configuration -- (world, arm, pack, model, label) -- and a rollup
reports each series as distributions over its runs, never as a mean alone.
The one rule with teeth: **a series whose runs read different data files is
two series wearing one name, and this refuses to average them.** Each of
`rewards`, `cadence`, `energy`, `metabolism` and `questions` changes the
regime; a rollup that quietly pooled a 45-points-an-hour week with a
30-points-an-hour one would report a robot that exists in neither.

`current` on a series says whether its hashes are today's data files, so
the committed rollup goes stale -- and the spec fails -- the moment a data
file is edited, which is the cheapest possible place to be told that the
committed results now describe a previous regime.

Issue #117 adds the two halves of "under what conditions": the DEADLINE is
part of the regime as well (no hash catches it, and it caps how much of a
day the model decided at all), and the `label` is part of the series key,
so a quiet box and a loaded one are two series rather than one average of
both. And a run whose fallback rate says the BOX decided too much of its
day is dropped from the survival statistics -- never deleted, and never
silently: see `FALLBACK_LIMIT`.
"""

import json
from collections import Counter
from pathlib import Path

from pluggybot.evaluation.notes import NOTES_NAME
from pluggybot.evaluation.record import (
  LATENCY_PERCENTILES, SCHEMA, data_hashes, dist, problems,
)

ROLLUP_NAME = "rollup.json"

#: The most of a run's decisions that may have come from the SCRIPTED
#: ROTATION before the run stops being a measurement of a model (issue
#: #117; Evaluation.md section 5). Every fallback is `scripted()` deciding,
#: and the rotation never chooses `charge` -- so a `guarded` day at 40 %
#: fallback is two-fifths a `scripted` day wearing the guarded name, and
#: on `autonomous`, where no rail charges either, that is what killed pass
#: 1b's one `flat` death: two timeouts, `fallback:explore`, the street, and
#: zero on the way back.
#:
#: ⚠ A JUDGEMENT CALL, and that is exactly why it is a value the rollup
#: WRITES DOWN (`fallbackLimit`, per series) rather than a comment beside a
#: filter. Whoever disagrees with the number can see which one was applied.
#:
#: The arms want different numbers because a fallback costs them different
#: things. On `guarded` the rails still charge the robot, so a fallback
#: dilutes the result; on `autonomous` nothing else is looking after the
#: pack, so a fallback is the one decision that can end the run. `scripted`
#: has no limit at all: the rotation is not a failure mode there, it is the
#: arm.
FALLBACK_LIMIT: dict[str, float | None] = {
  "scripted": None,
  "guarded": 0.25,
  "autonomous": 0.10,
}


class MixedRegime(ValueError):
  """Two runs in one series read different data files."""


def series_key(record: dict) -> tuple:
  return (record["world"], record["arm"], record["pack"],
          record.get("model") or "none", record.get("label") or "")


def _series(records: list[dict], current) -> dict:
  runs = sorted(records, key=lambda r: r["runId"])
  regimes = {json.dumps(r["dataHashes"], sort_keys=True) for r in runs}
  if len(regimes) > 1:
    key = series_key(runs[0])
    seen = Counter(json.dumps(r["dataHashes"], sort_keys=True) for r in runs)
    detail = "; ".join(f"{n} run(s) on {json.loads(h)}" for h, n in seen.items())
    raise MixedRegime(
      f"refusing to aggregate {key}: its {len(runs)} runs span "
      f"{len(regimes)} regimes (data files + world) -- {detail}. Re-run the series on "
      "one set of data files, or move the older runs to results/archive/")
  hashes = runs[0]["dataHashes"]
  # ...and so is THE DEADLINE (issue #117). It is not a data file, so no
  # hash catches it, and it is the one configuration value that decides
  # how much of a day the model decided at all: pooling an 8 s series with
  # a 20 s one averages two different experiments. Raising it is what this
  # check was written for -- the old runs do not become wrong, they become
  # a different series.
  deadlines = {r["config"].get("deadlineS") for r in runs}
  if len(deadlines) > 1:
    raise MixedRegime(
      f"refusing to aggregate {series_key(runs[0])}: its {len(runs)} runs "
      f"span {len(deadlines)} decision deadlines "
      f"({sorted(d for d in deadlines if d is not None)} s). "
      "A deadline is a cap on how many decisions the model made at all -- "
      "re-fly the series, label the new runs (--label), or move the older "
      "ones to results/archive/")
  arm = runs[0]["arm"]
  limit = FALLBACK_LIMIT.get(arm)

  def excluded_because(r: dict) -> str:
    """Why this run is not a survival data point, or "" (Evaluation.md §5).

    Three exclusions, one shape: the first measured an ADMIN, the second
    measured the BOX's clock, and the third measured the box's LOAD
    through a deadline. None of them is a deletion -- the run stays
    committed, still validates, and the reason travels with it here.
    """
    if r["end"] == "killed":
      return "killed on wall clock"
    if r["interventions"]:
      return f"{len(r['interventions'])} admin intervention(s)"
    rate = r["mind"]["fallbackRate"]
    if limit is not None and rate is not None and rate > limit:
      return (f"fallbackRate {rate:.4g} over the {limit:.4g} limit for the "
              f"{arm} arm: {rate:.0%} of its decisions were the rotation")
    return ""

  excluded = [{"runId": r["runId"], "why": excluded_because(r)} for r in runs
              if excluded_because(r)]
  clean = [r for r in runs if not excluded_because(r)]
  ch = [r["charging"] for r in runs]
  mind = [r["mind"] for r in runs]
  eco = [r["economy"] for r in runs]
  world, arm, pack, model, label = series_key(runs[0])
  return {
    "world": world, "arm": arm, "pack": pack, "model": model,
    # What the box was, as the run itself recorded it: the label it was
    # flown under, the deadline it was held to, and how many sims shared
    # the machine. The pair this issue commits -- one quiet series beside
    # one loaded one -- is only readable because these are here.
    "label": label,
    "deadlineS": runs[0]["config"].get("deadlineS"),
    # WHICH RUNG, where there is a ladder. Not in the series key: a rung is
    # a change to what the model is SHOWN, so two rungs are two series and
    # the label is what keeps them apart -- naming the rung here means a
    # reader can see which one without opening a record.
    "rung": runs[0]["config"].get("rung"),
    "parallel": sorted({r["config"].get("parallel") for r in runs}),
    "n": len(runs), "runIds": [r["runId"] for r in runs],
    "commits": sorted({r["commit"] for r in runs}),
    "dataHashes": dict(hashes),
    # `current` is judged against THIS series' world: the regime is the
    # five data files AND the world model (issue #110).
    "current": ((dict(hashes) == current(world)) if current is not None
                else None),
    "ends": dict(Counter(r["end"] for r in runs)),
    "killed": sum(1 for r in runs if r["end"] == "killed"),
    "withInterventions": sum(1 for r in runs if r["interventions"]),
    # The threshold that was applied, in the artifact rather than in a
    # comment -- a reader who disagrees with it can see the number and the
    # runs it cost (issue #117).
    "fallbackLimit": limit,
    "overFallback": sum(1 for e in excluded if "fallbackRate" in e["why"]),
    "survival": {
      "n": len(clean),
      "excluded": excluded,
      "survivalS": dist([s for r in clean for s in r["survival"]["survivalS"]]),
      "deaths": {"flat": sum(r["survival"]["deaths"]["flat"] for r in clean),
                 "stuck": sum(r["survival"]["deaths"]["stuck"] for r in clean)},
      "minFraction": dist([r["survival"]["minFraction"] for r in clean]),
    },
    "charging": {
      "forced": dist([c["forced"] for c in ch]),
      "deferred": dist([c["deferred"] for c in ch]),
      "voluntaryChosen": dist([c["voluntary"]["chosen"] for c in ch]),
      "voluntaryHonoured": dist([c["voluntary"]["honoured"] for c in ch]),
      "voluntaryFrac": dist([f for c in ch for f in c["voluntary"]["chosenFrac"]]),
      "docked": dist([c["docked"] for c in ch]),
      "anticipationOffer": dist([c["anticipation"]["offer"] for c in ch]),
      "anticipationAction": dist([c["anticipation"]["action"] for c in ch]),
    },
    "mind": {
      "decisions": dist([m["decisions"] for m in mind]),
      "llmCalls": dist([m["llmCalls"] for m in mind]),
      "fallbacks": dist([m["fallbacks"] for m in mind]),
      "fallbackRate": dist([m["fallbackRate"] for m in mind]),
      "fallbackReasons": dict(sum((Counter(m["fallbackReasons"]) for m in mind),
                                  Counter())),
      # Pooled across the series and read at its TAIL, because the
      # deadline is a cap on this one distribution (issue #117).
      "wallS": dist([w for m in mind for w in m["wallS"]["values"]],
                    LATENCY_PERCENTILES),
      "constrained": sorted({str(m.get("constrained")) for m in mind}),
      "longestStreak": dist([m["longestStreak"] for m in mind]),
      "usd": dist([m.get("usd") for m in mind]),
    },
    "whFailed": dist([r["whFailed"] for r in runs]),
    "memory": {
      "learn": dist([r["memory"]["learn"] for r in runs]),
      "forget": dist([r["memory"]["forget"] for r in runs]),
      "refusals": dist([len(r["memory"]["refusals"]) for r in runs]),
    },
    "economy": {
      "earned": dist([e["earned"] for e in eco]),
      "balance": dist([e["balance"] for e in eco]),
      "spilled": dist([e["spilled"] for e in eco]),
      "identityHolds": sorted({str(e["identityHolds"]) for e in eco}),
      "hungerEnd": dict(Counter(str(e["hungerEnd"]) for e in eco)),
      "tasksDone": dist([e["tasks"].get("done") for e in eco]),
      "tasksFailed": dist([e["tasks"].get("failed") for e in eco]),
      "tasksExpired": dist([e["tasks"].get("expired") for e in eco]),
    },
  }


def rollup(records: list[dict], current=None) -> dict:
  """Aggregate records by series. Raises `MixedRegime` rather than pooling
  two regimes under one name. `current` is a callable `world -> hashes`
  (today's `data_hashes`), or None to leave `current` unset."""
  bad = {r.get("runId", "?"): problems(r) for r in records}
  bad = {k: v for k, v in bad.items() if v}
  if bad:
    raise ValueError("invalid record(s): " + "; ".join(
      f"{k}: {', '.join(v)}" for k, v in bad.items()))
  groups: dict[tuple, list[dict]] = {}
  for r in records:
    groups.setdefault(series_key(r), []).append(r)
  return {"schema": SCHEMA, "runs": len(records),
          "series": [_series(groups[k], current) for k in sorted(groups)]}


#: Files in `results/` that are not runs. A record is anything else, so a
#: new sidecar has to be named here or it is read as a malformed run.
NOT_RECORDS = (ROLLUP_NAME, NOTES_NAME)


def load_records(results_dir: Path) -> list[dict]:
  out = []
  for path in sorted(Path(results_dir).glob("*.json")):
    if path.name in NOT_RECORDS:
      continue
    out.append(json.loads(path.read_text()))
  return out


def write_rollup(results_dir: Path) -> Path:
  results_dir = Path(results_dir)
  doc = rollup(load_records(results_dir), current=data_hashes)
  target = results_dir / ROLLUP_NAME
  target.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
  return target
