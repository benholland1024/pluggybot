"""Many records -> one rollup, per series (issue #106; Evaluation.md §4).

A SERIES is one configuration -- (world, arm, pack, model) -- and a rollup
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
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from pluggybot.evaluation.notes import NOTES_NAME
from pluggybot.evaluation.record import SCHEMA, data_hashes, problems

ROLLUP_NAME = "rollup.json"


class MixedRegime(ValueError):
  """Two runs in one series read different data files."""


def series_key(record: dict) -> tuple:
  return (record["world"], record["arm"], record["pack"],
          record.get("model") or "none")


def _dist(values: list) -> dict:
  vals = [v for v in values if v is not None]
  if not vals:
    return {"n": 0, "min": None, "median": None, "max": None, "values": []}
  return {"n": len(vals), "min": min(vals), "median": statistics.median(vals),
          "max": max(vals), "values": vals}


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
  # Survival statistics EXCLUDE a run with an intervention in it and a run
  # that was killed on wall clock (Evaluation.md §5): the first measured an
  # admin, the second measured the box.
  clean = [r for r in runs if not r["interventions"] and r["end"] != "killed"]
  ch = [r["charging"] for r in runs]
  mind = [r["mind"] for r in runs]
  eco = [r["economy"] for r in runs]
  world, arm, pack, model = series_key(runs[0])
  return {
    "world": world, "arm": arm, "pack": pack, "model": model,
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
    "survival": {
      "n": len(clean),
      "survivalS": _dist([s for r in clean for s in r["survival"]["survivalS"]]),
      "deaths": {"flat": sum(r["survival"]["deaths"]["flat"] for r in clean),
                 "stuck": sum(r["survival"]["deaths"]["stuck"] for r in clean)},
      "minFraction": _dist([r["survival"]["minFraction"] for r in clean]),
    },
    "charging": {
      "forced": _dist([c["forced"] for c in ch]),
      "deferred": _dist([c["deferred"] for c in ch]),
      "voluntaryChosen": _dist([c["voluntary"]["chosen"] for c in ch]),
      "voluntaryHonoured": _dist([c["voluntary"]["honoured"] for c in ch]),
      "voluntaryFrac": _dist([f for c in ch for f in c["voluntary"]["chosenFrac"]]),
      "docked": _dist([c["docked"] for c in ch]),
      "anticipationOffer": _dist([c["anticipation"]["offer"] for c in ch]),
      "anticipationAction": _dist([c["anticipation"]["action"] for c in ch]),
    },
    "mind": {
      "decisions": _dist([m["decisions"] for m in mind]),
      "llmCalls": _dist([m["llmCalls"] for m in mind]),
      "fallbacks": _dist([m["fallbacks"] for m in mind]),
      "fallbackRate": _dist([m["fallbackRate"] for m in mind]),
      "fallbackReasons": dict(sum((Counter(m["fallbackReasons"]) for m in mind),
                                  Counter())),
      "wallS": _dist([w for m in mind for w in m["wallS"]["values"]]),
      "constrained": sorted({str(m.get("constrained")) for m in mind}),
      "longestStreak": _dist([m["longestStreak"] for m in mind]),
      "usd": _dist([m.get("usd") for m in mind]),
    },
    "whFailed": _dist([r["whFailed"] for r in runs]),
    "memory": {
      "learn": _dist([r["memory"]["learn"] for r in runs]),
      "forget": _dist([r["memory"]["forget"] for r in runs]),
      "refusals": _dist([len(r["memory"]["refusals"]) for r in runs]),
    },
    "economy": {
      "earned": _dist([e["earned"] for e in eco]),
      "balance": _dist([e["balance"] for e in eco]),
      "spilled": _dist([e["spilled"] for e in eco]),
      "identityHolds": sorted({str(e["identityHolds"]) for e in eco}),
      "hungerEnd": dict(Counter(str(e["hungerEnd"]) for e in eco)),
      "tasksDone": _dist([e["tasks"].get("done") for e in eco]),
      "tasksFailed": _dist([e["tasks"].get("failed") for e in eco]),
      "tasksExpired": _dist([e["tasks"].get("expired") for e in eco]),
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
