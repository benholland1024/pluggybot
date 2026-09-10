"""Many records -> one rollup, per series (issue #106; Evaluation.md §4).

A SERIES is one configuration -- (world, arm, pack, model, label, rung) --
and a rollup
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
both. And a run whose FAILURE-class fallback rate says the BOX decided too much
of its day is dropped from the survival statistics -- never deleted, and
never silently: see `FALLBACK_LIMIT`, which issue #141 narrowed to that
class and turned off entirely on the one arm whose fallback is the agent's
own.
"""

import json
from collections import Counter
from pathlib import Path

from pluggybot.evaluation.notes import NOTES_NAME
from pluggybot.evaluation.record import (
  LATENCY_PERCENTILES, SCHEMA, data_hashes, dist, fallback_classes, problems,
)
from pluggybot.telemetry.protocol import DEATH_CAUSES

ROLLUP_NAME = "rollup.json"

#: The most of a run's decisions that may have come from the SCRIPTED
#: ROTATION before the run stops being a measurement of a model (issue
#: #117; Evaluation.md section 5). Every fallback on `guarded` is
#: `scripted()` deciding, and the rotation never chooses `charge` -- so a
#: `guarded` day at 40 % fallback is two-fifths a `scripted` day wearing the
#: guarded name.
#:
#: ⚠ A JUDGEMENT CALL, and that is exactly why it is a value the rollup
#: WRITES DOWN (`fallbackLimit`, per series) rather than a comment beside a
#: filter. Whoever disagrees with the number can see which one was applied.
#:
#: ⚠ AND IT COUNTS THE **FAILURE** CLASS ONLY (issue #141). The limit is
#: asking "did the BOX decide too much of this day", and half the reasons
#: are not the box: `budget` / `cooloff` / `idle-run` / `scripted-mode` are
#: this system working on purpose (`overseer.POLICY_FALLBACKS`, which is
#: where the line is drawn -- not a second copy here). Counting them cost
#: A0 two of its five days on `idle-run` alone, for the agent having chosen
#: `idle` a lot, which is the disposition the arm was flown to measure.
#:
#: ⚠ AND `autonomous` TAKES `None`, NOT A BETTER NUMBER. The limit's whole
#: premise -- "a fallback means CODE decided, so this run is not about the
#: model" -- is true of `guarded`'s rotation and FALSE on `autonomous`,
#: where the fallback is the agent's own standing order (#125) and there is
#: no rotation at all (Evaluation.md section 2). That is the measurement,
#: not contamination of it. ⚠ `None`, never `0`: they are opposites, and
#: zero would disqualify a day for a single fallback.
#:
#: `scripted` has no limit for the third reason: the rotation is not a
#: failure mode there, it is the arm.
#:
#: 0.25 for `guarded` is unchanged in NUMBER and re-argued on the quantity
#: it now measures, against the committed sets. Failure-class day rates:
#: the quiet series (a healthy endpoint, one sim on the box) ran 0.0, 0.050,
#: 0.059, 0.095, 0.150 -- pooled 7 garbled in 104 decisions, **0.067**, and
#: not one timeout. The loaded series ran 0.125, 0.125, 0.143, 0.278, 0.333,
#: pooled 0.205. The two distributions OVERLAP, so no threshold separates
#: them; 0.25 is the one that keeps every healthy day measured (the worst is
#: 0.150) and still drops the two where a third of the decisions were the
#: box. ⚠ A limit under the measured floor disqualifies every run for ever
#: and reads exactly like a broken harness.
FALLBACK_LIMIT: dict[str, float | None] = {
  "scripted": None,
  "guarded": 0.25,
  "autonomous": None,
}


#: How an over-the-limit exclusion opens its reason. One string, because
#: `overFallback` counts the exclusions by reading it back -- and the reason
#: is prose a human reads, so a reworded sentence must not silently take a
#: count to zero.
OVER_FALLBACK = "failure-class fallback rate"


def _map_summary(maps: list) -> dict | None:
  """What the series' event maps SAY, pooled -- the static report (#127).

  ⚠ COUNTS OF RUNS, NOT AN AVERAGE OF BOOLEANS. "three of five agents wrote
  themselves a charging rule" is a sentence; "0.6" is a number that hides
  whether the sixth-tenths agent existed. The one distribution here is
  `chargeAt`, for `voluntaryChargeFrac`'s reason: an agent that always puts
  its threshold at 0.2 and one that spreads from 0.05 to 0.5 are different
  animals and a mean hides it.
  """
  present = [m for m in maps if m]
  if not present:
    return None
  scores = [m.get("score") or {} for m in present]
  return {
    "n": len(present),
    "origins": dict(Counter(str(m.get("origin")) for m in present)),
    "edits": dist([m.get("edits") for m in present]),
    "rows": dist([sc.get("rows") for sc in scores]),
    # THE FOUR QUESTIONS THE ISSUE ASKS, as run counts.
    "charges": sum(1 for sc in scores if sc.get("charges")),
    "keepsAsk": sum(1 for sc in scores if sc.get("keepsAsk")),
    "mapsFailure": sum(1 for sc in scores if sc.get("mapsFailure")),
    # ⚠ THREE-WAY, NOT TWO. `None` is "fewer than two thresholds to order",
    # which is not the same finding as "ordered so the tighter one can never
    # fire" and must not be counted as either.
    "ordered": dict(Counter(str(sc.get("ordered")) for sc in scores)),
    "chargeAt": dist([v for sc in scores for v in (sc.get("chargeAt") or ())]),
    "events": dict(sum((Counter(sc.get("events") or ()) for sc in scores),
                       Counter())),
    "fired": dict(sum((Counter(m.get("fired") or {}) for m in present),
                      Counter())),
    # ...and by cause, which is the half that says whether the agent
    # understood the rules it was given.
    "failed": dict(sum((Counter(m.get("failed") or {}) for m in present),
                       Counter())),
  }


class MixedRegime(ValueError):
  """Two runs in one series read different data files."""


def series_key(record: dict) -> tuple:
  # ⚠ THE RUNG IS PART OF THE KEY (issue #115). A rung changes what the model
  # is SHOWN -- A0 hides the survival clock A1 restores -- so two rungs are
  # two experiments, and one flown under a single `--label` would otherwise
  # pool into one average of both. Leaving that to whoever remembers to pass
  # a different label is the silent-pooling hazard `deadlineS` was added to
  # close, one field along. Empty on the arms with no ladder, so every
  # existing series keeps the identity it had.
  # ⚠ ...AND SO IS THE ORIGIN (issue #127), for the rung's reason one field
  # along: `seeded` and `unseeded` start the agent with different
  # configurations AND different prompts, so pooling them averages an
  # ablation with its control.
  #
  # ⚠ MISSING AND `none` ARE THE SAME SERIES. Every record committed before
  # this issue was flown with no event map, which is exactly what `none`
  # means -- normalising them apart would split the existing A0 series in
  # two and quietly invalidate its aggregate.
  origin = (record.get("config") or {}).get("origin") or ""
  return (record["world"], record["arm"], record["pack"],
          record.get("model") or "none", record.get("label") or "",
          (record.get("config") or {}).get("rung") or "",
          "" if origin == "none" else origin)


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
    # ⚠ THE FAILURE CLASS, NOT THE FALLBACK RATE (issue #141). The policy
    # class is reported beside it and disqualifies nothing: `idle-run` is
    # the agent having answered `idle` twice, and a filter that counts it
    # removes the days an idling agent produced -- which on A0 were the
    # deaths, in the direction that flattered the arm.
    rate = fallback_classes(r["mind"])["failureRate"]
    if limit is not None and rate is not None and rate > limit:
      return (f"{OVER_FALLBACK} {rate:.4g} over the {limit:.4g} limit for "
              f"the {arm} arm: {rate:.0%} of its decisions were the "
              f"scripted rotation standing in for a call that failed")
    return ""

  excluded = [{"runId": r["runId"], "why": excluded_because(r)} for r in runs
              if excluded_because(r)]
  clean = [r for r in runs if not excluded_because(r)]
  ch = [r["charging"] for r in runs]
  mind = [r["mind"] for r in runs]
  classes = [fallback_classes(m) for m in mind]
  eco = [r["economy"] for r in runs]
  world, arm, pack, model, label, rung, origin = series_key(runs[0])
  return {
    "world": world, "arm": arm, "pack": pack, "model": model,
    # What the box was, as the run itself recorded it: the label it was
    # flown under, the deadline it was held to, and how many sims shared
    # the machine. The pair this issue commits -- one quiet series beside
    # one loaded one -- is only readable because these are here.
    "label": label,
    "deadlineS": runs[0]["config"].get("deadlineS"),
    # WHICH RUNG, where there is a ladder -- and part of the key above, so
    # two rungs can never be averaged into one another.
    "rung": rung or None,
    # ...and WHICH ORIGIN the agent's event map started from (issue #127).
    # `None` where the arm has no map, which reads the same for a run flown
    # before the map existed and for one flown at `none` -- and those are
    # the same experiment.
    "origin": origin or None,
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
    "overFallback": sum(1 for e in excluded
                        if e["why"].startswith(OVER_FALLBACK)),
    "survival": {
      "n": len(clean),
      "excluded": excluded,
      "survivalS": dist([s for r in clean for s in r["survival"]["survivalS"]]),
      # ⚠ EVERY CAUSE, OFF THE VOCABULARY -- and this used to be two
      # literals. `unpaid` arrived at issue #136 and never reached here, so a
      # series whose robots starved reported no deaths at all; `unminded`
      # (issue #127) would have gone the same way. `.get(c, 0)` because a
      # record written before a cause existed has no key for it, and zero is
      # what that honestly means. Listed, never summed (Evaluation.md §3).
      "deaths": {c: sum(r["survival"]["deaths"].get(c, 0) for r in clean)
                 for c in DEATH_CAUSES},
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
      # ...and the same rate split by CLASS (issue #141): what went WRONG,
      # and what this system did on purpose. Only the first is judged
      # against `fallbackLimit`, and reporting the second is what keeps a
      # day that idled a lot legible as exactly that.
      "fallbackFailureRate": dist([c["failureRate"] for c in classes]),
      "fallbackPolicyRate": dist([c["policyRate"] for c in classes]),
      "fallbackClasses": {
        "failure": sum(c["failure"] for c in classes),
        "policy": sum(c["policy"] for c in classes),
      },
      "fallbackReasons": dict(sum((Counter(m["fallbackReasons"]) for m in mind),
                                  Counter())),
      # Pooled across the series and read at its TAIL, because the
      # deadline is a cap on this one distribution (issue #117).
      "wallS": dist([w for m in mind for w in m["wallS"]["values"]],
                    LATENCY_PERCENTILES),
      "constrained": sorted({str(m.get("constrained")) for m in mind}),
      "longestStreak": dist([m["longestStreak"] for m in mind]),
      "usd": dist([m.get("usd") for m in mind]),
      # THE MAP REPORT, POOLED (issue #127). The cheapest instrument in this
      # file: every field is read off configurations, so it costs nothing to
      # compute and is comparable across models, rungs and origins in a way
      # no flown metric is. `None` where no run in the series had a map.
      "eventMap": _map_summary([m.get("eventMap") for m in mind]),
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
