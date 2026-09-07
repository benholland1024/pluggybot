# results/ — committed measurement records (M14, issue #106)

One JSON file per run, written by `scripts/experiment.py`, plus
`rollup.json`, which aggregates them by series, and `notes.json`, which says
what each series MEANT. The record format is `docs/Evaluation.md` §4 (schema
v1); the code is `src/pluggybot/evaluation/`.

**A result set lands with its write-up** (§8). `notes.json` carries one entry
per series -- what was run, what the numbers say, what changed since the last
set, and what the set does **not** show -- and the suite fails on a series
with no entry, on an entry for a series that is gone, and on an empty
`notShown`. It is prose and the records are data: an entry never restates a
number the rollup already carries, because the second copy is the one that
goes stale. The website's data page renders it (rooftop-media-2026 #187), so
a number and its reading travel together or not at all.

These are vendored the way `protocol/` fixtures are: generated, checked in,
and stale-checked by `tests/test_experiment.py`. Do not edit a record by
hand — re-fly it. After editing any of the five data files (`rewards`,
`cadence`, `energy`, `metabolism`, `questions`) run

    uv run python scripts/experiment.py --rollup

so `rollup.json`'s `current` flags say the committed numbers describe a
previous regime. A regime is the five data files AND the world model (its
XML, includes and assets -- issue #110 changed one attribute of the robot
and every day after it was a different trajectory). Runs made on different
data files or a different world never share a series; move superseded runs
to `results/archive/` (not read by the rollup) rather than deleting them.

`archive/` holds the first committed set (2026-09-07, PR #109): five
`guarded` and five `scripted` days flown BEFORE the MSAA fix of issue #110,
when the world was not repeatable. They are the record of that spread --
the scripted five gave three trajectories -- and they carry no `world`
hash because the field did not exist yet.

A record with `end: "killed"` was stopped on wall clock and is excluded from
survival statistics; one with a non-empty `interventions` list likewise.
