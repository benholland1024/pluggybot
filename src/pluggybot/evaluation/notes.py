"""What a result set MEANT, written when it was collected (Evaluation.md §8).

`results/` holds numbers; nothing in it holds what they say. The website's
`/experiments/pluggyworld/data` page (rooftop-media-2026 #187) renders this
file, and the sequencing rule that page is built under is the reason it
exists: the PAGE is built late, so it does not shape the experiments around
what renders nicely, but the EXPLANATION is written when the data lands.
A page over undocumented numbers would have to invent the interpretation at
render time, which is the failure mode `docs/Evaluation.md` is about.

An entry is per SERIES -- the rollup's own `(world, arm, pack, model)` -- and
its shape is §3's baseline sections reduced to their four moving parts: what
was RUN, what the numbers SAID, what CHANGED since the last set, and what the
set does NOT show. The fourth is not decoration; every baseline section here
has one and it is the half a reader skips.

⚠ **This file is PROSE and the records are DATA, and neither restates the
other.** An entry never carries a number the rollup already carries -- a
second copy of `fallbackRate` here is a copy that goes stale silently, and
the page has the rollup open beside it. What belongs here is the sentence a
column cannot hold.

⚠ **THE FIELDS ARE PLAIN PROSE, NOT MARKDOWN.** The one consumer renders
them as text, so a backtick renders as a backtick and a `--` as two hyphens —
which is what the first draft of this file did, having been written in a repo
where both are punctuation. Name a file or a flag in words, and use real
dashes. `results/README.md` beside it is markdown and is the place for the
other register.

⚠ **`schema` is this file's own, not the record's.** Prose gains a field far
more readily than a measurement does, and a page pinning one version should
not be broken by the other moving.
"""

import json
from pathlib import Path

NOTES_NAME = "notes.json"
SCHEMA = 1

#: Every entry carries all of these. `changed` may be empty prose (a first
#: set changed nothing) but the KEY is required, because an absent key and a
#: deliberate "nothing changed" are different claims and only one of them
#: says somebody looked.
FIELDS = ("series", "title", "date", "ran", "found", "changed", "notShown")


def series_id(world: str, arm: str, pack: str, model: str | None,
              label: str = "") -> str:
  """The rollup's series key as one string. `none` for a model-less arm,
  matching `rollup.series_key`, so an entry names a series the same way the
  numbers do and a typo cannot silently address nothing.

  A `label` (issue #117 -- the conditions a run was flown under) is
  APPENDED and only when there is one, so the same series keeps the same
  id it had before labels existed. Two series that differ only by label
  are two write-ups, which is the point of flying the pair: what a quiet
  box was worth is a sentence, and it goes in the labelled one."""
  return f"{world}/{arm}/{pack}/{model or 'none'}" + (f"/{label}" if label
                                                      else "")


def load(results_dir: Path) -> dict:
  """The committed notes, or an empty document if the file is absent."""
  path = Path(results_dir) / NOTES_NAME
  if not path.exists():
    return {"schema": SCHEMA, "entries": []}
  return json.loads(path.read_text())


def problems(doc: dict, series_ids: list[str] | None = None) -> list[str]:
  """Everything wrong with a notes document, as sentences. Empty means valid.

  With `series_ids` (every series in the rollup, superseded ones included)
  it also checks the two directions that matter: every committed series is
  explained, and no entry explains a series that is not there. The second
  catches a renamed world or a re-flown model, where the prose survives its
  data and quietly describes runs nobody can look at. A series that has
  gone stale keeps its entry -- what it meant is still what it meant."""
  out = []
  if doc.get("schema") != SCHEMA:
    out.append(f"schema {doc.get('schema')!r}, expected {SCHEMA}")
  entries = doc.get("entries")
  if not isinstance(entries, list):
    return out + ["entries is not a list"]
  seen = set()
  for i, entry in enumerate(entries):
    where = entry.get("series", f"entry {i}")
    for key in FIELDS:
      if key not in entry:
        out.append(f"{where}: missing {key!r}")
    if not isinstance(entry.get("found"), list) or not entry.get("found"):
      out.append(f"{where}: found is not a non-empty list")
    if not entry.get("notShown"):
      out.append(f"{where}: notShown is empty -- every set has a limit, and "
                 "it is the half a reader skips")
    if entry.get("series") in seen:
      out.append(f"{where}: two entries for one series")
    seen.add(entry.get("series"))
  if series_ids is not None:
    for sid in series_ids:
      if sid not in seen:
        out.append(f"{sid}: a committed series with no write-up "
                   "(Evaluation.md §8)")
    for sid in sorted(seen - set(series_ids)):
      out.append(f"{sid}: a write-up for a series that is not in the rollup")
  return out
