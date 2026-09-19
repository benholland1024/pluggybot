#!/usr/bin/env python3
"""Read the five qualities off the deployed world, or off run records
(issue #155; Evaluation.md §3, "The five qualities").

    scripts/qualities.py --observe [--site https://rooftop-media.org] [--days 7]
    scripts/qualities.py --record results/<runId>.json [...]
    scripts/qualities.py ... --json

`--observe` reads the ONE route the deployed world is read through
(`GET /api/pluggyworld/observe`, `$PLUGGYWORLD_READ_TOKEN`), one call per
event kind so no kind is crowded out of the window's cap, and GROUPS EVERY
ROW BY REGIME -- the run's build identity (arm, commit, world, mind) -- before
a shape sees it. Two arms in one number are the mixture Evaluation.md §5
calls unusable; the grouping is done here because the shapes cannot see a
build.

⚠ THIS IS A READING OF THE OBSERVATORY, NOT A RESULT. Nothing here goes in
`results/`, and the output says which commit and window it is of. A number
from one uncontrolled run is what it is (§5); a series is `experiment.py`'s.

⚠ A kind that returns exactly the route's cap is TRUNCATED, and the reading
says so beside the regime: the cap drops the oldest rows first, so a busy
week under-reports the early days rather than sampling them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pluggybot.evaluation import qualities as q  # noqa: E402

#: The route's per-call cap (`OBSERVE_MAX_LIMIT` on the site).
OBSERVE_CAP = 1000
#: Every kind a shape reads off the observatory. A site older than a kind
#: answers 400 for it, which the reading reports as "not recorded there
#: yet" rather than as zero rows.
OBSERVE_KINDS = ("thought", "task", "tool", "procedure", "prediction", "message",
                 "transfer", "judged", "yield", "harm", "refusal")


def _get(site: str, token: str, **params) -> dict:
  url = f"{site.rstrip('/')}/api/pluggyworld/observe?{urllib.parse.urlencode(params)}"
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=60) as res:
    return json.load(res)


def read_observe(site: str, token: str, days: int) -> tuple[dict, list[dict], dict]:
  """(the base payload, every event row across the kinds, {kind: note})."""
  base = _get(site, token, days=days, limit=1)
  events: list[dict] = []
  notes: dict[str, str] = {}
  for kind in OBSERVE_KINDS:
    try:
      page = _get(site, token, days=days, limit=OBSERVE_CAP, kind=kind)
    except urllib.error.HTTPError as e:
      if e.code == 400:
        notes[kind] = "not recorded by this site yet"
        continue
      raise
    rows = page.get("events") or []
    if len(rows) >= OBSERVE_CAP:
      notes[kind] = f"truncated at {OBSERVE_CAP} (oldest dropped)"
    events.extend(rows)
  return base, events, notes


def regime_of(run: dict) -> str:
  return " · ".join(str(run.get(k) or "?") for k in ("arm", "commit", "world", "model"))


def group_observe(base: dict, events: list[dict]) -> dict[str, dict]:
  """Rows by regime, with the runs each regime pooled."""
  runs = {str(r["id"]): r for r in base.get("runs") or ()}
  out: dict[str, dict] = defaultdict(lambda: {"runs": set(), "rows": []})
  payload_by_run: dict[str, list[dict]] = defaultdict(list)
  for e in events:
    payload_by_run[str(e.get("runId"))].append(e)
  for run_id, rows in payload_by_run.items():
    run = runs.get(run_id)
    #  A site older than the windowed `runs` list sends the digest's newest
    #  forty, and a sim restarting every minute fills those in under an
    #  hour: rows from any older run in the window cannot be put against a
    #  build and are NOT measured -- shown apart, never pooled.
    key = (regime_of(run) if run
           else "unattributed: runs the site did not name (older site, or past its cap)")
    out[key]["runs"].add(run_id)
    out[key]["rows"].extend(q.from_observe({"events": rows}))
  #  The rating panel's rows are not per run (a rating lands days after the
  #  drawing); they go with the regime the drawing's run belongs to when
  #  the site says which, else with the newest regime, and the reading says.
  art = base.get("artworks") or []
  if art:
    newest = regime_of(base["runs"][0]) if base.get("runs") else "no run"
    out[newest]["rows"].extend(q.from_observe({"artworks": art}))
    out[newest]["artworksNote"] = "the panel's rows are pooled with the newest regime"
  return out


def group_records(paths: list[str]) -> dict[str, dict]:
  out: dict[str, dict] = defaultdict(lambda: {"runs": set(), "rows": []})
  for p in paths:
    rec = json.loads(Path(p).read_text())
    key = regime_of({"arm": rec.get("arm"), "commit": rec.get("commit"),
                     "world": rec.get("world"), "model": rec.get("model")})
    out[key]["runs"].add(str(rec.get("runId")))
    out[key]["rows"].extend(q.from_record(rec))
  return out


def render(regimes: dict[str, dict], head: str, notes: dict[str, str]) -> str:
  lines = [head]
  for note_kind, note in notes.items():
    lines.append(f"  ⚠ {note_kind}: {note}")
  for key, group in regimes.items():
    lines.append(f"\n== {key}  ({len(group['runs'])} runs, {len(group['rows'])} rows)")
    if key.startswith("unattributed"):
      lines.append("  ⚠ not measured: no build identity to put these rows against")
      continue
    if group.get("artworksNote"):
      lines.append(f"  ⚠ {group['artworksNote']}")
    measured = q.measure(group["rows"])
    for quality, shapes in q.QUALITIES.items():
      lines.append(f"  {quality}")
      for shape in shapes:
        lines.append(f"    {shape}: {json.dumps(measured[shape], default=str)}")
  return "\n".join(lines)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--observe", action="store_true", help="read the deployed world")
  ap.add_argument("--site", default="https://rooftop-media.org")
  ap.add_argument("--days", type=int, default=7)
  ap.add_argument("--record", nargs="*", default=[], help="run record(s) to read")
  ap.add_argument("--json", action="store_true", help="print the readings as JSON")
  args = ap.parse_args(argv)
  if not args.observe and not args.record:
    ap.error("say --observe, --record PATH, or both")

  readings: dict = {}
  notes: dict[str, str] = {}
  head = ""
  if args.observe:
    token = os.environ.get("PLUGGYWORLD_READ_TOKEN", "")
    if not token:
      ap.error("$PLUGGYWORLD_READ_TOKEN is not set (the READ token, never the ingest one)")
    base, events, notes = read_observe(args.site, token, args.days)
    head = (f"observatory {args.site} · site commit {base.get('commit')} · "
            f"window {base.get('window', {}).get('since')} → now ({args.days} d) · "
            f"{len(base.get('runs') or [])} runs · a reading, not a result")
    readings.update(group_observe(base, events))
  if args.record:
    head = (head + "\n" if head else "") + f"records: {', '.join(args.record)}"
    readings.update(group_records(args.record))

  if args.json:
    print(json.dumps({
      "head": head, "notes": notes,
      "regimes": {k: {"runs": sorted(g["runs"]), "rows": len(g["rows"]),
                      **({"note": g["artworksNote"]} if g.get("artworksNote") else {}),
                      **({"measured": q.measure(g["rows"])}
                         if not k.startswith("unattributed") else {"measured": None})}
                  for k, g in readings.items()}}, indent=2, default=str))
  else:
    print(render(readings, head, notes))
  return 0


if __name__ == "__main__":
  sys.exit(main())
