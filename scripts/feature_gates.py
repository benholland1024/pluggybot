#!/usr/bin/env python3
"""What followed each probe: the feature-gate checklist, read off the
deployed world (issue #264; docs/Observatory.md, "The feature gates").

    scripts/feature_gates.py --observe [--site https://rooftop-media.org] [--days 2]
    scripts/feature_gates.py --observe --json

After every deploy an admin asks each robot, through the visitor channel,
to use each feature -- the checklist's probes. The site marks each such
message a PROBE and hands back the hour it prompted (`evaluation/observe.
py`); this script reads ONE `/observe` window and prints, per probe, what
was asked, what the robot said, and every decision and event of that
robot inside the window, grouped by kind -- the evidence a pass or a fail
is written from. The VERDICT IS THE READER'S: a pass is the feature used
without a problem or a reasoned refusal; a fail is a garbled attempt, a
refusal the robot could not explain, an action that errored, or silence.
The script names the cheap half -- `silent` (no answer inside the window,
and the window is over), `errored` (a garbled decision or a refused /
aborted / failed row of the feature's kind), `open` (the window has not
closed yet) -- and quotes the rest.

⚠ A READING, NOT A RESULT: nothing here goes in `results/`. The outcome
goes in the dated entry under the checklist in docs/Observatory.md, one
line per probe, and a fail becomes an issue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pluggybot.evaluation import observe  # noqa: E402

#: Rows that say an attempt went wrong, by kind -> the subjects that mean it.
#: The words are the site's own (each kind's outcome vocabulary); a kind
#: not listed has no failing subject a gate reads.
ERROR_SUBJECTS: dict[str, tuple[str, ...]] = {
  "procedure": ("refused", "aborted"),
  "tool": ("refused",),
  "read": ("failed",),
  "task": ("failed",),
  "thought": ("refused",),
}
#: A decision the model did not produce (`fallback:<why>`); `garbled` is
#: the one that means the attempt itself was malformed.
GARBLED = "fallback:garbled"


def gate(probe: dict, window: observe.Window | None, decisions: list[dict],
         events: list[dict], now: datetime) -> dict:
  """One probe's evidence: the ask, the answer, the rows, and the cheap
  half of the verdict."""
  by_kind: dict[str, Counter] = defaultdict(Counter)
  errored: list[str] = []
  for e in events:
    by_kind[str(e.get("kind"))][str(e.get("subject"))] += 1
    if str(e.get("subject")) in ERROR_SUBJECTS.get(str(e.get("kind")), ()):
      errored.append(f"{e.get('kind')}:{e.get('subject')} {e.get('detail') or ''}".strip())
  garbled = [d for d in decisions if d.get("source") == GARBLED]
  if garbled:
    errored.append(f"{len(garbled)} garbled decision(s)")
  answered = probe.get("status") not in (None, "", "pending", "delivered")
  over = window is not None and window.end <= now
  if window is None:
    cheap = "undelivered"
  elif not answered and over:
    cheap = "silent"
  elif errored:
    cheap = "errored"
  elif not over:
    cheap = "open"
  else:
    cheap = "answered"
  return {
    "id": probe.get("id"), "by": probe.get("by"), "robot": probe.get("robot"),
    "at": probe.get("at"), "from": probe.get("from"), "until": probe.get("until"),
    "text": probe.get("text"), "status": probe.get("status"),
    "reply": probe.get("reply"), "action": probe.get("action"),
    "decisions": [{"t": d.get("simTime"), "action": d.get("action"),
                   "detail": d.get("detail"), "source": d.get("source"),
                   "reason": d.get("reason")} for d in decisions],
    "events": {k: dict(v) for k, v in sorted(by_kind.items())},
    "errored": errored,
    "cheap": cheap,
  }


def gates(base: dict, events: list[dict], now: datetime | None = None) -> list[dict]:
  """Every probe in the payload with what followed it, oldest first."""
  now = now or datetime.now(timezone.utc)
  probes = base.get("probes")
  if not isinstance(probes, list):
    return []
  spans = {w.probe.get("id"): w for w in observe.windows(probes)}
  decisions = base.get("decisionRows") or []
  out = []
  for probe in probes:
    span = spans.get(probe.get("id"))
    own = [span] if span is not None else []
    out.append(gate(probe, span,
                    sorted((d for d in decisions if observe.prompted(d, own)), key=_order),
                    sorted((e for e in events if observe.prompted(e, own)), key=_order), now))
  return out


def _order(row: dict) -> tuple:
  return (str(row.get("createdAt") or ""), row.get("id") or 0)


def render(base: dict, found: list[dict], notes: dict[str, str]) -> str:
  lines = [f"observatory {base.get('commit')} · live {base.get('live')} · "
           f"window {base.get('window', {}).get('since')} → now · "
           f"{len(found)} probes · a reading, not a result"]
  for kind, note in notes.items():
    lines.append(f"  ⚠ {kind}: {note}")
  if "probes" not in base:
    lines.append("  ⚠ the site sends no probes (older than #264): nothing to read")
  for g in found:
    lines.append(f"\n== {g['at']} · {g['by']} → {g['robot']} · {g['status']} · {g['cheap']}")
    lines.append(f"  asked: {g['text']}")
    if g["reply"]:
      lines.append(f"  said:  {g['reply']}" + (f"  [{g['action']}]" if g["action"] else ""))
    lines.append(f"  window: {g['from']} → {g['until']}")
    for d in g["decisions"]:
      what = f"{d['action']}" + (f" ({d['detail']})" if d["detail"] else "")
      src = "" if d["source"] in ("llm", None) or str(d["source"]).startswith("llm:") else f" [{d['source']}]"
      lines.append(f"    {str(d['t'] or '?'):>9}s  {what}{src}" + (f" — {d['reason']}" if d["reason"] else ""))
    for kind, subjects in g["events"].items():
      lines.append(f"    {kind}: " + ", ".join(f"{s} ×{n}" for s, n in sorted(subjects.items())))
    for err in g["errored"]:
      lines.append(f"    ⚠ {err}")
  return "\n".join(lines)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--observe", action="store_true", help="read the deployed world")
  ap.add_argument("--site", default="https://rooftop-media.org")
  ap.add_argument("--days", type=int, default=2,
                  help="the window to read (a checklist is run inside a day of its deploy)")
  ap.add_argument("--json", action="store_true", help="print the gates as JSON")
  args = ap.parse_args(argv)
  if not args.observe:
    ap.error("say --observe")
  token = os.environ.get("PLUGGYWORLD_READ_TOKEN", "")
  if not token:
    ap.error("$PLUGGYWORLD_READ_TOKEN is not set (the READ token, never the ingest one)")
  base, events, notes = observe.read_observe(args.site, token, args.days)
  found = gates(base, events)
  if args.json:
    print(json.dumps({"commit": base.get("commit"), "live": base.get("live"),
                      "window": base.get("window"), "notes": notes, "gates": found},
                     indent=2, default=str))
  else:
    print(render(base, found, notes))
  return 0


if __name__ == "__main__":
  sys.exit(main())
