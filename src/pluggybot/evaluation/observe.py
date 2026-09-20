"""Reading the deployed world (Evaluation.md §5, "How the observatory is
read"): the ONE route, and the rule that tells prompted rows from the
robot's own (issue #264).

`GET /api/pluggyworld/observe` on the website, behind the READ token, is
how a person, a script or an agent reads the observatory. Two scripts read
it -- `scripts/qualities.py --observe` (the six qualities, organic
adoption) and `scripts/feature_gates.py --observe` (what followed each
probe) -- and this module is what they share, so the route's cap, its
per-kind pull and the window rule are stated once.

THE WINDOW RULE. After every deploy the feature-gate checklist
(docs/Observatory.md, "The feature gates") is run through the visitor
channel: an admin asks the robot to use each feature, and reads its answer
for a problem. A feature used because it was asked for is not a feature
adopted, so the site stamps an admin's message a PROBE (`pw_messages.
probed_by`) and hands every reader the span it prompted: `from` is the
delivery, `until` is an hour after the robot's answer (or the delivery,
where none came), both null for a probe the robot never received. The
hour is the site's `PROBE_WINDOW_MS`; a reader never restates it. A row
is PROMPTED when it is the probed robot's and its wall-clock `createdAt`
falls inside a span. ⚠ A row with no `robot` (a site older than #264 sent
none on its event rows) is prompted if ANY span covers its time: the
conservative reading, because an organic count that quietly kept a
prompted row is the error this exists to prevent, and the other way round
is a smaller reading, said so.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

#: The route's per-call cap (`OBSERVE_MAX_LIMIT` on the site). A kind that
#: comes back at exactly the cap is TRUNCATED, oldest rows first.
OBSERVE_CAP = 1000
#: Every event kind the site files (its `PW_EVENT_KINDS`), one pull each. A
#: site older than a kind answers 400 for it, which a reading reports as
#: "not recorded there yet" rather than as zero rows; a reader that wants
#: fewer passes its own tuple.
OBSERVE_KINDS = ("death", "charge", "task", "intervention", "hunger", "thought",
                 "tool", "procedure", "encounter", "prediction", "message",
                 "transfer", "judged", "yield", "recall", "harm", "refusal",
                 "event_map", "read", "care", "finding", "conversation",
                 "constitution", "heart")


def get(site: str, token: str, **params) -> dict:
  url = f"{site.rstrip('/')}/api/pluggyworld/observe?{urllib.parse.urlencode(params)}"
  req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
  with urllib.request.urlopen(req, timeout=60) as res:
    return json.load(res)


def read_observe(site: str, token: str, days: int,
                 kinds: tuple[str, ...] = OBSERVE_KINDS) -> tuple[dict, list[dict], dict]:
  """(the base payload, every event row across the kinds, {kind: note}).

  The base call is made AT THE CAP for its `decisionRows` (the site sends
  the newest `limit`), and its unfiltered `events` are not used -- each
  kind is pulled on its own so no kind is crowded out of the window's cap.
  """
  base = get(site, token, days=days, limit=OBSERVE_CAP)
  events: list[dict] = []
  notes: dict[str, str] = {}
  if len(base.get("decisionRows") or []) >= OBSERVE_CAP:
    notes["decisions"] = f"truncated at {OBSERVE_CAP} (oldest dropped)"
  for kind in kinds:
    try:
      page = get(site, token, days=days, limit=OBSERVE_CAP, kind=kind)
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


# ---- the probes and their windows ---------------------------------------------


@dataclass(frozen=True)
class Window:
  """One probe's prompted span, as the site states it: the robot it was
  addressed to and the wall-clock `[start, end]`, plus the probe itself
  for a reading that wants to say what was asked."""

  robot: str
  start: datetime
  end: datetime
  probe: dict


def _when(value) -> datetime | None:
  if not isinstance(value, str) or not value:
    return None
  try:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
  except ValueError:
    return None


def windows(probes) -> list[Window]:
  """The spans off the site's `probes` list, in the order given. A probe
  with no `from` or `until` (never delivered) opens no window."""
  out: list[Window] = []
  for probe in probes or ():
    if not isinstance(probe, dict):
      continue
    start, end = _when(probe.get("from")), _when(probe.get("until"))
    if start is None or end is None or not probe.get("robot"):
      continue
    out.append(Window(robot=str(probe["robot"]), start=start, end=end, probe=probe))
  return out


def prompted(row: dict, spans: list[Window]) -> Window | None:
  """The window a row falls in, or None: the robot's own span, or any
  span where the row names no robot (the module docstring's rule)."""
  at = _when(row.get("createdAt"))
  if at is None:
    return None
  robot = row.get("robot") or None
  for span in spans:
    if robot is not None and span.robot != robot:
      continue
    if span.start <= at <= span.end:
      return span
  return None


def split(rows: list[dict], probes) -> tuple[list[dict], list[dict]]:
  """(the robot's own rows, the prompted rows) -- every row of `rows` in
  exactly one of the two."""
  spans = windows(probes)
  own, asked = [], []
  for row in rows:
    (asked if prompted(row, spans) else own).append(row)
  return own, asked
