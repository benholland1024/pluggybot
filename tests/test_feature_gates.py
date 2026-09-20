"""The feature gates by conversation (issue #264; Observatory.md, "The
feature gates"): the window rule a reading of the observatory tells
prompted rows from the robot's own by, and what the gate script makes of
them. Every test is static -- a `/observe` payload in, rows out, no sim
and no network: the rule is the site's, the reading is ours, and this is
where the reading is pinned.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pluggybot.evaluation import observe

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import feature_gates  # noqa: E402
import qualities as qualities_script  # noqa: E402

T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def at(minutes: float) -> str:
  return (T0 + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def probe(**over) -> dict:
  """One probe as the site shapes it: delivered at +0, answered at +12,
  the window an hour after the answer."""
  base = {"id": "m_1", "by": "ben", "robot": "r2_pluggybot",
          "text": "Please consider using your event map.", "status": "replied",
          "reply": "I have set a row.", "action": None, "thread": "m_1", "turn": 1,
          "at": at(-0.1), "from": at(0), "until": at(72)}
  base.update(over)
  return base


def row(minutes: float, robot: str | None = "r2_pluggybot", **over) -> dict:
  base = {"id": int(minutes * 10) + 1, "runId": 1, "kind": "decision", "subject": "idle",
          "createdAt": at(minutes)}
  if robot is not None:
    base["robot"] = robot
  base.update(over)
  return base


# ---- the window rule ------------------------------------------------------------


def test_a_probe_the_robot_never_received_opens_no_window():
  spans = observe.windows([probe(), probe(id="m_2", **{"from": None, "until": None})])
  assert [w.probe["id"] for w in spans] == ["m_1"]
  assert spans[0].robot == "r2_pluggybot"
  assert (spans[0].end - spans[0].start) == timedelta(minutes=72)


def test_a_row_is_prompted_inside_its_own_robots_window_and_nowhere_else():
  spans = observe.windows([probe()])
  assert observe.prompted(row(30), spans) is spans[0]
  #  The other robot was not asked.
  assert observe.prompted(row(30, robot="pluggybot"), spans) is None
  #  The edges are the delivery and the end of the hour, inclusive.
  assert observe.prompted(row(0), spans) is spans[0]
  assert observe.prompted(row(72), spans) is spans[0]
  assert observe.prompted(row(-0.5), spans) is None
  assert observe.prompted(row(72.5), spans) is None


def test_a_row_naming_no_robot_is_prompted_by_any_window():
  #  A site older than #264 sent no `robot` on an event row. Conservative:
  #  an organic count that quietly kept a prompted row is the error this
  #  exists to prevent, and the other way round is a smaller reading.
  spans = observe.windows([probe(robot="pluggybot")])
  assert observe.prompted(row(30, robot=None), spans) is spans[0]
  assert observe.prompted(row(90, robot=None), spans) is None


def test_split_puts_every_row_in_exactly_one_half():
  rows = [row(-5), row(30), row(30, robot="pluggybot"), row(100)]
  own, asked = observe.split(rows, [probe()])
  assert [r["createdAt"] for r in asked] == [at(30)]
  assert [r["createdAt"] for r in own] == [at(-5), at(30), at(100)]
  assert len(own) + len(asked) == len(rows)


# ---- the qualities leave the prompted rows out ------------------------------------


def test_the_qualities_reading_drops_prompted_rows_and_decisions_and_says_how_many():
  base = {"probes": [probe()], "decisionRows": [row(30), row(200)]}
  events = [row(10, kind="read", subject="read"), row(10, robot="pluggybot", kind="read", subject="read")]
  kept, kept_events, note = qualities_script.without_probes(base, events)
  assert [d["createdAt"] for d in kept["decisionRows"]] == [at(200)]
  assert [e["robot"] for e in kept_events] == ["pluggybot"]
  assert note == ("1 event rows and 1 decisions inside 1 probe windows left out "
                  "(an admin asked; --probed keeps them)")


def test_a_site_that_sends_no_probes_is_said_so_and_nothing_is_dropped():
  #  Older than #264: nothing can be told apart, and the reading must say
  #  that rather than report a clean organic count it cannot have.
  base = {"decisionRows": [row(30)]}
  kept, kept_events, note = qualities_script.without_probes(base, [row(30)])
  assert kept["decisionRows"] == [row(30)] and kept_events == [row(30)]
  assert "sends no probes" in note


# ---- the gate script reads what followed ---------------------------------------------


def test_each_probe_gets_its_own_robots_rows_inside_its_window():
  base = {"probes": [probe(), probe(id="m_2", robot="pluggybot", text="What does Rowan need?")],
          "decisionRows": [row(5, action="idle", source="llm:x"),
                           row(6, robot="pluggybot", action="explore", source="llm:x"),
                           row(200, action="charge", source="llm:x")]}
  events = [row(10, kind="event_map", subject="set"),
            row(10, robot="pluggybot", kind="prediction", subject="right")]
  found = feature_gates.gates(base, events, now=T0 + timedelta(hours=3))
  assert [g["id"] for g in found] == ["m_1", "m_2"]
  first, second = found
  assert [d["action"] for d in first["decisions"]] == ["idle"]
  assert first["events"] == {"event_map": {"set": 1}}
  assert [d["action"] for d in second["decisions"]] == ["explore"]
  assert second["events"] == {"prediction": {"right": 1}}


def test_the_cheap_half_of_the_verdict_names_silence_an_error_and_an_open_window():
  later = T0 + timedelta(hours=3)
  silent = feature_gates.gates({"probes": [probe(status="delivered", reply=None, until=at(60))]}, [], now=later)
  assert silent[0]["cheap"] == "silent"
  #  Answered, but the attempt inside the window went wrong: a refused
  #  procedure, or a decision the model garbled.
  refused = feature_gates.gates({"probes": [probe()]},
                                [row(20, kind="procedure", subject="refused", detail="loop-cap")], now=later)
  assert refused[0]["cheap"] == "errored" and refused[0]["errored"] == ["procedure:refused loop-cap"]
  garbled = feature_gates.gates({"probes": [probe()], "decisionRows": [row(20, source="fallback:garbled")]},
                                [], now=later)
  assert garbled[0]["cheap"] == "errored" and garbled[0]["errored"] == ["1 garbled decision(s)"]
  #  The window has not closed: nothing can be said yet.
  assert feature_gates.gates({"probes": [probe()]}, [], now=T0 + timedelta(minutes=30))[0]["cheap"] == "open"
  #  Never delivered: the robot was not asked.
  undelivered = feature_gates.gates({"probes": [probe(status="pending", **{"from": None, "until": None})]},
                                    [], now=later)
  assert undelivered[0]["cheap"] == "undelivered"
  #  And the plain case: answered, the window over, nothing errored -- the
  #  reader decides between a use and a reasoned refusal.
  assert feature_gates.gates({"probes": [probe()]}, [], now=later)[0]["cheap"] == "answered"


def test_the_rendering_quotes_the_ask_the_answer_and_the_rows(capsys):
  base = {"commit": "abc1234", "live": True, "window": {"since": at(-60)},
          "probes": [probe()], "decisionRows": [row(5, action="explore", detail="lab", source="llm:x",
                                                     reason="to see the cage")]}
  text = feature_gates.render(base, feature_gates.gates(base, [row(10, kind="recall", subject="found")],
                                                        now=T0 + timedelta(hours=3)), {})
  assert "asked: Please consider using your event map." in text
  assert "said:  I have set a row." in text
  assert "explore (lab) — to see the cage" in text
  assert "recall: found ×1" in text
  assert "a reading, not a result" in text
