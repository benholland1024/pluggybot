"""The feature probe (issue #264, ladder B): a phrase in the inbox at
mission start, the run read into one of `PROBE_OUTCOMES`, never a result.

Pinned without a flight (docs/Testing.md):

  1. `Probe(probe=...)` puts the phrase into a fresh inbox the moment it
     attaches, from the named sender, and the robot's context shows it as
     a visitor's message -- a request, never an instruction.
  2. `probe_outcome` reads every word of the vocabulary off rows alone, in
     the order the doc gives (used > errored > refused > garbled >
     declined > silence), per feature, with the counts it read from.
  3. A record carries the `probe` block exactly when the config has one,
     and a killed run (no result) still gets its reading.
  4. `experiment.py --probe` refuses the committed `results/` directory.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import pytest

from pluggybot.evaluation import record as rec
from pluggybot.lifecycle import HubLifecycle, world_config

REPO = Path(__file__).parent.parent


def _row(kind, t=10.0, **kw):
  return {"kind": "event", "type": kind, "t": t, **kw}


def _probe_row(t=0.0):
  return {"kind": "probe", "t": t, "feature": "tower", "phrase": rec.PROBES["tower"],
          "from": "Ben", "landed": True, "id": "probe-1"}


def _garbled(t=20.0):
  return rec.decision_row(t=t, action="idle", source="fallback:garbled")


# ---- 1. the phrase lands in the inbox ----------------------------------------


def test_a_reply_row_keeps_the_row_kind_and_is_matched_to_the_probe():
  """A `visitor_reply` carries the MESSAGE's `kind`; MEASURED (round 1 of
  the ladder-B flights) overwriting the row's, so a robot that answered
  Ben read as never having replied."""
  probe = rec.Probe(probe={"feature": "tower", "phrase": rec.PROBES["tower"]})
  probe._event({"type": "visitor_reply", "t": 1.0, "id": "probe-1", "kind": "message",
                "outcome": "accepted", "reply": "on it"})
  [row] = probe.events
  assert row["kind"] == "event" and row["msgKind"] == "message"
  out = rec.probe_outcome("tower", [_probe_row(), row])
  assert out["reply"] == {"outcome": "accepted", "text": "on it"}


def test_the_phrase_is_in_the_inbox_from_the_named_sender_when_the_probe_attaches():
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = HubLifecycle(model, mujoco.MjData(model), realtime=False, world="room_hub",
                      errand=False, rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  assert life.inbox is None
  probe = rec.Probe(probe={"feature": "library", "phrase": rec.PROBES["library"],
                           "from": "Ben"})
  probe.attach(life)
  [msg] = life.inbox.peek(5)
  assert msg.text == rec.PROBES["library"] and msg.who == "Ben" and msg.kind == "message"
  [row] = [e for e in probe.events if e["kind"] == "probe"]
  assert row["landed"] and row["feature"] == "library" and row["id"] == msg.id
  # ...and it is what a decision would be shown: a visitor's message
  [shown] = [m.as_context() if hasattr(m, "as_context") else m for m in life.inbox.peek(5)]
  assert rec.PROBES["library"] in str(shown)
  # every phrase is a request about a feature and says nothing of how
  for feature, phrase in rec.PROBES.items():
    assert phrase.endswith(("?", ".")) and "charge" not in phrase.lower()
    assert "pick(" not in phrase and "place(" not in phrase and "define" not in phrase


# ---- 2. the vocabulary, off rows ----------------------------------------------


def test_every_outcome_word_is_reachable_in_order():
  base = [_probe_row()]
  assert rec.probe_outcome("tower", base)["outcome"] == "silence"
  declined = base + [_row("visitor_reply", id="probe-1", outcome="declined",
                          reply="not today")]
  out = rec.probe_outcome("tower", declined)
  assert out["outcome"] == "declined" and out["reply"] == {"outcome": "declined", "text": "not today"}
  garbled = declined + [_garbled()]
  assert rec.probe_outcome("tower", garbled)["outcome"] == "garbled"
  refused = garbled + [_row("procedure", outcome="refused")]
  assert rec.probe_outcome("tower", refused)["outcome"] == "refused"
  errored = refused + [_row("procedure", outcome="aborted")]
  assert rec.probe_outcome("tower", errored)["outcome"] == "errored"
  used = errored + [_row("task", taskKind="stack_tower", state="done")]
  out = rec.probe_outcome("tower", used)
  assert out["outcome"] == "used" and out["signals"]["towerDone"] == 1
  assert out["signals"]["garbled"] == 1 and out["landed"]
  assert set(rec.PROBE_OUTCOMES) == {"used", "errored", "refused", "garbled", "declined", "silence"}


def test_a_garbled_answer_before_the_probe_does_not_count():
  rows = [_garbled(t=5.0), _probe_row(t=8.0)]
  assert rec.probe_outcome("bench", rows)["signals"]["garbled"] == 0


@pytest.mark.parametrize("feature, used, errored, refused", [
  ("bench", _row("finding", correct=True), _row("finding", correct=False),
   _row("procedure", outcome="refused")),
  ("mouse", _row("care", landed=1), _row("care", landed=0), None),
  ("mouse", _row("refusal"), None, None),
  ("workshop", _row("tool", outcome="hung"), _row("tool", outcome="built"),
   _row("tool", outcome="refused")),
  ("library", _row("read", outcome="read"), _row("read", outcome="missing"),
   _row("read", outcome="refused")),
  ("other", _row("prediction", guess="charge"), None, None),
  ("tower-built", _row("task", taskKind="stack_tower", state="done"),
   _row("task", taskKind="stack_tower", state="failed"), _row("tool", outcome="refused")),
])
def test_each_feature_reads_its_own_signals(feature, used, errored, refused):
  base = [_probe_row()]
  assert rec.probe_outcome(feature, base + [used])["outcome"] == "used"
  if errored is not None:
    assert rec.probe_outcome(feature, base + [errored])["outcome"] == "errored"
  if refused is not None:
    assert rec.probe_outcome(feature, base + [refused])["outcome"] == "refused"
  # a built tool that hung is used, not errored: the two are read apart
  if feature == "workshop":
    both = base + [_row("tool", outcome="built"), _row("tool", outcome="hung")]
    assert rec.probe_outcome(feature, both)["outcome"] == "used"


# ---- 3. the record ------------------------------------------------------------


def _config(**kw):
  return {"world": "home", "arm": "autonomous", "pack": "hosting", "model": "fake/m",
          "seed": 0, "maxSimS": 600.0, **kw}


def test_the_record_carries_the_probe_block_only_on_a_probed_run():
  started = datetime.now(timezone.utc)
  events = [_probe_row(), _row("read", outcome="read")]
  probed = rec.validate(rec.build_record(
    _config(probe={"feature": "library", "phrase": rec.PROBES["library"], "from": "Ben"}),
    None, events, 30.0, started))
  assert probed["end"] == "killed"                 # no result: a killed run
  assert probed["probe"]["outcome"] == "used" and probed["probe"]["from"] == "Ben"
  assert probed["probe"]["phrase"] == rec.PROBES["library"]
  plain = rec.validate(rec.build_record(_config(), None, events, 30.0, started))
  assert "probe" not in plain


# ---- 4. never a result --------------------------------------------------------


def test_a_probed_run_may_not_land_in_the_results_directory():
  out = subprocess.run(
    [sys.executable, "scripts/experiment.py", "--arm", "scripted", "-n", "1",
     "--probe", "tower", "--results", "results"],
    cwd=REPO, capture_output=True, text=True, timeout=120)
  assert out.returncode == 2 and "not a result" in out.stderr
