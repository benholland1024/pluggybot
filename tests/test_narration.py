"""A Python exception is not something the robot says (issue #76).

Three places turned a CAUGHT exception into visitor-facing prose. All three
are paths where the code recovered correctly -- nothing crashed, the tool went
back to its bay, the rotation decided, the mission carried on -- so the only
thing broken was what the robot said about itself:

  DECIDE explore: ... [fallback:RuntimeError: Connection reset by peer]
  USE_TOOL FAILED: IndexError: list index out of range -- stowing the tool anyway
  VISITOR rating 7 ignored: 'pluggybot: no ledger entry 7'

The first of those is the worst, because it does not stop at the status line.
`Decision.summary()` carries `source`, `lifecycle._after_decision` puts the
summary through `_say` (every telemetry frame) AND `_remember` (`History.md`,
which the site badges as the robot's own paper trail) -- so a vendor's HTTP
error class became something the robot had written down about its day.

The rule these hold down, and it is one rule with three faces:

  WHAT THE ROBOT SAYS IS A SENTENCE; WHAT THE OPERATOR NEEDS IS EVIDENCE, AND
  THEY TRAVEL SEPARATELY. `_say(msg, detail=...)` is the seam -- `msg` reaches
  `status`, the frames and `say_hooks`; `detail` reaches the console and
  `self.log` and nothing else. On the overseer's side the same split is
  `FALLBACK_REASONS` (a closed vocabulary, on the wire) against `Usage.errors`
  (the class and the message, for whoever is debugging).

⚠ `ThoughtRefused` is deliberately NOT covered. Its message is authored prose
that says which file refused a write and why ("Main.md is written by a person
editing the file..."), the class docstring says the caller narrates it, and a
robot saying it out loud is the feature. The rule is about exception REPRS --
a class name, a re-quoted KeyError argument -- not about every value that
happens to arrive in an `except` clause.
"""

import re

import pytest

from pluggybot.economy.ledger import Ledger
from pluggybot.economy.scoring import evaluate
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.overseer import Overseer

from test_overseer import FakeClient, _lifecycle, full


#: What a Python exception looks like when it reaches prose. Deliberately a
#: SHAPE rather than a list of the three known strings: the point is that the
#: next `except` clause somebody writes cannot quietly reintroduce this, and a
#: test enumerating today's offenders would not notice tomorrow's.
LOOKS_LIKE_AN_EXCEPTION = re.compile(
  r"""\b\w*(?:Error|Exception|Interrupt)\b   # RuntimeError, KeyError, ...
    | ^\s*Traceback
    | \bobject\ at\ 0x                       # a bare repr
  """, re.VERBOSE)


def narration(life) -> list[str]:
  """Every line the robot SAYS, captured off `say_hooks`.

  The honest instrument, and `life.status` is not: `status` holds only the
  most recent line, so an assertion against it silently checks whatever was
  said LAST rather than the line under test -- the use-phase failure below is
  followed immediately by `SWAP_RETURN`, and reading `status` passed happily
  against the unfixed code. `say_hooks` is what `publisher.event` consumes, so
  this is the wire.
  """
  said: list[str] = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  return said


def assert_sounds_like_a_robot(line: str) -> None:
  hit = LOOKS_LIKE_AN_EXCEPTION.search(line)
  assert hit is None, \
      f"a Python exception reached the robot's own voice: {line!r} " \
      f"(matched {hit.group(0)!r} at {hit.start()})"


# ---- 1. an API outage is not the robot's memory ------------------------------


def test_the_fallback_vocabulary_is_closed_and_documented():
  """The tokens are a two-repo vocabulary: the site renders `source`, so
  ADDING one is additive and RENAMING one is breaking. Pinning the set is what
  makes that rule enforceable rather than a wish -- and it is what caught
  `busy`, a ninth token issue #76 did not list and the docs had never had."""
  assert ov.FALLBACK_REASONS == (
    "timeout", "offline", "garbled", "budget", "cooloff", "busy",
    "idle-run", "no-client", "scripted-mode")
  # No duplicates, and every one of them is a token a human could read rather
  # than a class name: lowercase, no colons, no spaces.
  assert len(set(ov.FALLBACK_REASONS)) == len(ov.FALLBACK_REASONS)
  for why in ov.FALLBACK_REASONS:
    assert re.fullmatch(r"[a-z][a-z-]*", why), why
  # ...and the table in the doc lists all of them, because a closed set
  # nobody wrote down is an open one.
  from pathlib import Path
  doc = (Path(__file__).parent.parent / "docs" / "Overseer.md").read_text()
  for why in ov.FALLBACK_REASONS:
    assert f"`fallback:{why}`" in doc, f"{why} is not in docs/Overseer.md"


@pytest.mark.parametrize("boom, expect", [
  (RuntimeError("Connection reset by peer"), "offline"),
  (OSError("[Errno 111] Connection refused"), "offline"),
  (TimeoutError("timed out"), "timeout"),
  (ValueError("no JSON object could be decoded"), "garbled"),
  (KeyError("content"), "garbled"),
])
def test_every_transport_failure_buckets_to_a_token(boom, expect):
  assert ov.fallback_reason(boom) == expect
  assert expect in ov.FALLBACK_REASONS


def test_an_api_outage_is_not_what_the_robot_remembers():
  """THE acceptance case. An endpoint that raises, and then the two things
  `_after_decision` does with the summary: the status line every frame carries
  and the line written into `History.md`.

  Shown to fail before the fix by restoring
  `slot = {"error": f"{type(e).__name__}: {e}"[:200]}` in `Overseer._call` --
  both assertions then read `... [fallback:RuntimeError: Connection reset by
  peer]`.
  """
  life = _lifecycle("room_hub", errand=False)
  life.overseer = Overseer(_menu(), client=FakeClient(
    RuntimeError("Connection reset by peer")))
  said = narration(life)
  life._decide()

  assert life.decisions[-1]["source"] == "fallback:offline"
  decided = [ln for ln in said if ln.startswith("DECIDE")]
  assert decided, said
  for line in said:
    assert_sounds_like_a_robot(line)
  # ...and the robot's own paper trail, which is the half that outlives the
  # run. `_remember` writes History.md through the same summary.
  for line in life.thoughts.lines("History.md"):
    assert_sounds_like_a_robot(line)


def test_the_operator_still_gets_the_class_and_the_message():
  """The other half of the split, and the reason the fix is not simply
  deleting the detail: `Usage.errors` is where somebody debugging a dead
  endpoint looks, and "offline" alone would not tell them it was a DNS
  failure rather than a 401."""
  boss = Overseer(_menu(), client=FakeClient(
    RuntimeError("Connection reset by peer")))
  boss.decide({"decisions": 0})
  assert any("RuntimeError" in e and "Connection reset by peer" in e
             for e in boss.usage.errors), boss.usage.errors
  # ...exactly once. `_call` writes the detailed line and `_record` skips the
  # bucket for these two, or two entries per failure would push the detail out
  # of the five `stats()` shows.
  assert len(boss.usage.errors) == 1, boss.usage.errors
  assert boss.stats()["fallbacks"] == 1


def test_a_policy_fallback_is_still_not_an_incident():
  """The pre-existing rule, re-checked because #76 moved the list it lives in:
  `budget` and friends are the policy WORKING, and a run that filed every one
  of them as an error would read like an incident report."""
  boss = Overseer(_menu(), client=FakeClient(full(action="dance", reason=".")),
                  calls_per_hour=1)
  boss.decide({"decisions": 0})
  boss.decide({"decisions": 1})
  assert boss.decisions[-1].source == "fallback:budget"
  assert boss.usage.errors == []


# ---- 2. a recovered use-phase failure is not a traceback ---------------------


def test_a_use_phase_that_raises_is_narrated_as_a_sentence():
  """Nothing crashed: the tool goes back to its bay, the evaluator grades the
  job honestly and the loop carries on. Shown to fail before the fix by
  restoring `self._say(f"USE_TOOL FAILED: {used['error']} -- ...")`, which
  puts `IndexError: list index out of range` under the robot's portrait.
  """
  from pluggybot.mission.errand import Errand

  life = _lifecycle("room_hub", errand=False)

  def explodes(_life):
    raise IndexError("list index out of range")

  errand = Errand(name="carry:test", module="module_lcd", station_y=0.0,
                  use_at=(1.0, 1.0), use=explodes, needs_use_pose=False)
  life.mission.drive_to = lambda *a, **kw: True
  life.mission.swap_at_bay = lambda *a, **kw: None
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True,
                                                     "hung": True}
  said = narration(life)
  result = life.run_errand(errand)

  spoken = [ln for ln in said if "USE_TOOL FAILED" in ln]
  assert spoken, said
  for line in said:
    assert_sounds_like_a_robot(line)
  # The machine record is untouched -- scoring and `errand_results` read it,
  # and it is a dict field rather than something the robot says.
  assert result["error"] == "IndexError: list index out of range"
  # ...and the operator's console line still carries it, in the detail slot
  # that reaches `self.log` and nothing else.
  logged = [ln for ln in life.log if "USE_TOOL FAILED" in ln]
  assert "IndexError: list index out of range" in logged[0]
  assert "IndexError" not in spoken[0], "the detail rode the wire"


def test_the_neighbouring_lines_stopped_shouting():
  """Two lines over from the traceback, and the same audience: a visitor
  reading `NEVER GOT THERE -- BUT DROPPED THE TOOL` under a robot's portrait
  is being shouted at about a recovery."""
  from pluggybot.mission.errand import Errand

  life = _lifecycle("room_hub", errand=False)
  errand = Errand(name="carry:test", module="module_lcd", station_y=0.0,
                  use_at=(1.0, 1.0), use=None, needs_use_pose=False)
  life.mission.drive_to = lambda *a, **kw: False        # never got there
  life.mission.swap_at_bay = lambda *a, **kw: None
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False,
                                                     "hung": False}
  said = narration(life)
  life.run_errand(errand)
  arrival = [ln for ln in said if ln.startswith("USE_TOOL:")]
  assert arrival, said
  assert "NEVER GOT THERE" not in arrival[0] and "DROPPED" not in arrival[0]
  assert "never got there" in arrival[0] and "dropped the tool" in arrival[0]


# ---- 3. a visitor's rating that misses ---------------------------------------


def _rating(life, seq: int, quality: float = 0.8) -> list[str]:
  said = narration(life)
  life.inbox = Inbox()
  life.inbox.offer({"type": "rating", "seq": seq, "quality": quality}, t=0.0)
  life._visitor_step()
  return said


def test_a_rating_for_a_job_that_is_not_there_reads_as_a_sentence():
  """`str()` of a KeyError re-quotes its argument, so the old shared line read
  literally as `ignored: 'pluggybot: no ledger entry 7'` -- stray single quotes
  and the robot's own BODY name -- and it fires exactly when a person has just
  done something on the site. Shown to fail before the fix by restoring the
  single `except (KeyError, ValueError) as e: self._say(f"... {e}")`.
  """
  life = _lifecycle("room_hub", errand=False)
  life.ledger = Ledger()
  said = _rating(life, seq=7)

  spoken = [ln for ln in said if "ignored" in ln]
  assert spoken, said
  assert_sounds_like_a_robot(spoken[0])
  assert "'" not in spoken[0], f"re-quoted KeyError argument: {spoken[0]!r}"
  assert "pluggybot:" not in spoken[0], "the robot said its own body name"
  assert "7" in spoken[0], "the robot should still say which job it meant"
  # The evidence is still in the log, off the wire.
  logged = [ln for ln in life.log if "ignored" in ln]
  assert "KeyError" in logged[0] and "KeyError" not in spoken[0]


def test_a_rating_for_a_job_that_is_not_pending_says_so_differently():
  """The two misses are caught apart so the robot can say WHICH happened --
  'I have no job 7' and 'job 1 is not waiting on a rating' are different
  facts, and the website is holding a stale row in only one of them."""
  life = _lifecycle("room_hub", errand=False)
  life.ledger = Ledger()
  entry = life.ledger.award(evaluate("carry", {"picked": True, "stowed": True,
                                               "module": "module_lcd"}), t=1.0)
  said = _rating(life, seq=entry["seq"])   # a `carry` is auto-tiered, not pending

  spoken = [ln for ln in said if "ignored" in ln]
  assert spoken, said
  assert_sounds_like_a_robot(spoken[0])
  assert "not waiting on a rating" in spoken[0]
  logged = [ln for ln in life.log if "ignored" in ln]
  assert "ValueError" in logged[0] and "ValueError" not in spoken[0]


def _menu():
  from pluggybot.lifecycle import board_book
  return ov.Menu.for_world("room_hub", board_book("room_hub"))
