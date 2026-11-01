"""Where the commissioned fixture puts the robot (issue #476), the loop's
half: a charge that finds no station where the walk arrived is a robot
lost -- looked for, never the same walk again -- and every belief event is
a `drift` row on the wire, the ones the robot should know of in History.
Pinned on the stub's loop, nothing stepped. The humanoid's own looks, fixes
and search at its station are `test_station.py`'s; what they measure is
`scripts/station_spike.py --lost`."""

import math

from pluggybot.body import StubBody
from pluggybot.lifecycle import DRIFT_EVERY_S
from pluggybot.mind.thoughts import HISTORY
from pluggybot.navigator import wire_pose
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _no_steps(result=True, into=None):
  """A routine that steps nothing: `result`, and its arguments into `into`."""
  def routine(*args, **kw):
    if into is not None:
      into.append(args)
    return result
    yield
  return routine


# ---- lost ---------------------------------------------------------------------


def test_a_charge_with_no_board_where_it_arrived_searches_and_docks_again():
  # Rowan walked to the same wrong place 31 times and died flat: no board
  # from the standoff the walk ARRIVED at is a loss, searched for, and the
  # standoff walked to again once a fixture put it right
  body = StubBody()
  life = stub_life(body=body)
  docks = []

  def dock():
    docks.append(1)
    body.on_charger = len(docks) > 1
    return "no board" if len(docks) == 1 else "stopped"
    yield

  body.dock_routine = dock
  body.finds_fixture = True
  assert life.go_charge() is True
  assert body.lost_at == ["dock"] and len(docks) == 2
  assert body.went[-1] == body.went[-2], "the standoff was not walked to again"
  # ...and a walk back that gives up near the standoff lets the board decide,
  # as the first walk's does
  body = StubBody()
  life = stub_life(body=body)
  docks.clear()
  body.dock_routine = dock
  body.finds_fixture = True
  walks = []

  def walk(x, y, timeout=90.0, stop=None):
    walks.append((x, y))
    body.last_drive = {"why": "" if len(walks) == 1 else "stalled",
                       "goal": (float(x), float(y)), "seconds": 10.0,
                       "shortM": 0.0 if len(walks) == 1 else 0.3}
    return len(walks) == 1
    yield

  body.go_to_routine = walk
  assert life.go_charge() is True and len(docks) == 2 and len(walks) == 2
  # ...and found nowhere, the charge fails as it did, searched for once
  body = StubBody()
  life = stub_life(body=body)
  docks.clear()
  body.dock_routine = lambda: _no_steps("no board")()
  assert life.go_charge() is False
  assert body.lost_at == ["dock"]


def test_no_board_from_near_the_standoff_is_a_loss_searched_once_a_charge():
  # Within NEAR_STANDOFF_M the board is in plain sight (#422), so none there
  # is a robot lost too: searched once a charge, and the walk's retries go
  # on after a search that found nothing
  body = StubBody()
  life = stub_life(body=body)
  walks = []

  def short(x, y, timeout=90.0, stop=None):
    walks.append((x, y))
    body.last_drive = {"why": "stalled", "goal": (float(x), float(y)),
                       "seconds": 10.0, "shortM": 0.3}
    return False
    yield

  body.go_to_routine = short
  body.dock_routine = lambda: _no_steps("no board")()
  assert life.go_charge() is False
  assert body.lost_at == ["dock"] and len(walks) == 2
  # ...and a walk that gave up FAR off saw no board from anywhere it should:
  # no loss, the retries as before
  body = StubBody()
  life = stub_life(body=body)

  def far(x, y, timeout=90.0, stop=None):
    body.last_drive = {"why": "no_route", "goal": (float(x), float(y)),
                       "seconds": 1.0, "shortM": 6.0}
    return False
    yield

  body.go_to_routine = far
  assert life.go_charge() is False
  assert body.lost_at == []


# ---- on the wire, and in History -------------------------------------------------


def _event(seq, why, before, after, truth, **kw):
  return {"seq": seq, "t": 5.0, "why": why, "before": wire_pose(before),
          "after": wire_pose(after), "truth": wire_pose(truth), **kw}


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.texts[HISTORY].splitlines() if ln.strip()]


def test_every_belief_event_is_a_drift_row_and_a_loss_is_history():
  body = StubBody()
  life = stub_life(body=body)
  rows = []
  life.on_event.append(rows.append)
  life._drift_due = math.inf                         # no sample in the way
  body.belief_events.append(_event(1, "lost", (1.0, 2.0, 0.0), (1.0, 2.0, 0.0),
                                   (3.0, 2.0, 0.0), fixture="station"))
  body.belief_events.append(_event(2, "fixed", (1.0, 2.0, 0.0), (3.0, 2.0, 0.3),
                                   (3.0, 2.0, 0.3), fixture="station", moved=True,
                                   dropped=True))
  body.belief_events.append(_event(3, "fixed", (3.0, 2.0, 0.3), (3.1, 2.0, 0.3),
                                   (3.1, 2.0, 0.3), fixture="station", moved=True,
                                   dropped=False))
  life._belief_step()
  life._belief_step()                                # ...each once
  drift = [r for r in rows if r.get("type") == "drift"]
  assert [r["why"] for r in drift] == ["lost", "fixed", "fixed"]
  assert drift[0]["errorM"] == 2.0 and drift[0]["pose"] == wire_pose((3.0, 2.0, 0.0))
  assert drift[1]["errorM"] == 0.0 and drift[1]["dropped"] and drift[1]["fixture"] == "station"
  assert all(r["robot"] == life.root and "seq" not in r for r in drift)
  said = _history(life)
  assert any("my charging station was not in sight" in ln and "from nearer" in ln
             for ln in said)
  assert any("my charging station put me 2.0 m and 17 deg" in ln and "is gone" in ln
             for ln in said)
  assert len(said) == 2, "a fix within the map's reach went into History"
  # ...and the search after a loss ends in History, its fix's line or its own
  for seq, kw, words in ((5, dict(found=False, fixture="station"),
                          "I looked for my charging station for 156 s and did not see it"),
                         (6, dict(found=True, fixture="station", dropped=False),
                          "found my charging station, near where I believed I was"),
                         (7, dict(found=True, fixture="station", dropped=True), None)):
    body.belief_events.append(_event(seq, "searched", (3.1, 2.0, 0.3), (3.1, 2.0, 0.3),
                                     (3.1, 2.0, 0.3), seconds=156.4, **kw))
    life._belief_step()
    now = _history(life)
    assert (now[-1].endswith(words) if words else len(now) == len(said)), (seq, now[-1])
    said = now
  # ...and no pose of the truth's in what the robot reads
  assert not any("3.0" in ln for ln in said)


def test_the_belief_against_the_truth_is_sampled_with_the_verdicts_since():
  # Every DRIFT_EVERY_S on the physics seam, with the matcher's verdicts
  # since the last: a jump (`relocated`) and a slide (all `ok`) read apart
  body = StubBody(pose=(1.0, 2.0, 0.0))
  life = stub_life(body=body)
  assert life._belief_step in body.step_hooks
  rows = []
  life.on_event.append(rows.append)
  body.match_counts = {"ok": 7}                      # ...a restored life's own
  life._belief_step()                                # ...the first a period on
  body.match_counts = {"ok": 17}
  life.data.time += DRIFT_EVERY_S
  life._belief_step()
  body.match_counts = {"ok": 32, "relocated": 1}
  life.data.time += DRIFT_EVERY_S - 1.0
  life._belief_step()
  life.data.time += 1.0
  life._belief_step()
  body.match_counts = {"ok": 3}                      # a map forgotten: from zero
  life.data.time += DRIFT_EVERY_S
  life._belief_step()
  samples = [r for r in rows if r.get("type") == "drift" and r["why"] == "sample"]
  assert [s["matched"] for s in samples] == [{"ok": 10}, {"ok": 15, "relocated": 1},
                                             {"ok": 3}]
  assert samples[0]["errorM"] == 0.0 and samples[0]["believed"] == wire_pose((1.0, 2.0, 0.0))
  assert samples[0]["posture"] == "standing" and samples[0]["state"] == life.state
