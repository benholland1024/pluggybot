"""Where the commissioned fixtures put the robot (issue #476): a look at the
dock's board or the rack's tags puts a drifted belief right, a fix past the
matcher's own search drops the map laid while lost, a fixture missing where
the belief puts it in plain sight is a robot lost -- looked for, never the
same walk again -- and each of them is a `drift` row on the wire. Pinned on
the served body with its decodes and fits stubbed, nothing stepped, and on
the stub's loop; what they measure is `scripts/drift_spike.py --looks` and
`--lived`."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.legs import dock as dk
from pluggybot.legs import fixtures as fx
from pluggybot.legs import rack as rk
from pluggybot.lifecycle import DRIFT_EVERY_S, QUAD_HOME, world_config
from pluggybot.mapping import scan_match as sm
from pluggybot.mind.thoughts import HISTORY
from pluggybot.navigator import wire_pose
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


@pytest.fixture(scope="module")
def quad_world():
  from pluggybot.legs import world as lw
  return lw.home_spec().compile()


def _quad(model):
  """The served body standing in the living room, a scan of it mapped;
  nothing stepped."""
  from pluggybot.legs.body import QuadMission
  m = QuadMission(model, mujoco.MjData(model), realtime=False,
                  grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  m.start_at(1.5, 0.5, 0.0)
  m._scan_step()
  return m


def _fit(m, fixture: str, pose, n: int = 6):
  """What a look from `pose` fits of a fixture: its commissioned pose in
  that pose's frame."""
  prior = m.dock_prior if fixture == "dock" else m.tool_rack_prior
  x, y, yaw = dk.relative(prior, pose)
  return (dk.DockFix if fixture == "dock" else rk.RackFix)(x, y, yaw, n, 0.002)


def _off(m, dx: float, dy: float, dth_deg: float = 0.0):
  x, y, th = m.true_pose()
  m.odo.correct(x + dx, y + dy, th + math.radians(dth_deg))


def _at_truth(m) -> bool:
  (bx, by, bth), (tx, ty, tth) = m.pose, m.true_pose()
  return math.hypot(bx - tx, by - ty) < 1e-6 and abs(dk._wrap(bth - tth)) < 1e-6


def _no_steps(result=True, into=None):
  """A routine that steps nothing: `result`, and its arguments into `into`."""
  def routine(*args, **kw):
    if into is not None:
      into.append(args)
    return result
    yield
  return routine


# ---- a fix --------------------------------------------------------------------


def test_a_look_at_a_fixture_puts_a_drifted_belief_right_and_lays_the_map_again(quad_world):
  # Off by 0.36 m and 2 deg, inside the matcher's own search: the belief
  # goes where the rack puts it, and the map is laid again round it
  m = _quad(quad_world)
  truth = m.true_pose()
  _off(m, 0.3, -0.2, 2.0)
  mapped = m.grid.grid.copy()
  rec = m.fixture_fix("rack", _fit(m, "rack", truth), 0.55)
  assert rec["why"] == "fixed" and rec["fixture"] == "rack" and rec["moved"]
  assert not rec["dropped"] and _at_truth(m)
  assert m.matcher.anchoring == sm.ANCHORED_SCANS
  assert np.array_equal(m.grid.grid, mapped), "a map within the matcher's reach went"
  assert m.belief_events[-1] is rec and rec["before"] != rec["after"]
  m.close()


@pytest.mark.parametrize("off", [(1.0, 0.5, 0.0), (0.0, 0.0, 8.0)])
def test_a_fix_past_the_matchers_search_drops_the_map_laid_while_lost(quad_world, off):
  # Past ASKEW_* the robot had been lost past what its map could put right;
  # kept, such a map relocated a corrected belief onto a copy within 20 s.
  # The map goes, and what was laid in it.
  m = _quad(quad_world)
  truth = m.true_pose()
  m.places.see(36, 25.0, 4.4, 0.0, 0.0)
  _off(m, *off)
  rec = m.fixture_fix("dock", _fit(m, "dock", truth, n=4), 0.7)
  assert rec["dropped"] and _at_truth(m)
  assert not m.grid.grid.any(), "the map laid while lost was kept"
  assert len(m.places) == 0 and m.matcher.anchoring == 0
  m.close()


def test_no_fix_off_too_few_tags_too_far_or_a_belief_already_right(quad_world):
  # A look is a fix only where it was measured to be one: a bay's pair
  # within NEAR_FIX_M, and a lost robot's four tags within FAR_FIX_M; and
  # none where the belief is within the matcher's own few centimetres
  m = _quad(quad_world)
  truth = m.true_pose()
  _off(m, 0.3, 0.0)
  before = m.pose
  assert m.fixture_fix("rack", _fit(m, "rack", truth, n=fx.NEAR_FIX_TAGS - 1), 0.5) is None
  assert m.fixture_fix("rack", _fit(m, "rack", truth), fx.NEAR_FIX_M + 0.1) is None
  assert m.pose == before and not m.belief_events
  m._seeking = True                                  # ...a lost robot's look
  assert m.fixture_fix("rack", _fit(m, "rack", truth, n=fx.FAR_FIX_TAGS - 1), 1.5) is None
  assert m.fixture_fix("rack", _fit(m, "rack", truth), fx.FAR_FIX_M + 0.1) is None
  assert m.pose == before
  assert m.fixture_fix("rack", _fit(m, "rack", truth, n=fx.FAR_FIX_TAGS), 1.5) is not None
  assert _at_truth(m)
  m._seeking = False
  _off(m, 0.06, 0.04, 1.5)
  before = m.pose
  assert m.fixture_fix("dock", _fit(m, "dock", truth, n=4), 0.6) is None
  assert m.pose == before
  m.close()


def test_lined_up_at_a_bay_the_rack_is_a_fix_and_never_on_the_way_there(quad_world, monkeypatch):
  # Lined up at a bay, one look was within 0.3 cm and 0.22 deg of the
  # truth; from an approach's start, up to 26 cm and 6.6 deg out, past the
  # line that drops a map, and a walk-in's looks steer. (The dock's fix is
  # its anchor, `test_the_dock_lain_on_drops_a_map_laid_while_lost`.)
  m = _quad(quad_world)
  truth = m.true_pose()
  fit = _fit(m, "rack", truth, n=4)
  m.detect_board = lambda: {}
  monkeypatch.setattr(rk, "fit_rack", lambda seen, spec: fit)
  monkeypatch.setattr(dk, "fit_dock", lambda seen: _fit(m, "dock", truth, n=4))
  m.fixture_fit = lambda f, seen: (fit, 0.55)
  _off(m, 0.4, 0.2)
  before = m.pose
  m.look_at_rack()
  assert m.run(m._find_rack_routine()) and m.run(m._find_board_routine())
  assert m.pose == before, "an approach's start or walk-in moved the belief"
  m._rack_walk_in_routine = _no_steps("stopped")
  m._drive_routine = _no_steps(None)
  m.bay_aim = lambda bay: rk.BayAim(0.45, 0.5, 0.0, 0.0)
  assert m.run(m._lined_up_routine(1, {})) is not None and _at_truth(m)
  assert math.dist(m.tool_rack_seen[:2], m.tool_rack_prior[:2]) < 1e-6, \
    "the rack believed in the old frame"
  m.close()


def test_the_dock_lain_on_drops_a_map_laid_while_lost(quad_world, monkeypatch):
  # The dock's own anchor puts the belief right as it always did (#42), and
  # a belief that far off is the map's too
  m = _quad(quad_world)
  m.detect_board = lambda: {}
  monkeypatch.setattr(dk, "fit_dock", lambda seen: dk.DockFix(0.0, 0.0, 0.0, 4, 0.002))
  m.odo.correct(m.dock_prior[0] + 2.0, m.dock_prior[1] - 1.0, m.dock_prior[2])
  m.anchor_at_dock()
  rec = m.belief_events[-1]
  assert rec["why"] == "fixed" and rec["fixture"] == "dock" and rec["dropped"]
  assert not m.grid.grid.any() and m.matcher.anchoring == 0
  m.close()


# ---- lost ---------------------------------------------------------------------


def test_a_fixture_missing_where_the_belief_puts_it_is_a_robot_lost(quad_world):
  # Its map goes, and it looks round for a fixture; with another robot
  # reported at the fixture, whose body can hide it, nothing goes
  m = _quad(quad_world)
  searched = []
  m.find_fixture_routine = _no_steps({"found": True}, searched)
  m.others = [lambda: m.fixture_face("dock")]
  mapped = m.grid.grid.copy()
  assert m.run(m.lost_routine("dock")) is False
  assert np.array_equal(m.grid.grid, mapped) and not searched and not m.belief_events
  m.others = []
  assert m.run(m.lost_routine("dock")) is True
  assert searched and not m.grid.grid.any()
  assert m.belief_events[-1]["why"] == "lost" and m.belief_events[-1]["fixture"] == "dock"
  m.close()


def test_a_lost_robots_search_walks_to_a_far_sighting_and_ends_at_a_fix(quad_world):
  # The rack sighted too far off to fix from is walked toward and looked at
  # from its approach; the look there fixes, and the search ends
  m = _quad(quad_world)
  truth = m.true_pose()
  _off(m, 1.8, -0.9)
  lost_at = m.pose
  looks, walked = [], []
  m._board_detector = lambda: SimpleNamespace(detect=lambda data: (looks.append(1), {})[1])
  far = rk.RackFix(*dk.relative(m.tool_rack_prior, truth), 6, 0.002)

  def fit(fixture, seen):
    if fixture != "rack":
      return None
    return (far, 3.6) if len(looks) == 1 else (_fit(m, "rack", truth), 1.3)

  m.fixture_fit = fit
  m.drive_to_routine = _no_steps(True, walked)
  m.face_routine = _no_steps(True)
  rec = m.run(m.find_fixture_routine(m.pose_xy(), 300.0))
  assert rec["found"] and rec["why"] == "found" and _at_truth(m)
  # ...to where it looks at the rack from, off the far sighting, in the map
  there = dk.compose(dk.compose(lost_at, (far.x, far.y, far.yaw)), fx.LOOK_FROM["rack"])
  assert len(walked) == 1 and math.dist(walked[0][:2], there[:2]) < 1e-6, walked
  assert [e["why"] for e in m.belief_events][-2:] == ["fixed", "searched"]
  assert m.belief_events[-1]["fixture"] == "rack" and m.belief_events[-1]["dropped"]
  assert not m._seeking, "a search left every decode trying the fixtures"
  m.close()


def test_a_lost_robots_search_walks_to_one_sighting_once(quad_world, monkeypatch):
  # A fixture seen no better from where its sighting sent the robot (three
  # tags at 1.5 m, every look) sends it nowhere new: walked to again, it
  # stood looking there until its patience ran out
  m = _quad(quad_world)
  truth = m.true_pose()
  walked = []

  def walk(x, y, timeout=90.0, stop=None, beyond=()):
    walked.append((x, y))
    assert len(walked) < 4, "walked to one sighting again and again"
    return True
    yield

  m._board_detector = lambda: SimpleNamespace(detect=lambda data: {})
  m.fixture_fit = lambda f, seen: (_fit(m, "rack", truth, n=3), 1.5) if f == "rack" else None
  m.drive_to_routine = walk
  m.face_routine = _no_steps(True)
  m._look_around_routine = _no_steps(False)
  m._search_map = lambda looked, near: (None,) * 5
  monkeypatch.setattr(fx, "next_viewpoint", lambda *a, **kw: None)
  rec = m.run(m.find_fixture_routine(m.pose_xy(), 300.0))
  assert len(walked) == 1 and rec["why"] == "not found" and not rec["found"]
  m.close()


def test_a_rack_not_there_is_searched_for_and_its_approach_walked_to_again(quad_world):
  # Found by a fixture, the walk to the approach and the look are made once
  # more; found nowhere, it says it saw no rack, never "no route"
  m = _quad(quad_world)
  walked, lost = [], []
  m.drive_to_routine = _no_steps(True, walked)
  m.face_routine = _no_steps(True)
  finds = iter([False, True])

  def find_rack():
    return next(finds)
    yield

  m._find_rack_routine = find_rack
  m.lost_routine = lambda fixture, stop=None: _no_steps(True, lost)(fixture)
  assert m.run(m._to_the_bay_routine(1, {})) == "ok"
  assert lost == [("rack",)] and len(walked) == 2
  walked.clear()
  lost.clear()
  finds = iter([False])
  m.lost_routine = lambda fixture, stop=None: _no_steps(False, lost)(fixture)
  assert m.run(m._to_the_bay_routine(1, {})) == "no rack"
  assert lost == [("rack",)] and len(walked) == 1
  m.close()


def test_a_charge_with_no_board_where_it_arrived_searches_and_docks_again():
  # Rowan walked to the same wrong place 31 times and died flat: no board
  # from the standoff the walk ARRIVED at is a loss, searched for, and the
  # standoff walked to again once a fixture put it right
  body = StubBody(rack=world_config("home_quad")["rack"])
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
  body = StubBody(rack=world_config("home_quad")["rack"])
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
  body = StubBody(rack=world_config("home_quad")["rack"])
  life = stub_life(body=body)
  docks.clear()
  body.dock_routine = lambda: _no_steps("no board")()
  assert life.go_charge() is False
  assert body.lost_at == ["dock"]


def test_no_board_from_near_the_standoff_is_a_loss_searched_once_a_charge():
  # Within NEAR_STANDOFF_M the board is in plain sight (#422), so none there
  # is a robot lost too: searched once a charge, and the walk's retries go
  # on after a search that found nothing
  body = StubBody(rack=world_config("home_quad")["rack"])
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
  body = StubBody(rack=world_config("home_quad")["rack"])
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
                                   (3.0, 2.0, 0.0), fixture="dock"))
  body.belief_events.append(_event(2, "fixed", (1.0, 2.0, 0.0), (3.0, 2.0, 0.3),
                                   (3.0, 2.0, 0.3), fixture="rack", moved=True,
                                   dropped=True))
  body.belief_events.append(_event(3, "fixed", (3.0, 2.0, 0.3), (3.1, 2.0, 0.3),
                                   (3.1, 2.0, 0.3), fixture="dock", moved=True,
                                   dropped=False))
  body.belief_events.append(_event(4, "fixed", (2.0, 2.0, 0.3), (3.1, 2.0, 0.3),
                                   (3.1, 2.0, 0.3), fixture="dock", anchor="seat",
                                   moved=True, dropped=True))
  life._belief_step()
  life._belief_step()                                # ...each once
  drift = [r for r in rows if r.get("type") == "drift"]
  assert [r["why"] for r in drift] == ["lost", "fixed", "fixed", "fixed"]
  assert drift[0]["errorM"] == 2.0 and drift[0]["pose"] == wire_pose((3.0, 2.0, 0.0))
  assert drift[1]["errorM"] == 0.0 and drift[1]["dropped"] and drift[1]["fixture"] == "rack"
  assert all(r["robot"] == life.root and "seq" not in r for r in drift)
  said = _history(life)
  assert any("the dock's board was not in sight" in ln for ln in said)
  assert any("the rack put me 2.0 m and 17 deg" in ln and "is gone" in ln for ln in said)
  assert any(ln.endswith("the dock put me 1.1 m and 0 deg from where I believed I was: "
                         "the map I had was laid askew, and it is gone") for ln in said), \
    "the seat is no board"
  assert len(said) == 3, "a fix within the map's reach went into History"
  # ...and the search after a loss ends in History, its fix's line or its own
  for seq, kw, words in ((5, dict(found=False), "saw neither"),
                         (6, dict(found=True, fixture="rack", dropped=False),
                          "found the rack, near where I believed I was"),
                         (7, dict(found=True, fixture="rack", dropped=True), None)):
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


def test_a_fetch_that_saw_no_rack_says_so_and_never_no_route():
  life = stub_life()
  said = life.pick_failure("module_pen", 0.0, "no rack")
  assert "the rack was not in sight" in said and "no route" not in said
