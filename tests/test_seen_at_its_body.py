"""A robot standing or walking that the other's LIDAR sees is kept clear of
at its body (issue #675).

Every walk and every step keeps off the other robot where
`HubLifecycle.keep_clear` puts it, and standing or walking that was its
REPORTED pose: its belief, in its own map. Beside the other at the counter on
11-07 and 11-08 the two lay 0.49 m apart in the other's map at the median
(1.13 m at the 90th percentile), so the disc kept round a robot at work stood
where it was not -- a step refused for nothing, saying a distance that was
not true, and none refused where the robot was. Seen by the LIDAR (its scans'
split, #316), it is now placed at its body as the observer's own sensors put
it, as a robot resting already was (#455); out of sight, where it says.

Each rule pinned as cheaply as it fails:

  1. THE RULE, on the stub: seen, at its body as seen; out of sight, where
     it says; the sighting held `SEEN_HOLD_S`, and past it only while the
     robot has not moved since.
  2. THE SIGHTING: a scan names whom its band's returns came off
     (`Scan.peer_roots`), the navigator stamps it, and `seen_at` holds it
     `SEEN_HOLD_S` of its own scans.
  3. THE PLAN hears of a robot coming into sight, or leaving it, at once.
  4. THE WIRING, on the pair as built: a scan, the disc at the body through
     both drifts, a warp forgetting it, and a restart keeping it.

The day's numbers and the flights are in docs/SimNotes.md, "A robot at work
kept off where its body stands".
"""

import json
import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import KeepClear, StubBody
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.way import STILL
from pluggybot.navigator import DOWN_CHECK_S, SEEN_HOLD_S, Navigator
from pluggybot.perception.lidar import Lidar, Scan
from pluggybot.robot import FIRST, SECOND
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_downed_peer import STEPPER, _Falls
from test_resting_in_the_way import _believe, _close, _pair, _rest
from test_two_robots import W0, _stand
from test_two_robots import _pair as _pair_model


# ---- 1. the rule ---------------------------------------------------------------------


def _watched(now: float = 10.0):
  """A robot on the stub standing where it says it is, (1.0, 2.0), its body
  0.5 m off that, and a second body watching it whose own map is 0.1 m and
  0.2 m off the world (`as_seen`)."""
  life = stub_life()
  life.data.time = now
  life.body.x, life.body.y = 1.0, 2.0
  life.body.footprint_centre = lambda: (1.4, 2.3)
  eye = StubBody()
  eye.as_seen = lambda x, y: (x - 0.1, y + 0.2)
  return life, eye


def test_a_robot_standing_that_the_lidar_sees_is_kept_clear_of_at_its_body():
  """Out of sight, where it says it is, as every walk and step kept it; seen,
  where its body stands as the watcher's own sensors put it -- and said
  seen, so a plan hears of it (`KeepClear.seen`)."""
  life, eye = _watched()
  keep = life.keep_clear(seen_by=eye)
  assert (keep.x, keep.y) == (1.0, 2.0) and not keep.seen, "out of sight: where it says"
  eye.sightings[life.root] = 10.0
  keep = life.keep_clear(seen_by=eye)
  assert (keep.x, keep.y) == pytest.approx((1.3, 2.5)) and keep.seen
  assert not (keep.down or keep.resting or keep.dead) and keep.root == life.root
  # ...another robot's sighting is not this one's, and nobody watching sees nothing
  eye.sightings = {"r9_pluggybot": 10.0}
  assert not life.keep_clear(seen_by=eye).seen
  assert (life.keep_clear().x, life.keep_clear().y) == (1.0, 2.0)


def test_a_sighting_outlives_the_hold_only_while_the_robot_has_not_moved():
  """`SEEN_HOLD_S` of now, seen, moved or not; past it -- the watcher taking
  no scan since, a hand at work -- only a robot that has not moved since it
  was seen (`Body.moved_at`): one that walked off meanwhile would be placed
  where it went, unseen."""
  life, eye = _watched(now=10.0 + 0.5 * SEEN_HOLD_S)
  eye.sightings[life.root] = 10.0
  life.body.moved_at = 10.2
  assert life.keep_clear(seen_by=eye).seen, "within the hold, moving or not"
  life.data.time = 10.0 + SEEN_HOLD_S + 5.0
  assert not life.keep_clear(seen_by=eye).seen, "moved since, unseen"
  life.body.moved_at = 9.0
  keep = life.keep_clear(seen_by=eye)
  assert keep.seen and (keep.x, keep.y) == pytest.approx((1.3, 2.5)), "still since"


def test_a_robot_down_resting_dead_or_stepping_aside_is_where_it_was():
  """Nothing else moves: lying, resting, dead or stepping aside it is at
  its body seen or not (#365, #455), and says why, never `seen`."""
  life, eye = _watched()
  life.body.resting = True
  keep = life.keep_clear(seen_by=eye)
  assert keep.resting and not keep.seen and (keep.x, keep.y) == pytest.approx((1.3, 2.5))
  life.body.resting = False
  life.body.making_way = "r2_pluggybot"
  keep = life.keep_clear(seen_by=eye)
  assert not keep.seen and (keep.x, keep.y) == pytest.approx((1.3, 2.5))


# ---- 2. the sighting ------------------------------------------------------------------


def test_a_scan_names_the_robot_its_returns_in_the_band_came_off():
  """`Lidar.band`: each robot its returns in the band came off, by its root
  (the split's labels, #316) -- one a metre ahead is named, one past the
  house's walls is not, and a LIDAR told of nobody names nobody."""
  model = _pair_model(second_at=(2.5, 0.5))
  data = mujoco.MjData(model)
  _stand(model, data, FIRST, 1.5, 0.5, 0.0)
  _stand(model, data, SECOND, 2.5, 0.5, 0.0)
  lidar = Lidar(model, seed=3)
  scan = lidar.band(data, np.eye(3), hm.stand_height(), joints=W0)
  assert scan.peer_roots == (), "told of nobody"
  lidar.exclude_robot(SECOND.root)
  scan = lidar.band(data, np.eye(3), hm.stand_height(), joints=W0)
  assert len(scan.peer_ranges) and scan.peer_roots == (SECOND.root,)
  _stand(model, data, SECOND, -8.0, 4.0, 0.0)
  scan = lidar.band(data, np.eye(3), hm.stand_height(), joints=W0)
  assert not len(scan.peer_ranges) and scan.peer_roots == ()


def _navigator(roots: list):
  """The scan step's own `self`, every scan naming `roots[0]`."""
  nav = object.__new__(Navigator)
  nav.data, nav._next_scan, nav.backoff_until = NS(time=3.0), 0.0, 99.0
  nav.LIDAR_ORIGIN, nav.peer_returns = (0.0, 0.0), None
  nav.scanned_t, nav.seen_t = None, {}
  nav._scan = lambda: Scan(np.array([0.0]), np.array([2.0]), np.array([0.1]),
                           np.array([0.6]), peer_roots=roots[0])
  nav.maps = lambda: False
  nav.matcher = None
  return nav


def test_the_scans_stamp_whom_they_saw_and_a_sighting_holds_of_the_scans_since():
  """The wire from the LIDAR to `seen_at`: each scan stamps the robots it
  named; one missed by every scan since for `SEEN_HOLD_S` is out of sight --
  a frame or a few that miss it (the watcher's own fin, an arm, the part's
  dropouts) move nothing -- and one never named never was. Standing still
  the LIDAR takes no scan, and its newest scans' word stands."""
  roots = [("r2_pluggybot",)]
  nav = _navigator(roots)
  assert nav.seen_at("r2_pluggybot") is None, "never scanned"
  Navigator._scan_step(nav)
  assert nav.scanned_t == 3.0 and nav.seen_at("r2_pluggybot") == 3.0
  assert nav.seen_at("pluggybot") is None
  roots[0] = ()
  for k in range(1, round(SEEN_HOLD_S / 0.1)):
    nav.data.time, nav._next_scan = 3.0 + 0.1 * k, 0.0
    Navigator._scan_step(nav)
    assert nav.seen_at("r2_pluggybot") == 3.0, f"missed {0.1 * k:.1f} s, within the hold"
  nav.data.time, nav._next_scan = 3.05 + SEEN_HOLD_S, 0.0
  Navigator._scan_step(nav)
  assert nav.seen_at("r2_pluggybot") is None, "missed past the hold"
  roots[0] = ("r2_pluggybot",)
  nav.data.time, nav._next_scan = 3.15 + SEEN_HOLD_S, 0.0
  Navigator._scan_step(nav)
  assert nav.seen_at("r2_pluggybot") == pytest.approx(3.15 + SEEN_HOLD_S)


# ---- 3. the plan -------------------------------------------------------------------------


def test_a_robot_coming_into_sight_or_leaving_it_is_planned_round_at_once():
  """Seen, its disc moves from where it says it is to where its body
  stands -- half a metre at the counter -- so the plan hears of it within
  `DOWN_CHECK_S`, as of a fall (#365), not at its next 2 s replan."""
  seen, flips = [False], []

  def flip() -> None:                        # seen at 0.5 s, out of sight at 1.0 s
    due = (0.5, 1.0)
    if len(flips) < 2 and walk.data.time >= due[len(flips)]:
      seen[0] = not seen[0]
      flips.append(float(walk.data.time))

  walk = _Falls(flip)
  walk.others = [lambda: KeepClear(0.5, 1.5, seen=seen[0])]   # off the path
  tick.run(STEPPER, walk.drive_to_routine(20.0, 0.0, 1.5))
  assert len(flips) == 2
  for when in flips:
    after = [t for t in walk.plans if t >= when]
    assert after and after[0] - when <= DOWN_CHECK_S + 0.011, \
      f"planned {[round(t, 2) for t in walk.plans]}, flipped at {when:.2f}"
  assert len(walk.plans) == 3, "the premise: no routine replan in 1.5 s"


# ---- 4. the wiring -----------------------------------------------------------------------


def test_the_pair_keeps_clear_of_a_robot_its_lidar_saw_where_its_body_stands(tmp_path):
  """The pair as built (`pair.build_pair` hands each mission `keep_clear(
  seen_by=)`): Luca's scan sees Rowan standing 1.8 m off, its belief 2.9 m
  off its body, and keeps clear of its body, placed through Luca's own map
  drifted as on 10-08 -- where before it kept clear of Rowan's word. A warp
  forgets what it saw, and a restart keeps it."""
  me, peer = _pair(tmp_path)
  try:
    _rest(peer, resting=False)
    _believe(peer, 0.0, 2.9)
    keep = me.body.others[0]()
    assert not keep.seen and (keep.x, keep.y) == pytest.approx(peer.body.pose_xy()), \
      "the premise: unseen, where it says it is"
    m = me.body.mission
    m._next_scan = 0.0
    m._scan_step()
    assert me.body.seen_at(peer.root) == float(me.data.time)
    keep = me.body.others[0]()
    assert keep.seen
    assert (keep.x, keep.y) == pytest.approx(me.body.as_seen(*peer.body.footprint_centre()),
                                             abs=1e-9)
    assert math.dist((keep.x, keep.y), peer.body.pose_xy()) > 2.5
    _believe(me, 1.2, -2.9, math.radians(-9.5))
    keep = me.body.others[0]()
    seen = me.body.as_seen(*peer.body.footprint_centre())
    assert keep.seen and (keep.x, keep.y) == pytest.approx(seen, abs=1e-9)
    assert math.dist(seen, peer.body.footprint_centre()) > 2.5, "the premise: Luca's drift"
    # ...a restart keeps whom it saw, and when, and when the other last went
    kept, arrays = m.kept_state()
    kept = json.loads(json.dumps(kept))
    t, saw = m.scanned_t, dict(m.seen_t)
    m.scanned_t, m.seen_t = None, {}
    m.restore_kept(kept, arrays)
    assert (m.scanned_t, m.seen_t) == (t, saw)
    pm = peer.body.mission
    went = pm.went_t
    pm.went_t = went - 1.0
    kept, arrays = pm.kept_state()
    pm.went_t = 99.0
    pm.restore_kept(json.loads(json.dumps(kept)), arrays)
    assert pm.went_t == went - 1.0
    # ...its LIDAR idle past the hold, a hand at work taking no scan: the
    # newest scans' word stands while the other has not gone anywhere since
    me.data.time += SEEN_HOLD_S + 5.0
    assert me.body.others[0]().seen
    # ...a walk command is going somewhere, and a hand at rest is not
    pm.legs = "walk"
    pm._before_step(STILL)
    assert peer.body.moved_at == went - 1.0
    pm._before_step((0.2, 0.0, 0.0))
    assert peer.body.moved_at == float(me.data.time) and not me.body.others[0]().seen
    pm.went_t = went
    assert me.body.others[0]().seen
    # ...so is a warp: placed by an old sighting, it would be where it was put
    x, y, th = peer.body.true_pose()
    peer.body.start_at(x - 0.5, y, th)
    assert peer.body.moved_at == float(me.data.time) and not me.body.others[0]().seen
    # ...and its own warp forgets what it saw
    me.data.time -= SEEN_HOLD_S + 5.0
    peer.body.start_at(x, y, th)
    pm.went_t = 0.0
    assert me.body.others[0]().seen
    x, y, th = me.body.true_pose()
    me.body.start_at(x, y, th)
    assert me.body.seen_at(peer.root) is None and not me.body.others[0]().seen
  finally:
    _close(me, peer)
