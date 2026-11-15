"""Scan matching (issue #386), pinned on synthetic scans: rooms and
corridors drawn as segments, scans cast analytically with the LIDAR's own
noise, maps fused from known poses, and the body's wiring called at its
seam -- the humanoid standing in its house, its own LIDAR's scans. Nothing
here flies."""

import copy
import gc
import math
import time

import numpy as np
import pytest

from pluggybot.mapping import scan_match as sm
from pluggybot.mapping.occupancy_grid import OccupancyGrid

ANGLES = np.linspace(-math.pi, math.pi, 360, endpoint=False)


def cast(segments, pose, rng=None, max_range=8.0):
  """Ranges from the pose (the sensor at the pose point) to the nearest
  segment, with the LIDAR model's noise (10 mm + 1 % of range) if `rng`."""
  x, y, th = pose
  a = th + ANGLES
  dx, dy = np.cos(a), np.sin(a)
  best = np.full(len(a), max_range)
  for x1, y1, x2, y2 in segments:
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
      t = ((x1 - x) * ey - (y1 - y) * ex) / den
      u = ((x1 - x) * dy - (y1 - y) * dx) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    best = np.where(ok & (t < best), t, best)
  if rng is not None:
    hit = best < max_range
    best = np.where(hit, best + rng.normal(0.0, 0.010 + 0.01 * best), best)
  return np.clip(best, 0.02, max_range)


def box(x0, y0, x1, y1):
  return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


#: A 6 x 4 m room with three stubs, its walls off the cell boundaries.
ROOM = box(0.013, 0.021, 6.013, 4.021) + [
  (2.013, 0.021, 2.013, 1.5), (4.013, 4.021, 4.013, 2.5), (1.0, 3.011, 1.6, 3.011)]
#: A 1.5 m corridor longer than the LIDAR reaches either way.
CORRIDOR = [(-30.0, 0.0, 40.0, 0.0), (-30.0, 1.5, 40.0, 1.5)]


def mapped(segments, bounds, poses, seed=1):
  rng = np.random.default_rng(seed)
  g = OccupancyGrid(*bounds, resolution=0.05)
  for p in poses:
    g.update(p, ANGLES, cast(segments, p, rng), 8.0, origin=(0.0, 0.0))
  return g


@pytest.fixture(scope="module")
def room():
  rng = np.random.default_rng(0)
  poses = ([(rng.uniform(0.5, 1.8), rng.uniform(0.5, 3.5), rng.uniform(-3, 3)) for _ in range(20)]
           + [(rng.uniform(2.3, 5.5), rng.uniform(0.3, 3.7), rng.uniform(-3, 3)) for _ in range(20)])
  return mapped(ROOM, (-1, -1, 7, 5), poses)


@pytest.fixture(scope="module")
def corridor():
  return mapped(CORRIDOR, (-25, -1, 35, 3), [(x, 0.75, 0.0) for x in np.arange(-12, 20, 0.5)])


TRUE = (3.2, 1.8, 0.3)


def test_one_scan_recovers_a_known_offset_in_a_known_room(room, monkeypatch):
  # Without odometry's prior, the fit alone: 5 cm, 3 cm and 1.1 deg off,
  # back to the pose the map says -- to within the map's own quantisation
  # (a wall's surface sits a quarter of a cell into the cell it marks).
  monkeypatch.setattr(sm, "PRIOR_XY", 0.0)
  monkeypatch.setattr(sm, "PRIOR_TH", 0.0)
  m = sm.ScanMatcher(room)
  ranges = cast(ROOM, TRUE, np.random.default_rng(7))
  got = m.match((TRUE[0] + 0.05, TRUE[1] - 0.03, TRUE[2] + 0.02), ANGLES, ranges)
  assert got.accepted and got.why == "ok"
  assert math.hypot(got.pose[0] - TRUE[0], got.pose[1] - TRUE[1]) < 0.02
  assert abs(math.degrees(got.pose[2] - TRUE[2])) < 0.2
  assert got.held == 0


def test_odometrys_prior_damps_a_scan_and_successive_scans_converge(room):
  # With the prior, one scan moves the pose PART of the way -- the map's
  # quantisation must not land in the pose whole each scan -- and a robot
  # standing still converges over a second's worth of scans.
  m = sm.ScanMatcher(room)
  rng = np.random.default_rng(11)
  start = (TRUE[0] + 0.05, TRUE[1] - 0.03, TRUE[2] + 0.02)
  first = m.match(start, ANGLES, cast(ROOM, TRUE, rng))
  moved = math.hypot(first.pose[0] - start[0], first.pose[1] - start[1])
  assert first.accepted and 0.005 < moved < 0.05
  pose = first.pose
  for _ in range(10):
    pose = m.match(pose, ANGLES, cast(ROOM, TRUE, rng)).pose
  assert math.hypot(pose[0] - TRUE[0], pose[1] - TRUE[1]) < 0.02
  assert abs(math.degrees(pose[2] - TRUE[2])) < 0.2


def test_a_corridor_is_fixed_across_and_held_along(corridor, monkeypatch):
  # A straight corridor constrains the robot across it and not along it:
  # the fit is degenerate along x, and there the pose stays odometry's
  # rather than sliding to wherever the jags of the cells put it.
  monkeypatch.setattr(sm, "PRIOR_XY", 0.0)
  monkeypatch.setattr(sm, "PRIOR_TH", 0.0)
  m = sm.ScanMatcher(corridor)
  true = (4.0, 0.7, 0.05)
  ranges = cast(CORRIDOR, true, np.random.default_rng(3))
  guess = (true[0] + 0.10, true[1] + 0.05, true[2] + 0.03)
  got = m.match(guess, ANGLES, ranges)
  assert got.accepted and got.held == 1
  assert abs(got.weak_dir[0]) > 0.95                  # the weak direction is x
  assert abs(got.pose[0] - guess[0]) < 0.005          # held along
  assert abs(got.pose[1] - true[1]) < 0.01            # fixed across
  assert abs(math.degrees(got.pose[2] - true[2])) < 0.2


def test_a_fit_that_would_slide_past_the_gate_is_refused_unless_the_search_finds_it(
    room, monkeypatch):
  # 17 cm off is a pose the fit can pull back, but not in one scan: a
  # correction past MAX_STEP may be another wall to sit on, so the fit's is
  # never taken -- the bounded search's is, where the walls agree with it
  # (DOUBT_FOUND_SHARE), and otherwise the scan is refused.
  ranges = cast(ROOM, TRUE, np.random.default_rng(5))
  guess = (TRUE[0] + 0.14, TRUE[1] - 0.10, TRUE[2] + 0.08)
  got = sm.ScanMatcher(room).match(guess, ANGLES, ranges)
  assert got.why == "found"
  assert math.hypot(got.pose[0] - TRUE[0], got.pose[1] - TRUE[1]) < 0.02
  monkeypatch.setattr(sm, "DOUBT_FOUND_SHARE", 1.01)
  got = sm.ScanMatcher(room).match(guess, ANGLES, ranges)
  assert not got.accepted and got.why == "slid" and not got.fuse
  assert got.pose == guess


#: A storeroom as the counter has it (#559, T5's night of 11-10): 6 x 6 m, and
#: the robot 0.6 m from its east wall facing it, so that the returns ahead
#: fall under MATCH_MIN_M and the long north and south walls make most of the
#: scan, as the counter's own returns are left out at a work spot.
STORE = box(0.013, 0.021, 6.013, 6.021)
AT_COUNTER = (5.4, 3.02, 0.0)


@pytest.fixture(scope="module")
def store():
  rng = np.random.default_rng(4)
  poses = [(rng.uniform(0.6, 5.4), rng.uniform(0.6, 5.4), rng.uniform(-3, 3)) for _ in range(40)]
  return mapped(STORE, (-1, -1, 7, 7), poses)


@pytest.mark.parametrize("off", [0.3, -0.3])
def test_a_fit_its_side_walls_hold_while_its_end_walls_disagree_is_searched(store, off):
  # MEASURED at the counter (T5): the side walls stay on their walls whatever
  # the offset along them, so a fit 0.3-0.4 m off was judged `ok` and its
  # scan laid in, and the copy it laid kept the error for hours. Doubted, the
  # search takes the pose the end walls agree with, and the scan is not laid
  # at the pose it disagreed with.
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(9))
  guess = (AT_COUNTER[0] + off, AT_COUNTER[1], AT_COUNTER[2])
  got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
  assert got.why == "found" and got.fuse, got
  assert abs(got.pose[0] - AT_COUNTER[0]) < 0.05, got


def test_a_doubted_fits_take_is_judged_at_its_own_bar(store, monkeypatch):
  # ...at DOUBT_FOUND_SHARE, never the base search's FOUND_SHARE: at 0.75 a
  # take moved a true pose over 0.1 m at 11 places of 7,840 (the hall's
  # stairs), at 0.9 at the base's own 2 (#681's review: no test read it).
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(9))
  guess = (AT_COUNTER[0] + 0.3, AT_COUNTER[1], AT_COUNTER[2])
  monkeypatch.setattr(sm, "DOUBT_FOUND_SHARE", 1.01)
  got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
  assert got.why == "doubted", got


#: A corridor fenced in panels, both sides (#681's sidewalks; D4, the day
#: shift of 11-11): 0.7 m of fence and a 0.3 m gap every metre, so a place a
#: metre along explains a scan as well as the truth does.
PANELS = [(x, y, x + 0.7, y) for x in np.arange(-30.0, 40.0) + 0.013 for y in (0.0, 1.5)]


@pytest.fixture(scope="module")
def panelled():
  return mapped(PANELS, (-25, -1, 35, 3), [(x, 0.75, 0.0) for x in np.arange(-12, 20, 0.5)])


#: The panels' ends fix the fit along the corridor at 0.2 of its points, so
#: these doubt it at a bar over that: what is pinned is what the doubt's
#: search does with a fit, whatever doubted it (the sidewalks' were held).
PANELS_WEAK = 0.3


def test_a_doubted_fits_search_takes_no_place_another_explains_as_well(panelled, monkeypatch):
  # MEASURED (the drift flight seed 19 from the hall): along a sidewalk every
  # place scored 0.95-1.00 of the search's best, and taking the best, a few
  # points ahead each time, walked the belief 5.95 m off. Here the best is
  # the place a metre behind the truth, and the truth, 0.4 m ahead of the
  # belief, explains the scan as well: neither is taken, and the fit is as
  # it would have been -- its scan laid in, as an `ok` one is (not laid in,
  # doubted scans starved the map of new ground on the way to the mall).
  monkeypatch.setattr(sm, "DOUBT_WEAK_SHARE", PANELS_WEAK)
  true = (4.0, 0.75, 0.0)
  ranges = cast(PANELS, true, np.random.default_rng(3))
  guess = (true[0] - 0.4, true[1], true[2])
  got = sm.ScanMatcher(panelled).match(guess, ANGLES, ranges)
  assert got.why == "doubted" and got.accepted and got.fuse, got
  assert abs(got.pose[0] - guess[0]) < 0.05, got


def test_a_doubted_fit_the_search_finds_nothing_for_keeps_its_own_correction(panelled, monkeypatch):
  # ...found nothing, a doubted fit is the fit it was, correction and all:
  # 0.4 m behind the truth and 1.7 deg turned, the panels fix its heading,
  # and the fit's turn is taken as an `ok` one's would be. Laid at the pose
  # asked about instead, every doubted scan -- 229 of ~720 along the mall's
  # sidewalk -- would lose its correction across and its turn (#681's review).
  monkeypatch.setattr(sm, "DOUBT_WEAK_SHARE", PANELS_WEAK)
  true = (4.0, 0.75, 0.0)
  ranges = cast(PANELS, true, np.random.default_rng(3))
  guess = (true[0] - 0.4, true[1], true[2] + 0.03)
  got = sm.ScanMatcher(panelled).match(guess, ANGLES, ranges)
  assert got.why == "doubted" and got.fuse, got
  assert abs(math.degrees(got.pose[2] - true[2])) < 0.3, got
  assert abs(got.pose[0] - guess[0]) < 0.05, got


def test_only_the_base_search_takes_the_best_of_two_places_alike(room, monkeypatch):
  # Which search a fit gets is the rule (#681): an inconsistent fit the
  # base's, whose best is taken as it always was; a slid or a doubted one the
  # doubt's, whose best is taken only where no place UNIQUE_M off explains
  # the scan as well (`unique`). Pinned at the wiring: on one noise draw in
  # the panelled corridor a slid fit's refusal hung on one point (#681's
  # review).
  calls = []
  real = sm.ScanMatcher._search

  def spy(self, pose, pts, **kw):
    calls.append(kw.get("unique", False))
    return real(self, pose, pts, **kw)

  monkeypatch.setattr(sm.ScanMatcher, "_search", spy)
  slid = (TRUE[0] + 0.14, TRUE[1] - 0.10, TRUE[2] + 0.08)
  got = sm.ScanMatcher(room).match(slid, ANGLES, cast(ROOM, TRUE, np.random.default_rng(5)))
  assert got.why == "found" and calls == [True], (got, calls)
  calls.clear()
  far = (TRUE[0] + 0.30, TRUE[1] - 0.26, TRUE[2] + 0.04)
  got = sm.ScanMatcher(room).match(far, ANGLES, cast(ROOM, TRUE, np.random.default_rng(21)))
  assert got.why == "found" and calls == [False], (got, calls)


def test_a_straight_corridor_held_along_is_not_doubted(corridor):
  # ...where nothing disagrees: a corridor's fit held along it is laid in,
  # 0.3 m off along it as anywhere (no end in reach says otherwise).
  true = (4.0, 0.7, 0.0)
  ranges = cast(CORRIDOR, true, np.random.default_rng(3))
  got = sm.ScanMatcher(corridor).match((true[0] + 0.3, true[1], true[2]), ANGLES, ranges)
  assert got.why == "ok" and got.held == 1 and got.fuse


def test_a_fit_held_along_with_a_tenth_of_its_points_off_the_walls_is_doubted(corridor):
  # ...and where something does: a held fit with 0.89-0.95 of its points on
  # walls is searched (DOUBT_HELD_SHARE), as the sidewalks' were on the drift
  # flight 19h (#681's review: no test read the bar). Here a post the map
  # never saw takes a tenth of the scan; nothing along the corridor explains
  # it better, so the fit is what it was, held and laid in.
  true = (4.0, 0.7, 0.0)
  ranges = cast(CORRIDOR + [(5.5, 0.5, 5.5, 1.0)], true, np.random.default_rng(3))
  m = sm.ScanMatcher(corridor)
  got = m.match((true[0] + 0.3, true[1], true[2]), ANGLES, ranges)
  assert got.why == "doubted" and got.held == 1 and got.fuse, got


#: The free_from the band lays a scan from: about 1 m ahead, and 2.8 m
#: behind, where the head's back hides the rings under -26 deg.
BAND_FF = np.where(np.abs(ANGLES) < math.pi / 2, 1.0, 2.8)


@pytest.fixture(scope="module")
def store_copy(store):
  """The storeroom with a copy of its counter's end laid 0.3 m along it, as
  a matcher that kept a belief 0.3 m off laid it (#681's review: the copies
  the live maps hold): 72 scans from 24 work poses round the counter, each
  laid where the belief put it, its rays clearing what they crossed."""
  g = copy.deepcopy(store)
  rng = np.random.default_rng(5)
  work = [(rng.uniform(5.0, 5.4), AT_COUNTER[1] + rng.normal(0, 0.15), rng.uniform(-0.6, 0.6))
          for _ in range(24)]
  noise = np.random.default_rng(11)
  for p in work:
    for _ in range(3):
      g.update((p[0] + 0.3, p[1], p[2]), ANGLES, cast(STORE, p, noise), 8.0, origin=(0.0, 0.0),
               free_from=BAND_FF)
  return g, [AT_COUNTER] + work[:6]


def test_a_copy_of_the_room_in_the_map_does_not_capture_a_right_belief(store_copy):
  # #681's review: where a copy's scans cleared the true walls' cells, a fit
  # at the truth read 0.77-0.80 on walls; doubted for being under 0.8, its
  # search took the copy, which explains more (11-12 work poses of 12 on the
  # review's synthetic counter; 35 of 40 poses round the counter on Luca's
  # live map of 11-09, the base none). A fit with every direction fixed by
  # many points is not doubted, whatever its share: the copy pulls no right
  # belief the base's fit would have kept.
  g, poses = store_copy
  for k, p in enumerate(poses):
    ranges = cast(STORE, p, np.random.default_rng(30 + k))
    m = sm.ScanMatcher(g)
    got = m.match(p, ANGLES, ranges)
    assert got.why == "ok" and abs(got.pose[0] - p[0]) < 0.05, (k, got)
  # ...the case it pins: at the counter the copy explains the scan better
  m = sm.ScanMatcher(g)
  m.refresh(*AT_COUNTER[:2])
  pts = m.points(ANGLES, cast(STORE, AT_COUNTER, np.random.default_rng(30)))
  d, _, _, _, live = m.distance(AT_COUNTER, pts)
  assert (np.abs(d[live]) < sm.INLIER_M).mean() < 0.8
  start = m._search(AT_COUNTER, pts, unique=True)
  assert start is not None and abs(start[0] - AT_COUNTER[0] - 0.3) < 0.06, start


def test_a_full_fit_is_doubted_where_its_weakest_direction_rests_on_few_points(
    store_copy, monkeypatch):
  # ...and one whose weakest direction rests on under DOUBT_WEAK_SHARE of its
  # points is doubted as a held one is: MEASURED at the counter, a belief
  # 0.2-0.4 m off toward it is often a full fit, the counter's front, its
  # props and the wall behind "fixing" the direction at 0.05-0.19 of its
  # points (median 0.075), where a right belief beside a copy reads 0.12-0.29
  # (Luca's live map of 11-09) and the tests' copies 0.17-0.30. Pinned at the
  # bar: the right belief by the copy is doubted (and searched onto the copy)
  # with the bar just over its weakest direction's share, and not just under.
  g, _ = store_copy
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(30))
  m = sm.ScanMatcher(g)
  m.refresh(*AT_COUNTER[:2])
  pts = m.points(ANGLES, ranges)
  fit = m._fit(AT_COUNTER, pts)
  _, why, kw = m._judge(AT_COUNTER, pts, fit)
  weak = kw["weakest"] / float(fit[2].sum())
  assert why == "ok" and kw["held"] == 0 and weak > sm.DOUBT_WEAK_SHARE, (why, kw, weak)
  monkeypatch.setattr(sm, "DOUBT_WEAK_SHARE", weak + 0.01)
  got = sm.ScanMatcher(g).match(AT_COUNTER, ANGLES, ranges)
  assert got.why == "found" and abs(got.pose[0] - AT_COUNTER[0] - 0.3) < 0.06, got
  monkeypatch.setattr(sm, "DOUBT_WEAK_SHARE", weak - 0.01)
  assert sm.ScanMatcher(g).match(AT_COUNTER, ANGLES, ranges).why == "ok"


def test_the_doubts_search_takes_no_pose_it_would_doubt_again(store, monkeypatch):
  # #681's review of 11-13: the take was judged at DOUBT_FOUND_SHARE with
  # every direction fixed, never by the doubt's own test, so a pose whose
  # weakest direction rests on a few aliasing points, at 0.90-0.95 on walls,
  # could be taken, and the next scan from it doubted and searched again.
  # Pinned at the bars: the storeroom's take-back, its found fit put just
  # under DOUBT_WEAK_SHARE and DOUBT_HELD_SHARE, is refused (doubted, as it
  # was), and just over either, taken.
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(9))
  guess = (AT_COUNTER[0] + 0.3, AT_COUNTER[1], AT_COUNTER[2])
  m = sm.ScanMatcher(store)
  m.refresh(*guess[:2])
  pts = m.points(ANGLES, ranges)
  start = m._search(guess, pts, unique=True)
  fit = m._fit(start, pts)
  at, why, kw = m._judge(guess, pts, fit, found=True, doubt=True)
  live = float(fit[2].sum())
  weak, share = kw["weakest"] / live, kw["inliers"] / live
  assert why == "found" and weak > sm.DOUBT_WEAK_SHARE, (why, weak, share)
  monkeypatch.setattr(sm, "DOUBT_HELD_SHARE", share + 0.01)
  monkeypatch.setattr(sm, "DOUBT_WEAK_SHARE", weak + 0.01)
  got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
  assert got.why == "doubted" and abs(got.pose[0] - guess[0]) < 0.05, got
  for name, bar in (("DOUBT_WEAK_SHARE", weak - 0.01), ("DOUBT_HELD_SHARE", share - 0.01)):
    with monkeypatch.context() as patch:
      patch.setattr(sm, name, bar)
      got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
      assert got.why == "found" and abs(got.pose[0] - AT_COUNTER[0]) < 0.05, (name, got)


def test_the_doubts_search_takes_only_a_place_that_explains_the_scan_much_better(
    store, monkeypatch):
  # MEASURED (#681's third round, a drift flight's captured takes): on open
  # ground a weakly fixed fit's search found places 1.00-1.11 as good as the
  # pose asked about, and four takes in 5 s moved the belief 10 cm the wrong
  # way and 9 cm back; the counter's take-backs gained 1.25-2.1. Pinned at
  # the bar: the storeroom's take-back, refused with DOUBT_GAIN just over
  # its gain, taken just under.
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(9))
  guess = (AT_COUNTER[0] + 0.3, AT_COUNTER[1], AT_COUNTER[2])
  m = sm.ScanMatcher(store)
  m.refresh(*guess[:2])
  pts = m.points(ANGLES, ranges)
  on_wall = np.abs(np.nan_to_num(m.field, nan=np.inf)) < sm.INLIER_M
  start = m._search(guess, pts, unique=True)
  gain = m._on_wall_count(start, pts, on_wall) / m._on_wall_count(guess, pts, on_wall)
  assert gain > sm.DOUBT_GAIN, gain
  monkeypatch.setattr(sm, "DOUBT_GAIN", gain + 0.01)
  got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
  assert got.why == "doubted" and abs(got.pose[0] - guess[0]) < 0.05, got
  monkeypatch.setattr(sm, "DOUBT_GAIN", gain - 0.01)
  got = sm.ScanMatcher(store).match(guess, ANGLES, ranges)
  assert got.why == "found" and abs(got.pose[0] - AT_COUNTER[0]) < 0.05, got


def test_the_doubts_gain_is_over_the_fit_it_would_replace(store, monkeypatch):
  # ...over the pose the doubted fit would lay the scan at, never the pose
  # asked about (#681's second review): a fit corrected across a sidewalk
  # made any place along it a gain over the asked pose. The wiring, and the
  # bar against a pose: no place gains over itself.
  ranges = cast(STORE, AT_COUNTER, np.random.default_rng(9))
  guess = (AT_COUNTER[0] + 0.3, AT_COUNTER[1], AT_COUNTER[2])
  m = sm.ScanMatcher(store)
  m.refresh(*guess[:2])
  pts = m.points(ANGLES, ranges)
  at, why, _ = m._judge(guess, pts, m._fit(guess, pts))
  assert why == "doubted" and at != guess, (why, at)
  start = m._search(guess, pts, unique=True)
  assert start is not None and m._search(guess, pts, unique=True, against=start) is None
  seen = []
  real = sm.ScanMatcher._search

  def spy(self, pose, pts, **kw):
    seen.append(kw.get("against"))
    return real(self, pose, pts, **kw)

  monkeypatch.setattr(sm.ScanMatcher, "_search", spy)
  assert sm.ScanMatcher(store).match(guess, ANGLES, ranges).why == "found"
  assert seen == [at], (seen, at)


def test_a_slid_fits_search_is_taken_only_where_the_fit_slid(room, monkeypatch):
  # A fit that slid past MAX_STEP is searched, and the search's pose taken
  # only within DOUBT_SLID_M of where the fit itself slid: on Luca's live map
  # of 11-09 a belief 0.1 m off slid 0.17 m toward one copy of the counter
  # and the search found another 0.25 m from it (13 poses of 40, the base
  # refusing every one); the slides the search agreed with, within 0.05 m
  # on a map laid true, were the counter's take-backs.
  ranges = cast(ROOM, TRUE, np.random.default_rng(5))
  guess = (TRUE[0] + 0.14, TRUE[1] - 0.10, TRUE[2] + 0.08)
  assert sm.ScanMatcher(room).match(guess, ANGLES, ranges).why == "found"
  for name in ("DOUBT_SLID_M", "DOUBT_SLID_RAD"):     # ...where it slid and turned
    with monkeypatch.context() as patch:
      patch.setattr(sm, name, 0.0)
      got = sm.ScanMatcher(room).match(guess, ANGLES, ranges)
      assert got.why == "slid" and not got.accepted and got.pose == guess, (name, got)


def test_the_doubts_search_runs_once_in_its_scans_and_a_restart_keeps_the_count(
    corridor, monkeypatch):
  # #681's review: ~25 ms a doubted scan, and 600 of 600 doubted in a minute
  # standing among clutter the map lacks. The search a doubted or a slid fit
  # gets runs at most once every DOUBT_EVERY scans; between, the fit is what
  # it was (doubted laid in, slid refused, as before #681). The post the map
  # never saw doubts every scan here, and nothing explains it better.
  calls = []
  real = sm.ScanMatcher._search

  def spy(self, pose, pts, **kw):
    calls.append(len(calls))
    return real(self, pose, pts, **kw)

  monkeypatch.setattr(sm.ScanMatcher, "_search", spy)
  true = (4.0, 0.7, 0.0)
  ranges = cast(CORRIDOR + [(5.5, 0.5, 5.5, 1.0)], true, np.random.default_rng(3))
  guess = (true[0] + 0.3, true[1], true[2])
  m = sm.ScanMatcher(corridor)
  whys = [m.match(guess, ANGLES, ranges).why for _ in range(2 * sm.DOUBT_EVERY + 1)]
  assert set(whys) == {"doubted"} and len(calls) == 3, (whys, calls)
  # ...and the count rides a restart
  m = sm.ScanMatcher(corridor)
  for _ in range(3):
    m.match(guess, ANGLES, ranges)
  state, arrays = m.kept_state()
  again = sm.ScanMatcher(corridor)
  again.restore_kept(state, arrays)
  calls.clear()
  for _ in range(sm.DOUBT_EVERY - 3):
    again.match(guess, ANGLES, ranges)
  assert calls == [], calls
  again.match(guess, ANGLES, ranges)
  assert len(calls) == 1
  # ...and starts afresh where the belief is put somewhere new -- the window
  # of a fix opened or closed (`anchored`), the map forgotten at a stand-up
  # -- and never at `fuse_next`, which the Navigator calls at every scan it
  # does not map (#681's second review: a robot bowing in and out of
  # mapping searched at every mapped scan)
  for reset, afresh in ((again.fuse_next, False), (lambda: again.anchored(on=False), True),
                        (again.forget, True)):
    calls.clear()
    again.match(guess, ANGLES, ranges)
    assert calls == [], calls
    reset()
    again.match(guess, ANGLES, ranges)
    assert len(calls) == afresh, (reset, calls)


def test_a_scan_with_no_map_under_it_is_no_match():
  g = OccupancyGrid(-1, -1, 7, 5, 0.05)
  got = sm.ScanMatcher(g).match(TRUE, ANGLES, cast(ROOM, TRUE))
  assert not got.accepted and got.why == "no map"


def test_the_matcher_reads_every_return_short_of_the_lidars_reach():
  # #560: the park's middle is 8 m from its nearest fence, and the band
  # reads a wall within 2-3 mm out to 7.9 m (#581) -- so every return the
  # map reads is a point of the fit, and only "no return" (the reach
  # itself) is none. At the C1's 6.5 m the probe left the park's middle
  # unpinned at 71 % of places.
  assert sm.MATCH_RANGE_M >= 8.0
  m = sm.ScanMatcher(OccupancyGrid(-1, -1, 9, 5, 0.05), max_range=8.0)
  ranges = np.array([1.0, 6.6, 7.5, 7.99, 8.0])
  pts = m.points(np.zeros(5), ranges)
  assert np.allclose(pts[:, 0], [1.0, 6.6, 7.5, 7.99])


def test_the_field_is_signed_and_crosses_zero_on_the_walls_first_cell(room):
  # + in front of a wall, - inside the band of cells it is marked with,
  # zero on the centre of the first: a point behind the face is pulled back
  # to it, never through the band's far side.
  m = sm.ScanMatcher(room)
  m.refresh(3.0, 2.0)
  res = room.resolution
  ix, iy = room.world_to_cell(5.70, 2.0)
  row = m.field[iy - m.corner[1], ix - m.corner[0]:ix - m.corner[0] + 10]
  first = int(np.argmax(np.nan_to_num(row, nan=1.0) <= 0.0))
  assert row[first] == pytest.approx(0.0)
  assert np.all(np.diff(row[:first + 1]) < 0)             # falling to the face
  inside = row[first:][~np.isnan(row[first:])]
  # ...and still falling through the band and the unseen cells just behind
  # it: measured from the band's back, it read 0, 0, +5, +5 cm
  assert len(inside) >= 3
  assert np.all(inside <= 0.0) and np.all(np.diff(inside) < 0)
  assert room.cell_to_world(ix + first, iy)[0] == pytest.approx(6.025, abs=res / 2)


def test_new_geometry_in_unseen_space_pulls_on_nothing():
  # A wall the map has never seen: its returns land in unknown cells, and
  # pulled to the nearest known one they would drag the pose to the edge of
  # what was mapped. Seen, they are no information.
  room3 = box(0.013, 0.021, 6.013, 4.021)
  seen = [s for s in room3 if s[0] != 6.013 or s[2] != 6.013]      # the east wall unseen
  rng = np.random.default_rng(2)
  g = OccupancyGrid(-1, -1, 7, 5, 0.05)
  for p in [(1.0, 2.0, 0.0), (2.0, 1.0, 1.0), (1.5, 3.0, -1.0)]:
    r = cast(seen + [(4.5, -1.0, 4.5, 5.0)], p, rng)      # a screen hides it
    g.update(p, ANGLES, r, 8.0, origin=(0.0, 0.0))
  m = sm.ScanMatcher(g)
  m.refresh(1.5, 2.0)
  east = [(0.013, 0.021, 6.013, 0.021), (6.013, 0.021, 6.013, 4.021)]
  pts = m.points(ANGLES, cast(east, (5.0, 2.0, 0.0), max_range=8.0))
  d, gx, gy, gth, live = m.distance((5.0, 2.0, 0.0), pts)
  east_points = pts[:, 0] > 0.9
  assert not live[east_points].any()


def test_a_match_costs_less_than_the_map_update_it_precedes(room):
  # The cost pinned as a RATIO, which the machine's load moves far less
  # than a wall-clock bound: one match against the grid update of the same
  # scan on the home world's grid (469,200 cells), interleaved. MEASURED
  # ~0.4 ms against ~1.6 ms. ⚠ With the collector OFF, as `timeit` has it:
  # a pass in the loop frees whatever the tests before it left (#473: their
  # memory stores, each closed with an fsync, put 0.4-1.2 s into one match).
  # Its threshold is a pass at every allocation, so a loop run with it on
  # fails here at once.
  big = OccupancyGrid(-17.5, -11.5, 33.5, 11.5, 0.05)
  ix0, iy0 = big.world_to_cell(-1.0, -1.0)
  big.grid[iy0:iy0 + room.grid.shape[0], ix0:ix0 + room.grid.shape[1]] = room.grid
  m = sm.ScanMatcher(big)
  rng = np.random.default_rng(4)
  scans = [cast(ROOM, TRUE, rng) for _ in range(20)]
  m.match(TRUE, ANGLES, scans[0])
  scratch = OccupancyGrid(-17.5, -11.5, 33.5, 11.5, 0.05)
  t_match = t_update = 0.0
  passes = []

  def counted(phase, info):
    passes.append(phase)

  enabled, threshold = gc.isenabled(), gc.get_threshold()
  gc.disable()
  gc.set_threshold(1)
  gc.callbacks.append(counted)
  try:
    for r in scans:
      t = time.perf_counter()
      m.match(TRUE, ANGLES, r)
      t_match += time.perf_counter() - t
      t = time.perf_counter()
      scratch.update(TRUE, ANGLES, r, 8.0, origin=(0.0, 0.0))
      t_update += time.perf_counter() - t
  finally:
    gc.callbacks.remove(counted)
    gc.set_threshold(*threshold)
    if enabled:
      gc.enable()
  assert not passes, f"the collector ran {len(passes) // 2} times inside the timed loop"
  assert t_match < t_update, f"{t_match * 50:.2f} ms against {t_update * 50:.2f} ms a scan"


def test_one_scan_matches_the_same_twice_and_across_a_restart(room):
  # Deterministic: the same inputs give the same pose to the bit, and a
  # matcher restored from `kept_state` answers the next scan exactly as the
  # one that was never stopped.
  rng = np.random.default_rng(9)
  scans = [cast(ROOM, TRUE, rng) for _ in range(30)]
  a, b = sm.ScanMatcher(room), sm.ScanMatcher(room)
  pa = pb = (TRUE[0] + 0.03, TRUE[1], TRUE[2])
  for k, r in enumerate(scans[:25]):
    pa = a.match(pa, ANGLES, r).pose
    pb = b.match(pb, ANGLES, r).pose
    a.fused(pa, 0.1 * k)
    b.fused(pb, 0.1 * k)
  assert pa == pb
  state, arrays = a.kept_state()
  c = sm.ScanMatcher(room)
  c.restore_kept(state, arrays)
  pc = pa
  for r in scans[25:]:
    pa = a.match(pa, ANGLES, r).pose
    pc = c.match(pc, ANGLES, r).pose
  assert pa == pc


def test_the_prior_lets_go_of_odometry_that_pumps(room, monkeypatch):
  # A robot standing still whose odometry claims 10 mm a scan (spinning on the
  # lab's feed plate, measured): the robust prior lets the walls win, and
  # the pose stays. A plain prior lags the pump by its share of the fit --
  # 6 cm here, among four walls; in the lab, where the walls said less, the
  # lag left the fit's basin and the pose ran 1 m (the premise, shown by
  # lifting the robustness).
  rng = np.random.default_rng(12)
  scans = [cast(ROOM, TRUE, rng) for _ in range(60)]

  def stand(matcher):
    pose = TRUE
    for r in scans:
      c, s_ = math.cos(pose[2]), math.sin(pose[2])
      pose = (pose[0] + 0.010 * c, pose[1] + 0.010 * s_, pose[2])   # the pump
      pose = matcher.match(pose, ANGLES, r).pose
    return math.hypot(pose[0] - TRUE[0], pose[1] - TRUE[1])

  assert stand(sm.ScanMatcher(room)) < 0.025
  monkeypatch.setattr(sm, "PRIOR_SCALE", 1e9)
  assert stand(sm.ScanMatcher(room)) > 0.045


def test_a_robot_standing_still_fuses_a_scan_every_few_seconds(room):
  # Docked for minutes, every scan fused at a pose jittering by a
  # millimetre walked the map and the pose together; fused only once the
  # robot moves, or every FUSE_S, it stays.
  m = sm.ScanMatcher(room)
  ok = sm.Match(TRUE, True, "ok", 300, 0.02, 100.0, (1.0, 0.0, 0.0), 0, sm.NO_STEP)
  assert m.fuses(ok, 0.0)                              # nothing fused yet
  m.fused(TRUE, 0.0)
  jitter = ok._replace(pose=(TRUE[0] + 0.002, TRUE[1], TRUE[2] + 0.001))
  assert not m.fuses(jitter, 0.1)
  assert m.fuses(jitter, sm.FUSE_S)
  moved = ok._replace(pose=(TRUE[0] + 1.5 * sm.FUSE_M, TRUE[1], TRUE[2]))
  assert m.fuses(moved, 0.1)
  turned = ok._replace(pose=(TRUE[0], TRUE[1], TRUE[2] + 1.5 * sm.FUSE_RAD))
  assert m.fuses(turned, 0.1)
  # ...and a fit that slid is never laid into the map, however long it has been
  assert not m.fuses(moved._replace(why="slid", accepted=False), 99.0)
  assert m.fuses(moved._replace(why="sparse", accepted=False), 99.0)


# ---- the body's wiring ---------------------------------------------------------


class _FixedMatcher(sm.ScanMatcher):
  """Answers every scan with one verdict."""

  def __init__(self, grid, answer):
    super().__init__(grid)
    self.answer = answer

  def match(self, pose, angles, ranges, origin=None):
    return self.answer


@pytest.fixture(scope="module")
def house():
  """The humanoids' house with one humanoid in it, compiled once."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  return world_spec(world_config(HUMANOID_HOME)["model"]).compile()


def _humanoid(model):
  """The humanoid standing in the living room; nothing stepped."""
  import mujoco

  from pluggybot.body import body_for
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  body = body_for(model, mujoco.MjData(model),
                  grid_bounds=world_config(HUMANOID_HOME)["grid_bounds"])
  body.start_at(1.5, 0.5, 0.0)
  return body


def test_each_scan_is_laid_through_the_pose_its_match_found(house):
  # The matched pose is the belief -- the reckoning is moved to it -- and
  # the pose the scan goes into the map at; a refused fit leaves the belief
  # where odometry had it, and the map untouched. The scan's own seam,
  # called directly: the rule is the wiring, not a walk.
  from pluggybot.perception.lidar import LIDAR_PERIOD
  body = _humanoid(house)
  m = body.mission
  laid = []
  real = m.grid.update
  m.grid.update = lambda pose, *a, **kw: (laid.append(tuple(pose)), real(pose, *a, **kw))
  target = (1.7, 0.4, 0.05)
  m.matcher = _FixedMatcher(m.grid, sm.Match(target, True, "ok", 300, 0.02, 100.0,
                                             (1.0, 0.0, 0.0), 0, (0.2, -0.1, 0.05)))
  m._scan_step()
  assert laid == [target]
  assert math.hypot(m.odo.x - target[0], m.odo.y - target[1]) < 0.01
  laid.clear()
  before = m.pose
  m.matcher.answer = m.matcher.answer._replace(why="slid", accepted=False, pose=before)
  m.matcher.fused_pose = None
  m.data.time += LIDAR_PERIOD
  m._scan_step()
  assert not laid, "a scan whose fit slid was laid into the map"
  assert m.pose == before
  body.close()


def test_the_served_body_set_down_far_from_its_belief_finds_itself(house):
  # The served body's own LIDAR in its house, through the scan's own seam
  # (issue #422): the living room mapped by a scan standing, the belief then
  # moved 1.5 m and 5 deg off -- a robot come home from a long explore. Its
  # map refuses two runs of LOST_RUN scans and none of them is laid in; the
  # wide search, finding the same pose twice, puts the belief back where the
  # body stands. Nothing stepped: the scans are the standing body's.
  from pluggybot.perception.lidar import LIDAR_PERIOD
  body = _humanoid(house)
  m = body.mission
  m._scan_step()
  laid = []
  real = m.grid.update
  m.grid.update = lambda pose, *a, **kw: (laid.append(tuple(pose)), real(pose, *a, **kw))
  x, y, th = m.pose
  m.odo.correct(x - 1.2, y + 0.9, th + math.radians(5.0))
  whys = []
  for _ in range(2 * sm.LOST_RUN):
    m.data.time += LIDAR_PERIOD
    m._scan_step()
    whys.append(m.matcher.last.why)
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN - 1) + ["relocated"], whys
  assert not laid, "a refused scan was laid into the map"
  (bx, by, bth), (tx, ty, tth) = m.pose, m.true_pose()
  assert math.hypot(bx - tx, by - ty) < 0.05
  assert abs(math.degrees(math.atan2(math.sin(bth - tth), math.cos(bth - tth)))) < 1.0
  # ...and the jump is said (issue #476): onto a copy the map held, a jump
  # like it lost a belief put right, and only its record tells them apart
  jumps = [e for e in m.belief_events if e["why"] == "relocated"]
  assert len(jumps) == 1 and jumps[0]["before"]["x"] == pytest.approx(x - 1.2, abs=1e-3)
  assert jumps[0]["after"]["x"] == pytest.approx(bx, abs=1e-3)
  body.close()


def test_the_served_body_up_from_a_fall_its_heading_turned_finds_itself(house):
  # The served body's own LIDAR in its house, through the scan's own seam
  # (issue #690): the living room mapped by a scan standing, then its belief
  # 0.5 m off and turned 120 deg -- what the other robot's blocking interrupt
  # left Luca with on 11-12 -- and its get-up's stand called as the get-up
  # calls it. The served wide search finds nothing; at every heading, the
  # pose found twice puts the belief back. Never up from a fall, the same
  # scans are refused.
  from pluggybot.perception.lidar import LIDAR_PERIOD

  def turned(stood):
    body = _humanoid(house)
    m = body.mission
    m._scan_step()
    x, y, th = m.pose
    m.odo.correct(x - 0.4, y + 0.3, th + math.radians(120.0))
    if stood:
      m._stood_up()
    whys = []
    for _ in range(2 * sm.LOST_RUN):
      m.data.time += LIDAR_PERIOD
      m._scan_step()
      whys.append(m.matcher.last.why)
    (bx, by, bth), (tx, ty, tth) = m.pose, m.true_pose()
    body.close()
    return whys, math.hypot(bx - tx, by - ty), _deg(bth - tth)

  whys, off, deg = turned(stood=False)
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN) and deg > 100.0
  whys, off, deg = turned(stood=True)
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN - 1) + ["relocated"], whys
  assert off < 0.05 and deg < 1.0


def test_a_restart_keeps_the_matchers_field_and_the_imus_stream(house):
  # A new piece of state that decides anything is kept (issue #345): the
  # field the next match reads and where the IMU's noise had got to.
  from pluggybot.perception.lidar import LIDAR_PERIOD
  a = _humanoid(house)
  m = a.mission
  for _ in range(3):
    m._scan_step()
    m.data.time += LIDAR_PERIOD
  imu = m.stand.trunk.imu
  for _ in range(100):
    imu.gyro_z(0.2, 0.002)
  assert m.matcher.field is not None, "the premise: a field to keep"
  state, arrays = m.kept_state()
  assert state["trunk"]["imu"]["rng"] and state["match"] is not None
  b = _humanoid(house)
  n = b.mission
  n.restore_kept(state, arrays)
  assert n.stand.trunk.imu.gyro_z(0.1, 0.002) == imu.gyro_z(0.1, 0.002)
  assert np.array_equal(n.matcher.field, m.matcher.field, equal_nan=True)
  assert n.matcher.fused_pose == m.matcher.fused_pose
  a.close()
  b.close()


def test_a_pose_past_the_fits_reach_is_found_by_the_search(room):
  # 0.4 m off, the walls are past the field's reach and the fit cannot
  # pull the pose back: MEASURED, an odometry pump while tilted left a robot
  # 0.45 m out, and every scan it fused painted a second house. The fit's
  # verdict is "inconsistent", the search finds the pose, and the fit from
  # there is taken.
  m = sm.ScanMatcher(room)
  ranges = cast(ROOM, TRUE, np.random.default_rng(21))
  got = m.match((TRUE[0] + 0.30, TRUE[1] - 0.26, TRUE[2] + 0.04), ANGLES, ranges)
  assert got.why == "found" and got.accepted and got.fuse
  assert math.hypot(got.pose[0] - TRUE[0], got.pose[1] - TRUE[1]) < 0.03
  assert abs(math.degrees(got.pose[2] - TRUE[2])) < 0.3


def test_a_scan_the_map_cannot_explain_is_refused_and_not_fused(room):
  # A scan of somewhere else -- another room's walls where the map has
  # its own -- is no correction and no map: refused, and never laid in.
  m = sm.ScanMatcher(room)
  elsewhere = box(0.4, 0.7, 3.1, 2.9) + [(1.7, 0.7, 1.7, 1.6)]
  got = m.match(TRUE, ANGLES, cast(elsewhere, (1.2, 1.6, 0.9), np.random.default_rng(22)))
  assert got.why == "inconsistent" and not got.accepted and not got.fuse
  assert got.pose == TRUE


# ---- a robot come home lost (issue #422) -----------------------------------------


#: Lost 1.2 m and 4 deg from TRUE: past the search and the fit from its best.
LOST = (TRUE[0] - 0.9, TRUE[1] + 0.8, TRUE[2] + math.radians(4.0))


def test_a_robot_lost_past_the_search_is_relocated_after_two_runs_of_refusals(room):
  # Home from a long explore 0.8-1.4 m out, past the ±0.6 m search, every
  # scan was refused (issue #422). LOST_RUN refusals running, the wide
  # search finds the one pose the room agrees with; LOST_RUN more, it finds
  # it again and takes it. Matched from there, the next scan is laid in.
  m = sm.ScanMatcher(room)
  rng = np.random.default_rng(23)
  for _ in range(2 * sm.LOST_RUN - 1):
    got = m.match(LOST, ANGLES, cast(ROOM, TRUE, rng))
    assert got.why == "inconsistent" and got.pose == LOST
  assert m.pending is not None, "the first search found nothing"
  m.doubt_wait = sm.DOUBT_EVERY      # a doubt searched just before: relocated, it waits no more
  got = m.match(LOST, ANGLES, cast(ROOM, TRUE, rng))
  assert got.why == "relocated" and got.accepted and not got.fuse and m.doubt_wait == 0
  assert math.hypot(got.pose[0] - TRUE[0], got.pose[1] - TRUE[1]) < 0.02
  assert abs(math.degrees(got.pose[2] - TRUE[2])) < 0.3
  assert m.match(got.pose, ANGLES, cast(ROOM, TRUE, rng)).why == "ok"


def test_the_wide_search_judges_at_its_bar_as_it_reads_now(room, monkeypatch):
  # `_relocate` judges what it found at FOUND_SHARE as the module reads it
  # when it runs, as the bounded search does: a bar bound when `_judge` was
  # defined left a test or a sweep of it flying the old one (#681's review).
  m = sm.ScanMatcher(room)
  m.refresh(LOST[0], LOST[1])
  pts = m.points(ANGLES, cast(ROOM, TRUE, np.random.default_rng(23)))
  assert m._relocate(LOST, pts) is not None, "the premise: the wide search finds it"
  monkeypatch.setattr(sm, "FOUND_SHARE", 1.01)
  assert m._relocate(LOST, pts) is None


def test_a_correction_the_wide_search_finds_once_is_not_taken(room):
  # Once is a coincidence: on a sidewalk whose thin wall the map had eroded
  # (#401), a pose 1.8 m along it explained a scan better than the truth
  # did, and the next search took a different one. Here the robot is lost
  # by one correction for the first run of refusals and another for the
  # second -- it is moved, its belief left where it was -- and nothing is
  # taken until a third run finds the second again.
  m = sm.ScanMatcher(room)
  rng = np.random.default_rng(25)
  moved = (TRUE[0] + 0.4, TRUE[1] - 0.8, TRUE[2] - math.radians(3.0))
  lost = (2.0, 2.2, TRUE[2])
  for k in range(3 * sm.LOST_RUN):
    truth = TRUE if k < sm.LOST_RUN else moved
    got = m.match(lost, ANGLES, cast(ROOM, truth, rng))
    if k < 3 * sm.LOST_RUN - 1:
      assert got.why == "inconsistent", k
  assert got.why == "relocated"
  assert math.hypot(got.pose[0] - moved[0], got.pose[1] - moved[1]) < 0.02


def test_the_wide_search_takes_nothing_where_two_places_explain_the_scan():
  # Two rooms alike, the robot lost between them: the scan cannot say which
  # it stands in, and a relocation there would be a coin flip whose wrong
  # side lays the rest of the day's map through the wrong room.
  a = box(0.013, 0.021, 3.013, 2.521)
  b = [(x1 + 3.4, y1, x2 + 3.4, y2) for x1, y1, x2, y2 in a]
  rng = np.random.default_rng(3)
  poses = [(rng.uniform(0.4, 2.6), rng.uniform(0.4, 2.1), rng.uniform(-3, 3)) for _ in range(15)]
  true, lost = (1.5, 1.2, 0.2), (3.2, 2.0, 0.25)

  def lost_for(segments, poses):
    m = sm.ScanMatcher(mapped(segments, (-1, -1, 7.5, 3.5), poses))
    for _ in range(2 * sm.LOST_RUN):
      got = m.match(lost, ANGLES, cast(a, true, rng))
    return got

  both = lost_for(a + b, poses + [(x + 3.4, y, th) for x, y, th in poses])
  assert both.why == "inconsistent" and both.pose == lost
  # ...the premise: with only the room it stands in mapped, the same search
  # finds it
  one = lost_for(a, poses)
  assert one.why == "relocated"
  assert math.hypot(one.pose[0] - true[0], one.pose[1] - true[1]) < 0.03


def test_any_scan_not_refused_breaks_the_run_and_what_it_found(room, monkeypatch):
  # LOST_RUN refusals RUNNING: a scan the map did not refuse -- matched, or
  # too little map to say -- breaks the run, and the correction the run's
  # search found goes with it; carried across, it would confirm a search
  # made far down the walk, when the drift was another.
  m = sm.ScanMatcher(room)
  rng = np.random.default_rng(26)
  searched = []
  real = m._relocate
  monkeypatch.setattr(m, "_relocate", lambda pose, pts: (searched.append(1), real(pose, pts))[1])
  for _ in range(sm.LOST_RUN + 2):
    m.match(LOST, ANGLES, cast(ROOM, TRUE, rng))
  assert searched and m.pending is not None, "the premise: a search found the pose"
  empty = np.full(len(ANGLES), 8.0)                     # nothing in reach: no points
  assert m.match(LOST, ANGLES, empty).why == "sparse"
  assert (m.lost_run, m.pending) == (0, None)
  for _ in range(sm.LOST_RUN - 1):
    assert m.match(LOST, ANGLES, cast(ROOM, TRUE, rng)).why == "inconsistent"
  assert len(searched) == 1, "searched again before LOST_RUN refusals running"


def test_the_wide_search_weighs_places_not_headings(room):
  # Its candidates are PLACES, UNIQUE_M apart at any heading: deduped only
  # where position AND heading were near, its three slots went to one spot
  # at three headings, which all fitted back to one pose -- and a twin
  # ranked fourth would never have been looked at.
  m = sm.ScanMatcher(room)
  m.refresh(*LOST[:2])
  pts = m.points(ANGLES, cast(ROOM, TRUE, np.random.default_rng(27)))
  cands = m._wide_candidates(LOST, pts)
  assert len(cands) == sm.WIDE_TRIES
  for i, a in enumerate(cands):
    for b in cands[i + 1:]:
      assert math.hypot(a[0] - b[0], a[1] - b[1]) >= sm.UNIQUE_M, (a, b)


def test_a_restart_keeps_the_run_of_refusals_and_what_it_found(room):
  # Both decide when the robot relocates (issue #345's rule): restarted
  # after the first wide search, it relocates on the same scan as a robot
  # never stopped.
  rng = np.random.default_rng(24)
  scans = [cast(ROOM, TRUE, rng) for _ in range(2 * sm.LOST_RUN)]
  a = sm.ScanMatcher(room)
  for r in scans[:sm.LOST_RUN + 2]:
    a.match(LOST, ANGLES, r)
  state, arrays = a.kept_state()
  assert state["pending"] is not None and state["lostRun"] == sm.LOST_RUN + 2
  b = sm.ScanMatcher(room)
  b.restore_kept(state, arrays)
  for r in scans[sm.LOST_RUN + 2:]:
    got_a, got_b = a.match(LOST, ANGLES, r), b.match(LOST, ANGLES, r)
  assert got_a.why == got_b.why == "relocated" and got_a.pose == got_b.pose


# ---- up from a fall, a take is checked by a wider look (issue #681) ----------


#: The storeroom with its counter (#681's fourth round): the counter's face
#: 0.6 m in front of the east wall, along most of it. A scan reads the face,
#: the nearer, and the map holds both lines -- the wall's was laid over the
#: counter's top while the robot worked at it.
COUNTER = STORE + [(5.413, 0.6, 5.413, 5.4), (5.413, 0.6, 6.013, 0.6), (5.413, 5.4, 6.013, 5.4)]
#: Up from a fall at the counter, as flown: the get-up staggered the body a
#: metre back and turned it, the odometry held through it, so the belief
#: stands 0.99 m east and 0.30 m south of the body, its heading right.
STOOD = (4.1, 3.3, -1.8)
BEHIND = (STOOD[0] + 0.99, STOOD[1] - 0.30, STOOD[2])
#: ...and one that left the belief half a metre east: the fit itself slides
#: the face's returns onto the wall's line, 9 cm on.
BESIDE = (STOOD[0] + 0.52, STOOD[1] - 0.04, STOOD[2])
#: ...and one from where every fit slides and its search finds nothing where
#: it slid: refused scan after scan, each breaking the run the wide search
#: waits on.
SLIDING = (STOOD[0] + 0.40, STOOD[1] + 0.10, STOOD[2])
#: ...and one that carried the body past the wide search's own reach (flown:
#: 2.47 m).
FAR = (STOOD[0] + 2.37, STOOD[1] - 0.71, STOOD[2])
#: A wider look's take-back, in verdicts: the look's own scan and the
#: LOST_RUN - 1 after it refused, then the next look finds the place again.
TAKEN_BACK = ["outdone"] * sm.LOST_RUN + ["relocated"]


@pytest.fixture(scope="module")
def counter():
  rng = np.random.default_rng(4)
  poses = [(rng.uniform(0.6, 5.0), rng.uniform(0.6, 5.4), rng.uniform(-3, 3)) for _ in range(40)]
  g = mapped(STORE, (-1, -1, 7, 7), poses[:12])
  rng = np.random.default_rng(5)
  for p in poses[12:]:
    g.update(p, ANGLES, cast(COUNTER, p, rng), 8.0, origin=(0.0, 0.0))
  return g


def _fallen(grid):
  """A matcher on `grid`, up from a fall."""
  m = sm.ScanMatcher(grid)
  m.fell()
  return m


def _stood(m, belief, rng, scans):
  """`scans` scans of the counter from STOOD matched as the Navigator
  matches them, the belief `belief` moved by each take: the verdicts and
  where the belief ended."""
  whys = []
  for _ in range(scans):
    got = m.match(belief, ANGLES, cast(COUNTER, STOOD, rng))
    whys.append(got.why)
    if got.accepted:
      belief = got.pose
  return whys, belief


def _looks(monkeypatch):
  """Every wider look and plain wide search from here on, by name."""
  made = []
  found, relocate = sm.ScanMatcher._wide_found, sm.ScanMatcher._relocate
  inside = []

  def wide_found(self, pose, pts, doubt=False):
    if not inside:
      made.append("look")
    return found(self, pose, pts, doubt)

  def relocated(self, pose, pts):
    made.append("search")
    inside.append(1)
    try:
      return relocate(self, pose, pts)
    finally:
      inside.pop()
  monkeypatch.setattr(sm.ScanMatcher, "_wide_found", wide_found)
  monkeypatch.setattr(sm.ScanMatcher, "_relocate", relocated)
  return made


def _place(off, inliers, turn=0.0):
  """A place a wider look found, `off` (m east, north) of BESIDE and `turn`
  rad from its heading."""
  return ((BESIDE[0] + off[0], BESIDE[1] + off[1], BESIDE[2] + turn),
          {"inliers": inliers, "rms_m": 0.02, "weakest": 80.0, "weak_dir": (1.0, 0.0, 0.0),
           "held": 0, "step": (off[0], off[1], turn)})


def _doubt(m):
  """The fall's doubt, as a matcher holds it."""
  return m.fallen, m.looks_left, m.wider_wait, m.better, m.rivals


SETTLED = (False, 0, 0, None, False)


def test_up_from_a_fall_a_find_the_search_cannot_see_past_is_refused_for_a_better_place_wider(
    counter, monkeypatch):
  # MEASURED (#681's fourth round; the wake of 11-14, four falls of five): a
  # get-up at the storeroom's counter leaves the belief a metre from the
  # body, past the bounded search's 0.6 m, and its best there is the place
  # where the counter's face lies on the wall behind it, 0.6 m east of the
  # truth -- 76 % of the scan on walls, every direction fixed, taken as
  # `found`, and every fit after it `ok` for 33-87 minutes. A wider look,
  # that find one more of its places, holds the truth: the find is refused,
  # and the truth taken once the next look finds it again.
  looks = _looks(monkeypatch)
  whys, belief = _stood(_fallen(counter), BEHIND, np.random.default_rng(31), sm.LOST_RUN + 1)
  assert whys == TAKEN_BACK and looks == ["look", "look"], (whys, looks)
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03, belief
  # ...the premise, and what a robot not up from a fall still does: the
  # search's find is that place, taken at once, no wider look made
  looks.clear()
  whys, belief = _stood(sm.ScanMatcher(counter), BEHIND, np.random.default_rng(31), 1)
  assert whys == ["found"] and 0.5 < belief[0] - STOOD[0] < 0.7 and abs(belief[1] - STOOD[1]) < 0.03
  assert not looks


def test_up_from_a_fall_a_fit_is_refused_for_a_better_place_wider(counter):
  # ...and where the get-up left the belief half a metre east, the fit
  # itself slides onto that place, `ok` (flown: 0.51 m east at the stand,
  # 0.60 after its first scan): the fit is checked as the search's find is.
  # Not up from a fall, a fit is what it was.
  whys, belief = _stood(_fallen(counter), BESIDE, np.random.default_rng(32), sm.LOST_RUN + 1)
  assert whys == TAKEN_BACK, whys
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03, belief
  whys, belief = _stood(sm.ScanMatcher(counter), BESIDE, np.random.default_rng(32), 3)
  assert whys == ["ok"] * 3 and 0.5 < belief[0] - STOOD[0] < 0.7, (whys, belief)


def test_up_from_a_fall_a_fit_that_only_slides_is_looked_round(counter, monkeypatch):
  # FLOWN (one fall of the counter's 84; and 20-37 poses of 40 on three of
  # the live maps probed, a belief left (+0.8, +0.3) m): every fit slid and
  # its search found nothing where it slid -- 563 scans refused in a minute,
  # the belief 0.86 m off -- and a fit that slid is no take to check and no
  # refusal of the run the wide search waits on. Up from a fall it is looked
  # round as a take is, at the heading the robot believes, and the place two
  # looks running find is taken. Not up from a fall it is what it was.
  looks = _looks(monkeypatch)
  whys, belief = _stood(_fallen(counter), SLIDING, np.random.default_rng(44), sm.LOST_RUN + 1)
  assert whys == ["slid"] * sm.LOST_RUN + ["relocated"] and looks == ["look", "look"], (whys, looks)
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03, belief
  looks.clear()
  whys, belief = _stood(sm.ScanMatcher(counter), SLIDING, np.random.default_rng(44),
                        2 * sm.LOST_RUN + 1)
  assert whys == ["slid"] * (2 * sm.LOST_RUN + 1) and belief == SLIDING and not looks, whys


def test_up_from_a_fall_the_wide_searchs_own_take_is_checked_by_the_next_look(counter, monkeypatch):
  # FLOWN (one fall of 85 on the second pod): a get-up carried the body 2.47
  # m from the belief, past the wide search's 2 m, and that search's best in
  # reach -- found twice, unrivalled -- was the counter's face on the wall
  # behind it, 0.58 m east of the truth, every fit `ok` from there. Up from
  # a fall that take is checked as any other: the next scan's look is made
  # from where the belief was put, and the truth, now in reach, is taken once
  # the look after finds it again. Not up from a fall the wide search's take
  # is what it was.
  looks = _looks(monkeypatch)
  m = _fallen(counter)
  whys, belief = _stood(m, FAR, np.random.default_rng(45), 2 * sm.LOST_RUN)
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN - 1) + ["relocated"], whys
  assert 0.5 < belief[0] - STOOD[0] < 0.7 and looks == ["search", "search"], (belief, looks)
  assert _doubt(m) == (True, sm.FALLEN_LOOKS, 0, None, False) and m.turned_from is None
  whys, belief = _stood(m, belief, np.random.default_rng(46), sm.LOST_RUN + 1)
  assert whys == TAKEN_BACK and looks == ["search", "search", "look", "look"], (whys, looks)
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03 and _doubt(m) == SETTLED
  m = sm.ScanMatcher(counter)
  whys, belief = _stood(m, FAR, np.random.default_rng(45), 3 * sm.LOST_RUN + 1)
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN - 1) + ["relocated"] + ["ok"] * (sm.LOST_RUN + 1)
  assert 0.5 < belief[0] - STOOD[0] < 0.7 and looks[4:] == ["search", "search"], (belief, looks)
  # ...what the last look held was held from the old belief: it goes
  m = _fallen(counter)
  m.better, m.rivals, m.wider_wait, m.turned_from = (1.0, 0.0, 0.0), True, 4, (0.0, 0.0)
  m._put()
  assert _doubt(m) == (True, sm.FALLEN_LOOKS, 0, None, False) and m.turned_from is None


def test_the_search_at_every_headings_take_is_checked_by_the_next_look_too(nook):
  # ...as the wide search's own is: put by the search at every heading (its
  # heading turned unseen in the fall, #690), the belief is still a fall's,
  # and the next scan's fit is looked round from where it was put. There it
  # is the best, and the doubt is over.
  rng = np.random.default_rng(41)
  scans = [cast(NOOK, TRUE, rng) for _ in range(2 * sm.LOST_RUN + 1)]
  m = _fallen(nook)
  whys = [m.match(TURNED, ANGLES, r).why for r in scans[:-1]]
  assert whys == ["inconsistent"] * (2 * sm.LOST_RUN - 1) + ["relocated"], whys
  assert _doubt(m) == (True, sm.FALLEN_LOOKS, 0, None, False) and m.turned_from is None
  got = m.match(m.last.pose, ANGLES, scans[-1])
  assert got.why == "ok" and _doubt(m) == SETTLED and m.looks_left == 0


def test_a_take_is_checked_once_and_stands_where_the_wider_look_agrees(counter, monkeypatch):
  # A get-up that left the belief right, or within the search's reach: the
  # look finds the take's own place, the take stands at once, and the fall's
  # doubt of the pose is over -- one look a fall.
  looks = _looks(monkeypatch)
  for belief, first in (((STOOD[0] + 0.03, STOOD[1] - 0.02, STOOD[2]), "ok"),
                        ((STOOD[0] + 0.45, STOOD[1] + 0.40, STOOD[2]), "found")):
    looks.clear()
    m = _fallen(counter)
    whys, end = _stood(m, belief, np.random.default_rng(33), 5)
    assert whys == [first] + ["ok"] * 4 and looks == ["look"], (whys, looks)
    assert _doubt(m) == SETTLED
    assert math.hypot(end[0] - STOOD[0], end[1] - STOOD[1]) < 0.03, end


def test_a_take_stands_where_the_wider_look_cannot_tell_it_from_the_best(counter, monkeypatch):
  # A place that explains the scan as well as the best does is its rival
  # (RIVAL_SHARE), and the take is one more of the look's places: the best's
  # rival, or itself the best, it stands as it did before. Here the bar is
  # set so that the counter's two lines are such a pair.
  m = _fallen(counter)
  m.refresh(BEHIND[0], BEHIND[1])
  pts = m.points(ANGLES, cast(COUNTER, STOOD, np.random.default_rng(31)))
  start = m._search(BEHIND, pts)
  at, why, kw = m._judge(BEHIND, pts, m._fit(start, pts), found=True)
  assert why == "found" and 0.5 < at[0] - STOOD[0] < 0.7, "the premise: the search's find"
  best = max(m._look(BEHIND, pts, (at, kw)), key=lambda f: f[1]["inliers"])
  assert math.hypot(best[0][0] - STOOD[0], best[0][1] - STOOD[1]) < 0.03, "...and the look's: the truth"
  share = kw["inliers"] / best[1]["inliers"]
  assert 0.6 < share < sm.RIVAL_SHARE, share
  assert m._outdone(BEHIND, pts, (at, kw)) == (True, None) and m.better is not None
  # ...itself the best: here beside a place made up, which explains less
  m = _fallen(counter)
  monkeypatch.setattr(m, "_wide_found", lambda pose, pts, doubt=False: [_place((-1.0, 0.3), 100)])
  assert m._outdone(BESIDE, pts, _place((0.0, 0.0), 200)) == (False, None)
  # ...and the best's rival
  monkeypatch.setattr(sm, "RIVAL_SHARE", share - 0.01)
  whys, belief = _stood(_fallen(counter), BEHIND, np.random.default_rng(31), 1)
  assert whys == ["found"] and 0.5 < belief[0] - STOOD[0] < 0.7, (whys, belief)


def test_a_take_two_better_places_outdo_is_refused_until_its_looks_are_spent(counter, monkeypatch):
  # The look cannot tell two places apart, and both explain the scan far
  # better than the take: the take is refused all the same -- taken, it is
  # the place the scan fits worst of the three -- and nothing waits to be
  # found again (#681's review: read as "nothing better", the rivalry let
  # the take stand). ⚠ Its looks spent, the doubt is over and the take is
  # the base's again, the bound's price: refused for as long as the robot
  # stood there, nothing was matched and nothing mapped for an hour at a
  # counter (the review again). The last look is made only to find again
  # what the one before found: with nothing waiting it is not made.
  m = _fallen(counter)
  looks = []
  monkeypatch.setattr(m, "_wide_found", lambda pose, pts, doubt=False: (
    looks.append(1), [_place((-0.52, 0.04), 340), _place((-1.42, 0.04), 335)])[1])
  spent = (sm.FALLEN_LOOKS - 1) * sm.LOST_RUN
  whys, belief = _stood(m, BESIDE, np.random.default_rng(38), spent)
  assert whys == ["outdone"] * spent and belief == BESIDE, whys
  assert m.fallen and m.rivals and m.better is None and len(looks) == sm.FALLEN_LOOKS - 1
  whys, belief = _stood(m, BESIDE, np.random.default_rng(38), 2)
  assert whys == ["ok", "ok"] and len(looks) == sm.FALLEN_LOOKS - 1 and _doubt(m) == SETTLED, whys


def test_the_last_look_finds_again_what_the_one_before_found(counter, monkeypatch):
  # ...and with a place waiting it is made: found alone by the look before,
  # the place is taken; not found again, the doubt is over all the same and
  # the take the base's from the next scan.
  for last, end in (((-0.52, 0.04), "relocated"), ((-1.42, 0.04), "outdone")):
    m = _fallen(counter)
    looks = []

    def wide_found(pose, pts, doubt=False):
      looks.append(1)
      if len(looks) < sm.FALLEN_LOOKS - 1:
        return [_place((-0.52, 0.04), 340), _place((-1.42, 0.04), 335)]
      return [_place((-0.52, 0.04) if len(looks) < sm.FALLEN_LOOKS else last, 340)]
    monkeypatch.setattr(m, "_wide_found", wide_found)
    whys, belief = _stood(m, BESIDE, np.random.default_rng(38), (sm.FALLEN_LOOKS - 1) * sm.LOST_RUN + 1)
    assert whys[:-1] == ["outdone"] * ((sm.FALLEN_LOOKS - 1) * sm.LOST_RUN) and whys[-1] == end, whys
    assert len(looks) == sm.FALLEN_LOOKS and _doubt(m) == SETTLED, (len(looks), _doubt(m))
    assert (belief == BESIDE) == (end == "outdone")


def test_a_wider_look_weighs_places_at_the_takes_heading_and_at_the_doubts_bar(
    counter, monkeypatch):
  # MEASURED from the truth at 7,840 places over the house: by the hall's
  # flight a scan from the truth puts 0.61-0.75 of its points on walls and
  # poses turned 14-16 deg 0.78-0.96, and weighed as the wide search finds
  # its places (FOUND_SHARE, any heading in WIDE_RAD) a right belief up from
  # a fall was refused at 11 places and moved at 3. A get-up moves the body;
  # the heading is the IMU's (right to 1.6 deg in 84 falls): a place turned
  # UNIQUE_RAD from the take -- from the belief, for a fit that slid -- is
  # no better place, and one that explains under DOUBT_FOUND_SHARE of the
  # scan is none.
  m = _fallen(counter)
  m.refresh(BESIDE[0], BESIDE[1])
  pts = m.points(ANGLES, cast(COUNTER, STOOD, np.random.default_rng(42)))
  take = _place((0.0, 0.0), 200, turn=0.05)
  for turn, outdone in ((math.radians(14.0), False), (0.05 - sm.UNIQUE_RAD - 0.01, False),
                        (math.radians(4.0), True)):
    asked = []
    monkeypatch.setattr(m, "_wide_found", lambda pose, pts, doubt=False: (
      asked.append(doubt), [_place((-1.0, 0.3), 320, turn)])[1])
    m.fell()
    assert m._outdone(BESIDE, pts, take)[0] is outdone and asked == [True], turn
    # ...and for a fit that slid, from the heading the robot believes
    assert bool(m._look(BESIDE, pts, None)) == (abs(turn) < sm.UNIQUE_RAD), turn
  # ⚠ ...its places the wide search's own, each at the heading the scan fits
  # it best in WIDE_RAD: looked for at the take's heading alone, places by
  # the flight that fit better turned were found there too, and three right
  # beliefs were refused and moved 0.5-1.7 m
  monkeypatch.undo()
  seen = []
  real = sm.ScanMatcher._wide_candidates
  monkeypatch.setattr(sm.ScanMatcher, "_wide_candidates", lambda self, pose, pts: (
    seen.append(1), real(self, pose, pts))[1])
  m = _fallen(counter)
  m.refresh(BESIDE[0], BESIDE[1])
  assert m._look(BESIDE, pts, None) and seen == [1]
  # ...at the doubt's bar: set past any scan, the counter's truth is no place
  # of the look's, and the take it would have refused stands
  monkeypatch.setattr(sm, "DOUBT_FOUND_SHARE", 1.01)
  whys, belief = _stood(_fallen(counter), BEHIND, np.random.default_rng(31), 1)
  assert whys == ["found"] and 0.5 < belief[0] - STOOD[0] < 0.7, (whys, belief)


def test_a_wider_look_is_made_once_in_its_scans_whatever_comes_between(counter, monkeypatch):
  # ~80 ms on the physics thread: a look is made once in LOST_RUN scans the
  # matcher is handed, and what it found waits for the next whatever comes
  # between -- a scan with too little under it, a fit that slid, which each
  # break the plain run of refusals (#681's review: looked at afresh after
  # every such scan, a take was searched every other scan; and flown, fits
  # slid and takes refused in turn never came to a second search, 563 scans
  # in a minute). A take between two looks is refused as the last one was.
  looks = _looks(monkeypatch)
  m = _fallen(counter)
  rng = np.random.default_rng(39)
  empty = np.full(len(ANGLES), 8.0)
  whys = []
  for k in range(sm.LOST_RUN + 1):
    whys.append(m.match(BESIDE, ANGLES, empty if k % 2 else cast(COUNTER, STOOD, rng)).why)
  assert whys == ["outdone", "sparse"] * (sm.LOST_RUN // 2) + ["relocated"], whys
  assert looks == ["look", "look"] and m.lost_run == 0, looks


def test_at_the_next_look_the_take_is_weighed_afresh(counter, monkeypatch):
  # Refused at the stand for a better place, a take that is that place's
  # rival by the time of the next look is outdone no longer: nothing is
  # relocated, and the take stands.
  m = _fallen(counter)
  rng = np.random.default_rng(35)
  whys, belief = _stood(m, BESIDE, rng, sm.LOST_RUN)
  assert whys == ["outdone"] * sm.LOST_RUN and m.better is not None, whys
  monkeypatch.setattr(sm, "RIVAL_SHARE", 0.6)
  whys, belief = _stood(m, belief, rng, 1)
  assert whys == ["ok"] and _doubt(m) == SETTLED, whys
  assert 0.5 < belief[0] - STOOD[0] < 0.7, belief


def test_up_from_a_fall_a_copy_of_the_room_a_step_away_takes_no_right_belief(store_copy):
  # A wider look's find is a PLACE, as the wide search's are (UNIQUE_M): on
  # the storeroom with a copy laid 0.3 m along its counter the look finds
  # the copy at every pose -- it explains more of a scan than the room does,
  # the room 0.78-0.81 of its points on walls -- and a right belief up from
  # a fall stands, as one not up from a fall does (the third round's rule: a
  # copy captures no right belief). Refused for any better POSE, the belief
  # was moved onto the copy at 7 poses of 7.
  g, poses = store_copy
  for k, p in enumerate(poses):
    ranges = cast(STORE, p, np.random.default_rng(30 + k))
    m = _fallen(g)
    got = m.match(p, ANGLES, ranges)
    assert got.why == "ok" and abs(got.pose[0] - p[0]) < 0.05 and not m.fallen, (k, got)
    # ...the premise: the look's best place is the copy
    pts = m.points(ANGLES, ranges)
    at, _, kw = m._judge(p, pts, m._fit(p, pts))
    best = max(m._look(p, pts, (at, kw)), key=lambda f: f[1]["inliers"])
    assert 0.2 < best[0][0] - p[0] < 0.4 and best[1]["inliers"] > kw["inliers"], (k, best)


def test_up_from_a_fall_the_map_is_believed_and_a_copy_further_off_that_explains_more_is_taken(
    store):
  # What the wider look costs, stated: it believes the MAP. Where a map holds
  # a copy of the room UNIQUE_M or more away that explains more of a scan
  # than the room does -- one laid through an error that size, its rays
  # having cleared the room's own wall -- a robot up from a fall is taken to
  # the copy, even one the get-up left where it believed (1 fall of 84 at
  # the counter did, within 0.3 m). Not up from a fall its fit stands, as it
  # did. On the four live maps of 11-08 and 11-09, which hold copies, no
  # right belief was moved at 160 poses (#681's review; docs/SimNotes.md).
  g = copy.deepcopy(store)
  rng = np.random.default_rng(5)
  work = [(rng.uniform(5.0, 5.4), AT_COUNTER[1] + rng.normal(0, 0.15), rng.uniform(-0.6, 0.6))
          for _ in range(24)]
  noise = np.random.default_rng(11)
  for p in work:
    for _ in range(3):
      g.update((p[0] + 0.7, p[1], p[2]), ANGLES, cast(STORE, p, noise), 8.0, origin=(0.0, 0.0),
               free_from=BAND_FF)
  for fallen, moved in ((False, 0.0), (True, 0.7)):
    m = sm.ScanMatcher(g)
    if fallen:
      m.fell()
    rng, belief = np.random.default_rng(30), AT_COUNTER
    for _ in range(sm.LOST_RUN + 1):
      got = m.match(belief, ANGLES, cast(STORE, AT_COUNTER, rng))
      belief = got.pose if got.accepted else belief
    assert abs(belief[0] - AT_COUNTER[0] - moved) < 0.05, (fallen, belief, got)


def test_a_verdict_that_neither_takes_nor_slid_is_not_looked_round(room, monkeypatch):
  # Only a take is checked, and a fit that slid looked round: a verdict with
  # too little under it to say (no direction fixed) is what it was, up from
  # a fall too, and costs no look -- nor does a fit the map refused, which
  # counts toward the wide search's own run as it always did.
  m = _fallen(room)
  kw = {"inliers": 50, "rms_m": 0.02, "weakest": 0.0, "weak_dir": (1.0, 0.0, 0.0), "held": 3}
  monkeypatch.setattr(m, "_wide_found", lambda *a, **k: pytest.fail("a wider look for no take"))
  monkeypatch.setattr(m, "_search", lambda *a, **k: None)
  for why, run in (("unconstrained", 0), ("inconsistent", 1)):
    monkeypatch.setattr(m, "_judge", lambda asked, pts, fit, **k: (asked, why, dict(kw)))
    got = m.match(TRUE, ANGLES, cast(ROOM, TRUE, np.random.default_rng(36)))
    assert got.why == why and not got.accepted and m.fallen and m.lost_run == run
    assert m.looks_left == sm.FALLEN_LOOKS


def test_a_better_place_wider_waits_for_the_next_look_and_a_restart_keeps_it(counter, monkeypatch):
  # Once is a coincidence, for a wider look too: its find waits to be found
  # again by the next, LOST_RUN scans on -- no look between, every scan
  # refused by a verdict of its own (`outdone`: not laid in, the belief
  # unmoved, and no refusal of the run the plain wide search waits on) --
  # and a restart between the two relocates on the same scan (issue #345's
  # rule).
  looks = _looks(monkeypatch)
  rng = np.random.default_rng(34)
  scans = [cast(COUNTER, STOOD, rng) for _ in range(sm.LOST_RUN + 1)]
  a = _fallen(counter)
  for r in scans[:4]:
    got = a.match(BESIDE, ANGLES, r)
    assert got.why == "outdone" and got.pose == BESIDE and got.step == sm.NO_STEP
    assert not got.accepted and not got.fuse and not a.fuses(got, 9.0)
  assert a.better is not None and a.fallen and looks == ["look"] and (a.lost_run, a.pending) == (0, None)
  assert a.counts == {"outdone": 4}
  state, arrays = a.kept_state()
  assert state["better"] == list(a.better) and state["rivals"] is False
  assert (state["looksLeft"], state["widerWait"]) == (sm.FALLEN_LOOKS - 1, sm.LOST_RUN - 3)
  b = sm.ScanMatcher(counter)
  b.restore_kept(state, arrays)
  assert _doubt(b) == _doubt(a) and b.turned_from == a.turned_from
  for r in scans[4:]:
    got_a, got_b = a.match(BESIDE, ANGLES, r), b.match(BESIDE, ANGLES, r)
  assert got_a.why == got_b.why == "relocated" and got_a.pose == got_b.pose
  assert looks == ["look"] * 3 and not a.fallen and not b.fallen
  # ...places it could not tell apart are kept too; a state from before the
  # rule is no fall's
  a = _fallen(counter)
  a.rivals, a.wider_wait = True, 3
  b.restore_kept(*a.kept_state())
  assert _doubt(b) == _doubt(a) == (True, sm.FALLEN_LOOKS, 3, None, True)
  old = {k: v for k, v in a.kept_state()[0].items()
         if k not in ("looksLeft", "widerWait", "better", "rivals")}
  b.restore_kept(old, {})
  assert _doubt(b) == SETTLED


def test_between_two_looks_the_last_ones_word_stands(counter):
  # Flown at the cube's end of the counter: the first find refused, the
  # search's next lay 0.43 m from the truth -- one place with it, to the
  # wide search -- between the truth and the place refused, and the fits
  # after it slid back onto that place. Until the next look a take is the
  # find that waits, to a fit's own width, or it waits too.
  m = _fallen(counter)
  m.refresh(BEHIND[0], BEHIND[1])
  pts = m.points(ANGLES, cast(COUNTER, STOOD, np.random.default_rng(31)))
  kw = {"inliers": 200, "rms_m": 0.03, "weakest": 40.0, "weak_dir": (1.0, 0.0, 0.0), "held": 0}
  waits = (STOOD[0] - BEHIND[0], STOOD[1] - BEHIND[1], 0.0)
  for off, outdone in ((0.43, True), (sm.DOUBT_SLID_M + 0.02, True), (0.03, False)):
    m.better, m.wider_wait = waits, 5
    step = (waits[0] + off, waits[1], 0.0)
    take = ((BEHIND[0] + step[0], BEHIND[1] + step[1], BEHIND[2]), dict(kw, step=step))
    assert m._outdone(BEHIND, pts, take) == (outdone, None), off
    assert (m.better, m.wider_wait, m.looks_left) == (waits, 5, sm.FALLEN_LOOKS), "no look while one waits"
  # ...places the look could not tell apart outdo every take, a look that
  # held no place leaves it the base's, and a fit that slid is not judged
  m.better, m.rivals = None, True
  assert m._outdone(BEHIND, pts, take) == (True, None)
  m.rivals = False
  assert m._outdone(BEHIND, pts, take) == (False, None)
  m.better, m.rivals = waits, True
  assert m._outdone(BEHIND, pts, None) == (False, None)
  assert m.wider_wait == 5 and m.looks_left == sm.FALLEN_LOOKS


def test_a_wider_look_is_found_again_by_the_next_look_and_never_by_the_plain_wide_search(
    counter, monkeypatch):
  # The wide search's rule, for a wider look too: once is a coincidence. But
  # the second find is the next LOOK's -- at the take's heading and the
  # doubt's bar, as the first -- never the plain wide search's, which looks
  # at any heading at its own bar and whose turn a run of refusals standing
  # just short of LOST_RUN brought two scans on (#681's review): a take the
  # look refused is no refusal of that run, and breaks it as any other
  # verdict does; and a place the plain search holds for its own second
  # (`pending`) refuses no take.
  looks = _looks(monkeypatch)
  m = _fallen(counter)
  m.lost_run, m.pending, m.turned_seen = sm.LOST_RUN - 2, (-1.8, 0.0, 0.0), (BESIDE, BESIDE)
  whys, belief = _stood(m, BESIDE, np.random.default_rng(37), 1)
  assert whys == ["outdone"] and (m.lost_run, m.pending, m.turned_seen) == (0, None, None)
  m.lost_run, m.pending = sm.LOST_RUN - 2, (-1.8, 0.0, 0.0)
  whys, belief = _stood(m, belief, np.random.default_rng(37), sm.LOST_RUN)
  assert whys == TAKEN_BACK[1:] and looks == ["look", "look"], (whys, looks)
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03, belief
  # ...and a right take stands at once beside it
  looks.clear()
  m = _fallen(counter)
  m.lost_run, m.pending = sm.LOST_RUN + 1, (-1.8, 0.0, 0.0)
  whys, belief = _stood(m, (STOOD[0] + 0.45, STOOD[1] + 0.40, STOOD[2]), np.random.default_rng(33), 1)
  assert whys == ["found"] and looks == ["look"] and m.pending is None, (whys, looks)
  assert math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03, belief


def test_the_falls_doubt_ends_once_the_belief_has_walked_on(counter, monkeypatch):
  # ...or with a take that stands, a fix or a warp. Walked FALLEN_NEAR_M from
  # where it stood up with nothing taken -- open ground, its scans too thin
  # to say -- the robot is no longer a get-up's to answer for: its first fit
  # there is taken as any robot's (#681's review: the doubt had no end, and
  # a fit minutes and tens of metres on was looked at as a fall's).
  looks = _looks(monkeypatch)
  m = _fallen(counter)
  empty = np.full(len(ANGLES), 8.0)
  assert m.match((STOOD[0] - sm.FALLEN_NEAR_M + 0.1, STOOD[1], STOOD[2]), ANGLES, empty).why == "sparse"
  assert m.fallen and m.turned_from == (STOOD[0] - sm.FALLEN_NEAR_M + 0.1, STOOD[1])
  assert m.match((STOOD[0] + 0.03, STOOD[1], STOOD[2]), ANGLES, empty).why == "sparse" and m.fallen
  whys, belief = _stood(m, BESIDE, np.random.default_rng(40), 1)
  assert whys == ["ok"] and not m.fallen and not looks, (whys, looks)
  for on in (True, False):
    m = _fallen(counter)
    m.match(BESIDE, ANGLES, cast(COUNTER, STOOD, np.random.default_rng(41)))
    assert m.fallen and m.wider_wait and m.better is not None and m.looks_left == sm.FALLEN_LOOKS - 1
    m.anchored(on)
    assert _doubt(m) == SETTLED
  m = _fallen(counter)
  m.match(BESIDE, ANGLES, cast(COUNTER, STOOD, np.random.default_rng(41)))
  m.forget()
  assert _doubt(m) == SETTLED
  m.rivals = True
  m.fell()
  assert _doubt(m) == (True, sm.FALLEN_LOOKS, 0, None, False)


def test_not_up_from_a_fall_a_find_stands_whatever_the_wide_search_holds(counter, monkeypatch):
  # Only a fall puts the pose in doubt: a robot whose map refused it, and
  # whose wide search holds a place still to be found again, takes its
  # bounded search's find at once as it always did -- on a sidewalk whose
  # wall the map had eroded that place was 1.8 m along and wrong (#401), and
  # the find the truth (#681's review).
  looks = _looks(monkeypatch)
  m = sm.ScanMatcher(counter)
  m.pending, m.lost_run = (-1.8, 0.0, 0.0), sm.LOST_RUN + 1
  whys, belief = _stood(m, (STOOD[0] + 0.45, STOOD[1] + 0.40, STOOD[2]), np.random.default_rng(33), 1)
  assert whys == ["found"] and math.hypot(belief[0] - STOOD[0], belief[1] - STOOD[1]) < 0.03
  assert not looks and m.pending is None


def test_a_fall_closes_an_anchors_window(counter):
  # Fixed off the station's tags and shoved over before the window closed
  # (the station's own fall scene), the scans after the get-up were laid
  # unmatched at a belief the get-up had just left behind: a copy of the
  # room (#681's review). Up from a fall they are matched.
  m = sm.ScanMatcher(counter)
  m.anchored()
  assert m.match(BESIDE, ANGLES, cast(COUNTER, STOOD, np.random.default_rng(43))).why == "anchored"
  m.fell()
  assert m.anchoring == 0
  assert m.match(BESIDE, ANGLES, cast(COUNTER, STOOD, np.random.default_rng(43))).why == "outdone"


# ---- a robot up from a fall, its heading turned (issue #690) -----------------


#: Up from a fall 0.5 m from TRUE, its heading turned 120 deg: what the other
#: robot's blocking interrupt left Luca with on 11-12.
TURNED = (TRUE[0] - 0.4, TRUE[1] + 0.3, TRUE[2] + math.radians(120.0))
#: ROOM is nearly the same turned half round (its two long stubs are), and
#: so is any box: an L, its north-east corner walled off, is not.
NOOK = [(0.013, 0.021, 6.013, 0.021), (6.013, 0.021, 6.013, 2.521), (6.013, 2.521, 4.513, 2.521),
        (4.513, 2.521, 4.513, 4.021), (4.513, 4.021, 0.013, 4.021), (0.013, 4.021, 0.013, 0.021),
        (2.013, 0.021, 2.013, 1.5), (1.0, 3.011, 1.6, 3.011)]


@pytest.fixture(scope="module")
def nook():
  rng = np.random.default_rng(0)
  poses = ([(rng.uniform(0.5, 1.8), rng.uniform(0.5, 3.5), rng.uniform(-3, 3)) for _ in range(20)]
           + [(rng.uniform(2.3, 4.2), rng.uniform(0.3, 3.7), rng.uniform(-3, 3)) for _ in range(20)]
           + [(rng.uniform(4.6, 5.7), rng.uniform(0.3, 2.2), rng.uniform(-3, 3)) for _ in range(8)])
  return mapped(NOOK, (-1, -1, 7, 5), poses)


def _deg(a):
  return abs(math.degrees(math.remainder(a, 2 * math.pi)))


def test_a_robot_up_from_a_fall_its_heading_turned_is_found_at_every_heading(nook):
  # Its odometry held through the get-up and its heading its IMU's, a
  # heading the fall turned unseen is past the wide search's 15 deg for good:
  # live, every scan refused for 26 minutes. Up from a fall (`fell`), a wide
  # search that finds nothing looks again at every heading, and a pose found
  # twice is taken. Never fallen, the same scans are refused.
  rng = np.random.default_rng(41)
  scans = [cast(NOOK, TRUE, rng) for _ in range(2 * sm.LOST_RUN)]
  never = sm.ScanMatcher(nook)
  assert [never.match(TURNED, ANGLES, r).why for r in scans] == ["inconsistent"] * len(scans)
  m = sm.ScanMatcher(nook)
  m.fell()
  whys = [m.match(TURNED, ANGLES, r).why for r in scans]
  assert whys == ["inconsistent"] * (len(scans) - 1) + ["relocated"], whys
  assert m.last.accepted and not m.last.fuse
  assert math.hypot(m.last.pose[0] - TRUE[0], m.last.pose[1] - TRUE[1]) < 0.03
  assert _deg(m.last.pose[2] - TRUE[2]) < 0.5
  assert m.turned_left == 0, "a pose taken still searches every heading"


#: A hall symmetric under a half turn, as the lobby (3 x 12 m) nearly is.
HALL = box(0.013, 0.021, 9.013, 3.021)


def test_the_search_at_every_heading_takes_nothing_the_scan_turned_half_round_fits_as_well(
    monkeypatch):
  # MEASURED in the lobby: a scan turned 180 deg put 0.90-1.11 of the
  # truth's points on walls (median 0.99) and was the best in 115 of 312
  # searches; taken, the robot would face the wrong way down the hall. Two
  # places at once are a rival (TURNED_RIVAL_SHARE), and nothing is taken.
  rng = np.random.default_rng(42)
  hall = mapped(HALL, (-1, -1, 10, 4), [(rng.uniform(0.6, 8.4), rng.uniform(0.6, 2.4),
                                         rng.uniform(-3, 3)) for _ in range(30)])
  # ...its half turn 1.1 m from the truth, both within the search's reach
  true = (4.0, 1.3, 0.2)
  turned = (3.6, 1.6, 0.2 + math.radians(120.0))
  scans = [cast(HALL, true, rng) for _ in range(2 * sm.LOST_RUN)]
  m = sm.ScanMatcher(hall)
  m.fell()
  assert [m.match(turned, ANGLES, r).why for r in scans] == ["inconsistent"] * len(scans)
  # ...the premise: with no rival counted, it takes one of the two
  monkeypatch.setattr(sm, "TURNED_RIVAL_SHARE", 2.0)
  m = sm.ScanMatcher(hall)
  m.fell()
  assert [m.match(turned, ANGLES, r).why for r in scans][-1] == "relocated"


def test_the_search_at_every_heading_waits_on_the_served_wide_search(room, monkeypatch):
  # Up from a fall with its heading right and its belief past the bounded
  # search -- most get-ups: a fall moves the body up to 2 m (the lobby's 118
  # stood 0.71 m off at the median) -- the served wide search relocates it as
  # before, and the search at every heading never runs. MEASURED, run
  # first in the lobby, it refused 10 relocations the served one made right:
  # the hall's scan turned half round was its rival.
  m = sm.ScanMatcher(room)
  m.fell()
  ran = []
  real = m._relocate_turned
  monkeypatch.setattr(m, "_relocate_turned", lambda pose, pts: (ran.append(1), real(pose, pts))[1])
  rng = np.random.default_rng(43)
  whys = [m.match(LOST, ANGLES, cast(ROOM, TRUE, rng)).why for _ in range(2 * sm.LOST_RUN)]
  assert whys[-1] == "relocated" and not ran


def test_the_search_at_every_heading_takes_no_pose_a_turned_one_explains_nearly_as_well(
    room, monkeypatch):
  # MEASURED walking on knocked 40 deg in the lobby: the scan turned half
  # round put 333, then 317, points on walls where the truth's put 296 and
  # 275 -- 0.889 and 0.867 of them, under RIVAL_SHARE's 0.9 -- and was taken
  # twice running, 3.03 m and 180 deg off. At every heading the half turn is
  # always looked at: a find TURNED_RIVAL_SHARE as good is its rival.
  half = (TRUE[0] + 2.0, TRUE[1], TRUE[2] + math.pi)
  inliers = {TRUE: 296, half: 333}
  m = sm.ScanMatcher(room)
  monkeypatch.setattr(m, "_turned_candidates", lambda pose, pts: [TRUE, half])
  monkeypatch.setattr(m, "_search", lambda start, pts, **kw: start)
  monkeypatch.setattr(m, "_fit", lambda start, pts: start)
  monkeypatch.setattr(m, "_judge", lambda asked, pts, fit, found=False, doubt=False: (
    fit, "found", {"inliers": inliers[fit], "step": (0.0, 0.0, 0.0)}))
  assert m._relocate_turned(TURNED, None) is None
  # ...the premise: at the wide search's own share, the half turn is taken
  monkeypatch.setattr(sm, "TURNED_RIVAL_SHARE", sm.RIVAL_SHARE)
  assert m._relocate_turned(TURNED, None)[0] == half


def _walking(k, per_scan):
  """Scan k of a robot up from a fall standing its first LOST_RUN scans,
  then walking `per_scan` a scan: (the truth, the belief that walked the
  same steps through its heading turned 120 deg)."""
  d = per_scan * max(0, k - (sm.LOST_RUN - 1))
  return ((TRUE[0] + d * math.cos(TRUE[2]), TRUE[1] + d * math.sin(TRUE[2]), TRUE[2]),
          (TURNED[0] + d * math.cos(TURNED[2]), TURNED[1] + d * math.sin(TURNED[2]), TURNED[2]))


def _walked(nook, per_scan, seed=44):
  m = sm.ScanMatcher(nook)
  m.fell()
  rng = np.random.default_rng(seed)
  for k in range(2 * sm.LOST_RUN):
    truth, belief = _walking(k, per_scan)
    got = m.match(belief, ANGLES, cast(NOOK, truth, rng))
  return m, got, truth


def test_a_robot_walking_on_turned_is_found_again_where_odometry_carried_it(nook, monkeypatch):
  # Walking on with its heading turned, its belief goes another way than its
  # body, and the correction from it the next search finds is another: 0.4 m
  # between them at 120 deg off moves it 0.7 m. The pose found, carried by
  # odometry from the belief it was found from to the next search's, is the
  # same.
  m, got, truth = _walked(nook, 0.04)
  assert got.why == "relocated"
  assert math.hypot(got.pose[0] - truth[0], got.pose[1] - truth[1]) < 0.03
  # ...the premise: compared as the served search compares, its corrections
  monkeypatch.setattr(sm, "_carried", lambda found, then, now: (
    found[0] - then[0] + now[0], found[1] - then[1] + now[1], found[2] - then[2] + now[2]))
  m, got, truth = _walked(nook, 0.04)
  assert got.why == "inconsistent" and m.turned_seen is not None


def test_a_robot_that_walked_on_from_where_it_stood_up_searches_no_more(nook, monkeypatch):
  # Walked on with its heading off, its belief has walked another way than
  # its body and the truth leaves the lattice round it, and in a hall the
  # scan turned half round is left there to fit unrivalled: MEASURED,
  # walking on knocked 120 deg in the lobby, 5 of 118 were taken onto it,
  # each 1.24-1.73 m from where it stood up; every pose taken right was found
  # within 0.88 m. Past TURNED_NEAR_M, the search at every heading is over.
  m, got, _ = _walked(nook, 0.07)
  assert got.why == "inconsistent" and m.turned_left == 0
  # ...the premise: with the stand anywhere near, the same walk is taken
  monkeypatch.setattr(sm, "TURNED_NEAR_M", 2.0)
  m, got, truth = _walked(nook, 0.07)
  assert got.why == "relocated"
  assert math.hypot(got.pose[0] - truth[0], got.pose[1] - truth[1]) < 0.03


def _counted(m, monkeypatch, found_from=None):
  """`m`'s searches at every heading counted, each finding nothing -- or,
  from the `found_from`th on, a pose where the robot stands -- and the
  searches it does not count stubbed: nothing to find, in 0 s."""
  ran = []
  kw = {"inliers": 300, "rms_m": 0.01, "weakest": 100.0, "weak_dir": (1.0, 0.0, 0.0),
        "held": 0, "step": (0.0, 0.0, 0.0)}

  def turned(pose, pts):
    ran.append(1)
    return (tuple(pose), kw) if found_from is not None and len(ran) >= found_from else None
  monkeypatch.setattr(m, "_relocate_turned", turned)
  monkeypatch.setattr(m, "_search", lambda *a, **k: None)
  monkeypatch.setattr(m, "_relocate", lambda pose, pts: None)
  return ran


#: A scan of another room's walls where the map has its own: every fit refused.
ELSEWHERE = box(0.4, 0.7, 3.1, 2.9) + [(1.7, 0.7, 1.7, 1.6)]


def test_the_search_at_every_heading_ends_at_its_count_or_a_scan_that_fits(room, monkeypatch):
  # TURNED_SEARCHES of them, ~0.1 s each on the physics thread, and only
  # until the map fits a scan: a robot lost for good otherwise searched
  # every heading every second (Luca's 26 minutes). The last runs only to
  # confirm the one before (#690's review): what it found alone, no search
  # after it would confirm.
  m = sm.ScanMatcher(room)
  m.fell()
  ran = _counted(m, monkeypatch)
  elsewhere = cast(ELSEWHERE, (1.2, 1.6, 0.9), np.random.default_rng(45))
  for _ in range((sm.TURNED_SEARCHES + 2) * sm.LOST_RUN):
    assert m.match(TRUE, ANGLES, elsewhere).why == "inconsistent"
  assert len(ran) == sm.TURNED_SEARCHES - 1 and m.turned_left == 0
  # ...and one found by the last but one is confirmed by the last
  m = sm.ScanMatcher(room)
  m.fell()
  ran = _counted(m, monkeypatch, found_from=sm.TURNED_SEARCHES - 1)
  whys = [m.match(TRUE, ANGLES, elsewhere).why for _ in range(sm.TURNED_SEARCHES * sm.LOST_RUN)]
  assert whys[-1] == "relocated" and len(ran) == sm.TURNED_SEARCHES
  m.fell()
  assert m.match(TRUE, ANGLES, cast(ROOM, TRUE, np.random.default_rng(46))).accepted
  assert m.turned_left == 0 and m.turned_seen is None


def test_a_fall_starts_the_run_of_refusals_again(room, monkeypatch):
  # #690's review: what the wide search found before the fall is gone with
  # the run -- the fall moved the body under it, up to 2 m and turned -- and
  # the next wide search comes LOST_RUN refusals after the stand, never the
  # rest of a run begun before the fall.
  m = sm.ScanMatcher(room)
  rng = np.random.default_rng(48)
  for _ in range(sm.LOST_RUN + 3):
    m.match(LOST, ANGLES, cast(ROOM, TRUE, rng))
  assert m.pending is not None and m.lost_run == sm.LOST_RUN + 3, "the premise: a run, and a find"
  m.fell()
  assert (m.lost_run, m.pending) == (0, None)
  searched = []
  monkeypatch.setattr(m, "_relocate", lambda pose, pts: searched.append(1))
  monkeypatch.setattr(m, "_relocate_turned", lambda pose, pts: None)
  elsewhere = cast(ELSEWHERE, (1.2, 1.6, 0.9), np.random.default_rng(49))
  for _ in range(sm.LOST_RUN - 1):
    m.match(LOST, ANGLES, elsewhere)
  assert not searched
  m.match(LOST, ANGLES, elsewhere)
  assert searched == [1]


def test_a_warp_ends_the_search_at_every_heading(room):
  # #690's review: `start_at` (a warp: a restart put clear of the other
  # robot, a script) puts the belief somewhere new, closing the anchor's
  # window (`anchored(on=False)`); a search armed by a fall before it is
  # over too, or it would carry what it found to a belief already right.
  m = sm.ScanMatcher(room)
  m.fell()
  m.match(TURNED, ANGLES, cast(ROOM, TRUE, np.random.default_rng(50)))
  assert m.turned_left and m.turned_from is not None, "the premise: armed, and where it stood up"
  m.anchored(on=False)
  assert (m.turned_left, m.turned_seen, m.turned_from) == (0, None, None)


def test_a_scan_with_too_few_returns_near_offers_no_heading(room):
  # #690's review: the lattice scores the returns within TURNED_RANGE_M, and
  # on open ground with its walls 6-8 m off there are none: every pose read
  # alike, and five fits ran from arbitrary starts for nothing.
  m = sm.ScanMatcher(room)
  m.refresh(*TRUE[:2])
  far = np.full(len(ANGLES), sm.TURNED_RANGE_M + 0.5)
  pts = m.points(ANGLES, far)
  assert len(pts) >= sm.MIN_INLIERS, "the premise: a scan the matcher reads"
  assert m._turned_candidates(TRUE, pts) == []
  near = m.points(ANGLES, cast(ROOM, TRUE, np.random.default_rng(51)))
  assert len(m._turned_candidates(TRUE, near)) == sm.TURNED_TRIES


def test_a_restart_keeps_the_search_at_every_heading(nook):
  # Both decide when the robot relocates (issue #345's rule): restarted
  # after its first search at every heading, it relocates on the same scan
  # as a robot never stopped.
  rng = np.random.default_rng(47)
  scans = [cast(NOOK, TRUE, rng) for _ in range(2 * sm.LOST_RUN)]
  a = sm.ScanMatcher(nook)
  a.fell()
  for r in scans[:sm.LOST_RUN + 2]:
    a.match(TURNED, ANGLES, r)
  state, arrays = a.kept_state()
  assert state["turnedLeft"] == sm.TURNED_SEARCHES - 1 and state["turnedSeen"] is not None
  assert state["turnedFrom"] == list(TURNED[:2])
  b = sm.ScanMatcher(nook)
  b.restore_kept(state, arrays)
  for r in scans[sm.LOST_RUN + 2:]:
    got_a, got_b = a.match(TURNED, ANGLES, r), b.match(TURNED, ANGLES, r)
  assert got_a.why == got_b.why == "relocated" and got_a.pose == got_b.pose


#: The copy a robot come home lost lays: the room's walls, laid through
#: poses this far off the ones they were seen from.
ASKEW = (0.6, -0.5, math.radians(2.0))


def _askew_room():
  rng = np.random.default_rng(31)
  g = OccupancyGrid(-2, -2, 8, 6, 0.05)
  for _ in range(30):
    p = (rng.uniform(1.5, 4.5), rng.uniform(1.0, 3.0), rng.uniform(-3, 3))
    laid = (p[0] + ASKEW[0], p[1] + ASKEW[1], p[2] + ASKEW[2])
    g.update(laid, ANGLES, cast(ROOM, p, rng), 8.0, origin=(0.0, 0.0))
  return g


def _back_off(m, g, rng, steps):
  """A back-out from the fixture at TRUE, 3 cm a scan, each scan matched and
  laid in as `Navigator._scan_step` does: the verdicts and the last pose."""
  whys, pose = [], TRUE
  for k in range(steps):
    at = (TRUE[0] - 0.03 * k * math.cos(TRUE[2]), TRUE[1] - 0.03 * k * math.sin(TRUE[2]), TRUE[2])
    got = m.match(at, ANGLES, cast(ROOM, at, rng))
    whys.append(got.why)
    pose = got.pose
    if m.fuses(got, 0.1 * k):
      g.update(pose, ANGLES, cast(ROOM, at, rng), 8.0, origin=(0.0, 0.0))
      m.fused(pose, 0.1 * k)
  return whys, pose


def test_an_anchor_lays_the_room_again_over_a_copy_laid_askew():
  # The commissioned fixture outranks the map (issue #422; the quadrupeds'
  # dock then, the humanoids' station now): anchored on it, a robot come
  # home lost had a copy of the living room laid 0.9 m askew, and matching
  # against it pulled the belief 0.62 m back into it as the robot backed
  # off. Anchored, the back-out is laid in unmatched, the copy is
  # overwritten, and the next scan matches where the robot stands.
  g = _askew_room()
  m = sm.ScanMatcher(g)
  m.anchored()
  rng = np.random.default_rng(32)
  whys, pose = _back_off(m, g, rng, sm.ANCHORED_SCANS)
  assert whys == ["anchored"] * sm.ANCHORED_SCANS
  at = (TRUE[0] - 0.03 * sm.ANCHORED_SCANS * math.cos(TRUE[2]),
        TRUE[1] - 0.03 * sm.ANCHORED_SCANS * math.sin(TRUE[2]), TRUE[2])
  got = m.match(at, ANGLES, cast(ROOM, at, rng))
  assert got.why == "ok" and math.hypot(got.pose[0] - at[0], got.pose[1] - at[1]) < 0.03
  # ...the premise: matched as it backs off, the copy has the belief
  m = sm.ScanMatcher(_askew_room())
  _, pose = _back_off(m, m.grid, np.random.default_rng(32), sm.ANCHORED_SCANS)
  assert math.hypot(pose[0] - at[0], pose[1] - at[1]) > 0.3


def test_a_restart_keeps_an_anchors_window(room):
  m = sm.ScanMatcher(room)
  m.anchored()
  m.fused(TRUE, 0.0)
  state, arrays = m.kept_state()
  n = sm.ScanMatcher(room)
  n.restore_kept(state, arrays)
  assert n.anchoring == sm.ANCHORED_SCANS - 1
  assert n.match(TRUE, ANGLES, cast(ROOM, TRUE)).why == "anchored"


def test_the_stations_look_alone_opens_the_window(house):
  # The commissioned fixture's look is the one anchor (#422, #476; the
  # quadrupeds' was their dock's board): the humanoid's is its station, off
  # its two tags. The start opens no window -- a look-around laid unmatched
  # smears the gyro's scale error -- nor does a look that agrees with the
  # belief; a look that moves the belief does, and a stand-up there closes
  # the window a death at the station left.
  from pluggybot.body import RackPose
  body = _humanoid(house)
  m, s = body.mission, body._station
  assert m.matcher.anchoring == 0, "the start opened the window"
  assert body._fix_at(s) is None and m.matcher.anchoring == 0, "a look that agrees"
  body._fix_at(RackPose(s.x + 0.3, s.y - 0.2, s.yaw))     # the belief 0.36 m off
  assert m.matcher.anchoring == sm.ANCHORED_SCANS
  body.start_at(*m.pose)                                   # ...and a stand-up
  assert m.matcher.anchoring == 0, "a stand-up kept the station's window"
  body.close()


def test_a_return_within_the_hands_reach_is_no_point_of_the_fit():
  """What the band sees nearer than `MATCH_MIN_M` is what the hands work on
  -- the station's box, the plug in the hand, a drawer -- struck by a ring
  or two at the part's noisiest (#581's stage 2): left out of the fit, and
  the walls beyond it in."""
  m = sm.ScanMatcher(OccupancyGrid(-3, -3, 7, 7), origin=(0.0, 0.0))
  ranges = [0.4, sm.MATCH_MIN_M - 0.01, sm.MATCH_MIN_M, 3.0, sm.MATCH_RANGE_M + 0.1, 8.0]
  pts = m.points(np.zeros(len(ranges)), ranges)
  assert np.allclose(pts[:, 0], [sm.MATCH_MIN_M, 3.0]) and np.allclose(pts[:, 1], 0.0)
