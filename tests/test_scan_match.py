"""Scan matching (issue #386), pinned on synthetic scans: rooms and
corridors drawn as segments, scans cast analytically with the LIDAR's own
noise, maps fused from known poses. Nothing here flies but the one proof
behind `--endurance` at the end; the flights that measured it are
`scripts/drift_spike.py`."""

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


def test_a_fit_that_would_slide_past_the_gate_is_refused(room):
  # 17 cm off is a pose the fit can pull back, but not in one scan: a
  # correction past MAX_STEP is another wall to sit on, and is refused.
  m = sm.ScanMatcher(room)
  ranges = cast(ROOM, TRUE, np.random.default_rng(5))
  guess = (TRUE[0] + 0.14, TRUE[1] - 0.10, TRUE[2] + 0.08)
  got = m.match(guess, ANGLES, ranges)
  assert not got.accepted
  assert got.pose == guess


def test_a_scan_with_no_map_under_it_is_no_match():
  g = OccupancyGrid(-1, -1, 7, 5, 0.05)
  got = sm.ScanMatcher(g).match(TRUE, ANGLES, cast(ROOM, TRUE))
  assert not got.accepted and got.why == "no map"


def test_returns_near_the_lidars_reach_are_left_out():
  # A wall at the edge of reach is seen short: its long draws are clipped
  # to "no return", and the map clears it with the same rays. Matched, the
  # street's far wall pulled the pose 3 cm toward it every scan (MEASURED
  # on a recorded lab trip: +31.6 mm mean, 0.0 once cut at 6 m).
  m = sm.ScanMatcher(OccupancyGrid(-1, -1, 7, 5, 0.05), max_range=8.0)
  ranges = np.array([1.0, sm.MATCH_RANGE_M - 0.01, sm.MATCH_RANGE_M + 0.01, 7.9, 8.0])
  pts = m.points(np.zeros(5), ranges)
  assert len(pts) == 2


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
  # ~0.4 ms against ~1.6 ms.
  big = OccupancyGrid(-17.5, -11.5, 33.5, 11.5, 0.05)
  ix0, iy0 = big.world_to_cell(-1.0, -1.0)
  big.grid[iy0:iy0 + room.grid.shape[0], ix0:ix0 + room.grid.shape[1]] = room.grid
  m = sm.ScanMatcher(big)
  rng = np.random.default_rng(4)
  scans = [cast(ROOM, TRUE, rng) for _ in range(20)]
  m.match(TRUE, ANGLES, scans[0])
  scratch = OccupancyGrid(-17.5, -11.5, 33.5, 11.5, 0.05)
  t_match = t_update = 0.0
  for r in scans:
    t = time.perf_counter()
    m.match(TRUE, ANGLES, r)
    t_match += time.perf_counter() - t
    t = time.perf_counter()
    scratch.update(TRUE, ANGLES, r, 8.0, origin=(0.0, 0.0))
    t_update += time.perf_counter() - t
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
  # A robot standing still whose wheels claim 10 mm a scan (spinning on the
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


# ---- the rover's wiring ----------------------------------------------------------


class _FixedMatcher(sm.ScanMatcher):
  """Answers every scan with one verdict, and records what went into the map."""

  def __init__(self, grid, answer):
    super().__init__(grid)
    self.answer, self.laid = answer, []

  def match(self, pose, angles, ranges):
    return self.answer


def _rover():
  import mujoco

  from pluggybot.mission.mission import HubMission
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  m = HubMission(model, mujoco.MjData(model), viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  return m


def test_the_rover_lays_each_scan_through_the_pose_its_match_found():
  # The matched pose is the belief -- the reckoner is moved to it -- and the
  # pose the scan goes into the map at; a refused fit leaves the belief
  # where odometry had it, and the map untouched.
  m = _rover()
  laid = []
  real = m.grid.update
  m.grid.update = lambda pose, *a, **kw: (laid.append(tuple(pose)), real(pose, *a, **kw))
  target = (1.2, 0.9, 0.05)
  m.matcher = _FixedMatcher(m.grid, sm.Match(target, True, "ok", 300, 0.02, 100.0,
                                             (1.0, 0.0, 0.0), 0, (0.2, -0.1, 0.05)))
  m._next_scan = 0.0
  m._drive(0.15, 0.0, 0.0)
  assert laid and laid[0] == target
  assert math.hypot(m.swap.reckoner.x - target[0], m.swap.reckoner.y - target[1]) < 0.01
  laid.clear()
  before = m.pose
  m.matcher.answer = m.matcher.answer._replace(why="slid", accepted=False, pose=before)
  m.matcher.fused_pose = None
  m._drive(0.15, 0.0, 0.0)
  assert not laid, "a scan whose fit slid was laid into the map"


def test_a_restart_keeps_the_matchers_field_and_the_imus_stream():
  # A new piece of state that decides anything is kept (issue #345): the
  # field the next match reads and where the IMU's noise had got to.
  m = _rover()
  m._drive(1.0, 0.0, 0.0)
  state, arrays = m.kept_state()
  assert state["imu"]["rng"] and state["match"] is not None
  n = _rover()
  n.restore_kept(state, arrays)
  assert n.swap.imu.gyro_z(0.1, 0.002) == m.swap.imu.gyro_z(0.1, 0.002)
  if m.matcher.field is None:
    assert n.matcher.field is None
  else:
    assert np.array_equal(n.matcher.field, m.matcher.field, equal_nan=True)
  assert n.matcher.fused_pose == m.matcher.fused_pose


def test_a_pose_past_the_fits_reach_is_found_by_the_search(room):
  # 0.4 m off, the walls are past the field's reach and the fit cannot
  # pull the pose back: MEASURED, a wheel pump while tilted left the rover
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


# ⚠ BEHIND `--endurance`: one lab trip on the honest sensors, ~45 s of
# wall clock. Its RULES are pinned above on synthetic scans;
# what only a flight proves is the whole chain -- a gyro with its
# calibration residue and counted wheels, every scan laid through the pose
# the walls gave -- still gets the rover through the lobby's door to the
# cage. On odometry alone it gave up 0.2 m short of that door (`no route`
# at t = 83 s); matched, its worst on the way was 0.21 m.
@pytest.mark.slow
@pytest.mark.endurance
def test_the_rover_reaches_the_cage_on_honest_sensors(tmp_path):
  """The feed errand from the start pose, matcher on: the program completes
  and the belief stays within 0.3 m of the truth. Stops on its claim, the
  errand's result."""
  from pluggybot.lifecycle import cage_errand, run_demo
  worst = [0.0]

  def on_ready(life):
    life.errands[:] = [cage_errand("home", "feed")]

    def step():
      tx, ty, _ = life.body.true_pose()
      bx, by, _ = life.body.pose
      worst[0] = max(worst[0], math.hypot(bx - tx, by - ty))
    life.body.step_hooks.append(step)

  r = run_demo(view=False, realtime=False, world="home", pack="hosting",
               errand="none", max_sim_time=400.0, tasks=False, metabolism=False,
               overseer=False, thoughts_root=str(tmp_path / "thoughts"),
               ledger_state=str(tmp_path / "ledger.json"),
               board_state=str(tmp_path / "boards.json"),
               spend_state=str(tmp_path / "spend.json"),
               on_ready=on_ready, stop_when=lambda life: bool(life.errand_results))
  feed = next(e for e in r["errands"] if e["errand"] == "care:feed")
  assert feed["procedure"]["ok"], feed["procedure"]
  assert worst[0] < 0.3, worst[0]
