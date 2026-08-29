"""The vectorized scan update must be the old per-ray loop, only faster.

The pure-Python loop over 360 rays cost ~9.4 ms per scan -- the single most
expensive Python in the mission loop (issue #2). The vectorized version must
produce the same map (the log-odds semantics are load-bearing: frontier
choices and A* costs read this grid) at >=5x the speed.

Fixture: tests/fixtures/lidar_scans.npz -- 12 real Lidar scans recorded in
room_hub.xml (noise, dropout and the self-filter all active, ranges down to
2 cm from a pose overlapping scenery, so the shorter-than-one-cell edge case
is exercised).

Tolerance, stated per the acceptance criteria: the two implementations pick
bit-identical CELLS (same sample positions, same truncation), but accumulate
in a different order -- the loop adds L_FREE k times where the vectorized
version adds k*L_FREE once -- so values may differ in the last float ulps.
We assert atol=1e-9 (six orders below the 0.4 evidence quantum) and that the
thresholded to_image() views are exactly identical.
"""
import math
import time
from pathlib import Path

import numpy as np

from pluggybot.mapping.occupancy_grid import L_FREE, L_OCC, OccupancyGrid

FIXTURE = Path(__file__).parent / "fixtures" / "lidar_scans.npz"


def reference_update(grid, pose, angles, ranges, max_range, origin=(0.03, 0.03)):
  """The pre-vectorization OccupancyGrid.update, verbatim (milestone 4)."""
  px, py, theta = pose
  fwd, left = origin
  ox = px + fwd * math.cos(theta) - left * math.sin(theta)
  oy = py + fwd * math.sin(theta) + left * math.cos(theta)

  rows, cols = grid.grid.shape
  for angle, r in zip(angles, ranges):
    a = theta + angle
    hit = r < max_range - 1e-6

    free_len = r - grid.resolution if hit else r
    n = max(2, int(free_len / (grid.resolution / 2)))
    ts = np.linspace(0.0, free_len, n)
    ixs = ((ox + ts * np.cos(a) - grid.x_min) / grid.resolution).astype(int)
    iys = ((oy + ts * np.sin(a) - grid.y_min) / grid.resolution).astype(int)
    valid = (ixs >= 0) & (ixs < cols) & (iys >= 0) & (iys < rows)
    grid.grid[iys[valid], ixs[valid]] += L_FREE

    if hit:
      ix, iy = grid.world_to_cell(ox + r * math.cos(a), oy + r * math.sin(a))
      if 0 <= ix < cols and 0 <= iy < rows:
        grid.grid[iy, ix] += L_OCC

  np.clip(grid.grid, -5.0, 5.0, out=grid.grid)


def load_scans():
  f = np.load(FIXTURE)
  scans = [(tuple(f[f"pose_{i}"]), f[f"angles_{i}"], f[f"ranges_{i}"])
           for i in range(int(f["n_scans"]))]
  return scans, float(f["max_range"]), tuple(f["origin"])


def make_grid():
  # room_hub's mission grid: -3..7 m at 5 cm, same as mission/mission.py
  return OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)


def test_vectorized_update_matches_reference_on_recorded_scans():
  """Replaying the fixture through both implementations builds the same map.

  All 12 scans go into ONE grid per implementation, because accumulation and
  the shared +/-5 clip are part of the semantics, not just single scans.
  """
  scans, max_range, origin = load_scans()
  ref, vec = make_grid(), make_grid()
  for pose, angles, ranges in scans:
    reference_update(ref, pose, angles, ranges, max_range, origin=origin)
    vec.update(pose, angles, ranges, max_range, origin=origin)

  assert ref.grid.any()                       # the fixture actually painted evidence
  np.testing.assert_allclose(vec.grid, ref.grid, atol=1e-9)
  assert (vec.to_image() == ref.to_image()).all()


def test_vectorized_update_matches_reference_on_synthetic_edge_cases():
  """Hand-built scans covering what a recorded fixture might not:
  max-range misses, a ray shorter than one cell, rays leaving the grid,
  an off-grid origin ray, and a rotated pose with a sensor offset."""
  cases = [
    ((0.0, 0.0, 0.3), [0.0, 1.0, -2.0], [5.0, 0.02, 8.0]),      # miss at exactly max_range
    ((6.5, 6.5, 2.4), [0.5, 3.0], [4.0, 4.0]),                  # rays exit the grid
    ((-2.9, -2.9, -1.6), [0.0, 0.1, 0.2], [0.04, 2.0, 7.99]),   # near-corner, sub-cell ray
    ((2.0, 3.0, math.pi / 2), list(np.linspace(-3, 3, 40)), list(np.full(40, 1.5))),
  ]
  for pose, angles, ranges in cases:
    ref, vec = make_grid(), make_grid()
    reference_update(ref, pose, np.array(angles), np.array(ranges), 8.0,
                     origin=(0.1, 0.0))
    vec.update(pose, np.array(angles), np.array(ranges), 8.0, origin=(0.1, 0.0))
    np.testing.assert_allclose(vec.grid, ref.grid, atol=1e-9)


def test_vectorized_update_is_5x_faster():
  """The issue-#2 acceptance bar. Relative (both timed here, same machine,
  best-of-5) so machine speed cancels; fails against the old implementation
  by construction, since the reference IS the old implementation.

  The two are timed INTERLEAVED (ref, vec, ref, vec, ...) rather than in two
  blocks, because the suite runs in parallel and its load varies over time: a
  mission test starting during a solid block of vectorized reps penalises
  that half alone and deflates the ratio. Measured 6.8-7.1x on a quiet
  machine and 6.2x under six steady CPU burners, but a blocked measurement
  read 4.9x once inside the real suite -- a false alarm, since the regression
  this guards (the loop coming back) reads ~1x. Interleaving costs nothing
  and puts both halves in the same load window. Note that timing by
  `process_time` instead does NOT help: the contention is real
  memory-bandwidth cost, not scheduler preemption (7.16x vs 7.04x quiet,
  6.34x vs 6.18x loaded), so the clock is not the lever.
  """
  scans, max_range, origin = load_scans()

  def time_one(fn):
    grid = make_grid()
    t0 = time.perf_counter()
    for pose, angles, ranges in scans:
      fn(grid, pose, angles, ranges)
    return time.perf_counter() - t0

  def ref_fn(g, p, a, r):
    reference_update(g, p, a, r, max_range, origin=origin)

  def vec_fn(g, p, a, r):
    g.update(p, a, r, max_range, origin=origin)

  t_ref = t_vec = math.inf
  for _ in range(5):
    t_ref = min(t_ref, time_one(ref_fn))
    t_vec = min(t_vec, time_one(vec_fn))
  speedup = t_ref / t_vec
  print(f"\nper-scan: reference {t_ref / len(scans) * 1e3:.2f} ms, "
        f"vectorized {t_vec / len(scans) * 1e3:.2f} ms  ({speedup:.1f}x)")
  assert speedup >= 5.0


def test_update_accepts_empty_scan():
  """Every bearing self-occluded is a real Lidar output (the arm crosses the
  scan plane); it must be a no-op, not an indexing error."""
  grid = make_grid()
  grid.update((0.0, 0.0, 0.0), np.array([]), np.array([]), 8.0)
  assert not grid.grid.any()
