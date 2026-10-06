"""What a depth cloud says of a box (issue #466, stage 2; `perception/box.py`):
its top, an edge seen from in front, an edge whose face looks at the camera,
which way it faces, and a face toward the robot. The clouds here are made up
-- the rules are about points, and a frame the D435 renders is
`tests/test_probe.py`'s.
"""

import math

import numpy as np
import pytest

from pluggybot.perception import box

RNG = np.random.default_rng(466)


def rows(lo: float, hi: float, spacing: float, noise: float, per_row: int = 40,
         phase: float | None = None) -> np.ndarray:
  """A face's points along a direction as a stereo camera lays them at a
  grazing angle: rows `spacing` apart from `lo` to `hi`, each row `per_row`
  points, blurred by `noise` -- the rows' phase against the edge at random."""
  first = lo + (RNG.uniform(0.0, spacing) if phase is None else phase)
  at = np.arange(first, hi, spacing)
  x = np.repeat(at, per_row)
  return x + RNG.normal(0.0, noise, len(x))


# ---- the top ---------------------------------------------------------------------

def test_the_top_is_the_highest_level_a_large_share_of_points_stand_at():
  floor = np.column_stack([RNG.uniform(0, 1, (4000, 2)), RNG.normal(0, 0.001, 4000)])
  face = np.column_stack([RNG.uniform(0, 1, (6000, 2)), RNG.uniform(0.0, 0.14, 6000)])
  top = np.column_stack([RNG.uniform(0, 1, (3000, 2)), 0.152 + RNG.normal(0, 0.001, 3000)])
  knob = np.column_stack([RNG.uniform(0, 1, (300, 2)), RNG.uniform(0.075, 0.101, 300)])
  stray = np.column_stack([RNG.uniform(0, 1, (20, 2)), RNG.uniform(0.2, 0.3, 20)])
  pts = np.concatenate([floor, face, top, knob, stray])
  assert box.top_height(pts) == pytest.approx(0.152, abs=0.0005)
  assert box.on_top(pts, 0.152).sum() >= 2900
  assert box.top_height(floor) is None, "the floor is no top"


# ---- the edges -------------------------------------------------------------------

@pytest.mark.parametrize("spacing, noise", [(0.0046, 0.0015), (0.002, 0.0005), (0.006, 0.003)])
def test_an_edge_seen_from_in_front_is_read_off_the_mean_of_a_window(spacing, noise):
  # A top's far edge from the stance: rows 4-5 mm apart, 1.4 mm of noise.
  # Each slice across lays its last row somewhere inside the edge; the
  # window's mean reads the edge whatever the rows' phase.
  ends = [box.edge(np.concatenate([rows(0.0, 0.22, spacing, noise) for _ in range(12)]))
          for _ in range(8)]
  assert np.abs(np.array(ends) - 0.22).max() < 0.0012
  assert box.edge(np.arange(10) * 0.01) is None, "too few points"


def test_a_face_at_the_edge_pulls_a_windows_mean_and_not_a_low_percentile():
  # A bracket's tip from the measuring stance: its top's points even along
  # it, and its front face's points stood AT the tip -- a quarter of the
  # window's. The window read the tip 4 mm past it; a low percentile reads
  # it where it is (0.4 mm of noise there).
  tip = 0.0
  top = rows(tip, tip + 0.056, 0.0012, 0.0004, per_row=12)
  near = top[top < tip + box.EDGE_WINDOW_M]
  face = tip + RNG.normal(0.0, 0.0004, len(near) // 3)
  along = np.concatenate([top, face])
  assert box.edge(along, far=False) < tip - 0.003, "the premise: the window is pulled"
  assert box.faced_edge(along, far=False) == pytest.approx(tip, abs=0.001)
  assert box.faced_edge(-along, far=True) == pytest.approx(-tip, abs=0.001)


# ---- which way it faces, and a face -------------------------------------------------

def lid(yaw: float, n: int = 90000) -> np.ndarray:
  """A lid's top, 0.22 deep and 0.30 across, as many points as three frames
  lay on it from the measuring stance, three times denser toward the robot
  than at its back, turned `yaw`: (x, y)."""
  u = np.concatenate([RNG.uniform(0, 0.22, 2 * n // 3), RNG.uniform(0, 0.08, n // 3)])
  v = RNG.uniform(-0.15, 0.15, len(u))
  c, s = math.cos(yaw), math.sin(yaw)
  return np.column_stack([c * u - s * v, s * u + c * v]) + RNG.normal(0, 0.001, (len(u), 2))


def test_a_box_faces_square_to_its_tops_far_edge(monkeypatch):
  # Roughly the least rectangle, then square to the far edge laid slice by
  # slice, from a hint 20 deg off either way. Trimmed, the least rectangle's
  # area is flat within a degree or two of its least, and alone it read
  # these lids up to 2.5 deg off.
  lids = {yaw: lid(yaw) for yaw in (0.0, 0.33, -0.4, 1.0)}
  for yaw, xy in lids.items():
    for hint in (yaw + 0.35, yaw - 0.35):
      assert box.facing(xy, hint) == pytest.approx(yaw, abs=math.radians(0.3))
  assert box.facing(lids[0.0][:10], 0.0) is None
  monkeypatch.setattr(box, "SQUARE_PASSES", 0)
  rough = [abs(box.facing(xy, yaw + h) - yaw) for yaw, xy in lids.items() for h in (0.35, -0.35)]
  assert max(rough) > math.radians(1.0), "the premise: the least rectangle alone drifts"


def test_a_face_is_where_it_stands_its_middle_and_its_turn():
  # A front face's points, the knob's 7 cm across taken out of its middle,
  # one side seen twice as densely as the other: its middle is between its
  # edges, never the points' mean.
  across = np.concatenate([RNG.uniform(-0.15, -0.035, 4000), RNG.uniform(0.035, 0.15, 2000)])
  slope = math.tan(math.radians(1.5))
  along = 0.07 + slope * across + RNG.normal(0, 0.0008, len(across))
  place, middle, got = box.face(along, across + 0.01)
  assert middle == pytest.approx(0.01, abs=0.002)
  assert place == pytest.approx(0.07, abs=0.0005)
  assert got == pytest.approx(slope, abs=0.002)
  assert box.face(along[:10], across[:10]) is None
