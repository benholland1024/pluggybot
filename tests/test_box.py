"""What a depth cloud says of a box (issue #466, stage 2; `perception/box.py`):
its top, an edge seen from in front, an edge whose face looks at the camera,
which way it faces, and a face toward the robot. The clouds here are made up
-- the rules are about points, and a frame the D435 renders is
`tests/test_probe.py`'s. Each test draws from a generator of its own, so
what it draws is the same whichever tests ran before it (#479's review: one
shared generator failed the noisiest edge in 139 of 300 orders).
"""

import math

import numpy as np
import pytest

from pluggybot.perception import box

def rows(rng, lo: float, hi: float, spacing: float, noise: float,
         per_row: int = 40) -> np.ndarray:
  """A face's points along a direction as a stereo camera lays them at a
  grazing angle: rows `spacing` apart from `lo` to `hi`, each row `per_row`
  points, blurred by `noise` -- the rows' phase against the edge at random."""
  at = np.arange(lo + rng.uniform(0.0, spacing), hi, spacing)
  x = np.repeat(at, per_row)
  return x + rng.normal(0.0, noise, len(x))


# ---- the top ---------------------------------------------------------------------

def test_the_top_is_the_highest_level_a_large_share_of_points_stand_at():
  rng = np.random.default_rng(466)
  floor = np.column_stack([rng.uniform(0, 1, (4000, 2)), rng.normal(0, 0.001, 4000)])
  face = np.column_stack([rng.uniform(0, 1, (6000, 2)), rng.uniform(0.0, 0.14, 6000)])
  top = np.column_stack([rng.uniform(0, 1, (3000, 2)), 0.152 + rng.normal(0, 0.001, 3000)])
  knob = np.column_stack([rng.uniform(0, 1, (300, 2)), rng.uniform(0.075, 0.101, 300)])
  stray = np.column_stack([rng.uniform(0, 1, (20, 2)), rng.uniform(0.2, 0.3, 20)])
  pts = np.concatenate([floor, face, top, knob, stray])
  assert box.top_height(pts) == pytest.approx(0.152, abs=0.0005)
  assert box.on_top(pts, 0.152).sum() >= 2900
  assert box.top_height(floor) is None, "the floor is no top"


# ---- the edges -------------------------------------------------------------------

#: Spacing and noise, and the bars: the mean of 32 reads and the worst, m.
#: Over 500 draws of each the mean was within 0.31, 0.13 and 0.70 mm, the
#: worst read 1.65, 0.80 and 2.46; the rough percentile's mean was 1.96,
#: 2.13 and 1.37 mm off at its nearest.
EDGE_CASES = [(0.0046, 0.0015, 0.0006, 0.0022), (0.002, 0.0005, 0.0005, 0.0012),
              (0.006, 0.003, 0.001, 0.0032)]

@pytest.mark.parametrize("spacing, noise, bias, worst", EDGE_CASES)
def test_an_edge_seen_from_in_front_is_read_off_the_mean_of_a_window(spacing, noise, bias,
                                                                     worst):
  # A top's far edge from the stance: rows 4-5 mm apart, 1.4 mm of noise.
  # Each slice across lays its last row somewhere inside the edge; the
  # window's mean reads the edge whatever the rows' phase -- no bias, and
  # every read within `worst` (`EDGE_CASES`).
  rng = np.random.default_rng(466)
  ends = np.array([box.edge(np.concatenate([rows(rng, 0.0, 0.22, spacing, noise)
                                            for _ in range(12)])) for _ in range(32)]) - 0.22
  assert abs(ends.mean()) < bias and np.abs(ends).max() < worst
  assert box.edge(np.arange(10) * 0.01) is None, "too few points"


def test_a_face_at_the_edge_pulls_a_windows_mean_and_not_a_low_percentile():
  # A bracket's tip from the measuring stance: its top's points even along
  # it, and its front face's points stood AT the tip -- a quarter of the
  # window's. The window read the tip 4 mm past it; a low percentile reads
  # it where it is (0.4 mm of noise there).
  rng = np.random.default_rng(466)
  tip = 0.0
  top = rows(rng, tip, tip + 0.056, 0.0012, 0.0004, per_row=12)
  near = top[top < tip + box.EDGE_WINDOW_M]
  face = tip + rng.normal(0.0, 0.0004, len(near) // 3)
  along = np.concatenate([top, face])
  assert box.edge(along, far=False) < tip - 0.003, "the premise: the window is pulled"
  assert box.faced_edge(along, far=False) == pytest.approx(tip, abs=0.001)
  assert box.faced_edge(-along, far=True) == pytest.approx(-tip, abs=0.001)


# ---- which way it faces, and a face -------------------------------------------------

#: How far off the facing may read: over 60 draws of the four lids, at most
#: 0.49 deg off squared, and at least 1.03 off its worst unsquared.
FACING_RAD = math.radians(0.75)

def lid(rng, yaw: float, n: int = 90000) -> np.ndarray:
  """A lid's top, 0.22 deep and 0.30 across, as many points as three frames
  lay on it from the measuring stance, three times denser toward the robot
  than at its back, turned `yaw`: (x, y)."""
  u = np.concatenate([rng.uniform(0, 0.22, 2 * n // 3), rng.uniform(0, 0.08, n // 3)])
  v = rng.uniform(-0.15, 0.15, len(u))
  c, s = math.cos(yaw), math.sin(yaw)
  return np.column_stack([c * u - s * v, s * u + c * v]) + rng.normal(0, 0.001, (len(u), 2))


def test_a_box_faces_square_to_its_tops_far_edge(monkeypatch):
  # Roughly the least rectangle, then square to the far edge laid slice by
  # slice, from a hint 20 deg off either way. Trimmed, the least rectangle's
  # area is flat within a degree or two of its least, and alone it read
  # these lids up to 2.5 deg off.
  rng = np.random.default_rng(466)
  lids = {yaw: lid(rng, yaw) for yaw in (0.0, 0.33, -0.4, 1.0)}
  for yaw, xy in lids.items():
    for hint in (yaw + 0.35, yaw - 0.35):
      assert box.facing(xy, hint) == pytest.approx(yaw, abs=FACING_RAD)
  assert box.facing(lids[0.0][:10], 0.0) is None
  monkeypatch.setattr(box, "SQUARE_PASSES", 0)
  rough = [abs(box.facing(xy, yaw + h) - yaw) for yaw, xy in lids.items() for h in (0.35, -0.35)]
  assert max(rough) > math.radians(0.9), "the premise: the least rectangle alone drifts"


def test_a_face_is_where_it_stands_its_middle_and_its_turn():
  # A front face's points, the knob's 7 cm across taken out of its middle,
  # one side seen twice as densely as the other: its middle is between its
  # edges, never the points' mean.
  rng = np.random.default_rng(466)
  across = np.concatenate([rng.uniform(-0.15, -0.035, 4000), rng.uniform(0.035, 0.15, 2000)])
  slope = math.tan(math.radians(1.5))
  along = 0.07 + slope * across + rng.normal(0, 0.0008, len(across))
  place, middle, got = box.face(along, across + 0.01)
  assert middle == pytest.approx(0.01, abs=0.002)
  assert place == pytest.approx(0.07, abs=0.0005)
  assert got == pytest.approx(slope, abs=0.002)
  assert box.face(along[:10], across[:10]) is None
