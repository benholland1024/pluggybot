"""What a depth cloud says of a box (issue #466, stage 2; `perception/box.py`):
its top, an edge seen from in front, an edge whose face looks at the camera,
which way it faces, and a face toward the robot. The clouds here are made up
-- the rules are about points, and a frame the D435 renders is the
opener's to fly (`evaluation/opener.py`). Each test draws from a generator of its own, so
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


# ---- an end seen past (#636) -------------------------------------------------------

W, H, F = 40, 40, 40.0
EYE = np.array([0.0, 0.0, 1.0])


def _point(row: int, col: int, height: float) -> list[float]:
  """Where pixel (row, col) of a camera looking straight down from `EYE`
  meets a level `height`, the world frame."""
  depth = EYE[2] - height
  return [(col - W / 2 + 0.5) / F * depth, -(row - H / 2 + 0.5) / F * depth, height]


def picture(heights: np.ndarray, near: float = 0.17, reach: float = 3.0):
  """A camera straight down over `heights` (H x W, NaN where it reads
  nothing), its rays as `DepthCamera` lays them: each row of `z` its pixel's
  depth, its points in reading order."""
  from pluggybot.perception.depth import DepthPicture
  z = EYE[2] - heights
  pts = np.array([_point(r, c, heights[r, c]) for r in range(H) for c in range(W)
                  if not np.isnan(heights[r, c])]).reshape(-1, 3)
  return DepthPicture(z=z, points=pts, eye=EYE.copy(), turn=np.eye(3), focal=F, near=near,
                      reach=reach, noise_k=0.0)


def _end(heights: np.ndarray, end_col: int, way=(1.0, 0.0, 0.0), top: float = 0.5,
         start_col: int | None = None, **kw) -> dict:
  """`past_end` of the top's points in column `start_col` (by default the
  end's own), walked along `way`, the end read at column `end_col`."""
  look = box.Look.of([picture(heights, **kw)])
  col = end_col if start_col is None else start_col
  picks = np.flatnonzero((look.cols == col) & (look.points[:, 2] == top))
  ended = np.flatnonzero((look.cols == end_col) & (look.points[:, 2] == top))
  return box.past_end(look, picks, np.asarray(way), top, float(np.max(look.points[ended] @ way)))


def test_a_look_keeps_each_points_pixel_and_a_picture_projects_points_back_to_them():
  h = np.full((H, W), 0.5)
  h[3, 4] = np.nan
  look = box.Look.of([picture(h), picture(h)])
  assert len(look.points) == 2 * (W * H - 1) and set(look.frame) == {0, 1}
  rows, cols = look.pictures[0].pixels()
  assert np.array_equal(look.rows[look.frame == 1], rows)
  assert (3, 4) not in set(zip(rows.tolist(), cols.tolist()))
  back = look.pictures[0].project(look.points[look.frame == 0])
  assert back == pytest.approx(np.column_stack([rows, cols]), abs=1e-9)


def test_past_the_end_of_a_top_a_lower_pixel_says_it_stopped_there():
  h = np.full((H, W), 0.0)
  h[:, :21] = 0.5
  saw = _end(h, 20)
  assert saw["lower"] == 1.0 and saw["cut"] == 0.0 and not saw["reach"]


def test_a_walk_goes_on_over_the_top_and_a_gap_in_its_points():
  # the walk starts on the top, 5 cm of its own points to the edge, one of
  # them dropped: lower at the end all the same
  h = np.full((H, W), 0.0)
  h[:, :21] = 0.5
  h[20, 19] = np.nan
  saw = _end(h, 20, start_col=16)
  assert saw["lower"] == 1.0 and saw["runs"] == 0.0


def test_a_top_running_on_past_where_its_end_reads_is_no_end():
  # what cut the end's points stopped short of it (the window's rim): just
  # past them is more of the top, never anything lower
  h = np.full((H, W), 0.0)
  h[:, :33] = 0.5
  saw = _end(h, 20)
  assert saw["runs"] == 1.0 and saw["lower"] == 0.0
  # ...and a pixel or two of the top past it is an end's own blur, not more of it
  h = np.full((H, W), 0.0)
  h[:, :22] = 0.5
  assert _end(h, 20)["lower"] == 1.0
  # ...and a lone pixel does not tell a walk: one low column before a taller
  # thing, and one high column before a drop, each column of pixels alone
  h = np.full((H, W), 0.5)
  h[:, 21:] = 0.9
  h[:, 18] = 0.2
  assert _end(h, 20, start_col=15)["taller"] == 1.0, "a lone low pixel told the walk"
  h = np.full((H, W), 0.0)
  h[:, :21] = 0.5
  h[:, 18] = 0.9
  assert _end(h, 20, start_col=15)["lower"] == 1.0, "a lone high pixel told the walk"
  # ...a surface is seen on more than one pixel: two low of three, and the walk is told
  h = np.full((H, W), 0.5)
  h[:, 21] = 0.2
  h[:, 23:] = 0.2
  assert _end(h, 20)["lower"] == 1.0


def test_a_top_the_picture_cut_has_no_end_seen():
  h = np.full((H, W), 0.5)
  saw = _end(h, W - 2)
  assert saw["cut"] == 1.0 and saw["lower"] == 0.0 and saw["runs"] == 0.0


def test_a_dropout_or_two_at_the_border_is_the_picture_cutting_a_top():
  h = np.full((H, W), 0.5)
  h[:, W - 2:] = np.nan
  assert _end(h, W - 5, start_col=W - 8)["cut"] == 1.0, "two columns unread is not the top stopping"
  h = np.full((H, W), 0.5)
  h[:, W - box.PAST_CUT_PX:] = np.nan
  assert _end(h, W - 1 - box.PAST_CUT_PX)["lower"] == 1.0, (
    f"{box.PAST_CUT_PX} columns unread: the top stopped")


def test_the_shadow_is_looked_past_to_what_stands_behind_it():
  # the band of nothing beside a near thing, then the floor (or the taller
  # thing that cast it): told by what stands behind it
  h = np.full((H, W), 0.0)
  h[:, :21] = 0.5
  h[:, 21:29] = np.nan
  assert _end(h, 20)["lower"] == 1.0
  h[:, 29:] = 0.8
  assert _end(h, 20)["taller"] == 1.0, "the taller thing behind its shadow"


def test_nothing_read_all_the_way_past_an_end_is_the_top_stopped():
  # a table's far edge, the floor past the camera's reach: walked to the
  # border or to the walk's end with no return, the top stopped
  h = np.full((H, W), 0.5)
  h[:, 21:] = np.nan
  assert _end(h, 20)["lower"] == 1.0, "unread to the border"
  # ...and unread over the walk's whole length, the border further than it goes
  big = np.full((H, W), 0.5)
  big[:, 21:] = np.nan
  try:
    box_steps = box.PAST_STEPS
    box.PAST_STEPS = 10
    assert _end(big, 20)["lower"] == 1.0
  finally:
    box.PAST_STEPS = box_steps


def test_something_taller_past_the_end_is_its_free_surfaces_end():
  h = np.full((H, W), 0.5)
  h[:, 21:] = 0.9
  saw = _end(h, 20)
  assert saw["taller"] == 1.0 and saw["cut"] == 0.0


def test_an_end_at_the_cameras_nearest_or_farthest_reading_is_the_ranges():
  # past either, a pixel reads nothing whatever stands there, so a walk over
  # them would say the top stopped
  h = np.full((H, W), 0.5)
  h[:, 21:] = np.nan
  assert _end(h, 20, reach=0.55)["reach"]
  assert not _end(h, 20, reach=0.7)["reach"]
  assert _end(h, 20, near=0.49)["near"]
  assert not _end(h, 20, near=0.4)["near"]


def test_a_diagonal_walk_reads_no_pixel_twice():
  # a step of the walk's unit length revisits a pixel across a diagonal, and
  # then one noisy pixel is two agreeing: a low one stood before a taller
  # thing, and the walk read `lower`
  h = np.full((H, W), 0.5)
  for r in range(H):
    for c in range(W):
      if (r - 10) + (c - 10) >= 2 * 9:
        h[r, c] = 0.9
  h[14, 14] = 0.2          # on the walk's way, 4 pixels along it
  look = box.Look.of([picture(h)])
  picks = np.flatnonzero((look.rows == 10) & (look.cols == 10))
  way = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
  saw = box.past_end(look, picks, way, 0.5, 9.0)
  assert saw["taller"] == 1.0 and saw["lower"] == 0.0


def test_a_walk_goes_where_the_end_points_in_the_picture():
  # `way` is the end's way in the world, whichever way that runs in the
  # picture: +x is along the rows, +y up them (the camera's x and y)
  h = np.full((H, W), 0.5)
  h[:, 21:] = 0.0
  assert _end(h, 20)["lower"] == 1.0, "+x runs toward the higher columns"
  h = np.full((H, W), 0.5)
  h[:20, :] = 0.0
  look = box.Look.of([picture(h)])
  picks = np.flatnonzero((look.rows == 20) & (look.points[:, 2] == 0.5))
  up = np.array([0.0, 1.0, 0.0])
  assert box.past_end(look, picks, up, 0.5, float(np.max(look.points[picks] @ up)))["lower"] == 1.0
  picks = np.flatnonzero((look.rows == H - 1) & (look.points[:, 2] == 0.5))
  assert box.past_end(look, picks, -up, 0.5, float(np.max(look.points[picks] @ -up)))["cut"] == 1.0


def test_a_bodys_pictures_are_its_cloud_each_point_projecting_back_to_its_pixel():
  # `Body.depth_pictures` is `depth_cloud` with each frame's picture beside
  # its points -- the same points, the same order, drawing the same noise
  # (the opener and `measure` before it stand on those) -- and the picture
  # carries the very camera that laid them: a point projects back onto its
  # own pixel, the waist's reading in the composition too, or a walk goes
  # the wrong way (`past_end`)
  import mujoco
  from pluggybot.body import body_for
  from pluggybot.humanoid import model as hm

  def floor(waist: float = 0.0):
    model = mujoco.MjModel.from_xml_string(hm.body_xml())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[model.jnt_qposadr[model.joint("waist_yaw").id]] = waist
    mujoco.mj_forward(model, data)
    return model, data

  a, b = body_for(*floor(), noise="p"), body_for(*floor(), noise="p")
  cloud = a.run(a.depth_cloud(64, 36, 0.2, frames=2, apart=0.02))
  pictures = b.run(b.depth_pictures(64, 36, 0.2, frames=2, apart=0.02))
  assert np.array_equal(np.concatenate([p.points for p in pictures]), cloud)
  turned = body_for(*floor(0.5), noise="q")
  for pic in (pictures[1], turned.run(turned.depth_pictures(64, 36, 0.2))[0]):
    rows, cols = pic.pixels()
    assert len(rows) and pic.project(pic.points) == pytest.approx(
      np.column_stack([rows, cols]), abs=1e-6)
    # ...and each pixel's rise is what a metre of its depth does to its height
    assert pic.rises(rows, cols) == pytest.approx(
      (pic.points[:, 2] - pic.eye[2]) / pic.z[rows, cols], abs=1e-6)


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


def test_an_edge_says_which_points_it_was_read_off():
  # #630: `measure` asks whether those points stand at its window's rim, so
  # they are the window inside the rough edge -- at either end -- and the
  # edge is `edge`'s own
  rng = np.random.default_rng(630)
  along = rng.uniform(0.0, 1.0, 5000)
  for far in (True, False):
    end, strip = box.edge_read(along, far=far)
    assert end == box.edge(along, far=far)
    lo, hi = along[strip].min(), along[strip].max()
    assert (hi > 0.99 and lo > 0.965) if far else (lo < 0.01 and hi < 0.035)
    assert strip.sum() == pytest.approx(0.03 * len(along), rel=0.25)


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
