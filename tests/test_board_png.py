"""scripts/board_png.py: a board's ink as an image the website can hang
(rooftop-media-2026 #128).

Two things can go wrong silently, and each is pinned: the image can come out
MIRRORED (every figure the pen draws is a symmetric diagnostic, so nothing
upstream notices -- the website flips +lat to the viewer's left once, in
`surfaces/board.ts`, and this has to agree with it), and a recording replay
can ignore a `board_cleared` and hang a drawing that was erased.
"""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "board_png.py"


@pytest.fixture(scope="module")
def board_png():
  spec = importlib.util.spec_from_file_location("board_png", SCRIPT)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _ink_columns(img):
  """Mean x of every inked pixel, as a fraction of the width."""
  a = np.asarray(img.convert("L"))
  ys, xs = np.nonzero(a < 128)
  assert len(xs) > 0, "nothing inked"
  return xs.mean() / a.shape[1]


def test_plus_lat_is_the_viewers_left(board_png):
  # One vertical stroke at +lat only. The website paints +lat on the LEFT
  # (the robot's left, facing the board), so the ink must sit in the left
  # half of a whole-face render -- a mirrored image puts it on the right.
  stroke = [[0.04, -0.05], [0.04, 0.05]]
  img = board_png.render([stroke], margin=0.0, whole=True)
  assert _ink_columns(img) < 0.4
  # ...and height grows UP the image: a stroke at +height is near the top.
  img = board_png.render([[[-0.05, 0.05], [0.05, 0.05]]], margin=0.0, whole=True)
  a = np.asarray(img.convert("L"))
  ys, _ = np.nonzero(a < 128)
  assert ys.mean() / a.shape[0] < 0.4


def test_crop_hugs_the_ink_and_whole_keeps_the_face(board_png):
  stroke = [[-0.01, -0.01], [0.01, 0.01]]
  cropped = board_png.render([stroke], margin=0.005, whole=False)
  whole = board_png.render([stroke], margin=0.005, whole=True)
  assert cropped.size == (round(0.03 * board_png.PX_PER_M), round(0.03 * board_png.PX_PER_M))
  assert whole.size == (480, 390)


def test_recording_replay_honours_a_clear_and_a_time(board_png, tmp_path):
  messages = [
    {"type": "board_snapshot", "t": 0.0, "board": "whiteboard_a",
     "strokes": [{"program": "old", "stroke": 0, "points": [[0, 0], [0.01, 0.01]]}]},
    {"t": 1.0, "robot": "pluggybot"},                           # a frame: ignored
    {"type": "draw", "t": 5.0, "board": "whiteboard_b", "points": [[0, 0], [0.02, 0]]},
    {"type": "board_cleared", "t": 10.0, "board": "whiteboard_a"},
    {"type": "draw", "t": 20.0, "board": "whiteboard_a", "points": [[0, 0], [0.03, 0.03]]},
    {"type": "draw", "t": 30.0, "board": "whiteboard_a", "points": [[0, 0], [-0.03, 0.03]]},
  ]
  path = tmp_path / "rec.jsonl.gz"
  with gzip.open(path, "wt") as fh:
    for m in messages:
      fh.write(json.dumps(m) + "\n")

  # At the end: the two strokes after the clear, and nothing from before it
  # (a picture of an erased drawing is a picture of something that is not
  # on the wall), and nothing from the other board.
  assert board_png.strokes_from_recording(path, "whiteboard_a", None) == [
    [[0, 0], [0.03, 0.03]], [[0, 0], [-0.03, 0.03]]]
  # At t=25: one stroke.  At t=12: cleared, so nothing to draw.
  assert board_png.strokes_from_recording(path, "whiteboard_a", 25.0) == [[[0, 0], [0.03, 0.03]]]
  assert board_png.strokes_from_recording(path, "whiteboard_a", 12.0) == []
  # Before the clear: the snapshot's stroke.
  assert board_png.strokes_from_recording(path, "whiteboard_a", 2.0) == [[[0, 0], [0.01, 0.01]]]
  with pytest.raises(SystemExit):
    board_png.render([], margin=0.0, whole=False)


def test_boards_state_file_reads_the_lines(board_png, tmp_path):
  state = {"version": 1, "boards": {"whiteboard_a": {
    "lines": [{"program": "house", "points": [[0, 0], [0.01, 0]]}]}}}
  path = tmp_path / "boards.json"
  path.write_text(json.dumps(state))
  assert board_png.strokes_from_boards(path, "whiteboard_a") == [[[0, 0], [0.01, 0]]]
  with pytest.raises(SystemExit):
    board_png.strokes_from_boards(path, "whiteboard_b")


def test_the_committed_home_recording_yields_the_house(board_png):
  rec = Path(__file__).resolve().parents[1] / "protocol" / "telemetry.home_lifecycle.jsonl.gz"
  strokes = board_png.strokes_from_recording(rec, "whiteboard_a", None)
  assert len(strokes) >= 5
  img = board_png.render(strokes, margin=0.012, whole=False)
  assert min(img.size) > 100
