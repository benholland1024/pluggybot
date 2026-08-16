"""Guards for whiteboards-as-state and the generalized errand (issue #12)."""

import json
import math

import mujoco
import pytest

from pluggybot.hub.boards import CELL, BoardBook, BoardRecord, decimate
from pluggybot.hub.drawing import Board, Envelope, board_standoff
from pluggybot.hub.errand import carry_errand, drawing_errand
from pluggybot.hub.lifecycle import board_book, errands_for
from pluggybot.hub import strokes

META = json.load(open("models/home_world.meta.json"))
BOARD_A = Board.from_meta(META["boards"]["whiteboard_a"])


def book(path=None) -> BoardBook:
  return BoardBook.for_meta(META, path=path, clock=lambda: "2026-08-16T00:00:00")


def square(size=0.05, n=40):
  """A closed square as a traced polyline, in board coordinates."""
  h = size / 2
  corners = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]
  pts = []
  for a, b in zip(corners, corners[1:]):
    pts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            for k in range(n)]
  return pts + [corners[-1]]


# ---- decimation -------------------------------------------------------------


def test_decimate_keeps_the_endpoints_and_the_corners():
  """A stroke is decimated by DISTANCE, not by index.

  The pen slows into a corner, so every-nth sampling drops the straights and
  the corners at the same rate and rounds the figure off. And the last sample
  must survive whatever else does: a stroke that loses its endpoint stops
  meeting the stroke it was drawn to meet, which on a letter is visible.
  """
  raw = square()
  thin = decimate(raw, eps=0.005)
  assert thin[0] == raw[0] and thin[-1] == raw[-1]
  assert len(thin) < len(raw) / 2, "nothing was actually thinned"
  # every original corner is still within a decimation step of a kept point
  for corner in [(-0.025, -0.025), (0.025, -0.025), (0.025, 0.025)]:
    assert min(math.dist(corner, p) for p in thin) < 0.005


def test_decimate_caps_a_pathological_stroke():
  """One websocket message must stay bounded. A figure with thousands of
  genuinely-distinct points degrades to a coarse polyline rather than
  stalling the sender thread."""
  raw = [(0.001 * k, 0.0) for k in range(5000)]
  thin = decimate(raw, eps=0.0)
  assert len(thin) <= 401 and thin[-1] == raw[-1]


def test_decimate_survives_a_stroke_with_no_ink():
  """A press that never found the board leaves nothing to draw."""
  assert decimate([]) == []
  assert decimate([(0.0, 0.0)]) == [(0.0, 0.0)]


# ---- board state ------------------------------------------------------------


def test_a_stroke_fills_the_board_and_names_its_program():
  b = book()
  msg = b.stroke("whiteboard_a", "square", square(), t=12.5)
  rec = b["whiteboard_a"]
  assert msg["type"] == "draw" and msg["board"] == "whiteboard_a"
  assert msg["stroke"] == 0 and msg["program"] == "square"
  assert rec.programs == ["square"] and rec.strokes == 1
  assert rec.ink_m == pytest.approx(0.2, abs=0.01)   # 4 x 50 mm
  assert 0.0 < rec.fill < 1.0
  assert rec.drawn_at is not None


def test_fill_is_measured_against_the_pens_reach_not_the_slab():
  """The home boards are 320 x 260 mm; the carriage has 110 mm of travel and
  the base is parked for the whole drawing. Scoring fill against the slab
  would report a board the robot has no way to fill as 12 % after its best
  possible effort, and "is this board full" would never be true."""
  rec = book()["whiteboard_a"]
  env = Envelope.for_board(BOARD_A)
  assert rec.reach == env.size
  slab = (BOARD_A.half[1] * 2, BOARD_A.half[2] * 2)
  assert rec.reach[0] < slab[0], "reach must be the intersection, not the slab"


def test_redrawing_the_same_place_does_not_inflate_the_fill():
  """Coverage is a SET of cells. Two passes over one line is one line's worth
  of board used up, which is what a person looking at the board would say."""
  b = book()
  b.stroke("whiteboard_a", "square", square())
  once = b["whiteboard_a"].fill
  b.stroke("whiteboard_a", "square", square())
  assert b["whiteboard_a"].fill == once
  assert b["whiteboard_a"].strokes == 2, "the stroke itself still counted"


def test_a_long_segment_marks_every_cell_it_crosses():
  """Segments are WALKED at half a cell, not sampled at their endpoints.

  A 100 mm line decimated to its two endpoints would otherwise mark 2 cells
  out of the 10 it crosses, and a board's fill would depend on how its
  polylines happened to be thinned rather than on how much board is inked.
  """
  b = book()
  b.stroke("whiteboard_a", "line", [(-0.05, 0.0), (0.05, 0.0)])
  assert len(b["whiteboard_a"].cells) >= int(0.10 / CELL)


def test_clearing_is_an_action_that_reports_itself():
  b = book()
  b.stroke("whiteboard_a", "square", square())
  seen: list = []
  b.on_event.append(seen.append)
  b.clear("whiteboard_a", t=99.0)
  rec = b["whiteboard_a"]
  assert rec.blank and rec.fill == 0.0 and rec.programs == [] and rec.ink_m == 0
  assert rec.clears == 1 and rec.cleared_at is not None
  assert seen == [{"type": "board_cleared", "t": 99.0, "robot": "pluggybot",
                   "board": "whiteboard_a",
                   "clearedAt": "2026-08-16T00:00:00"}]


def test_an_empty_stroke_is_not_a_drawing():
  """A press that never found the board leaves no ink. Recording it would
  claim a stroke the viewer cannot see and add a program name to the board."""
  b = book()
  assert b.stroke("whiteboard_a", "square", [(0.0, 0.0)]) is None
  assert b["whiteboard_a"].blank and b["whiteboard_a"].programs == []


def test_draw_events_carry_board_local_metres():
  """The polyline is what the browser paints into the canvas, so it must be in
  the board's own frame -- and rounded like poses, to 0.1 mm."""
  b = book()
  msg = b.stroke("whiteboard_a", "line", [(-0.0123456, 0.02), (0.03, 0.02)])
  assert msg["points"][0] == [-0.0123, 0.02]
  assert all(abs(y) <= BOARD_A.half[1] and abs(z) <= BOARD_A.half[2]
             for y, z in msg["points"])


# ---- persistence ------------------------------------------------------------


def test_board_state_survives_a_restart(tmp_path):
  """The acceptance criterion, and the reason the state file exists: a board
  is world state, not run state."""
  path = tmp_path / "boards.json"
  first = book(path)
  first.stroke("whiteboard_a", "house", square())
  first.clear("whiteboard_b")

  second = book(path)                        # a fresh process would do this
  a, b2 = second["whiteboard_a"], second["whiteboard_b"]
  assert a.programs == ["house"] and a.strokes == 1
  assert a.fill == first["whiteboard_a"].fill
  assert a.cells == first["whiteboard_a"].cells
  assert b2.clears == 1 and b2.cleared_at == "2026-08-16T00:00:00"


def test_state_is_written_on_every_event_not_at_shutdown(tmp_path):
  """A board that survives only a CLEAN shutdown does not survive the thing
  restarts are usually about."""
  path = tmp_path / "boards.json"
  b = book(path)
  b.stroke("whiteboard_a", "house", square())
  assert path.exists(), "nothing was written until close()"
  assert json.loads(path.read_text())["boards"]["whiteboard_a"]["strokes"] == 1


def test_no_state_file_means_boards_are_per_run(tmp_path):
  b = book()
  b.stroke("whiteboard_a", "house", square())
  assert b.save() is None
  assert not list(tmp_path.iterdir())


def test_a_state_file_cannot_resurrect_a_removed_board(tmp_path):
  """The world decides which boards exist; the file only says what is on
  them. A layout change that takes a whiteboard off the wall must not have it
  reappear in telemetry with last month's drawing on it."""
  path = tmp_path / "boards.json"
  path.write_text(json.dumps({"version": 1, "boards": {
    "whiteboard_ghost": {"reach": [0.1, 0.2], "strokes": 3, "cells": [[0, 0]]},
  }}))
  b = book(path)
  assert "whiteboard_ghost" not in b.names


def test_a_saved_reach_never_overrides_the_worlds(tmp_path):
  """`reach` is a property of the pen and the board, not of the drawing. A
  file written before a board was resized must not keep scoring against the
  old denominator."""
  path = tmp_path / "boards.json"
  path.write_text(json.dumps({"version": 1, "boards": {
    "whiteboard_a": {"reach": [9.0, 9.0], "strokes": 1, "cells": [[0, 0]]},
  }}))
  assert book(path)["whiteboard_a"].reach == Envelope.for_board(BOARD_A).size


def test_an_unreadable_state_version_is_loud(tmp_path):
  path = tmp_path / "boards.json"
  path.write_text(json.dumps({"version": 99, "boards": {}}))
  with pytest.raises(ValueError, match="board state version"):
    book(path)


def test_an_older_state_file_still_loads(tmp_path):
  """Newer refuses, OLDER loads -- and the deploy is what pays for it.

  `/var/lib/pluggybot` outlives the image, so an upgrade that adds a field
  must not crash the sim on start over a file whose old shape is a strict
  subset of the new one. A version-1 file has statistics and no polylines,
  which is exactly what it recorded.
  """
  path = tmp_path / "boards.json"
  path.write_text(json.dumps({"version": 1, "boards": {
    "whiteboard_a": {"reach": [0.11, 0.2], "strokes": 3, "programs": ["house"],
                     "cells": [[0, 0], [1, 1]]},
  }}))
  b = book(path)
  assert b["whiteboard_a"].strokes == 3
  assert b["whiteboard_a"].lines == []
  assert b.snapshots() == [], "a board with no remembered lines has no snapshot"


# ---- catching a late viewer up (protocol 0.5.0) -----------------------------


def test_the_polylines_themselves_survive_a_restart(tmp_path):
  """Board STATISTICS surviving a restart is not enough, and this is the gap
  that proves it: a viewer arriving after the producer restarted would see a
  board reporting 19 % fill with nothing painted on it. Ink is not a body --
  no keyframe re-ships it and no `draw` event will describe it again -- so
  the lines have to live in the file too."""
  path = tmp_path / "boards.json"
  first = book(path)
  first.stroke("whiteboard_a", "house", square())

  second = book(path)
  lines = second["whiteboard_a"].lines
  assert len(lines) == 1 and lines[0]["program"] == "house"
  assert lines[0]["points"] == first["whiteboard_a"].lines[0]["points"]


def test_a_snapshot_carries_every_stroke_on_the_board():
  b = book()
  b.stroke("whiteboard_a", "house", square())
  b.stroke("whiteboard_a", "house", square(size=0.03))
  b.stroke("whiteboard_b", "tree", square(size=0.04))
  snaps = {s["board"]: s for s in b.snapshots(t=12.5)}
  assert set(snaps) == {"whiteboard_a", "whiteboard_b"}
  assert snaps["whiteboard_a"]["type"] == "board_snapshot"
  assert snaps["whiteboard_a"]["t"] == 12.5
  assert [s["stroke"] for s in snaps["whiteboard_a"]["strokes"]] == [0, 1]
  # ...and the points are the SAME polyline the `draw` event carried, or the
  # catch-up would render a different drawing than the live viewer saw.
  live = b.stroke("whiteboard_b", "tree", square(size=0.02))
  assert b.snapshots()[1]["strokes"][-1]["points"] == live["points"]


def test_a_blank_board_has_no_snapshot():
  """Nothing to say is said with silence: a snapshot per blank board would
  be pure overhead on every connect, and a joiner's canvas already starts
  empty."""
  b = book()
  assert b.snapshots() == []
  b.stroke("whiteboard_a", "house", square())
  assert [s["board"] for s in b.snapshots()] == ["whiteboard_a"]


def test_erasing_drops_the_remembered_lines_too():
  """`board_cleared` means drop everything painted for that board -- and the
  producer must agree, or the next snapshot would re-paint the drawing the
  robot just erased."""
  b = book()
  b.stroke("whiteboard_a", "house", square())
  b.clear("whiteboard_a")
  assert b["whiteboard_a"].lines == []
  assert b.snapshots() == []


def test_the_remembered_lines_are_capped_and_say_so():
  """An un-erased board must not grow an unbounded state file or an
  unbounded join message. Oldest-first, and counted: a snapshot missing the
  start of a long drawing reports how much it is missing."""
  from pluggybot.hub.boards import MAX_LINES
  b = book()
  for _ in range(MAX_LINES + 5):
    b.stroke("whiteboard_a", "house", square(size=0.01))
  rec = b["whiteboard_a"]
  assert len(rec.lines) == MAX_LINES and rec.dropped == 5
  snap = b.snapshots()[0]
  assert snap["dropped"] == 5
  # Stroke indices stay ABSOLUTE across the eviction, so they still line up
  # with the `stroke` field of the live events the viewer may also have.
  assert snap["strokes"][0]["stroke"] == 5
  assert rec.strokes == MAX_LINES + 5


def test_a_save_is_atomic(tmp_path):
  """Written to a temp file and renamed over the target: a crash mid-write
  leaves the PREVIOUS state rather than a truncated file, which would load as
  an erased board."""
  path = tmp_path / "boards.json"
  b = book(path)
  b.stroke("whiteboard_a", "house", square())
  assert [p.name for p in tmp_path.iterdir()] == ["boards.json"], \
    "a .tmp file was left behind"


# ---- errands ----------------------------------------------------------------


def test_a_drawing_errand_goes_to_the_plotters_own_standoff():
  """The mission's A* destination and the plotter's fine approach must be the
  SAME pose, or the errand navigates somewhere the drawing then has to undo.
  One formula, in drawing.board_standoff."""
  e = drawing_errand(book(), "whiteboard_a", BOARD_A)
  assert e.use_at == board_standoff(BOARD_A)
  assert e.module == "module_pen" and e.use is not None


@pytest.mark.parametrize("name", ["whiteboard_a", "whiteboard_b"])
def test_a_drawing_errand_stands_on_free_floor(name):
  """The same geometric check world_config's use_at gets, for a destination
  that is now computed rather than written down. A standoff inside a wall
  aims a 60 s drive at scenery, and nothing raises."""
  import numpy as np
  board = Board.from_meta(META["boards"][name])
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  x, y = board_standoff(board)
  hit = np.zeros(1, dtype=np.int32)
  dist = mujoco.mj_ray(model, data, np.array([x, y, 3.0]),
                       np.array([0.0, 0.0, -1.0]), None, 1, -1, hit)
  assert dist >= 0 and 3.0 - float(dist) < 0.05, (
    f"the {name} standoff is blocked by {model.geom(int(hit[0])).name}")


def test_an_oversized_figure_is_shrunk_not_clipped():
  """`targets_for` CLIPS, so an oversized figure draws flattened against the
  carriage's travel limit and reports a PERFECT trace -- the pen went exactly
  where it was told. The errand sizes to the ENVELOPE before anything moves."""
  huge = strokes.program("square", size=0.30)
  assert not huge.fits(Envelope.for_board(BOARD_A))
  e = drawing_errand(book(), "whiteboard_a", BOARD_A, program=huge)
  assert e.detail["figure"] == "square"
  assert e.detail["ink_m"] < huge.ink_length, "the figure was not shrunk"


def test_the_carry_errand_is_still_the_milestone_8_one():
  """Existing lifecycle behaviour must survive the generalization: same
  module, same bay, and a `use` phase that does nothing."""
  e = carry_errand()
  assert (e.module, e.station_y, e.use) == ("module_lcd", 0.125, None)


def test_the_errand_menu_is_by_name():
  """What the overseer will choose from (issue #15): a name, not a pile of
  flags."""
  b = board_book("home")
  assert [e.detail["board"] for e in errands_for("draw2", "home", b)] == \
    ["whiteboard_a", "whiteboard_b"]
  assert len(errands_for("draw", "home", b)) == 1
  assert errands_for("none", "home", b) == []
  with pytest.raises(ValueError, match="unknown errand queue"):
    errands_for("juggle", "home", b)


def test_a_world_without_boards_cannot_be_asked_to_draw():
  """room_hub has no whiteboards -- the standing board lives in the bare
  hub_world, which is not a navigated room. Failing here beats a mission that
  fetches the pen and drives at a wall."""
  assert board_book("room_hub") is None
  with pytest.raises(ValueError, match="no whiteboards"):
    errands_for("draw", "room_hub", None)


# ---- the plotter -> board wiring, against real physics ---------------------


@pytest.mark.slow
def test_a_real_drawing_reaches_the_board_state():
  """The seam between the two halves of issue #12, with an actual pen.

  Everything above works on polylines handed in by a test. This is the claim
  that the polylines a DRAWING produces are the ones the board records: one
  `draw` event per stroke, carrying what the pen inked rather than what it
  was commanded, arriving as each stroke finishes rather than in a lump at
  the end (which is what makes a drawing watchable).
  """
  from pluggybot.hub.coupling import HUB_STATION_YS
  from pluggybot.hub.drawing import PenPlotter
  from pluggybot.hub.swap import HubSwap

  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()

  board = Board.hub()
  b = BoardBook([BoardRecord("board", Envelope.for_board(board).size)])
  events: list = []
  b.on_event.append(events.append)
  plotter = PenPlotter(model, data, swap, board=board)
  plotter.on_stroke = lambda i, pts, name: b.stroke(
    "board", name or "?", pts, t=float(data.time))
  assert plotter.drive_to_board(), "never reached the board"

  program = strokes.program("text", text="HI", cap_height=0.025)
  r = plotter.draw_program(program)
  assert r["drew"], f"never got the pen on the board: {r}"

  rec = b["board"]
  assert rec.programs == ["text"]
  assert rec.strokes == len(events) == len(program.strokes)
  assert rec.ink_m > 0.02, f"a whole word left {rec.ink_m * 1000:.0f} mm of ink"
  assert 0.0 < rec.fill < 1.0
  # Streamed as it was drawn, not flushed at the end: each event carries the
  # sim time of the stroke that produced it, and they are strictly ordered.
  assert all(y["t"] > x["t"] for x, y in zip(events, events[1:])), \
    "the strokes did not arrive one at a time"
  # What is recorded is what the pen DID. The plotter's own form error is
  # ~1 mm, so a polyline identical to the command would mean the trace was
  # never consulted.
  commanded = plotter.commanded[0]
  inked = events[0]["points"]
  assert max(min(math.dist(p, c) for c in commanded) for p in inked) > 0.0
  assert all(abs(y) <= board.half[1] and abs(z) <= board.half[2]
             for y, z in inked), "ink landed off the slab"


def test_board_records_round_trip_through_json():
  rec = BoardRecord("b", (0.1, 0.2), programs=["house"], strokes=2,
                    ink_m=0.5, cells={(1, 2), (3, 4)})
  back = BoardRecord.from_json("b", rec.to_json())
  assert (back.reach, back.programs, back.strokes, back.cells) == \
    (rec.reach, rec.programs, rec.strokes, rec.cells)
