"""Guards for stroke programs and the Hershey font (tools/strokes.py, issue #11).

Pure content: no MuJoCo, no model, no data. That is the point of the module
and the reason this whole file runs in milliseconds -- everything expensive
about drawing lives on the other side of the split, in test_drawing.py.
"""

import json
import math

import numpy as np
import pytest

from pluggybot.tools import hershey
from pluggybot.tools import strokes as S
from pluggybot.rack.coupling import PEN_TRAVEL
from pluggybot.tools.drawing import Board, Envelope, PenPlotter, square_path


@pytest.fixture(scope="module")
def boards():
  """Every board the pen can be taken to: the hub's standing one and the
  home world's two wall-mounted ones."""
  meta = json.load(open("models/home_world.meta.json"))
  return [Board.hub()] + [Board.from_meta(s) for s in meta["boards"].values()]


# ---- the font --------------------------------------------------------------


def test_font_covers_printable_ascii():
  for code in range(32, 127):
    assert chr(code) in hershey.GLYPHS, f"no glyph for {chr(code)!r}"
  assert hershey.CAP_UNITS == 21.0
  assert hershey.DESCENDER_UNITS < 0, "descenders must fall below the baseline"


def test_glyphs_sit_on_the_baseline_with_y_up():
  """JHF's y axis points DOWN with the baseline at +9. The parser flips it
  once so nothing downstream has to remember; if that flip is ever dropped,
  every letter draws upside down and every bounds check still passes."""
  cap = max(y for s in hershey.glyph("A").strokes for _, y in s)
  foot = min(y for s in hershey.glyph("A").strokes for _, y in s)
  assert cap > 0 and abs(foot) < 1e-9, "an 'A' stands on the baseline"
  assert min(y for s in hershey.glyph("g").strokes for _, y in s) < 0, \
    "a 'g' has to descend"


def test_unknown_characters_fall_back_rather_than_raise():
  """Text comes from an LLM eventually; one stray character is not a reason
  to lose a three-minute errand."""
  assert not hershey.supported("café")
  assert hershey.glyph("é") is hershey.glyph("?")
  assert S.program("text", text="café").points > 0


def test_text_width_scales_with_cap_height():
  a = hershey.text_width("PLUGGY", 0.010)
  b = hershey.text_width("PLUGGY", 0.030)
  assert b == pytest.approx(3 * a, rel=1e-9)
  assert hershey.text_width("", 0.01) == 0.0
  # A running sum of advances, so a longer string is strictly wider.
  assert hershey.text_width("PLUGGYBOT", 0.01) > hershey.text_width("PLUGGY", 0.01)


def test_wrap_respects_the_line_width():
  lines = hershey.wrap("GOOD MORNING PLUGGY", 0.012, S.TEXT_WIDTH)
  assert len(lines) > 1, "that does not fit on one 100 mm line at 12 mm caps"
  for line in lines:
    assert hershey.text_width(line, 0.012) <= S.TEXT_WIDTH + 1e-9

  # A word wider than the whole line is hard-broken rather than allowed to
  # run off the board -- the carriage has 110 mm of travel and no more.
  broken = hershey.wrap("SUPERCALIFRAGILISTIC", 0.020, 0.050)
  assert len(broken) > 1
  assert "".join(broken) == "SUPERCALIFRAGILISTIC"


def test_layout_is_centred_on_the_origin():
  polys = hershey.layout("HELLO\nWORLD", 0.012)
  xs = [x for p in polys for x, _ in p]
  ys = [y for p in polys for _, y in p]
  assert abs((min(xs) + max(xs)) / 2) < 1e-3
  assert abs((min(ys) + max(ys)) / 2) < 5e-3


# ---- the board's coordinate convention -------------------------------------


def test_text_advances_away_from_the_viewers_left():
  """Board `lat` is measured LEFT of the robot's approach heading, so text
  reading left-to-right advances toward NEGATIVE lat.

  This is the one convention in the module that cannot be checked by looking
  at a number: a mirrored figure has identical bounds, identical ink length
  and an identical form error, and every symmetric test figure -- which is
  both diagnostics -- passes either way. The first thing that can tell the
  difference is a letter.
  """
  first = S.program("text", text="A")
  last = S.program("text", text="Z")
  both = S.program("text", text="AZ")
  a_lat = np.mean([y for s in both.strokes[:len(first.strokes)] for y, _ in s])
  z_lat = np.mean([y for s in both.strokes[len(first.strokes):] for y, _ in s])
  assert a_lat > z_lat, (
    "the first letter must sit to the viewer's LEFT of the last, i.e. at "
    "greater lat -- this text would be drawn mirrored")
  assert len(last.strokes) >= 1


def test_asymmetric_figures_are_flipped_once():
  """The robot's arm is authored to the reading-frame right, so it must come
  out at negative lat. Same flip as the text, same single place it happens."""
  arm = max((s for s in S.program("robot").strokes),
            key=lambda s: max(abs(y) for y, _ in s))
  assert min(y for y, _ in arm) < 0


# ---- programs --------------------------------------------------------------


def test_every_program_fits_every_board(boards):
  """Default parameters must be drawable as-is on every board the pen can
  reach, or the defaults are decoration."""
  for name in sorted(S.PROGRAMS):
    prog = S.program(name)
    for board in boards:
      env = Envelope.for_board(board)
      assert prog.fits(env), (
        f"{name} is {prog.size[0] * 1000:.0f} x {prog.size[1] * 1000:.0f} mm, "
        f"outside the {env.size[0] * 1000:.0f} x {env.size[1] * 1000:.0f} mm "
        f"the pen can reach on {board.geom}")
      assert prog.points >= 2 and prog.ink_length > 0.0


def test_library_figures_have_no_detail_finer_than_the_machine():
  """A stroke shorter than ~10 mm lands visibly detached from what it belongs
  to, because each stroke re-presses and presses scatter by ~4 mm. Measured on
  the board: the house's first window bars were 8 mm and drew as a line
  floating outside the window frame; the robot's first eyes were 4 mm.

  Text is exempt -- its short strokes are parts of a letter, and its size is
  set by cap height (see TEXT_CAP), which was measured the same way.
  """
  for name in S.LIBRARY:
    for stroke in S.program(name).strokes:
      length = sum(math.dist(a, b) for a, b in zip(stroke, stroke[1:]))
      assert length >= S.MIN_FEATURE, (
        f"{name} has a {length * 1000:.1f} mm stroke; under "
        f"{S.MIN_FEATURE * 1000:.0f} mm the plotter cannot place it where it "
        f"belongs")
  assert set(S.LIBRARY) <= set(S.PROGRAMS)


def test_closed_figures_close():
  for name in ("square", "circle"):
    stroke = S.program(name).strokes[0]
    assert math.dist(stroke[0], stroke[-1]) < 1e-9, f"{name} must close"
  # The sun's disc and the tree's canopy are closed too; the rays are not.
  disc = max(S.program("sun").strokes, key=len)
  assert math.dist(disc[0], disc[-1]) < 1e-9


def test_diagnostics_are_ported_not_rewritten():
  """PluggyPlan's 0.59 mm form error was measured against these exact samples.
  Regenerating them 'more cleanly' would quietly move the baseline."""
  assert S.program("square").strokes[0] == tuple(square_path(S.FIGURE_SIZE))
  assert S.program("square", size=0.05).strokes[0] == tuple(square_path(0.05))


def test_registry_rejects_unknown_and_duplicate_names():
  with pytest.raises(KeyError):
    S.program("mona-lisa")
  with pytest.raises(ValueError):
    S.register("square")(lambda: None)


def test_a_program_must_actually_draw_something():
  with pytest.raises(ValueError):
    S.StrokeProgram("empty", ())
  with pytest.raises(ValueError):
    S.StrokeProgram("dot", (((0.0, 0.0),),))
  with pytest.raises(ValueError):
    S.program("text", text="   ")


def test_transforms():
  sq = S.program("square", size=0.06)
  assert sq.ink_length == pytest.approx(4 * 0.06, rel=1e-6)
  assert sq.scaled(2.0).size == pytest.approx((0.12, 0.12))
  moved = sq.translated(0.01, -0.02)
  assert moved.bounds()[0] == pytest.approx(sq.bounds()[0] + 0.01)
  assert moved.centered().bounds()[0] == pytest.approx(-0.03)


def test_fitted_shrinks_but_never_grows(boards):
  env = Envelope.for_board(boards[0])
  big = S.program("square", size=0.30)
  assert not big.fits(env)
  assert big.fitted(env).fits(env)
  small = S.program("square", size=0.02)
  assert small.fitted(env).size == pytest.approx(small.size)


def test_shrink_keeps_words_whole():
  """Without the shrink pass, "GOOD MORNING" at a 12 mm cap becomes
  "GOOD MORNI / NG": still a legal drawing, still inside every bound, and not
  what anyone asked for."""
  cap, lines = S.fit_text("GOOD MORNING", 0.020, shrink=False)
  assert cap == 0.020
  assert lines != ["GOOD", "MORNING"], "expected the un-shrunk layout to break"

  cap, lines = S.fit_text("GOOD MORNING", 0.020)
  assert cap < 0.020, "the message did not fit and was not shrunk"
  assert lines == ["GOOD", "MORNING"], f"a word got broken: {lines}"
  assert S.program("text", text="GOOD MORNING",
                   cap_height=0.020).size[0] <= S.TEXT_WIDTH + 1e-9

  # Shrinking is one-way: text that already fits is drawn at the size asked.
  assert S.fit_text("HI", 0.020)[0] == 0.020


# ---- the measurement the programs are scored with --------------------------


def test_envelope_is_the_carriage_not_the_board(boards):
  """The board is 320 mm wide; the pen can reach 110 mm of it.

  Sizing content to the board rather than to the carriage is the obvious
  mistake here -- `targets_for` CLIPS an out-of-range command, so the figure
  comes out flattened against the travel limit while the error stats report
  the pen tracked its commands perfectly. It went exactly where it was told.
  """
  for board in boards:
    env = Envelope.for_board(board)
    assert env.size[0] == pytest.approx(2 * PEN_TRAVEL)
    assert env.size[0] < 2 * board.half[1], "the face is scenery, not reach"


def test_nearest_does_not_invent_segments_between_strokes():
  """Two strokes are not one polyline.

  Concatenating a program's points before measuring creates a phantom segment
  from the end of one stroke to the start of the next, and ink measured
  against a line the pen was never asked to draw scores as a GOOD drawing --
  the failure is silent and it flatters. Here the phantom segment runs
  diagonally through (0.5, 0.5), where the real strokes are 0.5 away.
  """
  strokes = [((0.0, 0.0), (1.0, 0.0)), ((0.0, 1.0), (1.0, 1.0))]
  d, _ = PenPlotter._nearest(np.array([[0.5, 0.5]]), strokes)
  assert d[0] == pytest.approx(0.5), \
    "measured against a segment joining two separate strokes"
