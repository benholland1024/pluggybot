"""How far the ink sits from the glyphs it was meant to be (issue #22).

The measurement `economy/questions.py`'s fidelity bar is set from, and the one
that has to be re-run whenever the pen, the board or the answer's cap height
moves. It draws with the REAL plotter on a real board and reports, for each
drawing, the symmetric distance to a handful of candidate answers plus how
much ink was used against how much the answer needs.

Two rows are the whole point:

  the answer it was asked for, correctly written  -- what a pass looks like
  a library FIGURE drawn instead of the answer    -- what wrong work looks
                                                     like, and the number
                                                     that sets the bar

The near-miss digits are there to show what the bar CANNOT do: a 6 and an 8
are closer together than a correct drawing is to its own ideal, which is why
correctness is decided against the answer the mind committed to and never by
reading the board (economy/questions.py, "Why the ink is a FIDELITY check").

Usage:
  MUJOCO_GL=egl uv run python scripts/answer_spike.py
  MUJOCO_GL=egl uv run python scripts/answer_spike.py --answers 5 12 42
"""

import argparse

import mujoco

from pluggybot.economy import questions
from pluggybot.tools import strokes
from pluggybot.tools.boards import decimate
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.tools.drawing import PenPlotter
from pluggybot.rack.swap import HubSwap

CANDIDATES = ("5", "6", "8", "12", "15")


def draw(model, program) -> tuple[list, dict]:
  """Draw one program with the pen module and return the ink the BOARD would
  have recorded -- decimated, exactly as `BoardBook.stroke` stores it, since
  that is what the evaluator reads."""
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  swap.pick()
  plotter = PenPlotter(model, data, swap)
  if not plotter.drive_to_board():
    raise SystemExit("never reached the board")
  inked: list = []
  plotter.on_stroke = lambda i, pts, name: inked.append(decimate(pts))
  result = plotter.draw_program(program)
  return [s for s in inked if len(s) >= 2], result


def report(label: str, ink, result, want: str, candidates) -> None:
  print(f"\n-- {label}: drew {result.get('strokes_drawn')}/"
        f"{result.get('strokes')} strokes, form "
        f"{result.get('form_rms_mm', float('nan')):.2f} mm, "
        f"{result.get('inked_fraction', 0):.0%} inked")
  for cand in candidates:
    m = questions.ink_match(ink, cand)
    if m is None:
      print(f"   vs {cand!r:5}: no ink on the board at all")
      continue
    verdict = "PASS" if (m["matchMm"] <= questions.ANSWER_MATCH_MM
                         and questions.INK_RATIO[0] <= m["inkRatio"]
                         <= questions.INK_RATIO[1]) else "fail"
    mark = " <- asked for" if cand == want else ""
    print(f"   vs {cand!r:5}: {m['matchMm']:6.2f} mm  "
          f"(ink->glyph {m['inkToGlyphMm']:5.2f}, glyph->ink "
          f"{m['glyphToInkMm']:5.2f})  ink x{m['inkRatio']:.2f}  "
          f"{verdict}{mark}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--answers", nargs="*", default=["5", "12"],
                      help="answers to write with the real pen")
  parser.add_argument("--figure", default="robot",
                      help="a library figure to draw INSTEAD of the answer")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  print(f"bar: {questions.ANSWER_MATCH_MM} mm, ink ratio "
        f"{questions.INK_RATIO[0]}-{questions.INK_RATIO[1]}x, "
        f"cap {questions.ANSWER_CAP * 1000:.0f} mm")
  wanted = list(dict.fromkeys([*args.answers, *CANDIDATES]))
  for answer in args.answers:
    ink, result = draw(model, strokes.program("answer", text=answer))
    report(f"answer {answer!r}", ink, result, answer, wanted)
  if args.figure:
    ink, result = draw(model, strokes.program(args.figure))
    report(f"{args.figure} figure, drawn instead of an answer", ink, result,
           "", args.answers)


if __name__ == "__main__":
  main()
