"""Errands: a tool, a place, and something to DO there (issue #12).

The hub lifecycle used to have exactly one errand hardcoded into it -- fetch
`module_lcd`, drive to a hardcoded point, drive back -- and its `USE_TOOL`
state did no work at all. That was honest for milestone 8, whose claim was
about the swap rather than about the tool, but it means the robot's whole
life is "carry a thing somewhere and put it back".

An errand is that shape with the middle filled in:

    SWAP_PICK  ->  USE_TOOL  ->  SWAP_RETURN
    fetch `module`  drive to `use_at`,   hang it back
    from `station_y`  then run `use()`     in `station_y`

`use` is an arbitrary callable handed the live `HubLifecycle`, so the drawing
errand below and the claw/garden errands that come later are the SAME machinery
with a different middle -- and the pick/carry/stow half, which is the part that
took two issues to make repeatable, has exactly one implementation.

Two things the split buys immediately:

  - `scripts/home_draw.py` stops being a parallel mission stack. It was a
    second copy of "spawn, explore-ish, fetch, drive, draw, stow" that drifted
    from the lifecycle's copy; now it builds an errand and hands it over.
  - The overseer (issue #15) gets a menu it can choose from without knowing
    any physics: an errand is a name, a tool and a destination.

The one rule an errand's `use` must respect: LEAVE THE TOOL IN ITS CARRY
CONFIGURATION. Every stow computes its release heights from the lift it starts
at, and a tool axis parked where the last stroke left it fouls the bay's
brackets (docs/SimNotes.md, "The pen would not stow"). `PenPlotter.carry_config`
is that for the pen; a tool without a moving axis has nothing to do.
"""

from dataclasses import dataclass, field
from typing import Callable

from pluggybot.hub.coupling import HUB_STATION_YS
from pluggybot.hub.drawing import Board, Envelope, PenPlotter, board_standoff
from pluggybot.hub.strokes import StrokeProgram, from_cli

LCD_BAY = HUB_STATION_YS[0]
PEN_BAY = HUB_STATION_YS[2]      # bay C, as every pen demo has used


@dataclass
class Errand:
  """One fetch-use-stow job.

  `use` is called with the running `HubLifecycle` once the robot has arrived
  at `use_at` with the tool on the fork, and whatever dict it returns is
  reported alongside the swap verdicts. Returning nothing is fine -- that is
  the milestone-8 carry errand, which proves the swap and does no work.
  """

  name: str
  module: str
  station_y: float
  use_at: tuple[float, float]
  use: Callable[["object"], dict | None] | None = None
  #: free-form, for the demo scripts and the overseer's ledger
  detail: dict = field(default_factory=dict)


def carry_errand(module: str = "module_lcd", station_y: float = LCD_BAY,
                 use_at: tuple[float, float] = (-1.2, 2.5),
                 name: str = "carry") -> Errand:
  """The milestone-8 errand: fetch a tool, take it somewhere, bring it back.

  Kept as its own thing rather than deleted, and not for nostalgia: it is the
  only errand that exercises the swap stack with NOTHING else running, so when
  a drawing errand fails this is how you find out whether the pick broke or the
  pen did.
  """
  return Errand(name=name, module=module, station_y=station_y, use_at=use_at)


def drawing_errand(book, board_name: str, board: Board,
                   program: StrokeProgram | None = None,
                   program_name: str = "house",
                   size: float | None = None, text: str | None = None,
                   erase: bool = True, station_y: float = PEN_BAY,
                   module: str = "module_pen", on_drawn=None) -> Errand:
  """Fetch the pen, square up to a board, erase it, draw, restore the carry pose.

  `use_at` is the plotter's own board standoff, so the mission's A* takes the
  robot to the exact pose the fine approach expects and the fine approach is
  then a settle rather than a drive.

  ERASING IS PART OF THE ERRAND, by default. A task should not have to share a
  board with whatever was there before -- and it is the sim, not the browser,
  that decides a board is blank, because the sim is what owns board state.

  The figure is sized against the pen's ENVELOPE, never against the slab
  (`Envelope.for_board`): `targets_for` CLIPS, so an oversized figure draws
  flattened against the carriage's travel limit and reports a PERFECT trace,
  because the pen went exactly where it was told to go.
  """
  envelope = Envelope.for_board(board)
  figure = program if program is not None else from_cli(program_name, size, text)
  if not figure.fits(envelope):
    figure = figure.fitted(envelope)

  def use(life) -> dict:
    plotter = PenPlotter(life.model, life.data, life.mission.swap, board=board)
    # The stroke hook is wired HERE rather than in the plotter's constructor
    # signature at the call site, because it needs the errand's program name:
    # a bare polyline list has no name, and "which programs are on this board"
    # is the board state's whole point.
    plotter.on_stroke = lambda i, points, name: book.stroke(
      board_name, name or figure.name, points, t=float(life.data.time))
    squared = plotter.drive_to_board()
    if erase:
      book.clear(board_name, t=float(life.data.time))
      life._say(f"USE_TOOL: erased {board_name}")
    life._say(f"USE_TOOL: drawing {figure.name} on {board_name} "
              f"({len(figure.strokes)} strokes, {figure.ink_length:.2f} m of ink)")
    result = plotter.draw_program(figure)
    # `on_drawn(life, plotter, result)` fires with the robot STILL AT THE
    # BOARD, which is the only moment a photograph of the drawing is worth
    # taking -- scripts/home_draw.py's filmstrip hangs off this, and without
    # the seam the demo would have to re-implement the errand to get it.
    if on_drawn is not None:
      on_drawn(life, plotter, result)
    # Back to the carry pose BEFORE the stow drives anywhere: lift to carry
    # height and centre the carriage, or the module fouls its bay's brackets
    # on the way in (issue #10).
    plotter.carry_config()
    rec = book[board_name]
    life._say(f"USE_TOOL: drew {result.get('strokes_drawn')}/"
              f"{result.get('strokes')} strokes, "
              f"{result.get('inked_fraction', 0):.0%} inked, "
              f"{board_name} now {rec.fill:.0%} full")
    return {"squared": squared, "board": board_name, "figure": figure.name,
            "fill": rec.fill, "plotter": plotter, **result}

  return Errand(name=f"draw:{board_name}", module=module,
                station_y=station_y, use_at=board_standoff(board), use=use,
                detail={"board": board_name, "figure": figure.name,
                        "strokes": len(figure.strokes),
                        "ink_m": figure.ink_length})
