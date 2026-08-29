"""Whiteboards as persistent world state, not scenery (issue #12).

A board that only ever gets written on fills up and stops being interesting.
So the sim OWNS each drawing surface: which stroke programs are on it, how
much of the pen's reach carries ink, and when it was last cleared. That state
outlives a mission (`--boards state.json`) the same way a real whiteboard
outlives the person who drew on it.

Three rules, and the first is the whole design:

  STROKES ARE NEVER MUJOCO GEOMETRY. Ink is layer 3 of the PluggyWorld model
  (activity/base.py): rigid bodies in MuJoCo, discrete state in Python,
  everything continuous or organic in the browser. A drawn line is a
  `draw` EVENT -- board id plus the polyline the pen actually traced, in
  board-local metres -- which the website paints into a canvas texture. Adding
  a geom per stroke would put thousands of collision-eligible slabs into a
  world the robot has to drive through, and MuJoCo cannot add geometry to a
  compiled model anyway.

  WHAT IS RECORDED IS WHAT THE PEN DID, not what it was asked to do. The
  polyline comes off `PenPlotter`'s trace, filtered to the samples where the
  shaft was actually touching the board. The same doctrine as every other
  criterion in this repo: a commanded path is a belief, ink is a fact. It also
  means the board shows the robot's real 0.6-1.1 mm of form error, which is
  the honest picture and the more charming one.

  ERASING IS AN ACTION, not a reset. `clear()` is called by the drawing errand
  before it starts (a task should not have to share a board), and is equally
  available to an overseer as a standalone choice (issue #15). v1 erasing is
  bookkeeping plus a `board_cleared` event; a robot physically wiping a board
  with an eraser module is a later task, and when it lands it calls this same
  method -- the state machine does not care what moved the ink.

`fill` is measured against the pen's REACH, not against the slab. The home
world's boards are 320 x 260 mm but the carriage has 110 mm of travel and the
base is parked for the whole drawing (drawing.Envelope), so a board is "full"
when the window the robot can actually reach is inked. Scoring against the
slab would report a board the robot has no way to fill as 12 % after its best
possible effort.
"""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.telemetry.protocol import ROBOT_ROOT

CELL = 0.01          # m of board per coverage cell. The plotter's own form
                     # error is ~1 mm and its finest authored feature is 10 mm
                     # (strokes.MIN_FEATURE), so a 10 mm cell is the smallest
                     # square that means "there is a drawing here" rather than
                     # "a line passed through here".
POINT_EPS = 0.001    # m. Trace samples land ~0.04 mm apart at DRAW_SPEED, so a
                     # raw stroke is thousands of points for a figure a browser
                     # draws in a few dozen. Decimated by distance, which keeps
                     # corners (they are far from the previous KEPT point)
                     # while dropping the straights.
MAX_POINTS = 400     # per stroke, after decimation: a hard ceiling on one
                     # websocket message, so a pathological figure degrades to
                     # a coarse polyline instead of stalling the sender thread.
NDIGITS = 4          # 0.1 mm, as for poses
MAX_LINES = 240      # strokes KEPT per board for the catch-up snapshot
                     # (issue #13 / rooftop-media-2026 #28). A board that
                     # deep is visually saturated long before it is full, and
                     # the cap is what stops an un-erased board growing an
                     # unbounded state file and an unbounded join message.

STATE_VERSION = 2    # 2 adds `lines`: the polylines themselves, so a board's
                     # INK survives a restart and not merely its statistics.
                     # A version-1 file still loads -- its boards come back
                     # with their counters and no strokes to redraw, which is
                     # exactly what they had recorded.


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def decimate(points, eps: float = POINT_EPS,
             cap: int = MAX_POINTS) -> list[tuple[float, float]]:
  """Thin a traced polyline to what a browser needs to redraw it.

  Distance-based rather than every-nth: the pen slows into corners, so index
  decimation drops the straights and the corners at the same rate and rounds
  the figure off. The first and last samples are always kept -- a stroke that
  loses its endpoint stops meeting the one it was drawn to meet.
  """
  pts = [(float(y), float(z)) for y, z in points]
  if len(pts) <= 2:
    return pts
  kept = [pts[0]]
  for p in pts[1:-1]:
    if math.dist(p, kept[-1]) >= eps:
      kept.append(p)
  kept.append(pts[-1])
  if len(kept) > cap:
    step = math.ceil(len(kept) / cap)
    kept = kept[::step]
    if kept[-1] != pts[-1]:      # the endpoint survives the thinning too
      kept.append(pts[-1])
  return kept


@dataclass
class BoardRecord:
  """One drawing surface's persistent state."""

  name: str
  #: (width, height) of the pen's reachable window on this board, m -- the
  #: denominator of `fill`. Comes from drawing.Envelope, which is where the
  #: carriage travel and lift range live.
  reach: tuple[float, float]
  programs: list[str] = field(default_factory=list)
  strokes: int = 0
  ink_m: float = 0.0
  clears: int = 0
  cleared_at: str | None = None
  drawn_at: str | None = None
  #: inked coverage cells, board-local indices. A set, so re-drawing over the
  #: same place does not inflate the fill.
  cells: set[tuple[int, int]] = field(default_factory=set)
  #: the strokes THEMSELVES, {"program", "points"} each, oldest first. The
  #: counters above describe a drawing; this is the drawing. Kept because ink
  #: is not a body: no keyframe re-ships it, so a viewer who arrives after
  #: the pen has moved on -- or after the producer restarted -- has no other
  #: way to be told what is on the wall (protocol 0.5.0, `board_snapshot`).
  lines: list[dict] = field(default_factory=list)
  #: strokes dropped off the front by MAX_LINES. Reported rather than hidden:
  #: a snapshot that is missing the start of the drawing should say so.
  dropped: int = 0

  @property
  def fill(self) -> float:
    area = self.reach[0] * self.reach[1]
    if area <= 0:
      return 0.0
    return min(len(self.cells) * CELL * CELL / area, 1.0)

  @property
  def blank(self) -> bool:
    return not self.strokes

  def flags(self) -> dict:
    """The telemetry block: small, JSON-ready, re-shipped on every keyframe."""
    return {
      "programs": list(self.programs),
      "strokes": self.strokes,
      "inkM": round(self.ink_m, 3),
      "fill": round(self.fill, 3),
      "clears": self.clears,
      "clearedAt": self.cleared_at,
      "drawnAt": self.drawn_at,
    }

  # ---- persistence ---------------------------------------------------------

  def to_json(self) -> dict:
    return {"reach": list(self.reach), "programs": list(self.programs),
            "strokes": self.strokes, "inkM": self.ink_m,
            "clears": self.clears, "clearedAt": self.cleared_at,
            "drawnAt": self.drawn_at, "dropped": self.dropped,
            "lines": [dict(line) for line in self.lines],
            "cells": sorted([int(i), int(j)] for i, j in self.cells)}

  @classmethod
  def from_json(cls, name: str, spec: dict,
                reach: tuple[float, float] | None = None) -> "BoardRecord":
    # `reach` comes from the WORLD when one is supplied: the saved figure is
    # a property of the pen and the board geometry, and a state file written
    # before a board was resized must not keep scoring against the old one.
    return cls(
      name=name,
      reach=tuple(reach if reach is not None else spec["reach"]),  # type: ignore[arg-type]
      programs=list(spec.get("programs", [])),
      strokes=int(spec.get("strokes", 0)),
      ink_m=float(spec.get("inkM", 0.0)),
      clears=int(spec.get("clears", 0)),
      cleared_at=spec.get("clearedAt"),
      drawn_at=spec.get("drawnAt"),
      dropped=int(spec.get("dropped", 0)),
      # A version-1 file has no `lines`, and that is not an error: the board
      # comes back with its statistics and nothing to redraw.
      lines=[{"program": str(line["program"]),
              "points": [[float(a), float(b)] for a, b in line["points"]]}
             for line in spec.get("lines", [])],
      cells={(int(i), int(j)) for i, j in spec.get("cells", [])},
    )


class BoardBook:
  """Every drawing surface in one world: state, events, and a file to survive in.

  The telemetry surface is deliberately the same duck type an `ActivitySet`
  presents (`names` + `snapshot()`), because the frame builder diffs both the
  same way -- and for the same reason. Like an activity's, a board's visible
  effect is invisible to the pose stream by construction: ink is not a body,
  so these flags plus the `draw` events are the ONLY record of it on the wire.

  `on_event` receives complete protocol messages (`draw`, `board_cleared`) as
  they happen. Wire the live publisher and the recorder into it; both are
  sinks, and neither knows the other exists.
  """

  def __init__(self, records=(), path: str | os.PathLike | None = None,
               clock: Callable[[], str] = _now) -> None:
    self.boards: dict[str, BoardRecord] = {r.name: r for r in records}
    self.path = Path(path) if path is not None else None
    self.clock = clock
    self.on_event: list[Callable[[dict], None]] = []

  # ---- construction --------------------------------------------------------

  @classmethod
  def for_meta(cls, meta: dict, path: str | os.PathLike | None = None,
               clock: Callable[[], str] = _now) -> "BoardBook":
    """Build from a world generator's sidecar (`models/<world>.meta.json`).

    The reachable window per board comes from `drawing.Envelope`, so the
    fill denominator is the pen's actual travel intersected with the slab --
    never the slab alone.
    """
    from pluggybot.tools.drawing import Board, Envelope
    records = []
    for name, spec in (meta.get("boards") or {}).items():
      env = Envelope.for_board(Board.from_meta(spec))
      records.append(BoardRecord(name=name, reach=env.size))
    book = cls(records, path=path, clock=clock)
    if book.path is not None and book.path.exists():
      book.load()
    return book

  # ---- the telemetry surface (matches ActivitySet) -------------------------

  @property
  def names(self) -> list[str]:
    return list(self.boards)

  def snapshot(self) -> dict:
    return {name: rec.flags() for name, rec in self.boards.items()}

  def __len__(self) -> int:
    return len(self.boards)

  def __contains__(self, name: str) -> bool:
    return name in self.boards

  def __getitem__(self, name: str) -> BoardRecord:
    return self.boards[name]

  def snapshots(self, t: float = 0.0, by: str = ROBOT_ROOT) -> list[dict]:
    """One `board_snapshot` message per board that is carrying ink.

    The catch-up channel, and the reason it has to exist: a keyframe
    re-ships every board's COUNTERS, but there is no keyframe for the lines
    themselves -- ink is not a body, and a `draw` event happens once. So a
    browser that opens the page after the pen has moved on, or after the
    producer restarted onto a state file, sees a board reporting 40 % fill
    with nothing painted on it. This is what closes that gap: sent when a
    stream OPENS, cached by the relay hub, and skipped entirely for blank
    boards (protocol 0.5.0).
    """
    out = []
    for name, rec in self.boards.items():
      if not rec.lines:
        continue
      out.append({
        "type": "board_snapshot", "t": round(float(t), 3), "robot": by,
        "board": name, "clearedAt": rec.cleared_at, "drawnAt": rec.drawn_at,
        "dropped": rec.dropped,
        "strokes": [{"program": line["program"], "stroke": i + rec.dropped,
                     "points": [list(p) for p in line["points"]]}
                    for i, line in enumerate(rec.lines)],
      })
    return out

  # ---- the actions ---------------------------------------------------------

  def clear(self, board: str, t: float = 0.0,
            by: str = ROBOT_ROOT) -> BoardRecord:
    """Erase a board: drop its contents, emit `board_cleared`.

    A first-class action, not a private reset -- the drawing errand calls it
    before it draws, and an overseer may choose it on its own.
    """
    rec = self.boards[board]
    rec.programs.clear()
    rec.cells.clear()
    rec.lines.clear()
    rec.dropped = 0
    rec.strokes = 0
    rec.ink_m = 0.0
    rec.drawn_at = None
    rec.clears += 1
    rec.cleared_at = self.clock()
    self._emit({"type": "board_cleared", "t": round(float(t), 3),
                "robot": by, "board": board, "clearedAt": rec.cleared_at})
    return rec

  def stroke(self, board: str, program: str, points, t: float = 0.0,
             by: str = ROBOT_ROOT) -> dict | None:
    """Record one traced stroke and stream it as a `draw` event.

    `points` are board-local `(lat, height)` metres straight off the
    plotter's trace -- +lat is the viewer's LEFT (drawing.pen_board), which
    the browser flips once at paint time, exactly as `strokes.py` flips once
    at authoring time. Returns the emitted message, or None for a stroke with
    nothing in it (a press that never found the board leaves no ink, and an
    empty polyline is not a drawing).
    """
    pts = decimate(points)
    if len(pts) < 2:
      return None
    rec = self.boards[board]
    if program not in rec.programs:
      rec.programs.append(program)
    rec.strokes += 1
    rec.ink_m += sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    rec.drawn_at = self.clock()
    self._mark(rec, pts)
    points = [[round(y, NDIGITS), round(z, NDIGITS)] for y, z in pts]
    # Remembered, not just streamed: this is the only copy of the line that
    # outlives the message. Oldest-first eviction at the cap, counted.
    rec.lines.append({"program": program, "points": points})
    if len(rec.lines) > MAX_LINES:
      rec.dropped += len(rec.lines) - MAX_LINES
      del rec.lines[:len(rec.lines) - MAX_LINES]
    msg = {"type": "draw", "t": round(float(t), 3), "robot": by,
           "board": board, "program": program, "stroke": rec.strokes - 1,
           "points": points}
    self._emit(msg)
    return msg

  @staticmethod
  def _mark(rec: BoardRecord, pts) -> None:
    """Paint the polyline into the coverage grid.

    Segments are WALKED at half a cell, not sampled at their endpoints: a
    75 mm edge drawn as two points would otherwise mark 2 cells out of the 8
    it crosses, and the fill of a finished figure would depend on how the
    polyline happened to be decimated rather than on how much board it covers.
    """
    for a, b in zip(pts, pts[1:]):
      steps = max(int(math.dist(a, b) / (CELL / 2)), 1)
      for k in range(steps + 1):
        f = k / steps
        y = a[0] + (b[0] - a[0]) * f
        z = a[1] + (b[1] - a[1]) * f
        rec.cells.add((math.floor(y / CELL), math.floor(z / CELL)))

  def _emit(self, msg: dict) -> None:
    for hook in self.on_event:
      hook(msg)
    # Saved on every event rather than at mission end: a board that survives
    # only a CLEAN shutdown does not survive the thing restarts are usually
    # about. The file is a few KB and events arrive at most once a second.
    self.save()

  # ---- persistence ---------------------------------------------------------

  def save(self, path: str | os.PathLike | None = None) -> Path | None:
    """Write the state file, atomically. No path configured means no file --
    board state is then per-run, which is what every test and every demo that
    was not asked to remember anything wants."""
    target = Path(path) if path is not None else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(
      {"version": STATE_VERSION,
       "boards": {n: r.to_json() for n, r in self.boards.items()}},
      indent=1) + "\n")
    # Rename over the target: a crash mid-write leaves the PREVIOUS state
    # intact rather than a truncated file that loads as an erased board.
    os.replace(tmp, target)
    return target

  def load(self, path: str | os.PathLike | None = None) -> "BoardBook":
    """Restore from a state file. Boards the file does not mention keep the
    blank state the world gave them; boards the WORLD does not have are
    dropped -- a state file outliving a layout change must not resurrect a
    whiteboard that is no longer on any wall.

    OLDER versions load; newer ones refuse. The asymmetry is deliberate and
    the deploy is what pays for it: `/var/lib/pluggybot` outlives the image,
    so an upgrade that added a field would otherwise crash the sim on start
    over a file whose old shape is a strict subset of the new one. A file
    from the FUTURE is a different story -- that is a downgrade, and reading
    it would silently drop whatever the newer version knew.
    """
    target = Path(path) if path is not None else self.path
    if target is None or not target.exists():
      return self
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if not 1 <= version <= STATE_VERSION:
      raise ValueError(
        f"{target}: board state version {doc.get('version')!r}, expected "
        f"1..{STATE_VERSION} -- delete the file to start the boards blank")
    for name, spec in doc.get("boards", {}).items():
      if name not in self.boards:
        continue
      self.boards[name] = BoardRecord.from_json(
        name, spec, reach=self.boards[name].reach)
    return self
