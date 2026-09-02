"""Save what is on a whiteboard as a PNG -- so a drawing the robot made can be
hung on the website's walls (rooftop-media-2026 #128).

The website hangs framed pictures on interior walls from image files checked
into ITS repo; the robot's own drawings get there by hand, every once in a
while, through this script. Nothing is automated on purpose: a gallery that
curates itself hangs everything, and the ink is already durable in two places
this reads from --

  --boards PATH      the boards state file (`--boards` / `$PLUGGY_BOARDS` on
                     a served world; `/var/lib/pluggybot/boards.json` on the
                     deploy). What is on every board RIGHT NOW.
  --recording PATH   a telemetry recording (`.jsonl.gz`). The board as it
                     stood at the END of the mission, or at `--at T` sim
                     seconds -- the ink a viewer would have seen then.

The image is CROPPED TO THE INK, not the board: the pen's reach is 110 x 200
mm inside a 320 x 260 mm slab, so a whole-board render is mostly blank
whiteboard around a small figure. `--margin` says how much board to keep
round it, and `--whole` keeps all of it.

The one coordinate rule, the website's (`surfaces/board.ts`): +lat is the
VIEWER'S LEFT, so it decreases across the image; height grows upward, so it
decreases down the image. Both flips happen here and nowhere else.

Usage:
  uv run python scripts/board_png.py --boards /var/lib/pluggybot/boards.json \
      --board whiteboard_a house.png
  uv run python scripts/board_png.py --recording protocol/telemetry.home_lifecycle.jsonl.gz \
      --board whiteboard_a house.png
"""

import argparse
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw

# The website's numbers, so a picture on the wall is inked like the board
# behind it: 1.5 px/mm, a 4 mm marker tip, the same two colours.
PX_PER_M = 1500
INK_WIDTH_M = 0.004
INK = "#1d2330"
BOARD = "#f4f2ec"
# Board face, metres: the generator's slab is 320 x 260 mm; an ink-cropped
# image never needs it, `--whole` does.
FACE_W, FACE_H = 0.32, 0.26


def strokes_from_boards(path: Path, board: str) -> list[list[list[float]]]:
  doc = json.loads(path.read_text())
  rec = doc.get("boards", {}).get(board)
  if rec is None:
    raise SystemExit(f"{path} has no board {board!r}; it has {sorted(doc.get('boards', {}))}")
  return [line["points"] for line in rec.get("lines", [])]


def strokes_from_recording(path: Path, board: str, at: float | None) -> list[list[list[float]]]:
  """Replay the ink messages up to `at` (or the end): a snapshot restores,
  a draw adds, a clear wipes -- the website's `boards.ts` rule."""
  strokes: list[list[list[float]]] = []
  with gzip.open(path, "rt") as fh:
    for line in fh:
      msg = json.loads(line)
      kind = msg.get("type")
      if kind not in ("board_snapshot", "draw", "board_cleared"):
        continue
      if msg.get("board") != board:
        continue
      if at is not None and float(msg.get("t", 0.0)) > at:
        break
      if kind == "board_snapshot":
        strokes = [s["points"] for s in msg.get("strokes", [])]
      elif kind == "draw":
        strokes.append(msg["points"])
      else:
        strokes = []
  return strokes


def render(strokes: list[list[list[float]]], margin: float, whole: bool,
           px_per_m: float = PX_PER_M) -> Image.Image:
  pts = [p for s in strokes for p in s]
  if not pts:
    raise SystemExit("nothing is drawn on that board")
  if whole:
    lat0, lat1 = -FACE_W / 2, FACE_W / 2
    h0, h1 = -FACE_H / 2, FACE_H / 2
  else:
    lat0 = min(p[0] for p in pts) - margin
    lat1 = max(p[0] for p in pts) + margin
    h0 = min(p[1] for p in pts) - margin
    h1 = max(p[1] for p in pts) + margin
  width = max(1, round((lat1 - lat0) * px_per_m))
  height = max(1, round((h1 - h0) * px_per_m))
  img = Image.new("RGB", (width, height), BOARD)
  draw = ImageDraw.Draw(img)
  pen = max(1, round(INK_WIDTH_M * px_per_m))

  def px(p):
    # +lat is the viewer's LEFT; height grows upward. Both flip here.
    return ((lat1 - p[0]) * px_per_m, (h1 - p[1]) * px_per_m)

  for s in strokes:
    if len(s) < 2:
      continue
    draw.line([px(p) for p in s], fill=INK, width=pen, joint="curve")
  return img


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  src = ap.add_mutually_exclusive_group(required=True)
  src.add_argument("--boards", type=Path, help="boards state file (JSON)")
  src.add_argument("--recording", type=Path, help="telemetry recording (.jsonl.gz)")
  ap.add_argument("--board", default="whiteboard_a")
  ap.add_argument("--at", type=float, default=None,
                  help="recording only: the board at this sim time, not the end")
  ap.add_argument("--margin", type=float, default=0.012,
                  help="board kept around the ink, metres (default 12 mm)")
  ap.add_argument("--whole", action="store_true", help="the whole face, uncropped")
  ap.add_argument("--px-per-m", type=float, default=4 * PX_PER_M,
                  help="resolution; the board texture itself is 1500, and a "
                       "picture wants more because it hangs closer to the "
                       "camera than a board does (default 6000)")
  ap.add_argument("out", type=Path)
  args = ap.parse_args()

  if args.boards:
    strokes = strokes_from_boards(args.boards, args.board)
  else:
    strokes = strokes_from_recording(args.recording, args.board, args.at)
  img = render(strokes, args.margin, args.whole, args.px_per_m)
  img.save(args.out)
  print(f"{args.out}: {img.width} x {img.height} px, {len(strokes)} strokes")


if __name__ == "__main__":
  main()
