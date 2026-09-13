"""Build an agent-described tool and run it through the coupling rig.

  MUJOCO_GL=egl uv run python scripts/workshop.py --example
  MUJOCO_GL=egl uv run python scripts/workshop.py --spec my_tool.json

Validates the spec against the envelope (every reason at once), builds the
module, and answers the four questions of docs/ToolPattern.md §5 off the
world: does it hang, is it picked, does it conduct, does it stow -- plus
every axis run to its far end and back. Saves a filmstrip PNG named after
the script. `--sweep` also runs the lateral envelope (±4 mm, ±8 mm).
"""

import argparse
import json
import sys
import time

import numpy as np

from pluggybot.workshop import build, validate
from pluggybot.workshop.spec import Refused

EXAMPLE = {
  "name": "scoop",
  "parts": [
    {"id": "hinge", "part": "servo_fs90", "pos": [-20, 0, -45],
     "axis": {"verb": "tilt", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
    {"id": "blade", "part": "scaffold_pla_box", "size": [60, 30, 4],
     "pos": [-30, 0, -8], "on": "hinge"},
  ],
}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--spec", help="a tool spec, JSON")
  parser.add_argument("--example", action="store_true", help="the scoop")
  parser.add_argument("--sweep", action="store_true", help="the lateral envelope too")
  parser.add_argument("--frames", type=int, default=8)
  parser.add_argument("--out", default="workshop.png")
  args = parser.parse_args()
  raw = EXAMPLE if args.example or not args.spec else json.load(open(args.spec))

  try:
    tool = validate.check(raw)
  except Refused as e:
    print("REFUSED:")
    for r in e.reasons:
      print("  -", r)
    sys.exit(1)
  mass = (build.coupling.MODULE_MASS + tool.mass) * 1000
  print(f"{tool.body}: {len(tool.parts)} parts, {mass:.0f} g with plate and peg, "
        f"{validate.worst_moment(tool):.3f} N·m worst about the peg, "
        f"verbs {[p.axis.verb for p in tool.axes]}")

  t0 = time.time()
  res, frames = build.rig(tool, n_frames=args.frames)
  print(f"rig ({time.time() - t0:.1f} s): " + ", ".join(
    f"{k}={v}" for k, v in res.items() if k != "poles"))
  if args.sweep:
    for dy in (0.004, -0.004, 0.008):
      r, _ = build.rig(tool, dy=dy)
      print(f"  dy {dy * 1000:+.0f} mm: picked={r['picked']} conducts={r['conducts']} "
            f"stowed={r['stowed']}")
  if frames:
    from PIL import Image
    strip = np.concatenate(frames, axis=1)
    Image.fromarray(strip).save(args.out)
    print(f"filmstrip: {args.out}")


if __name__ == "__main__":
  main()
