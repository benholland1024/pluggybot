"""Hub coupling tolerance sweep (milestone-8 prep; schuko_spike's sibling).

Measures what the fork-and-peg gravity latch actually forgives before any
hub layout or controller depends on it:
  - pick-and-return cycles across lateral / vertical / yaw misalignment
  - retention: how hard the carried tool can be shaken before it drops

Usage:
  MUJOCO_GL=egl uv run python scripts/hub_spike.py            # the sweep
  MUJOCO_GL=egl uv run python scripts/hub_spike.py --film     # + filmstrip
"""

import argparse

import numpy as np
from PIL import Image

from pluggybot.rack.coupling import run_cycle, run_pick


def sweep(name: str, values, kw_of) -> None:
  print(f"\n-- {name} sweep (pick + return cycle) --")
  for v in values:
    res, _ = run_cycle(**kw_of(v))
    verdict = ("ok" if res["picked"] and res["returned"]
               else "pick-only" if res["picked"] else "FAIL")
    print(f"  {name}={v:+.3f}: {verdict:9s} max force {res['max_force_n']:5.1f} N")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--film", action="store_true",
                      help="save hub_spike.png filmstrip of an aligned cycle")
  args = parser.parse_args()

  if args.film:
    res, frames = run_cycle(n_frames=12)
    h, w, _ = frames[0].shape
    sheet = np.zeros((h * 3, w * 4, 3), dtype=np.uint8)
    for i, f in enumerate(frames[:12]):
      sheet[(i // 4) * h:(i // 4 + 1) * h, (i % 4) * w:(i % 4 + 1) * w] = f
    Image.fromarray(sheet).save("hub_spike.png")
    print(f"aligned cycle: {res}  -> hub_spike.png")

  res, _ = run_cycle()
  print(f"aligned: picked={res['picked']} returned={res['returned']} "
        f"max force {res['max_force_n']:.1f} N")

  sweep("dy", [d * 1e-3 for d in (-8, -6, -4, -2, 2, 4, 6, 8)],
        kw_of=lambda v: {"dy": v})
  sweep("dz", [d * 1e-3 for d in (-8, -6, -4, -2, 2, 4, 6)],
        kw_of=lambda v: {"dz": v})
  sweep("yaw", [-6, -4, -2, 2, 4, 6],
        kw_of=lambda v: {"yaw_deg": v})

  print("\n-- retention: shake the carried tool (drive-accel analog) --")
  for accel in (2.0, 4.0, 6.0, 8.0):
    res, _ = run_pick(shake_accel=accel)
    print(f"  {accel:.0f} m/s^2: {'held' if res['carried'] else 'DROPPED'}")

  print("\n-- tool mass budget --")
  for mass in (0.12, 0.20, 0.30):
    res, _ = run_cycle(tool_mass=mass)
    print(f"  {mass * 1000:.0f} g: "
          f"{'ok' if res['picked'] and res['returned'] else 'FAIL'}")


if __name__ == "__main__":
  main()
