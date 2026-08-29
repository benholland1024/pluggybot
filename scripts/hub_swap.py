"""Tool-swap demo (milestone 8): the fork robot exchanges modules at the hub.

The robot starts at the hand-off pose in front of station A, picks the LCD
module off the hub, carries it away, brings it back, and hangs it up again —
the coupling spike's verbs driven by the real base, lift, and odometry.

Usage:
  MUJOCO_GL=egl uv run python scripts/hub_swap.py             # headless + film
  MUJOCO_GL=egl uv run python scripts/hub_swap.py --dy 0.003  # with jitter
"""

import argparse

import mujoco
import numpy as np
from PIL import Image

from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.swap import HubSwap


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dy", type=float, default=0.0, help="lateral jitter (m)")
  parser.add_argument("--dyaw", type=float, default=0.0, help="yaw jitter (deg)")
  parser.add_argument("--film", action="store_true", help="save hub_swap.png")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  frames: list[np.ndarray] = []
  renderer = mujoco.Renderer(model, 360, 480) if args.film else None

  def snap():
    if renderer is not None:
      renderer.update_scene(data, camera="hub_watch")
      frames.append(renderer.render().copy())

  swap.place_at_standoff(HUB_STATION_YS[0], dy=args.dy, dyaw_deg=args.dyaw)
  snap()

  why = swap.pick()
  state = swap.module_state("module_lcd")
  print(f"pick   : approach={why}  on_fork={state['on_fork']} "
        f"pos=({state['pos'][0]:+.3f}, {state['pos'][1]:+.3f}, {state['pos'][2]:.3f})")
  snap()

  why = swap.put_back()
  state = swap.module_state("module_lcd")
  print(f"return : approach={why}  hung={state['hung']} "
        f"pos=({state['pos'][0]:+.3f}, {state['pos'][1]:+.3f}, {state['pos'][2]:.3f})")
  snap()

  ok = state["hung"]
  print("SWAP CYCLE:", "OK" if ok else "FAILED")

  if renderer is not None:
    h, w, _ = frames[0].shape
    sheet = np.zeros((h, w * len(frames), 3), dtype=np.uint8)
    for i, f in enumerate(frames):
      sheet[:, i * w:(i + 1) * w] = f
    Image.fromarray(sheet).save("hub_swap.png")
    renderer.close()
    print("film: hub_swap.png")


if __name__ == "__main__":
  main()
