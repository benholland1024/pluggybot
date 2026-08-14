"""Dummy WebSocket sink: what serve.py talks to when there is no website.

Listens, counts, and -- the part a plain `websocat` cannot do -- measures
FRAME GAPS: the wall-clock spacing between telemetry frames as received.
At 20 Hz a healthy live stream arrives every 50 ms; a stall in the sim (or
a publisher blocking the physics step, the bug this exists to catch) shows
up here as a gap. Prints a summary per connection.

Usage:
  uv run python scripts/ws_sink.py [--port 8765] [--quiet]
"""

import argparse
import json
import time


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--quiet", action="store_true",
                      help="no per-event lines, summaries only")
  args = parser.parse_args()

  from websockets.sync.server import serve

  def handle(ws) -> None:
    print(f"[sink] connection from {ws.remote_address}")
    counts: dict[str, int] = {}
    gaps: list[float] = []
    last_frame_wall = None
    t0 = t1 = None
    try:
      for raw in ws:
        msg = json.loads(raw)
        kind = msg.get("type", "frame")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "frame":
          now = time.monotonic()
          if last_frame_wall is not None:
            gaps.append(now - last_frame_wall)
          last_frame_wall = now
          t0 = msg["t"] if t0 is None else t0
          t1 = msg["t"]
        elif kind == "event" and not args.quiet:
          print(f"[sink] event t={msg['t']:7.1f}  {msg['line']}")
    except Exception as e:
      print(f"[sink] connection ended: {type(e).__name__}")
    print(f"[sink] received: {counts}")
    if gaps:
      gaps.sort()
      n = len(gaps)
      print(f"[sink] frame gaps (wall ms): median {gaps[n // 2] * 1e3:.0f}"
            f", p99 {gaps[int(n * 0.99)] * 1e3:.0f}"
            f", max {gaps[-1] * 1e3:.0f}"
            f"  over {n + 1} frames, sim t {t0:.1f} -> {t1:.1f}")

  with serve(handle, "localhost", args.port) as server:
    print(f"[sink] listening on ws://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
  main()
