"""Dummy WebSocket sink: what serve.py talks to when there is no website.

Listens, counts, and -- the part a plain `websocat` cannot do -- measures
FRAME GAPS: the wall-clock spacing between telemetry frames as received.
At 20 Hz a healthy live stream arrives every 50 ms; a stall in the sim (or
a publisher blocking the physics step, the bug this exists to catch) shows
up here as a gap. Prints a summary per connection.

It also stands in for the website's relay hub (rooftop-media-2026 #22) on
the two things the hub needs from the producer: --token rejects a
connection whose `Authorization: Bearer` does not match (the ingest
secret), and the summary reports the KEYFRAME spacing -- the worst-case
wait before a browser joining mid-stream is rendering a complete world,
and the depth of cache the real hub must hold to spare it that wait.

Usage:
  uv run python scripts/ws_sink.py [--port 8765] [--quiet] [--token SECRET]
"""

import argparse
import json
import time


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--quiet", action="store_true",
                      help="no per-event lines, summaries only")
  parser.add_argument("--token", default=None,
                      help="require this ingest secret (default: accept all)")
  args = parser.parse_args()

  from websockets.sync.server import serve

  def check_auth(connection, request):
    if args.token is None:
      return None                                    # open sink, as before
    if request.headers.get("Authorization") != f"Bearer {args.token}":
      print("[sink] REJECTED a connection: bad or missing token")
      return connection.respond(401, "unauthorized\n")
    return None

  def handle(ws) -> None:
    print(f"[sink] connection from {ws.remote_address}")
    counts: dict[str, int] = {}
    gaps: list[float] = []
    key_gaps: list[float] = []
    last_frame_wall = None
    last_key_t = None
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
          if msg.get("key"):
            counts["keyframe"] = counts.get("keyframe", 0) + 1
            if last_key_t is not None:
              key_gaps.append(msg["t"] - last_key_t)
            last_key_t = msg["t"]
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
    if key_gaps:
      print(f"[sink] keyframes: {counts.get('keyframe', 0)}"
            f", spacing (sim s) median {sorted(key_gaps)[len(key_gaps) // 2]:.2f}"
            f", max {max(key_gaps):.2f}"
            "  <- a mid-stream joiner's worst-case wait")
    elif counts.get("keyframe"):
      # One keyframe and no second one is normal on a run shorter than the
      # interval -- only complain once the run was long enough to show two.
      span = (t1 - t0) if t0 is not None else 0.0
      print(f"[sink] 1 keyframe in {span:.1f} sim s: too short to measure"
            " the cadence")
    elif counts.get("frame"):
      print("[sink] NO keyframes at all: a mid-stream joiner would never"
            " complete (--keyframe-s 0?)")

  with serve(handle, "localhost", args.port, process_request=check_auth) as server:
    print(f"[sink] listening on ws://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
  main()
