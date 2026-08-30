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

As of protocol 0.7.0 it talks BACK (issue #16): type a line and it goes
down the socket as a visitor message, and the robot's `visitor_reply`
comes back up. That makes this the smallest possible end-to-end test of
the visitor channel -- no website, no database, no rate limiter, just you
and the robot deciding whether to take your advice.

Usage:
  uv run python scripts/ws_sink.py [--port 8765] [--quiet] [--token SECRET]
  ... then type at it:
      draw a tree on whiteboard_b        # just talk to it -- as of 0.14.0
      what are you doing                 # there is one kind of message, and
      hello!                             # working out which is the robot's job
      rate 3 0.8                         # rate ledger entry 3 at 80 %
"""

import argparse
import json
import sys
import threading
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

  def typed_lines(ws) -> None:
    """stdin -> visitor messages, on this connection (issue #16).

    A daemon thread per connection, and deliberately crude: it is a dev
    tool, and the real sender is the website's relay. The ONE thing it takes
    seriously is giving every message an id, because the id is what the
    robot's answer comes back on.
    """
    n = 0
    for line in sys.stdin:
      line = line.strip()
      if not line:
        continue
      n += 1
      if line.startswith("rate "):
        parts = line.split()
        try:
          msg = {"type": "rating", "id": f"r{n}", "seq": int(parts[1]),
                 "quality": float(parts[2])}
        except (IndexError, ValueError):
          print("[sink] usage: rate <ledger seq> <0..1>")
          continue
      else:
        msg = {"type": "message", "id": f"m{n}", "from": "console",
               "text": line}
      try:
        ws.send(json.dumps(msg))
        print(f"[sink] -> {msg['type']} {msg['id']}")
      except Exception as e:
        print(f"[sink] could not send: {type(e).__name__}")
        return

  def handle(ws) -> None:
    print(f"[sink] connection from {ws.remote_address}")
    if not args.quiet:
      print("[sink] say anything to the robot, or 'rate <seq> <0..1>'")
      threading.Thread(target=typed_lines, args=(ws,), daemon=True).start()
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
        elif kind == "visitor_reply":
          # The other half of the loop (issue #16): what the robot decided
          # about something somebody said. Printed even when quiet -- it is
          # the whole reason this direction exists.
          print(f"[sink] <- {msg['outcome'].upper()} {msg['id']}: "
                f"{msg.get('reply') or '(no reply)'}")
        elif kind == "journal" and not args.quiet:
          print(f"[sink] journal t={msg['t']:7.1f}  {msg['text']}")
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
