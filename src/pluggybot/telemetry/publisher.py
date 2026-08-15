"""Live WebSocket publisher for PluggyWorld telemetry (webserver v1).

A CLIENT, not a server: the sim connects OUTBOUND to a configured endpoint
(the website's Node process, or scripts/ws_sink.py) and streams the same
frames the v0 recorder writes -- the recorder and the publisher share one
FrameBuilder, so a live viewer and a replayed recording see identical
data. The sim owns no public surface; if nobody is listening, it simply
keeps living (design doc: rooftop-media-2026/docs/pluggyworld.md).

The physics step NEVER blocks on the network. The step hook builds a dict
and put_nowait()s it on a bounded queue; a sender thread owns the socket,
the JSON serialization, and the PNG encoding. Every failure mode degrades
to dropping messages:

  - endpoint down       -> sender retries with a delay; queue fills; frames
                           drop on the floor; physics unaffected
  - socket dies mid-run -> same: the send raises in the sender thread, the
                           in-flight message is lost, reconnect begins
  - consumer too slow   -> TCP backpressure blocks the SENDER thread, the
                           queue fills, frames drop

Sparse frames make drops dangerous: a frame is only deltas against what
was PREVIOUSLY EMITTED, so a consumer that missed anything holds stale
poses forever. Whenever continuity breaks -- a (re)connect, or any dropped
frame -- the hook resets the FrameBuilder, and the next frame is a full
keyframe. Each connection also opens with the stream header, so a late
consumer joins as if the stream had just begun.

That covers OUR socket, but in production the far end is a relay hub
(rooftop-media-2026 #22) with browsers behind it: a browser joining
mid-mission is invisible to us -- we never reconnect, so we never re-key
for it. Which is why keyframes also RECUR on a timer (FrameBuilder's
keyframe_s), marked "key": true. The hub then needs to cache only the
last keyframe plus the frames since it, and needs no knowledge of the
body census to recognize one.

The hub's ingest path is authenticated, so the publisher presents a
shared secret as `Authorization: Bearer <token>` on connect. A dev sink
(scripts/ws_sink.py) ignores it unless started with --token.

Beyond frames, two lower-frequency message types ride the same socket
(consumers must ignore types they do not know; frames are the messages
with no "type" field):

  {"type": "grid", ...}    the robot's occupancy-grid BELIEF as a base64
                           PNG, ~1 Hz. The hook snapshots the uint8 image
                           (cheap numpy); the sender encodes it.
  {"type": "event", ...}   lifecycle narration lines (_say), as they occur.
"""

import base64
import io
import json
import queue
import threading
from typing import Callable

from PIL import Image

from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.telemetry.recorder import FRAME_HZ, KEYFRAME_S, FrameBuilder

GRID_HZ = 1.0          # occupancy-grid messages per sim-second
QUEUE_MAX = 256        # ~13 s of frames at 20 Hz; beyond that, drop
RECONNECT_DELAY = 1.0  # wall-seconds between connection attempts
CONNECT_TIMEOUT = 2.0  # wall-seconds before a connection attempt fails


class WsPublisher:
  """Streams telemetry to a ws:// endpoint from the step-hook seam.

  Append `step_hook` to HubMission.step_hooks; wire `event` into the
  lifecycle's say_hooks for narration lines; call `close()` at mission end.
  `grid`, if given, is an OccupancyGrid whose to_image() is shipped at
  grid_hz. `status_fn` is merged into every frame's robot record, exactly
  as in the recorder. `token`, if given, is the ingest shared secret.
  """

  def __init__(self, model, data, endpoint: str, hz: float = FRAME_HZ,
               status_fn: Callable[[], dict] | None = None,
               model_name: str | None = None,
               grid=None, grid_hz: float = GRID_HZ,
               token: str | None = None,
               keyframe_s: float = KEYFRAME_S,
               activities=None) -> None:
    if token is not None and not token.strip():
      # An empty PLUGGYWORLD_TOKEN is the classic systemd/.env mis-deploy.
      # Falsy would silently mean "send no header at all", so the sim would
      # publish unauthenticated and blame the server's 401.
      raise ValueError("token is empty: unset it to publish unauthenticated")
    self.endpoint = endpoint
    self.grid = grid
    self._headers = {"Authorization": f"Bearer {token}"} if token else None
    self._builder = FrameBuilder(model, data, hz=hz, status_fn=status_fn,
                                 model_name=model_name, keyframe_s=keyframe_s,
                                 activities=activities)
    self.data = data
    self._grid_interval = 1.0 / grid_hz
    self._next_grid = 0.0
    self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
    # Set by the sender (on connect) or the hook (on drop); cleared by the
    # hook once it has actually re-keyed the stream.
    self._need_keyframe = threading.Event()
    self._stop = threading.Event()
    self.frames_sent = 0
    self.frames_dropped = 0
    self.events_dropped = 0
    self.connections = 0
    self.last_error: str | None = None
    self._thread = threading.Thread(target=self._send_loop, daemon=True)
    self._thread.start()

  # ---- the hook (runs inside every physics step; must never block) ---------

  def step_hook(self) -> None:
    if self._need_keyframe.is_set() and not self._queue.full():
      # Re-key only when the keyframe has room to actually go out --
      # resetting into a full queue would just drop it and lose the reset.
      self._need_keyframe.clear()
      self._builder.reset()
    frame = self._builder.build()
    if frame is not None:
      try:
        self._queue.put_nowait(("frame", frame))
      except queue.Full:
        self.frames_dropped += 1
        self._need_keyframe.set()
      if self.grid is not None and frame["t"] >= self._next_grid:
        self._next_grid = frame["t"] + self._grid_interval
        msg = {"type": "grid", "t": frame["t"], "robot": ROBOT_ROOT,
               "extent": [self.grid.x_min, self.grid.y_min,
                          self.grid.x_max, self.grid.y_max],
               "resolution": self.grid.resolution}
        try:                                  # PNG encoding is the SENDER's job
          self._queue.put_nowait(("grid", (msg, self.grid.to_image())))
        except queue.Full:
          pass                                # next second's grid supersedes it

  def event(self, t: float, line: str) -> None:
    """Queue a narration line (wire into HubLifecycle.say_hooks)."""
    try:
      self._queue.put_nowait(("event", {"type": "event", "t": round(t, 3),
                                        "robot": ROBOT_ROOT, "line": line}))
    except queue.Full:
      self.events_dropped += 1

  # ---- the sender (its own thread; owns socket, json, png) -----------------

  def _send_loop(self) -> None:
    from websockets.sync.client import connect

    while not self._stop.is_set():
      try:
        with connect(self.endpoint, open_timeout=CONNECT_TIMEOUT,
                     additional_headers=self._headers) as ws:
          self.connections += 1
          self.last_error = None       # "error since the last good connect"
          # A fresh consumer starts from nothing: drain whatever went stale
          # while disconnected, open with the header, and have the hook cut
          # a keyframe so sparse frames have a base to build on.
          self._drain()
          ws.send(json.dumps(self._builder.header(), separators=(",", ":")))
          self._need_keyframe.set()
          while not self._stop.is_set():
            try:
              kind, payload = self._queue.get(timeout=0.25)
            except queue.Empty:
              continue
            if kind == "grid":
              msg, img = payload
              buf = io.BytesIO()
              Image.fromarray(img).save(buf, format="PNG")
              msg["png"] = base64.b64encode(buf.getvalue()).decode("ascii")
              payload = msg
            ws.send(json.dumps(payload, separators=(",", ":")))
            if kind == "frame":
              self.frames_sent += 1
      except Exception as e:
        # Connection refused, reset, timeout, handshake failure -- all the
        # same story: nobody is listening right now. The sim does not care.
        # It is recorded, though, because one of those failures is a
        # REJECTED TOKEN, and a silent 1 s retry loop looks identical to a
        # server that is merely down. serve.py prints this when it never
        # connected.
        self.last_error = f"{type(e).__name__}: {e}"
        self._stop.wait(RECONNECT_DELAY)

  def _drain(self) -> None:
    try:
      while True:
        self._queue.get_nowait()
    except queue.Empty:
      pass

  def close(self, timeout: float = 5.0) -> None:
    """Stop the sender thread. Best-effort: unsent messages are dropped
    (a live stream has no obligation to flush -- that is what recordings
    are for). Idempotent."""
    self._stop.set()
    self._thread.join(timeout=timeout)
