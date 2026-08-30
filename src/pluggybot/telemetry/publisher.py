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

INBOUND (issue #16). The socket is bidirectional as of protocol 0.7.0: the
server may send visitor messages and ratings back down it. Two
deliberate choices about how:

  IT IS THE SAME THREAD, not a reader thread. `recv(timeout=0)` is a
  non-blocking poll, so the sender loop checks for inbound between sends and
  the connection is only ever touched by one thread. No lock, no second
  failure mode, and no way for a reader to outlive the socket it was reading.

  NOTHING IS DELIVERED TO A DEAD SOCKET. The poll lives inside the `with
  connect(...)` block, so a message can only arrive while a connection is
  genuinely up; a reconnect starts a fresh poll against the fresh socket, and
  anything in flight when the old one broke is simply gone. `on_inbound` is
  called on this thread and must never block or touch the sim -- it hands the
  message to `hub.inbox.Inbox`, which is a bounded deque and nothing else.
"""

import json
import queue
import threading
from typing import Callable

from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.telemetry.recorder import (FRAME_HZ, GRID_HZ, KEYFRAME_S,
                                          FrameBuilder, GridSampler,
                                          encode_grid_png)

QUEUE_MAX = 256        # ~13 s of frames at 20 Hz; beyond that, drop
RECONNECT_DELAY = 1.0  # wall-seconds between connection attempts
CONNECT_TIMEOUT = 2.0  # wall-seconds before a connection attempt fails
INBOUND_PER_PASS = 8   # inbound messages taken per send-loop pass (issue #16)


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
               activities=None, boards=None, screens=None,
               ledger=None, tasks=None, accepts=(), goals: str = "",
               thoughts=None, spend=None, mode=None, metabolism=None,
               steering: bool = False,
               robot_name: str | None = None) -> None:
    if token is not None and not token.strip():
      # An empty PLUGGYWORLD_TOKEN is the classic systemd/.env mis-deploy.
      # Falsy would silently mean "send no header at all", so the sim would
      # publish unauthenticated and blame the server's 401.
      raise ValueError("token is empty: unset it to publish unauthenticated")
    self.endpoint = endpoint
    # dedupe=False: the hub caches the most recent grid per robot for late
    # joiners, so the last message sent is what a new browser is handed --
    # and a live stream that fell silent because the map stopped changing
    # would look exactly like one whose grid path had broken.
    self._grid = GridSampler(grid, hz=grid_hz, dedupe=False)
    self._headers = {"Authorization": f"Bearer {token}"} if token else None
    self._builder = FrameBuilder(model, data, hz=hz, status_fn=status_fn,
                                 model_name=model_name, keyframe_s=keyframe_s,
                                 activities=activities, boards=boards,
                                 screens=screens, ledger=ledger,
                                 tasks=tasks, accepts=accepts, goals=goals,
                                 thoughts=thoughts, spend=spend,
                                 mode=mode, metabolism=metabolism,
                                 steering=steering, robot_name=robot_name)
    self.data = data
    self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
    # Set by the sender (on connect) or the hook (on drop); cleared by the
    # hook once it has actually re-keyed the stream.
    self._need_keyframe = threading.Event()
    # Set by the sender on CONNECT ONLY, and deliberately not on a dropped
    # frame: a re-key is how the stream repairs itself under load, and
    # answering congestion by enqueuing every stroke on every board would
    # make the queue that just overflowed overflow harder. The boards are
    # read on the physics thread, which is the thread that owns them.
    self._need_boards = threading.Event()
    # ...and on the same terms, what the robot is FOR (0.8.0). Set on connect
    # only: goals do not change during a run, but a NEW consumer has never
    # seen them and no keyframe carries them.
    self._need_goals = threading.Event()
    # ...and the memory documents (0.11.0, issue #38). Set on connect for the
    # goals reason, and UNLIKE goals these change during a run -- but a
    # change is published by `ThoughtFiles.on_event` as it happens, so this
    # flag stays what it says: catching a new consumer up, not polling.
    self._need_thoughts = threading.Event()
    self.boards = boards
    self._stop = threading.Event()
    self.frames_sent = 0
    self.frames_dropped = 0
    self.events_dropped = 0
    self.connections = 0
    self.last_error: str | None = None
    # Called with each raw inbound message, on the SENDER thread (issue #16).
    # Wire `hub.inbox.Inbox.offer` into it. Must not block: whatever it does
    # happens between two outbound sends.
    self.on_inbound: list[Callable[[object], None]] = []
    self.inbound_received = 0
    self._thread = threading.Thread(target=self._send_loop, daemon=True)
    self._thread.start()

  # ---- the hook (runs inside every physics step; must never block) ---------

  def step_hook(self) -> None:
    if self._need_goals.is_set() and not self._queue.full():
      self._need_goals.clear()
      goals = self._builder.goals_message(float(self.data.time))
      if goals is not None:
        self.message(goals)
    if self._need_thoughts.is_set() and not self._queue.full():
      self._need_thoughts.clear()
      for thought in self._builder.thought_messages(float(self.data.time)):
        self.message(thought)
    if self._need_boards.is_set() and not self._queue.full():
      self._need_boards.clear()
      if self.boards is not None:
        for snapshot in self.boards.snapshots(t=float(self.data.time)):
          self.message(snapshot)
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
      sample = self._grid.due(frame["t"])
      if sample is not None:
        try:                                  # PNG encoding is the SENDER's job
          self._queue.put_nowait(("grid", sample))
        except queue.Full:
          pass                                # next second's grid supersedes it

  def event(self, t: float, line: str) -> None:
    """Queue a narration line (wire into HubLifecycle.say_hooks)."""
    self.message({"type": "event", "t": round(t, 3),
                  "robot": ROBOT_ROOT, "line": line})

  def message(self, msg: dict) -> None:
    """Queue any typed low-frequency message (wire into BoardBook.on_event
    for `draw` / `board_cleared`, and Ledger.on_event for `earned`).

    Dropped rather than blocking, like everything else here -- but a dropped
    STROKE is not like a dropped frame: there is no later message that
    supersedes it, so the browser's canvas is missing that line until the
    board is next erased. That is the accepted cost of never blocking the
    physics on a socket, and it is why recordings, not the live stream, are
    the lossless artifact. `events_dropped` counts it.
    """
    try:
      self._queue.put_nowait(("event", dict(msg)))
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
          # ...and on what the robot is for (0.8.0), which is likewise in no
          # keyframe: a viewer who joins an hour in still has to be told.
          self._need_goals.set()
          # ...and on what it is working from (0.11.0, issue #38), which no
          # keyframe carries either.
          self._need_thoughts.set()
          # ...and catch the new consumer up on the ink already on the walls
          # (0.5.0). No keyframe carries it: a stroke is an event that
          # happened once, so without this a viewer who joined late watches
          # a robot admire a blank board.
          self._need_boards.set()
          while not self._stop.is_set():
            self._poll_inbound(ws)
            try:
              kind, payload = self._queue.get(timeout=0.25)
            except queue.Empty:
              continue
            if kind == "grid":
              msg, img = payload
              msg["png"] = encode_grid_png(img)
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

  def _poll_inbound(self, ws) -> None:
    """Take whatever the server has sent, without waiting for it (issue #16).

    `timeout=0` is "is there one already?", so this costs a queue check per
    loop and never delays a frame. Bounded per pass: a server that floods must
    not be able to keep the sender from ever sending again -- the rest waits
    for the next pass, or is dropped by the inbox's own bound. That is the
    acceptance criterion "flooding input is dropped without affecting the
    outbound stream", and this loop is the half of it the inbox cannot do.

    ConnectionClosed is left to propagate: it means the socket is gone, which
    is the sender loop's business and its reconnect.
    """
    from websockets.exceptions import ConnectionClosed
    for _ in range(INBOUND_PER_PASS):
      try:
        raw = ws.recv(timeout=0)
      except ConnectionClosed:
        raise
      except Exception:                     # noqa: BLE001
        # TimeoutError -- nothing waiting, which is the common case. Anything
        # else on a live socket is a malformed frame, and dropping it is the
        # same answer.
        return
      self.inbound_received += 1
      for hook in self.on_inbound:
        try:
          hook(raw)
        except Exception:                   # noqa: BLE001
          # A handler that raises must never take the OUTBOUND stream down
          # with it -- that would let a visitor message stop the telemetry.
          pass

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
