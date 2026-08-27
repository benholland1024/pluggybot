"""Webserver v1 guards: real-time pacer + live WebSocket publisher.

The live stream's failure modes are all about who waits on whom, so these
tests run a real in-process WebSocket sink (websockets' sync server on an
ephemeral port) rather than mocking the socket layer. The three contracts
under guard:

  - the pacer holds sim time to wall time at the requested rate;
  - a dead endpoint costs the physics loop nothing (frames drop, steps
    continue at full speed);
  - a reconnected consumer gets header + KEYFRAME. Sparse frames only
    carry what moved since last emitted, so without the re-key a consumer
    that joins mid-stream would hold rest poses forever for any body that
    stopped moving before it connected -- verified to fail with the
    builder reset removed.
  - keyframes RECUR, so a consumer that joins without us noticing -- a
    browser behind the website's relay hub, which never touches our
    socket -- still converges. The keyframe_s=0 arm of that test is the
    pre-0.2.0 behaviour, and it fails the same assertion.
  - the ingest secret rides on the connection, and a rejected one is
    REPORTED rather than retried in silence.
"""

import importlib.util
import json
import math
import sys
import threading
import time
import types
from pathlib import Path

import mujoco
import pytest

from test_telemetry import MINI_XML

from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.telemetry.pacer import RealTimePacer
from pluggybot.telemetry.publisher import WsPublisher
from pluggybot.telemetry.recorder import FrameBuilder


@pytest.fixture()
def mini_model():
  return mujoco.MjModel.from_xml_string(MINI_XML)


def wait_for(cond, timeout: float = 5.0) -> bool:
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    if cond():
      return True
    time.sleep(0.02)
  return False


class Sink:
  """In-process ws sink: every connection's parsed messages, in order.

  `token`, if given, makes it stand in for the website's authenticated
  ingest path: a connection without the matching bearer is refused at the
  handshake, exactly as the hub will refuse it.

  It also drives the OTHER direction (issue #16): `send()` pushes a message
  down the live connection, which is what makes it a fake SERVER rather than
  a sink -- and is the acceptance criterion "test with a fake server driving
  both directions".
  """

  def __init__(self, port: int = 0, token: str | None = None) -> None:
    from websockets.sync.server import serve
    self.sessions: list[list] = []
    self.auth_headers: list[str | None] = []
    self.refused = 0
    self._token = token
    self._live: list = []                   # connections currently open
    self._server = serve(self._handle, "localhost", port,
                         process_request=self._check_auth)
    self.port = self._server.socket.getsockname()[1]
    self._thread = threading.Thread(target=self._server.serve_forever,
                                    daemon=True)
    self._thread.start()

  @property
  def endpoint(self) -> str:
    return f"ws://localhost:{self.port}"

  def _check_auth(self, connection, request):
    self.auth_headers.append(request.headers.get("Authorization"))
    if self._token is None:
      return None
    if request.headers.get("Authorization") != f"Bearer {self._token}":
      self.refused += 1
      return connection.respond(401, "unauthorized\n")
    return None

  def _handle(self, ws) -> None:
    msgs: list = []
    self.sessions.append(msgs)
    self._live.append(ws)
    try:
      for raw in ws:
        msgs.append(json.loads(raw))
    except Exception:
      pass                    # connection torn down mid-recv: session over
    finally:
      if ws in self._live:
        self._live.remove(ws)

  def send(self, msg) -> bool:
    """Push one message down to the producer. False if nobody is connected.

    Deliberately reports rather than raises: "no sim is connected" is a
    first-class state the website has to handle (rooftop-media-2026 #29's
    "the sim-absent path is explicit, not silent"), and a fake server that
    hid it would let a test pass that the real one could not.
    """
    raw = msg if isinstance(msg, str) else json.dumps(msg)
    for ws in list(self._live):
      try:
        ws.send(raw)
        return True
      except Exception:
        continue
    return False

  def wait_live(self, timeout: float = 5.0) -> bool:
    return wait_for(lambda: bool(self._live), timeout)

  def stop(self) -> None:
    self._server.shutdown()
    self._thread.join(timeout=5.0)


def step_seconds(model, data, seconds: float, *hooks) -> None:
  for _ in range(round(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    for hook in hooks:
      hook()


# ---- pacer -----------------------------------------------------------------

def test_pacer_holds_requested_rate(mini_model):
  """2 sim-seconds at 4x must take ~0.5 wall-seconds -- and measurably
  longer than the same loop free-running (i.e. the pacer actually paces)."""
  data = mujoco.MjData(mini_model)
  t0 = time.monotonic()
  step_seconds(mini_model, data, 2.0)
  free_wall = time.monotonic() - t0

  data = mujoco.MjData(mini_model)
  pacer = RealTimePacer(data, rate=4.0)
  t0 = time.monotonic()
  step_seconds(mini_model, data, 2.0, pacer.step_hook)
  wall = time.monotonic() - t0

  assert wall == pytest.approx(0.5, abs=0.15)
  assert wall > free_wall * 2, "pacing must dominate this tiny model's step cost"
  s = pacer.stats()
  assert abs(s["drift_s"]) < 0.15
  assert s["multiple"] == pytest.approx(4.0, rel=0.3)


def test_pacer_rejects_nonpositive_rate(mini_model):
  with pytest.raises(ValueError):
    RealTimePacer(mujoco.MjData(mini_model), rate=0.0)


# ---- frame builder re-key --------------------------------------------------

def test_builder_reset_forces_keyframe(mini_model):
  """After reset() the next frame re-ships every dynamic body, moved or
  not -- the primitive the publisher's reconnect recovery is built on."""
  mini_model.opt.gravity[:] = 0            # nothing will ever move
  data = mujoco.MjData(mini_model)
  mujoco.mj_forward(mini_model, data)
  builder = FrameBuilder(mini_model, data)
  first = builder.build()
  assert set(first["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
  assert first.get("key") is True
  step_seconds(mini_model, data, 0.1)
  sparse = builder.build()
  assert "bodies" not in sparse["robots"]["pluggybot"]
  assert "key" not in sparse, "a sparse frame must never claim to be a keyframe"
  builder.reset()
  step_seconds(mini_model, data, 0.1)
  rekeyed = builder.build()
  assert set(rekeyed["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
  assert set(rekeyed["world"]) == {"ball"}
  # complete is not enough: the hub recognizes a cache boundary by the
  # MARKER, so a recovery keyframe that forgets to say so is invisible to it
  assert rekeyed.get("key") is True


def replay(frames) -> dict:
  """The world a consumer holds after applying `frames` in order.

  Exactly what a browser does with sparse frames: overwrite the bodies a
  frame mentions, hold the last value seen for every body it does not.
  """
  world: dict = {}
  for f in frames:
    world.update(f["robots"]["pluggybot"].get("bodies", {}))
    world.update(f.get("world", {}))
  return world


@pytest.mark.parametrize("keyframe_s, converges", [(1.0, True), (0.0, False)])
def test_recurring_keyframes_let_an_unnoticed_joiner_converge(
    mini_model, keyframe_s, converges):
  """A consumer that starts reading mid-stream must agree with one that
  read from the top -- within keyframe_s.

  This is the browser-behind-the-relay-hub case (rooftop-media-2026 #22):
  it joins the HUB, never our socket, so no reconnect fires and nothing
  re-keys on its behalf. Sparse frames are deltas against the whole
  history, so every body that settled before it joined is simply missing
  from its world, forever. The keyframe_s=0 arm is the pre-0.2.0
  publisher and fails on exactly that: the ball has stopped moving by the
  time our joiner arrives, so it never hears about the ball at all.
  """
  data = mujoco.MjData(mini_model)
  builder = FrameBuilder(mini_model, data, keyframe_s=keyframe_s)
  frames = []
  for _ in range(round(6.0 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    frame = builder.build()
    if frame is not None:
      frames.append(frame)

  join_t = 4.0                            # everything has long since settled
  joined = [f for f in frames if f["t"] >= join_t]
  # what the joiner holds one keyframe-interval in, vs. the full stream
  # truncated to that same instant -- so any still-moving body is compared
  # at one moment, not two.
  window = [f for f in joined if f["t"] <= join_t + 1.0]
  seen_from_top = replay([f for f in frames if f["t"] <= window[-1]["t"]])
  assert (replay(window) == seen_from_top) is converges
  if not converges:
    assert "ball" not in replay(window), \
      "without recurring keyframes the joiner should be MISSING the settled ball"


def test_keyframes_are_marked_and_complete(mini_model):
  """Every frame carrying "key" re-ships every dynamic body, and they
  arrive on the requested cadence. The marker is the contract: it is how
  the relay hub recognizes a cache boundary without knowing the census."""
  data = mujoco.MjData(mini_model)
  builder = FrameBuilder(mini_model, data, keyframe_s=1.0)
  frames = []
  for _ in range(round(5.0 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    frame = builder.build()
    if frame is not None:
      frames.append(frame)

  keys = [f for f in frames if f.get("key")]
  assert frames[0]["key"] is True, "the first frame is always a keyframe"
  assert len(keys) == 5, "one keyframe per sim-second over 5 s"
  for f in keys:
    assert set(f["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
    assert set(f["world"]) == {"ball"}
  spacing = [b["t"] - a["t"] for a, b in zip(keys, keys[1:])]
  assert all(1.0 <= s < 1.0 + 2 / 20.0 for s in spacing), spacing
  assert builder.keyframes == len(keys)
  # and the cost stays marginal: keyframes are the rare frame, not the norm
  assert len(keys) < 0.1 * len(frames)


def test_bad_config_is_refused_at_construction(mini_model):
  """Two mis-deploys that would otherwise fail silently and far away: a
  negative interval keys every frame while advertising a negative cache
  depth, and an EMPTY token (a set-but-blank PLUGGYWORLD_TOKEN) is falsy,
  so the bearer header would simply be omitted and the server's 401 would
  look like a wrong secret rather than a missing one."""
  data = mujoco.MjData(mini_model)
  with pytest.raises(ValueError):
    FrameBuilder(mini_model, data, keyframe_s=-1.0)
  with pytest.raises(ValueError):
    WsPublisher(mini_model, data, "ws://localhost:1", token="")
  with pytest.raises(ValueError):
    WsPublisher(mini_model, data, "ws://localhost:1", token="   ")


# ---- publisher -------------------------------------------------------------

def test_publisher_streams_protocol_messages(mini_model):
  """One live session: header first, then frames identical in shape to the
  recorder's (keyframe -> sparse, monotonic t, status merged), grid
  messages at ~1 Hz with a decodable PNG, and event lines."""
  data = mujoco.MjData(mini_model)
  grid = OccupancyGrid(x_min=-1, y_min=-1, x_max=1, y_max=1, resolution=0.1)
  sink = Sink()
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini",
                    status_fn=lambda: {"state": "EXPLORE", "status": "hi",
                                       "battery": {"frac": 0.5, "watts": 8.5,
                                                   "charging": False}},
                    grid=grid)
  try:
    assert wait_for(lambda: sink.sessions), "publisher never connected"
    time.sleep(0.2)                        # let the post-header re-key land
    step_seconds(mini_model, data, 2.0, pub.step_hook)
    pub.event(float(data.time), "the errand is done")
    # Wait for the EVENT as well as the frames. `close()` drops whatever is
    # still queued -- deliberately, since a live stream has no obligation to
    # flush -- and a frame count says nothing about a message queued after
    # those frames. Waiting only on frames made this test fail about 1 run
    # in 5 (measured: 4/20), with the event assertion below reading an empty
    # list. Wait for the thing you are about to assert on.
    assert wait_for(lambda: sink.sessions[0]
                    and sum(1 for m in sink.sessions[0]
                            if "type" not in m) >= 30
                    and any(m.get("type") == "event"
                            for m in sink.sessions[0]))
  finally:
    pub.close()
    sink.stop()

  msgs = sink.sessions[0]
  assert msgs[0]["type"] == "header"
  assert msgs[0]["robots"] == {"pluggybot": ["pluggybot"]}
  frames = [m for m in msgs if "type" not in m]
  times = [f["t"] for f in frames]
  assert times == sorted(times)
  # the first frame of the session is a keyframe: every dynamic body
  assert set(frames[0]["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
  assert set(frames[0]["world"]) == {"ball"}
  assert frames[0]["robots"]["pluggybot"]["state"] == "EXPLORE"

  grids = [m for m in msgs if m.get("type") == "grid"]
  assert 1 <= len(grids) <= 4              # ~1 Hz over 2 sim-seconds
  import base64
  import io

  from PIL import Image
  img = Image.open(io.BytesIO(base64.b64decode(grids[0]["png"])))
  assert img.size == (20, 20)              # 2 m x 2 m at 0.1 m cells
  assert grids[0]["extent"] == [-1, -1, 1, 1]

  events = [m for m in msgs if m.get("type") == "event"]
  assert events and events[-1]["line"] == "the errand is done"


def test_publisher_keyframes_reach_the_wire(mini_model):
  """The cadence must survive the trip through WsPublisher, not just live
  inside FrameBuilder: the header advertises it and marked, complete
  keyframes actually arrive. Drop the keyframe_s pass-through in the
  publisher's constructor and this is the test that notices."""
  data = mujoco.MjData(mini_model)
  sink = Sink()
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini",
                    keyframe_s=0.5)
  try:
    assert wait_for(lambda: sink.sessions), "publisher never connected"
    time.sleep(0.2)
    step_seconds(mini_model, data, 3.0, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent >= 55)
  finally:
    pub.close()
    sink.stop()

  msgs = sink.sessions[0]
  assert msgs[0]["keyframeS"] == 0.5, "the header must advertise the cadence"
  keys = [m for m in msgs if "type" not in m and m.get("key")]
  assert len(keys) >= 5, f"expected ~6 keyframes over 3 s at 0.5 s, got {len(keys)}"
  for f in keys:
    assert set(f["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
    assert set(f["world"]) == {"ball"}
  assert pub.frames_dropped == 0, "a live sink should not have lost frames"


def test_dead_endpoint_does_not_touch_physics(mini_model):
  """Nobody listening: steps run at full speed, the queue fills, frames
  drop, and nothing raises. 15 sim-seconds of hook calls must cost wall
  time like physics alone, not like network timeouts."""
  data = mujoco.MjData(mini_model)
  pub = WsPublisher(mini_model, data, "ws://localhost:1", model_name="mini")
  try:
    t0 = time.monotonic()
    step_seconds(mini_model, data, 15.0, pub.step_hook)
    wall = time.monotonic() - t0
  finally:
    pub.close()
  assert wall < 5.0, "a dead endpoint must never slow the step loop"
  assert pub.frames_dropped > 0, "the bounded queue must drop, not grow"
  assert pub.frames_sent == 0


def test_publisher_presents_the_ingest_token(mini_model):
  """The website's /ingest path is authenticated by a shared secret, so
  the publisher must offer it at the handshake -- and must offer nothing
  when it has none (a dev sink stays open)."""
  data = mujoco.MjData(mini_model)
  sink = Sink(token="s3cret")
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini",
                    token="s3cret")
  try:
    assert wait_for(lambda: sink.sessions), "authenticated connect failed"
    assert sink.auth_headers[0] == "Bearer s3cret"
    assert sink.refused == 0
  finally:
    pub.close()

  open_sink = Sink()
  pub2 = WsPublisher(mini_model, data, open_sink.endpoint, model_name="mini")
  try:
    assert wait_for(lambda: open_sink.sessions)
    assert open_sink.auth_headers[0] is None
  finally:
    pub2.close()
    open_sink.stop()
    sink.stop()


def test_rejected_token_is_reported_not_swallowed(mini_model):
  """A wrong secret retries exactly like a server that is down -- so the
  sim must keep stepping (it is still just a missing consumer) AND say
  which it was, or a mis-deployed secret is a silent black hole."""
  data = mujoco.MjData(mini_model)
  sink = Sink(token="right")
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini",
                    token="wrong")
  try:
    t0 = time.monotonic()
    step_seconds(mini_model, data, 3.0, pub.step_hook)
    wall = time.monotonic() - t0
    assert wait_for(lambda: sink.refused > 0), "the sink never refused it"
  finally:
    pub.close()
    sink.stop()
  assert wall < 3.0, "a refused handshake must not slow the step loop"
  assert pub.connections == 0 and pub.frames_sent == 0
  assert not sink.sessions, "a refused publisher must not stream"
  assert pub.last_error is not None and "401" in pub.last_error, pub.last_error


def test_last_error_clears_on_a_good_connect(mini_model):
  """last_error means "since the last good connect", not "ever". A sim
  started before its server is up records failures and then succeeds; if
  the field never cleared, a health check reading it would call a healthy
  publisher broken forever."""
  data = mujoco.MjData(mini_model)
  probe = Sink()                        # take a free port, then release it
  port = probe.port
  probe.stop()
  pub = WsPublisher(mini_model, data, f"ws://localhost:{port}",
                    model_name="mini")
  try:
    assert wait_for(lambda: pub.last_error is not None), \
      "connecting to a closed port must record something"
    sink = Sink(port=port)              # the server finally comes up
    try:
      assert wait_for(lambda: sink.sessions, timeout=10.0)
      step_seconds(mini_model, data, 0.5, pub.step_hook)
      assert wait_for(lambda: pub.frames_sent > 0)
      assert pub.last_error is None, "a good connect must clear the error"
    finally:
      sink.stop()
  finally:
    pub.close()


def test_reconnect_resends_header_and_keyframe(mini_model):
  """Kill the sink mid-stream, bring it back on the same port: the sim
  must keep stepping throughout, and the new session must open with the
  header and a full keyframe even though nothing moved while it was gone
  (remove FrameBuilder.reset from the hook and this fails: the second
  session's frames never mention any body again)."""
  mini_model.opt.gravity[:] = 0
  data = mujoco.MjData(mini_model)
  sink = Sink()
  port = sink.port
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini")
  try:
    assert wait_for(lambda: sink.sessions)
    time.sleep(0.2)
    step_seconds(mini_model, data, 1.0, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent >= 10)
    sink.stop()                            # the socket dies mid-mission

    step_seconds(mini_model, data, 1.0, pub.step_hook)   # sim carries on

    sink2 = Sink(port=port)                # the consumer comes back
    try:
      assert wait_for(lambda: sink2.sessions, timeout=10.0), \
        "publisher never reconnected"
      time.sleep(0.2)
      step_seconds(mini_model, data, 1.0, pub.step_hook)
      assert wait_for(lambda: sink2.sessions[0]
                      and sum(1 for m in sink2.sessions[0]
                              if "type" not in m) >= 5)
    finally:
      sink2.stop()
  finally:
    pub.close()
    sink.stop()

  msgs = sink2.sessions[0]
  assert msgs[0]["type"] == "header", "every session must open with the header"
  framed = [m for m in msgs if "type" not in m and
            "bodies" in m["robots"]["pluggybot"]]
  assert framed, "no keyframe after reconnect: sparse frames have no base"
  assert set(framed[0]["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
  assert set(framed[0]["world"]) == {"ball"}
  assert framed[0].get("key") is True, \
    "the recovery keyframe must be MARKED, or the relay hub cannot see it"


def test_every_connection_is_caught_up_on_the_ink(mini_model):
  """0.5.0: a viewer joining a mission in progress must be told what is
  already on the boards.

  This is the half of rooftop-media-2026 #28 no keyframe can cover. Ink is
  not a body: the `boards` block re-ships the counters on every keyframe,
  but a `draw` event happens once and is gone. So a browser opening the page
  after the pen has moved on gets a board that reports strokes and paints
  none -- and a RECONNECT is the same problem, which is why the snapshot
  rides every session rather than only the first.
  """
  from pluggybot.hub.boards import BoardBook, BoardRecord

  mini_model.opt.gravity[:] = 0
  data = mujoco.MjData(mini_model)
  book = BoardBook([BoardRecord(name="whiteboard_a", reach=(0.11, 0.2))],
                   clock=lambda: "2026-08-16T00:00:00")
  book.stroke("whiteboard_a", "house", [(0.0, 0.0), (0.02, 0.0), (0.02, 0.02)])
  sink = Sink()
  port = sink.port
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini",
                    boards=book)
  book.on_event.append(pub.message)
  try:
    assert wait_for(lambda: sink.sessions)
    time.sleep(0.2)
    step_seconds(mini_model, data, 0.5, pub.step_hook)
    assert wait_for(lambda: any(m.get("type") == "board_snapshot"
                                for m in sink.sessions[0]))
    sink.stop()

    # A second stroke lands while nobody is listening, then a viewer returns.
    book.stroke("whiteboard_a", "house", [(0.03, 0.0), (0.05, 0.02)])
    sink2 = Sink(port=port)
    try:
      assert wait_for(lambda: sink2.sessions, timeout=10.0)
      time.sleep(0.2)
      step_seconds(mini_model, data, 0.5, pub.step_hook)
      assert wait_for(lambda: any(m.get("type") == "board_snapshot"
                                  for m in sink2.sessions[0]))
    finally:
      sink2.stop()
  finally:
    pub.close()
    sink.stop()

  first = [m for m in sink.sessions[0] if m.get("type") == "board_snapshot"]
  again = [m for m in sink2.sessions[0] if m.get("type") == "board_snapshot"]
  assert len(first[0]["strokes"]) == 1
  # The second session is caught up on BOTH strokes, including the one drawn
  # while it was away -- which no `draw` event will ever mention again.
  assert len(again[0]["strokes"]) == 2
  assert again[0]["strokes"][1]["points"][0] == [0.03, 0.0]


def test_a_publisher_without_boards_sends_no_snapshots(mini_model):
  """room_hub has no whiteboards, and a world with nothing to say about ink
  must not open every connection with an empty message."""
  data = mujoco.MjData(mini_model)
  sink = Sink()
  pub = WsPublisher(mini_model, data, sink.endpoint, model_name="mini")
  try:
    assert wait_for(lambda: sink.sessions)
    time.sleep(0.2)
    step_seconds(mini_model, data, 0.5, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent >= 5)
  finally:
    pub.close()
    sink.stop()
  assert not [m for m in sink.sessions[0] if m.get("type") == "board_snapshot"]


# ---- scripts/serve.py: the world the live demo actually serves --------------
# serve.py is what visitors see, and every world-dependent constant it gets
# wrong fails SILENTLY -- a short explore budget just stops mapping early, a
# stale use_at just drives at a wall, a stale model_name just makes the site
# render the other house. So the guard is the whole wiring, checked by
# running main() with the heavy collaborators faked out: real argparse, real
# world_config, real MjModel load, no physics.

_SERVE = Path(__file__).parent.parent / "scripts" / "serve.py"


# ---- the inbound direction (issue #16) ---------------------------------------


def _publishing(mini_model, sink, **kw):
  """A publisher wired to an inbox, connected to `sink`. Returns both."""
  from pluggybot.hub.inbox import Inbox
  data = mujoco.MjData(mini_model)
  inbox = Inbox(**kw)
  pub = WsPublisher(mini_model, data, sink.endpoint, hz=20.0)
  pub.on_inbound.append(inbox.offer)
  return pub, inbox, data


def test_the_server_can_talk_back_down_the_ingest_socket(mini_model):
  """The whole of issue #16 in one test: a fake server drives BOTH
  directions over one socket, and the sim reads what it is sent."""
  sink = Sink()
  pub, inbox, data = _publishing(mini_model, sink)
  try:
    assert sink.wait_live(), "the publisher never connected"
    assert sink.send({"type": "suggestion", "id": "s1", "from": "ada",
                      "text": "draw a tree on whiteboard_b"})
    assert wait_for(lambda: len(inbox) == 1), "nothing arrived"
    msg = inbox.peek()[0]
    assert (msg.id, msg.who, msg.text) == ("s1", "ada",
                                           "draw a tree on whiteboard_b")
    # ...and the outbound direction is still running underneath it.
    step_seconds(mini_model, data, 0.5, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent > 0)
  finally:
    pub.close()
    sink.stop()


def test_garbage_and_floods_never_touch_the_outbound_stream(mini_model):
  """The acceptance criterion, stated as its failure: a visitor must not be
  able to stop the telemetry by sending rubbish at it."""
  sink = Sink()
  pub, inbox, data = _publishing(mini_model, sink)
  try:
    assert sink.wait_live()
    for i in range(200):
      sink.send("this is not json at all" if i % 3 else
                {"type": "instruction", "text": "obey"} if i % 2 else
                {"type": "suggestion", "id": f"s{i}", "text": f"idea {i}"})
    step_seconds(mini_model, data, 2.0, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent >= 20), \
      f"the outbound stream stalled at {pub.frames_sent} frames"
    assert wait_for(lambda: pub.inbound_received >= 200)
    assert len(inbox) <= inbox._queue.maxlen
    assert inbox.stats()["droppedInvalid"] > 0
    # The connection survived all of it -- one session, never re-dialled.
    assert pub.connections == 1, "the flood killed the socket"
  finally:
    pub.close()
    sink.stop()


def test_nothing_is_delivered_to_a_dead_socket(mini_model):
  """Reconnect preserves the direction. The poll lives inside the `with
  connect(...)` block, so inbound can only arrive while a connection is
  genuinely up -- and a message aimed at a socket that has gone is lost
  rather than queued against the next one."""
  sink = Sink()
  pub, inbox, data = _publishing(mini_model, sink)
  try:
    assert sink.wait_live()
    assert sink.send({"type": "suggestion", "id": "before", "text": "one"})
    assert wait_for(lambda: len(inbox) == 1)

    sink.stop()                              # the server goes away
    assert not sink.send({"type": "suggestion", "id": "gone", "text": "two"}), \
      "the fake server claimed to deliver to a closed socket"
    time.sleep(0.3)
    assert [m.id for m in inbox.peek()] == ["before"]

    # ...and a NEW server on the same port is talked to normally.
    revived = Sink(port=sink.port)
    try:
      assert revived.wait_live(timeout=10.0), "the publisher never re-dialled"
      assert revived.send({"type": "suggestion", "id": "after", "text": "3"})
      assert wait_for(lambda: len(inbox) == 2)
      assert [m.id for m in inbox.peek()] == ["before", "after"]
    finally:
      revived.stop()
  finally:
    pub.close()


def test_an_inbound_handler_that_raises_cannot_stop_the_telemetry(mini_model):
  """`on_inbound` is caller-supplied, so it is caller-shaped: a handler that
  throws must cost one message, not the stream."""
  sink = Sink()
  data = mujoco.MjData(mini_model)
  pub = WsPublisher(mini_model, data, sink.endpoint, hz=20.0)
  pub.on_inbound.append(lambda raw: (_ for _ in ()).throw(RuntimeError("no")))
  try:
    assert sink.wait_live()
    sink.send({"type": "suggestion", "id": "s1", "text": "boom"})
    assert wait_for(lambda: pub.inbound_received == 1)
    step_seconds(mini_model, data, 1.0, pub.step_hook)
    assert wait_for(lambda: pub.frames_sent >= 10)
    assert pub.connections == 1
  finally:
    pub.close()
    sink.stop()


def _load_serve():
  spec = importlib.util.spec_from_file_location("serve", _SERVE)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class _FakeLife:
  """Stands in for HubLifecycle: records how it was built and run."""

  def __init__(self, model, data, **kw):
    self.init_kwargs = kw
    self.mission = types.SimpleNamespace(step_hooks=[], grid=None)
    self.say_hooks: list = []
    # Where a `visitor_reply` goes on its way to the socket (issue #16).
    self.visitor_hooks: list = []
    self.run_args: tuple = ()
    self.run_kwargs: dict = {}

  def telemetry_status(self) -> dict:
    return {}

  def run(self, *a, **kw):
    self.run_args, self.run_kwargs = a, kw
    return {"state": "DONE", "swaps_done": 2, "charge_cycles": 1,
            "module_stowed": True, "sim_time": 100.0,
            "errands": [], "boards": {}, "verdicts": [], "points": 0,
            "earned": 0}


class _FakePublisher:
  def __init__(self, model, data, endpoint, **kw):
    self.init_kwargs = kw
    self.messages: list = []
    self.frames_sent = self.frames_dropped = self.connections = 0
    self.last_error = None

  def step_hook(self) -> None:
    pass

  def event(self, t, msg) -> None:
    pass

  def message(self, msg) -> None:
    # `draw` / `board_cleared` reach the browser through here (issue #12)
    self.messages.append(msg)

  def close(self) -> None:
    pass


def _serve_wiring(monkeypatch, argv):
  """Run serve.main() with fakes; return (fake life, fake publisher)."""
  serve = _load_serve()
  built: dict = {}

  def life_factory(model, data, **kw):
    built["life"] = _FakeLife(model, data, **kw)
    built["model"] = model
    return built["life"]

  def pub_factory(model, data, endpoint, **kw):
    built["pub"] = _FakePublisher(model, data, endpoint, **kw)
    return built["pub"]

  monkeypatch.setattr(serve, "HubLifecycle", life_factory)
  monkeypatch.setattr(serve, "WsPublisher", pub_factory)
  monkeypatch.setattr(sys, "argv", ["serve.py", *argv])
  serve.main()
  return built["life"], built["pub"], built["model"]


@pytest.mark.parametrize("world", ["room_hub", "home"])
def test_serve_takes_every_world_constant_from_world_config(monkeypatch, world):
  """--world must thread the WHOLE config through, not just the model.

  Both traps this test exists for are silent: serve.py used to pass no
  explore_budget at all (home wants 240 s, the default is 90 s, so the map
  simply stops filling), and it used to inherit room_hub's use_at of
  (-1.2, 2.5) -- which in the home world is inside wall_divider_0, so the
  errand drives at a wall instead of into the living room.
  """
  from pluggybot.hub.lifecycle import world_config
  cfg = world_config(world)

  life, pub, model = _serve_wiring(monkeypatch, ["--world", world, "--free-run"])

  assert life.init_kwargs["battery_wh"] == cfg["battery_wh"]
  assert life.init_kwargs["low_battery_wh"] == cfg["low_battery_wh"]
  assert life.init_kwargs["grid_bounds"] == cfg["grid_bounds"]
  assert life.init_kwargs["rack"] == cfg["rack"]
  assert life.run_args[0] == cfg["start"]
  assert life.run_kwargs["use_at"] == cfg["use_at"]
  assert life.run_kwargs["explore_budget"] == cfg["explore_budget"], \
    "serve.py must pass the world's explore budget, not run()'s default"
  # the header field the website selects its scene off
  assert pub.init_kwargs["model_name"] == cfg["model_name"]
  # ...and the model really is that world, not just a matching label
  assert model.body(f"{'wall_divider_0' if world == 'home' else 'rack'}").id > 0


def test_serve_defaults_to_room_hub_unchanged(monkeypatch):
  """The regression arm: no --world must behave exactly as webserver v1 did."""
  life, pub, _ = _serve_wiring(monkeypatch, ["--free-run"])
  assert pub.init_kwargs["model_name"] == "room_hub"
  assert life.run_args[0] == (0.5, 3.0, math.pi / 2)
  assert life.run_kwargs["use_at"] == (-1.2, 2.5)
  assert life.run_kwargs["explore_budget"] == 90.0
  assert life.init_kwargs["rack"] is None


def test_serve_wires_the_drawing_errand_and_its_boards(monkeypatch, tmp_path):
  """`--errand draw` has to reach three places or it half-applies silently
  (issue #12): the lifecycle needs the errand QUEUE, the lifecycle and the
  publisher both need the same BOOK -- one so the boards block is in the
  frames, the other so `draw` events reach the browser at all -- and the
  book needs the state file, or a restart forgets the drawing.

  Missing any one of them looks like success from the terminal: the robot
  still fetches the pen and still draws. The website just shows a blank
  wall, which is not something this repo's tests can see from here.
  """
  state = tmp_path / "boards.json"
  life, pub, _ = _serve_wiring(
    monkeypatch, ["--world", "home", "--free-run", "--errand", "draw",
                  "--boards", str(state)])
  errands = life.init_kwargs["errands"]
  assert [e.name for e in errands] == ["draw:whiteboard_a"]
  assert errands[0].module == "module_pen" and errands[0].use is not None
  book = life.init_kwargs["boards"]
  assert book is pub.init_kwargs["boards"], \
    "the publisher and the lifecycle must share ONE book"
  assert book.path == state
  assert pub.message in book.on_event, "strokes never reach the socket"


def test_serve_wires_the_ledger_to_the_lifecycle_and_the_socket(monkeypatch,
                                                                tmp_path):
  """`--ledger` has the same three-places problem as `--boards` (issue #14).

  The lifecycle needs the ledger to bank verdicts into, the publisher needs
  the SAME one so the balance rides in the frames, and it needs the event
  hook or no `earned` message ever reaches the site. Any one missing looks
  like success from the terminal -- the robot still does the work and the
  log still prints the score -- while the site's scoreboard sits at zero.
  """
  state = tmp_path / "ledger.json"
  life, pub, _ = _serve_wiring(
    monkeypatch, ["--free-run", "--ledger", str(state)])
  ledger = life.init_kwargs["ledger"]
  assert ledger is pub.init_kwargs["ledger"], \
    "the publisher and the lifecycle must share ONE ledger"
  assert ledger.path == state, "points would not survive the restart"
  assert pub.message in ledger.on_event, "awards never reach the socket"
  # ...and a run without the flag still scores, it just does not remember:
  # a ledger is not optional the way a board book is, because every world
  # charges.
  bare, _, _ = _serve_wiring(monkeypatch, ["--free-run"])
  assert bare.init_kwargs["ledger"].path is None


def test_serve_without_boards_is_the_pre_0_4_0_wiring(monkeypatch):
  """room_hub has no whiteboards, so there is no book -- and every board hook
  has to tolerate that rather than being wired to None."""
  life, pub, _ = _serve_wiring(monkeypatch, ["--free-run"])
  assert life.init_kwargs["boards"] is None
  assert pub.init_kwargs["boards"] is None


def test_serve_recorder_labels_the_world_it_recorded(monkeypatch, tmp_path):
  """--record writes the replay artifact; a room_hub label on a home-world
  recording would pose the wrong scene on the website's replay path."""
  serve = _load_serve()
  seen: dict = {}
  real_recorder = serve.TelemetryRecorder

  def recorder_factory(model, data, path, **kw):
    seen.update(kw)
    return real_recorder(model, data, path, **kw)

  monkeypatch.setattr(serve, "HubLifecycle",
                      lambda model, data, **kw: _FakeLife(model, data, **kw))
  monkeypatch.setattr(serve, "WsPublisher",
                      lambda model, data, endpoint, **kw:
                      _FakePublisher(model, data, endpoint, **kw))
  monkeypatch.setattr(serve, "TelemetryRecorder", recorder_factory)
  monkeypatch.setattr(sys, "argv",
                      ["serve.py", "--world", "home", "--free-run",
                       "--record", str(tmp_path / "out.jsonl.gz")])
  serve.main()
  assert seen["model_name"] == "home_world"


def test_serve_names_the_robot_in_both_artifacts(monkeypatch, tmp_path):
  """--robot-name is this instance's IDENTITY on the wire (issue #39), and
  `serve.py --record` writes a recording of the SAME run it streams -- so
  the name must reach the publisher AND the recorder, or a replay of a
  stream would disagree with the stream about who was flying."""
  serve = _load_serve()
  seen: dict = {}
  pub_kw: dict = {}
  real_recorder = serve.TelemetryRecorder

  def recorder_factory(model, data, path, **kw):
    seen.update(kw)
    return real_recorder(model, data, path, **kw)

  def pub_factory(model, data, endpoint, **kw):
    pub_kw.update(kw)
    return _FakePublisher(model, data, endpoint, **kw)

  monkeypatch.setattr(serve, "HubLifecycle",
                      lambda model, data, **kw: _FakeLife(model, data, **kw))
  monkeypatch.setattr(serve, "WsPublisher", pub_factory)
  monkeypatch.setattr(serve, "TelemetryRecorder", recorder_factory)
  monkeypatch.setattr(sys, "argv",
                      ["serve.py", "--free-run", "--robot-name", "Luca",
                       "--record", str(tmp_path / "out.jsonl.gz")])
  serve.main()
  assert pub_kw["robot_name"] == "Luca"
  assert seen["robot_name"] == "Luca"
  # ...and without the flag the wiring passes None through untouched:
  # resolution (env, then 'Pluggy') is the FrameBuilder's alone, so both
  # sinks of one run cannot resolve differently.
  _, pub, _ = _serve_wiring(monkeypatch, ["--free-run"])
  assert pub.init_kwargs["robot_name"] is None
