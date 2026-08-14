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

import json
import threading
import time

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
  """

  def __init__(self, port: int = 0, token: str | None = None) -> None:
    from websockets.sync.server import serve
    self.sessions: list[list] = []
    self.auth_headers: list[str | None] = []
    self.refused = 0
    self._token = token
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
    try:
      for raw in ws:
        msgs.append(json.loads(raw))
    except Exception:
      pass                    # connection torn down mid-recv: session over

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
    assert wait_for(lambda: sink.sessions[0]
                    and sum(1 for m in sink.sessions[0]
                            if "type" not in m) >= 30)
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
