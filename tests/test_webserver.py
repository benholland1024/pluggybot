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
  """In-process ws sink: every connection's parsed messages, in order."""

  def __init__(self, port: int = 0) -> None:
    from websockets.sync.server import serve
    self.sessions: list[list] = []
    self._server = serve(self._handle, "localhost", port)
    self.port = self._server.socket.getsockname()[1]
    self._thread = threading.Thread(target=self._server.serve_forever,
                                    daemon=True)
    self._thread.start()

  @property
  def endpoint(self) -> str:
    return f"ws://localhost:{self.port}"

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
  step_seconds(mini_model, data, 0.1)
  assert "bodies" not in builder.build()["robots"]["pluggybot"]
  builder.reset()
  step_seconds(mini_model, data, 0.1)
  rekeyed = builder.build()
  assert set(rekeyed["robots"]["pluggybot"]["bodies"]) == {"pluggybot"}
  assert set(rekeyed["world"]) == {"ball"}


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
