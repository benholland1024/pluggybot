"""User control (issue #530): a person's procedure flown as one attempt in a
world of its own, streamed live and graded as the robot's are -- the stream's
shape, the host's rules (one run at a time, a refusal before anything is
built, a child that holds the session's token and never the secret), the
counter world's committed fixtures, and the one physics claim, that
watching a flight never moves it.

Every test but the last two is the stub body, a fake child or a direct
call; `test_watching_never_moves_the_flight` flies the humanoid for a
second twice.
"""

import gzip
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.evaluation import attempts as at
from pluggybot.evaluation import counter as ct
from pluggybot.evaluation.practice import job_for
from pluggybot.humanoid.body import HEAD_CAMERA
from pluggybot.mind.practice import system_sections
from pluggybot.procedure import lang
from pluggybot.session import attempt as sa
from pluggybot.session import host as sh
from pluggybot.session import wire
from pluggybot.telemetry.protocol import PROTOCOL_VERSION
from pluggybot.telemetry.recorder import FrameBuilder, TelemetryRecorder
from pluggybot.telemetry.scene import scene_dict

from test_telemetry import PROTOCOL, frames_of

SESSION, TOKEN = "s3ssion-0001", "k" * 32
INGEST = f"/api/pluggyworld/sessions/{SESSION}/ingest"
SOURCE = ('def look():\n    reach("left", 0.3, 0.2, 0.9)\n    x = read("left.x")\n'
          '    withdraw("left")\n')


def _lines(path) -> list[dict]:
  with gzip.open(path, "rt") as f:
    return [json.loads(line) for line in f]


def _keys(o) -> set:
  if isinstance(o, dict):
    return set(o) | set().union(*(_keys(v) for v in o.values()))
  if isinstance(o, list):
    return set().union(*(_keys(v) for v in o)) if o else set()
  return set()


# ---- the scene names the cameras its producer asks for -----------------------------

CAMERA_XML = """<mujoco><worldbody>
  <body name="pluggybot" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1"/>
    <body name="head" pos="0 0 .2"><geom type="sphere" size=".05"/>
      <camera name="eye" pos="0.01 0 0" euler="0 -90 0" fovy="50"/>
      <camera name="chase" mode="track" pos="0 -1 0"/>
    </body>
  </body>
</worldbody></mujoco>"""


def test_a_scene_carries_the_cameras_its_producer_names_and_only_those():
  """`cameras` is additive and asked for: a scene that names none carries
  no block (the served fixtures are unchanged), one that does carries each
  camera's body-local pose under its role, and a camera that moves on its
  own has no body-local pose to give."""
  m = mujoco.MjModel.from_xml_string(CAMERA_XML)
  assert "cameras" not in scene_dict(m, "cam")
  got = scene_dict(m, "cam", cameras={"head": "eye"})["cameras"]
  cid = m.camera("eye").id
  assert got == {"head": {"name": "eye", "body": "head", "pos": [0.01, 0.0, 0.0],
                          "quat": [round(float(v), 6) for v in m.cam_quat[cid]],
                          "fovy": 50.0}}
  with pytest.raises(ValueError, match="moves on its own"):
    scene_dict(m, "cam", cameras={"head": "chase"})


# ---- a stream opens with what its producer names ------------------------------------


def test_a_recording_opens_with_its_producers_opening_after_the_header(tmp_path):
  m = mujoco.MjModel.from_xml_string(CAMERA_XML)
  d = mujoco.MjData(m)
  opening = [{"type": "scene", "scene": {"model": "cam"}}]
  rec = TelemetryRecorder(m, d, str(tmp_path / "r.jsonl.gz"), model_name="cam",
                          opening=opening)
  rec.step_hook()
  rec.close()
  lines = _lines(tmp_path / "r.jsonl.gz")
  assert [x.get("type") for x in lines[:2]] == ["header", "scene"]
  assert lines[1]["scene"] == {"model": "cam"}
  assert FrameBuilder(m, d).opening == []


def test_a_live_consumer_is_sent_the_opening_on_every_connect():
  """The publisher's opening goes right after the header on EVERY connect:
  a session's viewer that joined late, or a stream that reconnected, has
  no other way to learn the set-out's world."""
  from websockets.sync.server import serve
  from pluggybot.telemetry.publisher import WsPublisher
  m = mujoco.MjModel.from_xml_string(CAMERA_XML)
  d = mujoco.MjData(m)
  firsts: list[list[str]] = []
  done = threading.Event()

  def handler(ws):
    got = [json.loads(ws.recv())["type"] for _ in range(2)]
    firsts.append(got)
    if len(firsts) == 2:
      done.set()
    ws.close()                                       # make it come back
  with serve(handler, "localhost", 0) as server:
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pub = WsPublisher(m, d, f"ws://localhost:{port}", model_name="cam",
                      opening=[wire.scene_message({"model": "cam"})])
    try:
      assert done.wait(10.0), firsts
    finally:
      pub.close()
      server.shutdown()
  assert firsts == [["header", "scene"], ["header", "scene"]]


def test_a_child_starts_once_its_stream_is_open_and_nothing_it_says_is_drained(monkeypatch):
  """`connections` counts a stream once its drain is behind it: a child that
  starts flying the moment it reads one queues nothing the drain can take
  (#530's review: counted before the drain, a GIL switch between the two
  dropped a run's first step). A slow drain stands for the switch."""
  from websockets.sync.server import serve
  from pluggybot.telemetry.publisher import WsPublisher
  m = mujoco.MjModel.from_xml_string(CAMERA_XML)
  d = mujoco.MjData(m)
  got, heard = [], threading.Event()

  def handler(ws):
    for raw in ws:
      got.append(json.loads(raw).get("line"))
      if "first" in got:
        heard.set()
  drain = WsPublisher._drain

  def slow_drain(self):
    time.sleep(0.3)
    drain(self)
  monkeypatch.setattr(WsPublisher, "_drain", slow_drain)
  with serve(handler, "localhost", 0) as server:
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pub = WsPublisher(m, d, f"ws://localhost:{port}", model_name="cam")
    try:
      assert sa._wait_connected(pub, 5.0)
      pub.message({"type": "event", "line": "first"})
      assert heard.wait(5.0), got
    finally:
      pub.close()
      server.shutdown()


def test_a_last_word_waits_for_room_and_one_dropped_all_the_same_is_said(tmp_path, capsys):
  """The verdict is the one message a session's stream must not lose: it
  waits for room in the queue (`FLUSH_S`), and one dropped all the same, or
  a stream that never emptied, is said where the host's log sees it -- the
  host's `ended` carries the verdict besides."""
  waits = []

  class Choked:
    def step_hook(self):
      pass

    def message(self, msg, wait=0.0):
      waits.append((msg.get("type"), wait))
      return msg.get("type") != "attempt"

    def flush(self, timeout):
      return False

    def close(self):
      pass
  sa.fly(sa.task_of(7), lambda m, d, **s: Choked(), proc=lang.compile_procedure(SOURCE, sa.facts()),
         build=robot_world, hold_s=0.2)
  assert ("attempt", sa.FLUSH_S) in waits and {w for t, w in waits if t != "attempt"} == {0.0}
  err = capsys.readouterr().err
  assert "verdict was dropped" in err and "had not left" in err


def test_a_publishers_message_may_wait_for_room_and_says_whether_it_went():
  from pluggybot.telemetry import publisher as tp
  m = mujoco.MjModel.from_xml_string(CAMERA_XML)
  pub = tp.WsPublisher(m, mujoco.MjData(m), "ws://127.0.0.1:9", model_name="cam")
  try:
    for _ in range(tp.QUEUE_MAX):
      pub._queue.put_nowait(("event", {}))
    assert pub.message({"type": "x"}) is False
    threading.Timer(0.1, pub._queue.get_nowait).start()
    assert pub.message({"type": "x"}, wait=2.0) is True
  finally:
    pub.close()


# ---- an attempt's stream: its scene, its steps, its senses, its verdict ------------


def robot_world(task: at.Task) -> at.World:
  """`test_lid`'s chest on a bare floor and the stub body, with a body for
  the stream's census to call the robot (a stream names its robot's)."""
  from pluggybot.activity import chest as ch
  from pluggybot.body import StubBody
  from test_lid import BARE
  spec = mujoco.MjSpec.from_string(BARE)
  robot = spec.worldbody.add_body(name="pluggybot", pos=[-1.0, 0.0, 0.3])
  robot.add_freejoint()
  robot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
  ch.attach_chest(spec, ch.draw(task.chest), (0.0, 0.0), 0.0, tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  chest = ch.Chest(model, data, ch.draw(task.chest))
  body.step_hooks.append(lambda: chest.sense(model, data))
  return at.World(task=task, model=model, data=data, body=body, chest=chest,
                  start=(0.0, 0.0, 0.0))


def _fly(source=SOURCE, watch=True, **kw):
  path = kw.pop("path")
  proc = lang.compile_procedure(source, sa.facts())
  flight = sa.fly(at.Task(chest=7, body="humanoid", mount="counter"),
                  lambda m, d, **s: TelemetryRecorder(m, d, str(path), **s),
                  proc=proc, watch=watch, build=robot_world, hold_s=0.2, **kw)
  return flight, _lines(path)


def test_a_session_streams_its_scene_its_steps_its_senses_and_its_verdict(tmp_path):
  flight, lines = _fly(path=tmp_path / "s.jsonl.gz")
  header = lines[0]
  assert header["model"] == sa.WORLD == "home_humanoid_counter"
  assert header["activities"] == ["chest"]
  assert lines[1] == wire.scene_message(flight.scene) and lines[1]["scene"]["model"] == sa.WORLD
  # each verb as it began, at the line it stands on
  steps = [x for x in lines if x.get("type") == "step"]
  assert [(s["n"], s["line"], s["verb"]) for s in steps] == [(1, 2, "reach"), (2, 4, "withdraw")]
  assert steps[0]["args"] == {"hand": "left", "x": 0.3, "y": 0.2, "z": 0.9}
  assert any("PROCEDURE look step 1 (line 2)" in x["line"] for x in lines
             if x.get("type") == "event")
  # the robot's senses: every sensor a procedure may read, None before a
  # verb's report exists -- never a made-up 0
  reads = [x for x in lines if x.get("type") == "readouts"]
  assert reads and set(reads[0]["values"]) == set(sa.facts().sensors)
  assert reads[0]["values"]["spot.x"] is None and reads[0]["values"]["pull.force"] is None
  # ...and what its depth camera sees, a picture a second on the grid's path
  depth = [x for x in lines if x.get("type") == "depth"]
  assert depth and all(d["png"] and (d["width"], d["height"]) == (wire.DEPTH_W, wire.DEPTH_H)
                       for d in depth)
  # the end: the verdict and the report the robot's mind is shown of a try,
  # and never a secret metric (the catch is the lid's)
  end = lines[-1]
  assert end["type"] == "attempt" and end == flight.message
  assert end["verdict"] == flight.row["verdict"] and not end["verdict"]["ok"]
  assert "secret" not in _keys(end) and "catchN" not in _keys(end)
  assert flight.row["secret"]["catchN"] is not None
  assert end["report"].startswith("TRY 1") and "THE GRADE: failed" in end["report"]
  assert "Write the procedure" not in end["report"]
  assert [s["verb"] for s in end["run"]["steps"]] == ["reach", "withdraw"]
  assert end["run"]["locals"]["x"] == pytest.approx(0.3) and "source" not in end["run"]


def test_a_report_is_the_minds_less_what_it_is_told_to_do_next():
  """A person reads the report the robot's mind is shown of a try, the same
  words but its last line (`run_report`)."""
  from pluggybot.mind import practice as mp
  ran = {"program": "a", "total": 1, "steps": [{"i": 0, "verb": "wait", "line": 2, "ok": True}],
         "seconds": 1.0, "locals": {}}
  verdict = {"ok": False, "reason": "the lid never opened"}
  full = mp.report_text(1, ran, verdict, 1.0, False, 2)
  assert full == mp.run_report(1, ran, verdict, 1.0, False) + \
    "\n\nWrite the procedure for try 2 (2 left)."


def test_an_attempt_that_raises_says_so_on_the_stream_and_has_no_verdict(tmp_path):
  def broken(task):
    w = robot_world(task)
    w.body.step_hooks.append(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    return w
  path = tmp_path / "x.jsonl.gz"
  with pytest.raises(RuntimeError, match="boom"):
    sa.fly(at.Task(chest=7, body="humanoid", mount="counter"),
           lambda m, d, **s: TelemetryRecorder(m, d, str(path), **s),
           proc=lang.compile_procedure(SOURCE, sa.facts()), build=broken, hold_s=0.2)
  end = _lines(path)[-1]
  assert end["type"] == "attempt" and "verdict" not in end and "boom" in end["error"]


def test_a_try_its_time_cut_says_what_it_ran_and_where_the_time_ran_out(tmp_path):
  """A procedure the try's limit cuts mid-verb ran: its verbs, the one in
  flight and its report go out -- never "no procedure that compiled" (the
  record was kept only when the run returned, and a cut run never does) --
  and the hold that follows reads as the hold."""
  flight, lines = _fly('def w():\n    x = 1\n    wait(0.2)\n    wait(5)\n',
                       path=tmp_path / "c.jsonl.gz", budget_s=1.0)
  end = lines[-1]
  assert end["cut"] and end["steps"] == 2 and flight.row["steps"] == 2
  assert [s["verb"] for s in end["run"]["steps"]] == ["wait"]
  assert end["run"]["inFlight"] == {"line": 4, "verb": "wait"}
  assert end["run"]["locals"] == {"x": 1.0} and end["run"]["seconds"] == pytest.approx(1.0, abs=0.05)
  assert "no procedure that compiled" not in end["report"]
  assert "line 3 wait: ok" in end["report"] and "x = 1" in end["report"]
  assert "was at line 4, wait, when the time ran out" in end["report"]
  states = [f.get("robots", {}).get("pluggybot", {}).get("state") for f in frames_of(lines)]
  assert states.index("HOLD") > states.index("ATTEMPT")


def test_a_flight_that_cannot_start_says_why_and_leaves_nothing_open(tmp_path, monkeypatch):
  """A pace it cannot keep is refused before anything is built, and a
  setup that raises is that error said, its sink and world closed -- never
  `list.remove`'s, with both left open (#530's review: a pace of -1)."""
  closed = []

  def build(task):
    w = robot_world(task)
    close = w.close
    w.close = lambda: (closed.append("world"), close())
    return w

  class Sink(TelemetryRecorder):
    def close(self):
      closed.append("sink")
      super().close()
  proc = lang.compile_procedure(SOURCE, sa.facts())

  def fly(**kw):
    return sa.fly(sa.task_of(7), lambda m, d, **s: Sink(m, d, str(tmp_path / "p.jsonl.gz"), **s),
                  proc=proc, build=build, **kw)
  with pytest.raises(ValueError, match="pace"):
    fly(pace=-1.0)
  assert closed == []

  class Stuck:
    def __init__(self, *a, **kw):
      raise RuntimeError("no pacer")
  monkeypatch.setattr("pluggybot.telemetry.pacer.RealTimePacer", Stuck)
  with pytest.raises(RuntimeError, match="no pacer"):
    fly(pace=1.0)
  assert closed == ["sink", "world"]


# ---- the host: one run at a time, refused before anything is built ---------------


class Child:
  """A run's child as a test drives it: running until it ends (`end`), its
  word in the host's file as the real one writes it (`attempt.say`)."""

  def __init__(self, job, token):
    self.job, self.token, self.code = job, token, None
    self.terminated = self.killed = False

  def end(self, code, said=None):
    if said is not None:
      sa.say(self.job, said)
    self.code = code

  def poll(self):
    return self.code

  def wait(self, timeout=None):
    if self.code is None and self.terminated:
      self.code = -15
    return self.code

  def terminate(self):
    self.terminated = True

  def kill(self):
    self.killed = True


@pytest.fixture
def results(tmp_path, monkeypatch):
  """The host's files for its children's words, in the test's own folder."""
  made = []

  def result_file():
    made.append(str(tmp_path / f"run{len(made)}.json"))
    return made[-1]
  monkeypatch.setattr(sh, "result_file", result_file)
  return made


def _host(spawn=None, **kw):
  children, sent, now = [], [], [0.0]

  def spawn_fake(job, token):
    children.append(Child(job, token))
    return children[-1]
  host = sh.Host("ws://web:3000/api/pluggyworld/sessions/host", record="/rec",
                 spawn=spawn or spawn_fake, clock=lambda: now[0], **kw)
  return host, children, sent, now


def _run(attempt=1, source=SOURCE, **kw):
  return {"type": "run", "session": SESSION, "attempt": attempt, "chest": 3, "source": source,
          "ingest": INGEST, "token": TOKEN} | kw


def _cancel(attempt=1, session=SESSION):
  return {"type": "cancel", "session": session, "attempt": attempt}


SAID = {"ok": False, "reason": "the lid never opened"}


def test_a_procedure_is_refused_with_every_reason_and_nothing_is_built(results):
  host, children, sent, _ = _host()
  host.handle(_run(source='def a():\n    nonsense()\n    x = 1 % 2\n'), sent.append)
  assert not children and host.run is None and not results
  (no,) = sent
  assert no["type"] == "refused" and len(no["reasons"]) >= 2
  assert any("nonsense" in r for r in no["reasons"]) and any("%" in r for r in no["reasons"])


def test_one_run_at_a_time_and_its_child_flies_on_the_sessions_own_path(results):
  host, children, sent, _ = _host()
  host.handle(_run(), sent.append)
  host.handle(_run(attempt=2), sent.append)
  assert [m["type"] for m in sent] == ["started", "busy"] and len(children) == 1
  job = children[0].job
  assert job["endpoint"] == f"ws://web:3000{INGEST}" and job["record"] == "/rec"
  assert (job["chest"], job["attempt"], job["source"]) == (3, 1, SOURCE)
  assert job["result"] == results[0]
  assert children[0].token == TOKEN and "token" not in job
  children[0].end(0, SAID)
  host.poll(sent.append)
  assert sent[-1] == wire.ended_message(SESSION, 1, "graded", verdict=SAID) and host.run is None
  assert sent[-1]["verdict"] == SAID
  host.handle(_run(attempt=2), sent.append)             # the seat is free again
  assert sent[-1]["type"] == "started" and len(children) == 2


def test_a_child_holds_the_sessions_token_and_never_a_secret(monkeypatch):
  monkeypatch.setenv(sh.SECRET_ENV, "the-shared-secret")
  for name in ("HF_TOKEN", "PLUGGY_OVERSEER_KEY", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD"):
    monkeypatch.setenv(name, "a credential")
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  env = sh.child_env(TOKEN)
  assert env[sa.TOKEN_ENV] == TOKEN and env["MUJOCO_GL"] == "osmesa"
  assert "a credential" not in env.values() and "the-shared-secret" not in env.values()


def test_a_run_that_ends_says_how(results):
  host, children, sent, now = _host()
  host.handle(_run(), sent.append)
  children[-1].end(1, {"why": "Unstable: reset"})
  host.poll(sent.append)
  assert sent[-1] == wire.ended_message(SESSION, 1, "failed", "Unstable: reset")
  host.handle(_run(attempt=2), sent.append)
  children[-1].end(1)                                   # it wrote nothing
  host.poll(sent.append)
  assert sent[-1] == wire.ended_message(SESSION, 2, "failed", "exit 1")
  host.handle(_run(attempt=3), sent.append)
  host.handle(_cancel(3, session="another-000001"), sent.append)
  assert not children[-1].terminated, "a cancel for another session stops nothing"
  host.handle(_cancel(3), sent.append)
  host.poll(sent.append)
  assert children[-1].terminated and sent[-1] == wire.ended_message(SESSION, 3, "cancelled")
  host.handle(_run(attempt=4), sent.append)
  now[0] += sh.CHILD_S + 1
  host.poll(sent.append)
  assert children[-1].terminated and sent[-1]["why"] == "out of time"
  assert not [f for f in results if Path(f).exists()], "every run's file went with it"


def test_a_cancel_names_its_attempt_and_one_for_the_run_before_stops_nothing(results):
  """The site's cancel for attempt 1, late, must not stop attempt 2 (#530's
  review: a cancel matched the session alone)."""
  host, children, sent, _ = _host()
  host.handle(_run(attempt=1), sent.append)
  children[-1].end(0, SAID)
  host.poll(sent.append)
  host.handle(_run(attempt=2), sent.append)
  host.handle(_cancel(1), sent.append)
  host.handle({"type": "cancel", "session": SESSION}, sent.append)
  assert not children[-1].terminated
  host.handle(_cancel(2), sent.append)
  assert children[-1].terminated


def test_a_run_its_child_graded_is_graded_whatever_ended_the_process(results):
  """Graded is the child's word, written the moment it graded: a cancel
  that lands after a clean exit, or one that kills the child as its stream
  closes, is still a graded run with its verdict (#530's review: both read
  `cancelled`)."""
  host, children, sent, _ = _host()
  host.handle(_run(attempt=1), sent.append)
  children[-1].end(0, SAID)                             # exited, not yet polled
  host.handle(_cancel(1), sent.append)
  host.poll(sent.append)
  assert sent[-1] == wire.ended_message(SESSION, 1, "graded", verdict=SAID)
  host.handle(_run(attempt=2), sent.append)
  sa.say(children[-1].job, SAID)                        # graded, still closing
  host.handle(_cancel(2), sent.append)
  host.poll(sent.append)
  assert children[-1].terminated
  assert sent[-1] == wire.ended_message(SESSION, 2, "graded", verdict=SAID)


def test_a_run_whose_process_would_not_start_failed_and_the_host_goes_on(results):
  """A fork that fails is the run's end, said -- never taken for a dropped
  connection (#530's review: an OSError from Popen tore the socket down)."""
  def broken(job, token):
    raise OSError(12, "Cannot allocate memory")
  host, _, sent, _ = _host(spawn=broken)
  host.handle(_run(), sent.append)
  (ended,) = sent
  assert ended["type"] == "ended" and ended["why"] == "failed"
  assert "did not start" in ended["detail"] and "Cannot allocate memory" in ended["detail"]
  assert host.run is None and not Path(results[0]).exists()


def test_an_ending_waits_for_a_connection_and_follows_the_next_hello(results):
  """A run that ends while the site is away is still said: watched with no
  connection (its time bound too), named by the next hello, its ending
  after it -- and one whose send failed is kept for the next."""
  host, children, sent, now = _host()
  host.handle(_run(), sent.append)
  sent.clear()
  now[0] += sh.CHILD_S + 1
  host.poll(None)                                      # no connection
  assert children[-1].terminated and host.running() == {"session": SESSION, "attempt": 1}

  def dropped(msg):
    raise ConnectionError("the site went away")
  with pytest.raises(ConnectionError):
    host.poll(dropped)
  assert sh.hello({"commit": "x"}, host.running())["running"] == {"session": SESSION,
                                                                    "attempt": 1}
  host.poll(sent.append)
  assert sent == [wire.ended_message(SESSION, 1, "out of time")] and host.running() is None


def test_the_host_reconnects_after_a_refused_handshake_and_says_what_ended_meanwhile(results):
  """The site answering 503 (it is restarting) or 401 is no connection now,
  never the host's end (#530's review: `InvalidStatus` is no OSError) --
  and while it is away the host watches its child, so a run past its time
  is stopped then and its ending follows the next hello."""
  from http import HTTPStatus
  from websockets.sync.server import serve
  said, tries, done = [], [0], threading.Event()

  def process_request(connection, request):
    tries[0] += 1
    if tries[0] == 1:
      return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "restarting\n")
    return None

  def handler(ws):
    said.extend(json.loads(ws.recv()) for _ in range(2))
    done.set()
    ws.recv()                                          # until the host goes
  host, children, _, now = _host()
  host.handle(_run(), lambda msg: None)
  now[0] += sh.CHILD_S + 1                             # its time is up meanwhile
  polled = []
  poll = host.poll
  host.poll = lambda send=None: (polled.append(send is None), poll(send))[1]
  stop = threading.Event()
  with serve(handler, "localhost", 0, process_request=process_request) as server:
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    thread = threading.Thread(target=sh.serve, args=(f"ws://localhost:{port}", TOKEN, None),
                              kwargs={"host": host, "stop": stop, "reconnect_s": 0.05},
                              daemon=True)
    thread.start()
    try:
      assert done.wait(10.0), "the host never came back"
    finally:
      stop.set()
      thread.join(5.0)
      server.shutdown()
  assert tries[0] >= 2 and not thread.is_alive()
  assert said[0]["type"] == "hello" and said[0]["running"] == {"session": SESSION, "attempt": 1}
  assert said[1] == wire.ended_message(SESSION, 1, "out of time")
  assert children[0].terminated and polled[0] is True, "watched while there was no connection"


def test_a_run_is_the_sessions_own_whole():
  """A session id and a token are read whole: a trailing newline or a NUL
  is no session's (#530's review: `$` let the one through, and the other
  made `Popen` raise in the host)."""
  for bad in ({"session": SESSION + "\n", "ingest": INGEST.replace(SESSION, SESSION + "\n")},
              {"token": "k" * 20 + "\x00" + "k" * 11}, {"token": "k" * 15},
              {"token": "k" * 129}):
    with pytest.raises(wire.BadRun):
      wire.parse_run(_run() | bad, sh.CHESTS)
  assert wire.parse_run(_run(), sh.CHESTS)["token"] == TOKEN


def test_a_run_that_is_not_the_sessions_own_is_failed_not_flown(results):
  """A malformed `run` is the site's fault, not the person's: an `ended`
  that failed, saying why -- never a refusal of their procedure."""
  host, children, sent, _ = _host()
  for bad in (_run(ingest="/api/pluggyworld/ingest"), _run(chest=at.HELD_OUT_FROM),
              _run(token="short"), _run(session="../x"), _run(attempt=0)):
    host.handle(bad, sent.append)
  assert not children and {m["type"] for m in sent} == {"ended"}
  assert {m["why"] for m in sent} == {"failed"}
  assert "ingest" in sent[0]["detail"] and "chest" in sent[1]["detail"]


def test_an_ending_carries_a_verdict_if_and_only_if_it_was_graded():
  assert wire.ended_message(SESSION, 1, "graded", verdict=SAID)["verdict"] == SAID
  assert "verdict" not in wire.ended_message(SESSION, 1, "failed", "x")
  with pytest.raises(ValueError):
    wire.ended_message(SESSION, 1, "graded")
  with pytest.raises(ValueError):
    wire.ended_message(SESSION, 1, "cancelled", verdict=SAID)


def test_hello_says_what_the_robot_is_told():
  """The page's reference is the robot's own text (`mind.practice`): the
  language and the body, byte for byte, and the job as the robot is told
  it -- so the two never drift apart."""
  said = sh.hello({"commit": "abc1234"}, None)
  sections = dict(system_sections(sa.facts()))
  assert [v["heading"] for v in said["vocabulary"]] == ["the language", "your body"]
  assert [v["text"] for v in said["vocabulary"]] == [sections["the language"],
                                                     sections["your body"]]
  assert said["job"] == job_for(sa.task_of(0))
  assert said["task"]["limitS"] == pytest.approx(sa.task_of(0).limit_s, abs=0.01)
  assert said["task"]["chests"] == [0, at.PUBLIC.stop - 1]
  assert said["protocolVersion"] == PROTOCOL_VERSION and said["running"] is None


def test_a_refused_job_builds_no_world(monkeypatch):
  def built(*a, **kw):
    raise AssertionError("a world was built for a procedure the language refused")
  monkeypatch.setattr(at, "set_out", built)
  code, said = sa.run_job({"source": "def a(:\n", "chest": 0, "attempt": 1,
                           "endpoint": "ws://nowhere", "session": SESSION}, TOKEN)
  assert code == 2 and said["refused"]
  # ...and a flight builds through the module's set-out as it is NOW, so the
  # patch above is what a refusal that slipped through would have met
  with pytest.raises(AssertionError, match="a world was built"):
    sa.fly(sa.task_of(0), None, proc=lang.compile_procedure(SOURCE, sa.facts()))


def test_a_graded_run_is_kept_and_said_before_its_last_message_goes(tmp_path, monkeypatch):
  """The child's word to its host and the record's row are written the
  moment the attempt is graded, BEFORE the stream's last message: a cancel
  or a kill that lands as the stream closes keeps both (#530's review: the
  row was appended after the stream had closed)."""
  seen = []

  class Sink(TelemetryRecorder):
    def emit(self, message):
      if message.get("type") == "attempt":
        seen.append((Path(job["result"]).exists(), (tmp_path / "rec" / "rows.jsonl").exists()))
      super().emit(message)
  job = {"session": SESSION, "attempt": 2, "chest": 7, "source": SOURCE,
         "endpoint": "ws://nowhere", "record": str(tmp_path / "rec"),
         "result": str(tmp_path / "said.json"), "pace": None}
  real_fly = sa.fly

  def fly(task, open_sink, **kw):
    return real_fly(task, lambda m, d, **s: Sink(m, d, str(tmp_path / "s.jsonl.gz"), **s),
                    build=robot_world, hold_s=0.2, **kw)
  monkeypatch.setattr(sa, "fly", fly)
  code, said = sa.run_job(job, TOKEN)
  assert code == 0 and seen == [(True, True)]
  assert json.loads(Path(job["result"]).read_text()) == said == {
    "ok": False, "reason": said["reason"]}
  (row,) = [json.loads(x) for x in (tmp_path / "rec" / "rows.jsonl").read_text().splitlines()]
  assert row["attempt"] == 2 and row["session"] == SESSION and row["source"] == SOURCE


# ---- the counter world's committed fixtures (`scripts/attempts.py --record`) -------

SCENE = PROTOCOL / f"scene.{ct.WORLD}.json"
RECORDING = PROTOCOL / f"telemetry.{ct.WORLD}.jsonl.gz"


def test_the_counter_scene_is_its_set_outs_world():
  """The demo chest's set-out at the counter, the humanoid at its start and
  its head camera named: stale on any change to the house, the body, the
  counter or the chest's drawing."""
  scene = json.loads(SCENE.read_text())
  model = ct.model_of(ct.set_out(0))
  assert scene == ct.scene(model, {"head": HEAD_CAMERA}), (
    "stale fixture: MUJOCO_GL=egl uv run python scripts/attempts.py --body humanoid "
    f"--record protocol/telemetry.{ct.WORLD}.jsonl.gz")
  owners = {b["name"]: b["robot"] for b in scene["bodies"]}
  assert owners["left_hand"] == "pluggybot" and owners["counter"] is None
  assert scene["cameras"]["head"]["body"] in owners


def test_the_counter_recording_is_one_attempt_of_the_reference():
  """The site's first proof (#530, stage 1): the reference's attempt at
  chest 0, as a session streams one -- the scene after the header, the
  humanoid's every body, its senses, and one verdict, which passed."""
  lines = _lines(RECORDING)
  header, frames = lines[0], frames_of(lines)
  assert header["model"] == ct.WORLD and header["protocolVersion"] == PROTOCOL_VERSION
  assert lines[1] == wire.scene_message(json.loads(SCENE.read_text()))
  assert {"left_hand", "right_jaw_a", "nav_cam_body"} <= set(header["robots"]["pluggybot"])
  assert "chest_lid_body" in header["world"] and header["activities"] == ["chest"]
  assert {"ATTEMPT", "HOLD"} <= {f["robots"]["pluggybot"].get("state") for f in frames}
  opened = [f["activities"]["chest"].get("opened") for f in frames
            if "chest" in f.get("activities", {})]
  assert True in opened
  reads = [x for x in lines if x.get("type") == "readouts"]
  assert reads and set(reads[-1]["values"]) == set(sa.facts().sensors)
  assert [x for x in lines if x.get("type") == "depth" and x.get("png")]
  (end,) = [x for x in lines if x.get("type") == "attempt"]
  assert end is lines[-1] and end["verdict"]["ok"] and end["report"] is None
  assert "secret" not in _keys(lines) and "catchN" not in _keys(lines)


# ---- the one physics claim: watching never moves the flight --------------------------


def test_watching_never_moves_the_flight(tmp_path):
  """The robot's senses a page shows are read on streams of the watcher's
  own (`Body.hand_watched`, `Body.depth_watched`): a humanoid's attempt
  streamed into a real recording, its senses watched or not, ends in the
  same state bit for bit -- and its own depth camera, opened at the set-out
  as a `measure` opens it, has drawn the same noise. Read off the flight's
  own (`hand_at`, its depth camera), the two parted (#530's PR)."""
  proc = lang.compile_procedure('def w():\n    wait(1.2)\n', sa.facts())
  ends = []
  for watch in (False, True):
    kept = {}

    def build(task):
      w = at.set_out(task)
      # the flight's own depth camera, as a `measure` opens it (a cheap one)
      w.body.run(w.body.depth_cloud(16, 9, 0.17, frames=1, apart=0.0))
      close = w.close

      def closed():
        kept["qpos"], kept["t"] = w.data.qpos.copy(), float(w.data.time)
        kept["depth"] = w.body._depth[1].rng.bit_generator.state
        close()
      w.close = closed
      return w
    path = tmp_path / f"watched-{watch}.jsonl.gz"
    sa.fly(sa.task_of(0), lambda m, d, **s: TelemetryRecorder(m, d, str(path), **s), proc=proc,
           watch=watch, build=build, hold_s=0.3)
    kept["depth pictures"] = sum(x.get("type") == "depth" for x in _lines(path))
    ends.append(kept)
  assert ends[0]["depth pictures"] == 0 and ends[1]["depth pictures"] >= 2
  assert ends[0]["t"] == ends[1]["t"]
  assert np.array_equal(ends[0]["qpos"], ends[1]["qpos"])
  assert ends[0]["depth"] == ends[1]["depth"]
