"""The eye (issue #275): the robot looks at the world as the site draws it.

Every rule pinned without a renderer, a network or a model, in
milliseconds: the door (an `image` inbound kind with its own byte cap,
checked for being a JPEG), the eye (one open request, a stale picture
dropped), the mind (`look` on a mind, and on nothing else), the orders
(never a standing order, never a map
row), the delivery (a fake renderer answering on the socket mid-wait; the
deadline; the picture as an IMAGE PART of the next turn and never text;
shown once, kept across a fallback; the run cap), the build identity, and
the rule's text. The whole mission is never flown.
"""

import base64
import inspect
import json
import math

import mujoco
import pytest

from pluggybot.evaluation.identity import build_identity
from pluggybot.lifecycle import (
  LOOK_SLICE_S, QUAD_HOME, HubLifecycle, overseer_context, world_config,
)
from pluggybot.mind import events, llm, look
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import (
  JPEG_MAGIC, MAX_IMAGE_BYTES, MAX_IMAGE_RAW_BYTES, MAX_RAW_BYTES, Inbox,
)
from pluggybot.mind.overseer import LOOK_S, MAX_LOOK_RUN, Menu, Overseer
from pluggybot.robot import FIRST, world_spec
from pluggybot.telemetry.protocol import (
  CODE_HANDLED_TYPES, INBOUND_TYPES, LOOK_OUTCOMES, LOOK_WHYS,
  WORLD_INBOUND_TYPES,
)

from test_body import stub_life
from test_overseer import FakeClient, full

#: A JPEG in name only: the magic bytes and some payload. The door checks
#: the magic and the size, never the picture -- decoding is the model's.
#: Its message is past `MAX_RAW_BYTES`, as a frame the site sends is.
JPEG = JPEG_MAGIC + bytes(range(256)) * 200


def image_message(ref: str, jpeg: bytes = JPEG, robot: str = "pluggybot") -> str:
  return json.dumps({"type": "image", "robot": robot, "ref": ref,
                     "jpeg": base64.b64encode(jpeg).decode("ascii")})


def renderer_message(connected: bool) -> str:
  """The website's word on whether a renderer is there (issue #357)."""
  return json.dumps({"type": "renderer", "connected": connected})


# ---- the door ---------------------------------------------------------------------


def test_the_image_kind_is_inbound_and_code_does_not_handle_it():
  assert "image" in INBOUND_TYPES
  assert "image" not in CODE_HANDLED_TYPES, "a picture is for a mind"
  assert LOOK_OUTCOMES == ("asked", "seen", "none")


def test_the_renderer_word_is_a_state_the_door_keeps_and_never_queues():
  """Whether a renderer is there (issue #357) is a STATE: the newest word
  stands, nothing is queued (a burst that evicts messages cannot evict
  it, and an evicted one would go out as a `dropped` visitor reply), and
  a link that drops forgets it. It is about the world, so a pair's router
  hands it to both robots; and a mind's, like the picture it gates."""
  assert "renderer" in INBOUND_TYPES and WORLD_INBOUND_TYPES == ("renderer",)
  assert "renderer" not in CODE_HANDLED_TYPES, "only a mind looks"
  inbox = Inbox()
  assert inbox.renderer is None, "nobody has said"
  assert inbox.offer(renderer_message(True)).connected is True
  assert inbox.renderer is True and len(inbox) == 0 and inbox.received == 0
  inbox.offer(renderer_message(False))
  assert inbox.renderer is False and len(inbox) == 0
  # A word that might mean either is not taken as one, and leaves the last.
  for junk in ({"type": "renderer", "connected": "yes"},
               {"type": "renderer", "connected": 1}, {"type": "renderer"}):
    assert inbox.offer(json.dumps(junk)) is None
  assert inbox.renderer is False and inbox.dropped_invalid == 3
  inbox.offer(renderer_message(True))
  inbox.forget_renderer()
  assert inbox.renderer is None


def test_a_picture_passes_its_own_cap_and_nothing_else_does():
  """The sentence-sized cap stays for every other kind; a picture has its
  own, on the decoded bytes and the raw message. Shown to fail without the
  door's second cap: a 68 kB image message is over `MAX_RAW_BYTES`."""
  inbox = Inbox()
  raw = image_message("look:pluggybot:1")
  assert len(raw) > MAX_RAW_BYTES
  msg = inbox.offer(raw)
  assert msg is not None and msg.kind == "image"
  assert msg.ref == "look:pluggybot:1" and msg.image == JPEG
  assert msg.as_dict()["bytes"] == len(JPEG) and "jpeg" not in msg.as_dict()
  # A message that needed the room for anything else is dropped unread.
  big_text = json.dumps({"type": "message", "id": "m1", "text": "x" * 70000})
  assert MAX_RAW_BYTES < len(big_text) < MAX_IMAGE_RAW_BYTES
  assert inbox.offer(big_text) is None
  # ...and a picture over its own cap, or not a picture, or unnamed.
  assert inbox.offer(image_message("r", JPEG_MAGIC + b"\0" * MAX_IMAGE_BYTES)) is None
  assert inbox.offer(image_message("r", b"GIF89a" + b"\0" * 100)) is None
  assert inbox.offer(json.dumps({"type": "image", "ref": "r", "jpeg": "not base64!"})) is None
  assert inbox.offer(json.dumps({"type": "image", "ref": "", "jpeg": "AAAA"})) is None
  assert inbox.dropped_invalid == 5
  assert [m.ref for m in inbox.drain(("image",))] == ["look:pluggybot:1"]


# ---- the eye ----------------------------------------------------------------------


def _eye_row(eye: look.Eye, t: float = 1.0) -> dict:
  camera = {"pos": [0.0, 0.0, 0.18], "forward": [1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0], "fovy": 41.0, "width": 640, "height": 480}
  return eye.ask(camera, t=t, x=0.5, y=3.0, heading=1.5707963)


def test_the_eye_answers_the_open_request_and_drops_every_other_picture():
  eye = look.Eye("pluggybot", wait_s=2.0)
  inbox = Inbox()
  row = _eye_row(eye, t=1.0)
  assert row["ref"] == "look:pluggybot:1" and row["outcome"] == "asked"
  assert row["at"] == {"x": 0.5, "y": 3.0, "headingDeg": 90.0}
  # A second ask while one is open is a programming error, never a queue.
  with pytest.raises(RuntimeError):
    _eye_row(eye, t=1.5)
  # A picture for the wrong request is dropped; the right one resolves.
  inbox.offer(image_message("look:pluggybot:0"))
  inbox.offer(image_message("look:pluggybot:1"))
  results = [eye.offer(m, t=2.4) for m in inbox.drain(("image",))]
  assert results[0] is None and results[1] is row
  assert (row["outcome"], row["bytes"], row["waitS"]) == ("seen", len(JPEG), 1.4)
  assert eye.dropped == {"stale": 1}
  # Late twice: a second copy is stale too, and the wire row has no bytes.
  inbox.offer(image_message("look:pluggybot:1"))
  assert eye.offer(inbox.drain(("image",))[0], t=3.0) is None
  assert "_jpeg" in row and "_jpeg" not in look.wire_row(row)
  assert eye.give_up(t=9.0) is None, "nothing open, nothing to give up on"
  # The deadline, on a fresh request.
  row2 = _eye_row(eye, t=10.0)
  assert not eye.overdue(11.9) and eye.overdue(12.0)
  assert eye.give_up(t=12.0) is row2
  assert (row2["outcome"], row2["why"], row2["waitS"]) == ("none", "unanswered", 2.0)
  assert eye.stats() == {"asked": 2, "seen": 1, "none": 1, "dropped": {"stale": 2}}


#: A head with a camera on it, looking along its own +x with +z up, and
#: nothing else: what the eye reads off a world. Mocap, so it stays put.
EYE_WORLD = """<mujoco model="eye">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="head" mocap="true" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
      <camera name="eye" pos="0.05 0 0.02" xyaxes="0 -1 0 0 0 1" fovy="41"/>
    </body>
  </worldbody>
</mujoco>"""


def test_the_camera_pose_is_the_cameras_and_says_which_way_it_looks():
  """`forward` is the way the body faces and `up` is world +z, read off
  the camera's own world pose -- so the picture is taken from exactly where
  the camera is."""
  model = mujoco.MjModel.from_xml_string(EYE_WORLD)
  data = mujoco.MjData(model)
  x, y, yaw = 1.0, 2.0, 0.5
  data.mocap_pos[0] = (x, y, 0.2)
  data.mocap_quat[0] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(model, data)
  pose = look.camera_pose(model, data, "eye")
  assert pose["forward"] == pytest.approx([math.cos(yaw), math.sin(yaw), 0.0], abs=1e-4)
  assert pose["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)
  assert pose["fovy"] == 41.0
  assert (pose["width"], pose["height"]) == (look.WIDTH, look.HEIGHT)
  cid = model.camera("eye").id
  assert pose["pos"] == pytest.approx(list(data.cam_xpos[cid]), abs=1e-4)
  assert pose["pos"][:2] == pytest.approx([x + 0.05 * math.cos(yaw), y + 0.05 * math.sin(yaw)],
                                          abs=1e-4)


def test_a_quadruped_looks_through_its_own_head_camera():
  """The camera is the BODY's (issue #408). The eye asked every body for
  the rover's `left_eye`; on legs that was a KeyError on the physics
  thread, and on the served pair it took the process down -- 35 exits in
  48 h, three of them resetting the world from XML. A quadruped's look
  goes out from its nose camera, facing the way the body faces."""
  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body="quadruped").compile()
  data = mujoco.MjData(model)
  inbox = Inbox()
  inbox.offer(renderer_message(True))      # a renderer is there (issue #357)
  life = HubLifecycle(model, data, realtime=False, world=QUAD_HOME,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], inbox=inbox)
  seen = []
  life.on_event.append(seen.append)
  try:
    life.body.start_at(*cfg["start"])
    assert life.body.head_camera == FIRST.el("nav_eye")
    routine = life._look_routine()
    next(routine)                            # the request is out
    asked = [m for m in seen if m["type"] == "look"]
    assert [m["outcome"] for m in asked] == ["asked"]
    camera = asked[0]["camera"]
    assert camera == look.camera_pose(model, data, FIRST.el("nav_eye"))
    yaw = life.body.true_pose()[2]
    assert camera["forward"] == pytest.approx([math.cos(yaw), math.sin(yaw), 0.0], abs=1e-3)
    assert camera["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-3)
    assert 0.3 < camera["pos"][2] < 0.5, "the nose's height, not the floor's"
    routine.close()
  finally:
    life.body.close()


# ---- the arm ----------------------------------------------------------------------


def test_look_is_a_minds_and_a_bare_menu_has_none_of_it():
  auto = ov.build(QUAD_HOME, enabled=True, client=FakeClient())
  assert auto.menu.look and "look" in auto.menu.available()
  assert "look" in auto.menu.schema()["properties"]["action"]["enum"]
  assert "look" not in auto.menu.schema(look=False)["properties"]["action"]["enum"]
  assert dict(auto.sections)["LOOKING"] == ov.LOOK_RULE
  # A menu no mind was built over offers none of it, and the rules do not
  # name it.
  bare = Overseer(Menu.for_world(QUAD_HOME), client=FakeClient())
  assert not bare.menu.look and "look" not in bare.menu.available()
  assert "look" not in bare.menu.schema()["properties"]["action"]["enum"]
  assert "LOOKING" not in dict(bare.sections)
  assert "`look`" not in ov.RULES
  # The prefix's action list names it where it is offered and not elsewhere.
  prefix = lambda boss: "".join(t for _, t in boss.sections)   # noqa: E731
  assert '"look":' in prefix(auto) and '"look":' not in prefix(bare)
  with pytest.raises(ValueError):
    bare.menu.validate(full(action="look"))
  assert auto.menu.validate(full(action="look")).action == "look"
  with pytest.raises(ValueError, match="off the menu"):
    auto.menu.validate(full(action="look"), look=False)


def test_a_look_is_never_an_order_and_never_a_map_row():
  auto = ov.build(QUAD_HOME, enabled=True, client=FakeClient())
  assert "look" not in auto.menu.orderable(())
  assert "look" not in auto.menu.schema(standing_orders=True)["properties"]["standing_order"]["enum"]
  with pytest.raises(ValueError, match="cannot be `look`"):
    ov.standing_order("look", auto.menu)
  assert not ov.order_runnable(auto.menu, "look", {"looksLeft": 2})
  with pytest.raises(ValueError):
    events.row({"event": "nothing_to_do", "action": "look"}, auto.menu)


def test_looks_left_takes_the_action_off_the_call_and_the_state_says_so():
  assert ov._look_allowed({}) and ov._look_allowed({"looksLeft": 1})
  assert not ov._look_allowed({"looksLeft": 0})
  assert MAX_LOOK_RUN == 2 and LOOK_S == look.LOOK_S


# ---- the delivery ---------------------------------------------------------------------


def _eyed_stub():
  """A stub body whose head camera is `EYE_WORLD`'s: the eye's bookkeeping
  with no robot to look through (the quadruped's own is pinned above)."""
  from pluggybot.body import StubBody
  from pluggybot.economy import energy
  cfg = world_config(QUAD_HOME)
  model = mujoco.MjModel.from_xml_string(EYE_WORLD)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data, rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                  charge_w=energy.load(QUAD_HOME).charge_w)
  body.head_camera = "eye"
  return body


def _looker(*answers, inbox=None, renderer: bool | None = True):
  """A mind that looks, on the eyed stub. `renderer` is the website's word
  (issue #357): a renderer is there unless a test says otherwise, and None
  says nothing at all."""
  boss = ov.build(QUAD_HOME, enabled=True, client=FakeClient(*answers))
  inbox = Inbox() if inbox is None else inbox
  if renderer is not None:
    inbox.offer(renderer_message(renderer))
  life = stub_life(body=_eyed_stub(), overseer=boss, inbox=inbox)
  life.body.start_at(0.5, 3.0, math.pi / 2)
  life.max_sim_time = 0.0            # a fallback's explore ends where it starts
  return boss, life


def _renderer(life, inbox, seen, after_s: float, jpeg: bytes = JPEG):
  """A fake website: answers the first `asked` on the socket `after_s`
  sim-seconds later, once."""
  done = []

  def hook():
    asked = [m for m in seen if m["type"] == "look" and m["outcome"] == "asked"]
    if asked and not done and life.data.time >= asked[0]["t"] + after_s:
      done.append(True)
      inbox.offer(image_message(asked[0]["ref"], jpeg))
  life.body.step_hooks.append(hook)
  return done


def test_a_picture_arrives_next_turn_as_an_image_part_and_never_as_text():
  """The whole round trip, without a renderer: the `look` request goes out
  with the camera's pose, the fake website answers 1.3 s later, the robot
  stops waiting the moment it does, and the NEXT call carries the JPEG as
  an image part in the backend's shape beside a user turn whose JSON says
  `attached` and holds not one byte of base64. Shown once."""
  inbox = Inbox()
  boss, life = _looker(full(action="look"), full(action="idle"), full(action="idle"),
                       inbox=inbox)
  seen = []
  life.on_event.append(seen.append)
  _renderer(life, inbox, seen, after_s=1.3)
  # ...less the think slices the call in flight was stood out in: as many as
  # its worker thread took, which is the box's, not the look's
  thinks = []
  real = life.body.hold_routine
  life.body.hold_routine = lambda s: (thinks.append(s == ov.THINK_SLICE_S), real(s))[1]
  try:
    t0 = float(life.data.time)
    life._decide()
    waited = float(life.data.time) - t0 - sum(thinks) * ov.THINK_SLICE_S
    assert 1.3 <= waited < 1.3 + 2 * LOOK_SLICE_S + 0.2, waited
    rows = [m for m in seen if m["type"] == "look"]
    assert [(m["outcome"], m["ref"]) for m in rows] == [
      ("asked", "look:pluggybot:1"), ("seen", "look:pluggybot:1")]
    assert rows[0]["camera"] == look.camera_pose(life.model, life.data, "eye")
    assert rows[0]["robot"] == "pluggybot" and rows[0]["at"]["x"] == 0.5
    assert rows[1]["bytes"] == len(JPEG) and 1.3 <= rows[1]["waitS"] < 1.8
    assert "_jpeg" not in rows[1] and "jpeg" not in rows[1]
    assert life.state == "LOOK"
    # What the NEXT call is shown.
    state = overseer_context(life)
    block = state["seen"][0]
    assert block["from"] == look.SENDER and block["image"] == "attached"
    assert block["id"] == "look:pluggybot:1"
    assert block["at"]["headingDeg"] == pytest.approx(90.0, abs=1.0)   # the belief's (#386)
    assert base64.b64decode(block["jpeg"]) == JPEG
    assert state["looksLeft"] == MAX_LOOK_RUN - 1
    life._decide()
    call = boss.client.calls[-1]
    content = call["messages"][0]["content"]
    assert isinstance(content, list) and [p["type"] for p in content] == ["image", "text"]
    assert content[0]["source"]["data"] == block["jpeg"]
    assert content[0]["source"]["media_type"] == "image/jpeg"
    text = content[1]["text"]
    assert '"image": "attached"' in text and '"from": "your head camera"' in text
    assert block["jpeg"][:64] not in text, "the picture is never text"
    assert '"looksLeft": 1' in text
    assert block["jpeg"][:64] not in json.dumps(call["system"])
    # Shown once: the second decision saw it, and the run reset on `idle`.
    after = overseer_context(life)
    assert after["seen"] == [] and after["looksLeft"] == MAX_LOOK_RUN
    assert isinstance(boss.client.calls[-1]["messages"][0]["content"], list)
    life._decide()
    assert isinstance(boss.client.calls[-1]["messages"][0]["content"], str)
    assert "a picture came" in life.thoughts.read("History.md")
  finally:
    life.body.close()
  assert life.eye.stats() == {"asked": 1, "seen": 1, "none": 0, "dropped": {}}
  # ...and the summary's rows are the wire's: never the bytes.
  assert "looks" in inspect.getsource(HubLifecycle.end)
  assert [r["outcome"] for r in map(look.wire_row, life.eye.looks)] == ["seen"]
  assert all("_jpeg" not in look.wire_row(r) for r in life.eye.looks)


def test_nobody_answering_is_none_said_so_after_the_deadline():
  """A renderer the website says is there, that never answers: the robot
  stands still `LOOK_S`, the row says `none` and why, and the next turn is
  told -- as text, because there is no picture to attach."""
  boss, life = _looker(full(action="look"), full(action="idle"))
  seen = []
  life.on_event.append(seen.append)
  # The served deadline is ten seconds; the flight here waits one, so the
  # rule (stand still exactly the deadline, then say `none`) is pinned in
  # a second rather than ten.
  assert life.eye.wait_s == LOOK_S == 10.0
  life.eye.wait_s = 1.0
  try:
    life._decide()
    rows = [m for m in seen if m["type"] == "look"]
    assert [(m["outcome"], m["why"]) for m in rows] == [("asked", ""), ("none", "unanswered")]
    # The deadline, to within the slice the stand-still is cut into.
    assert 1.0 <= rows[1]["waitS"] <= 1.0 + 2 * LOOK_SLICE_S
    assert rows[1]["t"] == rows[0]["t"], "the same row, resolved"
    block = overseer_context(life)["seen"][0]
    assert block["image"] == "none" and block["why"] == "unanswered" and "jpeg" not in block
    life._decide()
    content = boss.client.calls[-1]["messages"][0]["content"]
    assert isinstance(content, str) and '"image": "none"' in content
    assert "no picture came back" in life.thoughts.read("History.md")
  finally:
    life.body.close()


def test_a_late_picture_is_dropped_and_a_picture_waits_for_the_models_own_turn():
  """A renderer that answers after the deadline hands the robot nothing
  (a picture of where it used to be); and a picture that DID come stays on
  the shelf across a fallback, which made no call and saw nothing."""
  inbox = Inbox()
  boss, life = _looker(full(action="look"), RuntimeError("endpoint down"),
                       full(action="idle"), inbox=inbox)
  seen = []
  life.on_event.append(seen.append)
  life.eye.wait_s = 1.0
  _renderer(life, inbox, seen, after_s=1.5)
  try:
    life._decide()
    assert [m["outcome"] for m in seen if m["type"] == "look"] == ["asked", "none"]
    # The late picture is on the socket now; the next pass drains it.
    life.body.run(life.body.hold_routine(1.5))
    life._look_step()
    assert life.eye.dropped == {"stale": 1}
    assert overseer_context(life)["seen"][0]["image"] == "none"
    life._decide()                            # the fallback
    assert boss.decisions[-1].scripted
    assert len(overseer_context(life)["seen"]) == 1, "lost to an outage"
    life._decide()                            # the model's own
    assert overseer_context(life)["seen"] == []
  finally:
    life.body.close()


def test_the_run_cap_takes_look_off_the_menu_then_gives_it_back():
  """Two looks in a row and the third call's grammar has no `look`; an
  answer that names it anyway is malformed; any other action resets."""
  inbox = Inbox()
  boss, life = _looker(full(action="look"), full(action="look"),
                       full(action="look"), full(action="idle"), inbox=inbox)
  life.eye.wait_s = 0.5
  try:
    life._decide()
    assert overseer_context(life)["looksLeft"] == 1
    life._decide()
    assert overseer_context(life)["looksLeft"] == 0
    life._decide()                            # the third `look` is malformed
    assert boss.decisions[-1].scripted and boss.decisions[-1].source == "fallback:garbled"
    assert "look" not in boss.client.calls[-1]["output_config"]["format"]["schema"]["properties"]["action"]["enum"]
    assert life._look_run == 0, "a fallback is not a look"
    life._decide()
    assert boss.decisions[-1].action == "idle" and not boss.decisions[-1].scripted
    assert "look" in boss.client.calls[-1]["output_config"]["format"]["schema"]["properties"]["action"]["enum"]
    assert overseer_context(life)["looksLeft"] == MAX_LOOK_RUN
  finally:
    life.body.close()


def test_with_no_renderer_look_is_off_the_call_and_the_state_says_why():
  """While nothing can take a picture (issue #357) the call's grammar has
  no `look` and the state says so in `camera`; the website's word that a
  renderer is there puts it back, and its word that none is takes it off
  again. Nobody having said is no renderer: a demo with no website, a sim
  before the hub's first word, a link that dropped. On the deployed pair
  every look stood its ten seconds for a renderer never deployed."""
  inbox = Inbox()
  boss, life = _looker(full(action="idle"), full(action="idle"), inbox=inbox,
                       renderer=None)
  schema = lambda: boss.client.calls[-1]["output_config"]["format"]["schema"]  # noqa: E731
  try:
    assert overseer_context(life)["camera"] == look.NO_PICTURE
    life._decide()
    assert "look" not in schema()["properties"]["action"]["enum"]
    inbox.offer(renderer_message(True))
    assert "camera" not in overseer_context(life)
    life._decide()
    assert "look" in schema()["properties"]["action"]["enum"]
    # (a third idle in a row is the idle-run policy, which calls nobody)
    inbox.offer(renderer_message(False))
    state = overseer_context(life)
    assert state["camera"] == look.NO_PICTURE and not ov._look_allowed(state)
  finally:
    life.body.close()
  assert not ov._look_allowed({"looksLeft": 2, "camera": look.NO_PICTURE})
  assert ov._look_allowed({"looksLeft": 2})


def test_a_look_that_races_the_word_is_answered_at_once_and_never_stood_out():
  """`look` is off the menu with no renderer there, so a look that runs
  anyway raced the website's word: it resolves at once, `none` /
  `unanswerable`, with no ten seconds stood for nobody, and the robot is
  told so in words that say nothing can take a picture, not that one was
  late. Shown to fail without the at-once branch: the row stood `LOOK_S`
  and said `unanswered`."""
  boss, life = _looker(renderer=False)
  seen = []
  life.on_event.append(seen.append)
  try:
    t0 = float(life.data.time)
    life.body.run(life._look_routine())
    assert float(life.data.time) == t0, "no time stood"
    rows = [m for m in seen if m["type"] == "look"]
    assert [(m["outcome"], m["why"], m["waitS"]) for m in rows] == [
      ("asked", "", 0.0), ("none", look.UNANSWERABLE, 0.0)]
    block = overseer_context(life)["seen"][0]
    assert block["text"] == look.NO_PICTURE and block["why"] == look.UNANSWERABLE
    assert block["image"] == "none" and "jpeg" not in block
    assert "nothing could take a picture" in life.thoughts.read("History.md")
  finally:
    life.body.close()
  assert LOOK_WHYS == ("unanswered", "unanswerable", "aborted")


def test_the_word_decides_whether_a_wait_begins_and_never_ends_one():
  """A renderer that drops mid-render and comes back still sends its
  picture: `renderer/eye.js` keeps the request across its own reconnect,
  and a hub may relay one across the sim's blip too. So the word that it
  went (issue #357) takes `look` off the NEXT call, and the wait already
  begun runs to its picture or its deadline. Shown to fail when the word
  ended the wait: the picture came a second later and was dropped as
  stale, and the robot was told nothing could take one."""
  inbox = Inbox()
  boss, life = _looker(inbox=inbox)
  seen = []
  life.on_event.append(seen.append)
  said = []

  def renderer_blinks():
    pending = life.eye.pending
    if pending is None:
      return
    if not said and life.data.time >= pending["t"] + 1.0:
      said.append("gone")
      inbox.offer(renderer_message(False))
    elif said == ["gone"] and life.data.time >= pending["t"] + 2.0:
      said.append("back")
      inbox.offer(renderer_message(True))
      inbox.offer(image_message(pending["ref"]))
  life.body.step_hooks.append(renderer_blinks)
  try:
    life.body.run(life._look_routine())
    row = [m for m in seen if m["type"] == "look"][-1]
    assert (row["outcome"], row["bytes"]) == ("seen", len(JPEG)), row
    assert 2.0 <= row["waitS"] <= 2.0 + 2 * LOOK_SLICE_S, row["waitS"]
    assert life.eye.dropped == {}
  finally:
    life.body.close()


def test_a_mind_with_no_eye_has_no_seen_block_and_its_turn_is_a_string():
  boss = Overseer(Menu.for_world(QUAD_HOME, None), client=FakeClient())
  life = stub_life(overseer=boss)
  try:
    state = overseer_context(life)
    assert "seen" not in state and "looksLeft" not in state
    life._decide()
    assert isinstance(boss.client.calls[-1]["messages"][0]["content"], str)
  finally:
    life.body.close()


# ---- the request, and the identity -------------------------------------------------


def test_the_user_content_is_the_user_turn_unless_a_picture_is_attached():
  state = {"battery": {"fraction": 0.5}, "looksLeft": 2, "seen": []}
  turn = lambda st, backend: ov._user_content(   # noqa: E731
    ov.model_state(st, True), ov._pictures(st), backend)
  assert turn(state, "huggingface") == ov._user_turn(state)
  with_none = {**state, "seen": [{"id": "look:pluggybot:1", "image": "none"}]}
  assert turn(with_none, "huggingface") == ov._user_turn(with_none)
  b64 = base64.b64encode(JPEG).decode()
  with_pic = {**state, "seen": [{"id": "look:pluggybot:1", "image": "attached", "jpeg": b64}]}
  parts = turn(with_pic, "huggingface")
  assert parts[0] == {"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64," + b64}}
  assert parts[1] == {"type": "text", "text": ov._user_turn(
    {**state, "seen": [{"id": "look:pluggybot:1", "image": "attached"}]})}
  assert turn(with_pic, "anthropic")[0] == llm.image_part("anthropic", b64)
  assert llm.image_part("anthropic", b64)["source"]["media_type"] == "image/jpeg"
  assert llm.image_part("local", b64)["type"] == "image_url"
  assert with_pic["seen"][0]["jpeg"] == b64, "the state is not edited"


def test_an_escalation_carries_the_picture_in_its_own_backends_shape():
  src = inspect.getsource(Overseer._maybe_escalate)
  assert "_user_content(" in src and "self.escalate_backend" in src
  assert "_user_turn(" not in src


def test_the_picture_never_reaches_any_turn_as_text_including_an_interrupt():
  """`model_state` is where the bytes leave, for every turn, so the ONE turn
  that is built off the state without `_user_content` -- the mid-errand
  interrupt -- carries no base64 either. Shown to fail with the strip in
  `_user_content` alone: an interrupt fired while a picture waited on
  the shelf (a fallback's standing order started the errand, and the
  shelf keeps a picture across a fallback) dumped 16 kB of base64 into
  the question."""
  b64 = base64.b64encode(JPEG).decode()
  state = {"battery": {"fraction": 0.1, "wh": 0.1}, "looksLeft": 1,
           "seen": [{"id": "look:pluggybot:1", "image": "attached", "jpeg": b64}]}
  for survival in (False, True):
    shown = ov.model_state(state, survival=survival)
    assert "jpeg" not in json.dumps(shown) and shown["seen"][0]["image"] == "attached"
  assert ov._pictures(state) == [b64]
  assert state["seen"][0]["jpeg"] == b64, "the raw state keeps it for the image part"
  assert ov.model_state({"battery": {}}) is not None
  # ...and the interrupt's own turn.
  assert b64[:32] not in ov._interrupt_turn(ov.model_state(state), "draw", "low")
  auto = ov.build(QUAD_HOME, enabled=True,
                  client=FakeClient({"continue_errand": True, "reason": "nearly done"}))
  auto.start_interrupt(state, "draw", "your pack is at 10%")
  while auto.interrupt_pending:
    pass
  assert auto.interrupt_result()["continue"] is True
  content = auto.client.calls[-1]["messages"][0]["content"]
  assert isinstance(content, str) and b64[:32] not in content


def test_a_picture_arriving_with_no_look_open_is_dropped_at_the_next_pass():
  """A renderer answering after the deadline: the picture is drained off
  the inbox at the top of the next arbitration pass and counted stale --
  never left to sit in the queue (an evicted picture would go out as a
  `dropped` visitor reply for a message nobody sent)."""
  inbox = Inbox()
  boss, life = _looker(full(action="idle"), inbox=inbox)
  replies = []
  life.visitor_hooks.append(replies.append)
  try:
    inbox.offer(image_message("look:pluggybot:7"))
    assert len(inbox) == 1
    life._visitor_step()
    assert len(inbox) == 0 and life.eye.dropped == {"stale": 1}
    assert replies == [] and life._seen == []
  finally:
    life.body.close()


def test_a_stop_thrown_into_a_look_closes_the_request():
  """`stop_when`'s `MissionAborted` lands mid-wait: the request resolves
  `none` (`aborted`) so the eye is never left holding one -- `Eye.ask`
  refuses a second request while one is open."""
  from pluggybot.tick import MissionAborted
  boss, life = _looker(full(action="look"), full(action="idle"))
  seen = []
  life.on_event.append(seen.append)
  try:
    routine = life._look_routine()
    next(routine)                            # the request is out
    assert life.eye.pending is not None
    with pytest.raises(MissionAborted):
      routine.throw(MissionAborted("stopped"))
    assert life.eye.pending is None
    assert [(m["outcome"], m["why"]) for m in seen if m["type"] == "look"] == [
      ("asked", ""), ("none", "aborted")]
    # A second look can be asked for.
    life.eye.ask({}, t=1.0, x=0.0, y=0.0, heading=0.0)
  finally:
    life.body.close()


def test_the_eye_can_be_switched_off_for_a_mind_that_takes_no_picture(monkeypatch):
  monkeypatch.setenv(ov.LOOK_ENV, "0")
  off = ov.build(QUAD_HOME, enabled=True, client=FakeClient())
  assert not off.menu.look and "look" not in off.menu.available()
  assert "LOOKING" not in dict(off.sections)
  monkeypatch.delenv(ov.LOOK_ENV)
  on = ov.build(QUAD_HOME, enabled=True, client=FakeClient())
  assert on.menu.look
  assert not ov.build(QUAD_HOME, enabled=True, client=FakeClient(),
                      look=False).menu.look


def test_a_heading_is_reported_wrapped():
  assert [look.wrap_degrees(d) for d in (0, 90, 180, -180, 450, -190, 360)] == [
    0.0, 90.0, 180.0, 180.0, 90.0, 170.0, 0.0]
  eye = look.Eye("pluggybot")
  assert eye.ask({}, t=0.0, x=0.0, y=0.0, heading=math.radians(450))["at"]["headingDeg"] == 90.0


def test_which_model_looked_is_in_the_build_identity_and_absent_without_an_eye():
  seen = build_identity(QUAD_HOME, arm="autonomous", model="org/m:cheapest",
                        backend="huggingface", eyes="org/m:cheapest", commit="abc")
  assert seen["eyes"] == "org/m:cheapest" and seen["model"] == "org/m:cheapest"
  blind = build_identity(QUAD_HOME, arm="autonomous", model="org/m", backend="huggingface",
                         commit="abc")
  assert "eyes" not in blind


# ---- the rule --------------------------------------------------------------------------


def test_the_rule_says_what_the_action_does_and_prescribes_no_looking():
  rule = ov.LOOK_RULE
  assert "`look`" in rule and "`seen`" in rule and "`looksLeft`" in rule
  assert "`camera`" in rule, "the key that says nothing can take a picture"
  assert "head camera" in rule and "as the people watching you see it" in rule
  for word in ("charge", "battery", "rack", "renderer", "website", "tresjs"):
    assert word not in rule.lower(), word
  for word in ("for example", "e.g.", "such as", "you should", "describe"):
    assert word not in rule.lower(), word
  # The lifecycle never captions: nothing it emits or shelves is a sentence
  # about what is IN the picture.
  for fn in (HubLifecycle._look_routine, HubLifecycle._resolve_look, look.as_context):
    src = inspect.getsource(fn).lower()
    for word in ("fence", "tree", "wall", "caption", "describe"):
      assert word not in src, (fn.__name__, word)
