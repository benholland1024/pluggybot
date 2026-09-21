"""The eye (issue #275): the robot looks at the world as the site draws it.

Every rule pinned without a renderer, a network or a model, in
milliseconds: the door (an `image` inbound kind with its own byte cap,
checked for being a JPEG), the eye (one open request, a stale picture
dropped), the arm (`look` on `autonomous` alone, `guarded`'s menu, schema
and prefix unchanged), the orders (never a standing order, never a map
row), the delivery (a fake renderer answering on the socket mid-wait; the
deadline; the picture as an IMAGE PART of the next turn and never text;
shown once, kept across a fallback; the run cap), the build identity, and
the rule's text. The whole mission is never flown.
"""

import base64
import hashlib
import inspect
import json
import math

import mujoco
import pytest

from pluggybot.evaluation.record import build_identity
from pluggybot.lifecycle import (
  LOOK_SLICE_S, HubLifecycle, overseer_context, world_config,
)
from pluggybot.mind import events, llm, look
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import (
  JPEG_MAGIC, MAX_IMAGE_BYTES, MAX_IMAGE_RAW_BYTES, MAX_RAW_BYTES, Inbox,
)
from pluggybot.mind.overseer import LOOK_S, MAX_LOOK_RUN, Menu, Overseer
from pluggybot.robot import FIRST
from pluggybot.telemetry.protocol import (
  CODE_HANDLED_TYPES, INBOUND_TYPES, LOOK_OUTCOMES,
)

from test_autonomous import GUARDED_RULES_SHA
from test_overseer import FakeClient, _lifecycle, full

#: A JPEG in name only: the magic bytes and some payload. The door checks
#: the magic and the size, never the picture -- decoding is the model's.
JPEG = JPEG_MAGIC + bytes(range(256)) * 40


def image_message(ref: str, jpeg: bytes = JPEG, robot: str = "pluggybot") -> str:
  return json.dumps({"type": "image", "robot": robot, "ref": ref,
                     "jpeg": base64.b64encode(jpeg).decode("ascii")})


# ---- the door ---------------------------------------------------------------------


def test_the_image_kind_is_inbound_and_code_does_not_handle_it():
  assert "image" in INBOUND_TYPES
  assert "image" not in CODE_HANDLED_TYPES, "a picture is for a mind"
  assert LOOK_OUTCOMES == ("asked", "seen", "none")


def test_a_picture_passes_its_own_cap_and_nothing_else_does():
  """The sentence-sized cap stays for every other kind; a picture has its
  own, on the decoded bytes and the raw message. Shown to fail without the
  door's second cap: a 16 kB image message is over `MAX_RAW_BYTES`."""
  inbox = Inbox()
  raw = image_message("look:pluggybot:1")
  assert len(raw) > MAX_RAW_BYTES
  msg = inbox.offer(raw)
  assert msg is not None and msg.kind == "image"
  assert msg.ref == "look:pluggybot:1" and msg.image == JPEG
  assert msg.as_dict()["bytes"] == len(JPEG) and "jpeg" not in msg.as_dict()
  # A message that needed the room for anything else is dropped unread.
  big_text = json.dumps({"type": "message", "id": "m1", "text": "x" * 9000})
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


def test_the_camera_pose_is_the_head_cameras_and_says_which_way_it_looks():
  """`forward` is the way the body faces and `up` is world +z, read off the
  camera the tag detector renders from -- so the picture is taken from
  exactly where the metric camera is."""
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  x, y, yaw = 1.0, 2.0, 0.5
  q = FIRST.qpos_adr(model)
  data.qpos[q:q + 2] = (x, y)
  data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(model, data)
  pose = look.camera_pose(model, data, FIRST.el(look.CAMERA))
  assert pose["forward"] == pytest.approx([math.cos(yaw), math.sin(yaw), 0.0], abs=1e-4)
  assert pose["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)
  assert pose["fovy"] == float(model.cam_fovy[model.camera(FIRST.el(look.CAMERA)).id])
  assert (pose["width"], pose["height"]) == (look.WIDTH, look.HEIGHT)
  cid = model.camera(FIRST.el(look.CAMERA)).id
  assert pose["pos"] == pytest.approx(list(data.cam_xpos[cid]), abs=1e-4)
  assert 0.1 < pose["pos"][2] < 0.3, "the head's height, not the floor's"


# ---- the arm ----------------------------------------------------------------------


def test_look_is_offered_on_autonomous_alone_and_guarded_is_unchanged():
  auto = ov.build("room_hub", enabled=True, client=FakeClient(), autonomous=True)
  assert auto.menu.look and "look" in auto.menu.available()
  assert "look" in auto.menu.schema()["properties"]["action"]["enum"]
  assert "look" not in auto.menu.schema(look=False)["properties"]["action"]["enum"]
  assert dict(auto.sections)["LOOKING"] == ov.LOOK_RULE
  guarded = ov.build("room_hub", enabled=True, client=FakeClient())
  assert not guarded.menu.look and "look" not in guarded.menu.available()
  assert "look" not in guarded.menu.schema()["properties"]["action"]["enum"]
  assert "LOOKING" not in dict(guarded.sections)
  assert "`look`" not in ov.RULES and "`look`" not in ov.RULES_AUTONOMOUS
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  # The prefix's action list names it on one arm and not the other.
  prefix = lambda boss: "".join(t for _, t in boss.sections)   # noqa: E731
  assert '"look":' in prefix(auto) and '"look":' not in prefix(guarded)
  with pytest.raises(ValueError):
    guarded.menu.validate(full(action="look"))
  assert auto.menu.validate(full(action="look")).action == "look"
  with pytest.raises(ValueError, match="off the menu"):
    auto.menu.validate(full(action="look"), look=False)


def test_a_look_is_never_an_order_and_never_a_map_row():
  auto = ov.build("room_hub", enabled=True, client=FakeClient(), autonomous=True)
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


def _looker(*answers, inbox=None):
  boss = ov.build("room_hub", enabled=True, client=FakeClient(*answers),
                  autonomous=True)
  life = _lifecycle("room_hub", overseer=boss, errand=False, inbox=inbox)
  life.mission.start_at(*world_config("room_hub")["start"])
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
  life.mission.step_hooks.append(hook)
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
  try:
    t0 = float(life.data.time)
    life._decide()
    waited = float(life.data.time) - t0
    assert 1.3 <= waited < 1.3 + 2 * LOOK_SLICE_S + 0.2, waited
    rows = [m for m in seen if m["type"] == "look"]
    assert [(m["outcome"], m["ref"]) for m in rows] == [
      ("asked", "look:pluggybot:1"), ("seen", "look:pluggybot:1")]
    assert rows[0]["camera"]["fovy"] == 41.0 and rows[0]["camera"]["width"] == look.WIDTH
    assert rows[0]["robot"] == "pluggybot" and rows[0]["at"]["x"] == 0.5
    assert rows[1]["bytes"] == len(JPEG) and 1.3 <= rows[1]["waitS"] < 1.8
    assert "_jpeg" not in rows[1] and "jpeg" not in rows[1]
    assert life.state == "LOOK"
    # What the NEXT call is shown.
    state = overseer_context(life)
    block = state["seen"][0]
    assert block["from"] == look.SENDER and block["image"] == "attached"
    assert block["id"] == "look:pluggybot:1" and block["at"]["headingDeg"] == 90.0
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
    life.mission.close()
  assert life.eye.stats() == {"asked": 1, "seen": 1, "none": 0, "dropped": {}}
  # ...and the record's rows are the wire's: never the bytes.
  assert "looks" in inspect.getsource(HubLifecycle.end)
  assert [r["outcome"] for r in map(look.wire_row, life.eye.looks)] == ["seen"]
  assert all("_jpeg" not in look.wire_row(r) for r in life.eye.looks)


def test_nobody_answering_is_none_said_so_after_the_deadline():
  """No inbox at all (a demo, a test): the robot stands still `LOOK_S`,
  the row says `none` and why, and the next turn is told -- as text,
  because there is no picture to attach."""
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
    life.mission.close()


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
    for _ in range(3):
      life.mission.run(life.mission._drive_routine(0.5, 0.0, 0.0))
    life._look_step()
    assert life.eye.dropped == {"stale": 1}
    assert overseer_context(life)["seen"][0]["image"] == "none"
    life._decide()                            # the fallback
    assert boss.decisions[-1].scripted
    assert len(overseer_context(life)["seen"]) == 1, "lost to an outage"
    life._decide()                            # the model's own
    assert overseer_context(life)["seen"] == []
  finally:
    life.mission.close()


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
    life.mission.close()


def test_a_guarded_world_has_no_seen_block_and_its_turn_is_a_string():
  boss = Overseer(Menu.for_world("room_hub", None), client=FakeClient())
  life = _lifecycle("room_hub", overseer=boss, errand=False)
  try:
    state = overseer_context(life)
    assert "seen" not in state and "looksLeft" not in state
    life._decide()
    assert isinstance(boss.client.calls[-1]["messages"][0]["content"], str)
  finally:
    life.mission.close()


# ---- the request, and the identity -------------------------------------------------


def test_the_user_content_is_the_user_turn_unless_a_picture_is_attached():
  state = {"battery": {"fraction": 0.5}, "looksLeft": 2, "seen": []}
  assert ov._user_content(state, "huggingface") == ov._user_turn(state)
  with_none = {**state, "seen": [{"id": "look:pluggybot:1", "image": "none"}]}
  assert ov._user_content(with_none, "huggingface") == ov._user_turn(with_none)
  b64 = base64.b64encode(JPEG).decode()
  with_pic = {**state, "seen": [{"id": "look:pluggybot:1", "image": "attached", "jpeg": b64}]}
  parts = ov._user_content(with_pic, "huggingface")
  assert parts[0] == {"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64," + b64}}
  assert parts[1] == {"type": "text", "text": ov._user_turn(
    {**state, "seen": [{"id": "look:pluggybot:1", "image": "attached"}]})}
  assert ov._user_content(with_pic, "anthropic")[0] == llm.image_part("anthropic", b64)
  assert llm.image_part("anthropic", b64)["source"]["media_type"] == "image/jpeg"
  assert llm.image_part("local", b64)["type"] == "image_url"
  assert with_pic["seen"][0]["jpeg"] == b64, "the state is not edited"


def test_an_escalation_carries_the_picture_in_its_own_backends_shape():
  src = inspect.getsource(Overseer._maybe_escalate)
  assert "_user_content(" in src and "self.escalate_backend" in src
  assert "_user_turn(" not in src


def test_which_model_looked_is_in_the_build_identity_and_absent_without_an_eye():
  seen = build_identity("home", arm="autonomous", model="org/m:cheapest",
                        backend="huggingface", eyes="org/m:cheapest", commit="abc")
  assert seen["eyes"] == "org/m:cheapest" and seen["model"] == "org/m:cheapest"
  blind = build_identity("home", arm="guarded", model="org/m", backend="huggingface",
                         commit="abc")
  assert "eyes" not in blind


# ---- the rule --------------------------------------------------------------------------


def test_the_rule_says_what_the_action_does_and_prescribes_no_looking():
  rule = ov.LOOK_RULE
  assert "`look`" in rule and "`seen`" in rule and "`looksLeft`" in rule
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
