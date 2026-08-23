"""PluggyWorld protocol guards (webserver v0): transpiler + recorder.

The transpiler's conversions are exactly the kind of pure math that renders
wrong silently -- a half-extent box is a quarter the volume and still looks
like a box, a Z-axis cylinder rendered on Y is a fallen column. Each
conversion is asserted numerically here, on a tiny inline model where the
right answer is known by construction, plus a coverage pass over the real
room_hub. The recorder tests drive the real seam contract: decimation,
keyframe-then-sparse frames, and everything queued reaching the file.
"""

import gzip
import json
import queue
import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pluggybot.telemetry.protocol import PROTOCOL_VERSION, body_census, dynamic_flags
from pluggybot.telemetry.recorder import FrameBuilder, TelemetryRecorder
from pluggybot.telemetry.scene import geom_size, quat_mul, scene_dict

REPO = Path(__file__).parent.parent

# One of everything the protocol must carry: all five geom types, a textured
# material, an invisible collision geom, a free robot body (named pluggybot,
# which the census keys on), a free world body, and a static body.
MINI_XML = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="16" height="16"/>
    <material name="mat_checker" texture="checker"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 2 0.1"/>
    <body name="pluggybot" pos="0 0 0.5">
      <freejoint/>
      <geom name="chassis" type="box" size="0.1 0.2 0.3"/>
      <geom name="mast" type="cylinder" size="0.05 0.15" rgba="1 0 0 1"/>
      <geom name="bumper" type="capsule" size="0.02 0.1" euler="0 90 0"/>
      <geom name="ghost" type="box" size="0.1 0.1 0.1" rgba="1 1 1 0"
            contype="0" conaffinity="0"/>
    </body>
    <body name="ball" pos="1 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.04"/>
    </body>
    <body name="post" pos="2 0 0.2">
      <geom name="post_tag" type="box" size="0.1 0.1 0.2"
            material="mat_checker"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def mini_model():
  return mujoco.MjModel.from_xml_string(MINI_XML)


@pytest.fixture()
def mini_scene(mini_model):
  return scene_dict(mini_model, "mini")


def find_geom(scene, name):
  for body in scene["bodies"]:
    for geom in body["geoms"]:
      if geom["name"] == name:
        return geom
  raise AssertionError(f"geom {name} not in scene")


def rotate(quat_wxyz, vec):
  w, x, y, z = quat_wxyz
  return Rotation.from_quat([x, y, z, w]).apply(vec)


# ---- transpiler conversions ------------------------------------------------

def test_sizes_are_full_extents(mini_scene):
  """MuJoCo stores halves; the protocol ships the full figure."""
  assert find_geom(mini_scene, "chassis")["size"] == [0.2, 0.4, 0.6]
  assert find_geom(mini_scene, "mast")["size"] == [0.05, 0.3]
  assert find_geom(mini_scene, "bumper")["size"] == [0.02, 0.2]
  assert find_geom(mini_scene, "ball")["size"] == [0.04]
  assert find_geom(mini_scene, "floor")["size"] == [6.0, 4.0]


def test_cylinder_axis_swap(mini_scene):
  """A Y-axis ThreeJS cylinder under the emitted quat must stand on
  MuJoCo's axis (local +Z here: the mast is unrotated)."""
  q = find_geom(mini_scene, "mast")["quat"]
  assert np.allclose(rotate(q, [0, 1, 0]), [0, 0, 1], atol=1e-6)


def test_capsule_axis_swap_composes_with_geom_rotation(mini_scene):
  """The bumper is rotated 90 deg about Y, laying its MuJoCo axis along
  world +X -- the composed quat must carry ThreeJS's +Y all the way there,
  not just perform the generic fix."""
  q = find_geom(mini_scene, "bumper")["quat"]
  assert np.allclose(rotate(q, [0, 1, 0]), [1, 0, 0], atol=1e-6)


def test_plane_needs_no_axis_fix(mini_scene):
  """Both sides agree a plane faces local +Z; the quat stays identity."""
  assert np.allclose(find_geom(mini_scene, "floor")["quat"], [1, 0, 0, 0])


def test_box_quat_untouched(mini_scene):
  assert np.allclose(find_geom(mini_scene, "chassis")["quat"], [1, 0, 0, 0])


def test_mesh_refused():
  """The primitives-only world is a design decision, not an accident --
  a mesh must fail the transpile loudly."""
  with pytest.raises(ValueError, match="primitives-only"):
    geom_size(int(mujoco.mjtGeom.mjGEOM_MESH), [0.1, 0.1, 0.1])


def test_quat_mul_matches_scipy():
  a, b = (0.5, 0.5, 0.5, 0.5), (0.8, 0.0, 0.6, 0.0)
  got = quat_mul(a, b)
  ra = Rotation.from_quat([a[1], a[2], a[3], a[0]])
  rb = Rotation.from_quat([b[1], b[2], b[3], b[0]])
  x, y, z, w = (ra * rb).as_quat()
  assert np.allclose(got, [w, x, y, z], atol=1e-9)


# ---- scene structure -------------------------------------------------------

def test_invisible_geoms_skipped(mini_scene):
  with pytest.raises(AssertionError):
    find_geom(mini_scene, "ghost")


def test_texture_reference_and_table(mini_scene):
  assert find_geom(mini_scene, "post_tag")["texture"] == "checker"
  (tex,) = mini_scene["textures"]
  assert tex["name"] == "checker" and tex["file"] == "checker.png"
  assert tex["width"] == 16 and tex["height"] == 16


def test_scene_is_json_serializable(mini_scene):
  """Regression: dynamic_flags once handed numpy.bool_ through `or`, and
  json.dumps refuses numpy scalars -- the scene compiled fine in every test
  and died on the first real write."""
  json.dumps(mini_scene)


def test_dynamic_census(mini_model, mini_scene):
  robot, world = body_census(mini_model)
  assert robot == ["pluggybot"] and world == ["ball"]
  flags = {b["name"]: b["dynamic"] for b in mini_scene["bodies"]}
  assert flags["pluggybot"] and flags["ball"]
  assert not flags["post"] and not flags["world"]
  robots = {b["name"]: b["robot"] for b in mini_scene["bodies"]}
  assert robots["pluggybot"] == "pluggybot" and robots["ball"] is None


def test_body_poses_are_world_frame(mini_scene):
  """The post's rest pose must be its WORLD position -- a client renders
  straight from these with no kinematic-tree math."""
  post = next(b for b in mini_scene["bodies"] if b["name"] == "post")
  assert post["pos"] == [2.0, 0.0, 0.2]
  assert post["parent"] == "world" and post["visual"] is None


def test_room_hub_coverage():
  """Every visible geom of the real world transpiles, all five primitive
  types are exercised, and every AprilTag texture is referenced."""
  model = mujoco.MjModel.from_xml_path(str(REPO / "models" / "room_hub.xml"))
  scene = scene_dict(model, "room_hub")
  assert scene["protocolVersion"] == PROTOCOL_VERSION
  geoms = [g for b in scene["bodies"] for g in b["geoms"]]
  visible = sum(1 for g in range(model.ngeom) if model.geom_rgba[g][3] > 0)
  assert len(geoms) == visible
  assert {g["type"] for g in geoms} == {"plane", "box", "cylinder",
                                        "capsule", "sphere"}
  referenced = {g["texture"] for g in geoms if g["texture"]}
  assert len(referenced) == model.ntex        # all 12 tags in use
  assert {t["name"] for t in scene["textures"]} == referenced
  robot, world = body_census(model)
  # 21 = 7 robot links + the rack + five modules + the pen's two moving
  # parts + the dispenser's shuttle + three loose seeds. A census, so it
  # fails whenever the world gains or loses a dynamic body -- which is
  # the point: every one of them costs a pose in every keyframe.
  assert sum(dynamic_flags(model)) == len(robot) + len(world) == 21
  assert len(robot) == 7
  assert {"rack", "module_lcd", "module_seed", "seed_0"} <= set(world)


# ---- recorder --------------------------------------------------------------

def record(model, seconds=2.0, path=None, status_fn=None, tmp=None,
           **kwargs):
  data = mujoco.MjData(model)
  path = path or str(tmp / "out.jsonl")
  rec = TelemetryRecorder(model, data, path, status_fn=status_fn,
                          model_name="mini", **kwargs)
  for _ in range(round(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    rec.step_hook()
  rec.close()
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt") as f:
    lines = [json.loads(line) for line in f]
  return rec, lines


def test_recorder_honours_the_keyframe_cadence(mini_model, tmp_path):
  """The recorder's keyframe_s must reach its builder: `serve.py --record`
  writes a recording of the SAME run it streams, so a cadence that applied
  to one and not the other would make the two artifacts disagree."""
  rec, lines = record(mini_model, seconds=3.0, tmp=tmp_path, keyframe_s=0.5)
  header, frames = lines[0], lines[1:]
  assert header["keyframeS"] == 0.5
  keys = [f for f in frames if f.get("key")]
  assert len(keys) >= 5, f"expected ~6 keyframes over 3 s, got {len(keys)}"
  for f in keys:
    assert set(f["robots"]["pluggybot"]["bodies"]) == set(header["robots"]["pluggybot"])
    assert set(f["world"]) == set(header["world"])


def test_recorder_header_and_decimation(mini_model, tmp_path):
  rec, lines = record(mini_model, seconds=2.0, tmp=tmp_path)
  header, frames = lines[0], lines[1:]
  assert header["type"] == "header"
  assert header["protocolVersion"] == PROTOCOL_VERSION
  assert header["robots"] == {"pluggybot": ["pluggybot"]}
  assert header["world"] == ["ball"]
  # ~20 Hz of sim time out of 500 Hz of steps, spacing never under 1/hz
  assert len(frames) == pytest.approx(2.0 * header["hz"], abs=2)
  times = [f["t"] for f in frames]
  assert all(b - a >= 1 / header["hz"] - 1e-6 for a, b in zip(times, times[1:]))
  # everything queued reached the file: close() drains before returning
  assert len(frames) == rec.frames


def test_first_frame_is_keyframe_then_sparse(mini_model, tmp_path):
  """Frame 0 carries every dynamic body; once the ball has settled on the
  floor it stops being shipped (absent = unchanged, the replayer holds)."""
  _, lines = record(mini_model, seconds=3.0, tmp=tmp_path)
  first, last = lines[1], lines[-1]
  assert "pluggybot" in first["robots"]["pluggybot"]["bodies"]
  assert "ball" in first["world"]
  pose = first["world"]["ball"]
  assert len(pose) == 7 and pose[:3] == [1.0, 0.0, 0.5]
  assert "world" not in last, "a settled body must stop being shipped"


def test_static_scene_sends_no_poses_after_keyframe(mini_model, tmp_path):
  mini_model.opt.gravity[:] = 0                # nothing will ever move
  _, lines = record(mini_model, seconds=1.0, tmp=tmp_path)
  for frame in lines[2:]:
    assert "bodies" not in frame["robots"]["pluggybot"]
    assert "world" not in frame


def test_recorder_status_fn_and_gzip(mini_model, tmp_path):
  """The lifecycle's status dict rides in every frame, and a .gz path
  records through gzip transparently."""
  calls = {"n": 0}

  def status():
    calls["n"] += 1
    return {"state": "EXPLORE", "status": f"frame {calls['n']}",
            "battery": {"frac": 0.5, "watts": 8.5, "charging": False}}

  _, lines = record(mini_model, seconds=1.0, status_fn=status,
                    path=str(tmp_path / "out.jsonl.gz"))
  frames = lines[1:]
  assert calls["n"] == len(frames), "status_fn runs once per frame, not per step"
  assert frames[0]["robots"]["pluggybot"]["state"] == "EXPLORE"
  assert frames[-1]["robots"]["pluggybot"]["status"] == f"frame {len(frames)}"
  assert frames[0]["robots"]["pluggybot"]["battery"]["watts"] == 8.5


# ---- board state and the mixed stream (0.4.0, issue #12) -------------------


def test_the_scene_maps_board_names_to_their_geometry():
  """A `draw` event names a BOARD ("whiteboard_a") and gives points in that
  board's own frame. The geom it lives on is called "board_b". Without the
  scene's board table the client has a polyline it cannot place, and every
  other test here would still pass -- the geometry is present, it is just
  unreachable by the name the events use."""
  model = mujoco.MjModel.from_xml_path(str(REPO / "models" / "home_world.xml"))
  meta = json.loads((REPO / "models" / "home_world.meta.json").read_text())
  scene = scene_dict(model, "home_world", meta=meta)
  geoms = {g["name"] for b in scene["bodies"] for g in b["geoms"]}
  assert scene["boards"], "the home world's drawing surfaces are missing"
  for name, spec in scene["boards"].items():
    assert spec["geom"] in geoms, f"{name} points at a geom nobody renders"
    assert len(spec["half"]) == 3 and len(spec["pos"]) == 3


class FakeBook:
  """The duck type the frame builder wants: `names` + `snapshot()`.

  A stand-in rather than a real BoardBook, because what is under test here is
  the SPARSE-EMISSION contract, not the drawing -- and the contract has to
  hold for whatever the flags happen to be.
  """

  def __init__(self, **boards):
    self.boards = boards

  @property
  def names(self):
    return list(self.boards)

  def snapshot(self):
    return {k: dict(v) for k, v in self.boards.items()}


def test_board_flags_are_sparse_and_re_ship_on_keyframes(mini_model, tmp_path):
  """Ink has no body, so the pose stream says nothing about a drawing at all.

  That makes the boards block the ONLY channel there is -- exactly the
  argument that put activities in the frame -- so it has to obey both halves
  of the rule: quiet when nothing changed, and complete again on every
  keyframe, or a browser that joined mid-mission never learns what is on the
  wall.
  """
  book = FakeBook(whiteboard_a={"strokes": 0, "fill": 0.0})
  data = mujoco.MjData(mini_model)
  builder = FrameBuilder(mini_model, data, model_name="mini", boards=book,
                         keyframe_s=0.5)
  assert builder.header()["boards"] == ["whiteboard_a"]

  frames = []
  for _ in range(round(2.0 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    if float(data.time) > 0.9:
      book.boards["whiteboard_a"] = {"strokes": 3, "fill": 0.21}
    f = builder.build()
    if f is not None:
      frames.append(f)

  assert "boards" in frames[0], "the first frame must carry the whole board"
  quiet = [f for f in frames[1:6] if "boards" in f]
  assert not quiet, f"an unchanged board was re-shipped: {quiet}"
  changed = [f for f in frames if f.get("boards", {}).get(
    "whiteboard_a", {}).get("strokes") == 3]
  assert changed, "a board that changed was never shipped"
  # ...and every keyframe carries it again, whether or not it changed
  for f in [f for f in frames if f.get("key")]:
    assert "boards" in f, f"keyframe at t={f['t']} dropped the board state"


def test_two_sinks_over_one_world_do_not_eat_each_others_deltas(mini_model):
  """`serve.py --record` runs a publisher AND a recorder over one book. The
  already-emitted memory therefore lives on the BUILDER, never on the board --
  shared, each sink would ship a random half of the changes."""
  book = FakeBook(whiteboard_a={"strokes": 0})
  data = mujoco.MjData(mini_model)
  a = FrameBuilder(mini_model, data, model_name="mini", boards=book)
  b = FrameBuilder(mini_model, data, model_name="mini", boards=book)
  assert "boards" in a.build() and "boards" in b.build()
  mujoco.mj_step(mini_model, data)
  book.boards["whiteboard_a"] = {"strokes": 1}
  for _ in range(round(0.1 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
  fa, fb = a.build(), b.build()
  assert fa["boards"] == fb["boards"] == {"whiteboard_a": {"strokes": 1}}


def test_recorded_events_interleave_with_frames(mini_model, tmp_path):
  """A stroke is not a per-tick quantity: decimating one to 20 Hz would mean
  shipping the same polyline a hundred times or dropping it. So it rides as
  its own line, and a recording becomes a MIXED stream -- which is the part of
  0.4.0 a 0.3.0 replayer trips over, since it assumed every line after the
  header was a frame."""
  data = mujoco.MjData(mini_model)
  path = str(tmp_path / "out.jsonl")
  rec = TelemetryRecorder(mini_model, data, path, model_name="mini")
  for step in range(round(1.0 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    rec.step_hook()
    if step == 200:
      rec.emit({"type": "draw", "t": round(float(data.time), 3),
                "board": "whiteboard_a", "points": [[0.0, 0.0], [0.01, 0.0]]})
  rec.close()
  with open(path) as f:
    lines = [json.loads(x) for x in f]

  frames = [x for x in lines[1:] if "type" not in x]
  events = [x for x in lines[1:] if x.get("type") == "draw"]
  assert len(events) == 1 and frames, "the event replaced the frames"
  assert all(b["t"] >= a["t"] for a, b in zip(lines[1:], lines[2:])), \
    "the stream must stay ordered in sim time across both message kinds"
  # the event landed in the middle, not flushed to the end at close()
  assert 0 < lines.index(events[0]) < len(lines) - 1


def test_a_recording_opens_with_the_ink_already_on_the_walls(mini_model,
                                                              tmp_path):
  """0.5.0: a mission recorded against boards that survived a previous run
  opens with the robot standing in front of a drawing it did not make.

  Without the snapshot a replayer paints a blank wall while the `boards`
  block insists the wall is 19 % full -- and nothing later in the stream
  repairs it, because a keyframe re-ships the counters and never the lines.
  """
  from pluggybot.hub.boards import BoardBook, BoardRecord

  data = mujoco.MjData(mini_model)
  book = BoardBook([BoardRecord(name="whiteboard_a", reach=(0.11, 0.2))],
                   clock=lambda: "2026-08-16T00:00:00")
  book.stroke("whiteboard_a", "house", [(0.0, 0.0), (0.02, 0.0), (0.02, 0.02)])
  path = str(tmp_path / "out.jsonl")
  rec = TelemetryRecorder(mini_model, data, path, model_name="mini",
                          boards=book)
  step_seconds = round(0.2 / mini_model.opt.timestep)
  for _ in range(step_seconds):
    mujoco.mj_step(mini_model, data)
    rec.step_hook()
  rec.close()
  lines = [json.loads(x) for x in open(path)]

  assert lines[0]["type"] == "header"
  snaps = [x for x in lines if x.get("type") == "board_snapshot"]
  assert len(snaps) == 1, "the recording never said what was on the board"
  assert lines.index(snaps[0]) == 1, "the snapshot must precede the frames"
  assert snaps[0]["board"] == "whiteboard_a"
  assert len(snaps[0]["strokes"]) == 1
  assert snaps[0]["strokes"][0]["points"][0] == [0.0, 0.0]


def test_screens_ride_the_same_sparse_rule_as_boards(mini_model, tmp_path):
  """0.5.0, and the third block to follow one rule. What makes it worth
  asserting separately: a face is not a pose either, so a screen missing
  from a frame means "unchanged" and never "dark"."""
  from pluggybot.hub.screen import ScreenSet

  class FakeScreen:
    name = "module_lcd"
    flags = {"mode": "face", "powered": True, "face": "idle", "hint": "blink"}

  data = mujoco.MjData(mini_model)
  screens = ScreenSet([FakeScreen()])
  path = str(tmp_path / "out.jsonl")
  rec = TelemetryRecorder(mini_model, data, path, model_name="mini",
                          keyframe_s=0.5, screens=screens)
  for step in range(round(1.2 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    if step == 100:
      FakeScreen.flags = {"mode": "count", "powered": True, "face": "happy",
                          "hint": "none", "count": 4, "label": "plants"}
    rec.step_hook()
  rec.close()
  lines = [json.loads(x) for x in open(path)]
  assert lines[0]["screens"] == ["module_lcd"]
  frames = [x for x in lines[1:] if "type" not in x]
  carrying = [f for f in frames if "screens" in f]
  keyed = [f for f in frames if f.get("key")]
  assert len(carrying) < len(frames), "an unchanged screen was re-sent"
  assert all("screens" in f for f in keyed), "a keyframe skipped the screen"
  counts = [f["screens"]["module_lcd"].get("count") for f in carrying]
  assert 4 in counts


def test_the_ledger_rides_the_same_sparse_rule_and_pays_only_through_it(
    mini_model, tmp_path):
  """0.6.0, and the fourth block to follow one rule (issue #14).

  Two claims in one recording, because they are the two halves of the
  streaming acceptance criterion. The BALANCE rides in the frames -- sparse,
  re-shipped on every keyframe -- so a browser that joins late is caught up
  without a snapshot message of its own; and each award also arrives as an
  `earned` message carrying the verdict behind it, interleaved with the
  frames exactly as `draw` is.
  """
  from pluggybot.hub.ledger import Ledger
  from pluggybot.hub.scoring import evaluate

  data = mujoco.MjData(mini_model)
  ledger = Ledger()
  path = str(tmp_path / "out.jsonl")
  rec = TelemetryRecorder(mini_model, data, path, model_name="mini",
                          keyframe_s=0.5, ledger=ledger)
  ledger.on_event.append(rec.emit)
  for step in range(round(1.2 / mini_model.opt.timestep)):
    mujoco.mj_step(mini_model, data)
    if step == 100:
      ledger.award(evaluate("carry", {"picked": True, "stowed": True,
                                      "module": "module_lcd"}),
                   t=float(data.time))
    rec.step_hook()
  rec.close()
  lines = [json.loads(x) for x in open(path)]
  assert lines[0]["ledger"] == ["pluggybot"]
  frames = [x for x in lines[1:] if "type" not in x]
  carrying = [f for f in frames if "ledger" in f]
  keyed = [f for f in frames if f.get("key")]
  assert len(carrying) < len(frames), "an unchanged balance was re-sent"
  assert all("ledger" in f for f in keyed), "a keyframe skipped the balance"
  balances = [f["ledger"]["pluggybot"]["balance"] for f in carrying]
  assert balances[0] == 0 and balances[-1] == ledger.balance() > 0
  earned = [x for x in lines[1:] if x.get("type") == "earned"]
  assert len(earned) == 1 and earned[0]["task"] == "carry"
  assert earned[0]["points"] == ledger.balance() and earned[0]["reason"]


# ---- what the robot is for (0.8.0, rooftop-media-2026 #30) -----------------

def test_a_recording_opens_by_saying_what_the_robot_is_for(mini_model,
                                                            tmp_path):
  """0.8.0: the goals prose rides the `board_snapshot` slot, for its reason.

  Goals are not a pose and no keyframe re-ships them, so this one line is
  the only place in the whole stream a reader can learn them. A recording
  that emits it after the frames, or not at all, leaves the site's goals
  panel permanently blank.
  """
  rec, lines = record(mini_model, seconds=0.4, tmp=tmp_path,
                      goals="Keep the house in good order.", steering=True)
  goals = [x for x in lines if x.get("type") == "goals"]
  assert len(goals) == 1, "the recording never said what the robot is for"
  assert lines.index(goals[0]) == 1, "goals must precede the frames"
  assert goals[0]["text"] == "Keep the house in good order."
  assert goals[0]["robot"] == "pluggybot"
  assert goals[0]["steering"] is True
  # It is emitted ONCE. A per-frame block would put up to MAX_GOALS_CHARS of
  # unchanging prose on the wire twenty times a second.
  assert all("goals" not in f for f in lines if "type" not in f)


def test_goals_say_whether_anything_is_actually_reading_them(mini_model,
                                                             tmp_path):
  """The `accepts` lesson, applied to the other end of the same loop.

  The goals file is read on EVERY run, overseer or not -- so a producer that
  streamed the prose without saying which of the two this is lets the site
  report "following its goals" about a robot flying a scripted rotation with
  nothing reading them. `steering` is the whole difference, and it defaults
  to the honest answer.
  """
  _, scripted = record(mini_model, seconds=0.4, tmp=tmp_path,
                       goals="Water the garden.")
  msg = next(x for x in scripted if x.get("type") == "goals")
  assert msg["steering"] is False, \
    "a scripted rotation claimed an overseer was steering by these"

  # ...and a run with no goals at all emits no message, rather than an empty
  # one a consumer would render as a robot that wants nothing.
  _, silent = record(mini_model, seconds=0.4, tmp=tmp_path, goals="")
  assert not [x for x in silent if x.get("type") == "goals"]


def test_the_goals_file_is_read_whether_or_not_an_overseer_runs():
  """`overseer.build` answers (None, None) when disabled, which is why the
  telemetry path cannot get its prose from there. It reads the file itself.
  """
  from pluggybot.hub import overseer as ov

  assert ov.build("home", enabled=False) == (None, None)
  # ...and yet there is prose to stream, which is the whole point of the
  # helper: a scripted mission still has a purpose to display.
  assert ov.goals_text(None).strip()


def test_a_live_consumer_is_told_the_goals_on_every_connect(mini_model):
  """A goals message per CONNECT, not per stream.

  Same argument as the board snapshots beside it: a browser that opens the
  page an hour into a mission has missed the only line that carried them,
  and the hub relays rather than re-keys on its behalf. So the publisher
  re-sends on connect -- and the flag lives on the physics thread, because
  that is the thread that owns the clock the message is stamped with.
  """
  from pluggybot.telemetry.publisher import WsPublisher

  data = mujoco.MjData(mini_model)
  pub = WsPublisher.__new__(WsPublisher)          # no socket, no sender thread
  pub._builder = FrameBuilder(mini_model, data, model_name="mini",
                              goals="Tidy the blocks.", steering=False)
  pub.data = data
  pub._queue = queue.Queue(maxsize=64)
  pub._need_goals = threading.Event()
  pub._need_boards = threading.Event()
  pub._need_keyframe = threading.Event()
  pub.boards = None
  pub.grid = None
  pub.frames_dropped = 0
  pub.events_dropped = 0

  pub.step_hook()
  assert not _typed(pub._queue, "goals"), "goals went out with nobody connected"

  pub._need_goals.set()                            # ...as the sender does on connect
  pub.step_hook()
  first = _typed(pub._queue, "goals")
  assert len(first) == 1 and first[0]["text"] == "Tidy the blocks."

  pub.step_hook()
  assert not _typed(pub._queue, "goals"), "goals repeat on every physics step"

  pub._need_goals.set()                            # ...and a reconnect
  pub.step_hook()
  assert len(_typed(pub._queue, "goals")) == 1


def _typed(q, kind):
  """Drain a publisher queue, returning the typed messages of one kind."""
  out = []
  while True:
    try:
      k, payload = q.get_nowait()
    except queue.Empty:
      return out
    if k == "event" and payload.get("type") == kind:
      out.append(payload)


# ---- the committed fixtures ------------------------------------------------
# The same checks the website repo runs against its vendored copies: if these
# fail, regenerate the fixtures or bump the protocol version deliberately.

PROTOCOL = REPO / "protocol"

# Every world the website can render needs BOTH artifacts: a scene to build
# and a recording to replay in it. Serving one world's telemetry against the
# other's scene is the failure this table exists to make impossible.
WORLDS = [
  # (scene fixture, model, model name, generator sidecar, recording, draws?)
  # `draws` marks the recording that must exercise the DRAWING errand: the
  # home world is the one the website serves, so its fixture is what the
  # canvas-painting code on the other side is built against (issue #12).
  ("scene.room_hub.json", "room_hub.xml", "room_hub", None,
   "telemetry.hub_lifecycle.jsonl.gz", False),
  ("scene.home_world.json", "home_world.xml", "home_world",
   "home_world.meta.json", "telemetry.home_lifecycle.jsonl.gz", True),
]
SCENE_CASES = [(w[0], w[1], w[2], w[3]) for w in WORLDS]
TELEMETRY_CASES = [(w[4], w[2], w[5]) for w in WORLDS]


@pytest.mark.parametrize("fixture,world_xml,model_name,meta_file", SCENE_CASES,
                         ids=[c[2] for c in SCENE_CASES])
def test_scene_fixture_current(fixture, world_xml, model_name, meta_file):
  """A committed scene must match the committed WORLD, not an old one.

  home_world's is the one that rots fastest: the world is GENERATED, so a
  layout constant changed in home/world.py and regenerated into models/
  leaves this fixture describing last week's house -- and the website
  renders walls where the robot no longer sees them.
  """
  path = PROTOCOL / fixture
  scene = json.loads(path.read_text())
  assert scene["protocolVersion"] == PROTOCOL_VERSION
  assert scene["model"] == model_name and scene["upAxis"] == "z"
  for tex in scene["textures"]:
    assert (path.parent / "textures" / tex["file"]).exists()
  meta = (json.loads((REPO / "models" / meta_file).read_text())
          if meta_file else None)
  model = mujoco.MjModel.from_xml_path(str(REPO / "models" / world_xml))
  assert scene == scene_dict(model, model_name, meta=meta), \
    f"stale fixture: uv run python -m pluggybot.telemetry.scene " \
    f"models/{world_xml}"


def test_the_home_fixture_shows_the_census_answer():
  """An errand's RESULT has to survive at least one frame (issue #13).

  Python between two physics steps takes zero sim time, so a use-phase that
  sets the screen and returns has its result overwritten by the next state's
  automatic face before a single 20 Hz frame is built. The first version of
  the census did exactly that: the recorded showcase mission carried the
  right answer in its result dict and in NONE of its 10 850 frames, which is
  the same as not having computed it -- the website renders the wire, not the
  dict. Delete the `_drive(PRESENT_S, 0, 0)` hold and this is what fails.
  """
  with gzip.open(PROTOCOL / "telemetry.home_lifecycle.jsonl.gz", "rt") as f:
    lines = [json.loads(line) for line in f]
  frames = [x for x in lines[1:] if "type" not in x]
  shown = [(fr["t"], s) for fr in frames
           for s in (fr.get("screens") or {}).values()]
  counts = [(t, s) for t, s in shown if s["mode"] == "count"]
  assert counts, "the census answer never reached a frame"
  assert counts[0][1]["label"] == "plants"
  assert isinstance(counts[0][1]["count"], int)

  # ...and it stayed up long enough to read. Measured as a DURATION, not as a
  # frame count: the block is sparse, so a state that persists for five
  # seconds appears once and is merely re-shipped on the next keyframe. The
  # first version of this assertion counted frames and failed on a working
  # fix, which is its own small lesson about sparse encodings.
  first = counts[0][0]
  after = [t for t, s in shown if t > first and s["mode"] != "count"]
  held = (after[0] if after else frames[-1]["t"]) - first
  assert held >= 4.0, f"the answer was on screen for {held:.1f} sim-seconds"


@pytest.mark.parametrize("fixture,model_name,draws", TELEMETRY_CASES,
                         ids=[c[1] for c in TELEMETRY_CASES])
def test_telemetry_fixture_is_a_full_mission(fixture, model_name, draws):
  with gzip.open(PROTOCOL / fixture, "rt") as f:
    lines = [json.loads(line) for line in f]
  # Dispatch on "type"; no "type" means frame (0.4.0). A recording is a MIXED
  # stream now -- `draw` and `board_cleared` events ride between the frames,
  # because a stroke is not a per-tick quantity.
  header = lines[0]
  frames = [x for x in lines[1:] if "type" not in x]
  assert header["protocolVersion"] == PROTOCOL_VERSION
  # the header field the website selects its scene off -- a recording
  # mislabelled here poses one world's robot inside the other's rooms
  assert header["model"] == model_name
  robot_names = set(header["robots"]["pluggybot"])
  first = frames[0]["robots"]["pluggybot"]
  assert set(first["bodies"]) == robot_names, "first frame must be a keyframe"
  assert set(frames[0]["world"]) == set(header["world"])

  # Keyframes recur (0.2.0): the website's relay hub caches "last keyframe
  # + frames since" to serve a browser that joins mid-mission, so a
  # recording without them would be testing a stream shape we never send.
  keys = [f for f in frames if f.get("key")]
  assert frames[0].get("key") is True
  assert len(keys) > 1, "the fixture must exercise RECURRING keyframes"
  for f in keys:
    assert set(f["robots"]["pluggybot"]["bodies"]) == robot_names
    assert set(f["world"]) == set(header["world"])
  gaps = [b["t"] - a["t"] for a, b in zip(keys, keys[1:])]
  # the cadence re-anchors on the frame that carried the keyframe, so it can
  # run one frame interval late -- but no more, or it has silently regressed
  assert max(gaps) <= header["keyframeS"] + 2.0 / header["hz"], \
    f"keyframe spacing drifted past the advertised cadence: max {max(gaps):.2f} s"
  times = [f["t"] for f in frames]
  assert all(b > a for a, b in zip(times, times[1:]))
  states = {f["robots"]["pluggybot"]["state"] for f in frames}
  assert {"EXPLORE", "GO_CHARGE", "CHARGE",
          "SWAP_PICK", "USE_TOOL", "SWAP_RETURN"} <= states, \
    "the fixture must cover the full battery-driven mission"

  # What the robot is FOR (0.8.0). The site opens on a recording rather than
  # on a live sim -- that is what `replay` is for -- so if the fixtures do not
  # carry this line the goals panel is blank in the case a visitor actually
  # meets. It is also the one message with no keyframe behind it: lose it and
  # nothing later in the stream repairs it.
  events = [x for x in lines[1:] if "type" in x]
  goals = [e for e in events if e["type"] == "goals"]
  assert len(goals) == 1, "the fixture never says what the robot is for"
  assert goals[0]["text"].strip(), "an empty statement of purpose"
  # ...and it says so honestly: these missions run the scripted rotation, so
  # nothing is reading the goals and the fixture must not claim otherwise.
  assert goals[0]["steering"] is False
  assert lines.index(goals[0]) <= 2, "goals must precede the frames"

  # The scoreboard half (0.6.0). Every mission charges, and charging is a
  # scored task, so BOTH worlds' fixtures must carry a ledger that moves --
  # this is what the site's scoreboard is built against, and the balance is
  # the only part of it that survives a mid-mission join (there is no
  # snapshot message for points; `recent` in the keyframe is the catch-up).
  events = [x for x in lines[1:] if "type" in x]
  assert header["ledger"] == ["pluggybot"]
  banked = [e for e in events if e["type"] == "earned"]
  assert banked, "a whole mission earned nothing: no task was ever evaluated"
  assert {"charge"} <= {e["task"] for e in banked}
  for e in banked:
    assert e["reason"] and e["tier"] in ("auto", "hidden", "visitor")
    # A hidden-truth task must not publish its answer: this stream reaches
    # both the website and (issue #15) the robot's own context.
    assert "truth" not in e["metrics"]
  with_ledger = [f for f in frames if "ledger" in f]
  assert len(with_ledger) < len(frames), "an unchanged balance was re-sent"
  # ...and the block agrees with the events it summarizes. Not necessarily
  # with the LAST one: a mission whose final award lands on its final physics
  # step ends before another frame is built, and a fixture is not a worse
  # fixture for that.
  final = with_ledger[-1]["ledger"]["pluggybot"]
  assert 0 < final["balance"] <= banked[-1]["balance"]
  assert final["balance"] in {e["balance"] for e in banked}
  assert final["recent"] and final["tasks"] >= len(final["recent"])

  if not draws:
    return
  # The drawing half (0.4.0): the board is erased, then inked, and BOTH facts
  # reach a consumer only through these lines and the boards block. There is
  # no body to watch -- a fixture that lost them replays a robot miming at a
  # blank wall, and every assertion above would still pass.
  events = [x for x in lines[1:] if "type" in x]
  cleared = [e for e in events if e["type"] == "board_cleared"]
  drawn = [e for e in events if e["type"] == "draw"]
  # The drawing errand is scored too, and on a real mission rather than on a
  # measurement handed in by a test: the strokes that paid are the ones the
  # pen wrote into the board book (issue #14).
  draw_award = next(e for e in banked if e["task"] == "draw")
  assert draw_award["ok"] and draw_award["points"] > 0
  assert draw_award["metrics"]["strokesInked"] == len(drawn)
  assert cleared, "no board_cleared event: the errand must erase before it draws"
  assert len(drawn) > 1, f"only {len(drawn)} draw events in a whole drawing"
  assert set(header["boards"]), "the header must name the world's boards"
  for e in drawn:
    assert e["board"] in header["boards"]
    assert len(e["points"]) >= 2 and all(len(p) == 2 for p in e["points"])
  assert cleared[0]["t"] < drawn[0]["t"], "drew before erasing"
  # ...and the board state itself moved off blank, in the frames
  final = [f for f in frames if "boards" in f][-1]["boards"]
  assert any(b["strokes"] > 0 and b["fill"] > 0 for b in final.values()), \
    "the boards block never showed any ink"
  for f in frames:
    bat = f["robots"]["pluggybot"]["battery"]
    assert 0.0 <= bat["frac"] <= 1.0
    for pose in list(f["robots"]["pluggybot"].get("bodies", {}).values()) \
        + list(f.get("world", {}).values()):
      assert len(pose) == 7
