"""PluggyWorld protocol guards (webserver v0): transpiler + recorder.

The transpiler's conversions are exactly the kind of pure math that renders
wrong silently -- a half-extent box is a quarter the volume and still looks
like a box, a Z-axis cylinder rendered on Y is a fallen column. Each
conversion is asserted numerically here, on a tiny inline model where the
right answer is known by construction, plus a coverage pass over the served
house. The recorder tests drive the real seam contract: decimation,
keyframe-then-sparse frames, and everything queued reaching the file.
"""

import gzip
import json
import math
import queue
import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pluggybot.telemetry.protocol import (ENCOUNTER_PHASES, PROTOCOL_VERSION,
                                          body_census, dynamic_flags)
from pluggybot.telemetry.recorder import (FrameBuilder,
                                          TelemetryRecorder)
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


def test_the_served_house_transpiles_whole():
  """Every visible geom of the served world transpiles, all five primitive
  types are exercised, and every AprilTag texture is referenced."""
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  scene = scene_dict(model, QUAD_HOME, meta=json.loads(Path(cfg["meta"]).read_text()))
  assert scene["protocolVersion"] == PROTOCOL_VERSION
  geoms = [g for b in scene["bodies"] for g in b["geoms"]]
  visible = sum(1 for g in range(model.ngeom) if model.geom_rgba[g][3] > 0)
  assert len(geoms) == visible
  assert {g["type"] for g in geoms} == {"plane", "box", "cylinder",
                                        "capsule", "sphere"}
  referenced = {g["texture"] for g in geoms if g["texture"]}
  assert len(referenced) == model.ntex        # every tag in use
  assert {t["name"] for t in scene["textures"]} == referenced
  robot, world = body_census(model)
  # 34 = the quadruped's 19 links + 15 of the world's: the tools, the
  # tower's blocks and the bench's cubes, the lab's plates and its mouse,
  # the garden's plate and the dock's pins. A census, so it fails whenever
  # the world gains or loses a dynamic body -- which is the point: every
  # one of them costs a pose in every keyframe.
  assert sum(dynamic_flags(model)) == len(robot) + len(world) == 34
  assert len(robot) == 19
  assert {"module_lcd", "module_pen", "module_claw", "block_0", "lab_mouse",
          "dock_pole_l"} <= set(world)


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


def frames_of(lines):
  """Just the pose frames: no `type` means frame, which is the dispatch rule
  the protocol has had since 0.4.0.

  ⚠ FILTER, NEVER SLICE. These tests used `lines[1:]` and `lines[2:]`, which
  silently became wrong the moment the recording opened with one more typed
  message than it used to -- 0.19.0's always-emitted `goals` line did exactly
  that. A count of frames should not depend on how many things precede them.
  """
  return [ln for ln in lines if "type" not in ln]


def test_recorder_honours_the_keyframe_cadence(mini_model, tmp_path):
  """The recorder's keyframe_s must reach its builder: `serve.py --record`
  writes a recording of the SAME run it streams, so a cadence that applied
  to one and not the other would make the two artifacts disagree."""
  rec, lines = record(mini_model, seconds=3.0, tmp=tmp_path, keyframe_s=0.5)
  header, frames = lines[0], frames_of(lines)
  assert header["keyframeS"] == 0.5
  keys = [f for f in frames if f.get("key")]
  assert len(keys) >= 5, f"expected ~6 keyframes over 3 s, got {len(keys)}"
  for f in keys:
    assert set(f["robots"]["pluggybot"]["bodies"]) == set(header["robots"]["pluggybot"])
    assert set(f["world"]) == set(header["world"])


def test_recorder_header_and_decimation(mini_model, tmp_path, monkeypatch):
  monkeypatch.delenv("PLUGGY_ROBOT_NAME", raising=False)
  rec, lines = record(mini_model, seconds=2.0, tmp=tmp_path)
  header, frames = lines[0], frames_of(lines)
  assert header["type"] == "header"
  assert header["protocolVersion"] == PROTOCOL_VERSION
  assert header["robots"] == {"pluggybot": ["pluggybot"]}
  # An unconfigured run still has a NAME (0.10.0, issue #39): absent config
  # degrades to a default, never to a blank identity header on the site.
  assert header["robotNames"] == {"pluggybot": "Pluggy"}
  assert header["world"] == ["ball"]
  # ~20 Hz of sim time out of 500 Hz of steps, spacing never under 1/hz
  assert len(frames) == pytest.approx(2.0 * header["hz"], abs=2)
  times = [f["t"] for f in frames]
  assert all(b - a >= 1 / header["hz"] - 1e-6 for a, b in zip(times, times[1:]))
  # everything queued reached the file: close() drains before returning
  assert len(frames) == rec.frames


def test_a_named_robot_re_keys_nothing(mini_model, tmp_path):
  """The name is IDENTITY, and the species is the KEY (0.10.0, issue #39).

  'Luca the pluggybot' must put Luca in `robotNames` and change nothing
  else: `robots`, the frame keys and ROBOT_ROOT itself all stay the MJCF
  body name, because every body-name-keyed structure (body_census, the
  scene transpiler, the ledger, every fixture) rides on it. A rename that
  re-keyed telemetry would orphan every consumer mid-stream.
  """
  from pluggybot.telemetry.protocol import ROBOT_ROOT
  rec, lines = record(mini_model, seconds=1.0, tmp=tmp_path,
                      robot_name="Luca")
  header, frames = lines[0], frames_of(lines)
  assert header["robotNames"] == {"pluggybot": "Luca"}
  assert ROBOT_ROOT == "pluggybot"
  assert header["robots"] == {"pluggybot": ["pluggybot"]}
  assert all(set(f["robots"]) == {"pluggybot"} for f in frames)
  # ...and the name rides the header alone: 20 Hz frames repeat what can
  # change, and a name cannot.
  assert all("robotNames" not in f for f in frames)


def test_the_name_comes_from_the_environment_and_degrades_to_a_default(
    mini_model, tmp_path, monkeypatch):
  """$PLUGGY_ROBOT_NAME names a deployed sim without a rebuild (the boards/
  ledger/energy pattern); an explicit name beats it; blank degrades to the
  default rather than putting an empty identity on the wire."""
  from pluggybot.telemetry.protocol import robot_display_name
  monkeypatch.setenv("PLUGGY_ROBOT_NAME", "Beryl")
  _, lines = record(mini_model, seconds=0.2, tmp=tmp_path)
  assert lines[0]["robotNames"] == {"pluggybot": "Beryl"}
  assert robot_display_name("Luca") == "Luca"      # flag beats env
  monkeypatch.setenv("PLUGGY_ROBOT_NAME", "   ")
  assert robot_display_name() == "Pluggy"          # blank is not a name
  monkeypatch.delenv("PLUGGY_ROBOT_NAME")
  assert robot_display_name("") == "Pluggy"
  # A runaway env var is a loud config error, not a broken site layout.
  with pytest.raises(ValueError):
    robot_display_name("x" * 61)


def test_first_frame_is_keyframe_then_sparse(mini_model, tmp_path):
  """Frame 0 carries every dynamic body; once the ball has settled on the
  floor it stops being shipped (absent = unchanged, the replayer holds)."""
  _, lines = record(mini_model, seconds=3.0, tmp=tmp_path)
  first, last = frames_of(lines)[0], frames_of(lines)[-1]
  assert "pluggybot" in first["robots"]["pluggybot"]["bodies"]
  assert "ball" in first["world"]
  pose = first["world"]["ball"]
  assert len(pose) == 7 and pose[:3] == [1.0, 0.0, 0.5]
  assert "world" not in last, "a settled body must stop being shipped"


def test_static_scene_sends_no_poses_after_keyframe(mini_model, tmp_path):
  mini_model.opt.gravity[:] = 0                # nothing will ever move
  _, lines = record(mini_model, seconds=1.0, tmp=tmp_path)
  for frame in frames_of(lines)[1:]:
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
  frames = frames_of(lines)
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
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.robot import world_spec
  cfg = world_config(QUAD_HOME)
  model = world_spec(cfg["model"], body=cfg["body"]).compile()
  scene = scene_dict(model, QUAD_HOME, meta=json.loads(Path(cfg["meta"]).read_text()))
  geoms = {g["name"] for b in scene["bodies"] for g in b["geoms"]}
  assert scene["boards"], "the house's drawing surfaces are missing"
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
  from pluggybot.tools.boards import BoardBook, BoardRecord

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
  assert lines.index(snaps[0]) < lines.index(frames_of(lines)[0]), \
    "the snapshot must precede the frames"
  assert snaps[0]["board"] == "whiteboard_a"
  assert len(snaps[0]["strokes"]) == 1
  assert snaps[0]["strokes"][0]["points"][0] == [0.0, 0.0]


def test_screens_ride_the_same_sparse_rule_as_boards(mini_model, tmp_path):
  """0.5.0, and the third block to follow one rule. What makes it worth
  asserting separately: a face is not a pose either, so a screen missing
  from a frame means "unchanged" and never "dark"."""
  from pluggybot.tools.screen import ScreenSet

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
  from pluggybot.economy.ledger import Ledger
  from pluggybot.economy.scoring import evaluate

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
  assert lines.index(goals[0]) < lines.index(frames_of(lines)[0]), \
    "goals must precede the frames"
  assert goals[0]["text"] == "Keep the house in good order."
  assert goals[0]["robot"] == "pluggybot"
  assert goals[0]["steering"] is True
  # It is emitted ONCE. A per-frame block would put up to MAX_GOALS_CHARS of
  # unchanging prose on the wire twenty times a second.
  assert all("goals" not in f for f in lines if "type" not in f)


def test_a_mindless_mode_stops_advertising_what_only_a_mind_can_hear(mini_model,
                                                                    tmp_path):
  """The `accepts` lesson, applied to the operator's switch (issue #37).

  `accepts` is fixed at construction from whether an overseer was BUILT, but
  `scripted` hands deciding back to the rotation and `paused` stops it
  altogether -- and in neither does anything read a visitor's message.
  Advertising
  the full vocabulary there reintroduces exactly what the field exists to
  prevent: a site marking a message "delivered" to a robot with nothing
  listening. A rating and a reset are CODE's to apply, so they survive every
  mode.
  """
  from pluggybot.mind.mode import ModeSwitch
  from pluggybot.telemetry.protocol import CODE_HANDLED_TYPES, INBOUND_TYPES

  data = mujoco.MjData(mini_model)
  path = tmp_path / "mode.json"

  def hears(mode: str, advertised=INBOUND_TYPES) -> tuple:
    path.write_text(json.dumps({"mode": mode}))
    builder = FrameBuilder(mini_model, data, model_name="mini",
                           accepts=advertised, mode=ModeSwitch(path))
    return tuple(builder.header()["accepts"])

  assert hears("llm") == INBOUND_TYPES
  assert hears("scripted") == CODE_HANDLED_TYPES, \
    "free mode promised a conversation the rotation cannot have"
  assert hears("paused") == CODE_HANDLED_TYPES, \
    "a paused robot promised to act on a visitor's message"

  # A world that never had a mind advertises the same thing in every mode --
  # the narrowing only ever removes, so it cannot invent an ability.
  assert hears("llm", CODE_HANDLED_TYPES) == CODE_HANDLED_TYPES

  # ...and with no switch at all (every run before 0.12.0, and every
  # recording) the advertised list is passed through untouched.
  bare = FrameBuilder(mini_model, data, model_name="mini",
                      accepts=INBOUND_TYPES)
  assert tuple(bare.header()["accepts"]) == INBOUND_TYPES


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

  # ⚠ ...AND A RUN WITH NO GOALS STILL EMITS THE MESSAGE (0.19.0, issue
  # #154). `Goals.md` is the ROBOT's now and starts empty, so "absent when
  # empty" would take this message off every scripted world -- and `steering`
  # rides HERE AND NOWHERE ELSE, so the flag saying whether anything is
  # deciding would vanish from exactly the streams it describes. An empty
  # `text` with `steering: false` says both things at once.
  _, silent = record(mini_model, seconds=0.4, tmp=tmp_path, goals="")
  empty = [x for x in silent if x.get("type") == "goals"]
  assert len(empty) == 1, "the steering flag went missing with the prose"
  assert empty[0]["text"] == "" and empty[0]["steering"] is False


def test_the_goals_file_is_read_whether_or_not_an_overseer_runs(tmp_path):
  """`overseer.build` answers (None, None) when disabled, which is why the
  telemetry path cannot get its prose from there. It reads the file itself.

  ⚠ AND SINCE ISSUE #154 IT IS USUALLY EMPTY, which is the honest answer
  rather than a gap: `Goals.md` is the ROBOT's now and starts blank, so a
  scripted world has no goals to stream and never will. What a scripted
  world still has is a CONSTITUTION, and that rides a `thought` document
  like the other three. The helper must keep READING the right file either
  way -- returning the constitution here would report a person's hopes as
  the robot's own goals.
  """
  from pluggybot.mind import overseer as ov
  from pluggybot.mind.thoughts import ThoughtFiles

  assert ov.build("home", enabled=False) is None
  assert ov.goals_text(None) == "", "a fresh robot reported goals it never set"
  # ...and when the robot HAS set one, that is what the stream carries.
  files = ThoughtFiles(tmp_path / "thoughts")
  files.intend("ink both boards this week", t=1.0)
  assert ov.goals_text(thoughts=files).strip() == "ink both boards this week"


def test_a_recording_opens_with_the_robots_memory(mini_model, tmp_path):
  """0.11.0: the thought files ride the `goals` slot, four times over
  (pluggybot #38).

  Same argument as goals and board snapshots: a document is not a pose and
  no keyframe re-ships one, so these lines are the only place in the stream
  a reader learns what the robot is working from. A recording that emitted
  them after the frames -- or not at all -- leaves the site's Thoughts tab
  permanently empty, and that is the case almost every visitor meets,
  because the default view is a recording.
  """
  from pluggybot.mind.thoughts import TOP_OF_MIND, NAMES, ThoughtFiles

  memory = ThoughtFiles()
  memory.pin("whiteboard_b is the one people look at")
  _, lines = record(mini_model, seconds=0.4, tmp=tmp_path,
                    goals="Keep the house in good order.", thoughts=memory)
  thoughts = [x for x in lines if x.get("type") == "thought"]
  assert [t["name"] for t in thoughts] == list(NAMES)
  first_frame = next(i for i, x in enumerate(lines) if "type" not in x)
  assert all(lines.index(t) < first_frame for t in thoughts), \
    "the documents arrived after the frames that depend on them"
  by_name = {t["name"]: t for t in thoughts}
  assert by_name[TOP_OF_MIND]["text"] == "whiteboard_b is the one people look at"
  # Each one says who may write it -- four panels that look alike on a page
  # and are not alike at all (the `steering` lesson, one loop over).
  assert {t["writer"] for t in thoughts} == {"human", "system", "robot"}
  # ...and `goals` is UNCHANGED and still sent: it is the only carrier of
  # `steering`, which says who is READING rather than who may write.
  assert len([x for x in lines if x.get("type") == "goals"]) == 1
  # A run with no files at all emits none, on the same terms as goals.
  _, silent = record(mini_model, seconds=0.4, tmp=tmp_path, goals="")
  assert not [x for x in silent if x.get("type") == "thought"]


class _MappedMind:
  """A stand-in overseer with a map and a prompt: the two methods the
  builder reads on open."""
  def __init__(self, rows):
    self.rows = rows

  def event_map_message(self, t, robot):
    return {"type": "event_map", "t": t, "robot": robot, "origin": "unseeded",
            "why": "origin", "source": None, "edits": 0, "rows": self.rows}

  def prompt_message(self, t, robot):
    return {"type": "prompt", "t": t, "robot": robot, "sha": "abc",
            "sections": [{"name": "WHO YOU ARE", "text": "WHO YOU ARE\n\nx"}]}


def test_a_recording_opens_with_the_rows_and_the_map(mini_model, tmp_path):
  """Issue #238: beside the documents, the ROWS they are rendered from (one
  `records` snapshot per robot) and the mind's event map, all before the
  first frame for the reason every message in this slot is there -- no
  keyframe re-ships one. A world with no map opens with no `event_map`
  line, which is what every committed fixture is."""
  from pluggybot.mind.thoughts import ThoughtFiles

  memory = ThoughtFiles()
  memory.pin("whiteboard_b is the one people look at", t=0.0)
  memory.remember("woke up", t=0.0)
  mind = _MappedMind([{"event": "nothing_to_do", "action": "ask"}])
  _, lines = record(mini_model, seconds=0.4, tmp=tmp_path, thoughts=memory,
                    overseer=mind)
  first_frame = next(i for i, x in enumerate(lines) if "type" not in x)
  opening = [x.get("type") for x in lines[1:first_frame]]
  assert opening.index("records") > opening.index("thought")
  assert "event_map" in opening
  snap = next(x for x in lines if x.get("type") == "records")
  assert snap["robot"] == "pluggybot" and snap["generation"] == 1
  assert [r["kind"] for r in snap["records"]] == ["core", "history"]
  emap = next(x for x in lines if x.get("type") == "event_map")
  assert emap["robot"] == "pluggybot" and emap["rows"] == mind.rows
  # ...and what the mind is TOLD (issue #241), in the same slot: once, on
  # open, before the frames.
  assert opening.count("prompt") == 1
  prompt = next(x for x in lines if x.get("type") == "prompt")
  assert prompt["robot"] == "pluggybot" and prompt["sha"] == "abc"
  assert [s["name"] for s in prompt["sections"]] == ["WHO YOU ARE"]
  # ...and a scripted world -- no mind -- has no map line and no prompt at
  # all: nothing is being told anything, which is not an empty prompt.
  _, plain = record(mini_model, seconds=0.4, tmp=tmp_path, thoughts=ThoughtFiles())
  assert not [x for x in plain if x.get("type") in ("event_map", "prompt")]
  assert len([x for x in plain if x.get("type") == "records"]) == 1


def test_a_live_consumer_is_told_the_memory_on_every_connect(mini_model):
  """A thought message per CONNECT, like the goals beside it -- a browser
  that opened the page an hour in has missed the only lines that carried
  them, and the hub relays rather than re-keys on its behalf."""
  from pluggybot.mind.thoughts import NAMES, ThoughtFiles
  from pluggybot.telemetry.publisher import WsPublisher

  data = mujoco.MjData(mini_model)
  pub = WsPublisher.__new__(WsPublisher)          # no socket, no sender thread
  pub._builder = FrameBuilder(mini_model, data, model_name="mini",
                              thoughts=ThoughtFiles(),
                              overseer=_MappedMind([]))
  pub.data = data
  pub._queue = queue.Queue(maxsize=64)
  pub._need_goals = threading.Event()
  pub._need_thoughts = threading.Event()
  pub._need_boards = threading.Event()
  pub._need_keyframe = threading.Event()
  pub.boards = None
  pub._grids = []
  pub.frames_dropped = 0
  pub.events_dropped = 0
  pub.events_queued = pub.events_sent = 0

  pub.step_hook()
  assert not _drain(pub._queue), "sent with nobody connected"

  pub._need_thoughts.set()                         # ...as the sender does
  pub.step_hook()
  sent = _drain(pub._queue)
  assert [m["name"] for m in sent if m["type"] == "thought"] == list(NAMES)
  # ...with the rows, the map and the prompt behind them (issues #238,
  # #241), same connect.
  assert [m["type"] for m in sent if m["type"] in ("records", "event_map", "prompt")] \
      == ["records", "event_map", "prompt"]

  pub.step_hook()
  assert not _drain(pub._queue), "repeated on every physics step"


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
  pub._need_thoughts = threading.Event()
  pub._need_boards = threading.Event()
  pub._need_keyframe = threading.Event()
  pub.boards = None
  pub._grids = []
  pub.frames_dropped = 0
  pub.events_dropped = 0
  pub.events_queued = pub.events_sent = 0

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


def _drain(q):
  """Drain a publisher queue, returning every typed message."""
  out = []
  while True:
    try:
      k, payload = q.get_nowait()
    except queue.Empty:
      return out
    if k == "event":
      out.append(payload)


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
#: The one recording: the served pair on legs, in the first quadruped
#: period's shape (no offers, no upkeep).
PAIR_RECORDING = PROTOCOL / "telemetry.home_quad_pair.jsonl.gz"


def _pair_recording() -> list[dict]:
  with gzip.open(PAIR_RECORDING, "rt") as f:
    return [json.loads(line) for line in f]


def test_the_pair_recording_gives_every_robot_the_same_shape():
  """The pair fixture (0.20.0): two robots from one loop, and everything
  the wire keys by robot present for BOTH -- bodies, status, goals,
  documents, map. The served shape (#181, #387): the first robot walks to
  its dock and charges, the second explores."""
  from pluggybot.mind.thoughts import NAMES
  from pluggybot.robot import FIRST, SECOND
  lines = _pair_recording()
  header, frames = lines[0], frames_of(lines)
  events = [x for x in lines[1:] if "type" in x]
  roots = [FIRST.root, SECOND.root]
  assert header["protocolVersion"] == PROTOCOL_VERSION
  assert header["model"] == "home_quad_pair"
  assert list(header["robots"]) == roots
  assert header["robotNames"] == {FIRST.root: "Pluggy", SECOND.root: "Rowan"}
  assert header["robots"][SECOND.root] == [SECOND.el(n) for n in header["robots"][FIRST.root]]
  assert header["ledger"] == roots
  assert "encounters" in header["activities"]
  # Every frame carries both; the keyframes carry both bodies whole.
  assert all(set(f["robots"]) == set(roots) for f in frames)
  for root in roots:
    assert set(frames[0]["robots"][root]["bodies"]) == set(header["robots"][root])
    states = {f["robots"][root]["state"] for f in frames}
    assert "DEAD" not in states, f"{root} died"
    # ...and each moved: the FARTHEST it got from where it started, since a
    # robot standing by goes back to its own start pose
    xy = [f["robots"][root]["bodies"][root][:2] for f in frames
          if root in f["robots"][root].get("bodies", {})]
    travelled = max(math.dist(xy[0], p) for p in xy)
    assert travelled > 1.0, f"{root} barely moved: {travelled:.2f} m"
  # One goals message per robot, and every document for each, and a map.
  goals = [e for e in events if e["type"] == "goals"]
  assert [g["robot"] for g in goals] == roots
  for root in roots:
    docs = [e["name"] for e in events if e["type"] == "thought" and e["robot"] == root]
    assert docs[:len(NAMES)] == list(NAMES), f"{root}: opening documents {docs[:4]}"
    assert any(e["type"] == "grid" and e["robot"] == root for e in events), \
      f"{root}: no map of its own"
  # Whatever encounter a flight carries is well-formed.
  for e in (e for e in events if e["type"] == "encounter"):
    assert e["phase"] in ENCOUNTER_PHASES and set(e["robots"]) == set(roots)


@pytest.mark.parametrize("pair", [False, True], ids=["home_quad", "home_quad_pair"])
def test_the_quadruped_scene_fixtures_are_current(pair):
  """The home world with legs in it (issue #387): the rover taken out and
  the quadruped -- or the served pair -- and its dock put in at load, so
  the scene is built from the world's SPEC, as the sim builds it. Stale on
  any change to the house, the body or the dock."""
  from pluggybot.lifecycle import QUAD_HOME, world_config
  from pluggybot.robot import SECOND, pair_model_name, world_spec
  cfg = world_config(QUAD_HOME)
  name = pair_model_name(cfg["model_name"]) if pair else cfg["model_name"]
  scene = json.loads((PROTOCOL / f"scene.{name}.json").read_text())
  assert scene["protocolVersion"] == PROTOCOL_VERSION and scene["model"] == name
  model = world_spec(cfg["model"], second_at=cfg["start2"][:2] if pair else None,
                     body="quadruped").compile()
  meta = json.loads(Path(cfg["meta"]).read_text())
  flag = " --pair" if pair else ""
  assert scene == scene_dict(model, name, meta=meta), \
    f"stale fixture: uv run python -m pluggybot.telemetry.scene --world {QUAD_HOME}{flag}"
  owners = {b["name"]: b["robot"] for b in scene["bodies"]}
  assert "FL_thigh" in owners and owners["dock"] is None and "chassis" not in owners
  if pair:
    assert [n for n, o in owners.items() if o == SECOND.root] == \
      [SECOND.el(n) for n, o in owners.items() if o == "pluggybot"]


def test_the_quadruped_pair_recording_is_the_periods_shape():
  """The first quadruped period's fixture (issue #387): no offers and no
  upkeep -- no `tasks` block, no appetite, `hungerStates: []` -- and every
  frame says each robot's POSTURE, lying among them: the first walks to
  the dock and lies on it to charge, the second explores and rests."""
  from pluggybot.robot import FIRST, SECOND
  with gzip.open(PROTOCOL / "telemetry.home_quad_pair.jsonl.gz", "rt") as f:
    lines = [json.loads(line) for line in f]
  header, frames = lines[0], frames_of(lines)
  roots = [FIRST.root, SECOND.root]
  assert header["model"] == "home_quad_pair" and list(header["robots"]) == roots
  assert header["hungerStates"] == [] and not header["taskKinds"]
  for root in roots:
    recs = [f["robots"][root] for f in frames if root in f["robots"]]
    assert all("posture" in r and "metabolism" not in r for r in recs)
    assert {"standing", "lying"} <= {r["posture"] for r in recs}
  first = {f["robots"][FIRST.root]["state"] for f in frames if "state" in f["robots"][FIRST.root]}
  assert "CHARGE" in first, "the first robot lies on the dock and charges"
  assert not any(x.get("type") == "tasks" or "tasks" in x for x in lines[1:])


def test_the_recording_is_a_whole_served_day():
  """What the site builds against, on the one committed recording: the
  stream's shape, the memory it opens with, and the scoreboard -- each
  something no keyframe repairs if a fixture loses it."""
  from pluggybot.mind import constitution as constitutions
  from pluggybot.mind.thoughts import HISTORY, MAIN, NAMES, TOP_OF_MIND
  from pluggybot.robot import FIRST, SECOND
  lines = _pair_recording()
  roots = [FIRST.root, SECOND.root]
  # Dispatch on "type"; no "type" means frame (0.4.0). A recording is a
  # MIXED stream: events ride between the frames.
  header = lines[0]
  frames = [x for x in lines[1:] if "type" not in x]
  assert header["protocolVersion"] == PROTOCOL_VERSION
  first = frames[0]
  for root in roots:
    assert set(first["robots"][root]["bodies"]) == set(header["robots"][root]), \
      "first frame must be a keyframe"
  assert set(first["world"]) == set(header["world"])

  # Keyframes recur (0.2.0): the website's relay hub caches "last keyframe
  # + frames since" to serve a browser that joins mid-day, so a recording
  # without them would be testing a stream shape we never send.
  keys = [f for f in frames if f.get("key")]
  assert first.get("key") is True
  assert len(keys) > 1, "the fixture must exercise RECURRING keyframes"
  for f in keys:
    for root in roots:
      assert set(f["robots"][root]["bodies"]) == set(header["robots"][root])
    assert set(f["world"]) == set(header["world"])
  gaps = [b["t"] - a["t"] for a, b in zip(keys, keys[1:])]
  # the cadence re-anchors on the frame that carried the keyframe, so it can
  # run one frame interval late -- but no more, or it has silently regressed
  assert max(gaps) <= header["keyframeS"] + 2.0 / header["hz"], \
    f"keyframe spacing drifted past the advertised cadence: max {max(gaps):.2f} s"
  times = [f["t"] for f in frames]
  assert all(b > a for a, b in zip(times, times[1:]))
  states = {f["robots"][FIRST.root]["state"] for f in frames}
  assert {"EXPLORE", "GO_CHARGE", "CHARGE"} <= states, \
    "the fixture must cover the battery-driven day"
  for f in frames:
    for root in roots:
      bat = f["robots"][root]["battery"]
      assert 0.0 <= bat["frac"] <= 1.0
    for pose in [p for root in roots
                 for p in f["robots"][root].get("bodies", {}).values()] \
        + list(f.get("world", {}).values()):
      assert len(pose) == 7

  # What the robot is FOR (0.8.0), which the site's goals panel shows on a
  # recording -- the case a visitor actually meets -- and the one message
  # with no keyframe behind it. ⚠ ITS TEXT IS EMPTY, WHICH IS THE POINT
  # (0.19.0, issue #154): `Goals.md` is the ROBOT's, this day runs without
  # a mind, and `steering` rides on this message and nowhere else.
  events = [x for x in lines[1:] if "type" in x]
  first_frame = next(i for i, x in enumerate(lines) if "type" not in x)
  goals = [e for e in events if e["type"] == "goals"]
  assert [g["robot"] for g in goals] == roots, "the fixture lost the steering flag"
  for g in goals:
    assert g["text"] == "" and g["steering"] is False
    assert lines.index(g) < first_frame, "goals must precede the frames"

  # ...and the memory documents behind it (0.11.0, issue #38), on the same
  # terms: the site's Thoughts tab is built against these lines.
  docs = [e for e in events if e["type"] == "thought"]
  # ⚠ THE PERSONA IN THE FIXTURE IS THE ONE IN THE CODE (issue #39): the
  # default constitution in the quadruped's words (#387). A fixture carrying
  # last month's persona is what a visitor reads.
  persona = constitutions.for_body(constitutions.resolve(constitutions.DEFAULT_NAME),
                                   "quadruped").text.strip()
  for root in roots:
    mine = [d for d in docs if d["robot"] == root]
    opening = [d for d in mine if lines.index(d) < first_frame]
    assert [d["name"] for d in opening] == list(NAMES), \
      f"{root}: the fixture does not open with the robot's memory"
    assert {d["writer"] for d in opening} == {"human", "system", "robot"}, \
      "documents claiming one writer render as identical panels"
    # Goals.md rides the wire twice by design, and the two must agree
    assert next(d for d in opening if d["name"] == "Goals.md")["text"].strip() == ""
    # History is written DURING the day, so it must actually move
    later = [d for d in mine if lines.index(d) >= first_frame]
    assert later and {d["name"] for d in later} == {HISTORY}, \
      f"{root}: nothing, or something other than History, was written mid-day"
    # ...and the robot's OWN file is empty here, honestly: nothing without
    # a mind writes an opinion
    assert next(d for d in opening if d["name"] == TOP_OF_MIND)["text"] == ""
    main = next(d for d in opening if d["name"] == MAIN)["text"]
    assert main == persona, "the recording's persona is stale: re-record"
    # ...and it never claims the SPECIES as the robot's name
    assert "You are PluggyBot" not in main
  # The ROWS behind the documents (issue #238): one `records` snapshot per
  # robot before the first frame, and a `record` line for every History
  # line written after it.
  snaps = [e for e in events if e["type"] == "records"]
  assert [s["robot"] for s in snaps] == roots
  assert all(lines.index(s) < first_frame and s["generation"] >= 1 for s in snaps)
  rows = [e for e in events if e["type"] == "record"]
  assert rows and {r["record"]["kind"] for r in rows} == {"history"}, \
    "a mindless day writes History rows and nothing else"
  assert all(r["record"]["writer"] == "system" and r["record"]["status"] == "active"
             and r["t"] == r["record"]["t"] for r in rows)
  for root in roots:
    ids = [r["record"]["id"] for r in rows if r["robot"] == root]
    assert ids == sorted(ids)
  # ...and no map and no prompt: a world with no mind says nothing, rather
  # than an empty map or an empty prompt
  assert not [e for e in events if e["type"] in ("event_map", "prompt")]

  # The scoreboard half (0.6.0): a charge is a scored task, so a day that
  # charges banks it -- the site's scoreboard is built against this, and the
  # balance is the only part of it that survives a mid-day join.
  assert header["ledger"] == roots
  banked = [e for e in events if e["type"] == "earned"]
  assert banked and {e["task"] for e in banked} == {"charge"}
  for e in banked:
    assert e["reason"] and e["tier"] in ("auto", "hidden", "visitor")
    assert "truth" not in e["metrics"]
  with_ledger = [f for f in frames if "ledger" in f]
  assert with_ledger and len(with_ledger) < len(frames), "an unchanged balance was re-sent"
  # ...and the block's own arithmetic, which is why `consumed` and `spent`
  # are on the wire: a balance the site cannot reconstruct is points leaking
  final = with_ledger[-1]["ledger"]
  for root in roots:
    b = final[root]
    assert b["earned"] - b["consumed"] - b["spent"] - b["given"] + b["received"] \
        == b["balance"]
  assert final[FIRST.root]["recent"] and final[FIRST.root]["tasks"] >= 1


# ---- the occupancy map in a RECORDING (rooftop-media-2026 #78) -------------
#
# The grid had been a LIVE-ONLY message since 0.2.0: the publisher shipped it
# and the recorder did not, so a replayed mission was of a robot that never
# had a map. The website's default view IS a recording -- live falls back to
# one whenever no sim is publishing -- so a map panel reading only the live
# stream would be blank for almost every visitor. These four guards are the
# producer half of that fix.

def _stub_grid(marks=()):
  """A small OccupancyGrid with cells stamped directly in log-odds.

  Stamping beats driving a lidar here: what is under test is the recorder's
  seam, and a scan would make the assertions depend on ray casting.
  """
  from pluggybot.mapping.occupancy_grid import OccupancyGrid
  grid = OccupancyGrid(x_min=-1.0, y_min=-0.5, x_max=1.0, y_max=0.5,
                       resolution=0.05)
  for ix, iy, odds in marks:
    grid.grid[iy, ix] = odds
  return grid


def _record_with_grid(model, path, grid, seconds, grid_hz, on_step=None):
  data = mujoco.MjData(model)
  rec = TelemetryRecorder(model, data, path, model_name="mini",
                          grid=grid, grid_hz=grid_hz)
  for step in range(round(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    if on_step is not None:
      on_step(step)
    rec.step_hook()
  rec.close()
  return rec, [json.loads(x) for x in open(path)]


def test_a_recording_carries_the_robots_map_belief(mini_model, tmp_path):
  """The map rides a recording as its own typed line, like a stroke does.

  Everything a renderer needs to place it must be ON that line: `extent` and
  `resolution` are what scale it, and hardcoding a world size instead is the
  bug this guards: every world's grid is its own size.
  """
  grid = _stub_grid([(5, 3, -2.0), (6, 3, 2.0)])
  path = str(tmp_path / "out.jsonl")
  rec, lines = _record_with_grid(mini_model, path, grid, seconds=2.0,
                                 grid_hz=2.0)

  grids = [x for x in lines if x.get("type") == "grid"]
  assert grids, "the recording carries no map at all"
  assert rec.grids == len(grids)
  assert all(g["extent"] == [-1.0, -0.5, 1.0, 0.5] for g in grids), \
    "a renderer cannot scale a map whose extent it is not told"
  assert all(g["resolution"] == 0.05 for g in grids)
  assert all(g["robot"] == "pluggybot" for g in grids)
  # ...and the lines are ordered in sim time with the frames around them, so
  # a replayer sampling by one clock gets the map the robot held then.
  assert all(b["t"] >= a["t"] for a, b in zip(lines[1:], lines[2:]))
  assert 0 < lines.index(grids[0]) < len(lines) - 1


def test_row_zero_of_the_map_png_is_the_y_min_edge(mini_model, tmp_path):
  """The orientation contract, pinned because it is invisible until it is
  wrong: a renderer that assumes PNG row 0 is the TOP of the world draws the
  robot's map upside down, and a symmetric room hides it completely."""
  import base64
  import io

  from PIL import Image

  # Nothing but one confidently-free cell, hard against the y_min edge.
  grid = _stub_grid([(4, 0, -2.0)])
  path = str(tmp_path / "out.jsonl")
  _, lines = _record_with_grid(mini_model, path, grid, seconds=1.0,
                               grid_hz=2.0)
  first = next(x for x in lines if x.get("type") == "grid")
  img = np.array(Image.open(io.BytesIO(base64.b64decode(first["png"]))))

  assert img.shape == (20, 40), "rows are y, columns are x"
  assert img[0, 4] == 255, "the y_min row is not row 0"
  assert img[-1, 4] == 127, "something was written to the y_max row"


def test_an_unchanged_map_is_not_written_twice(mini_model, tmp_path):
  """The map stops changing for minutes at a time -- through a charge,
  through a drawing -- and a byte-identical PNG re-written every interval is
  pure weight in a file the website ships to every visitor.

  The live stream deliberately does NOT skip; see GridSampler.
  """
  grid = _stub_grid([(5, 3, -2.0)])

  def change(step):
    if step == round(1.0 / mini_model.opt.timestep):
      grid.grid[9, 3] = 2.0

  path = str(tmp_path / "out.jsonl")
  rec, lines = _record_with_grid(mini_model, path, grid, seconds=2.0,
                                 grid_hz=4.0, on_step=change)

  grids = [x for x in lines if x.get("type") == "grid"]
  assert len(grids) == 2, \
    f"an unchanged map was written {len(grids)} times, not twice"
  assert rec._grids[0].skipped >= 5, "nothing was skipped: the dedupe is inert"
  # The second one is the change, not a re-run of the first.
  assert grids[0]["png"] != grids[1]["png"]


def test_the_map_png_is_encoded_off_the_physics_thread(mini_model, tmp_path,
                                                       monkeypatch):
  """The rule the whole module is shaped around: no I/O, and no work of this
  size, inside a physics step. Encoding a 280 x 200 PNG is milliseconds; the
  hook that queues it runs at 500 Hz."""
  import pluggybot.telemetry.recorder as recorder_mod

  threads = []
  real = recorder_mod.encode_grid_png

  def spy(img):
    threads.append(threading.current_thread())
    return real(img)

  monkeypatch.setattr(recorder_mod, "encode_grid_png", spy)
  grid = _stub_grid([(5, 3, -2.0)])
  path = str(tmp_path / "out.jsonl")
  _record_with_grid(mini_model, path, grid, seconds=1.0, grid_hz=2.0)

  assert threads, "nothing was encoded"
  assert all(t is not threading.current_thread() for t in threads), \
    "the PNG was encoded inside the physics step"


def test_the_recorder_and_the_publisher_describe_one_map(mini_model):
  """Recorder and publisher are the same producer with different sinks, and
  the map is the fifth block to have to prove it: a replay that scaled its
  map differently from the live stream would be a second renderer."""
  from pluggybot.telemetry.recorder import GridSampler

  grid = _stub_grid([(5, 3, -2.0)])
  recorded, _ = GridSampler(grid, hz=1.0, dedupe=True).due(0.0)
  live, _ = GridSampler(grid, hz=1.0, dedupe=False).due(0.0)
  assert recorded == live


# ---- build identity in the header (issue #132) --------------------------------


def test_the_header_says_which_build_produced_the_stream(mini_model):
  """The deployed world is an observatory, and an observation nobody can
  attribute is not weaker data -- it is unusable data (Evaluation.md §5).

  The things a regime is made of, so a regime change is visible in the
  header itself.
  """
  from pluggybot.evaluation.identity import build_identity

  data = mujoco.MjData(mini_model)
  identity = build_identity("home_quad", arm="guarded", model="a/b",
                            backend="huggingface", pack_wh=8.0,
                            reserve_wh=0.9, deadline_s=90.0,
                            hashes={"rewards": "ab" * 32}, commit="deadbee")
  header = FrameBuilder(mini_model, data, model_name="mini",
                        build=identity).header()

  assert header["build"]["commit"] == "deadbee"
  assert header["build"]["arm"] == "guarded"
  assert header["build"]["model"] == "a/b"
  assert header["build"]["backend"] == "huggingface"
  assert header["build"]["dataHashes"] == {"rewards": "ab" * 32}
  assert (header["build"]["packWh"], header["build"]["reserveWh"],
          header["build"]["deadlineS"]) == (8.0, 0.9, 90.0)

  # ⚠ NESTED, and this is the assertion that says why: the header's own
  # `model` is the WORLD -- the field a replayer picks its scene off
  # (protocol/README.md) -- and the mind's model is a different string one
  # field away. Flattened, a consumer reading `model` would get whichever
  # was written last.
  assert header["model"] == "mini"

  # ...and the stream still has frames in it. `build` is FrameBuilder's own
  # frame-building method, so holding the identity under that name replaces
  # it with a dict and every frame after the header stops -- silently, and
  # only on the one path (serve.py) that passes an identity at all.
  builder = FrameBuilder(mini_model, data, model_name="mini", build=identity)
  assert isinstance(builder.build(), dict)


def test_a_consumer_that_never_heard_of_the_build_block_still_works(mini_model):
  """ADDITIVE, so no `protocolVersion` bump (protocol/README.md's own rule).

  Two halves, and the second is the one that rots: the version does not
  move, AND a header built without an identity is byte-identical to the one
  0.15.0 always produced -- which is what keeps every committed fixture,
  and every recording made before this, valid.
  """
  data = mujoco.MjData(mini_model)
  bare = FrameBuilder(mini_model, data, model_name="mini").header()

  assert bare["protocolVersion"] == PROTOCOL_VERSION == "0.21.0"
  assert "build" not in bare, \
    "a run that was handed no identity must not invent one"

  from pluggybot.evaluation.identity import build_identity
  stamped = FrameBuilder(mini_model, data, model_name="mini",
                         build=build_identity("home_quad", arm="scripted",
                                              hashes={}, commit="x")).header()
  assert {k: v for k, v in stamped.items() if k != "build"} == bare, \
    "the identity changed a field a 0.15.0 consumer already reads"


def test_the_header_hashes_the_files_the_sim_reads(monkeypatch, tmp_path):
  """One implementation, not two (issue #132): the header calls
  `data_hashes`, which resolves each data file exactly as the sim does --
  the env override wins -- and the proof is that re-pointing a data file
  moves the header's hash of it and nothing else."""
  from pluggybot.economy import scoring
  from pluggybot.evaluation import identity

  before = identity.build_identity("home_quad", arm="scripted")["dataHashes"]
  assert before == identity.data_hashes("home_quad")

  tweaked = tmp_path / "rewards.json"
  tweaked.write_text(Path(scoring.TABLE_PATH).read_text() + "\n")
  monkeypatch.setenv(scoring.TABLE_ENV, str(tweaked))
  after = identity.build_identity("home_quad", arm="scripted")["dataHashes"]

  assert after["rewards"] != before["rewards"], \
    "the env override the sim reads is not the file the header hashed"
  assert after["world"] == before["world"]
