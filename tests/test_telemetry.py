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

  if not draws:
    return
  # The drawing half (0.4.0): the board is erased, then inked, and BOTH facts
  # reach a consumer only through these lines and the boards block. There is
  # no body to watch -- a fixture that lost them replays a robot miming at a
  # blank wall, and every assertion above would still pass.
  events = [x for x in lines[1:] if "type" in x]
  cleared = [e for e in events if e["type"] == "board_cleared"]
  drawn = [e for e in events if e["type"] == "draw"]
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
