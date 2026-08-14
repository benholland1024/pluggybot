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
from pluggybot.telemetry.recorder import TelemetryRecorder
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
  assert len(referenced) == model.ntex        # all 10 tags in use
  assert {t["name"] for t in scene["textures"]} == referenced
  robot, world = body_census(model)
  assert sum(dynamic_flags(model)) == len(robot) + len(world) == 16
  assert len(robot) == 7 and "rack" in world and "module_lcd" in world


# ---- recorder --------------------------------------------------------------

def record(model, seconds=2.0, path=None, status_fn=None, tmp=None):
  data = mujoco.MjData(model)
  path = path or str(tmp / "out.jsonl")
  rec = TelemetryRecorder(model, data, path, status_fn=status_fn,
                          model_name="mini")
  for _ in range(round(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
    rec.step_hook()
  rec.close()
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt") as f:
    lines = [json.loads(line) for line in f]
  return rec, lines


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


# ---- the committed fixtures ------------------------------------------------
# The same checks the website repo runs against its vendored copies: if these
# fail, regenerate the fixtures or bump the protocol version deliberately.

SCENE_FIXTURE = REPO / "protocol" / "scene.room_hub.json"
TELEMETRY_FIXTURE = REPO / "protocol" / "telemetry.hub_lifecycle.jsonl.gz"


def test_scene_fixture_current():
  scene = json.loads(SCENE_FIXTURE.read_text())
  assert scene["protocolVersion"] == PROTOCOL_VERSION
  assert scene["model"] == "room_hub" and scene["upAxis"] == "z"
  for tex in scene["textures"]:
    assert (SCENE_FIXTURE.parent / "textures" / tex["file"]).exists()
  # the committed fixture must match the committed WORLD, not an old one
  model = mujoco.MjModel.from_xml_path(str(REPO / "models" / "room_hub.xml"))
  assert scene == scene_dict(model, "room_hub"), \
    "stale fixture: uv run python -m pluggybot.telemetry.scene"


def test_telemetry_fixture_is_a_full_mission():
  with gzip.open(TELEMETRY_FIXTURE, "rt") as f:
    lines = [json.loads(line) for line in f]
  header, frames = lines[0], lines[1:]
  assert header["protocolVersion"] == PROTOCOL_VERSION
  robot_names = set(header["robots"]["pluggybot"])
  first = frames[0]["robots"]["pluggybot"]
  assert set(first["bodies"]) == robot_names, "first frame must be a keyframe"
  assert set(frames[0]["world"]) == set(header["world"])
  times = [f["t"] for f in frames]
  assert all(b > a for a, b in zip(times, times[1:]))
  states = {f["robots"]["pluggybot"]["state"] for f in frames}
  assert {"EXPLORE", "GO_CHARGE", "CHARGE",
          "SWAP_PICK", "USE_TOOL", "SWAP_RETURN"} <= states, \
    "the fixture must cover the full battery-driven mission"
  for f in frames:
    bat = f["robots"]["pluggybot"]["battery"]
    assert 0.0 <= bat["frac"] <= 1.0
    for pose in list(f["robots"]["pluggybot"].get("bodies", {}).values()) \
        + list(f.get("world", {}).values()):
      assert len(pose) == 7
