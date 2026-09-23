"""The near-field map in the loop and on the wire (issue #34, stage 2).

The rules, each pinned cheaply: the seam ticks the sensor at its own rate
and only where it is on; the map is placed by the BELIEVED pose; the draw
is a load only while the sensor runs; a recompile rebinds the camera; the
pair drops the other robot; and the `heightmap` line beside `grid` carries
what a renderer needs and decodes back to the heights. No mission flown.
"""

import base64
import io
import json
import math

import mujoco
import numpy as np
from PIL import Image

from pluggybot import power
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.perception import depth as nf
from pluggybot.perception.heightmap import HeightMap
from pluggybot.robot import world_spec
from pluggybot.telemetry.recorder import (GridSampler, HeightMapSampler,
                                          StreamRobot, TelemetryRecorder,
                                          grid_samplers)


def _life(near_field: bool, props=()) -> HubLifecycle:
  cfg = world_config("room_hub")
  spec = world_spec(cfg["model"])
  for i, (x, y, side) in enumerate(props):
    g = spec.worldbody.add_geom()
    g.name = f"prop_{i}"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [side / 2] * 3
    g.pos = [x, y, side / 2]
  model = spec.compile()
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world="room_hub",
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      spec=spec, errand=False, near_field=near_field)
  life.mission.start_at(0.0, 0.0, 0.0)
  for _ in range(300):
    mujoco.mj_step(model, data)
  return life


def test_the_seam_ticks_the_sensor_at_its_rate_and_only_where_it_is_on():
  """A frame is ~7 ms of Python, so the hook is throttled to `nf.PERIOD`
  and absent altogether when the sensor is off -- a mission test pays
  nothing for a sensor it did not ask for."""
  off = _life(near_field=False)
  assert off.near_field is None and off.depth_camera is None
  assert off._near_field_step not in off.mission.step_hooks
  on = _life(near_field=True)
  assert on._near_field_step in on.mission.step_hooks
  n0 = on.near_field_frames
  assert n0 > 0, "the seam did not tick while the lifecycle settled"
  on.data.time += nf.PERIOD
  on._near_field_step()
  on._near_field_step()
  assert on.near_field_frames == n0 + 1, "two calls in one period took two frames"
  on.data.time += nf.PERIOD
  on._near_field_step()
  assert on.near_field_frames == n0 + 2
  assert on.near_field.seen_fraction() > 0.05


def test_the_map_is_placed_by_the_believed_pose_not_the_true_one():
  """The sensor never learns where the robot is: a cube 0.9 m ahead lands
  in the map where the RECKONER says the robot is, so a reckoner told it
  stands a metre further along puts the cube a metre further along."""
  life = _life(near_field=True, props=((0.9 + 0.025, 0.0, 0.05),))
  life.near_field = HeightMap()                      # forget the settling frames
  life.mission.swap.reckoner.x += 1.0                # a belief, deliberately wrong
  life.data.time += nf.PERIOD
  life._near_field_step()
  cubes = [t for t in life.near_field.things()
           if abs(t["y"]) < 0.3 and 0.03 < t["height"] < 0.08]
  assert len(cubes) == 1, cubes
  assert abs(cubes[0]["x"] - (0.925 + 1.0)) < 0.05, cubes[0]


def test_the_camera_draws_power_only_while_the_map_is_built():
  """`DEPTH_CAMERA_W` is a load like the module's: on the pack while the
  sensor streams, absent where it is off. Nothing else about the two
  lifecycles differs, so the pack gap over a second is that draw."""
  on, off = _life(near_field=True), _life(near_field=False)
  dt = on.model.opt.timestep
  steps = round(1.0 / dt)
  for life in (on, off):
    life.battery.energy_wh = 1.0
    for _ in range(steps):
      life._power_step()
  gap_wh = off.battery.energy_wh - on.battery.energy_wh
  assert math.isclose(gap_wh, power.DEPTH_CAMERA_W / 3600.0, rel_tol=0.05), \
    f"the sensor cost {gap_wh * 3600:.2f} W, not {power.DEPTH_CAMERA_W}"


def test_a_recompile_rebinds_the_camera():
  """The lifecycle owns the camera, so `rebind` re-points it (issue #168's
  fence: a holder of the old model keeps casting into a stale world)."""
  life = _life(near_field=True)
  model2 = life.spec.compile()
  data2 = mujoco.MjData(model2)
  life.rebind(model2, data2)
  assert life.depth_camera.model is model2
  assert life.depth_camera.cam_id == model2.camera("depth_eye").id


def test_a_pair_drops_the_other_robot_from_each_frame():
  """As the LIDAR does: another robot standing in the frame is no
  information about the floor."""
  from pluggybot.pair import build_pair
  lives = build_pair("room_hub", near_field=True, errands=("none", "none"))
  assert all(life.depth_camera is not None for life in lives)
  roots = [life.mission.handle.root for life in lives]
  assert lives[0].depth_camera._other_roots == [roots[1]]
  assert lives[1].depth_camera._other_roots == [roots[0]]
  assert lives[1].depth_camera.camera_name == "r2_depth_eye"


# ---- the peer channel (issue #328) ------------------------------------------

def _pair_frame(gap: float, across: float):
  """One depth frame of the first robot, with the second `gap` m ahead and
  `across` m to its left. Returns the camera, its data, and the frame."""
  from pluggybot.robot import FIRST, SECOND, world_with_robots
  model = world_with_robots("models/room_hub.xml", second_at=(3.0, 3.0))
  data = mujoco.MjData(model)
  cam = nf.DepthCamera(model, handle=FIRST)
  cam.exclude_robot(SECOND.root)
  adr = SECOND.qpos_adr(model)
  data.qpos[adr:adr + 2] = [gap, across]        # the first robot is at 0,0
  mujoco.mj_forward(model, data)
  return cam, data, cam.frame(data)


def test_the_frame_answers_the_peer_apart_from_the_room():
  """Issue #328. The height map must not hold a body that drives off, and
  the thing that must not drive into it must see it -- so the same casts
  come back sorted, exactly as `Lidar.scan_split` sorts the scan (#316).
  The peer's pixels are in `peers` and in nothing else."""
  cam, _, frame = _pair_frame(1.0, 0.0)
  assert len(frame.peers) > 100, "the other robot is not in the frame at all"
  # ...and the map's cloud has nothing standing where the robot is: every
  # room point is either well short of it or well past it.
  ahead = frame.points[np.abs(frame.points[:, 1]) < 0.15]
  on_the_peer = ahead[(ahead[:, 0] > 0.75) & (ahead[:, 0] < 1.05)
                      & (ahead[:, 2] > 0.05)]
  assert len(on_the_peer) == 0, "the peer got into the map's cloud"
  # the peer's own points are where the peer is, standing off the floor
  assert 0.75 < float(frame.peers[:, 0].min()) < 1.05
  assert float(frame.peers[:, 2].max()) > 0.2, "only the floor came back"
  # a world with nobody else in it pays nothing and carries nothing
  from pluggybot.robot import FIRST
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  alone = nf.DepthCamera(model, handle=FIRST).frame(data)
  assert len(alone.peers) == 0


def test_the_peer_channel_does_not_draw_on_the_maps_noise_stream():
  """The lesson of issue #316's `Lidar.peer_rng`, one sensor along: peer
  pixels reach the noise and the dropout now, and taking those draws off
  the map's stream would move every reading the height map takes the
  moment a second robot walked into frame. `peer_rng` is the second
  stream; the map's is drawn from exactly twice a frame either way."""

  class Counting:
    def __init__(self, inner):
      self.inner, self.draws = inner, 0

    def random(self, *a, **kw):
      self.draws += 1
      return self.inner.random(*a, **kw)

    def normal(self, *a, **kw):
      self.draws += 1
      return self.inner.normal(*a, **kw)

  cam, data, _ = _pair_frame(1.0, 0.0)
  cam.rng = counted = Counting(np.random.default_rng(3))
  frame = cam.frame(data)
  assert len(frame.peers) > 100, "no peer in the frame to draw for"
  assert counted.draws == 2, \
    f"the map's stream was drawn {counted.draws} times, not twice"


# ---- the wire --------------------------------------------------------------

def _stub_map(cells) -> HeightMap:
  """A map with the robot at the origin and `(ix, iy, height)` cells set,
  indices relative to the window's corner."""
  hm = HeightMap()
  hm.recentre(0.0, 0.0)
  for ix, iy, h in cells:
    hm.height[iy, ix] = h
    hm.count[iy, ix] = 1
  return hm


def test_the_heightmap_line_carries_what_a_renderer_needs_and_decodes(tmp_path):
  """Beside `grid`, its own typed line: a moving `extent`, `resolution`,
  and the encoding's two numbers (`zMax`, `steps`) so a consumer decodes it
  without the sim's constants -- 0 is never measured, else 1 cm steps.
  Row 0 is the y_min edge, as the grid's is."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  hm = _stub_map([(10, 20, 0.10), (150, 150, 0.30), (5, 5, 0.9), (6, 5, -0.02)])
  path = str(tmp_path / "out.jsonl")
  rec = TelemetryRecorder(model, data, path, model_name="room_hub",
                          heightmap=hm, grid_hz=2.0)
  assert [s._interval for s in rec._grids
          if isinstance(s, HeightMapSampler)] == [10.0], "recorded at 0.1 Hz"
  for _ in range(round(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    rec.step_hook()
  rec.close()
  lines = [json.loads(x) for x in open(path)]
  hms = [x for x in lines if x.get("type") == "heightmap"]
  assert len(hms) == 1, "one map, unchanged: written once"
  assert rec.heightmaps == 1 and rec.grids == 0
  m = hms[0]
  assert m["robot"] == "pluggybot"
  assert m["extent"] == hm.extent and m["extent"][2] - m["extent"][0] == 4.0
  assert m["resolution"] == 0.02 and m["zMax"] == 0.5 and m["steps"] == 50
  img = np.array(Image.open(io.BytesIO(base64.b64decode(m["png"]))))
  assert img.shape == (200, 200) and img.dtype == np.uint8
  h = HeightMap.from_image(img)
  assert np.isnan(h[0, 0]), "an unmeasured cell must decode as unknown"
  assert abs(h[20, 10] - 0.10) < 0.006 and abs(h[150, 150] - 0.30) < 0.006
  assert h[5, 5] == 0.5, "above zMax clips to zMax"
  assert h[5, 6] == 0.0, "below the floor clips to the floor"
  assert img[5, 6] == 1, "a measured floor cell is 1, never the unseen 0"


def test_the_recorder_and_the_publisher_describe_one_height_map():
  """Same sampler, two sinks (the grid's rule): the recorded message and the
  live one are equal, and the extent follows the window."""
  hm = _stub_map([(10, 20, 0.10)])
  recorded, _ = HeightMapSampler(hm, hz=1.0, dedupe=True).due(0.0)
  live, _ = HeightMapSampler(hm, hz=1.0, dedupe=False).due(0.0)
  assert recorded == live
  hm.recentre(3.0, 0.0)
  moved, _ = HeightMapSampler(hm, hz=1.0).due(0.0)
  assert moved["extent"][0] == recorded["extent"][0] + 3.0


def test_samplers_follow_each_robots_map():
  """A single-robot caller's map is the first robot's; a `StreamRobot`
  without one gets no sampler; a second robot's rides its root."""
  hm = _stub_map([])
  only_grid = grid_samplers(None, [], 1.0, False)
  assert not any(isinstance(s, HeightMapSampler) for s in only_grid)
  both = grid_samplers(None, [StreamRobot("r2_pluggybot", heightmap=hm)],
                       1.0, False, heightmap=hm)
  hms = [s for s in both if isinstance(s, HeightMapSampler)]
  assert [s.root for s in hms] == ["pluggybot", "r2_pluggybot"]
  assert isinstance(both[0], GridSampler) and not isinstance(both[0], HeightMapSampler)
