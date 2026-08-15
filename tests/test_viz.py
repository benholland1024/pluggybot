"""Guards for the camera dashboard (viz.py, issue #1) and the video recorder."""

import numpy as np
import pytest

from pluggybot.viz import SEAM, TILE_H, TILE_W, Recorder, ViewDashboard


def test_dashboard_composite_layout(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    frame = dash.render(playground_data)
  finally:
    dash.close()

  assert frame.shape == (2 * TILE_H + SEAM, 2 * TILE_W + SEAM, 3)
  left = frame[:TILE_H, :TILE_W]
  right = frame[:TILE_H, TILE_W + SEAM:]
  dock = frame[TILE_H + SEAM:, TILE_W + SEAM:]
  assert left.std() > 0, "left-eye tile is blank"
  assert dock.std() > 0, "dock-eye tile is blank"
  assert not np.array_equal(left, right), "stereo tiles show no parallax"


def test_dashboard_embeds_map_tile(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    blank = dash.render(playground_data, map_img=None)
    # A recognizable map: white field with a black block off-center.
    map_img = np.full((200, 200), 255, dtype=np.uint8)
    map_img[40:60, 40:60] = 0
    with_map = dash.render(playground_data, map_img=map_img)
  finally:
    dash.close()

  map_tile_blank = blank[TILE_H + SEAM:, :TILE_W]
  map_tile = with_map[TILE_H + SEAM:, :TILE_W]
  assert not np.array_equal(map_tile, map_tile_blank), "map tile ignored map_img"
  # The map must arrive un-mirrored: the black block sits in the tile's
  # upper-left quadrant, as it does in the source image.
  h, w = map_tile.shape[:2]
  upper_left = map_tile[:h // 2, :w // 2].mean()
  lower_right = map_tile[h // 2:, w // 2:].mean()
  assert upper_left < lower_right, "map tile flipped or misplaced"


def test_battery_glyph_renders_and_reflects_state(playground_model, playground_data):
  import mujoco
  mujoco.mj_forward(playground_model, playground_data)
  dash = ViewDashboard(playground_model)
  try:
    plain = dash.render(playground_data)
    full = dash.render(playground_data, battery=0.9)
    low = dash.render(playground_data, battery=0.1)
    charging = dash.render(playground_data, battery=0.1, charging=True)
  finally:
    dash.close()
  region = np.s_[TILE_H + SEAM:TILE_H + SEAM + 24, TILE_W - 100:TILE_W]
  assert not np.array_equal(plain[region], full[region]), "glyph missing"
  assert not np.array_equal(full[region], low[region]), \
    "urgency color must change with charge level"
  assert not np.array_equal(low[region], charging[region]), \
    "charging must be visually distinct"


# --- video recorder ------------------------------------------------------

SMALL = (320, 180)          # keep the encoder cheap in the test suite


def _step_for(model, data, seconds, recorder=None):
  import mujoco
  until = data.time + seconds
  while data.time < until:
    mujoco.mj_step(model, data)
    if recorder is not None:
      recorder.maybe_grab(data)


def _step_n(model, data, n, recorder=None):
  """A FIXED step count, so anything that steps the sim behind our back shows
  up as extra elapsed time rather than just ending the loop sooner."""
  import mujoco
  for _ in range(n):
    mujoco.mj_step(model, data)
    if recorder is not None:
      recorder.maybe_grab(data)


def test_recorder_samples_on_the_sim_clock(playground_model, playground_data,
                                           tmp_path):
  """Frame count follows sim time / speed, not wall time -- that is the whole
  reason to render offscreen instead of screen-recording the viewer."""
  import imageio.v2 as imageio

  out = tmp_path / "clip.mp4"
  rec = Recorder(playground_model, str(out), fps=30, speed=1.0, size=SMALL)
  try:
    _step_for(playground_model, playground_data, 1.0, rec)
  finally:
    rec.close()

  assert abs(rec.frames_written - 31) <= 2, \
    f"1.0 s at 30 fps should be ~31 frames, got {rec.frames_written}"
  clip = imageio.mimread(str(out), memtest=False)
  assert len(clip) > 0, "encoder produced no readable frames"
  # The encoded frames must be exactly what was rendered: if the size is off
  # the 16-px macroblock grid, ffmpeg resamples them behind our back.
  assert clip[0].shape[:2] == (rec.height, rec.width), \
    "encoder resized the rendered frames"


def test_recorder_speed_scales_the_sample_interval(playground_model,
                                                   playground_data, tmp_path):
  out = tmp_path / "fast.mp4"
  rec = Recorder(playground_model, str(out), fps=30, speed=3.0, size=SMALL)
  try:
    _step_for(playground_model, playground_data, 1.0, rec)
  finally:
    rec.close()
  # 3 sim seconds per played second -> a third of the frames for the same sim.
  assert abs(rec.frames_written - 11) <= 2, \
    f"speed=3 should thin 31 frames to ~11, got {rec.frames_written}"


def test_recorder_does_not_disturb_the_sim(playground_model, tmp_path):
  """Recording must not move the numbers a demo reports.

  Fails if anything in the capture path steps the sim -- the way an
  end-of-clip 'hold on the final pose' does when it runs before the result
  dict is built, which silently shifted pickup.py's reported settle state.
  """
  import mujoco

  def moving():
    # The scene must actually be EVOLVING or the assertion cannot fail: at
    # rest, extra hidden steps leave qpos bit-identical and the test is decor.
    d = mujoco.MjData(playground_model)
    d.qvel[:] = 0.5
    return d

  plain = moving()
  _step_n(playground_model, plain, 250)

  recorded = moving()
  rec = Recorder(playground_model, str(tmp_path / "probe.mp4"), size=SMALL)
  try:
    _step_n(playground_model, recorded, 250, rec)
  finally:
    rec.close()

  assert recorded.time == plain.time, "recording changed the sim clock"
  np.testing.assert_array_equal(
    recorded.qpos, plain.qpos, err_msg="recording perturbed the physics")


def test_recorder_azimuth_takes_the_shortest_arc(playground_model, tmp_path):
  """350 deg -> 10 deg is +20, not -340.

  A naive lerp toward the raw target swings the camera the long way round the
  scene -- three-quarters of a circle of scenery whipping past mid-clip.
  """
  rec = Recorder(playground_model, str(tmp_path / "pan.mp4"),
                 azimuth=350.0, size=SMALL)
  try:
    rec.set_camera(azimuth=10.0)
    seen = []
    for _ in range(40):
      rec._apply_camera(0.10)
      seen.append(rec.cam.azimuth % 360.0)
  finally:
    rec.close()

  assert seen[0] > 350.0, \
    f"first step went backwards ({seen[0]:.1f}); took the long way round"
  # The short arc only ever visits 350..360..10. The long way passes through
  # the opposite side of the scene, so nothing may land near 180.
  assert not any(90.0 < a < 270.0 for a in seen), \
    "camera swung through the far side of the scene"
  assert abs((seen[-1] - 10.0 + 180.0) % 360.0 - 180.0) < 0.5, \
    f"never settled on the target azimuth (ended {seen[-1]:.1f})"


def test_recorder_rejects_unknown_track_body(playground_model, tmp_path):
  with pytest.raises(KeyError):
    Recorder(playground_model, str(tmp_path / "x.mp4"),
             track_body="no_such_body", size=SMALL)
