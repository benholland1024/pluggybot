"""Guards for the video recorder (viz.py)."""

import numpy as np
import pytest

from pluggybot.viz import Recorder


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
