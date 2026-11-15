"""Guards for the video recorder (viz.py) and the clips it writes of the
humanoid's scenes (`humanoid_spike.py --record`, #517)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot import viz
from pluggybot.humanoid import model as hm
from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.viz import Recorder

ROOT = Path(__file__).resolve().parents[1]


# --- video recorder ------------------------------------------------------

SMALL = (320, 180)          # keep the encoder cheap in the test suite


@pytest.fixture(scope="module")
def world_model():
  """The humanoid alone on a floor."""
  return mujoco.MjModel.from_xml_string(hm.body_xml(hm.CHOSEN))


@pytest.fixture
def world_data(world_model):
  return mujoco.MjData(world_model)


def _step_for(model, data, seconds, recorder=None):
  until = data.time + seconds
  while data.time < until:
    mujoco.mj_step(model, data)
    if recorder is not None:
      recorder.maybe_grab(data)


def _step_n(model, data, n, recorder=None):
  """A FIXED step count, so anything that steps the sim behind our back shows
  up as extra elapsed time rather than just ending the loop sooner."""
  for _ in range(n):
    mujoco.mj_step(model, data)
    if recorder is not None:
      recorder.maybe_grab(data)


def test_recorder_samples_on_the_sim_clock(world_model, world_data,
                                           tmp_path):
  """Frame count follows sim time / speed, not wall time -- that is the whole
  reason to render offscreen instead of screen-recording the viewer."""
  import imageio.v2 as imageio

  out = tmp_path / "clip.mp4"
  rec = Recorder(world_model, str(out), fps=30, speed=1.0, size=SMALL)
  try:
    _step_for(world_model, world_data, 1.0, rec)
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


def test_recorder_speed_scales_the_sample_interval(world_model,
                                                   world_data, tmp_path):
  out = tmp_path / "fast.mp4"
  rec = Recorder(world_model, str(out), fps=30, speed=3.0, size=SMALL)
  try:
    _step_for(world_model, world_data, 1.0, rec)
  finally:
    rec.close()
  # 3 sim seconds per played second -> a third of the frames for the same sim.
  assert abs(rec.frames_written - 11) <= 2, \
    f"speed=3 should thin 31 frames to ~11, got {rec.frames_written}"


def test_recorder_does_not_disturb_the_sim(world_model, tmp_path):
  """Recording must not move the numbers a demo reports.

  Fails if anything in the capture path steps the sim -- the way an
  end-of-clip 'hold on the final pose' does when it runs before the result
  dict is built, which silently shifted a demo's reported settle state.
  """

  def moving():
    # The scene must actually be EVOLVING or the assertion cannot fail: at
    # rest, extra hidden steps leave qpos bit-identical and the test is decor.
    d = mujoco.MjData(world_model)
    d.qvel[:] = 0.5
    return d

  plain = moving()
  _step_n(world_model, plain, 250)

  recorded = moving()
  rec = Recorder(world_model, str(tmp_path / "probe.mp4"), size=SMALL)
  try:
    _step_n(world_model, recorded, 250, rec)
  finally:
    rec.close()

  assert recorded.time == plain.time, "recording changed the sim clock"
  np.testing.assert_array_equal(
    recorded.qpos, plain.qpos, err_msg="recording perturbed the physics")


def test_recorder_azimuth_takes_the_shortest_arc(world_model, tmp_path):
  """350 deg -> 10 deg is +20, not -340.

  A naive lerp toward the raw target swings the camera the long way round the
  scene -- three-quarters of a circle of scenery whipping past mid-clip.
  """
  rec = Recorder(world_model, str(tmp_path / "pan.mp4"),
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


def test_recorder_rejects_unknown_track_body(world_model, tmp_path):
  with pytest.raises(KeyError):
    Recorder(world_model, str(tmp_path / "x.mp4"),
             track_body="no_such_body", size=SMALL)


def _frames(path):
  import imageio.v2 as imageio
  return imageio.mimread(str(path), memtest=False)


def test_recorder_brightens_its_frames_after_rendering_never_the_light(world_model,
                                                                       tmp_path):
  """#517: the frames are lifted after they are rendered, and every frame
  is rendered in the scene's own light -- the robot's cameras see by it
  (#507: brightened, a tag decoded 3 cm elsewhere)."""
  light = (world_model.vis.headlight.ambient.copy(),
           world_model.vis.headlight.diffuse.copy())
  means = {}
  for lift in (1.0, 1.5):
    data = mujoco.MjData(world_model)
    rec = Recorder(world_model, str(tmp_path / f"lift{lift}.mp4"), size=SMALL,
                   track_body=ROBOT_ROOT, distance=3.0, brighten=lift)
    render, lit = rec.renderer.render, []

    def spied(*a, **k):
      lit.append((world_model.vis.headlight.ambient.copy(),
                  world_model.vis.headlight.diffuse.copy()))
      return render(*a, **k)
    rec.renderer.render = spied
    try:
      _step_n(world_model, data, 20, rec)
    finally:
      rec.close()
    assert lit, "nothing was rendered"
    for ambient, diffuse in lit:
      np.testing.assert_array_equal(ambient, light[0])
      np.testing.assert_array_equal(diffuse, light[1])
    means[lift] = float(np.mean(_frames(tmp_path / f"lift{lift}.mp4")))
  assert means[1.5] > means[1.0] + 10.0, means


def test_recorder_holds_its_last_frame_without_stepping(world_model, tmp_path):
  """A clip's end is held so its last caption is read: the frame rendered
  once and written again, never the sim stepped (a hold that stepped it
  moved a demo's reported state)."""
  data = mujoco.MjData(world_model)
  data.qvel[:] = 0.5                       # moving, or a step would not show
  rec = Recorder(world_model, str(tmp_path / "held.mp4"), size=SMALL)
  try:
    _step_n(world_model, data, 50, rec)
    t, qpos, n = data.time, data.qpos.copy(), rec.frames_written
    rec.hold(data, 1.0)
  finally:
    rec.close()
  assert rec.frames_written - n == 30
  assert data.time == t
  np.testing.assert_array_equal(data.qpos, qpos)


def test_recorder_cuts_a_caption_to_the_frame(world_model, tmp_path):
  """A line longer than the frame is cut at a word, an ellipsis where it
  was: drawn whole, it ran under the clock and off the frame."""
  rec = Recorder(world_model, str(tmp_path / "x.mp4"), size=SMALL)
  try:
    cut = rec._fit("shut at 40 N, 0.2 kg a jaw " * 8, 200)
  finally:
    rec.close()
  assert cut.endswith(" ...") and rec._font.getlength(cut) <= 200
  one = rec._fit("x" * 200, 100)                  # a first word too wide alone
  assert one.endswith(" ...") and rec._font.getlength(one) <= 100
  with pytest.raises(ValueError):
    Recorder(world_model, str(tmp_path / "y.mp4"), size=SMALL, brighten=0.0)


def test_recorder_keeps_real_time_over_a_long_clip(world_model, tmp_path):
  """Each sample is due an interval after the last was due: counted from
  the step that took it, 120 s at a 2 ms step gave 3,538 frames for 3,601
  and the clip played 1.7 % fast; and a second call at one instant (a line
  yielded with no step) never writes a second frame. The clock alone, no
  physics, no picture."""
  rec = Recorder(world_model, str(tmp_path / "long.mp4"), size=SMALL)
  rec._frame = lambda data: None
  rec.writer.append_data = lambda frame: None
  data = mujoco.MjData(world_model)
  try:
    for k in range(1, 60_001):
      data.time = k * 0.002
      rec.maybe_grab(data)
      rec.maybe_grab(data)
  finally:
    rec.close()
  assert abs(rec.frames_written - 3601) <= 1, rec.frames_written


def test_recorder_films_on_after_the_clock_is_set_back(world_model, world_data,
                                                       tmp_path):
  """A scene that starts over resets its data, and its clock goes back to 0
  (`getup`'s three lyings, `jaws`' shuts): held to the old next sample, the
  recorder filmed nothing of it until the clock caught up."""
  rec = Recorder(world_model, str(tmp_path / "again.mp4"), size=SMALL)
  try:
    _step_for(world_model, world_data, 0.5, rec)
    mujoco.mj_resetData(world_model, world_data)
    _step_for(world_model, world_data, 0.5, rec)
  finally:
    rec.close()
  assert abs(rec.frames_written - 32) <= 2, \
    f"twice 0.5 s at 30 fps should be ~32 frames, got {rec.frames_written}"


def test_recorder_writes_a_gif_beside_its_clip(world_model, world_data, tmp_path):
  from PIL import Image
  out, gif = tmp_path / "clip.mp4", tmp_path / "clip.gif"
  rec = Recorder(world_model, str(out), size=SMALL)
  try:
    _step_for(world_model, world_data, 1.0, rec)
  finally:
    rec.close(gif=str(gif))
  assert out.exists() and gif.exists()
  with Image.open(gif) as g:
    assert g.size[0] == viz.GIF_W
    assert abs(g.n_frames - viz.GIF_FPS) <= 2, g.n_frames


def test_record_flies_a_scene_once_and_writes_its_clip(tmp_path, monkeypatch):
  """`humanoid_spike.py --record SCENE PATH --gif` (#517): the scene's world
  handed to the recorder, a frame sampled on the sim clock as its lines go
  by, brightened after rendering, the part it is at captioned and under it
  how the last part ended (the latest word alone: each kept, the caption ran
  off the frame), the last frame held, the GIF beside the MP4, the world
  closed; and nothing stepped but the scene's own steps. Flown on a
  stand-in scene: no scene is flown in the suite."""
  monkeypatch.syspath_prepend(str(ROOT / "scripts"))
  import humanoid_spike as sp
  from pluggybot import batch
  monkeypatch.setattr(batch, "watched", lambda: None)     # the runner's, never this process's

  model = mujoco.MjModel.from_xml_string(hm.body_xml(hm.CHOSEN))
  data = mujoco.MjData(model)
  steps = int(round(0.5 / model.opt.timestep))

  def lines():
    yield "lying on its back"
    for k in range(steps):
      mujoco.mj_step(model, data)
      yield "  falling; the get-up has it" if k == steps // 2 else None
    yield "  stood in 0.5 s"
  closed, made = [], {}
  monkeypatch.setattr(sp, "scene_clip", lambda scene, at_once=False: {
    "model": model, "data": data, "lines": lines(), "close": lambda: closed.append(scene)})
  monkeypatch.setattr(sp, "CLIP_SIZE", SMALL)

  class Spied(Recorder):
    def __init__(self, *a, **k):
      made.update(k)
      super().__init__(*a, **k)

    def set_label(self, text):
      made["label"] = text
      super().set_label(text)
  monkeypatch.setattr(viz, "Recorder", Spied)
  out = tmp_path / "getup.mp4"
  sp.main(["--record", "getup", str(out), "--gif"])

  assert round(data.time / model.opt.timestep) == steps, "recording stepped the sim"
  assert closed == ["getup"]
  assert made["track_body"] == ROBOT_ROOT and made["fps"] == 30
  assert made["brighten"] == sp.CLIP_BRIGHTEN > 1.0
  assert made["label"] == "lying on its back\nlying on its back: stood in 0.5 s"
  assert abs(len(_frames(out)) - (16 + 30 * sp.CLIP_HOLD_S)) <= 2
  assert out.with_suffix(".gif").exists()
  for refused in (["--record", "no_such_scene", str(tmp_path / "x.mp4")],
                  ["--record", "getup", str(tmp_path / "x.gif")],
                  ["--record", "getup", str(tmp_path / "x.mp4"), "--strip", "getup"],
                  ["--strip", "getup", "--gif", "--out", str(tmp_path / "strip.png")],
                  ["--record", "getup", str(tmp_path / "x.mp4"), "--stand-at-once"]):
    with pytest.raises(SystemExit):
      sp.main(refused)
  # ...and a world drawn brighter for our camera already is not lifted again
  model.vis.headlight.ambient[:] = 0.35
  assert sp.clip_lift(model) == 1.0

  def failing():
    yield "walking up the flight"
    for _ in range(20):
      mujoco.mj_step(model, data)
      yield None
    raise RuntimeError("a flight that fails")
  monkeypatch.setattr(sp, "scene_clip", lambda scene, at_once=False: {
    "model": model, "data": data, "lines": failing()})
  broken = tmp_path / "stairs.mp4"
  for earlier in (broken, broken.with_suffix(".gif")):     # an earlier run's
    earlier.write_bytes(b"an earlier run's clip")
  with pytest.raises(RuntimeError):
    sp.main(["--record", "stairs", str(broken), "--gif"])
  # ...leaves no clip at PATH that reads as the whole scene
  assert not broken.exists() and not broken.with_suffix(".gif").exists()
  assert (tmp_path / "stairs.part.mp4").exists()
