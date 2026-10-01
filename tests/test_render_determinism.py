"""The robot's cameras render the same frame twice (issue #110).

A day must be the same day twice, and the first committed `scripted`
series broke it: five identical days, three trajectories. Traced to the
offscreen renderer -- with multisample antialiasing on, one static scene
rendered to a different image every time (+-1 in a few dozen shadow-edge
pixels), the AprilTag decode moved on ~0.6 % of looks, and a moved decode
is a moved belief and, some minutes later, a different walk.
`offsamples="0"` in the house's `<visual>` block, which the quadruped's
world inherits, is the fix, and these tests are what keeps it.
"""

import hashlib
import os
import warnings

import mujoco
import numpy as np
import pytest

from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.legs import world as lw
from pluggybot.rack import tags

#: Where the nose camera looks across the living room's lit, shadowed floor
#: -- where MSAA's frames measured apart.
LIVING_ROOM = (-1.0, 1.0, 0.5)

#: Whether MSAA renders that scene differently from one render to the next,
#: by the RASTERISER that draws it (`GL_RENDERER`, lower case), never the
#: backend: EGL is a GPU on one box and llvmpipe on another (issue #440).
#: MEASURED at LIVING_ROOM, offsamples 4: #110's GTX 1660 SUPER, ten renders
#: ten images; Mesa's Intel driver on Meteor Lake and llvmpipe (osmesa, so
#: the deployed box), one image of 64 each -- and no #110 there to fix.
MSAA_VARIES = {"gtx 1660 super": True, "intel(r) graphics (mtl)": False,
               "llvmpipe": False}


@pytest.fixture(scope="module")
def model():
  return lw.home_spec().compile()


def _frames(model, n: int, pose, camera: str = "nav_eye") -> tuple[set[str], str]:
  """The distinct images of `n` renders, and the rasteriser that drew them."""
  data = mujoco.MjData(model)
  lw.stand(model, data, "", *pose)
  renderer = mujoco.Renderer(model, 720, 1280)
  # Its context is current. Read before the first render, which leaves a
  # GL_INVALID_ENUM pending that PyOpenGL would raise against this call.
  from OpenGL import GL
  rasteriser = GL.glGetString(GL.GL_RENDERER).decode()
  seen = set()
  for _ in range(n):
    renderer.update_scene(data, camera=camera)
    seen.add(hashlib.sha256(np.ascontiguousarray(renderer.render()).tobytes())
             .hexdigest())
  renderer.close()
  return seen, rasteriser


def test_the_nose_camera_renders_one_static_scene_to_one_image(model):
  assert model.vis.quality.offsamples == 0, \
    "the robot's cameras render with MSAA on"
  assert len(_frames(model, 8, LIVING_ROOM)[0]) == 1, "the same scene rendered differently"


def test_multisampling_varies_the_frames_where_it_was_measured_to(model, record_property):
  """The premise pin: put MSAA back on the shipped model and it reaches the
  renderer (no frame is the plain one), and the frames differ again where
  that was measured -- and stay one image where it was not. Without it the
  test above could pass for a reason that has nothing to do with the
  setting, as it does, harmlessly, wherever MSAA never varied."""
  plain, rasteriser = _frames(model, 1, LIVING_ROOM)
  model.vis.quality.offsamples = 4
  try:
    distinct, _ = _frames(model, 16, LIVING_ROOM)
  finally:
    model.vis.quality.offsamples = 0
  record_property("rasteriser", rasteriser)
  where = f"{rasteriser} (MUJOCO_GL={os.environ.get('MUJOCO_GL')})"
  assert not plain & distinct, (
    f"offsamples 4 rendered the plain image on {where}: MSAA never reached "
    "the renderer, so nothing here measures it")
  varies = next((v for k, v in MSAA_VARIES.items() if k in rasteriser.lower()), None)
  seen = f"16 MSAA renders of one scene gave {len(distinct)} distinct image(s) on {where}"
  if varies is None:
    warnings.warn(f"{seen}, a rasteriser not in MSAA_VARIES: measure it and add it")
    return
  assert (len(distinct) > 1) == varies, (
    f"{seen}, measured to give {'more than one' if varies else 'one'} -- the "
    "premise of the fix has changed here; re-measure with "
    "scripts/determinism_spike.py before trusting either")


def test_the_tag_decode_is_the_same_on_the_same_frame(model):
  """Downstream of the render: the shared detector, two threads and all,
  gives one answer for one image -- so with identical frames the belief
  cannot drift. The robot at a bay of its rack, reading the rack's tags."""
  data = mujoco.MjData(model)
  lw.stand(model, data, "", *dk.compose(lw.rack_pose(), rk.work_pose(rk.DEFAULT, 0)))
  eye = tags.TagDetector(model, "nav_eye", tag_size=tags.DOCK_TAG_SIZE)
  try:
    answers = {repr(sorted((k, tuple(round(x, 9) for x in v["t"]))
                           for k, v in eye.detect(data).items()))
               for _ in range(6)}
  finally:
    eye.close()
  assert len(answers) == 1
  assert "[]" not in answers, "no tag in view -- the pose is wrong"
