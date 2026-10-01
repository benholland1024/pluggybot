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


@pytest.fixture(scope="module")
def model():
  return lw.home_spec().compile()


def _frames(model, n: int, pose, camera: str = "nav_eye") -> set[str]:
  data = mujoco.MjData(model)
  lw.stand(model, data, "", *pose)
  renderer = mujoco.Renderer(model, 720, 1280)
  seen = set()
  for _ in range(n):
    renderer.update_scene(data, camera=camera)
    seen.add(hashlib.sha256(np.ascontiguousarray(renderer.render()).tobytes())
             .hexdigest())
  renderer.close()
  return seen


def test_the_nose_camera_renders_one_static_scene_to_one_image(model):
  assert model.vis.quality.offsamples == 0, \
    "the robot's cameras render with MSAA on"
  assert len(_frames(model, 8, LIVING_ROOM)) == 1, "the same scene rendered differently"


def test_multisampling_is_what_made_the_frames_differ(model):
  """The premise pin: put MSAA back on the shipped model and the frames
  differ again. Without it the test above could pass for a reason that has
  nothing to do with the setting."""
  model.vis.quality.offsamples = 4
  try:
    distinct = _frames(model, 16, LIVING_ROOM)
  finally:
    model.vis.quality.offsamples = 0
  assert len(distinct) > 1, (
    "MSAA rendered identically here -- the premise of the fix has changed; "
    "re-measure with scripts/determinism_spike.py before trusting either")


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
