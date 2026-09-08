"""The robot's cameras render the same frame twice (issue #110).

Evaluation.md §1 rests on the world being the same world twice, and the
first committed `scripted` series broke it: five identical days, three
trajectories. Traced to the offscreen renderer -- with multisample
antialiasing on, one static scene rendered to a different image every
time (+-1 in a few dozen shadow-edge pixels), the AprilTag decode moved on
~0.6 % of looks, and a moved decode is a moved rack belief and, some
minutes later, a different drive. `offsamples="0"` in the robot models is
the fix, and these tests are what keeps it.
"""

import hashlib

import mujoco
import numpy as np
import pytest

from pluggybot.rack import tags
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.swap import HubSwap

WORLDS = ("models/hub_world.xml", "models/home_world.xml",
          "models/room_hub.xml")


def _frames(model, n: int, camera: str = "dock_eye") -> set[str]:
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  mujoco.mj_forward(model, data)
  renderer = mujoco.Renderer(model, 720, 1280)
  seen = set()
  for _ in range(n):
    renderer.update_scene(data, camera=camera)
    seen.add(hashlib.sha256(np.ascontiguousarray(renderer.render()).tobytes())
             .hexdigest())
  renderer.close()
  return seen


@pytest.mark.parametrize("xml", WORLDS)
def test_the_dock_camera_renders_one_static_scene_to_one_image(xml):
  model = mujoco.MjModel.from_xml_path(xml)
  assert model.vis.quality.offsamples == 0, \
    f"{xml} renders the robot's cameras with MSAA on"
  assert len(_frames(model, 8)) == 1, "the same scene rendered differently"


def test_multisampling_is_what_made_the_frames_differ():
  """The premise pin: put MSAA back on the shipped model and the frames
  differ again. Without it the test above could pass for a reason that has
  nothing to do with the setting."""
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  model.vis.quality.offsamples = 4
  distinct = _frames(model, 8)
  assert len(distinct) > 1, (
    "MSAA rendered identically here -- the premise of the fix has changed; "
    "re-measure with scripts/determinism_spike.py before trusting either")


def test_the_tag_decode_is_the_same_on_the_same_frame():
  """Downstream of the render: the shared detector, two threads and all,
  gives one answer for one image -- so with identical frames the belief
  cannot drift."""
  model = mujoco.MjModel.from_xml_path("models/hub_world.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])
  mujoco.mj_forward(model, data)
  eye = tags.TagDetector(model, "dock_eye")
  answers = {repr(sorted((k, tuple(round(x, 9) for x in v["t"]))
                         for k, v in eye.detect(data).items()))
             for _ in range(6)}
  assert len(answers) == 1
  assert "[]" not in answers, "no tag in view -- the pose is wrong"
