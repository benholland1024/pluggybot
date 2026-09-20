"""The tag camera renders without shadows (rooftop-media-2026 #296).

MEASURED on the deploy box (MUJOCO_GL=osmesa, llvmpipe): one 1280x720
frame of the home world costs 1113 ms with shadows and 32 ms without --
sixteen lights each casting into a 4096^2 shadow map, on a software
rasteriser. The served pair spent ~80 % of its CPU there and ran at 0.23x
real time, and the site's real-time clock then played a frame of motion
and froze until the next. The premise cannot be reproduced on an EGL
box; what is pinned is the flag, that it survives `update_scene`, and --
the second bug the same look found -- that a recompile moves the rack
finder's camera too.
"""

import mujoco

from pluggybot.mission.mission import HubMission
from pluggybot.rack.tags import TagDetector
from pluggybot.robot import FIRST

F = mujoco.mjtRndFlag


def test_the_detector_renders_without_shadows_or_reflections_and_keeps_it_so():
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  # The world is what makes the pass dear: many shadow-casting lights.
  assert model.nlight >= 8 and all(model.light_castshadow)
  det = TagDetector(model, "dock_eye")
  assert det.renderer.scene.flags[F.mjRND_SHADOW] == 0
  assert det.renderer.scene.flags[F.mjRND_REFLECTION] == 0
  # `update_scene` runs on every look and must not put them back.
  det.detect(data)
  det.detect(data)
  assert det.renderer.scene.flags[F.mjRND_SHADOW] == 0
  assert det.renderer.scene.flags[F.mjRND_REFLECTION] == 0
  # ...and a fresh Renderer's default is the opposite, so this is a choice.
  plain = mujoco.Renderer(model, 720, 1280)
  assert plain.scene.flags[F.mjRND_SHADOW] == 1
  plain.close()
  det.close()


def test_a_recompile_moves_the_rack_finders_camera_too():
  """The rack finder kept the OLD model's renderer across a rebind (issue
  #168's seam): a stale world, rendered. Shown to fail without the fix:
  `after is before`."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False, handle=FIRST)
  mission.start_discovery()
  before = mission.finder.spotter.detector.renderer
  new_model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  mission.rebind(new_model, mujoco.MjData(new_model))
  after = mission.finder.spotter.detector.renderer
  assert after is not before and after.model is new_model
  assert mission.tags.detector.renderer.model is new_model
  mission.close()
