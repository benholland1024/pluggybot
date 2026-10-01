"""The tag camera renders without shadows (rooftop-media-2026 #296).

MEASURED on the deploy box (MUJOCO_GL=osmesa, llvmpipe): one 1280x720
frame of the home world costs 1113 ms with shadows and 32 ms without --
sixteen lights each casting into a 4096^2 shadow map, on a software
rasteriser. The served pair spent ~80 % of its CPU there and ran at 0.23x
real time. The premise cannot be reproduced on an EGL box; what is pinned
is the flag, and that it survives `update_scene`.
"""

import mujoco

from pluggybot.legs import world as lw
from pluggybot.rack.tags import DOCK_TAG_SIZE, TagDetector

F = mujoco.mjtRndFlag


def test_the_detector_renders_without_shadows_or_reflections_and_keeps_it_so():
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  lw.stand(model, data, "", 1.5, 0.5, 0.0)
  # The world is what makes the pass dear: many shadow-casting lights.
  assert model.nlight >= 8 and all(model.light_castshadow)
  det = TagDetector(model, "nav_eye", tag_size=DOCK_TAG_SIZE)
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
