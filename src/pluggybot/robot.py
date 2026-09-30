"""One robot's names in a world that may hold two (issue #167, M12).

Every body, joint, actuator, site, camera and sensor of the robot lives under
bare names -- `pluggybot`, `FL_hip_abd`, `nav_eye` -- and the FIRST robot
keeps them. A second robot is the same model ATTACHED with a prefix
(`MjSpec.attach`), so its names are `r2_pluggybot`, `r2_FL_hip_abd`; a
`RobotHandle` is that prefix,
and every class that resolves a robot element does so through
`handle.el(name)` rather than the bare string. What is NOT per robot: the
rack, its bays and the tools -- they are the world's, shared, and contended
for by the minds rather than namespaced (the issue's rule).
"""

from dataclasses import dataclass

import mujoco

from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The second robot's prefix. Numbered rather than named: the NAME is per
#: instance and rides the header (#39); the prefix is structure.
SECOND_PREFIX = "r2_"


@dataclass(frozen=True)
class RobotHandle:
  """The prefix one robot's elements carry. `""` is the first robot."""

  prefix: str = ""

  def el(self, name: str) -> str:
    """The world's name for this robot's `name`."""
    return f"{self.prefix}{name}"

  @property
  def root(self) -> str:
    """This robot's root body: `pluggybot`, or `r2_pluggybot`."""
    return self.el(ROBOT_ROOT)

  def qpos_adr(self, model) -> int:
    """Where this robot's free joint starts in `qpos` (7 numbers: xyz, wxyz).
    Read by name: a world's robots are attached after its house."""
    root = model.body(self.root).id
    return int(model.jnt_qposadr[model.body_jntadr[root]])

  def dof_adr(self, model) -> int:
    root = model.body(self.root).id
    return int(model.jnt_dofadr[model.body_jntadr[root]])


FIRST = RobotHandle("")
SECOND = RobotHandle(SECOND_PREFIX)


#: The first robot's paint: every geom carrying exactly this colour is the
#: robot's LIVERY, and a second robot is repainted there and nowhere else
#: (the tags and the hardware greys stay).
CHASSIS_RGBA = (0.2, 0.4, 0.8, 1.0)
#: ...and the second robot's: purple, so two robots in one room are told
#: apart at a glance -- on the site, which keeps the producer's colour per
#: geom, and in the other robot's cameras, which render rgba. Paint, not a
#: hint (CLAUDE.md: hints never ride as colours): a real second robot would
#: be painted too.
SECOND_CHASSIS_RGBA = (0.55, 0.3, 0.8, 1.0)


def paint(spec: mujoco.MjSpec, rgba, was=CHASSIS_RGBA) -> int:
  """Repaint every geom of `spec` carrying `was` to `rgba`; the count."""
  n = 0
  for geom in spec.geoms:
    if all(abs(float(a) - b) < 1e-6 for a, b in zip(geom.rgba, was)):
      geom.rgba = list(rgba)
      n += 1
  return n


PAIR_SUFFIX = "_pair"


def pair_model_name(model_name: str) -> str:
  """The wire's name for a world with the second robot attached (protocol
  0.20.0): `room_hub` -> `room_hub_pair`. A replayer picks its scene off the
  header's `model`, and the second robot's bodies are in no single-robot
  scene, so the pair world is a world of its own to a consumer -- one scene
  fixture and one recording under this name, like any other."""
  return f"{model_name}{PAIR_SUFFIX}"


#: The bodies a world's robots may have (`world_config`'s `body`): the
#: quadruped (issue #387), since the rover was deleted (#376).
BODIES = ("quadruped",)


def world_spec(path: str, second_at=None, prefix: str = SECOND_PREFIX,
               chassis_rgba=SECOND_CHASSIS_RGBA,
               body: str = "quadruped") -> mujoco.MjSpec:
  """A world's SPEC with the quadruped and its dock put in, and optionally a
  second robot at `second_at` in its own livery (`legs.world.home_spec`).
  Kept by the lifecycle (issue #168 slice C) so a tool can be hung mid-run
  by editing it and recompiling."""
  if body not in BODIES:
    raise ValueError(f"unknown body {body!r} ({' or '.join(BODIES)})")
  from pluggybot.legs.world import home_spec
  return home_spec(second_at=second_at, second_prefix=prefix,
                   second_rgba=chassis_rgba, path=path)
