"""One robot's names in a world that may hold two (issue #167, M12).

Every body, joint, actuator, site, camera and sensor of the robot lives in
`models/pluggybot_fork.xml` under bare names -- `chassis`, `lift`,
`dock_eye` -- and the FIRST robot keeps them, so a single-robot world is the
world it always was, byte for byte (the parity instrument, Evaluation.md §1).
A second robot is the same file ATTACHED with a prefix (`MjSpec.attach`),
so its names are `r2_chassis`, `r2_lift`, `r2_dock_eye`; a `RobotHandle`
is that prefix, and every class that resolves a robot element does so
through `handle.el(name)` rather than the bare string. What is NOT per
robot: the rack, its bays and the modules -- they are the world's, shared,
and contended for by the minds rather than namespaced (the issue's rule).

Measured on the way in: a second robot parked in the same world leaves the
first robot's trajectory byte-identical over a spin, a drive and a drawing,
and a full `home` day diverges at t = 98 s by 1e-15 -- the constraint
solver's rounding with an extra island, not a code path (SimNotes, "A
second robot perturbs the first at the last bit"). The namespacing itself
is parity-exact: the refactored code flown ALONE hashes identical to the
day before it.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco

from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The robot's model file, a `<mujocoinclude>` every world includes for its
#: first robot.
ROBOT_XML = Path(__file__).resolve().parents[2] / "models" / "pluggybot_fork.xml"
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
    0 for the first robot, because it is the first body in every world."""
    root = model.body(self.root).id
    return int(model.jnt_qposadr[model.body_jntadr[root]])

  def dof_adr(self, model) -> int:
    root = model.body(self.root).id
    return int(model.jnt_dofadr[model.body_jntadr[root]])


FIRST = RobotHandle("")
SECOND = RobotHandle(SECOND_PREFIX)


def robot_spec() -> mujoco.MjSpec:
  """The robot file as a model spec of its own (it is an include)."""
  text = ROBOT_XML.read_text()
  text = text.replace("<mujocoinclude>", "<mujoco>").replace(
    "</mujocoinclude>", "</mujoco>")
  return mujoco.MjSpec.from_string(text)


#: The first robot's paint, as `models/pluggybot_fork.xml` has it on the
#: chassis and the head mount: every geom carrying exactly this colour is
#: the robot's LIVERY, and an attached robot is repainted there and nowhere
#: else (the battery's red, the tags, the hardware greys stay).
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


def attach_robot(spec: mujoco.MjSpec, pos, prefix: str = SECOND_PREFIX,
                 chassis_rgba=SECOND_CHASSIS_RGBA) -> None:
  """Attach a second robot to a world spec at `pos`, every element prefixed
  and its livery repainted `chassis_rgba` (None keeps the first robot's).
  The attach is one call; the WORK is that everything reads through a
  handle afterwards."""
  robot = robot_spec()
  if chassis_rgba is not None:
    paint(robot, chassis_rgba)
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), 0.0]
  spec.attach(robot, prefix=prefix, frame=frame)


PAIR_SUFFIX = "_pair"


def pair_model_name(model_name: str) -> str:
  """The wire's name for a world with the second robot attached (protocol
  0.20.0): `room_hub` -> `room_hub_pair`. A replayer picks its scene off the
  header's `model`, and the second robot's bodies are in no single-robot
  scene, so the pair world is a world of its own to a consumer -- one scene
  fixture and one recording under this name, like any other."""
  return f"{model_name}{PAIR_SUFFIX}"


def world_spec(path: str, second_at=None, prefix: str = SECOND_PREFIX,
               chassis_rgba=SECOND_CHASSIS_RGBA) -> mujoco.MjSpec:
  """A world's SPEC with, optionally, a second robot parked at `second_at`.
  Kept by the lifecycle (issue #168 slice C) so a tool can be hung mid-run
  by editing it and recompiling. MEASURED: `spec.compile()` is trajectory-
  identical to `MjModel.from_xml_path` (tests/test_recompile.py)."""
  spec = mujoco.MjSpec.from_file(path)
  if second_at is not None:
    attach_robot(spec, second_at, prefix, chassis_rgba)
  return spec


def world_with_robots(path: str, second_at=None,
                      prefix: str = SECOND_PREFIX,
                      chassis_rgba=SECOND_CHASSIS_RGBA) -> mujoco.MjModel:
  """A world's model with, optionally, a second robot parked at `second_at`
  in its own livery. `None` compiles the world exactly as `from_xml_path`
  would."""
  if second_at is None:
    return mujoco.MjModel.from_xml_path(path)
  return world_spec(path, second_at, prefix, chassis_rgba).compile()
