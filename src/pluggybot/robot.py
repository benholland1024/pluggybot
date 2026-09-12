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


def attach_robot(spec: mujoco.MjSpec, pos, prefix: str = SECOND_PREFIX) -> None:
  """Attach a second robot to a world spec at `pos`, every element prefixed.
  The attach is one call; the WORK is that everything reads through a
  handle afterwards."""
  frame = spec.worldbody.add_frame()
  frame.pos = [float(pos[0]), float(pos[1]), 0.0]
  spec.attach(robot_spec(), prefix=prefix, frame=frame)


def world_with_robots(path: str, second_at=None,
                      prefix: str = SECOND_PREFIX) -> mujoco.MjModel:
  """A world's model with, optionally, a second robot parked at `second_at`.
  `None` compiles the world exactly as `from_xml_path` would."""
  if second_at is None:
    return mujoco.MjModel.from_xml_path(path)
  spec = mujoco.MjSpec.from_file(path)
  attach_robot(spec, second_at, prefix)
  return spec.compile()
