"""Guards for the single always-on solver policy (issue #3).

PluggyWorld puts two robots in one world, so solver settings cannot be
phase-scoped per robot. The settled policy, measured by
scripts/noslip_spike.py (the sweep is in docs/SimNotes.md):
`noslip_iterations` is 0, always, everywhere, and no code mutates solver
options at runtime. At >= 1 iteration the coupling half-seats under +/-3 mm
of lateral jitter (on the fork but not powered): the peg seats by SLIDING,
and the pass suppresses exactly that.
"""

import ast
import pathlib

import mujoco

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pluggybot"


def _worlds():
  """Every world the repo builds: the house, the served world alone and as
  a pair, the quadruped alone, its dock's rig and the coupling's."""
  from pluggybot.legs import dock as dk
  from pluggybot.legs import world as lw
  from pluggybot.rack import coupling
  yield "models/home_world.xml", mujoco.MjModel.from_xml_path("models/home_world.xml")
  yield "models/quadruped.xml", mujoco.MjModel.from_xml_path("models/quadruped.xml")
  yield "home_quad", lw.home_spec().compile()
  yield "home_quad_pair", lw.home_spec(second_at=(0.0, 1.0)).compile()
  xml, assets = dk.world_xml()
  yield "the dock's rig", mujoco.MjModel.from_xml_string(xml, assets)
  yield "the coupling's rig", mujoco.MjModel.from_xml_string(coupling.scene_xml())


def test_no_world_enables_noslip():
  """One solver policy: the pass is off in every world, including the
  GENERATED ones -- the peg's mu=0.4-in-the-spike / 1.0-in-the-world bug
  is exactly the class of divergence this guards against."""
  for name, model in _worlds():
    assert model.opt.noslip_iterations == 0, (
      f"{name} enables noslip_iterations="
      f"{model.opt.noslip_iterations}; the policy is 0 everywhere")


def _solver_writes(tree) -> list[int]:
  """Lines assigning `<anything>.opt.<option>`: a solver option set in code."""
  lines = []
  for node in ast.walk(tree):
    targets = (node.targets if isinstance(node, ast.Assign)
               else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
               else [])
    for t in targets:
      if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Attribute)
          and t.value.attr == "opt"):
        lines.append(node.lineno)
  return lines


def test_no_code_mutates_the_solver_at_runtime():
  """A solver mode is global state: setting it per-object once leaked a
  mutated model through a module-scoped fixture into a later test's
  coupling pick, and in a shared two-robot world one robot's 'phase' would
  be everyone's physics. Walked off the syntax tree of `src/`."""
  bad = [f"{p.relative_to(SRC.parent)}:{line}"
         for p in sorted(SRC.rglob("*.py"))
         for line in _solver_writes(ast.parse(p.read_text()))]
  assert not bad, "solver options set in code:\n  " + "\n  ".join(bad)
  # ...and the walk finds one where there is one
  assert _solver_writes(ast.parse("model.opt.noslip_iterations = 3\n")) == [1]
  assert _solver_writes(ast.parse("self.model.opt.timestep *= 0.5\n")) == [1]
