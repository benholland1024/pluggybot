"""Guards for the single always-on solver policy (issue #3).

PluggyWorld puts two robots in one world, so solver settings cannot be
phase-scoped per robot. The settled policy, measured by scripts/noslip_spike.py
(sweep table in docs/SimNotes.md):

  `noslip_iterations` is 0, always, everywhere. No code mutates solver
  options at runtime. What the per-phase noslip pass had been papering over
  was mostly not solver drift at all: the base was ROLLING under sustained
  tool loads, because a wheel velocity servo commanded 0 resists speed, not
  force. The fix is the parking brake the physical gearbox provides for
  free -- `frictionloss` on the wheel joints (pen square: 63 % -> 99 %
  inked with no solver pass and no step-cost increase).

The sweep also measured why always-on noslip LOST: at >=1 iteration the
robot-level coupling half-seats under +/-3 mm lateral jitter (on the fork
but not electrically powered), and at 3 iterations returns start missing
the tray by ~35 mm -- the peg seats by sliding, and the pass suppresses
exactly that.
"""

import mujoco
import pytest

from pluggybot.control import W_BREAKAWAY, turn_command
from pluggybot.hub.coupling import HUB_STATION_YS
from pluggybot.hub.drawing import PenPlotter, square_path
from pluggybot.hub.gripper import ClawTool
from pluggybot.hub.swap import HubSwap

WORLDS = ("models/world.xml", "models/world_fork.xml", "models/playground.xml",
          "models/room_1.xml", "models/room_hub.xml", "models/hub_world.xml")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def test_no_world_enables_noslip():
  """One solver policy: the pass is off in every world, including the
  GENERATED ones -- the peg's mu=0.4-in-the-spike / 1.0-in-the-world bug
  is exactly the class of divergence this guards against."""
  for path in WORLDS:
    model = mujoco.MjModel.from_xml_path(path)
    assert model.opt.noslip_iterations == 0, (
      f"{path} enables noslip_iterations="
      f"{model.opt.noslip_iterations}; the policy is 0 everywhere")


def test_tool_controllers_never_mutate_the_solver(hub_model):
  """A solver mode is global state: setting it per-object once leaked a
  mutated model through a module-scoped fixture into a later test's
  coupling pick, and in a shared two-robot world one robot's 'phase' would
  be everyone's physics. The old per-phase toggles must stay no-ops."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  plotter = PenPlotter(hub_model, data, swap)
  claw = ClawTool(hub_model, data, swap)
  for on in (True, False, True):
    plotter.contact_physics(on)
    claw.grasp_physics(on)
    assert hub_model.opt.noslip_iterations == 0, \
      "a tool controller mutated the global solver policy"


def test_wheels_have_a_parking_brake(hub_model):
  """The plotter's half of the policy: a velocity servo commanded 0 resists
  SPEED, not force, so any sustained push slowly rolls the parked base --
  pen drag walked the plotter sideways (63 % inked), and the stow's lower
  phase walked the base 9 mm into the rack and jammed the peg past the
  tray. `frictionloss` on the wheel joints is the gearbox's Coulomb
  friction, the parking brake the real 37D has anyway."""
  for name in WHEEL_JOINTS:
    dof = hub_model.joint(name).dofadr[0]
    assert hub_model.dof_frictionloss[dof] >= 0.05, (
      f"{name} has no parking brake (frictionloss="
      f"{hub_model.dof_frictionloss[dof]}); the parked base will roll "
      f"under sustained tool loads")


def test_turn_command_clears_the_stiction_deadband():
  """The brake's side effect, caught by Ben watching pickup.py: a P-turn
  controller that shrinks its command with the error parks itself in the
  stiction deadband -- the wheel servo's torque is kv*(target-actual), so a
  target under frictionloss/kv moves a stopped wheel not at all, and _face
  took 55 s to settle 0.23 deg (9.5 s before the brake). turn_command floors
  every nonzero command at the breakaway yaw rate; commanding less is
  indistinguishable from commanding zero, so the floor costs nothing."""
  assert turn_command(0.0) == 0.0, "zero command must stay zero"
  small = turn_command(0.004)
  assert small >= W_BREAKAWAY, \
    f"near-tolerance command {small:.3f} is inside the stiction deadband"
  assert turn_command(-0.004) <= -W_BREAKAWAY, "floor must preserve sign"
  assert turn_command(1.0) == pytest.approx(0.5), "the clamp must survive"


def test_face_settles_promptly_despite_the_brake(hub_model):
  """End to end: squaring up must not crawl through the deadband. Measured
  55 s on the braked wheels with the raw P-controller, ~10 s with the
  breakaway floor -- the 25 s budget has 2x margin over the fix and 2x
  slack under the failure."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  claw = ClawTool(hub_model, data, swap)
  t0 = float(data.time)
  residual = claw._face(0.35)          # ~20 deg off the parked heading
  took = float(data.time) - t0
  assert abs(residual) < 0.016, f"never squared up ({residual:.3f} rad off)"
  assert took < 25.0, (
    f"facing took {took:.1f} s of sim time -- the turn controller is "
    f"crawling through the stiction deadband again")


@pytest.mark.slow
def test_pen_square_inks_without_solver_help(hub_model):
  """The failure mode the spike surfaced, pinned end to end: the SQUARE is
  the figure that holds an extreme carriage offset for whole edges, so it
  is the one that exposes creep under sustained load. Measured: 63 % inked
  / 1.94 mm form without the wheel brake (noslip 0), 99 % / 0.60 with it --
  better than the 2x-cost noslip=3 pass ever bought (93 % / 0.61)."""
  data = mujoco.MjData(hub_model)
  swap = HubSwap(hub_model, data)
  swap.place_at_standoff(HUB_STATION_YS[2])    # bay C, the pen module
  swap.pick()
  plotter = PenPlotter(hub_model, data, swap)
  assert plotter.drive_to_board(), "never reached the board"
  r = plotter.draw(square_path())
  assert r["drew"], f"never got the pen on the board: {r}"
  assert hub_model.opt.noslip_iterations == 0, \
    "the draw ran with solver help; this test exists to measure it without"
  assert r["inked_fraction"] > 0.85, (
    f"square only {r['inked_fraction']:.0%} inked at the global policy -- "
    f"the parked base is creeping again (no brake measured 63 %)")
  assert r["form_rms_mm"] < 1.5, (
    f"square form error {r['form_rms_mm']:.2f} mm -- creep under sustained "
    f"load is back (no brake measured 1.94 mm)")
