"""Guards for the mission state machine in scripts/lifecycle.py.

Loaded by path, like tests/test_dataset_generation.py, since scripts/ is not
part of the installed package.
"""

import importlib.util
import math
from pathlib import Path

import mujoco
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "lifecycle.py"

# Nose-on to L-box-north, whose near face is x=3.5 (body at x=4, half-len 0.5).
# The camera sits 0.05 m behind the body origin, so it reads 3.55 - x metres of
# range; x=3.34 gives 0.21 m (inside the 0.25 m reflex threshold) while the
# chassis front edge stops at 3.46, clear of the box.
NOSE_ON_TO_LBOX = (3.34, 1.0, 0.0)


def _load():
  spec = importlib.util.spec_from_file_location("lifecycle", _SCRIPT)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


@pytest.fixture(scope="module")
def lifecycle_module():
  return _load()


@pytest.fixture
def sim(lifecycle_module):
  # weights=None -> no detector; these tests are about navigation safety.
  return lifecycle_module.Lifecycle(
    headless=True, max_sim_time=10.0, weights=None, explore_budget=10.0)


def _place(sim, x, y, yaw):
  mujoco.mj_resetData(sim.model, sim.data)
  sim.data.qpos[0:3] = [x, y, 0.045]
  sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(sim.model, sim.data)


def test_reflex_arms_while_spinning(sim):
  """The reflex must fire in every maneuver, not just while following
  waypoints. explore.py armed it only when waypoints existed; measured, its
  look-around spins passed within 0.257 m of the L-box -- 7 mm outside the
  0.25 m threshold. A slightly different trajectory ground the chassis for
  503 steps. Spinning is not a reason to stop watching where you are."""
  _place(sim, *NOSE_ON_TO_LBOX)
  sim.mode = "spin"
  sim.waypoints = []                  # the condition that used to disarm it
  sim.perceive()
  assert sim.backoff_until > 0.0, "reflex did not arm during a spin"


def test_reflex_ignores_open_space(sim):
  """...and must not fire when nothing is close, or the robot never moves."""
  _place(sim, 1.0, 3.0, math.pi / 2)  # facing north up an empty corridor
  sim.mode = "spin"
  sim.waypoints = []
  sim.perceive()
  assert sim.backoff_until == 0.0


def test_backoff_is_a_bounded_pulse(sim):
  """Backoff must not re-arm while already backing off: an open-ended
  reverse would just drive into whatever is behind the robot."""
  _place(sim, *NOSE_ON_TO_LBOX)
  sim.waypoints = []
  sim.perceive()
  first = sim.backoff_until
  assert first > 0.0
  sim.step_count = 0                  # force another scan on the next perceive
  sim.perceive()
  assert sim.backoff_until == first, "reflex re-armed mid-backoff"


def test_leaving_explore_without_outlets_finishes(sim):
  """No confirmed outlet means there is nowhere to charge: the mission ends
  rather than entering GO_CHARGE with target=None (which would crash)."""
  sim.explore()                       # nothing seen yet
  sim.leave_explore("battery-low")
  assert sim.state == "DONE"
  assert sim.target is None


def test_leaving_explore_with_an_outlet_goes_to_charge(sim):
  for _ in range(3):                  # three sightings = confirmed
    sim.landmarks.add_sighting(-1.99, 1.0, 0.24, seen_from=(-1.2, 1.0))
  sim.leave_explore("battery-low")
  assert sim.state == "GO_CHARGE"
  assert sim.target is not None
  sx, sy, _ = sim.standoff_of(sim.target)
  assert sx > -1.99, "standoff must sit on the room side of the outlet"


def test_go_charge_hands_off_on_arrival(sim):
  """Standing on the standoff point must advance to FACE_OUTLET. Arrival is
  checked before replanning: A* from the goal cell to itself returns a
  one-cell path that follow_waypoints instantly empties, so replanning first
  would oscillate between 'arrived' and 'replan' forever."""
  # Odometry starts at the origin, so put the standoff point there: an outlet
  # 0.6 m ahead whose wall faces back toward the robot.
  sim.target = sim.landmarks.add_sighting(0.6, 0.0, 0.24, seen_from=(0.0, 0.0))
  sim.standoff_dir = (-1.0, 0.0)
  sim.waypoints = []
  v, w = sim.go_charge()
  assert sim.state == "FACE_OUTLET"
  assert (v, w) == (0.0, 0.0)


def test_face_outlet_turns_then_stops(sim, lifecycle_module):
  """Pivot while misaligned, stop inside FACING_TOLERANCE. The tolerance is
  the docking controller's yaw budget being spent, so it must be small."""
  sim.state = "FACE_OUTLET"
  # Outlet behind the robot: target heading is pi, odometry heading is 0.
  sim.target = sim.landmarks.add_sighting(1.0, 0.0, 0.24, seen_from=(0.0, 0.0))
  sim.standoff_dir = (1.0, 0.0)
  v, w = sim.face_outlet()
  assert sim.state == "FACE_OUTLET"           # still turning, not handed off
  assert v == 0.0 and abs(w) > 0.0

  # Outlet dead ahead of the robot's current heading: already squared up --
  # and the mission continues into DOCK, no longer ending at the pose.
  sim.target = sim.landmarks.add_sighting(-1.0, 0.0, 0.24, seen_from=(0.0, 0.0))
  sim.standoff_dir = (-1.0, 0.0)
  v, w = sim.face_outlet()
  assert sim.state == "DOCK"
  assert sim.dock_stage == "align"
  assert (v, w) == (0.0, 0.0)
  assert abs(sim.final_heading_error) <= lifecycle_module.FACING_TOLERANCE


def test_mechanical_dock_from_perfect_standoff(lifecycle_module):
  """The full mechanical docking stack -- creep, feeler squaring, press-sag
  compensation, RCC funnel capture, charge-contact detection -- must seat the
  plug from a perfect standoff with no vision in the loop (weights=None means
  the servo runs a straight creep). This is the deterministic floor under the
  visual controller: if THIS fails, the machine broke, not the model."""
  lc = lifecycle_module
  sim = lc.Lifecycle(headless=True, max_sim_time=9999, weights=None,
                     explore_budget=9999)
  ox, oy, oz, yaw = -1.953, 1.0, 0.26, math.pi
  sx = ox + 0.6 - 0.05 * math.sin(yaw)
  sy = oy + 0.05 * math.cos(yaw)
  sim.data.qpos[0:3] = [sx + 0.08 * math.cos(yaw), sy + 0.08 * math.sin(yaw), 0.045]
  sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
  mujoco.mj_forward(sim.model, sim.data)
  sim.reckoner.x, sim.reckoner.y, sim.reckoner.theta = sx, sy, yaw
  sim.target = sim.landmarks.add_sighting(ox, oy, oz, seen_from=(sx, sy))
  sim.state = "DOCK"
  sim.dock_stage = "align"
  sim.stage_t0 = 0.0
  sim.model.opt.timestep = lc.DOCK_TIMESTEP
  sim.model.actuator_forcerange[sim.model.actuator("arm").id] = \
      [-lc.DOCK_ARM_FORCE, lc.DOCK_ARM_FORCE]
  t0 = sim.data.time
  while sim.data.time - t0 < 60 and sim.state == "DOCK":
    sim.perceive()
    v, w = sim.dock()
    sim.actuate(v, w)
    mujoco.mj_step(sim.model, sim.data)
    sim.step_count += 1
  assert sim.docked, f"mechanical dock failed (stage {sim.dock_stage})"
  # and the charge criterion is telling the truth: face near the well floor
  face = sim.data.site_xpos[sim.model.site("plug_face").id]
  assert abs(face[1] - oy) < 0.004, "seated plug should be laterally centred"
  assert abs(face[2] - oz) < 0.006, "seated plug should be at socket height"


def test_no_map_sightings_while_docking(sim):
  """Sightings logged mid-dock are poison: the head camera sits closer than
  its calibrated regime, and a long jam racks up dozens of biased sightings
  that DRAG the landmark -- measured: outlet C's estimate ended 8 cm inside
  the wall after failed dock attempts, wall_normal() over the in-wall point
  gave a -67 deg standoff, and every subsequent approach was worse. The map
  spotter must go quiet in DOCK/CHARGE."""
  class FakeSpotter:
    def spot(self, data, pose):
      return [(1.0, 2.0, 0.30, 0.99)]
  sim.spotter = FakeSpotter()
  sim.next_spot = 0.0
  sim.state = "DOCK"
  sim.perceive()
  assert len(sim.landmarks.landmarks) == 0, "DOCK-state sighting was recorded"
  sim.state = "EXPLORE"
  sim.perceive()
  assert len(sim.landmarks.landmarks) == 1, "EXPLORE sighting should record"


def test_low_battery_triggers_charge_seeking(sim):
  """The battery is now the real trigger the explore_budget timer stood in
  for: a drained cell must end EXPLORE with the battery-low reason."""
  sim.explore_budget = 1e9              # timer out of the way
  sim.battery.energy_wh = sim.battery.capacity_wh * 0.10
  for _ in range(3):
    sim.landmarks.add_sighting(-1.99, 1.0, 0.24, seen_from=(-1.2, 1.0))
  sim.explore()
  assert sim.state == "GO_CHARGE"
  assert sim.explore_done_reason == "battery-low"


def test_charge_reserve_is_absolute_energy_not_fraction(sim):
  """The go-charge reserve must be ABSOLUTE energy: the drive home + dock
  costs a fixed ~0.2 Wh in this room regardless of pack size. The original
  35%-of-capacity trigger starved a 0.55 Wh pack mid-insertion (dead at
  81.5 s, four seconds short of charge contact) because 35% of a small pack
  is not the mission's cost. At 0.24 Wh remaining -- 44% of a small pack, so
  the fraction rule would sit tight -- the mission must head home."""
  sim.explore_budget = 1e9
  sim.battery.capacity_wh = 0.55
  sim.battery.energy_wh = 0.24
  for _ in range(3):
    sim.landmarks.add_sighting(-1.99, 1.0, 0.24, seen_from=(-1.2, 1.0))
  sim.explore()
  assert sim.state == "GO_CHARGE"
  assert sim.explore_done_reason == "battery-low"


def test_charge_fills_then_resumes_exploring(sim, monkeypatch):
  """CHARGE must actually charge (only while contact reports true), undock,
  and -- because the map was NOT finished -- go back to exploring with the
  cruise physics restored. The contact signal is faked; the real seat is
  covered by the mechanical dock test."""
  lc = _load()
  sim.state = "CHARGE"
  sim.charge_stage = "charging"
  sim.explore_done_reason = "battery-low"
  sim.model.opt.timestep = lc.DOCK_TIMESTEP         # as DOCK leaves things
  sim.battery.capacity_wh = 0.05                    # tiny cell: fast test
  sim.battery.energy_wh = sim.battery.capacity_wh * 0.2
  monkeypatch.setattr(sim, "charging_contact", lambda: True)
  t0 = sim.data.time
  while sim.state == "CHARGE" and sim.data.time - t0 < 120:
    sim.perceive()
    v, w = sim.charge()
    sim.actuate(v, w)
    mujoco.mj_step(sim.model, sim.data)
    sim.step_count += 1
  assert sim.battery.fraction >= lc.CHARGED
  assert sim.state == "EXPLORE", "unfinished map must resume exploring"
  assert sim.model.opt.timestep == sim.cruise_timestep, \
    "cruise physics must be restored after undocking"
  assert sim.mode == "spin", "resume opens with a look-around"


def test_charge_with_finished_map_ends_mission(sim, monkeypatch):
  lc = _load()
  sim.state = "CHARGE"
  sim.charge_stage = "charging"
  sim.explore_done_reason = "no-frontiers"
  sim.battery.capacity_wh = 0.05                    # tiny cell: fast test
  sim.battery.energy_wh = sim.battery.capacity_wh * 0.85
  monkeypatch.setattr(sim, "charging_contact", lambda: True)
  t0 = sim.data.time
  while sim.state == "CHARGE" and sim.data.time - t0 < 120:
    sim.perceive()
    v, w = sim.charge()
    sim.actuate(v, w)
    mujoco.mj_step(sim.model, sim.data)
    sim.step_count += 1
  assert sim.state == "DONE"
  assert sim.battery.fraction >= lc.CHARGED


def test_charging_keeps_the_press_on(sim, monkeypatch, lifecycle_module):
  """While charging, the wheels must keep the gentle insert-stage press.
  Wheels-at-zero was measured fatal: the RCC relaxes, the pin floats off the
  socket floor, contact drops, and the mission oscillated dock->charge->lost
  14 times before starving at 1%. Press is the springless sim socket's
  analog of a real Schuko's pin-gripping contacts."""
  sim.state = "CHARGE"
  sim.charge_stage = "charging"
  sim.battery.energy_wh = sim.battery.capacity_wh * 0.2
  monkeypatch.setattr(sim, "charging_contact", lambda: True)
  sim.data.ctrl[sim.model.actuator("arm").id] = 0.06   # as a frozen seat would
  sim.perceive()
  v, w = sim.charge()
  assert v > 0.0, "charging must press, or the plug floats off the contacts"
  assert v <= lifecycle_module.DOCK_PRESS * 0.045 + 1e-9, "press stays gentle"
  assert sim.data.ctrl[sim.model.actuator("arm").id] == pytest.approx(0.20), \
    "the ARM must keep pressing too: frozen at ext, the wheel press routes " \
    "through the prongs and the pin floats off the contacts"


def test_lost_contact_mid_charge_returns_to_dock(sim):
  """A plug that creeps out mid-charge is a DOCK problem, not a new mission
  phase: after the timeout with no contact, CHARGE hands back to the servo."""
  sim.state = "CHARGE"
  sim.charge_stage = "charging"
  sim.last_charge_contact = sim.data.time
  sim.battery.energy_wh = sim.battery.capacity_wh * 0.2
  # charging_contact stays honest (and false: nothing is docked)
  for _ in range(int(6.0 / sim.model.opt.timestep)):
    sim.perceive()
    v, w = sim.charge()
    if sim.state != "CHARGE":
      break
    sim.actuate(v, w)
    mujoco.mj_step(sim.model, sim.data)
    sim.step_count += 1
  assert sim.state == "DOCK" and sim.dock_stage == "servo"


def test_dead_battery_ends_the_mission(lifecycle_module):
  sim = lifecycle_module.Lifecycle(
    headless=True, max_sim_time=5.0, weights=None, explore_budget=1e9)
  sim.battery.energy_wh = 1e-6
  sim.run()
  assert sim.state == "DONE"
  assert sim.explore_done_reason == "battery-dead"


def test_jam_benches_but_phantom_forgets(sim):
  """A jam at a real outlet must NOT erase it from the map (watched failure:
  two jams deleted a true charging spot forever); only a phantom -- a target
  that fails close-range verification -- is forgotten."""
  for _ in range(3):
    real = sim.landmarks.add_sighting(-1.95, 1.0, 0.26, seen_from=(-1.2, 1.0))
  sim.target = real
  sim.reject_target("jammed", forget=False)
  assert real in sim.landmarks.landmarks, "jam erased a real outlet"
  assert id(real) in sim.set_aside, "jammed outlet should be benched"
  for _ in range(3):
    ghost = sim.landmarks.add_sighting(4.5, -1.95, 0.28, seen_from=(4.5, -1.2))
  sim.target = ghost
  sim.reject_target("phantom", forget=True)
  assert ghost not in sim.landmarks.landmarks, "phantom survived"
