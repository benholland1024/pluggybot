"""Guards for the shared driving law (behavior/navigation.py).

These integrate a kinematic unicycle rather than stepping MuJoCo, deliberately:
the failure being pinned here is a property of the CONTROL LAW, not of the
plant, so it reproduces without contacts, slew or wheel friction in the way --
and it runs in milliseconds instead of a minute.
"""

import math

import numpy as np

from pluggybot.behavior.navigation import (
  TERMINAL_CONE, V_MAX, W_MAX, drive_toward,
)
from pluggybot.control import wrap_angle


def roll_out(start, target, slow_radius=None, arrive=0.015,
             dt=0.01, horizon=60.0):
  """Integrate a unicycle under `drive_toward`. Returns (seconds, |turning|,
  arrived) -- turning accumulated as absolute heading change, in radians."""
  x, y, th = start
  t = turned = 0.0
  while t < horizon:
    if math.hypot(target[0] - x, target[1] - y) < arrive:
      return t, turned, True
    v, w = drive_toward((x, y, th), target, slow_radius=slow_radius)
    x += v * math.cos(th) * dt
    y += v * math.sin(th) * dt
    th_new = th + w * dt
    turned += abs(wrap_angle(th_new - th))
    th = th_new
    t += dt
  return t, turned, False


# A 0.20 m hop to a point 90 deg off the bow -- the seed dispenser's errand
# between two points of a row, and the shape that exposed the bug.
SHORT_HOP_START = (0.0, 0.0, 0.0)
SHORT_HOP_TARGET = (0.0, 0.20)


def test_pursuit_orbits_a_nearby_target():
  """The bug, pinned so it cannot come back unnoticed.

  Pure pursuit cannot converge on a destination this close: the robot
  overshoots, the target ends up beside it, and `w` saturates while
  `cos(heading_err)` keeps just enough forward speed alive to fly a circle
  AROUND the point. Measured in the real sim before the fix: heading error
  pinned at 84-87 deg for ten seconds, ~900 deg of turning per 200 mm hop.

  This asserts the DEFECT still exists in the default law, which is what
  makes the terminal-mode test below meaningful. If this ever starts
  failing, pure pursuit has been improved -- re-measure and rewrite both,
  do not just delete it.
  """
  _, turned, arrived = roll_out(SHORT_HOP_START, SHORT_HOP_TARGET)
  assert not arrived or turned > 4 * math.pi, (
    "pure pursuit reached a 0.2 m target without orbiting -- the premise of "
    "the terminal-approach fix no longer holds")


def test_terminal_approach_does_not_orbit():
  """The fix: a hard cone plus a distance taper, which cannot limit-cycle.

  The geometric minimum for this hop is ~90 deg (pivot onto the bearing) and
  the caller squares up afterwards, so anything near a full revolution means
  the orbit is back. Fails at ~11 rad without `slow_radius`.
  """
  t, turned, arrived = roll_out(
    SHORT_HOP_START, SHORT_HOP_TARGET, slow_radius=0.25)
  assert arrived, "terminal approach never reached the target"
  assert turned < math.radians(140), (
    f"terminal approach turned {math.degrees(turned):.0f} deg for a 90 deg "
    "hop -- it is orbiting again")
  assert t < 12.0, f"terminal approach took {t:.1f} s to cover 0.2 m"


def test_terminal_approach_pivots_before_it_translates():
  """Outside the cone the command must translate EXACTLY zero.

  This is the specific property that kills the orbit: the limit cycle lives
  on the sliver of forward speed that `cos(84 deg)` still allows, so a soft
  taper is not enough and the gate has to be hard.
  """
  for err_deg in (35, 60, 84, 90, 150):
    err = math.radians(err_deg)
    v, w = drive_toward((0.0, 0.0, -err), (1.0, 0.0), slow_radius=0.25)
    assert v == 0.0, f"terminal mode still translates at {err_deg} deg off-bow"
    assert abs(w) > 0.0
  v, _ = drive_toward((0.0, 0.0, 0.0), (1.0, 0.0), slow_radius=0.25)
  assert v > 0.0, "terminal mode refuses to move when aimed straight at it"
  assert TERMINAL_CONE < math.radians(90)


def test_terminal_speed_tapers_with_distance():
  """Speed falls to zero over the last `slow_radius` metres, so the robot
  arrives rather than overshooting and re-entering the chase."""
  far, _ = drive_toward((0.0, 0.0, 0.0), (5.0, 0.0), slow_radius=0.25)
  near, _ = drive_toward((0.0, 0.0, 0.0), (0.05, 0.0), slow_radius=0.25)
  assert far == V_MAX, "taper should not slow a distant target"
  assert near < far / 4, "no meaningful taper close in"
  # ...but never so slow that the wheels park in their stiction deadband
  # (frictionloss/kv = 0.1 rad/s of wheel, i.e. 0.0045 m/s of body).
  assert near / 0.045 > 0.1, "terminal speed lands in the stiction deadband"


def test_path_following_is_untouched():
  """The default law is the one every path-follower uses, and this change
  must not have altered it: sweeping THROUGH waypoints is what it is for."""
  for err_deg in (0, 30, 60, 89):
    err = math.radians(err_deg)
    v, w = drive_toward((0.0, 0.0, -err), (1.0, 0.0))
    assert v == V_MAX * math.cos(err)
    assert abs(w - min(W_MAX, 2.5 * err)) < 1e-9


# ---- the inflation-halo escape (issue #92) -----------------------------------


def _halo_world():
  """A robot sealed inside an obstacle's inflation ring, with the rest of the
  room known-free and a frontier at the far end.

  The geometry that trapped the real robot, minimised: a couch-shaped
  occupied block, the robot one cell from it -- deep inside the 7-cell
  inflation ring -- and open mapped floor beyond, fading to unknown at the
  far wall (which is what makes the far edge a frontier).
  """
  from pluggybot.mapping.occupancy_grid import OccupancyGrid

  g = OccupancyGrid(x_min=0.0, y_min=0.0, x_max=4.0, y_max=2.0,
                    resolution=0.05)
  g.grid[:, :] = -5.0                  # known free everywhere...
  g.grid[:, 70:] = 0.0                 # ...fading to unknown past x=3.5 m
  g.grid[16:24, 8:12] = 5.0            # the couch: occupied block
  pose = (0.65, 1.0, 0.0)              # one cell east of it: sealed in
  return g, pose


def test_a_sealed_in_robot_can_still_plan_to_a_frontier():
  """THE EXPLORE TRAP (issue #92), minimised and pinned.

  `astar`'s start-cell exemption is one cell deep, so a robot whose start
  AND four neighbours are inside an obstacle's inflation ring cannot take a
  first step; each failed frontier is then blacklisted forever and a few
  strikes later explore declares a two-thirds-unmapped house finished.
  Measured on the expanded home world: gave up at 61 s of a 900 s budget
  with 366 of 367 frontiers reachable and the nearest traversable cell
  15 cm away.

  Shown to fail before the fix by replacing `nearest_traversable(trav, ...)`
  with the raw robot cell in `plan()`: status comes back "no-reachable" and
  the blacklist eats a frontier per attempt.
  """
  from pluggybot.behavior.navigation import plan
  from pluggybot.mapping.frontier import traversable_mask

  g, pose = _halo_world()
  trav = traversable_mask(np.asarray(g.grid))
  rix, riy = g.world_to_cell(pose[0], pose[1])
  assert not trav[riy, rix], "the fixture robot is not actually sealed in"
  assert not any(trav[riy + dy, rix + dx]
                 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))), \
      "a traversable neighbour means astar's own exemption suffices " \
      "and this test is about nothing"

  blacklist: set = set()
  path, status = plan(g, pose, blacklist)
  assert status == "ok", f"a sealed-in robot could not plan: {status}"
  assert path, "ok with no path"
  assert blacklist == set(), \
      f"planning burned frontiers onto the blacklist: {blacklist}"


def test_the_escape_is_bounded_a_lost_robot_is_not_teleported():
  """The other half of `nearest_traversable`'s contract: a nearest cell
  beyond the bound means the robot is off its own map -- lost, not merely
  sealed -- and planning from half a metre of pretend-position would paper
  that over. The reserve probe hit exactly this: a robot placed on the
  unmapped street 'failed to dock' in a way that looked like a route
  problem and was actually a robot outside its own known world."""
  from pluggybot.mapping.astar import nearest_traversable

  trav = np.zeros((100, 100), dtype=bool)
  trav[50:60, 50:60] = True
  assert nearest_traversable(trav, (55, 55)) == (55, 55)
  assert nearest_traversable(trav, (48, 55)) == (50, 55)   # inside the bound
  assert nearest_traversable(trav, (10, 10)) is None       # genuinely lost


def test_the_real_couch_pose_that_ended_exploration_can_plan_again():
  """The trap flown, at the measured pose, on the real house.

  The synthetic fixture above is the iterate-loop guard; this is the claim
  that the fixture describes reality. The robot is placed where the real
  explore run parked and gave up -- (+4.00, +0.92), 30 cm east of the couch,
  between it and the east wall -- given one look around to map its
  surroundings, and must then be sealed in by the couch's inflation ring and
  STILL able to plan to a frontier.

  Measured before the fix, from this exact situation: explore ended at 61 s
  of a 900 s budget on "no-reachable", kitchen 0.0%% mapped, 60 frontiers
  blacklisted. After: 469 s, "no-frontiers", kitchen 86.6%%, blacklist 0.
  The full flown run is 10 minutes of wall clock, so what is pinned here is
  its first domino -- sealed, and planning anyway -- which takes seconds.
  """
  import mujoco

  from pluggybot.behavior.navigation import plan
  from pluggybot.lifecycle import world_config
  from pluggybot.mapping.frontier import traversable_mask
  from pluggybot.mission.mission import HubMission

  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  mission = HubMission(model, mujoco.MjData(model), viewer=None,
                       realtime=False, rack=cfg["rack"],
                       grid_bounds=cfg["grid_bounds"])
  try:
    mission.start_at(4.0, 0.92, math.pi)     # the measured stuck pose
    mission._spin()                          # one look: maps couch + wall
    trav = traversable_mask(np.asarray(mission.grid.grid))
    rix, riy = mission.grid.world_to_cell(4.0, 0.92)
    assert not trav[riy, rix], \
        "the couch's halo no longer seals this pose; the premise moved -- " \
        "re-measure before weakening anything"
    path, status = plan(mission.grid, mission.pose, set())
    assert status == "ok", \
        f"sealed in beside the real couch and could not plan: {status}"
    assert path
  finally:
    mission.close()
