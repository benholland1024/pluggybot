"""Guards for the shared driving law (behavior/navigation.py).

These integrate a kinematic unicycle rather than stepping MuJoCo, deliberately:
the failure being pinned here is a property of the CONTROL LAW, not of the
plant, so it reproduces without contacts, slew or wheel friction in the way --
and it runs in milliseconds instead of a minute.
"""

import math

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
