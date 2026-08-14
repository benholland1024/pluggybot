"""Differential-drive command helpers shared by teleop and autonomous driving."""

import math

WHEEL_RADIUS = 0.045    # m — Pololu 90x10 wheel (see docs/Parts.md)
TRACK_WIDTH = 0.21      # m — distance between wheel centers
MAX_WHEEL_ACCEL = 30.0  # rad/s^2 — slew limit, like a real motor controller's ramp


def wheel_targets(vel: float, ang_vel: float) -> tuple[float, float]:
  """Body command (v m/s, w rad/s) -> (left, right) wheel angular velocities."""
  left = (vel - ang_vel * TRACK_WIDTH / 2) / WHEEL_RADIUS
  right = (vel + ang_vel * TRACK_WIDTH / 2) / WHEEL_RADIUS
  return left, right


def slew(current: float, target: float, dt: float, max_accel: float = MAX_WHEEL_ACCEL) -> float:
  """Move current toward target, changing by at most max_accel * dt."""
  step = max_accel * dt
  return current + max(-step, min(step, target - current))


def wrap_angle(a: float) -> float:
  """Wrap an angle to (-pi, pi]."""
  return math.atan2(math.sin(a), math.cos(a))


W_BREAKAWAY = 0.08      # rad/s of body yaw -- the smallest in-place turn
                        # command that reliably breaks the wheels' static
                        # friction. The wheel joints carry frictionloss=0.05
                        # (the gearbox parking brake, issue #3) and the
                        # velocity servo's torque is kv*(target-actual), so a
                        # wheel target under frictionloss/kv = 0.1 rad/s
                        # cannot move a stopped wheel at all: a P-turn
                        # controller that shrinks its command with the error
                        # parks itself in the stiction deadband and creeps.
                        # Measured: the claw's _face took 55 s to settle
                        # (9.5 s before the brake). 0.1 rad/s of wheel is
                        # ~0.05 rad/s of yaw; this floors above it. Real
                        # motor controllers do the same thing (deadband
                        # compensation): commanding less than breakaway is
                        # indistinguishable from commanding zero, so there is
                        # nothing to lose by rounding up.


def turn_command(err: float, gain: float = 1.2, limit: float = 0.5) -> float:
  """P-controller turn command with a stiction breakaway floor."""
  w = max(-limit, min(limit, gain * err))
  if w != 0.0 and abs(w) < W_BREAKAWAY:
    w = math.copysign(W_BREAKAWAY, w)
  return w
