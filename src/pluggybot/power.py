"""Battery model (milestone 7): honest power, adjustable capacity.

The POWER side is anchored to the real parts (docs/Parts.md):
  - Drive motors: Pololu 37D 50:1 at 12 V — 5.5 A stall at 2.06 N·m, 0.2 A
    no-load. A brushed DC motor's current is proportional to torque, so
    I = I_noload·(speed) + I_stall·|τ|/τ_stall, and the sim knows each
    wheel's applied torque (actuator_force) and speed every step.
  - Electronics: Pi 5 + cameras + IMU + LIDAR, a steady ~8.5 W (was ~6 W
    before the hub robot traded its stereo pair for a scanning LIDAR — the
    unit costs ~2.5 W continuously, which is the real price of the swap).
  - Lift/arm: igus lead-screw steppers draw only while moving (the dryspin
    screw holds position unpowered — Parts.md), ~5 W each in motion.
  - Charging: ~1C on the 5 Ah 3S pack ≈ 55 W into the battery. The battery
    does not know or care WHAT it is plugged into — the charge signal is
    the electrical contact criterion (docking/contact.py), which is exactly
    the abstraction milestone 8's hub will reuse.

The CAPACITY side is deliberately a knob: the real ~55 Wh pack would take
hours of sim time to drain, so the default is a scaled "demo cell" that runs
flat in minutes. Scale capacity, never the physics — the same lesson as the
schuko chamfer (tune the world honestly or not at all): power draw numbers
stay honest, and `--battery-wh 55.5` runs the real pack.
"""

import numpy as np

NOMINAL_V = 11.1        # 3S LiPo nominal
STALL_A = 5.5           # per drive motor, at
STALL_TORQUE = 2.06     # N·m (Pololu #4753)
NOLOAD_A = 0.2          # per drive motor, spinning free
NOLOAD_SPEED = 21.0     # rad/s: no-load current scales up to full speed
ELECTRONICS_W = 8.5     # Pi 5 + cameras + IMU + LIDAR, always on. Was 6.0 for
                        # the stereo era; the RPLIDAR C1-class unit adds ~2.5 W,
                        # a 40 % increase that comes straight off run time.
ACTUATOR_W = 5.0        # each lead-screw stepper, only while moving
ACTUATOR_MOVING = 2e-3  # m/s: slower than this counts as holding (unpowered)
CHARGE_W = 55.0         # ~1C into the 5 Ah pack
MODULE_IDLE_W = 0.6     # a coupled tool module's own electronics: one
                        # ESP32-class board per module (Parts.md), awake only
                        # while the coupling conducts. Power-only coupling
                        # with wireless data means the module is a load the
                        # moment it is picked up, and dead the moment it is
                        # hung back up -- so the errand has an energy cost the
                        # milestone-7 battery model never had to carry.

DEMO_CAPACITY_WH = 1.0  # scaled demo cell (see module docstring)

#: The test suite's charge-rate knob (issue #84). See `Battery.charge_scale`.
#:
#: ⚠ THIS IS A WORLD PARAMETER, NOT A HARDWARE-ACCURACY EXEMPTION, and the
#: distinction is worth stating because it looks like one. The rule this repo
#: holds is that MEASUREMENTS ARE HONEST AND CAPABILITIES ARE NEVER FAKED.
#: Charge duration is neither: it is in the same class as pack capacity, which
#: is already a knob with two named values (`demo`, `hosting`) plus a free
#: override. Nothing the robot can DO changes -- it still has to find the
#: dock, align on the bay's own tag, seat the pins and hold a press until the
#: ELECTRICAL criterion says it is charging. What changes is how long the
#: waiting takes, and the waiting is what lands on the suite's clock.
#:
#: ⚠ THE DEPLOYED WORLD RUNS AT 1.0, and `tests/test_battery.py` asserts it.
#: Charging is a scored task and `economy/metabolism.json` was calibrated
#: against measured throughput (102 points/sim-hour at the honest rate), so a
#: faster charge means more cycles an hour, more points, and a metabolism
#: tuned against a world that does not exist.
CHARGE_SCALE_ENV = "PLUGGY_CHARGE_SCALE"


def charge_scale_from_env(default: float = 1.0) -> float:
  """`$PLUGGY_CHARGE_SCALE`, or `default`. An unreadable value is the
  default and says so -- failing safe here means failing HONEST, because a
  typo that silently sped the world up would be discovered as a metabolism
  that no longer matches its own measurement."""
  import os
  raw = os.environ.get(CHARGE_SCALE_ENV)
  if not raw:
    return default
  try:
    value = float(raw)
  except ValueError:
    print(f"{CHARGE_SCALE_ENV}={raw!r} is not a number; using {default}")
    return default
  if value <= 0.0:
    print(f"{CHARGE_SCALE_ENV}={raw!r} must be > 0; using {default}")
    return default
  return value


class Battery:
  """Tracks stored energy against the robot's actual actuator effort."""

  def __init__(self, model, capacity_wh: float = DEMO_CAPACITY_WH,
               fraction: float = 1.0, charge_scale: float = 1.0) -> None:
    if charge_scale <= 0.0:
      raise ValueError(f"charge_scale must be > 0, got {charge_scale}")
    self.capacity_wh = capacity_wh
    self.energy_wh = capacity_wh * fraction
    #: How many times faster than the hardware the pack fills (issue #84).
    #: 1.0 everywhere except the test suite; see `update` for the semantics
    #: and `docs/PluggyPlan.md` for why this is a world parameter and not a
    #: capability. THE DEPLOYED WORLD RUNS AT 1.0 and a test asserts it: a
    #: faster charge means more cycles an hour, more points, and a
    #: metabolism calibrated against a throughput that is not the real one.
    self.charge_scale = float(charge_scale)
    self._wheel_acts = [model.actuator("left_motor").id,
                        model.actuator("right_motor").id]
    self._wheel_dofs = [model.joint("left_wheel_joint").dofadr[0],
                        model.joint("right_wheel_joint").dofadr[0]]
    self._screw_dofs = [model.joint("lift_joint").dofadr[0],
                        model.joint("arm_joint").dofadr[0]]
    self.last_power_w = 0.0

  def power_draw(self, data) -> float:
    """Instantaneous electrical load in watts (excluding charging)."""
    p = ELECTRONICS_W
    for act, dof in zip(self._wheel_acts, self._wheel_dofs):
      tau = abs(float(data.actuator_force[act]))
      speed = min(abs(float(data.qvel[dof])) / NOLOAD_SPEED, 1.0)
      p += NOMINAL_V * (NOLOAD_A * speed + STALL_A * min(tau / STALL_TORQUE, 1.0))
    for dof in self._screw_dofs:
      if abs(float(data.qvel[dof])) > ACTUATOR_MOVING:
        p += ACTUATOR_W
    return p

  def update(self, data, dt: float, charging: bool = False,
             tool_w: float = 0.0) -> None:
    """tool_w is whatever a coupled module is drawing this step -- gated on
    the ELECTRICAL criterion, not on "are we carrying something", so a tool
    that is held but not conducting correctly costs nothing and a properly
    seated one does."""
    p = self.power_draw(data) + tool_w
    if charging:
      # ⚠ THE SCALE MULTIPLIES THE NET, not `CHARGE_W`, and the difference is
      # the whole usability of the knob. Charging is `CHARGE_W` in less
      # whatever the held press is drawing -- 55 W against a measured ~35 W
      # of stalled wheels, so ~19 W net. Scaling `CHARGE_W` by 2 would give
      # (110 - 35) = 75 W net, nearly FOUR times the fill rate, and nobody
      # could reason about the suite's clock from the number they typed.
      # Scaling the net means `charge_scale=5` fills the pack five times
      # faster, exactly.
      #
      # At 1.0 this is arithmetically identical to the `p -= CHARGE_W` it
      # replaces, which is what lets every existing mission, recording and
      # measurement stand unchanged.
      p = (p - CHARGE_W) * self.charge_scale
    self.last_power_w = p
    self.energy_wh = float(np.clip(self.energy_wh - p * dt / 3600.0,
                                   0.0, self.capacity_wh))

  @property
  def fraction(self) -> float:
    return self.energy_wh / self.capacity_wh

  @property
  def empty(self) -> bool:
    return self.energy_wh <= 0.0
