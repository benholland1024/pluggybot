"""Battery model (milestone 7): honest power, adjustable capacity.

The POWER side is anchored to the real parts (docs/Parts.md):
  - Drive motors: Pololu 37D 50:1 at 12 V — 5.5 A stall at 2.06 N·m, 0.2 A
    no-load. A brushed DC motor's current is proportional to torque, so
    I = I_noload·(speed) + I_stall·|τ|/τ_stall, and the sim knows each
    wheel's applied torque (actuator_force) and speed every step.
  - Electronics: Pi 5 + cameras + IMU, a steady ~6 W.
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
ELECTRONICS_W = 6.0     # Pi 5 + cameras + IMU, always on
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


class Battery:
  """Tracks stored energy against the robot's actual actuator effort."""

  def __init__(self, model, capacity_wh: float = DEMO_CAPACITY_WH,
               fraction: float = 1.0) -> None:
    self.capacity_wh = capacity_wh
    self.energy_wh = capacity_wh * fraction
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
      p -= CHARGE_W
    self.last_power_w = p
    self.energy_wh = float(np.clip(self.energy_wh - p * dt / 3600.0,
                                   0.0, self.capacity_wh))

  @property
  def fraction(self) -> float:
    return self.energy_wh / self.capacity_wh

  @property
  def empty(self) -> bool:
    return self.energy_wh <= 0.0
