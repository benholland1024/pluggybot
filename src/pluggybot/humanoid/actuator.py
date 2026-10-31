"""The honest actuator (issue #377 item 2): each joint is the real part.

A joint is a brushless motor and its planetary, driven by an FOC driver that
runs a PD loop on a position target. What the sim models, and where each
number comes from (each part's own, `humanoid/parts.py`):

  the torque-speed ENVELOPE   the maker's curve: a line from the saturation
                              torque at standstill to zero at no-load speed,
                              clipped at the driver's peak (`envelope`),
                              scaled with the pack's voltage
  reflected rotor INERTIA     `armature` = J_rotor x N^2
  the driver's PD loop        at the physics rate, on the target the policy
                              last sent
  HEAT                        the maker's Kt and resistance

Every constant names its source. #379's bench leg replaces them once the
order gate passes.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Motor:
  """One actuator as sold, numbers at its OUTPUT shaft unless named."""

  name: str
  #: Continuous torque (the thermal rating) and peak torque, N*m.
  rated_torque: float
  peak_torque: float
  #: The torque-speed line at `bus_v`: zero torque at `noload_speed` (rad/s),
  #: `saturation_torque` where it meets zero speed (N*m), before the
  #: driver's current clip at `peak_torque`.
  noload_speed: float
  saturation_torque: float
  bus_v: float
  #: Output-side torque constant, N*m/A, and phase resistance, ohm: the
  #: pair whose 1.5*R*I^2 fits under the maker's own efficiency curve.
  kt: float
  r_phase: float
  #: Rotor inertia on the MOTOR side, kg*m^2: `rotor_range` where no maker
  #: publishes it, the nominal inside it.
  rotor_inertia: float
  ratio: float
  mass: float
  #: Outer diameter and axial length, m.
  diameter: float
  length: float
  rotor_range: tuple[float, float] | None = None
  #: Winding limit (C), and a first-order thermal model: resistance winding
  #: -> ambient (K/W) and heat capacity (J/K). None where unpublished.
  max_winding_c: float | None = None
  thermal_r: float | None = None
  thermal_c: float | None = None
  price_eur: float | None = None
  source: str = ""
  #: The continuous torque held STILL, N*m, where the maker publishes one:
  #: stalled, one phase carries the current (RobStride: 1.414x a turning
  #: phase's heat), so a pose held is rated below a joint that turns.
  stall_torque: float | None = None

  @property
  def armature(self) -> float:
    return self.rotor_inertia * self.ratio ** 2

  def at_voltage(self, volts: float) -> "Motor":
    """The same motor on another bus: speed and the voltage line scale with
    it, the current clip (peak) does not."""
    k = volts / self.bus_v
    return replace(self, bus_v=volts, noload_speed=self.noload_speed * k,
                   saturation_torque=self.saturation_torque * k)


def envelope(qd: np.ndarray, sat: np.ndarray, peak: np.ndarray,
             noload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The torque a joint can make at speed `qd`: (lo, hi).

  The DC-motor line through (0, sat) and (noload, 0), clipped at the peak --
  the formulation legged-robot simulators use (Isaac Lab's `DCMotor`). A
  joint driven BACKWARDS (braking) keeps the peak."""
  hi = np.clip(sat * (1.0 - qd / noload), 0.0, peak)
  lo = np.clip(sat * (-1.0 - qd / noload), -peak, 0.0)
  return lo, hi


#: The pack's voltage across a discharge (12S Li-ion, `parts.PACK`): 3.0 to
#: 4.2 V a cell. The tables fly the nominal 3.6 V a cell, where a motor is
#: 10 % slower than its 48 V curve.
BUS_V_RANGE = (36.0, 50.4)
BUS_V_NOMINAL = 43.2

#: The driver's PD loop in MIT mode, set from the reflected inertia: a 10 Hz
#: loop at damping ratio 2, mjlab's Go1 recipe. Every policy is trained on
#: these gains (`training/` reads them from `models/humanoid.json`) and the
#: served model's drivers run them.
DRIVER_HZ, DRIVER_ZETA = 10.0, 2.0


def driver_gains(motor: Motor) -> tuple[float, float]:
  """(stiffness N*m/rad, damping N*m*s/rad), one pair for every joint."""
  w = 2 * np.pi * DRIVER_HZ
  return motor.armature * w ** 2, 2 * DRIVER_ZETA * motor.armature * w


@dataclass(frozen=True)
class JointLimits:
  """One number per joint, in the body's order, at the JOINT."""

  peak: np.ndarray
  rated: np.ndarray
  saturation: np.ndarray
  noload: np.ndarray
  #: Joint-side torque constant, N*m/A, and phase resistance.
  kt: np.ndarray
  r_phase: np.ndarray
  #: The pack's voltage these limits are at.
  bus_v: float = BUS_V_NOMINAL

  @classmethod
  def of_motors(cls, motors, bus_v: float | None = BUS_V_NOMINAL) -> "JointLimits":
    """One motor a joint, each driving its joint directly, in the order
    given, each its own part (#507)."""
    if bus_v is not None:
      motors = [m.at_voltage(bus_v) for m in motors]
    volts = {m.bus_v for m in motors}
    if len(volts) != 1:
      raise ValueError(f"one bus a body, not {sorted(volts)}")

    def col(attr):
      return np.array([getattr(m, attr) for m in motors], dtype=float)
    return cls(peak=col("peak_torque"), rated=col("rated_torque"),
               saturation=col("saturation_torque"), noload=col("noload_speed"),
               kt=col("kt"), r_phase=col("r_phase"), bus_v=volts.pop())

  def clip(self, tau: np.ndarray, qd: np.ndarray) -> np.ndarray:
    lo, hi = envelope(qd, self.saturation, self.peak, self.noload)
    return np.clip(tau, lo, hi)

  def dcmotor(self, stiffness, damping) -> list[dict]:
    """Each joint's driver as MuJoCo's own `dcmotor` in position mode
    (issue #385), so the PD and the envelope run in C on every physics step.

    The motor is the envelope's line: K = bus / no-load speed and R = K *
    bus / saturation torque (an equivalent DC pair, NOT the windings' Kt
    and R, which `copper_w` keeps). The dcmotor's controller makes a
    VOLTAGE, v = kp (target - q) - kd qd, clamped at the bus, and the motor
    turns it into K (v - K qd) / R, clamped at the peak. So the driver's
    torque PD is kp = Kp R / K and kd = Kd R / K - K: negative, because
    the FOC driver cancels the back-EMF the voltage model would add as
    damping. Inside the no-load speed the force equals `clip` of the PD to
    1e-14 N*m; `implicitfast` then integrates its damping implicitly,
    which is what makes a flight differ (SimNotes, "The served sim's
    speed"). One gain pair for every joint, or a pair a joint."""
    n = len(self.peak)
    kp, kd = np.broadcast_to(stiffness, n), np.broadcast_to(damping, n)
    rows = []
    for j in range(n):
      k = self.bus_v / self.noload[j]
      r = k * self.bus_v / self.saturation[j]
      rows.append({"motorconst": (k, 0.0), "resistance": r,
                   "saturation": (self.peak[j], 0.0, 0.0),
                   "controller": (float(kp[j]) * r / k, 0.0,
                                  float(kd[j]) * r / k - k, 0.0, 0.0, self.bus_v)})
    return rows

  def copper_w(self, tau: np.ndarray) -> np.ndarray:
    """Heat in each winding at joint torque `tau`, W: 1.5 * R * I^2 with
    I = tau / Kt the phase-current amplitude (amplitude-invariant Park)."""
    i = tau / self.kt
    return 1.5 * self.r_phase * i * i
