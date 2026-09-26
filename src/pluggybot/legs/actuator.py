"""The honest actuator (issue #377 item 2): each leg joint is the real part.

A joint is a brushless motor and its planetary (and, for the knee, a belt
after it), driven by an FOC driver that runs a PD loop on a position target.
What the sim models, and where each number comes from:

  the torque-speed ENVELOPE   the maker's curve: a line from the saturation
                              torque at standstill to zero at no-load speed,
                              clipped at the driver's peak (`envelope`),
                              scaled with the pack's voltage
  reflected rotor INERTIA     `armature` = J_rotor x N^2; J_rotor is NOT
                              published for the GIM8108, so a range
                              (`Motor.rotor_range`)
  FRICTION                    NOT published: Coulomb `frictionloss`,
                              randomised in training across `FRICTION_NM`
  the driver's PD loop        at the physics rate, on the target the policy
                              last sent
  command LATENCY             NOT published: randomised across `LATENCY_S`
  HEAT                        the maker's Kt and resistance; the thermal
                              resistance and mass are a published
                              measurement of the nearest comparable

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


#: The leg actuator (#377): Steadywin GIM8108-8 with its GDS68 driver, from
#: the maker's two selection tables (GDS and GDZ .xlsx, linked from
#: steadywin-motor.com), which DISAGREE; where they do, the lower figure.
#:   rated 6.71 N*m (GDZ; GDS says 7.5), peak 22 (GDS; GDZ 23.09).
#:   The torque-speed line is the GDZ sheet's 48 V curve, ~300 - 5.9*tau rpm:
#:   300 rpm = 31.4 rad/s at no load (KV 6.04 rpm/V x 48 V = 290, within 4 %)
#:   and 50.8 N*m at its zero-speed intercept -- Kt*V/(1.5 R) = 52.9 agrees.
#:   Kt 1.19 N*m/A with R 0.72 ohm (both GDZ): 1.5*R*I^2 stays under that
#:   curve's total loss at every point (76 of 96 W at 10 N*m, 369 of 512 W
#:   at 22); the GDS pair (1.0, 0.67) over-reads its own curve by half.
#:   Rotor inertia: NOT PUBLISHED (the one transcribed 4.55e-6 is 10-25x
#:   under every comparable). The range is the same class's: AK70-10 4.1e-5
#:   (MEVIUS model) .. AK80-9 1.1e-4 (CubeMars, maker); nominal the Mini
#:   Cheetah's 7.2e-5 (Katz 2018: 21 pole pairs, the GIM8108's count).
#:   396 g with the driver; OD 97 mm off the 2D drawing (the tables' 92 is
#:   the bolt circle), 55 mm long with the driver.
#:   Thermal: the GDS68's motor-temperature alarm, 90 C (adjustable); no
#:   Steadywin part publishes a thermal resistance, so the Mini Cheetah
#:   actuator's MEASURED 1.23 K/W and 32 J/K without airflow (Katz 2018).
#:   EUR 124.95 with the GDS68 (OpenELAB, Munich); $129.20 at Steadywin.
GIM8108_8 = Motor(
  name="Steadywin GIM8108-8 + GDS68", rated_torque=6.71, peak_torque=22.0,
  noload_speed=31.4, saturation_torque=50.8, bus_v=48.0, kt=1.19,
  r_phase=0.72, rotor_inertia=7.2e-5, rotor_range=(4.1e-5, 1.1e-4),
  ratio=8.0, mass=0.396, diameter=0.097, length=0.055, max_winding_c=90.0,
  thermal_r=1.23, thermal_c=32.0, price_eur=124.95,
  source="Steadywin selection tables (GDS, GDZ), 2D drawing, GDS68 manual")

#: Stanford Pupper v3's actuator, for the small body the table rejects:
#: Steadywin GIM4305-10 (maker tables; the lower figure where they differ),
#: rated 1.0 / peak 3.47 N*m at 24 V; the GDZ 24 V curve runs 408 rpm at
#: no load to 159 rpm at 3.82 N*m (42.7 rad/s, a 6.2 N*m intercept); Kt 0.39,
#: R 1.11 (GDZ). Rotor 1.6e-5 kg*m^2 is Pupper's own model (armature
#: 0.0016 at 10:1, cs123 pupper-mjlab). 150 g, 53 x 32 mm. $127.80.
GIM4305_10 = Motor(
  name="Steadywin GIM4305-10 + GDS34", rated_torque=1.0, peak_torque=3.47,
  noload_speed=42.7, saturation_torque=6.2, bus_v=24.0, kt=0.39,
  r_phase=1.11, rotor_inertia=1.6e-5, ratio=10.0, mass=0.150,
  diameter=0.053, length=0.032, source="Steadywin selection tables; "
  "Pupper v3 tech specs; cs123 pupper-mjlab model")

#: The pack's voltage across a discharge (12S Li-ion, #377's pack): 3.0 to
#: 4.2 V a cell. Training randomises the bus across it; the tables fly the
#: nominal 3.6 V a cell, where the motor is 10 % slower than its 48 V curve.
BUS_V_RANGE = (36.0, 50.4)
BUS_V_NOMINAL = 43.2

#: What no datasheet gives, randomised in training across these ranges and
#: flown at `*_NOMINAL` in the tables (Parts.md, "The quadruped body"):
#:   Coulomb friction at the joint, N*m: Katz measured 0.09 + 4 % of load on
#:   the Mini Cheetah actuator; the Go1 and Berkeley Humanoid models carry
#:   0.3-1.0, which fold other losses in.
FRICTION_NM = (0.05, 0.6)
FRICTION_NOMINAL = 0.1
#:   Command latency, host -> driver -> shaft, s: 0.2 ms of CAN wire time an
#:   exchange (Katz) up to Pupper v3's modelled 15.4 ms end to end.
LATENCY_S = (0.0, 0.02)
#:   Backlash, rad: 15 arcmin in both maker tables (Katz measured 0.28 deg).
#:   Not in the dynamics; an encoder that reads the motor side sees the
#:   output only to within it, so it is observation noise to the policy.
BACKLASH_RAD = 0.0044


@dataclass(frozen=True)
class JointLimits:
  """One number per leg joint (twelve, in `model.JOINT_NAMES` order), at
  the JOINT: the knee's belt multiplies torque and divides speed."""

  peak: np.ndarray
  rated: np.ndarray
  saturation: np.ndarray
  noload: np.ndarray
  #: Joint-side torque constant, N*m/A, and phase resistance.
  kt: np.ndarray
  r_phase: np.ndarray

  @classmethod
  def of(cls, motor: Motor, knee_ratio: float = 1.0,
         bus_v: float | None = BUS_V_NOMINAL) -> "JointLimits":
    if bus_v is not None:
      motor = motor.at_voltage(bus_v)
    g = np.array([1.0, 1.0, knee_ratio] * 4)
    return cls(peak=motor.peak_torque * g, rated=motor.rated_torque * g,
               saturation=motor.saturation_torque * g,
               noload=motor.noload_speed / g, kt=motor.kt * g,
               r_phase=np.full(12, motor.r_phase))

  def clip(self, tau: np.ndarray, qd: np.ndarray) -> np.ndarray:
    lo, hi = envelope(qd, self.saturation, self.peak, self.noload)
    return np.clip(tau, lo, hi)

  def copper_w(self, tau: np.ndarray) -> np.ndarray:
    """Heat in each winding at joint torque `tau`, W: 1.5 * R * I^2 with
    I = tau / Kt the phase-current amplitude (amplitude-invariant Park)."""
    i = tau / self.kt
    return 1.5 * self.r_phase * i * i
