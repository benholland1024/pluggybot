"""What the humanoid is built from (issue #507): every actuator, the
gripper, the pack and the sensors a part on sale, each number its sheet's
with the source at the constant, or `None` with its reason -- never a guess
(the catalog's rule, `humanoid/catalog.py`). The model that flies them is
`humanoid.model`; the bill that buys them is `humanoid.bill`.
"""

from dataclasses import dataclass
import math

from pluggybot.humanoid.actuator import BUS_V_NOMINAL, Motor


# ---- the drivers' reply, as the robot reads a joint ---------------------------------

@dataclass(frozen=True)
class Fields:
  """A driver's reply over CAN, as the robot is told a joint: each value a
  fixed-point field over a symmetric range (an MIT-style reply), and the
  torque read off the phase current with the current sense's noise."""
  position_bits: int
  position_range: float          # +- rad
  velocity_bits: int
  velocity_range: float          # +- rad/s
  torque_bits: int
  torque_range: float            # +- N*m
  #: The phase current's sense noise, A RMS: no maker of these publishes
  #: it; stated where it is set.
  current_noise_a: float

  @property
  def position_lsb(self) -> float:
    return 2 * self.position_range / (2 ** self.position_bits - 1)

  @property
  def velocity_lsb(self) -> float:
    return 2 * self.velocity_range / (2 ** self.velocity_bits - 1)

  @property
  def torque_lsb(self) -> float:
    return 2 * self.torque_range / (2 ** self.torque_bits - 1)


# ---- the actuators: RobStride's RS line --------------------------------------------
#
# One maker's line for every joint, so one protocol, one driver's algebra and
# one spare a size (the quadruped's arm took the legs' motor for that). Read
# 2026-10-16 off RobStride's own manuals and spec sheet (`RS_DOCS`); where
# two of its documents disagree, the weaker torque and the heavier mass.
# K-Scale's K-Bot, open since K-Scale stopped selling it (November 2025),
# uses the same line (RS04 at hip pitch and knee, RS03 at hip roll and yaw). Every rating is at 48 V; the pack's 43.2 V
# nominal scales the speeds (`Motor.at_voltage`).

#: RobStride's documents, by tag: the spec sheet (2026-10-09, the RS line's
#: pages and its summary table), and each part's own manual.
RS_DOCS = {
  "spec": "https://github.com/RobStride/Product_Information/blob/main/"
          "%E7%81%B5%E8%B6%B3%E6%97%B6%E4%BB%A3RS%E7%B3%BB%E5%88%97%E4%BA%A7%E5%93%81"
          "%E8%A7%84%E6%A0%BC%E4%BB%8B%E7%BB%8D%282026.010.09%29.pdf",
  **{part: "https://github.com/RobStride/Product_Information/blob/main/"
           f"Product%20Literature/CAN2.0/{part}/{part}User%20Manual{date}.pdf"
     for part, date in (("RS04", "260713"), ("RS03", "261013"), ("RS02", "260713"),
                        ("RS00", "260713"), ("RS05", "260713"))},
}
#: The windings' limit, C: "145 C (this is the constraint temperature; the
#: actual temperature is 180 C)" in every RS manual, but 135 C for the RS05
#: and where the RS02's over-temperature fault trips: the lower. Every
#: overload table is measured at 25 C ambient.
RS_WINDING_C, RS_RATED_AMBIENT_C = 145.0, 25.0


def _rpm(v: float) -> float:
  return v * 2 * math.pi / 60


def _r_th(rated: float, kt: float, r_phase: float, limit: float = RS_WINDING_C) -> float:
  """A winding's thermal resistance, K/W, off its maker's continuous
  rating: the torque that holds it at its limit in the rating's ambient,
  heating 1.5 R I^2 (DERIVED: no RobStride sheet publishes one)."""
  return (limit - RS_RATED_AMBIENT_C) / (1.5 * r_phase * (rated / kt) ** 2)


#: RobStride RS04 (`RS_DOCS`): peak 120 N*m (held 3 s turning at 25 C by the
#: manual's table, 10 s by the spec sheet's); continuous 40 N*m on a 345 mm
#: square heat sink, 35 on a 220 x 200 one -- the nearer a leg's brackets, so
#: the rating here -- and 28.5 stalled. No-load 200 rpm (+-10 %) at 48 V;
#: the torque-speed line through it and the peak's corner, 120 N*m at 95
#: rpm (the sheet's 48 V table), so 228.6 N*m where it meets standstill.
#: Kt 2.1 N*m per A RMS (1.485 per A of amplitude: its rated 40 N*m at
#: 27 A peak agrees); 0.16 ohm line to line, 0.08 a phase. 9:1 planetary,
#: its output-side inertia 0.04 kg*m^2 (the sheet's "low-speed equivalent
#: inertia"; the rotor's own unpublished, so it is that over 81).
#: 1.42 kg in the manual, 1.51 on the sheet's page: the heavier. 120 mm
#: across, 52 mm long. Backlash unpublished. EUR 254.75 (OpenELAB, incl.
#: VAT, pre-order, 2026-10-16); its maker lists it at CNY 1,199.
RS04 = Motor(
  name="RobStride RS04", rated_torque=35.0, peak_torque=120.0,
  noload_speed=_rpm(200.0), saturation_torque=120.0 * 200.0 / (200.0 - 95.0),
  bus_v=48.0, kt=2.1 / math.sqrt(2), r_phase=0.16 / 2,
  rotor_inertia=0.04 / 9 ** 2, ratio=9.0, mass=1.51, diameter=0.120,
  length=0.052, max_winding_c=RS_WINDING_C,
  thermal_r=_r_th(35.0, 2.1 / math.sqrt(2), 0.16 / 2), price_eur=254.75,
  source=RS_DOCS["RS04"], stall_torque=28.5)

#: RobStride RS03 (`RS_DOCS`): peak 60 N*m; continuous 20 N*m on a 215 x
#: 220 mm plate (21 on 200 x 200: the lower), 13 stalled. No-load 200 rpm
#: in its manual, 195 on the sheet (the slower); the line through it and
#: 60 N*m at 120 rpm (the sheet's 48 V table): 156 N*m at standstill. Kt
#: 2.36 N*m per A RMS; 0.39 ohm line to line. 9:1, its output-side inertia
#: 0.02 kg*m^2. 880 g in its manual, 928 on the sheet's page: the heavier.
#: 98 mm across, 54.1 mm long. EUR 215.95 (OpenELAB, incl. VAT, 2026-10-16).
RS03 = Motor(
  name="RobStride RS03", rated_torque=20.0, peak_torque=60.0,
  noload_speed=_rpm(195.0), saturation_torque=60.0 * 195.0 / (195.0 - 120.0),
  bus_v=48.0, kt=2.36 / math.sqrt(2), r_phase=0.39 / 2,
  rotor_inertia=0.02 / 9 ** 2, ratio=9.0, mass=0.928, diameter=0.098,
  length=0.0541, max_winding_c=RS_WINDING_C,
  thermal_r=_r_th(20.0, 2.36 / math.sqrt(2), 0.39 / 2), price_eur=215.95,
  source=RS_DOCS["RS03"], stall_torque=13.0)


#: RobStride RS02 (`RS_DOCS`): peak 17 N*m; continuous 6 N*m on a 260 x 280
#: mm sink in its manual, 7 on 150 x 150 on the sheet (the lower), and 6
#: stalled. No-load 410 rpm at 48 V; the line through it and 17 N*m at 219
#: rpm: 36.5 N*m at standstill. Kt 1.22 N*m per A RMS; 0.55 ohm line to
#: line. 7.75:1, its output-side inertia 4.2e-3 kg*m^2. 380 g in its manual,
#: 418 on the sheet's page. 78.5 mm across, 41.5 mm long. Its fault trips at
#: 135 C. EUR 149.95 (OpenELAB, incl. VAT, 2026-10-16).
RS02 = Motor(
  name="RobStride RS02", rated_torque=6.0, peak_torque=17.0,
  noload_speed=_rpm(410.0), saturation_torque=17.0 * 410.0 / (410.0 - 219.0),
  bus_v=48.0, kt=1.22 / math.sqrt(2), r_phase=0.55 / 2,
  rotor_inertia=4.2e-3 / 7.75 ** 2, ratio=7.75, mass=0.418, diameter=0.0785,
  length=0.0415, max_winding_c=135.0,
  thermal_r=_r_th(6.0, 1.22 / math.sqrt(2), 0.55 / 2, 135.0), price_eur=149.95,
  source=RS_DOCS["RS02"], stall_torque=6.0)

#: RobStride RS00 (`RS_DOCS`): peak 14 N*m (the sheet lists a newer 17 N*m
#: version beside it; this is the one its manual documents); continuous 5
#: N*m, 3.6 stalled. No-load 315 rpm at 48 V; the line through it and 14
#: N*m at 100 rpm: 20.5 N*m at standstill. Kt 1.48 N*m per A RMS; 1.5 ohm
#: line to line. 10:1, its output-side inertia 0.001 kg*m^2. 310 g, 57 mm
#: across, 51 mm long. EUR 129.95 (OpenELAB, incl. VAT, 2026-10-16).
RS00 = Motor(
  name="RobStride RS00", rated_torque=5.0, peak_torque=14.0,
  noload_speed=_rpm(315.0), saturation_torque=14.0 * 315.0 / (315.0 - 100.0),
  bus_v=48.0, kt=1.48 / math.sqrt(2), r_phase=1.5 / 2,
  rotor_inertia=0.001 / 10 ** 2, ratio=10.0, mass=0.310, diameter=0.057,
  length=0.051, max_winding_c=RS_WINDING_C,
  thermal_r=_r_th(5.0, 1.48 / math.sqrt(2), 1.5 / 2), price_eur=129.95,
  source=RS_DOCS["RS00"], stall_torque=3.6)

#: RobStride RS05 (`RS_DOCS`): peak 5.5 N*m; continuous 1.6 N*m on a 70 x
#: 70 mm sink in its manual (1.8 on 150 x 150: the lower); stalled
#: unpublished (1.6 N*m held 175 s), so the manual's own word -- a stalled
#: phase heats 1.414x a turning one -- sets it: 1.6 / sqrt(1.414) (DERIVED).
#: No-load 480 rpm at 48 V; the line through it and 5.5 N*m at 70 rpm:
#: 6.44 N*m at standstill. Kt 0.94 N*m per A RMS; 2.72 ohm line to line.
#: 7.75:1, its output-side inertia 7e-4 kg*m^2. 191 g in its manual, 243 on
#: the sheet's page. 46 mm across, 44 mm long. Its winding's limit 135 C.
#: EUR 106.95 (OpenELAB, incl. VAT, 2026-10-16).
RS05 = Motor(
  name="RobStride RS05", rated_torque=1.6, peak_torque=5.5,
  noload_speed=_rpm(480.0), saturation_torque=5.5 * 480.0 / (480.0 - 70.0),
  bus_v=48.0, kt=0.94 / math.sqrt(2), r_phase=2.72 / 2,
  rotor_inertia=7e-4 / 7.75 ** 2, ratio=7.75, mass=0.243, diameter=0.046,
  length=0.044, max_winding_c=135.0,
  thermal_r=_r_th(1.6, 0.94 / math.sqrt(2), 2.72 / 2, 135.0), price_eur=106.95,
  source=RS_DOCS["RS05"], stall_torque=1.6 / math.sqrt(1.414))

#: Each part's MIT reply (its manual's section 6.5; RS04's 6.1/6.5): the
#: angle in 16 bits over +-12.57 rad, the speed and the torque in 12 bits
#: over the part's own ranges. ⚠ THE RS04'S SPEED FIELD STOPS AT 15 rad/s,
#: under its 20.9 rad/s no-load speed (the RS03's at 20, under its 20.4):
#: a joint turning faster reads the field's last count. The current
#: sense's noise is unpublished: the quadruped's stated 25 mA RMS
#: (`perception.encoders.CURRENT_NOISE_A`).
FIELDS = {
  RS04.name: Fields(16, 12.57, 12, 15.0, 12, 120.0, 0.025),
  RS03.name: Fields(16, 12.57, 12, 20.0, 12, 60.0, 0.025),
  RS02.name: Fields(16, 12.57, 12, 44.0, 12, 17.0, 0.025),
  RS00.name: Fields(16, 12.57, 12, 33.0, 12, 14.0, 0.025),
  RS05.name: Fields(16, 12.57, 12, 50.0, 12, 5.5, 0.025),
}

#: How long each part holds a torque past its continuous rating before its
#: winding reaches its limit, s, from 25 C (the sheets' overload tables,
#: measured on the heat sink each rating names): turning, and stalled.
#: (N*m, s) pairs, rising torque.
OVERLOAD = {
  RS04.name: {"turning": ((50, 290), (70, 55), (90, 24), (110, 10.6), (120, 5)),
              "still": ((40, 130), (50, 74), (60, 33), (70, 10), (80, 6), (90, 3),
                        (100, 1.5), (110, 1), (120, 1))},
  RS03.name: {"turning": ((30, 393), (40, 67), (50, 34), (60, 13)),
              "still": ((20, 190), (30, 25), (40, 9), (50, 4), (60, 1))},
  RS02.name: {"turning": ((8, 840), (11, 75), (14, 20), (17, 9)),
              "still": ((7, 867), (11, 40), (15, 10), (17, 6))},
  RS00.name: {"turning": ((7, 120), (10, 18), (12, 10), (14, 5)),
              "still": ((5, 110), (9, 9), (12, 2), (14, 1.5))},
  RS05.name: {"turning": ((2, 438), (3, 42), (4, 13), (5, 5.4), (5.5, 3.7)),
              "still": ((1.6, 175), (4, 4), (5, 2), (5.5, 1))},
}


def held_for(motor: Motor, torque: float, still: bool) -> float | None:
  """How long the part holds `torque`, s, by its sheet's overload table:
  None within its continuous rating (for as long as wanted); between the
  rating and the table's first row, that row's time (it lasts longer, by
  how much the sheet does not say -- the table's bound, `lasts_at_least`);
  between rows, the two rows' times on a log scale."""
  rating = stall_torque(motor) if still else motor.rated_torque
  if torque <= rating:
    return None
  rows = OVERLOAD[motor.name]["still" if still else "turning"]
  if torque <= rows[0][0]:
    return float(rows[0][1])        # at least: `lasts_at_least`
  for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
    if torque <= t1:
      k = (torque - t0) / (t1 - t0)
      return float(math.exp(math.log(s0) + k * (math.log(s1) - math.log(s0))))
  return float(rows[-1][1])


def lasts_at_least(motor: Motor, torque: float, still: bool) -> bool:
  """Whether `held_for`'s time is only a floor: the torque is past the
  rating but under the overload table's first row."""
  rating = stall_torque(motor) if still else motor.rated_torque
  rows = OVERLOAD[motor.name]["still" if still else "turning"]
  return rating < torque < rows[0][0]


#: Gear backlash, rad: unpublished for every RS part; the quadruped's
#: 15 arcmin (both Steadywin tables, a planetary of the class) stands in.
BACKLASH_RAD = math.radians(15 / 60)


def backlash_rad(motor: Motor) -> float:
  return BACKLASH_RAD


#: A joint's Coulomb friction, as a share of its part's peak torque:
#: unpublished for every RS part. Katz measured 0.09 N*m on the Mini
#: Cheetah actuator, 17 N*m peak -- 0.5 % -- stated for the line; the
#: sheets' no-load currents (0.14-2 A RMS) times Kt bound it from above,
#: since they carry the gears' and the iron's losses at full speed too.
FRICTION_SHARE = 0.005


def friction_nm(motor: Motor) -> float:
  return FRICTION_SHARE * motor.peak_torque


def stall_torque(motor: Motor) -> float:
  """The continuous torque held still: the sheet's stalled rating."""
  return motor.stall_torque if motor.stall_torque is not None else motor.rated_torque


#: Which part drives each joint, both sides alike, and why that one
#: (SimNotes, "The humanoid body", has the tables): the RS04 wherever the
#: whole body's weight passes -- the hip's pitch and roll, the knee, the
#: ankle's pitch, which pushes off -- the RS03 for the hip's yaw, the
#: ankle's roll and the shoulder's two (an arm held out level with a load),
#: the RS02 down the arm, the RS00 at the wrist's pitch (the gripper and
#: what it holds, ahead of it) and the RS05 for the roll after it, which
#: turns only the gripper.
DRIVES = {
  "hip_pitch": RS04, "hip_roll": RS04, "hip_yaw": RS03, "knee": RS04,
  "ankle_pitch": RS04, "ankle_roll": RS03,
  "shoulder_pitch": RS03, "shoulder_roll": RS03, "shoulder_yaw": RS02,
  "elbow": RS02, "wrist_pitch": RS00, "wrist_roll": RS05,
}


# ---- the gripper --------------------------------------------------------------------

@dataclass(frozen=True)
class Gripper:
  """A parallel gripper as sold, in the sim's units: its jaws one motor's,
  slid toward each other along the hand's x."""
  name: str
  #: Total travel between the jaws, m, and the most each pad squeezes
  #: with, N (the sheet's grip force: the force on the thing held).
  stroke: float
  grip_force_n: float
  #: The jaws' fastest opening or closing, m/s, at `speed_v` volts.
  speed: float
  speed_v: float
  mass: float
  #: Its body (across the jaws' travel, across, along its axis) and its
  #: fingers' length, m.
  body: tuple[float, float, float]
  finger_len: float
  #: The opening as it reports it, m a count (the fingertip's resolution).
  position_lsb: float
  #: Its squeeze's noise as it reports it, N: unpublished, stated where set.
  force_noise_n: float
  #: The mounting face the wrist's adapter plate bolts to, m square.
  flange: float
  price_eur: float | None
  source: str

  @property
  def housing_half(self) -> tuple[float, float, float]:
    return tuple(v / 2 for v in self.body)

  @property
  def pad_half(self) -> tuple[float, float, float]:
    """A finger's pad: 12 mm thick, 30 mm wide, the finger's length."""
    return (0.006, 0.015, self.finger_len / 2)

  @property
  def finger_kg(self) -> float:
    """A finger's share of the sheet's mass, which is not split there:
    its pad's volume in aluminium at half fill (stated)."""
    x, y, z = self.pad_half
    return 8 * x * y * z * 2700 * 0.5

  @property
  def housing_kg(self) -> float:
    return self.mass - 2 * self.finger_kg

  #: The jaws' reflected inertia, kg: its motor's rotor through its gear is
  #: unpublished. Stated so the 2 ms step stays stable at the stiffness a
  #: squeeze commands (`GRIP_KP`); the sheet's closing time is the check.
  jaw_armature: float = 0.2

  def dcmotor(self, bus_v: float = BUS_V_NOMINAL) -> dict:
    """The actuated jaw's driver as a position-mode `dcmotor`, as a joint's
    is (`humanoid.actuator.JointLimits.dcmotor`): its force clipped at twice
    the grip force -- the second jaw is held to the first, so each pad
    squeezes half of what the motor pushes -- and its line falling to no
    force at the sheet's top speed, scaled to the bus (the line's
    standstill end is unpublished; the clip stands in)."""
    v0 = (self.speed / 2) * bus_v / self.speed_v
    peak = 2 * self.grip_force_n
    k = bus_v / v0
    r = k * bus_v / peak
    return {"motorconst": (k, 0.0), "resistance": r, "saturation": (peak, 0.0, 0.0),
            "controller": (GRIP_KP * r / k, 0.0, GRIP_KD * r / k - k, 0.0, 0.0, bus_v)}


#: The squeeze the gripper is commanded with, N/m and N*s/m: its stiffness
#: and damping (it takes both in each command); stiff enough that shut on
#: anything wider than 1 cm short of closed it pushes at its clip -- a
#: servo held at its stall, the claw's lesson (#407: below it, a held cube
#: crept out of the jaws).
GRIP_KP, GRIP_KD = 24000.0, 300.0

#: Unitree Dex1-1 (Standard), the parallel gripper of Unitree's G1, H2 and
#: R1 (https://unitree.com/Dex1-1): stroke 90 mm; grip force 5-120 N,
#: adjustable; 50-375 mm/s, a full stroke shut in 0.24 s at the shortest,
#: "measured under varying conditions (24V-60V)"; 550 g; 143 x 78 x 67 mm;
#: fingers 80 mm; reports angle, speed, torque and temperature, the
#: fingertip to 0.1 mm off a 15-bit absolute encoder; takes angle, speed,
#: torque, stiffness and damping at 1000 Hz over RS-485; 24-64 V, 3.5 A at
#: peak. Its flange, its rotor and its torque estimate's resolution are
#: unpublished. EUR 703.45 (OpenELAB, Garching; pre-order, 2026-10-16).
#: Chosen over the ROBOTIS RH-P12-RN(A) (106 mm, 170 N, 500 g, every
#: register published: EUR 3,912.36 a gripper, 5.6x the price) and Inspire's
#: EG2-5C, which the T2 ships (364 g, 0-20 N, its registers unpublished).
DEX1_1 = Gripper(
  name="Unitree Dex1-1", stroke=0.090, grip_force_n=120.0, speed=0.375,
  speed_v=60.0, mass=0.550, body=(0.078, 0.067, 0.063), finger_len=0.080,
  position_lsb=0.0001,
  # Unpublished: a FOC driver's current sense at rest (the quadruped's
  # stated 25 mA, `perception.encoders`) through the M4010's 5 N*m peak at
  # its 3.5 A peak line current, over its 7 mm effective pinion (its 27
  # rad/s top speed against the jaws' 0.19 m/s).
  force_noise_n=0.025 * (5.0 / 3.5) / 0.007,
  flange=0.06, price_eur=703.45, source="https://unitree.com/Dex1-1")
GRIPPER = DEX1_1


# ---- the pack -------------------------------------------------------------------------

@dataclass(frozen=True)
class Pack:
  """The pack, built here as the quadruped's was (Parts.md's bill: the
  cells, a BMS, nickel strip and a spot welder): `series` x `parallel`
  Molicel P45B, laid `rows` x `cols` on end in a printed case."""
  series: int
  parallel: int
  rows: int
  cols: int
  name: str = "12S Molicel P45B"

  #: The P45B (Molicel's sheet; the quadruped's bill before it): 21700, 4.5 Ah,
  #: 3.6 V nominal, 3.0-4.2 V, 70 g, 45 A continuous.
  CELL_AH, CELL_V, CELL_KG, CELL_D, CELL_L = 4.5, 3.6, 0.070, 0.021, 0.070

  @property
  def cells(self) -> int:
    return self.series * self.parallel

  @property
  def wh(self) -> float:
    return self.cells * self.CELL_AH * self.CELL_V

  @property
  def half(self) -> tuple[float, float, float]:
    """The case's half-extents, m: the cells on end with a 2 mm wall."""
    return (self.rows * self.CELL_D / 2 + 0.002, self.cols * self.CELL_D / 2 + 0.002,
            self.CELL_L / 2 + 0.002)

  @property
  def kg(self) -> float:
    """Cells, the BMS (the bill's JBD, 100 g), the strip (two 10 x 0.15 mm
    Hilumin links a cell, 50 mm each: 15 g/m, the bill's 150 g for 10 m)
    and the printed case: its surface at 2 mm of PETG."""
    x, y, z = (2 * v for v in self.half)
    case = 2 * (x * y + x * z + y * z) * 0.002 * 1270.0
    return self.cells * self.CELL_KG + 0.100 + self.cells * 2 * 0.05 * 0.015 + case


#: 12S like the quadruped's was (36-50.4 V; the drivers are rated at 48 V and
#: take 24-60), four in parallel: 778 Wh (`humanoid_spike.py --energy`).
PACK = Pack(series=12, parallel=4, rows=6, cols=8)


# ---- the sensors and the electronics --------------------------------------------------

def _catalog_kg(part: str) -> float:
  """A part's mass as the catalog has it (`humanoid.catalog.PARTS`), kg:
  one fact, read here rather than typed again."""
  from pluggybot.humanoid import catalog
  g = catalog.by_id()[part].massG
  if g is None:
    raise ValueError(f"{part} has no published mass")
  return g / 1000


#: The head's sensors, the catalog's parts: the D435 (75 g, its sheet of
#: March 2026), the Camera Module 3 (4 g) and the RPLIDAR C1 (110 g; its
#: puck 75.6 mm across and 41 mm tall, as the quadruped's model had it at
#: the tag `quadruped-final`).
D435_KG = _catalog_kg("realsense_d435")
CAMERA_KG = _catalog_kg("pi_camera_3")


#: The charge port on the torso's front (#559 stage 2): Rosenberger's RoPD
#: panel plug, the C003-B1 (25 A on 2.5 mm^2, 60 V DC, two power contacts and
#: four signal), by its sheet (C003-B1-XXX-Y_DB.pdf): 15 g, a 28.5 mm square
#: flange on M3 non-magnetic screws, 18.8 mm deep behind a 25 mm cut-out.
#: The flange's thickness, which stands proud of the torso, is unpublished:
#: 2 mm. ⚠ It is not hot-pluggable: power only once its signal pins are
#: mated, so the station's charger runs off them (`activity.station`).
PORT_KG = 0.015
#: Its flange, m: (proud of the torso, across).
PORT_FLANGE = (0.002, 0.0285)
#: ...and the funnel printed round it, m: its mouth across and how far its
#: rim stands proud of the connector's face. A plug pushed anywhere in its
#: mouth is guided onto the connector: the hand knows where it holds a plug
#: to 1-2 cm (its arm's encoders and backlash, and the plug free along the
#: Dex1-1's 80 mm pads: pushed to the body it landed 12-33 mm across the
#: port and 7-10 mm low), and the RoPD itself centres a few mm. 70 mm is as
#: wide as the gripper's housing, 40 mm off the plug's axis, clears. Its mass
#: is its drawing's: a 3 mm PETG cone wall, 18 g (1.27 g/cm^3).
PORT_FUNNEL = (0.070, 0.015)
PORT_FUNNEL_KG = 0.018


@dataclass(frozen=True)
class Lidar:
  radius: float
  height: float
  kg: float


LIDAR = Lidar(radius=0.0378, height=0.041, kg=_catalog_kg("rplidar_c1"))

#: What rides the torso besides the pack, by the catalog's parts and their
#: published masses: the Pi 5 with its cooler and SSD (only their
#: packed weights are published: 65, 29 and 35 g, upper bounds), the two CAN
#: HATs, the 5 V converter, and the e-stop chain's robot half -- the
#: contactor, the button, the receiver, its relay, the 12 V converter and
#: the pre-charge resistor and relay (the transmitter and its cells are in
#: a hand). The fuse, its holder, the connectors and the harness are
#: `model.ALLOWANCE`'s.
ELECTRONICS = {
  "pi5 (packed)": 0.065, "pi5 cooler (packed)": 0.029, "pi5 ssd kit (packed)": 0.035,
  "waveshare_can_hat_plus": _catalog_kg("waveshare_can_hat_plus"),
  "waveshare_can_fd_hat": _catalog_kg("waveshare_can_fd_hat"),
  "meanwell_ddr_60l_5": _catalog_kg("meanwell_ddr_60l_5"),
  "albright_sw80b549": _catalog_kg("albright_sw80b549"),
  "schneider_xb5as8442": _catalog_kg("schneider_xb5as8442"),
  "radiomaster_er6": _catalog_kg("radiomaster_er6"),
  "pololu_rc_relay": _catalog_kg("pololu_rc_relay"),
  "meanwell_ddr_30l_12": _catalog_kg("meanwell_ddr_30l_12"),
  "arcol_hs25_47r": _catalog_kg("arcol_hs25_47r"),
  "hongfa_hf115f_012": _catalog_kg("hongfa_hf115f_012"),
}
ELECTRONICS_KG = sum(ELECTRONICS.values())

#: What is powered whatever the joints do, W.
ELECTRONICS_W = {
  # The quadruped's figures (`legs.model.ELECTRONICS_W` at `quadruped-final`): the Pi 5, its
  # camera and the IMU (Raspberry Pi publishes no load figure), the
  # RPLIDAR C1's 230 mA at 5 V, the D435 streaming depth.
  "compute + cameras": 6.0,
  "lidar": 1.15,
  "depth camera": 2.0,
  # Twenty-four drivers enabled: the RS03's sheet's "standby current <= 40
  # mA" (the only RS part that publishes one) at the pack's 43.2 V -- an
  # upper bound, carried as the figure.
  "drivers": 24 * 0.040 * BUS_V_NOMINAL,
  # The two grippers' "static current 0.03 A" (Dex1-1's page).
  "grippers": 2 * 0.03 * BUS_V_NOMINAL,
  # The e-stop's contactor held closed: the SW80's coil, "7-13 W
  # (continuous)" by Albright's sheet -- the dearer -- through its 12 V
  # converter (89 % typical, the DDR-30L-12's sheet); and the chain's RC
  # relay, about 100 mA at 5 V with the relay on (Pololu's page).
  "contactor": 13.0 / 0.89,
  "e-stop relay": 0.1 * 5.0,
  # The BMS running: "<= 10 mA" (JBD's sheet).
  "bms": 0.010 * BUS_V_NOMINAL,
}
