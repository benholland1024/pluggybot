"""The parts catalog as data (issue #185): `protocol/parts.json`.

`docs/Parts.md` names the purchasable part behind each sim constant; this
module is that list as data, in two shelves:

- **body** -- what Pluggy and its rack are made of. Fixed. Each entry names
  the part, what it costs, what it weighs, and which sim constant it FEEDS.
- **catalog** -- what the agent may build a tool from (issue #168). The
  body's parts where they make sense on a module, parts that exist only for
  building, and ONE scaffold primitive: a printed PLA box at a density and
  within a print bed.

  ⚠ EVERY `feeds` VALUE IS READ OFF THE SIM AT BUILD TIME, never typed here.
  `build()` opens `models/room_hub.xml` through `MjSpec` and imports the
  constants; a fixture that carried a hand-copied number would drift from
  the model silently and then be worse than nothing, because it looks
  authoritative. So a literal that moves fails `tests/test_catalog.py`'s
  stale check, and a part whose datasheet number IS the sim's number pins it
  with `expect` -- the igus actuator's 50 N is the lift's `forcerange`, and
  either side moving fails the suite rather than the website.

  ⚠ A NUMBER THE DOC DOES NOT KNOW IS `null` WITH A `why`, NEVER A GUESS.
  Same rule as the overseer's three money states: "unknown" and a number
  are different facts. `partNumber` is what you order by -- a vendor's
  catalog number where there is one, the manufacturer's own designation
  otherwise (`RPLIDAR C1`, `Camera Module 3`); a series or a class of part
  ("any roller-lever micro") is not a part number and stays null.

  `status` is the design's word on the part: `chosen` is in the build,
  `candidate` was considered and is not (or not yet) in it, `removed` was in
  the design and taken out. All three are shown, because a catalog that only
  listed what won would hide the decisions.

Regenerate: `uv run python -m pluggybot.rack.catalog`. Vendored to the
website beside `hints.json`; the parts page is its first consumer.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCHEMA = 1

KINDS = ("motor", "actuator", "sensor", "structure", "power", "electronics",
         "fastener", "scaffold")
STATUSES = ("chosen", "candidate", "removed")
SHELVES = ("body", "catalog")

#: Fields a part may not know. Each null needs its reason in `why`, keyed by
#: the field, and `why` may name nothing that is not null (a stale excuse).
NULLABLE = ("partNumber", "source", "massG", "dimensionsMm", "priceEur")

#: The compiled world the model-side feeds are read from: the fork robot,
#: the rack and all five modules in one file.
WORLD = "models/room_hub.xml"


# ---- readers: a feed's value comes from HERE, never from a literal ---------

Reader = Callable[[Any], Any]


def _num(v: Any) -> Any:
  if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
    return [_num(x) for x in v]
  f = float(v)
  return int(f) if f.is_integer() else round(f, 6)


def const(path: str) -> Reader:
  """A Python constant by dotted path under `pluggybot`."""
  mod, _, name = path.rpartition(".")

  def read(_spec):
    return _num(getattr(importlib.import_module(f"pluggybot.{mod}"), name))
  return read


def geom(name: str, attr: str, index: int | None = None) -> Reader:
  def read(spec):
    g = next(g for g in spec.geoms if g.name == name)
    v = getattr(g, attr)
    return _num(v if index is None else v[index])
  return read


def joint(name: str, attr: str, index: int | None = None) -> Reader:
  def read(spec):
    j = next(j for j in spec.joints if j.name == name)
    v = getattr(j, attr)
    return _num(v if index is None else v[index])
  return read


def actuator(name: str, attr: str, index: int = 1) -> Reader:
  """`forcerange` / `ctrlrange` are pairs; index 1 is the positive limit."""
  def read(spec):
    a = next(a for a in spec.actuators if a.name == name)
    return _num(getattr(a, attr)[index])
  return read


def gain(name: str) -> Reader:
  def read(spec):
    a = next(a for a in spec.actuators if a.name == name)
    return _num(a.gainprm[0])
  return read


def camera(name: str) -> Reader:
  def read(spec):
    return _num(next(c for c in spec.cameras if c.name == name).fovy)
  return read


def body_pos(name: str, axis: int) -> Reader:
  def read(spec):
    return _num(next(b for b in spec.bodies if b.name == name).pos[axis])
  return read


def same(*readers: Reader) -> Reader:
  """Two elements the design keeps symmetric (the two wheels, the two
  motors): one number, and a refusal if they have come apart."""
  def read(spec):
    vals = [r(spec) for r in readers]
    if any(v != vals[0] for v in vals):
      raise ValueError(f"asymmetric: {vals}")
    return vals[0]
  return read


def neg(reader: Reader) -> Reader:
  return lambda spec: _num(-reader(spec))


def total(*readers: Reader) -> Reader:
  def read(spec):
    return _num(sum(r(spec) for r in readers))
  return read


@dataclass
class Feed:
  """One sim constant this part sets. `constant` is its name as a reader
  would look it up (a dotted path under `pluggybot`, or `<element>.<attr>`
  in the model); `where` says which. `expect` is the part's own number in
  the same unit, where the datasheet and the sim are meant to agree."""
  constant: str
  read: Reader
  unit: str
  note: str = ""
  expect: Any = None
  where: str = "model"

  def as_dict(self, spec) -> dict:
    out = {"constant": self.constant, "where": self.where,
           "value": self.read(spec), "unit": self.unit, "note": self.note}
    if self.expect is not None:
      out["expect"] = _num(self.expect)
    return out


def code(constant: str, unit: str, note: str = "", expect: Any = None) -> Feed:
  return Feed(constant, const(constant), unit, note, expect, where="code")


@dataclass
class Part:
  id: str
  name: str
  kind: str
  status: str
  shelves: tuple[str, ...]
  usedBy: tuple[str, ...]
  partNumber: str | None = None
  source: str | None = None
  massG: float | None = None
  dimensionsMm: dict | None = None
  priceEur: float | None = None
  priceFor: str = "each"
  quantity: int = 0
  capabilities: dict = field(default_factory=dict)
  feeds: tuple[Feed, ...] = ()
  why: dict = field(default_factory=dict)
  note: str = ""

  def as_dict(self, spec) -> dict:
    return {
      "id": self.id, "name": self.name, "kind": self.kind,
      "status": self.status, "shelves": list(self.shelves),
      "usedBy": list(self.usedBy), "partNumber": self.partNumber,
      "source": self.source, "massG": self.massG,
      "dimensionsMm": self.dimensionsMm, "priceEur": self.priceEur,
      "priceFor": self.priceFor, "quantity": self.quantity,
      "capabilities": self.capabilities,
      "feeds": [f.as_dict(spec) for f in self.feeds],
      "why": self.why, "note": self.note,
    }


ECKSTEIN = "https://eckstein-shop.de/"
TBD = "Parts.md: TBD (verify on the datasheet)"
PRINTED = "3D-printed, not bought"
STOCK = "stock material, not a catalog part"

PARTS: tuple[Part, ...] = (
  # ---- drive ---------------------------------------------------------------
  Part(
    "gearmotor_37d_50", "Pololu 50:1 Metal Gearmotor 37Dx70L mm 12V with "
    "64 CPR Encoder (Helical Pinion)", "motor", "chosen", ("body",),
    ("pluggybot",), partNumber="4753",
    source=ECKSTEIN + "Pololu-501-Metal-Gearmotor-37Dx70L-mm-12V-with-64CPR-"
    "EncoderHelical-Pinion-EN",
    massG=205, dimensionsMm={"diameter": 37, "length": 70}, priceEur=84.43,
    quantity=2,
    capabilities={"gearRatio": 50, "noLoadRpm": 200, "noLoadRadS": 20.9,
                  "stallTorqueNm": 2.06, "stallA": 5.5, "noLoadA": 0.2,
                  "encoderCprOutput": 3200, "voltageV": 12, "shaftMm": 6},
    feeds=(
      Feed("left_motor.forcerange",
           same(actuator("left_motor", "forcerange"),
                actuator("right_motor", "forcerange")),
           "N·m", "stall torque at 12 V, both motors", expect=2.06),
      Feed("left_motor.ctrlrange",
           same(actuator("left_motor", "ctrlrange"),
                actuator("right_motor", "ctrlrange")),
           "rad/s", "the 20.9 rad/s no-load speed, rounded"),
      Feed("left_wheel_joint.armature",
           same(joint("left_wheel_joint", "armature"),
                joint("right_wheel_joint", "armature")),
           "kg·m²", "reflected rotor inertia, which grows with the square of "
           "the 50:1 ratio; load-bearing for sim stability (SimNotes)"),
      Feed("left_wheel_joint.damping",
           same(joint("left_wheel_joint", "damping", 0),
                joint("right_wheel_joint", "damping", 0)),
           "N·m·s/rad", "the gearbox's ~30-35 % torque loss (Pololu's ~65 % "
           "efficiency)"),
      Feed("left_wheel_joint.frictionloss",
           same(joint("left_wheel_joint", "frictionloss"),
                joint("right_wheel_joint", "frictionloss")),
           "N·m", "the parking brake a real gearbox provides"),
      code("power.STALL_TORQUE", "N·m", expect=2.06),
      code("power.STALL_A", "A", "per motor; brushed-DC current scales with "
           "torque", expect=5.5),
      code("power.NOLOAD_A", "A", expect=0.2),
      code("power.NOLOAD_SPEED", "rad/s", "the no-load speed, rounded"),
    ),
    note="Passed over: the 30:1 sibling (#4752, same price) -- push force "
         "over top speed. The motor has no geom of its own; its 205 g is "
         "inside the chassis slab's 0.9 kg.",
  ),
  Part(
    "gearmotor_37d_30", "Pololu 30:1 Metal Gearmotor 37Dx70L mm 12V with "
    "64 CPR Encoder (Helical Pinion)", "motor", "candidate", ("body",),
    ("pluggybot",), partNumber="4752",
    source="https://www.pololu.com/product/4752", priceEur=84.43,
    capabilities={"gearRatio": 30, "noLoadRpm": 330, "stallTorqueNm": 1.37},
    why={"massG": "Parts.md does not weigh the runner-up",
         "dimensionsMm": "same 37D family; not recorded for the runner-up"},
    note="The runner-up: 330 rpm and 1.37 N·m against the 50:1's 200 rpm "
         "and 2.06 N·m. Speed is a low priority for this robot.",
  ),
  Part(
    "wheel_90x10", "Pololu Wheel 90×10 mm pair", "structure", "chosen",
    ("body",), ("pluggybot",), partNumber="1435-1439 (by colour)",
    source=ECKSTEIN + "Pololu-Wheel-90x10mm-Pair-Red-for-Micro-Metal-"
    "Gearmotors-EN",
    dimensionsMm={"diameter": 90, "width": 10}, priceEur=11.13,
    priceFor="pair", quantity=2,
    capabilities={"diameterMm": 90, "widthMm": 10, "mountingHoles": "6× M3"},
    feeds=(
      code("control.WHEEL_RADIUS", "m", "half the 90 mm diameter",
           expect=0.045),
      Feed("left_tire.size[0]",
           same(geom("left_tire", "size", 0), geom("right_tire", "size", 0)),
           "m", "tyre radius", expect=0.045),
      Feed("left_tire.size[1]",
           same(geom("left_tire", "size", 1), geom("right_tire", "size", 1)),
           "m", "half the 10 mm width", expect=0.005),
      Feed("left_tire.mass",
           same(geom("left_tire", "mass"), geom("right_tire", "mass")),
           "kg", "a placeholder: the part's mass is TBD"),
    ),
    why={"massG": TBD},
    note="Pololu's 60-70 mm wheels fit 3 mm shafts only; for the 37D's 6 mm "
         "D-shaft the verified path is a 90 mm wheel on a universal hub. "
         "Top speed 20.9 × 0.045 ≈ 0.94 m/s, stall push ≈ 46 N per wheel.",
  ),
  Part(
    "hub_6mm_m3", "Pololu Universal Aluminum Mounting Hub, 6 mm shaft, M3 "
    "holes (2-pack)", "structure", "chosen", ("body",), ("pluggybot",),
    partNumber="1999",
    source=ECKSTEIN + "PololuUniversalAluminumMountingHubfor6mmShaft2CM3Holes"
    "2-PackEN",
    priceEur=11.95, priceFor="2-pack", quantity=2,
    capabilities={"shaftMm": 6, "holes": "M3"},
    why={"massG": "Parts.md does not weigh it",
         "dimensionsMm": "not recorded in Parts.md"},
    note="Set-screw hub for the 6 mm D-shaft; the wheel bolts to it.",
  ),
  Part(
    "wheel_80x10_multihub", "Pololu Multi-Hub Wheel 80×10 mm (2-pack)",
    "structure", "candidate", ("body",), ("pluggybot",),
    source=ECKSTEIN + "Pololu-Multi-Hub-Wheel-w-Inserts-for-3mm-and-4mm-"
    "Shafts-8010mm-Black-2-pack-EN",
    dimensionsMm={"diameter": 80, "width": 10}, priceEur=14.20,
    priceFor="2-pack",
    why={"partNumber": "Parts.md names no number for the alternative",
         "massG": "Parts.md does not weigh it"},
    note="The alternative. Its inserts are 3 and 4 mm only -- whether it "
         "accepts the 6 mm universal hub is unverified.",
  ),
  # ---- chassis -------------------------------------------------------------
  Part(
    "ball_caster_19mm", "Pololu Ball Caster with 3/4″ metal ball",
    "structure", "chosen", ("body",), ("pluggybot",), partNumber="955",
    source="https://www.exp-tech.de/zubehoer/mechanische-bauteile/5551/"
    "pololu-ball-caster-with-3/4-metal-ball",
    dimensionsMm={"ballDiameter": 19, "height": "21-25 (spacers)"},
    priceEur=4.50, quantity=1,
    capabilities={"ballDiameterMm": 19},
    feeds=(
      Feed("caster.size[0]", geom("caster", "size", 0), "m",
           "sphere radius. ⚠ 20 mm: the sim's sphere is 40 mm across, the "
           "caster's stance rather than its 19 mm ball"),
      Feed("caster.priority", geom("caster", "priority"), "",
           "without it MuJoCo takes the pair MAX and the caster drags -- "
           "SimNotes, THE caster lesson", expect=1),
      Feed("caster.condim", geom("caster", "condim"), "",
           "frictionless, normal force only", expect=1),
      Feed("caster.mass", geom("caster", "mass"), "kg", "a placeholder"),
    ),
    why={"massG": TBD},
  ),
  Part(
    "bracket_37d", "Pololu 37D Metal Gearmotor Bracket (pair)", "structure",
    "candidate", ("body",), ("pluggybot",),
    source=ECKSTEIN + "Pololu-Motor-Mounts-Wheel-EN", quantity=1,
    why={"partNumber": "Parts.md names the category, not a number",
         "massG": "not recorded", "dimensionsMm": "not recorded",
         "priceEur": "Parts.md: TBD"},
    note="Sets the motor axle height above the chassis plate, and the "
         "bracket spacing sets the track width. Open decision 4 (chassis "
         "material) is blocked on this choice.",
  ),
  Part(
    "chassis_plate", "Chassis plate (laser-cut acrylic/alu or a stock "
    "profile)", "structure", "candidate", ("body",), ("pluggybot",),
    quantity=1,
    feeds=(
      Feed("chassis.size", geom("chassis", "size"), "m",
           "half-extents: a 24 × 18 × 6 cm slab"),
      Feed("chassis.mass", geom("chassis", "mass"), "kg",
           "the plate plus everything unmodelled on it: motors, driver, Pi"),
      code("control.TRACK_WIDTH", "m", "wheel centre to wheel centre",
           expect=0.21),
      Feed("left_wheel.pos[1]",
           same(body_pos("left_wheel", 1), neg(body_pos("right_wheel", 1))),
           "m", "half the track width: the wheels sit at ±0.105",
           expect=0.105),
    ),
    why={"partNumber": "no supplier chosen (open decision 4)",
         "source": "no supplier chosen", "massG": "not recorded",
         "dimensionsMm": "set by the motor brackets, once chosen",
         "priceEur": "Parts.md: TBD"},
  ),
  Part(
    "bumper_switch", "Front bumper: a sprung bar over 1-2 roller-lever "
    "microswitches (Omron D2F class)", "sensor", "chosen",
    ("body", "catalog"), ("pluggybot",), quantity=2,
    capabilities={"mountHeightMm": [60, 120]},
    feeds=(
      code("rack.swap.PRESS_RELEASE_S", "s",
           "a press is held this long past the last contact: a rigid chassis "
           "bounces off a rigid fence, a sprung bar stays pressed through it"),
    ),
    why={"partNumber": "any roller-lever micro; Parts.md names Omron's D2F "
         "series as the example, and a series is not a part number",
         "source": "no supplier chosen", "massG": "not recorded",
         "dimensionsMm": "not recorded",
         "priceEur": "€2-5 depending on the switch (Parts.md)"},
    note="Feeds `HubSwap.pressing`: dead reckoning holds its travel while "
         "the bumper is pressed on the side the wheels roll toward. In the "
         "sim the switch is the chassis box's front face read off the "
         "contact list -- nothing a €1 switch does not report. Blind spot: "
         "anything between the bumper's top (~12 cm) and the lidar plane "
         "(22 cm) meets the fork, not the bar.",
  ),
  # ---- vision & ranging ----------------------------------------------------
  Part(
    "lidar_rplidar_c1", "Slamtec RPLIDAR C1 (or A1M8) 2D LIDAR", "sensor",
    "chosen", ("body",), ("pluggybot",), partNumber="RPLIDAR C1", massG=110,
    quantity=1,
    capabilities={"rangeM": 12, "rateHz": 10, "fovDeg": 360,
                  "accuracyMm": 30, "powerW": 2.5, "interface": "USB/UART"},
    feeds=(
      code("perception.lidar.LIDAR_PERIOD", "s", "10 Hz, the part's rate",
           expect=0.1),
      code("perception.lidar.MAX_RANGE", "m",
           "8 m, UNDER the part's 12 m on purpose: a no-return ray only "
           "clears free space the robot will scan again, and the per-scan "
           "cost scales with it"),
      Feed("lidar_body.mass", geom("lidar_body", "mass"), "kg",
           "the unit, mounted high (body-local z 0.15) for a clear scan "
           "plane at 0.223 m", expect=0.11),
      Feed("lidar_body.size", geom("lidar_body", "size"), "m",
           "cylinder radius and half-height"),
      code("power.ELECTRONICS_W", "W",
           "Pi 5 + cameras + IMU + LIDAR, always on; was 6.0 before the "
           "unit's ~2.5 W"),
    ),
    why={"source": "Parts.md lists stockists (Botland, EXP-Tech, Welectron) "
         "without a link", "dimensionsMm": "not recorded",
         "priceEur": "€65-100 depending on the stockist and the A1M8 "
         "alternative (Parts.md)"},
    note="Replaced the stereo pair (Aug 2026): a ~17° blind sector "
         "behind-right where the mast stands in the scan plane is measured "
         "by `Lidar.blind_fraction`, and self-hits are dropped rather than "
         "reported as free space.",
  ),
  Part(
    "pi_camera_3", "Raspberry Pi Camera Module 3", "sensor", "chosen",
    ("body", "catalog"), ("pluggybot",), partNumber="Camera Module 3",
    source="https://www.welectron.com/Official-Raspberry-Pi-Camera-Module-3",
    massG=4, dimensionsMm={"length": 25, "width": 24, "height": 11.5},
    priceEur=25.50, quantity=2,
    capabilities={"sensor": "IMX708", "fovHDeg": 66, "fovVDeg": 41,
                  "autofocus": "PDAF, ~10 cm to infinity", "powerW": None},
    feeds=(
      Feed("left_eye.fovy", camera("left_eye"), "°",
           "the navigation camera on the head: AprilTags", expect=41),
      Feed("dock_eye.fovy", camera("dock_eye"), "°",
           "the docking camera on the lift carriage, so it rises with the "
           "fork", expect=41),
    ),
    note="Two cameras on the Pi 5's two CSI ports -- no multiplexer. Size "
         "and mass from raspberrypi.com's camera documentation; its draw "
         "is not published there, so a tool cannot budget for it yet.",
  ),
  Part(
    "realsense_d435", "RealSense D435 depth camera (active IR stereo)",
    "sensor", "chosen", ("body",), ("pluggybot",), partNumber="D435",
    massG=72, dimensionsMm={"length": 90, "width": 25, "height": 25},
    quantity=1,
    capabilities={"fovHDeg": 87, "fovVDeg": 58, "depthStream": "848x480 @ 30 Hz",
                  "minZM": 0.28, "specRangeM": 10, "baselineMm": 50,
                  "projector": "IR dot pattern: depth on textureless "
                               "surfaces, which passive stereo could not do "
                               "here", "interface": "USB 3", "powerW": None},
    feeds=(
      Feed("depth_eye.fovy", camera("depth_eye"), "°",
           "the near-field camera on the mast top, pitched 40° at the "
           "floor ahead", expect=58),
      Feed("depth_cam_body.mass", geom("depth_cam_body", "mass"), "kg",
           "the unit, on the mast top over the axle: CoM +10.7 mm, cruise "
           "launch pitch 0.9 -> 1.3°", expect=0.072),
      Feed("depth_cam_body.size", geom("depth_cam_body", "size"), "m",
           "box half-extents"),
      code("perception.depth.MIN_Z", "m",
           "the datasheet's min-Z at full resolution; does not bind on the "
           "mast-top mount", expect=0.28),
      code("perception.depth.MAX_Z", "m",
           "3 m, UNDER the part's 10 m: σ is 32 mm there and the map is "
           "near-field; also the ray cutoff, so the cost"),
      code("perception.depth.BASELINE", "m",
           "the imager baseline: the occlusion shadow's width and, with the "
           "sub-pixel error, the noise", expect=0.05),
      code("perception.depth.NOISE_K", "1/m",
           "σ_z = NOISE_K · z²: 0.08 px of disparity error on the real "
           "447 px focal length and 50 mm baseline"),
      code("perception.depth.PERIOD", "s",
           "10 Hz in sim, the LIDAR's rate; the part streams 30"),
    ),
    why={"source": "RealSense left Intel in 2025; order from a distributor "
         "(Mouser, Reichelt) -- link not verified",
         "priceEur": "roughly €300-400 at EU distributors in 2026, "
         "unverified"},
    note="Issue #34: the near-field sensor, so the robot can find things "
         "on the floor the scan plane looks over. Chosen over the D405 "
         "(7-50 cm, but PASSIVE stereo: fails on painted floors and matte "
         "printed modules exactly as the DIY pair did), a CSI time-of-flight "
         "module (the Pi 5's two CSI ports are the two cameras) and a second "
         "LIDAR tilted at the floor (a line, only while moving; cannot look "
         "at a thing from a standstill). Its draw is NOT in "
         "`power.ELECTRONICS_W` yet: nothing in the mission loop reads it, "
         "and the electrical budget lands with the loop integration and the "
         "energy table's re-measure (docs/Parts.md \"near-field depth camera\").",
  ),
  Part(
    "stereo_pair_diy", "DIY stereo: 2× Camera Module 3 on a custom 60 mm "
    "bracket", "sensor", "removed", ("body",), ("pluggybot",),
    capabilities={"baselineMm": 60},
    why={"partNumber": "an assembly, not a part", "source": "an assembly",
         "massG": "not recorded", "dimensionsMm": "not recorded",
         "priceEur": "two cameras and a printed bracket; never priced as one"},
    note="Superseded Aug 2026 by LIDAR + one camera. Measured on this sim's "
         "own pair: real SGBM produced disparity for 49.7 % of the mapper's "
         "scan row at 593 mm median error, against a 50 mm grid cell. Flat "
         "painted walls are the classic no-disparity case. The plug robot "
         "(`pluggybot.xml`) keeps its pair, frozen for milestone 6-7.",
  ),
  Part(
    "oak_d_lite", "Luxonis OAK-D Lite", "sensor", "candidate", ("body",),
    ("pluggybot",), partNumber="OAK-D Lite",
    capabilities={"baselineMm": 75, "minDepthCm": 35},
    why={"source": "no stockist recorded", "massG": "not recorded",
         "dimensionsMm": "not recorded",
         "priceEur": "€193-199 (Parts.md)"},
    note="The stereo alternative the DIY pair was picked over, before the "
         "question of whether any stereo pair could build the map was asked.",
  ),
  # ---- arm & docking -------------------------------------------------------
  Part(
    "igus_dle_la_0001", "igus drylin E lead-screw stepper linear actuator, "
    "NEMA11", "actuator", "chosen", ("body",), ("pluggybot",),
    partNumber="DLE-LA-0001",
    source="https://www.igus.com/product/DLE-LA-0001", quantity=2,
    capabilities={"motion": "slide", "forceN": 50, "holdingTorqueNm": 0.12,
                  "leadMmPerRev": 5.08, "stepMm": 0.0254,
                  "flange": "NEMA11 / 28 mm",
                  "strokeMm": {"lift": "~250 wanted", "reach": "~200 wanted"}},
    feeds=(
      Feed("lift.forcerange", actuator("lift", "forcerange"), "N",
           "max thrust: 6× the worst-case 7.8 N Schuko insertion", expect=50),
      Feed("arm.forcerange", actuator("arm", "forcerange"), "N",
           "the same unit on the reach axis", expect=50),
      Feed("lift_joint.range[1]", joint("lift_joint", "range", 1), "m",
           "the model's lift travel; the part wants ~0.25 m"),
      Feed("arm_joint.range[1]", joint("arm_joint", "range", 1), "m",
           "the model's reach"),
      code("power.ACTUATOR_W", "W", "drawn only while moving: the 0.12 N·m "
           "holding torque holds position unpowered"),
      Feed("lift_motor.mass", geom("lift_motor", "mass"), "kg",
           "a placeholder pending the igus quote"),
      Feed("mast.mass", geom("mast", "mass"), "kg", "a placeholder"),
    ),
    why={"massG": "TBD: igus's datasheet is per configured stroke",
         "dimensionsMm": "stroke-configured; the length depends on the quote",
         "priceEur": "quote-only: igus prices stroke-configured units "
         "through a configurator, not a list price"},
    note="Open decision 9: get one quote for both axes. The lift must span "
         "outlet heights 0.26-0.38 m and carry the docking camera high enough "
         "to keep a 0.38 m outlet in frame.",
  ),
  Part(
    "schuko_plug", "Rewireable Schuko CEE 7/7 right-angle plug (Type F)",
    "power", "chosen", ("body", "catalog"), ("module_plug",),
    source="https://leadsdirect.co.uk/shop/schuko-cee77-plug-rewireable-"
    "black-right-angle/",
    dimensionsMm={"bodyDiameter": 36.7, "pinLength": 19, "pinDiameter": 4.8,
                  "pinPitch": 19}, quantity=1,
    capabilities={"ratingA": 16, "ratingV": 250},
    feeds=(
      code("docking.schuko.PIN_LEN", "m", expect=0.019),
      code("docking.schuko.R_PIN", "m", "half the 4.8 mm pin", expect=0.0024),
      code("docking.schuko.PIN_SEP", "m", "half the 19 mm pin pitch",
           expect=0.0095),
      code("docking.schuko.R_BODY", "m",
           "⚠ 35.5 mm across, against the real 36.7: 0.15 mm of clearance "
           "per side in a 37 mm recess, not 0.75 -- open decision 8"),
      Feed("module_plug_barrel.size[0]", geom("module_plug_barrel", "size", 0),
           "m", "the module's barrel, the spike's radius"),
    ),
    why={"partNumber": "a class of part (any rewireable CEE 7/7); Parts.md "
         "links one example", "massG": "not recorded",
         "priceEur": "€3-6 depending on the example (Parts.md)"},
  ),
  Part(
    "rcc_wrist", "Compliant wrist: four compression springs and a floating "
    "plate (remote centre compliance -- build, don't buy)", "structure",
    "chosen", ("body",), ("pluggybot",), priceEur=10, quantity=1,
    capabilities={"recoversMm": 4, "recoversDeg": 9},
    feeds=(
      code("rack.coupling.LAT_STIFFNESS", "N/m",
           "a GUESS: measure the built part; the whole tolerance envelope "
           "scales with it"),
      code("rack.coupling.YAW_STIFFNESS", "N·m/rad", "a guess, likewise"),
      Feed("fork_lat_y.stiffness",
           same(joint("fork_lat_y", "stiffness", 0),
                joint("fork_lat_z", "stiffness", 0)),
           "N/m", "the fork's wrist joints", expect=150),
      Feed("fork_rot_z.stiffness",
           same(joint("fork_rot_z", "stiffness", 0),
                joint("fork_rot_y", "stiffness", 0)),
           "N·m/rad", expect=1.0),
    ),
    why={"partNumber": "built, not bought", "source": "built, not bought",
         "massG": "not weighed", "dimensionsMm": "not recorded"},
    note="Published RCC devices recover ~4 mm and ~9°, more than the "
         "±3 mm / ±3° docking budget, with no control loop.",
  ),
  Part(
    "alignment_feelers", "Alignment feelers: two prongs on the lift carriage "
    "straddling the socket", "structure", "removed", ("body",),
    ("pluggybot",),
    why={"partNumber": PRINTED, "source": PRINTED, "massG": "not weighed",
         "dimensionsMm": "not recorded", "priceEur": PRINTED},
    note="Removed from the hub robot: they bake in an outlet-housing width "
         "real outlets do not standardise. The plug robot keeps them "
         "(`prong_l`/`prong_r`) frozen for milestone 6-7; the fork's tines "
         "are a different part.",
  ),
  # ---- power ---------------------------------------------------------------
  Part(
    "battery_3s_5000", "3S LiPo pack, ≥ 5000 mAh (~55 Wh), XT60",
    "power", "candidate", ("body",), ("pluggybot",), quantity=1,
    capabilities={"nominalV": 11.1, "chargedV": 12.6, "capacityMah": 5000,
                  "capacityWh": 55.5, "dischargeC": 20, "connector": "XT60"},
    feeds=(
      code("power.NOMINAL_V", "V", expect=11.1),
      code("power.CHARGE_W", "W", "~1C into the 5 Ah pack"),
      code("power.DEMO_CAPACITY_WH", "Wh",
           "a knob, not the pack: the demo cell; `--battery-wh 55.5` runs "
           "the real one"),
      Feed("battery.mass", geom("battery", "mass"), "kg",
           "the ~400 g placeholder -- the traction ballast"),
      Feed("battery.size", geom("battery", "size"), "m",
           "half-extents: a 13 × 4.6 × 2.6 cm bay for a 5000 mAh 3S pack"),
      Feed("battery.pos", geom("battery", "pos"), "m",
           "ahead of centre for tipping margin, +y as the counterweight to "
           "the arm at y = -0.05; without it the robot veers 26 cm right "
           "over 4 m open-loop"),
    ),
    why={"partNumber": "part TBD: pick a specific pack from a German "
         "retailer and check its footprint against the chassis plate",
         "source": "part TBD", "massG": "no pack chosen; ~400 g is the "
         "target and the sim's placeholder", "dimensionsMm": "no pack chosen",
         "priceEur": "no pack chosen"},
    note="Open decision 6. A real pack of a different mass or footprint "
         "moves every physics threshold derived from the model.",
  ),
  Part(
    "charger_3s_board", "12.6 V CC/CV charger board (3S) fed by a mains "
    "adapter", "power", "candidate", ("body",), ("rack",), quantity=1,
    why={"partNumber": "no board chosen", "source": "no board chosen",
         "massG": "not recorded", "dimensionsMm": "not recorded",
         "priceEur": "€10-15 class (Parts.md)"},
    note="Hub power: replaces wall-outlet charging as the primary path; "
         "balance leads handled robot-side by a 3S BMS.",
  ),
  Part(
    "pogo_pins", "Charge contacts: spring-loaded pogo-pin pairs on the hub "
    "face, pads on the robot", "power", "chosen", ("body",),
    ("rack", "pluggybot"), priceEur=5, quantity=1,
    feeds=(
      code("rack.coupling.CHARGE_PIN_Z", "m",
           "pins at bumper height (chassis 0.06-0.12): preload comes from "
           "the drive-in press"),
      Feed("rack_pin_l.mass",
           same(geom("rack_pin_l", "mass"), geom("rack_pin_r", "mass")),
           "kg"),
    ),
    why={"partNumber": "no pin chosen", "source": "no supplier chosen",
         "massG": "not recorded", "dimensionsMm": "not recorded"},
    note="Open for the physical design: placement that engages by the same "
         "drive-in motion, with no extra alignment.",
  ),
  # ---- electronics ---------------------------------------------------------
  Part(
    "raspberry_pi_5", "Raspberry Pi 5, 8 GB", "electronics", "candidate",
    ("body",), ("pluggybot",), partNumber="Raspberry Pi 5 8GB",
    source="https://www.berrybase.de/en/raspberry-pi-5-8gb-ram",
    dimensionsMm={"length": 85, "width": 56}, priceEur=202.90, quantity=1,
    capabilities={"csiPorts": 2},
    why={"massG": "not recorded"},
    note="€202.90 is the cheapest Geizhals listing (July 2026) INCLUDING an "
         "active-cooler kit, well above the historical ~€90 board price; "
         "verify the standalone board before budgeting. No accelerator HAT.",
  ),
  Part(
    "motor_driver_mdd10a", "Cytron MDD10A dual-channel motor driver",
    "electronics", "candidate", ("body",), ("pluggybot",),
    partNumber="MDD10A",
    source="https://botland.store/drivers-for-dc-motors/15818-cytron-mdd10a-"
    "dual-channel-30v-10a-motor-controller-5904422350444.html",
    quantity=1,
    capabilities={"channels": 2, "continuousA": 10, "peakA": 30,
                  "voltageV": [5, 30]},
    why={"massG": "not recorded", "dimensionsMm": "not recorded",
         "priceEur": "Parts.md: TBD, ~€20 class"},
    note="10 A continuous per channel comfortably covers the 5.5 A stall "
         "current per motor.",
  ),
  Part(
    "module_esp32", "ESP32-class board, one per module", "electronics",
    "chosen", ("body", "catalog"),
    ("module_lcd", "module_plug", "module_pen", "module_claw", "module_seed"),
    priceEur=5, quantity=5,
    feeds=(
      code("power.MODULE_IDLE_W", "W",
           "a coupled module's own electronics, drawn only while the "
           "coupling conducts"),
    ),
    why={"partNumber": "any ESP32 dev board; none chosen",
         "source": "no board chosen", "massG": "not recorded",
         "dimensionsMm": "not recorded"},
    note="Power-only coupling, wireless data: keeps the mating interface "
         "dumb and tolerant.",
  ),
  Part(
    "lcd_display", "Small SPI/I2C display driven by the module's ESP32",
    "electronics", "candidate", ("body", "catalog"), ("module_lcd",),
    quantity=1,
    feeds=(
      code("rack.coupling.LCD_SCREEN_HALF", "m",
           "half-extents of the screen geom: a 56 × 76 mm face"),
    ),
    why={"partNumber": "no display chosen", "source": "no display chosen",
         "massG": "not recorded", "dimensionsMm": "not recorded",
         "priceEur": "no display chosen"},
    note="Display-only; the face is drawn in the browser off a streamed "
         "enum.",
  ),
  # ---- rack & modules ------------------------------------------------------
  Part(
    "peg_rod_6mm", "Tool peg axle: 6 mm steel rod, 150 mm, two 63 mm "
    "conductors on a 24 mm insulating bush", "structure", "chosen",
    ("body", "catalog"),
    ("module_lcd", "module_plug", "module_pen", "module_claw", "module_seed"),
    dimensionsMm={"diameter": 6, "length": 150, "conductor": 63,
                  "bush": 24}, quantity=1,
    capabilities={"conductive": True, "poles": 2},
    feeds=(
      code("rack.coupling.PEG_R", "m", "half the 6 mm rod", expect=0.003),
      code("rack.coupling.PEG_HALF", "m", "half the 150 mm rod",
           expect=0.075),
      code("rack.coupling.PEG_INSUL_HALF", "m", "half the 24 mm bush",
           expect=0.012),
      code("rack.coupling.PEG_COND_HALF", "m", "half a 63 mm conductor",
           expect=0.0315),
      code("rack.coupling.PEG_FRICTION", "", "honest peg friction: the "
           "measured worst power outage under hard driving is 178 ms, which "
           "sizes the module's holding capacitor (~200 ms)"),
      code("rack.coupling.PEG_MASS", "kg", "the rod's share of the module "
           "budget, split 8 + 8 + 4 g over the three sections"),
      Feed("module_lcd_peg_l.mass",
           total(geom("module_lcd_peg_l", "mass"),
                 geom("module_lcd_peg_r", "mass"),
                 geom("module_lcd_peg_insul", "mass")),
           "kg", "the three sections together", expect=0.02),
    ),
    why={"partNumber": STOCK, "source": STOCK,
         "massG": "not weighed. ⚠ The sim's peg totals 20 g; a 150 mm "
         "length of 6 mm steel would weigh ~33 g -- an open discrepancy "
         "for #168's validator to carry, not a number to invent here",
         "priceEur": STOCK},
    note="The one loaded part AND the electrical connector: split peg + "
         "the fork's two V-notch pairs are a two-pole coupling with "
         "0.43-0.47 N of gravity preload per plate, self-wiping on the "
         "seating slide.",
  ),
  Part(
    "module_frame", "Module frame: a 3D-printed plate on the common peg "
    "interface", "structure", "chosen", ("body", "catalog"),
    ("module_lcd", "module_plug", "module_pen", "module_claw", "module_seed"),
    dimensionsMm={"x": 20, "y": 40, "z": 60}, quantity=1,
    capabilities={"massBudgetG": 120, "massCeilingG": 250,
                  "momentBudgetNm": 0.45},
    feeds=(
      code("rack.coupling.MODULE_MASS", "kg",
           "the plate + peg budget every module is emitted at; the face "
           "sits on top (measured modules: 143-211 g)", expect=0.12),
      Feed("module_lcd_body.mass", geom("module_lcd_body", "mass"), "kg",
           "the plate alone: the budget less the peg. ⚠ 100 g is a budget, "
           "not a print: this plate in PLA would weigh ~60 g", expect=0.10),
      Feed("module_lcd_body.size", geom("module_lcd_body", "size"), "m",
           "half-extents of the plate"),
    ),
    why={"partNumber": PRINTED, "source": PRINTED,
         "massG": "not weighed; the sim budgets 100 g for the plate",
         "priceEur": "printed: filament only, unpriced"},
    note="The moment budget is the number that shapes tools: the gravity "
         "latch takes ~0.45 N·m of pitch before the peg rides out of its V, "
         "so reach is far dearer than mass (ToolPattern.md §2).",
  ),
  Part(
    "rack_rail_stock", "Rack rail, posts and base from stock material "
    "(wood, or 2020 aluminium extrusion)", "structure", "chosen", ("body",),
    ("rack",), dimensionsMm={"railLength": 1860, "width": 1890, "depth": 170,
                             "height": 550}, quantity=1,
    capabilities={"bays": 5, "pitchMm": 250},
    feeds=(
      code("rack.coupling.RACK_HALF_W", "m", "half the rail", expect=0.93),
      code("rack.coupling.RACK_RAIL_Z", "m", "rail height"),
      code("rack.coupling.HUB_STATION_YS", "m",
           "the five tool bays at 0.25 m pitch; appended to, never "
           "reordered (bay-tag pairing is by index)"),
      Feed("rack_shelf.mass", geom("rack_shelf", "mass"), "kg", "the base"),
      Feed("rack_rail.mass", geom("rack_rail", "mass"), "kg"),
      Feed("rack_post_l.mass",
           same(geom("rack_post_l", "mass"), geom("rack_post_r", "mass")),
           "kg"),
    ),
    why={"partNumber": STOCK, "source": STOCK, "massG": "not weighed",
         "priceEur": STOCK},
    note="Does NOT fit a 220 × 220 mm print bed and should not try. 2020 "
         "extrusion makes the bay pitch adjustable later. Only the parts "
         "that touch the peg need print accuracy.",
  ),
  Part(
    "rack_printed_parts", "Rack V-trays, tray brackets, tag plates and the "
    "robot's fork: 3D-printed (PETG)", "structure", "chosen", ("body",),
    ("rack", "pluggybot"), quantity=1,
    capabilities={"material": "PETG", "trayLoadN": 3},
    feeds=(
      code("rack.coupling.V_HALF_LEN", "m",
           "tilted plate half-length: ~8 mm of usable V depth"),
      code("rack.coupling.V_THICK", "m"),
      code("rack.coupling.TRAY_Y", "m", "the shelf's inboard V-tray pair"),
      code("rack.coupling.FORK_Y", "m",
           "the fork's prongs grab outboard of the trays: ±58 mm stance"),
      code("rack.coupling.PLATE_HALF_T", "m",
           "every fiducial plate is a 4 mm box"),
      Feed("fork_vl_a.mass",
           same(geom("fork_vl_a", "mass"), geom("fork_vl_b", "mass"),
                geom("fork_vr_a", "mass"), geom("fork_vr_b", "mass")),
           "kg", "one fork V plate"),
    ),
    why={"partNumber": PRINTED, "source": PRINTED, "massG": "not weighed",
         "dimensionsMm": "many parts; each is a `rack/coupling.py` constant",
         "priceEur": PRINTED},
    note="Open: whether the trays need steel wear inserts.",
  ),
  Part(
    "m3_fasteners", "M3 screws and nuts (assorted)", "fastener", "chosen",
    ("body", "catalog"), ("pluggybot", "rack"),
    why={"partNumber": "assorted stock hardware", "source": "any",
         "massG": "not recorded", "dimensionsMm": "assorted",
         "priceEur": "not priced"},
    note="The wheel's six mounting holes and the universal hub are M3.",
  ),
  # ---- catalog-only: what a module is built from and nobody has chosen ----
  Part(
    "module_lead_screw", "Small lead-screw slide for a module axis "
    "(unspecified)", "actuator", "candidate", ("catalog",), ("module_pen",),
    capabilities={"motion": "slide", "forceN": None, "strokeMm": 110,
                  "powerW": None},
    feeds=(
      Feed("pen_carriage.forcerange", actuator("pen_carriage", "forcerange"),
           "N", "a GUESS: no part chosen"),
      Feed("pen_carriage.gainprm[0]", gain("pen_carriage"), "N/m",
           "kp is a LEAD SCREW's stiffness, not a hobby servo's: at 120 the "
           "figure came out 12 mm RMS off because the pen drags"),
      code("rack.coupling.PEN_TRAVEL", "m",
           "± along the peg axis: 110 mm of drawing width"),
      code("rack.coupling.PEN_CARRIAGE_MASS", "kg"),
      code("rack.coupling.PEN_RAIL_MASS", "kg"),
    ),
    why={"partNumber": "no part chosen", "source": "no part chosen",
         "massG": "no part chosen", "dimensionsMm": "no part chosen",
         "priceEur": "no part chosen"},
    note="The pen module's own axis: it brings an axis the base does not "
         "have. Its ±15 N is a sim guess with no datasheet behind it.",
  ),
  Part(
    "module_servo", "Small servo for a module axis (unspecified)", "actuator",
    "candidate", ("catalog",), ("module_claw", "module_seed"),
    capabilities={"motion": "hinge", "forceN": None, "powerW": None},
    feeds=(
      Feed("claw_l.forcerange",
           same(actuator("claw_l", "forcerange"),
                actuator("claw_r", "forcerange")),
           "N", "a GUESS: no part chosen"),
      Feed("seed_gate.forcerange", actuator("seed_gate", "forcerange"), "N",
           "a GUESS: no part chosen"),
      code("rack.coupling.CLAW_GRIP_KP", "N/m",
           "grip force = kp × squeeze past contact"),
      code("rack.coupling.CLAW_JAW_TRAVEL", "m", "inward travel to an 8 mm gap"),
      code("rack.coupling.DISP_GATE_KP", "N/m",
           "a small linear servo, not a solenoid: metering"),
      code("rack.coupling.DISP_STROKE", "m", "pocket to exit"),
    ),
    why={"partNumber": "no part chosen", "source": "no part chosen",
         "massG": "no part chosen", "dimensionsMm": "no part chosen",
         "priceEur": "no part chosen"},
    note="Both modules' ±20 N is a sim guess. Prices and models TBD "
         "(Parts.md, 'Tool hub & modules').",
  ),
  Part(
    "servo_fs90", "FEETECH FS90-FB micro servo (analog, position feedback)",
    "actuator", "chosen", ("catalog",), (), partNumber="FS90-FB",
    source="https://botland.store/micro-servos/17177-feetech-fs90-fb-micro-"
    "servo-with-position-feedback-5904422327088.html",
    massG=13, dimensionsMm={"length": 23, "width": 13, "height": 22},
    priceEur=2.90,
    capabilities={"motion": "hinge", "angleDeg": 120, "torqueNm": 0.147,
                  "speedDegS": 600, "voltageV": [4.8, 6.0], "stallA": 0.8,
                  "powerW": 4.8},
    note="The first catalog actuator with every number a tool needs: "
         "Botland's page gives mass, size, 1.5 kg·cm at 6 V, 0.10 s/60° "
         "and the 0-120° range; the stall current (800 mA at 6 V, so 4.8 W) "
         "is from Feetech's FS90 datasheet (pololu.com/file/0J1435), the "
         "same servo with a position wire added. 9 g there, 13 g here.",
  ),
  Part(
    "module_camera", "Module camera: wide-angle, wireless through the "
    "module's ESP32 (unspecified)", "sensor", "candidate", ("catalog",),
    ("module_claw",), capabilities={"fovDeg": None, "powerW": None},
    feeds=(
      Feed("claw_eye.fovy", camera("claw_eye"), "°",
           "wider than the Camera Module 3's 41°: it works at ~140 mm and "
           "no part has been chosen for it"),
    ),
    why={"partNumber": "no part chosen", "source": "no part chosen",
         "massG": "no part chosen", "dimensionsMm": "no part chosen",
         "priceEur": "no part chosen"},
    note="Mounted on the MODULE, a first: module data crosses the coupling "
         "wirelessly, so a tool camera costs no CSI port on the Pi.",
  ),
  Part(
    "scaffold_pla_box", "Printed PLA box or plate -- the one scaffold "
    "primitive", "scaffold", "chosen", ("catalog",), (),
    capabilities={"densityKgM3": 1240,
                  "printBedMm": {"x": 220, "y": 220, "z": 250}},
    why={"partNumber": "a primitive, not a part: any PLA filament",
         "source": "any", "massG": "density × the box the spec asks for",
         "dimensionsMm": "chosen by the spec, within the print bed",
         "priceEur": "priced per kg of filament, not per part; #168 sets "
         "the fabrication cost"},
    note="Rigid, at PLA's density, and no larger than a 220 × 220 × 250 mm "
         "bed (an Ender-3 class printer) in any one piece. The rack's trays "
         "and brackets and every module frame are this primitive; the rack "
         "itself is not, and could not be.",
  ),
)


def by_id() -> dict[str, Part]:
  return {p.id: p for p in PARTS}


def validate(entry: dict) -> list[str]:
  """Every reason one emitted entry is not honest, or `[]`.

  Used by the tests on every committed entry and shown to fail on a bad
  one; kept as a function so #168's validator can reuse it on a spec that
  names a part."""
  out: list[str] = []
  required = ("id", "name", "kind", "status", "shelves", "usedBy",
              "partNumber", "source", "massG", "dimensionsMm", "priceEur",
              "priceFor", "quantity", "capabilities", "feeds", "why", "note")
  for key in required:
    if key not in entry:
      out.append(f"missing {key}")
  if out:
    return out
  if entry["kind"] not in KINDS:
    out.append(f"kind {entry['kind']!r} not in {KINDS}")
  if entry["status"] not in STATUSES:
    out.append(f"status {entry['status']!r} not in {STATUSES}")
  if not entry["shelves"] or any(s not in SHELVES for s in entry["shelves"]):
    out.append(f"shelves {entry['shelves']!r} not a non-empty subset of {SHELVES}")
  for key in NULLABLE:
    if entry[key] is None and not entry["why"].get(key):
      out.append(f"{key} is null with no why")
  for key in entry["why"]:
    if key not in NULLABLE:
      out.append(f"why names {key!r}, which cannot be null")
    elif entry[key] is not None:
      out.append(f"why names {key!r}, which is not null (a stale excuse)")
  for key in ("partNumber", "source", "note", "name"):
    if entry[key] is not None and not isinstance(entry[key], str):
      out.append(f"{key} must be a string or null")
  if entry["source"] is not None and not entry["source"].startswith("http"):
    out.append("source must be a URL")
  for key in ("massG", "priceEur"):
    v = entry[key]
    if v is not None and not (isinstance(v, (int, float)) and v >= 0
                              and math.isfinite(v)):
      out.append(f"{key} must be a non-negative number or null")
  for f in entry["feeds"]:
    for key in ("constant", "where", "value", "unit", "note"):
      if key not in f:
        out.append(f"feed {f.get('constant')!r} missing {key}")
    if f.get("where") not in ("code", "model"):
      out.append(f"feed {f.get('constant')!r}: where must be code or model")
  if entry["status"] == "chosen" and "body" in entry["shelves"] \
      and entry["kind"] != "fastener" and not entry["feeds"] \
      and entry["partNumber"] is None:
    out.append("a chosen body part with neither a part number nor a feed "
               "is a part nobody can point at")
  return out


def mismatches(fixture: dict) -> list[str]:
  """Every feed whose sim value disagrees with the part's own number.

  THE pin: the igus's 50 N and the lift's `forcerange` are one fact, and a
  change to either side lands here, in the suite, rather than on the
  website as a spec sheet that quietly stopped describing the sim."""
  out = []
  for p in fixture["parts"]:
    for f in p["feeds"]:
      if "expect" in f and f["value"] != f["expect"]:
        out.append(f"{p['id']}: {f['constant']} is {f['value']}, the part "
                   f"says {f['expect']} {f['unit']}")
  return out


def build(world: str = WORLD) -> dict:
  """Read every feed off the compiled world and emit the fixture."""
  import mujoco
  from pluggybot.workshop.spec import unbuildable   # lazy: it imports this module
  spec = mujoco.MjSpec.from_file(world)
  parts = []
  for p in PARTS:
    entry = p.as_dict(spec)
    # WHETHER THE WORKSHOP MAY BUILD FROM IT (issue #168): the validator's
    # own predicate, so the page marks exactly the parts a spec may name
    # -- and says why the rest cannot be, in the validator's words.
    why = unbuildable(p)
    entry["workshop"] = {"usable": why is None, **({"why": why} if why else {})}
    parts.append(entry)
  ids = [p["id"] for p in parts]
  assert len(ids) == len(set(ids)), "duplicate part id"
  return {
    "schema": SCHEMA,
    "world": world,
    "kinds": list(KINDS),
    "statuses": list(STATUSES),
    "shelves": list(SHELVES),
    "nullable": list(NULLABLE),
    "parts": parts,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-o", "--out", default="protocol/parts.json")
  args = parser.parse_args()
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fixture = build()
  bad = {p["id"]: validate(p) for p in fixture["parts"]}
  bad = {k: v for k, v in bad.items() if v}
  if bad:
    raise SystemExit(f"refusing to write a dishonest catalog: {bad}")
  if off := mismatches(fixture):
    raise SystemExit("refusing to write a catalog the sim disagrees with: "
                     + "; ".join(off))
  out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
  shelves = {s: sum(s in p["shelves"] for p in fixture["parts"])
             for s in SHELVES}
  print(f"{out}: {len(fixture['parts'])} parts, {shelves}, "
        f"{sum(len(p['feeds']) for p in fixture['parts'])} feeds")


if __name__ == "__main__":
  main()
