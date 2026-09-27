"""The parts catalog as data (issue #185): `protocol/parts.json`.

`docs/Parts.md` names the purchasable part behind each sim constant; this
module is that list as data, in three shelves:

- **body** -- what Pluggy and its rack are made of. Fixed. Each entry names
  the part, what it costs, what it weighs, and which sim constant it FEEDS.
- **catalog** -- what the agent may build a tool from (issue #168). The
  body's parts where they make sense on a module, parts that exist only for
  building, and ONE scaffold primitive: a printed PLA box at a density and
  within a print bed.
- **build** -- the hardware build's bill of materials (issue #379): the
  quadruped (#377), its arm, rack, dock and tools (#378), and the rigs,
  spares and shop tools that build them, each part read against the
  quadruped's own model. `LINES` buys them: a quantity, a lead time, and an
  allowance where nothing is designed yet; `bom()` totals them against the
  budget, and Parts.md's bill is RENDERED from them (`bom_markdown`), so the
  doc cannot drift from the data either. The `body` shelf stays the rover's
  until it goes (#376 stage C).

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

Regenerate: `uv run python -m pluggybot.rack.catalog` (it rewrites
Parts.md's bill too). Vendored to the website beside `hints.json`; the parts
page is its first consumer, and shows the `build` shelf once its half lands.
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
         "fastener", "scaffold", "equipment")
STATUSES = ("chosen", "candidate", "removed")
SHELVES = ("body", "catalog", "build")

#: Fields a part may not know. Each null needs its reason in `why`, keyed by
#: the field, and `why` may name nothing that is not null (a stale excuse).
NULLABLE = ("partNumber", "source", "massG", "dimensionsMm", "priceEur")

#: The compiled world a part's model-side feeds and `usedBy` are read
#: against, by the robot the part is for: the rover's (the fork robot, the
#: rack and all five modules in one file) and the quadruped's body as
#: trained and served (#377). The arm, rack and dock are not in a served
#: world yet (#375 step 4): their feeds are their specs' constants.
WORLDS = {"rover": "models/room_hub.xml", "quadruped": "models/quadruped.xml"}
WORLD = WORLDS["rover"]


# ---- readers: a feed's value comes from HERE, never from a literal ---------

Reader = Callable[[Any], Any]


def _num(v: Any) -> Any:
  if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
    return [_num(x) for x in v]
  f = float(v)
  return int(f) if f.is_integer() else round(f, 6)


def _resolve(path: str) -> Any:
  """A dotted path under `pluggybot`: the longest prefix that imports as a
  module, then attributes (`legs.actuator.GIM8108_8.peak_torque` is a field
  of the motor that module defines; a dataclass's field reads its
  default)."""
  names = path.split(".")
  for i in range(len(names) - 1, 0, -1):
    try:
      obj = importlib.import_module("pluggybot." + ".".join(names[:i]))
    except ModuleNotFoundError:
      continue
    for name in names[i:]:
      obj = getattr(obj, name)
    return obj
  raise ModuleNotFoundError(path)


def const(path: str) -> Reader:
  """A Python constant by dotted path under `pluggybot`."""
  return lambda _spec: _num(_resolve(path))


def entry(path: str, key: str) -> Reader:
  """One value of a dict constant (`legs.model.ELECTRONICS_W`'s "lidar")."""
  return lambda _spec: _num(_resolve(path)[key])


def legs(kind: Callable[..., Reader], attr: str, *args) -> Reader:
  """The quadruped's twelve joints or their drivers as one number:
  `legs(actuator, "forcerange")`, refused if they have come apart."""
  def read(spec):
    from pluggybot.legs.model import JOINT_NAMES
    return same(*(kind(name, attr, *args) for name in JOINT_NAMES))(spec)
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


def site(name: str, attr: str, index: int | None = None) -> Reader:
  def read(spec):
    v = getattr(next(s for s in spec.sites if s.name == name), attr)
    return _num(v if index is None else v[index])
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


def switched_bays() -> Reader:
  """How many bays the compiled world gives a presence switch (issue #351):
  those whose switch V (`coupling.BAY_SWITCH_PLATES`) is in the model."""
  def read(spec):
    from pluggybot.rack.coupling import BAY_SWITCH_PLATES, STATION_YS, bay_prefix
    names = {g.name for g in spec.geoms}
    return sum(all(bay_prefix(i) + p in names for p in BAY_SWITCH_PLATES)
               for i in range(len(STATION_YS)))
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


def code_entry(constant: str, key: str, unit: str, note: str = "",
               expect: Any = None) -> Feed:
  return Feed(f"{constant}[{key!r}]", entry(constant, key), unit, note, expect,
              where="code")


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
  #: Which world its model-side feeds and `usedBy` are read against.
  robot: str = "rover"

  def as_dict(self, spec) -> dict:
    return {
      "id": self.id, "name": self.name, "kind": self.kind,
      "status": self.status, "shelves": list(self.shelves),
      "robot": self.robot,
      "usedBy": list(self.usedBy), "partNumber": self.partNumber,
      "source": self.source, "massG": self.massG,
      "dimensionsMm": self.dimensionsMm, "priceEur": self.priceEur,
      "priceFor": self.priceFor, "quantity": self.quantity,
      "capabilities": self.capabilities,
      "feeds": [f.as_dict(spec) for f in self.feeds],
      "why": self.why, "note": self.note,
    }


ECKSTEIN = "https://eckstein-shop.de/"
FUNDUINO_C1 = ("https://funduinoshop.com/en/electronic-modules/sensors/"
               "movement-distance/rplidar-c1-dtof-lidar-3600-laser-range-"
               "scanner-12m-ip54-slamtec")
DIGIKEY_D435 = "https://www.digikey.de/de/products/detail/realsense/82635AWGDVKPRQ/9926002"
BERRYBASE_CAM3 = "https://www.berrybase.de/raspberry-pi-camera-module-3-12mp"
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
      code("perception.encoders.WHEEL_COUNTS_PER_REV", "counts",
           "the encoder the reckoner counts: 64 a motor turn, x 50 (#386)",
           expect=3200),
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
    "bumper_switch", "Front bumper: a sprung bar over 2 Omron D2F-01L2 "
    "hinge-roller-lever microswitches", "sensor", "chosen",
    ("body", "catalog"), ("pluggybot",), partNumber="D2F-01L2",
    source="https://www.digikey.de/de/products/detail/omron-electronics-inc-"
    "emc-div/D2F-01L2/368444",
    massG=0.5, dimensionsMm={"length": 12.8, "width": 5.8, "height": 16.5},
    priceEur=2.59, quantity=2,
    capabilities={"mountHeightMm": [60, 120], "sense": "contact",
                  "operatingForceN": 0.78, "releasingForceN": 0.05,
                  "overtravelMm": 0.55, "contactRating": "0.1 A 30 V DC",
                  "powerW": 0},
    feeds=(
      code("rack.swap.PRESS_RELEASE_S", "s",
           "a press is held this long past the last contact: a rigid chassis "
           "bounces off a rigid fence, a sprung bar stays pressed through it"),
    ),
    note="Feeds `HubSwap.pressing`: dead reckoning holds its travel while "
         "the bumper is pressed on the side the wheels roll toward. In the "
         "sim the switch is the chassis box's front face read off the "
         "contact list -- nothing a €1 switch does not report. Blind spot: "
         "anything between the bumper's top (~12 cm) and the lidar plane "
         "(22 cm) meets the fork, not the bar. Sourced for issue #199: "
         "Omron's D2F datasheet (omronfs.omron.com/en_US/ecb/products/pdf/"
         "en-d2f.pdf) gives the 12.8 x 5.8 body, the 16.5 mm free position "
         "of the roller lever (the height here) and OF 0.78 N; its "
         "'approx. 0.5 g' is stated for the pin-plunger body, and the "
         "0.3 mm stainless lever adds an unpublished fraction of a gram. "
         "A switch draws nothing of its own (`powerW` 0). On a TOOL it is "
         "a `contact` sense: `<tool>.<id>.contact` reads 1 while the part "
         "touches anything outside the module (workshop/build.py).",
  ),
  # ---- vision & ranging ----------------------------------------------------
  Part(
    "lidar_rplidar_c1", "Slamtec RPLIDAR C1 (or A1M8) 2D LIDAR", "sensor",
    "chosen", ("body",), ("pluggybot",), partNumber="RPLIDAR C1", massG=110,
    source=FUNDUINO_C1, priceEur=84.90, quantity=1,
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
    why={"dimensionsMm": "not recorded"},
    note="Replaced the stereo pair (Aug 2026): a ~17° blind sector "
         "behind-right where the mast stands in the scan plane is measured "
         "by `Lidar.blind_fraction`, and self-hits are dropped rather than "
         "reported as free space.",
  ),
  Part(
    "pi_camera_3", "Raspberry Pi Camera Module 3", "sensor", "chosen",
    ("body", "catalog"), ("pluggybot",), partNumber="Camera Module 3",
    source=BERRYBASE_CAM3,
    massG=4, dimensionsMm={"length": 25, "width": 24, "height": 11.5},
    priceEur=28.90, quantity=2,
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
         "is not published there, so a tool cannot budget for it. Looked "
         "for again for issue #199 (2026-09-14): neither product brief "
         "(RP-008151, RP-009789) states it, the forum thread on it (t="
         "347469) has one whole-Pi-Zero figure and no camera-alone one, "
         "and Arducam's and InnoMaker's IMX708 CSI sheets omit it too. "
         "The tool eye is `esp32_cam`, whose maker does publish its draw.",
  ),
  Part(
    "realsense_d435", "RealSense D435 depth camera (active IR stereo)",
    "sensor", "chosen", ("body",), ("pluggybot",), partNumber="D435",
    source=DIGIKEY_D435, massG=75,
    dimensionsMm={"length": 90, "width": 25, "height": 25}, priceEur=340.01,
    quantity=1,
    capabilities={"fovHDeg": 87, "fovVDeg": 58, "depthStream": "848x480 @ 30 Hz",
                  "minZM": 0.28, "specRangeM": 10, "baselineMm": 50,
                  "projector": "IR dot pattern: depth on textureless "
                               "surfaces, which passive stereo could not do "
                               "here", "interface": "USB 3", "powerW": 2.0},
    feeds=(
      Feed("depth_eye.fovy", camera("depth_eye"), "°",
           "the near-field camera on the mast top, pitched 40° at the "
           "floor ahead", expect=58),
      Feed("depth_cam_body.mass", geom("depth_cam_body", "mass"), "kg",
           "the unit, on the mast top over the axle: CoM +10.7 mm, cruise "
           "launch pitch 0.9 -> 1.3°; 72 g, inside the datasheet's 75 ±10 %"),
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
      code("power.DEPTH_CAMERA_W", "W",
           "drawn only while the near-field map is built; ~1.9 W measured "
           "by users streaming depth with the projector, 3.5 W the USB "
           "budget, the datasheet figure is Parts.md's open decision 10"),
    ),

    note="Issue #34: the near-field sensor, so the robot can find things "
         "on the floor the scan plane looks over. Chosen over the D405 "
         "(7-50 cm, but PASSIVE stereo: fails on painted floors and matte "
         "printed modules exactly as the DIY pair did), a CSI time-of-flight "
         "module (the Pi 5's two CSI ports are the two cameras) and a second "
         "LIDAR tilted at the floor (a line, only while moving; cannot look "
         "at a thing from a standstill). Its draw is `power.DEPTH_CAMERA_W`, "
         "on the pack only while the map is built -- on where served, off in "
         "a test (docs/Parts.md \"near-field depth camera\").",
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
      Feed("module_plug_pin_l.size[0]",
           same(geom("module_plug_pin_l", "size", 0),
                geom("module_plug_pin_r", "size", 0)),
           "m", "half the 4.8 mm pin", expect=0.0024),
      Feed("module_plug_pin_l.fromto[1]", geom("module_plug_pin_l", "fromto", 1),
           "m", "half the 19 mm pin pitch", expect=0.0095),
      Feed("module_plug_barrel.size[0]", geom("module_plug_barrel", "size", 0),
           "m", "the module's barrel: ⚠ 35.5 mm across, against the real "
           "36.7"),
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
    ("body",), ("pluggybot",), partNumber="SC1112",
    source="https://www.berrybase.de/raspberry-pi-5-8gb-ram",
    dimensionsMm={"length": 85, "width": 56}, priceEur=184.90, quantity=1,
    capabilities={"csiPorts": 2},
    why={"massG": "Raspberry Pi publishes none; BerryBase's 65 g is packed"},
    note="The board alone at BerryBase (2026-09-28); the active cooler is "
         "SC1148. No accelerator HAT.",
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
    "chosen", ("body", "catalog", "build"),
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
    "electronics", "candidate", ("body", "catalog", "build"), ("module_lcd",),
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
    "built_rack_rail_stock", "Built-tool rail: a second, shorter rack of the "
    "same stock beside the first, for the tools the robot builds",
    "structure", "chosen", ("body",),
    ("rack",), dimensionsMm={"railLength": 710, "width": 740, "depth": 170,
                             "height": 550}, quantity=1,
    capabilities={"bays": 3, "pitchMm": 250},
    feeds=(
      code("rack.coupling.BUILT_RACK_HALF_W", "m", "half the rail", expect=0.355),
      code("rack.coupling.BUILT_RACK_Y", "m",
           "its centre in the first rack's frame, continuing the pitch past bay E"),
      code("rack.coupling.BUILT_STATION_YS", "m",
           "the three built-tool bays at 0.25 m pitch, in the first rack's "
           "frame; appended to, never reordered (bay-tag pairing is by index)"),
      Feed("rack_built_shelf.mass", geom("rack_built_shelf", "mass"), "kg", "the base"),
      Feed("rack_built_rail.mass", geom("rack_built_rail", "mass"), "kg"),
      Feed("rack_built_post_l.mass",
           same(geom("rack_built_post_l", "mass"), geom("rack_built_post_r", "mass")),
           "kg"),
    ),
    why={"partNumber": STOCK, "source": STOCK, "massG": "not weighed",
         "priceEur": STOCK},
    note="The five hand-built modules are permanent and hang on the first "
         "rack; a built tool hangs here and only here (issue #277). Same "
         "frame as the first rack, no charge bay, no rack tag: it is "
         "surveyed beside the first and found through it.",
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
    "bay_switch", "Bay presence switch: one Omron D2F-01L2 hinge-roller-lever "
    "microswitch in each tool bay's V-tray", "sensor", "chosen",
    ("body", "build"), ("rack", "rack_built"), partNumber="D2F-01L2",
    source="https://www.digikey.de/de/products/detail/omron-electronics-inc-"
    "emc-div/D2F-01L2/368444",
    massG=0.5, dimensionsMm={"length": 12.8, "width": 5.8, "height": 16.5},
    priceEur=2.59, quantity=8,
    capabilities={"sense": "contact", "operatingForceN": 0.78,
                  "releasingForceN": 0.05, "overtravelMm": 0.55,
                  "contactRating": "0.1 A 30 V DC", "powerW": 0},
    feeds=(
      Feed("bay*_tray_l_*", switched_bays(), "bays",
           "one switch in each bay's +y V (`coupling.BAY_SWITCH_PLATES`): "
           "five on the first rail, three on the built one", expect=8),
    ),
    note="Feeds `coupling.bay_switches`, all the rack reports over the "
         "network: which bays are occupied, never by which module (issue "
         "#351). The robot's `rack` context is built off these, its own "
         "fork and what the other robot says it carries, and nothing else. "
         "In the sim the switch is its V's two plates read off the contact "
         "list, as the bumper is. ⚠ Force is not modelled, and a bare "
         "switch under one tray would not close for every module: the LCD "
         "(143 g) and the plug (156 g) put 0.70 and 0.77 N on each tray "
         "against the 0.78 N operating force. The physical design needs the "
         "lever to carry the tray, or a lighter switch; open, not guessed.",
  ),
  Part(
    "rack_controller", "ESP32-class board on the rack: reads the bay "
    "switches and reports them over the network", "electronics",
    "candidate", ("body",), ("rack",), priceEur=5, quantity=1,
    why={"partNumber": "any ESP32 dev board; none chosen",
         "source": "no board chosen", "massG": "not recorded",
         "dimensionsMm": "not recorded"},
    note="The rack's only electronics besides the charger: eight inputs, "
         "one per `bay_switch`, and the radio that makes the rack a "
         "network fact rather than something a robot has to go and look "
         "at (issue #351). A candidate, like the charger board, until a "
         "board is chosen.",
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
    "(unspecified)", "actuator", "candidate", ("body",), ("module_pen",),
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
         "have. Its ±15 N is a sim guess with no datasheet behind it. Off "
         "the catalog shelf since issue #199: the workshop builds slides "
         "from `slide_l12_100`, a real part; this stays the pen's record. "
         "The pen's 110 mm travel is 10 mm more than that part's stroke, "
         "so the pen itself is not yet claimed to be built from it.",
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
    "servo_fs90mg", "FEETECH FS90MG micro servo (digital, metal gears)",
    "actuator", "chosen", ("catalog", "build"), (), partNumber="FS90MG",
    source="https://eckstein-shop.de/Feetech-FS90MG-6V-22kgcm-Digital-Servo",
    massG=12.7, dimensionsMm={"length": 22.5, "width": 12.1, "height": 26.7},
    priceEur=5.95,
    capabilities={"motion": "hinge", "angleDeg": 180, "torqueNm": 0.216,
                  "speedDegS": 857, "voltageV": [4.8, 6.0], "stallA": 0.8,
                  "powerW": 4.8, "gears": "copper"},
    note="The second servo size (issue #199): 2.2 kg·cm at 6 V is 0.216 "
         "N·m, half again the FS90-FB's 0.147, at the SAME 800 mA stall -- "
         "so it fits the peg's budget exactly as the FS90 does. Every "
         "number is Feetech's own FS90MG product specification (the PDF "
         "Feetech distributes through resellers, e.g. aifitlab.com's "
         "FS90MG-PRODUCT_SPECIFICATION.pdf): 0.07 s/60° no-load at 6 V is "
         "857°/s, 12.7 ± 1 g, 22.5 × 12.1 × 26.7 mm, 180° limit angle. "
         "Passed over: Tower Pro's MG90S (2.2 kg·cm, 13.4 g), whose maker "
         "publishes no current, and Power HD's HD-1810MG (3.9 kg·cm), "
         "whose 1.4 A stall at 6 V is 8.4 W -- over the peg on its own.",
  ),
  Part(
    "slide_l12_100", "Actuonix L12-100-50-6-R micro linear servo (100 mm "
    "stroke, 50:1, 6 V, RC input)", "actuator", "chosen", ("catalog", "build"),
    (), partNumber="L12-100-50-6-R",
    source="https://www.digikey.de/de/products/detail/actuonix-motion-devices-"
    "inc/L12-100-50-6-R/11689540",
    massG=56, dimensionsMm={"length": 152, "width": 15, "height": 14.9},
    priceEur=77.05,
    capabilities={"motion": "slide", "strokeMm": 100, "forceN": 22,
                  "speedMmS": 25, "peakPowerPoint": "17 N at 14 mm/s",
                  "backDriveN": 12, "voltageV": 6, "stallA": 0.46,
                  "powerW": 2.76, "dutyCycle": 0.2},
    note="The first slide the workshop can build from (issue #199). Every "
         "number is the L12 datasheet's (actuonix.com/assets/images/"
         "datasheets/ActuonixL12Datasheet.pdf): 22 N max force lifted and "
         "25 mm/s no-load at 50:1, 460 mA stall on the 6 V winding (2.76 "
         "W), 56 g and 152 mm hole-to-hole closed for the 100 mm stroke, "
         "a 15 × 14.9 mm body. 20 % duty cycle, which the sim does not "
         "model. `forceN` is the lifted maximum; at speed it is the 17 N "
         "peak-power point. ⚠ Passed over: the L16-140-35-6-R, the one "
         "stroke that covers the pen's 110 mm travel, because its datasheet "
         "gives stall current at 12 V only (650 mA) and the 6 V winding's "
         "is not published -- so its draw would be a guess.",
  ),
  Part(
    "esp32_cam", "Ai-Thinker ESP32-CAM (ESP32-S, OV2640 2 MP camera, Wi-Fi)",
    "sensor", "chosen", ("catalog",), (), partNumber="ESP32-CAM",
    source="https://www.berrybase.de/en/esp32-cam-development-board-incl.-"
    "ov2640-camera-module",
    massG=10, dimensionsMm={"length": 40.5, "width": 27, "height": 4.5},
    priceEur=8.60,
    capabilities={"sensor": "OV2640", "resolution": "1600 × 1200",
                  "fovDeg": None, "voltageV": 5, "powerW": 1.55,
                  "powerFlashOffW": 0.9, "radio": "Wi-Fi 802.11 b/g/n"},
    note="An eye a tool can carry (issue #199): the camera whose maker "
         "publishes its draw, and the one that fits the module design -- "
         "power-only coupling, wireless data -- because it IS the radio. "
         "Ai-Thinker's ESP32-CAM specification V1.0 gives 180 mA at 5 V "
         "with the flash off (0.9 W) and 310 mA with it at full brightness "
         "(1.55 W, the ceiling `powerW` carries, on a servo's stall terms), "
         "27 × 40.5 × 4.5 mm and 10 g. On a real module it would replace "
         "the module's own ESP32 rather than sit beside it; the sim sums "
         "both, the dearer case. The lens's field of view is not on the "
         "sheet (`fovDeg` null) -- the Pi's cameras keep their measured "
         "fovy and a built eye renders nothing yet. EU stock is marketplace "
         "clones of the same design; the price is BerryBase's.",
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
         "wirelessly, so a tool camera costs no CSI port on the Pi. The "
         "workshop's eye is `esp32_cam` (issue #199); this stays the claw "
         "module's record until a lens with a published field of view is "
         "chosen for it.",
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
  # ---- the build (#379): the quadruped and what builds it ------------------
  # Read against the quadruped's model (`robot="quadruped"`). A part both
  # robots carry has an entry per robot, each reading its own model, and
  # one price and one mass between them (tests/test_catalog.py).
  Part(
    "quad_rplidar_c1", "Slamtec RPLIDAR C1 2D LIDAR", "sensor", "chosen",
    ("build",), ("lidar_mast",), partNumber="RPLIDAR C1", source=FUNDUINO_C1,
    massG=110, priceEur=84.90, quantity=1, robot="quadruped",
    capabilities={"rangeM": 12, "rateHz": 10, "fovDeg": 360, "supplyV": 5,
                  "currentMa": 230, "startCurrentMa": 800,
                  "interface": "USB/UART"},
    feeds=(
      code_entry("legs.model.ELECTRONICS_W", "lidar", "W",
                 "230 mA typical at 5 V (Slamtec datasheet rev 1.2)",
                 expect=1.15),
      code("legs.model.MassBudget.scanner", "kg", "the unit's 110 g and its mast"),
      Feed("lidar.pos[2]", site("lidar", "pos", 2), "m",
           "the scan head on its mast: the scan plane 0.51 m up, over the "
           "couch and the bed"),
      code("perception.lidar.LIDAR_PERIOD", "s", "10 Hz, the part's rate",
           expect=0.1),
      code("perception.lidar.MAX_RANGE", "m",
           "8 m, UNDER the part's 12 m on purpose (the rover's entry says why)"),
    ),
    why={"dimensionsMm": "not recorded"},
    note="Funduino (DE) has it in stock; OpenELAB lists it at €79.95 with 4 "
         "left from a warehouse it does not name (it ships from China in "
         "10-20 days otherwise). It draws 800 mA for a moment at start-up.",
  ),
  Part(
    "quad_realsense_d435", "RealSense D435 depth camera (active IR stereo)",
    "sensor", "chosen", ("build",), ("depth_cam_body",), partNumber="D435",
    source=DIGIKEY_D435, massG=75,
    dimensionsMm={"length": 90, "width": 25, "height": 25}, priceEur=340.01,
    quantity=1, robot="quadruped",
    capabilities={"fovHDeg": 87, "fovVDeg": 58, "baselineMm": 50,
                  "interface": "USB 3", "maxPowerW": 3.4},
    feeds=(
      Feed("depth_eye.fovy", camera("depth_eye"), "°",
           "the depth field's vertical angle", expect=58),
      code_entry("legs.model.ELECTRONICS_W", "depth camera", "W",
                 "depth with its projector; the datasheet's 3.40 W is depth "
                 "and 1080p colour at once"),
      code("legs.model.MassBudget.depth_cam", "kg", "the unit's 75 g and a bracket"),
      code("perception.depth.MIN_Z", "m", "the datasheet's min-Z", expect=0.28),
      code("perception.depth.BASELINE", "m", expect=0.05),
    ),
    note="DigiKey DE's kit (961448, with cable and tripod), none in stock; "
         "MyBotShop (DE) €419.99 in 21 days; RealSense's own store $314, "
         "backordered 6-8 weeks, import duty on top. Cognex agreed to buy "
         "RealSense on 2026-09-22 (closing in Q4 2026): the bill's longest "
         "lead, and the one to watch.",
  ),
  Part(
    "quad_pi_camera_3", "Raspberry Pi Camera Module 3", "sensor", "chosen",
    ("build",), ("nav_cam_body",), partNumber="Camera Module 3",
    source=BERRYBASE_CAM3, massG=4,
    dimensionsMm={"length": 25, "width": 24, "height": 11.5}, priceEur=28.90,
    quantity=1, robot="quadruped",
    capabilities={"sensor": "IMX708", "fovHDeg": 66, "fovVDeg": 41},
    feeds=(
      Feed("nav_eye.fovy", camera("nav_eye"), "°",
           "the one navigation camera: the rack's and the dock's tags",
           expect=41),
      code("legs.model.MassBudget.cameras", "kg",
           "budgets two modules; the model carries one"),
    ),
    note="One on the body: the model has one navigation camera, where the "
         "rover had two. It ships with the old 15-pin cable; a Pi 5 needs "
         "`pi5_camera_cable`.",
  ),
  Part(
    "pi5_camera_cable", "Raspberry Pi camera cable, standard to mini, 200 mm",
    "electronics", "chosen", ("build",), ("nav_cam_body",), partNumber="SC1128",
    source="https://www.berrybase.de/raspberry-pi-camera-cable-standard-mini-200mm",
    priceEur=1.20, quantity=1, robot="quadruped",
    capabilities={"pins": "22 (Pi 5) to 15 (camera)", "lengthMm": 200},
    why={"massG": "only a shipping weight is given",
         "dimensionsMm": "a 200 mm flat cable"},
  ),
  Part(
    "tdk_ev_icm42688p", "TDK InvenSense ICM-42688-P evaluation board (IMU)",
    "sensor", "chosen", ("build",), ("pluggybot",), partNumber="EV_ICM-42688-P",
    source="https://www.digikey.de/de/products/detail/tdk-invensense/"
           "EV-ICM-42688-P/18634550",
    priceEur=36.79, quantity=1, robot="quadruped",
    capabilities={"interface": "I2C, I3C or SPI", "logicV": "1.8 or 3.0"},
    feeds=(
      code("perception.imu.GYRO_NOISE", "rad/s/√Hz",
           "0.0028 °/s/√Hz (DS-000347, tested in production)",
           expect=math.radians(0.0028)),
      code("perception.imu.ACCEL_NOISE", "m/s²/√Hz",
           "65 µg/√Hz on x and y, 70 on z",
           expect=[65e-6 * 9.80665, 65e-6 * 9.80665, 70e-6 * 9.80665]),
      code("perception.imu.GYRO_SCALE", "1", "±0.5 % initial tolerance",
           expect=0.005),
    ),
    why={"massG": "not published for the board",
         "dimensionsMm": "not published for the board"},
    note="TDK's own board, a 2×10 header, used with jump wires or soldered "
         "(TDK AN-000488): the one ICM-42688-P board in stock at an EU "
         "storefront; the supply of the chip is short (a Tindie maker "
         "raised its price for it).",
  ),
  # ---- actuation and compute -----------------------------------------------
  Part(
    "gim8108_8", "Steadywin GIM8108-8 joint motor with the GDS68 driver "
    "(8:1, FOC, CAN, MIT mode), 48 V", "motor", "chosen", ("build",),
    ("pluggybot",), partNumber="SW-GIM8108-8-S68",
    source="https://openelab.io/products/steadywin-gim8108-8-robot-motor?"
           "variant=46976063668422",
    massG=396, dimensionsMm={"diameter": 97, "length": 55}, priceEur=124.95,
    quantity=14, robot="quadruped",
    capabilities={"ratedTorqueNm": 7.5, "peakTorqueNm": 22, "ratedRpm": 110,
                  "maxRpm": 320, "ratio": 8, "busV": "12-56",
                  "connector": "XT30(2+2): power and CAN in one"},
    feeds=(
      Feed("FL_hip_abd.forcerange", legs(actuator, "forcerange"), "N·m",
           "the peak torque, all twelve leg drivers alike", expect=22),
      Feed("FL_hip_abd.armature", legs(joint, "armature"), "kg·m²",
           "the rotor's inertia x 8²: the class's nominal, unpublished"),
      code("legs.actuator.GIM8108_8.rated_torque", "N·m",
           "the lower of the maker's two tables (this seller's says 7.5)"),
      code("legs.actuator.GIM8108_8.peak_torque", "N·m", expect=22),
      code("legs.actuator.GIM8108_8.noload_speed", "rad/s",
           "at 48 V, off the maker's curve"),
      code("legs.actuator.GIM8108_8.mass", "kg", "with the driver",
           expect=0.396),
      code("legs.actuator.GIM8108_8.price_eur", "€",
           "the price the sizing tables were drawn at", expect=124.95),
    ),
    note="Twelve on the legs and two on the arm (#378): one part, one driver, "
         "one set of gains. €124.95 is a sale on €134.95, and on pre-order: "
         "none in OpenELAB's Munich or US warehouse, \"restock in 10-20 "
         "days\", no count, so sixteen go through its quote form. Steadywin's "
         "own store is $129.20 plus $185 of express to Germany for fourteen, "
         "12-26 days, VAT and duty unstated. No cable comes with it; power "
         "and CAN share one XT30(2+2) (manual rev 1.4).",
  ),
  Part(
    "amass_xt30_22_pair", "Amass XT30(2+2) connector pair: power and signal "
    "in one (male, female and pins)", "electronics", "chosen", ("build",),
    ("pluggybot",), partNumber="XT30(2+2)",
    source="https://www.3dptronics.com/electronics/xt3022-connectors-pair",
    priceEur=5.00, priceFor="pair", quantity=14, robot="quadruped",
    capabilities={"wiring": "16, 18 or 20 AWG (Amass)",
                  "ratedA": "15 with 18 AWG (Amass)"},
    why={"massG": "not stated", "dimensionsMm": "not recorded"},
    note="The female mates each GDS68's power-and-CAN socket. No German "
         "shop stocks it: 3DPTronics (Italy) sells pairs only, its cart "
         "says VAT in; LCSC has the female alone (C19268028) at $0.38. "
         "Amass lists 16-20 AWG for its cups, not 14.",
  ),
  Part(
    "lapp_unitronic_can", "LAPP UNITRONIC BUS CAN 1 x 2 x 0.22 mm², "
    "shielded, by the metre", "electronics", "chosen", ("build",),
    ("pluggybot",), partNumber="2170260/1",
    source="https://www.conrad.de/de/p/lapp-2170260-1-busleitung-unitronic-"
           "bus-1-x-2-x-0-22-mm-violett-meterware-603992.html",
    priceEur=1.98, quantity=10, robot="quadruped",
    capabilities={"impedanceOhm": 120, "outerDiameterMm": 5.7},
    why={"massG": "not stated", "dimensionsMm": "5.7 mm across, by the metre"},
    note="The four buses. Rated for fixed installation, and a leg flexes "
         "it; LAPP's flexible CAN FD P is sold at Conrad only by the 100 m "
         "to businesses.",
  ),
  Part(
    "silicone_wire_14awg", "Silicone wire, 14 AWG (2.08 mm²), 1 m",
    "power", "chosen", ("build",), ("pluggybot",), partNumber="SK14S / SK14R",
    source="https://www.mylipo.de/Silikonkabel-14AWG-208mm-schwarz_1",
    priceEur=2.25, quantity=20, robot="quadruped",
    why={"massG": "not stated per metre", "dimensionsMm": "a metre of 2.08 mm²"},
    note="Each driver's drop from the 10 AWG bus, red and black. Amass lists "
         "16-20 AWG for the XT30(2+2)'s cups: step down to 16 at the plug.",
  ),
  Part(
    "raspberry_pi_5_quad", "Raspberry Pi 5, 8 GB", "electronics", "chosen",
    ("build",), ("pluggybot",), partNumber="SC1112",
    source="https://www.berrybase.de/raspberry-pi-5-8gb-ram",
    dimensionsMm={"length": 85, "width": 56}, priceEur=184.90, quantity=1,
    robot="quadruped", capabilities={"csiPorts": 2, "power": "5 V 5 A"},
    feeds=(
      code_entry("legs.model.ELECTRONICS_W", "compute + cameras", "W",
                 "the rover's figure less its LIDAR; Raspberry Pi publishes "
                 "no load figure"),
    ),
    why={"massG": "Raspberry Pi publishes none; BerryBase's 65 g is packed"},
    note="The rover's board. The 16 GB (SC1113) is €319.20 in stock, if "
         "#379's budget for the computer asks for it.",
  ),
  Part(
    "pi5_active_cooler", "Raspberry Pi Active Cooler for the Pi 5",
    "electronics", "chosen", ("build",), ("pluggybot",), partNumber="SC1148",
    source="https://www.berrybase.de/raspberry-pi-active-cooler-luefter-fuer-"
           "raspberry-pi-5",
    priceEur=5.90, quantity=1, robot="quadruped",
    why={"massG": "only a packed weight (29 g) is given",
         "dimensionsMm": "the Pi 5's footprint"},
  ),
  Part(
    "pi_ssd_kit_256", "Raspberry Pi SSD Kit, 256 GB (M.2 HAT+ and NVMe)",
    "electronics", "chosen", ("build",), ("pluggybot",), partNumber="SC1675",
    source="https://www.berrybase.de/raspberry-pi-ssd-kit-fuer-raspberry-pi-"
           "5-256gb",
    priceEur=54.90, quantity=1, robot="quadruped",
    capabilities={"capacityGb": 256, "iops": "40k read, 70k write (4 kB)"},
    why={"massG": "only a packed weight (35 g) is given",
         "dimensionsMm": "a HAT on the Pi 5's footprint"},
    note="Out of stock at every EU seller read; the 512 GB kit (SC1676) is "
         "€166.10 in stock at Reichelt, and the M.2 HAT+ alone (SC1166) "
         "€12.50 at BerryBase.",
  ),
  Part(
    "waveshare_can_hat_plus", "Waveshare 2-CH CAN HAT+ (MCP2515, isolated) "
    "for the Raspberry Pi", "electronics", "chosen", ("build",),
    ("pluggybot",), partNumber="27338",
    source="https://eckstein-shop.de/WaveShare-2CH-Isolated-CAN-Bus-"
           "Expansion-HAT-for-Raspberry-Pi",
    massG=45, priceEur=29.95, quantity=1, robot="quadruped",
    capabilities={"channels": 2, "controller": "MCP2515 (CAN 2.0)",
                  "bitrate": "1 Mbit/s (its wiki's example)", "pi5": "listed"},
    why={"dimensionsMm": "a HAT on the Pi's footprint"},
    note="Two of the four buses, one a pair of legs. Its wiki lists the Pi "
         "5 and documents the stack with the CAN FD HAT: four channels.",
  ),
  Part(
    "waveshare_can_fd_hat", "Waveshare 2-CH CAN FD HAT (MCP2518FD, "
    "isolated) for the Raspberry Pi", "electronics", "chosen", ("build",),
    ("pluggybot",), partNumber="17075",
    source="https://www.welectron.com/Waveshare-17075-2-CH-CAN-FD-HAT_1",
    massG=40, priceEur=51.90, quantity=1, robot="quadruped",
    capabilities={"channels": 2, "controller": "MCP2518FD (CAN FD)",
                  "isolationKv": 5},
    why={"dimensionsMm": "a HAT on the Pi's footprint"},
    note="The other two buses: the arm's and a spare. Its own wiki lists "
         "Pis up to the 4B; the Pi 5 stack is the HAT+'s wiki's. Four USB "
         "adapters (`ucan_v1`, €8.99) are the fallback.",
  ),
  Part(
    "ucan_v1", "UCAN V1.0 USB-CAN adapter (candleLight firmware)",
    "electronics", "chosen", ("build",), (), partNumber="OP00014",
    source="https://eckstein-shop.de/ucan-v1-usb-can-adapter",
    priceEur=8.99, quantity=1, robot="quadruped",
    capabilities={"firmware": "candleLight (Linux gs_usb, socketcan)"},
    why={"massG": "not on the page", "dimensionsMm": "not on the page"},
    note="The bench leg's bus from a PC. The page gives no bitrate; "
         "candleLight runs a bus at 1 Mbit/s. PEAK's PCAN-USB is €199 ex VAT "
         "and sold to businesses only.",
  ),
  Part(
    "pi5_psu_27w", "Raspberry Pi 27 W USB-C power supply (EU)",
    "power", "chosen", ("build",), (), partNumber="SC1152",
    source="https://www.berrybase.de/raspberry-pi-27w-usb-c-power-supply-"
           "netzteil-weiss",
    priceEur=12.40, quantity=1, robot="quadruped",
    capabilities={"output": "5.1 V 5 A (PD)"},
    why={"massG": "only a packed weight (60 g) is given",
         "dimensionsMm": "not recorded"},
    note="The Pi on the bench, before the robot's 5 V rail exists.",
  ),
  # ---- power ---------------------------------------------------------------
  Part(
    "molicel_p45b", "Molicel INR21700-P45B Li-ion cell (21700, 4.5 Ah)",
    "power", "chosen", ("build",), ("pluggybot",), partNumber="INR21700-P45B",
    source="https://www.akkuteile.de/en/lithium-ionen-battery/size-21700/"
           "molicel/molicel-inr21700-p45b-4500mah-li-ion-battery-3-6v-3-7v_"
           "100852_3119",
    massG=70, priceEur=7.00, quantity=12, robot="quadruped",
    capabilities={"capacityAh": 4.5, "nominalV": 3.6, "chargeV": 4.2,
                  "continuousA": 45},
    feeds=(
      code("legs.model.PACK_WH", "Wh", "twelve in series, 3.6 V x 4.5 Ah",
           expect=194.4),
      code("legs.actuator.BUS_V_NOMINAL", "V", "12 x 3.6", expect=43.2),
      code("legs.actuator.BUS_V_RANGE", "V",
           "12 x 3.0 to 12 x 4.2: the top is the cell's charge voltage, the "
           "bottom the sim's cut-off"),
      code("legs.model.MassBudget.battery", "kg",
           "the pack: twelve 70 g cells, the BMS, a case and leads"),
    ),
    why={"dimensionsMm": "the 21700 format; the sheet's figures not read"},
    note="€7.00 each for eight or more (€7.90 for one to three) at "
         "akkuteile.de, the same SKU at LiPo24. No German seller publishes "
         "a 12S1P built to order: Enerprof's smallest 12S Molicel pack is a "
         "12S3P at €860 in 14-28 days, the rest quote on request, so the pack "
         "is built here (`spot_welder`).",
  ),
  Part(
    "jbd_bms_12s_120a", "JBD smart BMS, 10S-17S, 120 A, Bluetooth",
    "power", "chosen", ("build",), ("pluggybot",),
    partNumber="10S-17S-120A-BT-UART",
    source="https://bikebattery.de/JBD-10S-17S-120A-360A-Intelligentes-Smart-"
           "Bluetooth-BMS-RS485-Android-iOS-APP-11S-12S-13S-14S-15S-16S",
    massG=100, dimensionsMm={"length": 105, "width": 55, "height": 12},
    priceEur=112.99, quantity=1, robot="quadruped",
    capabilities={"seriesCells": "10-17", "continuousA": 120,
                  "balancing": "active; current not stated"},
    note="The seller's 60 A and 80 A versions are sold out; the JK "
         "JK-BD6A20S6P (60 A, 0.6 A active balancing) is €60.82 ex VAT from "
         "JK's own store in China with no delivery time stated. Mass is the "
         "seller's article weight.",
  ),
  Part(
    "noeifevo_charger_12s", "NOEIFEVO 50.4 V 5 A Li-ion charger (12S, CC/CV)",
    "power", "chosen", ("build",), (), partNumber="HRH-300L-50405-XT60",
    source="https://www.noeifevo.de/products/noeifevo-50-4v-5a-lithium-"
           "ladegerat-fur-12s-44-4v-li-ionen-lipo-akku-e-bike-roller-ladegerat-"
           "led-anzeige-aluminiumgehause",
    priceEur=69.00, quantity=1, robot="quadruped",
    capabilities={"outputV": 50.4, "outputA": 5, "inputV": "220-240 AC"},
    feeds=(
      code("legs.dock.CHARGE_A", "A", "the charger's CC current", expect=5.0),
      code("legs.dock.CHARGE_W", "W", "5 A at the pack's nominal 43.2 V"),
    ),
    why={"massG": "not published", "dimensionsMm": "not published"},
    note="The dock's charger. The page states no VAT and no warehouse: its "
         "policy is 3-7 working days to Germany from Poland, 7-15 from "
         "China, and a sold-out label sits beside the add-to-cart. "
         "BOUNDMOTOR's (Parts.md's first pick) is $99 plus import duty, 12-20 "
         "days by air, with a barrel plug needing an adapter.",
  ),
  Part(
    "meanwell_ddr_60l_5", "Mean Well DDR-60L-5 DC-DC converter, 18-75 V in, "
    "5 V 12 A, isolated (DIN rail)", "power", "chosen", ("build",),
    ("pluggybot",), partNumber="DDR-60L-5",
    source="https://www.reichelt.com/de/en/dc-dc-converter-60w-5v-12a-ddr-"
           "60l-5-p256746.html",
    massG=216, priceEur=43.80, quantity=1, robot="quadruped",
    capabilities={"inputV": "18-75", "outputV": "5 (4.5-5.5 adjustable)",
                  "outputA": 12, "isolationKv": 4},
    feeds=(
      code_entry("legs.model.ELECTRONICS_W", "compute + cameras", "W",
                 "the Pi 5, its camera and the IMU on the 5 V rail"),
    ),
    why={"dimensionsMm": "not recorded"},
    note="Set to 5.1 V for the Pi 5. The DDR-60G is 9-36 V in and would not "
         "take the pack.",
  ),
  Part(
    "amass_xt90s", "Amass XT90-S anti-spark connector pair", "power",
    "chosen", ("build",), (), partNumber="XT90S",
    source="https://www.mhm-modellbau.de/part-AM-XT90S.php",
    priceEur=4.05, priceFor="pair", quantity=1, robot="quadruped",
    why={"massG": "not on the page", "dimensionsMm": "not on the page"},
    note="The pack's lead, the charger's and the bench supply's: the "
         "anti-spark half takes the drivers' inrush as it mates.",
  ),
  Part(
    "victron_midi_60a", "Victron MIDI fuse 60 A / 58 V", "power", "chosen",
    ("build",), (), partNumber="CIP133060010",
    source="https://www.klimaworld.com/products/victron-midi-fuse-"
           "sicherungen-58v-48v-60a",
    priceEur=14.29, quantity=1, robot="quadruped",
    capabilities={"currentA": 60, "voltageV": 58, "dropMv": 70},
    why={"massG": "only a shipping weight is given",
         "dimensionsMm": "not recorded"},
    note="The pack's main fuse. Victron's sheet does not state its breaking "
         "capacity.",
  ),
  Part(
    "victron_midi_holder", "Victron MIDI fuse holder", "power", "chosen",
    ("build",), (), partNumber="CIP000050001",
    source="https://www.offgridtec.com/victron-midi-fuse-sicherung-halter.html",
    priceEur=10.12, quantity=1, robot="quadruped",
    capabilities={"currentA": 200, "maxV": 58},
    why={"massG": "the page gives 0.2 with no unit",
         "dimensionsMm": "not recorded"},
  ),
  Part(
    "silicone_wire_10awg", "Silicone wire, 10 AWG (6 mm²), 1 m", "power",
    "chosen", ("build",), (), partNumber="2392 (red), 2393 (black)",
    source="https://www.modellbau-skeries.de/p/silikonkabel-10awg-6mm-x-"
           "1000mm-rot",
    priceEur=4.40, quantity=1, robot="quadruped",
    why={"massG": "not stated per metre", "dimensionsMm": "a metre of 6 mm²"},
    note="The bus from the pack through the fuse and the contactor to the "
         "legs' drivers.",
  ),
  # ---- safety --------------------------------------------------------------
  Part(
    "albright_sw80b549", "Albright SW80B contactor, 12 V coil, with blowouts "
    "(100 A, 96 V DC)", "power", "chosen", ("build",), (),
    partNumber="SW80B549",
    source="https://www.rijregeling.nl/en/product/albright-contactors-and-"
           "accessories/sw80-series/sw80b549-albright-contactor-12v-"
           "contactor/",
    massG=400, priceEur=82.28, quantity=1, robot="quadruped",
    capabilities={"coilV": 12, "continuousA": 100, "maxV": 96,
                  "breaksA": "190 A resistive at 96 V"},
    why={"dimensionsMm": "not recorded"},
    note="The e-stop's switch: the motor bus opens when the button on the "
         "robot or the wireless relay breaks its coil circuit. The B "
         "version, with blowouts: the plain SW80 is rated to break load "
         "only to 48 V, and the pack charges to 50.4. ⚠ Confirm the coil is "
         "CONTINUOUS-rated before ordering: Albright's sheet rates SW80 "
         "coils by class (continuous 7-13 W down to 25 % duty at 20-30 W) "
         "and neither it nor the seller says which the 549 is; ARC "
         "Components lists the SW80B-5 and -35 as 12 V continuous, at £51.05 "
         "ex VAT and 7-9 weeks. €68.00 plus 21 % Dutch VAT as the page shows "
         "it; about 350 g and 50 g of blowouts (Albright's sheet). TE's "
         "Kilovac LEV100 (190 g) sells to businesses only, at €127.44 ex "
         "VAT.",
  ),
  Part(
    "schneider_xb5as8442", "Schneider Harmony XB5AS8442 emergency stop, "
    "40 mm twist-release head, 1 NC", "electronics", "chosen", ("build",), (),
    partNumber="XB5AS8442",
    source="https://www.reichelt.com/de/en/shop/product/emergency_stop_"
           "switch_harmony_xb5_twist_unlock_22_mm_1_nc-382385",
    massG=61, priceEur=30.95, quantity=1, robot="quadruped",
    capabilities={"contact": "1 NC", "mountMm": 22, "ip": "IP66"},
    why={"dimensionsMm": "a 40 mm head on a 22 mm hole"},
    note="One on the robot, one on the bench rig. Eaton's M22 in three "
         "parts (head, adapter, NC contact) is €34.42 and in stock.",
  ),
  Part(
    "radiomaster_pocket", "RadioMaster Pocket ELRS transmitter (EU LBT)",
    "electronics", "chosen", ("build",), (), partNumber="Pocket ELRS EU-LBT",
    source="https://n-factory.de/RadioMaster-Pocket-ELRS-Remote-Control-"
           "EU-LBT",
    massG=288, priceEur=79.95, quantity=1, robot="quadruped",
    why={"dimensionsMm": "not published"},
    note="The wireless e-stop's hand unit: a switch on it holds the "
         "receiver's relay closed. It runs on two 18650 cells, not "
         "included. A commercial wireless e-stop (Tyro Indus 1S) is €725 ex "
         "VAT, the receiver not stated.",
  ),
  Part(
    "radiomaster_er6", "RadioMaster ER6 ELRS PWM receiver (LBT)",
    "electronics", "chosen", ("build",), ("pluggybot",),
    partNumber="ER6 ELRS LBT",
    source="https://n-factory.de/RadioMaster-ER6-ELRS-LBT-24-GHz-PWM-Receiver",
    massG=14.5, priceEur=31.95, quantity=1, robot="quadruped",
    capabilities={"failsafe": "at link quality 0 or 1 s without a packet, "
                              "per channel 988-2012 µs"},
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "pololu_rc_relay", "Pololu RC Switch with Relay (assembled)",
    "electronics", "chosen", ("build",), ("pluggybot",), partNumber="2804",
    source="https://eckstein-shop.de/Pololu-RC-Switch-with-Relay-Assembled-EN",
    massG=17, priceEur=21.36, quantity=1, robot="quadruped",
    capabilities={"relayA": 10, "threshold": "on above about 1700 µs"},
    why={"dimensionsMm": "not recorded"},
    note="In the contactor's coil circuit, in series with the e-stop "
         "button: the receiver's failsafe at 988 µs opens it, and Pololu's "
         "safe-start needs off-then-on to close it again.",
  ),
  Part(
    "meanwell_ddr_30l_12", "Mean Well DDR-30L-12 DC-DC converter, 18-75 V "
    "in, 12 V 2.5 A, isolated (DIN rail)", "power", "chosen", ("build",),
    ("pluggybot",), partNumber="DDR-30L-12",
    source="https://www.reichelt.com/de/en/shop/product/dc_dc_converter_30w_"
           "12v_2_5a-256735",
    massG=120, priceEur=27.10, quantity=1, robot="quadruped",
    capabilities={"inputV": "18-75", "outputV": "12 (9-13.2 adjustable)",
                  "outputA": 2.5},
    why={"dimensionsMm": "not recorded"},
    note="The contactor's coil. 30 W covers every class of SW80 coil; the "
         "15 W DDR-15L-12 (€23.30) holds only a continuous one.",
  ),
  Part(
    "molicel_p28a", "Molicel INR18650-P28A Li-ion cell (18650, flat top)",
    "power", "chosen", ("build",), (), partNumber="INR18650-P28A",
    source="https://www.lipo24.de/products/molicel-inr18650-p28a-2800mah-"
           "35a-lithium-ionen-akku-3-6v-3-7v",
    massG=48, dimensionsMm={"diameter": 18.45, "length": 65}, priceEur=5.95,
    quantity=2, robot="quadruped",
    note="The wireless e-stop's transmitter runs on two.",
  ),
  Part(
    "arcol_hs25_47r", "Arcol HS25 wirewound resistor, 47 Ω, 25 W, aluminium "
    "housing", "power", "chosen", ("build",), ("pluggybot",),
    partNumber="HS25 47R F",
    source="https://www.reichelt.com/de/en/shop/product/wirewound_resistor_"
           "axial_25_w_47_ohm_1_-233471",
    massG=14, priceEur=2.99, quantity=1, robot="quadruped",
    capabilities={"ohms": 47, "watts": "25 on a heatsink, 12.5 free"},
    why={"dimensionsMm": "not recorded"},
    note="Pre-charge: the drivers' capacitors fill through it, about 1 A at "
         "50.4 V, before the contactor closes; the sheet gives no pulse "
         "rating, only continuous power.",
  ),
  Part(
    "hongfa_hf115f_012", "Hongfa HF115F relay, 12 V coil, 16 A changeover",
    "power", "chosen", ("build",), ("pluggybot",),
    partNumber="HF115F-012-1ZS3A",
    source="https://www.reichelt.com/de/en/shop/product/miniature_power_"
           "relay_12_v_dc_16_a_1_changeover_contact-260113",
    massG=13, priceEur=3.10, quantity=1, robot="quadruped",
    capabilities={"coil": "12 V, 360 Ω", "maxV": "300 V DC"},
    why={"dimensionsMm": "not recorded"},
    note="Switches the pre-charge resistor in, then out once the contactor "
         "has taken the bus; what it breaks at 60 V DC is a curve on the "
         "sheet, not a number.",
  ),
  # ---- frame, legs, arm ----------------------------------------------------
  Part(
    "quad_frame_plates", "The torso's frame: two 3 mm aluminium side plates, "
    "end caps and cross members", "structure", "chosen", ("build",),
    ("pluggybot",), robot="quadruped",
    feeds=(
      code("legs.model.MassBudget.frame", "kg", "estimated"),
      code("legs.model.CHOSEN.torso", "m", "the torso's half-extents"),
    ),
    why={"partNumber": "no design yet", "source": "a laser cutter, once drawn",
         "massG": "estimated at 1.0 kg (`MassBudget.frame`), no design yet",
         "dimensionsMm": "side plates 400 x 110 mm; the rest undrawn",
         "priceEur": "no design yet: priced from its drawing"},
  ),
  Part(
    "cf_tube_25_22", "Carbon fibre tube, roll-wrapped, 25 mm OD / 22 mm ID, "
    "1 m (Easy Composites)", "structure", "chosen", ("build",),
    ("FL_thigh", "FL_shank"), partNumber="CFT-WF-25-22-1",
    source="https://www.easycomposites.eu/25mm-23mm-woven-finish-carbon-"
           "fibre-tube",
    massG=168.5, dimensionsMm={"outer": 25, "inner": 22, "length": 1000},
    priceEur=46.05, quantity=2, robot="quadruped",
    capabilities={"wallMm": 1.5, "tensileModulusGpa": 64},
    feeds=(
      code("legs.model.CHOSEN.thigh", "m", "four thighs"),
      code("legs.model.CHOSEN.shank", "m", "and four shanks: 1.68 m of tube"),
      code("legs.model.CHOSEN.thigh_mass", "kg",
           "the link with its clamps, estimated"),
      code("legs.model.CHOSEN.shank_mass", "kg", "estimated"),
    ),
    note="The 1.5 mm wall, for a leg; the 1 mm 25/23 (€37.72, 113 g a "
         "metre) is the arm's class. The page's variant list sits under the "
         "25/23's address.",
  ),
  Part(
    "quad_leg_brackets", "The legs' motor mounts, hip brackets, knee housings "
    "and tube clamps, with their fasteners", "structure", "chosen", ("build",),
    ("FL_hip", "FL_thigh"), robot="quadruped",
    feeds=(
      code("legs.model.CHOSEN.hip_out", "m",
           "abduction axis to the leg's plane: half the motor stack and a "
           "bracket"),
    ),
    why={"partNumber": "no design yet", "source": "a job shop, once drawn",
         "massG": "in `thigh_mass` and the frame's budget, estimated",
         "dimensionsMm": "no design yet",
         "priceEur": "no design yet: priced from its drawings"},
  ),
  Part(
    "squash_balls_12", "Dunlop Pro squash balls, box of 12 (40 mm, 24 g each)",
    "structure", "chosen", ("build",), ("FL_shank",), partNumber="700108",
    source="https://www.squashpoint.de/dunlop-pro-squashball-12er-box.html",
    massG=288, dimensionsMm={"diameter": 40}, priceEur=37.50, quantity=1,
    robot="quadruped",
    feeds=(
      code("legs.model.CHOSEN.foot_r", "m",
           "the model's foot is 44 mm across; a squash ball is 40"),
    ),
    note="The feet, four and eight spares. What hobby builders use; whether "
         "one holds up under a 10 kg trot is on no page, and the box is the "
         "cheap way to find out. A sale price (was €51.00).",
  ),
  Part(
    "cf_tube_20_18", "Carbon fibre tube, roll-wrapped, 20 mm OD / 18 mm ID, "
    "1 m (Easy Composites)", "structure", "chosen", ("build",), (),
    partNumber="CFT-WF-20-18-1",
    source="https://www.easycomposites.eu/20mm-woven-finish-carbon-fibre-tube",
    massG=86.5, dimensionsMm={"outer": 20, "inner": 18, "length": 1000},
    priceEur=27.91, quantity=1, robot="quadruped",
    capabilities={"wallMm": 1, "tensileModulusGpa": 64},
    feeds=(
      code("legs.arm.ArmSpec.upper", "m", "the upper arm"),
      code("legs.arm.ArmSpec.fore", "m", "and the forearm, 0.60 m of tube"),
      code("legs.arm.TUBE_R", "m",
           "the capsule the sim collides: the tube and its rod ends"),
    ),
    note="€23.45 a metre ex VAT, as Parts.md had it (#378).",
  ),
  Part(
    "skf_61800_2rs1", "SKF 61800-2RS1 deep-groove ball bearing, 10 x 19 x 5 mm",
    "structure", "chosen", ("build",), (), partNumber="61800-2RS1",
    source="https://www.kugellager-shop.net/61800-2rs1-skf-rillenkugellager-"
           "10x19x5mm.html",
    massG=4.7, dimensionsMm={"bore": 10, "outer": 19, "width": 5},
    priceEur=7.60, quantity=4, robot="quadruped",
    capabilities={"dynamicKn": 1.72, "staticKn": 0.83},
    note="The elbow's and the wrist's pivots, two each. €6.84 from ten; the "
         "mass is another seller's page (SKF's did not load).",
  ),
  Part(
    "igus_kcrm_05", "igus igubal KCRM-05 rod end, M5 female, right-hand",
    "structure", "chosen", ("build",), (), partNumber="KCRM-05",
    source="https://www.igus.de/product/igubal_KCRM_KCLM?artNr=KCRM-05",
    massG=3.1, priceEur=5.43, quantity=6, robot="quadruped",
    capabilities={"thread": "M5 x 0.8", "boreMm": 5,
                  "staticRadialN": "1200 short-term, 600 long-term"},
    feeds=(
      code("legs.arm.ArmSpec.rods_mass", "kg",
           "the three rods and their six ends, estimated"),
    ),
    why={"dimensionsMm": "not recorded"},
    note="The parallelogram's rods' ends. igus shows €4.56 ex VAT; €4.37 at "
         "ten. A steel PHSA5 is €2.87 at Dold, and 20 g.",
  ),
  Part(
    "quad_arm_fork", "The arm's end plate, fork and lean-pad (two prongs, "
    "four 60° V's, two 53° end-ramps), and the parallelogram's three rods",
    "structure", "chosen", ("build",), (), robot="quadruped",
    feeds=(
      code("legs.arm.ArmSpec.plate_mass", "kg", "estimated"),
      code("legs.arm.ForkSpec.flank_deg", "°", "the V's flanks, for the stairs"),
      code("legs.arm.ForkSpec.fork_y", "m", "the V's at ±85 mm"),
      code("legs.arm.ForkSpec.v_half_len", "m", "the flanks 31 mm long"),
    ),
    why={"partNumber": "no design yet", "source": "a job shop, once drawn",
         "massG": "estimated at 0.08 kg (`ArmSpec.plate_mass`)",
         "dimensionsMm": "no design yet",
         "priceEur": "no design yet: priced from its drawing"},
  ),
  Part(
    "ptfe_tape_glass", "PTFE glass-fabric adhesive tape, 0.13 mm, 25 mm x 30 m",
    "structure", "chosen", ("build",), (),
    partNumber="PTFE Klebeband 0.13 SW",
    source="https://shop.hightechflon.com/PTFE-Teflon-Klebeband-0-13%20mm-"
           "ptfe-glasgewebe/Page-16-1-96-183.aspx",
    priceEur=29.67, quantity=1, robot="quadruped",
    feeds=(code("legs.arm.RAMP_MU", "1", "the end-ramps' slippery face"),),
    why={"massG": "255 g/m² on the page; the roll is not weighed",
         "dimensionsMm": "25 mm x 30 m, 0.13 mm thick"},
    note="The ramps' face. The smoother pure-PTFE film is €76.42 for the "
         "same roll; acetal inserts machined into the fork are the other "
         "way.",
  ),
  # ---- tools, rack, dock ---------------------------------------------------
  Part(
    "quad_tool_peg", "Tool peg, 220 mm: two 6 mm steel conductors on an "
    "insulating bush", "structure", "chosen", ("build",), (),
    robot="quadruped",
    feeds=(
      code("legs.rack.PEG_HALF", "m", "half its 220 mm"),
      code("legs.rack.PEG_MASS", "kg", "the rover's grams a millimetre"),
    ),
    why={"partNumber": "no design yet", "source": "turned in the shop, once drawn",
         "massG": "29 g, the rover's peg's grams a millimetre (`PEG_MASS`)",
         "dimensionsMm": "6 x 220 mm", "priceEur": "no design yet"},
  ),
  Part(
    "dold_3030_1m", "Aluminium extrusion 30x30 light, B-type slot 8, cut to "
    "length (1 m)", "structure", "chosen", ("build",), (),
    partNumber="67700-Z",
    source="https://www.dold-mechatronik.de/Aluminum-Profile-30x30L-B-Type-"
           "Groove-8-084kg-m-Customized-Cutting-50-to-6000mm",
    massG=843, dimensionsMm={"width": 30, "height": 30, "length": 1000},
    priceEur=10.40, quantity=1, robot="quadruped",
    feeds=(
      code("legs.rack.RackSpec.peg_z", "m", "the rack's pegs 0.50 m up"),
      code("legs.rack.RackSpec.rail_z", "m", "its rail"),
    ),
    note="€9.90 a metre and €0.50 a cut, so a metre in one piece is €10.40.",
  ),
  Part(
    "dold_angle_30", "Angle bracket 30, B-type slot 8, with its screws and "
    "hammer nuts", "structure", "chosen", ("build",), (),
    partNumber="66918-BSA",
    source="https://www.dold-mechatronik.de/Angle-30-B-type-groove-8-with-"
           "mounting-kit-and-cap",
    massG=60, priceEur=1.79, quantity=1, robot="quadruped",
    why={"dimensionsMm": "not recorded"},
    note="€1.79 from ten, €2.20 singly: the bill buys fourteen.",
  ),
  Part(
    "quad_rack_board", "The rack's back board, its six printed V-trays, and "
    "the rack's and the dock's printed tags", "structure", "chosen",
    ("build",), (), robot="quadruped",
    feeds=(
      code("legs.rack.TRAY_Y", "m", "the trays at ±45 mm"),
      code("legs.rack.RACK_TAG_IDS", "id", "tags 29-34, a pair a bay"),
      code("legs.dock.DockSpec.board_x", "m", "the dock's tag board"),
    ),
    why={"partNumber": "no design yet", "source": "a board and the shop's PETG",
         "massG": "no design yet", "dimensionsMm": "about 1.0 x 0.6 m",
         "priceEur": "no design yet"},
  ),
  Part(
    "millmax_0858", "Mill-Max 0858-0-15-20-82-14-11-0 spring-loaded pin "
    "(gold, 1.27 mm plunger, 2.29 mm stroke)", "electronics", "chosen",
    ("build",), (), partNumber="0858-0-15-20-82-14-11-0",
    source="https://www.digikey.de/de/products/detail/mill-max-manufacturing-"
           "corp/0858-0-15-20-82-14-11-0/7667985",
    priceEur=2.12, quantity=4, robot="quadruped",
    capabilities={"currentA": 12, "forceFreeG": 25, "forceMidG": 120},
    feeds=(
      code("legs.dock.PIN_TRAVEL", "m", "the rated travel", expect=0.001143),
      code("legs.dock.PIN_STROKE", "m", "the full stroke", expect=0.002286),
      code("legs.dock.PIN_TIP_R", "m", "the 1.27 mm plunger's radius",
           expect=0.000635),
      code("legs.dock.PIN_FREE_N", "N", "25 g at the start of travel"),
      code("legs.dock.PIN_MID_N", "N", "120 g mid-stroke"),
    ),
    why={"massG": "not on DigiKey; the datasheet not read",
         "dimensionsMm": "7.32-9.60 mm working height"},
    note="€1.781 ex VAT each at ten (€2.499 singly, VAT in). Parts.md's "
         "$2.45 was DigiKey US's.",
  ),
  Part(
    "aisler_belly_pads", "Belly pads: 2-layer ENIG PCB, 160 x 20 mm, a "
    "batch of 6 (AISLER Budget)", "electronics", "chosen", ("build",), (),
    source="https://aisler.net/en/products/boards",
    dimensionsMm={"length": 160, "width": 20, "thickness": 1.6},
    priceEur=38.83, quantity=1, robot="quadruped",
    feeds=(
      code("legs.model.PAD_HALF", "m", "half the 160 x 20 mm strip"),
      code("legs.model.PAD_Y", "m", "the two pads 60 mm apart"),
    ),
    why={"partNumber": "a board of our own, ordered by its gerbers",
         "massG": "not given"},
    note="Made and shipped in Germany; asking for five prices six. JLCPCB's "
         "five are €19.90 with no VAT shown, plus shipping.",
  ),
  Part(
    "uhmwpe_300_10", "UHMW-PE (PE 1000) sheet, natural, 300 x 300 x 10 mm",
    "structure", "chosen", ("build",), (),
    partNumber="0100-RI45070-0300-0300-N",
    source="https://www.kunststoffhaus.de/0020-RI45070-DIV-10-25-mm-PE-UHMW-"
           "Platte-PE-1000-Polyethylen-Abm-und-Farbe-waehlbar",
    dimensionsMm={"length": 300, "width": 300, "thickness": 10},
    priceEur=46.67, quantity=1, robot="quadruped",
    feeds=(
      code("legs.dock.DockSpec.cradle_mu", "1",
           "flown at 0.3; UHMW-PE slides at 0.12-0.17 (a supplier's figure)"),
    ),
    why={"massG": "not given; about 840 g at the material's 0.93 g/cm³"},
    note="The cradle's bed and funnel faces. ADS cuts one for €32.17 "
         "(€25 of it the saw) and states no lead time.",
  ),
  Part(
    "quad_cradle_work", "The dock's cradle: the bed and funnel faces cut "
    "from the sheet, on a base", "structure", "chosen", ("build",), (),
    robot="quadruped",
    feeds=(
      code("legs.dock.DockSpec.mouth_half", "m", "the funnel's mouth"),
      code("legs.dock.DockSpec.bed_half", "m", "the bed"),
    ),
    why={"partNumber": "no design yet", "source": "a job shop, once drawn",
         "massG": "no design yet", "dimensionsMm": "a 0.30 m cradle",
         "priceEur": "no design yet"},
  ),
  Part(
    "quad_pack_materials", "The pack's printed case, fish paper, heat-shrink "
    "and 12S balance lead", "power", "chosen", ("build",), ("pluggybot",),
    robot="quadruped",
    feeds=(code("legs.model.CHOSEN.belly_depth", "m",
                "the belly pack's depth under the hips"),),
    why={"partNumber": "no design yet", "source": "small parts, not priced",
         "massG": "in `MassBudget.battery`", "dimensionsMm": "no design yet",
         "priceEur": "small parts, not priced"},
  ),
  # ---- the bench leg's stand and the gantry's fittings ----------------------
  Part(
    "dold_mgn12_rail_1000", "Linear rail MGN12, 1000 mm (Dold)", "structure",
    "chosen", ("build",), (), partNumber="MGN12R-1000",
    source="https://www.dold-mechatronik.de/Linearfuehrung-MGN12R-1000mm",
    massG=650, priceEur=47.00, quantity=1, robot="quadruped",
    why={"dimensionsMm": "a 1000 mm MGN12 rail"},
    note="The bench leg hangs on it and hops. The MGN15's are sold out to "
         "late November.",
  ),
  Part(
    "dold_mgn12h", "Linear carriage MGN12H (Dold)", "structure", "chosen",
    ("build",), (), partNumber="MGN12H",
    source="https://www.dold-mechatronik.de/Linearwagen-MGN12H",
    massG=54, priceEur=13.00, quantity=1, robot="quadruped",
    capabilities={"dynamicKn": 3.72, "staticKn": 5.88},
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "quad_bench_mount", "The bench leg's mount: the carriage's plate holding "
    "the hip, and the hop's end stops", "structure", "chosen", ("build",), (),
    robot="quadruped",
    feeds=(code("legs.model.CHOSEN.stand_height", "m",
                "the hop's travel sized from the standing height"),),
    why={"partNumber": "no design yet", "source": "a laser cutter, once drawn",
         "massG": "no design yet", "dimensionsMm": "no design yet",
         "priceEur": "no design yet"},
  ),
  Part(
    "quad_gantry_fittings", "The gantry's fittings: angle brackets and T-nuts "
    "for 40x40 slot 8, and two eye bolts", "fastener", "chosen", ("build",),
    (), robot="quadruped",
    why={"partNumber": "not read for 40x40", "source": "Dold, not read",
         "massG": "not read", "dimensionsMm": "slot 8",
         "priceEur": "not read for 40x40"},
  ),
  # ---- the bench leg, the gantry, the shop ---------------------------------
  Part(
    "korad_ka6005p", "Korad KA6005P lab power supply, 0-60 V, 0-5 A, safety "
    "terminals", "equipment", "chosen", ("build",), (), partNumber="KA6005P",
    source="https://www.welectron.com/Korad-KA6005P-Benchtop-Power-Supply-"
           "Safety-Terminals",
    massG=7500, priceEur=239.00, quantity=1, robot="quadruped",
    capabilities={"outputV": "0-60", "outputA": "0-5", "powerW": 300,
                  "modes": "CC/CV, USB and RS232"},
    why={"dimensionsMm": "not recorded"},
    note="The bench leg's first power-ups, current-limited. A lab supply "
         "cannot take back what the motors return on landing, so the leg "
         "hops on the pack (and a clamp), never on this. Joy-IT's RD6012 "
         "with its 65 V supply and case gives 12 A for €239.28, on an "
         "open-frame supply.",
  ),
  Part(
    "hilumin_strip_10m", "Hilumin (nickel-plated steel) cell strip, 10 x "
    "0.15 mm, 10 m", "power", "chosen", ("build",), (),
    partNumber="2645314",
    source="https://www.akkuman.de/shop/Schweissband-aus-Hilumin-10-mm-015-"
           "dick-10-m-lang",
    massG=150, priceEur=19.95, quantity=1, robot="quadruped",
    capabilities={"widthMm": 10, "thicknessMm": 0.15},
    why={"dimensionsMm": "10 m of 10 x 0.15 mm"},
    note="Nickel-plated steel, not nickel: more resistance, so the links that "
         "carry the whole pack current are doubled. Pure nickel was in stock "
         "at no EU seller a page could be read from.",
  ),
  Part(
    "kweld_kit", "keenlab kWeld spot welder, complete kit, cables assembled",
    "equipment", "chosen", ("build",), (), partNumber="kWeld complete kit",
    source="https://www.keenlab.de/index.php/product/kweld-complete-kit/",
    massG=800, priceEur=210.63, quantity=1, robot="quadruped",
    capabilities={"strip": "up to about 0.3 mm", "pulses": "energy-regulated"},
    why={"dimensionsMm": "not recorded"},
    note="€177.00 ex VAT as the page shows it. It needs a source of 800 A: "
         "`kweld_kcap`. Malectrics' V4 bundle is €148.75 without its LiPo; "
         "FNIRSI's SWM-10 is €42.90 and welds 0.1-0.25 mm.",
  ),
  Part(
    "kweld_kcap", "keenlab kCap ultracapacitor module for the kWeld",
    "equipment", "chosen", ("build",), (), partNumber="kCap",
    source="https://www.keenlab.de/index.php/product/kweld-ultracapacitor-"
           "module/",
    massG=600, priceEur=153.51, quantity=1, robot="quadruped",
    capabilities={"pulseA": 1300, "chargeV": 8.1},
    why={"dimensionsMm": "not recorded"},
    note="€129.00 ex VAT as the page shows it; the bench supply charges it.",
  ),
  Part(
    "engineer_pa09", "Engineer PA-09 crimping tool (JST-PH/GH class, AWG "
    "32-20)", "equipment", "chosen", ("build",), (), partNumber="PA-09",
    source="https://www.kiwi-electronics.com/en/jst-crimping-tool-pa-09-916",
    massG=135, priceEur=47.59, quantity=1, robot="quadruped",
    why={"dimensionsMm": "not recorded"},
    note="€39.99 ex VAT; ships from the Netherlands. The drivers' CAN and "
         "power leads.",
  ),
  Part(
    "knipex_9762145a", "Knipex 97 62 145 A ferrule crimping pliers, "
    "0.25-2.5 mm²", "equipment", "chosen", ("build",), (),
    partNumber="97 62 145 A",
    source="https://www.reichelt.com/de/en/shop/product/crimping_pliers_for_"
           "end_sleeves_ferrules_-184328",
    massG=170, priceEur=30.80, quantity=1, robot="quadruped",
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "pinecil_v2", "Pine64 Pinecil V2 soldering iron", "equipment", "chosen",
    ("build",), (), partNumber="PINECIL-BB2",
    source="https://eleshop.de/pinecil-smart-mini-tragbarer-lotkolben.html",
    massG=28, priceEur=39.90, quantity=1, robot="quadruped",
    capabilities={"powerW": 88, "supply": "USB-C PD 12-20 V or DC 12-24 V"},
    why={"dimensionsMm": "not recorded"},
    note="No supply included; the bench supply runs it off its barrel jack.",
  ),
  Part(
    "unit_ut161e", "UNI-T UT161E true-RMS multimeter", "equipment", "chosen",
    ("build",), (), partNumber="UT161E",
    source="https://eleshop.de/uni-t-ut161e.html",
    massG=400, priceEur=95.60, quantity=1, robot="quadruped",
    capabilities={"counts": 22000, "category": "CAT III 1000 V"},
    why={"dimensionsMm": "not recorded"},
    note="The EU name of the UT61E+; Reichelt has it at €109.00.",
  ),
  Part(
    "proxxon_mc5", "Proxxon MicroClick MC 5 torque wrench, 1-5 N·m, 1/4\"",
    "equipment", "chosen", ("build",), (), partNumber="23347",
    source="https://www.reichelt.com/de/en/shop/product/micro-click_torque_"
           "screwdriver_5_s-91768",
    massG=431, priceEur=77.99, quantity=1, robot="quadruped",
    capabilities={"rangeNm": "1-5", "accuracy": "±6 %"},
    why={"dimensionsMm": "not recorded"},
    note="The actuators' and frame's screws to their torques.",
  ),
  Part(
    "wera_950pks9", "Wera 950 PKS/9 SM N metric hex key set, 1.5-10 mm",
    "equipment", "chosen", ("build",), (), partNumber="05133163001",
    source="https://www.reichelt.com/de/en/shop/product/spanner_set_hex_950_"
           "pks_smn_9-pieces-169289",
    massG=265, priceEur=21.80, quantity=1, robot="quadruped",
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "bambu_p1s", "Bambu Lab P1S 3D printer (printer only)", "equipment",
    "chosen", ("build",), (), partNumber="P1S",
    source="https://eu.store.bambulab.com/products/p1s",
    massG=12950, priceEur=379.00, quantity=1, robot="quadruped",
    capabilities={"buildVolumeMm": {"x": 256, "y": 256, "z": 256},
                  "petg": "ideal (maker)"},
    why={"dimensionsMm": "not recorded"},
    note="Enclosed, for PETG: the rack's trays, the tools' frames, the "
         "cable guides, the pack's case. The page says the VAT may change "
         "at checkout. The A1 (open) is €259.",
  ),
  Part(
    "petg_1kg", "PETG filament, 1.75 mm, 1 kg (DAS FILAMENT)", "structure",
    "chosen", ("build",), (), partNumber="F10765",
    source="https://dasfilament.de/produkt/petg-filament-175-mm-schwarz-1-kg/",
    massG=1000, priceEur=21.00, quantity=1, robot="quadruped",
    why={"dimensionsMm": "a 1 kg spool"},
  ),
  Part(
    "dold_4040_1m", "Aluminium extrusion 40x40 light, I-type slot 8, cut "
    "to length (1 m)", "structure", "chosen", ("build",), (),
    partNumber="60800-Z",
    source="https://www.dold-mechatronik.de/Aluminum-Profile-40x40L-I-Type-"
           "Groove-8-176kg-m-Customized-Cutting-50-to-6000mm",
    massG=1759, dimensionsMm={"width": 40, "height": 40, "length": 1000},
    priceEur=18.90, quantity=1, robot="quadruped",
    note="€17.90 a metre and €1.00 a cut, so a metre in one piece is "
         "€18.90.",
  ),
  Part(
    "donges_rope_ratchet_113", "Dönges rope ratchet, 113 kg, 5 m",
    "equipment", "chosen", ("build",), (),
    partNumber="Seilzugratsche 113 kg",
    source="https://www.feuerwehrdiscount.de/seilzugratsche-rope-ratchet/"
           "auslastung-bis-max.-68-kg",
    priceEur=33.99, quantity=1, robot="quadruped",
    capabilities={"ratedKg": 113, "ropeM": 5},
    why={"massG": "given for the 68 kg one only",
         "dimensionsMm": "not recorded"},
    note="The body's hanger, raised and lowered by hand. The 68 kg one is "
         "€14.99; the 113 kg one leaves the catch load its margin (the "
         "page's variant).",
  ),
  Part(
    "mammut_magic_sling_120", "Mammut Magic Sling 12.0, 120 cm (22 kN)",
    "equipment", "chosen", ("build",), (), partNumber="314-0107",
    source="https://www.bergfreunde.de/mammut-magic-sling-120-bandschlinge/",
    priceEur=18.00, quantity=1, robot="quadruped",
    capabilities={"breakingKn": 22, "widthMm": 12},
    why={"massG": "not on the page", "dimensionsMm": "a 120 cm sling"},
    note="No harness is sold for a robot: two slings round the torso, fore "
         "and aft, meet at the ratchet.",
  ),
  Part(
    "hummelt_bungee_10m", "Bungee cord, 8 mm, 10 m (Hummelt)", "equipment",
    "chosen", ("build",), (), partNumber="1037",
    source="https://hummelt-shop.de/produkt/expanderseil-gummiseil-8mm-blau/",
    massG=420, priceEur=12.49, quantity=1, robot="quadruped",
    capabilities={"breakingKg": 70, "stretch": "95-110 %"},
    why={"dimensionsMm": "10 m of 8 mm cord"},
    note="Between the ratchet and the slings, doubled or tripled, so a fall "
         "is caught softly.",
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
  required = ("id", "name", "kind", "status", "shelves", "robot", "usedBy",
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
  if entry["robot"] not in WORLDS:
    out.append(f"robot {entry['robot']!r} not in {tuple(WORLDS)}")
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
  if entry["status"] == "chosen" and {"body", "build"} & set(entry["shelves"]) \
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


# ---- the build: the quadruped's bill of materials (#379) ---------------------

#: What the build may cost (#379: "priced against the $20,000 budget"), and
#: how a line's euros become dollars: the ECB's reference rate on RATE_DATE.
BUDGET_USD = 20_000
USD_PER_EUR = 1.1403
RATE_DATE = "2026-09-25"
RATE_SOURCE = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
#: Every price is what a buyer in Germany pays, VAT in (a seller quoting ex
#: VAT or dollars is converted, and the line says so), read on CHECKED. A
#: business that reclaims the VAT spends 16 % less.
VAT = 0.19
CHECKED = "2026-09-28"
#: On top of every line: shipping, the drift between reading a price and
#: ordering, and the part bought twice. A decision, not a measurement.
CONTINGENCY = 0.15
#: The bill's sections, in the order the build needs them (#379, "The
#: first build step": one leg on a bench, then the body on a gantry).
GROUPS = ("actuation", "power", "compute", "sensors", "frame and legs", "arm",
          "tools", "rack", "dock", "safety", "bench leg", "gantry", "spares",
          "shop")


@dataclass
class Line:
  """One line of the bill: `quantity` of a `build`-shelf part (in its
  `priceFor` units), bought for `group`. A part with no price -- nothing
  designed yet -- is carried at an ALLOWANCE, a sum set aside for the whole
  line and never a price, with the `basis` it rests on. `leadTime` is the
  seller's own words on CHECKED; a null one needs its `why`."""
  part: str
  quantity: int
  group: str
  leadTime: str | None
  allowanceEur: float | None = None
  basis: str = ""
  why: dict = field(default_factory=dict)
  note: str = ""


#: A lead time this bill did not read: the part's price is the workshop
#: catalog's (#199, 2026-09-14), not a page read on CHECKED.
NOT_READ = {"leadTime": "not read for this bill; the price is the workshop "
                        "catalog's (#199, 2026-09-14)"}
#: ...and a line with no drawing yet: nothing to quote, so no lead.
UNDRAWN = {"leadTime": "nothing drawn to quote"}
EASY_COMPOSITES = ("more than 50 in stock, dispatched the same day before "
                   "13:00, DHL 2 days (Easy Composites, NL)")

LINES: tuple[Line, ...] = (
  # ---- actuation -----------------------------------------------------------
  Line("gim8108_8", 14, "actuation",
       "pre-order: none in stock, \"restock in 10-20 days\" (OpenELAB)",
       note="twelve on the legs, two on the arm"),
  Line("amass_xt30_22_pair", 16, "actuation",
       "dispatched within 2-3 days, UPS or DHL (3DPTronics, Italy)",
       note="one a driver, spares' too"),
  Line("lapp_unitronic_can", 10, "actuation",
       "176 m in stock, delivered on 2026-09-30 (Conrad)"),
  Line("silicone_wire_14awg", 20, "actuation",
       "available now, 1-3 working days (MyLipo)",
       note="ten metres red, ten black"),
  # ---- compute -------------------------------------------------------------
  Line("raspberry_pi_5_quad", 1, "compute", "in stock, 1-3 days (BerryBase)"),
  Line("pi5_active_cooler", 1, "compute", "in stock, 1-3 days (BerryBase)"),
  Line("pi_ssd_kit_256", 1, "compute",
       "not deliverable now (BerryBase); the 512 GB kit is in stock at "
       "Reichelt for €166.10"),
  Line("waveshare_can_hat_plus", 1, "compute", "in stock, 1-3 days (Eckstein)"),
  Line("waveshare_can_fd_hat", 1, "compute",
       "in stock (short supply), 1-3 business days (Welectron)"),
  # ---- frame and legs ------------------------------------------------------
  Line("quad_frame_plates", 1, "frame and legs", None, why=UNDRAWN,
       allowanceEur=300.0,
       basis="six to eight 3 mm AlMg3 parts: AluFritze's online configurator "
             "prices a plain 300 x 150 mm rectangle at €20.02 (four for "
             "€80.10, 10-14 working days); a plate with holes is priced from "
             "its drawing"),
  Line("cf_tube_25_22", 2, "frame and legs", EASY_COMPOSITES),
  Line("quad_leg_brackets", 1, "frame and legs", None, why=UNDRAWN,
       allowanceEur=800.0,
       basis="twelve motor mounts, four hip brackets and eight tube clamps, "
             "nothing drawn, sized as machined aluminium at a job shop; "
             "printed on the shop's P1S they cost their filament"),
  Line("squash_balls_12", 1, "frame and legs",
       "in stock, 1-3 working days (squashpoint.de)",
       note="four feet and eight spares"),
  # ---- arm -----------------------------------------------------------------
  Line("cf_tube_20_18", 1, "arm", EASY_COMPOSITES),
  Line("skf_61800_2rs1", 4, "arm", "in stock, 1-4 days (kugellager-shop.net)"),
  Line("igus_kcrm_05", 6, "arm", "ready to ship in 24 hours (igus)"),
  Line("quad_arm_fork", 1, "arm", None, why=UNDRAWN, allowanceEur=150.0,
       basis="one machined aluminium plate with its prongs, V's and ramps, "
             "and three 5 mm rods threaded M5: a plain 4 mm plate is €23.12 "
             "at AluFritze, the V's and ramps are machining"),
  Line("ptfe_tape_glass", 1, "arm", "2-4 working days (High-tech-flon)"),
  # ---- tools ---------------------------------------------------------------
  Line("quad_tool_peg", 4, "tools", None, why=UNDRAWN, allowanceEur=30.0,
       basis="four pegs cut and turned from a metre of 6 mm silver steel "
             "and an acetal bush each, as the rover's peg (`peg_rod_6mm`)"),
  Line("servo_fs90mg", 2, "tools", None, why=NOT_READ,
       note="the claw's jaws and the seed dispenser's gate"),
  Line("slide_l12_100", 1, "tools", None, why=NOT_READ,
       note="the pen's sideways carriage: 100 mm of the 110 it wants, the "
            "nearest slide with a published draw"),
  Line("module_esp32", 4, "tools", None,
       why={"leadTime": "no board chosen"}, note="one a tool"),
  Line("lcd_display", 1, "tools", None, why={"leadTime": "no display chosen"},
       allowanceEur=20.0,
       basis="a small SPI display for the LCD tool's ESP32; the rover's LCD "
             "module never chose one"),
  # ---- rack ----------------------------------------------------------------
  Line("bay_switch", 3, "rack", None, why=NOT_READ,
       note="one a bay, as the rover's rack reports its bays (#351)"),
  Line("module_esp32", 1, "rack", None, why={"leadTime": "no board chosen"},
       note="the rack's board, which reports the switches"),
  Line("dold_3030_1m", 4, "rack", "in stock, 4-5 working days (Dold)",
       note="two posts, the rail and its feet"),
  Line("dold_angle_30", 8, "rack", "in stock, 3-4 working days (Dold)"),
  Line("quad_rack_board", 1, "rack", None,
       why={"leadTime": "a board from a DIY store; the trays and tags printed "
                        "in the shop"},
       allowanceEur=40.0,
       basis="a 1.0 x 0.6 m plywood back board, six V-trays printed in PETG "
             "from the shop's spools, and the rack's six and the dock's four "
             "60 mm tags printed and laminated"),
  # ---- power ---------------------------------------------------------------
  Line("molicel_p45b", 12, "power", "ready to ship in 1-2 days (akkuteile.de)"),
  Line("jbd_bms_12s_120a", 1, "power",
       "short stock, 1-5 working days (bikebattery.de)"),
  Line("hilumin_strip_10m", 1, "power", "in stock, 1-3 working days (AKKUman)"),
  Line("meanwell_ddr_60l_5", 1, "power",
       "in stock, 1-2 business days (Reichelt)"),
  Line("amass_xt90s", 3, "power", "in stock, 1-2 working days (MHM)",
       note="the pack's lead, the charger's and the bench supply's"),
  Line("victron_midi_60a", 1, "power", "in stock, 1-3 working days (Klimaworld)"),
  Line("victron_midi_holder", 1, "power", "ships in 1-3 working days (Offgridtec)"),
  Line("silicone_wire_10awg", 4, "power", "in stock, 1-3 days (Skeries)",
       note="two metres red, two black"),
  Line("quad_pack_materials", 1, "power", None, why=UNDRAWN, allowanceEur=30.0,
       basis="the case printed in the shop's PETG; fish paper, heat-shrink "
             "and a 12S balance lead"),
  # ---- sensors -------------------------------------------------------------
  Line("quad_rplidar_c1", 1, "sensors", "1-3 business days (Funduino)"),
  Line("quad_realsense_d435", 1, "sensors",
       "none in stock, the maker's standard lead 14 weeks (DigiKey DE); "
       "21 days at MyBotShop for €419.99"),
  Line("quad_pi_camera_3", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("pi5_camera_cable", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("tdk_ev_icm42688p", 1, "sensors",
       "138 in stock, the maker's standard lead 16 weeks (DigiKey DE)"),
  # ---- dock ----------------------------------------------------------------
  Line("millmax_0858", 4, "dock",
       "18,820 in stock, the maker's standard lead 4 weeks (DigiKey DE)",
       note="two a pole"),
  Line("aisler_belly_pads", 1, "dock",
       "dispatched in about 10 business days, then 2 days by post (AISLER)"),
  Line("uhmwpe_300_10", 1, "dock",
       "available now, 3-5 working days (kunststoffhaus.de)"),
  Line("quad_cradle_work", 1, "dock", None, why=UNDRAWN, allowanceEur=100.0,
       basis="the bed and the funnel's faces routed from the sheet and "
             "screwed to a base: an hour at a job shop, or by hand"),
  Line("noeifevo_charger_12s", 1, "dock",
       "3-7 working days from its Polish warehouse, 7-15 from China; the page "
       "does not say which"),
  # ---- safety --------------------------------------------------------------
  Line("albright_sw80b549", 1, "safety", None,
       why={"leadTime": "the page states none (it can be put in the cart)"}),
  Line("schneider_xb5as8442", 2, "safety", "available on 2026-10-01 (Reichelt)",
       note="one on the robot, one on the bench rig"),
  Line("radiomaster_pocket", 1, "safety", "ships the same day (n-Factory)"),
  Line("radiomaster_er6", 1, "safety", "ships the same day (n-Factory)"),
  Line("pololu_rc_relay", 1, "safety", "about 8-10 days (Eckstein)"),
  Line("molicel_p28a", 2, "safety", "available, 1-3 days (LiPo24)",
       note="the transmitter's"),
  Line("meanwell_ddr_30l_12", 1, "safety",
       "in stock, 1-2 business days (Reichelt)", note="the contactor's coil"),
  Line("arcol_hs25_47r", 1, "safety", "in stock, 1-2 business days (Reichelt)",
       note="pre-charge"),
  Line("hongfa_hf115f_012", 1, "safety",
       "in stock, 1-2 business days (Reichelt)", note="pre-charge"),
  # ---- the bench leg -------------------------------------------------------
  Line("korad_ka6005p", 1, "bench leg", "in stock, 1-3 business days (Welectron)"),
  Line("ucan_v1", 1, "bench leg", "in stock, 1-3 days (Eckstein)"),
  Line("dold_mgn12_rail_1000", 1, "bench leg", "in stock, 3-4 working days (Dold)"),
  Line("dold_mgn12h", 1, "bench leg", "in stock, 3-4 working days (Dold)"),
  Line("dold_3030_1m", 3, "bench leg", "in stock, 4-5 working days (Dold)",
       note="the stand: a 1.2 m upright and its base"),
  Line("dold_angle_30", 6, "bench leg", "in stock, 3-4 working days (Dold)"),
  Line("quad_bench_mount", 1, "bench leg", None, why=UNDRAWN,
       allowanceEur=60.0,
       basis="a 4 mm AlMg3 plate at AluFritze (a plain 300 x 150 mm one is "
             "€23.12) and printed end stops"),
  Line("pi5_psu_27w", 1, "bench leg", "in stock, 1-3 days (BerryBase)"),
  # ---- the gantry ----------------------------------------------------------
  Line("dold_4040_1m", 8, "gantry", "in stock, 4-5 working days (Dold)",
       note="two 2 m uprights, a 1.5 m beam and two 1 m feet, priced as "
            "metres; a longer cut saves a euro"),
  Line("donges_rope_ratchet_113", 1, "gantry",
       "in stock, 1-2 working days (feuerwehrdiscount.de)"),
  Line("mammut_magic_sling_120", 2, "gantry", "2-3 working days (Bergfreunde)"),
  Line("hummelt_bungee_10m", 1, "gantry", "in stock, 2-3 days (Hummelt)"),
  Line("quad_gantry_fittings", 1, "gantry", None,
       why={"leadTime": "not read for 40x40"}, allowanceEur=40.0,
       basis="eight angle brackets with T-nuts for 40x40 slot 8 and two eye "
             "bolts; the 30x30's brackets are €1.79-2.20 at Dold, the 40x40's "
             "were not read"),
  # ---- spares --------------------------------------------------------------
  Line("gim8108_8", 2, "spares",
       "pre-order: none in stock, \"restock in 10-20 days\" (OpenELAB)"),
  Line("molicel_p45b", 2, "spares", "ready to ship in 1-2 days (akkuteile.de)"),
  Line("victron_midi_60a", 2, "spares", "in stock, 1-3 working days (Klimaworld)"),
  Line("millmax_0858", 6, "spares",
       "18,820 in stock, the maker's standard lead 4 weeks (DigiKey DE)",
       note="with the dock's four, ten: the price at ten"),
  Line("skf_61800_2rs1", 2, "spares", "in stock, 1-4 days (kugellager-shop.net)"),
  # ---- the shop ------------------------------------------------------------
  Line("kweld_kit", 1, "shop", "28 in stock (keenlab)"),
  Line("kweld_kcap", 1, "shop", "41 in stock (keenlab)"),
  Line("engineer_pa09", 1, "shop", "31 in stock, ships the same day (Kiwi, NL)"),
  Line("knipex_9762145a", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("pinecil_v2", 1, "shop", "in stock, ships the same day (eleshop)"),
  Line("unit_ut161e", 1, "shop", "expected on 2026-10-07 (eleshop)"),
  Line("proxxon_mc5", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("wera_950pks9", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("bambu_p1s", 1, "shop",
       "ships from the EU warehouse in 1-3 business days (Bambu Lab)"),
  Line("petg_1kg", 3, "shop", None,
       why={"leadTime": "in stock; the page and its terms give no delivery time"},
       note="the rack's trays, the tools' frames, the pack's case, cable guides"),
)


def validate_line(line: dict, parts: dict[str, dict]) -> list[str]:
  """Every reason one emitted line of the bill is not honest, or `[]`."""
  out: list[str] = []
  p = parts.get(line["part"])
  if p is None:
    return [f"no part {line['part']!r}"]
  if "build" not in p["shelves"]:
    out.append(f"{line['part']} is not on the build shelf")
  if not (isinstance(line["quantity"], int) and line["quantity"] > 0):
    out.append("quantity must be a positive whole number")
  if line["group"] not in GROUPS:
    out.append(f"group {line['group']!r} not in {GROUPS}")
  if p["priceEur"] is None:
    a = line["lineEur"]
    if not (isinstance(a, (int, float)) and a > 0) or not line["basis"]:
      out.append("a part with no price needs an allowance and its basis")
  elif line["basis"] or line["eachEur"] is None:
    out.append("a priced part carries its price, not an allowance")
  if (line["leadTime"] is None) != ("leadTime" in line["why"]):
    out.append("a lead time is the seller's words, or null with a why")
  if set(line["why"]) - {"leadTime"}:
    out.append(f"why names {sorted(set(line['why']) - {'leadTime'})}")
  return out


def bom(parts: dict[str, dict]) -> dict:
  """The bill as data: every line priced off its part (or its allowance),
  and the totals against the budget -- summed here, never typed."""
  lines = []
  for ln in LINES:
    p = parts.get(ln.part, {})
    each = p.get("priceEur")
    lines.append({
      "part": ln.part, "name": p.get("name"), "group": ln.group,
      "quantity": ln.quantity, "priceFor": p.get("priceFor"), "eachEur": each,
      "lineEur": (round(each * ln.quantity, 2) if each is not None
                  else ln.allowanceEur),
      "allowance": each is None, "basis": ln.basis, "leadTime": ln.leadTime,
      "why": ln.why, "note": ln.note,
    })

  def total(pick) -> float:
    return round(sum(ln["lineEur"] or 0 for ln in lines if pick(ln)), 2)
  sourced = total(lambda ln: not ln["allowance"])
  allowed = total(lambda ln: ln["allowance"])
  contingency = round((sourced + allowed) * CONTINGENCY, 2)
  eur = round(sourced + allowed + contingency, 2)
  return {
    "budgetUsd": BUDGET_USD, "usdPerEur": USD_PER_EUR, "rateDate": RATE_DATE,
    "rateSource": RATE_SOURCE, "vat": VAT, "checked": CHECKED,
    "contingency": CONTINGENCY, "groups": list(GROUPS), "lines": lines,
    "totals": {
      "sourcedEur": sourced, "allowanceEur": allowed,
      "contingencyEur": contingency, "totalEur": eur,
      "totalUsd": round(eur * USD_PER_EUR, 2),
      "budgetEur": round(BUDGET_USD / USD_PER_EUR, 2),
      "byGroup": {g: total(lambda ln, g=g: ln["group"] == g) for g in GROUPS},
    },
  }


def _eur(v: float | None) -> str:
  return "—" if v is None else f"{v:,.2f}"


#: Parts.md's bill sits between these; `main()` rewrites what is between.
BOM_START = ("<!-- bom: rendered by `uv run python -m pluggybot.rack.catalog` "
             "from rack/catalog.py's LINES; edit those, not this -->")
BOM_END = "<!-- /bom -->"
#: A line's feeds, as the bill shows them: the first few constants' names.
FEEDS_SHOWN = 3


def bom_markdown(fixture: dict) -> str:
  """The bill as Parts.md shows it: one row a line under its group's
  subtotal, the allowances' bases as notes, and the totals."""
  b, parts = fixture["build"], {p["id"]: p for p in fixture["parts"]}
  t = b["totals"]
  rows = [f"Prices and lead times read {b['checked']}; € incl. {b['vat']:.0%} "
          f"VAT; $1 = €{1 / b['usdPerEur']:.4f} (ECB, {b['rateDate']}).", "",
          "| part | qty | € each | € line | lead time | feeds |",
          "|---|---|---|---|---|---|"]
  notes = []
  for g in b["groups"]:
    mine = [ln for ln in b["lines"] if ln["group"] == g]
    if not mine:
      continue
    rows.append(f"| **{g}** | | | **{_eur(t['byGroup'][g])}** | | |")
    for ln in mine:
      p = parts[ln["part"]]
      name = f"[{p['name']}]({p['source']})" if p["source"] else p["name"]
      if p["partNumber"]:
        name += f" `{p['partNumber']}`"
      unit = "" if ln["priceFor"] == "each" else f" /{ln['priceFor']}"
      each = "allowance" if ln["allowance"] else _eur(ln["eachEur"]) + unit
      if ln["allowance"]:
        notes.append(f"- **{p['name']}** ({g}), €{_eur(ln['lineEur'])}: "
                     f"{ln['basis']}")
      feeds = [f"`{f['constant']}`" for f in p["feeds"]]
      if len(feeds) > FEEDS_SHOWN:
        feeds = feeds[:FEEDS_SHOWN] + [f"+{len(feeds) - FEEDS_SHOWN}"]
      lead = ln["leadTime"] if ln["leadTime"] is not None else \
        f"unknown: {ln['why']['leadTime']}"
      rows.append(f"| {name} | {ln['quantity']} | {each} | {_eur(ln['lineEur'])} "
                  f"| {lead} | {', '.join(feeds) or '—'} |")
  rows += ["", "| | € | $ |", "|---|---|---|",
           f"| priced lines | {_eur(t['sourcedEur'])} | |",
           f"| allowances (nothing designed yet) | {_eur(t['allowanceEur'])} | |",
           f"| contingency, {b['contingency']:.0%} | {_eur(t['contingencyEur'])} | |",
           f"| **total** | **{_eur(t['totalEur'])}** | **{_eur(t['totalUsd'])}** |",
           f"| budget | {_eur(t['budgetEur'])} | {_eur(b['budgetUsd'])} |"]
  if notes:
    rows += ["", "**The allowances**, each a sum set aside, never a price:", *notes]
  return "\n".join(rows)


def build(world: str = WORLD) -> dict:
  """Read every feed off its robot's compiled world and emit the fixture.
  `world` stands in for the rover's (the tests' moved-model probes)."""
  import mujoco
  from pluggybot.workshop.spec import unbuildable   # lazy: it imports this module
  specs = {robot: mujoco.MjSpec.from_file(world if robot == "rover" else path)
           for robot, path in WORLDS.items()}
  parts = []
  for p in PARTS:
    entry = p.as_dict(specs[p.robot])
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
    "worlds": dict(WORLDS),
    "kinds": list(KINDS),
    "statuses": list(STATUSES),
    "shelves": list(SHELVES),
    "nullable": list(NULLABLE),
    "parts": parts,
    "build": bom({p["id"]: p for p in parts}),
  }


def with_bill(doc: str, fixture: dict) -> str:
  """Parts.md with its bill re-rendered between the markers."""
  head, start, rest = doc.partition(BOM_START)
  _, end, tail = rest.partition(BOM_END)
  if not start or not end:
    raise SystemExit(f"Parts.md has no bill markers ({BOM_START!r} ... {BOM_END!r})")
  return f"{head}{BOM_START}\n{bom_markdown(fixture)}\n{BOM_END}{tail}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-o", "--out", default="protocol/parts.json")
  parser.add_argument("--doc", default="docs/Parts.md",
                      help="the doc whose bill is re-rendered")
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
  parts = {p["id"]: p for p in fixture["parts"]}
  bad = {ln["part"]: r for ln in fixture["build"]["lines"]
         if (r := validate_line(ln, parts))}
  if bad:
    raise SystemExit(f"refusing to write a dishonest bill: {bad}")
  out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
  doc = Path(args.doc)
  doc.write_text(with_bill(doc.read_text(), fixture))
  shelves = {s: sum(s in p["shelves"] for p in fixture["parts"])
             for s in SHELVES}
  print(f"{out}: {len(fixture['parts'])} parts, {shelves}, "
        f"{sum(len(p['feeds']) for p in fixture['parts'])} feeds")


if __name__ == "__main__":
  main()
