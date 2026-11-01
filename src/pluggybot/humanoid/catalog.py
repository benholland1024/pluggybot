"""The parts catalog as data (issue #185): what the humanoid's bill buys off
a seller's page, and the machinery a bill is built and checked with.

`docs/Parts.md` names the purchasable part behind each sim constant; the
humanoid's bill (`humanoid/bill.py`, issue #379) is that list as data, on
one shelf, `build`: its own parts (the actuators, the grippers, what is made
to a drawing) and the ones bought off a seller's page, whose facts live
here (`PARTS`) -- one part number, one price, one mass, read by the bill
and by `humanoid.parts` alike. `bom()` totals a bill against its budget, and
Parts.md's bill is RENDERED from it (`bom_markdown`), so the doc cannot
drift from the data either. `uv run python -m pluggybot.humanoid.bill`
writes both, and `protocol/parts.json`. The quadruped's bill went with it
(#559); the tag `quadruped-final` keeps it.

  ⚠ EVERY `feeds` VALUE IS READ OFF THE SIM AT BUILD TIME, never typed here.
  The bill opens the robot's model through `MjSpec` and imports the
  constants; a fixture that carried a hand-copied number would drift from
  the model silently and then be worse than nothing, because it looks
  authoritative. So a literal that moves fails `tests/test_catalog.py`'s
  stale check, and a part whose datasheet number IS the sim's number pins it
  with `expect` -- an RS04's 120 N·m is every one of its drivers'
  `forcerange`, and either side moving fails the suite rather than the
  website.

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

Vendored to the website beside `hints.json`; the parts page is its first
consumer.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

SCHEMA = 1

KINDS = ("motor", "actuator", "sensor", "structure", "power", "electronics",
         "fastener", "equipment")
STATUSES = ("chosen", "candidate", "removed")
SHELVES = ("build",)

#: Fields a part may not know. Each null needs its reason in `why`, keyed by
#: the field, and `why` may name nothing that is not null (a stale excuse).
NULLABLE = ("partNumber", "source", "massG", "dimensionsMm", "priceEur")


# ---- readers: a feed's value comes from HERE, never from a literal ---------

Reader = Callable[[Any], Any]


def _num(v: Any) -> Any:
  if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
    return [_num(x) for x in v]
  f = float(v)
  return int(f) if f.is_integer() else round(f, 6)


def _resolve(path: str) -> Any:
  """A dotted path under `pluggybot`: the longest prefix that imports as a
  module, then attributes (`humanoid.parts.RS04.peak_torque` is a field of
  the motor that module defines; a dataclass's field reads its default)."""
  names = path.split(".")
  for i in range(len(names) - 1, 0, -1):
    mod = "pluggybot." + ".".join(names[:i])
    try:
      obj = importlib.import_module(mod)
    except ModuleNotFoundError as e:
      if e.name != mod:
        raise
      continue
    for name in names[i:]:
      obj = getattr(obj, name)
    return obj
  raise ModuleNotFoundError(path)


def const(path: str) -> Reader:
  """A Python constant by dotted path under `pluggybot`."""
  return lambda _spec: _num(_resolve(path))


def entry(path: str, key: str) -> Reader:
  """One value of a dict constant (`humanoid.parts.ELECTRONICS_W`'s "lidar")."""
  return lambda _spec: _num(_resolve(path)[key])


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


def site(name: str, attr: str, index: int | None = None) -> Reader:
  def read(spec):
    v = getattr(next(s for s in spec.sites if s.name == name), attr)
    return _num(v if index is None else v[index])
  return read


def camera(name: str) -> Reader:
  def read(spec):
    return _num(next(c for c in spec.cameras if c.name == name).fovy)
  return read


def same(*readers: Reader) -> Reader:
  """Elements the design keeps alike (every driver of one part): one
  number, and a refusal if they have come apart."""
  def read(spec):
    vals = [r(spec) for r in readers]
    if any(v != vals[0] for v in vals):
      raise ValueError(f"asymmetric: {vals}")
    return vals[0]
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
  #: Which model its model-side feeds and `usedBy` are read against.
  robot: str = "humanoid"

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


PRINTED = "3D-printed, not bought"

PARTS: tuple[Part, ...] = (
  # ---- what the humanoid's bill buys off a seller's page (`humanoid.bill`) ----
  Part(
    "amass_xt30_22_pair", "Amass XT30(2+2) connector pair: power and signal in one (male, "
    "female and pins)",
    "electronics", "chosen", ("build",), (),
    partNumber="XT30(2+2)",
    source="https://www.3dptronics.com/electronics/xt3022-connectors-pair",
    priceEur=5.0,
    priceFor="pair",
    capabilities={"wiring": "16, 18 or 20 AWG (Amass)", "ratedA": "15 with 18 AWG (Amass)"},
    why={"massG": "not stated", "dimensionsMm": "not recorded"},
    note="Power and CAN in one, a driver's (RobStride's drivers take an XT30 and a GH1.25). No "
         "German shop stocks it: 3DPTronics (Italy) sells pairs only, its cart says VAT in; "
         "LCSC has the female alone (C19268028) at $0.38. Amass lists 16-20 AWG for its cups, "
         "not 14.",
  ),
  Part(
    "lapp_unitronic_can", "LAPP UNITRONIC BUS CAN 1 x 2 x 0.22 mm², shielded, by the metre",
    "electronics", "chosen", ("build",), (),
    partNumber="2170260/1",
    source="https://www.conrad.de/de/p/lapp-2170260-1-busleitung-unitronic-bus-1-x-2-x-0-22-mm-violett-meterware-603992.html",
    priceEur=1.98,
    capabilities={"impedanceOhm": 120, "outerDiameterMm": 5.7},
    why={"massG": "not stated", "dimensionsMm": "5.7 mm across, by the metre"},
    note="Four buses, six drivers each, daisy-chained. Rated for fixed installation, and a leg "
         "flexes it; LAPP's flexible CAN FD P is sold at Conrad only by the 100 m to "
         "businesses.",
  ),
  Part(
    "silicone_wire_14awg", "Silicone wire, 14 AWG (2.08 mm²), 1 m",
    "power", "chosen", ("build",), (),
    partNumber="SK14S / SK14R",
    source="https://www.mylipo.de/Silikonkabel-14AWG-208mm-schwarz_1",
    priceEur=2.25,
    why={"massG": "not stated per metre", "dimensionsMm": "a metre of 2.08 mm²"},
    note="Each driver's drop from the 10 AWG bus, red and black. Amass lists 16-20 AWG for the "
         "XT30(2+2)'s cups: step down to 16 at the plug.",
  ),
  Part(
    "molicel_p45b", "Molicel INR21700-P45B Li-ion cell (21700, 4.5 Ah)",
    "power", "chosen", ("build",), (),
    partNumber="INR21700-P45B",
    source="https://www.akkuteile.de/en/lithium-ionen-battery/size-21700/molicel/molicel-inr21700-p45b-4500mah-li-ion-battery-3-6v-3-7v_100852_3119",
    massG=70,
    priceEur=7.0,
    capabilities={"capacityAh": 4.5, "nominalV": 3.6, "chargeV": 4.2, "continuousA": 45},
    why={"dimensionsMm": "the 21700 format; the sheet's figures not read"},
    note="€7.00 each for eight or more (€7.90 for one to three) at akkuteile.de, the same SKU "
         "at LiPo24. No German seller was found publishing a 12S pack built to order: "
         "Enerprof's smallest 12S Molicel pack is a 12S3P at €860 in 14-28 days, the rest "
         "quote on request, so the pack is built here (`kweld_kit`).",
  ),
  Part(
    "jbd_bms_12s_120a", "JBD smart BMS, 10S-17S, 120 A, Bluetooth",
    "power", "chosen", ("build",), (),
    partNumber="10S-17S-120A-BT-UART",
    source="https://bikebattery.de/JBD-10S-17S-120A-360A-Intelligentes-Smart-Bluetooth-BMS-RS485-Android-iOS-APP-11S-12S-13S-14S-15S-16S",
    massG=100,
    dimensionsMm={"length": 105, "width": 55, "height": 12},
    priceEur=112.99,
    capabilities={"seriesCells": "10-17",
                  "continuousA": 120,
                  "balancing": "active; current not stated"},
    note="The seller's 60 A and 80 A versions are sold out; the JK JK-BD6A20S6P (60 A, 0.6 A "
         "active balancing) is €60.82 ex VAT from JK's own store in China with no delivery "
         "time stated. Mass is the seller's article weight.",
  ),
  Part(
    "hilumin_strip_10m", "Hilumin (nickel-plated steel) cell strip, 10 x 0.15 mm, 10 m",
    "power", "chosen", ("build",), (),
    partNumber="2645314",
    source="https://www.akkuman.de/shop/Schweissband-aus-Hilumin-10-mm-015-dick-10-m-lang",
    massG=150,
    priceEur=19.95,
    capabilities={"widthMm": 10, "thicknessMm": 0.15},
    why={"dimensionsMm": "10 m of 10 x 0.15 mm"},
    note="Nickel-plated steel, not nickel: more resistance, so the links that carry the whole "
         "pack current are doubled. Pure nickel was in stock at no EU seller a page could be "
         "read from.",
  ),
  Part(
    "meanwell_ddr_60l_5", "Mean Well DDR-60L-5 DC-DC converter, 18-75 V in, 5 V 12 A, isolated "
    "(DIN rail)",
    "power", "chosen", ("build",), (),
    partNumber="DDR-60L-5",
    source="https://www.reichelt.com/de/en/dc-dc-converter-60w-5v-12a-ddr-60l-5-p256746.html",
    massG=216,
    priceEur=43.8,
    capabilities={"inputV": "18-75",
                  "outputV": "5 (4.5-5.5 adjustable)",
                  "outputA": 12,
                  "isolationKv": 4},
    why={"dimensionsMm": "not recorded"},
    note="Set to 5.1 V for the Pi 5. The DDR-60G is 9-36 V in and would not take the pack.",
  ),
  Part(
    "amass_xt90s", "Amass XT90-S anti-spark connector pair",
    "power", "chosen", ("build",), (),
    partNumber="XT90S",
    source="https://www.mhm-modellbau.de/part-AM-XT90S.php",
    priceEur=4.05,
    priceFor="pair",
    why={"massG": "not on the page", "dimensionsMm": "not on the page"},
    note="The pack's lead, the charger's and the bench supply's: the anti-spark half takes the "
         "drivers' inrush as it mates.",
  ),
  Part(
    "victron_midi_60a", "Victron MIDI fuse 60 A / 58 V",
    "power", "chosen", ("build",), (),
    partNumber="CIP133060010",
    source="https://www.klimaworld.com/products/victron-midi-fuse-sicherungen-58v-48v-60a",
    priceEur=14.29,
    capabilities={"currentA": 60, "voltageV": 58, "dropMv": 70},
    why={"massG": "only a shipping weight is given", "dimensionsMm": "not recorded"},
    note="The pack's main fuse. Victron's sheet does not state its breaking capacity.",
  ),
  Part(
    "victron_midi_holder", "Victron MIDI fuse holder",
    "power", "chosen", ("build",), (),
    partNumber="CIP000050001",
    source="https://www.offgridtec.com/victron-midi-fuse-sicherung-halter.html",
    priceEur=10.12,
    capabilities={"currentA": 200, "maxV": 58},
    why={"massG": "the page gives 0.2 with no unit", "dimensionsMm": "not recorded"},
  ),
  Part(
    "silicone_wire_10awg", "Silicone wire, 10 AWG (6 mm²), 1 m",
    "power", "chosen", ("build",), (),
    partNumber="2392 (red), 2393 (black)",
    source="https://www.modellbau-skeries.de/p/silikonkabel-10awg-6mm-x-1000mm-rot",
    priceEur=4.4,
    why={"massG": "not stated per metre", "dimensionsMm": "a metre of 6 mm²"},
    note="The bus from the pack through the fuse and the contactor to the drivers.",
  ),
  Part(
    "noeifevo_charger_12s", "NOEIFEVO 50.4 V 5 A Li-ion charger (12S, CC/CV)",
    "power", "chosen", ("build",), (),
    partNumber="HRH-300L-50405-XT60",
    source="https://www.noeifevo.de/products/noeifevo-50-4v-5a-lithium-ladegerat-fur-12s-44-4v-li-ionen-lipo-akku-e-bike-roller-ladegerat-led-anzeige-aluminiumgehause",
    priceEur=69.0,
    capabilities={"outputV": 50.4, "outputA": 5, "inputV": "220-240 AC"},
    why={"massG": "not published", "dimensionsMm": "not published"},
    note="The bench charger: 216 W at the pack's nominal 43.2 V, 3.6 h from empty; the "
         "station's own is the NPB-750-48 (#559). The page states no VAT and no warehouse: its "
         "policy is 3-7 working days to Germany from Poland, 7-15 from China, and a sold-out "
         "label sits beside the add-to-cart. BOUNDMOTOR's (Parts.md's first pick) is $99 plus "
         "import duty, 12-20 days by air, with a barrel plug needing an adapter.",
  ),
  Part(
    "raspberry_pi_5", "Raspberry Pi 5, 8 GB",
    "electronics", "chosen", ("build",), (),
    partNumber="SC1112",
    source="https://www.berrybase.de/raspberry-pi-5-8gb-ram",
    dimensionsMm={"length": 85, "width": 56},
    priceEur=184.9,
    capabilities={"csiPorts": 2, "power": "5 V 5 A"},
    why={"massG": "Raspberry Pi publishes none; BerryBase's 65 g is packed"},
    note="The 16 GB (SC1113) is €319.20 in stock, if #379's budget for the computer asks for it.",
  ),
  Part(
    "pi5_active_cooler", "Raspberry Pi Active Cooler for the Pi 5",
    "electronics", "chosen", ("build",), (),
    partNumber="SC1148",
    source="https://www.berrybase.de/raspberry-pi-active-cooler-luefter-fuer-raspberry-pi-5",
    priceEur=5.9,
    why={"massG": "only a packed weight (29 g) is given", "dimensionsMm": "the Pi 5's footprint"},
  ),
  Part(
    "pi_ssd_kit_256", "Raspberry Pi SSD Kit, 256 GB (M.2 HAT+ and NVMe)",
    "electronics", "chosen", ("build",), (),
    partNumber="SC1675",
    source="https://www.berrybase.de/raspberry-pi-ssd-kit-fuer-raspberry-pi-5-256gb",
    priceEur=54.9,
    capabilities={"capacityGb": 256, "iops": "40k read, 70k write (4 kB)"},
    why={"massG": "only a packed weight (35 g) is given",
         "dimensionsMm": "a HAT on the Pi 5's footprint"},
    note="Out of stock at every EU seller read; the 512 GB kit (SC1676) is €166.10 in stock at "
         "Reichelt, and the M.2 HAT+ alone (SC1166) €12.50 at BerryBase.",
  ),
  Part(
    "waveshare_can_hat_plus", "Waveshare 2-CH CAN HAT+ (MCP2515, isolated) for the Raspberry Pi",
    "electronics", "chosen", ("build",), (),
    partNumber="27338",
    source="https://eckstein-shop.de/WaveShare-2CH-Isolated-CAN-Bus-Expansion-HAT-for-Raspberry-Pi",
    massG=45,
    priceEur=29.95,
    capabilities={"channels": 2,
                  "controller": "MCP2515 (CAN 2.0)",
                  "bitrate": "1 Mbit/s (its wiki's example)",
                  "pi5": "listed"},
    why={"dimensionsMm": "a HAT on the Pi's footprint"},
    note="Two of the four buses. Its wiki lists the Pi 5 and documents the stack with the CAN "
         "FD HAT: four channels.",
  ),
  Part(
    "waveshare_can_fd_hat", "Waveshare 2-CH CAN FD HAT (MCP2518FD, isolated) for the Raspberry Pi",
    "electronics", "chosen", ("build",), (),
    partNumber="17075",
    source="https://www.welectron.com/Waveshare-17075-2-CH-CAN-FD-HAT_1",
    massG=40,
    priceEur=51.9,
    capabilities={"channels": 2, "controller": "MCP2518FD (CAN FD)", "isolationKv": 5},
    why={"dimensionsMm": "a HAT on the Pi's footprint"},
    note="The other two buses. Its own wiki lists Pis up to the 4B; the Pi 5 stack is the "
         "HAT+'s wiki's. Four USB adapters (`ucan_v1`, €8.99) are the fallback.",
  ),
  Part(
    "realsense_d435", "RealSense D435 depth camera (active IR stereo)",
    "sensor", "chosen", ("build",), (),
    partNumber="D435",
    source="https://www.digikey.de/de/products/detail/realsense/82635AWGDVKPRQ/9926002",
    massG=75,
    dimensionsMm={"length": 90, "width": 25, "height": 25},
    priceEur=340.01,
    capabilities={"fovHDeg": 87,
                  "fovVDeg": 58,
                  "baselineMm": 50,
                  "interface": "USB 3",
                  "maxPowerW": 3.4},
    note="DigiKey DE's kit (961448, with cable and tripod), none in stock; MyBotShop (DE) "
         "€419.99 in 21 days; RealSense's own store $314, backordered 6-8 weeks, import duty "
         "on top. Cognex agreed to buy RealSense on 2026-09-22 (closing in Q4 2026): the "
         "bill's longest lead, and the one to watch.",
  ),
  Part(
    "rplidar_c1", "Slamtec RPLIDAR C1 2D LIDAR",
    "sensor", "chosen", ("build",), (),
    partNumber="RPLIDAR C1",
    source="https://funduinoshop.com/en/electronic-modules/sensors/movement-distance/rplidar-c1-dtof-lidar-3600-laser-range-scanner-12m-ip54-slamtec",
    massG=110,
    priceEur=84.9,
    capabilities={"rangeM": 12,
                  "rateHz": 10,
                  "fovDeg": 360,
                  "supplyV": 5,
                  "currentMa": 230,
                  "startCurrentMa": 800,
                  "interface": "USB/UART"},
    why={"dimensionsMm": "not recorded"},
    note="Funduino (DE) has it in stock; OpenELAB lists it at €79.95 with 4 left from a "
         "warehouse it does not name (it ships from China in 10-20 days otherwise). It draws "
         "800 mA for a moment at start-up.",
  ),
  Part(
    "pi_camera_3", "Raspberry Pi Camera Module 3",
    "sensor", "chosen", ("build",), (),
    partNumber="Camera Module 3",
    source="https://www.berrybase.de/raspberry-pi-camera-module-3-12mp",
    massG=4,
    dimensionsMm={"length": 25, "width": 24, "height": 11.5},
    priceEur=28.9,
    capabilities={"sensor": "IMX708",
                  "fovHDeg": 66,
                  "fovVDeg": 41,
                  "autofocus": "PDAF, ~10 cm to infinity",
                  "powerW": None},
    note="The body's navigation camera. It ships with the old 15-pin cable; a Pi 5 needs "
         "`pi5_camera_cable`. Size and mass from raspberrypi.com's camera documentation; its "
         "draw is not published there. Looked for again for issue #199 (2026-09-14): neither "
         "product brief (RP-008151, RP-009789) states it, the forum thread on it (t=347469) "
         "has one whole-Pi-Zero figure and no camera-alone one, and Arducam's and InnoMaker's "
         "IMX708 CSI sheets omit it too.",
  ),
  Part(
    "pi5_camera_cable", "Raspberry Pi camera cable, standard to mini, 200 mm",
    "electronics", "chosen", ("build",), (),
    partNumber="SC1128",
    source="https://www.berrybase.de/raspberry-pi-camera-cable-standard-mini-200mm",
    priceEur=1.2,
    capabilities={"pins": "22 (Pi 5) to 15 (camera)", "lengthMm": 200},
    why={"massG": "only a shipping weight is given", "dimensionsMm": "a 200 mm flat cable"},
  ),
  Part(
    "tdk_ev_icm42688p", "TDK InvenSense ICM-42688-P evaluation board (IMU)",
    "sensor", "chosen", ("build",), (),
    partNumber="EV_ICM-42688-P",
    source="https://www.digikey.de/de/products/detail/tdk-invensense/EV-ICM-42688-P/18634550",
    priceEur=36.79,
    capabilities={"interface": "I2C, I3C or SPI", "logicV": "1.8 or 3.0"},
    why={"massG": "not published for the board", "dimensionsMm": "not published for the board"},
    note="TDK's own board, a 2×10 header, used with jump wires or soldered (TDK AN-000488): "
         "the one ICM-42688-P board in stock at an EU storefront; the supply of the chip is "
         "short (a Tindie maker raised its price for it).",
  ),
  Part(
    "igus_kcrm_05", "igus igubal KCRM-05 rod end, M5 female, right-hand",
    "structure", "chosen", ("build",), (),
    partNumber="KCRM-05",
    source="https://www.igus.de/product/igubal_KCRM_KCLM?artNr=KCRM-05",
    massG=3.1,
    priceEur=5.43,
    capabilities={"thread": "M5 x 0.8",
                  "boreMm": 5,
                  "staticRadialN": "1200 short-term, 600 long-term"},
    why={"dimensionsMm": "not recorded"},
    note="The ankles' rods: one at each end. igus shows €4.56 ex VAT; €4.37 at ten. A steel "
         "PHSA5 is €2.87 at Dold, and 20 g.",
  ),
  Part(
    "petg_1kg", "PETG filament, 1.75 mm, 1 kg (DAS FILAMENT)",
    "structure", "chosen", ("build",), (),
    partNumber="F10765",
    source="https://dasfilament.de/produkt/petg-filament-175-mm-schwarz-1-kg/",
    massG=1000,
    priceEur=21.0,
    why={"dimensionsMm": "a 1 kg spool"},
    note="The head's shell, the torso's cover, the pack's case and the cameras' and the "
         "LIDAR's mounts.",
  ),
  Part(
    "albright_sw80b549", "Albright SW80B contactor, 12 V coil, with blowouts (100 A, 96 V DC)",
    "power", "chosen", ("build",), (),
    partNumber="SW80B549",
    source="https://www.rijregeling.nl/en/product/albright-contactors-and-accessories/sw80-series/sw80b549-albright-contactor-12v-contactor/",
    massG=400,
    priceEur=82.28,
    capabilities={"coilV": 12,
                  "continuousA": 100,
                  "maxV": 96,
                  "breaksA": "190 A resistive at 96 V"},
    why={"dimensionsMm": "not recorded"},
    note="The e-stop's switch: the motor bus opens when the button on the robot or the "
         "wireless relay breaks its coil circuit. The B version, with blowouts: the plain SW80 "
         "is rated to break load only to 48 V, and the pack charges to 50.4. ⚠ Confirm the "
         "coil is CONTINUOUS-rated before ordering: Albright's sheet rates SW80 coils by class "
         "(continuous 7-13 W down to 25 % duty at 20-30 W) and neither it nor the seller says "
         "which the 549 is; ARC Components lists the SW80B-5 and -35 as 12 V continuous, at "
         "£51.05 ex VAT and 7-9 weeks. €68.00 plus 21 % Dutch VAT as the page shows it; about "
         "350 g and 50 g of blowouts (Albright's sheet). TE's Kilovac LEV100 (190 g) sells to "
         "businesses only, at €127.44 ex VAT.",
  ),
  Part(
    "schneider_xb5as8442", "Schneider Harmony XB5AS8442 emergency stop, 40 mm twist-release "
    "head, 1 NC",
    "electronics", "chosen", ("build",), (),
    partNumber="XB5AS8442",
    source="https://www.reichelt.com/de/en/shop/product/emergency_stop_switch_harmony_xb5_twist_unlock_22_mm_1_nc-382385",
    massG=61,
    priceEur=30.95,
    capabilities={"contact": "1 NC", "mountMm": 22, "ip": "IP66"},
    why={"dimensionsMm": "a 40 mm head on a 22 mm hole"},
    note="One on the robot. Eaton's M22 in three parts (head, adapter, NC contact) is €34.42 "
         "and in stock.",
  ),
  Part(
    "radiomaster_pocket", "RadioMaster Pocket ELRS transmitter (EU LBT)",
    "electronics", "chosen", ("build",), (),
    partNumber="Pocket ELRS EU-LBT",
    source="https://n-factory.de/RadioMaster-Pocket-ELRS-Remote-Control-EU-LBT",
    massG=288,
    priceEur=79.95,
    why={"dimensionsMm": "not published"},
    note="The wireless e-stop's hand unit: a switch on it holds the receiver's relay closed. "
         "It runs on two 18650 cells, not included. A commercial wireless e-stop (Tyro Indus "
         "1S) is €725 ex VAT, the receiver not stated.",
  ),
  Part(
    "radiomaster_er6", "RadioMaster ER6 ELRS PWM receiver (LBT)",
    "electronics", "chosen", ("build",), (),
    partNumber="ER6 ELRS LBT",
    source="https://n-factory.de/RadioMaster-ER6-ELRS-LBT-24-GHz-PWM-Receiver",
    massG=14.5,
    priceEur=31.95,
    capabilities={"failsafe": "at link quality 0 or 1 s without a packet, per channel 988-2012 µs"},
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "pololu_rc_relay", "Pololu RC Switch with Relay (assembled)",
    "electronics", "chosen", ("build",), (),
    partNumber="2804",
    source="https://eckstein-shop.de/Pololu-RC-Switch-with-Relay-Assembled-EN",
    massG=17,
    priceEur=21.36,
    capabilities={"relayA": 10, "threshold": "on above about 1700 µs"},
    why={"dimensionsMm": "not recorded"},
    note="In the contactor's coil circuit, in series with the e-stop button: the receiver's "
         "failsafe at 988 µs opens it, and Pololu's safe-start needs off-then-on to close it "
         "again.",
  ),
  Part(
    "molicel_p28a", "Molicel INR18650-P28A Li-ion cell (18650, flat top)",
    "power", "chosen", ("build",), (),
    partNumber="INR18650-P28A",
    source="https://www.lipo24.de/products/molicel-inr18650-p28a-2800mah-35a-lithium-ionen-akku-3-6v-3-7v",
    massG=48,
    dimensionsMm={"diameter": 18.45, "length": 65},
    priceEur=5.95,
    note="The wireless e-stop's transmitter runs on two.",
  ),
  Part(
    "meanwell_ddr_30l_12", "Mean Well DDR-30L-12 DC-DC converter, 18-75 V in, 12 V 2.5 A, "
    "isolated (DIN rail)",
    "power", "chosen", ("build",), (),
    partNumber="DDR-30L-12",
    source="https://www.reichelt.com/de/en/shop/product/dc_dc_converter_30w_12v_2_5a-256735",
    massG=120,
    priceEur=27.1,
    capabilities={"inputV": "18-75", "outputV": "12 (9-13.2 adjustable)", "outputA": 2.5},
    why={"dimensionsMm": "not recorded"},
    note="The contactor's coil. 30 W covers every class of SW80 coil; the 15 W DDR-15L-12 "
         "(€23.30) holds only a continuous one.",
  ),
  Part(
    "arcol_hs25_47r", "Arcol HS25 wirewound resistor, 47 Ω, 25 W, aluminium housing",
    "power", "chosen", ("build",), (),
    partNumber="HS25 47R F",
    source="https://www.reichelt.com/de/en/shop/product/wirewound_resistor_axial_25_w_47_ohm_1_-233471",
    massG=14,
    priceEur=2.99,
    capabilities={"ohms": 47, "watts": "25 on a heatsink, 12.5 free"},
    why={"dimensionsMm": "not recorded"},
    note="Pre-charge: the drivers' capacitors fill through it, about 1 A at 50.4 V, before the "
         "contactor closes; the sheet gives no pulse rating, only continuous power.",
  ),
  Part(
    "hongfa_hf115f_012", "Hongfa HF115F relay, 12 V coil, 16 A changeover",
    "power", "chosen", ("build",), (),
    partNumber="HF115F-012-1ZS3A",
    source="https://www.reichelt.com/de/en/shop/product/miniature_power_relay_12_v_dc_16_a_1_changeover_contact-260113",
    massG=13,
    priceEur=3.1,
    capabilities={"coil": "12 V, 360 Ω", "maxV": "300 V DC"},
    why={"dimensionsMm": "not recorded"},
    note="Switches the pre-charge resistor in, then out once the contactor has taken the bus; "
         "what it breaks at 60 V DC is a curve on the sheet, not a number.",
  ),
  # ---- the shop's tools and the bench's supply: what a build needs to make
  # its pack, its harness and its printed parts, and to power a leg up first
  Part(
    "korad_ka6005p", "Korad KA6005P lab power supply, 0-60 V, 0-5 A, safety terminals",
    "equipment", "chosen", ("build",), (),
    partNumber="KA6005P",
    source="https://www.welectron.com/Korad-KA6005P-Benchtop-Power-Supply-Safety-Terminals",
    massG=7500,
    priceEur=239.0,
    capabilities={"outputV": "0-60",
                  "outputA": "0-5",
                  "powerW": 300,
                  "modes": "CC/CV, USB and RS232"},
    why={"dimensionsMm": "not recorded"},
    note="A leg's first power-ups on the bench, current-limited. A lab supply cannot take back "
         "what the motors return on landing, so a leg hops on the pack (and a clamp), never on "
         "this. Joy-IT's RD6012 with its 65 V supply and case gives 12 A for €239.28, on an "
         "open-frame supply.",
  ),
  Part(
    "ucan_v1", "UCAN V1.0 USB-CAN adapter (candleLight firmware)",
    "electronics", "chosen", ("build",), (),
    partNumber="OP00014",
    source="https://eckstein-shop.de/ucan-v1-usb-can-adapter",
    priceEur=8.99,
    capabilities={"firmware": "candleLight (Linux gs_usb, socketcan)"},
    why={"massG": "not on the page", "dimensionsMm": "not on the page"},
    note="A leg's bus from a PC, on the bench. The page gives no bitrate; candleLight runs a "
         "bus at 1 Mbit/s. PEAK's PCAN-USB is €199 ex VAT and sold to businesses only.",
  ),
  Part(
    "kweld_kit", "keenlab kWeld spot welder, complete kit, cables assembled",
    "equipment", "chosen", ("build",), (),
    partNumber="kWeld complete kit",
    source="https://www.keenlab.de/index.php/product/kweld-complete-kit/",
    massG=800,
    priceEur=210.63,
    capabilities={"strip": "up to about 0.3 mm", "pulses": "energy-regulated"},
    why={"dimensionsMm": "not recorded"},
    note="€177.00 ex VAT as the page shows it. It needs a source of 800 A: `kweld_kcap`. "
         "Malectrics' V4 bundle is €148.75 without its LiPo; FNIRSI's SWM-10 is €42.90 and "
         "welds 0.1-0.25 mm.",
  ),
  Part(
    "kweld_kcap", "keenlab kCap ultracapacitor module for the kWeld",
    "equipment", "chosen", ("build",), (),
    partNumber="kCap",
    source="https://www.keenlab.de/index.php/product/kweld-ultracapacitor-module/",
    massG=600,
    priceEur=153.51,
    capabilities={"pulseA": 1300, "chargeV": 8.1},
    why={"dimensionsMm": "not recorded"},
    note="€129.00 ex VAT as the page shows it; the bench supply charges it.",
  ),
  Part(
    "engineer_pa09", "Engineer PA-09 crimping tool (JST-PH/GH class, AWG 32-20)",
    "equipment", "chosen", ("build",), (),
    partNumber="PA-09",
    source="https://www.kiwi-electronics.com/en/jst-crimping-tool-pa-09-916",
    massG=135,
    priceEur=47.59,
    why={"dimensionsMm": "not recorded"},
    note="€39.99 ex VAT; ships from the Netherlands. The drivers' CAN and power leads.",
  ),
  Part(
    "knipex_9762145a", "Knipex 97 62 145 A ferrule crimping pliers, 0.25-2.5 mm²",
    "equipment", "chosen", ("build",), (),
    partNumber="97 62 145 A",
    source="https://www.reichelt.com/de/en/shop/product/crimping_pliers_for_end_sleeves_ferrules_-184328",
    massG=170,
    priceEur=30.8,
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "pinecil_v2", "Pine64 Pinecil V2 soldering iron",
    "equipment", "chosen", ("build",), (),
    partNumber="PINECIL-BB2",
    source="https://eleshop.de/pinecil-smart-mini-tragbarer-lotkolben.html",
    massG=28,
    priceEur=39.9,
    capabilities={"powerW": 88, "supply": "USB-C PD 12-20 V or DC 12-24 V"},
    why={"dimensionsMm": "not recorded"},
    note="No supply included; the bench supply runs it off its barrel jack.",
  ),
  Part(
    "unit_ut161e", "UNI-T UT161E true-RMS multimeter",
    "equipment", "chosen", ("build",), (),
    partNumber="UT161E",
    source="https://eleshop.de/uni-t-ut161e.html",
    massG=400,
    priceEur=95.6,
    capabilities={"counts": 22000, "category": "CAT III 1000 V"},
    why={"dimensionsMm": "not recorded"},
    note="The EU name of the UT61E+; Reichelt has it at €109.00.",
  ),
  Part(
    "proxxon_mc5", "Proxxon MicroClick MC 5 torque wrench, 1-5 N·m, 1/4\"",
    "equipment", "chosen", ("build",), (),
    partNumber="23347",
    source="https://www.reichelt.com/de/en/shop/product/micro-click_torque_screwdriver_5_s-91768",
    massG=431,
    priceEur=77.99,
    capabilities={"rangeNm": "1-5", "accuracy": "±6 %"},
    why={"dimensionsMm": "not recorded"},
    note="The actuators' and frame's screws to their torques.",
  ),
  Part(
    "wera_950pks9", "Wera 950 PKS/9 SM N metric hex key set, 1.5-10 mm",
    "equipment", "chosen", ("build",), (),
    partNumber="05133163001",
    source="https://www.reichelt.com/de/en/shop/product/spanner_set_hex_950_pks_smn_9-pieces-169289",
    massG=265,
    priceEur=21.8,
    why={"dimensionsMm": "not recorded"},
  ),
  Part(
    "bambu_p1s", "Bambu Lab P1S 3D printer (printer only)",
    "equipment", "chosen", ("build",), (),
    partNumber="P1S",
    source="https://eu.store.bambulab.com/products/p1s",
    massG=12950,
    priceEur=379.0,
    capabilities={"buildVolumeMm": {"x": 256, "y": 256, "z": 256}, "petg": "ideal (maker)"},
    why={"dimensionsMm": "not recorded"},
    note="Enclosed, for PETG: the head's shell, the torso's cover, the charge port's funnel, "
         "the pack's case and the sensors' mounts. The page says the VAT may change at "
         "checkout. The A1 (open) is €259.",
  ),
)


def by_id() -> dict[str, Part]:
  return {p.id: p for p in PARTS}


def validate(entry: dict, worlds: dict, shelves: tuple = SHELVES) -> list[str]:
  """Every reason one emitted entry is not honest, or `[]`: `worlds` maps
  each robot to the model its entries are read against (the bill's).

  Used by the tests on every committed entry and shown to fail on a bad
  one."""
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
  if not entry["shelves"] or any(s not in shelves for s in entry["shelves"]):
    out.append(f"shelves {entry['shelves']!r} not a non-empty subset of {shelves}")
  if entry["robot"] not in worlds:
    out.append(f"robot {entry['robot']!r} not in {tuple(worlds)}")
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
  if entry["status"] == "chosen" and "build" in entry["shelves"] \
      and entry["kind"] != "fastener" and not entry["feeds"] \
      and entry["partNumber"] is None:
    out.append("a chosen build part with neither a part number nor a feed "
               "is a part nobody can point at")
  return out


def mismatches(fixture: dict) -> list[str]:
  """Every feed whose sim value disagrees with the part's own number.

  THE pin: an RS04's 120 N·m and its drivers' `forcerange` are one fact,
  and a change to either side lands here, in the suite, rather than on the
  website as a spec sheet that quietly stopped describing the sim."""
  out = []
  for p in fixture["parts"]:
    for f in p["feeds"]:
      if "expect" in f and f["value"] != f["expect"]:
        out.append(f"{p['id']}: {f['constant']} is {f['value']}, the part "
                   f"says {f['expect']} {f['unit']}")
  return out


# ---- a bill of materials (#379): what a bill is built and checked with -------

#: How a line's euros become dollars: the ECB's reference rate on RATE_DATE.
USD_PER_EUR = 1.1403
RATE_DATE = "2026-09-25"
RATE_SOURCE = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
#: Every price is what a buyer in Germany pays, VAT in (a seller quoting ex
#: VAT or dollars is converted, and the line says so). A business that
#: reclaims the VAT spends 16 % less.
VAT = 0.19
#: When `PARTS`' prices were read off their pages.
CHECKED = "2026-09-28"
#: On top of every line: shipping, the drift between reading a price and
#: ordering, and the part bought twice. A decision, not a measurement.
CONTINGENCY = 0.15


@dataclass
class Line:
  """One line of the bill: `quantity` of a `build`-shelf part (in its
  `priceFor` units), bought for `group`. A part with no price -- nothing
  designed yet -- is carried at an ALLOWANCE, a sum set aside for the whole
  line and never a price, with the `basis` it rests on. `leadTime` is the
  seller's own words on the bill's reading date; a null one needs its
  `why`."""
  part: str
  quantity: int
  group: str
  leadTime: str | None
  allowanceEur: float | None = None
  basis: str = ""
  why: dict = field(default_factory=dict)
  note: str = ""


def validate_line(line: dict, parts: dict[str, dict], groups: tuple) -> list[str]:
  """Every reason one emitted line of a bill is not honest, or `[]`;
  `groups` the bill's sections."""
  out: list[str] = []
  p = parts.get(line["part"])
  if p is None:
    return [f"no part {line['part']!r}"]
  if "build" not in p["shelves"]:
    out.append(f"{line['part']} is not on the build shelf")
  if not (isinstance(line["quantity"], int) and line["quantity"] > 0):
    out.append("quantity must be a positive whole number")
  if line["group"] not in groups:
    out.append(f"group {line['group']!r} not in {groups}")
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


def _day(text: str) -> bool:
  """Whether `text` is a calendar day as the site reads one: `2026-09-28`."""
  try:
    return date.fromisoformat(text).isoformat() == text
  except ValueError:
    return False


def bom(parts: dict[str, dict], lines: tuple, groups: tuple, checked: str,
        budget_usd: float | None = None) -> dict:
  """A bill as data: every line priced off its part (or its allowance), and
  the totals against the budget -- summed here, never typed: its lines,
  sections, the day its oldest price was read (`checked`), and its budget,
  or None where it has none. The rate, the VAT and the contingency are one
  for every bill."""
  if not _day(checked):
    # the site prints it as a day, and a page that cannot read it renders nothing
    raise ValueError(f"checked is the day the bill's oldest price was read, a calendar "
                     f"day: {checked!r}")
  src = lines
  lines = []
  for ln in src:
    p = parts.get(ln.part, {})
    each = p.get("priceEur")
    if each is not None and ln.allowanceEur is not None:
      raise ValueError(f"{ln.part} ({ln.group}): a priced part carries its "
                       "price, not an allowance")
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
    "budgetUsd": budget_usd, "usdPerEur": USD_PER_EUR, "rateDate": RATE_DATE,
    "rateSource": RATE_SOURCE, "vat": VAT, "checked": checked,
    "contingency": CONTINGENCY, "groups": list(groups), "lines": lines,
    "totals": {
      "sourcedEur": sourced, "allowanceEur": allowed,
      "contingencyEur": contingency, "totalEur": eur,
      "totalUsd": round(eur * USD_PER_EUR, 2),
      "budgetEur": (None if budget_usd is None
                    else round(budget_usd / USD_PER_EUR, 2)),
      "byGroup": {g: total(lambda ln, g=g: ln["group"] == g) for g in groups},
    },
  }


def _eur(v: float | None) -> str:
  return "—" if v is None else f"{v:,.2f}"


def _cell(text: str) -> str:
  """Text for a Markdown table cell: a `|` would end the cell."""
  return text.replace("|", "\\|")


#: A line's feeds, as the bill shows them: the first few constants' names.
FEEDS_SHOWN = 3


def bom_markdown(fixture: dict, key: str = "build", read_on: str | None = None) -> str:
  """A bill as Parts.md shows it: one row a line under its group's
  subtotal, the allowances' bases as notes, and the totals. `key` names
  the fixture's bill; `read_on` says when its prices were read, in words,
  where they were read on more days than its one `checked`."""
  b, parts = fixture[key], {p["id"]: p for p in fixture["parts"]}
  t = b["totals"]
  rows = [f"Prices and lead times read {read_on or b['checked']}; € incl. {b['vat']:.0%} "
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
      name = (f"[{_cell(p['name'])}]({p['source']})" if p["source"]
              else _cell(p["name"]))
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
      lead = _cell(ln["leadTime"] if ln["leadTime"] is not None else
                   f"unknown: {ln['why']['leadTime']}")
      rows.append(f"| {name} | {ln['quantity']} | {each} | {_eur(ln['lineEur'])} "
                  f"| {lead} | {', '.join(feeds) or '—'} |")
  rows += ["", "| | € | $ |", "|---|---|---|",
           f"| priced lines | {_eur(t['sourcedEur'])} | |",
           f"| allowances (nothing designed yet) | {_eur(t['allowanceEur'])} | |",
           f"| contingency, {b['contingency']:.0%} | {_eur(t['contingencyEur'])} | |",
           f"| **total** | **{_eur(t['totalEur'])}** | **{_eur(t['totalUsd'])}** |"]
  if b["budgetUsd"] is not None:
    rows.append(f"| budget | {_eur(t['budgetEur'])} | {_eur(b['budgetUsd'])} |")
  if notes:
    rows += ["", "**The allowances**, each a sum set aside, never a price:", *notes]
  return "\n".join(rows)


def with_bill(doc: str, fixture: dict, markers: tuple[str, str], key: str = "build",
              read_on: str | None = None) -> str:
  """Parts.md with a bill re-rendered between its markers."""
  first, last = markers
  head, start, rest = doc.partition(first)
  _, end, tail = rest.partition(last)
  if not start or not end:
    raise SystemExit(f"Parts.md has no bill markers ({first!r} ... {last!r})")
  return f"{head}{first}\n{bom_markdown(fixture, key, read_on)}\n{last}{tail}"
