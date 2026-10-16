"""The humanoid's bill of materials (issue #507): what it is built from,
kept as `rack/catalog.py` keeps the quadruped's and checked by the same
rules -- a part's facts once, every feed READ off `models/humanoid.xml` or
the live constant (an `expect` where the sheet's number IS the sim's), a
number nobody publishes `null` with its why, and a part nothing has been
drawn for an ALLOWANCE with its basis, never a price. Parts.md's "The
humanoid body" carries the rendering (`uv run python -m
pluggybot.humanoid.bill`), and a test fails if the two differ.

A part the quadruped's bill already buys keeps its facts there
(`_shared`): one part number, one price and one mass. The shop's tools,
the gantry and the bench rig are that bill's too, and not bought twice.

Not in `protocol/parts.json`: the site's parts page shows the served
robot's build (#185), and this bill joins it when the humanoid is served.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.rack import catalog as cat
from pluggybot.rack.catalog import (PRINTED, Feed, Line, Part, actuator, camera, code,
                                    code_entry, joint, same)

ROBOT = "humanoid"
WORLDS = {ROBOT: hm.MODEL_XML}
SHELVES = ("build",)
#: When this bill's own lines were read; the parts it shares with the
#: quadruped's carry that bill's reading (`cat.CHECKED`).
CHECKED = f"2026-10-16 (the parts shared with the quadruped's bill: {cat.CHECKED})"
GROUPS = ("actuation", "grippers", "power", "compute", "sensors", "frame",
          "safety", "spares")
BOM_START = ("<!-- humanoid bom: rendered by `uv run python -m pluggybot.humanoid.bill` "
             "from humanoid/bill.py's LINES; edit those, not this -->")
BOM_END = "<!-- /humanoid bom -->"

OPENELAB = "https://openelab.io/products/"
#: OpenELAB's own words on its RobStride pages, read 2026-10-16: the RS04 a
#: pre-order, shipping from Shenzhen; the others in its page data's
#: inventory or ordered through it from China.
OE_LEAD = ("pre-order where none is in local stock; ships from Shenzhen "
           "(OpenELAB's terms)")
UNDRAWN = {"leadTime": "nothing drawn to quote"}


def driven_by(motor) -> list[str]:
  """Every joint a part drives, both sides."""
  return [j for j in hm.JOINT_NAMES if hm.CHOSEN.motor(j).name == motor.name]


def _bodies(motor) -> tuple[str, ...]:
  return tuple(f"{j.split('_', 1)[0]}_{hm.BODY_OF[hm.kind(j)]}" for j in driven_by(motor))


def _actuator(pid: str, motor, page: str, size: dict, inertia: float, note: str) -> Part:
  """A RobStride part as the bill buys it: its sheet's facts, and the
  model's drivers read back -- its clip, its rotor through its gear."""
  joints = driven_by(motor)
  name = motor.name.split()[-1]
  return Part(
    pid, f"{motor.name} joint actuator ({motor.ratio:g}:1 planetary, FOC driver, "
    "CAN, MIT mode), 48 V", "motor", "chosen", SHELVES, _bodies(motor),
    partNumber=name, source=OPENELAB + page, massG=round(motor.mass * 1000, 1),
    dimensionsMm=size, priceEur=motor.price_eur, quantity=len(joints), robot=ROBOT,
    capabilities={"peakNm": motor.peak_torque, "ratedNm": motor.rated_torque,
                  "stallRatedNm": round(hp.stall_torque(motor), 2),
                  "noLoadRpm48V": round(motor.noload_speed * 60 / (2 * 3.141592653589793), 1),
                  "ktNmPerArms": round(motor.kt * 2 ** 0.5, 3),
                  "windingLimitC": motor.max_winding_c},
    feeds=(
      Feed(f"{joints[0]}.forcerange", same(*(actuator(j, "forcerange") for j in joints)),
           "N·m", f"every {name} driver's clip: {', '.join(joints)}",
           expect=motor.peak_torque),
      Feed(f"{joints[0]}.armature", same(*(joint(j, "armature") for j in joints)),
           "kg·m²", "the sheet's output-side inertia", expect=inertia),
      code(f"humanoid.parts.{name}.rated_torque", "N·m",
           "continuous, turning, on the sheet's heat sink"),
      code(f"humanoid.parts.{name}.stall_torque", "N·m", "continuous, held still"),
      code(f"humanoid.parts.{name}.saturation_torque", "N·m",
           "the torque-speed line at standstill (DERIVED off the sheet's 48 V table)"),
    ),
    note=note + " Its facts are RobStride's own manuals and spec sheet "
    "(`humanoid.parts.RS_DOCS`); the link is where it is bought.")


def _shared(pid: str, *, usedBy=(), feeds=(), quantity: int = 1, note: str | None = None,
            id: str | None = None) -> Part:
  """A part the quadruped's bill already buys: its facts from there, its
  use and its feeds this body's."""
  p = cat.by_id()[pid]
  return replace(p, id=id or pid, robot=ROBOT, usedBy=tuple(usedBy), feeds=tuple(feeds),
                 quantity=quantity, note=p.note if note is None else note)


def _structure_g(names: tuple[str, ...]):
  """A reader: the named fabricated parts' mass as drawn, g, all of each."""
  def read(_spec):
    s = hm.structure()
    return round(sum(s[n][0].kg * s[n][1] for n in names) * 1000, 3)
  return read


def _brackets_g(_spec) -> float:
  return round(sum(hm._bracket(hm.CHOSEN.motor(j)).kg for j in hm.JOINT_NAMES) * 1000, 3)


def _fabricated(pid: str, name: str, parts: tuple[str, ...], usedBy=(), note: str = "",
                printed: bool = False) -> Part:
  """A part made to a drawing (`model.structure`): its mass its volume's,
  its price an allowance in the bill until the drawing is quoted."""
  s = hm.structure()
  read = _structure_g(parts)
  why = {"partNumber": PRINTED if printed else "made to a drawing not yet drawn",
         "source": PRINTED if printed else "made to a drawing not yet drawn",
         "dimensionsMm": "the drawing's; `humanoid.model.structure` sizes it",
         "priceEur": "printed: filament only, unpriced" if printed
         else "nothing drawn to quote: an allowance in the bill"}
  return Part(pid, name, "structure", "chosen", SHELVES, tuple(usedBy),
              massG=read(None), quantity=sum(s[n][1] for n in parts), robot=ROBOT,
              why=why, note=note,
              feeds=(Feed(f"humanoid.model.structure[{', '.join(parts)}]", read, "g",
                          "their volume's mass, the fasteners' and wiring's allowance "
                          "on it (`humanoid.model.ALLOWANCE`)", where="code"),))


PARTS: tuple[Part, ...] = (
  # ---- actuation ---------------------------------------------------------------
  _actuator("rs04", hp.RS04, "robstride04-qdd-120n-m-integrated-joint-motor-module",
            {"diameter": 120, "length": 52}, 0.04,
            "The hips' pitch and roll, the knees and the ankles' pitch: wherever the "
            "body's weight passes. K-Scale's open K-Bot uses it at hip pitch and knee. "
            "The sheet's mass is 1420 g in its manual and 1510 on its page (the "
            "heavier here), and its 120 N·m is held 3 s turning in one and 10 s in "
            "the other."),
  _actuator("rs03", hp.RS03, "robstride03-qdd-60n-m-integrated-joint-motor-module",
            {"diameter": 98, "length": 54.1}, 0.02,
            "The hips' yaw, the ankles' roll and the shoulders' pitch and roll. "
            "K-Bot uses it at hip roll and yaw. 880 g in its manual, 928 on the "
            "sheet's page."),
  _actuator("rs02", hp.RS02, "robstride02-qdd-17n-m-joint-motor-module-dual-encoders",
            {"diameter": 78.5, "length": 41.5}, 4.2e-3,
            "The upper arms' twist and the elbows. K-Bot uses it at shoulder yaw "
            "and elbow. 380 g in its manual, 418 on the sheet's page."),
  _actuator("rs00", hp.RS00, "robstride00-qdd-14n-m-integrated-joint-motor-module",
            {"diameter": 57, "length": 51}, 1e-3,
            "The wrists' pitch: the gripper and what it holds, ahead of it."),
  _actuator("rs05", hp.RS05, "robstride05-qdd-joint-motor-module",
            {"diameter": 46, "length": 44}, 7e-4,
            "The wrists' roll, after the pitch: it turns only the gripper. 191 g in "
            "its manual, 243 on the sheet's page."),
  _shared("amass_xt30_22_pair", usedBy=(hm.ROBOT_ROOT,), quantity=26,
          note="Power and CAN in one, a driver's (RobStride's drivers take an XT30 "
               "and a GH1.25)."),
  _shared("lapp_unitronic_can", usedBy=(hm.ROBOT_ROOT,), quantity=15,
          note="Four buses, six drivers each, daisy-chained."),
  _shared("silicone_wire_14awg", usedBy=(hm.ROBOT_ROOT,), quantity=30),
  # ---- grippers ----------------------------------------------------------------
  Part(
    "dex1_1", "Unitree Dex1-1 parallel gripper (Standard)", "actuator", "chosen",
    SHELVES, tuple(f"{s}_hand" for s in hm.SIDES), partNumber="Dex1-1",
    source="https://openelab.io/de/products/unitree-dex1-1-robotic-gripper",
    massG=550, dimensionsMm={"length": 143, "width": 78, "height": 67},
    priceEur=hp.DEX1_1.price_eur, quantity=2, robot=ROBOT,
    capabilities={"strokeMm": 90, "gripForceN": [5, 120], "speedMmS": [50, 375],
                  "closeS": 0.24, "fingerMm": 80, "supplyV": [24, 64], "peakA": 3.5,
                  "bus": "RS-485, 6 Mbps, 1000 Hz", "reports": "angle, speed, torque, "
                  "temperature; fingertip to 0.1 mm off a 15-bit absolute encoder",
                  "commands": "angle, speed, torque, stiffness, damping"},
    feeds=(
      Feed("left_gripper.forcerange", same(*(actuator(f"{s}_gripper", "forcerange")
                                             for s in hm.SIDES)), "N",
           "the actuated jaw's clip: twice the grip force, each pad squeezing half",
           expect=2 * 120.0),
      Feed(f"left_{hm.JAWS[0]}.range", same(*(joint(f"{s}_{hm.JAWS[0]}", "range", 1)
                                              for s in hm.SIDES)), "m",
           "a jaw's travel: half the stroke", expect=0.045),
      code("humanoid.parts.DEX1_1.grip_force_n", "N", "", expect=120.0),
      code("humanoid.parts.DEX1_1.position_lsb", "m", "the fingertip's resolution",
           expect=0.0001),
    ),
    note="Unitree's maker page (unitree.com/Dex1-1) gives every number; its flange, "
         "its rotor and its torque estimate's resolution are unpublished. OpenELAB "
         "shows both incl. and excl. VAT beside the price. Chosen over the ROBOTIS "
         "RH-P12-RN(A) (every register published; €3,912.36 incl. VAT a gripper) and "
         "Inspire's EG2-5C, the T2's own (0-20 N, its registers unpublished).",
  ),
  Part("rs485_adapter", "A USB to RS-485 adapter that runs at the grippers' 6 Mbps",
       "electronics", "candidate", SHELVES, (hm.ROBOT_ROOT,), quantity=1, robot=ROBOT,
       why={"partNumber": "none chosen", "source": "none chosen",
            "massG": "none chosen", "dimensionsMm": "none chosen",
            "priceEur": "none chosen: an allowance in the bill"},
       note="The Dex1-1 talks RS-485 at 6 Mbps through Unitree's own library; a "
            "Pi's CAN HATs do not."),
  # ---- power -------------------------------------------------------------------
  _shared("molicel_p45b", usedBy=(hm.ROBOT_ROOT,), quantity=hp.PACK.cells,
          feeds=(code("humanoid.parts.PACK.wh", "Wh", f"{hp.PACK.series}S{hp.PACK.parallel}P"),
                 code("humanoid.parts.PACK.kg", "kg", "the cells, the BMS, the strip and "
                      "the printed case"))),
  _shared("jbd_bms_12s_120a", usedBy=(hm.ROBOT_ROOT,)),
  _shared("hilumin_strip_10m"),
  _shared("meanwell_ddr_60l_5", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS_W", "compute + cameras", "W"),)),
  _shared("amass_xt90s", quantity=3),
  _shared("victron_midi_60a"),
  _shared("victron_midi_holder"),
  _shared("silicone_wire_10awg", quantity=4),
  _shared("noeifevo_charger_12s",
          note="The bench charger: 216 W at the pack's nominal 43.2 V, 3.6 h from "
               "empty. A dock of its own is not this issue's."),
  # ---- compute -----------------------------------------------------------------
  _shared("raspberry_pi_5_quad", id="raspberry_pi_5", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "pi5 (packed)", "kg",
                            "its packed weight, an upper bound"),)),
  _shared("pi5_active_cooler"),
  _shared("pi_ssd_kit_256"),
  _shared("waveshare_can_hat_plus", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "waveshare_can_hat_plus", "kg"),)),
  _shared("waveshare_can_fd_hat", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "waveshare_can_fd_hat", "kg"),)),
  # ---- sensors -----------------------------------------------------------------
  _shared("quad_realsense_d435", id="realsense_d435", usedBy=("depth_cam_body",),
          feeds=(Feed("depth_eye.fovy", camera("depth_eye"), "°",
                      "the depth field's vertical angle", expect=58),
                 code("humanoid.parts.D435_KG", "kg"),
                 code("humanoid.model.CHOSEN.depth_pitch_deg", "°",
                      "fixed in the head, pitched at the counter and the floor ahead"))),
  _shared("quad_rplidar_c1", id="rplidar_c1", usedBy=("lidar_mount",),
          feeds=(code("humanoid.parts.LIDAR.kg", "kg"),
                 code_entry("humanoid.parts.ELECTRONICS_W", "lidar", "W", expect=1.15))),
  _shared("pi_camera_3", usedBy=("nav_cam_body",),
          feeds=(Feed("nav_eye.fovy", camera("nav_eye"), "°", "the tags' camera",
                      expect=41),)),
  _shared("pi5_camera_cable", usedBy=("nav_cam_body",)),
  _shared("tdk_ev_icm42688p", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code("perception.imu.GYRO_NOISE", "rad/s/√Hz", "its sheet's noise"),)),
  # ---- frame -------------------------------------------------------------------
  Part("humanoid_brackets", "The motors' brackets: two 5 mm 6082 plates a joint, each "
       "its motor's diameter square, machined, with their fasteners", "structure",
       "chosen", SHELVES, (hm.ROBOT_ROOT,), massG=_brackets_g(None), quantity=48,
       robot=ROBOT,
       why={"partNumber": "made to a drawing not yet drawn",
            "source": "made to a drawing not yet drawn",
            "dimensionsMm": "each its motor's diameter square (`humanoid.model._bracket`)",
            "priceEur": "nothing drawn to quote: an allowance in the bill"},
       feeds=(Feed("humanoid.model._bracket", _brackets_g, "g",
                   "all forty-eight, riding each motor's housing in the model "
                   "(`model._housing`)", where="code"),)),
  _fabricated("humanoid_trunk_plates", "The pelvis's and the torso's frame: 3 mm 6082 "
              "plates", ("pelvis_side", "pelvis_deck", "torso_side", "torso_back",
                         "torso_top"), usedBy=(hm.ROBOT_ROOT,)),
  _fabricated("humanoid_leg_plates", "The thighs' and shanks' side plates, 5 and 4 mm "
              "6082", ("thigh_plate", "shank_plate"),
              usedBy=tuple(f"{s}_{b}" for s in hm.SIDES for b in ("thigh", "shank"))),
  _fabricated("humanoid_soles", "The soles: a 6 mm 6082 plate and its 5 mm rubber",
              ("sole_plate", "sole_rubber"), usedBy=tuple(f"{s}_foot" for s in hm.SIDES)),
  _fabricated("humanoid_arm_tubes", "The upper arms and forearms: 40 x 2 and 35 x 2 mm "
              "6082 round tube, cut to length", ("upper_arm_tube", "forearm_tube"),
              usedBy=tuple(f"{s}_{b}" for s in hm.SIDES for b in ("upper_arm", "forearm"))),
  _fabricated("humanoid_ankle_rods", "The ankles' parallelogram rods: 8 x 1.5 mm tube, "
              "an M5 rod end at each end", ("ankle_rod",),
              usedBy=tuple(f"{s}_shank" for s in hm.SIDES)),
  _fabricated("humanoid_gripper_adapters", "The plates the grippers bolt to on the "
              "wrists' roll, 6 mm 6082", ("gripper_adapter",),
              usedBy=tuple(f"{s}_hand" for s in hm.SIDES)),
  _fabricated("humanoid_shells", "The head's shell and the torso's front cover, "
              "printed in PETG", ("head_shell", "torso_cover"), usedBy=(hm.ROBOT_ROOT,),
              printed=True),
  _shared("igus_kcrm_05", usedBy=tuple(f"{s}_shank" for s in hm.SIDES), quantity=4,
          note="The ankles' rods: one at each end."),
  _shared("petg_1kg", quantity=2,
          note="The head's shell, the torso's cover, the pack's case and the "
               "cameras' and the LIDAR's mounts."),
  # ---- safety ------------------------------------------------------------------
  _shared("albright_sw80b549", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS_W", "contactor", "W",
                            "its coil held, through the 12 V converter"),)),
  _shared("schneider_xb5as8442", usedBy=(hm.ROBOT_ROOT,)),
  _shared("radiomaster_pocket"),
  _shared("radiomaster_er6", usedBy=(hm.ROBOT_ROOT,)),
  _shared("pololu_rc_relay", usedBy=(hm.ROBOT_ROOT,)),
  _shared("molicel_p28a", quantity=2),
  _shared("meanwell_ddr_30l_12", usedBy=(hm.ROBOT_ROOT,)),
  _shared("arcol_hs25_47r", usedBy=(hm.ROBOT_ROOT,)),
  _shared("hongfa_hf115f_012", usedBy=(hm.ROBOT_ROOT,)),
)

#: What the allowances rest on: a machined 6082 part at a job shop, as the
#: quadruped's bill sizes its legs' (€800 for twenty-four parts: €33.33 a
#: part), and AluFritze's configurator for a plain plate (€20.02 for 300 x
#: 150 mm in 3 mm).
MACHINED_EUR = 800.0 / 24

LINES: tuple[Line, ...] = (
  # ---- actuation ---------------------------------------------------------------
  Line("rs04", 8, "actuation", OE_LEAD, note="the hips' pitch and roll, the knees, "
       "the ankles' pitch"),
  Line("rs03", 8, "actuation", OE_LEAD, note="the hips' yaw, the ankles' roll, the "
       "shoulders' pitch and roll"),
  Line("rs02", 4, "actuation", OE_LEAD, note="the upper arms' twist and the elbows"),
  Line("rs00", 2, "actuation", OE_LEAD, note="the wrists' pitch"),
  Line("rs05", 2, "actuation", OE_LEAD, note="the wrists' roll"),
  Line("amass_xt30_22_pair", 30, "actuation",
       "dispatched within 2-3 days, UPS or DHL (3DPTronics, Italy)",
       note="one a driver, four spare"),
  Line("lapp_unitronic_can", 15, "actuation", "176 m in stock, delivered on 2026-09-30 (Conrad)"),
  Line("silicone_wire_14awg", 30, "actuation", "available now, 1-3 working days (MyLipo)",
       note="fifteen metres red, fifteen black"),
  # ---- grippers ----------------------------------------------------------------
  Line("dex1_1", 2, "grippers", "pre-order: estimated dispatch 10-20 days after "
       "ordering; Standard marked sold out (OpenELAB)"),
  Line("rs485_adapter", 1, "grippers", None, why={"leadTime": "none chosen"},
       allowanceEur=30.0,
       basis="an FTDI-class USB to RS-485 adapter good for 6 Mbps; none chosen"),
  # ---- power -------------------------------------------------------------------
  Line("molicel_p45b", hp.PACK.cells, "power", "ready to ship in 1-2 days (akkuteile.de)",
       note=f"{hp.PACK.series}S{hp.PACK.parallel}P, {hp.PACK.wh:.0f} Wh"),
  Line("jbd_bms_12s_120a", 1, "power", "short stock, 1-5 working days (bikebattery.de)"),
  Line("hilumin_strip_10m", 1, "power", "in stock, 1-3 working days (AKKUman)"),
  Line("meanwell_ddr_60l_5", 1, "power", "in stock, 1-2 business days (Reichelt)"),
  Line("amass_xt90s", 3, "power", "in stock, 1-2 working days (MHM)"),
  Line("victron_midi_60a", 1, "power", "in stock, 1-3 working days (Klimaworld)"),
  Line("victron_midi_holder", 1, "power", "ships in 1-3 working days (Offgridtec)"),
  Line("silicone_wire_10awg", 4, "power", "in stock, 1-3 days (Skeries)"),
  Line("noeifevo_charger_12s", 1, "power",
       "3-7 working days from its Polish warehouse, 7-15 from China; the page "
       "does not say which"),
  # ---- compute -----------------------------------------------------------------
  Line("raspberry_pi_5", 1, "compute", "in stock, 1-3 days (BerryBase)"),
  Line("pi5_active_cooler", 1, "compute", "in stock, 1-3 days (BerryBase)"),
  Line("pi_ssd_kit_256", 1, "compute", "not deliverable now (BerryBase); the 512 GB kit "
       "is in stock at Reichelt for €166.10"),
  Line("waveshare_can_hat_plus", 1, "compute", "in stock, 1-3 days (Eckstein)"),
  Line("waveshare_can_fd_hat", 1, "compute", "in stock (short supply), 1-3 business days "
       "(Welectron)"),
  # ---- sensors -----------------------------------------------------------------
  Line("realsense_d435", 1, "sensors", "none in stock, the maker's standard lead 14 "
       "weeks (DigiKey DE); 21 days at MyBotShop for €419.99"),
  Line("rplidar_c1", 1, "sensors", "1-3 business days (Funduino)"),
  Line("pi_camera_3", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("pi5_camera_cable", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("tdk_ev_icm42688p", 1, "sensors", "138 in stock, the maker's standard lead 16 "
       "weeks (DigiKey DE)"),
  # ---- frame -------------------------------------------------------------------
  Line("humanoid_brackets", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(48 * MACHINED_EUR, 2),
       basis="forty-eight machined 6082 plates with their fasteners, at the "
             "quadruped's bill's rate for its legs' (€800 for twenty-four parts)"),
  Line("humanoid_trunk_plates", 1, "frame", None, why=UNDRAWN, allowanceEur=300.0,
       basis="eight 3 mm 6082 plates, as the quadruped's frame (€300 for six to "
             "eight): AluFritze's configurator prices a plain 300 x 150 mm one at "
             "€20.02; a plate with holes is priced from its drawing"),
  Line("humanoid_leg_plates", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(8 * MACHINED_EUR, 2),
       basis="eight machined side plates at the quadruped's bill's rate a part"),
  Line("humanoid_soles", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(2 * MACHINED_EUR + 20.0, 2),
       basis="two machined sole plates at the quadruped's bill's rate a part, and "
             "€20 of 5 mm rubber sheet"),
  Line("humanoid_arm_tubes", 1, "frame", None, why=UNDRAWN, allowanceEur=40.0,
       basis="0.8 m of 6082 round tube in two sizes, cut to length; not priced for "
             "this bill"),
  Line("humanoid_ankle_rods", 1, "frame", None, why=UNDRAWN, allowanceEur=20.0,
       basis="two 0.34 m rods cut from 8 mm tube and threaded M5 at each end"),
  Line("humanoid_gripper_adapters", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(2 * MACHINED_EUR, 2),
       basis="two machined plates at the quadruped's bill's rate a part, drawn when "
             "the gripper's flange is known (Unitree does not publish it)"),
  Line("igus_kcrm_05", 4, "frame", "ready to ship in 24 hours (igus)"),
  Line("petg_1kg", 2, "frame", None,
       why={"leadTime": "in stock; the page and its terms give no delivery time"}),
  # ---- safety ------------------------------------------------------------------
  Line("albright_sw80b549", 1, "safety", None,
       why={"leadTime": "the page states none (it can be put in the cart)"}),
  Line("schneider_xb5as8442", 1, "safety", "available on 2026-10-01 (Reichelt)"),
  Line("radiomaster_pocket", 1, "safety", "ships the same day (n-Factory)"),
  Line("radiomaster_er6", 1, "safety", "ships the same day (n-Factory)"),
  Line("pololu_rc_relay", 1, "safety", "about 8-10 days (Eckstein)"),
  Line("molicel_p28a", 2, "safety", "available, 1-3 days (LiPo24)", note="the transmitter's"),
  Line("meanwell_ddr_30l_12", 1, "safety", "in stock, 1-2 business days (Reichelt)",
       note="the contactor's coil"),
  Line("arcol_hs25_47r", 1, "safety", "in stock, 1-2 business days (Reichelt)",
       note="pre-charge"),
  Line("hongfa_hf115f_012", 1, "safety", "in stock, 1-2 business days (Reichelt)",
       note="pre-charge"),
  # ---- spares ------------------------------------------------------------------
  Line("rs04", 2, "spares", OE_LEAD),
  Line("rs03", 1, "spares", OE_LEAD),
  Line("rs02", 1, "spares", OE_LEAD),
  Line("molicel_p45b", 4, "spares", "ready to ship in 1-2 days (akkuteile.de)"),
  Line("victron_midi_60a", 2, "spares", "in stock, 1-3 working days (Klimaworld)"),
)


def build() -> dict:
  """Every feed read off the humanoid's model and the live constants: the
  catalog's own reading (`rack.catalog.build`), on this bill's parts."""
  import mujoco
  spec = mujoco.MjSpec.from_file(WORLDS[ROBOT])
  parts = [p.as_dict(spec) for p in PARTS]
  ids = [p["id"] for p in parts]
  assert len(ids) == len(set(ids)), "duplicate part id"
  return {
    "worlds": dict(WORLDS), "shelves": list(SHELVES), "parts": parts,
    "humanoid": cat.bom({p["id"]: p for p in parts}, lines=LINES, groups=GROUPS,
                        checked=CHECKED, budget_usd=None),
  }


def problems(fixture: dict) -> dict:
  """Every dishonest part or line, by id, and every feed the sim
  disagrees with: `{}` when the bill may be written."""
  parts = {p["id"]: p for p in fixture["parts"]}
  out = {pid: r for pid, p in parts.items()
         if (r := cat.validate(p, worlds=WORLDS, shelves=SHELVES))}
  out.update({f"line:{ln['part']}": r for ln in fixture["humanoid"]["lines"]
              if (r := cat.validate_line(ln, parts, groups=GROUPS))})
  if off := cat.mismatches(fixture):
    out["mismatches"] = off
  return out


def with_bill(doc: str, fixture: dict) -> str:
  return cat.with_bill(doc, fixture, key="humanoid", markers=(BOM_START, BOM_END))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--doc", default="docs/Parts.md",
                      help="the doc whose humanoid bill is re-rendered")
  args = parser.parse_args()
  fixture = build()
  if bad := problems(fixture):
    raise SystemExit(f"refusing to write a dishonest bill: {bad}")
  doc = Path(args.doc)
  doc.write_text(with_bill(doc.read_text(), fixture))
  t = fixture["humanoid"]["totals"]
  print(f"{doc}: {len(fixture['parts'])} parts, {len(LINES)} lines, "
        f"€{t['totalEur']:,.2f} (${t['totalUsd']:,.2f})")


if __name__ == "__main__":
  main()
