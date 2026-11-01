"""The humanoid's bill of materials (issues #379, #507): what it is built
from, and the build the site's parts page shows (#185) -- checked by the
catalog's rules (`humanoid/catalog.py`): a part's facts once, every feed
READ off `models/humanoid.xml` or the live constant (an `expect` where the
sheet's number IS the sim's), a number nobody publishes `null` with its
why, and a part nothing has been drawn for an ALLOWANCE with its basis,
never a price. `uv run python -m pluggybot.humanoid.bill` writes Parts.md's
"The humanoid body" bill and `protocol/parts.json`, and a test fails if
either differs.

A part bought off a seller's page keeps its facts in the catalog
(`catalog.PARTS`, `_bought`): one part number, one price and one mass,
`humanoid.parts` reading the same masses. The shop's tools are bought here
too: the humanoid is the one build since the quadruped's went (#559). ⚠
NOT IN IT YET: a gantry to hang a 43 kg body from and a bench rig for its
first leg, neither drawn; the quadruped's were sized for 10 kg.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid import catalog as cat
from pluggybot.humanoid.catalog import (PRINTED, Feed, Line, Part, actuator, camera, code,
                                        code_entry, joint, same)

ROBOT = "humanoid"
WORLDS = {ROBOT: hm.MODEL_XML}
SHELVES = ("build",)
#: The days this bill's prices were read: its own parts, its charging lines,
#: and the catalog's parts (the shop's among them) at the catalog's reading.
#: The site prints the oldest, a day, as "on or after"; Parts.md says all three.
READ_ON = ("2026-10-16", "2026-10-30", cat.CHECKED)
CHECKED = (f"{READ_ON[0]}, the charging lines {READ_ON[1]} (the catalog's parts, the "
           f"shop's among them: {READ_ON[2]})")
GROUPS = ("actuation", "grippers", "power", "compute", "sensors", "frame",
          "safety", "charging", "spares", "shop")
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


def _bought(pid: str, *, usedBy=(), feeds=(), quantity: int = 1) -> Part:
  """A part bought off a seller's page: its facts the catalog's
  (`catalog.PARTS`), its use and its feeds this body's."""
  return replace(cat.by_id()[pid], robot=ROBOT, usedBy=tuple(usedBy), feeds=tuple(feeds),
                 quantity=quantity)


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
  _bought("amass_xt30_22_pair", usedBy=(hm.ROBOT_ROOT,), quantity=26),
  _bought("lapp_unitronic_can", usedBy=(hm.ROBOT_ROOT,), quantity=15),
  _bought("silicone_wire_14awg", usedBy=(hm.ROBOT_ROOT,), quantity=30),
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
  _bought("molicel_p45b", usedBy=(hm.ROBOT_ROOT,), quantity=hp.PACK.cells,
          feeds=(code("humanoid.parts.PACK.wh", "Wh", f"{hp.PACK.series}S{hp.PACK.parallel}P"),
                 code("humanoid.parts.PACK.kg", "kg", "the cells, the BMS, the strip and "
                      "the printed case"))),
  _bought("jbd_bms_12s_120a", usedBy=(hm.ROBOT_ROOT,)),
  _bought("hilumin_strip_10m"),
  _bought("meanwell_ddr_60l_5", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS_W", "compute + cameras", "W"),)),
  _bought("amass_xt90s", quantity=3),
  _bought("victron_midi_60a"),
  _bought("victron_midi_holder"),
  _bought("silicone_wire_10awg", quantity=4),
  _bought("noeifevo_charger_12s"),
  # ---- charging by its cable (#559 stage 2: `activity.station`) ----------------
  Part("ropd_panel_plug", "Rosenberger RoPD panel plug, C003-B1 (magnetic, 60 V DC, "
       "25 A): the charge port on the torso's front", "power", "candidate", SHELVES,
       (hm.ROBOT_ROOT,), source="https://products.rosenberger.com/_ocassets/db/"
       "C003-B1-XXX-Y_DB.pdf", massG=15.0, dimensionsMm="28.5 x 28.5 x 18.8",
       quantity=1, robot=ROBOT,
       feeds=(code("humanoid.parts.PORT_KG", "kg", expect=0.015),
              code("activity.station.BREAK_N", "N", "its disengagement force, the "
                   "sheets' minimum (no maximum published)", expect=20.0)),
       why={"partNumber": "the C003-B1 family; its variant code (cable length, "
                          "coding) not chosen",
            "priceEur": "no price published for the panel plug alone"},
       note="Magnetic and self-locating, 2 power contacts and 4 signal; not "
            "hot-pluggable, so the charger runs off its signal pins. Its capture "
            "range is unpublished (\"100 % self-location\"): the printed funnel "
            "round it is what the hand needs."),
  Part("charge_funnel", "The charge port's funnel, printed in PETG: 70 mm across its "
       "mouth, 15 mm deep", "structure", "candidate", SHELVES, (hm.ROBOT_ROOT,),
       massG=hp.PORT_FUNNEL_KG * 1000, dimensionsMm="70 x 70 x 15", quantity=1,
       robot=ROBOT,
       feeds=(code("humanoid.parts.PORT_FUNNEL", "m", "its mouth and depth",
                   expect=(0.07, 0.015)),),
       why={"partNumber": "printed to its drawing", "source": "printed to its drawing",
            "priceEur": "printed: its filament is the printed parts' line"},
       note="A 3 mm cone wall; its mass is its drawing's volume at 1.27 g/cm^3."),
  Part("ropd_cable_jack", "Rosenberger RoPD cable jack, C003-03 (25 A), on "
       "Rosenberger's C003 cable: the station's plug", "power", "candidate", SHELVES,
       (), source="https://products.rosenberger.com/_ocassets/db/C003-03-XXX-Y_DB.pdf",
       massG=20.0, dimensionsMm="24.5 dia. x 43", quantity=1, robot=ROBOT,
       feeds=(code("activity.station.JACK_LEN_M", "m", expect=0.043),
              code("activity.station.PLUG_R_M", "m", "its body's radius",
                   expect=0.01225)),
       why={"partNumber": "the C003-03 family; its cable length not chosen",
            "priceEur": "listed at $43.58 a piece (Powell, 6-week lead), no euro price"},
       note="Its cable: 2 x 2.5 mm^2 power and 4 x 0.35 mm^2 signal, PU, 8.3 mm, "
            "101 g/m; 2 m of it drop 0.32 V at 10 A."),
  Part("meanwell_npb_750_48", "Mean Well NPB-750-48 charger (42-80 V CV, 11.3 A CC, "
       "93 %): the station's", "power", "candidate", SHELVES, (),
       partNumber="NPB-750-48",
       source="https://www.meanwell.com/Upload/PDF/NPB-750/NPB-750-SPEC.PDF",
       dimensionsMm="230 x 158 x 67", priceEur=164.87, quantity=1, robot=ROBOT,
       feeds=(code("activity.station.CHARGE_A", "A", "its CC set to 10 A of its 11.3",
                   expect=10.0),),
       why={"massG": "only its packed weight is published: 1.84 kg"},
       note="Its CV programmed to 50.4 V (its default boost, 57.6 V, is past the "
            "pack's) over its CANBus or the SBP-001; its REMOTE input on the "
            "connector's signal pins. RS Belgium's price, excluding VAT."),
  Part("tecna_9300", "Tecna 9300 tool balancer (0.2-0.5 kg, 1.6 m stroke): draws "
       "the plug home", "structure", "candidate", SHELVES, (), partNumber="9300",
       source="https://www.westermans.com/assets/products/18927/files/8886/"
              "dcb0000002_rev.04_v2_generale-bilanciatori_en_09-2021r-1.pdf",
       massG=430.0, quantity=1, robot=ROBOT,
       feeds=(code("activity.station.BALANCER_N", "N", "its least, a 0.2 kg load",
                   expect=0.2 * 9.81),),
       why={"dimensionsMm": "not in its catalogue",
            "priceEur": "no price found"},
       note="Its rope clipped to the plug's back, its stop at 0.6 m; how its pull "
            "varies over its stroke is unpublished."),
  # ---- compute -----------------------------------------------------------------
  _bought("raspberry_pi_5", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "pi5 (packed)", "kg",
                            "its packed weight, an upper bound"),)),
  _bought("pi5_active_cooler"),
  _bought("pi_ssd_kit_256"),
  _bought("waveshare_can_hat_plus", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "waveshare_can_hat_plus", "kg"),)),
  _bought("waveshare_can_fd_hat", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS", "waveshare_can_fd_hat", "kg"),)),
  # ---- sensors -----------------------------------------------------------------
  _bought("realsense_d435", usedBy=("depth_cam_body",),
          feeds=(Feed("depth_eye.fovy", camera("depth_eye"), "°",
                      "the depth field's vertical angle", expect=58),
                 code("humanoid.parts.D435_KG", "kg"),
                 code("humanoid.model.CHOSEN.depth_pitch_deg", "°",
                      "fixed in the head, pitched at the counter and the floor ahead"),
                 Feed("color_eye.fovy", camera("color_eye"), "°",
                      "its colour imager: the tags within reach", expect=42))),
  _bought("livox_mid360s", usedBy=("lidar_mount",),
          feeds=(code("humanoid.parts.LIDAR.kg", "kg", expect=0.265),
                 code_entry("humanoid.parts.ELECTRONICS_W", "lidar", "W", "its sheet's "
                            "6.5 W through its 12 V converter, 89 %"),
                 code("perception.lidar.FOV_DEG", "°", "its field upside down, the sheet's "
                      "-7 to +52 turned over", expect=(-52.0, 7.0)),
                 code("perception.lidar.LIDAR_PERIOD", "s", "a frame", expect=0.1),
                 code("perception.lidar.BLIND_M", "m", "its blind zone", expect=0.1),
                 code("perception.lidar.SIGMA", "m", "a range's 1 sigma at 0.2 m and at "
                      "10 m", expect=((0.2, 0.04), (10.0, 0.02))),
                 code("humanoid.model.CHOSEN.lidar_rise", "m", "its rays' origin over the "
                      "head's top, upside down on its bracket"))),
  _bought("livox_m12_cable", usedBy=("lidar_mount",)),
  _bought("pi_camera_3", usedBy=("nav_cam_body",),
          feeds=(Feed("nav_eye.fovy", camera("nav_eye"), "°",
                      "the places' tags on the walls, and the eye a look sees through",
                      expect=41),)),
  _bought("pi5_camera_cable", usedBy=("nav_cam_body",)),
  _bought("tdk_ev_icm42688p", usedBy=(hm.ROBOT_ROOT,),
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
  _fabricated("humanoid_lidar_bracket", "The LIDAR's bracket: a 20 x 3 mm 6082 post up "
              "the head's back and the plate its base bolts under", ("lidar_post", "lidar_roof"),
              usedBy=("lidar_mount",)),
  _fabricated("humanoid_shells", "The head's shell and the torso's front cover, "
              "printed in PETG", ("head_shell", "torso_cover"), usedBy=(hm.ROBOT_ROOT,),
              printed=True),
  _bought("igus_kcrm_05", usedBy=tuple(f"{s}_shank" for s in hm.SIDES), quantity=4),
  _bought("petg_1kg", quantity=2),
  # ---- safety ------------------------------------------------------------------
  _bought("albright_sw80b549", usedBy=(hm.ROBOT_ROOT,),
          feeds=(code_entry("humanoid.parts.ELECTRONICS_W", "contactor", "W",
                            "its coil held, through the 12 V converter"),)),
  _bought("schneider_xb5as8442", usedBy=(hm.ROBOT_ROOT,)),
  _bought("radiomaster_pocket"),
  _bought("radiomaster_er6", usedBy=(hm.ROBOT_ROOT,)),
  _bought("pololu_rc_relay", usedBy=(hm.ROBOT_ROOT,)),
  _bought("molicel_p28a", quantity=2),
  _bought("meanwell_ddr_30l_12", usedBy=(hm.ROBOT_ROOT,)),
  _bought("arcol_hs25_47r", usedBy=(hm.ROBOT_ROOT,)),
  _bought("hongfa_hf115f_012", usedBy=(hm.ROBOT_ROOT,)),
  # ---- shop: what the build needs to make its pack, its harness and its
  # printed parts, and to power its first leg up on the bench ----------------
  *(_bought(pid) for pid in ("korad_ka6005p", "ucan_v1", "kweld_kit", "kweld_kcap",
                             "engineer_pa09", "knipex_9762145a", "pinecil_v2",
                             "unit_ut161e", "proxxon_mc5", "wera_950pks9", "bambu_p1s")),
)

#: What the allowances rest on: a machined 6082 part at a job shop, at the
#: rate the quadruped's legs were priced at in #379 (€800 for twenty-four
#: parts: €33.33 a part), and AluFritze's configurator for a plain plate
#: (€20.02 for 300 x 150 mm in 3 mm).
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
  # ---- charging by its cable ---------------------------------------------------
  Line("ropd_panel_plug", 1, "charging", None,
       why={"leadTime": "Rosenberger sells in lots (Rodantech: 150); none quoted"},
       allowanceEur=40.0, basis="the cable jack's listed $43.58"),
  Line("charge_funnel", 1, "charging", None, why={"leadTime": "printed"},
       allowanceEur=1.0, basis="25 g of PETG"),
  Line("ropd_cable_jack", 1, "charging", "6 weeks (Powell)",
       allowanceEur=45.0, basis="$43.58 at Powell"),
  Line("meanwell_npb_750_48", 1, "charging", None,
       why={"leadTime": "RS Belgium gives none; Jameco 14 weeks"}),
  Line("tecna_9300", 1, "charging", None, why={"leadTime": "none quoted"},
       allowanceEur=120.0, basis="a light tool balancer; none quoted"),
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
  Line("livox_mid360s", 1, "sensors", "only 3 left in stock, 1-3 working days (Funduino)"),
  Line("livox_m12_cable", 1, "sensors", "currently unavailable (Funduino)"),
  Line("meanwell_ddr_30l_12", 1, "sensors", "in stock, 1-2 business days (Reichelt)",
       note="the LIDAR's 12 V"),
  Line("pi_camera_3", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("pi5_camera_cable", 1, "sensors", "in stock, 1-3 days (BerryBase)"),
  Line("tdk_ev_icm42688p", 1, "sensors", "138 in stock, the maker's standard lead 16 "
       "weeks (DigiKey DE)"),
  # ---- frame -------------------------------------------------------------------
  Line("humanoid_brackets", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(48 * MACHINED_EUR, 2),
       basis="forty-eight machined 6082 plates with their fasteners, at €33.33 a "
             "machined part (#379's rate for the quadruped's legs: €800 for "
             "twenty-four)"),
  Line("humanoid_trunk_plates", 1, "frame", None, why=UNDRAWN, allowanceEur=300.0,
       basis="eight 3 mm 6082 plates, as #379 priced the quadruped's frame (€300 "
             "for six to eight): AluFritze's configurator prices a plain 300 x 150 mm "
             "one at €20.02; a plate with holes is priced from its drawing"),
  Line("humanoid_leg_plates", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(8 * MACHINED_EUR, 2),
       basis="eight machined side plates at €33.33 a machined part (#379's rate)"),
  Line("humanoid_soles", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(2 * MACHINED_EUR + 20.0, 2),
       basis="two machined sole plates at €33.33 a machined part (#379's rate), "
             "and €20 of 5 mm rubber sheet"),
  Line("humanoid_arm_tubes", 1, "frame", None, why=UNDRAWN, allowanceEur=40.0,
       basis="0.8 m of 6082 round tube in two sizes, cut to length; not priced for "
             "this bill"),
  Line("humanoid_ankle_rods", 1, "frame", None, why=UNDRAWN, allowanceEur=20.0,
       basis="two 0.34 m rods cut from 8 mm tube and threaded M5 at each end"),
  Line("humanoid_gripper_adapters", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(2 * MACHINED_EUR, 2),
       basis="two machined plates at €33.33 a machined part (#379's rate), drawn "
             "when the gripper's flange is known (Unitree does not publish it)"),
  Line("humanoid_lidar_bracket", 1, "frame", None, why=UNDRAWN,
       allowanceEur=round(2 * MACHINED_EUR, 2),
       basis="two machined parts at €33.33 a machined part (#379's rate)"),
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
  # ---- shop --------------------------------------------------------------------
  Line("korad_ka6005p", 1, "shop", "in stock, 1-3 business days (Welectron)",
       note="a leg's first power-ups; it charges the kCap and runs the Pinecil"),
  Line("ucan_v1", 1, "shop", "in stock, 1-3 days (Eckstein)"),
  Line("kweld_kit", 1, "shop", "28 in stock (keenlab)"),
  Line("kweld_kcap", 1, "shop", "41 in stock (keenlab)"),
  Line("engineer_pa09", 1, "shop", "31 in stock, ships the same day (Kiwi, NL)"),
  Line("knipex_9762145a", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("pinecil_v2", 1, "shop", "in stock, ships the same day (eleshop)"),
  Line("unit_ut161e", 1, "shop", "expected on 2026-10-07 (eleshop)"),
  Line("proxxon_mc5", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("wera_950pks9", 1, "shop", "in stock, 1-2 business days (Reichelt)"),
  Line("bambu_p1s", 1, "shop", "ships from the EU warehouse in 1-3 business days (Bambu Lab)"),
)


def build() -> dict:
  """The catalog as the site's parts page reads it (`protocol/parts.json`,
  issue #185): every part with each feed read off the humanoid's model and
  the live constants, and the bill off this bill's lines (`build`) -- no
  budget: #379's was set for the quadruped's build."""
  import mujoco
  spec = mujoco.MjSpec.from_file(WORLDS[ROBOT])
  parts = [p.as_dict(spec) for p in PARTS]
  ids = [p["id"] for p in parts]
  assert len(ids) == len(set(ids)), "duplicate part id"
  return {
    "schema": cat.SCHEMA, "worlds": dict(WORLDS), "kinds": list(cat.KINDS),
    "statuses": list(cat.STATUSES), "shelves": list(SHELVES),
    "nullable": list(cat.NULLABLE), "parts": parts,
    "build": cat.bom({p["id"]: p for p in parts}, lines=LINES, groups=GROUPS,
                     checked=min(READ_ON), budget_usd=None),
  }


def problems(fixture: dict) -> dict:
  """Every dishonest part or line, by id, and every feed the sim
  disagrees with: `{}` when the bill may be written."""
  parts = {p["id"]: p for p in fixture["parts"]}
  out = {pid: r for pid, p in parts.items()
         if (r := cat.validate(p, worlds=WORLDS, shelves=SHELVES))}
  out.update({f"line:{ln['part']}": r for ln in fixture["build"]["lines"]
              if (r := cat.validate_line(ln, parts, groups=GROUPS))})
  if off := cat.mismatches(fixture):
    out["mismatches"] = off
  return out


def with_bill(doc: str, fixture: dict) -> str:
  return cat.with_bill(doc, fixture, (BOM_START, BOM_END), read_on=CHECKED)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("-o", "--out", default="protocol/parts.json",
                      help="the catalog the site's parts page reads")
  parser.add_argument("--doc", default="docs/Parts.md",
                      help="the doc whose bill is re-rendered")
  args = parser.parse_args()
  fixture = build()
  if bad := problems(fixture):
    raise SystemExit(f"refusing to write a dishonest bill: {bad}")
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
  doc = Path(args.doc)
  doc.write_text(with_bill(doc.read_text(), fixture))
  t = fixture["build"]["totals"]
  print(f"{out}, {doc}: {len(fixture['parts'])} parts, {len(LINES)} lines, "
        f"{sum(len(p['feeds']) for p in fixture['parts'])} feeds, "
        f"€{t['totalEur']:,.2f} (${t['totalUsd']:,.2f})")


if __name__ == "__main__":
  main()
