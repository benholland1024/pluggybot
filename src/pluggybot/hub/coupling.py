"""Standalone hub tool-coupling spike (milestone-8 prep).

No robot here (schuko_spike's little sibling): a compliant carrier moves a
two-prong FORK through a scripted pick or return against a fixed hub shelf.
The questions this answers before any controller or hub layout depends on
them:
  - does the fork-and-peg gravity latch actually work in MuJoCo contact?
  - how much lateral / vertical / yaw misalignment does a pick forgive?
  - how hard can the carried tool be shaken before it hops out of the fork?

The coupling, designed for a robot with NO WRIST (latch verbs are slide-in
and lift/lower only):

  tool module   a plate hanging by a long horizontal PEG AXLE (along y)
  hub shelf     two upward-open V-TRAYS catch the peg near its ends
  arm fork      two prongs tipped with upward-open V-NOTCHES that grab the
                peg OUTBOARD of the trays -- nothing sits next to the grab
                points, so the lateral capture envelope is set by peg
                overhang length, not by machined clearances

Pick: slide the fork in under the peg overhangs, lift 15 mm (pegs seat into
the fork Vs, tool rises off the trays), back away. Return is the reverse.
Gravity is the latch; V depth is the retention -- both measured here.

Like the schuko recess, a V-notch is concave, so each one is COMPOSED of two
45-degree tilted boxes (convex pieces).
"""

import math

import mujoco
import numpy as np

from pluggybot.hub.tags import (
  BAY_TAG_IDS, CHARGE_TAG_ID, MODULE_TAG_IDS, RACK_TAG_ID,
  RACK_TAG_SIZE, SMALL_TAG_SIZE, asset_xml, plate_half_extent,
  write_tag_pngs,
)

# -- geometry (meters) --------------------------------------------------------
PEG_R = 0.003           # peg axle radius (6 mm rod)
PEG_HALF = 0.075        # peg half-length: 150 mm rod
TOOL_HALF_Y = 0.020     # tool body half-width: 40 mm plate
TOOL_HALF_Z = 0.030
TOOL_HALF_X = 0.010
PEG_ABOVE_BODY = 0.022  # peg axis above tool body centre
PEG_Z = 0.150           # peg rest height on the shelf (spike-local; the real
                        # hub sits wherever the lift can reach)
TRAY_Y = 0.040          # shelf V-trays, inboard pair
FORK_Y = 0.058          # fork prongs grab outboard of the trays
V_HALF_LEN = 0.011      # tilted plate half-length -> ~8 mm usable V depth
V_THICK = 0.003
V_RISE = V_HALF_LEN / math.sqrt(2) * 2   # vertex-to-plate-top height (~16 mm)
FORK_DROP = 0.022       # fork V vertex this far under the peg line on approach:
                        # the plate tips must pass BELOW the peg (first film
                        # showed the +x flank ramming it horizontally)
LIFT_STEP = 0.036       # m raised during a pick: FORK_DROP + seating + enough
                        # that the peg clears the tray plate TOPS -- 8 mm less
                        # and the peg exits by grinding up the tray flank at
                        # the full push cap (measured 10 N; clean is ~2 N)
TRAY_VERTEX_DROP = 0.008  # tray V vertex under the nominal peg line

# -- carrier ("the robot", simplified) ---------------------------------------
PUSH_FORCE = 10.0       # N cap on the approach axis
LAT_STIFFNESS = 150.0   # N/m lateral compliance (same guess as schuko spike)
YAW_STIFFNESS = 1.0     # N*m/rad
START_X = 0.16          # carrier start: fork tips well clear of the peg


def _v_notch_xml(prefix: str, pos: tuple[float, float, float],
                 half_y: float, rgba: str) -> str:
  """Upward-open V (peg axis along y) from two 45-degree boxes whose upper
  faces form the funnel: mouth ~16 mm wide, bottom self-centring in x."""
  x, y, z = pos
  off = V_HALF_LEN / math.sqrt(2)
  out = []
  for s, name in ((-1, "a"), (1, "b")):
    out.append(
      f'<geom name="{prefix}{name}" type="box" '
      f'size="{V_HALF_LEN:.4f} {half_y:.4f} {V_THICK:.4f}" '
      f'pos="{x + s * off:.4f} {y:.4f} {z + off:.4f}" '
      f'euler="0 {-s * 45:.0f} 0" rgba="{rgba}"/>')
  return "\n      ".join(out)


def scene_xml(dy: float = 0.0, dz: float = 0.0, yaw_deg: float = 0.0,
              tool_mass: float = 0.12) -> str:
  """Hub shelf + hanging tool at the origin; carrier approaching from +x.

  dy/dz: lateral/vertical offset of the carrier's approach line (m).
  yaw_deg: angular misalignment about z. tool_mass: the module's mass budget
  under test (the interface spec caps it -- confirm the latch holds it).
  """
  peg_len_color = "0.75 0.75 0.78 1"
  # The tool spawns HANGING: peg resting at the tray vertices (+ its radius).
  peg_rest_z = PEG_Z - TRAY_VERTEX_DROP + PEG_R
  tool_body_z = peg_rest_z - PEG_ABOVE_BODY

  trays = "\n      ".join(
    _v_notch_xml(f"tray_{lbl}_", (0.0, s * TRAY_Y, PEG_Z - TRAY_VERTEX_DROP),
                 0.008, "0.55 0.57 0.60 1")
    + f'\n      <geom name="tray_{lbl}_post" type="box" '
      f'size="0.006 0.008 {(PEG_Z - 0.02) / 2:.4f}" '
      f'pos="0 {s * TRAY_Y:.4f} {(PEG_Z - 0.02) / 2:.4f}" '
      f'rgba="0.45 0.47 0.50 1"/>'
    for lbl, s in (("l", 1), ("r", -1)))

  # Fork frame (carrier-local): V vertices at x=-0.055, z=-FORK_DROP, so the
  # whole fork -- plate tips included -- passes UNDER the peg on approach.
  fz = -FORK_DROP
  forks = "\n      ".join(
    _v_notch_xml(f"fork_{lbl}_", (-0.055, s * FORK_Y, fz),
                 0.006, "0.30 0.32 0.36 1")
    + f'\n      <geom name="fork_{lbl}_prong" type="box" '
      f'size="0.035 0.005 0.004" pos="{-0.055 + 0.035:.4f} {s * FORK_Y:.4f} '
      f'{fz - 0.004:.4f}" rgba="0.30 0.32 0.36 1"/>'
    for lbl, s in (("l", 1), ("r", -1)))

  return f"""
<mujoco model="hub_coupling_spike">
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <default>
    <geom friction="0.4" solref="0.005 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.2 0.5" dir="-0.5 0.35 -0.8"/>
    <light pos="0.2 0.3 0.4" dir="-0.4 -0.6 -0.7"/>
    <camera name="side" pos="0.30 -0.30 0.28" xyaxes="0.707 0.707 0 -0.25 0.25 0.93"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>

    <!-- Hub shelf: back wall + two V-tray posts -->
    <body name="hub">
      <geom name="hub_back" type="box" size="0.004 0.10 0.10"
            pos="-0.018 0 0.10" rgba="0.60 0.62 0.65 1"/>
      {trays}
    </body>

    <!-- Tool module: plate hanging by its peg axle in the trays -->
    <body name="tool" pos="0 0 {tool_body_z:.4f}">
      <freejoint/>
      <geom name="tool_body" type="box"
            size="{TOOL_HALF_X} {TOOL_HALF_Y} {TOOL_HALF_Z}"
            mass="{tool_mass - 0.02:.3f}" rgba="0.20 0.45 0.75 1"/>
      <geom name="tool_peg" type="cylinder" size="{PEG_R} {PEG_HALF}"
            zaxis="0 1 0" pos="0 0 {PEG_ABOVE_BODY:.4f}" mass="0.02"
            rgba="{peg_len_color}"/>
    </body>

    <!-- Carrier: approach rail with force-limited x + position-servo lift z,
         the fork hanging through compliant y/yaw (arm + base flex). -->
    <body name="rail" pos="{START_X:.4f} {dy:.4f} {PEG_Z + dz:.4f}"
          euler="0 0 {yaw_deg:.3f}">
      <inertial pos="0 0 0" mass="0.10" diaginertia="2e-5 2e-5 2e-5"/>
      <joint name="advance" type="slide" axis="-1 0 0" damping="{PUSH_FORCE / 0.03:.0f}"/>
      <joint name="lift" type="slide" axis="0 0 1" damping="20"/>
      <body name="fork" pos="0 0 0">
        <joint name="lat_y" type="slide" axis="0 1 0"
               stiffness="{LAT_STIFFNESS}" damping="4" armature="1e-5"/>
        <joint name="rot_z" type="hinge" axis="0 0 1"
               stiffness="{YAW_STIFFNESS}" damping="0.05" armature="1e-5"/>
        <geom name="fork_bridge" type="box" size="0.006 {FORK_Y + 0.006:.4f} 0.005"
              pos="-0.014 0 {-FORK_DROP - 0.004:.4f}" rgba="0.30 0.32 0.36 1"/>
        {forks}
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="advance" ctrlrange="-{PUSH_FORCE} {PUSH_FORCE}"/>
    <position name="lift" joint="lift" kp="400" kv="30"
              ctrlrange="-0.05 0.08" forcerange="-30 30"/>
  </actuator>
</mujoco>"""


def module_xml(name: str, x: float, y: float, peg_z: float,
               rgba: str, mass: float = 0.12, face: str = "",
               yaw_deg: float = 0.0) -> str:
  """A tool module as a FREE body hanging at a station. `face` injects
  extra visual-only geoms (screen, plug, tag) in the module's frame."""
  peg_rest_z = peg_z - TRAY_VERTEX_DROP + PEG_R
  body_z = peg_rest_z - PEG_ABOVE_BODY
  return f"""
    <body name="{name}" pos="{x:.4f} {y:.4f} {body_z:.4f}" euler="0 0 {yaw_deg:.1f}">
      <freejoint/>
      <geom name="{name}_body" type="box"
            size="{TOOL_HALF_X} {TOOL_HALF_Y} {TOOL_HALF_Z}"
            mass="{mass - 0.02:.3f}" rgba="{rgba}"/>
      <geom name="{name}_peg" type="cylinder" size="{PEG_R} {PEG_HALF}"
            zaxis="0 1 0" pos="0 0 {PEG_ABOVE_BODY:.4f}" mass="0.02"
            rgba="0.75 0.75 0.78 1"/>
      {face}
    </body>"""


# ---- rack v2 layout (the "bike rack for tools", designed with Ben) ---------
RACK_HANG_X = 0.09        # the hang plane: pegs this far out from the wall,
                          # leaving ~54 mm behind for business ends that face
                          # the WALL (a carried plug must point forward)
RACK_BRACKET_X = 0.07     # tray brackets drop from the rail BEHIND the hang
                          # plane, so a lifted peg rises into free air
RACK_RAIL_Z = 0.40        # rail height: carried peg tops out ~55 mm below
HUB_PEG_Z = 0.30          # module peg height (mid lift range)
HUB_STATION_YS = (0.125, -0.125)  # tool bays at 0.25 m pitch
CHARGE_BAY_Y = -0.375     # charge bay continues the pitch at the rack's end
RACK_HALF_W = 0.48        # side posts
CHARGE_PIN_Z = 0.09       # pogo pins at bumper height (chassis 0.06-0.12)
# Fiducial plates carry real tag36h11 AprilTags (hub/tags.py). Plate sizes
# follow from the marker sizes, which are themselves a range decision: a
# tag must span ~25-30 px to decode, so the rack's marker is large (read
# from across the room) and the bay/module markers are small (read from
# arm's length).
RACK_PLATE_HALF = plate_half_extent(RACK_TAG_SIZE)
SMALL_PLATE_HALF = plate_half_extent(SMALL_TAG_SIZE)


def rack_charge_contact(model, data) -> bool:
  """Both charge-bay pins touching the robot's bumper face: the rack-side
  sibling of the plug's electrical criterion, and milestone 8's charge hook."""
  pins = {model.geom("rack_pin_l").id, model.geom("rack_pin_r").id}
  chassis = model.geom("chassis").id
  touching = set()
  for i in range(data.ncon):
    c = data.contact[i]
    pair = {c.geom1, c.geom2}
    if chassis in pair:
      touching |= pins & pair
  return len(touching) == 2


def _bay_xml(prefix: str, y: float, tag_id: int) -> str:
  """One tool bay: two rail-hung brackets ending in V-trays at the hang
  plane, plus a small bay tag (fiducial placeholder) on the rail above."""
  parts = []
  col_h = (RACK_RAIL_Z - 0.283) / 2
  for lbl, s in (("l", 1), ("r", -1)):
    yy = y + s * TRAY_Y
    parts.append(
      f'<geom name="{prefix}col_{lbl}" type="box" size="0.006 0.008 {col_h:.4f}" '
      f'pos="{RACK_BRACKET_X:.4f} {yy:.4f} {0.283 + col_h:.4f}" '
      f'rgba="0.45 0.47 0.50 1"/>')
    parts.append(
      f'<geom name="{prefix}foot_{lbl}" type="box" size="0.014 0.008 0.004" '
      f'pos="{RACK_BRACKET_X + 0.010:.4f} {yy:.4f} 0.284" '
      f'rgba="0.45 0.47 0.50 1"/>')
    parts.append(_v_notch_xml(f"{prefix}tray_{lbl}_",
                              (RACK_HANG_X, yy, HUB_PEG_Z - TRAY_VERTEX_DROP),
                              0.008, "0.55 0.57 0.60 1"))
  parts.append(
    f'<geom name="{prefix}bay_tag" type="box" '
    f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
    f'pos="{RACK_HANG_X + 0.002:.4f} {y:.4f} {HUB_PEG_Z + 0.055:.4f}" '
    f'contype="0" conaffinity="0" material="tagmat{tag_id}"/>')
  return "\n      ".join(parts)


def _rack_body_xml(pos: tuple[float, float, float] = (0, 0, 0),
                   yaw_deg: float = 0.0) -> str:
  """The unified rack as ONE free body at an arbitrary room pose."""
  ya, yb = HUB_STATION_YS
  return f"""<body name="rack" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {yaw_deg:.1f}">
      <freejoint/>
      <!-- frame: side posts, rail, anti-tip feet, ballast/PSU shelf (1.5 kg
           low -- the ballast IS the stability budget) -->
      <geom name="rack_post_l" type="box" size="0.012 0.012 {RACK_RAIL_Z / 2:.3f}"
            pos="{RACK_BRACKET_X:.4f} {RACK_HALF_W:.4f} {RACK_RAIL_Z / 2:.3f}"
            mass="0.15" rgba="0.50 0.52 0.55 1"/>
      <geom name="rack_post_r" type="box" size="0.012 0.012 {RACK_RAIL_Z / 2:.3f}"
            pos="{RACK_BRACKET_X:.4f} {-RACK_HALF_W:.4f} {RACK_RAIL_Z / 2:.3f}"
            mass="0.15" rgba="0.50 0.52 0.55 1"/>
      <geom name="rack_rail" type="box" size="0.012 {RACK_HALF_W:.3f} 0.012"
            pos="{RACK_BRACKET_X:.4f} 0 {RACK_RAIL_Z:.3f}" mass="0.20"
            rgba="0.50 0.52 0.55 1"/>
      <geom name="rack_foot_l" type="box" size="0.070 0.015 0.010"
            pos="0.10 {RACK_HALF_W:.4f} 0.010" mass="0.10" friction="1.0"
            rgba="0.35 0.37 0.40 1"/>
      <geom name="rack_foot_r" type="box" size="0.070 0.015 0.010"
            pos="0.10 {-RACK_HALF_W:.4f} 0.010" mass="0.10" friction="1.0"
            rgba="0.35 0.37 0.40 1"/>
      <geom name="rack_shelf" type="box" size="0.030 {RACK_HALF_W - 0.02:.3f} 0.012"
            pos="0.08 0 0.032" mass="1.50" rgba="0.40 0.42 0.45 1"/>
      <!-- wall braces: the rack LEANS on the wall, so every robot press
           transfers straight into it -- without these it scooted 9.6 mm
           under a sustained charge-bay press (measured; free-body rack) -->
      <geom name="rack_brace_l" type="box" size="{RACK_BRACKET_X / 2:.4f} 0.010 0.010"
            pos="{RACK_BRACKET_X / 2:.4f} {RACK_HALF_W:.4f} 0.35" mass="0.05"
            rgba="0.50 0.52 0.55 1"/>
      <geom name="rack_brace_r" type="box" size="{RACK_BRACKET_X / 2:.4f} 0.010 0.010"
            pos="{RACK_BRACKET_X / 2:.4f} {-RACK_HALF_W:.4f} 0.35" mass="0.05"
            rgba="0.50 0.52 0.55 1"/>
      <geom name="rack_tag" type="box"
            size="0.002 {RACK_PLATE_HALF:.4f} {RACK_PLATE_HALF:.4f}"
            pos="{RACK_BRACKET_X + 0.014:.4f} 0 {RACK_RAIL_Z + 0.075:.3f}"
            contype="0" conaffinity="0" material="tagmat{RACK_TAG_ID}"/>
      <geom name="rack_tag_mast" type="box" size="0.006 0.006 0.045"
            pos="{RACK_BRACKET_X:.4f} 0 {RACK_RAIL_Z + 0.045:.3f}" mass="0.02"
            rgba="0.50 0.52 0.55 1"/>
      {_bay_xml("baya_", ya, BAY_TAG_IDS[0])}
      {_bay_xml("bayb_", yb, BAY_TAG_IDS[1])}
      <!-- charge bay: panel + two pogo pins at bumper height + its own tag
           (the terminal servo needs a mark here like anywhere else) -->
      <geom name="rack_charge_panel" type="box" size="0.008 0.06 0.05"
            pos="0.10 {CHARGE_BAY_Y:.4f} {CHARGE_PIN_Z:.3f}" mass="0.10"
            rgba="0.30 0.32 0.36 1"/>
      <geom name="rack_charge_tag" type="box"
            size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}"
            pos="0.109 {CHARGE_BAY_Y:.4f} {CHARGE_PIN_Z + 0.030:.3f}"
            contype="0" conaffinity="0" material="tagmat{CHARGE_TAG_ID}"/>
      <geom name="rack_pin_l" type="cylinder" size="0.004 0.006" zaxis="1 0 0"
            pos="0.114 {CHARGE_BAY_Y + 0.03:.4f} {CHARGE_PIN_Z:.3f}" mass="0.005"
            rgba="0.85 0.75 0.30 1"/>
      <geom name="rack_pin_r" type="cylinder" size="0.004 0.006" zaxis="1 0 0"
            pos="0.114 {CHARGE_BAY_Y - 0.03:.4f} {CHARGE_PIN_Z:.3f}" mass="0.005"
            rgba="0.85 0.75 0.30 1"/>
    </body>"""


def _module_faces() -> tuple[str, str]:
  lcd_face = (f'<geom name="module_lcd_screen" type="box" '
              f'size="0.002 0.014 0.020" pos="{-TOOL_HALF_X:.4f} 0 0" '
              f'contype="0" conaffinity="0" rgba="0.05 0.08 0.10 1"/>'
              f'\n      <geom name="module_lcd_tag" type="box" '
              f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
              f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
              f'material="tagmat{MODULE_TAG_IDS["module_lcd"]}"/>')
  plug_face = (
    f'<geom name="module_plug_barrel" type="cylinder" size="0.01775 0.015" '
    f'zaxis="1 0 0" pos="{-(TOOL_HALF_X + 0.013):.4f} 0 0" '
    f'contype="0" conaffinity="0" rgba="0.15 0.25 0.6 1"/>'
    f'\n      <geom name="module_plug_pin_l" type="capsule" size="0.0024" '
    f'fromto="{-(TOOL_HALF_X + 0.028):.4f} 0.0095 0 {-(TOOL_HALF_X + 0.044):.4f} 0.0095 0" '
    f'contype="0" conaffinity="0" rgba="0.75 0.75 0.78 1"/>'
    f'\n      <geom name="module_plug_pin_r" type="capsule" size="0.0024" '
    f'fromto="{-(TOOL_HALF_X + 0.028):.4f} -0.0095 0 {-(TOOL_HALF_X + 0.044):.4f} -0.0095 0" '
    f'contype="0" conaffinity="0" rgba="0.75 0.75 0.78 1"/>'
    f'\n      <geom name="module_plug_tag" type="box" '
    f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
    f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
    f'material="tagmat{MODULE_TAG_IDS["module_plug"]}"/>')
  return lcd_face, plug_face


def write_hub_world(path: str = "models/hub_world.xml") -> None:
  """Bare hub world: floor, wall, the unified tool RACK, modules, fork robot.

  The rack is ONE freestanding structure leaning against the wall -- and a
  FREE BODY, deliberately: whether the robot's presses scoot or tip it is a
  question the sim should measure, not assume. Modules hang business-end
  toward the wall, so a picked tool faces the robot's driving direction. The
  charge bay ends the rail: pogo pins at bumper height, pressed by the
  milestone-7 charge-press behavior, usable whatever the fork carries. White
  plates are AprilTag placeholders (rack pose, bay identity, module
  identity) until the perception pass.
  """
  ya, yb = HUB_STATION_YS
  lcd_face, plug_face = _module_faces()
  tag_ids = write_tag_pngs()
  xml = f"""<!-- GENERATED by pluggybot.hub.coupling.write_hub_world().
     Regenerate: uv run python -m pluggybot.hub.coupling
     Bare hub world (milestone 8): fork robot + wall + the unified tool rack
     (free body: stability is measured, not assumed) with hanging modules and
     a charge bay. Geometry constants live in hub/coupling.py. -->
<mujoco model="hub_world">
  <option timestep="0.001" integrator="implicitfast"/>
  <include file="pluggybot_fork.xml"/>
  <asset>
    {asset_xml(tag_ids)}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <light pos="1.5 0 1.5" dir="-0.7 0 -0.7"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall" type="box" size="0.01 1.2 0.5" pos="-0.01 0 0.5"
          rgba="0.72 0.70 0.62 1"/>

    {_rack_body_xml()}

    {module_xml("module_lcd", RACK_HANG_X, ya, HUB_PEG_Z,
                "0.20 0.45 0.75 1", face=lcd_face)}
    {module_xml("module_plug", RACK_HANG_X, yb, HUB_PEG_Z,
                "0.25 0.27 0.30 1", face=plug_face)}
    <camera name="hub_watch" pos="0.85 -0.85 0.70"
            xyaxes="0.707 0.707 0 -0.32 0.32 0.89"/>
  </worldbody>
</mujoco>
"""
  with open(path, "w") as fh:
    fh.write(xml)


# ---- room placement (room_hub.xml) -----------------------------------------
RACK_ROOM_POS = (-0.90, 5.99)   # rack origin: north wall of room 1, braces
                                # touching the wall face at y=5.99
RACK_ROOM_YAW = -90.0           # rack local +x (its outward normal) -> -y:
                                # facing south, into room 1


def rack_frame_to_world(x_local: float, y_local: float) -> tuple[float, float]:
  """A rack-frame point in room coordinates (rotation by RACK_ROOM_YAW)."""
  th = math.radians(RACK_ROOM_YAW)
  c, s = math.cos(th), math.sin(th)
  return (RACK_ROOM_POS[0] + x_local * c - y_local * s,
          RACK_ROOM_POS[1] + x_local * s + y_local * c)


def write_hub_rack(path: str = "models/hub_rack.xml") -> None:
  """The rack + modules as a room include, placed on room 1's north wall."""
  ya, yb = HUB_STATION_YS
  lcd_face, plug_face = _module_faces()
  tag_ids = write_tag_pngs()
  ax, ay_ = rack_frame_to_world(RACK_HANG_X, ya)
  bx, by_ = rack_frame_to_world(RACK_HANG_X, yb)
  xml = f"""<!-- GENERATED by pluggybot.hub.coupling.write_hub_rack().
     Regenerate: uv run python -m pluggybot.hub.coupling
     The tool rack + hanging modules placed against room 1's north wall,
     included by room_hub.xml. Same generator as hub_world.xml's rack. -->
<mujocoinclude>
  <asset>
    {asset_xml(tag_ids)}
  </asset>
  <worldbody>
    {_rack_body_xml((RACK_ROOM_POS[0], RACK_ROOM_POS[1], 0.0), RACK_ROOM_YAW)}
    {module_xml("module_lcd", ax, ay_, HUB_PEG_Z,
                "0.20 0.45 0.75 1", face=lcd_face, yaw_deg=RACK_ROOM_YAW)}
    {module_xml("module_plug", bx, by_, HUB_PEG_Z,
                "0.25 0.27 0.30 1", face=plug_face, yaw_deg=RACK_ROOM_YAW)}
  </worldbody>
</mujocoinclude>
"""
  with open(path, "w") as fh:
    fh.write(xml)


def run_pick(dy: float = 0.0, dz: float = 0.0, yaw_deg: float = 0.0,
             tool_mass: float = 0.12, shake_accel: float = 0.0,
             n_frames: int = 0) -> tuple[dict, list[np.ndarray]]:
  """One scripted pick: approach, lift, retreat (then optionally shake).

  Success = the tool left the trays and followed the carrier out. shake_accel
  (m/s^2) oscillates the advance axis after the pick to probe retention.
  Returns (result, frames): picked, carried (still held after any shake),
  max contact force, and the tool's final peg seating error in the fork.
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(dy, dz, yaw_deg, tool_mass))
  data = mujoco.MjData(model)
  push = model.actuator("push").id
  lift = model.actuator("lift").id
  tool_bid = model.body("tool").id

  frames: list[np.ndarray] = []
  renderer = mujoco.Renderer(model, 360, 480) if n_frames else None

  def step(n, ctrl_push, ctrl_lift):
    nonlocal max_force
    f6 = np.zeros(6)
    for _ in range(n):
      data.ctrl[push] = ctrl_push
      data.ctrl[lift] = ctrl_lift
      mujoco.mj_step(model, data)
      for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)
        max_force = max(max_force, float(np.linalg.norm(f6[:3])))
      if renderer is not None and len(frames) < n_frames and \
         data.time > len(frames) * total_time / n_frames:
        renderer.update_scene(data, camera="side")
        frames.append(renderer.render().copy())

  max_force = 0.0
  total_time = 15.0 + (2.0 if shake_accel else 0.0)
  adv_qadr = model.joint("advance").qposadr[0]

  step(1500, 0.0, 0.0)                   # settle: tool takes its true hang
  z0_tool = float(data.xpos[tool_bid][2])

  # Approach until the fork bottoms out (bridge on the tool face is the depth
  # reference -- feeler thinking); force-limited, so over-advance is gentle.
  step(5000, PUSH_FORCE, 0.0)
  step(2000, 0.0, LIFT_STEP)             # lift: peg into the fork Vs
  step(6500, -PUSH_FORCE * 0.4, LIFT_STEP)   # retreat with the tool
  tool_x = float(data.xpos[tool_bid][0])
  lifted = float(data.xpos[tool_bid][2]) - z0_tool
  picked = lifted > 0.005 and tool_x > 0.05

  carried = picked
  if picked and shake_accel:
    # retention probe: oscillate the advance axis (drive accel analog)
    for k in range(4):
      sign = 1 if k % 2 == 0 else -1
      step(500, sign * min(PUSH_FORCE, 0.12 * shake_accel), LIFT_STEP)
    fork_x = START_X - float(data.qpos[adv_qadr])
    carried = (float(data.xpos[tool_bid][2]) - z0_tool > 0.005
               and abs(float(data.xpos[tool_bid][0]) - fork_x) < 0.10)

  if renderer is not None:
    renderer.close()
  return {
    "picked": picked,
    "carried": carried,
    "lifted_mm": lifted * 1000.0,
    "max_force_n": max_force,
  }, frames


def run_cycle(dy: float = 0.0, dz: float = 0.0, yaw_deg: float = 0.0,
              tool_mass: float = 0.12, n_frames: int = 0,
              ) -> tuple[dict, list[np.ndarray]]:
  """Full pick-and-RETURN cycle: take the tool off the hub, carry it out,
  bring it back, hang it up, leave empty. The offsets apply to BOTH
  approaches (the same navigation errors happen twice).

  Success ("returned") = the tool ends hanging in the trays again -- peg
  near its rest height, body near the hub centreline -- with the fork clear.
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(dy, dz, yaw_deg, tool_mass))
  data = mujoco.MjData(model)
  push = model.actuator("push").id
  lift = model.actuator("lift").id
  tool_bid = model.body("tool").id

  frames: list[np.ndarray] = []
  renderer = mujoco.Renderer(model, 360, 480) if n_frames else None
  total_time = 26.0
  max_force = 0.0

  def step(n, ctrl_push, ctrl_lift):
    nonlocal max_force
    f6 = np.zeros(6)
    for _ in range(n):
      data.ctrl[push] = ctrl_push
      data.ctrl[lift] = ctrl_lift
      mujoco.mj_step(model, data)
      for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)
        max_force = max(max_force, float(np.linalg.norm(f6[:3])))
      if renderer is not None and len(frames) < n_frames and \
         data.time > len(frames) * total_time / n_frames:
        renderer.update_scene(data, camera="side")
        frames.append(renderer.render().copy())

  step(1500, 0.0, 0.0)
  z_rest = float(data.xpos[tool_bid][2])

  step(5000, PUSH_FORCE, 0.0)                 # pick: slide under
  step(2000, 0.0, LIFT_STEP)                  #   lift off the trays
  step(6500, -PUSH_FORCE * 0.4, LIFT_STEP)    #   carry out
  picked = (float(data.xpos[tool_bid][2]) - z_rest > 0.005
            and float(data.xpos[tool_bid][0]) > 0.05)

  step(5000, PUSH_FORCE, LIFT_STEP)           # return: slide back in, high
  step(2000, 0.0, 0.0)                        #   lower: peg into the trays
  step(4000, -PUSH_FORCE * 0.4, 0.0)          #   leave empty
  tp = data.xpos[tool_bid]
  returned = (abs(float(tp[2]) - z_rest) < 0.006
              and abs(float(tp[0])) < 0.012
              and abs(float(tp[1])) < 0.02)

  if renderer is not None:
    renderer.close()
  return {
    "picked": picked,
    "returned": returned,
    "max_force_n": max_force,
  }, frames


if __name__ == "__main__":
  write_hub_world()
  write_hub_rack()
  print("wrote models/hub_world.xml and models/hub_rack.xml")
