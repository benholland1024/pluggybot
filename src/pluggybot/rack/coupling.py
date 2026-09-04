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

from pluggybot.rack.tags import (
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

# -- the peg as the module's ELECTRICAL interface ----------------------------
# Measured (SimNotes): the peg already sits in four V-notch plates carrying
# 0.43-0.49 N each of gravity preload -- 4-9x what a lean-pad could ever
# supply, because a lean-pad's preload is capped by the same weak 22 mm
# geometry that made the pad necessary in the first place. So the power
# contacts go where the force already is. Splitting the peg into two
# conductors either side of an insulated centre makes the LEFT V-notch pair
# and the RIGHT V-notch pair the two poles of a power-only coupling: no extra
# parts, no extra alignment, and the seating slide wipes the contact clean.
# The peg is also already the one metal part in the design (6 mm steel rod).
# Steel rod on printed V-notches. MuJoCo's DEFAULT is 1.0, whose friction
# angle is 45 degrees -- exactly the V's flank angle, so the peg sat right on
# the sliding threshold and barely self-centred. The coupling SPIKE that
# measured the +/-4 mm envelope has always set 0.4; the generated hub world
# was silently running the same coupling at 1.0, so the measured tolerances
# were never the ones in use. `priority=1` because MuJoCo combines pair
# friction as the elementwise MAX -- the caster lesson, again: setting a low
# friction without priority does nothing at all.
# This is also what removes a nasty conflict: at 1.0 the noslip pass needed
# for grasping BROKE the coupling (peg on the fork but not electrically
# seated), because the peg seats by sliding and noslip suppresses sliding. At
# 0.4 the peg self-centres properly and both work together.
PEG_FRICTION = 0.4
PEG_INSUL_HALF = 0.012                            # insulated centre section
PEG_COND_HALF = (PEG_HALF - PEG_INSUL_HALF) / 2   # each conductor
PEG_COND_Y = PEG_INSUL_HALF + PEG_COND_HALF       # conductor centre offset
# Which fork geoms are which pole. Left/right are the fork's own frame; a
# module always presents the same face to the fork, so the pairing holds at
# any rack yaw.
FORK_POLE_GEOMS = {"l": ("fork_vl_a", "fork_vl_b"),
                   "r": ("fork_vr_a", "fork_vr_b")}

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
              tool_mass: float = 0.12, noslip: int = 0) -> str:
  """Hub shelf + hanging tool at the origin; carrier approaching from +x.

  dy/dz: lateral/vertical offset of the carrier's approach line (m).
  yaw_deg: angular misalignment about z. tool_mass: the module's mass budget
  under test (the interface spec caps it -- confirm the latch holds it).
  noslip: `noslip_iterations` for the run -- the peg seats by SLIDING into
  its V, so the coupling is the behavior a global noslip policy is most
  likely to break, and the spike must be able to measure it under that
  policy (issue #3).
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
  <option timestep="0.001" integrator="implicitfast" noslip_iterations="{noslip}"/>
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
      {peg_xml(name)}
      {face}
    </body>"""


def peg_xml(name: str, z: float = PEG_ABOVE_BODY) -> str:
  """The hang peg, built as two conductors around an insulated centre.

  Mechanically this is still one 150 mm rod -- the sections are collinear and
  the same radius, and the V-notches they seat in are far outboard of the
  centre (fork plates at |y| 52-64 mm, tray plates at 32-48 mm), so the
  insulator never touches anything and the latch behaves exactly as measured.
  Electrically it is a two-pole connector that costs nothing to align,
  because the alignment is the gravity latch that was already there.

  (The RACK's trays are the same geometry, so a hub that wanted to power or
  charge modules on the shelf could use the identical seam. Not modelled --
  here the robot is what powers its tools.)
  """
  out = []
  for lbl, s in (("l", 1), ("r", -1)):
    out.append(
      f'<geom name="{name}_peg_{lbl}" type="cylinder" '
      f'size="{PEG_R} {PEG_COND_HALF:.4f}" zaxis="0 1 0" '
      f'pos="0 {s * PEG_COND_Y:.4f} {z:.4f}" mass="0.008" '
      f'friction="{PEG_FRICTION}" priority="1" '
      f'rgba="0.75 0.75 0.78 1"/>')
  out.append(
    f'<geom name="{name}_peg_insul" type="cylinder" '
    f'size="{PEG_R} {PEG_INSUL_HALF:.4f}" zaxis="0 1 0" '
    f'pos="0 0 {z:.4f}" mass="0.004" rgba="0.12 0.12 0.14 1"/>')
  return "\n      ".join(out)


def module_power_state(model, data, name: str = "module_lcd") -> dict:
  """Is this module's coupling conducting, and if not, which pole is open?

  The rack-side sibling of `rack_charge_contact`, and the same lesson behind
  both: an electrical criterion beats a positional one. Milestone 6 burned
  four position-based seat detectors before the charging voltage settled it;
  here the question "is the tool powered" has exactly one honest answer, and
  it is the same one the hardware will have.

  Reporting the poles separately is deliberate. A half-seated coupling --
  one conductor on, one off -- is a real failure mode of a two-point latch
  (the feelers taught this: one prong 39 mm short wrecked everything while
  the other looked perfect), and a bare boolean would hide it as "off".
  """
  poles = {}
  for side, plates in FORK_POLE_GEOMS.items():
    try:
      peg = model.geom(f"{name}_peg_{side}").id
      plate_ids = {model.geom(g).id for g in plates}
    except KeyError:
      poles[side] = False
      continue
    touching = False
    for i in range(data.ncon):
      pair = {data.contact[i].geom1, data.contact[i].geom2}
      if peg in pair and plate_ids & pair:
        touching = True
        break
    poles[side] = touching
  return {"left": poles["l"], "right": poles["r"],
          "powered": poles["l"] and poles["r"]}


def module_power_contact(model, data, name: str = "module_lcd") -> bool:
  """Both poles conducting -- the module is coupled and powered."""
  return module_power_state(model, data, name)["powered"]


# ---- rack v2 layout (the "bike rack for tools", designed with Ben) ---------
RACK_HANG_X = 0.09        # the hang plane: pegs this far out from the wall,
                          # leaving ~54 mm behind for business ends that face
                          # the WALL (a carried plug must point forward)
RACK_BRACKET_X = 0.07     # tray brackets drop from the rail BEHIND the hang
                          # plane, so a lifted peg rises into free air
RACK_RAIL_Z = 0.40        # rail height: carried peg tops out ~55 mm below
HUB_PEG_Z = 0.30          # module peg height (mid lift range)
HUB_STATION_YS = (0.125, -0.125, 0.375, 0.625, 0.875)  # tool bays at 0.25 m
                          # pitch. C/D/E mirror the charge bay's place at the
                          # far end, and are APPENDED rather than inserted in
                          # y order: the bay<->tag pairing is by index, and
                          # every demo and test names its bay as
                          # HUB_STATION_YS[0].
CHARGE_BAY_Y = -0.375     # charge bay continues the pitch at the rack's end
RACK_HALF_W = 0.93        # side posts. Grew 0.48 -> 0.68 for the fourth tool
                          # bay and 0.68 -> 0.93 for the fifth (the seed
                          # dispenser): five modules at 0.25 m pitch plus the
                          # charge bay is 1.86 m of rail. Re-checked against
                          # BOTH rooms it stands in, which is the step a bay
                          # addition must not skip -- the rail grows along a
                          # wall and the far post is the thing that hits
                          # something. room_hub: rack at world (-0.90, 5.99)
                          # yaw -90, so the rail now spans x -1.83..0.03,
                          # still clear of room 1's west wall at -2.0 (0.17 m
                          # margin) and of the floor-box at (0, 4) (wrong y).
                          # home_world: rack at (0.5, -1.98) yaw 90, rail
                          # spans x -0.43..1.43 inside a house wall at -2.0.
CHARGE_PIN_Z = 0.09       # pogo pins at bumper height (chassis 0.06-0.12)
CHARGE_TAG_X = 0.109      # rack-local x of the charge tag's PLATE CENTRE --
                          # the anchor a measured standoff fix is computed
                          # from (mission/mission.py, issue #32). The face the
                          # camera sees is PLATE_HALF_T proud of it; the 2 mm
                          # is inside the creep's electrical stop, so the
                          # reach is left as measured.
RACK_TAG_X = RACK_BRACKET_X + 0.014  # rack-local x of the big rack tag's
                          # plate centre, on its mast (localize.TAG_LOCAL_X)
PLATE_HALF_T = 0.002      # every fiducial plate is a 4 mm box; PnP returns
                          # the printed FACE, one half-thickness proud
# Fiducial plates carry real tag36h11 AprilTags (rack/tags.py). Plate sizes
# follow from the marker sizes, which are themselves a range decision: a
# tag must span ~25-30 px to decode, so the rack's marker is large (read
# from across the room) and the bay/module markers are small (read from
# arm's length).
RACK_PLATE_HALF = plate_half_extent(RACK_TAG_SIZE)
SMALL_PLATE_HALF = plate_half_extent(SMALL_TAG_SIZE)


# Where a bay tag's FACE sits, rack-local: plate centre 2 mm proud of the
# hang plane plus its own 2 mm half-thickness. The terminal travel ranges
# off this face (mission._terminal_travel), so the offset is a constant of
# the rack's geometry, not a calibration number.
BAY_TAG_FACE_X = RACK_HANG_X + 0.004

#: Rack-local (x, y) of every RACK-FIXED tag's printed face, keyed by id --
#: the layout `localize.fit_rack_facing` fits to what the dock camera
#: decodes (issue #88). Commissioning knowledge, like `RackPose.prior`: a
#: real multi-tag board ships with its geometry. Module tags are NOT here:
#: a module hangs a few mm off the plane and may be on the fork.
RACK_TAG_FACES: dict[int, tuple[float, float]] = {
  RACK_TAG_ID: (RACK_TAG_X + PLATE_HALF_T, 0.0),
  CHARGE_TAG_ID: (CHARGE_TAG_X + PLATE_HALF_T, CHARGE_BAY_Y),
  **{BAY_TAG_IDS[i]: (BAY_TAG_FACE_X, y) for i, y in enumerate(HUB_STATION_YS)},
}


def bay_tag_id(station_y: float) -> int:
  """The AprilTag id of the bay nearest this station y.

  Was a hardcoded two-bay equality check in mission.py -- fine while there
  were exactly two bays and silently wrong the moment there were three.
  Pairing is by INDEX into HUB_STATION_YS, so bays may sit anywhere.
  """
  i = min(range(len(HUB_STATION_YS)),
          key=lambda k: abs(HUB_STATION_YS[k] - station_y))
  return BAY_TAG_IDS[i]


def bay_prefix(i: int) -> str:
  """Geom-name prefix for bay i: baya_, bayb_, bayc_, ..."""
  return f"bay{chr(ord('a') + i)}_"


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
            pos="{RACK_TAG_X:.4f} 0 {RACK_RAIL_Z + 0.075:.3f}"
            contype="0" conaffinity="0" material="tagmat{RACK_TAG_ID}"/>
      <geom name="rack_tag_mast" type="box" size="0.006 0.006 0.045"
            pos="{RACK_BRACKET_X:.4f} 0 {RACK_RAIL_Z + 0.045:.3f}" mass="0.02"
            rgba="0.50 0.52 0.55 1"/>
      {chr(10).join(f'      {_bay_xml(bay_prefix(i), y, BAY_TAG_IDS[i])}'
                    for i, y in enumerate(HUB_STATION_YS))}
      <!-- charge bay: panel + two pogo pins at bumper height + its own tag
           (the terminal servo needs a mark here like anywhere else) -->
      <geom name="rack_charge_panel" type="box" size="0.008 0.06 0.05"
            pos="0.10 {CHARGE_BAY_Y:.4f} {CHARGE_PIN_Z:.3f}" mass="0.10"
            rgba="0.30 0.32 0.36 1"/>
      <geom name="rack_charge_tag" type="box"
            size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}"
            pos="{CHARGE_TAG_X:.3f} {CHARGE_BAY_Y:.4f} {CHARGE_PIN_Z + 0.030:.3f}"
            contype="0" conaffinity="0" material="tagmat{CHARGE_TAG_ID}"/>
      <geom name="rack_pin_l" type="cylinder" size="0.004 0.006" zaxis="1 0 0"
            pos="0.114 {CHARGE_BAY_Y + 0.03:.4f} {CHARGE_PIN_Z:.3f}" mass="0.005"
            rgba="0.85 0.75 0.30 1"/>
      <geom name="rack_pin_r" type="cylinder" size="0.004 0.006" zaxis="1 0 0"
            pos="0.114 {CHARGE_BAY_Y - 0.03:.4f} {CHARGE_PIN_Z:.3f}" mass="0.005"
            rgba="0.85 0.75 0.30 1"/>
    </body>"""


# ---- the drawing module (milestone 8): a tool with its own actuated axis ---
# The robot is nonholonomic: the base owns x and yaw, the lift owns z, the arm
# owns reach, and NOTHING owns lateral. So a pen carriage running along the
# module's own y axis -- parallel to the peg -- supplies the one DoF the robot
# structurally lacks, and pairs with the existing lift to make an X-Y plotter
# against a vertical board. The module is not cargo; it is a kinematic
# extension the robot acquires by picking it up.
PEN_TRAVEL = 0.055       # +/- along the peg axis: 110 mm of drawing width.
                         # Bounded by the peg's own half-length (0.075) and the
                         # fork's axial end-stops at 0.079 -- the rail must not
                         # reach either.
PEN_RAIL_Z = -0.025      # rail below the module body centre, i.e. 47 mm below
                         # the peg: clear under the lean-pad (which stops
                         # 36 mm below the peg) and under the rack's tray
                         # brackets. Guarded by a clearance sweep, not by this
                         # comment.
PEN_LEN = 0.045          # pen protrusion past the carriage
PEN_MOUNT_X = -(TOOL_HALF_X + 0.016)   # rail/carriage stand off IN FRONT of
                         # the module plate. The first version put them at
                         # the plate's own x, so the carriage block and pen
                         # shaft were buried inside it -- and since the quill
                         # is a GRANDCHILD of the module body, MuJoCo does not
                         # filter that pair (only parent-child is excluded).
                         # The carriage jammed against the plate's corner at
                         # +21.9 mm and stopped tracking for most of a figure.
                         # The clearance sweep missed it because it checked
                         # the pen against the ROBOT, never against the
                         # module's own frame.
PEN_RAIL_DZ = -0.012     # rail rides BELOW the pen line, so a fully
                         # compressed quill retracts the shaft past the rail
                         # rather than into it. The SIGN is issue #10's whole
                         # fix, and it is a stow-clearance decision, not a
                         # drawing one. Above the pen line, the rail and the
                         # carriage that wraps it both topped out 16 mm under
                         # the bay's bracket feet -- but setting a module down
                         # needs the peg lifted 14.7 mm to pass over the tray
                         # flanks, so the pen had a ~1 mm window between "peg
                         # too low to clear the flanks" and "rail jammed under
                         # the feet", and RETURN_CLEARANCE (20 mm) sits inside
                         # the foul band. Every other module clears because
                         # nothing of theirs reaches into the bracket band at
                         # all. Below the pen line the tallest thing left is
                         # the quill itself, the window opens to 14.7-28 mm,
                         # and -- because the PEN LINE DOES NOT MOVE -- the
                         # drawing geometry, the lift presets and the pen's
                         # moment arm about the peg are all untouched. Sweep
                         # table in SimNotes ("The pen would not stow").
PEN_BLOCK_LO = PEN_RAIL_DZ - 0.006   # carriage block, relative to the pen
PEN_BLOCK_HI = 0.004                 # line: wraps the rail at one end and
                         # reaches over the quill at the other. It is sized
                         # from the two things it joins because it is the
                         # TALLEST part of the assembly whenever the rail is
                         # not -- a block left at its old extent would have
                         # kept the old ceiling with the rail already moved.
PEN_CARRIAGE_MASS = 0.030
PEN_RAIL_MASS = 0.020
PEN_QUILL_TRAVEL = 0.020     # sprung pen holder: absorbs arm-position error
PEN_QUILL_STIFFNESS = 60.0   # N/m. SOFT and long-travel, which is the whole
                             # trick. A first attempt at 200 N/m held only
                             # ~0.5 mm of deflection and the pen lifted clean
                             # off the top edge of the figure (0 % ink there,
                             # 98 % at the bottom): the arm's droop GROWS with
                             # lift height, so the pen retreats a few mm as it
                             # draws upward. A stiff spring cannot absorb that
                             # without the force swinging wildly; a soft one
                             # at ~10 mm nominal press holds 0.6 N and varies
                             # only 0.4-0.8 N across the whole figure. Sized
                             # against the peg's limit too -- the lean-pad
                             # reacts pen torque at a 29 mm lever, and past
                             # ~1.5 N the peg rides up its V-notch and out.

# The board the pen draws on. In the bare hub world it stands opposite the
# rack, so the robot picks the tool at x~0 and drives out to x~1 to use it --
# the errand shape the lifecycle already runs. Placed on the FORK line
# (y = -PLUG_LATERAL for a robot heading +x at y=0), because the pen inherits
# the fork's 5 cm lateral offset and a board centred on the chassis would put
# every drawing off to one side.
# ---- the claw module: reach DOWN, never forward -----------------------------
# Measured (SimNotes): the fork-and-peg gravity latch takes about **0.45 N.m**
# of pitch moment before the peg rides out of its V. A payload at forward
# offset L costs W*L, so reach is far more expensive than mass: 800 g hangs
# happily on the peg axis, and 400 g unseats the module at 150 mm out. Hence a
# pendant straight down the peg's own axis (L = 0) rather than the angled arm
# the idea started as. The chassis is nowhere near the limit -- 800 g at the
# peg only drops wheel load 15.0 -> 12.6 N, and static tipping needs ~5 kg.
CLAW_JAW_Z = -0.132       # jaw centre. Peg sits 172 mm up at lift 0, so this
                          # puts the jaws astride an object on the floor with
                          # the lift near its bottom stop.
CLAW_JAW_OPEN = 0.035     # jaw stand-off from centre when open (70 mm span)
CLAW_JAW_TRAVEL = 0.027   # inward travel to a 8 mm gap
CLAW_PAD_HALF_H = 0.020   # pad half-height. 40 mm pads grip and lift
                          # reliably (verified: 99.6 mm off the floor). They
                          # do force the grip ~9 mm ABOVE a 26 mm block's
                          # centre of mass, because pads cannot go below the
                          # floor, and that offset is a lever -- the first
                          # hard turn pivots the block out. Shortening them to
                          # 24 mm to grip across the CoM was tried and made
                          # the GRASP fail outright, so the tall pads stay
                          # until that is understood. See docs/SimNotes.md:
                          # carrying through a turn is an open item.
# A HARD contact constraint on the pads, and this is what lets the whole robot
# run one solver policy instead of two.
#
# MuJoCo's contact constraints are soft by default (solimp 0.9 0.95 0.001), and
# a soft tangential constraint DRIFTS under sustained load: a gripped block
# creeps out of the jaws at ~8 mm/s regardless of clamp force. The obvious cure
# is the `noslip` post-solve pass, and it works -- but it costs 2.8x the step
# time globally and had to be toggled per phase, which is a footgun.
#
# Stiffening the constraint on the two PADS fixes it at the source: slip over a
# 100 mm lift goes -21.7 mm (soft, no noslip) -> -0.13 mm (hard, no noslip),
# against +0.28 mm for the noslip cure. Same result, no solver mode, no toggle,
# and 3x faster. It is also the more honest model -- a rigid printed pad on a
# rigid block should not squash -- and it follows the house pattern of putting
# contact behaviour on the specific geoms that need it (condim="1"
# priority="1" on the caster; friction+priority on the peg).
GRIP_SOLIMP = "0.99 0.999 0.0001"
CLAW_GRIP_KP = 600.0      # grip force = kp x squeeze past contact. 200 held
                          # a static lift but lost the block during a TURN:
                          # the pendant swings, and 1.8 N per pad was not
                          # enough against that. 600 is an MG996R-class servo
                          # (11 kg.cm) rather than a micro one -- a real part
                          # choice, not a tuning knob.
# The drop tube must STOP at the jaw tops. A first version ran it to -0.140,
# which is 28 mm INTO the grip zone: descending on an object, the tube reached
# the target first and shoved the block 17 mm forward and 3 mm up, and the
# jaws then closed on empty air while a graze still read as "both pads in
# contact". The structure that carries a gripper must not occupy the space
# the gripper needs.
CLAW_PENDANT_BOT = CLAW_JAW_Z + CLAW_PAD_HALF_H   # = the jaw tops
CLAW_PENDANT_TOP = -TOOL_HALF_Z           # = the module plate's underside
# The arm ANGLES FORWARD as it drops rather than hanging straight down. Two
# reasons, and neither is reach for its own sake:
#   * a straight pendant puts the grip directly under the module, so a
#     module-mounted camera has to look down the shaft it is bolted to and
#     sees its own arm. Angling forward opens a clear sightline.
#   * it costs almost nothing. The coupling budget is 0.45 N.m, and 60 g at
#     55 mm forward of the peg is 0.032 N.m -- 7 % of it. Reach is only
#     expensive when it is LONG (400 g at 150 mm was 0.59 N.m and unseated
#     the module).
# Bounded by the RACK, not by the moment: modules hang business-end-inward, so
# this arm points at the wall when stowed, and there are exactly 80 mm between
# a racked module's front face and the wall.
CLAW_REACH = 0.055        # forward offset of the grip from the module centre

BOARD_X = 1.30
BOARD_Y = -0.05
BOARD_Z = 0.30
BOARD_HALF = (0.010, 0.16, 0.13)


# The claw's own camera. Mounted on the MODULE -- a first; every other camera
# is on the chassis -- and offset to the SIDE so the arm does not block the
# view of its own grip point. Module data crosses the coupling wirelessly like
# every other module signal, which is why a tool camera costs no CSI port on
# the Pi, unlike a fourth chassis camera would. Wide fovy: it works at ~140 mm.
CLAW_EYE_POS = (0.033, 0.030, -0.079)     # (forward, lateral, vertical),
                          # i.e. partway DOWN the arm rather than above its
                          # root. Set by rendering, in three passes: mounted
                          # high the arm filled the frame and clipped the
                          # block; moving outboard helped; putting the camera
                          # partway down the arm puts the arm BEHIND it and
                          # halves the working distance to ~65 mm.


# ---- the seed dispenser (the fifth tool): metering, not manipulation -------
#: The first tool built AGAINST docs/ToolPattern.md rather than mined out of
#: the ones before it. Its envelope check (zero moment on the peg axis, no
#: lean-pad demand because a dispenser releases rather than presses, ~178 g in
#: class) is that doc's worked stage-0 example -- ToolPattern.md section 5.0.
#
# The mechanism is a SLIDE-VALVE ESCAPEMENT, which is what a real seed meter
# is: one actuator, one moving part, exactly one seed per cycle.
#
#   shuttle home (0)      a pocket sits under the tube mouth; the seed in it
#                         rests on a fixed shelf. The blanking slab is parked
#                         off to one side.
#   shuttle out (STROKE)  the pocket -- seed and all -- has travelled past
#                         the end of the shelf, so that ONE seed drops; and
#                         the blanking slab has arrived under the tube mouth,
#                         holding the rest of the stack up.
#   home again            the slab retracts and the stack settles one seed
#                         into the pocket, ready for the next cycle.
#
# Metering by GEOMETRY, never by timing: "open the gate for 200 ms" dispenses
# a different number of seeds on a slower machine.
SEED_R = 0.008            # 16 mm seed body. Sized UP from a literal seed: the
                          # claw's grip physics is tuned for ~5 cm objects and
                          # true seed-scale bodies are miserable contact work,
                          # so this is a bean/bulb, the smallest thing worth
                          # simulating rather than the smallest thing real.
SEED_MASS = 0.004
# A DROPPED SPHERE ROLLS FOREVER, and no amount of sliding friction stops it.
# Measured on a seed released with 0.15 m/s of residual horizontal motion --
# which is just the base's settle, nothing dramatic:
#
#   condim=3, friction 0.7                586 mm travelled, still doing 97 mm/s at 6 s
#   condim=3, friction 1.0 priority=1     586 mm, 97 mm/s      (IDENTICAL)
#   condim=6, friction 0.9 0.02 0.005      14 mm, stopped
#
# The middle row is the lesson. Sliding friction resists SLIDING, and a
# rolling ball is not sliding -- so cranking mu does literally nothing, to
# the millimetre. Rolling resistance is a different friction dimension and
# needs `condim="6"` before MuJoCo will even solve for it. Exactly the shape
# of the issue-3 wheel finding ("a velocity servo commanded 0 resists speed,
# not force") and of the caster lesson, met on a third axis. Sowing accuracy
# was 220 mm before this and is bounded by the drop, not the roll, after it.
# `priority="1"` because pair friction combines as the elementwise MAX, so
# the floor's default would otherwise define the contact.
SEED_FRICTION = "0.9 0.02 0.005"     # sliding, torsional, rolling
SEED_COUNT = 3            # a magazine, not a hopper: three is enough to prove
                          # the escapement meters ONE per cycle (which one
                          # seed cannot) and cheap in contacts.
DISP_BORE = 0.010         # half-width of the square bore (20 mm), 4 mm of
                          # slop around the seed
DISP_WALL = 0.002
DISP_TUBE_TOP = -TOOL_HALF_Z   # flush with the plate's underside: the PLATE is
                          # the magazine's cap, and everything below it is in
                          # air the rack never occupies
DISP_MOUTH_Z = -0.118     # tube mouth. Leaves 88 mm of tube -- 5 seeds' worth
                          # of room for a 3-seed load, so a bounced seed in
                          # transit has somewhere to land that is still inside.
DISP_POCKET_HALF_Z = 0.010     # pocket 20 mm deep: taller than the seed, so a
                          # metered seed is caged on four sides and cannot be
                          # shaken out of the pocket while the tool is carried
DISP_STROKE = 0.024       # pocket -> exit travel. Must exceed the bore's full
                          # width (20 mm) so the blanking slab covers the mouth
                          # completely at the out position, and so the pocket
                          # clears the shelf's end at the same moment.
DISP_SHELF_END = 0.012    # the shelf stops here (+y); past it the pocket has
                          # no floor. This single number IS the exit hole --
                          # cheaper and more robust than a 4-box frame with a
                          # hole in it, and it cannot be mis-decomposed.
DISP_SHELF_HALF_Z = 0.002
DISP_GATE_KP = 800.0      # a small linear servo, not a solenoid: metering
                          # wants a POSITION, and the whole point of the
                          # escapement is that the count comes from geometry
                          # rather than from how long a coil stayed energised.
DISP_SHELF_FRICTION = 0.30     # the seed slides across this shelf on its way
                          # out, and MuJoCo combines pair friction as the
                          # elementwise MAX -- so without `priority` the
                          # seed's own value would win and the shelf would
                          # drag. The caster lesson, fifth appearance.
# Derived, never re-typed (the claw's pendant constant broke twice by being
# typed rather than derived).
DISP_POCKET_Z = DISP_MOUTH_Z - DISP_POCKET_HALF_Z          # pocket centre
DISP_SHELF_TOP = DISP_MOUTH_Z - 2 * DISP_POCKET_HALF_Z     # pocket floor
DISP_OUTLET_Z = DISP_SHELF_TOP - 0.0005 - 2 * DISP_SHELF_HALF_Z   # underside


def seed_stack_zs(n: int = SEED_COUNT) -> list[float]:
  """Module-local z of each loaded seed: one in the pocket, the rest stacked
  in the tube above it. Spaced 1 mm over a seed diameter so the world loads
  without interpenetration -- a stack that starts overlapping starts with an
  impulse, and the first frame of a demo is a bad place to learn that."""
  z0 = DISP_SHELF_TOP + SEED_R + 0.001
  return [z0 + k * (2 * SEED_R + 0.001) for k in range(n)]


def seed_bodies_xml(module_x: float, module_y: float, module_body_z: float,
                    yaw_deg: float = 0.0, n: int = SEED_COUNT) -> str:
  """The loaded seeds, as FREE bodies riding in the dispenser's magazine.

  Free bodies rather than geoms on the module, and the reason is the whole
  reason a dispenser is worth building: a dispensed seed must become a thing
  in the world that the sim (and the website) can see fall, land and stay
  put. It also gives the tool an honest physical criterion -- "a seed left
  the tube and reached the floor" is a fact, where "the gate was commanded
  open" is a belief.

  They are retained by GEOMETRY, not by grip: a capped tube over a shelf,
  with the only exit blocked by the shuttle unless it is deliberately
  driven out. So unlike the claw's payload, a carried magazine needs no
  swing analysis -- there is no pose in which the seeds have anywhere to go.
  """
  th = math.radians(yaw_deg)
  c, s = math.cos(th), math.sin(th)
  out = []
  for k, z in enumerate(seed_stack_zs(n)):
    # The magazine sits on the module's own axis (x = y = 0 module-local), so
    # the yaw rotation is a no-op for position -- but it is written out
    # anyway, because the next tool to load something off-axis will copy this
    # and a silently-omitted rotation is exactly how the rack-frame verdict
    # bug happened.
    wx = module_x + 0.0 * c - 0.0 * s
    wy = module_y + 0.0 * s + 0.0 * c
    out.append(
      f"""
    <body name="seed_{k}" pos="{wx:.4f} {wy:.4f} {module_body_z + z:.4f}">
      <freejoint/>
      <geom name="seed_{k}_body" type="sphere" size="{SEED_R}" """
      f"""mass="{SEED_MASS}" condim="6" friction="{SEED_FRICTION}" """
      f"""priority="1" rgba="0.85 0.72 0.35 1"/>
    </body>""")
  return "".join(out)


def dispenser_actuator_xml() -> str:
  """The escapement's one actuator. Lives on the MODULE, like the pen
  carriage and the claw's jaws -- driven by the module's own ESP32 across the
  wireless data link, powered through the peg."""
  return (f'<position name="seed_gate" joint="seed_gate_joint" '
          f'kp="{DISP_GATE_KP}" kv="12" '
          f'ctrlrange="0 {DISP_STROKE:.4f}" forcerange="-20 20"/>')


def _look_at(pos, target, up=(0.0, 0.0, 1.0)) -> str:
  """MJCF `xyaxes` for a camera at `pos` aimed at `target`.

  Computed rather than hand-typed: a camera bolted to a swappable tool aims at
  a point fixed by other constants, and hand-derived direction cosines are
  exactly what rots silently when one of those constants moves.
  """
  z = np.array(pos, dtype=float) - np.array(target, dtype=float)
  z /= np.linalg.norm(z)
  x = np.cross(np.array(up, dtype=float), z)
  if np.linalg.norm(x) < 1e-6:
    x = np.array([1.0, 0.0, 0.0])
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  return " ".join(f"{v:.4f}" for v in (*x, *y))


#: The LCD's display panel, half-extents (depth, width, height) in metres.
#: 56 x 76 mm, deliberately OVERHANGING the 40 x 60 mm carrier plate: this
#: is the module whose entire job is being looked at, and issue #28's
#: acceptance is "legible at the distance a visitor's camera actually sits
#: at" -- which a panel inset into the plate is not. Costs nothing anywhere:
#: the geom is visual-only (contype/conaffinity 0), it clears the pegs in x
#: (they span +/-3 mm about the plate centre plane, the panel sits 8-12 mm
#: in front of it) and the fork plates in y (|y| 52-64 mm, panel 28 mm).
LCD_SCREEN_HALF = (0.002, 0.028, 0.038)


def _module_faces() -> tuple[str, str]:
  lcd_face = (f'<geom name="module_lcd_screen" type="box" '
              f'size="{LCD_SCREEN_HALF[0]} {LCD_SCREEN_HALF[1]} '
              f'{LCD_SCREEN_HALF[2]}" pos="{-TOOL_HALF_X:.4f} 0 0" '
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
  pen_face = (
    # Rail: the carriage's track, spanning further in y than the module plate
    # so the pen can reach past the tool's own width.
    f'<geom name="module_pen_rail" type="box" '
    f'size="0.004 {PEN_TRAVEL + 0.012:.4f} 0.004" '
    f'pos="{PEN_MOUNT_X:.4f} 0 {PEN_RAIL_Z + PEN_RAIL_DZ:.4f}" '
    f'mass="{PEN_RAIL_MASS}" rgba="0.55 0.57 0.60 1"/>'
    f'\n      <geom name="module_pen_tag" type="box" '
    f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
    f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
    f'material="tagmat{MODULE_TAG_IDS["module_pen"]}"/>'
    # The actuated axis. A lead screw HOLDS position unpowered (the dryspin
    # argument from Parts.md that milestone 7's power model already relies
    # on), so a position servo parked at its target is the honest model of a
    # module sitting unpowered on the rack.
    f'\n      <body name="module_pen_carriage" '
    f'pos="{PEN_MOUNT_X:.4f} 0 {PEN_RAIL_Z + PEN_RAIL_DZ:.4f}">'
    f'\n        <joint name="pen_carriage_joint" type="slide" axis="0 1 0" '
    f'range="{-PEN_TRAVEL:.4f} {PEN_TRAVEL:.4f}" damping="2"/>'
    f'\n        <geom name="module_pen_block" type="box" '
    f'size="0.008 0.010 {(PEN_BLOCK_HI - PEN_BLOCK_LO) / 2:.4f}" '
    f'pos="0 0 {(PEN_BLOCK_HI + PEN_BLOCK_LO) / 2 - PEN_RAIL_DZ:.4f}" '
    f'mass="{PEN_CARRIAGE_MASS}" rgba="0.30 0.32 0.36 1"/>'
    # SPRUNG QUILL. The plug's RCC lesson, one tool along: the wrist has
    # compliance in y, z, and both yaws, but NONE along the approach axis,
    # so pen pressure would otherwise be (arm position error) x (the arm's
    # 1200 N/m servo) -- 6 mm of overshoot is 7 N, and past ~1.5 N the peg
    # rides up its own V-notch. A spring behind the pen turns pressure into
    # a design constant instead of a positioning problem, and absorbs the
    # lift droop that varies with height. This is what a real plotter pen is.
    f'\n        <body name="module_pen_quill" pos="0 0 {-PEN_RAIL_DZ:.4f}">'
    f'\n          <joint name="pen_quill_joint" type="slide" axis="1 0 0" '
    f'range="0 {PEN_QUILL_TRAVEL:.4f}" stiffness="{PEN_QUILL_STIFFNESS}" '
    f'damping="2" armature="1e-6"/>'
    f'\n          <geom name="module_pen_shaft" type="capsule" size="0.0025" '
    f'fromto="-0.008 0 0 {-(0.008 + PEN_LEN):.4f} 0 0" '
    f'mass="0.006" friction="0.25" priority="1" '
    f'solimp="{GRIP_SOLIMP}" rgba="0.90 0.30 0.25 1"/>'
    f'\n          <site name="pen_tip" pos="{-(0.008 + PEN_LEN):.4f} 0 0" '
    f'size="0.002" rgba="0.9 0.3 0.25 1"/>'
    f'\n        </body>'
    f'\n      </body>')
  eye = (-CLAW_EYE_POS[0], CLAW_EYE_POS[1], CLAW_EYE_POS[2])
  eye_axes = _look_at(eye, (-CLAW_REACH, 0.0, CLAW_JAW_Z))
  claw_face = (
    # Drop tube on the peg's own axis (module x = 0). That is the whole
    # design: L = 0 spends none of the 0.45 N.m coupling budget on reach.
    f'<geom name="module_claw_pendant" type="capsule" size="0.008" '
    f'fromto="0 0 {CLAW_PENDANT_TOP:.4f} '
    f'{-CLAW_REACH:.4f} 0 {CLAW_PENDANT_BOT:.4f}" mass="0.045" '
    f'rgba="0.45 0.47 0.50 1"/>'
    f'\n      <camera name="claw_eye" pos="{eye[0]:.4f} {eye[1]:.4f} '
    f'{eye[2]:.4f}" xyaxes="{eye_axes}" fovy="58"/>'
    f'\n      <geom name="module_claw_tag" type="box" '
    f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
    f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
    f'material="tagmat{MODULE_TAG_IDS["module_claw"]}"/>'
    # Parallel jaws. Slides rather than pivots: a parallel gripper keeps the
    # pads flat on the object through the whole closing stroke, so grip force
    # does not fight a changing contact angle. High-friction pads with
    # priority, because MuJoCo combines pair friction as the MAX and the
    # floor's 1.0 would otherwise define the grip (the caster lesson).
    + "".join(
      f'\n      <body name="module_claw_jaw_{lbl}" '
      f'pos="{-CLAW_REACH:.4f} {s * CLAW_JAW_OPEN:.4f} {CLAW_JAW_Z:.4f}">'
      f'\n        <joint name="claw_{lbl}" type="slide" axis="0 {s} 0" '
      f'range="{-CLAW_JAW_TRAVEL:.4f} 0" damping="1"/>'
      f'\n        <geom name="module_claw_pad_{lbl}" type="box" '
      f'size="0.014 0.004 {CLAW_PAD_HALF_H:.4f}" mass="0.020" '
      f'friction="1.5" priority="1" solimp="{GRIP_SOLIMP}" '
      f'rgba="0.30 0.32 0.36 1"/>'
      f'\n      </body>'
      for lbl, s in (("l", 1), ("r", -1)))
    + f'\n      <site name="claw_grip" pos="{-CLAW_REACH:.4f} 0 {CLAW_JAW_Z:.4f}" size="0.003" '
      f'rgba="0.9 0.6 0.2 1"/>')
  tube_bot = DISP_MOUTH_Z
  tube_half_h = (DISP_TUBE_TOP - tube_bot) / 2
  tube_z = (DISP_TUBE_TOP + tube_bot) / 2
  shelf_z = DISP_SHELF_TOP - 0.0005 - DISP_SHELF_HALF_Z
  seed_face = (
    f'<geom name="module_seed_tag" type="box" '
    f'size="0.002 {SMALL_PLATE_HALF:.4f} {SMALL_PLATE_HALF:.4f}" '
    f'pos="{TOOL_HALF_X:.4f} 0 0" contype="0" conaffinity="0" '
    f'material="tagmat{MODULE_TAG_IDS["module_seed"]}"/>'
    # The magazine: a square tube of four walls, hanging entirely below the
    # plate. Square rather than round because a box is a convex primitive and
    # a tube is not -- the same decomposition the V-notches and the schuko
    # recess needed.
    + "".join(
      f'\n      <geom name="module_seed_tube_{lbl}" type="box" '
      f'size="{sx:.4f} {sy:.4f} {tube_half_h:.4f}" '
      f'pos="{px:.4f} {py:.4f} {tube_z:.4f}" mass="0.008" '
      f'rgba="0.50 0.42 0.30 1"/>'
      for lbl, sx, sy, px, py in (
        ("px", DISP_WALL, DISP_BORE + 2 * DISP_WALL, DISP_BORE + DISP_WALL, 0.0),
        ("nx", DISP_WALL, DISP_BORE + 2 * DISP_WALL, -(DISP_BORE + DISP_WALL), 0.0),
        ("py", DISP_BORE, DISP_WALL, 0.0, DISP_BORE + DISP_WALL),
        ("ny", DISP_BORE, DISP_WALL, 0.0, -(DISP_BORE + DISP_WALL))))
    # The shelf the metered seed rides out on. It simply STOPS at
    # DISP_SHELF_END -- that absence is the exit hole.
    + f'\n      <geom name="module_seed_shelf" type="box" '
      f'size="{DISP_BORE + DISP_WALL:.4f} '
      f'{(DISP_SHELF_END + DISP_BORE + 2 * DISP_WALL) / 2:.4f} '
      f'{DISP_SHELF_HALF_Z:.4f}" '
      f'pos="0 {(DISP_SHELF_END - DISP_BORE - 2 * DISP_WALL) / 2:.4f} '
      f'{shelf_z:.4f}" mass="0.006" '
      f'friction="{DISP_SHELF_FRICTION}" priority="1" '
      f'rgba="0.50 0.42 0.30 1"/>'
    f'\n      <site name="seed_outlet" pos="0 {DISP_SHELF_END + 0.010:.4f} '
    f'{DISP_OUTLET_Z:.4f}" size="0.003" rgba="0.85 0.72 0.35 1"/>'
    # The escapement shuttle: ONE moving part, on ONE joint. Its pocket and
    # its blanking slab are DISP_STROKE apart, which is what makes "seed
    # released" and "stack held" the same motion rather than two that have
    # to be sequenced against each other.
    f'\n      <body name="module_seed_shuttle" pos="0 0 {DISP_POCKET_Z:.4f}">'
    f'\n        <joint name="seed_gate_joint" type="slide" axis="0 1 0" '
    f'range="0 {DISP_STROKE:.4f}" damping="1"/>'
    f'\n        <geom name="module_seed_blank" type="box" '
    f'size="{DISP_BORE + 2 * DISP_WALL:.4f} {DISP_BORE + 2 * DISP_WALL:.4f} '
    f'{DISP_POCKET_HALF_Z:.4f}" pos="0 {-DISP_STROKE:.4f} 0" mass="0.010" '
    f'rgba="0.35 0.37 0.40 1"/>'
    + "".join(
      f'\n        <geom name="module_seed_pocket_{lbl}" type="box" '
      f'size="{sx:.4f} {sy:.4f} {DISP_POCKET_HALF_Z:.4f}" '
      f'pos="{px:.4f} {py:.4f} 0" mass="0.004" rgba="0.35 0.37 0.40 1"/>'
      for lbl, sx, sy, px, py in (
        ("py", DISP_BORE + 2 * DISP_WALL, DISP_WALL + 0.003,
         0.0, DISP_BORE + DISP_WALL + 0.003),
        ("px", DISP_WALL + 0.003, DISP_BORE + DISP_WALL + 0.003,
         DISP_BORE + DISP_WALL + 0.003, 0.0),
        ("nx", DISP_WALL + 0.003, DISP_BORE + DISP_WALL + 0.003,
         -(DISP_BORE + DISP_WALL + 0.003), 0.0)))
    + '\n      </body>')
  return lcd_face, plug_face, pen_face, claw_face, seed_face


def claw_actuator_xml() -> str:
  """Both jaws, driven together. The module's own ESP32 owns this servo --
  power-only coupling, wireless data, exactly as the LCD and pen modules."""
  return "\n    ".join(
    f'<position name="claw_{lbl}" joint="claw_{lbl}" kp="{CLAW_GRIP_KP}" '
    f'kv="6" ctrlrange="{-CLAW_JAW_TRAVEL:.4f} 0" forcerange="-20 20"/>'
    for lbl in ("l", "r"))


def pen_actuator_xml() -> str:
  """The pen carriage's own actuator. Lives on the MODULE, not the robot --
  which is the whole point of the drawing tool: it brings an axis the base
  does not have."""
  # kp is a LEAD SCREW's stiffness, not a hobby servo's. The first version
  # used kp=120, which needs ~8 mm of tracking error to make 1 N -- and the
  # pen drags against the board the whole time it draws, so the figure came
  # out 12 mm RMS off. A screw does not yield to friction; neither should
  # this.
  return (f'<position name="pen_carriage" joint="pen_carriage_joint" '
          f'kp="2000" kv="80" '
          f'ctrlrange="{-PEN_TRAVEL:.4f} {PEN_TRAVEL:.4f}" '
          f'forcerange="-15 15"/>')


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
  ya, yb, yc, yd, ye = HUB_STATION_YS
  lcd_face, plug_face, pen_face, claw_face, seed_face = _module_faces()
  tag_ids = write_tag_pngs()
  xml = f"""<!-- GENERATED by pluggybot.rack.coupling.write_hub_world().
     Regenerate: uv run python -m pluggybot.rack.coupling
     Bare hub world (milestone 8): fork robot + wall + the unified tool rack
     (free body: stability is measured, not assumed) with hanging modules and
     a charge bay. Geometry constants live in rack/coupling.py. -->
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

    <!-- Drawing board for the pen module: a whiteboard on a stand, facing
         the rack. Static geoms (no freejoint) -- a board that slides away
         under the pen would be measuring the board, not the plotter. -->
    <!-- friction + priority, not friction alone: MuJoCo combines pair
         friction as the elementwise MAX, so a slippery board against the
         default 1.0 floor material would still drag like rubber. This is
         THE caster lesson, and it costs a drawing dearly -- pen drag is what
         the carriage servo has to fight. -->
    <geom name="board" type="box"
          size="{BOARD_HALF[0]} {BOARD_HALF[1]} {BOARD_HALF[2]}"
          pos="{BOARD_X} {BOARD_Y} {BOARD_Z}" friction="0.25" priority="1"
          rgba="0.95 0.95 0.93 1"/>
    <geom name="board_leg_l" type="box" size="0.012 0.012 {BOARD_Z / 2:.3f}"
          pos="{BOARD_X + 0.02} {BOARD_Y + BOARD_HALF[1] - 0.02} {BOARD_Z / 2:.3f}"
          rgba="0.45 0.47 0.50 1"/>
    <geom name="board_leg_r" type="box" size="0.012 0.012 {BOARD_Z / 2:.3f}"
          pos="{BOARD_X + 0.02} {BOARD_Y - BOARD_HALF[1] + 0.02} {BOARD_Z / 2:.3f}"
          rgba="0.45 0.47 0.50 1"/>

    {_rack_body_xml()}

    {module_xml("module_lcd", RACK_HANG_X, ya, HUB_PEG_Z,
                "0.20 0.45 0.75 1", face=lcd_face)}
    {module_xml("module_plug", RACK_HANG_X, yb, HUB_PEG_Z,
                "0.25 0.27 0.30 1", face=plug_face)}
    {module_xml("module_pen", RACK_HANG_X, yc, HUB_PEG_Z,
                "0.65 0.35 0.30 1", face=pen_face)}
    {module_xml("module_claw", RACK_HANG_X, yd, HUB_PEG_Z,
                "0.35 0.55 0.35 1", face=claw_face)}
    {module_xml("module_seed", RACK_HANG_X, ye, HUB_PEG_Z,
                "0.55 0.45 0.25 1", face=seed_face)}
    {seed_bodies_xml(RACK_HANG_X, ye,
                     HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R - PEG_ABOVE_BODY)}

    <!-- Something to pick up. A 26 mm, 60 g block on open floor: small enough
         for a 70 mm jaw span, heavy enough that dropping it is visible. -->
    <body name="pickup" pos="1.10 0.60 0.013">
      <freejoint/>
      <geom name="pickup_box" type="box" size="0.013 0.013 0.013" mass="0.06"
            friction="1.2" rgba="0.90 0.60 0.20 1"/>
    </body>
    <camera name="hub_watch" pos="0.85 -0.85 0.70"
            xyaxes="0.707 0.707 0 -0.32 0.32 0.89"/>
  </worldbody>
  <actuator>
    {pen_actuator_xml()}
    {claw_actuator_xml()}
    {dispenser_actuator_xml()}
  </actuator>
</mujoco>
"""
  with open(path, "w") as fh:
    fh.write(xml)


# ---- room placement (room_hub.xml) -----------------------------------------
RACK_ROOM_POS = (-0.90, 5.99)   # rack origin: north wall of room 1, braces
                                # touching the wall face at y=5.99
RACK_ROOM_YAW = -90.0           # rack local +x (its outward normal) -> -y:
                                # facing south, into room 1


def rack_frame_to_world(x_local: float, y_local: float,
                        pos: tuple[float, float] | None = None,
                        yaw_deg: float | None = None) -> tuple[float, float]:
  """A rack-frame point in room coordinates (room_hub's placement by
  default; pass pos/yaw for a rack placed anywhere else, e.g. home_world)."""
  px, py = RACK_ROOM_POS if pos is None else pos
  th = math.radians(RACK_ROOM_YAW if yaw_deg is None else yaw_deg)
  c, s = math.cos(th), math.sin(th)
  return (px + x_local * c - y_local * s,
          py + x_local * s + y_local * c)


def rack_and_modules_xml(pos: tuple[float, float], yaw_deg: float) -> str:
  """The rack body + its four hanging modules at an arbitrary room pose --
  the one worldbody snippet every world that has a hub shares (hub_world,
  room_hub via hub_rack.xml, home_world)."""
  ya, yb, yc, yd, ye = HUB_STATION_YS
  lcd_face, plug_face, pen_face, claw_face, seed_face = _module_faces()
  ax, ay_ = rack_frame_to_world(RACK_HANG_X, ya, pos, yaw_deg)
  bx, by_ = rack_frame_to_world(RACK_HANG_X, yb, pos, yaw_deg)
  cx, cy_ = rack_frame_to_world(RACK_HANG_X, yc, pos, yaw_deg)
  dx, dy_ = rack_frame_to_world(RACK_HANG_X, yd, pos, yaw_deg)
  ex, ey_ = rack_frame_to_world(RACK_HANG_X, ye, pos, yaw_deg)
  body_z = HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R - PEG_ABOVE_BODY
  return f"""{_rack_body_xml((pos[0], pos[1], 0.0), yaw_deg)}
    {module_xml("module_lcd", ax, ay_, HUB_PEG_Z,
                "0.20 0.45 0.75 1", face=lcd_face, yaw_deg=yaw_deg)}
    {module_xml("module_plug", bx, by_, HUB_PEG_Z,
                "0.25 0.27 0.30 1", face=plug_face, yaw_deg=yaw_deg)}
    {module_xml("module_pen", cx, cy_, HUB_PEG_Z,
                "0.65 0.35 0.30 1", face=pen_face, yaw_deg=yaw_deg)}
    {module_xml("module_claw", dx, dy_, HUB_PEG_Z,
                "0.35 0.55 0.35 1", face=claw_face, yaw_deg=yaw_deg)}
    {module_xml("module_seed", ex, ey_, HUB_PEG_Z,
                "0.55 0.45 0.25 1", face=seed_face, yaw_deg=yaw_deg)}
    {seed_bodies_xml(ex, ey_, body_z, yaw_deg)}"""


def write_hub_rack(path: str = "models/hub_rack.xml") -> None:
  """The rack + modules as a room include, placed on room 1's north wall."""
  tag_ids = write_tag_pngs()
  xml = f"""<!-- GENERATED by pluggybot.rack.coupling.write_hub_rack().
     Regenerate: uv run python -m pluggybot.rack.coupling
     The tool rack + hanging modules placed against room 1's north wall,
     included by room_hub.xml. Same generator as hub_world.xml's rack. -->
<mujocoinclude>
  <asset>
    {asset_xml(tag_ids)}
  </asset>
  <worldbody>
    {rack_and_modules_xml(RACK_ROOM_POS, RACK_ROOM_YAW)}
  </worldbody>
  <actuator>
    {pen_actuator_xml()}
    {claw_actuator_xml()}
    {dispenser_actuator_xml()}
  </actuator>
</mujocoinclude>
"""
  with open(path, "w") as fh:
    fh.write(xml)


def run_pick(dy: float = 0.0, dz: float = 0.0, yaw_deg: float = 0.0,
             tool_mass: float = 0.12, shake_accel: float = 0.0,
             n_frames: int = 0, noslip: int = 0) -> tuple[dict, list[np.ndarray]]:
  """One scripted pick: approach, lift, retreat (then optionally shake).

  Success = the tool left the trays and followed the carrier out. shake_accel
  (m/s^2) oscillates the advance axis after the pick to probe retention.
  Returns (result, frames): picked, carried (still held after any shake),
  max contact force, and the tool's final peg seating error in the fork.
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(dy, dz, yaw_deg, tool_mass, noslip))
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
              tool_mass: float = 0.12, n_frames: int = 0, noslip: int = 0,
              ) -> tuple[dict, list[np.ndarray]]:
  """Full pick-and-RETURN cycle: take the tool off the hub, carry it out,
  bring it back, hang it up, leave empty. The offsets apply to BOTH
  approaches (the same navigation errors happen twice).

  Success ("returned") = the tool ends hanging in the trays again -- peg
  near its rest height, body near the hub centreline -- with the fork clear.
  """
  model = mujoco.MjModel.from_xml_string(
    scene_xml(dy, dz, yaw_deg, tool_mass, noslip))
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
