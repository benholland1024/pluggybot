"""The tool coupling (milestone 8): a module hangs by a split PEG, and a
fork of V-notches takes it -- the interface every tool presents, on the
rover's lift until #376 and on the quadruped's arm since #405 (`legs.arm`,
whose fork takes the same peg, longer).

  tool module   a plate hanging by a long horizontal PEG AXLE (along y)
  rack          two upward-open V-TRAYS catch the peg near its ends
  fork          two prongs tipped with upward-open V-NOTCHES that grab the
                peg OUTBOARD of the trays, so the lateral capture envelope
                is set by peg overhang, not by machined clearances

Gravity is the latch; V depth is the retention. The peg's two conductors
are the module's power (`module_power_state`). A V-notch is concave, and
MuJoCo collides convex pieces only, so each is two tilted boxes.

Here too: the rover's coupling rig (`scene_xml`, the noslip policy's
test compiles it), the contact list read as an array (`touching`), and the
bay index space (`STATION_YS`) both of the quadruped's rows read their bays
by.
"""

import math

import numpy as np

from pluggybot.rack.tags import (
  SMALL_TAG_SIZE, plate_half_extent,
)

# -- geometry (meters) --------------------------------------------------------
#: The rover's plate + 150 mm peg budget; the `face` adds its own on top.
#: `PEG_MASS` is the rod's share, split 8 + 8 + 4 g over the two conductors
#: and the bush by `peg_xml`. The quadruped's tools are these with the
#: 220 mm peg (`legs.rack.MODULE_MASS`, `PEG_MASS`). The envelope a new tool
#: must fit is the arm's (`legs/arm.py`; `workshop/validate.py` refuses by
#: it; ToolPattern.md §2).
MODULE_MASS = 0.12
PEG_MASS = 0.02

#: DESIGN DECISION, NOT MEASURED: what the two-pole peg coupling delivers,
#: at the 12 V pack. 12 W is 1 A. Was 6 W (0.5 A), sized so one hobby servo
#: at stall (4.8 W) plus the module's ESP32 fits and nothing else does --
#: which #199's parts made binding: a servo and an eye together were refused.
#: Raised 2026-09-15 (Ben) on this argument: a light steel-on-steel point
#: contact is tens of milliohms to ~0.1 ohm, so 1 A dissipates ~0.1 W at
#: the contact and heating is not the limit; what limits plain steel is
#: voltage drop and fretting/oxidation, and 1 A is inside what such a
#: contact is rated for. The validator sums each part's CEILING (stall,
#: flash at full) as if simultaneous, so the budget is already pessimistic.
#: Still a paper number until the built coupling is measured; if the
#: physical build ever wants more, the upgrade that keeps "the peg IS the
#: connector" is brass sleeves on the conductor segments and plated
#: V-plates, or a sprung contact in the V (Parts.md, "Module power
#: contacts"). The holding capacitor (~200 ms) is the other half.
PEG_POWER_W = 12.0
PEG_R = 0.003           # peg axle radius (6 mm rod)
PEG_HALF = 0.075        # peg half-length: 150 mm rod
TOOL_HALF_Y = 0.020     # tool body half-width: 40 mm plate
TOOL_HALF_Z = 0.030
TOOL_HALF_X = 0.010
PEG_ABOVE_BODY = 0.022  # peg axis above tool body centre
PEG_Z = 0.150           # peg rest height on the rig's shelf (spike-local)
TRAY_Y = 0.040          # shelf V-trays, inboard pair
FORK_Y = 0.058          # fork prongs grab outboard of the trays
V_HALF_LEN = 0.011      # tilted plate half-length -> ~8 mm usable V depth
V_THICK = 0.003
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
LAT_STIFFNESS = 150.0   # N/m lateral compliance (a guess: the rover's RCC wrist)
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
              tool_mass: float = MODULE_MASS, noslip: int = 0,
              face: str = "", actuators: str = "",
              push: float = PUSH_FORCE, back_x: float = -0.018,
              peg_z: float = PEG_Z) -> str:
  """Hub shelf + hanging tool at the origin; carrier approaching from +x.

  dy/dz: lateral/vertical offset of the carrier's approach line (m).
  yaw_deg: angular misalignment about z. tool_mass: the module's mass budget
  under test (the interface spec caps it -- confirm the latch holds it).
  noslip: `noslip_iterations` for the run -- the peg seats by SLIDING into
  its V, so the coupling is the behavior a global noslip policy is most
  likely to break, and the spike must be able to measure it under that
  policy (issue #3).
  face / actuators: a module's own parts and servos, strings in its frame.
  With a face the tool carries the real split peg (`peg_xml`) so the rig
  can read its poles; without, the plain rod the envelope was measured
  with.
  push / back_x / peg_z: the approach force, the back wall's x and the peg's
  rest height. The defaults are the SPIKE's -- a 10 N push reacted by a
  wall 4 mm behind the plate, a stand-in for the robot's measured approach
  depth, with the peg at 0.15 -- and are what the ±4 mm envelope was swept
  with.
  """
  peg_len_color = "0.75 0.75 0.78 1"
  # The tool spawns HANGING: peg resting at the tray vertices (+ its radius).
  peg_rest_z = peg_z - TRAY_VERTEX_DROP + PEG_R
  tool_body_z = peg_rest_z - PEG_ABOVE_BODY

  trays = "\n      ".join(
    _v_notch_xml(f"tray_{lbl}_", (0.0, s * TRAY_Y, peg_z - TRAY_VERTEX_DROP),
                 0.008, "0.55 0.57 0.60 1")
    + f'\n      <geom name="tray_{lbl}_post" type="box" '
      f'size="0.006 0.008 {(peg_z - 0.02) / 2:.4f}" '
      f'pos="0 {s * TRAY_Y:.4f} {(peg_z - 0.02) / 2:.4f}" '
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
  <visual><global offwidth="960" offheight="720"/><quality offsamples="0"/></visual>
  <default>
    <geom friction="0.4" solref="0.005 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.2 0.5" dir="-0.5 0.35 -0.8"/>
    <light pos="0.2 0.3 0.4" dir="-0.4 -0.6 -0.7"/>
    <camera name="side" pos="0.30 -0.30 {peg_z + 0.13:.2f}" xyaxes="0.707 0.707 0 -0.25 0.25 0.93"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>

    <!-- Hub shelf: back wall + two V-tray posts -->
    <body name="hub">
      <geom name="hub_back" type="box" size="0.004 0.10 {peg_z / 1.5:.4g}"
            pos="{back_x:.4g} 0 {peg_z / 1.5:.4g}" rgba="0.60 0.62 0.65 1"/>
      {trays}
    </body>

    <!-- Tool module: plate hanging by its peg axle in the trays -->
    <body name="tool" pos="0 0 {tool_body_z:.4f}">
      <freejoint/>
      <geom name="tool_body" type="box"
            size="{TOOL_HALF_X} {TOOL_HALF_Y} {TOOL_HALF_Z}"
            mass="{tool_mass - PEG_MASS:.3f}" rgba="0.20 0.45 0.75 1"/>
      {peg_xml("tool") if face else
       f'<geom name="tool_peg" type="cylinder" size="{PEG_R} {PEG_HALF}" '
       f'zaxis="0 1 0" pos="0 0 {PEG_ABOVE_BODY:.4f}" mass="{PEG_MASS}" '
       f'rgba="{peg_len_color}"/>'}
      {face}
    </body>

    <!-- Carrier: approach rail with force-limited x + position-servo lift z,
         the fork hanging through compliant y/yaw (arm + base flex). -->
    <body name="rail" pos="{START_X:.4f} {dy:.4f} {peg_z + dz:.4f}"
          euler="0 0 {yaw_deg:.3f}">
      <inertial pos="0 0 0" mass="0.10" diaginertia="2e-5 2e-5 2e-5"/>
      <joint name="advance" type="slide" axis="-1 0 0" damping="{push / 0.03:.1f}"/>
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
    <motor name="push" joint="advance" ctrlrange="-{push} {push}"/>
    <position name="lift" joint="lift" kp="400" kv="30"
              ctrlrange="-0.05 0.08" forcerange="-30 30"/>
    {actuators}
  </actuator>
</mujoco>"""


def module_xml(name: str, x: float, y: float, peg_z: float,
               rgba: str, mass: float = MODULE_MASS, face: str = "",
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
            mass="{mass - PEG_MASS:.3f}" rgba="{rgba}"/>
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


# ---- reading the contact list (rooftop-media-2026 #296) -----------------------
#
# ⚠ MEASURED: the electrical criteria below and the rover's two chassis
# scans ran EVERY physics step, per robot,
# as Python loops over `data.contact[i]` -- a pybind struct per contact,
# 83 contacts at rest in the home world, 500 steps a sim-second, two
# robots: ~330 000 struct constructions a sim-second. A py-spy profile of
# the served pair put 49 % of the physics thread there against 13 % in
# `mj_step`. Every scan now reads `data.contact.geom`, the (ncon, 2) int
# view, once; and a geom's id is resolved by name ONCE per model, because
# `model.geom(name)` is a string lookup and was paid twice a step.


_GEOM_IDS: dict[tuple[int, str], tuple[object, int | None]] = {}


def geom_id(model, name: str) -> int | None:
  """`model.geom(name).id`, or None where no such geom -- cached per
  model OBJECT (the entry holds the model, so a recycled `id()` cannot
  alias a recompiled world, whose ids differ)."""
  key = (id(model), name)
  hit = _GEOM_IDS.get(key)
  if hit is not None and hit[0] is model:
    return hit[1]
  try:
    gid = int(model.geom(name).id)
  except KeyError:
    gid = None
  _GEOM_IDS[key] = (model, gid)
  return gid


def contact_pairs(data) -> np.ndarray:
  """The active contacts' geom pairs, an (ncon, 2) int array."""
  return data.contact.geom[:data.ncon]


def touching(data, a: int, others) -> bool:
  """Is geom `a` in contact with any geom in `others`?

  The rows holding `a` first, then those few in Python: `np.isin`'s fixed
  cost, twice a call, twice a step per robot for the tool's poles, was 11 %
  of the served pair's physics thread (issue #385)."""
  g = contact_pairs(data)
  mine = g[(g[:, 0] == a) | (g[:, 1] == a)]
  if mine.shape[0] == 0:
    return False
  others = set(others)
  return any((x == a and y in others) or (y == a and x in others)
             for x, y in mine.tolist())


def module_power_state(model, data, name: str = "module_lcd",
                       prefix: str = "", poles: dict | None = None) -> dict:
  """Is this module's coupling conducting, and if not, which pole is open?

  `prefix` names WHOSE fork (issue #167): the first robot's plates are the
  bare `fork_v*` geoms, the second's carry its prefix. A module seated on
  the other robot's fork is not powered by this one.

  The same lesson as the dock's pins (`legs.dock.dock_charge_contact`): an
  electrical criterion beats a positional one. Milestone 6 burned
  four position-based seat detectors before the charging voltage settled it;
  here the question "is the tool powered" has exactly one honest answer, and
  it is the same one the hardware will have.

  Reporting the poles separately is deliberate. A half-seated coupling --
  one conductor on, one off -- is a real failure mode of a two-point latch
  (the feelers taught this: one prong 39 mm short wrecked everything while
  the other looked perfect), and a bare boolean would hide it as "off".
  """
  plates_by_side = poles or FORK_POLE_GEOMS
  poles = {}
  for side, plates in plates_by_side.items():
    peg = geom_id(model, f"{name}_peg_{side}")
    plate_ids = [geom_id(model, prefix + g) for g in plates]
    if peg is None or any(p is None for p in plate_ids):
      poles[side] = False
      continue
    poles[side] = touching(data, peg, plate_ids)
  return {"left": poles["l"], "right": poles["r"],
          "powered": poles["l"] and poles["r"]}


def module_power_contact(model, data, name: str = "module_lcd",
                         prefix: str = "") -> bool:
  """Both poles conducting -- the module is coupled and powered."""
  return module_power_state(model, data, name, prefix)["powered"]


# ---- the rack's layout (the "bike rack for tools", designed with Ben) -------
# The rover's rack (deleted with it in #376; `rover-final` has it): what is
# left is the INDEX SPACE every bay lives in, which the quadruped's rows
# read their bays by (`legs.rack.RackSpec.stations`, `legs.swap`).
HUB_STATION_YS = (0.125, -0.125, 0.375, 0.625, 0.875)  # tool bays at 0.25 m
                          # pitch, APPENDED rather than inserted in y order:
                          # the bay<->tag pairing is by index
# ---- the BUILT-TOOL rail (issue #277) ---------------------------------------
# For the tools the robot builds: the hand-built tools are permanent and a
# built tool never takes one of their bays. Its stations follow the first
# rack's, so `STATION_YS` is one index space for every bay. On legs the
# rail is `legs.rack.BUILT` (#407), at its own positions on the rack's
# board: these y's are its stations' place in the index space and nothing
# more.
BUILT_STATION_YS = (1.175, 1.425, 1.675)  # the rover's three bays
#: EVERY BAY, BY INDEX: the hand-built five, then the built rail's -- so it
#: is APPENDED to, never reordered; a bay index in
#: `HubLifecycle.rack_inventory` indexes this.
STATION_YS = HUB_STATION_YS + BUILT_STATION_YS
SMALL_PLATE_HALF = plate_half_extent(SMALL_TAG_SIZE)


def bay_prefix(i: int) -> str:
  """Geom-name prefix for bay i: baya_, bayb_, bayc_, ... (the built
  rail's are bayf_, bayg_, bayh_: one letter sequence over STATION_YS)."""
  return f"bay{chr(ord('a') + i)}_"


def built_bay_index(bay: int) -> int:
  """A built-rail bay (0..len(BUILT_STATION_YS)-1, the letter the robot
  names) as its index into STATION_YS -- the space `rack_inventory` and the
  tag pairing use. Refused, not clamped, outside the rail."""
  if not 0 <= bay < len(BUILT_STATION_YS):
    raise ValueError(f"no built-rail bay {bay}; it has {len(BUILT_STATION_YS)}")
  return len(HUB_STATION_YS) + bay


def is_built_bay(index: int) -> bool:
  """Whether a STATION_YS index is on the built rail."""
  return index >= len(HUB_STATION_YS)


#: The V a bay's presence switch sits in (issue #351): ONE switch a bay, in
#: the +y tray, so a module hanging by its other end alone reads absent --
#: as it would on a rack with one switch a bay.
BAY_SWITCH_PLATES = ("tray_l_a", "tray_l_b")


def bay_switches(model, data) -> tuple[bool | None, ...]:
  """Each bay's presence switch, by `STATION_YS` index: True while anything
  rests in its V, None where this world has no such bay (no built rail).

  What the rack reports over the network, and all it reports (issue #351):
  a bay is occupied, never by WHICH module -- a module hung in another's
  bay presses that bay's switch. Read off the contact list, as the bumper
  is (catalog `bay_switch`); the switch's force is not modelled.

  No debounce, MEASURED: on home, 2.0 M samples over a 472 s charge press
  and a carry, a switch disagreed with its module only while that module
  was being swapped (at most 44 ms) and for one step as the world settled
  in its first second, before any decision."""
  g = contact_pairs(data)
  out: list[bool | None] = []
  for i in range(len(STATION_YS)):
    ids = [geom_id(model, bay_prefix(i) + p) for p in BAY_SWITCH_PLATES]
    if any(gid is None for gid in ids):
      out.append(None)
      continue
    out.append(bool(g.shape[0]) and bool(np.isin(g, ids).any()))
  return tuple(out)


#: The built-tool rail's body name (issue #277): what a world that has one
#: carries, and what `HubLifecycle` reads to know whether a tool can hang.
BUILT_RACK_BODY = "rack_built"


#: Near-rigid contact for what is gripped and stacked: the challenge blocks
#: carry it, or the grade is the solver's (CLAUDE.md, "A challenge is a
#: task...").
GRIP_SOLIMP = "0.99 0.999 0.0001"


def rack_frame_to_world(x_local: float, y_local: float,
                        pos: tuple[float, float],
                        yaw_deg: float) -> tuple[float, float]:
  """A rack-frame point in world coordinates, for a rack at `pos`, `yaw_deg`."""
  px, py = pos
  th = math.radians(yaw_deg)
  c, s = math.cos(th), math.sin(th)
  return (px + x_local * c - y_local * s,
          py + x_local * s + y_local * c)
