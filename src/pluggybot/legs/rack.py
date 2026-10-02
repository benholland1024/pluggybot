"""The rack for legs (issue #378): tools hang at the arm's working height,
and a row of tags tells the robot where each bay is.

A tool is the rover's module -- a plate and a split peg whose two
conductors are the coupling's poles (`rack.coupling.peg_xml`) -- on a peg
`PEG_HALF` long either side, so the arm's fork (`legs.arm`) can take it
further out than the rack's trays hold it. The rack holds each peg in two
V-trays hung from a rail above, inboard of where the fork takes it, with
nothing under a tool: its business end and its pendant hang in free air.

The tags are a pair a bay, either side of it, at the height the nose
camera reads from the working stance. Their layout is the rack's drawing
(`tag_layout`), and the rack's pose is fitted to the tags together
(`fit_rack`, the dock's Kabsch fit and the #88 rule: the facing comes from
the baseline, never one tag's yaw). SimNotes, "The quadruped's arm", has
the tables; `scripts/arm_spike.py` flies them.

Frames: the RACK's -- the origin on the floor under the middle bay's peg,
+x out of the rack toward a robot at work (the robot faces -x), y along the
rail, z up.
"""

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.rack.coupling import (PEG_ABOVE_BODY, PEG_FRICTION, PEG_INSUL_HALF,
                                     PEG_R, TOOL_HALF_X, TOOL_HALF_Y, TOOL_HALF_Z,
                                     bay_prefix, geom_id, touching)
from pluggybot.rack.tags import LEGS_RACK_TAG_IDS, LEGS_RACK_TAG_SIZE

#: The peg's half-length: the rover's 75 mm plus the fork's lateral capture
#: (`legs.arm.ForkSpec`): the fork takes it at +-85 mm, the rack at +-45.
PEG_HALF = 0.110
#: The rack's two V-trays per bay, at +-TRAY_Y: outboard of the tool's
#: plate (+-TOOL_HALF_Y), inboard of the fork. ⚠ The plate's room between
#: them is what a return turns on: at +-35 mm (9 mm of room) a tool hung
#: back from 10-15 mm off set its plate on a tray's corner, whatever the
#: fork's stops allowed; at +-45 (19 mm) it hangs back from -15..+20 mm.
TRAY_Y = 0.045
#: A tray's V: the rover's (`rack.coupling`), its vertex this far under the
#: peg's axis at rest.
V_HALF_LEN = 0.011
V_THICK = 0.003
TRAY_VERTEX_DROP = PEG_R * math.sqrt(2)
#: A tray's half-width along y.
TRAY_HALF_W = 0.006


@dataclass(frozen=True)
class RackSpec:
  """A rack for legs. Lengths in metres."""

  #: Tool bays along y, the rack's frame; their count is the rack's.
  bays: tuple[float, ...] = (-0.30, 0.0, 0.30)
  #: The pegs' axis, above the floor.
  peg_z: float = 0.50
  #: The rail the trays hang from, above the pegs; the back board behind.
  rail_z: float = 0.62
  back_x: float = -0.09
  #: The tags: a pair a bay, `tag_dy` either side of it at `tag_z` on the
  #: back board's face -- under the pegs' ends, clear of a pendant's
  #: +-35 mm, and 20 deg off the nose camera's axis from the working pose.
  #: ⚠ Between the bays (+-150 mm) they sat 25 deg off it: 2.7 deg of yaw
  #: took one past the camera's 33.5 deg edge, and a fetch at the bay had
  #: one tag and no fit.
  tag_z: float = 0.40
  tag_dy: float = 0.075
  tag_size: float = LEGS_RACK_TAG_SIZE
  #: The trays' y (either side of a bay) and the tools' peg: the module
  #: constants, carried here so the spike can fly the rover's (`--rover`).
  tray_y: float = TRAY_Y
  peg_half: float = PEG_HALF

  @property
  def tag_ys(self) -> tuple[float, ...]:
    return tuple(y for b in self.bays for y in (b - self.tag_dy, b + self.tag_dy))


DEFAULT = RackSpec()
#: The rack's tag ids, two a bay in `RackSpec.tag_ys`' order (`rack.tags`,
#: the one registry of ids).
RACK_TAG_IDS = LEGS_RACK_TAG_IDS


def tag_layout(spec: RackSpec = DEFAULT) -> dict[int, tuple[float, float, float]]:
  """Each rack tag's printed face in the rack frame, by id: the drawing."""
  return {i: (spec.back_x + 0.006, y, spec.tag_z)
          for i, y in zip(RACK_TAG_IDS, spec.tag_ys)}


def _f(v: float) -> str:
  return f"{v:.6g}"


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def _quat_y(angle: float) -> str:
  return _v(math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def _quat_x(angle: float) -> str:
  return _v(math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def peg_xml(name: str, z: float = PEG_ABOVE_BODY, peg_half: float = PEG_HALF) -> str:
  """The split peg, `peg_half` either side: two conductors either side of
  an insulated centre, named as `rack.coupling.peg_xml` names them
  (`{name}_peg_l`, `_r`, `_insul`), so the rover's criterion reads it."""
  cond_half = (peg_half - PEG_INSUL_HALF) / 2
  cond_kg = (peg_kg(peg_half) - INSUL_KG) / 2
  out = []
  for lbl, s in (("l", 1), ("r", -1)):
    out.append(f'<geom name="{name}_peg_{lbl}" type="cylinder" '
               f'size="{_v(PEG_R, cond_half)}" zaxis="0 1 0" '
               f'pos="{_v(0, s * (PEG_INSUL_HALF + cond_half), z)}" mass="{_f(cond_kg)}" '
               f'friction="{_f(PEG_FRICTION)}" priority="1" rgba="0.75 0.75 0.78 1"/>')
  out.append(f'<geom name="{name}_peg_insul" type="cylinder" '
             f'size="{_v(PEG_R, PEG_INSUL_HALF)}" zaxis="0 1 0" pos="{_v(0, 0, z)}" '
             f'mass="{_f(INSUL_KG)}" rgba="0.12 0.12 0.14 1"/>')
  return "\n      ".join(out)


#: The peg's mass: the rover's 20 g for its 150 mm (`rack.coupling.PEG_MASS`,
#: split 8 + 8 + 4 g over the conductors and the bush) at the same grams a
#: millimetre, the bush unchanged.
INSUL_KG = 0.004


def peg_kg(peg_half: float = PEG_HALF) -> float:
  return INSUL_KG + (0.020 - INSUL_KG) * (2 * peg_half - 2 * PEG_INSUL_HALF) / (0.150 - 2 * PEG_INSUL_HALF)


PEG_MASS = peg_kg()
#: The plate + peg budget: the rover's `MODULE_MASS` less its 150 mm peg,
#: plus this one.
MODULE_MASS = 0.12 - 0.020 + PEG_MASS


def tool_xml(name: str, pos=(0.0, 0.0, 0.0), yaw: float = 0.0,
             mass: float = MODULE_MASS, face: str = "",
             rgba: str = "0.20 0.45 0.75 1", peg_half: float = PEG_HALF,
             group: int = 0) -> str:
  """A tool module as a free body whose PEG'S AXIS is at `pos` (world), its
  +x face (the robot's side) turned `yaw` about z. `face` is the tool's own
  parts in the module's frame (`rack.coupling._module_faces`'). `group` is
  its geoms' visual group: ⚠ a policy's height scan (`legs.policy`) reads
  group 0 as TERRAIN, so a tool the robot carries over its nose reads as a
  0.4 m obstacle unless the robot filters it, as it filters its own body."""
  x, y, z = pos
  bz = z - PEG_ABOVE_BODY
  q = _v(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
  return f"""
    <body name="{name}" pos="{_v(x, y, bz)}" quat="{q}" childclass="{name}_tool">
      <freejoint name="{name}_free"/>
      <geom name="{name}_body" type="box" size="{_v(TOOL_HALF_X, TOOL_HALF_Y, TOOL_HALF_Z)}"
            mass="{_f(mass - peg_kg(peg_half))}" rgba="{rgba}"/>
      {peg_xml(name, peg_half=peg_half)}
      {face}
    </body>"""


def tool_default(name: str, group: int = 0) -> str:
  """The `<default>` class a `tool_xml` body's geoms take (its group)."""
  return f'<default class="{name}_tool"><geom group="{group}"/></default>'


def _v_notch(prefix: str, x: float, y: float, z: float, half_w: float) -> str:
  off = V_HALF_LEN / math.sqrt(2)
  out = []
  for s, lbl in ((-1, "a"), (1, "b")):
    out.append(f'<geom name="{prefix}{lbl}" type="box" '
               f'size="{_v(V_HALF_LEN, half_w, V_THICK)}" '
               f'pos="{_v(x + s * off, y, z + off - V_THICK * math.sqrt(2))}" '
               f'quat="{_quat_y(-s * math.pi / 4)}" rgba="0.55 0.57 0.60 1"/>')
  return "\n      ".join(out)


def rack_xml(spec: RackSpec = DEFAULT, pos=(0.0, 0.0), yaw: float = 0.0,
             name: str = "rack") -> str:
  """The rack as one static body: a back board, a rail, and each bay's two
  trays on brackets hung from the rail; its tags on the board. The tags'
  materials are `rack.tags.asset_xml(RACK_TAG_IDS)`'s. A bay's parts are
  named as the rover's are, `bay<letter>_` (`coupling.bay_prefix`): its
  presence switch is `coupling.bay_switches`' reading, the one the rack
  view is built from (#351, #405)."""
  from pluggybot.rack.tags import plate_half_extent
  s = spec
  half_y = max(abs(y) for y in s.tag_ys) + s.tag_size + 0.05
  g = [f'<geom name="{name}_board" type="box" size="{_v(0.006, half_y, s.rail_z / 2)}" '
       f'pos="{_v(s.back_x, 0, s.rail_z / 2)}" rgba="0.85 0.85 0.82 1"/>',
       f'<geom name="{name}_rail" type="box" size="{_v(0.03, half_y, 0.01)}" '
       f'pos="{_v(s.back_x + 0.03, 0, s.rail_z)}" rgba="0.45 0.47 0.50 1"/>']
  vz = s.peg_z - TRAY_VERTEX_DROP
  for k, by in enumerate(s.bays):
    bay = bay_prefix(k)
    for side, lbl in ((1, "l"), (-1, "r")):
      ty = by + side * s.tray_y
      g.append(_v_notch(f"{bay}tray_{lbl}_", 0.0, ty, vz, TRAY_HALF_W))
      # The bracket: from the rail down behind the tray to below its V.
      top = s.rail_z - 0.01
      bot = vz - 0.012
      g.append(f'<geom name="{bay}bracket_{lbl}" type="box" '
               f'size="{_v(0.004, TRAY_HALF_W, (top - bot) / 2)}" '
               f'pos="{_v(-0.022, ty, (top + bot) / 2)}" rgba="0.45 0.47 0.50 1"/>')
      g.append(f'<geom name="{bay}arm_{lbl}" type="box" '
               f'size="{_v(0.012, TRAY_HALF_W, 0.003)}" '
               f'pos="{_v(-0.012, ty, bot)}" rgba="0.45 0.47 0.50 1"/>')
  half = plate_half_extent(s.tag_size)
  for tag_id, (x, y, z) in tag_layout(s).items():
    g.append(f'<geom name="{name}_tag{tag_id}" type="box" size="{_v(0.001, half, half)}" '
             f'pos="{_v(x, y, z)}" contype="0" conaffinity="0" material="tagmat{tag_id}"/>')
  body = "\n      ".join(g)
  return (f'<body name="{name}" pos="{_v(pos[0], pos[1], 0)}" '
          f'quat="{_v(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))}">\n'
          f'      {body}\n    </body>')


# ---- the served rack (#405) -----------------------------------------------------

#: The served world's rack body: not the rover's `rack`, whose name the site
#: draws with the rover's rack (`visualHints`).
RACK_BODY = "tool_rack"
#: The tools it ships, by bay (issue #405): three of #378's four survivors,
#: one a bay -- the LCD, the pen (#406) and the claw (#407's); the seed
#: dispenser waits for a fourth. The rover's module names, so a tool is the
#: same tool to everything that names it (the rack view, the lost-tool
#: clock, a program's `fetch`, an admin's reset).
TOOL_BAYS = {"module_lcd": 0, "module_pen": 1, "module_claw": 2}
#: Each tool's mass, kg: the rover's module (`models/home_world.xml`) with
#: its 150 mm peg (20 g) swapped for this one. Until #407 rebuilds them the
#: LCD and the claw are a plate, a peg and a face that says which
#: (`tool_face`), the mass on the plate: the envelope's "on its peg" case.
#: The pen is built (#406, `pen_face`), its parts' masses out of its plate's.
TOOL_KG = {"module_lcd": 0.1426 - 0.020 + PEG_MASS,
           "module_pen": 0.1816 - 0.020 + PEG_MASS,
           "module_claw": 0.2106 - 0.020 + PEG_MASS}
#: The LCD's screen, half-extents: named `module_lcd_screen`, so the served
#: face finds it (`tools/screen.py`).
SCREEN_HALF = (0.002, 0.028, 0.038)


# ---- the pen (#406) ----------------------------------------------------------------

#: The pen's sideways carriage: an Actuonix L12-100 (Parts.md,
#: `slide_l12_100`), its 100 mm stroke centred on the module -- the one axis
#: an arm moving in its own plane lacks. A lead screw at 50:1: it holds where
#: it was sent (12 N back-drives it), at most 25 mm/s and 22 N.
PEN_TRAVEL = 0.050
PEN_SPEED = 0.025
PEN_FORCE_N = 22.0
#: The rail (the slide's body) runs UNDER the plate, across it, and the
#: carriage rides it with the pen's line above the rail (module frame, m).
#: ⚠ The module balances on its peg: the rover's carriage stood off in front
#: of its plate (`rover-final`), 56 g 26 mm ahead, which on the legs' rack
#: hung the tool 16 deg off plumb (`rack.on_bay` asks 2), so the rail sits
#: 2 mm behind the peg. The block's top stays under the plate's bottom.
PEN_MOUNT_X = 0.002
PEN_RAIL_Z = -0.048
PEN_LINE_DZ = 0.012
#: The pen past its holder, m: its tip 48 mm ahead of the peg and 58 under
#: it, 36 mm short of the rack's back board with the tool hung, which leans
#: 0.6 deg (its CoM 0.3 mm ahead of the peg).
PEN_LEN = 0.042
#: The pen's radius, m: its point is the shaft's rounded end, this far past
#: the `pen_tip` site at the end's centre -- what touches a board first.
PEN_TIP_R = 0.0025
#: The sprung quill: the pen's pressure is the spring's, not the arm's
#: position. Soft and long, the rover's: at 200 N/m the pen lifted off where
#: the arm drooped (0 % ink at the top of a figure); at 60 N/m, 10 mm in is
#: 0.6 N.
PEN_QUILL_TRAVEL = 0.020
PEN_QUILL_STIFFNESS = 60.0
#: The parts' masses, kg (the rover's): the slide's body is the rail, its rod
#: end with the pen's holder the carriage, the quill the pen.
PEN_RAIL_KG, PEN_CARRIAGE_KG, PEN_QUILL_KG = 0.020, 0.030, 0.006
PEN_PARTS_KG = PEN_RAIL_KG + PEN_CARRIAGE_KG + PEN_QUILL_KG
#: Its actuator, joints, shaft and tip, by name: the world's, as the module is.
PEN_ACTUATOR = "pen_carriage"
PEN_CARRIAGE_JOINT = "pen_carriage_joint"
PEN_QUILL_JOINT = "pen_quill_joint"
PEN_SHAFT = "module_pen_shaft"
PEN_TIP = "pen_tip"


def pen_face() -> str:
  """The pen's parts in the module's frame: the rail, the carriage sliding
  along it (the peg's axis) and on the carriage the quill, sprung toward the
  board. Ink is a contact of the shaft (`tools.drawing.pen_on_board`)."""
  from pluggybot.rack.coupling import GRIP_SOLIMP
  block_lo, block_hi = -0.018, 0.004            # about the pen's line
  return (
    f'<geom name="module_pen_rail" type="box" '
    f'size="{_v(0.004, PEN_TRAVEL + 0.012, 0.004)}" pos="{_v(PEN_MOUNT_X, 0, PEN_RAIL_Z)}" '
    f'mass="{_f(PEN_RAIL_KG)}" rgba="0.55 0.57 0.60 1"/>'
    f'<body name="module_pen_carriage" pos="{_v(PEN_MOUNT_X, 0, PEN_RAIL_Z)}">'
    f'<joint name="{PEN_CARRIAGE_JOINT}" type="slide" axis="0 1 0" '
    f'range="{_v(-PEN_TRAVEL, PEN_TRAVEL)}" damping="2"/>'
    f'<geom name="module_pen_block" type="box" '
    f'size="{_v(0.008, 0.010, (block_hi - block_lo) / 2)}" '
    f'pos="{_v(0, 0, PEN_LINE_DZ + (block_hi + block_lo) / 2)}" '
    f'mass="{_f(PEN_CARRIAGE_KG)}" rgba="0.30 0.32 0.36 1"/>'
    f'<body name="module_pen_quill" pos="{_v(0, 0, PEN_LINE_DZ)}">'
    f'<joint name="{PEN_QUILL_JOINT}" type="slide" axis="1 0 0" '
    f'range="{_v(0, PEN_QUILL_TRAVEL)}" stiffness="{_f(PEN_QUILL_STIFFNESS)}" '
    f'damping="2" armature="1e-6"/>'
    f'<geom name="{PEN_SHAFT}" type="capsule" size="{_f(PEN_TIP_R)}" '
    f'fromto="{_v(-0.008, 0, 0, -(0.008 + PEN_LEN), 0, 0)}" mass="{_f(PEN_QUILL_KG)}" '
    f'friction="0.25 0.005 0.0001" priority="1" solimp="{GRIP_SOLIMP}" '
    f'rgba="0.90 0.30 0.25 1"/>'
    f'<site name="{PEN_TIP}" pos="{_v(-(0.008 + PEN_LEN), 0, 0)}" size="0.002"/>'
    f'</body></body>')


def tool_actuators_xml() -> str:
  """The tools' own actuators, for the world's `<actuator>`: the pen's
  carriage, a position servo as stiff as a lead screw -- at a hobby
  servo's kp the pen's drag on the board took 8 mm of its error, and the
  rover's figure came out 12 mm off."""
  return (f'<position name="{PEN_ACTUATOR}" joint="{PEN_CARRIAGE_JOINT}" kp="2000" '
          f'kv="80" ctrlrange="{_v(-PEN_TRAVEL, PEN_TRAVEL)}" '
          f'forcerange="{_v(-PEN_FORCE_N, PEN_FORCE_N)}"/>')


def tool_face(name: str) -> str:
  """What a tool shows: the pen's working parts (`pen_face`), and, visual
  only, the LCD's screen facing away from the robot that carries it and the
  claw's pendant and jaws (they collide as a plate and a peg until #407). A
  module faces the robot with its +x, so its business end is at -x."""
  back = -TOOL_HALF_X
  vis = 'contype="0" conaffinity="0" mass="0"'
  if name == "module_lcd":
    return (f'<geom name="module_lcd_screen" type="box" size="{_v(*SCREEN_HALF)}" '
            f'pos="{_v(back - SCREEN_HALF[0], 0, 0)}" {vis} rgba="0.05 0.08 0.10 1"/>')
  if name == "module_pen":
    return pen_face()
  if name == "module_claw":
    return (f'<geom name="module_claw_pendant" type="box" size="0.008 0.012 0.060" '
            f'pos="{_v(back - 0.010, 0, -0.080)}" {vis} rgba="0.30 0.32 0.36 1"/>'
            + "".join(f'<geom name="module_claw_jaw_{lbl}" type="box" size="0.006 0.004 0.020" '
                      f'pos="{_v(back - 0.010, s * 0.022, -0.150)}" {vis} '
                      f'rgba="0.25 0.25 0.28 1"/>' for s, lbl in ((1, "l"), (-1, "r"))))
  return ""


def tools_xml(pos=(0.0, 0.0), yaw: float = 0.0, spec: RackSpec = DEFAULT) -> tuple[str, str]:
  """The served rack's tools hung on their bays -- (the defaults they use,
  the bodies) -- each compiled hanging, so a world's `qpos0` is every tool
  on its bay (what `HubLifecycle._return_module` puts a lost one back to).
  A tool faces the robot at work: its yaw is the rack's. ⚠ EXACTLY AT
  REST: compiled 0.3 mm up (the spike's settle) a tool read not hung until
  the world's first steps, and a first rack view would have said so."""
  defaults, bodies = [], []
  for name, bay in TOOL_BAYS.items():
    defaults.append(tool_default(name))
    parts = PEN_PARTS_KG if name == "module_pen" else 0.0
    bodies.append(tool_xml(name, bay_peg(spec, bay, pos=pos, yaw=yaw), yaw=yaw,
                           mass=TOOL_KG[name] - parts, face=tool_face(name)))
  return "".join(defaults), "".join(bodies)


def bay_peg(spec: RackSpec, bay: int, pos=(0.0, 0.0), yaw: float = 0.0):
  """Where a tool's peg rests on bay `bay`, in the world: (x, y, z)."""
  c, s = math.cos(yaw), math.sin(yaw)
  y = spec.bays[bay]
  return (pos[0] - s * y, pos[1] + c * y, spec.peg_z)


# ---- the criterion ----------------------------------------------------------------

#: Which of the arm's fork geoms are which pole (`legs.arm.fork_xml`).
FORK_POLES = {"l": ("arm_vl_a", "arm_vl_b"), "r": ("arm_vr_a", "arm_vr_b")}


def pole_ids(model, name: str, prefix: str = "") -> tuple | None:
  """The coupling's conductors as geom ids, `((peg, (v_a, v_b)), ...)` left
  then right, or None where the tool or the fork has none."""
  poles = []
  for side, plates in FORK_POLES.items():
    peg = geom_id(model, f"{name}_peg_{side}")
    ids = tuple(geom_id(model, prefix + g) for g in plates)
    if peg is None or None in ids:
      return None
    poles.append((peg, ids))
  return tuple(poles)


def tool_power(model, data, name: str, prefix: str = "") -> dict:
  """Is this tool's coupling conducting on the arm's fork, and which pole
  is open if not: the rover's criterion (`module_power_state`), each
  conductor on a V of its own side."""
  poles = pole_ids(model, name, prefix)
  left, right = ((False, False) if poles is None
                 else (touching(data, *poles[0]), touching(data, *poles[1])))
  return {"left": left, "right": right, "powered": left and right}


def on_bay(model, data, name: str, spec: RackSpec, bay: int) -> bool:
  """Is the tool HUNG on bay `bay`: its peg down in both trays' V's -- on
  both flanks of each -- and the tool plumb? A peg on one flank, or a plate
  resting on a tray's corner, is a tool jammed on the rack, not hung."""
  pegs = [geom_id(model, f"{name}_peg_{s}") for s in ("l", "r")]
  pegs = [p for p in pegs if p is not None]
  for lbl in ("l", "r"):
    for ab in ("a", "b"):
      flank = geom_id(model, f"{bay_prefix(bay)}tray_{lbl}_{ab}")
      if not any(touching(data, p, [flank]) for p in pegs):
        return False
  z = data.xmat[model.body(name).id].reshape(3, 3)[2, 2]
  return z > math.cos(math.radians(HUNG_TILT_DEG))


#: A hung tool hangs plumb: within this of it, deg.
HUNG_TILT_DEG = 2.0


# ---- the rack's pose, off its tags ------------------------------------------------

#: A fit needs its tags to span this much of the rack (the dock's rule).
MIN_BASELINE_M = 0.1
MAX_FIT_RMS_M = 0.03


@dataclass(frozen=True)
class RackFix:
  x: float
  y: float
  yaw: float
  n: int
  rms: float


def fit_rack(seen: dict[int, tuple[float, float]],
             spec: RackSpec = DEFAULT) -> RackFix | None:
  """The rack's frame in the observer's horizontal frame, fitted to its
  decoded tags (`legs.dock.fit_dock`'s Kabsch; None without a baseline or
  past MAX_FIT_RMS_M)."""
  layout = tag_layout(spec)
  ids = [i for i in seen if i in layout]
  if len(ids) < 2:
    return None
  p = np.array([layout[i][:2] for i in ids], dtype=float)
  q = np.array([seen[i] for i in ids], dtype=float)
  if np.linalg.norm(p - p[0], axis=1).max() < MIN_BASELINE_M:
    return None
  pm, qm = p.mean(axis=0), q.mean(axis=0)
  h = (p - pm).T @ (q - qm)
  yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
  c, s = math.cos(yaw), math.sin(yaw)
  t = qm - np.array([[c, -s], [s, c]]) @ pm
  rms = float(np.sqrt(np.mean(np.sum((q - (p @ np.array([[c, s], [-s, c]]) + t)) ** 2, axis=1))))
  if rms > MAX_FIT_RMS_M:
    return None
  return RackFix(float(t[0]), float(t[1]), yaw, len(ids), rms)


# ---- walking in to a bay ------------------------------------------------------------

#: Where the robot works a bay from: its torso's centre this far out from
#: the bay's peg along the rack's normal, facing it -- the arm's comfortable
#: mid-reach with the plate level (`legs.arm`, the spike's `--reach`).
WORK_X = 0.45
#: The walk in, on the dock's rules (`legs.dock`): the policy does not walk
#: below ~0.2 m/s, so it walks through at APPROACH_V and cuts the command
#: STOP_M short (the flat policy's coast from 0.3 m/s, 22 +- 5.5 mm).
APPROACH_V = 0.3
STOP_M = 0.022
LOOKAHEAD_M = 0.25
K_YAW, YAW_MAX = 2.0, 0.5
K_VY, VY_MAX = 1.0, 0.15
TURN_FIRST = math.radians(25.0)
TURN_W = 0.4


#: Stopped, how far off the bay the robot may believe it is and still reach
#: for the tool: inside the coupling's measured capture (+-20 mm across,
#: +-6 deg; the spike's `--capture`), with room for what a look misses.
LINEUP_ACROSS, LINEUP_YAW = 0.015, math.radians(4.0)
#: ⚠ A STOPPED ROBOT KEEPS TURNING: the flat policy holds no heading (its
#: yaw-rate dead band never corrects a slow turn), and after the walk-in's
#: stop it turned 1.4 deg in 0.5 s and 2.7 deg by 6 s -- 7 mm a second at
#: the fork, so a pick that started inside the capture ended outside it (3
#: of 21 walk-ins). The robot stands this long, then measures the bay again.
SETTLE_AFTER_WALK_S = 3.0


def work_pose(spec: RackSpec, bay: int) -> tuple[float, float, float]:
  """The pose a robot works bay `bay` from, in the RACK frame: (x, y, yaw)."""
  return (WORK_X, spec.bays[bay], math.pi)


#: ⚠ ...AND IT TURNS ONE WAY. On the served body the settle's drift was
#: always counter-clockwise, +1.2 to +2.7 deg over the 3 s while the torso
#: stayed within 4 mm (6 walk-ins at bay A, #405): at the peg that is 5-20
#: mm across, and a walk-in stopping square failed the 15 mm gate on 3 of
#: 6 and every try from the kitchen. The walk-in stops turned this far
#: clockwise, so the settle brings it square.
SETTLE_DRIFT = math.radians(1.9)


def walk_in_twist(ex: float, ey: float, eth: float, heading: float = 0.0):
  """The command for one step of the walk to a bay's working pose, from
  where the robot believes it stands in the WORK frame (the working pose's:
  origin there, +x toward the rack). Zero once within STOP_M of it.
  `heading` is where it should stop facing, in that frame (the served
  body's `-SETTLE_DRIFT`)."""
  from pluggybot.legs.policy import Twist
  if ex >= -STOP_M:
    return Twist()
  aim = math.atan2(-ey, LOOKAHEAD_M) + heading
  e = math.atan2(math.sin(aim - eth), math.cos(aim - eth))
  if abs(e) > TURN_FIRST:
    return Twist(yaw_rate=math.copysign(TURN_W, e))
  vy = max(-VY_MAX, min(VY_MAX, -K_VY * ey))
  return Twist(vx=APPROACH_V, vy=vy, yaw_rate=max(-YAW_MAX, min(YAW_MAX, K_YAW * e)))


# ---- measuring a bay for the arm ----------------------------------------------------

def seen_in_torso(model, data, dets: dict, camera: str, root: int) -> dict:
  """Decoded tags as (x, y, z) in the TORSO frame, not levelled: the arm
  works in it. A detection's "t" is the camera's OpenCV frame (x right, y
  down, z forward); the camera's mount is the robot's own kinematics."""
  cam = model.camera(camera).id
  rot = data.xmat[root].reshape(3, 3)
  p_mount = rot.T @ (data.cam_xpos[cam] - data.xpos[root])
  r_mount = rot.T @ data.cam_xmat[cam].reshape(3, 3)
  return {tag_id: r_mount @ np.array([det["t"][0], -det["t"][1], -det["t"][2]]) + p_mount
          for tag_id, det in dets.items()}


@dataclass(frozen=True)
class BayAim:
  """A bay as the arm needs it, in the torso frame: where its peg crosses
  the arm's plane (x, z), how far the peg's middle is across it, and the
  bay's yaw off square-on."""

  x: float
  z: float
  across: float
  yaw: float

  @property
  def lined_up(self) -> bool:
    return abs(self.across) <= LINEUP_ACROSS and abs(self.yaw) <= LINEUP_YAW


def bay_aim(seen: dict, spec: RackSpec, bay: int, arm_y: float = 0.0) -> BayAim | None:
  """The bay's peg off the rack's decoded tags (`seen_in_torso`): the rack's
  facing from its tags' baseline (`fit_rack`), its height from theirs."""
  fix = fit_rack({i: (p[0], p[1]) for i, p in seen.items()}, spec)
  if fix is None:
    return None
  layout = tag_layout(spec)
  z = float(np.mean([p[2] - layout[i][2] for i, p in seen.items() if i in layout]))
  c, s = math.cos(fix.yaw), math.sin(fix.yaw)
  by = spec.bays[bay]
  px, py = fix.x - s * by, fix.y + c * by
  # Along the peg's axis (the rack's y) to the arm's plane.
  t = (arm_y - py) / c if abs(c) > 1e-9 else 0.0
  # The rack's +x faces the robot: square-on, its yaw is pi.
  yaw = math.atan2(math.sin(fix.yaw - math.pi), math.cos(fix.yaw - math.pi))
  return BayAim(px - s * t, z + spec.peg_z, py, yaw)
