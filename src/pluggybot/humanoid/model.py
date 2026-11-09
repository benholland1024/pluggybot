"""The humanoid's MJCF, generated from a `HumanoidSpec` (issue #507).

Our own body, drawn from a reference: Booster Robotics' T2 as its maker
publishes it (`BoosterRobotics/booster_assets`,
`robots/T2/T2_31dof_JZgripper/T2_31dof_JZgripper.xml`, BSD-3-Clause,
Copyright (c) 2025 BoosterRobotics; `models/booster_assets.LICENSE`), a
1.40 m, 42 kg body its maker walks. What comes from it: the proportions
(`T2`, below), the legs' joint order, the legs' ranges and the arm's to
the elbow, and its masses as a check on ours (`scripts/humanoid_spike.py
--mass`).
What does not: the geometry (our own primitives, never its meshes), the
actuators (`humanoid.parts`, each on sale with a datasheet), and three of
its joint groups. Every joint here has its reason, because a joint costs
money, training and serving speed (#507, rule 4):

  legs, six a side   hip pitch, roll, yaw, knee, ankle pitch, ankle roll:
                     walking and turning on the spot need the hip's three,
                     stairs and getting up the knee's and the ankle pitch's
                     range, and the ankle roll keeps a foot flat on a tread
                     while the body sways over it (the H1's five-joint leg
                     walks with its feet tilting under a wide stance)
  arms, six a side   shoulder pitch, roll, yaw, elbow, wrist pitch, wrist
                     roll: the fewest that place a gripper anywhere in reach
                     AND turn it to the handle -- position and orientation
                     apart, which a block stacked square on another needs;
                     the roll last, so the jaws turn about their own axis
  a gripper an arm   a parallel gripper on sale (rule 2), its jaws one
                     motor's
  the waist's yaw    at the pelvis's top (#560, stage (c3)): each hand
                     reaches past the middle of a table and every pick-up
                     is made standing tall (SimNotes, "A waist, measured
                     before it is built", and "The waist built")
  not here           the T2's waist's pitch and roll: the hips bend the
                     body to a low reach (Ben, 2026-11-05: a pitch only if
                     the stance still turns as an arm goes out);
                     its head's two joints: the depth camera is fixed,
                     pitched at the counter and the floor ahead, and a
                     shelf above the head is out of its view; its seventh
                     arm joint (a wrist's third axis): redundancy, not reach

Zero joint angles are the robot standing straight, arms hanging at its
sides. The axes are the reference's: pitch about y, roll about x, yaw about
z, so the legs' ranges carry over as published; a knee flexes positive, a
hip pitch flexes NEGATIVE (the thigh forward), an ankle pitch positive
points the toes down. The arms hang where the reference's are held out
level, so their ranges are its own moved to that zero; the elbow and the
wrist pitch turn about -y, so that flexing is positive.

Masses: an actuator's is its datasheet's, a geom of its own size on the
link that carries its housing; a FABRICATED part's is its drawing's volume
times its material's density (`structure`), plus `ALLOWANCE` for fasteners
and wiring; inertia is MuJoCo's, from each geom's shape. The bought parts
are `humanoid.parts`; the bill is Parts.md's, "The humanoid body".
"""

from dataclasses import dataclass, field, replace
import math

from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.actuator import JointLimits, Motor, driver_gains
from pluggybot.telemetry.protocol import ROBOT_ROOT

SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
       "wrist_pitch", "wrist_roll")
#: The waist's joints, the trunk's own (no side): its yaw, at the pelvis's
#: top (#560, stage (c3)).
WAIST = ("waist_yaw",)
#: Every actuated joint but the grippers', in the order qpos carries them:
#: the left leg, the right leg, the waist, then each arm (its gripper's jaws
#: follow its wrist in qpos; read them by name). A body drawn without the
#: waist (`HumanoidSpec.waist_yaw`, the instruments' today's body) has
#: `joint_names(spec)`.
JOINT_NAMES = (tuple(f"{s}_{j}" for s in SIDES for j in LEG) + WAIST
               + tuple(f"{s}_{j}" for s in SIDES for j in ARM))
#: A gripper's two jaws: one slide each, the second held to the first.
JAWS = ("jaw_a", "jaw_b")
#: The body each kind of joint turns, a side's name before it (the waist's,
#: no side's: `body_of`).
BODY_OF = {"hip_pitch": "hip_pitch_link", "hip_roll": "hip_roll_link",
           "hip_yaw": "thigh", "knee": "shank", "ankle_pitch": "ankle_pitch_link",
           "ankle_roll": "foot", "shoulder_pitch": "shoulder_pitch_link",
           "shoulder_roll": "shoulder_roll_link", "shoulder_yaw": "upper_arm",
           "elbow": "forearm", "wrist_pitch": "wrist", "wrist_roll": "hand",
           "waist_yaw": "waist_yaw_link"}


def kind(joint: str) -> str:
  """`left_knee` -> `knee`: which part drives a joint; a waist joint is its
  own kind."""
  return joint if joint in WAIST else joint.split("_", 1)[1]


def side(joint: str) -> int:
  """+1 for the robot's left, -1 for its right, 0 for the waist's."""
  if joint in WAIST:
    return 0
  return 1 if joint.startswith("left") else -1


def body_of(joint: str) -> str:
  """The body a joint turns: `left_knee` -> `left_shank`."""
  if joint in WAIST:
    return BODY_OF[joint]
  return f"{joint.split('_', 1)[0]}_{BODY_OF[kind(joint)]}"


def joint_names(spec: "HumanoidSpec") -> tuple[str, ...]:
  """A body's actuated joints, `JOINT_NAMES`' order: without the waist where
  the spec has none."""
  return tuple(j for j in JOINT_NAMES if spec.waist_yaw or j not in WAIST)


# ---- the reference -----------------------------------------------------------

@dataclass(frozen=True)
class Reference:
  """The T2 as its published MJCF has it, m and kg, read 2026-10-16 at
  booster_assets commit 3c2dfa9 (`scripts/humanoid_spike.py --mass` sets
  our segments beside its). Each length is between two joint axes with the
  legs straight, the T2's own canted hip axis (15 deg) projected flat."""
  #: Hip pitch axis -> knee, knee -> ankle pitch, ankle -> sole.
  thigh: float = 0.346
  shank: float = 0.335
  ankle_to_sole: float = 0.0405
  #: The legs' lines either side of the middle.
  hip_y: float = 0.100
  #: The hip pitch axes -> the shoulder's pitch axis, and its width.
  shoulder_z: float = 0.421
  shoulder_y: float = 0.163
  #: Shoulder roll axis -> elbow axis -> wrist flexion axis.
  upper_arm: float = 0.217
  forearm: float = 0.212
  #: The sole: length and width.
  foot: tuple[float, float] = (0.216, 0.082)
  #: Masses by segment: a leg from its hip pitch link down, an arm from
  #: its shoulder pitch link with its gripper, the head, and the rest (the
  #: trunk with the waist's three joints).
  leg_kg: float = 8.267
  arm_kg: float = 4.515
  head_kg: float = 1.563
  rest_kg: float = 14.932
  total_kg: float = 42.006


T2 = Reference()


# ---- fabricated parts ---------------------------------------------------------

#: Material densities, kg/m^3: 6082-T6 aluminium plate (2.70 g/cm^3, the
#: alloy's datasheet), PETG printed solid (1.27: a 2 mm shell is all wall),
#: the sole's rubber (NR/SBR sheet, 1.2-1.5 g/cm^3; the lower), and the
#: head's padded rim's TPU (1.23, its filament's sheet: `catalog`'s
#: `tpu_1kg`).
ALUMINIUM = 2700.0
PETG = 1270.0
RUBBER = 1200.0
TPU = 1230.0
#: Fasteners and wiring on top of a fabricated part's own mass: a stated
#: allowance, 15 % (the quadruped's frame budget carried its fasteners in
#: its estimate; a drawing replaces this with a count).
ALLOWANCE = 0.15


@dataclass(frozen=True)
class Plate:
  """A flat part cut from sheet: `area` m^2 of outline at `thick` m."""
  area: float
  thick: float
  density: float = ALUMINIUM

  @property
  def kg(self) -> float:
    return self.area * self.thick * self.density * (1 + ALLOWANCE)


@dataclass(frozen=True)
class Tube:
  """A round tube, `od` x `wall` m, `length` m long."""
  od: float
  wall: float
  length: float
  density: float = ALUMINIUM

  @property
  def kg(self) -> float:
    r_o, r_i = self.od / 2, self.od / 2 - self.wall
    return math.pi * (r_o ** 2 - r_i ** 2) * self.length * self.density * (1 + ALLOWANCE)


def _bracket(motor: Motor, thick: float = 0.005) -> Plate:
  """The two side plates a motor's housing bolts between and its link
  carries on: each its diameter square, 5 mm 6082 (the quadruped's hip
  brackets' stock)."""
  return Plate(area=2 * motor.diameter ** 2, thick=thick)


@dataclass(frozen=True)
class HumanoidSpec:
  """A candidate body. Lengths in metres, masses in kg; the parts that
  drive it are `drives` (joint kind -> motor)."""

  name: str = "humanoid"
  drives: dict = field(default_factory=lambda: dict(hp.DRIVES))
  # -- the legs (`T2`'s proportions) --
  #: The hip pitch joints' origins either side, and the legs' lines.
  hip_pitch_y: float = 0.08
  hip_y: float = 0.10
  #: Hip pitch axis -> roll axis -> the thigh's top (the yaw motor) -> knee.
  hip_roll_drop: float = 0.067
  hip_yaw_drop: float = 0.069
  thigh: float = 0.21
  shank: float = 0.335
  #: Ankle pitch axis -> roll axis -> the sole. The roll motor sits ON the
  #: roll axis (direct drive), so the axis is its radius and the sole plate
  #: above the floor: 25 mm higher than the T2's, whose ankle has no motor.
  ankle_roll_drop: float = 0.01
  sole_drop: float = 0.055
  #: The sole: length, width, and how far its heel is behind the ankle.
  foot_len: float = 0.22
  foot_w: float = 0.09
  heel: float = 0.07
  # -- the trunk: the pelvis, and over the waist the torso and the head --
  #: The pelvis, round the hip pitch motors, and the torso above it:
  #: half-extents and the torso's centre over the hip axes.
  pelvis_half: tuple[float, float, float] = (0.07, 0.115, 0.06)
  torso_half: tuple[float, float, float] = (0.085, 0.115, 0.18)
  torso_z: float = 0.24
  #: The waist's yaw (#560, stage (c3); Ben, 2026-11-05: as #620 measured it,
  #: `scripts/waist_spike.py`'s `yaw`): its axis up the pelvis's middle, its
  #: link at the pelvis's top, `waist_z` over the hip axes, its motor stood
  #: in the pelvis's middle between the hips' pitch motors (they leave |y| <
  #: 0.054 free, an RS03's 98 mm fits); the torso, the pack, the head and its
  #: sensors, the port and the arms ride it, the IMU and the legs the
  #: pelvis. False draws the body before it: one rigid trunk, the drawing
  #: #620's and #621's instruments measure from.
  waist_yaw: bool = True
  waist_z: float = 0.06
  # -- the arms --
  #: The shoulder pitch joint (on the torso's side) over the hip axes, and
  #: the roll axis outboard of it: the arm hangs on the roll's line.
  shoulder_z: float = 0.421
  shoulder_pitch_y: float = 0.13
  shoulder_roll_out: float = 0.05
  #: Shoulder roll axis -> the upper arm's top (its yaw motor) -> elbow ->
  #: wrist pitch -> wrist roll.
  shoulder_yaw_drop: float = 0.05
  upper_arm: float = 0.17
  forearm: float = 0.21
  wrist: float = 0.03
  # -- the head: fixed on the torso, the sensors in it (`parts`), built
  # round the LIDAR (`_head`) --
  #: The head's radius, m: the face's below the LIDAR's window and the
  #: cap's over it, one cylinder (#581, stage 3: a crown narrower than the
  #: face read as a puck on top of the head).
  head_r: float = 0.068
  #: The face's foot over the hip axes, m; its top is where the field's
  #: steepest ray passes its edge (`face_top`).
  face_lo: float = 0.475
  #: The depth camera's height over the hip axes, on the face's front,
  #: where it rode before the head was built round the LIDAR; and the
  #: navigation camera's over it, brought 3.5 cm down into the face with
  #: the LIDAR (#581, stage 3: on the dome, its front held the rays at 0.711).
  depth_cam_z: float = 0.535
  nav_cam_z: float = 0.565
  #: The depth camera's pitch below level, deg: a counter's height (0.94 m)
  #: from 0.12 m ahead and the floor from 0.48 m (`humanoid_spike.py
  #: --sensors`).
  depth_pitch_deg: float = 40.0
  #: The navigation camera's (the tags on the walls, 0.4-1.2 m up, read
  #: from a metre on).
  nav_pitch_deg: float = 20.0
  #: The LIDAR's rays' origin over the hip axes, m (#581, stage 3: the unit
  #: upside down in the head, `_lidar`): 4 mm over the lowest from which its
  #: steepest ray, -52 deg, clears the navigation camera's front, 9 over the
  #: depth camera's corners (at 0.67 the -52 deg ray met the navigation
  #: camera ahead). The shoulders' motors hide that ray and the -48 deg one
  #: aside, 0.5 % of a frame more than at stage 1's 0.735 (`lidar_spike.py
  #: --hidden --heights`).
  lidar_z: float = 0.68
  # -- the charge port (#559 stage 2): on the torso's front --
  #: Right of its middle and over the hip axes, m: where the right hand
  #: carries a plug straight in from ahead with its arm clear of the trunk
  #: (an IK sweep of the robot's model of itself: 26-57 mm clear along
  #: 17-26 cm of carry, `activity.station`'s seat), 0.90 m up standing.
  #: ⚠ Where the head's cameras see it there is none: they ride the trunk,
  #: and no point of its front or sides comes within 21 deg of their view.
  port_y: float = -0.075
  port_z: float = 0.168

  @property
  def leg_length(self) -> float:
    """Hip pitch axis to the sole, legs straight."""
    return (self.hip_roll_drop + self.hip_yaw_drop + self.thigh + self.shank
            + self.ankle_roll_drop + self.sole_drop)

  def motor(self, joint: str) -> Motor:
    return self.drives[kind(joint)]

  def with_(self, **kw) -> "HumanoidSpec":
    return replace(self, **kw)


CHOSEN = HumanoidSpec()
MODEL_XML = "models/humanoid.xml"
#: The arms' commands carry their joints' policy gains (`driver_gains`, a
#: 10 Hz loop at damping ratio 2) times these, stiffness and damping: at the
#: policies' own, an arm held out sagged 12 cm at the hand. Inside every
#: part's MIT range (the RS00's Kp to 500). The arms' weight rides each
#: command too (`humanoid.arms`), and training's scripted arms hold the same
#: (`training/humanoid_train`).
ARM_GAIN_SCALE = (4.0, 2.0)
#: ...and the waist's commands theirs (`humanoid.arms.Waist`), sized to what
#: it turns: 0.46 kg*m^2 about its axis with the arms hanging, 1.14 with both
#: out level, where the arms' scale left it ringing at 4.2 and 2.7 Hz at a
#: damping ratio of 0.42 and 0.27 -- here 5.9 and 3.7 Hz at 1.48 and 0.94
#: (kp 632 N*m/rad, kd 50 N*m*s/rad; the RS03's MIT range, its manual's
#: 4.1.2: Kp 0-5000, Kd 0-100). Training's scripted waist holds the same.
WAIST_GAIN_SCALE = (8.0, 10.0)
#: The body's constants for `training/`, which cannot import this package.
CONSTANTS_JSON = "models/humanoid.json"


def structure(spec: HumanoidSpec = CHOSEN) -> dict[str, tuple]:
  """Every fabricated part the body carries but the motors' brackets
  (`_bracket`), one entry a part: (its stock and size, as a `Plate` or a
  `Tube`, so its mass is its volume's; how many the body carries; what it
  is). The generator lays them and the bill buys them."""
  px, py, pz = spec.pelvis_half
  tx, ty, tz = spec.torso_half
  r, top, cap_lo = spec.head_r, face_top(spec), _rim_lo(spec) + RIM[1]
  foot = spec.foot_len * spec.foot_w
  waist = {"waist_plate": (Plate(2 * px * 2 * py, 0.005), 1,
                           "the waist's turntable plate over the pelvis, the torso's "
                           "frame on it, 5 mm 6082")} if spec.waist_yaw else {}
  return {
    **waist,
    "pelvis_side": (Plate(2 * px * 2 * pz, 0.003), 2, "the pelvis's sides, 3 mm 6082"),
    "pelvis_deck": (Plate(2 * px * 2 * py, 0.003), 2, "the pelvis's top and bottom"),
    "torso_side": (Plate(2 * tx * 2 * tz, 0.003), 2, "the torso's sides, 3 mm 6082"),
    "torso_back": (Plate(2 * ty * 2 * tz, 0.003), 1, "the torso's back"),
    "torso_top": (Plate(2 * tx * 2 * ty, 0.003), 1,
                  "the torso's top, the shoulders' and the neck's mount"),
    "torso_cover": (Plate(2 * ty * 2 * tz, 0.002, PETG), 1,
                    "the torso's front cover, printed in PETG"),
    "head_shell": (Plate(2 * math.pi * r * (top - spec.face_lo) + math.pi * r ** 2, 0.002,
                         PETG), 1, "the face, the head's shell under its window, printed in PETG"),
    "head_visor": (Plate(2 * VISOR[2] * (_visor_top(spec) - VISOR[3]), 0.002, PETG), 1,
                   "the visor over the face's front, the cameras in it, printed in PETG"),
    "head_dome": (Plate(2 * math.pi * DOME_R * (_dome_z(spec) + DOME_R - top), 0.002, PETG), 1,
                  "the dome over the face, under the window, printed in PETG"),
    "head_back": (Plate(math.radians(2 * BACK[0]) * r * (_back_top(spec) - top), BACK[2], PETG),
                  1, "the back's panels, round its back half up to the window, printed in PETG"),
    "head_cap": (Plate(2 * math.pi * r * (spec.lidar_z + CAP_TOP - cap_lo) + math.pi * r ** 2,
                       CAP_WALL, PETG), 1, "the cap over the LIDAR's base, printed in PETG"),
    "head_rim": (Plate(2 * math.pi * (r + RIM[0] / 2) * RIM[1], RIM[0], TPU), 1,
                 "the cap's padded rim, printed in TPU"),
    "thigh_plate": (Plate(0.06 * spec.thigh, 0.005), 4,
                    "a thigh's side plates, 60 mm wide, 5 mm 6082"),
    "shank_plate": (Plate(0.05 * spec.shank, 0.004), 4,
                    "a shank's side plates, 50 mm wide, 4 mm 6082"),
    "ankle_rod": (Tube(0.008, 0.0015, spec.shank), 2,
                  "the ankle's parallelogram rod, 8 x 1.5 mm tube"),
    "sole_plate": (Plate(foot, 0.006), 2, "a sole, 6 mm 6082"),
    "sole_rubber": (Plate(foot, 0.005, RUBBER), 2, "a sole's rubber, 5 mm sheet"),
    "upper_arm_tube": (Tube(0.040, 0.002, spec.upper_arm), 2, "an upper arm, 40 x 2 mm 6082 tube"),
    "forearm_tube": (Tube(0.035, 0.002, spec.forearm), 2, "a forearm, 35 x 2 mm 6082 tube"),
    "gripper_adapter": (Plate(hp.GRIPPER.flange ** 2, 0.006), 2,
                        "the plate a gripper bolts to on the wrist's roll"),
    "lidar_fin": (Plate((r - FIN[1]) * (spec.lidar_z + hp.LIDAR.origin - top), FIN[0]), 1,
                  "the fin up the head's back that carries the cap, 6 mm 6082"),
    "lidar_plate": (Plate(hp.LIDAR.size[1] * (hp.LIDAR.size[0] / 2 + r), PLATE_T), 1,
                    "the plate the LIDAR's base bolts under, on the fin, 3 mm 6082"),
  }


# The head built round the LIDAR (#581, stage 3; `_head`; SimNotes, "The
# sensor in the head"): the unit upside down, its window open all round --
# its manual: nothing in its field, "including glass" -- between the face
# and the cap, one cylinder (`HumanoidSpec.head_r`); the cap carried by a
# fin behind, edge-on to the sensor, and the back half closed under the
# rings the band is laid from. Every part keeps `FIELD_CLEAR` out of the
# field but the fin and the back's panels, which hide what they are drawn to.
#: The LIDAR's field upside down, deg (`perception.lidar.FOV_DEG`: its
#: sheet's -7 to +52 turned over), and how far the head keeps out of it, m.
FIELD_DEG = (-52.0, 7.0)
FIELD_CLEAR = 0.002
#: The colour imager's vertical field, deg (the D435's, at 720p): what
#: `spot`'s and `watch`'s texts count their picture off (`steps.SPOT_TOP_DEG`
#: and the rest; a test reads them against the model, #660).
COLOR_FOVY_DEG = 42
#: The D435's two imagers on its body, m: ahead of its middle along the
#: camera's own axis, and the colour imager's across it.
CAM_AHEAD_M, COLOR_ACROSS_M = 0.013, -0.015
#: Where the cameras ride, m ahead of the hip axes: the depth camera's
#: middle and the navigation camera's, as on stage 1's face.
DEPTH_CAM_X, NAV_CAM_X = 0.0775, 0.069
#: The navigation camera's housing, half-extents, m (a Camera Module 3).
NAV_CAM_HALF = (0.006, 0.012, 0.012)
#: The visor, the plate over the face's front the cameras sit in, m: its
#: back and front, half its width and its foot over the hip axes. Its top
#: is under the steepest ray at its corners (`_visor_top`).
VISOR = (0.045, 0.070, 0.050, 0.487)
#: The dome over the face, the slope under the window, m: a sphere inside
#: the steepest ray's cone (the site draws no ellipsoid), rising from the
#: face's top 6 mm in from its edge.
DOME_R = 0.066
#: The cap's padded rim, TPU: how far it stands proud of the head, the
#: first of it a fall meets, and how tall it is, m.
RIM = (0.004, 0.012)
#: The cap's top over the rays' origin, and its wall, m: 18 mm over the
#: unit's base, its plate and the cap's wall between them, so that no plane
#: meets the unit before the head does (8 mm, 0.01 m lower).
CAP_TOP, CAP_WALL = 0.065, 0.003
#: The fin that carries the cap, 6082, m: its thickness (its shadow, 4 deg
#: each side of straight back) and its inner face behind the sensor, the
#: manual's 10 mm of air behind the unit; its outer face is the head's.
FIN = (0.006, 0.0425)
#: The back's panels, PETG, round its back half: how far round from
#: straight back each way, deg; how many; their thickness, m; and the ray
#: their top edge lies on at their middle, deg -- the ring above it, -24
#: deg, passes 1 mm over their corners, and the one under it, -28, meets
#: them. The band's walls and the fences come off its rings from -8 to -24
#: deg; the steeper ones behind see the floor within 2.6 m. Over the
#: house, #572's probe pins every outdoor place with the panels as without
#: them (closed to every ring over +-55 deg, the loop's places went 3 %
#: unpinned and the rooms' 7: `lidar_spike.py --probe`).
BACK = (90.0, 7, 0.004, -26.0)
#: The plate under the cap that the unit's base bolts to, m.
PLATE_T = 0.003
#: The visor's, the dome's and the back's inner face's colour, the PETG's
#: own black, unpainted (the head's shell is painted the robot's livery);
#: the rim's black TPU.
VISOR_RGBA, RIM_RGBA = "0.09 0.1 0.12 1", "0.06 0.06 0.07 1"


def _tan(deg: float) -> float:
  return math.tan(math.radians(deg))


def _under(spec: HumanoidSpec, r: float) -> float:
  """The highest a part `r` m off the LIDAR's axis may reach under its
  window, over the hip axes, m: `FIELD_CLEAR` under the steepest ray."""
  return spec.lidar_z - _tan(-FIELD_DEG[0]) * r - FIELD_CLEAR


def face_top(spec: HumanoidSpec = CHOSEN) -> float:
  """The face's top over the hip axes, m: its edge under the steepest ray."""
  return _under(spec, spec.head_r)


def _visor_top(spec: HumanoidSpec) -> float:
  return _under(spec, math.hypot(VISOR[1], VISOR[2]))


def _dome_z(spec: HumanoidSpec) -> float:
  """The dome's middle over the hip axes, m: deep enough that the sphere
  stays `FIELD_CLEAR` inside the steepest ray's cone."""
  return spec.lidar_z - (DOME_R + FIELD_CLEAR) / math.cos(math.radians(-FIELD_DEG[0]))


def _rim_lo(spec: HumanoidSpec) -> float:
  """The rim's underside over the hip axes, m: over the unit's window and
  `FIELD_CLEAR` over the field's highest ray at its edge."""
  out = _tan(FIELD_DEG[1]) * (spec.head_r + RIM[0]) + FIELD_CLEAR
  return spec.lidar_z + max(hp.LIDAR.window, out)


def _back_panel(spec: HumanoidSpec) -> tuple[float, float]:
  """A back panel's half-width and its top over the hip axes, m: wide
  enough to meet the next one, its top on the ray `BACK[3]` at its middle."""
  half_w = spec.head_r * math.tan(math.radians(BACK[0] / BACK[1])) + 0.0005
  return half_w, spec.lidar_z - _tan(-BACK[3]) * (spec.head_r - BACK[2] / 2)


def _back_top(spec: HumanoidSpec) -> float:
  return _back_panel(spec)[1]


def _kg(spec: HumanoidSpec, *names: str, each: bool = False) -> float:
  """The named fabricated parts' mass, kg: all of each, or one of each."""
  parts = structure(spec)
  return sum(parts[n][0].kg * (1 if each else parts[n][1]) for n in names)


def _f(v: float) -> str:
  return f"{v:.6g}"


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def _exact(*vs: float) -> str:
  """Round-trip precision, for the drivers: their gains are the policy's."""
  return " ".join(repr(float(v)) for v in vs)


#: Joint ranges, rad, and their source: the T2's where the joint is its
#: (the legs, the shoulder's pitch and roll, the upper arm's twist and the
#: elbow, moved to our zero; the wrist's roll is its forearm's twist), ours
#: for the wrist's pitch. Left side; the right mirrors roll and yaw
#: (`joint_range`).
RANGES = {
  "hip_pitch": (-2.932153, 2.234021),
  "hip_roll": (-0.575959, 1.535890),
  "hip_yaw": (-2.757620, 2.757620),
  "knee": (0.0, 2.583087),
  "ankle_pitch": (-0.837758, 0.575959),
  "ankle_roll": (-0.314159, 0.314159),
  "shoulder_pitch": (-3.019420, 1.710423),
  # The T2's +-1.518 about its arm held level; hanging is its -1.518.
  "shoulder_roll": (0.0, 3.036873),
  "shoulder_yaw": (-2.059489, 2.059489),
  "elbow": (0.0, 2.103122),
  # Ours: the T2's 7-joint wrist does not map onto a 6-joint one. A pitch of
  # +-90 deg points the gripper down at a counter with the forearm level.
  "wrist_pitch": (-1.6, 1.6),
  "wrist_roll": (-2.059489, 2.059489),
  # Ours (Ben, 2026-11-05): +-60 deg, all of it used by #620's table
  # reaches (the T2 turns +-138); to the left positive.
  "waist_yaw": (-1.05, 1.05),
}
#: Joints whose range mirrors on the right (a roll or yaw turns the other way).
MIRRORED = ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw",
            "wrist_roll")


def joint_range(joint: str) -> tuple[float, float]:
  lo, hi = RANGES[kind(joint)]
  if side(joint) < 0 and kind(joint) in MIRRORED:
    return -hi, -lo
  return lo, hi


def stand_qpos(spec: HumanoidSpec = CHOSEN, bend: float = 0.2) -> dict[str, float]:
  """The joints standing on flat feet with the knees bent: the hip and the
  ankle each take half the knee's angle, so the shank and the thigh lean
  by the same amount and the sole stays level; the arms hang, a little
  out and the elbows a little bent."""
  q = {}
  for s in SIDES:
    q[f"{s}_hip_pitch"] = -bend
    q[f"{s}_knee"] = 2 * bend
    q[f"{s}_ankle_pitch"] = -bend
    q[f"{s}_shoulder_roll"] = 0.1 if s == "left" else -0.1
    q[f"{s}_elbow"] = 0.3
  return q


def stand_height(spec: HumanoidSpec = CHOSEN, bend: float = 0.2) -> float:
  """The hip axes over the floor in `stand_qpos(bend)`: the hip pitch axis
  to the knee is the thigh's whole length (the roll and yaw offsets lie
  along it), leaning `bend`, the shank leaning as much the other way, and
  below the ankle the foot level."""
  upper = spec.hip_roll_drop + spec.hip_yaw_drop + spec.thigh
  return (upper + spec.shank) * math.cos(bend) + spec.ankle_roll_drop + spec.sole_drop


def color_camera_z(spec: HumanoidSpec = CHOSEN) -> float:
  """The colour imager's height over the floor in `stand_qpos`: its body's
  height, `CAM_AHEAD_M` of its look pitched down. What `spot`'s text counts
  the top of its picture off (`steps.SPOT_TOP_AT_FEET_M`)."""
  return (stand_height(spec) + spec.depth_cam_z
          - CAM_AHEAD_M * math.sin(math.radians(spec.depth_pitch_deg)))


def color_camera_ahead(spec: HumanoidSpec = CHOSEN, bend: float = 0.2) -> float:
  """How far the colour imager stands ahead of the soles' middle, the
  stance frame's origin, in `stand_qpos(bend)`, m: its body's place and
  `CAM_AHEAD_M` of its look pitched down, less where the soles' middle is --
  the ankle off the hips by the thigh's lean less the shank's, the sole's
  middle ahead of the ankle."""
  upper = spec.hip_roll_drop + spec.hip_yaw_drop + spec.thigh
  soles = (upper - spec.shank) * math.sin(bend) + spec.foot_len / 2 - spec.heel
  return (DEPTH_CAM_X + CAM_AHEAD_M * math.cos(math.radians(spec.depth_pitch_deg))
          - soles)


# ---- the MJCF ------------------------------------------------------------------

MOTOR_RGBA = "0.25 0.25 0.28 1"
LIMB_RGBA = "0.2 0.4 0.8 1"          # robot.CHASSIS_RGBA, so `paint` re-liveries it
SOLE_RGBA = "0.1 0.1 0.1 1"
#: The jaws' pads: stiff, hard contact, the quadruped claw's lesson
#: (#407): on MuJoCo's soft default a held cube crept out of the jaws.
PAD_SOLREF = "0.006 1"
PAD_SOLIMP = "0.99 0.999 0.00001"
#: The soles' and the pads' friction, with `priority` so it is theirs and
#: not the floor's MAX (SimNotes, "THE caster lesson"): rubber on a floor,
#: and the pads' rubber on a handle.
SOLE_FRICTION = "0.9 0.02 0.01"
PAD_FRICTION = "1.2 0.01 0.0002"


def _axis_vec(axis: str) -> tuple[float, float, float]:
  return {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]


def _housing(name: str, motor: Motor, at, axis: str) -> str:
  """A motor's housing: a cylinder its own diameter and length about its
  axis, at its datasheet mass, with the bracket it bolts into."""
  ax = _axis_vec(axis)
  h = motor.length / 2
  a = [c - h * d for c, d in zip(at, ax)]
  b = [c + h * d for c, d in zip(at, ax)]
  return (f'<geom name="{name}" class="motor" type="cylinder" '
          f'size="{_f(motor.diameter / 2)}" fromto="{_v(*a, *b)}" '
          f'mass="{_f(motor.mass + _bracket(motor).kg)}"/>')


def _joint(name: str, axis: str, motor: Motor | None, sign: float = 1.0) -> str:
  lo, hi = joint_range(name)
  ax = [sign * c for c in _axis_vec(axis)]
  attrs = ""
  if motor is not None:
    attrs = (f' armature="{_f(motor.armature)}" '
             f'frictionloss="{_f(hp.friction_nm(motor))}"')
  return (f'<joint name="{name}" axis="{_v(*ax)}" range="{_v(lo, hi)}"{attrs}/>')


def _leg(spec: HumanoidSpec, s: str) -> str:
  k = 1 if s == "left" else -1
  m = {j: spec.motor(f"{s}_{j}") for j in LEG}
  out = spec.hip_y - spec.hip_pitch_y
  thigh = 2 * _kg(spec, "thigh_plate", each=True)
  shank = 2 * _kg(spec, "shank_plate", each=True)
  rod = _kg(spec, "ankle_rod", each=True)
  sole = _kg(spec, "sole_plate", "sole_rubber", each=True)
  fx = spec.foot_len / 2 - spec.heel           # the sole's middle, ahead of the ankle
  sole_t = 0.011                                # plate and rubber
  return f"""
      <body name="{s}_hip_pitch_link" pos="{_v(0, k * spec.hip_pitch_y, 0)}">
        {_joint(f"{s}_hip_pitch", "y", m["hip_pitch"])}
        {_housing(f"{s}_hip_roll_motor", m["hip_roll"], (0, k * out, -spec.hip_roll_drop), "x")}
        <body name="{s}_hip_roll_link" pos="{_v(0, k * out, -spec.hip_roll_drop)}">
          {_joint(f"{s}_hip_roll", "x", m["hip_roll"])}
          {_housing(f"{s}_hip_yaw_motor", m["hip_yaw"], (0, 0, -spec.hip_yaw_drop), "z")}
          <body name="{s}_thigh" pos="{_v(0, 0, -spec.hip_yaw_drop)}">
            {_joint(f"{s}_hip_yaw", "z", m["hip_yaw"])}
            <geom name="{s}_thigh" class="limb" fromto="{_v(0, 0, -0.04, 0, 0, -spec.thigh + 0.04)}"
                  size="0.05" mass="{_f(thigh)}"/>
            {_housing(f"{s}_knee_motor", m["knee"], (0, 0, -spec.thigh), "y")}
            <body name="{s}_shank" pos="{_v(0, 0, -spec.thigh)}">
              {_joint(f"{s}_knee", "y", m["knee"])}
              <geom name="{s}_shank" class="limb" fromto="{_v(0, 0, -0.05, 0, 0, -spec.shank + 0.05)}"
                    size="0.04" mass="{_f(shank)}"/>
              {_housing(f"{s}_ankle_pitch_motor", m["ankle_pitch"], (0, 0, -ANKLE_MOTOR_DROP), "y")}
              <geom name="{s}_ankle_rod" class="visual" type="capsule" size="0.004"
                    fromto="{_v(-ANKLE_CRANK, 0, -ANKLE_MOTOR_DROP, -ANKLE_CRANK, 0, -spec.shank)}"
                    mass="{_f(rod)}" rgba="0.6 0.6 0.62 1"/>
              <body name="{s}_ankle_pitch_link" pos="{_v(0, 0, -spec.shank)}">
                {_joint(f"{s}_ankle_pitch", "y", m["ankle_pitch"])}
                {_housing(f"{s}_ankle_roll_motor", m["ankle_roll"], (0, 0, -spec.ankle_roll_drop), "x")}
                <body name="{s}_foot" pos="{_v(0, 0, -spec.ankle_roll_drop)}">
                  {_joint(f"{s}_ankle_roll", "x", m["ankle_roll"])}
                  <geom name="{s}_sole" class="sole" type="box"
                        size="{_v(spec.foot_len / 2, spec.foot_w / 2, sole_t / 2)}"
                        pos="{_v(fx, 0, -spec.sole_drop + sole_t / 2)}"
                        mass="{_f(sole)}"/>
                  <site name="{s}_foot" type="box" group="4"
                        size="{_v(spec.foot_len / 2, spec.foot_w / 2, 0.006)}"
                        pos="{_v(fx, 0, -spec.sole_drop + 0.003)}"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>"""


#: The ankle pitch's motor rides the shank's top, below the knee's, and
#: drives the ankle through a rod: a PARALLELOGRAM (equal cranks, the rod as
#: long as the pivots are apart), so the ankle turns exactly as the motor
#: does, 1:1 at every angle, and its rotor is the joint's armature -- while
#: its 1.5 kg ride 0.24 m nearer the knee than at the ankle. m: the motor's
#: axis below the knee's, and the cranks' length.
ANKLE_MOTOR_DROP = 0.10
ANKLE_CRANK = 0.035


def _arm(spec: HumanoidSpec, s: str) -> str:
  k = 1 if s == "left" else -1
  m = {j: spec.motor(f"{s}_{j}") for j in ARM}
  g = hp.GRIPPER
  upper = _kg(spec, "upper_arm_tube", each=True)
  fore = _kg(spec, "forearm_tube", each=True)
  adapter = _kg(spec, "gripper_adapter", each=True)
  half = g.housing_half
  jaw_z = -2 * half[2] - g.finger_len / 2         # the fingers' middle
  return f"""
      <body name="{s}_shoulder_pitch_link" pos="{_v(0, k * spec.shoulder_pitch_y, spec.shoulder_z)}">
        {_joint(f"{s}_shoulder_pitch", "y", m["shoulder_pitch"])}
        {_housing(f"{s}_shoulder_roll_motor", m["shoulder_roll"], (0, k * spec.shoulder_roll_out, 0), "x")}
        <body name="{s}_shoulder_roll_link" pos="{_v(0, k * spec.shoulder_roll_out, 0)}">
          {_joint(f"{s}_shoulder_roll", "x", m["shoulder_roll"])}
          {_housing(f"{s}_shoulder_yaw_motor", m["shoulder_yaw"], (0, 0, -spec.shoulder_yaw_drop), "z")}
          <body name="{s}_upper_arm" pos="{_v(0, 0, -spec.shoulder_yaw_drop)}">
            {_joint(f"{s}_shoulder_yaw", "z", m["shoulder_yaw"])}
            <geom name="{s}_upper_arm" class="limb" fromto="{_v(0, 0, -0.03, 0, 0, -spec.upper_arm + 0.03)}"
                  size="0.04" mass="{_f(upper)}"/>
            {_housing(f"{s}_elbow_motor", m["elbow"], (0, 0, -spec.upper_arm), "y")}
            <body name="{s}_forearm" pos="{_v(0, 0, -spec.upper_arm)}">
              {_joint(f"{s}_elbow", "y", m["elbow"], sign=-1.0)}
              <geom name="{s}_forearm" class="limb" fromto="{_v(0, 0, -0.03, 0, 0, -spec.forearm + 0.03)}"
                    size="0.035" mass="{_f(fore)}"/>
              {_housing(f"{s}_wrist_pitch_motor", m["wrist_pitch"], (0, 0, -spec.forearm), "y")}
              <body name="{s}_wrist" pos="{_v(0, 0, -spec.forearm)}">
                {_joint(f"{s}_wrist_pitch", "y", m["wrist_pitch"], sign=-1.0)}
                {_housing(f"{s}_wrist_roll_motor", m["wrist_roll"], (0, 0, -spec.wrist), "z")}
                <body name="{s}_hand" pos="{_v(0, 0, -spec.wrist - m["wrist_roll"].length / 2)}">
                  {_joint(f"{s}_wrist_roll", "z", m["wrist_roll"])}
                  <geom name="{s}_gripper" class="limb" type="box" size="{_v(*half)}"
                        pos="{_v(0, 0, -half[2])}" mass="{_f(g.housing_kg + adapter)}"/>
                  <site name="{s}_grip" pos="{_v(0, 0, jaw_z)}" size="0.01" group="4"/>
                  {_jaw(s, JAWS[0], +1, jaw_z)}
                  {_jaw(s, JAWS[1], -1, jaw_z)}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>"""


def _jaw(s: str, name: str, sign: int, z: float) -> str:
  """One finger of the parallel gripper, sliding along the hand's x: its
  joint reads the jaw's travel out from closed, 0 to half the stroke."""
  g = hp.GRIPPER
  pad = g.pad_half
  return f"""<body name="{s}_{name}" pos="{_v(sign * pad[0], 0, z)}">
                    <joint name="{s}_{name}" type="slide" axis="{_v(sign, 0, 0)}"
                           range="{_v(0, g.stroke / 2)}" armature="{_f(g.jaw_armature)}"/>
                    <geom name="{s}_{name}_pad" class="pad" type="box" size="{_v(*pad)}"
                          mass="{_f(g.finger_kg)}"/>
                  </body>"""


def _actuators(spec: HumanoidSpec) -> str:
  """Every joint's driver as a position-mode `dcmotor` on its own part's
  line and its own gains (`humanoid.actuator.JointLimits.dcmotor`, issue
  #385's algebra), then the grippers'."""
  rows = []
  names = joint_names(spec)
  lim = JointLimits.of_motors([spec.motor(j) for j in names])
  gains = [driver_gains(spec.motor(j)) for j in names]
  for jn, d in zip(names, lim.dcmotor([g[0] for g in gains], [g[1] for g in gains])):
    rows.append(
      f'    <dcmotor name="{jn}" joint="{jn}" input="position" '
      f'motorconst="{_exact(*d["motorconst"])}" resistance="{_exact(d["resistance"])}" '
      f'saturation="{_exact(*d["saturation"])}" controller="{_exact(*d["controller"])}"/>')
  for s in SIDES:
    d = hp.GRIPPER.dcmotor()
    rows.append(
      f'    <dcmotor name="{s}_gripper" joint="{s}_{JAWS[0]}" input="position" '
      f'motorconst="{_exact(*d["motorconst"])}" resistance="{_exact(d["resistance"])}" '
      f'saturation="{_exact(*d["saturation"])}" controller="{_exact(*d["controller"])}"/>')
  return "\n".join(rows)


def _trunk(spec: HumanoidSpec) -> dict[str, str]:
  """The pelvis, the torso and the head, with the hips' and shoulders'
  pitch motors, the pack, the electronics and the sensors, in the pieces
  `body_xml` lays: the pelvis's own (its IMU and box), the hips' motors on
  it, and what rides above the waist -- the torso with the pack and the
  electronics, the shoulders' motors, and the neck, the head, the port and
  the sensors. Without the waist they are one rigid body, in that order but
  the hips' motors after the torso's three (the drawing before #560's
  stage (c3), to the byte)."""
  px, py, pz = spec.pelvis_half
  tx, ty, tz = spec.torso_half
  pelvis = _kg(spec, "pelvis_side", "pelvis_deck")
  torso = _kg(spec, "torso_side", "torso_back", "torso_top", "torso_cover")
  pack = hp.PACK
  hips = "".join(_housing(f"{s}_hip_pitch_motor", spec.motor(f"{s}_hip_pitch"),
                          (0, (1 if s == "left" else -1) * spec.hip_pitch_y, 0), "y")
                 for s in SIDES)
  shoulders = "".join(
    _housing(f"{s}_shoulder_pitch_motor", spec.motor(f"{s}_shoulder_pitch"),
             (0, (1 if s == "left" else -1) * spec.shoulder_pitch_y, spec.shoulder_z), "y")
    for s in SIDES)
  dp = math.radians(spec.depth_pitch_deg)
  npch = math.radians(spec.nav_pitch_deg)
  return {"pelvis": f"""
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="pelvis" class="limb" type="box" size="{_v(px, py, pz)}"
            mass="{_f(pelvis)}"/>""", "hips": hips, "torso": f"""
      <geom name="torso" class="limb" type="box" size="{_v(tx, ty, tz)}"
            pos="{_v(0, 0, spec.torso_z)}" mass="{_f(torso)}"/>
      <geom name="pack" class="visual" type="box" size="{_v(*pack.half)}"
            pos="{_v(-tx + pack.half[0] + 0.01, 0, spec.torso_z - tz + pack.half[2] + 0.01)}"
            mass="{_f(pack.kg)}" rgba="0.15 0.15 0.2 1"/>
      <geom name="electronics" class="visual" type="box" size="0.04 0.08 0.05"
            pos="{_v(tx - 0.05, 0, spec.torso_z + 0.06)}" mass="{_f(hp.ELECTRONICS_KG)}"
            rgba="0.1 0.5 0.2 1"/>""", "shoulders": shoulders, "head": f"""
      <geom name="neck" class="visual" type="cylinder" size="0.025"
            fromto="{_v(0, 0, spec.torso_z + tz, 0, 0, spec.face_lo)}" mass="0" rgba="0.3 0.3 0.32 1"/>{_head(spec)}
      <geom name="charge_port" class="visual" type="box"
            size="{_v(hp.PORT_FLANGE[0] / 2, hp.PORT_FLANGE[1] / 2, hp.PORT_FLANGE[1] / 2)}"
            pos="{_v(tx + hp.PORT_FLANGE[0] / 2, spec.port_y, spec.port_z)}"
            mass="{_f(hp.PORT_KG)}" rgba="0.75 0.6 0.2 1"/>
      <geom name="charge_funnel" class="visual" type="cylinder"
            size="{_v(hp.PORT_FUNNEL[0] / 2, hp.PORT_FUNNEL[1] / 2)}" zaxis="1 0 0"
            pos="{_v(tx + hp.PORT_FLANGE[0] + hp.PORT_FUNNEL[1] / 2, spec.port_y, spec.port_z)}"
            mass="{_f(hp.PORT_FUNNEL_KG)}" rgba="0.2 0.2 0.22 0.35"/>
      <site name="charge_port" pos="{_v(tx + hp.PORT_FLANGE[0], spec.port_y, spec.port_z)}"
            zaxis="1 0 0" size="0.004"/>
      <body name="depth_cam_body" pos="{_v(DEPTH_CAM_X, 0, spec.depth_cam_z)}" euler="{_v(0, dp, 0)}">
        <geom name="depth_cam" class="visual" type="box" size="0.0125 0.045 0.0125"
              mass="{_f(hp.D435_KG)}" rgba="0.2 0.2 0.2 1"/>
        <camera name="depth_eye" pos="{_v(CAM_AHEAD_M, 0, 0)}" xyaxes="0 -1 0 0 0 1" fovy="58"/>
        <camera name="color_eye" pos="{_v(CAM_AHEAD_M, COLOR_ACROSS_M, 0)}" xyaxes="0 -1 0 0 0 1" fovy="{_f(COLOR_FOVY_DEG)}"/>
      </body>
      <body name="nav_cam_body" pos="{_v(NAV_CAM_X, 0, spec.nav_cam_z)}" euler="{_v(0, npch, 0)}">
        <geom name="nav_cam" class="visual" type="box" size="{_v(*NAV_CAM_HALF)}"
              mass="{_f(hp.CAMERA_KG)}" rgba="0.1 0.1 0.1 1"/>
        <camera name="nav_eye" pos="0.007 0 0" xyaxes="0 -1 0 0 0 1" fovy="41"/>
      </body>{_lidar(spec)}"""}


#: The waist's turntable plate's colour: the aluminium's own, unpainted.
WAIST_PLATE_RGBA = "0.6 0.6 0.62 1"


def _waist(spec: HumanoidSpec, above: str) -> str:
  """The waist's yaw (`HumanoidSpec.waist_yaw`): its motor stood in the
  pelvis's middle, the link it turns at the pelvis's top with the turntable
  plate on it, and `above` -- everything that rides it -- laid through a
  frame that undoes the link's offset, so the waist at zero is the body
  without it, every part where it was (#620's `waist_spike.waist_xml("yaw")`,
  pinned to it)."""
  m, z = spec.motor("waist_yaw"), spec.waist_z
  px, py, _ = spec.pelvis_half
  plate = structure(spec)["waist_plate"][0]
  return f"""
      {_housing("waist_yaw_motor", m, (0, 0, z - m.length / 2), "z")}
      <body name="{BODY_OF['waist_yaw']}" pos="{_v(0, 0, z)}">
        {_joint("waist_yaw", "z", m)}
        <geom name="waist_plate" class="visual" type="box" size="{_v(px, py, plate.thick / 2)}"
              pos="{_v(0, 0, plate.thick / 2)}" mass="{_f(plate.kg)}" rgba="{WAIST_PLATE_RGBA}"/>
        <frame pos="{_v(0, 0, -z)}">{above}
        </frame>
      </body>"""


def _head(spec: HumanoidSpec) -> str:
  """The head built round the LIDAR (#581, stage 3): the face, the cylinder
  the cameras ride in its visor, and the dome over it, the slope under the
  window; the window, the unit's own (`_lidar`), open all round but where
  the fin that carries the cap crosses it, and where the back's panels
  close it under the band's rings; the cap over the unit's base, on its
  padded rim. Every part collides but the panels' painted inner faces, and
  a fall on the head is the rim's and the cap's."""
  r, z = spec.head_r, spec.lidar_z
  lo, top = spec.face_lo, face_top(spec)
  back, front, half_w, foot = VISOR
  rim_lo = _rim_lo(spec)
  cap_lo, cap_hi = rim_lo + RIM[1], z + CAP_TOP
  thick, inner = FIN
  fin_lo, fin_hi = top - 0.004, z + hp.LIDAR.origin
  out = f"""
      <geom name="head" class="limb" type="cylinder" size="{_v(r, (top - lo) / 2)}"
            pos="{_v(0, 0, (lo + top) / 2)}" mass="{_f(_kg(spec, 'head_shell'))}"/>
      <geom name="head_visor" class="limb" type="box"
            size="{_v((front - back) / 2, half_w, (_visor_top(spec) - foot) / 2)}"
            pos="{_v((back + front) / 2, 0, (foot + _visor_top(spec)) / 2)}"
            mass="{_f(_kg(spec, 'head_visor'))}" rgba="{VISOR_RGBA}"/>
      <geom name="head_dome" class="limb" type="sphere" size="{_f(DOME_R)}"
            pos="{_v(0, 0, _dome_z(spec))}" mass="{_f(_kg(spec, 'head_dome'))}" rgba="{VISOR_RGBA}"/>
      <geom name="head_rim" class="limb" type="cylinder" size="{_v(r + RIM[0], RIM[1] / 2)}"
            pos="{_v(0, 0, rim_lo + RIM[1] / 2)}" mass="{_f(_kg(spec, 'head_rim'))}"
            rgba="{RIM_RGBA}"/>
      <geom name="head_cap" class="limb" type="cylinder" size="{_v(r, (cap_hi - cap_lo) / 2)}"
            pos="{_v(0, 0, (cap_lo + cap_hi) / 2)}"
            mass="{_f(_kg(spec, 'head_cap', 'lidar_plate'))}"/>
      <geom name="head_fin" class="limb" type="box"
            size="{_v((r - inner) / 2, thick / 2, (fin_hi - fin_lo) / 2)}"
            pos="{_v(-(r + inner) / 2, 0, (fin_lo + fin_hi) / 2)}"
            mass="{_f(_kg(spec, 'lidar_fin'))}"/>"""
  arc, n, wall, _ = BACK
  half_w, back_hi = _back_panel(spec)
  step = math.radians(2 * arc / n)
  for k in range(n):
    a = math.pi - math.radians(arc) + step * (k + 0.5)
    c, s = math.cos(a), math.sin(a)
    for name, rad, half_t, kind, extra in (
        (f"head_back{k}", r - wall / 2, wall / 2, "limb",
         f'mass="{_f(_kg(spec, "head_back") / n)}"'),
        (f"head_back{k}_inside", r - wall - 0.0005, 0.0005, "visual",
         f'mass="0" rgba="{VISOR_RGBA}"')):
      out += f"""
      <geom name="{name}" class="{kind}" type="box" size="{_v(half_t, half_w, (back_hi - top) / 2 + 0.002)}"
            pos="{_v(rad * c, rad * s, (top + back_hi) / 2 - 0.002)}" euler="{_v(0, 0, a)}" {extra}/>"""
  return out


def _lidar(spec: HumanoidSpec) -> str:
  """The LIDAR upside down in the head (#581): its base bolted under the
  cap's plate -- on rubber isolators, the bill's, rigid here -- its window
  under it (`_head`), its rays' origin `lidar_z` over the hip axes. A body
  of its own, so that its rays never meet its own housing -- they leave
  through its window (`perception.lidar.Lidar`) -- and it collides as the
  head does."""
  ux, uy, uz = (v / 2 for v in hp.LIDAR.size)
  return f"""
      <body name="lidar_mount" pos="{_v(0, 0, spec.lidar_z)}">
        <geom name="lidar" class="limb" type="box" size="{_v(ux, uy, uz)}"
              pos="{_v(0, 0, hp.LIDAR.origin - uz)}" mass="{_f(hp.LIDAR.kg)}"
              rgba="0.05 0.05 0.05 1"/>
        <site name="lidar" pos="0 0 0" size="0.005"/>
      </body>"""


def body_xml(spec: HumanoidSpec = CHOSEN, *, root: str = ROBOT_ROOT,
             standalone: bool = True, scenery: str = "", assets: str = "",
             after: str = "") -> str:
  """The robot as MJCF. `standalone` wraps it with a floor, a light, the
  solver options and any `scenery` (bodies and geoms for the worldbody, and
  the `assets` they use), so it compiles alone; otherwise a
  `<mujocoinclude>`, which `attachable` turns into a spec. ⚠ Scenery with
  JOINTS goes in `after`, a worldbody after the robot's, so the robot's
  free joint stays at qpos[0].

  A joint's `ctrl` is a target its driver -- its part's torque-speed line
  and a PD whose gains each command carries (`humanoid.drivers`) -- runs in
  C, the quadruped's algebra (#385)."""
  h0 = stand_height(spec)
  legs = "".join(_leg(spec, s) for s in SIDES)
  arms = "".join(_arm(spec, s) for s in SIDES)
  t = _trunk(spec)
  if spec.waist_yaw:
    body = (t["pelvis"] + t["hips"] + legs
            + _waist(spec, t["torso"] + t["shoulders"] + t["head"] + arms))
  else:
    body = t["pelvis"] + t["torso"] + t["hips"] + t["shoulders"] + t["head"] + legs + arms
  robot = f"""
  <default>
    <default class="humanoid">
      <joint damping="0"/>
      <default class="motor">
        <geom contype="0" conaffinity="0" group="2" rgba="{MOTOR_RGBA}"/>
      </default>
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="limb">
        <geom type="capsule" contype="1" conaffinity="0" group="1" rgba="{LIMB_RGBA}"/>
      </default>
      <default class="sole">
        <geom condim="3" priority="1" friction="{SOLE_FRICTION}" contype="1"
              conaffinity="0" group="1" rgba="{SOLE_RGBA}"/>
      </default>
      <default class="pad">
        <geom condim="4" priority="1" friction="{PAD_FRICTION}" solref="{PAD_SOLREF}"
              solimp="{PAD_SOLIMP}" contype="1" conaffinity="0" group="1"
              rgba="0.1 0.1 0.1 1"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="{root}" pos="{_v(0, 0, h0)}" childclass="humanoid">
      <freejoint name="{root}_root"/>{body}
    </body>
  </worldbody>

  <equality>
{_jaw_equalities()}
  </equality>

  <contact>
{_leg_pairs()}
  </contact>

  <actuator>
{_actuators(spec)}
  </actuator>

  <sensor>
    <framequat name="pelvis_quat" objtype="site" objname="imu"/>
    <gyro name="imu_ang_vel" site="imu"/>
    <accelerometer name="imu_lin_acc" site="imu"/>
    <velocimeter name="imu_lin_vel" site="imu"/>
    <subtreeangmom name="root_angmom" body="{root}"/>
{_foot_sensors()}
  </sensor>
"""
  key = _keyframes(spec)
  if not standalone:
    return f"<mujocoinclude>{robot}</mujocoinclude>\n"
  return f"""<mujoco model="{spec.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality offsamples="0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.8"
             rgb2="0.7 0.7 0.68" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>{assets}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid"/>{scenery}
  </worldbody>
{robot}{key}{f'''
  <worldbody>
    {after}
  </worldbody>''' if after else ""}
</mujoco>
"""


#: The jaws' coupling, as stiff as the step allows (twice the 2 ms step): a
#: gripper's jaws are geared together, and on MuJoCo's default (0.02 s)
#: the second lagged the first 2.8 mm under a full squeeze, so the gap the
#: gripper reports off its one encoder read 37 mm on a 40 mm block (0.7 mm
#: and 39.3 here).
JAWS_SOLREF = "0.004 1"


def _jaw_equalities() -> str:
  """Each gripper's second jaw held to its first: one motor drives both
  through the gripper's own gear, so they close together."""
  return "\n".join(f'    <joint name="{s}_jaws" joint1="{s}_{JAWS[1]}" '
                   f'joint2="{s}_{JAWS[0]}" polycoef="0 1 0 0 0" solref="{JAWS_SOLREF}"/>'
                   for s in SIDES)


#: The legs' parts that meet each other in a stride, a crossing or a stumble:
#: the robot's own geoms touch nothing of the robot's (its policies' model
#: has no self-contact), so without these a gait could pass one leg through
#: the other. At the stand they are 10-12 cm apart.
LEG_GEOMS = ("thigh", "shank", "sole")
#: ...and an arm's against the trunk, the legs' thighs and shanks and the
#: other arm: a robot getting up off the floor leans on its arms, and an arm
#: that passed through its own trunk or thigh would push off nothing a real
#: one could (#507, stage 5). At the stand they are 3.3 cm apart or more.
ARM_GEOMS = ("upper_arm", "forearm", "gripper")
#: The head's parts (`_head`) and the LIDAR in it, which an arm raised by
#: the head meets as it meets the trunk.
HEAD_GEOMS = (("head", "head_visor", "head_dome", "head_rim", "head_cap", "head_fin", "lidar")
              + tuple(f"head_back{k}" for k in range(BACK[1])))
TRUNK_GEOMS = ("pelvis", "torso") + HEAD_GEOMS


def _leg_pairs() -> str:
  """The left leg's parts against the right's, each pair explicit; then
  each arm's against the trunk, both legs and the other arm."""
  legs = [f'    <pair geom1="left_{a}" geom2="right_{b}"/>'
          for a in LEG_GEOMS for b in LEG_GEOMS]
  limbs = TRUNK_GEOMS + tuple(f"{t}_{g}" for t in SIDES for g in LEG_GEOMS[:2])
  arms = [f'    <pair geom1="{s}_{a}" geom2="{b}"/>'
          for s in SIDES for a in ARM_GEOMS for b in limbs]
  arms += [f'    <pair geom1="left_{a}" geom2="right_{b}"/>'
           for a in ARM_GEOMS for b in ARM_GEOMS]
  return "\n".join(legs + arms)


def _foot_sensors() -> str:
  return "\n".join(f'    <touch name="{s}_foot_touch" site="{s}_foot"/>' for s in SIDES)


def qpos_order(spec: HumanoidSpec = CHOSEN) -> list[str]:
  """Every hinge and slide joint in the order qpos carries them after the
  free joint's seven: the tree's (legs, the waist, then each arm and its
  jaws)."""
  order = [f"{s}_{j}" for s in SIDES for j in LEG]
  order += [j for j in WAIST if spec.waist_yaw]
  for s in SIDES:
    order += [f"{s}_{j}" for j in ARM] + [f"{s}_{j}" for j in JAWS]
  return order


def _keyframes(spec: HumanoidSpec) -> str:
  rows = []
  for name, q, h in (("stand", stand_qpos(spec), stand_height(spec)),
                     ("zero", {}, spec.leg_length)):
    joints = " ".join(_f(q.get(j, 0.0)) for j in qpos_order(spec))
    rows.append(f'    <key name="{name}" qpos="0 0 {_f(h)} 1 0 0 0 {joints}"/>')
  return "\n  <keyframe>\n" + "\n".join(rows) + "\n  </keyframe>"


def mass(spec: HumanoidSpec = CHOSEN) -> float:
  """The whole robot, kg, off the compiled model: the parts' and the
  fabricated parts' masses as the generator laid them."""
  import mujoco
  model = mujoco.MjModel.from_xml_string(body_xml(spec))
  return float(model.body_subtreemass[model.body(ROBOT_ROOT).id])


def attachable(spec: HumanoidSpec = CHOSEN, **kw):
  """The robot as an `MjSpec` to attach into another world, its angles in
  RADIANS: the include form carries no `<compiler>`, and parsed on its own
  MuJoCo reads a joint's range in degrees (the quadruped's knee was thrown
  0.4 m into the air by its own limits so, #386)."""
  import mujoco
  xml = body_xml(spec, standalone=False, **kw)
  xml = xml.replace("<mujocoinclude>",
                    '<mujoco>\n  <compiler angle="radian" autolimits="true"/>')
  return mujoco.MjSpec.from_string(xml.replace("</mujocoinclude>", "</mujoco>"))


def training_constants(spec: HumanoidSpec = CHOSEN) -> dict:
  """What `training/` needs of the body, as data: it cannot import pluggybot
  (mjlab caps numpy below pluggybot's), so it reads this file. Each joint
  its own part's numbers, on the pack's nominal bus."""
  from pluggybot.humanoid import actuator as act
  joints = {}
  names = joint_names(spec)
  for j in names:
    m = spec.motor(j).at_voltage(act.BUS_V_NOMINAL)
    kp, kd = driver_gains(spec.motor(j))
    joints[j] = {
      "motor": m.name, "peak_torque": m.peak_torque, "rated_torque": m.rated_torque,
      "saturation_torque": round(m.saturation_torque, 6),
      "noload_speed": round(m.noload_speed, 6), "armature": round(m.armature, 9),
      "frictionloss": round(hp.friction_nm(m), 6),
      "stiffness": round(float(kp), 6), "damping": round(float(kd), 6),
      "range": [round(v, 6) for v in joint_range(j)],
    }
  return {
    "root": ROBOT_ROOT,
    "joints": joints,
    "order": list(names),
    # ...and by what drives them: the legs the policies', the waist and the
    # arms scripted in training as the served reach drives them
    "groups": {"legs": [j for j in names if kind(j) in LEG],
               "waist": [j for j in names if j in WAIST],
               "arms": [j for j in names if kind(j) in ARM]},
    "feet": [f"{s}_foot" for s in SIDES],
    "stand_height": round(stand_height(spec), 6),
    "stand_qpos": {j: round(v, 6) for j, v in stand_qpos(spec).items()},
    "bus_v": act.BUS_V_NOMINAL,
    "gripper": {"name": hp.GRIPPER.name, "stroke": hp.GRIPPER.stroke,
                "grip_force_n": hp.GRIPPER.grip_force_n},
    "arm_gain_scale": list(ARM_GAIN_SCALE),
    "waist_gain_scale": list(WAIST_GAIN_SCALE),
  }


def write_body(path: str = MODEL_XML, spec: HumanoidSpec = CHOSEN) -> None:
  import json
  with open(path, "w") as fh:
    fh.write("<!-- GENERATED by pluggybot.humanoid.model (issue #507): uv run python -m "
             "pluggybot.humanoid.model. Proportions and joint ranges from Booster T2's "
             "published model, BSD-3-Clause, Copyright (c) 2025 BoosterRobotics "
             "(models/booster_assets.LICENSE). -->\n")
    fh.write(body_xml(spec))
  with open(CONSTANTS_JSON, "w") as fh:
    json.dump(training_constants(spec), fh, indent=1)
    fh.write("\n")


if __name__ == "__main__":
  write_body()
  print(f"wrote {MODEL_XML} and {CONSTANTS_JSON}")
