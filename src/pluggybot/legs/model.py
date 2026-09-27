"""The quadruped's MJCF, generated from a `BodySpec` (issue #377).

One function, `body_xml(spec)`, so a sizing sweep can build every candidate
in memory and the chosen one is written to `models/quadruped.xml` and tested
against the generator (the home world's pattern). Nothing deployed loads it.

The layout is the Mini Cheetah / Go1 one: a hip ABDUCTION motor in each
corner of the torso (axis along x), and on its link the hip FLEXION motor
and the KNEE motor stacked coaxially (axis along y), the knee driven down the
thigh by a belt of ratio `knee_ratio`. So each leg's two upper motors ride
the abduction link and the thigh and shank carry only tube, belt and foot:
light legs are what let a policy swing them fast. Zero joint angles are the
leg hanging straight down; the knee bends BACK (hip flexion positive, knee
negative), the `>` every quadruped of this class stands in.

Masses are a BUDGET, each line a part (Parts.md, "The quadruped body"):
actuators from the datasheet, the rest estimated from the parts named at
`MassBudget`, and #378's arm a placeholder until #378 sizes it. Inertia is
MuJoCo's, from each geom's shape at uniform density.
"""

from dataclasses import dataclass, field, replace
import math

from pluggybot.legs.actuator import FRICTION_NOMINAL, GIM4305_10, GIM8108_8, Motor
from pluggybot.telemetry.protocol import ROBOT_ROOT

LEGS = ("FL", "FR", "HL", "HR")
JOINTS = ("hip_abd", "hip_flex", "knee")
#: Every leg joint, in the order qpos/ctrl/the policy's action carry them.
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEGS for j in JOINTS)


@dataclass(frozen=True)
class MassBudget:
  """Everything the torso carries that is not an actuator, in kg."""

  #: Frame plates, standoffs, hip brackets' torso half, fasteners: two 3 mm
  #: aluminium side plates 0.40 x 0.11 m (0.36 kg each at 2.7 g/cm^3) plus
  #: end caps and cross members.
  frame: float = 1.00
  #: The pack (`battery_*`); sized by the energy table, #377 item 5.
  battery: float = 0.95
  #: Compute, the drivers' CAN adapter, the buck converters, wiring.
  electronics: float = 0.40
  #: The 2D laser scanner (RPLIDAR C1 class, 110 g, Parts.md) + its mast.
  #: (`scanner`, not `lidar`: tests/test_body.py fences `.lidar` as the
  #: rover's sensor object.)
  scanner: float = 0.13
  #: RealSense D435 (72 g, Parts.md) + bracket.
  depth_cam: float = 0.09
  #: Two camera modules + mounts.
  cameras: float = 0.03
  #: #378's arm, stowed, WITHOUT the tool: two actuators and two links.
  #: A placeholder budget until #378 sizes it -- the torque table is flown at
  #: this figure and must be re-flown if #378 needs more.
  arm: float = 0.90
  #: The tool at the arm's tip (the rover's ~250 g practical ceiling,
  #: ToolPattern.md "Mass and geometry class").
  tool: float = 0.25

  @property
  def total(self) -> float:
    return (self.frame + self.battery + self.electronics + self.scanner
            + self.depth_cam + self.cameras + self.arm + self.tool)


@dataclass(frozen=True)
class BodySpec:
  """A candidate body. Lengths in metres, masses in kg."""

  name: str = "quadruped"
  thigh: float = 0.21
  shank: float = 0.21
  foot_r: float = 0.022
  #: Torso half-extents (x, y, z).
  torso: tuple[float, float, float] = (0.21, 0.10, 0.055)
  #: The abduction axis's corner, from the torso centre (x, |y|).
  hip_x: float = 0.19
  hip_y: float = 0.05
  #: Abduction axis -> the leg's plane, outboard: half the flexion/knee motor
  #: stack plus a bracket.
  hip_out: float = 0.085
  motor: Motor = GIM8108_8
  #: The knee's belt reduction AFTER the motor's own gearbox (1.0 = direct).
  knee_ratio: float = 1.0
  #: Thigh and shank structure (tube, belt, pulleys, foot shell), kg each.
  thigh_mass: float = 0.12
  shank_mass: float = 0.07
  foot_mass: float = 0.02
  masses: MassBudget = field(default_factory=MassBudget)
  #: The arm's pose for the torque table: stowed along the torso's top, or
  #: reaching straight forward at full length with the tool.
  arm_reach: bool = False
  arm_length: float = 0.55
  #: Hip axis -> the belly's underside. The battery hangs below the torso as
  #: a belly pack so the robot can LIE on it with its shanks flat and its
  #: legs unloaded: a folded leg (knee at its stop) holds the hips 0.10 m up,
  #: so a belly any shallower leaves the legs carrying the robot at rest.
  belly_depth: float = 0.105

  @property
  def leg_length(self) -> float:
    return self.thigh + self.shank

  @property
  def stand_height(self) -> float:
    """The torso's nominal standing height (hip axis over the floor)."""
    return round(0.72 * self.leg_length + self.foot_r, 4)

  @property
  def mass(self) -> float:
    """The whole robot, kg."""
    legs = 4 * (3 * self.motor.mass + self.thigh_mass + self.shank_mass
                + self.foot_mass)
    return legs + self.masses.total

  def with_(self, **kw) -> "BodySpec":
    return replace(self, **kw)


#: The body #377 proposes; `models/quadruped.xml` is its standalone MJCF.
CHOSEN = BodySpec()
MODEL_XML = "models/quadruped.xml"
#: The body's constants for `training/`, which cannot import this package.
CONSTANTS_JSON = "models/quadruped.json"

#: The small body the table rejects: Stanford Pupper v3's published geometry
#: (cs123 pupper-mjlab: thigh 0.0845, shank 0.088, foot 0.02, hips +-0.075
#: fore-aft and ~0.078 out) on its GIM4305-10s, 3.0 kg as built (the model
#: says 3.17). `PUPPER_WITH_SUITE` is the same body carrying what #377's
#: robot must: the sensors, #378's arm and a tool.
PUPPER_CLASS = BodySpec(
  name="pupper-class", thigh=0.0845, shank=0.088, foot_r=0.02,
  torso=(0.10, 0.05, 0.03), hip_x=0.075, hip_y=0.035, hip_out=0.043,
  motor=GIM4305_10, thigh_mass=0.036, shank_mass=0.04, foot_mass=0.01,
  belly_depth=0.05, arm_length=0.30,
  masses=MassBudget(frame=0.90, battery=0.0, electronics=0.0, scanner=0.0,
                    depth_cam=0.0, cameras=0.0, arm=0.0, tool=0.0))
PUPPER_WITH_SUITE = PUPPER_CLASS.with_(
  name="pupper-class + suite",
  masses=MassBudget(frame=0.90, battery=0.0, electronics=0.0))


def _f(v: float) -> str:
  return f"{v:.6g}"


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def leg_ik(spec: BodySpec, x: float, z: float) -> tuple[float, float]:
  """Hip flexion and knee angles putting the foot at (x, z) in the leg's
  plane, relative to the flexion axis (z < 0 is down). Knee bent back."""
  a, b = spec.thigh, spec.shank
  d2 = x * x + z * z
  c = (d2 - a * a - b * b) / (2 * a * b)
  knee = -math.acos(max(-1.0, min(1.0, c)))
  # Foot = -a*(sin t, cos t) - b*(sin(t+k), cos(t+k)); solve for t.
  alpha = math.atan2(-x, -z)
  beta = math.atan2(b * math.sin(-knee), a + b * math.cos(knee))
  return alpha + beta, knee


def lie_qpos(spec: BodySpec) -> list[float]:
  """The rest posture: belly down, each shank flat on the floor ahead of
  its knee and the thigh sloping up to the hip. Nothing is held."""
  flex = math.acos((spec.belly_depth - spec.foot_r) / spec.thigh)
  return [0.0, flex, -math.pi / 2 - flex] * 4


def pose_qpos(spec: BodySpec, height: float, x_off: float = 0.0,
              abd: float = 0.0) -> list[float]:
  """Twelve joint angles for every foot at `height` under its flexion axis
  (torso level), shifted `x_off` forward."""
  flex, knee = leg_ik(spec, x_off, -(height - spec.foot_r))
  return [abd, flex, knee] * 4


#: The belly's charge pads (#378, `legs/dock.py`): two strips along x on
#: the pack's underside, FLUSH with it, so the robot rests on them anywhere
#: it lies and they meet the dock's pins where it docks. Half-extents, and
#: the strips' centres at y = +-PAD_Y. Their mass is the pack's. 3 mm deep
#: in the sim (a real pad is a plated strip): a pin's tip is 1.27 mm across,
#: and landing hard it sank past a 1 mm pad into the case behind it.
PAD_HALF = (0.08, 0.01, 0.0015)
PAD_Y = 0.03


def _pads(spec: BodySpec) -> str:
  z = -spec.belly_depth + PAD_HALF[2]
  return "".join(
    f"""
      <geom name="belly_pad_{lbl}" type="box" size="{_v(*PAD_HALF)}"
            pos="{_v(0, side * PAD_Y, z)}" mass="0" contype="1" conaffinity="0"
            group="1" rgba="0.85 0.65 0.2 1"/>"""
    for side, lbl in ((1, "l"), (-1, "r")))


def body_xml(spec: BodySpec, *, root: str = ROBOT_ROOT,
             standalone: bool = True, scenery: str = "", assets: str = "",
             after: str = "") -> str:
  """The robot as MJCF. `standalone` wraps it with a floor, a light, the
  solver options and any `scenery` (MJCF bodies/geoms for the worldbody, and
  the `assets` they use), so it compiles alone; otherwise a
  `<mujocoinclude>`. ⚠ Scenery with JOINTS goes in `after`, a worldbody
  after the robot's: the keyframes (and `quad_spike`'s routines) index the
  robot's free joint from qpos[0]."""
  m = spec.motor
  tx, ty, tz = spec.torso
  h0 = spec.stand_height
  budget = spec.masses
  # Flexion + knee motor stack on the abduction link, one cylinder along y.
  stack_r, stack_h = m.diameter / 2, 2 * m.length
  armature = m.rotor_inertia * m.ratio ** 2
  knee_arm = armature * spec.knee_ratio ** 2
  j_range = {"hip_abd": (-0.8, 0.8), "hip_flex": (-1.6, 3.3),
             "knee": (-2.75, -0.35)}
  # A joint's torque limit at the joint, which the actuator's ctrlrange is.
  peak = {"hip_abd": m.peak_torque, "hip_flex": m.peak_torque,
          "knee": m.peak_torque * spec.knee_ratio}
  arm_x = tx - 0.04

  legs = []
  for leg in LEGS:
    sx = 1 if leg[0] == "F" else -1
    sy = 1 if leg[1] == "L" else -1
    out = sy * spec.hip_out
    legs.append(f"""
      <body name="{leg}_hip" pos="{_v(sx * spec.hip_x, sy * spec.hip_y, 0)}">
        <joint name="{leg}_hip_abd" class="abd" range="{_v(*j_range['hip_abd'])}"
               armature="{_f(armature)}"/>
        <geom name="{leg}_stack" class="visual" type="cylinder" size="{_v(stack_r, stack_h / 2)}"
              pos="{_v(0, out * 0.55, 0)}" euler="1.5708 0 0" mass="{_f(2 * m.mass)}"
              rgba="0.25 0.25 0.28 1"/>
        <body name="{leg}_thigh" pos="{_v(0, out, 0)}">
          <joint name="{leg}_hip_flex" class="flex" range="{_v(*j_range['hip_flex'])}"
                 armature="{_f(armature)}"/>
          <geom name="{leg}_thigh" class="limb" fromto="0 0 0 0 0 {_f(-spec.thigh)}"
                size="0.022" mass="{_f(spec.thigh_mass)}"/>
          <body name="{leg}_shank" pos="{_v(0, 0, -spec.thigh)}">
            <joint name="{leg}_knee" class="flex" range="{_v(*j_range['knee'])}"
                   armature="{_f(knee_arm)}"/>
            <geom name="{leg}_shank" class="limb" fromto="0 0 0 0 0 {_f(-spec.shank + spec.foot_r)}"
                  size="0.016" mass="{_f(spec.shank_mass)}"/>
            <geom name="{leg}_foot" class="foot" pos="{_v(0, 0, -spec.shank)}"
                  mass="{_f(spec.foot_mass)}"/>
            <site name="{leg}_foot" type="sphere" pos="{_v(0, 0, -spec.shank)}"
                  size="{_f(spec.foot_r + 0.004)}" group="4"/>
          </body>
        </body>
      </body>""")

  if spec.arm_reach:
    # Straight ahead from the shoulder at full length, the tool at the tip.
    arm_geoms = f"""
      <geom name="arm" class="visual" type="capsule" size="0.025"
            fromto="{_v(arm_x, 0, tz + 0.03, arm_x + spec.arm_length, 0, tz + 0.03)}"
            mass="{_f(budget.arm)}" rgba="0.9 0.6 0.1 1"/>
      <geom name="tool" class="visual" type="box" size="0.04 0.04 0.04"
            pos="{_v(arm_x + spec.arm_length, 0, tz + 0.03)}" mass="{_f(budget.tool)}"
            rgba="0.9 0.8 0.2 1"/>"""
  else:
    # Folded back along the torso's top, below the LIDAR's scan plane.
    arm_geoms = f"""
      <geom name="arm" class="visual" type="capsule" size="0.025"
            fromto="{_v(arm_x, 0, tz + 0.03, arm_x - 0.24, 0, tz + 0.03)}"
            mass="{_f(budget.arm)}" rgba="0.9 0.6 0.1 1"/>
      <geom name="tool" class="visual" type="box" size="0.04 0.04 0.035"
            pos="{_v(arm_x - 0.05, 0, tz + 0.09)}" mass="{_f(budget.tool)}"
            rgba="0.9 0.8 0.2 1"/>"""

  # The actuators write torque: the driver's PD loop and the torque-speed
  # envelope are `legs.actuator`'s, run in Python at the physics rate, so
  # the XML carries only the peak as a hard stop.
  actuators = "\n".join(
    f'    <motor name="{jn}" joint="{jn}" ctrlrange="{_v(-peak[jn.split("_", 1)[1]], peak[jn.split("_", 1)[1]])}"/>'
    for jn in JOINT_NAMES)
  # Named as mjlab's velocity task reads them (`robot/imu_ang_vel`, ...), so
  # the training side (`training/`) uses this file unchanged.
  sensors = "\n".join(
    ['    <framequat name="torso_quat" objtype="site" objname="imu"/>',
     '    <gyro name="imu_ang_vel" site="imu"/>',
     '    <accelerometer name="imu_lin_acc" site="imu"/>',
     '    <velocimeter name="imu_lin_vel" site="imu"/>',
     f'    <subtreeangmom name="root_angmom" body="{root}"/>']
    + [f'    <touch name="{leg}_touch" site="{leg}_foot"/>' for leg in LEGS])

  robot = f"""
  <default>
    <default class="quad">
      <joint damping="0" frictionloss="{_f(FRICTION_NOMINAL)}"/>
      <default class="abd"><joint axis="1 0 0"/></default>
      <default class="flex"><joint axis="0 1 0"/></default>
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2" density="0"/>
      </default>
      <default class="limb">
        <geom type="capsule" contype="1" conaffinity="0" group="1"
              rgba="0.55 0.55 0.6 1"/>
      </default>
      <!-- Rubber on the floor. priority="1": friction combines as the pair's
           MAX otherwise, and the floor's 1.0 would win whatever the foot
           says (SimNotes, "THE caster lesson"), so the foot's own value is
           what training randomises. -->
      <default class="foot">
        <geom type="sphere" size="{_f(spec.foot_r)}" condim="3" priority="1"
              friction="0.9 0.02 0.01" contype="1" conaffinity="0" group="1"
              rgba="0.1 0.1 0.1 1"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="{root}" pos="{_v(0, 0, h0)}" childclass="quad">
      <freejoint name="{root}_root"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="torso" type="box" size="{_v(tx, ty, tz)}" mass="{_f(budget.frame)}"
            contype="1" conaffinity="0" group="1" rgba="0.2 0.4 0.8 1"/>
      <!-- The belly pack: the battery, below the torso. It is what the robot
           lies on (and #378's dock contacts are on its underside). -->
      <geom name="belly" type="box" size="{_v(0.11, 0.06, (spec.belly_depth - tz) / 2)}"
            pos="{_v(0, 0, -(spec.belly_depth + tz) / 2)}" mass="{_f(budget.battery)}"
            contype="1" conaffinity="0" group="1" rgba="0.15 0.15 0.2 1"/>{_pads(spec)}
      <geom name="electronics" class="visual" type="box" size="0.06 0.05 0.015"
            pos="-0.05 0 0.03" mass="{_f(budget.electronics)}" rgba="0.1 0.5 0.2 1"/>
      <geom name="abd_motors" class="visual" type="box" size="{_v(tx - 0.02, ty - 0.01, tz - 0.01)}"
            mass="{_f(4 * m.mass)}" rgba="0.25 0.25 0.28 0.3"/>
      <!-- Sensors (#377 item 7). The LIDAR rides a short mast at the rear so
           its scan plane clears the stowed arm; the depth camera looks down
           and forward off the torso's nose, under the arm's shoulder. -->
      <body name="lidar_mast" pos="{_v(-tx + 0.06, 0, tz)}">
        <geom name="lidar_mast" class="visual" type="cylinder" size="0.012 0.04"
              pos="0 0 0.04" mass="{_f(min(0.02, budget.scanner))}" rgba="0.3 0.3 0.3 1"/>
        <geom name="lidar" class="visual" type="cylinder" size="0.0378 0.0205"
              pos="0 0 0.1" mass="{_f(budget.scanner - min(0.02, budget.scanner))}" rgba="0.05 0.05 0.05 1"/>
        <site name="lidar" pos="0 0 0.1" size="0.005"/>
      </body>
      <body name="depth_cam_body" pos="{_v(tx + 0.012, 0, 0.0)}" euler="0 0.5236 0">
        <geom name="depth_cam" class="visual" type="box" size="0.0125 0.045 0.0125"
              mass="{_f(budget.depth_cam)}" rgba="0.2 0.2 0.2 1"/>
        <camera name="depth_eye" pos="0.013 0 0" xyaxes="0 -1 0 0 0 1" fovy="58"/>
      </body>
      <body name="nav_cam_body" pos="{_v(tx + 0.005, 0, tz - 0.012)}">
        <geom name="nav_cam" class="visual" type="box" size="0.006 0.012 0.012"
              mass="{_f(budget.cameras)}" rgba="0.1 0.1 0.1 1"/>
        <camera name="nav_eye" pos="0.007 0 0" xyaxes="0 -1 0 0 0 1" fovy="41"/>
      </body>
      <site name="arm_shoulder" pos="{_v(arm_x, 0, tz + 0.03)}" size="0.01"/>{arm_geoms}
{''.join(legs)}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
{sensors}
  </sensor>
"""
  stand = " ".join(_f(q) for q in pose_qpos(spec, h0))
  lie = " ".join(_f(q) for q in lie_qpos(spec))
  key = f"""
  <keyframe>
    <key name="stand" qpos="0 0 {_f(h0)} 1 0 0 0 {stand}"/>
    <key name="lie" qpos="0 0 {_f(spec.belly_depth + 0.001)} 1 0 0 0 {lie}"/>
  </keyframe>"""
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
{robot}{key}{f"""
  <worldbody>
    {after}
  </worldbody>""" if after else ""}
</mujoco>
"""


def training_constants(spec: BodySpec = CHOSEN) -> dict:
  """What `training/` needs of the body, as data: it cannot import pluggybot
  (mjlab 1.5 caps numpy below pluggybot's), so it reads this file."""
  from pluggybot.legs import actuator as act
  m = spec.motor.at_voltage(act.BUS_V_NOMINAL)
  return {
    "root": ROBOT_ROOT,
    "joints": list(JOINT_NAMES),
    "legs": list(LEGS),
    "stand_height": spec.stand_height,
    "stand_qpos": [round(q, 6) for q in pose_qpos(spec, spec.stand_height)],
    "lie_qpos": [round(q, 6) for q in lie_qpos(spec)],
    "belly_depth": spec.belly_depth,
    "mass": round(spec.mass, 6),
    "knee_ratio": spec.knee_ratio,
    "motor": {
      "name": m.name, "peak_torque": m.peak_torque,
      "rated_torque": m.rated_torque,
      "saturation_torque": round(m.saturation_torque, 6),
      "noload_speed": round(m.noload_speed, 6), "bus_v": m.bus_v,
      "armature": m.armature,
      "armature_range": [r * m.ratio ** 2 for r in spec.motor.rotor_range],
    },
    "friction_nominal": act.FRICTION_NOMINAL,
    "friction_range": list(act.FRICTION_NM),
    "latency_s": list(act.LATENCY_S),
    "bus_v_range": list(act.BUS_V_RANGE),
    "backlash_rad": act.BACKLASH_RAD,
  }


def write_body(path: str = MODEL_XML, spec: BodySpec = CHOSEN) -> None:
  import json
  with open(path, "w") as fh:
    fh.write("<!-- GENERATED by pluggybot.legs.model (issue #377): "
             "uv run python -m pluggybot.legs.model -->\n")
    fh.write(body_xml(spec))
  with open(CONSTANTS_JSON, "w") as fh:
    json.dump(training_constants(spec), fh, indent=1)
    fh.write("\n")


if __name__ == "__main__":
  write_body()
  print(f"wrote {MODEL_XML} and {CONSTANTS_JSON}")
