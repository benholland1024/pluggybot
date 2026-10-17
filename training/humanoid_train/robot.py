"""The humanoid as an mjlab entity (#507), read off the files pluggybot
generates: `models/humanoid.xml` and `models/humanoid.json` (`uv run python
-m pluggybot.humanoid.model`). Nothing is typed twice: each joint is driven
by its own part's numbers, the five RobStride sizes each an actuator group.
"""

import json
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "models" / "humanoid.xml"
BODY = json.loads((REPO / "models" / "humanoid.json").read_text())
ROOT = BODY["root"]
JOINTS = BODY["joints"]
#: The legs' twelve and the arms' twelve, in the order qpos carries them.
LEGS = tuple(j for j in BODY["order"] if "_shoulder" not in j and "_elbow" not in j
             and "_wrist" not in j)
ARMS = tuple(j for j in BODY["order"] if j not in LEGS)
#: The soles' sites and geoms, and the hands (the bodies a payload or a pull
#: is put on).
FEET = tuple(BODY["feet"])
SOLES = tuple(f"{s}_sole" for s in ("left", "right"))
HANDS = ("left_hand", "right_hand")

#: The whole robot, kg, off its compiled drawing (a get-up's lift is a share
#: of its weight).
MASS_KG = float(mujoco.MjModel.from_xml_path(str(XML)).body_subtreemass[1])

#: The physics step training runs at, s, and the policy's every
#: `DECIMATION` of them (mjlab's velocity task: 50 Hz); the served sim runs
#: 0.002 and flies the policy there.
TRAIN_DT = 0.005
DECIMATION = 4
#: The drivers' latency, physics steps: a stated choice, as no RobStride
#: sheet gives one -- an MIT command answered within its 1 kHz CAN cycle,
#: and one cycle of a bus carrying six joints' frames on top.
DELAY_MAX_STEPS = 1


def get_spec() -> mujoco.MjSpec:
  """The robot alone: the standalone file's floor (and its texture), light,
  keyframes and drivers go (mjlab brings a terrain and builds its own actuators), and the
  grippers' jaws are held shut (`freeze_jaws`)."""
  spec = mujoco.MjSpec.from_file(str(XML))
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for asset in [*spec.materials, *spec.textures]:
    spec.delete(asset)
  for key in list(spec.keys):
    spec.delete(key)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  freeze_jaws(spec)
  return spec


def freeze_jaws(spec: mujoco.MjSpec) -> None:
  """Each gripper's jaws held where they are built (shut), their slides and
  the gear between them taken out: a balance policy's joints are the legs',
  and a jaw's 40 g travelling 45 mm moves nothing it feels."""
  jaws = {f"{s}_{j}" for s in ("left", "right") for j in ("jaw_a", "jaw_b")}
  for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    for joint in list(body.joints):
      if joint.name in jaws:
        spec.delete(joint)
  for eq in list(spec.equalities):
    spec.delete(eq)


def _group(motor: str) -> list[str]:
  return [j for j, v in JOINTS.items() if v["motor"] == motor]


def _actuator(motor: str, gain_scale: tuple[float, float] = (1.0, 1.0),
              joints: tuple[str, ...] | None = None) -> DcMotorActuatorCfg:
  """One part's joints as mjlab's DC motor: its torque-speed line through
  its no-load speed on the pack's bus, clipped at its peak, its reflected
  inertia and the friction its part is given; the PD its driver runs."""
  names = [j for j in _group(motor) if joints is None or j in joints]
  first = JOINTS[names[0]]
  return DcMotorActuatorCfg(
    target_names_expr=tuple(names),
    stiffness=first["stiffness"] * gain_scale[0],
    damping=first["damping"] * gain_scale[1],
    effort_limit=first["peak_torque"],
    saturation_effort=first["saturation_torque"],
    velocity_limit=first["noload_speed"],
    armature=first["armature"],
    frictionloss=first["frictionloss"],
    delay_min_lag=0,
    delay_max_lag=DELAY_MAX_STEPS,
  )


MOTORS = sorted({v["motor"] for v in JOINTS.values()})
#: The arms' drivers in training: the scripted arms (`stand.ScriptedArms`)
#: hold their targets at the gains the served arms' commands carry
#: (`pluggybot.humanoid.model.ARM_GAIN_SCALE` times the policy's), with the
#: arm's own weight fed forward, as the served arms do.
ARM_GAIN_SCALE = tuple(BODY["arm_gain_scale"])


def actuators() -> tuple[DcMotorActuatorCfg, ...]:
  out = []
  for motor in MOTORS:
    legs = tuple(j for j in _group(motor) if j in LEGS)
    arms = tuple(j for j in _group(motor) if j in ARMS)
    if legs:
      out.append(_actuator(motor, joints=legs))
    if arms:
      out.append(_actuator(motor, ARM_GAIN_SCALE, joints=arms))
  return tuple(out)


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, BODY["stand_height"] + 0.01),
  joint_pos={**{j: 0.0 for j in BODY["order"]}, **BODY["stand_qpos"]},
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=actuators(),
  soft_joint_pos_limit_factor=0.9,
)

#: An action of 1 moves a joint's target a quarter of its part's peak
#: torque's worth of PD error (the Go1 and G1 recipe) -- but the sagittal
#: joints a posture moves (`stand.PostureCommandCfg`: a crouch and a bow
#: take the hip 2 rad from the stand) half a radian.
ACTION_SCALE = {
  j: (0.5 if any(k in j for k in ("hip_pitch", "knee", "ankle_pitch"))
      else 0.25 * JOINTS[j]["peak_torque"] / JOINTS[j]["stiffness"])
  for j in LEGS
}


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(init_state=INIT_STATE, spec_fn=get_spec, articulation=ARTICULATION)
