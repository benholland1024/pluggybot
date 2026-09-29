"""The quadruped as an mjlab entity (#377), read off the files pluggybot
generates: `models/quadruped.xml` and `models/quadruped.json`
(`uv run python -m pluggybot.legs.model`). Nothing is typed twice."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "models" / "quadruped.xml"
BODY = json.loads((REPO / "models" / "quadruped.json").read_text())
MOTOR = BODY["motor"]
ROOT = BODY["root"]
FEET = tuple(f"{leg}_foot" for leg in BODY["legs"])

#: The physics step training runs at, s (mjlab's velocity task); the served
#: sim runs 0.002, and `scripts/quad_policy.py` flies the policy there.
TRAIN_DT = 0.005

#: The driver's PD gains (`pluggybot.legs.actuator.driver_gains`: a 10 Hz
#: loop at damping ratio 2 on the reflected inertia), which the served
#: model's dcmotors run too.
STIFFNESS = BODY["driver"]["stiffness"]
DAMPING = BODY["driver"]["damping"]


def get_spec() -> mujoco.MjSpec:
  """The robot alone: the standalone file's floor, light, keyframes and
  torque motors go (mjlab brings a terrain and builds its own actuators),
  and the arm is fixed at its stow (`freeze_arm`)."""
  spec = mujoco.MjSpec.from_file(str(XML))
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for key in list(spec.keys):
    spec.delete(key)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  freeze_arm(spec)
  return spec


def freeze_arm(spec: mujoco.MjSpec) -> None:
  """The arm (#378) as the served body holds it while it walks and gets up
  (#405): each joint's body turned to the stow its driver holds and the
  joint taken out, with the linkages that bound them. A policy's joints
  are the legs' twelve, and the arm's geometry is there to fall on --
  #377's placeholder collided with nothing. `quadruped.json`'s `arm` says
  which joints, the stow, and their axis."""
  arm = BODY.get("arm")
  if not arm:
    return
  held = dict(zip(arm["joints"], arm["stow_qpos"]))
  axis = np.array(arm["axis"], dtype=float)
  for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
    for joint in list(body.joints):
      if joint.name not in held:
        continue
      q = held.pop(joint.name)
      turn = np.array([math.cos(q / 2), *(math.sin(q / 2) * axis)])
      quat = np.zeros(4)
      mujoco.mju_mulQuat(quat, np.array(body.quat, dtype=float), turn)
      body.quat = quat
      spec.delete(joint)
  if held:
    raise ValueError(f"the arm's joints {sorted(held)} are not in {XML.name}")
  for tendon in list(spec.tendons):
    if tendon.name in arm["tendons"]:
      spec.delete(tendon)
  for eq in list(spec.equalities):
    if eq.name in arm["equalities"]:
      spec.delete(eq)


LEG_ACTUATOR = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_abd", ".*_hip_flex", ".*_knee"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  # The driver's current clip is the hard limit; the continuous rating is a
  # thermal one, checked afterwards against the flown torques.
  effort_limit=MOTOR["peak_torque"],
  saturation_effort=MOTOR["saturation_torque"],
  velocity_limit=MOTOR["noload_speed"],
  armature=MOTOR["armature"],
  frictionloss=BODY["friction_nominal"],
  delay_min_lag=0,
  delay_max_lag=round(BODY["latency_s"][1] / TRAIN_DT),
)

_q = dict(zip(BODY["joints"], BODY["stand_qpos"]))
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, BODY["stand_height"] + 0.01),
  joint_pos={
    ".*_hip_abd": _q["FL_hip_abd"],
    ".*_hip_flex": _q["FL_hip_flex"],
    ".*_knee": _q["FL_knee"],
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(LEG_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)

#: An action of 1 moves a joint's target a quarter of the peak torque's
#: worth of PD error (the Go1 recipe).
ACTION_SCALE = {n: 0.25 * MOTOR["peak_torque"] / STIFFNESS
                for n in LEG_ACTUATOR.target_names_expr}


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(init_state=INIT_STATE, spec_fn=get_spec,
                   articulation=ARTICULATION)
