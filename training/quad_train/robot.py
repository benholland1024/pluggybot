"""The quadruped as an mjlab entity (#377), read off the files pluggybot
generates: `models/quadruped.xml` and `models/quadruped.json`
(`uv run python -m pluggybot.legs.model`). Nothing is typed twice."""

import json
from pathlib import Path

import mujoco
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
  torque motors go (mjlab brings a terrain and builds its own actuators)."""
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
  return spec


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
