"""The humanoid's standing policy in numpy (issue #507, stage 2), as
`legs.policy` runs the quadruped's: an `.npz` the trainer exported and
checked against its ONNX file (`training/quad_train/export.py --body
humanoid`), a normaliser and an MLP with ELU in float64.

What it reads is the robot's, never the simulator's: the rate off the IMU
part (`perception.imu.Imu`, its noise, offset and scale), gravity in the
trunk's frame off that part's attitude filter (`imu.Attitude`), and every
joint off its driver's reply (`humanoid.senses.JointSense`: counts, the
gearbox's backlash). What it commands is the legs' twelve drivers, each a
target with the gains the policy was trained on (`legs.drivers`); the arms
are the reach's (`humanoid.arms`), and the policy sees their encoders.

The command is the posture: the pelvis's height off the stand's and the
trunk's pitch forward (`training/humanoid_train/stand.py`), moved toward a
new one no faster than training moved it (`StandDriver.ask`).
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid.senses import JointSense
from pluggybot.legs.drivers import Drivers
from pluggybot.legs.policy import _elu
from pluggybot.perception.imu import Attitude, Imu

STAND_NPZ = Path(__file__).resolve().parents[3] / "models" / "humanoid_stand.npz"
#: The posture's fastest change, m/s and rad/s: training's
#: (`stand.HEIGHT_RATE`, `PITCH_RATE`), which the policy was shown.
HEIGHT_RATE, PITCH_RATE = 0.15, 0.6
#: ...and the postures it was trained over: the height off the stand's, m,
#: and the pitch, rad (`stand.HEIGHT`, `stand.PITCH`).
HEIGHT_RANGE, PITCH_RANGE = (-0.25, 0.0), (-0.1, 1.1)
#: The posture shown is the one asked within this, m and rad.
SETTLED = np.array([0.005, 0.02])


class StandPolicy:
  """The exported actor: `act(obs)` is its deterministic action."""

  def __init__(self, path: str | Path = STAND_NPZ):
    raw = Path(path).read_bytes()
    #: The weights' own hash, for a build's identity.
    self.sha256 = hashlib.sha256(raw).hexdigest()
    with np.load(path) as z:
      self.mean = z["obs_mean"].astype(np.float64)
      self.div = z["obs_div"].astype(np.float64)
      n = int(z["layers"])
      self.layers = [(z[f"w{k}"].astype(np.float64), z[f"b{k}"].astype(np.float64))
                     for k in range(n)]
      self.meta = json.loads(str(z["meta"]))
      self.training = json.loads(str(z["training"]))
    #: Every joint the policy sees, in its order, and where each stands
    #: when its observation reads zero.
    self.joint_names = tuple(self.meta["joint_names"].split(","))
    assert set(self.joint_names) == set(hm.JOINT_NAMES), self.joint_names
    self.default_q = np.array([float(v) for v in self.meta["default_joint_pos"].split(",")])
    #: The joints it drives (the legs), and an action's scale on each.
    self.action_joints = tuple(self.training["action_joints"])
    self.action_scale = np.array([float(v) for v in self.meta["action_scale"].split(",")])
    self.observation_names = tuple(self.meta["observation_names"].split(","))
    #: Each term's scale as training applied it (the height scan's 1/5).
    self.observation_scales = tuple(
      float(v) for v in self.meta.get("observation_terms_scale", "").split(",") if v
    ) or (1.0,) * len(self.observation_names)
    self.stiffness = np.array(self.training["stiffness"], dtype=float)
    self.damping = np.array(self.training["damping"], dtype=float)
    #: Seconds between two policy steps.
    self.period = self.training["train_dt"] * self.training["decimation"]

  def act(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean) / self.div
    for w, b in self.layers[:-1]:
      x = _elu(x @ w + b)
    w, b = self.layers[-1]
    return x @ w + b


class TrunkSense:
  """The IMU on the pelvis, as the robot reads it: its rate and its tilt,
  stepped every physics step (the filter integrates the gyro). Started from
  the orientation the body boots in, as the quadruped's is."""

  def __init__(self, model, data, prefix: str = "", key: str = "humanoid"):
    self.m, self.d = model, data
    self.imu = Imu(f"{key}:pelvis")
    gyro, accel = model.sensor(f"{prefix}imu_ang_vel"), model.sensor(f"{prefix}imu_lin_acc")
    self.gyro_adr, self.accel_adr = int(gyro.adr[0]), int(accel.adr[0])
    root = model.body(f"{prefix}{hm.ROBOT_ROOT}").id
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[root])
    self.attitude = Attitude(quat)
    self.rate = np.zeros(3)

  def step(self) -> None:
    dt = self.m.opt.timestep
    d = self.d
    self.rate = self.imu.gyro(d.sensordata[self.gyro_adr:self.gyro_adr + 3], dt)
    accel = self.imu.accel(d.sensordata[self.accel_adr:self.accel_adr + 3], dt)
    self.attitude.step(self.rate, accel, dt)

  def gravity(self) -> np.ndarray:
    """Gravity's direction in the trunk's frame, as the filter has it."""
    return self.attitude.matrix().T @ np.array([0.0, 0.0, -1.0])


class StandDriver:
  """The standing policy on one body: `step()` once a physics step, before
  it. The legs' targets change on a policy step and their drivers hold
  them between; `ask(height, pitch)` sets the posture it walks toward."""

  def __init__(self, model, data, policy: StandPolicy | None = None, prefix: str = "",
               key: str = "humanoid", trunk: TrunkSense | None = None,
               joints: JointSense | None = None):
    self.policy = policy or StandPolicy()
    self.m, self.d = model, data
    self.trunk = trunk or TrunkSense(model, data, prefix, key)
    self.joints = joints or JointSense(model, prefix=prefix, key=key)
    self.legs = Drivers(model, data, prefix, joints=self.policy.action_joints)
    # The policy's joint order against the senses' (`hm.JOINT_NAMES`).
    order = {n: k for k, n in enumerate(hm.JOINT_NAMES)}
    self.seen = np.array([order[n] for n in self.policy.joint_names])
    legs = {n: k for k, n in enumerate(self.policy.joint_names)}
    self.leg_default = self.policy.default_q[[legs[n] for n in self.policy.action_joints]]
    self.every = max(1, round(self.policy.period / model.opt.timestep))
    self.last_action = np.zeros(len(self.policy.action_joints))
    self.goal = np.zeros(2)
    self.posture = np.zeros(2)
    self.target = self.leg_default.copy()
    self.steps = 0

  def ask(self, height: float = 0.0, pitch: float = 0.0) -> None:
    """The posture to walk toward: the pelvis `height` m off the stand's and
    the trunk pitched `pitch` rad forward, held inside what it trained on."""
    self.goal = np.array([min(max(height, HEIGHT_RANGE[0]), HEIGHT_RANGE[1]),
                          min(max(pitch, PITCH_RANGE[0]), PITCH_RANGE[1])])

  @property
  def settled(self) -> bool:
    """Whether the posture it is shown has reached the one asked, to
    `SETTLED`: a trim of a few millimetres is not a new posture."""
    return bool((np.abs(self.posture - self.goal) <= SETTLED).all())

  def observation(self) -> np.ndarray:
    q, qd, _ = self.joints.read(self.d)
    terms = {
      "base_ang_vel": lambda: self.trunk.rate,
      "projected_gravity": self.trunk.gravity,
      "joint_pos": lambda: q[self.seen] - self.policy.default_q,
      "joint_vel": lambda: qd[self.seen],
      "actions": lambda: self.last_action,
      "posture": lambda: self.posture,
    }
    return np.concatenate([np.asarray(terms[name](), dtype=float)
                           for name in self.policy.observation_names])

  def step(self) -> None:
    """Before each physics step: the trunk's senses stepped and, on a policy
    step, the posture moved on and new targets sent to the legs with the
    gains the policy was trained on."""
    self.trunk.step()
    if self.steps % self.every == 0:
      rate = np.array([HEIGHT_RATE, PITCH_RATE]) * self.policy.period
      self.posture = self.posture + np.clip(self.goal - self.posture, -rate, rate)
      action = self.policy.act(self.observation())
      self.last_action = action
      self.target = self.leg_default + self.policy.action_scale * action
      self.legs.pd(self.target, self.policy.stiffness, self.policy.damping)
    self.steps += 1
