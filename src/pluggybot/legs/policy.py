"""The walking policy as plain numpy (issue #377 item 4).

The served sim may not import a training stack (`deploy/requirements-serve.txt`,
`tests/test_deploy.py`), so the policy trained in `training/` is exported to
an `.npz` of arrays (`quad_train.export`) and run here: a normaliser and an
MLP with ELU, in float64 so a day hashes the same twice.

⚠ The process runs BLAS on ONE thread (`OPENBLAS_NUM_THREADS=1` before numpy
loads): these products are small enough that six threads made a step twice
as slow (238 against 119 us, measured under load). A flight hashes identical
at 1 and at 6 threads (`scripts/quad_spike.py --determinism`), so the pin is
for speed; the float64 forward pass is what keeps two days one day.

`PolicyDriver` is the robot's side of it: every `decimation` physics steps it
builds the observation mjlab trained on -- the gyro, gravity in the body
frame, joint positions and speeds off the encoders, the last action and the
command -- asks the policy for joint targets, and on EVERY physics step runs
the driver's PD toward them through the actuator's envelope
(`actuator.JointLimits`). The observation reads no velocity the body could
not measure: the policy was trained without the base's linear velocity.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.legs.actuator import JointLimits
from pluggybot.legs.model import JOINT_NAMES
from pluggybot.telemetry.protocol import ROBOT_ROOT

POLICY_NPZ = Path(__file__).resolve().parents[3] / "models" / "quadruped_policy.npz"


def _elu(x: np.ndarray) -> np.ndarray:
  return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class WalkingPolicy:
  """The exported actor: `act(obs)` is the deterministic action (the
  Gaussian's mean), one array in, one array out."""

  def __init__(self, path: str | Path = POLICY_NPZ):
    raw = Path(path).read_bytes()
    #: What the build identity carries: the weights' own hash.
    self.sha256 = hashlib.sha256(raw).hexdigest()
    with np.load(path) as z:
      self.mean = z["obs_mean"].astype(np.float64)
      self.div = z["obs_div"].astype(np.float64)
      n = int(z["layers"])
      self.layers = [(z[f"w{k}"].astype(np.float64), z[f"b{k}"].astype(np.float64))
                     for k in range(n)]
      self.meta = json.loads(str(z["meta"]))
      self.training = json.loads(str(z["training"]))
    names = self.meta["joint_names"].split(",")
    assert tuple(names) == JOINT_NAMES, names
    self.default_q = np.array([float(v) for v in self.meta["default_joint_pos"].split(",")])
    self.action_scale = np.array([float(v) for v in self.meta["action_scale"].split(",")])
    self.stiffness = float(self.training["stiffness"])
    self.damping = float(self.training["damping"])
    #: Seconds between two policy steps (the trainer's physics step x its
    #: decimation).
    self.period = self.training["train_dt"] * self.training["decimation"]

  @property
  def obs_size(self) -> int:
    return self.mean.shape[0]

  def act(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean) / self.div
    for w, b in self.layers[:-1]:
      x = _elu(x @ w + b)
    w, b = self.layers[-1]
    return x @ w + b


@dataclass
class Twist:
  """The command the policy tracks, in the body's heading frame."""

  vx: float = 0.0
  vy: float = 0.0
  yaw_rate: float = 0.0

  def array(self) -> np.ndarray:
    return np.array([self.vx, self.vy, self.yaw_rate])


class PolicyDriver:
  """The policy on one body: `torque(twist)` once per physics step."""

  def __init__(self, model, data, policy: WalkingPolicy, limits: JointLimits,
               prefix: str = ""):
    self.m, self.d, self.policy, self.limits = model, data, policy, limits
    self.root = model.body(f"{prefix}{ROBOT_ROOT}").id
    ids = [model.joint(f"{prefix}{n}").id for n in JOINT_NAMES]
    self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
    self.vadr = np.array([model.jnt_dofadr[j] for j in ids])
    self.gyro = model.sensor(f"{prefix}imu_ang_vel").id
    self.gyro_adr = model.sensor_adr[self.gyro]
    self.every = max(1, round(policy.period / model.opt.timestep))
    self.last_action = np.zeros(len(JOINT_NAMES))
    self.target = policy.default_q.copy()
    self.steps = 0

  def observation(self, twist: Twist) -> np.ndarray:
    d = self.d
    rot = d.xmat[self.root].reshape(3, 3)
    gravity = rot.T @ np.array([0.0, 0.0, -1.0])
    return np.concatenate([
      d.sensordata[self.gyro_adr:self.gyro_adr + 3],
      gravity,
      d.qpos[self.qadr] - self.policy.default_q,
      d.qvel[self.vadr],
      self.last_action,
      twist.array(),
    ])

  def torque(self, twist: Twist) -> np.ndarray:
    if self.steps % self.every == 0:
      action = self.policy.act(self.observation(twist))
      self.last_action = action
      self.target = self.policy.default_q + self.policy.action_scale * action
    self.steps += 1
    q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
    tau = self.policy.stiffness * (self.target - q) - self.policy.damping * qd
    return self.limits.clip(tau, qd)

  def step(self, twist: Twist) -> None:
    """One physics step with the policy driving (the standalone model's
    actuators are the twelve leg motors, in joint order)."""
    self.d.ctrl[:12] = self.torque(twist)
    mujoco.mj_step(self.m, self.d)
