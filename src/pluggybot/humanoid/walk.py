"""The humanoid's walking policy in numpy (issue #507, stage 4), as the
standing one is run (`humanoid.policy`): an `.npz` the trainer exported and
checked against its ONNX file, reading only the robot's senses -- the IMU
part's rate and its filter's gravity, every joint off its driver -- and
driving the legs' twelve drivers with the gains it trained on. Its command
is the TWIST: forward and sideways speed in the trunk's heading frame, m/s,
and the turn, rad/s (`training/humanoid_train/walk.py`).
"""

from pathlib import Path

import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid.policy import StandPolicy, TrunkSense
from pluggybot.humanoid.senses import JointSense
from pluggybot.legs.drivers import Drivers

WALK_NPZ = Path(__file__).resolve().parents[3] / "models" / "humanoid_walk.npz"
#: The twist it trained on, at its widest (`walk.LIN_X`, `LIN_Y`, `ANG_Z`):
#: a command is held inside it.
LIN_X, LIN_Y, ANG_Z = (-0.8, 1.4), (-0.4, 0.4), (-1.0, 1.0)


class WalkDriver:
  """The walking policy on one body: `step()` once a physics step, before
  it, after `ask(vx, vy, wz)`. It may share the trunk's and the joints'
  senses with the standing policy's driver (one IMU, one set of drivers)."""

  def __init__(self, model, data, policy: StandPolicy | None = None, prefix: str = "",
               key: str = "humanoid", trunk: TrunkSense | None = None,
               joints: JointSense | None = None):
    self.policy = policy or StandPolicy(WALK_NPZ)
    self.m, self.d = model, data
    self.trunk = trunk or TrunkSense(model, data, prefix, key)
    self.joints = joints or JointSense(model, prefix=prefix, key=key)
    self.legs = Drivers(model, data, prefix, joints=self.policy.action_joints)
    order = {n: k for k, n in enumerate(hm.JOINT_NAMES)}
    self.seen = np.array([order[n] for n in self.policy.joint_names])
    legs = {n: k for k, n in enumerate(self.policy.joint_names)}
    self.leg_default = self.policy.default_q[[legs[n] for n in self.policy.action_joints]]
    self.every = max(1, round(self.policy.period / model.opt.timestep))
    self.last_action = np.zeros(len(self.policy.action_joints))
    self.twist = np.zeros(3)
    self.target = self.leg_default.copy()
    self.steps = 0

  def ask(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
    """The twist to track, held inside what it trained on."""
    self.twist = np.array([min(max(vx, LIN_X[0]), LIN_X[1]),
                           min(max(vy, LIN_Y[0]), LIN_Y[1]),
                           min(max(wz, ANG_Z[0]), ANG_Z[1])])

  def observation(self) -> np.ndarray:
    q, qd, _ = self.joints.read(self.d)
    terms = {
      "base_ang_vel": lambda: self.trunk.rate,
      "projected_gravity": self.trunk.gravity,
      "joint_pos": lambda: q[self.seen] - self.policy.default_q,
      "joint_vel": lambda: qd[self.seen],
      "actions": lambda: self.last_action,
      "command": lambda: self.twist,
    }
    return np.concatenate([np.asarray(terms[name](), dtype=float)
                           for name in self.policy.observation_names])

  def step(self, trunk_stepped: bool = False) -> None:
    """Before each physics step: the trunk's senses stepped (unless the
    standing driver beside it did) and, on a policy step, new targets to
    the legs with the gains it trained on."""
    if not trunk_stepped:
      self.trunk.step()
    if self.steps % self.every == 0:
      action = self.policy.act(self.observation())
      self.last_action = action
      self.target = self.leg_default + self.policy.action_scale * action
      self.legs.pd(self.target, self.policy.stiffness, self.policy.damping)
    self.steps += 1
