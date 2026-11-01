"""The humanoid falls and gets up (issue #507, stage 5): the get-up policy in
numpy (`training/humanoid_train/getup.py`, exported `--joints all`), as the
standing and walking ones are run (`humanoid.policy`), reading only the
robot's senses -- the IMU part's rate and its filter's gravity, every joint
off its driver -- and driving all twenty-four joints, arms too, each with
the gains it trained on.

It takes a FALLING body over: the policy that had the legs (standing or
walking) hands it the legs and the arms once the trunk has tipped
`FALL_TILT_RAD` past the posture those legs hold, by the IMU, or the pelvis
has sunk under `COLLAPSED_M` over the sole it stands on by its own model of
itself (`FallWatch`: shoved hard from the front, the standing policy folded
straight down into a squat, tilted 32 deg, and its hands struck the floor)
-- trained from bodies tipping over at a shove's speed, it lands on its
hands and knees rather than its head -- and it hands them back once the
robot stands (`stood`): upright, its pelvis at the stand's height over that
sole, and still, for `STOOD_S`. ⚠ PAST THE POSTURE, NEVER PAST UPRIGHT
(#520): the IMU is on the pelvis and the body has no waist, so a bow the
standing policy is asked tilts the IMU by the whole bow -- read off upright,
a reach to a low shelf (0.8 rad) or the floor (1.1) was a fall, and the
get-up took the joints. ⚠ THE LOWER SOLE, NEVER BOTH: standing across two
treads, the soles' middle is half a riser under the stand, and the get-up
would never hand back; and a foot lifted on a 0.22 m flight brought the
middle within 0.10 m of a collapse (the lower sole, 0.19).
"""

import math
from pathlib import Path

import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import SelfModel
from pluggybot.humanoid.policy import StandPolicy, TrunkSense
from pluggybot.humanoid.senses import JointSense
from pluggybot.humanoid.drivers import Drivers

GETUP_NPZ = Path(__file__).resolve().parents[3] / "models" / "humanoid_getup.npz"
#: The trunk tipped this far off the posture its legs hold, by the IMU, is
#: a fall the get-up policy takes over: the standing policy stood through
#: shoves that tipped it 15 deg (`humanoid_spike.py --shove`), and by 35 deg
#: it is going over. A bow's trunk follows the posture it is shown within
#: 2.5 deg on its way down and 1.5 on its way back up.
FALL_TILT_RAD = math.radians(35.0)
#: ...or the pelvis this low over the sole it stands on, m: the standing
#: policy's deepest crouch holds it at 0.48 (`policy.HEIGHT_RANGE`).
COLLAPSED_M = 0.40
#: Stood: the trunk within this of upright, the pelvis within `STOOD_M` of
#: the stand's height over the lower sole, the gyro under `STOOD_RATE`,
#: for `STOOD_S` running.
STOOD_TILT_RAD, STOOD_M, STOOD_RATE, STOOD_S = math.radians(10.0), 0.05, 0.3, 0.5


class GetupDriver:
  """The get-up policy on one body: `step()` once a physics step, before it,
  while it has the joints. It may share the trunk's and the joints' senses
  with the other policies' drivers (one IMU, one set of drivers)."""

  def __init__(self, model, data, policy: StandPolicy | None = None, prefix: str = "",
               key: str = "humanoid", trunk: TrunkSense | None = None,
               joints: JointSense | None = None):
    self.policy = policy or StandPolicy(GETUP_NPZ)
    self.m, self.d = model, data
    self.trunk = trunk or TrunkSense(model, data, prefix, key)
    self.joints = joints or JointSense(model, prefix=prefix, key=key)
    self.drivers = Drivers(model, data, prefix, joints=self.policy.action_joints)
    order = {n: k for k, n in enumerate(hm.JOINT_NAMES)}
    self.seen = np.array([order[n] for n in self.policy.joint_names])
    names = {n: k for k, n in enumerate(self.policy.joint_names)}
    self.default = self.policy.default_q[[names[n] for n in self.policy.action_joints]]
    self.every = max(1, round(self.policy.period / model.opt.timestep))
    self.last_action = np.zeros(len(self.policy.action_joints))
    self.target = self.default.copy()
    self.steps = 0
    self.sm = SelfModel()
    self._still = 0.0

  def begin(self) -> None:
    """Take the joints over: the next step is a policy step, from rest."""
    self.last_action[:] = 0.0
    self.steps = 0
    self._still = 0.0

  def observation(self) -> np.ndarray:
    q, qd, _ = self.joints.read(self.d)
    terms = {
      "base_ang_vel": lambda: self.trunk.rate,
      "projected_gravity": self.trunk.gravity,
      "joint_pos": lambda: q[self.seen] - self.policy.default_q,
      "joint_vel": lambda: qd[self.seen],
      "actions": lambda: self.last_action,
    }
    return np.concatenate([np.asarray(terms[name](), dtype=float) * scale
                           for name, scale in zip(self.policy.observation_names,
                                                  self.policy.observation_scales)])

  def step(self, trunk_stepped: bool = False) -> None:
    """Before each physics step: the trunk's senses stepped (unless another
    driver did) and, on a policy step, new targets to every joint with the
    gains the policy trained on."""
    if not trunk_stepped:
      self.trunk.step()
    if self.steps % self.every == 0:
      action = self.policy.act(self.observation())
      self.last_action = action
      self.target = self.default + self.policy.action_scale * action
      self.drivers.pd(self.target, self.policy.stiffness, self.policy.damping)
      self._watch()
    self.steps += 1

  def _watch(self) -> None:
    """How long the body has stood still, s (`stood`)."""
    q, _, _ = self.joints.read(self.d)
    height = over_lower_sole(self.sm, q, self.trunk.gravity())
    upright = self.trunk.attitude.tilt() < STOOD_TILT_RAD
    tall = abs(height - hm.stand_height()) < STOOD_M
    quiet = float(np.linalg.norm(self.trunk.rate)) < STOOD_RATE
    self._still = self._still + self.policy.period if (upright and tall and quiet) else 0.0

  @property
  def stood(self) -> bool:
    """Whether it has stood, still, for `STOOD_S`: the joints may go back."""
    return self._still >= STOOD_S


class FallWatch:
  """Whether the body is falling, from its senses (module docstring): its
  trunk's tilt off the IMU, its pelvis's height over its soles off the
  legs' encoders through its model of itself."""

  def __init__(self, trunk: TrunkSense, joints: JointSense):
    self.trunk, self.joints = trunk, joints
    self.sm = SelfModel()

  def falling(self, data, pitch: float = 0.0) -> bool:
    """`pitch`: the trunk's pitch forward the legs' policy holds, rad -- the
    standing policy's posture (`Reach.held_pitch`), upright for any other."""
    gravity = self.trunk.gravity()
    if tipped(gravity, pitch) > FALL_TILT_RAD:
      return True
    q, _, _ = self.joints.read(data)
    return over_lower_sole(self.sm, q, gravity) < COLLAPSED_M


def tipped(gravity_b: np.ndarray, pitch: float) -> float:
  """How far the trunk has tipped off a posture pitched `pitch` forward,
  rad: the angle between gravity as the IMU reads it in the trunk's frame
  and as that posture would put it there -- a tip about any level axis,
  sideways as much as forward. At a pitch of 0, the IMU's tilt."""
  want = (math.sin(pitch), 0.0, -math.cos(pitch))
  dot = sum(float(g) * w for g, w in zip(gravity_b, want))
  return math.acos(max(-1.0, min(1.0, dot)))


def over_lower_sole(sm: SelfModel, q: np.ndarray, gravity_b: np.ndarray) -> float:
  """The pelvis's height over its lower sole's bottom, m, off its joints and
  its tilt through its model of itself (module docstring)."""
  sm.pose(q, gravity_b)
  d, m = sm.d, sm.m
  return float(d.qpos[2] - min(d.geom_xpos[g][2] - d.geom_xmat[g][8] * m.geom_size[g][2]
                               for g in sm.soles))
