"""The humanoid's joints and grippers as their drivers report them (issue
#507): the sim's angles are exact, a robot's are whole counts of each
driver's own reply fields with the gearbox's backlash under them, and its
torque is read off the phase current, whose sense is noisy
(`humanoid.parts.Fields`; the quadruped's are `perception/encoders.py`).
The IMU is `perception/imu.py`'s part and the depth camera and the LIDAR
`perception/`'s, each on the head (`humanoid.model`).

The noise is one stream a robot, as the IMU's is, keyed by a checksum of
its name -- never `hash()`, which Python salts per process.
"""

import zlib

import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.perception.encoders import quantised


class JointSense:
  """Every joint's position, speed and torque, as its driver reports them,
  in `model.JOINT_NAMES` order."""

  def __init__(self, model, spec: hm.HumanoidSpec = hm.CHOSEN, prefix: str = "",
               key: str = "humanoid"):
    self.rng = np.random.default_rng(zlib.crc32(f"joints:{key}".encode()))
    motors = [spec.motor(j) for j in hm.JOINT_NAMES]
    fields = [hp.FIELDS[m.name] for m in motors]
    self.pos_lsb = np.array([f.position_lsb for f in fields])
    self.vel_lsb = np.array([f.velocity_lsb for f in fields])
    self.tau_lsb = np.array([f.torque_lsb for f in fields])
    self.tau_top = np.array([f.torque_range for f in fields])
    #: The torque's noise, N*m: the current sense's times the output Kt.
    self.tau_noise = np.array([f.current_noise_a * m.kt for f, m in zip(fields, motors)])
    self.backlash = np.array([hp.backlash_rad(m) for m in motors])
    self.qadr = np.array([model.jnt_qposadr[model.joint(f"{prefix}{j}").id]
                          for j in hm.JOINT_NAMES])
    self.vadr = np.array([model.jnt_dofadr[model.joint(f"{prefix}{j}").id]
                          for j in hm.JOINT_NAMES])
    self.act = np.array([model.actuator(f"{prefix}{j}").id for j in hm.JOINT_NAMES])

  def read(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position rad, speed rad/s, torque N*m) as the drivers report them:
    the output shaft known to within its backlash, each value in its
    field's whole counts, and the torque saturating at its field's last
    count."""
    n = len(self.qadr)
    q = data.qpos[self.qadr] + self.rng.uniform(-0.5, 0.5, n) * self.backlash
    tau = data.actuator_force[self.act] + self.rng.normal(0.0, 1.0, n) * self.tau_noise
    tau = np.clip(quantised(tau, self.tau_lsb), -self.tau_top, self.tau_top)
    return (quantised(q, self.pos_lsb), quantised(data.qvel[self.vadr], self.vel_lsb), tau)


class GripperSense:
  """A gripper as its own driver reports it (`parts.GRIPPER`): the opening
  between its jaws off its motor's encoder, in the fingertip's published
  resolution, and the squeeze off its torque estimate -- never what the jaws
  touch, which a hand's sensors cannot tell (#493: the quadruped's `held()`
  read the world)."""

  def __init__(self, model, side: str, prefix: str = "", key: str = "humanoid"):
    self.rng = np.random.default_rng(zlib.crc32(f"gripper:{key}:{side}".encode()))
    jaw = model.joint(f"{prefix}{side}_{hm.JAWS[0]}").id
    self.qadr = int(model.jnt_qposadr[jaw])
    self.act = int(model.actuator(f"{prefix}{side}_gripper").id)

  def read(self, data) -> tuple[float, float]:
    """(opening m, squeeze N): the jaws' gap and the force each pad pushes
    with, as the gripper reports them."""
    g = hp.GRIPPER
    gap = float(quantised(2 * data.qpos[self.qadr], g.position_lsb))
    # The actuated jaw's force is both pads' squeeze (`parts.Gripper`).
    squeeze = -float(data.actuator_force[self.act]) / 2
    squeeze += float(self.rng.normal(0.0, g.force_noise_n))
    return gap, squeeze
