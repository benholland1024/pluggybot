"""The humanoid's joints and grippers as their drivers report them (issue
#507): the sim's angles are exact, a robot's are whole counts of each
driver's own reply fields with the gearbox's backlash under them, and its
torque is read off the phase current, whose sense is noisy
(`humanoid.parts.Fields`, counted by `perception/encoders.py`).
The IMU is `perception/imu.py`'s part and the depth camera and the LIDAR
`perception/`'s, each on the head (`humanoid.model`).

The noise is one stream a robot, as the IMU's is, keyed by a checksum of
its name -- never `hash()`, which Python salts per process -- and a part
added to the body since draws from one of its own (`SHARED`).
"""

import zlib

import numpy as np

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.perception.encoders import quantised

#: The drivers whose noise is one block of the robot's stream, a read at a
#: time, in this order: the legs' and the arms', the body before its waist
#: (#560's stage (c3)). A joint added since draws from a stream of its own,
#: keyed by its name, so adding a part re-deals no other part's noise. ⚠
#: DEALT OVER EVERY JOINT IN ORDER, THE WAIST RE-DREW EVERY OTHER JOINT'S
#: NOISE -- and every set-out of a check replays the robot's noise at the
#: same instants, so with it the one realization the whole check flies:
#: today's body placed the cube 22 times of 32 as served and 9 dealt the
#: waist's way, 16-21 at three other seeds (SimNotes, "The cube and the
#: robot's noise").
SHARED = (tuple(f"{s}_{j}" for s in hm.SIDES for j in hm.LEG)
          + tuple(f"{s}_{j}" for s in hm.SIDES for j in hm.ARM))


class JointSense:
  """Every joint's position, speed and torque, as its driver reports them,
  in the order the body's drawing carries them (`model.joint_names`: the
  served body's are `model.JOINT_NAMES`)."""

  def __init__(self, model, spec: hm.HumanoidSpec = hm.CHOSEN, prefix: str = "",
               key: str = "humanoid"):
    self.names = names = hm.joint_names(spec)
    #: Where the shared block's joints sit in this body's order.
    self.shared = np.array([names.index(j) for j in SHARED if j in names], dtype=int)
    self.reseed(key)
    motors = [spec.motor(j) for j in names]
    fields = [hp.FIELDS[m.name] for m in motors]
    self.pos_lsb = np.array([f.position_lsb for f in fields])
    self.vel_lsb = np.array([f.velocity_lsb for f in fields])
    self.tau_lsb = np.array([f.torque_lsb for f in fields])
    self.tau_top = np.array([f.torque_range for f in fields])
    #: The torque's noise, N*m: the current sense's times the output Kt.
    self.tau_noise = np.array([f.current_noise_a * m.kt for f, m in zip(fields, motors)])
    self.backlash = np.array([hp.backlash_rad(m) for m in motors])
    self.qadr = np.array([model.jnt_qposadr[model.joint(f"{prefix}{j}").id] for j in names])
    self.vadr = np.array([model.jnt_dofadr[model.joint(f"{prefix}{j}").id] for j in names])
    self.act = np.array([model.actuator(f"{prefix}{j}").id for j in names])

  def reseed(self, key: str) -> None:
    """Every stream drawn afresh as a robot keyed `key` draws them: the
    shared block's (`rng`), and each other joint's own (`own`, the
    waist's)."""
    self.rng = np.random.default_rng(zlib.crc32(f"joints:{key}".encode()))
    self.own = [(i, np.random.default_rng(zlib.crc32(f"joints:{key}:{j}".encode())))
                for i, j in enumerate(self.names) if j not in SHARED]

  def read(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position rad, speed rad/s, torque N*m) as the drivers report them:
    the output shaft known to within its backlash, each value in its
    field's whole counts, and the torque saturating at its field's last
    count."""
    n, k = len(self.qadr), len(self.shared)
    slack, current = np.empty(n), np.empty(n)
    slack[self.shared] = self.rng.uniform(-0.5, 0.5, k)
    current[self.shared] = self.rng.normal(0.0, 1.0, k)
    for i, rng in self.own:
      slack[i], current[i] = rng.uniform(-0.5, 0.5), rng.normal(0.0, 1.0)
    q = data.qpos[self.qadr] + slack * self.backlash
    tau = data.actuator_force[self.act] + current * self.tau_noise
    tau = np.clip(quantised(tau, self.tau_lsb), -self.tau_top, self.tau_top)
    return (quantised(q, self.pos_lsb), quantised(data.qvel[self.vadr], self.vel_lsb), tau)

  def kept_state(self) -> list:
    """Every stream's state, the shared block's first: a restart's (#345)."""
    return [self.rng.bit_generator.state] + [r.bit_generator.state for _, r in self.own]

  def restore_kept(self, states) -> None:
    """`kept_state` put back -- or the shared block's alone, a save from
    before the waist's own stream, its own begun afresh."""
    states = states if isinstance(states, list) else [states]
    self.rng.bit_generator.state = states[0]
    for (_, r), s in zip(self.own, states[1:]):
      r.bit_generator.state = s


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
