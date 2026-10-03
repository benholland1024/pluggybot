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
command -- asks the policy for joint targets and hands them to the drivers
with the gains it was trained on (`legs.drivers`, issue #385: MuJoCo's
`dcmotor`s run the PD and the torque-speed envelope on EVERY physics step,
in C, with no Python between two policy steps). The observation reads no
velocity the body could not measure: the policy was trained without the
base's linear velocity.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.legs.arm import ArmDriver
from pluggybot.legs.drivers import Drivers
from pluggybot.legs.model import CHOSEN, JOINT_NAMES
from pluggybot.telemetry.protocol import ROBOT_ROOT

POLICY_NPZ = Path(__file__).resolve().parents[3] / "models" / "quadruped_policy.npz"

#: The perceptive policy's height scan (`training/quad_train/task.py` SCAN):
#: the body's height over the terrain under a grid 1.6 m long and 1.0 m wide
#: at 0.1 m, turned with the heading, the rays cast down from 1 m above the
#: body, a miss reading `SCAN_MAX_M`. Ordered as mjlab's grid: x fastest.
#: On the robot it is the D435's height map sampled at these points; here it
#: is the terrain itself (ray casts), the ideal that map approximates.
SCAN_SIZE_M, SCAN_RES_M, SCAN_RAISE_M, SCAN_MAX_M = (1.6, 1.0), 0.1, 1.0, 5.0


def scan_offsets() -> np.ndarray:
  xs = np.arange(-SCAN_SIZE_M[0] / 2, SCAN_SIZE_M[0] / 2 + SCAN_RES_M / 2, SCAN_RES_M)
  ys = np.arange(-SCAN_SIZE_M[1] / 2, SCAN_SIZE_M[1] / 2 + SCAN_RES_M / 2, SCAN_RES_M)
  gx, gy = np.meshgrid(xs, ys, indexing="xy")
  return np.stack([gx.ravel(), gy.ravel()], axis=1)


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
    #: The observation's terms, in order, and the scale each is multiplied
    #: by (the exported file's metadata; the height scan's is 1/5).
    self.observation_names = tuple(self.meta["observation_names"].split(","))
    self.observation_scales = tuple(
      float(v) for v in self.meta.get("observation_terms_scale", "").split(",") if v
    ) or (1.0,) * len(self.observation_names)
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
  """The command the policy tracks: a velocity in the body's heading frame
  and, for a posture policy, the torso's height offset from the stand, its
  pitch and its roll."""

  vx: float = 0.0
  vy: float = 0.0
  yaw_rate: float = 0.0
  height: float = 0.0
  pitch: float = 0.0
  roll: float = 0.0

  def array(self) -> np.ndarray:
    return np.array([self.vx, self.vy, self.yaw_rate])

  def posture(self) -> np.ndarray:
    return np.array([self.height, self.pitch, self.roll])


class PolicyDriver:
  """The policy on one body: `command(twist)` once per physics step, before
  it; the joint targets change on a policy step and the drivers hold them
  between. A routine may take the drivers in between (`drivers.torque`);
  the policy's next command brings its own gains back."""

  def __init__(self, model, data, policy: WalkingPolicy, prefix: str = "",
               scan=None):
    self.policy, self.prefix = policy, prefix
    self.drivers = Drivers(model, data, prefix)
    self._resolve(model, data)
    self.scan_xy = scan_offsets()
    #: Terrain only: the robot's own geoms (groups 1, 2) never block a ray...
    self.scan_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    #: ...nor a tool riding its fork, a body of its own in group 0: carried
    #: over the nose it read as a 0.4 m obstacle (#378). Its body id, or -1.
    self.scan_exclude = -1
    #: Where the scan comes from: the ideal casts, or what the robot has
    #: (`legs.scan.MapScan.scan`, #388).
    self.scan = scan if scan is not None else self.height_scan
    self.last_action = np.zeros(len(JOINT_NAMES))
    self.target = policy.default_q.copy()
    self.steps = 0
    #: The arm (#378), where the body has one: held where it stands by
    #: `step`, the instruments' path, as the served body holds it (a limp arm
    #: falls across the nose camera). Built at the first `step`: the served
    #: body drives its own (`QuadMission.arm`) and calls `command`, never
    #: `step`.
    self.arm: ArmDriver | None = None
    self._arm_prefix: str | None = prefix      # None once looked for

  def _resolve(self, model, data) -> None:
    prefix = self.prefix
    self.m, self.d = model, data
    self.root = model.body(f"{prefix}{ROBOT_ROOT}").id
    ids = [model.joint(f"{prefix}{n}").id for n in JOINT_NAMES]
    self.qadr = np.array([model.jnt_qposadr[j] for j in ids])
    self.vadr = np.array([model.jnt_dofadr[j] for j in ids])
    self.act = self.drivers.act
    self.gyro = model.sensor(f"{prefix}imu_ang_vel").id
    self.gyro_adr = model.sensor_adr[self.gyro]
    self.every = max(1, round(self.policy.period / model.opt.timestep))

  def rebind(self, model, data) -> None:
    """A recompiled world (issue #407): its drivers and its ids, by name;
    what it is in the middle of -- its last action, its target, its step
    count -- is state and stays. A carried tool's scan exclusion is the
    body's to set again (`QuadMission.carry`)."""
    self.drivers.rebind(model, data)
    self._resolve(model, data)
    if self.arm is not None:
      self.arm.rebind(model, data)

  def observation(self, twist: Twist) -> np.ndarray:
    """The policy's input, term by term in the order its file names them
    (`observation_names`): a walking, a posture and a get-up policy share the
    terms and differ in which they carry."""
    d = self.d
    rot = d.xmat[self.root].reshape(3, 3)
    terms = {
      "base_ang_vel": lambda: d.sensordata[self.gyro_adr:self.gyro_adr + 3],
      "projected_gravity": lambda: rot.T @ np.array([0.0, 0.0, -1.0]),
      "joint_pos": lambda: d.qpos[self.qadr] - self.policy.default_q,
      "joint_vel": lambda: d.qvel[self.vadr],
      "actions": lambda: self.last_action,
      "command": twist.array,
      "posture": twist.posture,
      "height_scan": self.scan,
    }
    return np.concatenate([
      np.asarray(terms[name](), dtype=float) * scale
      for name, scale in zip(self.policy.observation_names,
                             self.policy.observation_scales)])

  def height_scan(self) -> np.ndarray:
    """The body's height over the terrain under each scan point."""
    d = self.d
    body = d.xpos[self.root]
    rot = d.xmat[self.root].reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    xy = self.scan_xy @ np.array([[c, s], [-s, c]])
    top = body[2] + SCAN_RAISE_M
    down = np.array([0.0, 0.0, -1.0])
    geomid = np.zeros(1, dtype=np.int32)
    out = np.full(len(xy), SCAN_MAX_M)
    # One ray per point: parallel rays from separate origins, which
    # `mj_multiRay` (one origin, many directions) cannot cast.
    for i, (x, y) in enumerate(xy):
      dist = mujoco.mj_ray(self.m, d, np.array([body[0] + x, body[1] + y, top]),
                           down, self.scan_groups, 1, self.scan_exclude, geomid)
      if 0 <= dist <= SCAN_MAX_M + SCAN_RAISE_M:
        out[i] = body[2] - (top - dist)
    return out

  def set_bus(self, volts: float) -> None:
    self.drivers.set_bus(volts)

  def command(self, twist: Twist) -> None:
    """Before each physics step: on a policy step, new joint targets, sent
    with the gains the policy was trained on (a GDS68 takes both in every
    command)."""
    decide = self.steps % self.every == 0
    if decide:
      action = self.policy.act(self.observation(twist))
      self.last_action = action
      self.target = self.policy.default_q + self.policy.action_scale * action
    if decide or self.drivers.torqued:
      self.drivers.pd(self.target, self.policy.stiffness, self.policy.damping)
    self.steps += 1

  def step(self, twist: Twist) -> None:
    """One physics step with the policy driving, the arm held."""
    self.command(twist)
    if self._arm_prefix is not None:
      self.arm = _arm_held(self.m, self.d, self._arm_prefix)
      self._arm_prefix = None
    if self.arm is not None:
      self.arm.step()
    mujoco.mj_step(self.m, self.d)


def _arm_held(model, data, prefix: str = "") -> ArmDriver | None:
  """The body's arm held where it stands, or None on a body without one."""
  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}arm_shoulder") < 0:
    return None
  return ArmDriver(model, data, CHOSEN.arm, prefix=prefix)
