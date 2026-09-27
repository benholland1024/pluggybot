"""A scripted gait for SIZING the body (issue #377): torques before learning.

A virtual-model controller of the kind MIT's Cheetah papers start from:
stance feet push (tau = -J^T f) so the torso holds its height, attitude and
commanded velocity; swing feet follow a Raibert foot placement under a
Cartesian PD (tau = J^T F). It is a measuring instrument for the torque,
thermal and energy tables, never the robot's gait: the walking policy is
learned (#377 item 3), and its torques are measured again when it lands.
The instrument can under-read a policy that stamps harder, which is why the
table states margins rather than bare pass/fail.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from pluggybot.legs.model import JOINT_NAMES, LEGS, BodySpec

G = 9.81
#: A foothold keeps this far from a height change (the foot is 22 mm).
EDGE_CLEAR_M = 0.05
#: The body the gains were tuned on, kg.
REFERENCE_KG = 9.34
#: Time constant of the height and pitch references over terrain, s.
REF_TAU_S = 0.25

#: Leg phase offsets and duty factor per gait. A trot moves diagonal pairs;
#: a walk one leg at a time, three feet down (a lateral sequence).
GAITS = {
  "stand": ({leg: 0.0 for leg in LEGS}, 1.0),
  "walk": ({"HL": 0.0, "FL": 0.25, "HR": 0.5, "FR": 0.75}, 0.75),
  "trot": ({"FL": 0.0, "HR": 0.0, "FR": 0.5, "HL": 0.5}, 0.5),
}


@dataclass
class Command:
  vx: float = 0.0
  vy: float = 0.0
  yaw_rate: float = 0.0
  height: float | None = None
  pitch: float = 0.0
  gait: str = "stand"
  period: float = 0.4
  swing_height: float = 0.08


def _quat_rpy(q) -> tuple[float, float, float]:
  w, x, y, z = q
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return roll, pitch, yaw


class VirtualModel:
  """Torques for all twelve joints from a `Command`, one physics step at a
  time. `root` is the robot's free body; `prefix` names WHICH robot (a
  second one's elements are `r2_...`, `robot.RobotHandle`)."""

  def __init__(self, model, data, spec: BodySpec, root: str = "pluggybot",
               terrain=None, prefix: str = ""):
    self.m, self.d, self.spec = model, data, spec
    #: (x, y) -> ground height. The scripted gait is TOLD the terrain (a
    #: blind policy is not); it is how the table measures a climb's torques.
    self.terrain = terrain or (lambda x, y: 0.0)
    self.root = model.body(prefix + root).id
    self.mass = float(model.body_subtreemass[self.root])
    self.qadr = np.array([model.jnt_qposadr[model.joint(prefix + j).id]
                          for j in JOINT_NAMES])
    self.vadr = np.array([model.jnt_dofadr[model.joint(prefix + j).id]
                          for j in JOINT_NAMES])
    self.sites = [model.site(f"{prefix}{leg}_foot").id for leg in LEGS]
    self.hips = [model.body(f"{prefix}{leg}_thigh").id for leg in LEGS]
    self.jac = np.zeros((3, model.nv))
    #: The swing PD was tuned on the 9.3 kg body; a lighter one gets gains
    #: in proportion, or its swing saturates small actuators.
    self.swing_gain = self.mass / REFERENCE_KG
    self.reset()
    # The whole robot's rotational inertia about its root, off the mass
    # matrix's free-joint block: enough for an attitude PD's feed-forward.
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    fa = model.body_dofadr[self.root]
    self.inertia = np.diag(full[fa + 3:fa + 6, fa + 3:fa + 6]).copy()
    # A fresh MjData's mass matrix is all zeros until a forward pass: built
    # on one, the attitude loop ran without its feed-forward and a lie-down
    # moved the body 44.6 mm instead of 34.8 (#378).
    if not np.all(self.inertia > 0):
      raise ValueError("forward the data (mj_forward) before building the gait")

  def reset(self) -> None:
    """Re-anchor on the body as it stands now: where to hold, which way to
    face, the gait's clock and the ground under the feet. Called whenever
    the gait takes over from something else (a joint-space fold, a fall)."""
    d, spec = self.d, self.spec
    self.t0 = float(d.time)
    self.liftoff = {leg: None for leg in LEGS}
    self.stance_prev = {leg: True for leg in LEGS}
    self.x_des = None
    self.yaw_des = None
    self.ground = [self.terrain(*d.site_xpos[sid][:2]) for sid in self.sites]
    self.ref_ground = sum(self.ground) / 4
    self.ref_slope = math.atan2(
      (self.ground[0] + self.ground[1] - self.ground[2] - self.ground[3]) / 2,
      2 * spec.hip_x)

  def _foothold(self, x: float, y: float) -> tuple[float, float]:
    """Move a foothold off an edge: the nearest x whose +-EDGE_CLEAR_M is
    one flat height (a perceptive controller's rule; a blind one lands on
    the corner and the table would measure the stumble)."""
    # Forward first: a foot kept short of an edge leaves the body walking
    # on while the hind legs trail, and it tips onto the step.
    for shift in (0.0, 0.03, 0.06, 0.09, 0.12, -0.03, -0.06, -0.09):
      xs = x + shift
      hs = {self.terrain(xs + dx, y) for dx in (-EDGE_CLEAR_M, 0, EDGE_CLEAR_M)}
      if len(hs) == 1:
        return xs, y
    return x, y

  def _state(self):
    d = self.d
    q = d.xquat[self.root]
    com = d.subtree_com[self.root].copy()
    rot = d.xmat[self.root].reshape(3, 3)
    fa = self.m.body_dofadr[self.root]
    v = d.qvel[fa:fa + 3].copy()
    w_body = d.qvel[fa + 3:fa + 6].copy()
    return com, rot, v, rot @ w_body, _quat_rpy(q)

  def torque(self, cmd: Command) -> np.ndarray:
    m, d, spec = self.m, self.d, self.spec
    com, rot, v, w, (roll, pitch, yaw) = self._state()
    offsets, duty = GAITS[cmd.gait]
    t = float(d.time) - self.t0
    height = cmd.height if cmd.height is not None else spec.stand_height
    # Height and pitch are referenced to the ground the FEET stand on, not
    # the ground under the hips: that jumps a whole riser in one step as a
    # hip crosses an edge, and the attitude loop flips the body backwards.
    # Low-passed, so a foot landing on a step raises the body over a stride.
    for i, leg in enumerate(LEGS):
      if self.stance_prev[leg]:
        p = d.site_xpos[self.sites[i]]
        self.ground[i] = self.terrain(p[0], p[1])
    k = min(1.0, m.opt.timestep / REF_TAU_S)
    self.ref_ground += k * (sum(self.ground) / 4 - self.ref_ground)
    rise = (self.ground[0] + self.ground[1] - self.ground[2] - self.ground[3]) / 2
    self.ref_slope += k * (math.atan2(rise, 2 * spec.hip_x) - self.ref_slope)
    height += self.ref_ground
    slope = self.ref_slope
    if self.yaw_des is None:
      self.yaw_des, self.x_des = yaw, com[:2].copy()
    dt = m.opt.timestep
    self.yaw_des += cmd.yaw_rate * dt
    cy, sy = math.cos(yaw), math.sin(yaw)
    v_des = np.array([cy * cmd.vx - sy * cmd.vy, sy * cmd.vx + cy * cmd.vy, 0])
    self.x_des = self.x_des + v_des[:2] * dt
    stance = {}
    for leg in LEGS:
      ph = ((t / cmd.period) + offsets[leg]) % 1.0
      stance[leg] = (cmd.gait == "stand") or ph < duty
      if not stance[leg] and self.stance_prev[leg]:
        self.liftoff[leg] = d.site_xpos[self.sites[LEGS.index(leg)]].copy()
      self.stance_prev[leg] = stance[leg]

    # Stance: the wrench on the CoM, shared out by least squares.
    pos_err = np.zeros(3)
    pos_err[:2] = self.x_des - com[:2]
    pos_err[:2] = np.clip(pos_err[:2], -0.05, 0.05)
    pos_err[2] = height - com[2]
    acc = (np.array([40.0, 40.0, 250.0]) * pos_err
           + np.array([12.0, 12.0, 25.0]) * (v_des - v))
    force = self.mass * (acc + np.array([0, 0, G]))
    yaw_err = math.atan2(math.sin(self.yaw_des - yaw), math.cos(self.yaw_des - yaw))
    # Roll and pitch are about the HEADING's axes, so the attitude loop runs
    # in the heading frame and its moment is turned back into the world's
    # (applied about the world's axes, a correction reverses past 90 deg of
    # heading, and a turn on the spot flipped the body at 140 deg).
    heading = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    att = np.array([-roll, cmd.pitch - slope - pitch, yaw_err])
    alpha = np.array([300.0, 300.0, 120.0]) * att + np.array(
      [30.0, 30.0, 20.0]) * (np.array([0, 0, cmd.yaw_rate]) - heading.T @ w)
    moment = heading @ (self.inertia * alpha)
    #: Which legs carry weight this step (LEGS order): the table splits
    #: stance load from how hard the instrument swings a leg.
    self.stance_now = np.array([stance[leg] for leg in LEGS])
    down = [i for i, leg in enumerate(LEGS) if stance[leg]]
    tau = np.zeros(12)
    if down:
      a = np.zeros((6, 3 * len(down)))
      for k, i in enumerate(down):
        r = d.site_xpos[self.sites[i]] - com
        a[:3, 3 * k:3 * k + 3] = np.eye(3)
        a[3:, 3 * k:3 * k + 3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]],
                                           [-r[1], r[0], 0]])
      b = np.concatenate([force, moment])
      wts = np.diag([1, 1, 1, 4, 4, 1.0])
      aa = wts @ a
      f = np.linalg.solve(aa.T @ aa + 1e-4 * np.eye(aa.shape[1]), aa.T @ (wts @ b))
      mu = 0.6
      for k, i in enumerate(down):
        fi = f[3 * k:3 * k + 3]
        fi[2] = max(fi[2], 2.0)
        lim = mu * fi[2]
        n = math.hypot(fi[0], fi[1])
        if n > lim:
          fi[:2] *= lim / n
        mujoco.mj_jacSite(m, d, self.jac, None, self.sites[i])
        cols = self.vadr[3 * i:3 * i + 3]
        tau[3 * i:3 * i + 3] = -self.jac[:, cols].T @ fi

    # Swing: Raibert placement under the hip, a raised-sine height profile.
    swing_t = (1.0 - duty) * cmd.period
    stance_t = duty * cmd.period
    for i, leg in enumerate(LEGS):
      if stance[leg]:
        continue
      ph = ((t / cmd.period) + offsets[leg]) % 1.0
      s = (ph - duty) / (1.0 - duty)
      hip = d.xpos[self.hips[i]]
      # The hip's own velocity: a turn moves every hip sideways (w x r),
      # and a foot placed for the torso's velocity alone lands behind it.
      r = hip - com
      v_hip = v + np.cross(w, r)
      v_hip_des = v_des + np.cross([0.0, 0.0, cmd.yaw_rate], r)
      target = hip.copy()
      target[:2] = hip[:2] + v_hip[:2] * stance_t / 2 + 0.05 * (v_hip[:2] - v_hip_des[:2])
      target[:2] = self._foothold(target[0], target[1])
      target[2] = self.terrain(target[0], target[1]) + spec.foot_r
      start = self.liftoff[leg] if self.liftoff[leg] is not None else target
      # Lift, travel, land: the foot clears the higher of the two footholds
      # before it moves across, so a riser's edge is stepped over, not into.
      apex = max(start[2], target[2]) + cmd.swing_height
      # Stepping UP, the foot travels only once it has risen past the upper
      # foothold, or a hind foot catches the riser's edge and flips the body.
      u0 = 0.4 if target[2] > start[2] + 0.02 else 0.2
      u = min(max((s - u0) / (0.8 - u0), 0.0), 1.0)
      du = (1 / (0.8 - u0)) if u0 < s < 0.8 else 0.0
      p_des = start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * u))
      v_des_f = (target - start) * 0.5 * math.pi * math.sin(math.pi * u) * du / swing_t
      if s < 0.4:
        z0, z1, blend = start[2], apex, s / 0.4
      else:
        z0, z1, blend = apex, target[2], max(0.0, (s - 0.4) / 0.6)
      p_des[2] = z0 + (z1 - z0) * (0.5 - 0.5 * math.cos(math.pi * blend))
      span = 0.4 if s < 0.4 else 0.6
      v_des_f[2] = ((z1 - z0) * 0.5 * math.pi * math.sin(math.pi * blend)
                    / (span * swing_t))
      mujoco.mj_jacSite(m, d, self.jac, None, self.sites[i])
      cols = self.vadr[3 * i:3 * i + 3]
      jl = self.jac[:, cols]
      v_foot = self.jac @ d.qvel
      f = self.swing_gain * (700.0 * (p_des - d.site_xpos[self.sites[i]])
                             + 25.0 * (v_des_f - v_foot))
      tau[3 * i:3 * i + 3] = jl.T @ f + d.qfrc_bias[cols]
    return tau
