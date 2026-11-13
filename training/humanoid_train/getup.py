"""Falling and getting up (#507, stage 5): the humanoid from the floor to its
stand -- lying on its back, on its front, dropped any way up, or tipping
over from standing -- by a policy of its own, which the served body hands
its legs AND arms to once it is falling, and back once it stands.

The quadruped's get-up (#389, at `quadruped-final`) is the template, with
what a biped needs:
  - the legs' and the arms' twenty-four joints are the policy's (the
    waist's yaw, #560's stage (c3), held home, as the served body holds
    it while it gets up): a body on the floor pushes off
    with its hands, and its arms meet its trunk and legs
    (`models/humanoid.xml`'s pairs), so a push is one a real arm could make;
  - a LIFT at the pelvis, a share of the body's weight that falls to
    nothing over training (`ASSIST_STAGES`): standing up from the floor is
    found far sooner held up than not (HoST, Huang et al. 2025, pulled its
    G1 up so);
  - the rise is paid as it comes, the pelvis's height off the floor, and
    being upright from the release (a reward held back killed the
    quadruped's roll, #389); the STAND -- the height, the pose, both soles
    down -- only from `BUDGET_S`, so standing sooner earns nothing;
  - a fall is part of it: a quarter of the episodes start TIPPING, the body
    upright and turning over at a shove's speed, and a hit harder than
    `IMPACT_FREE` body weights on the trunk, the head or a limb is charged,
    so the policy that takes a falling body over learns to land on its
    arms and legs rather than its head.

The charges set the pace, as the quadruped's do: torque past each part's
continuous rating and harder past 60 % of its peak, the joints' speed and
acceleration, the pelvis rising faster than `RISE_FREE_MPS`, the action's
rate -- ramped up over training (`GENTLE_STAGES`).

`-Getup-Wall` (#679) carries `-Getup-Head` on with a wall or the counter
beside the fallen body in a share of its episodes (`OBSTACLE`): the get-up
trained on open floor alone never stood a robot lying against one.
"""

import copy
import dataclasses
import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from humanoid_train import robot
from humanoid_train.walk import blown_up

_ROBOT = SceneEntityCfg("robot")
JOINTS = robot.LEGS + robot.ARMS

#: Episode seconds: a fall or a body lying still, then time to get up and
#: stand quietly.
EPISODE_S = 10.0
#: How each episode starts, and the share of them: on its back, on its
#: front or on a side (each a few centimetres up, its joints near the
#: stand's), dropped from 0.85-1.05 m any way up with its joints anywhere
#: near the stand's, TIPPING from its stand at a shove's speed, or POSED
#: where a get-up was seen to end (`POSES`; none but `-Getup-Wall2`'s).
KINDS = ("back", "front", "side", "dropped", "tipping", "posed")
STARTS = {"back": 0.3, "front": 0.25, "side": 0.0, "dropped": 0.2, "tipping": 0.25}
#: A side start's pelvis, m: clear of the floor with an arm under it.
SIDE_M = 0.22
#: A tipping start's speed at the pelvis, m/s, turning the body over its
#: feet (0.23 m/s is a 10 N*s shove, which the standing policy stands
#: through: `humanoid_spike.py --shove`).
TIP_MPS = (0.6, 1.4)
#: The stand pays from this long after the release, s.
BUDGET_S = 3.0
#: The pelvis's height lying, m: the rise is paid from here to the stand's.
LYING_M = 0.10
#: Past 60 % of a part's peak is charged harder (#389's bound).
CAP_SHARE = 0.6
#: The pelvis may rise this fast for nothing, m/s.
RISE_FREE_MPS = 0.5
#: A hit on anything but a sole up to this many body weights is free; past
#: it each body weight is charged (`impact`).
IMPACT_FREE = 2.0
#: The charges for getting up hard, at full weight, and their share of it
#: from each iteration on.
GENTLE = {
  "torque_over_rating": -0.5,
  "torque_over_cap": -5.0,
  "joint_vel_l2": -0.005,
  "joint_acc_l2": -1e-7,
  "rise_speed": -5.0,
  "action_rate_l2": -0.02,
  "impact": -0.2,
}
GENTLE_STAGES = ((0, 0.1), (500, 0.25), (1000, 0.5), (1500, 1.0))
#: The lift at the pelvis, a share of the body's weight, from each
#: iteration on: held up, the body finds its way to its feet; then it must
#: carry itself.
ASSIST_STAGES = ((0, 0.5), (800, 0.3), (1600, 0.15), (2400, 0.0))
STEPS_PER_ITERATION = 24
#: `-Getup-Head`, a get-up that stands carried on: the head's own hits
#: charged past `HEAD_FREE` body weights at `HEAD_WEIGHT` a body weight
#: (the rewards are scaled by the step: a 5 kN hit costs a tenth of an
#: episode's pay), and more of the episodes a fall (`HEAD_STARTS`). ⚠ At
#: `impact`'s charge alone, taking a falling body over it let the head
#: strike at 5.5-6.9 m/s falling forward or sideways, harder than the
#: standing policy's own falls (`humanoid_spike.py --fall`).
HEAD_FREE, HEAD_WEIGHT = 0.5, -20.0
#: ⚠ With no side among its starts (dropped any way up was 0.2), it lost
#: getting up off one side, which the get-up before it had (getup2).
HEAD_STARTS = {"back": 0.15, "front": 0.15, "side": 0.15, "dropped": 0.15, "tipping": 0.4}

#: `-Getup-Wall` (#679): a box beside the fallen body in `share` of the
#: episodes, its top a counter's (`counter` of them: the storeroom's is
#: 0.72 m, a table's 0.75) or a room's wall's; the rest open floor, as
#: `-Getup-Head`'s. Where its face stands is drawn as a GAP past the body's
#: own reach toward it (`_reach`, off the body as it starts), so a start
#: never begins inside it: beside the trunk, beyond the head or the feet of
#: a body lying, and in the way a body tips -- into it, or along its face --
#: as the counter's falls went (a robot at its work spot, falling forward
#: and to one side, lay along the counter's front, 0.23 m off it). `front`:
#: the share of a body on its side whose box is beside its FRONT, not its
#: back -- the one way of lying the served get-up never stood from -- and
#: `left` the share of a body on its back or front with the box at its left;
#: `share` is the lying starts', `tipping_share` the tipping starts'.
OBSTACLE = {"share": 0.5, "tipping_share": 0.25, "posed_share": 0.5, "counter": 0.6,
            "counter_top_m": (0.6, 0.95),
            "wall_top_m": 2.4, "lying": {"beside": 0.7, "head": 0.15, "feet": 0.15},
            "front": 0.9, "left": 0.5, "tipping": {"into": 0.5, "along": 0.5, "far": 0.0},
            "gap_m": {"beside": (0.0, 0.25), "head": (0.0, 0.35), "feet": (0.0, 0.35),
                      "into": (0.05, 0.5), "along": (0.0, 0.3), "far": (0.7, 1.4),
                      "posed": (0.0, 0.4)}}
#: `-Getup-Wall2`'s box (#679's second round): with half the tipping starts,
#: and FAR in 0.4 of those -- into the way it tips, its face 0.7-1.4 m past
#: the standing body's reach, a wall its head meets as it lands (the hall's
#: walls 1.48 m off its middle, the storeroom's north wall 1.2 m off the
#: chest's work spot); a sitter's at any side in half of the posed starts.
OBSTACLE2 = {**OBSTACLE, "tipping_share": 0.5, "tipping": {"into": 0.3, "along": 0.3, "far": 0.4}}

#: `-Getup-Wall2` (#679's second round): where D3's wall policy was seen to
#: stand still in four falls today's get-up stands from -- three sitting on
#: the floor, one propped on a wall -- and each fall as the get-up was handed
#: it, moving (`humanoid_spike.py --fall-ends`): each drawn evenly, mirrored
#: half the time, at any heading, each joint off its pose by up to
#: `POSE_SPREAD_RAD` and the trunk's roll and pitch by up to `POSE_TILT_RAD`,
#: its lowest part `POSE_DROP_M` over the floor. One recorded `against` its
#: face starts against it (a wall's, where it was); the others take the box
#: at any side in `posed_share` of them.
POSES = Path(__file__).with_name("getup_poses.json")
POSE_SPREAD_RAD, POSE_TILT_RAD, POSE_DROP_M = 0.2, 0.1, 0.02
#: `-Getup-Wall3` (#679's third round): the get-up HANDS BACK as served --
#: upright within `STOOD_UP` and the pelvis `HAND_BACK_M` of the stand's
#: height for `HAND_BACK_S` (`humanoid.getup`'s `STOOD_*`, which a test reads
#: these against), the standing policy (`STAND_NPZ`, served) has the legs and
#: the arms held at its pose, until a fall past `RETAKE_DEG` or the pelvis
#: under `RETAKE_M` (the served watch's). ⚠ Never handed back in training, a
#: get-up stood where the standing policy could not keep it (its soles
#: 0.53-0.65 m apart: it fell within 2 s). ⚠ No gyro test, which the served
#: one has: under the action's noise the trunk never settled, and nothing was
#: handed back. (SimNotes, "The wall get-up's second and third rounds".)
STAND_NPZ = robot.REPO / "models" / "humanoid_stand.npz"
HAND_BACK_M, HAND_BACK_S = 0.05, 0.5
#: Handed back is paid this a second (`upright`'s weight).
HANDED_WEIGHT = 1.0
RETAKE_DEG, RETAKE_M = 35.0, 0.40
#: Its stand is the served get-up's: within 10 deg of upright
#: (`getup.STOOD_TILT_RAD`), and touching nothing of the box. ⚠ Paid at the
#: open floor's 0.9 (25 deg), it stood leaning on the counter 18 deg off
#: upright, and the served get-up, which hands back at 10, never did.
STOOD_UP = math.cos(math.radians(10.0))
#: Still down -- the pelvis under `DOWN_M` -- past `BUDGET_S` is charged
#: `DOWN_WEIGHT` a second (`down`).
DOWN_M, DOWN_WEIGHT = 0.4, -1.0
#: `-Getup-Wall`'s starts: `-Getup-Head`'s with twice the sides (a body on
#: its side, front to the box, is the one getup3 never stood from).
WALL_STARTS = {**HEAD_STARTS, "side": 0.3}
#: `-Getup-Wall2`'s: `-Getup-Wall`'s, and POSED in 0.2 beside their 1.15.
WALL2_STARTS = {**WALL_STARTS, "posed": 0.2}
#: Its episodes, s: a slower stand beside a face still earns.
WALL_EPISODE_S = 20.0
#: Its PPO: the entropy bonus kept (at 0 the action's noise fell from 0.23
#: to 0.06, and a get-up stood shaking in the served physics), and small
#: steps from getup3 (at `desired_kl` 0.01 the get-up off its back beside
#: a box fell from getup3's 0.99 to 0.51 in 450 iterations, in the training
#: physics; at 0.004, 0.96).
WALL_PPO = {"entropy_coef": 0.005, "desired_kl": 0.004}
#: The box, half-sizes, m: as deep as the counter, 4 m of face, and as tall
#: as a wall -- sunk into the floor to a counter's top (a mocap body is
#: welded to the world, and meets no floor), and its top under the floor
#: where an episode has none.
OBSTACLE_HALF = (0.3, 2.0, 1.2)
PARKED_TOP_M = -1.0

#: An action of 1 moves a joint's target this far, rad: the joints a
#: get-up swings through their ranges (the hips' pitch, the knees, the
#: shoulders, the elbows) the furthest.
ACTION_SCALE = {
  j: (0.6 if any(k in j for k in ("hip_pitch", "knee", "shoulder_pitch", "shoulder_roll",
                                   "elbow"))
      else 0.15 if "ankle_roll" in j else 0.3)
  for j in JOINTS
}
#: The head's parts, built round the LIDAR (#581's stage 3): a hit on any
#: of them is the head's (the head was one cylinder when getup3 trained).
HEAD = ("head", "head_visor", "head_dome", "head_rim", "head_cap", "head_fin", r"head_back\d+",
        "lidar")
#: The trunk's and the limbs' geoms, whose hits on the floor are charged.
STRUCK = (("pelvis", "torso") + HEAD
          + tuple(f"{s}_{g}" for s in ("left", "right")
                  for g in ("thigh", "shank", "upper_arm", "forearm", "gripper")))


def _up(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """1 standing, 0 on its side or lying, -1 on its head."""
  return -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]


def _z(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]


def _since_release(env) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt


def _paid(env, budget_s: float) -> torch.Tensor:
  return (_since_release(env) >= budget_s).float()


# ---- the starts --------------------------------------------------------------------

def _quat_from_rpy(roll, pitch, yaw):
  cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
  cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
  cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
  return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
                     dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """The product a b of (w, x, y, z) quaternions."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw],
                     dim=-1)


def _quat_apply(quat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """v turned by the (w, x, y, z) quaternion."""
  w, u = quat[:, :1], quat[:, 1:]
  t = 2.0 * torch.cross(u, v, dim=-1)
  return v + w * t + torch.cross(u, t, dim=-1)


def _poses(env, asset, names: tuple[str, ...] | None):
  """The posed starts' table (`POSES`) in the asset's joint order, read once:
  each pose's joints and their speeds and their mirror image's, its trunk
  with no heading (w, x, y, z) and its velocity (linear, angular: the world's
  turned by that heading), whether it starts against its face, and the
  face's way off the heading (rad) and distance from the pelvis (m)."""
  key = ("_getup_poses", names)
  cache = getattr(env, "_getup_pose_cache", {})
  if key not in cache:
    table = json.loads(POSES.read_text())["poses"]
    chosen = [k for k in table if names is None or k in names]
    if not chosen:
      raise ValueError(f"no pose of {names} in {POSES}")
    joints = list(asset.joint_names)
    default = asset.data.default_joint_pos[0]
    twins = [_mirrored(name) for name in joints]
    index, sign = [joints.index(t) for t, _ in twins], [s for _, s in twins]
    q, qd, quat, vel, lean, ang, dist = [], [], [], [], [], [], []
    for k in chosen:
      pose = table[k]
      row, speed = default.clone(), torch.zeros_like(default)
      for j, name in enumerate(joints):
        if name in pose["joints"]:
          row[j] = pose["joints"][name]
          speed[j] = pose.get("joint_vel", {}).get(name, 0.0)
      q.append(row)
      qd.append(speed)
      quat.append(pose["quat_level"])
      vel.append(pose.get("lin_vel", [0.0] * 3) + pose.get("ang_vel", [0.0] * 3))
      lean.append(bool(pose["against"]))
      ang.append((pose["face"] or {"angle": 0.0})["angle"])
      dist.append((pose["face"] or {"dist": 0.0})["dist"])
    dev = env.device
    q, qd = torch.stack(q).to(dev), torch.stack(qd).to(dev)
    index, sign = torch.tensor(index, device=dev), torch.tensor(sign, device=dev)
    vel = torch.tensor(vel, device=dev, dtype=torch.float32)
    # the mirror image's: a polar vector's y turned over, an axial one's x and z
    flip = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=dev)
    cache[key] = {"q": q, "q_mirror": q[:, index] * sign, "qd": qd, "qd_mirror": qd[:, index] * sign,
                  "quat": torch.tensor(quat, device=dev, dtype=torch.float32),
                  "vel": vel, "vel_mirror": vel * flip,
                  "lean": torch.tensor(lean, device=dev), "angle": torch.tensor(ang, device=dev),
                  "dist": torch.tensor(dist, device=dev)}
    env._getup_pose_cache = cache
  return cache[key]


def reset_fallen(env, env_ids, starts: dict | None = None,
                 asset_cfg: SceneEntityCfg = _ROBOT, obstacle: dict | None = None,
                 poses: tuple[str, ...] | None = None) -> None:
  """Each episode's start, drawn by `starts` (`STARTS`'s kinds), and with
  `obstacle` (`OBSTACLE`'s draws) the box placed against it; a posed start
  among `poses` (`POSES`' names; all of them by default)."""
  env_ids = envs_mdp.events.resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  shares = [(starts or STARTS).get(k, 0.0) for k in KINDS]
  n, dev = len(env_ids), env.device
  u = torch.rand(n, device=dev) * sum(shares)
  edges = torch.cumsum(torch.tensor(shares, device=dev), 0)
  kind = torch.searchsorted(edges, u.contiguous()).clamp(max=len(KINDS) - 1)
  back, front, side, dropped, tipping, posed = (kind == k for k in range(len(KINDS)))
  yaw = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  roll = torch.empty(n, device=dev).uniform_(-0.3, 0.3)
  pitch = torch.where(back, torch.full_like(yaw, -math.pi / 2),
                      torch.full_like(yaw, math.pi / 2))
  flip = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
  roll = torch.where(side, flip * math.pi / 2 + roll, roll)
  pitch = torch.where(side, torch.zeros_like(pitch), pitch)
  roll = torch.where(dropped, torch.empty(n, device=dev).uniform_(-math.pi, math.pi), roll)
  pitch = torch.where(dropped, torch.empty(n, device=dev).uniform_(-math.pi / 2, math.pi / 2),
                      pitch)
  roll = torch.where(tipping, torch.zeros_like(roll), roll)
  pitch = torch.where(tipping, torch.zeros_like(pitch), pitch)
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, 2] = torch.where(dropped, torch.empty(n, device=dev).uniform_(0.85, 1.05),
                          torch.full_like(yaw, 0.12))
  pos[:, 2] = torch.where(tipping, torch.full_like(yaw, robot.BODY["stand_height"] + 0.01),
                          pos[:, 2])
  # On a side the arm under it reaches 0.2 m from the pelvis.
  pos[:, 2] = torch.where(side, torch.full_like(yaw, SIDE_M), pos[:, 2])
  vel = torch.zeros(n, 6, device=dev)
  way = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  speed = torch.empty(n, device=dev).uniform_(*TIP_MPS) * tipping.float()
  vel[:, 0], vel[:, 1] = speed * torch.cos(way), speed * torch.sin(way)
  turn = speed / robot.BODY["stand_height"]
  vel[:, 3], vel[:, 4] = -turn * torch.sin(way), turn * torch.cos(way)
  quat = _quat_from_rpy(roll, pitch, yaw)
  q0 = asset.data.default_joint_pos[env_ids]
  spread = torch.where(dropped, 0.6, torch.where(tipping, 0.0, 0.3))[:, None]
  q = q0 + spread * (2 * torch.rand_like(q0) - 1)
  lean = face_ang = face_dist = None
  qd = torch.zeros_like(q0)
  if bool(posed.any()):
    # ...where a get-up was seen to end: its pose (mirrored half the time),
    # turned to `yaw`, tilted a little, its joints a little off
    table = _poses(env, asset, poses)
    pick = torch.randint(len(table["q"]), (n,), device=dev)
    mirrored = torch.rand(n, device=dev) < 0.5
    pose_q = torch.where(mirrored[:, None], table["q_mirror"][pick], table["q"][pick])
    level = table["quat"][pick]
    level = torch.where(mirrored[:, None], level * torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev),
                        level)
    tilt = _quat_from_rpy(_uniform(n, (-POSE_TILT_RAD, POSE_TILT_RAD), dev),
                          _uniform(n, (-POSE_TILT_RAD, POSE_TILT_RAD), dev), yaw)
    quat = torch.where(posed[:, None], _quat_mul(tilt, level), quat)
    q = torch.where(posed[:, None], pose_q + POSE_SPREAD_RAD * (2 * torch.rand_like(q0) - 1), q)
    qd = torch.where(posed[:, None], torch.where(mirrored[:, None], table["qd_mirror"][pick],
                                                 table["qd"][pick]), qd)
    moving = torch.where(mirrored[:, None], table["vel_mirror"][pick], table["vel"][pick])
    turned = torch.cat([_quat_apply(tilt, moving[:, :3]), _quat_apply(tilt, moving[:, 3:])], -1)
    vel = torch.where(posed[:, None], turned, vel)
    lean = posed & table["lean"][pick]
    face_ang = torch.where(mirrored, -table["angle"][pick], table["angle"][pick])
    face_dist = table["dist"][pick]
    pos[:, 2] = torch.where(posed, torch.full_like(yaw, 1.0), pos[:, 2])
  limits = asset.data.soft_joint_pos_limits[env_ids]
  q = torch.max(torch.min(q, limits[..., 1]), limits[..., 0])
  asset.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids)
  asset.write_joint_state_to_sim(q, qd, env_ids=env_ids)
  if lean is not None:
    # ...its lowest part `POSE_DROP_M` over the floor, off where its geoms are
    with wp.ScopedDevice(env.sim.wp_device):
      mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
    down = torch.zeros(n, 3, device=dev)
    down[:, 2] = -1.0
    below = _reach(env, env_ids, down, pos, asset_cfg)     # how far under the root it reaches
    floor = env.scene.env_origins[env_ids][:, 2]
    pos[:, 2] = torch.where(posed, floor + below + POSE_DROP_M, pos[:, 2])
    asset.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids)
  if obstacle is not None:
    _place_obstacle(env, env_ids, obstacle, pos, quat, tipping, way, asset_cfg,
                    posed=posed, lean=lean, face_ang=face_ang, face_dist=face_dist, yaw=yaw)


def _uniform(n, lo_hi, device) -> torch.Tensor:
  return torch.empty(n, device=device).uniform_(*lo_hi)


def _trunk_axis(quat: torch.Tensor) -> torch.Tensor:
  """The trunk's own up (pelvis to head) in the world, from (w, x, y, z)."""
  w, x, y, z = quat.unbind(-1)
  return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def _trunk_front(quat: torch.Tensor) -> torch.Tensor:
  """The trunk's own ahead in the world, from (w, x, y, z)."""
  w, x, y, z = quat.unbind(-1)
  return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)


def _trunk_left(quat: torch.Tensor) -> torch.Tensor:
  """The trunk's own left in the world, from (w, x, y, z)."""
  w, x, y, z = quat.unbind(-1)
  return torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)


def _robot_extent(env, env_ids, asset_cfg: SceneEntityCfg):
  """The robot's colliding geoms (their ids, each one's own box in its
  frame: centre, half-size), read once off the compiled model."""
  cache = getattr(env, "_getup_extent", None)
  if cache is None:
    mj = env.sim.mj_model
    ids = env.scene[asset_cfg.name].indexing.geom_ids.cpu().numpy()
    ids = ids[(mj.geom_contype[ids] > 0) | (mj.geom_conaffinity[ids] > 0)]
    box = torch.as_tensor(mj.geom_aabb[ids], dtype=torch.float32, device=env.device)
    cache = (torch.as_tensor(ids, device=env.device, dtype=torch.long), box[:, :3], box[:, 3:])
    env._getup_extent = cache
  return cache


def _reach(env, env_ids, toward: torch.Tensor, root: torch.Tensor,
           asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """How far the robot's body reaches from its root along `toward` (unit,
  world), m, each geom its own box as it stands now (after a forward)."""
  ids, centre, half = _robot_extent(env, env_ids, asset_cfg)
  xpos = env.sim.data.geom_xpos[env_ids][:, ids]                  # [B, G, 3]
  xmat = env.sim.data.geom_xmat[env_ids][:, ids]                  # [B, G, 3, 3]
  mid = xpos + torch.einsum("bgij,gj->bgi", xmat, centre)
  along = torch.einsum("bgi,bi->bg", mid - root[:, None, :], toward)
  local = torch.einsum("bgji,bj->bgi", xmat, toward)              # R^T toward
  return (along + (local.abs() * half[None]).sum(-1)).amax(dim=1)


def _place_obstacle(env, env_ids, draws: dict, pos, quat, tipping, way,
                    asset_cfg: SceneEntityCfg, posed=None, lean=None, face_ang=None,
                    face_dist=None, yaw=None) -> None:
  """The box against each start (`OBSTACLE`): drawn present, a counter's
  top or a wall's, its face square to a way from the root -- beside the
  trunk, beyond the head or the feet of a body lying; into, along or FAR
  into the way a body tips; at any side of a body posed, or where a posed
  body leant (`lean`: a wall, `face_dist` from the pelvis at `face_ang` off
  its heading `yaw`) -- a gap past the body's reach that way; else parked."""
  n, dev = len(env_ids), env.device
  none = torch.zeros(n, dtype=torch.bool, device=dev)
  posed = none if posed is None else posed
  lean = none if lean is None else lean
  with wp.ScopedDevice(env.sim.wp_device):   # the starts just written: where each geom is
    mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
  share = torch.where(tipping, draws["tipping_share"], draws["share"])
  share = torch.where(posed, draws.get("posed_share", draws["share"]), share)
  present = (torch.rand(n, device=dev) < share) | lean
  counter = (torch.rand(n, device=dev) < draws["counter"]) & ~lean
  top = torch.where(counter, _uniform(n, draws["counter_top_m"], dev),
                    torch.full((n,), draws["wall_top_m"], device=dev))
  top = torch.where(present, top, torch.full_like(top, PARKED_TOP_M))
  # The lying body's long way, its head's side (dropped near upright: any).
  head = _trunk_axis(quat)[:, :2]
  long = torch.atan2(head[:, 1], head[:, 0])
  long = torch.where(head.norm(dim=-1) > 0.3, long, _uniform(n, (-math.pi, math.pi), dev))
  side = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
  lying_p = torch.tensor([draws["lying"][k] for k in ("beside", "head", "feet")], device=dev)
  tip_p = torch.tensor([draws["tipping"].get(k, 0.0) for k in ("into", "along", "far")],
                       device=dev)
  lie = torch.multinomial(lying_p, n, replacement=True)
  tip = torch.multinomial(tip_p, n, replacement=True)
  # Beside a body on its side: at its front (`front` of them) or its back;
  # on its back or its front: at its left (`left` of them) or its right.
  ahead, lefts = _trunk_front(quat)[:, :2], _trunk_left(quat)[:, :2]
  facing = torch.atan2(ahead[:, 1], ahead[:, 0]) + torch.where(
    torch.rand(n, device=dev) < draws["front"], 0.0, math.pi)
  leftward = torch.atan2(lefts[:, 1], lefts[:, 0]) + torch.where(
    torch.rand(n, device=dev) < draws["left"], 0.0, math.pi)
  across = torch.where(ahead.norm(dim=-1) > 0.7, facing,
                       torch.where(lefts.norm(dim=-1) > 0.7, leftward, long + side * math.pi / 2))
  ang = torch.where(lie == 0, across, torch.where(lie == 1, long, long + math.pi))
  ang = ang + _uniform(n, (-0.25, 0.25), dev)
  tipped = torch.where(tip == 0, way + _uniform(n, (-0.6, 0.6), dev),
                       torch.where(tip == 1, way + side * math.pi / 2 + _uniform(n, (-0.3, 0.3), dev),
                                   way + _uniform(n, (-0.5, 0.5), dev)))
  ang = torch.where(tipping, tipped, ang)
  gaps = {k: _uniform(n, draws["gap_m"][k], dev) for k in draws["gap_m"]}
  gap = torch.where(lie == 0, gaps["beside"], torch.where(lie == 1, gaps["head"], gaps["feet"]))
  gap = torch.where(tipping, torch.where(tip == 0, gaps["into"],
                                         torch.where(tip == 1, gaps["along"], gaps["far"])), gap)
  if bool(posed.any()):
    ang = torch.where(posed, _uniform(n, (-math.pi, math.pi), dev), ang)
    ang = torch.where(lean, yaw + face_ang, ang)
    gap = torch.where(posed, gaps["posed"], gap)
  toward = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
  reach = _reach(env, env_ids, toward, pos, asset_cfg)
  face = reach + gap
  if bool(lean.any()):
    # ...where it leant, never inside the body as its joints now lie
    face = torch.where(lean, torch.maximum(face_dist, reach + 0.005), face)
  centre = pos.clone()
  centre[:, :2] = pos[:, :2] + toward[:, :2] * (face + OBSTACLE_HALF[0])[:, None]
  centre[:, 2] = env.scene.env_origins[env_ids][:, 2] + top - OBSTACLE_HALF[2]
  turn = torch.stack([torch.cos(ang / 2), torch.zeros_like(ang), torch.zeros_like(ang),
                      torch.sin(ang / 2)], -1)
  env.scene["obstacle"].write_mocap_pose_to_sim(torch.cat([centre, turn], -1), env_ids)
  if not hasattr(env, "_getup_obstacle"):
    env._getup_obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
  env._getup_obstacle[env_ids] = present


def obstacle_spec() -> mujoco.MjSpec:
  """The box (`OBSTACLE_HALF`) on a mocap body, parked: it meets the
  robot's geoms (contype 1) and nothing of the world."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="obstacle", mocap=True,
                                 pos=[0.0, 0.0, PARKED_TOP_M - OBSTACLE_HALF[2]])
  body.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(OBSTACLE_HALF),
                contype=0, conaffinity=1, rgba=[0.86, 0.85, 0.82, 1.0])
  return spec


def assist(env, env_ids, share: float, asset_cfg: SceneEntityCfg) -> None:
  """The lift at the pelvis for the episodes starting: `share` of the
  body's weight, upward, held until their next start. ⚠ `asset_cfg` must be
  in the event's params, where the manager resolves its body: a default
  here stays unresolved, and the lift went on every body (the robot flew)."""
  env_ids = envs_mdp.events.resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  forces = torch.zeros(len(env_ids), 1, 3, device=env.device)
  forces[..., 2] = share * robot.MASS_KG * 9.81
  asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces), env_ids=env_ids,
                                     body_ids=asset_cfg.body_ids)


def assist_curriculum(env, env_ids, stages=ASSIST_STAGES):
  """The lift's share for the iteration training is at."""
  it = env.common_step_counter // STEPS_PER_ITERATION
  share = [s for at, s in stages if it >= at][-1]
  env.event_manager.get_term_cfg("assist").params["share"] = share
  return torch.tensor(share)


# ---- rewards -----------------------------------------------------------------------

def upright(env) -> torch.Tensor:
  """The right way up, from the release."""
  return torch.square((1.0 + _up(env)) / 2.0)


def rise(env) -> torch.Tensor:
  """The pelvis off the floor, from lying to the stand's height, from the
  release: kneeling is half way."""
  return torch.clamp((_z(env) - LYING_M) / (robot.BODY["stand_height"] - LYING_M), 0.0, 1.0)


def _free(env, free: str | None) -> torch.Tensor:
  """1 where the contact sensor `free` (none: no check) finds nothing."""
  if free is None:
    return torch.ones(env.num_envs, device=env.device)
  return (env.scene[free].data.found == 0).all(dim=1).float()


def stand_height(env, std: float, budget_s: float, upright: float = 0.9,
                 free: str | None = None) -> torch.Tensor:
  gate = (_up(env) > upright).float() * _free(env, free)
  return _paid(env, budget_s) * gate * torch.exp(
    -torch.square(_z(env) - robot.BODY["stand_height"]) / std ** 2)


def stand_pose(env, std: float, budget_s: float, asset_cfg: SceneEntityCfg = _ROBOT,
               upright: float = 0.9, free: str | None = None):
  asset = env.scene[asset_cfg.name]
  err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
  gate = (_up(env) > upright).float() * _free(env, free)
  return _paid(env, budget_s) * gate * torch.exp(-err / std ** 2)


def feet_down(env, sensor_name: str, budget_s: float, free: str | None = None) -> torch.Tensor:
  """Both soles on the floor, standing tall."""
  found = env.scene[sensor_name].data.found
  tall = (_z(env) > robot.BODY["stand_height"] - 0.1).float()
  return _paid(env, budget_s) * tall * (found > 0).all(dim=1).float() * _free(env, free)


def down(env, budget_s: float, below_m: float) -> torch.Tensor:
  """Still down -- the pelvis under `below_m` -- from `budget_s`: 1 a second."""
  return _paid(env, budget_s) * (_z(env) < below_m).float()


def _ratio(env, asset_cfg: SceneEntityCfg, of: torch.Tensor) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.actuator_ids]
  return tau.abs() / of.to(tau.device)


def torque_over_rating(env, asset_cfg: SceneEntityCfg, rated: torch.Tensor):
  """Each joint's torque past its part's continuous rating, in ratings,
  squared."""
  return torch.sum(torch.square(torch.clamp(_ratio(env, asset_cfg, rated) - 1.0, min=0.0)), 1)


def torque_over_cap(env, asset_cfg: SceneEntityCfg, cap: torch.Tensor):
  """Each joint's torque past `CAP_SHARE` of its part's peak, in caps."""
  return torch.sum(torch.clamp(_ratio(env, asset_cfg, cap) - 1.0, min=0.0), 1)


def rise_speed(env, free: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """The pelvis rising faster than `free`, squared (falling is free)."""
  vz = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2]
  return torch.square(torch.clamp(vz - free, min=0.0))


def impact(env, sensor_name: str, free: float) -> torch.Tensor:
  """The hardest hit this policy step on each struck part past `free` body
  weights, in body weights, summed over the parts."""
  hist = env.scene[sensor_name].data.force_history          # [B, N, H, 3]
  peak = torch.norm(hist, dim=-1).amax(dim=-1) / (robot.MASS_KG * 9.81)
  return torch.sum(torch.clamp(peak - free, min=0.0), dim=1)


def gentle_curriculum(stages=GENTLE_STAGES) -> dict:
  return {f"gentle_{name}": CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={"reward_name": name, "stages": [
      {"step": it * STEPS_PER_ITERATION, "weight": w * share} for it, share in stages]})
    for name, w in GENTLE.items()}


# ---- the critic's extras -------------------------------------------------------------

def root_height(env) -> torch.Tensor:
  return _z(env)[:, None]


def since_release(env) -> torch.Tensor:
  return _since_release(env)[:, None] / EPISODE_S


def joint_torque(env, asset_cfg: SceneEntityCfg, peak: torch.Tensor) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.actuator_ids]
  return tau / peak.to(tau.device)


# ---- the mirror image (#679's symmetry, rsl_rl's data augmentation) ----------------

#: The joints whose angle a mirror image flips (about the trunk's x-z plane):
#: the rolls and the yaws -- a left and a right joint's axes are each
#: other's reflection, so a pitch keeps its sign (`models/humanoid.xml`).
FLIPS = ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw", "wrist_roll")
#: Each observation term's mirror: a joint term swaps left and right, a
#: polar vector flips y, an axial one x and z, a scalar stays.
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions", "joint_torque")
_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
MIRROR_TERMS = {"base_lin_vel": _POLAR, "projected_gravity": _POLAR, "base_ang_vel": _AXIAL,
                "root_height": (1.0,), "since_release": (1.0,)}


def _mirrored(name: str) -> tuple[str, float]:
  """A joint's twin in the mirror image and the sign its angle takes: a
  left's is the right's and back, a joint on the middle (the waist's yaw)
  its own; a roll or a yaw turns over."""
  side, _, part = name.partition("_")
  if side not in ("left", "right"):
    return name, -1.0 if part.endswith("yaw") else 1.0
  return ("right" if side == "left" else "left") + "_" + part, -1.0 if part in FLIPS else 1.0


@functools.cache
def _joint_mirror(device):
  twins = [_mirrored(name) for name in JOINTS]
  return (torch.tensor([JOINTS.index(t) for t, _ in twins], device=device),
          torch.tensor([s for _, s in twins], device=device))


def _mirror_group(x: torch.Tensor, terms, dims) -> torch.Tensor:
  index, sign = _joint_mirror(x.device)
  out, at = [], 0
  for name, dim in zip(terms, dims):
    part = x[:, at:at + dim]
    if name in _JOINT_TERMS:
      out.append(part[:, index] * sign)
    else:
      out.append(part * torch.tensor(MIRROR_TERMS[name], device=x.device))
    at += dim
  assert at == x.shape[1], (at, x.shape)
  return torch.cat(out, dim=1)


def mirror(env, obs=None, actions=None):
  """rsl_rl's symmetry augmentation: each sample and its mirror image, the
  batch doubled (originals first). A blind get-up that rolls one way off its
  back was stopped by a box on that side (wall8: 10 % stood with the box at
  its left, 100 % at its right, in the training physics); a mirrored batch
  trains both ways at once."""
  out_obs = out_act = None
  if obs is not None:
    om = getattr(env, "unwrapped", env).observation_manager
    mirrored = {}
    for group in obs.keys():
      terms = om.active_terms[group]
      dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[group]]
      mirrored[group] = torch.cat([obs[group], _mirror_group(obs[group], terms, dims)], dim=0)
    out_obs = obs.__class__(mirrored, batch_size=[2 * obs.batch_size[0]])
  if actions is not None:
    index, sign = _joint_mirror(actions.device)
    out_act = torch.cat([actions, actions[:, index] * sign], dim=0)
  return out_obs, out_act


# ---- what an episode ended as (`-Getup-Wall`'s metrics, each the share of
# the episodes ending: standing with the box beside it, standing on open
# floor, and with the box; the first over the last is the rate) ------------------

def _beside(env) -> torch.Tensor:
  got = getattr(env, "_getup_obstacle", None)
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) if got is None else got


def _standing(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
  """Standing as the served get-up hands back (`STOOD_UP`, its height to
  5 cm) on both soles, and leaning on no box."""
  found = env.scene[sensor_name].data.found
  tall = (_z(env) - robot.BODY["stand_height"]).abs() < 0.05
  free = _free(env, "against_obstacle" if "against_obstacle" in env.scene.sensors else None)
  return (_up(env) > STOOD_UP) & tall & (found > 0).all(dim=1) & (free > 0)


def stood_beside(env) -> torch.Tensor:
  return (_standing(env) & _beside(env)).float()


def stood_open(env) -> torch.Tensor:
  return (_standing(env) & ~_beside(env)).float()


def beside(env) -> torch.Tensor:
  return _beside(env).float()


@dataclass(kw_only=True)
class HandBackActionCfg(JointPositionActionCfg):
  """The get-up's joints, handed to the standing policy as served
  (`HandBackAction`)."""

  stand_npz: str = str(STAND_NPZ)

  def build(self, env) -> "HandBackAction":
    return HandBackAction(self, env)


class HandBackAction(JointPositionAction):
  """The get-up policy's targets until it stands as the served get-up hands
  back; then the standing policy's on the legs (its observation as the
  served `StandDriver` builds it, its last action started at none, its
  posture the stand's) and the arms held at its pose -- until a fall the
  served watch would take, when the get-up has them again."""

  def __init__(self, cfg: HandBackActionCfg, env):
    super().__init__(cfg, env)
    import numpy as np
    dev = self.device
    with np.load(cfg.stand_npz) as z:
      self.mean = torch.tensor(z["obs_mean"], device=dev, dtype=torch.float32)
      self.div = torch.tensor(z["obs_div"], device=dev, dtype=torch.float32)
      self.layers = [(torch.tensor(z[f"w{k}"], device=dev, dtype=torch.float32),
                      torch.tensor(z[f"b{k}"], device=dev, dtype=torch.float32))
                     for k in range(int(z["layers"]))]
      meta, training = json.loads(str(z["meta"])), json.loads(str(z["training"]))
    assert meta["observation_names"] == "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,posture"
    names = list(self._entity.joint_names)
    self.seen = torch.tensor([names.index(n) for n in meta["joint_names"].split(",")], device=dev)
    self.seen_default = torch.tensor([float(v) for v in meta["default_joint_pos"].split(",")],
                                     device=dev)
    driven = [names[int(i)] for i in self._target_ids]
    legs = list(training["action_joints"])
    self.legs = torch.tensor([driven.index(n) for n in legs], device=dev)
    scale = torch.tensor([float(v) for v in meta["action_scale"].split(",")], device=dev)
    default = dict(zip(meta["joint_names"].split(","), self.seen_default.tolist()))
    self.leg_default = torch.tensor([default[n] for n in legs], device=dev)
    self.leg_scale = scale
    #: what the standing policy holds every driven joint at, its legs' aside
    self.hold = torch.tensor([default[n] for n in driven], device=dev)
    n = self.num_envs
    self.handed = torch.zeros(n, dtype=torch.bool, device=dev)
    self.still = torch.zeros(n, device=dev)
    self.last = torch.zeros(n, len(legs), device=dev)

  def _stand_act(self) -> torch.Tensor:
    data = self._entity.data
    q, qd = data.joint_pos[:, self.seen], data.joint_vel[:, self.seen]
    obs = torch.cat([data.root_link_ang_vel_b, data.projected_gravity_b, q - self.seen_default, qd,
                     self.last, torch.zeros(self.num_envs, 2, device=self.device)], dim=-1)
    x = (obs - self.mean) / self.div
    for w, b in self.layers[:-1]:
      x = torch.nn.functional.elu(x @ w + b)
    w, b = self.layers[-1]
    return x @ w + b

  def fallen(self) -> torch.Tensor:
    """Where the body lies as a fall the served watch would take
    (`RETAKE_DEG`, `RETAKE_M`)."""
    up, z = _up(self._env), _z(self._env)
    return (up < math.cos(math.radians(RETAKE_DEG))) | (z < RETAKE_M)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    up, z = _up(self._env), _z(self._env)
    stood = (up > STOOD_UP) & ((z - robot.BODY["stand_height"]).abs() < HAND_BACK_M)
    fell = self.fallen()
    retaken = self.handed & fell
    if bool(retaken.any()):
      # ...a take's first action charged as from none, as it was seen
      # (`retake_last_action`): what it put out while handed back moved nothing
      self._env.action_manager._prev_action[retaken] = 0.0
    self.handed &= ~fell
    self.still = torch.where(stood & ~self.handed, self.still + self._env.step_dt,
                             torch.zeros_like(self.still))
    took = ~self.handed & (self.still >= HAND_BACK_S)
    self.last[took] = 0.0
    self.handed |= took
    if bool(self.handed.any()):
      act = self._stand_act()
      self.last = torch.where(self.handed[:, None], act, self.last)
      target = self.hold.expand(self.num_envs, -1).clone()
      target[:, self.legs] = self.leg_default + self.leg_scale * act
      self._processed_actions = torch.where(self.handed[:, None], target, self._processed_actions)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.handed[ids] = False
    self.still[ids] = 0.0
    self.last[ids] = 0.0


def retake_last_action(env) -> torch.Tensor:
  """The last action, none where the get-up is about to take a fall back
  from the standing policy (`HandBackAction`): the served get-up begins a
  take seeing none (`GetupDriver.begin`). ⚠ Read where the retake is
  decided it came a step late: the take's first action was chosen seeing
  the last it had put out while handed back, and the next saw none."""
  action = env.action_manager.action
  term = env.action_manager.get_term("joint_pos")
  if not isinstance(term, HandBackAction):
    return action
  pending = term.handed & term.fallen()
  return torch.where(pending[:, None], torch.zeros_like(action), action)


def handed_back(env) -> torch.Tensor:
  """1 where the standing policy has the legs (`HandBackAction`)."""
  term = env.action_manager.get_term("joint_pos")
  return getattr(term, "handed", torch.zeros(env.num_envs, device=env.device))[:, None].float()


def getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}
  cfg.episode_length_s = EPISODE_S
  cfg.sim.njmax = 600
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  terrain = ContactMatch(mode="body", pattern="terrain")
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=robot.SOLES, entity="robot"),
      secondary=terrain, fields=("found", "force"), reduce="netforce", num_slots=1),
    ContactSensorCfg(
      name="struck",
      primary=ContactMatch(mode="geom", pattern=STRUCK, entity="robot"),
      secondary=terrain, fields=("found", "force"), reduce="netforce", num_slots=1,
      history_length=robot.DECIMATION),
  )

  # Nothing to track: the twist stays at zero for the stock terms that read it.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = JOINTS
  action.preserve_order = True
  action.scale = dict(ACTION_SCALE)

  actor = cfg.observations["actor"].terms
  for name in ("base_lin_vel", "command", "height_scan"):
    del actor[name]
  critic = cfg.observations["critic"].terms
  for name in ("command", "height_scan", "foot_height", "foot_air_time", "foot_contact",
               "foot_contact_forces"):
    critic.pop(name, None)
  # ⚠ The joints it SEES are the ones it drives, as the served get-up reads
  # them (`policy.joint_names`): over all the body's, the waist (#560's
  # stage (c3)) made 80 inputs, and getup3's 78 no longer carried on.
  seen = SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)
  for terms in (actor, critic):
    for name in ("joint_pos", "joint_vel"):
      terms[name].params = {**terms[name].params, "asset_cfg": seen}
  driven = SceneEntityCfg("robot", actuator_names=JOINTS, preserve_order=True)
  peak = torch.tensor([robot.JOINTS[j]["peak_torque"] for j in JOINTS])
  rated = torch.tensor([robot.JOINTS[j]["rated_torque"] for j in JOINTS])
  critic["root_height"] = ObservationTermCfg(func=root_height)
  critic["since_release"] = ObservationTermCfg(func=since_release)
  critic["joint_torque"] = ObservationTermCfg(
    func=joint_torque, params={"asset_cfg": driven, "peak": peak})

  joints = SceneEntityCfg("robot", joint_names=(".*",))
  events = {k: cfg.events[k] for k in ("foot_friction", "encoder_bias", "base_com")}
  events["foot_friction"].params["asset_cfg"].geom_names = robot.SOLES
  events["base_com"].params["asset_cfg"].body_names = (robot.ROOT,)
  events["rotor_inertia"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.8, 1.2)})
  events["joint_friction"] = EventTermCfg(
    mode="startup", func=dr.joint_friction,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.5, 2.0)})
  events["body_mass"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,)),
            "operation": "scale", "ranges": (0.9, 1.1)})
  events["effort_limit"] = EventTermCfg(
    mode="startup", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale", "effort_limit_range": (0.9, 1.1)})
  events["reset_base"] = EventTermCfg(func=reset_fallen, mode="reset")
  events["assist"] = EventTermCfg(
    func=assist, mode="reset",
    params={"share": ASSIST_STAGES[0][1],
            "asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,))})
  cfg.events = events

  share = GENTLE_STAGES[0][1]
  gentle = {
    "torque_over_rating": (torque_over_rating, {"asset_cfg": driven, "rated": rated}),
    "torque_over_cap": (torque_over_cap, {"asset_cfg": driven, "cap": CAP_SHARE * peak}),
    "joint_vel_l2": (envs_mdp.rewards.joint_vel_l2, {}),
    "joint_acc_l2": (envs_mdp.rewards.joint_acc_l2, {}),
    "rise_speed": (rise_speed, {"free": RISE_FREE_MPS}),
    "action_rate_l2": (mdp.action_rate_l2, {}),
    "impact": (impact, {"sensor_name": "struck", "free": IMPACT_FREE}),
  }
  assert gentle.keys() == GENTLE.keys()
  cfg.rewards = {
    "upright": RewardTermCfg(func=upright, weight=1.0),
    "rise": RewardTermCfg(func=rise, weight=1.0),
    "stand_height": RewardTermCfg(func=stand_height, weight=1.0,
                                  params={"std": 0.05, "budget_s": BUDGET_S}),
    "stand_pose": RewardTermCfg(func=stand_pose, weight=0.5,
                                params={"std": 0.8, "budget_s": BUDGET_S}),
    "feet_down": RewardTermCfg(func=feet_down, weight=0.5,
                               params={"sensor_name": "feet_ground_contact",
                                       "budget_s": BUDGET_S}),
    **{name: RewardTermCfg(func=func, weight=GENTLE[name] * share, params=params)
       for name, (func, params) in gentle.items()},
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }
  cfg.curriculum = {**gentle_curriculum(),
                    "assist": CurriculumTermCfg(func=assist_curriculum, params={})}
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan": TerminationTermCfg(func=nan_detection),
    "blown_up": TerminationTermCfg(func=blown_up),
  }
  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["assist"].params["share"] = 0.0
  return cfg


def getup_head_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Getup-Head`: the get-up carried on, falling without its head
  (`HEAD_FREE`): no lift and every charge at its full weight from the
  start, as the get-up ended."""
  cfg = getup_env_cfg(play)
  cfg.scene.sensors = cfg.scene.sensors + (
    ContactSensorCfg(
      name="head_struck", primary=ContactMatch(mode="geom", pattern=HEAD, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
      reduce="netforce", num_slots=1, history_length=robot.DECIMATION),
  )
  cfg.rewards["head_struck"] = RewardTermCfg(
    func=impact, weight=HEAD_WEIGHT, params={"sensor_name": "head_struck", "free": HEAD_FREE})
  for name, w in GENTLE.items():
    cfg.rewards[name].weight = w
  cfg.curriculum = {}
  cfg.events["assist"].params["share"] = 0.0
  cfg.events["reset_base"].params["starts"] = dict(HEAD_STARTS)
  cfg.metrics = {**cfg.metrics, **{
    name: MetricsTermCfg(func=func, reduce="last")
    for name, func in (("stood_beside", stood_beside), ("stood_open", stood_open),
                       ("beside", beside))}}
  return cfg


def getup_wall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Getup-Wall` (#679): `-Getup-Head` with the box beside the fallen
  body in a share of its episodes (`OBSTACLE`), its hits charged as the
  floor's -- carried on from getup3 (`--agent.resume True --agent.load-run
  '.*_getup3$'`): its inputs are the same 78."""
  cfg = getup_head_env_cfg(play)
  cfg.scene.entities["obstacle"] = EntityCfg(
    spec_fn=obstacle_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, PARKED_TOP_M - OBSTACLE_HALF[2])))
  box = ContactMatch(mode="body", pattern="obstacle", entity="obstacle")
  cfg.scene.sensors = cfg.scene.sensors + tuple(
    ContactSensorCfg(name=name, primary=ContactMatch(mode="geom", pattern=parts, entity="robot"),
                     secondary=box, fields=("found", "force"), reduce="netforce", num_slots=1,
                     history_length=robot.DECIMATION)
    for name, parts in (("struck_obstacle", STRUCK), ("head_struck_obstacle", HEAD))) + (
    ContactSensorCfg(name="against_obstacle",
                     primary=ContactMatch(mode="subtree", pattern=robot.ROOT, entity="robot"),
                     secondary=box, fields=("found",), reduce="netforce", num_slots=1),)
  for name in ("stand_height", "stand_pose"):
    cfg.rewards[name].params.update(upright=STOOD_UP, free="against_obstacle")
  cfg.rewards["feet_down"].params["free"] = "against_obstacle"
  # ⚠ Lying still beside the box cost nothing but the stand it forwent, and
  # fine-tuned (wall5, 750 iterations) the mean action lay still on its back
  # beside it in 136 of 256 starts that getup3 stood from in 252.
  cfg.rewards["down"] = RewardTermCfg(func=down, weight=DOWN_WEIGHT,
                                      params={"budget_s": BUDGET_S, "below_m": DOWN_M})
  cfg.rewards["impact_obstacle"] = RewardTermCfg(
    func=impact, weight=GENTLE["impact"],
    params={"sensor_name": "struck_obstacle", "free": IMPACT_FREE})
  cfg.rewards["head_struck_obstacle"] = RewardTermCfg(
    func=impact, weight=HEAD_WEIGHT,
    params={"sensor_name": "head_struck_obstacle", "free": HEAD_FREE})
  cfg.events["reset_base"].params["obstacle"] = copy.deepcopy(OBSTACLE)
  cfg.events["reset_base"].params["starts"] = dict(WALL_STARTS)
  if not play:
    cfg.episode_length_s = WALL_EPISODE_S
  return cfg


def getup_wall2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Getup-Wall2` (#679's second round): `-Getup-Wall` with the bodies
  posed where its policy was seen to end (`POSES`: three sitting, one propped
  on a wall) among its starts, and a wall FAR in the way a body tips
  (`OBSTACLE2`) -- carried on from getupwall10 (`--agent.load-run
  '.*_getupwall10$' --agent.load-checkpoint model_8450.pt`)."""
  cfg = getup_wall_env_cfg(play)
  cfg.events["reset_base"].params["obstacle"] = copy.deepcopy(OBSTACLE2)
  cfg.events["reset_base"].params["starts"] = dict(WALL2_STARTS)
  return cfg


def getup_wall3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Getup-Wall3` (#679's third round): `-Getup-Wall2` handing back to the
  served standing policy (`HandBackAction`), and paid while it has -- carried
  on from getupwall12 (`--agent.load-run '.*_getupwall12$'
  --agent.load-checkpoint model_9050.pt`)."""
  cfg = getup_wall2_env_cfg(play)
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = HandBackActionCfg(
    **{f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init})
  for group in ("actor", "critic"):
    cfg.observations[group].terms["actions"].func = retake_last_action
  # Handed back is paid and waiting charged, the stand paid only handed back,
  # and the get-up's own charges charged only while it holds the legs (handed
  # back, they read the standing policy's stepping): ⚠ any of the three
  # missing, it learned to stand just outside the hand-back, which served is
  # the `stuck` death at 20 s (SimNotes, "The wall get-up's second and third
  # rounds"). The critic is not told: a 108th input would not load getupwall12's.
  cfg.rewards["handed_back"] = RewardTermCfg(func=lambda env: handed_back(env)[:, 0],
                                             weight=HANDED_WEIGHT)
  cfg.rewards["not_handed"] = RewardTermCfg(
    func=lambda env: _paid(env, BUDGET_S) * (1.0 - handed_back(env)[:, 0]), weight=-HANDED_WEIGHT)
  for name in ("stand_height", "stand_pose", "feet_down"):
    cfg.rewards[name].func = _handed_only(cfg.rewards[name].func)
  for name in GENTLE:
    if name != "impact":
      cfg.rewards[name].func = _handed_only(cfg.rewards[name].func, handed=False)
  return cfg


def _handed_only(func, handed: bool = True):
  """`func`'s pay only while the standing policy has the legs (or, with
  `handed` False, only while it does not)."""
  def paid(env, **params):
    on = handed_back(env)[:, 0]
    return func(env, **params) * (on if handed else 1.0 - on)
  paid.__name__ = f"{func.__name__}_{'handed' if handed else 'unhanded'}"
  return paid
