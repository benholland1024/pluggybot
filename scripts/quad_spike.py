"""The quadruped body spike (issue #377): size it, then measure what it costs.

The body is `pluggybot.legs.model.BodySpec`, its actuators
`pluggybot.legs.actuator`, and until the walking policy is trained the gait
is `pluggybot.legs.scripted.VirtualModel` -- a measuring instrument, not the
robot's gait. Every table here is SimNotes, "The quadruped body". The
torque-driven tables (the scripted gait's) fly #377's sizing premise,
`model.SIZING`: the arm a placeholder on the robot's back. The policy
flights fly the chosen body, its arm held stowed (`PolicyDriver.step`).

  --view        the robot in the MuJoCo viewer, cycling through stand,
                crouch, lie down, walk, trot, fast trot and a turn, six
                seconds each (ENTER skips ahead; every letter is already one
                of the viewer's own keys); the terminal says which.
                `--pupper` shows the small body carrying the suite.
  --torque      the torque table: peak and RMS torque per joint against the
                actuator's peak and continuous ratings, for each activity
  --sweep       the design sweep behind the choice: knee belt ratio x leg
                length, the knee's worst row
  (default)     a filmstrip of the activities, quad_spike.png

Usage:
  MUJOCO_GL=egl uv run python scripts/quad_spike.py [--torque|--sweep]
  uv run python scripts/quad_spike.py --view
"""

import os

# One BLAS thread before numpy loads: the policy's matrix products are small
# enough that six threads DOUBLED a step (238 against 119 us, measured under
# load). Not for determinism: a flight hashes identical at 1 and 6 threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.legs.actuator import BUS_V_NOMINAL, JointLimits  # noqa: E402
from pluggybot.legs.model import (CHOSEN, ELECTRONICS_W, LEGS, PUPPER_CLASS, SIZING,  # noqa: E402
                                  PUPPER_WITH_SUITE, BodySpec, attachable, body_xml,
                                  pose_qpos)
from pluggybot.legs.odometry import LegOdometry  # noqa: E402
from pluggybot.legs.policy import POLICY_NPZ, PolicyDriver, Twist, WalkingPolicy  # noqa: E402
from pluggybot.legs.scan import MapScan  # noqa: E402
from pluggybot.legs.scripted import Command, VirtualModel, _quat_rpy  # noqa: E402
from pluggybot.legs import posture as moves  # noqa: E402

#: The climbs: a curb and a house stair's riser (#280's world builds both).
CURB_M, RISER_M = 0.12, 0.18
#: The step's edge, in the world's x; the push-up poses straddle it.
STEP_EDGE_X = 0.0
#: A push up a riser starts with the hips this fraction of the standing
#: height over their own feet, and rises to the stand in `PUSH_S`.
PUSH_LOW, PUSH_S = 0.62, 0.6
#: Seconds each activity is flown, and how much of the start is settling.
FLY_S, SETTLE_S = 5.0, 1.5
#: A push up a riser: a hold, the push, a hold.
PUSH_FLY_S, PUSH_SETTLE_S = 2.0, 0.3
#: A deep crouch, as a fraction of the standing height.
CROUCH = 0.46
#: The "peak" torque is this percentile of the samples (the max is kept).
PEAK_PCT = 99.5
#: The pack's voltage the tables fly at; `--bus` changes it (an empty 12S
#: pack is 36 V, and the motor is a fifth slower there).
BUS_V = BUS_V_NOMINAL


@dataclass
class Activity:
  name: str
  command: object  # t -> Command
  start_height: float | None = None
  #: A step of this height across the robot's path (0: flat).
  step: float = 0.0
  arm_reach: bool = False


def activities(spec: BodySpec) -> list[Activity]:
  h0 = spec.stand_height
  lie = spec.belly_depth
  crouch = round(CROUCH * h0, 3)

  def rise(t):
    s = min(max((t - 1.0) / 1.5, 0.0), 1.0)
    return Command(height=lie + (h0 - lie) * (0.5 - 0.5 * math.cos(math.pi * s)))

  # Swing heights scale with the leg and the period with its square root
  # (Froude); the speeds and the obstacles are the world's.
  k = spec.leg_length / CHOSEN.leg_length
  kt = math.sqrt(k)

  low = PUSH_LOW * h0

  def push(t):
    return Command(height=low + _ease((t - 0.5) / PUSH_S) * (h0 - low))

  def trot(v, period):
    return lambda t: Command(gait="trot", period=period * kt,
                             swing_height=0.08 * k,
                             vx=v * min(max(t - 0.5, 0.0) / 1.0, 1.0))

  return [
    Activity("stand", lambda t: Command()),
    Activity("stand, arm at full reach", lambda t: Command(), arm_reach=True),
    Activity(f"deep crouch ({crouch:.2f} m)", lambda t: Command(height=crouch),
             start_height=crouch),
    Activity("stand up from lying", rise, start_height=lie),
    Activity("walk: trot 0.3 m/s", trot(0.3, 0.4)),
    Activity("trot 1.0 m/s", trot(1.0, 0.35)),
    Activity("trot 1.5 m/s", trot(1.5, 0.3)),
    Activity(f"push up a {CURB_M:.2f} m curb", push, step=CURB_M),
    Activity(f"push up a {RISER_M:.2f} m riser", push, step=RISER_M),
  ]


def step_scenery(height: float) -> str:
  return (f'\n    <geom name="step" type="box" size="1.0 1.0 {height / 2}" '
          f'pos="{STEP_EDGE_X + 1.0} 0 {height / 2}" rgba="0.6 0.5 0.4 1"/>')


def terrain_of(act: "Activity"):
  if not act.step:
    return None
  return lambda x, y: act.step if x >= STEP_EDGE_X else 0.0


def solve_pose(model, data, root_qpos, feet_world, iters: int = 60) -> None:
  """Joint angles putting each foot site at `feet_world[leg]`, the torso held
  at `root_qpos` (7 numbers): damped least squares on MuJoCo's kinematics."""
  data.qpos[:7] = root_qpos
  jac = np.zeros((3, model.nv))
  for _ in range(iters):
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    for i, leg in enumerate(LEGS):
      sid = model.site(f"{leg}_foot").id
      err = feet_world[leg] - data.site_xpos[sid]
      mujoco.mj_jacSite(model, data, jac, None, sid)
      cols = 6 + 3 * i + np.arange(3)
      j = jac[:, cols]
      dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), err)
      data.qpos[7 + 3 * i:10 + 3 * i] += dq
  mujoco.mj_forward(model, data)


def build(spec: BodySpec, act: Activity):
  """Model and data at the activity's start pose. A push up a riser starts
  straddling it: the front feet on the step, the hind feet on the floor, each
  under its hip, the torso pitched along the line between and crouched."""
  spec = spec.with_(arm_reach=act.arm_reach)
  scenery = step_scenery(act.step) if act.step else ""
  model = mujoco.MjModel.from_xml_string(
    body_xml(spec, scenery=scenery, drive="torque"))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  if act.start_height is not None:
    data.qpos[2] = act.start_height
    data.qpos[7:19] = pose_qpos(spec, act.start_height)
  if act.step:
    pitch = -math.atan2(act.step, 2 * spec.hip_x)
    root = [STEP_EDGE_X, 0.0, act.step / 2 + PUSH_LOW * spec.stand_height,
            math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    data.qpos[:7] = root
    data.qpos[7:19] = pose_qpos(spec, PUSH_LOW * spec.stand_height)
    mujoco.mj_kinematics(model, data)
    feet = {}
    for leg in LEGS:
      hip = data.xpos[model.body(f"{leg}_thigh").id]
      ground = act.step if leg[0] == "F" else 0.0
      feet[leg] = np.array([hip[0], hip[1], ground + spec.foot_r])
    solve_pose(model, data, root, feet)
  mujoco.mj_forward(model, data)
  return spec, model, data


def fly(spec: BodySpec, act: Activity, seconds: float | None = None,
        viewer_hook=None):
  """Fly one activity under the scripted gait and the honest actuator
  envelope. Returns (torques, joint speeds, fell) after the settle."""
  spec, model, data = build(spec, act)
  vm = VirtualModel(model, data, spec, terrain=terrain_of(act))
  lim = JointLimits.of(spec.motor, spec.knee_ratio, BUS_V)
  taus, qds, stance = [], [], []
  fell = False
  seconds = seconds or (PUSH_FLY_S if act.step else FLY_S)
  settle = PUSH_SETTLE_S if act.step else SETTLE_S
  for _ in range(int(seconds / model.opt.timestep)):
    t = float(data.time)
    cmd = act.command(t)
    tau = vm.torque(cmd)
    data.ctrl[:] = lim.clip(tau, data.qvel[vm.vadr])
    mujoco.mj_step(model, data)
    if viewer_hook:
      viewer_hook(model, data)
    if t >= settle:
      taus.append(data.ctrl.copy())
      qds.append(data.qvel[vm.vadr].copy())
      stance.append(np.repeat(vm.stance_now, 3))
    roll, p, _ = _quat_rpy(data.qpos[3:7])
    if data.qpos[2] < 0.05 or abs(roll) > 1.0 or abs(p) > 1.0:
      fell = True
      break
  # A push that did not fall but never stood is not a climb.
  if act.step and not fell:
    fell = data.qpos[2] < act.step / 2 + 0.95 * spec.stand_height
  return np.array(taus), np.array(qds), np.array(stance), fell


def _by_joint(x: np.ndarray, fn) -> np.ndarray:
  """(abd, flex, knee) of `fn` over every sample and all four legs."""
  return np.array([fn(x[:, j::3]) for j in range(3)])


def torque_table(spec: BodySpec = SIZING) -> list[dict]:
  lim = JointLimits.of(spec.motor, spec.knee_ratio, BUS_V)
  rows = []
  for act in activities(spec):
    tau, qd, stance, fell = fly(spec, act)
    if len(tau) == 0:  # down before the measurement window opened
      nan = np.full(3, np.nan)
      rows.append({"activity": act.name, "fell": True, "peak": nan,
                   "peak_max": nan, "swing_peak": nan, "rms": nan,
                   "speed": nan, "speed_margin": nan, "peak_margin": nan,
                   "rms_margin": nan})
      continue
    load = np.where(stance, np.abs(tau), 0.0)
    swing = np.where(stance, 0.0, np.abs(tau))
    # 99.5th percentile over every sample of all four legs: a touchdown
    # spike lasts a step or two and says more about the instrument's
    # footfall than the joint's load; `max` keeps it visible.
    peak = _by_joint(load, lambda a: float(np.percentile(a, PEAK_PCT)))
    peak_max = _by_joint(load, lambda a: float(a.max()))
    swing_peak = _by_joint(swing, lambda a: float(np.percentile(a, PEAK_PCT)))
    rms = _by_joint(tau, lambda a: float(np.sqrt((a ** 2).mean(axis=0)).max()))
    speed = _by_joint(np.abs(qd), lambda a: float(a.max()))
    rows.append({"activity": act.name, "fell": fell, "peak": peak,
                 "peak_max": peak_max,
                 "swing_peak": swing_peak, "rms": rms, "speed": speed,
                 "speed_margin": 1 - speed / lim.noload[:3],
                 "peak_margin": 1 - peak / lim.peak[:3],
                 "rms_margin": 1 - rms / lim.rated[:3]})
  return rows


def print_torque_table(spec: BodySpec = SIZING) -> None:
  lim = JointLimits.of(spec.motor, spec.knee_ratio, BUS_V)
  print(f"{spec.name}: {spec.mass:.2f} kg, {spec.motor.name}, knee belt "
        f"{spec.knee_ratio}:1, thigh/shank {spec.thigh}/{spec.shank} m")
  print(f"ratings at the joint (abd, flex, knee): peak "
        f"{lim.peak[:3].round(1)} N*m, continuous {lim.rated[:3].round(1)} N*m")
  print(f"{'activity':28s} {'stance p99.5 a/f/k':>18s} {'max knee':>8s} "
        f"{'swing p99.5 a/f/k':>18s} "
        f"{'worst-leg RMS a/f/k':>20s} {'knee margin: peak, RMS':>24s} "
        f"{'fastest joint, % no-load':>25s}")
  for r in torque_table(spec):
    flag = ("  DID NOT RISE" if "push" in r["activity"] else "  FELL") if r["fell"] else ""
    print(f"{r['activity']:28s} {'/'.join(f'{v:4.1f}' for v in r['peak']):>18s} "
          f"{r['peak_max'][2]:8.1f} "
          f"{'/'.join(f'{v:4.1f}' for v in r['swing_peak']):>18s} "
          f"{'/'.join(f'{v:4.1f}' for v in r['rms']):>20s} "
          f"{r['peak_margin'][2] * 100:12.0f} % {r['rms_margin'][2] * 100:7.0f} % "
          f"{(r['speed'] / lim.noload[:3]).max() * 100:17.0f} %{flag}")


def power_row(spec: BodySpec, act: Activity) -> dict:
  """Mean electrical draw of one activity, W, split by where it goes."""
  lim = JointLimits.of(spec.motor, spec.knee_ratio)
  tau, qd, _, fell = fly(spec, act)
  mech = tau * qd
  copper = lim.copper_w(tau)
  return {
    "activity": act.name, "fell": fell,
    # Heat in the windings: what holding a pose costs.
    "copper_w": float(copper.sum(axis=1).mean()),
    # Shaft work, the positive half only: no credit for regeneration,
    # because a driver that can return it to the pack is not assumed.
    "mech_w": float(np.clip(mech, 0, None).sum(axis=1).mean()),
    "regen_w": float(-np.clip(mech, None, 0).sum(axis=1).mean()),
    # The worst winding's mean heat, for the thermal model.
    "worst_copper_w": float(copper.mean(axis=0).max()),
    "joint_copper_w": copper.mean(axis=0),
  }


#: The fold's PD and crouch, and the two moves, are the body's own
#: (`legs/posture.py`, issue #387); the names stay here for the tables.
FOLD_KP, FOLD_KD, FOLD_H = moves.FOLD_KP, moves.FOLD_KD, moves.FOLD_H


class Meter:
  """Steps the physics through the actuator envelope and keeps the bill."""

  def __init__(self, spec: BodySpec, model, data):
    self.lim = JointLimits.of(spec.motor, spec.knee_ratio)
    self.m, self.d = model, data
    self.copper_j = self.mech_j = 0.0
    self.peak = np.zeros(12)

  def step(self, tau: np.ndarray) -> None:
    """A torque-drive step: the envelope here, then the bill."""
    d = self.d
    t = self.lim.clip(tau, d.qvel[6:18])
    d.ctrl[:] = t
    mujoco.mj_step(self.m, d)
    self._bill(t)

  def bill(self) -> None:
    """After a position-drive step: what the drivers applied."""
    self._bill(self.d.actuator_force[:12].copy())

  def _bill(self, t: np.ndarray) -> None:
    dt = self.m.opt.timestep
    self.copper_j += float(self.lim.copper_w(t).sum()) * dt
    self.mech_j += float(np.clip(t * self.d.qvel[6:18], 0, None).sum()) * dt
    np.maximum(self.peak, np.abs(t), out=self.peak)

  def pd(self, target: np.ndarray) -> np.ndarray:
    return FOLD_KP * (target - self.d.qpos[7:19]) - FOLD_KD * self.d.qvel[6:18]

  @property
  def wh(self) -> float:
    return (self.copper_j + self.mech_j) / 3600


_ease = moves.ease


def lie_down_routine(spec: BodySpec, data, vm: VirtualModel,
                     lower_s: float = 1.2, fold_s: float = 1.0):
  """Stand -> belly on the standalone model (`legs.posture.lie_down_routine`)."""
  return moves.lie_down_routine(spec, data, vm, moves.Joints.of(vm.m),
                                lower_s, fold_s)


def stand_up_routine(spec: BodySpec, data, vm: VirtualModel,
                     fold_s: float = 1.0, rise_s: float = 1.2):
  """Belly -> stand on the standalone model (`legs.posture.stand_up_routine`)."""
  return moves.stand_up_routine(spec, data, vm, moves.Joints.of(vm.m),
                                fold_s, rise_s)


def _metered(spec: BodySpec, key: int, routine):
  model = mujoco.MjModel.from_xml_string(body_xml(spec, drive="torque"))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, key)
  mujoco.mj_forward(model, data)
  meter = Meter(spec, model, data)
  vm = VirtualModel(model, data, spec)
  for tau in routine(spec, data, vm):
    meter.step(tau)
  return model, data, meter


def stand_up(spec: BodySpec):
  """Returns (seconds, Wh in the legs, peak torques, stood)."""
  _, data, meter = _metered(spec, 1, stand_up_routine)
  stood = abs(data.qpos[2] - spec.stand_height) < 0.03
  return data.time, meter.wh, meter.peak, stood


def lie_down(spec: BodySpec):
  """Returns (seconds, Wh in the legs, peak torques, resting on the belly)."""
  _, data, meter = _metered(spec, 0, lie_down_routine)
  resting = data.qpos[2] < spec.belly_depth + 0.01
  return data.time, meter.wh, meter.peak, resting


#: Ambient air for the thermal table, C: a warm room; `--thermal` also
#: flies a hot day.
AMBIENT_C = (30.0, 40.0)


def energy_table(spec: BodySpec = SIZING) -> None:
  """Watts per activity, and what a transition costs."""
  base = sum(ELECTRONICS_W.values())
  print(f"electronics, always on: {base:.1f} W "
        + ", ".join(f"{k} {v:g}" for k, v in ELECTRONICS_W.items()))
  print(f"{'activity':28s} {'windings W':>10s} {'shaft W':>8s} {'regen W':>8s} "
        f"{'total W':>8s} {'Wh/h':>6s}")
  print(f"{'lying (drivers holding 0)':28s} {0:10.1f} {0:8.1f} {0:8.1f} "
        f"{base:8.1f} {base:6.1f}")
  keep = ("stand", "stand, arm at full reach", "walk: trot 0.3 m/s",
          "trot 1.0 m/s", "trot 1.5 m/s")
  rows = {}
  for act in activities(spec):
    if act.name not in keep:
      continue
    r = power_row(spec, act)
    total = base + r["copper_w"] + r["mech_w"]
    rows[act.name] = total
    print(f"{act.name:28s} {r['copper_w']:10.1f} {r['mech_w']:8.1f} "
          f"{r['regen_w']:8.1f} {total:8.1f} {total:6.1f}")
  t_up, wh_up, _, _ = stand_up(spec)
  t_dn, wh_dn, _, _ = lie_down(spec)
  cycle_wh = wh_up + wh_dn + base * (t_up + t_dn) / 3600
  bleed = rows["stand"] - base
  print(f"stand up {t_up:.1f} s {wh_up * 1000:.1f} mWh (legs); lie down "
        f"{t_dn:.1f} s {wh_dn * 1000:.1f} mWh; the round trip with the "
        f"electronics {cycle_wh * 1000:.1f} mWh")
  print(f"standing costs {bleed:.1f} W more than lying: a wait longer than "
        f"{cycle_wh * 3600 / bleed:.1f} s is cheaper lying down "
        f"(ignoring the {t_up:.1f} s it then takes to stand)")


def thermal_table(spec: BodySpec = SIZING) -> None:
  """Each activity SUSTAINED: the hottest winding's steady temperature.

  A first-order winding (`Motor.thermal_r`, `thermal_c`) driven by
  non-negative heat never passes the highest steady state among the
  activities it runs through, so this table bounds ANY day built from
  them; the time constant is R*C (about 40 s), so an activity held for
  a few minutes reaches its row."""
  m = spec.motor
  tau_s = m.thermal_r * m.thermal_c
  print(f"{m.name}: {m.thermal_r} K/W, {m.thermal_c} J/K (tau {tau_s:.0f} s), "
        f"limit {m.max_winding_c:.0f} C; ambient {AMBIENT_C[0]:.0f} / "
        f"{AMBIENT_C[1]:.0f} C")
  print(f"{'activity':28s} {'hottest winding W':>17s} {'joint':>12s} "
        f"{'steady C':>9s} {'hot day C':>9s} {'margin K':>8s}")
  names = [f"{leg}_{j}" for leg in LEGS for j in ("abd", "flex", "knee")]
  for act in activities(spec):
    r = power_row(spec, act)
    w = r["joint_copper_w"]
    k = int(np.argmax(w))
    t = [a + w[k] * m.thermal_r for a in AMBIENT_C]
    print(f"{act.name:28s} {w[k]:17.1f} {names[k]:>12s} {t[0]:9.1f} "
          f"{t[1]:9.1f} {m.max_winding_c - t[1]:8.1f}")


#: A Livox Mid-360 frame: 360 deg by -7..52 deg, 200 000 points a second at
#: 10 Hz (maker). The sim casts a fraction of them; the table prices each.
MID360_VFOV = (-7.0, 52.0)


def ray_cost(world: str = "models/home_world.xml") -> None:
  """What a scan costs the physics thread, in the home world: the 2D LIDAR
  (360 rays in one `mj_multiRay`, as `perception/lidar.py` casts them, the
  noise drawn ray by ray) against a 3D LIDAR frame of N rays, also one
  `mj_multiRay`."""
  from pluggybot.perception.lidar import Lidar
  model = mujoco.MjModel.from_xml_path(world)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  lidar = Lidar(model)
  t = time.perf_counter()
  for _ in range(50):
    lidar.scan(data)
  two_d = (time.perf_counter() - t) / 50
  print(f"{'scan':34s} {'rays':>7s} {'ms a scan':>9s} {'ms per sim s at 10 Hz':>22s}")
  print(f"{'2D LIDAR (lidar.py)':34s} {360:7d} {two_d * 1e3:9.2f} "
        f"{two_d * 1e4:22.1f}")
  pos = data.site("lidar").xpos.copy() if model.nsite and any(
    model.site(i).name == "lidar" for i in range(model.nsite)) else data.qpos[:3] + [0, 0, 0.45]
  for n in (4000, 10000, 20000):
    rows = max(8, round(math.sqrt(n * (MID360_VFOV[1] - MID360_VFOV[0]) / 360)))
    cols = n // rows
    az = np.linspace(-math.pi, math.pi, cols, endpoint=False)
    el = np.radians(np.linspace(*MID360_VFOV, rows))
    a, e = np.meshgrid(az, el)
    dirs = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                     np.sin(e)], axis=-1).reshape(-1, 3)
    dist = np.zeros(len(dirs))
    geomid = np.zeros(len(dirs), dtype=np.int32)
    body = model.body("pluggybot").id
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
      mujoco.mj_multiRay(model, data, pos, dirs.reshape(-1), None, 1, body,
                         geomid, dist, None, len(dirs), 40.0)
    ms = (time.perf_counter() - t) / reps * 1e3
    print(f"{'3D LIDAR frame (mj_multiRay)':34s} {len(dirs):7d} {ms:9.2f} "
          f"{ms * 10:22.1f}")


def _driver(model, data, policy: WalkingPolicy, prefix: str = "") -> PolicyDriver:
  """A policy on one body, its drivers on the pack the tables fly (`--bus`)."""
  drv = PolicyDriver(model, data, policy, prefix=prefix)
  drv.set_bus(BUS_V)
  return drv


def quad_pair_world(policy: WalkingPolicy):
  """The home world with its rover taken out and two quadrupeds in, 1.5 m
  apart, each standing; returns (model, data, drivers)."""
  spec = mujoco.MjSpec.from_file("models/home_world.xml")
  for el in (list(spec.actuators) + list(spec.sensors) + list(spec.tendons)
             + list(spec.equalities) + list(spec.excludes)):
    spec.delete(el)
  spec.delete(spec.body("pluggybot"))
  for prefix, x in (("", 0.0), ("r2_", 1.5)):
    frame = spec.worldbody.add_frame()
    frame.pos = [x, 0.0, 0.0]
    spec.attach(attachable(CHOSEN), prefix=prefix, frame=frame)
  model = spec.compile()
  data = mujoco.MjData(model)
  drivers = []
  for prefix in ("", "r2_"):
    root = model.jnt_qposadr[model.joint(f"{prefix}pluggybot_root").id]
    data.qpos[root + 2] = CHOSEN.stand_height
    j = model.jnt_qposadr[model.joint(f"{prefix}FL_hip_abd").id]
    data.qpos[j:j + 12] = pose_qpos(CHOSEN, CHOSEN.stand_height)
  mujoco.mj_forward(model, data)
  for prefix in ("", "r2_"):
    drivers.append(_driver(model, data, policy, prefix=prefix))
  return model, data, drivers


def served_cost(path=POLICY_NPZ, sim_s: float = 10.0, rounds: int = 3) -> None:
  """What a pair's physics thread spends per sim second on what the BODY
  changes -- the physics and the body's own controller -- for two rovers
  (their wheel servos are MuJoCo's) against two quadrupeds (the policy in
  numpy, the drivers' PD in MuJoCo's dcmotor), interleaved A B A B (CLAUDE.md: wall clock tracks the
  machine). The sensors' costs are the same rays at the same rates for either
  body, so the ratio scales the rover pair's measured multiple on the box."""
  from pluggybot.robot import world_with_robots
  policy = WalkingPolicy(path)
  rover = world_with_robots("models/home_world.xml", second_at=(1.5, 0.0))
  results = {"rover pair": [], "quadruped pair": []}
  twist = Twist(vx=0.5)
  for _ in range(rounds):
    data = mujoco.MjData(rover)
    mujoco.mj_forward(rover, data)
    n = int(sim_s / rover.opt.timestep)
    t = time.perf_counter()
    for _ in range(n):
      mujoco.mj_step(rover, data)
    results["rover pair"].append((time.perf_counter() - t) / sim_s)
    model, data, drivers = quad_pair_world(policy)
    t = time.perf_counter()
    for _ in range(int(sim_s / model.opt.timestep)):
      for drv in drivers:
        drv.command(twist)
      mujoco.mj_step(model, data)
    results["quadruped pair"].append((time.perf_counter() - t) / sim_s)
  for name, xs in results.items():
    print(f"{name:16s} {np.median(xs) * 1e3:7.1f} ms of wall per sim second "
          f"(median of {rounds}: " + ", ".join(f"{x * 1e3:.0f}" for x in xs) + ")")
  ratio = np.median(results["quadruped pair"]) / np.median(results["rover pair"])
  print(f"quadruped / rover: {ratio:.2f}")


def sweep() -> None:
  """Knee belt ratio x leg length x the unpublished rotor inertia: each
  candidate's worst row per column."""
  lo, hi = CHOSEN.motor.rotor_range
  print(f"{'knee belt':>9s} {'leg':>5s} {'rotor':>7s} {'kg':>5s} {'knee p99.5':>10s} "
        f"{'% peak':>6s} {'worst RMS':>9s} {'% cont.':>7s} "
        f"{'fastest':>7s} {'% no-load':>9s} {'falls':>5s}")
  for kr in (1.0, 1.5):
    for leg in (0.19, 0.21, 0.23):
      for rotor in (lo, hi):
        motor = replace(CHOSEN.motor, rotor_inertia=rotor)
        spec = SIZING.with_(knee_ratio=kr, thigh=leg, shank=leg, motor=motor)
        lim = JointLimits.of(motor, kr, BUS_V)
        rows = torque_table(spec)
        peak = max(r["peak"][2] for r in rows)
        rms = max(r["rms"].max() / 1.0 for r in rows)
        rms_frac = max((r["rms"] / lim.rated[:3]).max() for r in rows)
        spd_frac = max((r["speed"] / lim.noload[:3]).max() for r in rows)
        falls = sum(r["fell"] for r in rows)
        print(f"{kr:9.1f} {leg:5.2f} {rotor:7.1e} {spec.mass:5.2f} {peak:10.1f} "
              f"{peak / lim.peak[2] * 100:5.0f}% {rms:9.1f} {rms_frac * 100:6.0f}% "
              f"{'':7s} {spd_frac * 100:8.0f}% {falls:5d}")


# The viewer cycles through these, DEMO_S each: a name and a routine
# factory (spec, data, vm) -> one torque per physics step, forever.
DEMO_S = 6.0


def _gait(cmd: Command):
  def routine(spec, data, vm):
    vm.reset()
    while True:
      yield vm.torque(cmd)
  return routine


def _then_rest(spec, data, vm):
  yield from lie_down_routine(spec, data, vm)
  while True:  # on the belly, the drivers hold nothing
    yield np.zeros(12)


def _then_stand(spec, data, vm):
  yield from stand_up_routine(spec, data, vm)
  vm.reset()
  while True:
    yield vm.torque(Command())


DEMO = [
  ("stand", _gait(Command())),
  ("deep crouch", _gait(Command(height=CROUCH * CHOSEN.stand_height))),
  ("stand", _gait(Command())),
  ("lie down on the belly, then let go", _then_rest),
  ("stand up from the belly", _then_stand),
  ("walk (trot 0.3 m/s)", _gait(Command(gait="trot", vx=0.3, period=0.4))),
  ("trot 1.0 m/s", _gait(Command(gait="trot", vx=1.0, period=0.35))),
  ("fast trot 1.5 m/s", _gait(Command(gait="trot", vx=1.5, period=0.3))),
  ("turn on the spot", _gait(Command(gait="trot", yaw_rate=0.8, period=0.35))),
]


def view(spec: BodySpec) -> None:
  from mujoco import viewer as mj_viewer
  model = mujoco.MjModel.from_xml_string(body_xml(spec, drive="torque"))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  vm = VirtualModel(model, data, spec)
  lim = JointLimits.of(spec.motor, spec.knee_ratio, BUS_V)
  state = {"i": -1, "since": 0.0, "routine": None}

  def advance():
    state["i"] = (state["i"] + 1) % len(DEMO)
    state["since"] = data.time
    name, factory = DEMO[state["i"]]
    state["routine"] = factory(spec, data, vm)
    print(f"  -> {name}")

  def key(code):
    if code == 257:  # ENTER (GLFW); every letter is one of the viewer's own
      state["skip"] = True

  print(f"{spec.name}: {spec.mass:.2f} kg, stands {spec.stand_height:.3f} m. "
        f"Each activity runs {DEMO_S:.0f} s; ENTER skips. Close the window "
        "to quit.")
  advance()
  with mj_viewer.launch_passive(model, data, key_callback=key) as viewer:
    viewer.cam.distance, viewer.cam.elevation = 1.8, -18
    viewer.cam.trackbodyid = vm.root
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    wall = time.time() - data.time
    while viewer.is_running():
      if data.time - state["since"] >= DEMO_S or state.pop("skip", False):
        advance()
      data.ctrl[:] = lim.clip(next(state["routine"]), data.qvel[vm.vadr])
      mujoco.mj_step(model, data)
      if int(round(data.time / model.opt.timestep)) % 10 == 0:
        viewer.sync()
        ahead = data.time - (time.time() - wall)
        if ahead > 0:
          time.sleep(ahead)


def filmstrip(out: str, spec: BodySpec = SIZING) -> None:
  """Each activity a moment in, then the robot lying on its belly."""
  from PIL import Image, ImageDraw
  tiles = []

  def shoot(model, data, label):
    renderer = mujoco.Renderer(model, 300, 400)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3] - [0, 0, 0.1]
    cam.distance, cam.azimuth, cam.elevation = 1.5, 125, -12
    renderer.update_scene(data, cam)
    img = Image.fromarray(renderer.render())
    renderer.close()
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(img)

  for act in activities(spec):
    spec_a, model, data = build(spec, act)
    vm = VirtualModel(model, data, spec_a, terrain=terrain_of(act))
    lim = JointLimits.of(spec_a.motor, spec_a.knee_ratio, BUS_V)
    for _ in range(int((1.2 if act.step else 2.35) / model.opt.timestep)):
      cmd = act.command(float(data.time))
      data.ctrl[:] = lim.clip(vm.torque(cmd), data.qvel[vm.vadr])
      mujoco.mj_step(model, data)
    shoot(model, data, act.name)
  model, data, _ = _metered(spec, 0, lie_down_routine)
  shoot(model, data, "lying on the belly pack")
  rows = (len(tiles) + 3) // 4
  strip = Image.new("RGB", (400 * 4, 300 * rows), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


#: The policy's flight: (seconds, command), flown in the served sim's
#: physics (2 ms steps), not the trainer's.
POLICY_SCHEDULE = [
  (2.0, Twist()),
  (4.0, Twist(vx=0.5)),
  (4.0, Twist(vx=1.0)),
  (3.0, Twist(yaw_rate=0.8)),
  (3.0, Twist(vy=0.3)),
  (2.0, Twist()),
]


#: The odometry course: ~20 m of straight, arc, turn and sidestep, then back.
ODOMETRY_COURSE = [
  (2.0, Twist()),
  (10.0, Twist(vx=0.6)),
  (8.0, Twist(vx=0.5, yaw_rate=0.4)),
  (4.0, Twist(yaw_rate=0.8)),
  (6.0, Twist(vy=0.3)),
  (10.0, Twist(vx=0.8)),
  (2.0, Twist()),
]


def odometry(path=POLICY_NPZ, seeds: int = 5) -> None:
  """The policy walks the course; the legs and the IMU reckon it. Drift is
  per metre WALKED (the SimNotes rule: a denominator that cannot vanish)."""
  policy = WalkingPolicy(path)
  print(f"{'seed':>4s} {'walked m':>8s} {'position error m':>16s} {'% of distance':>13s} "
        f"{'heading error deg':>17s}")
  for seed in range(seeds):
    model = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    drv = _driver(model, data, policy)
    odo = LegOdometry(model, data, seed=seed)
    for seconds, twist in ODOMETRY_COURSE:
      t0 = data.time
      while data.time - t0 < seconds:
        drv.step(twist)
        odo.step()
    err, dyaw = odo.error()
    print(f"{seed:4d} {odo.distance:8.1f} {err:16.3f} "
          f"{err / odo.distance * 100:12.1f}% {math.degrees(dyaw):17.2f}")


#: A house stair's tread, m (#280 builds the stairs to it).
TREAD_M = 0.28


def staircase_scenery(rise: float, steps: int, edge: float = 0.6,
                      down: bool = False) -> str:
  """`steps` risers of `rise` on a `TREAD_M` tread from x = `edge`: up to a
  landing, or `down` from a landing (the robot starts on it) to the floor."""
  box = ('\n    <geom name="{}" type="box" size="{} 1.0 {}" pos="{} 0 {}" '
         'rgba="0.6 0.5 0.4 1"/>')
  if down:
    top = steps * rise
    geoms = [box.format("landing", 5.0, top / 2, edge - 5.0, top / 2)]
    for i in range(steps - 1):
      h = (steps - 1 - i) * rise
      geoms.append(box.format(f"stair{i}", TREAD_M / 2, h / 2,
                              edge + (i + 0.5) * TREAD_M, h / 2))
    return "".join(geoms)
  geoms = []
  for i in range(steps):
    h = (i + 1) * rise
    geoms.append(box.format(f"stair{i}", TREAD_M / 2, h / 2,
                            edge + (i + 0.5) * TREAD_M, h / 2))
  # A long landing: a flight is flown for a fixed time, and a policy that
  # climbed quickly must not "fail" by walking off the far end of it.
  h = steps * rise
  geoms.append(box.format("landing", 5.0, h / 2, edge + steps * TREAD_M + 5.0, h / 2))
  return "".join(geoms)


def _policy_world(path, scenery: str = "", key: int = 0):
  policy = WalkingPolicy(path)
  model = mujoco.MjModel.from_xml_string(body_xml(CHOSEN, scenery=scenery))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, key)
  mujoco.mj_forward(model, data)
  drv = _driver(model, data, policy)
  return model, data, drv


#: What `--climb` flies (#388): a name, the risers, and up or down. A flight
#: of ten is the house's floor to its second floor.
CLIMBS = (("step up", 1, True), ("flight of 4", 4, True), ("flight of 10", 10, True),
          ("step down", 1, False), ("10 down", 10, False))
#: Where a seeing policy's scan comes from: the ideal casts it trained on,
#: or the D435's map laid through the true pose or the legs' own reckoning
#: (`legs/scan.py`).
SCANS = ("ideal", "map", "odometry")
#: The first riser's edge (or the top step's nose, going down), in x.
CLIMB_EDGE_M = 0.6
#: A climb counts if the robot still stands where it arrived this long after.
SETTLE_ON_ARRIVAL_S = 2.0


def fly_climb(path, rise: float, steps: int, up: bool, trial: int,
              scan: str = "ideal", speed: float = 0.4) -> dict:
  """One climb, `0.6 + 0.1 * trial` m short of the first riser, commanded
  straight at it: whether the robot arrived (on the landing, or on the floor
  past the last riser) and still stood there `SETTLE_ON_ARRIVAL_S` later,
  legged odometry's drift at the moment it arrived, and -- on a map's scan
  -- how far that scan was from the ideal one."""
  model, data, drv = _policy_world(path, staircase_scenery(rise, steps, CLIMB_EDGE_M,
                                                           down=not up))
  data.qpos[0] -= 0.1 * trial
  if not up:
    data.qpos[2] += steps * rise
  mujoco.mj_forward(model, data)
  odo = LegOdometry(model, data, seed=trial)
  eye = None
  if scan != "ideal" and "height_scan" in drv.policy.observation_names:
    eye = MapScan(model, data, odometry=odo if scan == "odometry" else None, seed=trial)
    drv.scan = eye.scan
  top = steps * rise

  def there() -> bool:
    if up:
      return data.qpos[2] > top + 0.8 * CHOSEN.stand_height
    return (data.qpos[0] > CLIMB_EDGE_M + (steps - 1) * TREAD_M + 0.3
            and abs(data.qpos[2] - CHOSEN.stand_height) < 0.08)

  errors, unseen, arrived, taus = [], [], None, []
  margin = np.full(12, math.inf)
  lo, hi = model.jnt_range[1:13].T
  for _ in range(int((4.0 + 2.5 * steps) / model.opt.timestep)):
    if eye is not None and drv.steps % drv.every == 0:
      errors.append(np.abs(eye.scan() - drv.height_scan()))
      unseen.append(float(1.0 - eye.seen().mean()))
    drv.step(Twist(vx=speed))
    odo.step()
    if eye is not None:
      eye.step()
    if data.xmat[drv.root][8] < math.cos(math.radians(70)):
      break
    taus.append(np.abs(data.actuator_force[drv.act]))
    q = data.qpos[drv.qadr]
    margin = np.minimum(margin, np.minimum(q - lo, hi - q))
    if arrived is None and there():
      arrived = {"at_s": data.time, "drift_pct": odo.error()[0] / max(odo.distance, 1e-6) * 100,
                 "height_err": odo.height_error()}
    if arrived is not None and data.time - arrived["at_s"] > SETTLE_ON_ARRIVAL_S:
      break
  made = arrived is not None and there() and data.xmat[drv.root][8] > 0.9
  err = np.concatenate(errors) if errors else np.zeros(0)
  return {"rise": rise, "steps": steps, "up": up, "trial": trial, "made": bool(made),
          **(arrived or {}), "scan_err": err,
          "margin_rad": _by_joint(margin[None, :], np.min),
          "torque": _by_joint(np.array(taus), lambda a: float(np.percentile(a, PEAK_PCT))),
          "unseen": float(np.mean(unseen)) if unseen else float("nan")}


def _fly_climb(args):
  return fly_climb(*args)


def climb(path=POLICY_NPZ, risers=(0.10, 0.12, 0.15, 0.18, 0.20, 0.22),
          trials: int = 5, scan: str = "ideal", jobs: int = 1) -> list[dict]:
  """The tallest riser a policy clears: each of `CLIMBS`, `trials` times, in
  our physics. Odometry's drift is over the climbs that succeeded; a map's
  scan error is over every decision of every flight."""
  todo = [(path, rise, steps, up, trial, scan)
          for rise in risers for _, steps, up in CLIMBS for trial in range(trials)]
  if jobs > 1:
    from multiprocessing import Pool
    with Pool(jobs) as pool:
      rows = pool.map(_fly_climb, todo, chunksize=1)
  else:
    rows = [fly_climb(*t) for t in todo]
  print(f"policy {Path(path).name}, scan: {scan}, {trials} trials a case")
  print(f"{'riser m':>7s} " + " ".join(f"{name:>12s}" for name, _, _ in CLIMBS))
  for rise in risers:
    cells = []
    for _, steps, up in CLIMBS:
      mine = [r for r in rows if r["rise"] == rise and r["steps"] == steps and r["up"] == up]
      cells.append(f"{sum(r['made'] for r in mine)}/{len(mine)}")
    print(f"{rise:7.2f} " + " ".join(f"{c:>12s}" for c in cells))
  print(f"the flights of ten that arrived: p{PEAK_PCT} torque abd/flex/knee, N*m "
        f"(peak {CHOSEN.motor.peak_torque:.0f}); nearest each came to a stop, rad")
  for rise in risers:
    for up in (True, False):
      ok = [r for r in rows if r["rise"] == rise and r["steps"] == 10 and r["up"] == up
            and r["made"]]
      if ok:
        t = np.max([r["torque"] for r in ok], axis=0)
        print(f"  {rise:.2f} {'up  ' if up else 'down'}: {'/'.join(f'{x:4.1f}' for x in t)}"
              f"   stop margin {'/'.join(f'{x:.2f}' for x in np.min([r['margin_rad'] for r in ok], axis=0))}")
  print("legged odometry on the flights of ten that arrived: drift % of distance; "
        "height error, cm (+ reads high)")
  for rise in risers:
    for up in (True, False):
      ok = [r for r in rows if r["rise"] == rise and r["steps"] == 10 and r["up"] == up
            and r["made"]]
      if ok:
        print(f"  {rise:.2f} {'up  ' if up else 'down'}: drift "
              f"{np.mean([r['drift_pct'] for r in ok]):4.1f} %  height "
              + " ".join(f"{r['height_err'] * 100:+.0f}" for r in ok))
  errs = [r for r in rows if len(r["scan_err"])]
  if errs:
    print("the map's scan against the ideal one, per flight of ten (|difference|, mm):")
    for rise in risers:
      for up in (True, False):
        mine = [r for r in errs if r["rise"] == rise and r["steps"] == 10 and r["up"] == up]
        if not mine:
          continue
        e = np.concatenate([r["scan_err"] for r in mine]) * 1000
        print(f"  {rise:.2f} {'up  ' if up else 'down'}: median {np.median(e):5.1f}  "
              f"p95 {np.percentile(e, 95):6.1f}  over 5 cm {np.mean(e > 50):.0%}  "
              f"unseen {np.mean([r['unseen'] for r in mine]):.0%}")
  return rows


#: The posture flight: (seconds, command). Heights are offsets from the stand.
POSTURE_SCHEDULE = [
  (2.0, Twist()),
  (3.0, Twist(height=-0.10)),
  (3.0, Twist(height=-0.14)),
  (3.0, Twist(pitch=0.2)),
  (3.0, Twist(pitch=-0.2)),
  (3.0, Twist(roll=0.15)),
  (3.0, Twist(vx=0.5, height=-0.08)),
  (3.0, Twist(vx=0.5, pitch=0.15)),
  (2.0, Twist()),
]


def posture(path) -> None:
  """A posture policy holding commanded heights, pitches and rolls, still
  and walking: what it was told against what it did, the last 1.5 s of each."""
  model, data, drv = _policy_world(path)
  print(f"{'command':34s} {'height m (want)':>16s} {'pitch deg (want)':>17s} "
        f"{'roll deg (want)':>16s} {'vx':>5s}")
  for seconds, twist in POSTURE_SCHEDULE:
    t0, rows = data.time, []
    while data.time - t0 < seconds:
      drv.step(twist)
      if data.time - t0 > seconds - 1.5:
        r, p, _ = _quat_rpy(data.qpos[3:7])
        rot = data.xmat[drv.root].reshape(3, 3)
        rows.append((data.qpos[2], p, r, (rot.T @ data.qvel[:3])[0]))
      if data.qpos[2] < 0.08:
        break
    z, p, r, vx = np.mean(rows, axis=0)
    label = (f"vx {twist.vx:.1f} h {twist.height:+.2f} p {twist.pitch:+.2f} "
             f"r {twist.roll:+.2f}")
    print(f"{label:34s} {z:7.3f} ({CHOSEN.stand_height + twist.height:5.3f}) "
          f"{math.degrees(p):8.1f} ({math.degrees(twist.pitch):5.1f}) "
          f"{math.degrees(r):7.1f} ({math.degrees(twist.roll):5.1f}) {vx:5.2f}")


#: A drop's landing, s from its first touch, reported apart from what
#: follows: the legs meet the floor wherever the fall threw them. What peaks
#: in it is still mostly the policy's own target -- the drivers' stiffness
#: term, not the impact's speed: #377's policy passed 60 % of the peak in
#: 19 of 20 landings, the stiffness term the larger in all 19 (#389).
LANDING_S = 0.3


def getup_drops(trials: int = 20, seed: int = 0) -> list[tuple]:
  """#377's random drops, drawn in its order: (roll, pitch, yaw, twelve
  joint angles), the joints anywhere within 90 % of their range."""
  lo, hi = mujoco.MjModel.from_xml_string(body_xml(CHOSEN)).jnt_range[1:13].T
  rng = np.random.default_rng(seed)
  return [(rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi / 2, math.pi / 2),
           rng.uniform(-math.pi, math.pi), rng.uniform(lo * 0.9, hi * 0.9))
          for _ in range(trials)]


def getup_trial(path, drop: tuple | None = None, seconds: float = 6.0) -> dict:
  """One get-up in our physics: from lying on the belly, or dropped from
  0.45 m as `drop` says (`getup_drops`). Stood is the served body's test
  (`legs.posture`: right way up at the standing height, held), and every
  time is from the release:

    stood       when the held stand began, or None
    up          the stand-up: s from the first touch to `stood`
    wh          the legs' energy to `stood`
    peak        |torque| by joint over the whole flight, N*m
    peak_after  ...from `LANDING_S` past the first touch (the belly: all)
    vz          the torso's fastest rise over the same span, m/s"""
  model, data, drv = _policy_world(path, key=1 if drop is None else 0)
  if drop is not None:
    data.qpos[:7] = [0, 0, 0.45, *_rpy_quat(*drop[:3])]
    data.qpos[7:19] = drop[3]
    mujoco.mj_forward(model, data)
  floor, dt = model.geom("floor").id, model.opt.timestep
  meter = Meter(CHOSEN, model, data)
  out = {"stood": None, "up": None, "wh": None, "peak_after": np.zeros(12), "vz": 0.0}
  touch = 0.0 if drop is None else None
  held, began, began_wh = 0.0, None, 0.0
  while data.time < seconds:
    drv.step(Twist())
    meter.bill()
    t, z = data.time, float(data.qpos[2])
    if touch is None and (data.contact.geom[:data.ncon] == floor).any():
      touch = t
    if touch is not None and t >= touch + (LANDING_S if drop is not None else 0.0):
      np.maximum(out["peak_after"], np.abs(data.actuator_force[:12]), out=out["peak_after"])
      out["vz"] = max(out["vz"], float(data.qvel[2]))
    if out["stood"] is not None:
      continue
    if (data.xmat[drv.root][8] > moves.UPRIGHT_COS
            and abs(z - CHOSEN.stand_height) < moves.STAND_TOL_M):
      if held == 0.0:
        began, began_wh = t, meter.wh
      held += dt
    else:
      held = 0.0
    if held >= moves.STOOD_HOLD_S:
      out.update(stood=began, up=max(0.0, began - touch), wh=began_wh)
  out["peak"] = meter.peak.copy()
  return out


def _afk(peak: np.ndarray) -> np.ndarray:
  """Twelve joints' peaks as the worst abduction, flexion and knee."""
  return peak.reshape(4, 3).max(axis=0).round(1)


def getup(path, trials: int = 20, seconds: float = 6.0) -> None:
  """The get-up policy in our physics, from `trials` random drops and from
  the belly (`getup_trial` says what each number is)."""
  falls = [getup_trial(path, d, seconds) for d in getup_drops(trials)]
  belly = getup_trial(path, None, seconds)
  ok = [r for r in falls if r["stood"] is not None]
  worst = max(range(len(falls)), key=lambda k: falls[k]["peak_after"].max())
  print(f"from a random fall: stood {len(ok)} of {len(falls)}"
        + (f", median {np.median([r['stood'] for r in ok]):.1f} s from the release "
           f"(slowest {max(r['stood'] for r in ok):.1f})" if ok else ""))
  if ok:
    ups = [r["up"] for r in ok]
    print(f"  the stand-up, first touch to stood: median {np.median(ups):.1f} s, "
          f"shortest {min(ups):.1f} s, longest {max(ups):.1f} s")
  after = np.max([r["peak_after"] for r in falls], axis=0)
  print(f"  peak a/f/k over the flight {_afk(np.max([r['peak'] for r in falls], axis=0))} "
        f"N*m; from {LANDING_S} s after the landing {_afk(after)} N*m (the worst, "
        f"fall {worst}); fastest rise {max(r['vz'] for r in falls):.2f} m/s")
  if ok:
    print(f"  energy to the stand: median {np.median([r['wh'] for r in ok]) * 1000:.0f} mWh")
  if belly["stood"] is None:
    print(f"from the belly: DID NOT STAND; peak a/f/k {_afk(belly['peak'])} N*m")
    return
  print(f"from the belly: stood in {belly['stood']:.1f} s, "
        f"{belly['wh'] * 1000:.0f} mWh to the stand; peak "
        f"a/f/k {_afk(belly['peak'])} N*m; fastest rise {belly['vz']:.2f} m/s")


#: Where `--shove` pushes the served body over, (x, y, heading) in the
#: house: the open floor, beside the couch, against the south wall, the
#: hall, the kitchen counter, a doorway (#387's six).
SHOVE_SPOTS = {"open": (1.5, 0.5, 1.57), "couch": (3.9, 1.3, 1.57),
               "south wall": (2.0, -1.55, 0.0), "hall": (-3.5, -2.0, 1.57),
               "counter": (-8.0, 4.5, 0.0), "doorway": (-1.9, -0.5, 3.14)}
#: A shove that has not put the body down in this long was not a fall, s.
SHOVE_FALL_S = 3.0


def shove(path=None, per_spot: int = 8, seed: int = 7, limit_s: float = 40.0) -> None:
  """The served body in the house, standing, shoved `per_spot` times at each
  of `SHOVE_SPOTS` (a random kick to the torso's speed and spin): how long
  from the fall to standing again, by the body's own posture machine, and
  how many were still down after `limit_s`. `QuadBody.stuck_after_s` is
  read off it. The torque peak is from `LANDING_S` after the landing -- the
  first touch of anything but a foot, which comes a few tenths after the
  fall is called at 60 deg. `path` flies a get-up policy other than the
  committed one."""
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  if path is not None:
    qb.policies()
    qb._POLICIES["getup"] = WalkingPolicy(path)
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=home.GRID_BOUNDS)
  mis, v = body.mission, body.handle.dof_adr(model)
  act = mis.drivers.act
  own = np.zeros(model.ngeom, dtype=bool)
  own[mis.body_gids] = True
  rng = np.random.default_rng(seed)
  ups, never, spots = [], [], {}
  peak, whole = np.zeros(12), np.zeros(12)
  for name, (x, y, yaw) in SHOVE_SPOTS.items():
    for _ in range(per_spot):
      body.start_at(x, y, yaw)
      body.run(mis._drive_routine(0.5, 0.0, 0.0))
      data.qvel[v:v + 3] += rng.uniform([-1.5, -2.5, 0.5], [1.5, 2.5, 1.5])
      data.qvel[v + 3:v + 6] += rng.uniform([-14, -14, -3], [14, 14, 3])
      t0, fell, landed, up = float(data.time), None, None, None
      while data.time - t0 < limit_s and up is None:
        body.run(body.hold_routine(0.02))
        if fell is None and mis.posture == qb.GETTING_UP:
          fell = float(data.time)
        elif fell is None and data.time - t0 > SHOVE_FALL_S:
          break
        if fell is not None and landed is None:
          g = data.contact.geom[:data.ncon]
          if ((mis._is_limb[g[:, 0]] & ~own[g[:, 1]])
                  | (mis._is_limb[g[:, 1]] & ~own[g[:, 0]])).any():
            landed = float(data.time)
        if fell is not None:
          np.maximum(whole, np.abs(data.actuator_force[act]), out=whole)
        if landed is not None and data.time - landed >= LANDING_S:
          np.maximum(peak, np.abs(data.actuator_force[act]), out=peak)
        if fell is not None and mis.posture == qb.STANDING:
          up = float(data.time) - fell
      if fell is None:
        continue
      spots.setdefault(name, []).append(up)
      (ups if up is not None else never).append(up)
  print(f"shoved {per_spot * len(SHOVE_SPOTS)} times, fell {len(ups) + len(never)}: "
        f"up {len(ups)}, still down after {limit_s:.0f} s {len(never)}")
  if ups:
    print(f"  fall to standing: median {np.median(ups):.1f} s, p95 "
          f"{np.percentile(ups, 95):.1f} s, slowest {max(ups):.1f} s")
  for name, xs in spots.items():
    got = [u for u in xs if u is not None]
    print(f"  {name:11s} {len(got)}/{len(xs)} up" + (f", slowest {max(got):.1f} s" if got else ""))
  print(f"  peak a/f/k from each fall on {_afk(whole)} N*m; from {LANDING_S} s "
        f"after each landing {_afk(peak)} N*m")


def _rpy_quat(r: float, p: float, y: float) -> list[float]:
  cr, sr, cp, sp = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2)
  cy, sy = math.cos(y / 2), math.sin(y / 2)
  return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def trace_policy(path, every_s: float = 0.2) -> list[str]:
  """The policy's flight hashed: qpos + qvel + ctrl every `every_s` sim s."""
  import hashlib
  policy = WalkingPolicy(path)
  model = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  drv = _driver(model, data, policy)
  every = round(every_s / model.opt.timestep)
  hashes = []
  for seconds, twist in POLICY_SCHEDULE:
    t0 = data.time
    while data.time - t0 < seconds:
      drv.step(twist)
      if drv.steps % every == 0:
        h = hashlib.sha256()
        for arr in (data.qpos, data.qvel, data.ctrl):
          h.update(arr.tobytes())
        hashes.append(h.hexdigest()[:16])
  return hashes


def determinism(path) -> bool:
  """The same flight in two fresh processes, BLAS pinned: identical?"""
  import json
  import subprocess
  runs = []
  for _ in range(2):
    out = subprocess.run(
      [sys.executable, __file__, "--trace", str(path)], capture_output=True,
      text=True, check=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
  a, b = runs
  first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
  same = first is None and len(a) == len(b)
  print(f"{len(a)} samples over {sum(s for s, _ in POLICY_SCHEDULE):.0f} sim s: "
        + ("IDENTICAL" if same else f"first differs at sample {first}"))
  return same


def fly_policy(path=POLICY_NPZ, spec: BodySpec = CHOSEN, view: bool = False):
  """The trained policy in OUR sim: each segment's tracking, the falls, the
  torques and what a policy step costs. Returns the rows."""
  policy = WalkingPolicy(path)
  model = mujoco.MjModel.from_xml_string(body_xml(spec))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  drv = _driver(model, data, policy)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
    viewer.cam.distance, viewer.cam.elevation = 1.8, -18
    viewer.cam.trackbodyid = drv.root
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  rows, fell, wall0 = [], False, time.time()
  act_s, n_act = 0.0, 0
  lim = JointLimits.of(spec.motor, spec.knee_ratio, BUS_V)
  for seconds, twist in POLICY_SCHEDULE:
    t0, vel, yaw_rate, taus, tilt, copper, mech = data.time, [], [], [], [], [], []
    while data.time - t0 < seconds and not fell:
      decide = drv.steps % drv.every == 0
      t = time.perf_counter()
      drv.command(twist)
      if decide:
        act_s += time.perf_counter() - t
        n_act += 1
      mujoco.mj_step(model, data)
      if data.time - t0 > 1.0:  # settled into the new command
        rot = data.xmat[drv.root].reshape(3, 3)
        vel.append(rot.T @ data.qvel[0:3])
        yaw_rate.append(data.qvel[5])
        tau, qd = data.actuator_force[drv.act].copy(), data.qvel[drv.vadr]
        taus.append(tau)
        copper.append(float(lim.copper_w(tau).sum()))
        mech.append(float(np.clip(tau * qd, 0, None).sum()))
        r_, p_, _ = _quat_rpy(data.qpos[3:7])
        tilt.append((r_, p_))
      if viewer is not None and drv.steps % 10 == 0:
        if not viewer.is_running():
          return rows
        viewer.sync()
        ahead = data.time - (time.time() - wall0)
        if ahead > 0:
          time.sleep(ahead)
      roll, pitch, _ = _quat_rpy(data.qpos[3:7])
      fell = data.qpos[2] < 0.15 or abs(roll) > 1.0 or abs(pitch) > 1.0
    v = np.mean(vel, axis=0) if vel else np.full(3, np.nan)
    tau = np.abs(np.array(taus)) if taus else np.full((1, 12), np.nan)
    tl = np.degrees(np.array(tilt)) if tilt else np.full((1, 2), np.nan)
    rows.append({"command": twist, "vx": v[0], "vy": v[1],
                 "yaw_rate": float(np.mean(yaw_rate)) if yaw_rate else np.nan,
                 "peak": _by_joint(tau, lambda a: float(np.percentile(a, PEAK_PCT))),
                 "rms": _by_joint(np.array(taus) if taus else tau,
                                  lambda a: float(np.sqrt((a ** 2).mean(axis=0)).max())),
                 "copper_w": float(np.mean(copper)) if copper else np.nan,
                 "mech_w": float(np.mean(mech)) if mech else np.nan,
                 "tilt_deg": float(np.abs(tl - tl.mean(axis=0)).max()),
                 "fell": fell})
  if viewer is not None:
    viewer.close()
  print(f"policy {Path(path).name} sha256 {policy.sha256[:12]}: trained "
        f"{policy.training['env_steps'] / 1e6:.0f} M steps on {policy.training['envs']} "
        f"envs, {policy.training['gpu']}, {policy.training['wall']}")
  print(f"one policy step: {act_s / max(n_act, 1) * 1e6:.0f} us "
        f"(every {drv.every} physics steps)")
  base = sum(ELECTRONICS_W.values())
  print(f"{'command vx/vy/yaw':>18s} {'got vx/vy/yaw':>18s} {'p99.5 tau a/f/k':>16s} "
        f"{'RMS a/f/k':>15s} {'W (windings+shaft+electronics)':>31s} {'tilt deg':>8s}")
  for r in rows:
    c = r["command"]
    watts = r["copper_w"] + r["mech_w"] + base
    print(f"{c.vx:5.2f}/{c.vy:5.2f}/{c.yaw_rate:5.2f} "
          f"{r['vx']:6.2f}/{r['vy']:5.2f}/{r['yaw_rate']:5.2f} "
          f"{'/'.join(f'{x:4.1f}' for x in r['peak']):>16s} "
          f"{'/'.join(f'{x:3.1f}' for x in r['rms']):>15s} "
          f"{watts:7.1f} ({r['copper_w']:5.1f}+{r['mech_w']:5.1f}+{base:4.1f}) "
          f"{r['tilt_deg']:8.2f}"
          + ("  FELL" if r["fell"] else ""))
  return rows


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--view", action="store_true")
  ap.add_argument("--torque", action="store_true")
  ap.add_argument("--sweep", action="store_true")
  ap.add_argument("--energy", action="store_true")
  ap.add_argument("--thermal", action="store_true")
  ap.add_argument("--served", nargs="?", const=str(POLICY_NPZ), default=None,
                  help="a pair's physics + controller cost, quadrupeds vs rovers")
  ap.add_argument("--rays", action="store_true",
                  help="what a 2D and a 3D LIDAR scan cost the physics thread")
  ap.add_argument("--policy", nargs="?", const=str(POLICY_NPZ), default=None,
                  help="fly the exported walking policy (default: the committed "
                       "one) in the served sim's physics; with --view, watch it")
  ap.add_argument("--determinism", nargs="?", const=str(POLICY_NPZ),
                  default=None, help="the policy's flight in two processes, "
                                     "hashed: identical or not")
  ap.add_argument("--trace", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--climb", nargs="?", const=str(POLICY_NPZ), default=None,
                  help="the tallest riser a policy clears, as a step, flights of "
                       "4 and 10 up and 10 down")
  ap.add_argument("--scan", choices=SCANS, default="ideal",
                  help="--climb: a seeing policy's scan source (legs/scan.py)")
  ap.add_argument("--trials", type=int, default=None,
                  help="--climb: trials a case (5); --getup: random drops (20)")
  ap.add_argument("--risers", default=None,
                  help="--climb: comma-separated riser heights, m")
  ap.add_argument("--jobs", type=int, default=1, help="--climb: processes")
  ap.add_argument("--posture", default=None, metavar="NPZ",
                  help="a posture policy holding commanded heights and tilts")
  ap.add_argument("--getup", default=None, metavar="NPZ",
                  help="a get-up policy from random falls and from the belly")
  ap.add_argument("--shove", nargs="?", const="", default=None, metavar="NPZ",
                  help="the served body shoved over at six places in the house: "
                       "the get-up's time and the ones still down (default: "
                       "the committed get-up policy)")
  ap.add_argument("--odometry", nargs="?", const=str(POLICY_NPZ), default=None,
                  help="legged odometry's drift on the course, the policy walking")
  ap.add_argument("--pupper", action="store_true",
                  help="the Pupper-class body, as built and with the suite")
  ap.add_argument("--bus", type=float, default=BUS_V_NOMINAL,
                  help="pack voltage the tables fly at (12S: 36-50.4 V)")
  ap.add_argument("--out", default="quad_spike.png")
  args = ap.parse_args(argv)
  global BUS_V
  BUS_V = args.bus
  if args.trace:
    import json
    print(json.dumps(trace_policy(args.trace)))
  elif args.climb:
    risers = tuple(float(r) for r in args.risers.split(",")) if args.risers else None
    climb(args.climb, **({"risers": risers} if risers else {}), trials=args.trials or 5,
          scan=args.scan, jobs=args.jobs)
  elif args.getup:
    getup(args.getup, trials=args.trials or 20)
  elif args.shove is not None:
    shove(args.shove or None)
  elif args.posture:
    posture(args.posture)
  elif args.odometry:
    odometry(args.odometry)
  elif args.determinism:
    sys.exit(0 if determinism(args.determinism) else 1)
  elif args.policy:
    fly_policy(args.policy, view=args.view)
  elif args.view:
    view(PUPPER_WITH_SUITE if args.pupper else SIZING)
  elif args.torque and args.pupper:
    print_torque_table(PUPPER_CLASS)
    print()
    print_torque_table(PUPPER_WITH_SUITE)
  elif args.torque:
    print_torque_table(SIZING)
  elif args.sweep:
    sweep()
  elif args.energy:
    energy_table(SIZING)
  elif args.thermal:
    thermal_table(SIZING)
  elif args.rays:
    ray_cost()
  elif args.served:
    served_cost(args.served)
  else:
    filmstrip(args.out)


if __name__ == "__main__":
  sys.exit(main())
