"""The quadruped body spike (issue #377): size it, then measure what it costs.

The body is `pluggybot.legs.model.BodySpec`, its actuators
`pluggybot.legs.actuator`, and until the walking policy is trained the gait
is `pluggybot.legs.scripted.VirtualModel` -- a measuring instrument, not the
robot's gait. Every table here is SimNotes, "The quadruped body".

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
from pluggybot.legs.model import (CHOSEN, LEGS, PUPPER_CLASS, PUPPER_WITH_SUITE,  # noqa: E402
                                  BodySpec, body_xml, lie_qpos, pose_qpos)
from pluggybot.legs.policy import POLICY_NPZ, PolicyDriver, Twist, WalkingPolicy  # noqa: E402
from pluggybot.legs.scripted import Command, VirtualModel, _quat_rpy  # noqa: E402

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
  model = mujoco.MjModel.from_xml_string(body_xml(spec, scenery=scenery))
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


def torque_table(spec: BodySpec = CHOSEN) -> list[dict]:
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


def print_torque_table(spec: BodySpec = CHOSEN) -> None:
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


#: The driver's PD for a joint-space move (lying down, folding the legs):
#: stiff enough to fold a leg under the belly in a second, N*m/rad, N*m*s/rad.
FOLD_KP, FOLD_KD = 60.0, 1.5
#: Crouch height the legs fold to before the push up, m.
FOLD_H = 0.16


class Meter:
  """Steps the physics through the actuator envelope and keeps the bill."""

  def __init__(self, spec: BodySpec, model, data):
    self.lim = JointLimits.of(spec.motor, spec.knee_ratio)
    self.m, self.d = model, data
    self.copper_j = self.mech_j = 0.0
    self.peak = np.zeros(12)

  def step(self, tau: np.ndarray) -> None:
    d, dt = self.d, self.m.opt.timestep
    t = self.lim.clip(tau, d.qvel[6:18])
    d.ctrl[:] = t
    mujoco.mj_step(self.m, d)
    self.copper_j += float(self.lim.copper_w(t).sum()) * dt
    self.mech_j += float(np.clip(t * d.qvel[6:18], 0, None).sum()) * dt
    np.maximum(self.peak, np.abs(t), out=self.peak)

  def pd(self, target: np.ndarray) -> np.ndarray:
    return FOLD_KP * (target - self.d.qpos[7:19]) - FOLD_KD * self.d.qvel[6:18]

  @property
  def wh(self) -> float:
    return (self.copper_j + self.mech_j) / 3600


def _ease(s: float) -> float:
  s = min(max(s, 0.0), 1.0)
  return 0.5 - 0.5 * math.cos(math.pi * s)


def _pd(data, target: np.ndarray) -> np.ndarray:
  """The driver's own PD toward joint targets (the standalone model: the
  robot's joints are qpos[7:19])."""
  return FOLD_KP * (target - data.qpos[7:19]) - FOLD_KD * data.qvel[6:18]


def lie_down_routine(spec: BodySpec, data, vm: VirtualModel,
                     lower_s: float = 1.2, fold_s: float = 1.0):
  """Stand -> belly, one torque a physics step: lower under the scripted
  gait to the fold height, then fold the legs out to the rest pose."""
  vm.reset()
  h0, t0 = float(data.qpos[2]), data.time
  while data.time - t0 < lower_s:
    s = _ease((data.time - t0) / lower_s)
    yield vm.torque(Command(height=h0 + s * (FOLD_H - h0)))
  start, lie, t0 = data.qpos[7:19].copy(), np.array(lie_qpos(spec)), data.time
  while data.time - t0 < fold_s:
    yield _pd(data, start + _ease((data.time - t0) / fold_s) * (lie - start))


def stand_up_routine(spec: BodySpec, data, vm: VirtualModel,
                     fold_s: float = 1.0, rise_s: float = 1.2):
  """Belly -> stand: fold the feet under the hips (the belly still takes
  the weight), then push up under the scripted gait."""
  start, t0 = data.qpos[7:19].copy(), data.time
  crouch = np.array(pose_qpos(spec, FOLD_H))
  while data.time - t0 < fold_s:
    yield _pd(data, start + _ease((data.time - t0) / fold_s) * (crouch - start))
  vm.reset()
  z0, t0 = float(data.qpos[2]), data.time
  while data.time - t0 < rise_s:
    s = _ease((data.time - t0) / rise_s)
    yield vm.torque(Command(height=z0 + s * (spec.stand_height - z0)))


def _metered(spec: BodySpec, key: int, routine):
  model = mujoco.MjModel.from_xml_string(body_xml(spec))
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


#: What is powered whatever the legs do, W (Parts.md, "The quadruped body").
ELECTRONICS_W = {
  # Pi 5 + two cameras + IMU: the rover's `power.ELECTRONICS_W` less its
  # LIDAR share. Raspberry Pi publishes no load figure.
  "compute + cameras": 6.0,
  # RPLIDAR C1: 230 mA typical at 5 V (Slamtec datasheet rev 1.1).
  "lidar": 1.15,
  # RealSense D435 streaming depth with its projector: the rover's
  # `power.DEPTH_CAMERA_W` (the maker's 3.40 W is depth AND 1080p colour).
  "depth camera": 2.0,
  # Twelve GDS68 drivers powered, the maker's standby current (< 10 mA) at
  # 48 V. An enabled FOC loop at zero torque is unpublished and draws more.
  "drivers": 12 * 0.48,
}
#: Ambient air for the thermal table, C: a warm room; `--thermal` also
#: flies a hot day.
AMBIENT_C = (30.0, 40.0)


def energy_table(spec: BodySpec = CHOSEN) -> None:
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


def thermal_table(spec: BodySpec = CHOSEN) -> None:
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
        spec = CHOSEN.with_(knee_ratio=kr, thigh=leg, shank=leg, motor=motor)
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
  model = mujoco.MjModel.from_xml_string(body_xml(spec))
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


def filmstrip(out: str, spec: BodySpec = CHOSEN) -> None:
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


def trace_policy(path, every_s: float = 0.2) -> list[str]:
  """The policy's flight hashed: qpos + qvel + ctrl every `every_s` sim s."""
  import hashlib
  policy = WalkingPolicy(path)
  model = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  drv = PolicyDriver(model, data, policy, JointLimits.of(CHOSEN.motor, 1.0, BUS_V))
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
  drv = PolicyDriver(model, data, policy, JointLimits.of(spec.motor, spec.knee_ratio, BUS_V))
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
    viewer.cam.distance, viewer.cam.elevation = 1.8, -18
    viewer.cam.trackbodyid = drv.root
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  rows, fell, wall0 = [], False, time.time()
  act_s, n_act = 0.0, 0
  for seconds, twist in POLICY_SCHEDULE:
    t0, vel, yaw_rate, taus = data.time, [], [], []
    while data.time - t0 < seconds and not fell:
      decide = drv.steps % drv.every == 0
      t = time.perf_counter()
      data.ctrl[:12] = drv.torque(twist)
      if decide:
        act_s += time.perf_counter() - t
        n_act += 1
      mujoco.mj_step(model, data)
      if data.time - t0 > 1.0:  # settled into the new command
        rot = data.xmat[drv.root].reshape(3, 3)
        vel.append(rot.T @ data.qvel[0:3])
        yaw_rate.append(data.qvel[5])
        taus.append(data.ctrl[:12].copy())
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
    rows.append({"command": twist, "vx": v[0], "vy": v[1],
                 "yaw_rate": float(np.mean(yaw_rate)) if yaw_rate else np.nan,
                 "peak": _by_joint(tau, lambda a: float(np.percentile(a, PEAK_PCT))),
                 "fell": fell})
  if viewer is not None:
    viewer.close()
  print(f"policy {Path(path).name} sha256 {policy.sha256[:12]}: trained "
        f"{policy.training['env_steps'] / 1e6:.0f} M steps on {policy.training['envs']} "
        f"envs, {policy.training['gpu']}, {policy.training['wall']}")
  print(f"one policy step: {act_s / max(n_act, 1) * 1e6:.0f} us "
        f"(every {drv.every} physics steps)")
  print(f"{'command vx/vy/yaw':>18s} {'got vx/vy/yaw':>18s} {'p99.5 tau a/f/k':>16s}")
  for r in rows:
    c = r["command"]
    print(f"{c.vx:5.2f}/{c.vy:5.2f}/{c.yaw_rate:5.2f} "
          f"{r['vx']:6.2f}/{r['vy']:5.2f}/{r['yaw_rate']:5.2f} "
          f"{'/'.join(f'{x:4.1f}' for x in r['peak']):>16s}"
          + ("  FELL" if r["fell"] else ""))
  return rows


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--view", action="store_true")
  ap.add_argument("--torque", action="store_true")
  ap.add_argument("--sweep", action="store_true")
  ap.add_argument("--energy", action="store_true")
  ap.add_argument("--thermal", action="store_true")
  ap.add_argument("--policy", nargs="?", const=str(POLICY_NPZ), default=None,
                  help="fly the exported walking policy (default: the committed "
                       "one) in the served sim's physics; with --view, watch it")
  ap.add_argument("--determinism", nargs="?", const=str(POLICY_NPZ),
                  default=None, help="the policy's flight in two processes, "
                                     "hashed: identical or not")
  ap.add_argument("--trace", default=None, help=argparse.SUPPRESS)
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
  elif args.determinism:
    sys.exit(0 if determinism(args.determinism) else 1)
  elif args.policy:
    fly_policy(args.policy, view=args.view)
  elif args.view:
    view(PUPPER_WITH_SUITE if args.pupper else CHOSEN)
  elif args.torque and args.pupper:
    print_torque_table(PUPPER_CLASS)
    print()
    print_torque_table(PUPPER_WITH_SUITE)
  elif args.torque:
    print_torque_table(CHOSEN)
  elif args.sweep:
    sweep()
  elif args.energy:
    energy_table(CHOSEN)
  elif args.thermal:
    thermal_table(CHOSEN)
  else:
    filmstrip(args.out)


if __name__ == "__main__":
  sys.exit(main())
