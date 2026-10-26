"""The humanoid's arm joints' friction, calibrated once and stored (issue
#493, stage 2): what each arm joint's driver reads beyond holding the arm's
weight while that joint turns slowly in free air, the rest held -- fitted
per joint as a dry friction against the motion, N*m, and a viscous one, N*m
a rad/s (`calibrate_routine`; `scripts/humanoid_spike.py --friction`).
Kept in `FILE`, beside the robot's drawing, and taken off the drivers'
torques where a joint turns (`taken`, `SelfModel.hand_force`): so the force
a hand reads is net of its arm's weight AND its joints' friction, and no
probe flies the calibration again.

⚠ HELD STILL a joint's friction is anything up to its dry friction either
way, which nothing reads: none is taken off a joint turning slower than
`MOVING_RAD_S`, and a hand's force held still keeps that much in it -- the
whole body's reading weighs each joint by it (`still`, `SelfModel.felt`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pluggybot.humanoid import model as hm

#: Where the table is kept: beside the robot's drawing.
FILE = Path(__file__).resolve().parents[3] / "models" / "humanoid_friction.json"
#: The arm's pose the joints are swept about (the left's; the right mirrors
#: its roll, yaw and wrist roll): out in front and down, clear of the body
#: and every joint inside its range by the sweep.
POSE = {"shoulder_pitch": -0.6, "shoulder_roll": 0.45, "shoulder_yaw": 0.0, "elbow": 0.9,
        "wrist_pitch": 0.0, "wrist_roll": 0.0}
_MIRRORED = ("shoulder_roll", "shoulder_yaw", "wrist_roll")
#: Each joint is swept this far either side of the pose, rad, there and
#: back at each of these speeds, rad/s, each end settled this long, s. ⚠ AT
#: ONE SPEED the dry and the viscous terms are one column: fitted, the
#: shoulder read 1.78 N*m dry and -4.97 viscous, which sum right at 0.3
#: rad/s and nowhere else.
SWEEP_RAD, SWEEP_SPEEDS, SETTLE_S = 0.35, (0.15, 0.45), 0.4
#: Only the steady part of a sweep is read: a joint turning within this
#: share of its sweep's speed (its ramp's ends are the arm accelerating).
STEADY = 0.2
#: A joint turning slower than this, rad/s, has no friction taken off it,
#: and no sample is read off it.
MOVING_RAD_S = 0.05


def pose(side: str) -> dict:
  """The calibration pose of one arm, joint name -> angle."""
  return {f"{side}_{j}": (-v if side == "right" and j in _MIRRORED else v)
          for j, v in POSE.items()}


def fit(samples: list[tuple[float, float, float]]) -> dict:
  """One joint's friction off its samples (its speed, rad/s, its driver's
  torque beyond the arm's weight, N*m, and the sweep's speed), read where
  it turned steadily (`STEADY`): torque = coulomb * sign(speed) + viscous *
  speed, least squares."""
  keep = [s for s in samples
          if abs(s[0]) > MOVING_RAD_S and abs(abs(s[0]) - s[2]) <= STEADY * s[2]]
  qd = np.array([s[0] for s in keep])
  tau = np.array([s[1] for s in keep])
  if len(qd) < 10:
    raise ValueError(f"{len(qd)} samples turning: too few to fit")
  a = np.column_stack([np.sign(qd), qd])
  (coulomb, viscous), *_ = np.linalg.lstsq(a, tau, rcond=None)
  return {"coulomb": round(float(coulomb), 4), "viscous": round(float(viscous), 4),
          "samples": int(len(qd))}


def calibrate_routine(body):
  """Each arm joint swept about `pose` in free air, its driver's torque
  read beyond the arm's weight (the arm's own model, posed off its
  encoders and IMU), and fitted (`fit`): a routine returning the table,
  joint name -> {coulomb, viscous, samples}. The body stands where
  nothing is within its arm's reach."""
  reach, data = body.reach, body.data
  table = {}
  for side in hm.SIDES:
    start = pose(side)
    for j in hm.ARM:
      name = f"{side}_{j}"
      yield from body.ramp_routine(body.actuator(name), start[name], SWEEP_SPEEDS[0])
    yield from body.hold_routine(SETTLE_S)
    for j in hm.ARM:
      name = f"{side}_{j}"
      act, k = body.actuator(name), hm.ARM.index(j)
      samples: list = []

      for speed in SWEEP_SPEEDS:
        def record(side=side, k=k, samples=samples, speed=speed):
          # ...its weight where its encoders put it: between its commands,
          # the model is posed at them, and the lag read as friction
          reach._pose()
          _, qd, tau = reach.stand.joints.read(data)
          arm = reach.self_model.arm[side]
          samples.append((float(qd[arm][k]),
                          float(tau[arm][k] - reach.self_model.gravity(side)[k]), speed))
        for to in (start[name] + SWEEP_RAD, start[name] - SWEEP_RAD):
          body.step_hooks.append(record)
          try:
            yield from body.ramp_routine(act, to, speed)
          finally:
            body.step_hooks.remove(record)
          yield from body.hold_routine(SETTLE_S)
      yield from body.ramp_routine(act, start[name], SWEEP_SPEEDS[0])
      table[name] = fit(samples)
    hang = hm.stand_qpos()
    for j in reversed(hm.ARM):
      name = f"{side}_{j}"
      yield from body.ramp_routine(body.actuator(name), hang.get(name, 0.0), SWEEP_SPEEDS[0])
  return table


def load(path: Path | str = FILE) -> dict:
  """The stored table, joint name -> {coulomb, viscous}; empty where none
  is kept (nothing is taken off then)."""
  p = Path(path)
  if not p.exists():
    return {}
  return json.loads(p.read_text())["joints"]


def save(table: dict, path: Path | str = FILE) -> None:
  Path(path).write_text(json.dumps(
    {"about": "the humanoid's arm joints' friction, calibrated (humanoid/friction.py; "
              "scripts/humanoid_spike.py --friction --write)",
     "joints": table}, indent=1) + "\n")


def still(table: dict, drawn: np.ndarray) -> np.ndarray:
  """What each joint's friction may hold while it is still, N*m, in
  `hm.JOINT_NAMES` order: an arm joint's dry friction as calibrated, where
  the table has it -- else `drawn`, the robot's drawing's (the legs': none
  is calibrated)."""
  out = np.array(drawn, dtype=float)
  for k, name in enumerate(hm.JOINT_NAMES):
    row = table.get(name)
    if row is not None:
      out[k] = row["coulomb"]
  return out


def taken(table: dict, side: str, qd: np.ndarray) -> np.ndarray:
  """The friction torque, N*m, each of `side`'s arm joints turns against at
  speeds `qd` (`hm.ARM` order): nothing below `MOVING_RAD_S`, or where the
  table has no row."""
  out = np.zeros(len(hm.ARM))
  for k, j in enumerate(hm.ARM):
    row = table.get(f"{side}_{j}")
    if row is not None and abs(qd[k]) > MOVING_RAD_S:
      out[k] = row["coulomb"] * np.sign(qd[k]) + row["viscous"] * qd[k]
  return out
