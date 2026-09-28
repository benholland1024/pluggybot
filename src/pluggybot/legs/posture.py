"""Lying down and standing up (issues #377, #387): the body's two scripted
moves, and the rules for when it makes them.

Each move is a generator of twelve joint TORQUES, one per physics step,
through the drivers' envelope (`legs.drivers.Drivers.torque`): the
scripted gait (`legs.scripted.VirtualModel`) lowers or raises the torso
while the feet carry it, and a joint-space PD folds the legs between the
stand and the rest pose (`model.lie_qpos`). #377 measured both at 2.2 s,
and #378's dock is built on where the lie-down puts the body
(`dock.LIE_SHIFT_M`): a change here re-flies the dock.

⚠ THE REST IS A REFLEX, and the mind is not asked (Ben, 2026-09-27, on
#387): the body lies down after `T_REST_S` without a motion command and
stands before the next one (`legs.body.QuadStepper`). It decides nothing
the mind decides -- the rover's parking brake is the precedent -- and the
posture rides the wire as a fact.
"""

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.body import TOPPLE_TILT_RAD
from pluggybot.legs.model import JOINT_NAMES, BodySpec, lie_qpos, pose_qpos
from pluggybot.legs.scripted import Command, VirtualModel
from pluggybot.telemetry.protocol import ROBOT_ROOT

#: The rest reflex's delay, s: standing costs 39 W more than lying, and a
#: stand-up plus a lie-down 93 mWh, so any wait longer than this is cheaper
#: lying down (`quad_spike.py --energy`; SimNotes, "The quadruped body").
T_REST_S = 8.6
#: The driver's PD for a joint-space move (lying down, folding the legs):
#: stiff enough to fold a leg under the belly in a second, N*m/rad, N*m*s/rad.
FOLD_KP, FOLD_KD = 60.0, 1.5
#: Crouch height the legs fold to before the push up, m.
FOLD_H = 0.16
#: After the fold the drivers hold nothing this long before the body counts
#: as lying, s: the belly and the limp legs settle (#378's `SETTLE_S`).
LIE_SETTLE_S = 1.0
#: After the push up the scripted gait holds the stand this long before the
#: policy takes the body back, s.
STAND_HOLD_S = 0.3
#: FALLEN: the torso's up axis this far from the world's, rad -- the
#: topple angle every body is judged down at, so "down" means one thing.
FALL_TILT_RAD = TOPPLE_TILT_RAD
#: ...or, on its feet by its own account, the torso this low for this long:
#: legs that buckled under it leave it level on its belly (m, s).
SLUMP_Z_M, SLUMP_S = 0.20, 0.5
#: STOOD, after a get-up: right way up and at the standing height, held
#: (what `quad_spike.py --getup` reads its table by).
UPRIGHT_COS, STAND_TOL_M, STOOD_HOLD_S = 0.95, 0.04, 0.5


def ease(s: float) -> float:
  """0 -> 1 over s in [0, 1], a half cosine: no step at either end (the
  ramping rule, CLAUDE.md)."""
  s = min(max(s, 0.0), 1.0)
  return 0.5 - 0.5 * math.cos(math.pi * s)


@dataclass(frozen=True)
class Joints:
  """Where one robot's twelve leg joints and its root height live in
  `qpos` / `qvel`, by name (a world may carry other joints first)."""

  qadr: np.ndarray
  vadr: np.ndarray
  #: The root's height, `qpos[z]`.
  z: int

  @classmethod
  def of(cls, model, prefix: str = "") -> "Joints":
    ids = [model.joint(f"{prefix}{n}").id for n in JOINT_NAMES]
    root = model.body(f"{prefix}{ROBOT_ROOT}").id
    return cls(np.array([model.jnt_qposadr[j] for j in ids]),
               np.array([model.jnt_dofadr[j] for j in ids]),
               int(model.jnt_qposadr[model.body_jntadr[root]]) + 2)


def fold_pd(data, joints: Joints, target: np.ndarray) -> np.ndarray:
  """The driver's own PD toward joint targets, as a torque."""
  return (FOLD_KP * (target - data.qpos[joints.qadr])
          - FOLD_KD * data.qvel[joints.vadr])


def lie_down_routine(spec: BodySpec, data, vm: VirtualModel, joints: Joints,
                     lower_s: float = 1.2, fold_s: float = 1.0):
  """Stand -> belly, one torque a physics step: lower under the scripted
  gait to the fold height, then fold the legs out to the rest pose."""
  vm.reset()
  h0, t0 = float(data.qpos[joints.z]), data.time
  while data.time - t0 < lower_s:
    s = ease((data.time - t0) / lower_s)
    yield vm.torque(Command(height=h0 + s * (FOLD_H - h0)))
  start, lie, t0 = data.qpos[joints.qadr].copy(), np.array(lie_qpos(spec)), data.time
  while data.time - t0 < fold_s:
    yield fold_pd(data, joints, start + ease((data.time - t0) / fold_s) * (lie - start))


def stand_up_routine(spec: BodySpec, data, vm: VirtualModel, joints: Joints,
                     fold_s: float = 1.0, rise_s: float = 1.2):
  """Belly -> stand: fold the feet under the hips (the belly still takes
  the weight), then push up under the scripted gait."""
  start, t0 = data.qpos[joints.qadr].copy(), data.time
  crouch = np.array(pose_qpos(spec, FOLD_H))
  while data.time - t0 < fold_s:
    yield fold_pd(data, joints, start + ease((data.time - t0) / fold_s) * (crouch - start))
  vm.reset()
  z0, t0 = float(data.qpos[joints.z]), data.time
  while data.time - t0 < rise_s:
    s = ease((data.time - t0) / rise_s)
    yield vm.torque(Command(height=z0 + s * (spec.stand_height - z0)))
