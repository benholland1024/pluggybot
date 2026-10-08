"""A rollout (issue #466): a record's commands replayed, step for step, in a
world of the robot's own (`compile.Imagined`), and what the arm's drivers
would have read. The instrument stage 3's fitter calls, in the worker.

The robot starts as the record's `start` says it knew itself -- at its
believed pose, tilted as its IMU read, its joints at its encoders', lying on
its belly as its CAD lies, the tool seated on its fork -- the scene's joints
where the world says they start (`Imagined.start`), and is given `SETTLE_S`
holding the first command to come to rest there. Then each row:
the arm's drivers run on its targets and gains as the served body's run
(the same `legs.arm.ArmDriver`), the claw's servos take its commands, the
legs hold nothing (it lies), the world steps, and the hooks (a scene's
catches) set what they set. The body's half of all this is
`legs.imagined.ImaginedBody`.

What it returns is the drivers' torques and the encoders: the readings'
EXPECTATION -- a real driver's carry its noise and its counts
(`perception.encoders.torque_readings`), which `noise_seed` adds, keyed on
the seed and never on the world's. ⚠ A world MuJoCo finds unstable is reset
in place and steps on, finite and plausible: a rollout that saw one is
refused (`Diverged`), never returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pluggybot.imagination.compile import Imagined
from pluggybot.imagination.record import SENSED, Record

#: How long the robot holds the record's first command before its first row,
#: s: it is put down 1 mm over the floor and its tool on its fork, and the
#: claw settles on its peg within a second (#469's `mount`).
SETTLE_S = 1.0
#: MuJoCo's counters of a world it found unstable and reset: a 10 g part on
#: a spring of 40 N*m/rad was reset 124 times in one rollout, every reading
#: finite (the review of #473).
UNSTABLE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC)


class Diverged(RuntimeError):
  """A rollout whose world went unstable and was reset: its readings are
  not the document's."""


def _stable(data, when: str) -> None:
  bad = [w.name for w in UNSTABLE if data.warning[w].number > 0]
  if bad:
    raise Diverged(f"the world went unstable {when} ({', '.join(bad)}) and MuJoCo reset "
                   f"it: these readings are not the document's")


@dataclass(frozen=True)
class Readings:
  """A row per command: `sensed` is `record.SENSED`'s columns, and `joints`
  each scene joint's coordinate as imagined (rad, or m)."""
  t: np.ndarray
  sensed: np.ndarray
  joints: dict[str, np.ndarray]

  def column(self, name: str) -> np.ndarray:
    return self.sensed[:, SENSED.index(name)]

  def to_wire(self) -> tuple[dict, dict[str, np.ndarray]]:
    return ({"joints": sorted(self.joints)},
            {"t": self.t, "sensed": self.sensed,
             **{f"joint_{k}": v for k, v in self.joints.items()}})

  @classmethod
  def from_wire(cls, head: dict, arrays: dict) -> "Readings":
    return cls(t=arrays["t"], sensed=arrays["sensed"],
               joints={k: arrays[f"joint_{k}"] for k in head["joints"]})


def _body(world: Imagined, data):
  from pluggybot.legs.imagined import ImaginedBody
  return ImaginedBody(world.model, data, world.carrying)


def settled(world: Imagined, record: Record, settle_s: float = SETTLE_S) -> mujoco.MjData:
  """The record's start in `world`, held at its first command for
  `settle_s`: where its first row begins."""
  m = world.model
  dt = float(m.opt.timestep)
  if abs(record.dt - dt) > 1e-12:
    raise ValueError(f"a record is replayed a row a physics step: its dt is "
                     f"{record.dt:g} s, the world's step {dt:g}")
  if record.start.carrying != world.carrying:
    raise ValueError(f"the record's robot carried {record.start.carrying!r} and this "
                     f"world's carries {world.carrying!r}: compile it with the record's")
  d = mujoco.MjData(m)
  body = _body(world, d)
  st, first = record.start, record.commands[0]
  body.place(st.pose, st.attitude, st.legs, st.arm, first)
  if world.start:
    for name, q in world.start.items():
      d.qpos[world.joints[name][0]] = q
    mujoco.mj_forward(m, d)
  for _ in range(round(settle_s / dt)):
    body.apply(first)
    mujoco.mj_step(m, d)
    for hook in world.hooks:
      hook(m, d)
  _stable(d, "settling")
  return d


def rollout(world: Imagined, record: Record, settle_s: float = SETTLE_S,
            noise_seed: int | None = None) -> Readings:
  """`record`'s commands replayed in `world` (the module docstring)."""
  m = world.model
  d = settled(world, record, settle_s)
  body = _body(world, d)
  hooks, cmds, n = world.hooks, record.commands, record.n
  t0 = float(d.time)
  t = np.zeros(n)
  sensed = np.zeros((n, len(SENSED)))
  joints = {name: np.zeros(n) for name in world.joints}
  adr = {name: q for name, (q, _) in world.joints.items()}
  for k in range(n):
    body.apply(cmds[k])
    mujoco.mj_step(m, d)
    for hook in hooks:
      hook(m, d)
    t[k] = d.time - t0
    body.read(sensed[k])
    for name, q in adr.items():
      joints[name][k] = d.qpos[q]
  _stable(d, "in its rows")
  if noise_seed is not None:
    from pluggybot.perception.encoders import torque_readings
    for c, which in ((0, "shoulder"), (1, "elbow")):
      sensed[:, c] = torque_readings(sensed[:, c], f"imagination{noise_seed}/{which}")
  return Readings(t=t, sensed=sensed, joints=joints)
