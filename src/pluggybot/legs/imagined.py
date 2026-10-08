"""The quadruped in a world of its own imagining (issue #466): the body's half
of `imagination/`. Its CAD with the tool it carries (`body_spec`), put down
as it knew itself and driven by a record's commands (`ImaginedBody`) -- the
same arm driver and leg drivers the served body runs, in a world that is
not the world. The imagination reaches the machine through nothing else
(`tests/test_body.py`'s fence: a body's classes stay on its side).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot.legs import rack as rk
from pluggybot.legs.arm import ArmDriver, tool_forces, tool_payload
from pluggybot.legs.drivers import Drivers
from pluggybot.legs.model import CHOSEN, JOINT_NAMES, attachable
from pluggybot.rack.coupling import PEG_ABOVE_BODY
from pluggybot.telemetry.protocol import ROBOT_ROOT


def body_spec(carrying: str | None = None) -> mujoco.MjSpec:
  """The robot as its CAD has it (`legs.model.attachable`), and `carrying`
  -- a module of its rack (`legs.rack.TOOL_BAYS`) -- loose beside it until
  `ImaginedBody.place` seats it on the fork."""
  spec = mujoco.MjSpec.from_string('<mujoco><compiler angle="radian" autolimits="true"/>'
                                   '</mujoco>')
  spec.attach(attachable(CHOSEN), prefix="", frame=spec.worldbody.add_frame())
  if carrying is not None:
    if carrying not in rk.TOOL_BAYS:
      raise ValueError(f"{carrying!r} is no tool of the robot's: it carries one of "
                       f"{', '.join(rk.TOOL_BAYS)}")
    tool = mujoco.MjSpec.from_string(
      f'<mujoco><compiler angle="radian"/><default>{rk.tool_default(carrying)}</default>'
      f'<worldbody>'
      + rk.tool_xml(carrying, (0.0, 0.0, 1.0),
                    mass=rk.TOOL_KG[carrying] - rk.FACE_KG.get(carrying, 0.0),
                    face=rk.tool_face(carrying))
      + f'</worldbody><actuator>{rk.tool_actuators_xml((carrying,))}</actuator></mujoco>')
    spec.attach(tool, prefix="", frame=spec.worldbody.add_frame())
  return spec


def force_apart(sensed: np.ndarray, other: np.ndarray) -> np.ndarray:
  """What two sets of the arm's readings put on the tool apart, N, row by
  row (forward, up, the torso's frame): `other`'s torques less `sensed`'s,
  through the arm's Jacobian at `sensed`'s pose (`legs.arm.tool_force`).
  Both are rows of `record.SENSED`'s columns: the two torques, then the
  shoulder and the forearm."""
  sensed, other = np.asarray(sensed, dtype=float), np.asarray(other, dtype=float)
  dtau = other[:, :2] - sensed[:, :2]
  return tool_forces(CHOSEN.arm, sensed[:, 2], sensed[:, 3], dtau[:, 0], dtau[:, 1])


def level_of(attitude) -> np.ndarray:
  """Level from the torso, off a roll and a pitch (`attitude`)."""
  roll, pitch = attitude
  cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
  return (np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
          @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))


#: The encoders' rates for friction's sign are read off positions smoothed
#: over this many rows (the record carries positions, quantised).
RATE_ROWS = 25


def felt(record, friction) -> np.ndarray:
  """What the world put on the tool, N, row by row (forward, up, the
  torso's frame), off a record alone: its torque readings less the arm's
  own weight -- its CAD's links at its encoders, levelled by its start's
  tilt (`ArmDriver.gravity_sensed`) -- and less the `friction` it
  calibrated where a joint turns (`legs.arm.less_friction`), through the
  arm's Jacobian: `legs.probe.Recorder.force`, after the fact."""
  from pluggybot.imagination.compile import robot_spec
  from pluggybot.legs.arm import less_friction
  m = robot_spec(record.start.carrying).compile()
  body = ImaginedBody(m, mujoco.MjData(m), record.start.carrying)
  level = level_of(record.start.attitude)
  q = np.asarray(record.sensed[:, 2:4], dtype=float)
  weight = np.array([body.arm.gravity_sensed(row, level) for row in q])
  # padded with its ends, never zeros: a zero pad read an arm holding still
  # turning 45 rad/s at either end, and took its friction off there
  kernel = np.ones(RATE_ROWS) / RATE_ROWS
  half = RATE_ROWS // 2
  smooth = np.column_stack([
    np.convolve(np.pad(q[:, i], (half, RATE_ROWS - 1 - half), mode="edge"), kernel,
                mode="valid") for i in range(2)])
  rates = np.gradient(smooth, record.dt, axis=0)
  tau = less_friction(record.sensed[:, :2] - weight, rates, friction)
  return tool_forces(CHOSEN.arm, q[:, 0], q[:, 1], tau[:, 0], tau[:, 1])


def _quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
  """Yaw about z, then pitch about the new y, then roll about the new x."""
  cr, sr = math.cos(roll / 2), math.sin(roll / 2)
  cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
  cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
  return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                   cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr])


class ImaginedBody:
  """The robot's drivers in an imagined world: put down (`place`), driven a
  command row a step (`apply`: the arm's targets and gains, the claw's
  servos; the legs hold nothing, as lying), and read (`read`: the arm
  motors' torques, the arm's two coordinates). A row is `record.COMMANDS`'
  order: shoulder, forearm, Kp, Kd, slide, jaws -- the slide and the jaws
  the claw's (`tools.claw` sends both jaws one command, each one's travel
  off shut), and with another tool sent nowhere."""

  def __init__(self, model, data, carrying: str | None) -> None:
    self.carrying = carrying
    self.rebind(model, data)

  def rebind(self, model, data) -> None:
    """Onto a world, by name: an imagined world is built for a rollout and
    never recompiled, but a holder of one is rebindable all the same."""
    self.model, self.data = model, data
    self.legs = Drivers(model, data)
    self.legs.limp()
    self.arm = ArmDriver(model, data, CHOSEN.arm)
    self.slide = self.jaws = None
    self.arm.payload = tool_payload(model, self.carrying, CHOSEN.arm)
    if self.carrying == "module_claw":
      self.slide = model.actuator(rk.CLAW_SLIDE).id
      self.jaws = [model.actuator(j).id for j in rk.CLAW_JAWS]

  def _set(self, name: str, q) -> None:
    j = self.model.joint(name).id
    adr = int(self.model.jnt_qposadr[j])
    q = np.atleast_1d(np.asarray(q, dtype=float))
    self.data.qpos[adr:adr + len(q)] = q

  def place(self, pose, attitude, legs, arm, first) -> None:
    """Down as the robot knew itself: at its believed `pose` (x, y, yaw),
    tilted by `attitude` (roll, pitch), lying on its belly as its CAD's
    `lie` keyframe has it, its legs' and arm's joints at their encoders
    (`arm`: the shoulder and the forearm's absolute angle), the tool it
    carries seated on its fork, square to its plate, and its servos where
    the `first` row sends them -- the jaws with nothing in them: a record
    begins before the grip. Forwards the data."""
    m, d = self.model, self.data
    x, y, yaw = pose
    self._set(f"{ROBOT_ROOT}_root",
              [x, y, CHOSEN.belly_depth + 0.001, *_quat(*attitude, yaw)])
    for name, q in zip(JOINT_NAMES, legs):
      self._set(name, q)
    shoulder, fore = arm
    self._set("arm_shoulder", shoulder)
    self._set("arm_elbow", fore - shoulder)
    self._set("arm_wrist", -fore)                     # the plate kept level
    if self.carrying is not None:
      mujoco.mj_kinematics(m, d)
      root = m.body(ROBOT_ROOT).id
      torso = d.xmat[root].reshape(3, 3)
      seat = d.site_xpos[m.site("arm_seat").id]
      peg = seat + torso @ np.array([0.0, 0.0, CHOSEN.arm.fork.seat_rise() + 0.0003])
      # in the plate's frame, which is the torso's (the parallelogram), and
      # facing the robot: seated by yaw alone, a robot rolled 0.13 rad
      # dropped it off its fork (the review of #473)
      quat = np.zeros(4)
      mujoco.mju_mulQuat(quat, d.xquat[root], np.array([0.0, 0.0, 0.0, 1.0]))
      self._set(f"{self.carrying}_free",
                [*(peg - torso @ np.array([0.0, 0.0, PEG_ABOVE_BODY])), *quat])
      if self.slide is not None:
        self._set(rk.CLAW_SLIDE_JOINT, first[4])
        for jaw in rk.CLAW_JAWS:
          self._set(jaw, first[5])
    mujoco.mj_forward(m, d)

  def apply(self, row) -> None:
    """One row of commands, before the step: its targets are already ramped
    by the robot that sent them, so the driver runs on them as they are."""
    arm, d = self.arm, self.data
    arm.target[0] = arm.goal[0] = row[0]
    arm.target[1] = arm.goal[1] = row[1]
    arm.kp, arm.kd = float(row[2]), float(row[3])
    if self.slide is not None:
      d.ctrl[self.slide] = row[4]
      d.ctrl[self.jaws[0]] = d.ctrl[self.jaws[1]] = row[5]
    arm.step()

  def read(self, out: np.ndarray) -> None:
    """After the step: the two arm motors' torques (N*m) and the arm's two
    coordinates (rad) into `out`."""
    d, act = self.data, self.arm.act
    out[0] = d.actuator_force[act[0]]
    out[1] = d.actuator_force[act[1]]
    out[2:4] = self.arm.q()
