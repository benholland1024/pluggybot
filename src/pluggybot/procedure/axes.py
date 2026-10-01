"""The motor-and-sensor level: named AXES a program may move and named
SENSORS it may read (issue #166; the floor issue #168's tools stand on).

A real servo takes a setpoint, a lead screw takes a target at a bounded
speed, a motor controller takes (v, w). That is the finest grain an agent's
program can honestly reach on real hardware, and it is the grain here:
`move(axis, target)` walks ONE actuator's setpoint through
`Body.ramp_routine` (the ramping rule, inside the primitive), and
`read(sensor)` is one scalar off the world. Nothing below this level is
reachable from a program -- the fence in tests/test_procedure.py lists who
writes `data.ctrl`, and this module is not on it.

Both are REGISTRIES, not constants: an axis or a sensor is present when the
tool that carries it is on the fork (`requires`), and a tool built from a
spec (#168) registers its own by the same call. A body's own axes are
always present -- the quadruped's `shoulder` and `elbow` (`BODY_AXES`) --
and its senses.
"""

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pluggybot.legs.arm import ARM_SLEW, ArmSpec
from pluggybot.tick import Routine


@dataclass(frozen=True)
class Axis:
  """One position setpoint a program may move, with its envelope."""

  name: str
  lo: float
  hi: float
  speed: float                  # units per second, the setpoint's ceiling
  unit: str
  doc: str
  #: the module that must be on the fork for this axis to exist ("" = body)
  requires: str = ""
  #: how to run it: (life, target) -> Routine. Default: one actuator by name.
  actuator: str = ""
  run: Callable[..., Routine] | None = None
  #: what it is commanded to, where that is not one actuator's ctrl:
  #: (life) -> float, in the axis's own units (`setpoints`)
  setpoint: Callable[..., float] | None = None


@dataclass(frozen=True)
class Sensor:
  """One scalar a program may read, measured off the world."""

  name: str
  read: Callable[..., float]    # (life) -> float
  doc: str
  requires: str = ""


#: The quadruped's arm (issue #405): each joint is an axis and a sensor,
#: in radians, walked by the arm's own driver (`legs.arm.ArmDriver`, at most
#: `ARM_SLEW`) -- the shoulder's elevation off straight ahead, the elbow's
#: angle off the upper arm's line. The ranges are the joints' own.
ARM_JOINTS = ("shoulder", "elbow")
_ARM = ArmSpec()


def _ramp(actuator: str, settle: float = 0.5):
  def run(life, target: float) -> Routine:
    axis = next(a for a in AXES.values() if a.actuator == actuator)
    # a body axis is THIS robot's; a tool's axis is the module's and shared
    # (issue #167) -- the handle says which
    name = actuator if axis.requires else life.body.handle.el(actuator)
    act = life.model.actuator(name).id
    return (yield from life.body.ramp_routine(act, float(target), axis.speed,
                                              settle=settle))
  return run


def _held(actuator: str):
  """What the body holds one of its joints to (`Body.setpoint`): a torque
  motor's `ctrl` is no setpoint."""
  return lambda life: life.body.setpoint(life.body.actuator(actuator))


AXES: dict[str, Axis] = {
  "shoulder": Axis("shoulder", *_ARM.shoulder_range, ARM_SLEW, "rad",
                   "the arm's shoulder: 0 is the upper arm straight ahead, + "
                   f"raises it; {_ARM.stow[0]:.2f} lies it back along the body, "
                   "folded", actuator="arm_shoulder", setpoint=_held("arm_shoulder")),
  "elbow": Axis("elbow", *_ARM.elbow_range, ARM_SLEW, "rad",
                "the arm's elbow: the forearm off the upper arm's line, 0 "
                f"straight, - bends it down; {_ARM.stow[1]:.2f} folds it back "
                "along the upper arm", actuator="arm_elbow",
                setpoint=_held("arm_elbow")),
}
for _a in list(AXES.values()):
  if _a.run is None:
    AXES[_a.name] = Axis(**{**_a.__dict__, "run": _ramp(_a.actuator)})


def _joint(life, name: str, body: bool = True) -> float:
  """A joint position, measured. `body` joints are this robot's (prefixed
  by its handle, issue #167); a module's joint is the world's."""
  if body:
    name = life.body.handle.el(name)
  return float(life.data.qpos[life.model.joint(name).qposadr[0]])


def geom_contact(life, geom: str) -> float:
  """Is this geom touching anything outside its own root body -- a bumper's
  criterion, read off the contact list: what a microswitch a built tool
  carries reads (workshop/build.py, issue #199). A geom the world does not
  have reads 0: the tool is not on the rack yet, or was retired."""
  model, data = life.model, life.data
  try:
    gid = model.geom(geom).id
  except KeyError:
    return 0.0
  own = int(model.body_rootid[model.geom_bodyid[gid]])
  for i in range(data.ncon):
    c = data.contact[i]
    if gid in (c.geom1, c.geom2):
      other = c.geom2 if c.geom1 == gid else c.geom1
      if int(model.body_rootid[model.geom_bodyid[other]]) != own:
        return 1.0
  return 0.0


def noise(life, key: str, sigma: float) -> float:
  """One Gaussian sample, `sigma` wide, DETERMINISTIC on the physics step
  and the sensor: the same step reads the same value twice (a sensor
  sampled faster than it updates does), the next step reads a fresh one,
  and one world replays byte-identical. Seeded off a checksum rather than
  `hash()`, which Python salts per process."""
  step = int(round(float(life.data.time) / float(life.model.opt.timestep)))
  prefix = getattr(getattr(life, "body", None), "handle", None)
  tag = f"{key}:{getattr(prefix, 'prefix', '')}:{step}"
  seed = zlib.crc32(tag.encode())
  return float(np.random.default_rng(seed).normal(0.0, sigma))


SENSORS: dict[str, Sensor] = {
  "battery.frac": Sensor("battery.frac", lambda life: float(life.battery.fraction),
                         "pack fraction, 0..1"),
  "battery.wh": Sensor("battery.wh", lambda life: float(life.battery.energy_wh),
                       "pack energy, Wh"),
  "points": Sensor("points", lambda life: float(life.ledger.balance())
                   if life.ledger is not None else 0.0, "the balance"),
  "time": Sensor("time", lambda life: float(life.data.time), "sim seconds"),
  "bumper": Sensor("bumper", lambda life: 1.0 if life.body.pressing else 0.0,
                   "1 while its body presses against something"),
  "shoulder": Sensor("shoulder", lambda life: _joint(life, "arm_shoulder"),
                     "the arm's shoulder, rad, measured"),
  "elbow": Sensor("elbow", lambda life: _joint(life, "arm_elbow"),
                  "the arm's elbow, rad, measured"),
}
#: What every body reads (issue #387): its pack, its wallet, the clock and
#: its bumper...
BODY_SENSORS = ("battery.frac", "battery.wh", "points", "time", "bumper")
#: ...and a legged body its arm's two joints (issue #405).
LEGS_SENSORS = BODY_SENSORS + ARM_JOINTS
#: Each body's own axes, present whatever is on its fork.
BODY_AXES = {"quadruped": ARM_JOINTS}


#: The ramp as the public name a built tool registers its axes with
#: (workshop/build.py): one actuator, the axis's own speed, settled.
ramped = _ramp


def register_axis(axis: Axis) -> None:
  """A tool's axis, added by the thing that built the tool (#168)."""
  AXES[axis.name] = axis


def register_sensor(sensor: Sensor) -> None:
  SENSORS[sensor.name] = sensor


def setpoints(life, module: str | None) -> dict[str, float]:
  """What every axis present is COMMANDED to: the body's, and those of the
  tool on the fork (`module`; None is an empty fork). What a death reads
  (issue #362). A setpoint, not a position: a joint stalled against
  something reads what it was asked for, and the gap is the finding. An
  axis whose actuator the world no longer has (a retired tool) is left out."""
  out: dict[str, float] = {}
  for axis in AXES.values():
    if axis.requires and axis.requires != module:
      continue
    try:
      if axis.setpoint is not None:
        value = axis.setpoint(life)
      elif axis.requires:
        value = float(life.data.ctrl[life.model.actuator(axis.actuator).id])
      else:
        value = life.body.setpoint(life.body.actuator(axis.actuator))
    except KeyError:
      continue
    out[axis.name] = round(value, 4)
  return out


def describe() -> dict:
  """The registries as data, for a prompt and for a validator's message."""
  return {"axes": [{"name": a.name, "lo": a.lo, "hi": a.hi, "speed": a.speed,
                    "unit": a.unit, "doc": a.doc, "requires": a.requires}
                   for a in AXES.values()],
          "sensors": [{"name": s.name, "doc": s.doc, "requires": s.requires}
                      for s in SENSORS.values()]}
