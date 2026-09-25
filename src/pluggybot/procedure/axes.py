"""The motor-and-sensor level: named AXES a program may move and named
SENSORS it may read (issue #166; the floor issue #168's tools stand on).

A real servo takes a setpoint, a lead screw takes a target at a bounded
speed, a motor controller takes (v, w). That is the finest grain an agent's
program can honestly reach on real hardware, and it is the grain here:
`move(axis, target)` walks ONE actuator's setpoint through
`HubSwap.ramp_routine` (the ramping rule, inside the primitive), and
`read(sensor)` is one scalar off the world. Nothing below this level is
reachable from a program -- the fence in tests/test_procedure.py lists who
writes `data.ctrl`, and this module is not on it.

Both are REGISTRIES, not constants: an axis or a sensor is present when the
tool that carries it is on the fork (`requires`), and a tool built from a
spec (#168) registers its own by the same call. The body's axes (`lift`,
`arm`) and the mission's sensors are always present.

Ranges and speeds are the numbers the tools already ramp with -- measured,
and re-typed here from their constants rather than from nothing.
"""

import math
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pluggybot.rack.coupling import CLAW_JAW_TRAVEL, PEN_TRAVEL
from pluggybot.tick import Routine
from pluggybot.tools.drawing import CARRIAGE_SPEED, LIFT_MAX, LIFT_MIN, PRESS_MAX
from pluggybot.tools.gripper import JAW_SPEED, LIFT_SPEED


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


#: The arm's press probes at PRESS_STEP per 0.12 s; this is that rate, so a
#: program's `move("arm", ...)` cannot outrun what the plotter does itself.
ARM_SPEED = 0.02
ARM_MAX = PRESS_MAX
#: The seed gate's ctrlrange (rack/coupling.py's dispenser actuator).
GATE_MAX = 0.024
GATE_SPEED = 0.02


def _ramp(actuator: str, settle: float = 0.5):
  def run(life, target: float) -> Routine:
    axis = next(a for a in AXES.values() if a.actuator == actuator)
    # a body axis (lift, arm) is THIS robot's; a tool's axis is the module's
    # and shared (issue #167) -- the handle says which
    name = actuator if axis.requires else life.mission.swap.handle.el(actuator)
    act = life.model.actuator(name).id
    yield from life.mission.swap.ramp_routine(act, float(target), axis.speed,
                                              settle=settle)
  return run


def _jaws(life, opening: float) -> Routine:
  """The claw's two actuators as one axis: 0 open .. 1 closed, ramped."""
  from pluggybot.tools.gripper import ClawTool
  claw = ClawTool(life.model, life.data, life.mission.swap)
  yield from claw.jaws_routine(float(opening))


def _jaws_setpoint(life) -> float:
  """The opening `_jaws` last commanded, off one jaw's ctrl (both carry it)."""
  return -float(life.data.ctrl[life.model.actuator("claw_l").id]) / CLAW_JAW_TRAVEL


AXES: dict[str, Axis] = {
  "lift": Axis("lift", LIFT_MIN, LIFT_MAX, LIFT_SPEED, "m",
               "the mast, height of the fork", actuator="lift"),
  "arm": Axis("arm", 0.0, ARM_MAX, ARM_SPEED, "m",
              "the fork's reach; 0 is stowed, the driving configuration",
              actuator="arm"),
  "pen.carriage": Axis("pen.carriage", -PEN_TRAVEL, PEN_TRAVEL, CARRIAGE_SPEED,
                       "m", "the pen's slide along the peg; + is the "
                       "viewer's left", requires="module_pen",
                       actuator="pen_carriage"),
  "claw.jaws": Axis("claw.jaws", 0.0, 1.0, JAW_SPEED / CLAW_JAW_TRAVEL, "",
                    "the claw's opening, 0 wide .. 1 closed",
                    requires="module_claw", run=_jaws, setpoint=_jaws_setpoint),
  "seed.gate": Axis("seed.gate", 0.0, GATE_MAX, GATE_SPEED, "m",
                    "the dispenser's gate", requires="module_seed",
                    actuator="seed_gate"),
}
for _a in list(AXES.values()):
  if _a.run is None:
    AXES[_a.name] = Axis(**{**_a.__dict__, "run": _ramp(_a.actuator)})


def _joint(life, name: str, body: bool = True) -> float:
  """A joint position, measured. `body` joints are this robot's (prefixed
  by its handle, issue #167); a module's joint is the world's."""
  if body:
    name = life.mission.swap.handle.el(name)
  return float(life.data.qpos[life.model.joint(name).qposadr[0]])


def geom_contact(life, geom: str) -> float:
  """Is this geom touching anything outside its own root body -- the
  bumper's criterion (`HubSwap.pressing`), read off the contact list. The
  pen's shaft reads it; so does a microswitch a built tool carries
  (workshop/build.py, issue #199). A geom the world does not have reads 0:
  the tool is not on the rack yet, or was retired."""
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


def _pen_contact(life) -> float:
  return geom_contact(life, "module_pen_shaft")


def _claw_holding(life) -> float:
  from pluggybot.procedure.steps import _claw, _holding_anything
  claw = _claw(life)
  return 1.0 if claw is not None and _holding_anything(claw) else 0.0


def _seated(life) -> float:
  from pluggybot.procedure.steps import _carried
  from pluggybot.rack.coupling import module_power_contact
  tool = _carried(life)
  return 1.0 if tool and module_power_contact(
    life.model, life.data, tool, life.mission.swap.handle.prefix) else 0.0


def _last_look(life, key: str, default: float) -> float:
  look = getattr(life, "_last_look", None) or {}
  return float(look.get(key, default))


#: The noise on a load reading, N, one standard deviation. A 50 N-class
#: load cell (the igus lead screw's own rating) at 0.06 % of full scale --
#: the HX711-and-strain-gauge tier a real robot would carry, not a lab
#: balance. Set against the bench's job (challenge/bench.py, issue #227):
#: a reading is `load - tare` and both carry it, so one careful pair
#: scatters by ~4 g, inside the grade's 10 % on the lightest cube the bank
#: sets out; averaging a few beats it, and that is the robot's to find.
#: Deterministic, on purpose (`noise`): a sensor that read differently on
#: two runs of one world would make every mission test a different test.
LOAD_NOISE_N = 0.03


def noise(life, key: str, sigma: float) -> float:
  """One Gaussian sample, `sigma` wide, DETERMINISTIC on the physics step
  and the sensor: the same step reads the same value twice (a sensor
  sampled faster than it updates does), the next step reads a fresh one,
  and one world replays byte-identical. Seeded off a checksum rather than
  `hash()`, which Python salts per process."""
  step = int(round(float(life.data.time) / float(life.model.opt.timestep)))
  prefix = getattr(getattr(getattr(life, "mission", None), "swap", None), "handle", None)
  tag = f"{key}:{getattr(prefix, 'prefix', '')}:{step}"
  seed = zlib.crc32(tag.encode())
  return float(np.random.default_rng(seed).normal(0.0, sigma))


def _lift_force(life) -> float:
  """What the mast's lead screw is pushing with, N, + up: at rest, the
  weight it carries -- the carriage and the fork, a module, whatever the
  module holds. MEASURED (challenge/bench.py): a cube in the claw's jaws
  moves it by exactly `dm * g`. Read off the actuator's own force with a
  load cell's noise, so it is a scale exactly as far as a real one is."""
  act = life.mission.swap.lift_act
  return float(life.data.actuator_force[act]) + noise(life, "lift.force", LOAD_NOISE_N)


SENSORS: dict[str, Sensor] = {
  "battery.frac": Sensor("battery.frac", lambda life: float(life.battery.fraction),
                         "pack fraction, 0..1"),
  "battery.wh": Sensor("battery.wh", lambda life: float(life.battery.energy_wh),
                       "pack energy, Wh"),
  "points": Sensor("points", lambda life: float(life.ledger.balance())
                   if life.ledger is not None else 0.0, "the balance"),
  "time": Sensor("time", lambda life: float(life.data.time), "sim seconds"),
  "lift": Sensor("lift", lambda life: _joint(life, "lift_joint"),
                 "the mast's height, m, measured"),
  "arm": Sensor("arm", lambda life: _joint(life, "arm_joint"),
                "the fork's reach, m, measured"),
  "lift.force": Sensor("lift.force", _lift_force,
                       "what the mast's lead screw is pushing with, N (+ up): "
                       "at rest, the weight it carries, read with a load "
                       "cell's noise"),
  "bumper": Sensor("bumper", lambda life: 1.0 if life.mission.swap.pressing else 0.0,
                   "1 while the chassis presses against something"),
  "module.seated": Sensor("module.seated", _seated,
                          "1 when a module is on the fork and powered"),
  "pen.contact": Sensor("pen.contact", _pen_contact,
                        "1 while the pen touches something", requires="module_pen"),
  "pen.carriage": Sensor("pen.carriage",
                         lambda life: _joint(life, "pen_carriage_joint", body=False),
                         "the carriage's position, m", requires="module_pen"),
  "claw.holding": Sensor("claw.holding", _claw_holding,
                         "1 while both pads hold something", requires="module_claw"),
  "look.tag": Sensor("look.tag", lambda life: _last_look(life, "tag", -1.0),
                     "the nearest tag id the last look() decoded, -1 for none"),
  "look.range": Sensor("look.range", lambda life: _last_look(life, "range", math.inf),
                       "its forward distance, m"),
  "look.lateral": Sensor("look.lateral", lambda life: _last_look(life, "lateral", 0.0),
                         "its lateral offset, m, + to the camera's left"),
}


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
  (issue #362). A setpoint, not a position: a lift stalled against
  something reads what it was asked for, and the gap is the finding. An
  axis whose actuator the world no longer has (a retired tool) is left out."""
  out: dict[str, float] = {}
  for axis in AXES.values():
    if axis.requires and axis.requires != module:
      continue
    try:
      if axis.setpoint is not None:
        value = axis.setpoint(life)
      else:
        name = (axis.actuator if axis.requires
                else life.mission.swap.handle.el(axis.actuator))
        value = float(life.data.ctrl[life.model.actuator(name).id])
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
