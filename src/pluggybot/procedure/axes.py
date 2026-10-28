"""The motor-and-sensor level: named AXES a program may move and named
SENSORS it may read (issue #166).

A real servo takes a setpoint, a lead screw takes a target at a bounded
speed, a motor controller takes (v, w). That is the finest grain an agent's
program can honestly reach on real hardware, and it is the grain here:
`move(axis, target)` walks ONE actuator's setpoint through
`Body.ramp_routine` (the ramping rule, inside the primitive), and
`read(sensor)` is one scalar off the world. Nothing below this level is
reachable from a program -- the fence in tests/test_procedure.py lists who
writes `data.ctrl`, and this module is not on it.

Both are REGISTRIES, not constants: a new tool or body adds its entries
and the language does not change. A body's own axes are always present --
the quadruped's `shoulder` and `elbow` (`BODY_AXES`) -- and its senses; a
sensor a tool carries is present when that tool is on the fork
(`requires`).
"""

import math
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pluggybot.legs.arm import ARM_SLEW, ArmSpec
from pluggybot.perception.watch import WATCH_HZ
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
  #: how often its part reads it afresh while the body moves, s -- what a
  #: hand's move may stop on, read no oftener (#554) -- or None for what a
  #: verb last reported, which no move changes
  every_s: float | None = None
  #: how a stop condition reads it where `read` reads its part afresh: as
  #: the body's own loop last read it (`Body.hand_sensed`), so a condition
  #: that never comes true moves the flight not at all (#554)
  live: Callable[..., float] | None = None


#: How often a hand's readings are fresh while it moves, s: once a command
#: of its arm's (`humanoid.reach.EVERY_S`, which a test holds this to) --
#: its place off its encoders through its model of itself, its force off
#: the whole body's balance (#547), its gripper's word -- and the clock
#: they are read by.
HAND_EVERY_S = 0.01


#: The quadruped's arm (issue #405): each joint is an axis and a sensor,
#: in radians, walked by the arm's own driver (`legs.arm.ArmDriver`, at most
#: `ARM_SLEW`) -- the shoulder's elevation off straight ahead, the elbow's
#: angle off the upper arm's line. The ranges are the joints' own.
ARM_JOINTS = ("shoulder", "elbow")
_ARM = ArmSpec()


def _ramp(actuator: str, settle: float = 0.5):
  def run(life, target: float) -> Routine:
    axis = next(a for a in AXES.values() if a.actuator == actuator)
    # THIS robot's (issue #167): the handle says which
    act = life.model.actuator(life.body.handle.el(actuator)).id
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
  "time": Sensor("time", lambda life: float(life.data.time), "sim seconds",
                 every_s=HAND_EVERY_S),
  "bumper": Sensor("bumper", lambda life: 1.0 if life.body.pressing else 0.0,
                   "1 while its body presses against something"),
  "shoulder": Sensor("shoulder", lambda life: _joint(life, "arm_shoulder"),
                     "the arm's shoulder, rad, measured"),
  "elbow": Sensor("elbow", lambda life: _joint(life, "arm_elbow"),
                  "the arm's elbow, rad, measured"),
  "claw.holding": Sensor(
    "claw.holding", lambda life: 1.0 if life.body.held_cube() is not None else 0.0,
    "1 while the claw's jaws hold a cube, both pads on it", requires="module_claw"),
  # THE ARM'S SCALE (issue #407): each motor's torque as its driver reports
  # it, off the current (`perception.encoders.torque_reading`)
  "shoulder.torque": Sensor(
    "shoulder.torque", lambda life: life.body.arm_torque(life.body.actuator("arm_shoulder")),
    "the shoulder motor's torque, N*m, as its driver reports it: what it "
    "holds the arm up with, + raising"),
  "elbow.torque": Sensor(
    "elbow.torque", lambda life: life.body.arm_torque(life.body.actuator("arm_elbow")),
    "the elbow motor's torque, N*m, as its driver reports it: what it holds "
    "the forearm and whatever it carries up with, + raising"),
}
#: What every body reads (issue #387): its pack, its wallet, the clock and
#: its bumper...
BODY_SENSORS = ("battery.frac", "battery.wh", "points", "time", "bumper")
#: ...and a legged body its arm's two joints (issue #405) and its motors'
#: torques (#407).
ARM_TORQUES = ("shoulder.torque", "elbow.torque")
LEGS_SENSORS = BODY_SENSORS + ARM_JOINTS + ARM_TORQUES
#: ...and what the claw's verbs feel (issue #407), where the world has it.
CLAW_SENSORS = ("claw.holding",)


# ---- the humanoid's hands (issue #493) -------------------------------------------
#
# A hand's place and force are said in the body's STANCE frame (`Body.
# stance`): from the floor between its feet, x ahead, y to its left, z up, m
# -- what a procedure reaches in, so it needs no trigonometry to say "in
# front of me". A verb's report is read as sensors too (`life.reports`, the
# last of each verb's): the language returns numbers only through `read`.

#: The humanoid's two hands, by the names a verb takes them by.
HANDS = ("left", "right")


def in_stance(life, p, point: bool = True) -> tuple[float, float, float]:
  """A believed world point (or, `point` False, a direction) in the body's
  stance frame."""
  x, y, yaw = life.body.stance
  c, s = math.cos(yaw), math.sin(yaw)
  dx, dy = (p[0] - x, p[1] - y) if point else (p[0], p[1])
  return c * dx + s * dy, -s * dx + c * dy, float(p[2])


def from_stance(life, p, point: bool = True) -> tuple[float, float, float]:
  """A stance-frame point (or direction) in the believed world frame."""
  x, y, yaw = life.body.stance
  c, s = math.cos(yaw), math.sin(yaw)
  wx, wy = c * p[0] - s * p[1], s * p[0] + c * p[1]
  return (wx + x, wy + y, float(p[2])) if point else (wx, wy, float(p[2]))


class NoReading(LookupError):
  """A sensor read before anything gave it a value: a verb's report read
  before the verb ran. The language stops the run, saying which."""


def reported(verb: str, key: str, life) -> float:
  """One number of a verb's last report (`life.reports`)."""
  rep = (getattr(life, "reports", None) or {}).get(verb)
  if rep is None or key not in rep:
    raise NoReading(f"{verb}.{key} has no reading: `{verb}` has not run")
  return float(rep[key])


def _hand_at(side: str, k: int):
  return lambda life: in_stance(life, life.body.hand_at(side))[k]


def _hand_sensed(side: str, k: int):
  return lambda life: in_stance(life, life.body.hand_sensed(side))[k]


def _hand_force(side: str, k: int):
  return lambda life: in_stance(life, life.body.hand_force(side), point=False)[k]


for _s in HANDS:
  for _k, _a in enumerate("xyz"):
    SENSORS[f"{_s}.{_a}"] = Sensor(
      f"{_s}.{_a}", _hand_at(_s, _k),
      f"where your {_s} hand's grip is, m, your stance frame ({_a})", every_s=HAND_EVERY_S,
      live=_hand_sensed(_s, _k))
    SENSORS[f"{_s}.f{_a}"] = Sensor(
      f"{_s}.f{_a}", _hand_force(_s, _k),
      f"the force your {_s} hand pushes with, N, your stance frame ({_a}), off your "
      "motor currents less your own weight and your joints' friction -- your whole "
      "body's while that hand is out and you stand, else its arm's alone; it reads "
      "truest held still", every_s=HAND_EVERY_S)
  SENSORS[f"{_s}.held"] = Sensor(
    f"{_s}.held", (lambda side: lambda life: 1.0 if life.body.gripped(side) else 0.0)(_s),
    f"1 while your {_s} gripper says it holds something: its jaws stopped short of "
    "closed, squeezing", every_s=HAND_EVERY_S)
  SENSORS[f"{_s}.gap"] = Sensor(
    f"{_s}.gap", (lambda side: lambda life: life.body.grip_gap(side))(_s),
    f"how far apart your {_s} gripper's jaws are, m: what it holds, how wide",
    every_s=HAND_EVERY_S)
for _k, _a in enumerate("xyz"):
  SENSORS[f"spot.{_a}"] = Sensor(
    f"spot.{_a}", (lambda key: lambda life: reported("spot", key, life))(_a),
    f"where the tag the last `spot` saw is, m, your stance frame ({_a}): the middle "
    "of the face it read")
  SENSORS[f"spot.n{_a}"] = Sensor(
    f"spot.n{_a}", (lambda key: lambda life: reported("spot", key, life))(f"n{_a}"),
    f"which way that face looks ({_a}): a unit vector from it into the thing it is "
    "printed on")
  SENSORS[f"spot.e{_a}"] = Sensor(
    f"spot.e{_a}", (lambda key: lambda life: reported("spot", key, life))(f"e{_a}"),
    f"which way that face's edges run ({_a}): a unit vector along one of them, the "
    "other at right angles to it and to the face's look")
SENSORS["spot.range"] = Sensor(
  "spot.range", lambda life: reported("spot", "range", life),
  "how far that tag was from the camera, m")
#: What the humanoid's procedures read (#493): the clock, each hand's place,
#: its force and its gripper's word, and the last `spot`'s tag.
HAND_SENSORS = (("time",) + tuple(f"{s}.{k}" for s in HANDS
                                  for k in ("x", "y", "z", "fx", "fy", "fz", "held", "gap"))
                + tuple(f"spot.{k}" for k in ("x", "y", "z", "nx", "ny", "nz", "ex", "ey", "ez",
                                              "range")))

#: A tag WATCHED while the body moves (#554's stage 2, `Body.watch`): the
#: newest reading the robot has -- whether its picture shows the tag -- and
#: where the newest picture that showed it put the tag, how its face looked
#: and how long ago; fresh at the camera's rate (`perception.watch`).
WATCH_EVERY_S = 1.0 / WATCH_HZ


def _sighting(name: str, life) -> dict:
  got = life.body.sighting()
  if got is None:
    raise NoReading(f"{name} has no reading: `watch` has not run")
  return got


def _last_seen(name: str, life) -> dict:
  got = _sighting(name, life)
  if got["last"] is None:
    raise NoReading(f"{name} has no reading: tag {got['tag']} has not been seen since "
                    "`watch` began")
  return got["last"]


def _watched(name: str, k: int, key: str | None):
  """`watch.<a>` off the newest picture that showed the tag: its place
  (`key` None), or its `normal` or `edge`, in the stance frame now."""
  def read(life) -> float:
    last = _last_seen(name, life)
    if key is None:
      return in_stance(life, (last["x"], last["y"], last["z"]))[k]
    return in_stance(life, last[key], point=False)[k]
  return read


for _k, _a in enumerate("xyz"):
  SENSORS[f"watch.{_a}"] = Sensor(
    f"watch.{_a}", _watched(f"watch.{_a}", _k, None),
    f"where the tag you watch was in the newest picture that showed it, m, your stance "
    f"frame ({_a}): the middle of the face it read", every_s=WATCH_EVERY_S)
  SENSORS[f"watch.n{_a}"] = Sensor(
    f"watch.n{_a}", _watched(f"watch.n{_a}", _k, "normal"),
    f"which way that face looked ({_a}): a unit vector from it into the thing it is "
    "printed on", every_s=WATCH_EVERY_S)
  SENSORS[f"watch.e{_a}"] = Sensor(
    f"watch.e{_a}", _watched(f"watch.e{_a}", _k, "edge"),
    f"which way that face's edges ran ({_a}): a unit vector along one of them",
    every_s=WATCH_EVERY_S)
SENSORS["watch.range"] = Sensor(
  "watch.range", lambda life: float(_last_seen("watch.range", life)["range"]),
  "how far that tag was from the camera then, m", every_s=WATCH_EVERY_S)
SENSORS["watch.seen"] = Sensor(
  "watch.seen", lambda life: 1.0 if _sighting("watch.seen", life)["seen"] else 0.0,
  "1 while your newest picture shows the tag you watch, 0 while it does not: out of "
  "view, or hidden -- by your own hand too", every_s=WATCH_EVERY_S)
SENSORS["watch.age"] = Sensor(
  "watch.age", lambda life: float(life.data.time) - _last_seen("watch.age", life)["at"],
  "how long ago the newest picture that showed it was taken, s: how old `watch.x` and "
  "the rest are", every_s=WATCH_EVERY_S)
#: ...what a procedure with a watch reads.
WATCH_SENSORS = tuple(f"watch.{k}" for k in ("x", "y", "z", "nx", "ny", "nz", "ex", "ey", "ez",
                                             "range", "seen", "age"))

# The humanoid's arm joints: each an axis (the motor level a hand's verbs
# are built from), walked by `Body.ramp_routine`. ⚠ No sensor reads one:
# the quadruped's `shoulder` and `elbow` read the simulator's `qpos`, a
# stand-in no new sense extends -- a hand's place is read off its encoders
# through the robot's model of itself (`left.x`, `Body.hand_at`).
#: How fast a humanoid's arm joint is walked, rad/s (`humanoid.reach.JOINT_RATE`).
HUMANOID_SLEW = 1.0


def _humanoid_axes() -> tuple[str, ...]:
  from pluggybot.humanoid import model as hm
  names = []
  for s in HANDS:
    for j in hm.ARM:
      name = f"{s}_{j}"
      lo, hi = hm.joint_range(name)
      AXES[name] = Axis(name, float(lo), float(hi), HUMANOID_SLEW, "rad",
                        f"your {s} arm's {j.replace('_', ' ')} joint", actuator=name,
                        run=_ramp(name), setpoint=_held(name))
      names.append(name)
  return tuple(names)


HUMANOID_JOINTS = _humanoid_axes()
#: Each body's own axes, present whatever is on its fork.
BODY_AXES = {"quadruped": ARM_JOINTS, "humanoid": HUMANOID_JOINTS}
#: ...and the reports of a pull (#493, the force move #525): how far it
#: went, the hardest it pushed and the way the hand was going at its end.
PULL_SENSORS = ("pull.moved", "pull.force", "pull.dx", "pull.dy", "pull.dz")
SENSORS["pull.moved"] = Sensor(
  "pull.moved", lambda life: reported("pull", "moved", life),
  "how far the last `pull` took the hand from where it began, m")
SENSORS["pull.force"] = Sensor(
  "pull.force", lambda life: reported("pull", "force", life),
  "the hardest the last `pull` pushed along its way, N, as your hand felt it")
#: ...and where the last step took the stance (#525), in the stance frame
#: it stood in before: what a place read before it is turned into.
STEP_SENSORS = ("step.dx", "step.dy", "step.turn")
SENSORS["step.dx"] = Sensor(
  "step.dx", lambda life: reported("step", "dx", life),
  "how far ahead your feet went in the last `step`, m, in the stance frame you stood in "
  "before it")
SENSORS["step.dy"] = Sensor(
  "step.dy", lambda life: reported("step", "dy", life),
  "how far to your left your feet went in the last `step`, m, in that frame")
SENSORS["step.turn"] = Sensor(
  "step.turn", lambda life: reported("step", "turn", life),
  "how far you turned in the last `step`, rad (left, +)")
#: ...and what `measure` saw (#493, the arm with sight): a flat top's height
#: and where it ends, the stance frame.
MEASURE_SENSORS = tuple(f"object.{k}" for k in ("top", "near", "far", "left", "right"))
for _k, _doc in (("top", "the height of the flat top `measure` last saw, m"),
                 ("near", "how far ahead that top begins, m"),
                 ("far", "how far ahead that top ends, m"),
                 ("left", "how far to your left that top reaches, m"),
                 ("right", "how far to your left its right edge is (negative: to your right), m")):
  SENSORS[f"object.{_k}"] = Sensor(
    f"object.{_k}", (lambda key: lambda life: reported("measure", key, life))(_k), _doc)
for _k, _a in enumerate("xyz"):
  SENSORS[f"pull.d{_a}"] = Sensor(
    f"pull.d{_a}", (lambda key: lambda life: reported("pull", key, life))(f"d{_a}"),
    f"the way your hand was going at the end of the last `pull` ({_a}), your stance "
    "frame: a unit vector, where what it held let it go")


#: ...and whether the last of each hand's move stopped on its reading
#: (#554, `until`): 1 where it did, 0 where it ended otherwise.
STOPPED_SENSORS = ("reach.stopped", "move_hand.stopped", "pull.stopped")


def _stopped(verb: str):
  def read(life) -> float:
    got = ((getattr(life, "reports", None) or {}).get("stopped") or {}).get(verb)
    if got is None:
      raise NoReading(f"{verb}.stopped has no reading: `{verb}` has not run")
    return float(got)
  return read


for _v in ("reach", "move_hand", "pull"):
  SENSORS[f"{_v}.stopped"] = Sensor(
    f"{_v}.stopped", _stopped(_v),
    f"1 if your last `{_v}` stopped on its `until`, 0 if it ended otherwise")


def setpoints(life) -> dict[str, float]:
  """What every axis is COMMANDED to, the body's own. What a death reads
  (issue #362). A setpoint, not a position: a joint stalled against
  something reads what it was asked for, and the gap is the finding. An
  axis the body has no actuator for (the stub's) is left out."""
  out: dict[str, float] = {}
  for axis in AXES.values():
    try:
      value = (axis.setpoint(life) if axis.setpoint is not None
               else life.body.setpoint(life.body.actuator(axis.actuator)))
    except KeyError:
      continue
    out[axis.name] = round(value, 4)
  return out


def describe() -> dict:
  """The registries as data, for a prompt and for a validator's message."""
  return {"axes": [{"name": a.name, "lo": a.lo, "hi": a.hi, "speed": a.speed,
                    "unit": a.unit, "doc": a.doc}
                   for a in AXES.values()],
          "sensors": [{"name": s.name, "doc": s.doc, "requires": s.requires}
                      for s in SENSORS.values()]}
