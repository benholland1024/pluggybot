"""The probe against the truth (issue #466, stage 2): OUR grading of what the
robot sensed (`legs/probe.py`), by code that knows where the chest stands.
The robot never sees any of it.

  set_out     where drawn chest k stands in the house's empty storeroom and
              where the robot starts, seeded
  geometry    the robot's guess of the box against the truth, EACH IN ITS
              ROBOT'S FRAME: the guess through the belief it was laid in,
              the truth through the true pose at that moment -- the belief's
              own error the probe never pays (it measures and plans through
              one belief) and the imagination never sees (its record is laid
              in one)
  in_map      the true chest in the robot's map at the record's start: the
              world's pose carried through (true pose)^-1 and the belief
  replay      the record through the world's chest and through the
              best-expressible reference, each placed in the map, against
              what the robot sensed: what a perfect model leaves, and what
              the language's best leaves -- stage 3's bar for a poor fit
              (#466's decisions: "judged against what the best-expressible
              reference leaves on the same probe")
  turned      how far the held handle turned on its pin while the catch
              held, in the flight and in each replay: a record where one
              turned over is FLAGGED, never dropped

`scripts/probe_chest.py` flies the set-outs and reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity import chest as ch
from pluggybot.evaluation import imagined as im
from pluggybot.imagination.rollout import Readings, rollout

#: Where the robot starts: the house's storeroom, empty (`home.world`'s
#: x 22-28, y -6-0), facing +x -- #469's spike stood there.
START = (24.0, -3.0)
#: A set-out draws: the robot's heading off +x; the knob's PLUMB place this
#: far from the torso's centre, this far off that heading; and the chest
#: turned this far off the line to the knob (rad, m). Where it rests, swung
#: on its pin, the knob hangs 33 mm further: 0.78-0.98 m off and up to 10.1
#: deg wide over 128 set-outs -- past a cube's 0.9 m (`legs.claw`) in 51 --
#: and the probe found its tag in every one.
HEADING = math.radians(8.0)
KNOB_FROM = (0.75, 0.95)
BEARING = math.radians(10.0)
TURNED = math.radians(20.0)
#: `set_out(k)`'s seed is this plus k.
SETOUT_SEED = 4660
#: The held handle TURNED OVER on its pin past this while the catch held
#: its lid shut, deg: over two batches of 128 set-outs (#479, before its
#: review and after) the arm's pull turned it a median 4 and at most 15.8,
#: 21.0 and 23.1 in the two that began to turn over and fell back, and
#: 33.5-42.2 in the thirteen that turned over, every catch 3.3 N or more.
#: What follows such a release measures no fit (SimNotes, "The probe from
#: the robot's own senses").
TWISTED_DEG = 28.0


@dataclass(frozen=True)
class SetOut:
  """Chest k: its hidden parameters (`chest.draw(k)`), where it stands
  (its frame's origin on the floor, its yaw), and where the robot starts
  (x, y, heading)."""
  k: int
  lid: ch.Lid
  chest: tuple[float, float, float]
  start: tuple[float, float, float]


def set_out(k: int) -> SetOut:
  """Chest `k` in the storeroom (seeded): the robot at `START` turned up to
  `HEADING`, the knob's plumb place `KNOB_FROM` off it up to `BEARING` off
  its heading, the chest turned up to `TURNED` off the line to its knob."""
  if k < 0:
    raise ValueError(f"a set-out is a whole number from 0, not {k}")
  rng = np.random.default_rng(SETOUT_SEED + k)
  heading = float(rng.uniform(-HEADING, HEADING))
  bearing = heading + float(rng.uniform(-BEARING, BEARING))
  dist = float(rng.uniform(*KNOB_FROM))
  yaw = bearing + float(rng.uniform(-TURNED, TURNED))
  kx, ky = START[0] + dist * math.cos(bearing), START[1] + dist * math.sin(bearing)
  off = ch.HINGE + ch.PIN + ch.KNOB_OFF                  # the plumb knob, chest frame
  c, s = math.cos(yaw), math.sin(yaw)
  chest = (kx - (c * off[0] - s * off[1]), ky - (s * off[0] + c * off[1]), yaw)
  return SetOut(k=k, lid=ch.draw(k), chest=chest, start=(START[0], START[1], heading))


def truth(chest_pose) -> dict:
  """The chest's hinge (a point on its line, and its axis), its pin and its
  lid's front-top edge, world frame, off where it stands: the chest's own
  drawing (`activity.chest`)."""
  x, y, yaw = chest_pose
  c, s = math.cos(yaw), math.sin(yaw)

  def at(p) -> np.ndarray:
    return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])
  return {"hinge": at(ch.HINGE), "pin": at(ch.HINGE + ch.PIN),
          "axis": np.array([-s, c, 0.0]), "yaw": yaw}


def in_frame(p, pose) -> np.ndarray:
  """A world (or map) point in a robot's heading frame at `pose` (x, y,
  yaw): x ahead, y left, z as it is."""
  x, y, th = pose
  c, s = math.cos(th), math.sin(th)
  dx, dy = float(p[0]) - x, float(p[1]) - y
  return np.array([c * dx + s * dy, -s * dx + c * dy, float(p[2])])


def _wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


def geometry(guess, belief, true_pose, world: dict, knob) -> dict:
  """The guess against the truth, each in its own robot's frame (the
  module docstring), mm and deg: the hinge line's place along the box (+
  further in) and up, measured square to the true axis; the pin along,
  across and up; the knob; the facing; and the pin's radius off the hinge.
  `knob` is where it truly hung, world frame."""
  g_h, g_p = in_frame(guess.hinge, belief), in_frame(guess.pin, belief)
  g_k = in_frame(guess.knob, belief)
  t_h, t_p = in_frame(world["hinge"], true_pose), in_frame(world["pin"], true_pose)
  t_k = in_frame(knob, true_pose)
  yaw_t = _wrap(world["yaw"] - true_pose[2])
  u = np.array([math.cos(yaw_t), math.sin(yaw_t), 0.0])
  v = np.array([-u[1], u[0], 0.0])
  z = np.array([0.0, 0.0, 1.0])
  dh = g_h - t_h
  dh -= (dh @ v) * v                                     # along the hinge's line is no error
  dp, dk = g_p - t_p, g_k - t_k
  r = t_p - t_h
  r_true = float(np.linalg.norm(r - (r @ v) * v))
  return {"hingeAlongMm": round(1000 * float(dh @ u), 2), "hingeUpMm": round(1000 * float(dh @ z), 2),
          "pinAlongMm": round(1000 * float(dp @ u), 2), "pinAcrossMm": round(1000 * float(dp @ v), 2),
          "pinUpMm": round(1000 * float(dp @ z), 2),
          "knobMm": [round(1000 * float(dk @ e), 2) for e in (u, v, z)],
          "facingDeg": round(math.degrees(_wrap(guess.yaw - belief[2] - yaw_t)), 3),
          "radiusMm": round(1000 * (guess.radius - r_true), 2)}


def in_map(world_pose, true_pose, belief) -> tuple[float, float, float]:
  """A world pose (x, y, yaw) in the robot's map: where it stands from the
  robot's true pose, laid off its belief."""
  x, y, yaw = world_pose
  rel = in_frame((x, y, 0.0), true_pose)
  bx, by, bth = belief
  c, s = math.cos(bth), math.sin(bth)
  return (bx + c * rel[0] - s * rel[1], by + s * rel[0] + c * rel[1],
          _wrap(yaw - true_pose[2] + bth))


def flown(record, lid_angle) -> Readings:
  """What the robot sensed as rollout readings: its torques and encoders,
  and the lid's TRUE angle row by row (ours: the robot never sensed it)."""
  return Readings(t=np.arange(1, record.n + 1) * record.dt, sensed=record.sensed,
                  joints={"hinge": np.asarray(lid_angle, dtype=float)})


def turned(pin, hinge, phases: dict) -> float | None:
  """How far the held handle turned on its pin while the catch held its
  lid shut, deg: from the first sweep up's start until the lid is past the
  catch's reach (`chest.CAUGHT_OFF`), or the sweep's end. `pin` and `hinge`
  are the two joints' angles row by row (rad); shut, the handle's angle on
  the lid is its angle in the world. None where no sweep began."""
  if "up0" not in phases:
    return None
  a, b = phases["up0"]
  pin, hinge = np.asarray(pin, dtype=float), np.asarray(hinge, dtype=float)
  out = np.flatnonzero(hinge[a:b] > ch.CAUGHT_OFF)
  end = a + (int(out[0]) if len(out) else b - a)
  return math.degrees(float(np.abs(pin[a:end] - pin[a]).max())) if end > a else 0.0


@dataclass(frozen=True)
class Replayed:
  """A record through one world: the `imagined.Gap`s phase by phase, and
  how far its held handle turned on its pin while the catch held (deg,
  `turned`)."""
  gaps: list
  turned: float | None


def replay(record, phases: dict, lid: ch.Lid, chest_map, pin: float, lid_angle) -> dict:
  """The record through the world's chest and through the best-expressible
  reference, each set at `chest_map` in the robot's map with its handle
  starting `pin` rad on its pin, each against what the robot sensed with
  the lid's true angle (`flown`), as `Replayed` -- what a perfect model
  leaves, and what the language's best leaves."""
  setting = im.Setting(chest_x=chest_map[0], chest_y=chest_map[1], chest_yaw=chest_map[2])
  sensed = flown(record, lid_angle)
  rows = {k: slice(*v) for k, v in phases.items()}
  out = {}
  for name, world in (("world", im.truth_world(lid, setting, pin)),
                      ("reference", im.reference_world(lid, setting, pin))):
    got = rollout(world, record)
    out[name] = Replayed(gaps=im.compare(sensed, got, rows),
                         turned=turned(got.joints["pin"], got.joints["hinge"], phases))
  return out
