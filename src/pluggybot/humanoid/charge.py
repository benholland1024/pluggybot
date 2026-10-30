"""Charging by its cable (issue #559, stage 2; SimNotes, "The humanoid's
charging station"): how a humanoid plugs itself in at a station
(`activity.station`) and unplugs. OUR routine, never one it learns (Ben's
call 1: charging is how it survives), off its own senses alone -- the
station off its two tags (`fit`), its port off its model of itself
(`port`), whether the connector mated off its pack's charge current
(`Body.charging`).

THE PLUG-IN (`plug_in_routine`), its right hand, standing at the seat:
  the hand 5 cm off the port, fingers to its left and jaws upright;
  open, slid along the plug's axis over the parked plug to its grip --
  ⚠ NEVER IN FROM ITS SIDE: the arm's swing up to a grasp beside the
    station read 35-49 N of force it was not feeling, and stopped short;
  shut on it, drawn back along its line to 1 cm off the port, and pushed
  to the body (`PUSH_N`): the funnel round the port takes it within 12 mm,
  and a miss is drawn back and pushed again a step to one side.
THE UNPLUG (`unplug_routine`): the open jaws slid along the seated plug from
  behind its back, shut, pulled straight off past the connector's 20 N
  (`UNPLUG_N`), carried to the cup's mouth -- the balancer takes it from
  there -- and let go.
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity import station as st
from pluggybot.body import RackPose
from pluggybot.humanoid.arms import frame
from pluggybot.rack.tags import STATION_TAG_IDS

#: The hand that plugs in: the port is right of the trunk's middle.
SIDE = "right"
#: The squeeze, N, of the Dex1-1's 5-120: the plug weighs 0.5 N and the
#: balancer pulls 2; and to unplug it, past the connector's 20 N (at 40 the
#: plug slipped out of the jaws).
GRIP_N, UNPLUG_GRIP_N = 40.0, 80.0
#: Where the hand takes the plug's line, m short of its grip on it, and
#: where it carries it to, m off the port.
LINE_BACK_M, OUT_M = 0.10, 0.05
#: The push to the body, N and m: from 1 cm off the port, until the hand
#: feels this or has gone this far.
PUSH_N, PUSH_M, PUSH_FROM_M = 10.0, 0.03, 0.010
#: ...and a miss -- the balancer takes the plug home, and the hand fetches
#: it again -- pushed this far to one side, m, along the fingers: the one
#: way the jaws leave the plug free (they centre it across the other).
SEARCH_M = (-0.012, 0.012, -0.024)
TRIES = 4
#: The balancer's walk home with the plug, s (0.25 m at 0.25 m/s, its swing
#: and its cup's draw-in): at 2.5 the hand closed on the plug's place before
#: it was there, 2 tries in 15.
PARK_WAIT_S = 4.0
#: The pull that unplugs it, N, past the connector's hold (`st.BREAK_N`):
#: standing, the robot pulls about 35 N.
UNPLUG_N = 30.0
#: The seat's tolerance, m and rad: off it further, the robot steps again.
#: Its walker's least step is a stride, about 10 cm (`step_routine`), and
#: the carry takes a plug a few cm across. ⚠ NEARER THE STATION ONLY
#: `SEAT_IN_M`: 4.9 cm in from its seat, a robot's hand swung into the
#: station's box on its way to the plug.
SEAT_TOL_M, SEAT_IN_M, SEAT_TOL_RAD = 0.06, 0.03, math.radians(6.0)
#: ...in at most this many looks and steps (the last a look alone).
SEAT_STEPS = 6


def fit(spots: dict) -> RackPose | None:
  """The station's frame (`activity.station`'s: +x into the wall) off its
  two tags' places in the believed world, or None without both: its facing
  is their baseline's (#88)."""
  a, b = (spots.get(i) for i in STATION_TAG_IDS)
  if a is None or b is None:
    return None
  a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
  y = a - b
  if np.linalg.norm(y) < 0.5 * st.TAG_Y:
    return None
  y = y / np.linalg.norm(y)
  mid = (a + b) / 2
  return RackPose(float(mid[0]), float(mid[1]), math.atan2(-y[0], y[1]))


def look(body) -> RackPose | None:
  """Both tags spotted from where it stands, and the station fitted."""
  spots = {}
  for tag in STATION_TAG_IDS:
    s = body.spot(tag)
    spots[tag] = None if s is None else (s["x"], s["y"], s["z"])
  return fit(spots)


def port(body) -> tuple[np.ndarray, np.ndarray]:
  """Its port's place and outward normal in the believed world, off its
  model of itself posed by its senses (the encoders, the IMU)."""
  body.reach._pose()
  sm = body.reach.self_model
  sid = sm.m.site("charge_port").id
  p = body.to_world(sm.d.site_xpos[sid].copy())
  out = sm.d.site_xmat[sid].reshape(3, 3)[:, 2]
  c, s = math.cos(body.stance[2]), math.sin(body.stance[2])
  return (np.asarray(p, float),
          np.array([c * out[0] - s * out[1], s * out[0] + c * out[1], out[2]]))


def _grip(out: np.ndarray) -> np.ndarray:
  """The hand on a plug lying along `out`, its nose the other way: fingers
  to the robot's left of it, jaws upright (`arms.frame`)."""
  left = np.cross((0.0, 0.0, 1.0), out)
  return frame(left / np.linalg.norm(left), (0.0, 0.0, 1.0))


def parked_grip(station: RackPose) -> np.ndarray:
  """Where the parked plug is gripped, in the world."""
  x, y = st.to_world((station.x, station.y, station.yaw),
                     -st.BOSS_M - 0.003 - st.PLUG_LEN_M + st.GRIP_BACK_M, 0.0)
  return np.array([x, y, st.CUP_Z])


def plug_in_routine(body, station: RackPose, trace: list):
  """The plug-in (the module docstring): "plugged in", or why not."""
  into = np.array([math.cos(station.yaw), math.sin(station.yaw), 0.0])    # into the wall
  p, out = port(body)
  seat = p + out * st.GRIP_BACK_M                     # the hand, the plug on the port
  left = np.cross((0.0, 0.0, 1.0), out)
  left = left / np.linalg.norm(left)
  for k, aside in enumerate((0.0, *SEARCH_M)[:TRIES]):
    if k:
      yield from body.hold_routine(PARK_WAIT_S)       # the balancer takes it home
    yield from body.grip_routine(SIDE, False)
    # ...on the plug's line LINE_BACK_M short of its grip, wherever the
    # robot stands: 5 cm off the port, a robot standing 5 cm in from its
    # seat swung its jaws into the parked plug's nose
    line = parked_grip(station) - into * LINE_BACK_M
    if not (yield from body.reach_routine(SIDE, line, _grip(into))):
      return f"its hand did not get to the plug's line ({body.hand_why(SIDE)})"
    r = yield from body.move_hand_routine(SIDE, parked_grip(station), speed=0.06)
    if r["why"] != "arrived" or not (yield from body.grip_routine(SIDE, True, force=GRIP_N)):
      yield from body.grip_routine(SIDE, False)
      yield from body.release_routine(SIDE)
      return "it could not grip the plug"
    r = yield from body.move_hand_routine(SIDE, seat + out * OUT_M, speed=0.06,
                                          turns=[_grip(out)])
    yield from body.move_hand_routine(SIDE, seat + out * PUSH_FROM_M + left * aside, speed=0.04)
    pushed = yield from body.pull_routine(SIDE, -out, PUSH_N, PUSH_M, speed=0.02, give=False)
    # ...let go in the funnel: the magnet and the funnel draw it home, as
    # they never could against the hand
    yield from body.grip_routine(SIDE, False)
    yield from body.hold_routine(0.3)
    trace.append({"try": k, "aside": aside, "push": pushed["why"],
                  "along": pushed.get("along"), "mated": body.charging()})
    yield from body.release_routine(SIDE)
    if body.charging():
      return "plugged in"
  return "it pushed the plug in and the port never took it"


def unplug_routine(body, station: RackPose, trace: list):
  """The unplug (the module docstring): "unplugged", or why not."""
  p, out = port(body)
  seat = p + out * st.GRIP_BACK_M
  turn = _grip(out)
  yield from body.grip_routine(SIDE, False)
  # ...3 cm behind the plug's back: further out, from a robot standing in
  # from its seat the jaws swung into the station's boss
  behind = seat + out * (st.PLUG_LEN_M - st.GRIP_BACK_M + 0.03)
  if not (yield from body.reach_routine(SIDE, behind, turn)):
    return f"its hand did not get behind the plug ({body.hand_why(SIDE)})"
  yield from body.move_hand_routine(SIDE, seat, speed=0.05)
  if not (yield from body.grip_routine(SIDE, True, force=UNPLUG_GRIP_N)):
    yield from body.release_routine(SIDE)
    return "it could not grip the plug"
  pulled = yield from body.pull_routine(SIDE, out, UNPLUG_N, 0.05, speed=0.03, give=False)
  trace.append({"unplug": pulled["why"], "along": pulled.get("along"),
                "mated": body.charging()})
  if body.charging():
    yield from body.grip_routine(SIDE, False)
    yield from body.release_routine(SIDE)
    return f"it could not pull the plug off ({pulled['why']})"
  yield from body.move_hand_routine(SIDE, parked_grip(station), speed=0.06,
                                        turns=[_grip(np.array([math.cos(station.yaw),
                                                               math.sin(station.yaw), 0.0]))])
  yield from body.hold_routine(0.4)
  yield from body.grip_routine(SIDE, False)
  yield from body.release_routine(SIDE)
  return "unplugged"
