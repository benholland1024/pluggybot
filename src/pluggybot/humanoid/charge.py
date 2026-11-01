"""Charging by its cable (issue #559, stage 2; SimNotes, "The humanoid's
charging station"): how a humanoid plugs itself in at a station
(`activity.station`) and unplugs. OUR routine, never one it learns (Ben's
call 1: charging is how it survives), off its own senses alone -- the
station off its two tags (`fit`), its port off its model of itself
(`port`), whether the connector mated off its pack's charge current
(`Body.charging`).

THE PLUG-IN (`plug_in_routine`), its right hand, standing at the seat:
  the hand on the parked plug's line short of its grip, jaws upright, the
  fingers across it (`GRIPS`, the first that gets round it);
  open, slid along the plug's axis over the parked plug to its grip --
  ⚠ NEVER IN FROM ITS SIDE: the arm's swing up to a grasp beside the
    station read 35-49 N of force it was not feeling, and stopped short;
  shut on it, carried to 5 cm off the port as the body then stands, and
  pushed to the body (`PUSH_N`): the funnel round the port takes it within
  15 mm, and a miss is fetched again and pushed a step to one side.
THE UNPLUG (`unplug_routine`): the open jaws slid along the seated plug from
  behind its back, shut, pulled straight off past the connector's 20 N
  (`UNPLUG_N`), carried to the cup's mouth -- the balancer takes it from
  there -- and let go.
LOST (`HumanoidBody.lost_routine`): no station where the belief puts it; it
  looks for its tags by its head's camera, round where it stands and then
  outward from where it believes the station is, and seen, walks nearer by
  what it saw (`rough`) until a look fits the station near enough to fix the
  belief by (`LOST_*`, `NEAR_*`).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity import station as st
from pluggybot.body import RackPose
from pluggybot.humanoid.arms import GRIP_MIN_N, frame
from pluggybot.perception.tags import STATION_TAG_IDS

#: The hand that plugs in: the port is right of the trunk's middle.
SIDE = "right"
#: The squeeze, N, of the Dex1-1's 5-120: the plug weighs 0.5 N and the
#: balancer pulls 2; and to unplug it, past the connector's 20 N (at 40 the
#: plug slipped out of the jaws).
GRIP_N, UNPLUG_GRIP_N = 40.0, 80.0
#: How the hand takes the parked plug, tried in turn until one gets round
#: it: the fingers swung this far forward of straight across, rad, about the
#: jaws' line -- the pads still pinch it from above and below -- and the
#: grip this far behind its face, m. MEASURED, 140 seats over what the seat
#: check accepts (`SEAT_*`), each started there: straight across at the
#: jack's back end the plug went in from 110 -- 3 cm and more behind the
#: seat the arm reached the plug only bowed, the wrist bent past 90 deg, and
#: the hand stopped short of it -- swung 20 deg from 140. ⚠ THE SWUNG GRIP
#: FURTHER BACK, 45 MM: at the jack's back end the gripper's housing met the
#: belly with the plug 21 mm off the port, past the catch's 15; at 50 the
#: pads' far edge came 4 mm off the station's boss, and walked in from
#: across the house, 12 first pushes in 15 were made with a hand that had
#: shut on the boss. ⚠ STRAIGHT ACROSS AFTER IT: of the seats the swung hand
#: missed, 3 cm in and 4-6 cm to the right, its way in along the fingers
#: pressed on the trunk.
GRIPS = ((math.radians(20.0), 0.045), (0.0, st.GRIP_BACK_M))
#: A squeeze holds the plug only where its jaws stop this near the jack's
#: width, m (the gripper's encoder reads the gap to about a millimetre):
#: on the boss, 40 mm across, a squeeze held and the hand pushed nothing.
#: Shut wider, the hand opens, draws back this far along the plug, m, and
#: shuts again, once: the station's fit is 1-2 cm off along the plug (its
#: tags' range off the colour camera), and the grip's pads end 12 mm off
#: the boss.
PLUG_GAP_TOL_M, BOSS_BACK_M = 0.004, 0.025
#: Nearer the station than its seat the tags are under the colour camera's
#: view (54 deg below level at the seat itself), so a look there that sees
#: no pair is taken again bowed this far, rad: fitted on the look from
#: before its last step instead, the step's odometry put the hand 2-3 cm off
#: along the plug, and both grips shut on the boss.
LOW_LOOK_RAD = 0.2
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
#: ...and where it lets the plug go, m short of its seat in the cup: the
#: cup's mouth (`st.CUP_MOUTH_M`) and the station's fit, 1-2 cm off along
#: the plug.
CUP_SHORT_M = 0.025
#: The seat's tolerance, m and rad: off it further, the robot steps again.
#: Its walker's least step is a stride, about 10 cm (`step_routine`), and
#: the carry takes a plug a few cm across. ⚠ NEARER THE STATION ONLY
#: `SEAT_IN_M`: 4.9 cm in from its seat, a robot's hand swung into the
#: station's box on its way to the plug.
SEAT_TOL_M, SEAT_IN_M, SEAT_TOL_RAD = 0.06, 0.03, math.radians(6.0)
#: ...in at most this many looks and steps (the last a look alone).
SEAT_STEPS = 6
#: A LOST robot looks for its station by its head's camera (#572), which
#: reads the station's 60 mm tags much further than the colour imager's
#: 1.07 m. MEASURED (`station_spike.py --reach`): both square-on to 4.5 m,
#: one from 30 deg off the station's face to 5 m -- the parked plug hides
#: the far tag past about 20 deg -- and from 50 deg off within 1.5 m. It
#: looks round where it stands, then from viewpoints outward from where it
#: believes the station is (`humanoid.places`' search), for this long at
#: most, s -- or until `stop`, the pack's (`HubLifecycle.go_charge`).
LOST_PATIENCE_S = 600.0
#: ...and seen, it walks this far out of the station's face, m, as `rough`
#: places it -- the colour imager reads the pair from there -- faces it and
#: looks, in this many tries at most. ⚠ NEARER BY WHAT IT SAW, NEVER A FIX
#: OFF A FAR LOOK: past 2 m the head's camera places a tag to 1-8 cm, but
#: the pair is 18 cm across, and its facing came out 1-5.5 deg off at 2-2.5
#: m and up to 29 deg at 3-4.5 m (66 in a probe; once both tags 0.26 m long).
NEAR_M, NEAR_TRIES = 1.0, 4
#: ...and seen again after tries that never fitted it, nearer by that look
#: again, this many times in all.
NEAR_ROUNDS = 2
#: A look of the head's camera fits the station to fix by once both its tags
#: are this near, m: within 1.5 m -- both read only within 15 deg of its
#: face -- the pair's facing came out within 1.9 deg, its place 1.3 cm.
NEAR_FIT_M = 1.6
#: ...and a far look's facing is taken only within this of the way it was
#: seen from, rad (a sign is read only from in front of it): else the robot
#: walks in along its line of sight.
VIEW_RAD = math.radians(60.0)


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


def look_low_routine(body):
  """`look`, and where it sees no pair, a look bowed `LOW_LOOK_RAD`,
  standing again after it: the station's frame, or None."""
  seen = look(body)
  if seen is not None:
    return seen
  yield from body.gaze_routine(LOW_LOOK_RAD)
  seen = look(body)
  yield from body.gaze_routine(0.0)
  return seen


def look_routine(body):
  """`look`, and where it spots one tag of the two, facing that one and
  looking again: the other is beside it, past the edge of the view. The
  station's frame, or None."""
  spots = {t: body.spot(t) for t in STATION_TAG_IDS}
  seen = fit({t: None if s is None else (s["x"], s["y"], s["z"]) for t, s in spots.items()})
  one = [s for s in spots.values() if s is not None]
  if seen is not None or len(one) != 1:
    return seen
  x, y, _ = body.pose
  yield from body.face_routine(math.atan2(one[0]["y"] - y, one[0]["x"] - x))
  return look(body)


def rough(seen: dict) -> RackPose:
  """The station as one look of the head's camera places it (`HumanoidPlaces.
  see_station`), to walk nearer by: the pair fitted where both tags are in
  it, else the one tag where it is, facing as its own rotation says --
  either facing only within `VIEW_RAD` of the way it was seen from, and
  otherwise that way."""
  tags = seen["tags"]
  ex, ey = seen["eye"]
  fitted = fit({t: p for t, (p, _) in tags.items()}) if len(tags) == 2 else None
  if fitted is not None:
    x, y, out = fitted.x, fitted.y, fitted.yaw + math.pi
  else:
    t, (p, out) = next(iter(tags.items()))
    x, y = p[0], p[1]
  view = math.atan2(ey - y, ex - x)
  if abs(math.remainder(out - view, 2 * math.pi)) > VIEW_RAD:
    out = view
  if fitted is None:
    # ...the one tag where its drawing puts it either side of the cup
    ly = st.tag_layout()[t][1]
    x, y = x - math.sin(out) * ly, y + math.cos(out) * ly
  return RackPose(float(x), float(y), math.remainder(out + math.pi, 2 * math.pi))


def near_fit(seen: dict | None) -> RackPose | None:
  """The station fitted off one look of the head's camera where both its
  tags were in it within `NEAR_FIT_M` of its eye -- near enough to fix the
  belief by -- else None."""
  if seen is None or len(seen["tags"]) != 2:
    return None
  ex, ey = seen["eye"]
  if any(math.hypot(p[0] - ex, p[1] - ey) > NEAR_FIT_M for p, _ in seen["tags"].values()):
    return None
  return fit({t: p for t, (p, _) in seen["tags"].items()})


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


def _grip(out: np.ndarray, swing: float = 0.0) -> np.ndarray:
  """The hand on a plug lying along `out`, its nose the other way: fingers
  to the robot's left of it, swung `swing` toward `out`, jaws upright
  (`arms.frame`)."""
  left = np.cross((0.0, 0.0, 1.0), out)
  left = left / np.linalg.norm(left)
  return frame(math.cos(swing) * left + math.sin(swing) * out, (0.0, 0.0, 1.0))


def parked_grip(station: RackPose, back: float = st.GRIP_BACK_M) -> np.ndarray:
  """Where the parked plug is gripped `back` behind its face, in the world."""
  x, y = st.to_world((station.x, station.y, station.yaw),
                     -st.BOSS_M - 0.003 - st.PLUG_LEN_M + back, 0.0)
  return np.array([x, y, st.CUP_Z])


def _squeeze_routine(body, force: float):
  """Shut on a plug a catch holds: at the least the gripper squeezes, until
  both pads are on it, and then at `force`. ⚠ SHUT AT ONCE, the first pad to
  meet a seated plug pushed it across the port with the whole squeeze, past
  the connector's 20 N, and it broke away: from 110 seats, 46 unplugs could
  not grip it, 45 of them having knocked it off its catch."""
  yield from body.grip_routine(SIDE, True, force=GRIP_MIN_N)
  return (yield from body.grip_routine(SIDE, True, force=force))


def _take_routine(body, grip: np.ndarray, into: np.ndarray):
  """Shut on the plug, the hand at `grip`: None, or why not -- shut wider
  than its jack (`PLUG_GAP_TOL_M`), on the station's boss, opened, drawn
  `BOSS_BACK_M` back along the plug and shut again, once."""
  for k in range(2):
    if not (yield from _squeeze_routine(body, GRIP_N)):
      return "it could not grip the plug"
    gap = body.grip_gap(SIDE)
    if abs(gap - 2.0 * st.PLUG_R_M) <= PLUG_GAP_TOL_M:
      return None
    if k or gap < 2.0 * st.PLUG_R_M:
      return f"its jaws shut {gap * 1000:.0f} mm apart, on something not the plug"
    yield from body.grip_routine(SIDE, False)
    r = yield from body.move_hand_routine(SIDE, grip - into * BOSS_BACK_M, speed=0.04)
    if r["why"] != "arrived":
      return f"its hand did not draw back off the station's boss ({r['why']})"


def _fetch_routine(body, station: RackPose, swing: float, back: float):
  """The open jaws onto the parked plug along its axis (`GRIPS`' one) and
  shut on it: None, or why not -- the hand let go and home again."""
  into = np.array([math.cos(station.yaw), math.sin(station.yaw), 0.0])    # into the wall
  grip = parked_grip(station, back)
  yield from body.grip_routine(SIDE, False)
  # ...on the plug's line LINE_BACK_M short of its grip, wherever the
  # robot stands: 5 cm off the port, a robot standing 5 cm in from its
  # seat swung its jaws into the parked plug's nose
  why = None
  if not (yield from body.reach_routine(SIDE, grip - into * LINE_BACK_M, _grip(into, swing))):
    why = f"its hand did not get to the plug's line ({body.hand_why(SIDE)})"
  else:
    r = yield from body.move_hand_routine(SIDE, grip, speed=0.06)
    why = (f"its hand did not get round the plug ({r['why']})" if r["why"] != "arrived"
           else (yield from _take_routine(body, grip, into)))
  if why is not None:
    yield from body.grip_routine(SIDE, False)
    yield from body.release_routine(SIDE)
  return why


def plug_in_routine(body, station: RackPose, trace: list):
  """The plug-in (the module docstring): "plugged in", or why not."""
  for k, aside in enumerate((0.0, *SEARCH_M)[:TRIES]):
    if k:
      yield from body.hold_routine(PARK_WAIT_S)       # the balancer takes it home
    for swing, back in GRIPS:
      why = yield from _fetch_routine(body, station, swing, back)
      if why is None:
        break
    else:
      return why
    # ...the port where the body now stands it: a posture taken to reach
    # the plug moves it
    p, out = port(body)
    seat = p + out * back                               # the hand, the plug on the port
    left = np.cross((0.0, 0.0, 1.0), out)
    left = left / np.linalg.norm(left)
    yield from body.move_hand_routine(SIDE, seat + out * OUT_M, speed=0.06,
                                      turns=[_grip(out, swing)])
    yield from body.move_hand_routine(SIDE, seat + out * PUSH_FROM_M + left * aside, speed=0.04)
    pushed = yield from body.pull_routine(SIDE, -out, PUSH_N, PUSH_M, speed=0.02, give=False)
    # ...let go in the funnel: the magnet and the funnel draw it home, as
    # they never could against the hand
    yield from body.grip_routine(SIDE, False)
    yield from body.hold_routine(0.3)
    trace.append({"try": k, "aside": aside, "swing": round(math.degrees(swing)),
                  "push": pushed["why"], "along": pushed.get("along"),
                  "mated": body.charging()})
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
    yield from body.release_routine(SIDE)
    return f"its hand did not get behind the plug ({body.hand_why(SIDE)})"
  yield from body.move_hand_routine(SIDE, seat, speed=0.05)
  if not (yield from _squeeze_routine(body, UNPLUG_GRIP_N)):
    yield from body.grip_routine(SIDE, False)
    yield from body.release_routine(SIDE)
    return "it could not grip the plug"
  pulled = yield from body.pull_routine(SIDE, out, UNPLUG_N, 0.05, speed=0.03, give=False)
  trace.append({"unplug": pulled["why"], "along": pulled.get("along"),
                "mated": body.charging()})
  if body.charging():
    yield from body.grip_routine(SIDE, False)
    yield from body.release_routine(SIDE)
    return f"it could not pull the plug off ({pulled['why']})"
  # ...carried to its cup's mouth short of the seat, let go, and the jaws
  # slid back off it along its line: carried to the seat as the station was
  # fitted, the plug was driven into the boss at up to 59 N, and the robot
  # fell, 5 unplugs in 140 -- the balancer takes it home from the mouth
  into = np.array([math.cos(station.yaw), math.sin(station.yaw), 0.0])
  yield from body.move_hand_routine(SIDE, parked_grip(station) - into * CUP_SHORT_M,
                                    speed=0.06, turns=[_grip(into)])
  yield from body.hold_routine(0.4)
  yield from body.grip_routine(SIDE, False)
  yield from body.move_hand_routine(SIDE, parked_grip(station) - into * LINE_BACK_M,
                                    speed=0.06)
  yield from body.release_routine(SIDE)
  return "unplugged"
