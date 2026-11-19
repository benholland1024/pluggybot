"""Seeded random days of the stub pair at the world's three stations, for
#731 (a station the other robot is plugged into is taken, never a robot
lost): each robot charges -- waiting or passing with `free` --, plugs in
and stays a drawn while, stands on a seat with its plug out, or near one,
or walks about; beliefs drawn off (0-1 m, 0-15 deg) and put right by a
search; a fall plugged in (the breakaway: its plug out, its body down on the
seat a moment), stand-ups (a warp home, or by or onto a seat, its belief
right), restarts mid-trip (the trip closed where it stands, a new one begun:
mid-wait too) and, on some days, the run's hour (#654) -- on one physics
loop, with the invariants a served day needs read at every step (`Watch`):

  1. no charge waits past its bound: no wait holds on past the bound
     (`wait_bound`) from the first time its trip found the station taken;
  2. never a dock tried at a station another robot says it is plugged in at
     (its own word, `on_charger`) -- so never two plugged into one;
  3. never a lost search begun at a station another robot says it is
     plugged in at;
  4. never a robot that really lost its station left unsearched: a trip
     whose last dock saw no station while nothing truly stood by its seat
     (no body within `SEAT_HELD_M`, nobody plugged in) and that gave up at
     that station searched in it -- unless a restart closed it, the run's
     hour ended it, or the other robot is plugged in there as it gives up
     (a search then is the third's: the station is taken, whoever is lost);
  5. a `free` charge passes a taken station `MAX_STATION_SWITCHES` times
     at most;
  6. the loop never goes round at one sim instant (`ZERO_TIME_ROUNDS`).

Not collected (no `test_` prefix): `test_station_taken.py` flies a cheap
seeded slice of it in the default run; the seeds the branch reports were a
measurement, flown once by hand.
"""
from __future__ import annotations

import math
import random

import mujoco

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot import body as body_mod
from pluggybot.body import NOT_SEEN, STUB_WORLD, RackPose, StubBody
from pluggybot.economy import energy
from pluggybot.robot import FIRST, SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

WORLD = "home_humanoid"
FAST_WORLD = STUB_WORLD.replace('<mujoco model="stub">',
                                '<mujoco model="stub"><option timestep="0.05"/>')
SEATS = {"station": (3.575, -1.53), "station_facility": (20.575, -5.53),
         "station_mall": (17.575, -24.53)}
#: In the fuzz's world: a body this near a seat hides the station's tags,
#: and a dock sees its station only from this near the true standoff.
HIDES_M, SEES_M = 0.35, 0.45
#: The loop's own calls at one sim instant before it is called a spin.
ZERO_TIME_ROUNDS = 2000
#: What "nothing truly by the seat" is read against (`body.SEAT_HELD_M`;
#: the same width where a tree before #731 has none, to fly the fuzz there).
SEAT_HELD_M = getattr(body_mod, "SEAT_HELD_M", 0.54)


class Violation(AssertionError):
  """An invariant a served day needs, broken: what, when, and the seed."""


class _Error:
  """A belief laid off the truth by a fixed rigid move (`b = E(t)`), as a
  map laid through a wrong belief is: where it SAYS it is, `E(truth)`."""

  def __init__(self, dx=0.0, dy=0.0, deg=0.0, about=(0.0, 0.0)):
    self.dx, self.dy, self.turn, self.about = dx, dy, math.radians(deg), about

  def fwd(self, x, y):
    c, s = math.cos(self.turn), math.sin(self.turn)
    ax, ay = self.about
    return ax + c * (x - ax) - s * (y - ay) + self.dx, ay + s * (x - ax) + c * (y - ay) + self.dy

  def back(self, x, y):
    c, s = math.cos(-self.turn), math.sin(-self.turn)
    ax, ay = self.about
    x, y = x - self.dx - ax, y - self.dy - ay
    return ax + c * x - s * y, ay + s * x + c * y

  @property
  def off(self) -> bool:
    return bool(self.dx or self.dy or self.turn)


class FuzzBody(StubBody):
  """The stub with a truth of its own: its pose is its BELIEF (what it says
  and walks by), its body `E⁻¹` of it, what it sees placed through `E`; a
  dock that sees the station only from near the true standoff with no body
  on the seat, and a search that finds it (its belief put right) unless a
  body hides it, after a drawn while."""

  def __init__(self, *a, rng=None, watch=None, **kw):
    super().__init__(*a, **kw)
    self.rng, self.watch = rng, watch
    self.err = _Error()
    self.world: list[FuzzBody] = []
    self.still_since = 0.0

  @property
  def resting(self) -> bool:
    """Resting standing once it has moved nothing for `REST_S` (8.6 s), as
    the humanoid does plugged in: placed by its body, then (#455)."""
    return float(self.data.time) - self.still_since >= 8.6

  @resting.setter
  def resting(self, value) -> None:
    pass

  def go_to_routine(self, x, y, timeout=90.0, stop=None):
    self.still_since = float(self.data.time)
    return (yield from super().go_to_routine(x, y, timeout, stop))

  def truth(self):
    return self.err.back(self.x, self.y)

  def _bodies(self):
    from pluggybot.body import KeepClear
    return [KeepClear(*where()) for where in self.others]

  def footprint_centre(self):
    return self.truth()

  def as_seen(self, x, y):
    return self.err.fwd(x, y)

  def _hidden(self, seat) -> bool:
    return any(math.dist(b.truth(), seat) < HIDES_M or
               (b.on_charger and math.dist(b.charge_seat(), seat) < 0.05)
               for b in self.world if b is not self)

  def peer_on_the_goal(self, x, y):
    near = [math.dist((k.x, k.y), (x, y)) for k in self._bodies()]
    near = [d for d in near if d <= 0.45]
    return min(near) if near else None

  def dock_routine(self):
    seat = self.charge_seat()
    self.watch.docked(self, seat)
    yield from self._wait(self.rng.uniform(2.0, 20.0))
    standoff = self.rack.to_world(0.5, 0.0)
    if math.dist(self.truth(), standoff) > SEES_M or self._hidden(seat):
      self.watch.not_seen(self, seat)
      return NOT_SEEN
    if any(b.on_charger and math.dist(b.charge_seat(), seat) < 0.05
           for b in self.world if b is not self):
      return "plug not in its cup"
    self.x, self.y = self.err.fwd(*seat)        # ...its body on the seat
    self.still_since = float(self.data.time)
    self.on_charger = True
    return "docked"

  def lost_routine(self, fixture, stop=None):
    self.lost_at.append(fixture)
    self.watch.searched(self)
    t_end = float(self.data.time) + self.rng.uniform(30.0, 300.0)
    while float(self.data.time) < t_end:
      if stop is not None and stop():
        return False
      yield from self._wait(1.0)
    if self._hidden(self.charge_seat()):
      return False
    self.x, self.y = self.truth()               # ...the belief put right
    self.err = _Error()
    return True


class Watch:
  """Every invariant, read as the day goes."""

  def __init__(self, seed: int, lives: list) -> None:
    self.seed, self.lives = seed, lives
    self.trips: dict = {}                       # root -> the trip under way
    self.counts = {"trips": 0, "waits": 0, "searches": 0, "notSeen": 0, "hidden": 0,
                   "lost": 0, "switches": 0, "charged": 0, "restarts": 0,
                   "restartsMidWait": 0, "falls": 0, "standUps": 0, "hours": 0}
    self.waiting: dict = {}                     # root -> inside a bay wait now
    self._instant, self._rounds = None, 0

  def fail(self, life, what: str) -> None:
    raise Violation(f"seed {self.seed}, t={float(life.data.time):.1f}, "
                    f"{life.robot_name}: {what}")

  def _others(self, body):
    return [b for b in body.world if b is not body]

  def begin(self, life) -> None:
    self.counts["trips"] += 1
    self.trips[life.root] = {"lostAt": None, "searched": False, "logAt": len(life.log)}

  def end(self, life, ok: bool, how: str) -> None:
    trip = self.trips.pop(life.root, None)
    if trip is None:
      return
    if ok:
      self.counts["charged"] += 1
    lost = trip["lostAt"]
    if (lost is not None and not ok and not trip["searched"] and how == "returned"
        and not life._hour_up and math.dist(life.body.charge_seat(), lost) < 0.05
        and not any(b.on_charger and math.dist(b.charge_seat(), lost) < 0.05
                    for b in self._others(life.body))):
      self.fail(life, f"really lost and never searched: {life.charge_failure}")
    switched = sum(1 for ln in life.log[trip.get("logAt", 0):]
                   if "on to the nearest free charging station" in ln)
    self.counts["switches"] += switched
    if switched > lc.MAX_STATION_SWITCHES:
      self.fail(life, f"passed {switched} taken stations")

  def docked(self, body, seat) -> None:
    life = self._life(body)
    if any(b.on_charger and math.dist(b.charge_seat(), seat) < 0.05 for b in self._others(body)):
      self.fail(life, "a dock tried where the other says it is plugged in")
    if life.root in self.trips:
      self.trips[life.root]["lostAt"] = None    # ...this dock's answer decides

  def not_seen(self, body, seat) -> None:
    self.counts["notSeen"] += 1
    life = self._life(body)
    if body._hidden(seat):
      self.counts["hidden"] += 1
    elif all(math.dist(b.truth(), seat) >= SEAT_HELD_M for b in self._others(body)):
      self.counts["lost"] += 1
      if life.root in self.trips:
        self.trips[life.root]["lostAt"] = seat

  def searched(self, body) -> None:
    self.counts["searches"] += 1
    life = self._life(body)
    seat = body.charge_seat()
    if any(b.on_charger and math.dist(b.charge_seat(), seat) < 0.05 for b in self._others(body)):
      self.fail(life, "a lost search where the other says it is plugged in")
    if life.root in self.trips:
      self.trips[life.root]["searched"] = True

  def wait(self, life, since: float, t_in: float) -> None:
    self.counts["waits"] += 1
    bound, _ = life.wait_bound("charge", 0.0, 0.0)
    t = float(life.data.time)
    if t > max(t_in, since + bound) + lc.BAY_POLL_S + 1e-6:
      self.fail(life, f"waited until {t - since:.0f} s past the bound's {bound:.0f} s")

  def spin(self, life) -> None:
    t = float(life.data.time)
    self._rounds = self._rounds + 1 if t == self._instant else 0
    self._instant = t
    if self._rounds > ZERO_TIME_ROUNDS:
      self.fail(life, "the loop goes round at one instant")

  def _life(self, body):
    return next(life for life in self.lives if life.body is body)


def _until(routine, stop):
  """`routine`, closed where it stands once `stop()` says so (a restart)."""
  try:
    cmd = next(routine)
    while True:
      if stop():
        routine.close()
        return "restarted", None
      cmd = routine.send((yield cmd))
  except StopIteration as e:
    return "returned", e.value


def _robot(seed, model, data, handle, name, watch_box):
  rng = random.Random(f"{seed}:{handle.root}")
  cfg = lc.world_config(WORLD)
  body = FuzzBody(model, data, handle=handle, grid_bounds=cfg["grid_bounds"],
                  charge_w=energy.load(WORLD).charge_w * rng.choice((4.0, 20.0)),
                  rng=rng, watch=None)
  body.stations = {n: RackPose(*s, math.pi / 2) for n, s in SEATS.items()}
  life = stub_life(WORLD, body=body, handle=handle, robot_name=name, mortal=False)
  life.home_pose = tuple(float(v) for v in cfg["start" if handle is FIRST else "start2"])
  life.max_sim_time = 1e12
  return life, rng


def _wrap(life, watch) -> None:
  """The lifecycle's own seams, watched: the bay wait's bound, the loop's
  instant."""
  wait, holder = life._await_bay_routine, life._bay_holder

  def watched_wait(sx, sy, kind, since, seat=None, **kw):
    t_in = float(life.data.time)
    watch.waiting[life.root] = True
    try:
      out = yield from wait(sx, sy, kind, since, seat=seat, **kw)
    finally:
      watch.waiting[life.root] = False
    watch.wait(life, since, t_in)
    return out

  def watched_holder(*a, **kw):
    watch.spin(life)
    return holder(*a, **kw)
  life._await_bay_routine, life._bay_holder = watched_wait, watched_holder


def _station(rng) -> str:
  """Mostly the house's, where both robots charge from their starts."""
  return "station" if rng.random() < 0.7 else rng.choice(list(SEATS))


def _day(life, rng, watch, sim_s: float):
  """One robot's day: charges, stays, seats, walks; restarts mid-trip."""
  body, t_end = life.body, sim_s
  while float(life.data.time) < t_end:
    roll = rng.random()
    if roll < 0.45:
      if body.on_charger:
        body.on_charger = False
      free = rng.random() < 0.5
      watch.begin(life)
      cut = (float(life.data.time) + rng.uniform(5.0, 900.0)
             if rng.random() < 0.15 else math.inf)
      mid_wait = []
      how, ok = yield from _until(life.go_charge_routine(free=free),
                                  lambda cut=cut: float(life.data.time) >= cut
                                  and not mid_wait.append(watch.waiting.get(life.root)))
      watch.counts["restarts"] += how == "restarted"
      watch.counts["restartsMidWait"] += how == "restarted" and bool(mid_wait and mid_wait[-1])
      watch.end(life, bool(ok), how)
      if ok:
        yield from body._wait(rng.choice((rng.uniform(5.0, 120.0), rng.uniform(300.0, 2400.0))))
        if rng.random() < 0.2:
          # ...shoved over plugged in: the breakaway lets the plug go, its body
          # down on the seat a moment, then up and gone
          watch.counts["falls"] += 1
          body.on_charger = False
          body.attitude = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
          yield from body._wait(1.5)
          body.attitude = (1.0, 0.0, 0.0, 0.0)
        body.on_charger = False
        yield from body.go_to_routine(*life.home_pose[:2])
    elif roll < 0.6:
      seat = SEATS[_station(rng)]                   # on a seat, its plug out
      yield from body.go_to_routine(*body.err.fwd(*seat))
      yield from body._wait(rng.uniform(5.0, 300.0))
      yield from body.go_to_routine(*life.home_pose[:2])
    elif roll < 0.72:
      sx, sy = SEATS[_station(rng)]                 # near one, never on it
      a = rng.uniform(0.0, 2 * math.pi)
      r = rng.uniform(0.6, 2.0)
      yield from body.go_to_routine(*body.err.fwd(sx + r * math.cos(a), sy + r * math.sin(a)))
      yield from body._wait(rng.uniform(5.0, 300.0))
    elif roll < 0.9:
      body.err = (_Error() if rng.random() < 0.3 else
                  _Error(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-15, 15),
                         about=body.truth()))
      yield from body._wait(rng.uniform(1.0, 30.0))
    else:
      if rng.random() < 0.6:                       # a stand-up, its belief right:
        watch.counts["standUps"] += 1              # home, or by or onto a seat
        body.on_charger = False
        body.err = _Error()
        if rng.random() < 0.5:
          body.x, body.y = life.home_pose[0], life.home_pose[1]
        else:
          sx, sy = SEATS[_station(rng)]
          r, a = rng.choice((0.0, rng.uniform(0.2, 1.5))), rng.uniform(0.0, 2 * math.pi)
          body.x, body.y = sx + r * math.cos(a), sy + r * math.sin(a)
      yield from body._wait(rng.uniform(10.0, 200.0))


def fly(seed: int, sim_s: float = 7200.0) -> dict:
  """One seeded day of the pair; raises `Violation` at the first broken
  invariant. Returns what it flew, counted."""
  model = mujoco.MjModel.from_xml_string(FAST_WORLD)
  data = mujoco.MjData(model)
  lives, rngs = [], []
  for handle, name in ((FIRST, "Luca"), (SECOND, "Rowan")):
    life, rng = _robot(seed, model, data, handle, name, None)
    lives.append(life)
    rngs.append(rng)
  watch = Watch(seed, lives)
  hour = random.Random(f"{seed}:hour")
  if hour.random() < 0.3:
    # ...a day that ends on the run's hour, the next run carrying on (#654)
    watch.counts["hours"] = 1
    end = hour.uniform(600.0, sim_s)
    for life in lives:
      life.continuing, life.max_sim_time = True, end
  for life in lives:
    others = [o for o in lives if o is not life]
    life.peers = others
    life.body.others = [(lambda other=other, me=life.body: other.keep_clear(seen_by=me))
                        for other in others]
    life.body.world = [o.body for o in lives]
    life.body.watch = watch
    _wrap(life, watch)
    life.body.x, life.body.y = life.home_pose[0], life.home_pose[1]
  tick.run_many([(life.body.stepper, _day(life, rng, watch, sim_s))
                 for life, rng in zip(lives, rngs)])
  return dict(watch.counts, seed=seed, simS=round(float(data.time), 1))


if __name__ == "__main__":
  import json
  import sys
  first, n = int(sys.argv[1]), int(sys.argv[2])
  totals: dict = {}
  import contextlib
  import io
  for seed in range(first, first + n):
    with contextlib.redirect_stdout(io.StringIO()):
      try:
        out = fly(seed)
      except Violation as v:
        out = {"seed": seed, "violation": str(v)}
    for k, v in out.items():
      if isinstance(v, (int, float)) and k not in ("seed", "simS"):
        totals[k] = totals.get(k, 0) + v
    print(json.dumps(out), flush=True)
  print("totals", json.dumps(totals))
