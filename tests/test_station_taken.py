"""A station the other robot is plugged into is TAKEN, never a robot lost
(issue #731; SimNotes, "A station the other robot is plugged into").

From 14:52 to 16:39 UTC on 11-18 Rowan was plugged in at the research
center lobby's station. Luca, its belief 0.57-0.81 m and 12-15 deg off,
charged five times at that station: the taken test (`_bay_holder`) placed
Rowan's body through Luca's wrong belief, 0.6 m off the seat, and read the
station free; Luca walked in, its dock saw no station (Rowan's body hid both
tags), and it searched as a robot lost, 422-468 s each, and failed "no
charge contact (station not seen)" -- 78 minutes, 29 % down to 7 %.

Two rules, Ben's options 1 with 2: a robot that SAYS it is plugged in at a
station holds its seat, wherever the asker's belief places its body (its
own port's word, `KeepClear.plugged`); and a station not in sight from its
standoff with another robot by its seat -- where it says it stands, or where
this robot's sensors see it -- is taken, not lost: the charge waits for it as
#346's wait does, or with `free` goes on to another station, and searches
only once nobody is by it, or once the wait's bound is spent with nobody
plugged in there. Each is pinned here on TWO lifecycles of stubs wired as
`pair.build_pair` wires a pair's bodies (`others`, `peers`): the asker reads
the other robot's own report (`keep_clear`) through its own body, never the
other lifecycle's state. Shown failing on loop-staging 0fb85dc2, where the
first two rules are not; the guards (a robot really lost still searches)
pass there and fail on the rule widened (the report's `pins` list).
"""
import math

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.body import NOT_SEEN, STUB_WORLD, KeepClear, RackPose, StubBody
from pluggybot.economy import energy
from pluggybot.robot import FIRST, SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

WORLD = "home_humanoid"
#: The stub's world at a coarser step (the fuzz's, #730): what a charge's
#: wait decides is in seconds, and its bound is 23 minutes.
FAST_WORLD = STUB_WORLD.replace('<mujoco model="stub">',
                                '<mujoco model="stub"><option timestep="0.02"/>')

#: The world's three stations as the stub frames a rack: its origin the
#: humanoid's seat (`activity.station.SEAT_X`, `SEAT_Y`), +x out into the
#: room -- all three face +y.
SEATS = {"station": (3.575, -1.53), "station_facility": (20.575, -5.53),
         "station_mall": (17.575, -24.53)}
LOBBY = "station_facility"
#: #731's five lost poses (the drift rows of 11-18): Luca's truth and its
#: belief, (x, y, deg) -- 0.57-0.81 m and 12-15 deg apart -- and Rowan's
#: believed pose at each. Rowan's body, plugged in: its soles' middle.
LOST = [((20.568, -4.546, -118.6), (20.629, -5.111, -106.2), (20.681, -5.573)),
        ((20.575, -4.428, -120.4), (20.568, -5.074, -106.3), (20.702, -5.574)),
        ((20.596, -4.457, -98.7), (20.541, -5.063, -84.2), (20.754, -5.555)),
        ((20.617, -4.335, -99.3), (20.529, -5.066, -84.2), (20.796, -5.546)),
        ((20.692, -4.285, -98.8), (20.545, -5.077, -84.1), (20.844, -5.532))]
ROWAN_BODY = (20.55, -5.722)
#: A body this near a seat hides the station's tags from its standoff, in
#: this test's world (the flown probe's is SimNotes').
HIDES_M = 0.35


def _rack(name: str) -> RackPose:
  return RackPose(*SEATS[name], math.pi / 2)


def _belief(true_pose, believed_pose):
  """Where a robot whose belief is `believed_pose` while it stands at
  `true_pose` places a TRUE world point (`Body.as_seen`): seen from where
  it stands, laid down through where it believes it stands."""
  (tx, ty, td), (bx, by, bd) = true_pose, believed_pose
  turn = math.radians(bd - td)
  c, s = math.cos(turn), math.sin(turn)

  def seen(x, y):
    dx, dy = x - tx, y - ty
    return bx + c * dx - s * dy, by + s * dx + c * dy
  return seen


class _SeatBody(StubBody):
  """The stub as the pair's bodies are: the other robots placed as `others`
  answers them (`HubLifecycle.keep_clear`), a world truth of its own for its
  body (`body_xy`) and its sensors (`seen_as`), and a dock that sees no
  station while another robot's body stands on its seat."""

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.body_xy = None          # where its body truly is; None: its pose
    self.seen_as = None          # its belief's error: a true point as it places it
    self.world = []              # every robot's body, for the dock's truth
    self.lost = False            # ...and a dock that sees nothing anyway
    self.docks: list = []        # (t, station seat) of every dock tried

  def footprint_centre(self):
    return self.body_xy or (self.x, self.y)

  def as_seen(self, x, y):
    return (x, y) if self.seen_as is None else self.seen_as(x, y)

  def peer_on_the_goal(self, x, y):
    """`Navigator.peer_on_the_goal`: where `others` places each robot."""
    near = [math.dist((b.x, b.y), (x, y)) for b in (KeepClear(*w()) for w in self.others)]
    near += [math.dist(p, (x, y)) for p in self.peers_on]
    near = [d for d in near if d <= 0.45]
    return min(near) if near else None

  def dock_routine(self):
    seat = self.charge_seat()
    self.docks.append((float(self.data.time), seat))
    hid = any(math.dist(b.footprint_centre(), seat) < HIDES_M or
              (b.on_charger and math.dist(b.charge_seat(), seat) < 0.05)
              for b in self.world if b is not self)
    if self.lost or hid:
      return NOT_SEEN
    self.on_charger = True
    return "docked"
    yield


def _pair(luca_pose=(20.5, -2.0, -90.0), rowan_at=LOBBY, rowan_kw=None):
  """Luca (the first robot) in the lobby, and Rowan plugged in at
  `rowan_at`'s station (None: not plugged in), resting there, its body on
  the seat: two lifecycles on one stub world, each body told of the other as
  `pair.build_pair` tells it."""
  model = mujoco.MjModel.from_xml_string(FAST_WORLD)
  data = mujoco.MjData(model)
  cfg = lc.world_config(WORLD)
  lives = []
  for handle, name in ((FIRST, "Luca"), (SECOND, "Rowan")):
    body = _SeatBody(model, data, handle=handle, grid_bounds=cfg["grid_bounds"],
                     charge_w=energy.load(WORLD).charge_w)
    body.stations = {n: _rack(n) for n in SEATS}
    life = stub_life(WORLD, body=body, handle=handle, robot_name=name,
                     **((rowan_kw or {}) if handle is SECOND else {}))
    life.home_pose = tuple(float(v) for v in cfg["start" if handle is FIRST else "start2"])
    lives.append(life)
  luca, rowan = lives
  for life in lives:
    others = [o for o in lives if o is not life]
    life.peers = others
    life.body.others = [(lambda other=other, me=life.body: other.keep_clear(seen_by=me))
                        for other in others]
    life.body.world = [o.body for o in lives]
  luca.body.x, luca.body.y, luca.body.theta = luca_pose[0], luca_pose[1], math.radians(luca_pose[2])
  rb = rowan.body
  rb.rack = rb.rack_prior = _rack(rowan_at or LOBBY)
  rb.x, rb.y = ROWAN_BODY
  rb.body_xy = ROWAN_BODY
  rb.on_charger = rowan_at is not None
  rb.resting = True
  return luca, rowan


def _live(luca, rowan, k: int) -> None:
  """#731's k-th lost pose: Luca's belief as it was off, Rowan's report as
  Rowan believed it, Rowan's body on the seat."""
  true, believed, rowan_said = LOST[k]
  luca.body.seen_as = _belief(true, believed)
  rowan.body.x, rowan.body.y = rowan_said


def _leaves_at(luca, rowan, t: float) -> None:
  """Rowan unplugs and walks home at sim time `t`, as Luca's clock runs."""
  def hook():
    if float(luca.data.time) >= t and rowan.body.on_charger:
      rowan.body.on_charger, rowan.body.resting = False, False
      rowan.body.x, rowan.body.y = rowan.home_pose[0], rowan.home_pose[1]
      rowan.body.body_xy = None
  luca.body.step_hooks.append(hook)


def _said(life, text: str) -> bool:
  return any(text in line for line in life.log)


# ---- rule 2: the plugged-in robot read off its own report --------------------


@pytest.mark.parametrize("k", range(len(LOST)))
def test_a_station_the_other_says_it_is_plugged_into_is_held_wherever_the_belief_puts_it(k):
  """T1's probe on the stub pair: through Luca's live belief Rowan's body
  lies off the seat, and the taken test read the lobby's station free from
  3 of 5 poses (here all five, the placement exact). Rowan's own word that
  its plug is in that station holds the seat, wherever it is placed --
  and the station is no free one to pass to."""
  luca, rowan = _pair()
  _live(luca, rowan, k)
  luca.body.choose_station()
  sx, sy, _ = luca.body.charge_standoff()
  seat = luca.body.charge_seat()
  assert luca.body.peer_on_the_goal(*seat) is None, "the premise: placed off the seat"
  held = luca._bay_holder(sx, sy, seat)
  assert held is not None and held[3] == "plugged", held
  assert not luca._bay_free((sx, sy), seat)
  assert luca.peer_at(*held[:2])[0] == "Rowan", "the holder is named"
  # ...and the plug, never the placement: unplugged, the same body is not on
  # the seat for the base's test, and plugged in at another station, it is not
  # at this one
  for plugged in (None, "station_mall"):
    luca, rowan = _pair(rowan_at=plugged)
    _live(luca, rowan, k)
    luca.body.choose_station()
    sx, sy, _ = luca.body.charge_standoff()
    assert luca._bay_holder(sx, sy, luca.body.charge_seat()) is None, plugged


# ---- rule 1 with 2: Luca's five charges, flown on the stub pair ---------------


@pytest.mark.parametrize("k", range(len(LOST)))
def test_lucas_charge_to_the_station_rowan_is_plugged_into_waits_and_is_never_lost(k):
  """Each of #731's five charges as a stub scene: no "lost", no search, the
  station waited for (`charge`, the default) -- Rowan unplugs 5 minutes in,
  and Luca walks in and plugs in -- never a dock tried while Rowan's plug
  was in."""
  luca, rowan = _pair()
  _live(luca, rowan, k)
  t0 = float(luca.data.time)
  _leaves_at(luca, rowan, t0 + 300.0)
  assert luca.body.run(luca.go_charge_routine()) is True, luca.log[-3:]
  assert luca.body.lost_at == [], "searched as a robot lost"
  assert _said(luca, "WAIT: Rowan is plugged in at the charging station"), luca.log
  assert all(t >= t0 + 300.0 for t, _ in luca.body.docks), luca.body.docks
  assert rowan.body.way_asked == [] and luca.body.asked_off == [], "a plugged-in robot asked off"
  assert luca.body.on_charger and not rowan.body.on_charger


@pytest.mark.parametrize("k", range(len(LOST)))
def test_lucas_free_charge_goes_on_to_a_free_station(k):
  """...and decided `charge (free)`, as all five were: on to the nearest
  station nobody holds, never into the lobby's."""
  luca, rowan = _pair()
  _live(luca, rowan, k)
  assert luca.body.run(luca.go_charge_routine(free=True)) is True, luca.log[-3:]
  assert luca.body.lost_at == []
  assert _said(luca, "GO_CHARGE on to the nearest free charging station -- Rowan is at this one")
  assert [seat for _, seat in luca.body.docks] == [luca.body.charge_seat()]
  assert math.dist(luca.body.charge_seat(), SEATS[LOBBY]) > 1.0


# ---- rule 1: not plugged in, by the seat -------------------------------------


@pytest.mark.parametrize("standing", [True, False], ids=["seen", "resting"])
def test_a_station_out_of_sight_with_the_other_by_its_seat_is_taken_not_lost(standing):
  """Rowan on the seat with its plug OUT -- unplugging, or about to plug in
  -- its body hiding the tags, and Luca's belief as live: Luca's LIDAR
  places it, or a body at rest is placed (#455, #675), through that belief,
  0.9 m off the seat, where the base's test reads the station free; where
  Rowan SAYS it stands is 0.14 m from the seat. Not lost: waited for -- "by"
  the station, one at rest asked off its way (#455) -- never searched, and
  plugged in once Rowan walks off (an unplug takes 26 s, a plug-in from the
  standoff 27)."""
  luca, rowan = _pair(rowan_at=None)
  _live(luca, rowan, 1)
  rowan.body.resting = not standing
  if standing:
    luca.body.sightings[rowan.root] = 0.0           # seen, placed through its belief
    rowan.body.moved_at = -1.0
  seat = SEATS[LOBBY]
  luca.body.choose_station()
  assert luca.body.peer_on_the_goal(*seat) is None, "the premise: the base reads it free"
  t0 = float(luca.data.time)
  luca.body.step_hooks.append(lambda: _walks_off(luca, rowan, t0 + 60.0))
  assert luca.body.run(luca.go_charge_routine()) is True, luca.log[-4:]
  assert luca.body.lost_at == [], "searched as a robot lost"
  assert _said(luca, "WAIT: Rowan is standing at rest by the charging station" if not standing
               else "WAIT: Rowan is standing by the charging station"), luca.log
  assert luca.body.on_charger and luca.body.asked_off, "one by the seat asked off its way"
  assert len(luca.body.docks) == 2 and luca.body.docks[-1][0] >= t0 + 60.0


def _walks_off(luca, rowan, t: float) -> None:
  """Rowan, its plug out, walks home at sim time `t`."""
  if float(luca.data.time) >= t and rowan.body.body_xy is not None:
    rowan.body.x, rowan.body.y = rowan.home_pose[0], rowan.home_pose[1]
    rowan.body.body_xy = None
    rowan.body.resting = False
    luca.body.sightings.pop(rowan.root, None)


@pytest.mark.parametrize("how", ["down", "getting up"])
def test_a_robot_on_the_floor_says_nothing_of_where_it_is(how):
  """A robot down cannot report itself (`keep_clear`, #365: its reckoning
  runs on through a fall, measured 2.2 m off its body in 10 s), nor while
  its get-up has its legs, tilted or not, so where it SAYS it is holds no
  seat: knocked over 1.5 m off the lobby's seat, its belief still on it, its
  body is what is read -- and standing again, its word."""
  luca, rowan = _pair(rowan_at=None)
  rowan.body.x, rowan.body.y = SEATS[LOBBY]                    # what it believes
  rowan.body.body_xy = (SEATS[LOBBY][0] + 1.5, SEATS[LOBBY][1])  # where it lies
  if how == "down":
    rowan.body.attitude = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    assert rowan.down()
  else:
    rowan.body.posture = "getting_up"                          # ...kneeling, under 60 deg
    assert not rowan.down()
  assert luca.body.peer_by_seat(*SEATS[LOBBY]) is None
  rowan.body.attitude, rowan.body.posture = (1.0, 0.0, 0.0, 0.0), "standing"
  assert luca.body.peer_by_seat(*SEATS[LOBBY]) == pytest.approx(0.0)


def test_a_plug_in_waited_for_leaves_the_while_by_the_seat_for_its_unplug():
  """Rowan plugs in as Luca's dock looks, and Luca waits for its plug; Rowan
  unplugs and stands on the seat a moment, turning away (an unplug takes 26
  s), and Luca's next dock is out of sight again: still a station taken --
  the while a robot by the seat holds it is its own, never spent by the plug
  -- and Luca plugs in once Rowan goes (the first shape searched there)."""
  luca, rowan = _pair(rowan_at=None)
  _live(luca, rowan, 1)
  rowan.body.x, rowan.body.y = 30.0, 0.0                # ...not by the seat yet
  rowan.body.body_xy = None
  dock = luca.body.dock_routine
  marks = {}

  def docking(*a, **kw):
    if not marks:                                       # its first look: Rowan plugs in
      marks["plugged"] = float(luca.data.time)
      rowan.body.x, rowan.body.y = 20.702, -5.574
      rowan.body.body_xy = ROWAN_BODY
      rowan.body.on_charger = True
    return (yield from dock(*a, **kw))
  luca.body.dock_routine = docking

  def hook():
    t = float(luca.data.time)
    if "plugged" in marks and rowan.body.on_charger and t >= marks["plugged"] + 200.0:
      rowan.body.on_charger = False                      # unplugged, still on the seat
      marks["unplugged"] = t
    if "unplugged" in marks and rowan.body.body_xy is not None and t >= marks["unplugged"] + 30.0:
      rowan.body.x, rowan.body.y = rowan.home_pose[0], rowan.home_pose[1]
      rowan.body.body_xy = None
  luca.body.step_hooks.append(hook)
  assert luca.body.run(luca.go_charge_routine()) is True, luca.log[-4:]
  assert luca.body.lost_at == [], "searched behind Rowan's unplug"
  assert _said(luca, "WAIT: Rowan is plugged in at the charging station")
  assert _said(luca, "by the charging station, which I do not see")


def test_a_while_by_the_seat_is_120_s_a_station_its_walks_included_each_wait_counted():
  """Rowan by the seat goes after 100 s and is back by it as Luca walks in
  again, every walk 30 s long: the while it holds the station is 120 s a
  station a charge in all, the walks out of its way counted in it (the
  first shape began a second wait with 20 s left and walked 30), and the
  wait is said, counted and remembered once, as #346's are."""
  luca, rowan = _pair(rowan_at=None)
  _live(luca, rowan, 1)
  luca.body.lost = True                               # ...and Luca's dock sees nothing
  walk = luca.body.go_to_routine

  def slow(x, y, timeout=90.0, stop=None):
    yield from luca.body._wait(min(30.0, timeout))
    return (yield from walk(x, y, timeout, stop))
  luca.body.go_to_routine = slow
  t0 = float(luca.data.time)
  steps = {"away": False}

  def hook():
    t = float(luca.data.time)
    if not steps["away"] and t >= t0 + 100.0:
      steps["away"] = True
      rowan.body.x, rowan.body.y = 30.0, 0.0           # gone...
    elif steps["away"] and t >= t0 + 101.0 and rowan.body.x > 25.0:
      rowan.body.x, rowan.body.y = 20.702, -5.574      # ...and back by the seat
  luca.body.step_hooks.append(hook)
  held = []
  oos = luca._out_of_sight_routine

  def timed(*a, **kw):
    t = float(luca.data.time)
    out = yield from oos(*a, **kw)
    held.append(float(luca.data.time) - t)
    return out
  luca._out_of_sight_routine = timed
  waits0 = luca.bay_waits
  assert luca.body.run(luca.go_charge_routine()) is False
  assert len(held) == 2 and sum(held) <= lc.UNSEEN_WAIT_S + lc.BAY_POLL_S + 1e-6, held
  assert luca.bay_waits - waits0 == 1
  said = [ln for ln in luca.thoughts.read("History.md").splitlines()
          if "which I did not see; I waited" in ln]
  assert len(said) == 1, said
  assert luca.body.lost_at == ["station"]


def test_a_robot_by_the_seat_that_plugs_in_is_waited_for_as_one_plugged_in():
  """...and one by the seat that plugs in as Luca waits on it a while is a
  robot plugged in: said so, and waited for as long as a charge, never the
  while's 120 s it was first said."""
  luca, rowan = _pair(rowan_at=None)
  _live(luca, rowan, 1)
  t0 = float(luca.data.time)

  def hook():
    if float(luca.data.time) >= t0 + 30.0 and not rowan.body.on_charger:
      rowan.body.on_charger = True
  luca.body.step_hooks.append(hook)
  assert luca.body.run(luca.go_charge_routine()) is False
  bound, _ = luca.wait_bound("charge", 0.0, 0.0)
  assert float(luca.data.time) - t0 >= bound, "the while's 120 s ended a wait for a plug"
  assert _said(luca, "WAIT: Rowan is plugged in at the charging station -- waiting up to 1386 s")
  assert luca.body.lost_at == [] and "Rowan was plugged in at it" in luca.charge_failure


def test_no_lost_search_behind_a_plug_in_made_as_the_robot_looked_round():
  """Walks that never got there are a robot lost too (#572) -- but never
  searched for behind a robot that plugged in there as this one looked
  round after its second walk: the charge ends, naming it, and the mind
  decides again."""
  luca, rowan = _pair(rowan_at=None)
  rowan.body.x, rowan.body.y = 30.0, 0.0
  rowan.body.body_xy = None
  rounds = []

  def far(x, y, timeout=90.0, stop=None):
    luca.body.last_drive = {"why": "no_route", "goal": (float(x), float(y)),
                            "seconds": 1.0, "shortM": 6.0}
    return False
    yield

  def looks():
    rounds.append(1)
    if len(rounds) == 2:                                # ...plugged in as it looked again
      rowan.body.x, rowan.body.y = ROWAN_BODY
      rowan.body.body_xy = ROWAN_BODY
      rowan.body.on_charger = True
    return
    yield
  luca.body.go_to_routine, luca.body.look_around_routine = far, looks
  assert luca.body.run(luca.go_charge_routine()) is False
  assert luca.body.lost_at == [], "a lost search behind a robot plugged in"
  assert "Rowan was plugged in at it" in luca.charge_failure, luca.charge_failure


def test_a_robot_plugged_in_elsewhere_is_never_by_this_seat():
  """Where a robot plugged in at the mall says it stands is nothing to the
  lobby's seat: its plug puts it at the mall, whatever its belief."""
  luca, rowan = _pair(rowan_at="station_mall")
  rowan.body.x, rowan.body.y = SEATS[LOBBY][0] + 0.2, SEATS[LOBBY][1]   # its belief, lost
  rowan.body.body_xy = (SEATS["station_mall"][0], SEATS["station_mall"][1])
  assert luca.body.peer_by_seat(*SEATS[LOBBY]) is None
  assert luca.body.plugged_on(*SEATS["station_mall"]) == rowan.root


# ---- the guards: a robot really lost still searches --------------------------


@pytest.mark.parametrize("rowan", ["far", "1.5 m off", "beside"])
def test_a_station_truly_out_of_sight_is_still_a_robot_lost_and_searched(rowan):
  """The rule must not hide a robot really lost: its station empty, the
  other robot elsewhere, or standing 1.5 m off the seat, or beside it 0.6 m
  off by its report and its look alike -- nobody on it -- is searched for at
  once, as #476 and #572 built it."""
  luca, other = _pair(rowan_at=None)
  luca.body.lost = True                              # no station in sight
  other.body.resting = False
  where = {"far": (other.home_pose[0], other.home_pose[1]),
           "1.5 m off": (SEATS[LOBBY][0] + 1.5, SEATS[LOBBY][1]),
           "beside": (SEATS[LOBBY][0] - 0.6, SEATS[LOBBY][1])}[rowan]
  other.body.x, other.body.y = where
  other.body.body_xy = where
  assert luca.body.run(luca.go_charge_routine()) is False
  assert luca.body.lost_at == ["station"], luca.log[-4:]
  assert not _said(luca, "WAIT:"), luca.log


def test_a_station_out_of_sight_is_held_by_the_other_by_its_seat_a_while_at_most():
  """...but by its seat by its own word alone, nobody plugged in there, it
  holds the station `UNSEEN_WAIT_S` at most: a robot itself lost says it
  stands where it does not (the fuzz's seed 73), and the one waiting may be
  lost too. Then the station is searched for -- here behind Rowan's body,
  never found -- and the charge's failure is the base's own."""
  luca, rowan = _pair(rowan_at=None)
  _live(luca, rowan, 1)                             # by the seat by its report alone
  t0 = float(luca.data.time)
  assert luca.body.run(luca.go_charge_routine()) is False
  assert lc.UNSEEN_WAIT_S <= float(luca.data.time) - t0 < lc.UNSEEN_WAIT_S + 5.0
  assert luca.body.lost_at == ["station"], luca.log[-4:]
  assert _said(luca, "WAIT: Rowan is standing at rest by the charging station, which I do not "
               f"see -- waiting up to {lc.UNSEEN_WAIT_S:.0f} s"), luca.log
  assert _said(luca, "WAIT: Rowan was still by the charging station after 12"), luca.log
  assert luca.charge_failure == f"no charge contact ({NOT_SEEN})", luca.charge_failure


def test_a_station_found_by_its_search_is_no_longer_held_out_of_sight():
  """Rowan, lost itself, SAYS it stands by the lobby's seat while its body
  is across the street, and Luca, lost, saw no station: held a while, then
  searched for and found -- and found, the station is in sight, so Rowan's
  word holds it no more: Luca walks in and plugs in (the first shape held
  it again, its wait's bound spent, and failed)."""
  luca, rowan = _pair(rowan_at=None)
  rowan.body.x, rowan.body.y = SEATS[LOBBY][0] + 0.3, SEATS[LOBBY][1]
  rowan.body.body_xy = (20.0, -12.0)
  luca.body.lost = True

  def found(fixture, stop=None):
    luca.body.lost_at.append(fixture)
    luca.body.lost = False                         # ...its belief put right
    return True
    yield
  luca.body.lost_routine = found
  assert luca.body.run(luca.go_charge_routine()) is True, luca.log[-4:]
  assert luca.body.lost_at == ["station"] and luca.body.on_charger


def test_one_come_onto_the_way_in_of_a_station_out_of_sight_holds_it_a_while_too():
  """Rowan comes onto the way in as Luca's dock looks, its plug out, and
  stays, at rest: on the way in by the planner's own disc, but plugged in
  nowhere -- a while, as one by the seat holds it, asked off its way, never
  a charge's 23 minutes (the first shape waited its bound), then searched."""
  luca, rowan = _pair(rowan_at=None)
  rowan.body.x, rowan.body.y = 50.0, 50.0           # away, as Luca walks in
  rowan.body.body_xy = None
  luca.body.lost = True
  dock = luca.body.dock_routine

  def comes(*a, **kw):
    out = yield from dock(*a, **kw)
    rowan.body.x, rowan.body.y = SEATS[LOBBY][0], SEATS[LOBBY][1] + 0.25  # on its way in
    return out
  luca.body.dock_routine = comes
  t0 = float(luca.data.time)
  assert luca.body.run(luca.go_charge_routine()) is False
  assert float(luca.data.time) - t0 < lc.UNSEEN_WAIT_S + 5.0
  assert luca.body.lost_at == ["station"] and luca.body.asked_off, luca.log[-4:]


def test_a_station_never_in_sight_waited_for_to_its_bound_is_searched_after_all():
  """...and one never in sight that is held to its wait's bound -- Rowan
  plugged in as the dock looked, then unplugged and at rest on the
  standoff's far side, out of the seat's width, asked off its way and never
  stepping off -- nobody plugged in by then, is searched for: a robot lost
  and never searched dies flat."""
  luca, rowan = _pair(rowan_at=None)
  rowan.body.x, rowan.body.y = 50.0, 50.0           # away, as Luca walks in
  rowan.body.body_xy = None
  luca.body.lost = True
  dock = luca.body.dock_routine
  t0 = float(luca.data.time)

  def comes(*a, **kw):
    out = yield from dock(*a, **kw)
    if not rowan.body.on_charger and float(luca.data.time) < t0 + 1.0:
      rowan.body.x, rowan.body.y = ROWAN_BODY       # plugged in as Luca looked
      rowan.body.on_charger = True
    return out
  luca.body.dock_routine = comes

  def hook():
    if rowan.body.on_charger and float(luca.data.time) >= t0 + 60.0:
      rowan.body.on_charger = False                 # ...out, and at rest past the standoff
      rowan.body.x, rowan.body.y = SEATS[LOBBY][0], SEATS[LOBBY][1] + 0.8
  luca.body.step_hooks.append(hook)
  assert luca.body.run(luca.go_charge_routine()) is False
  bound, _ = luca.wait_bound("charge", 0.0, 0.0)
  assert float(luca.data.time) - t0 >= bound
  assert luca.body.lost_at == ["station"], luca.log[-4:]
  assert luca.body.asked_off, "a robot at rest on the standoff is asked off"


# ---- what still ends a wait --------------------------------------------------


def test_the_bound_and_the_runs_hour_still_end_a_wait_for_a_plugged_in_robot():
  """Rowan charged 107 minutes on 11-18: a station it never leaves is waited
  for to the bound (3x a charge's typical occupancy) and the charge fails
  naming it -- never searched, its plug in all along -- and the run's hour
  ends the wait as #654 built it."""
  luca, rowan = _pair()
  _live(luca, rowan, 0)
  t0 = float(luca.data.time)
  assert luca.body.run(luca.go_charge_routine()) is False
  bound, _ = luca.wait_bound("charge", 0.0, 0.0)
  assert float(luca.data.time) - t0 == pytest.approx(bound, abs=2.0)
  assert luca.body.lost_at == [] and luca.body.docks == []
  assert ("Rowan was plugged in at it" in luca.charge_failure
          and "3x how long one charge" in luca.charge_failure), luca.charge_failure
  # ...and the hour
  luca, rowan = _pair()
  _live(luca, rowan, 0)
  luca.continuing, luca.max_sim_time = True, float(luca.data.time) + 120.0
  assert luca.body.run(luca.go_charge_routine()) is False
  assert float(luca.data.time) < luca.max_sim_time + 5.0
  assert luca.body.lost_at == [] and _said(luca, "this run's time is up"), luca.log[-3:]


def test_a_charge_a_persons_message_broke_still_holds_its_station():
  """#730: a person's message ends Rowan's charge's hold with its plug in,
  and the loop asks. Its plug is still in, so the station is still held:
  Luca waits on, never walks in on a charge the message only paused."""
  from pluggybot.mind import events as ev
  from pluggybot.mind import overseer as ov
  from pluggybot.mind.inbox import Inbox
  from test_overseer import FakeClient, full
  menu = ov.Menu.for_world(WORLD, lc.board_book(WORLD))
  mind = ov.Overseer(menu, origin="unseeded", client=FakeClient(full()),
                     event_map=ev.EventMap([ev.Row(event="message_received", action=ev.ASK)]))
  luca, rowan = _pair(rowan_kw={"overseer": mind, "inbox": Inbox()})
  _live(luca, rowan, 2)
  rowan.max_sim_time = 1e9
  rowan.battery.energy_wh = rowan.battery.capacity_wh * 0.4
  rowan.inbox.offer({"type": "message", "id": "m_1", "from": "a visitor", "text": "hello?"})
  rowan.body.run(rowan._after_decision_routine(ov.Decision(action="charge")))
  assert rowan._open_charge is not None and rowan.body.on_charger, "the break, its plug in"
  luca.body.choose_station()
  sx, sy, _ = luca.body.charge_standoff()
  held = luca._bay_holder(sx, sy, luca.body.charge_seat())
  assert held is not None and held[3] == "plugged", held


@pytest.mark.parametrize("seed", [37, 5])
def test_seeded_days_of_the_pair_keep_every_invariant_at_the_stations(seed):
  """The station's fuzz (#731, `fuzz_station_taken`): seeded random days of
  the stub pair at the world's three stations -- charges waiting or passing
  with `free`, plugged in a while at rest on the seat, on a seat with the
  plug out or near one, beliefs off and put right by a search, a fall
  plugged in, stand-ups by or onto a seat, restarts mid-trip and mid-wait --
  every invariant read at every step. A cheap slice, 0.6 s: on the tree
  before, seed 37 tried a dock where the other said it was plugged in at
  21 s and seed 5 a lost search at 42 s; hundreds of seeds flew offline as a
  measurement, never a test kept out of the run."""
  from fuzz_station_taken import fly
  out = fly(seed, sim_s=600.0)
  assert out["simS"] >= 600.0 and out["trips"] >= 2, out
