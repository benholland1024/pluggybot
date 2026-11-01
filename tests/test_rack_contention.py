"""The pair at the charging station (issue #346): done there means gone, and
a taken charge bay is waited for. "The rack" in the loop's words is the
frame its charge logic works round (`Body.rack_prior`), the humanoids'
charging station since the quadruped's tool rack went with it (#559), and
with it the swaps' waits and the stow's retries.

The loop's half, on the stub: the body's rule for a peer on a goal
(`Body.peer_on_the_goal`) is stubbed with a reported pose, the drives are
stubbed, and a clock only advances (`_clock`) -- nothing flies.
"""
import inspect
import math
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.body import RackPose, StubBody
from pluggybot.mind import events as ev
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang
from pluggybot.robot import SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

#: The stand-in for the body's rule (`Navigator.peer_on_the_goal`): a peer
#: this near a goal makes it unreachable.
ON_THE_GOAL_M = 0.45


def _peer(x: float, y: float, state: str = "IDLE"):
  """The other robot's PUBLIC surface and no more: a name, a state, a
  reported pose -- mutable, so a test can walk it away -- and whether it is
  lying down (issue #365; it is not)."""
  pos = [x, y]
  return SimpleNamespace(robot_name="Rowan", root=SECOND.root, state=state,
                         pos=pos, down=lambda: False,
                         body=SimpleNamespace(pose_xy=lambda: (pos[0], pos[1])))


def _reported(life, *poses) -> None:
  """The body answers a goal off these reported poses (callables), as the
  body's rule does off the others' broadcast."""
  def rule(x, y):
    near = [math.hypot(px - x, py - y) for px, py in (p() for p in poses)]
    near = [d for d in near if d < ON_THE_GOAL_M]
    return min(near) if near else None
  life.body.peer_on_the_goal = rule


def _life(peer=None):
  """The stub with its charge logic round the humanoids' station, framed as
  `HumanoidBody.rack_prior` frames it, and home at the house's start."""
  cfg = lc.world_config("home_humanoid")
  x, y, yaw = cfg["dock"]
  life = stub_life(body=StubBody(rack=RackPose(x, y, yaw + math.pi),
                                 grid_bounds=cfg["grid_bounds"]))
  life.home_pose = tuple(float(v) for v in cfg["start"])
  if peer is not None:
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
  return life


def _clock(life, on_tick=None):
  """Standing still advances the clock and does nothing else."""
  def still(seconds, *a, **kw):
    life.data.time += seconds
    if on_tick is not None:
      on_tick(float(life.data.time))
    return tick.result(None)
  life.body.hold_routine = still


def _at(life, x, y):
  life.body.x, life.body.y = x, y


def _standoff(life):
  """The charge bay's standoff: where a robot stands to walk in and plug in."""
  sx, sy, _ = life.body.charge_standoff()
  return sx, sy


# ---- rule 1: done at the rack means gone ----------------------------------


def test_a_clear_that_did_not_arrive_tries_elsewhere_and_says_so():
  """Rowan narrated "clearing the rack for the others" and stood 0.08 m
  from the charge bay for 23 minutes: `_clear_rack_routine` threw the
  drive's answer away. It is read now -- a clear that ends within
  `RACK_CLEAR_M` tries other spots and writes where it ended in History."""
  life = _life()
  sx, sy = _standoff(life)
  _at(life, sx, sy)
  drives = []
  life.body.go_to_routine = lambda *a, **kw: (drives.append(a[:2]), tick.result(False))[1]
  assert life.body.run(life._clear_rack_routine()) is False
  assert drives[0] == life.home_pose[:2]
  assert len(drives) == 1 + lc.CLEAR_SPOTS, drives
  assert all(life.rack_distance(*d) >= lc.RACK_CLEAR_M for d in drives[1:])
  history = life.thoughts.read("History.md")
  assert "could not get clear of the rack" in history, history

  # ...and a clear that got away is one drive and nothing to report
  life = _life()
  _at(life, sx, sy)
  drives = []

  def goes(x, y, *a, **kw):
    drives.append((x, y))
    _at(life, x, y)
    return tick.result(True)

  life.body.go_to_routine = goes
  assert life.body.run(life._clear_rack_routine()) is True
  assert len(drives) == 1
  assert "clear of the rack" not in life.thoughts.read("History.md")


def test_the_loop_leaves_the_rack_before_it_thinks_but_only_for_somebody():
  """A charge, or one that failed, ends at the station, and the next thing
  the loop does with a mind is ASK -- standing in the other robot's lane
  for as long as the call flies. So the two branches that do not go
  to the rack themselves leave it first; alone in the world, nothing is
  blocked and nothing moves (a single robot's day is unchanged)."""
  src = inspect.getsource(lc.HubLifecycle._day_routine)
  lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
  for call in ("yield from self._arbitrate_routine()",
               "yield from self._grade_routine()"):
    at = lines.index(call)
    assert lines[at - 1] == "yield from self._leave_rack_routine()", call

  drives = []
  for peer in (None, _peer(9.0, 9.0)):
    life = _life(peer)
    _at(life, *_standoff(life))
    life.body.go_to_routine = lambda *a, **kw: (drives.append(a[:2]), tick.result(False))[1]
    life.body.run(life._leave_rack_routine())
    assert bool(drives) is (peer is not None), drives


def test_a_robot_left_standing_at_the_rack_is_logged_once():
  life = _life(_peer(9.0, 9.0))
  sx, sy = _standoff(life)
  _at(life, sx, sy)
  life.state = "DECIDE"
  for t in range(0, int(lc.RACK_LINGER_S) + 5):
    life.data.time = float(t)
    life._rack_linger_step()
  said = [line for line in life.log if "RACK: lingering" in line]
  assert len(said) == 1, said
  # ...and a robot AT the rack for what it came for is not lingering
  life = _life(_peer(9.0, 9.0))
  _at(life, sx, sy)
  life.state = "CHARGE"
  for t in range(0, int(lc.RACK_LINGER_S) + 5):
    life.data.time = float(t)
    life._rack_linger_step()
  assert not [line for line in life.log if "RACK: lingering" in line]


def test_a_refused_procedure_beside_the_station_is_not_lingering():
  """A procedure opens in SWAP_PICK, the quadruped's swap state the frames
  still carry (#559), and one its world refused stays in it while the loop
  goes on: that is no linger, as it never was."""
  life = _life(_peer(9.0, 9.0))
  _at(life, *_standoff(life))
  life.run_errand(programmed_errand(lang.parse('def x():\n  move("shoulder", 0.1)\n')))
  assert life.state == "SWAP_PICK"
  for t in range(0, int(lc.RACK_LINGER_S) + 5):
    life.data.time = float(t)
    life._rack_linger_step()
  assert not [line for line in life.log if "RACK: lingering" in line]


# ---- rule 2: a taken charge bay is waited for ------------------------------


def test_a_taken_bay_is_waited_for_out_of_the_way_and_a_freed_one_proceeds():
  """#313's early return failed the job the moment a drive found another
  robot on the standoff. Now the robot backs off to a spot beside the
  holder's lane, waits, and goes on once the holder has gone -- and a
  wait for the same bay asked again (the drive in found it taken again)
  goes back to the spot, rather than waiting where it stopped, and is
  said once."""
  life = _life()
  sx, sy = _standoff(life)
  peer = _peer(sx + 0.2, sy)
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  _clock(life, lambda t: peer.pos.__setitem__(0, sx + 5.0) if t >= 20.0 else None)
  drives = []
  life.body.go_to_routine = lambda x, y, *a, **kw: (drives.append((x, y)),
                                                    tick.result(True))[1]
  assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is True
  assert life.data.time == pytest.approx(20.0, abs=lc.BAY_POLL_S)
  spot = drives[0]
  assert math.hypot(spot[0] - sx, spot[1] - sy) > 0.6, "waited in the lane"
  assert "Rowan was at the charge bay I needed" in life.thoughts.read("History.md")
  assert any("WAIT: the charge bay is free after 20 s" in line for line in life.log)
  assert life.bay_waits == 1

  peer.pos[0] = sx + 0.2                       # ...and it comes back as we go in
  assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is True
  assert drives[1] == spot, drives
  assert life.bay_waits == 1
  assert life.thoughts.read("History.md").count(
    "Rowan was at the charge bay I needed") == 1


def test_the_wait_is_bounded_at_three_typical_occupancies_of_a_charge():
  """Ben, 2026-09-24: 3x the MEASURED typical occupancy of that bay type --
  a charge's, whatever the holder is doing, the swaps' bay having gone
  with the rack (#559)."""
  assert lc.WAIT_OCCUPANCIES == 3.0 and lc.CHARGE_OCCUPANCY_S == 462.0
  for state in ("IDLE", "CHARGE"):
    life = _life()
    sx, sy = _standoff(life)
    peer = _peer(sx + 0.2, sy, state)
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
    _clock(life)
    life.body.go_to_routine = lambda *a, **kw: tick.result(False)
    assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is False
    assert life.data.time == pytest.approx(3 * lc.CHARGE_OCCUPANCY_S, abs=lc.BAY_POLL_S)
    said = life.held_for(("Rowan", 0.2))
    assert said.startswith("Rowan was standing 0.20 m from the bay")
    assert f"I waited {3 * lc.CHARGE_OCCUPANCY_S:.0f} s for it to leave" in said, said
    assert "charge usually takes" in said


def test_nothing_ends_a_charges_wait_early_but_its_bound():
  """A CHARGE's wait is not interrupted, the robot's own `battery_below`
  row included: charging is what the row asks for, and giving up on the
  charge is how the robot dies. (The picks' waits it ended went with the
  rack, #559.)"""
  life = _life()
  sx, sy = _standoff(life)
  peer = _peer(sx + 0.2, sy, "CHARGE")
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  row = ev.Row(event="battery_below", action="charge", value=0.3)

  def fire(t):
    if t == 10.0:
      life._interrupt_pending = row
  _clock(life, fire)
  life._in_errand = True
  assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is False
  assert life.last_bay_wait["why"] == "bound"
  assert life.data.time == pytest.approx(3 * lc.CHARGE_OCCUPANCY_S, abs=lc.BAY_POLL_S)


def test_a_taken_charge_bay_is_waited_for_before_the_drive_in():
  """The charge bay is the one whose loss is a death: all 4 `flat`
  deaths on 42f4a11 followed a refused charge. The drive in waits for it,
  and a look round for a bay somebody is standing on is not taken."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  peer = _peer(sx + 0.2, sy, "IDLE")
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  _clock(life)
  spins = []
  life.body.look_around_routine = lambda *a, **kw: (spins.append(1), tick.result(None))[1]
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  assert life.body.run(life.go_charge_routine()) is False
  assert life.data.time == pytest.approx(3 * lc.CHARGE_OCCUPANCY_S,
                                         abs=lc.BAY_POLL_S)
  assert spins == [], "looked round for a bay somebody was standing on"
  said = next(line for line in life.log if "GO_CHARGE: never reached the charge bay" in line)
  assert "Rowan was standing 0.20 m from it" in said, said
  assert "I waited 1386 s for it to leave, 3x how long one charge" in said


def test_a_robot_on_the_dock_holds_the_charge_bay_and_is_not_walked_in_on():
  """Two bodies on one dock (#455): the charge bay was taken only by a
  robot on its STANDOFF, and one lying on the dock -- a metre past it --
  was walked in on: both charged at once, and backed out onto each other
  at the standoff, where neither could plan a step. A robot on the seat
  holds the bay as one on the standoff does: waited for out of the lane,
  docked once it has gone, and named "on the dock" if it never goes --
  and one that lay down there while this one walked to the standoff is
  found there before the walk in."""
  for leaves, late in ((True, False), (False, False), (True, True), (False, True)):
    life = _life()
    sx, sy, _ = life.body.charge_standoff()
    seat = life.body.charge_seat()
    peer = _peer(*((seat[0] + 5.0, seat[1]) if late else seat), "CHARGE")
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
    assert life.body.peer_on_the_goal(sx, sy) is None, "the premise: the standoff is free"
    _clock(life, (lambda t: peer.pos.__setitem__(0, seat[0] + 5.0) if t >= 30.0 else None)
           if leaves else None)
    drives, docks = [], []

    def go_to(x, y, *a, late=late, peer=peer, seat=seat, drives=drives, **kw):
      if late and not drives:               # ...it docks as this one walks
        peer.pos[:] = list(seat)
      drives.append((x, y))
      return tick.result(True)
    life.body.go_to_routine = go_to

    def dock():
      docks.append(float(life.data.time))
      life.body.on_charger = True
      return "docked"
      yield
    life.body.dock_routine = dock
    assert life.body.run(life.go_charge_routine()) is leaves
    assert any("WAIT: Rowan is standing on the dock" in line for line in life.log)
    assert math.dist(drives[int(late)], (sx, sy)) > 0.6, "waited in the lane"
    if leaves:
      assert docks == [pytest.approx(30.0, abs=lc.BAY_POLL_S)], docks
      continue
    assert docks == [], "walked in on a robot on the dock"
    said = next(line for line in life.log if "GO_CHARGE: never reached the charge bay" in line)
    assert "-- Rowan was standing on the dock; I waited 1386 s for it to leave" in said, said


def test_a_robot_on_the_docks_way_in_holds_it_as_on_its_standoff_or_seat():
  """The way in from the standoff to the seat is the dock's too (#455's
  review): with the pair colliding, a robot lying across it short of the
  seat -- out of reach of both the standoff's test and the seat's -- turned
  every walk in back, where before the PR it was walked through."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  seat = life.body.charge_seat()
  mx, my = (sx + seat[0]) / 2, (sy + seat[1]) / 2
  ux, uy = (seat[0] - sx) / math.dist((sx, sy), seat), (seat[1] - sy) / math.dist((sx, sy), seat)
  peer = _peer(mx - 0.40 * uy, my + 0.40 * ux, "IDLE")   # 0.40 m off the way, half way in
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  assert life.body.peer_on_the_goal(sx, sy) is None and life.body.peer_on_the_goal(*seat) is None
  _clock(life)
  docks = []
  life.body.go_to_routine = lambda *a, **kw: tick.result(True)

  def dock():
    docks.append(float(life.data.time))
    return "docked"
    yield
  life.body.dock_routine = dock
  assert life.body.run(life.go_charge_routine()) is False
  assert docks == [], "walked in across the robot on the dock's way in"
  assert any("WAIT: Rowan is standing on the dock's way in" in line for line in life.log)
  said = next(line for line in life.log if "GO_CHARGE: never reached the charge bay" in line)
  assert "-- Rowan was standing on the dock's way in; I waited 1386 s" in said, said


def test_a_resting_holder_is_asked_afresh_after_a_no_or_a_yes_that_did_not_move_it():
  """A yes counts `MAKE_WAY_WAIT_S` from the first (`Navigator._ask_way`),
  and after that the robot was never asked again: one whose step aside
  ended short and lay down on the bay again was waited on for the rest of
  the bound (#455's review). The wait asks it afresh once its yes has run
  out -- and after a no, only that often: each ask plans a way and an aside
  (~30 ms on the physics thread), and a no is narrated that often anyway."""
  from pluggybot.navigator import MAKE_WAY_WAIT_S, OTHER_WAIT_S
  for says in ("yes", "no"):
    life = _life()
    sx, sy = _standoff(life)
    peer = _peer(sx + 0.2, sy)
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
    _clock(life)
    asked_at = []

    def ask_off(x, y, asked, beyond=(), says=says, peer=peer, life=life, asked_at=asked_at):
      now = float(life.data.time)
      if now - asked.get(peer.root, now) > MAKE_WAY_WAIT_S:
        return False                        # the real `_ask_way` passes it over
      asked_at.append(now)
      if says == "no":
        return False
      asked.setdefault(peer.root, now)      # ...says yes, and never gets off
      return True
    life.body.ask_off = ask_off
    assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is False
    if says == "no":
      assert len(asked_at) >= 3
      assert asked_at == pytest.approx([k * MAKE_WAY_WAIT_S for k in range(len(asked_at))])
      continue
    window = [t for t in asked_at if t <= MAKE_WAY_WAIT_S]
    assert window == pytest.approx([k * OTHER_WAIT_S for k in range(len(window))])
    again = [t for t in asked_at if t > MAKE_WAY_WAIT_S]
    assert again, "never asked again after its first yes ran out"
    assert again[0] == pytest.approx(window[-1] + OTHER_WAIT_S + MAKE_WAY_WAIT_S)


def test_a_dock_found_again_after_a_lost_search_is_looked_at_again():
  """#476's lost search from the standoff takes up to 300 s, and the other
  robot can dock meanwhile: the walk back looks at the bay again, as every
  walk to it does (#455's review), and never walks in on the one there."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  seat = life.body.charge_seat()
  peer = _peer(seat[0] + 5.0, seat[1], "CHARGE")
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  _clock(life, lambda t: peer.pos.__setitem__(0, seat[0] + 5.0) if t >= 40.0 else None)
  life.body.go_to_routine = lambda *a, **kw: tick.result(True)
  docks = []

  def dock():
    docks.append((float(life.data.time), list(peer.pos)))
    if len(docks) == 1:
      return "no board"
    life.body.on_charger = True
    return "docked"
    yield

  def lost(fixture, stop=None):
    peer.pos[:] = list(seat)                 # ...and it docked while this one looked
    life.data.time += 30.0
    return True
    yield
  life.body.dock_routine, life.body.lost_routine = dock, lost
  assert life.body.run(life.go_charge_routine()) is True
  assert len(docks) == 2 and docks[1][0] >= 40.0, docks
  assert math.dist(docks[1][1], seat) > 1.0, "walked in on the robot on the dock"


def test_a_robot_resting_on_a_bay_is_asked_off_it_while_it_is_waited_for():
  """A wait for a bay held by a robot lying down to rest ran its bound and
  never asked it to move: 1387 s at the charge bay, 90 s at the rack's, on
  afb473a every time (#455). The wait asks, every `OTHER_WAIT_S`, for the
  bay it waits on and on past it to the station's seat (`Body.ask_off`:
  the body's rule is who), and goes on once the robot is off it."""
  from pluggybot.navigator import OTHER_WAIT_S
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  seat, beyond = life.body.charge_seat(), (tuple(life.body.charge_seat()),)
  peer = _peer(sx + 0.2, sy)
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  _clock(life)
  asks = []

  def ask_off(x, y, asked, beyond=()):
    asks.append((float(life.data.time), (x, y), tuple(beyond)))
    if len(asks) == 3:                      # ...and on the third, it is off
      peer.pos[0] += 5.0
    return True
  life.body.ask_off = ask_off
  assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0,
                                               seat=seat)) is True
  assert [a[1:] for a in asks] == [((sx, sy), beyond)] * 3
  assert [a[0] for a in asks] == pytest.approx([0.0, OTHER_WAIT_S, 2 * OTHER_WAIT_S])
  assert life.data.time < 3 * OTHER_WAIT_S + lc.BAY_POLL_S


# ---- found reviewing the PR ---------------------------------------------------


def test_a_clear_that_failed_is_not_driven_again_from_the_same_place():
  """`_leave_rack_routine` runs before EVERY decision: a robot that could
  not get away re-drove every spot (minutes) and wrote a History line per
  pass. From where it gave up, it does not try again until it has moved."""
  life = _life(_peer(9.0, 9.0))
  sx, sy = _standoff(life)
  _at(life, sx, sy)
  drives = []
  life.body.go_to_routine = lambda *a, **kw: (drives.append(a[:2]), tick.result(False))[1]
  life.body.run(life._leave_rack_routine())
  tried = len(drives)
  assert tried == 1 + lc.CLEAR_SPOTS
  life.body.run(life._leave_rack_routine())
  assert len(drives) == tried, "drove the same failed clear again"
  _at(life, sx + 0.5, sy)                      # it went somewhere, and came back
  life.body.run(life._leave_rack_routine())
  assert len(drives) == 2 * tried


def test_an_old_wait_is_not_told_as_this_failures_story():
  """`held_for` adds "I waited N s" only for the wait that ended THIS
  failure: a charge refused after a spin, hours after a wait for the same
  robot, claimed a 1386 s wait that never happened."""
  life = _life(_peer(0.0, 0.0))
  life.last_bay_wait = {"who": "Rowan", "why": "bound", "s": 1386.0,
                        "bound": 1386.0, "of": "charge", "end": 10.0}
  life.data.time = 10.0
  assert "I waited 1386 s" in life.held_for(("Rowan", 0.2))
  life.data.time = 4000.0
  assert life.held_for(("Rowan", 0.2)) == ("Rowan was standing 0.20 m from "
                                           "the bay, nearer than the planner may route")


def test_a_peer_the_lifecycle_cannot_name_is_still_waited_for():
  """The wait is keyed on the RULE (`peer_on_the_goal`), never on having a
  name for the robot: answering "free" for a peer the planner routes round
  sent a wait round its loop at no sim time at all."""
  life = _life()
  sx, sy = _standoff(life)
  _reported(life, lambda: (sx + 0.2, sy))      # routed round, unnamed
  _clock(life)
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  assert life.body.run(life._await_bay_routine(sx, sy, "charge", 0.0)) is False
  assert life.data.time == pytest.approx(3 * lc.CHARGE_OCCUPANCY_S,
                                         abs=lc.BAY_POLL_S)
  assert "another robot" in life.thoughts.read("History.md")
