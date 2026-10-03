"""The pair at the rack (issue #346): done there means gone, a taken bay is
waited for, and a refused return is tried again before anything else.

The loop's half, on the stub: the body's rule for a peer on a goal
(`Body.peer_on_the_goal`) is stubbed with a reported pose, the drives are
stubbed, and a clock only advances (`_clock`) -- nothing flies. A bay wait
inside the quadruped's own swap is #418's.
"""
import inspect
import math
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.mind import events as ev
from pluggybot.mission.errand import Errand
from pluggybot.rack.coupling import STATION_YS
from pluggybot.robot import SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

STATION = STATION_YS[0]
#: The stand-in for the body's rule (`Navigator.peer_on_the_goal`): a peer
#: this near a goal makes it unreachable.
ON_THE_GOAL_M = 0.45


def _peer(x: float, y: float, state: str = "IDLE"):
  """The other robot's PUBLIC surface and no more: a name, a state, a
  reported pose -- mutable, so a test can walk it away -- whether it is
  lying down (issue #365; it is not), and a fork with nothing on it."""
  pos = [x, y]
  return SimpleNamespace(robot_name="Rowan", root=SECOND.root, state=state,
                         pos=pos, down=lambda: False, body=SimpleNamespace(
                           pose_xy=lambda: (pos[0], pos[1]),
                           module_state=lambda t: {"on_fork": False, "hung": True}))


def _reported(life, *poses) -> None:
  """The body answers a goal off these reported poses (callables), as the
  body's rule does off the others' broadcast."""
  def rule(x, y):
    near = [math.hypot(px - x, py - y) for px, py in (p() for p in poses)]
    near = [d for d in near if d < ON_THE_GOAL_M]
    return min(near) if near else None
  life.body.peer_on_the_goal = rule


def _life(peer=None):
  life = stub_life()
  life.home_pose = tuple(float(v) for v in lc.world_config("home_quad")["start"])
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
  """A bay's standoff: where a robot stands to swap there."""
  sx, sy, _ = life.body.bay_standoff(STATION)
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
  """A swap, a charge or a failed pick ends at the rack, and the next
  thing the loop does with a mind is ASK -- standing in the other robot's
  lane for as long as the call flies. So the two branches that do not go
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
  life.state = "SWAP_PICK"
  for t in range(0, int(lc.RACK_LINGER_S) + 5):
    life.data.time = float(t)
    life._rack_linger_step()
  assert not [line for line in life.log if "RACK: lingering" in line]


# ---- rule 2: a taken bay is waited for --------------------------------------


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
  assert life.body.run(life._await_bay_routine(sx, sy, "pick", 0.0)) is True
  assert life.data.time == pytest.approx(20.0, abs=lc.BAY_POLL_S)
  spot = drives[0]
  assert math.hypot(spot[0] - sx, spot[1] - sy) > 0.6, "waited in the lane"
  assert "Rowan was at the bay I needed" in life.thoughts.read("History.md")
  assert any("WAIT: the bay is free after 20 s" in line for line in life.log)
  assert life.bay_waits == 1

  peer.pos[0] = sx + 0.2                       # ...and it comes back as we go in
  assert life.body.run(life._await_bay_routine(sx, sy, "pick", 0.0)) is True
  assert drives[1] == spot, drives
  assert life.bay_waits == 1
  assert life.thoughts.read("History.md").count("Rowan was at the bay I needed") == 1


def test_the_wait_is_bounded_at_three_typical_occupancies_of_what_holds_it():
  """Ben, 2026-09-24: 3x the MEASURED typical occupancy of that bay type,
  and what the holder is doing -- its public state -- picks which."""
  assert lc.WAIT_OCCUPANCIES == 3.0
  assert (lc.SWAP_OCCUPANCY_S, lc.CHARGE_OCCUPANCY_S) == (30.0, 462.0)
  for state, occupancy in (("SWAP_PICK", lc.SWAP_OCCUPANCY_S),
                           ("CHARGE", lc.CHARGE_OCCUPANCY_S)):
    life = _life()
    sx, sy = _standoff(life)
    peer = _peer(sx + 0.2, sy, state)
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
    _clock(life)
    life.body.go_to_routine = lambda *a, **kw: tick.result(False)
    assert life.body.run(life._await_bay_routine(sx, sy, "pick", 0.0)) is False
    assert life.data.time == pytest.approx(3 * occupancy, abs=lc.BAY_POLL_S)
    said = life.held_for(("Rowan", 0.2))
    assert said.startswith("Rowan was standing 0.20 m from the bay")
    assert f"I waited {3 * occupancy:.0f} s for it to leave" in said, said
    assert ("charge" if state == "CHARGE" else "swap") + " usually takes" in said


def test_the_robots_own_battery_row_interrupts_a_picks_wait_and_nothing_else():
  """A pick's wait is the errand's, so the errand's interrupt reaches it --
  a `battery_below` row that names an action ends it without a call. A
  CHARGE's wait is not interrupted (charging is what the row asks for),
  and nor is a RETURN's: abort means stow, and a procedure's `stow()` runs
  inside its errand -- ended early it left the tool riding the fork."""
  for kind, stops in (("pick", True), ("return", False), ("charge", False)):
    life = _life()
    sx, sy = _standoff(life)
    peer = _peer(sx + 0.2, sy, "CHARGE")
    life.peers = [peer]
    _reported(life, peer.body.pose_xy)
    life.body.go_to_routine = lambda *a, **kw: tick.result(False)
    row = ev.Row(event="battery_below", action="charge", value=0.3)

    def fire(t, life=life, row=row):
      if t == 10.0:
        life._interrupt_pending = row
    _clock(life, fire)
    life._in_errand = True
    free = life.body.run(life._await_bay_routine(sx, sy, kind, 0.0))
    assert free is False
    if stops:
      assert life.last_bay_wait["why"] == "interrupted"
      assert life.data.time == pytest.approx(10.0, abs=lc.BAY_POLL_S)
      assert life._aborting
    else:
      assert life.last_bay_wait["why"] == "bound"


def test_an_errand_whose_wait_was_interrupted_is_not_an_error():
  """An act of caution, on SAFE POINT TWO's terms: `interrupted`, never
  `error`, and no "could not pick up" line blaming the tool."""
  life = _life()

  def gave_up(*a, **kw):
    life._aborting = True
    return tick.result("peer-at-bay")

  life.body.fetch_tool_routine = gave_up
  errand = Errand(name="carry:test", module="module_lcd", station_y=STATION,
                  use_at=(1.0, 1.0), use=lambda _l: {}, needs_use_pose=False)
  result = life.run_errand(errand)
  assert result.get("interrupted") is True and "error" not in result
  assert "could not pick up" not in life.thoughts.read("History.md")


def test_a_taken_charge_bay_is_waited_for_before_the_drive_in():
  """The charge bay is the one whose loss is a death: all 4 `flat`
  deaths on 42f4a11 followed a refused charge. The drive in waits for it,
  and a look round for a bay somebody is standing on is not taken."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  peer = _peer(sx + 0.2, sy, "SWAP_PICK")
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


# ---- rule 3: a refused return is retried first ------------------------------


def test_a_refused_return_is_retried_before_the_next_errand(monkeypatch):
  """Luca, three times: a stow refused, the next drawing "picked" the pen
  already on its fork, the drive gave up, the stow was refused again. A
  robot never starts a job with a tool it failed to hang still on its
  fork -- and the retries are bounded, or a stow that misses for a
  mechanical reason would starve the loop (and its mind) for good."""
  from pluggybot.procedure import steps
  cfg = lc.world_config("home_quad")
  for clears in (True, False):
    life = _life()
    held = {"tool": "module_pen"}
    order = []
    monkeypatch.setattr(steps, "_carried", lambda _l: held["tool"])

    def stow(_l, _args, clears=clears):
      order.append("stow")
      if clears:
        held["tool"] = None
      return tick.result({"ok": clears, "reason": "missed"})

    monkeypatch.setattr(steps, "_stow", stow)

    def errand(e, life=life):
      order.append("errand")
      life._end_run = True
      return tick.result({})

    life.run_errand_routine = errand
    life.errands = [SimpleNamespace(name="draw:x", module="module_pen")]
    life._afford_next = lambda: True
    life.run(cfg["start"], max_sim_time=60.0)
    assert order == (["stow", "errand"] if clears
                     else ["stow"] * lc.STOW_RETRIES + ["errand"]), order


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
  sent the swap round its loop at no sim time at all."""
  life = _life()
  sx, sy = _standoff(life)
  _reported(life, lambda: (sx + 0.2, sy))      # routed round, unnamed
  _clock(life)
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  assert life.body.run(life._await_bay_routine(sx, sy, "pick", 0.0)) is False
  assert life.data.time == pytest.approx(3 * lc.SWAP_OCCUPANCY_S,
                                         abs=lc.BAY_POLL_S)
  assert "another robot" in life.thoughts.read("History.md")


# ---- rule 3, on legs (#418): a robot holds at its approach's start --------------


def test_a_quadruped_holds_where_it_stands_and_the_lifecycle_is_who_it_asks():
  """The quadruped's swap waits at its approach's start, a metre behind the
  bay and out of the holder's way already (Ben, #418): `hold` walks
  nowhere, where the rover's wait backs off to a spot. The body asks the
  lifecycle's wait (`Body.bay_wait`)."""
  life = _life()
  assert life.body.bay_wait == life._await_bay_routine
  sx, sy = _standoff(life)
  peer = _peer(sx + 0.2, sy)
  life.peers = [peer]
  _reported(life, peer.body.pose_xy)
  _clock(life, lambda t: peer.pos.__setitem__(0, sx + 5.0) if t >= 12.0 else None)
  drives = []
  life.body.go_to_routine = lambda x, y, *a, **kw: (drives.append((x, y)),
                                                    tick.result(True))[1]
  assert life.body.run(life._await_bay_routine(sx, sy, "pick", 0.0, hold=True)) is True
  assert drives == [], "a held wait walked"
  assert life.data.time == pytest.approx(12.0, abs=lc.BAY_POLL_S)


from pluggybot.legs.swap import ToolSwap  # noqa: E402


class _Bay(ToolSwap):
  """`legs.swap.ToolSwap`'s wait and fetch alone: a work pose, a peer at a
  distance from it (the planner's disc, 0.55 m standing), and a walk to the
  approach that always arrives."""

  def __init__(self, peer_m, free=True):
    self.peer_m, self.free, self.asked, self.at_bay = peer_m, free, [], False
    self.data = SimpleNamespace(time=3.0)
    self.bay_wait = self._wait
    self.peer_at_bay_m = None
    self.tool_rack_prior = object()

  def work_pose(self, bay):
    return (1.0 + 0.3 * bay, 2.0, 0.0)

  def peer_on_the_goal(self, wx, wy):
    return self.peer_m if self.peer_m is not None and self.peer_m < 0.55 else None

  def _wait(self, wx, wy, kind, since, hold=False):
    self.asked.append((wx, wy, kind, since, hold))
    if self.free:
      self.peer_m = None
    return self.free
    yield

  def stand_routine(self):
    return
    yield

  stow_arm_routine = stand_routine

  def _to_the_bay_routine(self, bay, rec):
    return "ok"
    yield

  def _at_the_bay(self, bay):
    import contextlib

    @contextlib.contextmanager
    def at():
      self.at_bay = True
      yield
    return at()

  def _fetch_at_routine(self, bay, module, rec):
    return "arrived"
    yield


def _drive(gen):
  try:
    while True:
      next(gen)
  except StopIteration as stop:
    return stop.value


def test_the_swap_waits_on_the_work_pose_and_a_wait_that_gives_up_is_blocked():
  """The next bay's working pose is 0.30 m off and two over 0.60: the
  wait is the planner's disc (0.55 m standing) round THIS bay's working
  pose, never its standoff. A wait that gave up is `blocked`, said with
  how far off the robot in the way stood, and the robot never walks in."""
  clear = _Bay(peer_m=0.60)
  assert _drive(ToolSwap.fetch_routine(clear, 0, "module_lcd")) == "arrived"
  assert clear.asked == [] and clear.at_bay
  near = _Bay(peer_m=0.30)
  assert _drive(ToolSwap.fetch_routine(near, 1, "module_lcd")) == "arrived"
  assert near.asked == [(1.3, 2.0, "pick", 3.0, True)]
  held = _Bay(peer_m=0.30, free=False)
  rec = {}
  assert _drive(ToolSwap._bay_free_routine(held, 1, "return", rec)) is False
  assert rec["why"] == "blocked" and held.peer_at_bay_m == 0.30
  held = _Bay(peer_m=0.30, free=False)
  assert _drive(ToolSwap.fetch_routine(held, 1, "module_lcd")) == "blocked"
  assert not held.at_bay, "it walked in to a bay another robot held"


def test_who_held_a_bay_is_forgotten_by_the_next_swap():
  """A wait given up for the other robot keeps how far off it stood, for
  the failure's words; the next swap starts without it -- kept, a later
  fetch that found no route was told "Rowan was 0.30 m from the bay"."""
  held = _Bay(peer_m=0.30, free=False)
  assert _drive(ToolSwap.fetch_routine(held, 1, "module_lcd")) == "blocked"
  assert held.peer_at_bay_m == 0.30
  held.peer_m = None

  def nowhere(bay, rec):
    return "no-route"
    yield
  held._to_the_bay_routine = nowhere
  assert _drive(ToolSwap.fetch_routine(held, 1, "module_lcd")) == "no-route"
  assert held.peer_at_bay_m is None


def test_a_stow_given_up_for_the_other_robot_says_who_held_the_bay():
  """A fetch given up for the other robot named it (`held_for`); a stow
  given up the same way said only that the tool was still on the fork."""
  from pluggybot.procedure import steps as st

  def stow(station, tool):
    return "blocked"
    yield
  body = SimpleNamespace(module_state=lambda t: {"on_fork": t == "module_pen",
                                                 "hung": False},
                         stow_tool_routine=stow, retract_arm_routine=lambda: tick.result(None),
                         held_cube=lambda: None, swap_trace=lambda: "")
  life = SimpleNamespace(rack_inventory=dict(st.TOOL_BAYS),
                         swaps_done=0, model=None, world="home_quad", body=body,
                         peer_at_the_bay=lambda station: ("Rowan", 0.3),
                         held_for=lambda b: f"{b[0]} was {b[1]:.2f} m from the bay")
  out = tick.run(SimpleNamespace(step=lambda *a: None), st._stow(life, {}))
  assert not out["ok"]
  assert out["reason"].endswith("still on the fork: Rowan was 0.30 m from the bay")
