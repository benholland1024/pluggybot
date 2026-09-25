"""The pair at the rack (issue #346): done there means gone, a taken bay is
waited for, and a refused return is tried again before anything else.

Every rule here is pinned with a stubbed peer pose, a stubbed drive and a
clock that only advances (`_clock`) -- nothing flies. The flown evidence
(the 60 s drive away from the rack, the charge tag seen from beside the
standoff) is in docs/SimNotes.md, "Two robots at one rack".
"""
import inspect
import math
from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.mind import events as ev
from pluggybot.mission.errand import Errand
from pluggybot.mission.mission import bay_standoff, charge_standoff
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.robot import FIRST, SECOND

from test_overseer import _lifecycle  # noqa: I001 -- tests/ is on sys.path

STATION = HUB_STATION_YS[0]


def _peer(x: float, y: float, state: str = "IDLE"):
  """The other robot's PUBLIC surface and no more: a name, a state and a
  reported pose -- mutable, so a test can walk it away."""
  pos = [x, y]
  return SimpleNamespace(robot_name="Rowan", root=SECOND.root, state=state,
                         pos=pos, mission=SimpleNamespace(
                           pose_xy=lambda: (pos[0], pos[1])))


def _life(peer=None):
  life = _lifecycle("room_hub", errand=False)
  mujoco.mj_forward(life.model, life.data)     # modules where they hang
  life.home_pose = tuple(float(v) for v in lc.world_config("room_hub")["start"])
  if peer is not None:
    life.peers = [peer]
    life.mission.others = [peer.mission.pose_xy]
  return life


def _clock(life, on_tick=None):
  """Standing still advances the clock and does nothing else."""
  def still(seconds, v, w):
    life.data.time += seconds
    if on_tick is not None:
      on_tick(float(life.data.time))
    return tick.result(None)
  life.mission._drive_routine = still


def _at(life, x, y):
  life.mission.swap.reckoner.x, life.mission.swap.reckoner.y = x, y


class Arrived(Exception):
  """Raised by a stubbed drive that reached the standoff: the swap past the
  route is not what these tests are about."""


# ---- rule 1: done at the rack means gone ----------------------------------


def test_a_clear_that_did_not_arrive_tries_elsewhere_and_says_so():
  """Rowan narrated "clearing the rack for the others" and stood 0.08 m
  from the charge bay for 23 minutes: `_clear_rack_routine` threw the
  drive's answer away. It is read now -- a clear that ends within
  `RACK_CLEAR_M` tries other spots and writes where it ended in History."""
  life = _life()
  sx, sy, _ = bay_standoff(STATION, life.mission.rack)
  _at(life, sx, sy)
  life.mission.grid.grid[:] = -5.0              # a mapped, empty room
  drives = []
  life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a[:2]),
                                                    tick.result(False))[1]
  assert life.mission.run(life._clear_rack_routine()) is False
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

  life.mission.drive_to_routine = goes
  assert life.mission.run(life._clear_rack_routine()) is True
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
    sx, sy, _ = bay_standoff(STATION, life.mission.rack)
    _at(life, sx, sy)
    life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a[:2]),
                                                      tick.result(False))[1]
    life.mission.run(life._leave_rack_routine())
    assert bool(drives) is (peer is not None), drives


def test_a_robot_left_standing_at_the_rack_is_logged_once():
  life = _life(_peer(9.0, 9.0))
  sx, sy, _ = bay_standoff(STATION, life.mission.rack)
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
  robot on the standoff -- 6 of 47 picks and 7 of 38 stows on the deployed
  pair in one afternoon. Now the robot backs off to a spot beside the
  holder's lane, waits, and drives in once the holder has gone."""
  life = _life()
  sx, sy, _ = bay_standoff(STATION, life.mission.rack)
  peer = _peer(sx + 0.2, sy)
  life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
  life.mission.grid.grid[:] = -5.0
  _clock(life, lambda t: peer.pos.__setitem__(0, sx + 5.0) if t >= 20.0 else None)
  drives = []

  def drive(x, y, *a, **kw):
    drives.append((x, y))
    if math.hypot(x - sx, y - sy) < 1e-6:
      raise Arrived
    return tick.result(True)

  life.mission.drive_to_routine = drive
  with pytest.raises(Arrived):
    life.mission.run(life.mission.swap_at_bay_routine(STATION, "pick",
                                                      module="module_lcd"))
  assert life.data.time == pytest.approx(20.0, abs=lc.BAY_POLL_S)
  spot = drives[0]
  assert math.hypot(spot[0] - sx, spot[1] - sy) > 0.6, "waited in the lane"
  assert drives[-1] == (sx, sy)
  assert "Rowan was at the bay I needed" in life.thoughts.read("History.md")
  assert any("WAIT: the bay is free after 20 s" in line for line in life.log)
  assert life.bay_waits == 1

  # ...and a drive that finds it taken AGAIN goes back to the spot, rather
  # than waiting where it stopped: the edge of the holder's disc is its
  # way out.
  life = _life()
  peer = _peer(sx + 0.2, sy)
  life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
  life.mission.grid.grid[:] = -5.0
  back = {"again": False}

  def holder(t):
    if t >= (25.0 if back["again"] else 10.0):
      peer.pos[0] = sx + 5.0                 # it leaves

  _clock(life, holder)
  drives = []

  def drive_back(x, y, *a, **kw):
    drives.append((x, y))
    if math.hypot(x - sx, y - sy) < 1e-6:
      if back["again"]:
        raise Arrived
      back["again"] = True                   # ...and comes back as we go in
      peer.pos[0] = sx + 0.2
      return tick.result(False)
    return tick.result(True)

  life.mission.drive_to_routine = drive_back
  with pytest.raises(Arrived):
    life.mission.run(life.mission.swap_at_bay_routine(STATION, "pick",
                                                      module="module_lcd"))
  spots = [d for d in drives if d != (sx, sy)]
  assert len(spots) == 2 and spots[0] == spots[1], drives


def test_the_wait_is_bounded_at_three_typical_occupancies_of_what_holds_it():
  """Ben, 2026-09-24: 3x the MEASURED typical occupancy of that bay type.
  A charge holds the neighbouring tool bay (0.200 m off) for the whole
  charge, so what the holder is doing -- its public state -- picks which."""
  assert lc.WAIT_OCCUPANCIES == 3.0
  assert (lc.SWAP_OCCUPANCY_S, lc.CHARGE_OCCUPANCY_S) == (30.0, 462.0)
  for state, occupancy in (("SWAP_PICK", lc.SWAP_OCCUPANCY_S),
                           ("CHARGE", lc.CHARGE_OCCUPANCY_S)):
    life = _life()
    sx, sy, _ = bay_standoff(STATION, life.mission.rack)
    peer = _peer(sx + 0.2, sy, state)
    life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
    _clock(life)
    life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
    why = life.mission.run(life.mission.swap_at_bay_routine(
      STATION, "pick", module="module_lcd"))
    assert why == "peer-at-bay"
    assert life.data.time == pytest.approx(3 * occupancy, abs=lc.BAY_POLL_S)
    said = life.pick_failure("module_lcd", STATION, why)
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
    sx, sy, _ = bay_standoff(STATION, life.mission.rack)
    peer = _peer(sx + 0.2, sy, "CHARGE")
    life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
    life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
    row = ev.Row(event="battery_below", action="charge", value=0.3)

    def fire(t, life=life, row=row):
      if t == 10.0:
        life._interrupt_pending = row
    _clock(life, fire)
    life._in_errand = True
    free = life.mission.run(life._await_bay_routine(sx, sy, kind, 0.0))
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

  life.mission.swap_at_bay_routine = gave_up
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": True}
  errand = Errand(name="carry:test", module="module_lcd", station_y=STATION,
                  use_at=(1.0, 1.0), use=lambda _l: {}, needs_use_pose=False)
  result = life.run_errand(errand)
  assert result.get("interrupted") is True and "error" not in result
  assert "could not pick up" not in life.thoughts.read("History.md")


def test_a_taken_charge_bay_is_waited_for_before_and_during_the_approach():
  """The charge bay is the one whose loss is a death: all 4 `flat`
  deaths on 42f4a11 followed a refused charge. The drive in waits for it,
  and so does the terminal approach -- a peer that arrives while this
  robot lines up leaves it looking from beside the standoff, where the
  tag is out of the camera's view (the six "no-tag" charges)."""
  life = _life()
  sx, sy, _ = charge_standoff(life.mission.rack)
  peer = _peer(sx + 0.2, sy, "SWAP_PICK")
  life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
  _clock(life)
  spins = []
  life.mission._spin_routine = lambda *a, **kw: (spins.append(1), tick.result(None))[1]
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  assert life.mission.run(life.go_charge_routine()) is False
  assert life.data.time == pytest.approx(3 * lc.CHARGE_OCCUPANCY_S,
                                         abs=lc.BAY_POLL_S)
  assert spins == [], "spun for a bay somebody was standing on"
  said = next(line for line in life.log if "GO_CHARGE: no route" in line)
  assert "Rowan was standing 0.20 m from it" in said, said
  assert "I waited 1386 s for it to leave, 3x how long one charge" in said

  # ...and the approach asks the same hook before it looks
  m = life.mission
  asked = []
  m.bay_wait = lambda *a: (asked.append(a[2]), tick.result(False))[1]
  m.swap._run_routine = lambda *a, **kw: tick.result(None)
  assert m.run(m.charge_approach_routine(0.5, 0.05)) == "peer-at-bay"
  assert asked == ["charge"]


# ---- rule 3: a refused return is retried first ------------------------------


def test_a_refused_return_is_retried_before_the_next_errand(monkeypatch):
  """Luca, three times: a stow refused, the next drawing "picked" the pen
  already on its fork, the drive gave up, the stow was refused again. A
  robot never starts a job with a tool it failed to hang still on its
  fork -- and the retries are bounded, or a stow that misses for a
  mechanical reason would starve the loop (and its mind) for good."""
  from pluggybot.procedure import steps
  cfg = lc.world_config("room_hub")
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
    life.mission._spin_routine = lambda *a, **kw: tick.result(None)
    life.errands = [SimpleNamespace(name="draw:x", module="module_pen")]
    life._afford_next = lambda: True
    life.run(cfg["start"], max_sim_time=60.0)
    assert order == (["stow", "errand"] if clears
                     else ["stow"] * lc.STOW_RETRIES + ["errand"]), order


# ---- the evidence: what a failed charge approach saw ------------------------


def test_the_charge_trace_names_what_stands_between_the_camera_and_the_tag():
  """Rule 4's instrument: six "no-tag" charges on the deployed pair left
  nothing to read. The trace says, per look, whether the tag fixed, how
  far off the standoff the robot stood, the peer's distance, and what the
  camera's line to the tag met first -- here, the other robot's body."""
  from pluggybot.pair import build_pair
  from pluggybot.mission.mission import charge_trace
  me, peer = build_pair("room_hub", errands=("none", "none"))
  m, d = me.model, me.data
  sx, sy, hd = charge_standoff(me.mission.rack_prior)

  def place(handle, x, y, yaw):
    adr = handle.qpos_adr(m)
    d.qpos[adr:adr + 2] = [x + 0.08 * math.cos(yaw), y + 0.08 * math.sin(yaw)]
    d.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

  place(FIRST, sx, sy, hd)
  place(SECOND, 5.5, 5.5, 0.0)
  mujoco.mj_forward(m, d)
  assert me.mission.line_of_sight("dock_eye", "rack_charge_tag") == "clear"
  back = 0.6                                  # 0.6 m out, square in the lane
  place(FIRST, sx + back * math.cos(hd + math.pi), sy + back * math.sin(hd + math.pi), hd)
  place(SECOND, sx + 0.2 * math.cos(hd + math.pi), sy + 0.2 * math.sin(hd + math.pi), hd)
  mujoco.mj_forward(m, d)
  sight = me.mission.line_of_sight("dock_eye", "rack_charge_tag")
  assert sight.startswith(SECOND.prefix), sight
  me.mission.last_charge = {"attempts": []}
  me.mission.others = [lambda: (sx, sy)]
  me.mission._charge_look(None)
  line = charge_trace(me.mission.last_charge)
  assert line.startswith("#1 no tag") and "peer 0.0 m from it" in line
  assert f"sight {SECOND.prefix}" in line


# ---- found reviewing the PR ---------------------------------------------------


def test_a_clear_that_failed_is_not_driven_again_from_the_same_place():
  """`_leave_rack_routine` runs before EVERY decision: a robot that could
  not get away re-drove every spot (minutes) and wrote a History line per
  pass. From where it gave up, it does not try again until it has moved."""
  life = _life(_peer(9.0, 9.0))
  sx, sy, _ = bay_standoff(STATION, life.mission.rack)
  _at(life, sx, sy)
  life.mission.grid.grid[:] = -5.0
  drives = []
  life.mission.drive_to_routine = lambda *a, **kw: (drives.append(a[:2]),
                                                    tick.result(False))[1]
  life.mission.run(life._leave_rack_routine())
  tried = len(drives)
  assert tried == 1 + lc.CLEAR_SPOTS
  life.mission.run(life._leave_rack_routine())
  assert len(drives) == tried, "drove the same failed clear again"
  _at(life, sx + 0.5, sy)                      # it went somewhere, and came back
  life.mission.run(life._leave_rack_routine())
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
  sx, sy, _ = bay_standoff(STATION, life.mission.rack)
  life.mission.others = [lambda: (sx + 0.2, sy)]    # routed round, unnamed
  _clock(life)
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(False)
  assert life.mission.run(life._await_bay_routine(sx, sy, "pick", 0.0)) is False
  assert life.data.time == pytest.approx(3 * lc.SWAP_OCCUPANCY_S,
                                         abs=lc.BAY_POLL_S)
  assert "another robot" in life.thoughts.read("History.md")


def test_the_charge_approach_faces_the_rack_before_it_looks_after_a_wait():
  """Back from the waiting spot the robot arrives at whatever heading the
  drive left it, and a look from there is the "no-tag" this exists to end."""
  life = _life()
  m = life.mission
  sx, sy, hd = charge_standoff(m.rack)
  m.others = [lambda: (sx + 0.2, sy)]
  calls = []
  m.bay_wait = lambda *a: (calls.append("wait"), tick.result(True))[1]
  m.drive_to_routine = lambda *a, **kw: (calls.append("drive"), tick.result(True))[1]
  m.face_routine = lambda h: (calls.append(("face", round(h, 3))), tick.result(True))[1]
  m.swap._run_routine = lambda *a, **kw: tick.result(None)

  class Looked(Exception):
    pass

  def look():
    calls.append("look")
    raise Looked

  m.charge_bay_fix = look
  with pytest.raises(Looked):
    m.run(m.charge_approach_routine(0.5, 0.05))
  assert calls == ["wait", "drive", ("face", round(hd, 3)), "look"], calls
