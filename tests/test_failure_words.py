"""A failed job says what failed (issue #350).

Most of the deployed robots' open tickets, and the belief that kept both
from the lab, came from words that named the wrong failure: "no ink reached
whiteboard_b" fourteen times in a day for a pen that never got there, "the
feed plate was never pressed" for a route that gave up in the living room,
"no count came back from None", "stopped 9.1 m short" read as the pack. One
fast test per rule, each stubbing what is not under test; nothing flies.
"""

from types import SimpleNamespace

import pytest

from pluggybot import tick
from pluggybot.economy import scoring
from pluggybot.lifecycle import HubLifecycle
from pluggybot.mission.errand import Errand
from pluggybot.mission.mission import DRIVE_GAVE_UP, HubMission, KeepClear, gave_up
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.robot import SECOND

from test_overseer import _lifecycle  # noqa: I001 -- tests/ is on sys.path

STATION = HUB_STATION_YS[2]
STEPPER = SimpleNamespace(_step_once=lambda *a: None)


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.read("History.md").splitlines() if ln.strip()]


def _stalled_at(life, seconds=60.0, short=3.0):
  """A drive stub that gives up the way the real one records it."""
  def drive(x, y, timeout=90.0):
    life.mission.last_drive = {"why": "stalled", "goal": (x, y),
                               "seconds": seconds, "shortM": short}
    return tick.result(False)
  return drive


def _pen_errand():
  return Errand(name="draw:whiteboard_a", module="module_pen", station_y=STATION,
                use_at=(1.0, 1.0), use=lambda _l: {"drew": True},
                detail={"board": "whiteboard_a"})


# ---- 1. the errand's own failure comes first ---------------------------------


def test_a_refused_pick_leads_the_verdict_and_is_said_once():
  """Five of the fourteen "no ink reached whiteboard_b" were a pick refused
  at the rack. The verdict leads with the pick; the grade is still the
  board's (no ink, 0 points), and History says it once, not twice."""
  life = _lifecycle("room_hub", errand=False)
  life.mission.swap_at_bay_routine = lambda *a, **kw: tick.result(None)
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": False, "hung": False}
  result = life.run_errand(_pen_errand())
  reason = result["verdict"]["reason"]
  assert reason.startswith("could not pick up module_pen: it was not on its bay"), reason
  assert reason.endswith("no ink reached whiteboard_a"), "the measured grade still follows"
  assert result["verdict"]["ok"] is False and result["verdict"]["points"] == 0
  said = [ln for ln in _history(life) if "could not pick up" in ln]
  assert len(said) == 1 and "draw: could not pick up module_pen" in said[0], said


def test_a_drive_that_never_arrived_leads_the_verdict_with_why():
  """Six of the fourteen were "USE_TOOL: never got there" -- no History line
  and no reason. The narration and the verdict now carry both."""
  life = _lifecycle("room_hub", errand=False)
  life.mission.swap_at_bay_routine = lambda *a, **kw: tick.result(None)
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True, "hung": True}
  life.mission.drive_to_routine = _stalled_at(life)
  result = life.run_errand(_pen_errand())
  why = "the drive gave up 3.0 m short after 60 s (stalled, no progress for 10 s)"
  assert result["verdict"]["reason"] == (f"never reached whiteboard_a: {why} -- "
                                         "no ink reached whiteboard_a")
  assert any(f"USE_TOOL: never got there -- {why}" in ln for ln in life.log)
  assert sum(why in ln for ln in _history(life)) == 1


def test_a_failure_a_passed_verdict_does_not_carry_is_still_in_history():
  """A carry whose drive fell short and whose tool went home was done, and
  its verdict is left alone -- so History says what failed on its own."""
  life = _lifecycle("room_hub", errand=False)
  life.mission.swap_at_bay_routine = lambda *a, **kw: tick.result(None)
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True, "hung": True}
  life.mission.drive_to_routine = _stalled_at(life)
  errand = Errand(name="carry", module="module_lcd", station_y=HUB_STATION_YS[0],
                  use_at=(1.0, 1.0), use=lambda _l: {}, task="carry")
  result = life.run_errand(errand)
  assert result["verdict"]["ok"] and "never reached" not in result["verdict"]["reason"]
  assert any("carry: never reached the use pose: the drive gave up" in ln
             for ln in _history(life))


def test_the_lead_moves_no_point_and_leaves_a_pass_alone():
  failed = "never reached the cage: the drive gave up"
  v = scoring.evaluate("feed", {"feeds": 0, "feedsBefore": 0}, failed=failed)
  assert v.reason == f"{failed} -- the feed plate was never pressed" and v.points == 0
  bare = scoring.evaluate("feed", {"feeds": 0, "feedsBefore": 0})
  assert (v.ok, v.points, v.metrics) == (bare.ok, bare.points, bare.metrics)
  won = scoring.evaluate("feed", {"feeds": 1, "feedsBefore": 0}, failed=failed)
  assert won.ok and not won.reason.startswith(failed)


def test_a_board_it_never_squared_up_to_leads_with_which_of_the_two():
  """The use-phase's own word that its work never began leads the same way."""
  life = _lifecycle("room_hub", errand=False)
  life.mission.swap_at_bay_routine = lambda *a, **kw: tick.result(None)
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True, "hung": True}
  life.mission.drive_to_routine = lambda *a, **kw: tick.result(True)
  errand = _pen_errand()
  errand.use = lambda _l: {"error": "never squared up to the board",
                           "failedBefore": "never squared up to whiteboard_a: the "
                                           "turn to face it ran out of time"}
  result = life.run_errand(errand)
  assert result["verdict"]["reason"].startswith(
    "never squared up to whiteboard_a: the turn to face it ran out of time -- ")


# ---- 2. a drive that gives up says why ----------------------------------------


class _Drive:
  """`drive_to_routine`'s own `self`, with the planner and the wheels
  stubbed: every command is 0.1 s, and the robot moves `step` m along x."""
  drive_to_routine = HubMission.drive_to_routine
  _drove = HubMission._drove
  _at_stand_in = HubMission._at_stand_in
  _other_in_the_way = HubMission._other_in_the_way
  _bodies = HubMission._bodies
  peer_on_the_goal = HubMission.peer_on_the_goal

  def __init__(self, plan, step=0.0, others=(), sighting=None, cut=False):
    self.data = SimpleNamespace(time=0.0)
    self.grid = SimpleNamespace(resolution=0.05)
    self._cut = cut
    self.pose = (0.0, 0.0, 0.0)
    self.others = list(others)
    self.backoff_until, self.peer_holds = 0.0, 0
    self.swap = SimpleNamespace(pressing=False)
    self.last_drive, self._stand_in = None, None
    self._plan, self._step, self._sighting = plan, step, sighting

  def _plan_to(self, wx, wy):
    return self._plan(self)

  def _route_cut_by_others(self, wx, wy):
    return self._cut

  def peer_sighting(self):
    return self._sighting

  def _nav_routine(self, v, w):
    self.data.time += 0.1
    self.pose = (self.pose[0] + self._step, self.pose[1], 0.0)
    yield v, w

  def _drive_routine(self, seconds, v, w):
    self.data.time += seconds
    yield v, w


def _drive(fake, goal=(5.0, 0.0), timeout=60.0):
  arrived = tick.run(STEPPER, fake.drive_to_routine(*goal, timeout))
  return arrived, fake.last_drive


def _stand_in(fake):
  fake._stand_in = (0.2, 0.0)
  return [(0.2, 0.0)]


CAUSES = {
  # the planner found nothing at all
  "no plan": (_Drive(lambda f: None), "no_route"),
  # ...or only a stand-in, and the robot stood at it
  "stand-in": (_Drive(_stand_in), "no_route"),
  # a plan it could not make progress along
  "stalled": (_Drive(lambda f: [(5.0, 0.0)]), "stalled"),
  # the other robot's disc is what cut the route
  "disc": (_Drive(lambda f: None, others=[lambda: (5.0, 0.3)], cut=True), "peer"),
  # waited for the other robot beside it until the time ran out
  "waited": (_Drive(lambda f: [(5.0, 0.0)], others=[lambda: (0.5, 0.0)]), "peer"),
  # held for a body the depth camera saw in the corridor
  "held": (_Drive(lambda f: [(5.0, 0.0)], sighting=0.3), "peer"),
  # a robot LYING on the goal: never waited for (#365), and its disc made
  # the goal a stand-in -- "no route" until the review
  "lying": (_Drive(_stand_in, others=[lambda: KeepClear(5.0, 0.1, True)]), "peer"),
  # still making progress when the budget ran out
  "clock": (_Drive(lambda f: [(100.0, 0.0)], step=0.02), "timeout"),
}


@pytest.mark.parametrize("case", list(CAUSES))
def test_each_way_a_drive_gives_up_is_recorded_as_itself(case):
  fake, cause = CAUSES[case]
  arrived, rec = _drive(fake, goal=(100.0, 0.0) if case == "clock" else (5.0, 0.0),
                        timeout=5.0 if case == "clock" else 60.0)
  assert arrived is False and rec["why"] == cause, rec
  assert rec["why"] in DRIVE_GAVE_UP


def test_the_four_causes_read_differently_and_an_arrival_reads_none():
  said = {cause: gave_up(_drive(CAUSES[case][0], goal=(100.0, 0.0) if case == "clock"
                                else (5.0, 0.0), timeout=5.0 if case == "clock" else 60.0)[1])
          for case, (_, cause) in CAUSES.items() if case in ("no plan", "stalled",
                                                              "disc", "clock")}
  assert "no route over the floor mapped so far" in said["no_route"]
  assert "stalled, no progress for 10 s" in said["stalled"]
  assert "the other robot standing 0.3 m from where it was going" in said["peer"]
  assert "out of time" in said["timeout"]
  assert len({s[s.index("("):] for s in said.values()}) == 4
  near = _Drive(lambda f: [])
  near.pose = (4.95, 0.0, 0.0)
  arrived, rec = _drive(near)
  assert arrived is True and rec["why"] == ""


def test_a_robot_lying_on_the_goal_is_said_to_lie_there():
  _, rec = _drive(CAUSES["lying"][0])
  assert (rec["peerAt"], rec["peerM"], rec["peerDown"]) == ("goal", 0.1, True)
  assert gave_up(rec, "Rowan").endswith("(Rowan lying 0.1 m from where it was going)")
  beside = {**rec, "peerAt": "here", "peerM": 0.4}
  assert gave_up(beside, "Rowan").endswith("(Rowan lying in the way, 0.4 m off)")


def test_a_stand_in_from_an_earlier_drive_is_not_this_drives():
  """`_stand_in` is the last PLAN's, and a drive starts without one."""
  fake = _Drive(lambda f: [(5.0, 0.0)])           # plans, and no stand-in
  fake._stand_in = (0.1, 0.0)                     # ...one left by a drive before
  assert _drive(fake)[1]["why"] == "stalled"


def test_the_cut_route_check_answers_as_a_plan_would_without_the_other_robot():
  """`_route_cut_by_others` labels the floor `_plan_to` last used instead of
  planning again (a plan across the home loop is ~1.3 s on the physics
  thread): held here to a REAL plan with the other robot left out, on a
  wall with a door the other robot stands in, and one with no door."""
  from pluggybot.lifecycle import world_config
  life = _lifecycle("room_hub", errand=False)
  m = life.mission
  m.start_at(*world_config("room_hub")["start"])          # (0.5, 3.0)
  door = KeepClear(2.0, 3.0)
  for gap in (True, False):
    m.grid.grid[:] = -5.0                                   # mapped and free...
    wall = m.grid.world_to_cell(2.0, 0.0)[0]
    lo, hi = m.grid.world_to_cell(0, 2.5)[1], m.grid.world_to_cell(0, 3.5)[1]
    m.grid.grid[:, wall] = 5.0                              # ...but a wall at x = 2
    if gap:
      m.grid.grid[lo:hi, wall] = -5.0                       # with a 1 m door in it
    for goal in [(4.0, 3.0), (2.0, 5.0), (1.0, 1.0)]:     # beyond, in the wall, this side
      m.others = [lambda: door]
      masked = m._plan_to(*goal)
      cut = m._route_cut_by_others(*goal)
      m.others = []
      alone = m._plan_to(*goal)
      assert cut == (alone is not None), (gap, goal)
      if goal == (4.0, 3.0):
        assert masked is None and cut is gap, "the door is the other robot's"


def test_the_lifecycle_names_the_other_robot_and_reads_no_stale_record():
  rec = {"why": "peer", "goal": (5.0, 0.0), "seconds": 60.0, "shortM": 4.2,
         "peerAt": "goal", "peerM": 0.3, "peerXY": (5.0, 0.3), "peerDown": False}
  rowan = SimpleNamespace(robot_name="Rowan", root=SECOND.root,
                          mission=SimpleNamespace(pose_xy=lambda: (5.0, 0.3)))
  me = SimpleNamespace(peers=[rowan], mission=SimpleNamespace(last_drive=rec))
  me._nearest_peer = lambda *a: HubLifecycle._nearest_peer(me, *a)
  assert HubLifecycle.drive_why(me, 5.0, 0.0) == (
    "the drive gave up 4.2 m short after 60 s (Rowan standing 0.3 m from where it "
    "was going)")
  # a record of a drive somewhere else is not this drive's reason
  assert HubLifecycle.drive_why(me, 9.0, 9.0) == "the drive gave up"


def test_a_charge_trip_that_never_arrived_says_why_and_so_does_the_death():
  """ "GO_CHARGE: no route to the charge bay" was said for every failed
  drive, and the `stuck` death that followed carried nothing."""
  life = _lifecycle("room_hub", errand=False)
  life.mission.drive_to_routine = _stalled_at(life, seconds=90.0, short=2.0)
  life.mission._spin_routine = lambda *a, **kw: tick.result(None)
  assert life.mission.run(life.go_charge_routine()) is False
  why = "the drive gave up 2.0 m short after 90 s (stalled, no progress for 10 s)"
  assert any(f"GO_CHARGE: never reached the charge bay: {why}" in ln
             for ln in life.log), life.log[-3:]
  life._strand()
  assert life.dead["why"].endswith(f": never reached the charge bay: {why}")


def test_a_charge_bay_waited_for_and_given_up_reads_no_older_drive():
  """A wait that gave up drove nowhere: the record of an earlier drive to
  the same standoff is not this trip's reason."""
  from test_rack_contention import _clock, _peer
  from pluggybot.mission.mission import charge_standoff
  life = _lifecycle("room_hub", errand=False)
  sx, sy, _ = charge_standoff(life.mission.rack)
  life.mission.last_drive = {"why": "stalled", "goal": (sx, sy), "seconds": 90.0,
                             "shortM": 2.0}                      # a drive long ago
  peer = _peer(sx + 0.2, sy, "CHARGE")
  life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
  _clock(life)
  life.mission.drive_to_routine = lambda *a, **kw: pytest.fail("the wait drove")
  assert life.mission.run(life.go_charge_routine()) is False
  assert life.charge_failure == "never reached the charge bay"
  assert any("GO_CHARGE: never reached the charge bay -- Rowan was standing" in ln
             for ln in life.log), life.log[-2:]


def test_a_charge_trip_that_ends_in_a_wait_is_the_waits_not_an_earlier_drives():
  """Drive, fail, find the bay taken, wait, give up: the trip ended in the
  wait, and the drive before it is not what the death should name."""
  from test_rack_contention import _clock, _peer
  from pluggybot.mission.mission import charge_standoff
  life = _lifecycle("room_hub", errand=False)
  sx, sy, _ = charge_standoff(life.mission.rack)
  peer = _peer(sx + 3.0, sy, "CHARGE")
  life.peers, life.mission.others = [peer], [peer.mission.pose_xy]
  _clock(life)
  stalled = _stalled_at(life, seconds=90.0, short=2.0)

  def drive(x, y, timeout=90.0):
    peer.pos[:] = [x + 0.2, y]                 # ...and it takes the bay meanwhile
    return stalled(x, y, timeout)
  life.mission.drive_to_routine = drive
  life.mission._spin_routine = lambda *a, **kw: tick.result(None)
  assert life.mission.run(life.go_charge_routine()) is False
  assert life.charge_failure == "never reached the charge bay"
  assert any("GO_CHARGE: never reached the charge bay -- Rowan was standing" in ln
             for ln in life.log), life.log[-2:]


# ---- 3. narration names only what happened -------------------------------------


def test_an_act_whose_route_gave_up_never_reached_the_cage():
  """Three of the lab's "nothing registered" / "never pressed" were the
  route's first leg giving up beside the rack."""
  run = {"failedAt": 0, "steps": [{"i": 0, "verb": "drive_to", "ok": False,
                                   "why": "the drive gave up 1.0 m short after 180 s "
                                          "(stalled, no progress for 10 s)"}]}
  errand = SimpleNamespace(detail={"cage": "lab", "routeLegs": 5})
  # ...or its budget or an interrupt stopped it on the way
  cut = {"completed": 2, "stopped": "interrupted"}
  assert HubLifecycle._program_failure(errand, {"procedure": cut}) == (
    "never reached the cage: it was interrupted, after 2 of the 5 legs of the way there")
  assert HubLifecycle._program_failure(
    errand, {"procedure": {**cut, "completed": 5}}) == "", "stopped in the lab"
  assert HubLifecycle._program_failure(errand, {"procedure": run}) == (
    "never reached the cage: the drive gave up 1.0 m short after 180 s (stalled, no "
    "progress for 10 s), on leg 1 of 5 of the way there")
  # ...and a failure IN the lab -- the pass over the plate -- is not one
  in_lab = {**run, "failedAt": 5, "steps": [{**run["steps"][0], "i": 5}]}
  assert HubLifecycle._program_failure(errand, {"procedure": in_lab}) == ""
  assert HubLifecycle._program_failure(SimpleNamespace(detail={}), {"procedure": run}) == ""


def test_the_care_line_says_the_route_and_not_the_mouse(tmp_path):
  from test_mouse import _life as _lab_life
  from pluggybot.mind import overseer as ov
  from pluggybot.lifecycle import errand_from
  import mujoco
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  life = _lab_life(model, tmp_path)
  errand = errand_from(ov.Decision(action="care", care="company", reason=""), "home",
                       from_xy=life.mission.pose_xy())
  assert errand.detail["routeLegs"] == 5, "from the rack, the whole route"
  failed = "never reached the cage: the drive gave up (why), on leg 1 of 5 of the way there"
  life._cage_record(errand, {"procedure": {"ok": False}}, None,
                    scoring.cage_before(life, errand), failed)
  assert life.status.endswith(f"set off to company for the mouse and {failed}")
  assert "nothing registered" not in life.thoughts.read("History.md")


def test_a_job_on_the_cage_that_never_got_there_is_said_once(tmp_path):
  """The paid job's verdict leads with the route's failure, so its cage
  line is narrated and not written to History a second time."""
  from test_mouse import _life as _lab_life, _shock_errand
  import mujoco
  model = mujoco.MjModel.from_xml_path("models/home_world.xml")
  life = _lab_life(model, tmp_path)
  _, errand = _shock_errand(life)
  failed = "never reached the cage: the drive gave up (why), on leg 1 of 5 of the way there"
  before = scoring.cage_before(life, errand)
  result = {"procedure": {"ok": False}, "points": 0}
  verdict = scoring.evaluate("shock", scoring.sample_shock(life, errand, result, before),
                             failed=failed)
  life._bank(verdict)
  life._cage_record(errand, result, verdict, before, failed)
  assert sum(failed in ln for ln in _history(life)) == 1
  assert life.status.endswith(f"and {failed}"), "...and the narration still says it"


def test_a_census_that_never_ran_names_its_zone_and_never_none():
  errand = Errand(name="census:garden", module="module_lcd", station_y=HUB_STATION_YS[0],
                  use_at=(1.0, 1.0), task="census", needs_use_pose=False,
                  detail={"zone": "garden"})
  measured = scoring.sample_census(None, errand, {"errand": "census:garden"}, {})
  assert scoring.evaluate("census", measured).reason == "no count came back from garden"
  assert scoring.evaluate("census", {}).reason == "no count came back"


# ---- 4. `mapDone` is what it means ----------------------------------------------


def test_the_context_says_the_floor_is_explored_not_that_a_map_is_done():
  """Luca read `mapDone: false` as its event map not landing and filed
  tk_0001. It meant: exploring the floor has finished."""
  from pluggybot.mind import overseer as ov
  life = _lifecycle("room_hub", errand=False)
  life.floor_explored = True
  state = ov.context_for(life)
  assert state["floorExplored"] is True and "mapDone" not in state
  from pluggybot.lifecycle import board_book
  menu = ov.Menu.for_world("home", board_book("home"))
  done = {"tasksThisMission": ["draw", "census", "dance", "carry"], "decisions": 3}
  assert ov.scripted(menu, {**done, "floorExplored": False}, "test").action == "explore"
  assert ov.scripted(menu, {**done, "floorExplored": True}, "test").action != "explore"


def test_a_continuation_saved_before_the_rename_still_says_the_floor_is_explored(tmp_path):
  from test_continuation import _life as _kept_life, _prelude, _saved
  life = _kept_life(tmp_path)
  life.floor_explored = True
  snap = _saved(life, tmp_path)
  state = snap.meta["robots"][life.root]
  state["mapDone"] = state.pop("floorExplored")          # what 42f4a11 wrote
  back = _kept_life(tmp_path)
  _prelude(back, snap)
  assert back.floor_explored
