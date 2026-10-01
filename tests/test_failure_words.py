"""A failed job says what failed (issue #350).

Most of the deployed robots' open tickets, and the belief that kept both
from the lab, came from words that named the wrong failure: "no ink reached
whiteboard_b" fourteen times in a day for a pen that never got there, "the
feed plate was never pressed" for a route that gave up in the living room,
"no count came back from None", "stopped 9.1 m short" read as the pack. One
fast test per rule, each stubbing what is not under test; nothing flies.
"""

import math
import re
from types import SimpleNamespace

import pytest

from pluggybot import tick
from pluggybot.body import KeepClear, StubBody
from pluggybot.economy import scoring
from pluggybot.legs.body import QuadMission
from pluggybot.lifecycle import NEAR_STANDOFF_M, QUAD_HOME, HubLifecycle, board_book, world_config
from pluggybot.mission.errand import Errand
from pluggybot.navigator import DRIVE_GAVE_UP, gave_up
from pluggybot.procedure.steps import TOOL_BAYS
from pluggybot.rack.coupling import STATION_YS
from pluggybot.robot import SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

STATION = STATION_YS[TOOL_BAYS["module_pen"]]
STEPPER = SimpleNamespace(step=lambda *a: None)


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.read("History.md").splitlines() if ln.strip()]


def _life(**kw):
  """The loop on a stub body that says why a drive gave up in the served
  body's own words (`navigator.gave_up`), with the house's boards."""
  cfg = world_config(QUAD_HOME)
  body = StubBody(rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  body.gave_up = gave_up
  return stub_life(body=body, boards=board_book(QUAD_HOME), **kw)


def _stalled_at(life, seconds=60.0, short=3.0):
  """A drive stub that gives up the way the real one records it."""
  def drive(x, y, timeout=90.0, stop=None):
    life.body.last_drive = {"why": "stalled", "goal": (x, y),
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
  life = _life()
  life.body.fetch_tool_routine = lambda *a, **kw: tick.result("arrived")
  life.body.module_state = lambda *a, **kw: {"on_fork": False, "hung": False}
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
  life = _life()
  life.body.go_to_routine = _stalled_at(life)
  result = life.run_errand(_pen_errand())
  why = "the drive gave up 3.0 m short after 60 s (stalled, no progress for 10 s)"
  assert result["verdict"]["reason"] == (f"never reached whiteboard_a: {why} -- "
                                         "no ink reached whiteboard_a")
  assert any(f"USE_TOOL: never got there -- {why}" in ln for ln in life.log)
  assert sum(why in ln for ln in _history(life)) == 1


def test_a_failure_a_passed_verdict_does_not_carry_is_still_in_history():
  """A carry whose drive fell short and whose tool went home was done, and
  its verdict is left alone -- so History says what failed on its own."""
  life = _life()
  life.body.go_to_routine = _stalled_at(life)
  errand = Errand(name="carry", module="module_lcd", station_y=STATION_YS[0],
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
  life = _life()
  errand = _pen_errand()
  errand.use = lambda _l: {"error": "never squared up to the board",
                           "failedBefore": "never squared up to whiteboard_a: the "
                                           "turn to face it ran out of time"}
  result = life.run_errand(errand)
  assert result["verdict"]["reason"].startswith(
    "never squared up to whiteboard_a: the turn to face it ran out of time -- ")


# ---- 2. a drive that gives up says why ----------------------------------------


class _Drive:
  """`drive_to_routine`'s own `self` -- the served body's (`QuadMission`,
  whose drive is the `Navigator`'s) -- with the planner and the legs
  stubbed: every command is 0.1 s, and the robot moves `step` m along x."""
  drive_to_routine = QuadMission.drive_to_routine
  _drove = QuadMission._drove
  _at_stand_in = QuadMission._at_stand_in
  _other_in_the_way = QuadMission._other_in_the_way
  _bodies = QuadMission._bodies
  peer_on_the_goal = QuadMission.peer_on_the_goal
  _cells = QuadMission._cells
  _left = QuadMission._left
  _known_cells = QuadMission._known_cells
  PROGRESS_ALONG_ROUTE = QuadMission.PROGRESS_ALONG_ROUTE
  PROGRESS_MAP_GROWTH = QuadMission.PROGRESS_MAP_GROWTH
  NEW_ROUTE_M = QuadMission.NEW_ROUTE_M
  # ...its measured sizes, which the drive reads off its body
  BACKOFF_V, PEER_CLEARANCE_M = QuadMission.BACKOFF_V, QuadMission.PEER_CLEARANCE_M
  OTHER_ROBOT_CELLS = QuadMission.OTHER_ROBOT_CELLS
  DOWN_ROBOT_CELLS = QuadMission.DOWN_ROBOT_CELLS
  pressing = False

  def __init__(self, plan, step=0.0, others=(), sighting=None, cut=False):
    import numpy as np
    self.data = SimpleNamespace(time=0.0)
    self.grid = SimpleNamespace(resolution=0.05, grid=np.zeros((4, 4)))
    self._cut = cut
    self.pose = (0.0, 0.0, 0.0)
    self.others = list(others)
    self.backoff_until, self.peer_holds = 0.0, 0
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
  import mujoco

  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  model = lw.home_spec().compile()
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  m = body.mission
  m.start_at(0.5, 3.0, 0.0)
  door = KeepClear(2.0, 3.0)
  for gap in (True, False):
    m.grid.grid[:] = -5.0                                   # mapped and free...
    wall = m.grid.world_to_cell(2.0, 0.0)[0]
    lo, hi = m.grid.world_to_cell(0, 2.5)[1], m.grid.world_to_cell(0, 3.5)[1]
    m.grid.grid[:, wall] = 5.0                              # ...but a wall at x = 2
    if gap:
      m.grid.grid[lo:hi, wall] = -5.0                       # with a 1 m door in it
    for goal in [(4.0, 3.0), (2.0, 5.0), (1.0, 1.0)]:     # beyond, in the wall, this side
      m.others, m._plan_memo = [lambda: door], None
      masked = m._plan_to(*goal)
      cut = m._route_cut_by_others(*goal)
      m.others, m._plan_memo = [], None
      alone = m._plan_to(*goal)
      assert cut == (alone is not None), (gap, goal)
      if goal == (4.0, 3.0):
        assert masked is None and cut is gap, "the door is the other robot's"
  body.close()


def test_the_lifecycle_names_the_other_robot_and_reads_no_stale_record():
  rec = {"why": "peer", "goal": (5.0, 0.0), "seconds": 60.0, "shortM": 4.2,
         "peerAt": "goal", "peerM": 0.3, "peerXY": (5.0, 0.3), "peerDown": False}
  rowan = SimpleNamespace(robot_name="Rowan", root=SECOND.root,
                          body=SimpleNamespace(pose_xy=lambda: (5.0, 0.3)))
  me = SimpleNamespace(peers=[rowan], body=SimpleNamespace(last_drive=rec, gave_up=gave_up))
  me._nearest_peer = lambda *a: HubLifecycle._nearest_peer(me, *a)
  assert HubLifecycle.drive_why(me, 5.0, 0.0) == (
    "the drive gave up 4.2 m short after 60 s (Rowan standing 0.3 m from where it "
    "was going)")
  # a record of a drive somewhere else is not this drive's reason
  assert HubLifecycle.drive_why(me, 9.0, 9.0) == "the drive gave up"


def test_a_charge_trip_that_never_arrived_says_why_and_so_does_the_death():
  """ "GO_CHARGE: no route to the charge bay" was said for every failed
  drive, and the `stuck` death that followed carried nothing."""
  life = _life()
  life.body.go_to_routine = _stalled_at(life, seconds=90.0, short=2.0)
  assert life.body.run(life.go_charge_routine()) is False
  why = "the drive gave up 2.0 m short after 90 s (stalled, no progress for 10 s)"
  assert any(f"GO_CHARGE: never reached the charge bay: {why}" in ln
             for ln in life.log), life.log[-3:]
  life._strand()
  assert life.dead["why"].endswith(f": never reached the charge bay: {why}")


def test_a_walk_home_that_gives_up_near_the_standoff_goes_on_to_the_board():
  """Home from a long explore 0.9 m out, three robots of eight believed
  themselves 0.2-0.3 m short of the standoff, pressed the couch until the
  walk gave up, and the charge ended there: the board, 2.5 m off, was never
  looked for (issue #422). Near enough, the approach goes on and the board
  decides; farther, or the other robot the cause, the walk's retries and
  its failure stand."""
  def trip(short, why="stalled"):
    life = _life()
    sx, sy, _ = life.body.charge_standoff()
    life.body.x, life.body.y = sx + short, sy

    def drive(x, y, timeout=90.0, stop=None):
      life.body.last_drive = {"why": why, "goal": (x, y), "seconds": 12.0, "shortM": short}
      return tick.result(False)

    looks = []
    life.body.go_to_routine = drive
    life.body.look_around_routine = lambda: (looks.append(1), tick.result(None))[1]
    return life.body.run(life.go_charge_routine()), life.body.on_charger, len(looks)

  assert trip(0.2) == (True, True, 0)
  assert trip(NEAR_STANDOFF_M + 0.5) == (False, False, 2)
  assert trip(0.2, why="peer") == (False, False, 2)


def _peer(x: float, y: float, state: str = "IDLE"):
  """The other robot's PUBLIC surface and no more: a name, a state, a
  reported pose -- mutable, so a test can walk it away -- standing, and a
  fork with nothing on it."""
  pos = [x, y]
  return SimpleNamespace(robot_name="Rowan", root=SECOND.root, state=state,
                         pos=pos, down=lambda: False, body=SimpleNamespace(
                           pose_xy=lambda: (pos[0], pos[1]),
                           module_state=lambda t: {"on_fork": False, "hung": True}))


def _taken_by(life, peer, within: float = 0.45):
  """The other robot on the goal is the body's to say (`peer_on_the_goal`):
  here, within `within` of it; and standing still advances the clock and
  does nothing else."""
  def on_goal(x, y):
    d = math.hypot(peer.pos[0] - x, peer.pos[1] - y)
    return d if d < within else None

  def still(seconds):
    life.data.time += seconds
    return tick.result(None)
  life.peers = [peer]
  life.body.peer_on_the_goal = on_goal
  life.body.hold_routine = still


def test_a_charge_bay_waited_for_and_given_up_reads_no_older_drive():
  """A wait that gave up drove nowhere: the record of an earlier drive to
  the same standoff is not this trip's reason."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  life.body.last_drive = {"why": "stalled", "goal": (sx, sy), "seconds": 90.0,
                          "shortM": 2.0}                           # a drive long ago
  _taken_by(life, _peer(sx + 0.2, sy, "CHARGE"))

  def drive(x, y, timeout=90.0, stop=None):
    if math.hypot(x - sx, y - sy) < 1e-6:
      pytest.fail("the wait drove to the bay")
    return tick.result(True)                     # ...to where it waits, at most
  life.body.go_to_routine = drive
  assert life.body.run(life.go_charge_routine()) is False
  _held_by_rowan(life)
  # ...and who held it reaches a stranded robot's death, not only the log
  # (issue #424: the narration was the one place it was kept)
  life._strand()
  assert life.dead["why"].endswith(f": {life.charge_failure}")


def _held_by_rowan(life):
  """The reason is the wait's, whole -- who held the bay -- and names no
  drive; the narration says that same sentence (issue #424)."""
  assert life.charge_failure.startswith("never reached the charge bay -- Rowan was "
                                        "standing"), life.charge_failure
  assert "the drive gave up" not in life.charge_failure
  assert any(ln.endswith(f"GO_CHARGE: {life.charge_failure}") for ln in life.log), \
    life.log[-2:]


def test_a_charge_trip_that_ends_in_a_wait_is_the_waits_not_an_earlier_drives():
  """Drive, fail, find the bay taken, wait, give up: the trip ended in the
  wait, and the drive before it is not what the death should name."""
  life = _life()
  sx, sy, _ = life.body.charge_standoff()
  peer = _peer(sx + 3.0, sy, "CHARGE")
  _taken_by(life, peer)
  stalled = _stalled_at(life, seconds=90.0, short=2.0)

  def drive(x, y, timeout=90.0, stop=None):
    if math.hypot(x - sx, y - sy) < 1e-6:
      peer.pos[:] = [x + 0.2, y]               # ...and it takes the bay meanwhile
      return stalled(x, y, timeout)
    return tick.result(True)
  life.body.go_to_routine = drive
  assert life.body.run(life.go_charge_routine()) is False
  _held_by_rowan(life)


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


def test_the_care_line_says_the_route_and_not_the_mouse():
  from pluggybot.lifecycle import errand_from
  from pluggybot.mind import overseer as ov
  life = _life()
  errand = errand_from(ov.Decision(action="care", care="toy", reason=""), QUAD_HOME)
  failed = "never reached the cage: the drive gave up (why), on leg 1 of 1 of the way there"
  life._cage_record(errand, {"procedure": {"ok": False}}, None,
                    scoring.cage_before(life, errand), failed)
  assert life.status.endswith(f"set off to toy for the mouse and {failed}")
  assert "nothing registered" not in life.thoughts.read("History.md")


def test_a_job_on_the_cage_that_never_got_there_is_said_once():
  """The paid job's verdict leads with the route's failure, so its cage
  line is narrated and not written to History a second time."""
  from pluggybot.lifecycle import cage_errand
  life = _life()
  errand = cage_errand(QUAD_HOME, "feed", task="feed")
  errand.task_id, errand.detail["predicted"] = "t_0001", "eating"
  failed = "never reached the cage: the drive gave up (why), on leg 1 of 1 of the way there"
  before = scoring.cage_before(life, errand)
  result = {"procedure": {"ok": False}, "points": 0}
  verdict = scoring.evaluate("feed", scoring.sample_feed(life, errand, result, before),
                             failed=failed)
  life._bank(verdict)
  life._cage_record(errand, result, verdict, before, failed)
  assert sum(failed in ln for ln in _history(life)) == 1
  assert life.status.endswith(f"and {failed}"), "...and the narration still says it"


def test_a_census_that_never_ran_names_its_zone_and_never_none():
  errand = Errand(name="census:garden", module="module_lcd", station_y=STATION_YS[0],
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
  life = _life()
  life.floor_explored = True
  state = ov.context_for(life)
  assert state["floorExplored"] is True and "mapDone" not in state
  from dataclasses import replace
  # a body that takes a tool (`Menu.tools`, #406/#407), so the rotation
  # has something besides exploring to turn to
  menu = replace(ov.Menu.for_world(QUAD_HOME, board_book(QUAD_HOME)), tools=True)
  done = {"tasksThisMission": ["draw", "census", "dance", "carry"], "decisions": 3}
  assert ov.scripted(menu, {**done, "floorExplored": False}, "test").action == "explore"
  assert ov.scripted(menu, {**done, "floorExplored": True}, "test").action != "explore"


def test_a_continuation_saved_before_the_rename_still_says_the_floor_is_explored(tmp_path):
  from pluggybot import continuation
  life = stub_life()
  life.floor_explored = True
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  snap = continuation.read(path)
  state = snap.meta["robots"][life.root]
  state["mapDone"] = state.pop("floorExplored")          # what 42f4a11 wrote
  back = stub_life()
  # ...into a body whose map came back with it: the floor is only said to
  # be explored alongside the map it was explored in
  back.body.restore_kept = lambda state, arrays: True
  back.begin(world_config(QUAD_HOME)["start"], max_sim_time=600.0)
  continuation.restore([back], snap)
  assert back.floor_explored


# ---- 5. a decided charge or explore says how it ended (#424) --------------------
# On the live pair in 50 h, 31 decided charges failed and 122 decided explores
# to a zone ended, and History heard of none: the robot read `chose charge`
# and then nothing, and filed tickets saying its actions vanished. Each is a
# day on the stub: what is pinned is where the outcome is WRITTEN.


def _gives_up(body, why="no_route"):
  """A stub drive that gives up the way the real one records it."""
  def drive(x, y, timeout=90.0, stop=None):
    body.last_drive = {"why": why, "goal": (float(x), float(y)),
                       "seconds": 0.0, "shortM": 2.5}
    return tick.result(False)
  return drive


@pytest.mark.parametrize("fails", ["dock", "drive"])
def test_a_decided_charge_that_failed_says_why_in_history(fails):
  from test_body import StubBody, stub_life
  from pluggybot.mind import overseer as ov
  body = StubBody()
  life = stub_life(body=body)
  if fails == "dock":
    body.dock_routine = lambda: tick.result("no board")   # the board never in view
    why = "no charge contact (no board)"
  else:
    body.go_to_routine = _gives_up(body)
    why = "never reached the charge bay: the drive gave up (no_route)"
  life._after_decision(ov.Decision(action="charge", reason="topping up"))
  assert _history(life)[-1].endswith(f"charge: did not charge -- {why}"), _history(life)[-2:]
  assert life.charge_cycles == 0 and not life.verdicts, "no charge, and no verdict for one"


def test_a_decided_explore_whose_zone_walk_gave_up_says_why_and_how_it_ended():
  """26 of the 122 ended at once, `no-reachable`, and a walk to the zone that
  gave up was not even narrated: the robot decided its explores were being
  "dropped pre-run"."""
  from test_body import StubBody, stub_life
  from pluggybot.mind import overseer as ov
  body = StubBody()
  life = stub_life(body=body)
  life.begin((0.0, 0.0, 0.0))                    # the day's setup, and no day
  body.go_to_routine = _gives_up(body)
  body.plan_frontier = lambda blacklist: (None, "no-reachable")
  life._after_decision(ov.Decision(action="explore", zone="lab", reason="the lab"))
  assert _history(life)[-1].endswith(
    "explore (lab): never got there -- the drive gave up (no_route) -- and explored "
    "where it stopped for 0 s, until none of the floor I have not seen could be "
    "reached"), _history(life)[-2:]
  assert any("EXPLORE: never reached lab -- the drive gave up (no_route)" in ln
             for ln in life.log)


def test_every_way_an_explore_ends_is_said_and_the_runs_own_end_is_not():
  """The words are `explore_outcome`'s, a pure function: every ending
  `explore_routine` answers reads as itself, and the run's end writes
  nothing -- the next run carries on (#345)."""
  from pluggybot.lifecycle import EXPLORE_ENDS, explore_outcome
  assert set(EXPLORE_ENDS) == {"budget", "no-frontiers", "no-reachable", "only-near",
                               "blocked", "battery"}
  said = {ended: explore_outcome("", ended, 45.0) for ended in EXPLORE_ENDS}
  assert len(set(said.values())) == len(said)
  assert explore_outcome("lab", "time", 12.0) == ""
  assert explore_outcome("lab", "budget", 45.0) == (
    "explore (lab): got there and explored for 45 s, all the time one explore is given")


def test_a_decided_explore_that_ran_its_time_says_so():
  """83 of the 122 ended `budget spent`, which is the explore working -- and
  said nowhere the robot reads."""
  from test_body import StubBody, stub_life
  from pluggybot.lifecycle import DECIDED_EXPLORE_S
  from pluggybot.mind import overseer as ov
  body = StubBody()
  life = stub_life(body=body)
  life.begin((0.0, 0.0, 0.0))                    # the day's setup, and no day
  body.plan_frontier = lambda blacklist: ([(80, 80)], "ok")    # always somewhere new
  body.go_to_routine = lambda *a, **kw: body.hold_routine(5.0)  # ...5 s away
  life._after_decision(ov.Decision(action="explore", reason="mapping"))
  line = _history(life)[-1]
  assert re.search(r"\] explore: explored for (\d+) s, all the time one explore is "
                   r"given$", line), _history(life)[-2:]
  assert int(re.search(r"for (\d+) s", line).group(1)) >= DECIDED_EXPLORE_S
