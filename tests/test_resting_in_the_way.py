"""A robot resting in the other's way, where the other's map put it
elsewhere (issue #455).

On the live pair (8a61ada, its log's last 16 hours) thirteen walks gave up
naming a robot lying down to rest, and only one had an ask beside it. The
encounters say why: each time the two bodies were within 2 m, while the
words said 3.3-3.6 m. Luca's map of the lab sat 2.4-2.9 m and 9.5 deg off
for hours, and a robot resting was kept clear of where it SAID it was --
its belief. So the planner routed round a spot it was not at, the depth
camera held the walk at its body, a stagnated drive found nobody near
enough to ask, and the words named it 3.4 m off. The one ask, Rowan's of
Luca at the dock, was answered five times by a step of 0.0 m: by its own
map Luca was already off the way it was sent. And three walks to the
workshop gave up at once, 10-25 m from Rowan walking the hall.

Each rule pinned as cheaply as it fails, on the pair as built with
nothing stepped, or on the drive with its legs stubbed:

  1. WHERE A ROBOT LYING STILL IS: its body, placed as the other's own
     sensors would (`HubLifecycle.keep_clear`), resting or dead; a drive
     stalled beside one asks it, where its belief said 3 m off.
  2. THE WAY IT IS ASKED OFF is told relative to its body (`make_way`).
  3. A FAR ROBOT is planned past (`PAST_M`), and the walk walks on.
  4. THE PRESS: the walk in rides the ask, and is cleared before it is
     walked -- or not walked, "in the way"; and a goal one lies on, its
     standoff, is asked off at the plan that finds it there (#439).
  5. THE WORDS: dead is dead, and a no is said, once a while.

The day's attribution is in docs/SimNotes.md, "A robot resting where the
other's map put it elsewhere".
"""

import math

import mujoco
import pytest

from pluggybot import navigator as nav
from pluggybot import tick
from pluggybot.body import KeepClear
from pluggybot.legs import body as qb
from pluggybot.legs import world as lw
from pluggybot.legs.body import QuadMission
from pluggybot.lifecycle import QUAD_HOME, posture
from pluggybot.navigator import MAKE_WAY_WAIT_S, PAST_M, STAGNATION_S, gave_up
from pluggybot.procedure import steps as st
from test_make_way import (BOUNDS, DOOR, EAST, GOAL, STEPPER, _door_map, _Drive,  # noqa: I001
                           _off, _resting)
from test_places import FEED, SOUTH, _quad, _sign

Y = -0.5
LYING_AT = (0.2, Y)


def _pair(tmp_path, near_field=False):
  """The quadruped pair in the living room's open floor, both maps laid
  free: Luca at (-1.6, -0.5) facing +x, Rowan lying down to rest at
  (0.2, -0.5) facing back at it. Beliefs exact until a test moves one."""
  from pluggybot.pair import build_pair
  me, peer = build_pair(QUAD_HOME, errands=("none", "none"), names=("Luca", "Rowan"),
                        thoughts_root=str(tmp_path / "thoughts"), near_field=near_field)
  me.body.start_at(-1.6, Y, 0.0)
  peer.body.start_at(*LYING_AT, math.pi)
  for life in (me, peer):
    life.body.grid.grid[:] = -5.0
  peer.body.mission.posture = qb.LYING
  return me, peer


def _close(*lives):
  for life in lives:
    life.body.close()


def _believe(life, dx, dy, dyaw=0.0):
  """Move one robot's belief off its body, its body left where it is."""
  x, y, th = life.body.true_pose()
  life.body.mission._set_pose(x + dx, y + dy, th + dyaw)


# ---- 1. where a robot lying still is ------------------------------------------------


def test_a_robot_lying_still_is_kept_clear_of_where_its_body_lies_as_the_other_sees_it(tmp_path):
  """The rule and its line of wiring (`pair.build_pair` hands each mission
  `keep_clear(seen_by=)`): a robot lying down to rest is where the other's
  sensors put its body, not where it says it is -- its belief here 2.9 m
  off, as Luca's was -- and placed through the other's map, drifted as
  Luca's lab was. Dead, likewise, and said dead; standing, where it says."""
  me, peer = _pair(tmp_path)
  try:
    _believe(peer, 0.0, 2.9)
    keep = me.body.others[0]()
    assert (keep.x, keep.y) == pytest.approx(me.body.as_seen(*peer.body.footprint_centre()),
                                             abs=1e-9)
    assert math.dist((keep.x, keep.y), peer.body.pose_xy()) > 2.5, "the premise"
    assert keep.resting and not keep.dead
    _believe(me, 1.2, -2.9, math.radians(-9.5))
    keep = me.body.others[0]()
    seen = me.body.as_seen(*peer.body.footprint_centre())
    assert (keep.x, keep.y) == pytest.approx(seen, abs=1e-9)
    assert math.dist(seen, peer.body.footprint_centre()) > 2.5, "the premise: Luca's drift"
    peer.dead = {"cause": "unminded"}
    keep = me.body.others[0]()
    assert keep.dead and not keep.resting and (keep.x, keep.y) == pytest.approx(seen, abs=1e-9)
    assert posture([peer], "Rowan") == "lying dead"
    peer.dead = None
    peer.body.mission.posture = qb.STANDING
    keep = me.body.others[0]()
    assert (keep.x, keep.y) == peer.body.pose_xy() and not (keep.resting or keep.dead)
  finally:
    _close(me, peer)


def test_a_robot_lying_still_to_rest_or_dead_is_not_held_for(tmp_path):
  """The depth camera's hold is for a robot that will MOVE (#365). Placed
  where its body lies, a robot resting is planned round, and asked off a
  way it cuts; held for, a walk past one resting mid-hall stood at its
  disc's edge facing it while the two ways round it swapped plan by plan,
  and timed out. Dead, likewise; standing -- stepping aside -- it moves,
  and is held for."""
  me, peer = _pair(tmp_path, near_field=True)
  try:
    me.body.start_at(LYING_AT[0] - 0.6, Y, 0.0)
    frame = me.depth_camera.frame(me.data)
    assert me.body.mission.peer_ahead(frame.peers) is not None, \
      "the premise: the camera sees it in the corridor ahead"
    m = me.body.mission
    for how, held in (("resting", False), ("dead", False), ("standing", True)):
      peer.body.mission.posture = qb.STANDING if how == "standing" else qb.LYING
      peer.dead = {"cause": "unminded"} if how == "dead" else None
      m.peer_seen_m = None
      me.data.time += 0.1
      me._next_near_field = 0.0
      me._near_field_step()
      assert (m.peer_sighting() is not None) is held, how
  finally:
    _close(me, peer)


def test_a_stalled_drive_asks_the_robot_resting_on_its_goal_whose_belief_is_metres_off(tmp_path):
  """The live press, its standoff 0.2 m from Rowan's body and Rowan said
  3.3 m off: no progress for `STAGNATION_S`, and a drive waits on, and asks,
  a robot near it or its goal -- none, by the belief. By the body, Rowan is
  near the goal, it is asked off the way there, and it starts to step off."""
  me, peer = _pair(tmp_path)
  try:
    _believe(peer, 0.0, 3.0)
    goal = (LYING_AT[0] - 0.2, Y)
    m = me.body.mission
    assert m._other_in_the_way(*goal), "nobody near enough to wait on"
    assert m._ask_way(*goal, {}, near=True) and peer.body.making_way == me.root
  finally:
    _close(me, peer)


# ---- 2. the way it is asked off ----------------------------------------------------


def test_the_way_is_told_relative_to_the_body_it_is_asked_off(tmp_path):
  """Rowan's charge at the dock: Luca's belief 1.1 m off its body, asked off
  a way running through its body, which by its own map ran 1.1 m off it --
  so it stood up, stepped 0.0 m aside, and lay down again, five times. Told
  where the way runs against its body, it steps `aside_clear_m` off it."""
  me, peer = _pair(tmp_path)
  try:
    _believe(peer, 0.0, 1.1)
    way = [me.body.pose_xy(), LYING_AT, (1.5, Y)]
    assert me.body.ask_way(peer.root, way)
    at, to = peer.body.pose_xy(), peer.body.mission._aside["to"]
    assert math.dist(at, to) > 0.5, f"stepped {math.dist(at, to):.2f} m aside"
    told = [peer.body.as_seen(*me.body.from_seen(*p)) for p in way]
    assert _off(told, *at) < 0.1, "the premise: the way runs through it"
    assert _off(told, *to) >= peer.body.mission.aside_clear_m - 1e-6
  finally:
    _close(me, peer)


# ---- 3. a far robot ------------------------------------------------------------------


@pytest.fixture(scope="module")
def house():
  return lw.home_spec().compile()


def test_a_route_only_a_far_robots_disc_cuts_is_walked_and_a_near_ones_is_not(house):
  """Rowan walking the hall, the one way into the workshop: Luca's plan
  took its disc for a wall and the walk gave up at once, 25 m off, with
  Rowan long gone before Luca could get there. Further off than `PAST_M`,
  only its disc cutting the way, it is planned past and the walk walks;
  nearer, it is the wall it was, and the walk gives up as before."""
  m = QuadMission(house, mujoco.MjData(house), realtime=False, grid_bounds=BOUNDS)
  try:
    _door_map(m)
    m.others = [lambda: KeepClear(*DOOR, root="r2_pluggybot")]
    for off, walks in ((PAST_M + 2.0, True), (1.5, False)):
      m._set_pose(DOOR[0] - off, DOOR[1], 0.0)
      m._plan_memo = None
      assert (m._plan_to(*EAST) is not None) is walks, off
      walk = m.drive_to_routine(*EAST, 60.0)
      step = next(walk, None)
      walk.close()
      assert (step is not None and qb.is_motion(step)) is walks, (off, m.last_drive)
      if not walks:
        assert m.last_drive["why"] == "peer" and m.last_drive["seconds"] == 0.0
  finally:
    m.close()


# ---- 4. the press --------------------------------------------------------------------


def test_the_ask_carries_the_way_on_past_the_goal():
  """A robot asked off the walk to a press's standoff -- one backed out of
  its own press lies 0.25 m short of it -- is asked off the walk in beyond
  it too (`beyond`), or the nearest spot off the way to the standoff is
  on the way to the plate."""
  drive = _Drive([_resting()], opens_at=7.0)
  assert tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0, beyond=[(6.0, 0.0)]))
  root, way = drive.asked[0]
  assert way == [(0.0, 0.0), GOAL, (6.0, 0.0)]


def test_a_drive_whose_goal_a_resting_robot_lies_on_asks_it_at_the_plan_that_finds_it():
  """#439's flights of the live press, the walker's map 2.9 m off as Luca's
  was: a robot lying 0.25 m short of the standoff put the standoff inside
  its disc, every plan ended at a stand-in beside it, and only a
  stagnation asked. None came -- the drive steered at the goal from one
  stand-in, planned from the far side of the disc, and every new route was
  progress -- and the walk circled it for its 85 s, unasked. A goal a
  resting robot lies on is a way it cuts: asked at the plan that finds it
  there, within `PAST_M` of it, and waited for while it steps off; from
  further off it is walked toward, as a far disc across the way is."""
  class OnTheGoal(_Drive):
    def _plan_to(self, wx, wy):
      self._stand_in = None if self._open() else (wx - 0.6, wy)
      return [(wx, wy)] if self._open() else [self._stand_in]

    def _nav_routine(self, v, w):
      self.data.time += 0.1
      if self._open():
        self.pose = (GOAL[0], GOAL[1], 0.0)
      yield v, w

  def walker(at=PAST_M - 0.5, **kw):
    lying = {k: kw.pop(k) for k in ("down",) if k in kw}
    drive = OnTheGoal([_resting(GOAL[0] + 0.25, GOAL[1], **lying)], **kw)
    drive.pose = (GOAL[0] - at, GOAL[1], 0.0)
    return drive

  drive = walker(opens_at=8.0)
  asked_at = []
  ask = drive.ask_way
  drive.ask_way = lambda root, way: asked_at.append(drive.data.time) or ask(root, way)
  assert tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0))
  assert asked_at and asked_at[0] == 0.0, asked_at
  assert drive.last_drive["askedWay"] == ["r2_pluggybot"]
  # ...from across the house it is walked toward, and nobody is asked yet
  far = walker(at=PAST_M + 1.0, opens_at=8.0)
  assert tick.run(STEPPER, far.drive_to_routine(*GOAL, 60.0)) and far.asked == []
  # ...and one that says no is not waited for on its account, nor a fallen
  # one asked (#365): each drive ends as it did
  no = walker(says=False)
  assert not tick.run(STEPPER, no.drive_to_routine(*GOAL, 30.0))
  assert no.asked and no.last_drive["why"] == "peer"
  fallen = walker(down=True)
  assert not tick.run(STEPPER, fallen.drive_to_routine(*GOAL, 30.0))
  assert not fallen.asked and fallen.last_drive["why"] == "peer"
  assert fallen.last_drive["seconds"] <= STAGNATION_S + 0.5


def test_a_robot_on_the_goal_that_says_no_is_asked_once_a_wait_at_most():
  """#439's review: a drive at its stand-in, its waypoints spent, replans
  every step, and the ask plans a way past the robot on the goal (~29 ms,
  past the plan's memo). One that said no was asked 4 729 times in an 85 s
  walk, 139 s of planning on the pair's one physics thread. Once every
  `OTHER_WAIT_S` at most."""
  class AtTheStandIn(_Drive):
    def _plan_to(self, wx, wy):
      self._stand_in = (wx - 0.6, wy)
      return [self._stand_in]

    def _nav_routine(self, v, w):
      self.data.time += 0.1
      yield v, w

  drive = AtTheStandIn([_resting(GOAL[0] + 0.25, GOAL[1])], says=False)
  drive.pose = (GOAL[0] - 0.6, GOAL[1], 0.0)      # ...standing on it
  assert not tick.run(STEPPER, drive.drive_to_routine(*GOAL, 30.0))
  assert 0 < len(drive.asked) <= 2 * 30.0 / nav.OTHER_WAIT_S, len(drive.asked)


class _Clear(_Drive):
  clear_way_routine = nav.Navigator.clear_way_routine
  _across = nav.Navigator._across


@pytest.mark.parametrize("who", ["resting", "refuses", "standing", "dead", "nobody"])
def test_a_way_no_plan_walks_is_cleared_by_asking_and_waiting_or_not_walked(who):
  """`clear_way_routine`, the press's walk in: a robot resting across it is
  asked off it and waited for while it steps off; one standing on it while
  it moves on; one that says no, or is dead, is not waited for -- the way
  is not walked -- and nobody across it is no wait at all."""
  way = [(0.0, 0.0), (1.2, 0.0)]
  gone_at = 5.0

  def there(**kw):
    def where():
      if who in ("resting", "standing") and drive.data.time >= gone_at:
        return KeepClear(0.6, 2.0, root="r2_pluggybot")
      return KeepClear(0.6, 0.1, root="r2_pluggybot", **kw)
    return where

  others = {"resting": [there(resting=True)], "refuses": [there(resting=True)],
            "standing": [there()], "dead": [there(dead=True)], "nobody": []}[who]
  drive = _Clear(others, says=who != "refuses")
  across = tick.run(STEPPER, drive.clear_way_routine(way))
  if who in ("resting", "standing", "nobody"):
    assert across is None
    assert drive.data.time == (0.0 if who == "nobody" else pytest.approx(6.0)), drive.data.time
  else:
    assert across is not None and across.root == "r2_pluggybot" and drive.data.time == 0.0
  assert [r for r, _ in drive.asked] == (["r2_pluggybot"] * 3 if who == "resting"
                                         else ["r2_pluggybot"] if who == "refuses" else [])
  if drive.asked:
    assert drive.asked[0][1] == way


def test_a_robot_waiting_for_its_way_in_is_never_asked_aside(house):
  """The press's wait is a walk of its own on the quadruped (`driving`):
  8.6 s into it the rest reflex lies the body down, and a body lying free
  says yes to an ask -- it would walk off its standoff before its walk
  in, and walk in from wherever it stood aside."""
  m = QuadMission(house, mujoco.MjData(house), realtime=False, grid_bounds=BOUNDS)
  try:
    m.grid.grid[:] = -5.0
    m._set_pose(0.0, 0.0, 0.0)
    m.others = [lambda: KeepClear(1.0, 0.0, root="r2_pluggybot", resting=True)]
    m.ask_way = lambda root, way: True
    wait = m.clear_way_routine([(0.0, 0.0), (2.0, 0.0)])
    next(wait)
    m.posture = qb.LYING
    assert m.make_way([(4.0, 3.0), (6.0, 3.0)], "r2_pluggybot") is False, m._aside
    assert m.way_refused == "walking"
    wait.close()
    assert m.driving == 0
  finally:
    m.close()


def test_a_press_asks_off_its_walk_in_and_does_not_walk_into_who_stays(house):
  """The press's two halves (`PlaceWalk.press_routine`): the walk to its
  standoff carries the walk in beyond it, and before the walk in -- steered
  by the sign, no planner under it -- the way to the plate is cleared. One
  still across it ends the try "in the way", and nothing walks into it."""
  body = _quad(house, 25.0, 2.0, SOUTH)
  m = body.mission
  beyonds, cleared, walked_in = [], [], []
  stays = KeepClear(25.0, 1.4, root="r2_pluggybot", resting=True)

  def drive(x, y, timeout=90.0, stop=None, beyond=()):
    beyonds.append(list(beyond))
    return tick.result(True)

  try:
    m.places.see(FEED, *_sign("feed"), 0.0, view=SOUTH, tag_facing=SOUTH)
    m.drive_to_routine = drive
    m.face_routine = lambda h: tick.result(True)
    m.look_for_places = lambda: [FEED]
    m._press_walk_in_routine = lambda tag: walked_in.append(tag) or tick.result("stopped")
    m._hold_on_routine = lambda pad: tick.result(True)
    m._press_back_out_routine = lambda: tick.result(None)
    m.clear_way_routine = lambda way: cleared.append(list(way)) or tick.result(stays)
    rec = body.run(m.press_routine(FEED, patience=st.PRESS_PATIENCE_S))
    press = m.press_pose(FEED)[:2]
    assert beyonds and all(b == [pytest.approx(press)] for b in beyonds)
    assert cleared and cleared[0][-1] == pytest.approx(press)
    assert rec["why"] == "in the way" and not walked_in
    assert rec["attempts"][-1]["across"] == {"root": "r2_pluggybot", "m": pytest.approx(0.6, abs=0.01),
                                            "rests": True, "dead": False, "down": False}
    m.clear_way_routine = lambda way: tick.result(None)
    rec = body.run(m.press_routine(FEED, patience=st.PRESS_PATIENCE_S))
    assert rec["pressed"] and walked_in == [FEED]
  finally:
    body.close()


def test_a_press_that_was_not_walked_onto_says_who_lay_across_the_way():
  """The verb's words for "in the way": who, how and how far -- and its
  trace -- so the robot reads a body across its way, not a plate."""
  rec = {"pressed": False, "why": "in the way",
         "attempts": [{"standoff": [25.0, 2.0], "why": "in the way",
                       "across": {"root": "r2_pluggybot", "m": 0.6, "rests": True,
                                  "dead": False, "down": False}}]}
  rowan = type("Peer", (), {"robot_name": "Rowan", "root": "r2_pluggybot"})()
  life = type("Life", (), {"_peer": lambda self, root: rowan if root == rowan.root else None})()
  life.body = type("Body", (), {"press_plate_routine":
                                lambda self, tag, patience, stop=None: tick.result(rec)})()
  life.data = type("Data", (), {"time": 0.0})()
  out = tick.run(STEPPER, st._press(life, {"tag": FEED}))
  assert out["reason"] == (f"did not walk onto tag {FEED}'s plate: Rowan lay across the "
                           "way onto it, 0.6 m off, and did not step off it")
  assert "walk in across" in out["trace"]


# ---- 5. the words --------------------------------------------------------------------


def test_a_dead_robot_is_neither_waited_for_nor_asked_and_is_said_dead():
  """Rowan's charge four minutes after Luca's `unminded` death read "Luca
  lying down to rest in the way, 1.1 m off". It lies where it died until
  the world stands it up: never asked (#441 refuses it), never waited on,
  and said dead."""
  dead = lambda: KeepClear(2.0, 0.0, root="r2_pluggybot", dead=True)  # noqa: E731
  drive = _Drive([dead])
  assert not tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0))
  rec = drive.last_drive
  assert (rec["why"], rec["seconds"], drive.asked) == ("peer", 0.0, [])
  assert gave_up(rec, "Luca").endswith("(Luca lying dead in the way, 2.0 m off)")
  beside = _Drive([lambda: KeepClear(0.8, 0.0, root="r2_pluggybot", dead=True)], plans=[GOAL])
  tick.run(STEPPER, beside.drive_to_routine(*GOAL, STAGNATION_S + 30.0))
  assert beside.last_drive["seconds"] < STAGNATION_S + 1.0, "it waited on the dead"
  assert beside.asked == []


def test_a_no_is_said_once_a_while_and_with_why(tmp_path):
  """Of thirteen give-ups naming a robot resting, one had an ask beside it
  in the log -- and a no left no line at all. The robot asked says why it
  cannot, once a `MAKE_WAY_WAIT_S`; standing it is in nobody's way by its
  own rule, and says nothing."""
  me, peer = _pair(tmp_path)
  try:
    way = [me.body.pose_xy(), LYING_AT, (1.5, Y)]
    said = lambda: [ln for ln in peer.log if "asked me off its way" in ln]  # noqa: E731
    peer.body.mission.docked = True
    assert not me.body.ask_way(peer.root, way)
    assert peer.status == "MAKE WAY: Luca asked me off its way, and I cannot: I am on the dock"
    assert not me.body.ask_way(peer.root, way) and len(said()) == 1
    me.data.time += MAKE_WAY_WAIT_S
    assert not me.body.ask_way(peer.root, way) and len(said()) == 2
    peer.body.mission.docked = False
    peer.body.mission.posture = qb.STANDING
    assert not me.body.ask_way(peer.root, way) and len(said()) == 2
  finally:
    _close(me, peer)
