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

Each rule pinned as cheaply as it fails, on the humanoid pair as built
with nothing stepped (a humanoid rests standing), or on the drive with its
legs stubbed:

  1. WHERE A ROBOT RESTING IS: its body, placed as the other's own sensors
     would (`HubLifecycle.keep_clear`), resting or dead; a drive stalled
     beside one asks it, where its belief said 3 m off.
  2. THE WAY IT IS ASKED OFF is told relative to its body (`make_way`).
  3. A FAR ROBOT is planned past (`PAST_M`), and the walk walks on.
  4. THE WAY ON: an ask carries the way on past the goal, a way no plan
     walks is cleared before it is walked -- or not walked -- and a goal
     one rests on is asked off at the plan that finds it there (#439).
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
from pluggybot.humanoid import world as hw
from pluggybot.humanoid.body import HumanoidBody
from pluggybot.humanoid.way import REST_S
from pluggybot.lifecycle import HUMANOID_HOME, posture
from pluggybot.navigator import MAKE_WAY_WAIT_S, PAST_M, STAGNATION_S, gave_up
from test_make_way import (BOUNDS, DOOR, EAST, GOAL, STEPPER, _door_map, _Drive,  # noqa: I001
                           _off, _resting)

Y = -0.5
REST_AT = (0.2, Y)


def _pair(tmp_path, near_field=False):
  """The humanoid pair in the living room's open floor, both maps laid
  free: Luca at (-1.6, -0.5) facing +x, Rowan resting at (0.2, -0.5) facing
  back at it. Beliefs exact until a test moves one."""
  from pluggybot.pair import build_pair
  me, peer = build_pair(HUMANOID_HOME, errands=("none", "none"), names=("Luca", "Rowan"),
                        thoughts_root=str(tmp_path / "thoughts"), near_field=near_field)
  me.body.start_at(-1.6, Y, 0.0)
  peer.body.start_at(*REST_AT, math.pi)
  for life in (me, peer):
    life.body.grid.grid[:] = -5.0
  _rest(peer)
  return me, peer


def _rest(life, resting=True):
  """At rest -- no motion command for `REST_S` -- or just moved."""
  life.body.mission.last_motion_t = (float(life.data.time) - REST_S if resting
                                     else float(life.data.time))


def _close(*lives):
  for life in lives:
    life.body.close()


def _believe(life, dx, dy, dyaw=0.0):
  """Move one robot's belief off its body, its body left where it is."""
  x, y, th = life.body.true_pose()
  life.body.mission._set_pose(x + dx, y + dy, th + dyaw)


# ---- 1. where a robot resting is ----------------------------------------------------


def test_a_robot_resting_is_kept_clear_of_where_its_body_stands_as_the_other_sees_it(tmp_path):
  """The rule and its line of wiring (`pair.build_pair` hands each mission
  `keep_clear(seen_by=)`): a robot at rest is where the other's sensors put
  its body, not where it says it is -- its belief here 2.9 m off, as Luca's
  was -- and placed through the other's map, drifted as Luca's lab was.
  Dead, likewise, and said dead; moving, where it says."""
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
    _rest(peer, resting=False)
    keep = me.body.others[0]()
    assert (keep.x, keep.y) == peer.body.pose_xy() and not (keep.resting or keep.dead)
  finally:
    _close(me, peer)


def test_a_robot_resting_or_dead_is_not_held_for(tmp_path):
  """The depth camera's hold is for a robot that will MOVE (#365). Placed
  where its body stands, a robot resting is planned round, and asked off a
  way it cuts; held for, a walk past one resting mid-hall stood at its
  disc's edge facing it while the two ways round it swapped plan by plan,
  and timed out. Dead, likewise; moving -- stepping aside -- it is held
  for."""
  me, peer = _pair(tmp_path, near_field=True)
  try:
    me.body.start_at(REST_AT[0] - 0.6, Y, 0.0)
    frame = me.depth_camera.frame(me.data)
    assert me.body.mission.peer_ahead(frame.peers) is not None, \
      "the premise: the camera sees it in the corridor ahead"
    m = me.body.mission
    for how, held in (("resting", False), ("dead", False), ("moving", True)):
      _rest(peer, resting=how != "moving")
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
    goal = (REST_AT[0] - 0.2, Y)
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
  where the way runs against its body, it steps `aside_clear_m` off it
  (on the planner's 10 cm lattice: a cell's middle half a cell nearer)."""
  me, peer = _pair(tmp_path)
  try:
    _believe(peer, 0.0, 1.1)
    way = [me.body.pose_xy(), REST_AT, (1.5, Y)]
    assert me.body.ask_way(peer.root, way)
    at, to = peer.body.pose_xy(), peer.body.mission._aside["to"]
    assert math.dist(at, to) > 0.5, f"stepped {math.dist(at, to):.2f} m aside"
    told = [peer.body.as_seen(*me.body.from_seen(*p)) for p in way]
    assert _off(told, *at) < 0.1, "the premise: the way runs through it"
    assert _off(told, *to) >= peer.body.mission.aside_clear_m - 0.05
  finally:
    _close(me, peer)


# ---- 3. a far robot ------------------------------------------------------------------


@pytest.fixture(scope="module")
def house():
  return hw.home_spec().compile()


def test_a_route_only_a_far_robots_disc_cuts_is_walked_and_a_near_ones_is_not(house):
  """Rowan walking the hall, the one way into the workshop: Luca's plan
  took its disc for a wall and the walk gave up at once, 25 m off, with
  Rowan long gone before Luca could get there. Further off than `PAST_M`,
  only its disc cutting the way, it is planned past and the walk walks;
  nearer, it is the wall it was, and the walk gives up as before."""
  body = HumanoidBody(house, mujoco.MjData(house), realtime=False, grid_bounds=BOUNDS)
  m = body.mission
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
      assert (step is not None and any(step)) is walks, (off, m.last_drive)
      if not walks:
        assert m.last_drive["why"] == "peer" and m.last_drive["seconds"] == 0.0
  finally:
    body.close()


# ---- 4. the way on ------------------------------------------------------------------


def test_the_ask_carries_the_way_on_past_the_goal():
  """A robot asked off a walk that goes on past its goal (`beyond`: a
  standoff, and the walk in from it) is asked off the way beyond too, or
  the nearest spot off the way to the goal is on the way past it."""
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
  cannot, once a `MAKE_WAY_WAIT_S`; not yet at rest it is in nobody's way
  by its own rule, and says nothing."""
  me, peer = _pair(tmp_path)
  try:
    way = [me.body.pose_xy(), REST_AT, (1.5, Y)]
    said = lambda: [ln for ln in peer.log if "asked me off its way" in ln]  # noqa: E731
    (port,) = peer.body.mission.port_eqs
    peer.data.eq_active[port] = 1                       # plugged in
    assert not me.body.ask_way(peer.root, way)
    assert peer.status == ("MAKE WAY: Luca asked me off its way, and I cannot: "
                           "I am plugged in at my charging station")
    assert not me.body.ask_way(peer.root, way) and len(said()) == 1
    me.data.time += MAKE_WAY_WAIT_S
    _rest(peer)
    assert not me.body.ask_way(peer.root, way) and len(said()) == 2
    peer.data.eq_active[port] = 0
    _rest(peer, resting=False)
    assert not me.body.ask_way(peer.root, way) and len(said()) == 2
  finally:
    _close(me, peer)
