"""A robot knocked over is avoided where it lies (issue #365).

On 2026-09-23 one robot drove into the other, which had been on its side
for three minutes, and fell over too. The planner's disc round a peer is
taken round where the peer SAYS it is, and a robot knocked over mid-errand
keeps reckoning as its errand drives it until the errand returns: MEASURED
up to 2.2 m of reported pose off the body in 10 s. So the disc went with
the belief, and nothing routed round the body.

The world here is the quadruped pair in the living room's open floor: the
driver starts at (-1.6, -0.5) facing +x, and the other robot lies across
its path at (-0.3, -0.5). Nothing is stepped but where a rule is the
drive's, and there a stub walks. The flown reproduction, guard by guard,
is in docs/SimNotes.md, "A robot lying down was avoided where it said it
was".
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import KeepClear
from pluggybot.legs import world as lw
from pluggybot.lifecycle import QUAD_HOME, world_config
from pluggybot.navigator import CLOSE_ENOUGH_M, DOWN_CHECK_S
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import SECOND
from test_unknown import STEPPER, _Walk  # noqa: I001 -- tests/ is on sys.path

S = math.sqrt(0.5)
#: The root quaternion of a robot facing +x and lying on its left side,
#: its right side, its front or its back.
LYING = {"left": (S, S, 0.0, 0.0), "right": (S, -S, 0.0, 0.0),
         "front": (S, 0.0, S, 0.0), "back": (S, 0.0, -S, 0.0)}
Y = -0.5
AT = (-0.3, Y)
GOAL = (1.3, Y)


def _downed_pair(lie: str = "left", near_field: bool = False):
  """The quadruped pair with the second robot lying across the first one's
  path, its reckoning 1.8 m off its body the way its errand's walk puts it
  -- set rather than flown, which is seconds of physics this rule does not
  need. The pair's OWN wiring is left in place: it is half of what is
  under test."""
  from pluggybot.pair import build_pair
  me, peer = build_pair(QUAD_HOME, near_field=near_field, errands=("none", "none"))
  me.body.start_at(-1.6, Y, 0.0)
  peer.home_pose = tuple(world_config(QUAD_HOME)["start2"])
  adr = SECOND.qpos_adr(me.model)
  me.data.qpos[adr:adr + 3] = [AT[0], AT[1], 0.15]
  me.data.qpos[adr + 3:adr + 7] = LYING[lie]
  mujoco.mj_forward(me.model, me.data)
  peer.body.mission.odo.y += 1.8
  me.body.grid.grid[:] = -5.0                     # a mapped, empty room
  return me, peer


def _plan(life, x, y):
  """A fresh plan: the quadruped keeps its last one for a moment from the
  same spot (`QuadMission.REPLAN_HOLD_S`)."""
  life.body.mission._plan_memo = None
  return life.body.mission._plan_to(x, y)


def _clearance(point, peer) -> float:
  """How far `point` is from the other robot's BODY: its nearest geom's
  bounding circle on the floor. Negative is inside it."""
  gids = sorted(robot_geoms(peer.model, SECOND.root))
  xy = peer.data.geom_xpos[gids, :2]
  r = peer.model.geom_rbound[gids]
  return float((np.hypot(*(xy - np.asarray(point)).T) - r).min())


@pytest.mark.parametrize("lie", sorted(LYING))
def test_a_robot_lying_down_is_avoided_where_it_lies_not_where_it_says(lie):
  """The rule and its one line of wiring (`pair.build_pair` hands every
  mission `HubLifecycle.keep_clear`). Without it the disc sits 1.8 m off
  at the reckoning's answer and the plan runs straight through the body."""
  me, peer = _downed_pair(lie)
  assert peer.down()
  body = peer.body.footprint_centre()
  said = peer.body.pose_xy()
  assert math.dist(body, said) > 1.5, "the premise: the reckoning left the body"
  keep = me.body.others[0]()
  mission = me.body.mission
  assert keep.down and mission._cells(keep) == mission.DOWN_ROBOT_CELLS
  x, y = keep.x, keep.y
  assert (x, y) == pytest.approx(me.body.as_seen(*body), abs=1e-9)
  assert math.dist((x, y), body) < 0.02         # a driver that knows where it is
  path = _plan(me, *GOAL)
  assert path is not None and math.dist(path[-1], GOAL) < 0.1
  # The plan keeps this robot's centre a swing (the map's own inflation)
  # off every part of the body, less one cell of rasterising.
  swing = mission.INFLATION_CELLS * me.body.grid.resolution
  near = min(_clearance(p, peer) for p in path)
  assert near > swing - me.body.grid.resolution, \
    f"on its {lie}: a waypoint {near:.3f} m from the body"
  # ...where the wiring before #365 -- the reported pose, a standing
  # robot's disc -- planned the straight line, through the robot.
  mission.others = [peer.body.pose_xy]
  straight = _plan(me, *GOAL)
  assert min(_clearance(p, peer) for p in straight) < 0.0


def test_the_body_is_placed_where_the_drivers_own_sensors_would_put_it():
  """The driver's drift cancels: the disc goes where a lump in its depth
  image would land in its map -- seen from where it truly stands, placed
  through where it believes it stands (`Navigator.as_seen`). Placed at the
  body's world position instead, a driver 0.4 m out would plan to pass
  0.4 m nearer the robot than it thinks."""
  me, peer = _downed_pair()
  body = peer.body.footprint_centre()
  odo = me.body.mission.odo
  odo.correct(odo.x + 0.4, odo.y - 0.1, odo.yaw + 0.2)
  x, y, _ = me.body.others[0]()
  # as far ahead, and as far round from its heading, as it truly is
  tx, ty, tth = me.body.true_pose()
  bx, by, bth = me.body.pose
  assert math.dist((x, y), (bx, by)) == pytest.approx(
    math.dist(body, (tx, ty)), abs=1e-9)
  assert math.atan2(y - by, x - bx) - bth == pytest.approx(
    math.atan2(body[1] - ty, body[0] - tx) - tth, abs=1e-9)
  assert math.dist((x, y), body) > 0.3, "the premise: the driver has drifted"
  # ...and a standing robot is where it SAYS it is, for everyone
  peer.stand_up("a test", auto=False)
  assert me.body.others[0]() == peer.body.pose_xy()


def test_a_robot_lying_down_is_not_held_for():
  """The one line at the seam: the depth camera's peer channel sees a
  robot lying down, and the hold that channel drives is for a robot that
  will MOVE. One on the floor will not, and a detour that runs straight at
  it and turns at its disc's edge was held 0.52 m short until the drive
  timed out."""
  me, peer = _downed_pair("front", near_field=True)
  me.body.start_at(-0.9, Y, 0.0)              # its body 0.6 m ahead
  frame = me.depth_camera.frame(me.data)
  assert me.body.mission.peer_ahead(frame.peers) is not None, \
    "the premise: the camera sees the robot on the floor in the corridor"
  me._next_near_field = 0.0
  me._near_field_step()
  assert me.body.mission.peer_sighting() is None, "held for a robot lying down"
  # ...stood up where it lay, it is a robot that moves again, and held for
  lw.stand(me.model, me.data, SECOND.prefix, AT[0], AT[1], 0.0)
  assert not peer.down()
  me.data.time += 0.1                         # the camera's next frame
  me._next_near_field = 0.0
  me._near_field_step()
  assert me.body.mission.peer_sighting() is not None


def test_standing_up_lets_go_of_where_it_lay():
  """The stand-up is a warp to the start pose that resets the reckoning:
  from then on the robot is avoided where it says it is, with a standing
  robot's disc, and the floor where it lay is floor again."""
  me, peer = _downed_pair()
  peer.stand_up("a test", auto=False)
  assert not peer.down()
  assert me.body.others[0]() == peer.body.pose_xy()
  assert math.dist(peer.body.pose_xy(), peer.home_pose[:2]) < 0.01
  path = _plan(me, *GOAL)
  assert max(abs(y - Y) for _, y in path) < 0.1, "still routing round the old spot"


def test_a_goal_beside_a_robot_lying_down_is_out_of_reach_further_off():
  """#313's arithmetic, with the wider disc: the planner may end no nearer
  the goal than `radius - distance`, and a stagnated drive is arrived only
  inside CLOSE_ENOUGH_M -- so a goal is out of reach nearer a standing
  robot than its disc less that, and a lying one's likewise with its own.
  `peer_on_the_goal` must say what the planner does, or a bay would be
  waited for that can be reached, and one tried for ever that cannot."""
  me, _ = _downed_pair()
  m = me.body.mission
  me.body.start_at(1.0, Y, 0.0)
  gx, gy, beside = 2.5, Y, 0.7
  res = m.grid.resolution
  assert m.OTHER_ROBOT_CELLS * res - CLOSE_ENOUGH_M < beside \
      < m.DOWN_ROBOT_CELLS * res - CLOSE_ENOUGH_M
  for standing, reachable in ((True, True), (False, False)):
    m.others = [(lambda: (gx + beside, gy)) if standing
                else (lambda: KeepClear(gx + beside, gy, down=True))]
    near = m.peer_on_the_goal(gx, gy)
    path = _plan(me, gx, gy)
    reached = m._stand_in is None or math.dist(path[-1], (gx, gy)) < CLOSE_ENOUGH_M
    assert (near is None) is reachable, near
    assert reached is reachable, f"{math.dist(path[-1], (gx, gy)):.3f} m short"


def test_a_drive_does_not_wait_on_a_robot_lying_down():
  """A stagnated drive WAITS for another robot near it or its goal: stand
  still, let it move. One on the floor will not until the world stands it
  up, and a drive whose goal lay in its disc stood beside it for its whole
  timeout. It plans round the body instead, or ends as a drive with
  nowhere to go does."""
  me, _ = _downed_pair()
  m = me.body.mission
  me.body.start_at(1.0, 1.0, 0.0)
  m.others = [lambda: (2.6, 1.2)]
  assert m._other_in_the_way(2.6, 1.2), "the premise: a robot standing there"
  m.others = [lambda: KeepClear(2.6, 1.2, down=True)]
  assert not m._other_in_the_way(2.6, 1.2), "waited on a robot lying down"


class _Falls(_Walk):
  """The drive's own `self`, walking 1 cm every 10 ms along x with the
  plan stubbed, every plan's time recorded, and a hook each step."""

  def __init__(self, hook):
    super().__init__(growth=False)
    self.hook, self.plans = hook, []
    self.PROGRESS_MAP_GROWTH = False

  def _plan_to(self, wx, wy):
    self.plans.append(self.data.time)
    return [(wx, wy)]

  def _nav_routine(self, v, w):
    self.data.time += 0.01
    self.pose = (self.pose[0] + 0.01, self.pose[1], 0.0)
    self.hook()
    yield v, w


def test_a_fall_and_a_stand_up_are_planned_round_at_once():
  """The camera does not hold for a robot on the floor, so the plan hears
  of a fall within `DOWN_CHECK_S` rather than at its next 2 s replan --
  MEASURED without that, a robot knocked flat 0.5 m ahead just after a
  replan was met by the LIDAR's stop -- and of a stand-up likewise, or it
  routes round an empty floor for up to 2 s. A 1.5 s drive has room for
  one routine plan and no second, so every later plan is one of these."""
  lying, flips = [False], []

  def flip() -> None:                        # falls at 0.5 s, stood up at 1.0 s
    due = (0.5, 1.0)
    if len(flips) < 2 and walk.data.time >= due[len(flips)]:
      lying[0] = not lying[0]
      flips.append(float(walk.data.time))

  walk = _Falls(flip)
  walk.others = [lambda: KeepClear(0.5, 1.5, down=lying[0])]   # off the path
  tick.run(STEPPER, walk.drive_to_routine(20.0, 0.0, 1.5))
  assert len(flips) == 2
  for when in flips:
    after = [t for t in walk.plans if t >= when]
    assert after and after[0] - when <= DOWN_CHECK_S + 0.011, \
      f"planned {[round(t, 2) for t in walk.plans]}, flipped at {when:.2f}"
  assert len(walk.plans) == 3, "the premise: no routine replan in 1.5 s"


def test_history_says_a_robot_on_the_floor_is_lying_there():
  """A line saying a robot on the floor STOOD at the bay it blocked reads
  as a robot in the way on purpose, and History is what the mind reads
  back (`posture`, shared with the `WAIT:` narration)."""
  me, peer = _downed_pair()
  blocked = (peer.robot_name or peer.root, 0.4)
  assert "was lying knocked over 0.40 m from the bay" in me.held_for(blocked)
  peer.stand_up("a test", auto=False)
  assert "was standing 0.40 m from the bay" in me.held_for(blocked)
