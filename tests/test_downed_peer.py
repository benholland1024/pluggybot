"""A robot lying on the floor is avoided where it lies (issue #365).

On 2026-09-23 Rowan drove into Luca, who had been on its side for three
minutes, and fell over too. The planner's disc round a peer is taken round
where the peer SAYS it is, and a robot knocked over mid-errand keeps
turning its wheels until the errand returns: a wheel in the air is travel
to the reckoner, MEASURED up to 2.2 m off the body in 10 s. So the disc
went with the belief, and nothing routed round the body.

The world here is room_hub's open floor, from the west wall (x = -2) to the
floor box (x = 2): the driver starts at (-1.6, 0) facing +x, and the other
robot lies across its path at (-0.3, 0). The flown reproduction, guard by
guard, is in docs/SimNotes.md, "A robot lying down was avoided where it
said it was".
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.mission.mission import (
  CLOSE_ENOUGH_M, DOWN_CHECK_S, DOWN_ROBOT_CELLS, OTHER_ROBOT_CELLS,
  HubMission, KeepClear,
)
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import FIRST, SECOND

S = math.sqrt(0.5)
#: The root quaternion of a robot facing +x and lying on its left side,
#: its right side, its front or its back.
LYING = {"left": (S, S, 0.0, 0.0), "right": (S, -S, 0.0, 0.0),
         "front": (S, 0.0, S, 0.0), "back": (S, 0.0, -S, 0.0)}
AT = (-0.3, 0.0)
GOAL = (1.3, 0.0)


def _downed_pair(lie: str = "left", near_field: bool = False):
  """A pair on room_hub's open floor with the second robot lying across the
  first one's path, its reckoner 1.8 m off its body the way its errand's
  wheels put it -- set rather than flown, which is seconds of physics this
  rule does not need. The pair's OWN wiring is left in place: it is half of
  what is under test."""
  from pluggybot.pair import build_pair
  me, peer = build_pair("room_hub", near_field=near_field,
                        errands=("none", "none"))
  me.mission.start_at(-1.6, 0.0, 0.0)
  peer.home_pose = (3.0, 3.0, math.pi / 2)             # room_hub's start2
  adr = SECOND.qpos_adr(me.model)
  me.data.qpos[adr:adr + 3] = [AT[0], AT[1], 0.1]
  me.data.qpos[adr + 3:adr + 7] = LYING[lie]
  mujoco.mj_forward(me.model, me.data)
  peer.mission.swap.reckoner.y += 1.8
  me.mission.grid.grid[:] = -5.0                     # a mapped, empty room
  return me, peer


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
  at the reckoner's answer and the plan runs straight through the body."""
  me, peer = _downed_pair(lie)
  assert peer.down()
  body = peer.mission.footprint_centre()
  said = peer.mission.pose_xy()
  assert math.dist(body, said) > 1.5, "the premise: the reckoner left the body"
  keep = me.mission.others[0]()
  assert keep.down and keep.cells == DOWN_ROBOT_CELLS
  x, y = keep.x, keep.y
  assert (x, y) == pytest.approx(me.mission.as_seen(*body), abs=1e-9)
  assert math.dist((x, y), body) < 1e-3       # a driver that knows where it is
  path = me.mission._plan_to(*GOAL)
  assert path is not None and math.dist(path[-1], GOAL) < 0.1
  # The plan keeps this robot's centre a swing (0.35 m, the map's own
  # inflation) off every part of the body, less one cell of rasterising.
  near = min(_clearance(p, peer) for p in path)
  assert near > 0.35 - me.mission.grid.resolution, \
    f"on its {lie}: a waypoint {near:.3f} m from the body"
  # ...where the wiring before #365 -- the reported pose, a standing
  # robot's disc -- planned the straight line, through the robot.
  me.mission.others = [peer.mission.pose_xy]
  straight = me.mission._plan_to(*GOAL)
  assert min(_clearance(p, peer) for p in straight) < 0.0


def test_the_body_is_placed_where_the_drivers_own_sensors_would_put_it():
  """The driver's drift cancels: the disc goes where a lump in its depth
  image would land in its map -- seen from where it truly stands, placed
  through where it believes it stands (`HubMission.as_seen`). Placed at the
  body's world position instead, a driver 0.4 m out would plan to pass
  0.4 m nearer the robot than it thinks, and the deployed pair's drifts ran
  0.24-0.55 m against 0.37 m of floor between the detour and the body."""
  me, peer = _downed_pair()
  body = peer.mission.footprint_centre()
  r = me.mission.swap.reckoner
  r.x, r.y, r.theta = r.x + 0.4, r.y - 0.1, r.theta + 0.2
  x, y, _ = me.mission.others[0]()
  # as far ahead, and as far round from its heading, as it truly is
  tx, ty, tth = me.mission.true_pose()
  bx, by, bth = me.mission.pose
  assert math.dist((x, y), (bx, by)) == pytest.approx(
    math.dist(body, (tx, ty)), abs=1e-9)
  assert math.atan2(y - by, x - bx) - bth == pytest.approx(
    math.atan2(body[1] - ty, body[0] - tx) - tth, abs=1e-9)
  assert math.dist((x, y), body) > 0.3, "the premise: the driver has drifted"
  # ...and a standing robot is where it SAYS it is, for everyone
  peer.stand_up("a test", auto=False)
  assert me.mission.others[0]() == peer.mission.pose_xy()


def test_a_robot_lying_down_is_not_held_for():
  """The one line at the seam: the depth camera's peer channel sees a
  robot lying down (120-590 points of it at 0.5-1.2 m, every orientation
  measured), and the hold that channel drives is for a robot that will
  MOVE. One on the floor will not, and a detour that runs straight at it
  and turns at its disc's edge was held 0.52 m short until the drive timed
  out."""
  me, peer = _downed_pair("front", near_field=True)
  me.mission.start_at(-0.9, 0.0, 0.0)            # its body 0.5 m ahead
  frame = me.depth_camera.frame(me.data)
  assert me.mission.peer_ahead(frame.peers) is not None, \
    "the premise: the camera sees the robot on the floor in the corridor"
  me._next_near_field = 0.0
  me._near_field_step()
  assert me.mission.peer_sighting() is None, "held for a robot lying down"
  # ...stood up where it lay, it is a robot that moves again, and held for
  adr = SECOND.qpos_adr(me.model)
  me.data.qpos[adr + 2] = 0.045
  me.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
  mujoco.mj_forward(me.model, me.data)
  assert not peer.down()
  me._next_near_field = 0.0
  me._near_field_step()
  assert me.mission.peer_sighting() is not None


def test_standing_up_lets_go_of_where_it_lay():
  """The stand-up is a warp to the start pose that resets the reckoner:
  from then on the robot is avoided where it says it is, with a standing
  robot's disc, and the floor where it lay is floor again."""
  me, peer = _downed_pair()
  peer.stand_up("a test", auto=False)
  assert not peer.down()
  assert me.mission.others[0]() == peer.mission.pose_xy()
  assert math.dist(peer.mission.pose_xy(), peer.home_pose[:2]) < 0.01
  path = me.mission._plan_to(*GOAL)
  assert max(abs(y) for _, y in path) < 0.1, "still routing round the old spot"


def test_a_goal_beside_a_robot_lying_down_is_out_of_reach_further_off():
  """#313's arithmetic, with the wider disc: A* may plan no nearer the
  goal than `radius - distance`, and a stagnated drive is arrived only
  inside CLOSE_ENOUGH_M -- so a goal is out of reach 0.45 m round a
  standing robot and 0.55 m round a lying one. `peer_on_the_goal` must say
  what the planner does, or a bay would be waited for that can be reached,
  and one tried for ever that cannot."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  m.grid.grid[:] = -5.0
  gx, gy = 2.0, 1.0
  res = m.grid.resolution
  assert OTHER_ROBOT_CELLS * res - CLOSE_ENOUGH_M < 0.5 \
      < DOWN_ROBOT_CELLS * res - CLOSE_ENOUGH_M
  for standing, reachable in ((True, True), (False, False)):
    m.others = [(lambda: (gx + 0.5, gy)) if standing
                else (lambda: KeepClear(gx + 0.5, gy, down=True))]
    near = m.peer_on_the_goal(gx, gy)
    short = math.dist(m._plan_to(gx, gy)[-1], (gx, gy))
    assert (near is None) is reachable, near
    assert (short < CLOSE_ENOUGH_M) is reachable, f"{short:.3f} m short"


def test_a_drive_does_not_wait_on_a_robot_lying_down():
  """A stagnated drive WAITS for another robot near it or its goal: stand
  still, let it move. One on the floor will not until the world stands it
  up, and a drive whose goal lay in its disc stood beside it for its whole
  timeout. It plans round the body instead, or ends as a drive with
  nowhere to go does."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  m = HubMission(model, mujoco.MjData(model), viewer=None, realtime=False)
  m.start_at(1.0, 1.0, 0.0)
  m.others = [lambda: (2.6, 1.2)]
  assert m._other_in_the_way(2.6, 1.2), "the premise: a robot standing there"
  m.others = [lambda: KeepClear(2.6, 1.2, down=True)]
  assert not m._other_in_the_way(2.6, 1.2), "waited on a robot lying down"


def test_a_fall_and_a_stand_up_are_planned_round_at_once():
  """The camera does not hold for a robot on the floor, so the plan hears
  of a fall within `DOWN_CHECK_S` rather than at its next 2 s replan --
  MEASURED without that, a robot knocked flat 0.5 m ahead just after a
  replan was met by the LIDAR's 0.25 m stop -- and of a stand-up likewise,
  or it routes round an empty floor for up to 2 s. A 1.5 s drive has room
  for one routine plan and no second, so every later plan is one of these."""
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  m = HubMission(model, data, viewer=None, realtime=False)
  m.start_at(-1.6, 0.0, 0.0)
  m.grid.grid[:] = -5.0
  lying = [False]
  m.others = [lambda: KeepClear(0.5, 1.5, down=lying[0])]   # off the path
  plans, real = [], m._plan_to

  def plan(*a, **kw):
    plans.append(float(data.time))
    return real(*a, **kw)

  m._plan_to = plan
  t0, flips = float(data.time), []

  def flip() -> None:                        # falls at 0.5 s, stood up at 1.0 s
    due = (0.5, 1.0)
    if len(flips) < 2 and data.time - t0 >= due[len(flips)]:
      lying[0] = not lying[0]
      flips.append(float(data.time))

  m.step_hooks.append(flip)
  m.drive_to(*GOAL, timeout=1.5)
  assert len(flips) == 2
  for when in flips:
    after = [t for t in plans if t >= when]
    assert after and after[0] - when <= DOWN_CHECK_S + 0.01, \
      f"planned {[round(t - t0, 2) for t in plans]}, flipped at {when - t0:.2f}"


def test_history_says_a_robot_on_the_floor_is_lying_there():
  """A line saying a robot on the floor STOOD at the bay it blocked reads
  as a robot in the way on purpose, and History is what the mind reads
  back (`posture`, shared with the `WAIT:` narration)."""
  me, peer = _downed_pair()
  blocked = (peer.robot_name or peer.root, 0.4)
  assert "was lying knocked over 0.40 m from the bay" in me.held_for(blocked)
  peer.stand_up("a test", auto=False)
  assert "was standing 0.40 m from the bay" in me.held_for(blocked)


def test_a_driving_robot_goes_round_a_robot_lying_face_down():
  """The integration, in the deployed configuration: the near-field
  camera on, whose peer channel sees a robot lying down. Before #365 the
  camera held the drive ~0.6 m short and the drive stood there until it
  gave up, because the plan it held on went through the body -- MEASURED,
  0 of 8 lying poses arrived. With the disc on the body and no hold, 8 of
  8; face-down is the pose the hold alone still stopped. Stopped on the
  claim: arrival."""
  me, peer = _downed_pair("front", near_field=True)
  mine = np.fromiter(robot_geoms(me.model, FIRST.root), dtype=np.int32)
  theirs = np.fromiter(robot_geoms(me.model, SECOND.root), dtype=np.int32)
  touched = []

  def watch() -> None:
    g = me.data.contact.geom[:me.data.ncon]
    if (np.isin(g[:, 0], mine) & np.isin(g[:, 1], theirs)
        | np.isin(g[:, 1], mine) & np.isin(g[:, 0], theirs)).any():
      touched.append(float(me.data.time))

  me.mission.step_hooks.append(watch)
  assert me.mission.drive_to(*GOAL, timeout=40.0), "never got past it"
  assert not touched, f"touched the robot on the floor at t={touched[0]:.2f}"
