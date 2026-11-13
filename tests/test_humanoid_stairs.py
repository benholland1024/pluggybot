"""The humanoid on stairs, falling and getting up (issue #507, stage 5): what
it sees of the ground ahead (`humanoid.scan`), how its legs' odometry keeps
its height and its planted foot on a flight (`humanoid.odometry`), when it
counts itself as falling (`humanoid.getup`), and an arm that meets its own
trunk. Each rule pinned as cheaply as it fails for the right reason: a
synthetic stance, a map written by hand, a pose set and forwarded. SimNotes,
"The humanoid climbs, falls and gets up", carries the story."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import SelfModel
from pluggybot.humanoid.odometry import StanceOdometry
from pluggybot.humanoid.scan import HeadScan
from pluggybot.perception.depth import DepthCamera

#: The waist's reading at its zero, where these tests' bodies stand (#560's stage (c3):
#: the head's sensors ride the waist, and a frame is laid through its reading).
W0 = {"waist_yaw": 0.0}

#: The get-up's allowance in the test, s: it stood from its back in 3.0 s
#: (`humanoid_spike.py --view getup`).
GETUP_S = 6.0


class _Stance(StanceOdometry):
  """The odometry with its soles set by the test, not posed off encoders:
  each tick's (sole middles, corners) relative to the pelvis, level frame."""

  def __init__(self, ticks, **kw):
    super().__init__(SelfModel(), _Level(), **kw)
    self.ticks = iter(ticks)

  def _soles(self, q, gravity_b):
    mids, corners = next(self.ticks)
    self.corners = np.array(corners, dtype=float)
    self.low = self.corners[:, :, 2].min(axis=1)
    self.flat = np.array([True, True])
    return np.array(mids, dtype=float)


class _Level:
  """An IMU that says the trunk is level, facing +x."""

  def yaw(self) -> float:
    return 0.0


def _sole(x: float, y: float, z: float):
  """A sole's middle and its four bottom corners, flat at height z."""
  mid = [x, y, z]
  corners = [[x + a * 0.11, y + b * 0.045, z] for a in (-1, 1) for b in (-1, 1)]
  return mid, corners


def _tick(left, right):
  (ml, cl), (mr, cr) = _sole(*left), _sole(*right)
  return [ml, mr], [cl, cr]


def test_on_a_stair_the_planted_foot_is_the_one_on_its_ground_not_the_lower():
  # Climbing: the leading (left) foot is planted on a tread 0.18 m up, the
  # trailing (right) one on the floor; then the trailing foot lifts 3 cm.
  # It is still the LOWER foot -- stage 4's rule kept it planted, and the
  # height sank a riser a step -- but it is clear of its ground, and the
  # leading foot is on its own. The mapped ground: the tread under the left
  # sole, the floor under the right.
  ground = lambda x, y, yaw: 0.18 if x > 0.1 else 0.0  # noqa: E731
  ticks = [_tick((0.2, 0.1, -0.53), (0.0, -0.1, -0.73)),   # the left about to land
           _tick((0.2, 0.1, -0.55), (0.0, -0.1, -0.73)),   # both down
           _tick((0.2, 0.1, -0.55), (0.0, -0.1, -0.70))]   # the right lifting
  odo = _Stance(ticks, z=0.73, terrain=ground)
  odo.step(None, None)
  odo.step(None, None)
  assert odo.foot == 1                          # both down: it stays
  odo.step(None, None)
  assert odo.foot == 0, "the lifting trailing foot was kept planted"
  assert odo.z == pytest.approx(0.73)           # the soles did not move it
  # The same soles with no map: the lower foot, as stage 4 had it.
  blind = _Stance(ticks)
  for _ in range(3):
    blind.step(None, None)
  assert blind.foot == 1


def test_the_height_is_never_laid_off_the_map():
  # Landing on a tread, the planted foot changes -- and once the height was
  # laid off the ground the map had under the new sole. A footprint believed
  # a few cm up the flight read the next tread (0.54 m where the foot landed
  # on 0.36), and the height jumped a riser. The planted sole did not move:
  # neither does the height.
  ground = lambda x, y, yaw: 0.54 if x > 0.1 else 0.0  # noqa: E731
  ticks = [_tick((0.2, 0.1, -0.10), (0.0, -0.1, -0.73)),   # the left swung high
           _tick((0.2, 0.1, -0.37), (0.0, -0.1, -0.70))]   # it lands; the right lifts
  odo = _Stance(ticks, z=0.73, terrain=ground)
  odo.step(None, None)
  odo.step(None, None)
  assert odo.foot == 0
  assert odo.z == pytest.approx(0.73), "the height was laid off the map's tread"


def test_a_sole_tipped_over_an_edge_is_followed_by_its_heel():
  # Stepping down off the landing (0.9 m), the planted sole tips over its
  # edge: its heel stays on the landing and its toe hangs over the tread
  # below (0.72). Followed by its lowest corner -- the hanging toe -- the
  # height rose 9 cm as the body stepped down 10; the heel did not move.
  edge = 0.25
  ground_at = lambda x, y: 0.9 if x < edge else 0.72  # noqa: E731

  def tipped(toe_drop):
    mid = [0.2, -0.1, -0.73 - toe_drop / 2]
    heel = [[0.2 - 0.11, -0.1 + b * 0.045, -0.73] for b in (-1, 1)]
    toe = [[0.2 + 0.11, -0.1 + b * 0.045, -0.73 - toe_drop] for b in (-1, 1)]
    return mid, [heel[0], heel[1], toe[0], toe[1]]

  (m0, c0), (m1, c1) = tipped(0.02), tipped(0.10)
  left_mid, left_corners = _sole(0.0, 0.1, -0.60)        # the other foot, swinging high
  ticks = [([left_mid, m0], [left_corners, c0]), ([left_mid, m1], [left_corners, c1])]
  odo = _Stance(ticks, z=1.63, ground_at=ground_at)
  odo.step(None, None)
  odo.step(None, None)
  assert odo.foot == 1
  assert odo.z == pytest.approx(1.63), "followed by its hanging toe"


def test_on_one_floor_the_lower_foot_rule_stands():
  # Both soles over ground of one height (a floor's, within `STEP_M`): the
  # clearance rule would swap on the map's centimetre of noise, and read a
  # walk 10 % short; the lower foot it is.
  ground = lambda x, y, yaw: 0.008 if x > 0.1 else 0.0  # noqa: E731
  ticks = [_tick((0.2, 0.1, -0.73), (0.0, -0.1, -0.735)),
           _tick((0.2, 0.1, -0.725), (0.0, -0.1, -0.735))]
  odo = _Stance(ticks, z=0.73, terrain=ground)
  odo.step(None, None)
  odo.step(None, None)
  assert odo.foot == 1


@pytest.fixture(scope="module")
def head():
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def _scan(head):
  model, data = head
  return HeadScan(model, data, belief=lambda: (0.0, 0.0, 0.73, 0.0))


def test_the_ground_under_a_sole_is_only_what_was_seen_or_stood_on(head):
  # One cell seen 0.6 m off, high (a wall's top): the scan's fill reads it
  # for every point unseen -- and once the ground under a sole read it too,
  # and the height it anchored at boot was 0.67 m up.
  s = _scan(head)
  m = s.map
  m.recentre(0.0, 0.0)
  ix, iy = int(math.floor(0.6 / m.cell_m)) - m.origin[0], -m.origin[1]
  m.height[iy, ix] = 0.67
  s._redraw()
  assert s.scan()[0] == pytest.approx(0.73 - 0.67)   # the fill: nearest seen
  assert s.under(0.0, 0.0, 0.0) is None
  s.feel(0.0, 0.0, 0.02, 0.0)                        # a sole planted there
  assert s.under(0.0, 0.0, 0.0) == pytest.approx(0.02)


def test_a_frames_height_is_registered_cell_against_cell(head):
  # A floor laid from noisy points: each cell keeps its HIGHEST. A frame of
  # the same floor, as noisy, compared point by point against those maxima,
  # read the belief ~7 mm too low every frame and ratcheted it up 9 cm in
  # 7 s of walking; its own maxima against them read nothing.
  s = _scan(head)
  rng = np.random.default_rng(0)
  xy = rng.uniform(0.5, 2.0, (40000, 2))

  def floor():
    return np.column_stack([xy, rng.normal(0.0, 0.005, len(xy))])

  s.map.update((0.0, 0.0, 0.0), floor())
  assert abs(s._drift(0.0, 0.0, 0.0, floor())) < 0.002


def _frame(rng, x, y, z, n):
  """`n` points uniform over the box x * y * z (each a (lo, hi)), world
  heights, in the robot's frame at the origin facing +x."""
  return np.column_stack([rng.uniform(*x, n), rng.uniform(*y, n), rng.uniform(*z, n)])


def test_a_wall_beside_the_way_is_not_ground_the_walker_is_shown(head):
  # The house's flight runs between the hall's wall and the deck's face.
  # Training had no walls; shown them as they stand, the walker fell over
  # backwards before the first riser. A floor with a wall's face 0.5 m to
  # the left, seen only to 0.7 m. And a sole set against the wall stands on
  # the floor: read as on the wall's top, the odometry's height would jump
  # by the wall's (the span rule; the scan alone is the jump's, below).
  s = _scan(head)
  rng = np.random.default_rng(1)
  floor = _frame(rng, (-0.6, 1.2), (-0.6, 0.6), (0.0, 0.0), 40000)
  wall = _frame(rng, (-0.6, 1.2), (0.46, 0.56), (0.0, 0.7), 20000)
  s.map.update((0.0, 0.0, 0.0), np.vstack([floor, wall]))
  s._redraw()
  assert np.allclose(s.scan(), 0.73, atol=0.005), "the wall was shown as ground"
  assert s.under(0.0, 0.5, 0.0) < 0.01, "a sole against the wall stood on its top"


def test_ground_too_high_to_step_onto_from_here_is_not_shown(head):
  # The house's counter's top seen from the floor, 0.6-1.2 m ahead and 0.9 m
  # up: its cells hold only its top, no wall's face, and by height alone it
  # is a flight's fifth tread. Behind the jump at its edge, it is not ground
  # the walker could step onto.
  s = _scan(head)
  rng = np.random.default_rng(2)
  floor = _frame(rng, (-0.6, 0.6), (-0.6, 0.6), (0.0, 0.0), 30000)
  top = _frame(rng, (0.6, 1.2), (-0.6, 0.6), (0.9, 0.9), 30000)
  s.map.update((0.0, 0.0, 0.0), np.vstack([floor, top]))
  s._redraw()
  assert np.allclose(s.scan(), 0.73, atol=0.005), "the counter's top was shown as ground"


def test_a_sole_beside_a_half_wall_stands_on_the_ground_not_on_the_walls_top(head):
  # From the landing the half wall beside the flight shows its top alone:
  # its cells hold no face, so the span rule cannot tell it, and a sole set
  # beside it, two thirds of its footprint over the top, read the ground
  # under it 1.4-1.6 m up. The top is not joined to the ground it stands on.
  s = _scan(head)
  rng = np.random.default_rng(3)
  floor = _frame(rng, (-0.6, 1.2), (-0.6, 0.12), (0.0, 0.0), 40000)
  top = _frame(rng, (-0.6, 1.2), (0.12, 0.22), (0.9, 0.9), 20000)
  s.map.update((0.0, 0.0, 0.0), np.vstack([floor, top]))
  s._redraw()
  assert s.under(0.0, 0.12, 0.0) < 0.01, "a sole beside the half wall stood on its top"


def test_the_camera_sees_the_floor_far_from_where_its_plane_is_anchored(head):
  # MuJoCo's cast skips a geom whose bounding sphere lies past its cutoff,
  # and a plane's is its anchor point: 4 m from the bare floor's, a camera
  # cast to its range saw no floor, and the seeing walker walked on a frozen
  # map from 2.75 m out and fell.
  model, _ = head
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[0] = 4.0
  mujoco.mj_forward(model, data)
  assert len(DepthCamera(model, mount="body").frame(data, W0).points) == 0
  assert len(HeadScan(model, data).cam.frame(data, W0).points) > 1000


def test_a_body_folded_into_a_squat_is_falling_though_it_is_upright(head):
  # Shoved hard from the front, the standing policy folded straight down,
  # its trunk tilted 32 deg -- under the 35 the IMU alone waits for -- and
  # its hands struck the floor.
  from pluggybot.humanoid.getup import FallWatch
  from pluggybot.humanoid.policy import TrunkSense
  from pluggybot.humanoid.senses import JointSense
  model, data = head
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  mujoco.mj_forward(model, d)
  watch = FallWatch(TrunkSense(model, d), JointSense(model))
  assert not watch.falling(d)
  for s in hm.SIDES:
    for j, v in (("hip_pitch", -2.0), ("knee", 2.4), ("ankle_pitch", -0.4)):
      d.qpos[model.jnt_qposadr[model.joint(f"{s}_{j}").id]] = v
  mujoco.mj_forward(model, d)
  assert watch.trunk.attitude.tilt() < math.radians(5.0)
  assert watch.falling(d)


def _turn(axis, angle: float) -> np.ndarray:
  """A rotation about `axis` by `angle`, rad (Rodrigues)."""
  a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
  return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_a_tip_is_read_off_the_posture_its_legs_hold_whichever_way():
  # #520: the IMU is on the pelvis and the body has no waist, so a bow
  # tilts the IMU by the whole bow. A tip is read off the posture the legs
  # hold: about any level axis, sideways as much as forward, by the angle
  # the body turned. At the stand it IS the IMU's tilt, to the bit, so a
  # fall shoved from the stand is taken at the step it always was.
  from pluggybot.humanoid.getup import tipped
  from pluggybot.perception.imu import Attitude
  down = np.array([0.0, 0.0, -1.0])
  for pitch in (0.0, 0.55, 1.1):
    bowed = _turn((0.0, 1.0, 0.0), pitch)
    assert tipped(bowed.T @ down, pitch) == pytest.approx(0.0, abs=1e-9)
    for azimuth in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
      tip = _turn((math.cos(azimuth), math.sin(azimuth), 0.0), 0.6) @ bowed
      assert tipped(tip.T @ down, pitch) == pytest.approx(0.6, abs=1e-9), (pitch, azimuth)
  rng = np.random.default_rng(520)
  for _ in range(100):
    q = rng.normal(size=4)
    att = Attitude(q / np.linalg.norm(q))
    assert tipped(att.matrix().T @ down, 0.0) == att.tilt()


def _posed(model, height: float, pitch: float):
  """The body standing as the standing policy holds a posture: the
  statue's legs for it (`reach.static_legs`) and the trunk pitched."""
  from pluggybot.humanoid.reach import static_legs
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  for name, v in static_legs(height, pitch).items():
    d.qpos[model.jnt_qposadr[model.joint(name).id]] = v
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  d.qpos[3:7] = quat
  mujoco.mj_forward(model, d)
  return d


def test_a_bow_its_legs_hold_is_no_fall_however_deep(head):
  # #520: read off upright, every posture a reach takes past 35 deg -- a
  # low shelf's 0.8 rad, the floor's 1.1 -- was a fall, and the get-up
  # took the joints. Read off the posture the legs hold, none is.
  from pluggybot.humanoid.getup import FALL_TILT_RAD, FallWatch
  from pluggybot.humanoid.policy import TrunkSense
  from pluggybot.humanoid.reach import POSTURES
  from pluggybot.humanoid.senses import JointSense
  model, _ = head
  deep = []
  for height, pitch in POSTURES:
    d = _posed(model, height, pitch)
    watch = FallWatch(TrunkSense(model, d), JointSense(model))
    assert not watch.falling(d, pitch), (height, pitch)
    if pitch > FALL_TILT_RAD:
      deep.append(pitch)
      assert watch.falling(d, 0.0), (height, pitch)         # off upright: the premise
  assert deep


def test_the_mission_reads_a_tip_off_the_posture_its_legs_hold():
  # The watch's posture is the standing policy's while it has the legs --
  # the trim its ask carries taken out, so upright at the stand to the bit,
  # and none at the deepest bow, where the ask is held at the policy's
  # range -- and upright under the walker.
  from pluggybot.body import body_for
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  m = body.mission
  body.run(body.settle_routine(0.5))
  assert m.reach.trim[1] != 0.0 and m.reach.held_pitch() == 0.0
  bowed = _posed(model, -0.25, 1.1)
  data.qpos[:], data.qvel[:] = bowed.qpos, 0.0
  mujoco.mj_forward(model, data)
  m.fresh_drivers()
  m.reach.trim[1] = 0.05                                    # an ask past 1.1 rad is held at it
  m.reach.hold_posture(-0.25, 1.1)
  m.stand.posture[:] = m.stand.goal                         # shown the bow it holds
  assert m.reach.held_pitch() == pytest.approx(1.1)         # ...so the trim is carried by none
  m._steps = 0                                              # the watch looks next step
  body.run(body.settle_routine(model.opt.timestep))
  assert m.falls == 0 and m.legs == "stand"
  m.legs, m._steps = "walk", 0
  body.run(body.settle_routine(model.opt.timestep))
  assert m.falls == 1 and m.legs == "getup"


def test_standing_across_two_treads_is_standing(head):
  # Up off the floor on a flight, a foot on each of two treads: over its
  # soles' middle the pelvis stood half a riser short of the stand, and the
  # get-up would never hand the legs back. Over the sole it stands on, it
  # is the stand's height.
  from pluggybot.humanoid.getup import STOOD_M, over_lower_sole
  from pluggybot.humanoid.policy import TrunkSense
  from pluggybot.humanoid.senses import JointSense
  model, _ = head
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  for j, v in (("right_hip_pitch", -0.9), ("right_knee", 1.7), ("right_ankle_pitch", -0.8)):
    d.qpos[model.jnt_qposadr[model.joint(j).id]] = v
  mujoco.mj_forward(model, d)
  sm = SelfModel()
  height = over_lower_sole(sm, JointSense(model).read(d)[0], TrunkSense(model, d).gravity())
  bottoms = [sm.d.geom_xpos[g][2] - sm.m.geom_size[g][2] for g in sm.soles]
  assert abs(bottoms[0] - bottoms[1]) > 0.15                      # a riser between them
  assert abs(sm.d.qpos[2] - hm.stand_height()) > STOOD_M          # the middle: never stood
  assert height == pytest.approx(hm.stand_height(), abs=STOOD_M)


def test_an_arm_meets_its_own_trunk(head):
  # The robot's own geoms touch nothing of the robot's but the pairs the
  # model names: an arm swung across the chest passed through it, and a
  # body getting up off the floor would push off nothing a real arm could.
  model, _ = head
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  for j, v in (("left_shoulder_pitch", -1.4), ("left_shoulder_roll", 0.0),
               ("left_shoulder_yaw", -1.6), ("left_elbow", 2.0)):
    d.qpos[model.jnt_qposadr[model.joint(j).id]] = v
  mujoco.mj_forward(model, d)
  geoms = {model.geom(n).id for n in ("left_upper_arm", "left_forearm", "left_gripper")}
  trunk = {model.geom(n).id for n in hm.TRUNK_GEOMS}
  met = [(c.geom1, c.geom2) for c in d.contact[:d.ncon]
         if {c.geom1, c.geom2} & geoms and {c.geom1, c.geom2} & trunk]
  assert met, "the arm passed through the trunk"


class _Rising:
  """A get-up that stands the body after `n` steps, whatever it does."""

  def __init__(self, n: int):
    self.n, self.steps = n, 0

  def begin(self) -> None:
    pass

  def step(self) -> None:
    self.steps += 1

  @property
  def stood(self) -> bool:
    return self.steps >= self.n


def test_a_walk_asked_while_it_gets_up_waits_until_it_stands(head):
  # A fall mid-walk, the arms home: the walk's next step asks the legs back
  # for the walker, and found the arms at rest and the posture settled --
  # so it took them off the get-up policy on the floor. It waits.
  from pluggybot.body import body_for
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  m = body.mission
  m.getup, m.legs = _Rising(50), "getup"
  body.run(m.to_walk_routine())
  assert m.getup.steps == 50, "the walker took the legs before the body stood"
  assert m.legs == "walk"


def test_a_fall_while_the_walker_settles_is_the_getups_until_it_stands(head):
  # The legs come back from the walker after it has stood still SETTLE_S. A
  # fall in that wait went to the get-up -- and at the wait's end to the
  # standing policy anyway, mid-fall (#520).
  from pluggybot.body import body_for
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  m = body.mission
  m.legs, m.getup = "walk", _Rising(600)                    # longer than the settle's 400
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  body.run(m.to_stand_routine())
  assert m.getup.steps == 600, "the standing policy took the legs before the body stood"
  assert m.legs == "stand" and m.falls == 1


def test_a_fall_as_the_arms_come_home_is_the_getups_until_it_stands(head):
  # The legs go to the walker once the arms are home and the posture has
  # settled. A fall on the step that settled it went to the walker (#520).
  from pluggybot.body import body_for
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  m = body.mission
  m.getup = _Rising(50)
  m.stand.posture[:] = (0.0, 0.03)                          # not settled yet
  falls = iter([True])                                      # the fall: on the settling step
  m.fall.falling = lambda data, pitch=0.0: m.stand.settled and next(falls, False)
  body.run(m.to_walk_routine())
  assert m.getup.steps == 50, "the walker took the legs before the body stood"
  assert m.legs == "walk" and m.falls == 1


def test_a_stand_up_keeps_what_the_grippers_were_asked(head):
  # No policy drives the grippers, so a fall leaves them as they were asked;
  # the stand-up's fresh reach built its own, and a gripper shut on a knob
  # read as never shut (`Gripper.held` asks `shut_asked`, #520).
  from pluggybot.body import body_for
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  m = body.mission
  m.reach.grippers["left"].shut()
  m.getup = _Rising(5)
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  body.run(body.settle_routine(model.opt.timestep))         # the watch takes a fall
  assert m.legs == "getup"
  body.run(m._getup_routine())                              # ...and it stands
  assert m.legs == "stand" and m.reach.phase["left"] == "rest"
  assert m.reach.grippers["left"].shut_asked


def test_the_getup_policy_stands_it_up_off_its_back(head):
  # Its get-up policy (`humanoid.getup`) in our physics, off its own senses:
  # lying on its back, it stands -- upright, its pelvis at the stand's
  # height over its soles, still -- and by our grading off the truth.
  from pluggybot.humanoid.getup import GetupDriver
  model, _ = head
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  q = np.zeros(4)
  mujoco.mju_euler2Quat(q, np.array([0.0, -math.pi / 2, 0.0]), "XYZ")
  d.qpos[2], d.qpos[3:7] = 0.15, q
  mujoco.mj_forward(model, d)
  g = GetupDriver(model, d)
  g.begin()
  for _ in range(int(GETUP_S / model.opt.timestep)):
    g.step()
    mujoco.mj_step(model, d)
    if g.stood:
      break
  root = model.body(hm.ROBOT_ROOT).id
  assert g.stood, f"still down after {GETUP_S:g} s"
  assert d.xmat[root][8] > 0.95 and d.xpos[root][2] > 0.65


#: How the body lies along a counter's face (its front's way and its head's,
#: in the face's terms: n off it into the room, a along it), and its nearest
#: part's gap to it, m -- two of #679's lying table's falls.
ALONG_A_COUNTER = {"on its front, head back, 0.1 m off": ("-z", "-a", 0.1),
                   "on its side facing it, 0.25 m off": ("-n", "+a", 0.25)}


@pytest.mark.parametrize("lying", sorted(ALONG_A_COUNTER))
def test_the_getup_stands_it_up_lying_along_a_counter(lying):
  # #679: the get-up trained on open floor (getup3, served until then) never
  # stood a body lying along the storeroom's counter like this -- 20 s on, a
  # `stuck` death -- and the served one, trained beside a box and handing
  # back as served, stands in 2-4 s; here on a bare floor beside a box the
  # counter's size, the waist held home as the mission holds it.
  from pluggybot.evaluation.counter import COUNTER_D, COUNTER_H
  from pluggybot.humanoid.body import HumanoidBody
  from pluggybot.humanoid.getup import GetupDriver
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  face_x = 0.6
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[COUNTER_D / 2, 1.2, COUNTER_H / 2],
                          pos=[face_x + COUNTER_D / 2, 0.0, COUNTER_H / 2])
  model = spec.compile()
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  front, head, gap = ALONG_A_COUNTER[lying]
  axes = {"-n": (1.0, 0.0, 0.0), "+a": (0.0, 1.0, 0.0), "-a": (0.0, -1.0, 0.0),
          "-z": (0.0, 0.0, -1.0)}
  front, head = np.array(axes[front]), np.array(axes[head])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.column_stack([front, np.cross(head, front), head]).flatten())
  root = model.body(hm.ROBOT_ROOT).id
  adr = model.jnt_qposadr[model.body_jntadr[root]]
  d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = [face_x - 1.0, 0.0, 0.4], quat
  mujoco.mj_forward(model, d)
  # ...its lowest part 1 cm over the floor and its nearest `gap` off the face,
  # by the geoms that collide (`humanoid_spike.py`'s lying table's)
  ours = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0
          and (model.geom_contype[g] or model.geom_conaffinity[g])]
  boxes = [(d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3],
            np.abs(d.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]) for g in ours]
  low, near = min(c[2] - h[2] for c, h in boxes), max(c[0] + h[0] for c, h in boxes)
  d.qpos[adr:adr + 3] += [face_x - gap - near, 0.0, 0.01 - low]
  mujoco.mj_forward(model, d)
  stand = StandDriver(model, d)
  reach = Reach(model, d, stand)
  g = GetupDriver(model, d, trunk=stand.trunk, joints=stand.joints)
  g.begin()
  while d.time < HumanoidBody.GETUP_BUDGET_S and not g.stood:
    reach.hold_waist()
    g.step()
    mujoco.mj_step(model, d)
  assert g.stood, f"still down {HumanoidBody.GETUP_BUDGET_S:g} s on: the stuck death"
  assert d.xmat[root][8] > 0.95 and d.xpos[root][2] > 0.65


def _getups():
  from pluggybot.humanoid.getup import GETUP_NPZ
  return sorted(p.name for p in GETUP_NPZ.parent.glob("humanoid_getup*.npz"))


def test_the_getup_charges_the_head_as_built(head):
  # #679: a get-up's training charges a hit on the head's parts (`HEAD`, in
  # `training/`, which cannot import this package: read off its source) --
  # matched whole, as mjlab's sensors match a pattern -- and they are the
  # head's parts as the body builds them (`model.HEAD_GEOMS`): getup3 trained
  # when the head was one cylinder, and charged that one alone.
  import ast
  import re
  from pathlib import Path
  model, _ = head
  src = Path(__file__).resolve().parents[1] / "training" / "humanoid_train" / "getup.py"
  patterns = next(ast.literal_eval(node.value) for node in ast.parse(src.read_text()).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "HEAD" for t in node.targets))
  names = {model.geom(g).name for g in range(model.ngeom)}
  struck = {n for n in names if any(re.fullmatch(p, n) for p in patterns)}
  assert struck == set(hm.HEAD_GEOMS)


def test_the_wall_getup_trains_handing_back_as_the_served_getup_hands_back():
  # #679's third round: training hands the legs to the standing policy where
  # the served get-up does (`getup.STOOD_*`) and takes them back where the
  # served watch would (`FALL_TILT_RAD`, `COLLAPSED_M`) -- read off its source,
  # as `training/` cannot import this package. Never handed back in training,
  # round 2's pick stood with its soles 0.6 m apart, and the standing policy
  # given that stance fell within 2 s.
  import ast
  from pathlib import Path
  from pluggybot.humanoid import getup as served
  src = Path(__file__).resolve().parents[1] / "training" / "humanoid_train" / "getup.py"
  values = {}
  for node in ast.parse(src.read_text()).body:
    if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], (ast.Name, ast.Tuple)):
      continue
    target, value = node.targets[0], node.value
    names = [t.id for t in target.elts] if isinstance(target, ast.Tuple) else [target.id]
    parts = value.elts if isinstance(value, ast.Tuple) and len(names) > 1 else [value]
    for name, part in zip(names, parts):
      values[name] = part
  # STOOD_UP is cos(radians(deg)): its degrees are the literal inside
  stood_deg = ast.literal_eval(values["STOOD_UP"].args[0].args[0])
  assert stood_deg == pytest.approx(math.degrees(served.STOOD_TILT_RAD))
  read = {k: ast.literal_eval(values[k]) for k in ("HAND_BACK_M", "HAND_BACK_S", "RETAKE_DEG",
                                                  "RETAKE_M")}
  assert read["HAND_BACK_M"] == served.STOOD_M
  assert read["HAND_BACK_S"] == served.STOOD_S
  assert read["RETAKE_DEG"] == pytest.approx(math.degrees(served.FALL_TILT_RAD))
  assert read["RETAKE_M"] == served.COLLAPSED_M


@pytest.mark.parametrize("npz", _getups())
def test_a_getup_sees_the_joints_it_drives(head, npz):
  # #679: a get-up's inputs are the joints it drives (getup3's 78, carried on
  # since the waist), and the served runtime builds them off the joints its
  # file lists -- but mjlab lists every joint of the body (25 since the
  # waist), so `quad_train.export` writes the ones its task's actor names.
  # Exported over every joint, the get-up built 80 inputs for a 78-input net.
  from pluggybot.humanoid.getup import GETUP_NPZ, GetupDriver
  from pluggybot.humanoid.policy import StandPolicy
  model, _ = head
  policy = StandPolicy(GETUP_NPZ.parent / npz)
  assert policy.joint_names == policy.action_joints
  # ...and every list the file keeps a joint at a time lines up with them
  for key in ("default_joint_pos", "joint_stiffness", "joint_damping"):
    assert len(policy.meta[key].split(",")) == len(policy.joint_names), key
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  mujoco.mj_forward(model, d)
  g = GetupDriver(model, d, policy=policy)
  g.begin()
  assert g.observation().shape == policy.mean.shape
  g.step()
