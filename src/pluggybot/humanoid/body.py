"""The humanoid behind `Body` (issue #507, stages 3 and 4): its posture,
its senses and its arms -- what manipulation needs -- over its standing
policy (`humanoid.policy`) and its reach (`humanoid.reach`); and its walk,
over the `Navigator` (`humanoid.mission`: its map, its planner, its leg
odometry and IMU, its walking policy). A member this body cannot serve yet
says why (`NotImplementedError`).

WHERE IT BELIEVES IT IS: its pelvis, off its leg odometry and IMU, matched
to its map (`HumanoidMission.pose`); and its STANCE, the frame between its
soles that its reach works in (`humanoid.arms`), off that pose and where
its legs' encoders put the pelvis over its feet. Through the stance every
point it senses or is sent to goes between the world's frame and its own:
a tag its colour imager decodes, a hand's goal.
"""

import math

import mujoco
import numpy as np

from pluggybot import tick
from pluggybot.body import TOPPLE_HOLD_S, Body
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid.arms import HELD_GAP_M, HandGoal
from pluggybot.humanoid.policy import PITCH_RANGE, StandPolicy
from pluggybot.humanoid.mission import DEPTH_PERIOD, MOTION_EPS, V_MIN, HumanoidMission
from pluggybot.humanoid import reach as reach_
from pluggybot.humanoid.reach import HOLD_SPEED
from pluggybot.perception.depth import DepthFrame
from pluggybot.perception.watch import TagWatch
from pluggybot.power import Pack
from pluggybot.robot import FIRST, RobotHandle

#: The command that holds it where it is: (forward m/s, sideways, turn
#: rad/s), the walking command stage 4's policy tracks.
STILL = (0.0, 0.0, 0.0)
#: Level enough for a reading to be a map's: its IMU's tilt under this.
LEVEL_TILT_RAD = math.radians(5.0)
#: The camera its tags are read off: the D435's colour imager, pitched at a
#: counter and the floor ahead (`model._trunk`), and its eye (#275): the
#: Camera Module 3 on the head, looking ahead.
TAG_CAMERA, HEAD_CAMERA = "color_eye", "nav_eye"
#: How long a gripper is given to close or open, s: the Dex1-1 shuts its
#: stroke in 0.24 s at its fastest, and a squeeze settles after.
GRIP_S = 0.6
#: ...and how much longer, s, while its jaws stand open on something and
#: squeeze it too little to hold, read every `GRIP_READ_S`: a chest's knob
#: on its drop handle swung off a pad and back between the jaws, and read
#: at `GRIP_S` alone the grip said NOT HELD in 7 of 24 set-outs where the
#: pads held the knob 2 s later (#493).
GRIP_SETTLE_S, GRIP_READ_S = 1.5, 0.1
#: A gaze's bow is waited for this long at most, s (`gaze_routine`), and a
#: hand's way out and home this long (`release_routine`).
GAZE_WAIT_S, RELEASE_S = 4.0, 15.0
#: A hand's carry or pull stops once the body's own odometry puts its
#: stance this far from where it began, m: a pull toward the robot dragged
#: it 0.3 m to the counter, its standing policy stepping under the arm
#: (#493).
FEET_MOVED_M = 0.05
#: A STEP (`step_routine`, #525; SimNotes, "Stepping toward the work"): its
#: pace forward and back, m/s -- at the walker's slowest, 0.2, its feet slid
#: rather than lifted and its odometry counted the first stride backward:
#: asked 5 cm it went 16 and believed 5; backing up at 0.3 it went 4 cm and
#: believed it had gone FORWARD 1 -- and its turn, rad/s: at 0.6 the heading
#: sprang back 6-8 deg as the legs changed hands, at 1.0 it lands within 2
#: deg of the turn asked.
STEP_V, BACK_V, TURN_W = 0.3, 0.4, 1.0
#: ...stopped once what it believes it has walked, and its speed times this,
#: s, reach the step asked -- its stride carries on that far, about 0.2-0.27
#: s of its pace -- and a turn this much short, rad.
STEP_STOP_S, TURN_STOP_RAD = 0.22, math.radians(0.5)
#: ...BLOCKED when it walks less than this, m, in this long, s, and given
#: this long, s; and stood still this long after, s, the standing policy's
#: own settle.
STEP_STUCK_M, STEP_STUCK_S, STEP_PATIENCE_S, STEP_SETTLE_S = 0.01, 1.0, 15.0, 0.5
#: Its measured depth's own noise stream (`depth_cloud`) and how far its
#: rays reach, m: the probe's seed, and a counter's chest within arm's reach.
DEPTH_SEED, DEPTH_REACH_M = 466, 3.0
#: A watcher's depth camera's own noise stream (`depth_watched`, #530): never
#: the measured depth's, so watching draws nothing a flight reads.
WATCH_SEED = 530
#: What the humanoid does not do yet answers.
NOT_HERE = "not on the humanoid yet: the quadruped's world (its dock, rack and tools)"


def _yaw(rot: np.ndarray) -> float:
  return math.atan2(rot[1, 0], rot[0, 0])


def _step_walks(ahead: float, left: float) -> list[tuple[float, float]]:
  """A step's walks: `ahead` and `left` as one motion -- or, where its
  forward speed would fall in the walker's dead band (`mission.command_for`
  raises one under `V_MIN` to it, and an oblique step walked 12 cm further
  ahead than asked, #525), the one and then the other."""
  length = math.hypot(ahead, left)
  if not length:
    return []
  vx = (BACK_V if ahead < -abs(left) else STEP_V) * ahead / length
  if abs(vx) < MOTION_EPS or abs(vx) >= V_MIN:
    return [(ahead, left)]
  return [w for w in ((ahead, 0.0), (0.0, left)) if any(w)]


def _rz(yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class HumanoidPack(Pack):
  """The pack's book with the humanoid's draw: its electronics always on
  (`parts.ELECTRONICS_W`), and every winding's heat at the torque its
  driver makes -- holding still does no work."""

  def __init__(self, body: "HumanoidBody", capacity_wh: float, charge_scale: float):
    super().__init__(capacity_wh, charge_scale=charge_scale,
                     draw_w=sum(hp.ELECTRONICS_W.values()))
    self.act = body.stand.joints.act
    motors = [hm.CHOSEN.motor(j) for j in hm.JOINT_NAMES]
    self.kt = np.array([m.kt for m in motors])
    self.r = np.array([m.r_phase for m in motors])

  def power_draw(self, data) -> float:
    i = data.actuator_force[self.act] / self.kt
    return self.draw_w + float((1.5 * self.r * i * i).sum())


class _SharedDepth:
  """Its depth camera as the loop's near-field reads it (#559): the SAME
  frames its planner's layer is laid from, levelled into the robot frame,
  at the camera's own period (`mission.DEPTH_PERIOD`). Its draw is in the
  body's electronics (`parts.ELECTRONICS_W`), so the loop adds none."""

  draw_w = 0.0
  period = DEPTH_PERIOD

  def __init__(self, mission: HumanoidMission) -> None:
    self.mission = mission

  @property
  def rng(self):
    return self.mission.depth.rng

  @property
  def peer_rng(self):
    return self.mission.depth.peer_rng

  def frame(self, data) -> DepthFrame:
    frame = self.mission.latest_depth()
    if frame is None:
      return DepthFrame(z=np.zeros((0, 0)), points=np.zeros((0, 3)), self_fraction=0.0,
                        peers=np.zeros((0, 3)), peer_geoms=np.zeros(0, dtype=np.int32))
    return frame

  def exclude_robot(self, root_name: str) -> None:
    pass


class HumanoidBody(Body):
  """The humanoid (module docstring)."""

  STILL = STILL
  level_tilt_rad = LEVEL_TILT_RAD
  #: How long a fall may last before it is the `stuck` death, s; without
  #: its get-up policy, the topple's hold. MEASURED (`humanoid_spike.py
  #: --fall`): it stood from six shoves on a bare floor in 2.7-4.2 s, and
  #: from 17 of 20 in the house within 15 s, the slowest 11.9 s; of the
  #: three still down, one stood at 30 s, at the foot of the flight it had
  #: fallen down. The quadruped's 20 (its slowest 4.6), the margin for
  #: falls those places do not make.
  GETUP_BUDGET_S = 20.0

  def __init__(self, model, data, handle: RobotHandle = FIRST,
               policy: StandPolicy | None = None, walk: StandPolicy | None = None,
               grid_bounds=(-3, -3, 7, 7), viewer=None, realtime: bool = False):
    self.handle = handle
    self._m, self._d = model, data
    self._policies = (policy, walk)
    self.root = model.body(handle.root).id
    self.mission = HumanoidMission(model, data, handle, stand=policy, walk=walk,
                                   grid_bounds=grid_bounds, viewer=viewer,
                                   realtime=realtime)
    self._tags = None
    #: The tag its colour imager keeps reading while it moves (`watch`).
    self.mission.sight = TagWatch(model.opt.timestep, self._sighted)
    #: A watcher's own joint sense and model of itself (`hand_watched`), and
    #: its own depth camera (`depth_watched`).
    self._watch = self._depth_watch = None
    #: ...an array, as the loop indexes it (#559: a frozenset broke a pair's
    #: restore, `HubLifecycle.unclip`, and the near field's robots lying down)
    self.geom_ids = np.array([g for g in range(model.ngeom)
                              if self._under_root(int(model.geom_bodyid[g]))], dtype=np.int32)
    # The members a standing body answers plainly.
    self.places = None
    self.rack = self.rack_prior = None
    self.rack_discovered = False
    self.swapping_at = None
    self.peer_at_bay_m = None
    #: Why each hand's last reach ended short of holding (`hand_why`), and
    #: the hands whose reach `stop_hand` stopped (#554).
    self._why: dict[str, str] = {}
    self._halted: set[str] = set()
    self.bay_wait = None
    self.docked = False
    self.head_camera = handle.el(HEAD_CAMERA)
    self.collision_steps = self.press_steps = 0

  falls = property(lambda self: self.mission.falls)

  @property
  def rights_itself(self) -> bool:
    """Whether it gets itself up: with its get-up policy, it does."""
    return self.mission.getup is not None

  @property
  def stuck_after_s(self) -> float:
    return self.GETUP_BUDGET_S if self.rights_itself else TOPPLE_HOLD_S

  @property
  def posture(self) -> str:
    """`getting_up` while its get-up policy has the joints, else `standing`
    (it rests standing)."""
    return "getting_up" if self.mission.legs == "getup" else "standing"

  # Its drivers are the mission's: the standing policy's, the walking
  # policy's and the arms'.
  stand = property(lambda self: self.mission.stand)
  walker = property(lambda self: self.mission.walker)
  reach = property(lambda self: self.mission.reach)

  def _under_root(self, body: int) -> bool:
    while body > 0:
      if body == self.root:
        return True
      body = int(self._m.body_parentid[body])
    return False

  # ---- which robot, and the seam -------------------------------------------

  @property
  def model(self):
    return self._m

  @property
  def data(self):
    return self._d

  @property
  def step_hooks(self) -> list:
    return self.mission.step_hooks

  @property
  def stepper(self):
    return self.mission.stepper

  def run(self, routine, name: str = ""):
    return tick.run(self.mission.stepper, routine, name)

  def kept_state(self):
    """What it believes and what its controllers hold, for a restart (#559:
    `HumanoidMission.kept_state`), and whether it rests."""
    return self.mission.kept_state()

  def restore_kept(self, state, arrays) -> bool:
    if "legs" not in state:
      return False                      # saved before a humanoid kept any
    return self.mission.restore_kept(state, arrays)

  @property
  def working(self) -> bool:
    """A hand out (#559): its move is a routine's, which no file holds, so
    a restart's save waits it out (`continuation.Keeper.busy`)."""
    return any(p != "rest" for p in self.reach.phase.values())

  def close(self) -> None:
    if self._tags is not None:
      self._tags.close()
      self._tags = None

  # ---- where it is ---------------------------------------------------------

  def _true_stance(self) -> tuple[float, float, float]:
    """The stance as the simulator has it: the soles' middle on the floor
    and their mean heading -- the sim's own record, never an act."""
    m, d = self._m, self._d
    soles = [m.geom(self.handle.el(f"{s}_sole")).id for s in hm.SIDES]
    heads = [math.atan2(d.geom_xmat[g][3], d.geom_xmat[g][0]) for g in soles]
    yaw = math.atan2(sum(math.sin(h) for h in heads), sum(math.cos(h) for h in heads))
    mid = np.mean([d.geom_xpos[g] for g in soles], axis=0)
    return float(mid[0]), float(mid[1]), yaw

  @property
  def stance(self) -> tuple[float, float, float]:
    """Where it believes its stance is, (x, y, heading): its pelvis where it
    believes it, less where its legs' encoders put the pelvis over its feet
    (its odometry's model of itself, posed every tick)."""
    x, y, yaw = self.mission.pose
    sm = self.mission.odo.sm
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, sm.d.qpos[3:7])
    syaw = yaw - _yaw(rot.reshape(3, 3))
    c, s = math.cos(syaw), math.sin(syaw)
    px, py = float(sm.d.qpos[0]), float(sm.d.qpos[1])
    return x - (c * px - s * py), y - (s * px + c * py), syaw

  def to_stance(self, p) -> np.ndarray:
    """A believed world point in its stance frame."""
    x, y, yaw = self.stance
    return _rz(yaw).T @ (np.asarray(p, dtype=float) - [x, y, 0.0])

  def to_world(self, p) -> np.ndarray:
    """A point in its stance frame in the believed world."""
    x, y, yaw = self.stance
    return _rz(yaw) @ np.asarray(p, dtype=float) + [x, y, 0.0]

  def start_at(self, x: float, y: float, yaw: float) -> None:
    """Upright at (x, y) facing `yaw`, standing, its arms hanging, and told
    so: its odometry there and its drivers fresh. Steps nothing
    (`Body.start_at`); its map stays."""
    m, d = self._m, self._d
    adr = m.jnt_qposadr[m.joint(self.handle.el(f"{hm.ROBOT_ROOT}_root")).id]
    d.qpos[adr:adr + 3] = [x, y, hm.stand_height()]
    d.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    stand = hm.stand_qpos()
    for name in hm.qpos_order():
      d.qpos[m.jnt_qposadr[m.joint(self.handle.el(name)).id]] = stand.get(name, 0.0)
    dadr = m.jnt_dofadr[m.joint(self.handle.el(f"{hm.ROBOT_ROOT}_root")).id]
    d.qvel[dadr:dadr + 6] = 0.0
    for name in hm.qpos_order():
      d.qvel[m.jnt_dofadr[m.joint(self.handle.el(name)).id]] = 0.0
    mujoco.mj_forward(m, d)
    self.mission.fresh_drivers()
    if self.mission.matcher is not None:
      self.mission.matcher.anchored(on=False)

  @property
  def pose(self):
    return self.mission.pose

  def pose_xy(self):
    x, y, _ = self.pose
    return x, y

  def true_pose(self):
    return self.mission.true_pose()

  def root_xy(self):
    return float(self._d.xpos[self.root][0]), float(self._d.xpos[self.root][1])

  def orientation(self):
    return tuple(float(v) for v in self._d.xquat[self.root])

  def footprint_centre(self):
    sx, sy, _ = self._true_stance()
    return sx, sy

  def as_seen(self, x, y):
    return self.mission.as_seen(x, y)

  def from_seen(self, x, y):
    return self.mission.from_seen(x, y)

  def level(self) -> bool:
    return self.mission.level()

  # ---- standing, and the motor level ---------------------------------------

  def hold_routine(self, seconds: float):
    return self.mission._drive_routine(seconds, 0.0, 0.0)

  def settle_routine(self, seconds: float):
    for _ in range(int(round(seconds / self._m.opt.timestep))):
      yield STILL

  def _arm_joint(self, act: int) -> tuple[str, int] | None:
    name = self._m.actuator(act).name[len(self.handle.prefix):]
    for s in hm.SIDES:
      joints = [f"{s}_{j}" for j in hm.ARM]
      if name in joints:
        return s, joints.index(name)
    return None

  def actuator(self, name: str) -> int:
    """An arm joint's driver or a gripper's, by its bare name
    (`left_elbow`, `right_gripper`): its id. The legs are the standing
    policy's, never a caller's."""
    if name in hm.JOINT_NAMES and hm.kind(name) in hm.LEG:
      raise KeyError(f"{name} is the standing policy's")
    return int(self._m.actuator(self.handle.el(name)).id)

  def ramp_routine(self, act: int, target: float, speed: float, settle: float = 0.0):
    """An arm joint walked to `target` at `speed` (its arm's other joints
    held where they are, the arm's weight fed forward), or a gripper's
    jaws to `target` m apart; then settled. ⚠ A fall lets go of the arm and
    ends the ramp (#520): ramped on, it drove the stand-up's fresh reach,
    whose hanging arms a walk's hand-over reads as home."""
    dt = self._m.opt.timestep
    joint = self._arm_joint(act)
    if joint is None:
      yield from self.mission.to_stand_routine()
      g = next(g for g in self.reach.grippers.values() if g.act == act)
      width = self.setpoint(act)
      want = min(max(target, 0.0), hp.GRIPPER.stroke)
      while abs(width - want) > 1e-9:
        width += max(-speed * dt, min(speed * dt, want - width))
        g.open(width)
        yield STILL
    else:
      yield from self.mission.to_stand_routine()
      side, k = joint
      falls = self.mission.falls
      q = self.reach.q_cmd[side].copy()
      self.reach.set_joints(side, q)
      while abs(q[k] - target) > 1e-9:
        if self.mission.falls != falls:
          return
        q[k] += max(-speed * dt, min(speed * dt, target - q[k]))
        self.reach.q_to[side] = q.copy()
        yield STILL
      while not self.reach.done(side):
        if self.mission.falls != falls:
          return
        yield STILL
    yield from self.settle_routine(settle)

  def setpoint(self, act: int) -> float:
    joint = self._arm_joint(act)
    if joint is None:
      return 2 * float(self._d.ctrl[act])
    side, k = joint
    return float(self.reach.q_cmd[side][k])

  def arm_torque(self, act: int) -> float:
    joint = self._arm_joint(act)
    if joint is None:
      raise KeyError(f"actuator {act} is not an arm joint's")
    side, k = joint
    _, _, tau = self.stand.joints.read(self._d)
    return float(tau[hm.JOINT_NAMES.index(f"{side}_{hm.ARM[k]}")])

  def retract_arm_routine(self):
    for s in hm.SIDES:
      if not self.reach.home(s):
        yield from self.release_routine(s)

  # ---- hands ---------------------------------------------------------------

  def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                    turns=None):
    """⚠ True only once the hand HOLDS at the goal, never on `Reach.done`
    (#520): that is also a hand stopped short (`blocked`) and a hand
    hanging, as every hand is once a fall has let go of the reach and the
    stand-up has built a fresh one. A fall ends it at once; the next
    routine waits out the get-up."""
    yield from self.mission.to_stand_routine()
    p = self.to_stance(pos)
    r = None if rot is None else _rz(self.stance[2]).T @ np.asarray(rot, dtype=float)
    self._why[side] = "out of reach"
    self._halted.discard(side)
    postures = reach_.POSTURES
    if path is not None:
      # the posture the hand then follows its path in, and that one alone
      ways = [self.to_stance(q) for q in path]
      if turns is not None and len(turns) != len(ways):
        raise ValueError(f"a turn for each of the path's {len(ways)} points, not {len(turns)}")
      follow = self.reach.path_for(side, [p] + ways,
                                   rot=[r] + self._turns(list(turns)) if turns is not None else r)
      if follow is None:
        return False
      postures = (follow,)
    if not self.reach.reach(side, HandGoal(p, r, stiffness), postures=postures,
                            flip=path is None):
      # ...or where its approach is out of reach in that posture, in any: the
      # hand takes the path's posture once out (`Reach.carry`)
      # -- the grip asked, never half a turn round: the path's turns are its
      if path is None or not self.reach.reach(side, HandGoal(p, r, stiffness), flip=False):
        return False
    falls = self.mission.falls
    for _ in range(int(round(patience / self._m.opt.timestep))):
      if self.mission.falls != falls or self.reach.done(side):
        break
      yield STILL
    held = self.mission.falls == falls and self.reach.phase[side] == "hold"
    self._why[side] = ("" if held else "fell" if self.mission.falls != falls
                       else "until" if side in self._halted
                       else "stopped short" if self.reach.done(side) else "out of time")
    return held

  def hand_why(self, side) -> str:
    return self._why.get(side, "")

  def stop_hand(self, side) -> bool:
    # ⚠ only a hand of the standing body's: waiting out a get-up, the reach a
    # fall let go of still says its hand moves, and stopping that used up a
    # condition the stand-up's fresh reach then ran without
    if self.mission.legs != "stand" or not self.reach.halt(side, "until"):
      return False
    self._halted.add(side)
    return True

  def release_routine(self, side):
    """Out and home, waited for until its drivers read the arm there
    (`Reach.home`, #539), the posture back at the stand: "home"; "fell" -- a
    fall ends it at once, the next routine waiting out the get-up, as a
    reach's does -- or "held", not home in `RELEASE_S`: a hand that cannot
    come home ends the routine there, never the attempt it is in."""
    yield from self.mission.to_stand_routine()
    self.reach.retreat(side)
    falls = self.mission.falls
    for _ in range(int(round(RELEASE_S / self._m.opt.timestep))):
      if self.mission.falls != falls or self.reach.home(side):
        break
      yield STILL
    while self.mission.falls == falls and not self.stand.settled:
      yield STILL
    if self.mission.falls != falls:
      return "fell"
    return "home" if self.reach.home(side) else "held"

  def grip_routine(self, side, closed, force=None):
    """Shut -- squeezing with `force`, N a pad, or its sheet's (#525) -- its
    word is read once its jaws have closed or held (#493: `GRIP_SETTLE_S`),
    never while they stand open on something they do not yet squeeze."""
    yield from self.mission.to_stand_routine()
    g = self.reach.grippers[side]
    if closed:
      g.shut(force)
    else:
      g.open()
    yield from self.hold_routine(GRIP_S)
    if not closed:
      return True
    waited = 0.0
    while not g.held() and g.read()[0] > HELD_GAP_M and waited < GRIP_SETTLE_S:
      yield from self.hold_routine(GRIP_READ_S)
      waited += GRIP_READ_S
    return g.held()

  def gripped(self, side) -> bool:
    return self.reach.grippers[side].held()

  def grip_gap(self, side) -> float:
    return self.reach.grippers[side].read()[0]

  def hand_at(self, side):
    p = self.to_world(self.reach.grip_now(side)[0])
    return float(p[0]), float(p[1]), float(p[2])

  def hand_sensed(self, side):
    got = self.reach.sensed_grip.get(side)
    p = self.to_world(self.reach.grip_now(side)[0] if got is None else got)
    return float(p[0]), float(p[1]), float(p[2])

  def hand_watched(self, side):
    if self._watch is None:
      from pluggybot.humanoid.arms import SelfModel
      from pluggybot.humanoid.senses import JointSense
      self._watch = (JointSense(self._m, prefix=self.handle.prefix,
                                key=f"watch:{self.handle.prefix}"), SelfModel())
    sense, sm = self._watch
    q, _, _ = sense.read(self._d)
    sm.pose(q, self.mission.stand.trunk.gravity())
    p = self.to_world(sm.grip_pose(side)[0])
    return float(p[0]), float(p[1]), float(p[2])

  def depth_watched(self, width, height):
    from pluggybot.perception.depth import DepthCamera
    key = (int(width), int(height))
    if self._depth_watch is None or self._depth_watch[0] != key:
      self._depth_watch = (key, DepthCamera(self._m, self.handle, width=key[0], height=key[1],
                                            mount="body", seed=WATCH_SEED, reach=DEPTH_REACH_M))
    return self._depth_watch[1].frame(self._d).z

  def hand_follows(self, side, poses, rot=None) -> bool:
    return self.reach.path_for(side, [self.to_stance(p) for p in poses],
                               rot=self._turns(rot)) is not None

  def hand_reaches(self, side, pos, rot=None) -> bool:
    r = None if rot is None else _rz(self.stance[2]).T @ np.asarray(rot, dtype=float)
    return self.reach.plans(side, HandGoal(self.to_stance(pos), r, 1.0))

  def _turns(self, rot):
    """A turn, or a list of them, the believed world frame's, in the stance
    frame's."""
    yaw = _rz(self.stance[2]).T
    if rot is None:
      return None
    if isinstance(rot, (list, tuple)):
      return [None if r is None else yaw @ np.asarray(r, dtype=float) for r in rot]
    return yaw @ np.asarray(rot, dtype=float)

  def gaze_routine(self, down):
    """A bow of `down` rad at the height it stands at, inside what the
    standing policy trained on, until its trunk is still there -- ⚠ NEVER
    WITH A HAND OUT, which keeps the posture it has: the bow carries the
    hand with it, and a look from a low reach's bow lifted the hand 43 cm
    (#525)."""
    yield from self.mission.to_stand_routine()
    if any(p != "rest" for p in self.reach.phase.values()):
      return
    lo, hi = PITCH_RANGE
    self.reach.hold_posture(float(self.reach.posture[0]), min(max(float(down), lo), hi))
    waited = 0.0
    while not self.reach._posture_held() and waited < GAZE_WAIT_S:
      yield from self.hold_routine(GRIP_READ_S)
      waited += GRIP_READ_S

  def depth_cloud(self, width, height, min_z, frames=1, apart=0.1):
    """Its D435's points, each frame placed through its pelvis in its model
    of itself (posed off its encoders and IMU) and its stance: the camera
    sits on its trunk, which no waist turns."""
    from pluggybot.perception.depth import DepthCamera
    key = (int(width), int(height), float(min_z))
    if getattr(self, "_depth", (None,))[0] != key:
      self._depth = (key, DepthCamera(self._m, self.handle, width=key[0], height=key[1],
                                      min_z=key[2], mount="body", seed=DEPTH_SEED,
                                      reach=DEPTH_REACH_M))
    cam = self._depth[1]
    out = []
    for _ in range(int(frames)):
      yield from self.hold_routine(apart)
      pts = cam.frame(self._d).points
      self.reach._pose()
      sm = self.reach.self_model
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, sm.d.qpos[3:7])
      inside = pts @ rot.reshape(3, 3).T + sm.d.qpos[0:3]
      x, y, yaw = self.stance
      out.append(inside @ _rz(yaw).T + [x, y, 0.0])
    return np.concatenate(out) if out else np.zeros((0, 3))

  def _report(self, rep: dict) -> dict:
    """A carry's or a pull's report (`Reach.report`), its stance frame's
    ways turned into the believed world frame."""
    out = dict(rep)
    yaw = _rz(self.stance[2])
    if "dx" in out:
      out["dx"], out["dy"], out["dz"] = (float(v) for v in yaw @ [rep["dx"], rep["dy"], rep["dz"]])
    if "direction" in out:
      out["direction"] = [float(v) for v in yaw @ np.asarray(rep["direction"], dtype=float)]
    return out

  def _held_routine(self, side, patience):
    """Wait out a hand's carry or pull: what it reported, or why it was cut
    short -- a fall (which lets go of the arms), its own feet moved
    (`FEET_MOVED_M`), or `patience`."""
    falls = self.mission.falls
    x0, y0, _ = self.stance
    for _ in range(int(round(patience / self._m.opt.timestep))):
      if self.mission.falls != falls:
        return {"why": "fell", "seconds": 0.0, "moved": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0}
      if self.reach.phase[side] != "carry":
        return self._report(self.reach.report[side])
      x, y, _ = self.stance
      if math.hypot(x - x0, y - y0) > FEET_MOVED_M:
        self.reach.stop(side, "feet moved")
        return self._report(self.reach.report[side])
      yield STILL
    self.reach.stop(side, "out of time")
    return self._report(self.reach.report[side])

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=HOLD_SPEED, patience=20.0,
                        path=None, turns=None):
    """...`path`, points it goes through on its way, in turn (#493: a lid's
    arc carried as one motion, from one posture that follows it all), and
    `turns`, the grip's turn at each."""
    yield from self.mission.to_stand_routine()
    ways = None if path is None else [self.to_stance(p) for p in path]
    if not self.reach.carry(side, self.to_stance(pos), stiffness, speed, path=ways,
                            turns=None if turns is None else self._turns(list(turns))):
      rep = self.reach.report[side] if self.reach.out(side) else None
      return self._report(rep or {"why": "not out", "seconds": 0.0, "moved": 0.0,
                                  "dx": 0.0, "dy": 0.0, "dz": 0.0})
    return (yield from self._held_routine(side, patience))

  def pull_routine(self, side, direction, limit_n, distance_m, speed=HOLD_SPEED,
                   patience=30.0, give=True):
    """THE PULL, a force move (#525): the hand that is out pulled -- or
    pushed -- along `direction` with at most `limit_n`, N, as the hand reads
    it, at most `speed`, m/s, going where what it holds takes it across that
    way, its turn giving -- or, not `give`, holding its line and its turn
    (`Reach.pull`); stopped at `distance_m` along its
    way, held fast, its grip lost ("let go"), out of reach, its feet moved,
    or `patience` run out."""
    yield from self.mission.to_stand_routine()
    d = _rz(self.stance[2]).T @ np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    t0 = float(self._d.time)
    if not self.reach.out(side):
      return self._report({"why": "not out", "seconds": 0.0, "moved": 0.0, "along": 0.0,
                           "dx": 0.0, "dy": 0.0, "dz": 0.0, "direction": list(d), "force": 0.0})
    start = self.reach.grip_now(side)[0].copy()
    gripped = self.gripped(side)
    if not self.reach.pull(side, d, limit_n, distance_m, speed, give):
      rep = {**(self.reach.report[side] or {}), "why": "out of reach"}
    else:
      rep = yield from self._held_routine(side, patience)
      rep = dict(self.reach.report[side] or rep) if rep["why"] not in ("fell", "feet moved",
                                                                     "out of time") else rep
    why = rep["why"]
    # ...a grip lost ends it "let go", but where its stop condition ended it
    # first, that is how it ended (the condition may be the grip's own word)
    if gripped and not self.gripped(side) and why not in ("fell", "until"):
      why = "let go"
    h = self.reach.grip_now(side)[0]
    disp = h - start
    n = float(np.linalg.norm(disp))
    way = rep.get("way") or list(disp / n if n > 1e-9 else d)
    return self._report({"why": why, "seconds": round(float(self._d.time) - t0, 3),
                         "moved": round(n, 4), "along": round(float(rep.get("went", disp @ d)), 4),
                         "dx": float(disp[0]), "dy": float(disp[1]), "dz": float(disp[2]),
                         "direction": [float(v) for v in way],
                         "force": round(float(rep.get("force") or 0.0), 2)})

  def hand_force(self, side):
    f = _rz(self.stance[2]) @ self.reach.force[side]
    return float(f[0]), float(f[1]), float(f[2])

  def step_routine(self, ahead, left=0.0, turn=0.0, patience=STEP_PATIENCE_S):
    """THE STEP (#525): its hands home and empty, it turns `turn` on the
    spot, walks `ahead` and `left` in the frame it then stands in -- one
    motion at `STEP_V` (`BACK_V` backing up), or the one and then the other
    where the walker would not walk that way (`_step_walks`) -- and stands:
    a walk stops once what it believes it has walked, and its speed times
    `STEP_STOP_S`, reach its length, so a stride, about 10 cm, is the least
    it goes. Its legs are the walking policy's meanwhile. Returns how it
    ended (`why`: "stepped", "holding" -- a hand holds something, and it
    does not move -- "blocked", "fell", "out of time"), in how long, and
    where its stance went, by its own reckoning, in the stance frame it
    stood in before (`dx`, `dy`, m; `turn`, rad)."""
    mi = self.mission
    yield from mi._getup_routine()
    t0, s0, falls = float(self._d.time), self.stance, mi.falls
    if any(self.gripped(s) for s in hm.SIDES):
      return {"why": "holding", "seconds": 0.0, "dx": 0.0, "dy": 0.0, "turn": 0.0}

    def over() -> str | None:
      if mi.falls != falls:
        return "fell"
      return "out of time" if float(self._d.time) - t0 > patience else None
    length, why = math.hypot(ahead, left), "stepped"
    if abs(turn) > TURN_STOP_RAD or length:
      # the legs to the walker, its arms home, before anything is measured:
      # out of a deep bow the pelvis came 16 cm forward on its own, and the
      # hand-over outlasted `STEP_STUCK_S` (#525)
      yield from mi.to_walk_routine()
      why = over() or why
    # the turn, its heading counted unwrapped: wrapped, one past pi never ended
    heading, turned = mi.pose[2], 0.0
    while (why == "stepped" and abs(turn) > TURN_STOP_RAD
           and turned * math.copysign(1.0, turn) < abs(turn) - TURN_STOP_RAD):
      yield from mi._twist_routine(0.0, 0.0, math.copysign(TURN_W, turn))
      turned += math.remainder(mi.pose[2] - heading, 2 * math.pi)
      heading = mi.pose[2]
      why = over() or why
    # ⚠ A TURN IS TAKEN AS ON THE SPOT: its feet shuffle round, the stance
    # truly moves 2-6 cm and the odometry is 3-7 cm off over it (turns of
    # 17-86 deg) -- so the step's move is the walks' alone, measured from
    # where the turn left the belief
    s_mid = self.stance
    for a_, l_ in _step_walks(ahead, left):
      if why == "stepped":
        why = yield from self._walk_routine(a_, l_, over)
    yield from mi.to_stand_routine()
    for _ in range(int(round(STEP_SETTLE_S / self._m.opt.timestep))):
      yield STILL
    if mi.falls != falls:
      why = "fell"
    x1, y1, yaw1 = self.stance
    if not length:
      x1, y1 = s_mid[0], s_mid[1]
    d = _rz(s0[2]).T @ np.array([x1 - s_mid[0], y1 - s_mid[1], 0.0])
    return {"why": why, "seconds": round(float(self._d.time) - t0, 3), "dx": float(d[0]),
            "dy": float(d[1]), "turn": math.remainder(yaw1 - s0[2], 2 * math.pi)}

  def _walk_routine(self, ahead, left, over):
    """One straight walk of a step, its legs the walker's: "stepped" once
    what it believes it has walked, and its speed times `STEP_STOP_S`, reach
    its length; "blocked" where it gets nowhere; else what `over` says."""
    mi, dt = self.mission, self._m.opt.timestep
    length = math.hypot(ahead, left)
    x0, y0, yaw = mi.pose
    # the walk's way in the world, and its twist in the trunk's frame
    ux, uy = ahead / length, left / length
    wx, wy = math.cos(yaw) * ux - math.sin(yaw) * uy, math.sin(yaw) * ux + math.cos(yaw) * uy
    pace = BACK_V if ahead < -abs(left) else STEP_V
    speed, went, mark = 0.0, 0.0, (float(self._d.time), 0.0)
    while True:
      x, y, _ = mi.pose
      now = wx * (x - x0) + wy * (y - y0)
      speed += min(1.0, dt / 0.2) * ((now - went) / dt - speed)
      went = now
      if went + max(speed, 0.0) * STEP_STOP_S >= length:
        return "stepped"
      if float(self._d.time) - mark[0] >= STEP_STUCK_S:
        if went - mark[1] < STEP_STUCK_M:
          return "blocked"
        mark = (float(self._d.time), went)
      why = over()
      if why is not None:
        return why
      yield from mi._twist_routine(pace * ux, pace * uy, 0.0)

  # ---- senses --------------------------------------------------------------

  def pack(self, capacity_wh: float, charge_scale: float):
    return HumanoidPack(self, capacity_wh, charge_scale)

  def depth_camera(self):
    """Its D435 as the loop's near-field reads it (#559): the frames its
    walk lays its low layer from (`HumanoidMission.latest_depth`)."""
    return _SharedDepth(self.mission)

  def know_peer(self, root: str) -> None:
    """The other robot's body out of its LIDAR's scan and its depth
    camera's points (#559), as the quadruped's: painted into the map, a
    robot walked past walls in the one that passed it."""
    self.mission.lidar.exclude_robot(root)
    self.mission.depth.exclude_robot(root)

  def _detector(self):
    if self._tags is None:
      from pluggybot.rack.tags import TagDetector
      self._tags = TagDetector(self._m, self.handle.el(TAG_CAMERA))
    return self._tags

  def detect_tags(self) -> dict:
    return self._detector().detect(self._d)

  def spot(self, tag: int, at_height: float | None = None) -> dict | None:
    """One tag's centre in the believed world frame off one decode of its
    colour imager, through the camera's pose in its model of itself (the
    joints off their drivers, the tilt off its IMU) and its stance; or None
    where it does not decode. With `at_height`, the ray through the tag's
    centre is cut at that height instead of PnP's range: {x, y, z, range,
    normal}, the normal into the tag's face, world frame. ⚠ A CUBE WEARS
    ITS TAG ON EVERY FACE (`rack.tags.TagDetector.detect_all`): the face
    seen most square-on is the one read -- looking down on the chest's
    knob, the last decoded was its top, seen edge-on."""
    return self._decoded(tag, at_height)

  def watch(self, tag):
    if tag is None:
      self.mission.sight.stop()
    else:
      self.mission.sight.start(int(tag))

  def sighting(self):
    return self.mission.sight.newest

  def _sighted(self, tag: int) -> dict | None:
    """A watch's picture (`perception.watch.TagWatch`): `spot`'s decode,
    the camera placed through its model of itself as its arm's loop last
    posed it (#554) -- read afresh, every picture drew on its encoders'
    noise, and a watch nothing read changed the flight. A picture before
    that loop has posed it (a reach built by a stand-up) shows nothing."""
    if any(g is None for g in self.reach.sensed_grip.values()):
      return None
    return self._decoded(tag, fresh=False)

  def _decoded(self, tag: int, at_height: float | None = None,
               fresh: bool = True) -> dict | None:
    """`spot`'s reading, its model of itself posed afresh off its senses
    (`fresh`) or as its arm's loop last posed it."""
    faces = [d for d in self._detector().detect_all(self._d) if d["id"] == tag]
    if not faces:
      return None

    def square_on(d):
      t = np.asarray(d["t"])
      return float(np.asarray(d["normal"]) @ (t / np.linalg.norm(t)))
    seen = max(faces, key=square_on)
    if fresh:
      self.reach._pose()
    sm = self.reach.self_model
    # Kinematics places bodies; a camera is placed by `mj_camlight`.
    mujoco.mj_camlight(sm.m, sm.d)
    cam = sm.m.camera(TAG_CAMERA).id
    origin, rot = sm.d.cam_xpos[cam].copy(), sm.d.cam_xmat[cam].reshape(3, 3)
    # The detector's camera is x right, y down, z ahead; MuJoCo's looks along
    # its -z with y up.
    t = np.array(seen["t"])
    p = origin + rot @ np.array([t[0], -t[1], -t[2]])
    if at_height is not None:
      ray = p - origin
      cut = (at_height - origin[2]) / ray[2] if abs(ray[2]) > 1e-9 else 1.0
      p = origin + ray * cut
    n = rot @ np.array([seen["normal"][0], -seen["normal"][1], -seen["normal"][2]])
    e = rot @ np.array([seen["edge"][0], -seen["edge"][1], -seen["edge"][2]])
    w = self.to_world(p)
    nw, ew = _rz(self.stance[2]) @ n, _rz(self.stance[2]) @ e
    return {"x": float(w[0]), "y": float(w[1]), "z": float(w[2]),
            "range": float(np.linalg.norm(p - origin)), "normal": tuple(float(v) for v in nw),
            "edge": tuple(float(v) for v in ew)}

  # ---- posture and rest ----------------------------------------------------

  def rest_routine(self):
    """It rests STANDING, by itself: `resting` once it has stood still long
    enough (`humanoid.way`, #559); nothing to do."""
    return
    yield

  def stand_routine(self):
    return
    yield

  resting = property(lambda self: self.mission.resting)

  # ---- going places: its mission's -------------------------------------------

  def go_to_routine(self, x, y, timeout=None, stop=None):
    kw = {k: v for k, v in (("timeout", timeout), ("stop", stop)) if v is not None}
    return self.mission.drive_to_routine(x, y, **kw)

  last_drive = property(lambda self: self.mission.last_drive)

  def gave_up(self, record, peer="the other robot") -> str:
    """...a humanoid rests standing (#559): the other at rest, standing."""
    from pluggybot.navigator import gave_up
    return (gave_up(record, peer).replace("the drive gave up", "the walk gave up", 1)
            .replace("lying down to rest", "standing at rest"))

  def face_routine(self, heading):
    return self.mission.face_routine(heading)

  def look_around_routine(self):
    return self.mission._spin_routine()

  def velocity_routine(self, seconds, v, w):
    return self.mission._drive_routine(seconds, v, w)

  def travel_routine(self, distance, v):
    return self.mission.travel_routine(distance, v)

  grid = property(lambda self: self.mission.grid)
  belief_events = property(lambda self: self.mission.belief_events)

  @property
  def match_counts(self) -> dict:
    m = self.mission.matcher
    return dict(m.counts) if m is not None else {}

  def forget_world(self) -> None:
    """A true death (#419): the map, its layer, the walls it is kept off and
    the plan it last made, gone, as the quadruped's."""
    m = self.mission
    m.forget_map()
    m.low[:] = 0.0
    m._walls_t, m._walls_mask, m._plan_memo = None, None, None

  def plan_frontier(self, blacklist):
    from pluggybot.behavior.navigation import plan
    return plan(self.grid, self.pose, blacklist, self.mission.INFLATION_CELLS,
                own_component=True)

  def reachable(self, points):
    return self.mission.reachable(points)

  def in_sight(self, x, y) -> bool:
    return self.mission.in_sight(x, y)

  def lost_routine(self, fixture, stop=None):
    raise NotImplementedError(f"{NOT_HERE}: its fixtures are a dock's and a rack's")
    yield

  def find_tag_routine(self, tag, near, patience, stop=None):
    raise NotImplementedError(f"{NOT_HERE}: a place's walk is the quadruped's (#419)")
    yield

  # ---- the quadruped's world: not the humanoid's yet -----------------------

  def press_plate_routine(self, tag, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def draw_routine(self, board, program, patience, stop=None, on_stroke=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def pick_cube_routine(self, tag, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def place_cube_routine(self, tag, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def put_cube_routine(self, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def claw_routine(self, closed):
    raise NotImplementedError(NOT_HERE)
    yield

  def held_cube(self):
    return None

  def survey_routine(self, tag, patience, stop=None, on_count=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def hide_routine(self, away_from, reach_m, clear_of_m, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def seek_routine(self, base, cover_m, patience, stop=None):
    raise NotImplementedError(NOT_HERE)
    yield

  def start_discovery(self) -> None:
    """No rack in its world to watch for (#559): the quadruped's went with
    it, and the day begins here all the same."""

  def refresh_rack(self):
    return None

  def bay_standoff(self, station_y):
    raise NotImplementedError(NOT_HERE)

  def fetch_tool_routine(self, station_y, module):
    raise NotImplementedError(NOT_HERE)
    yield

  def stow_tool_routine(self, station_y, module):
    raise NotImplementedError(NOT_HERE)
    yield

  def place_tool_routine(self, module):
    raise NotImplementedError(NOT_HERE)
    yield

  def module_state(self, module):
    raise NotImplementedError(NOT_HERE)

  def tool_powered(self, module) -> bool:
    return False

  def seated_on(self, module):
    return None

  def swap_trace(self) -> str:
    return ""

  def charge_standoff(self):
    raise NotImplementedError(NOT_HERE)

  def charge_seat(self):
    raise NotImplementedError(NOT_HERE)

  def dock_routine(self):
    raise NotImplementedError(NOT_HERE)
    yield

  def charging(self) -> bool:
    return False

  def anchor_at_dock(self) -> None:
    raise NotImplementedError(NOT_HERE)

  def dock_hold_routine(self, seconds):
    raise NotImplementedError(NOT_HERE)
    yield

  def redock_routine(self):
    raise NotImplementedError(NOT_HERE)
    yield

  def undock_routine(self):
    raise NotImplementedError(NOT_HERE)
    yield

  def charge_trace(self) -> str:
    return ""

  def make_way(self, route, by) -> bool:
    return self.mission.make_way(route, by)

  making_way = property(lambda self: self.mission.making_way)
  way_refused = property(lambda self: self.mission.way_refused)
  asides = property(lambda self: self.mission.asides)
  last_aside = property(lambda self: self.mission.last_aside)

  def ask_off(self, x, y, asked, beyond=()) -> bool:
    return self.mission.ask_off(x, y, asked, beyond)

  def peer_on_the_goal(self, x, y):
    return self.mission.peer_on_the_goal(x, y)

  def watch_for_peers(self, points):
    return self.mission.watch_for_peers(points)

  pressing = property(lambda self: self.mission.pressing)
  peer_holds = property(lambda self: self.mission.peer_holds)

  @property
  def others(self):
    return self.mission.others

  @others.setter
  def others(self, where) -> None:
    self.mission.others = where

  @property
  def ask_way(self):
    return self.mission.ask_way

  @ask_way.setter
  def ask_way(self, ask) -> None:
    self.mission.ask_way = ask
