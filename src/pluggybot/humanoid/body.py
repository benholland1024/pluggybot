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
from pluggybot.humanoid.mission import HumanoidMission
from pluggybot.humanoid import reach as reach_
from pluggybot.humanoid.reach import HOLD_SPEED
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
#: A gaze's bow is waited for this long at most, s (`gaze_routine`).
GAZE_WAIT_S = 4.0
#: A hand's carry or pull stops once the body's own odometry puts its
#: stance this far from where it began, m: a pull toward the robot dragged
#: it 0.3 m to the counter, its standing policy stepping under the arm
#: (#493).
FEET_MOVED_M = 0.05
#: Its measured depth's own noise stream (`depth_cloud`) and how far its
#: rays reach, m: the probe's seed, and a counter's chest within arm's reach.
DEPTH_SEED, DEPTH_REACH_M = 466, 3.0
#: What the humanoid does not do yet answers.
NOT_HERE = "not on the humanoid yet: the quadruped's world (its dock, rack and tools)"


def _yaw(rot: np.ndarray) -> float:
  return math.atan2(rot[1, 0], rot[0, 0])


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
               grid_bounds=(-3, -3, 7, 7)):
    self.handle = handle
    self._m, self._d = model, data
    self._policies = (policy, walk)
    self.root = model.body(handle.root).id
    self.mission = HumanoidMission(model, data, handle, stand=policy, walk=walk,
                                   grid_bounds=grid_bounds)
    self._tags = None
    self.geom_ids = frozenset(g for g in range(model.ngeom)
                              if self._under_root(int(model.geom_bodyid[g])))
    # The members a standing body answers plainly.
    self.places = None
    self.rack = self.rack_prior = None
    self.rack_discovered = False
    self.swapping_at = None
    self.working = False
    self.peer_at_bay_m = None
    #: Why each hand's last reach ended short of holding (`hand_why`).
    self._why: dict[str, str] = {}
    self.bay_wait = None
    self.docked = False
    self.head_camera = handle.el(HEAD_CAMERA)
    self.making_way = None
    self.way_refused = ""
    self.asides = 0
    self.last_aside = None
    self.collision_steps = self.press_steps = 0
    self.resting = False

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
    """Nothing it believes outlives a restart yet: its map is a walk's
    (a served humanoid's restart comes after stage 5)."""
    return {}, {}

  def restore_kept(self, state, arrays) -> bool:
    return True

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
      if self.reach.phase[s] not in ("rest", "retreat"):
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
    postures = reach_.POSTURES
    if path is not None:
      # the posture the hand then follows its path in, and that one alone
      ways = [self.to_stance(q) for q in path]
      follow = self.reach.path_for(side, [p] + ways,
                                   rot=[r] + self._turns(list(turns)) if turns is not None else r)
      if follow is None:
        return False
      postures = (follow,)
    if not self.reach.reach(side, HandGoal(p, r, stiffness), postures=postures,
                            flip=path is None):
      return False
    falls = self.mission.falls
    for _ in range(int(round(patience / self._m.opt.timestep))):
      if self.mission.falls != falls or self.reach.done(side):
        break
      yield STILL
    held = self.mission.falls == falls and self.reach.phase[side] == "hold"
    self._why[side] = ("" if held else "fell" if self.mission.falls != falls
                       else "stopped short" if self.reach.done(side) else "out of time")
    return held

  def hand_why(self, side) -> str:
    return self._why.get(side, "")

  def release_routine(self, side):
    self.reach.retreat(side)
    while self.reach.phase[side] != "rest":
      yield STILL
    while not self.stand.settled:
      yield STILL

  def grip_routine(self, side, closed):
    """Shut, its word is read once its jaws have closed or held (#493:
    `GRIP_SETTLE_S`), never while they stand open on something they do not
    yet squeeze."""
    yield from self.mission.to_stand_routine()
    g = self.reach.grippers[side]
    if closed:
      g.shut()
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

  def hand_at(self, side):
    p = self.to_world(self.reach.grip_now(side)[0])
    return float(p[0]), float(p[1]), float(p[2])

  def hand_reaches(self, side, pos, rot=None) -> bool:
    p = self.to_stance(pos)
    r = None if rot is None else _rz(self.stance[2]).T @ np.asarray(rot, dtype=float)
    return self.reach.posture_for(side, p, rot=r) is not None

  def hand_follows(self, side, poses, rot=None) -> bool:
    return self.reach.path_for(side, [self.to_stance(p) for p in poses],
                               rot=self._turns(rot)) is not None

  def _turns(self, rot):
    """A turn, or a list of them, the believed world frame's, in the stance
    frame's."""
    yaw = _rz(self.stance[2]).T
    if rot is None:
      return None
    if isinstance(rot, (list, tuple)):
      return [yaw @ np.asarray(r, dtype=float) for r in rot]
    return yaw @ np.asarray(rot, dtype=float)

  def gaze_routine(self, down):
    """A bow of `down` rad at the height it stands at, inside what the
    standing policy trained on, until its trunk is still there."""
    yield from self.mission.to_stand_routine()
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
      if self.reach.phase[side] not in ("carry", "yield"):
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

  def pull_routine(self, side, direction, limit_n, distance_m, stiffness=1.0,
                   speed=HOLD_SPEED, patience=30.0):
    """It EXPLORES its first `reach.EXPLORE_M`, carried straight the way
    asked on a soft arm (`reach.EXPLORE_K`), then FOLLOWS the way the hand
    went --
    fitted to the hand's own path (`reach.fit_way`), planned and carried
    `reach.FOLLOW_M` at a time, its turn yielding to what turns it
    (`Reach.carry(give=)`) -- fitting again as it goes, until it has gone
    `distance_m` the way asked or a stop: a carry held back is "held
    fast", a grip lost "let go"."""
    yield from self.mission.to_stand_routine()
    d = _rz(self.stance[2]).T @ np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    t0 = float(self._d.time)
    start = self.reach.grip_now(side)[0].copy()
    gripped = self.gripped(side)
    if not self.reach.out(side):
      return self._report({"why": "not out", "seconds": 0.0, "moved": 0.0, "dx": 0.0,
                           "dy": 0.0, "dz": 0.0, "direction": list(d), "force": 0.0})
    trail, peak, way, why = [], 0.0, d, "held fast"
    for k in (min(stiffness, reach_.EXPLORE_K), stiffness):
      h = self.reach.grip_now(side)[0]
      if not self.reach.carry(side, h + min(distance_m, reach_.EXPLORE_M) * d, k, speed,
                              give=True, limit_n=limit_n):
        why = "out of reach"
        break
      rep = yield from self._held_routine(side, patience + 1.0)
      trail += list((self.reach.report[side] or {}).get("path") or [])
      peak = max(peak, float(rep.get("force") or 0.0))
      gone = float((self.reach.grip_now(side)[0] - start) @ d)
      if rep["why"] not in ("arrived", "held back"):
        why = rep["why"]
        break
      if gone >= reach_.EXPLORED_M:
        why = "distance"
        break
    while why == "distance":
      h = self.reach.grip_now(side)[0]
      left_m = distance_m - float((h - start) @ d)
      spent = float(self._d.time) - t0
      if left_m <= reach_.ARRIVE_M:
        break
      if spent >= patience:
        why = "out of time"
        break
      if gripped and not self.gripped(side):
        why = "let go"
        break
      fitted = reach_.fit_way(trail)
      ahead = reach_.continue_way(fitted, h, reach_.FOLLOW_M)
      # ...no further the way asked than it was asked
      points = [q for q in ahead if float((q - start) @ d) <= distance_m] or ahead[:1]
      if not self.reach.carry(side, points[-1], stiffness, speed, path=points[:-1], give=True,
                              limit_n=limit_n):
        why = "out of reach"
        break
      rep = yield from self._held_routine(side, patience - spent + 1.0)
      trail += list((self.reach.report[side] or {}).get("path") or [])[1:]
      peak = max(peak, float(rep.get("force") or 0.0))
      way = points[-1] - h
      onward = float(np.linalg.norm(self.reach.grip_now(side)[0] - h)) >= reach_.ONWARD_M
      why = ("distance" if rep["why"] == "arrived" or (rep["why"] == "held back" and onward)
             else "held fast" if rep["why"] == "held back" else rep["why"])
    h = self.reach.grip_now(side)[0]
    disp = h - start
    n = float(np.linalg.norm(way))
    return self._report({"why": why, "seconds": round(float(self._d.time) - t0, 3),
                         "moved": round(float(np.linalg.norm(disp)), 4),
                         "along": round(float(disp @ d), 4),
                         "dx": float(disp[0]), "dy": float(disp[1]), "dz": float(disp[2]),
                         "direction": [float(v) for v in (way / n if n > 1e-9 else d)],
                         "force": round(peak, 2)})

  def hand_force(self, side):
    f = _rz(self.stance[2]) @ self.reach.force[side]
    return float(f[0]), float(f[1]), float(f[2])

  # ---- senses --------------------------------------------------------------

  def pack(self, capacity_wh: float, charge_scale: float):
    return HumanoidPack(self, capacity_wh, charge_scale)

  def depth_camera(self):
    """None yet: the D435's depth is a walking body's map (stage 4)."""
    return None

  def know_peer(self, root: str) -> None:
    """One robot alone yet: a pair is a served world's (after stage 5)."""

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
    faces = [d for d in self._detector().detect_all(self._d) if d["id"] == tag]
    if not faces:
      return None

    def square_on(d):
      t = np.asarray(d["t"])
      return float(np.asarray(d["normal"]) @ (t / np.linalg.norm(t)))
    seen = max(faces, key=square_on)
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
    w = self.to_world(p)
    nw = _rz(self.stance[2]) @ n
    return {"x": float(w[0]), "y": float(w[1]), "z": float(w[2]),
            "range": float(np.linalg.norm(p - origin)), "normal": tuple(float(v) for v in nw)}

  # ---- posture and rest ----------------------------------------------------

  def rest_routine(self):
    """It rests STANDING: its standing policy holds it at the stand, and
    lying down is a get-up's partner (stage 5)."""
    self.resting = True
    return
    yield

  def stand_routine(self):
    self.resting = False
    return
    yield

  # ---- going places: its mission's -------------------------------------------

  def go_to_routine(self, x, y, timeout=None, stop=None):
    kw = {k: v for k, v in (("timeout", timeout), ("stop", stop)) if v is not None}
    return self.mission.drive_to_routine(x, y, **kw)

  last_drive = property(lambda self: self.mission.last_drive)

  def gave_up(self, record, peer="the other robot") -> str:
    from pluggybot.navigator import gave_up
    return gave_up(record, peer).replace("the drive gave up", "the walk gave up", 1)

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
    self.mission.forget_map()
    self.mission.low[:] = 0.0

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
    raise NotImplementedError(NOT_HERE)

  def refresh_rack(self):
    raise NotImplementedError(NOT_HERE)

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
    """It steps aside for nobody yet: one robot alone (a pair is a served
    world's, after stage 5)."""
    self.way_refused = "standing"
    return False

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
