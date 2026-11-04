"""A waist for the humanoid, measured before it is built (#560, stage (c)).

Today the pelvis, the torso and the head are one rigid body
(`humanoid.model`: "there is no waist"). This instrument builds the
candidates on the robot's MODEL OF ITSELF -- today's generated drawing with
a waist put in at the pelvis's top -- and measures each the way the reach
plans today (`reach.plan`): a posture tried on a statue's legs balanced over
the soles (`reach.static_legs`), the arm's six joints solved by its own
inverse kinematics, least bent first. A posture here is the legs' (the
pelvis's height and pitch, `reach.POSTURES`) and the waist's angles.

  none       today's body
  pitch      a waist pitch at `PITCH_Z` over the hip axes
  yaw        a waist yaw at the pelvis's top, `YAW_Z`
  pitch_yaw  both, the yaw below the pitch (the T2's and the G1's order: a
             bow is toward where the shoulders face)

Nothing here is served, trained or flown: no policy holds a waist yet, so
every number is a statue's -- what the body could reach and hold still,
an upper bound on what a policy will.

  --bands          the table's bands: each hand's reach at the table's
                   height, fingers down and tilted 45 deg, every candidate
  --window         the stance window at the table's place: how far the
                   stance may land from where our procedure asks and its
                   hand still reach each piece and where it goes
  --envelope       the reach scene's envelope (SimNotes, "The humanoid
                   stands, reaches and grasps"): the floor to over the head
  --across         at the counter's height, how far across the middle each
                   hand reaches
  --part           the waist's held torques at the postures its reaches
                   take, with a load, against each part's stall rating;
                   its mass and where it moves the centre of mass; its draw
  --candidates A,B which candidates (default all)
  --jobs N         processes (default 3); run it under a memory cap
"""

import argparse
import itertools
import math
import multiprocessing as mp
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pluggybot.humanoid import arms as ha
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.humanoid import reach as rc
from pluggybot.telemetry.protocol import ROBOT_ROOT

# ---- the candidates' drawing -------------------------------------------------

#: The waist's axes over the hip pitch axes, m. The yaw at the pelvis's top
#: (its motor stood in the pelvis's middle, between the hips' pitch motors:
#: they leave |y| < 0.054 free, an RS03's 98 mm fits); the pitch at the
#: T2's height (its waist pitch 0.182 m over its hips), the pack below it.
YAW_Z = 0.06
PITCH_Z = 0.18
#: The waist's ranges, rad: the pitch is the T2's (-43 deg back... +32 deg
#: here forward is POSITIVE, the trunk bowed over its hips as the posture's
#: pitch is), the yaw +-60 deg (the T2's turns +-138; past 60 here no reach
#: at a table grew, below).
PITCH_RANGE = (-0.558505, 0.750492)
YAW_RANGE = (-1.05, 1.05)
#: The parts: the yaw an RS03 (it holds no weight standing upright), the
#: pitch an RS04 (the trunk's whole upper half bows on it).
YAW_MOTOR, PITCH_MOTOR = hp.RS03, hp.RS04
#: The link between them: a 5 mm 6082 turntable plate the pelvis's
#: footprint, with `model.ALLOWANCE`.
LINK_PLATE = hm.Plate(2 * hm.CHOSEN.pelvis_half[0] * 2 * hm.CHOSEN.pelvis_half[1], 0.005)

CANDIDATES = {"none": (), "pitch": ("pitch",), "yaw": ("yaw",),
              "pitch_yaw": ("yaw", "pitch")}
WAIST_JOINTS = {"yaw": "waist_yaw", "pitch": "waist_pitch"}

#: What rides the pelvis whatever the waist: the root's own.
PELVIS_PARTS = {"imu", "pelvis", "left_hip_pitch_motor", "right_hip_pitch_motor",
                "left_hip_pitch_link", "right_hip_pitch_link"}
#: What sits below the pitch's axis, and so rides the pelvis (or the yaw's
#: link) when there is a pitch: the pack and the torso's lower box.
BELOW_PITCH = {"pack", "torso_low"}


def _f(v) -> str:
  return f"{v:.6g}"


def _el_name(el) -> str:
  return el.get("name", "")


def _housing(name: str, motor, at, axis: str) -> ET.Element:
  return ET.fromstring(hm._housing(name, motor, at, axis))


def waist_xml(kind: str, spec: hm.HumanoidSpec = hm.CHOSEN, pack_below: bool = True,
              scenery: str = "") -> str:
  """The robot's drawing with the candidate's waist (`CANDIDATES[kind]`):
  today's generated MJCF with the trunk cut at the waist -- the root keeps
  the pelvis, its IMU and the legs; a yaw's link the pelvis's top; the
  torso, the head, its sensors, the port and the arms ride above. The
  torso's box is cut at the pitch's axis, the pack and the lower part
  below it (`pack_below`; else the pack rides the bow too). Every element
  above keeps its place through a `<frame>` that undoes its link's offset,
  so the zero pose is today's body, plus the waist's motors and link."""
  joints = CANDIDATES[kind]
  root = ET.fromstring(hm.body_xml(spec, scenery=scenery))
  for key in root.findall("keyframe"):
    root.remove(key)
  body = next(b for b in root.iter("body") if b.get("name") == ROBOT_ROOT)
  if not joints:
    return ET.tostring(root, encoding="unicode")
  tx, ty, tz = spec.torso_half
  children = list(body)
  for el in children:
    body.remove(el)
  keep, upper = [], []
  for el in children:
    (keep if el.tag == "freejoint" or _el_name(el) in PELVIS_PARTS else upper).append(el)
  if "pitch" in joints:
    # The torso's box cut at the pitch's axis: its plates' mass shared by height.
    torso = next(e for e in upper if _el_name(e) == "torso")
    upper.remove(torso)
    lo, hi = spec.torso_z - tz, spec.torso_z + tz
    kg = float(torso.get("mass"))
    gap = 0.004
    for name, a, b in (("torso_low", lo, PITCH_Z - gap), ("torso", PITCH_Z + gap, hi)):
      e = ET.Element("geom", {"name": name, "class": "limb", "type": "box",
                              "size": f"{_f(tx)} {_f(ty)} {_f((b - a) / 2)}",
                              "pos": f"0 0 {_f((a + b) / 2)}",
                              "mass": _f(kg * (b - a) / (hi - lo))})
      upper.append(e)
  lower = [e for e in upper if _el_name(e) in BELOW_PITCH and pack_below and "pitch" in joints]
  upper = [e for e in upper if e not in lower]
  for el in keep:
    body.append(el)
  parent, at = body, 0.0
  for j in joints:
    z = YAW_Z if j == "yaw" else PITCH_Z
    motor = YAW_MOTOR if j == "yaw" else PITCH_MOTOR
    axis = "z" if j == "yaw" else "y"
    # the motor's housing on the link below its joint (its stator's), on
    # the axis: the yaw's stood under the pelvis's top, the pitch's across.
    centre = (0.0, 0.0, z - motor.length / 2) if j == "yaw" else (0.0, 0.0, z)
    parent.append(_housing(f"waist_{j}_motor", motor, (centre[0], centre[1], centre[2] - at), axis))
    link = ET.SubElement(parent, "body", {"name": f"waist_{j}_link", "pos": f"0 0 {_f(z - at)}"})
    lo, hi = YAW_RANGE if j == "yaw" else PITCH_RANGE
    ET.SubElement(link, "joint", {
      "name": WAIST_JOINTS[j], "axis": "0 0 1" if j == "yaw" else "0 1 0",
      "range": f"{_f(lo)} {_f(hi)}", "armature": _f(motor.armature),
      "frictionloss": _f(hp.friction_nm(motor))})
    if j == "yaw":
      # the turntable plate the yaw turns, just over the pelvis
      ET.SubElement(link, "geom", {"name": "waist_plate", "class": "visual", "type": "box",
                                   "size": f"{_f(spec.pelvis_half[0])} {_f(spec.pelvis_half[1])} 0.0025",
                                   "pos": "0 0 0.0025", "mass": _f(LINK_PLATE.kg)})
    parent, at = link, z
    if j == "yaw" and "pitch" in joints and lower:
      frame = ET.SubElement(link, "frame", {"pos": f"0 0 {_f(-at)}"})
      for el in lower:
        frame.append(el)
      lower = []
  if lower:
    # pitch alone: what is below its axis rides the pelvis
    for el in lower:
      body.append(el)
  frame = ET.SubElement(parent, "frame", {"pos": f"0 0 {_f(-at)}"})
  for el in upper:
    frame.append(el)
  return ET.tostring(root, encoding="unicode")


# ---- the robot's model of itself, with a waist ------------------------------

class WaistModel(ha.SelfModel):
  """`arms.SelfModel` on a candidate's drawing, its waist set by `waist`
  (joint name -> rad) whenever it is posed."""

  def __init__(self, kind: str = "none", pack_below: bool = True, scenery: str = ""):
    self.kind = kind
    self.m = mujoco.MjModel.from_xml_string(waist_xml(kind, pack_below=pack_below,
                                                      scenery=scenery))
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in hm.JOINT_NAMES])
    self.ranges = np.array([hm.joint_range(n) for n in hm.JOINT_NAMES])
    self.grip = {s: m.site(f"{s}_grip").id for s in hm.SIDES}
    self.soles = [m.geom(f"{s}_sole").id for s in hm.SIDES]
    self.root = m.body(ROBOT_ROOT).id
    self.arm = {s: np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for j in hm.ARM])
                for s in hm.SIDES}
    self.leg = {s: np.array([hm.JOINT_NAMES.index(f"{s}_{j}") for j in hm.LEG])
                for s in hm.SIDES}
    self.q = np.zeros(len(hm.JOINT_NAMES))
    self.waist_adr = {WAIST_JOINTS[j]: int(m.jnt_qposadr[m.joint(WAIST_JOINTS[j]).id])
                      for j in CANDIDATES[kind]}
    self.waist_dof = {WAIST_JOINTS[j]: int(m.jnt_dofadr[m.joint(WAIST_JOINTS[j]).id])
                      for j in CANDIDATES[kind]}
    self.waist = {n: 0.0 for n in self.waist_adr}

  def solve_chain(self, side: str, pos, rot=None, iters: int = 80):
    """`SelfModel.solve` with the waist's joints in `side`'s arm's chain: the
    damped least-squares solve over the arm's six and the waist's, from where
    they are, every joint `RANGE_MARGIN` inside its range; the waist's angles
    left in `self.waist`. Returns the arm's joints and how far off, m and
    rad."""
    m, d = self.m, self.d
    arm = self.arm[side]
    names = list(self.waist_adr)
    wj = [m.joint(n).id for n in names]
    lo = np.concatenate([self.ranges[arm, 0], m.jnt_range[wj, 0]]) + ha.RANGE_MARGIN
    hi = np.concatenate([self.ranges[arm, 1], m.jnt_range[wj, 1]]) - ha.RANGE_MARGIN
    cols = np.concatenate([self.vadr[arm], [self.waist_dof[n] for n in names]]).astype(int)
    q = np.clip(np.concatenate([self.q[arm], [self.waist[n] for n in names]]), lo, hi)

    def put(q):
      self.q[arm] = q[:6]
      d.qpos[self.qadr[arm]] = q[:6]
      for k, n in enumerate(names):
        self.waist[n] = float(q[6 + k])
        d.qpos[self.waist_adr[n]] = q[6 + k]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)

    put(q)
    for _ in range(iters):
      have_p, have_r = self.grip_pose(side)
      e_p = np.asarray(pos, dtype=float) - have_p
      jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
      mujoco.mj_jacSite(m, d, jp, jr, self.grip[side])
      if rot is None:
        e, j = e_p, jp[:, cols]
        er = 0.0
      else:
        e_r = ha._rot_error(np.asarray(rot), have_r)
        er = float(np.linalg.norm(e_r))
        e = np.concatenate([e_p, ha.IK_ROT_WEIGHT * e_r])
        j = np.vstack([jp[:, cols], ha.IK_ROT_WEIGHT * jr[:, cols]])
      if float(np.linalg.norm(e_p)) < ha.IK_POS_TOL and er < ha.IK_ROT_TOL:
        break
      dq = j.T @ np.linalg.solve(j @ j.T + ha.IK_DAMPING ** 2 * np.eye(len(e)), e)
      q = np.clip(q + np.clip(dq, -ha.IK_STEP, ha.IK_STEP), lo, hi)
      put(q)
    have_p, have_r = self.grip_pose(side)
    ep = float(np.linalg.norm(np.asarray(pos) - have_p))
    er = 0.0 if rot is None else float(np.linalg.norm(ha._rot_error(np.asarray(rot), have_r)))
    return q[:6].copy(), ep, er

  def set_waist(self, waist: dict) -> None:
    self.waist = {n: float(waist.get(n, 0.0)) for n in self.waist_adr}

  def pose(self, q, gravity_b) -> None:
    """`SelfModel.pose`, the waist's joints set too."""
    g = np.asarray(gravity_b, dtype=float)
    g = g / np.linalg.norm(g)
    pitch = math.asin(max(-1.0, min(1.0, g[0])))
    roll = math.atan2(-g[1], -g[2])
    self.q = np.array(q, dtype=float)
    d = self.d
    d.qpos[:] = 0.0
    d.qpos[self.qadr] = self.q
    for n, a in self.waist_adr.items():
      d.qpos[a] = self.waist[n]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), "XYZ")
    d.qpos[3:7] = quat
    mujoco.mj_kinematics(self.m, d)
    heads = [math.atan2(d.geom_xmat[g_][3], d.geom_xmat[g_][0]) for g_ in self.soles]
    yaw = math.atan2(sum(math.sin(h) for h in heads), sum(math.cos(h) for h in heads))
    bottoms = [d.geom_xpos[g_] - d.geom_xmat[g_].reshape(3, 3)[:, 2] * self.m.geom_size[g_][2]
               for g_ in self.soles]
    mid = np.mean(bottoms, axis=0)
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), -yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    d.qpos[0:3] = rz @ (-mid)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, turn, quat)
    d.qpos[3:7] = out
    mujoco.mj_kinematics(self.m, d)
    mujoco.mj_comPos(self.m, d)


# ---- postures: the legs' and the waist's ------------------------------------

#: The waist's angles a posture may take, rad: the pitch forward in steps
#: of 0.15 to its range's end, the yaw either way in steps of 15 deg.
WAIST_PITCHES = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)
WAIST_YAWS = tuple(math.radians(a) for a in (0, 15, -15, 30, -30, 45, -45, 60, -60))
#: A statue balances when its soles are flat on the floor and its centre of
#: mass over their middle within this, m (`reach.static_legs`' residuals).
BALANCED_M = 0.005


def waist_settings(kind: str) -> list[dict]:
  joints = CANDIDATES[kind]
  pitches = WAIST_PITCHES if "pitch" in joints else (0.0,)
  yaws = WAIST_YAWS if "yaw" in joints else (0.0,)
  out = []
  for p, y in itertools.product(pitches, yaws):
    w = {}
    if "pitch" in joints:
      w["waist_pitch"] = p
    if "yaw" in joints:
      w["waist_yaw"] = y
    out.append(w)
  return out


def bent(legs_index: int, waist: dict) -> float:
  """How bent a posture is, to try the least bent first (as `reach.plan`
  tries `POSTURES` in order): the legs' place in `reach.POSTURES`, a waist
  pitch of 0.3 rad as one more, a waist yaw of 20 deg as one more."""
  return (legs_index + max(0.0, waist.get("waist_pitch", 0.0)) / 0.3
          + abs(waist.get("waist_yaw", 0.0)) / math.radians(20))


def static_legs(sm: WaistModel, height: float, pitch: float, arms: dict | None = None):
  """`reach.static_legs` with the waist as `sm.waist` has it: the legs'
  sagittal joints that hold the pelvis `height` off the stand's and pitched
  `pitch`, the soles flat and the centre of mass over their middle -- and
  whether they do (`BALANCED_M`): a deep bow on a waist may have no statue."""
  m, d = sm.m, sm.d
  root, sole = sm.root, sm.soles[0]
  adr = {n: sm.qadr[k] for k, n in enumerate(hm.JOINT_NAMES)}
  h = hm.stand_height() + height
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
  stand = {**hm.stand_qpos(), **(arms or {})}

  def put(v):
    x, hip, knee, ankle = v
    d.qpos[:] = 0.0
    d.qpos[0], d.qpos[2] = x, h
    d.qpos[3:7] = quat
    for n, val in stand.items():
      d.qpos[adr[n]] = val
    for n, a in sm.waist_adr.items():
      d.qpos[a] = sm.waist[n]
    for s in hm.SIDES:
      d.qpos[adr[f"{s}_hip_pitch"]] = hip
      d.qpos[adr[f"{s}_knee"]] = knee
      d.qpos[adr[f"{s}_ankle_pitch"]] = ankle
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)

  def resid(v):
    put(v)
    r = d.geom_xmat[sole].reshape(3, 3)
    bottom = d.geom_xpos[sole][2] - m.geom_size[sole][2]
    return [10 * bottom, 3 * r[0, 2], 10 * (d.subtree_com[root][0] - d.geom_xpos[sole][0])]

  best = None
  for guess in ([0.0, -0.2 - pitch, 0.4, -0.2], [0.0, -0.8 - pitch, 1.6, -0.8]):
    fit = least_squares(resid, guess, bounds=([-1.0, -3.0, 0.0, -0.9], [1.0, 2.2, 2.6, 0.6]))
    if best is None or fit.cost < best.cost:
      best = fit
  r = resid(best.x)
  ok = abs(r[0]) / 10 < BALANCED_M and abs(r[2]) / 10 < BALANCED_M and abs(r[1]) / 3 < 0.02
  _, hip, knee, ankle = best.x
  legs = {f"{s}_{k}": v for s in hm.SIDES
          for k, v in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle))}
  return legs, ok


@dataclass
class Posture:
  height: float
  pitch: float
  waist: dict
  legs_index: int
  q: np.ndarray            # every joint, the arms hanging
  ok: bool                 # a statue stands there
  shoulder: dict = field(default_factory=dict)   # each shoulder's place, stance frame

  @property
  def cost(self) -> float:
    return bent(self.legs_index, self.waist)

  def label(self) -> str:
    w = " ".join(f"{k[6:]} {math.degrees(v):+.0f}" for k, v in self.waist.items() if v)
    return f"({self.height:+.2f}, {self.pitch:.2f}){' ' + w if w else ''}"


def postures(sm: WaistModel, legs=rc.POSTURES) -> list[Posture]:
  """Every posture the candidate may take -- the legs' (`reach.POSTURES`)
  with each of its waist's settings -- least bent first, each with its
  statue (the arms hanging, as `reach.plan` balances them)."""
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  out = []
  for i, (height, pitch) in enumerate(legs):
    for w in waist_settings(sm.kind):
      sm.set_waist(w)
      leg_q, ok = static_legs(sm, height, pitch)
      q = q0.copy()
      for n, v in leg_q.items():
        q[hm.JOINT_NAMES.index(n)] = v
      p = Posture(height, pitch, dict(w), i, q, ok)
      if ok:
        pose(sm, p)
        for s in hm.SIDES:
          p.shoulder[s] = sm.d.xpos[sm.m.body(f"{s}_shoulder_roll_link").id].copy()
      out.append(p)
  out.sort(key=lambda p: p.cost)
  return out


def pose(sm: WaistModel, p: Posture) -> None:
  sm.set_waist(p.waist)
  sm.pose(p.q, np.array([math.sin(p.pitch), 0.0, -math.cos(p.pitch)]))


def arm_reach(sm: WaistModel, side: str = "left") -> float:
  """The furthest the grip gets from its shoulder's roll axis, m: the arm
  straight, the wrist straight."""
  q0 = np.zeros(len(hm.JOINT_NAMES))
  sm.set_waist({})
  sm.pose(q0, np.array([0.0, 0.0, -1.0]))
  sh = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id]
  return float(np.linalg.norm(sm.d.site_xpos[sm.grip[side]] - sh))


def plan(sm: WaistModel, side: str, goal: ha.HandGoal, table: list[Posture],
         reach_m: float, flip: bool = True):
  """`reach.plan` over `table`'s postures: the least bent from which the
  hand reaches `goal` and its pre-grasp WITH `REACH_MARGIN_M` to spare,
  else the least bent at full stretch -- the grip's jaws either way round
  where `flip` (as `Reach._planned` tries them). Returns (posture, arm at
  the goal, slack) or None."""
  rots = [goal.rot]
  if flip and goal.rot is not None:
    rots.append(goal.rot @ np.diag([-1.0, -1.0, 1.0]))
  pos = np.asarray(goal.pos, dtype=float)

  def reaches(pos, rot, start=None):
    q, ep, er = rc.solve_seeded(sm, side, pos, rot, start)
    return q if ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL else None

  # the grip asked, then -- only where it plans nothing -- the other way
  # round (`Reach._planned`)
  for rot in rots:
    g = ha.HandGoal(pos, rot, goal.stiffness)
    pre = rc.pregrasp(g)
    for slack in (True, False):
      for p in table:
        if not p.ok:
          continue
        out = pos - p.shoulder[side]
        if np.linalg.norm(out) > reach_m + (rc.REACH_MARGIN_M if slack else 0.0) + 0.005:
          continue
        pose(sm, p)
        if slack and reaches(pos + rc.REACH_MARGIN_M * out / np.linalg.norm(out), rot) is None:
          continue
        q_goal = reaches(pos, rot)
        if q_goal is None:
          continue
        q_pre = reaches(pre.pos, rot, start=q_goal)
        if q_pre is not None:
          return p, q_goal, slack, q_pre, rot
  return None


# ---- what a reach found costs: the table, the balance, the torques ----------

def tilt(deg: float) -> np.ndarray:
  """Fingers ahead and down `deg` below level, the jaws closing across."""
  c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
  return ha.frame((c, 0.0, -s), (0.0, 1.0, 0.0))


UP45 = ha.frame((math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4)),
                (-math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))
#: The grips measured: #615's bands' (fingers down and tilted, the jaws
#: across), our table procedure's (45 deg), and the reach scene's envelope's
#: (`humanoid_spike.envelope_table`: F forward, U tilted 45 deg up, D down).
GRIPS = {"down": ha.frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), "t45": tilt(45.0),
         "t60": tilt(60.0), "F": ha.FORWARD, "U": UP45, "D": ha.DOWN}
#: The table's set-out (`evaluation.counter`): its top, the top's thickness,
#: and the soles' middle 13 cm out of its edge.
TABLE_H, TABLE_TOP_T, TABLE_GAP = 0.72, 0.03, 0.13
#: A reach's load for the part's sizing, kg: the torque table's "an arm out
#: level, 1 kg in the hand" (SimNotes, "The humanoid body").
LOAD_KG = 1.0


def _samples(m, d, g):
  """A solid geom's points with their radius: a box's 27 (its corners, its
  edges' and faces' middles and its centre), a capsule's or a cylinder's
  axis at five points, a sphere's centre."""
  pos, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
  t = m.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    grid = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3))) * size
    return pos + grid @ rot.T, 0.0
  if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
    ks = np.linspace(-1.0, 1.0, 5)[:, None]
    return pos + ks * size[1] * rot[:, 2], float(size[0])
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    return pos[None, :], float(size[0])
  return pos[None, :], 0.0


def into_table(sm: WaistModel, edge: float, top: float = TABLE_H,
               thick: float = TABLE_TOP_T) -> float:
  """How far the robot's solid parts go into the table's top as posed, m: a
  slab `thick` deep under `top`, from `edge` ahead of the soles' middle on
  (a top on legs: under it is room)."""
  m, d = sm.m, sm.d
  worst = 0.0
  for g in range(m.ngeom):
    if m.geom_contype[g] == 0 or m.geom_bodyid[g] == 0:
      continue
    pts, r = _samples(m, d, g)
    inside = (pts[:, 0] + r > edge) & (pts[:, 2] - r < top) & (pts[:, 2] + r > top - thick)
    if inside.any():
      worst = max(worst, float((top - (pts[inside, 2] - r)).max()))
  return worst


def support(sm: WaistModel) -> np.ndarray:
  """The soles' corners on the floor as posed, stance frame (x, y)."""
  pts = []
  for g in sm.soles:
    pos, rot, size = sm.d.geom_xpos[g], sm.d.geom_xmat[g].reshape(3, 3), sm.m.geom_size[g]
    for sx, sy in itertools.product((-1.0, 1.0), repeat=2):
      pts.append((pos + rot @ np.array([sx * size[0], sy * size[1], -size[2]]))[:2])
  return np.array(pts)


def margin(points: np.ndarray, c: np.ndarray) -> float:
  """How far `c` is inside the convex hull of `points`, m (negative: out)."""
  from scipy.spatial import ConvexHull
  eq = ConvexHull(points).equations
  return float(-(eq[:, :2] @ c + eq[:, 2]).max())


def held(sm: WaistModel, payload_kg: float = 0.0, side: str = "left") -> np.ndarray:
  """Every dof's torque holding the statue still as posed, N*m: gravity's
  generalized force (`mj_rne`, still), `payload_kg` hanging at `side`'s
  grip, and the floor's wrench on each sole the least that holds the
  trunk's six rows. Above the hips it is exact (no contact there but the
  payload); the legs' share is that least-squares split between the feet."""
  m, d = sm.m, sm.d
  d.qvel[:] = 0.0
  mujoco.mj_comVel(m, d)
  bias = np.zeros(m.nv)
  mujoco.mj_rne(m, d, 0, bias)
  jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  mujoco.mj_jacSite(m, d, jp, jr, sm.grip[side])
  need = bias - jp.T @ np.array([0.0, 0.0, -9.81 * payload_kg])
  rows = []
  for g in sm.soles:
    a, b = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    bottom = d.geom_xpos[g] - d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][2]
    mujoco.mj_jac(m, d, a, b, bottom, m.geom_bodyid[g])
    rows += [a, b]
  jc = np.vstack(rows)
  f, *_ = np.linalg.lstsq(jc[:, :6].T, need[:6], rcond=None)
  return need - jc.T @ f


def gravity_of(pitch: float) -> np.ndarray:
  return np.array([math.sin(pitch), 0.0, -math.cos(pitch)])


_CACHE: dict = {}


def _models(kind: str):
  if kind not in _CACHE:
    sm = WaistModel(kind)
    _CACHE[kind] = (sm, postures(sm), arm_reach(sm))
  return _CACHE[kind]


def evaluate(task: dict) -> dict:
  """One reach planned on `task["kind"]`'s model of itself: whether and from
  which posture it reaches, with slack or not; then, the statue balanced
  again with the arm out (as the policy moves its pelvis back for one),
  whether it still reaches, its centre of mass's margin inside the soles,
  how far it goes into the table (at its goal and its pre-grasp), and the
  torques it holds with `LOAD_KG` in the hand."""
  sm, table, reach_m = _models(task["kind"])
  side = task.get("side", "left")
  pos = np.array(task["pos"], dtype=float)
  out = dict(task, reached=False)
  t0 = time.perf_counter()
  got = plan(sm, side, ha.HandGoal(pos, GRIPS[task["grip"]]), table, reach_m)
  out["plan_s"] = round(time.perf_counter() - t0, 3)
  if got is None:
    return out
  p, q_goal, slack, q_pre, rot = got
  out.update(reached=True, slack=bool(slack), posture=p.label(), legs=p.legs_index,
             height=p.height, pitch=p.pitch, waist={k: round(v, 4) for k, v in p.waist.items()},
             flipped=bool(not np.allclose(rot, GRIPS[task["grip"]])))
  out.update(costs(sm, side, p, q_goal, q_pre, pos, rot, task.get("edge")))
  return out


def self_clear(sm: WaistModel, side: str) -> float:
  """How far `side`'s arm is from the robot's own trunk, head, legs and other
  arm as posed, m (`mj_geomDistance`; negative, it passes through them):
  the robot's model meets its own parts only in the pairs the world's names
  (`model.ARM_GEOMS`), and a waist turned to reach across may put the arm
  into the torso."""
  m, d = sm.m, sm.d
  other = "right" if side == "left" else "left"
  mine = [f"{side}_{g}" for g in hm.ARM_GEOMS] + [f"{side}_{j}_pad" for j in hm.JAWS]
  theirs = (list(hm.TRUNK_GEOMS) + ["torso_low"]
            + [f"{s}_{g}" for s in hm.SIDES for g in ("thigh", "shank")]
            + [f"{other}_{g}" for g in hm.ARM_GEOMS])
  ids = [m.geom(n).id for n in theirs if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
  worst = 0.05
  for a in (m.geom(n).id for n in mine):
    for b in ids:
      worst = min(worst, float(mujoco.mj_geomDistance(m, d, a, b, 0.05, None)))
  return worst


def costs(sm: WaistModel, side: str, p: "Posture", q_goal, q_pre, pos, rot, edge) -> dict:
  """What a planned reach costs once the statue is balanced again with the
  arm out at its goal (as the policy moves its pelvis back for one): whether
  it balances and still reaches, its centre of mass's margin inside the
  soles, how far it goes into the table and through the robot's own parts
  (at the goal and the pre-grasp), and the torques held with `LOAD_KG` in
  the hand."""
  out = {}
  arm_goal = rc._arm(side, q_goal)
  sm.set_waist(p.waist)
  legs, ok = static_legs(sm, p.height, p.pitch, arms=arm_goal)
  q = p.q.copy()
  for n, v in {**arm_goal, **legs}.items():
    q[hm.JOINT_NAMES.index(n)] = v
  sm.pose(q, gravity_of(p.pitch))
  out["balanced"] = bool(ok)
  com = sm.d.subtree_com[sm.root][:2].copy()
  out["com"] = [round(float(v), 4) for v in com]
  out["margin"] = round(margin(support(sm), com), 4)
  if edge is not None:
    out["into_goal"] = round(into_table(sm, edge), 4)
  out["self_goal"] = round(self_clear(sm, side), 4)
  tau = held(sm, LOAD_KG, side)
  out["tau"] = {n: round(float(tau[sm.waist_dof[n]]), 3) for n in sm.waist_dof}
  for k in ("hip_pitch", "knee", "ankle_pitch"):
    out["tau"][k] = round(max(abs(float(tau[sm.vadr[hm.JOINT_NAMES.index(f"{s}_{k}")]]))
                              for s in hm.SIDES), 3)
  for k in ("shoulder_pitch", "shoulder_roll"):
    out["tau"][k] = round(abs(float(tau[sm.vadr[hm.JOINT_NAMES.index(f"{side}_{k}")]])), 3)
  _, ep, er = rc.solve_seeded(sm, side, pos, rot, q_goal)
  out["still_reaches"] = bool(ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL)
  sm._set_arm(side, q_pre)
  if edge is not None:
    out["into_pre"] = round(into_table(sm, edge), 4)
  out["self_pre"] = round(self_clear(sm, side), 4)
  return out


#: A carry's points this far apart, m, and the lift our procedure gives a
#: piece before it is carried, and lowers it by at the end (`pull`, 5 cm).
CARRY_STEP_M, LIFT_M = 0.02, 0.05


def carry_points(pick, put) -> list[np.ndarray]:
  """Our procedure's carry (`hands_check.TABLE`): the piece lifted, carried
  straight across at that height, and set down where it goes."""
  a = np.asarray(pick, dtype=float) + (0.0, 0.0, LIFT_M)
  b = np.asarray(put, dtype=float) + (0.0, 0.0, LIFT_M)
  n = max(1, math.ceil(float(np.linalg.norm(b - a)) / CARRY_STEP_M))
  pts = [a + (b - a) * (k / n) for k in range(n + 1)]
  m = max(1, math.ceil(LIFT_M / CARRY_STEP_M))
  return pts + [b - np.array([0.0, 0.0, LIFT_M * k / m]) for k in range(1, m + 1)]


def carry_plan(sm: WaistModel, side: str, table: list[Posture], points, rot,
               reach_m: float, chain: bool = False, edge: float | None = None):
  """The least bent posture from which `side`'s grip FOLLOWS `points` in
  turn at `rot` -- the first solved from the seeds, each next on from the
  last, as a carried hand goes (`Reach.carry`, `path_for`) -- within the
  reach's own tolerance; or None. The waist HELD at the posture's (as a
  carry holds its posture today), or, `chain`, IN THE ARM'S CHAIN: turned by
  the same solve as the hand goes, from the posture's, and the statue
  balanced again at the end with the waist and the arm where they got to,
  which must still reach the last point. Returns (posture, the arm at the
  end, the waist at the end, the way's least clearance of the robot's own
  parts and deepest into the table, m: `edge` the table's)."""
  pts = [np.asarray(p, dtype=float) for p in points]
  # a waist in the chain moves the shoulder too: at most about 0.18 m, a turn
  # of 60 deg on the yaw's 0.18 m radius, or the pitch's 0.75 rad on 0.24
  reach_far = reach_m + 0.005 + (0.25 if chain else 0.0)
  for p in table:
    if not p.ok or any(np.linalg.norm(x - p.shoulder[side]) > reach_far for x in pts):
      continue
    pose(sm, p)
    q, ep, er = rc.solve_seeded(sm, side, pts[0], rot)
    ok = ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL
    # the carry's worst clearance of the robot's own parts and of the table
    # on its way, m (`self_clear`, `into_table`)
    worst_self, worst_into = 1.0, 0.0
    for x in pts[1:] if ok else ():
      sm._set_arm(side, q)
      if chain:
        q, ep, er = sm.solve_chain(side, x, rot)
      else:
        q, ep, er = sm.solve(side, x, rot, iters=80)
      if not (ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL):
        ok = False
        break
      worst_self = min(worst_self, self_clear(sm, side))
      if edge is not None:
        worst_into = max(worst_into, into_table(sm, edge))
    if ok and chain:
      # the statue balanced again as it ended, the waist where it got to
      w = dict(sm.waist)
      arm_end = rc._arm(side, q)
      legs, balanced = static_legs(sm, p.height, p.pitch, arms=arm_end)
      qq = p.q.copy()
      for n, v in {**arm_end, **legs}.items():
        qq[hm.JOINT_NAMES.index(n)] = v
      sm.set_waist(w)
      sm.pose(qq, gravity_of(p.pitch))
      _, ep, er = sm.solve_chain(side, pts[-1], rot)
      ok = balanced and ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL
    if ok:
      return p, q, dict(sm.waist), worst_self, worst_into
  return None


def evaluate_piece(task: dict) -> dict:
  """A piece at the table from a stance: picked up (`evaluate`), then carried
  as our procedure carries it to `task["put"]`, the grip's turn the one the
  pick-up took."""
  out = evaluate(task)
  if not out.get("reached"):
    return out
  sm, table, reach_m = _models(task["kind"])
  side = task.get("side", "left")
  rot = GRIPS[task["grip"]]
  if out.get("flipped"):
    rot = rot @ np.diag([-1.0, -1.0, 1.0])
  pts = carry_points(task["pos"], task["put"])
  # the pick-up's own posture first: a carry begins where the reach ended
  first = [p for p in table if p.label() == out["posture"]]
  order = first + [p for p in table if p.label() != out["posture"]]
  for chain, key in ((False, "carried"), (True, "chained")):
    if chain and not sm.waist_adr:
      continue
    t0 = time.perf_counter()
    got = carry_plan(sm, side, order, pts, rot, reach_m, chain=chain, edge=task["edge"])
    out[f"{key}_s"] = round(time.perf_counter() - t0, 3)
    out[key] = got is not None
    if got is not None:
      p, q_end, w_end, worst_self, worst_into = got
      out[f"{key}_posture"] = p.label()
      out[f"{key}_waist"] = {k: round(v, 4) for k, v in w_end.items()}
      out[f"into_{key}"] = round(worst_into, 4)
      out[f"self_{key}"] = round(worst_self, 4)
  return out


def plan_in(sm: WaistModel, side: str, goal: ha.HandGoal, p: "Posture", reach_m: float):
  """`plan`, from posture `p` alone and with slack: the arm at the goal and at
  its pre-grasp, the grip either way round as `plan` takes it; or None."""
  got = plan(sm, side, goal, [p], reach_m)
  return got if got is not None and got[2] else None


#: Each part a joint holds with, for a reach's worst share of its stall
#: rating (`alternatives`): the legs' and the shoulder's as `parts.DRIVES`
#: has them, the waist's the candidates'.
HOLDERS = {"waist_pitch": PITCH_MOTOR, "waist_yaw": YAW_MOTOR, "hip_pitch": hp.RS04,
           "knee": hp.RS04, "ankle_pitch": hp.RS04, "shoulder_pitch": hp.RS03,
           "shoulder_roll": hp.RS03}


def share(tau: dict) -> tuple[float, str]:
  """A reach's worst joint: its held torque's share of its part's stall
  rating, and which joint."""
  return max((abs(v) / hp.stall_torque(HOLDERS[k]), k) for k, v in tau.items())


def alternatives(task: dict) -> dict:
  """EVERY posture from which `task`'s reach plans with slack, each with what
  it costs (`costs`): the part is sized by the least a reach needs, never the
  first posture found."""
  sm, table, reach_m = _models(task["kind"])
  side = task.get("side", "left")
  pos = np.array(task["pos"], dtype=float)
  goal = ha.HandGoal(pos, GRIPS[task["grip"]])
  alts = []
  for p in table:
    if not p.ok:
      continue
    got = plan_in(sm, side, goal, p, reach_m)
    if got is None:
      continue
    _, q_goal, _, q_pre, rot = got
    c = costs(sm, side, p, q_goal, q_pre, pos, rot, task.get("edge"))
    alts.append({"posture": p.label(), "legs": p.legs_index, "height": p.height, "pitch": p.pitch,
                 "waist": {k: round(v, 4) for k, v in p.waist.items()}, **c})
  return dict(task, alts=alts)


def _evaluate(task: dict) -> dict:
  try:
    if task.get("sweep") == "alternatives":
      return alternatives(task)
    return evaluate_piece(task) if "put" in task else evaluate(task)
  except Exception as e:      # a row says what went wrong, never the batch
    return dict(task, error=repr(e))


def run(tasks: list[dict], jobs: int, into: str | None) -> list[dict]:
  """Every task evaluated, `jobs` processes, each row written to
  `into`/rows.jsonl as it comes (one kind's tasks together, so a process
  builds a kind's postures once)."""
  import json
  import os
  tasks = sorted(tasks, key=lambda t: list(CANDIDATES).index(t["kind"]))
  path = None
  if into:
    os.makedirs(into, exist_ok=True)
    path = os.path.join(into, "rows.jsonl")
  rows, t0 = [], time.perf_counter()
  fh = open(path, "a") if path else None
  try:
    if jobs <= 1:
      it = map(_evaluate, tasks)
    else:
      pool = mp.get_context("spawn").Pool(jobs)
      it = pool.imap(_evaluate, tasks, chunksize=4)
    for k, row in enumerate(it, 1):
      rows.append(row)
      if fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
      if k % 50 == 0 or k == len(tasks):
        print(f"  {k}/{len(tasks)} rows, {time.perf_counter() - t0:.0f} s", flush=True)
    if jobs > 1:
      pool.close()
      pool.join()
  finally:
    if fh:
      fh.close()
  return rows


# ---- the sweeps -----------------------------------------------------------------

#: The bands' grid, the left hand at the table's height (a grip's middle 5 cm
#: over its top: the plate's wall gripped at 0.767, the cup at 0.78), m.
BAND_Z = 0.77
BAND_X = tuple(round(0.10 + 0.04 * i, 2) for i in range(14))       # 0.10-0.62
BAND_Y = tuple(round(-0.20 + 0.04 * i, 2) for i in range(21))      # -0.20-0.60


def band_tasks(kinds) -> list[dict]:
  return [{"sweep": "bands", "kind": k, "grip": g, "pos": [x, y, BAND_Z], "edge": TABLE_GAP}
          for k in kinds for g in ("down", "t45", "t60") for x in BAND_X for y in BAND_Y]


#: Our table procedure's reaches (`hands_check.TABLE`), stance frame at the
#: start (the soles' middle 13 cm out of the edge, on the place's line), the
#: set-out's middle (`counter.TABLE_SOURCES`, undrawn): each piece's grip
#: where it is picked up and where it is set down, its hand, and the step
#: the procedure takes first (the stance's offset across, m: 0.26 to the
#: right for the plate, then 0.17 back for the fork).
PIECES = {
  "cup": ("right", (0.42, -0.31, 0.78), (0.42, -0.24, 0.78), 0.0),
  "plate": ("left", (0.43, 0.246, 0.767), (0.30, 0.081, 0.767), -0.26),
  "fork": ("left", (0.43, 0.32, 0.762), (0.30, 0.16, 0.762), -0.09),
}
WINDOW_DY = tuple(round(-0.42 + 0.01 * i, 2) for i in range(69))   # -0.42-0.26
WINDOW_DX = tuple(round(-0.06 + 0.01 * i, 2) for i in range(17))   # -0.06-0.10


def window_tasks(kinds) -> list[dict]:
  """Each piece's two grips from a stance moved (dx ahead, dy across) off
  the start: across at the procedure's own distance, and ahead (nearer the
  table, under it with knee room) where the procedure steps across."""
  out = []
  for k in kinds:
    for name, (side, pick, put, step) in PIECES.items():
      offsets = {(0.0, dy) for dy in WINDOW_DY} | {(dx, step) for dx in WINDOW_DX}
      for dx, dy in sorted(offsets):
        moved = [[round(x - dx, 4), round(y - dy, 4), z] for x, y, z in (pick, put)]
        base = {"sweep": "window", "kind": k, "piece": name, "side": side, "grip": "t45",
                "dx": dx, "dy": dy, "edge": round(TABLE_GAP - dx, 4)}
        # picked up and carried there; and, apart, reached where it goes
        out.append({**base, "which": "pick", "pos": moved[0], "put": moved[1]})
        out.append({**base, "which": "put", "pos": moved[1]})
  return out


#: The reach scene's envelope (`humanoid_spike.ENVELOPE_*`): the left hand
#: in front of its shoulder, 0.2 m out, the floor to over the head.
ENVELOPE_Z = (0.09, 0.2, 0.35, 0.49, 0.65, 0.8, 0.94, 1.1, 1.25, 1.43, 1.55)
ENVELOPE_X = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75)


def envelope_tasks(kinds) -> list[dict]:
  return [{"sweep": "envelope", "kind": k, "grip": g, "pos": [x, 0.2, z]}
          for k in kinds for g in ("F", "U", "D") for z in ENVELOPE_Z for x in ENVELOPE_X]


#: Across the middle: the left hand at the counter's height (0.94 m, fingers
#: forward) and the table's (tilted 45 deg), from its own side to the other.
ACROSS_Y = tuple(round(-0.44 + 0.04 * i, 2) for i in range(27))    # -0.44-0.60
ACROSS = (("counter", 0.94, "F", (0.35, 0.45)), ("table", BAND_Z, "t45", (0.30, 0.42)))


def across_tasks(kinds) -> list[dict]:
  return [{"sweep": "across", "kind": k, "where": w, "grip": g, "pos": [x, y, z],
           **({"edge": TABLE_GAP} if w == "table" else {})}
          for k in kinds for w, z, g, xs in ACROSS for x in xs for y in ACROSS_Y]


# ---- the reports ----------------------------------------------------------------

#: A reach goes INTO the table, or THROUGH the robot's own parts, past this, m.
INTO_M, SELF_M = 0.01, 0.01


def _cell(r: dict | None) -> str:
  """S slack, F full stretch, '.' none; lower case where the arm or the
  trunk goes over a centimetre into the table's top (at the goal or its
  pre-grasp); '?' where the statue balanced again with the arm out no
  longer reaches or balances; '!' where the arm passes over a centimetre
  through the robot's own trunk, legs or other arm."""
  if r is None or not r.get("reached"):
    return "."
  c = "S" if r["slack"] else "F"
  if max(r.get("into_goal", 0.0), r.get("into_pre", 0.0)) > INTO_M:
    c = c.lower()
  if not r.get("still_reaches", True) or not r.get("balanced", True):
    c += "?"
  if min(r.get("self_goal", 1.0), r.get("self_pre", 1.0)) < -SELF_M:
    c += "!"
  return c


def _ok(r: dict) -> bool:
  """Reached, clear of the table and of the robot's own parts, and still
  reached once balanced."""
  return (bool(r.get("reached")) and max(r.get("into_goal", 0.0), r.get("into_pre", 0.0)) <= INTO_M
          and min(r.get("self_goal", 1.0), r.get("self_pre", 1.0)) >= -SELF_M
          and r.get("still_reaches", True) and r.get("balanced", True))


def _by(rows, **want):
  return [r for r in rows if all(r.get(k) == v for k, v in want.items())]


def _kinds(rows):
  return [k for k in CANDIDATES if any(r["kind"] == k for r in rows)]


def report_bands(rows) -> None:
  rows = _by(rows, sweep="bands")
  if not rows:
    return
  print("\nTHE TABLE'S BANDS: the left hand at z", BAND_Z, "m, stance frame (x ahead of the soles' "
        "middle, y to its left), the table's edge", TABLE_GAP, "m ahead. S slack, F full "
        "stretch, lower case over 1 cm into the table's top, ? not once balanced again, . none")
  for kind in _kinds(rows):
    for grip in ("down", "t45", "t60"):
      cells = {(r["pos"][0], r["pos"][1]): r for r in _by(rows, kind=kind, grip=grip)}
      if not cells:
        continue
      print(f"\n== {kind}, {grip}   y \\ x " + " ".join(f"{x:>4.2f}" for x in BAND_X))
      for y in reversed(BAND_Y):
        print(f"   {y:+.2f}          " + " ".join(f"{_cell(cells.get((x, y))):>4s}" for x in BAND_X))
      good = [r for r in cells.values() if _ok(r)]
      if good:
        ys = sorted({r["pos"][1] for r in good})
        print(f"   reached clear: y {ys[0]:+.2f} to {ys[-1]:+.2f}; across the middle (|y| < 0.08): "
              + ", ".join(f"{y:+.2f} at x {min(r['pos'][0] for r in good if r['pos'][1] == y):.2f}-"
                          f"{max(r['pos'][0] for r in good if r['pos'][1] == y):.2f}"
                          for y in ys if abs(y) < 0.08) or "none")
      used = {}
      for r in good:
        for n, v in r.get("waist", {}).items():
          if v:
            used.setdefault(n, []).append(v)
      for n, vs in used.items():
        print(f"   {n} used in {len(vs)} of {len(good)}: {math.degrees(min(vs)):+.0f} to "
              f"{math.degrees(max(vs)):+.0f} deg")


def _intervals(vals: list[float], step: float) -> list[tuple[float, float]]:
  out = []
  for v in sorted(vals):
    if out and v - out[-1][1] <= step + 1e-9:
      out[-1] = (out[-1][0], v)
    else:
      out.append((v, v))
  return out


def _fmt(iv) -> str:
  return ", ".join(f"{a:+.2f}..{b:+.2f} ({100 * (b - a) + 1:.0f} cm)" for a, b in iv) or "none"


def report_window(rows) -> None:
  rows = _by(rows, sweep="window")
  if not rows:
    return
  print("\nTHE STANCE WINDOW: where the stance may be (m off the start: dy to its left, dx "
        "toward the table) and its hand pick a piece up -- the grip our procedure takes, 45 "
        "deg, the jaws across, clear of the table and still once balanced -- AND carry it "
        "where it goes as our procedure does (lifted 5 cm, straight across, set down), from "
        "one posture. A width counts each 1 cm cell.")
  for kind in _kinds(rows):
    print(f"\n== {kind}")
    across, across_chain = {}, {}
    for name, (side, _, _, step) in PIECES.items():
      both = {}
      for r in _by(rows, kind=kind, piece=name):
        v = _ok(r)
        if r["which"] == "pick":
          for key, name_ in (("carried", "carry"), ("chained", "chain")):
            both.setdefault((r["dx"], r["dy"]), {})[name_] = (
              v and r.get(key, False) and r.get(f"into_{key}", 0.0) <= INTO_M
              and r.get(f"self_{key}", 1.0) >= -SELF_M)
        both.setdefault((r["dx"], r["dy"]), {})[r["which"]] = v
      lat = [dy for (dx, dy), v in both.items() if dx == 0.0 and v.get("carry")]
      pick = [dy for (dx, dy), v in both.items() if dx == 0.0 and v.get("pick")]
      put = [dy for (dx, dy), v in both.items() if dx == 0.0 and v.get("put")]
      ahead = [dx for (dx, dy), v in both.items() if dy == step and v.get("carry")]
      chain = [dy for (dx, dy), v in both.items() if dx == 0.0 and v.get("chain")]
      chain_ahead = [dx for (dx, dy), v in both.items() if dy == step and v.get("chain")]
      across[name] = set(lat)
      across_chain.setdefault(name, set()).update(chain)
      print(f"  {name:5s} ({side} hand; the procedure stands at dy {step:+.2f}): picked up and carried {_fmt(_intervals(lat, 0.01))}")
      print(f"        picked up alone {_fmt(_intervals(pick, 0.01))}; set down alone {_fmt(_intervals(put, 0.01))}")
      print(f"        ahead, at dy {step:+.2f}: picked up and carried at dx {_fmt(_intervals(ahead, 0.01))}")
      if any("chained" in r for r in _by(rows, kind=kind, piece=name)):
        print(f"        the waist in the arm's chain: carried {_fmt(_intervals(chain, 0.01))}; "
              f"ahead {_fmt(_intervals(chain_ahead, 0.01))}")
    one = set.intersection(*across.values()) if across else set()
    print(f"  ONE STANCE for the whole place (dx 0): {_fmt(_intervals(sorted(one), 0.01))}")
    if across_chain and all(across_chain.values()):
      one = set.intersection(*across_chain.values())
      print(f"  ...the waist in the arm's chain: {_fmt(_intervals(sorted(one), 0.01))}")


def report_envelope(rows) -> None:
  rows = _by(rows, sweep="envelope")
  if not rows:
    return
  print("\nTHE ENVELOPE: the left hand in front of its shoulder (y 0.2), stance frame; a cell the "
        "grip that plans, with slack first (F forward, U tilted 45 deg up, D down; lower case at "
        "full stretch; ? not once balanced again; . none)")
  for kind in _kinds(rows):
    print(f"\n== {kind}   z \\ x " + " ".join(f"{x:>5.2f}" for x in ENVELOPE_X))
    for z in reversed(ENVELOPE_Z):
      cells = []
      for x in ENVELOPE_X:
        got = [r for r in _by(rows, kind=kind) if r["pos"][0] == x and r["pos"][2] == z and r.get("reached")]
        if not got:
          cells.append(".")
          continue
        best = min(got, key=lambda r: (not r["slack"], not r.get("still_reaches", True), "FUD".index(r["grip"])))
        c = best["grip"] if best["slack"] else best["grip"].lower()
        if not best.get("still_reaches", True) or not best.get("balanced", True):
          c += "?"
        cells.append(c)
      print(f"   {z:5.2f}        " + " ".join(f"{c:>5s}" for c in cells))


def report_across(rows) -> None:
  rows = _by(rows, sweep="across")
  if not rows:
    return
  print("\nACROSS THE MIDDLE: the left hand from its own side (+y) to the other (-y) (S/F/s/f/? "
        "as the bands)")
  for kind in _kinds(rows):
    for where, z, grip, xs in ACROSS:
      for x in xs:
        cells = {r["pos"][1]: r for r in _by(rows, kind=kind, where=where) if r["pos"][0] == x}
        if not cells:
          continue
        good = sorted(y for y, r in cells.items() if _ok(r))
        print(f"  {kind:9s} {where:7s} z {z:.2f} x {x:.2f}: " + "".join(
          f"{_cell(cells.get(y))[:1]}" for y in ACROSS_Y) + f"   clear from y {good[0]:+.2f}" if good
          else f"  {kind:9s} {where:7s} z {z:.2f} x {x:.2f}: none")
  print("  (cells from y " + f"{ACROSS_Y[0]:+.2f} to {ACROSS_Y[-1]:+.2f} by 0.04)")


#: The reaches every posture is tried at (`alternatives`): our table
#: procedure's six grips where it stands for them, the plate's and the fork's
#: from the start too (with a waist, no step?), the table's middle, and the
#: reach scene's four with the floor further out.
ALTERNATIVES = (
  ("cup up", "right", (0.42, -0.31, 0.78), "t45", TABLE_GAP),
  ("cup down", "right", (0.42, -0.24, 0.78), "t45", TABLE_GAP),
  ("plate up, stepped", "left", (0.43, 0.506, 0.767), "t45", TABLE_GAP),
  ("plate down, stepped", "left", (0.30, 0.341, 0.767), "t45", TABLE_GAP),
  ("fork up, stepped", "left", (0.43, 0.41, 0.762), "t45", TABLE_GAP),
  ("fork down, stepped", "left", (0.30, 0.25, 0.762), "t45", TABLE_GAP),
  ("plate up, unstepped", "left", (0.43, 0.246, 0.767), "t45", TABLE_GAP),
  ("plate down, unstepped", "left", (0.30, 0.081, 0.767), "t45", TABLE_GAP),
  ("fork up, unstepped", "left", (0.43, 0.32, 0.762), "t45", TABLE_GAP),
  ("fork down, unstepped", "left", (0.30, 0.16, 0.762), "t45", TABLE_GAP),
  ("table middle, near", "left", (0.30, 0.0, 0.77), "t45", TABLE_GAP),
  ("table middle, far", "left", (0.42, 0.0, 0.77), "t45", TABLE_GAP),
  ("counter 0.94", "left", (0.45, 0.2, 0.94), "F", None),
  ("high shelf 1.43", "left", (0.35, 0.2, 1.43), "U", None),
  ("low shelf 0.49", "left", (0.45, 0.2, 0.49), "F", None),
  ("floor at the toes", "left", (0.17, 0.2, 0.09), "D", None),
  ("floor 0.25 out", "left", (0.25, 0.2, 0.09), "D", None),
  ("floor 0.35 out", "left", (0.35, 0.2, 0.09), "D", None),
)


def alternative_tasks(kinds) -> list[dict]:
  return [{"sweep": "alternatives", "kind": k, "name": n, "side": side, "pos": list(pos),
           "grip": g, **({"edge": e} if e is not None else {})}
          for k in kinds for n, side, pos, g, e in ALTERNATIVES]


def report_alternatives(rows) -> None:
  rows = _by(rows, sweep="alternatives")
  if not rows:
    return
  print("\nEVERY POSTURE THAT REACHES (with slack, clear, still once balanced), 1 kg in the hand: "
        "how many; the one whose worst joint holds the least share of its part's stall "
        "rating (its posture, that share and joint); and, with a waist pitch, the least it holds")
  for name, *_ in ALTERNATIVES:
    print(f"  {name}")
    for kind in _kinds(rows):
      r = next((r for r in rows if r["kind"] == kind and r["name"] == name), None)
      if r is None:
        continue
      alts = [a for a in r.get("alts", []) if _ok({**a, "reached": True})]
      if not alts:
        print(f"    {kind:9s} none")
        continue
      best = min(alts, key=lambda a: share(a["tau"])[0])
      sh, joint = share(best["tau"])
      line = (f"    {kind:9s} {len(alts):3d}: best {best['posture']}, its worst {joint} "
              f"{abs(best['tau'][joint]):.1f} N*m ({100 * sh:.0f} %), knee {best['tau']['knee']:.1f}")
      if "waist_pitch" in best["tau"]:
        least = min(alts, key=lambda a: abs(a["tau"]["waist_pitch"]))
        line += (f"; least waist pitch {abs(least['tau']['waist_pitch']):.1f} N*m at "
                 f"{least['posture']}")
      print(line)


def report_part(rows) -> None:
  """The waist's held torques over every reach the sweeps found clear, with
  `LOAD_KG` in the hand, against each part's stall rating; the legs'."""
  good = [r for r in rows if _ok(r) and "tau" in r]
  if not good:
    return
  print("\nBALANCE over the reaches found (the statue balanced again with the arm out): how many "
        "balanced, and the centre of mass's least margin inside the soles, m")
  found = [r for r in rows if r.get("reached") and "margin" in r]
  for kind in _kinds(found):
    rs = [r for r in found if r["kind"] == kind]
    mg = np.array([r["margin"] for r in rs])
    print(f"  {kind:9s} {sum(r['balanced'] for r in rs)} of {len(rs)} balanced; margin least "
          f"{mg.min():.3f}, 5th percentile {np.percentile(mg, 5):.3f}")
  print(f"\nHELD TORQUES over the reaches found clear, {LOAD_KG:g} kg in the hand, N*m "
        "(the worst; and the 95th percentile):")
  for kind in _kinds(good):
    rs = [r for r in good if r["kind"] == kind]
    keys = [k for k in rs[0]["tau"]]
    line = []
    for k in keys:
      v = np.array([abs(r["tau"][k]) for r in rs])
      line.append(f"{k} {v.max():.1f} ({np.percentile(v, 95):.1f})")
    print(f"  {kind:9s} " + ", ".join(line))
    for n, motor in (("waist_pitch", PITCH_MOTOR), ("waist_yaw", YAW_MOTOR)):
      if n not in keys:
        continue
      v = np.array([abs(r["tau"][n]) for r in rs])
      worst = max(rs, key=lambda r: abs(r["tau"][n]))
      for m_ in (hp.RS03, hp.RS04):
        over = (v > hp.stall_torque(m_)).mean()
        held_s = hp.held_for(m_, float(v.max()), still=True)
        print(f"     {n} on an {m_.name.split()[-1]} (stall {hp.stall_torque(m_):g}): "
              f"{100 * over:.0f} % of reaches past it; the worst held "
              f"{'as long as wanted' if held_s is None else f'{held_s:.0f} s'}")
      print(f"     the worst: {worst['sweep']} {worst.get('piece', '')} {worst['pos']} "
            f"{worst['grip']} from {worst['posture']}")


def mass_table(kinds) -> None:
  """--mass: each candidate's whole mass, what the waist adds, the centre of
  mass standing, and what bows on the pitch (its mass and its centre over
  the axis)."""
  base = None
  for kind in kinds:
    sm = WaistModel(kind)
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    sm.pose(q0, np.array([0.0, 0.0, -1.0]))
    m, d = sm.m, sm.d
    total = float(m.body_subtreemass[sm.root])
    base = total if base is None else base
    com = d.subtree_com[sm.root]
    line = (f"{kind:9s} {total:6.2f} kg (+{total - base:4.2f}); the centre of mass standing "
            f"{com[0]:+.3f} ahead, {com[2]:.3f} up")
    for j in CANDIDATES[kind]:
      b = m.body(f"waist_{j}_link").id
      sub = float(m.body_subtreemass[b])
      c = d.subtree_com[b]
      axis = d.xpos[b]
      line += (f"; on the {j}: {sub:.2f} kg, its centre {c[2] - axis[2]:.3f} over the axis, "
               f"{c[0] - axis[0]:+.3f} ahead")
    print(line)
  print("(a pose held is a stalled motor: the parts' stall ratings, N*m -- "
        + ", ".join(f"{mm.name.split()[-1]} {hp.stall_torque(mm):g}" for mm in
                    (hp.RS02, hp.RS03, hp.RS04)) + ")")


#: The picture's reaches (`--strip`): the cup and the plate where our
#: procedure picks them up from its start, unstepped, and the table's middle.
STRIP = (("the cup, picked up", "right", (0.42, -0.31, 0.78)),
         ("the plate's wall, unstepped", "left", (0.43, 0.246, 0.767)),
         ("the table's middle", "left", (0.42, 0.0, 0.77)))
#: The table as the set-out has it, for the picture: its top's half-size
#: (`home.world.DINING_TABLE_HALF`), the edge `TABLE_GAP` ahead.
TABLE_HALF = 0.4


def strip(kinds, out: str) -> None:
  """--strip: each candidate's statue at each of `STRIP`'s reaches: the
  least bent posture that reaches it with slack and passes every check the
  sweeps make (`_ok`), the arm at its goal and the legs balanced for it, the
  table drawn and a marker where the grip goes; a tile a reach, a row a
  candidate. (The posture's label: the pelvis's height off the stand's and
  its pitch, then the waist's angles, deg.)"""
  from PIL import Image, ImageDraw
  w, h = 420, 340
  table = (f'<geom name="table_top" type="box" size="{TABLE_HALF} {TABLE_HALF} {TABLE_TOP_T / 2}" '
           f'pos="{TABLE_GAP + TABLE_HALF} 0 {TABLE_H - TABLE_TOP_T / 2}" rgba="0.55 0.45 0.35 1" '
           f'contype="0" conaffinity="0"/>')
  marks = "".join(f'<site name="goal{k}" pos="{x} {y} {z}" size="0.018" rgba="0.95 0.2 0.2 1"/>'
                  for k, (_, _, (x, y, z)) in enumerate(STRIP))
  tiles = []
  for kind in kinds:
    sm = WaistModel(kind, scenery=table + marks)
    table_ = postures(sm)
    reach_m = arm_reach(sm)
    r = mujoco.Renderer(sm.m, h, w)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    for name, side, pos in STRIP:
      goal = ha.HandGoal(np.array(pos), GRIPS["t45"])
      got = None
      for p in table_:
        if not p.ok:
          continue
        planned = plan_in(sm, side, goal, p, reach_m)
        if planned is None:
          continue
        _, q_goal, _, q_pre, rot = planned
        if _ok({"reached": True, **costs(sm, side, p, q_goal, q_pre, np.array(pos), rot, TABLE_GAP)}):
          got = (p, q_goal)
          break
      label = f"{kind}: {name}"
      if got is None:
        pose(sm, table_[0])
        label += ": no posture reaches it"
      else:
        p, q_goal = got
        arm_goal = rc._arm(side, q_goal)
        sm.set_waist(p.waist)
        legs, _ = static_legs(sm, p.height, p.pitch, arms=arm_goal)
        q = p.q.copy()
        for n, v in {**arm_goal, **legs}.items():
          q[hm.JOINT_NAMES.index(n)] = v
        sm.pose(q, gravity_of(p.pitch))
        label += f": {p.label()}"
      cam = mujoco.MjvCamera()
      cam.lookat[:] = (0.2, 0.0, 0.8)
      cam.distance, cam.azimuth, cam.elevation = 2.3, 200.0, -26.0
      r.update_scene(sm.d, cam, opt)
      img = Image.fromarray(r.render())
      draw = ImageDraw.Draw(img)
      draw.rectangle((0, 0, w, 18), fill=(20, 20, 24))
      draw.text((6, 4), label, fill=(255, 255, 255))
      tiles.append(img)
    r.close()
  cols = len(STRIP)
  rows = (len(tiles) + cols - 1) // cols
  sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
  for k, tile in enumerate(tiles):
    sheet.paste(tile, (w * (k % cols), h * (k // cols)))
  sheet.save(out)
  print(f"wrote {out}")


def read_rows(into: str) -> list[dict]:
  import json
  import os
  rows = []
  with open(os.path.join(into, "rows.jsonl")) as fh:
    for line in fh:
      rows.append(json.loads(line))
  return rows


def report(rows) -> None:
  errs = [r for r in rows if "error" in r]
  if errs:
    print(f"⚠ {len(errs)} rows raised; the first: {errs[0]['error']}")
  report_bands(rows)
  report_window(rows)
  report_envelope(rows)
  report_across(rows)
  report_alternatives(rows)
  report_part(rows)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--bands", action="store_true")
  ap.add_argument("--window", action="store_true")
  ap.add_argument("--envelope", action="store_true")
  ap.add_argument("--across", action="store_true")
  ap.add_argument("--alternatives", action="store_true",
                  help="every posture at a dozen named reaches, for the part")
  ap.add_argument("--mass", action="store_true", help="each candidate's mass and centre of mass")
  ap.add_argument("--strip", metavar="PNG", help="each candidate's statue at three of the table's reaches")
  ap.add_argument("--all", action="store_true", help="every sweep")
  ap.add_argument("--candidates", default=",".join(CANDIDATES))
  ap.add_argument("--jobs", type=int, default=3)
  ap.add_argument("--into", help="a directory: its rows.jsonl, a row a reach")
  ap.add_argument("--read", metavar="DIR", help="the reports again, off DIR's rows")
  ap.add_argument("--sample", type=int, default=0, help="every Nth task only (a smoke)")
  ap.add_argument("--check", action="store_true", help="compile each candidate and print its mass")
  args = ap.parse_args()
  kinds = [k for k in args.candidates.split(",") if k]
  for k in kinds:
    if k not in CANDIDATES:
      sys.exit(f"no candidate {k!r}: {', '.join(CANDIDATES)}")
  if args.check:
    for kind in kinds:
      sm = WaistModel(kind)
      print(kind, f"{sm.m.body_subtreemass[sm.root]:.2f} kg", sm.m.nq, sm.m.nv, sm.waist_adr)
    return
  if args.read:
    report(read_rows(args.read))
    return
  if args.mass:
    mass_table(kinds)
    return
  if args.strip:
    strip(kinds, args.strip)
    return
  tasks = []
  if args.bands or args.all:
    tasks += band_tasks(kinds)
  if args.window or args.all:
    tasks += window_tasks(kinds)
  if args.envelope or args.all:
    tasks += envelope_tasks(kinds)
  if args.across or args.all:
    tasks += across_tasks(kinds)
  if args.alternatives or args.all:
    tasks += alternative_tasks(kinds)
  if not tasks:
    ap.print_help()
    return
  if args.sample > 1:
    tasks = tasks[::args.sample]
  print(f"{len(tasks)} reaches to plan on {', '.join(kinds)}, {args.jobs} processes", flush=True)
  rows = run(tasks, args.jobs, args.into)
  report(rows)


if __name__ == "__main__":
  main()
