#!/usr/bin/env python3
"""Carrying while walking, measured (#560's next task: the table set from the
restaurant kitchen's shelf). Today a walk carries nothing -- `step` refuses
while a grip reads held, and every walk's hand-over lets go and brings a hand
that is out home (#669, `HumanoidMission.to_walk_routine`) -- so this
INSTRUMENT does what the served loop never does: it poses an arm holding a
piece of the table's dishware and hands the legs to the served walker (walk2)
with the arm held where it is, then walks a course of twists and stands.
Never a served change: the hand-over is this script's own.

On a bare floor, an arm holds a piece:

  CASES  none            the arms home, nothing held: walk2 as served
         <held>_<pose>   held: empty (the left arm posed, nothing in it),
                         plate (left hand), cup (right hand) -- or both,
                         the plate left and the cup right (`both_out`,
                         `both_low`); pose:
           low, tray, out, close   the arm in joint space (`POSES`), its
                                   wrist solved to hold the piece level,
                                   the piece put into the open jaws where
                                   the reference's own shut left it
                                   (`PINCH`) and shut on
           inside                  an empty arm with every joint inside
                                   what walk2's arms trained in
           ref                     the reference's own grip and lift off a
                                   pedestal at the table's height (taken
                                   away once lifted), the arm left as the
                                   reach left it, the posture back at the
                                   stand
  COURSES open loop from standing: the Navigator's walk (0.4 m/s), twice it,
          a half turn on the spot at its clamp (1 rad/s), a walk with a turn,
          a back-off, the expo counter's dining side -> table 0 (`PASS`,
          4.9 m) and the kitchen shelf -> table 0 through the staff door
          (`ROUTE`, ~13 m); closed loop (`GOALS`): a goal 4 m ahead and one
          behind, walked to by the Navigator as served (`goto_route`, the
          route's corners in turn, stalls at them arms home on a bare floor)
  --unseen      the held pieces hidden from every sensor, from the start:
                what a self-filter dropping what the hands hold would see
  --blind-arms  the walker shown its arms at its policy's default, still
  --kg X        every held piece weighs X kg; --squeeze N  the jaws shut at N
  --served      the served walk's own hand-over (#682's stage A), which carries
                a grip that reads held at the carry pose; `ref` left as its
                lift leaves it
  --walker NPZ  another walking policy (#682's stage B), walk2 unset
  --held-drop M the self-filter's radius round a held grip (0.12 unset)

  scripts/carry_spike.py --fly [--cases A,B] [--courses A,B] [--draws N]
                         [--first K] [--jobs J] [--out rows.jsonl] [--trace DIR]
                         [--unseen] [--blind-arms] [--kg X] [--squeeze N]
                         (rows APPENDED to --out: a batch's loops share one file)
  scripts/carry_spike.py --read rows.jsonl[,more]       the tables
  scripts/carry_spike.py --poses                        each pose: its joints against
                                                        walk2's training range, the hand,
                                                        the piece's tilt
  scripts/carry_spike.py --strip CASE:COURSE[:K]        its filmstrip (--strip-out)
  scripts/carry_spike.py --view CASE:COURSE[:K]         in the viewer (MUJOCO_GL unset)

Everything graded here is OUR reading off the simulator's truth (falls, where
the pelvis went, the piece in the jaws); what steers is the robot's own
walker, asked open-loop twists. Each flight draws noise of its own
(`senses.Noise`, keyed by its case, course and draw) and its piece's hidden
load (`dishware.draw(k)`).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from pluggybot.activity import dishware as dw
from pluggybot.body import body_for
from pluggybot.contacts import contact_pairs
from pluggybot.evaluation import counter
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.mission import STILL

#: The dining table's top, m, and the pedestal's (`ref`) near edge this far
#: ahead of the soles' middle: the table check's own (`counter`); and the
#: pedestal's half-extents.
TABLE_H, GAP_M = counter.TABLE_H, counter.TABLE_GAP
PED_HALF = (0.25, 0.35)
#: Each piece as our reference takes it (`hands_check.TABLE`), for `ref`:
#:   side, where it is set out (stance frame at boot, m), the grip off its
#:   middle (x, y, z over the table), the fingers' way and the jaws'
#:   (stance frame), and the reference's squeeze, N (`SQUEEZE_N` shuts it).
#: The plate's wall 1.5 cm inside, the jaws the other way round at 60 deg:
#: with the hand 0.35 m ahead and 0.30 m left the reach plans a square
#: posture; nearer the middle only a waist turned 15-30 deg. The cup from
#: above, 3 cm short of its middle, 3 cm under its lid: at 0.35 m ahead and
#: 0.2 m right the stand itself reaches it.
PIECE = {
  "plate": ("left", (0.35, 0.234), (0.0, 0.066, 0.0534), (0.5, 0.0, -0.866), (0.0, -1.0, 0.0), 40.0),
  "cup": ("right", (0.38, -0.20), (-0.03, 0.0, dw.CUP_H - 0.03), (0.707, 0.0, -0.707),
          (0.0, 1.0, 0.0), 40.0),
}
#: What each case holds, (piece, side): the empty hand's arm is the plate's.
HELD = {"none": (), "empty": ((None, "left"),), "plate": (("plate", "left"),),
        "cup": (("cup", "right"),), "both": (("plate", "left"), ("cup", "right"))}
#: The squeeze a posed piece is shut on, N a pad (the reference's 40);
#: `--squeeze N` another (the Dex1-1's 5-120).
SQUEEZE_N = float(os.environ.get("CARRY_SQUEEZE", "40"))
#: THE PINCH: where the reference's shut and lift left each piece in its
#: jaws, the piece's frame in the grip site's (m, rotation): flown off the
#: pedestal (`ref`'s grip; the plate's mean of draws 0, 1 and 3, 4-11 deg
#: apart; the cup's of 1 and 2 -- the other draws' shuts closed on nothing).
PINCH = {
  "plate": ((0.0852, -0.0147, -0.0562),
            ((0.1853, -0.982, 0.0375), (0.8488, 0.1792, 0.4975), (-0.4952, -0.0603, 0.8667))),
  "cup": ((0.0023, 0.0303, -0.0567),
          ((0.9699, 0.2384, -0.0496), (-0.2007, 0.6674, -0.7172), (-0.1379, 0.7055, 0.6952))),
}
#: The cup's grip is come down to from this far over it (the reference's).
CUP_OVER_M = 0.10
#: The lift: 5 cm up by a pull holding its line, at most 20 N (the reference's).
LIFT_M, LIFT_N = 0.05, 20.0
#: THE CARRY POSES, the left arm's shoulder pitch, roll and yaw and elbow,
#: rad (the right mirrors roll and yaw); the wrist's pitch and roll are
#: solved to hold the piece level (`level_wrist`). Walk2 trained with its
#: arms sent within half of `ARM_GOALS` of hanging (`--poses` says which
#: joint is outside):
#:   low    the forearm forward and down, the hand at the hip's height
#:   tray   the upper arm hanging, the forearm level: a waiter's
#:   out    the arm forward and down, near straight
#:   close  the forearm folded up, the hand in front of the chest
POSES = {
  "low": (-0.40, 0.15, 0.0, 1.00),
  "tray": (-0.20, 0.15, 0.0, 1.60),
  "out": (-0.90, 0.15, 0.0, 0.60),
  "close": (0.15, 0.25, 0.0, 2.00),
  # every joint inside what walk2's arms were sent to, the wrist as it hangs:
  # an empty hand's control (nothing it holds would be level)
  "inside": (-0.60, 0.30, 0.0, 0.90),
}
#: A pose whose wrist is fixed, not solved: (pitch, roll), the left's.
WRIST_FIXED = {"inside": (0.0, 0.0)}
#: walk2's arm goals (`training/humanoid_train/stand.py` `ARM_GOALS`, sent
#: `reach` 0.5 of the way from hanging, `rest_share` 0.5 of the time:
#: `walk.py`'s `ScriptedArmsCfg`), and the hanging pose, the left arm's.
ARM_GOALS = {"shoulder_pitch": (-2.7, 0.6), "shoulder_roll": (0.0, 1.5),
             "shoulder_yaw": (-1.3, 1.3), "elbow": (0.0, 2.0),
             "wrist_pitch": (-1.4, 1.4), "wrist_roll": (-1.6, 1.6)}
ARM_REST = {"shoulder_pitch": 0.0, "shoulder_roll": 0.1, "shoulder_yaw": 0.0, "elbow": 0.3,
            "wrist_pitch": 0.0, "wrist_roll": 0.0}
MIRRORED = ("shoulder_roll", "shoulder_yaw", "wrist_roll")
WALK2_REACH = 0.5
#: The courses: segments of (forward m/s, sideways m/s, turn rad/s, s).
#: 0.4 m/s is the Navigator's cruise (`behavior.navigation.V_MAX`), 1 rad/s
#: the humanoid's clamp on its turns (`mission.W_MAX`), -0.3 its back-off.
V, W = 0.4, 1.0


def _turn(deg: float) -> tuple:
  return (0.0, 0.0, math.copysign(W, deg), math.radians(abs(deg)) / W)


def _walk(m: float, v: float = V) -> tuple:
  return (v, 0.0, 0.0, m / v)


#: THE ROUTE, the restaurant kitchen's shelf to table 0's place
#: (`home.world`, assumed: the shelf's dishware taken standing at its
#: front, no rack or tray): from facing the shelf (27.45, -12.5) to the
#: staff door (25.6, -16.1), through it along the dining room's south wall
#: to the aisle between the tables (21.9, -16.1), up it to (21.9, -12.5),
#: across to table 0's place line (20.8, -12.5) and in to its place (20.8,
#: -11.8): ~13 m and 450 deg of turns, every one on the spot.
ROUTE = (_turn(-117.2), _walk(4.05), _turn(-62.8), _walk(3.7), _turn(-90.0), _walk(3.6),
         _turn(90.0), _walk(1.1), _turn(-90.0), _walk(0.7))
#: ...and THE PASS, the expo counter's dining side (24.5, -13.0, facing it)
#: to table 0's place: half a turn, 3.7 m down the middle aisle, a quarter
#: turn and 1.2 m in -- the carry left where the kitchen hands a piece
#: through the pass (its kitchen side is 2 m and half a turn from the shelf).
PASS = (_turn(180.0), _walk(3.7), _turn(-90.0), _walk(1.2))
COURSES = {
  "straight": (_walk(4.0),),
  "fast": (_walk(4.8, 0.8),),
  "spin": (_turn(180.0),),
  "arc": ((V, 0.0, 0.5, 6.0),),
  "back": ((-0.3, 0.0, 0.0, 3.0),),
  "pass": PASS,
  "route": ROUTE,
}
CARRIES = ("low", "tray", "out", "close")
CASES = ("none",) + tuple(f"{h}_{p}" for h in ("empty", "plate", "cup") for p in CARRIES) \
  + ("empty_inside", "both_out", "both_low", "plate_ref", "cup_ref")
#: Sampled every this many physics steps (10 ms).
EVERY = 5
#: A fall by our reading: the pelvis more than 60 deg off upright, or under
#: this, m.
FALLEN_Z = 0.45
#: Where each flight's samples go, a file a flight (`--trace DIR`): none, unset.
TRACE = os.environ.get("CARRY_TRACE")
#: `--unseen`: the held pieces put in the geom group no sensor casts against
#: (`perception.cull.PARKED_GROUP`) once posed -- what a self-filter that
#: drops what the hands hold would see. Off: every sensor sees them.
UNSEEN = os.environ.get("CARRY_UNSEEN") == "1"
#: `--house`: the served house (`models/home_world.xml`), the robot standing a
#: step back from the restaurant kitchen's shelf facing it (`HOUSE_START`,
#: x, y, heading), the held pieces parked over it until put into its jaws;
#: courses `house_route` (below). Off: a bare floor.
HOUSE = os.environ.get("CARRY_HOUSE") == "1"
HOUSE_START = (26.95, -12.5, 0.0)
#: `--served`: the legs handed over by the SERVED hand-over (#682's stage A,
#: `HumanoidMission.to_walk_routine`), which carries a grip that reads held
#: at the carry pose (`Reach.CARRY_POSE`) -- never this instrument's own --
#: and, `ref`, the arm left as the reference's lift leaves it, its posture
#: as it is: the served walk does the rest. Off: the instrument's hand-over.
SERVED = os.environ.get("CARRY_SERVED") == "1"
#: `--walker NPZ`: the walking policy flown (the served one, walk2, unset):
#: #682's stage B's carrying walker beside it.
WALKER = os.environ.get("CARRY_WALKER")
if WALKER:
  from pluggybot.humanoid import mission as _mission
  _mission.WALK_NPZ = Path(WALKER)
#: `--held-drop M`: the self-filter's radius round a held grip, m
#: (`mission.HELD_DROP_M`, Ben's "about 0.12" unset).
HELD_DROP = os.environ.get("CARRY_HELD_DROP")
if HELD_DROP:
  from pluggybot.humanoid import mission as _mission2
  _mission2.HELD_DROP_M = float(HELD_DROP)
#: `--blind-arms`: the walker shown every arm joint where its policy's
#: default puts it, still (`ArmsAtRest`), whatever the arm does -- does the
#: veer come from the arm's weight, or from inputs walk2 never trained on?
BLIND = os.environ.get("CARRY_BLIND") == "1"
#: `--kg X`: every held piece weighs X kg, not its draw's -- a heavier load
#: than the table's dishware (a stack, a full bowl) for the walker's margin.
KG = float(os.environ["CARRY_KG"]) if os.environ.get("CARRY_KG") else None
#: Held still after a course, s, before its end is read (the walker's settle
#: and the stand's hand-back, then this).
END_HOLD_S = 1.0


def stance_rot(f, j) -> np.ndarray:
  """The hand's turn, stance frame, off its fingers' way and its jaws'
  (`procedure.steps._grip_turn`'s)."""
  a = np.asarray(f, float) / np.linalg.norm(f)
  c = np.asarray(j, float)
  x = c - (c @ a) * a
  x /= np.linalg.norm(x)
  z = -a
  y = np.cross(z, x)
  return np.stack([x, y, z], axis=1)


def _rz(yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _split(case: str) -> tuple[str, str | None]:
  if case == "none":
    return "none", None
  held, pose = case.split("_")
  return held, pose


def pose_joints(pose: str, side: str, piece: str | None) -> dict:
  """The arm's six joints at `pose` for `side`, by name: the shoulder and
  elbow `POSES`', mirrored for the right; the wrist levelling `piece` (the
  plate's pinch for an empty hand), solved on the bare robot standing at
  its keyframe."""
  sp, sr, sy, el = POSES[pose]
  if side == "right":
    sr, sy = -sr, -sy
  q = {"shoulder_pitch": sp, "shoulder_roll": sr, "shoulder_yaw": sy, "elbow": el}
  if pose in WRIST_FIXED:
    wp, wr = WRIST_FIXED[pose]
    return {**q, "wrist_pitch": wp, "wrist_roll": -wr if side == "right" else wr}
  wp, wr, _ = level_wrist(side, q, piece or "plate")
  return {**q, "wrist_pitch": wp, "wrist_roll": wr}


_BARE = None


def _bare():
  global _BARE
  if _BARE is None:
    m = mujoco.MjModel.from_xml_string(hm.body_xml())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    _BARE = (m, d, d.qpos.copy())
  return _BARE


def _tilt_at(side: str, q: dict, piece: str) -> tuple[float, np.ndarray, np.ndarray]:
  """The piece's tilt off level (deg), and the grip site's place and turn
  (world, the bare robot at its keyframe), with the arm at `q`."""
  m, d, q0 = _bare()
  d.qpos[:] = q0
  for name, v in q.items():
    d.qpos[m.jnt_qposadr[m.joint(f"{side}_{name}").id]] = v
  mujoco.mj_kinematics(m, d)
  s = m.site(f"{side}_grip").id
  rg = d.site_xmat[s].reshape(3, 3).copy()
  r = rg @ np.array(PINCH[piece][1])
  return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2])))), d.site_xpos[s].copy(), rg


def level_wrist(side: str, q: dict, piece: str) -> tuple[float, float, float]:
  """The wrist's pitch and roll (inside their ranges) that hold the piece's
  floor most nearly level, the shoulder and elbow at `q`: a grid, then a
  finer one round its best. (pitch, roll, the tilt left, deg)."""
  m, _, _ = _bare()
  lo_p, hi_p = m.jnt_range[m.joint(f"{side}_wrist_pitch").id]
  lo_r, hi_r = m.jnt_range[m.joint(f"{side}_wrist_roll").id]
  best = (math.inf, 0.0, 0.0)
  for wp in np.linspace(lo_p, hi_p, 33):
    for wr in np.linspace(lo_r, hi_r, 41):
      t = _tilt_at(side, {**q, "wrist_pitch": wp, "wrist_roll": wr}, piece)[0]
      if t < best[0]:
        best = (t, wp, wr)
  t, wp0, wr0 = best
  for wp in np.linspace(max(lo_p, wp0 - 0.1), min(hi_p, wp0 + 0.1), 21):
    for wr in np.linspace(max(lo_r, wr0 - 0.1), min(hi_r, wr0 + 0.1), 21):
      t2 = _tilt_at(side, {**q, "wrist_pitch": wp, "wrist_roll": wr}, piece)[0]
      if t2 < best[0]:
        best = (t2, wp, wr)
  return float(best[1]), float(best[2]), float(best[0])


def trained(side: str, name: str) -> tuple[float, float]:
  """The range walk2's scripted arms were sent within for joint `name`."""
  lo, hi = ARM_GOALS[name]
  r = ARM_REST[name]
  if side == "right" and name in MIRRORED:
    lo, hi, r = -hi, -lo, -r
  return r + WALK2_REACH * (lo - r), r + WALK2_REACH * (hi - r)


def build(case: str, k: int, course: str = ""):
  """The robot at its keyframe on a bare floor and, for a held piece, the
  piece -- for `ref`, on a pedestal at the table's height, set out where the
  reach takes it with a square posture; otherwise parked off to the side,
  to be put into the jaws. The model, the data, the body (its noise keyed by
  case, course and draw) and what the instrument knows of its set-out."""
  if HOUSE:
    return _build_house(case, k, course)
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  held, pose = _split(case)
  pieces = [(p, s) for p, s in HELD[held] if p]
  load = dw.draw(k)
  # where the soles' middle stands at the keyframe, by the truth: the
  # instrument's set-out, never anything the robot reads (off the bare
  # robot: a spec compiled once finds no body added after)
  m0, d0, q0 = _bare()
  d0.qpos[:] = q0
  mujoco.mj_kinematics(m0, d0)
  soles = [d0.geom_xpos[m0.geom(f"{s}_sole").id][:2] for s in hm.SIDES]
  sx, sy = (soles[0] + soles[1]) / 2
  info = {"case": case, "held": held, "pose": pose, "arms": HELD[held], "pieces": pieces,
          "k": k, "stance0": [float(sx), float(sy)], "at": {}, "kg": {},
          "load": {"plate_kg": load.plate_kg, "cup_kg": load.cup_kg, "friction": load.friction}}
  for n, (piece, _) in enumerate(pieces):
    if pose == "ref":
      _, at, _, _, _, _ = PIECE[piece]
      ped = spec.worldbody.add_body(name="pedestal", mocap=True,
                                    pos=[sx + GAP_M + PED_HALF[0], sy, 0.0])
      ped.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[PED_HALF[0], PED_HALF[1], TABLE_H / 2], pos=[0, 0, TABLE_H / 2],
                   rgba=[0.55, 0.42, 0.30, 1])
      xy, z = (sx + at[0], sy + at[1]), TABLE_H
    else:
      xy, z = (sx - 3.0, sy + 3.0 + n), 0.0
    kg = KG if KG is not None else getattr(load, f"{piece}_kg")
    dw.ATTACH[piece](spec, xy, 0.0, z, kg, load.friction, tags=False)
    if pose != "ref":
      # ⚠ weightless until held (`setup_posed`): a body's gravity compensation
      # switched on only at run time does nothing where the model compiled
      # with none (MuJoCo's `ngravcomp`), and the plate fell 19 cm in 0.2 s
      spec.body(piece).gravcomp = 1.0
    info["at"][piece] = [float(xy[0]), float(xy[1])]
    info["kg"][piece] = kg
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for piece, _ in pieces:
    # the keyframe knows no piece: its free joint as the spec put it
    a = int(model.jnt_qposadr[model.joint(f"{piece}_free").id])
    data.qpos[a:a + 7] = model.qpos0[a:a + 7]
  mujoco.mj_forward(model, data)
  body = body_for(model, data, noise=f"carry-{case}-{course}-{k}")
  return model, data, body, info


def _build_house(case: str, k: int, course: str):
  """`build` in the served house: the robot at `HOUSE_START`, standing as it
  boots, its map's bounds the house's, the pieces parked 3 m over it
  (weightless: out of every sensor's band) until put into its jaws."""
  import json
  from pluggybot.perception.cull import fix_static_inertials
  held, pose = _split(case)
  if pose == "ref":
    raise ValueError("--house puts pieces into the jaws: no `ref` (its pedestal is a bare floor's)")
  pieces = [(p, s) for p, s in HELD[held] if p]
  load = dw.draw(k)
  spec = mujoco.MjSpec.from_file(counter.HOME_XML)
  x0, y0, yaw = HOUSE_START
  frame = spec.worldbody.add_frame()
  frame.pos = [x0, y0, 0.0]
  frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  spec.attach(hm.attachable(), prefix="", frame=frame)
  info = {"case": case, "held": held, "pose": pose, "arms": HELD[held], "pieces": pieces,
          "k": k, "at": {}, "kg": {}, "house": True,
          "load": {"plate_kg": load.plate_kg, "cup_kg": load.cup_kg, "friction": load.friction}}
  for n, (piece, _) in enumerate(pieces):
    kg = KG if KG is not None else getattr(load, f"{piece}_kg")
    dw.ATTACH[piece](spec, (x0 - 0.3 * n, y0), 0.0, 3.0, kg, load.friction, tags=False)
    spec.body(piece).gravcomp = 1.0
    info["kg"][piece] = kg
  fix_static_inertials(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  for name, v in hm.stand_qpos().items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = v
  mujoco.mj_forward(model, data)
  b = json.load(open(counter.HOME_META))["gridBounds"]
  bounds = (b[0], b[1], b[2], b[3]) if isinstance(b, list) else (b["xMin"], b["yMin"], b["xMax"], b["yMax"])
  body = body_for(model, data, grid_bounds=bounds, noise=f"carry-house-{case}-{course}-{k}")
  return model, data, body, info


class Reading:
  """Our reading of a flight, every `EVERY` physics steps: the truth."""

  def __init__(self, model, data, body, info):
    self.m, self.d, self.body, self.info = model, data, body, info
    self.root = model.body(hm.ROBOT_ROOT).id
    self.pieces = list(info["pieces"])
    self.rows: list = []
    #: corners of a route the drive gave up on near enough to go on from
    self.stalls = [0]
    self.robot = set()
    for g in range(model.ngeom):
      b = int(model.geom_bodyid[g])
      while b > 0 and b != self.root:
        b = int(model.body_parentid[b])
      if b == self.root:
        self.robot.add(g)
    self.robot_ids = np.array(sorted(self.robot))
    #: by piece, every geom of the world it touched while sampled
    self.touched: dict[str, set] = {}
    self.grip, self.pb, self.pads, self.mine = {}, {}, {}, {}
    self.mine_ids, self.pad_ids, self.other_ids = {}, {}, {}
    for piece, side in self.pieces:
      self.grip[piece] = model.site(f"{side}_grip").id
      self.pb[piece] = model.body(piece).id
      self.pads[piece] = {model.geom(f"{side}_{j}_pad").id for j in hm.JAWS}
      self.mine[piece] = {g for g in range(model.ngeom) if model.geom_bodyid[g] == self.pb[piece]}
      self.mine_ids[piece] = np.array(sorted(self.mine[piece]))
      self.pad_ids[piece] = np.array(sorted(self.pads[piece]))
      self.other_ids[piece] = np.array(sorted(self.robot - self.pads[piece]))

  def pelvis(self):
    d = self.d
    r = d.xmat[self.root].reshape(3, 3)
    return (float(d.xpos[self.root][0]), float(d.xpos[self.root][1]),
            math.atan2(r[1, 0], r[0, 0]), float(d.xpos[self.root][2]),
            math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2])))))

  def in_jaws(self, piece: str):
    """`piece`'s pose in its grip site's frame: (position m, rotation), how
    many of its hand's pads touch it, and whether anything else of the robot
    does."""
    d = self.d
    gs, pb = self.grip[piece], self.pb[piece]
    rg = d.site_xmat[gs].reshape(3, 3)
    p = rg.T @ (d.xpos[pb] - d.site_xpos[gs])
    r = rg.T @ d.xmat[pb].reshape(3, 3)
    pairs = contact_pairs(d)
    hit = np.isin(pairs, self.mine_ids[piece])
    met = np.concatenate((pairs[hit[:, 0], 1], pairs[hit[:, 1], 0]))
    pads = int(np.isin(self.pad_ids[piece], met).sum())
    other = bool(np.isin(met, self.other_ids[piece]).any())
    # ...and what it meets of the world: neither the robot nor itself
    world = met[~np.isin(met, self.robot_ids) & ~np.isin(met, self.mine_ids[piece])]
    self.touched.setdefault(piece, set()).update(
      self.m.geom(int(g)).name or f"geom{int(g)}" for g in world)
    return p, r, pads, other

  def piece_tilt(self, piece: str) -> float:
    r = self.d.xmat[self.pb[piece]].reshape(3, 3)
    return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))

  def sample(self, phase: str) -> None:
    x, y, h, z, tilt = self.pelvis()
    row = {"t": float(self.d.time), "phase": phase, "x": x, "y": y, "h": h, "z": z, "tilt": tilt,
           "falls": int(self.body.mission.falls), "legs": self.body.mission.legs}
    for piece, side in self.pieces:
      p, r, pads, other = self.in_jaws(piece)
      row[piece] = {"pp": p.tolist(), "pr": r.tolist(), "pads": pads, "other": other,
                    "pz": float(self.d.xpos[self.pb[piece]][2]), "ptilt": self.piece_tilt(piece),
                    "held": bool(self.body.gripped(side))}
    self.rows.append(row)


class ArmsAtRest:
  """A walker's joint senses with its arms read where its policy's default
  puts them and still (`--blind-arms`): the legs' and the waist's as read,
  the same draws of the same noise."""

  def __init__(self, walker):
    self.inner = walker.joints
    order = {n: k for k, n in enumerate(hm.JOINT_NAMES)}
    arm = [(order[n], walker.policy.default_q[k]) for k, n in enumerate(walker.policy.joint_names)
           if n.split("_", 1)[1] in hm.ARM]
    self.idx = np.array([a for a, _ in arm])
    self.rest = np.array([r for _, r in arm])

  def read(self, data):
    q, qd, tau = self.inner.read(data)
    q, qd = q.copy(), qd.copy()
    q[self.idx], qd[self.idx] = self.rest, 0.0
    return q, qd, tau


def _angle(r: np.ndarray) -> float:
  return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def _stand_routine(mi, budget_s: float = 6.0):
  """The posture back at the stand and the waist home, waited for: True once
  the standing policy holds it still."""
  mi.reach.hold_posture(0.0, 0.0, 0.0)
  for _ in range(int(round(budget_s / mi.model.opt.timestep))):
    if mi.stand.settled and mi.reach.waist_home() and mi.reach._posture_held():
      return True
    yield STILL
  return False


def _joints_routine(mi, side: str, q, settle_s: float = 0.5):
  """The arm walked in joint space to `q` (`Reach.set_joints`, its own
  ramp), then `settle_s` still."""
  mi.reach.set_joints(side, np.asarray(q, dtype=float))
  for _ in range(int(round(8.0 / mi.model.opt.timestep))):
    if mi.reach.done(side):
      break
    yield STILL
  for _ in range(int(round(settle_s / mi.model.opt.timestep))):
    yield STILL


def _pedestal_away(model, data) -> None:
  ped = model.body("pedestal").id
  data.mocap_pos[model.body_mocapid[ped]] = [50.0, 50.0, -5.0]
  for g in range(model.ngeom):
    if model.geom_bodyid[g] == ped:
      model.geom_contype[g] = model.geom_conaffinity[g] = 0


def setup_ref(body, model, data, info) -> dict:
  """`ref`: the piece gripped as the reference grips it and lifted 5 cm off
  the pedestal, the pedestal taken away; the arm HELD IN JOINT SPACE where
  the reach left it and the posture back at the stand (the hand rides up
  with the trunk: held in the stance frame instead, the arm's solve ran off
  onto a branch with the shoulder thrown back)."""
  mi = body.mission
  (piece, _), = info["pieces"]
  side, _, off, f, j, _ = PIECE[piece]
  squeeze = SQUEEZE_N
  px, py = info["at"][piece]
  _, _, byaw = body.stance
  rot = _rz(byaw) @ stance_rot(f, j)
  grip = np.array([px + off[0], py + off[1], TABLE_H + off[2]])
  out = {"ok": False}
  body.run(body.grip_routine(side, False))
  if piece == "cup":
    if not body.run(body.reach_routine(side, grip + [0.0, 0.0, CUP_OVER_M], rot)):
      return {**out, "why": f"reach over the cup: {body.hand_why(side)}"}
    # ...and down onto it as the reference comes down, a pull of 12 N giving
    # across its way (`hands_check.TABLE`'s `go`)
    out["down"] = body.run(body.pull_routine(side, (0.0, 0.0, -1.0), 12.0, CUP_OVER_M,
                                             give=True))["why"]
  elif not body.run(body.reach_routine(side, grip, rot)):
    return {**out, "why": f"reach: {body.hand_why(side)}"}
  out["posture_at_grip"] = [float(v) for v in mi.reach.posture]
  held = body.run(body.grip_routine(side, True, squeeze))
  if not held:
    body.run(body.grip_routine(side, False))
    held = body.run(body.grip_routine(side, True, squeeze - 10.0))
  if not held:
    return {**out, "why": "not held"}
  out["lift"] = body.run(body.pull_routine(side, (0.0, 0.0, 1.0), LIFT_N, LIFT_M,
                                           give=False))["why"]
  if not body.gripped(side):
    return {**out, "why": f"lift: {out['lift']}"}
  _pedestal_away(model, data)
  if SERVED:
    # ...the arm as the lift left it: the served walk's hand-over carries it
    out["ok"] = True
    return out
  body.run(_joints_routine(mi, side, mi.reach.q_cmd[side].copy(), settle_s=0.2))
  out["stood"] = body.run(_stand_routine(mi))
  if not out["stood"]:
    # ...handed to the walker bowed or turned, a flight would book the
    # hand-over's own stumble to the carry
    return {**out, "why": "never stood back at the stand"}
  out["ok"] = True
  return out


def _hold_out(mi, side: str) -> None:
  """The hand held out at the grip its commanded arm puts it at, as a reach
  holding there leaves it (phase `hold`, its goal and its command that
  pose): an instrument's set-out for the served hand-over, which carries only
  a hand out at a pose of its own."""
  from pluggybot.humanoid.arms import HandGoal
  r = mi.reach
  sm = r._planner_model()
  sm.q = r.self_model.q.copy()
  sm.d.qpos[:] = r.self_model.d.qpos
  sm._set_arm(side, r.q_cmd[side])
  pos, rot = sm.grip_pose(side)
  r.goal[side] = HandGoal(pos.copy(), rot.copy(), 1.0)
  r.cmd[side] = HandGoal(pos.copy(), rot.copy(), 1.0)
  r.phase[side] = "hold"


def setup_posed(body, model, data, info) -> dict:
  """A pose of `POSES` for each arm the case holds with: the jaws opened,
  the arm walked in joint space to the pose (its wrist levelling its piece),
  the piece put into the open jaws where the reference's shut left it
  (`PINCH`) -- weightless and still while they shut on it at `SQUEEZE_N`,
  its weight its own once held."""
  mi = body.mission
  pose = info["pose"]
  out = {"ok": False, "wrist": {}}
  for piece, side in info["arms"]:
    q = pose_joints(pose, side, piece)
    out["wrist"][side] = [q["wrist_pitch"], q["wrist_roll"]]
    body.run(body.grip_routine(side, False))
    body.run(_joints_routine(mi, side, [q[n] for n in hm.ARM]))
    if piece is None:
      body.run(body.grip_routine(side, True, SQUEEZE_N))
      continue
    pb = model.body(piece).id
    gs = model.site(f"{side}_grip").id
    p_rel, r_rel = np.array(PINCH[piece][0]), np.array(PINCH[piece][1])
    rg = data.site_xmat[gs].reshape(3, 3)
    pos = data.site_xpos[gs] + rg @ p_rel
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rg @ r_rel).flatten())
    jnt = model.joint(f"{piece}_free").id
    a, v = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
    data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = pos, quat
    data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(model, data)
    held = body.run(body.grip_routine(side, True, SQUEEZE_N))
    model.body_gravcomp[pb] = 0.0
    if not held:
      return {**out, "why": f"{piece} not held"}
    # ...the arm held where it is commanded while the other is posed -- or,
    # `SERVED`, the hand held OUT at the pose it is at, as a reach leaves one
    # holding there: the served hand-over lifts and carries only such a hand
    mi.reach.set_joints(side, mi.reach.q_cmd[side].copy())
    if SERVED:
      _hold_out(mi, side)
  out["ok"] = True
  return out


def setup(body, model, data, info, reading: Reading | None) -> dict:
  """Stood a second, then the arms and the pieces as the case says, and a
  second still: what happened, or why it did not. `UNSEEN`, the pieces are
  hidden from every cast from the start (`hide`): shown during the setup,
  frames of them stood in the map and the camera's layer before the walk."""
  mi = body.mission
  if UNSEEN and reading is not None:
    hide(model, info, reading)
  body.run(body.hold_routine(1.0))
  if info["held"] == "none":
    return {"ok": True}
  out = setup_ref(body, model, data, info) if info["pose"] == "ref" else \
    setup_posed(body, model, data, info)
  if not out.get("ok"):
    return out
  # ...each arm held in joint space where it is commanded: the trunk carries it
  # (the served hand-over takes it from where it is, `SERVED`)
  if not SERVED:
    for _, side in info["arms"]:
      mi.reach.set_joints(side, mi.reach.q_cmd[side].copy())
  body.run(body.hold_routine(1.0))
  out["q_arm"] = {side: [float(v) for v in mi.reach.q_cmd[side]] for _, side in info["arms"]}
  out["falls"] = int(mi.falls)
  out["hand"] = {side: [float(v) for v in body.to_stance(body.hand_at(side))]
                 for _, side in info["arms"]}
  if reading is not None:
    reading.sample("posed")
  for piece, side in info["pieces"]:
    held = bool(body.gripped(side))
    out.setdefault("held", {})[piece] = held
    if reading is not None:
      _, _, pads, other = reading.in_jaws(piece)
      out.setdefault("pads", {})[piece] = pads
      out.setdefault("other", {})[piece] = other
      out.setdefault("ptilt", {})[piece] = reading.piece_tilt(piece)
    if not held:
      return {**out, "ok": False, "why": f"{piece} lost as it was posed"}
  if mi.falls:
    return {**out, "ok": False, "why": "fell as it was posed"}
  return out


#: Closed-loop courses: a goal in the frame the robot stands in at the start
#: (ahead m, left m), walked to by the Navigator as served (`go_to_routine`:
#: its plan, its drive, its front stop), its odometry its belief.
GOALS = {"goto": ((4.0, 0.0),), "goto_round": ((-2.5, 1.5),),
         # `--house` alone: the shelf -> table 0 route's corners in the house,
         # from `HOUSE_START` -- before the staff door (25.6, -16.1), the
         # aisle's foot (21.9, -16.1) and head (21.9, -12.5), and table 0's
         # place's standoff (20.8, -12.3), half a metre short of its edge (a
         # job's `step` walks the last of it)
         "house_route": ((-1.35, -3.6), (-5.05, -3.6), (-5.05, 0.0), (-6.15, 0.2)),
         # the shelf -> table 0 route's corners (`ROUTE`), in the frame of the
         # robot facing the shelf: each walked to in turn
         "goto_route": ((-1.85, -3.6), (-5.55, -3.6), (-5.55, 0.0), (-6.65, 0.0), (-6.65, 0.7))}
GOAL_PATIENCE_S = 60.0
#: A route's corner the drive gave up on within this, by its belief, m, is
#: passed on from (`course_routine`).
CORNER_M = 0.75


def hide(model, info, reading: Reading) -> None:
  """The held pieces put in the geom group no sensor casts against
  (`perception.cull.PARKED_GROUP`)."""
  from pluggybot.perception.cull import PARKED_GROUP
  for piece, _ in info["pieces"]:
    for g in reading.mine[piece]:
      model.geom_group[g] = PARKED_GROUP


def before_walk(model, body, info, reading: Reading) -> dict:
  """The switches a flight walks under, once posed: the held pieces hidden
  from every sensor (`UNSEEN`; `setup` hid them already), the walker shown
  its arms at rest (`BLIND`). What was switched, for the row."""
  out = {}
  if UNSEEN:
    hide(model, info, reading)
    out["unseen"] = True
  if BLIND:
    body.mission.walker.joints = ArmsAtRest(body.mission.walker)
    out["blind"] = True
  return out


def course_routine(body, course: str, reading: Reading | None, falls0: int):
  """The legs handed to the walker WITH the arms held where they are (never
  the served hand-over, which lets go and brings an arm home), then the
  course -- each segment's twist, one physics step at a time, or a goal the
  Navigator walks to -- then still and the legs back to the stand. Returns
  how it ended."""
  mi = body.mission
  if not SERVED:
    mi.walker.last_action[:] = 0.0
    mi.legs = "walk"
  n = [0]

  def tick(phase: str):
    n[0] += 1
    if reading is not None and n[0] % EVERY == 0:
      reading.sample(phase)

  stalls = reading.stalls if reading is not None else [0]
  if course in GOALS:
    bx, by, bh = mi.pose
    c, s = math.cos(bh), math.sin(bh)
    arrived = True
    for gx, gy in GOALS[course]:
      drive = body.go_to_routine(bx + c * gx - s * gy, by + s * gx + c * gy,
                                 timeout=GOAL_PATIENCE_S)
      try:
        command = next(drive)
        while True:
          command = yield command
          tick("walk")
          if mi.falls != falls0:
            drive.close()
            return "fell"
          command = drive.send(command)
      except StopIteration as done:
        arrived = bool(done.value)
      if not arrived:
        # ...a CORNER given up within `CORNER_M` by its belief goes on to the
        # next: the arms-home walk stalled 0.54 m short of the route's first
        # (the Navigator's own last half metre, its turns on the spot slow);
        # the walk's last goal given up is given up
        last = (gx, gy) == GOALS[course][-1]
        if last or (mi.last_drive or {}).get("shortM", math.inf) > CORNER_M:
          break
        stalls[0] += 1
        arrived = True
  else:
    arrived = True
    for vx, vy, wz, secs in COURSES[course]:
      for _ in range(int(round(secs / mi.model.opt.timestep))):
        yield from mi._twist_routine(vx, vy, wz)
        tick("walk")
        if mi.falls != falls0:
          return "fell"
  yield from mi.to_stand_routine(getup=False)
  if mi.falls != falls0:
    return "fell standing"
  for _ in range(int(round(END_HOLD_S / mi.model.opt.timestep))):
    yield STILL
    tick("end")
    if mi.falls != falls0:
      return "fell standing"
  return "walked" if arrived else f"gave up: {(mi.last_drive or {}).get('why')}"


def asked_end(course) -> tuple[float, float, float]:
  """Where the course ends as asked, in the frame it began in: (x m, y m,
  turn deg) -- its twists integrated, or its goal (its turn unasked: 0)."""
  if course in GOALS:
    return (*GOALS[course][-1], 0.0)
  segments = COURSES[course]
  x = y = h = 0.0
  dt = 0.01
  for vx, vy, wz, secs in segments:
    for _ in range(int(round(secs / dt))):
      x += (vx * math.cos(h) - vy * math.sin(h)) * dt
      y += (vx * math.sin(h) + vy * math.cos(h)) * dt
      h += wz * dt
  return x, y, math.degrees(math.remainder(h, 2 * math.pi))


def fly(job) -> dict:
  """One flight: a case, a course, a draw. A row of what happened."""
  case, course, k = job
  t_wall = time.time()
  model, data, body, info = build(case, k, course)
  reading = Reading(model, data, body, info)
  row = {"case": case, "course": course, "k": k, "load": info["load"], "kg": info.get("kg")}
  if KG is not None:
    row["kg_override"] = KG
  if HOUSE:
    row["house"] = True
  if SQUEEZE_N != 40.0:
    row["squeeze"] = SQUEEZE_N
  try:
    s = setup(body, model, data, info, reading)
    row["setup"] = s
    if not s.get("ok"):
      row["result"] = "setup failed"
      return row
    mi = body.mission
    row.update(before_walk(model, body, info, reading))
    falls0 = mi.falls
    x0, y0, h0, _, _ = reading.pelvis()
    bx0, by0, bh0 = mi.pose
    t0 = float(data.time)
    reading.sample("start")
    i0 = len(reading.rows)
    how = body.run(course_routine(body, course, reading, falls0))
    x1, y1, h1, _, _ = reading.pelvis()
    c, sn = math.cos(h0), math.sin(h0)
    dx, dy = x1 - x0, y1 - y0
    ex, ey, eh = asked_end(course)
    walk = [r for r in reading.rows[i0:] if r["phase"] == "walk"] or reading.rows[i0:]
    tilts = [r["tilt"] for r in walk] or [0.0]
    after = reading.rows[i0:]
    fell_truth = any(r["tilt"] > 60.0 or r["z"] < FALLEN_Z for r in after)
    first_fall = next((r["t"] - t0 for r in after
                       if r["falls"] != falls0 or r["tilt"] > 60.0 or r["z"] < FALLEN_Z), None)
    bx1, by1, bh1 = mi.pose
    row.update({
      "result": how, "fell": bool(how.startswith("fell") or fell_truth or mi.falls != falls0),
      "fall_s": first_fall, "seconds": float(data.time) - t0,
      "end": [c * dx + sn * dy, -sn * dx + c * dy,
              math.degrees(math.remainder(h1 - h0, 2 * math.pi))],
      "asked": [ex, ey, eh],
      "belief_end": [math.cos(bh0) * (bx1 - bx0) + math.sin(bh0) * (by1 - by0),
                     -math.sin(bh0) * (bx1 - bx0) + math.cos(bh0) * (by1 - by0),
                     math.degrees(math.remainder(bh1 - bh0, 2 * math.pi))],
      "tilt_max": float(max(tilts)), "tilt_p95": float(np.percentile(tilts, 95)),
      "tilt_mean": float(np.mean(tilts)),
    })
    if course in GOALS:
      row["drive"] = mi.last_drive
      row["corners_stalled"] = reading.stalls[0]
    if SERVED:
      row["served"] = True
      row["carried"] = {s_: bool(mi.reach.carried(s_)) for _, s_ in info["arms"]}
      row["q_carried"] = {s_: [float(v) for v in mi.reach.q_cmd[s_]] for _, s_ in info["arms"]}
    if WALKER:
      row["walker"] = os.path.basename(WALKER)
    if HELD_DROP:
      row["held_drop"] = float(HELD_DROP)
    per = {}
    for piece, _ in info["pieces"]:
      start = reading.rows[i0 - 1][piece]
      seen = [r[piece] for r in after]
      p0, r0 = np.array(start["pp"]), np.array(start["pr"])
      slip = [float(np.linalg.norm(np.array(r["pp"]) - p0)) for r in seen]
      turn = [_angle(r0.T @ np.array(r["pr"])) for r in seen]
      per[piece] = {
        "slip_max_cm": 100 * max(slip), "slip_end_cm": 100 * slip[-1],
        "turn_max_deg": max(turn), "turn_end_deg": turn[-1],
        "pads_lost_s": sum(r["pads"] < 2 for r in seen) * EVERY * model.opt.timestep,
        # out of the jaws, whatever else of the robot it may rest on
        "dropped": bool(seen[-1]["pads"] == 0),
        "piece_z_end": seen[-1]["pz"], "held_end": seen[-1]["held"],
        "touched_body": any(r["other"] for r in seen),
        "touched_world": sorted(reading.touched.get(piece, ())),
        "ptilt_start": start["ptilt"], "ptilt_max": max(r["ptilt"] for r in seen)}
    if per:
      # the case's reading: the worse piece's
      row["pieces"] = per
      row.update({
        "slip_max_cm": max(v["slip_max_cm"] for v in per.values()),
        "turn_max_deg": max(v["turn_max_deg"] for v in per.values()),
        "pads_lost_s": max(v["pads_lost_s"] for v in per.values()),
        "dropped": any(v["dropped"] for v in per.values()),
        "held_end": all(v["held_end"] for v in per.values()),
        "touched_body": any(v["touched_body"] for v in per.values()),
        "ptilt_start": max(v["ptilt_start"] for v in per.values()),
        "ptilt_max": max(v["ptilt_max"] for v in per.values())})
    if TRACE:
      Path(TRACE).mkdir(parents=True, exist_ok=True)
      with open(Path(TRACE) / f"{case}-{course}-{k}.json", "w") as fh:
        json.dump(reading.rows, fh)
  except Exception as e:  # a flight that raises is a row of its own
    row["result"] = f"raised {type(e).__name__}: {e}"
  finally:
    row["wall_s"] = round(time.time() - t_wall, 1)
    body.close()
  return row


def _fly(job):
  try:
    return fly(job)
  except Exception as e:
    return {"case": job[0], "course": job[1], "k": job[2], "result": f"raised {e!r}"}


def fly_all(cases, courses, first: int, draws: int, jobs: int, out: str) -> None:
  todo = [(c, r, k) for c in cases for r in courses for k in range(first, first + draws)]
  print(f"{len(todo)} flights (draws {first}-{first + draws - 1}), {jobs} at a time -> {out}",
        flush=True)
  t = time.time()
  with open(out, "a") as fh:
    if jobs <= 1:
      for job in todo:
        row = _fly(job)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(_line(row), flush=True)
    else:
      with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_fly, todo, chunksize=1):
          fh.write(json.dumps(row) + "\n")
          fh.flush()
          print(_line(row), flush=True)
  print(f"done in {time.time() - t:.0f} s", flush=True)


def _line(r: dict) -> str:
  if "end" not in r:
    return f"{r['case']:12s} {r['course']:9s} k{r['k']}: {r.get('result')} {r.get('setup', '')}"
  s = (f"{r['case']:12s} {r['course']:9s} k{r['k']}: {r['result']:7s} fell={r['fell']!s:5s} "
       f"end {r['end'][0]:+.2f} {r['end'][1]:+.2f} {r['end'][2]:+6.1f} (asked {r['asked'][0]:+.2f} "
       f"{r['asked'][1]:+.2f} {r['asked'][2]:+6.1f}) tilt max {r['tilt_max']:.1f}")
  if "slip_max_cm" in r:
    s += (f" | slip {r['slip_max_cm']:.1f} cm {r['turn_max_deg']:.1f} deg, pads lost "
          f"{r['pads_lost_s']:.2f} s, dropped={r['dropped']}, held={r['held_end']}")
  return s + f" [{r.get('wall_s')} s]"


# ---- reading the rows ---------------------------------------------------------

def _wrap(deg: float) -> float:
  return math.degrees(math.remainder(math.radians(deg), 2 * math.pi))


def read(paths: list[str]) -> None:
  """The tables: for each course, each case's falls, its end against the
  asked end and against the arms-home walk, the trunk's tilt and the piece's
  slip in the jaws."""
  rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
  for r in rows:
    if r.get("unseen"):
      r["case"] += "*"
    if r.get("kg_override"):
      r["case"] += f"@{r['kg_override']:g}kg"
    if r.get("blind"):
      r["case"] += "~blind"
    if r.get("squeeze"):
      r["case"] += f"^{r['squeeze']:g}N"
    if r.get("house"):
      r["case"] += "#house"
  cases = list(dict.fromkeys(c for c0 in CASES for r in rows for c in (r["case"],)
                             if c.split("#")[0].split("^")[0].split("~")[0].split("*")[0].split("@")[0] == c0))
  courses = [c for c in (*COURSES, *GOALS) if any(r["course"] == c for r in rows)]

  def errs(r, course):
    a = asked_end(course)
    e = r["end"]
    return (e[0] - a[0], e[1] - a[1], _wrap(e[2] - a[2]))

  base = {}
  for course in courses:
    rs = [errs(r, course) for r in rows if r["case"] == "none" and r["course"] == course
          and "end" in r and not r["fell"]]
    if rs:
      base[course] = np.mean(rs, axis=0)
  print("CARRYING WHILE WALKING: each case a course from standing; per flight the pelvis's end "
        "off the asked end (along the start's heading m, left m, turned deg, wrapped), the "
        "mean; 'home' is how far it lands from where the arms-home walk lands, mean [worst]; "
        "drop/slip/turn: the piece in the jaws")
  for course in courses:
    ex, ey, eh = asked_end(course)
    print(f"\n{course}: asked {ex:+.2f} {ey:+.2f} m {eh:+.1f} deg"
          + (f"; arms home off it by {base[course][0]:+.2f} {base[course][1]:+.2f} m "
             f"{base[course][2]:+.1f} deg" if course in base else ""))
    print(f"  {'case':18s} {'n':>2s} {'fell':>4s} {'nosetup':>7s} {'off asked m, m, deg':>21s} "
          f"{'home m [worst]':>14s} {'home deg [worst]':>16s} {'tilt max [worst]':>16s} "
          f"{'drop':>5s} {'slip cm [worst]':>15s} {'turn deg [worst]':>16s}")
    for case in cases:
      rs = [r for r in rows if r["case"] == case and r["course"] == course]
      if not rs:
        continue
      ok = [r for r in rs if "end" in r]
      fell = sum(r["fell"] for r in ok)
      line = f"  {case:18s} {len(rs):2d} {fell:4d} {len(rs) - len(ok):7d}"
      if ok:
        e = np.array([errs(r, course) for r in ok])
        line += f" {e[:, 0].mean():+6.2f} {e[:, 1].mean():+6.2f} {e[:, 2].mean():+7.1f}"
        if course in base:
          off = np.hypot(e[:, 0] - base[course][0], e[:, 1] - base[course][1])
          offh = np.array([_wrap(v - base[course][2]) for v in e[:, 2]])
          line += (f"  {off.mean():5.2f} [{off.max():5.2f}]  {offh.mean():+6.1f} "
                   f"[{offh[np.argmax(np.abs(offh))]:+6.1f}]")
        else:
          line += " " * 33
        line += (f"  {np.mean([r['tilt_max'] for r in ok]):5.1f} [{max(r['tilt_max'] for r in ok):5.1f}]")
        if "slip_max_cm" in ok[0]:
          line += (f"  {sum(r['dropped'] for r in ok):2d}/{len(ok):<2d}"
                   f" {np.mean([r['slip_max_cm'] for r in ok]):5.1f} [{max(r['slip_max_cm'] for r in ok):5.1f}]"
                   f" {np.mean([r['turn_max_deg'] for r in ok]):6.1f} [{max(r['turn_max_deg'] for r in ok):5.1f}]")
      print(line)
  fails = [r for r in rows if "end" not in r]
  if fails:
    print("\nflights that never walked:")
    for r in fails:
      print(f"  {r['case']} {r['course']} k{r['k']}: {r.get('result')} "
            f"{ {k: v for k, v in (r.get('setup') or {}).items() if k in ('why', 'pads', 'held')} }")


# ---- the poses ---------------------------------------------------------------------

def poses_table() -> None:
  """--poses: each pose's joints for each piece, against the range walk2's
  scripted arms were sent within; where the hand is (stance frame at the
  keyframe); and how level the wrist holds the piece."""
  m, d, q0 = _bare()
  d.qpos[:] = q0
  mujoco.mj_kinematics(m, d)
  soles = [d.geom_xpos[m.geom(f"{s}_sole").id] for s in hm.SIDES]
  mid = (soles[0] + soles[1]) / 2
  for held in ("plate", "cup"):
    side = HELD[held][0][1]
    print(f"{held} ({side} arm):")
    for pose in POSES:
      q = pose_joints(pose, side, held)
      tilt, at, _ = _tilt_at(side, q, held)
      hand = at - [mid[0], mid[1], 0.0]
      outside = [n for n in hm.ARM if not trained(side, n)[0] - 1e-9 <= q[n] <= trained(side, n)[1] + 1e-9]
      print(f"  {pose:6s} " + " ".join(f"{q[n]:+5.2f}" for n in hm.ARM)
            + f"  hand {hand[0]:+.2f} {hand[1]:+.2f} {hand[2]:.2f} m, piece tilt {tilt:4.1f} deg"
            + (f"  OUTSIDE walk2's arms: {', '.join(outside)}" if outside else "  inside walk2's arms"))
  print("walk2's arms (left): " + ", ".join(
    f"{n} [{trained('left', n)[0]:+.2f}, {trained('left', n)[1]:+.2f}]" for n in hm.ARM))


# ---- watching one ------------------------------------------------------------------

def _parse(spec: str):
  parts = spec.split(":")
  return parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 0


def strip(spec: str, out: str, every_s: float = 1.5) -> None:
  from PIL import Image, ImageDraw
  case, course, k = _parse(spec)
  model, data, body, info = build(case, k, course)
  model.vis.headlight.ambient[:] = 0.35
  model.vis.headlight.diffuse[:] = 0.55
  reading = Reading(model, data, body, info)
  s = setup(body, model, data, info, reading)
  print(f"setup: {s}")
  rend = mujoco.Renderer(model, 360, 480)
  cam = mujoco.MjvCamera()
  shown = mujoco.MjvOption()
  shown.geomgroup[:] = 1    # an `--unseen` piece too: hidden from the robot, not from us
  root = model.body(hm.ROBOT_ROOT).id
  tiles = []

  # the plate is white on a pale floor: drawn in a colour of the strip's own
  for piece, _ in info["pieces"]:
    for g in reading.mine[piece]:
      model.geom_rgba[g] = [0.95, 0.55, 0.15, 1.0]

  def tile(label):
    # from the robot's front left, as it turns
    r = data.xmat[root].reshape(3, 3)
    cam.lookat[:] = data.xpos[root] - [0.0, 0.0, 0.1]
    cam.distance, cam.elevation = 2.4, -18.0
    cam.azimuth = math.degrees(math.atan2(r[1, 0], r[0, 0])) + 180.0 + 35.0
    rend.update_scene(data, cam, shown)
    img = Image.fromarray(rend.render())
    ImageDraw.Draw(img).text((8, 8), label[:70], fill=(255, 255, 255))
    tiles.append(img)

  said = before_walk(model, body, info, reading)
  tile(f"{case} {course} k{k}: posed {' '.join(said)}")
  falls0 = body.mission.falls
  routine = course_routine(body, course, reading, falls0)
  last = data.time
  try:
    command = next(routine)
    while True:
      body.stepper.step(command)
      if data.time - last >= every_s:
        last = data.time
        r = reading.rows[-1] if reading.rows else {}
        extra = "".join(f" {p} pads {r.get(p, {}).get('pads')}" for p, _ in info["pieces"])
        tile(f"t {data.time:5.1f} s tilt {r.get('tilt', 0):4.1f}{extra}")
      command = routine.send(None)
  except StopIteration as done:
    print(f"course: {done.value}")
  tile(f"end: falls {body.mission.falls - falls0}")
  rend.close()
  body.close()
  cols = 5
  sheet = Image.new("RGB", (480 * min(cols, len(tiles)), 360 * ((len(tiles) + cols - 1) // cols)),
                    (255, 255, 255))
  for i, t in enumerate(tiles):
    sheet.paste(t, (480 * (i % cols), 360 * (i // cols)))
  sheet.save(out)
  print(f"wrote {out} ({len(tiles)} tiles)")


def view(spec: str) -> None:
  from mujoco import viewer as mj_viewer
  case, course, k = _parse(spec)
  model, data, body, info = build(case, k, course)
  with mj_viewer.launch_passive(model, data) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = model.body(hm.ROBOT_ROOT).id
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.6, 150.0, -15.0
    reading = Reading(model, data, body, info)
    print(setup(body, model, data, info, reading), before_walk(model, body, info, reading))
    routine = course_routine(body, course, None, body.mission.falls)
    wall = time.time() - data.time
    try:
      command = next(routine)
      while v.is_running():
        body.stepper.step(command)
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          v.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)
        command = routine.send(None)
    except StopIteration as done:
      print(done.value)


def main(argv=None) -> None:
  global TRACE, UNSEEN, KG, BLIND, SQUEEZE_N, HOUSE, SERVED
  from pluggybot.batch import watched
  watched()
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--fly", action="store_true")
  ap.add_argument("--cases", default=",".join(CASES))
  ap.add_argument("--courses", default=",".join((*COURSES, *GOALS)))
  ap.add_argument("--draws", type=int, default=4)
  ap.add_argument("--first", type=int, default=0, help="the first draw's number")
  ap.add_argument("--jobs", type=int, default=1)
  ap.add_argument("--out", default="carry_rows.jsonl")
  ap.add_argument("--trace", default=None, help="a directory: each flight's samples, a file each")
  ap.add_argument("--kg", type=float, default=None, help="every held piece's mass, kg")
  ap.add_argument("--squeeze", type=float, default=None, help="the posed pieces' squeeze, N")
  ap.add_argument("--house", action="store_true",
                  help="the served house, the robot a step back from the kitchen's shelf")
  ap.add_argument("--blind-arms", action="store_true",
                  help="the walker shown its arms at its policy's default, still")
  ap.add_argument("--unseen", action="store_true",
                  help="the held pieces hidden from every sensor once posed (a self-filter)")
  ap.add_argument("--served", action="store_true",
                  help="the served walk's own hand-over (#682), which carries a held grip")
  ap.add_argument("--walker", default=None, help="another walking policy's .npz")
  ap.add_argument("--held-drop", type=float, default=None,
                  help="the self-filter's radius round a held grip, m")
  ap.add_argument("--read", default=None)
  ap.add_argument("--poses", action="store_true")
  ap.add_argument("--strip", default=None)
  ap.add_argument("--strip-out", default="carry_spike.png")
  ap.add_argument("--view", default=None)
  args = ap.parse_args(argv)
  if args.trace:
    TRACE = os.environ["CARRY_TRACE"] = str(Path(args.trace).expanduser().resolve())
  if args.unseen:
    os.environ["CARRY_UNSEEN"] = "1"
    UNSEEN = True
  if args.kg is not None:
    os.environ["CARRY_KG"] = str(args.kg)
    KG = args.kg
  if args.blind_arms:
    os.environ["CARRY_BLIND"] = "1"
    BLIND = True
  if args.house:
    os.environ["CARRY_HOUSE"] = "1"
    HOUSE = True
  if args.squeeze is not None:
    os.environ["CARRY_SQUEEZE"] = str(args.squeeze)
    SQUEEZE_N = args.squeeze
  if args.served:
    os.environ["CARRY_SERVED"] = "1"
    SERVED = True
  if args.held_drop is not None:
    os.environ["CARRY_HELD_DROP"] = str(args.held_drop)
    from pluggybot.humanoid import mission as _m2
    _m2.HELD_DROP_M = args.held_drop
  if args.walker:
    os.environ["CARRY_WALKER"] = str(Path(args.walker).expanduser().resolve())
    from pluggybot.humanoid import mission as _m
    _m.WALK_NPZ = Path(os.environ["CARRY_WALKER"])
  if args.fly:
    cases, courses = args.cases.split(","), args.courses.split(",")
    for c in cases:
      if c not in CASES:
        sys.exit(f"no case {c!r}: {CASES}")
    for c in courses:
      if c not in COURSES and c not in GOALS:
        sys.exit(f"no course {c!r}: {(*COURSES, *GOALS)}")
    fly_all(cases, courses, args.first, args.draws, args.jobs, args.out)
  elif args.read:
    read(args.read.split(","))
  elif args.poses:
    poses_table()
  elif args.strip:
    strip(args.strip, args.strip_out)
  elif args.view:
    view(args.view)
  else:
    ap.print_help()


if __name__ == "__main__":
  main()
