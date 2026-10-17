"""The humanoid spike (issue #507): our own body from real parts, sized
before anything is trained, and a scene for each part to watch.

The body is `pluggybot.humanoid.model.HumanoidSpec`, its parts
`pluggybot.humanoid.parts`; SimNotes, "The humanoid body", has every table
below and Parts.md, "The humanoid body", the parts and the bill.

  --view SCENE  a scene in MuJoCo's viewer, at real time, over and over
                until the window closes (MUJOCO_GL unset); the terminal says
                what is moving. Scenes: `body` -- the humanoid in the house's
                living room, held up on a gantry, moving each joint through
                its range in turn, then its grippers; `stand` -- on its own
                feet, its standing policy, shoved, crouching and bowing;
                `reach` -- a hand to a counter, a high shelf, a low shelf
                and the floor, each held; `grasp` -- a drawer's handle
                gripped (judged by the gripper's own reading) and the drawer
                drawn out with a compliant arm, what the hand felt said
                beside the truth
  --strip SCENE the scene's filmstrip, headless (`--out` names it)
  --policy NPZ  fly another standing policy (default models/humanoid_stand.npz)
  --mass        the mass budget by segment, beside the T2's published model
  --torque      the torque table: each joint's load against its actuator's
                peak and continuous ratings, standing, crouched, on one
                leg, reaching, and -- ESTIMATES until stages 4 and 5 have
                policies -- walking and on the house's stairs
  --thermal     the hottest winding, each activity sustained
  --energy      watts by activity, and what the pack buys
  --sensors     what the head's cameras and the LIDAR see from the stand
  --shove       the shoves it stands through each way, standing and crouched
  --envelope    where a hand reaches, floor to over the head, flown balancing
  --hold        a locked handle gripped and pulled harder until something gives
  (default)     the `body` scene's filmstrip, humanoid_spike.png

Usage:
  uv run python scripts/humanoid_spike.py --view body|stand|reach|grasp
  MUJOCO_GL=egl uv run python scripts/humanoid_spike.py [--torque|--shove|...]
"""

import os

# One BLAS thread before numpy loads (the quadruped's lesson: small matrix
# products ran slower on six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.humanoid import model as hm  # noqa: E402
from pluggybot.humanoid import parts as hp  # noqa: E402
from pluggybot.legs.actuator import driver_gains  # noqa: E402
from pluggybot.legs.drivers import Drivers  # noqa: E402
from pluggybot.telemetry.protocol import ROBOT_ROOT  # noqa: E402

HOME_XML = "models/home_world.xml"

# ---- the scene: `body` -------------------------------------------------------------

#: Where the gantry stands: the living room's west half, the body facing
#: east down the room, clear of the couch and with room for a camera.
GANTRY_XY, GANTRY_YAW = (0.5, 0.5), 0.0
#: The scenes' camera: south-east of the body, looking at its front-right.
CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE = 150.0, -8.0, 2.7
#: How far the gantry holds the feet off the floor, m.
HANG_CLEAR = 0.03
#: Each move: (what the terminal says, {joint kind: the LEFT side's target,
#: rad -- the right's mirrors where the joint does, `hm.MIRRORED`}, the
#: grippers' opening as a share of their stroke). Each lasts `MOVE_S`: a
#: cosine ease out to its targets over the first half, a hold, and the
#: next move eases on from there (position targets are ramped, never
#: stepped). Amplitudes are a demonstration's, inside every range.
MOVE_S = 2.0
MOVES = [
  ("hanging, at rest", {}, 0.0),
  ("hip pitch: thighs forward", {"hip_pitch": -1.3}, 0.0),
  ("hip pitch: thighs back", {"hip_pitch": 0.45}, 0.0),
  ("hip roll: legs out", {"hip_roll": 0.45}, 0.0),
  ("hip yaw: toes out", {"hip_yaw": 0.6, "knee": 0.9}, 0.0),
  ("hip yaw: toes in", {"hip_yaw": -0.5, "knee": 0.9}, 0.0),
  ("knee: bent", {"knee": 2.0, "hip_pitch": -0.8}, 0.0),
  ("ankle pitch: toes up", {"ankle_pitch": -0.7, "knee": 0.6}, 0.0),
  ("ankle pitch: toes down", {"ankle_pitch": 0.5, "knee": 0.6}, 0.0),
  ("ankle roll", {"ankle_roll": 0.3}, 0.0),
  ("ankle roll, the other way", {"ankle_roll": -0.3}, 0.0),
  ("shoulder pitch: arms forward and up", {"shoulder_pitch": -2.6}, 0.0),
  ("shoulder pitch: arms back", {"shoulder_pitch": 0.8}, 0.0),
  ("shoulder roll: arms out", {"shoulder_roll": 1.6}, 0.0),
  ("elbow: bent", {"elbow": 1.9, "shoulder_pitch": -0.6}, 0.0),
  ("shoulder yaw: forearms out", {"shoulder_yaw": 1.0, "elbow": 1.5,
                                  "shoulder_pitch": -0.6}, 0.0),
  ("shoulder yaw: forearms in", {"shoulder_yaw": -0.6, "elbow": 1.5,
                                 "shoulder_pitch": -0.6}, 0.0),
  ("wrist pitch", {"wrist_pitch": 1.2, "elbow": 1.5, "shoulder_pitch": -0.6}, 0.0),
  ("wrist pitch, the other way", {"wrist_pitch": -1.2, "elbow": 1.5,
                                  "shoulder_pitch": -0.6}, 0.0),
  ("wrist roll", {"wrist_roll": 1.5, "elbow": 1.5, "shoulder_pitch": -0.6}, 0.0),
  ("grippers open", {"elbow": 1.5, "shoulder_pitch": -0.6}, 1.0),
  ("grippers shut", {"elbow": 1.5, "shoulder_pitch": -0.6}, 0.0),
  # The grips 0.94 m up as it would stand (the gantry holds it 3 cm higher),
  # 0.48 m ahead, at the scene's gains (0.935 measured): a counter's height,
  # open to a handle.
  ("hands at a counter's height", {"shoulder_pitch": -0.76, "elbow": 0.4,
                                   "wrist_pitch": 0.8, "knee": 0.4,
                                   "hip_pitch": -0.2, "ankle_pitch": -0.2}, 0.6),
]


def targets(move: dict) -> np.ndarray:
  """Every joint's target for one move, in `hm.JOINT_NAMES` order: the
  left side's value, mirrored on the right where the joint's sense is."""
  out = []
  for j in hm.JOINT_NAMES:
    v = move.get(hm.kind(j), 0.0)
    if hm.side(j) < 0 and hm.kind(j) in hm.MIRRORED:
      v = -v
    lo, hi = hm.joint_range(j)
    out.append(min(max(v, lo), hi))
  return np.array(out)


def _ease(s: float) -> float:
  s = min(max(s, 0.0), 1.0)
  return 0.5 - 0.5 * math.cos(math.pi * s)


def _gantry(spec: mujoco.MjSpec, x: float, y: float, yaw: float, top: float) -> None:
  """The gantry the body hangs from (Parts.md's bill: 40 x 40 extrusion, a
  rope ratchet and slings): two posts either side and a crossbar, drawn
  only -- the hold itself is a weld, the strap's work."""
  c, s = math.cos(yaw), math.sin(yaw)
  body = spec.worldbody.add_body(name="gantry", pos=[x, y, 0.0])
  half = 0.75                                   # the posts either side, m
  for k, sign in enumerate((1, -1)):
    # Either side of the body: along its own y.
    px, py = -s * sign * half, c * sign * half
    body.add_geom(name=f"gantry_post_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.02, 0.02, top / 2], pos=[px, py, top / 2],
                  rgba=[0.75, 0.76, 0.78, 1], contype=0, conaffinity=0)
  body.add_geom(name="gantry_bar", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.02, 0, 0],
                fromto=[s * half, -c * half, top, -s * half, c * half, top],
                rgba=[0.75, 0.76, 0.78, 1], contype=0, conaffinity=0)


def held_world(at=GANTRY_XY, yaw: float = GANTRY_YAW) -> tuple:
  """The house with the humanoid held up on its gantry in the living room: the
  model, its data at the stand pose, and the drivers. The house is the
  generator's file as it is (no robot in it, and no change to the served
  world: `legs.world` builds that one)."""
  spec = mujoco.MjSpec.from_file(HOME_XML)
  robot = hm.attachable()
  h = hm.stand_height() + HANG_CLEAR
  frame = spec.worldbody.add_frame()
  frame.pos = [at[0], at[1], h - hm.stand_height()]
  frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  spec.attach(robot, prefix="", frame=frame)
  top = h + hm.CHOSEN.head_z + 0.35
  _gantry(spec, at[0], at[1], yaw, top)
  strap = spec.worldbody.add_geom(
    name="gantry_strap", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.008, 0, 0],
    fromto=[at[0] - 0.12 * math.cos(yaw), at[1] - 0.12 * math.sin(yaw), top,
            at[0] - 0.12 * math.cos(yaw), at[1] - 0.12 * math.sin(yaw),
            h + hm.CHOSEN.torso_z + hm.CHOSEN.torso_half[2]],
    rgba=[0.9, 0.5, 0.1, 1], contype=0, conaffinity=0)
  del strap
  # The hold: the pelvis welded to a hook where it hangs (the strap's
  # work), the hook's frame the pelvis's own. ⚠ The weld's data is written:
  # left at MjSpec's default its anchor sat a metre off the pelvis, and the
  # body sagged 42 mm swinging its legs (turned 90 deg, to the floor).
  spec.worldbody.add_body(name="gantry_hook", pos=[at[0], at[1], h],
                          quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=ROBOT_ROOT,
                    name2="gantry_hook", objtype=mujoco.mjtObj.mjOBJ_BODY,
                    data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  model = spec.compile()
  # The house's corner is dim for a camera of ours: a brighter headlight,
  # drawn only (nothing a sensor of the robot's renders reads it).
  model.vis.headlight.ambient[:] = 0.35
  model.vis.headlight.diffuse[:] = 0.55
  data = mujoco.MjData(model)
  _pose(model, data, hm.stand_qpos())
  drivers = Drivers(model, data, joints=hm.JOINT_NAMES)
  return model, data, drivers


def _pose(model, data, q: dict) -> None:
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = q.get(name, 0.0)
  mujoco.mj_forward(model, data)


#: The gains the scenes' commands carry: each joint's loop tuned on its own
#: rotor as the policies' are (`legs.actuator.driver_gains`, 10 Hz at
#: damping ratio 2) times the arms' scale (`hm.ARM_GAIN_SCALE`): a scene
#: has no policy offsetting its targets. Inside every part's MIT range (the
#: RS00's Kp to 500, the RS04's to 5000).
GAINS = np.array([driver_gains(hm.CHOSEN.motor(j)) for j in hm.JOINT_NAMES]) * hm.ARM_GAIN_SCALE


class BodyScene:
  """The `body` scene's moves, one physics step at a time."""

  def __init__(self, model, data, drivers: Drivers):
    self.m, self.d, self.drivers = model, data, drivers
    self.grippers = [model.actuator(f"{s}_gripper").id for s in hm.SIDES]
    self.k = 0
    self.t0 = data.time
    self.prev, self.prev_g = targets({}), 0.0

  @property
  def label(self) -> str:
    return MOVES[self.k][0]

  def step(self) -> bool:
    """One physics step of the current move; True when a new one began."""
    name, move, g = MOVES[self.k]
    s = _ease((self.d.time - self.t0) / (MOVE_S / 2))
    q = self.prev + s * (targets(move) - self.prev)
    self.drivers.pd(q, GAINS[:, 0], GAINS[:, 1])
    stroke = hp.GRIPPER.stroke / 2
    self.d.ctrl[self.grippers] = stroke * (self.prev_g + s * (g - self.prev_g))
    mujoco.mj_step(self.m, self.d)
    if self.d.time - self.t0 >= MOVE_S:
      self.prev, self.prev_g = targets(move), g
      self.k = (self.k + 1) % len(MOVES)
      self.t0 = self.d.time
      return True
    return False


def view_body() -> None:
  from mujoco import viewer as mj_viewer
  model, data, drivers = held_world()
  scene = BodyScene(model, data, drivers)
  print(f"the humanoid: {hm.mass():.1f} kg, {len(hm.JOINT_NAMES)} joints and two "
        f"grippers; each move {MOVE_S:.0f} s. Close the window to quit.")
  print(f"  -> {scene.label}")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(ROBOT_ROOT).id
    viewer.cam.distance, viewer.cam.azimuth = CAM_DISTANCE, CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION
    wall = time.time() - data.time
    while viewer.is_running():
      if scene.step():
        print(f"  -> {scene.label}")
      if int(round(data.time / model.opt.timestep)) % 10 == 0:
        viewer.sync()
        ahead = data.time - (time.time() - wall)
        if ahead > 0:
          time.sleep(ahead)


#: The filmstrip's moments: the moves whose end is shot.
STRIP = ("hanging, at rest", "hip pitch: thighs forward", "hip roll: legs out",
         "knee: bent", "ankle pitch: toes up", "shoulder pitch: arms forward and up",
         "shoulder roll: arms out", "shoulder yaw: forearms out", "wrist pitch",
         "grippers open", "hands at a counter's height")


def filmstrip(out: str) -> None:
  """The `body` scene, one tile at the end of each move in `STRIP`."""
  from PIL import Image, ImageDraw
  model, data, drivers = held_world()
  scene = BodyScene(model, data, drivers)
  r = mujoco.Renderer(model, 300, 400)
  root = model.body(ROBOT_ROOT).id
  tiles = []
  shot = set()
  while len(shot) < len(STRIP):
    label = scene.label
    if scene.step() and label in STRIP and label not in shot:
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[root] + [0.0, 0.0, 0.1]
      cam.distance, cam.azimuth, cam.elevation = CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
      tiles.append(img)
      shot.add(label)
  r.close()
  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  strip = Image.new("RGB", (400 * cols, 300 * rows), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % cols), 300 * (k // cols)))
  strip.save(out)
  print(f"wrote {out}")


# ---- the torque table's instrument: a pose held on the floor ------------------------

#: A pose is held by a PD stiff enough that it sags under `SAG_RAD` at its
#: actuator's peak: a measuring instrument's gains, carried in its commands
#: as any command's are, never the policy's (`GAINS`). Damped at
#: `SAG_DAMP_S` seconds of its stiffness.
SAG_RAD, SAG_DAMP_S = 0.05, 0.04
#: Seconds a pose is held, and how much of it is settling before the
#: measurement.
HOLD_S, HOLD_SETTLE_S = 3.0, 2.0
#: The house's stair (SimNotes, "The stairs curriculum"): a 0.18 m riser on
#: a 0.28 m tread.
#: ⚠ Its rows are ESTIMATES (below): held as a statue, all the weight on a
#: leading knee bent as far as the riser asks needs the ankle past the T2's
#: 48 deg of dorsiflexion upright, and past its 18 deg of roll leaned
#: forward 0.35 rad -- a climb shares that instant with the trailing leg,
#: which is a policy's to do (stage 5), not a statue's.
RISER_M, TREAD_M = 0.18, 0.28


def stiff_gains(spec: hm.HumanoidSpec = hm.CHOSEN) -> tuple[np.ndarray, np.ndarray]:
  kp = np.array([spec.motor(j).peak_torque for j in hm.JOINT_NAMES]) / SAG_RAD
  return kp, SAG_DAMP_S * kp


def leg_pose(q: dict, s: str, bend: float, lean: float = 0.0, side_lean: float = 0.0,
             lift: float = 0.0) -> None:
  """One leg's joints with its foot level under a level pelvis: knee `2 *
  bend`, the hip and ankle pitch half of it each, the whole leg leaned
  `lean` forward about the ankle and `side_lean` toward the robot's right
  (the hip and ankle rolls opposite), and the foot drawn up `lift` rad of
  extra knee with the hip taking it (a swing foot off the floor)."""
  q[f"{s}_hip_pitch"] = -bend - lean - lift
  q[f"{s}_knee"] = 2 * bend + 2 * lift
  q[f"{s}_ankle_pitch"] = -bend + lean - lift
  q[f"{s}_hip_roll"] = -side_lean
  q[f"{s}_ankle_roll"] = side_lean


def floor_world(scenery: str = "", payload: dict | None = None):
  """The bare robot on a floor (and any scenery), with `payload` kg held at
  each named hand's grip point: the model and its data."""
  spec = mujoco.MjSpec.from_string(hm.body_xml(scenery=scenery))
  for s, kg in (payload or {}).items():
    hand = spec.body(f"{s}_hand")
    grip = next(x for x in hand.sites if x.name == f"{s}_grip")
    hand.add_geom(name=f"{s}_payload", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[0.03, 0, 0], pos=list(grip.pos), mass=kg,
                  contype=0, conaffinity=0, rgba=[0.9, 0.7, 0.1, 1])
  model = spec.compile()
  return model, mujoco.MjData(model)


def place(model, data, q: dict, stance: tuple[str, ...]) -> None:
  """The robot in pose `q`, its pelvis level, set down so the `stance`
  soles rest on the floor and the first stance sole's middle is over the
  world's origin. Forwards."""
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = q.get(name, 0.0)
  mujoco.mj_kinematics(model, data)
  sole = model.geom(f"{stance[0]}_sole").id
  bottom = data.geom_xpos[sole][2] - model.geom_size[sole][2]
  data.qpos[0] -= data.geom_xpos[sole][0]
  data.qpos[1] -= data.geom_xpos[sole][1]
  data.qpos[2] -= bottom
  mujoco.mj_forward(model, data)


def balance(model, data, stance: str, bend: float, others: dict, iters: int = 8) -> dict:
  """Lean the stance leg until the whole robot's centre of mass is over its
  sole's middle (the pose a statue stands in), the other joints as
  `others`. Returns the pose."""
  lean = side = 0.0
  root = model.body(ROBOT_ROOT).id
  sole = model.geom(f"{stance}_sole").id
  height = hm.CHOSEN.leg_length - hm.CHOSEN.hip_roll_drop
  for _ in range(iters):
    q = dict(others)
    leg_pose(q, stance, bend, lean, side)
    place(model, data, q, (stance,))
    # A lean forward about the ankle carries the hips back (the foot is
    # planted), and a lean toward the right carries them left.
    err = data.subtree_com[root][:2] - data.geom_xpos[sole][:2]
    lean += err[0] / height
    side -= err[1] / height
  return q


def balance_both(model, data, q: dict, bend: float, iters: int = 8) -> dict:
  """Lean both legs about their ankles until the centre of mass is over the
  middle of the two soles: a load held out ahead is stood under, as a
  person leans back to carry one."""
  lean = 0.0
  root = model.body(ROBOT_ROOT).id
  soles = [model.geom(f"{s}_sole").id for s in hm.SIDES]
  height = hm.CHOSEN.leg_length - hm.CHOSEN.hip_roll_drop
  for _ in range(iters):
    out = dict(q)
    for s in hm.SIDES:
      leg_pose(out, s, bend, lean)
    place(model, data, out, hm.SIDES)
    mid = data.geom_xpos[soles].mean(axis=0)
    lean += (data.subtree_com[root][0] - mid[0]) / height
  return out


def hold(model, data, q: dict, stance: tuple[str, ...], seconds: float = HOLD_S) -> dict:
  """Place the robot in `q` and hold it with the instrument's gains:
  the mean and the largest |torque| each driver applied after the settle,
  whether it stood (the pelvis within 5 cm and 0.1 rad of the start), and
  the joints the pose asked past their range (`outside`), which a joint's
  stop holds instead of its driver."""
  outside = [j for j in hm.JOINT_NAMES
             if not hm.joint_range(j)[0] - 1e-9 <= q.get(j, 0.0) <= hm.joint_range(j)[1] + 1e-9]
  place(model, data, q, stance)
  drivers = Drivers(model, data, joints=hm.JOINT_NAMES)
  kp, kd = stiff_gains()
  target = np.array([q.get(j, 0.0) for j in hm.JOINT_NAMES])
  root = model.body(ROBOT_ROOT).id
  start = data.xpos[root].copy()
  taus = []
  stood = True
  for _ in range(int(seconds / model.opt.timestep)):
    drivers.pd(target, kp, kd)
    mujoco.mj_step(model, data)
    if data.time >= HOLD_SETTLE_S:
      taus.append(data.actuator_force[drivers.act].copy())
    tilt = math.acos(min(1.0, max(-1.0, data.xmat[root][8])))
    if np.linalg.norm(data.xpos[root] - start) > 0.05 or tilt > 0.1:
      stood = False
      break
  taus = np.abs(np.array(taus)) if taus else np.full((1, len(hm.JOINT_NAMES)), np.nan)
  return {"mean": taus.mean(axis=0), "max": taus.max(axis=0), "stood": stood,
          "outside": outside}


def static_poses() -> list[tuple[str, callable]]:
  """The torque table's static rows: each a name and a function that holds
  its pose and returns `hold`'s record. In our physics: a statue on its
  soles, every joint held where the pose puts it."""
  def stand():
    m, d = floor_world()
    return hold(m, d, hm.stand_qpos(), ("left", "right"))

  def crouch():
    m, d = floor_world()
    return hold(m, d, hm.stand_qpos(bend=0.75), ("left", "right"))

  def one_leg():
    m, d = floor_world()
    swing = {}
    leg_pose(swing, "right", 0.2, lift=0.35)
    q = balance(m, d, "left", 0.2, swing)
    return hold(m, d, q, ("left",))

  def reach(kg_each: float, sides: tuple[str, ...]):
    def run():
      m, d = floor_world(payload={s: kg_each for s in sides})
      q = hm.stand_qpos()
      for s in sides:
        q[f"{s}_shoulder_pitch"] = -math.pi / 2
        q[f"{s}_elbow"] = 0.0
        q[f"{s}_shoulder_roll"] = 0.0
      return hold(m, d, balance_both(m, d, q, 0.2), hm.SIDES)
    return run

  return [
    ("stand", stand),
    ("deep crouch", crouch),
    ("on one leg", one_leg),
    ("an arm out level, 1 kg in the hand", reach(1.0, ("right",))),
    ("both arms out level, 2 kg in each", reach(2.0, ("left", "right"))),
  ]


# ---- the tables ---------------------------------------------------------------------

#: The kinds of joint, in the order the tables print them.
KINDS = hm.LEG + hm.ARM


def by_kind(per_joint: np.ndarray) -> dict:
  """Each joint kind's worse side."""
  out = {}
  for k, j in enumerate(hm.JOINT_NAMES):
    out[hm.kind(j)] = max(out.get(hm.kind(j), 0.0), float(per_joint[k]))
  return out


def static_rows() -> list[tuple[str, dict]]:
  """Each static pose held in our physics: its name and the record, every
  joint's mean |torque| over the hold (a pose held is a load sustained)."""
  rows = []
  for name, fn in static_poses():
    rec = fn()
    rows.append((name, rec))
  return rows


def torque_table() -> None:
  spec = hm.CHOSEN
  print(f"{spec.name}: {hm.mass():.1f} kg; each joint's part, N*m peak / "
        f"continuous turning / continuous held still:")
  for k in KINDS:
    m = spec.drives[k]
    print(f"  {k:15s} {m.name:16s} {m.peak_torque:6g} / {m.rated_torque:4g} / "
          f"{hp.stall_torque(m):5.2f}")
  rows = static_rows()
  print()
  print("HELD IN OUR PHYSICS (a statue on its soles; the mean |torque| over the "
        "hold, the worse side, and its % of the part's continuous rating HELD "
        "STILL -- a joint holding a pose is a stalled motor, one phase "
        "carrying the current):")
  head = f"{'joint':15s}" + "".join(f"{n[:16]:>17s}" for n, _ in rows)
  print(head)
  for kind in KINDS:
    m = spec.drives[kind]
    cells = []
    for _, rec in rows:
      if not rec["stood"]:
        cells.append(f"{'FELL':>17s}")
        continue
      v = by_kind(rec["mean"])[kind]
      cells.append(f"{v:8.1f} ({100 * v / hp.stall_torque(m):4.0f} %)")
    print(f"{kind:15s}" + "".join(cells))
  outside = sorted({j for _, rec in rows for j in rec["outside"]})
  if outside:
    print(f"  ⚠ asked past their range: {', '.join(outside)}")


# ---- the estimates: walking and the stairs, until stages 4 and 5 fly them -----------

#: ⚠ ESTIMATES, NOT OUR PHYSICS: published peak joint loads, N*m per kg of
#: the whole body, scaled to this body by its mass alone (the gait lab's
#: and prosthesis design's rule; children of 1.47 m sit inside the adult
#: per-kg range at one speed, which supports it). The upper end of each
#: published range, and a robot's own where one was measured (an RL gait
#: bends its knees more than a person's). Stages 4 and 5 read the table
#: again off their policies (#507, rule 7). `ESTIMATE_SOURCES` names each.
ESTIMATES = {
  "walking ~1.25 m/s": {
    "hip_pitch": (1.3, "human"), "hip_roll": (1.05, "g1"), "hip_yaw": (0.75, "g1"),
    "knee": (1.15, "g1"), "ankle_pitch": (1.65, "g1"), "ankle_roll": (0.28, "g1")},
  "stairs up, 0.16-0.22 m": {
    "hip_pitch": (0.91, "human-stairs"), "hip_roll": (2.15, "adam"),
    "hip_yaw": (1.2, "adam"), "knee": (2.16, "adam"), "ankle_pitch": (1.7, "human-stairs")},
  "stairs down": {
    "hip_pitch": (0.6, "human-stairs"), "knee": (1.6, "human-stairs"),
    "ankle_pitch": (1.5, "human-stairs")},
}
#: ...and the one published RMS, for the windings: Adam Lite's legs up 15 cm
#: stairs on hardware, N*m per kg (its figure's mean and spread, the worse
#: side; 60 kg).
ESTIMATE_RMS = {"hip_pitch": 13.0 / 60, "hip_roll": 45.4 / 60, "hip_yaw": 14.6 / 60,
                "knee": 35.7 / 60, "ankle_pitch": 14.0 / 60}
ESTIMATE_SOURCES = {
  "human": "human gait, 1.22-1.33 m/s (Alkjaer 2006, Leijendekkers 2018, Nikolajsen 2013)",
  "human-stairs": "human stairs, 16-22 cm risers (Boehme 2022, Afschrift 2014, Riener 2002)",
  "g1": "Unitree G1 on hardware, 0.5-1.85 m/s, read off Feng 2026 "
        "(https://arxiv.org/html/2602.21666)",
  "adam": "PNDbotics Adam Lite up 15 cm stairs on hardware, 60 kg "
          "(https://arxiv.org/html/2509.12858, Fig. 9)",
}


def estimate_table(mass: float | None = None) -> None:
  spec = hm.CHOSEN
  mass = mass or hm.mass()
  print(f"ESTIMATES -- published peak loads scaled to {mass:.1f} kg by mass alone, "
        "N*m, and % of the part's PEAK (rule 7: stages 4 and 5 read these again off "
        "their policies):")
  acts = list(ESTIMATES)
  print(f"{'joint':15s}" + "".join(f"{a[:22]:>24s}" for a in acts))
  for kind in hm.LEG:
    m = spec.drives[kind]
    cells = []
    for a in acts:
      if kind not in ESTIMATES[a]:
        cells.append(f"{'no data':>24s}")
        continue
      per_kg, src = ESTIMATES[a][kind]
      tau = per_kg * mass
      cells.append(f"{tau:7.1f} ({100 * tau / m.peak_torque:3.0f} %) {src:>8s}")
    print(f"{kind:15s}" + "".join(cells))
  print("windings, Adam Lite's stairs RMS scaled, against the part's continuous "
        "rating turning, steady at 25 C:")
  for kind, per_kg in ESTIMATE_RMS.items():
    m = spec.drives[kind]
    rms = per_kg * mass
    t = hp.RS_RATED_AMBIENT_C + (m.max_winding_c - hp.RS_RATED_AMBIENT_C) * (rms / m.rated_torque) ** 2
    print(f"  {kind:15s} {rms:5.1f} N*m RMS ({100 * rms / m.rated_torque:3.0f} % of "
          f"{m.rated_torque:g}): {t:4.0f} C of {m.max_winding_c:.0f}")
  for tag, text in ESTIMATE_SOURCES.items():
    print(f"  [{tag}] {text}")


#: Ambient air for the thermal table, C: a warm room and a hot day.
AMBIENT_C = (30.0, 40.0)


def copper_w(motor, tau: float) -> float:
  """Heat in a winding at output torque `tau`, W: 1.5 R I^2 with I = tau /
  Kt the phase current's amplitude (`legs.actuator.JointLimits.copper_w`)."""
  i = tau / motor.kt
  return 1.5 * motor.r_phase * i * i


def thermal_table(rows=None) -> None:
  """Each held pose SUSTAINED: every winding's steady temperature, and,
  past its rating, how long its part holds it. A part's continuous rating
  is by its maker's definition the torque that holds its winding at its
  limit at 25 C on the named heat sink, and a winding heats as torque
  squared, so a torque `tau` held still settles at the ambient plus
  (limit - 25) * (tau / the stalled rating)^2 (DERIVED: no sheet gives a
  thermal resistance). Past the rating the sheet's overload table says
  how long (`parts.held_for`)."""
  rows = rows or static_rows()
  print(f"{'activity':38s} {'hottest winding':>16s} {'N*m':>6s} {'warm room C':>11s} "
        f"{'hot day C':>9s} {'limit C':>7s} {'held for':>13s}")
  for name, rec in rows:
    if not rec["stood"]:
      print(f"{name:38s} {'FELL':>16s}")
      continue
    worst = None
    for k, j in enumerate(hm.JOINT_NAMES):
      m = hm.CHOSEN.motor(j)
      tau = float(rec["mean"][k])
      rise = (m.max_winding_c - hp.RS_RATED_AMBIENT_C) * (tau / hp.stall_torque(m)) ** 2
      margin = m.max_winding_c - (AMBIENT_C[1] + rise)
      if worst is None or margin < worst[0]:
        worst = (margin, j, tau, rise, m)
    _, j, tau, rise, m = worst
    held = hp.held_for(m, tau, still=True)
    if held is None:
      lasts = "as long as wanted"
    else:
      lasts = f"{'over ' if hp.lasts_at_least(m, tau, still=True) else ''}{held:.0f} s"
    print(f"{name:38s} {j:>16s} {tau:6.1f} {AMBIENT_C[0] + rise:11.0f} "
          f"{AMBIENT_C[1] + rise:9.0f} {m.max_winding_c:7.0f} {lasts:>13s}")


def energy_table(rows=None) -> None:
  """Watts by held pose: the windings' heat (holding still does no work),
  and the electronics always on; what the pack buys of each."""
  rows = rows or static_rows()
  base = sum(hp.ELECTRONICS_W.values())
  print(f"electronics, always on: {base:.1f} W "
        + ", ".join(f"{k} {v:g}" for k, v in hp.ELECTRONICS_W.items()))
  print(f"the pack: {hp.PACK.wh:.0f} Wh ({hp.PACK.name})")
  print(f"{'activity':38s} {'windings W':>10s} {'total W':>8s} {'hours':>6s}")
  for name, rec in rows:
    if not rec["stood"]:
      print(f"{name:38s} {'FELL':>10s}")
      continue
    w = sum(copper_w(hm.CHOSEN.motor(j), float(rec["mean"][k]))
            for k, j in enumerate(hm.JOINT_NAMES))
    print(f"{name:38s} {w:10.1f} {base + w:8.1f} {hp.PACK.wh / (base + w):6.1f}")


#: ⚠ ESTIMATES, as `ESTIMATES` are: published walking draw per kg of body,
#: W/kg -- the T2's 480 Wh for "2 hours of continuous walking" on 42 kg
#: (booster.tech), the G1's 9 Ah 13S pack for "about 2 h" on 35 kg
#: (unitree.com/g1), each a whole robot's average, its electronics in it.
WALKING_W_PER_KG = {"Booster T2": 480 / 2 / 42.0, "Unitree G1": 9.0 * 13 * 3.6 / 2 / 35.0}


def walking_energy() -> None:
  mass = hm.mass()
  print(f"WALKING, ESTIMATED (published averages per kg, scaled to {mass:.1f} kg):")
  for robot, w_per_kg in WALKING_W_PER_KG.items():
    w = w_per_kg * mass
    print(f"  as the {robot}: {w_per_kg:.1f} W/kg, {w:.0f} W, "
          f"{hp.PACK.wh / w:.1f} h on the {hp.PACK.wh:.0f} Wh pack")


# ---- the mass budget, beside the reference ------------------------------------------

def segment_masses(spec: mujoco.MjSpec) -> dict:
  """Our body by the reference's segments: a leg from its hip pitch link
  down (the hip pitch motor rides the pelvis, as the T2's does its waist),
  an arm from its shoulder pitch link with its gripper, the head (its
  shell and the sensors in it), and the rest."""
  model = spec.compile()

  def subtree(name):
    return float(model.body_subtreemass[model.body(name).id])
  geoms = {g.name: g.mass for g in spec.geoms}
  head = sum(geoms[g] for g in ("head", "depth_cam", "nav_cam", "lidar"))
  total = subtree(ROBOT_ROOT)
  legs = [subtree(f"{s}_hip_pitch_link") for s in hm.SIDES]
  arms = [subtree(f"{s}_shoulder_pitch_link") for s in hm.SIDES]
  return {"leg": legs[0], "arm": arms[0], "head": head,
          "rest": total - sum(legs) - sum(arms) - head, "total": total}


def mass_table() -> None:
  ours = segment_masses(mujoco.MjSpec.from_string(hm.body_xml()))
  t2 = hm.T2
  ref = {"leg": t2.leg_kg, "arm": t2.arm_kg, "head": t2.head_kg,
         "rest": t2.rest_kg, "total": t2.total_kg}
  print(f"{'segment':10s} {'ours kg':>8s} {'T2 kg':>7s} {'ours/T2':>8s}")
  for k in ("leg", "arm", "head", "rest", "total"):
    print(f"{k:10s} {ours[k]:8.2f} {ref[k]:7.2f} {ours[k] / ref[k]:8.2f}")
  print("(a leg and an arm each side; the rest is the trunk, the pack, the "
        "electronics and the hips' and shoulders' pitch motors)")
  spec = hm.CHOSEN
  ours = {
    "thigh (hip pitch axis -> knee)": spec.hip_roll_drop + spec.hip_yaw_drop + spec.thigh,
    "shank (knee -> ankle)": spec.shank,
    "ankle -> sole": spec.ankle_roll_drop + spec.sole_drop,
    "the legs' lines, either side": spec.hip_y,
    "hip axes -> shoulder pitch axis": spec.shoulder_z,
    "the shoulder's roll axis, out": spec.shoulder_pitch_y + spec.shoulder_roll_out,
    "upper arm (roll axis -> elbow)": spec.shoulder_yaw_drop + spec.upper_arm,
    "forearm (elbow -> wrist)": spec.forearm,
    "sole, long": spec.foot_len,
    "sole, wide": spec.foot_w,
  }
  ref = dict(zip(ours, (t2.thigh, t2.shank, t2.ankle_to_sole, t2.hip_y, t2.shoulder_z,
                        t2.shoulder_y, t2.upper_arm, t2.forearm, *t2.foot)))
  print(f"\n{'length, m':34s} {'ours':>6s} {'T2':>6s}")
  for k, v in ours.items():
    print(f"{k:34s} {v:6.3f} {ref[k]:6.3f}")


# ---- the sensors from the stand ---------------------------------------------------

def _ray_floor(origin, direction, height: float) -> float | None:
  """Where a ray from `origin` meets the plane z = `height`: the horizontal
  distance ahead, or None if it never does."""
  if direction[2] >= 0:
    return None
  t = (height - origin[2]) / direction[2]
  return float(np.hypot(*(t * direction[:2])))


def sensors_table() -> None:
  """What the head's cameras and the LIDAR cover from the stand: each
  camera's view between its top and bottom rows, where it meets the floor
  and the counter's height, and the LIDAR's plane."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  counter = 0.94
  for name in ("depth_eye", "color_eye", "nav_eye"):
    cid = model.camera(name).id
    pos = data.cam_xpos[cid]
    mat = data.cam_xmat[cid].reshape(3, 3)
    half = math.radians(model.cam_fovy[cid]) / 2
    rows = {}
    for label, ang in (("bottom", -half), ("top", half)):
      # The camera looks along its -z, its image's up is +y.
      d = mat @ np.array([0.0, math.sin(ang), -math.cos(ang)])
      rows[label] = d
    near = _ray_floor(pos, rows["bottom"], 0.0)
    far = _ray_floor(pos, rows["top"], 0.0)
    c_near = _ray_floor(pos, rows["bottom"], counter)
    c_far = _ray_floor(pos, rows["top"], counter)
    pitch = math.degrees(math.atan2(-(mat @ [0, 0, -1])[2],
                                    np.hypot(*(mat @ [0, 0, -1])[:2])))

    def fmt(v):
      return "beyond" if v is None else f"{v:.2f} m"
    print(f"{name:10s} at {pos[2]:.2f} m, pitched {pitch:.0f} deg down, "
          f"{model.cam_fovy[cid]:.0f} deg tall: the floor from {fmt(near)} to "
          f"{fmt(far)} ahead, the counter's height ({counter} m) from "
          f"{fmt(c_near)} to {fmt(c_far)}")
  lidar = data.site_xpos[model.site("lidar").id]
  print(f"LIDAR plane at {lidar[2]:.2f} m: walls and anything taller; tables, "
        "the couch and the bed are under it (the depth camera's)")


# ---- stage 2: standing, reaching, grasping (the standing policy flying) -------------
#
# The humanoid on its own feet in the living room, its legs the standing
# policy's (`humanoid.policy`) and its arms the reach's (`humanoid.reach`),
# reading only its own senses. A scene STAGES with the truth and says so:
# where a target or a handle is put (told to the robot in its stance frame,
# as if it had seen it there), and a shove's push. What a scene reports of
# how well it went -- where the grip truly is, the force truly on a handle,
# whether it fell -- is OUR grading off the simulator, never the robot's.

from pluggybot.humanoid.arms import DOWN, FORWARD, HandGoal, frame  # noqa: E402
from pluggybot.humanoid.policy import StandDriver, StandPolicy  # noqa: E402
from pluggybot.humanoid.reach import Reach, plan  # noqa: E402

#: Where the robot stands: the body scene's spot, facing east down the room.
STAND_XY, STAND_YAW = GANTRY_XY, GANTRY_YAW
#: A fall, by our grading: the pelvis this low, m (a crouch puts it at 0.48).
FALLEN_M = 0.40
#: A shove: a force on the trunk for this long, s -- a hand's push, its
#: impulse the force times this.
SHOVE_S = 0.1


def _yaw_quat(yaw: float) -> list:
  return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def stance_offset() -> np.ndarray:
  """Where the stance frame's origin is from the pelvis, at the stand, m,
  off the robot's own drawing: what places scenery in front of it."""
  from pluggybot.humanoid.arms import SelfModel
  sm = SelfModel()
  sm.pose(np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES]),
          np.array([0.0, 0.0, -1.0]))
  return -sm.d.qpos[0:3].copy()


def to_world(p_stance, at=STAND_XY, yaw: float = STAND_YAW) -> np.ndarray:
  """A point in the stance frame of a robot standing at `at`, in the world:
  for building scenery round it before it stands there."""
  off = stance_offset()
  c, s = math.cos(yaw), math.sin(yaw)
  x, y, z = p_stance
  ox, oy = off[0] + x, off[1] + y
  return np.array([at[0] + c * ox - s * oy, at[1] + s * ox + c * oy, z])


def standing_world(scenery=None, at=STAND_XY, yaw: float = STAND_YAW):
  """The house with the humanoid standing on its soles in the living room,
  and whatever `scenery(spec)` adds: the model and its data at the stand."""
  spec = mujoco.MjSpec.from_file(HOME_XML)
  frame = spec.worldbody.add_frame()
  frame.pos = [at[0], at[1], 0.0]
  frame.quat = _yaw_quat(yaw)
  spec.attach(hm.attachable(), prefix="", frame=frame)
  if scenery is not None:
    scenery(spec)
  model = spec.compile()
  model.vis.headlight.ambient[:] = 0.35
  model.vis.headlight.diffuse[:] = 0.55
  data = mujoco.MjData(model)
  _pose(model, data, hm.stand_qpos())
  return model, data


def true_stance(model, data) -> tuple[np.ndarray, np.ndarray]:
  """The stance frame as the simulator has it (our staging and grading):
  its rotation (world from stance) and its origin."""
  soles = [model.geom(f"{s}_sole").id for s in hm.SIDES]
  heads = [math.atan2(data.geom_xmat[g][3], data.geom_xmat[g][0]) for g in soles]
  yaw = math.atan2(sum(math.sin(h) for h in heads), sum(math.cos(h) for h in heads))
  bottoms = [data.geom_xpos[g] - data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g][2]
             for g in soles]
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.mean(bottoms, axis=0)


class Humanoid:
  """The humanoid in a world: its standing policy and its reach, stepped
  together, one physics step at a time."""

  def __init__(self, model, data, policy: StandPolicy | None = None):
    self.m, self.d = model, data
    self.stand = StandDriver(model, data, policy)
    self.reach = Reach(model, data, self.stand)
    self.root = model.body(ROBOT_ROOT).id

  def step(self) -> None:
    self.stand.step()
    self.reach.step()
    mujoco.mj_step(self.m, self.d)

  def fallen(self) -> bool:
    return bool(self.d.xpos[self.root][2] < FALLEN_M)

  def heading(self) -> float:
    rot = self.d.xmat[self.root].reshape(3, 3)
    return math.atan2(rot[1, 0], rot[0, 0])


def wait(h: Humanoid, seconds: float):
  """`seconds` of the robot standing as it is."""
  for _ in range(int(round(seconds / h.m.opt.timestep))):
    h.step()
    yield None


def shove(h: Humanoid, toward, impulse: float):
  """Push the trunk `impulse` N*s toward `toward` (in the robot's frame,
  x ahead): a force at its centre of mass for `SHOVE_S` -- staged with the
  truth, as a scene's push is."""
  c, s = math.cos(h.heading()), math.sin(h.heading())
  u = np.array([toward[0], toward[1]], dtype=float)
  u /= np.linalg.norm(u)
  f = impulse / SHOVE_S
  for _ in range(int(round(SHOVE_S / h.m.opt.timestep))):
    h.d.xfrc_applied[h.root, :3] = [f * (c * u[0] - s * u[1]), f * (s * u[0] + c * u[1]), 0.0]
    h.step()
    yield None
  h.d.xfrc_applied[h.root, :3] = 0.0


def posture(h: Humanoid, height: float, pitch: float):
  """The standing policy asked a posture (through the reach, which holds
  it by its own loop), and stood until it has it."""
  h.reach.hold_posture(height, pitch)
  while not h.stand.settled:
    h.step()
    yield None


def reach_and_hold(h: Humanoid, side: str, goal: HandGoal, hold_s: float = 2.0,
                   record: dict | None = None, budget_s: float = 20.0,
                   say: str | None = None):
  """A hand to `goal` (stance frame), held `hold_s`, then home. Into
  `record`: whether it planned, how long it took, where the grip truly was
  through the hold against where the goal truly is (our grading), and
  whether the robot fell."""
  rec = record if record is not None else {}
  rot_w, origin = true_stance(h.m, h.d)
  want = origin + rot_w @ goal.pos
  t0 = h.d.time
  rec["planned"] = h.reach.reach(side, goal)
  if not rec["planned"]:
    return
  while not h.reach.done(side) and h.d.time - t0 < budget_s and not h.fallen():
    h.step()
    yield None
  rec["seconds"] = h.d.time - t0
  rec["blocked"] = h.reach.phase[side] == "blocked"
  rec["arrived"] = h.reach.done(side) and not rec["blocked"]
  site = h.m.site(f"{side}_grip").id
  misses = []
  if say:
    yield say
  for _ in wait(h, hold_s):
    misses.append(float(np.linalg.norm(h.d.site_xpos[site] - want)))
    yield None
  rec["miss_mean"], rec["miss_max"] = float(np.mean(misses)), float(np.max(misses))
  rec["posture"] = tuple(float(v) for v in h.reach.posture)
  rec["slack"] = h.reach.slack[side]
  h.reach.retreat(side)
  t0 = h.d.time
  while h.reach.phase[side] != "rest" and h.d.time - t0 < budget_s and not h.fallen():
    h.step()
    yield None
  yield from posture(h, 0.0, 0.0)
  rec["fell"] = h.fallen()


# ---- the scene: `stand` ------------------------------------------------------------

#: The `stand` scene's script: (what the terminal says, what happens). A
#: shove's direction is where it pushes, in the robot's frame (x ahead);
#: its size is inside every direction's measured margin (`--shove`: from
#: the front it stands through 10 N*s and falls at 15).
STAND_SHOVE_NS = 10.0
STAND_SCRIPT = (
  ("standing", ("wait", 2.0)),
  ("shoved from the front", ("shove", (-1.0, 0.0), STAND_SHOVE_NS)),
  ("recovering", ("wait", 2.5)),
  ("shoved from its left", ("shove", (0.0, -1.0), STAND_SHOVE_NS)),
  ("recovering", ("wait", 2.5)),
  ("shoved from behind", ("shove", (1.0, 0.0), STAND_SHOVE_NS)),
  ("recovering", ("wait", 2.5)),
  ("shoved from its right", ("shove", (0.0, 1.0), STAND_SHOVE_NS)),
  ("recovering", ("wait", 2.5)),
  ("crouching 0.2 m", ("posture", -0.2, 0.0)),
  ("crouched", ("wait", 1.5)),
  ("shoved from behind, crouched", ("shove", (1.0, 0.0), STAND_SHOVE_NS)),
  ("recovering", ("wait", 2.5)),
  ("bowing 0.8 rad", ("posture", -0.2, 0.8)),
  ("bowed", ("wait", 1.5)),
  ("standing again", ("posture", 0.0, 0.0)),
  ("standing", ("wait", 1.5)),
)


def stand_scene(h: Humanoid):
  """The `stand` scene, once: yields a line to print when a new part
  begins, else None, one physics step at a time."""
  for label, (what, *args) in STAND_SCRIPT:
    yield label
    if what == "wait":
      yield from wait(h, *args)
    elif what == "shove":
      yield from shove(h, *args)
    else:
      yield from posture(h, *args)
    if h.fallen():
      yield "FELL"
      return


# ---- the scene: `reach` ------------------------------------------------------------

#: The heights household objects cluster at (#507: 0.09, 0.49, 0.94 and
#: 1.43 m), each a target in the stance frame for the left hand: a counter
#: and a low shelf with the fingers forward; a high shelf with them tilted
#: 45 deg up (level, the wrist's pitch runs out of range over the
#: shoulder); the floor fingers down -- reachable only near the toes, at
#: full stretch from the deepest bow the policy holds (`reach.plan`;
#: SimNotes has the envelope).
UP45 = frame((math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4)),
             (-math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))
REACH_TARGETS = (
  ("a counter's height, 0.94 m", HandGoal(np.array([0.45, 0.2, 0.94]), FORWARD)),
  ("a high shelf, 1.43 m", HandGoal(np.array([0.35, 0.2, 1.43]), UP45)),
  ("a low shelf, 0.49 m", HandGoal(np.array([0.45, 0.2, 0.49]), FORWARD)),
  ("the floor, 0.09 m", HandGoal(np.array([0.17, 0.2, 0.09]), DOWN)),
)


def _marker(spec: mujoco.MjSpec, name: str, pos, rgba=(0.95, 0.3, 0.2, 0.9)) -> None:
  b = spec.worldbody.add_body(name=name, pos=list(pos))
  b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0],
             rgba=list(rgba), contype=0, conaffinity=0)


def reach_world():
  """The standing world with a marker at each of the reach scene's
  targets: drawn only (nothing collides with it)."""
  def scenery(spec):
    for k, (_, goal) in enumerate(REACH_TARGETS):
      _marker(spec, f"target_{k}", to_world(goal.pos))
  return standing_world(scenery)


def reach_scene(h: Humanoid, records: list | None = None):
  for label, goal in REACH_TARGETS:
    yield f"reaching {label}"
    rec = {"target": label}
    yield from reach_and_hold(h, "left", HandGoal(goal.pos.copy(), goal.rot), record=rec,
                              say=f"holding at {label}")
    if records is not None:
      records.append(rec)
    if not rec.get("planned"):
      yield "  no posture puts it in reach"
    elif rec.get("blocked"):
      yield (f"  stopped short: its own solve put the goal out of reach, or its hand "
             f"pushed on something; held {100 * rec['miss_mean']:.1f} cm off it")
    else:
      yield (f"  posture {rec['posture'][0]:+.2f} m, {rec['posture'][1]:.2f} rad"
             f"{'' if rec['slack'] else ', the arm at full stretch'}; there in "
             f"{rec['seconds']:.1f} s; the grip {100 * rec['miss_mean']:.1f} cm off it "
             f"held (worst {100 * rec['miss_max']:.1f})")
    if h.fallen():
      yield "FELL"
      return


# ---- the scene: `grasp` ------------------------------------------------------------

#: The counter: its top at a counter's 0.94 m, its front this far ahead of
#: the stance frame's origin, m; a drawer under its top whose bar handle
#: stands `HANDLE_OFF` proud of its front, at `HANDLE_Z`, in front of the
#: left shoulder. The drawer slides out against a spring and its runners'
#: friction, the feel of a drawer that closes itself.
COUNTER_TOP, COUNTER_FRONT, COUNTER_DEPTH = 0.94, 0.51, 0.6
HANDLE_OFF, HANDLE_Z, HANDLE_Y = 0.06, 0.84, 0.2
HANDLE_R, HANDLE_HALF = 0.012, 0.07
DRAWER_KG, DRAWER_SPRING, DRAWER_FRICTION, DRAWER_DAMPING = 3.0, 30.0, 4.0, 5.0
DRAWER_TRAVEL = 0.3
#: How far the grasp scene draws the drawer out, m, and the stiffness its
#: hand pulls with (a share of the reach's): compliant, so the hand gives.
PULL_M, PULL_STIFFNESS = 0.15, 0.5


def counter_scenery(spec: mujoco.MjSpec, locked: bool = False, at=STAND_XY,
                    yaw: float = STAND_YAW) -> None:
  """The counter, its drawer and the drawer's handle, in front of the
  robot standing at `at`. `locked`: the drawer welded shut, a handle to
  pull on."""
  front = to_world((COUNTER_FRONT, HANDLE_Y, 0.0), at, yaw)
  wood = [0.62, 0.48, 0.34, 1.0]
  c, sn = math.cos(yaw), math.sin(yaw)
  counter = spec.worldbody.add_body(
    name="counter", pos=[front[0] + c * COUNTER_DEPTH / 2, front[1] + sn * COUNTER_DEPTH / 2, 0.0],
    quat=_yaw_quat(yaw))
  counter.add_geom(name="counter_base", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[COUNTER_DEPTH / 2, 0.5, (COUNTER_TOP - 0.16) / 2],
                   pos=[0, 0, (COUNTER_TOP - 0.16) / 2], rgba=wood)
  counter.add_geom(name="counter_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[COUNTER_DEPTH / 2 + 0.02, 0.52, 0.015],
                   pos=[-0.02, 0, COUNTER_TOP - 0.015], rgba=[0.85, 0.85, 0.82, 1])
  drawer = counter.add_body(name="drawer", pos=[-COUNTER_DEPTH / 2, 0.0, HANDLE_Z])
  if not locked:
    drawer.add_joint(name="drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, DRAWER_TRAVEL], stiffness=DRAWER_SPRING,
                     frictionloss=DRAWER_FRICTION, damping=DRAWER_DAMPING)
  drawer.add_geom(name="drawer_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.25, 0.22, 0.06], pos=[0.25, 0, 0], mass=DRAWER_KG,
                  rgba=[0.7, 0.55, 0.4, 1], contype=0, conaffinity=0)
  drawer.add_geom(name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.01, 0.24, 0.07], pos=[-0.01, 0, 0], mass=0.3,
                  rgba=[0.55, 0.42, 0.3, 1], contype=0, conaffinity=0)
  for k, y in enumerate((-HANDLE_HALF + 0.01, HANDLE_HALF - 0.01)):
    drawer.add_geom(name=f"handle_post_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.006, HANDLE_OFF / 2, 0], pos=[-0.02 - HANDLE_OFF / 2, y, 0],
                    quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0],
                    rgba=[0.75, 0.75, 0.78, 1], contype=0, conaffinity=0, mass=0.02)
  drawer.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=[HANDLE_R, HANDLE_HALF, 0], pos=[-0.02 - HANDLE_OFF, 0, 0],
                  quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
                  rgba=[0.8, 0.8, 0.84, 1], mass=0.1, condim=4, priority=1,
                  friction=[1.0, 0.01, 0.0002])


def handle_goal(rot=FORWARD) -> HandGoal:
  """Where the handle is, told to the robot in its stance frame (staged):
  the grip's middle on the bar's."""
  return HandGoal(np.array([COUNTER_FRONT - 0.02 - HANDLE_OFF, HANDLE_Y, HANDLE_Z]), rot)


def handle_force(h: Humanoid, side: str = "left") -> np.ndarray:
  """The force the pads truly put on the handle, N, in the stance frame:
  OUR grading, off the contacts, never the robot's."""
  m, d = h.m, h.d
  handle = m.geom("handle").id
  pads = {m.geom(f"{side}_{j}_pad").id for j in hm.JAWS}
  total = np.zeros(3)
  f6 = np.zeros(6)
  for i in range(d.ncon):
    c = d.contact[i]
    pair = {int(c.geom1), int(c.geom2)}
    if handle in pair and pair & pads:
      mujoco.mj_contactForce(m, d, i, f6)
      fw = c.frame.reshape(3, 3).T @ f6[:3]
      # The contact's force acts on geom2 from geom1: on the handle when it
      # is geom2.
      total += fw if int(c.geom2) == handle else -fw
  rot_w, _ = true_stance(m, d)
  return rot_w.T @ total


def grasp_scene(h: Humanoid, records: list | None = None):
  """The `grasp` scene, once: the hand to the drawer's handle, open; shut
  and judged by the gripper's own reading; the drawer drawn out with a
  compliant arm, what the hand felt said beside the truth; let go."""
  side = "left"
  g = h.reach.grippers[side]
  rec = {}
  g.open()
  yield "reaching the drawer's handle, the gripper open"
  goal = handle_goal()
  rec["planned"] = h.reach.reach(side, goal)
  if not rec["planned"]:
    yield "  no posture puts the handle in reach"
    return
  t0 = h.d.time
  while not h.reach.done(side) and h.d.time - t0 < 20.0:
    h.step()
    yield None
  yield "shutting the gripper"
  g.shut()
  yield from wait(h, 0.6)
  gap, squeeze = g.read()
  rec["held"] = g.held()
  yield (f"  {'HELD' if rec['held'] else 'NOT HELD'} by its own reading: the jaws "
         f"{1000 * gap:.0f} mm apart, squeezing {squeeze:.0f} N")
  if rec["held"]:
    yield f"pulling the drawer {100 * PULL_M:.0f} cm out, the arm compliant"
    h.reach.stiffen(side, PULL_STIFFNESS)
    h.reach.move_goal(side, goal.pos - np.array([PULL_M, 0.0, 0.0]))
    felt, truth = [], []
    t0 = h.d.time
    while not h.reach.done(side) and h.d.time - t0 < 10.0:
      h.step()
      if h.reach.steps % 25 == 0:
        felt.append(h.reach.force[side].copy())
        truth.append(handle_force(h, side))
      yield None
    for _ in wait(h, 1.0):
      if h.reach.steps % 25 == 0:
        felt.append(h.reach.force[side].copy())
        truth.append(handle_force(h, side))
      yield None
    out = h.d.qpos[h.m.jnt_qposadr[h.m.joint("drawer").id]]
    rec["drawer_m"] = float(out)
    felt, truth = np.array(felt), np.array(truth)
    rec["felt"], rec["truth"] = felt, truth
    yield (f"  the drawer {100 * out:.1f} cm out; the hand felt a pull of "
           f"{-felt[-25:, 0].mean():.1f} N along it, truly {-truth[-25:, 0].mean():.1f} N")
  yield "letting go"
  g.open()
  yield from wait(h, 0.5)
  h.reach.stiffen(side, 1.0)
  h.reach.retreat(side)
  t0 = h.d.time
  while h.reach.phase[side] != "rest" and h.d.time - t0 < 20.0:
    h.step()
    yield None
  g.shut()
  yield from posture(h, 0.0, 0.0)
  rec["fell"] = h.fallen()
  if records is not None:
    records.append(rec)


# ---- the viewer and the filmstrips --------------------------------------------------

#: The standing policy a scene flies (`--policy`), or None: the committed one.
_POLICY = None


def _world_for(scene: str, policy=None):
  policy = policy or _POLICY
  if scene == "stand":
    model, data = standing_world()
  elif scene == "reach":
    model, data = reach_world()
  else:
    model, data = standing_world(counter_scenery)
  return model, data, Humanoid(model, data, policy)


def _scene_gen(scene: str, h: Humanoid):
  return {"stand": stand_scene, "reach": reach_scene, "grasp": grasp_scene}[scene](h)


def view_stage2(scene: str) -> None:
  from mujoco import viewer as mj_viewer
  model, data, h = _world_for(scene)
  print(f"the `{scene}` scene: the humanoid's legs on its standing policy, its "
        "arms on the reach, reading only its own senses. Close the window to quit.")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(ROBOT_ROOT).id
    az, el, dist, _ = SCENE_CAMS.get(scene, (CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE, None))
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = dist, az, el
    while viewer.is_running():
      # Again and again: each time from a fresh world, the robot standing.
      mujoco.mj_resetData(model, data)
      _pose(model, data, hm.stand_qpos())
      h = Humanoid(model, data, _POLICY)
      wall = time.time() - data.time
      for line in _scene_gen(scene, h):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)


#: Each scene's camera, where it is not the body scene's: (azimuth,
#: elevation, distance, look-at offset from the pelvis). The grasp is seen
#: from the robot's right, or the counter hides the hand.
SCENE_CAMS = {"grasp": (90.0, -15.0, 2.0, (0.3, 0.0, 0.15))}


def filmstrip_stage2(scene: str, out: str, every_s: float | None = None) -> None:
  """A scene's filmstrip: a tile when each part of it begins, and its
  last moment, each labelled."""
  from PIL import Image, ImageDraw
  model, data, h = _world_for(scene)
  r = mujoco.Renderer(model, 300, 400)
  root = model.body(ROBOT_ROOT).id
  tiles = []
  pending = None

  az, el, dist, off = SCENE_CAMS.get(scene, (CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE,
                                             (0.0, 0.0, -0.1)))

  def shoot(label):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[root] + off
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    ImageDraw.Draw(img).text((8, 8), label[:60], fill=(255, 255, 255))
    tiles.append(img)
  # A part is shot a moment in, once its motion has begun.
  delay = 0.0
  for line in _scene_gen(scene, h):
    if line and not line.startswith("  "):
      if pending is not None:
        shoot(pending)
      pending, delay = line, data.time + 0.9
    elif line:
      print(line)
    if pending is not None and data.time >= delay:
      shoot(pending)
      pending = None
  if pending is not None:
    shoot(pending)
  r.close()
  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  strip = Image.new("RGB", (400 * cols, 300 * rows), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % cols), 300 * (k // cols)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 2's measurements: on a bare floor, a fresh robot a trial ----------------

def floor_humanoid(scenery=None, policy=None) -> Humanoid:
  """The robot standing on a bare floor at the world's origin, facing +x,
  with `scenery(spec)` round it: a measurement's world (no house to
  compile)."""
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  if scenery is not None:
    scenery(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return Humanoid(model, data, policy)


def _drain(gen) -> None:
  for _ in gen:
    pass


#: The shoves `--shove` tries, N*s at the trunk over `SHOVE_S` (a 42.6 kg
#: body's velocity changed by impulse / 42.6), in eight directions.
SHOVE_LEVELS = (5, 10, 15, 20, 25, 30, 40, 50, 60)
#: A sole that moved this far was a step, m.
STEPPED_M = 0.03


def shove_trial(toward, impulse: float, policy=None, posture_=(0.0, 0.0)) -> str:
  """One shove on a fresh robot standing (at `posture_`): `held` (feet
  where they were), `stepped` (a sole moved, standing after), or `fell`."""
  h = floor_humanoid(policy=policy)
  _drain(posture(h, *posture_))
  _drain(wait(h, 1.0))
  soles = [h.m.geom(f"{s}_sole").id for s in hm.SIDES]
  before = h.d.geom_xpos[soles].copy()
  _drain(shove(h, toward, impulse))
  for _ in wait(h, 4.0):
    if h.fallen():
      return "fell"
  moved = float(np.linalg.norm(h.d.geom_xpos[soles][:, :2] - before[:, :2], axis=1).max())
  return "stepped" if moved > STEPPED_M else "held"


def shove_table(policy=None, postures=((0.0, 0.0), (-0.2, 0.0))) -> dict:
  """--shove: the largest shove each way the robot stands through with its
  feet where they were, and the largest it survives at all, by posture."""
  dirs = {"from the front": (-1, 0), "from behind": (1, 0), "from its left": (0, -1),
          "from its right": (0, 1), "front-left": (-1, -1), "front-right": (-1, 1),
          "back-left": (1, -1), "back-right": (1, 1)}
  out = {}
  for p in postures:
    print(f"posture {p[0]:+.2f} m, {p[1]:.2f} rad: shoves of {SHOVE_LEVELS} N*s, "
          f"{SHOVE_S:.1f} s each")
    print(f"  {'direction':16s} {'feet held to':>13s} {'survived to':>12s}  (N*s; 'fell at')")
    for name, toward in dirs.items():
      held = survived = 0
      fell_at, holding = None, True
      for level in SHOVE_LEVELS:
        result = shove_trial(toward, level, policy, p)
        if result == "fell":
          fell_at = level
          break
        survived = level
        holding = holding and result == "held"
        if holding:
          held = level
      out[(p, name)] = (held, survived, fell_at)
      print(f"  {name:16s} {held:13d} {survived:12d}  {'fell at %d' % fell_at if fell_at else 'never fell'}")
  return out


#: The reach envelope's grid: heights, m, from the floor to over the head,
#: and distances ahead of the stance frame's origin, m, the left hand in
#: front of its shoulder (0.2 m out).
ENVELOPE_Z = (0.09, 0.2, 0.35, 0.49, 0.65, 0.8, 0.94, 1.1, 1.25, 1.43, 1.55)
ENVELOPE_X = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)


def envelope_table(policy=None, fly: bool = True) -> dict:
  """--envelope: each grid point planned on the robot's model of itself
  (fingers forward, else down), and every planned one FLOWN: reached and
  held 2 s balancing, the grip's true miss through the hold."""
  out = {}
  print("the left hand's reach, stance frame: a cell is the grip's mean miss held, "
        "cm (F fingers forward, U tilted 45 deg up, D down), '--' planned but stopped "
        "short, missed or fell, '.' no posture the policy holds puts it in reach")
  print(f"{'z \\ x':>6s}" + "".join(f"{x:>8.2f}" for x in ENVELOPE_X))
  for z in reversed(ENVELOPE_Z):
    cells = []
    for x in ENVELOPE_X:
      cell = "."
      # The grips that plan, those with slack first (`reach.plan`).
      planned = []
      for tag, rot in (("F", FORWARD), ("U", UP45), ("D", DOWN)):
        goal = HandGoal(np.array([x, 0.2, z]), rot)
        found = plan("left", goal)
        if found is not None:
          planned.append((not found[3], tag, goal))
      if planned:
        _, tag, goal = min(planned, key=lambda p: p[0])
        if not fly:
          cell = tag
        else:
          h = floor_humanoid(policy=policy)
          rec = {}
          _drain(reach_and_hold(h, "left", goal, record=rec))
          out[(z, x)] = (tag, rec)
          ok = rec.get("arrived") and not rec.get("fell") and rec.get("miss_mean", 1) < 0.03
          cell = f"{100 * rec['miss_mean']:.1f}{tag}" if ok else "--"
      cells.append(f"{cell:>8s}")
    print(f"{z:6.2f}" + "".join(cells), flush=True)
  return out


def hold_table(policy=None, step_m: float = 0.01, step_s: float = 0.5,
               most_m: float = 0.25) -> dict:
  """--hold: the drawer locked, its handle gripped and pulled harder and
  harder -- the hand's goal drawn back `step_m` every `step_s` at the full
  stiffness -- until the grip slips, the body steps or falls, or the goal
  is `most_m` behind. The pull truly on the handle (our grading) beside
  what the hand felt off its motor currents, step by step."""
  h = floor_humanoid(lambda spec: counter_scenery(spec, locked=True, at=(0.0, 0.0), yaw=0.0),
                     policy)
  side = "left"
  g = h.reach.grippers[side]
  g.open()
  goal = handle_goal()
  if not h.reach.reach(side, goal):
    print("the handle is out of reach")
    return {}
  _drain(_until(h, lambda: h.reach.done(side), 20.0))
  g.shut()
  _drain(wait(h, 0.6))
  print(f"held by its own reading: {g.held()} (gap {1000 * g.read()[0]:.0f} mm, "
        f"squeeze {g.read()[1]:.0f} N)")
  site, handle = h.m.site(f"{side}_grip").id, h.m.geom("handle").id
  soles = [h.m.geom(f"{s}_sole").id for s in hm.SIDES]
  feet0 = h.d.geom_xpos[soles].copy()
  rows, why = [], f"the goal {100 * most_m:.0f} cm behind the handle"
  back = 0.0
  while back < most_m:
    back += step_m
    h.reach.move_goal(side, goal.pos - np.array([back, 0.0, 0.0]))
    felt, truth = [], []
    for _ in wait(h, step_s):
      felt.append(-h.reach.force[side][0])
      truth.append(-handle_force(h, side)[0])
    slip = float(np.linalg.norm(h.d.site_xpos[site] - h.d.geom_xpos[handle]))
    moved = float(np.linalg.norm(h.d.geom_xpos[soles][:, :2] - feet0[:, :2], axis=1).max())
    half = len(truth) // 2
    rows.append((back, float(np.mean(truth[half:])), float(np.mean(felt[half:])),
                 float(np.std(np.array(felt[half:]) - np.array(truth[half:]))), slip, moved))
    if h.fallen():
      why = "it fell"
      break
    if slip > 0.02:
      why = "the grip slipped"
      break
    if moved > STEPPED_M:
      why = "it stepped"
      break
  print(f"{'goal back cm':>12s} {'pull N, truly':>14s} {'felt N':>8s} {'felt sd':>8s} "
        f"{'grip off handle mm':>19s} {'feet moved mm':>14s}")
  for back, t, f, sd, slip, moved in rows:
    print(f"{100 * back:12.0f} {t:14.1f} {f:8.1f} {sd:8.1f} {1000 * slip:19.1f} {1000 * moved:14.1f}")
  print(f"stopped: {why}; the hardest steady pull, truly {max(r[1] for r in rows):.1f} N")
  return {"rows": rows, "why": why}


def _until(h: Humanoid, done, budget_s: float):
  t0 = h.d.time
  while not done() and h.d.time - t0 < budget_s and not h.fallen():
    h.step()
    yield None


SCENES = {"body": view_body, "stand": lambda: view_stage2("stand"),
          "reach": lambda: view_stage2("reach"), "grasp": lambda: view_stage2("grasp")}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--view", choices=sorted(SCENES), default=None, metavar="SCENE",
                  help=f"a scene in the viewer: {', '.join(sorted(SCENES))}")
  ap.add_argument("--strip", choices=sorted(SCENES), default=None, metavar="SCENE",
                  help="a scene's filmstrip, headless (--out names it)")
  ap.add_argument("--policy", default=None,
                  help="fly another standing policy's .npz (default models/humanoid_stand.npz)")
  ap.add_argument("--shove", action="store_true",
                  help="the shoves it stands through, each way, standing and crouched")
  ap.add_argument("--envelope", action="store_true",
                  help="where the hand reaches, floor to over the head, flown balancing")
  ap.add_argument("--hold", action="store_true",
                  help="a locked drawer's handle pulled harder until something gives")
  ap.add_argument("--mass", action="store_true", help="the mass budget beside the T2's")
  ap.add_argument("--torque", action="store_true",
                  help="each joint's load: held in our physics, then the estimates")
  ap.add_argument("--thermal", action="store_true", help="the hottest winding, held")
  ap.add_argument("--energy", action="store_true", help="watts by activity, the pack's hours")
  ap.add_argument("--sensors", action="store_true", help="what the head sees from the stand")
  ap.add_argument("--out", default="humanoid_spike.png")
  args = ap.parse_args(argv)
  policy = StandPolicy(args.policy) if args.policy else None
  if policy is not None:
    global _POLICY
    _POLICY = policy
  if args.view:
    SCENES[args.view]()
  elif args.strip and args.strip != "body":
    filmstrip_stage2(args.strip, args.out)
  elif args.shove:
    shove_table(policy)
  elif args.envelope:
    envelope_table(policy)
  elif args.hold:
    hold_table(policy)
  elif args.mass:
    mass_table()
  elif args.torque:
    torque_table()
    print()
    estimate_table()
  elif args.thermal:
    thermal_table()
  elif args.energy:
    energy_table()
    print()
    walking_energy()
  elif args.sensors:
    sensors_table()
  else:
    filmstrip(args.out)


if __name__ == "__main__":
  sys.exit(main())
