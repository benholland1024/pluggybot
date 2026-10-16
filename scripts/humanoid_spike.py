"""The humanoid spike (issue #507): our own body from real parts, sized
before anything is trained, and a scene for each part to watch.

The body is `pluggybot.humanoid.model.HumanoidSpec`, its parts
`pluggybot.humanoid.parts`; SimNotes, "The humanoid body", has every table
below and Parts.md, "The humanoid body", the parts and the bill.

  --view SCENE  a scene in MuJoCo's viewer, at real time, over and over
                until the window closes (MUJOCO_GL unset); the terminal says
                what is moving. Scenes: `body` -- the humanoid in the house's
                living room, held up on a gantry, moving each joint through
                its range in turn, then its grippers.
  --mass        the mass budget by segment, beside the T2's published model
  --torque      the torque table: each joint's load against its actuator's
                peak and continuous ratings, standing, crouched, on one
                leg, reaching, and -- ESTIMATES until stages 4 and 5 have
                policies -- walking and on the house's stairs
  --thermal     the hottest winding, each activity sustained
  --energy      watts by activity, and what the pack buys
  --sensors     what the head's cameras and the LIDAR see from the stand
  (default)     the `body` scene's filmstrip, humanoid_spike.png

Usage:
  uv run python scripts/humanoid_spike.py --view body
  MUJOCO_GL=egl uv run python scripts/humanoid_spike.py [--torque|--mass|...]
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
#: damping ratio 2) but at twice the frequency -- the stiffness four times,
#: the damping twice. At the policies' own an arm held out sagged 12 cm at
#: the hand; a scene has no policy offsetting its targets. Inside every
#: part's MIT range (the RS00's Kp to 500, the RS04's to 5000).
GAINS = np.array([driver_gains(hm.CHOSEN.motor(j)) for j in hm.JOINT_NAMES]) * [4.0, 2.0]


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


SCENES = {"body": view_body}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--view", choices=sorted(SCENES), default=None, metavar="SCENE",
                  help=f"a scene in the viewer: {', '.join(sorted(SCENES))}")
  ap.add_argument("--mass", action="store_true", help="the mass budget beside the T2's")
  ap.add_argument("--torque", action="store_true",
                  help="each joint's load: held in our physics, then the estimates")
  ap.add_argument("--thermal", action="store_true", help="the hottest winding, held")
  ap.add_argument("--energy", action="store_true", help="watts by activity, the pack's hours")
  ap.add_argument("--sensors", action="store_true", help="what the head sees from the stand")
  ap.add_argument("--out", default="humanoid_spike.png")
  args = ap.parse_args(argv)
  if args.view:
    SCENES[args.view]()
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
