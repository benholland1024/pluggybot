"""The humanoid spike (issue #507): our own body from real parts, sized
before anything is trained, and a scene for each part to watch.

The body is `pluggybot.humanoid.model.HumanoidSpec`, its parts
`pluggybot.humanoid.parts`; SimNotes, "The humanoid body" and the three
sections after it, has every table below and Parts.md, "The humanoid body",
the parts and the bill.

  --view SCENE  a scene in MuJoCo's viewer, at real time, over and over
                until the window closes (MUJOCO_GL unset); the terminal says
                what is moving. Scenes: `body` -- the humanoid in the house's
                living room, held up on a gantry, moving each joint through
                its range in turn, then its grippers; `stand` -- on its own
                feet, its standing policy, shoved, crouching and bowing;
                `reach` -- a hand to a counter, a high shelf, a low shelf
                and the floor, each held, through `Body` with its get-up
                loaded as served; `grasp` -- a drawer's handle
                gripped (judged by the gripper's own reading) and the drawer
                drawn out with a compliant arm, what the hand felt said
                beside the truth; `pull` -- a hand held out where a
                handle would be, pulling at the `pull` verb's most, 20 N,
                each way in turn and pushing away, the force staged on the
                hand (#544); `bar` -- the `pull` verb on a locked handle at
                its most, back, up, forward and down, how each ended and the
                most the bar took said (#547); `feel` -- a hand held out where the
                bar would be, 20 N staged at its grip pulling back, then
                pushing away, away and down, and up, what its arm read and
                what the whole body read said (#547); `demo` -- #466's chest
                on a counter, its
                knob found by its tag, reached and gripped, every act a
                `Body` member; `walk` -- from the living room into the
                kitchen and back, steered by its own map; `rough` -- the walker
                that sees the ground (not served) across doorsteps, paving, a
                slope and blocks; `stairs` -- it up the house's flight on its
                own map, round on the landing and down; `fall` -- shoved over
                three ways, the get-up taking each fall; `getup` -- up off its
                back, its front and its side
  --strip SCENE the scene's filmstrip, headless (`--out` names it)
  --policy NPZ  fly another standing policy (default models/humanoid_stand.npz)
  --mass        the mass budget by segment, beside the T2's published model
  --torque      the torque table: each joint's load against its actuator's
                peak and continuous ratings, standing, crouched, on one
                leg, reaching; walking and on the house's stairs as
                ESTIMATES; then walking FLOWN by the walking policy (rule 7)
  --thermal     the hottest winding, each activity sustained
  --energy      watts by activity, and what the pack buys
  --sensors     what the head's cameras and the LIDAR see from the stand
  --shove       the shoves it stands through each way, standing and crouched
  --envelope    where a hand reaches, floor to over the head, flown balancing
  --hold        a locked handle gripped and pulled harder until something gives
  --pulls       a hand held where a locked handle's bar is, a force staged on
                it each way, harder until the body falls (#544)
  --bar         the `pull` verb's force move on the locked handle, each way at
                10 and 20 N: the force on the bar against what the hand read,
                the postures taken, how it ended (#547); `--oracle` closes
                the move on the bar's true force instead: the loop's floor
  --feel        what the hand's readings -- its arm's own and the whole
                body's -- make of 20 N staged at the grip, each way, standing
                still and swaying (#547)
  --handover [N]  the walk's hand-over at a standstill, through the body's own
                `step`, at N counter set-outs (8)
  --demo-table [N]  the demo's first N set-outs (12), each flown through Body
  --track [NPZ,...]  the served walker and the one that sees the ground (or
                these) asked each command on a bare floor: what they walk
  --odometry    the served walker's leg odometry against the truth
  --rough [NPZ,...]  the served walker and the one that sees the ground (or
                these) over a doorstep, paving, a slope up and down and a
                grid of blocks
  (default)     the `body` scene's filmstrip, humanoid_spike.png

Usage:
  uv run python scripts/humanoid_spike.py --view body|stand|reach|grasp|pull|bar|feel|demo|walk|rough|stairs|fall|getup
  MUJOCO_GL=egl uv run python scripts/humanoid_spike.py [--torque|--shove|...]
"""

import os

# One BLAS thread before numpy loads (small matrix products ran slower on
# six).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.humanoid import model as hm  # noqa: E402
from pluggybot.humanoid import parts as hp  # noqa: E402
from pluggybot.humanoid.actuator import driver_gains  # noqa: E402
from pluggybot.humanoid.drivers import Drivers  # noqa: E402
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
  world: `humanoid.world` builds that one)."""
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


def _pose(model, data, q: dict, prefix: str = "") -> None:
  for name in hm.qpos_order():
    data.qpos[model.jnt_qposadr[model.joint(prefix + name).id]] = q.get(name, 0.0)
  mujoco.mj_forward(model, data)


#: The gains the scenes' commands carry: each joint's loop tuned on its own
#: rotor as the policies' are (`humanoid.actuator.driver_gains`, 10 Hz at
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


# ---- rule 7's re-read: walking, off the walking policy -----------------------------

#: The speeds the walking rows are flown at, m/s: a stroll, a walk, its top.
WALK_SPEEDS = (0.5, 1.0, 1.3)
#: Each is flown this long, s, its first `WALK_SETTLE_S` (the start) left
#: out of the record.
WALK_S, WALK_SETTLE_S = 10.0, 3.0


def walking_rows() -> list[tuple[str, dict]]:
  """Each speed walked on a bare floor by the walking policy (`models/
  humanoid_walk.npz`), its senses the robot's, its arms held hanging by the
  reach: every leg joint's |torque| peak and RMS after the start, and
  whether it kept its feet (`hold`'s record's shape)."""
  rows = []
  for v in WALK_SPEEDS:
    m, d = floor_world()
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    w = _walker(m, d, None)
    w.walk.ask(v, 0.0, 0.0)
    act = [m.actuator(j).id for j in hm.JOINT_NAMES]
    root = m.body(ROBOT_ROOT).id
    taus, stood = [], True
    for _ in range(int(WALK_S / m.opt.timestep)):
      w.step()
      if d.time >= WALK_SETTLE_S:
        taus.append(np.abs(d.actuator_force[act]))
      if d.xpos[root][2] < FALLEN_M:
        stood = False
        break
    t = np.array(taus) if taus else np.full((1, len(act)), np.nan)
    rows.append((f"walking {v:g} m/s", {"mean": np.sqrt((t ** 2).mean(axis=0)),
                                        "max": t.max(axis=0), "stood": stood, "outside": []}))
  return rows


def walking_table() -> None:
  """Rule 7: each leg joint walking, FLOWN -- its peak against its part's
  peak, its RMS against the part's continuous rating turning, and the
  winding that RMS holds steady at, 40 C ambient (`thermal_table`'s rule).
  A joint past either rating sends the design back (#507)."""
  rows = walking_rows()
  print("WALKING, FLOWN by the walking policy in our physics (rule 7's re-read; "
        "the estimates above are superseded for walking, the stairs' stand until stage 5):")
  print(f"{'joint':15s}" + "".join(f"{n:>37s}" for n, _ in rows))
  print(f"{'':15s}" + "".join(f"{'peak N*m (% peak)  RMS (% rated) C':>37s}" for _ in rows))
  for kind in hm.LEG:
    m = hm.CHOSEN.drives[kind]
    cells = []
    for _, rec in rows:
      if not rec["stood"]:
        cells.append(f"{'FELL':>37s}")
        continue
      ks = [k for k, j in enumerate(hm.JOINT_NAMES) if hm.kind(j) == kind]
      peak = float(rec["max"][ks].max())
      rms = float(rec["mean"][ks].max())
      winding = AMBIENT_C[1] + (m.max_winding_c - hp.RS_RATED_AMBIENT_C) * (rms / m.rated_torque) ** 2
      cells.append(f"{peak:6.1f} ({100 * peak / m.peak_torque:3.0f} %) {rms:5.1f} "
                   f"({100 * rms / m.rated_torque:3.0f} %) {winding:4.0f}")
    print(f"{kind:15s}" + "".join(f"{c:>37s}" for c in cells))


def stairs_rows(npz: str | None = None) -> list[tuple[str, dict]]:
  """The house's flight (0.18 m risers) walked up and down by the seeing
  walker on its own map's scan: every joint's |torque| peak and RMS from the
  foot of the flight to its top (`climb_trial`'s course), whether it got
  there, and how long it was on the flight (a burst, `flown_table`)."""
  rows = []
  for down in (False, True):
    model, data, foot, end = flight_world(hw.STAIRS_RISE, down=down)
    w = Seeing(model, data, scan="map", npz=npz)
    act = [model.actuator(j).id for j in hm.JOINT_NAMES]
    heading = math.pi if down else 0.0
    goal = foot - CLIMB_PAST_M if down else end + CLIMB_PAST_M
    taus, there, steer = [], False, max(1, round(0.1 / model.opt.timestep))
    for k in range(int(CLIMB_S / model.opt.timestep)):
      if k % steer == 0:
        w.steer(heading, 0.0)
      w.step()
      x = float(data.xpos[w.root][0])
      if foot - 0.3 < x < end + 0.3:
        taus.append(np.abs(data.actuator_force[act]))
      if data.xmat[w.root][8] < 0.5:
        break
      if (x < goal) if down else (x > goal):
        there = True
        break
    t = np.array(taus) if taus else np.full((1, len(act)), np.nan)
    rows.append((f"stairs {'down' if down else 'up'}", {
      "mean": np.sqrt((t ** 2).mean(axis=0)), "max": t.max(axis=0), "stood": there,
      "outside": [], "seconds": len(taus) * model.opt.timestep}))
  return rows


def getup_rows(npz: str | None = None) -> list[tuple[str, dict]]:
  """The get-up policy from lying on its back, its front and its side on a
  bare floor: every joint's |torque| peak and RMS until it stood, and
  whether it did (in 10 s)."""
  rows = []
  for name, (roll, pitch) in LYING.items():
    f = Falling(floor_humanoid(), npz)
    h = f.h
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, 0.0]), "XYZ")
    h.d.qpos[2], h.d.qpos[3:7] = 0.15, q
    mujoco.mj_forward(h.m, h.d)
    h.stand = StandDriver(h.m, h.d)
    h.reach = Reach(h.m, h.d, h.stand)
    f.rebuild()
    f.take()
    act = [h.m.actuator(j).id for j in hm.JOINT_NAMES]
    taus = []
    while f.up and h.d.time < 10.0:
      f.step()
      taus.append(np.abs(h.d.actuator_force[act]))
    t = np.array(taus)
    rows.append((f"up {name}", {"mean": np.sqrt((t ** 2).mean(axis=0)), "max": t.max(axis=0),
                                "stood": not f.up, "outside": [], "seconds": float(h.d.time)}))
  return rows


def flown_table(title: str, rows: list[tuple[str, dict]], kinds=hm.LEG) -> None:
  """Rule 7's re-read of FLOWN rows: each joint kind's peak against its
  part's peak, its RMS against the part's continuous rating, and the winding
  that RMS holds steady at, 40 C ambient (`walking_table`'s columns) -- or,
  for a burst of a few seconds (a row carrying `seconds`), the burst's
  length: a winding's time constant is minutes, so no steady state."""
  print(title)
  print(f"{'joint':15s}" + "".join(f"{n:>37s}" for n, _ in rows))
  print(f"{'':15s}" + "".join(f"{'peak N*m (% peak)  RMS (% rated) C':>37s}" for _ in rows))
  bursts = [r.get("seconds") for _, r in rows]
  if any(b is not None for b in bursts):
    print(f"{'(a burst, s)':15s}" + "".join(f"{'' if b is None else f'{b:.1f} s':>37s}"
                                            for b in bursts))
  for kind in kinds:
    m = hm.CHOSEN.drives[kind]
    cells = []
    for _, rec in rows:
      ks = [k for k, j in enumerate(hm.JOINT_NAMES) if hm.kind(j) == kind]
      peak = float(rec["max"][ks].max())
      rms = float(rec["mean"][ks].max())
      winding = AMBIENT_C[1] + (m.max_winding_c - hp.RS_RATED_AMBIENT_C) * (rms / m.rated_torque) ** 2
      mark = "" if rec["stood"] else "!"
      heat = "   -" if rec.get("seconds") is not None else f"{winding:4.0f}"
      cells.append(f"{mark}{peak:6.1f} ({100 * peak / m.peak_torque:3.0f} %) {rms:5.1f} "
                   f"({100 * rms / m.rated_torque:3.0f} %) {heat}")
    print(f"{kind:15s}" + "".join(f"{c:>37s}" for c in cells))
  print("  (! did not get there: its numbers are up to where it stopped)")


#: Ambient air for the thermal table, C: a warm room and a hot day.
AMBIENT_C = (30.0, 40.0)


def copper_w(motor, tau: float) -> float:
  """Heat in a winding at output torque `tau`, W: 1.5 R I^2 with I = tau /
  Kt the phase current's amplitude (`humanoid.actuator.JointLimits.copper_w`)."""
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


def friction_table(write: bool = False) -> None:
  """--friction: the arm joints' friction calibrated as the robot does it
  (`humanoid.friction.calibrate_routine`), standing in the living room,
  each joint's fit beside the friction its motor's sheet gives the
  model (`parts.friction_nm`, the sim's truth: ours, never the robot's)."""
  from pluggybot.body import body_for
  from pluggybot.humanoid import friction
  model, data = standing_world()
  body = body_for(model, data, policy=_POLICY)
  try:
    body.run(body.hold_routine(1.0))
    table = body.run(friction.calibrate_routine(body))
  finally:
    body.close()
  print(f"{'joint':22s} {'dry N*m':>8s} {'viscous':>8s} {'sheet':>7s}  samples")
  for name, row in table.items():
    sheet = hp.friction_nm(hm.CHOSEN.motor(name))
    print(f"{name:22s} {row['coulomb']:8.3f} {row['viscous']:8.3f} {sheet:7.3f}  {row['samples']}")
  if write:
    friction.save({n: {"coulomb": r["coulomb"], "viscous": r["viscous"]} for n, r in table.items()})
    print(f"kept in {friction.FILE}")


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
from pluggybot.procedure.steps import PULL_MOST_N  # noqa: E402

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


def standing_world(scenery=None, at=STAND_XY, yaw: float = STAND_YAW, z: float = 0.0):
  """The house with the humanoid standing on its soles in the living room
  (or on ground `z` up at `at`), and whatever `scenery(spec)` adds: the
  model and its data at the stand."""
  spec = mujoco.MjSpec.from_file(HOME_XML)
  frame = spec.worldbody.add_frame()
  frame.pos = [at[0], at[1], z]
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
    # the standing policy has the legs: a hand's force off the whole body
    self.reach.standing = True
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


def reach_body_world():
  """The `reach` scene's world and the humanoid behind `Body` in it, its
  get-up loaded as served -- its own fall watch at work, as wherever it
  stands (#520: flown on the standing rig alone, the scene never showed
  that the watch took the deepest bows for falls)."""
  from pluggybot.body import body_for
  model, data = reach_world()
  body = body_for(model, data, policy=_POLICY)
  return model, data, body


def reach_scene(body, model, data, records: list | None = None):
  """The `reach` scene, once, through `Body`: the left hand to each of
  `REACH_TARGETS` -- told to the robot in its stance frame -- held, and
  home. Yields a line when a part begins or ends. Where the grip truly is
  is OUR grading; its falls are its own watch's."""
  from pluggybot.humanoid.body import _rz
  site = model.site("left_grip").id
  body.start_at(*STAND_XY, STAND_YAW)
  # Its stance is its odometry's, posed by its first ticks.
  yield from drive(body, body.settle_routine(0.5))
  for label, goal in REACH_TARGETS:
    yield f"reaching {label}"
    rec = {"target": label}
    if records is not None:
      records.append(rec)
    falls, t0 = body.falls, data.time
    x, y, yaw = body._true_stance()
    truly = _rz(yaw) @ goal.pos + [x, y, 0.0]
    rot = None if goal.rot is None else _rz(body.stance[2]) @ goal.rot
    rec["reached"] = yield from drive(body, body.reach_routine(
      "left", body.to_world(goal.pos), rot, patience=15.0))
    rec["seconds"], rec["phase"] = data.time - t0, body.reach.phase["left"]
    rec["posture"] = tuple(float(v) for v in body.reach.posture)
    rec["slack"] = body.reach.slack["left"]
    if rec["reached"]:
      yield f"holding at {label}"
      misses = []
      for _ in drive(body, body.settle_routine(2.0)):
        misses.append(float(np.linalg.norm(data.site_xpos[site] - truly)))
        yield None
      rec["miss_mean"], rec["miss_max"] = float(np.mean(misses)), float(np.max(misses))
    yield from drive(body, body.release_routine("left"))
    rec["falls"] = body.falls - falls
    if rec["falls"]:
      yield (f"  FELL by its own watch ({rec['falls']}): the get-up took every joint; "
             f"the reach said {rec['reached']}")
    elif rec["reached"]:
      yield (f"  posture {rec['posture'][0]:+.2f} m, {rec['posture'][1]:.2f} rad"
             f"{'' if rec['slack'] else ', the arm at full stretch'}; there in "
             f"{rec['seconds']:.1f} s; the grip {100 * rec['miss_mean']:.1f} cm off it "
             f"held (worst {100 * rec['miss_max']:.1f}); no fall by its own watch")
    elif rec["phase"] == "blocked":
      yield ("  stopped short: its own solve put the goal out of reach, or its hand "
             "pushed on something")
    elif rec["phase"] == "rest":
      yield "  no posture puts it in reach"
    else:
      yield f"  not there in 15 s ({rec['phase']})"


def view_reach() -> None:
  from mujoco import viewer as mj_viewer
  model, data, body = reach_body_world()
  print("the `reach` scene: the left hand to a counter, a high shelf, a low shelf and the "
        "floor, each held, through Body with its get-up loaded. Close the window to quit.")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body.root
    viewer.cam.distance, viewer.cam.azimuth = CAM_DISTANCE, CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION
    while viewer.is_running():
      wall = time.time() - data.time
      for line in reach_scene(body, model, data):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)


def filmstrip_reach(out: str) -> None:
  """The `reach` scene's filmstrip: a tile as it reaches each target and
  as it holds there, labelled."""
  from PIL import Image, ImageDraw
  model, data, body = reach_body_world()
  r = mujoco.Renderer(model, 300, 400)
  tiles, pending, at = [], None, 0.0

  def shoot(label):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[body.root] + [0.0, 0.0, -0.1]
    cam.distance, cam.azimuth, cam.elevation = CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    ImageDraw.Draw(img).text((8, 8), label[:60], fill=(255, 255, 255))
    tiles.append(img)
  for line in reach_scene(body, model, data):
    if line and not line.startswith("  "):
      if pending is not None:
        shoot(pending)
      pending, at = line, data.time + 0.9
    elif line:
      print(line)
    if pending is not None and data.time >= at:
      shoot(pending)
      pending = None
  if pending is not None:
    shoot(pending)
  r.close()
  body.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)),
                    (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


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
  else:
    model, data = standing_world(None if scene in ("pull", "feel") else
                                 (lambda spec: counter_scenery(spec, locked=True)) if scene == "bar"
                                 else counter_scenery)
  return model, data, Humanoid(model, data, policy)


def _scene_gen(scene: str, h: Humanoid):
  return {"stand": stand_scene, "grasp": grasp_scene, "pull": pull_scene,
          "bar": bar_scene, "feel": feel_scene}[scene](h)


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
SCENE_CAMS = {"grasp": (90.0, -15.0, 2.0, (0.3, 0.0, 0.15)),
              "pull": (120.0, -12.0, 2.3, (0.25, 0.0, 0.0)),
              "bar": (120.0, -12.0, 2.3, (0.25, 0.0, 0.0)),
              "feel": (120.0, -12.0, 2.3, (0.25, 0.0, 0.0))}


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


SHOVE_DIRS = {"from the front": (-1, 0), "from behind": (1, 0), "from its left": (0, -1),
              "from its right": (0, 1), "front-left": (-1, -1), "front-right": (-1, 1),
              "back-left": (1, -1), "back-right": (1, 1)}


def _shove_row(job) -> tuple:
  """One way's shoves at one posture, rising until it falls: the largest
  with its feet held, the largest it stood, and the one it fell at."""
  p, toward, npz = job
  policy = StandPolicy(npz) if npz else None
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
  return held, survived, fell_at


def shove_table(npz: str | None = None, postures=((0.0, 0.0), (-0.2, 0.0)),
                jobs: int = 4) -> dict:
  """--shove: the largest shove each way the robot stands through with its
  feet where they were, and the largest it survives at all, by posture."""
  import multiprocessing as mp
  work = [(p, toward, npz) for p in postures for toward in SHOVE_DIRS.values()]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    got = iter(pool.map(_shove_row, work))
  out = {}
  print(f"policy {_name_stand(npz)}")
  for p in postures:
    print(f"posture {p[0]:+.2f} m, {p[1]:.2f} rad: shoves of {SHOVE_LEVELS} N*s, "
          f"{SHOVE_S:.1f} s each")
    print(f"  {'direction':16s} {'feet held to':>13s} {'survived to':>12s}  (N*s; 'fell at')")
    for name in SHOVE_DIRS:
      held, survived, fell_at = out[(p, name)] = next(got)
      print(f"  {name:16s} {held:13d} {survived:12d}  {'fell at %d' % fell_at if fell_at else 'never fell'}")
  return out


#: The reach envelope's grid: heights, m, from the floor to over the head,
#: and distances ahead of the stance frame's origin, m, the left hand in
#: front of its shoulder (0.2 m out).
ENVELOPE_Z = (0.09, 0.2, 0.35, 0.49, 0.65, 0.8, 0.94, 1.1, 1.25, 1.43, 1.55)
ENVELOPE_X = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)


def _envelope_cell(job) -> dict:
  """One cell flown: the hand reached to `goal` and held 2 s balancing."""
  goal, npz = job
  h = floor_humanoid(policy=StandPolicy(npz) if npz else None)
  rec = {}
  _drain(reach_and_hold(h, "left", goal, record=rec))
  return rec


def envelope_table(npz: str | None = None, fly: bool = True, jobs: int = 4) -> dict:
  """--envelope: each grid point planned on the robot's model of itself
  (fingers forward, else down), and every planned one FLOWN: reached and
  held 2 s balancing, the grip's true miss through the hold."""
  import multiprocessing as mp
  chosen = {}
  for z in ENVELOPE_Z:
    for x in ENVELOPE_X:
      # The grips that plan, those with slack first (`reach.plan`).
      planned = []
      for tag, rot in (("F", FORWARD), ("U", UP45), ("D", DOWN)):
        goal = HandGoal(np.array([x, 0.2, z]), rot)
        found = plan("left", goal)
        if found is not None:
          planned.append((not found[3], tag, goal))
      if planned:
        _, tag, goal = min(planned, key=lambda p: p[0])
        chosen[(z, x)] = (tag, goal)
  recs = {}
  if fly:
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
      recs = dict(zip(chosen, pool.map(_envelope_cell, [(g, npz) for _, g in chosen.values()])))
  out = {}
  print(f"policy {_name_stand(npz)}")
  print("the left hand's reach, stance frame: a cell is the grip's mean miss held, "
        "cm (F fingers forward, U tilted 45 deg up, D down), '--' planned but stopped "
        "short, missed or fell, '.' no posture the policy holds puts it in reach")
  print(f"{'z \\ x':>6s}" + "".join(f"{x:>8.2f}" for x in ENVELOPE_X))
  for z in reversed(ENVELOPE_Z):
    cells = []
    for x in ENVELOPE_X:
      cell = "."
      if (z, x) in chosen:
        tag = chosen[(z, x)][0]
        if not fly:
          cell = tag
        else:
          rec = recs[(z, x)]
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


#: `--pulls` (#544): the standing policy's own measure, a force at a hand. The
#: left hand held by the reach where the locked handle's bar is (0.84 m up,
#: 0.43 m ahead: a counter's height), pulling or pushing there along each
#: way, stance frame (x ahead, y left, z up), as the verb's way is -- "back"
#: draws the hand toward the robot, the force ON it forward -- `PULL_TIMES`
#: times, each built at `PULL_RATE_N_S` and held `PULL_HOLD_S`, at each force
#: in turn until the body falls: the verb's own force built on the locked bar
#: at up to 87 N/s (pulling up; back and left were slower), and a force
#: stepped on whole topples a body a lean would have held. Staged with the
#: truth. ⚠ NOT THE VERB
#: ON THE BAR: a `pull` takes the posture that reaches its way's end first,
#: and pulled 0.3 m down on a fixed bar the body bowed 48 deg, its grip
#: rolled off and it fell at 20 N -- the hand's stack, not the legs' (`--hold`
#: and the `pull` scene fly the verb on the bar).
PULL_WAYS = {"back": (-1, 0, 0), "forward": (1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0),
             "up": (0, 0, 1), "down": (0, 0, -1), "back-left": (-1, 1, 0),
             "back-right": (-1, -1, 0), "forward-left": (1, 1, 0), "forward-right": (1, -1, 0),
             "back-up": (-1, 0, 1), "back-down": (-1, 0, -1), "forward-up": (1, 0, 1),
             "forward-down": (1, 0, -1)}
PULL_FORCES = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
PULL_TIMES, PULL_RATE_N_S, PULL_HOLD_S, PULL_REST_S, PULL_AFTER_S = 2, 90.0, 3.0, 1.0, 3.0


def pull_trial(way: str, force: float, npz: str | None = None) -> dict:
  """A fresh robot, its left hand held where the bar is, pulling `force` N
  along `PULL_WAYS[way]` -- the force on it the other way -- `PULL_TIMES`
  times, then stood `PULL_AFTER_S`:
  whether it fell (its trunk `getup.FALL_TILT_RAD` off upright, the body's
  own line at the stand, or its pelvis under `FALLEN_M`), stepped (a sole
  moved `STEPPED_M`), the trunk's worst tilt and how far the hand gave."""
  h = floor_humanoid(policy=StandPolicy(npz) if npz else _POLICY)
  side = "left"
  out = {"way": way, "force": force, "fell": False, "stepped": False, "tilt_deg": 0.0,
         "gave_m": 0.0}
  if not h.reach.reach(side, handle_goal()):
    return {**out, "why": "the bar's place is out of reach"}
  _drain(_until(h, lambda: h.reach.done(side), 20.0))
  _drain(wait(h, 0.5))
  rot_w, _ = true_stance(h.m, h.d)
  f = -rot_w @ (np.array(PULL_WAYS[way], dtype=float) / np.linalg.norm(PULL_WAYS[way])) * force
  hand, site = h.m.body(f"{side}_hand").id, h.m.site(f"{side}_grip").id
  soles = [h.m.geom(f"{s}_sole").id for s in hm.SIDES]
  feet0, at0 = h.d.geom_xpos[soles][:, :2].copy(), h.d.site_xpos[site].copy()

  def phase(seconds: float, share) -> bool:
    for k in range(int(round(seconds / h.m.opt.timestep))):
      h.d.xfrc_applied[hand, :3] = f * share(k * h.m.opt.timestep / seconds)
      h.step()
      tilt = math.acos(min(1.0, max(-1.0, float(h.d.xmat[h.root][8]))))
      out["tilt_deg"] = max(out["tilt_deg"], math.degrees(tilt))
      out["fell"] |= h.fallen() or tilt > FALL_TILT_RAD
      out["stepped"] |= float(np.linalg.norm(h.d.geom_xpos[soles][:, :2] - feet0,
                                             axis=1).max()) > STEPPED_M
      out["gave_m"] = max(out["gave_m"], float(np.linalg.norm(h.d.site_xpos[site] - at0)))
      if out["fell"]:
        return False
    return True

  for _ in range(PULL_TIMES):
    if not (phase(force / PULL_RATE_N_S, lambda s: s) and phase(PULL_HOLD_S, lambda s: 1.0)
            and phase(PULL_REST_S, lambda s: 0.0)):
      break
  h.d.xfrc_applied[hand] = 0.0
  if not out["fell"]:
    phase(PULL_AFTER_S, lambda s: 0.0)
  return out


def _pull_way(job) -> list[dict]:
  """One way's trials, the force rising until the body falls."""
  way, npz = job
  rows = []
  for force in PULL_FORCES:
    rows.append(pull_trial(way, force, npz))
    if rows[-1]["fell"] or rows[-1].get("why"):
      break
  return rows


def pull_table(npz: str | None = None, jobs: int = 4) -> dict:
  """--pulls: each way, the hardest pull the body holds without a fall and
  with its feet where they were, `PULL_TIMES` pulls a trial."""
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    ways = pool.map(_pull_way, [(w, npz) for w in PULL_WAYS])
  print(f"THE PULL A HAND HOLDS, the left hand held where the locked handle's bar is "
        f"({HANDLE_Z:.2f} m up, {COUNTER_FRONT - 0.02 - HANDLE_OFF:.2f} m ahead) pulling each way "
        f"(back: toward the robot, the force on the hand forward) "
        f"{PULL_TIMES} times at each force ({', '.join(f'{f:g}' for f in PULL_FORCES)} N), each "
        f"built at {PULL_RATE_N_S:g} N/s and held {PULL_HOLD_S:g} s, until it falls; policy "
        f"{_name_stand(npz)}")
  print("  'stood to': the hardest with no fall; 'feet held to': no step either; 'fell at': "
        "the first that toppled it; the hand's give and the trunk's tilt at 'stood to'")
  print(f"  {'way':14s} {'stood to N':>10s} {'feet held to':>13s} {'fell at':>8s} "
        f"{'hand gave cm':>13s} {'tilt deg':>9s}")
  out = {}
  for way, rows in zip(PULL_WAYS, ways):
    stood = [r for r in rows if not r["fell"] and not r.get("why")]
    stepped = next((i for i, r in enumerate(stood) if r["stepped"]), len(stood))
    held = stood[:stepped]
    fell = next((r["force"] for r in rows if r["fell"]), None)
    out[way] = {"rows": rows, "stood": stood[-1]["force"] if stood else 0.0,
                "held": held[-1]["force"] if held else 0.0, "fell": fell}
    best = stood[-1] if stood else None
    print(f"  {way:14s} {out[way]['stood']:10.0f} {out[way]['held']:13.0f} "
          f"{fell if fell else '-':>8} {100 * best['gave_m'] if best else 0:13.1f} "
          f"{best['tilt_deg'] if best else 0:9.1f}"
          + (f"  {rows[-1]['why']}" if rows[-1].get("why") else ""), flush=True)
  return out


def _name_stand(npz: str | None) -> str:
  return os.path.basename(npz) if npz else "served (humanoid_stand.npz)"


#: The `pull` scene (#544): the left hand held where a handle would be,
#: pulling each way at the verb's most (`steps.PULL_MOST_N`), as `--pulls`
#: stages it: a force on the hand, built at `PULL_RATE_N_S`, held
#: `PULL_HOLD_S` -- what the body holds; the verb on a locked bar, and what
#: it puts on it, is the `bar` scene's.
PULL_SCENE_N = PULL_MOST_N
PULL_SCENE = ("back", "forward", "left", "right", "up", "down")


def pull_scene(h: Humanoid, records: list | None = None):
  """The `pull` scene, once: the left hand to where a counter's handle
  would be, then pulling `PULL_SCENE_N` along each of `PULL_SCENE`'s ways
  (forward is a push) -- staged with the truth, a force on the hand the
  other way -- each built at the verb's rate and held, then let go; home."""
  side = "left"
  yield "a hand out where a handle would be"
  if not h.reach.reach(side, handle_goal()):
    yield "  no posture puts it in reach"
    return
  t0 = h.d.time
  while not h.reach.done(side) and h.d.time - t0 < 20.0:
    h.step()
    yield None
  rot_w, _ = true_stance(h.m, h.d)
  hand = h.m.body(f"{side}_hand").id
  rec = {"ways": []}
  for way in PULL_SCENE:
    w = np.array(PULL_WAYS[way], dtype=float)
    f = -rot_w @ (w / np.linalg.norm(w)) * PULL_SCENE_N
    yield f"{'pushing away' if way == 'forward' else f'pulling {way}'} at {PULL_SCENE_N:.0f} N"
    build = PULL_SCENE_N / PULL_RATE_N_S
    for k in range(int(round((build + PULL_HOLD_S) / h.m.opt.timestep))):
      h.d.xfrc_applied[hand, :3] = f * min(1.0, k * h.m.opt.timestep / build)
      h.step()
      if h.fallen():
        h.d.xfrc_applied[hand] = 0.0
        yield "FELL"
        return
      yield None
    h.d.xfrc_applied[hand] = 0.0
    rec["ways"].append(way)
    yield from wait(h, 1.0)
  yield "home"
  h.reach.retreat(side)
  t0 = h.d.time
  while h.reach.phase[side] != "rest" and h.d.time - t0 < 20.0:
    h.step()
    yield None
  yield from posture(h, 0.0, 0.0)
  rec["fell"] = h.fallen()
  yield "standing, never fallen" if not rec["fell"] else "FELL"
  if records is not None:
    records.append(rec)


#: `--handover` (#544): the walk's hand-over at a standstill, through the
#: body's own `step` (`HumanoidBody.step_routine`: its legs to the walking
#: policy, a walk, back to the standing policy) at counter set-outs: back
#: and ahead, a turn and back, and back out of a bow -- each trial's sizes
#: drawn off its own seed (m, rad; the set-outs stand the robot alike).
HANDOVER_STEPS = ("back", "ahead", "turn", "turn back", "out of a bow")
HANDOVER_M, HANDOVER_RAD, HANDOVER_BOW = (0.10, 0.30), (0.3, 1.0), (0.25, 0.6)
#: Each hand-over to the standing policy is watched this long, s.
HANDOVER_S = 2.0


def handover_trial(k: int, npz: str | None = None) -> dict:
  """At counter set-out k (`evaluation.counter`), each of `HANDOVER_STEPS`
  in turn through the body's own `step`, the last from a bow (`gaze_
  routine`); each hand-over from the walker to the standing policy watched
  `HANDOVER_S`: the trunk's worst tilt off upright (its asked posture is the
  stand), how far the soles moved and the trunk's fastest across the floor
  -- off the truth, our grading -- and the falls its own watch took."""
  from pluggybot.evaluation import counter as ct
  model, data, body = ct.world(ct.set_out(k), StandPolicy(npz) if npz else _POLICY)
  mi = body.mission
  root, soles = model.body(ROBOT_ROOT).id, [model.geom(f"{s}_sole").id for s in hm.SIDES]
  # the robot is put in after the house: its own joint, by name
  dof = int(model.jnt_dofadr[model.body_jntadr[root]])
  rng = np.random.default_rng(k)
  far, turn, bow = (float(rng.uniform(*r)) for r in (HANDOVER_M, HANDOVER_RAD, HANDOVER_BOW))
  sizes = {"back": (-far, 0.0, 0.0), "ahead": (far, 0.0, 0.0), "turn": (0.0, 0.0, turn),
           "turn back": (0.0, 0.0, -turn), "out of a bow": (-far, 0.0, 0.0)}
  state = {"legs": mi.legs, "t": None, "feet": None, "tilt": 0.0, "moved": 0.0, "speed": 0.0}
  rows: list[dict] = []

  def watch():
    if mi.legs == "stand" and state["legs"] != "stand":
      state.update(t=float(data.time), feet=data.geom_xpos[soles][:, :2].copy(), tilt=0.0,
                   moved=0.0, speed=0.0)
    state["legs"] = mi.legs
    if state["t"] is None:
      return
    tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(data.xmat[root][8])))))
    state["tilt"] = max(state["tilt"], tilt)
    state["moved"] = max(state["moved"], float(np.linalg.norm(
      data.geom_xpos[soles][:, :2] - state["feet"], axis=1).max()))
    state["speed"] = max(state["speed"], float(np.linalg.norm(data.qvel[dof:dof + 2])))
    if float(data.time) - state["t"] >= HANDOVER_S:
      rows[-1].update(tilt_deg=state["tilt"], moved_m=state["moved"], speed_ms=state["speed"])
      state["t"] = None
  body.step_hooks.append(watch)
  try:
    body.run(body.hold_routine(1.0))
    for label in HANDOVER_STEPS:
      if label == "out of a bow":
        body.run(body.gaze_routine(bow))
      falls = mi.falls
      rows.append({"step": label, "size": sizes[label]})
      rep = body.run(body.step_routine(*sizes[label]))
      body.run(body.hold_routine(HANDOVER_S))
      rows[-1].update(why=rep["why"], fell=mi.falls != falls)
  finally:
    body.close()
  return {"k": k, "rows": rows}


def _handover_trial(job):
  return handover_trial(*job)


def handover_table(npz: str | None = None, n: int = 8, jobs: int = 4) -> list[dict]:
  """--handover: `HANDOVER_STEPS` at counter set-outs 0..n-1."""
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    got = pool.map(_handover_trial, [(k, npz) for k in range(n)])
  print(f"THE WALK'S HAND-OVER AT A STANDSTILL, through the body's own `step` at counter "
        f"set-outs 0-{n - 1} (steps {HANDOVER_M[0]:g}-{HANDOVER_M[1]:g} m, turns "
        f"{HANDOVER_RAD[0]:g}-{HANDOVER_RAD[1]:g} rad, bows {HANDOVER_BOW[0]:g}-{HANDOVER_BOW[1]:g} "
        f"rad, drawn); policy {_name_stand(npz)}: each hand-over back to the standing policy "
        f"watched {HANDOVER_S:g} s (our grading, off the truth)")
  print(f"  {'step':24s} {'ok':>5s} {'falls':>6s} {'worst tilt deg':>15s} {'soles moved mm':>15s} "
        f"{'trunk m/s':>10s}")
  for i, label in enumerate(HANDOVER_STEPS):
    rs = [g["rows"][i] for g in got if len(g["rows"]) > i]
    ok = sum(r.get("why") == "stepped" for r in rs)
    falls = sum(bool(r.get("fell")) for r in rs)
    tilt = [r["tilt_deg"] for r in rs if "tilt_deg" in r]
    moved = [1000 * r["moved_m"] for r in rs if "moved_m" in r]
    speed = [r["speed_ms"] for r in rs if "speed_ms" in r]
    print(f"  {label:24s} {ok:>2d}/{len(rs):<2d} {falls:6d} {_span(tilt):>15s} {_span(moved, '{:.0f}'):>15s} "
          f"{_span(speed, '{:.2f}'):>10s}")
  return got


#: `--bar` (#547): the force move against something that holds the hand --
#: the left hand shut on the locked drawer's handle and `pull`ed each of
#: `PULL_WAYS` at `BAR_FORCES` for `BAR_DISTANCE_M` at its own speed, a
#: fresh robot a trial. Nothing moves, so every newton on the bar is the
#: move's to keep: its peak, at an instant and over `BAR_MEAN_S`, is ours
#: (`handle_force`, off the contacts), beside the hardest the hand read; the
#: postures the body took with the hand on the bar; how and when the pull
#: ended -- or not, in `BAR_BUDGET_S`; a fall; how far the grip slid.
BAR_FORCES, BAR_DISTANCE_M, BAR_MEAN_S, BAR_BUDGET_S = (10.0, 20.0), 0.05, 0.1, 6.0


def _bar_world(spec: mujoco.MjSpec) -> None:
  counter_scenery(spec, locked=True, at=(0.0, 0.0), yaw=0.0)


def _grip_bar(h: Humanoid, side: str = "left"):
  """The hand to the locked handle's bar, open, then shut on it; the
  gripper's own word on whether it holds."""
  g = h.reach.grippers[side]
  g.open()
  if not h.reach.reach(side, handle_goal()):
    return False
  yield from _until(h, lambda: h.reach.done(side), 20.0)
  g.shut()
  yield from wait(h, 0.6)
  return g.held()


def _pull_bar(h: Humanoid, way: str, force: float, side: str = "left"):
  """One `pull` on the bar along `PULL_WAYS[way]`, watched to its end (at
  most `BAR_BUDGET_S`): what `bar_trial` reports."""
  w = np.array(PULL_WAYS[way], dtype=float)
  w /= np.linalg.norm(w)
  pads = [h.m.geom(f"{side}_{j}_pad").id for j in hm.JAWS]
  at0 = np.mean(h.d.geom_xpos[pads], axis=0)
  postures = [tuple(float(v) for v in h.reach.posture)]
  n = max(1, int(round(BAR_MEAN_S / h.m.opt.timestep)))
  hist, read, t0 = [], 0.0, h.d.time
  h.reach.pull(side, w, force, BAR_DISTANCE_M)
  while h.reach.report[side] is None and h.d.time - t0 < BAR_BUDGET_S and not h.fallen():
    h.step()
    hist.append(float(handle_force(h, side) @ w))
    st = h.reach.pulls[side]
    read = max(read, st["peak"]) if st else read
    p = tuple(float(v) for v in h.reach.posture)
    if p != postures[-1]:
      postures.append(p)
    yield None
  rep = h.reach.report[side] or {}
  means = np.convolve(hist, np.ones(n) / n, mode="valid") if len(hist) >= n else [0.0]
  return {"way": way, "force": force, "peak": max(hist, default=0.0),
          "peakMean": float(np.max(means)), "read": float(rep.get("force", read)),
          "postures": postures[1:], "why": rep.get("why", "no end"),
          "seconds": round(h.d.time - t0, 2), "fell": bool(h.fallen()),
          "slidMm": 1000 * float(np.linalg.norm(np.mean(h.d.geom_xpos[pads], axis=0) - at0))}


def _run(gen):
  """A scene's generator driven to its end, for its return value."""
  try:
    while True:
      next(gen)
  except StopIteration as stop:
    return stop.value


def bar_trial(way: str, force: float, npz: str | None = None, oracle: bool = False) -> dict:
  """A fresh robot's left hand on the locked bar, `pull`ed along
  `PULL_WAYS[way]` at `force` N (`--bar`); `oracle`, the move closed on
  the bar's true force (`closed_on_the_bar`)."""
  h = floor_humanoid(_bar_world, policy=StandPolicy(npz) if npz else _POLICY)
  if oracle:
    closed_on_the_bar(h)
  if not _run(_grip_bar(h)):
    return {"way": way, "force": force, "why": "no grip on the bar"}
  return _run(_pull_bar(h, way, force))


def closed_on_the_bar(h: Humanoid, side: str = "left") -> None:
  """The hand's reading replaced, every command, by the force the pads
  truly put on the bar (`handle_force`, ours): the force loop's own floor,
  what it does with a perfect reading (`--bar --oracle`, #547) -- never the
  robot's."""
  feel = h.reach._feel

  def oracle() -> None:
    feel()
    h.reach.force[side] = handle_force(h, side)
  h.reach._feel = oracle


def _bar_trial(job):
  return bar_trial(*job)


def bar_table(npz: str | None = None, jobs: int = 4, oracle: bool = False) -> list[dict]:
  """--bar: each way at each of `BAR_FORCES`, a process a trial."""
  import multiprocessing as mp
  trials = [(w, f, npz, oracle) for f in BAR_FORCES for w in PULL_WAYS]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_bar_trial, trials)
  print(f"THE FORCE MOVE ON THE LOCKED BAR: the left hand shut on the welded drawer's handle "
        f"({HANDLE_Z:.2f} m up, {COUNTER_FRONT - 0.02 - HANDLE_OFF:.2f} m ahead), `pull`ed "
        f"{100 * BAR_DISTANCE_M:.0f} cm each way at {', '.join(f'{f:g}' for f in BAR_FORCES)} N, "
        f"watched {BAR_BUDGET_S:g} s at most; policy {_name_stand(npz)}; the move closed on "
        + ("the bar's TRUE force (an oracle, ours: the loop's own floor)" if oracle
           else "the hand's reading"))
  print(f"  the force on the bar is the pads' (ours): its peak at an instant and over "
        f"{BAR_MEAN_S:g} s; 'read' the hardest the hand read; 'postures' those taken with the "
        f"hand on the bar")
  print(f"  {'way':14s} {'ask N':>5s} {'peak N':>7s} {'peak .1s':>8s} {'read N':>6s} "
        f"{'postures':>8s} {'ended':12s} {'s':>5s} {'slid mm':>7s}")
  for r in rows:
    if r.get("peak") is None:
      print(f"  {r['way']:14s} {r['force']:5.0f}  {r['why']}")
      continue
    print(f"  {r['way']:14s} {r['force']:5.0f} {r['peak']:7.1f} {r['peakMean']:8.1f} "
          f"{r['read']:6.1f} {len(r['postures']):8d} {r['why']:12s} {r['seconds']:5.1f} "
          f"{r['slidMm']:7.1f}" + ("  FELL" if r["fell"] else ""), flush=True)
  for f in BAR_FORCES:
    flown = [r for r in rows if r["force"] == f and r.get("peak") is not None]
    if flown:
      worst = max(flown, key=lambda r: r["peakMean"])
      print(f"  asked {f:g} N: the hardest {BAR_MEAN_S:g} s on the bar {worst['peakMean']:.1f} N "
            f"({worst['peakMean'] / f:.2f} times, {worst['way']}); over a quarter more on "
            f"{sum(r['peakMean'] > 1.25 * f for r in flown)} ways of {len(flown)}")
  return rows


#: `--feel` (#547): what the hand's readings make of a force we are sure of
#: -- the left hand held by the reach where the bar would be, no bar, and
#: `FEEL_N` staged at the grip (no moment there) so the hand pulls along
#: each of `PULL_WAYS`, built over `FEEL_BUILD_S` and held `FEEL_HOLD_S`:
#: its arm's own reading (`Reach.arm_force`) and the whole body's
#: (`Reach.force`, `SelfModel.felt`), each as the reach computed it as it
#: steered, over the hold's last `FEEL_READ_S` -- along the way and across
#: it -- and the worst either was along the way from the build's start;
#: standing still, and SWAYING: the posture asked rocking, the trunk's
#: pitch `SWAY_PITCH` rad either way at `SWAY_PITCH_HZ` and the pelvis down
#: `SWAY_DROP` m and up again at `SWAY_DROP_HZ`, the hand held where it is.
FEEL_N, FEEL_BUILD_S, FEEL_HOLD_S, FEEL_READ_S = 20.0, 0.25, 1.5, 1.0
SWAY_PITCH, SWAY_PITCH_HZ, SWAY_DROP, SWAY_DROP_HZ = 0.1, 0.4, 0.04, 0.25
FEELS = ("arm", "body")


def feel_trial(way: str, npz: str | None = None, sway: bool = False) -> dict:
  h = floor_humanoid(policy=StandPolicy(npz) if npz else _POLICY)
  side = "left"
  out = {"way": way, "sway": sway}
  if not h.reach.reach(side, handle_goal()):
    return {**out, "why": "the bar's place is out of reach"}
  _drain(_until(h, lambda: h.reach.done(side), 20.0))
  _drain(wait(h, 0.5))
  w = np.array(PULL_WAYS[way], dtype=float)
  w /= np.linalg.norm(w)
  rot_w, _ = true_stance(h.m, h.d)
  hand, site = h.m.body(f"{side}_hand").id, h.m.site(f"{side}_grip").id
  dt = h.m.opt.timestep
  build, hold = int(round(FEEL_BUILD_S / dt)), int(round(FEEL_HOLD_S / dt))
  late = build + hold - int(round(FEEL_READ_S / dt))
  reads, worst = {k: [] for k in FEELS}, {k: 0.0 for k in FEELS}
  for k in range(build + hold):
    share = min(1.0, k / build)
    f = -rot_w @ w * FEEL_N * share
    h.d.xfrc_applied[hand, :3] = f
    h.d.xfrc_applied[hand, 3:] = np.cross(h.d.site_xpos[site] - h.d.xipos[hand], f)
    if sway:
      t = k * dt
      h.reach.hold_posture(-SWAY_DROP / 2 * (1.0 - math.cos(2 * math.pi * SWAY_DROP_HZ * t)),
                           SWAY_PITCH * math.sin(2 * math.pi * SWAY_PITCH_HZ * t))
    h.step()
    if h.fallen():
      return {**out, "why": "fell"}
    if (h.reach.steps - 1) % h.reach.every:
      continue
    for key, felt in zip(FEELS, (h.reach.arm_force[side], h.reach.force[side])):
      worst[key] = max(worst[key], abs(float(felt @ w) - FEEL_N * share))
      if k >= late:
        reads[key].append(np.array(felt))
  for key in FEELS:
    along = np.array(reads[key]) @ w
    out[key] = {"read": float(np.mean(along)), "sd": float(np.std(along)),
                "across": float(np.linalg.norm(np.mean(reads[key], axis=0) - np.mean(along) * w)),
                "worst": worst[key]}
  return out


def _feel_trial(job):
  return feel_trial(*job)


def feel_table(npz: str | None = None, jobs: int = 4) -> list[dict]:
  """--feel: each way, still and swaying, a process a trial."""
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_feel_trial, [(w, npz, s) for s in (False, True) for w in PULL_WAYS])
  print(f"WHAT THE HAND READS of {FEEL_N:g} N staged at the grip, the left hand held by the reach "
        f"where the locked handle's bar is, no bar: its arm's own reading and the whole body's, "
        f"along the way and across it over the last {FEEL_READ_S:g} s of {FEEL_HOLD_S:g}, and "
        f"the worst along it from the build's start ({FEEL_BUILD_S:g} s); swaying, the trunk's "
        f"pitch +-{SWAY_PITCH:g} rad at {SWAY_PITCH_HZ:g} Hz and the pelvis {100 * SWAY_DROP:g} cm "
        f"down and up at {SWAY_DROP_HZ:g} Hz; policy {_name_stand(npz)}")
  cell = f" {'along':>5s} {'sd':>4s} {'acr':>4s} {'worst':>5s}"
  print(f"  {'':14s}" + "".join(f" | {s + ', ' + k:<22s}" for s in ("still", "swaying")
                                for k in FEELS))
  print(f"  {'way':14s}" + f" |{cell}" * 4)
  for way in PULL_WAYS:
    cells = []
    for sway in (False, True):
      r = next(r_ for r_ in rows if r_["way"] == way and r_["sway"] == sway)
      for key in FEELS:
        c = r.get(key)
        cells.append(f" | {r['why']:22s}" if c is None else
                     f" | {c['read']:5.1f} {c['sd']:4.1f} {c['across']:4.1f} {c['worst']:5.1f}")
    print(f"  {way:14s}" + "".join(cells), flush=True)
  return rows


#: The `feel` scene (#547): the left hand held out where the locked bar
#: is, no bar, `FEEL_N` staged at its grip along each of `FEEL_SCENE`'s
#: ways -- one its arm reads, then the three a straight arm is blindest to
#: (after them, the arm straightened, back read 5 N of 20) -- each
#: built over `FEEL_BUILD_S` and held `FEEL_HOLD_S`: what its arm read of it
#: and what the whole body read, over the hold's second half, said once it
#: is let go. Staged with the truth.
FEEL_SCENE = ("back", "forward", "forward-down", "up")


def feel_scene(h: Humanoid, records: list | None = None):
  """The `feel` scene, once: the hand out, then each of `FEEL_SCENE`'s ways
  staged and read (above); home."""
  side = "left"
  yield "a hand out where the bar would be"
  if not h.reach.reach(side, handle_goal()):
    yield "  no posture puts it in reach"
    return
  yield from _until(h, lambda: h.reach.done(side), 20.0)
  yield from wait(h, 0.5)
  rot_w, _ = true_stance(h.m, h.d)
  hand, site = h.m.body(f"{side}_hand").id, h.m.site(f"{side}_grip").id
  dt = h.m.opt.timestep
  build, hold = int(round(FEEL_BUILD_S / dt)), int(round(FEEL_HOLD_S / dt))
  rec = {"ways": []}
  for way in FEEL_SCENE:
    w = np.array(PULL_WAYS[way], dtype=float)
    w /= np.linalg.norm(w)
    yield f"  {FEEL_N:.0f} N {way} at the grip"
    reads = {k: [] for k in FEELS}
    for k in range(build + hold):
      f = -rot_w @ w * FEEL_N * min(1.0, k / build)
      h.d.xfrc_applied[hand, :3] = f
      h.d.xfrc_applied[hand, 3:] = np.cross(h.d.site_xpos[site] - h.d.xipos[hand], f)
      h.step()
      if h.fallen():
        h.d.xfrc_applied[hand] = 0.0
        yield "FELL"
        return
      if k >= build + hold // 2:
        for key, felt in zip(FEELS, (h.reach.arm_force[side], h.reach.force[side])):
          reads[key].append(float(felt @ w))
      yield None
    h.d.xfrc_applied[hand] = 0.0
    row = {"way": way, **{k: float(np.mean(v)) for k, v in reads.items()}}
    rec["ways"].append(row)
    yield (f"{FEEL_N:.0f} N {way}: the arm read {row['arm']:.0f}, the whole body "
           f"{row['body']:.0f}")
    yield from wait(h, 1.0)
  yield "home"
  h.reach.retreat(side)
  yield from _until(h, lambda: h.reach.phase[side] == "rest", 20.0)
  yield from posture(h, 0.0, 0.0)
  rec["fell"] = h.fallen()
  yield "standing, never fallen" if not rec["fell"] else "FELL"
  if records is not None:
    records.append(rec)


#: The `bar` scene (#547): the force move on the locked bar at `pull`'s
#: most, each of `BAR_SCENE`'s ways from a fresh grip.
BAR_SCENE = ("back", "up", "forward", "down")


def bar_scene(h: Humanoid, records: list | None = None):
  """The `bar` scene, once: for each of `BAR_SCENE`'s ways, the left hand
  shut on the locked handle's bar and `pull`ed `PULL_MOST_N` along it, how
  it ended and the most the bar took said; let go and home."""
  side = "left"
  rec = {"ways": []}
  for way in BAR_SCENE:
    yield "gripping the locked bar"
    held = yield from _grip_bar(h, side)
    if not held:
      yield "  NOT HELD"
      break
    yield f"{'pushing' if way in ('forward', 'down') else 'pulling'} {way} at {PULL_MOST_N:.0f} N"
    row = yield from _pull_bar(h, way, PULL_MOST_N, side)
    rec["ways"].append(row)
    yield (f"  {row['why']} in {row['seconds']:.1f} s: the bar took {row['peakMean']:.0f} N at "
           f"most, the hand read {row['read']:.0f}; {len(row['postures'])} postures taken")
    if row["fell"]:
      yield "FELL"
      break
    h.reach.grippers[side].open()
    yield from wait(h, 0.4)
    h.reach.retreat(side)
    yield from _until(h, lambda: h.reach.phase[side] == "rest", 20.0)
    yield from posture(h, 0.0, 0.0)
  rec["fell"] = h.fallen()
  yield "standing, never fallen" if not rec["fell"] else "FELL"
  if records is not None:
    records.append(rec)


def _until(h: Humanoid, done, budget_s: float):
  t0 = h.d.time
  while not done() and h.d.time - t0 < budget_s and not h.fallen():
    h.step()
    yield None


# ---- stage 3: the scene `demo`, through `Body` ----------------------------------------
#
# #466's chest on a counter in the storeroom (`evaluation.counter`, set-out
# `DEMO`), and the humanoid behind `Body` (`humanoid.body.HumanoidBody`):
# it finds the knob by its tag in its own colour imager, reaches it, shuts
# its gripper and judges the grip by the gripper's own reading -- every act
# a Body member. What the terminal says of the truth (where the knob truly
# is, whether the pads truly hold it) is OUR grading.

def drive(body, routine):
  """A body's routine stepped one physics step at a time (its stepper,
  as `tick.run` steps it), yielding between: what it returned."""
  try:
    command = next(routine)
    while True:
      body.stepper.step(command)
      yield None
      command = routine.send(None)
  except StopIteration as done:
    return done.value


def demo_scene(body, model, data, records: list | None = None):
  """The `demo` scene, once: yields a line when a part begins or ends."""
  from pluggybot.evaluation import counter as ct
  from pluggybot.activity.chest import KNOB
  from pluggybot.humanoid.body import _rz
  rec = {}
  if records is not None:
    records.append(rec)
  yield "looking for the chest's knob by its tag"
  yield from drive(body, body.hold_routine(1.0))
  seen = body.spot(ch_tag())
  truly = ct.knob_world(model, data)
  if seen is None:
    yield "  its tag does not decode from here"
    return
  # The knob is a cube: its middle is half a cube in from the face read.
  knob = np.array([seen["x"], seen["y"], seen["z"]]) + KNOB * np.array(seen["normal"])
  rec["seen_off_m"] = float(np.linalg.norm(knob - truly))
  yield (f"  the knob, by its tag, {seen['range']:.2f} m off: {100 * rec['seen_off_m']:.1f} cm "
         "from where it truly is")
  side = "left" if body.to_stance(knob)[1] > 0 else "right"
  yield f"reaching it with the {side} hand, the gripper open"
  yield from drive(body, body.grip_routine(side, False))
  # The first tilt from which a posture it holds puts the knob in reach.
  for tilt in KNOB_TILTS:
    rot = _rz(body.stance[2]) @ knob_grip(tilt)
    rec["reached"] = yield from drive(body, body.reach_routine(side, knob, rot, patience=20.0))
    if rec["reached"] or body.reach.phase[side] != "rest":
      break
  if not rec["reached"]:
    yield "  out of its reach"
    return
  yield f"  there, its fingers tilted {math.degrees(tilt):.0f} deg down"
  yield "shutting the gripper on the knob"
  rec["held"] = yield from drive(body, body.grip_routine(side, True))
  rec["pads"] = ct.pads_hold(model, data, side)
  yield (f"  {'HELD' if rec['held'] else 'NOT HELD'} by its own reading; the pads truly "
         f"{'on' if rec['pads'] else 'off'} the knob")
  yield "holding it"
  yield from drive(body, body.hold_routine(2.0))
  rec["still_pads"] = ct.pads_hold(model, data, side)
  yield "letting go"
  yield from drive(body, body.grip_routine(side, False))
  rec["released"] = yield from drive(body, body.release_routine(side))
  yield f"  its arm {rec['released']}"
  yield from drive(body, body.grip_routine(side, True))
  rec["fell"] = bool(data.xpos[body.root][2] < FALLEN_M)


#: The knob's grip: fingers forward and tilted 25 deg down onto it, the jaws
#: closing ACROSS it. ⚠ A drop handle's knob hangs nearly under its pin,
#: its drop bar and stem behind and above it: jaws closing up and down
#: brought a pad down onto them, swung the handle 13 deg and stalled 79 mm
#: open on them -- HELD by the gripper's reading, the pads truly off the
#: knob. Level, the plan crouched and the elbow met the counter; tilted
#: down, it reaches from the stand.
KNOB_TILTS = tuple(math.radians(d) for d in (25.0, 15.0, 35.0, 45.0))


def knob_grip(tilt: float) -> np.ndarray:
  return frame((math.cos(tilt), 0.0, -math.sin(tilt)), (0.0, 1.0, 0.0))


def ch_tag() -> int:
  from pluggybot.perception.tags import CHEST_TAG_IDS
  return CHEST_TAG_IDS[0]


def demo_world(k: int | None = None):
  """Set-out k's world (`evaluation.counter`). ⚠ In the house's own light:
  a headlight lights the robot's cameras too, and drawn brighter, as the
  other scenes are, it moved where the knob's tag decoded by 3 cm."""
  from pluggybot.evaluation import counter as ct
  so = ct.set_out(ct.DEMO if k is None else k)
  return ct.world(so, _POLICY)


def _demo_flight(k: int) -> dict:
  model, data, body = demo_world(k)
  records: list = []
  try:
    for _ in demo_scene(body, model, data, records):
      pass
  finally:
    body.close()
  return {"k": k, **records[0]}


def demo_table(n: int = 12, jobs: int = 4) -> None:
  """--demo-table: the demo's set-outs 0..n-1 (`evaluation.counter.set_out`),
  each flown through `Body` in a process of its own: where the tag put the
  knob, whether the hand got there, the gripper's word and the truth."""
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_demo_flight, range(n))
  print(f"THE DEMO, set-outs 0-{n - 1}: the knob by its tag (cm off the truth), reached, "
        "the gripper's own word, the pads truly on the knob at the grip and 2 s later:")
  for r in rows:
    seen = f"{100 * r['seen_off_m']:4.1f}" if "seen_off_m" in r else "  --"
    word = "" if "held" not in r else ("HELD" if r["held"] else "NOT HELD")
    truth = "" if "pads" not in r else (f"pads {'on' if r['pads'] else 'off'}, "
                                        f"{'on' if r['still_pads'] else 'off'} 2 s later")
    print(f"  {r['k']:2d}: {seen} cm  {'reached' if r.get('reached') else 'NOT reached':11s} "
          f"{word:9s} {truth}{'  FELL' if r.get('fell') else ''}")
  held = sum(bool(r.get("held") and r.get("pads") and r.get("still_pads")) for r in rows)
  agreed = sum(bool(r["held"]) == bool(r["pads"]) for r in rows if "held" in r)
  print(f"held, by its word and truly, through the hold: {held} of {n}; its word agreed "
        f"with the truth {agreed} times in {sum('held' in r for r in rows)}")


def view_demo() -> None:
  from mujoco import viewer as mj_viewer
  model, data, body = demo_world()
  print("the `demo` scene: #466's chest on a counter; the humanoid finds its knob by its "
        "tag, reaches it and grips it, through Body. Close the window to quit.")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body.root
    viewer.cam.distance, viewer.cam.azimuth = CAM_DISTANCE, CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION
    while viewer.is_running():
      body.start_at(*body.true_pose()[:2], 0.0)
      wall = time.time() - data.time
      for line in demo_scene(body, model, data):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)


def filmstrip_demo(out: str) -> None:
  from PIL import Image, ImageDraw
  model, data, body = demo_world()
  r = mujoco.Renderer(model, 300, 400)
  tiles, pending, at = [], None, 0.0

  def shoot(label):
    # From the robot's right (the counter hides the hand from the front),
    # the headlight brighter for this tile alone: the robot's own camera
    # renders in the house's light (`demo_world`).
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[body.root] + [0.25, 0.0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 90.0, -15.0
    light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.35, 0.55
    r.update_scene(data, cam)
    img = Image.fromarray(r.render())
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
    ImageDraw.Draw(img).text((8, 8), label[:60], fill=(255, 255, 255))
    tiles.append(img)
  for line in demo_scene(body, model, data):
    if line and not line.startswith("  "):
      if pending is not None:
        shoot(pending)
      pending, at = line, data.time + 0.9
    elif line:
      print(line)
    if pending is not None and data.time >= at:
      shoot(pending)
      pending = None
  if pending is not None:
    shoot(pending)
  r.close()
  body.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)),
                    (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 4: the scene `walk`, through `Body` ----------------------------------------
#
# The humanoid walks across the house and back, steered by the Navigator
# (`humanoid.mission`): its map off its head's LIDAR and the D435's points
# under it, its pose off its legs' odometry and IMU, matched to the map. The
# house's walls are a room's 2.4 m since stage 5 (the rover's 1.2 m were
# under the humanoid's LIDAR, which saw over every one).

#: The walk: from the living room, through the hall, into the kitchen and
#: back (`home.world`'s rooms; the doors are the planner's to find).
WALK_FROM, WALK_TO = (0.5, 0.5), (-8.5, 4.0)


def house_world(at=WALK_FROM, yaw: float = math.pi, walk: StandPolicy | None = None):
  """The house with the humanoid standing at `at`: the model, its data and
  its body (`body.body_for`, its walker `walk` -- the served one by
  default), its map's bounds the house's."""
  import json
  from pluggybot.body import body_for
  spec = mujoco.MjSpec.from_file(HOME_XML)
  frame = spec.worldbody.add_frame()
  frame.pos = [at[0], at[1], 0.0]
  frame.quat = _yaw_quat(yaw)
  spec.attach(hm.attachable(), prefix="", frame=frame)
  model = spec.compile()
  data = mujoco.MjData(model)
  _pose(model, data, hm.stand_qpos())
  meta = json.load(open(HOME_XML.replace(".xml", ".meta.json")))
  b = meta["gridBounds"]
  bounds = (b[0], b[1], b[2], b[3]) if isinstance(b, list) else (b["xMin"], b["yMin"], b["xMax"], b["yMax"])
  body = body_for(model, data, grid_bounds=bounds, walk=walk)
  return model, data, body


def walk_scene(body, model, data, records: list | None = None):
  """The `walk` scene, once: there and back, saying how each leg ended."""
  for name, (x, y) in (("into the kitchen", WALK_TO), ("back to the living room", WALK_FROM)):
    yield f"walking {name}"
    t0 = data.time
    arrived = yield from drive(body, body.go_to_routine(x, y, timeout=120.0))
    bx, by, _ = body.pose
    tx, ty, _ = body.true_pose()
    rec = {"leg": name, "arrived": arrived, "seconds": data.time - t0,
           "believed_off_m": math.hypot(bx - x, by - y),
           "truly_off_m": math.hypot(tx - x, ty - y),
           "belief_error_m": math.hypot(bx - tx, by - ty),
           "fell": bool(data.xpos[body.root][2] < FALLEN_M)}
    if records is not None:
      records.append(rec)
    why = "" if arrived else f" ({body.gave_up(body.last_drive or {})})"
    yield (f"  {'there' if arrived else 'did not arrive'} in {rec['seconds']:.0f} s{why}; "
           f"{100 * rec['truly_off_m']:.0f} cm from the goal truly, its belief "
           f"{100 * rec['belief_error_m']:.0f} cm off the truth")
    if rec["fell"]:
      yield "FELL"
      return


def view_walk() -> None:
  from mujoco import viewer as mj_viewer
  model, data, body = house_world()
  print("the `walk` scene: the humanoid walks from the living room to the kitchen and "
        "back, steered by its map. Close the window to quit.")
  with mj_viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body.root
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.0, CAM_AZIMUTH, -25.0
    while viewer.is_running():
      wall = time.time() - data.time
      for line in walk_scene(body, model, data):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)


def filmstrip_walk(out: str, every_s: float = 6.0) -> None:
  """The walk's filmstrip: a tile every `every_s`, from above and behind."""
  from PIL import Image, ImageDraw
  model, data, body = house_world()
  r = mujoco.Renderer(model, 300, 400)
  tiles, next_t, label = [], 0.0, ""
  for line in walk_scene(body, model, data):
    if line and not line.startswith("  "):
      label = line
    elif line:
      print(line)
    if data.time >= next_t:
      next_t += every_s
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[body.root]
      cam.distance, cam.azimuth, cam.elevation = 4.5, math.degrees(body.true_pose()[2]) + 180, -50
      # Brighter for this tile alone: the robot's cameras render in the
      # house's light.
      light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
      model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.4, 0.6
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
      ImageDraw.Draw(img).text((8, 8), f"{data.time:.0f} s: {label}"[:60], fill=(255, 255, 255))
      tiles.append(img)
  r.close()
  body.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 4: rough ground -----------------------------------------------------------
# A walking policy over what the rough task's terrains are at their hardest
# (`training/humanoid_train/walk.py`, `TERRAINS`), each course a world of its
# own, walked straight ahead off the robot's own senses: a doorstep twice,
# paving, a slope up and down, a grid of blocks. Graded by us: whether it
# fell, and how far it got. The served walker is the flat task's, blind; the
# walker that sees the ground (`SEEING_NPZ`) walks each on its own map.

from pluggybot.humanoid.walk import SEEING_NPZ, WALK_NPZ  # noqa: E402
#: Each course is walked at this speed, m/s, for this long, s.
ROUGH_V, ROUGH_S = 0.6, 12.0
#: Where each course begins ahead of the robot, m.
ROUGH_AT = 1.0
COURSES = ("flat", "doorstep", "paving", "slope", "blocks")
#: The courses, m: a doorstep's rise (the task's risers reach 8 cm); paving's
#: tallest and its side; the slope's rise over run (the task's plates tilt
#: to 10 deg); the blocks' tallest (the task's hardest grid is +-8 cm about
#: its platform) and their side.
DOORSTEP_M, PAVING_M, PAVING_W, SLOPE, BLOCK_M, BLOCK_W = 0.05, 0.04, 0.3, 0.17, 0.08, 0.45


def _lay(wb, kind: str, x0: float, rng, length: float = 7.2) -> float:
  """Course `kind` laid along +x from `x0` (a grid `length` long): how
  long it is, m."""
  grey = [0.55, 0.5, 0.45, 1.0]

  def box(name, x_from, x_to, top):
    wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, rgba=grey,
                size=[(x_to - x_from) / 2, 1.5, top / 2], pos=[(x_from + x_to) / 2, 0.0, top / 2])

  if kind == "doorstep":
    for k, x in enumerate((x0, x0 + 2.5)):
      box(f"step{k}", x, x + 0.6, DOORSTEP_M)
    return 3.1
  if kind in ("paving", "blocks"):
    side, tall = (PAVING_W, PAVING_M) if kind == "paving" else (BLOCK_W, BLOCK_M)
    n, across = int(length / side + 1e-9), int(3.0 / side)
    for i in range(n):
      for j in range(across):
        top = rng.uniform(0.0, tall)
        x, y = x0 + i * side, (j - across / 2) * side
        wb.add_geom(name=f"{kind}{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX, rgba=grey,
                    size=[side / 2, side / 2, top / 2 + 0.001],
                    pos=[x + side / 2, y + side / 2, top / 2])
    return n * side
  if kind == "slope":
    th, half = math.atan(SLOPE), 0.05
    for name, x, sign in (("up", x0, 1), ("down", x0 + 2.0, -1)):
      wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, rgba=grey,
                  quat=[math.cos(th / 2), 0.0, -sign * math.sin(th / 2), 0.0],
                  size=[0.5 / math.cos(th), 1.5, half],
                  pos=[x + 0.5 + sign * half * math.sin(th), 0.0, SLOPE / 2 - half * math.cos(th)])
    box("top", x0 + 1.0, x0 + 2.0, SLOPE)
    return 3.0
  return 0.0


def course_world(kind: str, seed: int = 0):
  """The bare robot at its stand on a floor with course `kind` ahead of it,
  its random parts drawn from `seed` (`"all"`: every course, a shorter grid
  each, end to end): the model and its data."""
  rng = np.random.default_rng(seed)
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  if kind == "all":
    x = ROUGH_AT
    for k in COURSES[1:]:
      x += _lay(spec.worldbody, k, x, rng, length=3.0) + 1.5
  else:
    _lay(spec.worldbody, kind, ROUGH_AT, rng)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  return model, data


class Blind:
  """A walker shown no ground on a bare body, and (`odometry`) its legs'
  odometry fed as the mission feeds it, every 10 ms: `step()` once a
  physics step, in place of the walker's and the arms'."""

  def __init__(self, m, d, policy: StandPolicy | None = None, odometry: bool = False):
    from pluggybot.humanoid.arms import SelfModel
    from pluggybot.humanoid.odometry import StanceOdometry
    from pluggybot.humanoid.walk import WalkDriver
    self.m, self.d = m, d
    self.root = m.body(ROBOT_ROOT).id
    self.stand = StandDriver(m, d)
    self.reach = Reach(m, d, self.stand)
    self.walk = WalkDriver(m, d, policy=policy, trunk=self.stand.trunk, joints=self.stand.joints)
    self.odo = (StanceOdometry(SelfModel(), self.stand.trunk.attitude,
                               float(d.xpos[self.root][0]), float(d.xpos[self.root][1]))
                if odometry else None)
    self.every = max(1, round(0.01 / m.opt.timestep))
    self.n = 0

  def step(self) -> None:
    self.walk.step()
    self.reach.step()
    mujoco.mj_step(self.m, self.d)
    if self.odo is not None and self.n % self.every == 0:
      q, _, _ = self.stand.joints.read(self.d)
      self.odo.step(q, self.stand.trunk.gravity())
    self.n += 1


def _walker(m, d, npz: str | None, odometry: bool = False):
  """The walking policy at `npz` (the served one by default) on a bare body:
  one that sees the ground shown its own map, as it would be served
  (`Seeing`, its odometry always beside it); a blind one alone (`Blind`)."""
  policy = StandPolicy(npz or str(WALK_NPZ))
  if "height_scan" in policy.observation_names:
    return Seeing(m, d, scan="map", npz=npz or str(WALK_NPZ))
  return Blind(m, d, policy, odometry=odometry)


def course_flight(kind: str, seed: int, npz: str | None = None) -> dict:
  """Course `kind` walked once by the walking policy at `npz` (the served
  one by default), from a start drawn from `seed` -- up to 0.3 m back and
  0.15 rad askew -- so the feet meet each edge somewhere new."""
  m, d = course_world(kind, seed)
  rng = np.random.default_rng(1000 + seed)
  d.qpos[0] -= rng.uniform(0.0, 0.3)
  d.qpos[3:7] = _yaw_quat(rng.uniform(-0.15, 0.15))
  mujoco.mj_forward(m, d)
  w = _walker(m, d, npz)
  w.walk.ask(ROUGH_V, 0.0, 0.0)
  root = m.body(ROBOT_ROOT).id
  x0, fell = float(d.xpos[root][0]), False
  for _ in range(int(ROUGH_S / m.opt.timestep)):
    w.step()
    # Our grading off the truth: the trunk more than 60 deg off upright.
    if d.xmat[root][8] < 0.5:
      fell = True
      break
  return {"course": kind, "seed": seed, "npz": npz or "served", "fell": fell,
          "made_m": float(d.xpos[root][0]) - x0, "s": float(d.time)}


def _course_flight(job):
  return course_flight(*job)


def _name(npz: str | None) -> str:
  return os.path.basename(npz or "served (humanoid_walk.npz)")[:28]


def rough_table(npzs: list[str | None], seeds: int = 8, jobs: int = 4) -> None:
  """Each course walked `seeds` times by each policy: falls, and the mean
  distance made of the `ROUGH_V` * `ROUGH_S` asked. One process a flight."""
  import multiprocessing as mp
  work = [(kind, s, npz) for npz in npzs for kind in COURSES for s in range(seeds)]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_course_flight, work)
  print(f"ROUGH GROUND, each course walked {seeds} times at {ROUGH_V} m/s for {ROUGH_S:g} s "
        f"({ROUGH_V * ROUGH_S:.1f} m asked) from a start up to 0.3 m back and 0.15 rad askew; "
        f"falls of {seeds}, the mean distance made, m:")
  print(f"{'policy':28s}" + "".join(f"{k:>14s}" for k in COURSES))
  for npz in npzs:
    cells = []
    for kind in COURSES:
      rs = [r for r in rows if r["npz"] == (npz or "served") and r["course"] == kind]
      cells.append(f"{sum(r['fell'] for r in rs)} fell {np.mean([r['made_m'] for r in rs]):4.1f}")
    print(f"{_name(npz):28s}" + "".join(f"{c:>14s}" for c in cells))
  print(f"  doorsteps {100 * DOORSTEP_M:.0f} cm; paving {100 * PAVING_W:.0f} cm square to "
        f"{100 * PAVING_M:.0f} cm tall; a slope of {SLOPE} up 1 m, level 1 m, down 1 m; "
        f"blocks {100 * BLOCK_W:.0f} cm square to {100 * BLOCK_M:.0f} cm tall")


#: The tracking table's commands, (forward m/s, sideways m/s, turn rad/s):
#: what the Navigator asks -- a walk, a turn on the spot, a back-off
#: (`mission.BACKOFF_V`) -- and the sidestep it never asks.
TRACK_COMMANDS = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.3, 0.0, 0.0), (-0.3, 0.0, 0.0),
                  (-0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5),
                  (0.0, 0.0, 1.0))
#: Each is flown from standing on a bare floor, measured over `TRACK_S`
#: after `TRACK_SETTLE_S`, s, `TRACK_TRIALS` times, each asked after
#: standing a different while (up to `TRACK_WAIT_S`): the walk starts from
#: another moment of its stance. ⚠ One trial a command read a turn of
#: -0.18 and -0.08 rad/s from one policy at two checkpoints 700 iterations
#: apart.
TRACK_SETTLE_S, TRACK_S, TRACK_TRIALS, TRACK_WAIT_S = 3.0, 6.0, 3, 1.0


def track_row(npz: str | None, cmd, trial: int = 0) -> tuple[float, float, float]:
  """The velocity walked asked `cmd`: forward and sideways in the heading it
  had when measuring began, m/s, and its turn, rad/s (the truth: ours)."""
  m, d = floor_world()
  mujoco.mj_resetDataKeyframe(m, d, 0)
  mujoco.mj_forward(m, d)
  w = _walker(m, d, npz)
  root = m.body(ROBOT_ROOT).id

  def fly(s):
    for _ in range(int(round(s / m.opt.timestep))):
      w.step()

  def heading():
    r = d.xmat[root].reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])

  w.walk.ask(0.0, 0.0, 0.0)
  fly(TRACK_WAIT_S * trial / TRACK_TRIALS)
  w.walk.ask(*cmd)
  fly(TRACK_SETTLE_S)
  p0, h0, t0, turned, last = d.xpos[root][:2].copy(), heading(), d.time, 0.0, heading()
  for _ in range(int(TRACK_S / 0.1)):
    fly(0.1)
    turned += math.remainder(heading() - last, 2 * math.pi)
    last = heading()
  dt = d.time - t0
  dx, dy = d.xpos[root][:2] - p0
  c, s = math.cos(h0), math.sin(h0)
  return (c * dx + s * dy) / dt, (-s * dx + c * dy) / dt, turned / dt


def _track_row(job):
  return track_row(*job)


def track_table(npzs: list[str | None], jobs: int = 4) -> None:
  """--track: each command flown by each policy on a bare floor, the mean
  of `TRACK_TRIALS`, and the asked part's spread."""
  import multiprocessing as mp
  work = [(npz, cmd, k) for npz in npzs for cmd in TRACK_COMMANDS for k in range(TRACK_TRIALS)]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    got = dict(zip(work, pool.map(_track_row, work)))
  print(f"TRACKING on a bare floor, each command flown from standing {TRACK_TRIALS} times, "
        f"measured over {TRACK_S:g} s after {TRACK_SETTLE_S:g} s: forward and sideways m/s, "
        f"turn rad/s, the mean, and the asked part's lowest and highest")
  print(f"{'asked':18s}" + "".join(f"{_name(n):>40s}" for n in npzs))
  for cmd in TRACK_COMMANDS:
    axis = int(np.argmax(np.abs(cmd)))
    cells = []
    for n in npzs:
      rs = np.array([got[(n, cmd, k)] for k in range(TRACK_TRIALS)])
      mean = rs.mean(axis=0)
      cells.append("%+.2f %+.2f %+.2f (%+.2f..%+.2f)" % (*mean, rs[:, axis].min(), rs[:, axis].max()))
    print(f"{'%+.1f %+.1f %+.1f' % cmd:18s}" + "".join(f"{c:>40s}" for c in cells))


#: The odometry table's commands, (forward m/s, sideways m/s, turn rad/s),
#: each flown `ODO_S` s from standing on a bare floor (nothing to match).
ODO_COMMANDS = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.3, 0.0, 0.0), (-0.5, 0.0, 0.0),
                (0.0, 0.3, 0.0), (0.6, 0.0, 0.4))
ODO_S = 8.0


def odometry_row(cmd) -> tuple[float, float, float]:
  """The served walker asked `cmd`, its leg odometry fed as the mission
  feeds it (`humanoid.mission`, every 10 ms): the distance truly walked,
  the odometry's, and how far apart they ended, m."""
  m, d = floor_world()
  mujoco.mj_resetDataKeyframe(m, d, 0)
  mujoco.mj_forward(m, d)
  w = _walker(m, d, None, odometry=True)
  p0 = d.xpos[w.root][:2].copy()
  w.walk.ask(*cmd)
  for _ in range(int(ODO_S / m.opt.timestep)):
    w.step()
  true = d.xpos[w.root][:2] - p0
  believed = np.array([w.odo.x, w.odo.y]) - p0
  return float(np.linalg.norm(true)), float(np.linalg.norm(believed)), float(np.linalg.norm(believed - true))


def odometry_table() -> None:
  """--odometry: the served walker's leg odometry against the truth."""
  print(f"LEG ODOMETRY, the served walker on a bare floor, {ODO_S:g} s from standing "
        "(nothing for a scan to match): walked, believed, apart, m")
  for cmd in ODO_COMMANDS:
    true, believed, apart = odometry_row(cmd)
    print(f"  asked {'%+.1f %+.1f %+.1f' % cmd}: {true:5.2f} {believed:5.2f} {apart:5.2f} "
          f"({100 * apart / max(true, 1e-6):.0f} % of the walk)")


#: The rough scene's steering, off the robot's own leg odometry and IMU
#: (`humanoid.odometry`): aim at the courses' middle line this far ahead,
#: m, turning at this gain, 1/s, and no faster than `STEER_W`, rad/s. Walked
#: open-loop, a walker turned and drifted off the courses.
STEER_AHEAD_M, STEER_GAIN, STEER_W = 2.0, 1.5, 0.8


def rough_scene(model, data, npz: str | None = str(SEEING_NPZ)):
  """The `rough` scene, once: the walker that sees the ground across every course
  end to end, steered back to their middle line off its own senses, saying
  where it is."""
  w = _walker(model, data, npz, odometry=True)
  root, odo = w.root, w.odo
  steer = max(1, round(0.1 / model.opt.timestep))
  marks, x, k = [], ROUGH_AT, 0
  for kind in COURSES[1:]:
    marks.append((x, kind))
    x += {"doorstep": 3.1, "slope": 3.0}.get(kind, 3.0) + 1.5
  end = x
  sees = "seeing the ground" if isinstance(w, Seeing) else "blind"
  yield f"walking at {ROUGH_V} m/s, {sees}, onto the courses, steered by its own odometry"
  n = 0
  while data.time < 90.0:
    if n % steer == 0:
      aim = math.atan2(-odo.y, STEER_AHEAD_M)
      turn = STEER_GAIN * math.remainder(aim - odo.yaw, 2 * math.pi)
      w.walk.ask(ROUGH_V, 0.0, max(-STEER_W, min(STEER_W, turn)))
    w.step()
    n += 1
    if k < len(marks) and data.xpos[root][0] > marks[k][0]:
      yield f"  {data.time:4.1f} s: onto the {marks[k][1]}"
      k += 1
    if data.xmat[root][8] < 0.5:
      yield f"  FELL at {data.xpos[root][0]:.1f} m"
      return
    if data.xpos[root][0] > end:
      yield f"  across, {end:.0f} m, in {data.time:.0f} s"
      return
    yield None
  yield f"  still on the courses at {data.xpos[root][0]:.1f} m after 90 s"


def view_rough() -> None:
  from mujoco import viewer as mj_viewer
  print("the `rough` scene: the walker that sees the ground (models/humanoid_walk_seeing.npz, "
        "not the served one) across doorsteps, paving, a slope and blocks, on its own map. "
        "Close the window to quit.")
  while True:
    model, data = course_world("all")
    mujoco.mj_forward(model, data)
    with mj_viewer.launch_passive(model, data) as viewer:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = model.body(ROBOT_ROOT).id
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.0, 90.0, -15.0
      wall = time.time()
      for line in rough_scene(model, data):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)
      time.sleep(2.0)
      if not viewer.is_running():
        return


def filmstrip_rough(out: str, every_s: float = 3.0) -> None:
  """The rough scene's filmstrip: a tile every `every_s`, from the side."""
  from PIL import Image, ImageDraw
  model, data = course_world("all")
  mujoco.mj_forward(model, data)
  model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.4, 0.6
  r = mujoco.Renderer(model, 300, 400)
  root = model.body(ROBOT_ROOT).id
  tiles, next_t, label = [], 0.0, ""
  for line in rough_scene(model, data):
    if line:
      print(line)
      label = line.strip().split(": ")[-1]
    if data.time >= next_t:
      next_t += every_s
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[root]
      rot = data.xmat[root].reshape(3, 3)
      cam.distance, cam.elevation = 3.5, -10.0
      cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 90.0
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      ImageDraw.Draw(img).text((8, 8), f"{data.time:.0f} s: {label}"[:60], fill=(255, 255, 255))
      tiles.append(img)
  r.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 5: stairs -------------------------------------------------------------------
# The house's flight is five 0.18 m risers on 0.30 m treads, one doorway wide
# between the hall's wall and the landing's face (`home.world`). The walker
# that sees the ground (`walk.SEEING_NPZ`, `training/humanoid_train/walk.py`
# `-Walk-Seeing-Fine`) is shown a height scan: on the IDEAL scan, the terrain itself as
# training cast it (OUR instrument); on the MAP's, its own -- the head's D435
# laid into a map through its belief (`humanoid.scan.HeadScan`), the belief
# its legs' odometry with the map's ground under its soles
# (`humanoid.odometry`). A trial is graded by us off the truth.

from pluggybot.home import world as hw  # noqa: E402
from pluggybot.humanoid.arms import SelfModel  # noqa: E402
from pluggybot.humanoid.odometry import StanceOdometry  # noqa: E402
from pluggybot.humanoid.scan import HeadScan, ideal_scan  # noqa: E402
from pluggybot.humanoid.mission import DEPTH_PERIOD  # noqa: E402

#: The flight's foot ahead of the robot's start, m; a trial walks at
#: `CLIMB_V` for at most `CLIMB_S`, and reaches the landing (or the floor)
#: once its pelvis is `CLIMB_PAST_M` past the flight's end.
FLIGHT_AT, CLIMB_V, CLIMB_S, CLIMB_PAST_M = 1.0, 0.4, 25.0, 0.5
CLIMB_RISERS = (0.10, 0.15, 0.18, 0.20, 0.22)
#: The landing's depth at the top of a course, m (the house's is 1.5).
LANDING_M = 1.5


def flight_world(rise: float, steps: int = hw.STAIRS_STEPS, down: bool = False):
  """A flight of `steps` risers of `rise` on the house's tread, one flight
  wide between two walls a room tall, a landing at its top; the robot at
  its stand `FLIGHT_AT` short of the foot facing up it, or (`down`) on the
  landing facing down it. The model, its data, and the flight's foot and
  end (x, m)."""
  spec = mujoco.MjSpec.from_string(hm.body_xml())
  wb = spec.worldbody
  grey, wall = [0.58, 0.54, 0.50, 1.0], [0.8, 0.78, 0.74, 1.0]
  tread, width = hw.STAIRS_TREAD, hw.STAIRS_WIDTH
  x0, run, top = FLIGHT_AT, hw.STAIRS_TREAD * steps, rise * steps

  def box(x_from, x_to, h, y=0.0, w=width, z0=0.0, rgba=grey):
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=rgba,
                size=[(x_to - x_from) / 2, w / 2, h / 2], pos=[(x_from + x_to) / 2, y, z0 + h / 2])

  for i in range(steps):
    box(x0 + i * tread, x0 + run, rise * (i + 1))
  box(x0 + run, x0 + run + LANDING_M, top)
  for side in (1, -1):
    box(-1.5, x0 + run + LANDING_M, hw.WALL_HALF_H * 2 + top, y=side * (width / 2 + 0.02),
        w=0.04, rgba=wall)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  if down:
    data.qpos[0] = x0 + run + 0.6
    data.qpos[2] += top
    data.qpos[3:7] = _yaw_quat(math.pi)
  mujoco.mj_forward(model, data)
  return model, data, x0, x0 + run


class Seeing:
  """A walker that sees the ground on a bare body, its scan ideal or its own map's
  (`scan` "ideal" or "map"), and the odometry the map is laid through:
  `step()` once a physics step, in place of the walker's and the arms'."""

  def __init__(self, model, data, scan: str = "map", npz: str | None = None):
    from pluggybot.humanoid.walk import WalkDriver
    self.m, self.d = model, data
    self.root = model.body(ROBOT_ROOT).id
    self.stand = StandDriver(model, data)
    self.reach = Reach(model, data, self.stand)
    x, y = float(data.xpos[self.root][0]), float(data.xpos[self.root][1])
    rot = data.xmat[self.root].reshape(3, 3)
    att = self.stand.trunk.attitude
    self.head = HeadScan(model, data, belief=lambda: (self.odo.x, self.odo.y, self.odo.z, self.odo.yaw),
                         level=att.level, correct=self._correct, period=DEPTH_PERIOD)
    self.odo = StanceOdometry(SelfModel(), att, x, y, terrain=self.head.under,
                              felt=self.head.feel, ground_at=self.head.ground_at)
    self.odo.correct(x, y, math.atan2(rot[1, 0], rot[0, 0]))
    q, _, _ = self.stand.joints.read(data)
    self.odo.step(q, self.stand.trunk.gravity())
    see = (lambda: ideal_scan(model, data, self.root)) if scan == "ideal" else self.head.scan
    self.walk = WalkDriver(model, data, policy=StandPolicy(npz or str(SEEING_NPZ)),
                           trunk=self.stand.trunk, joints=self.stand.joints, scan=see)
    self.every = max(1, round(0.01 / model.opt.timestep))
    self.n = 0

  def _correct(self, dz: float) -> None:
    self.odo.z += dz

  def step(self) -> None:
    self.walk.step()
    self.reach.step()
    mujoco.mj_step(self.m, self.d)
    self.head.step()
    if self.n % self.every == 0:
      q, _, _ = self.stand.joints.read(self.d)
      self.odo.step(q, self.stand.trunk.gravity())
    self.n += 1

  def steer(self, heading: float, line_y: float, v: float = CLIMB_V) -> None:
    """Walk at `v` along `heading`, held to the line y = `line_y` off its own
    belief (the rough scene's steering, `STEER_*`)."""
    # Off the line to its left, along a heading of 0 or pi: turn back to it.
    off = (self.odo.y - line_y) * math.cos(heading)
    aim = heading - math.atan2(off, STEER_AHEAD_M)
    w = STEER_GAIN * math.remainder(aim - self.odo.yaw, 2 * math.pi)
    self.walk.ask(v, 0.0, max(-STEER_W, min(STEER_W, w)))


def climb_trial(rise: float, down: bool, scan: str = "map", npz: str | None = None,
                seed: int = 0, steps: int = hw.STAIRS_STEPS) -> dict:
  """The house's flight at `rise` (`steps` risers), walked up (or down) once
  at `CLIMB_V` from a start drawn from `seed` -- up to 0.2 m back, 5 cm off
  its middle, 0.08 rad askew -- graded by us: whether it got there, how
  long it took, whether it fell, and its legs' hardest torques."""
  model, data, foot, end = flight_world(rise, steps=steps, down=down)
  rng = np.random.default_rng(seed)
  heading = math.pi if down else 0.0
  data.qpos[0] += (1 if down else -1) * rng.uniform(0.0, 0.2)
  data.qpos[1] += rng.uniform(-0.05, 0.05)
  data.qpos[3:7] = _yaw_quat(heading + rng.uniform(-0.08, 0.08))
  mujoco.mj_forward(model, data)
  w = Seeing(model, data, scan=scan, npz=npz)
  goal = foot - CLIMB_PAST_M if down else end + CLIMB_PAST_M
  peak = np.zeros(len(hm.JOINT_NAMES))
  act = np.array([model.actuator(j).id for j in hm.JOINT_NAMES])
  fell, there, steer = False, False, max(1, round(0.1 / model.opt.timestep))
  for k in range(int(CLIMB_S / model.opt.timestep)):
    if k % steer == 0:
      w.steer(heading, 0.0)
    w.step()
    peak = np.maximum(peak, np.abs(data.actuator_force[act]))
    x = float(data.xpos[w.root][0])
    if data.xmat[w.root][8] < 0.5:
      fell = True
      break
    if (x < goal) if down else (x > goal):
      there = True
      break
  return {"rise": rise, "down": down, "scan": scan, "seed": seed, "there": there,
          "fell": fell, "s": float(data.time), "peak": peak.tolist(),
          "belief_z_m": float(w.odo.z), "true_z_m": float(data.xpos[w.root][2])}


def _climb_trial(job):
  return climb_trial(*job)


def climb_table(npz: str | None = None, trials: int = 5, jobs: int = 4,
                scans=("ideal", "map")) -> list[dict]:
  """--climb: flights of five at each riser, up and down, `trials` times on
  each scan. One process a flight."""
  import multiprocessing as mp
  work = [(r, down, sc, npz, s) for r in CLIMB_RISERS for down in (False, True)
          for sc in scans for s in range(trials)]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_climb_trial, work)
  print(f"STAIRS, flights of {hw.STAIRS_STEPS} risers on the house's {100 * hw.STAIRS_TREAD:.0f} cm "
        f"tread, walked at {CLIMB_V} m/s ({_name(npz or str(SEEING_NPZ))}): got there of "
        f"{trials}, falls, and the mean time, s")
  print(f"{'riser':>8s}" + "".join(f"{f'{sc} up':>18s}{f'{sc} down':>18s}" for sc in scans))
  for r in CLIMB_RISERS:
    cells = []
    for sc in scans:
      for down in (False, True):
        rs = [x for x in rows if x["rise"] == r and x["down"] == down and x["scan"] == sc]
        ok = [x for x in rs if x["there"]]
        t = f"{np.mean([x['s'] for x in ok]):.1f}" if ok else "-"
        cells.append(f"{len(ok)}/{trials} {sum(x['fell'] for x in rs)}f {t}s")
    print(f"{r:8.2f}" + "".join(f"{c:>18s}" for c in cells))
  return rows


# ---- stage 5: the scene `stairs`, in the house -------------------------------------------
# The humanoid in the hall, facing the flight's foot: up the flight on the
# walker that sees the ground, steered along its lane's middle by its own belief (its
# legs' odometry, the map's ground under its soles, its IMU's heading),
# round on the landing, and back down. Staged with the truth: where it
# starts. Graded by us: whether it got up and down, how long each took,
# whether it fell, and how far its belief was off the truth.

#: The flight's lane: its middle line (x, m), its foot and its top (y, m;
#: `home.world`), and where the scene starts the robot, facing up it.
LANE_X = hw.STAIRS_X[0] + hw.STAIRS_WIDTH / 2
FLIGHT_FOOT_Y = hw.STAIRS_Y[1]
FLIGHT_TOP_Y = hw.STAIRS_Y[1] - hw.STAIRS_TREAD * hw.STAIRS_STEPS
STAIRS_FROM = (LANE_X, FLIGHT_FOOT_Y + 1.4)
#: On the landing once its belief is this far past the top step, m; back
#: down once it is this far out into the hall.
LANDING_IN_M, HALL_OUT_M = 0.5, 1.0
#: The scene's camera: from the hall north of the flight, looking up it
#: (distance m, azimuth and elevation deg): the lane runs between a wall
#: a room tall and the deck's half wall, and seen from the side either hid
#: the robot.
STAIRS_CAM = (3.2, -90.0, -22.0)


def stairs_world(npz: str | None = None):
  """The house, the humanoid in the hall facing up the flight, its walker
  the one that sees the ground (`npz`, `walk.SEEING_NPZ` by default)."""
  return house_world(at=STAIRS_FROM, yaw=-math.pi / 2,
                     walk=StandPolicy(npz or str(SEEING_NPZ)))


def _lane_routine(mission, heading: float, done, budget_s: float = 30.0, v: float = CLIMB_V):
  """Walk along the lane's middle line along `heading` (+-pi/2) until
  `done(belief)`, off the robot's own belief: True, or False when the
  budget ran out."""
  t0 = float(mission.data.time)
  while not done(mission.pose):
    if float(mission.data.time) - t0 > budget_s:
      return False
    x, y, yaw = mission.pose
    off = (x - LANE_X) * -math.sin(heading)          # left of the line, facing along it
    aim = heading - math.atan2(off, STEER_AHEAD_M)
    w = STEER_GAIN * math.remainder(aim - yaw, 2 * math.pi)
    for _ in range(max(1, round(0.1 / mission.model.opt.timestep))):
      yield from mission._twist_routine(v, 0.0, max(-STEER_W, min(STEER_W, w)))
  return True


def _within(gen, data, until: float):
  """`gen` driven until sim time `until`: what it returned, or None when the
  time ran out (a get-up that never stood holds the legs for good: the
  lifecycle's `stuck` death, which a scene has not got)."""
  try:
    x = next(gen)
    while True:
      if data.time > until:
        gen.close()
        return None
      yield x
      x = gen.send(None)
  except StopIteration as e:
    return e.value


#: A leg of the `stairs` scene ends by this, sim s: its walk's own budget
#: (30 s) and a get-up's (`HumanoidBody.GETUP_BUDGET_S`) with room over.
LEG_S = 90.0


def stairs_scene(body, model, data, records: list | None = None):
  """The `stairs` scene, once: up the flight, round, down, saying how each
  part ended."""
  m = body.mission
  legs = (("up the flight", -math.pi / 2, lambda p: p[1] < FLIGHT_TOP_Y - LANDING_IN_M),
          ("down the flight", math.pi / 2, lambda p: p[1] > FLIGHT_FOOT_Y + HALL_OUT_M))
  for k, (name, heading, done) in enumerate(legs):
    if k:
      yield "turning round on the landing"
      if (yield from _within(drive(body, m.face_routine(heading)), data,
                             data.time + LEG_S)) is None:
        yield "  still down after a fall"
        return
    yield f"walking {name}"
    t0, falls = data.time, m.falls
    there = yield from _within(drive(body, _lane_routine(m, heading, done)), data, t0 + LEG_S)
    bx, by, _ = m.pose
    tx, ty, _ = m.true_pose()
    if there is None:
      if records is not None:
        records.append({"leg": name, "there": False, "seconds": data.time - t0,
                        "falls": m.falls - falls, "down": True, "belief_error_m": math.nan,
                        "height_m": float(data.xpos[body.root][2]), "belief_z_m": m.odo.z})
      yield "  still down after a fall"
      return
    rec = {"leg": name, "there": bool(there), "seconds": data.time - t0,
           "belief_error_m": math.hypot(bx - tx, by - ty),
           "height_m": float(data.xpos[body.root][2]), "belief_z_m": m.odo.z,
           "falls": m.falls - falls, "down": bool(data.xmat[body.root][8] < 0.5)}
    if records is not None:
      records.append(rec)
    fell = f", {rec['falls']} fall(s) taken by its get-up" if rec["falls"] else ""
    yield (f"  {'there' if there else 'not there'} in {rec['seconds']:.0f} s{fell}; its belief "
           f"{100 * rec['belief_error_m']:.0f} cm off the truth, its height believed "
           f"{rec['belief_z_m']:.2f} m against {rec['height_m']:.2f}")
    if rec["down"]:
      yield "DOWN"
      return
    if not there:
      return


#: The house's climbs (`--climb`'s second table): this many, each from the
#: scene's start drawn askew -- up to 0.3 m back, 5 cm off the lane's
#: middle, 0.08 rad -- up the flight and down it as the scene walks them.
HOUSE_CLIMBS = 10


def house_climb_trial(seed: int, npz: str | None = None) -> dict:
  """The `stairs` scene's legs from a start drawn from `seed`: how each
  ended, graded by us off the truth."""
  rng = np.random.default_rng(seed)
  at = (STAIRS_FROM[0] + rng.uniform(-0.05, 0.05), STAIRS_FROM[1] + rng.uniform(0.0, 0.3))
  model, data, body = house_world(at=at, yaw=-math.pi / 2 + rng.uniform(-0.08, 0.08),
                                  walk=StandPolicy(npz or str(SEEING_NPZ)))
  records = []
  try:
    for _ in stairs_scene(body, model, data, records):
      pass
  finally:
    body.close()
  return {"seed": seed, "legs": records}


def _house_climb_trial(job):
  return house_climb_trial(*job)


def house_climb_table(npz: str | None = None, jobs: int = 4) -> list[dict]:
  """--climb, in the house: the scene's legs from `HOUSE_CLIMBS` starts."""
  import multiprocessing as mp
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_house_climb_trial, [(k, npz) for k in range(HOUSE_CLIMBS)])
  print(f"THE HOUSE'S FLIGHT, the `stairs` scene from {HOUSE_CLIMBS} starts drawn askew "
        f"({_name(npz or str(SEEING_NPZ))}, its map laid every {DEPTH_PERIOD:g} s): up it, then "
        f"round and down it, each leg's end, time, falls its get-up took, and its belief "
        f"against the truth at the end, m")
  for leg in ("up the flight", "down the flight"):
    rs = [x for r in rows for x in r["legs"] if x["leg"] == leg]
    ok = [x for x in rs if x["there"] and not x["down"]]
    print(f"  {leg:16s} there {len(ok)}/{HOUSE_CLIMBS} (tried {len(rs)}), "
          f"{sum(x['falls'] for x in rs)} falls, "
          + (f"{np.median([x['seconds'] for x in ok]):.0f} s, belief off {np.median([x['belief_error_m'] for x in ok]):.2f} m "
             f"and its height {np.median([abs(x['belief_z_m'] - x['height_m']) for x in ok]):.2f} m (medians)"
             if ok else "-"))
  return rows


def view_stairs() -> None:
  from mujoco import viewer as mj_viewer
  print("the `stairs` scene: the humanoid up the house's flight on the walker that sees "
        "the ground, round on the landing and down. Close the window to quit.")
  while True:
    model, data, body = stairs_world()
    with mj_viewer.launch_passive(model, data) as viewer:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = body.root
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = STAIRS_CAM
      wall = time.time() - data.time
      for line in stairs_scene(body, model, data):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = data.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)
      time.sleep(2.0)
      if not viewer.is_running():
        return
    body.close()


def filmstrip_stairs(out: str, every_s: float = 2.0) -> None:
  """The stairs scene's filmstrip: a tile every `every_s`, from the hall
  behind the flight's foot (`STAIRS_CAM`)."""
  from PIL import Image, ImageDraw
  model, data, body = stairs_world()
  r = mujoco.Renderer(model, 300, 400)
  tiles, next_t, label = [], 0.0, ""
  for line in stairs_scene(body, model, data):
    if line and not line.startswith("  "):
      label = line
    elif line:
      print(line)
    if data.time >= next_t:
      next_t += every_s
      cam = mujoco.MjvCamera()
      cam.lookat[:] = data.xpos[body.root]
      cam.distance, cam.azimuth, cam.elevation = STAIRS_CAM
      light = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
      model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.4, 0.6
      r.update_scene(data, cam)
      img = Image.fromarray(r.render())
      model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = light
      ImageDraw.Draw(img).text((8, 8), f"{data.time:.0f} s: {label}"[:60], fill=(255, 255, 255))
      tiles.append(img)
  r.close()
  body.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 5: falling, and getting up ------------------------------------------------------
# On hardware a fall can break an actuator (#507): what one costs is read
# here, for when it is costed. The robot standing on a bare floor is shoved
# over -- staged with the truth, a push at its trunk's centre of mass -- and
# what follows is graded by us off the world: which part hit the floor
# first, how fast and how hard each part was struck, the hardest torque
# through each joint's gears -- its motor's, less what its rotor's inertia
# takes as the joint is stopped short (`dof_armature` times the joint's
# deceleration: the shock that strips a gearbox) -- against its part's
# peak, how hard each was driven into its hard stop, and how fast each was
# driven back against its part's no-load speed (back-driven past it, a
# motor pumps the bus). Three ways to
# fall: the standing policy keeps the legs; the drivers go limp once the
# trunk has tipped `getup.FALL_TILT_RAD` past its posture; or the get-up policy takes
# the legs and arms there (`humanoid.getup`), and gets up.

from pluggybot.humanoid.getup import (COLLAPSED_M, FALL_TILT_RAD, GETUP_NPZ,  # noqa: E402
                                      FallWatch, GetupDriver, over_lower_sole, tipped)

#: The shoves, the robot's frame (x ahead), and how hard, N*s: past what it
#: stands through every way (`--shove`: 10-20 N*s), it goes over.
FALL_WAYS = {"front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1),
             "front-left": (1, 1), "back-right": (-1, -1)}
FALL_NS = 45.0
FALL_S = 8.0
#: The robot's parts, by the geoms that strike the floor.
FALL_PARTS = {
  "head": ("head",), "trunk": ("torso", "pelvis"),
  "arms": tuple(f"{s}_{g}" for s in hm.SIDES for g in ("upper_arm", "forearm")),
  "hands": tuple(f"{s}_{g}" for s in hm.SIDES for g in ("gripper", "jaw_a_pad", "jaw_b_pad")),
  "legs": tuple(f"{s}_{g}" for s in hm.SIDES for g in ("thigh", "shank")),
}


def fall_trial(way: str, mode: str, npz: str | None = None, ns: float = FALL_NS) -> dict:
  """The robot shoved `way` at `ns` N*s, falling as `mode` has it ("stand",
  "limp" or "getup"): what hit first, the parts' peak forces (N), each
  joint's peak torque (N*m, its motor's and its hard stop's) and speed
  (rad/s), and -- the get-up's -- when it stood again."""
  h = floor_humanoid()
  m, d = h.m, h.d
  ids = {part: [m.geom(g).id for g in gs] for part, gs in FALL_PARTS.items()}
  floor = m.geom("floor").id
  act = np.array([m.actuator(j).id for j in hm.JOINT_NAMES])
  dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in hm.JOINT_NAMES])
  getup = GetupDriver(m, d, policy=StandPolicy(npz or str(GETUP_NPZ)), trunk=h.stand.trunk,
                      joints=h.stand.joints) if mode == "getup" else None
  legs = Drivers(m, d, "", joints=hm.JOINT_NAMES)
  for _ in range(int(1.0 / m.opt.timestep)):
    h.step()
  watch = FallWatch(h.stand.trunk, h.stand.joints)
  first, peaks = None, {p: 0.0 for p in FALL_PARTS}
  hit = {p: None for p in FALL_PARTS}
  vel = np.zeros(6)
  tau = np.zeros(len(hm.JOINT_NAMES))
  stop = np.zeros(len(hm.JOINT_NAMES))
  speed = np.zeros(len(hm.JOINT_NAMES))
  arm = m.dof_armature[dof]
  jid = {int(m.joint(j).id): k for k, j in enumerate(hm.JOINT_NAMES)}
  took_over, stood_at, gen = None, None, shove(h, FALL_WAYS[way], ns)
  force = np.zeros(6)
  t0 = d.time
  while d.time - t0 < FALL_S:
    if gen is not None:
      if next(gen, "done") == "done":
        gen = None
      continue
    if took_over is None and watch.falling(d, h.reach.held_pitch()):
      took_over = d.time - t0
      if mode == "getup":
        getup.begin()
    if took_over is not None and mode == "getup":
      getup.step(trunk_stepped=False)
      mujoco.mj_step(m, d)
      if getup.stood and stood_at is None:
        stood_at = d.time - t0
    elif took_over is not None and mode == "limp":
      h.stand.trunk.step()
      legs.limp()
      mujoco.mj_step(m, d)
    else:
      h.step()
    for i in range(d.ncon):
      c = d.contact[i]
      other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
      if other is None:
        continue
      mujoco.mj_contactForce(m, d, i, force)
      for part, gs in ids.items():
        if other in gs:
          peaks[part] = max(peaks[part], float(np.linalg.norm(force[:3])))
          if hit[part] is None:
            # The part's speed as it first touched: what a fall is, less
            # the solver's say in how hard the floor pushed back.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, other, vel, 0)
            hit[part] = float(np.linalg.norm(vel[3:]))
          if first is None:
            first = part
    tau = np.maximum(tau, np.abs(d.actuator_force[act] - arm * d.qacc[dof]))
    for r in np.flatnonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
      k = jid.get(int(d.efc_id[r]))
      if k is not None:
        stop[k] = max(stop[k], abs(float(d.efc_force[r])))
    speed = np.maximum(speed, np.abs(d.qvel[dof]))
  return {"way": way, "mode": mode, "first": first, "peaks": peaks, "hit": hit, "tau": tau.tolist(),
          "stop": stop.tolist(),
          "speed": speed.tolist(), "took_over_s": took_over, "stood_s": stood_at,
          "up_at_end": bool(d.xmat[h.root][8] > 0.95 and d.xpos[h.root][2] > 0.6)}


def _fall_trial(job):
  return fall_trial(*job)


def fall_table(npz: str | None = None, jobs: int = 4) -> list[dict]:
  """--fall: each way shoved over, falling each of three ways."""
  import multiprocessing as mp
  modes = ("stand", "limp", "getup")
  work = [(w, md, npz) for w in FALL_WAYS for md in modes]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_fall_trial, work)
  peak_tau = np.array([hm.CHOSEN.motor(j).peak_torque for j in hm.JOINT_NAMES])
  noload = np.array([hm.CHOSEN.motor(j).noload_speed for j in hm.JOINT_NAMES])
  print(f"FALLS, shoved {FALL_NS:g} N*s at the trunk each way, standing on a bare floor; the "
        f"get-up policy or the limp drivers take over {math.degrees(FALL_TILT_RAD):.0f} deg past its "
        "posture")
  print("  each part: the speed it first touched the floor at, m/s, and the hardest the floor "
        "pushed back, kN (MuJoCo's soft contact: an upper bound's shape, not a number to bank)")
  print("  gears past peak: joints whose torque through the gears passed their part's peak "
        "(the worst, x peak, and which); stops: the hardest a joint was driven into its "
        "hard stop, N*m; past no-load: joints back-driven past their part's no-load speed")
  print(f"{'way':11s}{'falling as':>11s}{'first hit':>10s}" + "".join(f"{p:>13s}" for p in FALL_PARTS)
        + f"{'gears past peak':>34s}{'stops':>8s}{'past no-load':>14s}{'stood':>8s}")
  for r in rows:
    share = np.array(r["tau"]) / peak_tau
    over = np.flatnonzero(share > 1.0)
    worst = hm.JOINT_NAMES[int(np.argmax(share))]
    fast = int((np.array(r["speed"]) > noload).sum())
    stood = f"{r['stood_s']:.1f}s" if r["stood_s"] else ("up" if r["up_at_end"] else "-")
    gears = f"{len(over)} ({share.max():.1f}x, {worst})" if len(over) else "0"
    print(f"{r['way']:11s}{r['mode']:>11s}{str(r['first']):>10s}"
          + "".join(f"{'-':>13s}" if r["hit"][p] is None
                    else f"{r['hit'][p]:6.1f} {r['peaks'][p] / 1000:5.1f}" for p in FALL_PARTS)
          + f"{gears:>34s}{max(r['stop']):8.0f}{fast:>14d}{stood:>8s}")
  return rows


#: Where the house's falls are taken (x, y, the ground's height, facing):
#: the hall's middle, a wall at its side, the couch ahead, the flight's
#: third tread facing up it, and the landing's edge facing down it.
HOUSE_FALLS = {
  "the hall": ((-3.5, 1.5), 0.0, math.pi / 2),
  "by a wall": ((-4.55, 0.5), 0.0, math.pi / 2),
  "by the couch": ((3.2, -0.05), 0.0, math.pi / 2),
  "on the flight": ((hw.STAIRS_X[0] + hw.STAIRS_WIDTH / 2, -3.75), 3 * hw.STAIRS_RISE, -math.pi / 2),
  "the landing's edge": ((hw.STAIRS_X[0] + hw.STAIRS_WIDTH / 2, -4.75),
                         hw.STAIRS_RISE * hw.STAIRS_STEPS, math.pi / 2),
}
HOUSE_WAYS = ("front", "back", "left", "right")
#: A house fall is watched this long from the shove, s.
HOUSE_FALL_S = 15.0


def house_fall_trial(place: str, way: str, npz: str | None = None, ns: float = FALL_NS) -> dict:
  """The robot standing at `place` in the house, shoved `way`, its get-up
  policy taking the fall as the mission hands it: what it struck first and
  how fast its head went into anything, whether and when it stood, and
  where its pelvis ended (graded by us off the truth)."""
  at, z, yaw = HOUSE_FALLS[place]
  m, d = standing_world(at=at, yaw=yaw, z=z)
  h = Humanoid(m, d)
  getup = GetupDriver(m, d, policy=StandPolicy(npz or str(GETUP_NPZ)), trunk=h.stand.trunk,
                      joints=h.stand.joints)
  watch = FallWatch(h.stand.trunk, h.stand.joints)
  ours = {g for g in range(m.ngeom) if _in_robot(m, int(m.geom_bodyid[g]), h.root)}
  head = m.geom("head").id
  for _ in range(int(1.0 / m.opt.timestep)):
    h.step()
  gen, took, stood, first, head_v = shove(h, FALL_WAYS[way], ns), None, None, None, None
  vel = np.zeros(6)
  t0 = d.time
  while d.time - t0 < HOUSE_FALL_S:
    if gen is not None:
      if next(gen, "done") == "done":
        gen = None
      continue
    if took is None and watch.falling(d, h.reach.held_pitch()):
      took = d.time - t0
      getup.begin()
    if took is not None and stood is None:
      getup.step()
      mujoco.mj_step(m, d)
      if getup.stood:
        stood = d.time - t0
        h = Humanoid(m, d)
    else:
      h.step()
    for i in range(d.ncon):
      c = d.contact[i]
      if (c.geom1 in ours) == (c.geom2 in ours):
        continue
      mine = c.geom1 if c.geom1 in ours else c.geom2
      if m.geom(mine).name.endswith("_sole"):
        continue
      if first is None:
        first = next((part for part, gs in FALL_PARTS.items() if m.geom(mine).name in gs),
                     m.geom(mine).name)
      if mine == head and head_v is None:
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, head, vel, 0)
        head_v = float(np.linalg.norm(vel[3:]))
  return {"place": place, "way": way, "first": first, "head_v": head_v, "took_s": took,
          "stood_s": stood, "end_z": float(d.xpos[h.root][2]) - z}


def _in_robot(m, body: int, root: int) -> bool:
  while body > 0:
    if body == root:
      return True
    body = int(m.body_parentid[body])
  return False


def _house_fall_trial(job):
  return house_fall_trial(*job)


def house_fall_table(npz: str | None = None, jobs: int = 4) -> list[dict]:
  """--fall, in the house: each place shoved each way, the get-up taking it."""
  import multiprocessing as mp
  work = [(p, w, npz) for p in HOUSE_FALLS for w in HOUSE_WAYS]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_house_fall_trial, work)
  print(f"FALLS IN THE HOUSE, shoved {FALL_NS:g} N*s each way, the get-up policy taking each "
        f"fall: what struck first (a sole aside), the head's speed into anything, m/s, and when "
        f"it stood from the shove -- or where its pelvis lay over the ground it had stood on "
        f"after {HOUSE_FALL_S:g} s")
  print(f"{'place':20s}" + "".join(f"{w:>27s}" for w in HOUSE_WAYS))
  for place in HOUSE_FALLS:
    cells = []
    for w in HOUSE_WAYS:
      r = next(x for x in rows if x["place"] == place and x["way"] == w)
      head = "-" if r["head_v"] is None else f"{r['head_v']:.1f}"
      end = f"stood {r['stood_s']:.1f}s" if r["stood_s"] else (
        "held" if r["took_s"] is None else f"down {r['end_z']:+.2f}")
      cells.append(f"{(r['first'] or '-')[:6]} {head:>4s} {end:>12s}")
    print(f"{place:20s}" + "".join(f"{c:>27s}" for c in cells))
  print("  (down: still on the ground after the watch, its pelvis that high over the ground it "
        "had stood on, m)")
  return rows


# Falls from a bow (#520): the watch reads a tip off the posture the legs
# hold, so a bow a reach asks is no fall, and a shove from one still is.
# Each posture a reach takes, shoved each way at four strengths: once on the
# standing policy alone, graded by us off the world -- did it fall -- with
# the moment each check WOULD have taken it; once as served, the checks
# handing the fall to the get-up policy.

#: The postures (height off the stand's, m; pitch, rad): the stand and
#: `reach.POSTURES`' bows. The shoves' ways and strengths, N*s, and how long
#: each is watched, s.
BOW_POSTURES = ((0.0, 0.0), (-0.2, 0.55), (-0.22, 0.8), (-0.25, 0.95), (-0.25, 1.1))
BOW_WAYS, BOW_NS, BOW_WATCH_S = ("front", "back", "left", "right"), (10.0, 20.0, 30.0, 45.0), 6.0


def bow_fall_trial(held, way: str, ns: float, served: bool, npz: str | None = None) -> dict:
  """The robot on a bare floor settled in posture `held` (its reach holding
  it, as a plan's posture is), shoved `way` at `ns` N*s and watched
  `BOW_WATCH_S`: when each check fires -- the tilt past the posture (the
  watch's), off upright (the watch's as it was), the collapse -- what struck
  the floor first, a hand aside, and when, and the head's speed into it;
  whether it fell, by our grading (a part struck, or its pelvis 0.15 m under
  where it held); and `served`, the checks handing the fall to the get-up
  policy, when they did and when it stood. Times from the shove's start,
  as `fall_trial`'s."""
  h = floor_humanoid()
  m, d = h.m, h.d
  floor, head = m.geom("floor").id, m.geom("head").id
  part_of = {m.geom(g).id: p for p, gs in FALL_PARTS.items() for g in gs}
  watch = FallWatch(h.stand.trunk, h.stand.joints)
  getup = GetupDriver(m, d, policy=StandPolicy(npz or str(GETUP_NPZ)), trunk=h.stand.trunk,
                      joints=h.stand.joints) if served else None
  every = max(1, round(h.stand.policy.period / m.opt.timestep))
  _drain(wait(h, 1.0))
  _drain(posture(h, *held))
  _drain(wait(h, 1.5))
  z0 = float(d.xpos[h.root][2])
  rec = {"posture": tuple(held), "way": way, "ns": ns, "served": served, "tilt_s": None,
         "upright_s": None, "collapse_s": None, "first": None, "hit_s": None, "head_v": None,
         "took_s": None, "stood_s": None, "max_tip": 0.0}
  t0 = d.time
  _drain(shove(h, FALL_WAYS[way], ns))
  vel = np.zeros(6)
  n, up = 0, False
  while d.time - t0 < BOW_WATCH_S:
    if not up and n % every == 0:
      t, g = d.time - t0, h.stand.trunk.gravity()
      tip = tipped(g, h.reach.held_pitch())
      rec["max_tip"] = max(rec["max_tip"], tip)
      sunk = over_lower_sole(watch.sm, h.stand.joints.read(d)[0], g) < COLLAPSED_M
      for key, fired in (("tilt_s", tip > FALL_TILT_RAD), ("collapse_s", sunk),
                         ("upright_s", h.stand.trunk.attitude.tilt() > FALL_TILT_RAD)):
        if fired and rec[key] is None:
          rec[key] = t
      if served and (tip > FALL_TILT_RAD or sunk):
        up, rec["took_s"] = True, t
        getup.begin()
    if up:
      getup.step()
      mujoco.mj_step(m, d)
      if getup.stood:
        rec["stood_s"] = d.time - t0
        break
    else:
      h.step()
    n += 1
    for i in range(d.ncon):
      c = d.contact[i]
      other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
      if other not in part_of or part_of[other] == "hands":
        continue
      if rec["first"] is None:
        rec["first"], rec["hit_s"] = part_of[other], d.time - t0
      if other == head and rec["head_v"] is None:
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, head, vel, 0)
        rec["head_v"] = float(np.linalg.norm(vel[3:]))
  rec["fell"] = rec["hit_s"] is not None or float(d.xpos[h.root][2]) < z0 - 0.15
  return rec


def _bow_fall_trial(job):
  return bow_fall_trial(*job)


def _span(values, fmt: str = "{:.1f}") -> str:
  return "-" if not values else fmt.format(min(values)) + "-" + fmt.format(max(values))


def bow_fall_table(npz: str | None = None, jobs: int = 4) -> list[dict]:
  """--fall, from a bow: each posture a reach takes, shoved each way at each
  strength, on the standing policy alone and as served."""
  import multiprocessing as mp
  work = [(p, w, ns, s, npz) for s in (False, True) for p in BOW_POSTURES for w in BOW_WAYS
          for ns in BOW_NS]
  with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
    rows = pool.map(_bow_fall_trial, work)
  alone = [r for r in rows if not r["served"]]
  served = [r for r in rows if r["served"]]
  print(f"FALLS FROM A BOW, on a bare floor: each posture a reach takes, shoved {', '.join(BOW_WAYS)} "
        f"at {', '.join(f'{ns:g}' for ns in BOW_NS)} N*s. On the standing policy alone, graded by "
        "us: the falls, which check would have taken each first, how long before a part but a "
        "hand struck the floor, and the shoves it stood through that a check took for a fall -- "
        "the tilt past the posture (the watch) and off upright (the watch as it was). As served, "
        "the checks handing each fall to the get-up policy: stood, s from the shove; and the "
        "head into the floor, m/s, alone and served")
  print(f"{'posture':>14s}{'falls':>7s}{'first: tilt, collapse':>23s}{'ahead, s':>11s}"
        f"{'false: posture, upright':>25s}{'stood, served':>18s}{'head alone':>17s}"
        f"{'head served':>17s}")
  for p in BOW_POSTURES:
    fell = [r for r in alone if r["posture"] == p and r["fell"]]
    stood_through = [r for r in alone if r["posture"] == p and not r["fell"]]
    by_tilt = sum((r["tilt_s"] is not None)
                  and (r["collapse_s"] is None or r["tilt_s"] <= r["collapse_s"]) for r in fell)
    caught = [r for r in fell if r["tilt_s"] is not None or r["collapse_s"] is not None]
    ahead = [r["hit_s"] - min(t for t in (r["tilt_s"], r["collapse_s"]) if t is not None)
             for r in caught if r["hit_s"] is not None]
    false_now = sum(r["tilt_s"] is not None or r["collapse_s"] is not None for r in stood_through)
    false_was = sum(r["upright_s"] is not None or r["collapse_s"] is not None for r in stood_through)
    up = [r for r in served if r["posture"] == p and r["took_s"] is not None]
    stood = [r["stood_s"] for r in up if r["stood_s"] is not None]
    heads = [r["head_v"] for r in fell if r["head_v"]]
    heads_s = [r["head_v"] for r in up if r["head_v"]]
    print(f"{p[0]:+6.2f} m {p[1]:4.2f}{len(fell):>4d}/{len(fell) + len(stood_through)}"
          f"{f'{by_tilt}, {len(caught) - by_tilt}':>23s}{_span(ahead, '{:.2f}'):>11s}"
          f"{f'{false_now} of {len(stood_through)}, {false_was}':>25s}"
          f"{f'{len(stood)}/{len(up)}, {_span(stood)}':>18s}"
          f"{f'{len(heads)}, {_span(heads)}':>17s}{f'{len(heads_s)}, {_span(heads_s)}':>17s}")
  missed = [r for r in alone if r["fell"] and r["tilt_s"] is None and r["collapse_s"] is None]
  most = max((r["max_tip"] for r in alone if not r["fell"]), default=0.0)
  print(f"  falls neither check took: {len(missed)}; the most a shove it stood through tipped it "
        f"past its posture: {math.degrees(most):.1f} deg, against the watch's "
        f"{math.degrees(FALL_TILT_RAD):.0f}")
  return rows


# The scenes `fall` and `getup`: on a bare floor, the get-up policy taking
# over as the humanoid's mission hands it a falling body (`humanoid.mission`:
# its senses say it is falling, every joint goes to the get-up policy, and
# the standing policy has the legs again once it stands). Staged with the
# truth: the shove's push, and where a lying start puts the body.

#: The get-up policy the scenes fly (`--getup-policy`; the committed one by
#: default).
_GETUP = None
#: The `getup` scene's starts: lying on its back, its front, its side.
LYING = {"on its back": (0.0, -math.pi / 2), "on its front": (0.0, math.pi / 2),
         "on its side": (math.pi / 2, 0.0)}


class Falling:
  """The bare humanoid with its get-up policy beside its standing one, the
  fall handed over as the mission hands it (`humanoid.mission`)."""

  def __init__(self, h: Humanoid, npz: str | None = None):
    self.h = h
    self.policy = StandPolicy(npz or str(GETUP_NPZ))
    self.every = max(1, round(h.stand.policy.period / h.m.opt.timestep))
    self.rebuild()

  def rebuild(self) -> None:
    """Its watch and get-up driver over the body's senses as they stand
    (a fresh standing driver's)."""
    h = self.h
    self.watch = FallWatch(h.stand.trunk, h.stand.joints)
    self.getup = GetupDriver(h.m, h.d, policy=self.policy, trunk=h.stand.trunk,
                             joints=h.stand.joints)
    self.up = False
    self.n = 0

  def take(self) -> None:
    self.up = True
    self.getup.begin()

  def step(self) -> None:
    h = self.h
    if self.up:
      self.getup.step()
      mujoco.mj_step(h.m, h.d)
      if self.getup.stood:
        self.up = False
        h.stand.last_action[:] = 0.0
        h.stand.posture[:] = 0.0
        h.reach = Reach(h.m, h.d, h.stand)
    else:
      h.step()
      if self.n % self.every == 0 and self.watch.falling(h.d, h.reach.held_pitch()):
        self.take()
    self.n += 1


def fall_scene(f: Falling, ways=("front", "back-right", "left")):
  """The `fall` scene, once: shoved over each of `ways` in turn, the get-up
  policy taking the fall and standing the body again."""
  h = f.h
  for way in ways:
    yield "standing"
    for _ in range(int(1.5 / h.m.opt.timestep)):
      f.step()
      yield None
    yield f"shoved {FALL_NS:g} N*s from the {way}"
    for _ in shove(h, FALL_WAYS[way], FALL_NS):
      yield None
    t0, took = h.d.time, False
    while h.d.time - t0 < FALL_S:
      f.step()
      if f.up and not took:
        took = True
        yield f"  falling, {h.d.time - t0:.1f} s on: the get-up policy has every joint"
      if took and not f.up:
        yield f"  stood again {h.d.time - t0:.1f} s after the shove"
        break
      yield None
    else:
      yield "  still down after the shove" if took else "  stood through it"


def getup_scene(f: Falling):
  """The `getup` scene, once: from lying on its back, its front and its
  side, each time stood again by the get-up policy."""
  h = f.h
  for name, (roll, pitch) in LYING.items():
    yield f"lying {name}"
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, 0.0]), "XYZ")
    mujoco.mj_resetDataKeyframe(h.m, h.d, 0)
    h.d.qpos[2], h.d.qpos[3:7] = 0.15, q
    mujoco.mj_forward(h.m, h.d)
    h.stand = StandDriver(h.m, h.d)
    h.reach = Reach(h.m, h.d, h.stand)
    f.rebuild()
    f.take()
    t0 = h.d.time
    while h.d.time - t0 < 10.0:
      f.step()
      if not f.up:
        yield f"  stood in {h.d.time - t0:.1f} s"
        break
      yield None
    else:
      yield "  still down after 10 s"
    for _ in range(int(1.0 / h.m.opt.timestep)):
      f.step()
      yield None


def _run_view(make, scene, what: str) -> None:
  from mujoco import viewer as mj_viewer
  print(what + " Close the window to quit.")
  while True:
    f = make()
    h = f.h
    with mj_viewer.launch_passive(h.m, h.d) as viewer:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = h.root
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 120.0, -15.0
      wall = time.time() - h.d.time
      for line in scene(f):
        if line:
          print(f"  -> {line}")
        if not viewer.is_running():
          return
        if int(round(h.d.time / h.m.opt.timestep)) % 10 == 0:
          viewer.sync()
          ahead = h.d.time - (time.time() - wall)
          if ahead > 0:
            time.sleep(ahead)
      if not viewer.is_running():
        return


def view_fall() -> None:
  _run_view(lambda: Falling(floor_humanoid(), _GETUP), fall_scene,
            "the `fall` scene: the humanoid shoved over three ways, its get-up policy taking "
            "each fall and standing it again.")


def view_getup() -> None:
  _run_view(lambda: Falling(floor_humanoid(), _GETUP), getup_scene,
            "the `getup` scene: the humanoid stood up from lying on its back, its front and "
            "its side.")


def filmstrip_falls(scene: str, out: str, every_s: float = 0.3) -> None:
  """The `fall` or `getup` scene's filmstrip: a tile every `every_s`."""
  from PIL import Image, ImageDraw
  f = Falling(floor_humanoid(), _GETUP)
  h = f.h
  h.m.vis.headlight.ambient[:], h.m.vis.headlight.diffuse[:] = 0.4, 0.6
  r = mujoco.Renderer(h.m, 300, 400)
  tiles, next_t, label = [], 0.0, ""
  for line in (fall_scene(f) if scene == "fall" else getup_scene(f)):
    if line:
      print(line)
      label = line.strip()
      continue
    if h.d.time < next_t - every_s:             # a start's reset set the clock back
      next_t = h.d.time
    if h.d.time >= next_t and len(tiles) < 36:
      next_t = h.d.time + every_s
      cam = mujoco.MjvCamera()
      cam.lookat[:] = h.d.xpos[h.root]
      cam.lookat[2] = max(0.4, cam.lookat[2])
      cam.distance, cam.azimuth, cam.elevation = 3.0, 120.0, -15.0
      r.update_scene(h.d, cam)
      img = Image.fromarray(r.render())
      ImageDraw.Draw(img).text((8, 8), f"{h.d.time:.1f} s: {label}"[:60], fill=(255, 255, 255))
      tiles.append(img)
  r.close()
  strip = Image.new("RGB", (400 * min(4, len(tiles)), 300 * ((len(tiles) + 3) // 4)), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % 4), 300 * (k // 4)))
  strip.save(out)
  print(f"wrote {out}")


# ---- stage 5: whether it can be served --------------------------------------------------
# ONE physics thread steps a served pair (#385), so what decides whether the
# humanoid can be served is its loop's cost: each body through the
# Navigator, its LIDAR matched to its map, its depth camera's low layer, the
# same walks in the served house. Graded by the clock.

#: The walks each robot paces back and forth, (from, to): the first
#: through the hall into the kitchen, the second from the living room into
#: the bedroom.
SPEED_ROUTES = (((0.5, 0.5), (-8.5, 4.0)), ((3.0, -1.0), (1.5, 4.5)))
SPEED_WARM_S = 5.0


def speed_world(n: int, seeing: bool = False, bounds=None):
  """The served house (`humanoid.world.home_spec`: its fixtures, which every
  ray of the robots' LIDARs and depth cameras is cast against) with `n`
  humanoids standing at the starts of `SPEED_ROUTES`: the model, its data
  and the bodies, their maps over `bounds` (the house's own by default;
  `grid_spike.py` grows them)."""
  import json
  from pluggybot.body import body_for
  from pluggybot.humanoid.world import home_spec
  from pluggybot.robot import RobotHandle
  if bounds is None:
    b = json.load(open(HOME_XML.replace(".xml", ".meta.json")))["gridBounds"]
    bounds = (b[0], b[1], b[2], b[3]) if isinstance(b, list) else (b["xMin"], b["yMin"], b["xMax"], b["yMax"])
  handles = [RobotHandle(p) for p in ("", "r2_")[:n]]
  starts = [r[0] for r in SPEED_ROUTES[:n]]
  spec = home_spec(first_at=starts[0], second_at=starts[1] if n > 1 else None)
  model = spec.compile()
  data = mujoco.MjData(model)
  for h in handles:
    _pose(model, data, hm.stand_qpos(), h.prefix)
  walk = StandPolicy(str(SEEING_NPZ)) if seeing else None
  bodies = [body_for(model, data, h, grid_bounds=bounds, walk=walk) for h in handles]
  return model, data, bodies


def _paces(body, route, until: float, data):
  """Back and forth along `route` until sim time `until`."""
  k = 1
  while data.time < until:
    x, y = route[k]
    yield from body.go_to_routine(x, y, timeout=60.0)
    k = 1 - k


def speed_row(n: int, sim_s: float = 60.0, seeing: bool = False,
              profile: bool = False) -> dict:
  """`n` humanoids pacing their routes for `sim_s` after a warm-up: the
  wall seconds a sim second took, and (`profile`) where they went."""
  from pluggybot import tick
  model, data, bodies = speed_world(n, seeing)
  warm = list(zip([b.stepper for b in bodies],
                  [_paces(b, r, SPEED_WARM_S, data) for b, r in zip(bodies, SPEED_ROUTES)]))
  tick.run_many(warm)
  t_sim0 = data.time
  pairs = list(zip([b.stepper for b in bodies],
                   [_paces(b, r, t_sim0 + sim_s, data) for b, r in zip(bodies, SPEED_ROUTES)]))
  prof = None
  if profile:
    import cProfile
    prof = cProfile.Profile()
    prof.enable()
  t0 = time.perf_counter()
  tick.run_many(pairs)
  wall = time.perf_counter() - t0
  if prof is not None:
    import io
    import pstats
    prof.disable()
    out = io.StringIO()
    pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(18)
    print(out.getvalue())
  for b in bodies:
    b.close()
  return {"n": n, "seeing": seeing, "sim_s": data.time - t_sim0,
          "wall_s": wall, "rtf": (data.time - t_sim0) / wall}


def speed_table(sim_s: float = 60.0, profile: bool = False) -> None:
  """--speed: one humanoid and a pair, on the served walker and on the one
  that sees the ground, the same walks in the served house, one after
  another (never two at once: they would share the machine)."""
  print(f"SERVING SPEED, the same walks ({sim_s:g} sim s after {SPEED_WARM_S:g} s):")
  for n, seeing in ((1, False), (2, False), (1, True), (2, True)):
    r = speed_row(n, sim_s, seeing, profile)
    print(f"  humanoid x{r['n']}{' (the walker that sees the ground)' if seeing else ''}: "
          f"{r['sim_s']:.0f} sim s in {r['wall_s']:.0f} wall s, {r['rtf']:.2f}x real time",
          flush=True)


SCENES = {"body": view_body, "stand": lambda: view_stage2("stand"),
          "reach": view_reach, "grasp": lambda: view_stage2("grasp"),
          "pull": lambda: view_stage2("pull"), "bar": lambda: view_stage2("bar"),
          "feel": lambda: view_stage2("feel"),
          "demo": view_demo, "walk": view_walk, "rough": view_rough, "stairs": view_stairs,
          "fall": view_fall, "getup": view_getup}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--view", choices=sorted(SCENES), default=None, metavar="SCENE",
                  help=f"a scene in the viewer: {', '.join(sorted(SCENES))}")
  ap.add_argument("--strip", choices=sorted(SCENES), default=None, metavar="SCENE",
                  help="a scene's filmstrip, headless (--out names it)")
  ap.add_argument("--policy", default=None,
                  help="fly another standing policy's .npz (default models/humanoid_stand.npz)")
  ap.add_argument("--getup-policy", default=None,
                  help="the `fall` and `getup` scenes' get-up policy .npz (default "
                       "models/humanoid_getup.npz; `reach` flies the served one)")
  ap.add_argument("--shove", action="store_true",
                  help="the shoves it stands through, each way, standing and crouched")
  ap.add_argument("--envelope", action="store_true",
                  help="where the hand reaches, floor to over the head, flown balancing")
  ap.add_argument("--hold", action="store_true",
                  help="a locked drawer's handle pulled harder until something gives")
  ap.add_argument("--pulls", action="store_true",
                  help="a force staged at a hand held where the locked handle is, each way, harder "
                       "until the body falls (#544)")
  ap.add_argument("--bar", action="store_true",
                  help="the force move on the locked handle, each way at 10 and 20 N: the force on "
                       "the bar against what the hand read, the postures, how it ended (#547)")
  ap.add_argument("--oracle", action="store_true",
                  help="--bar: the move closed on the bar's true force, not the hand's reading "
                       "(ours: the loop's own floor)")
  ap.add_argument("--feel", action="store_true",
                  help="what the hand's readings, its arm's and the whole body's, make of 20 N "
                       "staged at the grip, each way, still and swaying (#547)")
  ap.add_argument("--handover", nargs="?", type=int, const=8, default=None, metavar="N",
                  help="the walk's hand-over at a standstill, through `step`, at N set-outs (8)")
  ap.add_argument("--demo-table", nargs="?", type=int, const=12, default=None, metavar="N",
                  help="the demo's first N set-outs (12), each flown through Body")
  ap.add_argument("--odometry", action="store_true",
                  help="the served walker's leg odometry against the truth, on a bare floor")
  ap.add_argument("--track", nargs="?", const="", default=None, metavar="NPZ,...",
                  help="the served walker and the one that sees the ground (or these policies) "
                       "asked each command on a bare floor")
  ap.add_argument("--rough", nargs="?", const="", default=None, metavar="NPZ,...",
                  help="the served walker and the one that sees the ground (or these policies) "
                       "over a doorstep, paving, a slope and blocks")
  ap.add_argument("--friction", action="store_true",
                  help="its arm joints' friction calibrated in free air, beside the sheets' "
                       "(--write keeps it in models/humanoid_friction.json)")
  ap.add_argument("--write", action="store_true", help="--friction: keep the table")
  ap.add_argument("--mass", action="store_true", help="the mass budget beside the T2's")
  ap.add_argument("--torque", action="store_true",
                  help="each joint's load: held in our physics, then the estimates")
  ap.add_argument("--thermal", action="store_true", help="the hottest winding, held")
  ap.add_argument("--energy", action="store_true", help="watts by activity, the pack's hours")
  ap.add_argument("--sensors", action="store_true", help="what the head sees from the stand")
  ap.add_argument("--climb", nargs="?", const="", default=None, metavar="NPZ",
                  help="the walker that sees the ground (or this one) up and down flights of five at each "
                       "riser, on the ideal scan and on its own map's")
  ap.add_argument("--fall", nargs="?", const="", default=None, metavar="NPZ",
                  help="shoved over each way: what hits first, how hard, each joint's torque "
                       "and speed, falling on the standing policy, limp, or the get-up's; in "
                       "the house; and from each bow a reach takes, what the watch says")
  ap.add_argument("--speed", nargs="?", type=float, const=60.0, default=None, metavar="SIM_S",
                  help="the loop's cost: one humanoid and a pair, the same walks in the "
                       "house, timed (--profile: where it goes)")
  ap.add_argument("--profile", action="store_true", help="with --speed")
  ap.add_argument("--jobs", type=int, default=4, help="processes for a table's flights")
  ap.add_argument("--out", default="humanoid_spike.png")
  args = ap.parse_args(argv)
  policy = StandPolicy(args.policy) if args.policy else None
  if policy is not None:
    global _POLICY
    _POLICY = policy
  if args.getup_policy:
    global _GETUP
    _GETUP = args.getup_policy
  if args.view:
    SCENES[args.view]()
  elif args.strip == "reach":
    filmstrip_reach(args.out)
  elif args.strip == "demo":
    filmstrip_demo(args.out)
  elif args.strip == "walk":
    filmstrip_walk(args.out)
  elif args.strip == "rough":
    filmstrip_rough(args.out)
  elif args.strip == "stairs":
    filmstrip_stairs(args.out)
  elif args.strip in ("fall", "getup"):
    filmstrip_falls(args.strip, args.out)
  elif args.strip and args.strip != "body":
    filmstrip_stage2(args.strip, args.out)
  elif args.shove:
    shove_table(args.policy, jobs=args.jobs)
  elif args.envelope:
    envelope_table(args.policy, jobs=args.jobs)
  elif args.hold:
    hold_table(policy)
  elif args.pulls:
    pull_table(args.policy, jobs=args.jobs)
  elif args.bar:
    bar_table(args.policy, jobs=args.jobs, oracle=args.oracle)
  elif args.feel:
    feel_table(args.policy, jobs=args.jobs)
  elif args.handover is not None:
    handover_table(args.policy, args.handover, jobs=args.jobs)
  elif args.demo_table is not None:
    demo_table(args.demo_table)
  elif args.rough is not None:
    rough_table([p or None for p in args.rough.split(",")] if args.rough else [None, str(SEEING_NPZ)])
  elif args.odometry:
    odometry_table()
  elif args.track is not None:
    track_table([p or None for p in args.track.split(",")] if args.track else [None, str(SEEING_NPZ)])
  elif args.climb is not None:
    climb_table(args.climb or None, jobs=args.jobs)
    house_climb_table(args.climb or None, jobs=args.jobs)
  elif args.speed is not None:
    speed_table(args.speed, args.profile)
  elif args.fall is not None:
    fall_table(args.fall or None, jobs=args.jobs)
    house_fall_table(args.fall or None, jobs=args.jobs)
    bow_fall_table(args.fall or None, jobs=args.jobs)
  elif args.friction:
    friction_table(write=args.write)
  elif args.mass:
    mass_table()
  elif args.torque:
    torque_table()
    print()
    estimate_table()
    print()
    walking_table()
    print()
    flown_table("STAIRS, FLOWN by the walker that sees the ground up and down the house's flight "
                "on its own map (rule 7's re-read; the stairs' estimates above are superseded):",
                stairs_rows())
    if GETUP_NPZ.exists() or _GETUP:
      print()
      flown_table("GETTING UP, FLOWN by the get-up policy from lying (rule 7's re-read):",
                  getup_rows(_GETUP), kinds=hm.LEG + hm.ARM)
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
