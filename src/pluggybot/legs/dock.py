"""The quadruped's dock (issue #378): a cradle it lies down onto.

A quadruped charges the way it rests: it walks over the dock, stops with
its belly above it, and lies down. Its weight seats the belly on the
cradle's bed; two poles of spring-loaded pins in the bed press up into two
pads on the belly pack's underside. Chamfered faces either side funnel the
belly into the bed as it comes down, so the walk only has to put the belly
within the funnel's mouth. A board of AprilTags ahead of the cradle is what
the walk steers by, and what the robot measures its pose off once it lies
there.

⚠ THE CONTACTS ARE SPRUNG, NOT RIGID: a lying quadruped rests on its belly
and four limp legs, so rigid contacts under it carry whatever the legs
leave them (SimNotes, "The quadruped's dock").

Everything is in the DOCK's frame: the origin on the floor under the docked
robot's body centre, +x the way the docked robot faces (toward the board).
The robot comes in from -x. SimNotes, "The quadruped's dock", has the
tables; `scripts/dock_spike.py` flies them.
"""

from dataclasses import dataclass
import math

import numpy as np

from pathlib import Path

from pluggybot.legs.actuator import BUS_V_NOMINAL
from pluggybot.legs.model import CHOSEN, body_xml
from pluggybot.legs.policy import Twist
from pluggybot.rack.coupling import geom_id, touching
from pluggybot.rack.tags import (DOCK_TAG_IDS, DOCK_TAG_SIZE, TAG_DIR, asset_xml,
                                 plate_half_extent)

#: The contacts (Parts.md, "The dock"): Mill-Max 0858-0-15-20-82-14-11-0, a
#: spring-loaded pin rated 12 A at a 30 C rise (9.6 A derated), 20 mOhm
#: max; its spring 25 g free and 120 g at its rated travel, 1.143 mm of a
#: 2.286 mm stroke; a 1.27 mm plunger. Standing its rated travel proud of
#: the bed, a pin under a belly lying on the bed is at mid-stroke.
PIN_TRAVEL = 0.001143
PIN_STROKE = 0.002286
PIN_FREE_N = 0.025 * 9.81            # 25 g
PIN_MID_N = 0.120 * 9.81             # 120 g
PIN_TIP_R = 0.000635
#: Two a pole, 20 mm apart along the pad: the charge current needs one.
PINS_PER_POLE = 2
PIN_DX = 0.010
#: The charger behind the pins (Parts.md, "The dock"): a 12S Li-ion CC-CV
#: charger, 50.4 V at 5 A -- 1.1C for the P45B pack, under the cell's
#: 13.5 A maximum (its standard charge is 4.5 A). In the sim, the constant
#: current at the pack's nominal voltage; the CV tail is not modelled (nor
#: is the rover's, `power.CHARGE_W`). #387 wires it into the pack.
CHARGE_A = 5.0
CHARGE_W = CHARGE_A * BUS_V_NOMINAL
#: How far the plate's collision box reaches under the floor, where it
#: cannot be seen: MuJoCo's soft contact let a 9 kg belly let go 30 mm up
#: sink through the 6 mm plate onto the floor.
BASE_KEEL = 0.05
#: The pins' contact: the stiffest the 2 ms step integrates (a time constant
#: of two steps) and a near-hard impedance. Measured lying on the bed: the
#: pins at 1.119 of their 1.143 mm, pressing 2.32 N a pole of the part's
#: 2.35; at MuJoCo's default the pad sank into them and they pressed 0.49.
PIN_SOLREF = "0.004 1"
PIN_SOLIMP = "0.99 0.999 0.001"
#: The pole's joint. The armature is a numerical stand-in for the plungers'
#: inertia: at a gram's worth (0.01 kg) a pole chattered against the stiff
#: pin contact -- the criterion flipped 1286 times in 10 s of lying there --
#: and at 0.1 kg it holds every step. Critically damped at that armature.
POLE_ARMATURE = 0.1
POLE_DAMPING = 25.0
#: A plated tip on a plated pad: unpublished, a stated choice between a
#: lubricated contact's ~0.2 and dry gold's >1 (the lie-down does not feel
#: it: 34.8 mm of shift at 0.2, 0.5 and 1.0).
PIN_MU = 0.5


@dataclass(frozen=True)
class DockSpec:
  """The cradle and its tag board. Lengths in metres."""

  #: The base plate the cradle is built on; its top is the bed.
  base_t: float = 0.006
  #: The cradle's length along x.
  length: float = 0.30
  #: The bed's half-width: the belly's 0.06 plus the clearance it sits in.
  bed_half: float = 0.064
  #: The funnel's mouth, half-width: what the walk must put the belly
  #: inside. Bounded by the feet, which pass either side of it.
  mouth_half: float = 0.085
  #: How far the funnel's faces rise above the bed, and their thickness.
  wall_h: float = 0.030
  face_t: float = 0.004
  #: The cradle's friction against the belly's case: its bed and faces are
  #: one UHMW-PE part (0.12-0.2 dry, flown at 0.3). priority="1", or the
  #: pair's MAX is the case's 1.0.
  cradle_mu: float = 0.3
  #: The two poles of pins, at y = +-pin_y: under the belly's pads.
  pin_y: float = 0.030
  #: The tag board: its face this far ahead of the origin, the tags at
  #: (+-tag_y) in two rows. The facing is fitted to the pair's BASELINE, and
  #: the seat is 0.62 m behind the board, so a facing error moves the seat
  #: sideways by 0.62 m x its size: at +-0.07 the believed seat swung +-10 mm
  #: walking in; the nose camera frames +-0.14 standing over it and lying.
  board_x: float = 0.62
  tag_y: float = 0.14
  tag_z: tuple[float, float] = (0.20, 0.34)

  @property
  def bed_z(self) -> float:
    """Height of the bed: where the belly rests."""
    return self.base_t

  @property
  def face_angle(self) -> float:
    """The funnel's faces, from horizontal, rad."""
    return math.atan2(self.wall_h, self.mouth_half - self.bed_half)

  @property
  def outer_half(self) -> float:
    """The dock's half-width over everything, faces' thickness included:
    what the feet walk either side of."""
    return self.mouth_half + self.face_t * math.sin(self.face_angle)


DEFAULT = DockSpec()


def tag_layout(spec: DockSpec = DEFAULT) -> dict[int, tuple[float, float, float]]:
  """Each dock tag's centre in the dock frame, by id: the board's drawing,
  which is what the robot knows of the dock before it sees it."""
  out = {}
  ids = iter(DOCK_TAG_IDS)
  for z in spec.tag_z:
    for y in (spec.tag_y, -spec.tag_y):
      out[next(ids)] = (spec.board_x, y, z)
  return out


def _f(v: float) -> str:
  return f"{v:.6g}"


def _quat_x(angle: float) -> str:
  """A rotation about x as a quaternion: unit-free, whatever a world's
  compiler says about angles."""
  return f"{_f(math.cos(angle / 2))} {_f(math.sin(angle / 2))} 0 0"


def pole_spring() -> tuple[float, float]:
  """(stiffness N/m, springref m) of one pole's slide joint: the pins'
  springs in parallel, a line through the datasheet's two points, with the
  joint's zero where a pin stands free (its upper stop)."""
  k = PINS_PER_POLE * (PIN_MID_N - PIN_FREE_N) / PIN_TRAVEL
  return k, PINS_PER_POLE * PIN_FREE_N / k


def dock_xml(spec: DockSpec = DEFAULT, pos=(0.0, 0.0), yaw_deg: float = 0.0,
             name: str = "dock") -> str:
  """The dock as one MJCF body for a world's worldbody: static, but for its
  two poles of pins, each on a sprung slide joint. Its tag materials are
  `rack.tags.asset_xml(DOCK_TAG_IDS)`'s."""
  s = spec
  g = []
  half_l = s.length / 2
  # The plate's collision box is keeled BASE_KEEL under the floor.
  depth = s.base_t + BASE_KEEL
  mu = f'friction="{_f(s.cradle_mu)} 0.005 0.0001" priority="1"'
  g.append(f'<geom name="{name}_base" type="box" size="{_f(half_l)} {_f(s.outer_half)} '
           f'{_f(depth / 2)}" pos="0 0 {_f(s.base_t - depth / 2)}" contype="2" '
           f'conaffinity="1" {mu} rgba="0.32 0.33 0.36 1"/>')
  k, ref = pole_spring()
  for side, lbl in ((1, "l"), (-1, "r")):
    # A face from the bed's edge up to the mouth, as a thin box whose top
    # surface is that line.
    ang = s.face_angle
    run = math.hypot(s.mouth_half - s.bed_half, s.wall_h)
    t = s.face_t
    ny, nz = -side * math.sin(ang), math.cos(ang)
    cy = side * (s.bed_half + s.mouth_half) / 2 - ny * t / 2
    cz = s.base_t + s.wall_h / 2 - nz * t / 2
    g.append(f'<geom name="{name}_face_{lbl}" type="box" size="{_f(half_l)} '
             f'{_f(run / 2)} {_f(t / 2)}" pos="0 {_f(cy)} {_f(cz)}" '
             f'quat="{_quat_x(side * ang)}" {mu} contype="2" conaffinity="1" '
             f'rgba="0.85 0.85 0.8 1"/>')
    # A pole: its pins' plungers ride one slide joint, the springs in
    # parallel (POLE_ARMATURE, PIN_SOLREF: why each is what it is). ⚠ A pin
    # meets the robot and nothing of its own dock (contype 0): it rides in a
    # hole in the plate, and MuJoCo's parent filter does not apply to a body
    # welded to the world -- the plate held the pins up.
    pins = "".join(
      f'\n        <geom name="{name}_pin_{lbl}{i}" type="sphere" size="{_f(PIN_TIP_R)}" '
      f'pos="{_f(dx)} 0 {_f(PIN_TRAVEL - PIN_TIP_R)}" contype="0" conaffinity="1" '
      f'friction="{_f(PIN_MU)} 0.005 0.0001" priority="2" solref="{PIN_SOLREF}" '
      f'solimp="{PIN_SOLIMP}" rgba="0.85 0.65 0.2 1"/>'
      for i, dx in enumerate((-PIN_DX, PIN_DX)))
    g.append(f'<body name="{name}_pole_{lbl}" pos="0 {_f(side * s.pin_y)} {_f(s.bed_z)}">'
             f'\n        <inertial pos="0 0 0" mass="0.002" diaginertia="1e-8 1e-8 1e-8"/>'
             f'\n        <joint name="{name}_pole_{lbl}" type="slide" axis="0 0 1" '
             f'range="{_f(-PIN_STROKE)} 0" stiffness="{_f(k)}" springref="{_f(ref)}" '
             f'damping="{_f(POLE_DAMPING)}" armature="{_f(POLE_ARMATURE)}"/>'
             f'{pins}\n      </body>')
  # The board on its post, facing -x.
  top = max(s.tag_z) + plate_half_extent(DOCK_TAG_SIZE) + 0.02
  g.append(f'<geom name="{name}_post" type="box" size="0.015 0.015 {_f(top / 2)}" '
           f'pos="{_f(s.board_x + 0.025)} 0 {_f(top / 2)}" rgba="0.32 0.33 0.36 1"/>')
  g.append(f'<geom name="{name}_board" type="box" size="0.005 {_f(s.tag_y + 0.06)} '
           f'{_f((top - min(s.tag_z) + 0.06) / 2)}" '
           f'pos="{_f(s.board_x + 0.005)} 0 {_f((top + min(s.tag_z) - 0.06) / 2)}" '
           f'rgba="0.95 0.95 0.95 1"/>')
  half = plate_half_extent(DOCK_TAG_SIZE)
  for tag_id, (x, y, z) in tag_layout(s).items():
    g.append(f'<geom name="{name}_tag{tag_id}" type="box" size="0.001 {_f(half)} {_f(half)}" '
             f'pos="{_f(x - 0.001)} {_f(y)} {_f(z)}" contype="0" conaffinity="0" '
             f'material="tagmat{tag_id}"/>')
  body = "\n      ".join(g)
  yaw = math.radians(yaw_deg)
  return (f'<body name="{name}" pos="{_f(pos[0])} {_f(pos[1])} 0" '
          f'quat="{_f(math.cos(yaw / 2))} 0 0 {_f(math.sin(yaw / 2))}">\n'
          f'      {body}\n    </body>')


def world_xml(spec: DockSpec = DEFAULT) -> tuple[str, dict[str, bytes]]:
  """#377's body standing at the origin and a dock whose frame is the
  world's, as MJCF and the assets it reads: what the spike and the tests
  fly. The dock's pin poles are joints, so it goes AFTER the robot."""
  xml = body_xml(CHOSEN, assets="\n    " + asset_xml(DOCK_TAG_IDS), after=dock_xml(spec))
  root = Path(__file__).resolve().parents[3]
  assets = {f"tags/tag{i}.png": (root / TAG_DIR / f"tag{i}.png").read_bytes()
            for i in DOCK_TAG_IDS}
  return xml, assets


def dock_charge_contact(model, data, prefix: str = "", name: str = "dock") -> bool:
  """Each belly pad on a pin of its own pole: the dock's ELECTRICAL
  criterion, as `rack_charge_contact` is the rover's. A pad on the other
  pole is a reversed robot, and is not charging. `prefix` names whose
  belly."""
  for lbl in ("l", "r"):
    pad = geom_id(model, f"{prefix}belly_pad_{lbl}")
    pins = [geom_id(model, f"{name}_pin_{lbl}{i}") for i in range(PINS_PER_POLE)]
    if pad is None or None in pins:
      raise KeyError("no belly pads or dock pins in this model")
    if not touching(data, pad, pins):
      return False
  return True


# ---- finding it: the board's tags --------------------------------------------

@dataclass(frozen=True)
class DockFix:
  """The dock's frame in an observer's horizontal frame, fitted to its tags:
  its origin (x, y) and the direction of its +x (yaw)."""

  x: float
  y: float
  yaw: float
  n: int          # tags the fit used
  rms: float      # m; how far the decoded tags sit from the fitted layout


#: A fit needs its tags to span this much of the board, m: its facing comes
#: from the baseline, and a column's two tags are ONE point in the plane --
#: seen alone, they fitted any heading as 0 with a perfect rms.
MIN_BASELINE_M = 0.1
#: ...and to sit where the drawing says within this, m: a misdecoded tag or
#: a range gone wrong is refused, not blended in. Real looks through the
#: approach: 1.6 mm median, 14 mm worst of 80.
MAX_FIT_RMS_M = 0.03


def fit_dock(seen: dict[int, tuple[float, float]],
             spec: DockSpec = DEFAULT) -> DockFix | None:
  """Fit the board's drawing to where its tags were decoded.

  `seen` maps tag id -> the horizontal (x, y) of that tag's decoded
  translation in the observer's frame; ids not on the board are ignored.
  A least-squares 2D rigid fit (Kabsch), so
  the facing comes from the BASELINE between tags and never from one tag's
  PnP yaw, which square-on is a coin flip (issue #88). None without a
  baseline (MIN_BASELINE_M) or beyond MAX_FIT_RMS_M."""
  layout = tag_layout(spec)
  ids = [i for i in seen if i in layout]
  if len(ids) < 2:
    return None
  p = np.array([layout[i][:2] for i in ids], dtype=float)
  q = np.array([seen[i] for i in ids], dtype=float)
  if np.linalg.norm(p - p[0], axis=1).max() < MIN_BASELINE_M:
    return None
  pm, qm = p.mean(axis=0), q.mean(axis=0)
  h = (p - pm).T @ (q - qm)
  yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
  c, s = math.cos(yaw), math.sin(yaw)
  rot = np.array([[c, -s], [s, c]])
  t = qm - rot @ pm
  rms = float(np.sqrt(np.mean(np.sum((q - (p @ rot.T + t)) ** 2, axis=1))))
  if rms > MAX_FIT_RMS_M:
    return None
  return DockFix(float(t[0]), float(t[1]), yaw, len(ids), rms)


def camera_mount(model, data, camera: str,
                 root_body: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The camera on the body, for a decode: `(level, r_mount, p_mount)` --
  its mount in the root's frame, the robot's own kinematics read off the
  model (fixed; never typed), and the rotation that levels the root's frame
  by its roll and pitch, which gravity gives the IMU. A MuJoCo camera-frame
  point v is at `level @ (r_mount @ v + p_mount)` in the HEADING frame;
  `seen_from` and the places' looks (`legs.places`) read tags through it."""
  cam = model.camera(camera).id
  rot = data.xmat[root_body].reshape(3, 3)
  p_mount = rot.T @ (data.cam_xpos[cam] - data.xpos[root_body])
  r_mount = rot.T @ data.cam_xmat[cam].reshape(3, 3)
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  c, s = math.cos(yaw), math.sin(yaw)
  level = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot
  return level, r_mount, p_mount


def seen_from(model, data, detections: dict, camera: str,
              root_body: int) -> dict[int, tuple[float, float]]:
  """Decoded tags as horizontal (x, y) in the body's HEADING frame: origin at
  the root, x forward, levelled by the IMU's gravity (`camera_mount`). A
  detection's "t" is the camera's OpenCV frame (x right, y down, z
  forward)."""
  level, r_mount, p_mount = camera_mount(model, data, camera, root_body)
  out = {}
  for tag_id, det in detections.items():
    tx, ty, tz = det["t"]
    in_cam = np.array([tx, -ty, -tz])      # OpenCV -> MuJoCo's camera frame
    p = level @ (r_mount @ in_cam + p_mount)
    out[tag_id] = (float(p[0]), float(p[1]))
  return out


#: Each look is blended into the dock's believed pose with this weight: the
#: dock does not move, so looks average; the legs' drift between two (0.25 s,
#: a few mm) is what the weight must still follow.
LOOK_WEIGHT = 0.5


def blend(old: tuple[float, float, float] | None, new: tuple[float, float, float],
          weight: float = LOOK_WEIGHT) -> tuple[float, float, float]:
  """A look's pose of the dock folded into the one believed so far."""
  if old is None:
    return new
  dyaw = _wrap(new[2] - old[2])
  return (old[0] + weight * (new[0] - old[0]), old[1] + weight * (new[1] - old[1]),
          _wrap(old[2] + weight * dyaw))


def compose(a: tuple[float, float, float],
            b: tuple[float, float, float]) -> tuple[float, float, float]:
  """Pose `b` (given in `a`'s frame) in the frame `a` is given in."""
  c, s = math.cos(a[2]), math.sin(a[2])
  return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1],
          _wrap(a[2] + b[2]))


def relative(pose: tuple[float, float, float],
             frame: tuple[float, float, float]) -> tuple[float, float, float]:
  """`pose` expressed in `frame` (both in one common frame)."""
  dx, dy = pose[0] - frame[0], pose[1] - frame[1]
  c, s = math.cos(frame[2]), math.sin(frame[2])
  return (c * dx + s * dy, -s * dx + c * dy, _wrap(pose[2] - frame[2]))


def _wrap(a: float) -> float:
  return math.atan2(math.sin(a), math.cos(a))


# ---- walking in -----------------------------------------------------------------

#: Where the approach begins: on the dock's axis, this far behind the seat,
#: facing the board. The board's four tags decode from there.
STANDOFF_M = 1.0
#: Lying down moves the body BACK: #377's scripted lower-and-fold
#: (`quad_spike.lie_down_routine`) folds the legs forward under the belly
#: and the body travels 34.8 mm rearward onto this dock from the policy's
#: stance, from wherever it stood (35.5-36.2 onto the bare floor). So the
#: robot stands this far AHEAD of the seat to lie onto it. Re-measure it
#: with any change to the lie-down.
LIE_SHIFT_M = 0.035
#: The walk in. The policy does not walk below ~0.2 m/s (a dead band,
#: measured: 0.15 m/s commanded walks 5 mm/s), so the approach walks through
#: at APPROACH_V and cuts the command STOP_M short of the seat: from 0.3 m/s
#: the flat policy comes to rest 22 +- 5.5 mm on (eight stops, every phase
#: of the gait).
APPROACH_V = 0.3
STOP_M = 0.022
#: Pure pursuit onto the axis: the robot steers for the point LOOKAHEAD_M
#: ahead of it on the axis, and sidesteps toward the axis as it walks (the
#: policy tracks half a small sideways command while it walks).
LOOKAHEAD_M = 0.25
K_YAW = 2.0
YAW_MAX = 0.5
K_VY = 1.0
VY_MAX = 0.15
#: Facing further off the aim point than this, it turns on the spot first,
#: at TURN_W (above the policy's ~0.2 rad/s dead band).
TURN_FIRST = math.radians(25.0)
TURN_W = 0.4
#: ⚠ OVER THE DOCK IT WALKS STRAIGHT: no sidestep, at most OVER_YAW_MAX of
#: steering, and never a turn on the spot. The nearest stance foot's centre
#: to the body's centreline, measured on the flat policy walking at 0.3 m/s:
#: 124 mm straight, FOOT_TRACK_M steering at 0.1 rad/s -- but 108 mm
#: sidestepping at 0.15 m/s and 74 mm turning on the spot, both inside the
#: dock (`DockSpec.outer_half`). A line-up the walk has not finished by the
#: time the front feet reach the dock is the check's to refuse, not the
#: walk's to finish over it.
OVER_YAW_MAX = 0.1
FOOT_TRACK_M = 0.120
#: How far ahead of its hip a front foot lands, m: 17-20 mm walking at
#: 0.3 m/s, sidestepping, steering or turning on the spot.
FOOT_REACH_M = 0.02
#: ...and it walks over the dock only near its axis: further off than this
#: it stops, and the check backs it out for another run (a start 0.28 m
#: across and 0.18 m close had a foot on a face for 0.42 s). Wider than the
#: feet's 10 mm of clearance on purpose: what it believes of the axis
#: swings a few mm look to look, and a tighter net stopped straight walks.
OVER_ACROSS_M = 0.020

#: How often the walk re-reads the board, s of sim (a render and a decode).
LOOK_EVERY_S = 0.25
#: No board in view: turn on the spot by these steps (deg, so +25, -25,
#: +50, -50, +75, -75 from where it started) and look after each.
SEARCH_STEPS = (25.0, -50.0, 75.0, -100.0, 125.0, -150.0)
#: Standing over the seat, how far off the robot may believe it is and still
#: lie down: well inside the funnel's capture.
LIE_ACROSS_M, LIE_YAW = 0.015, math.radians(6.0)
LIE_ALONG_M = 0.05
#: The walk in gives up after this long, s; a run that did not line up
#: backs out this far and settles before looking again (m, s, s).
WALK_IN_S = 20.0
BACK_OUT_M, BACK_OUT_S, BACK_OUT_SETTLE_S = 0.9, 8.0, 0.8
#: Stopped over the seat, it stands still this long before the look that
#: checks the line-up, s; a search turn settles for its own, s.
STOPPED_S, TURN_SETTLE_S = 0.8, 0.4
#: A search turn ends within this of its heading, or at its budget.
TURN_TOL, TURN_BUDGET_S = math.radians(3.0), 10.0
#: Attempts at the dock before the approach gives up.
TRIES = 3


def over_dock_x(spec: DockSpec = DEFAULT) -> float:
  """Where, in the dock's frame, the robot's front feet can land on the
  dock's rear edge: from here on it walks straight."""
  return -(spec.length / 2 + CHOSEN.hip_x + CHOSEN.foot_r + FOOT_REACH_M)


def walk_in_twist(ex: float, ey: float, eth: float,
                  spec: DockSpec = DEFAULT) -> Twist:
  """The command for one step of the walk in, from where the robot believes
  it stands in the dock's frame. Zero once it is within STOP_M of where it
  lies down from (LIE_SHIFT_M ahead of the seat): the stop is the command's
  last act."""
  if ex >= LIE_SHIFT_M - STOP_M:
    return Twist()
  aim = math.atan2(-ey, LOOKAHEAD_M)
  e = _wrap(aim - eth)
  over = ex > over_dock_x(spec)
  if over and abs(ey) > OVER_ACROSS_M:
    return Twist()
  if abs(e) > TURN_FIRST:
    return Twist() if over else Twist(yaw_rate=math.copysign(TURN_W, e))
  if over:
    return Twist(vx=APPROACH_V,
                 yaw_rate=max(-OVER_YAW_MAX, min(OVER_YAW_MAX, K_YAW * e)))
  vy = max(-VY_MAX, min(VY_MAX, -K_VY * ey))
  w = max(-YAW_MAX, min(YAW_MAX, K_YAW * e))
  return Twist(vx=APPROACH_V, vy=vy, yaw_rate=w)


if __name__ == "__main__":
  from pluggybot.rack.tags import write_tag_pngs
  print(f"wrote tags {write_tag_pngs(ids=DOCK_TAG_IDS)} to models/tags")
