"""The probe (issue #466, stage 2): the robot lifts a box's lid by its drop
handle from its own senses, and records what it sent its arm and what it
sensed -- the imagination's input (`imagination.record`). Nothing in it
reads the world's model of the box.

CODE'S FORESIGHT, stated -- stage 3's model of the box replaces it: the box
in front is lidded, its lid hinged along the back edge of its top, and the
knob that lifts it -- a cube the claw is proven on, its tag
(`rack.tags.CHEST_TAG_IDS`) on every face as the cubes' are -- hangs on a
drop handle from a pin at the tip of the bracket out of the lid's front
edge over it. Held, the knob moves as that pin does: planned as a point
fixed to the lid, it would be 66 mm off the pin's arc at the top
(`tests/test_probe.py`), past the 40 mm Kp 8 forgives (#469). Everything
else is MEASURED: the knob off its tag (PnP, `Probe.look`), and the top,
its back edge (the hinge) and the bracket's tip (the pin) off the D435's
depth at the part's own 848 x 480 (`guess`, over `perception.box`). The
hinge is the top's height, a board's thickness over the true one, and the
pin the bracket's top, half the bracket over it: what foresight costs.

THE STEPS (`Probe.routine`), from the robot's senses alone:
  find       the knob's tag in the colour imager, and the box off depth,
             standing where it starts
  calibrate  lying there, the arm sweeps the probe's arc in the air -- no
             point seen standing under it -- and its friction is read off its
             torques (`legs.arm.arm_friction`): once per robot, anywhere
             free (#469)
  walk in    to a look point on the box's axis, then steered by the knob's
             tag to `MEASURE_AT_M` off it, facing along the axis
  measure    standing still there: the knob and the box. Lying, neither
             imager sees the knob or the top -- the D435 sits 0.10 m over
             the floor and the knob hangs at 0.09
  lie        the last 12.5 cm on its belief (`STAND_AT_M`), down, and the box
             re-found by its front face, which the D435 still sees lying:
             the measures moved by what the walk and the lie-down did to the
             belief (`front_face`, `Move`; up to 25 mm), the path planned in
             that belief and kept (`Frame`). Out of reach, it stands, backs
             out and walks in again
  take hold  the jaws onto the knob once the claw hangs still, clear of the
             bracket (`TAKE_CLEAR_M`), at Kp 8 from here on (#469's gain: a
             hinge guessed within 4 cm keeps the claw seated)
  sweep      the lid up and down at two rates (`RATES`), the first pulling
             its catch off -- the record ends here
  let go     the jaws open and the arm let settle at Kp 8, then its working
             gains, the claw up and back, and onto its feet

THE RECORD (`Recorder`, a step hook): a row a physics step of what was SENT
-- the arm drivers' targets as they applied them, their gains, the claw's
slide and jaws -- and what was SENSED -- the drivers' torque readings
(`perception.encoders.torque_reading`, as `Body.arm_torque` reads them) and
the encoders in the MIT reply's field. Its start is the robot as it knew
itself, and its depth and detections are what the last measures were made
of, laid in the map as the robot knew it at the record's start. ⚠ NOTHING
HERE READS THE WORLD'S MODEL OF THE BOX -- no element of it by name, no grip
judged off the world (`ClawHand.held` is our grading's); `tests/test_probe.py`
walks the imports and the source. The claw is already on the fork: a real
run fetches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from pluggybot.imagination.record import Record, Start, plain
from pluggybot.legs import arm as am
from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.legs.model import CHOSEN, JOINT_NAMES
from pluggybot.perception import box
from pluggybot.perception.encoders import (LEG_POSITION_LSB, LEG_VELOCITY_LSB, quantised,
                                           torque_reading)
from pluggybot.rack.tags import CHEST_TAG_IDS
from pluggybot.tick import Routine
from pluggybot.tools import claw as tc
from pluggybot.tools import servo

# ---- the plan (#469's; `evaluation.imagined`'s oracle flies the same) -------------

#: The knob's tag, on every face of the knob.
KNOB_TAG = CHEST_TAG_IDS[0]
#: Lying, the knob this far ahead of the torso's centre (#469: the lid's arc
#: carries it 0.16 m further and 0.22 m up, and with it 0.60 m ahead the
#: arm's reach stopped a lid at 66 deg)...
KNOB_AHEAD_M = 0.46
#: ...so the walk-in stands this far off it: lying moves the body back
#: (`legs.dock.LIE_SHIFT_M`). The claw's walk-in stands a cube's distance
#: off (`legs.claw.STAND_AT_M`), 8 cm further.
STAND_AT_M = KNOB_AHEAD_M - dk.LIE_SHIFT_M
#: A look point this far off the knob on the box's axis: inside the colour
#: imager's window on a cube's tag standing (0.5-0.9 m, `legs.claw`).
LOOK_M = 0.85
#: It measures standing this far off the knob, then walks the rest of the
#: way to `STAND_AT_M` on its belief: from there the knob sits at the
#: bottom of the colour frame -- v 643-690 of 720 -- where a stance 15 mm
#: short cut its tag in half and no look read it, and the walk in steered on
#: looks gone stale and lay 5 cm off the axis. Here it sits mid-frame.
MEASURE_AT_M = 0.55
#: The probe's gain, and its damping kept at the working gains' ratio.
KP = 8.0
KD = am.ARM_KD * math.sqrt(KP / am.ARM_KP)
#: The sweeps (#469's `--fit`): up to `TOP` and back to `BOTTOM` (rad of
#: lid), at each rate (rad/s), held `TURN_S` at each end. ⚠ A chest is
#: drawn to shut over these and no faster or higher (`activity.chest
#: .SWEPT_TO`, `SWEPT_RATE`; pinned in `tests/test_probe.py`): a drop
#: handle only pulls.
RATES = (0.15, 0.45)
TOP, BOTTOM, TURN_S = 1.2, 0.05, 0.4
#: The jaws take the knob this far in front of the bracket's tip at most,
#: m: the claw's crossbar is 8 mm long either side of the jaws' middle
#: (`legs.rack.claw_face`), and 6 mm more. ⚠ A DROP HANDLE RESTS WITH ITS
#: CENTRE OF MASS UNDER ITS PIN, swung 37.6 deg, its knob then 9.6 mm in
#: front of the tip: straight down onto it the crossbar landed on the
#: bracket 42 mm over the knob, and the jaws shut on the handle's arm. The
#: grip takes the knob a few millimetres off its middle instead.
TAKE_CLEAR_M = 0.014
#: Over the knob before going down onto it, m, and how long the claw hangs
#: there first (#469: straight down after a short move it swung 3 deg on
#: its peg, and its crossbar met the handle's bracket).
OVER_M = 0.045
SWING_SETTLE_S = 1.0
#: Let go, the arm settles this long at the probe's gain before its working
#: gains come back: at Kp 8 under a lid's pull it trails its target by up to
#: a tenth of a radian, which Kp 60 would answer with a kick of 6 N*m.
RELEASE_S = 0.5

# ---- the senses -----------------------------------------------------------------

#: The probe's depth frames: the D435's own 848 x 480 (the served body's
#: 120 x 70 is a seventh of it, for the cost of ten a second). From the
#: stance the top's back edge is laid within a millimetre (SimNotes, "The
#: probe from the robot's own senses"); at 120 x 70 the back 3 cm of the top
#: went unsampled.
DEPTH_W, DEPTH_H = 848, 480
#: ...and that stream's nearest depth, m: Intel's tuning guide puts it at
#: the focal length (px) times the baseline over the 126 disparities the
#: ASIC searches, 17 cm at 848 x 480 (the served camera's 0.28 is 1280 x
#: 720's).
DEPTH_MIN_Z = 0.17
#: Its noise has a stream of its own (`perception.depth`'s seed).
DEPTH_SEED = 466
#: The measures: this many looks and depth frames, this far apart (s),
#: standing still.
LOOKS, FRAMES, APART_S = 5, 3, 0.1
#: At the measuring stance it stands this long first, s: the IMU's tilt
#: takes its time constant to forget a walk (`perception.imu.TAU_S`) --
#: MEASURED, its pitch 0.38 deg off a second after a walk stopped, 0.18 at
#: two, 0.10 at three: at the top's back edge 4, 2 and 1 mm. Its legs'
#: reckoning walks about 1 mm a second while it stands.
STILL_S = 2.0
#: What a guess reads: the points within this of the knob, m (the box is
#: 0.3 m wide and the knob hangs 0.1 m in front of it).
NEAR_M = 0.45
#: The bracket: the top's points within this of the knob across (m), more
#: than this in front of the lid's front edge; the lid's front edge off the
#: top's points further across than `BESIDE_M`.
TONGUE_HALF_M, TONGUE_GAP_M, BESIDE_M = 0.015, 0.004, 0.03
#: The bracket's tip is read with its face toward the camera
#: (`perception.box.faced_edge`): the D435's noise there, 0.32 m off from
#: the measuring stance, is 0.4 mm.
#: A guess whose pin is not this far off its hinge is no lid, m.
RADIUS_M = (0.10, 0.60)
#: The box's front face, as both stances see it: its points this high over
#: the floor (lying, the D435 sees no higher than 0.09 m on it), clear of
#: the knob and its handle across, within this of its place along (m).
FACE_Z = (0.02, 0.085)
FACE_CLEAR_M, FACE_BAND_M = 0.035, 0.006
#: Lying, it re-finds the box by its front face in this many frames.
LYING_FRAMES = 2
#: The record's clouds keep the points within this of the knob across the
#: floor and under this height, m: the box and what hangs off it.
KEEP_M, KEEP_UNDER_M = 0.40, 0.40
#: The encoders as the MIT reply carries them: the arm's GIM8108-8s report
#: in the legs' fields (`perception.encoders`).
POSITION_LSB, VELOCITY_LSB = LEG_POSITION_LSB, LEG_VELOCITY_LSB

# ---- the walk ---------------------------------------------------------------------

#: The walk to a stance, its looks, its tries; the stand before a measure;
#: the walk back out of a stance that missed (s, m) -- the claw's.
WALK_IN_S, LOOK_EVERY_S, TRIES, STOPPED_S = 20.0, 0.4, 3, 0.8
BACK_OUT_M, BACK_OUT_S = 0.45, 6.0
#: A look point within this of where it stands is where it stands, m.
AT_LOOK_M = 0.15
#: Looking for the knob's tag: where it stands, then turned this far each
#: way (deg).
SWEEP_DEG = (30.0, -60.0)
#: Lying, the knob must be inside this band ahead and across, m, and its
#: path within the arm's reach and the slide's stroke.
REACH_X = (0.40, 0.52)
REACH_Y = rk.CLAW_TRAVEL - 0.015
#: How the path's reach is checked: this many points along it.
PATH_CHECKS = 25
#: The probe's whole budget, sim s.
PATIENCE_S = 300.0

# ---- calibrating -----------------------------------------------------------------

#: The calibration flies the probe's arc at its slower rate with the jaws
#: shut, the knob's place `KNOB_AHEAD_M` ahead, in the air: the floor under
#: its footprint (`FREE_HALF_M` across, `FREE_PAD_M` past either end) must
#: hold no point SEEN standing higher than `FREE_OVER_M`. ⚠ Unseen is not
#: bare: standing, the D435 sees the floor there from 0.41 m ahead, and a
#: thing up to 0.12 m tall could stand unseen in the pad nearer; the claw
#: comes down over it no lower than 8 cm above that (set-out 0, measured).
#: A nearer stance, another camera pitch or a longer pad moves both.
FREE_HALF_M, FREE_PAD_M, FREE_OVER_M = 0.12, 0.08, 0.02

# ---- the record's own check ------------------------------------------------------

#: The probe's own word that it held the lid: the force at the tool down,
#: off its torques, its mean over the sweeps up past the catch at least this
#: (N) -- a drawn lid weighs 1.3 N at the knob or more (#469).
HELD_DOWN_N = 0.4
#: ...read past this long into each sweep up, s: the catch's pull first.
HELD_AFTER_S = 1.0


def rot_about(axis, angle: float) -> np.ndarray:
  """The rotation by `angle` (rad) about the unit `axis` (Rodrigues)."""
  a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
  return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def pin_path(knob, pin, hinge, axis):
  """The knob's middle with the lid at `s` (rad), in whatever frame the
  points are: held, it moves as its pin does about the hinge's line
  (through `hinge`, along `axis`), and does not turn."""
  r = np.subtract(pin, hinge)
  k0 = np.asarray(knob, dtype=float)

  def path(s: float) -> np.ndarray:
    return k0 + rot_about(axis, s) @ r - r
  return path


@dataclass(frozen=True)
class Guess:
  """What the robot makes of the box (the module docstring's foresight),
  in the map frame: the knob where it hangs, the box's facing (`yaw`, +x
  into the box), its top's height, a point on the hinge's line and the
  pin. `seen` is what each was read off, counted."""
  knob: tuple[float, float, float]
  yaw: float
  top: float
  hinge: tuple[float, float, float]
  pin: tuple[float, float, float]
  seen: dict = field(default_factory=dict)

  @property
  def axis(self) -> np.ndarray:
    """The hinge's axis: across the box, to its left seen from the front --
    the turn about it that lifts the lid's front edge is +."""
    return np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])

  @property
  def radius(self) -> float:
    """The pin's distance off the hinge's line, m."""
    r = np.subtract(self.pin, self.hinge)
    return float(np.linalg.norm(r - (r @ self.axis) * self.axis))

  def path(self, s: float) -> np.ndarray:
    """The knob's middle with the lid at `s` (rad), map frame."""
    return pin_path(self.knob, self.pin, self.hinge, self.axis)(s)

  def lid_at(self, knob) -> float:
    """The lid's angle that puts the knob at `knob` (map frame), rad: the
    turn of the pin's offset off the hinge that `knob` implies."""
    a = self.axis
    r = np.subtract(self.pin, self.hinge)
    q = np.asarray(knob, dtype=float) - np.asarray(self.knob) + r
    r, q = r - (r @ a) * a, q - (q @ a) * a
    return math.atan2(float(a @ np.cross(r, q)), float(r @ q))

  def as_dict(self) -> dict:
    return {"knob": list(self.knob), "yaw": self.yaw, "top": self.top,
            "hinge": list(self.hinge), "pin": list(self.pin), "radius": self.radius,
            "seen": dict(self.seen)}

  @classmethod
  def from_dict(cls, d: dict) -> "Guess":
    return cls(knob=tuple(d["knob"]), yaw=d["yaw"], top=d["top"], hinge=tuple(d["hinge"]),
               pin=tuple(d["pin"]), seen=dict(d.get("seen", {})))


def guess(cloud: np.ndarray, knob, hint: float) -> Guess | str:
  """The box off depth points (k, 3, map frame) and the knob where it
  hangs (map), `hint` the direction it was seen in (rad): a `Guess`, or why
  there is none. The foresight is the module docstring's: the hinge is the
  top's back edge, the pin the tip of the top's tongue over the knob."""
  pts = np.asarray(cloud, dtype=float)
  k = np.asarray(knob, dtype=float)
  near = pts[np.hypot(pts[:, 0] - k[0], pts[:, 1] - k[1]) < NEAR_M]
  top = box.top_height(near)
  if top is None:
    return "no top seen"
  on = near[box.on_top(near, top)]
  yaw = box.facing(on[:, :2], hint)
  if yaw is None:
    return "too little of the top seen"
  u = np.array([math.cos(yaw), math.sin(yaw)])
  v = np.array([-u[1], u[0]])
  rel = on[:, :2] - k[:2]
  pu, pv = rel @ u, rel @ v
  far = box.edge(pu, far=True)
  front = box.edge(pu[np.abs(pv) > BESIDE_M], far=False)
  if far is None or front is None:
    return "the top's edges not seen"
  tongue = (np.abs(pv) < TONGUE_HALF_M) & (pu < front - TONGUE_GAP_M)
  tip = box.faced_edge(pu[tongue], far=False)
  if tip is None:
    return "no handle's bracket seen"
  at = k[:2] + float(pv[tongue].mean()) * v
  g = Guess(knob=tuple(float(x) for x in k), yaw=float(yaw), top=float(top),
            hinge=(*(float(x) for x in at + far * u), float(top)),
            pin=(*(float(x) for x in at + tip * u), float(on[tongue, 2].mean())),
            seen={"near": int(len(near)), "top": int(len(on)), "tongue": int(tongue.sum())})
  if not RADIUS_M[0] <= g.radius <= RADIUS_M[1]:
    return f"no lid: its pin {g.radius:.2f} m off its hinge"
  return g


def front_face(cloud: np.ndarray, g: Guess) -> tuple[float, float, float] | None:
  """The box's front face off a cloud (map), in the guess's frame (origin
  the knob, x into the box): its place along, its middle across, and its
  turn off the guess's facing (rad). The face is the most points at one
  place along between the knob and the box's back, at `FACE_Z`, clear of
  the knob across. None where too little of it is seen."""
  c, s = math.cos(g.yaw), math.sin(g.yaw)
  rel = np.asarray(cloud, dtype=float)[:, :2] - np.asarray(g.knob[:2])
  pu, pv = rel @ np.array([c, s]), rel @ np.array([-s, c])
  z = np.asarray(cloud)[:, 2]
  far = float((np.subtract(g.hinge, g.knob)[:2]) @ np.array([c, s]))
  band = ((z > FACE_Z[0]) & (z < FACE_Z[1]) & (np.abs(pv) > FACE_CLEAR_M)
          & (np.abs(pv) < NEAR_M) & (pu > 0.0) & (pu < far))
  if band.sum() < box.MIN_POINTS:
    return None
  counts, edges = np.histogram(pu[band], np.arange(0.0, far + 0.002, 0.002))
  mode = float(edges[int(np.argmax(counts))]) + 0.001
  sel = band & (np.abs(pu - mode) < FACE_BAND_M)
  got = box.face(pu[sel], pv[sel])
  if got is None:
    return None
  along, across, slope = got
  return along, across, -math.atan(slope)


@dataclass(frozen=True)
class Move:
  """How far the box seems to have moved in the map between two looks at
  its front face (`front_face`'s, in guess `g`'s frame): the belief's own
  drift, undone by moving what the first look measured (`point`,
  `guess`)."""
  g: Guess
  before: tuple[float, float, float]
  after: tuple[float, float, float]

  @property
  def turn(self) -> float:
    return self.after[2] - self.before[2]

  def point(self, p) -> np.ndarray:
    """Map points (..., 2 or 3) as the second look lays them."""
    p = np.array(p, dtype=float)
    c, s = math.cos(self.g.yaw), math.sin(self.g.yaw)
    u, v = np.array([c, s]), np.array([-s, c])
    k = np.asarray(self.g.knob[:2])
    rel = p[..., :2] - k
    pu, pv = rel @ u - self.before[0], rel @ v - self.before[1]
    ct, st = math.cos(self.turn), math.sin(self.turn)
    nu, nv = ct * pu - st * pv + self.after[0], st * pu + ct * pv + self.after[1]
    p[..., 0] = k[0] + nu * u[0] + nv * v[0]
    p[..., 1] = k[1] + nu * u[1] + nv * v[1]
    return p

  def guess(self) -> Guess:
    g = self.g
    return Guess(knob=tuple(self.point(g.knob)), yaw=g.yaw + self.turn, top=g.top,
                 hinge=tuple(self.point(g.hinge)), pin=tuple(self.point(g.pin)),
                 seen=dict(g.seen))

  def pose(self, pose) -> list[float]:
    x, y = self.point(pose[:2])
    return [float(x), float(y), float(pose[2] + self.turn)]

  def as_dict(self) -> dict:
    """Along and across the box (mm) and the turn (deg)."""
    return {"alongMm": round(1000 * (self.after[0] - self.before[0]), 2),
            "acrossMm": round(1000 * (self.after[1] - self.before[1]), 2),
            "turnDeg": round(math.degrees(self.turn), 3)}


@dataclass(frozen=True)
class Frame:
  """The robot's HEADING frame (`tools.claw`'s: the torso's centre, x
  ahead, levelled) as it believed it at one moment: where it stood in the
  map, its heading, and the torso's centre over the floor. A plan is made
  in one and kept: the robot lies still while it works."""
  x: float
  y: float
  yaw: float
  floor: float

  def of(self, p_map) -> np.ndarray:
    """Map points (..., 3) in this frame."""
    p = np.asarray(p_map, dtype=float)
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    dx, dy = p[..., 0] - self.x, p[..., 1] - self.y
    return np.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - self.floor], axis=-1)

  def turn(self, v) -> np.ndarray:
    """A map direction in this frame."""
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], dtype=float)


# ---- the record ---------------------------------------------------------------------

def attitude(mis) -> tuple[float, float]:
  """Roll and pitch as the IMU says them (`perception.imu.Attitude`), in
  `legs.imagined`'s order: yaw, then pitch, then roll."""
  r = mis.odo.att.matrix()
  return (math.atan2(float(r[2, 1]), float(r[2, 2])),
          -math.asin(max(-1.0, min(1.0, float(r[2, 0])))))


def start_of(mis) -> Start:
  """The robot as it knows itself: its belief, its IMU's tilt, its legs'
  and its arm's encoders (the arm's as its drivers run them: the shoulder
  and the forearm's absolute angle), and the module on its fork."""
  m, d = mis.model, mis.data
  legs = [float(d.qpos[m.jnt_qposadr[m.joint(mis.handle.el(n)).id]]) for n in JOINT_NAMES]
  return Start(pose=tuple(float(v) for v in mis.pose), attitude=attitude(mis),
               legs=tuple(float(v) for v in quantised(legs, POSITION_LSB)),
               arm=tuple(float(v) for v in quantised(mis.arm.q(), POSITION_LSB)),
               carrying=mis.carrying)


class Recorder:
  """What the robot sent its arm and what it sensed, a row a physics step
  (the module docstring), while it is hooked on the mission's seam: after
  each step the arm driver's target is the one it applied. Beside the
  record it keeps the arm's weight as its own senses put it
  (`legs.arm.ArmDriver.gravity_sensed`) and the encoders' rates, row by
  row: what the force at the tool is read off (`force`)."""

  def __init__(self, mis) -> None:
    self.mis = mis
    hand = tc.ClawHand(mis)
    self.slide, self.jaw = hand.slide, hand.jaws[0]
    self.acts = tuple(mis.arm.act)
    self.keys = tuple(f"{mis.handle.prefix}{a}" for a in self.acts)
    self.commands: list[tuple] = []
    self.sensed: list[tuple] = []
    self.gravity: list[np.ndarray] = []
    self.rates: list[np.ndarray] = []
    self.marks: dict[str, int] = {}
    self.start: Start | None = None
    self.t0: float | None = None

  def __len__(self) -> int:
    return len(self.commands)

  def begin(self) -> None:
    self.start = start_of(self.mis)
    self.t0 = float(self.mis.data.time)
    self.mis.step_hooks.append(self._row)

  def mark(self, phase: str) -> None:
    """A phase begins at the next row."""
    self.marks[phase] = len(self.commands)

  def stop(self) -> None:
    if self._row in self.mis.step_hooks:
      self.mis.step_hooks.remove(self._row)

  def _row(self) -> None:
    mis, d = self.mis, self.mis.data
    arm = mis.arm
    step = int(round(float(d.time) / float(mis.model.opt.timestep)))
    self.commands.append((float(arm.target[0]), float(arm.target[1]), float(arm.kp),
                          float(arm.kd), float(d.ctrl[self.slide]), float(d.ctrl[self.jaw])))
    tau = [torque_reading(float(d.actuator_force[a]), key, step)
           for a, key in zip(self.acts, self.keys)]
    q = quantised(arm.q(), POSITION_LSB)
    self.sensed.append((*tau, *(float(v) for v in q)))
    self.gravity.append(arm.gravity_sensed(q, mis.odo.att.level()))
    self.rates.append(quantised(arm.qd(), VELOCITY_LSB))

  def record(self, depth=(), detections=()) -> Record:
    self.stop()
    return Record(start=self.start, dt=float(self.mis.model.opt.timestep),
                  commands=np.array(self.commands), sensed=np.array(self.sensed),
                  depth=tuple(depth), detections=tuple(detections))

  def phases(self) -> dict[str, tuple[int, int]]:
    """Each phase's rows, [first, past the last)."""
    names = list(self.marks)
    ends = [self.marks[n] for n in names[1:]] + [len(self.commands)]
    return {n: (self.marks[n], e) for n, e in zip(names, ends)}

  def beyond(self) -> tuple[np.ndarray, np.ndarray]:
    """What the motors read beyond the arm's own weight, and the encoders'
    rates: `legs.arm.arm_friction`'s and `less_friction`'s input."""
    s = np.array(self.sensed)
    return s[:, :2] - np.array(self.gravity), np.array(self.rates)

  def force(self, friction) -> np.ndarray:
    """The force at the tool, N, torso (forward, up), row by row: the
    readings less the arm's weight and, where a joint turns, its friction
    (`legs.arm.tool_force`)."""
    beyond, rates = self.beyond()
    tau = am.less_friction(beyond, rates, friction)
    s = np.array(self.sensed)
    spec = self.mis.arm_spec
    return np.array([am.tool_force(spec, a, b, c, e)
                     for a, b, (c, e) in zip(s[:, 2], s[:, 3], tau)])


@dataclass
class Probed:
  """A probe, done or stopped: `ok` once it took hold and swept the lid and
  its torques say it held it, else `why` not. `record` is the hold and the
  sweeps, its `phases` rows [first, past last); `calibration` the empty
  sweep and `friction` read off it; `guess` the measures it planned off and
  `frame` the belief it planned in; `t0` the sim time of the record's first
  row; `log` how each step went."""
  ok: bool = False
  why: str = ""
  record: Record | None = None
  phases: dict = field(default_factory=dict)
  calibration: Record | None = None
  friction: tuple[float, float] | None = None
  guess: Guess | None = None
  frame: Frame | None = None
  t0: float | None = None
  log: dict = field(default_factory=dict)

  def to_wire(self) -> tuple[dict, dict[str, np.ndarray]]:
    """A JSON-ready header and named float arrays (`imagination.worker.pack`'s)."""
    head = {"ok": self.ok, "why": self.why,
            "phases": {k: list(v) for k, v in self.phases.items()},
            "friction": None if self.friction is None else list(self.friction),
            "guess": None if self.guess is None else self.guess.as_dict(),
            "frame": None if self.frame is None else [self.frame.x, self.frame.y,
                                                      self.frame.yaw, self.frame.floor],
            "t0": self.t0, "log": plain(self.log, "the log")}
    arrays = {}
    for name, rec in (("record", self.record), ("calibration", self.calibration)):
      if rec is None:
        continue
      h, a = rec.to_wire()
      head[name] = h
      arrays.update({f"{name}.{k}": v for k, v in a.items()})
    return head, arrays

  @classmethod
  def from_wire(cls, head: dict, arrays: dict) -> "Probed":
    recs = {}
    for name in ("record", "calibration"):
      if name in head:
        mine = {k.split(".", 1)[1]: v for k, v in arrays.items() if k.startswith(f"{name}.")}
        recs[name] = Record.from_wire(head[name], mine)
    g, f = head.get("guess"), head.get("frame")
    return cls(ok=head["ok"], why=head["why"], record=recs.get("record"),
               phases={k: tuple(v) for k, v in head["phases"].items()},
               calibration=recs.get("calibration"),
               friction=None if head["friction"] is None else tuple(head["friction"]),
               guess=None if g is None else Guess.from_dict(g),
               frame=None if f is None else Frame(*f), t0=head["t0"], log=head["log"])


def level_of(attitude) -> np.ndarray:
  """Level from the torso, off a roll and a pitch (`attitude`)."""
  roll, pitch = attitude
  cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
  return (np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
          @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))


def jaws_seen(record: Record, frame: Frame, spec) -> np.ndarray:
  """The jaws' middle each row, map frame, as the robot can tell it: off
  its arm's encoders through its kinematics, straight under the seated peg
  (a module hangs plumb), the slide's command across, levelled by its
  start's tilt and laid in the map through the belief it planned in. ⚠ The
  claw swings on its peg -- +-4 deg, 12 mm at the jaws (#469) -- and this
  reads the swing as where the jaws are."""
  qs, qf = record.sensed[:, 2], record.sensed[:, 3]
  vx = (spec.shoulder_x + spec.upper * np.cos(qs) + spec.fore * np.cos(qf)
        + spec.fork.vertex_x)
  vz = (spec.shoulder_z + spec.upper * np.sin(qs) + spec.fore * np.sin(qf)
        + spec.fork.vertex_z + spec.fork.seat_rise())
  lv = level_of(record.start.attitude)
  seat = np.column_stack([vx, -record.commands[:, 4], vz]) @ lv.T
  seat[:, 2] -= rk.CLAW_JAW_DROP
  c, s = math.cos(frame.yaw), math.sin(frame.yaw)
  return np.column_stack([frame.x + c * seat[:, 0] - s * seat[:, 1],
                          frame.y + s * seat[:, 0] + c * seat[:, 1], seat[:, 2] + frame.floor])


def lid_from_arm(probed: "Probed", spec) -> np.ndarray:
  """The lid's angle each row of the record, rad, as the robot can tell it
  from its arm: the jaws where it believes them (`jaws_seen`), turned back
  into a lid's angle about its guessed hinge (`Guess.lid_at`), the knob
  held where it was taken."""
  jaws = jaws_seen(probed.record, probed.frame, spec)
  a, b = probed.phases["take"]
  g = probed.guess
  held = Guess(knob=tuple(jaws[b - 1]), yaw=g.yaw, top=g.top, hinge=g.hinge, pin=g.pin)
  return np.array([held.lid_at(p) for p in jaws])


# ---- the probe -------------------------------------------------------------------

class Probe:
  """The probe on one robot (`legs.body.QuadMission`, the claw on its fork):
  `routine()` is the module docstring's steps, returning a `Probed`."""

  def __init__(self, mis, seed: int = DEPTH_SEED) -> None:
    from pluggybot.perception.depth import DepthCamera
    self.mis = mis
    self.depth = DepthCamera(mis.model, mis.handle, width=DEPTH_W, height=DEPTH_H,
                             min_z=DEPTH_MIN_Z, mount="body", seed=seed)
    self.depth.carry(mis.carrying)
    #: What the measures at the stance were made of: the record's.
    self.detections: list[dict] = []
    self.clouds: list[np.ndarray] = []

  # ---- the senses, into the map ------------------------------------------------

  def floor(self) -> float:
    """The torso's centre over the floor, as its own senses put it: lying,
    its belly's depth; standing, its legs' (`LegOdometry.height`: the
    encoders, levelled by the IMU)."""
    from pluggybot.legs.body import LYING
    mis = self.mis
    return CHOSEN.belly_depth if mis.posture == LYING else mis.odo.height()

  def frame_now(self) -> Frame:
    mis = self.mis
    return Frame(*(float(v) for v in mis.pose), self.floor())

  def _to_map(self, pts: np.ndarray) -> np.ndarray:
    """Torso-frame points into the map through the belief: levelled by the
    IMU, over the floor by the legs (or the belly lying), then the believed
    heading and place."""
    mis = self.mis
    p = np.asarray(pts, dtype=float) @ mis.odo.att.level().T
    x, y, th = mis.pose
    c, s = math.cos(th), math.sin(th)
    return np.column_stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1],
                            p[:, 2] + self.floor()])

  def look(self) -> list[np.ndarray]:
    """One decode of the colour imager: each face of the knob's tag it read,
    the knob's middle in the map (its face's middle, a half-cube in along
    its normal). Each decode is kept for the record."""
    mis = self.mis
    det = mis.color_detector()
    _, r_mount, p_mount = dk.camera_mount(mis.model, mis.data, mis.handle.el("color_eye"),
                                          mis.root)
    out = []
    for d in det.detect_all(mis.data):
      if int(d["id"]) != KNOB_TAG:
        continue
      tx, ty, tz = d["t"]
      nx, ny, nz = d["normal"]
      centre = (r_mount @ np.array([tx, -ty, -tz]) + p_mount
                + tc.CUBE_HALF * (r_mount @ np.array([nx, -ny, -nz])))
      at = self._to_map(centre[None, :])[0]
      out.append(at)
      self.detections.append({**{k: plain(v) for k, v in d.items()},
                              "time": float(mis.data.time), "pose": list(mis.pose),
                              "attitude": list(attitude(mis)), "knob": [float(v) for v in at]})
    return out

  def cloud(self) -> np.ndarray:
    """One depth frame at the part's resolution, in the map."""
    return self._to_map(self.depth.frame(self.mis.data).points)

  @staticmethod
  def keep(cloud: np.ndarray, knob) -> np.ndarray:
    """A cloud's points the record keeps: within `KEEP_M` of the knob
    across the floor, off it and under `KEEP_UNDER_M`."""
    near = np.hypot(cloud[:, 0] - knob[0], cloud[:, 1] - knob[1]) < KEEP_M
    return cloud[near & (cloud[:, 2] > box.FLOOR_M) & (cloud[:, 2] < KEEP_UNDER_M)]

  def _hold(self, seconds: float) -> Routine:
    yield from self.mis._drive_routine(seconds, 0.0, 0.0)

  def measure_routine(self, keep: bool = False) -> Routine:
    """Standing still: `LOOKS` decodes and `FRAMES` depth frames, `APART_S`
    apart -- a `Guess` off the knob (the mean of every face read) and the
    frames, or why there is none -- and the frames' points. `keep` puts the
    frames and the decodes in the record."""
    if keep:
      self.detections, self.clouds = [], []
    knobs, clouds = [], []
    for k in range(max(LOOKS, FRAMES)):
      yield from self._hold(APART_S)
      if k < LOOKS:
        knobs += self.look()
      if k < FRAMES:
        clouds.append(self.cloud())
    cloud = np.concatenate(clouds)
    if not knobs:
      return "the knob's tag not seen", cloud
    knob = np.mean(knobs, axis=0)
    if keep:
      self.clouds = [self.keep(c, knob) for c in clouds]
    x, y, _ = self.mis.pose
    return guess(cloud, knob, math.atan2(knob[1] - y, knob[0] - x)), cloud

  def refind_routine(self, g: Guess, standing: np.ndarray, att: dict) -> Routine:
    """Lying, the box re-found by its front face (the module docstring's
    `lie`): `g` moved by what the walk and the lie-down did to the belief,
    and the record's standing clouds and decodes with it -- or `g` as it
    was, said in `att`, where either look missed the face."""
    before = front_face(standing, g)
    lying = []
    for _ in range(LYING_FRAMES):
      yield from self._hold(APART_S)
      lying.append(self.cloud())
    after = front_face(np.concatenate(lying), g)
    if before is None or after is None:
      att["refound"] = "the box's front face not seen " + ("standing" if before is None else "lying")
      self.clouds += [self.keep(c, g.knob) for c in lying]
      return g
    move = Move(g, before, after)
    att["refound"] = move.as_dict()
    self.clouds = [move.point(c) for c in self.clouds] + [self.keep(c, g.knob) for c in lying]
    for d in self.detections:
      d["knob"] = [float(v) for v in move.point(d["knob"])]
      d["pose"] = move.pose(d["pose"])
    return move.guess()

  # ---- the arm --------------------------------------------------------------------

  def _hand(self) -> tc.ClawHand:
    return tc.ClawHand(self.mis)

  def follow_routine(self, hand, path, s0: float, s1: float, rate: float,
                     level) -> Routine:
    """The jaws along `path(s)` (heading frame) from `s0` to `s1` at `rate`:
    the fork aimed every physics step and the slide sent at its own speed.
    False once a point is out of reach or past the slide's stroke, or the
    body fell."""
    mis = self.mis
    n = max(1, int(abs(s1 - s0) / rate / float(mis.model.opt.timestep)))
    for k in range(1, n + 1):
      vx, vz, slide = hand.vertex_for(*path(s0 + (s1 - s0) * k / n), level)
      g = mis.arm.goal
      q = am.solve_vertex(mis.arm_spec, vx, vz, near=(float(g[0]), float(g[1] - g[0])))
      if q is None or abs(slide) > rk.CLAW_TRAVEL or hand.fell():
        return False
      mis.arm.aim(*q)
      yield from servo.ramp_routine(mis, hand.slide, slide, tc.SLIDE_SPEED)
    return True

  def reaches(self, hand, path, level) -> str | None:
    """Why `path` (heading frame, over the lid's sweep) is out of the
    arm's reach or the slide's stroke, or None."""
    for s in np.linspace(0.0, TOP, PATH_CHECKS):
      vx, vz, slide = hand.vertex_for(*path(float(s)), level)
      if am.solve_vertex(self.mis.arm_spec, vx, vz) is None:
        return f"the lid at {math.degrees(s):.0f} deg is out of the arm's reach"
      if abs(slide) > rk.CLAW_TRAVEL:
        return f"the lid at {math.degrees(s):.0f} deg is past the slide's stroke"
    return None

  def _restore(self, gains) -> None:
    """The arm's working `gains` back on any way out, held where it IS: put
    back mid-sweep with the probe's target ahead of it, a stiff arm's step
    to it is the kick `RELEASE_S` lets go to avoid."""
    arm = self.mis.arm
    if (arm.kp, arm.kd) != tuple(gains):
      shoulder, fore = arm.q()
      arm.hold_at(shoulder, fore - shoulder)
      arm.kp, arm.kd = gains

  def _leave_routine(self, hand, gains) -> Routine:
    """Off the knob: the jaws open and the arm let settle at the probe's
    gain (`RELEASE_S`), then its working `gains`; the fork up and back, the
    slide to its middle, the carry pose, onto its feet -- after a fall, the
    gains alone."""
    mis = self.mis
    if hand.fell():
      mis.arm.kp, mis.arm.kd = gains
      return
    yield from hand.jaws_routine(closed=False)
    yield from self._hold(RELEASE_S)
    mis.arm.kp, mis.arm.kd = gains
    x, z = mis._vertex_goal()
    yield from hand.move_routine(x - 0.03, z + 0.05, speed=0.05)
    yield from hand.centre_routine()
    yield from hand.move_routine(*am.CARRY, speed=0.15)
    if hand.fell():
      return
    mis.arm.aim(*am.CARRY_Q)
    yield from self._hold(0.3)
    yield from mis.stand_routine()

  # ---- calibrating -----------------------------------------------------------------

  def bare(self, cloud: np.ndarray, frame: Frame, x0: float, x1: float) -> bool:
    """No point of `cloud` (map) standing on the floor between `x0` and
    `x1` ahead in `frame`, within `FREE_HALF_M` of its line: what was SEEN
    (`FREE_HALF_M`'s note says what that leaves out)."""
    h = frame.of(cloud)
    return not ((h[:, 0] > x0) & (h[:, 0] < x1) & (np.abs(h[:, 1]) < FREE_HALF_M)
                & (cloud[:, 2] > FREE_OVER_M)).any()

  def calibrate_routine(self, g: Guess, cloud: np.ndarray, log: dict) -> Routine:
    """The empty sweep (the module docstring's `calibrate`): the
    calibration's record and the arm's friction, or None with why in
    `log`. The arc is the guess's, square ahead with the knob's place
    `KNOB_AHEAD_M` ahead of the lying body -- the probe's arc as the
    walk-in leaves it -- and the floor under it must be bare."""
    mis = self.mis
    c, s = math.cos(g.yaw), math.sin(g.yaw)
    r = np.subtract(g.pin, g.hinge)
    pin_box = np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]])
    reach = [KNOB_AHEAD_M + float((rot_about((0.0, 1.0, 0.0), a) @ pin_box - pin_box)[0])
             for a in np.linspace(0.0, TOP, PATH_CHECKS)]
    # standing, the lying body's footprint is `LIE_SHIFT_M` nearer
    if not self.bare(cloud, self.frame_now(), min(reach) - dk.LIE_SHIFT_M - FREE_PAD_M,
                     max(reach) - dk.LIE_SHIFT_M + FREE_PAD_M):
      log["why"] = "the floor under the arc is not bare"
      return None
    yield from mis.rest_routine()
    f = self.frame_now()
    path = pin_path((KNOB_AHEAD_M, 0.0, g.knob[2] - f.floor), pin_box, (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0))
    hand = self._hand()
    level = mis.odo.att.level()
    why = self.reaches(hand, path, level)
    if why is not None:
      log["why"] = why
      return None
    gains = (mis.arm.kp, mis.arm.kd)
    mis.arm.kp, mis.arm.kd = KP, KD
    rec = Recorder(mis)
    try:
      yield from hand.jaws_routine(closed=True)
      ok = yield from hand.to_routine(*path(0.0), level)
      yield from self._hold(SWING_SETTLE_S)
      rec.begin()
      ok = ok and (yield from self.follow_routine(hand, path, 0.0, TOP, RATES[0], level))
      yield from self._hold(TURN_S)
      ok = ok and (yield from self.follow_routine(hand, path, TOP, BOTTOM, RATES[0], level))
      calibration = rec.record()
      # before the leave, which stands it up: `fell` reads any new posture
      fell = hand.fell()
      yield from self._leave_routine(hand, gains)
    finally:
      rec.stop()
      self._restore(gains)
    if not ok:
      log["why"] = "fell" if fell else "the arc left the arm's reach"
      return None
    friction = tuple(float(v) for v in am.arm_friction(*rec.beyond()))
    log["calibrated"] = [round(v, 4) for v in friction]
    return calibration, friction

  # ---- finding and walking in --------------------------------------------------

  def find_routine(self, log: dict) -> Routine:
    """The knob's tag from where it stands, else turned each way: a `Guess`
    (or why there is none), and the cloud it was read off."""
    mis = self.mis
    for turn in (0.0, *SWEEP_DEG):
      if turn:
        yield from mis._turn_by_routine(turn)
      yield from self._hold(STOPPED_S)
      if self.look():
        break
    else:
      return "the knob's tag not found", np.zeros((0, 3))
    log["foundTurnDeg"] = turn
    g, cloud = yield from self.measure_routine()
    log["found"] = g.as_dict() if isinstance(g, Guess) else g
    return g, cloud

  def walk_in_routine(self, g: Guess, until: float, log: dict) -> Routine:
    """To the measuring stance off the knob on the box's axis (the module
    docstring's `walk in`): a look point first, then steered by the knob's
    tag. "stopped" there, or "budget"."""
    mis = self.mis
    u = np.array([math.cos(g.yaw), math.sin(g.yaw)])
    lx, ly = np.array(g.knob[:2]) - LOOK_M * u
    px, py, _ = mis.pose
    if math.hypot(lx - px, ly - py) > AT_LOOK_M:
      yield from mis.drive_to_routine(lx, ly, timeout=max(1.0, until - float(mis.data.time)))
      log["drive"] = (mis.last_drive or {}).get("why")
    yield from mis.face_routine(g.yaw)
    return (yield from self._steer_routine(g.knob, g.yaw, MEASURE_AT_M, looks=True))

  def _steer_routine(self, knob, yaw: float, off: float, looks: bool) -> Routine:
    """Steered (`legs.rack.walk_in_twist`) to `off` short of the knob along
    the axis `yaw`, facing along it; with `looks`, the knob re-read off its
    tag every `LOOK_EVERY_S`, else where it was measured. "stopped", or
    "budget"."""
    mis = self.mis
    u = np.array([math.cos(yaw), math.sin(yaw)])
    knob = np.array(knob[:2], dtype=float)
    last = -math.inf
    t0 = float(mis.data.time)
    while mis.data.time - t0 < WALK_IN_S:
      if looks and mis.data.time - last >= LOOK_EVERY_S:
        last = float(mis.data.time)
        seen = self.look()
        if seen:
          knob = np.mean(seen, axis=0)[:2]
      stance = (*(knob - off * u), yaw)
      tw = rk.walk_in_twist(*dk.relative(mis.pose, stance))
      if (tw.vx, tw.vy, tw.yaw_rate) == (0.0, 0.0, 0.0):
        yield from self._hold(STOPPED_S)
        return "stopped"
      yield from mis._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
    return "budget"

  # ---- the whole of it --------------------------------------------------------------

  def routine(self, patience: float = PATIENCE_S) -> Routine:
    """The probe (the module docstring): a `Probed`."""
    mis = self.mis
    until = float(mis.data.time) + patience
    out = Probed(log={"tries": []})
    if mis.carrying != "module_claw":
      out.why = "no claw on the fork"
      return out
    was, mis.working = mis.working, True
    try:
      g, cloud = yield from self.find_routine(out.log)
      if not isinstance(g, Guess):
        out.why = g
        return out
      got = yield from self.calibrate_routine(g, cloud, out.log)
      if got is None:
        out.why = "not calibrated: " + out.log.get("why", "")
        return out
      out.calibration, out.friction = got
      for _ in range(TRIES):
        att: dict = {}
        out.log["tries"].append(att)
        if float(mis.data.time) >= until:
          out.why = "out of time"
          return out
        yield from mis.stand_routine()
        att["walkIn"] = yield from self.walk_in_routine(g, until, att)
        yield from self._hold(STILL_S)
        g2, standing = yield from self.measure_routine(keep=True)
        if not isinstance(g2, Guess):
          att["why"] = g2
          yield from self._back_out_routine()
          continue
        g = g2
        att["guess"] = g.as_dict()
        att["stand"] = yield from self._steer_routine(g.knob, g.yaw, STAND_AT_M, looks=False)
        yield from mis.rest_routine()
        g = yield from self.refind_routine(g, standing, att)
        frame = self.frame_now()
        knob = frame.of(g.knob)
        att["knob"] = [round(float(v), 4) for v in knob]
        hand = self._hand()
        level = mis.odo.att.level()
        take = self.take_point(g, frame)
        att["take"] = [round(float(v), 4) for v in take]
        path = pin_path(take, frame.of(g.pin), frame.of(g.hinge), frame.turn(g.axis))
        why = (None if REACH_X[0] <= knob[0] <= REACH_X[1] and abs(knob[1]) <= REACH_Y
               else "the knob out of reach") or self.reaches(hand, path, level)
        if why is not None:
          att["why"] = why
          yield from self._back_out_routine()
          continue
        out.guess, out.frame = g, frame
        yield from self._probe_routine(hand, path, level, out)
        return out
      out.why = "never lined up"
      return out
    finally:
      mis.working = was

  @staticmethod
  def take_point(g: Guess, frame: Frame) -> np.ndarray:
    """Where the jaws take the knob, heading frame: its middle, held back
    along the box's axis to `TAKE_CLEAR_M` in front of the pin -- the
    bracket's tip -- where it hangs nearer it. Held, the knob moves with the
    jaws, so the path begins here."""
    into = frame.turn((math.cos(g.yaw), math.sin(g.yaw), 0.0))
    knob = frame.of(g.knob)
    ahead = float((frame.of(g.pin) - knob) @ into)
    return knob - max(0.0, TAKE_CLEAR_M - ahead) * into

  def _back_out_routine(self) -> Routine:
    mis = self.mis
    yield from mis.stand_routine()
    yield from mis._back_out_by_routine(BACK_OUT_M, BACK_OUT_S, rk.APPROACH_V,
                                        dk.BACK_OUT_SETTLE_S)

  def _probe_routine(self, hand, path, level, out: Probed) -> Routine:
    """Take hold, sweep and let go, recording (the module docstring)."""
    mis = self.mis
    rec = Recorder(mis)
    gains = (mis.arm.kp, mis.arm.kd)
    mis.arm.kp, mis.arm.kd = KP, KD
    try:
      rec.begin()
      out.t0 = rec.t0
      rec.mark("take")
      yield from hand.jaws_routine(closed=False)
      ok = yield from hand.to_routine(*(path(0.0) + [0.0, 0.0, OVER_M]), level)
      yield from self._hold(SWING_SETTLE_S)
      ok = ok and (yield from hand.to_routine(*path(0.0), level, tc.WORK_V / 2))
      yield from hand.jaws_routine(closed=True, settle=tc.GRIP_SETTLE_S)
      if not ok:
        out.why = "fell" if hand.fell() else "the knob out of reach"
      s = 0.0
      for i, rate in enumerate(RATES):
        if not ok:
          break
        rec.mark(f"up{i}")
        ok = yield from self.follow_routine(hand, path, s, TOP, rate, level)
        yield from self._hold(TURN_S)
        rec.mark(f"down{i}")
        ok = ok and (yield from self.follow_routine(hand, path, TOP, BOTTOM, rate, level))
        yield from self._hold(TURN_S)
        s = BOTTOM
      out.record = rec.record(depth=self.clouds, detections=self.detections)
      out.phases = rec.phases()
      out.log["swept"] = ok
      if ok:
        down = self.held_down(rec, out.friction)
        out.log["heldDownN"] = round(down, 3)
        out.ok = down >= HELD_DOWN_N
        out.why = "probed" if out.ok else "nothing held"
      else:
        out.why = out.why or ("fell" if hand.fell() else "the path left the arm's reach")
      yield from self._leave_routine(hand, gains)
    finally:
      rec.stop()
      self._restore(gains)

  def held_down(self, rec: Recorder, friction) -> float:
    """The force at the tool down (N), its mean over the sweeps up past
    `HELD_AFTER_S` where both joints turn: the probe's own word that it
    held the lid."""
    f = rec.force(friction)
    _, rates = rec.beyond()
    turning = np.all(np.abs(rates) > am.FRICTION_BAND, axis=1)
    use = np.zeros(len(f), dtype=bool)
    skip = int(round(HELD_AFTER_S / float(self.mis.model.opt.timestep)))
    for name, (a, b) in rec.phases().items():
      if name.startswith("up"):
        use[a + skip:b] = True
    use &= turning
    return float(-f[use, 1].mean()) if use.any() else 0.0
