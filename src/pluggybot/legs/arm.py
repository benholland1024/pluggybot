"""The quadruped's arm (issue #378): two pitch joints on the torso, and the
fork at its tip that carries a tool.

The arm moves in the robot's forward-and-up plane: a SHOULDER on the torso's
top front and an ELBOW, both motors stacked at the shoulder -- the elbow is
driven through a rod parallel to the upper arm, so the forearm's absolute
angle is its motor's, as a palletising robot's is -- and a passive second
parallelogram keeps the END PLATE at the torso's own angle wherever the arm
is (`level="parallelogram"`, the choice; SimNotes, "The quadruped's arm",
has the table that chose it over a wrist motor and over pitching the body).

In the joint space MuJoCo integrates, the three are joints about y
(`arm_shoulder`, `arm_elbow`, `arm_wrist`, positive raises), and the
linkages are tendons over them: `arm_fore` = shoulder + elbow is what the
elbow's motor drives (the forearm's absolute angle), and `arm_level` =
shoulder + elbow + wrist is held at zero by an equality (the plate's
absolute angle). By virtual work that is the ideal linkage exactly; the
rods' own mass is lumped into the links.

The fork at the tip is the rover's coupling re-sized for a body that cannot
creep (#378, "the coupling"): a tool still hangs by its split peg, but the
fork's V-notches sit further out along a longer peg and its end-stops are
ramps that centre the peg along its axis as the fork lifts, so the lateral
capture is what the legs can deliver rather than the rover's 4 mm.

Frames: the TORSO's (x forward, z up, origin at its centre) for the mount;
the arm PLANE's (x, z at y = `ArmSpec.y`) for the kinematics; the PLATE's
(the end plate, level with the torso under the parallelogram) for the fork.
"""

from dataclasses import dataclass, field, replace
import math

import numpy as np

from pluggybot.legs.actuator import GIM8108_8, Motor
from pluggybot.rack.coupling import PEG_R

#: A link's tube and the rods beside it: radius of the capsule the sim
#: collides and draws, m (a 20/18 mm carbon tube with its rod ends).
TUBE_R = 0.011


@dataclass(frozen=True)
class ForkSpec:
  """The coupling's arm half, in the PLATE frame. Lengths in metres."""

  #: The V-notches' vertex (where a seated peg's axis sits, less
  #: `seat_rise`) ahead of and below the wrist axis.
  vertex_x: float = 0.085
  vertex_z: float = -0.030
  #: The two V-notches, either side of the arm's plane, under the peg's two
  #: conductors (the poles).
  fork_y: float = 0.085
  #: The V's flanks off the plate's plane, deg. ⚠ STEEPER THAN THE ROVER'S
  #: 45: the plate pitches with the torso, up to `STAIR_PITCH_DEG` down a
  #: flight, and that lays one flank down -- at 45, to 12 deg off level,
  #: under the peg's friction angle (21.8 deg), so a jolt that hops the peg
  #: leaves it on that flank or carries it over: a tool leaning 60 mm ahead
  #: was lost on 13 of 17 descents; at 60, on none of 18
  #: (`--retention --stairs [--first]`). The faces stay plain metal, the
  #: peg's 0.4: faced at 0.2 they let a jolt slide the peg along its axis
  #: and up an end-ramp, and the tool was lost.
  flank_deg: float = 60.0
  #: A V plate: half-length along its flank, half-width across, half-thick.
  #: ⚠ The flank's run across the plate is the V's mouth, its capture along
  #: the bay: 31 mm long keeps the rover's 15.6 at 60 deg. At the rover's
  #: 22 mm the mouth was 11, and the capture's corners went chaotic.
  v_half_len: float = 0.0156
  v_half_w: float = 0.006
  v_thick: float = 0.003
  #: The end-stops: a ramp outboard of each peg end, from `stop_y` (1.5 mm
  #: past a centred peg's end: a tool rides the fork within that) rising
  #: `ramp_h` over `ramp_w` (53 deg): a peg end the fork lifts under slides
  #: down it and inward, if the ramp's push beats the far V's grip, which
  #: the steeper V raised (`test_arm.py` has the arithmetic); the top stays
  #: 20 mm over the vertex, under a peg the fork passes beneath. The rover's
  #: stops, 4 mm past a peg with no ramps, took nothing 10 mm off
  #: (`--capture --rover`).
  stop_y: float = 0.1115
  ramp_w: float = 0.015
  ramp_h: float = 0.020
  #: The lean-pad, behind a seated module's back plate, `pad_drop` under
  #: the V's vertex: low enough that on the approach, FORK_DROP under a
  #: hanging tool's peg, it passes UNDER the tool's plate (52 mm under its
  #: peg) -- level with the vertex it rammed the hanging tool and flipped it
  #: onto the fork. ⚠ It stands 3 mm BEHIND the plate's back face
  #: (`pad_intrude` -3 mm): at the rover's +2.7 mm it met the plate's bottom
  #: edge while the peg was still on the rack's trays and levered the tool
  #: 15 deg up the V's flank, and at 0 an aim 5 mm deep did the same
  #: (`--capture`); 3 mm behind, the tool seats plumb. It is there for a
  #: tool that presses (the pen): the reaction swings it 4 deg onto the pad.
  pad_drop: float = 0.040
  pad_intrude: float = -0.003

  def seat_rise(self, peg_r: float = PEG_R) -> float:
    """A seated peg's axis over the V's vertex."""
    return peg_r / math.cos(math.radians(self.flank_deg))

  def flank_top(self) -> float:
    """The flanks' top edges over the V's vertex."""
    return 2 * self.v_half_len * math.sin(math.radians(self.flank_deg))


#: The plate's worst pitch off level coming down the house's flight (ten
#: 0.18 m risers), deg, measured (`--retention --stairs`): the parallelogram
#: holds the plate at the TORSO's angle, and a descent pitches the torso.
STAIR_PITCH_DEG = 33.0


@dataclass(frozen=True)
class ArmSpec:
  """A candidate arm. Lengths in metres, masses in kg, angles in radians."""

  name: str = "arm"
  #: "parallelogram" (two motors, the plate kept level by a passive linkage),
  #: "wrist" (a third motor pitches the plate) or "body" (the plate is fixed
  #: to the forearm: its angle is the joints' sum plus the body's pitch).
  level: str = "parallelogram"
  #: The shoulder axis in the torso frame (x, z) and the arm plane's y.
  shoulder_x: float = 0.15
  shoulder_z: float = 0.10
  y: float = 0.0
  #: Shoulder axis -> elbow axis, elbow axis -> wrist axis.
  upper: float = 0.25
  fore: float = 0.35
  #: Both arm motors, stacked at the shoulder (and the wrist's, for "wrist").
  motor: Motor = GIM8108_8
  wrist_motor: Motor | None = None
  #: Structure, ESTIMATED (Parts.md, "The arm"): each link's tube, end
  #: fittings and bearings; the rods of the two parallelograms; the end
  #: plate with the fork and the lean-pad.
  upper_mass: float = 0.12
  fore_mass: float = 0.12
  rods_mass: float = 0.06
  plate_mass: float = 0.08
  fork: ForkSpec = field(default_factory=ForkSpec)
  #: Joint ranges: the shoulder's elevation, the elbow's relative angle.
  shoulder_range: tuple[float, float] = (-1.2, 3.4)
  elbow_range: tuple[float, float] = (-math.pi, 0.4)
  #: Folded on the torso's top: the upper arm straight back, the forearm
  #: 10 deg up from it -- the fork then sits above the nose camera's view
  #: and under the LIDAR's scan plane, and the stowed arm hides nothing
  #: from either (the spike's `--sensors`: flat, the fork hid 13 % of the
  #: nose camera's image; at 15 deg the forearm blinds 9 LIDAR rays).
  stow: tuple[float, float] = (math.pi, -math.pi + 0.175)

  def with_(self, **kw) -> "ArmSpec":
    return replace(self, **kw)

  @property
  def reach(self) -> float:
    return self.upper + self.fore

  @property
  def mass(self) -> float:
    """The arm on the torso, without a tool: its motors included."""
    m = 2 * self.motor.mass + self.upper_mass + self.fore_mass + self.rods_mass \
        + self.plate_mass
    if self.level == "wrist":
      m += (self.wrist_motor or self.motor).mass
    return m


#: Where the fork carries a tool while the robot walks: its V vertex in the
#: torso frame, over the nose and high, so a pendant as long as the claw's
#: (0.2 m) hangs clear of the LIDAR's scan plane and the nose camera's view,
#: and only the upper arm's tube crosses the plane (5 of 360 rays; the
#: spike's `--sensors`).
CARRY = (0.40, 0.40)
#: The fall reflex: past this tilt of the torso's up axis (60 deg, the
#: body's FALLEN) the arm folds to its stow. It was what let #377's get-up
#: roll the robot -- held out, the arm props it on its side, and 3 of 6
#: pushes stayed down; #389's get-up rolls it either way (SimNotes, "A
#: gentler get-up").
FOLD_ON_FALL_COS = math.cos(math.radians(60.0))

#: THE TOOL ENVELOPE a tool on this coupling must fit (the workshop's
#: validator's, #375 step 4; `rack.coupling`'s are the rover's). Measured
#: (the spike's `--envelope`: carried through a 1.0 m/s trot and a stop;
#: `--retention --stairs`: the house's flight, both ways, at 0, 30 and
#: 60 mm ahead):
#:   mass  a tool of 0.60 kg stayed seated with its CoM up to 60 mm off its
#:         peg; the arm holds 0.61 kg (the claw and a 0.4 kg cube) straight
#:         out at 39 % of its motors' continuous rating. The ceiling keeps a
#:         payload's room under both.
#:   ahead a tool's CoM on or AHEAD of its peg (away from the robot) leans
#:         it onto the lean-pad, 6 deg; a mass behind it, below the peg,
#:         meets the pad's post. 60 mm is the stairs' tested range: at 90
#:         the flat and 27 descents held too, where the rover's 45 deg V's
#:         had flopped a 0.40 kg tool to 74 deg on the flat.
#:   drop  under the peg at the carry pose (`CARRY`) a pendant this long
#:         keeps 2 cm over the LIDAR's scan plane (`test_arm.py` reads it).
#:   power the peg is still the connector (`rack.coupling.PEG_POWER_W`), its
#:         preload the tool's weight; flown, it opened for at most 96 ms (down
#:         a flight of stairs, 60 mm ahead) against the 200 ms holding
#:         capacitor.
TOOL_MAX_KG = 0.40
TOOL_MAX_AHEAD_M = 0.060
TOOL_MAX_MOMENT_NM = 0.35
TOOL_MAX_DROP_M = 0.20

#: The coupling's verbs (the rover's, `rack.coupling`): the fork's V vertex
#: runs FORK_DROP under a hanging peg on the way in (its flanks' tops pass
#: 3.4 mm under the peg), the lift takes it LIFT above that, and it backs
#: out BACK_OUT; a return is the reverse. LIFT carries the peg's underside
#: 14 mm over the trays' V corners: the rover's 36 mm (with the 45 deg V)
#: cleared them by 4, and a return aimed 10 mm low knocked the tool off the
#: tray.
FORK_DROP = 0.0335
LIFT = 0.056
STANDOFF = 0.08
BACK_OUT = 0.10
#: The fork's speed along its straight lines, m/s.
FORK_V = 0.10


# ---- kinematics (the arm plane: x forward, z up, from the torso's centre) ----

def elbow_xz(spec: ArmSpec, qs: float) -> tuple[float, float]:
  return (spec.shoulder_x + spec.upper * math.cos(qs),
          spec.shoulder_z + spec.upper * math.sin(qs))


def wrist_xz(spec: ArmSpec, qs: float, qe: float) -> tuple[float, float]:
  ex, ez = elbow_xz(spec, qs)
  return ex + spec.fore * math.cos(qs + qe), ez + spec.fore * math.sin(qs + qe)


def plate_angle(spec: ArmSpec, qs: float, qe: float, qw: float = 0.0) -> float:
  """The end plate's pitch in the torso frame (+ raises its front)."""
  if spec.level == "parallelogram":
    return 0.0
  if spec.level == "wrist":
    return qs + qe + qw
  return qs + qe


def seat_xz(spec: ArmSpec, qs: float, qe: float, qw: float = 0.0,
            peg_r: float = PEG_R) -> tuple[float, float]:
  """Where a seated peg's axis sits, in the torso frame's x-z."""
  wx, wz = wrist_xz(spec, qs, qe)
  a = plate_angle(spec, qs, qe, qw)
  fx, fz = spec.fork.vertex_x, spec.fork.vertex_z + spec.fork.seat_rise(peg_r)
  return (wx + fx * math.cos(a) - fz * math.sin(a),
          wz + fx * math.sin(a) + fz * math.cos(a))


def solve(spec: ArmSpec, x: float, z: float, elbow: str = "up",
          near: tuple[float, float] | None = None) -> tuple[float, float] | None:
  """(shoulder, elbow) putting the WRIST axis at (x, z) in the torso frame,
  or None out of reach or out of the joints' ranges. `elbow` "up" keeps the
  elbow above the line from the shoulder to the wrist. The shoulder's angle
  is NOT wrapped: it is the turn inside its range (the stow is at pi, and a
  target close over the shoulder wants more -- wrapped to -pi, the joint
  limit swung the arm the other way). With `near`, the branch nearest it."""
  if near is not None:
    got = [q for e in ("up", "down")
           if (q := solve(spec, x, z, e)) is not None]
    if not got:
      return None
    return min(got, key=lambda q: abs(q[0] - near[0]) + abs(q[1] - near[1]))
  dx, dz = x - spec.shoulder_x, z - spec.shoulder_z
  a, b = spec.upper, spec.fore
  c = (dx * dx + dz * dz - a * a - b * b) / (2 * a * b)
  if abs(c) > 1.0:
    return None
  qe = math.acos(c) * (-1.0 if elbow == "up" else 1.0)
  qs = math.atan2(dz, dx) - math.atan2(b * math.sin(qe), a + b * math.cos(qe))
  lo, hi = spec.shoulder_range
  qs = lo + (qs - lo) % (2 * math.pi)
  q = (qs, qe)
  return q if within(spec, q) else None


def solve_seat(spec: ArmSpec, x: float, z: float, elbow: str = "up",
               peg_r: float = PEG_R, near=None) -> tuple[float, float] | None:
  """(shoulder, elbow) putting a seated peg's axis at (x, z), the plate
  level (the parallelogram's case: its angle does not depend on the joints)."""
  fx, fz = spec.fork.vertex_x, spec.fork.vertex_z + spec.fork.seat_rise(peg_r)
  return solve(spec, x - fx, z - fz, elbow, near)


def solve_vertex(spec: ArmSpec, x: float, z: float, near=None,
                 elbow: str = "up") -> tuple[float, float] | None:
  """(shoulder, elbow) putting the fork's V vertex at (x, z), plate level."""
  return solve(spec, x - spec.fork.vertex_x, z - spec.fork.vertex_z, elbow, near)


def within(spec: ArmSpec, q: tuple[float, float]) -> bool:
  qs, qe = q
  return (spec.shoulder_range[0] <= qs <= spec.shoulder_range[1]
          and spec.elbow_range[0] <= qe <= spec.elbow_range[1])


#: ...and the carry pose as the chosen arm's joint angles (shoulder, elbow):
#: where a walk puts an arm that carries a tool (`procedure.steps.travel_pose`)
#: and a rest leaves it.
CARRY_Q = solve_vertex(ArmSpec(), *CARRY, near=ArmSpec().stow)


# ---- the static load -------------------------------------------------------------

G = 9.81


def gravity_torques(spec: ArmSpec, qs: float, qe: float,
                    tool_kg: float = 0.0, tool_x: float = 0.0,
                    tool_z: float = -0.05) -> tuple[float, ...]:
  """What the motors hold against gravity, N*m: (shoulder, elbow), and the
  wrist's third for `level="wrist"`. The tool's CoM is (`tool_x`,
  `tool_z`) from the seated peg; the plate's mass sits at the wrist.

  The elbow's motor drives the forearm's ABSOLUTE angle, so it holds the
  forearm and the WEIGHT at the wrist about the elbow; the shoulder's
  holds the upper arm and the weight at the elbow. What the tip's weight
  does ahead of the wrist -- its moment about the wrist -- goes to the
  torso through the level parallelogram and no motor holds it (by
  virtual work: the equality's force acts on all three joints alike). A
  wrist motor holds it instead; with the plate fixed to the forearm
  (`"body"`), the elbow's does."""
  ex, _ = elbow_xz(spec, qs)
  wx, _ = wrist_xz(spec, qs, qe)
  px, _ = seat_xz(spec, qs, qe)
  sx = spec.shoulder_x
  fore_kg = spec.fore_mass + spec.rods_mass / 2
  upper_kg = spec.upper_mass + spec.rods_mass / 2
  tip = [(spec.plate_mass, wx), (tool_kg, px + tool_x)]
  if spec.level == "wrist":
    tip.append(((spec.wrist_motor or spec.motor).mass, wx))
  weight = sum(m for m, _ in tip)
  ahead = sum(m * G * (x - wx) for m, x in tip)
  tau_e = fore_kg * G * (wx - ex) / 2 + weight * G * (wx - ex)
  tau_s = upper_kg * G * (ex - sx) / 2 + (fore_kg + weight) * G * (ex - sx)
  if spec.level == "wrist":
    return tau_s, tau_e, ahead
  if spec.level == "body":
    return tau_s, tau_e + ahead
  return tau_s, tau_e


# ---- the MJCF --------------------------------------------------------------------

def _f(v: float) -> str:
  return f"{v:.6g}"


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


def _v_notch(prefix: str, x: float, y: float, z: float, f: ForkSpec) -> str:
  """An upward-open V (the peg's axis along y) of two plates `flank_deg` off
  level whose upper faces are the flanks, vertex at (x, y, z)
  (`rack.coupling`'s, at 45)."""
  a = math.radians(f.flank_deg)
  ox, oz = f.v_half_len * math.cos(a), f.v_half_len * math.sin(a)
  out = []
  for s, lbl in ((-1, "a"), (1, "b")):
    out.append(
      f'<geom name="{prefix}{lbl}" class="fork" type="box" '
      f'size="{_v(f.v_half_len, f.v_half_w, f.v_thick)}" '
      f'pos="{_v(x + s * ox, y, z + oz - f.v_thick / math.cos(a))}" '
      f'quat="{_quat_y(-s * a)}"/>')
  return "\n          ".join(out)


def _quat_y(angle: float) -> str:
  """A rotation about y as a quaternion: unit-free (either compiler angle)."""
  return _v(math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def fork_xml(spec: ArmSpec, prefix: str = "") -> str:
  """The fork's geoms in the plate body's frame."""
  f = spec.fork
  vx, vz = f.vertex_x, f.vertex_z
  g = []
  # A tool FACES the robot (its +x toward the arm), so its left conductor
  # rides the arm's right V: each V is named for the tool's pole it takes.
  for s, lbl in ((-1, "l"), (1, "r")):
    y = s * f.fork_y
    g.append(_v_notch(f"{prefix}arm_v{lbl}_", vx, y, vz, f))
    # The prong: from the plate forward to under the V.
    g.append(f'<geom name="{prefix}arm_prong_{lbl}" class="fork" type="box" '
             f'size="{_v((vx + 0.015) / 2, 0.005, 0.004)}" '
             f'pos="{_v((vx - 0.015) / 2, y, vz - 0.012)}"/>')
    # The end-stop: a ramp outboard of the peg's end, rising away from the
    # arm's plane, its low edge `stop_y` out at the V's height. A peg end
    # it meets while the fork lifts slides down it toward the middle.
    ang = math.atan2(f.ramp_h, f.ramp_w)
    half_t = 0.0015
    g.append(f'<geom name="{prefix}arm_ramp_{lbl}" class="fork" type="box" '
             f'friction="{_f(RAMP_MU)} 0.005 0.0001" priority="2" '
             f'size="{_v(0.006, math.hypot(f.ramp_w, f.ramp_h) / 2, half_t)}" '
             f'pos="{_v(vx, s * (f.stop_y + f.ramp_w / 2), vz + f.ramp_h / 2 - half_t / math.cos(ang))}" '
             f'quat="{_quat_x(s * ang)}"/>')
  # The bridge across the prongs' roots, and the lean-pad on its post.
  g.append(f'<geom name="{prefix}arm_bridge" class="fork" type="box" '
           f'size="{_v(0.006, f.stop_y, 0.005)}" pos="{_v(-0.009, 0, vz - 0.012)}"/>')
  # The pad is a round bar along y: rising into place, it meets the tool
  # plate's bottom edge and deflects it smoothly forward (a square pad's
  # corner caught that edge and levered the tool 16 deg up the V's flank).
  g.append(f'<geom name="{prefix}arm_pad" class="fork" type="capsule" '
           f'size="{_v(PAD_R, 0.024)}" pos="{_v(_pad_x(spec), 0, vz - f.pad_drop)}" '
           f'quat="{_quat_x(math.pi / 2)}"/>')
  # ...on a post from the bridge, under the tool's plate.
  g.append(f'<geom name="{prefix}arm_pad_post" class="fork" type="box" '
           f'size="{_v((_pad_x(spec) + 0.009) / 2, 0.006, 0.003)}" '
           f'pos="{_v((_pad_x(spec) - 0.009) / 2, 0, vz - f.pad_drop - 0.008)}"/>')
  g.append(f'<site name="{prefix}arm_seat" pos="{_v(vx, 0, vz)}" size="0.003"/>')
  return "\n          ".join(g)


def _quat_x(angle: float) -> str:
  return _v(math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


#: A module's back face is this far behind its peg's axis (`rack.coupling`'s
#: `TOOL_HALF_X`): what the lean-pad meets. A seated module faces the robot
#: with its +x, so in the plate frame its back face is BEHIND the peg.
MODULE_HALF_X = 0.010


#: The lean-pad bar's radius, m.
PAD_R = 0.005
#: The end-ramps' face: acetal or PTFE-faced (0.15; priority 2 over the
#: peg's 0.4). Sliding a peg in along its axis is the ramp's push against
#: the far V's grip: with 45 deg V's and ramps at the peg's 0.4 that was
#: 0.30 of the tool's weight against 0.28, and a pick from 15 mm off at
#: 4 deg left the peg's end on the ramp, its pole open. The 60 deg V grips
#: at 0.40; the 53 deg ramp's slippery face pushes 0.59.
RAMP_MU = 0.15


#: Each of the fork's geoms, kg: its plates, prongs, ramps and pad are the
#: plate's budget (`ArmSpec.plate_mass`), which the plate's own geom carries
#: the rest of.
FORK_GEOM_KG = 0.002


def _fork_kg(spec: ArmSpec) -> float:
  return fork_xml(spec).count("<geom ") * FORK_GEOM_KG


def _pad_x(spec: ArmSpec) -> float:
  """The lean-pad's axis in the plate frame: its front `pad_intrude` past a
  seated module's back face."""
  return spec.fork.vertex_x - MODULE_HALF_X + spec.fork.pad_intrude - PAD_R


def arm_mjcf(spec: ArmSpec, prefix: str = "") -> dict[str, str | list[float]]:
  """The arm as MJCF fragments for `model.body_xml(arm=...)`: `torso` (the
  motor stack, geoms on the torso), `body` (the links and the plate, a child
  of the torso after the legs), `tendon`, `equality`, `actuator`, and
  `qpos` (the joints' stowed values, appended to every keyframe)."""
  s = spec
  m = s.motor
  x0, y0, z0 = s.shoulder_x, s.y, s.shoulder_z
  stack = (f'<geom name="{prefix}arm_motors" class="visual" type="cylinder" '
           f'size="{_v(m.diameter / 2, m.length)}" pos="{_v(x0, y0, z0)}" '
           f'quat="{_quat_x(math.pi / 2)}" mass="{_f(2 * m.mass)}" '
           f'rgba="0.25 0.25 0.28 1"/>')
  rods = s.rods_mass / 2
  wrist_joint = ""
  wrist_geom = ""
  if s.level in ("parallelogram", "wrist"):
    wrist_joint = (f'<joint name="{prefix}arm_wrist" class="arm_joint" '
                   f'range="-6.3 6.3"/>')
    if s.level == "wrist":
      wm = s.wrist_motor or s.motor
      wrist_joint = wrist_joint.replace(
        "/>", f' armature="{_f(wm.armature)}" frictionloss="{_f(ARM_FRICTION_NM)}"/>')
  if s.level == "wrist":
    wm = s.wrist_motor or s.motor
    wrist_geom = (f'<geom name="{prefix}arm_wrist_motor" class="visual" type="cylinder" '
                  f'size="{_v(wm.diameter / 2, wm.length / 2)}" '
                  f'quat="{_quat_x(math.pi / 2)}" mass="{_f(wm.mass)}" '
                  f'rgba="0.25 0.25 0.28 1"/>')
  body = f"""
      <body name="{prefix}arm_upper" pos="{_v(x0, y0, z0)}" childclass="arm">
        <joint name="{prefix}arm_shoulder" class="arm_joint" range="{_v(*s.shoulder_range)}"
               armature="{_f(m.armature)}" frictionloss="{_f(ARM_FRICTION_NM)}"/>
        <geom name="{prefix}arm_upper" class="arm_link" fromto="{_v(0, 0, 0, s.upper, 0, 0)}"
              mass="{_f(s.upper_mass + rods)}"/>
        <body name="{prefix}arm_fore" pos="{_v(s.upper, 0, 0)}">
          <joint name="{prefix}arm_elbow" class="arm_joint" range="{_v(*s.elbow_range)}"/>
          <geom name="{prefix}arm_fore" class="arm_link" fromto="{_v(0, 0, 0, s.fore, 0, 0)}"
                mass="{_f(s.fore_mass + rods)}"/>
          {wrist_geom}
          <body name="{prefix}arm_plate" pos="{_v(s.fore, 0, 0)}">
            {wrist_joint}
            <geom name="{prefix}arm_plate" class="arm_link" type="box" size="0.03 0.03 0.012"
                  pos="0.02 0 -0.012" mass="{_f(s.plate_mass - _fork_kg(s))}"/>
          {fork_xml(s, prefix)}
          </body>
        </body>
      </body>"""
  tendon = (f'\n    <fixed name="{prefix}arm_fore" armature="{_f(m.armature)}" '
            f'frictionloss="{_f(ARM_FRICTION_NM)}">'
            f'<joint joint="{prefix}arm_shoulder" coef="1"/>'
            f'<joint joint="{prefix}arm_elbow" coef="1"/></fixed>')
  equality = ""
  if s.level == "parallelogram":
    tendon += (f'\n    <fixed name="{prefix}arm_level">'
               f'<joint joint="{prefix}arm_shoulder" coef="1"/>'
               f'<joint joint="{prefix}arm_elbow" coef="1"/>'
               f'<joint joint="{prefix}arm_wrist" coef="1"/></fixed>')
    equality = (f'\n    <tendon name="{prefix}arm_level" tendon1="{prefix}arm_level" '
                f'polycoef="0 0 0 0 0" solref="{LEVEL_SOLREF}"/>')
  act = [f'<motor name="{prefix}arm_shoulder" joint="{prefix}arm_shoulder" '
         f'ctrlrange="{_v(-m.peak_torque, m.peak_torque)}"/>',
         f'<motor name="{prefix}arm_elbow" tendon="{prefix}arm_fore" '
         f'ctrlrange="{_v(-m.peak_torque, m.peak_torque)}"/>']
  if s.level == "wrist":
    wm = s.wrist_motor or s.motor
    act.append(f'<motor name="{prefix}arm_wrist" joint="{prefix}arm_wrist" '
               f'ctrlrange="{_v(-wm.peak_torque, wm.peak_torque)}"/>')
  qs, qe = s.stow
  qpos = [qs, qe] + ([-(qs + qe)] if s.level in ("parallelogram", "wrist") else [])
  return {"torso": stack, "body": body, "tendon": tendon, "equality": equality,
          "actuator": "\n    ".join(act), "qpos": qpos,
          "default": ARM_DEFAULTS.format(friction=_f(BEARING_FRICTION_NM), tube=_f(TUBE_R),
                                         fork_kg=_f(FORK_GEOM_KG))}


#: The level parallelogram's compliance: a rod and two pin joints, stiff.
#: The equality's time constant, s (two physics steps is the stiffest the
#: 2 ms step integrates, as the dock's pins).
LEVEL_SOLREF = "0.004 1"
#: Coulomb friction at a motor's joint (the shoulder, and the elbow's drive
#: on its tendon), N*m: the legs' nominal (`actuator.FRICTION_NOMINAL`,
#: Katz's 0.09 + 4 % of load). A passive pivot is a ball bearing's.
ARM_FRICTION_NM = 0.1
BEARING_FRICTION_NM = 0.01

ARM_DEFAULTS = """
    <default class="arm">
      <joint type="hinge" axis="0 -1 0" armature="0" frictionloss="{friction}"
             damping="0"/>
      <default class="arm_joint"/>
      <default class="arm_link">
        <geom type="capsule" size="{tube}" contype="1" conaffinity="0" group="1"
              rgba="0.9 0.6 0.1 1"/>
      </default>
      <default class="fork">
        <geom contype="1" conaffinity="0" group="1" friction="0.4 0.005 0.0001"
              priority="1" mass="{fork_kg}" rgba="0.30 0.32 0.36 1"/>
      </default>
    </default>"""


# ---- driving it ------------------------------------------------------------------

#: The arm's servo loop: the GDS68's MIT mode (Kp 0-500 N*m/rad, Kd 0-5
#: N*m*s/rad) on each motor, plus the arm's own gravity as a feed-forward
#: torque off its model -- a real controller's URDF, and the tool's mass
#: from the catalog when one is carried. Kp over the arm's ~0.1 kg*m^2 about
#: the shoulder is a ~4 Hz loop at damping ratio ~0.8.
ARM_KP, ARM_KD = 60.0, 4.0
#: The fastest a joint target is ramped, rad/s (CLAUDE.md: every position
#: setpoint is ramped): a quarter of the motor's no-load speed at 48 V.
ARM_SLEW = 1.5


class ArmDriver:
  """One arm's motors on one robot: `aim(shoulder, elbow)` sets the joints'
  targets (`solve`'s), `step()` once per physics step before `mj_step`
  writes the two torques, clipped to the motor's envelope. The elbow's
  motor holds the forearm's ABSOLUTE angle (shoulder + elbow), so that is
  the coordinate its loop runs on.

  ⚠ EVERYTHING IT READS IS `qpos` AND `qvel` (issue #405): the gravity is
  the arm's own planar model (`gravity`), never the Jacobians of the last
  forward pass. A restart puts a world back and forwards it at the saved
  instant, where a running world's step reads the kinematics one step old
  -- a controller reading them every step parts the two worlds at the
  first step back (`scripts/determinism_spike.py --resume-at`) -- and the
  three Jacobians, 122 columns wide in the house, were 43 of its 100 us a
  step."""

  def __init__(self, model, data, spec: ArmSpec, prefix: str = ""):
    from pluggybot.legs.actuator import JointLimits
    from pluggybot.telemetry.protocol import ROBOT_ROOT
    if spec.level != "parallelogram":
      raise ValueError("this driver holds a shoulder and an elbow with the "
                       "plate kept level: the wrist (b) and the body (c) "
                       "options are the reach table's, never driven")
    self.m, self.d, self.spec = model, data, spec
    j = {n: model.joint(f"{prefix}arm_{n}").id
         for n in ("shoulder", "elbow", "wrist") if _has(model, f"{prefix}arm_{n}")}
    self.qadr = {n: model.jnt_qposadr[i] for n, i in j.items()}
    self.vadr = {n: model.jnt_dofadr[i] for n, i in j.items()}
    self._qs, self._qe, self._qw = (int(self.qadr[n]) for n in ("shoulder", "elbow", "wrist"))
    self._vs, self._ve = int(self.vadr["shoulder"]), int(self.vadr["elbow"])
    root = model.body(f"{prefix}{ROBOT_ROOT}").id
    self._quat = int(model.jnt_qposadr[model.body_jntadr[root]]) + 3
    self.act = [model.actuator(f"{prefix}arm_shoulder").id,
                model.actuator(f"{prefix}arm_elbow").id]
    self.bodies = [model.body(f"{prefix}arm_{b}").id for b in ("upper", "fore", "plate")]
    # The planar model: each link's mass and its CoM in its own frame (x
    # along it, z up off it), and the seat's place on the plate.
    self._links = [(float(model.body_mass[b]), float(model.body_ipos[b][0]),
                    float(model.body_ipos[b][2])) for b in self.bodies]
    seat = model.site(self._seat()).pos
    self._seat_xz = (float(seat[0]), float(seat[2]))
    lim = JointLimits.of(spec.motor)
    self.peak, self.sat, self.noload = (float(lim.peak[0]), float(lim.saturation[0]),
                                        float(lim.noload[0]))
    self.kp, self.kd = ARM_KP, ARM_KD
    #: How fast the target walks to the goal, rad/s: ARM_SLEW, or slower
    #: for a move that asks for it.
    self.slew = ARM_SLEW
    self.target = np.array(self.q())
    self.goal = self.target.copy()
    #: A carried tool's mass and its CoM, the SEAT's frame offset (level).
    self.payload = (0.0, (0.0, 0.0))

  def q(self) -> tuple[float, float]:
    """(shoulder, forearm's absolute angle)."""
    d = self.d
    qs = d.qpos[self.qadr["shoulder"]]
    return float(qs), float(qs + d.qpos[self.qadr["elbow"]])

  def qd(self) -> tuple[float, float]:
    d = self.d
    vs = d.qvel[self.vadr["shoulder"]]
    return float(vs), float(vs + d.qvel[self.vadr["elbow"]])

  def aim(self, shoulder: float, elbow: float) -> None:
    """The joint targets (the ELBOW's relative angle, as `solve` gives it);
    the driver ramps toward them at ARM_SLEW."""
    self.goal = np.array([shoulder, shoulder + elbow])

  def hold_at(self, shoulder: float, elbow: float) -> None:
    """Hold the arm where it IS, at these joint angles, with no ramp: for a
    body put there by hand (a warp, a stand-up), whose joints are there."""
    self.goal = np.array([shoulder, shoulder + elbow])
    self.target = self.goal.copy()

  def arrived(self, tol: float = 0.01) -> bool:
    return bool(np.all(np.abs(np.array(self.q()) - self.goal) < tol)
                and np.all(self.target == self.goal))

  def gravity(self) -> np.ndarray:
    """What the two motors hold against gravity, N*m: the arm's own links
    and the payload, split between the motors as the linkage splits it
    (`gravity_torques`' algebra). The arm moves in the torso's x-z plane
    about parallel axes, so each joint holds the moment of what hangs
    beyond it -- off the joint angles and the torso's attitude, both in
    `qpos` (the class docstring). Equal to MuJoCo's Jacobians to 1e-15 N*m
    at any pose, attitude and payload (`tests/test_arm.py`)."""
    return np.array(self._gravity())

  def _gravity(self) -> tuple[float, float]:
    q = self.d.qpos
    w, x, y, z = q[self._quat:self._quat + 4]
    # gravity in the torso frame: the third row of the torso's rotation
    gx = -G * 2.0 * float(x * z - w * y)
    gz = -G * (1.0 - 2.0 * float(x * x + y * y))
    qs = float(q[self._qs])
    a2 = qs + float(q[self._qe])
    a3 = a2 + float(q[self._qw])
    c1, s1 = math.cos(qs), math.sin(qs)
    c2, s2 = math.cos(a2), math.sin(a2)
    c3, s3 = math.cos(a3), math.sin(a3)
    (mu, ux, uz), (mf, fx, fz), (mp, px, pz) = self._links
    sx, sz = self.spec.shoulder_x, self.spec.shoulder_z
    ex, ez = sx + self.spec.upper * c1, sz + self.spec.upper * s1
    wx, wz = ex + self.spec.fore * c2, ez + self.spec.fore * s2
    bodies = [(mu, sx + ux * c1 - uz * s1, sz + ux * s1 + uz * c1),
              (mf, ex + fx * c2 - fz * s2, ez + fx * s2 + fz * c2),
              (mp, wx + px * c3 - pz * s3, wz + px * s3 + pz * c3)]
    kg, (ox, oz) = self.payload
    if kg:
      tx, tz = self._seat_xz[0] + ox, self._seat_xz[1] + oz
      bodies.append((kg, wx + tx * c3 - tz * s3, wz + tx * s3 + tz * c3))

    def held(ax: float, az: float, links) -> float:
      return sum(m * ((bz - az) * gx - (bx - ax) * gz) for m, bx, bz in links)
    hs, he, hw = held(sx, sz, bodies), held(ex, ez, bodies[1:]), held(wx, wz, bodies[2:])
    return hs - he, he - hw

  def _seat(self) -> str:
    name = self.m.body(self.bodies[2]).name
    return name.replace("arm_plate", "arm_seat")

  def step(self) -> np.ndarray:
    """The two torques for this physics step, written to `ctrl`. Two joints,
    so plain floats: numpy's per-call overhead was a fifth of the step."""
    d, t, g = self.d, self.target, self.goal
    lim = self.slew * self.m.opt.timestep
    t[0] += min(max(g[0] - t[0], -lim), lim)
    t[1] += min(max(g[1] - t[1], -lim), lim)
    qs = float(d.qpos[self._qs])
    qf = qs + float(d.qpos[self._qe])
    vs = float(d.qvel[self._vs])
    vf = vs + float(d.qvel[self._ve])
    gs, ge = self._gravity()
    tau_s = self._envelope(self.kp * (t[0] - qs) - self.kd * vs + gs, vs)
    tau_e = self._envelope(self.kp * (t[1] - qf) - self.kd * vf + ge, vf)
    d.ctrl[self.act[0]] = tau_s
    d.ctrl[self.act[1]] = tau_e
    return np.array([tau_s, tau_e])

  def _envelope(self, tau: float, qd: float) -> float:
    """`actuator.envelope` for one motor: the DC line through (0, sat) and
    (noload, 0), clipped at the peak; the forearm's motor turns at the
    forearm's absolute rate."""
    hi = min(max(self.sat * (1.0 - qd / self.noload), 0.0), self.peak)
    lo = min(max(self.sat * (-1.0 - qd / self.noload), -self.peak), 0.0)
    return min(max(tau, lo), hi)


def _has(model, joint: str) -> bool:
  try:
    model.joint(joint)
    return True
  except KeyError:
    return False
