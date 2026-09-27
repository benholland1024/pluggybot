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

#: A link's tube and the rods beside it: radius of the capsule the sim
#: collides and draws, m (a 20/18 mm carbon tube with its rod ends).
TUBE_R = 0.011


@dataclass(frozen=True)
class ForkSpec:
  """The coupling's arm half, in the PLATE frame. Lengths in metres."""

  #: The V-notches' vertex (where a seated peg's axis sits, less the peg's
  #: radius over sin 45 deg) ahead of and below the wrist axis.
  vertex_x: float = 0.085
  vertex_z: float = -0.030
  #: The two V-notches, either side of the arm's plane, under the peg's two
  #: conductors (the poles).
  fork_y: float = 0.085
  #: A V plate: half-length along its flank, half-width across, half-thick.
  v_half_len: float = 0.011
  v_half_w: float = 0.006
  v_thick: float = 0.003
  #: The end-stops: a ramp outboard of each peg end, from `stop_y` (1.5 mm
  #: past a centred peg's end: a tool rides the fork within that) rising
  #: `ramp_h` over `ramp_w`: a peg end the fork lifts under slides down it
  #: and inward. Steeper than the V's friction angle (tan^-1 0.57, 30 deg),
  #: or it holds. The rover's stops, 4 mm past a peg with no ramps, took
  #: nothing 10 mm off (`--capture --rover`).
  stop_y: float = 0.1115
  ramp_w: float = 0.020
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
#: body's FALLEN) the arm folds to its stow. Held out in its carry pose it
#: props the fallen robot on its side and the get-up policy -- trained with
#: a placeholder arm that collides with nothing -- cannot roll it (flown:
#: folded, it stood in 0.5 s; held out, never).
FOLD_ON_FALL_COS = math.cos(math.radians(60.0))

#: THE TOOL ENVELOPE a tool on this coupling must fit (the workshop's
#: validator's, #375 step 4; `rack.coupling`'s are the rover's). Measured
#: (the spike's `--envelope`: carried through a 1.0 m/s trot and a stop):
#:   mass  a tool of 0.60 kg stayed seated with its CoM up to 60 mm off its
#:         peg; the arm holds 0.61 kg (the claw and a 0.4 kg cube) straight
#:         out at 39 % of its motors' continuous rating. The ceiling keeps a
#:         payload's room under both.
#:   ahead a tool's CoM on or AHEAD of its peg (away from the robot) leans
#:         it onto the lean-pad, 6-8 deg; toward the robot it swings free
#:         (77 deg) or into the pad's post. 0.40 kg 90 mm ahead (0.35 N*m)
#:         slipped its plate past the pad and flopped to 56 deg; 0.60 kg 60
#:         mm ahead (the same moment) held: the lever binds, and the moment.
#:   drop  under the peg at the carry pose (`CARRY`) a pendant this long
#:         keeps 2 cm over the LIDAR's scan plane (`test_arm.py` reads it).
#:   power the peg is still the connector (`rack.coupling.PEG_POWER_W`), its
#:         preload the tool's weight; flown, it opened for at most 56 ms (down
#:         a flight of stairs) against the 200 ms holding capacitor.
TOOL_MAX_KG = 0.40
TOOL_MAX_AHEAD_M = 0.060
TOOL_MAX_MOMENT_NM = 0.35
TOOL_MAX_DROP_M = 0.20

#: The coupling's verbs (the rover's, `rack.coupling`): the fork's V vertex
#: runs FORK_DROP under a hanging peg on the way in, the lift takes it LIFT
#: above that, and it backs out BACK_OUT; a return is the reverse. LIFT
#: carries the peg 10 mm over the trays' V corners (15.6 mm over a vertex):
#: the rover's 36 mm cleared them by -0.4 mm, and a return aimed 10 mm low
#: knocked the tool off the tray.
FORK_DROP = 0.022
LIFT = 0.046
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
            peg_r: float = 0.003) -> tuple[float, float]:
  """Where a seated peg's axis sits, in the torso frame's x-z."""
  wx, wz = wrist_xz(spec, qs, qe)
  a = plate_angle(spec, qs, qe, qw)
  fx, fz = spec.fork.vertex_x, spec.fork.vertex_z + peg_r * math.sqrt(2)
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
               peg_r: float = 0.003, near=None) -> tuple[float, float] | None:
  """(shoulder, elbow) putting a seated peg's axis at (x, z), the plate
  level (the parallelogram's case: its angle does not depend on the joints)."""
  fx, fz = spec.fork.vertex_x, spec.fork.vertex_z + peg_r * math.sqrt(2)
  return solve(spec, x - fx, z - fz, elbow, near)


def solve_vertex(spec: ArmSpec, x: float, z: float, near=None,
                 elbow: str = "up") -> tuple[float, float] | None:
  """(shoulder, elbow) putting the fork's V vertex at (x, z), plate level."""
  return solve(spec, x - spec.fork.vertex_x, z - spec.fork.vertex_z, elbow, near)


def within(spec: ArmSpec, q: tuple[float, float]) -> bool:
  qs, qe = q
  return (spec.shoulder_range[0] <= qs <= spec.shoulder_range[1]
          and spec.elbow_range[0] <= qe <= spec.elbow_range[1])


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
  """An upward-open V (the peg's axis along y) of two 45 deg plates whose
  upper faces are the flanks, vertex at (x, y, z): `rack.coupling`'s."""
  off = f.v_half_len / math.sqrt(2)
  out = []
  for s, lbl in ((-1, "a"), (1, "b")):
    out.append(
      f'<geom name="{prefix}{lbl}" class="fork" type="box" '
      f'size="{_v(f.v_half_len, f.v_half_w, f.v_thick)}" '
      f'pos="{_v(x + s * off, y, z + off - f.v_thick * math.sqrt(2))}" '
      f'quat="{_quat_y(-s * math.pi / 4)}"/>')
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
    g.append(f'<geom name="{prefix}arm_ramp_{lbl}" class="fork" type="box" '
             f'friction="{_f(RAMP_MU)} 0.005 0.0001" priority="2" '
             f'size="{_v(0.006, math.hypot(f.ramp_w, f.ramp_h) / 2, 0.0015)}" '
             f'pos="{_v(vx, s * (f.stop_y + f.ramp_w / 2), vz + f.ramp_h / 2 - 0.0015 * math.sqrt(2))}" '
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
#: the far V's grip: at the peg's 0.4 that is 0.30 of the tool's weight
#: against 0.28, and a pick from 15 mm off at 4 deg left the peg's end on
#: the ramp and its pole open (with the rover's 36 mm lift; since the 46 mm
#: one both seat, and the face is the margin); at 0.15, 0.43 against 0.28.
RAMP_MU = 0.15


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
                  pos="0.02 0 -0.012" mass="{_f(s.plate_mass)}"/>
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
          "default": ARM_DEFAULTS.format(friction=_f(BEARING_FRICTION_NM), tube=_f(TUBE_R))}


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
              priority="1" mass="0.002" rgba="0.30 0.32 0.36 1"/>
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
  the coordinate its loop runs on."""

  def __init__(self, model, data, spec: ArmSpec, prefix: str = ""):
    import mujoco
    from pluggybot.legs.actuator import JointLimits
    self.m, self.d, self.spec = model, data, spec
    j = {n: model.joint(f"{prefix}arm_{n}").id
         for n in ("shoulder", "elbow", "wrist") if _has(model, f"{prefix}arm_{n}")}
    self.qadr = {n: model.jnt_qposadr[i] for n, i in j.items()}
    self.vadr = {n: model.jnt_dofadr[i] for n, i in j.items()}
    self.act = [model.actuator(f"{prefix}arm_shoulder").id,
                model.actuator(f"{prefix}arm_elbow").id]
    self.bodies = [model.body(f"{prefix}arm_{b}").id for b in ("upper", "fore", "plate")]
    lim = JointLimits.of(spec.motor)
    self.peak, self.sat, self.noload = lim.peak[0], lim.saturation[0], lim.noload[0]
    self.kp, self.kd = ARM_KP, ARM_KD
    self.target = np.array(self.q())
    self.goal = self.target.copy()
    #: A carried tool's mass and its CoM, the SEAT's frame offset (level).
    self.payload = (0.0, (0.0, 0.0))
    self._jac = np.zeros((3, model.nv))
    self._mj = mujoco

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

  def arrived(self, tol: float = 0.01) -> bool:
    return bool(np.all(np.abs(np.array(self.q()) - self.goal) < tol)
                and np.all(self.target == self.goal))

  def gravity(self) -> np.ndarray:
    """What the two motors hold against gravity, N*m: the arm's own bodies
    and the payload, each weight through its Jacobian, split between the
    motors as the linkage splits it (`gravity_torques`' algebra)."""
    m, d = self.m, self.d
    gen = np.zeros(m.nv)
    for b in self.bodies:
      self._mj.mj_jacBodyCom(m, d, self._jac, None, b)
      gen += self._jac[2] * m.body_mass[b] * 9.81
    kg, (ox, oz) = self.payload
    if kg:
      sid = m.site(self._seat()).id
      point = d.site_xpos[sid] + d.site_xmat[sid].reshape(3, 3) @ np.array([ox, 0.0, oz])
      self._mj.mj_jac(m, d, self._jac, None, point, m.site_bodyid[sid])
      gen += self._jac[2] * kg * 9.81
    gs = gen[self.vadr["shoulder"]]
    ge = gen[self.vadr["elbow"]]
    gw = gen[self.vadr["wrist"]] if "wrist" in self.vadr else 0.0
    return np.array([gs - ge, ge - gw])

  def _seat(self) -> str:
    name = self.m.body(self.bodies[2]).name
    return name.replace("arm_plate", "arm_seat")

  def step(self) -> np.ndarray:
    dt = self.m.opt.timestep
    self.target += np.clip(self.goal - self.target, -ARM_SLEW * dt, ARM_SLEW * dt)
    q, qd = np.array(self.q()), np.array(self.qd())
    tau = self.kp * (self.target - q) - self.kd * qd + self.gravity()
    # The motor's envelope at the joint's speed (`actuator.envelope`); the
    # forearm's motor turns at the forearm's absolute rate.
    from pluggybot.legs.actuator import envelope
    lo, hi = envelope(qd, np.full(2, self.sat), np.full(2, self.peak),
                      np.full(2, self.noload))
    tau = np.clip(tau, lo, hi)
    self.d.ctrl[self.act] = tau
    return tau


def _has(model, joint: str) -> bool:
  try:
    model.joint(joint)
    return True
  except KeyError:
    return False
