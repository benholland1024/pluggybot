"""What the arm can open from its lying stance, and what its torques can
measure (issue #469, the first step of #466; SimNotes, "Opening a box from
lying").

Three candidate mechanisms for demo 1, each alone with one robot in the
house's empty storeroom -- the spike's own scene, never the served world:

  handle  a box whose lid is lifted by a DROP HANDLE: a knob on an L hung
          from its own pin, parallel to the lid's hinge, so the claw's grip
          does not turn as the lid does
  lip     the same box, its lid lifted from under its front lip by the
          closed jaws pushing up (a finger notch takes the claw's pendant)
  drawer  a low drawer pulled straight back by a knob on its front

The robot lies with the claw on its fork and the knob (or the lip) in front
of it as a walk-in leaves it (`--walkin`), inside the claw's reach.

  --walkin      the spread a real walk-in leaves: the claw's tag-steered
                walk-in to a cube from jittered starts, lying where it
                stops -- the cube ahead and across, and the heading against
                the line it walked in on
  --window      the force window: a force ramped at the jaws each way until
                the claw leaves its seat, or swings off
  --table       each candidate over `--n` tries at the walk-in's spread, its
                path on the true arc: opens, the tool kept, the peak forces
                on the claw, at the stiff gains and a compliant one
  --tolerance   the path planned off a hinge guessed wrong (the radius off
                by up to 4 cm; the drawer's pull tilted), at four gains
  --torques     the force at the tool off the drivers' torque readings
                against the contact force, and the arm's own friction from
                an empty sweep
  --fit         the oracle fit: code that knows the truth fits the lid's
                gravity torque, spring, friction and damping to readings
                over sweeps up and down at two speeds, over `--n` set-outs;
                and mass against its lever arm
  --latch       a magnetic catch: its release force off the torques, and
                the jolt of its release
  (default)     a filmstrip of each candidate opening, mechanism_spike.png

  --all --into DIR  every table above, one file a mode in DIR, and the
                filmstrip: a batch, flown on a CPU pod (`training/pod.sh
                create-cpu`), or locally under a memory cap

Usage:
  MUJOCO_GL=egl uv run python scripts/mechanism_spike.py [--table|--fit|...]
  systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
    uv run python scripts/mechanism_spike.py --all --into DIR --jobs 4
"""

import os

# One BLAS thread before numpy loads (quad_spike.py: the policy's products
# are small enough that six threads doubled a step).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pluggybot.challenge.stack import BLOCK_HALF  # noqa: E402
from pluggybot.legs import arm as am  # noqa: E402
from pluggybot.legs import dock as dk  # noqa: E402
from pluggybot.legs import rack as rk  # noqa: E402
from pluggybot.rack.coupling import PEG_ABOVE_BODY  # noqa: E402

# ---- the candidates ------------------------------------------------------------
# The BOX frame: the origin on the floor under the middle of the box's front
# face (the robot's side), +x into the box (a robot square on faces +x), z up.

#: The box: a toy chest's size, its walls and lid 12 mm board.
BOX_D, BOX_W, BOX_H = 0.22, 0.30, 0.14
WALL_T = LID_T = 0.012
#: The knob the claw grips is the cube the claw is proven on (26 mm), on a
#: stem out of what it moves: the stem leaves the jaws out of their back,
#: under the crossbar that joins them.
KNOB = BLOCK_HALF
STEM, STEM_HALF_W = 0.030, 0.004
#: The drop handle: its pin this far in front of the lid's front edge, at
#: the lid's mid-thickness, and its arm down to the stem. The claw holds the
#: handle as it hangs while the lid turns under it, nearly 90 deg by the
#: top, its 40 mm arm then along the lid: 20 mm out, it lay in the lid's
#: front edge, unseen -- a body and its parent never collide, so the
#: handle's pairs with the board are named.
PIN_AHEAD, HANDLE_DROP = 0.050, 0.040
PIN_Z = LID_T / 2
#: The handle's parts, kg: its arm, its stem and the knob.
HANDLE_KG = (0.010, 0.005, 0.020)
#: The lip: the lid overhangs the front face, notched in its middle for the
#: claw's pendant (10 mm square) to rise into.
LIP, NOTCH_HALF_W = 0.030, 0.012
#: The crossbar's top over the jaws' middle (`legs.rack.claw_face`): what
#: pushes up under the lip.
BAR_TOP = rk.CLAW_PAD_HALF[2] + 0.008
#: ...and how far in under the lip its middle goes, from the lip's edge.
UNDER_LIP = 0.012
#: The drawer: its travel, the middle of its front panel's height, and the
#: carcass it slides in (the box's size, open at the front).
DRAWER_TRAVEL = 0.15
DRAWER_Z = 0.055
#: Each candidate's opening asked, and what counts as open: the lids' angle
#: (rad: 69 and 57 deg), the drawer's travel (m).
OPEN_TO = {"handle": 1.20, "lip": 1.20, "drawer": 0.12}
OPENED = {"handle": 1.0, "lip": 1.0, "drawer": 0.10}
#: The sweep's rate: rad/s for a lid, m/s for the drawer.
RATE = {"handle": 0.25, "lip": 0.25, "drawer": 0.03}
#: Where the sweep turns back down, short of shut: a lid let down onto its
#: stop rings the arm.
CLOSE_TO = {"handle": 0.05, "lip": 0.05, "drawer": 0.005}


@dataclass(frozen=True)
class Mech:
  """A candidate and its hidden parameters. The lids' are about the hinge
  (N*m, rad), the drawer's along its slide (N, m)."""

  kind: str = "handle"
  #: The lid's board, and a hidden weight in it this fraction of the way
  #: from the hinge to the front edge, this high over the hinge's line.
  lid_kg: float = 0.30
  lump_kg: float = 0.0
  lump_at: float = 0.5
  lump_z: float = LID_T / 2
  #: The drawer, loaded.
  drawer_kg: float = 0.5
  #: The joint: a spring (and where it is slack), damping, Coulomb friction.
  #: The lids' shut by `CLOSING_MARGIN_NM` at 69 deg swept at `RATE`.
  stiffness: float = 0.0
  springref: float = 0.0
  damping: float = 0.04
  friction: float = 0.04
  #: A magnetic catch holding it shut, N at the knob (0: none).
  latch_n: float = 0.0

  @classmethod
  def nominal(cls, kind: str) -> "Mech":
    if kind == "drawer":
      return cls(kind="drawer", damping=2.0, friction=1.0)
    return cls(kind=kind)


def _f(v: float) -> str:
  return f"{v:.6g}"


def _v(*vs: float) -> str:
  return " ".join(_f(v) for v in vs)


WOOD = 'rgba="0.55 0.42 0.30 1"'
LID_WOOD = 'rgba="0.70 0.55 0.38 1"'
METAL = 'rgba="0.20 0.45 0.75 1"'


def _knob(name: str, pos) -> str:
  return (f'<body name="{name}_knob_body" pos="{_v(*pos)}">'
          f'<geom name="{name}_knob" type="box" size="{_v(KNOB, KNOB, KNOB)}" '
          f'mass="{_f(HANDLE_KG[2])}" {METAL}/></body>')


def mechanism_xml(m: Mech, name: str = "mech") -> str:
  """The candidate as a `<mujoco>` to attach at the box's pose: a static
  base, and the moving part on its joint. The lid's rest on the walls is its
  hinge's lower limit, and the drawer's on its runners its slide: the lid
  and its base do not collide, nor the drawer and its carcass."""
  t, d, w, h = WALL_T, BOX_D, BOX_W, BOX_H
  g = [f'<geom name="{name}_floor" type="box" size="{_v(d / 2, w / 2, t / 2)}" '
       f'pos="{_v(d / 2, 0, t / 2)}" {WOOD}/>',
       f'<geom name="{name}_back" type="box" size="{_v(t / 2, w / 2, h / 2)}" '
       f'pos="{_v(d - t / 2, 0, h / 2)}" {WOOD}/>']
  for s, lbl in ((-1, "r"), (1, "l")):
    g.append(f'<geom name="{name}_side{lbl}" type="box" size="{_v(d / 2, t / 2, h / 2)}" '
             f'pos="{_v(d / 2, s * (w / 2 - t / 2), h / 2)}" {WOOD}/>')
  if m.kind == "drawer":
    g.append(f'<geom name="{name}_top" type="box" size="{_v(d / 2, w / 2, t / 2)}" '
             f'pos="{_v(d / 2, 0, h - t / 2)}" {WOOD}/>')
  else:
    g.append(f'<geom name="{name}_front" type="box" size="{_v(t / 2, w / 2, h / 2)}" '
             f'pos="{_v(t / 2, 0, h / 2)}" {WOOD}/>')
  joint_kw = (f'stiffness="{_f(m.stiffness)}" springref="{_f(m.springref)}" '
              f'damping="{_f(m.damping)}" frictionloss="{_f(m.friction)}"')
  if m.kind == "drawer":
    dd, dw, dh = (d - t) / 2 - 0.004, w / 2 - t - 0.003, 0.045
    moving = (
      f'<body name="{name}_drawer" pos="{_v(0, 0, DRAWER_Z)}">'
      f'<joint name="{name}_slide" type="slide" axis="1 0 0" '
      f'range="{_v(-DRAWER_TRAVEL, 0)}" {joint_kw}/>'
      f'<geom name="{name}_tray" type="box" size="{_v(dd, dw, 0.003)}" '
      f'pos="{_v(dd, 0, -dh + 0.003)}" mass="{_f(m.drawer_kg * 0.6)}" {LID_WOOD}/>'
      f'<geom name="{name}_panel" type="box" size="{_v(0.006, w / 2 - 0.002, dh + 0.008)}" '
      f'pos="{_v(-0.006, 0, 0)}" mass="{_f(m.drawer_kg * 0.4)}" {LID_WOOD}/>'
      f'<geom name="{name}_stem" type="box" size="{_v(STEM / 2, STEM_HALF_W, 0.004)}" '
      f'pos="{_v(-0.012 - STEM / 2, 0, 0)}" mass="{_f(HANDLE_KG[1])}" {METAL}/>'
      + _knob(name, (-0.012 - STEM - KNOB, 0, 0)) + '</body>')
    exclude = f'<exclude body1="{name}" body2="{name}_drawer"/>'
  else:
    lid = [f'<joint name="{name}_hinge" type="hinge" axis="0 1 0" range="0 1.9" '
           f'armature="0.0005" {joint_kw}/>',
           f'<geom name="{name}_lid" type="box" size="{_v(d / 2, w / 2, LID_T / 2)}" '
           f'pos="{_v(-d / 2, 0, LID_T / 2)}" mass="{_f(m.lid_kg)}" {LID_WOOD}/>']
    if m.lump_kg > 0.0:
      lid.append(f'<geom name="{name}_lump" type="box" size="0.01 0.01 0.004" '
                 f'pos="{_v(-d * m.lump_at, 0, m.lump_z)}" mass="{_f(m.lump_kg)}" '
                 f'contype="0" conaffinity="0" group="3" rgba="0.8 0.2 0.2 1"/>')
    if m.kind == "handle":
      lid.append(
        f'<geom name="{name}_bracket" type="box" size="{_v(PIN_AHEAD / 2 + 0.003, 0.006, 0.004)}" '
        f'pos="{_v(-d - PIN_AHEAD / 2 + 0.003, 0, PIN_Z)}" mass="0.005" {LID_WOOD}/>'
        f'<body name="{name}_handle" pos="{_v(-d - PIN_AHEAD, 0, PIN_Z)}">'
        f'<joint name="{name}_pin" type="hinge" axis="0 1 0" damping="0.002" '
        f'frictionloss="0.002" armature="0.0001"/>'
        f'<geom name="{name}_drop" type="box" size="{_v(0.003, STEM_HALF_W, HANDLE_DROP / 2)}" '
        f'pos="{_v(0, 0, -HANDLE_DROP / 2)}" mass="{_f(HANDLE_KG[0])}" {METAL}/>'
        f'<geom name="{name}_stem" type="box" size="{_v(STEM / 2 + 0.003, STEM_HALF_W, 0.004)}" '
        f'pos="{_v(-STEM / 2 + 0.003, 0, -HANDLE_DROP)}" mass="{_f(HANDLE_KG[1])}" {METAL}/>'
        + _knob(name, (-STEM - KNOB, 0, -HANDLE_DROP)) + '</body>')
    else:
      half = (w / 2 - NOTCH_HALF_W) / 2
      for s, lbl in ((-1, "r"), (1, "l")):
        # lacquered: the jaws' crossbar slides under it (priority over the
        # claw's own faces)
        lid.append(f'<geom name="{name}_lip{lbl}" type="box" size="{_v(LIP / 2, half, LID_T / 2)}" '
                   f'pos="{_v(-d - LIP / 2, s * (NOTCH_HALF_W + half), LID_T / 2)}" '
                   f'mass="0.008" friction="0.3 0.005 0.0001" priority="2" {LID_WOOD}/>')
    moving = (f'<body name="{name}_lid_body" pos="{_v(d, 0, h)}">' + "".join(lid) + '</body>')
    exclude = f'<exclude body1="{name}" body2="{name}_lid_body"/>'
    if m.kind == "handle":
      exclude += "".join(f'<pair geom1="{name}_{part}" geom2="{name}_lid"/>'
                         for part in ("drop", "stem", "knob"))
  return (f'<mujoco><compiler angle="radian"/><worldbody>'
          f'<body name="{name}">' + "".join(g) + moving + '</body>'
          f'</worldbody><contact>{exclude}</contact></mujoco>')


# ---- the geometry a path is planned on ---------------------------------------------

def _rot_y(th: float, v) -> np.ndarray:
  c, s = math.cos(th), math.sin(th)
  return np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])


HINGE = np.array([BOX_D, 0.0, BOX_H])
#: The handle's pin in the lid's frame, and the knob's middle off the pin
#: with the handle hanging plumb.
PIN = np.array([-BOX_D - PIN_AHEAD, 0.0, PIN_Z])
KNOB_OFF = np.array([-STEM - KNOB, 0.0, -HANDLE_DROP])
#: The lip's point the crossbar pushes, in the lid's frame (the lid's
#: underside passes through the hinge's axis).
LIP_POINT = np.array([-BOX_D - LIP + UNDER_LIP, 0.0, 0.0])


def jaws_at(kind: str, s: float, err: float = 0.0, at=None) -> np.ndarray:
  """Where the jaws' middle goes, box frame, with the mechanism opened to
  `s` (rad, or m out for the drawer) -- on a hinge `err` m further from the
  handle than it is (the radius guessed `err` long), or for the drawer a
  pull tilted to rise `err` over its travel. Shut, the jaws are on the knob
  where the robot saw it (`at`; else where the drawing puts it): the robot
  sees the knob, and guesses the hinge. The claw holds the handle as it
  hangs, so the knob moves as the pin does."""
  if kind == "drawer":
    x0 = np.array([-0.012 - STEM - KNOB, 0.0, DRAWER_Z]) if at is None else np.asarray(at)
    return x0 + np.array([-s, 0.0, err * s / OPEN_TO["drawer"]])
  p0 = PIN if kind == "handle" else LIP_POINT      # off the hinge, lid frame
  hinge = HINGE - err * p0 / np.linalg.norm(p0)
  closed = HINGE + p0
  point = hinge + _rot_y(s, closed - hinge)
  if kind == "handle":
    return point - closed + (closed + KNOB_OFF if at is None else np.asarray(at))
  return point - np.array([0.0, 0.0, BAR_TOP + 0.001])


# ---- the scene -------------------------------------------------------------------

#: The robot's start in the lab's storeroom (empty), facing +x.
START = (24.0, -3.0)
#: Lying, the knob's middle (the lip's point) this far ahead of the torso's
#: centre: the lids' arc carries it 0.16 m further and 0.22 m up, and with
#: the knob 0.60 m ahead the arm reaches only 66 deg of it.
LIE_AT_M = 0.46
#: The claw's working gains and the compliant ones: the driver's damping
#: kept at its ratio (Kd with the root of Kp).
GAINS = (am.ARM_KP, 30.0, 15.0, 8.0)


def kd_for(kp: float) -> float:
  return am.ARM_KD * math.sqrt(kp / am.ARM_KP)


class Scene:
  """The house, one candidate and the robot, which `lie` lays down in the
  storeroom with the claw on its fork; `place` sets the box in front of it.
  Every physics step a hook keeps the row `record` asks for."""

  def __init__(self, mech: Mech | None, name: str = "mech"):
    from pluggybot.home import world as home
    from pluggybot.legs import body as qb
    from pluggybot.legs import world as lw
    self.mech, self.name = mech, name
    spec = lw.home_spec(first_at=START)
    if mech is not None:
      frame = spec.worldbody.add_frame()
      frame.pos = [START[0] + 3.0, START[1], 0.0]
      spec.attach(mujoco.MjSpec.from_string(mechanism_xml(mech, name)), prefix="",
                  frame=frame)
    self.model = m = spec.compile()
    self.data = mujoco.MjData(m)
    self.body = qb.QuadBody(m, self.data, realtime=False, grid_bounds=home.GRID_BOUNDS)
    self.mis = self.body.mission
    self.claw = m.body("module_claw").id
    self.claw_kg = float(m.body_subtreemass[self.claw])
    self.grip_site = m.site(rk.CLAW_GRIP).id
    self.pads = [m.geom(p).id for p in rk.CLAW_PADS]
    self.acts = list(self.mis.arm.act)
    self.claw_geoms = self._geoms_under(self.claw)
    self.rows: list | None = None
    self.cmd = 0.0
    #: Where the jaws take hold, box frame (`look`), and the knob in the
    #: claw's frame once they have: what a slip along the pads is read off.
    self.grip_at = None
    self.grip0 = None
    self.mis.step_hooks.append(self._row)
    if mech is None:
      return
    self.box = m.body(name).id
    joint = m.joint(f"{name}_slide" if mech.kind == "drawer" else f"{name}_hinge").id
    self.qadr, self.vadr = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
    self.knob = m.geom(f"{name}_knob").id if mech.kind != "lip" else None
    self.mech_geoms = self._geoms_under(self.box)
    #: The catch: a torque (or force) on the joint, set each step.
    self.latch_on = mech.latch_n > 0.0

  def _geoms_under(self, root: int) -> np.ndarray:
    m = self.model
    out = []
    for b in range(m.nbody):
      x = b
      while x not in (0, root):
        x = int(m.body_parentid[x])
      if x == root:
        out.extend(range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b])))
    return np.array(sorted(out), dtype=np.int32)

  # ---- placing -----------------------------------------------------------------

  def lie(self) -> None:
    """Standing at the start with the claw, then down."""
    self.mount(*START, 0.0)
    self.body.run(self.mis.rest_routine())
    self.body.run(self.body.hold_routine(1.0))
    self.level, _, _ = dk.camera_mount(self.model, self.data,
                                       self.mis.handle.el("color_eye"), self.mis.root)

  def mount(self, x: float, y: float, heading: float) -> None:
    """Standing at (x, y, heading), the claw seated on the fork at the carry
    pose: a placement, not a swap (`draw_spike.mount_pen`'s)."""
    body, mis, m, d = self.body, self.mis, self.model, self.data
    body.start_at(x, y, heading)
    mis.arm.aim(*am.CARRY_Q)
    body.run(body.hold_routine(2.0))
    seat = m.site(mis.handle.el("arm_seat")).id
    adr = m.jnt_qposadr[m.joint("module_claw_free").id]
    dof = m.jnt_dofadr[m.joint("module_claw_free").id]
    rot = d.xmat[mis.root].reshape(3, 3)
    _, _, yaw = mis.true_pose()
    peg = d.site_xpos[seat] + rot @ np.array([0, 0, mis.arm_spec.fork.seat_rise() + 0.0003])
    d.qpos[adr:adr + 3] = peg - [0, 0, PEG_ABOVE_BODY]
    d.qpos[adr + 3:adr + 7] = [math.cos((yaw + math.pi) / 2), 0, 0,
                               math.sin((yaw + math.pi) / 2)]
    d.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(m, d)
    mis.carry("module_claw")
    body.run(body.hold_routine(1.0))

  def heading(self) -> tuple[np.ndarray, float]:
    """The torso's centre and its heading, world."""
    x, y, yaw = self.mis.true_pose()
    return np.array([x, y, float(self.data.xpos[self.mis.root][2])]), yaw

  def place(self, point, ahead: float, across: float = 0.0, yaw: float = 0.0) -> None:
    """The box set down so its `point` (box frame) is `ahead` m in front of
    the torso's centre and `across` m to its left, the box turned `yaw` off
    square about that point -- for a knob, the knob where it hangs."""
    m, d = self.model, self.data
    c, h = self.heading()
    a = h + yaw
    ca, sa = math.cos(a), math.sin(a)
    px, py = point[0], point[1]
    off = np.array([ca * px - sa * py, sa * px + ca * py])
    ch, sh = math.cos(h), math.sin(h)
    at = c[:2] + np.array([ch * ahead - sh * across, sh * ahead + ch * across])
    m.body_pos[self.box] = [at[0] - off[0], at[1] - off[1], 0.0]
    m.body_quat[self.box] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    mujoco.mj_forward(m, d)
    # ...and a second for the drop handle to swing to rest on its pin; then
    # the box moved so that the knob, where it hangs, is where it was asked
    self.body.run(self.body.hold_routine(1.0))
    if self.knob is not None:
      m.body_pos[self.box][:2] += at - d.geom_xpos[self.knob][:2]
      mujoco.mj_forward(m, d)
    #: Where it was set out: what a path is planned off, wherever it is now.
    self.placed = (d.xpos[self.box].copy(), d.xmat[self.box].reshape(3, 3).copy())
    self.grip_at = self.grip0 = None

  def look(self) -> None:
    """Where the jaws are to take hold, as the robot's eye would put it: the
    knob where it hangs -- the drop handle rests swung 19 deg on its pin, the
    knob 15 mm off its drawing -- or the lip's point."""
    if self.knob is None:
      self.grip_at = None
      return
    pos, rot = self.placed
    self.grip_at = rot.T @ (self.data.geom_xpos[self.knob] - pos)

  def path(self, s: float, err: float = 0.0) -> np.ndarray:
    """The jaws' target, heading frame, with the mechanism opened to `s`."""
    return self.to_heading(jaws_at(self.mech.kind, s, err, self.grip_at))

  def knob_in_claw(self) -> np.ndarray:
    """The knob's middle off the jaws' middle, in the claw's frame, m."""
    d = self.data
    return d.xmat[self.claw].reshape(3, 3).T @ (d.geom_xpos[self.knob]
                                                - d.site_xpos[self.grip_site])

  def away(self, gone: bool = True) -> None:
    """The box out of the arm's reach (the empty sweep's world), or back:
    lifted into the air, where its lid hangs shut on its stop -- under the
    floor, the floor's plane flung the lid open."""
    m = self.model
    m.body_pos[self.box] = self.placed[0] + ([0.0, 0.0, 3.0] if gone else 0.0)
    mujoco.mj_forward(m, self.data)

  def to_heading(self, p_box) -> np.ndarray:
    """A box-frame point, the box as it was set out, in the robot's heading
    frame (the torso's centre, x ahead, levelled)."""
    pos, rot = self.placed
    pw = pos + rot @ np.asarray(p_box)
    c, yaw = self.heading()
    ch, sh = math.cos(yaw), math.sin(yaw)
    r = pw - c
    return np.array([ch * r[0] + sh * r[1], -sh * r[0] + ch * r[1], r[2]])

  # ---- what the world reads ---------------------------------------------------------

  def opening(self) -> float:
    """The lid's angle (rad), or how far the drawer is out (m)."""
    q = float(self.data.qpos[self.qadr])
    return -q if self.mech.kind == "drawer" else q

  def contact_force(self) -> tuple[np.ndarray, np.ndarray]:
    """The mechanism's contact forces on the claw, N: in the heading frame,
    and in the torso's."""
    m, d = self.model, self.data
    total = np.zeros(3)
    if d.ncon:
      g = d.contact.geom[:d.ncon]
      on1 = np.isin(g[:, 0], self.claw_geoms) & np.isin(g[:, 1], self.mech_geoms)
      on2 = np.isin(g[:, 1], self.claw_geoms) & np.isin(g[:, 0], self.mech_geoms)
      f6 = np.zeros(6)
      for i in np.flatnonzero(on1 | on2):
        mujoco.mj_contactForce(m, d, int(i), f6)
        fw = d.contact.frame[i].reshape(3, 3).T @ f6[:3]
        # the normal points from geom1 to geom2: the force on geom2
        total += fw if on2[i] else -fw
    _, yaw = self.heading()
    ch, sh = math.cos(yaw), math.sin(yaw)
    head = np.array([ch * total[0] + sh * total[1], -sh * total[0] + ch * total[1], total[2]])
    return head, d.xmat[self.mis.root].reshape(3, 3).T @ total

  def seated(self) -> bool:
    return rk.tool_power(self.model, self.data, "module_claw")["powered"]

  def on_fork(self) -> bool:
    return self.mis.carrying == "module_claw"

  def gripped(self) -> bool:
    """Both pads within half a millimetre of the knob."""
    if self.knob is None:
      return False
    m, d = self.model, self.data
    return all(mujoco.mj_geomDistance(m, d, p, self.knob, 0.002, None) < 0.0005
               for p in self.pads)

  def claw_tilt(self) -> float:
    """The claw's swing on its peg off the plate's plumb, deg (+ its jaws
    away from the robot)."""
    d = self.data
    rel = d.xmat[self.mis.root].reshape(3, 3).T @ d.xmat[self.claw].reshape(3, 3)
    # the tool faces the robot: its x is the torso's -x
    return math.degrees(math.atan2(-rel[0, 2], rel[2, 2]))

  def readings(self) -> np.ndarray:
    """The two arm motors' torques as their drivers report them."""
    return np.array([self.body.arm_torque(a) for a in self.acts])

  # ---- the catch, and the row -------------------------------------------------------

  def latch_torque(self) -> float:
    """A magnetic catch at the knob's radius: its pull falls with the gap as
    a magnet's does, (1 + gap/g0)^-2, from `latch_n` shut to nothing past
    ten g0 (g0 0.5 mm at the knob)."""
    mech = self.mech
    r = {"handle": float(np.linalg.norm(PIN)), "lip": float(np.linalg.norm(LIP_POINT)),
         "drawer": 1.0}[mech.kind]
    gap = self.opening() * r
    g0 = 0.0005
    if gap > 10 * g0:
      return 0.0
    # ...less its pull at the cut, so it lets go to nothing
    pull = mech.latch_n * (1.0 + max(gap, 0.0) / g0) ** -2 - mech.latch_n * (1 + 10) ** -2
    # it holds the mechanism shut: against opening
    return -pull * r if mech.kind != "drawer" else pull

  def _row(self) -> None:
    if self.mech is not None and self.latch_on:
      self.data.qfrc_applied[self.vadr] = self.latch_torque()
    if self.rows is None:
      return
    head, torso = self.contact_force() if self.mech is not None else (np.zeros(3), np.zeros(3))
    arm = self.mis.arm
    slip = (1000 * float(np.linalg.norm(self.knob_in_claw() - self.grip0))
            if self.grip0 is not None else 0.0)
    self.rows.append((float(self.data.time), self.cmd,
                      self.opening() if self.mech is not None else 0.0,
                      *head, torso[0], torso[2],
                      self.seated(), self.gripped() if self.mech is not None else False,
                      *arm.q(), *arm.qd(), *self.readings(), *arm.gravity(),
                      self.claw_tilt(), slip))

  def record(self) -> None:
    self.rows = []

  def table(self) -> dict:
    """The rows kept since `record`, by column."""
    cols = ("t", "cmd", "open", "fx", "fy", "fz", "tx", "tz", "seated", "gripped",
            "qs", "qf", "vs", "vf", "rs", "re", "hs", "he", "tilt", "slip")
    a = np.array(self.rows, dtype=float) if self.rows else np.zeros((0, len(cols)))
    self.rows = None
    return {c: a[:, k] for k, c in enumerate(cols)}


# ---- the claw's motions ---------------------------------------------------------------

class Hand:
  """The claw at the mechanism: `tools.claw.ClawHand`'s motions, and a path
  followed step by step with the slide tracking it across."""

  def __init__(self, sc: Scene, kp: float = am.ARM_KP):
    from pluggybot.tools.claw import ClawHand
    self.sc = sc
    self.hand = ClawHand(sc.mis)
    sc.mis.arm.kp, sc.mis.arm.kd = kp, kd_for(kp)

  def run(self, routine):
    return self.sc.body.run(routine)

  def to(self, p_head, speed: float = 0.08) -> bool:
    return self.run(self.hand.to_routine(*p_head, self.sc.level, speed))

  def take(self) -> bool:
    """Onto the knob where it hangs and shut (the handle, the drawer); for
    the lip, the jaws shut and in under it."""
    sc, kind = self.sc, self.sc.mech.kind
    sc.look()
    p = sc.path(0.0)
    if kind == "lip":
      ok = self.to(p + [-0.06, 0.0, -0.012])
      ok = ok and self.to(p + [0.0, 0.0, -0.012], 0.04)
      return ok and self.to(p, 0.02)
    self.run(self.hand.jaws_routine(closed=False))
    ok = self.to(p + [0.0, 0.0, 0.045])
    ok = ok and self.to(p, 0.04)
    self.run(self.hand.jaws_routine(closed=True, settle=0.6))
    sc.grip0 = sc.knob_in_claw()
    return ok and sc.gripped()

  def follow(self, s0: float, s1: float, rate: float, err: float = 0.0) -> bool:
    """The jaws along the path from opening `s0` to `s1` at `rate` (rad/s,
    or m/s): the arm aimed and the slide set every physics step. False if a
    point is out of reach."""
    sc, m, d, mis = self.sc, self.sc.model, self.sc.data, self.sc.mis
    n = max(1, int(abs(s1 - s0) / rate / m.opt.timestep))
    for k in range(1, n + 1):
      s = s0 + (s1 - s0) * k / n
      sc.cmd = s
      p = sc.path(s, err)
      vx, vz, slide = self.hand.vertex_for(*p, sc.level)
      g = mis.arm.goal
      q = am.solve_vertex(mis.arm_spec, vx, vz, near=(float(g[0]), float(g[1] - g[0])))
      if q is None or abs(slide) > rk.CLAW_TRAVEL:
        return False
      mis.arm.aim(*q)
      d.ctrl[self.hand.slide] = slide
      self.run(mis._twist_routine(0.0, 0.0, 0.0))
    return True

  def hold(self, seconds: float) -> None:
    self.run(self.sc.body.hold_routine(seconds))

  def let_go(self) -> None:
    if self.sc.mech.kind != "lip":
      self.run(self.hand.jaws_routine(closed=False))
    x, z = self.sc.mis._vertex_goal()
    self.run(self.hand.move_routine(x - 0.03, z + 0.05, speed=0.05))


# ---- a try: up, down, and what happened -------------------------------------------------

def summary(t: dict, mech: Mech, claw_kg: float) -> dict:
  """A sweep's rows as a table row: how far it opened, the tool, the grip
  (and how far the knob slid along the pads, mm), the peak forces on the
  claw (heading frame: + up, + away from the robot, + to its left) and its
  swing on its peg (deg, + its jaws away from the robot)."""
  dt = np.diff(t["t"], prepend=t["t"][:1] - 0.002)
  out_run = longest = 0.0
  for s, step in zip(t["seated"], dt):
    out_run = 0.0 if s else out_run + step
    longest = max(longest, out_run)
  weight = claw_kg * 9.81
  opened = float(t["open"].max()) if len(t["open"]) else 0.0
  return {
    "opened": opened, "opens": opened >= OPENED[mech.kind],
    "seatedPct": round(100.0 * float(t["seated"].mean()), 1) if len(t["t"]) else 0.0,
    "openMs": round(1000 * longest),
    "gripPct": (round(100.0 * float(t["gripped"].mean()), 1)
                if mech.kind != "lip" and len(t["t"]) else None),
    "slipMm": (round(float(t["slip"].max()), 1)
               if mech.kind != "lip" and len(t["t"]) else None),
    "up": round(float(t["fz"].max()), 2), "down": round(float(-t["fz"].min()), 2),
    "away": round(float(t["fx"].max()), 2), "toward": round(float(-t["fx"].min()), 2),
    "side": round(float(np.abs(t["fy"]).max()), 2),
    "upOfWeight": round(float(t["fz"].max()) / weight, 2),
    "tilt": (round(float(t["tilt"].min()), 1), round(float(t["tilt"].max()), 1)),
  }


def fly(sc: Scene, hand: Hand, err: float = 0.0, rate: float | None = None,
        top: float | None = None) -> dict:
  """Take hold, open to `top` along the path (`err` off the true arc), hold,
  close again, let go: the summary and the rows."""
  kind = sc.mech.kind
  rate = rate or RATE[kind]
  top = OPEN_TO[kind] if top is None else top
  took = hand.take()
  sc.record()
  reached = hand.follow(0.0, top, rate, err)
  hand.hold(0.5)
  back = reached and hand.follow(top, CLOSE_TO[kind], rate, err)
  hand.hold(0.3)
  rows = sc.table()
  out = {"took": took, "reached": bool(reached and back), **summary(rows, sc.mech, sc.claw_kg)}
  out["closedTo"] = round(sc.opening(), 3)
  hand.let_go()
  hand.hold(0.5)
  out["seatedAfter"] = sc.seated()
  out["onFork"] = sc.on_fork()
  return out | {"rows": rows}


# ---- the walk-in's spread ------------------------------------------------------------

#: The claw's walk-in to a cube from LOOK_M back, the start jittered this
#: much across and in heading (the look point a walk leaves it at).
WALKIN_JITTER = (0.10, math.radians(8.0))


def walkin_one(k: int) -> dict:
  """One walk-in from a jittered start: where the cube is from the lying
  robot (true heading frame, m) and its heading against the line it came in
  on (deg)."""
  from pluggybot.legs import claw as lc
  from pluggybot.rack.tags import MASS_TAG_IDS
  sc = Scene(None)
  m, d, mis = sc.model, sc.data, sc.mis
  tag = MASS_TAG_IDS[0]
  cube = m.body("mass_known").id
  adr = m.jnt_qposadr[m.body_jntadr[cube]]
  at = np.array([START[0] + 1.6, START[1]])
  d.qpos[adr:adr + 3] = [at[0], at[1], BLOCK_HALF]
  d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
  mujoco.mj_forward(m, d)
  rng = np.random.default_rng(469 + k)
  dy, dh = (rng.uniform(-1, 1, 2) * WALKIN_JITTER) if k else (0.0, 0.0)
  sx, sy = at[0] - lc.LOOK_M, at[1] + dy
  sc.mount(sx, sy, dh)
  lines = []
  stance = mis._stance

  def watched(tag_, toward):
    out = stance(tag_, toward)
    x, y, _ = mis.true_pose()
    lines.append(math.atan2(at[1] - y, at[0] - x))
    return out
  mis._stance = watched
  rec: dict = {}
  got = sc.body.run(mis._approach_routine(tag, rec, (), float(d.time) + 150.0, None))
  x, y, yaw = mis.true_pose()
  ch, sh = math.cos(yaw), math.sin(yaw)
  rx, ry = at[0] - x, at[1] - y
  return {"k": k, "ok": got is not None, "why": rec.get("why"), "tries": len(rec.get("tries", [])),
          "ahead": round(ch * rx + sh * ry, 4), "across": round(-sh * rx + ch * ry, 4),
          "yaw": round(math.degrees(dk._wrap(yaw - lines[-1])), 2) if lines else None,
          "s": round(float(d.time), 1)}


#: MEASURED (`--walkin --n 16`): where the claw's walk-in left a cube from
#: the lying robot, the 14 that saw it from their start (two starts turned
#: away never did) -- ahead less `claw.LIE_AT_M` and across (m), and the
#: heading against the line walked in on (deg); each lined up on its first
#: try. The table sets each try's box out at one, re-centred on `LIE_AT_M`.
WALKIN = ((-0.0085, 0.0077, -0.50), (0.0044, 0.0167, -0.87), (0.0042, 0.0071, -0.10),
          (-0.0087, 0.0047, -0.42), (-0.0057, 0.0066, -0.67), (-0.0064, 0.0072, -0.50),
          (-0.0041, 0.0030, -0.01), (-0.0009, 0.0128, -1.34), (-0.0098, 0.0005, 0.04),
          (-0.0035, 0.0060, -0.86), (0.0020, 0.0123, -1.38), (-0.0112, -0.0021, 0.20),
          (-0.0025, 0.0070, -0.46), (-0.0044, 0.0063, -0.62))


def stances(n: int) -> list[tuple[float, float, float]]:
  """`n` set-outs at the walk-in's spread: square on, then each measured
  stance in turn."""
  return [(0.0, 0.0, 0.0)] + [WALKIN[k % len(WALKIN)] for k in range(n - 1)]


# ---- the force window ----------------------------------------------------------------

#: The pushes, heading frame, at the jaws' middle -- and `lift`, up through
#: the claw's centre of mass: the weight's own window.
WINDOW_DIRS = {"up": (0, 0, 1), "lift": (0, 0, 1), "down": (0, 0, -1),
               "toward": (-1, 0, 0), "away": (1, 0, 0), "side": (0, 1, 0)}
#: A push is ramped this fast, N/s, to at most WINDOW_MAX_N; it TIPS the claw
#: once it has swung this far on its peg, and UNSEATS it once the coupling's
#: criterion has been open this long -- longer than a contact's flicker.
WINDOW_RATE, WINDOW_MAX_N = 0.25, 6.0
TIP_DEG, UNSEAT_S = 10.0, 0.04


def window_one(direction: str) -> dict:
  """A push on the claw held at a working pose (the jaws `LIE_AT_M` ahead,
  a centimetre under the torso's centre), ramped until it unseats the claw
  or reaches `WINDOW_MAX_N`: where it tipped, where it unseated, and the
  swing along the way."""
  sc = Scene(None)
  sc.lie()
  m, d, mis = sc.model, sc.data, sc.mis
  hand = Hand(sc)
  hand.to(np.array([LIE_AT_M, 0.0, -0.01]))
  hand.hold(1.0)
  u = np.array(WINDOW_DIRS[direction], dtype=float)
  _, yaw = sc.heading()
  ch, sh = math.cos(yaw), math.sin(yaw)
  uw = np.array([ch * u[0] - sh * u[1], sh * u[0] + ch * u[1], u[2]])
  tilt0 = sc.claw_tilt()
  out = {"dir": direction, "tipN": None, "unseatN": None, "swing": [], "tilt0": round(tilt0, 1)}
  open_s, f, logged = 0.0, 0.0, 0.0
  while f < WINDOW_MAX_N:
    f += WINDOW_RATE * m.opt.timestep
    force = f * uw
    # at the jaws' middle -- or the lift's, the module's and its parts' CoM
    at = d.subtree_com[sc.claw] if direction == "lift" else d.site_xpos[sc.grip_site]
    d.xfrc_applied[sc.claw, :3] = force
    d.xfrc_applied[sc.claw, 3:] = np.cross(at - d.xipos[sc.claw], force)
    sc.body.run(mis._twist_routine(0.0, 0.0, 0.0))
    open_s = 0.0 if sc.seated() else open_s + m.opt.timestep
    swing = sc.claw_tilt() - tilt0
    if out["tipN"] is None and abs(swing) >= TIP_DEG:
      out["tipN"] = round(f, 2)
    if f >= logged + 0.5:
      logged += 0.5
      out["swing"].append((logged, round(swing, 1)))
    if open_s >= UNSEAT_S:
      out["unseatN"] = round(f, 2)
      break
  d.xfrc_applied[sc.claw] = 0.0
  hand.hold(1.0)
  out["onFork"] = sc.on_fork()
  out["seatedAfter"] = sc.seated()
  return out


# ---- the tables ------------------------------------------------------------------------

def table_one(args) -> dict:
  kind, k, stance, kp = args
  mech = Mech.nominal(kind)
  sc = Scene(mech)
  sc.lie()
  # the walk-in's yaw is the robot's heading less the line it came in on:
  # a box square to that line is turned the other way, seen from the robot
  sc.place(jaws_at(kind, 0.0), LIE_AT_M + stance[0], stance[1], -math.radians(stance[2]))
  hand = Hand(sc, kp)
  hand.hold(0.3)
  res = fly(sc, hand)
  res.pop("rows")
  return {"kind": kind, "k": k, "kp": kp, "stance": stance, **res}


def tolerance_one(args) -> dict:
  kind, err, kp, k, stance = args
  mech = Mech.nominal(kind)
  sc = Scene(mech)
  sc.lie()
  sc.place(jaws_at(kind, 0.0), LIE_AT_M + stance[0], stance[1], -math.radians(stance[2]))
  hand = Hand(sc, kp)
  res = fly(sc, hand, err=err)
  res.pop("rows")
  return {"kind": kind, "err": err, "kp": kp, "k": k, **res}


# ---- the torques: the force at the tool, and the arm's own friction ---------------------

def empty_sweep(sc: Scene, hand: Hand, rate: float, top: float | None = None) -> dict:
  """The jaws along the candidate's path, up and back, holding nothing, shut
  as they would be on the knob: the rows. The box is lifted out of the way,
  which a robot cannot do -- but the sim's friction is a constant, so a
  robot would calibrate its arm once, anywhere (two paths read the forearm
  0.02 N*m apart: its passive pivots turn their own ways)."""
  kind = sc.mech.kind
  top = OPEN_TO[kind] if top is None else top
  sc.look()
  sc.away()
  hand.run(hand.hand.jaws_routine(closed=True))
  hand.to(sc.path(0.0))
  hand.hold(0.3)
  sc.record()
  hand.follow(0.0, top, rate)
  hand.hold(0.5)
  hand.follow(top, CLOSE_TO[kind], rate)
  rows = sc.table()
  hand.to(sc.path(0.0) + [-0.04, 0.0, 0.06])
  sc.away(False)
  hand.hold(0.3)
  return rows


def beyond_hold(rows: dict) -> tuple[np.ndarray, np.ndarray]:
  """Each row's readings beyond the arm's own hold, and its coordinates'
  rates: what `legs.arm.arm_friction` and `less_friction` read."""
  return (np.column_stack([rows["rs"] - rows["hs"], rows["re"] - rows["he"]]),
          np.column_stack([rows["vs"], rows["vf"]]))


def arm_friction(rows: dict) -> np.ndarray:
  """Each motor's Coulomb friction off an empty sweep's rows."""
  return am.arm_friction(*beyond_hold(rows))


def force_off_torques(rows: dict, friction=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
  """The force at the tool, N, torso (x, z), off each row's torque READINGS
  (`legs.arm.tool_force`): less the arm's own hold and, where a joint turns,
  its friction (`legs.arm.less_friction`). And a mask of the rows where
  both joints turned."""
  spec = am.ArmSpec()
  beyond, rates = beyond_hold(rows)
  tau = am.less_friction(beyond, rates, friction)
  f = np.array([am.tool_force(spec, a, b, c, e)
                for a, b, (c, e) in zip(rows["qs"], rows["qf"], tau)])
  turning = np.all(np.abs(rates) > am.FRICTION_BAND, axis=1)
  return f.reshape(-1, 2), turning


def binned(x: np.ndarray, n: int = 50) -> np.ndarray:
  """Means over `n` rows (0.1 s of readings)."""
  k = len(x) // n
  return x[:k * n].reshape(k, n, *x.shape[1:]).mean(axis=1)


def torques_one(args) -> dict:
  """The candidate's oracle path flown empty (the arm's friction) and on the
  mechanism, at gain `kp`: the force off the readings against the contact
  force."""
  kind, kp = args
  mech = Mech.nominal(kind)
  sc = Scene(mech)
  sc.lie()
  sc.place(jaws_at(kind, 0.0), LIE_AT_M)
  hand = Hand(sc, kp)
  empty = empty_sweep(sc, hand, RATE[kind])
  fric = arm_friction(empty)
  res = fly(sc, hand)
  rows = res.pop("rows")
  truth = np.column_stack([rows["tx"], rows["tz"]])
  out = {"kind": kind, "kp": kp, "friction": [round(v, 3) for v in fric]}
  for label, f in (("raw", (0.0, 0.0)), ("corrected", fric)):
    est, moving = force_off_torques(rows, f)
    err = est - truth
    for part, mask in (("all", np.ones(len(err), bool)), ("moving", moving),
                       ("deadband", ~moving)):
      if mask.sum() < 50:
        continue
      e, eb = err[mask], binned(err[mask])
      out[f"{label}/{part}"] = {
        "n": int(mask.sum()), "bias": [round(v, 3) for v in e.mean(0)],
        "sd": [round(v, 3) for v in e.std(0)],
        "rms0.1s": [round(v, 3) for v in np.sqrt((eb ** 2).mean(0))],
        "max0.1s": [round(v, 3) for v in np.abs(eb).max(0)]}
  out["truthRange"] = [[round(float(truth[:, i].min()), 2), round(float(truth[:, i].max()), 2)]
                       for i in (0, 1)]
  return out


# ---- the oracle fit ------------------------------------------------------------------

#: The two sweep rates (rad/s), the range swept (rad), and the rows the fit
#: reads: off the lid's stop, moving, and both arm joints outside their
#: dead bands.
FIT_RATES = (0.15, 0.45)
FIT_TOP = 1.2
FIT_FROM = 0.08
FIT_MOVING = 0.05
#: The hidden parameters a set-out draws (seeded): the lid's board, a hidden
#: weight and where it sits, a spring helping it open (slack past the range,
#: so it eases the lift), its damping and its friction.
FIT_RANGES = {"lid_kg": (0.15, 0.45), "lump_kg": (0.0, 0.15), "lump_at": (0.2, 0.9),
              "stiffness": (0.0, 0.15), "damping": (0.0, 0.20), "friction": (0.0, 0.12)}
SPRING_SLACK = 2.5
G = 9.81
#: The fit's columns: the lid's gravity (cos and sin of its angle), the
#: spring (the angle and a constant), Coulomb friction, damping, inertia.
FIT_COLS = ("cos", "sin", "theta", "const", "coulomb", "damping", "inertia")
#: The models fitted: gravity alone (right where there is no spring), and
#: with its centre of mass's height off the hinge's line (sin); gravity and
#: a spring; and everything, the lid's inertia with it.
FIT_MODELS = {"gravity": [0, 4, 5], "gravity+height": [0, 1, 4, 5],
              "gravity+spring": [0, 2, 3, 4, 5], "full": [0, 1, 2, 3, 4, 5, 6]}


#: A DROP HANDLE ONLY PULLS: the claw hangs on its peg and swings off any
#: push away from the robot (`--window`), so a lid must close harder than
#: its spring, friction and damping hold it open, at every angle swept and
#: the faster rate -- by this much, N*m at the hinge (0.2 N at the knob). A
#: set-out whose spring won near the top pushed the handle back at the claw
#: on the fast sweep down: it swung 72 deg on its peg and rubbed the lid.
CLOSING_MARGIN_NM = 0.05


def closing_torque(m: Mech, th: np.ndarray, rate: float) -> np.ndarray:
  """What shuts the lid at `th` going down at `rate`, N*m: its gravity
  (the board, the hidden weight and the bracket, out along the lid and up
  off the hinge's line -- the height alone is up to 0.03 N*m at 69 deg)
  less its spring, friction and damping: the handle's tension times its
  lever."""
  out = (m.lid_kg * BOX_D / 2 + m.lump_kg * BOX_D * m.lump_at
         + 0.005 * (BOX_D + PIN_AHEAD / 2 - 0.003))
  up = (m.lid_kg + 0.005) * LID_T / 2 + m.lump_kg * m.lump_z
  return (G * (out * np.cos(th) - up * np.sin(th)) - m.stiffness * (m.springref - th)
          - m.friction - m.damping * rate)


def draw(k: int) -> Mech:
  """Set-out `k`'s hidden parameters (seeded): drawn from `FIT_RANGES` until
  the lid shuts harder than `CLOSING_MARGIN_NM` all the way up."""
  rng = np.random.default_rng(4690 + k)
  th = np.linspace(0.0, FIT_TOP, 25)
  while True:
    kw = {n: float(rng.uniform(*r)) for n, r in FIT_RANGES.items()}
    if k % 2 == 0:
      kw["stiffness"] = 0.0           # every other set-out has no spring
    m = Mech(kind="handle", springref=SPRING_SLACK, **kw)
    if closing_torque(m, th, max(FIT_RATES)).min() >= CLOSING_MARGIN_NM:
      return m


def lid_truth(sc: Scene) -> dict:
  """The fit's columns as the world has them: off the lid body's mass and
  centre of mass (in its own frame, about the hinge), its joint's stiffness,
  damping and friction, and its inertia about the hinge's axis."""
  m = sc.model
  lid = m.body(f"{sc.name}_lid_body").id
  mass = float(m.body_mass[lid])
  com = m.body_ipos[lid]
  j = m.joint(f"{sc.name}_hinge").id
  k = float(m.jnt_stiffness[j])
  # the inertia about the hinge's axis (y): the body's, rotated into its
  # frame, plus its mass at the CoM's distance from the axis, plus armature
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, m.body_iquat[lid])
  rot = rot.reshape(3, 3)
  iy = float((rot @ np.diag(m.body_inertia[lid]) @ rot.T)[1, 1])
  inertia = iy + mass * (com[0] ** 2 + com[2] ** 2) + float(m.dof_armature[m.jnt_dofadr[j]])
  return {"cos": -G * mass * com[0], "sin": -G * mass * com[2], "theta": k,
          "const": -k * float(m.qpos_spring[m.jnt_qposadr[j]]),
          "coulomb": float(m.dof_frictionloss[m.jnt_dofadr[j]]),
          "damping": float(m.dof_damping[m.jnt_dofadr[j]]), "inertia": inertia,
          "massKg": mass, "leverM": float(-com[0])}


def hinge_torque(sc: Scene, rows: dict, force: np.ndarray) -> np.ndarray:
  """What the handle puts on the lid about its hinge, N*m, off the force on
  the claw (torso x, z): the oracle's -- the lid's true angle places the
  pin; the handle's own weight rides the claw, so the pin carries it less
  the force on the claw."""
  level = sc.level
  head = (level @ np.column_stack([force[:, 0], np.zeros(len(force)), force[:, 1]]).T).T
  _, yaw = sc.heading()
  rot = sc.placed[1]
  # heading -> world -> box (the fit's box is set square, the algebra general)
  ch, sh = math.cos(yaw), math.sin(yaw)
  world = np.column_stack([ch * head[:, 0] - sh * head[:, 1], sh * head[:, 0] + ch * head[:, 1],
                           head[:, 2]])
  box = world @ rot
  w_h = np.array([0.0, 0.0, -G * sum(HANDLE_KG)])
  on_lid = w_h - box
  th = rows["open"]
  rx = PIN[0] * np.cos(th) + PIN[2] * np.sin(th)
  rz = -PIN[0] * np.sin(th) + PIN[2] * np.cos(th)
  return rz * on_lid[:, 0] - rx * on_lid[:, 2]


def design(th, thd, thdd) -> np.ndarray:
  return np.column_stack([np.cos(th), np.sin(th), th, np.ones_like(th), np.sign(thd), thd, thdd])


def fit_one(args) -> dict:
  """One set-out `mech`, named `name`, at gain `kp`: the arm's friction
  off an empty sweep, then the lid swept up and down at both rates; the
  oracle's fit of each column, the truth, the fit's own correlations, the
  claw's swing and slip, and the hinge's torque binned by angle (`curve`)."""
  name, mech, kp = args
  sc = Scene(mech)
  sc.lie()
  sc.place(jaws_at("handle", 0.0), LIE_AT_M)
  hand = Hand(sc, kp)
  fric = arm_friction(empty_sweep(sc, hand, FIT_RATES[0], FIT_TOP))
  took = hand.take()
  sc.record()
  ok, s = True, 0.0
  for rate in FIT_RATES:
    ok = ok and hand.follow(s, FIT_TOP, rate)
    hand.hold(0.4)
    ok = ok and hand.follow(FIT_TOP, CLOSE_TO["handle"], rate)
    hand.hold(0.4)
    s = CLOSE_TO["handle"]
  rows = sc.table()
  hand.let_go()
  truth = lid_truth(sc)
  force, both = force_off_torques(rows, fric)
  tau = hinge_torque(sc, rows, force)
  tau_true = hinge_torque(sc, rows, np.column_stack([rows["tx"], rows["tz"]]))
  th = rows["open"]
  dt = float(sc.model.opt.timestep)
  thd = np.gradient(th, dt)
  # the lid's rate off its angle, smoothed over 20 ms: its acceleration too
  ker = np.ones(10) / 10
  thd = np.convolve(thd, ker, mode="same")
  thdd = np.convolve(np.gradient(thd, dt), ker, mode="same")
  use = (th > FIT_FROM) & (np.abs(thd) > FIT_MOVING) & both
  a = design(th[use], thd[use], thdd[use])
  out = {"name": name, "kp": kp, "took": took, "reached": ok, "n": int(use.sum()),
         "friction": [round(v, 3) for v in fric], "truth": truth, "fits": {},
         "tiltRange": [round(float(rows["tilt"].min()), 1), round(float(rows["tilt"].max()), 1)],
         "slipMm": round(float(rows["slip"].max()), 1),
         "seatedPct": round(100.0 * float(rows["seated"].mean()), 1),
         "minMargin": round(float(closing_torque(mech, np.linspace(0, FIT_TOP, 25),
                                                 max(FIT_RATES)).min()), 3),
         "curve": curve(th, thd, tau, tau_true, use)}
  # the static torque the lid's gravity and spring make, at three angles: what
  # a fit's columns sum to, against the world's
  probe = np.array([0.2, 0.6, 1.0])

  def static(c: dict) -> np.ndarray:
    return (c.get("cos", 0.0) * np.cos(probe) + c.get("sin", 0.0) * np.sin(probe)
            + c.get("theta", 0.0) * probe + c.get("const", 0.0))
  for label, y in (("torques", tau[use]), ("contact", tau_true[use])):
    for model, cols in FIT_MODELS.items():
      x, *_ = np.linalg.lstsq(a[:, cols], y, rcond=None)
      r = y - a[:, cols] @ x
      s2 = float(r @ r) / max(1, len(y) - len(cols))
      cov = s2 * np.linalg.inv(a[:, cols].T @ a[:, cols])
      sd = np.sqrt(np.diag(cov))
      est = {FIT_COLS[c]: float(v) for c, v in zip(cols, x)}
      out["fits"][f"{label}/{model}"] = {
        "est": est, "se": {FIT_COLS[c]: float(v) for c, v in zip(cols, sd)},
        "corr": [[round(float(v), 3) for v in row] for row in cov / np.outer(sd, sd)],
        "rms": float(np.sqrt(s2)), "cols": [FIT_COLS[c] for c in cols],
        "curveErr": [round(float(v), 4) for v in static(est) - static(truth)]}
  return out


#: The hinge's torque binned by the lid's angle (rad), each way at each rate.
CURVE_BIN = 0.1


def curve(th, thd, tau, tau_true, use) -> list:
  """The hinge's torque, off the torques and off the contact force, binned
  by angle (`CURVE_BIN`), up and down, slow and fast: [rate, way, angle, n,
  torques, contact, the torques' scatter in the bin]."""
  out = []
  fast = np.abs(thd) > (FIT_RATES[0] + FIT_RATES[1]) / 2
  for r, rate in enumerate((~fast, fast)):
    for way in (1, -1):
      for lo in np.arange(FIT_FROM, FIT_TOP, CURVE_BIN):
        sel = use & rate & (np.sign(thd) == way) & (th >= lo) & (th < lo + CURVE_BIN)
        if sel.sum() >= 20:
          out.append([r, way, round(float(lo + CURVE_BIN / 2), 2), int(sel.sum()),
                      float(tau[sel].mean()), float(tau_true[sel].mean()),
                      float(tau[sel].std() / math.sqrt(sel.sum()))])
  return out


#: Set-outs at the edges of the drawn ranges, the drawn rule kept -- and
#: `open`, its premise: a spring that beats the lid's weight near the top.
CORNERS = {
  "heavy": Mech(kind="handle", lid_kg=0.45, lump_kg=0.15, lump_at=0.9, friction=0.12,
                damping=0.18),
  "light": Mech(kind="handle", lid_kg=0.15, friction=0.0, damping=0.0),
  "springy": Mech(kind="handle", lid_kg=0.45, lump_kg=0.10, lump_at=0.6, friction=0.02,
                  damping=0.02, stiffness=0.09, springref=SPRING_SLACK),
  "open": Mech(kind="handle", lid_kg=0.30, stiffness=0.15, springref=SPRING_SLACK),
}
#: Two lids with one first moment of mass and different masses: the board
#: alone (0.300 kg), and half the board with a weight out at its front edge
#: and up at its top face (0.229 kg, its inertia about the hinge 21 % more).
TWINS = {
  "twinA": Mech(kind="handle", lid_kg=0.30, friction=0.03, damping=0.03),
  "twinB": Mech(kind="handle", lid_kg=0.15, lump_kg=0.0789, lump_at=0.95, lump_z=0.0114,
                friction=0.03, damping=0.03),
}


#: The latch's pull: along the path to here (rad, or m), slowly -- a
#: compliant arm builds its pull only as its command runs ahead of a shut
#: lid: at Kp 15 a 4 N catch held until the command was 0.84 rad open.
LATCH_TOP = {"handle": 1.0, "lip": 1.0, "drawer": 0.10}
LATCH_RATE = {"handle": 0.15, "lip": 0.15, "drawer": 0.015}


def latch_one(args) -> dict:
  """A catch of `latch_n` N at the knob (the lip): the mechanism pulled from
  shut along its path, slowly. Over the whole pull: the most force on the
  claw, true and off the torques; how far the knob slid along the pads; the
  claw's widest swing off where it took hold; and where the mechanism ended
  against the command's top. At the release: the command, and the force in
  the half second before it. After it: the most the claw was pushed up, its
  swing, and the coupling."""
  kind, latch_n, kp = args
  mech = replace(Mech.nominal(kind), latch_n=latch_n)
  sc = Scene(mech)
  sc.lie()
  sc.place(jaws_at(kind, 0.0), LIE_AT_M)
  hand = Hand(sc)
  top, rate = LATCH_TOP[kind], LATCH_RATE[kind]
  fric = arm_friction(empty_sweep(sc, hand, rate, top))
  hand = Hand(sc, kp)
  took = hand.take()
  sc.record()
  ok = hand.follow(0.0, top, rate)
  hand.hold(1.0)
  rows = sc.table()
  est, _ = force_off_torques(rows, fric)
  mag_t = np.hypot(rows["tx"], rows["tz"])
  mag_e = np.hypot(est[:, 0], est[:, 1])
  tilt = rows["tilt"] - rows["tilt"][0]
  out = {"kind": kind, "latchN": latch_n, "kp": kp, "took": took, "reached": ok,
         "peakN": round(float(mag_t.max()), 2), "peakReadN": round(float(binned(mag_e, 10).max()), 2),
         "slipMm": round(float(rows["slip"].max()), 1) if kind != "lip" else None,
         "swingMax": round(float(tilt[np.argmax(np.abs(tilt))]), 1),
         "ended": round(float(rows["open"][-1]), 3), "top": top}
  opened = rows["open"] > (0.01 if kind != "drawer" else 0.005)
  if not opened.any():
    return out | {"released": False}
  i = int(np.argmax(opened))
  before = slice(max(0, i - 250), i + 1)
  after = slice(i, len(rows["t"]))
  out_run = longest = 0.0
  for seated in rows["seated"][after]:
    out_run = 0.0 if seated else out_run + 0.002
    longest = max(longest, out_run)
  return out | {
    "released": True, "atCmd": round(float(rows["cmd"][i]), 3),
    "trueN": round(float(mag_t[before].max()), 2),
    "readN": round(float(binned(mag_e[before], 10).max()), 2),
    "upAfterN": round(float(rows["fz"][after].max()), 2),
    "swingAfter": round(float(np.ptp(rows["tilt"][after])), 1),
    "openMs": round(1000 * longest), "gripped": sc.gripped() if kind != "lip" else None,
    "onFork": sc.on_fork(), "seated": sc.seated()}


# ---- the filmstrip ---------------------------------------------------------------------

def filmstrip(out: str, kinds=("handle", "lip", "drawer")) -> None:
  """Each candidate taken hold of, half open and open, from beside the box."""
  from PIL import Image, ImageDraw
  tiles = []
  for kind in kinds:
    sc = Scene(Mech.nominal(kind))
    sc.lie()
    sc.place(jaws_at(kind, 0.0), LIE_AT_M)
    hand = Hand(sc)
    hand.hold(0.3)
    hand.take()
    r = mujoco.Renderer(sc.model, 360, 480)
    c, yaw = sc.heading()
    look = c + np.array([math.cos(yaw), math.sin(yaw), 0.0]) * (LIE_AT_M + 0.02)
    look[2] = 0.20

    def shoot(label):
      # in profile from the robot's right (MuJoCo's azimuth 0 looks along
      # +x): the lid's arc and the claw on its peg in the picture's plane
      cam = mujoco.MjvCamera()
      cam.lookat[:] = look
      cam.distance, cam.azimuth, cam.elevation = 0.95, math.degrees(yaw) + 90.0, -8.0
      r.update_scene(sc.data, cam)
      img = Image.fromarray(r.render())
      ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
      tiles.append(img)
    shoot(f"{kind}: taken hold of")
    top = OPEN_TO[kind]
    hand.follow(0.0, top / 2, RATE[kind])
    hand.hold(0.3)
    shoot(f"{kind}: half open ({sc.opening():.2f})")
    hand.follow(top / 2, top, RATE[kind])
    hand.hold(0.3)
    shoot(f"{kind}: open ({sc.opening():.2f}), the claw seated {sc.seated()}")
    r.close()
  cols = 3
  rows = (len(tiles) + cols - 1) // cols
  strip = Image.new("RGB", (480 * cols, 360 * rows), (255, 255, 255))
  for k, t in enumerate(tiles):
    strip.paste(t, (480 * (k % cols), 360 * (k // cols)))
  strip.save(out)
  print(f"wrote {out}")


def _print_rows(rows, keys) -> None:
  for r in rows:
    print("  " + "  ".join(f"{k}={r.get(k)}" for k in keys))


def run_pool(fn, jobs, n_jobs):
  """`fn` over `jobs`, each in a process of its own: a world dropped in a
  reused one kept ~40 MB until the cycle collector ran, and 300 flights
  through six reused workers took the dev box into swap (2026-10-06)."""
  with Pool(max(1, n_jobs), maxtasksperchild=1) as pool:
    return pool.map(fn, jobs, chunksize=1)


#: The batch behind #469's report, mode by mode: its file in DIR, its flags.
ALL = (("walkin", ["--walkin", "--n", "16"]), ("window", ["--window"]),
       ("table", ["--table", "--n", "15"]),
       ("tolerance_handle", ["--tolerance", "--kinds", "handle", "--sets", "5"]),
       ("tolerance_others", ["--tolerance", "--kinds", "lip,drawer", "--sets", "3"]),
       ("torques", ["--torques", "--kp", "60,15,8"]),
       ("fit", ["--fit", "--n", "8", "--corners", "--twins", "--kp", "60,15,8"]),
       ("latch", ["--latch"]))


def batch(args, argv) -> None:
  """`--all`: every mode of `ALL` into DIR/<mode>.txt, then the filmstrip;
  or with `--into` alone, the one mode asked for into its file."""
  import contextlib
  import time
  into = Path(args.into)
  into.mkdir(parents=True, exist_ok=True)
  if args.all:
    modes = [(name, flags + ["--jobs", str(args.jobs)]) for name, flags in ALL]
    modes.append(("filmstrip", []))
  else:
    # `--into` goes in either spelling, or `main` hands the mode back here
    flags = [a for i, a in enumerate(argv) if not a.startswith("--into")
             and (i == 0 or argv[i - 1] != "--into")]
    modes = [(next((m for m in MODES if f"--{m}" in flags), "filmstrip"), flags)]
  for name, flags in modes:
    t0 = time.time()
    print(f"{name}: {' '.join(flags)}", file=sys.stderr, flush=True)
    if name == "filmstrip":
      main(flags + ["--out", str(into / "mechanism_spike.png")])
    else:
      with open(into / f"{name}.txt", "w") as f, contextlib.redirect_stdout(f):
        main(flags)
    print(f"{name}: {time.time() - t0:.0f} s", file=sys.stderr, flush=True)


#: The modes, as their flags name them.
MODES = ("walkin", "window", "table", "tolerance", "torques", "fit", "latch")


def main(argv=None) -> None:
  import json
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  for mode in MODES:
    ap.add_argument(f"--{mode}", action="store_true")
  ap.add_argument("--kinds", default="handle,lip,drawer",
                  help="the candidates flown, comma-separated (not --walkin, --window, --fit)")
  ap.add_argument("--n", type=int, default=15,
                  help="--walkin's starts, --table's set-outs, --fit's seeded set-outs")
  ap.add_argument("--sets", type=int, default=1,
                  help="--tolerance's set-outs a cell, square and then the walk-in's")
  ap.add_argument("--kp", default=f"{am.ARM_KP:g}",
                  help="--torques' and --fit's gains, comma-separated")
  ap.add_argument("--corners", action="store_true",
                  help="with --fit: the drawn ranges' corners, and the open-biased premise")
  ap.add_argument("--twins", action="store_true",
                  help="with --fit: two lids of one first moment and different masses")
  ap.add_argument("--jobs", type=int, default=3, help="processes in parallel")
  ap.add_argument("--out", default="mechanism_spike.png", help="the filmstrip's file")
  ap.add_argument("--all", action="store_true", help="every table and the filmstrip (with --into)")
  ap.add_argument("--into", default=None,
                  help="write a mode's rows to DIR/<mode>.txt rather than the terminal")
  args = ap.parse_args(argv)
  if args.all and not args.into:
    ap.error("--all writes a file a mode: give it --into DIR")
  if args.all or args.into:
    return batch(args, argv if argv is not None else sys.argv[1:])
  kinds = args.kinds.split(",")
  gains = [float(v) for v in args.kp.split(",")]
  if args.walkin:
    rows = run_pool(walkin_one, range(args.n), args.jobs)
    _print_rows(rows, ("k", "ok", "why", "tries", "ahead", "across", "yaw", "s"))
    ok = [r for r in rows if r["ok"]]
    for key in ("ahead", "across", "yaw"):
      v = np.array([r[key] for r in ok])
      print(f"{key}: mean {v.mean():+.4f} sd {v.std():.4f} min {v.min():+.4f} max {v.max():+.4f}")
    return
  if args.window:
    rows = run_pool(window_one, list(WINDOW_DIRS), args.jobs)
    _print_rows(rows, ("dir", "tilt0", "tipN", "unseatN", "onFork", "seatedAfter", "swing"))
    return
  if args.table:
    jobs = [(kind, k, st, kp) for kind in kinds for kp in (am.ARM_KP, 15.0)
            for k, st in enumerate(stances(args.n))]
    rows = run_pool(table_one, jobs, args.jobs)
    _print_rows(rows, ("kind", "kp", "k", "stance", "took", "reached", "opened", "opens",
                       "closedTo", "seatedPct", "openMs", "onFork", "gripPct", "slipMm", "up",
                       "down", "away", "toward", "side", "tilt"))
    return
  if args.tolerance:
    errs = (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)
    jobs = [(kind, e, kp, k, st) for kind in kinds for kp in GAINS for e in errs
            for k, st in enumerate(stances(args.sets))]
    rows = run_pool(tolerance_one, jobs, args.jobs)
    _print_rows(rows, ("kind", "kp", "err", "k", "took", "reached", "opened", "opens",
                       "seatedPct", "openMs", "onFork", "gripPct", "slipMm", "up", "down", "away",
                       "toward", "side", "tilt"))
    return
  if args.torques:
    for r in run_pool(torques_one, [(kind, kp) for kind in kinds for kp in gains], args.jobs):
      print(json.dumps(r))
    return
  if args.fit:
    named = {f"k{k}": draw(k) for k in range(args.n)}
    if args.corners:
      named.update(CORNERS)
    if args.twins:
      named.update(TWINS)
    jobs = [(name, mech, kp) for kp in gains for name, mech in named.items()]
    for r in run_pool(fit_one, jobs, args.jobs):
      print(json.dumps(r))
    return
  if args.latch:
    jobs = [(kind, n, kp) for kind in kinds for kp in (am.ARM_KP, 15.0)
            for n in (1.0, 2.0, 3.0, 4.0)]
    rows = run_pool(latch_one, jobs, args.jobs)
    _print_rows(rows, ("kind", "kp", "latchN", "took", "reached", "released", "atCmd",
                       "trueN", "readN", "peakN", "peakReadN", "slipMm", "swingMax", "ended",
                       "upAfterN", "swingAfter", "openMs", "gripped", "onFork", "seated"))
    return
  filmstrip(args.out, kinds)


if __name__ == "__main__":
  main()
