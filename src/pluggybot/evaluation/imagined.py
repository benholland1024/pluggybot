"""The imagination against the truth (issue #466): OUR grading, by code that
knows a chest's hidden parameters. The robot never sees any of it.

  oracle_record   a probe planned off the true hinge, as #469's sweeps were:
                  the claw onto the knob where it hangs, and the lid swept up
                  and down at two rates at Kp 8 -- commands only, no robot
                  flying (stage 2 flies the robot's own)
  truth_world     the world's own chest -- `chest_xml` and its magnet --
                  in a world of the robot's own, for the same rollout
  language_gap    one path through the truth and through the best-expressible
                  reference (`chest.reference_document`): how far apart they
                  behave, the language's own limit before any robot senses
                  anything
  mass_moments    what the arm's torques can see of a lid: its first and
                  second moments about its hinge, never its mass (#469)

`scripts/imagination_gap.py` measures it over drawn chests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from pluggybot.activity import chest as ch
from pluggybot.imagination.compile import Imagined, compile_scene, robot_spec
from pluggybot.imagination.record import Record, Start
from pluggybot.imagination.rollout import Readings, rollout, settled
from pluggybot.imagination.scene import parse
from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.legs.model import CHOSEN, lie_qpos
from pluggybot.tools import claw as cl

#: The probe's gain (#469's choice: a hinge guessed within 4 cm keeps the
#: claw seated), its damping kept at the working gains' ratio.
KP = 8.0
KD = am.ARM_KD * math.sqrt(KP / am.ARM_KP)
#: The sweeps: up to `TOP` and back to `BOTTOM` (rad), at each rate (rad/s),
#: held `TURN_S` at each end -- #469's `--fit`.
RATES = (0.15, 0.45)
TOP, BOTTOM, TURN_S = ch.SWEPT_TO, 0.05, 0.4
#: The knob, where it hangs, this far ahead of the torso's centre (#469's
#: `LIE_AT_M`: the lid's arc carries it 0.16 m further and 0.22 m up).
KNOB_AHEAD_M = 0.46
#: Over the knob before going down onto it, m; how long the claw hangs there
#: first (#469's `SWING_SETTLE_S`: straight down, its crossbar met the
#: handle's bracket); and the arm's time to arrive after a line, open loop.
OVER_M = 0.045
SWING_SETTLE_S = 1.0
ARRIVE_S = 0.5
JAWS_OPEN = rk.CLAW_JAW_OPEN - rk.CLAW_JAW_CLOSED


@dataclass(frozen=True)
class Setting:
  """Where the chest is: in front of the robot, which lies at the map's
  origin facing +x, turned `chest_yaw` (`place` sets it out square, its knob
  `KNOB_AHEAD_M` ahead)."""
  chest_x: float
  chest_y: float = 0.0
  chest_yaw: float = 0.0


def start() -> Start:
  """The robot lying at the map's origin with the claw on its fork, its arm
  at the carry pose: as the claw's walk-in leaves it."""
  qs, qe = am.CARRY_Q
  return Start(pose=(0.0, 0.0, 0.0), attitude=(0.0, 0.0), legs=tuple(lie_qpos(CHOSEN)),
               arm=(qs, qs + qe), carrying="module_claw")


def hold_record(n: int = 1) -> Record:
  """`n` rows holding the start: what a settle replays."""
  st = start()
  row = [st.arm[0], st.arm[1], KP, KD, 0.0, 0.0]
  return Record(start=st, dt=0.002, commands=np.tile(row, (n, 1)))


# ---- the world's own chest, in the robot's world ------------------------------------

def truth_world(lid: ch.Lid, setting: Setting) -> Imagined:
  """The world's chest -- its MJCF, its contact, its magnet -- beside the
  robot's own body, for the imagination's rollout: what the reference is
  measured against. Its joints are the reference's ids."""
  spec = robot_spec("module_claw")
  ch.attach_chest(spec, lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                  tags=False)
  model = spec.compile()
  chest = ch.Chest(model, None, lid)
  joints = {}
  for jid, name in (("hinge", "chest_hinge"), ("pin", "chest_pin")):
    j = model.joint(name).id
    joints[jid] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
  return Imagined(model=model, hooks=(chest.sense,), joints=joints, carrying="module_claw")


def place(lid: ch.Lid) -> tuple[Setting, float]:
  """The chest set out so its knob, where it hangs, is `KNOB_AHEAD_M` ahead
  of the settled robot's centre: (the setting, the handle's angle on its pin
  as it hangs there, rad)."""
  setting = Setting(chest_x=0.30)
  for _ in range(2):
    world = truth_world(lid, setting)
    d = settled(world, hold_record())
    root = world.model.body("pluggybot").id
    knob = d.geom_xpos[world.model.geom("chest_knob").id]
    setting = Setting(chest_x=setting.chest_x + float(d.xpos[root][0]) + KNOB_AHEAD_M
                      - float(knob[0]))
  world = truth_world(lid, setting)
  d = settled(world, hold_record())
  return setting, float(d.qpos[world.joints["pin"][0]])


def reference_world(lid: ch.Lid, setting: Setting, pin: float) -> Imagined:
  """The best-expressible reference, compiled: the chest as the language says
  it at its closest, the handle hanging as it hangs."""
  doc = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  return compile_scene(parse(doc), carrying="module_claw")


# ---- the oracle's probe ----------------------------------------------------------

class _Plan:
  """The arm's commands step by step, open loop, as the claw's routines
  would send them (`tools.claw.ClawHand`): the driver's slew on its target,
  the slide and the jaws ramped."""

  def __init__(self, st: Start, level: np.ndarray) -> None:
    self.level = level
    self.goal = np.array(st.arm, dtype=float)
    self.target = self.goal.copy()
    self.slide = self.jaws = 0.0
    self.rows: list[tuple] = []
    self.dt = 0.002
    self.spec = CHOSEN.arm
    self.rise = self.spec.fork.seat_rise()

  def step(self) -> None:
    lim = am.ARM_SLEW * self.dt
    self.target += np.clip(self.goal - self.target, -lim, lim)
    self.rows.append((*self.target, KP, KD, self.slide, self.jaws))

  def hold(self, seconds: float) -> None:
    for _ in range(round(seconds / self.dt)):
      self.step()

  def aim(self, vx: float, vz: float) -> None:
    g = self.goal
    q = am.solve_vertex(self.spec, vx, vz, near=(float(g[0]), float(g[1] - g[0])))
    if q is None:
      raise ValueError(f"the vertex ({vx:.3f}, {vz:.3f}) is out of the arm's reach")
    self.goal = np.array([q[0], q[0] + q[1]])

  def vertex_for(self, p) -> tuple[float, float, float]:
    """`ClawHand.vertex_for`: the fork's vertex and the slide for the jaws'
    middle at `p`, heading frame."""
    seat = self.level.T @ np.array([p[0], p[1], p[2] + rk.CLAW_JAW_DROP])
    return float(seat[0]), float(seat[2] - self.rise), -float(seat[1])

  def ramp(self, attr: str, value: float, speed: float, settle: float = 0.0) -> None:
    s = getattr(self, attr)
    n = max(1, int(abs(value - s) / speed / self.dt))
    for k in range(1, n + 1):
      setattr(self, attr, s + (value - s) * k / n)
      self.step()
    self.hold(settle)

  def to(self, p, speed: float = cl.WORK_V) -> None:
    """`ClawHand.to_routine`, open loop: the slide, then the vertex along a
    line, then `ARRIVE_S` to arrive."""
    vx, vz, slide = self.vertex_for(p)
    self.ramp("slide", slide, cl.SLIDE_SPEED)
    g = self.goal
    x0, z0 = am.wrist_xz(self.spec, float(g[0]), float(g[1] - g[0]))
    x0, z0 = x0 + self.spec.fork.vertex_x, z0 + self.spec.fork.vertex_z
    n = max(1, int(math.hypot(vx - x0, vz - z0) / speed / self.dt))
    for k in range(1, n + 1):
      self.aim(x0 + (vx - x0) * k / n, z0 + (vz - z0) * k / n)
      self.step()
    self.hold(ARRIVE_S)

  def follow(self, path, s0: float, s1: float, rate: float) -> None:
    """The jaws along `path(s)` (heading frame) from `s0` to `s1` at `rate`:
    the vertex aimed and the slide set every step (#469's `Hand.follow`)."""
    n = max(1, int(abs(s1 - s0) / rate / self.dt))
    for k in range(1, n + 1):
      vx, vz, slide = self.vertex_for(path(s0 + (s1 - s0) * k / n))
      if abs(slide) > rk.CLAW_TRAVEL:
        raise ValueError("the path runs past the slide's stroke")
      self.aim(vx, vz)
      self.slide = slide
      self.step()


def _rot_y(th: float) -> np.ndarray:
  c, s = math.cos(th), math.sin(th)
  return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def oracle_record(lid: ch.Lid, setting: Setting,
                  rates=RATES) -> tuple[Record, dict[str, slice]]:
  """The oracle's probe as a record of commands, and its phases' rows:
  `take` (onto the knob and shut), then each rate's sweep up and down
  (`up0`, `down0`, ...). Planned off the truth: the knob where it hangs in
  the settled world, and the true hinge."""
  world = truth_world(lid, setting)
  st = start()
  d = settled(world, hold_record())
  m = world.model
  root = m.body("pluggybot").id
  rot = d.xmat[root].reshape(3, 3)
  c = d.xpos[root].copy()
  yaw = math.atan2(rot[1, 0], rot[0, 0])
  cy, sy = math.cos(yaw), math.sin(yaw)
  heading = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
  level = heading @ rot
  knob = d.geom_xpos[m.geom("chest_knob").id].copy()
  ca, sa = math.cos(setting.chest_yaw), math.sin(setting.chest_yaw)
  box = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
  origin = np.array([setting.chest_x, setting.chest_y, 0.0])
  hinge = origin + box @ ch.HINGE
  pin = origin + box @ (ch.HINGE + ch.PIN)

  def path(s: float) -> np.ndarray:
    """The jaws' middle with the lid at `s`: the knob moves as the pin does
    (the claw holds the handle as it hangs)."""
    turned = hinge + box @ _rot_y(s) @ box.T @ (pin - hinge)
    return heading @ (turned - pin + knob - c)

  plan = _Plan(st, level)
  phases = {}
  k0 = 0
  plan.ramp("jaws", JAWS_OPEN, cl.JAW_SPEED, settle=0.3)
  plan.to(path(0.0) + [0.0, 0.0, OVER_M])
  plan.hold(SWING_SETTLE_S)
  plan.to(path(0.0), cl.WORK_V / 2)
  plan.ramp("jaws", 0.0, cl.JAW_SPEED, settle=cl.GRIP_SETTLE_S)
  phases["take"] = slice(k0, len(plan.rows))
  s = 0.0
  for i, rate in enumerate(rates):
    k0 = len(plan.rows)
    plan.follow(path, s, TOP, rate)
    plan.hold(TURN_S)
    phases[f"up{i}"] = slice(k0, len(plan.rows))
    k0 = len(plan.rows)
    plan.follow(path, TOP, BOTTOM, rate)
    plan.hold(TURN_S)
    phases[f"down{i}"] = slice(k0, len(plan.rows))
    s = BOTTOM
  return Record(start=st, dt=plan.dt, commands=np.array(plan.rows)), phases


# ---- the gap ---------------------------------------------------------------------

@dataclass(frozen=True)
class Gap:
  """Two worlds along one path, a phase at a time: the force at the tool
  their readings put apart (N: the RMS of 0.1 s means, and the largest), the
  lid's angle apart (deg: RMS, largest), and the lid's top in each (deg)."""
  phase: str
  force_rms: float
  force_max: float
  lid_rms: float
  lid_max: float
  top: tuple[float, float]


def _binned(x: np.ndarray, n: int = 50) -> np.ndarray:
  k = len(x) // n
  return x[:k * n].reshape(k, n, *x.shape[1:]).mean(axis=1)


def compare(truth: Readings, other: Readings, phases: dict[str, slice]) -> list[Gap]:
  """`other` against `truth`, phase by phase (`Gap`). The force is the
  readings' difference through the arm's Jacobian at the truth's pose
  (`legs.arm.tool_force`): what the two worlds put on the tool apart."""
  spec = CHOSEN.arm
  dtau = other.sensed[:, :2] - truth.sensed[:, :2]
  qs, qf = truth.column("shoulder"), truth.column("fore")
  force = np.array([am.tool_force(spec, a, b, c, e)
                    for a, b, (c, e) in zip(qs, qf, dtau)])
  lid_t, lid_o = np.degrees(truth.joints["hinge"]), np.degrees(other.joints["hinge"])
  out = []
  for name, rows in phases.items():
    f = _binned(force[rows])
    mag = np.hypot(f[:, 0], f[:, 1]) if len(f) else np.zeros(1)
    dl = lid_o[rows] - lid_t[rows]
    out.append(Gap(phase=name, force_rms=float(np.sqrt((mag ** 2).mean())),
                   force_max=float(mag.max()), lid_rms=float(np.sqrt((dl ** 2).mean())),
                   lid_max=float(np.abs(dl).max()),
                   top=(float(lid_t[rows].max()), float(lid_o[rows].max()))))
  return out


def language_gap(lid: ch.Lid, rates=RATES) -> dict:
  """One drawn chest's truth and its best-expressible reference along the
  oracle's probe: the gap phase by phase, and each world's readings."""
  setting, pin = place(lid)
  record, phases = oracle_record(lid, setting, rates)
  truth = rollout(truth_world(lid, setting), record)
  ref = rollout(reference_world(lid, setting, pin), record)
  return {"setting": setting, "pin": pin, "record": record, "phases": phases,
          "truth": truth, "reference": ref, "gap": compare(truth, ref, phases)}


def mass_moments(model: mujoco.MjModel, data: mujoco.MjData, body: int,
                 axis_point, axis) -> tuple[float, float]:
  """A subtree's first and second moments about an axis: (its gravity's
  torque about the axis over g, kg*m, + turning the axis's way; and sum m
  r^2 with each body's own inertia, kg*m^2), off a compiled world -- what
  the arm's torques can see of a lid (#469: never its mass)."""
  a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
  p0 = np.asarray(axis_point, dtype=float)
  first = second = 0.0
  for b in range(model.nbody):
    x = b
    while x not in (0, body):
      x = int(model.body_parentid[x])
    if x != body:
      continue
    m = float(model.body_mass[b])
    r = data.xipos[b] - p0
    perp = r - (r @ a) * a
    first += m * float(a @ np.cross(r, [0.0, 0.0, -1.0]))
    rot = data.ximat[b].reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[b]) @ rot.T
    second += float(a @ inertia @ a) + m * float(perp @ perp)
  return first, second
