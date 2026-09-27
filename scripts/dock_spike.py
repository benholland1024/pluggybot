"""The quadruped's dock spike (issue #378): a cradle the robot lies down onto.

The dock is `pluggybot.legs.dock` (the cradle, its board of tags, the
electrical criterion and the walk-in's arithmetic); the body is #377's
(`legs.model.CHOSEN`), walking on its committed flat policy, reckoning with
its legs (`legs.odometry`), and lying down and standing up with the scripted
lower-and-fold #377 measured (`quad_spike.lie_down_routine`). Every table is
SimNotes, "The quadruped's dock".

  --capture     the funnel's capture envelope: the robot PLACED standing at
                an offset over the dock, then lying down; the verdict is the
                electrical criterion. Premises: `--sticky` (the faces at the
                belly case's own friction) and `--flat` (no funnel at all).
  --approach    the charge success rate from the approaches the robot will
                make: standing at the standoff, TRULY off by an offset while
                it believes it stands exactly there, it looks at the board,
                walks in, stops, checks, lies down, and retries. A grid, then
                `--n` random starts. Premise: `--blind` (one look at the
                start of each run, then its legs alone).
  --hold        docked: the contact through a minute lying there, the
                preload on each pad, the pose the board gives it (the
                anchor), and standing up and backing off the dock
  --mouth M, --bed B   fly any table on a dock of another width (m)
  --view        dockings in the MuJoCo viewer, one after another, each from a
                new random start at the standoff; close the window to quit
  (default)     a filmstrip of one docking, dock_spike.png

Usage:
  MUJOCO_GL=egl uv run python scripts/dock_spike.py [--capture|--approach|--hold]
  uv run python scripts/dock_spike.py --view     # a window: MUJOCO_GL left off
"""

import os

# One BLAS thread before numpy loads (quad_spike.py: the policy's products
# are small enough that six threads doubled a step).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from dataclasses import replace  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quad_spike as qs  # noqa: E402

from pluggybot.legs import dock as dk  # noqa: E402
from pluggybot.legs.actuator import JointLimits  # noqa: E402
from pluggybot.legs.model import CHOSEN, LEGS  # noqa: E402
from pluggybot.legs.odometry import LegOdometry  # noqa: E402
from pluggybot.legs.policy import PolicyDriver, Twist, WalkingPolicy  # noqa: E402
from pluggybot.legs.scripted import Command, VirtualModel  # noqa: E402
from pluggybot.rack.tags import DOCK_TAG_SIZE, TagDetector  # noqa: E402

#: How long the robot lies before the verdict is read, s.
SETTLE_S = 1.0
LOOK_EVERY_S, SEARCH_STEPS = dk.LOOK_EVERY_S, dk.SEARCH_STEPS
#: The random starts: TRUE offsets from the standoff, uniform in +-these.
START_ACROSS_M, START_ALONG_M, START_YAW_DEG = 0.3, 0.3, 30.0


def world(spec: dk.DockSpec = dk.DEFAULT):
  """The chosen body standing at the origin, and a dock whose frame is the
  world's: the seat at (0, 0), the board ahead along +x."""
  model = mujoco.MjModel.from_xml_string(*dk.world_xml(spec))
  return model, mujoco.MjData(model)


def _yaw(q) -> float:
  w, x, y, z = q
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ViewerClosed(Exception):
  """The viewer's window was closed: the flight stops where it is."""


def handover(drv: PolicyDriver) -> None:
  """The policy takes the body back from a routine that is not its own (the
  scripted stand-up): its last action is none, and it decides at once."""
  drv.last_action[:] = 0.0
  drv.steps = 0


class Rig:
  """One robot, one dock, and everything the approach reads and drives."""

  def __init__(self, spec: dk.DockSpec = dk.DEFAULT, policy=None, seed: int = 0,
               look: bool = True):
    self.spec = spec
    self.model, self.data = world(spec)
    m = self.model
    # Standing, and forwarded, before the scripted gait is built: it reads
    # the body's inertia off the mass matrix once (all zeros on a fresh
    # MjData, which left its attitude loop without its feed-forward).
    mujoco.mj_resetDataKeyframe(m, self.data, 0)
    mujoco.mj_forward(m, self.data)
    self.lim = JointLimits.of(CHOSEN.motor)
    self.drv = PolicyDriver(m, self.data, policy or WalkingPolicy())
    self.vm = VirtualModel(m, self.data, CHOSEN)
    self.root = m.body("pluggybot").id
    self.det = TagDetector(m, "nav_eye", tag_size=DOCK_TAG_SIZE) if look else None
    self.seed = seed
    self.odo = None
    #: The dock's pose in the ODOMETRY frame, off the last look.
    self.dock = None
    self.looks = 0
    self._dock = np.array([i for i in range(m.ngeom)
                           if m.geom(i).name.startswith("dock_")])
    self._feet = np.array([m.geom(f"{leg}_foot").id for leg in LEGS])
    self._body = np.array([i for i in range(m.ngeom)
                           if m.body_rootid[m.geom_bodyid[i]] == self.root
                           and i not in self._feet])
    #: Sim seconds a foot, and anything else of the robot, touched the dock
    #: while it was on its feet (lying on it is the point).
    self.feet_on_dock_s = self.body_on_dock_s = 0.0
    self.on_feet = True
    #: What the legs and their drivers drew, J (the electronics are added by
    #: the reader: they draw whatever the legs do).
    self.leg_j = 0.0
    #: A passive viewer to hand every tenth step to, at real time (`--view`),
    #: and (wall, sim) time when its pacing last began.
    self.viewer = None
    self._pace = None

  def close(self) -> None:
    if self.det is not None:
      self.det.close()

  # ---- the world ------------------------------------------------------------

  def place(self, x: float, y: float, yaw: float, *, belief=None) -> None:
    """Stand the robot at a TRUE pose; its legs' reckoning starts from
    `belief` (default: the truth), which is how a standoff arrived at with
    a drifted estimate looks to the robot."""
    d = self.data
    mujoco.mj_resetDataKeyframe(self.model, d, 0)
    d.qpos[0], d.qpos[1] = x, y
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(self.model, d)
    self.odo = LegOdometry(self.model, d, seed=self.seed)
    if belief is not None:
      self.odo.x, self.odo.y, self.odo.yaw = belief
    self.dock = None
    self.hold(0.5)

  def truth(self) -> tuple[float, float, float]:
    """Where the robot IS, in the dock's frame (the world's, here)."""
    d = self.data
    return float(d.qpos[0]), float(d.qpos[1]), _yaw(d.qpos[3:7])

  def charging(self) -> bool:
    return dk.dock_charge_contact(self.model, self.data)

  def _on_dock(self, mine: np.ndarray) -> bool:
    g = self.data.contact.geom[:self.data.ncon]
    if not len(g):
      return False
    return bool(np.any((np.isin(g[:, 0], self._dock) & np.isin(g[:, 1], mine))
                       | (np.isin(g[:, 1], self._dock) & np.isin(g[:, 0], mine))))

  # ---- stepping ---------------------------------------------------------------

  def _after(self) -> None:
    d, dt = self.data, self.model.opt.timestep
    self.odo.step()
    tau, qd = d.actuator_force[self.drv.act], d.qvel[6:18]
    self.leg_j += float(self.lim.copper_w(tau).sum()
                        + np.clip(tau * qd, 0, None).sum()) * dt
    if self.on_feet:
      if self._on_dock(self._feet):
        self.feet_on_dock_s += dt
      if self._on_dock(self._body):
        self.body_on_dock_s += dt
    if self.viewer is not None:
      self._show()

  def _show(self) -> None:
    """Hand the frame to the viewer, holding the sim to real time."""
    d = self.data
    if not self.viewer.is_running():
      raise ViewerClosed
    if round(d.time / self.model.opt.timestep) % 10:
      return
    self.viewer.sync()
    now = time.time()
    if self._pace is None or d.time < self._pace[1]:   # `place` restarts time
      self._pace = (now, d.time)
    ahead = (d.time - self._pace[1]) - (now - self._pace[0])
    if ahead > 0:
      time.sleep(ahead)

  def step(self, twist: Twist) -> None:
    self.drv.step(twist)
    self._after()

  def torque_step(self, tau) -> None:
    """A routine's torque, through the drivers' envelope (`legs.drivers`)."""
    self.drv.drivers.torque(tau)
    mujoco.mj_step(self.model, self.data)
    self._after()

  def hold(self, seconds: float) -> None:
    t0 = self.data.time
    while self.data.time - t0 < seconds:
      self.step(Twist())

  def wh(self, since_j: float = 0.0, seconds: float = 0.0) -> float:
    """The legs since a reading of `leg_j`, plus the electronics over
    `seconds`."""
    return ((self.leg_j - since_j) + sum(qs.ELECTRONICS_W.values()) * seconds) / 3600

  # ---- senses -----------------------------------------------------------------

  def pose(self) -> tuple[float, float, float]:
    """Where the robot BELIEVES it is (its legs' reckoning)."""
    return self.odo.x, self.odo.y, self.odo.yaw

  def look(self) -> dk.DockFix | None:
    """One decode of the board; on a fit, the dock's believed pose moves to
    it, in the odometry frame the next steps act in."""
    self.looks += 1
    seen = dk.seen_from(self.model, self.data, self.det.detect(self.data),
                        "nav_eye", self.root)
    fix = dk.fit_dock(seen, self.spec)
    if fix is not None:
      self.dock = dk.blend(self.dock, dk.compose(self.pose(), (fix.x, fix.y, fix.yaw)))
    return fix

  def err(self) -> tuple[float, float, float]:
    """Where the robot believes it is in the dock's frame."""
    return dk.relative(self.pose(), self.dock)

  def preload(self) -> tuple[float, float]:
    """What each pole's pins press with, N: their springs' force at the
    travel they are held at. (A pin touches the pad AND the belly box the
    pad is flush with, and the solver splits the push between the two
    contacts; the part presses with its spring.)"""
    k, ref = dk.pole_spring()
    m, d = self.model, self.data
    return tuple(k * (ref - d.qpos[m.jnt_qposadr[m.joint(f"dock_pole_{lbl}").id]])
                 for lbl in ("l", "r"))

  # ---- lying down and standing up ---------------------------------------------

  def lie_down(self) -> None:
    self.on_feet = False
    for tau in qs.lie_down_routine(CHOSEN, self.data, self.vm):
      self.torque_step(tau)
    t0 = self.data.time
    while self.data.time - t0 < SETTLE_S:   # the drivers hold nothing
      self.torque_step(np.zeros(12))

  def stand_up(self) -> None:
    for tau in qs.stand_up_routine(CHOSEN, self.data, self.vm):
      self.torque_step(tau)
    self.vm.reset()
    for _ in range(int(0.3 / self.model.opt.timestep)):
      self.torque_step(self.vm.torque(Command()))
    handover(self.drv)
    self.on_feet = True


# ---- the capture envelope -----------------------------------------------------

def capture_one(spec: dk.DockSpec, dx: float, dy: float, dyaw_deg: float) -> dict:
  """Placed standing over the dock off by (dx, dy, dyaw), then lying down."""
  rig = Rig(spec, look=False)
  rig.place(dk.LIE_SHIFT_M + dx, dy, math.radians(dyaw_deg))
  rig.lie_down()
  x, y, th = rig.truth()
  return {"ok": rig.charging(), "x": x, "y": y, "yaw": math.degrees(th),
          "z": float(rig.data.qpos[2])}


def describe(spec: dk.DockSpec) -> str:
  return (f"mouth +-{spec.mouth_half * 1000:.0f} mm (outer +-{spec.outer_half * 1000:.0f}), "
          f"bed +-{spec.bed_half * 1000:.0f} mm, faces {math.degrees(spec.face_angle):.0f} deg "
          f"at mu {spec.cradle_mu}, {spec.wall_h * 1000:.1f} mm high")


def capture_table(spec: dk.DockSpec, label: str) -> None:
  print(f"{label}: {describe(spec)}")
  print(f"(placed standing {dk.LIE_SHIFT_M * 1000:.1f} mm ahead of the seat, off by:)")
  print(f"{'across mm':>9s} {'yaw deg':>7s} {'along mm':>8s} {'charging':>8s} "
        f"{'lies x/y mm':>12s} {'yaw deg':>7s} {'z mm':>6s}")
  cases = [(0.0, dy, dyaw) for dy in (0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05)
           for dyaw in (0, 5, 10, 15)]
  cases += [(dx, 0.0, 0) for dx in (-0.06, -0.03, 0.03, 0.06, 0.09)]
  for dx, dy, dyaw in cases:
    r = capture_one(spec, dx, dy, dyaw)
    print(f"{dy * 1000:9.0f} {dyaw:7d} {dx * 1000:8.0f} "
          f"{'yes' if r['ok'] else 'NO':>8s} {r['x'] * 1000:6.1f}/{r['y'] * 1000:5.1f} "
          f"{r['yaw']:7.1f} {r['z'] * 1000:6.1f}")


# ---- the approach -------------------------------------------------------------

LIE_ACROSS_M, LIE_YAW, LIE_ALONG_M = dk.LIE_ACROSS_M, dk.LIE_YAW, dk.LIE_ALONG_M


def walk_in(rig: Rig, source: str, budget_s: float = 20.0) -> str:
  """Walk from where it stands to the seat. `source` is what the robot
  steers by: "look" re-reads the board every LOOK_EVERY_S, "blind" steers
  by its legs alone after the first look, "truth" by the sim's own pose
  (the policy's floor, never an act)."""
  t0 = last = rig.data.time
  while rig.data.time - t0 < budget_s:
    ex, ey, eth = rig.truth() if source == "truth" else rig.err()
    tw = dk.walk_in_twist(ex, ey, eth)
    if tw == Twist():
      return "stopped"
    rig.step(tw)
    if source == "look" and rig.data.time - last >= LOOK_EVERY_S:
      rig.look()
      last = rig.data.time
  return "budget"


def back_out(rig: Rig, metres: float = 0.9, budget_s: float = 8.0) -> None:
  """Walk straight back the way it came."""
  x0, y0, _ = rig.pose()
  t0 = rig.data.time
  while (math.hypot(rig.odo.x - x0, rig.odo.y - y0) < metres
         and rig.data.time - t0 < budget_s):
    rig.step(Twist(vx=-dk.APPROACH_V))
  rig.hold(0.8)


def turn_by(rig: Rig, degrees: float, budget_s: float = 10.0) -> None:
  """Turn on the spot by `degrees` of the legs' own heading, then settle."""
  target = rig.odo.yaw + math.radians(degrees)
  t0 = rig.data.time
  while (abs(dk._wrap(target - rig.odo.yaw)) > math.radians(3.0)
         and rig.data.time - t0 < budget_s):
    rig.step(Twist(yaw_rate=math.copysign(dk.TURN_W, dk._wrap(target - rig.odo.yaw))))
  rig.hold(0.4)


def find_board(rig: Rig) -> bool:
  """Look; failing that, sweep the spot it stands on until the board
  decodes."""
  if rig.look() is not None:
    return True
  for step in SEARCH_STEPS:
    turn_by(rig, step)
    if rig.look() is not None:
      return True
  return False


def approach(rig: Rig, source: str = "look", tries: int = 3) -> dict:
  """From wherever it stands with the board in reach: look, walk in, stop,
  check, lie down, and take another run at it if the pads do not conduct.
  Returns what the last attempt saw."""
  out = {"ok": False, "tries": 0, "why": ""}
  t0, j0 = rig.data.time, rig.leg_j
  feet0, body0 = rig.feet_on_dock_s, rig.body_on_dock_s
  for attempt in range(tries):
    out["tries"] = attempt + 1
    if not find_board(rig):
      out["why"] = "no board"
      break
    why = walk_in(rig, source)
    rig.hold(0.8)
    if source == "look":
      rig.look()
    ex, ey, eth = rig.truth() if source == "truth" else rig.err()
    out.update(stopAt=rig.truth(), walk=why)
    if why != "stopped" or abs(ey) > LIE_ACROSS_M or abs(eth) > LIE_YAW \
        or abs(ex - dk.LIE_SHIFT_M) > LIE_ALONG_M:
      out["why"] = "not lined up"
      back_out(rig)
      continue
    rig.lie_down()
    out["liesAt"] = rig.truth()
    if rig.charging():
      out["ok"], out["why"] = True, ""
      break
    out["why"] = "no contact"
    rig.stand_up()
    back_out(rig)
  out.update(seconds=rig.data.time - t0, wh=rig.wh(j0, rig.data.time - t0),
             feet=rig.feet_on_dock_s - feet0, body=rig.body_on_dock_s - body0)
  return out


def _row(label: str, r: dict) -> str:
  nan3 = (math.nan,) * 3
  sx, sy, sth = r.get("stopAt", nan3)
  lx, ly, lth = r.get("liesAt", nan3)
  return (f"{label} {'yes' if r['ok'] else 'NO':>6s} {r['tries']:5d} "
          f"{sx * 1000:6.0f}/{sy * 1000:4.0f}/{math.degrees(sth):4.1f} "
          f"{ly * 1000:7.1f}/{math.degrees(lth):4.1f} {r['feet']:6.2f} {r['body']:6.2f} "
          f"{r['seconds']:6.1f} {r['wh'] * 1000:6.0f}  {r['why']}")


def approach_table(source: str, spec: dk.DockSpec = dk.DEFAULT, n: int = 0) -> None:
  """From the standoff, TRULY off by an offset while the robot believes it
  stands exactly there: a grid, then `n` random starts."""
  policy = WalkingPolicy()
  print(f"steering by {source}; {describe(spec)}")
  head = (f"{'docked':>6s} {'tries':>5s} {'stop x/y mm/deg':>16s} {'lies y mm/deg':>12s} "
          f"{'feet s':>6s} {'body s':>6s} {'sim s':>6s} {'mWh':>6s}")
  print(f"{'across m':>8s} {'yaw deg':>7s} {'along m':>7s} " + head)
  rows = []
  for across in (0.0, 0.1, 0.2, 0.3):
    for dyaw in (0.0, 15.0, 30.0):
      rig = Rig(spec, policy=policy)
      rig.place(-dk.STANDOFF_M, across, math.radians(dyaw),
                belief=(-dk.STANDOFF_M, 0.0, 0.0))
      r = approach(rig, source)
      rig.close()
      rows.append(r)
      print(_row(f"{across:8.2f} {dyaw:7.0f} {0.0:7.2f}", r))
  summary(f"grid ({source})", rows)
  if not n:
    return
  rng = np.random.default_rng(378)
  rand = []
  for k in range(n):
    across = rng.uniform(-START_ACROSS_M, START_ACROSS_M)
    along = rng.uniform(-START_ALONG_M, START_ALONG_M)
    dyaw = rng.uniform(-START_YAW_DEG, START_YAW_DEG)
    rig = Rig(spec, policy=policy, seed=k)
    rig.place(-dk.STANDOFF_M + along, across, math.radians(dyaw),
              belief=(-dk.STANDOFF_M, 0.0, 0.0))
    r = approach(rig, source)
    rig.close()
    rand.append(r)
    print(_row(f"{across:8.2f} {dyaw:7.0f} {along:7.2f}", r))
  summary(f"{n} random starts ({source})", rand)


def summary(label: str, rows: list[dict]) -> None:
  ok = [r for r in rows if r["ok"]]
  first = sum(r["ok"] and r["tries"] == 1 for r in rows)
  lies = np.array([r["liesAt"] for r in ok]) if ok else np.zeros((0, 3))
  print(f"{label}: docked {len(ok)} of {len(rows)} ({first} first try); "
        f"feet on the dock in {sum(r['feet'] > 0 for r in rows)} "
        f"(worst {max(r['feet'] for r in rows):.2f} s); "
        + (f"lies at y {lies[:, 1].min() * 1000:+.1f}..{lies[:, 1].max() * 1000:+.1f} mm, "
           f"yaw {np.degrees(lies[:, 2]).min():+.1f}..{np.degrees(lies[:, 2]).max():+.1f} deg, "
           f"x {lies[:, 0].min() * 1000:+.0f}..{lies[:, 0].max() * 1000:+.0f} mm; "
           f"median {np.median([r['seconds'] for r in ok]):.1f} s, "
           f"{np.median([r['wh'] for r in ok]) * 1000:.0f} mWh" if ok else ""))


# ---- docked -----------------------------------------------------------------

def hold_table(spec: dk.DockSpec = dk.DEFAULT, minutes: float = 1.0,
               trials: int = 5) -> None:
  """Dock from the standoff, lie a minute, read the board, stand up and
  back off: `trials` times, each from a different start."""
  policy = WalkingPolicy()
  print(f"docked: {describe(spec)}")
  print(f"{'trial':>5s} {'contact %':>9s} {'longest gap ms':>14s} {'preload N l/r':>14s} "
        f"{'anchor err mm/deg':>18s} {'stood':>5s} {'off the dock':>12s} {'feet s':>6s} "
        f"{'undock s':>8s} {'mWh':>5s}")
  rng = np.random.default_rng(3780)
  for k in range(trials):
    rig = Rig(spec, policy=policy, seed=k)
    across = rng.uniform(-0.2, 0.2)
    rig.place(-dk.STANDOFF_M, across, math.radians(rng.uniform(-20, 20)),
              belief=(-dk.STANDOFF_M, 0.0, 0.0))
    r = approach(rig, "look")
    if not r["ok"]:
      print(f"{k:5d}  did not dock: {r['why']}")
      rig.close()
      continue
    # Lying there, the drivers holding nothing: the criterion every step.
    steps = int(minutes * 60 / rig.model.opt.timestep)
    on, gap, worst = 0, 0, 0
    for _ in range(steps):
      rig.torque_step(np.zeros(12))
      if rig.charging():
        on += 1
        gap = 0
      else:
        gap += 1
        worst = max(worst, gap)
    pre = rig.preload()
    # The anchor: the robot's pose in the dock's frame, off the lower row.
    fix = rig.look()
    if fix is None:
      anchor = "no board"
    else:
      bx, by, bth = dk.relative((0.0, 0.0, 0.0), (fix.x, fix.y, fix.yaw))
      tx, ty, tth = rig.truth()
      anchor = (f"{math.hypot(bx - tx, by - ty) * 1000:5.1f}/"
                f"{math.degrees(dk._wrap(bth - tth)):+5.2f} ({fix.n})")
    # Off the dock again.
    t0, j0, feet0 = rig.data.time, rig.leg_j, rig.feet_on_dock_s
    rig.stand_up()
    stood = abs(rig.data.qpos[2] - CHOSEN.stand_height) < 0.03
    back_out(rig, 0.6)
    clear = rig.truth()[0] < -0.45
    print(f"{k:5d} {on / steps * 100:9.2f} {worst * rig.model.opt.timestep * 1000:14.0f} "
          f"{pre[0]:6.1f}/{pre[1]:6.1f} {anchor:>18s} {'yes' if stood else 'NO':>5s} "
          f"{'yes' if clear else 'NO':>12s} {rig.feet_on_dock_s - feet0:6.2f} "
          f"{rig.data.time - t0:8.1f} {rig.wh(j0, rig.data.time - t0) * 1000:5.0f}")
    rig.close()


# ---- the filmstrip ------------------------------------------------------------

def filmstrip(out: str, spec: dk.DockSpec = dk.DEFAULT) -> None:
  """One docking from 0.2 m off and 15 deg turned, and what the robot's nose
  camera sees standing to lie down and lying there."""
  from PIL import Image, ImageDraw
  rig = Rig(spec)
  rig.place(-dk.STANDOFF_M, 0.2, math.radians(15), belief=(-dk.STANDOFF_M, 0.0, 0.0))
  renderer = mujoco.Renderer(rig.model, 300, 400)
  tiles = []

  def shoot(label, lookat, azimuth, elevation, distance):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    renderer.update_scene(rig.data, cam)
    img = Image.fromarray(renderer.render())
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(img)

  def wide(label):
    shoot(label, (-0.45, 0.05, 0.1), 145.0, -28.0, 2.3)

  def close(label):
    x, y, _ = rig.truth()
    shoot(label, (x, y, 0.12), -35.0, -24.0, 1.3)

  def eye(label):
    """The nose camera's own frame, the decoded tags boxed."""
    dets = rig.det.detect(rig.data)
    img = Image.fromarray(rig.det.renderer.render()).resize((400, 225))
    draw = ImageDraw.Draw(img)
    k = 400 / rig.det.width
    for tag_id, det in dets.items():
      u, v = det["center"]
      draw.rectangle((u * k - 12, v * k - 12, u * k + 12, v * k + 12), outline=(255, 60, 60))
      draw.text((u * k + 14, v * k - 6), str(tag_id), fill=(255, 60, 60))
    tile = Image.new("RGB", (400, 300), (0, 0, 0))
    tile.paste(img, (0, 38))
    ImageDraw.Draw(tile).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(tile)

  wide("at the standoff: 0.2 m off, 15 deg turned")
  find_board(rig)
  t0, last, shot = rig.data.time, rig.data.time, 0
  while True:
    tw = dk.walk_in_twist(*rig.err())
    if tw == Twist():
      break
    rig.step(tw)
    if rig.data.time - last >= LOOK_EVERY_S:
      rig.look()
      last = rig.data.time
    if shot < 2 and rig.data.time - t0 > 1.5 * (shot + 1):
      shot += 1
      wide(f"walking in, steering by the board ({rig.data.time - t0:.1f} s)")
  rig.hold(0.8)
  rig.look()
  close("stopped where it lies down from")
  eye("its nose camera, standing: the upper pair")
  rig.on_feet = False
  for k, tau in enumerate(qs.lie_down_routine(CHOSEN, rig.data, rig.vm)):
    rig.torque_step(tau)
    if k == 900:
      close("lowering")
  for _ in range(int(SETTLE_S / rig.model.opt.timestep)):
    rig.torque_step(np.zeros(12))
  close(f"lying on the dock: charging {rig.charging()}")
  x, y, _ = rig.truth()
  # From behind, the hind legs hide the belly: drawn see-through for it.
  m = rig.model
  hind = [i for i in range(m.ngeom) if m.geom(i).name[:2] in ("HL", "HR")]
  alpha = m.geom_rgba[hind, 3].copy()
  m.geom_rgba[hind, 3] = 0.15
  shoot("from behind, hind legs see-through: the belly in the cradle",
        (x - 0.05, y, 0.04), 0.0, -6.0, 0.7)
  m.geom_rgba[hind, 3] = alpha
  eye("its nose camera, lying: the lower pair")
  renderer.close()
  rig.close()
  cols = 3
  rows = (len(tiles) + cols - 1) // cols
  strip = Image.new("RGB", (400 * cols, 300 * rows), (255, 255, 255))
  for k, tile in enumerate(tiles):
    strip.paste(tile, (400 * (k % cols), 300 * (k // cols)))
  strip.save(out)
  print(f"wrote {out}")


# ---- the viewer ---------------------------------------------------------------

def view(spec: dk.DockSpec = dk.DEFAULT, seed: int | None = None) -> None:
  """Dockings in the MuJoCo viewer, one after another, each from a new
  random start at the standoff: it looks, walks in steering by the board,
  lies down, lies there a moment, stands up and backs off. Close the window
  to quit."""
  from mujoco import viewer as mj_viewer
  rng = np.random.default_rng(seed)
  rig = Rig(spec)
  print("Each docking starts at the standoff, 1 m behind the seat, TRULY off by a "
        "random offset while the robot believes it stands exactly there. Close the "
        "window to quit.")
  with mj_viewer.launch_passive(rig.model, rig.data) as viewer:
    viewer.cam.lookat[:] = (-0.35, 0.0, 0.1)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 145.0, -25.0
    rig.viewer = viewer
    k = 0
    try:
      while viewer.is_running():
        k += 1
        across = rng.uniform(-START_ACROSS_M, START_ACROSS_M)
        along = rng.uniform(-START_ALONG_M, START_ALONG_M)
        yaw = rng.uniform(-START_YAW_DEG, START_YAW_DEG)
        print(f"docking {k}: {across:+.2f} m across, {along:+.2f} m along, "
              f"{yaw:+.0f} deg turned")
        rig.place(-dk.STANDOFF_M + along, across, math.radians(yaw),
                  belief=(-dk.STANDOFF_M, 0.0, 0.0))
        r = approach(rig, "look")
        tries = f"{r['tries']} {'try' if r['tries'] == 1 else 'tries'}"
        print(f"  {'charging' if r['ok'] else 'not docked (' + r['why'] + ')'} "
              f"after {tries}, {r['seconds']:.1f} s")
        if r["ok"]:
          t0 = rig.data.time
          while rig.data.time - t0 < 3.0:     # lying there, the drivers idle
            rig.torque_step(np.zeros(12))
          rig.stand_up()
          back_out(rig, 0.6)
    except ViewerClosed:
      pass
  rig.close()


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--capture", action="store_true")
  ap.add_argument("--sticky", action="store_true",
                  help="with --capture: the faces at the belly case's friction")
  ap.add_argument("--flat", action="store_true",
                  help="with --capture: no funnel, the bars alone")
  ap.add_argument("--approach", action="store_true")
  ap.add_argument("--blind", action="store_true",
                  help="with --approach: one look a run, then the legs' reckoning")
  ap.add_argument("--truth", action="store_true",
                  help="with --approach: steer by the sim's pose (the floor)")
  ap.add_argument("--n", type=int, default=0,
                  help="with --approach: this many random starts after the grid")
  ap.add_argument("--hold", action="store_true")
  ap.add_argument("--view", action="store_true",
                  help="dockings in the MuJoCo viewer until the window closes")
  ap.add_argument("--mouth", type=float, default=None)
  ap.add_argument("--bed", type=float, default=None)
  ap.add_argument("--out", default="dock_spike.png")
  args = ap.parse_args(argv)
  spec = dk.DEFAULT
  if args.mouth is not None:
    spec = replace(spec, mouth_half=args.mouth)
  if args.bed is not None:
    spec = replace(spec, bed_half=args.bed)
  if args.capture:
    if args.sticky:
      spec = replace(spec, cradle_mu=1.0)
    if args.flat:
      spec = replace(spec, wall_h=0.0005)
    capture_table(spec, "sticky faces" if args.sticky else "no funnel" if args.flat
                  else "the dock")
  elif args.approach:
    approach_table("blind" if args.blind else "truth" if args.truth else "look",
                   spec, args.n)
  elif args.hold:
    hold_table(spec)
  elif args.view:
    view(spec)
  else:
    filmstrip(args.out, spec)


if __name__ == "__main__":
  sys.exit(main())
