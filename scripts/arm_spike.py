"""The quadruped's arm spike (issue #378): the arm, the coupling at its tip,
the rack it takes tools from, and what carrying a tool through a walk, a
trot, a flight of stairs and a fall does to the coupling.

The arm is `pluggybot.legs.arm` (two pitch joints, both motors at the
shoulder, a passive parallelogram keeping the end plate level) on #377's
body (`legs.model.CHOSEN`, the placeholder arm replaced), standing and
walking on its committed policies; the rack and the tools are
`pluggybot.legs.rack`. Every table is SimNotes, "The quadruped's arm, its
coupling and the rack".

  --reach       the level-tool choice: which world targets (the floor ahead
                of the feet, a tabletop, a whiteboard's rows from one stance,
                a rack bay, the carry pose) each option reaches with the
                plate level -- (a) the parallelogram, (c) the body's pitch
                and crouch -- and what the motors hold, (b) a wrist motor's
                cost at full reach
  --capture     the coupling's envelope: the robot PLACED standing at the
                bay, off by the row (across, yaw, the aim along and in
                height), a pick and the tool hung back. Premises: `--rover`
                (the rover's fork and trays), `--narrow` (the trays at
                +-35 mm); `--loose` flies the stops at the rover's 4 mm
  --approach    the rack's success table: from the standoff, TRULY off by up
                to +-0.3 m and +-30 deg while it believes it stands exactly
                there, the robot finds the rack, walks in steering by its
                tags, waits, measures the bay, takes the tool if lined up
                (else backs out and tries again) and hangs it back; a
                centred start then `--n` random ones
  --retention   carrying a tool at the carry pose through a walk, trots,
                turns, sidesteps and hard stops: how often the coupling's
                criterion opens, the longest, the swing, the arm's and the
                knees' torques. `--stairs`: up and down the house's flight
                on the seeing policy, from rest, `--n` starts a row, a tool
                on its peg's line, 30 mm and 60 mm ahead of it (`--first`:
                the fork as first built, its V's at 45 deg -- the premise).
                `--fall`: pushed over trotting, with and without the fold
                reflex, then the get-up policy
  --getup       the get-up policy from 20 random drops, the arm folded,
                against the placeholder it was trained with
  --sensors     what the stowed arm, and the arm carrying, hide from the
                LIDAR and the nose and depth cameras
  --envelope    a tool's mass and how far its CoM sits off its peg, carried
                through a trot and a stop: seated, and at what angle
  --view [SCENE]  a scene in the MuJoCo viewer, over and over, at real time,
                until the window closes: `fetch` (the default: walk in, take
                the tool, hold it up, hang it back), `carry` (walks, trots,
                stops and turns with a tool), `stairs` (over a hill of the
                house's flight with one), `fall` (pushed over; the arm folds,
                or every other time does not), `reach` (the arm through the
                targets that chose it)
  (default)     a filmstrip of one fetch, arm_spike.png

Usage:
  MUJOCO_GL=egl uv run python scripts/arm_spike.py [--reach|--capture|--approach|...]
  uv run python scripts/arm_spike.py --view [fetch|carry|stairs|fall|reach]
                                                # a window: MUJOCO_GL left off
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

from pluggybot.legs import arm as am  # noqa: E402
from pluggybot.legs import rack as rk  # noqa: E402
from pluggybot.legs.actuator import GIM4310_10, GIM8108_8  # noqa: E402
from pluggybot.legs.model import CHOSEN, body_xml  # noqa: E402
from pluggybot.legs.policy import PolicyDriver, Twist, WalkingPolicy  # noqa: E402
from pluggybot.legs.scripted import _quat_rpy  # noqa: E402
from pluggybot.rack import coupling as rover_coupling  # noqa: E402
from pluggybot.rack.coupling import PEG_ABOVE_BODY  # noqa: E402
from pluggybot.rack.tags import TAG_DIR, asset_xml  # noqa: E402
from pluggybot.telemetry.protocol import ROBOT_ROOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: The spike's tool: the rover's heaviest module (the claw, 211 g) as a
#: plate, a peg and its mass.
TOOL = "tool"
TOOL_KG = 0.211
#: The tool's CoM from its peg, in the level plate frame: the plate's
#: centre (`rack.coupling.PEG_ABOVE_BODY` under the peg).
TOOL_COM = (0.0, -0.022)
FORK_DROP, LIFT, STANDOFF, BACK_OUT, FORK_V = (am.FORK_DROP, am.LIFT, am.STANDOFF,
                                              am.BACK_OUT, am.FORK_V)
#: The bay the spike works; the robot stands for it with the peg WORK_X
#: ahead of the torso's centre, on its centreline.
BAY = 1
WORK_X = rk.WORK_X
#: The standing policy turns ~1.5 deg while it settles from the keyframe
#: (arm or no arm): the rack is laid against the settled robot.
SETTLE_S = 3.0


#: The PREMISES: `--rover`, the rover's coupling on the arm -- its fork's
#: 45 deg V-notches at +-58 mm, end-stops 4 mm past the ends of a 150 mm peg
#: with no ramps, the trays at +-40 mm, its own 22 mm drop and 36 mm lift
#: (`rack.coupling`); `--loose`, this
#: fork's ramps with the rover's 4 mm of play at the stops.
ROVER_FORK = am.ForkSpec(fork_y=0.058, flank_deg=45.0, v_half_len=0.011, stop_y=0.079,
                         ramp_w=0.0005, ramp_h=0.012)
ROVER_RACK = replace(rk.DEFAULT, tray_y=0.040, peg_half=0.075)
LOOSE_FORK = replace(am.ForkSpec(), stop_y=rk.PEG_HALF + 0.004)
#: ...and `--narrow`: the trays at the rover's room for the plate (+-35 mm,
#: 9 mm either side), the fork and the peg 10 mm in with them.
NARROW_RACK = replace(rk.DEFAULT, tray_y=0.035, peg_half=0.100)
NARROW_FORK = replace(am.ForkSpec(), fork_y=0.075, stop_y=0.1015)
#: ...and the stairs' (`--retention --stairs --first`): the fork as this
#: issue first built it, its V's the rover's (45 deg, 22 mm flanks) and its
#: end-ramps at 45 deg.
FIRST_FORK = replace(am.ForkSpec(), flank_deg=45.0, v_half_len=0.011, ramp_w=0.020)


class ViewerClosed(Exception):
  """The viewer's window was closed: the scene stops where it is."""


class Rig:
  """One robot with the arm, one rack, one tool on a bay."""

  def __init__(self, spec: am.ArmSpec = am.ArmSpec(), rack: rk.RackSpec | None = rk.DEFAULT,
               policy=None, tool_kg: float = TOOL_KG, rover: bool = False,
               scenery: str = "", tool_group: int = 0, lump: tuple | None = None,
               loose: bool = False):
    #: The verbs' drop under a hanging peg and lift: the rover's own with
    #: its coupling.
    self.drop, self.lift = FORK_DROP, LIFT
    if rover:
      spec, rack = spec.with_(fork=ROVER_FORK), ROVER_RACK
      self.drop, self.lift = rover_coupling.FORK_DROP, rover_coupling.LIFT_STEP
    if loose:
      spec = spec.with_(fork=LOOSE_FORK)
    self.spec, self.rack = spec, rack
    peg_half = rack.peg_half if rack else rk.PEG_HALF
    face = ""
    if lump is not None:
      kg, lx, lz = lump
      face = (f'<geom name="{TOOL}_lump" type="sphere" size="0.012" pos="{lx} 0 {lz}" '
              f'mass="{kg}" rgba="0.8 0.2 0.2 1"/>')
      tool_kg -= kg
    after = rk.tool_xml(TOOL, (1.0, 0.0, 0.5), yaw=math.pi, mass=tool_kg,
                        peg_half=peg_half, face=face)
    assets_xml = f"\n    {asset_xml(rk.RACK_TAG_IDS)}" if rack else ""
    if rack:
      after = rk.rack_xml(rack, pos=(WORK_X + 1.0, 0.0), yaw=math.pi) + after
    xml = body_xml(CHOSEN, arm=am.arm_mjcf(spec), assets=assets_xml,
                   scenery=scenery, after=after)
    xml = xml.replace("  <default>\n", "  <default>\n    "
                      + rk.tool_default(TOOL, tool_group) + "\n", 1)
    assets = {f"tags/tag{i}.png": (ROOT / TAG_DIR / f"tag{i}.png").read_bytes()
              for i in rk.RACK_TAG_IDS} if rack else {}
    self.model = mujoco.MjModel.from_xml_string(xml, assets)
    self.data = mujoco.MjData(self.model)
    m, d = self.model, self.data
    mujoco.mj_resetDataKeyframe(m, d, 0)
    self.tool_adr = m.jnt_qposadr[m.joint(f"{TOOL}_free").id]
    self.tool_dof = m.jnt_dofadr[m.joint(f"{TOOL}_free").id]
    self.tool_body = m.body(TOOL).id
    self.rack_body = m.body("rack").id if rack else None
    self.root = m.body(ROBOT_ROOT).id
    self._park_tool()
    mujoco.mj_forward(m, d)
    policy = policy if policy is not None else WalkingPolicy()
    if isinstance(policy, (str, Path)):
      policy = WalkingPolicy(policy)
    self.drv = PolicyDriver(m, d, policy)
    self.arm = am.ArmDriver(m, d, spec)
    self.seat = m.site("arm_seat").id
    self.tool_kg = tool_kg
    #: A passive viewer to hand every tenth step to, at real time (`--view`),
    #: and (wall, sim) time when its pacing last began.
    self.viewer = None
    self._pace = None

  # ---- the world ------------------------------------------------------------

  def _park_tool(self) -> None:
    """The tool's free joint from the XML's pose (a keyframe zeroes it)."""
    m, d = self.model, self.data
    d.qpos[self.tool_adr:self.tool_adr + 3] = m.body(TOOL).pos
    d.qpos[self.tool_adr + 3:self.tool_adr + 7] = m.body(TOOL).quat

  def lay_rack(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0) -> None:
    """The rack laid in front of the robot as it stands NOW: bay `BAY`'s
    peg WORK_X ahead of the torso's centre, then off by (dx along, dy
    across, dyaw) in the robot's heading frame; the tool hung on the bay."""
    m, d = self.model, self.data
    _, _, yaw = _quat_rpy(d.qpos[3:7])
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = WORK_X + dx, dy
    peg = d.qpos[:2] + np.array([c * bx - s * by, s * bx + c * by])
    ryaw = yaw + math.pi + dyaw
    # The rack's origin is under its MIDDLE bay: step back from BAY's peg.
    off = self.rack.bays[BAY]
    origin = peg - np.array([-math.sin(ryaw) * off, math.cos(ryaw) * off])
    m.body_pos[self.rack_body] = [origin[0], origin[1], 0.0]
    m.body_quat[self.rack_body] = [math.cos(ryaw / 2), 0, 0, math.sin(ryaw / 2)]
    tq = [math.cos(ryaw / 2), 0, 0, math.sin(ryaw / 2)]
    d.qpos[self.tool_adr:self.tool_adr + 3] = [peg[0], peg[1], self.rack.peg_z
                                                - PEG_ABOVE_BODY + 0.0003]
    d.qpos[self.tool_adr + 3:self.tool_adr + 7] = tq
    d.qvel[self.tool_dof:self.tool_dof + 6] = 0.0
    mujoco.mj_forward(m, d)

  def mount_tool(self) -> None:
    """The tool seated on the fork as the arm stands now: its peg in the
    V's, hanging plumb (the retention flights start carrying it)."""
    m, d = self.model, self.data
    v = d.site_xpos[self.seat]
    rot = d.xmat[self.root].reshape(3, 3)
    _, _, yaw = _quat_rpy(d.qpos[3:7])
    peg = v + rot @ np.array([0.0, 0.0, self.spec.fork.seat_rise() + 0.0003])
    d.qpos[self.tool_adr:self.tool_adr + 3] = peg - [0, 0, PEG_ABOVE_BODY]
    ty = yaw + math.pi
    d.qpos[self.tool_adr + 3:self.tool_adr + 7] = [math.cos(ty / 2), 0, 0, math.sin(ty / 2)]
    d.qvel[self.tool_dof:self.tool_dof + 6] = 0.0
    self.arm.payload = (self.tool_kg, TOOL_COM)
    mujoco.mj_forward(m, d)

  def body_frame(self, p) -> np.ndarray:
    """A world point in the torso's frame."""
    d = self.data
    return d.xmat[self.root].reshape(3, 3).T @ (np.asarray(p) - d.xpos[self.root])

  def peg(self) -> np.ndarray:
    """The tool's peg axis, world."""
    d = self.data
    return d.xpos[self.tool_body] + d.xmat[self.tool_body].reshape(3, 3) @ [0, 0, PEG_ABOVE_BODY]

  def powered(self) -> bool:
    return rk.tool_power(self.model, self.data, TOOL)["powered"]

  def on_bay(self) -> bool:
    return rk.on_bay(self.model, self.data, TOOL, self.rack, BAY)

  def tool_tilt(self) -> float:
    """The tool's lean off plumb, deg."""
    z = self.data.xmat[self.tool_body].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, z[2]))))

  # ---- stepping ---------------------------------------------------------------

  def step(self, twist: Twist = Twist()) -> None:
    self.drv.command(twist)
    self.arm.step()
    mujoco.mj_step(self.model, self.data)
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
    if self._pace is None or d.time < self._pace[1]:     # a reset restarts time
      self._pace = (now, d.time)
    ahead = (d.time - self._pace[1]) - (now - self._pace[0])
    if ahead > 0:
      time.sleep(ahead)

  def hold(self, seconds: float, twist: Twist = Twist()) -> None:
    t0 = self.data.time
    while self.data.time - t0 < seconds:
      self.step(twist)

  def vertex(self) -> tuple[float, float]:
    """The fork's V vertex in the torso frame's (x, z)."""
    p = self.body_frame(self.data.site_xpos[self.seat])
    return float(p[0]), float(p[2])

  def fork_to(self, x: float, z: float, speed: float = FORK_V) -> bool:
    """The fork's V vertex along a straight line to (x, z), torso frame, the
    plate level; False if a point on the way is out of the arm's reach."""
    x0, z0 = self.vertex()
    dt = self.model.opt.timestep
    n = max(1, int(math.hypot(x - x0, z - z0) / speed / dt))
    arm = self.arm
    for k in range(1, n + 1):
      xt, zt = x0 + (x - x0) * k / n, z0 + (z - z0) * k / n
      near = (arm.goal[0], arm.goal[1] - arm.goal[0])
      q = am.solve_vertex(self.spec, xt, zt, near=near)
      if q is None:
        return False
      arm.goal = np.array([q[0], q[0] + q[1]])
      self.step()
    t0 = self.data.time
    while not arm.arrived(0.002) and self.data.time - t0 < 1.0:
      self.step()
    self.hold(0.2)
    return True

  # ---- the verbs --------------------------------------------------------------

  def pick(self, aim) -> dict:
    """Take the tool whose peg the robot believes is at `aim` (torso frame
    x, z): in under it, lift, back out. The verdict is the criterion's."""
    px, pz = aim
    out = {"lifted": False, "picked": False, "tilt": self.tool_tilt()}
    drop, lift = self.drop, self.lift
    if not (self.fork_to(px - STANDOFF, pz - drop, speed=0.2)
            and self.fork_to(px, pz - drop)):
      return {**out, "why": "out of reach"}
    self.arm.payload = (self.tool_kg, TOOL_COM)
    reached = self.fork_to(px, pz - drop + lift)
    out["lifted"] = self.powered()
    reached = self.fork_to(px - BACK_OUT, pz - drop + lift) and reached
    self.hold(0.5)
    out["picked"] = reached and self.powered() and not self.on_bay()
    out["tilt"] = self.tool_tilt()
    return out

  def put_back(self, aim) -> dict:
    """Hang the carried tool back on the bay the robot believes is at `aim`:
    over it, down, out."""
    px, pz = aim
    drop, lift = self.drop, self.lift
    if not (self.fork_to(px, pz - drop + lift) and self.fork_to(px, pz - drop)):
      return {"returned": False, "why": "out of reach"}
    self.arm.payload = (0.0, (0.0, 0.0))
    reached = self.fork_to(px - BACK_OUT, pz - drop)
    self.hold(0.5)
    return {"returned": reached and self.on_bay() and not self.powered()}


def settled(spec=am.ArmSpec(), **kw) -> Rig:
  rig = Rig(spec, **kw)
  rig.hold(SETTLE_S)
  return rig


# ---- the capture envelope -------------------------------------------------------

def aim_of(rig: Rig, dx: float = 0.0, dz: float = 0.0) -> tuple[float, float]:
  """The TRUE peg's axis where it crosses the arm's plane, off by an aim
  error (dx, dz): the placed capture measures the coupling, not the looking
  (the approach measures that, off the tags: `Approach.aim`)."""
  d = rig.data
  rot = d.xmat[rig.tool_body].reshape(3, 3)
  p = rig.body_frame(rig.peg())
  axis = d.xmat[rig.root].reshape(3, 3).T @ rot[:, 1]
  # Along the peg's axis to the arm's plane (y = the arm's y).
  t = (rig.spec.y - p[1]) / axis[1] if abs(axis[1]) > 1e-9 else 0.0
  q = p + t * axis
  return float(q[0] + dx), float(q[2] + dz)


def capture_one(args) -> dict:
  across, yaw_deg, along, height, premise = args
  kw = {"rover": premise == "rover", "loose": premise == "loose"}
  if premise == "narrow":
    kw.update(spec=am.ArmSpec().with_(fork=NARROW_FORK), rack=NARROW_RACK)
  rig = settled(**kw)
  rig.lay_rack(dy=across, dyaw=math.radians(yaw_deg))
  aim = aim_of(rig, along, height)
  out = rig.pick(aim)
  if out["picked"]:
    # Returning, the robot aims where it measured the bay: it has not moved.
    out.update(rig.put_back(aim))
  else:
    out["returned"] = False
  out.update(across=across, yaw=yaw_deg, along=along, height=height)
  return out


CAPTURE_ROWS = (
  [(a, 0, 0, 0) for a in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
                          -0.010, -0.015, -0.020, -0.025)]
  + [(0.0, y, 0, 0) for y in (2, 4, 6, 8, 10, 12, -6, -10)]
  + [(0.0, 0, e, 0) for e in (-0.010, -0.005, 0.005, 0.010, 0.015)]
  + [(0.0, 0, 0, h) for h in (-0.010, -0.005, 0.005, 0.010)]
  + [(0.015, 4, 0, 0), (0.020, 4, 0, 0), (-0.015, -4, 0, 0), (0.010, 6, 0.005, 0.005)])


def capture_table(premise: str = "", jobs: int = 3) -> list[dict]:
  todo = [(a, y, e, h, premise) for a, y, e, h in CAPTURE_ROWS]
  if jobs > 1:
    from multiprocessing import Pool
    with Pool(jobs) as pool:
      rows = pool.map(capture_one, todo, chunksize=1)
  else:
    rows = [capture_one(t) for t in todo]
  label = {"rover": "the rover's fork and trays (premise)",
           "loose": "the stops' play at the rover's 4 mm",
           "narrow": "the trays at +-35 mm, the fork at +-75 (premise)"}.get(
             premise, "the arm's fork")
  print(f"{label}: tool {TOOL_KG * 1000:.0f} g; the robot placed at the bay, off by:")
  print(f"{'across mm':>9s} {'yaw deg':>7s} {'aim along mm':>12s} {'aim height mm':>13s} "
        f"{'picked':>6s} {'hung back':>9s} {'tilt on the fork deg':>20s}")
  for r in rows:
    print(f"{r['across'] * 1000:9.0f} {r['yaw']:7.0f} {r['along'] * 1000:12.0f} "
          f"{r['height'] * 1000:13.0f} {'yes' if r['picked'] else 'NO':>6s} "
          f"{'yes' if r['returned'] else 'NO':>9s} {r['tilt']:20.1f}")
  return rows


def filmstrip(out: str) -> None:
  """One tool fetch from the approach's standoff, 0.2 m off and 11 deg
  turned (it lines up on its first walk in): the walk in, what the nose
  camera reads at the bay, the pick, the carry, the tool hung back, and the
  arm stowed."""
  from PIL import Image, ImageDraw
  rig = Approach(seed=0)
  rig.hold(1.0)
  rig.place(-APPROACH_STANDOFF, -0.2, math.radians(11.0),
            belief=(-APPROACH_STANDOFF, 0.0, 0.0))
  r = mujoco.Renderer(rig.model, 300, 400)
  tiles = []

  def shoot(label, lookat, az, el, dist):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r.update_scene(rig.data, cam)
    img = Image.fromarray(r.render())
    ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 255))
    tiles.append(img)

  # MuJoCo's azimuth 0 looks along +x, the way the robot walks in: these
  # stand behind and beside it, never behind the rack's board.
  def wide(label):
    x, y, _ = rig.truth()
    shoot(label, (x / 2 - 0.1, y / 2, 0.3), 25.0, -25.0, 2.4)

  def close(label, az=60.0, el=-12.0, dist=0.9):
    x, y, _ = rig.truth()
    shoot(label, (x + 0.35, y, 0.45), az, el, dist)

  def eye(label):
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

  wide("at the standoff: 0.2 m off, 11 deg turned")
  rig.look()
  t0, last, shot = rig.data.time, rig.data.time, False
  while True:
    tw = rk.walk_in_twist(*rig.err())
    if tw == Twist():
      break
    rig.step(tw)
    if rig.data.time - last >= LOOK_EVERY_S:
      rig.look()
      last = rig.data.time
    if not shot and rig.data.time - t0 > 2.0:
      shot = True
      wide("walking in, steering by the rack's tags")
  rig.hold(rk.SETTLE_AFTER_WALK_S)
  a = rig.aim()
  eye(f"the nose camera at the bay: {a.across * 1000:+.0f} mm across, "
      f"{math.degrees(a.yaw):+.1f} deg")
  px, pz = a.x, a.z
  rig.fork_to(px - STANDOFF, pz - FORK_DROP, speed=0.2)
  rig.fork_to(px, pz - FORK_DROP)
  close("slid in under the peg")
  rig.arm.payload = (rig.tool_kg, TOOL_COM)
  rig.fork_to(px, pz - FORK_DROP + LIFT)
  close(f"lifted: the tool powered {rig.powered()}", az=85.0, el=-5.0, dist=0.6)
  rig.fork_to(px - BACK_OUT, pz - FORK_DROP + LIFT)
  rig.fork_to(*CARRY, speed=0.2)
  rig.hold(0.5)
  close("carried over the nose, above the scan plane", az=75.0, el=-5.0, dist=1.5)
  rig.fork_to(px - BACK_OUT, pz - FORK_DROP + LIFT, speed=0.2)
  res = rig.put_back((px, pz))
  close(f"hung back: on its bay {res['returned']}")
  rig.arm.aim(*rig.spec.stow)
  rig.hold(3.0)
  close("stowed: the arm hides nothing from the sensors", az=60.0, el=-18.0, dist=1.4)
  r.close()
  rig.close()
  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  strip = Image.new("RGB", (400 * cols, 300 * rows), (255, 255, 255))
  for k, t in enumerate(tiles):
    strip.paste(t, (400 * (k % cols), 300 * (k // cols)))
  strip.save(out)
  print(f"wrote {out}")


# ---- carrying: retention and outages -----------------------------------------------

CARRY = am.CARRY
#: The flat flights, in order, on the committed flat policy.
FLAT_FLIGHTS = (
  ("stand", 2.0, Twist()),
  ("walk 0.3 m/s", 5.0, Twist(vx=0.3)),
  ("trot 0.6 m/s", 5.0, Twist(vx=0.6)),
  ("trot 1.0 m/s", 5.0, Twist(vx=1.0)),
  ("stop from 1.0 m/s", 2.0, Twist()),
  ("turn 0.8 rad/s", 4.0, Twist(yaw_rate=0.8)),
  ("sidestep 0.3 m/s", 4.0, Twist(vy=0.3)),
  ("start to 1.0 m/s", 2.0, Twist(vx=1.0)),
  ("stop", 2.0, Twist()),
)


class Watch:
  """The coupling's criterion every physics step: how often it is open,
  the longest it stays open, the tool's swing off plumb, the plate's tilt."""

  def __init__(self, rig: Rig):
    self.rig = rig
    self.steps = self.off = self.gap = self.worst = 0
    self.swing = self.plate = 0.0
    #: The arm's two motors (|tau| samples), and the legs' knees.
    self.arm_tau: list = []
    self.knee: list = []

  def tick(self) -> None:
    r = self.rig
    self.steps += 1
    self.arm_tau.append(np.abs(r.data.ctrl[r.arm.act]))
    self.knee.append(np.abs(r.data.actuator_force[r.drv.act][2::3]))
    if r.powered():
      self.gap = 0
    else:
      self.off += 1
      self.gap += 1
      self.worst = max(self.worst, self.gap)
    self.swing = max(self.swing, r.tool_tilt())
    z = r.data.xmat[r.model.body("arm_plate").id].reshape(3, 3)[:, 2]
    self.plate = max(self.plate, math.degrees(math.acos(max(-1.0, min(1.0, z[2])))))

  def row(self, name: str) -> str:
    dt = self.rig.model.opt.timestep
    tau = np.array(self.arm_tau) if self.arm_tau else np.zeros((1, 2))
    knee = np.array(self.knee) if self.knee else np.zeros((1, 4))
    return (f"{name:22s} {self.off / max(self.steps, 1) * 100:8.2f} {self.worst * dt * 1000:9.0f} "
            f"{self.swing:9.1f} {self.plate:9.1f} {'yes' if self.rig.powered() else 'NO':>7s} "
            f"{tau.max(axis=0)[0]:5.1f}/{tau.max(axis=0)[1]:4.1f} "
            f"{np.sqrt((tau ** 2).mean(axis=0)).max():5.2f} "
            f"{np.percentile(knee, 99.5):6.1f}")


RETENTION_HEAD = (f"{'flight':22s} {'open %':>8s} {'longest ms':>9s} {'swing deg':>9s} "
                  f"{'plate deg':>9s} {'seated':>7s} {'arm peak s/e':>10s} {'RMS':>5s} "
                  f"{'knee p99.5':>6s}")


def carrying(policy=None, scenery: str = "", key: int = 0, z_lift: float = 0.0,
             tool_kg: float = TOOL_KG, ahead: float = 0.0,
             fork: am.ForkSpec | None = None) -> Rig:
  """A robot standing with the tool seated on its fork at the carry pose. A
  tool whose CoM sits `ahead` of its peg (+: away from the robot) carries
  the offset in a lump: the plate's own mass sits on the peg's axis, so the
  whole tool's CoM is `ahead` off when the lump is ahead * kg / lump out (a
  tool's +x faces the robot: ahead is its -x). `fork`: another fork's."""
  lump = None
  if ahead:
    lump_kg = tool_kg - rk.MODULE_MASS
    lump = (lump_kg, -ahead * tool_kg / lump_kg, -PEG_ABOVE_BODY)
  spec = am.ArmSpec() if fork is None else am.ArmSpec().with_(fork=fork)
  rig = Rig(spec, rack=None, policy=policy, scenery=scenery, tool_group=3, tool_kg=tool_kg,
            lump=lump)
  if z_lift:
    rig.data.qpos[2] += z_lift
    mujoco.mj_forward(rig.model, rig.data)
  rig.hold(1.0)
  rig.fork_to(*CARRY, speed=0.2)
  rig.mount_tool()
  if ahead:
    rig.arm.payload = (tool_kg, (ahead, -PEG_ABOVE_BODY))
  rig.hold(1.0)
  return rig


def retention_flat(tool_kg: float = TOOL_KG) -> None:
  rig = carrying(tool_kg=tool_kg)
  print(f"carried at {CARRY} (torso frame), a {tool_kg * 1000:.0f} g tool, the flat policy:")
  print(RETENTION_HEAD)
  for name, seconds, twist in FLAT_FLIGHTS:
    w = Watch(rig)
    t0 = rig.data.time
    while rig.data.time - t0 < seconds:
      rig.step(twist)
      w.tick()
    print(w.row(name))


#: The stairs' table: a tool whose CoM sits on its peg, 30 mm ahead of it
#: and at the envelope's limit, up and down the house's flight from rest,
#: STAIR_FLIGHTS starts each STAIR_BACK_M further back.
STAIR_AHEAD = (0.0, 0.030, am.TOOL_MAX_AHEAD_M)
STAIR_FLIGHTS = 20
STAIR_BACK_M = 0.07


def stair_one(args) -> dict:
  """One flight from rest, facing along it (the seeing policy's own case),
  the tool carried: did the robot arrive, and is the tool still on its fork?"""
  up, ahead, k, fork, rise, steps = args
  sys.path.insert(0, str(ROOT / "scripts"))
  import quad_spike as qs
  from pluggybot.legs.policy import POLICY_NPZ
  scen = qs.staircase_scenery(rise, steps, qs.CLIMB_EDGE_M, down=not up)
  rig = carrying(policy=POLICY_NPZ.with_name("quadruped_rough_seeing.npz"), scenery=scen,
                 z_lift=0.0 if up else steps * rise, ahead=ahead, fork=fork)
  d = rig.data
  # Back along the flight, the tool with it: a robot carrying one is never
  # moved without it.
  d.qpos[0] -= STAIR_BACK_M * k
  d.qpos[rig.tool_adr] -= STAIR_BACK_M * k
  mujoco.mj_forward(rig.model, d)
  w = Watch(rig)
  top_x = qs.CLIMB_EDGE_M + steps * qs.TREAD_M
  t0 = d.time
  arrived = False
  while d.time - t0 < 4.0 + 2.5 * steps and not arrived:
    rig.step(Twist(vx=0.4))
    w.tick()
    if up:
      arrived = d.qpos[0] > top_x + 0.3 and d.qpos[2] > steps * rise + 0.8 * CHOSEN.stand_height
    else:
      arrived = (d.qpos[0] > top_x - qs.TREAD_M + 0.3
                 and abs(d.qpos[2] - CHOSEN.stand_height) < 0.08)
  rig.hold(1.0)
  dt = rig.model.opt.timestep
  tau = np.array(w.arm_tau)
  return {"up": up, "ahead": ahead, "arrived": arrived, "seated": rig.powered(),
          "open_ms": w.worst * dt * 1000, "swing": w.swing, "plate": w.plate,
          "arm": float(tau.max()), "knee": float(np.percentile(np.array(w.knee), 99.5))}


def retention_stairs(rise: float = 0.18, steps: int = 10, n: int = STAIR_FLIGHTS,
                     jobs: int = 3, fork: am.ForkSpec | None = None) -> None:
  """The house's flight up and down on the seeing policy, the tool carried,
  at each lean. A flight that never arrived is the policy's (a descent
  sometimes stalls at the top edge), counted apart from the coupling's."""
  todo = [(up, a, k, fork, rise, steps) for up in (True, False) for a in STAIR_AHEAD
          for k in range(n)]
  if jobs > 1:
    from multiprocessing import Pool
    with Pool(jobs) as pool:
      rows = pool.map(stair_one, todo, chunksize=1)
  else:
    rows = [stair_one(t) for t in todo]
  f = fork or am.ForkSpec()
  ramp = math.degrees(math.atan2(f.ramp_h, f.ramp_w))
  print(f"a flight of {steps} x {rise} m risers on the seeing policy (ideal scan), from "
        f"rest, a {TOOL_KG * 1000:.0f} g tool carried at {CARRY} on {f.flank_deg:.0f} deg "
        f"V's and {ramp:.0f} deg end-ramps; {n} starts each:")
  print(f"{'flight':6s} {'ahead mm':>8s} {'arrived':>8s} {'kept':>5s} {'lost':>5s} "
        f"{'longest open ms':>15s} {'swing deg':>9s} {'plate deg':>9s} {'arm peak':>8s} "
        f"{'knee p99.5':>10s}")
  for up in (True, False):
    for a in STAIR_AHEAD:
      mine = [r for r in rows if r["up"] == up and r["ahead"] == a]
      got = [r for r in mine if r["arrived"]]
      kept = [r for r in got if r["seated"]]
      worst = max((r["open_ms"] for r in kept), default=0.0)
      print(f"{'up' if up else 'down':6s} {a * 1000:8.0f} {len(got):4d}/{len(mine):<3d} "
            f"{len(kept):5d} {len(got) - len(kept):5d} {worst:15.0f} "
            f"{max((r['swing'] for r in kept), default=0.0):9.1f} "
            f"{max((r['plate'] for r in got), default=0.0):9.1f} "
            f"{max((r['arm'] for r in got), default=0.0):8.1f} "
            f"{max((r['knee'] for r in got), default=0.0):10.1f}")


def fall(push_ns: float = 30.0, reflex: bool = True, getup_s: float = 6.0) -> dict:
  """Trotting with a tool, pushed over sideways: the coupling through the
  fall, then the get-up policy with the arm on the robot's back. `reflex`
  folds the arm when the torso tips past 60 deg."""
  from pluggybot.legs.policy import POLICY_NPZ
  rig = carrying()
  w = Watch(rig)
  t0 = rig.data.time
  while rig.data.time - t0 < 2.0:
    rig.step(Twist(vx=0.6))
  d = rig.data
  d.xfrc_applied[rig.root] = [0.0, push_ns / 0.1, 0.0, 0.0, 0.0, 0.0]
  t0 = d.time
  while d.time - t0 < 0.1:
    rig.step(Twist(vx=0.6))
    w.tick()
  d.xfrc_applied[rig.root] = 0.0
  folded = False
  while d.time - t0 < 2.0:
    if reflex and not folded and rig.data.xmat[rig.root].reshape(3, 3)[2, 2] < am.FOLD_ON_FALL_COS:
      rig.arm.payload = (0.0, (0.0, 0.0))
      rig.arm.aim(*rig.spec.stow)
      folded = True
    rig.step(Twist())
    w.tick()
  out = {"row": w.row(f"pushed {push_ns:.0f} N*s"),
         "up": float(rig.data.xmat[rig.root].reshape(3, 3)[2, 2]),
         "tool_on_fork": rig.powered(), "tool_z": float(rig.data.xpos[rig.tool_body][2])}
  rig.drv = PolicyDriver(rig.model, rig.data,
                         WalkingPolicy(POLICY_NPZ.with_name("quadruped_getup.npz")))
  t0 = d.time
  stood = None
  held = 0.0
  while d.time - t0 < getup_s and stood is None:
    rig.step(Twist())
    upright = rig.data.xmat[rig.root].reshape(3, 3)[2, 2] > 0.95
    high = abs(d.qpos[2] - CHOSEN.stand_height) < 0.04
    held = held + rig.model.opt.timestep if (upright and high) else 0.0
    if held >= 0.5:
      stood = d.time - t0 - 0.5
  out["stood"] = stood
  return out


def retention_fall(push_ns: float = 30.0, reflex: bool = True) -> None:
  r = fall(push_ns, reflex)
  print(RETENTION_HEAD)
  print(r["row"] + f"   torso up-axis z {r['up']:+.2f}"
        + ("; the arm folded" if reflex else "; the arm held at its carry pose"))
  print(f"  after the fall: tool on the fork {r['tool_on_fork']}, tool at z {r['tool_z']:.2f} m")
  print(f"  the get-up policy: "
        f"{'stood in %.1f s' % r['stood'] if r['stood'] is not None else 'DID NOT STAND'}")


# ---- getting up with an arm on its back ---------------------------------------------

def getup_one(args) -> dict:
  """One of `quad_spike.getup`'s drops (the same seed's orientation and
  legs), with the arm folded and held at its stow -- or, `arm` False, the
  placeholder the policy trained with."""
  trial, with_arm = args
  from pluggybot.legs.model import SIZING
  from pluggybot.legs.model import body_xml as bx
  from pluggybot.legs.policy import POLICY_NPZ
  sys.path.insert(0, str(ROOT / "scripts"))
  import quad_spike as qs
  spec = am.ArmSpec()
  # the premise is #377's placeholder: `CHOSEN` with no arm given carries its own
  model = mujoco.MjModel.from_xml_string(
    bx(CHOSEN, arm=am.arm_mjcf(spec)) if with_arm else bx(SIZING))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  rng = np.random.default_rng(1000 + trial)
  r, p, y = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi / 2, math.pi / 2), \
    rng.uniform(-math.pi, math.pi)
  data.qpos[:7] = [0, 0, 0.45, *qs._rpy_quat(r, p, y)]
  lo, hi = model.jnt_range[1:13].T
  data.qpos[7:19] = rng.uniform(lo * 0.9, hi * 0.9)
  mujoco.mj_forward(model, data)
  drv = PolicyDriver(model, data, WalkingPolicy(POLICY_NPZ.with_name("quadruped_getup.npz")))
  arm = am.ArmDriver(model, data, spec) if with_arm else None
  if arm is not None:
    arm.aim(*spec.stow)
  root = model.body(ROBOT_ROOT).id
  held, stood = 0.0, None
  while data.time < 6.0:
    drv.command(Twist())
    if arm is not None:
      arm.step()
    mujoco.mj_step(model, data)
    upright = data.xmat[root].reshape(3, 3)[2, 2] > 0.95
    high = abs(data.qpos[2] - CHOSEN.stand_height) < 0.04
    held = held + model.opt.timestep if (upright and high) else 0.0
    if held >= 0.5 and stood is None:
      stood = data.time - 0.5
  return {"trial": trial, "arm": with_arm, "stood": stood}


def getup_table(trials: int = 20, jobs: int = 3) -> None:
  todo = [(k, a) for a in (False, True) for k in range(trials)]
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(getup_one, todo, chunksize=1)
  for with_arm in (False, True):
    mine = [r for r in rows if r["arm"] == with_arm]
    ok = [r["stood"] for r in mine if r["stood"] is not None]
    label = "the arm, folded" if with_arm else "the placeholder (as trained)"
    print(f"{label:30s} stood {len(ok)} of {len(mine)}"
          + (f", median {np.median(ok):.1f} s" if ok else "")
          + "; failed: " + ", ".join(str(r["trial"]) for r in mine if r["stood"] is None))


# ---- what the arm hides from the robot's sensors ----------------------------------

def occlusion(rig: Rig, cams=("nav_eye", "depth_eye")) -> dict:
  """What the arm (and a carried tool) hides, as the robot stands now: the
  LIDAR's rays whose first hit is the arm or the tool (the scan drops the
  robot's own body, so those bearings are blind), and each camera's pixels
  that see them (a segmentation render, the cameras' own resolution)."""
  m, d = rig.model, rig.data
  mine = {g for g in range(m.ngeom)
          if m.body(m.geom_bodyid[g]).name.startswith("arm_")
          or m.geom_bodyid[g] == rig.tool_body}
  site = m.site("lidar").id
  n = 360
  ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
  dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1) @ d.site_xmat[site].reshape(3, 3).T
  hit = np.zeros(n, np.int32)
  dist = np.zeros(n)
  mujoco.mj_multiRay(m, d, d.site_xpos[site].copy(), dirs.reshape(-1).copy(), None, 1, -1,
                     hit, dist, None, n, mujoco.mjMAXVAL)
  out = {"lidar": int(sum(1 for h in hit if h in mine))}
  for cam in cams:
    r = mujoco.Renderer(m, 240, 424)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render()[..., 0]
    r.close()
    out[cam] = float(np.isin(seg, list(mine)).mean())
  return out


def sensors_table() -> None:
  """Each pose the arm walks in, and what it hides."""
  print(f"{'pose':34s} {'LIDAR rays blind':>16s} {'nav camera %':>12s} {'depth camera %':>14s}")
  rig = Rig(rack=None)
  rig.hold(1.0)
  rig.arm.aim(*rig.spec.stow)
  rig.hold(3.0)
  rig.data.qpos[rig.tool_adr:rig.tool_adr + 3] = [5.0, 5.0, 0.1]     # out of the way
  mujoco.mj_forward(rig.model, rig.data)
  o = occlusion(rig)
  print(f"{'stowed, no tool':34s} {o['lidar']:16d} {o['nav_eye'] * 100:12.2f} "
        f"{o['depth_eye'] * 100:14.2f}")
  rig.fork_to(*CARRY, speed=0.2)
  rig.mount_tool()
  rig.hold(1.0)
  o = occlusion(rig)
  print(f"{'carrying at ' + str(CARRY):34s} {o['lidar']:16d} {o['nav_eye'] * 100:12.2f} "
        f"{o['depth_eye'] * 100:14.2f}")


# ---- the approach: walking in to a bay and taking its tool -------------------------

#: The approach starts here: on the bay's axis, this far behind the working
#: pose (the dock's standoff), facing the rack.
APPROACH_STANDOFF = 1.0
LOOK_EVERY_S = 0.25
START_ACROSS_M, START_ALONG_M, START_YAW_DEG = 0.3, 0.3, 30.0


class Approach(Rig):
  """A Rig whose rack stands at a fixed pose: bay BAY's working pose at the
  world's origin, the robot starting APPROACH_STANDOFF behind it."""

  def __init__(self, seed: int = 0, **kw):
    super().__init__(**kw)
    from pluggybot.legs.odometry import LegOdometry
    from pluggybot.rack.tags import TagDetector
    m = self.model
    wx, wy, _ = rk.work_pose(self.rack, BAY)
    # The rack frame -> world: yaw pi, and the working pose at the origin.
    m.body_pos[self.rack_body] = [wx, wy, 0.0]
    m.body_quat[self.rack_body] = [0.0, 0.0, 0.0, 1.0]
    px, py, pz = rk.bay_peg(self.rack, BAY, pos=(wx, wy), yaw=math.pi)
    self.data.qpos[self.tool_adr:self.tool_adr + 7] = [px, py, pz - PEG_ABOVE_BODY + 0.0003,
                                                       0, 0, 0, 1]
    mujoco.mj_forward(m, self.data)
    self.det = TagDetector(m, "nav_eye", tag_size=self.rack.tag_size)
    self.odo_cls = LegOdometry
    self.seed = seed
    self.rack_belief = None
    self.looks = 0

  def close(self) -> None:
    self.det.close()

  def place(self, x: float, y: float, yaw: float, belief) -> None:
    from pluggybot.legs import dock as dk  # noqa: F401 (the frame helpers)
    d = self.data
    d.qpos[0], d.qpos[1] = x, y
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(self.model, d)
    self.odo = self.odo_cls(self.model, d, seed=self.seed)
    self.odo.x, self.odo.y, self.odo.yaw = belief
    self.hold(0.5)

  def step(self, twist: Twist = Twist()) -> None:
    super().step(twist)
    if getattr(self, "odo", None) is not None:
      self.odo.step()

  def pose(self):
    return self.odo.x, self.odo.y, self.odo.yaw

  def truth(self):
    d = self.data
    return float(d.qpos[0]), float(d.qpos[1]), _quat_rpy(d.qpos[3:7])[2]

  def look(self):
    """One decode of the rack's tags; on a fit, the rack's believed pose
    (in the odometry frame) moves toward it."""
    from pluggybot.legs import dock as dk
    self.looks += 1
    seen = dk.seen_from(self.model, self.data, self.det.detect(self.data), "nav_eye",
                        self.root)
    fix = rk.fit_rack(seen, self.rack)
    if fix is not None:
      self.rack_belief = dk.blend(self.rack_belief, dk.compose(self.pose(),
                                                               (fix.x, fix.y, fix.yaw)))
    return fix

  def err(self):
    """Where the robot believes it stands in the WORK frame."""
    from pluggybot.legs import dock as dk
    wx, wy, wyaw = rk.work_pose(self.rack, BAY)
    work = dk.compose(self.rack_belief, (wx, wy, wyaw))
    return dk.relative(self.pose(), work)

  def aim(self) -> rk.BayAim | None:
    """The bay, measured off the rack's tags from where the robot stands."""
    seen = rk.seen_in_torso(self.model, self.data, self.det.detect(self.data), "nav_eye",
                            self.root)
    return rk.bay_aim(seen, self.rack, BAY, self.spec.y)


#: No rack in view: turn on the spot by these steps (deg: +25, -25, +50,
#: -50, +75, -75 from where it started) and look after each (the dock's).
SEARCH_STEPS = (25.0, -50.0, 75.0, -100.0, 125.0, -150.0)


def turn_by(rig: Approach, degrees: float, budget_s: float = 10.0) -> None:
  """Turn on the spot by `degrees` of the legs' own heading, then settle."""
  from pluggybot.legs import dock as dk
  target = rig.odo.yaw + math.radians(degrees)
  t0 = rig.data.time
  while (abs(dk._wrap(target - rig.odo.yaw)) > math.radians(3.0)
         and rig.data.time - t0 < budget_s):
    rig.step(Twist(yaw_rate=math.copysign(rk.TURN_W, dk._wrap(target - rig.odo.yaw))))
  rig.hold(0.4)


def find_rack(rig: Approach) -> bool:
  """Look; failing that, sweep the spot it stands on until the rack's tags
  fit (a start turned 26 deg away, 0.27 m off, saw none of them)."""
  if rig.look() is not None:
    return True
  for step in SEARCH_STEPS:
    turn_by(rig, step)
    if rig.look() is not None:
      return True
  return False


def walk_in(rig: Approach, budget_s: float = 20.0) -> str:
  t0 = last = rig.data.time
  while rig.data.time - t0 < budget_s:
    tw = rk.walk_in_twist(*rig.err())
    if tw == Twist():
      return "stopped"
    rig.step(tw)
    if rig.data.time - last >= LOOK_EVERY_S:
      rig.look()
      last = rig.data.time
  return "budget"


def back_out(rig: Approach, metres: float = 0.6, budget_s: float = 6.0) -> None:
  x0, y0, _ = rig.pose()
  t0 = rig.data.time
  while (math.hypot(rig.odo.x - x0, rig.odo.y - y0) < metres
         and rig.data.time - t0 < budget_s):
    rig.step(Twist(vx=-rk.APPROACH_V))
  rig.hold(0.8)


def approach(rig: Approach, tries: int = 3, carry_s: float = 0.0) -> dict:
  """From the standoff: look, walk in, stop, measure the bay, take the tool
  if lined up (else back out and try again), then hang it back -- holding
  it up at the carry pose `carry_s` first, for the viewer."""
  out = {"picked": False, "returned": False, "tries": 0, "why": ""}
  t0 = rig.data.time
  for attempt in range(tries):
    out["tries"] = attempt + 1
    if not find_rack(rig) and rig.rack_belief is None:
      out["why"] = "no rack"
      break
    why = walk_in(rig)
    rig.hold(rk.SETTLE_AFTER_WALK_S)
    a = rig.aim()
    if a is None:
      out["why"] = "no fit at the bay"
      back_out(rig)
      continue
    out.update(across=a.across, yaw=math.degrees(a.yaw), walk=why)
    if not a.lined_up:
      out["why"] = "not lined up"
      back_out(rig)
      continue
    # The truth, for the table: how far off the robot really stood.
    p = rig.body_frame(rig.peg())
    tx, tz = aim_of(rig)
    out.update(true_across=float(p[1]), aim_err_x=a.x - tx, aim_err_z=a.z - tz)
    res = rig.pick((a.x, a.z))
    out["picked"] = res["picked"]
    if res["picked"] and carry_s:
      rig.fork_to(*CARRY, speed=0.2)
      rig.hold(carry_s)
      rig.fork_to(a.x - BACK_OUT, a.z - rig.drop + rig.lift, speed=0.2)
      # Swinging a tool up and down shifts the stance under the arm (25 mm
      # of it, measured): an aim taken before is stale. Stand, look again.
      rig.hold(rk.SETTLE_AFTER_WALK_S)
      a = rig.aim() or a
    if res["picked"]:
      out["returned"] = rig.put_back((a.x, a.z))["returned"]
    out["why"] = "" if out["picked"] and out["returned"] else "coupling"
    break
  out["seconds"] = rig.data.time - t0
  return out


def approach_one(args) -> dict:
  k, across, along, yaw_deg = args
  rig = Approach(seed=k)
  rig.hold(1.0)
  rig.place(-APPROACH_STANDOFF + along, across, math.radians(yaw_deg),
            belief=(-APPROACH_STANDOFF, 0.0, 0.0))
  r = approach(rig)
  rig.close()
  r.update(start=(across, along, yaw_deg))
  return r


def approach_table(n: int = 20, jobs: int = 3) -> None:
  rng = np.random.default_rng(378)
  todo = [(k, 0.0, 0.0, 0.0) for k in range(1)]
  todo += [(k + 1, rng.uniform(-START_ACROSS_M, START_ACROSS_M),
            rng.uniform(-START_ALONG_M, START_ALONG_M),
            rng.uniform(-START_YAW_DEG, START_YAW_DEG)) for k in range(n)]
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(approach_one, todo, chunksize=1)
  print(f"{'start across/along m, yaw':>26s} {'tries':>5s} {'stood across mm':>15s} "
        f"{'yaw deg':>7s} {'aim err x/z mm':>14s} {'picked':>6s} {'returned':>8s} {'s':>5s}  why")
  for r in rows:
    a, al, y = r["start"]
    print(f"{a:+8.2f} {al:+6.2f} {y:+6.0f}       {r['tries']:5d} "
          f"{r.get('true_across', float('nan')) * 1000:15.1f} {r.get('yaw', float('nan')):7.1f} "
          f"{r.get('aim_err_x', float('nan')) * 1000:6.1f}/{r.get('aim_err_z', float('nan')) * 1000:5.1f}"
          f"   {'yes' if r['picked'] else 'NO':>6s} {'yes' if r['returned'] else 'NO':>8s} "
          f"{r['seconds']:5.1f}  {r['why']}")
  ok = [r for r in rows if r["picked"] and r["returned"]]
  print(f"picked and returned {len(ok)} of {len(rows)} "
        f"({sum(r['tries'] == 1 for r in ok)} first try)")


# ---- the level-tool choice: reach and holding torque -------------------------------

#: Working points from a seated peg, in the level plate frame (x forward,
#: z up): the rover's claw's grip (`rack.coupling.CLAW_REACH`, 154 mm under
#: its peg) and its pen's tip (71 mm ahead, 72 mm under, measured).
CLAW_GRIP = (0.055, -0.154)
PEN_TIP = (0.071, -0.072)
#: The posture policy's measured pitch (SimNotes, "The quadruped body": it
#: pitches +-11 deg) and height offsets (-0.14..+0.03 m trained).
BODY_PITCH = math.radians(11.0)
BODY_HEIGHT = (-0.14, 0.03)
#: The heaviest thing the arm lifts: the claw holding the bench's heaviest
#: cube (`challenge.bench.MAX_KG`).
CLAW_KG, CUBE_KG = 0.211, 0.40
#: The body: where the arm must not go (torso frame): below the torso's top
#: while behind the nose's sensors.
NOSE_X, TOP_Z = 0.26, CHOSEN.torso[2] + 0.011 + 0.005


def _clear(spec, q) -> bool:
  qs, qe = q
  pts = [(spec.shoulder_x, spec.shoulder_z), am.elbow_xz(spec, qs), am.wrist_xz(spec, qs, qe),
         am.seat_xz(spec, qs, qe)]
  for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
    for t in np.linspace(0, 1, 30):
      x, z = x0 + t * (x1 - x0), z0 + t * (z1 - z0)
      if x < NOSE_X and z < TOP_Z:
        return False
      if z < -CHOSEN.stand_height + 0.01:
        return False
  return True


def _reach(spec, x, z, level: str, height: float = 0.0):
  """A pose putting the seated peg at (x, z) (torso frame of a body standing
  `height` off its stand) with the plate level in the WORLD, or None.
  Returns (q, body pitch it needs)."""
  if level in ("parallelogram", "wrist"):
    for elbow in ("up", "down"):
      q = am.solve_seat(spec, x, z, elbow)
      if q is not None and _clear(spec, q):
        return q, 0.0
    return None
  # "body": the plate is the forearm's; level only where the forearm's
  # angle is the body's pitch undone -- search the pitch the posture has.
  best = None
  for pitch in np.linspace(-BODY_PITCH, BODY_PITCH, 23):
    c, sn = math.cos(pitch), math.sin(pitch)
    # The target in the pitched torso's frame (pitch about the torso's centre).
    xt, zt = c * x + sn * z, -sn * x + c * z
    fx, fz = spec.fork.vertex_x, spec.fork.vertex_z + spec.fork.seat_rise()
    phi = -pitch                       # the forearm's angle that levels the plate
    wx, wz = xt - (fx * math.cos(phi) - fz * math.sin(phi)), zt - (fx * math.sin(phi) + fz * math.cos(phi))
    ex, ez = wx - spec.fore * math.cos(phi), wz - spec.fore * math.sin(phi)
    d = math.hypot(ex - spec.shoulder_x, ez - spec.shoulder_z)
    if abs(d - spec.upper) > 0.004:
      continue
    qs = math.atan2(ez - spec.shoulder_z, ex - spec.shoulder_x)
    q = (qs, phi - qs)
    if _clear(spec.with_(level="parallelogram"), q):
      if best is None or abs(pitch) < abs(best[1]):
        best = (q, pitch)
  return best


def reach_table(spec: am.ArmSpec = am.ArmSpec()) -> None:
  h0 = CHOSEN.stand_height
  targets = []
  gx, gz = CLAW_GRIP
  for fx in (0.40, 0.45, 0.50):
    targets.append((f"claw on the floor, {fx:.2f} m ahead", fx - gx, 0.013 - h0 - gz, CLAW_KG + CUBE_KG))
  targets.append(("claw on a 0.8 m tabletop, 0.40 ahead", 0.40 - gx, 0.813 - h0 - gz,
                  CLAW_KG + CUBE_KG))
  px, pz = PEN_TIP
  for zb in (0.19, 0.30, 0.41):
    targets.append((f"pen at a board's z {zb:.2f}, 0.43 ahead", 0.43 - px, zb - h0 - pz, CLAW_KG))
  targets.append(("a rack bay's peg (0.50 m)", rk.WORK_X, rk.DEFAULT.peg_z - h0, CLAW_KG))
  targets.append((f"carrying at {CARRY}", CARRY[0], CARRY[1] + 0.0042, CLAW_KG))
  print(f"arm: shoulder ({spec.shoulder_x}, {spec.shoulder_z}) on the torso, links "
        f"{spec.upper} + {spec.fore} m, {spec.motor.name} x2 ({spec.mass:.2f} kg without a tool)")
  print(f"{'target':40s} {'(a) parallelogram':>22s} {'(c) the body pitches, crouches':>32s}")
  for name, x, z, kg in targets:
    a = _reach(spec, x, z, "parallelogram")
    if a is not None:
      ts, te = am.gravity_torques(spec, *a[0], tool_kg=kg, tool_x=CLAW_GRIP[0] if kg > CLAW_KG else 0.0,
                                  tool_z=-0.08)
      ca = f"yes {ts:+5.2f}/{te:+5.2f} N*m"
    else:
      ca = "NO"
    c = None
    for dh in np.linspace(*BODY_HEIGHT, 18):
      got = _reach(spec, x, z - dh, "body")
      if got is not None and (c is None or abs(dh) < abs(c[2])):
        c = (*got, dh)
    cc = (f"pitch {math.degrees(c[1]):+5.1f} deg, {c[2] * 100:+4.0f} cm" if c is not None
          else "NO (past +-11 deg)")
    print(f"{name:40s} {ca:>22s} {cc:>32s}")
  # (b): the wrist motor's weight at the tip, against (a), at full reach.
  print("holding at full horizontal reach, the claw and a 0.4 kg cube "
        "(shoulder / elbow [/ wrist] N*m, and their sum):")
  q = am.solve_seat(spec, spec.shoulder_x + spec.reach + spec.fork.vertex_x - 0.002,
                    spec.shoulder_z + spec.fork.vertex_z + 0.0042)
  base = am.gravity_torques(spec, *q, tool_kg=CLAW_KG + CUBE_KG, tool_x=CLAW_GRIP[0], tool_z=-0.08)
  print(f"  (a) parallelogram, no motor at the tip: {base[0]:5.2f} / {base[1]:5.2f}"
        f"          = {sum(base):5.2f}")
  for wm in (GIM4310_10, GIM8108_8):
    sb = spec.with_(level="wrist", wrist_motor=wm)
    tb = am.gravity_torques(sb, *q, tool_kg=CLAW_KG + CUBE_KG, tool_x=CLAW_GRIP[0], tool_z=-0.08)
    print(f"  (b) + a wrist motor, {wm.name} ({wm.mass * 1000:.0f} g): "
          f"{tb[0]:5.2f} / {tb[1]:5.2f} / {tb[2]:5.2f} = {sum(tb):5.2f} "
          f"(+{sum(tb) / sum(base) * 100 - 100:.0f} %)")


def envelope_one(args) -> dict:
  """A tool of `kg` whose CoM sits `ahead` of its peg (+: away from the
  robot, the business end's side, where a racked tool faces the wall),
  carried through a 1.0 m/s trot and a stop: still seated, and at what
  angle does it hang?"""
  kg, ahead = args
  rig = carrying(tool_kg=kg, ahead=ahead)
  w = Watch(rig)
  for seconds, twist in ((3.0, Twist(vx=1.0)), (2.0, Twist())):
    t0 = rig.data.time
    while rig.data.time - t0 < seconds:
      rig.step(twist)
      w.tick()
  return {"kg": kg, "ahead": ahead, "seated": rig.powered(), "longest_ms": w.worst * 2,
          "lean": rig.tool_tilt(), "swing": w.swing}


def envelope_table(jobs: int = 3) -> None:
  todo = [(kg, a) for kg in (0.25, 0.40, 0.60)
          for a in (-0.03, 0.0, 0.03, 0.06, 0.09)]
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(envelope_one, todo, chunksize=1)
  print("a tool whose CoM is off its peg, carried through a 1.0 m/s trot and a stop "
        "(+ ahead: away from the robot, which leans it onto the lean-pad):")
  print(f"{'tool kg':>7s} {'CoM ahead mm':>12s} {'moment N*m':>10s} {'seated':>7s} "
        f"{'longest open ms':>15s} {'hangs at deg':>12s} {'swing deg':>9s}")
  for r in rows:
    print(f"{r['kg']:7.2f} {r['ahead'] * 1000:12.0f} {r['kg'] * 9.81 * r['ahead']:10.3f} "
          f"{'yes' if r['seated'] else 'NO':>7s} {r['longest_ms']:15.0f} {r['lean']:12.1f} "
          f"{r['swing']:9.1f}")


# ---- the viewer ------------------------------------------------------------------

def watch(rig: Rig, scene, *, track: bool = True, lookat=None, distance: float = 2.2,
          azimuth: float = 150.0, elevation: float = -18.0) -> None:
  """Run `scene(rig)` over and over in the MuJoCo viewer, at real time, until
  the window closes. MuJoCo's azimuth 0 looks along +x, the way the robot
  faces at the start."""
  from mujoco import viewer as mj_viewer
  with mj_viewer.launch_passive(rig.model, rig.data) as viewer:
    if track:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = rig.root
    else:
      viewer.cam.lookat[:] = lookat
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = distance, azimuth, elevation
    rig.viewer = viewer
    try:
      while viewer.is_running():
        scene(rig)
    except ViewerClosed:
      pass
  rig.viewer = None


def view_fetch(seed: int | None = None) -> None:
  """Fetches, one after another, each from a new random start at the
  approach's standoff: the rack found, the walk in by its tags, the pick,
  the tool held up at the carry pose, hung back, the arm stowed."""
  rng = np.random.default_rng(seed)
  rig = Approach(seed=0)
  rig.hold(1.0)
  print("Each fetch starts 1 m behind the bay's working pose, TRULY off by a random "
        "offset while the robot believes it stands exactly there.")

  def scene(rig):
    across = rng.uniform(-START_ACROSS_M, START_ACROSS_M)
    along = rng.uniform(-START_ALONG_M, START_ALONG_M)
    yaw = rng.uniform(-START_YAW_DEG, START_YAW_DEG)
    print(f"fetch: {across:+.2f} m across, {along:+.2f} m along, {yaw:+.0f} deg turned")
    rig.place(-APPROACH_STANDOFF + along, across, math.radians(yaw),
              belief=(-APPROACH_STANDOFF, 0.0, 0.0))
    rig.rack_belief = None
    r = approach(rig, carry_s=3.0)
    print(f"  picked {r['picked']}, hung back {r['returned']} after {r['tries']} "
          f"{'try' if r['tries'] == 1 else 'tries'} {r['why']}")
    rig.arm.aim(*rig.spec.stow)
    rig.hold(2.0)
    back_out(rig, 0.8)
  try:
    watch(rig, scene, track=False, lookat=(-0.3, 0.0, 0.35), distance=2.6,
          azimuth=25.0, elevation=-20.0)
  finally:
    rig.close()


#: The carry scene's loop: out and back, so the robot stays in view.
CARRY_LOOP = (
  ("walk 0.3 m/s", 5.0, Twist(vx=0.3)),
  ("trot 0.6 m/s", 4.0, Twist(vx=0.6)),
  ("trot 1.0 m/s", 3.0, Twist(vx=1.0)),
  ("stop from 1.0 m/s", 2.0, Twist()),
  ("turn round", 4.0, Twist(yaw_rate=0.8)),
  ("trot 1.0 m/s", 3.0, Twist(vx=1.0)),
  ("trot 0.6 m/s", 4.0, Twist(vx=0.6)),
  ("walk 0.3 m/s", 5.0, Twist(vx=0.3)),
  ("stop", 2.0, Twist()),
  ("turn round", 4.0, Twist(yaw_rate=-0.8)),
  ("sidestep left", 3.0, Twist(vy=0.3)),
  ("sidestep right", 3.0, Twist(vy=-0.3)),
  ("stand", 2.0, Twist()),
)


def carry_scene(rig: Rig) -> None:
  for name, seconds, twist in CARRY_LOOP:
    print(f"  {name}")
    rig.hold(seconds, twist)


def view_carry() -> None:
  """A tool carried high over the nose through walks, trots, hard stops,
  turns and sidesteps, on the flat policy."""
  print("A tool carried at the carry pose, over the nose and above the LIDAR's "
        "scan plane: the gravity seat holds; watch the swing at the stops.")
  watch(carrying(), carry_scene, distance=2.0, azimuth=120.0, elevation=-12.0)


def _yaw(rig: Rig) -> float:
  return _quat_rpy(rig.data.qpos[3:7])[2]


def _turn_to(rig: Rig, heading: float, budget_s: float = 8.0) -> None:
  """Turn on the spot to a heading (the true one: a viewer's scene, not a
  measurement)."""
  t0 = rig.data.time
  while rig.data.time - t0 < budget_s:
    e = math.atan2(math.sin(heading - _yaw(rig)), math.cos(heading - _yaw(rig)))
    if abs(e) < math.radians(8.0):
      break
    rig.step(Twist(yaw_rate=math.copysign(0.8, e)))
  rig.hold(1.0)


def hill_scenery(rise: float, steps: int, edge: float, landing: float) -> str:
  """The house's flight up from x = `edge`, a `landing` across the top, and
  a flight down the far side: a robot crosses it without turning on it."""
  import quad_spike as qs
  box = ('\n    <geom name="{}" type="box" size="{} 1.0 {}" pos="{} 0 {}" '
         'rgba="0.6 0.5 0.4 1"/>')
  top_x = edge + steps * qs.TREAD_M
  g = [qs.staircase_scenery(rise, steps, edge).split('\n    <geom name="landing"')[0]]
  g.append(box.format("landing", landing / 2, steps * rise / 2, top_x + landing / 2,
                      steps * rise / 2))
  for i in range(steps - 1):
    h = (steps - 1 - i) * rise
    g.append(box.format(f"down{i}", qs.TREAD_M / 2, h / 2,
                        top_x + landing + (i + 0.5) * qs.TREAD_M, h / 2))
  return "".join(g)


def _cross(rig: Rig, heading: float, until) -> bool:
  """Walk at 0.4 m/s holding `heading` until `until()`, at most 60 s."""
  t0 = rig.data.time
  while not until() and rig.data.time - t0 < 60.0:
    e = math.atan2(math.sin(heading - _yaw(rig)), math.cos(heading - _yaw(rig)))
    rig.step(Twist(vx=0.4, yaw_rate=max(-0.5, min(0.5, 2.0 * e))))
  return until()


def view_stairs(rise: float = 0.18, steps: int = 10, landing: float = 1.5) -> None:
  """Over a hill of the house's flight -- up, across a landing, down the
  far side -- carrying a tool on the seeing policy, turning round on the
  flat at each end: the flights from rest, facing along them, that the
  stairs' table flies."""
  sys.path.insert(0, str(ROOT / "scripts"))
  import quad_spike as qs
  from pluggybot.legs.policy import POLICY_NPZ
  edge = qs.CLIMB_EDGE_M
  far = edge + (2 * steps - 1) * qs.TREAD_M + landing
  rig = carrying(policy=POLICY_NPZ.with_name("quadruped_rough_seeing.npz"),
                 scenery=hill_scenery(rise, steps, edge, landing))
  d = rig.data
  print(f"Over a hill of the house's flight ({steps} risers of {rise} m each way), "
        f"carrying a tool: the plate pitches with the torso, the tool hangs plumb.")

  def scene(rig):
    for heading, name, until in (
        (0.0, "over the hill", lambda: d.qpos[0] > far + 0.8),
        (math.pi, "and back", lambda: d.qpos[0] < edge - 0.8)):
      print(f"  {name}")
      if not _cross(rig, heading, until):
        print("  (stalled on the flight: the policy's, not the coupling's)")
      print(f"  the tool {'still on the fork' if rig.powered() else 'is OFF the fork'}")
      rig.hold(1.0)
      _turn_to(rig, math.pi - heading)
  watch(rig, scene, distance=3.0, azimuth=90.0, elevation=-10.0)


def view_fall(push_ns: float = 30.0) -> None:
  """Trotting with a tool, pushed over; the get-up policy stands it. Every
  other fall the arm stays at its carry pose -- the premise -- and the robot
  cannot get up."""
  from pluggybot.legs.policy import POLICY_NPZ
  rig = carrying()
  walking = rig.drv.policy
  getup = WalkingPolicy(POLICY_NPZ.with_name("quadruped_getup.npz"))
  count = [0]
  print("Pushed over while trotting with a tool. The tool is thrown either way; "
        "the arm folds as the torso passes 60 deg, except every other time.")

  def scene(rig):
    m, d = rig.model, rig.data
    reflex = count[0] % 2 == 0
    count[0] += 1
    print(f"  trotting ... pushed ({'the arm folds' if reflex else 'the arm held out: the premise'})")
    t0 = d.time
    while d.time - t0 < 2.0:
      rig.step(Twist(vx=0.6))
    d.xfrc_applied[rig.root] = [0.0, push_ns / 0.1, 0.0, 0.0, 0.0, 0.0]
    t0 = d.time
    while d.time - t0 < 0.1:
      rig.step(Twist(vx=0.6))
    d.xfrc_applied[rig.root] = 0.0
    folded = False
    while d.time - t0 < 2.0:
      if reflex and not folded and d.xmat[rig.root].reshape(3, 3)[2, 2] < am.FOLD_ON_FALL_COS:
        rig.arm.payload = (0.0, (0.0, 0.0))
        rig.arm.aim(*rig.spec.stow)
        folded = True
      rig.step(Twist())
    rig.drv = PolicyDriver(m, d, getup)
    t0, held, stood = d.time, 0.0, None
    while d.time - t0 < 5.0 and stood is None:
      rig.step(Twist())
      upright = d.xmat[rig.root].reshape(3, 3)[2, 2] > 0.95
      high = abs(d.qpos[2] - CHOSEN.stand_height) < 0.04
      held = held + m.opt.timestep if (upright and high) else 0.0
      if held >= 0.5:
        stood = d.time - t0 - 0.5
    print(f"  {'stood in %.1f s' % stood if stood is not None else 'cannot get up'}")
    rig.hold(1.5)
    # Stand it back up where it started, the arm stowed, the tool on the fork.
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[rig.tool_adr:rig.tool_adr + 3] = [5.0, 5.0, 0.1]
    mujoco.mj_forward(m, d)
    rig.drv = PolicyDriver(m, d, walking)
    rig.arm.target = np.array(rig.arm.q())
    rig.arm.goal = rig.arm.target.copy()
    rig.arm.payload = (0.0, (0.0, 0.0))
    rig.hold(1.0)
    rig.fork_to(*CARRY, speed=0.2)
    rig.mount_tool()
    rig.hold(1.0)
  watch(rig, scene, distance=2.2, azimuth=150.0, elevation=-18.0)


#: What `--view reach` visits, in the torso frame: the fork's V vertex for
#: each target (its working point is a marker), and how long it holds there.
def reach_targets(spec: am.ArmSpec) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
  """(name, the V vertex's (x, z), the working point's (x, z)), torso frame."""
  h0 = CHOSEN.stand_height
  r = spec.fork.seat_rise()
  gx, gz = CLAW_GRIP
  px, pz = PEN_TIP
  out = [("the carry pose", CARRY, (CARRY[0], CARRY[1] + r - 0.022))]
  out.append(("a rack bay's peg, 0.50 m up", (WORK_X, rk.DEFAULT.peg_z - h0 - r),
              (WORK_X, rk.DEFAULT.peg_z - h0)))
  for zb in (0.41, 0.30, 0.19):
    tip = (0.43, zb - h0)
    out.append((f"a pen at a board's {zb:.2f} m row, 0.43 m ahead",
                (tip[0] - px, tip[1] - pz - r), tip))
  grip = (0.45, 0.013 - h0)
  out.append(("a claw's grip on the floor, 0.45 m ahead", (grip[0] - gx, grip[1] - gz - r), grip))
  out.append(("the carry pose", CARRY, (CARRY[0], CARRY[1] + r - 0.022)))
  grip = (0.40, 0.813 - h0)
  out.append(("a claw's grip on a 0.8 m tabletop, 0.40 m ahead",
              (grip[0] - gx, grip[1] - gz - r), grip))
  return out


def view_reach() -> None:
  """The arm through the targets that chose it, the plate level at every one;
  a red marker at each working point (a claw's grip, a pen's tip)."""
  spec = am.ArmSpec()
  targets = reach_targets(spec)
  markers = "".join(
    f'\n    <geom name="mark{k}" type="sphere" size="0.012" pos="0 0 -1" '
    f'contype="0" conaffinity="0" rgba="0.9 0.1 0.1 0.8"/>' for k in range(len(targets)))
  rig = Rig(rack=None, scenery=markers)
  rig.hold(SETTLE_S)
  m, d = rig.model, rig.data
  rot = d.xmat[rig.root].reshape(3, 3)
  for k, (_, _, (wx, wz)) in enumerate(targets):
    m.geom_pos[m.geom(f"mark{k}").id] = d.xpos[rig.root] + rot @ np.array([wx, 0.0, wz])
  rig.fork_to(*CARRY, speed=0.2)
  rig.mount_tool()
  rig.hold(1.0)
  print("The arm through the targets that chose it; a red marker at each working "
        "point. The plate stays level at every one (the parallelogram).")

  def scene(rig):
    for name, (vx, vz), _ in targets:
      print(f"  {name}")
      rig.fork_to(vx, vz, speed=0.15)
      rig.hold(2.0)
  watch(rig, scene, track=False, lookat=d.xpos[rig.root] + np.array([0.3, 0.0, 0.1]),
        distance=1.9, azimuth=90.0, elevation=-5.0)


# ---- the served world: fetch and stow at the rack in the house (#405) ---------------

#: Where a flight "from across the house" starts: the house's spawns, each
#: facing its own way, the workshop's moved off its table (the rover's
#: spawn there is inside it: a quadruped set down on it fell), and the
#: bedroom's and the south garden's middles -- the kitchen, the workshop
#: and the garden are the far rooms.
HOUSE_STARTS = {"kitchen": (-8.5, 4.0, 0.0), "workshop": (-7.0, 0.5, 0.0),
                "hall": (-3.5, 1.0, 0.0), "bedroom": (1.5, 4.25, -1.19),
                "garden": (7.5, 2.0, 0.0), "garden_south": (4.0, -4.0, -2.41),
                "living": (1.5, 0.5, 1.5708)}
#: A flight's budget, sim s: each swap walks up to the drive's patience.
SERVED_S = 400.0


def _served_start(kind: str, k: int, body) -> tuple[float, float, float]:
  """Where flight `k` starts: `dock`, on the dock's approach standoff (where
  a robot that undocked stands) jittered 0.1 m and 15 deg, or `house`, one
  of `HOUSE_STARTS` in turn."""
  rng = np.random.default_rng(405 + k)
  if kind == "dock":
    x, y, yaw = body.charge_standoff()
    return (x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1),
            yaw + math.radians(rng.uniform(-15, 15)))
  return tuple(HOUSE_STARTS.values())[k % len(HOUSE_STARTS)]


def served_one(args) -> dict:
  """One served quadruped, a fresh map, from a start: fetch a tool off its
  bay and hang it back. Measured off the world: seated and conducting after
  the fetch, hung after the stow."""
  kind, k = args
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  from pluggybot.rack.coupling import STATION_YS
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=home.GRID_BOUNDS)
  start = _served_start(kind, k, body)
  body.start_at(*start)
  tool = tuple(rk.TOOL_BAYS)[k % len(rk.TOOL_BAYS)]
  station = STATION_YS[rk.TOOL_BAYS[tool]]
  out = {"kind": kind, "k": k, "tool": tool, "start": [round(v, 2) for v in start]}
  t0 = data.time
  out["fetchWhy"] = body.run(body.fetch_tool_routine(station, tool))
  out["fetchS"] = data.time - t0
  out["fetched"] = body.module_state(tool)["on_fork"] and body.tool_powered(tool)
  out["fetchTrace"] = body.swap_trace()
  if out["fetched"]:
    t0 = data.time
    out["stowWhy"] = body.run(body.stow_tool_routine(station, tool))
    out["stowS"] = data.time - t0
    out["stowTrace"] = body.swap_trace()
  st = body.module_state(tool)
  out["stowed"] = bool(out["fetched"] and st["hung"] and not st["on_fork"])
  out["falls"] = body.mission.falls
  body.close()
  return out


def served_table(n: int = 10, jobs: int = 3) -> None:
  """Fetch and stow in the served house: `n` flights from the dock and `n`
  from across the house, the tools in turn."""
  todo = [(kind, k) for kind in ("dock", "house") for k in range(n)]
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(served_one, todo, chunksize=1)
  _served_report(rows)


def _served_report(rows) -> None:
  for r in rows:
    print(f"{r['kind']:5s} {r['k']:2d} {r['tool']:11s} from {r['start']}: "
          f"fetch {'yes' if r['fetched'] else 'NO '} {r['fetchS']:5.1f} s ({r['fetchWhy']}), "
          f"stow {'yes' if r['stowed'] else 'NO '} {r.get('stowS', float('nan')):5.1f} s "
          f"({r.get('stowWhy', '-')}); falls {r['falls']}")
    for key in ("fetchTrace", "stowTrace"):
      if r.get(key) and ("->" in r[key] and "picked" not in r[key] and "hung" not in r[key]
                         or not r["fetched"] or not r["stowed"]):
        print(f"      {key}: {r[key]}")
  for kind in sorted({r["kind"] for r in rows}):
    mine = [r for r in rows if r["kind"] == kind]
    both = [r for r in mine if r["stowed"]]
    print(f"{kind}: fetched {sum(r['fetched'] for r in mine)} of {len(mine)}, "
          f"fetched and hung back {len(both)} of {len(mine)}"
          + (f"; fetch median {np.median([r['fetchS'] for r in both]):.1f} s, "
             f"stow median {np.median([r['stowS'] for r in both]):.1f} s" if both else ""))


#: The served carry (#405, stage C): the tool fetched, then a walk through
#: the house and back -- the hall, the kitchen, the garden by the east door
#: -- a trot and turns on the garden's lawn, and the stow.
CARRY_ROUTE = ((-3.5, 1.0), (-8.5, 4.0), (-3.5, 0.5), (7.5, 2.0))
CARRY_TROT = ((3.0, 0.8, 0.0, 0.0), (2.0, 0.0, 0.0, 0.8), (3.0, 0.8, 0.0, 0.0),
              (1.5, 0.0, 0.3, 0.0), (1.0, 0.0, 0.0, 0.0))


def served_carry_one(k: int) -> dict:
  """One served quadruped carries a tool through the house (`CARRY_ROUTE`,
  `CARRY_TROT`) and hangs it back: the coupling's criterion every step --
  how long it was open at worst, and whether the tool rode to the end."""
  from pluggybot.home import world as home
  from pluggybot.legs import body as qb
  from pluggybot.legs import world as lw
  from pluggybot.rack.coupling import STATION_YS
  model = lw.home_spec().compile()
  data = mujoco.MjData(model)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=home.GRID_BOUNDS)
  mis = body.mission
  body.start_at(*_served_start("dock", k, body))
  tool = tuple(rk.TOOL_BAYS)[k % len(rk.TOOL_BAYS)]
  station = STATION_YS[rk.TOOL_BAYS[tool]]
  out = {"k": k, "tool": tool, "legs": []}
  body.run(body.fetch_tool_routine(station, tool))
  out["fetched"] = body.tool_powered(tool)
  if not out["fetched"]:
    body.close()
    return out
  state = {"open": 0, "worst": 0, "steps": 0}

  def watch():
    state["steps"] += 1
    if body.tool_powered(tool):
      state["open"] = 0
    else:
      state["open"] += 1
      state["worst"] = max(state["worst"], state["open"])
  mis.step_hooks.append(watch)
  for x, y in CARRY_ROUTE:
    t0 = data.time
    arrived = body.run(body.go_to_routine(x, y, timeout=120.0))
    out["legs"].append({"to": [x, y], "arrived": bool(arrived), "s": round(data.time - t0, 1),
                        "seated": body.tool_powered(tool)})
  for seconds, vx, vy, w in CARRY_TROT:
    body.run(mis._drive_routine(seconds, 0.0, 0.0) if not (vx or vy or w) else
             _twist_for(mis, seconds, vx, vy, w))
  out["afterTrot"] = body.tool_powered(tool)
  out["carriedS"] = round(state["steps"] * model.opt.timestep, 1)
  out["worstOpenMs"] = round(state["worst"] * model.opt.timestep * 1000, 1)
  mis.step_hooks.remove(watch)
  out["stowWhy"] = body.run(body.stow_tool_routine(station, tool))
  out["stowed"] = body.module_state(tool)["hung"]
  out["falls"] = mis.falls
  body.close()
  return out


def _twist_for(mis, seconds, vx, vy, w):
  t0 = mis.data.time
  while mis.data.time - t0 < seconds:
    yield from mis._twist_routine(vx, vy, w)


def served_carry(n: int = 6, jobs: int = 3) -> None:
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(served_carry_one, range(n), chunksize=1)
  for r in rows:
    if not r["fetched"]:
      print(f"carry {r['k']} {r['tool']}: NOT FETCHED")
      continue
    legs = ", ".join(f"{'ok' if g['arrived'] else 'NO'} {g['s']:.0f} s"
                     + ("" if g["seated"] else " DROPPED") for g in r["legs"])
    print(f"carry {r['k']} {r['tool']:11s}: {r['carriedS']:.0f} s carried, worst open "
          f"{r['worstOpenMs']:.0f} ms; walks {legs}; after the trot "
          f"{'seated' if r['afterTrot'] else 'DROPPED'}; stow {r['stowWhy']} "
          f"{'hung' if r['stowed'] else 'NOT HUNG'}; falls {r['falls']}")
  ok = [r for r in rows if r["fetched"] and r["afterTrot"] and all(g["seated"] for g in r["legs"])]
  print(f"carried to the end {len(ok)} of {sum(r['fetched'] for r in rows)}; "
        f"hung back {sum(r.get('stowed', False) for r in rows)}")


def served_pair_one(k: int) -> dict:
  """The served PAIR (`pair.build_pair`, one world, one loop): each robot
  from its own start takes a tool -- the first bay A's, the second bay
  C's, 0.6 m apart -- at once, then hangs it back at once."""
  from pluggybot import tick
  from pluggybot.pair import build_pair
  from pluggybot.rack.coupling import STATION_YS
  lives = build_pair(world="home_quad", errands=("none", "none"))
  tools = (tuple(rk.TOOL_BAYS)[0], tuple(rk.TOOL_BAYS)[2])
  rng = np.random.default_rng(1405 + k)
  for life in lives:
    x, y, yaw = life.body.pose
    life.body.start_at(x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2),
                       yaw + rng.uniform(-0.5, 0.5))
  data = lives[0].data
  out = {"k": k, "tools": tools}
  t0 = data.time
  whys = tick.run_many([(life.body.stepper,
                         life.body.fetch_tool_routine(STATION_YS[rk.TOOL_BAYS[t]], t))
                        for life, t in zip(lives, tools)])
  out["fetchS"] = data.time - t0
  out["fetched"] = [life.body.module_state(t)["on_fork"] and life.body.tool_powered(t)
                    for life, t in zip(lives, tools)]
  out["fetchWhy"], out["fetchTrace"] = whys, [life.body.swap_trace() for life in lives]
  t0 = data.time
  whys = tick.run_many([(life.body.stepper,
                         life.body.stow_tool_routine(STATION_YS[rk.TOOL_BAYS[t]], t)
                         if ok else _nothing())
                        for life, t, ok in zip(lives, tools, out["fetched"])])
  out["stowS"] = data.time - t0
  out["stowed"] = [ok and life.body.module_state(t)["hung"]
                   for life, t, ok in zip(lives, tools, out["fetched"])]
  out["stowWhy"], out["stowTrace"] = whys, [life.body.swap_trace() for life in lives]
  out["falls"] = [life.body.mission.falls for life in lives]
  for life in lives:
    life.body.close()
  return out


def _nothing():
  return None
  yield


def served_pair(n: int = 5, jobs: int = 3) -> None:
  from multiprocessing import Pool
  with Pool(jobs) as pool:
    rows = pool.map(served_pair_one, range(n), chunksize=1)
  for r in rows:
    print(f"pair {r['k']}: fetched {r['fetched']} in {r['fetchS']:.1f} s ({r['fetchWhy']}), "
          f"hung back {r['stowed']} in {r['stowS']:.1f} s ({r['stowWhy']}); falls {r['falls']}")
    if not all(r["stowed"]):
      for key in ("fetchTrace", "stowTrace"):
        print(f"      {key}: {r[key]}")
  swaps = [ok for r in rows for ok in r["stowed"]]
  print(f"pair: {sum(swaps)} of {len(swaps)} swaps fetched and hung back, "
        f"{sum(all(r['stowed']) for r in rows)} of {len(rows)} flights both")


VIEWS = {"fetch": view_fetch, "carry": view_carry, "stairs": view_stairs,
         "fall": view_fall, "reach": view_reach}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--capture", action="store_true")
  ap.add_argument("--rover", action="store_true",
                  help="with --capture: the rover's fork and trays (the premise)")
  ap.add_argument("--loose", action="store_true",
                  help="with --capture: the stops' play at the rover's 4 mm")
  ap.add_argument("--narrow", action="store_true",
                  help="with --capture: the trays at +-35 mm (the premise)")
  ap.add_argument("--jobs", type=int, default=3)
  ap.add_argument("--retention", action="store_true")
  ap.add_argument("--stairs", action="store_true", help="with --retention")
  ap.add_argument("--first", action="store_true",
                  help="with --retention --stairs: the fork as first built, its V's "
                       "and ramps at 45 deg (the premise)")
  ap.add_argument("--fall", action="store_true", help="with --retention")
  ap.add_argument("--getup", action="store_true")
  ap.add_argument("--sensors", action="store_true")
  ap.add_argument("--approach", action="store_true")
  ap.add_argument("--reach", action="store_true")
  ap.add_argument("--envelope", action="store_true")
  ap.add_argument("--served", action="store_true",
                  help="the served quadruped fetching and stowing at the house's rack "
                       "(#405): --n flights from the dock and --n from across the house; "
                       "with --pair, the served pair swapping at once")
  ap.add_argument("--pair", action="store_true", help="with --served")
  ap.add_argument("--carry", action="store_true",
                  help="with --served: a tool carried through the house, trotted and "
                       "turned, then hung back -- the coupling's criterion every step")
  ap.add_argument("--view", nargs="?", const="fetch", choices=tuple(VIEWS),
                  help="watch a scene in the MuJoCo viewer (default: fetch) until "
                       "the window closes")
  ap.add_argument("--n", type=int, default=20,
                  help="--approach: random starts; --retention --stairs: starts per row")
  ap.add_argument("--out", default="arm_spike.png")
  args = ap.parse_args(argv)
  if args.capture:
    capture_table("rover" if args.rover else "loose" if args.loose
                  else "narrow" if args.narrow else "", args.jobs)
  elif args.retention and args.stairs:
    retention_stairs(n=args.n, jobs=args.jobs, fork=FIRST_FORK if args.first else None)
  elif args.retention and args.fall:
    retention_fall(reflex=True)
    retention_fall(reflex=False)
  elif args.retention:
    retention_flat()
  elif args.getup:
    getup_table(jobs=args.jobs)
  elif args.sensors:
    sensors_table()
  elif args.approach:
    approach_table(args.n, args.jobs)
  elif args.reach:
    reach_table()
  elif args.envelope:
    envelope_table(args.jobs)
  elif args.served and args.carry:
    served_carry(args.n, args.jobs)
  elif args.served and args.pair:
    served_pair(args.n, args.jobs)
  elif args.served:
    served_table(args.n, args.jobs)
  elif args.view:
    VIEWS[args.view]()
  else:
    filmstrip(args.out)


if __name__ == "__main__":
  sys.exit(main())
