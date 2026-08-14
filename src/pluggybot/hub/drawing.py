"""The drawing tool's plot controller (milestone 8).

An X-Y plotter assembled out of parts that already exist, plus the one the
robot never had:

  Y (horizontal)  the PEN MODULE's own carriage, along the peg axis. This is
                  the axis the robot structurally lacks -- a differential
                  drive cannot translate sideways -- and the whole argument
                  for the tool being a kinematic extension rather than cargo.
  Z (vertical)    the robot's existing lift.
  pen up/down     the arm's reach, which also sets pen PRESSURE. The lean-pad
                  is what makes that pressure possible at all: without it a
                  peg-hung module gives out at 0.1 N and flops over at 0.5 N
                  (docs/SimNotes.md).

The base is parked throughout. That is deliberate: wheel odometry has broken
something in this project four separate times, and a plotter that integrates
drift into its drawing would make the fifth. Everything that moves during a
drawing is a position-controlled joint on a lead screw.

Calibration is measured, not derived. `calibrate()` drives each axis to two
known commands and reads where the pen tip actually went, which gives the
board<->joint mapping including sign, scale and offset. Deriving it from the
model would bake in exactly the kind of transform error that made
`module_state` report every correct placement as a failure for weeks.
"""

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.control import turn_command, wheel_targets, wrap_angle
from pluggybot.hub.coupling import (
  BOARD_HALF, BOARD_X, BOARD_Y, BOARD_Z, PEN_TRAVEL,
)
from pluggybot.hub.swap import (
  ARM_EXT, DROOP_COMP, FORK_MOUNT_RAISE, PLUG_LATERAL,
)

PEN_MODULE = "module_pen"
DRAW_SPEED = 0.020        # m/s along the drawn path. The lift carries the
                          # whole arm assembly and lagged 5-7 mm at 35 mm/s;
                          # a plotter has no reason to hurry.
CARRIAGE_SPEED = 0.030    # m/s ceiling on SETPOINT motion, not just on the
                          # joint. A lead screw has a top speed, and a stiff
                          # position servo handed a step command delivers the
                          # whole gap as an impulse: commanding the carriage
                          # from +40 mm to -40 mm in one go threw the module
                          # clean out of the fork's V-notches (both electrical
                          # poles opened and it yawed ~100 deg), while walking
                          # the same 80 mm in 5 mm increments held seated the
                          # entire way. The wheels have had `control.slew` for
                          # exactly this reason since milestone 1; a position
                          # setpoint needs it just as much.
PRESS_STEP = 0.002        # m of extra arm reach per press probe
PRESS_EXTRA = 0.010       # m of arm reach past first contact. With the soft
                          # quill (60 N/m) that is ~0.6 N of pen force -- the
                          # SPRING sets pressure, not the arm's position
                          # accuracy. Deep nominal press is the point: it
                          # leaves ~10 mm of travel in hand to absorb the
                          # droop that grows as the lift rises. Without the
                          # quill the same reach would be 12 N through the
                          # arm's own 1200 N/m servo.
PRESS_MAX = 0.19          # m of arm extension before giving up on the board
LIFT_MIN, LIFT_MAX = 0.02, 0.30
BOARD_STANDOFF = 0.34     # m axle-to-board. MEASURED, not chosen: the pen tip
                          # rides 0.283 m ahead of the axle at the stowed-work
                          # extension, so a 0.42 m standoff left it 137 mm
                          # short with the arm at full stretch. This parks the
                          # press in the middle of the arm's travel.
PEN_BELOW_PEG = 0.072     # m the pen tip hangs below the peg line (measured).
                          # Estimating it from the model gave 47 mm -- the
                          # difference is the module's lean onto the lean-pad
                          # plus RCC droop, i.e. exactly the terms that are
                          # never in the nominal geometry.


def pen_on_board(model, data) -> bool:
  """Is the pen actually touching the board? The drawing equivalent of the
  electrical criterion: a commanded path is a belief, ink is a fact."""
  try:
    shaft = model.geom(f"{PEN_MODULE}_shaft").id
    board = model.geom("board").id
  except KeyError:
    return False
  for i in range(data.ncon):
    pair = {data.contact[i].geom1, data.contact[i].geom2}
    if shaft in pair and board in pair:
      return True
  return False


def square_path(size: float = 0.075, n: int = 240) -> list[tuple[float, float]]:
  """A closed square in board coordinates, centred on the origin.

  Chosen over a circle for the FIRST figure because each edge moves one axis
  alone: an error can be attributed to the carriage or to the lift rather
  than to 'the plotter'. The corners are where following lag shows up.
  """
  h = size / 2
  corners = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]
  pts = []
  for a, b in zip(corners, corners[1:]):
    for k in range(n // 4):
      f = k / (n // 4)
      pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
  pts.append(corners[-1])
  return pts


def circle_path(size: float = 0.075, n: int = 240) -> list[tuple[float, float]]:
  """A closed circle: every sample moves BOTH axes, so axis scale mismatch
  and following lag show up as shape distortion rather than as a number."""
  r = size / 2
  return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n))
          for k in range(n + 1)]


PATHS = {"square": square_path, "circle": circle_path}


class PenPlotter:
  """Drives the pen module against the board and records what it drew."""

  def __init__(self, model, data, swap) -> None:
    self.model, self.data, self.swap = model, data, swap
    self.pen_act = model.actuator("pen_carriage").id
    self.lift_act = model.actuator("lift").id
    self.arm_act = model.actuator("arm").id
    self.pen_site = model.site("pen_tip").id
    self.cal: dict = {}
    # (t, board_y, board_z, commanded_y, commanded_z, touching)
    self.trace: list[tuple[float, float, float, float, float, bool]] = []

  # ---- geometry ------------------------------------------------------------

  def pen_world(self) -> np.ndarray:
    return np.array(self.data.site_xpos[self.pen_site], dtype=float)

  def pen_board(self) -> tuple[float, float]:
    """Pen tip projected onto the board plane: (y, z) in board coordinates."""
    p = self.pen_world()
    return float(p[1] - BOARD_Y), float(p[2] - BOARD_Z)

  # ---- placement -----------------------------------------------------------

  def board_standoff(self, standoff: float = BOARD_STANDOFF) -> tuple[float, float]:
    """Axle pose the plotter works from: square to the board, and offset so
    the FORK line -- not the chassis centreline -- lands on the board's
    centre."""
    return BOARD_X - BOARD_HALF[0] - standoff, BOARD_Y + PLUG_LATERAL

  def drive_to_board(self, standoff: float = BOARD_STANDOFF,
                     timeout: float = 40.0) -> bool:
    """Carry the tool to the board and square up to it.

    This DRIVES rather than teleporting, and not for realism points: the
    module is a free body held only by gravity in the fork's V, so writing
    the robot's qpos leaves the tool behind in mid-air and it falls to the
    floor (measured -- the first version calibrated a pen lying on the
    ground, and the giveaway was dz/dlift coming back exactly 0.000).

    The arm is stowed for the drive, per the rule the rack taught: stowed is
    the driving configuration.
    """
    tx, ty = self.board_standoff(standoff)
    self.data.ctrl[self.arm_act] = 0.0
    self.swap._run(1.0, 0.0)
    t0 = self.data.time
    while self.data.time - t0 < timeout:
      pose = (self.swap.reckoner.x, self.swap.reckoner.y,
              self.swap.reckoner.theta)
      if math.hypot(tx - pose[0], ty - pose[1]) < 0.03:
        break
      v, w = drive_toward(pose, (tx, ty))
      tl, tr = wheel_targets(v, w)
      self.swap._step_once(tl, tr)
    self._face(0.0)                # board faces -x, so the robot heads +x
    lift0 = BOARD_Z + PEN_BELOW_PEG - 0.145 - FORK_MOUNT_RAISE + DROOP_COMP
    self.data.ctrl[self.lift_act] = lift0
    self.data.ctrl[self.arm_act] = ARM_EXT
    self.swap._run(2.0, 0.0)
    return math.hypot(tx - self.swap.reckoner.x,
                      ty - self.swap.reckoner.y) < 0.06

  def ramp(self, act: int, target: float, speed: float = CARRIAGE_SPEED,
           settle: float = 0.0) -> None:
    """Walk an actuator's SETPOINT to a target at a bounded speed.

    Never write a position setpoint directly across a gap -- see
    CARRIAGE_SPEED. This is `control.slew` for position actuators.
    """
    cur = float(self.data.ctrl[act])
    steps = max(int(abs(target - cur) / speed / self.model.opt.timestep), 1)
    for k in range(steps):
      self.data.ctrl[act] = cur + (target - cur) * (k + 1) / steps
      self.swap._step_once(0.0, 0.0)
    if settle:
      self.swap._run(settle, 0.0)

  def _face(self, heading: float, tol: float = 0.004,
            tries: int = 6) -> float:
    """Square up to a heading, then STOP AND CHECK -- repeatedly.

    A P-controller that simply stops commanding when the error goes small
    overshoots here, badly: `slew` rate-limits the wheel velocity command, so
    the wheels are still turning when the loop exits and the robot coasts
    past. Measured on the approach to the board: -9.52 deg in, +7.47 deg out,
    a 17 deg overshoot that the caller had no way to notice.

    Squareness matters more for drawing than for anything before it. The
    carriage sweeps 110 mm across the board, so a yaw error theta swings the
    pen's depth by 110*sin(theta) -- 14 mm at 7.5 deg, which is most of the
    quill's whole travel and is why one edge of the figure came out blank.

    Settle-and-recheck is the same shape as refine_standoff's back-up-and-
    try-again, one axis over. Returns the final error in radians.
    """
    for _ in range(tries):
      while abs(wrap_angle(heading - self.swap.reckoner.theta)) > tol:
        err = wrap_angle(heading - self.swap.reckoner.theta)
        tl, tr = wheel_targets(0.0, turn_command(err))
        self.swap._step_once(tl, tr)
      self.swap._run(0.6, 0.0)              # brake, let the slew unwind
      if abs(wrap_angle(heading - self.swap.reckoner.theta)) <= tol * 4:
        break
    return wrap_angle(heading - self.swap.reckoner.theta)

  def press(self) -> bool:
    """Extend the arm until the pen touches, then a little more.

    The same depth-referencing-by-gentle-press the docking and coupling work
    both settled on: drive until the sensor says contact, rather than to a
    believed distance. Returns whether the board was found.
    """
    ext = float(self.data.ctrl[self.arm_act])
    while ext < PRESS_MAX:
      ext += PRESS_STEP
      self.data.ctrl[self.arm_act] = ext
      self.swap._run(0.12, 0.0)
      if pen_on_board(self.model, self.data):
        self.data.ctrl[self.arm_act] = min(ext + PRESS_EXTRA, PRESS_MAX)
        self.swap._run(0.6, 0.0)
        return True
    return False

  def lift_pen(self, by: float = 0.02) -> None:
    self.data.ctrl[self.arm_act] = max(
      float(self.data.ctrl[self.arm_act]) - by, 0.0)
    self.swap._run(0.5, 0.0)

  # ---- calibration ---------------------------------------------------------

  def calibrate(self) -> dict:
    """Two-point calibration per axis: command, settle, read the pen tip.

    Yields board = origin + gain * command, per axis, with sign and scale
    measured rather than reasoned from the model's frames.
    """
    lift0 = float(self.data.ctrl[self.lift_act])
    self.ramp(self.pen_act, 0.0, settle=1.0)
    y0, z0 = self.pen_board()

    probe = PEN_TRAVEL * 0.6
    self.ramp(self.pen_act, probe, settle=1.0)
    y1, _ = self.pen_board()
    self.ramp(self.pen_act, 0.0, settle=0.5)

    dl = 0.04
    self.ramp(self.lift_act, lift0 + dl, settle=1.0)
    _, z1 = self.pen_board()
    self.ramp(self.lift_act, lift0, settle=1.0)

    self.cal = {
      "y0": y0, "z0": z0, "lift0": lift0,
      "dy_dcarriage": (y1 - y0) / probe,
      "dz_dlift": (z1 - z0) / dl,
    }
    return self.cal

  def calibrate_loaded(self, extent: float = PEN_TRAVEL * 0.6) -> dict:
    """Re-measure the carriage gain WITH THE PEN ON THE BOARD.

    The free-air calibration is a claim about a condition the tool never
    actually works in. What it gets wrong is the ORIGIN, not the scale:
    measured, pressing the pen moves its home position **10 mm** sideways
    (the press deflects the module and the compliant wrist), while the
    quasi-static gain barely budges, 0.999 -> 0.992. Re-zeroing there is
    what took a drawn circle from 9.2 mm to 2.2 mm of shape error.

    ⚠ Do NOT read this as correcting the scale. Measured separately, the pen
    tracks the carriage at only ~0.81:1 *while sweeping* with the pen down,
    against ~1:1 measured at rest -- that loss is kinetic (drag reverses with
    direction and yaws the module as it goes), so a calibration that settles
    before each reading cannot see it by construction. It is the largest
    remaining error, and it wants either a stiffer yaw constraint on the
    module or a sweep-while-measuring fit. Characterised, not fixed.

    `extent` is the figure's own half-width. It matters because the loss is
    not linear in carriage position, so a fit over 60 % of travel
    extrapolated to a square's corners made things WORSE than no loaded pass
    at all (8.2 -> 12.7 mm) while helping a circle. Fitting the SECANT across
    exactly the span the figure will use is the honest linear approximation.
    """
    self.ramp(self.pen_act, 0.0, settle=0.8)
    if not self.press():
      return self.cal
    probe = min(abs(extent), PEN_TRAVEL * 0.95)
    self.ramp(self.pen_act, -probe, settle=0.8)
    y_lo, _ = self.pen_board()
    self.ramp(self.pen_act, probe, settle=0.8)
    y_hi, _ = self.pen_board()
    self.ramp(self.pen_act, 0.0, settle=0.8)
    y_mid, _ = self.pen_board()
    self.cal["dy_dcarriage"] = (y_hi - y_lo) / (2 * probe)
    self.cal["y0"] = y_mid
    self.cal["loaded"] = True
    self.lift_pen()
    return self.cal

  def targets_for(self, by: float, bz: float) -> tuple[float, float]:
    c = self.cal
    carriage = (by - c["y0"]) / c["dy_dcarriage"]
    lift = c["lift0"] + (bz - c["z0"]) / c["dz_dlift"]
    return (float(np.clip(carriage, -PEN_TRAVEL, PEN_TRAVEL)),
            float(np.clip(lift, LIFT_MIN, LIFT_MAX)))

  # ---- drawing -------------------------------------------------------------

  def contact_physics(self, on: bool = True) -> None:
    """Deprecated no-op, kept so existing callers keep working.

    The plotter once toggled MuJoCo's `noslip` post-solve pass on for a
    drawing and off afterwards -- the last per-phase solver toggle in the
    repo, and PluggyWorld's shared two-robot world cannot phase-scope solver
    settings per robot. The spike that settled it (scripts/noslip_spike.py;
    table in docs/SimNotes.md) found the pass was curing the wrong thing:
    what moves under sustained pen drag is the PARKED BASE, and mostly it is
    not solver drift at all -- the wheels ROLL, because a velocity servo
    commanded 0 resists speed, not force. The wheel joints now carry
    `frictionloss` (pluggybot_fork.xml), the parking brake the physical
    gearbox provides for free, which draws a BETTER figure than the pass
    ever did (99 % vs 93 % inked, form 0.60 vs 0.61 mm on the square) with
    no solver mode, no toggle, and none of the 2x step cost.
    `noslip_iterations` is 0 everywhere, guarded by
    tests/test_noslip_policy.py -- and always-on noslip was measured WORSE
    than useless: at >=1 iteration the jittered coupling half-seats (on the
    fork, not powered), because the peg seats by sliding.
    """

  def draw(self, path: list[tuple[float, float]]) -> dict:
    """Follow a board-space path, recording commanded vs actual every step."""
    if not self.cal:
      self.calibrate()
    # Fit the loaded gain across the span this figure actually uses.
    half = max(abs(py) for py, _ in path) or PEN_TRAVEL * 0.6
    self.calibrate_loaded(half / abs(self.cal["dy_dcarriage"]))
    # Centre the figure on where the pen ACTUALLY is, not on the board's
    # middle. The base parks to a few cm and the pen inherits the fork's
    # lateral offset, so the pen's home sat 23 mm off board centre -- and a
    # figure drawn about board centre ran straight off the end of the
    # carriage's +/-55 mm travel, which showed up as 41 mm of "lag" on an
    # edge where the carriage was not asked to move at all. A machine homes
    # and then works in its own coordinates; so does this.
    oy, oz = self.cal["y0"], self.cal["z0"]
    path = [(py + oy, pz + oz) for py, pz in path]
    self.commanded = path
    # Move to the start with the pen clear, then press: dragging the pen to
    # the start would draw a line that is not part of the figure.
    self.lift_pen()
    c0, l0 = self.targets_for(*path[0])
    self.ramp(self.pen_act, c0)
    self.ramp(self.lift_act, l0, settle=1.0)
    if not self.press():
      return {"drew": False, "reason": "never reached the board"}

    ts = self.model.opt.timestep
    for (ay, az), (by, bz) in zip(path, path[1:]):
      seg = math.hypot(by - ay, bz - az)
      steps = max(int(seg / DRAW_SPEED / ts), 1)
      for k in range(steps):
        f = (k + 1) / steps
        cy, cz = ay + (by - ay) * f, az + (bz - az) * f
        carriage, lift = self.targets_for(cy, cz)
        self.data.ctrl[self.pen_act] = carriage
        self.data.ctrl[self.lift_act] = lift
        self.swap._step_once(0.0, 0.0)
        py, pz = self.pen_board()
        self.trace.append((float(self.data.time), py, pz, cy, cz,
                           pen_on_board(self.model, self.data)))
    self.lift_pen()
    return {"drew": True, **self.error_stats()}

  def error_stats(self) -> dict:
    """Two different questions, and they deserve two different numbers.

    `track` is the instantaneous error: how far the pen was from where it was
    supposed to be AT THAT MOMENT. It includes servo following lag.

    `shape` is the distance from each inked point to the NEAREST point on the
    commanded figure. Lag along the path direction does not count against it.
    That is the honest measure of a drawing, because a square traced a
    fraction of a second behind schedule is still a square -- nobody looking
    at the board can tell. Reporting only the first would condemn a perfect
    figure for being late.
    """
    inked = [t for t in self.trace if t[5]]
    if not inked or not getattr(self, "commanded", None):
      return {"inked_fraction": 0.0, "track_rms_mm": None, "shape_rms_mm": None}
    track = [math.hypot(py - cy, pz - cz) for _, py, pz, cy, cz, _ in inked]

    pts = np.array([[t[1], t[2]] for t in inked])
    poly = np.array(self.commanded)
    shape, _ = self._nearest(pts, poly)

    # Decompose: is the figure the WRONG SHAPE, or the right shape in the
    # wrong PLACE? Those are different problems with different fixes -- a
    # rigid offset is a calibration constant, distortion is mechanics -- and
    # lumping them together hides which one you have. (Ben spotted this
    # looking at draw.png: "it drew a great square, it was just to the right
    # of where we wanted it." The same decompose-before-you-fix lesson as the
    # 33 deg standoff miss, which turned out to be 0.02 deg of odometry and
    # all estimator.)
    offset = np.zeros(2)
    for _ in range(6):                       # translation-only ICP
      _, proj_i = self._nearest(pts + offset, poly)
      step = (proj_i - (pts + offset)).mean(axis=0)
      offset = offset + step
      if np.linalg.norm(step) < 1e-6:
        break
    form, _ = self._nearest(pts + offset, poly)

    return {
      "inked_fraction": len(inked) / len(self.trace),
      "track_rms_mm": math.sqrt(sum(e * e for e in track) / len(track)) * 1000,
      "track_max_mm": max(track) * 1000,
      "shape_rms_mm": float(np.sqrt((shape ** 2).mean()) * 1000),
      "shape_max_mm": float(shape.max() * 1000),
      "offset_mm": float(np.linalg.norm(offset) * 1000),
      "offset_y_mm": float(-offset[0] * 1000),
      "offset_z_mm": float(-offset[1] * 1000),
      "form_rms_mm": float(np.sqrt((form ** 2).mean()) * 1000),
      "form_max_mm": float(form.max() * 1000),
      "samples": len(self.trace),
    }

  @staticmethod
  def _nearest(pts: np.ndarray, poly: np.ndarray):
    """(distance, closest point on the polyline) for each point."""
    a, b = poly[:-1], poly[1:]
    seg = b - a
    denom = np.maximum((seg * seg).sum(1), 1e-12)
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip((rel * seg[None, :, :]).sum(2) / denom[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    d = np.sqrt(((pts[:, None, :] - proj) ** 2).sum(2))
    k = d.argmin(1)
    return d.min(1), proj[np.arange(len(pts)), k]
