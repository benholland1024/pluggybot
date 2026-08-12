"""The claw module's pick-and-place controller (milestone 8).

The fourth tool. Like the pen module it brings the robot a capability the base
does not have -- here a grip -- and like the pen module the robot supplies the
axis it already owns: the LIFT is the claw's up and down, so the claw itself
only opens and closes.

The shape of this tool was decided by measurement before any of it was drawn
(docs/SimNotes.md):

  REACH DOWN, NEVER FORWARD. The gravity latch takes ~0.45 N*m of pitch moment
  before the peg rides out of its V-notch. A payload at forward offset L costs
  W*L, so 800 g hangs happily on the peg's own axis while 400 g unseats the
  module at 150 mm out. The claw is therefore a pendant straight down the peg
  axis, not the angled arm the idea started as.

  THE CHASSIS WAS NEVER THE LIMIT. 800 g at the peg drops drive-wheel load
  only 15.0 -> 12.6 N; static forward tipping needs about 5 kg. The centre of
  mass was the obvious worry and turned out to be the wrong one.

  AND THE ROBOT CANNOT SEE WHAT IT IS PICKING UP. The nav camera sits 180 mm
  up with a 41 deg fovy, so the floor leaves its view inside **0.48 m** -- and
  the grip point is 235 mm ahead of the axle. The object is invisible at the
  moment of grasping, exactly as the socket is invisible at the moment of
  docking. A real pickup must memorise the target's pose while it can still
  see it and then run open-loop, which is what this does.
"""

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.control import wheel_targets, wrap_angle
from pluggybot.hub.coupling import CLAW_JAW_TRAVEL
from pluggybot.hub.swap import ARM_EXT, PLUG_LATERAL, VERTEX_AHEAD_OF_AXLE

CLAW_MODULE = "module_claw"
JAW_GEOMS = ("module_claw_pad_l", "module_claw_pad_r")
APPROACH_LIFT = 0.128     # the swap preset: jaws ride ~126 mm up, clear to drive
CARRY_LIFT = 0.160        # carrying height. Was 0.090 "clear of the floor,
                          # low CoM" -- and it was not clear at all: that put
                          # the block only 46 mm up, while a 172 mm pendant
                          # swinging the 10 deg a carried module was measured
                          # to swing moves its tip ~30 mm. The block struck
                          # the floor during the turn and was knocked out of
                          # the jaws. A pendant tool needs carry clearance
                          # sized by its SWING, not by its static height.
                          # An apparent lift CEILING at ~0.193 (the block let
                          # go with no contact but the pads) turned out to be
                          # the same contact creep, now cured by the pads' hard
                          # solimp: the higher the lift, the longer the grip
                          # was held and the further the block had crept. Not a
                          # ceiling at all -- a clock.
JAW_HALF_H = 0.020        # pad half-height (mirrors coupling.CLAW_PAD_HALF_H)
FLOOR_CLEARANCE = 0.002   # pad bottoms above the floor when closing. The grip
                          # target is set by the JAWS, not the object: aiming
                          # the pad CENTRE at the object's mid-height drove the
                          # 40 mm pads 11 mm into the floor, the module simply
                          # rode up in its fork (it is only gravity-latched),
                          # and the lift came up 12.8 mm short of command with
                          # nothing gripped.
GRIP_Z = JAW_HALF_H + FLOOR_CLEARANCE
APPROACH_RUNWAY = 0.45    # m of straight line before the grip pose. The
                          # lateral P-controller needs a runway to converge,
                          # and arriving beside the object instead means
                          # turning on the spot right next to it.
JAW_SPEED = 0.030         # m/s ceiling on jaw SETPOINT motion
LIFT_SPEED = 0.05         # m/s ceiling on lift SETPOINT motion (lead-screw
                          # class). See set_lift: a step command unseats the
                          # gravity-latched module.
SETTLE = 0.6


class ClawTool:
  """Drives the claw module through a pick-and-place."""

  def __init__(self, model, data, swap) -> None:
    self.model, self.data, self.swap = model, data, swap
    self.lift_act = model.actuator("lift").id
    self.arm_act = model.actuator("arm").id
    self.jaw_acts = [model.actuator("claw_l").id, model.actuator("claw_r").id]
    self.grip_site = model.site("claw_grip").id
    self._jaw_gids = {model.geom(g).id for g in JAW_GEOMS}
    self.offset = (VERTEX_AHEAD_OF_AXLE, -PLUG_LATERAL)   # nominal until measured

  def grasp_physics(self, on: bool = True) -> None:
    """Deprecated no-op, kept so existing callers keep working.

    Grip creep once needed MuJoCo's `noslip` post-solve pass, toggled ON for
    holding and OFF for coupling -- two solver policies and a genuine
    footgun. It is fixed at the source now: the jaw pads carry a HARD
    contact constraint (`coupling.GRIP_SOLIMP`), which removes the drift
    with no solver mode at all. Measured slip over a 100 mm lift:
    -21.7 mm soft, -0.13 mm hard, against +0.28 mm for the noslip cure.
    One policy everywhere, and ~3x faster.
    """

  def calibrate(self) -> tuple[float, float]:
    """Measure where the grip point actually sits relative to the axle.

    The nominal figure (fork vertex + the fork's lateral offset) was 12.5 mm
    out along track, because the module hangs a few mm ahead of the vertex and
    leans on the pad. Same story as the pen tip, which was 72 mm below the peg
    against a 47 mm estimate: nominal geometry never includes droop or lean.

    On hardware this is a one-off measurement against a jig, not a sensor --
    exactly the open item flagged for the plotter's pen-tip calibration.
    """
    g = self.grip_world()
    th = self.swap.reckoner.theta
    dx = float(g[0]) - self.swap.reckoner.x
    dy = float(g[1]) - self.swap.reckoner.y
    self.offset = (dx * math.cos(th) + dy * math.sin(th),
                   -dx * math.sin(th) + dy * math.cos(th))
    return self.offset

  # ---- state ---------------------------------------------------------------

  def grip_world(self) -> np.ndarray:
    return np.array(self.data.site_xpos[self.grip_site], dtype=float)

  def holding(self, target: str = "pickup_box") -> bool:
    """BOTH pads touching the object. The grip's own contact criterion --
    the same reasoning as the coupling's electrical one and the plug's
    charging one: a commanded grip is a belief, contact is a fact."""
    gid = self.model.geom(target).id
    touching = set()
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      pair = {c.geom1, c.geom2}
      if gid in pair:
        touching |= self._jaw_gids & pair
    return len(touching) == 2

  # ---- primitives ----------------------------------------------------------

  def jaws(self, opening: float, settle: float = SETTLE,
           speed: float = JAW_SPEED) -> None:
    """opening 0 = wide, 1 = fully closed. Ramped, like every other position
    setpoint on this robot.

    Third axis to need this and the most surprising: slamming the jaws shut
    in one command batted the block out of the gripper before contact could
    settle -- the closure ran to its full 27 mm stop on a 26 mm object that
    should have halted it at 18 mm, while a graze on the way past still
    registered as "both pads in contact". A servo has a closing speed.
    """
    target = -CLAW_JAW_TRAVEL * float(np.clip(opening, 0, 1))
    cur = float(self.data.ctrl[self.jaw_acts[0]])
    steps = max(int(abs(target - cur) / speed / self.model.opt.timestep), 1)
    for k in range(steps):
      for act in self.jaw_acts:
        self.data.ctrl[act] = cur + (target - cur) * (k + 1) / steps
      self.swap._step_once(0.0, 0.0)
    self.swap._run(settle, 0.0)

  def set_lift(self, target: float, settle: float = 1.2,
               speed: float = LIFT_SPEED) -> None:
    """Walk the lift setpoint at a bounded speed -- never write it across a
    gap. A stiff position servo handed a step delivers the whole difference
    as an impulse: raising to carry height in one command (184 mm) threw the
    module clean off the fork, both electrical poles open. Exactly the
    lesson the pen carriage taught, on a different axis. A lead screw has a
    top speed anyway.
    """
    target = float(np.clip(target, 0.0, 0.31))
    cur = float(self.data.ctrl[self.lift_act])
    steps = max(int(abs(target - cur) / speed / self.model.opt.timestep), 1)
    for k in range(steps):
      self.data.ctrl[self.lift_act] = cur + (target - cur) * (k + 1) / steps
      self.swap._step_once(0.0, 0.0)
    self.swap._run(settle, 0.0)

  def lower_grip_to(self, world_z: float, settle: float = 1.5) -> float:
    """Put the jaw centre at a world height, by measured correction.

    The lift-to-grip offset includes RCC droop and the module's lean, so it
    is measured rather than derived -- the pen module's 47 mm estimate was
    really 72 mm for exactly that reason.

    Iterated, not one-shot: the grip follows the lift command at only about
    0.87:1, because droop and the module's lean both change as the arm goes
    down. A single correction left the pads 11.6 mm high, which still LOOKED
    like a grip -- both pads touched -- but held the block by its top 12 mm
    and it slipped out the moment the lift took its weight. Converging the
    height is what makes the grasp deep enough to survive lifting.
    """
    for _ in range(5):
      now = float(self.data.ctrl[self.lift_act])
      err = world_z - float(self.grip_world()[2])
      if abs(err) < 0.0015:
        break
      self.set_lift(now + err, settle=settle)
    return float(self.grip_world()[2]) - world_z

  def _face(self, heading: float, tol: float = 0.004, tries: int = 6) -> float:
    """Settle-and-recheck, as the plotter needed: a P-controller that stops
    commanding at the target coasts past, because `slew` rate-limits the
    wheel command."""
    for _ in range(tries):
      while abs(wrap_angle(heading - self.swap.reckoner.theta)) > tol:
        err = wrap_angle(heading - self.swap.reckoner.theta)
        tl, tr = wheel_targets(0.0, max(-0.5, min(0.5, 1.2 * err)))
        self.swap._step_once(tl, tr)
      self.swap._run(SETTLE, 0.0)
      if abs(wrap_angle(heading - self.swap.reckoner.theta)) <= tol * 4:
        break
    return wrap_angle(heading - self.swap.reckoner.theta)

  # ---- driving -------------------------------------------------------------

  def axle_pose_for(self, obj_xy, heading: float) -> tuple[float, float]:
    """Where the axle must sit for the grip point to land on obj_xy.

    The grip hangs on the peg axis: VERTEX_AHEAD_OF_AXLE forward and
    PLUG_LATERAL right of the chassis centreline, because the claw inherits
    the fork's lateral offset exactly as the pen did.
    """
    ox, oy = obj_xy
    fwd, lat = self.offset
    fx, fy = math.cos(heading), math.sin(heading)
    ax = ox - fwd * fx + lat * fy
    ay = oy - fwd * fy - lat * fx
    return ax, ay

  def drive_over(self, obj_xy, heading: float, timeout: float = 45.0) -> bool:
    """Approach until the grip point is above the object, arm stowed.

    Stowed for the drive -- the rack taught that one, when an extended fork
    swept a module off its trays -- then deployed once lined up.
    """
    tx, ty = self.axle_pose_for(obj_xy, heading)
    self.set_lift(APPROACH_LIFT, settle=0.5)
    self.data.ctrl[self.arm_act] = 0.0
    self.swap._run(1.0, 0.0)
    t0 = self.data.time

    # STAGE 1: go to a staging point back along the approach line, and only
    # then turn onto the heading. Without this the robot arrives beside the
    # object -- the pick leaves it ~0.19 m away facing ~180 deg wrong -- so
    # `drive_toward` pirouettes on the spot next to the block, and every
    # refine retry backed up and pirouetted again. Four passes around the
    # thing it was trying to pick up. Lining up first and driving straight in
    # is both more repeatable and far easier to watch.
    sx = tx - APPROACH_RUNWAY * math.cos(heading)
    sy = ty - APPROACH_RUNWAY * math.sin(heading)
    while self.data.time - t0 < timeout:
      pose = (self.swap.reckoner.x, self.swap.reckoner.y,
              self.swap.reckoner.theta)
      if math.hypot(sx - pose[0], sy - pose[1]) < 0.03:
        break
      v, w = drive_toward(pose, (sx, sy))
      tl, tr = wheel_targets(v, w)
      self.swap._step_once(tl, tr)
    self._face(heading)

    def run_in():
      while self.data.time - t0 < timeout:
        pose = (self.swap.reckoner.x, self.swap.reckoner.y,
                self.swap.reckoner.theta)
        if math.hypot(tx - pose[0], ty - pose[1]) < 0.010:
          return
        v, w = drive_toward(pose, (tx, ty))
        tl, tr = wheel_targets(v, w)
        self.swap._step_once(tl, tr)

    run_in()
    self._face(heading)
    # Back up and take another run at it. drive_toward's arrival radius is
    # happy to stop a couple of cm off the line, and the open jaws only span
    # +/-22 mm around a 26 mm object -- the first approach measured 26 mm
    # lateral, just outside capture. The P-controller converges laterally
    # given a runway, which is the fix refine_standoff already uses at the
    # rack. There is no lateral DoF to trim with here: the claw hangs fixed,
    # so driving is the only correction available.
    for _ in range(3):
      dx, dy = tx - self.swap.reckoner.x, ty - self.swap.reckoner.y
      lat = -dx * math.sin(heading) + dy * math.cos(heading)
      if abs(lat) < 0.006:
        break
      self.swap._drive_until(0.30, -0.10, stall_stop=False)
      run_in()
      self._face(heading)

    # Null the remaining along-track error by creeping.
    dx, dy = tx - self.swap.reckoner.x, ty - self.swap.reckoner.y
    along = dx * math.cos(heading) + dy * math.sin(heading)
    if abs(along) > 0.003:
      self.swap._drive_until(abs(along), 0.04 * (1 if along > 0 else -1),
                             stall_stop=False)
    self.swap._run(SETTLE, 0.0)

    self.data.ctrl[self.arm_act] = ARM_EXT
    self.swap._run(1.5, 0.0)
    return math.hypot(tx - self.swap.reckoner.x,
                      ty - self.swap.reckoner.y) < 0.03

  # ---- the verbs -----------------------------------------------------------

  def pick_up(self) -> dict:
    """Open, drop astride the object, close, lift.

    Grasp physics stays ON from here until set_down(), because the carry in
    between is exactly the sustained hold that needs it.
    """
    self.jaws(0.0)
    residual = self.lower_grip_to(GRIP_Z)
    self.jaws(1.0, settle=1.2)
    gripped = self.holding()
    self.set_lift(CARRY_LIFT, settle=1.5)
    return {"gripped_before_lift": gripped, "holding": self.holding(),
            "lower_residual_mm": residual * 1000}

  def set_down(self) -> dict:
    """Lower until the object is back on the floor, release, retreat upward."""
    self.lower_grip_to(GRIP_Z)
    self.jaws(0.0, settle=1.0)
    self.set_lift(APPROACH_LIFT, settle=1.5)
    return {"released": not self.holding()}
