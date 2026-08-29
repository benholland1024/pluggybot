"""The seed dispenser's sowing controller (the fifth tool).

The first tool built AGAINST `docs/ToolPattern.md` instead of being mined for
it, and the doc's opening question did most of the design work: *which axis
does this tool bring, and which does it borrow?*

  the tool brings   a DISCRETE RELEASE. The robot can carry things and can
                    grip things, but it has no way to let go of exactly one
                    of something. A slide-valve escapement -- one actuator,
                    one moving part -- meters exactly one seed per cycle.
  it borrows        the LIFT for drop height and the BASE for placement.
                    Nothing else. There is no new axis and no new interface.

Three things fall out of that, and they are what make this tool cheap:

  NO MOMENT COST. The whole assembly hangs on the peg's own axis (L = 0), so
  it spends none of the coupling's 0.45 N.m budget however much it carries.
  The claw paid for that lesson; this tool just inherits it.

  NO FORCE DEMAND. A dispenser releases, it never presses, so it makes no
  claim at all on the lean-pad's ~1.5-2 N ceiling -- the constraint that
  shaped the entire drawing tool simply does not apply here.

  CENTIMETRE TOLERANCE, NOT MILLIMETRE. A sown seed lands where it lands; a
  couple of cm of scatter is a planting, not a failure. So this controller
  deliberately does NOT inherit the claw's back-up-and-take-another-run-at-it
  refinement, whose whole purpose is squeezing a 26 mm block into a 62 x 28 mm
  capture window. Matching the control effort to the tolerance class is the
  point -- and re-using the claw's approach would also have meant touching
  travel constants that implicitly contain mm-scale wheel slip (SimNotes).

What is NOT dodged: the payload is loose free bodies riding in a carried
module. They are retained by GEOMETRY -- a capped tube over a shelf whose
only exit is blocked by the shuttle -- so unlike the claw's grip there is no
pose in which they have anywhere to go, and no swing analysis is needed.
"""

import math

import numpy as np

from pluggybot.behavior.navigation import drive_toward
from pluggybot.control import turn_command, wheel_targets, wrap_angle
from pluggybot.rack.coupling import (
  DISP_STROKE, SEED_COUNT, SEED_R,
)
from pluggybot.rack.swap import ARM_EXT, PLUG_LATERAL, VERTEX_AHEAD_OF_AXLE

SEED_MODULE = "module_seed"
APPROACH_LIFT = 0.128     # the swap preset: the tool rides clear to drive
SOW_OUTLET_Z = 0.050      # m above the floor at the moment of release.
                          # Low, because drop height is scatter: a seed
                          # leaving the shelf keeps the base's residual motion
                          # and bounces on landing, and both errors grow with
                          # the fall. Not lower, because the outlet is the
                          # module's lowest part and the floor is not the only
                          # thing it could find.
LIFT_SPEED = 0.05         # m/s ceiling on lift SETPOINT motion (lead-screw
                          # class) -- see ClawTool.set_lift: a step command
                          # unseats the gravity-latched module.
GATE_SPEED = 0.020        # m/s ceiling on the escapement setpoint. Same rule
                          # as every other position axis on this robot, and
                          # here it does a second job: the shuttle drags a
                          # seed across the shelf, and slamming it would fling
                          # the seed rather than let it fall off the end.
SLOW_RADIUS = 0.25        # m over which a terminal approach tapers to a stop.
                          # Comfortably longer than a hop between seed points
                          # (0.20 m), so the whole approach is a decelerating
                          # glide rather than a chase that overshoots.
GATE_SETTLE = 0.8         # s of dwell at each end of a cycle. The OUT dwell
                          # is the seed's fall time; the HOME dwell is the
                          # stack settling one seed into the pocket, which is
                          # the part a hurried cycle would skip.
SETTLE = 0.6


class SeedDispenser:
  """Drives the seed module: fetch it, carry it, sow at chosen points."""

  def __init__(self, model, data, swap, ground: str = "floor") -> None:
    self.model, self.data, self.swap = model, data, swap
    self.lift_act = model.actuator("lift").id
    self.arm_act = model.actuator("arm").id
    self.gate_act = model.actuator("seed_gate").id
    self.outlet_site = model.site("seed_outlet").id
    self.module_bid = model.body(SEED_MODULE).id
    self._module_gids = {
      g for g in range(model.ngeom)
      if (model.geom(g).name or "").startswith(SEED_MODULE)}
    self.seed_bids = [model.body(f"seed_{k}").id for k in range(SEED_COUNT)]
    self._seed_gids = {model.geom(f"seed_{k}_body").id
                       for k in range(SEED_COUNT)}
    try:
      self.ground_gid = model.geom(ground).id
    except KeyError:                      # home_world names its floor
      self.ground_gid = model.geom("home_floor_geom").id
    self.offset = (VERTEX_AHEAD_OF_AXLE, -PLUG_LATERAL)   # until measured

  # ---- state ---------------------------------------------------------------

  def outlet_world(self) -> np.ndarray:
    return np.array(self.data.site_xpos[self.outlet_site], dtype=float)

  def _touching_module(self, gid: int) -> bool:
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      pair = {c.geom1, c.geom2}
      if gid in pair and self._module_gids & pair:
        return True
    return False

  def dispensed(self, k: int) -> bool:
    """Has seed k actually LEFT the tool?

    Two conditions, because either alone lies. "Not touching the module" is
    true for a fraction of a second whenever a seed bounces inside the
    magazine; "below the outlet" is true of a seed still riding in the pocket
    on a tilted module. Together they are the release, and it is the same
    reasoning as every other physical criterion in this repo: a commanded
    gate is a belief, a seed in free space is a fact.
    """
    gid = self.model.geom(f"seed_{k}_body").id
    below = float(self.data.xpos[self.seed_bids[k]][2]) < \
        float(self.outlet_world()[2]) - SEED_R
    return below and not self._touching_module(gid)

  def landed(self, k: int) -> bool:
    """Seed k is resting on the ground. The end of the errand, and the only
    outcome a gardener would call a planting."""
    gid = self.model.geom(f"seed_{k}_body").id
    for i in range(self.data.ncon):
      pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
      if gid in pair and self.ground_gid in pair:
        return True
    return False

  def remaining(self) -> int:
    """Seeds still in the magazine."""
    return sum(1 for k in range(SEED_COUNT) if not self.dispensed(k))

  # ---- primitives ----------------------------------------------------------

  def ramp(self, act: int, target: float, speed: float,
           settle: float = 0.0) -> None:
    """Walk an actuator's SETPOINT at a bounded speed. Never write a position
    setpoint across a gap -- the house rule, and the one this tool's own
    lift would break first (a step has thrown a module clean off the fork)."""
    cur = float(self.data.ctrl[act])
    steps = max(int(abs(target - cur) / speed / self.model.opt.timestep), 1)
    for k in range(steps):
      self.data.ctrl[act] = cur + (target - cur) * (k + 1) / steps
      self.swap._step_once(0.0, 0.0)
    if settle:
      self.swap._run(settle, 0.0)

  def set_lift(self, target: float, settle: float = 1.2) -> None:
    self.ramp(self.lift_act, float(np.clip(target, 0.0, 0.31)),
              LIFT_SPEED, settle)

  def lower_outlet_to(self, world_z: float, settle: float = 1.2) -> float:
    """Put the outlet at a world height, by measured correction, ITERATED.

    Converge it, do not correct it once: the outlet follows the lift command
    at less than 1:1 because droop and the module's lean on the pad both
    change as the arm descends. The claw learned this the expensive way --
    a single correction left its pads 11.6 mm high and the grasp looked
    right while holding the block by its top 12 mm.
    """
    for _ in range(5):
      now = float(self.data.ctrl[self.lift_act])
      err = world_z - float(self.outlet_world()[2])
      if abs(err) < 0.0015:
        break
      self.set_lift(now + err, settle=settle)
    return float(self.outlet_world()[2]) - world_z

  def _face(self, heading: float, tol: float = 0.01, tries: int = 4) -> float:
    """Square up, then settle and re-check -- `slew` rate-limits the wheel
    command, so a P-controller that stops commanding at the target coasts
    past it. Tolerance is looser than the plotter's 0.004 rad on purpose:
    heading only rotates this tool's centimetre-scale placement error, it
    does not sweep a pen across a board."""
    for _ in range(tries):
      while abs(wrap_angle(heading - self.swap.reckoner.theta)) > tol:
        err = wrap_angle(heading - self.swap.reckoner.theta)
        tl, tr = wheel_targets(0.0, turn_command(err))
        self.swap._step_once(tl, tr)
      self.swap._run(SETTLE, 0.0)
      if abs(wrap_angle(heading - self.swap.reckoner.theta)) <= tol * 3:
        break
    return wrap_angle(heading - self.swap.reckoner.theta)

  # ---- placement -----------------------------------------------------------

  def calibrate(self) -> tuple[float, float]:
    """Measure where the outlet actually sits relative to the axle.

    Measured, never derived: nominal geometry contains neither RCC droop nor
    the module's lean on the pad, and this repo has paid for that twice --
    the pen tip hangs 72 mm below the peg against a 47 mm estimate, and the
    claw's grip point was 12.5 mm off its nominal along-track offset.
    """
    o = self.outlet_world()
    th = self.swap.reckoner.theta
    dx = float(o[0]) - self.swap.reckoner.x
    dy = float(o[1]) - self.swap.reckoner.y
    self.offset = (dx * math.cos(th) + dy * math.sin(th),
                   -dx * math.sin(th) + dy * math.cos(th))
    return self.offset

  def axle_pose_for(self, xy, heading: float) -> tuple[float, float]:
    """Where the axle must sit for the outlet to be over xy. The outlet
    inherits the fork's 5 cm lateral offset exactly as the pen and claw do,
    plus the escapement's own +y exit -- both measured into `offset`."""
    ox, oy = xy
    fwd, lat = self.offset
    fx, fy = math.cos(heading), math.sin(heading)
    return (ox - fwd * fx + lat * fy, oy - fwd * fy - lat * fx)

  def drive_over(self, xy, heading: float, timeout: float = 45.0) -> float:
    """Approach until the outlet is above xy, arm stowed for the drive.

    Deliberately a single straight run with no refinement pass: see the
    module docstring on tolerance class. Returns the residual distance so
    the caller can report placement honestly rather than assume it.

    Uses `drive_toward`'s TERMINAL mode. This tool's hops are ~0.2 m, and a
    pure-pursuit chase of a destination that close orbits it instead of
    reaching it -- three revolutions and ten wasted seconds a point, before
    the fix. See `drive_toward` for the mechanism.
    """
    tx, ty = self.axle_pose_for(xy, heading)
    self.data.ctrl[self.arm_act] = 0.0        # stowed is the driving config
    self.swap._run(1.0, 0.0)
    t0 = self.data.time
    while self.data.time - t0 < timeout:
      pose = (self.swap.reckoner.x, self.swap.reckoner.y,
              self.swap.reckoner.theta)
      if math.hypot(tx - pose[0], ty - pose[1]) < 0.015:
        break
      v, w = drive_toward(pose, (tx, ty), slow_radius=SLOW_RADIUS)
      tl, tr = wheel_targets(v, w)
      self.swap._step_once(tl, tr)
    self._face(heading)
    self.data.ctrl[self.arm_act] = ARM_EXT
    self.swap._run(1.5, 0.0)
    return math.hypot(tx - self.swap.reckoner.x, ty - self.swap.reckoner.y)

  # ---- the verbs -----------------------------------------------------------

  def dispense(self) -> dict:
    """One escapement cycle: meter exactly one seed out, then re-arm.

    The count comes from GEOMETRY, not from timing -- the pocket carries one
    seed past the end of the shelf while the blanking slab arrives under the
    magazine mouth. "Hold the gate open for 200 ms" would dispense a
    different number of seeds on a differently-loaded run, and this repo has
    already been bitten once by a choreography that assumed its own timing.
    """
    before = {k for k in range(SEED_COUNT) if self.dispensed(k)}
    self.ramp(self.gate_act, DISP_STROKE, GATE_SPEED, settle=GATE_SETTLE)
    self.ramp(self.gate_act, 0.0, GATE_SPEED, settle=GATE_SETTLE)
    after = {k for k in range(SEED_COUNT) if self.dispensed(k)}
    dropped = sorted(after - before)
    return {"dropped": dropped, "count": len(dropped),
            "remaining": SEED_COUNT - len(after)}

  @staticmethod
  def row_heading(points) -> float:
    """The heading that runs ALONG a row of seed points.

    Sowing across the row is what a person would draw on paper and is the
    worst possible choice for a differential drive: with the robot square to
    the row, every hop to the next point is a pure SIDEWAYS translation --
    the one motion the base cannot make. It costs a ~90 deg pivot out and
    another ~90 deg back, per seed, forever.

    Driving along the row makes each hop a straight run with no turn at all,
    which is also how a real seed drill works. Measured on the demo's
    three-point row, with the terminal-approach fix already in place:
    **611 -> 288 deg** of total turning and 65 -> 56 s. (Against the
    unfixed original, the two together are 3130 -> 288 deg, 111 -> 56 s.)
    """
    pts = list(points)
    if len(pts) < 2:
      return 0.0
    return math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0])

  def sow_at(self, points, heading: float | None = None) -> dict:
    """Drive to each point in turn and drop one seed there.

    `heading` defaults to running ALONG the row (see `row_heading`) rather
    than across it. Pass one explicitly only when something else fixes the
    robot's facing -- and expect to pay a pivot per seed if it is not the
    row's own direction.

    Reports where each seed actually ENDED UP, not where it was aimed: the
    placement error is the sum of the drive's residual, the release scatter
    and the roll on landing, and only the last of those is invisible to the
    controller.
    """
    if heading is None:
      heading = self.row_heading(points)
    results = []
    for xy in points:
      residual = self.drive_over(xy, heading)
      self.lower_outlet_to(SOW_OUTLET_Z)
      r = self.dispense()
      self.set_lift(APPROACH_LIFT, settle=1.0)
      placed = None
      if r["dropped"]:
        p = self.data.xpos[self.seed_bids[r["dropped"][0]]]
        placed = (float(p[0]), float(p[1]))
      results.append({
        "target": tuple(xy), "drive_residual_mm": residual * 1000,
        "seed": r["dropped"][0] if r["dropped"] else None,
        "placed": placed,
        "error_mm": (math.hypot(placed[0] - xy[0], placed[1] - xy[1]) * 1000
                     if placed else None),
      })
    return {"sown": results,
            "landed": sum(1 for k in range(SEED_COUNT) if self.landed(k))}
