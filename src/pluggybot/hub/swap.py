"""Robot-side tool swap: the coupling spike's phases, driven by the real base.

The spike (hub/coupling.py) validated the fork-and-peg latch with an ideal
carrier; this runs the same slide-under / lift / retreat verbs through the
actual robot in hub_world.xml -- wheel velocity servos with slew, the real
lift lead-screw, odometry for the approach distance, and the fork hanging on
the RCC. Terminal alignment beyond odometry (a visual servo against a hub
fiducial) is deliberately NOT here yet: the swap assumes the hand-off pose
navigation already delivers, exactly as DOCK did in milestone 6.
"""

import math

import mujoco

from pluggybot.control import slew, wheel_targets
from pluggybot.hub.coupling import (
  HUB_PEG_Z, HUB_STATION_YS, LIFT_STEP, PEG_R, RACK_HANG_X, TRAY_VERTEX_DROP,
)
from pluggybot.odometry.dead_reckoning import DeadReckoner

PLUG_LATERAL = 0.05       # the fork line rides 5 cm right of the robot
                          # centerline, same as the plug did
ARM_EXT = 0.06            # arm extension for hub work: retracted, the fork
                          # vertex sits 25 mm BEHIND the front bumper, so the
                          # chassis bottoms on the tray posts before the fork
                          # reaches the peg (measured: 43 mm short, wheels
                          # slipping) -- the same reach problem the plug had
DROOP_COMP = 0.008        # RCC droop of the fork line under gravity
                          # (measured 7 mm here; the plug's was 7.8)
FORK_MOUNT_RAISE = 0.035  # the fork hangs this much higher on the wrist than
                          # the spike carrier's, in two instalments. 16 mm:
                          # at the original drop the right prong
                          # interpenetrated the chassis top by 9 mm at zero
                          # lift (caught by the geometric sweep, silent to
                          # contacts). +19 mm: the lean-pad hangs below
                          # everything else, and parked it sat ON the chassis
                          # and jacked the fork into the scanner's row. The
                          # lift preset absorbs the whole thing.
VERTEX_AHEAD_OF_AXLE = 0.175 + ARM_EXT   # fork V vertex forward of the axle
                          # (carriage -0.10 + arm tip 0.14 + vertex 0.055
                          #  + axle offset 0.08 + extension)
STANDOFF = 0.45           # m from the station at the hand-off pose
APPROACH_V = 0.05         # m/s creep toward the hub
RETREAT_DIST = 0.35       # m backed away between phases
STALL_TIME = 0.4          # s of no odometry progress = bottomed out
PICK_OVERSHOOT = 0.004    # m of deliberate overdrive past the peg line, the
                          # spike's own aligned figure: the peg lands low on
                          # the V's near flank and the LIFT self-centres it.
                          # 12 mm was tried and measured too greedy -- the peg
                          # caught the flank TIP, wedged between the fork V
                          # and the tray V during the hand-off. ⚠ This value
                          # implicitly contains the wheels' mm-scale slip
                          # over the approach: the issue-3 spike briefly
                          # hardened the tire contact, the slip shrank
                          # ~0.7 mm, and the front-heavy pen module's pick
                          # went to a knife edge. Anything that changes wheel
                          # contact or joint friction must re-verify the
                          # bay-C pick.
RETURN_CLEARANCE = 0.020  # m of extra lift while carrying a module INTO a
                          # bay, so the peg passes over the tray flanks
                          # (their tops sit above where a carried peg rides)
RELEASE_DROP = 0.014      # m the fork descends BELOW the pick height to let
                          # go. Capture and release are not symmetric: the
                          # pick only needs the fork V to reach peg height,
                          # but undoing exactly that leaves the V still
                          # touching (measured: peg correctly seated in both
                          # trays AND still resting on the fork's plates,
                          # 5 mm high, then dragged out on the way back).
CARRY_OFFSET = 0.007      # m the carried peg rides AHEAD of the vertex
                          # (fork tilts nose-down under the tool's weight;
                          # measured). The return compensates, so the peg --
                          # not the vertex -- parks over the tray line.
# Approach travel: fork vertex from the standoff to just past the peg line.
APPROACH_DIST = STANDOFF - VERTEX_AHEAD_OF_AXLE + PICK_OVERSHOOT
RETURN_DIST = RETREAT_DIST + PICK_OVERSHOOT - CARRY_OFFSET


def align_lift(peg_z: float = HUB_PEG_Z) -> float:
  """The lift setpoint that puts the FORK AXIS at `peg_z` -- the height a
  pick must enter at, and the reference every other height is measured from.

  Named because it was written out longhand in four places (the standoff
  preset, the mission's start pose, the plotter's carry and board heights),
  and one of those copies was already a bug once: they had been typed as
  bare literals, so raising the fork mount for the lean-pad would have left
  a copy pointing at the old geometry.
  """
  return peg_z - 0.145 - FORK_MOUNT_RAISE + DROOP_COMP


class HubSwap:
  """Scripted pick/return cycles for one robot in hub_world.xml."""

  def __init__(self, model, data) -> None:
    self.model, self.data = model, data
    m = model
    self.left_act = m.actuator("left_motor").id
    self.right_act = m.actuator("right_motor").id
    self.lift_act = m.actuator("lift").id
    self.left_adr = m.joint("left_wheel_joint").qposadr[0]
    self.right_adr = m.joint("right_wheel_joint").qposadr[0]
    self.gyro_adr = m.sensor("imu_gyro").adr[0]
    self.reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
    # Optional per-step callback. Every phase of both the swap and the
    # mission bottoms out in _step_once, so one hook here is enough to
    # drive a viewer (or any telemetry) through the whole run.
    self.on_step = None

  # ---- plumbing ------------------------------------------------------------

  def place_at_standoff(self, station_y: float, dy: float = 0.0,
                        dyaw_deg: float = 0.0) -> None:
    """Robot at the hand-off pose facing the hub (with test jitter), lift
    preset so the fork axis sits at peg height (the align stage)."""
    yaw = math.pi + math.radians(dyaw_deg)
    axle_x = RACK_HANG_X + STANDOFF
    axle_y = station_y - PLUG_LATERAL + dy
    d = self.data
    d.qpos[0] = axle_x + 0.08 * math.cos(yaw)
    d.qpos[1] = axle_y + 0.08 * math.sin(yaw)
    d.qpos[2] = 0.045
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    lift0 = align_lift()
    d.qpos[self.model.joint("lift_joint").qposadr[0]] = lift0
    d.ctrl[self.lift_act] = lift0
    d.ctrl[self.model.actuator("arm").id] = ARM_EXT
    d.qpos[self.model.joint("arm_joint").qposadr[0]] = ARM_EXT
    mujoco.mj_forward(self.model, d)
    self._run(1.0, 0.0)                  # settle
    self.reckoner.x, self.reckoner.y, self.reckoner.theta = axle_x, axle_y, yaw
    self.reckoner.update(float(d.qpos[self.left_adr]),
                         float(d.qpos[self.right_adr]))

  def _step_once(self, tl: float, tr: float) -> None:
    d, m = self.data, self.model
    ts = m.opt.timestep
    d.ctrl[self.left_act] = slew(d.ctrl[self.left_act], tl, ts)
    d.ctrl[self.right_act] = slew(d.ctrl[self.right_act], tr, ts)
    mujoco.mj_step(m, d)
    self.reckoner.update(float(d.qpos[self.left_adr]),
                         float(d.qpos[self.right_adr]),
                         gyro_yaw_rate=float(d.sensordata[self.gyro_adr + 2]),
                         dt=ts)
    if self.on_step is not None:
      self.on_step()

  def _run(self, seconds: float, v: float,
           lift_target: float | None = None) -> None:
    if lift_target is not None:
      self.data.ctrl[self.lift_act] = lift_target
    tl, tr = wheel_targets(v, 0.0)
    for _ in range(round(seconds / self.model.opt.timestep)):
      self._step_once(tl, tr)

  def _drive_until(self, distance: float, v: float, timeout: float = 20.0,
                   stall_stop: bool = True, steer_fn=None, stop_fn=None) -> str:
    """Travel `distance` of ODOMETRY path along the current heading; with
    stall_stop, ending early on no-progress (bottomed against the hub).
    steer_fn() -> w lets a terminal visual servo trim the heading while
    creeping (the dock_eye pattern); None drives straight.

    stop_fn() -> bool ends the drive on a SENSED event rather than a
    believed distance, and is the right stop for anything that ends in
    contact: pressing against the rack slips the wheels, dead reckoning
    counts the slip as progress, and neither the distance target nor the
    stall detector ever fires (measured: a charge approach that had been
    connected for 15 s kept pushing until it timed out, and the wasted
    press flattened the battery).
    """
    x0, y0 = self.reckoner.x, self.reckoner.y
    tl, tr = wheel_targets(v, 0.0)
    t0 = self.data.time
    last_dist, last_progress = 0.0, t0
    while self.data.time - t0 < timeout:
      if steer_fn is not None:
        tl, tr = wheel_targets(v, float(steer_fn()))
      self._step_once(tl, tr)
      if stop_fn is not None and stop_fn():
        return "stopped"
      dist = math.hypot(self.reckoner.x - x0, self.reckoner.y - y0)
      if dist >= distance:
        return "arrived"
      if dist > last_dist + 0.001:
        last_dist, last_progress = dist, self.data.time
      elif (stall_stop and self.data.time - last_progress > STALL_TIME
            and self.data.time - t0 > 0.5):
        return "stalled"
    return "timeout"

  # ---- the verbs -----------------------------------------------------------

  def pick(self, steer_fn=None, dist: float | None = None) -> str:
    """Slide under the module's peg, lift it off the trays, back away.

    dist overrides the approach travel: callers that know their believed
    distance to the hang plane (the mission) pass it explicitly -- a fixed
    travel assumes a perfect standoff, and the coupling's capture window is
    +/-11 mm while navigation's arrival radius is 80.
    """
    why = self._drive_until(APPROACH_DIST if dist is None else dist,
                            APPROACH_V, steer_fn=steer_fn)
    self._run(0.5, 0.0)                                     # settle
    lift_now = float(self.data.ctrl[self.lift_act])
    self._run(2.0, 0.0, lift_target=lift_now + LIFT_STEP)
    self._drive_until(RETREAT_DIST, -0.08, stall_stop=False)
    self._run(1.0, 0.0)
    return why

  def put_back(self, steer_fn=None, dist: float | None = None) -> str:
    """Carry the module back in, lower it onto the trays, leave empty.

    dist as in pick(). The default mirrors the bare-world choreography
    (re-approach from where pick's retreat ended); a mission approaching
    from the standoff must pass its own believed travel -- the default
    overdrove 12.5 cm there and rammed the carried module into the rack.

    Lift-clear-then-set-down, rather than driving straight in at carry
    height: a carried peg settles into the fork's V only ~9-18 mm above its
    hanging rest, which is BELOW the tray flanks' tops. Driving in flat
    means the peg strikes the flank instead of passing over it -- measured
    as a module that came down 29 mm too low, still on the fork, and rode
    away again. Exactly what a person does setting a tool on a rack.
    """
    lift_now = float(self.data.ctrl[self.lift_act])
    self._run(0.8, 0.0, lift_target=lift_now + RETURN_CLEARANCE)
    why = self._drive_until(RETURN_DIST if dist is None else dist,
                            APPROACH_V, steer_fn=steer_fn)
    self._run(0.5, 0.0)
    self._run(2.0, 0.0, lift_target=lift_now - LIFT_STEP - RELEASE_DROP)
    self._drive_until(RETREAT_DIST, -0.08, stall_stop=False)
    self._run(1.0, 0.0)
    return why

  # ---- truth checks (script-level verification only) -----------------------

  def rack_frame(self, world_pos) -> tuple[float, float, float]:
    """A world point in the RACK's own frame, read from the rack body's live
    pose. Hard-won: comparing world coordinates against rack-local constants
    works only where the rack sits at the origin unrotated (the bare hub
    world) and silently reports every correct placement as a failure once
    the rack is somewhere else in a room -- and the rack is a free body, so
    even in the bare world it may have shifted."""
    bid = self.model.body("rack").id
    rp = self.data.xpos[bid]
    rm = self.data.xmat[bid].reshape(3, 3)
    d = [float(world_pos[k]) - float(rp[k]) for k in range(3)]
    return tuple(sum(float(rm[i][j]) * d[i] for i in range(3)) for j in range(3))

  def module_state(self, name: str) -> dict:
    """Where a module is, in the terms every demo and test verifies: riding
    the fork, or hanging in a bay. Bay membership is checked too -- without
    it, a module dropped one bay over reads as correctly stowed."""
    p = self.data.xpos[self.model.body(name).id]
    vx = self.data.site_xpos[self.model.site("fork_vertex").id]
    lx, ly, lz = self.rack_frame(p)
    peg_rest_z = HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R
    on_fork = (abs(float(p[0]) - float(vx[0])) < 0.03
               and abs(float(p[1]) - float(vx[1])) < 0.04)
    bay_err = min(abs(ly - by) for by in HUB_STATION_YS)
    hung = (abs(lx - RACK_HANG_X) < 0.012
            and abs(lz - (peg_rest_z - 0.022)) < 0.008
            and bay_err < 0.030)
    return {"pos": [float(v) for v in p], "rack_frame": [lx, ly, lz],
            "on_fork": on_fork, "hung": hung, "bay_err_mm": bay_err * 1000}
