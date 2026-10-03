"""Taking a tool from the rack and hanging it back, on legs (issue #405):
#378's spike (`scripts/arm_spike.py --approach`) as the served body's own
routines.

From anywhere in the house the robot walks to the bay's approach start, a
metre behind its working pose, faces the rack and finds its tags; walks in
steering by them (`rack.walk_in_twist`, the dock's rules); stands
`rack.SETTLE_AFTER_WALK_S`, because a stopped robot keeps turning; measures
the bay off its tags (`rack.bay_aim`) and, lined up, runs the fork -- in
under the peg, up, back out -- or else backs out and tries again. A tool it
carries rides the fork at `arm.CARRY`, high over its nose. Hanging one back
is the reverse, with a second settle and a second look once the fork has
come down from the carry pose (swinging a tool down shifts the stance about
25 mm, and the aim taken before is stale).

The fork moves along straight lines of its V's vertex in the torso frame,
the plate level, the joint targets off the arm's own kinematics and the
driver's GOAL -- never a measurement of the last forward pass
(`legs.arm.ArmDriver` says why).

A mixin: `QuadMission` is the rest of the body. The lifecycle names a bay
by its `coupling.STATION_YS` entry (the bay index space), so a bay here is
that entry's index (`bay_of`), on whichever section of the rack has it
(`rack.spec_of`: the hand-built tools' three, the built rail's three).
"""

from __future__ import annotations

import contextlib
import math

import mujoco
import numpy as np

from pluggybot.legs import arm as am
from pluggybot.legs import dock as dk
from pluggybot.legs import rack as rk
from pluggybot.rack.coupling import PEG_ABOVE_BODY, STATION_YS, contact_pairs, touching
from pluggybot.telemetry.protocol import ROBOT_ROOT, robot_roots
from pluggybot.tick import Routine

#: The approach starts this far behind a bay's working pose, facing the
#: rack (the spike's `--approach`: 41 of 41 from up to 0.3 m and 30 deg off).
APPROACH_STANDOFF_M = 1.0
#: The walk there gives up after this long, s. MEASURED: the bench's claw
#: hung back from the lab is the street with a tool aboard, ~100 s (the
#: cube search's walk, `legs.claw.WALK_PATIENCE_S`), and at the walk's own
#: 90 s every weighing ended with the claw still on the fork.
TO_BAY_S = 240.0
#: Walk-ins a swap tries before it gives up, and each one's budget, s.
TRIES = 3
WALK_IN_S = 20.0
#: Backing out of a walk-in that did not line up, or out of the bay once
#: the fork is clear: how far, its budget, and the settle after, m / s.
BACK_OUT_M, BACK_OUT_S, BACK_OUT_SETTLE_S = 0.6, 6.0, 0.8
#: The rack's tags are read this often while walking in, s.
LOOK_EVERY_S = 0.25
#: The fork is given this long past its line to arrive, holds this long at
#: each stop, and this long with the tool lifted before it is judged, s.
#: ⚠ ARRIVING IS NOT THE VERDICT: at the bottom of a put the tray takes the
#: tool's weight an instant before the feed-forward lets go of it, and the
#: arm stood 0.016 rad off its goal -- judged by it, a tool already hung
#: was lifted straight back off (every pen and claw stow, #405). The world
#: says what happened: seated and conducting, or hung.
FORK_ARRIVE_S, FORK_HOLD_S, PICK_HOLD_S = 1.0, 0.2, 0.5
#: The fork's speed between the stow or the carry pose and the bay, m/s:
#: the spike's; the verbs' own lines at the bay are `arm.FORK_V`.
FORK_FAST_V = 0.2
#: A fork move waits for the arm's joints to come this near their goal,
#: rad (`FORK_ARRIVE_S` at most); one through a point out of reach fails.
FORK_TOL = 0.002
#: A tool whose origin is farther than this from the fork's plate cannot
#: touch it, m: the claw's geoms reach 0.17 m from its origin, the fork's
#: 0.16 from the plate's. The power check reads contacts only nearer, and a
#: carried tool farther off has fallen away (`QuadMission.carry`).
NEAR_FORK_M = 0.5


def bay_of(station_y: float) -> int:
  """The bay the lifecycle means by a `STATION_YS` entry (module docstring)."""
  return min(range(len(STATION_YS)), key=lambda i: abs(STATION_YS[i] - station_y))


def _pose_of(model, name: str) -> tuple[float, float, float] | None:
  """A body's commissioned pose on the floor, (x, y, yaw), or None."""
  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
    return None
  b = model.body(name)
  w, _, _, z = b.quat
  return float(b.pos[0]), float(b.pos[1]), 2.0 * math.atan2(float(z), float(w))


class ToolSwap:
  """The rack's routines (the module docstring), on `QuadMission`."""

  def _init_swap(self, model) -> None:
    #: Where the tool rack was installed (its body's pose: the commissioning)
    #: and what the robot believes of it this approach, off its tags.
    self.tool_rack_prior = _pose_of(model, rk.RACK_BODY)
    self.tool_rack_seen: tuple[float, float, float] | None = None
    #: The bay a swap is working, as the lifecycle keys bays (`station_y`).
    self.swapping_at: float | None = None
    #: The last swap's record: its attempts and why each ended.
    self.last_swap: dict | None = None
    #: The module riding the fork: set by a pick, cleared by a put or a fall.
    #: Its geoms are the robot's own to its senses while it rides.
    self.carrying: str | None = None
    self._carried_bid = -1
    plate = self._plate_bid = model.body(self.handle.el("arm_plate")).id
    self._fork_gids = np.array([g for g in range(model.ngeom)
                                if int(model.geom_bodyid[g]) == plate], dtype=np.int32)
    self._tool_gids: dict[str, np.ndarray] = {}
    self._poles: dict[str, tuple] = {}
    #: WHO WAITS FOR A TAKEN BAY (#418): the lifecycle's routine, `(wx, wy,
    #: kind, since, hold=True) -> True once free`; None waits for nobody.
    self.bay_wait = None
    #: How far off the robot was that made the last swap give its bay up.
    self.peer_at_bay_m: float | None = None

  def _rebind_swap(self, model) -> None:
    """The fork's geoms by name, the per-module caches emptied (a recompiled
    world, `QuadMission.rebind`); the rack's commissioning is the world's."""
    plate = self._plate_bid = model.body(self.handle.el("arm_plate")).id
    self._fork_gids = np.array([g for g in range(model.ngeom)
                                if int(model.geom_bodyid[g]) == plate], dtype=np.int32)
    self._tool_gids, self._poles = {}, {}

  # ---- where things are -------------------------------------------------------

  def work_pose(self, bay: int) -> tuple[float, float, float]:
    """A bay's working pose in the world, off the commissioned rack."""
    return dk.compose(self.tool_rack_prior, rk.work_pose(*rk.spec_of(bay)))

  def rack_standoff(self, bay: int) -> tuple[float, float, float]:
    """Where the approach to a bay starts: behind its working pose, facing it."""
    return dk.compose(self.work_pose(bay), (-APPROACH_STANDOFF_M, 0.0, 0.0))

  def tool_gids(self, module: str) -> np.ndarray:
    gids = self._tool_gids.get(module)
    if gids is None:
      body = self.model.body(module).id
      gids = np.array([g for g in range(self.model.ngeom)
                       if int(self.model.body_rootid[self.model.geom_bodyid[g]]) == body],
                      dtype=np.int32)
      self._tool_gids[module] = gids
    return gids

  def on_this_fork(self, module: str) -> bool:
    """Anything of the module touching this robot's fork, seated or only
    resting there."""
    g = contact_pairs(self.data)
    if not g.shape[0]:
      return False
    tool, fork = self.tool_gids(module), self._fork_gids
    return bool(((np.isin(g[:, 0], tool) & np.isin(g[:, 1], fork))
                 | (np.isin(g[:, 1], tool) & np.isin(g[:, 0], fork))).any())

  def tool_state(self, module: str) -> dict:
    """Where a module is, off the world: on this robot's fork, hung on a
    bay (both flanks of both trays, plumb: `rack.on_bay`), and which bay
    it is nearest along the rail. KeyError for a module this world lacks."""
    d, m = self.data, self.model
    p = d.xpos[m.body(module).id]
    spec, k = rk.DEFAULT, 0
    if self.tool_rack_prior is not None:
      rx, ry, ryaw = self.tool_rack_prior
      along = -math.sin(ryaw) * (p[0] - rx) + math.cos(ryaw) * (p[1] - ry)
      spec, k = min(((s, k) for s in rk.SPECS for k in range(len(s.bays))),
                    key=lambda sk: abs(sk[0].bays[sk[1]] - along))
    return {"pos": [float(v) for v in p], "on_fork": self.on_this_fork(module),
            "hung": rk.on_bay(m, d, module, spec, k), "bay": spec.stations[k]}

  def tool_powered(self, module: str | None) -> bool:
    """The module's coupling conducting on this robot's fork. Read every
    physics step for the module the lifecycle watches, all day, so off ids
    kept per module and no contact read for a tool away from the fork:
    `rack.tool_power`'s lookups by name cost 20-32 us a step, this 1."""
    if module is None:
      return False
    kept = self._poles.get(module)
    if kept is None:
      kept = self._poles[module] = (rk.pole_ids(self.model, module, self.handle.prefix),
                                    self.model.body(module).id)
    poles, bid = kept
    if poles is None or not self.within_fork_reach(bid):
      return False
    return all(touching(self.data, peg, plates) for peg, plates in poles)

  def within_fork_reach(self, bid: int) -> bool:
    """A body near enough this fork's plate that any of it could touch the
    fork (`NEAR_FORK_M`): floats, as it is read every physics step."""
    x = self.data.xpos
    t, f = x[bid], x[self._plate_bid]
    return (t[0] - f[0]) ** 2 + (t[1] - f[1]) ** 2 + (t[2] - f[2]) ** 2 <= NEAR_FORK_M ** 2

  def seated_on(self, module: str) -> str | None:
    """Which robot has the module electrically seated, by root: every
    robot's fork, as the module is the world's."""
    for root in robot_roots(self.model):
      prefix = root[:-len(ROBOT_ROOT)]
      if rk.tool_power(self.model, self.data, module, prefix)["powered"]:
        return root
    return None

  # ---- the rack's tags ---------------------------------------------------------

  def look_at_rack(self):
    """One decode from the nose camera; a fit moves the rack's believed
    pose (the odometry frame)."""
    from pluggybot.legs.body import NAV_EYE
    seen = dk.seen_from(self.model, self.data, self.detect_board(),
                        self.handle.el(NAV_EYE), self.root)
    fix = rk.fit_rack(seen, rk.SPECS)
    if fix is not None:
      self.tool_rack_seen = dk.blend(self.tool_rack_seen,
                                     dk.compose(self.pose, (fix.x, fix.y, fix.yaw)))
    return fix

  def bay_aim(self, bay: int) -> rk.BayAim | None:
    """The bay as the arm needs it, measured off the rack's tags from
    where the robot stands (torso frame)."""
    from pluggybot.legs.body import NAV_EYE
    seen = rk.seen_in_torso(self.model, self.data, self.detect_board(),
                            self.handle.el(NAV_EYE), self.root)
    spec, k = rk.spec_of(bay)
    return rk.bay_aim(seen, spec, k, self.arm_spec.y)

  def _rack_error(self, bay: int) -> tuple[float, float, float]:
    work = dk.compose(self.tool_rack_seen, rk.work_pose(*rk.spec_of(bay)))
    return dk.relative(self.pose, work)

  def _find_rack_routine(self) -> Routine:
    if self.look_at_rack() is not None:
      return True
    for step in dk.SEARCH_STEPS:
      yield from self._turn_by_routine(step)
      if self.look_at_rack() is not None:
        return True
    return False

  def _rack_walk_in_routine(self, bay: int) -> Routine:
    from pluggybot.legs.policy import Twist
    t0 = last = float(self.data.time)
    while self.data.time - t0 < WALK_IN_S:
      tw = rk.walk_in_twist(*self._rack_error(bay), heading=-rk.SETTLE_DRIFT)
      if tw == Twist():
        return "stopped"
      yield from self._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
      if self.data.time - last >= LOOK_EVERY_S:
        self.look_at_rack()
        last = float(self.data.time)
    return "budget"

  def _rack_back_out_routine(self) -> Routine:
    return (yield from self._back_out_by_routine(BACK_OUT_M, BACK_OUT_S, rk.APPROACH_V,
                                                 BACK_OUT_SETTLE_S))

  # ---- the fork ---------------------------------------------------------------

  def _vertex_goal(self) -> tuple[float, float]:
    """Where the fork's V vertex is SENT, torso frame: the driver's goal
    through the arm's kinematics, the plate level."""
    s = self.arm_spec
    qs, qf = self.arm.goal
    wx, wz = am.wrist_xz(s, float(qs), float(qf - qs))
    return wx + s.fork.vertex_x, wz + s.fork.vertex_z

  def _fork_to_routine(self, x: float, z: float, speed: float = am.FORK_V) -> Routine:
    """The fork's vertex along a straight line to (x, z), torso frame,
    standing: False if a point on the way is out of the arm's reach, or the
    body went down under it -- a fall folds the arm, and re-aimed along the
    line it was held out through the get-up."""
    from pluggybot.legs.body import STANDING
    s = self.arm_spec
    x0, z0 = self._vertex_goal()
    n = max(1, int(math.hypot(x - x0, z - z0) / speed / self.model.opt.timestep))
    for k in range(1, n + 1):
      if self.posture != STANDING:
        return False
      goal = self.arm.goal
      q = am.solve_vertex(s, x0 + (x - x0) * k / n, z0 + (z - z0) * k / n,
                          near=(float(goal[0]), float(goal[1] - goal[0])))
      if q is None:
        return False
      self.arm.aim(*q)
      yield from self._twist_routine(0.0, 0.0, 0.0)
    t0 = float(self.data.time)
    while not self.arm.arrived(FORK_TOL) and self.data.time - t0 < FORK_ARRIVE_S:
      yield from self._twist_routine(0.0, 0.0, 0.0)
    yield from self._drive_routine(FORK_HOLD_S, 0.0, 0.0)
    return self.posture == STANDING

  def _payload(self, module: str | None) -> None:
    """What the arm's feed-forward carries: the tool's mass, its CoM on
    its plate under the peg (`rack.tool_face`: the mass is the plate's)."""
    if module is None:
      self.arm.payload = (0.0, (0.0, 0.0))
      return
    kg = float(self.model.body_subtreemass[self.model.body(module).id])
    self.arm.payload = (kg, (0.0, self.arm_spec.fork.seat_rise() - PEG_ABOVE_BODY))

  def _pick_routine(self, aim: rk.BayAim, module: str) -> Routine:
    """In under the peg the robot measured, up, back out: True if the tool
    came off its bay seated and conducting."""
    px, pz = aim.x, aim.z
    if not ((yield from self._fork_to_routine(px - am.STANDOFF, pz - am.FORK_DROP,
                                              FORK_FAST_V))
            and (yield from self._fork_to_routine(px, pz - am.FORK_DROP))):
      return False
    self._payload(module)
    yield from self._fork_to_routine(px, pz - am.FORK_DROP + am.LIFT)
    yield from self._fork_to_routine(px - am.BACK_OUT, pz - am.FORK_DROP + am.LIFT)
    yield from self._drive_routine(PICK_HOLD_S, 0.0, 0.0)
    picked = self.tool_powered(module) and not self.tool_state(module)["hung"]
    if not picked:
      self._payload(None)
    return picked

  def _put_routine(self, aim: rk.BayAim, module: str) -> Routine:
    """Over the bay the robot measured, down, out: True if the tool hangs."""
    px, pz = aim.x, aim.z
    if not ((yield from self._fork_to_routine(px, pz - am.FORK_DROP + am.LIFT))
            and (yield from self._fork_to_routine(px, pz - am.FORK_DROP))):
      return False
    self._payload(None)
    yield from self._fork_to_routine(px - am.BACK_OUT, pz - am.FORK_DROP)
    yield from self._drive_routine(PICK_HOLD_S, 0.0, 0.0)
    return self.tool_state(module)["hung"] and not self.tool_powered(module)

  def carry_routine(self) -> Routine:
    """The fork to the carry pose, over the nose."""
    return (yield from self._fork_to_routine(*am.CARRY, FORK_FAST_V))

  # ---- the two swaps ----------------------------------------------------------

  def _to_the_bay_routine(self, bay: int, rec: dict) -> Routine:
    """The walk to the bay's approach start, facing the rack, and a look:
    "ok", or why not."""
    sx, sy, syaw = self.rack_standoff(bay)
    arrived = yield from self.drive_to_routine(sx, sy, timeout=TO_BAY_S)
    if not arrived:
      rec["drive"] = self.last_drive
      return "no-route"
    yield from self.face_routine(syaw)
    self.tool_rack_seen = None
    if not (yield from self._find_rack_routine()):
      return "no rack"
    return "ok"

  def _bay_free_routine(self, bay: int, kind: str, rec: dict) -> Routine:
    """Hold at the approach's start while another robot works this bay or
    the next (#418, decided: the rover's #346 wait): its REPORTED pose within
    the planner's disc of this bay's working pose (`peer_on_the_goal`: 0.55
    m standing -- the next bay's working pose is 0.30 m off, two bays over
    0.60). The minds choose who goes first; the wait breaks the symmetry two
    robots walking in at once cannot, and a bump that fires on both does
    not (#418: both flinched, 17 of 20). True once free; False if the wait
    gave up, said in `rec`."""
    self.peer_at_bay_m = None
    wx, wy, _ = self.work_pose(bay)
    if self.peer_on_the_goal(wx, wy) is None:
      return True
    free = False
    if self.bay_wait is not None:
      free = yield from self.bay_wait(wx, wy, kind, float(self.data.time), hold=True)
    if not free:
      self.peer_at_bay_m = self.peer_on_the_goal(wx, wy)
      rec["why"] = "blocked"
    return bool(free)

  def _lined_up_routine(self, bay: int, att: dict) -> Routine:
    """A walk in, the settle and the measurement: the bay's aim if lined
    up, else None (having backed out)."""
    walked = yield from self._rack_walk_in_routine(bay)
    yield from self._drive_routine(rk.SETTLE_AFTER_WALK_S, 0.0, 0.0)
    aim = self.bay_aim(bay)
    att["walk"] = walked
    if aim is not None:
      att.update(across=round(aim.across * 1000, 1), yaw=round(math.degrees(aim.yaw), 1))
    if aim is None or not aim.lined_up:
      att["why"] = "no fit at the bay" if aim is None else "not lined up"
      yield from self._rack_back_out_routine()
      if self.look_at_rack() is None:
        yield from self._find_rack_routine()
      return None
    return aim

  def fetch_routine(self, bay: int, module: str) -> Routine:
    """Walk to a bay and take its tool, carrying it at the carry pose:
    "arrived" once the fork went in (the verdict is the tool's own state),
    "no-route" if the walk found no way there, "blocked" if another robot
    held the bay past the wait (#418), "timeout" if no walk-in lined up."""
    rec = {"op": "fetch", "bay": bay, "module": module, "attempts": []}
    self.last_swap, self.peer_at_bay_m = rec, None
    if self.tool_rack_prior is None:
      rec["why"] = "no rack"
      return "no-route"
    yield from self.stand_routine()
    yield from self.stow_arm_routine()
    why = yield from self._to_the_bay_routine(bay, rec)
    if why != "ok":
      rec["why"] = why
      return "no-route"
    if not (yield from self._bay_free_routine(bay, "pick", rec)):
      return "blocked"
    with self._at_the_bay(bay):
      return (yield from self._fetch_at_routine(bay, module, rec))

  def _fetch_at_routine(self, bay: int, module: str, rec: dict) -> Routine:
    for _ in range(TRIES):
      att = {}
      rec["attempts"].append(att)
      aim = yield from self._lined_up_routine(bay, att)
      if aim is None:
        continue
      picked = yield from self._pick_routine(aim, module)
      # A tool on the fork is carried, seated or not: folded, it is thrown,
      # and unclaimed it was a bump on every step of a walk. ⚠ A tool is
      # the body's own to the bumper only while carried: a filter on every
      # tool against the fork let the pair's carried tools knock each other
      # off, where the bump backs the robots apart (4 of 12 hung back)
      held = picked or self.on_this_fork(module)
      att["why"] = ("picked" if picked else "on the fork, not seated" if held
                    else "the fork came out without it")
      if held:
        self.carry(module)
        yield from self.carry_routine()
      else:
        self.carry(None)
        yield from self.stow_arm_routine()
      yield from self._rack_back_out_routine()
      return "arrived"
    return "timeout"

  @contextlib.contextmanager
  def _at_the_bay(self, bay: int):
    """The part of a swap AT the bay: the rest reflex and a restart's save
    wait it out (`working`), the walk there need not -- a save held off
    through a 45 s walk outlasts a stop's grace."""
    was, self.working = self.working, True
    self.swapping_at = STATION_YS[bay]
    try:
      yield
    finally:
      self.swapping_at = None
      self.working = was

  def stow_routine(self, bay: int, module: str) -> Routine:
    """Walk to the tool's bay and hang it back, folding the arm after:
    "arrived" once the fork came down over the bay, "no-route", "blocked"
    (#418) or "timeout"."""
    rec = {"op": "stow", "bay": bay, "module": module, "attempts": []}
    self.last_swap, self.peer_at_bay_m = rec, None
    if self.tool_rack_prior is None:
      rec["why"] = "no rack"
      return "no-route"
    yield from self.stand_routine()
    why = yield from self._to_the_bay_routine(bay, rec)
    if why != "ok":
      rec["why"] = why
      return "no-route"
    if not (yield from self._bay_free_routine(bay, "return", rec)):
      return "blocked"
    with self._at_the_bay(bay):
      return (yield from self._stow_at_routine(bay, module, rec))

  def _stow_at_routine(self, bay: int, module: str, rec: dict) -> Routine:
    for _ in range(TRIES):
      att = {}
      rec["attempts"].append(att)
      aim = yield from self._lined_up_routine(bay, att)
      if aim is None:
        continue
      # Down from the carry pose to over the bay, then stand and look
      # again: the swing moved the stance under the arm (module docstring).
      yield from self._fork_to_routine(aim.x - am.BACK_OUT,
                                       aim.z - am.FORK_DROP + am.LIFT, FORK_FAST_V)
      yield from self._drive_routine(rk.SETTLE_AFTER_WALK_S, 0.0, 0.0)
      aim = self.bay_aim(bay) or aim
      hung = yield from self._put_routine(aim, module)
      att["why"] = "hung" if hung else "the fork came out and it did not hang"
      self.carry(None if hung or not self.on_this_fork(module) else module)
      if self.carrying is None:
        yield from self.stow_arm_routine()
      else:
        yield from self.carry_routine()
      yield from self._rack_back_out_routine()
      return "arrived"
    return "timeout"

  def swap_trace(self) -> str:
    rec = self.last_swap
    if not rec:
      return "no swap recorded"
    # ...the bay in its own row's letters, as the robot is told them (#407)
    spec, local = rk.spec_of(rec["bay"])
    where = ("rail bay " if spec is rk.BUILT else "bay ") + chr(ord("A") + local)
    parts = [f"{rec['op']} {rec['module']} at {where}"]
    if rec.get("why"):
      parts.append(str(rec["why"]))
    for i, a in enumerate(rec["attempts"], 1):
      parts.append(f"#{i} {a.get('walk', '')} across {a.get('across')} mm "
                   f"yaw {a.get('yaw')} deg -> {a.get('why')}")
    return "; ".join(parts)
