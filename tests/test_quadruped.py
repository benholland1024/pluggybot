"""The quadruped in the served world (issue #387, `legs/body.py`,
`legs/world.py`): each rule its first deploy rests on, pinned as cheaply as
it can be while still failing for the right reason -- the body's members,
the world it lives in, the rest reflex, the get-up and the `stuck` death,
the planner's sizes, the reflexes, the prompt in its own words, and upkeep
off as a configuration. Docking in the house is flown behind `--endurance`."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import navigator as nav
from pluggybot.behavior.navigation import plan
from pluggybot.body import StubBody, members
from pluggybot.legs import body as qb
from pluggybot.legs import posture as pz
from pluggybot.legs import world as lw
from pluggybot.legs.model import CHOSEN, body_xml
from pluggybot.lifecycle import QUAD_HOME, world_config, world_facts, world_for
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.mind import constitution as constitutions
from pluggybot.mind import overseer as ov
from pluggybot.perception.lidar import robot_geoms
from pluggybot.procedure import steps as st
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


@pytest.fixture(scope="module")
def quad_world():
  """The home world with one quadruped and its dock, compiled once."""
  return lw.home_spec().compile()


def quad(model):
  data = mujoco.MjData(model)
  cfg = world_config(QUAD_HOME)
  body = qb.QuadBody(model, data, realtime=False, grid_bounds=cfg["grid_bounds"])
  return body


# ---- the body and its world ------------------------------------------------------


def test_the_quadruped_implements_every_member_and_is_chosen_by_its_legs(quad_world):
  from pluggybot.body import body_for, is_quadruped
  body = quad(quad_world)
  try:
    missing = sorted(m for m in members() if not hasattr(body, m))
    assert not missing, missing
    assert body.STILL == (0.0, 0.0, 0.0) and body.rights_itself
    assert isinstance(body_for(quad_world, mujoco.MjData(quad_world)), qb.QuadBody)
    rover = mujoco.MjModel.from_xml_path("models/home_world.xml")
    assert is_quadruped(quad_world) and not is_quadruped(rover)
  finally:
    body.close()


def test_the_world_takes_out_the_rover_and_nothing_of_the_worlds(quad_world):
  """The rover's body, actuators, sensor and exclude go, and its rack with
  its modules, their drivers and the dispenser's seeds (#405: the arm's
  fork takes a longer peg); the plates' sensors are the world's and stay;
  the robot keeps its joint ranges in RADIANS (#386: in degrees the stand
  threw it 0.4 m up)."""
  m = quad_world
  names = lambda n, obj: {mujoco.mj_id2name(m, obj, i) for i in range(n)}  # noqa: E731
  acts = names(m.nu, mujoco.mjtObj.mjOBJ_ACTUATOR)
  assert not acts & {"left_motor", "right_motor", "lift", "arm", "pen_carriage",
                     "claw_l", "claw_r", "seed_gate"}
  assert "FL_knee" in acts
  bodies = names(m.nbody, mujoco.mjtObj.mjOBJ_BODY)
  assert not bodies & {"rack", "rack_built", "module_plug", "module_seed", "seed_0"}
  sensors = names(m.nsensor, mujoco.mjtObj.mjOBJ_SENSOR)
  assert "imu_gyro" not in sensors and "garden_plate_pos" in sensors
  assert m.nexclude == 0 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dock") >= 0
  assert m.jnt_range[m.joint("FL_knee").id] == pytest.approx((-2.75, -0.35))
  x, y, yaw = lw.dock_pose()
  assert (float(m.body("dock").pos[0]), float(m.body("dock").pos[1])) == pytest.approx((x, y))


def test_home_with_legs_is_its_own_world_and_offers_no_tool_errand():
  """No errand that needs a tool until the tools are rebuilt for its fork
  (#406, #407), no workshop; a program may fetch and stow its own rack's
  three tools (#405) and nothing else -- anything else refused with the
  reason, as any unknown name is -- and move only its own arm's joints.
  The lab's acts take no tool, so the lab is here (#403); the tower is not."""
  from pluggybot.legs import rack as legs_rack
  assert world_for("home", "quadruped") == QUAD_HOME == "home_quad"
  cfg = world_config(QUAD_HOME)
  assert cfg["body"] == "quadruped" and not cfg["tools"] and cfg["built_bays"] == 0
  assert cfg["swap"] and cfg["tool_bays"] == legs_rack.TOOL_BAYS
  assert "lab" in cfg and "tower" not in cfg
  menu = ov.Menu.for_world(QUAD_HOME)
  assert not {"carry", "dance", "draw", "artwork", "census"} & set(menu.available())
  assert {"explore", "charge", "idle"} <= set(menu.available())
  facts = world_facts(QUAD_HOME)
  assert facts.tools == tuple(legs_rack.TOOL_BAYS) and facts.axes == ("shoulder", "elbow")
  assert "lift" not in facts.sensors and "shoulder" in facts.sensors
  assert st.check_step(st.VERBS["fetch"], {"tool": "module_pen"}, facts) == []
  assert st.check_step(st.VERBS["fetch"], {"tool": "module_seed"}, facts)
  bad = st.check_step(st.VERBS["draw"], {"program": "circle", "board": "whiteboard_a"}, facts)
  assert bad and "not on this body yet" in bad[0]
  assert st.check_step(st.VERBS["drive_to"], {"x": 1.0, "y": 0.0}, facts) == []
  # ...and the rover's world keeps every one
  assert "carry" in ov.Menu.for_world("home").available()


# ---- the rest reflex, the get-up, and the `stuck` death -----------------------


def test_the_rest_reflex_lies_down_when_still_and_stands_before_a_walk(quad_world, monkeypatch):
  """Ben's rule (#387): code lies the body down after `T_REST_S` without a
  motion command, and stands it up before the next one -- the mind is not
  asked. The delay is shortened here: the RULE is the reflex."""
  monkeypatch.setattr(pz, "T_REST_S", 2.0)
  body = quad(quad_world)
  try:
    body.start_at(1.5, 0.5, 0.0)
    body.run(body.hold_routine(1.5))
    assert body.posture == qb.STANDING, "not before its delay"
    body.run(body.hold_routine(4.0))              # the delay, lie-down and settle
    assert body.posture == qb.LYING and body.resting
    x0 = body.true_pose()[0]
    body.run(body.velocity_routine(1.0, 0.4, 0.0))
    assert body.posture == qb.STANDING, "a motion command stands it first"
    assert body.true_pose()[0] > x0 + 0.1, "...and then walks"
  finally:
    body.close()


def test_standing_the_body_up_steps_nothing(quad_world):
  """#387, flown in a pair: the timer stood Rowan up with a second of
  settling stepped inside `start_at`, and Luca -- walking in the same loop --
  went that second with no policy and no odometry: it fell, came up 0.44 m
  from where it believed it was, and found no route to the dock. A stand-up
  lands on the seam of a loop another robot walks in, so placing a body
  steps nothing; it settles under the day's next command."""
  body = quad(quad_world)
  try:
    t0 = float(body.data.time)
    body.start_at(1.5, 0.5, 0.0)
    assert float(body.data.time) == t0
    assert body.posture == qb.STANDING and body.pose[:2] == pytest.approx((1.5, 0.5))
  finally:
    body.close()


def test_a_save_waits_out_every_move_the_body_makes():
  """A save mid-move holds half a scripted move no file keeps
  (`continuation.Keeper.busy`): its words are the body's, every one but
  standing and lying still."""
  from pluggybot import continuation
  assert set(continuation.MOVING_POSTURES) == set(qb.POSTURES) - {qb.STANDING, qb.LYING}
  assert set(qb.QuadMission.MOVING) == set(continuation.MOVING_POSTURES)


def test_lying_is_the_odometrys_rest_and_a_restart_keeps_what_it_learned(quad_world):
  """#425's wiring: the posture machine's `lying` is what tells the
  odometry the body rests (`LegOdometry.resting`; the rule is
  `test_a_body_lying_still_keeps_its_heading_on_a_biased_gyro`), a stand is
  not, and the gyro's offset learned lying rides a restart. The physics is
  not stepped: the bookkeeping after a step is called directly."""
  from pluggybot.perception import imu
  body, again = quad(quad_world), quad(quad_world)
  try:
    body.start_at(1.5, 0.5, 0.0)
    m = body.mission
    m.odo.imu.gyro_bias = np.array([0.0, 0.0, imu.GYRO_BIAS])
    m.posture = qb.LYING
    yaw = m.odo.yaw
    for _ in range(round(5.0 / body.model.opt.timestep)):
      m._after_physics()
    assert abs(math.degrees(m.odo.yaw - yaw)) < 0.01, "unheld, 0.25 deg"
    m.posture = qb.STANDING
    m._after_physics()
    assert not m.odo.resting
    again.restore_kept(*body.kept_state())
    assert again.mission.odo.still.bias == m.odo.still.bias != [0.0, 0.0, 0.0]
  finally:
    body.close()
    again.close()


def test_a_fall_is_got_up_from_by_the_policy(quad_world):
  body = quad(quad_world)
  try:
    body.start_at(1.5, 0.5, 0.0)
    q = body.handle.qpos_adr(quad_world)
    body.data.qpos[q + 2] = 0.25                 # on its side
    body.data.qpos[q + 3:q + 7] = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    mujoco.mj_forward(quad_world, body.data)
    downs, t0 = [], float(body.data.time)
    while body.data.time - t0 < 8.0:
      body.run(body.hold_routine(0.02))
      downs.append(body.posture)
    assert qb.GETTING_UP in downs and body.posture == qb.STANDING
    assert body.mission.falls == 1
  finally:
    body.close()


def _quad_spike():
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  import quad_spike
  return quad_spike


#: #389's bound on a joint while getting up: 60 % of the drivers' peak.
GETUP_CAP = 0.6 * CHOSEN.motor.peak_torque


def test_the_get_up_rises_from_the_belly_rather_than_springing():
  """#389: #377's get-up policy stood from the belly in 0.2 s by driving the
  knees to the drivers' 22 N*m peak -- hard on the gearboxes, a hazard to a
  hand, and it throws #378's arm. The committed one rises: no joint past
  60 % of the peak, a second or more to the stand, flown in our physics as
  `quad_spike.py --getup` flies it."""
  r = _quad_spike().getup_trial(qb.GETUP_NPZ)
  assert r["stood"] is not None and r["stood"] >= 1.0
  assert r["peak"].max() <= GETUP_CAP


def test_the_get_up_stands_from_nineteen_of_twenty_falls_without_springing():
  """...and from #377's twenty random drops: nineteen or more stand, the
  median a second or more from the first touch (#377's: 0.3 s), and at most
  two drive a joint past 60 % of the peak once the landing is over (#377's:
  ten). The landing itself is not held to it: the legs meet the floor
  wherever the fall threw them (SimNotes, "A gentler get-up")."""
  qs = _quad_spike()
  runs = [qs.getup_trial(qb.GETUP_NPZ, drop) for drop in qs.getup_drops()]
  stood = [r for r in runs if r["stood"] is not None]
  assert len(stood) >= 19
  assert np.median([r["up"] for r in stood]) >= 1.0
  assert sum(r["peak_after"].max() > GETUP_CAP for r in runs) <= 2


def test_a_body_that_rights_itself_is_stuck_only_past_its_budget():
  """The `stuck` death for a body that gets back up (#387): not at the
  rover's `TOPPLE_HOLD_S`, but once its get-up's budget has run out, and in
  words that say it fell and could not get up."""
  body = StubBody()
  body.rights_itself, body.stuck_after_s = True, 4.0
  life = stub_life(body=body, mortal=True)
  body.attitude = (math.cos(0.7), math.sin(0.7), 0.0, 0.0)   # 80 deg over
  body.run(body.hold_routine(3.0))
  assert life.dead is None, "past TOPPLE_HOLD_S, inside the get-up's budget"
  body.run(body.hold_routine(1.5))
  assert life.dead is not None and life.dead["cause"] == "stuck"
  assert "fell and could not get up" in life.dead["why"]


def test_the_rover_is_still_knocked_over_at_its_own_hold():
  body = StubBody()
  life = stub_life(body=body, mortal=True)
  body.attitude = (math.cos(0.7), math.sin(0.7), 0.0, 0.0)
  body.run(body.hold_routine(2.5))
  assert life.dead is not None and "knocked over" in life.dead["why"]


# ---- the planner's sizes and the reflexes ------------------------------------------


def _outline(model, data, root) -> np.ndarray:
  """Points on every geom of the robot, world frame."""
  pts = []
  for g in sorted(robot_geoms(model, "pluggybot")):
    s, p, r = model.geom_size[g], data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
      for c in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
        pts.append(p + r @ (c * s))
    else:
      rad, half = s[0], (s[1] if t != mujoco.mjtGeom.mjGEOM_SPHERE else 0.0)
      for e in (-1, 1):
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
          pts.append(p + r[:, 2] * half * e + rad * np.array([math.cos(a), math.sin(a), 0]))
  return np.array(pts)[:, :2] - data.xpos[root, :2]


def test_the_inflation_and_the_peer_disc_are_the_quadrupeds_own():
  """Measured off the body standing and lying (#387): the map's inflation
  covers its HALF-WIDTH on a diagonal (it walks forward through a door), and
  the disc round another quadruped covers that one's whole outline, lying
  included, plus this one's half-width -- and a fallen one's, from the
  middle of its footprint (worst of 20 random drops, SimNotes)."""
  m = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
  d = mujoco.MjData(m)
  root = m.body("pluggybot").id
  res, q = 0.05, qb.QuadMission
  widths, reach = [], []
  for key in (0, 1):                               # stand, lie
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    pts = _outline(m, d, root)
    widths.append(np.abs(pts[:, 1]).max())
    reach.append(np.hypot(pts[:, 0], pts[:, 1]).max())
  half = max(widths)
  assert half <= q.INFLATION_CELLS * res / math.sqrt(2), "a diagonal wall touches it"
  assert max(reach) + half <= q.OTHER_ROBOT_CELLS * res
  assert 0.63 + half <= q.DOWN_ROBOT_CELLS * res
  # ...and the corridor the peer channel watches is the body's
  assert half < q.PEER_STOP_HALF_M and q.PEER_CLEARANCE_M > 0.21


def test_the_front_stop_reads_the_corridor_ahead_not_a_cone():
  """A wall ALONGSIDE, 0.11-0.16 m out, fired the rover's 0.35 rad cone every scan
  at the quadruped's range, and it backed 2.3 m into a corner (#387)."""
  me = SimpleNamespace(FRONT_STOP_RANGE=qb.QuadMission.FRONT_STOP_RANGE,
                       FRONT_HALF_M=qb.QuadMission.FRONT_HALF_M)
  xs = np.linspace(0.05, 1.5, 80)
  wall = np.arctan2(-0.14, xs), np.hypot(xs, 0.14)          # alongside, to the right
  assert not qb.QuadMission._front_blocked(me, *wall)
  assert nav.Navigator._front_blocked(me, *wall), "the premise: the cone fires"
  ahead = np.array([0.0]), np.array([0.40])
  assert qb.QuadMission._front_blocked(me, *ahead)
  # ...it fires before the body's front-most point meets what is ahead --
  # the stowed arm's fork, not the nose (#405: at 0.45 the fork met the
  # wall first) -- and stops short of the clearance the planner grants (a
  # waypoint on the inflation's 0.35 m is dropped 0.08 m out; the LIDAR is
  # behind the torso's centre on the rear mast), or every waypoint along a
  # wall fires it
  lidar_behind = CHOSEN.torso[0] - 0.06
  m = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)                      # standing, stowed
  mujoco.mj_forward(m, d)
  front = _outline(m, d, m.body("pluggybot").id)[:, 0].max()
  assert front + lidar_behind < qb.QuadMission.FRONT_STOP_RANGE
  clearance = qb.QuadMission.INFLATION_CELLS * 0.05 + nav.WAYPOINT_REACHED_M
  assert qb.QuadMission.FRONT_STOP_RANGE < clearance + lidar_behind


def test_a_press_steps_away_from_where_it_was_touched():
  assert qb.retreat_from((0.2, 0.0), 0.3, 0.15) == (-0.3, 0.0)       # the nose
  assert qb.retreat_from((-0.3, 0.1), 0.3, 0.15) == (0.3, 0.0)       # hind knees
  assert qb.retreat_from((0.0, -0.18), 0.3, 0.15) == (0.0, 0.3)      # right flank
  assert qb.retreat_from((0.05, 0.18), 0.3, 0.15) == (0.0, -0.3)     # left flank


def test_the_dead_band_is_walked_out_of_and_a_slow_turn_is_a_pivot():
  """#378: the policy walks nothing under ~0.2 m/s and barely turns under
  ~0.2 rad/s. A crawl with a turn to make is a pivot (raised, it swept arcs
  into walls); a crawl straight on is raised; commands stay inside what the
  policy was trained on."""
  assert qb.command_for(0.0, 0.0, 0.0) == qb.STILL
  assert qb.command_for(0.1, 0.0, 0.0) == (qb.V_MIN, 0.0, 0.0)
  assert qb.command_for(0.1, 0.0, 1.2) == (0.0, 0.0, qb.W_MAX)
  assert qb.command_for(0.0, 0.0, 0.1) == (0.0, 0.0, qb.W_MIN)
  assert qb.command_for(-0.05, 0.0, 0.0) == (-qb.V_MIN, 0.0, 0.0)
  assert qb.command_for(2.0, 0.9, 0.0) == (qb.VX_RANGE[1], qb.VY_MAX, 0.0)


def test_progress_is_read_along_the_route_on_legs():
  """A detour round a wall is progress (#387): the straight line to the
  goal grows while the route shrinks."""
  me = SimpleNamespace(pose=(0.0, 0.0, 0.0), PROGRESS_ALONG_ROUTE=True)
  route = [(1.0, 0.0), (1.0, 2.0)]
  assert nav.Navigator._left(me, 5.0, route, 0.0, 2.0) == pytest.approx(1 + 2 + 1)
  me.PROGRESS_ALONG_ROUTE = False
  assert nav.Navigator._left(me, 5.0, route, 0.0, 2.0) == 5.0


def test_the_explore_drops_frontiers_off_its_own_floor():
  """Frontiers A* cannot reach are dropped before any A*, and none is
  blacklisted -- they may be reachable next round (#387)."""
  g = OccupancyGrid(0.0, 0.0, 6.0, 3.0, resolution=0.05)
  g.grid[:] = -5.0
  g.grid[:, 58:62] = 5.0                           # a wall, x 2.9..3.1
  g.grid[20:40, 100:110] = 0.0                     # unknown, beyond it
  bl = set()
  path, status = plan(g, (1.0, 1.5, 0.0), bl, own_component=True)
  assert (path, status) == (None, "no-reachable") and not bl
  path, status = plan(g, (1.0, 1.5, 0.0), bl)
  assert status == "no-reachable" and bl, "the rover's explorer tries and blacklists"


def test_the_depth_layer_is_what_the_lidar_cannot_see(quad_world):
  """Only near points, only three frames' evidence, never beside a wall the
  LIDAR maps -- the three rules a stray far pixel shut a door without."""
  body = quad(quad_world)
  try:
    m = body.mission
    m.start_at(1.5, 0.5, 0.0)
    m.low[:] = 0.0
    m.grid.grid[:] = -5.0
    box = np.array([[1.0, 0.0, 0.2]])            # a couch's face, 1 m ahead
    far = np.array([[2.5, 0.0, 0.2]])            # a pixel past the range
    cell = m.grid.world_to_cell(m.pose[0] + 1.0, m.pose[1])
    for k in range(3):
      m._fold_low(np.vstack([box, far]))
      assert (m.low[cell[1], cell[0]] > qb.LOW_OCC) == (k == 2), k
    far_cell = m.grid.world_to_cell(m.pose[0] + 2.5, m.pose[1])
    assert m.low[far_cell[1], far_cell[0]] == 0.0
    # ...beside a mapped wall, nothing
    m.low[:] = 0.0
    m.grid.grid[cell[1], cell[0] + 2] = 5.0
    m._walls_t = None
    for _ in range(3):
      m._fold_low(box)
    assert m.low[cell[1], cell[0]] == 0.0
  finally:
    body.close()


def test_a_replan_from_the_same_spot_is_the_last_plan(quad_world, monkeypatch):
  """At a stand-in the waypoints ran out where the robot stood, and the
  drive replanned every step: 4 190 plans in 90 s of a pair (#387)."""
  body = quad(quad_world)
  try:
    calls = []
    real = nav.Navigator._plan_to
    monkeypatch.setattr(nav.Navigator, "_plan_to",
                        lambda self, x, y: calls.append(1) or real(self, x, y))
    m = body.mission
    m.start_at(1.5, 0.5, 0.0)
    m._plan_to(2.0, 0.5)
    m._plan_to(2.0, 0.5)
    assert len(calls) == 1
    m._plan_to(2.5, 0.5)                           # another goal
    assert len(calls) == 2
  finally:
    body.close()


# ---- energy ------------------------------------------------------------------------


def test_the_pack_draws_what_the_drivers_do(quad_world):
  """#377's table, off the actuators: the electronics lying, ~54 W standing,
  and the dock's charger in (216 W)."""
  body = quad(quad_world)
  try:
    body.start_at(1.5, 0.5, 0.0)
    pack = body.pack(20.0, 1.0)
    watts = []
    for _ in range(250):
      body.run(body.hold_routine(0.002))
      watts.append(pack.power_draw(body.data))
    assert 40.0 < np.mean(watts) < 75.0, np.mean(watts)
    assert pack.charge_w == pytest.approx(5.0 * 43.2)
    # the computer, the LIDAR, the D435 and fourteen drivers' standby: the
    # legs' twelve and the arm's two (#405)
    assert pack.base_w == pytest.approx(6.0 + 1.15 + 2.0 + 14 * 0.48)
  finally:
    body.close()


# ---- the prompt in its own words, and upkeep off ------------------------------------


def test_upkeep_is_said_only_where_there_is_one():
  """#387: a world with no appetite has no upkeep to fail and no hours of
  paid work to price a heart in; a world WITH one reads the rule byte for
  byte (`guarded`'s experiments fly with it)."""
  assert ov.mortal_rule(True, "rover") == ov.MORTAL_RULE
  for body in ("rover", "quadruped"):
    text = ov.mortal_rule(False, body).lower()
    assert "upkeep" not in text and "hours of work" not in text
    assert "heartprice" in text and "buy_heart" in text
  assert "upkeep you cannot pay" in ov.mortal_rule(True, "quadruped")


def test_every_rule_a_quadruped_reads_is_in_its_own_words():
  """Where it charges is its DOCK, how it dies is a fall it cannot get up
  from, and nothing it reads offers it a lift or a wheel. Its fork is its
  arm's (#405). The rack's one mention is the constitution's: where the
  arm takes its tools."""
  from pluggybot.economy.ledger import Ledger
  from pluggybot.mind.thoughts import ThoughtFiles
  boss = ov.build(QUAD_HOME, None, enabled=True, client=object(), ledger=Ledger(),
                  thoughts=ThoughtFiles.open(None, body="quadruped"),
                  robot_name="Luca", mortal=True, hearts=True, autonomous=True,
                  origin="unseeded", standing_orders=True, others=("Rowan",))
  text = "\n".join(b["text"] for b in boss.system)
  assert "dock" in text and "a fall you cannot get up from" in text
  assert "THE LAB" in text and "when a foot presses it" in text   # #403
  for word in ("two-wheeled", "wheel", "chassis", "the mast", "set_lift", "lift",
               "hub's charge bay", "upkeep you cannot pay"):
    assert word not in text, word
  import re
  assert len(re.findall(r"\brack\b", text)) == 1
  assert "tools on the rack beside your dock" in text


def test_the_constitution_is_told_its_body_and_refuses_one_that_is_not_the_rovers():
  c = constitutions.load("default")
  legs = constitutions.for_body(c, "quadruped")
  assert legs.name == "default" and legs.sha != c.sha
  assert "two-wheeled" not in legs.text and "four-legged" in legs.text
  assert legs.text.split("\n\n")[1:] == c.text.split("\n\n")[1:], "only the body moves"
  assert constitutions.for_body(c, "rover") is c
  with pytest.raises(ValueError):
    constitutions.for_body(constitutions.Constitution.of("inline", "Be kind."), "quadruped")


def test_upkeep_off_is_a_configuration_no_unpaid_death_and_no_metabolism_on_the_wire(
    quad_world):
  """Mortal, broke and with no appetite: a day of nothing costs no heart,
  and the header advertises no hunger for a robot that has none."""
  from pluggybot.telemetry.recorder import FrameBuilder
  body = StubBody()
  life = stub_life(body=body, mortal=True)
  assert life.metabolism is None and life.ledger is None or life.ledger.balance() == 0
  body.run(body.hold_routine(30.0))
  assert life.dead is None and not life.deaths
  data = mujoco.MjData(quad_world)
  mujoco.mj_forward(quad_world, data)
  builder = FrameBuilder(quad_world, data, model_name=QUAD_HOME)
  assert builder.header()["hungerStates"] == []


def test_the_build_identity_names_the_body_and_hashes_its_policies():
  from pluggybot.evaluation.record import body_identity, build_identity, world_hash
  ident = body_identity(QUAD_HOME)
  assert ident["name"] == "quadruped"
  assert set(ident["policies"]) == {"walk", "getup"}
  assert all(len(s) == 64 for s in ident["policies"].values())
  assert body_identity("home") is None
  assert world_hash(QUAD_HOME) != world_hash("home")
  header = build_identity(QUAD_HOME, arm="autonomous", commit="x", hashes={},
                          body=ident)
  assert header["body"] == ident
  assert "body" not in build_identity("home", arm="guarded", commit="x", hashes={})


def test_serve_refuses_a_tool_errand_on_a_body_whose_arm_takes_no_tool():
  import subprocess
  import sys
  out = subprocess.run([sys.executable, "scripts/serve.py", "--world", "home",
                        "--body", "quadruped", "--errand", "draw"],
                       capture_output=True, text=True, timeout=120)
  assert out.returncode == 2 and "takes no tool yet" in out.stderr


# ---- docking in the house, flown ------------------------------------------------


@pytest.mark.slow
@pytest.mark.endurance(when=(
  "src/pluggybot/legs/", "models/quadruped", "models/home_world.xml",
  "src/pluggybot/navigator.py", "src/pluggybot/mapping/", "src/pluggybot/behavior/",
  "src/pluggybot/perception/", "src/pluggybot/power.py", "src/pluggybot/tick.py"))
def test_a_low_quadruped_walks_to_its_dock_and_charges_on_it(tmp_path):
  """The served robot's survival path, composed: stood at its commissioned
  start below the floor, it walks the house to its dock, finds the board,
  walks in, lies down onto the pins and charges -- the loop's own charge
  path on the real body. Each link is pinned fast (test_dock.py: the stop,
  the lie-down onto the pins, the criterion while lying, the pose off the
  board); only this flies them together. Stops once the pins have conducted
  for a second. Shown to fail by skipping the walk-in in
  `QuadMission.dock_routine`: every attempt stands short, and it never
  docks."""
  from pluggybot.lifecycle import run_demo
  seen: dict = {}
  states: list = []

  def docked(life) -> bool:
    if life.state not in states:
      states.append(life.state)
    if life.state == "CHARGE" and life.charging_now:
      seen.setdefault("t", float(life.data.time))
      seen.setdefault("wh", life.battery.energy_wh)
      seen.update(posture=life.body.posture, why=life.body.mission.last_charge,
                  gained=life.battery.energy_wh - seen["wh"])
      return float(life.data.time) - seen["t"] >= 1.0
    return False

  r = run_demo(view=False, realtime=False, world=QUAD_HOME, errand="none",
               pack="demo", battery_fraction=0.15, max_sim_time=120.0,
               overseer=False, thoughts_root=str(tmp_path / "thoughts"),
               ledger_state=str(tmp_path / "ledger.json"),
               board_state=str(tmp_path / "boards.json"),
               spend_state=str(tmp_path / "spend.json"), stop_when=docked)
  assert 0.15 * world_config(QUAD_HOME)["battery_wh"] < world_config(QUAD_HOME)[
    "low_battery_wh"], "the premise: it starts below the floor"
  assert "t" in seen, f"never charged on the dock: {states}, {r['sim_time']:.0f} sim s"
  assert states[-2:] == ["GO_CHARGE", "CHARGE"], states
  assert seen["posture"] == qb.LYING
  assert seen["why"]["attempts"][-1]["why"] == "docked", seen["why"]
  assert seen["gained"] > 0.03, "lying on the pins and nothing flowed"
  assert r["dead"] is None and not r["deaths"]
