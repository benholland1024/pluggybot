"""The humanoid pair served in the house (issue #559, stage 1): the world it
lives in, a world with nowhere to charge, the two costs of its walk that
grew with the map (#558's levers), and what it shares with the loop. Each
rule on the cheapest world that fails for its reason: the stub's
bookkeeping, the house compiled, one humanoid's mission in it. Its restart
-- across the change of body, and its own -- is `test_continuation.py`'s.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.body import is_humanoid
from pluggybot.evaluation import identity
from pluggybot.evaluation.identity import body_identity, world_hash
from pluggybot.humanoid import mission as hmi
from pluggybot.lifecycle import (HUMANOID_HOME, NO_CHARGER, HubLifecycle,
                                 world_config, world_for, world_targets)
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.navigator import Navigator
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import (BODIES, CHASSIS_RGBA, FIRST, SECOND, SECOND_CHASSIS_RGBA,
                             world_spec)
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _house(pair: bool = False):
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], cfg["start2"][:2] if pair else None).compile()
  return cfg, model, mujoco.MjData(model)


def _humanoid(near_field: bool = True) -> HubLifecycle:
  """One humanoid in its house, built and stood at the start."""
  cfg, model, data = _house()
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME,
                      battery_wh=cfg["battery_wh"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], near_field=near_field)
  life.body.start_at(*cfg["start"])
  return life


# ---- the world --------------------------------------------------------------------


def test_the_humanoids_house_is_its_own_world_and_the_body_names_it():
  """`$PLUGGY_WORLD=home` is the house with the humanoids in it, the one
  body (#559, stage 5: the quadrupeds' `home_quad` left with them, and
  `quadruped-final` serves it); a world names its body, and another body
  named is refused."""
  assert BODIES == ("humanoid",)
  assert world_for("home", "humanoid") == HUMANOID_HOME == world_for(HUMANOID_HOME, "humanoid")
  assert world_for("home") == HUMANOID_HOME == world_for(HUMANOID_HOME)
  for world, body in (("home", "quadruped"), (HUMANOID_HOME, "quadruped"),
                      ("home_quad", None)):
    with pytest.raises(ValueError):
      world_for(world, body)
  with pytest.raises(ValueError):
    world_config("home_quad")
  cfg = world_config(HUMANOID_HOME)
  assert cfg["body"] == "humanoid" and cfg["model_name"] == HUMANOID_HOME
  assert "module" not in world_targets(HUMANOID_HOME, procedures=True), "no rack, no carry"


def test_the_house_holds_two_humanoids_by_name_painted_and_meeting_each_other():
  """The quadrupeds' rule for a pair (`humanoid.world.PAIR_BITS`): each
  robot meets the other's body and never its own; the second in its own
  livery; the quadruped's dock and rack gone with it, the house's tags
  there."""
  _, model, _ = _house(pair=True)
  for h in (FIRST, SECOND):
    assert is_humanoid(model, h)
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_rack") < 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dock") < 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "tagmat44") >= 0

  def solid(root):
    return sorted(g for g in robot_geoms(model, root)
                  if model.geom_contype[g] or model.geom_conaffinity[g])
  one, two = solid(FIRST.root), solid(SECOND.root)

  def meets(a, b):
    return bool((model.geom_contype[a] & model.geom_conaffinity[b])
                or (model.geom_contype[b] & model.geom_conaffinity[a]))
  assert all(meets(a, b) for a in one for b in two)
  assert not any(meets(a, b) for a in one for b in one)
  def painted(root, rgba):
    return any(np.allclose(model.geom_rgba[g], rgba) for g in robot_geoms(model, root))
  assert painted(FIRST.root, CHASSIS_RGBA) and not painted(FIRST.root, SECOND_CHASSIS_RGBA)
  assert painted(SECOND.root, SECOND_CHASSIS_RGBA) and not painted(SECOND.root, CHASSIS_RGBA)


def test_the_header_names_the_humanoid_and_hashes_its_policies(monkeypatch):
  """The build's identity (#132) says which body flew and the sha of each
  policy it runs, and the world's hash reads the body's files too -- a
  retrained walker is a new regime."""
  who = body_identity(HUMANOID_HOME)
  assert who["name"] == "humanoid" and set(who["policies"]) == {"stand", "walk", "getup"}
  assert all(len(s) == 64 for s in who["policies"].values())
  header = identity.build_identity(HUMANOID_HOME, arm="autonomous", commit="x", hashes={},
                                   body=who)
  assert header["body"] == who
  assert "body" not in identity.build_identity(HUMANOID_HOME, arm="autonomous", commit="x",
                                               hashes={})
  with_body = world_hash(HUMANOID_HOME)
  monkeypatch.setattr(identity, "BODY_FILES", {"humanoid": ()})
  assert world_hash(HUMANOID_HOME) != with_body


# ---- nowhere to charge --------------------------------------------------------------


def test_the_humanoids_house_charges_at_its_station():
  """Stage 2: the humanoids' house has its charging station as its dock
  (`activity.station`, where the quadrupeds' dock lies), so the loop has
  somewhere to send it."""
  from pluggybot.humanoid.world import station_pose
  assert world_config(HUMANOID_HOME)["dock"] == station_pose()
  assert stub_life(world=HUMANOID_HOME).charger


def test_a_world_with_nowhere_to_charge_never_goes_and_says_why():
  """A world with nowhere this body charges (the humanoids' house before
  its station): a charge ends at once, said, and nothing sends it to one --
  a world with a charger and this pack below its reserve would have."""
  life = stub_life(world=HUMANOID_HOME)
  life.charger = False
  life.low_battery_wh = 5.0
  life.battery.energy_wh = 1.0
  assert not life.needs_charge
  t0 = float(life.data.time)
  assert life.body.run(life.go_charge_routine()) is False
  assert life.charge_failure == NO_CHARGER and float(life.data.time) == t0


def test_an_explore_the_run_ended_does_not_say_the_battery_did():
  """A run's end is not a low pack: in a world with nowhere to charge every
  explore ends there, and its line said "battery low" at a full pack."""
  life = stub_life(world=HUMANOID_HOME)
  life.max_sim_time = 0.0
  said = []
  life.say_hooks.append(lambda t, line: said.append(line))
  assert life.body.run(life.explore_routine(budget=60.0)) == "time"
  assert said[-1].endswith("EXPLORE: the run's time is up")


# ---- #558's two levers ------------------------------------------------------------------


def test_a_humanoid_holds_its_plan_as_the_quadruped_does():
  """A plan asked for again within `REPLAN_HOLD_S`, from within
  `REPLAN_MOVED_M`, is the last one -- the Navigator's now, so both bodies'
  (#558, #559): the humanoid's drive replanned every step its waypoints ran
  out short of the goal, 3 115 plans in 100 s of a pair."""
  assert hmi.HumanoidMission._plan_to is Navigator._plan_to
  life = _humanoid()
  m = life.body.mission
  made = []
  real = m._plan_round
  m._plan_round = lambda *a, **kw: made.append(1) or real(*a, **kw)
  try:
    goal = (life.body.pose[0] + 1.0, life.body.pose[1])
    m._plan_to(*goal)
    m._plan_to(*goal)
    assert len(made) == 1, "planned again at once"
    life.data.time += Navigator.REPLAN_HOLD_S
    m._plan_to(*goal)
    assert len(made) == 2, "held past its time"
  finally:
    life.body.close()


def test_a_humanoid_grows_the_walls_round_itself_never_the_whole_map():
  """The camera's layer is kept off what the LIDAR maps, grown -- over the
  window its points can land in (`LOW_RANGE_M`), as the quadruped's: grown
  over the whole grid every depth frame, it cost a pair 3.6 s of 155 in
  100 sim s, and it grows with the map (#558)."""
  life = _humanoid()
  m = life.body.mission
  try:
    g = m.grid
    g.grid[:] = -5.0
    x, y, _ = life.body.pose
    near = g.world_to_cell(x + 1.0, y)
    far = g.world_to_cell(x + hmi.LOW_RANGE_M + 2.0, y)
    for ix, iy in (near, far):
      g.grid[iy, ix] = OCC_THRESH + 5.0
    m._walls_t = None
    walls = m._walls()
    assert walls[near[1], near[0] - hmi.LOW_WALL_CELLS], "the near wall, grown"
    assert not walls[far[1], far[0] - 1], "a wall out of the camera's reach, grown"
  finally:
    life.body.close()


def test_a_thin_thing_goes_into_the_layer_whatever_floor_its_cell_shows():
  """A cell a frame's points hit is no floor in that frame (#559): the
  garden light's 4 cm pole, under the LIDAR's plane, fills part of its 5 cm
  cell and the floor round it the rest, and each floor point voted it down
  -- a walk into it stood there all day. Three frames of one point on it
  and ten on the floor beside it, and it is in the planner's layer."""
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    g = m.grid
    m.low[:] = 0.0
    x, y, yaw = m.pose
    ix, iy = g.world_to_cell(x + math.cos(yaw), y + math.sin(yaw))
    cx, cy = g.cell_to_world(ix, iy)                    # the cell's middle, in the world
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = c * (cx - x) + s * (cy - y), -s * (cx - x) + c * (cy - y)
    ring = [(ax + 0.015 * math.cos(k), ay + 0.015 * math.sin(k), 0.0) for k in range(10)]
    pts = np.array([(ax, ay, 0.5), *ring])
    for _ in range(3):
      m._fold_low(pts)
    assert m.low[iy, ix] >= hmi.LOW_OCC, m.low[iy, ix]
  finally:
    life.body.close()


# ---- what it shares with the loop and the other robot --------------------------------


def test_the_pair_drops_each_other_from_their_scans_and_shares_their_frames():
  """Each humanoid's LIDAR and depth camera keep the other's body out of its
  map (`Body.know_peer`), as the quadrupeds' do; and the loop's near field
  and peer channel read the frames its walk lays its layer from, at the
  camera's own 5 Hz (`Body.depth_camera`), never a render of their own."""
  from pluggybot.pair import build_pair
  lives = build_pair(HUMANOID_HOME, near_field=True)
  try:
    for life, other in zip(lives, reversed(lives)):
      m = life.body.mission
      theirs = set(robot_geoms(life.model, other.root))
      assert theirs <= m.lidar._other_geoms
      assert other.root in m.depth._other_roots
      assert life.depth_camera.period == hmi.DEPTH_PERIOD
    life = lives[0]
    life.body.start_at(*world_config(HUMANOID_HOME)["start"])
    renders = []
    real = life.body.mission.depth.frame
    life.body.mission.depth.frame = lambda d: renders.append(1) or real(d)
    life.body.run(life.body.hold_routine(1.0))
    assert len(renders) == pytest.approx(1.0 / hmi.DEPTH_PERIOD, abs=1)
    assert life.near_field_frames >= 4
  finally:
    for life in lives:
      life.body.close()


def test_a_bowed_humanoid_renders_nothing_for_its_layer_and_lays_no_frame_taken_so():
  """Bowed (a reach's posture tilts the pelvis) it takes no depth frame for
  its planner's layer, as before the near field shared them -- and a frame
  the near field asked for then is never folded into the layer once it is
  level again: a reading goes into a map only while the body is LEVEL."""
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    renders = []
    real = m.depth.frame
    m.depth.frame = lambda d: renders.append(1) or real(d)
    m.level = lambda: False
    life.body.run(life.body.hold_routine(1.0))
    assert renders == [], "rendered while bowed, with nothing asking"
    assert m.latest_depth() is not None and renders == [1]   # the near field's ask
    m.level = lambda: True
    low = m.low.copy()
    m._look_step()
    assert np.array_equal(m.low, low), "folded a frame taken bowed"
  finally:
    life.body.close()


def test_a_humanoids_true_death_forgets_its_plan_and_its_walls():
  """`Body.forget_world` (#419) leaves nothing of the map behind: the plan
  it held and the walls it grew, as the quadruped's."""
  life = _humanoid()
  m = life.body.mission
  try:
    m._plan_memo = ((1.0, 1.0), 0.0, (0.0, 0.0), [(1.0, 1.0)], None)
    m._walls_t, m._walls_mask = 1.0, np.ones_like(m.grid.grid, dtype=bool)
    life.body.forget_world()
    assert m._plan_memo is None and m._walls_mask is None and m._walls_t is None
  finally:
    life.body.close()


def test_a_humanoid_in_the_viewer_is_shown_and_paced():
  """`two_robots.py --view`: the viewer and the pacing reach its walk
  (`body_for`), or the window never moved and the day ran unpaced."""
  from pluggybot.body import body_for
  _, model, data = _house()
  seen = object()
  body = body_for(model, data, viewer=seen, realtime=True)
  try:
    assert body.mission.viewer is seen and body.mission.realtime
  finally:
    body.close()


# ---- making way (`humanoid.way`) -----------------------------------------------------


def _open_floor(m, x, y, half=2.0):
  """The map round (x, y) seen and free, `half` m each way: a synthetic room."""
  g = m.grid
  g.grid[:] = 0.0
  x0, y0 = g.world_to_cell(x - half, y - half)
  x1, y1 = g.world_to_cell(x + half, y + half)
  g.grid[y0:y1, x0:x1] = -5.0


def test_a_humanoid_rests_standing_once_it_has_stood_still():
  """A humanoid rests STANDING (#559): no motion command for the
  quadruped's reflex's 8.6 s, no walk of its own, no hand out and on its
  feet -- what a robot in the other's way is asked aside by (#415). It
  never rested, and a pair jammed a door for 500 s of a minded day. The
  clock rides a restart."""
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    m.last_motion_t = float(life.data.time)
    assert not life.body.resting
    life.data.time += REST_S
    assert life.body.resting
    m.driving = 1
    assert not life.body.resting, "inside a walk of its own"
    m.driving = 0
    m.reach.phase["right"] = "hold"
    assert not life.body.resting, "a hand out"
    m.reach.phase["right"] = "rest"
    m.legs = "getup"
    assert not life.body.resting, "getting up"
    m.legs = "walk"
    assert life.body.resting
    kept, _ = m.kept_state()
    assert kept["lastMotion"] == m.last_motion_t
  finally:
    life.body.close()


def test_a_resting_humanoid_says_yes_to_a_way_it_cuts_and_why_not_otherwise():
  """`Body.make_way`: a spot off the asker's way on floor it has seen; and
  no, with why, while it stands (not yet resting), works or walks."""
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    x, y, _ = life.body.pose
    _open_floor(m, x, y)
    way = [(x - 1.5, y), (x + 1.5, y)]
    m.last_motion_t = float(life.data.time)
    assert not life.body.make_way(way, "r2_pluggybot")
    assert life.body.way_refused == "standing"
    life.data.time += REST_S
    m.driving = 1
    assert not life.body.make_way(way, "r2_pluggybot") and life.body.way_refused == "walking"
    m.driving = 0
    assert life.body.make_way(way, "r2_pluggybot") and life.body.making_way == "r2_pluggybot"
    sx, sy = m._aside["to"]
    # ...measured on the planner's 10 cm lattice, cell to cell: a cell's
    # middle may lie half a cell nearer
    assert abs(sy - y) >= m.aside_clear_m - 0.05, "off the way, by the asker's disc"
    assert not life.body.resting, "stepping aside is no rest"
  finally:
    life.body.close()


def test_a_humanoid_on_a_plate_whose_sign_it_has_just_found_walks_off_it():
  """A humanoid's walks keep off no plate (#559's stage 4; SimNotes, "On
  the lab's plates"): its shock plate is switched off, and the feed and toy
  plates only feed and amuse the mouse. Kept off every pad it knew, as the
  quadruped's planner is (`PlaceWalk.keep_out`), one standing on a pad when
  it found the sign at the pad's far edge planned no walk off it."""
  import math
  from pluggybot.activity import cage
  from pluggybot.home import world as home
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    sx, sy = cage.sign_xy(home.LAB_CAGE_XY, "feed")
    south = -math.pi / 2
    px, py = cage.pad_from_sign(sx, sy, south)
    life.body.start_at(px, py, math.pi / 2)
    _open_floor(m, px, py, half=3.0)
    m.places.see(cage.PLATE_TAGS["feed"], sx, sy, float(life.data.time),
                 view=south, tag_facing=south)
    assert m._plan_to(px, py - 2.5) is not None
  finally:
    life.body.close()


def test_a_humanoid_plugged_in_at_its_station_says_no_to_a_way_it_cuts():
  """Its plug in its port's catch, a resting humanoid says no, as the
  quadruped on its dock does (#559's stage 4): asked aside in the pair's
  recording, it walked off with the plug in its port and pulled it out.
  Said in its own words, and read off the plug, never the lifecycle's
  `Body.docked`, which a restart forgets with the plug still in."""
  from pluggybot.humanoid.way import REST_S
  from pluggybot.lifecycle import WAY_REFUSED
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    x, y, _ = life.body.pose
    _open_floor(m, x, y)
    way = [(x - 1.5, y), (x + 1.5, y)]
    life.data.time += REST_S
    (port,) = m.port_eqs
    life.data.eq_active[port] = 1
    assert life.body.charging() and life.body.resting and not life.body.docked
    assert not life.body.make_way(way, "r2_pluggybot")
    assert life.body.way_refused == "charging" and life.body.making_way is None
    assert "dock" not in WAY_REFUSED["charging"]
    life.data.eq_active[port] = 0
    life.body.docked = True
    assert life.body.make_way(way, "r2_pluggybot"), "held, but nothing in its port"
  finally:
    life.body.close()


def test_a_humanoid_steps_aside_on_its_own_legs_beneath_a_routine_that_stands():
  """The step aside walks beneath whatever routine holds the body still --
  its legs handed to the walker where it rests -- to the spot, and ends
  there. ⚠ ITS FRONT STOP BACKS IT OFF, as a walk's does, and never ends
  it: resting in a doorway, its stop re-armed by every scan, the
  quadruped's rule ended 3 steps of 6 before they began."""
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    x, y, _ = life.body.pose
    _open_floor(m, x, y)
    life.data.time += REST_S
    assert life.body.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")
    m.backoff_until = float(life.data.time) + 0.5
    back = m._aside_command(hmi.STILL, float(life.data.time))
    assert back[0] < 0.0 and m._aside is not None, "the reflex, and the step goes on"
    m.backoff_until = 0.0
    said = []
    life.say_hooks.append(lambda t, line: said.append(line))
    life.body.run(life.body.hold_routine(12.0))
    rec = life.body.last_aside
    assert rec is not None and rec["why"] == "aside", rec
    moved = math.dist(rec["from"], rec["at"])
    assert moved > 0.5
    assert [ln for ln in said if ln.startswith("MADE WAY")] == [
      f"MADE WAY: stepped {moved:.1f} m aside for r2_pluggybot, whose way I was standing in"]
  finally:
    life.body.close()


def test_a_humanoid_left_bowed_stands_before_it_steps_aside():
  """A bow left held -- a `spot` cut off before its stand -- is no stance
  to walk from: asked aside, its posture comes back to the stand before the
  walker takes its legs, as `to_walk_routine` hands them over (#559
  review). The walker was handed them at the bow."""
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    x, y, _ = life.body.pose
    _open_floor(m, x, y)
    life.body.run(life.body.gaze_routine(0.4))
    assert m.legs == "stand" and m.reach.posture[1] == pytest.approx(0.4), "the premise"
    life.data.time += REST_S
    assert life.body.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")
    handed = []
    life.body.step_hooks.append(lambda: handed.append(
      (float(m.reach.posture[1]), float(m.stand.posture[1])))
      if m.legs == "walk" and not handed else None)
    life.body.run(life.body.hold_routine(4.0))
    assert handed, "it never walked"
    asked, shown = handed[0]
    assert asked == 0.0 and abs(shown) < 0.05, handed
  finally:
    life.body.close()


def test_a_routine_that_takes_the_legs_ends_a_step_aside():
  """The step aside walks beneath a routine that stands still, and one that
  takes the legs for the standing policy -- a reach, a gaze -- ends it (#559
  review): walked on, the aside walked the settle's STILL steps and the legs
  went to the standing policy mid-stride."""
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  m = life.body.mission
  try:
    x, y, _ = life.body.pose
    _open_floor(m, x, y)
    life.data.time += REST_S
    assert life.body.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")
    life.body.run(life.body.hold_routine(1.5))
    assert m._aside is not None and m.legs == "walk", "the premise: under way"
    life.body.run(m.to_stand_routine())
    assert m._aside is None and life.body.last_aside["why"] == "its own move"
    assert m.legs == "stand"
  finally:
    life.body.close()


def test_a_humanoid_at_rest_is_said_standing_never_lying():
  """What a mind and History are told of a humanoid resting in the way:
  standing at rest (#559) -- the quadruped's words, "lying down to rest",
  were false of it, in the other's context, a walk's end and its own
  narration."""
  from types import SimpleNamespace
  from pluggybot.lifecycle import posture
  from pluggybot.humanoid.way import REST_S
  life = _humanoid(near_field=False)
  try:
    life.data.time += REST_S
    assert life.body.resting
    peer = SimpleNamespace(robot_name="Rowan", root="r2_pluggybot", body=life.body,
                           dead=None, down=lambda: False)
    assert posture([peer], "Rowan") == "standing at rest"
    words = life.body.gave_up({"why": "peer", "goal": (1.0, 1.0), "seconds": 2.0,
                               "shortM": 3.0, "peerAt": "here", "peerM": 1.2,
                               "peerRests": True}, "Rowan")
    assert "standing at rest" in words and "lying" not in words, words
    said = []
    life.say_hooks.append(lambda t, line: said.append(line))
    x, y, _ = life.body.pose
    _open_floor(life.body.mission, x, y)
    assert life.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")
    assert said and "resting across" in said[-1] and "lying" not in said[-1], said
  finally:
    life.body.close()


def test_a_humanoids_look_is_taken_from_its_head():
  """`look` (#275) asks the site for a picture from `Body.head_camera`'s
  pose: the humanoid's is its head's Camera Module 3 (`nav_eye`), at a
  person's eye height and facing where the robot faces -- never another
  body's name (#408)."""
  from pluggybot.mind.look import camera_pose
  life = _humanoid()
  try:
    m, d = life.model, life.data
    mujoco.mj_forward(m, d)
    cam = camera_pose(m, d, life.body.head_camera)
    _, _, yaw = life.body.true_pose()
    assert life.body.head_camera == "nav_eye"
    assert 1.2 < cam["pos"][2] < 1.5
    assert cam["forward"][0] * math.cos(yaw) + cam["forward"][1] * math.sin(yaw) > 0.9
  finally:
    life.body.close()


def test_the_death_seam_reads_a_humanoids_tip_off_the_bow_its_fall_watch_reads(monkeypatch):
  """A reach's deepest bow (63 deg) is past the seam's topple (60): the seam
  reads the pelvis off the bow its legs hold (`Body.held_pitch`), which is
  its fall watch's own (`HumanoidMission._held_pitch`)."""
  life = _humanoid(near_field=False)
  try:
    bow = math.radians(63.0)
    life.data.xquat[life.body.root] = (math.cos(bow / 2), 0.0, math.sin(bow / 2), 0.0)
    assert life.down(), "the walker stands upright"
    monkeypatch.setattr(life.body.mission, "_held_pitch", lambda: bow)
    assert life.body.held_pitch == bow and not life.down()
  finally:
    life.body.close()


def test_a_humanoid_keeps_clear_of_its_station_and_with_a_hand_out_is_never_saved():
  """Its station is the frame the loop's standby keeps clear of
  (`HubLifecycle.rack_distance`, its commissioned pose, out into the room);
  and a restart's save waits out a hand's move (`Body.working`), which is a
  routine's, held in no file."""
  from pluggybot.humanoid.world import station_pose
  life = _humanoid()
  try:
    sx, sy, yaw = station_pose()
    r = life.body.rack_prior
    assert (r.x, r.y) == pytest.approx((sx, sy)) and math.cos(r.yaw - yaw) == pytest.approx(-1)
    assert life.rack_distance(sx, sy) < 1e-6 and life.rack_distance(0.0, 0.0) > 3.0
    assert not life.body.working
    life.body.reach.phase["left"] = "move"
    assert life.body.working
  finally:
    life.body.close()


def test_a_mind_judging_what_the_other_humanoid_needs_asks_after_no_tool():
  """A mind's first `other_needs` (#208) asked what was on the other
  humanoid's fork, and the body it asked had none -- the minded day's first
  answer ended it. The need is read off the loop's own state, never a body's
  fork (`carrying` is a vestige of the quadruped's: empty)."""
  from pluggybot.mind import acts
  need, state = acts.need_of(stub_life(world=HUMANOID_HOME))
  assert need == "nothing" and state["carrying"] == ""


def test_the_loop_with_no_mind_goes_to_no_charger_however_low_the_pack():
  """The loop's day in a world with nowhere to charge, on a pack a world
  with a charger would send it charging on: the start, the explore, the
  end, and no charge trip."""
  for charger, trips in ((True, 1), (False, 0)):
    life = stub_life(world=HUMANOID_HOME)
    life.charger = charger
    life.low_battery_wh = 5.0
    life.battery.energy_wh = 1.0
    summary = life.run(start=(0.5, 3.0, 0.0), max_sim_time=60.0)
    assert summary["charge_cycles"] == trips, charger
  assert not summary["stranded"] and summary["deaths"] == [], "sent to no charger, and stranded"
