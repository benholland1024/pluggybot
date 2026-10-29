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

from pluggybot.body import is_humanoid, is_quadruped
from pluggybot.evaluation.identity import body_identity, world_hash
from pluggybot.humanoid import mission as hmi
from pluggybot.lifecycle import (HUMANOID_HOME, NO_CHARGER, QUAD_HOME, HubLifecycle,
                                 world_config, world_for)
from pluggybot.legs.rack import RACK_BODY
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.navigator import Navigator
from pluggybot.perception.lidar import robot_geoms
from pluggybot.procedure import steps
from pluggybot.robot import (BODIES, CHASSIS_RGBA, FIRST, SECOND, SECOND_CHASSIS_RGBA,
                             world_spec)
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


def _house(pair: bool = False):
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"], cfg["start2"][:2] if pair else None,
                     body="humanoid").compile()
  return cfg, model, mujoco.MjData(model)


def _humanoid(near_field: bool = True) -> HubLifecycle:
  """One humanoid in its house, built and stood at the start."""
  cfg, model, data = _house()
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], near_field=near_field)
  life.body.start_at(*cfg["start"])
  return life


# ---- the world --------------------------------------------------------------------


def test_the_humanoids_house_is_its_own_world_and_the_body_names_it():
  """`$PLUGGY_BODY=humanoid` with `$PLUGGY_WORLD=home` is the house with the
  humanoids in it, as `quadruped` is the quadrupeds'; serve.py's `--body`
  offers both."""
  assert set(BODIES) == {"quadruped", "humanoid"}
  assert world_for("home", "humanoid") == HUMANOID_HOME == world_for(HUMANOID_HOME, "humanoid")
  assert world_for("home") == QUAD_HOME
  with pytest.raises(ValueError):
    world_for(QUAD_HOME, "humanoid")
  cfg = world_config(HUMANOID_HOME)
  assert cfg["body"] == "humanoid" and cfg["model_name"] == HUMANOID_HOME
  assert world_config(QUAD_HOME)["zones"] == cfg["zones"], "one house"


def test_the_house_holds_two_humanoids_by_name_painted_and_meeting_each_other():
  """The quadrupeds' rule for a pair (`legs.world.PAIR_BITS`): each robot
  meets the other's body and never its own; the second in its own livery;
  the quadruped's dock and rack gone with it, the house's tags there."""
  _, model, _ = _house(pair=True)
  for h in (FIRST, SECOND):
    assert is_humanoid(model, h) and not is_quadruped(model, h)
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RACK_BODY) < 0
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


def test_the_header_names_the_humanoid_and_hashes_its_policies():
  """The build's identity (#132) says which body flew and the sha of each
  policy it runs -- a retrained walker is a new regime."""
  who = body_identity(HUMANOID_HOME)
  assert who["name"] == "humanoid" and set(who["policies"]) == {"stand", "walk", "getup"}
  assert world_hash(HUMANOID_HOME) != world_hash(QUAD_HOME)


# ---- nowhere to charge --------------------------------------------------------------


def test_a_world_with_nowhere_to_charge_never_goes_and_says_why():
  """Until its cable's station (stage 2) the humanoid's house has no
  charger: a charge ends at once, said, and nothing sends it to one -- a
  quadruped's world with this pack below its reserve would have."""
  life = stub_life(world=HUMANOID_HOME)
  assert not life.charger and stub_life().charger
  life.low_battery_wh = 5.0
  life.battery.energy_wh = 1.0
  assert not life.needs_charge
  t0 = float(life.data.time)
  assert life.body.run(life.go_charge_routine()) is False
  assert life.charge_failure == NO_CHARGER and float(life.data.time) == t0
  # ...and no rack: no tool to stow
  assert steps._rack(life) == {}


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


def test_a_humanoid_is_near_no_rack_and_with_a_hand_out_is_never_saved():
  """Its house has no rack to keep clear of (`HubLifecycle.rack_distance`);
  and a restart's save waits out a hand's move (`Body.working`), which is a
  routine's, held in no file."""
  life = _humanoid()
  try:
    assert life.body.rack_prior is None and life.rack_distance(0.0, 0.0) == math.inf
    assert not life.body.working
    life.body.reach.phase["left"] = "move"
    assert life.body.working
  finally:
    life.body.close()


def test_a_mind_judging_what_the_other_humanoid_needs_asks_after_no_tool():
  """A world with no rack watches no module (`HubLifecycle.module`): a mind's
  first `other_needs` (#208) asked what was on the other humanoid's fork, and
  the body it asked had none -- the minded day's first answer ended it."""
  from pluggybot.mind import acts
  assert stub_life(world=HUMANOID_HOME).module == ""
  assert stub_life().module == "module_lcd", "the quadruped's world watches its LCD"
  life = _humanoid(near_field=False)
  try:
    need, state = acts.need_of(life)
    assert need == "nothing" and state["carrying"] == ""
  finally:
    life.body.close()


def test_the_loop_with_no_mind_goes_to_no_charger_however_low_the_pack():
  """The loop's day in a world with nowhere to charge, on a pack the
  quadrupeds' world would send it charging on: the start, the explore, the
  end, and no charge trip."""
  for world, trips in ((QUAD_HOME, 1), (HUMANOID_HOME, 0)):
    life = stub_life(world=world)
    life.low_battery_wh = 5.0
    life.battery.energy_wh = 1.0
    summary = life.run(start=(0.5, 3.0, 0.0), max_sim_time=60.0)
    assert summary["charge_cycles"] == trips, world
  assert not summary["stranded"] and summary["deaths"] == [], "sent to no charger, and stranded"
  assert summary["module_stowed"] is None, "nothing carried, in a world with no rack"
