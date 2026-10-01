"""Hide and seek (issue #167; on legs, #404): the first two-role game --
claimed per role, each robot its own part, one verdict for both off the
referee's measurement of the world, banked on the winner.

Each rule is pinned as cheaply as it fails: the referee on the quadruped
pair's world with the bodies set where a rule needs them and nothing
stepped; the claims, the queue and the pay on the pair as built; the roles'
verbs on stub bodies; the hider's spot and the seeker's search on a
quadruped's mission over a map drawn here. Games flown whole are
`scripts/solve.py --feature hide_and_seek` (SimNotes, "Hide and seek on
legs").
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.activity import hideseek as hs
from pluggybot.body import KeepClear, StubBody
from pluggybot.economy import scoring
from pluggybot.economy.cadence import default_cadence
from pluggybot.economy.ledger import Account
from pluggybot.economy.tasks import TaskBoard
from pluggybot.legs import body as qb
from pluggybot.legs import game as gm
from pluggybot.legs import world as lw
from pluggybot.legs.model import CHOSEN, lie_qpos
from pluggybot.lifecycle import (GAME_TARGET, QUAD_HOME, hide_and_seek_program,
                                 points_ledger, world_config, world_facts, world_targets)
from pluggybot.mind import overseer as ov
from pluggybot.mission.errand import programmed_errand
from pluggybot.pair import arrange_game, build_pair
from pluggybot.perception.lidar import robot_geoms
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st
from pluggybot.robot import FIRST, SECOND
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

TABLE = scoring.challenge_table()
STEPPER = SimpleNamespace(step=lambda *a: None)


# ---- 1. the referee sees a quadruped ------------------------------------------------


@pytest.fixture(scope="module")
def pair_world():
  return lw.home_spec(first_at=(0.0, -1.0), second_at=(1.0, -1.0)).compile()


def _stand(model, data, seeker, hider):
  lw.stand(model, data, FIRST.prefix, *seeker, 0.0)
  lw.stand(model, data, SECOND.prefix, *hider, 0.0)
  mujoco.mj_forward(model, data)


def _lie(model, data, prefix, x, y, z=0.12):
  """A robot lying on its belly at (x, y): its folded legs, its torso low."""
  lw.stand(model, data, prefix, x, y, 0.0)
  q = int(model.jnt_qposadr[model.body_jntadr[model.body(f"{prefix}pluggybot").id]])
  data.qpos[q + 2] = z
  j = int(model.jnt_qposadr[model.joint(f"{prefix}FL_hip_abd").id])
  data.qpos[j:j + 12] = lie_qpos(CHOSEN)
  mujoco.mj_forward(model, data)


def _first_hits(ref, model, data):
  """Where each of the referee's rays meets something first, as the
  rover's referee cast them: the seeker's root body left out, no more."""
  eye = data.site_xpos[ref.seeker_eye]
  vec = data.geom_xpos[ref.hider_gids] - eye
  dirs = vec / np.linalg.norm(vec, axis=1)[:, None]
  n = len(dirs)
  hit, far = np.zeros(n, np.int32), np.zeros(n)
  mujoco.mj_multiRay(model, data, eye, np.ascontiguousarray(dirs.reshape(-1)), None, 1,
                     ref.seeker_bid, hit, far, None, n, mujoco.mjMAXVAL)
  return hit


def test_the_referee_resolves_a_quadruped_pair(pair_world):
  """The rover's names are gone (a `chassis` geom, a `lidar` site on the
  root): the eye is the seeker's LIDAR on its rear mast, and the hider is
  every geom of its body -- what the quadruped has."""
  ref = hs.HideAndSeek(pair_world, hider=SECOND, seeker=FIRST)
  m = pair_world
  assert m.site(ref.seeker_eye).name == FIRST.el("lidar")
  assert m.body(int(m.site_bodyid[ref.seeker_eye])).name == FIRST.el("lidar_mast")
  assert set(ref.hider_gids.tolist()) == robot_geoms(m, SECOND.root)
  assert ref.seeker_root == FIRST.root


def test_it_sees_any_part_of_the_hider_and_never_its_own_body(pair_world):
  """Rays from the seeker's LIDAR to every geom of the hider: a mast or a
  leg in view is a sighting, a wall between is not, and the seeker's own
  body is no wall -- 0.51 m up on the rear mast, the eye looks down across
  its own stowed arm."""
  m, d = pair_world, mujoco.MjData(pair_world)
  ref = hs.HideAndSeek(m, hider=SECOND, seeker=FIRST)
  _stand(m, d, (0.0, -1.0), (1.4, -1.0))           # open floor, 1.4 m apart
  assert ref._line_of_sight(m, d)
  # ...where the rover's one ray, at the torso, met the seeker's own arm
  eye = d.site_xpos[ref.seeker_eye]
  v = d.xpos[ref.hider_bid] + (0.0, 0.0, 0.1) - eye
  one = np.zeros(1, np.int32)
  mujoco.mj_ray(m, d, eye, v / np.linalg.norm(v), None, 1, ref.seeker_bid, one)
  assert int(one[0]) not in ref._hider_set, "the premise: one ray at the torso"
  # lying 0.8 m in front: EVERY ray meets the seeker first, and goes on
  lw.stand(m, d, FIRST.prefix, 0.0, -1.0, 0.0)
  _lie(m, d, SECOND.prefix, 0.8, -1.0)
  mine = robot_geoms(m, FIRST.root)
  assert all(int(g) in mine for g in _first_hits(ref, m, d)), "the premise"
  assert ref._line_of_sight(m, d)
  # the living room's west wall between them, 1.2 m apart
  _stand(m, d, (-2.6, 0.6), (-1.4, 0.6))
  assert not ref._line_of_sight(m, d)


def test_a_find_needs_the_seeking_reach_and_sight_and_a_seek_run_out_is_the_hiders(pair_world):
  """Within `FIND_WITHIN_M` and in sight, once the head start is spent, is
  the seeker's; never during the head start; and out of reach to the end
  of the seeking, the hider's. No ray is cast beyond reach."""
  m, d = pair_world, mujoco.MjData(pair_world)
  ref = hs.HideAndSeek(m, hider=SECOND, seeker=FIRST, head_start_s=5.0, seek_s=10.0)
  _stand(m, d, (0.0, -1.0), (1.4, -1.0))
  ref.start(0.0)
  d.time = 2.0
  ref.sense(m, d)
  assert ref.phase == "hiding" and ref.over_at is None
  d.time = 6.0
  ref.sense(m, d)
  assert ref.phase == "found" and ref.flags["winner"] == "seeker"
  assert ref.flags["foundAtS"] == 6.0 and ref.measurements()["played"]
  far = hs.HideAndSeek(m, hider=SECOND, seeker=FIRST, head_start_s=5.0, seek_s=10.0)
  _stand(m, d, (0.0, -1.0), (1.6, -1.0))
  far.start(0.0)
  calls = []
  far._line_of_sight = lambda *a: calls.append(1) or True
  for t in (6.0, 14.9):
    d.time = t
    far.sense(m, d)
    assert far.phase == "seeking"
  d.time = 15.0
  far.sense(m, d)
  assert far.phase == "over" and far.flags["winner"] == "hider" and calls == []


def test_a_game_nobody_begins_is_called_off_and_pays_nobody(pair_world):
  """Roles taken and no role's errand begun: called off at
  `START_WITHIN_S`, no winner, not played -- a claimed job would otherwise
  hold its offer's target, and no game could be offered again."""
  m, d = pair_world, mujoco.MjData(pair_world)
  ref = hs.HideAndSeek(m, start_within_s=100.0)
  ref.assign(SECOND, FIRST, task_id="t_0001", t=0.0)
  called = []
  ref.on_over.append(called.append)
  d.time = 99.0
  ref.sense(m, d)
  assert not called and ref.playing("t_0001")
  d.time = 100.0
  ref.sense(m, d)
  assert called == [ref] and ref.over_for("t_0001") and ref.flags["winner"] == ""
  v = scoring.evaluate("hide_and_seek", ref.measurements(), table=TABLE)
  assert not v.ok and v.points == 0 and "not played to a decision" in v.reason


def test_each_game_ends_the_last_ones_record_and_a_restart_none(pair_world):
  """One referee, game after game: the next game's roles end the last
  one's record (its verdict stays on the wire until then); and a game does
  not outlive its process -- a restart's referee is idle."""
  ref = hs.HideAndSeek(pair_world)
  assert not ref.assigned and ref.phase == "idle"
  ref.assign(SECOND, FIRST, task_id="t_0001", t=0.0)
  ref.start(1.0)
  kept = ref.kept_state()
  ref.assign(FIRST, SECOND, task_id="t_0002", t=5.0)
  assert ref.over_for("t_0001") and ref.playing("t_0002") and ref.started_at is None
  fresh = hs.HideAndSeek(pair_world)
  fresh.restore_kept(kept)
  assert not fresh.assigned and fresh.phase == "idle"


# ---- 2. the board, the claims, the queue and the pay ----------------------------------


def test_a_two_role_task_is_claimed_per_role():
  board = TaskBoard(table=scoring.default_table())
  task = board.offer("hide_and_seek", GAME_TARGET, t=0.0)
  assert task.roles == ("hider", "seeker") and task.open_roles() == task.roles
  # the first robot takes the first open role; the offer stays open
  first = board.claim(task.id, robot=FIRST.root, t=1.0)
  assert first.state == "offered" and first.claims == {"hider": FIRST.root}
  assert first.claimable(2.0), "the second role is still on offer"
  assert board.claim(task.id, robot=FIRST.root, t=1.5) is None, "one role each"
  assert board.claim(task.id, robot=SECOND.root, role="hider", t=1.6) is None
  second = board.claim(task.id, robot=SECOND.root, t=2.0)
  assert second.state == "claimed" and second.claims["seeker"] == SECOND.root
  assert second.role_of(SECOND.root) == "seeker" and not second.claimable(3.0)
  assert second.as_dict()["claims"] == second.claims


def test_the_verdict_is_the_referees_flags_and_an_unfinished_game_pays_nobody():
  """One verdict off what the referee measured: a find is the seeker's, a
  seek run out the hider's, and a game not played to a decision -- or never
  refereed -- pays nobody."""
  base = {"seekS": 5.0, "findWithinM": hs.FIND_WITHIN_M}
  v = scoring.evaluate("hide_and_seek", {**base, "played": True, "winner": "seeker",
                                         "foundAtS": 3.0, "distanceM": 0.7, "los": True},
                       table=TABLE)
  assert v.ok and v.metrics["winner"] == "seeker" and "the seeker wins" in v.reason
  assert v.points == TABLE["hide_and_seek"].base
  v = scoring.evaluate("hide_and_seek", {**base, "played": True, "winner": "hider",
                                         "overAtS": 6.0, "distanceM": 2.0},
                       table=TABLE)
  assert v.ok and v.metrics["winner"] == "hider" and "the hider wins" in v.reason
  v = scoring.evaluate("hide_and_seek", {**base, "played": False, "winner": ""},
                       table=TABLE)
  assert not v.ok and v.points == 0 and "not played to a decision" in v.reason
  ok, _, reason = scoring.eval_hide_and_seek({})
  assert not ok and "never refereed" in reason


def _game_pair(tmp_path, autonomous=True):
  """The quadruped pair as served, on a shared board: the game's referee
  is wired where the board can offer it (`autonomous`, two robots)."""
  return build_pair(QUAD_HOME, errands=("none", "none"), tasks=True,
                    autonomous=autonomous, overseer=False,
                    task_state=str(tmp_path / "tasks.json"),
                    ledger_state=str(tmp_path / "ledger.json"),
                    thoughts_root=str(tmp_path / "thoughts"))


def test_a_role_queues_nothing_until_every_role_is_taken_and_then_each_its_own(tmp_path):
  """A role's claim queues no errand: the robot that took the first is
  free until the other takes the last. Then the referee is given the game
  and each robot its role's errand -- one each, on the job's id."""
  me, peer = _game_pair(tmp_path)
  try:
    game = me.game
    assert game is not None and peer.game is game
    task, state = arrange_game([me, peer])
    assert me._claim_task(task.id)
    assert me.errands == [] and not game.assigned and state["game"] is None
    assert "the seeker role is still open" in me.status
    assert me.tasks.get(task.id).state == "offered"
    assert not me._claim_task(task.id) and "hider role already" in me.status
    assert peer._claim_task(task.id) and state["game"] is game
    assert [(e.name, e.role, e.task_id) for e in me.errands] == [
      ("game:hide_and_seek:hider", "hider", task.id)]
    assert [(e.name, e.role, e.task_id) for e in peer.errands] == [
      ("game:hide_and_seek:seeker", "seeker", task.id)]
    assert game.playing(task.id) and game.hider is me.body.handle
    assert game.seeker is peer.body.handle
  finally:
    for life in (me, peer):
      life.body.close()


def test_the_winner_is_paid_once_and_nobody_for_a_game_called_off(tmp_path):
  """The referee calls it and the pair banks ONE verdict, on the winner's
  wallet, closing the job for both; a game called off pays nobody and
  fails the job."""
  me, peer = _game_pair(tmp_path)
  try:
    game = me.game
    task, _ = arrange_game([me, peer])
    me._claim_task(task.id)
    peer._claim_task(task.id)
    me.body.start_at(-3.5, 1.0, 0.0)                    # the hider, in the hall
    peer.body.start_at(1.5, 0.5, 0.0)                   # the seeker, nowhere near
    before = (me.ledger.balance(), peer.ledger.balance())
    game.start(0.0)
    me.data.time = game.head_start_s + game.seek_s
    game.sense(me.model, me.data)
    done = me.tasks.get(task.id)
    assert done.state == "done" and done.verdict["metrics"]["winner"] == "hider"
    assert me.ledger.balance() == before[0] + TABLE["hide_and_seek"].base
    assert peer.ledger.balance() == before[1], "the seeker was paid for losing"
    game.sense(me.model, me.data)
    assert me.ledger.balance() == before[0] + TABLE["hide_and_seek"].base, "paid twice"
    # ...and a game nobody began
    nxt, _ = arrange_game([me, peer], t=float(me.data.time))
    me._claim_task(nxt.id)
    peer._claim_task(nxt.id)
    me.data.time += hs.START_WITHIN_S
    game.sense(me.model, me.data)
    assert me.tasks.get(nxt.id).state == "failed"
    assert (me.ledger.balance(), peer.ledger.balance()) == (
      before[0] + TABLE["hide_and_seek"].base, before[1])
  finally:
    for life in (me, peer):
      life.body.close()


def test_the_game_is_offered_to_a_pair_on_autonomous_and_nowhere_else(tmp_path):
  """Its reward row is in challenges.json, shown to `autonomous` alone, and
  it takes two: its target (`world`, home) is named on that arm with both
  robots and nowhere else -- and the pair's referee is in the world where
  the board can offer it, and nowhere else (the stream's header lists the
  activities once)."""
  assert world_targets(QUAD_HOME, procedures=True, robots=("Luca", "Rowan"))["world"] \
      == [GAME_TARGET]
  assert "world" not in world_targets(QUAD_HOME, procedures=True, robots=("Luca",))
  assert "world" not in world_targets(QUAD_HOME, robots=("Luca", "Rowan"))
  assert "hide_and_seek" in default_cadence(QUAD_HOME).kinds
  for autonomous in (True, False):
    lives = _game_pair(tmp_path / str(autonomous), autonomous=autonomous)
    try:
      names = {a.name for a in lives[0].activities}
      assert (lives[0].game is not None) is autonomous
      assert ("hide_and_seek" in names) is autonomous
    finally:
      for life in lives:
        life.body.close()


def test_a_role_is_not_claimed_where_nothing_referees_the_game():
  """A world with no referee has no game to claim: the role would queue
  nothing, and nothing would ever call it."""
  board = TaskBoard(table=scoring.default_table())
  life = stub_life(tasks=board)
  task = board.offer("hide_and_seek", GAME_TARGET, t=0.0)
  assert not life._claim_task(task.id)
  assert "nothing here referees one" in life.status and board.get(task.id).claims == {}


# ---- 3. the roles' verbs ------------------------------------------------------------


class _Referee:
  """The referee as the roles see it: who seeks, the head start, the find,
  and when the game is over -- by the stub world's clock."""

  find_within_m = hs.FIND_WITHIN_M

  def __init__(self, data, seeker_root, task_id, head_s=3.0, over_t=12.0):
    self.data, self.seeker_root, self.task_id = data, seeker_root, task_id
    self.head_s, self.over_t = head_s, over_t
    self.started = None

  def start(self, t):
    self.started = t if self.started is None else self.started

  def playing(self, task_id):
    return task_id == self.task_id and float(self.data.time) < self.over_t

  def over_for(self, task_id):
    return not self.playing(task_id)

  def hiding_left(self, t):
    return max(0.0, (self.started or 0.0) + self.head_s - t)


def _stub_pair(tmp_path):
  cfg = world_config(QUAD_HOME)
  model, data = StubBody.world()
  board = TaskBoard(table=scoring.default_table())
  book = points_ledger(str(tmp_path / "ledger.json"), robots=(FIRST.root, SECOND.root))
  lives = []
  for handle, name in ((FIRST, "Luca"), (SECOND, "Rowan")):
    body = StubBody(model, data, handle=handle, rack=cfg["rack"],
                    grid_bounds=cfg["grid_bounds"])
    lives.append(stub_life(body=body, handle=handle, robot_name=name, tasks=board,
                           ledger=Account(book, handle.root)))
  for life in lives:
    life.peers = [other for other in lives if other is not life]
  return lives


def test_each_role_plays_its_part_and_the_seeker_is_never_told_where_the_hider_is(tmp_path):
  """The hider hides from where the seeker SAYS it is counting, with the
  head start's walk and clear of a find; the seeker counts out the head
  start where it stands, then searches outward from there with nothing but
  where it counted and how near a find is. Each stands by to the game's
  end, and each errand starts the referee's clock."""
  hider, seeker = _stub_pair(tmp_path)
  seeker.body.start_at(2.0, 3.0, 0.0)
  hider.body.start_at(-1.0, -1.0, 0.0)
  ref = _Referee(seeker.data, seeker.root, "t_0001")
  began = []
  real = seeker.body.seek_routine

  def seek(*a, **kw):
    began.append(float(seeker.data.time))
    return (yield from real(*a, **kw))
  seeker.body.seek_routine = seek
  runs = []
  for life, role in ((hider, "hider"), (seeker, "seeker")):
    life.game = ref
    errand = programmed_errand(hide_and_seek_program(), task="game", role=role)
    errand.task_id = "t_0001"
    runs.append((life, errand))
  results = tick.run_many([(life.body.stepper, life.run_errand_routine(e)) for life, e in runs],
                          name="game")
  for result in results:
    assert result["procedure"]["ok"], result
  assert hider.body.hid_from == [((2.0, 3.0), st.HIDE_MIN_REACH_M,
                                  hs.FIND_WITHIN_M + st.GAME_MARGIN_M)]
  assert seeker.body.sought == [((2.0, 3.0), hs.FIND_WITHIN_M - st.GAME_MARGIN_M)]
  assert began and began[0] >= ref.head_s, "it searched before it had counted"
  assert ref.started == 0.0
  assert float(hider.data.time) == pytest.approx(ref.over_t, abs=1.0)


def test_a_role_with_no_game_to_play_says_so():
  life = stub_life()
  life.game = None
  run = life.run_errand(programmed_errand(hide_and_seek_program(), task="game", role="seeker"))
  step = run["procedure"]["steps"][0]
  assert not step["ok"] and step["reason"] == "there is no game of hide and seek to seek in"


def test_the_games_verbs_are_its_programs_alone():
  """`hide` and `seek` are a game's program's, validated against the
  game's facts; the language the robot writes procedures in knows neither
  -- not a verb, not a name it reserves, not in a refusal's list -- and the
  prompt names neither."""
  plain, game = world_facts(QUAD_HOME), world_facts(QUAD_HOME, game=True)
  assert not set(st.GAME_VERB_NAMES) & set(plain.verbs)
  assert set(st.GAME_VERB_NAMES) <= set(game.verbs)
  with pytest.raises(st.Refused):
    st.compile_program(hide_and_seek_program(), plain)
  st.compile_program(hide_and_seek_program(), game)
  with pytest.raises(st.Refused) as refused:
    lang.compile_procedure("def go():\n  hide()\n", game)
  assert "unknown verb 'hide'" in str(refused.value) and "seek" not in str(refused.value)
  lang.compile_procedure("def go():\n  seek = 2\n  wait(seek)\n", plain)
  rule = ov.procedure_rule(armed=False, swaps=True, places=True, plates=True)
  assert "hide(" not in rule and "seek(" not in rule


# ---- 4. the hider's spot and the seeker's search, on a map drawn here -------------------


@pytest.fixture(scope="module")
def house():
  return lw.home_spec().compile()


#: Two rooms side by side, x 0..3 and 3..6, y 0..4, a 1 m door in the wall
#: between them at y 1.5..2.5; the seeker counts in the west room.
BASE = (1.0, 2.0)


def _rooms(m):
  g = m.grid
  g.grid[:] = 5.0
  row = lambda y: g.world_to_cell(0.0, y)[1]   # noqa: E731
  col = lambda x: g.world_to_cell(x, 0.0)[0]   # noqa: E731
  g.grid[row(0.0):row(4.0), col(0.0):col(6.0)] = -5.0
  g.grid[row(0.0):row(4.0), col(2.98):col(3.08)] = 5.0
  g.grid[row(1.5):row(2.5), col(2.98):col(3.08)] = -5.0


def _mission(house, at=(2.0, 3.0)):
  m = qb.QuadMission(house, mujoco.MjData(house), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  m.start_at(*at, 0.0)
  _rooms(m)
  m.others = []
  return m


def test_the_hider_picks_a_spot_out_of_sight_the_seekers_longest_walk_within_reach(house):
  """Out of the seeker's sight from where it counts, farther than a find,
  clear of the walls by `HIDE_CLEAR_M` (no doorway), within the walk the
  head start buys -- and of those, the one the seeker would walk furthest
  to; with no hidden spot in reach, the furthest in the open."""
  m = _mission(house)
  try:
    spot = m.hiding_spot(BASE, reach_m=8.0, clear_of_m=1.8)
    x, y = spot["at"]
    assert spot["hidden"] and x > 3.2, spot                  # through the door
    assert math.dist((x, y), BASE) > 1.8 and spot["walkM"] <= 8.0
    assert min(x - 3.08, 6.0 - x, y, 4.0 - y) >= gm.HIDE_CLEAR_M - 0.1
    # ...and no spot clear in the east room that the hider can walk to is a
    # longer walk for the seeker, by more than a lattice cell
    theirs = m.walk_field(m.seen_floor_lattice(), BASE)
    mine = m.walk_field(m.seen_floor_lattice(mask_others=True), m.pose_xy(), limit=8.0)
    cx, cy = m._lattice_xy(theirs.shape)
    inner = gm.HIDE_CLEAR_M + 0.1
    east = ((cx > 3.08 + inner) & (cx < 6.0 - inner) & (cy > inner) & (cy < 4.0 - inner)
            & np.isfinite(mine) & np.isfinite(theirs))
    assert east.any() and spot["seekerWalkM"] >= float(theirs[east].max()) - 0.1, spot
    near = m.hiding_spot(BASE, reach_m=1.5, clear_of_m=1.8)
    assert not near["hidden"] and near["at"][0] < 3.0 and near["walkM"] <= 1.5
  finally:
    m.close()


def _hall_and_alcove(m):
  """A long open hall, x 0..2, y 0..9, the seeker counting at its south
  end, and a small alcove off its east side near that end, x 2..4.5,
  y 1..3, behind a wall with a 1 m opening at y 2.5..3."""
  g = m.grid
  g.grid[:] = 5.0
  row = lambda y: g.world_to_cell(0.0, y)[1]   # noqa: E731
  col = lambda x: g.world_to_cell(x, 0.0)[0]   # noqa: E731
  g.grid[row(0.0):row(9.0), col(0.0):col(2.0)] = -5.0
  g.grid[row(1.0):row(3.6), col(2.1):col(4.5)] = -5.0
  g.grid[row(2.5):row(3.5), col(2.0):col(2.1)] = -5.0


def test_out_of_sight_comes_before_a_longer_walk_in_the_open(house):
  """Out of the seeker's sight first: the far end of an open hall is the
  seeker's longest walk, and in its plain view; an alcove round a corner,
  a shorter walk, is where a hider hides."""
  m = _mission(house, at=(1.0, 2.0))
  try:
    _hall_and_alcove(m)
    spot = m.hiding_spot((1.0, 0.8), reach_m=12.0, clear_of_m=1.8)
    assert spot["hidden"] and spot["at"][0] > 2.2, spot
    longest = m.walk_field(m.seen_floor_lattice(), (1.0, 0.8))
    assert spot["seekerWalkM"] < float(np.nanmax(np.where(np.isfinite(longest), longest,
                                                          np.nan))) - 2.0, "the premise"
  finally:
    m.close()


def test_the_dock_and_the_rack_are_no_hiding_place(house):
  """A robot resting before the dock or the rack is in the way of the
  other's charge or tool, and stood up to make way it is given away: the
  hider keeps `KEEP_CLEAR_M` off both, commissioned, as it knows them."""
  m = _mission(house)
  try:
    spot = m.hiding_spot(BASE, reach_m=8.0, clear_of_m=1.8)
    m.dock_prior = (*spot["at"], 0.0)            # the dock, where it would hide
    kept = m.hiding_spot(BASE, reach_m=8.0, clear_of_m=1.8)
    assert math.dist(kept["at"], spot["at"]) > gm.KEEP_CLEAR_M, kept
    assert all(math.dist(kept["at"], f) > gm.KEEP_CLEAR_M for f in m._facilities())
  finally:
    m.close()


def test_the_seekers_search_never_reads_where_the_other_robot_is(house):
  """The search is the seeker's own map and where it counted, and nothing
  else: with the hider's disc on its planner or not, it sets out for the
  same viewpoints in the same order -- first where a hider hides, out of
  its own sight from where it counted, then the rest, every viewpoint it
  can reach seen round."""
  went = {}
  for told in (False, True):
    m = _mission(house, at=BASE)
    in_view = m._seen_from(BASE, gm.SIGHT_M, m._sight_walls())
    side = m.grid.resolution * 2
    path: list = []

    def teleport(x, y, timeout=90.0, stop=None, m=m, path=path):
      m._set_pose(x, y, 0.0)
      m.data.time += 1.0
      path.append((round(x, 2), round(y, 2)))
      if stop is not None:
        stop()
      return True
      yield
    m.drive_to_routine = teleport
    if told:
      m.others = [lambda: KeepClear(4.5, 2.0, root=SECOND.root)]
    try:
      rec = tick.run(STEPPER, m.seek_routine(BASE, 1.2, patience=60.0))
      assert rec["why"] == "searched" and rec["targets"] == len(path) > 4
      went[told] = path
      seen = [bool(in_view[int((y - m.grid.y_min) // side), int((x - m.grid.x_min) // side)])
              for x, y in path]
      assert False in seen and True in seen, "the premise: both kinds of floor"
      first = seen.index(True)
      assert not any(seen[:first]) and all(seen[first:]), \
        f"a viewpoint in plain sight before the hidden floor was done: {seen}"
    finally:
      m.close()
  assert went[True] == went[False]
