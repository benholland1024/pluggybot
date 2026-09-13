"""The first two-role errand (issue #167, M12 slice D): hide and seek --
claimed per role, each robot its own steps, one verdict for both off the
referee's measurement of the world.
"""

import mujoco
import pytest

from pluggybot.activity.hideseek import (
  FIND_WITHIN_M, SEEK_HEAD_START_S, SEEK_S, HideAndSeek,
)
from pluggybot.economy import scoring
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import errand_for_task, hide_and_seek_program, world_facts
from pluggybot.pair import arrange_game, build_pair, run_pair
from pluggybot.procedure.steps import validate
from pluggybot.robot import FIRST, SECOND, world_with_robots

TABLE = scoring.challenge_table()


def test_the_program_validates_and_names_two_roles():
  p = hide_and_seek_program("room_hub")
  assert set(p.roles) == {"hider", "seeker"}
  assert validate(p, world_facts("room_hub")) == []
  assert validate(hide_and_seek_program("home"), world_facts("home")) == []
  assert p.steps("seeker")[0].verb == "wait"
  assert sum(s.args["seconds"] for s in p.steps("hider") if s.verb == "wait") \
    >= SEEK_HEAD_START_S + SEEK_S


def test_a_two_role_task_is_claimed_per_role_and_builds_each_roles_errand():
  board = TaskBoard(table=scoring.default_table())
  task = board.offer("hide_and_seek", "room_hub", t=0.0)
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
  # each role's errand is that role's steps, unscored by the lifecycle
  hider = errand_for_task(second, "room_hub", role="hider")
  seeker = errand_for_task(second, "room_hub", role="seeker")
  assert hider.role == "hider" and seeker.role == "seeker"
  assert hider.task == "game" and "game" not in scoring.EVALUATORS
  assert hider.program is not None and hider.program.steps("hider")[0].verb == "drive_to"
  assert errand_for_task(second, "room_hub", role="referee") is None


def _place(model, data, handle, x, y, yaw=0.0):
  import math
  q = handle.qpos_adr(model)
  data.qpos[q:q + 3] = [x, y, 0.045]
  data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


@pytest.fixture(scope="module")
def two_robot_hub():
  return world_with_robots("models/room_hub.xml", second_at=(3.0, 3.0))


def _referee(model, data, head_start=1.0, seek=5.0):
  game = HideAndSeek(model, hider=FIRST, seeker=SECOND, head_start_s=head_start,
                     seek_s=seek)
  game.start(data.time)
  return game


def _run(model, data, game, seconds):
  t0 = data.time
  while data.time - t0 < seconds:
    mujoco.mj_step(model, data)
    game.sense(model, data)


def test_the_referee_calls_a_find_off_both_poses_with_line_of_sight(two_robot_hub):
  """One verdict off the world: the seeker close with line of sight is a
  find; close behind a wall is not; far is not."""
  model = two_robot_hub
  data = mujoco.MjData(model)
  _place(model, data, FIRST, 2.0, 3.0)
  _place(model, data, SECOND, 2.7, 3.0)                # 0.7 m, open floor
  mujoco.mj_forward(model, data)
  game = _referee(model, data)
  _run(model, data, game, 1.5)
  assert game.phase == "found" and game.flags["winner"] == "seeker"
  assert game.flags["los"] and game.flags["distanceM"] < FIND_WITHIN_M
  v = scoring.evaluate("hide_and_seek", game.measurements(), table=TABLE)
  assert v.ok and v.metrics["winner"] == "seeker" and "the seeker wins" in v.reason
  assert v.points == TABLE["hide_and_seek"].base

  # behind the room's wall divider: close as the crow flies, no sight line
  data = mujoco.MjData(model)
  _place(model, data, FIRST, 1.8, 5.0)
  _place(model, data, SECOND, 1.8, 3.0)
  mujoco.mj_forward(model, data)
  game = _referee(model, data, seek=2.0)
  _run(model, data, game, 3.5)
  assert game.phase == "over" and game.flags["winner"] == "hider"
  v = scoring.evaluate("hide_and_seek", game.measurements(), table=TABLE)
  assert v.ok and v.metrics["winner"] == "hider" and "the hider wins" in v.reason

  # a game nobody finished pays nobody
  game = _referee(model, data, seek=100.0)
  _run(model, data, game, 0.5)
  v = scoring.evaluate("hide_and_seek", game.measurements(), table=TABLE)
  assert not v.ok and v.points == 0 and "not played to a decision" in v.reason
  ok, _, reason = scoring.eval_hide_and_seek({})
  assert not ok and "never refereed" in reason


def test_the_pair_arranges_the_game_and_pays_the_winner_only(monkeypatch):
  """The whole chain on stubbed drives: offer, two claims, the referee
  built once both roles are held, one verdict banked on the winner."""
  from test_procedure import _stub_swaps
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     tasks=True)
  for life in lives:
    _stub_swaps(life, monkeypatch)
  task, state = arrange_game(lives)
  assert task.state == "offered" and state["game"] is None
  a, b = lives
  # The referee is in the world's activities FROM THE OFFER (0.20.0): the
  # stream's header lists activities once, so one added at the claim would
  # ship flags under a name no consumer was told about. Idle and unassigned
  # until both roles are held; `start` before then is a no-op.
  assert "hide_and_seek" in a.activities.names and a.game is not None
  assert not a.game.assigned and a.game.phase == "idle"
  a.game.start(0.0)
  assert a.game.phase == "idle", "a game with no roles started"
  assert a._claim_task(task.id) and state["game"] is None
  assert b._claim_task(task.id) and state["game"] is not None
  game = state["game"]
  assert game.assigned and game.hider is a.mission.handle and game.seeker is b.mission.handle
  assert a.game is game and b.game is game
  assert a.role_in(task.id) == "hider" and b.role_in(task.id) == "seeker"
  assert a.tasks.get(task.id).state == "claimed"
  # ...the referee is fed the world and calls it: the two robots are
  # placed apart behind the divider, the seek phase runs out, the hider wins
  _place(a.model, a.data, FIRST, 1.8, 5.0)
  _place(a.model, a.data, SECOND, 1.8, 3.0)
  mujoco.mj_forward(a.model, a.data)
  game.head_start_s, game.seek_s = 0.5, 1.0
  game.start(a.data.time)
  before = (a.ledger.balance(), b.ledger.balance())
  _run(a.model, a.data, game, 2.0)
  assert game.phase == "over"
  assert a.tasks.get(task.id).state == "done"
  assert a.tasks.get(task.id).verdict["metrics"]["winner"] == "hider"
  assert a.ledger.balance() == before[0] + TABLE["hide_and_seek"].base
  assert b.ledger.balance() == before[1], "the seeker was paid for losing"


@pytest.mark.endurance
def test_hide_and_seek_is_played_by_two_robots():
  """The game flown on room_hub: both claim, the hider drives off, the
  seeker counts and sweeps, the referee calls it, and the task resolves
  with one verdict. Behind --endurance (~5 min of two robots' physics);
  every rule it exercises is pinned above."""
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     tasks=True)
  task, state = arrange_game(lives)
  results = run_pair(lives, max_sim_time=SEEK_HEAD_START_S + SEEK_S + 200.0,
                     stop_when=lambda ls: ls[0].tasks.get(task.id).state
                     in ("done", "failed"))
  game = state["game"]
  assert game is not None and game.over_at is not None
  final = lives[0].tasks.get(task.id)
  assert final.state == "done" and final.verdict["metrics"]["winner"] in ("hider", "seeker")
  assert all(r["dead"] is None for r in results)
  paid = [r["points"] for r in results]
  assert sorted(paid) == [0, TABLE["hide_and_seek"].base]
