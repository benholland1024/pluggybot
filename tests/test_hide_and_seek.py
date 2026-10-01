"""The first two-role errand (issue #167, M12 slice D): hide and seek --
claimed per role, each robot its own steps, one verdict for both off the
referee's measurement of the world. (The referee reads a body's names;
#404 re-keys it and writes the roles' programs for legs. What stays here is
the bookkeeping around it.)
"""

from types import SimpleNamespace

from pluggybot.activity import hideseek
from pluggybot.body import StubBody
from pluggybot.economy import scoring
from pluggybot.economy.ledger import Account
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import QUAD_HOME, points_ledger, world_config
from pluggybot.mission.errand import programmed_errand
from pluggybot.pair import arrange_game
from pluggybot.procedure.steps import Program, Step
from pluggybot.robot import FIRST, SECOND
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

TABLE = scoring.challenge_table()


def test_a_two_role_task_is_claimed_per_role():
  board = TaskBoard(table=scoring.default_table())
  task = board.offer("hide_and_seek", QUAD_HOME, t=0.0)
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
  base = {"seekS": 5.0, "findWithinM": hideseek.FIND_WITHIN_M}
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


def _pair(tmp_path):
  """Two stub lives in one world, on one board and one ledger."""
  cfg = world_config(QUAD_HOME)
  model, data = StubBody.world()
  board = TaskBoard(table=scoring.default_table())
  book = points_ledger(str(tmp_path / "ledger.json"), robots=(FIRST.root, SECOND.root))
  lives = []
  for handle, name in ((FIRST, "Pluggy"), (SECOND, "Rowan")):
    body = StubBody(model, data, handle=handle, rack=cfg["rack"],
                    grid_bounds=cfg["grid_bounds"])
    lives.append(stub_life(body=body, handle=handle, robot_name=name, tasks=board,
                           ledger=Account(book, handle.root)))
  return lives


def test_the_pair_arranges_the_game_and_pays_the_winner_only(tmp_path, monkeypatch):
  """The whole chain short of the referee's measurement: offer, two
  claims, the referee bound once both roles are held, and ONE verdict,
  banked on the winner's wallet, closing the task for both."""
  bound = []
  monkeypatch.setattr(hideseek.HideAndSeek, "assign", lambda self, hider, seeker: (
    setattr(self, "hider", hider), setattr(self, "seeker", seeker),
    bound.append((hider, seeker))))
  a, b = _pair(tmp_path)
  task, state = arrange_game([a, b])
  assert task.state == "offered" and state["game"] is None
  game = a.game
  assert game is not None and b.game is game
  assert not game.assigned and game.phase == "idle"
  game.start(0.0)
  assert game.phase == "idle", "a game with no roles started"
  assert a.tasks.claim(task.id, robot=a.root, t=1.0) and state["game"] is None
  assert a.tasks.claim(task.id, robot=b.root, t=2.0) and state["game"] is game
  assert bound == [(a.body.handle, b.body.handle)], "the roles bound once, as claimed"
  assert a.role_in(task.id) == "hider" and b.role_in(task.id) == "seeker"
  # the referee calls it: the seek ran out, the hider wins
  game.start(2.0)
  game.over_at = 8.0
  game.set(phase="over", winner="hider", overAtS=6.0)
  before = (a.ledger.balance(), b.ledger.balance())
  for hook in list(game.on_over):
    hook(game)
  done = a.tasks.get(task.id)
  assert done.state == "done" and done.verdict["metrics"]["winner"] == "hider"
  assert a.ledger.balance() == before[0] + TABLE["hide_and_seek"].base
  assert b.ledger.balance() == before[1], "the seeker was paid for losing"


def test_each_robot_plays_its_roles_steps_and_starts_the_referees_clock(tmp_path):
  """A two-role program runs ONE role per robot, named on its errand: each
  robot's run is its role's steps, to the end -- and the referee's clock
  starts as a role's errand begins."""
  a, b = _pair(tmp_path)
  program = Program(name="hide_and_seek", roles={
    "hider": (Step("drive_to", {"x": 3.0, "y": 1.0}), Step("wait", {"seconds": 2.0})),
    "seeker": (Step("wait", {"seconds": 1.0}), Step("drive_to", {"x": 3.5, "y": 1.0}),
               Step("face", {"heading": 0.0}))})
  started: list = []
  for life in (a, b):
    life.game = SimpleNamespace(start=started.append)
  for life, role in ((a, "hider"), (b, "seeker")):
    run = life.run_errand(programmed_errand(program, task="game", role=role))["procedure"]
    assert run["ok"] and run["role"] == role, run
    assert [s["verb"] for s in run["steps"]] == [s.verb for s in program.steps(role)]
  assert len(started) == 2, "a role's errand began without the referee's clock"
