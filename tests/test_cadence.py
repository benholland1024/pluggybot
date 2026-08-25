"""Guards for task generation, timing and expiry (issue #23).

hub/tasks.py's tests hold down what a task IS. These hold down when one turns
up and when it goes away, which is a different set of failures:

  1. THE WORLD DOES NOT GO EMPTY. A starter set that never grows back is a
     robot with nothing asked of it for the rest of a multi-hour run.
  2. ...AND IT DOES NOT GO MAD. Offers are capped, the board is bounded, and
     an offer nobody takes lapses on schedule rather than standing forever.
  3. NO TARGET IS BOOKED FOREVER, AND NO KIND IS STARVED. The same whiteboard
     back-to-back, or a third kind that never once gets a turn, are the two
     ways a rotation quietly stops being one.
  4. THE TIMING IS CONFIGURATION. Cadence, caps and expiry are read off
     hub/cadence.json, and a deploy can point $PLUGGY_CADENCE somewhere else.
  5. NOTHING HERE CAN DELAY A CHARGE. The producer runs on the physics seam,
     which means it runs DURING a charge -- so what it is allowed to touch is
     a property worth pinning rather than assuming.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from pluggybot.hub import lifecycle as lc
from pluggybot.hub.cadence import (
  CADENCE_ENV, CADENCE_VERSION, Cadence, TaskProducer, default_cadence,
  reload_cadence,
)
from pluggybot.hub.tasks import KINDS, TaskBoard

HOME_TARGETS = {"board": ["whiteboard_a", "whiteboard_b"],
                "zone": ["garden"], "module": ["module_lcd"]}


def cadence(**kw) -> Cadence:
  spec = {"firstAtS": 100.0, "everyS": 100.0, "ttlS": 300.0,
          "cooldownS": 150.0, "maxOffered": 4, "maxTasks": 20, "initial": 3,
          "kinds": {"whiteboard_answer": {},
                    "draw_figure": {"programs": ["house", "tree"]},
                    "rate_artwork": {"programs": ["robot"]},
                    "count_plants": {}}}
  spec.update(kw)
  return Cadence._build("test", spec, None)


def board(**kw) -> TaskBoard:
  return TaskBoard(clock=lambda: "2026-08-24T00:00:00", **kw)


def producer(beat=None, targets=None, b=None) -> TaskProducer:
  beat = beat if beat is not None else cadence()
  b = b if b is not None else board(max_tasks=beat.max_tasks,
                                    max_offered=beat.max_offered)
  return TaskProducer(b, beat, targets if targets is not None else HOME_TARGETS)


def run(maker: TaskProducer, until: float, step: float = 10.0,
        pack_wh: float | None = None, seed: bool = True) -> list:
  """Tick a producer over `until` sim seconds, expiring as the loop would.

  The mission loop's `_task_step` in miniature: expire, then offer. No
  physics, so a whole sim-day costs milliseconds -- which is the only way to
  assert anything about a MULTI-HOUR run in a suite that has to stay under
  ten minutes.
  """
  made = list(maker.seed(0.0, pack_wh)) if seed else []
  t = 0.0
  while t < until:
    t += step
    maker.board.expire_due(t)
    made += maker.tick(t, pack_wh)
  return made


# ---- the world keeps offering ------------------------------------------------


def test_a_worked_through_board_gets_more_work():
  """Issue #23's first line, and the whole reason `seed_tasks` was not enough.

  The robot finishes everything it is given; the world has to notice. Shown
  to fail by a producer that only ever seeds -- delete the body of `tick` and
  the board goes quiet at t=0 and stays that way for the rest of the run,
  which is exactly what `lifecycle.seed_tasks` did.
  """
  maker = producer()
  for task in maker.seed(0.0):
    # ...and worked through immediately, so nothing is booked or resting on
    # a target the way a real board would be.
    maker.board.tasks[task.id] = dataclasses.replace(
      task, state="done", resolved_t=1.0)
  assert not maker.board.open_tasks(), "the board should be clear"
  assert run(maker, until=1200.0, seed=False), \
    "the world offered a starter set and never asked for anything again"


def test_offers_arrive_on_the_configured_cadence():
  """`everyS` is what it says, and the FIRST one waits `firstAtS`."""
  maker = producer(cadence(everyS=100.0, firstAtS=250.0, initial=1,
                           cooldownS=0.0, ttlS=None, maxOffered=40,
                           maxTasks=40))
  maker.seed(0.0)
  at = [t.created_t for t in run(maker, until=1000.0, step=5.0, seed=False)]
  assert at, "nothing was ever offered"
  assert at[0] == pytest.approx(250.0, abs=5.0)
  gaps = [b - a for a, b in zip(at, at[1:])]
  assert all(g == pytest.approx(100.0, abs=5.0) for g in gaps), gaps


def test_a_tick_that_cannot_place_a_job_does_not_catch_up_later():
  """⚠ NO BURST. A robot that spent twenty minutes on one errand must not
  walk back into a board with eight new jobs on it -- that is the unbounded
  backlog this module exists to prevent, arriving through the front door.

  Shown to fail by advancing `next_at` by `every_s` instead of setting it
  from `t`: the ticks missed while the board was full come due at once.
  """
  maker = producer(cadence(everyS=60.0, firstAtS=0.0, initial=0, ttlS=None))
  # An hour in which nothing may be placed, because every target is booked.
  for target in ("whiteboard_a", "whiteboard_b", "garden"):
    kind = "draw_figure" if target.startswith("white") else "count_plants"
    assert maker.board.offer(kind, target, t=0.0) is not None
  maker.tick(3600.0)
  # ...and now one comes free. Exactly one job, not sixty.
  maker.board.expire_due(3600.0)
  for task in list(maker.board.open_tasks()):
    maker.board.tasks.pop(task.id)
  assert len(maker.tick(3660.0)) == 1
  assert len(maker.tick(3661.0)) == 0, "the missed ticks were banked"


# ---- ...and stays bounded ----------------------------------------------------


def test_a_multi_hour_run_stays_bounded_and_lapses_what_nobody_takes():
  """Issue #23's acceptance, first two boxes, in one four-sim-hour run.

  Nothing claims anything here, which is the WORST case for the bound: every
  offer runs its full deadline and then has to be aged out. The board must
  still be the size the config says, and the surplus must show as `expired`
  rather than as a silent deletion.
  """
  beat = cadence(maxOffered=4, maxTasks=20, ttlS=300.0)
  maker = producer(beat)
  run(maker, until=4 * 3600.0)
  stats = maker.board.stats()
  assert maker.made > 30, f"a four-hour world offered {maker.made} jobs"
  assert stats["held"] <= beat.max_tasks
  assert len(maker.board.offered()) <= beat.max_offered
  assert stats["expired"] > 0 and stats["dropped"] > 0
  # An aged-out task is not an expired one, and neither is a silent hole: the
  # two counters plus what is held have to account for every id ever issued.
  assert stats["held"] + stats["dropped"] == stats["total"]


def test_a_healthy_run_shows_claimed_completed_and_lapsed():
  """Issue #23's second acceptance box: a MIX, not one outcome repeated.

  The robot here is a stand-in -- it takes the oldest claimable job every
  ~400 sim-seconds and finishes it, which is roughly a home charge-to-charge
  cycle. What is under test is the cadence around it: offer faster than that
  and some jobs lapse; offer slower and the robot idles. Both are failures
  and only the middle passes.
  """
  from pluggybot.hub import scoring
  maker = producer(cadence(everyS=120.0, firstAtS=120.0, ttlS=420.0,
                           cooldownS=200.0))
  maker.seed(0.0)
  t, busy_until = 0.0, 0.0
  while t < 4 * 3600.0:
    t += 10.0
    maker.board.expire_due(t)
    maker.tick(t)
    if t < busy_until:
      continue
    ready = maker.board.claimable(t)
    if not ready:
      continue
    took = maker.board.claim(ready[0].id, t=t, answer="7")
    assert took is not None
    maker.board.start(took.id, t=t)
    busy_until = t + 400.0
    # Judged as a `carry`, which is the cheapest evaluator to satisfy from a
    # dict: what is under test is the cadence around a robot that finishes
    # things, not what any particular errand is worth.
    maker.board.resolve(took.id, scoring.evaluate(
      "carry", {"picked": True, "stowed": True, "module": "module_lcd"}), t=t)
  stats = maker.board.stats()
  assert stats["done"] > 3, f"nothing got finished: {stats}"
  assert stats["expired"] > 3, f"nothing ever lapsed: {stats}"
  # ...and the robot was never left standing about with an empty board.
  assert maker.board.offered(), "the world ran out of work"


# ---- fairness ----------------------------------------------------------------


def test_the_same_whiteboard_is_not_booked_back_to_back():
  """The per-target cooldown, which is the third thing issue #23 asks for."""
  beat = cadence(cooldownS=500.0, everyS=50.0, firstAtS=0.0, initial=0,
                 ttlS=60.0, kinds={"draw_figure": {}})
  maker = producer(beat, targets={"board": ["whiteboard_a"]})
  made = run(maker, until=2000.0, step=10.0, seed=False)
  at = [t.created_t for t in made]
  gaps = [b - a for a, b in zip(at, at[1:])]
  assert gaps, "only one offer in 2000 s: the cooldown is not being lifted"
  assert min(gaps) >= beat.cooldown_s, \
    f"whiteboard_a was re-booked after {min(gaps)} s"


def test_two_boards_are_used_evenly_rather_than_the_first_one_twice():
  """Least-recently-offered, not first: with `first`, the second whiteboard
  is scenery."""
  maker = producer(cadence(cooldownS=0.0, everyS=50.0, firstAtS=0.0,
                           initial=0, ttlS=40.0,
                           kinds={"draw_figure": {}}))
  used = [t.target for t in run(maker, until=1000.0, seed=False)]
  assert set(used) == {"whiteboard_a", "whiteboard_b"}
  assert abs(used.count("whiteboard_a") - used.count("whiteboard_b")) <= 1


def test_a_kind_behind_two_others_is_not_starved_forever():
  """⚠ MEASURED, and it is why the cursor keeps a passed-over kind at the
  head of the queue rather than advancing past whatever it placed.

  home has THREE board-shaped kinds and TWO whiteboards, so on any cycle one
  of them cannot be placed. With a cursor that simply advanced on each
  placement, the same two won every time and `rate_artwork` -- the entire
  visitor-rated tier -- was offered ZERO times in four sim-hours. Shown to
  fail by setting `self.cursor = (index + 1) % n` unconditionally.
  """
  maker = producer()
  made = run(maker, until=4 * 3600.0)
  offered = {task.kind for task in made}
  assert offered == set(maker.kinds), \
    f"never offered: {sorted(set(maker.kinds) - offered)}"


def test_successive_drawings_are_not_all_the_same_figure():
  """`programs` rotates, so a day of the site's stream is not one house over
  and over."""
  maker = producer(cadence(cooldownS=0.0, everyS=50.0, firstAtS=0.0,
                           initial=0, ttlS=40.0,
                           kinds={"draw_figure": {"programs": ["house",
                                                               "tree"]}}))
  drawn = [t.params["program"] for t in run(maker, until=600.0, seed=False)]
  assert set(drawn) == {"house", "tree"}, drawn


def test_the_rotation_is_deterministic():
  """A world that offered different work every run would make every mission
  test a different test -- the argument `QuestionBank.pick` is built on."""
  def once():
    return [(t.kind, t.target, t.created_t)
            for t in run(producer(), until=3000.0)]
  assert once() == once()


# ---- the world's furniture ---------------------------------------------------


def test_a_world_with_no_whiteboards_offers_fewer_jobs_not_broken_ones():
  maker = producer(targets={"module": ["module_lcd"]})
  assert maker.kinds == (), "a board kind was kept in a world with no boards"
  assert run(maker, until=2000.0) == []
  # ...and the same cadence over room_hub's actual furniture offers the carry.
  bare = producer(cadence(kinds={"fetch_module": {}}),
                  targets={"module": ["module_lcd"]})
  assert [t.kind for t in bare.seed(0.0)] == ["fetch_module"]


def test_the_seed_respects_the_per_target_rule_rather_than_double_booking():
  """home has three board-shaped kinds and two whiteboards, so a seed of four
  puts up three. Fewer offers, never two jobs on one board."""
  book = lc.board_book("home")
  b = lc.task_board(cadence=default_cadence("home"))
  seeded = lc.task_producer(b, "home", book).seed(0.0)
  targets = [t.target for t in seeded]
  assert len(targets) == len(set(targets)), targets


# ---- it is configuration -----------------------------------------------------


def test_every_world_block_in_the_shipped_file_loads():
  """Issue #23's last acceptance box, and the cheapest way for the file to be
  wrong: a kind renamed in hub/tasks.py and not here."""
  doc = json.loads((Path(__file__).parents[1]
                    / "src/pluggybot/hub/cadence.json").read_text())
  assert doc["version"] == CADENCE_VERSION
  for world in ("", *doc["worlds"]):
    beat = default_cadence(world)
    assert beat.kinds and all(name in KINDS for name in beat.kinds)
    assert beat.every_s > 0 and beat.max_offered <= beat.max_tasks


def test_a_deploy_can_point_the_cadence_somewhere_else(tmp_path, monkeypatch):
  """$PLUGGY_CADENCE, on the same terms as $PLUGGY_REWARDS: how busy a
  deployed world is has to be re-tunable on a mounted volume, not in a
  rebuild."""
  path = tmp_path / "cadence.json"
  path.write_text(json.dumps({
    "version": CADENCE_VERSION,
    "default": {"everyS": 42.0, "ttlS": 99.0, "maxOffered": 2, "maxTasks": 3,
                "initial": 1, "kinds": {"count_plants": {}}}}))
  monkeypatch.setenv(CADENCE_ENV, str(path))
  beat = reload_cadence(world="home")
  try:
    assert beat.every_s == 42.0 and beat.max_offered == 2
    assert list(beat.kinds) == ["count_plants"]
  finally:
    monkeypatch.delenv(CADENCE_ENV)
    reload_cadence(world="home")


def test_a_cadence_that_offers_a_job_nothing_can_judge_refuses_to_load():
  """The lock `Task.create` carries, one door up. A world scheduled to offer
  work nothing can build is worth finding out about at load rather than as a
  board that is quietly emptier than the file says."""
  with pytest.raises(ValueError, match="unknown task kind"):
    Cadence._build("test", {"kinds": {"paint_the_cat": {}}}, None)


@pytest.mark.parametrize("spec,match", [
  ({"everyS": 0.0}, "everyS"),
  ({"ttlS": -1.0}, "ttlS"),
  ({"maxOffered": 0}, "at least 1"),
  ({"maxOffered": 9, "maxTasks": 4}, "exceeds"),
])
def test_an_impossible_cadence_refuses_to_load(spec, match):
  with pytest.raises(ValueError, match=match):
    Cadence._build("test", {**{"kinds": {"count_plants": {}}}, **spec}, None)


def test_a_newer_cadence_version_refuses_to_load(tmp_path):
  path = tmp_path / "cadence.json"
  path.write_text(json.dumps({"version": CADENCE_VERSION + 1, "default": {}}))
  with pytest.raises(ValueError, match="cadence version"):
    Cadence.load("home", path)


# ---- and it cannot get in the robot's way ------------------------------------


def test_the_world_only_offers_work_it_could_ever_pay_the_energy_for():
  """A job whose estimate exceeds a CHARGED pack is never put up: the robot
  could not finish it after any amount of charging.

  ⚠ Measured against a charged pack, not the cell right now, and the
  distinction is not pedantry. Gating each tick on the instantaneous charge
  is the literal reading of issue #23's energy line and it takes home from 58
  offers in four sim-hours to 14 -- fewer than the robot can complete, which
  is the empty world this module exists to prevent. Deferring until after a
  charge is real and already lives in `Task.claimable`: the offer stands,
  unclaimable, until the pack is legal again.
  """
  maker = producer(cadence(kinds={"draw_figure": {}, "count_plants": {}}))
  tiny = min(KINDS["draw_figure"].estimate_wh,
             KINDS["count_plants"].estimate_wh) - 0.01
  assert run(maker, until=2000.0, pack_wh=tiny) == []
  assert maker.deferred > 0, "the deferral was not even noticed"
  # ...and the same world with a pack that could fund them offers both.
  ok = producer(cadence(kinds={"draw_figure": {}, "count_plants": {}}))
  assert {t.kind for t in run(ok, until=2000.0, pack_wh=9.9)} == \
    {"draw_figure", "count_plants"}


def test_the_producer_runs_during_a_charge_and_cannot_touch_the_robot():
  """⚠ THE NEW SEAM'S OWN RISK. `_task_step` hangs off the PHYSICS hook as of
  issue #23, so unlike every earlier task branch it runs while the robot is
  pressed against the charge pins. What keeps "a task never delays a charge"
  true is therefore no longer only the branch order in `run()` (which
  `test_tasks.py` pins with a real mission) -- it is that this seam is
  incapable of doing anything the robot does.

  Shown to fail by having `_task_step` queue the errand for a claimed job, or
  set `self.state`: both are one line, and both would let a job put up during
  a charge interrupt it.
  """
  import mujoco
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  # Priced for room_hub, as production does (issue #15): a bare board falls
  # back to `TaskKind.estimate_wh`, which is deliberately the dearest world's
  # figure and so refuses room_hub a carry it does for 0.57 Wh.
  from pluggybot.hub.energy import load as load_energy
  b = board(energy=load_energy("room_hub"))
  life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world="room_hub", errand=False, tasks=b,
                         producer=None, battery_wh=cfg["battery_wh"],
                         rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"])
  life.producer = TaskProducer(b, cadence(firstAtS=0.0, everyS=1.0, initial=0,
                                          kinds={"fetch_module": {}}),
                               {"module": ["module_lcd"]})
  life.state = "CHARGE"
  life.battery.energy_wh = life.low_battery_wh * 0.5
  before = (life.state, list(life.errands), life.battery.energy_wh,
            list(life.claimed))
  life.data.time = 500.0
  life._task_step()
  assert b.offered(), "the world stopped offering work during a charge"
  assert (life.state, list(life.errands), life.battery.energy_wh,
          list(life.claimed)) == before, \
    "the producer's seam reached into the mission loop"


def test_a_use_phase_is_skipped_when_the_robot_never_reached_the_board():
  """⚠ THE HANG ISSUE #23 FOUND, and it was never about cadence.

  `run_errand` called `mission.drive_to(*errand.use_at)` and THREW THE ANSWER
  AWAY, then narrated "arrived" whatever had happened. `drive_to` returns
  False when it stagnated or could not plan, so a use-phase ran anyway --
  the pen probing for a board seven metres away, forever, until the battery
  died. Measured: ten minutes of wall clock with nothing in the log after
  `USE_TOOL: arrived`, and no test in the repo could see it because nothing
  had ever sent the robot to the FAR whiteboard. The producer does, because
  it rotates targets instead of always naming the first one.

  What must happen instead is the boring thing: skip the use-phase, put the
  tool back, and let the evaluator find no ink and fail the job. A robot that
  could not get there and a robot that got there and drew badly are different
  events, and both have to end with the module on the rack.

  Shown to fail by restoring `self.mission.drive_to(...)` as a bare call: the
  use-phase runs, and this test hangs rather than failing -- which is exactly
  what it is guarding.
  """
  import mujoco
  from pluggybot.hub.errand import Errand
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world="room_hub", errand=False,
                         battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"])
  ran: list[bool] = []

  def use(_life):
    ran.append(True)
    return {"drew": True}

  # A destination the robot cannot get to, and a use-phase that records
  # whether it was reached. Everything physical is stubbed: what is under
  # test is one branch, not a mission.
  errand = Errand(name="draw:nowhere", module="module_pen", station_y=0.0,
                  use_at=(500.0, 500.0), use=use)
  life.mission.drive_to = lambda *a, **kw: False
  life.mission.swap_at_bay = lambda *a, **kw: None
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True,
                                                     "hung": True}
  result = life.run_errand(errand)
  assert not ran, "the use-phase ran at a board the robot never reached"
  assert result["error"] == "never reached the use pose"
  assert result["stowed"], "the tool was abandoned instead of being put back"


def test_an_errand_that_navigates_itself_is_not_skipped_for_falling_short():
  """...and the OTHER half of that gate, which the first version did not have.

  Skipping the use-phase is right for a pen, which must be at the board. It is
  wrong for an errand that does its own navigation: the census's `use_at` is
  the FIRST POINT OF THE SURVEY ROUTE its use-phase then drives itself, so the
  pre-positioning drive is redundant by construction.

  ⚠ MEASURED, and it cost a fixture. With one gate for every errand, the home
  showcase recording's drive stopped 1.96 m short of the garden vantage and
  the census was sent home -- from a spot it had previously surveyed 100 % of
  the zone from, counting 4 of 4 plants for +20. No committed recording then
  showed the LCD in `count` mode, which `test_the_home_fixture_shows_the_
  census_answer` guards and which is half of what the showcase queue exists to
  show. Set `needs_use_pose` back to True for every errand and that fixture
  test is what fails.
  """
  import mujoco
  from pluggybot.hub.errand import Errand
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world="room_hub", errand=False,
                         battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"])
  ran: list[bool] = []

  def use(_life):
    ran.append(True)
    return {"counted": 4, "truth": 4}

  errand = Errand(name="census:garden", module="module_lcd", station_y=0.0,
                  use_at=(500.0, 500.0), use=use, needs_use_pose=False)
  life.mission.drive_to = lambda *a, **kw: False      # the drive gave up
  life.mission.swap_at_bay = lambda *a, **kw: None
  life.mission.swap.module_state = lambda *a, **kw: {"on_fork": True,
                                                     "hung": True}
  result = life.run_errand(errand)
  assert ran, "an errand that navigates itself was skipped for a short drive"
  assert "error" not in result
  assert result["counted"] == 4
  assert result["stowed"]


def test_the_real_census_errand_navigates_itself():
  """The flag is only worth anything if the errand that needs it carries it,
  and a default of True means forgetting is silent. Asserted on the errand the
  home mission actually builds rather than on one written here."""
  census = [e for e in lc.errands_for("census", "home", None)
            if e.task == "census"]
  assert census, "the home world builds no census errand"
  assert all(e.needs_use_pose is False for e in census)
  # ...and the flag is not simply False everywhere: a pen MUST be at a board,
  # which is the case the gate was added for in the first place.
  assert all(e.needs_use_pose for e in lc.errands_for("carry", "room_hub", None))


def test_a_producer_world_stands_by_instead_of_calling_it_a_day():
  """⚠ MEASURED, and it is the difference between a mission and a world.

  `run()` broke the moment it had nothing to do, which was right when the
  only work was a preset queue -- a queue that never grows back. A world with
  a producer in it does grow back, so the break turned "the next job is 70
  seconds away" into "mission complete": a real home run mapped the house,
  did both the jobs it could reach and ended at t=410 with the next offer due
  at t=480.

  Standing by is bounded, so `needs_charge` is still re-checked every few
  seconds, and it happens only where a producer is attached -- an ordinary
  preset-errand mission ends exactly as it always did, which is what every
  other mission test in the suite is asserting.

  Shown to fail by restoring the bare `else: break`: the run ends at t~0 with
  the whole budget unspent.
  """
  import mujoco
  cfg = lc.world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  # Priced for room_hub, as production does (issue #15): a bare board falls
  # back to `TaskKind.estimate_wh`, which is deliberately the dearest world's
  # figure and so refuses room_hub a carry it does for 0.57 Wh.
  from pluggybot.hub.energy import load as load_energy
  b = board(energy=load_energy("room_hub"))
  life = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world="room_hub", errand=False, tasks=b,
                         battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"])
  # A world that will not offer anything for a long time, so the ONLY thing
  # keeping the loop alive is the decision to wait for it.
  life.producer = TaskProducer(b, cadence(firstAtS=1e6, everyS=1e6, initial=0,
                                          kinds={"fetch_module": {}}),
                               {"module": ["module_lcd"]})
  # Nothing physical: the branch under test is the last one in `run()`, and
  # mapping a room to reach it would make this a mission test.
  life.explore = lambda *a, **kw: setattr(life, "map_done", True)
  life.mission.drive_to = lambda *a, **kw: True

  # Two stand-by slices is the whole claim; 30 s of real physics for it was
  # 27 s of wall clock in a suite where this file costs under a second.
  budget = 12.0
  r = life.run(cfg["start"], max_sim_time=budget)

  assert r["state"] == "DONE"
  assert life.data.time >= budget, \
    f"the robot went home at t={life.data.time:.1f}s with work coming"
  # ...and a world with NO producer still ends the moment it is done, which
  # is what every other mission test in this suite depends on.
  quiet = lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                          world="room_hub", errand=False,
                          battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                          grid_bounds=cfg["grid_bounds"],
                          low_battery_wh=cfg["low_battery_wh"])
  quiet.explore = lambda *a, **kw: setattr(quiet, "map_done", True)
  quiet.mission.drive_to = lambda *a, **kw: True
  quiet.run(cfg["start"], max_sim_time=budget)
  assert quiet.data.time < budget, "a preset-errand mission stopped ending"
