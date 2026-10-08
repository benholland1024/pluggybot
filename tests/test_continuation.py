"""A restart is a continuation (issue #345).

The served world's process ends -- a deploy, a crash, the hourly ceiling --
and the next one carries on from the world it saved. These pin the rules
that make it so, each on the cheapest world that can fail for the right
reason: the physics by name and exactly, what a robot believes and whether
it is dead, the jobs it holds, the History line, the keeper's seam, and
the ways a saved world is NOT put back (a crash loop, a changed world, maps
of another epoch). The bookkeeping is pinned on the stub body
(`tests/test_body.py`); what is the body's -- its bodies, its maps, its
sensors stepping on -- on the served quadruped. The flown parity check is
`scripts/determinism_spike.py --resume-at`.
"""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import continuation, tick
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import (QUAD_HOME, STOW_RETRIES, HubLifecycle, cage_errand,
                                 task_board, world_config, world_facts)
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang
from pluggybot.mind.thoughts import HISTORY
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import world_spec
from pluggybot.tick import MissionAborted
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


class _Stop(Exception):
  """Raised at the top of the day loop: the prelude is what is under test."""


def _world():
  cfg = world_config(QUAD_HOME)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  return cfg, spec, model, mujoco.MjData(model)


def _board(tmp_path, tasks: bool, rebase: bool):
  return (task_board(str(tmp_path / "tasks.json"), world=QUAD_HOME, rebase=rebase)
          if tasks else None)


def _quad(tmp_path, tasks: bool = False, rebase: bool = True, **kw) -> HubLifecycle:
  """The served body in its house: for a claim about bodies, maps or senses."""
  cfg, spec, model, data = _world()
  return HubLifecycle(model, data, realtime=False, world=QUAD_HOME, spec=spec,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"],
                      tasks=_board(tmp_path, tasks, rebase), **kw)


def _stub(tmp_path, tasks: bool = False, rebase: bool = True, **kw) -> HubLifecycle:
  """The loop's bookkeeping on the stub body: every other claim."""
  return stub_life(tasks=_board(tmp_path, tasks, rebase), **kw)


def _restored(life, snap=None, max_sim_time: float = 600.0):
  """`begin` and the restore, as `run` does them: the day routine, undriven."""
  day = life.begin(world_config(life.world)["start"], max_sim_time=max_sim_time)
  if snap is not None:
    continuation.restore([life], snap)
  return day


def _prelude(life, snap=None, max_sim_time: float = 600.0, day=None):
  """...and the day routine driven up to its loop's first pass."""
  if day is None:
    day = _restored(life, snap, max_sim_time)

  def stop():
    raise _Stop

  life.at_loop_top.append(stop)
  with pytest.raises(_Stop):
    life.body.run(day)


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.texts[HISTORY].splitlines() if ln.strip()]


def _saved(life, tmp_path, last_step=None) -> continuation.Snapshot:
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint,
                                          last_step=last_step), path)
  return continuation.read(path)


#: A save AT one of the walking policy's decisions: 300 steps from a stand
#: (`walker.every` 10). The policy reads the forward pass (`xmat`, the gyro)
#: on the first step after the restore, which is why the last step is
#: stepped again (#420).
AT_A_DECISION_S = 0.6


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


# ---- the physics, by name and exactly ----------------------------------------


def test_the_physics_comes_back_exactly_and_the_warm_start_is_why():
  """MEASURED before the design: MuJoCo's state put back field by field
  steps on bit for bit -- and without the solver's warm start the same
  step parts by 1e-12, which a parity check would read as a changed world.
  The premise half fails if a save stops carrying `qacc_warmstart`."""
  _, _, model, data = _world()
  data.ctrl[:] = np.random.default_rng(0).uniform(-0.5, 0.5, model.nu)
  for _ in range(400):
    mujoco.mj_step(model, data)
  index, arrays = continuation.physics(model, data)
  for _ in range(400):
    mujoco.mj_step(model, data)
  want = np.concatenate([data.qpos, data.qvel])

  def again(drop_warm: bool) -> np.ndarray:
    d = mujoco.MjData(model)
    a = dict(arrays, warm=np.zeros_like(arrays["warm"])) if drop_warm else arrays
    continuation.put_physics(model, d, index, a)
    for _ in range(400):
      mujoco.mj_step(model, d)
    return np.concatenate([d.qpos, d.qvel])

  assert np.array_equal(again(drop_warm=False), want)
  assert not np.array_equal(again(drop_warm=True), want)


def test_a_restore_steps_again_the_step_it_was_saved_after():
  """A running world's forward pass -- its contacts, positions, sensors --
  is the one its last step began with, a step behind its `qpos`, and
  everything between two steps reads it: the walking policy deciding, a
  fork's contacts. Forwarded fresh at the saved instant, a restore parted
  wherever the policy decided first (#420). The save carries where that
  step began and the restore steps it again: what a reader sees comes back
  bit for bit. The premise half: forwarded at the instant, none of it
  does; and a step whose input changed after it is not replayed."""
  _, _, model, data = _world()
  rng = np.random.default_rng(0)
  last = continuation.LastStep(SimpleNamespace(model=model, data=data))
  for _ in range(400):
    data.ctrl[:] = rng.uniform(-0.5, 0.5, model.nu)
    mujoco.mj_step(model, data)
    last.hook()
  index, arrays = continuation.physics(model, data, last)

  def seen(d) -> list:
    return [d.xpos, d.xmat, d.sensordata, d.qacc, d.actuator_force,
            d.contact.geom[:d.ncon], d.contact.pos[:d.ncon]]

  back = mujoco.MjData(model)
  continuation.put_physics(model, back, index, arrays)
  assert not all(np.array_equal(a, b) for a, b in zip(seen(back), seen(data)))
  assert continuation.replay(model, back, index, arrays)
  assert all(np.array_equal(a, b) for a, b in zip(seen(back), seen(data)))
  assert np.array_equal(back.qpos, data.qpos) and back.time == data.time
  # ...and a step whose input moved after it lands elsewhere: the saved
  # state is put back, forwarded at the instant
  moved = dict(arrays, ctrl=arrays["ctrl"] + 0.1)
  other = mujoco.MjData(model)
  continuation.put_physics(model, other, index, moved)
  assert not continuation.replay(model, other, index, moved)
  assert np.array_equal(other.qpos, data.qpos) and other.time == data.time


def test_a_body_is_put_back_by_name_not_by_where_it_sits_in_the_state():
  """A built tool re-hung in another order, or a new build's world,
  reorders `qpos`: the save names every joint (an unnamed one by its body)
  and a world that lists them the other way round still gets each back."""
  xml = """<mujoco><worldbody>
    <body name="{a}" pos="0 0 1"><freejoint/><geom size=".1"/></body>
    <body name="{b}" pos="1 0 1"><freejoint/><geom size=".1"/></body>
  </worldbody></mujoco>"""
  one = mujoco.MjModel.from_xml_string(xml.format(a="alpha", b="beta"))
  two = mujoco.MjModel.from_xml_string(xml.format(a="beta", b="alpha"))
  d1 = mujoco.MjData(one)
  d1.qpos[0:3] = [5.0, 6.0, 7.0]                 # alpha
  d1.qpos[7:10] = [-1.0, -2.0, -3.0]             # beta
  index, arrays = continuation.physics(one, d1)
  d2 = mujoco.MjData(two)
  got = continuation.put_physics(two, d2, index, arrays)
  assert got == {"joints": 2, "jointsMissed": 0}
  assert list(d2.qpos[7:10]) == [5.0, 6.0, 7.0]  # alpha is second here
  assert list(d2.qpos[0:3]) == [-1.0, -2.0, -3.0]


# ---- what the robot believes, and whether it is alive -------------------------


def test_a_restart_restores_the_pack_the_pose_the_maps_and_the_clock(tmp_path):
  """The issue's round trip: step a world, save it, build a fresh one from
  the save. Pose, pack, belief, the occupancy grid, the dock's frame, the
  clock -- each asserted, because each one missing is a restart that
  resets something: a robot at 5 % in minute 59 woke at 100 %."""
  life = _quad(tmp_path)
  life.body.start_at(*world_config(QUAD_HOME)["start"])
  life.body.start_discovery()
  life.body.mission._drive(1.0, 0.15, 0.6)             # scans, travel
  life.battery.energy_wh = 0.37
  life.floor_explored = True
  life.explore_deadline = 12.5
  snap = _saved(life, tmp_path)

  back = _quad(tmp_path)
  _prelude(back, snap)
  assert back.data.time == life.data.time
  assert np.array_equal(back.data.qpos, life.data.qpos)
  assert back.battery.energy_wh == 0.37
  assert back.body.pose == life.body.pose
  assert np.array_equal(back.body.grid.grid, life.body.grid.grid)
  assert (life.body.grid.grid != 0).any()     # the map had something in it
  assert back.body.rack == life.body.rack
  assert back.floor_explored and back.explore_deadline == 12.5
  # ...and it never went back to the start pose or looked round: nothing moved
  assert back.data.time == life.data.time


def test_a_restored_robot_senses_on_exactly_as_if_nothing_had_stopped(tmp_path):
  """The parity rule, pinned in a second of physics: the same walk after a
  restore steps the same bodies AND paints the same maps. MEASURED, a
  scan's noise generator re-seeded at a restart painted a different map
  and the route off it parted 3 s later; the IMU, the odometry and the
  depth camera have one each. Saved where the walking policy decides next,
  which parted at once until the last step was stepped again (#420)."""
  life = _quad(tmp_path, near_field=True)
  start = world_config(QUAD_HOME)["start"]
  life.body.start_at(*start)
  last = continuation.LastStep(life)
  life.body.step_hooks.append(last.hook)
  life.body.mission._drive(AT_A_DECISION_S, 0.1, 0.4)
  walker = life.body.mission.walker
  assert walker.steps % walker.every == 0          # it decides next
  snap = _saved(life, tmp_path, last)
  life.body.mission._drive(0.6, 0.12, -0.3)

  back = _quad(tmp_path, near_field=True)
  back.begin(start)
  continuation.restore([back], snap)
  back.body.mission._drive(0.6, 0.12, -0.3)
  assert np.array_equal(back.data.qpos, life.data.qpos)
  assert back.body.pose == life.body.pose
  assert np.array_equal(back.body.grid.grid, life.body.grid.grid)
  assert np.array_equal(back.near_field.height, life.near_field.height,
                        equal_nan=True)


def test_a_dead_robot_is_still_dead_after_a_restart_and_pays_no_second_heart(tmp_path):
  """A robot saved dead at 0 % that came back alive would die again on its
  first step -- a second heart for one death -- or stand up free at 100 %.
  It comes back dead, its stand-up clock going on."""
  life = _stub(tmp_path, mortal=True, restart_after_s=300.0)
  life.home_pose = tuple(world_config(QUAD_HOME)["start"])
  life.battery.energy_wh = 0.0
  _hold(life, 0.5)
  assert life.dead is not None and len(life.deaths) == 1
  left = life.reset_in_s
  snap = _saved(life, tmp_path)

  back = _stub(tmp_path, mortal=True, restart_after_s=300.0)
  _prelude(back, snap)
  assert back.dead == life.dead
  assert back.reset_in_s == left
  _hold(back, 1.0)
  assert back.deaths == []                        # no second death
  assert any("still down (flat)" in ln for ln in _history(back))


# ---- the jobs --------------------------------------------------------------


def test_a_restart_during_an_errand_keeps_the_job_and_says_so(tmp_path, monkeypatch):
  """The robot that held the job is still that robot: an errand job comes
  back CLAIMED by it and its errand is queued again, a module the restart
  left on the fork is stowed first, and History names what was cut short.
  Until #345 the job came back `failed` ("interrupted by a restart")."""
  from pluggybot.procedure import steps
  life = _stub(tmp_path, tasks=True)
  task = life.tasks.offer("feed_mouse", "lab", t=0.0)
  assert life._claim_task(task.id, answer="eating")
  [errand] = life.errands
  life.tasks.start(task.id)                       # the errand is running...
  life._errand_now = errand                       # ...and the world stops
  snap = _saved(life, tmp_path)

  stowed = []
  monkeypatch.setattr(steps, "_carried", lambda life: "module_lcd")
  monkeypatch.setattr(steps, "_stow",
                      lambda life, args: (stowed.append(True), tick.result({"ok": True}))[1])
  back = _stub(tmp_path, tasks=True, rebase=False)
  assert back.tasks[task.id].state == "claimed"   # not failed
  assert back.tasks[task.id].claimed_by == back.root
  _prelude(back, snap)
  assert [e.task_id for e in back.errands] == [task.id]
  assert stowed == [True]
  assert any(f"the restart cut short {errand.name}; the job {task.id} "
             "(feed_mouse) is still mine, and is queued again" in ln
             for ln in _history(back))


def test_the_board_keeps_a_claim_and_gives_back_what_nobody_can_finish(tmp_path):
  """`TaskBoard.load`'s rule since #345: an errand job claimed or active
  comes back claimed by its robot, a procedure job stays active (its robot
  says `done`), and a game -- whose referee lived in the process -- fails
  if it was under way and is offered again if it was not. A claim held by
  a robot not in the new world is given back by `release_absent`."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  carry = b.offer("fetch_module", "module_lcd", t=0.0)
  tower = b.offer("stack_tower", "tower", t=0.0)
  game = b.offer("hide_and_seek", QUAD_HOME, t=0.0)
  other = b.offer("draw_figure", "whiteboard_a", params={"program": "house"}, t=0.0)
  b.claim(carry.id, robot="pluggybot", t=1.0)
  b.start(carry.id, t=2.0)
  b.claim(tower.id, robot="pluggybot", t=1.0)
  b.start(tower.id, t=1.0)
  b.claim(game.id, robot="pluggybot", t=1.0, role="hider")
  b.claim(game.id, robot="r2_pluggybot", t=1.0, role="seeker")
  b.start(game.id, t=2.0)
  b.claim(other.id, robot="r2_pluggybot", t=1.0)

  back = TaskBoard(path)
  assert (back[carry.id].state, back[carry.id].claimed_by) == ("claimed", "pluggybot")
  assert back[tower.id].state == "active"
  assert back[game.id].state == "failed"
  assert [t.id for t in back.interrupted] == [game.id]
  assert [t.id for t in back.held_by("pluggybot")] == [carry.id, tower.id]
  gone = back.release_absent({"pluggybot"})
  assert [t.id for t in gone] == [other.id]
  assert back[other.id].state == "offered" and back[other.id].claimed_by == ""


def test_an_offer_keeps_this_worlds_price_across_a_restart(tmp_path):
  """Found by the parity check: `Task.from_json` priced a reloaded offer at
  its kind's generic figure instead of the world's measured one, so after
  every restart a pack could not take a job it had been offered, or took
  one it could not fund."""
  from pluggybot.economy.tasks import KINDS
  path = tmp_path / "tasks.json"
  b = task_board(str(path), world=QUAD_HOME)
  task = b.offer("feed_mouse", "lab", t=0.0)
  back = task_board(str(path), world=QUAD_HOME)
  assert back[task.id].estimate_wh == task.estimate_wh == 1.77
  assert KINDS["feed_mouse"].estimate_wh != 1.77, "the premise: two prices"
  assert not back[task.id].claimable(10.0, pack_wh=1.5)


def test_a_world_that_carries_on_keeps_its_deadlines_on_its_own_clock(tmp_path):
  """The clock goes on (issue #345), so an offer's deadline means what it
  said; a world built from its XML starts at 0 and gets what was left."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("draw_figure", "whiteboard_a", params={"program": "house"},
                 ttl=720.0, t=3125.0)
  assert TaskBoard(path, rebase=False)[task.id].deadline == 3845.0
  assert TaskBoard(path)[task.id].deadline == 720.0


# ---- a tool on the fork (issue #420) -------------------------------------------


def _returns(life, day, passes: int) -> list[str]:
  """What `day` narrates of the tool on the fork, driven to its
  `passes`-th pass of the loop."""
  said, seen = [], []
  life.say_hooks.append(lambda t, line: said.append(line))

  def stop():
    seen.append(life.data.time)
    if len(seen) == passes:
      raise _Stop
  life.at_loop_top.append(stop)
  with pytest.raises(_Stop):
    life.body.run(day)
  return [ln for ln in said if "SWAP_RETURN" in ln or "on my fork" in ln]


def _returns_fail(life):
  """Every return this body makes leaves the tool on its fork."""
  life.body.stow_tool_routine = lambda station_y, module: tick.result("arrived")
  return life


def test_a_tool_on_the_fork_between_errands_is_the_loops_after_a_restart(tmp_path):
  """Saved between errands with a tool still on the fork after a failed
  return, the robot carries on as the day flown straight through does: the
  loop's next return, under the next number. The restart used to take it
  home on its own first -- a return nobody counted -- and forget the count,
  so every restart granted returns the straight day never had (#420)."""
  life = _returns_fail(_stub(tmp_path))
  life.body.holding = "module_lcd"
  life._stow_tries = STOW_RETRIES - 1               # one return left
  saved = []
  life.at_loop_top.append(lambda: saved or saved.append(_saved(life, tmp_path)))
  straight = _returns(life, life.begin(world_config(QUAD_HOME)["start"]), passes=2)
  assert straight == [f"SWAP_RETURN again ({STOW_RETRIES}/{STOW_RETRIES}): "
                      "module_lcd is still on my fork -- hanging it back before "
                      "anything else"]

  back = _returns_fail(_stub(tmp_path))
  assert _returns(back, _restored(back, saved[0]), passes=2) == straight


def test_a_return_a_restart_cut_short_is_made_again_under_its_number(tmp_path):
  """A return counts once it has run: one the world stopped in the middle
  of -- a keeper saves mid-walk -- is the next process's to make again,
  not one already spent (#420)."""
  life = _stub(tmp_path)
  life.body.holding = "module_lcd"

  def slow(station_y, module):
    yield from life.body.hold_routine(1.0)
    return "arrived"
  life.body.stow_tool_routine = slow
  cut = []

  def stop() -> None:
    if life.data.time >= 0.5 and not cut:
      cut.append(_saved(life, tmp_path))
      raise _Stop
  life.body.step_hooks.append(stop)
  with pytest.raises(_Stop):
    life.body.run(life._stow_retry_routine())

  back = _returns_fail(_stub(tmp_path))
  assert _returns(back, _restored(back, cut[0]), passes=2) == [
    f"SWAP_RETURN again (1/{STOW_RETRIES}): module_lcd is still on my fork -- "
    "hanging it back before anything else"]


# ---- History ---------------------------------------------------------------


def test_history_says_restarted_where_the_process_did_and_nothing_where_it_did_not(tmp_path):
  """"Woke up" / "finished the day" were written at every hourly restart,
  and the robots planned round a day that had not ended. A run that keeps
  its world says nothing at its end; the next says it restarted; a run
  with no saved world says it woke up, as it always did."""
  start = world_config(QUAD_HOME)["start"]
  life = _stub(tmp_path)
  life.continuing = True
  life.run(start, max_sim_time=0.0)
  lines = _history(life)
  assert any(f"woke up in {QUAD_HOME}" in ln for ln in lines)
  assert not any("finished the day" in ln or "day ended" in ln for ln in lines)
  snap = _saved(life, tmp_path)

  back = _stub(tmp_path)
  _prelude(back, snap)
  lines = _history(back)
  assert any("the world restarted; I carried on from" in ln for ln in lines)
  assert not any("woke up" in ln for ln in lines)

  plain = _stub(tmp_path / "plain")
  plain.run(start, max_sim_time=0.0)
  assert any("finished the day" in ln for ln in _history(plain))


def test_a_resumed_run_counts_its_budget_from_where_it_starts(tmp_path):
  """`max_sim_time` is a RUN's, now that the clock goes on: a world saved at
  t=4000 and given 600 s runs to 4600, where the old absolute reading would
  have ended it before its first pass -- for a robot the save never had as
  well, which starts from its start pose on the same clock."""
  life = _stub(tmp_path)
  life.data.time = 4000.0
  snap = _saved(life, tmp_path)
  snap.meta["robots"] = {}                         # nobody it knows
  back = _stub(tmp_path)
  _prelude(back, snap, max_sim_time=600.0)
  assert back.max_sim_time == 4600.0
  assert back.resumed is None                      # fresh, on the clock
  assert back.survival_since >= 4000.0


# ---- when a saved world is NOT put back ----------------------------------------


def test_a_changed_world_keeps_the_pack_and_the_clock_but_not_the_bodies(tmp_path):
  """A new build that moved a wall: a robot restored into it could be
  inside the wall, and its map is of a house that is gone. Pack, clock and
  jobs are the robot's whatever the house looks like; the bodies and the
  maps come back only into the world they were saved from."""
  start = world_config(QUAD_HOME)["start"]
  life = _quad(tmp_path)
  life.body.start_at(*start)
  life.body.mission._drive(0.5, 0.2, 0.3)
  life.battery.energy_wh = 0.42
  snap = _saved(life, tmp_path)
  snap.meta["fingerprint"] = "another-world"

  back = _quad(tmp_path)
  day = _restored(back, snap)
  assert back.battery.energy_wh == 0.42
  assert back.data.time == life.data.time
  assert not back.resumed["inPlace"]
  assert not np.array_equal(back.data.qpos, life.data.qpos)
  back.body.look_around_routine = lambda: tick.result(None)
  _prelude(back, day=day)
  assert back.body.pose == pytest.approx(start)    # from the start pose
  assert life.body.pose != pytest.approx(start)
  assert any("could not put me back where I was (the world itself changed"
             in ln for ln in _history(back))


@pytest.mark.parametrize("old", [None, 1])
def test_maps_of_another_epoch_are_dropped_as_a_changed_worlds_are(tmp_path, old):
  """#425: hours of heading walked while the quadrupeds lay, and every map
  kept since was laid askew -- and a place found in one is wrong too. The
  world is the same, so the fingerprint cannot say it: a save of an older
  `MAP_EPOCH` (one written before there was one included) keeps the pack
  and the clock and puts back no body, belief or map, and says why. #476:
  the deployed volume's saves are of epoch 1, and their maps were found
  askew again, so a save of epoch 1 is dropped too."""
  life = _quad(tmp_path)
  life.body.start_at(*world_config(QUAD_HOME)["start"])
  life.body.mission._drive(0.5, 0.2, 0.0)
  life.battery.energy_wh = 0.42
  snap = _saved(life, tmp_path)
  assert snap.meta["mapEpoch"] == continuation.MAP_EPOCH
  if old is None:
    del snap.meta["mapEpoch"]
  else:
    snap.meta["mapEpoch"] = old

  back = _quad(tmp_path)
  _restored(back, snap)
  assert back.battery.energy_wh == 0.42 and back.data.time == life.data.time
  assert not back.resumed["inPlace"]
  assert back.resumed["why"] == continuation.MAPS_DROPPED
  assert not np.array_equal(back.data.qpos, life.data.qpos)


def test_a_saved_world_that_keeps_crashing_is_left_after_three_tries(tmp_path):
  """A state that kills the process would otherwise be put back into the
  same death for ever. `load` counts itself before it is trusted and a
  save puts the count back to 0; three loads with no save in between and
  the next start is a fresh one, said why."""
  life = _stub(tmp_path)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  for _ in range(continuation.MAX_RESUMES):
    assert continuation.load(path, QUAD_HOME).snapshot is not None
  refused = continuation.load(path, QUAD_HOME)
  assert refused.snapshot is None and "never got past it" in refused.why
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  assert continuation.load(path, QUAD_HOME).snapshot is not None
  assert continuation.load(path, "another_world").snapshot is None


# ---- the keeper -------------------------------------------------------------


def test_the_keeper_saves_on_the_seam_and_a_signal_stops_at_a_step(tmp_path):
  """Saved every `every_s` of sim time on the first robot's hooks; and a
  stop only ASKS -- a SIGTERM lands between any two bytecodes, a ledger
  write or a death half done, so the next step boundary raises instead.
  A robot mid stand-up is half stood up, and neither happens then."""
  life = _stub(tmp_path)
  path = tmp_path / "world.npz"
  keeper = continuation.Keeper([life], path, every_s=0.1)
  assert life.continuing
  _hold(life, 0.25)
  assert keeper.saves == 2 and path.exists()
  meta = json.loads(str(np.load(path)["meta"]))
  assert meta["resumes"] == 0 and meta["world"] == QUAD_HOME

  life._standing_up = True
  keeper.request_stop("SIGTERM")
  _hold(life, 0.01)                                # deferred, not raised
  life._standing_up = False
  with pytest.raises(MissionAborted, match="SIGTERM"):
    _hold(life, 0.01)


def test_the_pair_carries_on_both_robots_from_one_saved_world(tmp_path):
  """The deployed shape (issue #181): one world, two robots, ONE save --
  both packs, both poses, both maps, keyed by each robot's root -- and the
  world's own activities with them (the pair's encounter count here)."""
  from pluggybot.pair import build_pair
  cfg = world_config(QUAD_HOME)
  starts = (cfg["start"], cfg["start2"])
  lives = build_pair(QUAD_HOME)
  for life, start in zip(lives, starts):
    life.body.start_at(*start)
  lives[1].body.mission._drive(0.5, 0.2, 0.3)
  lives[0].battery.energy_wh, lives[1].battery.energy_wh = 0.3, 0.6
  lives[0].encounters.count = 4
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)

  back = build_pair(QUAD_HOME, resume=snap)
  for life, start in zip(back, starts):
    life.begin(start)
  got = continuation.restore(back, snap)
  assert got["inPlace"] and got["robots"] == ["pluggybot", "r2_pluggybot"]
  assert np.array_equal(back[0].data.qpos, lives[0].data.qpos)
  for was, now in zip(lives, back):
    assert now.battery.energy_wh == was.battery.energy_wh
    assert now.body.pose == was.body.pose
    assert np.array_equal(now.body.grid.grid, was.body.grid.grid)
  assert back[0].encounters.count == 4


def test_a_world_saved_mid_charge_says_the_charge_was_cut_short(tmp_path):
  """A restart ends the charge routine: the robot is told its charge was
  cut short, rather than believe it charged on."""
  life = _stub(tmp_path)
  life.state = "CHARGE"
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path)
  _prelude(back, snap)
  assert any("it cut my charge short" in ln for ln in _history(back))


def test_the_pair_steps_on_exactly_after_a_restart(tmp_path):
  """The parity rule on the deployed shape: two robots from one loop, the
  depth cameras on, the pair's encounters sensing -- the same walk after a
  restore steps the same world and paints the same two maps, saved as the
  keeper saves it (its last step kept) where both policies decide next."""
  from pluggybot.pair import build_pair
  cfg = world_config(QUAD_HOME)
  starts = (cfg["start"], cfg["start2"])

  def fly(lives):
    tick.run_many([(life.body.stepper,
                    life.body.mission._drive_routine(AT_A_DECISION_S, 0.15, w))
                   for life, w in zip(lives, (-0.3, 0.4))])

  lives = build_pair(QUAD_HOME, near_field=True)
  for life, start in zip(lives, starts):
    life.body.start_at(*start)
  path = tmp_path / "world.npz"
  keeper = continuation.Keeper(lives, path, every_s=1e9)    # saved when told
  fly(lives)
  assert keeper.save()
  snap = continuation.read(path)
  assert "before" in snap.meta["physics"]
  fly(lives)

  back = build_pair(QUAD_HOME, near_field=True, resume=snap)
  for life, start in zip(back, starts):
    life.begin(start)
  continuation.restore(back, snap)
  fly(back)
  assert np.array_equal(back[0].data.qpos, lives[0].data.qpos)
  for was, now in zip(lives, back):
    assert now.body.pose == was.body.pose
    assert np.array_equal(now.body.grid.grid, was.body.grid.grid)
    assert np.array_equal(now.near_field.height, was.near_field.height,
                          equal_nan=True)


# ---- what the review found -----------------------------------------------------


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04 torn", None])
def test_a_save_that_cannot_be_read_is_a_fresh_start_never_a_crash(tmp_path, content):
  """Raised out of `load`, an empty or torn file killed the process before
  `MAX_RESUMES` could count, and `restart: unless-stopped` looped on it
  (found in review: EOFError, BadZipFile)."""
  life = _stub(tmp_path)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  data = path.read_bytes()
  path.write_bytes(content if content is not None else data[: len(data) // 2])
  got = continuation.load(path, QUAD_HOME)
  assert got.snapshot is None and "could not be read" in got.why


def test_a_kept_errand_claim_that_never_finishes_is_failed_after_three_restarts(tmp_path):
  """The world's crash-loop guard counts saves, and a claim lives on the
  board: a job whose errand crashes the process was re-queued by every
  process after it (found in review, six in a row). Each restart that
  takes the claim up counts; the fourth fails it, on the wire."""
  from pluggybot.economy.tasks import MAX_TAKE_UPS
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("fetch_module", "module_lcd", t=0.0)
  b.claim(task.id, robot="pluggybot", t=1.0)
  for n in range(1, MAX_TAKE_UPS + 1):
    b = TaskBoard(path, rebase=False)
    assert b.take_up(task.id, t=float(n)).state == "claimed"
  b = TaskBoard(path, rebase=False)
  heard = []
  b.on_event.append(heard.append)
  gone = b.take_up(task.id, t=9.0)
  assert gone.state == "failed" and "3 restarts" in gone.verdict["reason"]
  assert heard[0]["type"] == "task_resolved"
  assert "restarts" not in b[task.id].as_dict()   # the file's, not the wire's


def test_a_restart_mid_swap_backs_out_and_says_so(tmp_path, monkeypatch):
  """Mid-pick the fork is under a module still HANGING on its bay; the
  return backs it out, and History must not say a tool was carried off
  (found in review: "the restart left module_pen on my fork")."""
  from pluggybot.procedure import steps
  life = _stub(tmp_path)
  life._errand_now = programmed_errand(              # a fetch, mid-pick
    lang.compile_procedure('def job():\n  fetch("module_lcd")\n',
                           world_facts(QUAD_HOME)), task="program", name="procedure")
  snap = _saved(life, tmp_path)
  monkeypatch.setattr(steps, "_carried", lambda life: "module_lcd")
  monkeypatch.setattr(steps, "_stow", lambda life, args: tick.result({"ok": True}))
  back = _stub(tmp_path)
  _prelude(back, snap)
  assert any("the restart stopped me mid-swap at module_lcd's bay; I backed "
             "out and it hangs there" in ln for ln in _history(back))


def test_a_map_that_does_not_fit_this_build_is_explored_again(tmp_path):
  """The fingerprint is the geometry's; a new grid extent or resolution is
  not. The grid is then left empty -- and the map's verdicts with it, or a
  robot that believed its map complete would never explore the empty one
  it has (found in review). Where it IS stays put."""
  life = _quad(tmp_path)
  life.body.start_at(*world_config(QUAD_HOME)["start"])
  life.floor_explored = True
  life.blacklist = {(1, 2)}
  snap = _saved(life, tmp_path)
  snap.arrays["pluggybot/grid"] = np.zeros((4, 4))
  back = _quad(tmp_path)
  _prelude(back, snap)
  assert back.resumed["inPlace"] and back.body.pose == life.body.pose
  assert not back.floor_explored and back.blacklist == set()


def test_a_module_the_world_no_longer_has_is_not_restored_as_the_one_watched(tmp_path):
  """`module` names what the power model watches; a retired built tool's
  name restored there fails the next `module_state` read (found in
  review)."""
  life = _stub(tmp_path)
  snap = _saved(life, tmp_path)
  snap.meta["robots"]["pluggybot"]["module"] = "module_retired_long_ago"
  back = _stub(tmp_path)
  _prelude(back, snap)
  assert back.module == "module_lcd"


def test_what_begin_says_is_stamped_on_the_restored_clock(tmp_path):
  """`begin` announces what a restart failed (a game) before `restore`;
  with the clock set first, the line and its event carry the world's time,
  not 0 -- rows on the observatory in the order they happened."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  game = b.offer("hide_and_seek", QUAD_HOME, t=0.0)
  b.claim(game.id, robot="pluggybot", t=1.0, role="hider")
  b.claim(game.id, robot="r2_pluggybot", t=1.0, role="seeker")
  b.start(game.id, t=2.0)
  life = _stub(tmp_path / "saved")
  life.data.time = 500.0
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path, tasks=True, rebase=False)
  heard = []
  back.say_hooks.append(lambda t, line: heard.append((t, line)))
  back.run(world_config(QUAD_HOME)["start"], max_sim_time=0.0, resume=snap)
  [(t, _)] = [h for h in heard if "interrupted by a restart" in h[1]]
  assert t == 500.0


def test_an_offered_challenge_finds_its_props_where_the_offer_says(tmp_path):
  """The offer says where the blocks stand, and the hourly reset was what
  made that true: with the world carried on, a failed attempt would leave
  them wherever it dropped them, for good. The house sets them out as it
  offers the job (issue #345) -- all but one against a robot, which is
  somebody's mid-job."""
  from pluggybot.challenge import stack
  life = _quad(tmp_path, tasks=True)
  life.body.start_at(*world_config(QUAD_HOME)["start"])
  qadr = int(life.model.joint(int(life.model.body("block_1").jntadr[0])).qposadr[0])
  home = life.model.qpos0[qadr:qadr + 3]
  life.data.qpos[qadr:qadr + 3] = home + [0.6, -0.4, 0.0]  # knocked aside
  held = int(life.model.joint(int(life.model.body("block_2").jntadr[0])).qposadr[0])
  mujoco.mj_forward(life.model, life.data)            # inside the robot's torso
  life.data.qpos[held:held + 3] = life.data.geom_xpos[life.model.geom("torso").id]
  mujoco.mj_forward(life.model, life.data)
  against = life.data.qpos[held:held + 3].copy()
  block = life.model.body("block_2").id
  mine = list(robot_geoms(life.model, life.root))
  g = life.data.contact.geom[:life.data.ncon]
  at = g[(life.model.geom_bodyid[g] == block).any(axis=1)]
  assert np.isin(at, mine).any()                      # it IS touching the robot
  said = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life.tasks.offer("stack_tower", "tower", t=0.0)
  assert np.allclose(life.data.qpos[qadr:qadr + 7], life.model.qpos0[qadr:qadr + 7])
  assert np.array_equal(life.data.qpos[held:held + 3], against)
  assert any("set out block_1" in ln for ln in said)
  assert stack.BLOCKS[1] == "block_1"


# ---- the second look ------------------------------------------------------------


def test_a_pair_is_saved_after_every_robots_step_not_between_them(tmp_path):
  """A pair's step runs each robot's bookkeeping in turn (`tick.run_many`):
  hooked on the FIRST robot, the save caught the second one's pack and
  reckoner a step behind its body. Hooked on the last, everything the step
  does has been done."""
  from pluggybot.pair import build_pair
  lives = build_pair(QUAD_HOME)
  keeper = continuation.Keeper(lives, tmp_path / "world.npz")
  assert keeper.step_hook in lives[-1].body.step_hooks
  assert keeper.step_hook not in lives[0].body.step_hooks
  assert all(life.continuing for life in lives)


def test_a_new_claim_starts_its_restart_count_at_nothing(tmp_path):
  """The count is a CLAIM's: a job given back and taken by another robot
  must not arrive already most of the way to being failed."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("fetch_module", "module_lcd", t=0.0)
  b.claim(task.id, robot="r2_pluggybot", t=1.0)
  b.take_up(task.id)
  b.take_up(task.id)
  assert b[task.id].restarts == 2
  b.release(task.id)
  b.claim(task.id, robot="pluggybot", t=2.0)
  assert b[task.id].restarts == 0


def test_an_errand_cut_short_with_no_job_behind_it_is_named_and_left(tmp_path):
  """What a restart ended that no job asked for -- the run's own errand, a
  procedure the robot started -- is named in History and not queued again,
  and the line does not pretend nobody wanted it. And a second run on the
  same lifecycle does not carry the first one's restart into its own."""
  life = _stub(tmp_path)
  life._errand_now = cage_errand(QUAD_HOME, "feed")
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path)
  _prelude(back, snap)
  assert back.errands == []
  assert any(f"the restart cut short {life._errand_now.name}; it is not "
             "queued again" in ln for ln in _history(back))
  back.begin(world_config(QUAD_HOME)["start"])
  assert back.resumed is None
