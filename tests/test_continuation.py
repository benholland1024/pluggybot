"""A restart is a continuation (issue #345).

The served world's process ends -- a deploy, a crash, the hourly ceiling --
and the next one carries on from the world it saved. These pin the rules
that make it so, each on the cheapest world that can fail for the right
reason: the physics by name and exactly, what a robot believes and whether
it is dead, the jobs it holds, the History line, the keeper's seam, and
the ways a saved world is NOT put back (a crash loop, a changed world, maps
of another epoch). The bookkeeping is pinned on the stub body
(`tests/test_body.py`); what is the body's -- its bodies, its maps, its
sensors stepping on -- on the served humanoid. The flown parity check is
`scripts/determinism_spike.py --resume-at`.
"""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import continuation, tick
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import HUMANOID_HOME, HubLifecycle, task_board, world_config
from pluggybot.mission.errand import programmed_errand
from pluggybot.mind.thoughts import HISTORY
from pluggybot.procedure.steps import Program, Step
from pluggybot.robot import world_spec
from pluggybot.tick import MissionAborted
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


class _Stop(Exception):
  """Raised at the top of the day loop: the prelude is what is under test."""


def _world():
  cfg = world_config(HUMANOID_HOME)
  model = world_spec(cfg["model"]).compile()
  return cfg, model, mujoco.MjData(model)


def _board(tmp_path, tasks: bool, rebase: bool, world: str = HUMANOID_HOME):
  return (task_board(str(tmp_path / "tasks.json"), world=world, rebase=rebase)
          if tasks else None)


def _stub(tmp_path, tasks: bool = False, rebase: bool = True, **kw) -> HubLifecycle:
  """The loop's bookkeeping on the stub body: every other claim."""
  return stub_life(tasks=_board(tmp_path, tasks, rebase, kw.get("world", HUMANOID_HOME)),
                   **kw)


def _humanoid(tmp_path, **kw) -> HubLifecycle:
  """The served body in its house (#559): for a claim about its body, its
  maps or its senses."""
  cfg, model, data = _world()
  return HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME,
                      battery_wh=cfg["battery_wh"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], **kw)


def _walk(life, seconds: float, v: float, w: float) -> None:
  life.body.run(life.body.velocity_routine(seconds, v, w))


def _lap():
  """A procedure's errand, as the robot's own `procedure:<name>` builds it
  (`lifecycle.errand_from`): no job behind it."""
  return programmed_errand(Program.single("lap", [Step("wait", {"seconds": 1.0})]),
                           task="program", name="procedure")


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


#: ...and a humanoid's save on a step NO controller's period divides (#559):
#: 687 steps, against its reach's 5 and its policies' 10 -- on one they all
#: divide, a fresh controller ticks in step with the saved one by chance.
OFF_EVERY_PERIOD_S = 1.374


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


# ---- the physics, by name and exactly ----------------------------------------


def test_the_physics_comes_back_exactly_and_the_warm_start_is_why():
  """MEASURED before the design: MuJoCo's state put back field by field
  steps on bit for bit -- and without the solver's warm start the same
  step parts by 1e-12, which a parity check would read as a changed world.
  The premise half fails if a save stops carrying `qacc_warmstart`."""
  _, model, data = _world()
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
  _, model, data = _world()
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
  """A new build's world reorders `qpos`: the save names every joint (an
  unnamed one by its body) and a world that lists them the other way round
  still gets each back."""
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
  life = _humanoid(tmp_path)
  life.body.start_at(*world_config(HUMANOID_HOME)["start"])
  life.body.start_discovery()
  _walk(life, 1.0, 0.3, 0.6)                           # scans, travel
  life.battery.energy_wh = 0.37
  life.floor_explored = True
  life.explore_deadline = 12.5
  snap = _saved(life, tmp_path)

  back = _humanoid(tmp_path)
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


def test_a_restored_humanoid_steps_on_exactly_as_if_nothing_had_stopped(tmp_path):
  """The parity rule on the humanoid (#559), saved mid-walk where nothing
  in particular decides next: its drivers' steps and targets, the gains they
  set in the MODEL, its senses' noise, its reach and its reckoning's planted
  foot all ride the save, and the same walk after a restore steps the same
  body and paints the same maps."""
  life = _humanoid(tmp_path, near_field=True)
  start = world_config(HUMANOID_HOME)["start"]
  life.body.start_at(*start)
  last = continuation.LastStep(life)
  life.body.step_hooks.append(last.hook)
  m = life.body.mission
  m.run(m._drive_routine(OFF_EVERY_PERIOD_S, 0.5, 0.3))
  snap = _saved(life, tmp_path, last)
  m.run(m._drive_routine(1.0, 0.6, -0.4))

  back = _humanoid(tmp_path, near_field=True)
  back.begin(start)
  continuation.restore([back], snap)
  b = back.body.mission
  b.run(b._drive_routine(1.0, 0.6, -0.4))
  assert np.array_equal(back.data.qpos, life.data.qpos)
  assert back.body.pose == life.body.pose
  assert np.array_equal(back.body.grid.grid, life.body.grid.grid)
  assert np.array_equal(back.near_field.height, life.near_field.height, equal_nan=True)


def test_a_dead_robot_is_still_dead_after_a_restart_and_pays_no_second_heart(tmp_path):
  """A robot saved dead at 0 % that came back alive would die again on its
  first step -- a second heart for one death -- or stand up free at 100 %.
  It comes back dead, its stand-up clock going on."""
  life = _stub(tmp_path, mortal=True, restart_after_s=300.0)
  life.home_pose = tuple(world_config(HUMANOID_HOME)["start"])
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


def test_a_restart_keeps_the_hand_job_its_robot_holds_and_says_so(tmp_path):
  """The robot that held the job is still that robot: a hand job (#559)
  comes back held by it -- the procedure is its to run, and to say `done`
  -- nothing is queued, and History says so and names the procedure the
  world stopped in the middle of. Until #345 the job came back `failed`
  ("interrupted by a restart")."""
  life = _stub(tmp_path, tasks=True)
  task = life.tasks.offer("open_drawer", "drawer", t=0.0)
  assert life.tasks.claim(task.id, robot=life.root, t=0.0) is not None
  life._errand_now = _lap()                       # its procedure running, and
  snap = _saved(life, tmp_path)                   # the world stops

  back = _stub(tmp_path, tasks=True, rebase=False)
  assert back.tasks[task.id].state == "claimed"   # not failed
  assert back.tasks[task.id].claimed_by == back.root
  _prelude(back, snap)
  assert back.errands == []
  said = _history(back)
  assert any(f"the job {task.id} (open_drawer) is still mine -- the procedure is "
             "mine to run, and to say done" in ln for ln in said), said
  assert any("the restart cut short procedure; it is not queued again" in ln
             for ln in said), said


def test_a_restart_during_an_errand_keeps_the_job_and_says_so(tmp_path, monkeypatch):
  """...and a job with an ERRAND behind it comes back claimed by its robot,
  its errand queued again under the task's id, and History names what was
  cut short. No kind the humanoids are offered is discharged by an errand
  (#559), so the kind is a copy of the drawer's carrying its steps."""
  import dataclasses
  from pluggybot.economy.tasks import KINDS
  from pluggybot.procedure.steps import Program, Step
  monkeypatch.setitem(KINDS, "errand_job", dataclasses.replace(
    KINDS["open_drawer"], name="errand_job", discharge="errand", draws=False))
  life = _stub(tmp_path, tasks=True)
  steps = Program.single("job", [Step("wait", {"seconds": 1.0})]).as_dict()
  task = life.tasks.offer("errand_job", "drawer",
                          params={"minutes": 16, "procedure": steps}, t=0.0)
  assert life._claim_task(task.id)
  [errand] = life.errands
  life.tasks.start(task.id)                       # the errand is running...
  life._errand_now = errand                       # ...and the world stops
  snap = _saved(life, tmp_path)

  back = _stub(tmp_path, tasks=True, rebase=False)
  assert back.tasks[task.id].state == "claimed"   # not failed
  assert back.tasks[task.id].claimed_by == back.root
  _prelude(back, snap)
  assert [e.task_id for e in back.errands] == [task.id]
  assert any(f"the restart cut short {errand.name}; the job {task.id} "
             "(errand_job) is still mine, and is queued again" in ln
             for ln in _history(back)), _history(back)


def test_the_board_keeps_a_claim_and_gives_back_what_nobody_can_finish(tmp_path):
  """`TaskBoard.load`'s rule since #345: a procedure job claimed or active
  comes back as it was (its robot says `done`); a job nothing in the new
  process carries on -- a kind discharged neither by an errand nor by a
  procedure, here the act -- is given back to the offer if it was claimed
  and failed if it was under way. A claim held by a robot not in the new
  world is given back by `release_absent`."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  drawer = b.offer("open_drawer", "drawer", t=0.0)
  door = b.offer("open_door", "door", t=0.0)
  claimed = b.offer("take_points", "robot", t=0.0)
  begun = b.offer("take_points", "robot", t=0.0)
  other = b.offer("open_chest", "chest", t=0.0)
  b.claim(drawer.id, robot="pluggybot", t=1.0)
  b.claim(door.id, robot="pluggybot", t=1.0)
  b.start(door.id, t=2.0)
  b.claim(claimed.id, robot="pluggybot", t=1.0)
  b.claim(begun.id, robot="pluggybot", t=1.0)
  b.start(begun.id, t=2.0)
  b.claim(other.id, robot="r2_pluggybot", t=1.0)

  back = TaskBoard(path)
  assert (back[drawer.id].state, back[drawer.id].claimed_by) == ("claimed", "pluggybot")
  assert back[door.id].state == "active"
  assert (back[claimed.id].state, back[claimed.id].claimed_by) == ("offered", "")
  assert back[begun.id].state == "failed"
  assert [t.id for t in back.interrupted] == [begun.id]
  assert [t.id for t in back.held_by("pluggybot")] == [drawer.id, door.id]
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
  b = task_board(str(path), world=HUMANOID_HOME)
  task = b.offer("open_drawer", "drawer", t=0.0)
  back = task_board(str(path), world=HUMANOID_HOME)
  assert back[task.id].estimate_wh == task.estimate_wh == 7.082
  assert KINDS["open_drawer"].estimate_wh > 7.085, "the premise: two prices"
  assert back[task.id].claimable(10.0, pack_wh=7.085)


def test_a_world_that_carries_on_keeps_its_deadlines_on_its_own_clock(tmp_path):
  """The clock goes on (issue #345), so an offer's deadline means what it
  said; a world built from its XML starts at 0 and gets what was left."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("open_drawer", "drawer", ttl=720.0, t=3125.0)
  assert TaskBoard(path, rebase=False)[task.id].deadline == 3845.0
  assert TaskBoard(path)[task.id].deadline == 720.0


# ---- History ---------------------------------------------------------------


def test_history_says_restarted_where_the_process_did_and_nothing_where_it_did_not(tmp_path):
  """"Woke up" / "finished the day" were written at every hourly restart,
  and the robots planned round a day that had not ended. A run that keeps
  its world says nothing at its end; the next says it restarted; a run
  with no saved world says it woke up, as it always did."""
  start = world_config(HUMANOID_HOME)["start"]
  life = _stub(tmp_path)
  life.continuing = True
  life.run(start, max_sim_time=0.0)
  lines = _history(life)
  assert any(f"woke up in {HUMANOID_HOME}" in ln for ln in lines)
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
  start = world_config(HUMANOID_HOME)["start"]
  life = _humanoid(tmp_path)
  life.body.start_at(*start)
  _walk(life, 1.0, 0.3, 0.3)
  life.battery.energy_wh = 0.42
  snap = _saved(life, tmp_path)
  snap.meta["fingerprint"] = "another-world"

  back = _humanoid(tmp_path)
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


#: The quadrupeds' served pack, Wh (their 12S1P Molicel P45B, `quadruped-final`):
#: what a save of their house carried, and what one made before packs were
#: saved with their size is carried on from (`continuation.QUAD_PACK_WH`).
QUAD_PACK_WH = 12 * 3.6 * 4.5


def _quads(tmp_path, **kw) -> HubLifecycle:
  """A robot of the quadrupeds' house as the stub flies it, on their
  served pack: its save is relabelled theirs (`_theirs`), the world their
  served world was -- the change of body (#559)."""
  return _stub(tmp_path, battery_wh=QUAD_PACK_WH, mortal=True, restart_after_s=300.0, **kw)


def _theirs(life, tmp_path) -> continuation.Snapshot:
  """`life`'s save, as the quadrupeds' served world wrote it (`home_quad`)."""
  snap = _saved(life, tmp_path)
  snap.meta["world"] = "home_quad"
  continuation.write(snap, tmp_path / "world.npz")
  return snap


def _humanoids(tmp_path, **kw) -> HubLifecycle:
  return _stub(tmp_path, battery_wh=world_config(HUMANOID_HOME)["hosting_battery_wh"],
               mortal=True, restart_after_s=300.0, **kw)


def test_the_quadrupeds_house_carries_on_in_the_humanoids_at_the_same_share(tmp_path):
  """THE CHANGE OF BODY (#559): Luca and Rowan keep who they are in new
  bodies, so a save of the quadrupeds' house carries on in the humanoids' as
  a changed world does -- the clock and the pack, never the bodies or the
  maps -- the pack at the SHARE it was saved at (40 % of the quadruped's 194
  Wh is 40 % of the humanoid's 778, never 78 Wh of it), and History says
  why."""
  quad, humanoid = _quads(tmp_path), _humanoids(tmp_path)
  quad.battery.energy_wh = 0.4 * quad.battery.capacity_wh
  _hold(quad, 0.5)
  _theirs(quad, tmp_path)
  loaded = continuation.load(tmp_path / "world.npz", HUMANOID_HOME)
  assert loaded.snapshot is not None, loaded.why
  _prelude(humanoid, loaded.snapshot)
  assert humanoid.data.time >= quad.data.time
  assert humanoid.battery.capacity_wh > 3 * quad.battery.capacity_wh
  assert humanoid.battery.fraction == pytest.approx(0.4)
  assert not humanoid.resumed["inPlace"]
  assert any(f"could not put me back where I was ({continuation.BODY_CHANGED})" in ln
             for ln in _history(humanoid))


def test_a_save_from_before_packs_kept_their_size_carries_on_at_the_served_packs_share(tmp_path):
  """The deployed quadrupeds' saves carry no `capacityWh`: carried across
  the change of body, one is on the quadruped's served pack -- the one world
  ever served -- and a robot at 60 Wh of its 194 wakes at 31 %."""
  quad, humanoid = _quads(tmp_path), _humanoids(tmp_path)
  quad.battery.energy_wh = 60.0
  snap = _theirs(quad, tmp_path)
  del snap.meta["robots"][quad.root]["capacityWh"]
  _prelude(humanoid, snap)
  assert humanoid.battery.fraction == pytest.approx(60.0 / QUAD_PACK_WH)
  assert continuation.QUAD_PACK_WH == QUAD_PACK_WH


def test_a_robot_dead_in_its_old_body_is_dead_in_its_new_one(tmp_path):
  """Its deaths are the robot's, whatever body it wakes in: dead flat at the
  switch, it is still down, its stand-up clock going on."""
  quad, humanoid = _quads(tmp_path), _humanoids(tmp_path)
  quad.home_pose = tuple(world_config(HUMANOID_HOME)["start"])
  quad.battery.energy_wh = 0.0
  _hold(quad, 0.5)
  assert quad.dead is not None
  _prelude(humanoid, _theirs(quad, tmp_path))
  assert humanoid.dead == quad.dead and humanoid.reset_in_s == quad.reset_in_s


def test_a_held_job_this_world_cannot_do_is_given_back_after_a_restart(tmp_path):
  """The restart keeps the jobs, and a job the robot held that nothing in
  this world can do -- here a hand job whose prop this house does not have
  -- is back on offer, said in History, and nothing is queued (#559: the
  quadrupeds' jobs in the humanoids' house were the first)."""
  life = _stub(tmp_path, tasks=True)
  task = life.tasks.offer("open_drawer", "lab_drawer", t=0.0)
  assert life.tasks.claim(task.id, robot=life.root, t=0.0) is not None
  snap = _saved(life, tmp_path)

  back = _stub(tmp_path, tasks=True, rebase=False)
  _prelude(back, snap)
  assert back.errands == []
  assert back.tasks[task.id].state == "offered"
  assert any(f"the job {task.id} (open_drawer) is back on offer: nothing in this "
             "world can do it" in ln for ln in _history(back)), _history(back)


def test_the_same_body_on_another_pack_keeps_its_watt_hours(tmp_path):
  """The share is the change of BODY's alone (#559 review): the same body
  restarted on another pack -- an operator's `PLUGGY_BATTERY_WH` -- keeps
  the watt-hours it had, as it always has, up to what the pack holds."""
  small = _stub(tmp_path, battery_wh=100.0)
  small.battery.energy_wh = 60.0
  snap = _saved(small, tmp_path)
  big = _stub(tmp_path, battery_wh=200.0)
  _prelude(big, snap)
  assert big.battery.energy_wh == pytest.approx(60.0)


def test_another_houses_save_is_never_carried_on(tmp_path):
  """Only another body in the SAME house (`lifecycle.HOUSES`)."""
  snap = _theirs(_quads(tmp_path), tmp_path)
  snap.meta["world"] = "room_hub"
  continuation.write(snap, tmp_path / "world.npz")
  loaded = continuation.load(tmp_path / "world.npz", HUMANOID_HOME)
  assert loaded.snapshot is None and "room_hub" in loaded.why


@pytest.mark.parametrize("old", [None, 1, 2])
def test_maps_of_another_epoch_are_dropped_as_a_changed_worlds_are(tmp_path, old):
  """#425: hours of heading walked while the quadrupeds lay, and every map
  kept since was laid askew -- and a place found in one is wrong too. The
  world is the same, so the fingerprint cannot say it: a save of an older
  `MAP_EPOCH` (one written before there was one included) keeps the pack
  and the clock and puts back no body, belief or map, and says why. #476:
  the deployed volume's saves are of epoch 1, and their maps were found
  askew again, so a save of epoch 1 is dropped too -- and then of epoch 2,
  laid askew again before a fixture could drop a map that was."""
  life = _humanoid(tmp_path)
  life.body.start_at(*world_config(HUMANOID_HOME)["start"])
  _walk(life, 1.0, 0.3, 0.0)
  life.battery.energy_wh = 0.42
  snap = _saved(life, tmp_path)
  assert snap.meta["mapEpoch"] == continuation.MAP_EPOCH
  if old is None:
    del snap.meta["mapEpoch"]
  else:
    snap.meta["mapEpoch"] = old

  back = _humanoid(tmp_path)
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
    assert continuation.load(path, HUMANOID_HOME).snapshot is not None
  refused = continuation.load(path, HUMANOID_HOME)
  assert refused.snapshot is None and "never got past it" in refused.why
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  assert continuation.load(path, HUMANOID_HOME).snapshot is not None
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
  assert meta["resumes"] == 0 and meta["world"] == HUMANOID_HOME

  life._standing_up = True
  keeper.request_stop("SIGTERM")
  _hold(life, 0.01)                                # deferred, not raised
  life._standing_up = False
  with pytest.raises(MissionAborted, match="SIGTERM"):
    _hold(life, 0.01)


def test_a_save_waits_out_a_get_up_part_way(tmp_path):
  """A save mid get-up holds half a policy no file keeps (`Keeper.busy`):
  the humanoid is `getting_up` while its get-up policy has the legs
  (`HumanoidBody.posture`), the word a save waits on, and saves again once
  it stands."""
  from pluggybot.humanoid.body import HumanoidBody

  def posture(legs: str) -> str:
    return HumanoidBody.posture.fget(SimpleNamespace(mission=SimpleNamespace(legs=legs)))
  assert posture("getup") in continuation.MOVING_POSTURES
  assert posture("stand") not in continuation.MOVING_POSTURES
  assert posture("walk") not in continuation.MOVING_POSTURES
  life = _stub(tmp_path)
  keeper = continuation.Keeper([life], tmp_path / "world.npz", every_s=0.1)
  life.body.posture = posture("getup")
  assert keeper.busy()
  _hold(life, 0.25)
  assert keeper.saves == 0, "saved mid get-up"
  life.body.posture = posture("stand")
  assert not keeper.busy()
  _hold(life, 0.25)
  assert keeper.saves >= 1


def test_the_pair_carries_on_both_robots_from_one_saved_world(tmp_path):
  """The deployed shape (issue #181): one world, two robots, ONE save --
  both packs, both poses, both maps, keyed by each robot's root -- and the
  world's own activities with them (the pair's encounter count here)."""
  from pluggybot.pair import build_pair
  cfg = world_config(HUMANOID_HOME)
  starts = (cfg["start"], cfg["start2"])
  lives = build_pair(HUMANOID_HOME)
  for life, start in zip(lives, starts):
    life.body.start_at(*start)
  _walk(lives[1], 0.5, 0.2, 0.3)
  lives[0].battery.energy_wh, lives[1].battery.energy_wh = 0.3, 0.6
  lives[0].encounters.count = 4
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)

  back = build_pair(HUMANOID_HOME, resume=snap)
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


@pytest.mark.parametrize("where", ["the station", "its own start"])
def test_a_pair_saved_lying_on_each_other_is_put_apart_on_restore(tmp_path, where):
  """A world saved before the pair collided (#455) can hold the two on one
  spot -- the quadrupeds' jams at afb473a lay 0.05-0.2 m apart on their
  dock's standoff -- and stepped now, the contacts throw them out of each
  other: a robot down at each of three. The later robot is put at its first
  clear start (another robot's, where its own is the one taken), and the
  restart's History line says where it was and why it was moved; a pair
  saved apart comes back in place
  (`test_the_pair_carries_on_both_robots_from_one_saved_world`)."""
  import math

  from pluggybot.lifecycle import UNCLIP_M
  from pluggybot.pair import build_pair
  cfg = world_config(HUMANOID_HOME)
  starts = (cfg["start"], cfg["start2"])
  if where == "the station":
    a, b = (3.45, -0.35, math.radians(-90.0)), (3.40, -0.25, math.radians(-80.0))
  else:
    b = starts[1]
    a = (b[0] + 0.05, b[1], b[2])
  lives = build_pair(HUMANOID_HOME, thoughts_root=str(tmp_path / "before"))
  lives[0].body.start_at(*a)
  lives[1].body.start_at(*b)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)

  back = build_pair(HUMANOID_HOME, resume=snap, thoughts_root=str(tmp_path / "after"))
  for life, start in zip(back, starts):
    life.begin(start)
  got = continuation.restore(back, snap)
  assert got["inPlace"] and got["unclipped"] == [back[1].root]
  put = starts[1] if where == "the station" else starts[0]
  x, y, _ = back[1].body.true_pose()
  assert math.dist((x, y), put[:2]) < 0.05
  ax, ay, _ = back[0].body.true_pose()
  assert math.dist((ax, ay), a[:2]) < 0.05, "the first robot moved"
  line = back[1]._resumed_line(back[1].resumed)
  assert line.startswith(f"the world restarted; I carried on from ({b[0]:.1f}, {b[1]:.1f})"), line
  whose = "my start" if where == "the station" else f"{back[0].robot_name}'s start"
  assert (f"inside {back[0].robot_name}'s body there, and the world put me at "
          f"{whose} ({put[0]:.1f}, {put[1]:.1f})") in line, line
  mujoco.mj_forward(back[0].model, back[0].data)
  assert not back[1].unclip() and UNCLIP_M > 0.008


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
  keeper saves it (its last step kept), on a step no controller's period
  divides (#559)."""
  from pluggybot.pair import build_pair
  world, at = HUMANOID_HOME, OFF_EVERY_PERIOD_S / 2
  cfg = world_config(world)
  starts = (cfg["start"], cfg["start2"])

  def fly(lives):
    tick.run_many([(life.body.stepper,
                    life.body.mission._drive_routine(at, 0.15, w))
                   for life, w in zip(lives, (-0.3, 0.4))])

  lives = build_pair(world, near_field=True)
  for life, start in zip(lives, starts):
    life.body.start_at(*start)
  path = tmp_path / "world.npz"
  keeper = continuation.Keeper(lives, path, every_s=1e9)    # saved when told
  fly(lives)
  assert keeper.save()
  snap = continuation.read(path)
  assert "before" in snap.meta["physics"]
  fly(lives)

  back = build_pair(world, near_field=True, resume=snap)
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
  got = continuation.load(path, HUMANOID_HOME)
  assert got.snapshot is None and "could not be read" in got.why


def test_a_kept_claim_that_never_finishes_is_failed_after_three_restarts(tmp_path):
  """The world's crash-loop guard counts saves, and a claim lives on the
  board: a job whose errand crashed the process was re-queued by every
  process after it (found in review, six in a row). Each restart that
  takes the claim up counts; the fourth fails it, on the wire."""
  from pluggybot.economy.tasks import MAX_TAKE_UPS
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("open_drawer", "drawer", t=0.0)
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


def test_a_map_that_does_not_fit_this_build_is_explored_again(tmp_path):
  """The fingerprint is the geometry's; a new grid extent or resolution is
  not. The grid is then left empty -- and the map's verdicts with it, or a
  robot that believed its map complete would never explore the empty one
  it has (found in review). Where it IS stays put."""
  life = _humanoid(tmp_path)
  life.body.start_at(*world_config(HUMANOID_HOME)["start"])
  life.floor_explored = True
  life.blacklist = {(1, 2)}
  snap = _saved(life, tmp_path)
  snap.arrays["pluggybot/grid"] = np.zeros((4, 4))
  back = _humanoid(tmp_path)
  _prelude(back, snap)
  assert back.resumed["inPlace"] and back.body.pose == life.body.pose
  assert not back.floor_explored and back.blacklist == set()


def test_what_begin_says_is_stamped_on_the_restored_clock(tmp_path):
  """`begin` announces what a restart failed (a job under way that nothing
  in the new process carries on) before `restore`; with the clock set first,
  the line and its event carry the world's time, not 0 -- rows on the
  observatory in the order they happened."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  begun = b.offer("take_points", "robot", t=0.0)
  b.claim(begun.id, robot="pluggybot", t=1.0)
  b.start(begun.id, t=2.0)
  life = _stub(tmp_path / "saved")
  life.data.time = 500.0
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path, tasks=True, rebase=False)
  heard = []
  back.say_hooks.append(lambda t, line: heard.append((t, line)))
  back.run(world_config(HUMANOID_HOME)["start"], max_sim_time=0.0, resume=snap)
  [(t, _)] = [h for h in heard if "interrupted by a restart" in h[1]]
  assert t == 500.0


# ---- the second look ------------------------------------------------------------


def test_a_pair_is_saved_after_every_robots_step_not_between_them(tmp_path):
  """A pair's step runs each robot's bookkeeping in turn (`tick.run_many`):
  hooked on the FIRST robot, the save caught the second one's pack and
  reckoner a step behind its body. Hooked on the last, everything the step
  does has been done."""
  from pluggybot.pair import build_pair
  lives = build_pair(HUMANOID_HOME)
  keeper = continuation.Keeper(lives, tmp_path / "world.npz")
  assert keeper.step_hook in lives[-1].body.step_hooks
  assert keeper.step_hook not in lives[0].body.step_hooks
  assert all(life.continuing for life in lives)


def test_a_new_claim_starts_its_restart_count_at_nothing(tmp_path):
  """The count is a CLAIM's: a job given back and taken by another robot
  must not arrive already most of the way to being failed."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("open_drawer", "drawer", t=0.0)
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
  life._errand_now = _lap()
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path)
  _prelude(back, snap)
  assert back.errands == []
  assert any(f"the restart cut short {life._errand_now.name}; it is not "
             "queued again" in ln for ln in _history(back))
  back.begin(world_config(HUMANOID_HOME)["start"])
  assert back.resumed is None
