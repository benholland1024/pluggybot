"""A restart is a continuation (issue #345).

The served world's process ends -- a deploy, a crash, the hourly ceiling --
and until this every new process built the world from its XML: both robots
at their spawn poses on a full pack with an empty map, the job in hand
failed. These pin the rules that make a restart carry on instead, each on
the cheapest world that can fail for the right reason: the physics by name
and exactly, what a robot believes and whether it is dead, the jobs it
holds, the History line, the keeper's seam, and the two ways a saved world
is NOT put back (a crash loop, a changed world). The flown parity check is
`scripts/determinism_spike.py --resume-at`.
"""

import json

import mujoco
import numpy as np
import pytest

from pluggybot import continuation, tick
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import HubLifecycle, task_board, world_config
from pluggybot.mind.thoughts import HISTORY
from pluggybot.mission.mission import MissionAborted
from pluggybot.robot import world_spec


class _Stop(Exception):
  """Raised at the top of the day loop: the prelude is what is under test."""


def _world(world: str = "room_hub"):
  cfg = world_config(world)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  return cfg, spec, model, mujoco.MjData(model)


def _life(tmp_path, world: str = "room_hub", tasks: bool = False,
          rebase: bool = True, **kw) -> HubLifecycle:
  cfg, spec, model, data = _world(world)
  board = (task_board(str(tmp_path / "tasks.json"), world=world, rebase=rebase)
           if tasks else None)
  return HubLifecycle(model, data, realtime=False, world=world, spec=spec,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      tasks=board, **kw)


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
    life.mission.run(day)


def _history(life) -> list[str]:
  return [ln for ln in life.thoughts.texts[HISTORY].splitlines() if ln.strip()]


def _saved(life, tmp_path) -> continuation.Snapshot:
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  return continuation.read(path)


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
  the save. Pose, pack, belief, the occupancy grid, the rack's sightings,
  the clock -- each asserted, because each one missing is a restart that
  resets something: a robot at 5 % in minute 59 woke at 100 %."""
  life = _life(tmp_path)
  cfg = world_config("room_hub")
  life.mission.start_at(*cfg["start"])
  life.mission.start_discovery()
  life.mission._drive(1.5, 0.15, 0.6)             # scans, a look, travel
  life.battery.energy_wh = 0.37
  life.map_done = True
  life.explore_deadline = 12.5
  snap = _saved(life, tmp_path)

  back = _life(tmp_path)
  _prelude(back, snap)
  assert back.data.time == life.data.time
  assert np.array_equal(back.data.qpos, life.data.qpos)
  assert back.battery.energy_wh == 0.37
  assert back.mission.pose == life.mission.pose
  assert np.array_equal(back.mission.grid.grid, life.mission.grid.grid)
  assert (life.mission.grid.grid != 0).any()     # the map had something in it
  assert back.mission.rack == life.mission.rack
  assert ([lm.x for lm in back.mission.finder.landmarks.landmarks]
          == [lm.x for lm in life.mission.finder.landmarks.landmarks])
  assert back.map_done and back.explore_deadline == 12.5
  # ...and it never went back to the start pose or spun: nothing moved
  assert back.data.time == life.data.time


def test_a_restored_robot_senses_on_exactly_as_if_nothing_had_stopped(tmp_path):
  """The parity rule, pinned in a second of physics: the same drive after
  a restore steps the same bodies AND paints the same map. MEASURED, the
  scan's noise generator re-seeded at a restart painted a different map
  and the route off it parted 3 s later; the depth camera has one too."""
  life = _life(tmp_path, near_field=True)
  cfg = world_config("room_hub")
  life.mission.start_at(*cfg["start"])
  life.mission._drive(0.6, 0.1, 0.4)
  snap = _saved(life, tmp_path)
  life.mission._drive(0.6, 0.12, -0.3)

  back = _life(tmp_path, near_field=True)
  back.begin(cfg["start"])
  continuation.restore([back], snap)
  back.mission._drive(0.6, 0.12, -0.3)
  assert np.array_equal(back.data.qpos, life.data.qpos)
  assert back.mission.pose == life.mission.pose
  assert np.array_equal(back.mission.grid.grid, life.mission.grid.grid)
  assert np.array_equal(back.near_field.height, life.near_field.height,
                        equal_nan=True)


def test_a_dead_robot_is_still_dead_after_a_restart_and_pays_no_second_heart(tmp_path):
  """A robot saved dead at 0 % that came back alive would die again on its
  first step -- a second heart for one death -- or, before #345, stand up
  free at 100 %. It comes back dead, its stand-up clock going on."""
  life = _life(tmp_path, mortal=True, restart_after_s=300.0)
  cfg = world_config("room_hub")
  life.mission.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is not None and len(life.deaths) == 1
  left = life.reset_in_s
  snap = _saved(life, tmp_path)

  back = _life(tmp_path, mortal=True, restart_after_s=300.0)
  _prelude(back, snap)
  assert back.dead == life.dead
  assert back.reset_in_s == left
  back.mission._drive(1.0, 0.0, 0.0)
  assert back.deaths == []                        # no second death
  assert any("still down (flat)" in ln for ln in _history(back))


# ---- the jobs --------------------------------------------------------------


def test_a_restart_during_an_errand_keeps_the_job_and_says_so(tmp_path, monkeypatch):
  """The robot that held the job is still that robot: an errand job comes
  back CLAIMED by it and its errand is queued again, a module the restart
  left on the fork is stowed first, and History names what was cut short.
  Until #345 the job came back `failed` ("interrupted by a restart")."""
  from pluggybot.procedure import steps
  life = _life(tmp_path, tasks=True)
  task = life.tasks.offer("fetch_module", "module_lcd", t=0.0)
  assert life._claim_task(task.id)
  [errand] = life.errands
  life.tasks.start(task.id)                       # the errand is running...
  life._errand_now = errand                       # ...and the world stops
  snap = _saved(life, tmp_path)

  stowed = []
  monkeypatch.setattr(steps, "_carried", lambda life: "module_lcd")
  monkeypatch.setattr(steps, "_stow",
                      lambda life, args: (stowed.append(True), tick.result({"ok": True}))[1])
  back = _life(tmp_path, tasks=True, rebase=False)
  assert back.tasks[task.id].state == "claimed"   # not failed
  assert back.tasks[task.id].claimed_by == back.root
  _prelude(back, snap)
  assert [e.task_id for e in back.errands] == [task.id]
  assert stowed == [True]
  assert any(f"the restart cut short {errand.name}; the job {task.id} "
             "(fetch_module) is still mine, and is queued again" in ln
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
  game = b.offer("hide_and_seek", "room_hub", t=0.0)
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
  its kind's generic figure (0.93 Wh for a carry) instead of the world's
  measured one (room_hub, 0.817), so after every restart a pack charged to
  88 % could not take a job it had been offered."""
  path = tmp_path / "tasks.json"
  b = task_board(str(path), world="room_hub")
  task = b.offer("fetch_module", "module_lcd", t=0.0)
  back = task_board(str(path), world="room_hub")
  assert back[task.id].estimate_wh == task.estimate_wh == 0.817
  assert back[task.id].claimable(10.0, pack_wh=0.879)


def test_a_world_that_carries_on_keeps_its_deadlines_on_its_own_clock(tmp_path):
  """The clock goes on (issue #345), so an offer's deadline means what it
  said; a world built from its XML starts at 0 and gets what was left."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  task = b.offer("draw_figure", "whiteboard_a", params={"program": "house"},
                 ttl=720.0, t=3125.0)
  assert TaskBoard(path, rebase=False)[task.id].deadline == 3845.0
  assert TaskBoard(path)[task.id].deadline == 720.0


# ---- History ---------------------------------------------------------------


def test_history_says_restarted_where_the_process_did_and_nothing_where_it_did_not(tmp_path):
  """"Woke up" / "finished the day" were written at every hourly restart,
  and the robots planned round a day that had not ended. A run that keeps
  its world says nothing at its end; the next says it restarted; a run
  with no saved world says it woke up, as it always did."""
  life = _life(tmp_path)
  life.continuing = True
  life.run(world_config("room_hub")["start"], max_sim_time=0.0)
  lines = _history(life)
  assert any("woke up in room_hub" in ln for ln in lines)
  assert not any("finished the day" in ln or "day ended" in ln for ln in lines)
  snap = _saved(life, tmp_path)

  back = _life(tmp_path)
  _prelude(back, snap)
  lines = _history(back)
  assert any("the world restarted; I carried on from" in ln for ln in lines)
  assert not any("woke up" in ln for ln in lines)

  plain = _life(tmp_path / "plain")
  plain.run(world_config("room_hub")["start"], max_sim_time=0.0)
  assert any("finished the day" in ln for ln in _history(plain))


def test_a_resumed_run_counts_its_budget_from_where_it_starts(tmp_path):
  """`max_sim_time` is a RUN's, now that the clock goes on: a world saved at
  t=4000 and given 600 s runs to 4600, where the old absolute reading would
  have ended it before its first pass -- for a robot the save never had as
  well, which starts from its start pose on the same clock."""
  life = _life(tmp_path)
  life.data.time = 4000.0
  snap = _saved(life, tmp_path)
  snap.meta["robots"] = {}                         # nobody it knows
  back = _life(tmp_path)
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
  life = _life(tmp_path)
  cfg = world_config("room_hub")
  life.mission.start_at(*cfg["start"])
  life.mission._drive(1.0, 0.2, 0.0)
  life.battery.energy_wh = 0.42
  snap = _saved(life, tmp_path)
  snap.meta["fingerprint"] = "another-world"

  back = _life(tmp_path)
  day = _restored(back, snap)
  assert back.battery.energy_wh == 0.42
  assert back.data.time == life.data.time
  assert not back.resumed["inPlace"]
  assert not np.array_equal(back.data.qpos, life.data.qpos)
  _prelude(back, day=day)
  assert back.data.time > life.data.time           # from the start pose: spun
  assert any("could not put me back where I was (the world itself changed"
             in ln for ln in _history(back))


def test_a_saved_world_that_keeps_crashing_is_left_after_three_tries(tmp_path):
  """A state that kills the process would otherwise be put back into the
  same death for ever. `load` counts itself before it is trusted and a
  save puts the count back to 0; three loads with no save in between and
  the next start is a fresh one, said why."""
  life = _life(tmp_path)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  for _ in range(continuation.MAX_RESUMES):
    assert continuation.load(path, "room_hub").snapshot is not None
  refused = continuation.load(path, "room_hub")
  assert refused.snapshot is None and "never got past it" in refused.why
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  assert continuation.load(path, "room_hub").snapshot is not None
  assert continuation.load(path, "home").snapshot is None


# ---- the keeper -------------------------------------------------------------


def test_the_keeper_saves_on_the_seam_and_a_signal_stops_at_a_step(tmp_path):
  """Saved every `every_s` of sim time on the first robot's hooks; and a
  stop only ASKS -- a SIGTERM lands between any two bytecodes, a ledger
  write or a death half done, so the next step boundary raises instead.
  A robot mid stand-up is half stood up, and neither happens then."""
  life = _life(tmp_path)
  path = tmp_path / "world.npz"
  keeper = continuation.Keeper([life], path, every_s=0.1)
  assert life.continuing
  life.mission._drive(0.25, 0.0, 0.0)
  assert keeper.saves == 2 and path.exists()
  meta = json.loads(str(np.load(path)["meta"]))
  assert meta["resumes"] == 0 and meta["world"] == "room_hub"

  life._standing_up = True
  keeper.request_stop("SIGTERM")
  life.mission._drive(0.01, 0.0, 0.0)             # deferred, not raised
  life._standing_up = False
  with pytest.raises(MissionAborted, match="SIGTERM"):
    life.mission._drive(0.01, 0.0, 0.0)


def test_the_pair_carries_on_both_robots_from_one_saved_world(tmp_path):
  """The deployed shape (issue #181): one world, two robots, ONE save --
  both packs, both poses, both maps, keyed by each robot's root -- and the
  world's own activities with them (the pair's encounter count here)."""
  from pluggybot.pair import build_pair
  cfg = world_config("room_hub")
  starts = (cfg["start"], cfg["start2"])
  lives = build_pair("room_hub")
  for life, start in zip(lives, starts):
    life.mission.start_at(*start)
  lives[1].mission._drive(0.8, 0.2, 0.3)
  lives[0].battery.energy_wh, lives[1].battery.energy_wh = 0.3, 0.6
  lives[0].encounters.count = 4
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)

  back = build_pair("room_hub", resume=snap)
  for life, start in zip(back, starts):
    life.begin(start)
  got = continuation.restore(back, snap)
  assert got["inPlace"] and got["robots"] == ["pluggybot", "r2_pluggybot"]
  assert np.array_equal(back[0].data.qpos, lives[0].data.qpos)
  for was, now in zip(lives, back):
    assert now.battery.energy_wh == was.battery.energy_wh
    assert now.mission.pose == was.mission.pose
    assert np.array_equal(now.mission.grid.grid, was.mission.grid.grid)
  assert back[0].encounters.count == 4


def test_a_world_saved_mid_charge_counts_travel_again_and_says_so(tmp_path):
  """`pinned` is the charge routine's, set for the press and cleared in its
  `finally` -- and a restart ends that routine. Restored with the rest, the
  reckoner would never count travel again; the robot is told the charge
  was cut short instead."""
  life = _life(tmp_path)
  life.mission.start_at(*world_config("room_hub")["start"])
  life.mission.swap.pinned = True
  life.state = "CHARGE"
  snap = _saved(life, tmp_path)
  back = _life(tmp_path)
  _prelude(back, snap)
  assert back.mission.swap.pinned is False
  assert any("it cut my charge short" in ln for ln in _history(back))


def test_the_pair_steps_on_exactly_after_a_restart(tmp_path):
  """The parity rule on the deployed shape: two robots from one loop, the
  depth cameras on, the pair's encounters sensing -- the same drive after
  a restore steps the same world and paints the same two maps."""
  from pluggybot.pair import build_pair
  cfg = world_config("room_hub")
  starts = (cfg["start"], cfg["start2"])

  def fly(lives):
    tick.run_many([(life.mission.swap, life.mission._drive_routine(0.5, 0.15, w))
                   for life, w in zip(lives, (-0.3, 0.4))])

  lives = build_pair("room_hub", near_field=True)
  for life, start in zip(lives, starts):
    life.mission.start_at(*start)
  fly(lives)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture(lives, lives[0].world_fingerprint), path)
  snap = continuation.read(path)
  fly(lives)

  back = build_pair("room_hub", near_field=True, resume=snap)
  for life, start in zip(back, starts):
    life.begin(start)
  continuation.restore(back, snap)
  fly(back)
  assert np.array_equal(back[0].data.qpos, lives[0].data.qpos)
  for was, now in zip(lives, back):
    assert now.mission.pose == was.mission.pose
    assert np.array_equal(now.mission.grid.grid, was.mission.grid.grid)
    assert np.array_equal(now.near_field.height, was.near_field.height,
                          equal_nan=True)


# ---- what the review found -----------------------------------------------------


def test_a_pen_stowed_after_a_restart_has_its_carriage_centred_first(tmp_path):
  """A restart mid-drawing leaves the carriage where the last stroke did,
  and off-centre it jams on the bay's bracket feet: MEASURED in review, a
  pen stowed from 37 mm never hung and stayed on the fork for good. The
  carry configuration every stow goes through centres it, as the drawing
  errand's own does."""
  from pluggybot.procedure import steps
  life = _life(tmp_path)
  life.mission.start_at(*world_config("room_hub")["start"])
  act = life.model.actuator("pen_carriage")
  qadr = int(life.model.joint("pen_carriage_joint").qposadr[0])
  life.data.ctrl[act.id] = 0.037
  life.mission._drive(1.0, 0.0, 0.0)
  assert life.data.qpos[qadr] > 0.03
  life.mission.run(steps.carry_configuration_routine(life, "module_pen"))
  assert abs(life.data.qpos[qadr]) < 0.002


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04 torn", None])
def test_a_save_that_cannot_be_read_is_a_fresh_start_never_a_crash(tmp_path, content):
  """Raised out of `load`, an empty or torn file killed the process before
  `MAX_RESUMES` could count, and `restart: unless-stopped` looped on it
  (found in review: EOFError, BadZipFile)."""
  life = _life(tmp_path)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  data = path.read_bytes()
  path.write_bytes(content if content is not None else data[: len(data) // 2])
  got = continuation.load(path, "room_hub")
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
  life = _life(tmp_path)
  snap = _saved(life, tmp_path)
  monkeypatch.setattr(steps, "_carried", lambda life: "module_lcd")
  monkeypatch.setattr(steps, "_stow", lambda life, args: tick.result({"ok": True}))
  back = _life(tmp_path)
  _prelude(back, snap)
  assert any("the restart stopped me mid-swap at module_lcd's bay; I backed "
             "out and it hangs there" in ln for ln in _history(back))


def test_a_map_that_does_not_fit_this_build_is_explored_again(tmp_path):
  """The fingerprint is the geometry's; a new grid extent or resolution is
  not. The grid is then left empty -- and the map's verdicts with it, or a
  robot that believed its map complete would never explore the empty one
  it has (found in review). Where it IS stays put."""
  life = _life(tmp_path)
  life.mission.start_at(*world_config("room_hub")["start"])
  life.map_done = True
  life.blacklist = {(1, 2)}
  snap = _saved(life, tmp_path)
  snap.arrays["pluggybot/grid"] = np.zeros((4, 4))
  back = _life(tmp_path)
  _prelude(back, snap)
  assert back.resumed["inPlace"] and back.mission.pose == life.mission.pose
  assert not back.map_done and back.blacklist == set()


def test_a_module_the_world_no_longer_has_is_not_restored_as_the_one_watched(tmp_path):
  """`module` names what the power model watches; a retired built tool's
  name restored there fails the next `module_state` read (found in
  review)."""
  life = _life(tmp_path)
  snap = _saved(life, tmp_path)
  snap.meta["robots"]["pluggybot"]["module"] = "module_retired_long_ago"
  back = _life(tmp_path)
  _prelude(back, snap)
  assert back.module == "module_lcd"
  back.mission.swap.module_state(back.module)


def test_what_begin_says_is_stamped_on_the_restored_clock(tmp_path):
  """`begin` announces what a restart failed (a game) before `restore`;
  with the clock set first, the line and its event carry the world's time,
  not 0 -- rows on the observatory in the order they happened."""
  path = tmp_path / "tasks.json"
  b = TaskBoard(path)
  game = b.offer("hide_and_seek", "room_hub", t=0.0)
  b.claim(game.id, robot="pluggybot", t=1.0, role="hider")
  b.claim(game.id, robot="r2_pluggybot", t=1.0, role="seeker")
  b.start(game.id, t=2.0)
  life = _life(tmp_path / "saved")
  life.data.time = 500.0
  snap = _saved(life, tmp_path)
  back = _life(tmp_path, tasks=True, rebase=False)
  heard = []
  back.say_hooks.append(lambda t, line: heard.append((t, line)))
  back.run(world_config("room_hub")["start"], max_sim_time=0.0, resume=snap)
  [(t, _)] = [h for h in heard if "interrupted by a restart" in h[1]]
  assert t == 500.0


def test_an_offered_challenge_finds_its_props_where_the_offer_says(tmp_path):
  """The offer says where the blocks stand, and the hourly reset was what
  made that true: with the world carried on, a failed attempt would leave
  them wherever it dropped them, for good. The house sets them out as it
  offers the job (issue #345)."""
  from pluggybot.challenge import stack
  life = _life(tmp_path, world="home", tasks=True)
  qadr = int(life.model.joint(int(life.model.body("block_1").jntadr[0])).qposadr[0])
  home = life.model.qpos0[qadr:qadr + 3]
  life.data.qpos[qadr:qadr + 3] = home + [0.6, -0.4, 0.0]  # knocked aside
  # ...and one against a robot's chassis, which is somebody's mid-job
  held = int(life.model.joint(int(life.model.body("block_2").jntadr[0])).qposadr[0])
  mujoco.mj_forward(life.model, life.data)            # the chassis box's centre
  life.data.qpos[held:held + 3] = life.data.geom_xpos[life.mission.chassis_gid]
  mujoco.mj_forward(life.model, life.data)
  against = life.data.qpos[held:held + 3].copy()
  block = life.model.body("block_2").id
  g = life.data.contact.geom[:life.data.ncon]
  assert (life.model.geom_bodyid[g] == block).any()   # it IS touching
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
  lives = build_pair("room_hub")
  keeper = continuation.Keeper(lives, tmp_path / "world.npz")
  assert keeper.step_hook in lives[-1].mission.step_hooks
  assert keeper.step_hook not in lives[0].mission.step_hooks
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
  from pluggybot.mission.errand import carry_errand
  life = _life(tmp_path)
  life._errand_now = carry_errand("module_lcd")
  snap = _saved(life, tmp_path)
  back = _life(tmp_path)
  _prelude(back, snap)
  assert back.errands == []
  assert any(f"the restart cut short {life._errand_now.name}; it is not "
             "queued again" in ln for ln in _history(back))
  back.begin(world_config("room_hub")["start"])
  assert back.resumed is None
