"""Two robots, one world, one physics loop (issue #167, M12 slice B).

Each robot is a `HubLifecycle` of its own -- its own body, pack, odometry
and reserve, its own errand queue, its own day -- built against the same
`MjModel` and `MjData` through its `RobotHandle`. What is shared is the
WORLD: the model, the rack and its bays, the tools, the dock, the
whiteboards' book, the activities. The two day-routines are ticked in turn from
`tick.run_many`: every step, both robots' commands, one `mj_step`, both
robots' bookkeeping. No robot ever blocks the other.

Mutual awareness, the honest half: each mission is handed the other's
REPORTED pose (`Body.others`) -- what a robot may know of another over
the network -- so A* keeps clear of where it is now and a blocked drive waits
for it to move (or, while it lies on the floor, of where its BODY is: issue
#365, `HubLifecycle.keep_clear`); and each lidar DROPS the other's body from
its scans (`Lidar.exclude_robot`), the way a fleet subtracts a broadcast
footprint, because a robot that drives past otherwise paints a wake of
inflated obstacle into the map and walls in the robot it passed (measured,
and the reason the first attempt at a contested bay ended in "no route").
What is NOT here: the other robot as a mind -- that is the minds' slice (C)
and the context they are shown.

The rack is one rack and the dock one dock, contended. Contention is the
minds' to negotiate and this module arbitrates nothing.
"""

from typing import Callable

import os
from pathlib import Path

import mujoco

from pluggybot import tick
from pluggybot.economy.cadence import default_cadence
from pluggybot.lifecycle import (
  GAME_TARGET, QUAD_HOME, HubLifecycle, board_book, errand_for_task, errands_for,
  points_ledger, task_board, task_producer, world_config,
)
from pluggybot.mind import constitution as constitutions
from pluggybot.mind import events as ev
from pluggybot.robot import FIRST, SECOND, pair_model_name, world_spec
from pluggybot.tick import MissionAborted


#: The second robot's default display name; the first keeps `Pluggy`.
SECOND_NAME_ENV = "PLUGGY_ROBOT_NAME_2"
DEFAULT_SECOND_NAME = "Rowan"
#: ...and which constitution each reads (issue #263), on the name's terms:
#: the first robot's is `$PLUGGY_CONSTITUTION`, the second's its own
#: variable, so a pair can be given two dispositions in one world.
CONSTITUTION_ENVS = (constitutions.NAME_ENV, constitutions.SECOND_NAME_ENV)


def build_pair(world: str = QUAD_HOME, pack: str = "demo",
               errands=("none", "none"), board_state: str | None = None,
               view: bool = False, realtime: bool = False,
               handles: tuple = (FIRST, SECOND), names: tuple | None = None,
               overseer: bool | None = None, autonomous: bool = False,
               origin: str = ev.DEFAULT_ORIGIN, standing_orders: bool = False,
               thoughts_root: str | None = None, ledger_state: str | None = None,
               tasks: bool = False, metabolism: bool = False,
               mortal: bool | None = None, task_state: str | None = None,
               inboxes: tuple | None = None, mode=None,
               overseer_kw: dict | None = None, battery_wh: float | None = None,
               reserve_wh: float | None = None,
               constitutions_named: tuple | None = None,
               resume=None, **life_kw) -> list:
  """One world, two lifecycles -- and, with `overseer`, TWO MINDS.

  Two of everything a robot owns, one of everything the world does:

    per robot   a mission, a battery and a reserve, an errand queue, a
                thought-file root (the first robot's is `thoughts_root`
                itself, so an existing volume stays the first robot's; the
                second's is `<root>/<r2 root body>/`), a procedure library
                under it, a record store, a WALLET and an appetite, an overseer
                with its own event map and standing order
    the world   the model, the rack and its bays, the modules, the
                whiteboards' book, the activities, and ONE task board with
                one producer (ticked by the first robot's seam): an offer is
                the house's, whoever takes it, and a claim by one robot is
                the offer gone for the other.

  SEPARATE WALLETS, decided here: two ledgers, two balances, two upkeeps,
  two sets of hearts. A shared wallet would be a cooperation lever -- one
  robot's work paying the other's rent -- and is worth flying later as an
  ablation; separate is the cleaner measurement, because with it "did it
  help the other" cannot be confused with "did it help itself".

  Each mind is told the other's NAME in its prefix (`OTHER_ROBOT_RULE`) and
  what the other broadcasts in its context (`lifecycle.others_context`).

  The served world's knobs (issue #181): `task_state` persists the shared
  board; `inboxes` is one `Inbox` PER ROBOT (a reach-in is addressed to a
  robot, so each drains its own); `mode` is the operator's switch and goes
  to the FIRST robot only -- pausing blocks inside its step hook, and one
  loop steps both, so one pause stops the world; `overseer_kw` reaches
  every mind's `overseer.build` (backend, model, spend book...);
  `battery_wh` / `reserve_wh` override the pack's figures. `resume` is the
  saved world the pair will carry on from (issue #345; `run_pair` puts it
  back), which keeps the board's deadlines on the clock it goes on with.

  ⚠ THE DOCUMENTS AND THE STORE ARE PER ROBOT. The first robot's live at
  the thoughts root, as a single robot's did; the second's under ITS root
  (`<thoughts root>/<r2 root>/`, its own `memory.sqlite`) -- measured: one
  root for both handed two minds one memory.
  """
  from pluggybot.mind import overseer as ov
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.economy.metabolism import Appetite, Metabolism
  from pluggybot.telemetry.protocol import robot_display_name
  cfg = world_config(world)
  starts = (cfg["start"], cfg["start2"])
  # THE SPEC IS KEPT, and both lifecycles get it (issue #315): a pair is
  # compiled from a spec either way -- `world_with_robots` throws it away
  # after `compile()` -- and without it `can_reshape` refuses every build
  # with "this world was compiled without its spec", which is what the
  # deployed pair was hitting before it ever reached the pair rule below.
  spec = world_spec(cfg["model"], starts[1][:2], prefix=handles[1].prefix,
                    body=cfg["body"])
  model = spec.compile()
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  book = board_book(world, state=board_state)
  default_wh = cfg["battery_wh"] if pack == "demo" else cfg["hosting_battery_wh"]
  names = names or (robot_display_name(None),
                    robot_display_name(os.environ.get(SECOND_NAME_ENV)
                                       or DEFAULT_SECOND_NAME))
  # The world's task board, once, and its producer on the FIRST robot only.
  tasks = tasks or task_state is not None
  beat = default_cadence(world) if tasks else None
  board = (task_board(task_state, cadence=beat, world=world, rebase=resume is None)
           if tasks else None)
  # ...told both names, so a job done TO a robot (issue #228) can name
  # one; the target exists on the `autonomous` arm alone.
  maker = (task_producer(board, world, book, beat, procedures=autonomous,
                         robots=names)
           if board is not None else None)
  appetite = Appetite.load(world) if metabolism else None
  # ONE ledger file, one ACCOUNT per robot (issue #167 slice E): separate
  # wallets, one state, and every entry on the wire names its robot. Each
  # lifecycle holds its own `Account` view, so the calls it always made
  # address its own account.
  from pluggybot.economy.ledger import Account
  book_of_points = points_ledger(ledger_state, cap=appetite.cap if appetite else None,
                                 robots=tuple(h.root for h in handles))
  lives = []
  from pluggybot.mind.thoughts import ROOT_ENV
  thoughts_root = thoughts_root or os.environ.get(ROOT_ENV, "").strip() or None
  # Which constitution each robot reads (issue #263): named here, else
  # each robot's own environment variable, else the library's default.
  charters = tuple(constitutions.for_body(constitutions.resolve(
    (constitutions_named or (None, None))[i], env=CONSTITUTION_ENVS[i]),
    cfg["body"]) for i in range(len(handles)))
  for i, (handle, errand, name) in enumerate(zip(handles, errands, names)):
    if i == 0:
      memory = ThoughtFiles.open(thoughts_root, robot=handle.root,
                                 constitution=charters[i])
    else:
      root = None if thoughts_root is None else Path(thoughts_root) / handle.root
      memory = ThoughtFiles(str(root) if root is not None else None,
                            robot=handle.root, constitution=charters[i])
    ledger = Account(book_of_points, handle.root)
    hunger = (Metabolism(book_of_points, appetite, robot=handle.root)
              if appetite else None)
    boss = ov.build(world, book, enabled=overseer, thoughts=memory,
                    robot_name=name, ledger=ledger,
                             appetite=hunger is not None, mortal=bool(mortal),
                             hearts=bool(mortal) and ledger is not None,
                             standing_orders=standing_orders, origin=origin,
                             autonomous=autonomous,
                             others=tuple(n for n in names if n != name),
                             **(overseer_kw or {}))
    life = HubLifecycle(model, data, viewer=viewer if i == 0 else None,
                        realtime=realtime, battery_wh=battery_wh or default_wh,
                        rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                        low_battery_wh=(reserve_wh if reserve_wh is not None
                                        else cfg["low_battery_wh"]),
                        boards=book,
                        inbox=inboxes[i] if inboxes else None,
                        mode=mode if i == 0 else None,
                        world=world, errands=errands_for(errand, world, book),
                        handle=handle, robot_name=name, ledger=ledger,
                        overseer=boss, thoughts=memory, spec=spec,
                        metabolism=hunger, tasks=board,
                        producer=maker if i == 0 else None,
                        mortal=mortal, autonomous=autonomous, **life_kw)
    # The board grows for BOTH robots, though only the first ticks the
    # producer: the second stands by for work like the first does.
    life.expects_work = maker is not None
    lives.append(life)
  # ONE RACK FOR THE PAIR (issue #315). The rack, its bays and the modules
  # on it are the WORLD's and never prefixed -- tool contention is the
  # minds' to negotiate -- so the inventory the workshop edits is ONE dict
  # both lifecycles hold: a tool the second robot builds is a tool the
  # first can see, fetch and stand clear of. Two copies would diverge the
  # moment either robot hung anything, and `can_reshape` reads the other
  # robot's fork off it.
  for life in lives[1:]:
    life.rack_inventory = lives[0].rack_inventory
    # ...and the lost-tool clock is the world's (issue #347): one hand, on
    # the first robot's seam, or a tool would be put back twice
    life.lost_tool_after_s = None
  # Each mission is told where the OTHERS say they are (where they lie, once
  # knocked over), its lidar drops their bodies from the scan (see the module
  # doc and `Lidar.exclude_robot`), and its mind is shown what they broadcast
  # (`peers`). And a drive can ask one resting across its way to make way
  # (issue #415): a message between the two, as a reported pose is.
  by_root = {life.root: life for life in lives}
  for life in lives:
    others = [other for other in lives if other is not life]
    life.peers = others
    life.body.others = [
      (lambda other=other, me=life.body: other.keep_clear(seen_by=me))
      for other in others]
    life.body.ask_way = (lambda root, route, me=life.root:
                         root in by_root and by_root[root].make_way(route, me))
    for other in others:
      life.body.know_peer(other.root)
      if life.depth_camera is not None:
        life.depth_camera.exclude_robot(other.root)
  # The world's activities sense once per step, on the first robot's hooks:
  # they are the world's, and two copies would sense everything twice. The
  # pair's own -- the ENCOUNTERS between the two (activity/encounter.py) --
  # joins them, and its events reach whatever records the pair.
  from pluggybot.activity.base import ActivitySet
  from pluggybot.activity.encounter import Encounters
  activities = cfg["activities"](model, data) if cfg["activities"] else ActivitySet()
  meetings = Encounters(model, lives[0].body.handle, lives[1].body.handle,
                        lives=lives)
  activities.add(meetings)
  lives[0].body.step_hooks.append(activities.step_hook(model, data))
  for life in lives:
    life.activities = activities
    life.encounters = meetings
  # ...and A GAME FOR TWO (issue #404): where the board can offer hide and
  # seek -- its target is named on `autonomous` with two robots alone --
  # the pair's referee is in the world from the start
  if maker is not None and "hide_and_seek" in maker.kinds:
    referee_games(lives)
  return lives


def referee_games(lives: list):
  """The pair's games of hide and seek, refereed (issues #167, #404).

  ONE referee for the world (`activity/hideseek.py`), built idle in the
  world's activities -- the stream's header lists activities once -- and
  kept, game after game: each game on the pair's board, offered by the
  cadence or by `arrange_game`, is given it as its LAST role is claimed
  (whichever robot claims first holds the first open role, the hider), and
  each robot is then queued its role's errand (`errand_for_task(role=)`),
  nothing before. It senses on the first robot's seam, and when it calls a
  game, ONE verdict is evaluated off its flags and banked on the WINNER's
  wallet -- nobody's, when the game was called off -- and the task resolves
  with it for both. A pair already refereed keeps its referee, which is
  returned."""
  from pluggybot.activity.hideseek import HideAndSeek
  from pluggybot.economy import scoring
  first = lives[0]
  if first.game is not None:
    return first.game
  board = first.tasks
  if board is None:
    raise ValueError("a game needs the pair's task board (tasks=True)")
  by_root = {life.root: life for life in lives}
  game = HideAndSeek(first.model)
  if first.activities is not None:
    first.activities.add(game)
  else:
    first.body.step_hooks.append(lambda: game.sense(first.model, first.data))
  for life in lives:
    life.game = game

  def settle(game) -> None:
    task = board.get(game.task_id)
    if task is None or not task.open:
      return
    verdict = scoring.evaluate("hide_and_seek", game.measurements())
    winner = by_root.get(task.claims.get(verdict.metrics.get("winner") or "", ""))
    for life in lives:
      if verdict.ok and life is winner:
        life._bank(verdict)
      else:
        # ...and the one not paid reads it too: its game happened
        said = (f"hide_and_seek: {verdict.reason} -- nothing for the "
                f"{life.role_in(task.id) or 'other'}")
        life._say(f"GAME {said}")
        life._remember(said)
    board.resolve(task.id, verdict, t=float(first.data.time))
  game.on_over.append(settle)

  def on_claim(event: dict) -> None:
    if event.get("type") != "task_claimed":
      return
    task = board.get(event.get("id") or "")
    if (task is None or task.kind != "hide_and_seek" or task.state != "claimed"
        or set(task.claims) != {"hider", "seeker"}
        or not set(task.claims.values()) <= set(by_root)):
      return
    hider, seeker = by_root[task.claims["hider"]], by_root[task.claims["seeker"]]
    game.assign(hider=hider.body.handle, seeker=seeker.body.handle,
                task_id=task.id, t=float(first.data.time))
    for role, life in (("hider", hider), ("seeker", seeker)):
      errand = errand_for_task(task, life.world, role=role)
      if errand is not None:
        life.errands.append(errand)
    for life in lives:
      life._say(f"GAME hide_and_seek: {hider.robot_name} hides, "
                f"{seeker.robot_name} seeks")
  board.on_event.append(on_claim)
  return game


def arrange_game(lives: list, kind: str = "hide_and_seek", t: float = 0.0):
  """Put a game of hide and seek on the pair's board now (issue #167): the
  offer the cadence makes on `autonomous`, made by hand -- for a test, a
  demo or a recording -- and refereed by the pair's referee
  (`referee_games`). Returns the task, and a state whose `game` is the
  referee once THIS game's roles are taken."""
  game = referee_games(lives)
  board = lives[0].tasks
  task = board.offer(kind, GAME_TARGET, t=t)
  if task is None:
    raise ValueError(f"the board would not offer {kind}")
  state: dict = {"game": None}

  def on_claim(event: dict) -> None:
    if event.get("id") == task.id and game.task_id == task.id:
      state["game"] = game
  board.on_event.append(on_claim)
  return task, state


def arrange_hazards(lives: list, fall_at: float, drain_at: float,
                    who: int = 1) -> dict:
  """The two things a scripted day never does by itself, done to ONE robot
  of a quadruped pair on the clock (issue #387): it is knocked onto its side
  at the first step past `fall_at` where it stands, and its pack is emptied
  at `drain_at`. The body rights itself, and the loop kills it `flat` and
  stands it up `restart_after_s` later -- so a day arranged this way flies
  every posture and both deaths' halves the deployed pair can meet, and
  hashes like any other day: nothing here reads a clock but the sim's.
  Returns what it did, and when, for the caller to assert on."""
  import math
  from pluggybot.legs import body as qb
  life = lives[who]
  model, data = life.model, life.data
  q = life.body.handle.qpos_adr(model)
  v = life.body.handle.dof_adr(model)
  done: dict = {"fell": None, "drained": None}

  def hook() -> None:
    t = float(data.time)
    if (done["fell"] is None and t >= fall_at and life.dead is None
        and life.body.posture == qb.STANDING):
      data.qpos[q + 2] = 0.25                                    # on its side
      data.qpos[q + 3:q + 7] = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
      data.qvel[v:v + 6] = 0.0
      mujoco.mj_forward(model, data)
      done["fell"] = t
    if done["drained"] is None and t >= drain_at:
      life.battery.energy_wh = 0.0
      done["drained"] = t
  life.body.step_hooks.append(hook)
  return done


def record_pair(lives: list, path: str):
  """One recording of both robots (issue #167; protocol 0.20.0): the first
  robot's stream as it always was, the second under `robots[<r2 root>]`
  beside it with its own `metabolism`, the header naming both, one `goals`
  and one set of `thought` documents and one `grid` per robot, and every
  event keyed by the robot that emitted it -- two ledgers' accounts, the
  pair's encounters and the game's referee. The header's `model` is the
  PAIR world's name (`pair_model_name`), because a replayer picks its scene
  off it and the second robot's bodies are in no single-robot scene."""
  from pluggybot.mind import overseer as ov
  from pluggybot.telemetry.recorder import StreamRobot, TelemetryRecorder
  first, others = lives[0], lives[1:]
  cfg = world_config(first.world)
  recorder = TelemetryRecorder(
    first.model, first.data, path, model_name=pair_model_name(cfg["model_name"]),
    status_fn=first.telemetry_status, activities=first.activities,
    boards=first.boards, ledger=(first.ledger._ledger
                                 if hasattr(first.ledger, "_ledger") else first.ledger),
    tasks=first.tasks, thoughts=first.thoughts, metabolism=first.metabolism,
    grid=first.body.grid, robot_name=first.robot_name,
    heightmap=first.near_field,
    goals=ov.goals_text(thoughts=first.thoughts),
    steering=first.overseer is not None, overseer=first.overseer,
    tickets=first.tickets,
    others=[StreamRobot(o.root, o.robot_name, o.telemetry_status,
                        metabolism=o.metabolism, thoughts=o.thoughts,
                        goals=ov.goals_text(thoughts=o.thoughts),
                        steering=o.overseer is not None, grid=o.body.grid,
                        heightmap=o.near_field, overseer=o.overseer,
                        tickets=o.tickets)
            for o in others])
  first.body.step_hooks.append(recorder.step_hook)
  # ...and a recompiled world reaches its census (issues #168, #315). On
  # the FIRST robot alone, as its step hook is: `_recompile` rebinds every
  # lifecycle in the world, so a tool the second robot builds fires this
  # through the first robot's own rebind.
  first.on_rebind.append(recorder.rebind)
  if first.boards is not None:
    first.boards.on_event.append(recorder.emit)
  if first.ledger is not None:
    first.ledger.on_event.append(recorder.emit)
  if first.tasks is not None:
    first.tasks.on_event.append(recorder.emit)
  for life in lives:
    life.on_event.append(recorder.emit)
    life.thoughts.on_event.append(recorder.emit)
  first.encounters.on_event.append(recorder.emit)
  return recorder


def run_pair(lives: list, starts=None, max_sim_time: float = 600.0,
             explore_budget: float | None = None,
             stop_when: Callable | None = None,
             record: str | None = None, resume=None) -> list[dict]:
  """Both days from one loop; each robot's summary in order. `resume` is a
  saved world (issue #345), put back once both robots' `begin` has hung
  what they built and before either day moves."""
  cfg = world_config(lives[0].world)
  starts = starts or (cfg["start"], cfg["start2"])
  budget = explore_budget if explore_budget is not None else cfg["explore_budget"]
  if stop_when is not None:
    lives[0].stop_when(lambda: stop_when(lives))
  recorder = record_pair(lives, record) if record is not None else None
  if resume is not None:
    lives[0].data.time = resume.t        # what `begin` says, on the clock
  days = [life.begin(start, max_sim_time=max_sim_time, explore_budget=budget)
          for life, start in zip(lives, starts)]
  if resume is not None:
    from pluggybot import continuation
    continuation.restore(lives, resume)
  aborted = False
  try:
    tick.run_many([(life.body.stepper, day) for life, day in zip(lives, days)],
                  name="pair")
  except MissionAborted:
    aborted = True
  finally:
    for life in lives:
      life.body.close()
    if recorder is not None:
      recorder.close()
  return [life.end(aborted) for life in lives]


def run_demo_pair(world: str = QUAD_HOME, max_sim_time: float = 300.0,
                  view: bool = False, realtime: bool = True,
                  pack: str = "demo", errands=("none", "none"),
                  board_state: str | None = None, on_ready=None,
                  record: str | None = None, game: bool = False,
                  **kw) -> list[dict]:
  lives = build_pair(world, pack=pack, errands=errands, board_state=board_state,
                     view=view, realtime=realtime, tasks=kw.pop("tasks", False) or game,
                     **kw)
  if game:
    arrange_game(lives)
  if on_ready is not None:
    on_ready(lives)
  return run_pair(lives, max_sim_time=max_sim_time, record=record)



