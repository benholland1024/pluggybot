"""Two robots, one world, one physics loop (issue #167, M12 slice B).

Each robot is a `HubLifecycle` of its own -- its own mission, swap, battery,
odometry and reserve, its own errand queue, its own day -- built against
the same `MjModel` and `MjData` through its `RobotHandle`. What is shared is
the WORLD: the model, the rack and its bays, the modules, the whiteboards'
book, the activities. The two day-routines are ticked in turn from
`tick.run_many`: every step, both robots' commands, one `mj_step`, both
robots' bookkeeping. No robot ever blocks the other.

Mutual awareness, the honest half: each mission is handed the other's
REPORTED pose (`HubMission.others`) -- what a robot may know of another over
the network -- so A* keeps clear of where it is now and a blocked drive waits
for it to move; and each lidar DROPS the other's body from its scans
(`Lidar.exclude_robot`), the way a fleet subtracts a broadcast footprint,
because a robot that drives past otherwise paints a wake of inflated
obstacle into the map and walls in the robot it passed (measured, and the
reason the first attempt at a contested bay ended in "no route"). What is
NOT here: the other robot as a mind -- that is the minds' slice (C) and the
context they are shown.

The rack is one rack: one charge bay, five tool bays, contended. Tool
contention is the minds' to negotiate and this module arbitrates nothing; a
second charge bay is a generator parameter for the slice that flies both
robots' full days.
"""

from typing import Callable

import os
from pathlib import Path

import mujoco

from pluggybot import tick
from pluggybot.economy.cadence import default_cadence
from pluggybot.lifecycle import (
  HubLifecycle, board_book, errands_for, points_ledger, task_board,
  task_producer, world_config,
)
from pluggybot.mind import events as ev
from pluggybot.mission.mission import MissionAborted
from pluggybot.robot import FIRST, SECOND, pair_model_name, world_with_robots


#: The second robot's default display name; the first keeps `Pluggy`.
SECOND_NAME_ENV = "PLUGGY_ROBOT_NAME_2"
DEFAULT_SECOND_NAME = "Rowan"


def build_pair(world: str = "room_hub", pack: str = "demo",
               errands=("carry", "none"), board_state: str | None = None,
               view: bool = False, realtime: bool = False,
               handles: tuple = (FIRST, SECOND), names: tuple | None = None,
               overseer: bool | None = None, autonomous: bool = False,
               origin: str = ev.DEFAULT_ORIGIN, standing_orders: bool = False,
               thoughts_root: str | None = None, ledger_state: str | None = None,
               tasks: bool = False, metabolism: bool = False,
               mortal: bool | None = None, task_state: str | None = None,
               inboxes: tuple | None = None, mode=None,
               overseer_kw: dict | None = None, battery_wh: float | None = None,
               reserve_wh: float | None = None, goals_path: str | None = None,
               journal_path: str | None = None, **life_kw) -> list:
  """One world, two lifecycles -- and, with `overseer`, TWO MINDS.

  Two of everything a robot owns, one of everything the world does:

    per robot   a mission, a battery and a reserve, an errand queue, a
                thought-file root (the first robot's is `thoughts_root`
                itself, so an existing volume stays the first robot's; the
                second's is `<root>/<r2 root body>/`), a procedure library
                under it, a journal, a WALLET and an appetite, an overseer
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
  `battery_wh` / `reserve_wh` override the pack's figures.

  ⚠ THE DOCUMENTS ARE PER ROBOT, AND ONLY THE FIRST ROBOT'S COME FROM THE
  SINGLE-ROBOT KNOBS. The first robot keeps `goals_path` / `journal_path`
  and their environment fallbacks (`$PLUGGY_GOALS`, `$PLUGGY_JOURNAL`,
  which the image sets), so a volume that served one robot keeps that
  robot's goals and journal where they were. The second robot's live under
  ITS root (`<thoughts root>/<r2 root>/Goals.md`, `.../journal.json`) and
  ignore both flag and environment -- measured: the environment alone
  handed both minds one goals file and one journal.
  """
  from pluggybot.mind import overseer as ov
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.economy.metabolism import Appetite, Metabolism
  from pluggybot.telemetry.protocol import robot_display_name
  cfg = world_config(world)
  starts = (cfg["start"], cfg["start2"])
  model = world_with_robots(cfg["model"], second_at=starts[1][:2],
                            prefix=handles[1].prefix)
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
  board = task_board(task_state, cadence=beat, world=world) if tasks else None
  maker = (task_producer(board, world, book, beat, procedures=autonomous)
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
  for i, (handle, errand, name) in enumerate(zip(handles, errands, names)):
    if i == 0:
      memory = ThoughtFiles.open(thoughts_root, goals_path=goals_path,
                                 robot=handle.root)
      own_journal = journal_path
    else:
      root = None if thoughts_root is None else Path(thoughts_root) / handle.root
      # The constructor, not `open`: `open` falls back to $PLUGGY_GOALS.
      memory = ThoughtFiles(str(root) if root is not None else None,
                            goals_path=None, robot=handle.root)
      own_journal = str(root / "journal.json") if root is not None else None
    ledger = Account(book_of_points, handle.root)
    hunger = (Metabolism(book_of_points, appetite, robot=handle.root)
              if appetite else None)
    boss, journal = ov.build(world, book, enabled=overseer, thoughts=memory,
                             journal_path=own_journal,
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
                        overseer=boss, journal=journal, thoughts=memory,
                        metabolism=hunger, tasks=board,
                        producer=maker if i == 0 else None,
                        mortal=mortal, autonomous=autonomous, **life_kw)
    # The board grows for BOTH robots, though only the first ticks the
    # producer: the second stands by for work like the first does.
    life.expects_work = maker is not None
    lives.append(life)
  # Each mission is told where the OTHERS say they are, its lidar drops
  # their bodies from the scan (see the module doc and `Lidar.exclude_robot`),
  # and its mind is shown what they broadcast (`peers`).
  for life in lives:
    others = [other for other in lives if other is not life]
    life.peers = others
    life.mission.others = [other.mission.pose_xy for other in others]
    for other in others:
      life.mission.lidar.exclude_robot(other.mission.handle.root)
      if life.depth_camera is not None:
        life.depth_camera.exclude_robot(other.mission.handle.root)
  # The world's activities sense once per step, on the first robot's hooks:
  # they are the world's, and two copies would sense everything twice. The
  # pair's own -- the ENCOUNTERS between the two (activity/encounter.py) --
  # joins them, and its events reach whatever records the pair.
  from pluggybot.activity.base import ActivitySet
  from pluggybot.activity.encounter import Encounters
  activities = cfg["activities"](model, data) if cfg["activities"] else ActivitySet()
  meetings = Encounters(model, lives[0].mission.handle, lives[1].mission.handle)
  activities.add(meetings)
  lives[0].mission.step_hooks.append(activities.step_hook(model, data))
  for life in lives:
    life.activities = activities
    life.encounters = meetings
  return lives


def arrange_game(lives: list, kind: str = "hide_and_seek", t: float = 0.0):
  """Put a two-role game on the pair's board (issue #167) and referee it.

  The offer is the house's: whichever robot claims first takes the first
  open role (the hider), the other the second, and the referee
  (`activity/hideseek.py`) is built once both roles are held -- it has to
  know who is who. It senses on the first robot's seam, and when it calls
  the game, ONE verdict is evaluated off its flags and banked on the
  WINNER's wallet; the task resolves with that verdict for both.
  """
  from pluggybot.activity.hideseek import HideAndSeek
  from pluggybot.economy import scoring
  board = lives[0].tasks
  if board is None:
    raise ValueError("a game needs the pair's task board (tasks=True)")
  model, data = lives[0].model, lives[0].data
  task = board.offer(kind, lives[0].world, t=t)
  if task is None:
    raise ValueError(f"the board would not offer {kind}")
  by_root = {life.mission.handle.root: life for life in lives}
  # The referee is in the world's activities from the OFFER (idle, with no
  # roles yet), so the recording's header lists it and its flags ride every
  # frame; the roles are bound at the claim. It senses on the activity set's
  # hook like every other activity.
  game = HideAndSeek(model)
  if lives[0].activities is not None:
    lives[0].activities.add(game)
  for life in lives:
    life.game = game
  state: dict = {"game": None}

  def settle(game) -> None:
    verdict = scoring.evaluate(kind, game.measurements())
    claims = board.get(task.id).claims
    winner = by_root[claims[verdict.metrics["winner"]]]
    for life in lives:
      if life is winner:
        life._bank(verdict)
      else:
        life._say(f"GAME {kind}: {verdict.reason} -- nothing for the "
                  f"{life.role_in(task.id) or 'other'}")
    board.resolve(task.id, verdict, t=float(data.time))

  def on_claim(event: dict) -> None:
    if (event.get("type") != "task_claimed" or state["game"] is not None
        or event.get("id") != task.id):
      return
    claims = event.get("claims") or {}
    if set(claims) != {"hider", "seeker"}:
      return
    game.assign(hider=by_root[claims["hider"]].mission.handle,
                seeker=by_root[claims["seeker"]].mission.handle)
    state["game"] = game
    if lives[0].activities is None:
      lives[0].mission.step_hooks.append(lambda: game.sense(model, data))
    game.on_over.append(settle)
    for life in lives:
      life._say(f"GAME {kind}: {by_root[claims['hider']].robot_name} hides, "
                f"{by_root[claims['seeker']].robot_name} seeks")
  board.on_event.append(on_claim)
  return task, state


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
    grid=first.mission.grid, robot_name=first.robot_name,
    heightmap=first.near_field,
    goals=ov.goals_text(thoughts=first.thoughts),
    steering=first.overseer is not None,
    others=[StreamRobot(o.mission.handle.root, o.robot_name, o.telemetry_status,
                        metabolism=o.metabolism, thoughts=o.thoughts,
                        goals=ov.goals_text(thoughts=o.thoughts),
                        steering=o.overseer is not None, grid=o.mission.grid,
                        heightmap=o.near_field)
            for o in others])
  first.mission.step_hooks.append(recorder.step_hook)
  if first.boards is not None:
    first.boards.on_event.append(recorder.emit)
  if first.ledger is not None:
    first.ledger.on_event.append(recorder.emit)
  if first.tasks is not None:
    first.tasks.on_event.append(recorder.emit)
  for life in lives:
    life.on_event.append(recorder.emit)
    life.thoughts.on_event.append(recorder.emit)
    if life.journal is not None:
      life.journal.on_event.append(recorder.emit)
  first.encounters.on_event.append(recorder.emit)
  return recorder


def run_pair(lives: list, starts=None, max_sim_time: float = 600.0,
             explore_budget: float | None = None,
             stop_when: Callable | None = None,
             record: str | None = None) -> list[dict]:
  """Both days from one loop; each robot's summary in order."""
  cfg = world_config(lives[0].world)
  starts = starts or (cfg["start"], cfg["start2"])
  budget = explore_budget if explore_budget is not None else cfg["explore_budget"]
  if stop_when is not None:
    lives[0].stop_when(lambda: stop_when(lives))
  recorder = record_pair(lives, record) if record is not None else None
  days = [life.begin(start, max_sim_time=max_sim_time, explore_budget=budget)
          for life, start in zip(lives, starts)]
  aborted = False
  try:
    tick.run_many([(life.mission.swap, day) for life, day in zip(lives, days)],
                  name="pair")
  except MissionAborted:
    aborted = True
  finally:
    for life in lives:
      life.mission.close()
    if recorder is not None:
      recorder.close()
  return [life.end(aborted) for life in lives]


def run_demo_pair(world: str = "room_hub", max_sim_time: float = 300.0,
                  view: bool = False, realtime: bool = True,
                  pack: str = "demo", errands=("carry", "none"),
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



