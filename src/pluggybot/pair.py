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

import mujoco

from pluggybot import tick
from pluggybot.lifecycle import (
  HubLifecycle, board_book, errands_for, world_config,
)
from pluggybot.mission.mission import MissionAborted
from pluggybot.robot import FIRST, SECOND, world_with_robots


def build_pair(world: str = "room_hub", pack: str = "demo",
               errands=("carry", "none"), board_state: str | None = None,
               view: bool = False, realtime: bool = False,
               handles: tuple = (FIRST, SECOND), **life_kw) -> list:
  """One world, two lifecycles. `errands` names each robot's preset queue
  (`errands_for`'s names); everything else in `life_kw` goes to both."""
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
  lives = []
  for handle, errand in zip(handles, errands):
    life = HubLifecycle(model, data, viewer=viewer if handle is handles[0] else None,
                        realtime=realtime, battery_wh=default_wh,
                        rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                        low_battery_wh=cfg["low_battery_wh"], boards=book,
                        world=world, errands=errands_for(errand, world, book),
                        handle=handle, **life_kw)
    lives.append(life)
  # Each mission is told where the OTHERS say they are, and its lidar drops
  # their bodies from the scan (see the module doc and `Lidar.exclude_robot`).
  for life in lives:
    others = [other for other in lives if other is not life]
    life.mission.others = [other.mission.pose_xy for other in others]
    for other in others:
      life.mission.lidar.exclude_robot(other.mission.handle.root)
  # The world's activities sense once per step, on the first robot's hooks:
  # they are the world's, and two copies would sense everything twice.
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    lives[0].mission.step_hooks.append(activities.step_hook(model, data))
  return lives


def run_pair(lives: list, starts=None, max_sim_time: float = 600.0,
             explore_budget: float | None = None,
             stop_when: Callable | None = None) -> list[dict]:
  """Both days from one loop; each robot's summary in order."""
  cfg = world_config(lives[0].world)
  starts = starts or (cfg["start"], cfg["start2"])
  budget = explore_budget if explore_budget is not None else cfg["explore_budget"]
  if stop_when is not None:
    lives[0].stop_when(lambda: stop_when(lives))
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
  return [life.end(aborted) for life in lives]


def run_demo_pair(world: str = "room_hub", max_sim_time: float = 300.0,
                  view: bool = False, realtime: bool = True,
                  pack: str = "demo", errands=("carry", "none"),
                  board_state: str | None = None) -> list[dict]:
  lives = build_pair(world, pack=pack, errands=errands, board_state=board_state,
                     view=view, realtime=realtime)
  return run_pair(lives, max_sim_time=max_sim_time)



