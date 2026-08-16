"""The hub-era mission loop (milestone 8): explore, charge, swap tools.

Milestone 7's lifecycle closed the loop that names the project — explore
until the battery runs low, drive to an outlet, plug in, charge, resume. This
is the same shape with the hub as the destination, and one capability added:

  EXPLORE ---- battery low ----> GO_CHARGE --> CHARGE --+
     ^  frontier-drive the map      nose into    press  |
     |  while watching for the      the charge   until  |
     |  rack's fiducial             bay          full   |
     |                                                  |
     +--------- recharged, errand pending --------------+
                              |
                              v
                    SWAP_PICK -> (use the tool) -> SWAP_RETURN -> DONE

Two things are deliberately the same as milestone 7, because they were
measured there and the hub does not change them: the battery drains against
real actuator effort (pluggybot.power), and charging is confirmed by an
ELECTRICAL criterion rather than by position — here `rack_charge_contact`,
both pogo pins on the bumper. That criterion is why charging works whatever
the fork is carrying, and it is the same seam the plug module will use when
it charges away from the hub.
"""

import json
import math
from pathlib import Path
from typing import Literal

import mujoco

from pluggybot.behavior.navigation import STRIKES_TO_FINISH, plan
from pluggybot.hub.coupling import (
  HUB_STATION_YS, module_power_contact, rack_charge_contact,
)
from pluggybot.hub.errand import carry_errand, drawing_errand
from pluggybot.hub.mission import (
  MissionAborted, HubMission, RackPose, charge_standoff,
)
from pluggybot.power import MODULE_IDLE_W, Battery
from pluggybot.telemetry.recorder import TelemetryRecorder

State = Literal["EXPLORE", "GO_CHARGE", "CHARGE", "SWAP_PICK", "USE_TOOL",
                "SWAP_RETURN", "DONE"]

# Reserve is absolute energy, not a fraction of the pack -- the milestone-7
# lesson: the cost of getting home is set by the ROOM, not by the battery.
LOW_BATTERY_WH = 0.35
CHARGED = 0.90
DEMO_CAPACITY_WH = 0.7      # scaled demo cell: honest power draw, capacity
                            # sized so one explore + one errand actually
                            # runs the pack down and the loop has to charge

CHARGE_CREEP = 0.04         # m/s nosing into the pins
CHARGE_PRESS = 0.012        # m/s held press while charging: the milestone-7
                            # lesson -- contacts need sustained press, or the
                            # suspension relaxes and the circuit opens
CHARGE_APPROACH_MAX = 0.55  # m of creep before giving up on finding the pins
CHARGE_TIMEOUT = 400.0      # s of charging before calling it stuck
UNDOCK_REVERSE = 0.30       # m backed off the rack afterwards


class HubLifecycle:
  """Battery-driven mission over the hub: explore, charge, run a tool errand."""

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               battery_wh: float = DEMO_CAPACITY_WH,
               errand: bool = True, rack: RackPose | None = None,
               module: str = "module_lcd",
               grid_bounds: tuple[float, float, float, float] = (-3, -3, 7, 7),
               low_battery_wh: float = LOW_BATTERY_WH,
               errands=None, boards=None) -> None:
    self.model, self.data = model, data
    # The module whose electrical seating the power model watches. It follows
    # the errand queue -- a robot that draws and then grips is carrying a
    # different tool in each phase, and the coupling criterion has to be asked
    # about the one actually on the fork.
    self.module = errands[0].module if errands else module
    self.low_battery_wh = low_battery_wh
    self.boards = boards
    self.mission = HubMission(model, data, viewer=viewer, realtime=realtime,
                              rack=rack, grid_bounds=grid_bounds)
    self.battery = Battery(model, capacity_wh=battery_wh)
    self.mission.step_hooks.append(self._power_step)
    self.state: State = "EXPLORE"
    # A QUEUE, not a flag: "two drawings on two boards with charging in
    # between" is the acceptance test for issue #12, and a boolean cannot
    # express it. `errands=None` with `errand=True` keeps the milestone-8
    # call shape -- run() then builds the one carry errand out of its own
    # station_y / use_at arguments.
    self.errands: list = list(errands) if errands is not None else []
    self.want_default_errand = errand and errands is None
    self.errand_results: list[dict] = []
    self.charging_now = False
    self.tool_powered = False
    self.charge_cycles = 0
    self.swaps_done = 0
    self.tool_powered_s = 0.0
    self.log: list[str] = []
    self.status = ""                    # the latest _say message, bare
    # Callbacks fired with (sim_time, bare_message) on every _say line --
    # the live publisher streams narration through here as event messages.
    self.say_hooks: list = []

  # ---- power ---------------------------------------------------------------

  def _power_step(self) -> None:
    """Drain (or charge) once per physics step, whatever phase is running.

    A coupled module is a load, gated on the coupling's own electrical
    criterion rather than on "are we carrying it" -- the same reason charging
    is gated on contact and not on position.
    """
    dt = self.model.opt.timestep
    self.charging_now = rack_charge_contact(self.model, self.data)
    self.tool_powered = module_power_contact(self.model, self.data, self.module)
    if self.tool_powered:
      self.tool_powered_s += dt
    self.battery.update(self.data, dt, charging=self.charging_now,
                        tool_w=MODULE_IDLE_W if self.tool_powered else 0.0)

  def _say(self, msg: str) -> None:
    self.status = msg
    line = f"t={self.data.time:6.1f}s  bat={self.battery.fraction:5.0%}  {msg}"
    self.log.append(line)
    print(line, flush=True)
    for hook in self.say_hooks:
      hook(float(self.data.time), msg)

  def telemetry_status(self) -> dict:
    """The per-frame robot record for the telemetry recorder: lifecycle
    state, the _say narration, and the battery gauges. The bare message,
    not the formatted log line -- t and battery already ride in the frame,
    and the site wants "carrying the module", not a duplicate dashboard."""
    return {
      "state": self.state,
      "status": self.status,
      "battery": {"frac": round(self.battery.fraction, 4),
                  "watts": round(self.battery.last_power_w, 2),
                  "charging": self.charging_now},
    }

  @property
  def needs_charge(self) -> bool:
    # The reserve is a PARAMETER of the world, not of the pack (issue #6):
    # the cost of getting home is set by the floor plan, and home_world's
    # worst return trip is nearly twice room_hub's.
    return self.battery.energy_wh < self.low_battery_wh

  # ---- phases --------------------------------------------------------------

  def explore(self) -> None:
    """Frontier-drive the map until the battery calls, or the map is done.

    The rack's fiducial is watched for throughout (mission.start_discovery),
    so exploring is also how the robot learns where its hub is -- the same
    trip that maps the room localizes the dock.
    """
    strikes = 0
    while not self.needs_charge and self.data.time < self.max_sim_time:
      if self.data.time > self.explore_deadline:
        self.map_done = True
        self._say("EXPLORE: budget spent, stopping")
        return
      path, status = plan(self.mission.grid, self.mission.pose, self.blacklist)
      if status == "ok":
        strikes = 0
        wx, wy = self.mission.grid.cell_to_world(*path[-1])
        self.mission.drive_to(wx, wy, timeout=25.0)
        continue
      self.mission._spin()
      strikes += 1
      if status == "no-frontiers" or strikes >= STRIKES_TO_FINISH:
        self.map_done = True
        self._say(f"EXPLORE done ({status})")
        return
    self._say("EXPLORE -> GO_CHARGE (battery low)")

  def go_charge(self) -> bool:
    """Navigate to the charge bay and press until the pins connect."""
    self.mission.refresh_rack()
    sx, sy, hd = charge_standoff(self.mission.rack)
    if not self.mission.drive_to(sx, sy, timeout=90.0):
      self._say("GO_CHARGE: no route to the charge bay")
      return False
    self.mission.face(hd)
    self.mission.refine_standoff(sx, sy, hd)
    # Creep until the electrical criterion fires -- position is believed,
    # contact is known.
    why = self.mission.swap._drive_until(
      CHARGE_APPROACH_MAX, CHARGE_CREEP, stall_stop=True,
      stop_fn=lambda: rack_charge_contact(self.model, self.data))
    if not rack_charge_contact(self.model, self.data):
      self._say(f"GO_CHARGE: no charge contact ({why})")
      return False
    self._say("GO_CHARGE -> CHARGE (pins connected)")
    return True

  def charge(self) -> None:
    """Hold the press until full, then back off."""
    t0 = self.data.time
    while (self.battery.fraction < CHARGED
           and self.data.time - t0 < CHARGE_TIMEOUT):
      self.mission._drive(0.25, CHARGE_PRESS, 0.0)
      if not self.charging_now:
        # contact dropped: press again briefly, then give up on this attempt
        self.mission._drive(1.0, CHARGE_CREEP, 0.0)
        if not self.charging_now:
          self._say("CHARGE: lost the pins")
          return
    self.charge_cycles += 1
    self._say(f"CHARGE complete ({self.battery.fraction:.0%}) -- backing off")
    self.mission.swap._drive_until(UNDOCK_REVERSE, -0.08, stall_stop=False)

  def run_errand(self, errand) -> dict:
    """Fetch a tool, take it somewhere, DO something, and put it back.

    The middle is the errand's own `use` callable (hub/errand.py). Everything
    around it -- which bay, verifying the pick electrically, verifying the
    stow by hanging, restoring the arm -- is identical whatever the tool is,
    which is exactly why it lives here and only here.

    A use-phase that raises is caught and reported: the tool is still on the
    fork, and driving it back to its bay is strictly better than abandoning a
    module in the middle of the living room. `MissionAborted` (the viewer
    closing) is deliberately NOT caught -- that is a request to stop, not a
    failure to recover from.
    """
    self.module = errand.module
    self.state = "SWAP_PICK"
    self.mission.swap_at_bay(errand.station_y, "pick", module=self.module)
    carried = self.mission.swap.module_state(self.module)["on_fork"]
    self.swaps_done += 1
    self._say(f"SWAP_PICK {'done -- carrying the module' if carried else 'FAILED'}"
              f" ({errand.name})")

    self.state = "USE_TOOL"
    self.mission.drive_to(*errand.use_at, timeout=60.0)
    still = self.mission.swap.module_state(self.module)["on_fork"]
    self._say(f"USE_TOOL: arrived{'' if still else ' -- BUT DROPPED THE TOOL'}")
    used: dict = {}
    if errand.use is not None and still:
      try:
        used = errand.use(self) or {}
      except MissionAborted:
        raise
      except Exception as e:                      # noqa: BLE001 -- see docstring
        used = {"error": f"{type(e).__name__}: {e}"}
        self._say(f"USE_TOOL FAILED: {used['error']} -- stowing the tool anyway")

    self.state = "SWAP_RETURN"
    self.mission.swap_at_bay(errand.station_y, "return", module=self.module)
    stowed = self.mission.swap.module_state(self.module)["hung"]
    self.swaps_done += 1
    self._say(f"SWAP_RETURN {'done -- module stowed' if stowed else 'FAILED'}")
    result = {"errand": errand.name, "module": errand.module,
              "picked": carried, "stowed": stowed, **used}
    self.errand_results.append(result)
    return result

  # ---- the loop ------------------------------------------------------------

  def run(self, start: tuple[float, float, float],
          station_y: float = HUB_STATION_YS[0],
          use_at: tuple[float, float] = (-1.2, 2.5),
          max_sim_time: float = 600.0,
          explore_budget: float = 90.0) -> dict:
    self.max_sim_time = max_sim_time
    self.blacklist: set = set()
    self.map_done = False
    aborted = False
    if self.want_default_errand:
      self.errands = [carry_errand(self.module, station_y, use_at)]
    try:
      self.mission.start_at(*start)
      self.mission.start_discovery()
      self.mission._spin()               # seed the map before deciding anything
      self.explore_deadline = self.data.time + explore_budget
      self._say("mission start")

      # A real arbitration loop, not a fixed script. Priority order, and the
      # reasons: charging outranks everything (a flat robot does nothing at
      # all); then the ERRAND, because that is the job the robot was given
      # -- exploring is background work, and letting it go first meant the
      # robot mapped, ran flat, charged, mapped again, and never got round
      # to the task it existed for. Whatever the battery does mid-errand,
      # the next pass through here reacts to it.
      while self.data.time < max_sim_time and not self.battery.empty:
        if self.needs_charge:
          self.state = "GO_CHARGE"
          if not self.go_charge():
            break
          self.state = "CHARGE"
          self.charge()
        elif self.errands:
          # Pop BEFORE running: an errand that raises must not be retried
          # forever, and a queue that only shortens on success is an infinite
          # loop dressed as a task list.
          self.run_errand(self.errands.pop(0))
        elif not self.map_done:
          self.state = "EXPLORE"
          self.explore()
        else:
          break

      self.state = "DONE"
      self._say("mission complete" if not self.battery.empty
                else "BATTERY DEAD -- mission over")
    except MissionAborted:
      aborted = True
    finally:
      self.mission.close()

    module = self.mission.swap.module_state(self.module)
    return {
      "state": self.state,
      "aborted": aborted,
      "charge_cycles": self.charge_cycles,
      "swaps_done": self.swaps_done,
      "battery": self.battery.fraction,
      # The module the LAST errand carried. With a queue of errands over
      # different tools, "is it stowed" is per errand -- `errands` carries
      # each one's verdict, and this stays the single-errand summary the
      # milestone-8 demos print.
      "module_stowed": module["hung"],
      "errands": list(self.errand_results),
      "errands_left": len(self.errands),
      "boards": self.boards.snapshot() if self.boards is not None else {},
      "rack_discovered": self.mission.rack_discovered,
      "collision_steps": self.mission.collision_steps,
      "sim_time": float(self.data.time),
    }


def home_activities(model, data):
  """The home world's task state machines (issue #8).

  Built per model rather than per world-config, because an Activity binds to
  sensor addresses and geom ids -- both of which belong to one compiled
  MjModel and are meaningless against another.
  """
  from pluggybot.activity.base import ActivitySet
  from pluggybot.activity.plate import PlateGate
  return ActivitySet([PlateGate(model, data)])


def board_book(world: str, state: str | None = None):
  """The world's drawing surfaces as persistent state (issue #12), or None
  for a world with no boards in it.

  `state` is a JSON file the boards live in ACROSS runs. Without one they are
  blank at every mission start, which is what tests and one-off demos want;
  with one, the site's robot walks into a house whose whiteboards still carry
  yesterday's drawing.
  """
  from pluggybot.hub.boards import BoardBook
  cfg = world_config(world)
  if not cfg["meta"]:
    return None
  meta = json.loads(Path(cfg["meta"]).read_text())
  return BoardBook.for_meta(meta, path=state)


def errands_for(kind: str, world: str, book=None) -> list:
  """The named errand queues a demo or the website can ask for.

  This is the menu an overseer will eventually choose from (issue #15), which
  is why it is a lookup by NAME rather than a pile of flags: adding "draw a
  house on whiteboard_b" must not mean adding an argument to serve.py.
  """
  cfg = world_config(world)
  if kind == "carry":
    return [carry_errand(use_at=cfg["use_at"])]
  if kind == "none":
    return []
  if kind in ("draw", "draw2"):
    if book is None or not len(book):
      raise ValueError(f"the {world} world has no whiteboards to draw on")
    from pluggybot.hub.drawing import Board
    meta = json.loads(Path(cfg["meta"]).read_text())
    # Two boards, two different figures: the acceptance test for issue #12 is
    # "two drawings on two boards with charging in between", and drawing the
    # same figure twice would not catch a board id threaded through by
    # accident.
    names = list(meta["boards"])[:2 if kind == "draw2" else 1]
    figures = ("house", "tree", "sun", "robot")
    return [drawing_errand(book, name, Board.from_meta(meta["boards"][name]),
                           program_name=figures[i % len(figures)])
            for i, name in enumerate(names)]
  raise ValueError(f"unknown errand queue {kind!r} "
                   "(carry, draw, draw2 or none)")


def world_config(world: str) -> dict:
  """Everything the lifecycle needs to know about a world, in one place.

  room_hub keeps its historical constants; home_world's come from the
  generator's own module, so the demo can never disagree with the world it
  runs in (the sidecar and this dict are written from the same source).
  """
  if world == "home":
    from pluggybot.home import world as home
    return {
      "model": "models/home_world.xml", "model_name": "home_world",
      "rack": RackPose(home.HOME_RACK_POS[0], home.HOME_RACK_POS[1],
                       math.radians(home.HOME_RACK_YAW)),
      "grid_bounds": home.GRID_BOUNDS,
      "start": tuple(home.SPAWNS["start"]),
      "use_at": (1.5, 1.8),
      "battery_wh": home.HOME_DEMO_CAPACITY_WH,
      "low_battery_wh": home.HOME_LOW_BATTERY_WH,
      "explore_budget": 240.0,
      "activities": home_activities,
      # The generator sidecar, which is also where the BOARDS are described
      # (issue #12). One source again: the whiteboard the errand drives to is
      # the whiteboard the website renders.
      "meta": "models/home_world.meta.json",
    }
  if world == "room_hub":
    return {
      "model": "models/room_hub.xml", "model_name": "room_hub",
      "rack": None,                       # RackPose.prior() is this world's
      "grid_bounds": (-3, -3, 7, 7),
      "start": (0.5, 3.0, math.pi / 2),
      "use_at": (-1.2, 2.5),
      "battery_wh": DEMO_CAPACITY_WH,
      "low_battery_wh": LOW_BATTERY_WH,
      "explore_budget": 90.0,
      "activities": None,      # room_hub has no activities yet
      "meta": None,            # ...and no whiteboards: the standing board
                               # lives in the bare hub_world, which is not a
                               # navigated room
    }
  raise ValueError(f"unknown world {world!r} (room_hub or home)")


def run_demo(start=None, view: bool = False,
             realtime: bool = True, battery_wh: float | None = None,
             max_sim_time: float = 600.0,
             explore_budget: float | None = None,
             record: str | None = None,
             world: str = "room_hub",
             errand: str = "carry", board_state: str | None = None) -> dict:
  """Run a whole mission. `errand` names a queue off the menu (errands_for).

  Callers that want to hand in errands they built themselves -- the overseer,
  once issue #15 has it choosing rather than picking a preset -- should build
  the HubLifecycle directly, as scripts/home_draw.py does. The book has to
  travel WITH them: a drawing errand closes over the book it was built
  against, so a second one opened here would have the robot drawing into one
  book while telemetry reported the other.
  """
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  book = board_book(world, state=board_state)
  life = HubLifecycle(model, data, viewer=viewer, realtime=realtime,
                      battery_wh=battery_wh or cfg["battery_wh"],
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], boards=book,
                      errands=errands_for(errand, world, book))
  # Activities poll on the SAME per-step seam the battery drains through and
  # telemetry decimates from -- one hook for the whole world's state
  # machines, whatever their number.
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))
  recorder = None
  if record is not None:
    recorder = TelemetryRecorder(model, data, record,
                                 model_name=cfg["model_name"],
                                 status_fn=life.telemetry_status,
                                 activities=activities, boards=book)
    life.mission.step_hooks.append(recorder.step_hook)
    # Strokes and erasures are EVENTS, not poses: ink is not a body, so a
    # recording without these lines replays a robot miming at a blank wall.
    if book is not None:
      book.on_event.append(recorder.emit)
  try:
    return life.run(start or cfg["start"], use_at=cfg["use_at"],
                    max_sim_time=max_sim_time,
                    explore_budget=explore_budget or cfg["explore_budget"])
  finally:
    if recorder is not None:
      recorder.close()
    if viewer is not None:
      viewer.close()
