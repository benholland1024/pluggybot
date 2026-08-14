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

import math
from typing import Literal

import mujoco

from pluggybot.behavior.navigation import STRIKES_TO_FINISH, plan
from pluggybot.hub.coupling import (
  HUB_STATION_YS, module_power_contact, rack_charge_contact,
)
from pluggybot.hub.mission import (
  MissionAborted, HubMission, RackPose, charge_standoff,
)
from pluggybot.power import MODULE_IDLE_W, Battery

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
               module: str = "module_lcd") -> None:
    self.model, self.data = model, data
    self.module = module
    self.mission = HubMission(model, data, viewer=viewer, realtime=realtime,
                              rack=rack)
    self.battery = Battery(model, capacity_wh=battery_wh)
    self.mission.step_hooks.append(self._power_step)
    self.state: State = "EXPLORE"
    self.errand_pending = errand
    self.charging_now = False
    self.tool_powered = False
    self.charge_cycles = 0
    self.swaps_done = 0
    self.tool_powered_s = 0.0
    self.log: list[str] = []

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
    line = f"t={self.data.time:6.1f}s  bat={self.battery.fraction:5.0%}  {msg}"
    self.log.append(line)
    print(line, flush=True)

  @property
  def needs_charge(self) -> bool:
    return self.battery.energy_wh < LOW_BATTERY_WH

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

  def run_errand(self, station_y: float, use_at: tuple[float, float]) -> None:
    """Fetch a tool, take it somewhere, and put it back."""
    self.state = "SWAP_PICK"
    self.mission.swap_at_bay(station_y, "pick", module=self.module)
    carried = self.mission.swap.module_state(self.module)["on_fork"]
    self.swaps_done += 1
    self._say(f"SWAP_PICK {'done -- carrying the module' if carried else 'FAILED'}")

    self.state = "USE_TOOL"
    self.mission.drive_to(*use_at, timeout=60.0)
    still = self.mission.swap.module_state(self.module)["on_fork"]
    self._say(f"USE_TOOL: arrived{'' if still else ' -- BUT DROPPED THE TOOL'}")

    self.state = "SWAP_RETURN"
    self.mission.swap_at_bay(station_y, "return", module=self.module)
    stowed = self.mission.swap.module_state(self.module)["hung"]
    self.swaps_done += 1
    self._say(f"SWAP_RETURN {'done -- module stowed' if stowed else 'FAILED'}")
    self.errand_pending = False

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
        elif self.errand_pending:
          self.run_errand(station_y, use_at)
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
      "module_stowed": module["hung"],
      "rack_discovered": self.mission.rack_discovered,
      "collision_steps": self.mission.collision_steps,
      "sim_time": float(self.data.time),
    }


def run_demo(start=(0.5, 3.0, math.pi / 2), view: bool = False,
             realtime: bool = True, battery_wh: float = DEMO_CAPACITY_WH,
             max_sim_time: float = 600.0,
             explore_budget: float = 90.0) -> dict:
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  life = HubLifecycle(model, data, viewer=viewer, realtime=realtime,
                      battery_wh=battery_wh)
  try:
    return life.run(start, max_sim_time=max_sim_time,
                    explore_budget=explore_budget)
  finally:
    if viewer is not None:
      viewer.close()
