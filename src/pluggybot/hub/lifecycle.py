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
from pluggybot.hub.census import Zone
from pluggybot.hub.errand import (
  carry_errand, census_errand, dance_errand, drawing_errand,
)
from pluggybot.hub.mission import (
  MissionAborted, HubMission, RackPose, charge_standoff,
)
from pluggybot.hub.cadence import CHECK_S
from pluggybot.hub import energy as energy_model
from pluggybot.hub.overseer import THINK_SLICE_S
from pluggybot.hub.questions import clean_answer
from pluggybot.hub.screen import face_for
from pluggybot.hub import scoring, strokes
from pluggybot.power import MODULE_IDLE_W, Battery
from pluggybot.telemetry.protocol import ROBOT_ROOT
from pluggybot.telemetry.recorder import TelemetryRecorder

State = Literal["EXPLORE", "GO_CHARGE", "CHARGE", "DECIDE", "SWAP_PICK",
                "USE_TOOL", "SWAP_RETURN", "DONE"]

# Reserve is absolute energy, not a fraction of the pack -- the milestone-7
# lesson: the cost of getting home is set by the ROOM, not by the battery.
LOW_BATTERY_WH = 0.35
CHARGED = 0.90
DEMO_CAPACITY_WH = 0.7      # scaled demo cell: honest power draw, capacity
                            # sized so one explore + one errand actually
                            # runs the pack down and the loop has to charge
#: ...and the pack a WATCHED world runs on (issue #15, `--pack hosting`).
#: Sized from the measured errand costs rather than picked: room_hub's dearest
#: job is 0.57 Wh, so ~10 errands to a charge and a rhythm measured in hours
#: rather than minutes. See `home.HOME_HOSTING_CAPACITY_WH` for why the
#: reserve does NOT scale alongside it.
HOSTING_CAPACITY_WH = 6.0

CHARGE_CREEP = 0.04         # m/s nosing into the pins
CHARGE_PRESS = 0.012        # m/s held press while charging: the milestone-7
                            # lesson -- contacts need sustained press, or the
                            # suspension relaxes and the circuit opens
CHARGE_APPROACH_MAX = 0.55  # m of creep before giving up on finding the pins
#: Sim seconds of charging before calling it stuck, on a DEMO cell.
#:
#: ⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS, and this one was sized
#: against a 0.7 Wh pack. The deployed sim runs an 8 Wh one, which at the
#: measured rate (hub/energy.json, `chargeW`) needs ~1340 s to refill -- so
#: the fixed 400 s cap silently ended every charge partway up and narrated
#: "CHARGE complete (79 %)". `charge_timeout` below scales it with the pack.
#: It still computes 400 s on BOTH demo cells, where the arithmetic asks for
#: less, so nothing about an existing mission moves.
CHARGE_TIMEOUT = 400.0
#: ...and the floor under that scaling, so a tiny pack still gets long enough
#: to seat the pins and be believed.
CHARGE_TIMEOUT_MIN = 400.0
#: How much longer than the arithmetic says: the press is not perfectly
#: efficient, contact can drop and be re-made, and a cycle that times out one
#: second short of full is a charge that did not happen.
CHARGE_TIMEOUT_SLACK = 1.4
SCREEN_SENSE_S = 0.02       # sim seconds between power scans of a display
                            # the robot is NOT carrying (issue #13)
UNDOCK_REVERSE = 0.30       # m backed off the rack afterwards
#: Sim seconds an overseer-chosen `explore` runs for before the arbitration
#: loop gets to reconsider (issue #15). Bounded on purpose: without it one
#: `explore` decision eats the whole mission, and the point of an overseer is
#: that it decides repeatedly.
DECIDED_EXPLORE_S = 45.0
#: ...and how long `idle` stands still for. Long enough to read on the stream
#: as a deliberate pause, short enough not to be a way of doing nothing all day.
DECIDED_IDLE_S = 4.0
#: ...and how long the loop stands by when a PRODUCER world has momentarily
#: run out of work (issue #23). Short, because the only reason to bound it is
#: to keep re-checking `needs_charge`; the day ends on `max_sim_time`, not on
#: an idle moment.
WAIT_FOR_WORK_S = 5.0
#: Battery fraction below which a CHOSEN `charge` is worth making the trip for.
#:
#: ⚠ This closes a points farm, not a physics problem. `charge` is a scored
#: task (issue #14) and the drive to the rack costs energy, so without a floor
#: an overseer can spend battery driving out and then earn points for putting
#: it back, forever -- perpetual motion paid in points. The forced charge is
#: untouched: `needs_charge` fires on absolute reserve and never consults this.
TOP_UP_BELOW = 0.75
#: How many times one errand may be put back for a charge before it is given
#: up on (issue #15). The gate below is "charge, then try again", and a charge
#: that does not raise the pack -- lost pins, a timeout, a rack that cannot be
#: reached -- would otherwise turn that into a spin. Two, because the first
#: retry is the ordinary case (the charge worked) and the second is already
#: evidence that charging is not what is wrong.
MAX_ERRAND_DEFERRALS = 2
#: Visitor messages shown to the overseer at once (issue #16). Small: the
#: robot answers at most one per turn, and a wall of them is input tokens
#: spent on messages it is not going to get to -- the inbox keeps the rest.
VISITORS_SHOWN = 5
#: ...and offered tasks shown at once (issue #21). Small for the same reason:
#: the robot takes at most one per turn, and a wall of offers is input tokens
#: spent on jobs it will not reach.
TASKS_SHOWN = 5
#: ⚠ How long an offer stands, how often one appears, how many may stand at
#: once and how long a target rests are NO LONGER HERE. They are configuration
#: -- hub/cadence.json, per world, `$PLUGGY_CADENCE` to override -- because
#: issue #23's last acceptance line asks for exactly that, and because they
#: want re-tuning against a mission's own clock by somebody who is not editing
#: Python. `SEED_TTL_S` and `SEED_STANDING_TTL_S` are gone with the placeholder
#: `seed_tasks` they belonged to; see `hub/cadence.py`.


class HubLifecycle:
  """Battery-driven mission over the hub: explore, charge, run a tool errand."""

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               battery_wh: float = DEMO_CAPACITY_WH,
               errand: bool = True, rack: RackPose | None = None,
               module: str = "module_lcd",
               grid_bounds: tuple[float, float, float, float] = (-3, -3, 7, 7),
               low_battery_wh: float = LOW_BATTERY_WH,
               errands=None, boards=None, screen=None, ledger=None,
               overseer=None, journal=None, world: str = "room_hub",
               inbox=None, tasks=None, producer=None,
               energy=None) -> None:
    self.model, self.data = model, data
    # The visitor channel (issue #16). None -- the default -- means nobody can
    # talk to this robot, which is every test, every demo and every recording
    # except the served one.
    self.inbox = inbox
    self.replies: list[dict] = []
    #: fired with each `visitor_reply` message; wire the publisher and the
    #: recorder in, exactly as for boards, the ledger and the journal.
    self.visitor_hooks: list = []
    # Which world this is, which the overseer needs to build an errand out of
    # a decision (issue #15) -- the same name `world_config` is keyed by, so
    # there is no second place a world can be named.
    self.world = world
    # The LLM overseer, or None. None is the DEFAULT and the whole arbitration
    # loop below is unchanged without it: every existing demo, mission test and
    # recording has to behave exactly as it did.
    self.overseer = overseer
    self.journal = journal
    self.decisions: list[dict] = []
    # The module whose electrical seating the power model watches. It follows
    # the errand queue -- a robot that draws and then grips is carrying a
    # different tool in each phase, and the coupling criterion has to be asked
    # about the one actually on the fork.
    self.module = errands[0].module if errands else module
    self.low_battery_wh = low_battery_wh
    self.boards = boards
    # The points ledger (issue #14). Optional: a physics test or a spike has
    # nothing to score against and wants no state file. When it IS here, every
    # finished task is evaluated by hub/scoring.py and the verdict is banked
    # through it -- the lifecycle measures nothing and pays nothing itself.
    self.ledger = ledger
    self.verdicts: list[dict] = []
    # The task board (issue #21). Optional, like the ledger and for the same
    # reason: a physics test has nobody offering it work. When it IS here the
    # loop sweeps it for lapsed offers, takes claimable ones when it has
    # nothing else to do, and closes each one with the SAME verdict that pays
    # for it -- there is no second judgement of a task anywhere.
    self.tasks = tasks
    # ...and the thing that PUTS jobs on it (issue #23). Optional again, and
    # separately from the board: a test that hands in three offers of its own
    # wants the board without a world generating more behind its back, and a
    # restart against a persisted board resumes work rather than re-seeding.
    self.producer = producer
    self.claimed: list[str] = []
    self._next_task_check = 0.0
    # The LCD module's display (issue #13). The lifecycle drives the resting
    # face off its own state; an errand's use-phase may take the screen over
    # (`Screen.held`) and gets it back automatically at the next state change.
    self.screen = screen
    self._face_state: str | None = None
    self._face_shown: tuple[str, str] | None = None
    self._next_screen_sense = 0.0
    self.mission = HubMission(model, data, viewer=viewer, realtime=realtime,
                              rack=rack, grid_bounds=grid_bounds)
    self.battery = Battery(model, capacity_wh=battery_wh)
    # What an errand COSTS here, measured (issue #15). Read per world from
    # hub/energy.json, `$PLUGGY_ENERGY` to re-point -- and always present,
    # unlike the ledger or the task board: "can I finish this before the pack
    # runs out" is a question every mission asks, including the ones with
    # nobody scoring them.
    self.energy = energy or energy_model.load(world)
    self._deferrals: dict[str, int] = {}
    self.mission.step_hooks.append(self._power_step)
    # The world's own clock (issue #23): offers appear and lapse on the same
    # per-step seam the battery drains through, so a job put up while the
    # robot is halfway through an errand is put up THEN and not on whichever
    # arbitration pass happens next. See `_task_step`.
    self.mission.step_hooks.append(self._task_step)
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
    self._screen_step()

  def _screen_step(self) -> None:
    """Keep the display's power reading current, and its resting face.

    Two economies, both because this runs at 500-1000 Hz. The power scan is
    O(ncon) in Python, so when the display IS the errand's module the answer
    is already in hand (`tool_powered`) and is reused; otherwise the module
    is hanging on the rack, where "did it just get power" is a question worth
    asking ten times a second and not a thousand. And the automatic face is
    recomputed only when the mission state has actually moved, which is also
    the moment an errand's override is handed back.
    """
    if self.screen is None:
      return
    if self.screen.module == self.module:
      self.screen.sense(self.model, self.data, powered=self.tool_powered)
    elif self.data.time >= self._next_screen_sense:
      self._next_screen_sense = self.data.time + SCREEN_SENSE_S
      self.screen.sense(self.model, self.data)
    if self.state != self._face_state:
      self._face_state = self.state
      self.screen.release()
      self._face_shown = None
    if self.screen.held:
      # The errand owns the screen; forget what we last put there, so the
      # resting face is re-applied the moment it hands the screen back --
      # whether that is a state change or a direct `release()`.
      self._face_shown = None
      return
    # Compare the pair BEFORE handing it over: `Screen.face` builds a dict and
    # diffs it, which is cheap once and is not free a thousand times a second.
    resting = face_for(self.state, self.battery.fraction)
    if resting != self._face_shown:
      self._face_shown = resting
      self.screen.face(*resting)

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

  # ---- scoring (issue #14) -------------------------------------------------

  def _bank(self, verdict) -> dict | None:
    """Narrate an evaluator's verdict and hand it to the ledger.

    This is the lifecycle's ENTIRE role in scoring: it runs the tasks and it
    asks hub/scoring.py to judge the finished one. It never decides a verdict
    and never moves a balance -- scoring.py measures and judges, ledger.py
    pays, and neither will take an answer from the task itself.

    `None` means the task is not scoreable (no evaluator, or a hand-built
    errand with no task), which is one of the four tiers and not an error.
    """
    if verdict is None:
      return None
    self.verdicts.append(verdict.as_dict())
    if self.ledger is None:
      self._say(f"SCORE {verdict.task}: {verdict.reason} (no ledger)")
      return None
    entry = self.ledger.award(verdict, t=float(self.data.time))
    tail = " (pending a rating)" if entry["pending"] else ""
    self._say(f"SCORE {verdict.task}: {verdict.reason} -- "
              f"{entry['points']:+d} points{tail}, "
              f"balance {entry['balance']}")
    return entry

  @property
  def needs_charge(self) -> bool:
    # The reserve is a PARAMETER of the world, not of the pack (issue #6):
    # the cost of getting home is set by the floor plan, and home_world's
    # worst return trip is nearly twice room_hub's.
    return self.battery.energy_wh < self.low_battery_wh

  @property
  def charge_timeout(self) -> float:
    """How long a charge cycle is given, sized against THIS pack.

    A fixed 400 s was right for a 0.7 Wh demo cell and quietly wrong for the
    deployed 8 Wh one: at the measured rate it takes ~1340 s to refill, so
    every cycle hit the cap partway up and reported itself complete. The
    timeout is meant to catch a robot pressing on pins that are not
    conducting, which is a fault; it should never be what ends a charge that
    is working.

    ⚠ The rate it is sized against is the SLOWEST press measured, not the
    best one -- 19.4 W against 39.6 W on a different approach -- because the
    spread is geometry: how squarely the bumper meets the pins decides how
    hard the wheels stall against them. See hub/energy.json.
    """
    rate = self.energy.charge_w
    if rate <= 0.0:
      return CHARGE_TIMEOUT
    need = self.charged_wh * 3600.0 / rate * CHARGE_TIMEOUT_SLACK
    return max(CHARGE_TIMEOUT_MIN, need)

  # ---- phases --------------------------------------------------------------

  def explore(self, budget: float | None = None,
              mark_done: bool = True) -> None:
    """Frontier-drive the map until the battery calls, or the map is done.

    The rack's fiducial is watched for throughout (mission.start_discovery),
    so exploring is also how the robot learns where its hub is -- the same
    trip that maps the room localizes the dock.

    `budget` bounds ONE call (issue #15): an overseer that chose to explore
    gets a slice and then the arbitration loop reconsiders, rather than one
    decision consuming the mission. `mark_done=False` goes with it, because a
    slice running out is not the same fact as the map being finished -- and
    conflating them would let the first overseer explore permanently retire
    the branch. Running out of FRONTIERS still marks it done under either
    setting: that one really is "there is nothing left to see".
    """
    strikes = 0
    deadline = (self.data.time + budget if budget is not None
                else self.explore_deadline)
    while not self.needs_charge and self.data.time < self.max_sim_time:
      if self.data.time > deadline:
        self.map_done = mark_done
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
    """Navigate to the charge bay and press until the pins connect.

    The terminal half is `mission.charge_approach` (issue #32): the standoff
    computed from the believed rack pose is only how the robot gets to the
    NEIGHBOURHOOD -- the approach itself is re-measured off the charge bay's
    own tag and verified-retried, exactly as every tool-bay approach already
    was. The blind creep this replaces forgave ~6 cm / ~10 deg of belief
    error, and a long shift's accumulated drift walked straight out of that
    envelope roughly once an hour on a hosting pack.
    """
    self.mission.refresh_rack()
    sx, sy, hd = charge_standoff(self.mission.rack)
    # Route-failure retry, same as swap_at_bay's: when nothing is reachable,
    # spin to buy map (and possibly the rack tag) and try again.
    for _ in range(2):
      if self.mission.drive_to(sx, sy, timeout=90.0):
        break
      self.mission._spin()
      self.mission.refresh_rack()
      sx, sy, hd = charge_standoff(self.mission.rack)
    else:
      self._say("GO_CHARGE: no route to the charge bay")
      return False
    # Line up on the bay's own tag and creep until the electrical criterion
    # fires -- position is believed, contact is known.
    why = self.mission.charge_approach(CHARGE_APPROACH_MAX, CHARGE_CREEP)
    if not rack_charge_contact(self.model, self.data):
      self._say(f"GO_CHARGE: no charge contact ({why})")
      return False
    self._say("GO_CHARGE -> CHARGE (pins connected)")
    return True

  def charge(self) -> None:
    """Hold the press until full, then back off."""
    t0 = self.data.time
    # What the evaluator measures this cycle against (issue #14). Read BEFORE
    # the press, because "charged" is a gain in a real quantity -- a cycle
    # that sat on the pins conducting nothing has an end fraction and no
    # energy behind it.
    before = {"t": t0, "frac": self.battery.fraction,
              "wh": self.battery.energy_wh}
    # ⚠ THE PRESS IS NOT TRAVEL. The robot is held against the rack's pins
    # for the whole cycle -- minutes of it -- with the wheels turning and the
    # chassis stationary. Left to integrate that, dead reckoning gained
    # 828 mm of imaginary progress, and every pose downstream was computed in
    # the wrong frame: the next tool fetch drove to a standoff it believed it
    # had reached, a metre from the bay, and came away with nothing.
    # See `HubSwap.pinned`.
    self.mission.swap.pinned = True
    try:
      timeout = self.charge_timeout
      while (self.battery.fraction < CHARGED
             and self.data.time - t0 < timeout):
        self.mission._drive(0.25, CHARGE_PRESS, 0.0)
        if not self.charging_now:
          # contact dropped: press again briefly, then give up on this attempt
          self.mission._drive(1.0, CHARGE_CREEP, 0.0)
          if not self.charging_now:
            self._say("CHARGE: lost the pins")
            self._bank(scoring.score_charge(self, before))
            return
    finally:
      # Cleared before the undock, which is REAL travel and must be counted.
      self.mission.swap.pinned = False
    self.charge_cycles += 1
    self._say(f"CHARGE complete ({self.battery.fraction:.0%}) -- backing off")
    self._bank(scoring.score_charge(self, before))
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
    # The job this errand discharges is now genuinely under way (issue #21) --
    # `claimed` means taken, `active` means started, and the difference is
    # what a marker on the website shows.
    if errand.task_id and self.tasks is not None:
      self.tasks.start(errand.task_id, t=float(self.data.time))
    self.state = "SWAP_PICK"
    # What this errand actually took, measured (issue #15). Recorded on every
    # errand and not only in the spike, because an estimate that is never
    # compared against an outcome is a number nobody can tell has gone stale
    # -- and because SWAP_PICK to the end of SWAP_RETURN is exactly the span
    # the arbitration loop cannot interrupt, which is what makes it the span
    # worth pricing. `scripts/energy_spike.py` reads the same figure.
    spent_from = self.battery.energy_wh
    began_at = float(self.data.time)
    self.mission.swap_at_bay(errand.station_y, "pick", module=self.module)
    carried = self.mission.swap.module_state(self.module)["on_fork"]
    self.swaps_done += 1
    self._say(f"SWAP_PICK {'done -- carrying the module' if carried else 'FAILED'}"
              f" ({errand.name})")

    self.state = "USE_TOOL"
    # ⚠ THE ANSWER IS READ, and it used to be thrown away. `drive_to` returns
    # False when it stagnated or could not plan at all, and a use-phase run
    # anyway is a pen pressing at empty air: found by issue #23, whose
    # producer is the first thing that offers work on the FAR whiteboard, 7 m
    # away through a doorway the robot has not mapped yet. The drive gave up,
    # the loop narrated "arrived", the erase probe searched for a board that
    # was not there, and the mission hung until the battery died -- ten
    # minutes of wall clock with nothing in the log after `USE_TOOL: arrived`.
    #
    # Not reaching the board is an ordinary outcome, so it is reported and
    # not raised on: the tool still goes back to its bay, the evaluator still
    # measures the world (it will find no ink) and the job closes `failed`.
    # A robot that could not get there is a different thing from a robot that
    # got there and drew badly, and only one of them is a bug -- but they
    # must BOTH end with the module on the rack.
    #
    # ⚠ THE GATE IS PER-ERRAND, not universal (`Errand.needs_use_pose`). An
    # errand that DOES ITS OWN NAVIGATION does not need this drive to have
    # arrived -- the census's `use_at` is the first point of the survey route
    # its use-phase drives itself. Gating it too cost the recorded showcase
    # mission its census answer: the drive stopped 1.96 m short and the robot,
    # which could still see the whole garden from there, was sent home.
    arrived = self.mission.drive_to(*errand.use_at, timeout=60.0)
    still = self.mission.swap.module_state(self.module)["on_fork"]
    self._say(f"USE_TOOL: {'arrived' if arrived else 'NEVER GOT THERE'}"
              f"{'' if still else ' -- BUT DROPPED THE TOOL'}")
    # What the board looked like before this errand touched it (issue #14).
    # The evaluator counts the strokes that landed HERE, so a second drawing
    # on an un-erased board is not scored on the first one's ink.
    before = scoring.board_before(self, errand)
    used: dict = {}
    if errand.use is not None and not arrived and errand.needs_use_pose:
      used = {"error": "never reached the use pose"}
    elif errand.use is not None and still:
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
    spent = max(0.0, spent_from - self.battery.energy_wh)
    estimated = self.affords(errand).cost_wh
    result = {"errand": errand.name, "module": errand.module,
              "picked": carried, "stowed": stowed,
              # Measured against what hub/energy.json said it would be. Both,
              # deliberately: the estimate alone is a claim, and the two side
              # by side are what says the table still describes the world.
              "energyWh": round(spent, 4),
              "estimateWh": round(estimated, 4),
              "energySeconds": round(float(self.data.time) - began_at, 2),
              **used}
    if estimated > 0.0 and spent > estimated * energy_model.WARN_OVER:
      # Not a failure -- the errand finished -- but the table is the only
      # thing standing between an overseer and a mid-errand death, so an
      # under-estimate is said out loud rather than left in a dict.
      self._say(f"ENERGY {errand.name} cost {spent:.3f} Wh against an "
                f"estimate of {estimated:.3f} -- hub/energy.json is low "
                f"(scripts/energy_spike.py re-measures it)")
    self._deferrals.pop(errand.name, None)
    # And the verdict, LAST: an errand is judged on the finished job, which
    # includes putting the tool back. Measured off the sim by hub/scoring.py,
    # never off `used` alone -- see sample_draw, which counts the strokes the
    # pen actually wrote into the board book.
    verdict = scoring.score_errand(self, errand, result, before)
    entry = self._bank(verdict)
    if verdict is not None:
      result["verdict"] = verdict.as_dict()
      result["points"] = entry["points"] if entry is not None else 0
      # ...and the same verdict closes the OFFER, if this errand was one
      # (issue #21). The same object, deliberately: a task graded separately
      # from the errand that discharged it is a second scorer, and there is
      # only meant to be one.
      if errand.task_id and self.tasks is not None:
        closed = self.tasks.resolve(errand.task_id, verdict,
                                    t=float(self.data.time))
        if closed is not None:
          result["task_id"] = closed.id
          self._say(f"TASK {closed.id} {closed.state}: {closed.description}")
    self.errand_results.append(result)
    return result

  # ---- the energy gate (issue #15) -----------------------------------------

  def _afford_next(self) -> bool:
    """True if the head of the queue can be started RIGHT NOW.

    False means exactly one thing to the caller -- go and charge, then ask
    again -- and everything that is not that has already been dealt with
    here: an errand no pack in this world could ever cover is dropped, and so
    is one that has been put back for a charge too many times. Leaving either
    of those on the queue would make "charge and retry" a spin, which is the
    failure mode a guard like this invites.

    Draining the queue of unrunnable errands rather than reporting on just
    the head, because the caller's next branch pops whatever is in front and
    a head this method silently disliked would be run un-gated.
    """
    while self.errands:
      errand = self.errands[0]
      fit = self.affords(errand)
      if fit.state == energy_model.OVERSPEND:
        # A demo cell being honest about itself: the job is bigger than any
        # charge this world can give it, and it is run anyway because that is
        # what a cell sized smaller than its jobs means. Said out loud, once,
        # because "the robot ran flat" and "the robot was always going to run
        # flat" are different events and only the second is world tuning.
        self._say(f"ENERGY {errand.name}: {fit.why()}")
        return True
      if fit.ok:
        return True
      if fit.state == energy_model.BEYOND:
        self._say(f"SKIP {errand.name}: {fit.why()}")
        self._drop_errand(self.errands.pop(0), fit)
        continue
      seen = self._deferrals[errand.name] = \
          self._deferrals.get(errand.name, 0) + 1
      if seen > MAX_ERRAND_DEFERRALS:
        # Charging is not what is wrong. Said differently from a `beyond`
        # drop, because it is a different fault: the pack could hold this
        # job and does not, which means the charge cycle is failing.
        self._say(f"SKIP {errand.name}: still {fit.need_wh:.2f} Wh short "
                  f"after {MAX_ERRAND_DEFERRALS} charges -- giving up on it")
        self._drop_errand(self.errands.pop(0), fit)
        continue
      self._say(f"DEFER {errand.name}: {fit.why()}")
      return False
    return True

  def _drop_errand(self, errand, fit) -> None:
    """Give up on an errand, and close the job it was discharging.

    ⚠ DROPPING A TASK ERRAND FOR `beyond` SHOULD NOT BE REACHABLE, and the
    arithmetic is why: `Task.claimable` compares the job's estimate against
    `spendable_wh` (the pack, less the margin), which is never more than
    `fundable_wh` (a charged pack, less the same margin) -- so a job this
    world could never fund cannot be claimed in the first place. Dropping one
    because charging keeps failing IS reachable, which is the other half of
    why this exists: a task left `active` on a board the robot has walked away
    from is a marker that never resolves, and that reads as a bug rather than
    as the honest failure it is.
    """
    self.errand_results.append({
      "errand": errand.name, "module": errand.module,
      "picked": False, "stowed": True, "skipped": fit.state,
      "estimateWh": round(fit.cost_wh, 4), "energyWh": 0.0,
      "error": fit.why(),
    })
    if not errand.task_id or self.tasks is None:
      return
    # Judged by the same evaluator as a finished one, on the same evidence:
    # it will find no ink, no count and nothing carried, and close the job
    # `failed`. A task closed any other way would be a second scorer.
    verdict = scoring.score_errand(self, errand, self.errand_results[-1], {})
    self._bank(verdict)
    if verdict is not None:
      closed = self.tasks.resolve(errand.task_id, verdict,
                                  t=float(self.data.time))
      if closed is not None:
        self._say(f"TASK {closed.id} {closed.state}: {closed.description}")

  # ---- the visitor channel (issue #16) -------------------------------------

  def _visitor_step(self) -> None:
    """Apply whatever needs no decision, and hold the rest for one.

    Runs at the top of every arbitration pass. RATINGS are applied here and
    not by the overseer, deliberately: a rating settles a deferred verdict,
    and letting the model anywhere near that would hand it the "declare
    victory" button the whole reward design exists to keep out of its reach
    (issue #14). The rater supplies a 0..1 quality; `hub/rewards.json` turns
    it into points; the robot is not consulted.
    """
    if self.inbox is None:
      return
    for msg in self.inbox.drain(("reset_tool",)):
      self._reset_tool(msg)
    for msg in self.inbox.drain(("rating",)):
      if self.ledger is None:
        continue
      try:
        entry = self.ledger.settle(msg.seq, msg.quality,
                                   t=float(self.data.time))
      except (KeyError, ValueError) as e:
        # A rating for an entry that is gone, already settled, or was never
        # visitor-tiered. The website is allowed to be wrong about this --
        # it is a different process holding a stale row -- so it is a
        # narration line, not a crash.
        self._say(f"VISITOR rating {msg.seq} ignored: {e}")
        continue
      self._say(f"VISITOR rated task {msg.seq} ({entry['task']}) "
                f"{msg.quality:.0%} -- {entry['points']:+d} points, "
                f"balance {entry['balance']}")

  def _reset_tool(self, msg) -> None:
    """Put a lost module back on its bay, because an admin said so (#30).

    The recovery half of the tool-drop problem: prevention is the measured
    bay standoff, but a module that IS on the floor -- knocked there by a
    collision, an unlucky jam, anything the measurement cannot promise away
    -- is invisible to the whole swap stack (every pick finds an empty bay)
    and litters the rack's approach lane. On hardware this is a person
    picking the tool up; in the sim it is the same hand, reaching in through
    the admin page.

    Handled by CODE on the physics thread, like a rating: an admin command
    is not a thing the robot weighs, so it never reaches the overseer's
    context. Refused, with a narration, while the module is electrically
    seated on the fork -- a tool in use is not lost, and yanking it out of
    the coupling mid-errand would MAKE the mess this exists to clean up.

    The reset pose is `model.qpos0`: every world compiles its modules hung
    at their own bays, so "back where it belongs" is the model's own answer
    rather than a second copy of the rack geometry.
    """
    name, who = msg.module, msg.who or "an admin"
    if not name.startswith("module_"):
      self._say(f"ADMIN reset refused: {name!r} is not a module")
      return
    try:
      body = self.model.body(name)
    except KeyError:
      self._say(f"ADMIN reset refused: no {name!r} in this world")
      return
    jid = int(body.jntadr[0])
    if jid < 0 or self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
      self._say(f"ADMIN reset refused: {name!r} is not a free module")
      return
    if module_power_contact(self.model, self.data, name):
      self._say(f"ADMIN reset refused: {name} is seated on the fork -- "
                "a tool in use is not lost")
      return
    qadr = int(self.model.jnt_qposadr[jid])
    dadr = int(self.model.jnt_dofadr[jid])
    self.data.qpos[qadr:qadr + 7] = self.model.qpos0[qadr:qadr + 7]
    self.data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(self.model, self.data)
    self._say(f"ADMIN {who} reset {name} -- back on its bay")

  def _answer_visitor(self, decision) -> None:
    """Send one accept/decline/answer back out, and retire the message.

    The message is only taken off the queue once it has actually been
    answered. A decision that came back scripted (the API was down) responds
    to nobody, so the suggestion is still there for the next decision rather
    than silently discarded by an outage.
    """
    if self.inbox is None or not decision.responds:
      return
    msg = self.inbox.take(decision.respond_to)
    if msg is None:
      return                                # already dealt with; nothing owed
    reply = {"type": "visitor_reply", "t": round(float(self.data.time), 3),
             "robot": ROBOT_ROOT, "id": msg.id, "kind": msg.kind,
             "outcome": decision.outcome, "reply": decision.reply,
             "action": decision.action if decision.outcome == "accepted"
             else ""}
    for hook in self.visitor_hooks:
      hook(dict(reply))
    self.replies.append(reply)
    # Narrated as well as sent, because the two audiences are different: the
    # typed message closes the database row the website is holding, and the
    # event line is what a person watching the stream reads.
    self._say(f"VISITOR {decision.outcome} {msg.who or 'a visitor'}'s "
              f"{msg.kind}: {decision.reply or '(no reply)'}")

  # ---- tasks (issue #21) ----------------------------------------------------

  @property
  def charged_wh(self) -> float:
    """What a full pack holds here. `CHARGED` rather than capacity, because
    the charge cycle stops there and a pack is never actually filled."""
    return self.battery.capacity_wh * CHARGED

  @property
  def reserve_margin_wh(self) -> float:
    """The energy an errand must be expected to LEAVE BEHIND (issue #15).

    Zero on a demo cell and the return-trip reserve on a hosting-sized one --
    `energy.EnergyModel.margin_wh` is the rule and its docstring is the
    argument. One number for the world, so the errand gate, `Task.claimable`
    and the producer are all doing the same arithmetic rather than three
    slightly different ones.
    """
    return self.energy.margin_wh(self.charged_wh, self.low_battery_wh)

  @property
  def spendable_wh(self) -> float:
    """What a job's cost is compared against, right now.

    THE WHOLE CHARGE LESS THE MARGIN, and on a demo cell the margin is zero,
    so this is the whole charge -- which is what it always was, and the
    distinction was worth a wrong fixture to learn. The reserve is a
    RETURN-TRIP margin: on a cell smaller than one errand it is a margin the
    robot cannot afford to keep, because one errand costs roughly one full
    pack in both demo worlds (0.487-0.570 Wh in room_hub against a 0.700 Wh
    cell, 0.866-0.929 Wh in home against 1.100 Wh) while the energy ABOVE the
    reserve is 0.28 and 0.44 Wh. Gating on that would refuse every job in
    every world forever -- a task system that silently does nothing.

    On a hosting-sized pack there IS margin to keep, the errand is required to
    finish with the return trip still in hand, and the mid-errand death this
    number exists to prevent stops being reachable.
    """
    return max(0.0, self.battery.energy_wh - self.reserve_margin_wh)

  @property
  def fundable_wh(self) -> float:
    """What a CHARGED pack can fund here -- what the WORLD can pay for, as
    opposed to what the robot can afford this second.

    This is what the producer offers against (issue #23), and the difference
    from `spendable_wh` is the difference between "this world cannot pay for
    that job" and "the robot should charge first". Only the first is a reason
    not to put a job up; the second is what `Task.claimable` says about an
    offer that is already standing, and it says it every time anybody asks.
    """
    return max(0.0, self.charged_wh - self.reserve_margin_wh)

  def affords(self, errand) -> energy_model.Affordability:
    """Can this errand be started now, later, or not at all (issue #15).

    The whole point of the module is in the FOUR answers rather than two.
    `needs_charge` is checked between errands and never inside one, so an
    errand the pack cannot cover is a robot that dies holding the tool -- but
    "not now" wants a charge and a retry while "not ever here" wants the
    errand dropped, and answering both with False is how a charge/defer spin
    gets written.

    An errand that carries its own `estimate_wh` is priced by that: a task's
    figure is per KIND and knows which end of the house it is being asked
    about, which a per-action table cannot (CLAUDE.md, "the far board costs
    more than the near one").
    """
    return self.energy.afford(
      errand.task or errand.name, energy_wh=self.battery.energy_wh,
      charged_wh=self.charged_wh, reserve_wh=self.low_battery_wh,
      cost_wh=errand.estimate_wh or None,
      # WHICH board/zone/module, so a per-target row can win. `detail` is
      # where an errand already records what it went to; the name would work
      # for the drawing errands and not for the census.
      target=str(errand.detail.get("board") or errand.detail.get("zone") or
                 errand.module if errand.detail else ""))

  def _task_step(self) -> None:
    """Put up whatever is due and lapse whatever nobody got to.

    ⚠ THIS HANGS OFF THE PHYSICS SEAM, not off the arbitration loop, and that
    is issue #23's whole difference from #21. A mission pass happens between
    errands, so a producer ticked there could only offer work while the robot
    was standing still -- and an offer would only be seen to lapse minutes
    after it did, on whichever pass happened next. The world puts work up and
    takes it down on its own clock; the loop reacts to the board when it next
    looks at it. Throttled to `CHECK_S`, because sweeping forty tasks at
    500 Hz is Python spent to learn nothing.

    It is deliberately incapable of doing anything the robot does: it offers
    and it expires. Nothing here touches `state`, `errands` or the battery,
    which is what makes "a task never delays a charge" a property of the
    seam rather than of the numbers in hub/cadence.json.
    """
    if self.tasks is None or self.data.time < self._next_task_check:
      return
    self._next_task_check = float(self.data.time) + CHECK_S
    for task in self.tasks.expire_due(float(self.data.time)):
      self._say(f"TASK {task.id} expired: {task.description}")
    if self.producer is not None:
      for task in self.producer.tick(float(self.data.time), self.fundable_wh):
        self._say(f"TASK {task.id} offered: {task.description}")

  def _claim_task(self, task_id: str, answer: str = "") -> bool:
    """Take one offered job on and queue the errand that discharges it.

    False for every ordinary way this can not happen -- the offer is gone,
    somebody took it, it lapsed, it costs more than the pack has left, or
    this world cannot build an errand for it. False rather than an exception
    because the caller is a mission loop acting on an LLM's answer, and "that
    one is not available" is an answer.

    ⚠ The energy gate lives HERE and not in the errand: a task that cannot be
    afforded must not be claimable in the first place, because the reserve is
    only checked BETWEEN errands and a robot that starts a job it cannot
    finish dies holding the tool (CLAUDE.md, "A chosen errand can cost more
    than the whole pack").

    `answer` is what the MIND says the answer is, for a job that asks a
    question (issue #22). It is frozen into the task by `TaskBoard.claim` and
    read back out by the evaluator; the errand is handed the glyphs and never
    told what the question was, so nothing on the way to the board can revise
    what the robot committed to.
    """
    if self.tasks is None:
      return False
    now = float(self.data.time)
    task = self.tasks.get(task_id)
    if task is None or not task.claimable(now, self.spendable_wh):
      self._say(f"TASK {task_id}: not available")
      return False
    said = clean_answer(answer) if task.needs_answer else ""
    if task.needs_answer and not said:
      # Not a fault and not a failure of the task: a question is a job for a
      # mind, and the scripted rotation is not one. Left offered, so it
      # lapses honestly rather than being marked failed by a robot that never
      # touched it.
      self._say(f"TASK {task.id}: asks a question and nobody answered it")
      return False
    errand = errand_for_task(task, self.world, self.boards, answer=said)
    if errand is None:
      # Offered in a world that cannot build it. Not fatal and not a claim:
      # leaving it offered lets it lapse honestly rather than be marked
      # failed by a robot that never touched it.
      self._say(f"TASK {task.id}: nothing to build for {task.kind!r} here")
      return False
    if self.tasks.claim(task.id, t=now, pack_wh=self.spendable_wh,
                        answer=said) is None:
      return False
    self.claimed.append(task.id)
    self._say(f"TASK {task.id} claimed: {task.description}"
              + (f" -- answering {said}" if said else ""))
    # Queued rather than run inline, exactly as an overseer's chosen errand
    # is: if taking it dropped the battery below the reserve, the next pass
    # through the loop charges first.
    self.errands.append(errand)
    return True

  def _claim_next_task(self) -> bool:
    """Take the oldest claimable job, if there is one.

    This is the branch that makes tasks work WITHOUT an overseer. A world
    that only hands out work to a robot with an LLM attached would have no
    tasks in either committed recording and none on the deployed sim, which
    is off by default -- and a job offer nobody can accept is scenery.
    """
    if self.tasks is None:
      return False
    for task in self.tasks.claimable(float(self.data.time), self.spendable_wh):
      # A question is skipped rather than attempted (issue #22): there is
      # nobody here to work the answer out, and the two ways code could
      # supply one -- reading it out of the bank, or guessing -- are the sim
      # marking its own homework and a confident wrong number on a wall.
      if task.needs_answer:
        continue
      if self._claim_task(task.id):
        return True
    return False

  # ---- the one branch an LLM may replace (issue #15) ------------------------

  def _decide(self) -> None:
    """Ask the overseer what to do next, and do it.

    Reached ONLY when the battery is fine and the errand queue is empty --
    `run()` checks `needs_charge` first and always will. The overseer cannot
    reach this method's caller and has no action that suppresses charging.

    The `while pending` loop is the load-bearing line: it STEPS THE SIM while
    the API call is in flight, so the world keeps running and the telemetry
    stream keeps flowing during the pause. Blocking here instead would freeze
    every viewer for the length of an HTTP request -- and the pacer would then
    try to catch the missed sim time up in a burst, which is worse than the
    pause it was avoiding.
    """
    self.state = "DECIDE"
    state = overseer_context(self)
    self.overseer.start(state)
    while self.overseer.pending:
      self.mission._drive(THINK_SLICE_S, 0.0, 0.0)
    decision = self.overseer.result(state)
    self.decisions.append(decision.as_dict())
    self._say(f"DECIDE {decision.summary()}")
    # A note is written whatever the action was: "I chose X because Y" is
    # worth remembering regardless of what X turned out to be, and the model
    # may attach one to any decision.
    if decision.note and self.journal is not None:
      entry = self.journal.note(decision.note, t=float(self.data.time),
                                why=decision.reason)
      if entry is not None:
        self._say(f"JOURNAL {entry['text']}")
    # ...and the answer to whoever asked, if it answered anyone (issue #16).
    # Before the action runs, so a visitor whose suggestion was taken hears
    # so at the moment it is taken rather than five minutes later.
    self._answer_visitor(decision)

    if decision.action == "take_task":
      # The overseer accepting a job somebody offered (issue #21). Note where
      # this sits: after `needs_charge`, like every other action, and behind
      # the same claimability gate the scripted path uses -- an LLM cannot
      # take on a task the energy budget refuses.
      if not self._claim_task(decision.task, decision.answer):
        self.mission._drive(DECIDED_IDLE_S, 0.0, 0.0)
      return
    if decision.action == "charge":
      # Topping up EARLY is a real choice and this honours it. Note what it is
      # not: there is no action that declines to charge, because `needs_charge`
      # was already checked before this method was ever called.
      if self.battery.fraction >= TOP_UP_BELOW:
        # ...but "top up" has to mean there is something to top up. See
        # TOP_UP_BELOW: charging is a scored task, so an unconditional trip to
        # the rack is a points farm rather than a decision.
        self._say(f"DECIDE: already at {self.battery.fraction:.0%}, "
                  "not worth a trip to the rack")
        self.mission._drive(DECIDED_IDLE_S, 0.0, 0.0)
        return
      self.state = "GO_CHARGE"
      if self.go_charge():
        self.state = "CHARGE"
        self.charge()
      return
    if decision.action == "explore":
      self.state = "EXPLORE"
      if decision.zone:
        wx, wy = zone_centre(self.world, decision.zone)
        self._say(f"EXPLORE: heading for {decision.zone}")
        self.mission.drive_to(wx, wy, timeout=60.0)
      self.explore(budget=DECIDED_EXPLORE_S, mark_done=False)
      return
    if decision.action in ("idle", "journal"):
      self.mission._drive(DECIDED_IDLE_S, 0.0, 0.0)
      return
    errand = errand_from(decision, self.world, self.boards)
    if errand is None:
      # Vocabulary and world agreed on an action nothing can build. Not an
      # exception: the loop's next pass asks again, and the overseer's
      # consecutive-idle cap stops that becoming a spin.
      self._say(f"DECIDE: nothing to build for {decision.action!r}")
      self.mission._drive(DECIDED_IDLE_S, 0.0, 0.0)
      return
    fit = self.affords(errand)
    if fit.state == energy_model.BEYOND:
      # Chosen, buildable, and bigger than any charge this world can give it
      # (issue #15). Refused HERE rather than queued and dropped a moment
      # later, because the difference matters to whoever is watching: the
      # robot said no to its own idea, and the next decision is made knowing
      # that. The model is told which actions it can afford in the same
      # breath -- `affordableActions` in the context -- so this is a backstop
      # for a decision made against a stale reading, not the primary path.
      self._say(f"DECIDE: {fit.why()}")
      self.mission._drive(DECIDED_IDLE_S, 0.0, 0.0)
      return
    # Queued rather than run inline, so the errand goes through the SAME
    # arbitration the scripted queue does -- if the decision itself dropped
    # the battery below the reserve, the next pass charges first. A
    # `charge_first` errand is queued for exactly that reason: the loop's
    # energy gate turns it into a charge and then this errand.
    self.errands.append(errand)

  # ---- the loop ------------------------------------------------------------

  def run(self, start: tuple[float, float, float],
          station_y: float = HUB_STATION_YS[0],
          use_at: tuple[float, float] = (-1.2, 2.5),
          max_sim_time: float = 600.0,
          explore_budget: float = 90.0) -> dict:
    self.max_sim_time = max_sim_time
    self.blacklist: set = set()
    self.map_done = False
    self.stranded = False
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
        # Whatever visitors sent that needs no decision (issue #16). First,
        # so a rating lands on the ledger before the next frame carries the
        # balance -- and outside the priority order, because applying a
        # rating is bookkeeping rather than something the robot does.
        self._visitor_step()
        # ...and whatever the world put up or took down while the robot was
        # busy (issues #21, #23). Outside the priority order, because a job
        # appearing or lapsing is something that happens TO the world rather
        # than a thing the robot chose -- and mostly a no-op here, since the
        # same sweep runs on the physics seam. Kept so a lifecycle driven
        # without `mission` stepping still keeps its board honest.
        self._task_step()
        if self.needs_charge:
          self.state = "GO_CHARGE"
          if not self.go_charge():
            self.stranded = True
            break
          self.state = "CHARGE"
          self.charge()
        elif self.errands and not self._afford_next():
          # ⚠ AN ERRAND THAT WILL NOT FIT IS CHARGED FOR FIRST (issue #15).
          # `needs_charge` above is checked BETWEEN errands and never inside
          # one, so a job bigger than what is left in the pack cannot be
          # survived by any charging policy -- the robot leaves the rack,
          # works, and dies holding the tool. This is the one place that can
          # see it coming. `_afford_next` has already narrated why and, for
          # an errand no pack in this world could cover, has already dropped
          # it -- so False here always means "go and charge".
          self.state = "GO_CHARGE"
          if not self.go_charge():
            self.stranded = True
            break
          self.state = "CHARGE"
          self.charge()
        elif self.errands:
          # Pop BEFORE running: an errand that raises must not be retried
          # forever, and a queue that only shortens on success is an infinite
          # loop dressed as a task list.
          self.run_errand(self.errands.pop(0))
        elif self.overseer is not None:
          # THE ONE BRANCH THE LLM REPLACES (issue #15). Note where it sits:
          # after charging, which it cannot reach, and after the errand queue,
          # so an explicit order still outranks a chosen one.
          self._decide()
        elif self._claim_next_task():
          # An offered job, taken by the loop itself (issue #21). Unreachable
          # with an overseer, which is correct: a robot with a mind chooses
          # for itself and `take_task` is one of the things it may choose.
          # Without one, work still gets done -- and it outranks exploring for
          # the reason the errand queue does, because somebody asked for it.
          continue
        elif not self.map_done:
          self.state = "EXPLORE"
          self.explore()
        elif self.producer is not None:
          # WAITING FOR WORK IS NOT BEING FINISHED (issue #23). The loop used
          # to break the moment it had nothing to do, which was right when
          # the only work was a preset queue -- that queue never grows. A
          # world with a PRODUCER in it does: measured, a home run mapped the
          # house, did both the jobs it could reach and ended at t=410 with
          # the next offer due at t=480. Seventy seconds early, and it called
          # that a completed mission.
          #
          # Standing by rather than ending, in bounded slices so the loop
          # keeps re-checking `needs_charge` -- and deliberately without a
          # state of its own: `State` is a two-repo vocabulary the website
          # draws off, and "the robot paused" is what `idle` already looks
          # like from the outside. `max_sim_time` is still the thing that
          # ends the day.
          self.mission._drive(WAIT_FOR_WORK_S, 0.0, 0.0)
        else:
          break

      self.state = "DONE"
      # A robot that could not reach its charger has not completed anything
      # (issue #32): the old line said "mission complete" here because the
      # battery was not yet empty, which dressed the day's actual ending --
      # a failed dock -- as success.
      if self.stranded:
        self._say("GO_CHARGE FAILED -- mission over, stranded off the dock "
                  f"at {self.battery.fraction:.0%}")
      else:
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
      # A failed dock ended this run (issue #32) -- distinct from "aborted"
      # (the viewer closing is a request to stop, not a failure) and visible
      # here so a watcher can tell "finished the day" from "never got home".
      "stranded": self.stranded,
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
      # Every verdict this mission earned, in order, and what the robot is
      # worth afterwards (issue #14). The verdicts are here even without a
      # ledger -- a test or a spike wants the evaluation without a state file.
      "verdicts": list(self.verdicts),
      "points": self.ledger.balance() if self.ledger is not None else 0,
      "earned": sum(v["points"] for v in self.verdicts),
      "rack_discovered": self.mission.rack_discovered,
      "collision_steps": self.mission.collision_steps,
      "sim_time": float(self.data.time),
      # What the overseer chose and what it cost (issue #15). Empty without
      # one, so every existing caller's dict is unchanged in every value it
      # already read.
      "decisions": list(self.decisions),
      "overseer": self.overseer.stats() if self.overseer is not None else {},
      "journal": (self.journal.recent() if self.journal is not None else []),
      # What visitors said and what the robot said back (issue #16). Empty
      # without an inbox, which is every caller that does not serve.
      "visitors": self.inbox.stats() if self.inbox is not None else {},
      "replies": list(self.replies),
      # The jobs this world offered and what became of them (issue #21).
      # Empty without a task board, which is every caller that does not ask
      # for one.
      "tasks": self.tasks.snapshot() if self.tasks is not None else {},
      "task_stats": self.tasks.stats() if self.tasks is not None else {},
      "tasks_claimed": list(self.claimed),
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


def points_ledger(state: str | None = None, table=None):
  """The robots' points ledger (issue #14).

  `state` is a JSON file the balances and the earnings log live in ACROSS
  runs -- the same treatment the boards get, and for the same reason: points
  are world state, and every mission end is a restart. Without one the ledger
  is per-run, which is what tests and one-off demos want.
  """
  from pluggybot.hub.ledger import Ledger
  return Ledger(path=state, table=table)


def task_board(state: str | None = None, table=None, cadence=None,
               world: str = ""):
  """The world's job offers as persistent state (issue #21).

  `state` is a JSON file the tasks live in ACROSS runs -- the same treatment
  the boards and the ledger get, and for the same reason: an offer is world
  state, and every mission end is a restart. A task that vanished because the
  container cycled would be a job somebody asked for and nobody ever declined.

  The two CAPS come from `cadence` (issue #23) rather than from `hub/tasks.py`
  defaults, so how much work may stand at once is configuration like the rest
  of the timing policy. Without one the board keeps its own conservative
  defaults, which is what a unit test wants.
  """
  from pluggybot.hub.tasks import TaskBoard
  # ...and what a job COSTS here (issue #15), for the same reason: a board
  # that priced every world's `carry` the same either under-prices the big
  # floor plan or refuses the small one work it does perfectly well.
  costs = energy_model.load(world) if world else None
  if cadence is None:
    return TaskBoard(path=state, table=table, energy=costs)
  return TaskBoard(path=state, table=table, max_tasks=cadence.max_tasks,
                   max_offered=cadence.max_offered, energy=costs)


def world_targets(world: str, book=None) -> dict:
  """What this world has for a task to be ABOUT, by `TaskKind.target_kind`.

  The seam that keeps `hub/cadence.py` from knowing what a world is: the
  producer is handed the furniture and rotates over it, and a kind whose
  target_kind is missing here is simply not offered. Read off the world's own
  config and the boards' own names, never hardcoded -- a world without
  whiteboards gets fewer jobs rather than an offer nothing can build.
  """
  cfg = world_config(world)
  targets: dict[str, list[str]] = {}
  if book is not None and len(book):
    targets["board"] = list(book.names)
  if cfg.get("census_zone"):
    targets["zone"] = [cfg["census_zone"]["name"]]
  # One module, and deliberately the one every world has on its rack: the
  # carry is the job a bare room can still offer, and it is where the swap
  # stack gets exercised on its own.
  targets["module"] = ["module_lcd"]
  return targets


def task_producer(board, world: str, book=None, cadence=None):
  """The thing that keeps putting work into a world (issue #23).

  Replaces the `seed_tasks` placeholder. That one put up a starter set once
  and nothing ever added another, so a robot that worked through the board
  spent the rest of a multi-hour run with nothing asked of it -- and the
  three numbers behind it (how long an offer stands, how many, how often)
  were constants in this module rather than something a deploy could re-tune.
  Both halves are what issue #23 is.

  ⚠ `TaskProducer.seed` is still a separate call, and it must be made AFTER
  every hook is attached: `TaskBoard.offer` emits a `task_offered` the moment
  it is called.
  """
  from pluggybot.hub.cadence import TaskProducer, default_cadence
  return TaskProducer(board, cadence or default_cadence(world),
                      world_targets(world, book))


def world_screens(model, data):
  """Every display module in this world, as one telemetry-shaped set (#13).

  What counts as a display is decided in ONE place -- the `_screen` geom
  suffix `telemetry.scene.screen_map` keys the scene's `screens` block off --
  so the panel the website paints and the panel the sim drives can never be
  two different lists.
  """
  from pluggybot.hub.screen import Screen, ScreenSet
  from pluggybot.telemetry.scene import screen_map
  return ScreenSet([Screen(model, data, module=body)
                    for body in screen_map(model)])


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
  if kind == "dance":
    return [dance_errand(cfg["use_at"])]
  if kind == "census":
    zone = cfg.get("census_zone")
    if zone is None:
      raise ValueError(f"the {world} world has no zone to take a census of")
    return [census_errand(Zone.from_meta(zone), entry=cfg.get("census_entry"))]
  if kind == "showcase":
    # What the SITE serves (rooftop-media-2026 #28): one errand that leaves
    # ink on a board and one that puts a face on the screen, so a single
    # recording exercises both streamed surfaces. The battery arbitration
    # puts a charge between them without being asked -- that is the loop
    # doing its job, not a scripted interlude.
    return errands_for("draw", world, book) + errands_for("census", world)
  if kind == "artwork":
    # The visitor-judged tier (issue #14, made reachable by #16): the same
    # drawing errand, scored as `artwork`. Code confirms ink landed and banks
    # ZERO; the points arrive later, when somebody rates it over the inbound
    # channel. Kept as its own queue name so the deferred path can be flown
    # on demand rather than only when an overseer happens to pick it.
    if book is None or not len(book) or not cfg["meta"]:
      raise ValueError(f"the {world} world has no whiteboards to draw on")
    meta = json.loads(Path(cfg["meta"]).read_text())
    return [draw_errand_for(world, book, next(iter(meta["boards"])),
                            program_name="robot", task="artwork")]
  if kind in ("draw", "draw2"):
    if book is None or not len(book):
      raise ValueError(f"the {world} world has no whiteboards to draw on")
    meta = json.loads(Path(cfg["meta"]).read_text())
    # Two boards, two different figures: the acceptance test for issue #12 is
    # "two drawings on two boards with charging in between", and drawing the
    # same figure twice would not catch a board id threaded through by
    # accident.
    names = list(meta["boards"])[:2 if kind == "draw2" else 1]
    figures = ("house", "tree", "sun", "robot")
    return [draw_errand_for(world, book, name,
                            program_name=figures[i % len(figures)])
            for i, name in enumerate(names)]
  raise ValueError(f"unknown errand queue {kind!r} "
                   "(carry, draw, draw2, census, dance, showcase or none)")


def draw_errand_for(world: str, book, board_name: str,
                    program_name: str = "house", task: str = "draw",
                    program=None):
  """One drawing errand on a NAMED board with a NAMED figure.

  Split out of `errands_for` so the preset queue and the overseer's chosen
  drawing (issue #15) build the errand through the same code -- a chosen
  drawing that took a different path would be a second drawing stack, which is
  the exact thing issue #12 spent itself removing.
  """
  cfg = world_config(world)
  if book is None or not len(book) or not cfg["meta"]:
    raise ValueError(f"the {world} world has no whiteboards to draw on")
  meta = json.loads(Path(cfg["meta"]).read_text())
  if board_name not in meta["boards"]:
    raise ValueError(f"{world} has no board {board_name!r} "
                     f"(have: {', '.join(meta['boards'])})")
  from pluggybot.hub.drawing import Board
  return drawing_errand(book, board_name,
                        Board.from_meta(meta["boards"][board_name]),
                        program=program, program_name=program_name, task=task)


# ---- the overseer's seams (issue #15) ---------------------------------------


def errand_from(decision, world: str, book=None):
  """An overseer decision -> an errand, or None if this world cannot build it.

  None rather than an exception: a decision is untrusted input in exactly the
  way a visitor message will be (issue #16), and the mission loop's response
  to "I cannot do that" should be to ask again, not to end.
  """
  try:
    if decision.action in ("draw", "artwork"):
      # Same errand, different TIER. `artwork` is the visitor-judged slot
      # (issue #14): code confirms ink landed and banks zero, and the points
      # arrive later when somebody rates it over the inbound channel. That is
      # what makes `rating` a real path rather than a reserved word.
      return draw_errand_for(world, book, decision.board,
                             program_name=decision.program or "house",
                             task=decision.action)
    if decision.action in ("census", "dance", "carry"):
      return errands_for(decision.action, world, book)[0]
  except (ValueError, KeyError, IndexError):
    return None
  return None


def errand_for_task(task, world: str, book=None, answer: str = ""):
  """A claimed TASK -> the errand that discharges it, or None (issue #21).

  The sibling of `errand_from` and deliberately the same shape: a task is
  untrusted input in exactly the way an overseer decision is (it can come
  from a visitor), so a world that cannot build one answers None and the loop
  leaves the offer alone rather than ending.

  Note what is threaded through and what is not. The errand carries
  `task_id`, so the verdict that pays for the finished job also closes the
  offer -- one evaluation, two consumers. It does NOT carry the task's
  `secret`, and `whiteboard_answer` (issue #22) is what makes that load-
  bearing rather than merely tidy: the errand is handed the GLYPHS of the
  answer the mind committed to and is never told the question or the right
  answer. A use-phase is arbitrary caller code, so the less of the task it
  can see, the less there is for it to be wrong about -- and scoring reads
  the board and the frozen commitment instead.
  """
  from pluggybot.hub.tasks import KINDS
  spec = KINDS.get(task.kind)
  if spec is None:
    return None
  try:
    if task.kind == "whiteboard_answer":
      # A drawing errand like any other; only the figure is different. The
      # `answer` program is the one door text has into the plotter, and what
      # goes through it has already been through `questions.clean_answer`.
      errand = draw_errand_for(world, book, task.target, task="answer",
                               program=strokes.program(
                                 "answer", text=answer or task.answer))
    elif task.kind in ("draw_figure", "rate_artwork"):
      errand = draw_errand_for(world, book, task.target,
                               program_name=task.params.get("program")
                               or "house", task=spec.task)
    elif task.kind == "count_plants":
      cfg = world_config(world)
      zone = cfg.get("census_zone")
      if zone is None or zone["name"] != task.target:
        return None
      errand = census_errand(Zone.from_meta(zone), entry=cfg.get("census_entry"))
    elif task.kind == "fetch_module":
      errand = carry_errand(module=task.target,
                            use_at=world_config(world)["use_at"])
    else:
      return None
  except (ValueError, KeyError, IndexError):
    return None
  errand.task_id = task.id
  # The job's own energy figure travels with the errand (issue #15). Per
  # KIND rather than per action, which is strictly better information: it
  # knows which whiteboard was asked for, and the far one costs more than
  # the near one. The gate that refuses an unaffordable errand then agrees
  # by construction with the gate that refused to claim it.
  errand.estimate_wh = float(task.estimate_wh)
  return errand


def zone_centre(world: str, name: str) -> tuple[float, float]:
  """The middle of a named zone, for a `explore(zone)` decision."""
  for zone in world_config(world)["zones"]:
    if zone["name"] == name:
      return ((zone["min"][0] + zone["max"][0]) / 2.0,
              (zone["min"][1] + zone["max"][1]) / 2.0)
  raise ValueError(f"{world} has no zone {name!r}")


def overseer_context(life) -> dict:
  """The volatile half of the overseer's prompt, plus the decision counter
  the scripted fallback rotates on.

  Visitor messages are PEEKED, not taken: a decision can fail, come back
  scripted, or answer only one of several, and a suggestion is retired when it
  has been answered rather than when it has been read (issue #16).
  """
  from pluggybot.hub import overseer as ov
  visitors = life.inbox.peek(VISITORS_SHOWN) if life.inbox is not None else ()
  # The offers on the board, framed the way `TaskReward.as_context` frames a
  # payout: what the job is, what it pays, and whether it can be taken RIGHT
  # NOW given what is left in the pack. The claimability flag is computed
  # here rather than left to the model, because "can I afford this" is an
  # arithmetic question with a right answer and nothing is gained by asking
  # an LLM to do it (issue #21).
  offers = (life.tasks.context(float(life.data.time), life.spendable_wh,
                               limit=TASKS_SHOWN)
            if life.tasks is not None else [])
  # What the pack can pay for now, and what this world could ever do
  # (issue #15). TWO lists, because they are answers to different questions:
  # an errand the robot cannot afford this second is one the loop charges for
  # and then runs -- an ordinary plan, and one the model should be able to
  # make -- while an errand no charge here would cover is a dead end. The
  # decision is only ever filtered on the second.
  def priced(action: str, have: float):
    if action not in ov.ERRAND_ACTIONS:
      return True                       # free, bounded, or priced per offer
    return life.energy.afford(action, energy_wh=have,
                              charged_wh=life.charged_wh,
                              reserve_wh=life.low_battery_wh).ok

  menu = life.overseer.menu.available()
  affordable = [a for a in menu if priced(a, life.battery.energy_wh)]
  possible = [a for a in menu if priced(a, life.charged_wh)]
  state = ov.context_for(life, life.journal, visitors=visitors, tasks=offers,
                         affordable=affordable, possible=possible)
  state["decisions"] = len(life.overseer.decisions) if life.overseer else 0
  return state


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
      "hosting_battery_wh": home.HOME_HOSTING_CAPACITY_WH,
      "low_battery_wh": home.HOME_LOW_BATTERY_WH,
      "explore_budget": 240.0,
      "activities": home_activities,
      # The census's zone and the doorway it is entered by (issue #13). Off
      # the generator's own ZONES, so the rectangle the robot surveys is the
      # rectangle the website draws and the one the evaluator scores against.
      "census_zone": next(z for z in home.ZONES if z["kind"] == "garden"),
      "census_entry": (home.GARDEN_X[0] + 0.4,
                       sum(home.DOOR_GARDEN_Y) / 2),
      # Every named region, for an overseer's `explore(zone)` (issue #15).
      # Off the generator's own ZONES, like the census zone above -- the
      # region the LLM can name is the region the website draws.
      "zones": [dict(z) for z in home.ZONES],
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
      "hosting_battery_wh": HOSTING_CAPACITY_WH,
      "low_battery_wh": LOW_BATTERY_WH,
      "explore_budget": 90.0,
      "activities": None,      # room_hub has no activities yet
      "census_zone": None,     # ...and nothing countable to survey
      "census_entry": None,
      "zones": [],             # ...and one undivided room, so nothing to name
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
             errand: str = "carry", board_state: str | None = None,
             ledger_state: str | None = None,
             overseer: bool | None = None, goals: str | None = None,
             journal_state: str | None = None,
             tasks: bool = False, tasks_state: str | None = None,
             pack: str = "demo", reserve_wh: float | None = None) -> dict:
  """Run a whole mission. `errand` names a queue off the menu (errands_for).

  Callers that want to hand in errands they built themselves -- the overseer,
  once issue #15 has it choosing rather than picking a preset -- should build
  the HubLifecycle directly, as scripts/home_draw.py does. The book has to
  travel WITH them: a drawing errand closes over the book it was built
  against, so a second one opened here would have the robot drawing into one
  book while telemetry reported the other.
  """
  cfg = world_config(world)
  # Which CELL this run flies on (issue #15). `demo` flattens in minutes,
  # which is what every mission test and both committed recordings were made
  # against; `hosting` is the pack a watched world runs on, where one charge
  # buys hours of work and hub/energy.py's return-trip margin becomes real.
  # `--battery-wh` still overrides either.
  if pack not in ("demo", "hosting"):
    raise ValueError(f"unknown pack {pack!r} (demo or hosting)")
  default_wh = cfg["battery_wh"] if pack == "demo" else cfg["hosting_battery_wh"]
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  book = board_book(world, state=board_state)
  # Displays are a WORLD's, like activities and boards; the lifecycle drives
  # the first one (there is one LCD) and telemetry streams the set.
  screens = world_screens(model, data)
  # The ledger is the ROBOTS', not the world's -- it is the one piece of
  # persistent state that follows them between rooms.
  ledger = points_ledger(ledger_state)
  # Job offers (issue #21). OFF by default and staying that way: a task board
  # adds errands to a mission, which reshuffles the whole trajectory, and
  # every existing demo and mission test has to behave exactly as it did
  # unless somebody asks for tasks by name. A state path implies "yes".
  # ...and how often the world puts work up, how long it stands and how much
  # may stand at once (issue #23): hub/cadence.json, per world.
  from pluggybot.hub.cadence import default_cadence
  beat = default_cadence(world) if (tasks or tasks_state) else None
  board = (task_board(tasks_state, cadence=beat, world=world)
           if (tasks or tasks_state) else None)
  maker = task_producer(board, world, book, beat) if board is not None else None
  # The overseer chooses what to do once the queue below is empty (issue #15);
  # `None` reads $PLUGGY_OVERSEER, and off is the default everywhere.
  from pluggybot.hub import overseer as ov
  boss, journal = ov.build(world, book, enabled=overseer, goals_path=goals,
                           journal_path=journal_state)
  # Read for the STREAM whether or not an overseer reads it for decisions
  # (0.8.0): the goals panel on the site shows what the robot is for, and a
  # scripted rotation has a purpose too. `steering` is what keeps that
  # honest -- see FrameBuilder.goals_message.
  goals_prose = ov.goals_text(goals)
  life = HubLifecycle(model, data, viewer=viewer, realtime=realtime,
                      battery_wh=battery_wh or default_wh,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=(reserve_wh if reserve_wh is not None
                                      else cfg["low_battery_wh"]),
                      boards=book,
                      screen=next(iter(screens), None), ledger=ledger,
                      overseer=boss, journal=journal, world=world,
                      errands=errands_for(errand, world, book), tasks=board,
                      producer=maker)
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
                                 activities=activities, boards=book,
                                 screens=screens, ledger=ledger, tasks=board,
                                 goals=goals_prose,
                                 steering=boss is not None,
                                 # The occupancy map is a BELIEF, and a
                                 # recording that omits it replays a robot
                                 # that never had one -- which is what the
                                 # website's map panel was reading until
                                 # rooftop-media-2026 #78.
                                 grid=life.mission.grid)
    life.mission.step_hooks.append(recorder.step_hook)
    # Strokes and erasures are EVENTS, not poses: ink is not a body, so a
    # recording without these lines replays a robot miming at a blank wall.
    if book is not None:
      book.on_event.append(recorder.emit)
    # ...and so is an award: points are not a pose either (issue #14).
    ledger.on_event.append(recorder.emit)
    # ...and so is a job being offered, taken or resolved (issue #21). The
    # `tasks` block catches a late joiner up; these lines are the MOMENTS,
    # which is what the site animates a marker on.
    if board is not None:
      board.on_event.append(recorder.emit)
  # ⚠ SEEDED LAST, after every hook is attached. `TaskBoard.offer` emits a
  # `task_offered` the moment it is called, so seeding at construction time
  # put the offers on the floor before the recorder existed -- a recording
  # whose tasks block was populated and whose offer events were missing. The
  # `board_snapshot` lesson, arriving through a different door. Seeded only
  # when nothing is already outstanding, so a restart against a persisted
  # board resumes the jobs it left rather than re-offering them all.
  if maker is not None and not board.open_tasks():
    maker.seed(pack_wh=life.fundable_wh)
  try:
    return life.run(start or cfg["start"], use_at=cfg["use_at"],
                    max_sim_time=max_sim_time,
                    explore_budget=explore_budget or cfg["explore_budget"])
  finally:
    if recorder is not None:
      recorder.close()
    if viewer is not None:
      viewer.close()
