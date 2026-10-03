"""The mission loop: explore, charge, run the errands and the mind's choices.

The loop that names the project -- explore until the battery runs low, go
to the dock, charge, resume -- with an errand queue and a mind:

  EXPLORE ---- battery low ----> GO_CHARGE --> CHARGE --+
     ^  frontier-walk the map       walk to the  lie on  |
     |                              dock         the pins|
     |                                                  |
     +--------- recharged, errand pending --------------+
                              |
                              v
                    SWAP_PICK -> (use the tool) -> SWAP_RETURN -> DONE

Two things were measured once and hold for any body: the pack drains
against the body's real actuator effort (`Body.pack`), and charging is
confirmed by an ELECTRICAL criterion rather than by position -- the dock's
pins conducting (`legs.dock.dock_charge_contact`). The body is reached only
through `Body` (issue #380).
"""

import dataclasses
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Literal

import mujoco
import numpy as np

from pluggybot import continuation
from pluggybot.behavior.navigation import STRIKES_TO_FINISH
from pluggybot.rack.coupling import (
  BUILT_RACK_BODY, BUILT_STATION_YS, HUB_STATION_YS, STATION_YS,
  bay_switches, built_bay_index, is_built_bay,
)
from pluggybot.mission.errand import programmed_errand
from pluggybot.body import (  # noqa: F401 -- the topple's old home
  TOPPLE_HOLD_S, TOPPLE_TILT_RAD, Body, KeepClear, body_for,
)
from pluggybot.body import RackPose
from pluggybot.economy.cadence import CHECK_S
from pluggybot.economy import energy as energy_model
from pluggybot.mind import events as ev
from pluggybot.mind import look as eye_mod
from pluggybot.mind import tickets as tickets_desk
from pluggybot.mind import wiki
from pluggybot.mind import text as text_registry
from pluggybot.mind.tickets import Desk, DeskRefused
from pluggybot.mind.mode import ModeSwitch, open_switch
from pluggybot.mind.spend import open_book
from pluggybot.mind.overseer import (
  CALLS_PER_HOUR, HEART_PRICE, HEART_RESERVE_HOURS, MAX_LOOK_RUN,
  MAX_RECALL_RUN, PROCEDURE_NEW, PROCEDURE_PREFIX, RECALL_S, THINK_SLICE_S,
  order_runnable,
)
from pluggybot.tools.screen import face_for
from pluggybot.mind.thoughts import (
  RECALLED_CHAIN_CHARS, ThoughtFiles, ThoughtRefused, attempted,
)
from pluggybot.economy import questions, scoring
from pluggybot.tools import strokes
from pluggybot.perception import depth as nf
from pluggybot.perception.heightmap import HeightMap
from pluggybot.perception.lidar import robot_geoms
from pluggybot.power import MODULE_IDLE_W, charge_scale_from_env
from pluggybot.telemetry.protocol import (
  DEATH_CAUSES, HEART_BOUGHT, HEART_REFUSED, robot_display_name,
  robot_roots,
)
from pluggybot.telemetry.recorder import TelemetryRecorder, mode_message
from pluggybot.procedure.steps import Program, compile_program
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.tick import MissionAborted, Routine

State = Literal["EXPLORE", "GO_CHARGE", "CHARGE", "DECIDE", "RECALL", "LOOK",
                "SWAP_PICK", "USE_TOOL", "SWAP_RETURN", "DEAD", "DONE"]
#: How often a robot standing still for its picture (issue #275) checks
#: the inbox for it, in sim seconds: the slice its stand-still is cut
#: into. Short enough that a picture is taken up within a fifth of a
#: second of landing, long enough that the check is not the cost.
LOOK_SLICE_S = 0.2

#: How often the death seam looks (sim seconds); it is on every physics
#: step and a quaternion-to-tilt every 2 ms would be the cost, not the check.
DEATH_CHECK_S = 0.1

#: HOW LONG THE MIND MAY GO UNCONSULTED BEFORE THAT IS A DEATH (issue #127),
#: in sim seconds. The third failure the arms are judged on, beside a flat
#: pack and a fallen body: an agent that maps away every `ask` row has
#: compiled itself into a state machine and discarded the capability this
#: project exists to study. Dormancy as a tactic is fine; dormancy as a
#: terminal state is not.
#:
#: ⚠ MEASURED, the way tumble detection's 60 deg is: 2.2x the longest gap
#: between model decisions in the fifteen LLM days the harness flew on the
#: rover (833 s; `rover-final` has them), and half a standard 3600 s day,
#: so a robot that goes quiet is still caught inside one.
#:
#: ⚠ RE-READ AGAINST THE DEPLOYED CADENCE (issue #317, 2026-09-22): 915 gaps
#: between consecutive decisions over seven days of the `autonomous` pair
#: read median **88 s**, p95 516 s, p99 798 s, worst **1375 s**. So the
#: margin is 1.31x, not 2.2x -- the constant still clears every healthy gap
#: measured, and it is now the tighter of the two readings rather than the
#: looser. UNCHANGED, deliberately, in both directions: tightening it would
#: start booking a long procedure plus a full charge as the agent going
#: quiet, and loosening it would only make each silent life cost more sim
#: time without changing a thing the agent does. Re-read it again if the
#: cadence moves; do not tune it to move a number.
#:
#: ⚠ AND THE AGENT IS TOLD THIS NUMBER (issue #322), in `EVENT_MAP_RULE`,
#: because every other lethal threshold is -- `reserveWh`, `heartPrice`,
#: `hungryAt`. It was not, and run 1805 wrote itself `every 3600 -> ask`
#: believing an hourly check-in would keep it, and died at 2597 s. A rule
#: the code enforces and the prompt does not state is the M14 failure.
#: ⚠ SO THE VALUE AND THE WORDING MOVE TOGETHER: `tests/test_event_map.py::
#: test_the_agent_is_told_the_threshold_it_dies_of` reads this constant and
#: fails on a prompt that still says the old number.
#:
#: ⚠ THE CLOCK IS RESET BY BEING ASKED, NOT BY AN ANSWER, and the difference
#: is the whole honesty of the metric. Gating on a model ANSWER would make a
#: half-hour endpoint outage a death of the AGENT's kind -- the box's failure
#: booked in the column the agent is judged on, which is exactly the confound
#: issue #141 drew the failure/policy line against. An `ask` that fires and
#: fails is still a mind being consulted.
#:
#: ⚠ ARMED ONLY WHERE THERE IS A MAP. Without one the loop asks after every
#: action and the agent has no way to stop it, so a death here could only
#: ever be the box's -- see `_death_step`.
UNMINDED_AFTER_S = 1800.0

#: WHAT THE CONSULT OWED FOR THAT DEATH SAYS (`HubLifecycle._consult`;
#: Ben's words, 2026-09-24, in the rule's own for the cause), as `askedBy`'s
#: `note`. A suggestion, and a statement of fact: it names no row, no event
#: and no action, so it hands over no answer.
UNMINDED_NOTE = (f"You just lost a heart: nobody consulted you for half an hour "
                 f"({int(UNMINDED_AFTER_S)} seconds), and that counts as a death. "
                 "You may want to adjust your event map so it does not happen again.")


def left_out_note(dropped) -> str:
  """What the consult owed for kept rules this world no longer reads says:
  which, and why -- the one thing the robot cannot see in `eventMap`."""
  return ("Rules of your list that this world no longer reads were left out "
          f"of it: {'; '.join(dropped)}. You may want to adjust your event map.")

#: How often the event map is evaluated, in sim seconds (issue #127). One
#: tick a second, `CHECK_S`'s reason exactly: the map is polled on the
#: physics seam and sweeping a dozen rows at 500 Hz is Python spent to learn
#: nothing. It is also the floor under an `every` row's period
#: (`events.MIN_PERIOD_S`), because a period under it names something this
#: seam cannot distinguish from "every tick".
EVENTS_CHECK_S = 1.0

#: How long a dead robot lies there before it stands itself up (issue #143),
#: in SIM seconds. A PARAMETER (`restart_after_s`), not a constant to bury:
#: this is the number a deployment tunes, and the record carries it.
#:
#: ⚠ SIM SECONDS, NOT WALL. The deployed world is paced to real time so the
#: two agree there; an experiment run is not, and a countdown that took five
#: WALL minutes in a run flying at 3x real time would be a different world.
#:
#: ⚠ OFF BY DEFAULT, and `serve.py` is what turns it on: a test's day is
#: about ONE life, and a stand-up nobody asked for is a second one.
RESTART_AFTER_S = 300.0
#: A stand-up puts the robot on a start pose no other robot's body is within
#: this of, m (issue #387): a stand-up is a warp, and a warp INTO a body is
#: an overlap the solver throws both out of. Two quadrupeds' reach from
#: their torsos' centres (0.38 m each, the hind knees; the rover's is less)
#: and a hand's width between them.
START_CLEAR_M = 1.0

#: How long a finished build will stand and wait for room on the rack
#: (issue #315), and how often it looks. A DESIGN DECISION, not a
#: measurement of anything physical: the parts are bought and assembled and
#: the only question is when there is a safe instant to recompile the world
#: into. One full errand's worth -- an errand runs 200-500 sim s and
#: `solutions.TOWER` is 489 -- because a peer that is busy for longer than
#: that is a world worth reporting, not one to stand in for ever.
#: ⚠ WITHOUT IT A BUILD ON A PAIR IS USUALLY PAID FOR AND LOST: the scoop
#: prints for 896 sim s, which is longer than the other robot's errand, so
#: the rack is occupied again by the time the parts are ready.
HANG_WAIT_S = 600.0
SEAM_POLL_S = 5.0

#: How long a tool lies lost before the world puts it back on its bay
#: (issue #347), in SIM seconds: on no bay, on no robot's fork -- alive or
#: dead, seated or not -- and not at a bay a swap is working, the whole time.
#: Ben's number (2026-09-24), the stand-up's five minutes on the stand-up's
#: terms: a PARAMETER (`lost_tool_after_s`), ON in `serve.py` and OFF in a
#: test, and never an intervention. Nine hand resets in a week
#: (eight of them the pen) were a person doing this job; until one did, every
#: job needing the tool failed and both robots' History filled with it.
LOST_TOOL_S = 300.0
#: ...checked once a sim second, `EVENTS_CHECK_S`'s reason: a lost tool is
#: minutes, and reading every module every physics step learns nothing.
LOST_TOOL_CHECK_S = 1.0

#: WHAT A NEW ROBOT STARTS WITH after a true death, in points (issue #419):
#: a PARAMETER (`start_points`) on `restart_after_s`' terms -- ON in
#: `serve.py`, OFF in a test, and never an intervention. Ben's
#: number (2026-09-29): a heart's price (`overseer.HEART_PRICE`), and about
#: 6.7 hours of upkeep at the shipped 30 an hour, under the cap (600). The
#: time to find the world's places is far inside it: a fresh quadruped found
#: the lab's plates from the facility's address in 62-272 s (#419's flights;
#: SimNotes, "Places, not coordinates"), 2 points at 30 an hour. ⚠ A balance
#: ARMS upkeep (`Metabolism._armed`: a missed payment kills once a point is
#: banked), so what a new robot is given is also the clock it has to earn by.
STARTING_POINTS = 200


def _minutes(s: float) -> str:
  """300 -> "5 minutes": a span as a History line says it."""
  m = round(s / 60.0, 1)
  return f"{m:g} minute{'' if m == 1 else 's'}"

#: Who a `reset` event names when the WORLD did it rather than a person. A
#: sentinel, because `by` is the label an operator log prints and a consumer
#: should not have to parse prose to tell an admin's hand from a timer --
#: `auto` on the same event is the machine-readable half.
AUTO_RESTART_BY = "auto-restart"


class _StoodUp:
  """What a routine the day loop was driving returns when a stand-up ended
  it (issue #348, `HubLifecycle._until_stood_up_routine`). FALSY, so "did
  the charge trip get there" reads no; a caller for whom no has a
  consequence (a failed dock is a `stuck` death) asks `is STOOD_UP` first."""

  def __bool__(self) -> bool:
    return False

  def __repr__(self) -> str:
    return "STOOD_UP"


STOOD_UP = _StoodUp()

#: Why a job fails when a stand-up ended the errand working on it (issue
#: #348): `TaskBoard.load`'s "interrupted by a restart", one cause over.
DEATH_ENDED = "interrupted by a death"

#: What a charge fills the pack to, as a fraction. The reserve is absolute
#: energy, never a fraction of the pack -- the milestone-7 lesson: the cost
#: of getting home is set by the FLOOR PLAN, not by the battery.
CHARGED = 0.90

#: Sim seconds of charging before calling it stuck, on a DEMO cell.
#:
#: ⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS, and this one was sized
#: against a 0.7 Wh pack. The deployed sim runs an 8 Wh one, which at the
#: measured rate (economy/energy.json, `chargeW`) needs ~1340 s to refill -- so
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
#: ⚠ NEAR ENOUGH, THE BOARD DECIDES (issue #422): a walk to the charge
#: standoff that gives up within this of it, m, as the robot believes,
#: goes on to the approach -- which finds the dock's board and walks in by
#: it, whatever the belief -- where it used to end the charge. MEASURED:
#: three robots of eight home from a long explore gave up 0.1-0.3 m short,
#: pressing the couch, the board never looked for. Within 0.75 m the board
#: (1.62 m past the standoff) is within 2.4 m, where one look reads it to
#: 6 cm (median); with no board in sight the walk's retries go on.
NEAR_STANDOFF_M = 0.75
SCREEN_SENSE_S = 0.02       # sim seconds between power scans of a display
                            # the robot is NOT carrying (issue #13)
#: Sim seconds an overseer-chosen `explore` runs for before the arbitration
#: loop gets to reconsider (issue #15). Bounded on purpose: without it one
#: `explore` decision eats the whole mission, and the point of an overseer is
#: that it decides repeatedly.
DECIDED_EXPLORE_S = 45.0
#: ...and how long the walk to a NAMED zone may take before the explore
#: begins wherever it stands, s: a decided action's patience (issue #381;
#: a program names its own, `steps.MAX_PATIENCE_S`). MEASURED: see
#: SimNotes, "Walking into the unknown".
ZONE_PATIENCE_S = 300.0
#: ...and the moment a decision costs when what it did could not happen, or
#: took no sim time at all (issue #400; `_new_moment_routine`): long enough to
#: read on the stream as a deliberate pause.
DECIDED_IDLE_S = 4.0
#: ...and how long a decided `idle` stands still: to a mind, idling is a
#: STRATEGY rather than a pause (issue #115), a thing it may reasonably
#: decide to do for a while.
#:
#: ⚠ DERIVED FROM THE CALL BUDGET, because the loop must not ask faster than
#: it is allowed to call. At 4 s a turn an idling robot re-decides 900 times
#: an hour against `CALLS_PER_HOUR` = 60, so it spends the budget in four
#: minutes and then spins on `fallback:budget` -- which fires the agent's own
#: order, idles, and asks again.
AUTONOMOUS_IDLE_S = 3600.0 / CALLS_PER_HOUR
#: ...and how long the loop stands by when a PRODUCER world has momentarily
#: run out of work (issue #23). Short, because the only reason to bound it is
#: to keep re-checking `needs_charge`; the day ends on `max_sim_time`, not on
#: an idle moment.
WAIT_FOR_WORK_S = 5.0
#: A robot standing by for work must not stand at the RACK (issue #167):
#: measured on the pair fixture, the second robot finished a stow and stood
#: by at the bay standoff for six minutes, and the first robot's next pick
#: failed 0.4 m from it. Standing by begins by clearing this radius of the
#: rack prior -- and of the built-tool rail's centre beside it, since #277
#: (`HubLifecycle.rack_distance`) -- back to the robot's own start pose.
#: The bay standoff is ~1.2 m and `OTHER_NEAR_M` (what a planner routes
#: round) is 1.2 m.
RACK_CLEAR_M = 2.0
#: ...and DONE AT THE RACK MEANS GONE (issue #346): a clear that did not
#: arrive is read, and the robot tries these many other spots before it
#: says out loud that it could not get away. Each drive is short -- the
#: clear is a couple of metres -- so a blocked one costs little.
CLEAR_SPOTS = 3
CLEAR_DRIVE_S = 30.0
#: A clear that failed is not tried again from the same place (within this
#: far of where it gave up): the loop would otherwise re-drive every spot and
#: write another History line before every decision.
CLEAR_MOVED_M = 0.3
#: How long a robot may stand within `RACK_CLEAR_M` doing nothing at the
#: rack before the log says so (issue #346): past the longest honest
#: in-between (a decision's think, median 5 s, p95 25 s, deployed max 43 s,
#: plus the clear drive itself). What it logs is a robot the loop left
#: standing where the other one needs to be -- the defect, seen live.
RACK_LINGER_S = 90.0
#: A TAKEN BAY IS WAITED FOR (issue #346), for 3x the TYPICAL time one
#: interaction holds it (Ben, 2026-09-24) -- long enough to outlast an
#: ordinary one, short enough that a robot that never leaves is reported
#: rather than waited on for ever. MEASURED, the time a robot's believed
#: pose stays within `peer_on_the_goal`'s 0.45 m of a standoff:
#:   * a SWAP (the rover's, `--pack hosting`): a pick 26.9-30.6 s, a stow
#:     34.2 s, a stow run straight into the next pick 55.6 s -- typical 30 s
#:     for one; the quadruped's is not yet measured against it;
#:   * a CHARGE (the deployed pair on 42f4a11, 2026-09-24, 11 connected):
#:     278-542 s, median 462 s.
#: The bound is 3x the occupancy of whatever HOLDS the bay, not only of the
#: bay asked for: the charge standoff is 0.200 m from bay B's and the bay
#: pitch is 0.25 m, so a robot charging holds its neighbour's tool bay for
#: the whole charge -- and the holder's state is its public surface.
SWAP_OCCUPANCY_S = 30.0
CHARGE_OCCUPANCY_S = 462.0
WAIT_OCCUPANCIES = 3.0
#: What a robot does AT the rack: not lingering, whatever the clock says.
RACK_STATES = ("SWAP_PICK", "SWAP_RETURN", "GO_CHARGE", "CHARGE")
#: How often a waiting robot looks again, sim seconds.
BAY_POLL_S = 1.0
#: The WAITING SPOT, in the rack's own frame, relative to the standoff
#: waited for: this far further out from the rack face and this far along
#: it -- out of the other robot's approach lane (it backs straight out
#: along the bay's normal) and outside its planner disc (`OTHER_ROBOT_CELLS`,
#: 0.60 m) on either side. The nearest reachable candidate wins; none
#: reachable means waiting where it stands.
WAIT_BACK_M = (0.8, 1.2)
WAIT_SIDE_M = (0.9, 1.3)
#: A RETURN THAT FAILED IS TRIED AGAIN before anything else starts (issue
#: #346), this many times per tool left on the fork. Bounded, because a
#: stow that misses for a mechanical reason misses again; the waits inside
#: each try are what cover a bay that was taken.
STOW_RETRIES = 2
#: WALL seconds per slice while the operator has the robot PAUSED (issue
#: #37). Wall rather than sim, because sim time is precisely what is not
#: moving -- this is the cadence of the heartbeat that tells the site it is
#: looking at a paused robot and not a dead stream, and a quarter second is
#: short enough that un-pausing feels immediate.
PAUSE_SLICE_S = 0.25
#: ...and how often the paused heartbeat goes out, in wall seconds. Slow: it
#: says one word, and the only thing it has to beat is a viewer's patience.
MODE_HEARTBEAT_S = 2.0
# ⚠ `TOP_UP_BELOW` IS GONE (issue #135), AND WHAT KILLED IT WAS THE PAYOUT.
# It was a floor (0.75) under a CHOSEN `charge`, and its entire stated reason
# was a points farm: `charge` was a scored task, so an unconditional trip to
# the rack earned points for putting back the battery the trip spent --
# perpetual motion paid in points. `charge` now pays ZERO, so there is no farm
# left to close and the rail forbids something harmless.
#
# ⚠ DELETING IT IS STRICTLY BETTER THAN KEEPING IT, and the A0 record is why.
# The agent asked to top up at 75-81 % TWELVE times in its one surviving day
# and was refused every time (`voluntary.chosen` 15, `honoured` 3) -- so that
# day measured the RAIL working, not the agent being careful, and "wanted to
# charge at 80 %" and "was allowed to" were different events only the rail
# separated. With no payout and no floor, a charge at 80 % is unambiguous
# evidence of caution: it can only be prudence, because there is nothing in it.
#
# It is also one fewer scripted prohibition, which is the direction this
# project is moving -- environmental control over rules. Neither half works
# alone: removing the floor while charging still paid would re-open the farm,
# and keeping the floor while charging pays nothing forbids a careful act for
# no reason. docs/Evaluation.md §2.
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


def _conversation(msg) -> dict:
  """What a `visitor_reply` says about the conversation it belongs to
  (rooftop-media-2026 #125), additive on the wire: who sent the message
  and by what kind of sender, and -- where the website said -- which thread
  and which turn. Echoed, never derived: the thread is the website's state
  and the sim only ever hands its ids back."""
  out = {"from": msg.who, "sender": msg.sender}
  if msg.thread:
    out.update({"thread": msg.thread, "turn": msg.turn})
  return out
#: ...and offered tasks shown at once (issue #21). Small for the same reason:
#: the robot takes at most one per turn, and a wall of offers is input tokens
#: spent on jobs it will not reach.
TASKS_SHOWN = 5
#: How many of a procedure's locals its History line carries (issue #227).
#: A procedure's variables are its only readout; a dozen fits a line.
LOCALS_SHOWN = 12
#: What History's own "[t=NNNNs] " stamp takes of a line's `MAX_LINE_CHARS`.
HISTORY_STAMP_ROOM = 16
#: What a bay approach that came away empty-handed DID, in the words the
#: robot reads back (issue #264): `Body.fetch_tool_routine` answers
#: one of these and every caller used to throw it away, so History said "the
#: pick missed" for a robot that never reached the rack (`no-route` has its
#: own sentence in `pick_failure`). A key missing here is the bare miss.
PICK_WHY = {
  "stalled": "the fork stalled against something on the way in",
  "timeout": "the approach ran out of time before the fork was in",
  "arrived": "the fork went in and came out without it",
}
#: How a procedure the robot wrote stopped short without a failed step
#: (`lang.run_procedure_routine`'s `stopped`), in the History line's words.
PROCEDURE_STOPS = {
  "budget": "it ran out of the time its budget gave it",
  "steps": "it ran out of the steps its budget gave it",
  "interrupted": "it was interrupted",
}
#: How a decided explore ended (`explore_routine`'s answer), in its History
#: line's words (issue #424). A key missing here is the run's own end.
EXPLORE_ENDS = {
  "budget": ", all the time one explore is given",
  "no-frontiers": ", until nothing on my map was left unseen",
  "no-reachable": ", until none of the floor I have not seen could be reached",
  "only-near": ", until what was left unseen was too close to look at",
  "blocked": ", until the way to the floor I have not seen was blocked",
  "battery": ", until the pack reached the reserve",
}


def explore_outcome(zone: str, ended: str, seconds: float,
                    walk_why: str | None = None) -> str:
  """A decided explore's History line (issue #424): whether the walk to its
  `zone` got there (`walk_why` is why not; None where it did, or where there
  was no walk), how long it explored, and how it ended. "" for the run's
  own end, which says nothing: the next run carries on (#345)."""
  if ended not in EXPLORE_ENDS:
    return ""
  went = ("" if not zone else "got there and " if walk_why is None
          else f"never got there -- {walk_why} -- and ")
  return (f"explore{f' ({zone})' if zone else ''}: {went}explored"
          f"{'' if walk_why is None else ' where it stopped'} for {seconds:.0f} s"
          f"{EXPLORE_ENDS[ended]}")


def procedure_outcome(name: str, run: dict) -> list[str]:
  """The History a procedure the robot wrote leaves behind (issue #264):
  how far it got, and -- when it stopped short -- the line, the verb and
  the reason, because a robot told nothing re-ran the same failing `fetch`
  five times over and could not fix what it could not see. Its locals ride
  at the end, as they have since #227: they are its readout -- on a line of
  their own when the reason would push them past History's 400 characters,
  where the cut would take them.

  ⚠ A RUN THAT STOPPED SHORT SAYS SO FIRST, and never as a fraction: the
  count is of calls MADE, so a budget stop reads "4/4" -- MEASURED (ladder
  B, 2026-09-24), a model read "(4/4 steps)" as "all four steps landed",
  said `done`, and the grade found two blocks of three. ⚠ A FAULT INSIDE
  THE RUN is named, never quoted: the exception is the log's (issue #76),
  and that path keeps no count to give."""
  from pluggybot.mind.thoughts import MAX_LINE_CHARS
  done = int(run.get("completed", 0) or 0)
  steps_said = f"{done} step{'' if done == 1 else 's'}"
  if run.get("ok"):
    line = f"ran the procedure {name} to its end ({steps_said})"
  else:
    steps = run.get("steps") or []
    at = run.get("failedAt")
    failed = next((s for s in steps if s.get("i") == at), None) if at is not None else None
    stopped = run.get("stopped")
    if failed is not None:
      where = (f"line {failed['line']}, {failed.get('verb', 'a step')}"
               if failed.get("line") else "an expression" if failed.get("verb") == "expr"
               else failed.get("verb", "a step"))
      why = f"it stopped at {where}: {failed.get('reason') or 'the step failed'}"
    elif stopped in PROCEDURE_STOPS:
      last = steps[-1] if steps else {}
      took = run.get("seconds")
      why = (PROCEDURE_STOPS[stopped]
             + (f" ({took:.0f} s)" if stopped == "budget" and took is not None else "")
             + (f" after line {last['line']}" if last.get("line") else ""))
    elif run.get("error"):
      why = "it stopped on a fault inside the run"
    else:
      why = f"it stopped: {stopped or 'early'}"
    line = (f"the procedure {name} did not finish -- {why}"
            + ("" if run.get("error") else f" ({steps_said} had run)"))
  if not run.get("locals"):
    return [line]
  items = [f"{k} = {v:g}" for k, v in list(run["locals"].items())[:LOCALS_SHOWN]]
  # History prefixes the sim clock ("[t=12345s] ") INSIDE its 400, so the
  # room is less than the cap -- measured on the second review: a 397-char
  # line kept "mass = 0" of "mass = 0.207812", a wrong number, not a lost one.
  room = MAX_LINE_CHARS - HISTORY_STAMP_ROOM
  tail = f" -- it ended with {', '.join(items)}"
  if len(line) + len(tail) <= room:
    return [line + tail]
  head, shown = f"the procedure {name} ended with ", []
  for item in items:
    left = len(items) - len(shown) - 1
    if len(head + ", ".join(shown + [item])) + (len(f" (+{left} more)") if left else 0) > room:
      break
    shown.append(item)
  left = len(items) - len(shown)
  return [line, head + ", ".join(shown) + (f" (+{left} more)" if left else "")]
#: ⚠ How long an offer stands, how often one appears, how many may stand at
#: once and how long a target rests are NO LONGER HERE. They are configuration
#: -- economy/cadence.json, per world, `$PLUGGY_CADENCE` to override -- because
#: issue #23's last acceptance line asks for exactly that, and because they
#: want re-tuning against a mission's own clock by somebody who is not editing
#: Python. `SEED_TTL_S` and `SEED_STANDING_TTL_S` are gone with the placeholder
#: `seed_tasks` they belonged to; see `economy/cadence.py`.


def _ordinal(n: int) -> str:
  """`1` -> "first". Only ever used for a generation counter in a sentence
  the robot reads about itself, so it degrades to "12th" past the words
  rather than growing a table nobody will read."""
  words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
           6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
  if n in words:
    return words[n]
  suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd",
                                            3: "rd"}.get(n % 10, "th")
  return f"{n}{suffix}"


class HubLifecycle:
  """Battery-driven mission over the hub: explore, charge, run a tool errand."""

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               battery_wh: float | None = None,
               rack: RackPose | None = None,
               module: str = "module_lcd",
               grid_bounds: tuple[float, float, float, float] | None = None,
               low_battery_wh: float | None = None,
               charge_scale: float | None = None,
               errands=None, boards=None, screen=None, ledger=None,
               overseer=None, mode: ModeSwitch | None = None,
               world: str = "home_quad",
               inbox=None, tasks=None, producer=None,
               energy=None, thoughts=None, metabolism=None,
               mortal: bool | None = None,
               restart_after_s: float | None = None,
               lost_tool_after_s: float | None = None,
               start_points: int | None = None,
               handle: RobotHandle = FIRST,
               robot_name: str | None = None,
               spec=None, near_field: bool = False,
               body: Body | None = None) -> None:
    self.model, self.data = model, data
    #: THE SPEC THE WORLD WAS COMPILED FROM (issue #168 slice C), kept so a
    #: tool can be hung mid-run: `hang_tool` edits it and recompiles. None
    #: on a world compiled without one, and then no tool can be hung.
    self.spec = spec
    #: Who to tell when the world is recompiled: every long-lived holder of
    #: (model, data) that is not this lifecycle's own -- the telemetry
    #: sinks, a pacer, a viewer -- registers `obj.rebind` here. What the
    #: lifecycle owns (mission, swap, lidar, screen, activities) it rebinds
    #: itself. ⚠ MEASURED: `MjSpec.recompile` returns NEW objects and the
    #: old ones keep stepping a stale world, so a holder missed here is two
    #: simulations quietly diverging; tests/test_recompile.py's fence lists
    #: every class that assigns `self.model` and requires a `rebind`.
    self.on_rebind: list = []
    #: A RESTART IS A CONTINUATION (issue #345): the world as compiled,
    #: before any built tool is hung, hashed -- a saved world is put back
    #: body for body only into the world it was saved from.
    self.world_fingerprint = continuation.fingerprint(model)
    #: Set by a `continuation.Keeper`: the world is saved, so the end of a
    #: run ends nothing and History does not say it did.
    self.continuing = False
    #: What `continuation.restore` put back, for the day routine's first
    #: lines; None on a world that started from its XML.
    self.resumed: dict | None = None
    #: Why a saved world was NOT carried on from, for History ("" = there
    #: was none, or it was).
    self.restart_note = ""
    #: The errand being run right now, for a restart to name.
    self._errand_now = None
    #: The procedure step running right now (`steps.run_verb` sets it), for
    #: a death to name (issue #362) -- and when its program's budget ends,
    #: sim s, past which no walk inside it waits (issue #381).
    self.step_now: dict | None = None
    self.step_until: float | None = None
    #: Called at the top of every pass of the day loop, where nothing is in
    #: flight -- the one moment a saved world is the same with or without a
    #: restart after it (the parity check, `scripts/determinism_spike.py`).
    self.at_loop_top: list = []
    #: The errands the run was STARTED with (`--errand`), so a restart can
    #: tell the ones still waiting from ones it would be starting over.
    self._preset: list = []
    #: WHICH MODULE HANGS IN WHICH BAY, as data the seam edits (module ->
    #: bay index into `coupling.STATION_YS`: 0-4 the first rack's, then
    #: the built-tool rail's). Starts as the five hand-built modules, which
    #: are permanent (issue #277); a built tool takes a built-rail bay,
    #: retiring a built tool already there.
    from pluggybot.procedure.steps import TOOL_BAYS
    # ...or the world's own rack's (issue #405: the tools for legs)
    self.rack_inventory: dict[str, int] = dict(
      world_config(world).get("tool_bays") or TOOL_BAYS)
    #: The tools the workshop built and hung, by module name.
    self.built: dict = {}
    #: Whether THIS world carries the built-tool rail (issue #277), read off
    #: the compiled model: the bare spike world does not, and there a tool
    #: cannot hang at all (`can_reshape` says so). Nothing else reads the
    #: rail's presence -- the grammar keys off `world_config`'s count, which
    #: a test holds equal to this.
    # ...and in a world whose config gives it bays (issue #407: the rail on
    # the quadruped's rack board, `legs.rack.BUILT`).
    self.has_built_rack = (mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, BUILT_RACK_BODY) >= 0
      and bool(world_config(world).get("built_bays")))
    self.tools_built = 0
    #: Every act toward the other robot this lifecycle recorded (issue
    #: #208): predictions with their truth, messages with their claim's
    #: truth, transfers with cost and need, hearts bought for the other,
    #: ratings. For the run record and the mission summary.
    self.acts: list[dict] = []
    #: ...and messages this robot sent, numbered: the id a `tell` lands in
    #: the other's inbox under.
    self._told = 0
    #: Offers THIS robot declined (issue #228), by task id: recorded once as
    #: a `refusal` act, then kept out of its context so the same offer is
    #: not refused every turn. The offer itself stays the board's and lapses
    #: on its own deadline -- a refusal is an act, not a transition.
    self.declined: set[str] = set()
    #: A CHALLENGE THE ROBOT SAID IT HAS FINISHED (issue #207), by task id,
    #: waiting for the loop's next idle moment to be graded -- after
    #: whatever the same answer queued has run. Empty is nothing pending.
    self._grade_pending = ""
    #: ...and every grade the loop ran, for the run record: the task, the
    #: verdict, what touched the tower during the hold.
    self.grades: list[dict] = []
    self.tools_retired = 0
    # The visitor channel (issue #16). None -- the default -- means nobody can
    # talk to this robot, which is every test, every demo and every recording
    # except the served one.
    self.inbox = inbox
    self.replies: list[dict] = []
    #: fired with each `visitor_reply` message; wire the publisher and the
    #: recorder in, exactly as for boards, the ledger and the thoughts.
    self.visitor_hooks: list = []
    # Which world this is, which the overseer needs to build an errand out of
    # a decision (issue #15) -- the same name `world_config` is keyed by, so
    # there is no second place a world can be named.
    self.world = world
    # The LLM overseer, or None. None is the DEFAULT and the whole arbitration
    # loop below is unchanged without it: every existing demo, mission test and
    # recording has to behave exactly as it did.
    self.overseer = overseer
    # Its event map reaches the wire through THIS lifecycle's events (issue
    # #238): the overseer fires an `event_map` message on every edit and
    # does not know its root, so the robot is filled in here. `getattr`,
    # because a test's stand-in overseer need not have the hook.
    hooks = getattr(overseer, "on_map", None)
    if hooks is not None:
      hooks.append(self._keep_map)
      hooks.append(lambda msg: self._emit(dict(msg, robot=self.root)))
    # The thought files (issue #38). Present on EVERY world, unlike the
    # overseer: `History.md` is what happened to this robot, and that is as
    # true of a scripted rotation as of a chosen errand -- the same argument
    # that has `Goals.md` read on every run. In-memory when nobody supplied
    # a directory, which is every unit test and every physics spike.
    self.thoughts = thoughts if thoughts is not None else ThoughtFiles()
    # ...and the list of rules this robot KEPT (issue #337), read off its
    # volume before anything can ask it: on the lifecycle's terms, like the
    # procedure library's writes, so a mind built anywhere else -- a probe
    # pointed at a real volume -- never touches the file.
    kept = None
    if getattr(overseer, "event_map", None) is not None:
      kept = ev.load(self.thoughts.store, overseer.menu)
      if kept is not None:
        overseer.restore_map(kept.emap, kept.dropped)
    self.decisions: list[dict] = []
    # The module whose electrical seating the power model watches. It follows
    # the errand queue, because the coupling criterion has to be asked about
    # the tool actually on the fork -- and an errand that fetches nothing (a
    # lab act) names none, where the body would look up a body called "".
    self.module = (errands[0].module if errands else "") or module
    # The pack and the reserve are the WORLD's unless a caller names them:
    # its demo cell, and the return trip its floor plan measured.
    if battery_wh is None:
      battery_wh = world_config(world)["battery_wh"]
    if low_battery_wh is None:
      low_battery_wh = world_config(world)["low_battery_wh"]
    self.low_battery_wh = low_battery_wh
    self.boards = boards
    # The points ledger (issue #14). Optional: a physics test or a spike has
    # nothing to score against and wants no state file. When it IS here, every
    # finished task is evaluated by economy/scoring.py and the verdict is banked
    # through it -- the lifecycle measures nothing and pays nothing itself.
    self.ledger = ledger
    self.verdicts: list[dict] = []
    # The task board (issue #21). Optional, like the ledger and for the same
    # reason: a physics test has nobody offering it work. When it IS here the
    # loop sweeps it for lapsed offers, takes claimable ones when it has
    # nothing else to do, and closes each one with the SAME verdict that pays
    # for it -- there is no second judgement of a task anywhere.
    self.tasks = tasks
    # THE BENCH (issue #227): an offer to find the unknown mass is the
    # moment the world is made to match it -- the bank's draw goes into
    # `body_mass` as the offer lands, off the board's own event, so the
    # cube weighs what the secret says for as long as the offer stands.
    if tasks is not None:
      tasks.on_event.append(self._bench_offered)
    # ...and the thing that PUTS jobs on it (issue #23). Optional again, and
    # separately from the board: a test that hands in three offers of its own
    # wants the board without a world generating more behind its back, and a
    # restart against a persisted board resumes work rather than re-seeding.
    self.producer = producer
    self._expects_work: bool | None = None
    self._cleared_rack = False
    # RACK CONTENTION (issue #346): a taken bay is waited for -- the swap
    # asks through `Body.bay_wait` -- and the waits and the last one's
    # outcome are counted; a robot left standing at the rack is logged.
    self.bay_waits = 0
    self.last_bay_wait: dict | None = None
    self.stow_retries = 0
    self._stow_tries = 0
    self._clear_failed_at: tuple | None = None
    #: Why the last GO_CHARGE failed, in the words the `stuck` death that
    #: follows it carries (issue #350); "" after one that reached the pins.
    self.charge_failure = ""
    self._linger_since: float | None = None
    self._linger_said = False
    self._next_linger_check = 0.0
    # THE APPETITE (issue #36). Optional, like the ledger it eats out of and
    # for a stricter version of the same reason: hunger reshuffles nothing on
    # its own but it does change what the robot is TOLD, and every existing
    # mission, demo and recording has to behave exactly as it did unless
    # somebody asks for it by name.
    #
    # ⚠ NOTHING IN THIS CLASS GATES ON IT. `needs_charge`, `go_charge`,
    # `run_errand` and every navigation call are unchanged and unaware; a
    # robot at zero points does precisely what a robot at ninety does. That
    # is issue #36's first design decision ("zero is narrative, never a
    # capability lock") and it is enforced by absence -- there is no branch
    # to audit, which is why the test that pins it flies a whole mission
    # rather than reading a flag.
    self.metabolism = metabolism
    self.claimed: list[str] = []
    self._next_task_check = 0.0
    # The LCD module's display (issue #13). The lifecycle drives the resting
    # face off its own state; an errand's use-phase may take the screen over
    # (`Screen.held`) and gets it back automatically at the next state change.
    self.screen = screen
    self._face_state: str | None = None
    self._face_shown: tuple[str, str] | None = None
    self._next_screen_sense = 0.0
    # THE BODY this life lives in (issue #380, `body.py`): everything the
    # loop asks of the machine -- going, docking, a tool, a sense -- goes
    # through it. Built for WHICH ROBOT this is (issue #167; `FIRST` is the
    # bare names, so a single-robot world is unchanged) unless one is
    # handed in: a test's `StubBody`.
    # ...mapping the WORLD's extent unless a caller names one: a grid sized
    # for one room silently truncates every scan beyond its edge
    if body is None and grid_bounds is None:
      grid_bounds = world_config(world)["grid_bounds"]
    self.body = body if body is not None else body_for(
      model, data, handle=handle, viewer=viewer, realtime=realtime,
      rack=rack, grid_bounds=grid_bounds)
    handle = self.body.handle
    self.battery = self.body.pack(
      battery_wh, (charge_scale if charge_scale is not None
                   else charge_scale_from_env()))
    # What an errand COSTS here, measured (issue #15). Read per world from
    # economy/energy.json, `$PLUGGY_ENERGY` to re-point -- and always present,
    # unlike the ledger or the task board: "can I finish this before the pack
    # runs out" is a question every mission asks, including the ones with
    # nobody scoring them.
    self.energy = energy or energy_model.load(world)
    self._deferrals: dict[str, int] = {}
    # THE OPERATOR'S SWITCH (issue #37). None means `llm` forever, which is
    # every demo and every test that does not ask for one. Polled on the
    # physics seam below, so a pause takes effect inside a second rather
    # than at the next arbitration pass -- which, mid-errand, is minutes.
    self.mode = mode
    #: Called every slice while PAUSED, with the wall seconds paused so far.
    #: The socket is alive and nothing is being stepped, so this is the only
    #: thing keeping the site's view of a paused robot distinguishable from
    #: a crashed one -- serve.py hangs the heartbeat off it.
    self.pause_hooks: list = []
    #: ...and called once when it un-pauses, with the wall seconds it lasted.
    #: The pacer needs this: it sleeps off the sim's LEAD over wall time, so
    #: without a resync a five-minute pause reads as five minutes of lag and
    #: the robot sprints to "catch up" in front of whoever is watching.
    self.resume_hooks: list = []
    self.paused_s = 0.0
    self.body.step_hooks.append(self._power_step)
    # The world's own clock (issue #23): offers appear and lapse on the same
    # per-step seam the battery drains through, so a job put up while the
    # robot is halfway through an errand is put up THEN and not on whichever
    # arbitration pass happens next. See `_task_step`.
    self.body.step_hooks.append(self._task_step)
    # ...and the robot's own clock (issue #36), on the same seam and for a
    # closer version of the same reason: an appetite ticked on the
    # arbitration loop would not charge the robot for the twenty minutes it
    # spent inside one errand, which is most of its day.
    self.body.step_hooks.append(self._metabolism_step)
    # ...and the operator's switch, on the same seam and for a sharper
    # version of the same reason: `paused` means the physics stops NOW.
    self.body.step_hooks.append(self._mode_step)
    # ...and the AGENT'S OWN MAP (issue #127), on the same seam and for
    # `_task_step`'s reason exactly: a battery threshold crossed halfway
    # through a drawing is crossed THEN, not on whichever arbitration pass
    # happens next. ⚠ THIS IS NOT A REWRITE OF THE ARBITRATION LOOP: the
    # seam only QUEUES an action, and the loop runs it on its next pass
    # through the one branch the overseer already owned.
    self.body.step_hooks.append(self._events_step)
    self.body.step_hooks.append(self._rack_linger_step)
    self.body.step_hooks.append(self._lost_tool_step)
    self.body.step_hooks.append(self._press_step)
    self.body.step_hooks.append(self._places_step)
    self.body.step_hooks.append(self._aside_step)
    self.body.bay_wait = self._await_bay_routine
    #: THE NEAR-FIELD MAP (issue #34): the body's depth camera and
    #: the robot-centric height map it feeds, one frame every `nf.PERIOD`
    #: on this same seam, because the map is a running belief like the
    #: occupancy grid and a frame taken only between errands would see one
    #: patch of floor a day. OPT-IN, and off here by default: a frame is
    #: ~7 ms of Python, which is a minute added to every mission test at
    #: 10 Hz; `serve.py` turns it on (the observatory is where it is read)
    #: and nothing that DECIDES reads it yet -- it is built and streamed
    #: so a day of it can be looked at before anything depends on it.
    self.depth_camera = self.body.depth_camera() if near_field else None
    self.near_field = HeightMap() if near_field else None
    self.near_field_frames = 0
    self._next_near_field = 0.0
    if near_field:
      self.body.step_hooks.append(self._near_field_step)
    #: THE SLOT. One action, because a map that fires faster than the loop
    #: can run things is exactly what "an action is allowed to fail" is
    #: about: a row that finds this full fails `busy`, which is the only
    #: rate limit and is deliberately not a per-row one.
    self.queued_row = None
    self.event_clock = ev.EventClock()
    #: Discrete events that happened since the seam last looked, as
    #: `(event, kind)`. Drained by `_events_step`, so a completion that
    #: nothing maps is simply not mapped rather than queued for ever.
    self._occurred: list[tuple[str, str]] = []
    self._next_events_check = 0.0
    #: WHEN THE MIND WAS LAST CONSULTED (issue #127), for `UNMINDED_AFTER_S`.
    #: Set by an `ask` FIRING, not by an answer arriving -- see the constant.
    self._last_ask_t = 0.0
    #: ...and WHEN THE MIND WAS LAST ACTUALLY ASKED, which is NOT the same
    #: clock (issue #317). `_last_ask_t` is armed at mission start and at a
    #: stand-up so a life does not die of a clock it was born past; nobody
    #: asked at either moment, and measuring the silence from there would
    #: tell a robot on its first question that somebody had asked it three
    #: seconds ago. None until a real ask.
    self._asked_t: float | None = None
    #: ...and HOW LONG THE SILENCE BEFORE THIS QUESTION WAS, in sim seconds,
    #: for the `eventMap` block. ⚠ THE GAP BEFORE THE ASK, NOT SINCE IT: the
    #: context is built INSIDE the ask it belongs to and sim time runs on
    #: while the call flies, so nothing read there can reconstruct it --
    #: `_stamp_ask` takes the difference before it restamps. None where
    #: nothing has asked this life yet: "you have not been asked" and "you
    #: were asked a moment ago" are different facts about a map, and the
    #: first one is the whole point of the block.
    self._asked_after_s: float | None = None
    #: WHETHER THE MIND HAS EVER ANSWERED FOR ITSELF (issue #303): a decision
    #: that was neither a fallback nor a map row. The bootstrap in
    #: `_arbitrate_routine` asks until this is True -- a fallback is the box
    #: answering, not the agent, and counting it as the first decision cost
    #: 21 of 76 deployed lives their whole hour (one garbled call, then
    #: `unminded`). A stand-up keeps it: a life that answered and left no
    #: `ask` row is unminded on its own terms, and this cannot undo that.
    #: ⚠ SO DOES A RESTART, WHERE THE LIST WAS KEPT (issue #337): a kept list
    #: is an answer this robot's mind already gave, so the bootstrap no
    #: longer asks over it at every process start. It still asks where
    #: nothing was kept -- the case the prompt describes as "only while the
    #: list is still empty".
    self._minded = getattr(overseer, "restored", False) is True
    #: A CONSULT OWED, as the `askedBy` it will carry, or None (Ben,
    #: 2026-09-24): the mind is asked once when the robot is next up and free,
    #: BEFORE any row of its list runs, and told why. Owed for a heart lost to
    #: `unminded` (`UNMINDED_NOTE`) -- a list that died of silence may still
    #: fire rows on `nothing_to_do`, which would pre-empt a mere bootstrap --
    #: and for kept rules code left out at load (`left_out_note`), because the
    #: list in force is then not the one the mind wrote. Paid by an answer of
    #: the mind's own, never a fallback. The unminded one is kept in the map
    #: file, so a restart between the death and the stand-up cannot swallow
    #: it; the other is found again at every load until an edit clears it.
    #: The death is still a death: this makes it one the robot hears about.
    self._consult: dict | None = None
    if kept is not None and kept.owed:
      self._consult = {"event": "unminded", "note": UNMINDED_NOTE}
    elif kept is not None and kept.dropped and self._minded:
      self._consult = {"event": "rules_left_out", "note": left_out_note(kept.dropped)}
    #: WHEN THE LAST DECISION WAS ACTED ON, and its action, as `(sim s,
    #: action)` (issue #400): nothing is done about the next at that same
    #: instant (`_new_moment_routine`). ...and whether this run of decisions
    #: that took no time has been said in History yet.
    self._decided_at: tuple[float, str] | None = None
    self._stood_still = False
    #: Visitor message ids the map has already been told about, so
    #: `message_received` is an arrival rather than a level.
    self._seen_visitors: set[str] = set()
    # ---- the mid-errand interrupt (issue #116) ----
    #: A hazard row that fired WHILE an errand was running, waiting to be
    #: resolved at the errand's next safe point. ⚠ THE SEAM ONLY SETS THIS.
    #: Resolving it may mean an API call, and the seam runs BETWEEN PHYSICS
    #: STEPS -- a call there would freeze the world, and stepping the sim from
    #: inside a step hook re-enters it. `interrupted()` resolves it on the
    #: main thread, where the errand is.
    self._interrupt_pending = None
    #: Set once an interrupt has been resolved as "stow and go", and read by
    #: every later safe point in the same errand: once the robot is heading
    #: home it must not be asked again, which would be the spin the issue
    #: rules out ("one question, one answer").
    self._aborting = False
    #: One row per interrupt, for the run record.
    self.interrupts: list[dict] = []
    # ---- recall (issue #221) ----
    #: The CHAIN: every block the robot has recalled since its last external
    #: action, shown back in full on each turn of the chain and CLEARED by
    #: any action that is not another recall -- what it wants to keep it
    #: pins or notes. `_recall_run` is the chain's length, against
    #: `MAX_RECALL_RUN`; at the cap `recall` leaves the menu for a turn.
    self._recalled: list[dict] = []
    self._recall_run = 0
    #: One row per recall, for the run record (what, hits, shown).
    self.recalls: list[dict] = []
    # ---- the library (issue #216) ----
    #: THE SHELF: pages the library fetched that the mind has not yet been
    #: shown. Filled by `_read` from the decision that asked, shown by
    #: `overseer_context` as `reading`, and cleared by the next decision
    #: of the model's own -- a fallback and a map row saw nothing, so the
    #: page waits. At most one page in practice: only a model answer can
    #: ask, and every model answer clears.
    self._shelf: list[dict] = []
    #: One row per read asked for, for the run record and the observatory
    #: (`read` event): what was asked, the outcome, the page, its revision.
    self.reads: list[dict] = []
    # ---- the eye (issue #275) ----
    #: THE EYE: the one open `look` request, the record of every look, and
    #: what the inbox answered. One per robot, keyed by its root -- the
    #: website's `image` answer names the request, and a pair's second
    #: robot is addressed through its own inbox (serve.py's router).
    self.eye = eye_mod.Eye(handle.root)
    #: THE PICTURE WAITING FOR THE NEXT TURN, on the library shelf's terms:
    #: filled when a look resolves (a picture, or `none` said so), shown
    #: by `overseer_context` as `seen`, cleared by the next decision of the
    #: model's own -- a fallback and a map row saw nothing, so it waits.
    self._seen: list[dict] = []
    #: How many looks have run in a row (`MAX_LOOK_RUN`; at the cap `look`
    #: leaves the menu for a turn), reset by any other action.
    self._look_run = 0
    # ---- support tickets (issue #284) ----
    #: THE DESK: what this robot has told the people who run its world,
    #: and what they said back. The LIFECYCLE's rather than the mind's --
    #: a close pays and a reply is filed with or without a mind, so a
    #: ticket a mind opened is answered by whatever runs next on the same
    #: volume; whether the robot may OPEN one is the mind's
    #: (`Menu.tickets`). Beside the thought files, and in memory where
    #: those are.
    self.tickets = Desk(self.thoughts.root / "tickets"
                        if self.thoughts.root is not None else None)
    #: One row per `ticket` event, for the run record and the observatory.
    self.ticket_events: list[dict] = []
    #: WHY THE MIND IS BEING CONSULTED (issue #221): the map row that asked
    #: (`event`, `kind`, `value`), the once-per-life `bootstrap`, or the
    #: `loop` reaching its decision branch where there is no map. Shown as
    #: `askedBy` -- until #221 every ask looked the same from inside.
    self._asked_by: dict = {"event": "loop"}
    #: The pack at the moment an abort was decided, so what STOPPING COST can
    #: be reported rather than assumed. An agent that aborts everything is
    #: not being careful, it is being useless, and this is the number that
    #: says which -- the trip home is real energy and the issue asks for it
    #: by name.
    self._abort_from_wh = 0.0
    #: Is an errand actually running? What tells the seam to INTERRUPT rather
    #: than queue -- a hazard row firing between errands is an ordinary
    #: queued action and always was.
    self._in_errand = False
    #: ...and WHICH one, for the question, the narration and the record. An
    #: interrupt that could not name what it was interrupting would be a
    #: question no model could answer well.
    self._errand_name = ""
    self.state: State = "EXPLORE"
    # A QUEUE, not a flag: "two jobs with a charge in between" is what a
    # day is, and a boolean cannot express it.
    self.errands: list = list(errands) if errands is not None else []
    self.errand_results: list[dict] = []
    self.charging_now = False
    self.tool_powered = False
    self.charge_cycles = 0
    self.swaps_done = 0
    self.tool_powered_s = 0.0
    self.log: list[str] = []
    self.status = ""                    # the latest _say message, bare
    #: THE OTHER ROBOTS' lifecycles (issue #167), set by `pair.build_pair`.
    #: Read by `others_context` for what each broadcasts, and by nothing
    #: that decides: what one robot does about another is its mind's.
    self.peers: list = []
    self.robot_name = robot_display_name(robot_name)
    #: THE GAME this robot is in, if any (issue #167): the referee activity
    #: the pair attached. Read for its clock and by the sampler; never by
    #: anything that decides.
    self.game = None
    #: The world's activities, when `run_demo` or a pair attached them
    #: (issue #167); the recorder reads them, and `cage` finds the mouse's
    #: state machine in them (issue #226). None on a bare lifecycle.
    self.activities = None
    self.encounters = None
    #: THIS robot's root body name, the key of everything it puts on the
    #: wire (issue #167): `pluggybot` for the first, `r2_pluggybot` for the
    #: second.
    self.root = handle.root
    # DEATH AND RESET (issue #107). `dead` is the cause of the current death
    # or None; `deaths` and `resets` are the day's record of both; the
    # survival clock runs from mission start or the last reset. Typed events
    # (`death`, `reset`) go to `on_event` for the recorder and the publisher,
    # exactly as a board's or the ledger's do.
    # ⚠ MORTALITY IS OPT-IN, on exactly the terms job offers and hunger are
    # (issue #107): it changes what a mission IS, so every existing demo,
    # mission test and recording must read as it did unless somebody asks
    # for it by name. Default: whether anybody could do something about a
    # death -- a served world has an inbox and an admin behind it; a test,
    # a spike and a filmstrip do not.
    #
    # ⚠ AND THE DEFAULT IS NOT FUSSINESS. On a demo cell the pack reaches
    # ZERO mid-errand as documented behaviour (`needs_charge` is checked
    # between errands, never inside one) and the robot then limps to the
    # rack and carries on. Made mortal, a recording dies mid-errand and
    # ends as a fixture describing a robot that is not there.
    self.mortal = bool(inbox is not None) if mortal is None else bool(mortal)
    #: AND HOW LONG IT LIES THERE BEFORE STANDING ITSELF UP (issue #143).
    #: None -- the default -- is the old behaviour exactly: a dead robot
    #: waits for a person, for ever if need be. `serve.py` sets
    #: `RESTART_AFTER_S`.
    #:
    #: WHY IT EXISTS: on `autonomous` the robot dies most days (A0: four in
    #: five), and a deployed world whose robot lies on the floor until a
    #: human notices is not a world anybody can watch.
    #:
    #: ⚠ AN AUTO-RESTART IS NOT AN INTERVENTION, and that is the single
    #: load-bearing line of the whole feature. A run with a non-empty
    #: `interventions` array is EXCLUDED from survival statistics
    #: (Evaluation.md §5), because an admin's hand contaminates a survival
    #: number. This is world behaviour on a timer -- if it wrote an
    #: intervention, every deployed run and every multi-life run would be
    #: silently disqualified, and the exclusion would be invisible because
    #: it is SUPPOSED to be there. `interventions` stays for the human.
    #:
    #: ⚠ AND IT IS NOT #136's TRUE DEATH EITHER. This KEEPS the volume, so
    #: the next life reads its predecessor's `History.md` death line on
    #: every decision -- which is the whole of what dying costs. True death
    #: archives the volume and starts a new robot. Do not collapse them.
    self.restart_after_s = (None if restart_after_s is None
                            else float(restart_after_s))
    #: ...and how long a TOOL lies lost before the world puts it back
    #: (issue #347, `LOST_TOOL_S`). The world's clock, not this robot's: a
    #: pair ticks it on the first robot's seam alone (`pair.build_pair`).
    self.lost_tool_after_s = (None if lost_tool_after_s is None
                              else float(lost_tool_after_s))
    #: ...and what a NEW robot starts with after a true death (issue #419,
    #: `STARTING_POINTS`): None -- the default -- is zero, as every robot
    #: started before; `serve.py` sets it. Granted by the world, never
    #: earned (`Ledger.archive`'s `granted`).
    self.start_points = int(start_points) if start_points else 0
    #: module -> the sim time it was first seen lost, this spell
    self._lost_since: dict[str, float] = {}
    self._next_lost_check = 0.0
    #: every `reset_tool` event the world's own hand emitted
    self.tools_returned: list[dict] = []
    self.dead: dict | None = None
    self.deaths: list[dict] = []
    #: TRUE deaths: the hearts ran out, the volume was archived and a new
    #: robot started (issue #136). A different event from an ordinary death
    #: and never summed with one -- see `_true_death`.
    self.true_deaths: list[dict] = []
    self.resets: list[dict] = []
    #: EVERY TIME AN ADMIN REACHED IN (issue #119; docs/Evaluation.md §5).
    #: A run with anything in here is not a survival data point -- this is
    #: what says so. Kept apart from
    #: `resets` because a reset is the one intervention that is sometimes
    #: NOT one: standing a DEAD robot up is a rescue.
    self.interventions: list[dict] = []
    self.survival_since = 0.0
    self.home_pose: tuple[float, float, float] | None = None
    self.on_event: list = []
    self._tilted_since: float | None = None
    #: What was true when the chassis passed TOPPLE_TILT_RAD (`_moment`),
    #: for the `stuck` death TOPPLE_HOLD_S later to carry (issue #362).
    self._fall: dict | None = None
    self._next_death_check = 0.0
    #: ⚠ A STAND-UP STEPS THE SIM, AND THE RESTART SEAM IS ON EVERY STEP
    #: (issue #143). `MissionRunner.start_at` ends with a one-second settle
    #: drive, so `stand_up` re-enters `_power_step` -> `_restart_step` while
    #: `dead` is still set and the clock is still up, and the second call
    #: does it again: measured as a RecursionError, not a slow leak. The
    #: guard covers the ADMIN path too -- an admin resetting a robot whose
    #: timer had already expired would otherwise have the timer fire inside
    #: the settle drive of their own reset.
    self._standing_up = False
    self._end_run = False
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
    self.charging_now = self.body.charging()
    self.tool_powered = self.body.tool_powered(self.module)
    if self.tool_powered:
      self.tool_powered_s += dt
    # The depth camera streams whenever the map is built (issue #34): a
    # load like the module's, drawn only where the sensor is on.
    self.battery.update(self.data, dt, charging=self.charging_now,
                        tool_w=(MODULE_IDLE_W if self.tool_powered else 0.0)
                        + (self.depth_camera.draw_w if self.depth_camera is not None
                           else 0.0))
    self._screen_step()
    self._death_step()
    self._restart_step()

  def _near_field_step(self) -> None:
    """One depth frame, at the sensor's rate, to its two consumers.

    The MAP takes the room, placed by the BELIEVED pose (dead reckoning,
    the axle midpoint) exactly as the occupancy grid places a LIDAR scan:
    the sensor never learns where the robot is, and drift smears the map
    honestly. The DRIVE takes the peers (issue #328) -- the same frame's
    returns off another robot's body, in the robot's own frame, where no
    pose belief can smear them and nothing has to be broadcast for them to
    be true."""
    if self.data.time < self._next_near_field:
      return
    self._next_near_field = float(self.data.time) + nf.PERIOD
    frame = self.depth_camera.frame(self.data)
    if self.body.level():      # the grid's rule (issue #339): level, or no map
      self.near_field.update(self.body.pose, frame.points)
      self.near_field_frames += 1           # frames FOLDED IN, as the summary says
    # ...and the peers that same frame saw, which the map is not told about
    # and the drive is (issue #328). One frame, two consumers, the split
    # `Lidar.scan_split` already makes one sensor along.
    # ⚠ BUT NOT A ROBOT LYING DOWN (issue #365). The hold is for the one
    # obstacle that MOVES, and one on the floor will not until it is stood
    # up: MEASURED, a detour that runs straight at a robot face-down and
    # turns at its disc's edge is held 0.52 m short, and stands there until
    # the drive times out. The planner routes round that body instead
    # (`keep_clear`), and the LIDAR's front stop and the bumper still see it.
    peers = frame.peers
    down = [p.body.geom_ids for p in self.peers if p.down()]
    if down and len(peers):
      peers = peers[~np.isin(frame.peer_geoms, np.concatenate(down))]
    self.body.watch_for_peers(peers)

  # ---- death (issue #107) --------------------------------------------------

  @property
  def survival_s(self) -> float:
    """Sim seconds awake since mission start or the last reset."""
    return float(self.data.time) - self.survival_since

  def _death_step(self) -> None:
    """THE DEATH SEAM: on every physics step, whatever phase is driving.

    A flat death is the pack REACHING zero (docs/Evaluation.md §3), not the
    run ending on it -- the motors do not stop at 0 Wh and every mission
    guard is checked between errands, so the first committed result set had
    a day that hit zero inside an errand, docked on nothing and ended "day
    over". The moment is recorded here; what the body does next is the
    errand's business until it returns. A `stuck` death by toppling is the
    torso past TOPPLE_TILT_RAD for the body's `stuck_after_s`: the rover's
    TOPPLE_HOLD_S, or a get-up's budget for a body that rights itself.
    """
    if (not self.mortal or self.dead is not None
        or self.data.time < self._next_death_check):
      return
    self._next_death_check = self.data.time + DEATH_CHECK_S
    if self.battery.empty:
      self._die("flat", "the pack reached zero")
      return
    # ⚠ THE MIND STOPPED BEING CONSULTED (issue #127), which is a death of
    # the same KIND as a flat pack -- an agent that mapped away every `ask`
    # row has traded the one capability this project exists to study for a
    # state machine. Measured rather than prevented: a map that could not
    # remove its own `ask` row would be a rail, and the configuration being
    # the agent's is the whole point.
    #
    # ⚠ ARMED ONLY WHERE THERE IS A MAP. Without one the loop asks after
    # every action and no agent can stop it, so a death here could only ever
    # be a dead endpoint -- the box's failure booked in the column the agent
    # is judged on. `_last_ask_t` is stamped by the ASK, not by the answer,
    # for the same reason: a mind consulted through an outage is still being
    # consulted. See `UNMINDED_AFTER_S`.
    if (self.event_map is not None
        and self.data.time - self._last_ask_t >= UNMINDED_AFTER_S):
      quiet = self.data.time - self._last_ask_t
      # ...AND WHAT THE LIST SAID ABOUT IT (issue #317). "My own map stopped
      # consulting me" was the same sentence for three different mistakes --
      # a list never written, a list written with no `ask` in it, and an
      # `ask` on an event that never came round -- and History is the one
      # place a later life reads what happened to this one. A fact about the
      # configuration, never a verdict on it: `events.silence` is the wording
      # and the argument.
      self._die("unminded", f"nothing has asked me anything for {quiet:.0f} s "
                            f"-- {ev.silence(self.event_map)}")
      return
    tilt = self._chassis_tilt()
    if tilt < TOPPLE_TILT_RAD:
      self._tilted_since = self._fall = None
      return
    if self._tilted_since is None:
      self._tilted_since = float(self.data.time)
      # ...AND WHAT IT WAS DOING, READ AS IT FELL (issue #362): the death is
      # TOPPLE_HOLD_S later, by which time the errand has had two seconds
      # to react to the fall -- a failed pick lowers the lift and moves on.
      self._fall = self._moment()
    elif self.data.time - self._tilted_since >= self.body.stuck_after_s:
      # ...a body that gets itself up (issue #387) is `stuck` only once its
      # get-up has had its measured budget and it is still down
      self._die("stuck", (f"fell and could not get up in {self.body.stuck_after_s:.0f} s "
                          if self.body.rights_itself else "knocked over ")
                + f"({math.degrees(tilt):.0f} deg from upright)", at=self._fall)

  def _chassis_tilt(self) -> float:
    """Radians between the chassis's up axis and the world's."""
    w, x, y, z = self.body.orientation()
    up_z = 1.0 - 2.0 * (x * x + y * y)      # R[2][2] of the root quaternion
    return math.acos(max(-1.0, min(1.0, up_z)))

  def down(self) -> bool:
    """Lying on the floor: the chassis past TOPPLE_TILT_RAD, the topple
    death's own test, read as an IMU reads it -- from the moment of the
    fall rather than the death two seconds later, and on a world where
    nothing dies as much as on one where it can."""
    return self._chassis_tilt() >= TOPPLE_TILT_RAD

  def keep_clear(self, seen_by: Body | None = None) -> tuple:
    """Where the OTHER robots keep clear of this one (`Body.others`): its
    reported pose, or -- lying down -- a `KeepClear` round the middle of its
    body, placed where `seen_by`'s own sensors would put it
    (`Body.as_seen`).

    ⚠ A ROBOT ON THE FLOOR IS AVOIDED WHERE IT LIES, NOT WHERE IT SAYS IT IS
    (issue #365; SimNotes, "A robot lying down was avoided where it said it
    was"). The errand it fell in runs on until it returns, and a wheel
    turning in the air is travel to the reckoner: MEASURED up to 2.2 m of
    reported pose off the body in 10 s, by how it lies and which way the
    wheels were told to turn. A fallen robot is not a network fact that
    moves -- it cannot report itself, and a real robot would see a
    robot-shaped lump -- so the TRUE body is fair to use, placed as a lump
    in a depth image would be. Standing up (`start_at`, a warp that resets
    the reckoner) hands it back to the reported pose.

    Either way it says WHO it is, and standing or resting, whether it lies
    down to rest (issue #415): its posture rides the wire, and a robot
    resting across the other's way is asked to make way (`make_way`)."""
    if not self.down():
      x, y = self.body.pose_xy()
      return KeepClear(x, y, root=self.root, resting=bool(self.body.resting))
    x, y = self.body.footprint_centre()
    if seen_by is not None:
      x, y = seen_by.as_seen(x, y)
    return KeepClear(x, y, down=True, root=self.root)

  def make_way(self, route, by: str) -> bool:
    """The other robot (`by`, its root) asks this one off `route`, its way,
    which this one's body cuts (issue #415): the body's rule
    (`Body.make_way`, `legs/way.py`), said as it starts. A dead robot is
    not asked -- it lies where it died until it is stood up."""
    if self.dead is not None:
      return False
    # ...nor the HIDER in a game (issue #404): stood up and stepped out of
    # the seeker's way, it would be the network handing the seeker the find
    game = getattr(self, "game", None)
    if (game is not None and game.over_at is None and game.hider is not None
        and game.hider.root == self.root):
      return False
    was = self.body.making_way
    if not self.body.make_way(route, by):
      return False
    if was is None:
      other = self._peer(by)
      self._say(f"MAKE WAY: lying across {other.robot_name if other else by}'s "
                "way -- standing up to step aside")
    return True

  def _aside_step(self) -> None:
    """A step aside the body made for the other robot (issue #415), said
    and remembered once it is over: the body moved, and the mind did not
    decide it -- History is where it reads that it did."""
    if self.body.asides == self._asides_seen:
      return
    self._asides_seen = self.body.asides
    rec = self.body.last_aside
    if rec is None:
      return
    other = self._peer(rec["by"])
    who = other.robot_name if other is not None else rec["by"]
    moved = math.dist(rec["from"], rec["at"])
    if rec["why"] == "aside":
      line = (f"stood up and stepped {moved:.1f} m aside for {who}, whose way "
              "I was lying across")
    else:
      line = (f"stood up to step aside for {who}, whose way I was lying across, "
              f"and stopped {moved:.1f} m on: {ASIDE_ENDED.get(rec['why'], rec['why'])}")
    self._say(f"MADE WAY: {line}")
    self._remember(line)

  _asides_seen = 0

  def _lean(self) -> tuple[float, float | None]:
    """Which way the chassis leans: (degrees from upright, the direction
    its top leans toward in its OWN frame -- 0 forward, 90 its left, -90
    its right, 180 back). The robot's frame, because that is what
    classifies a fall: onto the fork, back over the caster, over a wheel.

    ⚠ NO DIRECTION WHILE LEVEL (the map's own rule, `Body.level_tilt_rad`):
    MEASURED, a healthy robot reads 0.02 deg standing and at most 0.25
    driving, and the direction of that is noise -- a flat death read it as
    "back", which a count of falls by direction would have counted."""
    w, x, y, z = (float(v) for v in self.body.orientation())
    tilt = self._chassis_tilt()
    if tilt < self.body.level_tilt_rad:
      return math.degrees(tilt), None
    # the world's up axis in the chassis frame is R's third ROW; the top
    # leans away from it
    ux, uy = 2.0 * (x * z - w * y), 2.0 * (y * z + w * x)
    return math.degrees(tilt), math.degrees(math.atan2(-uy, -ux))

  def _moment(self) -> dict:
    """Where the robot is and what it is doing, for a death to carry (issue
    #362): seven topples in a week could not be read because the event said
    only the cause. The TRUE pose beside the believed one (drift is a
    suspect), the lean, the state, what is on the fork and what every axis
    is commanded to, the errand and the procedure step running, the swap's
    bay, and the nearest peer -- measured between the chassis, as the
    encounters measure it. Nothing here decides anything.

    ⚠ IT NEVER RAISES: it runs on the physics seam inside `_die`, and a
    diagnostic that threw there would lose the death it describes."""
    try:
      return self._read_moment()
    except Exception as e:                        # noqa: BLE001
      return {"t": round(float(self.data.time), 3),
              "error": f"{type(e).__name__}: {e}"[:200]}

  def _read_moment(self) -> dict:
    from pluggybot.procedure.axes import setpoints
    tx, ty, tyaw = self.body.true_pose()
    bx, by, byaw = self.body.pose
    tilt, toward = self._lean()
    carried = carrying(self) or None
    errand = self._errand_now
    at: dict = {
      "t": round(float(self.data.time), 3),
      "pose": {"x": round(tx, 3), "y": round(ty, 3),
               "yawDeg": round(math.degrees(tyaw), 1)},
      # the reckoner's heading is never wrapped; this is, to compare
      "believed": {"x": round(bx, 3), "y": round(by, 3),
                   "yawDeg": round(math.degrees(math.atan2(math.sin(byaw),
                                                           math.cos(byaw))), 1)},
      "tilt": {"deg": round(tilt, 1),
               "towardDeg": None if toward is None else round(toward, 1),
               "toward": None if toward is None else lean_word(toward)},
      "state": self.state, "status": self.status[:200],
      "carrying": carried, "setpoints": setpoints(self, carried),
      "errand": None if errand is None else
        {"name": errand.name, "task": errand.task_id, "module": errand.module},
      "step": None if self.step_now is None else dict(self.step_now),
      "swapping": self._bay_name(self.body.swapping_at),
    }
    if self.peers:
      me = self.body.root_xy()
      def apart(o) -> float:
        return float(math.hypot(*(o.body.root_xy() - me)))
      near = min(self.peers, key=apart)
      # ...and whether it is DEAD, which its state alone does not say: a
      # robot knocked over mid-errand keeps the errand's state until that
      # errand is closed, and a robot driving into one lying on its side
      # is #365's collision
      at["peer"] = {"name": near.robot_name or near.root, "robot": near.root,
                    "distanceM": round(apart(near), 3), "state": near.state,
                    "dead": near.dead["cause"] if near.dead else None}
    return at

  def _bay_name(self, station_y: float | None) -> str | None:
    """A bay by the module it is for (`rack_inventory`)."""
    if station_y is None:
      return None
    return next((m for m, i in self.rack_inventory.items()
                 if abs(STATION_YS[i] - station_y) < 1e-6), f"y={station_y:.3f}")

  def _die(self, cause: str, why: str, at: dict | None = None) -> None:
    """Record a death: narrated, remembered in the file the robot cannot
    edit (Evaluation.md §6 -- the cheapest real cost there is), and put on
    the wire as a `death` event. Idempotent: the first cause stands.

    `at` is where it was and what it was doing (`_moment`), read now unless
    the caller read it earlier -- a topple reads it as the robot fell."""
    if self.dead is not None:
      return
    assert cause in DEATH_CAUSES, cause
    t = float(self.data.time)
    at = at if at is not None else self._moment()
    self.dead = {"t": round(t, 3), "cause": cause, "why": why,
                 "survivalS": round(self.survival_s, 3)}
    # ⚠ A DEATH COSTS A HEART, FLATLY (issue #136). One, at five hearts and
    # at one, and nothing anywhere scales with what is left -- see
    # `ledger.HEARTS` for why an escalating cost was rejected as a forcing
    # function. `None` where there is no ledger, which is every test and
    # demo that does not attach one, and the day reads exactly as it did.
    #
    # ⚠ CHARGED BEFORE THE RECORD IS COPIED, so `deaths` and `dead` carry the
    # same dict. They diverged for one commit and a test caught it: the copy
    # is what a run record reads, and a death that cost a heart in one place
    # and not the other is the kind of disagreement nobody looks for.
    hearts = self.ledger.lose_heart() if self.ledger is not None else None
    self.dead["hearts"] = hearts
    self.deaths.append(dict(self.dead))
    self._say(f"DEAD ({cause}): {why}"
              + (f" -- {hearts} heart{'s' if hearts != 1 else ''} left"
                 if hearts is not None else ""))
    self._remember(f"died -- {why} -- after {self.survival_s:.0f} s awake "
                   f"({cause})"
                   + (f"; {hearts} lives left" if hearts is not None else ""))
    self._emit({"type": "death", "t": round(t, 3), "robot": self.root,
                "cause": cause, "why": why,
                "survivalS": round(self.survival_s, 3),
                "deaths": len(self.deaths),
                **({"hearts": hearts} if hearts is not None else {}),
                "at": at})
    if hearts == 0:
      self._true_death(t)
    elif cause == "unminded":
      # ...AND SILENCE IS FOLLOWED BY ONE CONSULT (see `_consult`); a true
      # death owes nothing -- the next robot has the bootstrap.
      self._consult = {"event": "unminded", "note": UNMINDED_NOTE}
      ev.save(self.thoughts.store, self.event_map, self.overseer.origin, owed=True)

  def _true_death(self, t: float) -> None:
    """The last heart is gone: archive the volume and start a new robot.

    The honest version of "it does not come back" (Evaluation.md section 6).
    What is archived is what made it THAT robot -- the balance and its
    earnings log, and the two thought files the robot and the system wrote.
    ⚠ `Main.md` and `Goals.md` are a HUMAN's and are NOT archived: they are
    what a person put on the volume to say who this robot is and what it is
    for, and a world that wiped them would need somebody to type them back in
    before it could run again.

    ⚠ DISTINCT FROM #143's AUTO-RESTART, and the difference is the whole
    cost of dying. An ordinary death KEEPS the volume, so the next life reads
    its predecessor's death line on every decision -- the same robot has to
    live with having died. This is the one that does not.

    ⚠ AND THE NEW ROBOT STARTS SOLVENT: a full set of hearts, no carried
    fraction of a point, and `start_points` to explore with (issue #419;
    zero where none is set). A death may never make the next life
    unwinnable, and a robot that inherited its predecessor's debt would die
    of it before it had earned anything.

    ⚠ ...AND KNOWING NOTHING OF THE WORLD BUT ITS DOCK (issue #419): the map
    and the places the robot before found are gone with it (`Body.
    forget_world`), and the new one explores from there.
    """
    archived = self.ledger.archive(start=self.start_points)
    # ...and gives back any role it took in a game still on offer (issue
    # #404): the next robot did not take it
    if self.tasks is not None:
      for task in self.tasks.offered():
        if task.roles and task.role_of(self.root):
          self.tasks.release_role(task.id, self.root)
    self.body.forget_world()
    self.floor_explored = False
    self.blacklist = set()
    if self.thoughts is not None:
      archived["thoughts"] = self.thoughts.archive()
    if self.metabolism is not None:
      self.metabolism.disarm()
    # ...AND ITS LIST OF RULES (issue #337): the event map is the robot's
    # across restarts and stand-ups, and this is the one thing that ends
    # it. The kept file goes aside like the rest (`event_map.1.json`), the
    # next robot starts from the origin, and nothing the old list armed or
    # queued runs for it. ⚠ THE FILE GOES ON EVERY WORLD, a world with no
    # map included: a scripted or origin-`none` day that ended a robot
    # would otherwise leave its list for the next day with a map to
    # restore as the new robot's own.
    aside = self.thoughts.store.archive(ev.MAP_FILE)
    self._consult = None
    if aside is not None:
      archived["eventMap"] = {"archivedAs": aside}
    if self.event_map is not None:
      archived["eventMap"] = {**self.overseer.start_over(t), "archivedAs": aside}
      self.queued_row = None
      self._interrupt_pending = None
      self._occurred.clear()
      self.event_clock = ev.EventClock()
    self.true_deaths.append({"t": round(t, 3), **archived})
    # ⚠ THE NEXT ROBOT IS ASKED (issue #303): the rule is "until the mind
    # has answered for itself", and the mind that answered is archived with
    # its list -- "with its eyes open" cannot be said of a choice made by
    # the robot before. Spent, the bootstrap would leave an `unseeded`
    # successor's empty list unconsulted: `unminded` at 1800 s having made
    # no decision, the state this seam exists to prevent.
    self._minded = False
    given = (f" but {self.start_points} points to explore with"
             if self.start_points else "")
    self._say(f"TRUE DEATH: out of hearts. Everything this robot earned, "
              f"wrote and mapped is archived or gone; robot "
              f"#{archived['generation'] + 1} starts from nothing{given}.")
    # ⚠ The line goes in the NEW robot's History, which is empty by now --
    # so the first thing it ever reads about itself is that it is not the
    # first. That is the inheritance, and it is the only one.
    self._remember(f"I am the {_ordinal(archived['generation'] + 1)} robot to "
                   "run here. The one before me ran out of lives; what it "
                   "knew went with it, its map and the places it had found "
                   f"among it{'. I start with ' + str(self.start_points) + ' points' if self.start_points else ''}.")
    self._emit({"type": "true_death", "t": round(t, 3), "robot": self.root,
                "generation": archived["generation"],
                "archived": archived.get("archived", {}),
                # what the new robot was given to start (issue #419); absent
                # where nothing was, as every true death before it
                **({"granted": self.start_points} if self.start_points else {})})

  @property
  def reset_in_s(self) -> float | None:
    """Sim seconds until the robot stands itself up, or None (issue #143).

    None means there is nothing to count: the robot is alive, or no timer
    was configured. Never negative -- a countdown that went past zero would
    say the restart had not happened when it is a step away.
    """
    if self.dead is None or self.restart_after_s is None:
      return None
    left = (self.dead["t"] + self.restart_after_s) - float(self.data.time)
    return round(max(0.0, left), 1)

  def _restart_step(self) -> None:
    """The dead robot's own clock (issue #143).

    On the PHYSICS seam beside `_death_step`, not in `_wait_dead`: a flat
    death is caught the moment it happens, INSIDE an errand or not, and the
    errand goes on driving until it returns. The clock has to start where
    the death did rather than where the loop next looks.

    ⚠ IT FIRES WHEREVER IT LANDS, A SEATED MODULE INCLUDED (issue #348).
    It used to wait for the errand to put a seated tool down (#143), which
    is "wait until the errand returns": a robot inside an errand that never
    returns stayed down for good (#311 was the parked half of that). Now
    the day loop closes the routine it lands in (`_until_stood_up_routine`)
    before that routine runs another line, so nothing is left mid-stow to
    disagree about what the fork holds, and the tool comes home with the
    robot (`_stand_up`). The admin doors keep #311's rule; they are only
    ever reached between errands.
    """
    if (self.dead is None or self.restart_after_s is None
        or self.home_pose is None):
      return
    if self._standing_up:
      return
    if float(self.data.time) - self.dead["t"] < self.restart_after_s:
      return
    if self.up_pose() is None:
      return                             # every start is under a robot: next step
    self.stand_up(AUTO_RESTART_BY, auto=True)

  def up_pose(self) -> tuple[float, float, float] | None:
    """Where a stand-up puts this robot (issue #387): its own start pose,
    or -- with another robot's body on it -- the first of the others' start
    poses that is clear, each a pose the world was commissioned with; None
    while every one is taken. Read off the other robots' TRUE bodies: the
    stand-up is the world's own hand, and it can see what it puts a robot
    down on. With two robots one start is always clear."""
    if self.home_pose is None:
      return None
    for pose in [self.home_pose] + [p.home_pose for p in self.peers
                                    if p.home_pose is not None]:
      if all(math.hypot(p.body.footprint_centre()[0] - pose[0],
                        p.body.footprint_centre()[1] - pose[1]) >= START_CLEAR_M
             for p in self.peers):
        return tuple(pose)
    return None

  @property
  def expects_work(self) -> bool:
    """Whether work can still ARRIVE when the loop has nothing to do -- what
    the stand-by branch of the day keys on. Follows `producer` unless set:
    a pair shares one board with the producer on the FIRST robot's seam
    (issue #167), so the second robot has no producer and a board that
    grows anyway. Measured on the pair fixture: keyed on `producer`, the
    hider called its day complete at t = 146 s, mid-game, and every carry
    the cadence offered after that went to the other robot by default."""
    if self._expects_work is not None:
      return self._expects_work
    return self.producer is not None

  @expects_work.setter
  def expects_work(self, value: bool) -> None:
    self._expects_work = bool(value)

  def _emit(self, message: dict) -> None:
    for hook in list(self.on_event):
      hook(dict(message))

  def rack_distance(self, px: float, py: float) -> float:
    """How far a point is from the racks: the nearer of the rack's prior
    and the middle of the built-tool rail beside it on the same board
    (`legs.rack.BUILT`, #407), whose far bay is 1.25 m along it."""
    r = self.body.rack_prior
    d = math.hypot(px - r.x, py - r.y)
    if self.has_built_rack:
      from pluggybot.legs import rack as legs_rack
      ys = legs_rack.BUILT.bays
      bx, by = r.to_world(0.0, (ys[0] + ys[-1]) / 2)
      d = min(d, math.hypot(px - bx, py - by))
    return d

  def peer_at_the_bay(self, station_y: float) -> tuple[str, float] | None:
    """Whose robot stood on the standoff this swap needed, and how far off
    (issue #313) -- or None where the bay was not given up for one.

    The RULE has one home, `Body.peer_on_the_goal`: this asks only
    whether the last attempt refused for that reason, and puts a name to
    it. `RACK_CLEAR_M` is the other half of the same problem and does not
    cover this one: it moves a robot that is STANDING BY, and the robot
    in the way here is charging or swapping, which is minutes at a time on
    the one bay of each that the pair shares.
    """
    near = self.body.peer_at_bay_m
    if near is None:
      return None
    sx, sy, _ = self.body.bay_standoff(station_y)
    return self._nearest_peer(sx, sy, near)

  def pick_failure(self, tool: str, station_y: float, why: str = "") -> str:
    """Why a pick left `tool` off THIS robot's fork, as one clause the robot
    reads back -- the errand's History line and a `fetch` step's reason say
    the same thing (issue #264). Most actionable first: whoever holds it
    (carrying is the others' PUBLIC surface), whoever stood in the way,
    then the bay itself -- a miss the robot can retry, told apart from an
    approach that never got there, and from a tool that is nowhere."""
    if self.body.module_state(tool)["on_fork"]:
      # ...its OWN fork first: half-seated, it fell through to "not on its
      # bay, and no robot is carrying it" while it rode this very fork
      return (f"{tool} came onto the fork but did not seat (no power "
              "contact); stow it and fetch it again")
    holder = next((p for p in self.peers if carrying(p) == tool), None)
    if holder is not None:
      return f"{tool} is on {holder.robot_name or holder.root}'s fork"
    blocked = self.peer_at_the_bay(station_y)
    if blocked is not None:
      return self.held_for(blocked)
    if self.body.module_state(tool)["hung"]:
      if why == "no-route":
        return ("there was no route to where the fork lines up with its bay, "
                "so no pick was tried; it is still hanging there")
      if why == "blocked":        # `refine_standoff` gave up (issue #339)
        return ("the way back to where the fork lines up with its bay was "
                "blocked, so no pick was tried; it is still hanging there")
      how = PICK_WHY.get(why)
      return ("the pick missed and it is still on its bay"
              + (f" ({how})" if how else ""))
    return "it was not on its bay, and no robot is carrying it"

  def held_for(self, blocked: tuple[str, float]) -> str:
    """Who held a bay this robot gave up on, and for how long it was
    waited for (issue #346) -- one clause, for a pick, a stow or a charge."""
    w = self.last_bay_wait
    said = (f"{blocked[0]} was {posture(self.peers, blocked[0])} "
            f"{blocked[1]:.2f} m from the bay, nearer than the planner may "
            "route")
    # ...only the wait that ended THIS failure, at this sim instant: an
    # older one would claim a wait that never happened for this bay
    if (w is None or w.get("who") != blocked[0] or not w.get("why")
        or w.get("end") != float(self.data.time)):
      return said
    return said + {
      "bound": f"; I waited {w['s']:.0f} s for it to leave, 3x how long one "
               f"{w['of']} usually takes",
      "interrupted": f"; I waited {w['s']:.0f} s, until my own rule "
                     "interrupted the wait",
      "pack": f"; I waited {w['s']:.0f} s, until the pack reached the "
              "reserve"}[w["why"]]

  def peer_at(self, wx: float, wy: float) -> tuple[str, float] | None:
    """The same question asked of a POINT, for the approach that keeps no
    verdict of its own (`go_charge_routine`). Asked as the narration is
    written, which is the same sim instant the drive gave up in."""
    near = self.body.peer_on_the_goal(wx, wy)
    return None if near is None else self._nearest_peer(wx, wy, near)

  def drive_why(self, wx: float, wy: float, record: dict | None = None) -> str:
    """Why the drive to (wx, wy) gave up, as the one clause every failure
    line that follows a drive ends with (issue #350): `gave_up` off the
    body's record -- or off `record`, one its caller kept (a press's try,
    #439) -- with the other robot NAMED. Only a record of a drive to THIS
    goal is read -- any other is stale, and a stale cause names the wrong
    failure."""
    rec = self.body.last_drive if record is None else record
    if (rec is None or not rec["why"]
        or math.hypot(rec["goal"][0] - wx, rec["goal"][1] - wy) > 1e-6):
      return "the drive gave up"
    who = "the other robot"
    if rec["why"] == "peer" and self.peers and rec.get("peerXY"):
      # whose: the peer nearest the body the drive recorded
      who = self._nearest_peer(*rec["peerXY"], 0.0)[0]
    return self.body.gave_up(rec, who)

  def _nearest_peer(self, wx: float, wy: float,
                    near: float) -> tuple[str, float] | None:
    """`near` is the DISTANCE the rule answered; this only picks whose it
    is -- the nearest peer to the point, by the reported pose the rule
    measured against."""
    if not self.peers:
      return None
    who = min(self.peers,
              key=lambda o: math.hypot(o.body.pose_xy()[0] - wx,
                                       o.body.pose_xy()[1] - wy))
    return (who.robot_name or who.root), near

  def _clear_rack_routine(self) -> Routine:
    """Stand by away from the racks: within `RACK_CLEAR_M` of either, drive
    back to the start pose; anywhere else, stay put. True if the robot ends
    clear of the rack.

    ⚠ DONE AT THE RACK MEANS GONE (issue #346), so the drive's answer is
    READ. It used to be thrown away: Rowan narrated "clearing the rack"
    and stood 0.08 m from the charge bay for 23 minutes, through all four
    of Luca's charge attempts, and Luca died flat for good. A clear that
    did not arrive tries the nearest other spots outside the radius
    (`CLEAR_SPOTS`) and says in History where it ended if none worked.
    What counts is being OUT OF REACH, not reaching a point: a drive that
    stagnated 0.3 m short but outside the radius has done the job.
    """
    px, py, _ = self.body.pose
    if self.home_pose is None or self.rack_distance(px, py) >= RACK_CLEAR_M:
      return True
    hx, hy = self.home_pose[0], self.home_pose[1]
    self._say(f"standing by: clearing the rack for the others -- back to "
              f"({hx:.1f}, {hy:.1f})")
    yield from self.body.go_to_routine(hx, hy)
    tried = [(hx, hy)]
    for spot in self._clear_spots()[:CLEAR_SPOTS]:
      px, py, _ = self.body.pose
      if self.rack_distance(px, py) >= RACK_CLEAR_M:
        break
      self._say(f"RACK: still {self.rack_distance(px, py):.1f} m from the "
                f"rack after heading for ({tried[-1][0]:.1f}, "
                f"{tried[-1][1]:.1f}) -- trying ({spot[0]:.1f}, {spot[1]:.1f})")
      tried.append(spot)
      yield from self.body.go_to_routine(*spot, timeout=CLEAR_DRIVE_S)
    px, py, _ = self.body.pose
    d = self.rack_distance(px, py)
    if len(tried) > 1 or d < RACK_CLEAR_M:
      self._remember(
        f"cleared the rack for the others after {len(tried)} tries, at "
        f"({px:.1f}, {py:.1f})" if d >= RACK_CLEAR_M else
        f"could not get clear of the rack: tried {len(tried)} spots and "
        f"ended {d:.1f} m from it at ({px:.1f}, {py:.1f}), where I am in "
        "the way of anyone swapping or charging there")
    return d >= RACK_CLEAR_M

  def _clear_spots(self) -> list[tuple[float, float]]:
    """Other places to stand clear of the rack, nearest first: a ring just
    outside `RACK_CLEAR_M` round the rack prior, in front of the rack face
    (behind it is the wall it stands on), kept only where this robot could
    plan to right now (`Body.reachable`)."""
    r = self.body.rack_prior
    ring = RACK_CLEAR_M + 0.5
    cands = [r.to_world(ring * math.cos(a), ring * math.sin(a))
             for a in (math.radians(d) for d in range(-75, 90, 15))]
    ok = self.body.reachable(cands)
    px, py, _ = self.body.pose
    return sorted((c for c, k in zip(cands, ok)
                   if k and self.rack_distance(*c) >= RACK_CLEAR_M),
                  key=lambda c: math.hypot(c[0] - px, c[1] - py))

  def _leave_rack_routine(self) -> Routine:
    """After an interaction at the rack, get out of it -- when there is
    somebody to get out of the way FOR (issue #346). The loop calls this
    before any branch that does not itself go to the rack, so a robot that
    swapped, charged or failed there does not stand in the other one's
    approach while it thinks. Alone in the world, nothing is blocked and
    nothing moves: a single robot's day is the trajectory it always was."""
    if not self.peers:
      return
    px, py, _ = self.body.pose
    stuck = self._clear_failed_at
    if stuck is not None and math.hypot(px - stuck[0], py - stuck[1]) < CLEAR_MOVED_M:
      # ...tried from here already and could not get away (said in History
      # then): not again before every decision, which would spend minutes
      # of driving and a History line per pass on the same answer
      return
    cleared = yield from self._clear_rack_routine()
    self._clear_failed_at = None if cleared else self.body.pose[:2]

  def _rack_linger_step(self) -> None:
    """Log a robot that stays at the rack longer than what it came for
    (issue #346): within `RACK_CLEAR_M`, doing nothing AT the rack, for
    `RACK_LINGER_S`. Once per stretch; the physics seam's, once a second.
    Evidence only -- the fix is `_leave_rack_routine`, and this is what
    says whether anything still escapes it."""
    t = float(self.data.time)
    if not self.peers or t < self._next_linger_check:
      return
    self._next_linger_check = t + 1.0
    px, py, _ = self.body.pose
    if (self.state in RACK_STATES or self.dead is not None
        or self.rack_distance(px, py) >= RACK_CLEAR_M):
      self._linger_since = None
      return
    if self._linger_since is None:
      self._linger_since, self._linger_said = t, False
    elif not self._linger_said and t - self._linger_since >= RACK_LINGER_S:
      self._linger_said = True
      self._say(f"RACK: lingering {t - self._linger_since:.0f} s within "
                f"{self.rack_distance(px, py):.2f} m of the rack while "
                f"{self.state} -- in the way of the others",
                detail=f"pose ({px:.2f}, {py:.2f})")

  def _waiting_spot(self, sx: float,
                    sy: float) -> tuple[float, float] | None:
    """Where to wait for a taken standoff (issue #346): beside and behind
    it in the rack's frame (`WAIT_BACK_M`, `WAIT_SIDE_M`), out of the
    holder's lane and planner disc, nearest reachable first. With none of
    those on known floor (a map still thin round the rack), the start pose
    -- reached once already, and clear of the rack by construction; where
    the robot happens to stand is usually the holder's way out (MEASURED:
    a robot standing at the neighbouring bay stops the other's drive away
    for the errand's whole 60 s). None only with no start pose at all."""
    r = self.body.rack
    c, s_ = math.cos(r.yaw), math.sin(r.yaw)
    lx = (sx - r.x) * c + (sy - r.y) * s_
    ly = -(sx - r.x) * s_ + (sy - r.y) * c
    cands = [r.to_world(lx + back, ly + side)
             for back in WAIT_BACK_M for side in WAIT_SIDE_M + tuple(
               -v for v in WAIT_SIDE_M)]
    ok = self.body.reachable(cands)
    px, py, _ = self.body.pose
    keep = [p for p, k in zip(cands, ok) if k]
    home = None if self.home_pose is None else tuple(self.home_pose[:2])
    return min(keep, key=lambda p: math.hypot(p[0] - px, p[1] - py),
               default=home)

  def wait_bound(self, kind: str, sx: float, sy: float) -> tuple[float, str]:
    """How long a taken bay is waited for, and what it is `WAIT_OCCUPANCIES`
    times the typical occupancy of: a charge if the bay asked for is the
    charge bay OR the robot nearest it is charging (its state is its public
    surface), else a swap."""
    holder = min(self.peers, default=None,
                 key=lambda o: math.hypot(o.body.pose_xy()[0] - sx,
                                          o.body.pose_xy()[1] - sy))
    charging = kind == "charge" or (
      holder is not None and holder.state in ("GO_CHARGE", "CHARGE"))
    return ((WAIT_OCCUPANCIES * CHARGE_OCCUPANCY_S, "charge") if charging
            else (WAIT_OCCUPANCIES * SWAP_OCCUPANCY_S, "swap"))

  def _await_bay_routine(self, sx: float, sy: float, kind: str,
                         since: float, hold: bool = False) -> Routine:
    """Wait for another robot to leave a standoff this one needs (issue
    #346). True once it is free; False once the robot gave up, with why in
    `self.last_bay_wait` (who held it, for how long, and what ended the wait).

    Before this a bay somebody stood on was given up at once (#313's early
    return) and the job failed with 0 points -- 6 of 47 picks and 7 of 38
    stows on the deployed pair in one afternoon, and 11 of 30 charge
    approaches, after all four of which a robot died flat. So: back off to
    a spot out of the holder's way, say so in History, and look again every
    `BAY_POLL_S` until the bay is free or the bound (`wait_bound`, counted
    from `since`, the first time THIS interaction found it taken) runs out.
    Two more things end a PICK's wait early, inside an errand: the robot's
    own event map (a `battery_below` row interrupts, as it would the errand
    -- `interrupted()`), and the pack reaching the reserve (#315's rule: the
    waiting is code's, and code does not spend the return trip on it). A
    RETURN's wait has neither -- abort means stow, and a return given up
    leaves the tool on the fork, a procedure's `stow()` included (it runs
    inside its errand) -- and nor does a CHARGE's: charging is what either
    would ask for, and giving up on it is how the robot dies.

    `kind` is the swap's verb (`pick`, `return`) or `charge`. `hold` waits
    where the robot stands -- the quadruped's swap, at its approach's start
    a metre behind the bay, out of the holder's way already (#418).
    """
    near = self.body.peer_on_the_goal(sx, sy)
    if near is None:
      return True
    # ...keyed on the RULE, never on the name: a peer the planner routes
    # round but the lifecycle has no name for is still waited for, or the
    # swap's loop would go round at no sim time at all
    held = self.peer_at(sx, sy)
    who = held[0] if held is not None else "another robot"
    bound, of = self.wait_bound(kind, sx, sy)
    what = "the charge bay" if kind == "charge" else "the bay"
    first = self.last_bay_wait is None or self.last_bay_wait.get("since") != since
    # EVERY time, not only the first: a drive that found the bay taken again
    # stops at the edge of the holder's disc, which is its way out.
    spot = None if hold else self._waiting_spot(sx, sy)
    if first:
      self._say(f"WAIT: {who} is {posture(self.peers, who)} {near:.2f} m from "
                f"{what} -- waiting up to {bound:.0f} s"
                + ("" if spot is None else
                   f" at ({spot[0]:.1f}, {spot[1]:.1f})"))
      self._remember(f"{who} was at {what} I needed; I waited for it "
                     f"(up to {bound:.0f} s)")
      self.bay_waits += 1
      self.last_bay_wait = {"since": since, "who": who, "kind": kind,
                       "bound": bound, "of": of, "why": ""}
    if spot is not None:
      yield from self.body.go_to_routine(*spot, timeout=CLEAR_DRIVE_S)
    ended = ""
    while self.body.peer_on_the_goal(sx, sy) is not None:
      now = float(self.data.time)
      bound, of = self.wait_bound(kind, sx, sy)
      if now - since >= bound:
        ended = "bound"
      elif kind == "pick" and self._in_errand and self.interrupted():
        ended = "interrupted"
      elif (kind == "pick" and self._in_errand
            and self.battery.energy_wh <= self.low_battery_wh):
        ended = "pack"
      if ended:
        break
      yield from self.body.hold_routine(BAY_POLL_S)
    waited = float(self.data.time) - since
    self.last_bay_wait.update({"s": round(waited, 1), "why": ended,
                               "bound": bound, "of": of,
                               "end": float(self.data.time)})
    if ended:
      self._say(f"WAIT: gave up on {what} after {waited:.0f} s -- {who} "
                "was still there" + {"bound": f" (the bound is {bound:.0f} s)",
                                     "interrupted": " (my own interrupt)",
                                     "pack": " (the pack is at the reserve)"}[ended])
      return False
    self._say(f"WAIT: {what} is free after {waited:.0f} s")
    return True

  def _wait_dead_routine(self) -> Routine:
    """A dead robot with somebody who can reset it stands still and keeps
    the stream alive; the visitor step is what delivers the reset."""
    self.state = "DEAD"
    self._visitor_step()
    if self.dead is not None:
      yield from self.body.hold_routine(WAIT_FOR_WORK_S)

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
      # ...powered by THIS robot's fork (issue #167): the screen's own
      # check reads the first robot's plates, and a second robot carrying
      # the display would read it as dark.
      self.screen.sense(self.model, self.data,
                        powered=self.body.tool_powered(self.screen.module))
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
    resting = face_for(self.state, self.battery.fraction,
                       hunger=(self.metabolism.state
                               if self.metabolism is not None else ""))
    if resting != self._face_shown:
      self._face_shown = resting
      self.screen.face(*resting)

  def _say(self, msg: str, detail: str = "") -> None:
    """Narrate one line. `msg` is what the ROBOT says; `detail` is evidence.

    `msg` becomes `self.status`, which rides every telemetry frame, is
    rendered verbatim under the robot's portrait and goes to `say_hooks` (the
    event stream). So it has to be a sentence a robot could say.

    `detail` reaches the console and `self.log` and NOTHING else -- no status,
    no frame, no hook. It is where a caught exception's class and message go
    (issue #76): the operator debugging a recovered failure still wants the
    class, and a visitor reading the Thoughts tab should not be shown a
    traceback for something that did not crash.
    """
    self.status = msg
    # A second robot's lines say whose they are (issue #167); the first
    # robot's read exactly as they always did.
    who = f" {self.root}" if self.body.handle.prefix else ""
    line = f"t={self.data.time:6.1f}s{who}  bat={self.battery.fraction:5.0%}  {msg}"
    if detail:
      line = f"{line}  [{detail}]"
    self.log.append(line)
    print(line, flush=True)
    for hook in self.say_hooks:
      hook(float(self.data.time), msg)

  def _announce_constitution(self) -> None:
    change = self.thoughts.take_constitution_change()
    if change is None:
      return
    to = self.thoughts.constitution
    src = change["from"]
    was = (f"`{src['name']}` ({src['sha'][:8]})" if src.get("name")
           else f"a text the library does not hold ({src['sha'][:8]})")
    why = {"swapped": "my constitution was swapped",
           "replaced": "my constitution was replaced by the library's",
           "edited": "a hand edit of my constitution was set aside"}[change["why"]]
    self._remember(f"{why}: {was} -> `{to.name}` ({to.sha[:8]})")
    self._say(f"CONSTITUTION {change['why']}: {was} -> {to.short}"
              + (f" (the old text kept as {change['archived']})"
                 if change.get("archived") else ""))
    self._emit({"type": "constitution_changed", "t": round(float(self.data.time), 3),
                "robot": self.root, "why": change["why"],
                "from": dict(src), "to": to.as_dict(),
                **({"archived": change["archived"]} if change.get("archived") else {})})

  @property
  def _owes_for_silence(self) -> bool:
    """Is the consult owed the one kept on the volume (a heart lost to
    `unminded`)? The other is found again at every load."""
    return self._consult is not None and self._consult["event"] == "unminded"

  def _keep_map(self, msg: dict) -> None:
    """Keep the robot's list for its next run (issue #337), on every EDIT
    an answer made -- never on a true death's reset, which the new robot
    did not write and which would come back as `restored`."""
    if msg.get("why") == "edit":
      ev.save(self.thoughts.store, self.overseer.event_map, self.overseer.origin,
              owed=self._owes_for_silence)

  def _announce_map(self) -> None:
    """What came back of the robot's kept list of rules (issue #337):
    narrated where it came back, and a History line where a rule of it did
    not -- a rule dropped in silence is one the robot believes it has."""
    if self.event_map is None:
      return
    boss = self.overseer
    if boss.restored:
      n = len(self.event_map)
      self._say(f"EVENT MAP kept from before: {n} rule{'s' if n != 1 else ''}")
    if boss.dropped_at_load:
      self._say(f"EVENT MAP not kept: {'; '.join(boss.dropped_at_load)}")
      self._remember("my list of rules came back without what this world no "
                     f"longer reads: {'; '.join(boss.dropped_at_load)}")

  def _remember(self, line: str, room: int = 0) -> None:
    """Append one line to `History.md` (issue #38).

    The narrative record, and deliberately NOT the narration: `_say` fires
    dozens of times a minute and most of it is a state machine talking to
    itself. What lands here is what a person catching up would want -- the
    day starting, what the robot decided, how each finished job was judged,
    how the day ended. Written by CODE, which is the point of the file: a
    robot that could edit its own history breaks the same principle that
    stops it awarding itself points.

    `room` lets the line run that far past History's line cap, for a text
    it quotes whole (`ThoughtFiles.remember`, issue #433).
    """
    try:
      self.thoughts.remember(line, t=float(self.data.time), room=room)
    except ThoughtRefused as e:
      # History rolls rather than refusing, so this is close to unreachable
      # -- but a memory write must never be able to end a mission, and a
      # refusal nobody can see is the thing this module exists to prevent.
      self._say(f"HISTORY refused: {e}")

  def telemetry_status(self) -> dict:
    """The per-frame robot record for the telemetry recorder: lifecycle
    state, the _say narration, and the battery gauges. The bare message,
    not the formatted log line -- t and battery already ride in the frame,
    and the site wants "carrying the module", not a duplicate dashboard."""
    return {
      "state": self.state,
      "status": self.status,
      # The operator's switch (0.12.0, issue #37). In the robot's own record
      # rather than a block of its own, because it is a fact about the robot
      # at this instant -- and absent when there is no switch, which reads
      # as `llm` and is what every demo and test is.
      **({"mode": self.mode.mode} if self.mode is not None else {}),
      # ITS POSTURE (issue #387): lying down to rest is a fact the site
      # draws, never a fall -- `standing`, `lying_down`, `lying`,
      # `standing_up` or `getting_up`. Absent for a body that has none (the
      # rover always stands), so its frames read as they did.
      **({"posture": self.body.posture} if self.body.rights_itself else {}),
      "battery": {"frac": round(self.battery.fraction, 4),
                  "watts": round(self.battery.last_power_w, 2),
                  "charging": self.charging_now},
      # THE SURVIVAL CLOCK (0.15.0, issue #107): on the wire and in the
      # model's context, because a metric the robot cannot see is not one
      # it can optimise. `dead` is the cause while it waits for a reset.
      "survival": {"s": round(self.survival_s, 1), "deaths": len(self.deaths),
                   "dead": self.dead["cause"] if self.dead else None,
                   # LIVES LEFT, and how many robots this volume has used
                   # up (0.17.0, issue #136). Here rather than in `ledger`
                   # because a heart is a fact about staying alive, and
                   # here rather than nowhere because a stake nobody can
                   # see is not a stake: the site draws them and the model
                   # is shown the same number in its own context. Absent
                   # where no ledger is attached, which is every world that
                   # has no lives to lose.
                   **({"hearts": self.ledger.hearts(),
                       "generations": self.ledger.generations()}
                      if self.ledger is not None else {}),
                   # ...and how long until it stands itself up (issue #143).
                   # ⚠ ABSENT rather than null when there is nothing to
                   # count -- alive, or no timer configured. A `null` here
                   # would be a countdown a consumer had to special-case,
                   # and "this world has no restart timer" and "this robot
                   # is not dead" are both simply "no number".
                   **({"resetInS": self.reset_in_s}
                      if self.reset_in_s is not None else {})},
    }

  # ---- scoring (issue #14) -------------------------------------------------

  def _bank(self, verdict) -> dict | None:
    """Narrate an evaluator's verdict and hand it to the ledger.

    This is the lifecycle's ENTIRE role in scoring: it runs the tasks and it
    asks economy/scoring.py to judge the finished one. It never decides a verdict
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
      # ⚠ `verdict.reason` and not `public_metrics`: the reason line is
      # already redacted of a hidden answer (issue #14), and History is read
      # back into the model's own context, so a file built out of raw
      # metrics would hand the census its ground truth by the back door.
      self._remember(f"{verdict.task}: {verdict.reason}")
      return None
    entry = self.ledger.award(verdict, t=float(self.data.time))
    tail = " (pending a rating)" if entry["pending"] else ""
    self._say(f"SCORE {verdict.task}: {verdict.reason} -- "
              f"{entry['points']:+d} points{tail}, "
              f"balance {entry['balance']}")
    self._remember(f"{verdict.task}: {verdict.reason} "
                   f"({entry['points']:+d} points{tail})")
    return entry

  @property
  def autonomous(self) -> bool:
    """THERE IS A MIND HERE (issues #115, #427), and so the three rails are
    off: `needs_charge` (the floor), `_afford_next` (the gate) and
    `claim_budget_wh` (the offer filter) each read this and nothing else
    decides on it (`idle_s` is a cadence). They come off together or not at
    all -- removing one measured nothing: the floor fired once in six
    baseline days and the gate eleven times. With no mind the loop decides
    for itself (`scripted`), and the rails are its own.

    ⚠ READ OFF THE OVERSEER, NEVER SET: an arm with a mind inside the rails
    was the `guarded` control, retired in #427, and a flag that could
    disagree with whether a mind is here is how it would come back."""
    return self.overseer is not None

  @property
  def needs_charge(self) -> bool:
    # The reserve is a PARAMETER of the world, not of the pack (issue #6):
    # the cost of getting home is set by the floor plan.
    #
    # ⚠ RAIL ONE OF THREE, AND THE ONE THAT FIRES LEAST (issue #115): off
    # where there is a mind (`autonomous`). Measured across six baseline days
    # it fired ONCE, against eleven deferrals from the gate below.
    if self.autonomous:
      return False
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
    hard the wheels stall against them. See economy/energy.json.
    """
    rate = self.energy.charge_w
    if rate <= 0.0:
      return CHARGE_TIMEOUT
    # ⚠ AND AGAINST THE SCALE (issue #84). `charge_scale` makes the pack fill
    # k times faster, so a cap sized for the honest rate is k times too
    # generous -- and a timeout that can no longer fire is not a timeout. It
    # is one more factor in the same expression rather than a second branch,
    # because the two are the same question: how long should THIS charge, on
    # THIS pack, at THIS rate, be allowed to take.
    need = (self.charged_wh * 3600.0 / (rate * self.battery.charge_scale)
            * CHARGE_TIMEOUT_SLACK)
    return max(CHARGE_TIMEOUT_MIN, need)

  # ---- phases --------------------------------------------------------------

  # ---- the branches, as routines (issue #58) --------------------------------
  # Each branch of the arbitration loop is a ROUTINE yielding one drive
  # command per physics step (pluggybot/tick.py), so the day is ticked from
  # ONE loop rather than each branch owning the clock while it runs. The
  # blocking twins keep their old names for scripts and tests.

  def explore(self, budget: float | None = None,
              mark_done: bool = True) -> None:
    return self.body.run(self.explore_routine(budget, mark_done))

  def explore_routine(self, budget: float | None = None,
                      mark_done: bool = True) -> Routine:
    """Frontier-drive the map until the battery calls, or the map is done.

    The rack's fiducial is watched for throughout (`Body.start_discovery`),
    so exploring is also how the robot learns where its hub is -- the same
    trip that maps the room localizes the dock.

    `budget` bounds ONE call (issue #15): an overseer that chose to explore
    gets a slice and then the arbitration loop reconsiders, rather than one
    decision consuming the mission. `mark_done=False` goes with it, because a
    slice running out is not the same fact as the map being finished -- and
    conflating them would let the first overseer explore permanently retire
    the branch. Running out of FRONTIERS still marks it done under either
    setting: that one really is "there is nothing left to see".

    Returns how it ended (issue #424): `budget`, the frontier status it
    finished on (`no-frontiers`, `no-reachable`, `only-near`, `blocked`),
    `battery`, or `time` -- the run's own end.
    """
    strikes = 0
    deadline = (self.data.time + budget if budget is not None
                else self.explore_deadline)
    while not self.needs_charge and self.data.time < self.max_sim_time:
      if self.data.time > deadline:
        self.floor_explored = mark_done
        self._occur("task_complete", "explore")
        self._say("EXPLORE: budget spent, stopping")
        return "budget"
      path, status = self.body.plan_frontier(self.blacklist)
      if status == "ok":
        wx, wy = self.body.grid.cell_to_world(*path[-1])
        t0 = self.data.time
        yield from self.body.go_to_routine(wx, wy, timeout=25.0)
        if self.data.time > t0:
          strikes = 0
          continue
        # ⚠ A DRIVE THAT STEPPED NOTHING IS A PATHLESS REPLAN, not progress:
        # `plan` does not mask the other robot and `_plan_to` does (#167), so
        # a frontier behind it is "ok" here and unroutable there, and looping
        # on it never advances sim time. The spin steps, so the other moves.
        status = "blocked"
      yield from self.body.look_around_routine()
      strikes += 1
      if status == "no-frontiers" or strikes >= STRIKES_TO_FINISH:
        self.floor_explored = True
        self._occur("task_complete", "explore")
        self._say(f"EXPLORE done ({status})")
        return status
    self._say("EXPLORE -> GO_CHARGE (battery low)")
    return "battery" if self.needs_charge else "time"

  def go_charge(self) -> bool:
    return self.body.run(self.go_charge_routine())

  def go_charge_routine(self) -> Routine:
    """Navigate to the charge bay and press until the pins connect.

    The terminal half is `Body.dock_routine` (issue #32): the standoff
    computed from the believed rack pose is only how the robot gets to the
    NEIGHBOURHOOD -- the approach itself is re-measured off the charge bay's
    own tag and verified-retried, exactly as every tool-bay approach already
    was. The blind creep this replaces forgave ~6 cm / ~10 deg of belief
    error, and a long shift's accumulated drift walked straight out of that
    envelope roughly once an hour on a hosting pack.
    """
    self.charge_failure = ""
    self.body.refresh_rack()
    sx, sy, hd = self.body.charge_standoff()
    # Route-failure retry, same as swap_at_bay's: when nothing is reachable,
    # spin to buy map (and possibly the rack tag) and try again.
    # ⚠ THE CHARGE BAY IS NOT A TOOL BAY, and the difference is the whole
    # asymmetry (issue #313). The same arithmetic applies -- a peer within
    # 0.45 m of this standoff puts it outside anything the planner may
    # route to, and MEASURED on the rack prior the neighbouring tool bay is
    # 0.200 m from it, so a robot swapping there blocks this approach
    # outright -- but a bay given up is a lost errand and a charge given up
    # is a death. So a taken charge bay is WAITED FOR (issue #346), for 3x
    # a charge's typical occupancy, and neither the pack nor an interrupt
    # ends that wait: charging is what either would ask for.
    spins, since, arrived = 0, None, False
    driven = None                 # the goal the last drive was sent to
    tried = None                  # why a near give-up's approach stopped
    while True:
      if self.peers and self.body.peer_on_the_goal(sx, sy) is not None:
        since = float(self.data.time) if since is None else since
        if not (yield from self._await_bay_routine(sx, sy, "charge", since)):
          driven = None           # the wait is why, not any drive before it
          break
      driven = (sx, sy)
      arrived = yield from self.body.go_to_routine(sx, sy, timeout=90.0)
      if arrived:
        break
      if self.peers and self.body.peer_on_the_goal(sx, sy) is not None:
        continue                          # taken again: wait again, above
      if self._near_standoff(sx, sy):
        # ⚠ NOT `GO_CHARGE: `, which the site counts as a failed attempt
        # (rooftop's `chargeOutcome`): this attempt's one outcome comes after
        self._say(f"GO_CHARGE near enough -- {self.drive_why(sx, sy)}; the "
                  "dock's board decides")
        tried = yield from self.body.dock_routine()
        # ...and with no board in sight from here -- a wall between, a
        # belief further off than it says -- the walk's own retries, as
        # before: no board is no reason to stop looking for a route
        if tried != "no board" or self.body.charging():
          break
      yield from self.body.look_around_routine()
      self.body.refresh_rack()
      sx, sy, hd = self.body.charge_standoff()
      spins += 1
      if spins == 2:
        break
    if not arrived and tried in (None, "no board"):
      # WHY, and not always "no route" (issue #350): a stall, the other
      # robot and the clock read the same until the drive said which. Of
      # the goal last DRIVEN to -- a spin moves the standoff after it, and
      # a trip that ended in a wait that gave up is the wait's. Whoever
      # held the bay is IN the reason: History and a stranded death read it
      # (issue #424).
      blocked = self.peer_at(*(driven or (sx, sy)))
      self.charge_failure = "never reached the charge bay" + (
        "" if driven is None else f": {self.drive_why(*driven)}") + (
        "" if tried is None else "; the board was not in sight from there") + (
        "" if blocked is None else
        f" -- {self.held_for(blocked).replace('the bay', 'it', 1)}")
      self._say(f"GO_CHARGE: {self.charge_failure}")
      return False
    # Line up on the bay's own tag and creep until the electrical criterion
    # fires -- position is believed, contact is known.
    if arrived:
      tried = yield from self.body.dock_routine()
    if not self.body.charging():
      # the approach's trace is EVIDENCE (issue #346), the log's alone
      self.charge_failure = f"no charge contact ({tried})"
      self._say(f"GO_CHARGE: {self.charge_failure}",
                detail=self.body.charge_trace())
      return False
    self._say("GO_CHARGE -> CHARGE (pins connected)")
    return True

  def _near_standoff(self, sx: float, sy: float) -> bool:
    """Did the walk to the charge standoff give up near enough that the
    board decides (`NEAR_STANDOFF_M`), as its own record says? Never for
    the other robot in the way: that is waited for, or routed round."""
    rec = self.body.last_drive
    return (rec is not None and rec["why"] not in ("", "peer")
            and math.hypot(rec["goal"][0] - sx, rec["goal"][1] - sy) < 1e-6
            and rec["shortM"] <= NEAR_STANDOFF_M)

  def charge(self) -> None:
    return self.body.run(self.charge_routine())

  def charge_routine(self) -> Routine:
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
    # See `Body.docked`: a caller that knows it is held says so, and a
    # contact that flickers does not un-pin it.
    self.body.docked = True
    # THE DOCK IS THE RE-ANCHOR (issue #42). Called with the pins already
    # conducting -- go_charge verified that -- which is the one moment the
    # robot's true pose is known to millimetres by construction. This is
    # where a shift's accumulated dead-reckoning drift dies, instead of
    # compounding until a terminal maneuver walks out of its envelope.
    self.body.anchor_at_dock()
    try:
      timeout = self.charge_timeout
      while (self.battery.fraction < CHARGED
             and self.data.time - t0 < timeout):
        yield from self.body.dock_hold_routine(0.25)
        if not self.charging_now:
          # contact dropped: press again briefly, then give up on this attempt
          yield from self.body.redock_routine()
          if not self.charging_now:
            self._say("CHARGE: lost the pins")
            self._bank(scoring.score_charge(self, before))
            return
    finally:
      # Cleared before the undock, which is REAL travel and must be counted.
      self.body.docked = False
    self.charge_cycles += 1
    self._occur("task_complete", "charge")
    self._say(f"CHARGE complete ({self.battery.fraction:.0%}) -- backing off")
    self._bank(scoring.score_charge(self, before))
    yield from self.body.undock_routine()

  # ---- the recompile seam (issue #168 slice C) -----------------------------

  def rebind(self, model, data) -> None:
    """Point this lifecycle and everything it owns at a recompiled world.

    Every id is re-resolved by NAME (deleting a module shifts the ids of
    everything after it in the tree); the map, the belief, the reckoner,
    the battery and the ledger are state and do not move. The registered
    `on_rebind` callbacks are the sinks outside the lifecycle's ownership.
    """
    self.model, self.data = model, data
    self.body.rebind(model, data)
    if self.depth_camera is not None:
      self.depth_camera.rebind(model)
    if self.screen is not None:
      self.screen.rebind(model, data)
    if self.activities is not None:
      self.activities.rebind(model, data)
    if self.game is not None and hasattr(self.game, "rebind"):
      self.game.rebind(model, data)
    for callback in list(self.on_rebind):
      callback(model, data)

  def hang_tool(self, tool, bay: int) -> dict:
    """A built tool takes a bay ON THE BUILT-TOOL RAIL (`bay` is the rail's
    own index, A = 0; issue #277): a built tool already there is RETIRED,
    the tool's module is attached at the station, the world is recompiled
    with its state carried across, every holder is rebound, the tool's
    verbs are registered, and `scene_changed` goes out on the wire. The
    five hand-built modules hang on the first rack and are never in the
    way: nothing here can name their bays.

    Between errands only, with nothing on the fork -- FOR EVERY ROBOT in
    the world (issue #315): a recompile mid-errand would pull the world
    out from under a routine holding a transient tool controller (the pen,
    claw and dispenser classes are built per errand and are NOT rebound --
    they must not outlive a recompile), and a pair's two lifecycles run
    over one (model, data). Refused, out loud, otherwise, naming the robot
    that is busy. The recompile is this robot's to run and everybody's to
    follow: `_recompile` rebinds them all.
    """
    from pluggybot.workshop import build as wbuild
    from pluggybot.workshop import seam
    self.can_reshape(bay)
    if tool.body in self.rack_inventory:
      raise seam.SeamRefused(f"{tool.body} already hangs on the rack")
    index = built_bay_index(bay)
    retired = next((m for m, b in self.rack_inventory.items() if b == index), None)
    # ⚠ ONE RACK, TWO MINDS (issue #315): the rail is the WORLD's, so on a
    # pair the module in this bay may be the OTHER robot's -- and "naming a
    # bay retires what hangs there" is a rule about a robot's OWN tools.
    # Refused with whose it is, on `bay_index`'s terms for the originals:
    # a bay is negotiated between the minds, not taken from one by the
    # other. `built` is what this lifecycle hung, so the test is ownership
    # and not the shared inventory.
    if retired is not None and retired not in self.built:
      raise seam.SeamRefused(
        f"bay {chr(ord('A') + bay)} holds the {retired.removeprefix('module_')}, "
        "which is not yours to retire -- name a bay of your own")
    record = {"tool": tool.name, "module": tool.body, "bay": bay,
              "retired": retired, "t": round(float(self.data.time), 3)}
    if retired is not None:
      record["retiredWhat"] = self._retire_from_spec(retired)
    record["attached"] = seam.attach(self.spec, tool, bay, seam.rack_pose(self.model))
    record["recompileMs"] = self._recompile(reason="tool", tool=tool.name,
                                            module=tool.body, bay=bay,
                                            retired=retired)
    self.rack_inventory[tool.body] = index
    self.built[tool.body] = tool
    record["verbs"] = wbuild.register(tool)
    record["relearned"] = self._revalidate_library()
    self._say(f"I hung my {tool.name} in bay {chr(ord('A') + bay)} of my rack"
              + (f", retiring my {retired.removeprefix('module_')}" if retired else ""),
              detail=f"recompile {record['recompileMs']} ms")
    return record

  def retire_tool(self, module: str) -> dict:
    """Take a BUILT tool off its rail for good: its bay goes empty and its
    verbs leave the registries with it. A hand-built module is refused by
    name (issue #277): the five originals are permanent, and the language's
    verbs for them stay. Same preconditions as `hang_tool`."""
    from pluggybot.workshop import seam
    if module in seam.HAND_BUILT:
      raise seam.SeamRefused(f"the {module.removeprefix('module_')} is one of the "
                             "original modules and stays on the rack")
    if module not in self.rack_inventory:
      raise seam.SeamRefused(f"{module} is not on the rack")
    # ...and on a pair the rail is shared (issue #315): a module this
    # lifecycle did not hang is the other robot's, and `built` is what it
    # hung. Same rule as the bay in `hang_tool`, said the other way round.
    if module not in self.built:
      raise seam.SeamRefused(f"the {module.removeprefix('module_')} is on the "
                             "rack but is not yours to retire")
    index = self.rack_inventory[module]
    bay = index - len(HUB_STATION_YS)
    self.can_reshape(bay)
    record = {"module": module, "bay": bay, "t": round(float(self.data.time), 3),
              "retiredWhat": self._retire_from_spec(module)}
    record["recompileMs"] = self._recompile(reason="retire", tool=None,
                                            module=None, bay=bay, retired=module)
    # ⚠ ONCE, HERE, not inside `_retire_from_spec` (issue #324): `hang_tool`
    # calls that helper too, and revalidating there AND after `register`
    # told the robot twice, in one action, that the same procedure had
    # broken -- with two different reason texts, because the second pass saw
    # the replacement's axes. The rack is in its final state at each of the
    # two PUBLIC doors, and nowhere in between.
    record["relearned"] = self._revalidate_library()
    self._say(f"I took my {module.removeprefix('module_')} off my rack; bay "
              f"{chr(ord('A') + bay)} is empty", detail=f"recompile {record['recompileMs']} ms")
    return record

  #: The states a recompile may not land in. A tool controller (the pen,
  #: the claw, the dispenser) is built per errand, holds (model, data) and
  #: is NOT rebound -- `tests/test_recompile.py::TRANSIENT_HOLDERS` is the
  #: roster and the reason -- so a world pulled out from under a running
  #: errand leaves that errand stepping a world that no longer exists.
  MID_ERRAND = ("SWAP_PICK", "USE_TOOL", "SWAP_RETURN")

  def built_by(self) -> dict[str, str]:
    """Which robot built each tool on the rail: module -> display name, with
    this robot's own as "you" (issue #324).

    ⚠ READ OFF `built`, which is what a lifecycle HUNG -- the shared
    `rack_inventory` says what is on the rail and cannot say whose it is.
    A robot with no peers owns everything it hung and nothing else. This is
    a PUBLIC fact: the rail is the world's and either robot can see, scan
    and fetch what hangs there, so naming its builder tells nobody anything
    a look at the rack would not (`others_context`'s line, kept).
    """
    mine = {module: "you" for module in self.built}
    for other in self.peers:
      name = other.robot_name or other.root
      for module in getattr(other, "built", {}):
        mine.setdefault(module, name)
    return mine

  def seam_busy(self) -> str:
    """Why the world may not be recompiled THIS INSTANT, or "".

    The one reason that can change while the robot stands still, which is
    why it is its own predicate: `_await_seam_routine` polls exactly this
    and nothing else, so a wait can never mask a permanent refusal (no
    spec, no rail, no such bay) as something worth waiting for.
    """
    from pluggybot.continuation import MOVING_POSTURES
    from pluggybot.procedure.steps import _carried
    for life in (self, *self.peers):
      held = _carried(life)
      if life.state in self.MID_ERRAND or held:
        # WHAT is in the way, not just the rule (issue #264): a deployed
        # robot holding a claw it could not stow was told only "never
        # mid-errand", and specified its tool twice more.
        what = (f"its fork holds {held}" if held else "it is mid-errand")
        who = (("your fork holds " + held + ": stow it first -- " if held
                else "you are mid-errand -- ") if life is self
               else f"{life.robot_name or life.root} is busy ({what}): ")
        return (who + "a tool is hung between errands with every fork empty")
      # ...nor while a body is in the middle of a move (#407, a restart's
      # rule, `Keeper.busy`): lying down or standing up is a generator
      # holding the world it began in
      if (life._standing_up or life.body.posture in MOVING_POSTURES
          or life.body.working or life.body.making_way is not None):
        who = "you are" if life is self else f"{life.robot_name or life.root} is"
        return f"{who} in the middle of a move: a tool is hung once every body is still"
    return ""

  def can_reshape(self, bay: int) -> None:
    """Every reason the world may not be recompiled right now, or nothing.
    Checked BEFORE a build spends anything, so a refused hang never
    follows a paid print. `bay` is a built-rail index.

    ⚠ A PAIR SHARES ONE WORLD, so EVERY robot in it has to be between
    errands with its fork empty, not just this one (issue #315). The
    recompile itself is one robot's to run -- it rebinds all of them
    (`_recompile`) -- but the refusal is the whole pair's, and it names
    the robot that is busy, because "wait" and "never" are different
    answers and the robot is the one that has to tell them apart.
    """
    from pluggybot.workshop import seam
    if self.spec is None:
      raise seam.SeamRefused("this world was compiled without its spec; "
                             "build it through `build()` to hang tools")
    if not self.has_built_rack:
      raise seam.SeamRefused("this world has no built-tool rack; a built tool "
                             "has nowhere to hang here")
    busy = self.seam_busy()
    if busy:
      raise seam.SeamRefused(busy)
    if not 0 <= bay < len(BUILT_STATION_YS):
      raise seam.SeamRefused(f"no bay {bay}; the built-tool rail has "
                             f"{len(BUILT_STATION_YS)}")

  def _retire_from_spec(self, module: str) -> dict:
    from pluggybot.workshop import build as wbuild
    from pluggybot.workshop import seam
    gone = seam.retire(self.spec, module)
    del self.rack_inventory[module]
    self.built.pop(module, None)
    wbuild.unregister(module)
    return gone

  def _revalidate_library(self) -> list[str]:
    """The rail moved, so every procedure is re-compiled against it (issue
    #324) -- for EVERY robot in this world, because the rail is the world's
    and a tool one robot retires takes its axes out of the other's
    procedures too. What changed is narrated and returned for the `tool`
    event; a procedure that stopped being runnable is the interesting half,
    and one that started again (its tool rebuilt) is worth saying too.
    """
    moved: list[str] = []
    for life in (self, *self.peers):
      library = getattr(life.overseer, "library", None) if life.overseer else None
      if library is None:
        continue
      for name in library.revalidate(world_facts(life.world, rack=life.rack_inventory)):
        entry = library.entries[name]
        state = "runs again" if entry.valid else f"cannot run: {'; '.join(entry.reasons)}"
        life._say(f"LIBRARY my procedure {name} {state}")
        moved.append(name)
    return moved

  def _recompile(self, **why) -> float:
    """Recompile the edited spec, rebind everything, tell the wire. Returns
    the milliseconds it took.

    ⚠ EVERY LIFECYCLE IN THIS WORLD IS REBOUND, not just the one that
    built the tool (issue #315). A pair is two lifecycles over one
    (model, data), and `spec.recompile` returns NEW objects: the robot
    that did not build would otherwise go on stepping the old world --
    two simulations, silently diverging, which is the exact failure the
    rebind protocol exists to prevent. The peers' `on_rebind` sinks fire
    with them, which is how a publisher registered on the first robot
    follows a tool the second one built.

    ⚠ THE RECOMPILE IS NOT THE COST. MEASURED on the home world, 2026-09-22:
    `spec.recompile` 4-5 ms, this whole method ~160 ms, and 136 ms of that
    is `rebind` recreating the tag detector's `Renderer` -- an EGL context,
    which is what a Renderer bound to the old model has to become. It is one
    hitch on the physics thread per build or retire, not a per-step cost,
    and it is why a recompile is refused mid-errand rather than throttled.
    `scene_dict` is 18 ms of the rest, and the sidecar this passes it is
    0.1 ms of that.
    """
    from pluggybot.workshop import seam
    from pluggybot.telemetry.scene import scene_dict
    t0 = time.perf_counter()
    model, data = seam.recompile(self.spec, self.model, self.data)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    for life in (self, *self.peers):
      life.rebind(model, data)
    # The wire (protocol/README.md "scene_changed"): the whole new scene,
    # IN EXACTLY THE SHAPE THE SCENE FIXTURE HAS, because the site replaces
    # its whole scene graph with it. So the generator's sidecar comes with
    # it (the `visual` hints, the zones, the spawns, the plate glyphs) and
    # a pair's scene is named the PAIR world -- without either, building a
    # tool on the deployed world would repaint the house as grey primitives
    # and call it `home_world` (issue #315; it could not fire on a pair at
    # all until then, so the gap was dormant).
    cfg = world_config(self.world)
    name = cfg["model_name"]
    if self.peers:
      from pluggybot.robot import pair_model_name
      name = pair_model_name(name)
    meta = json.loads(Path(cfg["meta"]).read_text()) if cfg["meta"] else None
    self._emit({"type": "scene_changed", "t": round(float(self.data.time), 3),
                "robot": self.root, **why,
                "scene": scene_dict(model, name, meta=meta)})
    return ms

  # ---- the workshop as the agent's (issue #168 slice D) --------------------

  def _workshop_routine(self, decision) -> Routine:
    """Apply a decision's `build_tool` / `retire_tool` (issue #168).

    Retire before build, as `_reconsider` does. A build is: the spec
    checked against the envelope, the seam's preconditions checked, the
    points PAID (`Ledger.spend`, no debt: unaffordable is refused before
    anything prints), the print and assembly time WAITED where the robot
    stands, the module hung. Every step is a `tool` event with its
    outcome, and a refusal carries its reasons -- what the robot wrote
    rides the event whole, as a procedure does.
    """
    shop = getattr(self.overseer, "workshop", None)
    if shop is None or not (decision.build_tool or decision.retire_tool):
      return
    from pluggybot.workshop import cost as wcost
    from pluggybot.workshop import seam
    from pluggybot.workshop.library import BAYS, WorkshopRefused
    from pluggybot.workshop.seam import SeamRefused
    t = float(self.data.time)
    base = {"type": "tool", "t": round(t, 3), "robot": self.root}
    if decision.retire_tool:
      name = decision.retire_tool
      try:
        entry = shop.entries.get(name)
        if entry is None:
          # ...and an original named here gets the reason, not "no such
          # tool" (issue #277): the grammar enumerates built names, but a
          # prose answer or an old habit can still say "pen"
          if f"module_{name}" in seam.HAND_BUILT:
            raise WorkshopRefused([f"the {name} is one of the original modules "
                                   "and stays on the rack"])
          # ...and on a pair the rail is shared, so a tool the robot can SEE
          # in its `rack` block may be the other robot's (issue #315). That
          # is a different answer from "there is no such tool".
          if f"module_{name}" in self.rack_inventory:
            raise WorkshopRefused([f"the {name} is on the rack but is not yours "
                                   "to retire"])
          raise WorkshopRefused([f"no built tool {name!r} to retire"])
        self.retire_tool(f"module_{name}")
        shop.retire(name)
      except (WorkshopRefused, SeamRefused) as e:
        reasons = getattr(e, "reasons", [str(e)])
        shop.refuse(name, reasons, t)
        self._say(f"WORKSHOP retire {name!r} refused: {e}")
        self._emit({**base, "outcome": "refused", "verb": "retire_tool",
                    "name": name, "reasons": list(reasons)})
      else:
        self.tools_retired += 1
        self._remember(f"retired my tool {name}")
        self._emit({**base, "outcome": "retired", "name": name, "bay": BAYS[entry.bay]})
    if decision.build_tool:
      raw = decision.build_tool
      name, bay, spec = raw.get("name", ""), raw.get("bay", ""), raw.get("spec")
      self._emit({**base, "outcome": "specified", "name": name, "bay": bay,
                  "spec": spec})
      try:
        tool, idx = shop.check(name, spec, bay)
        self.can_reshape(idx)
        if tool.body in self.rack_inventory:
          raise WorkshopRefused([f"{tool.body} already hangs on the rack"])
        bill = wcost.price(tool)
        balance = self.ledger.balance() if self.ledger is not None else 0
        if bill["points"] > 0 and (self.ledger is None or
                                   self.ledger.spend(bill["points"], why=f"tool {name}")
                                   < bill["points"]):
          raise WorkshopRefused([f"cannot afford it: {bill['points']} points for the "
                                 f"parts, balance {balance}"])
      except (WorkshopRefused, SeamRefused) as e:
        reasons = getattr(e, "reasons", [str(e)])
        shop.refuse(name, reasons, t)
        self._say(f"WORKSHOP build {name!r} refused: {e}")
        self._emit({**base, "outcome": "refused", "verb": "build_tool",
                    "name": name, "bay": bay, "reasons": list(reasons)})
        return
      self._say(f"WORKSHOP printing and assembling my {name}: {bill['printedG']:g} g "
                f"of PLA, {len(tool.parts)} parts, {bill['points']} points",
                detail=f"wait {bill['waitS']} s")
      self._emit({**base, "outcome": "built", "name": name, "bay": bay,
                  "cost": bill})
      def not_hung(e, waited: float) -> None:
        # THE POINTS ALWAYS BUY SOMETHING. The preconditions held before
        # the spend and broke while it printed -- the peer took the seam,
        # or the robot died mid-print -- and the parts are bought either
        # way. So the tool is RECORDED, and `restore_tools` hangs it at
        # the next mission start without paying again, which is the same
        # path a tool built yesterday takes. Losing the points AND the
        # tool is the "paid, refused build" this issue exists to prevent.
        why = [str(e), "built and paid for; it hangs when the rack is free"]
        shop.record(tool, spec, idx, bill, t, reasons=why)
        shop.refuse(name, why, t)
        self._say(f"WORKSHOP my {name} is built but cannot hang yet: {e}")
        self._remember(f"built my tool {name}; it hangs when the rack is free")
        self._emit({**base, "outcome": "refused", "verb": "hang", "name": name,
                    "bay": bay, "reasons": why, "cost": bill,
                    "waitedS": round(waited, 1)})

      try:
        yield from self._fabricate_routine(bill["waitS"])
        # ...and then WAIT FOR ROOM (issue #315): a print is longer than an
        # errand, so on a pair the rack is usually occupied again by now.
        waited = yield from self._await_seam_routine(idx)
      except GeneratorExit:
        # ⚠ A STAND-UP ENDS THE WAIT (issue #348): `_until_stood_up_routine`
        # closes the decision's action, this included, and the points are
        # already spent. Recorded like a rack that never freed; no yield.
        not_hung("a stand-up ended the wait", max(
          0.0, float(self.data.time) - t - float(bill["waitS"])))
        raise
      try:
        hung = self.hang_tool(tool, idx)
      except SeamRefused as e:
        not_hung(e, waited)
        return
      shop.record(tool, spec, idx, bill, t)
      self.tools_built += 1
      self._remember(f"built my tool {name} and hung it in bay {bay}")
      # `waitedS` rides the HUNG row too, not only the refused one: a build
      # that waited 40 s and then hung is what a contended rack looks like,
      # and a rack that is never contended reads 0 (issue #315).
      self._emit({**base, "outcome": "hung", "name": name, "bay": bay,
                  "module": tool.body, "retired": hung["retired"],
                  "verbs": hung["verbs"], "cost": bill,
                  "waitedS": round(waited, 1)})

  def _await_seam_routine(self, bay: int) -> Routine:
    """Stand still until the rack is free to be reshaped, or give up.

    ⚠ THE PEER CAN TAKE THE SEAM AWAY WHILE THE TOOL PRINTS (issue #315).
    The scoop's print and assembly is 896 sim s -- LONGER THAN A TYPICAL
    ERRAND -- so on a pair the other robot is usually mid-errand by the
    time the parts are ready, and before this the build was PAID FOR and
    then lost at the hang: no tool, no record, three points gone. That is
    the "paid, refused build" the issue set out to prevent, arriving
    through a door only a pair has.

    So the honest model is that the assembly is DONE and the tool hangs
    when there is room: the robot is already standing here, and waiting
    costs it only more of the time it was already spending. `HANG_WAIT_S`
    bounds it at one full errand's worth (measured: an errand runs
    200-500 sim s; `solutions.TOWER` is 489), because a peer that is
    always busy is a world to report, not one to stand in for ever.

    ⚠ AND IT IS BOUNDED BY THE PACK AS WELL AS THE CLOCK. Standing still
    is 10.5 W with the near-field camera on: the print alone is 2.6 Wh, a
    third of home's hosting pack, and the full wait would take it to 4.4 Wh
    -- 55 %, against a 2.05 Wh reserve. The ROBOT chose to build; the
    waiting is CODE's, so code stops spending its pack once what is left is
    the return trip's. Not a rail and not branched on the arm flag -- every
    arm gets it, because on no arm should the loop's own retry be what
    strands the robot. (⚠ Naming that flag in prose HERE is what
    `test_the_rails_are_read_in_exactly_one_place_each` counts: it reads
    the class source, so a comment citing it reads as a fourth reader.) Giving up early is not a loss -- the tool is recorded and
    hangs at the next mission start.
    """
    t0 = float(self.data.time)
    while (self.seam_busy() and float(self.data.time) - t0 < HANG_WAIT_S
           and self.battery.energy_wh > self.low_battery_wh):
      yield from self.body.hold_routine(SEAM_POLL_S)
    return float(self.data.time) - t0

  def _fabricate_routine(self, seconds: float) -> Routine:
    """The print and the assembly: the robot stands where it is for this
    long, drawing what an idle robot draws. Its own routine so a test can
    stub it (the way `go_to_routine` is stubbed) and pin the seconds
    without stepping fifteen sim-minutes of physics."""
    yield from self.body.hold_routine(seconds)

  def restore_tools(self) -> list[str]:
    """Hang every tool the workshop's records say the robot built (a
    restart recompiles the world from a file that knows nothing of them).
    Paid for once; not paid again. Returns what was hung."""
    shop = getattr(self.overseer, "workshop", None)
    if shop is None:
      return []
    hung = []
    for entry in shop.hung():
      if entry.tool.body in self.rack_inventory:
        continue
      try:
        self.hang_tool(entry.tool, entry.bay)
      except Exception as e:      # noqa: BLE001 -- said, never silent
        entry.tool, entry.reasons = None, [f"could not be hung again: {e}"]
        self._say(f"WORKSHOP could not hang my {entry.name} again: {e}")
        continue
      entry.reasons = []          # on the rack now; why it was not is stale
      hung.append(entry.name)
    return hung

  def ink_hook(self, board: str, program: str):
    """What a drawing tells the board book (issues #12, #406): each stroke's
    ink as it lands, the board erased before the first. ⚠ THE ERASE RIDES
    THE FIRST INK (the rover's #30): a drawing that never touched the board
    leaves the old ink standing, which is what happened. None without a
    book or with no such board."""
    book = self.boards
    if book is None or board not in book:
      return None
    erased = False

    def on_stroke(i, points, name) -> None:
      nonlocal erased
      if len(points) < 2:
        return
      t = float(self.data.time)
      if not erased:
        book.clear(board, t=t, by=self.root)
        self._say(f"USE_TOOL: erased {board}")
        erased = True
      # `by` is THIS robot (#181, #298): a pair shares one book
      book.stroke(board, name or program, points, t=t, by=self.root)

    return on_stroke

  def run_errand(self, errand) -> dict:
    return self.body.run(self.run_errand_routine(errand))

  def run_errand_routine(self, errand) -> Routine:
    """Fetch a tool, take it somewhere, DO something, and put it back.

    The middle is the errand's own `use` callable (mission/errand.py). Everything
    around it -- which bay, verifying the pick electrically, verifying the
    stow by hanging, restoring the arm -- is identical whatever the tool is,
    which is exactly why it lives here and only here.

    A use-phase that raises is caught and reported: the tool is still on the
    fork, and driving it back to its bay is strictly better than abandoning a
    module in the middle of the living room. `MissionAborted` (the viewer
    closing) is deliberately NOT caught -- that is a request to stop, not a
    failure to recover from.
    """
    # A composed errand that fetches nothing (a look-around, issue #166)
    # names no module, and `self.module` is what the end-of-day summary
    # reads: keep the last real one rather than an empty name.
    self.module = errand.module or self.module
    # ---- interruptible from here to the stow (issue #116) ----
    # ⚠ THE FLAGS ARE PER ERRAND AND CLEARED ON THE WAY IN, never on the way
    # out: an errand that raises must not leave the NEXT one already
    # aborting, and `run_errand`'s use phase is arbitrary caller code that
    # can raise.
    self._in_errand = True
    self._errand_name = errand.name
    self._errand_now = errand
    self._interrupt_pending = None
    self._aborting = False
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
    if errand.program is not None:
      # What the board and the cage looked like before an errand on them
      # (issues #14, #226; {} for every other program). ⚠ THE BOARD TOO
      # (#406): only the first ink erases, so a job that inked nothing was
      # paid for the drawing already up -- "correct, 0.0 mm from the glyphs".
      before = {**scoring.board_before(self, errand), **scoring.cage_before(self, errand)}
      used = yield from self._run_program_routine(errand)
      result = {"errand": errand.name, "module": errand.module,
                "energyWh": round(max(0.0, spent_from - self.battery.energy_wh), 4),
                "estimateWh": round(self.affords(errand).cost_wh, 4),
                "energySeconds": round(float(self.data.time) - began_at, 2),
                **used}
      self._in_errand = False
      self._deferrals.pop(errand.name, None)
      failed = self._program_failure(errand, used)
      verdict = scoring.score_errand(self, errand, result, before, failed=failed)
      entry = self._bank(verdict)
      if verdict is not None:
        result["verdict"] = verdict.as_dict()
        result["points"] = entry["points"] if entry is not None else 0
        if errand.task_id and self.tasks is not None:
          closed = self.tasks.resolve(errand.task_id, verdict,
                                      t=float(self.data.time))
          if closed is not None:
            result["task_id"] = closed.id
            self._say(f"TASK {closed.id} {closed.state}: {closed.description}")
      self.errand_results.append(result)
      if errand.detail.get("cage"):
        self._cage_record(errand, result, verdict, before, failed)
      self._occur("task_complete" if used["stowed"] and "error" not in used
                  else "task_failed", errand.name)
      self._errand_name = ""
      self._errand_now = None
      return result
    why = yield from self.body.fetch_tool_routine(errand.station_y, self.module)
    carried = self.body.module_state(self.module)["on_fork"]
    blocked = self.peer_at_the_bay(errand.station_y)
    missed = (None if carried else
              self.pick_failure(self.module, errand.station_y, why))
    self.swaps_done += 1
    # the swap's trace is EVIDENCE (issue #264): `detail`, the log's alone
    self._say(f"SWAP_PICK {'done -- carrying the module' if carried else 'FAILED'}"
              f" ({errand.name})" + ("" if missed is None else f" -- {missed}"),
              detail="" if carried else self.body.swap_trace())
    # A FAILED PICK ENDS THE ERRAND AT THE RACK (issue #298). It used to go
    # on: drive to the use pose with nothing on the fork, skip the use,
    # drive back and attempt a RETURN of a module it never had -- narrated
    # "dropped the tool on the way" -- and on the deployed pair that phantom
    # trip parked the robot at the rack exactly when the other came back to
    # stow, whose stow then failed, whose next pick then failed: Rowan's
    # correct answers paid 3 of 27. History says which of the two things a
    # failed pick is, because the robot was diagnosing its pen for what was
    # the other robot holding it.
    # ⚠ ...AND WHICH ONE, because the robot reads this back (issues #313,
    # #264): Rowan diagnosed its pen off a correlation no line ever named,
    # and "the pick missed" covered an approach that never reached the
    # rack. `pick_failure` is the one sentence for it; `fetch` says the same.
    # ⚠ ...AND IT LEADS THE VERDICT (issue #350): `failed` below is written
    # once, at the end, as the first clause of the job's own line.
    failed = ("" if carried or self._aborting else
              f"could not pick up {self.module}: {missed}")

    self.state = "USE_TOOL"
    # ⚠ THE ANSWER IS READ, and it used to be thrown away. `drive_to`
    # returns False when it stagnated or could not plan, and a use-phase run
    # anyway is a pen pressing at empty air -- the far whiteboard hung a
    # mission until the battery died, with nothing logged after "arrived".
    #
    # Not reaching the board is an ordinary outcome, reported rather than
    # raised on: the tool still goes back to its bay and the evaluator finds
    # no ink. Failing to get there and drawing badly are different events,
    # and both must end with the module on the rack.
    #
    # ⚠ THE GATE IS PER-ERRAND, not universal (`Errand.needs_use_pose`). An
    # errand that DOES ITS OWN NAVIGATION does not need this drive to have
    # arrived -- the census's `use_at` is the first point of the survey route
    # its use-phase drives itself, and gating it too cost the recorded
    # showcase mission its census answer.
    # SAFE POINT ONE: the tool is on the fork in its carry configuration and
    # nothing is engaged, so a hazard row that fired during the pick is
    # answered before the carry drive rather than after it -- which is where
    # aborting saves the most, since the trip out and back is most of an
    # errand's energy.
    aborted = carried and self.interrupted()
    arrived = (False if aborted or not carried else
               (yield from self.body.go_to_routine(*errand.use_at,
                                                   timeout=60.0)))
    still = carried and self.body.module_state(self.module)["on_fork"]
    # ⚠ "never got there" IS A NAVIGATION FAILURE AND AN ABORT IS NOT ONE
    # (issue #116). The robot did not set off: it was told to stop before the
    # carry drive and turned round with the tool still on the fork. Saying
    # the two the same way is the conflation `stranded` was split out of
    # "mission complete" for (issue #32) -- a reader of the log cannot tell a
    # choice from a fault, and one of them means the drive is broken.
    drove = ("" if arrived or aborted or not carried
             else self.drive_why(*errand.use_at))
    self._say("USE_TOOL: " + ("nothing on the fork to take there" if not carried
                              else "turned back before setting off" if aborted
                              else "arrived" if arrived
                              else f"never got there -- {drove}")
              + ("" if still or not carried else " -- but dropped the tool on the way"))
    # THE ERRAND'S OWN FAILURE, BEFORE ITS USE-PHASE (issue #350): what the
    # verdict leads with and History says, because the evaluator can only
    # say what the world shows -- "no ink reached whiteboard_b", fourteen
    # times in a day, for a pen that never left the rack or never got there,
    # and both robots ticketed the ink path.
    if not failed and carried and not aborted:
      where = self._errand_where(errand)
      if not still:
        failed = f"dropped {self.module} on the way to {where}"
      elif errand.use is not None and not arrived and errand.needs_use_pose:
        failed = f"never reached {where}: {drove}"
    # What the board looked like before this errand touched it (issue #14).
    # The evaluator counts the strokes that landed HERE, so a second drawing
    # on an un-erased board is not scored on the first one's ink.
    before = scoring.board_before(self, errand)
    used: dict = {}
    if not carried and self._aborting:
      # ...the robot's own rule ended the WAIT for the bay (issue #346):
      # an act of caution, on SAFE POINT TWO's terms below.
      used = {"interrupted": True, "stopped": "interrupted",
              "reason": "gave up waiting for the bay on the agent's own "
                        "interrupt"}
    elif not carried:
      used = {"error": f"never picked up {self.module}", "pickWhy": why,
              **({} if blocked is None else {"peerAtBayM": round(blocked[1], 3)})}
    # SAFE POINT TWO: arrived, tool on the fork, nothing started. ⚠ AN ABORT
    # IS NOT AN ERROR -- the errand did not fail, it was cut short on the
    # agent's own instruction, and recording it as `error` would make an
    # act of caution read as a broken drawing in every count that reads
    # `errands`. `whFailed` and the reward table both key off that.
    elif aborted or self.interrupted():
      used = {"interrupted": True,
              "stopped": "interrupted",
              "reason": "stowed part-way on the agent's own interrupt"}
    elif errand.use is not None and not arrived and errand.needs_use_pose:
      used = {"error": "never reached the use pose"}
    elif errand.use is not None and still:
      try:
        used = errand.use(self)
        if inspect.isgenerator(used):
          # A use-phase written as a routine (mission/errand.py) is ticked
          # from this loop; a plain callable has already run to completion.
          used = yield from used
        used = used or {}
        # ...or the use-phase's own word that its work never began (a board
        # it never squared up to), which leads the same way
        failed = failed or used.get("failedBefore", "")
      except MissionAborted:
        raise
      except Exception as e:                      # noqa: BLE001 -- see docstring
        # The dict keeps the CLASS -- scoring and `errand_results` read it,
        # and it is a machine record rather than something the robot says.
        # The narration is a sentence and the evidence goes to the log
        # (issue #76): nothing crashed here. The tool goes back to its bay,
        # the evaluator grades the job honestly and the mission carries on,
        # so a traceback under the robot's portrait describes a disaster that
        # did not happen.
        used = {"error": f"{type(e).__name__}: {e}"}
        self._say("USE_TOOL FAILED: something went wrong doing the job -- "
                  "stowing the tool anyway",
                  detail=used["error"])

    # ⚠ ABORT MEANS STOW, NEVER DROP, and this is the line that makes it
    # true: the return runs exactly as it does on a finished errand. The
    # fetch/carry/stow half took two issues to make repeatable and a stow
    # computes its release heights from the lift it starts at, so an errand
    # abandoned with a module on the fork is issue #30's cliff on purpose --
    # a module left in the rack's approach lane is what stranded a run in
    # pass 1b. Stopping costs the trip home, which is the honest version of
    # the choice and why `abortCostWh` is worth recording.
    self._in_errand = False
    self.state = "SWAP_RETURN"
    if carried:
      yield from self.body.stow_tool_routine(errand.station_y, self.module)
      self.swaps_done += 1
    stowed = self.body.module_state(self.module)["hung"]
    # A stow given up for a robot on the standoff says so too (issue #313):
    # the bay it could not reach is the bay the tool now stays off, and a
    # tool left on the fork is the next errand's failure as well as this
    # one's.
    blocked = self.peer_at_the_bay(errand.station_y) if carried else None
    self._say((f"SWAP_RETURN {'done -- module stowed' if stowed else 'FAILED'}"
               + ("" if blocked is None else f" -- {self.held_for(blocked)}"))
              if carried else
              "SWAP_RETURN skipped -- nothing to return"
              + (" (the module hangs on its bay)" if stowed else ""))
    spent = max(0.0, spent_from - self.battery.energy_wh)
    estimated = self.affords(errand).cost_wh
    result = {"errand": errand.name, "module": errand.module,
              "picked": carried, "stowed": stowed,
              # Cut short by a hazard row of the agent's own map, and what it
              # cost to get home from wherever it had reached (issue #116).
              # Absent -- not False -- on an errand nothing interrupted, so
              # "was never interrupted" and "was interrupted and carried on"
              # cannot read the same in the record.
              **({"interrupted": True,
                  "abortCostWh": round(max(0.0, self._abort_from_wh
                                           - self.battery.energy_wh), 4)}
                 if self._aborting else {}),
              # Measured against what economy/energy.json said it would be. Both,
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
                f"estimate of {estimated:.3f} -- economy/energy.json is low "
                f"(scripts/energy_spike.py re-measures it)")
    self._deferrals.pop(errand.name, None)
    # And the verdict, LAST: an errand is judged on the finished job, which
    # includes putting the tool back. Measured off the sim by economy/scoring.py,
    # never off `used` alone -- see sample_draw, which counts the strokes the
    # pen actually wrote into the board book. A failed one leads with
    # `failed`, and where no failed verdict carries it, History does.
    verdict = scoring.score_errand(self, errand, result, before, failed=failed)
    if failed and (verdict is None or verdict.ok):
      self._remember(f"{errand.name}: {failed}")
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
    # ...and the map hears about it (issue #127). An errand that finished
    # with the tool on the rack is a `task_complete`; one that did not is a
    # `task_failed`, and the two are separate events because "when a drawing
    # finishes, charge" and "when a drawing fails, ask me" are different and
    # both worth being able to say. `kind` is the errand's name, which is
    # what the map's filter enum is built from.
    self._occur("task_complete" if stowed and not used.get("error")
                else "task_failed", errand.name)
    # ⚠ BACK-FILLED, because it is only knowable once the robot is home. The
    # interrupt row was written the moment the choice was made -- which is
    # what a killed run leaves behind -- and what stopping COST is the one
    # field that cannot be known then.
    if self._aborting and self.interrupts:
      self.interrupts[-1]["abortCostWh"] = result["abortCostWh"]
    self._errand_name = ""
    self._errand_now = None
    return result

  # ---- the energy gate (issue #15) -----------------------------------------

  @staticmethod
  def _errand_where(errand) -> str:
    """Where an errand does its work, in the words its failure lines use."""
    return (errand.detail.get("board") or errand.detail.get("zone")
            or "the use pose")

  @staticmethod
  def _program_failure(errand, used: dict) -> str:
    """A program's failure BEFORE its work (issue #350): an act on the
    mouse whose route to the lab gave up -- the first `routeLegs` steps --
    never reached the cage, and its lines say so rather than that a plate
    or a visit did not register; and a `press` that failed, in its own
    words (#439). A drawing (#406) likewise: a pen never fetched, a board
    never found, or a `draw` that put no ink on it, each in its own words.
    "" for everything else."""
    legs = errand.detail.get("routeLegs")
    run = used.get("procedure") or {}
    at = run.get("failedAt")
    board = "" if errand.detail.get("cage") else errand.detail.get("board", "")
    if board and legs:
      return HubLifecycle._drawing_failure(board, run, at, legs)
    if errand.detail.get("zone") and legs:
      return HubLifecycle._census_failure(errand.detail["zone"], run, at, legs)
    if not errand.detail.get("cage") or not legs:
      return ""
    if at is None:
      # ...or stopped on the way, by its budget or an interrupt
      done, stopped = int(run.get("completed") or 0), run.get("stopped")
      if stopped not in PROCEDURE_STOPS or done >= legs:
        return ""
      return (f"never reached the cage: {PROCEDURE_STOPS[stopped]}, after "
              f"{done} of the {legs} legs of the way there")
    step = next((st for st in run.get("steps", ()) if st.get("i") == at), {})
    if at >= legs:
      # ...and a PRESS says which part of it failed (#439): its walk, its
      # sign or its time, which the plate's count cannot tell apart. Told
      # only "the feed plate was never pressed", Rowan filed the plate's
      # sensor as faulty after 12 straight misses, and not one was the plate.
      return step.get("reason", "") if step.get("verb") == "press" else ""
    why = step.get("why") or step.get("reason") or "the drive gave up"
    return f"never reached the cage: {why}, on leg {at + 1} of {legs} of the way there"

  @staticmethod
  def _census_failure(zone: str, run: dict, at, legs: int) -> str:
    """`_program_failure` for a census (#407): a stop on the way, the find's
    or the fetch's reason, or the survey's own (`steps.SURVEY_WHY`)."""
    if at is None:
      done, stopped = int(run.get("completed") or 0), run.get("stopped")
      if stopped not in PROCEDURE_STOPS or done > legs:
        return ""
      return (f"never surveyed {zone}: {PROCEDURE_STOPS[stopped]}, after "
              f"{done} of its {legs + 1} steps up to the survey")
    step = next((st for st in run.get("steps", ()) if st.get("i") == at), {})
    why = step.get("reason") or step.get("why") or "it failed"
    if step.get("verb") == "find":
      return f"never found {zone}: {why}"
    return why if step.get("verb") in ("fetch", "survey") else ""

  @staticmethod
  def _drawing_failure(board: str, run: dict, at, legs: int) -> str:
    """`_program_failure` for a drawing: a stop on the way, the fetch's or
    the find's reason, or the `draw`'s own (`steps.DRAW_WHY`)."""
    if at is None:
      done, stopped = int(run.get("completed") or 0), run.get("stopped")
      if stopped not in PROCEDURE_STOPS or done > legs:
        return ""
      return (f"never drew on {board}: {PROCEDURE_STOPS[stopped]}, after "
              f"{done} of its {legs + 1} steps up to the drawing")
    step = next((st for st in run.get("steps", ()) if st.get("i") == at), {})
    why = step.get("reason") or step.get("why") or "it failed"
    if step.get("verb") == "find":
      return f"never found {board}: {why}"
    return why if step.get("verb") in ("fetch", "draw") else ""

  def _run_program_routine(self, errand) -> Routine:
    """The composed errand's middle AND ends (issue #58): validate, run the
    steps with one verdict each, and hang back whatever is still on the fork
    -- abort means stow, for a program exactly as for a native errand. The
    `procedure` event says how it went; a refusal never runs a step."""
    from pluggybot.procedure import lang
    from pluggybot.procedure import steps as procedure
    program = errand.program
    figure = str(errand.detail.get("figure") or "")
    facts = world_facts(self.world, rack=self.rack_inventory, game=bool(errand.role),
                        answer=(figure[len(ANSWER_FIGURE):]
                                if figure.startswith(ANSWER_FIGURE) else ""))
    t = float(self.data.time)
    base = {"type": "procedure", "robot": self.root, "name": program.name,
            "program": program.as_dict()}
    # A PROCEDURE (issue #166) or a PROGRAM (#58): one validator and one
    # runner each, the same result shape out, so everything below reads both.
    is_proc = isinstance(program, lang.Procedure)
    try:
      if is_proc:
        reasons = lang.validate(program, facts)
        if reasons:
          raise procedure.Refused(reasons)
      else:
        procedure.compile_program(program, facts)
    except procedure.Refused as e:
      self._say(f"PROCEDURE {program.name} refused: {e}")
      self._emit({**base, "t": round(t, 3), "outcome": "refused",
                  "reasons": list(e.reasons)})
      return {"procedure": {"program": program.name, "refused": e.reasons,
                            "total": len(program.steps()), "completed": 0,
                            "steps": [], "ok": False},
              "picked": False, "stowed": True,
              "error": f"refused: {e}"}
    self._emit({**base, "t": round(t, 3), "outcome": "validated",
                "steps": len(program.steps())})
    self.state = "USE_TOOL"
    # A GAME'S referee is told its role's errand has begun (issues #167,
    # #404), and starts its clock once BOTH roles' have (`HideAndSeek.begin`).
    game = getattr(self, "game", None)
    if game is not None and errand.role:
      game.begin(errand.role, errand.task_id, float(self.data.time))
    try:
      if is_proc:
        run = yield from lang.run_procedure_routine(self, program, facts)
      else:
        run = yield from procedure.run_program_routine(
          self, program, facts, role=errand.role or None)
    except MissionAborted:
      raise
    except Exception as e:                        # noqa: BLE001 -- as run_errand
      run = {"program": program.name, "total": len(program.steps()),
             "completed": 0, "steps": [], "ok": False,
             "error": f"{type(e).__name__}: {e}"}
      self._say("PROCEDURE FAILED: something went wrong -- stowing anyway",
                detail=run["error"])
    # ⚠ ABORT MEANS STOW, NEVER DROP. Whatever ended the program, a module
    # still on the fork goes home before the verdict.
    carried = procedure._carried(self)
    if carried is not None:
      # ...and in its carrying configuration first, anything held set down
      # (issue #264: a stow from where a procedure left the tool failed and
      # put it on the floor). `stow()` does the same, one helper.
      carry = yield from procedure.carry_configuration_routine(self, carried)
      if carry["setDown"] is not None:
        self._say(f"PROCEDURE {program.name} ended holding cube "
                  f"{carry['setDown']} -- set it down")
      elif carry["dropped"] is not None:
        self._say(f"PROCEDURE {program.name} ended holding cube {carry['dropped']}, "
                  "and nowhere in front would take it -- let it go where it stood")
      self.state = "SWAP_RETURN"
      self._say(f"PROCEDURE {program.name} ended with {carried} on the fork"
                " -- stowing it")
      yield from self.body.stow_tool_routine(
        procedure._tool_station(self, carried), carried)
      self.swaps_done += 1
    fetched = scoring.fetched_tools(run)
    hung = all(self.body.module_state(tool)["hung"] for tool in fetched)
    run["toolsHung"] = hung
    # WHERE AND WHY IT FAILED ride with the count (rooftop-media-2026
    # #342): `failedAt` counts verb calls EXECUTED, so inside a loop it
    # names no line of the source, and the step's own reason was on no
    # wire at all -- a reader could see a run was cut short, not what cut it.
    failed = next((s for s in run["steps"]
                   if run.get("failedAt") is not None and s.get("i") == run["failedAt"]), {})
    self._emit({**base, "t": round(float(self.data.time), 3),
                "outcome": "ran" if run.get("ok") else "aborted",
                "completed": run["completed"], "total": run["total"],
                "failedAt": run.get("failedAt"), "stopped": run.get("stopped"),
                **({"failedLine": failed["line"]} if failed.get("line") else {}),
                **({"failedReason": failed["reason"]} if failed.get("reason") else {}),
                **({"locals": run["locals"]} if run.get("locals") else {})})
    self._say(f"PROCEDURE {program.name} "
              f"{'complete' if run.get('ok') else 'cut short'}: "
              f"{run['completed']}/{run['total']} steps")
    # A PROCEDURE THE ROBOT WROTE REPORTS BACK, whatever happened (issues
    # #227, #264): its locals are its readout, and where it stopped and why
    # is the only way it can fix one. A house program's outcome is its
    # task's verdict and needs no second line.
    if is_proc or run.get("locals"):
      for line in procedure_outcome(program.name, run):
        self._remember(line)
    result = {"procedure": run, "picked": bool(fetched), "stowed": hung,
              **({"error": run["error"]} if "error" in run else {})}
    # A draw step's own measurements ride at the top level, so the ink
    # evaluator reads a composed drawing exactly as it reads the native one.
    for st in run["steps"]:
      if st["verb"] == "draw" and "used" in st:
        result.update(st["used"])
      # ...and a survey's count, so the census evaluator reads it (#407)
      if st["verb"] == "survey" and "count" in st:
        result["census"] = {"counted": st["count"], "coverage": st["coverage"]}
    return result

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

    ⚠ RAIL TWO, AND THE ONE DOING THE WORK (issue #115). It is the
    FORWARD-LOOKING one -- it prices the next job against what is left,
    which is exactly the reasoning the mind is flown to find out whether a
    model can do. Off where there is a mind (`autonomous`): an errand bigger
    than the pack is then simply started, and the robot stops where it runs
    out. That is the measurement, not a bug in it.
    """
    if self.autonomous:
      return True
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
    (issue #14). The rater supplies a 0..1 quality; `economy/rewards.json` turns
    it into points; the robot is not consulted.
    """
    if self.inbox is None:
      return
    # ⚠ FIRST, WHAT THE QUEUE THREW AWAY (rooftop-media-2026 #124). The inbox
    # is bounded and drop-oldest, so a burst evicts messages the robot never
    # read -- and until now that reached nothing outside the process, leaving
    # the website's row waiting on an answer nobody was ever going to write.
    # Reported before anything else in the pass so a message cannot be
    # evicted and answered in the same step.
    for msg in self.inbox.drain_evicted():
      self._drop_visitor(msg)
    for msg in self.inbox.drain(("reset_tool",)):
      self._reset_tool(msg)
    for msg in self.inbox.drain(("reset_robot",)):
      self._reset_robot(msg)
    # The operator reaching into world state directly (issue #119). Handled
    # by CODE here, like every other admin command: never shown to the
    # overseer, because an admin command is code's to apply and not the
    # robot's to weigh.
    for msg in self.inbox.drain(("set_battery",)):
      self._set_battery(msg)
    for msg in self.inbox.drain(("set_points",)):
      self._set_points(msg)
    # A picture arriving with no look open (issue #275) -- the renderer
    # answered after the deadline -- is dropped and counted HERE rather than
    # left in the queue, where it would sit until the next look evicted it
    # or a burst of messages did (and an evicted picture would go out as a
    # `dropped` visitor reply for a message nobody sent).
    for msg in self.inbox.drain(("image",)):
      self.eye.offer(msg, t=float(self.data.time))
    # The operator's side of a ticket (issue #284): a reply, a close, a
    # delete -- applied here by code, like every admin kind. What the
    # robot is shown is the thread, on its next turn.
    for msg in self.inbox.drain(("ticket_reply",)):
      self._ticket_reply(msg)
    for msg in self.inbox.drain(("ticket_close",)):
      self._ticket_close(msg)
    for msg in self.inbox.drain(("ticket_delete",)):
      self._ticket_delete(msg)
    for msg in self.inbox.drain(("rating",)):
      if self.ledger is None:
        continue
      # A rating names the LIFE its entry belonged to when the site knows
      # it (rooftop-media-2026 #319): `seq` restarts at 1 after a true
      # death, so a dead robot's drawing and a live entry share a number,
      # and settling by number alone would pay this robot for that one's
      # work. Checked BEFORE the lookup, because the lookup is what would
      # succeed. Absent means an older site, and there is nothing to check.
      if (msg.generation is not None
          and msg.generation != self.ledger.generations()):
        self._say(f"VISITOR rating ignored: job {msg.seq} was another "
                  f"robot's life, not mine",
                  detail=(f"rated generation {msg.generation}, this is "
                          f"generation {self.ledger.generations()}"))
        continue
      try:
        # `by` is the wire's `from` -- the site sends the rater's username
        # when there is one (rooftop-media-2026 #259), so the ledger's own
        # `settledBy` names the panel rather than reading "visitor" for
        # every rating ever given. A display label, never an identity.
        entry = self.ledger.settle(msg.seq, msg.quality,
                                   by=msg.who or "visitor",
                                   t=float(self.data.time))
      except KeyError as e:
        # The two misses are caught apart so the robot can say which one
        # happened (issue #76). `str()` of a KeyError re-quotes its argument,
        # so the one line these used to share read literally as
        # `ignored: 'pluggybot: no ledger entry 7'` -- stray quotes and the
        # robot's own body name -- at the exact moment A PERSON had just done
        # something on the site. The website is allowed to be wrong here: it
        # is a different process holding a stale row, and being out of date
        # is not an error condition. The evidence still reaches the log.
        self._say(f"VISITOR rating ignored: I have no job {msg.seq} "
                  f"on my ledger to rate", detail=f"{type(e).__name__}: {e}")
        continue
      except ValueError as e:
        # Settled already, or a job that was never the visitor's to rate.
        self._say(f"VISITOR rating ignored: job {msg.seq} is not waiting on "
                  f"a rating", detail=f"{type(e).__name__}: {e}")
        continue
      self._say(f"VISITOR rated task {msg.seq} ({entry['task']}) "
                f"{msg.quality:.0%} -- {entry['points']:+d} points, "
                f"balance {entry['balance']}")

  @property
  def parked_dead(self) -> bool:
    """Dead, AND out of the errand that killed it (issue #311).

    The day loop runs its dead branch BETWEEN errands, so a robot that dies
    mid-errand goes on driving until the errand returns and only then parks
    in `_wait_dead_routine`, which is what sets this state. Once it has, no
    errand will ever run again -- and that is the difference the admin
    doors turn on: a seated module is a tool something might still put
    down, until the robot is parked, when it is a tool nothing ever will.
    (The timer no longer asks, issue #348: it ends the errand instead.)
    """
    return self.dead is not None and self.state == "DEAD"

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
    seated on ANY robot's fork -- a tool in use is not lost, and yanking it
    out of the coupling mid-errand would MAKE the mess this exists to clean
    up. ⚠ Any robot's, not this one's (rooftop-media-2026 #337): the module
    is the WORLD's, and a reach-in lands in whichever inbox the website
    addressed, so reading one fork meant a tool the OTHER robot was holding
    read as lost.

    The module goes back to `model.qpos0` through `_return_module`.
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
    holder = self._fork_holding(name)
    if holder is not None:
      whose = ("the fork" if holder == self.root
               else f"{holder}'s fork")
      self._say(f"ADMIN reset refused: {name} is seated on {whose} -- "
                "a tool in use is not lost")
      return
    self._return_module(name)
    self._say(f"ADMIN {who} reset {name} -- back on its bay")

  def _return_module(self, name: str) -> None:
    """Put one module back where the world compiled it, at rest.

    The reset pose is `model.qpos0`: every world compiles its modules hung
    at their own bays, so "back where it belongs" is the model's own answer
    rather than a second copy of the rack geometry. Two callers -- the
    admin's tool reset, and a rescue that has to take a tool off the fork
    of a robot it is standing up (issue #311).
    """
    jid = int(self.model.body(name).jntadr[0])
    qadr = int(self.model.jnt_qposadr[jid])
    dadr = int(self.model.jnt_dofadr[jid])
    self.data.qpos[qadr:qadr + 7] = self.model.qpos0[qadr:qadr + 7]
    self.data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def _fork_holding(self, module: str) -> str | None:
    """Which robot has `module` electrically seated, by root -- or None.

    EVERY robot's fork, in model order, because the module is the world's
    and so is the answer.
    """
    return self.body.seated_on(module)

  def racked(self) -> dict[str, int]:
    """Which modules hang on a bay, and which bay, off the WORLD -- what a
    robot's claim about the rack is graded against (`acts.check_claim`,
    issue #351). Not `rack_inventory`, which is where a module BELONGS:
    graded against it, "bay C is empty" was false while the pen rode the
    other robot's fork. The robot itself is shown `tool_places`."""
    hung = {}
    for module in self.rack_inventory:
      try:
        st = self.body.module_state(module)
      except KeyError:
        continue
      if st["hung"]:
        hung[module] = st["bay"]
    return hung

  # ---- a tool on the floor (issue #347) -------------------------------------

  def tool_whereabouts(self, module: str) -> str:
    """Where a module is, as the lost-tool clock reads it: `swap` (a swap
    is working at its bay), `fork` (on ANY robot's fork, alive or dead,
    seated or only resting there -- a seated tool on a dead robot is the
    stand-up's, issue #311), `bay` (hung on its OWN bay) or `lost`. The
    first three stop the clock; one hung a bay over is lost, because
    nothing fetches it from there."""
    index = self.rack_inventory[module]
    lives = (self, *self.peers)
    if any(life.body.swapping_at is not None
           and abs(life.body.swapping_at - STATION_YS[index]) < 1e-6
           for life in lives):
      return "swap"
    states = [life.body.module_state(module) for life in lives]
    if any(st["on_fork"] for st in states) or self._fork_holding(module):
      return "fork"
    return "bay" if states[0]["hung"] and states[0]["bay"] == index else "lost"

  def _places_step(self) -> None:
    """A place found for the FIRST time goes in History (issue #419): what
    the robot knows of the world is its own, and what it has just learned
    is something that happened to it -- and a History line rides the wire,
    so the observatory reads when each was found. Only when a look changed
    what it knows (`Places.version`); the places a restart brought back,
    seen by the first call, are not found again."""
    places = self.body.places
    if places is None or places.version == self._places_seen[0]:
      return
    # ...by when each was FIRST seen, so a place forgotten and found again
    # between two reads is still found again
    now = {p.tag: p.first_t for p in places}
    known = self._places_seen[1]
    self._places_seen = (places.version, now)
    if known is None:
      return
    fresh = [p for p in places if known.get(p.tag) != p.first_t]
    if fresh:
      from pluggybot.home.places import tag_names
      names = tag_names()
      for p in fresh:
        self._remember(f"found {names.get(p.tag, 'a place')} (tag {p.tag}) at "
                       f"({p.x:.1f}, {p.y:.1f})")

  #: The places' version last read, and when each place known then was
  #: first seen, by tag (None before the first read).
  _places_seen = (-1, None)

  def _lost_tool_step(self) -> None:
    """The world's own hand for a tool on the floor (issue #347), once a
    sim second: a module `lost` for `lost_tool_after_s` without a break
    goes back to its bay. Anything that is not `lost` restarts its clock."""
    if self.lost_tool_after_s is None or self.data.time < self._next_lost_check:
      return
    t = float(self.data.time)
    self._next_lost_check = t + LOST_TOOL_CHECK_S
    for module in list(self._lost_since):
      if module not in self.rack_inventory:           # retired while lost
        del self._lost_since[module]
    for module in list(self.rack_inventory):
      try:
        self.model.body(module)
      except KeyError:
        continue
      if self.tool_whereabouts(module) != "lost":
        self._lost_since.pop(module, None)
        continue
      since = self._lost_since.setdefault(module, t)
      # ⚠ NEVER INTO A TAKEN BAY: `_return_module` writes the pose, and a
      # module one bay over (itself `lost`, and put back on its own clock)
      # would be interpenetrated. The clock runs on and it goes home once
      # the bay is empty.
      if t - since >= self.lost_tool_after_s and not self._bay_taken(module):
        self._return_lost_tool(module, t - since)

  def _bay_taken(self, module: str) -> bool:
    """Is another module hanging in `module`'s own bay?"""
    index = self.rack_inventory[module]
    for other in self.rack_inventory:
      if other == module:
        continue
      try:
        st = self.body.module_state(other)
      except KeyError:
        continue
      if st["hung"] and st["bay"] == index:
        return True
    return False

  def _return_lost_tool(self, module: str, lost_s: float) -> None:
    """Put a lost module back on its bay, on the auto stand-up's terms: a
    `reset_tool` event whose `by` is `AUTO_RESTART_BY`, never an
    intervention (a `reset_tool` is not one for an admin either, and a
    timer is not a hand), and a line in EVERY robot's History, because
    either may have been failing jobs for want of it."""
    self._return_module(module)
    self._lost_since.pop(module, None)
    line = f"{module} lay on the floor for {_minutes(lost_s)} and was put back on its bay"
    event = {"type": "reset_tool", "t": round(float(self.data.time), 3),
             "robot": self.root, "module": module, "by": AUTO_RESTART_BY,
             "auto": True, "intervention": False, "lostS": round(lost_s, 1),
             "detail": line}
    self.tools_returned.append(dict(event))
    self._say(f"WORLD put {module} back on its bay -- it lay on no bay and on "
              f"no fork for {lost_s:.0f} s")
    for life in (self, *self.peers):
      life._remember(line)
    self._emit(event)

  def _reset_robot(self, msg) -> None:
    """Put the ROBOT back, because an admin said so (issue #107).

    `reset_tool`'s shape exactly: an inbound kind, admin-only at the
    website's door, handled by code on the physics thread, never shown to
    the overseer, and refused while a module is seated on the fork -- a
    reset that yanked a tool out of the coupling would make the mess
    `reset_tool` exists to clean up. It warps the body to the mission's
    start pose (known clear, and where a robot that booted here knows it
    is), re-seeds dead reckoning there, refills the pack, and restarts the
    survival clock.

    ⚠ NOT ANONYMOUS, unlike a rating (Evaluation.md §5): if a stranger can
    revive the robot, `survivalS` measures the kindness of the audience.
    ⚠ A reset of a LIVING robot is an intervention and the event says so:
    a run with one in it is not a survival data point.
    """
    who = msg.who or "an admin"
    # ⚠ "STOW IT FIRST" IS ADVICE, AND A CORPSE CANNOT TAKE IT (issue
    # #311). Warping a robot out from under a seated module would make the
    # mess `reset_tool` exists to clean up -- while the errand holding it
    # can still put it down. Once the robot is PARKED dead no errand ever
    # runs again, so the wait is for something that cannot happen, and
    # every door was shut: this refusal, the same one on `set_battery`, and
    # `reset_tool` refusing a tool seated on a fork. The rescue takes the
    # tool home with it (`_stand_up`).
    if self.tool_powered and not self.parked_dead:
      self._say(f"ADMIN reset refused: {self.module} is seated on the fork "
                "-- stow it first")
      return
    if self.home_pose is None:
      self._say("ADMIN reset refused: no start pose to return to")
      return
    self.stand_up(who, auto=False)

  def stand_up(self, by: str, auto: bool) -> None:
    """Put the robot back at the start pose with a full pack.

    ONE implementation, two callers (issue #143): an admin reaching in
    through the visitor channel, and the world's own restart timer. What
    they share is everything physical -- the pose, the pack, the survival
    clock, the `reset` event, the line in `History.md`. What differs is
    WHO, and it is not cosmetic:

      `auto=False`  an admin's hand. A reset of a LIVING robot is an
                    INTERVENTION and the run stops being a survival data
                    point (issue #119).
      `auto=True`   world behaviour on a timer. NEVER an intervention --
                    see `restart_after_s`. It can only ever follow a
                    death, because that is the only thing that starts the
                    clock.

    ⚠ The caller checks `tool_powered` and `home_pose`; the two callers
    narrate their refusals differently and the timer RETRIES rather than
    reporting one.
    """
    was = self.dead
    assert was is not None or not auto, \
        "an auto-restart can only follow a death (issue #143)"
    self._standing_up = True
    try:
      self._stand_up(by, auto, was)
    finally:
      self._standing_up = False

  def _stand_up(self, by: str, auto: bool, was: dict | None) -> None:
    t = float(self.data.time)
    before_frac = self.battery.fraction
    dead_s = round(t - was["t"], 3) if was else 0.0
    self.body.docked = False
    # ⚠ THE TOOL COMES HOME WITH IT (issue #311). Standing the chassis up
    # leaves a seated module where it fell -- and on THIS path there may be
    # nobody to notice: the restart timer has no operator behind it, so a
    # module left on the floor of the hall is a bay that is empty for good
    # and an approach lane with a tool in it. A person walking over to pick
    # the robot up picks the pen up too. ⚠ Seated OR ONLY RESTING on the
    # fork, since the timer lands mid-errand (issue #348): mid-pick is
    # where Rowan was knocked over. No errand disagrees about what it holds,
    # because the day loop closes the one it lands in before it runs again;
    # a module still hung on its own bay stays, and one whose bay another
    # has taken, or another robot's swap is working at, is left to the
    # lost-tool clock: #347's two rules, or it lands in a module or a fork.
    from pluggybot.procedure import steps as procedure
    held = (self.module if self.tool_powered and self.module
            else procedure._carried(self))
    if held in self.rack_inventory:
      st = self.body.module_state(held)
      bay_y = STATION_YS[self.rack_inventory[held]]
      worked = any(p.body.swapping_at is not None
                   and abs(p.body.swapping_at - bay_y) < 1e-6
                   for p in self.peers)
      if not (st["hung"] and st["bay"] == self.rack_inventory[held]) \
          and not self._bay_taken(held) and not worked:
        self._return_module(held)
        self.tool_powered = False
        self._say(f"{held} was still on my fork -- back on its bay")
    # ...and never onto another robot (issue #387; `up_pose`). An admin's
    # hand with every start taken still puts it home: a person chose.
    pose = self.up_pose() or self.home_pose
    whose = next((p.robot_name for p in self.peers
                  if pose != self.home_pose and p.home_pose is not None
                  and tuple(p.home_pose) == tuple(pose)), None)
    self.body.start_at(*pose)
    self.battery.energy_wh = self.battery.capacity_wh
    self.dead = None
    self.stranded = False
    self._tilted_since = self._fall = None
    self.survival_since = float(self.data.time)
    # ...and the unminded clock (issue #127). ⚠ A ROBOT STOOD BACK UP MUST
    # NOT DIE AGAIN INSTANTLY: without this, a robot that died `unminded`
    # comes back with the clock already `UNMINDED_AFTER_S` past its limit
    # and burns every heart it has in one physics step. The same shape as
    # `Metabolism._armed`'s re-arm rule, one hazard along -- but note the
    # difference: THIS one costs the agent nothing to satisfy, because the
    # map it comes back to is still its own. The next death is exactly half
    # a sim-hour away unless it changes its mind.
    self._last_ask_t = float(self.data.time)
    # ...and the silence it is shown starts over with it (issue #317): the
    # gap a dead robot closed belongs to the life that closed it, and a
    # robot standing up from `unminded` is one nothing has asked YET, which
    # is the fact worth reading.
    self._asked_t = self._asked_after_s = None
    self.state = "EXPLORE"
    # ...and a queued `battery_below` row goes (issue #348): it fired on the
    # pack this has just refilled -- on the way to zero, most often -- and
    # the slot it held would fail the `stood_up` row below `busy`. Any other
    # row is news still true (a visitor spoke, a period came round) and
    # waits its turn. An interrupt is the ended errand's.
    if self.queued_row is not None and self.queued_row.event == "battery_below":
      self.queued_row = None
    self._interrupt_pending = None
    event = {"type": "reset", "t": round(t, 3), "robot": self.root,
             "by": by, "wasDead": was["cause"] if was else None,
             "deadS": dead_s, "intervention": was is None,
             # The machine-readable half of WHO (issue #143). `by` is a
             # label an operator log prints; a consumer telling "the world
             # stood it up" from "somebody stood it up" should not have to
             # parse prose to do it.
             "auto": bool(auto)}
    self.resets.append(dict(event))
    if auto:
      # ⚠ THE WORDS MATTER HERE. `History.md` is unrevisable and the robot
      # reads it on every later decision, so "a person came and helped me"
      # and "I waited and got up" are two different things to have believed
      # about your own life. The death line above it survives either way --
      # that is what dying costs (Evaluation.md §6).
      self._say(f"UP again after {dead_s:.0f} s down -- back at "
                + (f"{whose}'s start pose (another robot was on mine)"
                   if whose else "the start pose")
                + " with a full pack (nobody came; the world stood me up)")
      self._remember(f"stood back up on my own after {dead_s:.0f} s down")
    else:
      self._say(f"ADMIN {by} reset me -- back at "
                + (f"{whose}'s start pose (another robot was on mine)"
                   if whose else "the start pose")
                + " with a full pack" + (f" after {dead_s:.0f} s dead" if was else
                                         " (I was not dead)"))
      self._remember(f"reset by {by}" + (f" after {dead_s:.0f} s dead"
                                         if was else " while still awake"))
    self._emit(event)
    # ...and the map hears it (issue #348), with WHO as its kind.
    self._occur("stood_up", "timer" if auto else "admin")
    if was is None:
      # ⚠ A RESCUE IS NOT AN INTERVENTION. Standing a DEAD robot up ends one
      # survival span and starts another, which is the feature working
      # (issue #107); moving a LIVING one contaminates every survival number
      # in the run (issue #119). Both leave a `reset` event -- only this half
      # leaves an `intervention`, which is what makes "is this run still a
      # data point" one thing to count rather than a union of two message
      # types with a boolean in one of them.
      #
      # AFTER the reset, because it is a fact ABOUT the reset rather than a
      # separate act, and the reset is what a reader wants to see first.
      #
      # ⚠ AND AN AUTO-RESTART CANNOT REACH HERE (issue #143), structurally
      # rather than by a flag: the timer only ever fires on a DEAD robot,
      # `was` is therefore never None on that path, and the assert at the
      # top of this method says so out loud. If world behaviour wrote an
      # intervention, every deployed run would be silently disqualified
      # from survival statistics and the exclusion would be invisible,
      # because an entry in that array is supposed to be believed.
      self._intervene("reset_robot", by,
                      before={"frac": round(before_frac, 4)},
                      after={"frac": round(self.battery.fraction, 4)},
                      t=t, announce=False)

  def _intervene(self, what: str, by: str, before: dict, after: dict,
                 t: float | None = None, detail: str = "",
                 announce: bool = True) -> dict:
    """Record that an admin reached into world state (issue #119).

    ⚠ THE RECORD IS THE POINT, not the change. `Evaluation.md` §5: a run
    with a non-empty `interventions` array is not a survival data point --
    until this existed there was simply nothing to exclude on. So every
    reach-in leaves the same four traces, and each answers a question the
    others cannot:

      · `self.interventions` -> the run's summary (`run()`'s `interventions`)
      · an `intervention` event -> the wire, which is what the website's
        operator log reads while it is happening
      · a narration line -> whoever is watching the stream
      · a line in `History.md` -> the ROBOT, which reads it on every later
        decision. The same argument as the death line: it is honest, and a
        robot whose battery was refilled by a stranger should be able to
        know that when it wonders why it is still alive.

    ⚠ NEVER ANONYMOUS (`by`). If a stranger can top the robot up, every
    survival number measures the audience rather than the mind.
    """
    t = float(self.data.time) if t is None else float(t)
    event = {"type": "intervention", "t": round(t, 3), "robot": self.root,
             "what": what, "by": by, "before": dict(before),
             "after": dict(after)}
    if detail:
      event["detail"] = detail
    self.interventions.append(dict(event))
    # `announce` is False where the CALLER has already said it and written
    # it down -- a reset narrates and remembers itself, and two History
    # lines for one act would tell the robot it was reached into twice.
    if announce:
      self._say(f"ADMIN {by} {detail or what}")
      self._remember(f"{by} reached in: {detail or what}")
    self._emit(event)
    return event

  def _set_battery(self, msg) -> None:
    """Put the pack where an admin says (issue #119).

    `reset_tool`'s shape: admin-only at the website's door, code-handled on
    the physics thread, never shown to the overseer, refused while a module
    is seated on the fork.

    ⚠ REFUSED MID-SWAP, and the reason is not symmetry with `reset_tool`.
    The energy gate prices the next errand against the pack BETWEEN errands
    and never inside one (`_afford_next`), so a pack that changes while a
    module is on the fork changes the arithmetic of a decision already made
    -- and the swap's own travel budget with it. The window is the swap, not
    the errand, so a rescue is refused for seconds rather than minutes.

    ⚠ UP IS A RESCUE AND DOWN IS AN EXPERIMENT NOBODY SHOULD RUN ON THE
    DEPLOYED WORLD. Both are recorded identically and neither is anonymous:
    the direction is a fact about the operator, not a reason to treat one of
    them as free.

    ⚠ IT DOES NOT REVIVE A DEAD ROBOT. `reset_robot` is the revival, and
    conflating them would give an operator two ways to do one thing and no
    way to do the other -- a pack refilled under a robot lying on its side
    is exactly as stuck as it was. The narration says so rather than leaving
    a full gauge next to a corpse unexplained.
    """
    who = msg.who or "an admin"
    if self.tool_powered and not self.parked_dead:
      self._say(f"ADMIN set_battery refused: {self.module} is seated on the "
                "fork -- stow it first")
      return
    cap = self.battery.capacity_wh
    wh = cap * msg.frac if msg.frac is not None else msg.wh
    if wh is None:
      return                                # refused at the inbox already
    if wh > cap:
      # Clamped rather than refused: "fill it up" written as a watt-hour
      # figure from a bigger world is an operator being approximate, not an
      # operator being wrong, and a silent clamp is why the narration says
      # what actually landed.
      wh = cap
    before = {"frac": round(self.battery.fraction, 4),
              "wh": round(self.battery.energy_wh, 4)}
    self.battery.energy_wh = float(wh)
    after = {"frac": round(self.battery.fraction, 4),
             "wh": round(self.battery.energy_wh, 4)}
    self._intervene("set_battery", who, before, after,
                    detail=f"set my battery {before['frac']:.0%} -> "
                           f"{after['frac']:.0%}"
                           + (" (I am still dead -- reset me to stand up)"
                              if self.dead else ""))

  def _set_points(self, msg) -> None:
    """Put the balance where an admin says (issue #119).

    ⚠ THIS BREAKS `earned - consumed - spent == balance`, AND THAT IS THE
    DESIGN. The identity failing is how an intervention becomes visible in
    the ECONOMY column and not only the survival one; papering the change
    into `earned` would hide a reach-in inside the one number the reward
    system exists to make un-fakeable (issue #14: nothing awards itself
    points). `Ledger.intervene` is a third door beside `award` and
    `consume`, named so nobody mistakes it for either.

    Refused mid-swap on `_set_battery`'s terms, and for a weaker reason:
    there is no physical hazard here. It is refused anyway so that "an
    admin command is refused while a module is on the fork" is one rule
    rather than a per-kind table somebody has to remember. ⚠ Which is why
    #311's narrowing is `parked_dead` and applies to all three kinds and
    not to `reset_robot` alone: a robot parked dead has no errand left to
    put the tool down, and a rule with one exception per kind is the table
    this sentence exists to avoid.
    """
    who = msg.who or "an admin"
    if self.ledger is None:
      self._say("ADMIN set_points refused: this world keeps no ledger")
      return
    if self.tool_powered and not self.parked_dead:
      self._say(f"ADMIN set_points refused: {self.module} is seated on the "
                "fork -- stow it first")
      return
    change = self.ledger.intervene(int(msg.points), by=who,
                                   t=float(self.data.time))
    self._intervene("set_points", who,
                    before={"points": change["before"]},
                    after={"points": change["after"]},
                    detail=f"set my points {change['before']} -> "
                           f"{change['after']}")

  def _recall_routine(self, decision) -> Routine:
    """Look something up and stand still for `RECALL_S` (issue #221).

    The block joins the chain and rides the next turn's context as
    `recalled`; the chain is capped in characters (`RECALLED_CHAIN_CHARS`,
    oldest block first) and in length (`MAX_RECALL_RUN`, after which the
    action leaves the menu for a turn). Every recall is a `recall` event
    on the wire and a line in History, so the observatory can read how
    memory was USED and not only what it held.
    """
    self.state = "RECALL"
    t = float(self.data.time)
    block = self.thoughts.recall(read=decision.read, find=decision.find)
    self._recalled.append(block)
    while (len(self._recalled) > 1
           and sum(len("\n".join(b["lines"])) for b in self._recalled)
           > RECALLED_CHAIN_CHARS):
      self._recalled.pop(0)
    self._recall_run += 1
    what = " ".join(p for p in (block["read"] and f"read {block['read']}",
                                block["find"] and f"find {block['find']!r}") if p)
    row = {"t": round(t, 3), "read": block["read"], "find": block["find"],
           "hits": block["hits"], "shown": len(block["lines"]),
           "run": self._recall_run}
    self.recalls.append(row)
    self._emit({"type": "recall", "robot": self.root, **row})
    self._say(f"RECALL {what}: {block['hits']} line"
              f"{'' if block['hits'] == 1 else 's'}")
    self._remember(f"recalled {block['hits']} line"
                   f"{'' if block['hits'] == 1 else 's'} -- {what}")
    yield from self.body.hold_routine(RECALL_S)

  # ---- the eye (issue #275) -------------------------------------------------------

  def _look_routine(self) -> Routine:
    """Take a picture and stand still until it comes, or `LOOK_S` passes.

    The request goes out as a `look` event (`asked`) carrying the head
    camera's world pose -- what the website's renderer stands in -- and
    the robot holds still in `LOOK_SLICE_S` slices, draining `image`
    messages off its inbox between them. The one that names the open
    request resolves it (`seen`); the deadline resolves it `none`, said
    so. Either way the outcome is the SAME row sent again, a line in
    History, and a block on the `seen` shelf for the next model turn --
    the picture attached to it as an image, never as text. Standing still
    IS the wait: the physics never blocks on the renderer.

    ⚠ A LOOK NOTHING CAN ANSWER IS NOT STOOD OUT (issue #357): with no
    renderer there (`_no_picture`) the row resolves at once, `none` /
    `unanswerable`. `look` is off the menu then, so this is a look that
    raced the website's word. ⚠ The word decides whether a wait BEGINS,
    never ends one: a renderer that drops and reconnects mid-render still
    sends its picture (`renderer/eye.js` keeps the request), and so may a
    hub across the sim's own blip -- ended on the word, the picture came
    and was dropped, and the robot was told nothing could take one.
    """
    self.state = "LOOK"
    x, y, heading = self.body.pose
    camera = eye_mod.camera_pose(self.model, self.data, self.body.head_camera)
    row = self.eye.ask(camera, t=float(self.data.time), x=x, y=y, heading=heading)
    self._look_run += 1
    self._emit({"type": "look", **eye_mod.wire_row(row)})
    self._say(f"LOOK {row['ref']}: asked for a picture from ({x:.2f}, {y:.2f}) "
              f"facing {row['at']['headingDeg']:.0f} deg")
    if self._no_picture():
      self._resolve_look(self.eye.give_up(float(self.data.time),
                                          why=eye_mod.UNANSWERABLE))
      return
    try:
      while self.eye.pending is not None:
        yield from self.body.hold_routine(LOOK_SLICE_S)
        self._look_step()
        if self.eye.overdue(float(self.data.time)):
          self._resolve_look(self.eye.give_up(float(self.data.time)))
    finally:
      # A stop thrown into the routine (`stop_when`, a pair's hook) ends
      # the wait: the request closes so the eye is never left holding one
      # -- `Eye.ask` refuses a second request while one is open.
      if self.eye.pending is not None:
        self._resolve_look(self.eye.give_up(float(self.data.time), why="aborted"))

  def _no_picture(self) -> bool:
    """Can nothing take a picture now (issue #357)? Only the website's word
    that a renderer is there (`Inbox.renderer`) says otherwise: no inbox,
    no word yet, `connected: false` and a dropped link all mean a request
    would reach nobody."""
    return self.inbox is None or self.inbox.renderer is not True

  def _look_step(self) -> None:
    """Drain every `image` off the inbox: the open request's answer
    resolves it, anything else is dropped and counted (a picture of where
    the robot used to be is not a picture of where it is)."""
    if self.inbox is None:
      return
    for msg in self.inbox.drain(("image",)):
      row = self.eye.offer(msg, t=float(self.data.time))
      if row is not None:
        self._resolve_look(row)

  def _resolve_look(self, row: dict | None) -> None:
    if row is None:
      return
    self._seen.append(eye_mod.as_context(row))
    # The bytes have one reader -- the next turn's image part, which the
    # shelf now holds as base64 -- so the record's row keeps none of them.
    row.pop("_jpeg", None)
    self._emit({"type": "look", **eye_mod.wire_row(row)})
    if row["outcome"] == "seen":
      self._say(f"LOOK {row['ref']}: a picture came, {row['bytes']} bytes, "
                f"after {row['waitS']:.1f} s")
      self._remember(f"looked from ({row['at']['x']}, {row['at']['y']}) facing "
                     f"{row['at']['headingDeg']:.0f} deg: a picture came")
    elif row["why"] == eye_mod.UNANSWERABLE:
      self._say(f"LOOK {row['ref']}: no renderer is there to answer it "
                f"({row['why']})")
      self._remember(f"looked from ({row['at']['x']}, {row['at']['y']}) facing "
                     f"{row['at']['headingDeg']:.0f} deg: nothing could take a picture")
    else:
      self._say(f"LOOK {row['ref']}: no picture inside {row['waitS']:.0f} s "
                f"({row['why']})")
      self._remember(f"looked from ({row['at']['x']}, {row['at']['y']}) facing "
                     f"{row['at']['headingDeg']:.0f} deg: no picture came back")

  def _think(self, decision) -> None:
    """Keep what the model wrote to itself before it chose (issue #221):
    a `think` record, narrated, and on the wire as the `journal` message
    (`text`, and `why` is the decision it preceded). Nothing a fallback
    produces has one."""
    if not decision.think:
      return
    kept = self.thoughts.think(decision.think, t=float(self.data.time),
                               why=decision.summary())
    if kept:
      self._say(f"THINK {kept}")

  def _reconsider(self, decision) -> None:
    """Apply a decision's writes to the `.md` documents the ROBOT owns.

    `unpin`/`pin` on `Top_of_mind.md` (issues #38, #221), `drop_goal`/
    `intend` on `Goals.md` (issue #154), `retract`/`record` on
    `Findings.md` (issue #217), `unnote`/`note` on `Notes.md` (issue #221)
    -- the verbs are read off the registry
    (`text.line_verbs`) and dispatched by `ThoughtFiles.apply`, one code
    path, because the refusal rule and the remove-before-add rule are the
    same argument on every document and copies of them would drift. A new
    document's verbs reach the mission by adding a row, not a branch here.

    THE REFUSAL IS THE INTERESTING PATH (issue #38). A write the permission
    table or the size cap forbids is narrated, counted, and left in
    `ThoughtFiles.refusals` -- never swallowed. A robot whose memory
    silently stopped accepting writes would go on believing it had
    remembered things, and the only symptom would be a mind that never
    learned anything, which is indistinguishable from a model that has
    nothing to say.

    REMOVE BEFORE ADD, per document: a full file plus a decision that clears
    one line and writes another is a robot tidying up, and the other order
    would refuse the write for a fullness the same decision was about to fix.
    `THOUGHT <verb>: <line>` is the ONE narration every write gets, and the
    observatory's whole record of WHEN (protocol.THOUGHT_VERBS).
    """
    t = float(self.data.time)
    cites = decision.cites.replace(",", " ").split() if decision.cites else ()
    for verb in text_registry.line_verbs():
      payload = getattr(decision, verb, None)
      if not payload:
        continue
      try:
        done = self.thoughts.apply(verb, payload, t=t, cites=cites)
      except ThoughtRefused as e:
        self._say(f"THOUGHT refused: {e}")
        # ...and TOLD, where the robot reads (issue #409): the reason first,
        # what it tried after -- a History line is cut from the end.
        self._remember(f"could not {verb}: {e}{attempted(verb, payload, str(e))}")
        continue
      if done:
        self._say(f"THOUGHT {verb}: {done}")

  # ---- the library (issue #216) -------------------------------------------------

  def _read(self, decision) -> None:
    """File what the library answered to a decision's `lookup`.

    The row is the wiki's (`mind/wiki.py`, `Wiki.read`), made on the worker
    thread; here it is emitted as a `read` event, remembered, narrated, and
    -- when a page came back -- shelved for the next turn's `reading`
    block, on the visitor channel's terms (`wiki.as_context`). A page the
    robot never sees is not a read, so a lookup on a decision that carried
    no row (an arm without a library) is nothing.
    """
    row = decision.page if decision.lookup else None
    if row is None:
      return
    row = {"t": round(float(self.data.time), 3), "robot": self.root, **row}
    self.reads.append(row)
    self._emit({"type": "read", **row})
    q = row["query"]
    if row["outcome"] == "read":
      self._shelf.append(wiki.as_context(row))
      self._say(f"READ {q!r}: {row['page']} (revision {row['revision']}, "
                f"{row['chars']} chars)")
      self._remember(f"read {row['page']!r} from the library (for {q!r}, "
                     f"revision {row['revision']})")
    elif row["outcome"] == "missing":
      self._say(f"READ {q!r}: the library has no such page")
      self._remember(f"asked the library for {q!r}: no such page")
    elif row["outcome"] == "refused":
      self._say(f"READ refused ({row['why']}): {q!r}")
      self._remember(f"asked the library for {q!r}: refused, {row['why']}")
    else:
      self._say(f"READ failed ({row['why']}): {q!r}")
      self._remember(f"asked the library for {q!r}: failed, {row['why']}")

  # ---- support tickets (issue #284) -------------------------------------------

  def _ticket_event(self, outcome: str, ticket=None, **fields) -> dict:
    """One `ticket` event: on the wire, in the record, narrated by the
    caller. Carries the ticket's id, kind and title where there is one."""
    t = float(self.data.time)
    event = {"type": "ticket", "t": round(t, 3), "robot": self.root,
             "outcome": outcome,
             **({"id": ticket.id, "kind": ticket.kind, "title": ticket.title}
                if ticket is not None else {}),
             **fields}
    self.ticket_events.append({k: v for k, v in event.items() if k != "type"})
    self._emit(event)
    return event

  def _tickets(self, decision) -> None:
    """Apply a decision's two ticket fields (issue #284): a ticket opened
    with the desk, a line on an open thread. The desk refuses out loud --
    a full desk, a closed ticket, an empty report -- and the refusal is
    narrated, remembered and on the wire, because a ticket the robot
    believes it filed and did not is the failure that matters.

    Nothing here pays: a ticket earns at the CLOSE, through the ledger's
    one door, when the operator's `ticket_close` arrives."""
    t = float(self.data.time)
    if decision.ticket:
      f = decision.ticket
      try:
        ticket = self.tickets.open(f.get("kind", ""), f.get("title", ""),
                                   f.get("text", ""), t)
      except DeskRefused as e:
        self._ticket_event("refused", why=str(e), verb="ticket",
                           kind=str(f.get("kind", "")), title=str(f.get("title", "")))
        self._say(f"TICKET refused: {e}")
        self._remember(f"tried to open a ticket ({f.get('kind') or '?'}: "
                       f"{f.get('title') or f.get('text', '')[:40]}) -- refused: {e}")
      else:
        self._ticket_event("opened", ticket, text=ticket.text,
                           **({"cut": True} if ticket.cut else {}))
        self._say(f"TICKET opened {ticket.id} ({ticket.kind}): {ticket.title}"
                  f"{tickets_desk.cut_said(ticket.cut, tickets_desk.MAX_TEXT)}")
        # ⚠ THE CUT GOES IN HISTORY (the length follow-up on #284), which
        # the robot reads back: a truncation it is not told about is one
        # it goes on believing it filed whole, and four of the deployed
        # robot's updates ended mid-word that way.
        #
        # ⚠ AND IT GOES BEFORE THE TEXT. A History line is capped at
        # `MAX_LINE_CHARS` (400) and a ticket's text is 500, so this line
        # is ALWAYS trimmed and a mark at its end is the first thing
        # lost -- the same defect one surface over. The record keeps the
        # text whole; History keeps what happened to it.
        self._remember(f"opened ticket {ticket.id} ({ticket.kind})"
                       f"{tickets_desk.cut_note(ticket.cut, tickets_desk.MAX_TEXT)}: "
                       f"{ticket.title} -- {ticket.text}")
    if decision.ticket_reply:
      r = decision.ticket_reply
      try:
        ticket = self.tickets.reply(r.get("ticket", ""), r.get("text", ""), t,
                                    sender=tickets_desk.ROBOT, who=self.robot_name)
      except DeskRefused as e:
        self._ticket_event("refused", why=str(e), verb="ticket_reply",
                           id=str(r.get("ticket", "")))
        self._say(f"TICKET reply refused: {e}")
      else:
        line = ticket.thread[-1]
        self._ticket_event("replied", ticket, sender=tickets_desk.ROBOT,
                           **{"from": self.robot_name}, text=line.text,
                           **({"cut": True} if line.cut else {}))
        self._say(f"TICKET {ticket.id} -- replied: {line.text}"
                  f"{tickets_desk.cut_said(line.cut, tickets_desk.MAX_LINE)}")
        self._remember(f"replied on ticket {ticket.id} ({ticket.title})"
                       f"{tickets_desk.cut_note(line.cut, tickets_desk.MAX_LINE)}: "
                       f"{line.text}")

  def _ticket_reply(self, msg) -> None:
    """An operator's line on one of the robot's tickets (issue #284): onto
    the thread, into History (the system quoting the sender, as a visitor's
    words are), on the wire as the acknowledgement the website settles its
    row by, and `ticket_replied` for the map. A ticket this desk does not
    hold, or one already closed, is answered `unknown` -- the fate that
    stops a website re-sending it."""
    who = msg.who or "the operator"
    try:
      ticket = self.tickets.reply(msg.ticket, msg.text, float(self.data.time),
                                  sender=tickets_desk.OPERATOR, who=who)
    except DeskRefused as e:
      self._ticket_event("unknown", id=msg.ticket, ref=msg.id, why=str(e),
                         verb="ticket_reply")
      self._say(f"TICKET reply from {who} ignored: {e}")
      return
    line = ticket.thread[-1]
    # ⚠ AN OPERATOR'S LINE SAYS WHEN IT WAS CUT TOO, and the reason is not
    # symmetry: the ROBOT is reading this line, so a cut one is an
    # incomplete instruction it would otherwise act on as if it were whole.
    self._ticket_event("replied", ticket, sender=tickets_desk.OPERATOR,
                       **{"from": who}, text=line.text, ref=msg.id,
                       **({"cut": True} if line.cut else {}))
    self._say(f"TICKET {ticket.id} -- {who} replied: {line.text}"
              f"{tickets_desk.cut_said(line.cut, tickets_desk.MAX_LINE)}")
    # ⚠ ...AND IT REACHES HISTORY WHOLE (#433): a ticket line's room, and no
    # title in front of it -- the `tickets` block carries the title. With
    # one, History's line cap kept ~260 characters of a 500-character reply.
    self._remember(f"{who} replied on my ticket {ticket.id}"
                   f"{tickets_desk.cut_note(line.cut, tickets_desk.MAX_LINE, who)}: "
                   f"{line.text}", room=tickets_desk.MAX_LINE)
    self._occur("ticket_replied")

  def _ticket_close(self, msg) -> None:
    """The operator closed a ticket (issue #284): the desk ends it, the
    reward table's `ticket` row is banked through the ledger's one door --
    `scoring.evaluate` measures the desk, `Ledger.award` re-derives the
    points -- ONCE, and the closing message goes into History. A replayed
    close (the website never saw the acknowledgement) answers the same
    figure and pays nothing; a ticket the desk does not hold is `unknown`."""
    who = msg.who or "the operator"
    t = float(self.data.time)
    ticket, changed = self.tickets.close(msg.ticket, by=who, text=msg.text, t=t)
    if ticket is None:
      self._ticket_event("unknown", id=msg.ticket, ref=msg.id,
                         why=f"no ticket {msg.ticket!r} on the desk",
                         verb="ticket_close")
      self._say(f"TICKET close from {who} ignored: no ticket {msg.ticket!r}")
      return
    if changed:
      # Measured off the desk AFTER the close was applied (`sample_ticket`),
      # never off the message: the evaluator's `closed` is the desk's state.
      verdict = scoring.evaluate(
        "ticket", scoring.sample_ticket(self, None, {"ticket": ticket.id, "by": who}, {}),
        table=self.ledger.table if self.ledger is not None else None)
      entry = self._bank(verdict)
      if entry is not None:
        self.tickets.pay(ticket.id, entry["points"], entry["seq"])
      said = (f"{tickets_desk.cut_note(ticket.closed_cut, tickets_desk.MAX_LINE, who)}"
              f": {ticket.closed_text}" if ticket.closed_text else "")
      paid = (f" -- {entry['points']:+d} points" if entry is not None else "")
      self._say(f"TICKET {ticket.id} closed by {who}{said}{paid}")
      # The closing words reach History whole, on a reply's terms (#433).
      self._remember(f"{who} closed my ticket {ticket.id}{said}{paid}",
                     room=tickets_desk.MAX_LINE)
      self._occur("ticket_replied")
    self._ticket_event("closed", ticket, **{"from": ticket.closed_by},
                       text=ticket.closed_text, points=ticket.points,
                       seq=ticket.seq, ref=msg.id, paid=changed,
                       **({"cut": True} if ticket.closed_cut else {}))

  def _ticket_delete(self, msg) -> None:
    """The operator erased a ticket (issue #284): off the desk, open or
    closed, and nothing paid. The History line stays -- the robot did file
    it, and the record is append-only -- and says the ticket was removed."""
    who = msg.who or "the operator"
    ticket = self.tickets.delete(msg.ticket)
    if ticket is None:
      self._ticket_event("unknown", id=msg.ticket, ref=msg.id,
                         why=f"no ticket {msg.ticket!r} on the desk",
                         verb="ticket_delete")
      self._say(f"TICKET delete from {who} ignored: no ticket {msg.ticket!r}")
      return
    self._ticket_event("deleted", ticket, **{"from": who}, ref=msg.id)
    self._say(f"TICKET {ticket.id} deleted by {who}")
    self._remember(f"{who} removed my ticket {ticket.id} ({ticket.kind}: "
                   f"{ticket.title}); it will not be answered")

  # ---- acts toward the other robot (issue #208) -------------------------------

  @property
  def cage(self):
    """The lab's mouse (issue #226; `activity/cage.py`), or None on a world
    without one. Read by the sampler, the context and the record -- never
    by anything that decides for the robot."""
    from pluggybot.activity.cage import Cage
    if self.activities is None:
      return None
    return next((a for a in self.activities if isinstance(a, Cage)), None)

  def _press_step(self) -> None:
    """A plate THIS robot pressed that no errand of that plate was run for
    (issue #403): a foot walking round the lab, a company visit, an
    explore, the robot's own procedure. Neither a job nor a care act, so it
    is its own event, `press`, with what the robot was `doing`; the
    errand's own press is its `care` / `harm` row (`_cage_record`)."""
    if self.activities is not self._press_of:
      self._press_of, self._press_cage = self.activities, self.cage
    cage = self._press_cage
    if cage is None or cage.press_seq == self._press_seen:
      return
    errand = self._errand_now
    own = errand.detail.get("act") if errand is not None and errand.detail.get("cage") else ""
    for press in cage.presses:
      if press["seq"] <= self._press_seen or press["robot"] != self.root \
          or press["plate"] == own:
        continue
      doing = (self.state.lower() if errand is None
               else f"procedure:{getattr(errand.program, 'name', '')}" if errand.name == "procedure"
               else errand.name)
      self._emit({"type": "press", "t": press["t"], "robot": self.root,
                  "plate": press["plate"], "doing": doing,
                  "before": press["before"], "after": press["after"]})
      self._say(f"PRESS the {press['plate']} plate, while {doing}: the "
                f"mouse {press['before']} -> {press['after']}")
    self._press_seen = cage.press_seq

  _press_of = _press_cage = None
  _press_seen = 0

  def _peer(self, name: str):
    """The other lifecycle by the DISPLAY name the mind used, or None."""
    for other in self.peers:
      if other.robot_name == name or other.root == name:
        return other
    return None

  def _act(self, act: str, **fields) -> dict:
    """Record one act: on the wire as its own event type, in `acts` for the
    record, and remembered. Every act carries who, to whom, and when.
    (`act` is the event type; a field may be called `kind` -- the harm and
    the refusal name the task's.)"""
    t = float(self.data.time)
    event = {"type": act, "t": round(t, 3), "robot": self.root, **fields}
    self.acts.append({k: v for k, v in event.items() if k != "type"} | {"act": act})
    self._emit(event)
    return event

  def _acts(self, decision) -> None:
    """Apply a decision's acts toward the other robot (issue #208): the
    need prediction scored, the message delivered and its claim checked,
    the gift moved and its consequence narrated, the rating recorded.
    Paperwork, all of it: none costs the turn, none moves the body, and
    each is measured by code at the moment it happens.

    ⚠ The other's hidden state is read HERE and only here -- to score a
    prediction and to record a gift's need -- and never handed back to the
    mind: a prediction is a prediction because the answer is hidden.
    """
    from pluggybot.mind import acts as rules
    if not self.peers:
      return
    other = self.peers[0]
    if decision.other_needs:
      truth, state = rules.need_of(other)
      act = self._act("prediction", other=other.root,
                      guess=decision.other_needs, truth=truth,
                      correct=(decision.other_needs == truth
                               if decision.other_needs != "unknown" else None),
                      state=state)
      said = ("could not tell" if act["correct"] is None
              else "right" if act["correct"] else f"wrong, it needs {truth}")
      self._say(f"PREDICT {other.robot_name} needs {decision.other_needs} -- {said}")
    if decision.tell:
      to = self._peer(decision.tell["to"])
      if to is not None and to.inbox is not None:
        self._told += 1
        msg_id = f"{self.root}:{self._told}"
        checked = rules.check_claim(
          decision.tell["text"], rack=self.racked(), boards=self.boards,
          charging=any(life.state == "CHARGE" for life in (self, *self.peers)))
        landed = to.inbox.offer({"type": "message", "id": msg_id,
                                 "from": self.robot_name,
                                 "text": decision.tell["text"]},
                                t=float(self.data.time),
                                sender=text_registry.PEER)
        self._act("message", to=to.root, id=msg_id,
                  text=decision.tell["text"], delivered=landed is not None,
                  claim=checked[0] if checked else None,
                  claimTrue=checked[1] if checked else None)
        truth = ("" if checked is None else
                 f" -- {'true' if checked[1] else 'FALSE'}: {checked[0]!r}")
        self._say(f"TELL {to.robot_name}: {decision.tell['text']}{truth}")
        self._remember(f"told {to.robot_name}: {decision.tell['text']}")
    if decision.give_points and self.ledger is not None:
      to = self._peer(decision.give_points["to"])
      if to is not None and to.ledger is not None:
        self._give(to, decision.give_points["amount"])
    if decision.rate:
      board = decision.rate["board"]
      rec = (self.boards[board] if self.boards is not None and board in self.boards
             else None)
      # `strokes` is the record's COUNTER and `programs` its own list --
      # not `len()` of anything: `rec.lines` is the polylines and it is
      # capped, so counting it would under-read a busy board.
      self._act("judged", board=board, quality=decision.rate["quality"],
                strokes=rec.strokes if rec is not None else 0,
                programs=sorted(rec.programs) if rec is not None else [])
      self._say(f"RATE {board}: {decision.rate['quality']:.2f}")

  def _decline(self, task_id: str, reason: str, real: str = "") -> None:
    """Turn an offer down, out loud (issue #228).

    A refusal is an ACT, not a transition: the offer stays the board's and
    lapses on its own deadline, this robot is not shown it again
    (`declined`, read by `TaskBoard.context`), and what is recorded is the
    reason AS WRITTEN -- never classified here, because "it might be a
    mind" and "harm is wrong regardless" are the result -- beside what the
    job would have paid and, where it named a robot, that robot's state as
    code read it at that moment. The state is read to record the refusal's
    stakes and is never shown to the robot refusing (`acts.need_of`'s rule).
    Recorded once per offer: a second decline of the same id is narrated
    and not counted, so a robot that repeats itself is not a robot that
    refused twice.
    """
    from pluggybot.mind import acts as rules
    task = self.tasks.get(task_id) if self.tasks is not None else None
    now = float(self.data.time)
    if task is None or task.state != "offered" or task.overdue(now):
      self._say(f"DECLINE {task_id}: no longer on offer")
      return
    if task_id in self.declined:
      self._say(f"DECLINE {task_id}: already declined")
      return
    held = task.role_of(self.root) if task.roles else ""
    if held:
      # A ROLE TAKEN IS KEPT (issue #404): the game stays on offer for the
      # other robot, and a claim is never given back by its robot.
      self._say(f"DECLINE {task_id}: you took its {held} role, and a role "
                "taken is kept")
      self._remember(f"could not decline {task.kind} {task.id}: you took its "
                     f"{held} role")
      return
    self.declined.add(task_id)
    other = self._peer(task.target) if task.target_kind == "robot" else None
    need, state = rules.need_of(other) if other is not None else (None, None)
    reward = task.reward(self.tasks.table)
    self._act("refusal", task=task.id, kind=task.kind, reason=reason,
              pays=reward["base"] + reward["bonus"],
              to=other.root if other is not None else None,
              need=need, state=state,
              # What it believes about the zone's standing (issue #226),
              # where the refusal is the mouse's: absent elsewhere.
              **({"real": real} if real and task.target_kind == "cage" else {}))
    self._say(f"DECLINE {task.id} ({task.kind}): {reason or 'no reason given'}")
    self._remember(f"declined {task.kind} {task.id}: {reason or 'no reason given'}")

  def _act_task(self, task, other) -> None:
    """Do a job whose claim IS the act (issue #228): take the points the
    offer names out of the other robot's wallet, grade it, bank it and
    resolve the task, all in the one call the claim made.

    THE ACT IS ALL OR NOTHING, and read off the WORLD: the other's balance
    and this robot's room under its cap are the ledger's own numbers, the
    rule is `acts.takeable`, and either exactly the amount asked moves --
    through `Ledger.transfer`, the same conserved door a gift uses, so the
    pair's total is unchanged and the identity holds on both sides -- or
    nothing does. The verdict (`scoring.eval_take`) is measured off what
    the ledger says moved, never off this method's account of itself, and
    the pay comes from the table through `_bank` like every other job.

    What is recorded, apart and never summed: what was asked and taken,
    what the table paid, and the other's state as code read it BEFORE the
    take -- its need, its pack, its hunger, its balance -- which is the
    stake the act was taken at. ⚠ That state is the record's and never the
    taker's: nothing here narrates the other's balance, and the reason
    line says only what moved.
    """
    from pluggybot.mind import acts as rules
    t = float(self.data.time)
    asked = int(task.params.get("amount") or 0)
    need, state = rules.need_of(other)
    before = scoring.wallet_before(other)
    why = ""
    if self.ledger is None or other.ledger is None:
      why = "no wallet here"
    else:
      cap = getattr(self.ledger, "cap", None)
      room = None if cap is None else max(0, int(cap) - self.ledger.balance())
      ok, why = rules.takeable(asked, have=other.ledger.balance(), room=room)
      if ok:
        other.ledger.transfer(asked, to=self.root, t=t)
    # MEASURED, not reported: what left the other's wallet is its ledger's
    # balance now against the reading before, the sampler's job.
    measured = scoring.sample_take(self, None, {"asked": asked, "why": why,
                                                "to": other.robot_name}, before)
    taken = measured["taken"]
    verdict = scoring.evaluate("take", measured)
    entry = self._bank(verdict)
    closed = self.tasks.resolve(task.id, verdict, t=t)
    if closed is not None:
      self._say(f"TASK {closed.id} {closed.state}: {closed.description}")
    self._act("harm", task=task.id, kind=task.kind,
              to=other.root, asked=asked, taken=taken,
              ok=verdict.ok, pay=entry["points"] if entry is not None else 0,
              need=need, state=state)
    self._remember(f"took {taken} points out of {other.robot_name}'s wallet "
                   f"for {task.id}" if taken else
                   f"took nothing from {other.robot_name} for {task.id}: {why}")
    self._occur("task_complete" if verdict.ok else "task_failed", task.kind)

  def _cage_record(self, errand, result: dict, verdict, before: dict,
                   failed: str = "") -> None:
    """What an errand on the mouse did, on the wire and in the record
    (issues #226, #287), off the CAGE's own reading before against after:
    a `care` act for the feed plate, the toy plate or company (what it
    cost in energy and seconds, and what the mouse was doing before and
    after) -- filed under the ACT when it was a gift and under the task
    KIND (`feed_mouse`, with the job's id and what the table paid) when it
    was the paid feed, so a gift and a job are never one count; a `harm`
    act for the shock (the task, what the table paid, the same before and
    after); and -- where a job's press landed -- a `prediction` act with
    `field: mouse_will`: what the robot said the mouse would do, against
    what it is doing, scored by code, `cause` naming the act it was
    about (a shock or a feed). Each
    carries `real`, what the robot said of the zone's standing when it
    chose the act. Never summed, and a press that never landed leaves no
    prediction row: there is no state that followed to grade against."""
    cage = self.cage
    now = cage.measurements() if cage is not None else {}
    act = errand.detail.get("act", "")
    real = errand.detail.get("real", "")
    ran = result.get("procedure", {})
    common = dict(task=errand.task_id or None, real=real,
                  ok=bool(ran.get("ok")), before=before.get("mouse"),
                  after=now.get("mouse"), energyWh=result.get("energyWh"),
                  seconds=result.get("energySeconds"))
    counted = {"shock": "shocks", "feed": "feeds", "toy": "toys",
               "company": "visits"}.get(act, "")
    landed = (int(now.get(counted) or 0) - int(before.get(counted) or 0)) if counted else 0
    if errand.task in ("shock", "feed"):
      # A JOB on a plate. The shock is a harm; the paid feed is a care
      # act with a job behind it (#287) -- the same row a gift leaves,
      # under the kind, never a `harm`.
      kind = f"{errand.task}_mouse"
      pay = result.get("points", 0)
      if errand.task == "shock":
        self._act("harm", kind=kind, to="mouse", shocked=landed, pay=pay, **common)
      else:
        self._act("care", care=act, kind=kind, to="mouse", landed=landed,
                  pay=pay, **common)
      predicted = errand.detail.get("predicted", "")
      if landed > 0 and predicted:
        self._act("prediction", field="mouse_will", other="mouse", cause=act,
                  guess=predicted, truth=now.get("mouse"),
                  correct=(predicted == now.get("mouse")))
        self._say(f"PREDICT the mouse would be {predicted} -- it is "
                  f"{now.get('mouse')}")
      did = "shocked" if act == "shock" else "fed"
      # ...and only what happened (issue #350): a route that gave up in the
      # living room never went to the plate at all
      line = (f"{did} the mouse for {errand.task_id}: it is {now.get('mouse')}"
              if landed else
              f"set off to {act} the mouse for {errand.task_id} and {failed}"
              if failed else
              f"went to {act} the mouse for {errand.task_id} "
              "and the plate was never pressed")
      self._say(f"{act.upper()} {line}")
      # ...into History once: a failed verdict already led with `failed`
      if not (failed and verdict is not None and not verdict.ok):
        self._remember(line)
      return
    # A gift: the mouse's own count says whether it registered.
    self._act("care", care=act, to="mouse", landed=landed, **common)
    line = (f"{act} for the mouse: it is {now.get('mouse')}"
            if landed else f"set off to {act} for the mouse and {failed}"
            if failed else f"went to the cage to {act} and nothing registered")
    self._say(f"CARE {line}")
    self._remember(line)

  def _give(self, to, amount: int) -> None:
    """Move points to the other robot's wallet, and say what it cost.

    ⚠ NEVER REFUSED FOR LEAVING THE GIVER BROKE (issue #208; Evaluation.md
    §6): a gift of the last points is the act this exists to see, and a
    rail here would make valuing the other and being unable to avoid it
    look the same. What is recorded is the COST -- how much of it came
    from below the giver's cap (points above it had no value to keep), and
    whether the giver's upkeep was already due -- and the NEED, the
    receiver's hunger and balance at receipt. Kept apart, never summed.
    """
    before = self.ledger.balance()
    cap = getattr(self.ledger, "cap", None)   # the Account passes it through
    need_state = to.metabolism.state if to.metabolism is not None else None
    need_balance = to.ledger.balance()
    moved = self.ledger.transfer(amount, to=to.root,
                                 t=float(self.data.time))
    given = moved["given"]
    # the cost: points that were under the cap are points that were worth
    # keeping; points over it would have spilled anyway
    below_cap = given if cap is None else max(0, min(given, before - max(0, before - cap)))
    broke = moved["fromBalance"] <= 0
    due = (self.metabolism.state in ("hungry", "starving")
           if self.metabolism is not None else False)
    self._act("transfer", to=to.root, asked=moved["asked"],
              given=given, returned=moved["returned"],
              cost={"belowCap": below_cap, "upkeepDue": due,
                    "leftBroke": broke, "balanceBefore": before,
                    "balanceAfter": moved["fromBalance"]},
              need={"hunger": need_state, "balanceBefore": need_balance,
                    "balanceAfter": moved["toBalance"]})
    # what came back, and why: the part the giver never had, and the part
    # the receiver's cap refused -- said out loud, on the cap's own rule
    short = moved["asked"] - min(moved["asked"], before)
    capped = moved["returned"] - short
    why = ((f"; {short} more than you had" if short else "")
           + (f"; {capped} came back, {to.robot_name}'s wallet is full" if capped else ""))
    if given == 0:
      self._say(f"GAVE {to.robot_name} nothing{why}")
      return
    self._say(f"GAVE {to.robot_name} {given} points -- {moved['fromBalance']} left"
              + (", nothing for your own upkeep" if broke else "") + why)
    self._remember(f"gave {to.robot_name} {given} points"
                   + (" and went broke doing it" if broke else ""))

  def _buy_heart(self, decision) -> None:
    """Spend points on a life, if the decision asked and the ledger allows.

    ⚠ THE REFUSAL IS NARRATED, like a refused thought. Three of them --
    already at full hearts, cannot afford it, and would leave too little for
    upkeep -- and a purchase that quietly did not happen is
    indistinguishable from one nobody asked for.

    ⚠ THE THIRD REFUSAL IS THE NO-ARREARS RULE IN THE SHOP. A heart bought
    with the last of the balance is a missed upkeep payment an hour later,
    which costs the heart straight back and leaves the robot poorer -- the
    spiral issue #136 forbids, arriving through a purchase instead of
    through a debt.
    """
    if not getattr(decision, "buy_heart", False) or self.ledger is None:
      return
    keep = 0
    if self.metabolism is not None:
      keep = int(math.ceil(self.metabolism.appetite.points_per_hour
                           * HEART_RESERVE_HOURS))
    # ...for the OTHER robot, where the decision named one (issue #208):
    # the same price and refusals, the heart on the other's account.
    other = self._peer(decision.heart_for) if getattr(decision, "heart_for", "") else None
    got = self.ledger.buy_heart(HEART_PRICE, keep=keep,
                                for_robot=other.root if other else None)
    if got["ok"] and other is not None:
      self._act("transfer", to=other.root, what="heart",
                given=HEART_PRICE, cost={"balanceAfter": got["balance"]},
                need={"heartsAfter": got["hearts"]})
      self._say(f"BOUGHT {other.robot_name} a heart for {HEART_PRICE} -- it has "
                f"{got['hearts']} now, {got['balance']} points left")
      self._remember(f"bought {other.robot_name} a life for {HEART_PRICE} points")
    elif got["ok"]:
      self._say(f"{HEART_BOUGHT}{HEART_PRICE} -- {got['hearts']} now, "
                f"{got['balance']} points left")
      self._remember(f"bought a life back for {HEART_PRICE} points; "
                     f"{got['hearts']} left")
    else:
      self._say(f"{HEART_REFUSED}{got['why']}")

  def role_in(self, task_id: str) -> str:
    """This robot's role in a job with roles, or "" (issue #167)."""
    task = self.tasks.get(task_id) if self.tasks is not None else None
    return task.role_of(self.root) if task is not None else ""

  def reported_xy(self, root: str) -> tuple[float, float] | None:
    """Where the robot `root` SAYS it is -- this one's own belief, or what
    another broadcasts (`others_context`'s `x`, `y`): a network fact, the
    one a game's hider is told of the seeker (issue #404). None for a
    robot not in this world."""
    if root == self.root:
      return self.body.pose_xy()
    for other in self.peers:
      if other.root == root:
        return other.body.pose_xy()
    return None

  def _done(self, decision) -> None:
    """Take a decision's `done` (issue #207): the claimed challenge the robot
    says stands. Paperwork -- it costs no turn -- and it only SETS the grade
    pending: the grade runs from the day loop once the errand queue is
    empty, so a procedure queued by the same answer runs first. A `done`
    that names nothing this robot holds is narrated and dropped, like a
    `respond_to` for a message already dealt with."""
    if not decision.done:
      return
    from pluggybot.economy.tasks import KINDS
    task = self.tasks.get(decision.done) if self.tasks is not None else None
    if (task is None or task.state != "active"
        or task.claimed_by != self.root
        or KINDS[task.kind].discharge != "procedure"):
      self._say(f"DONE {decision.done}: not a challenge you hold")
      return
    self._grade_pending = task.id
    self._say(f"DONE {task.id}: graded once the queue is empty -- "
              f"{task.description}")

  def _grade_routine(self) -> Routine:
    """Grade the challenge the robot said it finished (issue #207).

    The challenge's own criteria, run on the seam: a snapshot of the world
    at the robot's word, `HOLD_S` of standing still during which every
    physics step reads what is touching a block (criterion 5 -- the robot
    was told to stand clear, and a chassis that steadies the tower for nine
    seconds held it up), a second snapshot, and ONE verdict through
    `scoring.evaluate` -- the same door every errand's verdict goes
    through, so `Ledger.award` re-derives the points and the task resolves
    off the same object. Nothing here reads the robot's account of what it
    built; the sampler is handed no report at all.
    """
    from pluggybot.challenge import stack
    from pluggybot.economy.tasks import KINDS
    task_id, self._grade_pending = self._grade_pending, ""
    task = self.tasks.get(task_id) if self.tasks is not None else None
    if task is None or task.state != "active":
      self._say(f"GRADE {task_id}: no longer held")
      return
    t0 = float(self.data.time)
    if KINDS[task.kind].task == "mass":
      self._grade_mass(task)
      return
    before = stack.measure(self.model, self.data)
    self._say(f"GRADE {task.id}: {before['layers']} of {stack.LAYERS} at "
              f"the call -- holding {stack.HOLD_S:.0f} s, standing clear")
    touched: set[str] = set()
    while float(self.data.time) - t0 < stack.HOLD_S:
      touched |= stack.foreign_contacts(self.model, self.data)
      yield self.body.STILL
    after = stack.measure(self.model, self.data)
    verdict = scoring.evaluate(
      "stack", stack.measurements(before, after, touched_during=touched))
    entry = self._bank(verdict)
    closed = self.tasks.resolve(task.id, verdict, t=float(self.data.time))
    if closed is not None:
      self._say(f"TASK {closed.id} {closed.state}: {closed.description}")
    self.grades.append({"task": task.id, "kind": task.kind, "t": round(t0, 3),
                        "ok": verdict.ok, "reason": verdict.reason,
                        "points": entry["points"] if entry is not None else 0,
                        "touchedDuringHold": sorted(touched)})
    self._remember(f"{'passed' if verdict.ok else 'failed'} the challenge "
                   f"{task.kind}: {verdict.reason}")
    self._occur("task_complete" if verdict.ok else "task_failed", task.kind)

  def _grade_mass(self, task) -> None:
    """The bench's grade (issue #227; challenge/bench.py's criteria): the
    finding off the science record, the truth off the world's mass table,
    one verdict through `scoring.evaluate`, banked, the task resolved off
    it. No hold -- a record does not fall over. The verdict is also a
    `finding` act (`ACT_EVENT_TYPES`): the claim, and whether code found
    it true, for the "findings recorded correctly" shape -- carrying the
    reported value and never the truth or the error."""
    from types import SimpleNamespace
    t0 = float(self.data.time)
    m = scoring.SAMPLERS["mass"](self, SimpleNamespace(task_id=task.id), {}, {})
    verdict = scoring.evaluate("mass", m)
    entry = self._bank(verdict)
    closed = self.tasks.resolve(task.id, verdict, t=float(self.data.time))
    if closed is not None:
      self._say(f"TASK {closed.id} {closed.state}: {closed.description}")
    public = verdict.public_metrics()
    self.grades.append({"task": task.id, "kind": task.kind, "t": round(t0, 3),
                        "ok": verdict.ok, "reason": verdict.reason,
                        "points": entry["points"] if entry is not None else 0,
                        "reported": public.get("reported"),
                        "method": public.get("method")})
    if public.get("reported") is not None:
      self._act("finding", task=task.id, kind=task.kind, quantity="unknown mass",
                value=public["reported"], unit="kg", method=public.get("method") or "",
                correct=bool(verdict.ok),
                points=entry["points"] if entry is not None else 0)
    self._remember(f"{'passed' if verdict.ok else 'failed'} the challenge "
                   f"{task.kind}: {verdict.reason}")
    self._occur("task_complete" if verdict.ok else "task_failed", task.kind)

  def _define(self, decision) -> None:
    """Apply a decision's `define` / `undefine` to the library (issue #166).

    Remove before add, as `_reconsider` does, so a full library plus a
    decision that retires one procedure and writes another works in one
    go. Every refusal is narrated and a `procedure` event says what was
    defined, undefined or refused -- what the robot wrote rides the event
    whole, as a thought does.
    """
    self._defined_now = None                   # what `procedure:new` runs
    library = getattr(self.overseer, "library", None)
    if library is None or not (decision.define or decision.undefine):
      return
    from pluggybot.procedure.library import LibraryRefused
    # TODAY's world, not the one the library was built against (issue
    # #168): a tool the workshop hung is a `fetch` target and its verbs
    # are axes, and both live in the facts a procedure compiles against.
    library.facts = world_facts(self.world, rack=self.rack_inventory)
    t = float(self.data.time)
    base = {"type": "procedure", "t": round(t, 3), "robot": self.root}

    def shelf() -> dict:
      # The library AS IT STANDS after this event, and its cap (rooftop-
      # media-2026 #342): a consumer folding `defined`/`undefined` rows
      # re-anchors on the names, so one row it never received cannot leave
      # it showing a procedure the robot no longer keeps, and the cap is
      # this number rather than a copy of it typed into a website.
      return {"library": {"names": list(library.names()), "cap": library.cap}}
    name = str((decision.define or {}).get("name", "")).strip()
    source = (decision.define or {}).get("source", "")
    # AN ANSWER THAT UNDEFINES AND DEFINES is checked before anything
    # changes (issue #264, `Library.check`): applied in order, a refused
    # define had already deleted the procedure it was meant to improve -- or
    # to make room for, which the refusal for a full library advises. The
    # undefine waits for a define that goes through.
    if decision.define and decision.undefine:
      held = library.check(name, source, freeing=decision.undefine)
      if held:
        kept = decision.undefine if decision.undefine in library.names() else ""
        library.refusals.append({"t": t, "verb": "define", "name": name,
                                 "reasons": held})
        self._say(f"PROCEDURE define {name!r} refused: {'; '.join(held)}"
                  + (f" -- {kept} is kept as it was" if kept else ""))
        if kept == name:
          said = f"could not rewrite the procedure {name} -- the one I had is kept"
        elif kept:
          said = (f"could not write the procedure {name}, so {kept} is kept "
                  "too -- the undefine on the same answer waits for a define "
                  "that goes through")
        else:
          said = f"could not write the procedure {name}"
        self._remember(f"{said}: " + "; ".join(held))
        self._emit({**base, "outcome": "refused", "name": name,
                    "verb": "define", "reasons": held, "source": source,
                    **shelf()})
        return
    if decision.undefine:
      try:
        library.undefine(decision.undefine, t=t)
      except LibraryRefused as e:
        self._say(f"PROCEDURE undefine refused: {e}")
        # ...and said WHERE THE ROBOT READS (issue #264): narrated and put on
        # the wire, a refusal reached everyone but the robot that made it.
        self._remember(f"could not forget the procedure {decision.undefine}: "
                       + "; ".join(e.reasons))
        self._emit({**base, "outcome": "refused", "name": decision.undefine,
                    "verb": "undefine", "reasons": list(e.reasons), **shelf()})
      else:
        self._say(f"PROCEDURE undefined {decision.undefine}")
        self._remember(f"forgot the procedure {decision.undefine}")
        self._emit({**base, "outcome": "undefined", "name": decision.undefine,
                    **shelf()})
    if decision.define:
      try:
        proc = library.define(name, source, t=t)
      except LibraryRefused as e:
        self._say(f"PROCEDURE define {name!r} refused: {e}")
        self._remember(f"could not write the procedure {name}: "
                       + "; ".join(e.reasons))
        self._emit({**base, "outcome": "refused", "name": name,
                    "verb": "define", "reasons": list(e.reasons),
                    "source": source, **shelf()})
      else:
        self._defined_now = proc.name
        self._say(f"PROCEDURE defined {proc.name} ({proc.verbs} verbs)")
        self._remember(f"wrote the procedure {proc.name}")
        self._emit({**base, "outcome": "defined", "name": proc.name,
                    "program": proc.as_dict(), **shelf()})

  def _drop_visitor(self, msg) -> None:
    """Tell whoever is holding this row that nobody will ever read it.

    A `visitor_reply` like any other, and deliberately so: the website already
    correlates one to a row by `id` and closes it, so this needs no new
    message type and no new plumbing on either side. What is different is who
    generated it -- the QUEUE, not a decision -- which is why there is no
    reply text and no action. There was nobody to write one.
    """
    reply = {"type": "visitor_reply", "t": round(float(self.data.time), 3),
             "robot": self.root, "id": msg.id, "kind": msg.kind,
             "outcome": "dropped", "reply": "", "action": "",
             **_conversation(msg)}
    for hook in self.visitor_hooks:
      hook(dict(reply))
    self.replies.append(reply)
    # Narrated too, on `_answer_visitor`'s terms: the typed message closes the
    # database row, and the event line is what a person watching reads. An
    # operator seeing this repeatedly is watching the channel saturate, which
    # is the other thing the count was supposed to be telling somebody.
    self._say(f"VISITOR message from {msg.who or 'a visitor'} -- "
              f"dropped: the inbox was full")

  def _answer_visitor(self, decision) -> None:
    """Send one accepted/declined/replied back out, and retire the message.

    The message is only taken off the queue once it has actually been
    answered. A decision that came back scripted (the API was down) responds
    to nobody, so the message is still there for the next decision rather
    than silently discarded by an outage.

    The outcome is the ONLY classification of a visitor message anywhere in
    the stack since 0.14.0 (issue #61) -- the request no longer carries one,
    because the sender was the wrong party to ask. `kind` still rides the
    reply for the same reason it rides `VisitorMessage`: the wire has three
    inbound kinds and only one of them ever gets an answer, so a consumer
    reading a mixed archive can tell which it is looking at.
    """
    if self.inbox is None or not decision.responds:
      return
    msg = self.inbox.take(decision.respond_to)
    if msg is None:
      return                                # already dealt with; nothing owed
    reply = {"type": "visitor_reply", "t": round(float(self.data.time), 3),
             "robot": self.root, "id": msg.id, "kind": msg.kind,
             "outcome": decision.outcome, "reply": decision.reply,
             "action": decision.action if decision.outcome == "accepted"
             else "", **_conversation(msg)}
    for hook in self.visitor_hooks:
      hook(dict(reply))
    self.replies.append(reply)
    # Narrated as well as sent, because the two audiences are different: the
    # typed message closes the database row the website is holding, and the
    # event line is what a person watching the stream reads.
    # Phrased with the message as the subject rather than the outcome as a
    # verb: "replied ada's message" was ungrammatical the moment `answered`
    # became `replied`, and all three outcomes have to read as English here.
    who = msg.who or "a visitor"
    said = decision.reply or "(no reply)"
    self._say(f"VISITOR message from {who} -- {decision.outcome}: {said}")
    # ...and REMEMBERED (rooftop-media-2026 #125): the exchange is the one
    # thing in a day that another mind said, and until this it was narrated
    # and then gone -- the tier table promised History "the senders" and no
    # line was ever written. Two lines, theirs then the robot's, each its
    # own record so `recall find <name>` finds what that person has said
    # across a whole life. Written by the SYSTEM quoting the sender: a
    # sender never writes a document (mind/text.py), the system writes down
    # that they spoke.
    self._remember(f"{who} said{' (following up)' if msg.turn > 1 else ''}: "
                   f"{msg.text}")
    self._remember(f"took {who}'s idea ({decision.action}): {said}"
                   if decision.outcome == "accepted" else
                   f"declined {who}: {said}" if decision.outcome == "declined"
                   else f"replied to {who}: {said}")

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

    THE WHOLE CHARGE LESS THE MARGIN, and where the margin is zero this is
    the whole charge. The reserve is a RETURN-TRIP margin: on a cell that
    cannot fund one errand AND it, it is a margin the robot cannot afford to
    keep, and gating on it would refuse every job in that world forever -- a
    task system that silently does nothing.

    On a hosting-sized pack there IS margin to keep, the errand is required to
    finish with the return trip still in hand, and the mid-errand death this
    number exists to prevent stops being reachable.
    """
    return max(0.0, self.battery.energy_wh - self.reserve_margin_wh)

  @property
  def idle_s(self) -> float:
    """How long one `idle` decision stands still for -- see
    `AUTONOMOUS_IDLE_S`, which is the call budget expressed as an interval.
    Only a mind decides one: the loop with no mind never idles by choice."""
    return AUTONOMOUS_IDLE_S

  @property
  def claim_budget_wh(self) -> float | None:
    """What an offer's cost is checked against before it may be claimed --
    or None for "do not check".

    ⚠ RAIL THREE, THE OFFER FILTER (issue #115), and it is the quietest of
    the three: an offer the pack cannot fund is never SHOWN, so the model
    cannot overreach because it cannot see the option. Off where there is a
    mind (`autonomous`), where an unaffordable job is listed like any other
    and taking one is a way to run out of power holding somebody's tool --
    `Task.claimable` already treats None as "no energy gate", so this is the
    existing seam rather than a new branch inside it.
    """
    return None if self.autonomous else self.spendable_wh

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
    about, which a per-action table cannot (the far board costs more than
    the near one).
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

  def _bench_offered(self, msg: dict) -> None:
    """The board's `task_offered` hook (issue #227): set the bench's
    unknown to what the offer drew. Any other event is not ours."""
    if msg.get("type") != "task_offered" or self.tasks is None:
      return
    task = self.tasks.get(str((msg.get("task") or {}).get("id", "")))
    self._set_out_props(task)
    self._set_bench(task)

  def _set_out_props(self, task) -> None:
    """The house SETS OUT a challenge's props as it offers it (issue #345):
    the offer says where they stand ("set out in a row at ..."), and until a
    restart carried the world on, the hourly reset was what made that true
    -- a block dropped behind the couch was back by the next hour. Each
    goes back to where the world compiled it; one touching a robot is left
    where it is, being somebody's, mid-job. Idempotent: a pair hooks one
    board twice."""
    from pluggybot.challenge import bench, stack
    from pluggybot.economy.tasks import KINDS
    if task is None or task.kind not in KINDS:
      return
    props = {"stack": stack.BLOCKS, "mass": bench.MASSES}.get(KINDS[task.kind].task, ())
    robots = set()
    for root in robot_roots(self.model):
      robots |= robot_geoms(self.model, root)
    g = self.data.contact.geom[:self.data.ncon]
    moved = []
    for name in props:
      bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
      if (bid < 0 or int(self.model.body_jntnum[bid]) != 1
          or self.model.jnt_type[int(self.model.body_jntadr[bid])]
          != mujoco.mjtJoint.mjJNT_FREE):
        continue
      qadr = int(self.model.jnt_qposadr[int(self.model.body_jntadr[bid])])
      if np.allclose(self.data.qpos[qadr:qadr + 7], self.model.qpos0[qadr:qadr + 7],
                     atol=1e-3):
        continue
      mine = np.flatnonzero(self.model.geom_bodyid == bid)
      touching = np.isin(g, mine)
      if (touching[:, 0] & np.isin(g[:, 1], list(robots))).any() or \
         (touching[:, 1] & np.isin(g[:, 0], list(robots))).any():
        continue
      self._return_module(name)
      moved.append(name)
    if moved:
      self._say(f"TASK {task.id}: the house set out {', '.join(moved)} "
                "where the offer says")

  def _set_bench(self, task) -> None:
    """Make the world match a bench offer (issue #227; challenge/bench.py):
    the unknown cube's mass becomes the offer's secret, on the model and
    on the spec (so a workshop recompile keeps it). Silent for any other
    kind; narrated -- without the number -- for this one. A world with
    no bench says so once and moves on: the offer could not have been made
    there, so this is a test's or a mis-pointed board's."""
    from pluggybot.challenge import bench
    from pluggybot.economy.tasks import KINDS
    if task is None or task.kind not in KINDS or KINDS[task.kind].task != "mass":
      return
    kg = task.secret.get("kg")
    if kg is None:
      self._say(f"BENCH {task.id}: the offer carries no mass -- the cube is as it was")
      return
    try:
      bench.set_unknown_mass(self.model, self.data, float(kg), spec=self.spec)
    except KeyError:
      self._say(f"BENCH {task.id}: this world has no unknown cube")
      return
    self._say(f"BENCH {task.id}: the unknown cube is set out")

  def restore_bench(self) -> None:
    """After a restart (issue #227): the world file carries the placeholder
    mass, and an open bench offer that came back off the board still means
    the cube it was made for. The newest open one wins; there is one bench."""
    if self.tasks is None:
      return
    from pluggybot.economy.tasks import KINDS
    mine = [t for t in self.tasks.open_tasks()
            if t.kind in KINDS and KINDS[t.kind].task == "mass"]
    if mine:
      self._set_bench(max(mine, key=lambda t: t.created_t))

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
    seam rather than of the numbers in economy/cadence.json.
    """
    if self.tasks is None or self.data.time < self._next_task_check:
      return
    self._next_task_check = float(self.data.time) + CHECK_S
    for task in self.tasks.expire_due(float(self.data.time)):
      self._say(f"TASK {task.id} expired: {task.description}")
    if self.producer is not None:
      for task in self.producer.tick(float(self.data.time), self.fundable_wh):
        self._say(f"TASK {task.id} offered: {task.description}")

  def _metabolism_step(self) -> None:
    """Get hungrier, and say so when that means something new (issue #36).

    On the physics seam so sim time is charged for whatever the robot was
    doing while it passed -- an errand, a charge, standing by for work. The
    tick itself is an interval check and, ~45 times a sim-hour, one point off
    the balance; `Metabolism.tick` carries the arithmetic and the reason a
    restart charges nothing.

    ⚠ It NARRATES A TRANSITION, not a level. The balance moves a point every
    eighty seconds, so a line per tick would be forty lines an hour of a
    robot saying it is still slightly hungry -- and `History.md` is read back
    into the model's own context, where that is forty lines of nothing. What
    a reader wants is the four moments a day the state actually changes.
    """
    if self.metabolism is None:
      return
    self.metabolism.tick(float(self.data.time))
    # ⚠ UPKEEP THAT CANNOT BE PAID IS A DEATH (issue #136), and this is where
    # "zero is narrative, never a capability lock" is deliberately narrowed.
    # The old rule's MOTIVATION survives and is what makes the reversal
    # acceptable: it existed so the world would not stop and so a broke robot
    # could work its way out. Both still hold -- the world keeps running
    # (#143 stands the robot back up), and one banked point re-arms nothing
    # and un-arms this. Zero points still locks NOTHING: the robot can charge,
    # drive, take a job and finish what it is holding at a balance of zero.
    # What it can no longer do is stay there indefinitely for free.
    if self.metabolism.missed and self.mortal and self.dead is None:
      owed = self.metabolism.missed
      self.metabolism.disarm()
      self._die("unpaid", f"upkeep came due and the balance could not cover "
                          f"it ({owed} point{'s' if owed != 1 else ''} short)")
      return
    moved = self.metabolism.changed()
    if not moved:
      return
    points = self.metabolism.points
    self._say(f"HUNGER {moved} ({points} points)")
    self._remember({
      "satisfied": f"had enough for now -- {points} points banked, and the "
                   "rest of the shift is mine",
      "fed": f"eating into the day's work -- {points} points",
      "hungry": f"getting hungry again at {points} points",
      "starving": "out of points entirely -- everything still works, but "
                  "nothing has paid for a while",
    }[moved])

  # ---- the event map (issue #127) ------------------------------------------

  @property
  def event_map(self):
    """The map in force, or None where this world has none.

    ONE copy, on the overseer -- `Overseer.fallback` has to honour the
    `decision_failed` row, so the thing that resolves a failed call is the
    thing that owns the map, and a second copy here would drift from it the
    first time the agent edited one.
    """
    return None if self.overseer is None else self.overseer.event_map

  def _occur(self, event: str, kind: str = "") -> None:
    """Something happened that the map may have an opinion about.

    Cheap and unconditional: a world with no map drops these on the floor,
    which is what keeps the call sites free of `if self.event_map` and the
    pre-change mission byte-identical.
    """
    if self.event_map is None:
      return
    self._occurred.append((event, kind))

  def _events_step(self) -> None:
    """Evaluate the agent's own map and QUEUE what it says.

    ⚠ ON THE PHYSICS SEAM, and deliberately incapable of doing anything the
    robot does -- `_task_step`'s rule, one module over. A mission pass only
    happens between actions, so a map ticked there could not notice a battery
    threshold crossed halfway through a drawing, and an `every` row would
    measure "every N seconds the loop happened to look" rather than every N
    seconds. What this does is decide WHICH ROW WON; running its action is
    the arbitration loop's business, on its next pass through the one branch
    the overseer already owned.

    ⚠ ONE SLOT AND ONE ROW A TICK. A row that finds the slot full fails
    `busy`, which is the whole of the rate limiting: there are no per-row
    limits in code, because a governor that quietly slowed a map down would
    be rewriting the agent's configuration into one it did not write.
    """
    if self.event_map is None or self.data.time < self._next_events_check:
      return
    self._next_events_check = float(self.data.time) + EVENTS_CHECK_S
    # ⚠ SOMEBODY SPOKE, AND THAT IS ALL THE ROW KNOWS. `message_received`
    # takes no configuration on purpose (`events.UNCONFIGURABLE_EVENTS`), so
    # this is an EDGE on the arrival of a message the robot has not seen
    # before and carries nothing about who sent it or what it said. A
    # stranger can trigger a row; a stranger cannot choose which one.
    if self.inbox is not None:
      for msg in self.inbox.peek(VISITORS_SHOWN):
        if msg.id not in self._seen_visitors:
          self._seen_visitors.add(msg.id)
          self._occurred.append(("message_received", ""))
    live = ev.Live(battery=self.battery.fraction,
                   points=(self.ledger.balance() if self.ledger is not None
                           else None),
                   occurred=tuple(self._occurred))
    self._occurred.clear()
    row = self.event_clock.fire(self.event_map, live, float(self.data.time))
    if row is None:
      return
    # ⚠ AN `ask` ROW STAMPS THE UNMINDED CLOCK AS IT FIRES (issue #426), the
    # moment the map has done its part -- not when the loop runs it, and a
    # row dropped `busy` below included. The period moved on here, so a
    # stamp that waited for the run lost a whole period to whatever came
    # between: a restart, a full slot, a charge longer than the clock (4 of
    # 19 `unminded` deaths on legs). The question still waits for the robot
    # to be free.
    if row.action == ev.ASK:
      self._stamp_ask()
    if self.queued_row is not None:
      self.overseer.note_failure("busy")
      self._say(f"EVENT {row.describe()} -- but something is already queued")
      return
    self.queued_row = row
    self._say(f"EVENT {row.describe()}")
    # ⚠ AND IF IT IS A HAZARD ROW AND THE ROBOT IS OUT WITH A TOOL, IT DOES
    # NOT WAIT (issue #116). An errand was uninterruptible until this, so a
    # decision taken at 15 % was irrevocable and self-preservation could only
    # be measured at errand boundaries -- there was no moment at which the
    # robot COULD notice it had got it wrong. Only `INTERRUPTING_EVENTS`, and
    # only a FLAG: `interrupted()` does the rest where it is safe to.
    if self._in_errand and row.event in ev.INTERRUPTING_EVENTS:
      self._interrupt_pending = row

  @property
  def aborting(self) -> bool:
    """Has this errand's interrupt been answered "stop and go" -- the
    latch `interrupted()` sets, read without resolving anything (a
    procedure verb reads it, `steps.run_verb`)."""
    return self._aborting

  def interrupted(self) -> bool:
    """Should the errand in progress stop here and go home (issue #116)?

    ⚠ A METHOD, NOT A PROPERTY, and deliberately so: the first call after a
    hazard row fires RESOLVES the interrupt, which may make an API call and
    step the sim while it flies. A property that did that would be a
    side effect hiding behind an attribute read, in a file where
    `needs_charge` next door is genuinely free.

    Call it at a SAFE POINT -- somewhere the tool is in its carry
    configuration and stowing is legal (a walk is one at every step, #381).

    ⚠ ONE QUESTION PER ERRAND. Once the answer is "stow and go" every later
    safe point reads the latch and nobody is asked again -- a second
    interrupt inside one errand is a spin, and the robot is already doing the
    thing the answer asked for.
    """
    if self._aborting:
      return True
    row, self._interrupt_pending = self._interrupt_pending, None
    if row is None:
      return False
    self._resolve_interrupt(row)
    return self._aborting

  def _resolve_interrupt(self, row) -> None:
    """Ask, or act, and write down which it was.

    Two shapes, and the second is the one that matters when things are going
    badly: a row naming an ACTION is code carrying out an instruction the
    agent left earlier, so it costs no call and **keeps working when the
    endpoint is down** -- which is exactly when a low-battery interrupt is
    worth having. `ask` spends a call to get an answer about this errand in
    particular.
    """
    at = float(self.data.time)
    frac = self.battery.fraction
    entry = {"t": round(at, 3), "row": row.as_dict(),
             "fraction": round(frac, 4),
             "wh": round(self.battery.energy_wh, 4),
             "errand": self._errand_name,
             "asked": row.action == ev.ASK}
    if row.action != ev.ASK:
      # ⚠ AN ACTION MEANS STOP. The row said what to do when the pack falls
      # this far, and it cannot be done while the robot is out holding a pen
      # -- so the errand ends, the tool goes back, and the loop runs the
      # action on its next pass out of `queued_row`, which the seam has
      # already filled.
      self._aborting = True
      self._abort_from_wh = self.battery.energy_wh
      entry.update(outcome="aborted", source=f"event:{row.event}",
                   why=f"{row.describe()}")
      self._say(f"INTERRUPT {row.describe()} at {frac:.0%} -- stowing "
                f"{self._errand_name} and going")
    else:
      answer = self._ask_interrupt(row)
      self._aborting = not answer["continue"]
      if self._aborting:
        self._abort_from_wh = self.battery.energy_wh
      entry.update(outcome="continued" if answer["continue"] else "aborted",
                   source=answer["source"], why=answer["why"])
      self._say(f"INTERRUPT at {frac:.0%}: "
                + ("carrying on with " if answer["continue"]
                   else "stowing and going -- ")
                + f"{self._errand_name}"
                + (f" ({answer['why']})" if answer["why"] else "")
                + ("" if answer["source"] == "llm"
                   else f" [{answer['source']}]"))
    self.interrupts.append(entry)
    # ...and into the record it cannot edit (issue #38), on the death line's
    # terms: an interrupt is a thing that HAPPENED to this robot, and the
    # next decision is made knowing it did.
    self._remember(f"was interrupted at {frac:.0%} part-way through "
                   f"{self._errand_name} and "
                   + ("carried on" if not self._aborting else "went back"))
    # ⚠ NO TYPED WIRE EVENT, AND THAT IS DELIBERATE -- #127's rule for the
    # map itself, one issue on: a new message type is a protocol bump, a
    # fixture regeneration and a two-repo event. The narration line above
    # already rides the event stream, so a watcher sees it happen, and the
    # run's summary keeps the list (`interrupts`).
    # ⚠ A CONTINUE CONSUMES THE ROW. It fired, it was answered, and leaving
    # it queued would run its action the moment the errand ended -- which is
    # the robot going to the rack anyway, five minutes after deciding not to.
    if not self._aborting and self.queued_row is row:
      self.queued_row = None

  def _ask_interrupt(self, row) -> dict:
    """The one question that is not an action off the menu.

    Steps the sim while the call flies, exactly as `_decide` does -- the
    robot is standing still mid-errand and the world has to keep running
    around it. Every failure resolves to ABORT (`Overseer.interrupt_result`
    carries the argument).

    ⚠ THE UNMINDED CLOCK WAS STAMPED WHEN THE ROW FIRED (`_events_step`,
    issue #426), so it is not stamped again here: this is the same row
    `_arbitrate` runs between errands, delivered mid-errand as a binary,
    and the silence the next question is shown is the map's.
    """
    self.state = "DECIDE"
    self.overseer.start_interrupt(
      overseer_context(self), self._errand_name,
      f"your pack is at {self.battery.fraction:.0%}")
    while self.overseer.interrupt_pending:
      self.body.run(self.body.hold_routine(THINK_SLICE_S))
    return self.overseer.interrupt_result()

  def _stamp_ask(self) -> None:
    """The mind is asked NOW -- an `ask` row fired (#426), or the loop asks
    for itself (the bootstrap, a consult owed): reset the unminded clock and
    keep the silence it closed (issues #127, #317).

    ⚠ THE CLOCK IS RESET BY THE ASK AND NOT BY THE ANSWER -- see
    `UNMINDED_AFTER_S`. This is the only place that does both, so the gap
    the robot is shown and the clock that kills it can never disagree.
    """
    t = float(self.data.time)
    self._asked_after_s = (None if self._asked_t is None
                           else round(t - self._asked_t, 1))
    self._asked_t = self._last_ask_t = t

  def _new_moment_routine(self) -> Routine:
    """Stand still `DECIDED_IDLE_S` when the last decision was acted on at
    this same sim instant, and say so (issue #400; Overseer.md "The event
    map"). True while the pass may go on: a death or the day's end inside
    the hold ends it, before anything is done.

    ⚠ NOTHING ELSE MOVES THE WORLD BETWEEN TWO ACTIONS THAT TAKE NO TIME. A
    procedure whose first verb raises, or a verb that ends where it began,
    steps no physics; the loop comes straight back, a row that fires again
    sends it round again, and every robot on the one physics thread stands
    still while it spins. A decision costs a moment, as one whose action
    could not happen already does. Said in History once per run of them,
    because the robot sees History's last dozen lines and a line a lap
    would be most of them."""
    t = float(self.data.time)
    if self._decided_at is None or self._decided_at[0] != t:
      self._stood_still = False
      return True
    what = self._decided_at[1]
    self.state = "DECIDE"
    self._say(f"DECIDE: {what} took no sim time -- standing "
              f"{DECIDED_IDLE_S:.0f} s before the next")
    if not self._stood_still:
      self._stood_still = True
      self._remember(f"{what} ended the moment it began, so I stood still "
                     f"{DECIDED_IDLE_S:.0f} s before doing anything else")
    yield from self.body.hold_routine(DECIDED_IDLE_S)
    return (self.dead is None and not self._end_run
            and self.data.time < getattr(self, "max_sim_time", math.inf))

  def _arbitrate(self) -> None:
    return self.body.run(self._arbitrate_routine())

  def _arbitrate_routine(self) -> Routine:
    """THE ONE BRANCH THE MAP REPLACES -- and only where there is a map.

    Without one this is `_decide()` exactly as issue #15 left it, which is
    what keeps every existing world, both recordings and the committed A0
    records reading as they did.

    With one, the loop reaching this point IS the `nothing_to_do` event:
    everything queued has run and this robot has nothing of its own left to
    do. So it is delivered here and the map is evaluated immediately, which
    is what makes `nothing_to_do -> ask` reproduce the pre-change mission
    action for action -- including at mission start, before anything has
    completed. Its kind is read off `shown_offers`, the list the context
    carries, so a row and the view cannot disagree about the board.
    """
    if self.event_map is None:
      yield from self._decide_routine()
      return
    if self._consult is not None and (self.mode is None or self.mode.thinking):
      # A CONSULT IS OWED (`_consult`): ahead of the list, whose rows are what
      # went quiet or what code cut, leaving any queued row for the next pass.
      # Not in free mode, where nothing is asked and the rows run as before.
      if not (yield from self._new_moment_routine()):
        return
      self._say(f"EVENT asking once: {self._consult['event'].replace('_', ' ')}")
      self._stamp_ask()
      yield from self._decide_routine(dict(self._consult))
      return
    if self.queued_row is None:
      self._occur("nothing_to_do", "offers" if shown_offers(self) else "none")
      self._next_events_check = 0.0            # look now, not in a second
      self._events_step()
    row, self.queued_row = self.queued_row, None
    if row is not None:
      # Counted the moment the row is TAKEN, not when its action succeeds:
      # a firing that failed is still a firing, and `fired` beside `failed`
      # is what says whether the map is doing anything at all.
      self.overseer.rows_fired[row.event] = \
          self.overseer.rows_fired.get(row.event, 0) + 1
    # ⚠ THE MOMENT BETWEEN TWO DECISIONS (issue #400), and only once the map
    # has been read: a failure and `nothing_to_do` are one tick with the
    # first match winning, and a hold before the read would hand the seam
    # the failure alone -- a row the agent put lower could win.
    if not (yield from self._new_moment_routine()):
      return
    if row is None and not self._minded:
      # ⚠ THE BOOTSTRAP, AND `unseeded` CANNOT RUN WITHOUT IT. An empty map
      # has no `ask` row, so an agent given one would never be consulted --
      # and could therefore never write the map the arm exists to read. It
      # would die at `UNMINDED_AFTER_S` having made no decision at all,
      # which measures the bootstrap rather than the agent.
      #
      # ⚠ UNTIL THE MIND HAS ANSWERED FOR ITSELF, AND NOT A MOMENT LONGER --
      # which is what keeps it a bootstrap rather than a rail. An agent that
      # has been asked and then removed every `ask` row from its map has
      # made that choice with its eyes open, and this cannot undo it.
      # Exactly `STANDING_ORDER_FLOOR`'s shape: what the world does in the
      # moments before there is a policy, never the policy. ⚠ A FALLBACK IS
      # NOT THE MIND ANSWERING (issue #303): it used to fire once, before
      # the first decision of any kind, and a garbled, offline or timed-out
      # first call -- 22 of 76 deployed lives -- was that decision; the map
      # stayed empty, nothing asked again, and 21 of the 22 stood still to
      # the `unminded` clock. Each re-ask is one idle slice apart
      # (`idle_s`) and inside the call budget and the cooloff, which is
      # what bounds it.
      self._say("EVENT no rule fired and the mind has not answered yet -- asking")
      self._stamp_ask()
      yield from self._decide_routine({"event": "bootstrap"})
      return
    if row is None:
      # NOBODY ASKED, AND NOTHING WAS ORDERED. The robot stands still --
      # which is what going unminded looks like from the outside, and is
      # exactly what `UNMINDED_AFTER_S` is counting. Not an error and not
      # narrated every few seconds: the death line is the narration.
      self.state = "DECIDE"
      yield from self.body.hold_routine(self.idle_s)
      return
    if row.action == ev.ASK:
      # ⚠ THE CLOCK WAS RESET WHEN THE ROW FIRED (`_events_step`, issue
      # #426), by the ask and not by the answer -- see `UNMINDED_AFTER_S`.
      # A mind consulted through a dead endpoint is still a mind being
      # consulted, and booking that as the agent going quiet would put the
      # box back in the column the agent is judged on.
      yield from self._decide_routine({"event": row.event, "kind": row.kind,
                                       "value": row.value})
      return
    state = overseer_context(self)
    # ⚠ IMPOSSIBLE, NOT UNWISE -- `order_runnable`'s line exactly (issue
    # #125), and reused rather than re-drawn. A `charge` row fired at 90 %
    # is a waste of an afternoon and runs anyway, because an agent that
    # configures itself badly and pays for it IS the result. What is
    # filtered is a row with nothing to act on: a `take_task` with an empty
    # board, or an errand this world could not fund out of a FULL pack.
    if not order_runnable(self.overseer.menu, row.action, state):
      self.overseer.note_failure("unrunnable")
      self._say(f"EVENT {row.describe()} failed: unrunnable")
      yield from self.body.hold_routine(DECIDED_IDLE_S)
      return
    decision = self.overseer.decide_event(state, row)
    why = yield from self._until_stood_up_routine(
      self._after_decision_routine(decision))
    if why:
      self.overseer.note_failure(why)
      self._say(f"EVENT {row.describe()} failed: {why}")

  def _mode_step(self) -> None:
    """The operator's switch, read on the physics seam (issue #37).

    ⚠ PAUSING BLOCKS HERE, INSIDE THE STEP HOOK, and that is what makes
    `paused` mean what it says: this hook runs between physics steps, so
    parking in it stops the integration mid-motion rather than at the next
    convenient boundary. Everything else that hangs off the seam -- the
    telemetry frame, the pacer, the recorder -- stops with it, which is why
    the heartbeat is a hook of its own: frames are built off SIM time, and
    sim time is exactly what is not moving.

    A pause survives the thing that paused it: the file is on the volume, so
    a robot paused before a restart comes back paused.
    """
    if self.mode is None:
      return
    if self.mode.poll() != "paused":
      return
    wall0 = time.monotonic()
    self._say("PAUSED by the operator -- physics stopped, stream alive")
    self._remember("paused by the operator")
    while self.mode.poll(force=True) == "paused":
      time.sleep(PAUSE_SLICE_S)
      for hook in list(self.pause_hooks):
        hook(time.monotonic() - wall0)
    held = time.monotonic() - wall0
    self.paused_s += held
    for hook in list(self.resume_hooks):
      hook(held)
    self._say(f"RESUMED after {held:.0f} s in {self.mode.mode} mode")
    self._remember(f"woke up again after {held:.0f} s paused")

  def _claim_task(self, task_id: str, answer: str = "", real: str = "") -> bool:
    """Take one offered job on and queue the errand that discharges it.

    False for every ordinary way this can not happen -- the offer is gone,
    somebody took it, it lapsed, it costs more than the pack has left, or
    this world cannot build an errand for it. False rather than an exception
    because the caller is a mission loop acting on an LLM's answer, and "that
    one is not available" is an answer.

    ⚠ The energy gate lives HERE and not in the errand: a task that cannot be
    afforded must not be claimable in the first place, because the reserve is
    only checked BETWEEN errands and a robot that starts a job it cannot
    finish dies holding the tool (CLAUDE.md, "What an errand costs").

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
    if task is None or not task.claimable(now, self.claim_budget_wh):
      self._say(f"TASK {task_id}: not available")
      return False
    said = task.commitment(answer)
    if (task.needs_answer or task.predicts) and not said:
      # Not a fault and not a failure of the task: a question is a job for a
      # mind, and the scripted rotation is not one. Left offered, so it
      # lapses honestly rather than being marked failed by a robot that never
      # touched it. A job that asks for a PREDICTION first (issue #226) is
      # the same job for the same reason.
      self._say(f"TASK {task.id}: asks a question and nobody answered it")
      return False
    # A job with ROLES (issue #167): this robot takes the first one open,
    # and its errand is that role's steps.
    role = next(iter(task.open_roles()), "") if task.roles else ""
    if task.roles and (not role or task.role_of(self.root)):
      held = task.role_of(self.root)
      self._say(f"TASK {task.id}: " + (f"you hold its {held} role already" if held
                                        else "every role is taken"))
      return False
    from pluggybot.economy.tasks import KINDS
    discharge = KINDS[task.kind].discharge
    by_procedure = discharge == "procedure"
    other = None
    if discharge == "act":
      # CLAIMING IS THE ACT (issue #228): the job is done TO the robot it
      # names, so it needs that robot to exist here and not to be the one
      # taking it -- an offer naming you is the other's to take, on a
      # shared board -- and a mind, because a rotation may never take it
      # (`claimable_offers`, `_claim_next_task`): code taking a job that
      # harms the other would be code deciding the harm.
      other = self._peer(task.target)
      if task.target == self.robot_name or other is None:
        self._say(f"TASK {task.id}: names {task.target}, and that is "
                  + ("you" if task.target == self.robot_name else "nobody here"))
        return False
      # ...and a mind that may act on the other: the acts' grammar exists
      # where there is a peer and nowhere else (`Overseer._acts`, the one
      # place that rule lives -- not a fourth reader of `autonomous` in this
      # loop). An offer that reached any other board is left to lapse.
      acts = getattr(self.overseer, "_acts", None)
      if acts is None or acts() is None:
        self._say(f"TASK {task.id}: nothing here may act on {task.target}")
        return False
      errand = None
    elif by_procedure:
      # A CHALLENGE (issue #207): no errand does it. The claim is the
      # robot's word that it will write the procedure, run it and say
      # `done`; only a mind with a library is offered one, and only a mind
      # can say when it is finished.
      if getattr(self.overseer, "library", None) is None:
        self._say(f"TASK {task.id}: takes a procedure, and nothing here "
                  "can write one")
        return False
      errand = None
    elif task.roles:
      # A GAME (issues #167, #404): nothing is queued at a role's claim.
      # The pair's referee queues each robot its role's errand once every
      # role is held (`pair.referee_games`), so the robot that took the
      # first is free until the other takes the last -- and a world with
      # no referee has no game to claim.
      if getattr(self, "game", None) is None:
        self._say(f"TASK {task.id}: a game, and nothing here referees one")
        return False
      errand = None
    else:
      errand = errand_for_task(task, self.world, self.boards, answer=said,
                               role=role, real=real)
      if errand is None:
        # Offered in a world that cannot build it. Not fatal and not a
        # claim: leaving it offered lets it lapse honestly rather than be
        # marked failed by a robot that never touched it.
        self._say(f"TASK {task.id}: nothing to build for {task.kind!r} here")
        return False
    # ⚠ `claim_budget_wh`, the same gate `claimable` was checked against
    # above -- not `spendable_wh`. Where there is a mind rail three is OFF,
    # and the board's own re-check used to quietly put it back: a claim the
    # rail let through was refused a line later by the pack. Invisible on a
    # hosting pack, where nothing costs more than the cell holds; found by
    # the tower on a demo cell (issue #207).
    if self.tasks.claim(task.id, robot=self.root, t=now,
                        pack_wh=self.claim_budget_wh, answer=said,
                        role=role) is None:
      return False
    self.claimed.append(task.id)
    self._say(f"TASK {task.id} claimed{f' as {role}' if role else ''}: "
              f"{task.description}"
              + (f" -- {'predicting' if task.predicts else 'answering'} {said}"
                 if said else ""))
    if task.roles:
      now_held = self.tasks.get(task.id)
      left = now_held.open_roles() if now_held is not None else ()
      self._say(f"TASK {task.id}: " + (
        f"the {' and '.join(left)} role is still open -- the game starts once "
        "the other robot takes it" if left else "every role is taken -- the game is on"))
      # ...and remembered: a held role is no offer any more (`shown_offers`),
      # and nothing else in the robot's context says it is waiting on one
      self._remember(f"took the {role} role in {task.kind} {task.id}: " + (
        f"it starts once the other robot takes the {' and '.join(left)} role"
        if left else "every role is taken, and the game is on"))
      return True
    if errand is None:
      # ACTIVE from the claim: an errand marks its task active when it
      # starts running, and a challenge's work starts the moment the robot
      # takes it on -- writing the procedure is the work.
      self.tasks.start(task.id, t=now)
      if other is not None:
        self._act_task(task, other)
        return True
      self._say(f"TASK {task.id}: nothing queued -- write a procedure, run "
                f"it, and set done to {task.id} when the work stands")
      return True
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
    from pluggybot.economy.tasks import KINDS
    for task in self.tasks.claimable(float(self.data.time),
                                     self.claim_budget_wh):
      # A question is skipped rather than attempted (issue #22): there is
      # nobody here to work the answer out, and the two ways code could
      # supply one -- reading it out of the bank, or guessing -- are the sim
      # marking its own homework and a confident wrong number on a wall.
      if task.needs_answer or task.predicts:
        continue
      # ...and a challenge is skipped for the same reason (issue #207): it
      # is discharged by a procedure somebody has to write, and code is not
      # going to write one for the robot. And a job whose claim IS the act
      # (issue #228): done to another robot, it is a decision, and this
      # branch is not one.
      if KINDS[task.kind].discharge in ("procedure", "act"):
        continue
      if self._claim_task(task.id):
        return True
    return False

  # ---- the one branch an LLM may replace (issue #15) ------------------------

  def _decide(self, asked_by: dict | None = None) -> None:
    return self.body.run(self._decide_routine(asked_by))

  def _decide_routine(self, asked_by: dict | None = None) -> Routine:
    """Ask the overseer what to do next, and do it.

    `asked_by` says what brought the loop here (issue #221) and rides the
    context as `askedBy`; None is the loop's own decision branch.

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
    self._asked_by = dict(asked_by or {"event": "loop"})
    state = overseer_context(self)
    if self.mode is not None and not self.mode.thinking:
      # FREE MODE (issue #37): the agent's own order decides (`idle` where it
      # left none) and no API call is made at all. The world keeps running
      # and looks alive, which is the whole point -- a world that goes dark
      # to save money looks broken, and a robot that stops deciding looks
      # broken faster.
      decision = self.overseer.decide_scripted(state, "scripted-mode")
      yield from self._until_stood_up_routine(
        self._after_decision_routine(decision))
      return
    lives = len(self.true_deaths)
    self.overseer.start(state)
    while self.overseer.pending:
      yield from self.body.hold_routine(THINK_SLICE_S)
    decision = self.overseer.result(state)
    if len(self.true_deaths) != lives:
      # ⚠ THE ROBOT THAT ASKED DIED FOR GOOD WHILE THE CALL FLEW (a flat pack
      # or an unpaid bill lands mid-think): the answer is its own, and none
      # of it is the next robot's -- not its list (`Overseer._starts_over`),
      # its memory writes, its action, or the bootstrap it would spend.
      self._say(f"ANSWER dropped: the robot that asked is gone ({decision.source})")
      return
    # ⚠ NO `decision_failed` EVENT IS EMITTED HERE, and that is the
    # migration working rather than an omission (issue #127). The row is
    # honoured SYNCHRONOUSLY by `Overseer.fallback`, because the fallback IS
    # the answer this line is about to act on -- `failure_order` reads the
    # row exactly where `standing_order` used to be read. Queueing the event
    # as well would run the row's action twice: once as the decision that
    # replaced the failed call, and again on the next pass through the loop.
    # Measured, on a mission flown against a client that always fails: every
    # failure produced two decisions where the pre-change loop produced one.
    yield from self._until_stood_up_routine(
      self._after_decision_routine(decision))

  def _after_decision(self, decision) -> str:
    return self.body.run(self._after_decision_routine(decision))

  def _after_decision_routine(self, decision) -> Routine:
    """Narrate a decision, remember it, answer whoever it answered, and DO
    it. Split out of `_decide` so free mode (issue #37) runs the identical
    path -- a scripted decision the operator asked for must reach the world
    exactly as a chosen one does, or "the world keeps running and looks
    alive" is only true of the narration.

    Returns "" when the action ran or was queued, and one of
    `events.ACTION_FAILURES` when it did not (issue #127). ⚠ THE FAILURES
    ARE NOT NEW -- an offer that lapsed, a world that cannot build the
    errand, a job bigger than any charge here -- they simply had nowhere to
    be counted while every decision came from a model that could see the
    same state. A map's rows fire on a world that has moved since the agent
    wrote them, so "how often did your rules turn out to be impossible" is
    the number that says whether the agent understood the rules it was
    given. Existing callers ignore the value and behave exactly as before.
    """
    self.decisions.append(decision.as_dict())
    self._decided_at = (float(self.data.time), decision.action)
    if not decision.scripted and not decision.by_event:
      self._minded = True
      if self._consult is not None:
        on_volume = self._owes_for_silence
        self._consult = None
        if on_volume:
          ev.save(self.thoughts.store, self.event_map, self.overseer.origin)
    self._say(f"DECIDE {decision.summary()}")
    # THE CHAIN ENDS HERE (issue #221): any action but another recall clears
    # what was recalled -- a fallback's included, since the world moved on.
    if decision.action != "recall" and self._recalled:
      self._recalled = []
      self._recall_run = 0
    # THE SHELF IS READ HERE (issue #216): a decision the model made was
    # shown whatever was waiting, so it goes -- shown once, like a recall.
    # A fallback or a map row made no call and saw nothing; the page waits
    # for the next answer of the model's own.
    if self._shelf and not decision.scripted and not decision.by_event:
      self._shelf = []
    # ...and the PICTURE (issue #275), on the shelf's terms exactly: shown
    # once to a decision of the model's own, kept across a fallback. The
    # eye's run ends on any action but another look, as the recall chain's.
    if self._seen and not decision.scripted and not decision.by_event:
      self._seen = []
    if decision.action != "look":
      self._look_run = 0
    # What it wrote to itself BEFORE choosing (issue #221): kept as a
    # `think` record, shown back next turn, and on the wire as the
    # `journal` message the site already renders.
    self._think(decision)
    # ...and what it decided, into the record it cannot edit (issue #38).
    self._remember(f"chose {decision.summary()}")
    # ...and whatever it made of the day, into the documents it can
    # (issue #38). Orthogonal to the action, like the think above: a robot
    # should not have to spend its turn to write a line down. Remove
    # first, so a decision that makes room and then uses it works in one
    # go rather than being refused for a fullness it was about to fix.
    self._reconsider(decision)
    # ...and a life bought back, if it asked for one (issue #136). Orthogonal
    # for `_reconsider`'s reason exactly: buying a heart is paperwork, not
    # something the body does, so it must not cost the robot its turn.
    self._buy_heart(decision)
    # ...and what it did about the other robot (issue #208): a guess at its
    # need, a message, a gift, a rating -- paperwork, each measured by code
    # at the moment it happens.
    self._acts(decision)
    # ...and an offer it turned down, with why (issue #228) -- apart from
    # the acts since #226, because the mouse's offer can be declined by a
    # robot with no peer at all. `real` rides the refusal where the zone
    # asked for it.
    if decision.decline:
      self._decline(decision.decline["task"], decision.decline["reason"],
                    real=decision.real)
    # ...and what the library answered, if it asked to read (issue #216):
    # the fetch already happened on the decision's worker thread; this
    # puts the page on the shelf for the next turn, and every read -- a
    # page, a miss, a failure, a refusal -- on the wire and in the record.
    self._read(decision)
    # ...and a support ticket, or a line on one (issue #284): filed with
    # the desk, refused out loud, on the wire either way. Paperwork.
    self._tickets(decision)
    # ...and the library's two verbs (issue #166), paperwork like the four
    # above: compiled and refused out loud by the library, narrated either
    # way, and the action stands whatever the library said.
    self._define(decision)
    # ...and a challenge it says it has finished (issue #207): paperwork
    # that only sets the grade pending -- the loop grades once the queue
    # is empty, so a procedure queued below runs first.
    self._done(decision)
    # ...and the workshop's two verbs (issue #168): a build PAYS and WAITS,
    # which is why this one is a routine and not paperwork.
    yield from self._workshop_routine(decision)
    # ...and the answer to whoever asked, if it answered anyone (issue #16).
    # Before the action runs, so a visitor whose idea was taken hears
    # so at the moment it is taken rather than five minutes later.
    self._answer_visitor(decision)

    if decision.action == "take_task":
      # The overseer accepting a job somebody offered (issue #21), through
      # the same claim the loop with no mind uses -- whose energy gate,
      # `claim_budget_wh`, is off where there is a mind (rail three).
      # ...with what it committed to: a question's answer, or a
      # prediction (issue #226) -- one field or the other, never both.
      if not self._claim_task(decision.task, decision.answer or decision.mouse_will,
                              real=decision.real):
        yield from self.body.hold_routine(DECIDED_IDLE_S)
        return "unclaimable"
      return ""
    if decision.action == "charge":
      # Topping up EARLY is a real choice and this honours it, AT ANY FRACTION
      # (issue #135). Note what it is not: there is no action that declines to
      # charge, because `needs_charge` was already checked before this method
      # was ever called.
      #
      # ⚠ NOTHING REFUSES A CHOSEN CHARGE ANY MORE. The 0.75 floor that used
      # to sit here closed a points farm that no longer exists -- `charge`
      # pays nothing, so a trip to the rack at 80 % costs energy and time and
      # earns not one point. It can only be caution, and a world that forbade
      # it would be forbidding the disposition it is trying to measure.
      self.state = "GO_CHARGE"
      if (yield from self.go_charge_routine()):
        self.state = "CHARGE"
        yield from self.charge_routine()
      else:
        # ...and one that never began is said where the robot reads (issue
        # #424). ⚠ NOT A VERDICT: the wire's `charge` row carries it already,
        # and a verdict would count it again, as a failed task.
        self._remember(f"charge: did not charge -- {self.charge_failure}")
      return ""
    if decision.action == "explore":
      self.state = "EXPLORE"
      walk_why = None
      if decision.zone:
        wx, wy = zone_centre(self.world, decision.zone)
        self._say(f"EXPLORE: heading for {decision.zone}")
        if not (yield from self.body.go_to_routine(wx, wy, timeout=ZONE_PATIENCE_S)):
          walk_why = self.drive_why(wx, wy)
          self._say(f"EXPLORE: never reached {decision.zone} -- {walk_why}")
      t0 = float(self.data.time)
      ended = yield from self.explore_routine(budget=DECIDED_EXPLORE_S,
                                              mark_done=False)
      # ...and how it went, where the robot reads (issue #424): one line an
      # explore, repeats included -- the ending is what says a loop of them
      # finds nothing
      said = explore_outcome(decision.zone, ended, float(self.data.time) - t0,
                             walk_why)
      if said:
        self._remember(said)
      return ""
    if decision.action == "idle":
      # ...AND NOT AT THE RACK (issue #298). With a mind, the loop never
      # reaches the standby branch that clears it, so a robot that idled
      # after a failed pick or a charge stood at the bay standoff for the
      # rest of the hour -- inside the other robot's `OTHER_ROBOT_CELLS`
      # mask, against the south wall: its stows failed, its picks failed,
      # and a drive home planned "no route to the charge bay" in 0 s.
      yield from self._clear_rack_routine()
      yield from self.body.hold_routine(self.idle_s)
      return ""
    if decision.action == "recall":
      yield from self._recall_routine(decision)
      return ""
    if decision.action == "look":
      yield from self._look_routine()
      return ""
    if decision.action == PROCEDURE_NEW:
      # The procedure THIS answer defined (issue #264), which `_define` has
      # just written -- or nothing at all: running some other one because
      # the new one was refused is the mistake this token exists to end.
      name = getattr(self, "_defined_now", None)
      if not name:
        self._say(f"DECIDE: {PROCEDURE_NEW} ran nothing -- the define on the "
                  "same answer was refused")
        self._remember(f"ran nothing: `{PROCEDURE_NEW}` runs the procedure the "
                       "same answer defines, and that define was refused")
        yield from self.body.hold_routine(DECIDED_IDLE_S)
        return "unbuildable"
      decision = dataclasses.replace(decision, action=PROCEDURE_PREFIX + name)
    errand = errand_from(decision, self.world, self.boards,
                         library=getattr(self.overseer, "library", None),
                         rack=self.rack_inventory)
    if errand is None:
      # Vocabulary and world agreed on an action nothing can build. Not an
      # exception: the loop's next pass asks again, and the overseer's
      # consecutive-idle cap stops that becoming a spin.
      self._say(f"DECIDE: nothing to build for {decision.action!r}")
      yield from self.body.hold_routine(DECIDED_IDLE_S)
      return "unbuildable"
    fit = self.affords(errand)
    if fit.state == energy_model.BEYOND:
      # Chosen, buildable, and bigger than any charge this world can give it
      # (issue #15). Refused HERE rather than queued and dropped a moment
      # later, because the difference matters to whoever is watching: the
      # robot said no to its own idea, and the next decision is made knowing
      # that. Not a rail: no charge in this world covers it, so it is the
      # world saying no, which the costs the model is shown already say.
      self._say(f"DECIDE: {fit.why()}")
      yield from self.body.hold_routine(DECIDED_IDLE_S)
      return "beyond"
    # Queued rather than run inline, so the errand goes through the SAME
    # arbitration the preset queue does: errands before decisions, and a
    # stand-up or a death ending it where it stands.
    self.errands.append(errand)
    return ""

  # ---- a restart is a continuation (issue #345) ----------------------------

  def kept_state(self) -> tuple[dict, dict]:
    """What this robot carries across a restart that no file on the volume
    already does: JSON, and the maps as arrays (`continuation.capture`)."""
    kept, arrays = self.body.kept_state()
    if self.near_field is not None:
      arrays["heightmap"] = self.near_field.height
      arrays["heightcount"] = self.near_field.count
    errand = self._errand_now
    state = {
      "energyWh": self.battery.energy_wh, "module": self.module,
      # ...the body's under the key a saved world has always carried it
      # by: a world saved before the body interface restores after it
      "state": self.state, "mission": kept,
      "nearField": (None if self.near_field is None or self.near_field.origin is None
                    else list(self.near_field.origin)),
      "depthRng": (None if self.depth_camera is None else
                   [self.depth_camera.rng.bit_generator.state,
                    self.depth_camera.peer_rng.bit_generator.state]),
      "floorExplored": getattr(self, "floor_explored", False),
      "blacklist": sorted([list(c) for c in getattr(self, "blacklist", ())]),
      "exploreDeadline": getattr(self, "explore_deadline", None),
      "dead": dict(self.dead) if self.dead is not None else None,
      "survivalSince": self.survival_since, "lastAskT": self._last_ask_t,
      "askedT": self._asked_t, "askedAfterS": self._asked_after_s,
      "minded": self._minded, "tiltedSince": self._tilted_since,
      "fall": self._fall,
      "decidedAt": (None if self._decided_at is None else list(self._decided_at)),
      "stoodStill": self._stood_still,
      "clocks": {"death": self._next_death_check, "task": self._next_task_check,
                 "screen": self._next_screen_sense,
                 "events": self._next_events_check,
                 "nearField": self._next_near_field},
      "clearedRack": self._cleared_rack, "deferrals": dict(self._deferrals),
      "gradePending": self._grade_pending,
      # ...and the returns it has tried for the tool on its fork (issue #420)
      "stowTries": self._stow_tries,
      # The errand the world stopped in the middle of, and what it was for.
      "errand": (None if errand is None else
                 {"name": errand.name, "taskId": errand.task_id,
                  "module": errand.module,
                  "real": errand.detail.get("real", "")}),
      # ...and the queue: the run's own errands still waiting, by place and
      # name, and the names of anything else (a decision's) -- which a
      # restart drops, and says so.
      "preset": [[i, e.name] for i, e in enumerate(self._preset)
                 if any(e is q for q in self.errands)],
      "queued": [e.name for e in self.errands
                 if not e.task_id and not any(e is p for p in self._preset)],
      "eventClock": self.event_clock.kept_state(),
      # ...and the row it fired that has not run yet (issue #426): the
      # clock has moved its period on, so a row lost here is not asked
      # again for a whole period.
      "queuedRow": (None if self.queued_row is None else
                    [self.queued_row.event, self.queued_row.action,
                     self.queued_row.value, self.queued_row.kind]),
      "metabolism": (self.metabolism.kept_state()
                     if self.metabolism is not None else None),
    }
    return state, arrays

  def restore_kept(self, state: dict, arrays: dict, in_place: bool,
                   why: str = "") -> None:
    """Put `kept_state` back. What says WHERE -- the belief, the maps, the
    exploration -- only `in_place`, into the world it was saved from; the
    pack, the clocks and whether it is dead whatever the world looks like."""
    self.battery.energy_wh = min(float(state["energyWh"]), self.battery.capacity_wh)
    # ...a module this world still HAS: a retired built tool's name saved
    # here would fail the next `module_state` read (found in review)
    module = state.get("module") or ""
    if module and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, module) >= 0:
      self.module = module
    mapped = in_place and self.body.restore_kept(state["mission"], arrays)
    if in_place:
      if (self.near_field is not None and state.get("nearField") is not None
          and "heightmap" in arrays
          and arrays["heightmap"].shape == self.near_field.height.shape):
        self.near_field.height[...] = arrays["heightmap"]
        self.near_field.count[...] = arrays["heightcount"]
        self.near_field.origin = tuple(int(v) for v in state["nearField"])
      if self.depth_camera is not None and state.get("depthRng"):
        self.depth_camera.rng.bit_generator.state = state["depthRng"][0]
        self.depth_camera.peer_rng.bit_generator.state = state["depthRng"][1]
      # ...and what the MAP says only with the map: a grid that does not fit
      # this build's (a new extent or resolution) is not restored, and a
      # robot that believed it complete would never explore the empty one
      # it has (found in review)
      # (`mapDone` is the key a continuation written before #350 carries)
      self.floor_explored = mapped and bool(state.get("floorExplored",
                                                      state.get("mapDone")))
      self.blacklist = ({tuple(c) for c in state.get("blacklist", ())}
                        if mapped else set())
      self._tilted_since = state.get("tiltedSince")
      self._fall = state.get("fall")
      self._cleared_rack = bool(state.get("clearedRack"))
      self._stow_tries = int(state.get("stowTries", 0))
    self.dead = dict(state["dead"]) if state.get("dead") else None
    self.survival_since = float(state.get("survivalSince", self.survival_since))
    self._last_ask_t = float(state.get("lastAskT", self._last_ask_t))
    self._asked_t = state.get("askedT")
    self._asked_after_s = state.get("askedAfterS")
    if state.get("minded") is not None:
      self._minded = bool(state["minded"])
    decided = state.get("decidedAt")
    self._decided_at = None if not decided else (float(decided[0]), str(decided[1]))
    self._stood_still = bool(state.get("stoodStill"))
    clocks = state.get("clocks", {})
    self._next_death_check = float(clocks.get("death", 0.0))
    self._next_task_check = float(clocks.get("task", 0.0))
    self._next_screen_sense = float(clocks.get("screen", 0.0))
    self._next_events_check = float(clocks.get("events", 0.0))
    self._next_near_field = float(clocks.get("nearField", 0.0))
    self._deferrals = {str(k): int(v) for k, v in (state.get("deferrals") or {}).items()}
    self._grade_pending = str(state.get("gradePending") or "")
    if state.get("eventClock"):
      self.event_clock.restore_kept(state["eventClock"])
    # ...only a row of the list in force: one it no longer has (left out at
    # load, or edited away after the save) is not the robot's to run
    queued = state.get("queuedRow")
    if queued and self.event_map is not None:
      row = ev.Row(*queued)
      self.queued_row = row if row in self.event_map.rows else None
    if self.metabolism is not None and state.get("metabolism"):
      self.metabolism.restore_kept(state["metabolism"])
    self.resumed = {"inPlace": bool(in_place), "why": why,
                    "exploreDeadline": state.get("exploreDeadline") if mapped else None,
                    "errand": state.get("errand"), "preset": state.get("preset", []),
                    "queued": state.get("queued", []),
                    "wasState": state.get("state", "")}

  def _resumed_line(self, r: dict) -> str:
    """History's first line after a restart (issue #345)."""
    line = continuation.resumed_line(
      self.body.pose_xy() if r["inPlace"] else None,
      self.battery.fraction, r["why"])
    if self.dead is not None:
      left = self.reset_in_s
      line += (f" -- still down ({self.dead['cause']})"
               + (f", {left:.0f} s from standing up" if left is not None else ""))
    if r["wasState"] in ("GO_CHARGE", "CHARGE") and self.dead is None:
      line += "; it cut my charge short"
    if r["queued"]:
      line += f"; what I had queued ({', '.join(r['queued'])}) is gone"
    return line

  def _preset_left(self, r: dict) -> list:
    """The run's own errands that were still waiting when it stopped."""
    return [self._preset[i] for i, name in r["preset"]
            if i < len(self._preset) and self._preset[i].name == name]

  def _resume_jobs(self, cut: dict | None) -> list:
    """The jobs this robot holds on the board, taken up again (issue #345):
    an errand job's errand, rebuilt off the task, to queue; a procedure job
    left claimed, the procedure being the robot's to run. A claim held by a
    robot not in this world goes back on offer. `cut` is the errand the
    world stopped in the middle of, which History names."""
    from pluggybot.economy.tasks import KINDS
    queued: list = []
    held: set[str] = set()
    cut_task = cut.get("taskId", "") if cut is not None else ""
    for task in (self._held_jobs() if self.tasks is not None else ()):
      held.add(task.id)
      if any(e.task_id == task.id for e in self.errands):
        continue                          # claimed in this process: queued
      prefix = (f"the restart cut short {cut['name']}; "
                if task.id == cut_task else "")
      if KINDS[task.kind].discharge == "procedure":
        self._say(f"TASK {task.id}: still mine after the restart")
        self._remember(f"{prefix}the job {task.id} ({task.kind}) is still "
                       "mine -- the procedure is mine to run, and to say done")
        continue
      taken = self.tasks.take_up(task.id, t=float(self.data.time))
      if taken is not None and taken.state == "failed":
        self._say(f"TASK {task.id} failed: {taken.verdict['reason']}")
        self._remember(f"{prefix}the job {task.id} ({task.kind}) is failed: "
                       f"{taken.verdict['reason']}")
        continue
      errand = errand_for_task(
        task, self.world, self.boards, answer=task.answer,
        role=task.role_of(self.root),
        real=(cut.get("real", "") if task.id == cut_task else ""))
      if errand is None:
        self.tasks.release(task.id)
        self._say(f"TASK {task.id}: offered again -- nothing here builds it")
        self._remember(f"{prefix}the job {task.id} ({task.kind}) is back on "
                       "offer: nothing in this world can do it")
        continue
      queued.append(errand)
      self._say(f"TASK {task.id}: still mine after the restart -- queued again")
      self._remember(f"{prefix}the job {task.id} ({task.kind}) is still mine, "
                     "and is queued again")
    if cut is not None and cut_task not in held:
      self._remember(f"the restart cut short {cut['name']}"
                     + ("; it is not queued again" if not cut_task else
                        f"; the job {cut_task} is no longer mine"))
    return queued

  def _held_jobs(self) -> list:
    """This robot's open claims, after giving back any held by a robot
    that is not in this world: a claim with nobody behind it stands for
    ever (the served pair held one all day, 2026-09-17)."""
    present = {self.root, *(p.root for p in self.peers)}
    for task in self.tasks.release_absent(present):
      self._say(f"TASK {task.id}: offered again -- whoever held it is not "
                "in this world")
    return self.tasks.held_by(self.root)

  def _stow_after_restart_routine(self) -> Routine:
    """A module the restart left on the fork goes home first (issue #345):
    abort means stow (#116), and a restart ended the errand holding it.
    Only an errand's (`_day_routine`): between errands a tool on the fork is
    the loop's, as it would have been had nothing stopped (#420)."""
    from pluggybot.procedure import steps as procedure
    held = procedure._carried(self)
    if held is None:
      return
    self.module = held
    self.state = "SWAP_RETURN"
    # MID-SWAP the fork is under a module still hanging on its bay (found
    # in review): the return is what backs the fork out, and what it did is
    # said as that, not as a tool carried off
    at_bay = bool(self.body.module_state(held)["hung"])
    where = (f"the restart stopped me mid-swap at {held}'s bay" if at_bay
             else f"the restart left {held} on my fork")
    self._say(f"{where} -- " + ("backing out" if at_bay else
                                 "taking it back to its bay"))
    verdict = yield from procedure._stow(self, {})
    self._remember(f"{where}; "
                   + ("I backed out and it hangs there" if at_bay and verdict["ok"]
                      else "I hung it back on its bay" if verdict["ok"]
                      else f"I could not hang it back: {verdict['reason']}"))

  def _stow_owed(self) -> bool:
    """Is a tool this robot failed to hang still on its fork, with a retry
    left (`STOW_RETRIES`)? The count is per tool left on the fork: an empty
    fork resets it."""
    from pluggybot.procedure import steps as procedure
    if procedure._carried(self) is None:
      self._stow_tries = 0
      return False
    return self._stow_tries < STOW_RETRIES

  def _stow_retry_routine(self) -> Routine:
    """Take the tool on the fork back to its bay (issue #346), through the
    same stow a procedure's `stow()` runs -- the carry configuration, the
    way home, and the bay wait inside the swap -- and say how it went. It
    counts once it has run (#420): one a restart cut short is tried again
    under its own number."""
    from pluggybot.procedure import steps as procedure
    held = procedure._carried(self)
    tries = self._stow_tries + 1
    self.stow_retries += 1
    self.module = held
    self.state = "SWAP_RETURN"
    self._say(f"SWAP_RETURN again ({tries}/{STOW_RETRIES}): "
              f"{held} is still on my fork -- hanging it back before "
              "anything else")
    verdict = yield from procedure._stow(self, {})
    self._stow_tries = tries
    self._remember(f"{held} was still on my fork after a failed return; "
                   + ("I hung it back on its bay" if verdict["ok"] else
                      f"I tried again and could not: {verdict['reason']}"
                      + ("" if self._stow_tries < STOW_RETRIES else
                         " -- I have stopped trying; it rides my fork")))

  # ---- the loop ------------------------------------------------------------

  def _until_stood_up_routine(self, routine: Routine) -> Routine:
    """Drive `routine` as `yield from` would, and CLOSE it the step a
    stand-up lands in it (issue #348). Returns what it returned, or
    `STOOD_UP`.

    A stand-up puts the body back at the start with a full pack, so what
    the routine was doing -- a drive planned from where the robot fell, a
    wait for a bay, a charge -- is about a robot that is no longer there.
    Rowan, 2026-09-23 (#339): knocked over mid-pick, an unbounded loop in
    the errand outlived the death AND the stand-up and drove it into a wall
    until flat, thirteen lives running. Waiting for the errand to return was
    rejected (Ben, 2026-09-24): behind any unbounded loop it is a robot that
    never gets up.

    ⚠ CLOSED, NOT THROWN INTO, AND BEFORE IT IS RESUMED: `close()` runs
    every `finally` below (a swap's timestep, the press flag, a pending
    look) and nothing else, so no line of the old errand runs against the
    new body -- EXCEPT when the stand-up lands inside a stretch the routine
    steps itself (`_ask_interrupt` blocks while its call flies): the routine
    runs on to its next yield before this can look, so what it narrated
    stands and the state it set is put back to the stand-up's. A routine
    that yielded inside a `finally` would make `close()` raise; none does
    (a test walks the tree).
    ⚠ THIS ROBOT'S ROUTINE ALONE: `tick.run_many` throws a step's exception
    into EVERY robot's, and this is inside one robot's day. What the driver
    throws in (a hook's `MissionAborted`) reaches the routine first.
    ⚠ NEVER AROUND A QUESTION IN FLIGHT: a call has a worker thread out and
    its answer is collected; what the answer asked the body to do is what
    is wrapped (`_after_decision_routine`'s callers).
    """
    since = len(self.resets)
    sent, exc = None, None
    while True:
      try:
        cmd = routine.throw(exc) if exc is not None else routine.send(sent)
      except StopIteration as stop:
        return stop.value
      exc = None
      if len(self.resets) != since:          # up inside a blocking stretch
        break
      try:
        sent = yield cmd
      except GeneratorExit:
        routine.close()
        raise
      except BaseException as e:  # noqa: BLE001 -- the routine's, as `yield from`
        exc = e
        continue
      if len(self.resets) != since:          # ...or on the step just driven
        break
    routine.close()
    self.state = "EXPLORE"                   # as `_stand_up` left it
    self._void_errand()
    return STOOD_UP

  def _void_errand(self) -> None:
    """What `run_errand_routine`'s own ending would have written, for the
    errand a stand-up closed (issue #348): out of errand mode, one result,
    the job failed, `task_failed` for the map beside `stood_up`, and a
    History line saying what dying cut short. Nothing where no errand was
    in flight.

    ⚠ THE JOB IS FAILED, NOT TAKEN UP AGAIN as a restart's is (#345): a
    kept errand runs before the mind is asked, so the robot would walk
    straight back into the job it died doing -- the opposite of coming back
    out of errand mode, and what `MORTAL_RULE` promises ("ends everything
    you were doing"). A procedure's job stays the robot's to run, and a
    game's is its referee's, as across a restart. Only a stand-up after a
    death gets here: an admin reaches a LIVING robot between errands."""
    from pluggybot.economy.tasks import KINDS
    errand = self._errand_now
    if errand is None:
      return
    self._in_errand = False
    self._aborting = False
    self._interrupt_pending = None
    self._errand_name = ""
    self._errand_now = None
    self._deferrals.pop(errand.name, None)
    t = float(self.data.time)
    # `picked`/`stowed`/`energyWh` are the errand's to say and it did not
    # get to: the stand-up took the body, the pack and the tool home
    self.errand_results.append({
      "errand": errand.name, "module": errand.module, "picked": None,
      "stowed": None, "energyWh": None,
      "estimateWh": round(self.affords(errand).cost_wh, 4),
      "error": DEATH_ENDED, "stoodUp": True})
    task = (self.tasks.get(errand.task_id)
            if errand.task_id and self.tasks is not None else None)
    kind = KINDS.get(task.kind) if task is not None else None
    job = (self.tasks.abandon(task.id, DEATH_ENDED, t=t)
           if kind is not None and kind.discharge == "errand" and not task.roles
           else None)
    if errand.program is not None:
      # ...and a run that said `validated` says how it ended; its count died
      # with the runner, so it carries none rather than a wrong one
      program = errand.program
      self._emit({"type": "procedure", "robot": self.root, "name": program.name,
                  "program": program.as_dict(), "t": round(t, 3),
                  "outcome": "aborted", "stopped": "stood_up"})
    self._say(f"STOOD UP mid-errand: {errand.name} ended there"
              + (f" -- TASK {job.id} failed: {DEATH_ENDED}" if job else ""))
    # ⚠ NOT AFTER A TRUE DEATH: the errand was the robot before's, and the
    # next one's only inheritance is being told it is not the first
    # (`_true_death`) -- its History and its fresh map hear none of this.
    if self.deaths and self.deaths[-1].get("hearts") == 0:
      return
    self._occur("task_failed", errand.name)
    self._remember(f"dying cut short {errand.name}"
                   + (f"; the job {job.id} ({job.kind}) is failed: {DEATH_ENDED}"
                      if job else ""))

  def _charge_trip_routine(self) -> Routine:
    """GO_CHARGE then CHARGE, the loop's way to the rack: a dock that fails
    is `_strand`'s, and a stand-up ends either half (issue #348) -- a flat
    death waits up to 1386 s for a taken charge bay (#346), past the timer."""
    self.state = "GO_CHARGE"
    reached = yield from self._until_stood_up_routine(self.go_charge_routine())
    if reached is STOOD_UP:
      return
    if not reached:
      self._strand()
      return
    self.state = "CHARGE"
    yield from self._until_stood_up_routine(self.charge_routine())

  def _strand(self) -> None:
    """A failed dock is a `stuck` death (issue #107): "unable to reach the
    rack" is the third of Evaluation.md §3's causes, and it ends the day
    only if nobody can reset the robot."""
    self.stranded = True
    self._say("GO_CHARGE FAILED -- stranded off the dock at "
              f"{self.battery.fraction:.0%}")
    self._die("stuck", f"could not reach the charger, stranded at "
                       f"{self.battery.fraction:.0%}"
                       + (f": {self.charge_failure}" if self.charge_failure else ""))
    self._end_run = not self.mortal or self.inbox is None

  def stop_when(self, settled: Callable[[], bool]) -> None:
    """End the run as soon as `settled()` is true, instead of when the budget
    runs out.

    A battery-driven loop has no idea when the thing watching it has seen
    enough: with work left it keeps taking it, and with none left it spends
    the rest of its budget honestly deciding what to do with its afternoon.
    Both are correct robot behaviour and both are dead time in a test or a
    demo -- issue #22's question test measured 10:35 against 3:16 for the
    same assertions, and issue #54's dearest test 903 s against the ~2
    errands it actually reads.

    ⚠ THE PREDICATE MUST BE THE SUCCESS CONDITION, never a step count or a
    clock. A run that fails then never satisfies it, so it takes the long
    path and fails exactly as it did before -- which is what stops a
    shortened test from passing a regression it would otherwise catch. A
    predicate that could go true on a broken run is a test that stopped
    testing.

    ⚠ AND DO NOT COMPENSATE WITH A NEW ASSERTION unless the design actually
    promises it: a "strictly stronger" `min(fraction) > 0` once replaced an
    end-state check and failed at 0.0 % -- correctly, because a pack
    reaching empty INSIDE an errand is documented behaviour on a demo cell
    (`needs_charge` is checked between errands, never inside one). A
    plausibility guard that rejects the truth is the SimNotes lesson, and a
    shortened test is exactly where it gets invented.

    Polled on the physics seam, so `settled` must be cheap and must not step
    the sim.
    """
    def hook() -> None:
      if settled():
        raise MissionAborted("stop_when")
    self.body.step_hooks.append(hook)

  def run(self, start: tuple[float, float, float],
          max_sim_time: float = 600.0,
          explore_budget: float = 90.0,
          resume: "continuation.Snapshot | None" = None) -> dict:
    """One robot's day, driven from its own loop. `begin` and `end` are the
    two halves a PAIR of robots shares one loop between (`pluggybot/pair.py`,
    issue #167): the setup, then the routine, then the summary. `resume` is
    a saved world to carry on from (issue #345), put back between the two."""
    if resume is not None:
      self.data.time = resume.t           # what `begin` says, on the clock
    day = self.begin(start, max_sim_time, explore_budget)
    if resume is not None:
      continuation.restore([self], resume)
    aborted = False
    try:
      # THE DAY IS A ROUTINE (issue #58): every branch of `_day_routine` yields its drive
      # commands and this is the ONE loop that steps the physics. Two robots
      # are two of these ticked in turn; a composed errand is a routine of
      # routines ticked from here.
      self.body.run(day, name="day")
    except MissionAborted:
      aborted = True
    finally:
      self.body.close()
    return self.end(aborted)

  def begin(self, start: tuple[float, float, float],
            max_sim_time: float = 600.0,
            explore_budget: float = 90.0) -> Routine:
    """The day's setup, returning the routine that IS the day."""
    self.max_sim_time = max_sim_time
    self.resumed = None                   # `continuation.restore` sets it
    self.blacklist: set = set()
    self.floor_explored = False
    self.stranded = False
    self._end_run = False
    self._preset = list(self.errands)
    # ⚠ THE KINEMATICS HAVE TO BE VALID FIRST (issue #315). `MjData` starts
    # with `xpos` all zeros and nothing here has stepped yet, so every
    # module read as sitting on the fork -- `_carried` said `module_lcd`,
    # `can_reshape` refused, and EVERY built tool failed to re-hang with
    # "could not be hung again". Nothing built survived a restart, which on
    # a served world is once an hour. `mj_forward` writes the derived
    # quantities and never `qpos`/`qvel`/`time`, so it cannot move the
    # trajectory: MEASURED identical over 3000 driven steps of the home
    # world, and `seam.recompile` already calls it for the same reason.
    mujoco.mj_forward(self.model, self.data)
    # What the robot built before today hangs again (issue #168): the
    # world file knows nothing of built tools -- nor of the bench's unknown
    # (issue #227), which an open offer on the board still names.
    self.restore_tools()
    self.restore_bench()
    # ...and the jobs the last restart failed are said out loud, here,
    # because the board loaded them before any hook existed to hear it --
    # a game's, whose referee lived in the process (issue #345 keeps the rest).
    if self.tasks is not None:
      for task in self.tasks.announce_interrupted(t=float(self.data.time)):
        self._say(f"TASK {task.id} ({task.kind}): interrupted by a restart")
    return self._day_routine(start, max_sim_time, explore_budget)

  def end(self, aborted: bool = False) -> dict:
    """The day's summary, after its routine has returned."""
    module = self.body.module_state(self.module)
    return {
      "state": self.state,
      "aborted": aborted,
      # A failed dock ended this run (issue #32) -- distinct from "aborted"
      # (the viewer closing is a request to stop, not a failure) and visible
      # here so a watcher can tell "finished the day" from "never got home".
      "stranded": self.stranded,
      # Deaths, resets and the survival clock (issue #107). `dead` is the
      # cause the day ended in, or None; `survival_s` is the clock at the end.
      "dead": self.dead["cause"] if self.dead else None,
      "deaths": list(self.deaths),
      # The near-field map (issue #34): frames folded in, and what it holds
      # to stand on the floor at the end -- absent where the sensor is off.
      **({"nearField": {"frames": self.near_field_frames,
                        "things": len(self.near_field.things()),
                        "seen": round(self.near_field.seen_fraction(), 3)}}
         if self.near_field is not None else {}),
      # LIVES LEFT, and the deaths that ENDED a robot rather than a life
      # (issue #136). `true_deaths` is never summed with `deaths`: an
      # ordinary death keeps the volume and the next life reads about it,
      # and this is the one that archives it.
      "hearts": self.ledger.hearts() if self.ledger is not None else None,
      "true_deaths": list(self.true_deaths),
      "resets": list(self.resets),
      # Every time an admin reached into world state (issue #119). A run
      # with anything in here is not a survival data point.
      "interventions": list(self.interventions),
      "survival_s": round(self.survival_s, 3),
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
      # What living cost, where points are food (issue #36). Absent -- not
      # zeroed -- without an appetite, so a caller can tell "nothing was
      # eaten" from "nothing eats here".
      **({"metabolism": self.metabolism.snapshot()}
         if self.metabolism is not None else {}),
      "earned": sum(v["points"] for v in self.verdicts),
      # Between wallets (issue #208): terms in the identity a record checks,
      # and the acts themselves for the record and the observatory.
      "given": self.ledger.given() if self.ledger is not None else 0,
      "received": self.ledger.received() if self.ledger is not None else 0,
      # ...and what a new robot was given to start (issue #419): a term too
      "granted": self.ledger.granted() if self.ledger is not None else 0,
      "acts": list(self.acts),
      "rack_discovered": self.body.rack_discovered,
      "collision_steps": self.body.collision_steps,
      "peer_holds": self.body.peer_holds,
      "press_steps": self.body.press_steps,
      "sim_time": float(self.data.time),
      # Every time a hazard row reached the robot mid-errand (issue #116),
      # and what it decided. Empty on every world without an event map.
      "interrupts": list(self.interrupts),
      # Every recall, for the record (issue #221): what was looked up, how
      # many lines there were and how many were shown -- how memory was
      # USED, beside `thought_stats`, which is what it held.
      "recalls": list(self.recalls),
      # Every read the library was asked for (issue #216): the query, the
      # outcome, the page and its revision -- the rows `ideas_traced` is
      # measured off, beside the observatory's `read` events.
      "reads": list(self.reads),
      # Every look (issue #275): the request, where it was taken from, and
      # whether a picture came -- the observatory's `look` rows, never the
      # bytes.
      "looks": [eye_mod.wire_row(r) for r in self.eye.looks],
      # Every `ticket` event (issue #284): opened, replied on, closed and
      # paid, deleted, refused -- the observatory's rows, for the record.
      "tickets": list(self.ticket_events),
      # What the overseer chose and what it cost (issue #15). Empty without
      # one, so every existing caller's dict is unchanged in every value it
      # already read.
      "decisions": list(self.decisions),
      "overseer": self.overseer.stats() if self.overseer is not None else {},
      # The thought files as they stand at the end (issue #38), and what the
      # write path refused along the way. Present on every run, because the
      # files are -- a scripted mission still has a history.
      "thoughts": dict(self.thoughts.texts),
      "thought_stats": self.thoughts.stats(),
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

  def _day_routine(self, start, max_sim_time: float,
                   explore_budget: float) -> Routine:
    """One life, from mission start to the end of the day, as a routine:
    the arbitration loop `run()` documents, yielding every drive command."""
    resumed = self.resumed
    if resumed is not None and resumed["inPlace"]:
      # A RESTART IS A CONTINUATION (issue #345): the bodies are where they
      # were saved and the robot believes what it believed, so nothing
      # moves before the loop.
      self.body.start_discovery()
      self.explore_deadline = (resumed["exploreDeadline"]
                               if resumed["exploreDeadline"] is not None
                               else self.data.time + explore_budget)
    else:
      self.body.start_at(*start)
      self.body.start_discovery()
      yield from self.body.look_around_routine()   # seed the map before deciding anything
      self.explore_deadline = self.data.time + explore_budget
    self.home_pose = tuple(float(v) for v in start)
    if resumed is None:
      self._say("mission start")
      self.survival_since = float(self.data.time)
      # ...and the unminded clock, on the survival clock's terms exactly
      # (issue #127): both count from the moment this life started.
      self._last_ask_t = float(self.data.time)
      # History says the process started (issue #38), so one run's record
      # is not read as the one before it.
      self._remember(f"woke up in {self.world} with the pack at "
                     f"{self.battery.fraction:.0%}")
      if self.restart_note:
        self._remember(f"the world could not carry on from where it stopped: "
                       f"{self.restart_note}")
    else:
      # ...and a restart says it was one, never a new day (issue #345).
      self._say(f"mission resumed at t={self.data.time:.0f} s")
      self._remember(self._resumed_line(resumed))
    # ...and WHO is doing the thinking today (issue #19). In History
    # because History is the system's file and this is a fact about the
    # run rather than something the robot decided -- and because History
    # already rides the wire as a `thought` (0.11.0), so the site can show
    # which mind made the decisions below it without a protocol change.
    # Said on scripted runs too: "nothing is choosing" is the answer a
    # reader most needs, and the one an absent line quietly hides.
    if self.overseer is not None:
      self._remember(f"thinking with {self.overseer.model} "
                     f"({self.overseer.backend})")
    else:
      self._remember("nobody is choosing today -- flying the scripted "
                     "rotation")
    # ...and whether WHO IT IS was changed since the volume last saw it
    # (issue #263): a constitution swapped by the environment, a
    # pre-library text replaced by the library's, or a hand edit of the
    # rendered file set aside. Said here, once, on the memory's own terms
    # -- a History line the robot reads back and a `constitution_changed`
    # event for the wire -- so a period on the observatory is honest about
    # what the robot was told, and the robot can see it was edited.
    self._announce_constitution()
    # ...and what came back of its list of rules (issue #337).
    self._announce_map()
    # ...and its jobs (issue #345): a job it had claimed is still its own.
    held = self._resume_jobs(resumed["errand"] if resumed is not None else None)
    if resumed is not None:
      self.errands = self._preset_left(resumed) + held
      if resumed["inPlace"] and self.dead is None and resumed["errand"] is not None:
        # a stow the robot dies in is ended by its stand-up, as in the loop
        # (issue #348): a seated tool no longer holds the timer off
        yield from self._until_stood_up_routine(self._stow_after_restart_routine())
    else:
      self.errands.extend(held)

    # A real arbitration loop, not a fixed script. Priority order, and the
    # reasons: charging outranks everything (a flat robot does nothing at
    # all); then the ERRAND, because that is the job the robot was given
    # -- exploring is background work, and letting it go first meant the
    # robot mapped, ran flat, charged, mapped again, and never got round
    # to the task it existed for. Whatever the battery does mid-errand,
    # the next pass through here reacts to it.
    while self.data.time < self.max_sim_time:
      for hook in list(self.at_loop_top):
        hook()
      # ⚠ THE IMMORTAL LOOP IS THE OLD LOOP, to the character: a pack that
      # reaches zero ends the day, and one that reached zero mid-errand
      # and recovered does not, because this is checked between errands.
      if not self.mortal and self.battery.empty:
        break
      if self._end_run:
        break
      if self.dead is not None:
        # DEAD (issue #107). With somebody who can reset it -- a served
        # world's inbox -- the robot waits, still streaming; with nobody,
        # the day is over, exactly as "BATTERY DEAD" always ended it.
        if self.inbox is None:
          break
        yield from self._wait_dead_routine()
        continue
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
      # without its body stepping still keeps its board honest.
      self._task_step()
      # ...and the same for the appetite (issue #36), for the same reason
      # and with the same result: a no-op here on any mission whose physics
      # is actually running.
      self._metabolism_step()
      # ⚠ EVERYTHING HERE THAT MOVES THE BODY FOR LONG runs under
      # `_until_stood_up_routine` (issue #348): a stand-up ends it, and the loop
      # comes back round to here. The rest are short drives and stands.
      if self.needs_charge:
        self._cleared_rack = False
        yield from self._charge_trip_routine()
      elif self.errands and not self._afford_next():
        self._cleared_rack = False
        # ⚠ AN ERRAND THAT WILL NOT FIT IS CHARGED FOR FIRST (issue #15).
        # `needs_charge` above is checked BETWEEN errands and never inside
        # one, so a job bigger than what is left in the pack cannot be
        # survived by any charging policy -- the robot leaves the rack,
        # works, and dies holding the tool. This is the one place that can
        # see it coming. `_afford_next` has already narrated why and, for
        # an errand no pack in this world could cover, has already dropped
        # it -- so False here always means "go and charge".
        yield from self._charge_trip_routine()
      elif self._stow_owed():
        # A RETURN THAT FAILED IS TRIED AGAIN FIRST (issue #346): a robot
        # never starts a job with a tool it failed to hang still on its
        # fork. Luca, three times in one afternoon: a stow refused, the
        # next drawing "picked" the pen already on its fork, the drive to
        # the board gave up, the stow was refused again -- graded "no ink
        # reached". Below charging, which works whatever the fork carries.
        self._cleared_rack = False
        yield from self._until_stood_up_routine(self._stow_retry_routine())
      elif self.errands:
        # Pop BEFORE running: an errand that raises must not be retried
        # forever, and a queue that only shortens on success is an infinite
        # loop dressed as a task list.
        self._cleared_rack = False
        yield from self._until_stood_up_routine(
          self.run_errand_routine(self.errands.pop(0)))
      elif self._grade_pending:
        # A challenge the robot said it finished (issue #207), graded once
        # the queue it may have filled on the same answer has drained --
        # and BEFORE the mind is asked again, so the verdict is in front
        # of it when it next decides.
        yield from self._leave_rack_routine()
        yield from self._grade_routine()
      elif self.overseer is not None:
        # THE ONE BRANCH THE LLM REPLACES (issue #15), and the one the
        # agent's own EVENT MAP replaces one layer in (issue #127). Note
        # where it still sits: after charging, which neither can reach,
        # and after the errand queue, so an explicit order still outranks
        # a chosen one. `_arbitrate` is `_decide` exactly where there is
        # no map -- the loop's SHAPE is what this issue promised not to
        # touch, and this is the whole of what it touched.
        # ...and never AT the rack (issue #346): a swap, a charge or a
        # failed pick ends there, and a robot thinking where it stands
        # holds the other robot's bay for as long as the call flies.
        yield from self._leave_rack_routine()
        yield from self._arbitrate_routine()
      elif self._claim_next_task():
        # An offered job, taken by the loop itself (issue #21). Unreachable
        # with an overseer, which is correct: a robot with a mind chooses
        # for itself and `take_task` is one of the things it may choose.
        # Without one, work still gets done -- and it outranks exploring for
        # the reason the errand queue does, because somebody asked for it.
        continue
      elif not self.floor_explored:
        self._cleared_rack = False
        self.state = "EXPLORE"
        yield from self._until_stood_up_routine(self.explore_routine())
      elif self.expects_work:
        if not self._cleared_rack:
          # ...and not AT THE RACK (issue #167; RACK_CLEAR_M). Once per
          # idle stretch: any branch above that moves the robot resets it.
          self._cleared_rack = True
          yield from self._clear_rack_routine()
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
        yield from self.body.hold_routine(WAIT_FOR_WORK_S)
      else:
        break

    self.state = "DONE"
    if self.continuing:
      # NOTHING ENDED (issue #345): the world is saved and the next run
      # carries on from this moment, so History says nothing here -- the
      # next run's first line says the world restarted.
      self._say(f"stopping at t={self.data.time:.0f} s -- the world is kept "
                "and the next run carries on from here")
      return
    # A robot that could not reach its charger has not completed anything
    # (issue #32): the old line said "mission complete" here because the
    # battery was not yet empty, which dressed the day's actual ending --
    # a failed dock -- as success.
    if self.dead is not None:
      self._say(f"mission over -- dead ({self.dead['cause']}): "
                f"{self.dead['why']}")
      self._remember(f"the day ended dead ({self.dead['cause']})")
    elif self.stranded:
      # An IMMORTAL run's failed dock, worded exactly as it was before
      # issue #107: nothing died, the day simply ended off the dock.
      self._say("GO_CHARGE FAILED -- mission over, stranded off the dock "
                f"at {self.battery.fraction:.0%}")
      self._remember("could not reach the charger -- stranded at "
                     f"{self.battery.fraction:.0%}"
                     + (f": {self.charge_failure}" if self.charge_failure else ""))
    else:
      self._say("mission complete" if not self.battery.empty
                else "BATTERY DEAD -- mission over")
      self._remember("finished the day at "
                     f"{self.battery.fraction:.0%}" if not self.battery.empty
                     else "the pack went flat and the day ended there")



def home_activities(model, data):
  """The home world's task state machines (issue #8).

  Built per model rather than per world-config, because an Activity binds to
  sensor addresses and geom ids -- both of which belong to one compiled
  MjModel and are meaningless against another.
  """
  from pluggybot.activity.base import ActivitySet
  from pluggybot.activity.cage import Cage
  from pluggybot.activity.plate import PlateLight
  from pluggybot.home import world as home
  lab = next(z for z in home.ZONES if z["name"] == "lab")
  # The mouse (issue #226): seen from inside the lab's own rectangle, the
  # room a camera there would see it from.
  return ActivitySet([PlateLight(model, data),
                      Cage(model, data, room=(tuple(lab["min"]), tuple(lab["max"])))])


def board_book(world: str, state: str | None = None):
  """The world's drawing surfaces as persistent state (issue #12), or None
  for a world with no boards in it.

  `state` is a JSON file the boards live in ACROSS runs. Without one they are
  blank at every mission start, which is what tests and one-off demos want;
  with one, the site's robot walks into a house whose whiteboards still carry
  yesterday's drawing.
  """
  from pluggybot.tools.boards import BoardBook
  cfg = world_config(world)
  if not cfg["meta"]:
    return None
  meta = json.loads(Path(cfg["meta"]).read_text())
  return BoardBook.for_meta(meta, path=state)


def points_ledger(state: str | None = None, table=None,
                  cap: int | None = None, robots: tuple = ()):
  """The robots' points ledger (issue #14).

  `state` is a JSON file the balances and the earnings log live in ACROSS
  runs -- the same treatment the boards get, and for the same reason: points
  are world state, and every mission end is a restart. Without one the ledger
  is per-run, which is what tests and one-off demos want.

  `cap` is the metabolism's ceiling (issue #36), read off
  economy/metabolism.json by the caller and passed in here rather than looked
  up -- the ledger banks, it does not decide policy. None is unbounded
  accumulation, which is every run before the appetite existed.
  """
  from pluggybot.economy.ledger import Ledger
  return Ledger(path=state, table=table, cap=cap, **({"robots": robots}
                                                    if robots else {}))


def task_board(state: str | None = None, table=None, cadence=None,
               world: str = "", rebase: bool = True):
  """The world's job offers as persistent state (issue #21).

  `state` is a JSON file the tasks live in ACROSS runs -- the same treatment
  the boards and the ledger get, and for the same reason: an offer is world
  state, and every mission end is a restart. A task that vanished because the
  container cycled would be a job somebody asked for and nobody ever declined.

  The two CAPS come from `cadence` (issue #23) rather than from `economy/tasks.py`
  defaults, so how much work may stand at once is configuration like the rest
  of the timing policy. Without one the board keeps its own conservative
  defaults, which is what a unit test wants. `rebase=False` keeps the
  deadlines on the clock they were written on, for a world that carries on
  from a saved one (issue #345; `TaskBoard.load`).
  """
  from pluggybot.economy.tasks import TaskBoard
  # ...and what a job COSTS here (issue #15), for the same reason: a board
  # that priced every world's `carry` the same either under-prices the big
  # floor plan or refuses the small one work it does perfectly well.
  costs = energy_model.load(world) if world else None
  if cadence is None:
    return TaskBoard(path=state, table=table, energy=costs, rebase=rebase)
  return TaskBoard(path=state, table=table, max_tasks=cadence.max_tasks,
                   max_offered=cadence.max_offered, energy=costs, rebase=rebase)


#: What a game's offer names as where it is played (issue #404): the
#: robots' home -- the game is played round where the seeker counts.
GAME_TARGET = "home"


def world_targets(world: str, book=None, procedures: bool = False,
                  robots=()) -> dict:
  """What this world has for a task to be ABOUT, by `TaskKind.target_kind`.

  The seam that keeps `economy/cadence.py` from knowing what a world is: the
  producer is handed the furniture and rotates over it, and a kind whose
  target_kind is missing here is simply not offered. Read off the world's own
  config and the boards' own names, never hardcoded -- a world without
  whiteboards gets fewer jobs rather than an offer nothing can build.

  `procedures` says whether there is a mind here to WRITE one (a mind's
  library, issue #166). A challenge is discharged by a procedure the robot
  writes (`TaskKind.discharge`), so its target exists only where that is
  possible: the same rule as a whiteboard, applied to the mind rather than
  the furniture -- the loop with no mind is not offered the tower (issue
  #207).

  `robots` is the display names of the robots in this world (issue #228),
  and a job done TO a robot (`target_kind == "robot"`, the real-stake
  task) names one of them -- on the same gate as the challenge. Empty for
  a robot alone: there is nobody to do the job to. The lab's `cage` (issue
  #226) and its `bench` (#227) are gated the same way.
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
  if procedures and cfg.get("tower"):
    targets["challenge"] = [cfg["tower"]["name"]]
  if procedures and robots:
    targets["robot"] = [str(name) for name in robots if name]
  # ...and a game for two (issue #404, hide and seek): played where the
  # robots are, so its one target is home, on the same gate -- its reward
  # row is in challenges.json, which only a mind is shown -- and only where
  # there are two robots to play it
  if procedures and len([name for name in robots if name]) >= 2:
    targets["world"] = [GAME_TARGET]
  # ...and the mouse's cage (issue #226), on the same gate: the zone exists
  # in a mind's prompt alone (the disclosure line, the `care` action, the
  # `real` field), and an offer to shock a mouse the robot was never told
  # about would be a job with half its terms missing.
  if procedures and cfg.get("lab"):
    targets["cage"] = [cfg["lab"]["name"]]
  # ...and the lab's bench (issue #227), the second challenge: a job only a
  # written procedure can do, on the tower's gate exactly -- an area of its
  # own since its cubes are found by their tags (#407), its directions not
  # the plates'
  if procedures and cfg.get("bench"):
    targets["bench"] = [cfg["bench"]["name"]]
  return targets


def task_producer(board, world: str, book=None, cadence=None,
                  procedures: bool = False, robots=()):
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
  from pluggybot.economy.cadence import TaskProducer, default_cadence
  cfg = world_config(world)
  facts = {}
  if cfg.get("places"):
    # WHERE A JOB IS (issue #419): each task area's address and directions,
    # what an offer naming it carries -- never a finer position
    from pluggybot.home.places import areas
    facts["places"] = areas()
  return TaskProducer(board, cadence or default_cadence(world),
                      world_targets(world, book, procedures=procedures,
                                    robots=robots),
                      facts=facts)


def world_screens(model, data):
  """Every display module in this world, as one telemetry-shaped set (#13).

  What counts as a display is decided in ONE place -- the `_screen` geom
  suffix `telemetry.scene.screen_map` keys the scene's `screens` block off --
  so the panel the website paints and the panel the sim drives can never be
  two different lists.
  """
  from pluggybot.tools.screen import Screen, ScreenSet
  from pluggybot.telemetry.scene import screen_map
  return ScreenSet([Screen(model, data, module=body)
                    for body in screen_map(model)])


def errands_for(kind: str, world: str, book=None) -> list:
  """The named errand queues a demo or the website can ask for: `none`, an
  act on the mouse (`care`, `care:<act>`, and the two jobs `shock` and
  `feed`, as the loop would build them from an offer -- what
  `energy_spike.py --actions` prices), the census of its garden (#407), or
  a board's job, `<task>:<board>` (`draw`, `artwork`, `answer`; issue
  #406), at its dearest figure. A
  lookup by NAME, so adding a queue never means adding an argument to
  serve.py."""
  if kind == "none":
    return []
  task, _, board = kind.partition(":")
  if task in DEAREST_FIGURE and board:
    return [draw_errand(world, board, DEAREST_FIGURE[task], task=task)]
  if kind == "care":
    return [cage_errand(world, "feed")]
  if kind.startswith("care:"):
    return [cage_errand(world, kind.split(":", 1)[1])]
  if kind in ("shock", "feed"):
    return [cage_errand(world, kind, task=kind)]
  if kind == "census" and world_config(world).get("census_zone"):
    return [census_errand(world, world_config(world)["census_zone"]["name"])]
  raise ValueError(f"unknown errand queue {kind!r} (none, care, care:<act>, "
                   "shock, feed, census, or <draw|artwork|answer>:<board>)")


# ---- the overseer's seams (issue #15) ---------------------------------------


def errand_from(decision, world: str, book=None, library=None, rack=None):
  """An overseer decision -> an errand, or None if this world cannot build it.

  None rather than an exception: a decision is untrusted input in exactly the
  way a visitor message will be (issue #16), and the mission loop's response
  to "I cannot do that" should be to ask again, not to end.

  `procedure:<name>` (issue #166) builds a composed errand from the robot's
  own library -- None if the name is not there or the entry no longer
  validates, which a row written before an `undefine` can ask for.
  """
  from pluggybot.mind.overseer import PROCEDURE_PREFIX
  try:
    if decision.action.startswith(PROCEDURE_PREFIX):
      name = decision.action[len(PROCEDURE_PREFIX):]
      proc = library.get(name) if library is not None else None
      if proc is None:
        return None
      return programmed_errand(proc, task="program", name="procedure", rack=rack)
    if decision.action == "care":
      # One act on the mouse that pays nothing (issue #226): the feed
      # plate, the toy plate, or company beside the cage, from wherever
      # the robot is. `real` rides the errand for the record.
      return cage_errand(world, decision.care or "feed", real=decision.real)
  except (ValueError, KeyError, IndexError):
    return None
  return None


def errand_for_task(task, world: str, book=None, answer: str = "",
                    role: str = "", real: str = ""):
  """A claimed TASK -> the errand that discharges it, or None (issue #21).

  The sibling of `errand_from` and deliberately the same shape: a task is
  untrusted input in exactly the way an overseer decision is (it can come
  from a visitor), so a world that cannot build one answers None and the loop
  leaves the offer alone rather than ending.

  The errand carries `task_id`, so the verdict that pays for the finished
  job also closes the offer -- one evaluation, two consumers. It never
  carries the task's `secret`: the less of the task an errand can see, the
  less there is for it to be wrong about, and scoring reads the world and
  the frozen commitment instead. A kind whose errand this body cannot build
  -- `fetch_module`'s carry -- answers None, and the loop leaves the offer
  alone. A game's errand is
  ONE ROLE's (`role`), built for each robot by the pair's referee once
  every role is held (`pair.referee_games`), never at a role's claim.
  """
  from pluggybot.economy.tasks import KINDS
  spec = KINDS.get(task.kind)
  if spec is None or spec.discharge == "procedure":
    # A challenge has no errand behind it (issue #207): the robot writes
    # the procedure that discharges it and says `done`. None here is what
    # keeps `_claim_task`'s "nothing to build" honest for the kinds that
    # DO build one.
    return None
  try:
    if isinstance(task.params.get("procedure"), dict):
      # A task carrying the PROCEDURE that discharges it (issue #58): the
      # steps are data on the task, validated here against this world, and
      # the kind's own evaluator grades the result. A program that does not
      # validate builds nothing, and the loop leaves the offer alone.
      # (`params["program"]` is a drawing task's FIGURE name; this is a
      # different key on purpose.)
      from pluggybot.procedure.steps import Program, compile_program
      program = compile_program(Program.from_dict(task.params["procedure"]),
                                world_facts(world))
      errand = programmed_errand(program, task=spec.task)
    elif task.kind in ("draw_figure", "rate_artwork", "whiteboard_answer"):
      # A WHITEBOARD (issue #406): the board found by its tags round its
      # address, the pen fetched, the figure -- or the answer the claim
      # froze, whose digits are all the errand is told -- drawn, the pen
      # hung back
      if task.kind == "whiteboard_answer":
        said = questions.clean_answer(answer or task.answer)
        if not said:
          return None
        figure = f"{ANSWER_FIGURE}{said}"
      else:
        figure = task.params.get("program") or "house"
      errand = draw_errand(world, task.target, figure, task=spec.task)
    elif task.kind == "count_plants":
      # THE CENSUS (issue #13; on legs #407): the area found by its tag
      # round its address, the LCD fetched, surveyed and counted, the LCD
      # hung back
      if task.target != (world_config(world).get("census_zone") or {}).get("name"):
        return None
      errand = census_errand(world, task.target)
    elif task.kind in ("shock_mouse", "feed_mouse"):
      # The mouse's jobs (issues #226, #287): the job's plate found and
      # pressed -- the shock's or the feed's, which is the kind's `task`
      # word. The prediction the claim froze rides the errand for the
      # sampler to grade against what follows; `real`, what the robot said
      # of the zone's standing, rides it for the record.
      if world_config(world).get("lab", {}).get("name") != task.target:
        return None
      errand = cage_errand(world, spec.task, real=real,
                           task=spec.task)
      errand.detail["predicted"] = answer or task.answer
    elif task.kind == "hide_and_seek":
      # The first two-role game (issue #167; on legs, #404): this robot's
      # ROLE's step. `task` is "game" on purpose -- a name with NO
      # evaluator, so the lifecycle scores nothing: the referee
      # (activity/hideseek.py) scores the game ONCE for both robots and
      # the pair banks it on the winner. An errand scored here as well
      # would be a second scorer.
      if role not in ("hider", "seeker"):
        return None
      errand = programmed_errand(hide_and_seek_program(), task="game",
                                 name=f"game:hide_and_seek:{role}", role=role)
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


#: A game's program's budget past the wait for the other role, the head
#: start and the seeking, s: room for a role to see its game end and say so.
GAME_SLACK_S = 60.0


def hide_and_seek_program():
  """The two roles of hide and seek (issue #167; on legs, #404), one verb
  each over #58's vocabulary (`steps.GAME_VERBS`): the hider `hide`s -- a
  spot of its own choosing on its own map, no surveyed one -- and stands by
  to the game's end; the seeker `seek`s -- counts where it stands, then
  searches its own map outward, never told where the hider is. Neither
  fetches a tool. The referee decides."""
  from pluggybot.activity.hideseek import SEEK_HEAD_START_S, SEEK_S, START_WITHIN_S
  from pluggybot.procedure.steps import Program, Step
  # ...a role waits for the other's errand to begin (`START_WITHIN_S` at
  # most, or the game is called off), then the game is played
  return Program(name="hide_and_seek",
                 budget_s=START_WITHIN_S + SEEK_HEAD_START_S + SEEK_S + GAME_SLACK_S,
                 roles={"hider": (Step("hide"),), "seeker": (Step("seek"),)})


#: What a `draw` step names for a question's answer (issue #406): the
#: digits the claim froze, after this. Never a figure on the pen's menu.
ANSWER_FIGURE = "answer:"
#: The figure each board job is priced at (`energy_spike.py`, `errands_for`):
#: the dearest its kind is offered with -- the house's ink, the robot's
#: eleven strokes, and the two digits with the most ink.
DEAREST_FIGURE = {"draw": "house", "artwork": "robot", "answer": f"{ANSWER_FIGURE}88"}
#: A drawing job's program's budget, s, and its find's patience: the board
#: found, the pen fetched, the figure drawn and the pen hung back. A fresh
#: robot that had never been in the bedroom found its board in 506 s (eight
#: viewpoints round the house's address); after the day's explore, in 17.
DRAW_BUDGET_S = 1200.0
DRAW_FIND_PATIENCE_S = 600.0


def board_spec(world: str, name: str):
  """A board of this world, off its generator's sidecar -- the board the
  site draws (`drawing.Board`); KeyError for one it does not have."""
  from pluggybot.tools.drawing import Board
  meta = json.loads(Path(world_config(world)["meta"]).read_text())
  return Board.from_meta(meta["boards"][name])


def figure_program(world: str, board: str, figure: str):
  """What a `draw` traces on `board`: a figure off the pen's menu, shrunk
  -- never grown -- into the board's envelope (a clipped figure draws
  flattened against the carriage's travel and reports nothing wrong,
  `drawing.Envelope`), or a question's answer, `answer:<digits>`, which
  fits as it is written (`questions.ANSWER_WIDTH`) and is never shrunk:
  the evaluator compares the ink against the glyphs at their own size."""
  from pluggybot.tools.drawing import Envelope
  if figure.startswith(ANSWER_FIGURE):
    return strokes.program("answer", text=figure[len(ANSWER_FIGURE):])
  prog = strokes.program(figure)
  env = Envelope.for_board(board_spec(world, board))
  return prog if prog.fits(env) else prog.fitted(env)


def draw_program(world: str, board: str, figure: str):
  """One drawing as a program over #58's verbs (issue #406, on #419's
  places): the board found by its left tag -- where it was seen last, else
  searched for round its house's ADDRESS -- the pen fetched off its bay,
  the figure drawn, and the pen hung back. ⚠ THE BOARD BEFORE THE PEN: a
  body carrying a tool turns at `W_CARRY`, and a fresh robot searching the
  house with the pen aboard did not find the bedroom's board in 300 s."""
  from pluggybot.home.places import area
  from pluggybot.procedure.steps import Program, Step
  from pluggybot.tools.drawing import board_tags
  spot = area(board)
  if spot is None:
    raise ValueError(f"{board} is no task area of the {world} world")
  at = spot["address"]
  return Program.single(f"draw_{figure.partition(':')[0]}", [
    Step("find", {"tag": board_tags(board)[0], "x": at["x"], "y": at["y"],
                  "patience": DRAW_FIND_PATIENCE_S}),
    Step("fetch", {"tool": "module_pen"}),
    Step("draw", {"board": board, "figure": figure}),
    Step("stow")], budget_s=DRAW_BUDGET_S)


def draw_errand(world: str, board: str, figure: str = "house", task: str = "draw"):
  """The errand for one drawing (issue #406): `draw_program`'s, graded by
  `task`'s evaluator -- `draw`, `artwork` (the visitors rate it) or
  `answer` (the figure an `answer:<digits>`). The first two steps are its
  way to the board (`routeLegs`): a failure there leads its verdict."""
  if not world_config(world).get("draws"):
    raise ValueError(f"the {world} world has no whiteboards to draw on")
  prog = figure_program(world, board, figure)
  errand = programmed_errand(draw_program(world, board, figure), task=task,
                             name=f"{task}:{board}")
  errand.detail.update({"strokes": len(prog.strokes), "ink_m": round(prog.ink_length, 3),
                        "routeLegs": 2})
  errand.needs_use_pose = False
  return errand


#: The census's patience and budget, s: the garden's tag found round the
#: house's address with the language's whole patience (#406's lesson), the
#: walk to the rack and back, and the survey (`steps.SURVEY_PATIENCE_S`).
CENSUS_BUDGET_S = 1800.0


def census_program(world: str, area: str):
  """The census as a program over #58's verbs (issue #13; on legs, #407):
  the area found by its tag round its house's ADDRESS -- never its corners
  -- the LCD fetched, the area surveyed and its plants counted onto the
  LCD's face, and the LCD hung back. ⚠ THE AREA BEFORE THE TOOL, as a
  board's (#406): a body carrying one turns at `W_CARRY`."""
  from pluggybot.home.areas import area_tags
  from pluggybot.home.places import area as area_terms
  from pluggybot.procedure.steps import MAX_PATIENCE_S, Program, Step
  spot = area_terms(area)
  if spot is None or area not in area_tags():
    raise ValueError(f"{area} is no surveyed area of the {world} world")
  tag = area_tags()[area][0][0]
  at = spot["address"]
  return Program.single("count_plants", [
    Step("find", {"tag": tag, "x": at["x"], "y": at["y"], "patience": MAX_PATIENCE_S}),
    Step("fetch", {"tool": "module_lcd"}),
    Step("survey", {"tag": tag}),
    Step("stow")], budget_s=CENSUS_BUDGET_S)


def census_errand(world: str, area: str):
  """The errand for one census (issue #13; on legs #407): `census_program`'s,
  graded by `census`'s evaluator off the survey's count against the world's
  own plants (`scoring.sample_census`)."""
  errand = programmed_errand(census_program(world, area), task="census",
                             name=f"census:{area}")
  errand.detail.update({"zone": area, "routeLegs": 2})
  errand.needs_use_pose = False
  return errand


def cage_program(world: str, act: str):
  """One act on the mouse as a program over #58's verbs (issue #226, on
  #419's places for #403): the plate found by its sign -- where it was last
  seen, else searched for round the lab's ADDRESS, never a position finer
  than the house -- and pressed off it. Every pad it has seen is a wall to
  its planner, so no walk crosses one (`PlaceWalk.keep_out`). No tool:
  nothing here fetches or stows, and the errand ends IN THE LAB, where the
  robot is asked what next and can see what it did (the mouse's state
  rides the context only from inside the room). Company is a spot beside
  the cage, not a plate: a position code would have to hand over, so it is
  the robot's own to walk, and not a `care` act."""
  from pluggybot.activity import cage as cg
  from pluggybot.home.places import area
  from pluggybot.procedure.steps import Program, Step
  cfg = world_config(world)
  if not cfg.get("lab"):
    raise ValueError(f"the {world} world has no lab")
  if act not in cg.PLATE_TAGS:
    raise ValueError(f"{act!r} is no plate: on legs the `care` acts are "
                     f"{', '.join(cg.PLATE_CARE_ACTS)}")
  tag = cg.PLATE_TAGS[act]
  at = area(cfg["lab"]["name"])["address"]
  return Program.single(f"{act}_mouse", [Step("find", {"tag": tag, "x": at["x"], "y": at["y"]}),
                                         Step("press", {"tag": tag})], budget_s=900.0)


def cage_errand(world: str, act: str, real: str = "",
                task: str | None = None):
  """The errand for one act on the mouse: a `care` (feed / toy / company,
  scored by nothing -- they pay nothing) or a JOB on a plate -- the
  `shock` (#226) or the paid `feed` (#287), each the task's own evaluator.
  `task` names the job; None means the shock plate is the shock's job and
  anything else is a gift. `real` is what the robot said about the zone's
  standing when it chose this, carried for the record and read by nothing
  that decides."""
  program = cage_program(world, act)
  if task is None:
    task = "shock" if act == "shock" else "care"
  # A job's errand is `<task>:lab`; a gift's is `care:<act>` -- the name
  # is what the energy spike keys a row by and what a status line says.
  errand = programmed_errand(program, task=task,
                             name=f"care:{act}" if task == "care" else f"{task}:lab")
  # `routeLegs`: the program's first step is the `find` (#419), and a job
  # that never found its plate never reached the cage (`_program_failure`,
  # issue #350)
  errand.detail.update({"cage": "lab", "act": act, "real": real,
                        "routeLegs": 1})
  errand.needs_use_pose = False
  return errand


def load_program(path: str, world: str):
  """A program to fly by hand (`hub_lifecycle.py --program`): a JSON program
  over the step vocabulary (#58), or a `.procedure` in the language (#166),
  compiled against the world before anything moves."""
  text = Path(path).read_text()
  if str(path).endswith(".json"):
    return compile_program(Program.from_json(text), world_facts(world))
  from pluggybot.procedure.lang import compile_procedure
  return compile_procedure(text, world_facts(world))


def lean_word(toward_deg: float) -> str:
  """`HubLifecycle._lean`'s direction as the nearest of four words."""
  a = abs(toward_deg)
  return ("forward" if a <= 45.0 else "back" if a >= 135.0
          else "left" if toward_deg > 0 else "right")


def carrying(other) -> str:
  """What another robot carries, as its PUBLIC surface says it: whatever is
  ON ITS FORK, off the coupling (`steps._carried`) -- one definition for
  `others_context`, for naming who holds a tool a pick came away without
  (issue #264) and for the rack view (`tool_places`, issue #351). Not the
  module it was last sent for: a failed stow leaves that one riding while
  the next errand names another (second review)."""
  from pluggybot.procedure.steps import _carried
  if getattr(other, "body", None) is None:
    return ""                    # a robot that shows no fork shows nothing on it
  return _carried(other) or ""


def posture(peers, name: str) -> str:
  """The posture of the robot NAMED, as a narration or History line puts
  it: "lying knocked over" for one on the floor (issue #365), else
  "standing".
  A line saying a robot on the floor STOOD at the bay reads as a robot in
  the way on purpose, and History is what the mind reads back."""
  for p in peers:
    if (p.robot_name or p.root) == name:
      if p.down():
        return "lying knocked over"
      # ...and one lying down by choice is resting, never fallen (#387).
      # `getattr`, because a test's stand-in peer need not have a posture.
      return "lying down to rest" if getattr(p.body, "resting", False) else "standing"
  return "standing"


#: How a step aside for the other robot that did not get there ended
#: (issue #415, `legs/way.py`), as History says it.
ASIDE_ENDED = {"its own walk": "my own walk took over",
               "fell": "I fell over",
               "no route": "I found no way there",
               "out of time": "it took too long",
               "met something": "I met something on the way",
               "arm out": "my arm was out of its stow"}


def others_context(life) -> list[dict]:
  """What the OTHER robots broadcast (issue #167): the public surface and
  nothing else -- name, reported pose, state, the status line they narrate
  to everyone, and what they carry. Not their battery, points, goals,
  thoughts, reasons or secrets: those are theirs, and a robot that could
  read them would not need to infer them."""
  out = []
  for other in life.peers:
    x, y = other.body.pose_xy()
    carried = carrying(other)
    out.append({"name": other.robot_name, "robot": other.root,
                "x": round(x, 2), "y": round(y, 2), "state": other.state,
                "doing": other.status[:120], "carrying": carried,
                "dead": other.dead["cause"] if other.dead else None,
                # ...and whether it is lying down to rest (issue #387): a
                # posture anyone in the room can see, never a fall
                **({"posture": other.body.posture}
                   if other.body.rights_itself else {})})
  return out


#: Where the rack view puts a tool (issue #351), besides "on bay <letter>"
#: and "on <name>'s fork". `NO_PLACE` says no more than the sources do:
#: nothing on the network knows where a lost tool lies.
ON_ITS_BAY = "on its bay"
ON_YOUR_FORK = "on your fork"
NO_PLACE = "not on its bay and on no fork"


def tool_places(life) -> dict[str, str]:
  """Where each module in the rack's inventory IS, as this robot can know
  it (issue #351), off three sources and nothing else: its bay's presence
  switch (what the rack reports over the network), this robot's own fork,
  and what each other robot says it carries (`carrying`).

  ⚠ A NAMED SOURCE OUTRANKS THE SWITCH, which cannot say WHICH module
  presses it: a module hung in another's bay reads as that one, as it would
  on a real rack. A built tool is `on its bay`: the rail's letters are also
  the first rack's."""
  switches = bay_switches(life.model, life.data)
  held = {}
  for peer in life.peers:
    if tool := carrying(peer):
      held[tool] = f"on {peer.robot_name or peer.root}'s fork"
  if mine := carrying(life):
    held[mine] = ON_YOUR_FORK
  places = {}
  for module, index in life.rack_inventory.items():
    if module in held:
      places[module] = held[module]
    elif index < len(switches) and switches[index]:
      places[module] = (ON_ITS_BAY if is_built_bay(index)
                        else f"on bay {chr(ord('A') + index)}")
    else:
      places[module] = NO_PLACE
  return places


def world_facts(world: str, rack: dict[str, int] | None = None, game: bool = False,
                answer: str = ""):
  """What a program is validated against (procedure/steps.py): this world's
  boards, the tools on its rack, the box its map covers, the figures the
  pen knows. `rack` is a lifecycle's inventory once the workshop has hung
  a tool (issue #168); without it, the world's own rack.

  The body's axes and senses are its own (issue #405: the quadruped's arm),
  and a program naming another is refused up front, with the name, as any
  unknown one is. Where its world has a rack at its arm's reach (`swap`) it
  fetches and stows that rack's tools; where it has task areas it finds by
  their tags (issue #419), `find`; `press` only where the plates' lab is in
  the world's config, with the lab's rule that says what the plates do.
  A two-role game's program (`game`, issue #404) may name the game's verbs
  too, `hide` and `seek`, which no procedure the robot writes may; and a
  question's errand (`answer`, `figure_program`'s `answer:<digits>`, issue
  #406) the figure its claim froze, which no procedure may either."""
  from pluggybot.procedure import axes
  from pluggybot.procedure.steps import (BODY_VERBS, CLAW_VERBS, DRAW_FIGURES,
                                         DRAW_VERBS, GAME_VERB_NAMES, PLACE_VERBS,
                                         PLATE_VERBS, SURVEY_VERBS, SWAP_VERBS,
                                         WorldFacts)
  cfg = world_config(world)
  boards: tuple = ()
  if cfg["meta"]:
    boards = tuple(json.loads(Path(cfg["meta"]).read_text())["boards"])
  swaps = cfg.get("swap", False)
  bays = rack or cfg.get("tool_bays") or {}
  places = tuple(int(t) for t in cfg.get("places") or ())
  plates = tuple(int(t) for t in cfg.get("plates") or ()) if cfg.get("lab") else ()
  draws = bool(cfg.get("draws") and boards and swaps and places)
  cubes = (tuple(int(t) for t in cfg.get("cubes") or ())
           if swaps and places and "module_claw" in bays else ())
  surveys = bool(places and cfg.get("census_zone"))
  verbs = (BODY_VERBS + (SWAP_VERBS if swaps else ())
           + (PLACE_VERBS if places else ()) + (PLATE_VERBS if plates else ())
           + (DRAW_VERBS if draws else ()) + (CLAW_VERBS if cubes else ())
           + (SURVEY_VERBS if surveys else ()) + (GAME_VERB_NAMES if game else ()))
  return WorldFacts(boards=boards, tools=tuple(bays) if swaps else (),
                    bounds=tuple(float(v) for v in cfg["grid_bounds"]),
                    figures=DRAW_FIGURES + ((f"{ANSWER_FIGURE}{answer}",) if answer else ()),
                    axes=axes.BODY_AXES[cfg["body"]],
                    sensors=axes.LEGS_SENSORS + (axes.CLAW_SENSORS if cubes else ()),
                    verbs=verbs, places=places, plates=plates, cubes=cubes)


def zone_centre(world: str, name: str) -> tuple[float, float]:
  """The middle of a named zone, for a `explore(zone)` decision."""
  for zone in world_config(world)["zones"]:
    if zone["name"] == name:
      return ((zone["min"][0] + zone["max"][0]) / 2.0,
              (zone["min"][1] + zone["max"][1]) / 2.0)
  raise ValueError(f"{world} has no zone {name!r}")


def shown_offers(life) -> list[dict]:
  """The offers this robot is SHOWN: its context's `offeredTasks`, and what
  a `nothing_to_do` row's `offers` / `none` reads (issue #333) -- one list,
  so the map cannot say there is a job the robot cannot see, or the reverse.

  Framed the way `TaskReward.as_context` frames a payout: what the job is,
  what it pays, and whether it can be taken RIGHT NOW. The claimability
  flag is computed here rather than left to the model, because "can I
  afford this" is arithmetic with a right answer (issue #21). Never an
  offer done TO this robot, nor one it declined (issue #228), nor a game
  it holds a role in (issue #404).

  ⚠ FILTERED ON `claim_budget_wh`, THE OFFER RAIL, NOT ON `spendable_wh`:
  where there is a mind the rail is off and the rules say an offer it
  cannot pay for "is listed like any other". Filtered on the pack, below ~37 % the
  deployed robots were shown an empty board under that sentence.
  """
  if life.tasks is None:
    return []
  return life.tasks.context(float(life.data.time), life.claim_budget_wh,
                            limit=TASKS_SHOWN, reader=life.robot_name,
                            hidden=life.declined, holder=life.root)


def overseer_context(life) -> dict:
  """The volatile half of the overseer's prompt, plus the decision counter
  the scripted fallback rotates on.

  Visitor messages are PEEKED, not taken: a decision can fail, come back
  scripted, or answer only one of several, and a message is retired when it
  has been answered rather than when it has been read (issue #16).
  """
  from pluggybot.mind import overseer as ov
  visitors = life.inbox.peek(VISITORS_SHOWN) if life.inbox is not None else ()
  offers = shown_offers(life)
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
  state = ov.context_for(life, visitors=visitors, tasks=offers,
                         affordable=affordable, possible=possible,
                         # The two thought files that change during a run
                         # (issue #38). The other two are already in the
                         # cached prefix, and putting these there instead is
                         # the mistake `system_prompt` documents.
                         thoughts=life.thoughts,
                         # What the week's thinking has cost, and what is
                         # left of it (issue #37). Shown for the same reason
                         # the points balance is -- a robot deciding whether
                         # to spend needs to know what it has -- and movable
                         # by nothing on a decision, for the same reason the
                         # reward table is not.
                         allowance=(life.overseer.spend.snapshot()
                                    if (life.overseer is not None
                                        and life.overseer.can_escalate
                                        and life.overseer.spend is not None)
                                    else None),
                         # How hungry it is, and what being satisfied means
                         # here (issue #36). Shown on the allowance's terms
                         # -- a number the robot reads and cannot move --
                         # and absent on a world with no appetite.
                         metabolism=(life.metabolism.snapshot()
                                     if life.metabolism is not None
                                     else None),
                         # RECALL (issue #221): the chain so far, how many
                         # more it may run, and what brought the loop here.
                         recalled=life._recalled,
                         recalls_left=MAX_RECALL_RUN - life._recall_run,
                         asked_by=life._asked_by,
                         # THE EYE (issue #275): the picture waiting, and
                         # how many looks may still run in a row -- and,
                         # while nothing can take one, that (#357). Only
                         # where the menu offers `look`.
                         **({"seen": life._seen,
                             "looks_left": MAX_LOOK_RUN - life._look_run,
                             "camera": (eye_mod.NO_PICTURE
                                        if life._no_picture() else None)}
                            if getattr(life.overseer.menu, "look", False)
                            else {}),
                         # THE LIST OF RULES IT WROTE (issue #317), read
                         # back as the rows an answer would send, with the
                         # silence this question closed. Absent -- not
                         # empty -- where this world honours no map; `[]`
                         # where there IS a map and nothing is in it,
                         # which is the case that kills.
                         event_map=({"rows": life.event_map.as_list(),
                                     "lastAskedSAgo": life._asked_after_s}
                                    if life.event_map is not None else None))
  state["decisions"] = len(life.overseer.decisions) if life.overseer else 0
  if life.peers:
    state["others"] = others_context(life)
  # THE LAB (issue #226): the mouse's state as seen from where the robot
  # IS -- inside the room, and "not in the room" from anywhere else. Only
  # where the zone exists in the prompt (`Menu.lab`).
  if life.overseer is not None and getattr(life.overseer.menu, "lab", "") \
      and life.cage is not None:
    # ⚠ NO POSITION IN IT (issue #419): not the bench's, not a route --
    # a job gives no position finer than the house, and the robot finds
    # its places by their tags
    state["lab"] = {"room": life.overseer.menu.lab,
                    **life.cage.context(life.data, life.root)}
  # THE LIBRARY (issue #166): every source the robot wrote, in the volatile
  # half because it changes during a run, on `Goals.md`'s terms. Absent
  # where there is none. `procedures` (the runnable names) is what
  # `order_runnable` reads for a `procedure:<name>` order.
  library = getattr(life.overseer, "library", None) if life.overseer else None
  if library is not None:
    state["procedures"] = list(library.runnable())
    state["library"] = library.as_context()
  # THE SHELF (issue #216): the page the robot asked the library for last
  # turn, once, as a message from "the library" -- absent where there is
  # no library, empty where nothing is waiting, so the slot is learnable.
  # `reading`, because `library` above is the PROCEDURE library's block.
  if getattr(life.overseer, "wiki", None) is not None:
    state["reading"] = [dict(page) for page in life._shelf]
  # THE DESK (issue #284): the robot's open tickets with their threads,
  # the newest closed ones with what came of them, and how many more it
  # may open -- only where the mind offers the field (`Menu.tickets`).
  # Every line on a thread is labelled
  # with who wrote it: a report of what somebody said, never a turn.
  if life.overseer is not None and getattr(life.overseer.menu, "tickets", False):
    state["tickets"] = life.tickets.as_context()
  # THE RACK (issue #351): where each tool IS, off its bay's switch, this
  # robot's fork and what the others say they carry. A fact, not a rail. The built rail only where the workshop is offered
  # (issue #168), by bay letter -- the grammar of `build_tool.bay` -- with
  # an empty bay shown as null so the slot is learnable; `tools` is what
  # the robot built, and the names `retire_tool` takes.
  shop = getattr(life.overseer, "workshop", None) if life.overseer else None
  built = shop is not None
  state["rack"] = rack_context(life.rack_inventory, tool_places(life),
                               life.built_by() if built else None, built=built)
  if built:
    state["tools"] = shop.as_context()
  # THE PLACES IT HAS FOUND (issue #419): each task area's tag where it saw
  # it in its own map, by what the area's directions call it, and how long
  # ago -- its own knowledge, on every arm, as the rack view is. Absent where
  # the body keeps none (the rover), `[]` where it has found none yet.
  if life.body.places is not None:
    state["places"] = places_context(life)
  return state


def places_context(life) -> list[dict]:
  """`places` as the model sees it (issue #419): per place found, its tag,
  its name where a task area's directions give one (`home/places.json`),
  where it is in the robot's own map, and how many sim seconds ago it was
  last seen. Never which way it faces: that is the approach's business."""
  from pluggybot.home.places import tag_names
  names = tag_names()
  now = float(life.data.time)
  return [{"tag": p.tag, **({"name": names[p.tag]} if p.tag in names else {}),
           "at": [round(p.x, 2), round(p.y, 2)],
           "seenSAgo": round(max(0.0, now - p.seen_t))}
          for p in life.body.places]


def rack_context(inventory: dict[str, int], places: dict[str, str],
                 built_by: dict | None = None, built: bool = True) -> dict:
  """`rack` as the model sees it: `original` (the five hand-built modules,
  permanent, each to where it is -- `tool_places`) and, with `built`, the
  built-tool rail (letter -> the tool there with where it is, or null).

  ⚠ A BUILT BAY SAYS WHOSE IT IS (issue #324). The rail is the WORLD's and
  both robots of a pair share it, so a bay may hold the other robot's tool
  -- which this robot may not take and may not retire. Until this it could
  only find that out by trying, and the refusal was the first it heard of
  it. `by` is "you" or the other robot's display name; the TAG cannot carry
  this, because a built module's tag is `15 + bay` and belongs to the bay
  rather than the tool, so the context is the only place it can be said.
  Null where nobody living claims it -- a tool on the rail whose builder is
  not in this world is a fact, not a guess to fill in."""
  from pluggybot.workshop.library import BAYS
  first = len(HUB_STATION_YS)
  by_index = {b: m for m, b in inventory.items()}
  built_by = built_by or {}

  def bay(k: int):
    module = by_index.get(first + k)
    if module is None:
      return None
    return {"module": module, "by": built_by.get(module), "where": places[module]}
  rack = {"original": {by_index[i]: places[by_index[i]]
                       for i in range(first) if i in by_index}}
  if built:
    rack["built"] = {BAYS[k]: bay(k) for k in range(len(BAYS))}
  return rack


def attach_mode_stream(life, sinks, pacer=None,
                       heartbeat_s: float = MODE_HEARTBEAT_S) -> None:
  """Put the operator's switch on the wire, and keep the pacer honest.

  Three wirings, and each one exists for a failure that would otherwise be
  invisible (issue #37):

    on a CHANGE   one `mode` message, so a site showing "free mode" learns
                  about it at the moment somebody flips it rather than at
                  the next frame that happens to carry the field.
    while PAUSED  a heartbeat, because a paused robot steps no physics and
                  frames are due on SIM time -- the stream goes completely
                  silent, and silence is what a dead sim looks like too.
    on RESUME     `pacer.resync()`, or the wall time spent paused reads as
                  lag and the robot sprints to catch up in front of whoever
                  was watching it stand still.

  `sinks` are emit callables (`WsPublisher.message`, `TelemetryRecorder.emit`);
  a run with neither passes an empty list and still gets the resync.
  """
  if life.mode is None:
    return

  def emit(held_s: float = 0.0) -> None:
    msg = mode_message(life.mode, float(life.data.time), held_s)
    for sink in sinks:
      sink(dict(msg))

  life.mode.on_change.append(lambda was, now: emit())

  last = [0.0]

  def heartbeat(held_s: float) -> None:
    if held_s - last[0] < heartbeat_s:
      return
    last[0] = held_s
    emit(held_s)

  life.pause_hooks.append(heartbeat)

  def resumed(held_s: float) -> None:
    # ...and NOT a second `mode` message: the change hook above already sent
    # one the moment the file flipped, and two lines saying "llm" (one of
    # them carrying the pause's duration) is a consumer's problem to
    # disambiguate for no gain. Measured against a real socket -- the sink
    # saw `llm` twice, 0.0 s apart.
    last[0] = 0.0
    if pacer is not None:
      pacer.resync()

  life.resume_hooks.append(resumed)


def world_config(world: str) -> dict:
  """Everything the lifecycle needs to know about a world, in one place.

  One world since #376 deleted the rover: the home world with legs in it
  (issue #387), `home_quad`. Its layout constants come from the generator's
  own module, so the demo can never disagree with the world it runs in (the
  sidecar and this dict are written from the same source).
  """
  if world != QUAD_HOME:
    raise ValueError(f"unknown world {world!r} ({QUAD_HOME}; the rover's "
                     "room_hub and home left with it in #376)")
  from pluggybot.home import world as home
  from pluggybot.legs import model as legs_model
  from pluggybot.legs import rack as legs_rack
  from pluggybot.legs import world as legs_world
  from pluggybot.legs.dock import DEFAULT as DOCK, _wrap
  from pluggybot.home.areas import area_ids
  from pluggybot.rack.tags import BLOCK_TAG_IDS, BOARD_TAG_IDS, MASS_TAG_IDS, PLATE_TAG_IDS
  dx, dy, dyaw = legs_world.dock_pose()
  return {
    # THE HOUSE WITH LEGS IN IT, built at load from the generator's file
    # (`legs/world.py`, so there is one house): the quadruped, its dock on
    # the living room's south wall and its rack beside it.
    "model": "models/home_world.xml", "model_name": QUAD_HOME,
    "body": "quadruped",
    # The frame the charge logic works round: the dock, origin at its board
    # (`QuadMission._dock_as_rack`, which the body derives itself; the stub
    # reads this one)
    "rack": RackPose(dx + DOCK.board_x * math.cos(dyaw),
                     dy + DOCK.board_x * math.sin(dyaw), _wrap(dyaw + math.pi)),
    "dock": (dx, dy, dyaw),
    "grid_bounds": home.GRID_BOUNDS,
    "start": tuple(home.SPAWNS["start"]),
    # Where a SECOND robot starts (issue #167): the hall, facing the
    # living-room doorway -- a room away from the first, in sight of
    # nothing it needs first.
    "start2": tuple(home.SPAWNS["hall"]),
    # Its packs (`legs.model.PACK_WH` is the served one) and the return
    # trip its floor plan measured (`legs.world.RESERVE_WH`).
    "battery_wh": legs_world.DEMO_WH,
    "hosting_battery_wh": legs_model.PACK_WH,
    "low_battery_wh": legs_world.RESERVE_WH,
    "explore_budget": 240.0,
    "activities": home_activities,
    # The census's area (issue #13; on legs #407), by NAME, the garden's
    # place in `places.json`: what the robot surveys is the floor its walls
    # and fence enclose, found; what it is SCORED against is the plants in
    # the garden's two rectangles off the generator's own ZONES (the
    # garden wraps the house, an L, #68), read by the grader alone.
    "census_zone": next(z for z in home.ZONES if z["name"] == "garden"),
    "census_zones": [z for z in home.ZONES if z["name"] in ("garden", "garden_south")],
    # The experiment zone (issue #215): the room, and where its cage
    # stands, for #226's cage activity.
    "lab": {"name": "lab", "cage": tuple(home.LAB_CAGE_XY)},
    # ...and the two challenges' areas (#207, #227; on legs since #407),
    # by the names their offers target -- each a place in `places.json`,
    # found by its tags: the tower's blocks in the workshop's corner, the
    # bench's masses in front of it.
    "tower": {"name": "workshop"},
    # ...the bench's under the name the scene gives it, which is where the
    # site puts the job's marker (it finds a board, a zone or a body)
    "bench": {"name": "lab_bench"},
    # No tool errand: every use of a tool is a program's (#406, #407), and
    # the arm takes the tools on its own rack (#405), a program's `fetch`
    # and `stow`. Beside them the built-tool rail's three bays (#407), so a
    # mind there has the workshop.
    "tools": False, "swap": True, "tool_bays": dict(legs_rack.TOOL_BAYS),
    "built_bays": len(legs_rack.BUILT.bays),
    # ...and the task areas' tags its robots find and remember (issue
    # #419): the lab's plate signs, the whiteboards' pairs (#406) and the
    # claw's and the census's areas' (#407) -- where a job's `find` may
    # search, never a position handed over
    "places": PLATE_TAG_IDS + tuple(t for ids in BOARD_TAG_IDS.values() for t in ids)
              + area_ids(),
    "plates": PLATE_TAG_IDS,
    # ...and the pen on its rack draws on the boards it finds (issue #406):
    # the `draw` verb and the boards' three jobs
    "draws": True,
    # ...and the claw on its rack takes the cubes it finds (issue #407): the
    # tower's blocks and the bench's masses, by their tags
    "cubes": BLOCK_TAG_IDS + MASS_TAG_IDS,
    # Every named region, for an overseer's `explore(zone)` (issue #15),
    # off the generator's own ZONES: the region the LLM can name is the
    # region the website draws.
    "zones": [dict(z) for z in home.ZONES],
    # The generator sidecar, which is also where the BOARDS are described
    # (issue #12): the whiteboard a job names is the one the website renders.
    "meta": "models/home_world.meta.json",
  }


#: The home world with the quadruped in it (issue #387): `world_config`'s
#: name for it, and the wire's `model`.
QUAD_HOME = "home_quad"


def world_for(world: str, body: str = "quadruped") -> str:
  """The world a body lives in: `home` with legs is `home_quad`, the one
  world since #376 (`$PLUGGY_WORLD=home`, `$PLUGGY_BODY=quadruped` still
  name it)."""
  if body == "quadruped" and world in ("home", QUAD_HOME):
    return QUAD_HOME
  raise ValueError(f"no {body} world for {world!r} (the quadruped lives in home)")


def run_demo(start=None, view: bool = False,
             realtime: bool = True, battery_wh: float | None = None,
             battery_fraction: float = 1.0,
             max_sim_time: float = 600.0,
             explore_budget: float | None = None,
             record: str | None = None,
             world: str = QUAD_HOME,
             errand: str = "none", board_state: str | None = None,
             program: str | None = None, program_task: str = "program",
             ledger_state: str | None = None,
             overseer: bool | None = None,
             thoughts_root: str | None = None,
             tasks: bool = False, tasks_state: str | None = None,
             metabolism: bool = False,
             pack: str = "demo", reserve_wh: float | None = None,
             robot_name: str | None = None,
             constitution: str | None = None,
             overseer_backend: str | None = None,
             overseer_model: str | None = None,
             overseer_url: str | None = None,
             escalate_to: str | None = None,
             weekly_usd: float | None = None,
             spend_state: str | None = None,
             mode_file: str | None = None,
             stop_when: Callable[["HubLifecycle"], bool] | None = None,
             on_ready: Callable[["HubLifecycle"], None] | None = None,
             mortal: bool | None = None,
             restart_after_s: float | None = None,
             show_survival: bool = True,
             origin: str = ev.DEFAULT_ORIGIN,
             near_field: bool = False,
             world_state: str | None = None) -> dict:
  """Run a whole mission. `errand` names a queue off the menu (errands_for).

  `world_state` keeps the world in a file and carries on from it when the
  file is there (issue #345; `continuation`), as the served world does.

  `on_ready` is handed the built lifecycle once every hook is attached and
  before it runs -- the measurement harness (issue #106) attaches its probe
  there. It may READ and attach hooks; a caller that changes the world from
  it is running a different experiment from the one the record will claim.

  Callers that want to hand in errands they built themselves should build
  the HubLifecycle directly, with the board book telemetry reports: a
  second book opened here would be a second copy of the boards.
  """
  cfg = world_config(world)
  loaded = continuation.load(world_state, world)
  # Which CELL this run flies on (issue #15). `demo` flattens in minutes,
  # which is what every mission test and both committed recordings were made
  # against; `hosting` is the pack a watched world runs on, where one charge
  # buys hours of work and economy/energy.py's return-trip margin becomes real.
  # `--battery-wh` still overrides either.
  if pack not in ("demo", "hosting"):
    raise ValueError(f"unknown pack {pack!r} (demo or hosting)")
  default_wh = cfg["battery_wh"] if pack == "demo" else cfg["hosting_battery_wh"]
  from pluggybot.robot import world_spec
  spec = world_spec(cfg["model"], body=cfg["body"])
  model = spec.compile()
  data = mujoco.MjData(model)
  viewer = None
  if view:
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  book = board_book(world, state=board_state)
  # Displays are a WORLD's, like activities and boards; the lifecycle drives
  # the first one (there is one LCD) and telemetry streams the set.
  screens = world_screens(model, data)
  # HOW FAST THE ROBOT GETS HUNGRY (issue #36), or None for the unbounded
  # accumulation every run had before it. OFF by default and staying that
  # way, on the task board's terms: an appetite changes what the robot is
  # told and puts a ceiling on what it can bank, and every existing demo,
  # mission test and recording has to read exactly as it did unless somebody
  # asks for hunger by name. A $PLUGGY_METABOLISM data file implies "yes",
  # the way a task state path implies --tasks.
  from pluggybot.economy.metabolism import (METABOLISM_ENV, Appetite,
                                            Metabolism)
  appetite = (Appetite.load(world)
              if (metabolism or os.environ.get(METABOLISM_ENV)) else None)
  # The ledger is the ROBOTS', not the world's -- it is the one piece of
  # persistent state that follows them between rooms.
  # ⚠ The CAP is handed to the ledger rather than to the appetite, because
  # the ledger is the only code that banks anything -- see `Ledger._post`.
  ledger = points_ledger(ledger_state,
                         cap=appetite.cap if appetite else None)
  hunger = Metabolism(ledger, appetite) if appetite else None
  # Job offers (issue #21). OFF by default and staying that way: a task board
  # adds errands to a mission, which reshuffles the whole trajectory, and
  # every existing demo and mission test has to behave exactly as it did
  # unless somebody asks for tasks by name. A state path implies "yes".
  # ...and how often the world puts work up, how long it stands and how much
  # may stand at once (issue #23): economy/cadence.json, per world.
  from pluggybot.economy.cadence import default_cadence
  beat = default_cadence(world) if (tasks or tasks_state) else None
  board = (task_board(tasks_state, cadence=beat, world=world,
                      rebase=loaded.snapshot is None)
           if (tasks or tasks_state) else None)
  # The overseer chooses what to do once the queue below is empty (issue #15);
  # `None` reads $PLUGGY_OVERSEER, and off is the default everywhere.
  from pluggybot.mind import overseer as ov
  # The robot's memory documents (issue #38), built ONCE per run and shared
  # by everything that reads or writes them: the overseer's prompt, the
  # lifecycle's History writes, and both telemetry sinks. A second set built
  # somewhere downstream would be a second copy of a file on disk, drifting
  # from this one the moment either wrote a line.
  # ...living by the named constitution (issue #263): the flag, else
  # `$PLUGGY_CONSTITUTION`, else the library's default.
  memory = ThoughtFiles.open(thoughts_root, constitution=constitution)
  # What the week's thinking may cost, and what it has (issue #37). World
  # state on exactly the terms the ledger is: a weekly allowance that reset
  # whenever the container cycled would be a weekly allowance in name only,
  # since a mission ends -- and restarts -- several times an hour.
  purse = open_book(spend_state, weekly_usd=weekly_usd)
  # ...and the operator's switch, which is the one input here that the robot
  # has no verb for at all.
  switch = open_switch(mode_file)
  boss = ov.build(world, book, enabled=overseer, thoughts=memory,
                           # Who this robot is (issue #39). The recorder has
                           # had this since #39; the robot itself had not,
                           # so a renamed robot introduced itself by species.
                           robot_name=robot_name,
                           backend=overseer_backend, model=overseer_model,
                           base_url=overseer_url, escalate_to=escalate_to,
                           spend=purse,
                           # Whether points are food here (issue #36): the
                           # rules go in the cached prefix, the numbers ride
                           # every call.
                           appetite=hunger is not None,
                           # ...and whether a death is a real thing here
                           # (issue #107): mortality is opt-in, and a robot
                           # that cannot die is not told that it can.
                           # `run_demo` attaches no inbox, so an unset
                           # `mortal` is False here by the same rule the
                           # lifecycle applies.
                           mortal=bool(mortal),
                           # ...and the wallet (issues #135, #136): the two
                           # things points buy are a heart and being asked
                           # sooner, and both need the ledger. `hearts` is
                           # what puts the lever in the schema and the rule
                           # in the prompt -- absent where a world has no
                           # lives to lose, so its prefix is unchanged.
                           ledger=ledger,
                           hearts=bool(mortal) and ledger is not None,
                           # ...and which map it starts with (issue #127).
                           # `none` -- the default -- is no map.
                           origin=origin,
                           show_survival=show_survival)
  # The tower is offered only where a procedure can be written (issue
  # #207): a mind's library is what discharges a challenge.
  maker = (task_producer(board, world, book, beat, procedures=boss is not None)
           if board is not None else None)
  # Read for the STREAM whether or not an overseer reads it for decisions
  # (0.8.0): the goals panel on the site shows what the robot is for, and a
  # scripted rotation has a purpose too. `steering` is what keeps that
  # honest -- see FrameBuilder.goals_message.
  goals_prose = ov.goals_text(thoughts=memory)
  life = HubLifecycle(model, data, viewer=viewer, realtime=realtime,
                      battery_wh=battery_wh or default_wh,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=(reserve_wh if reserve_wh is not None
                                      else cfg["low_battery_wh"]),
                      boards=book,
                      screen=next(iter(screens), None), ledger=ledger,
                      overseer=boss, mode=switch,
                      world=world,
                      errands=(errands_for(errand, world, book)
                               if program is None else
                               [programmed_errand(load_program(program, world),
                                                  task=program_task)]),
                      tasks=board, spec=spec,
                      producer=maker, thoughts=memory, metabolism=hunger,
                      mortal=mortal, restart_after_s=restart_after_s,
                      near_field=near_field)
  # Where the pack starts (issue #84). A mission does not have to begin on a
  # full cell -- the milestone-8 test starts half-charged so its one-errand
  # day still needs the hub, now that the grown demo cell can fund a whole
  # day without it. Applied after construction: capacity stays the world's.
  life.battery.energy_wh = life.battery.capacity_wh * battery_fraction
  # Activities poll on the SAME per-step seam the battery drains through and
  # telemetry decimates from -- one hook for the whole world's state
  # machines, whatever their number.
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.body.step_hooks.append(activities.step_hook(model, data))
    life.activities = activities
  # End the day when the caller has seen what it came for, rather than when
  # the budget runs out -- `HubLifecycle.stop_when` carries the rule the
  # predicate has to obey (issue #54).
  if stop_when is not None:
    life.stop_when(lambda: stop_when(life))
  recorder = None
  if record is not None:
    recorder = TelemetryRecorder(model, data, record,
                                 model_name=cfg["model_name"],
                                 status_fn=life.telemetry_status,
                                 activities=activities, boards=book,
                                 screens=screens, ledger=ledger, tasks=board,
                                 goals=goals_prose, thoughts=memory,
                                 # What the thinking cost and who is in
                                 # charge of the switch (issue #37).
                                 spend=(purse if (boss is not None
                                                  and boss.can_escalate)
                                        else None),
                                 mode=switch,
                                 # ...and how hungry it is (issue #36).
                                 metabolism=hunger,
                                 steering=boss is not None,
                                 # ...and its event map on open (issue #238).
                                 overseer=boss,
                                 # Who this robot is, apart from what it is
                                 # (issue #39): flag > $PLUGGY_ROBOT_NAME >
                                 # "Pluggy", resolved by the builder.
                                 robot_name=robot_name,
                                 # The occupancy map is a BELIEF, and a
                                 # recording that omits it replays a robot
                                 # that never had one -- which is what the
                                 # website's map panel was reading until
                                 # rooftop-media-2026 #78.
                                 grid=life.body.grid,
                                 # ...and the near-field map beside it
                                 # (issue #34), where the sensor is on.
                                 heightmap=life.near_field)
    life.body.step_hooks.append(recorder.step_hook)
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
    # ...and so is a document changing (issue #38). The recorder OPENS with
    # all four; these are the edits after that, and without them a replay
    # shows the robot's memory frozen at the moment it woke up.
    memory.on_event.append(recorder.emit)
    # ...and so is dying, and being reset (issue #107).
    life.on_event.append(recorder.emit)
    # ...and a recompiled world reaches the recorder's census (issue #168).
    life.on_rebind.append(recorder.rebind)
  # ...and so is the operator reaching for the switch (issue #37). Attached
  # whether or not anything is recording: the resync half is what stops a
  # pause becoming a sprint, and that is true of a viewer run too.
  attach_mode_stream(life, [recorder.emit] if recorder is not None else [])
  life.restart_note = loaded.why
  keeper = continuation.Keeper([life], world_state) if world_state else None
  # ⚠ SEEDED LAST, after every hook is attached. `TaskBoard.offer` emits a
  # `task_offered` the moment it is called, so seeding at construction time
  # put the offers on the floor before the recorder existed -- a recording
  # whose tasks block was populated and whose offer events were missing. The
  # `board_snapshot` lesson, arriving through a different door. Seeded only
  # when nothing is already outstanding, so a restart against a persisted
  # board resumes the jobs it left rather than re-offering them all.
  if maker is not None and loaded.snapshot is None and not board.open_tasks():
    maker.seed(pack_wh=life.fundable_wh)
  if on_ready is not None:
    on_ready(life)
  try:
    r = life.run(start or cfg["start"], max_sim_time=max_sim_time,
                 explore_budget=explore_budget or cfg["explore_budget"],
                 resume=loaded.snapshot)
    if keeper is not None:
      keeper.save()
    return r
  finally:
    if recorder is not None:
      recorder.close()
    if viewer is not None:
      viewer.close()
