"""Live PluggyWorld publisher (webserver v1): run the mission, stream it.

Runs the battery-driven hub lifecycle headless, paced to real time, while
publishing protocol frames + occupancy-grid images + event lines to a
WebSocket endpoint (an outbound CLIENT -- point it at the website's ingest
socket, or at scripts/ws_sink.py to watch the wire locally). The sim never
blocks on the network: if the endpoint is down, frames drop and the
mission carries on. Docs: docs/Webserver.md; wire format: protocol/README.md.

Usage:
  MUJOCO_GL=egl    uv run python scripts/serve.py --endpoint ws://localhost:8765
  MUJOCO_GL=osmesa uv run python scripts/serve.py --endpoint ws://localhost:8765
    # CPU-only rendering, the deploy-server configuration
  ... --world home      # the generated house + garden (issue #6); room_hub
                        # is the default. Everything the world implies --
                        # model, scene name, rack pose, grid extent, battery,
                        # start pose, errand destination, explore budget --
                        # comes from hub.lifecycle.world_config(), so this
                        # flag can never half-apply.
  ... --errand draw     # a real drawing errand (issue #12): fetch the pen,
                        # navigate to a whiteboard, erase it, draw, stow. The
                        # strokes stream as `draw` events for the browser to
                        # paint -- they are never MuJoCo geometry.
  ... --boards state.json     # whiteboard contents that survive a restart
  ... --ledger points.json    # the points ledger, likewise (issue #14): the
                              # balance and the earnings log the site shows
  ... --arm autonomous  # WHICH ARM this world flies (issue #142;
                        # $PLUGGY_ARM). Unset is what it always was: the
                        # --overseer flag decides and the arm is read off
                        # what got built. `--rung A0|A1` picks the rung of
                        # the autonomous ladder -- A0 HIDES the survival
                        # clock, and without that A0 and A1 are one run.
                        # ⚠ Flipping the DEPLOYED world to autonomous is a
                        # deliberate act, not a config change: Evaluation.md
                        # §2 argues it stays guarded, and that argument is
                        # updated in the PR that changes it, never
                        # silently contradicted.
  ... --restart-after 300     # SIM seconds a DEAD robot lies there before
                        # it stands itself up (issue #143). ON here and off
                        # in the experiment harness; 0 leaves it waiting for
                        # a person, as every world did before. ⚠ NOT an
                        # intervention: world behaviour on a timer is not an
                        # admin's hand, and `interventions` is what excludes
                        # a run from survival statistics.
  ... --rate 2.0        # sim seconds per wall second (default 1.0)
  ... --free-run        # no pacing: measure this machine's real-time multiple
  ... --record out.jsonl.gz   # also keep a v0 recording of the same run
  ... --token SECRET    # ingest shared secret (default: $PLUGGYWORLD_TOKEN)

The website's ingest path is authenticated; point --endpoint at
ws://host/api/pluggyworld/ingest and give it the same secret the server
holds. Prefer the environment variable to the flag -- an argument is
visible in `ps` to every user on the box.
"""

import argparse
import os
import time

import mujoco

from pluggybot.mind import llm, overseer
from pluggybot.mind.mode import open_switch
from pluggybot.mind.overseer import ESCALATE_MODEL
from pluggybot.mind.spend import WEEKLY_USD, open_book
from pluggybot.mind.inbox import Inbox
from pluggybot.economy.cadence import default_cadence
from pluggybot.evaluation.arms import (
  ARM_ENV, RUNGS, RUNG_ENV, arm_flags, rung_for,
)
from pluggybot.evaluation.record import build_identity
from pluggybot.economy.metabolism import METABOLISM_ENV, Appetite, Metabolism
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.lifecycle import (
  RESTART_AFTER_S, HubLifecycle, attach_mode_stream, board_book, errands_for,
  points_ledger, task_board, task_producer, world_config, world_screens,
)
from pluggybot.telemetry.pacer import RealTimePacer
from pluggybot.telemetry.protocol import CODE_HANDLED_TYPES, INBOUND_TYPES
from pluggybot.telemetry.publisher import WsPublisher
from pluggybot.telemetry.recorder import KEYFRAME_S, TelemetryRecorder


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--endpoint", default="ws://localhost:8765",
                      help="WebSocket endpoint to publish to")
  parser.add_argument("--world", choices=("room_hub", "home"),
                      default="room_hub",
                      help="which world to serve: room_hub (default) or the "
                           "generated home world (issue #6)")
  parser.add_argument("--rate", type=float, default=1.0,
                      help="pacing: sim seconds per wall second")
  parser.add_argument("--free-run", action="store_true",
                      help="disable pacing (real-time-multiple measurement)")
  parser.add_argument("--pack", choices=("demo", "hosting"), default="demo",
                      help="which cell to serve on (issue #15). `demo` is the "
                           "minutes-long cell the mission tests use; "
                           "`hosting` is the hours-long one a watched world "
                           "wants, where economy/energy.py's return-trip margin "
                           "becomes real and an errand is deferred rather "
                           "than started on a pack that cannot finish it")
  parser.add_argument("--battery-wh", type=float, default=None,
                      help="battery capacity (overrides --pack)")
  parser.add_argument("--reserve-wh", type=float, default=None,
                      help="override the world's go-charge reserve, in Wh")
  parser.add_argument("--max-sim-time", type=float, default=600.0)
  parser.add_argument("--restart-after", type=float, default=RESTART_AFTER_S,
                      metavar="S",
                      help="SIM seconds a dead robot lies there before it "
                           "stands itself up (issue #143; 0 disables it and "
                           "the robot then waits for a person, which is what "
                           "every world did before). ON here and off in the "
                           "experiment harness: a served world whose robot is "
                           "on the floor until somebody notices is not a "
                           "world anybody can watch, and a measured run is "
                           "about ONE life")
  parser.add_argument("--robot-name", default=None, metavar="NAME",
                      help="this robot's display name on the wire (issue "
                           "#39): the identity the site shows, e.g. 'Luca "
                           "the pluggybot'. Default $PLUGGY_ROBOT_NAME, then "
                           "'Pluggy'. Never the body name: renaming a robot "
                           "must not re-key its telemetry")
  parser.add_argument("--record", default=None, metavar="PATH",
                      help="also write a v0 JSONL recording of this run")
  parser.add_argument("--token", default=os.environ.get("PLUGGYWORLD_TOKEN"),
                      help="ingest shared secret (default $PLUGGYWORLD_TOKEN)")
  parser.add_argument("--keyframe-s", type=float, default=KEYFRAME_S,
                      metavar="S", help="sim seconds between full keyframes"
                                        " (0 disables; late joiners then wait"
                                        " forever)")
  parser.add_argument("--errand", choices=("carry", "draw", "draw2", "census",
                                          "dance", "artwork", "showcase",
                                          "none"),
                      default="carry",
                      help="what the robot is FOR this run (issue #12): carry "
                           "(the milestone-8 LCD errand), draw (fetch the pen, "
                           "erase a whiteboard and draw on it), draw2 (two "
                           "boards, charging in between), none")
  parser.add_argument("--boards", default=None, metavar="PATH",
                      help="JSON file the whiteboards' contents live in "
                           "between runs (default: blank boards every start)")
  parser.add_argument("--ledger", default=None, metavar="PATH",
                      help="JSON file the points ledger lives in between runs "
                           "(issue #14; default: the robot starts at zero)")
  parser.add_argument("--overseer", action="store_true",
                      help="let an LLM choose what to do next once --errand's "
                           "queue is empty (issue #15; $PLUGGY_OVERSEER). "
                           "Needs $ANTHROPIC_API_KEY -- without one the robot "
                           "runs the scripted fallback and the run reports it")
  parser.add_argument("--arm", choices=("scripted", "guarded", "autonomous"),
                      default=os.environ.get(ARM_ENV) or None,
                      help="WHICH ARM this world flies (issue #142; "
                           "$PLUGGY_ARM; docs/Evaluation.md §2). `scripted` "
                           "is the rotation with no mind, `guarded` is a mind "
                           "with every rail on -- what the deployed world has "
                           "always been -- and `autonomous` takes the three "
                           "rails off, corrects the prompt to match and makes "
                           "the fallback the agent's own standing order. "
                           "Unset behaves exactly as before: --overseer "
                           "decides, and the arm is READ OFF what was built")
  parser.add_argument("--rung", choices=tuple(sorted(RUNGS)),
                      default=os.environ.get(RUNG_ENV) or None,
                      help="which rung of the `autonomous` ladder "
                           "($PLUGGY_RUNG; default A0). A0 HIDES the survival "
                           "clock and A1 restores it -- without that they are "
                           "one run. Refused on an arm with no ladder rather "
                           "than ignored")
  parser.add_argument("--tasks", action="store_true",
                      help="offer the robot JOBS this run (issue #21): each "
                           "one has a description, a target, a reward off "
                           "economy/rewards.json and a deadline. More arrive on "
                           "the cadence in economy/cadence.json as the run goes "
                           "on (issue #23; $PLUGGY_CADENCE re-points it). "
                           "Off by default")
  parser.add_argument("--metabolism", action="store_true",
                      help="POINTS ARE FOOD (issue #36): the robot consumes "
                           "points at a steady rate on sim time, stops "
                           "banking at a cap, and once it has enough it is "
                           "SATISFIED and spends the rest of its time on its "
                           "goals. Off by default -- it changes what the "
                           "robot is told. Rate and cap are data "
                           "(economy/metabolism.json; $PLUGGY_METABOLISM "
                           "re-points it and implies this flag)")
  parser.add_argument("--task-state", default=None, metavar="PATH",
                      help="JSON file the task board lives in between runs "
                           "($PLUGGY_TASKS; implies --tasks)")
  parser.add_argument("--goals", default=None, metavar="PATH",
                      help="the overseer's long-term goals, as prose. Mount it "
                           "and edit it to change what the robot is for, no "
                           "redeploy ($PLUGGY_GOALS)")
  parser.add_argument("--journal", default=None, metavar="PATH",
                      help="JSON file the overseer's notes-to-self live in "
                           "between runs ($PLUGGY_JOURNAL)")
  parser.add_argument("--thoughts", default=None, metavar="DIR",
                      help="directory the robot's THOUGHT FILES live in "
                           "(issue #38; $PLUGGY_THOUGHTS). Main.md and "
                           "Goals.md are yours to edit, History.md is "
                           "append-only and written by the sim, and "
                           "Knowledge_and_Opinions.md is the robot's own")
  parser.add_argument("--overseer-backend", default=None,
                      choices=llm.BACKENDS, metavar="NAME",
                      help="WHICH MIND decides (issue #19; "
                           "$PLUGGY_OVERSEER_BACKEND): anthropic, "
                           "huggingface, local (a model on this machine -- "
                           "ollama by default, no network and no bill), "
                           "openai-compatible, or auto (the default: the "
                           "model id's shape, as before)")
  parser.add_argument("--overseer-model", default=None, metavar="ID",
                      help="the model that decides ($PLUGGY_MODEL). Defaults "
                           f"to {overseer.MODEL} on Anthropic and "
                           f"{llm.LOCAL_MODEL} on the local backend")
  parser.add_argument("--overseer-url", default=None, metavar="URL",
                      help="base URL of the local / openai-compatible "
                           f"endpoint ($PLUGGY_OVERSEER_URL; default "
                           f"{llm.LOCAL_URL}, ollama's). A key for a "
                           "third-party one goes in $PLUGGY_OVERSEER_KEY -- "
                           "never a flag, since ps is public")
  parser.add_argument("--escalate-to", default=None, metavar="ID",
                      help="a BIGGER mind the robot may buy a decision from "
                           "when it says it is unsure (issue #37; "
                           "$PLUGGY_ESCALATE_TO). Off unless set. The routing "
                           "costs no extra call -- the model asks on the "
                           "decision it was already making, and code decides "
                           f"whether it can afford one. Try {ESCALATE_MODEL}")
  parser.add_argument("--weekly-usd", type=float, default=None,
                      metavar="USD",
                      help="the soft weekly allowance escalations spend "
                           f"against ($PLUGGY_WEEKLY_USD; default {WEEKLY_USD})"
                           ". The HARD ceiling is the provider account "
                           "balance, which is deliberately not code")
  parser.add_argument("--spend-state", default=None, metavar="PATH",
                      help="JSON file the week's spending lives in between "
                           "runs ($PLUGGY_SPEND). A weekly budget that reset "
                           "on restart would be no budget at all")
  parser.add_argument("--mode-file", default=None, metavar="PATH",
                      help="the OPERATOR's control file ($PLUGGY_MODE_FILE): "
                           "{\"mode\": \"llm\"|\"scripted\"|\"paused\"}. "
                           "Polled, never written -- the robot has no verb "
                           "that reaches it. `scripted` is free mode (no API "
                           "calls, world still alive); `paused` stops the "
                           "physics and keeps the stream up")
  parser.add_argument("--overseer-budget", type=int, default=None,
                      metavar="N", help="hard cap on LLM calls per rolling "
                                        "hour (default 60)")
  args = parser.parse_args()

  # WHICH ARM (issue #142). Until this, `serve.py` REPORTED an arm and had no
  # way to set one: the identity header read `autonomous` off
  # `life.autonomous`, and nothing on this path could make that True.
  #
  # ⚠ ONE DEFINITION, IMPORTED (`evaluation/arms.py`). Re-deriving what an
  # arm means here is exactly how a stream comes to claim an arm nobody flew.
  #
  # ⚠ AND AN UNNAMED ARM CHANGES NOTHING. Without `--arm`/$PLUGGY_ARM the
  # overseer flag decides as it always has and the arm is read off what was
  # BUILT further down -- so every existing deployment, demo and test keeps
  # the behaviour it had, which is the same rule `--tasks` and `--metabolism`
  # are off under.
  flags = {}
  rung = None
  if args.arm:
    try:
      rung = rung_for(args.arm, args.rung)
      flags = arm_flags(args.arm, rung or "A0")
    except (ValueError, NotImplementedError) as e:
      parser.error(str(e))
    # ⚠ A CONTRADICTION IS REFUSED, NOT RESOLVED. `--overseer --arm scripted`
    # is somebody asking for two different worlds; picking one of them
    # quietly is how a header ends up honest about a run nobody meant.
    if args.overseer and not flags["overseer"]:
      parser.error(f"--overseer contradicts --arm {args.arm}, which has no "
                   "mind at all (docs/Evaluation.md §2)")
  elif args.rung:
    parser.error("--rung names a rung of the `autonomous` ladder, so it "
                 "needs --arm autonomous (docs/Evaluation.md §2)")

  cfg = world_config(args.world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  # Board state is the world's, not the run's: loaded before the mission and
  # written back on every stroke, so a restart walks into the house it left.
  book = board_book(args.world, state=args.boards)
  # The world's displays (issue #13): the lifecycle drives the LCD's face off
  # its own state, and every screen's content rides in the frames.
  screens = world_screens(model, data)
  # Points are world state too (issue #14), and the ROBOT's rather than the
  # room's: a balance that resets whenever the container cycles is not a
  # scoreboard. Every mission end is a restart, so the site's rivalry only
  # exists if this file does.
  # ...and how fast the robot gets HUNGRY (issue #36), which is what turns a
  # balance into a metabolism: consumed on sim time, capped rather than
  # accumulated, and once there is enough the robot has free time. World
  # state again -- the hunger lives in the balance, so the ledger file above
  # is what makes it survive the container cycling.
  appetite = (Appetite.load(args.world)
              if (args.metabolism or os.environ.get(METABOLISM_ENV)) else None)
  ledger = points_ledger(args.ledger, cap=appetite.cap if appetite else None)
  hunger = Metabolism(ledger, appetite) if appetite else None
  # Job offers (issue #21), and world state on exactly the terms the boards
  # and the ledger are: a task that vanished because the container cycled is
  # a job somebody asked for and nobody ever declined. Off unless asked for.
  # ...on the timing policy in economy/cadence.json (issue #23): how often work
  # appears, how long an offer stands, how much may stand at once and how long
  # a target rests. Configuration rather than constants, and $PLUGGY_CADENCE
  # re-tunes how busy a deployed world is without a rebuild.
  beat = default_cadence(args.world) if (args.tasks or args.task_state) else None
  tasks = (task_board(args.task_state, cadence=beat, world=args.world)
           if (args.tasks or args.task_state) else None)
  # ...and the thing that keeps putting work up, rather than a starter set
  # that never grows back.
  maker = (task_producer(tasks, args.world, book, beat)
           if tasks is not None else None)
  # The overseer decides what to do next once the preset queue is empty
  # (issue #15). Off unless asked for, and its memory is two more files in
  # the same volume the boards and the ledger live in: goals are read (and
  # human-edited between runs), the journal is written.
  overseer_kw = ({"calls_per_hour": args.overseer_budget}
                 if args.overseer_budget else {})
  # The thought files (issue #38), built ONCE and shared by everything that
  # reads or writes them -- the prompt, the History writes, and both wire
  # surfaces. Attached on EVERY served world, overseer or not: a scripted
  # rotation still has a history, and the site's Thoughts tab is what a
  # visitor opens first.
  memory = ThoughtFiles.open(args.thoughts, goals_path=args.goals)
  # The weekly allowance (issue #37), and world state on the same terms the
  # ledger is: a mission ends several times an hour here, so a budget that
  # lived in the process would be a budget that reset several times an hour.
  purse = open_book(args.spend_state, weekly_usd=args.weekly_usd)
  # ...and the operator's switch: a file this process only ever READS.
  switch = open_switch(args.mode_file)
  boss, journal = overseer.build(args.world, book,
                                 # A NAMED ARM IS THE STRONGER STATEMENT and
                                 # overrides $PLUGGY_OVERSEER, in both
                                 # directions: `--arm scripted` must be able
                                 # to turn a mind OFF, or the arm the header
                                 # reports is not the arm that flew.
                                 enabled=(flags["overseer"] if flags
                                          else args.overseer or None),
                                 goals_path=args.goals,
                                 journal_path=args.journal,
                                 thoughts=memory,
                                 # ...and its NAME (issue #39), so the robot
                                 # calls itself what the site's header calls
                                 # it rather than what its species is.
                                 robot_name=args.robot_name,
                                 backend=args.overseer_backend,
                                 model=args.overseer_model,
                                 base_url=args.overseer_url,
                                 escalate_to=args.escalate_to, spend=purse,
                                 # Whether points are food on this world
                                 # (issue #36): the RULES ride the cached
                                 # prefix, the numbers ride every call.
                                 appetite=hunger is not None,
                                 # ...and that it can DIE here (issue #107).
                                 # A served world is mortal -- it has an
                                 # inbox and an admin behind it -- and the
                                 # rule is only true where that holds, so it
                                 # is stated only there.
                                 mortal=True,
                                 # ...and WHOSE the fallback is, whether the
                                 # rails are off, and whether the robot may
                                 # see its own survival clock -- the three
                                 # things an arm IS (issue #142). Absent
                                 # means the arm was not named, and every
                                 # one of them keeps the default it had.
                                 **{k: v for k, v in flags.items()
                                    if k != "overseer"},
                                 **overseer_kw)
  # The goals file is read on every run, overseer or not: the site's goals
  # panel (rooftop-media-2026 #30) shows what the robot is FOR, and that is
  # as true of a scripted rotation as of a chosen errand. What is NOT the
  # same is whether anything is reading them, which is what `steering` says.
  goals_prose = overseer.goals_text(thoughts=memory)
  # The visitor channel (issue #16), attached ALWAYS as of issue #30 -- but
  # what this run advertises it can hear is per-kind (`accepts` below).
  # Suggestions and questions still need an overseer to read them; a rating
  # settles a ledger row and a `reset_tool` puts a dropped module back on its
  # bay, and BOTH are handled by code the moment the physics thread drains
  # them -- a scripted world's tools get lost (and its artwork rated) exactly
  # as often as a minded one's.
  inbox = Inbox()
  # The demo cell flattens in minutes, which reads on a watched stream as a
  # robot that only ever charges; `--pack hosting` is the hours-long one
  # (issue #15). The RESERVE is not scaled with it -- it is the absolute
  # energy needed to reach the dock, a property of the floor plan -- but on a
  # hosting pack it becomes a margin every errand must leave intact, which is
  # what economy/energy.py enforces and what stops a mid-errand death.
  pack_wh = (cfg["battery_wh"] if args.pack == "demo"
             else cfg["hosting_battery_wh"])
  life = HubLifecycle(model, data, inbox=inbox,
                      battery_wh=args.battery_wh or pack_wh,
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=(args.reserve_wh
                                      if args.reserve_wh is not None
                                      else cfg["low_battery_wh"]),
                      errands=errands_for(args.errand, args.world, book),
                      screen=next(iter(screens), None),
                      overseer=boss, journal=journal, mode=switch,
                      world=args.world,
                      boards=book, ledger=ledger, tasks=tasks,
                      producer=maker, thoughts=memory, metabolism=hunger,
                      # THE THREE RAILS, and this is the only reader of the
                      # flag that turns them off (issue #115): `needs_charge`,
                      # `_afford_next` and `claim_budget_wh`. The prompt is
                      # corrected in the same breath by `overseer.build`
                      # above, off the same dict.
                      autonomous=bool(flags.get("autonomous")),
                      # ...and the dead robot's own clock (issue #143). ON by
                      # default HERE and nowhere else: this world runs
                      # continuously and its robot dies most days on the
                      # `autonomous` arm, and a robot lying on the floor
                      # until a human notices is not a world anybody can
                      # watch. ⚠ NOT an intervention -- see
                      # `HubLifecycle.restart_after_s`.
                      restart_after_s=(args.restart_after
                                       if args.restart_after > 0 else None))
  # WHICH BUILD IS BEING WATCHED (issue #132; docs/Evaluation.md §5).
  #
  # This world is an OBSERVATORY, not an experiment: one uncontrolled
  # continuous run, an operator who can pause it and an admin who resets it,
  # and its numbers never enter a results table. What it can be is
  # ATTRIBUTABLE -- and until this block existed it was not, because the
  # header carried `protocolVersion` and nothing else, so a week of deployed
  # behaviour could not be told apart from the week before it under a
  # different build, a different model or a different rewards.json. That is
  # the exact failure `dataHashes` and `deadlineS` were added to the
  # experiment's series key to prevent; the experiment refuses to pool across
  # those regimes, and the observatory could not even detect one.
  #
  # Built HERE rather than inside the builders because it is the RUN's
  # identity, and both sinks of one run must carry the same one.
  # ⚠ READ OFF WHAT WAS BUILT, NEVER OFF WHAT WAS ASKED FOR. `--arm guarded`
  # on a box with no key builds an overseer that answers `fallback:no-client`
  # -- still `guarded`, and the fallback rate says the rest -- but an arm
  # whose mind could not be constructed at all is a `scripted` day, and the
  # header has to say the word for what ran.
  arm = ("scripted" if boss is None
         else "autonomous" if life.autonomous
         else "guarded")
  identity = build_identity(
    args.world, arm=arm,
    model=boss.model if boss is not None else None,
    backend=boss.backend if boss is not None else None,
    pack_wh=life.battery.capacity_wh,
    reserve_wh=life.low_battery_wh,
    # Not a data file, so no hash catches it, and it caps how much of a day
    # the model decided at all (issue #117).
    deadline_s=boss.timeout_s if boss is not None else None,
    # ...and which RUNG, where there is a ladder -- off the arm that was
    # BUILT, so a rung cannot outlive the arm it belongs to.
    rung=rung if arm == "autonomous" else None)
  print(f"build: {identity['commit']} / {identity['arm']}"
        + (f" {identity['rung']}" if identity.get("rung") else "")
        + (f" / {identity['model']} via {identity['backend']}"
           if identity["model"] else ""))

  # The world's task state machines, polled on the same per-step seam
  # everything else hangs off (issue #8). Their flags ride in the frames.
  activities = cfg["activities"](model, data) if cfg["activities"] else None
  if activities is not None:
    life.mission.step_hooks.append(activities.step_hook(model, data))
  publisher = WsPublisher(model, data, args.endpoint,
                          model_name=cfg["model_name"],
                          status_fn=life.telemetry_status,
                          grid=life.mission.grid, token=args.token,
                          keyframe_s=args.keyframe_s,
                          activities=activities, boards=book,
                          screens=screens, ledger=ledger, tasks=tasks,
                          # What this run can actually HEAR, per kind
                          # (issues #16, #30). The website reads it: a
                          # message is only "delivered" if somebody who can
                          # act on it got it, and that somebody is an overseer
                          # -- while a rating or an admin's tool reset is
                          # handled by code and heard on any served world.
                          accepts=(INBOUND_TYPES if boss is not None
                                   else CODE_HANDLED_TYPES),
                          goals=goals_prose, thoughts=memory,
                          # What the thinking costs, and who has the switch
                          # (0.12.0, issue #37). The spend block is only
                          # attached where there is spending to report --
                          # an all-zero money panel on a world that cannot
                          # spend is a panel that means nothing.
                          spend=(purse if (boss is not None and boss.can_escalate)
                                        else None),
                          mode=switch,
                          # ...and how hungry it is (0.13.0, issue #36).
                          metabolism=hunger,
                          steering=boss is not None,
                          robot_name=args.robot_name,
                          build=identity)
  life.mission.step_hooks.append(publisher.step_hook)
  life.say_hooks.append(publisher.event)
  # ...and dying / being reset are typed events too (issue #107).
  life.on_event.append(publisher.message)
  # A SERVED WORLD IS MORTAL (issue #107): there is an admin behind it who
  # can stand the robot back up, which is the whole condition. `HubLifecycle`
  # defaults to exactly this rule; it is asserted rather than set, so the
  # rule lives in one place.
  assert life.mortal, "a served world has an inbox and must be mortal"
  print("mortal: a death ends nothing -- an admin stands the robot up")
  if book is not None:
    # Strokes reach the browser as `draw` messages, never as geometry: the
    # website paints them into the board's canvas texture.
    book.on_event.append(publisher.message)
  # An award is an event on the same terms (issue #14): the balance rides in
  # the frames, and each `earned` message carries the verdict behind it.
  ledger.on_event.append(publisher.message)
  # ...and so is a job being offered, taken or resolved (issue #21).
  if tasks is not None:
    tasks.on_event.append(publisher.message)
  # ...and so are the robot's notes and its answers to visitors (#15, #16).
  if journal is not None:
    journal.on_event.append(publisher.message)
  # ...and so is a thought file changing (issue #38): the publisher opens a
  # stream with all four and these are the edits after that.
  memory.on_event.append(publisher.message)
  life.visitor_hooks.append(publisher.message)
  if inbox is not None:
    # THE OTHER DIRECTION. `offer` runs on the publisher's socket thread and
    # does nothing but validate and enqueue -- see mind/inbox.py for why that
    # is the whole of what it is allowed to do.
    publisher.on_inbound.append(inbox.offer)
  pacer = None
  if not args.free_run:
    pacer = RealTimePacer(data, rate=args.rate)
    life.mission.step_hooks.append(pacer.step_hook)
  recorder = None
  if args.record is not None:
    recorder = TelemetryRecorder(model, data, args.record,
                                 model_name=cfg["model_name"],
                                 status_fn=life.telemetry_status,
                                 keyframe_s=args.keyframe_s,
                                 activities=activities, boards=book,
                                 screens=screens, ledger=ledger, tasks=tasks,
                                 goals=goals_prose, thoughts=memory,
                                 spend=(purse if (boss is not None and boss.can_escalate)
                                        else None),
                                 mode=switch, metabolism=hunger,
                                 steering=boss is not None,
                                 robot_name=args.robot_name,
                                 build=identity,
                                 grid=life.mission.grid)
    life.mission.step_hooks.append(recorder.step_hook)
    if book is not None:
      book.on_event.append(recorder.emit)
    ledger.on_event.append(recorder.emit)
    if tasks is not None:
      tasks.on_event.append(recorder.emit)
    if journal is not None:
      journal.on_event.append(recorder.emit)
    memory.on_event.append(recorder.emit)
    life.visitor_hooks.append(recorder.emit)

  # The operator's switch, on the wire and off the pacer's clock (issue
  # #37). ⚠ The pacer is what makes this more than a display detail: a
  # PAUSED robot steps nothing, so without `resync` the wall time it stood
  # still reads as lag and the sim sprints to catch up the moment it is let
  # go -- at 2.9x, in front of whoever paused it to look at something.
  attach_mode_stream(life, [publisher.message]
                     + ([recorder.emit] if recorder is not None else []),
                     pacer=pacer)

  # ⚠ SEEDED LAST, after every hook above is attached: `offer` emits its
  # `task_offered` immediately, so seeding earlier drops those lines on the
  # floor -- see the note in `lifecycle.run_demo`. Only when nothing is
  # outstanding, so a restart resumes the jobs the last mission left.
  if maker is not None and not tasks.open_tasks():
    maker.seed(pack_wh=life.fundable_wh)

  wall0 = time.monotonic()
  try:
    r = life.run(cfg["start"], use_at=cfg["use_at"],
                 max_sim_time=args.max_sim_time,
                 explore_budget=cfg["explore_budget"])
  finally:
    publisher.close()
    if recorder is not None:
      recorder.close()
  wall = time.monotonic() - wall0

  print()
  print(f"mission state          : {r['state']}"
        f" (swaps={r['swaps_done']}, charges={r['charge_cycles']},"
        f" stowed={r['module_stowed']})")
  for e in r["errands"]:
    extra = (f"  {e['figure']} on {e['board']}, board {e['fill']:.0%} full"
             if e.get("board") else "")
    print(f"errand {e['errand']:<20s}: picked={e['picked']}"
          f" stowed={e['stowed']}{extra}")
  print(f"points balance         : {r['points']}"
        f" ({r['earned']} earned over {len(r['verdicts'])} scored task(s))")
  # `.get`, not `[...]`: this is a SUMMARY, and a summary line must never be
  # the thing that fails a run. tests/test_webserver.py drives this path with
  # a stubbed result dict, which is exactly the shape a caller on an older
  # lifecycle would hand it.
  if r.get("overseer"):
    o = r["overseer"]
    # Cost per SIM-hour, which is the number that matters for a box that runs
    # paced to real time: at --rate 1.0 a sim-hour is an hour of electricity.
    per_hour = o["usd"] / (r["sim_time"] / 3600.0) if r["sim_time"] else 0.0
    print(f"overseer               : {o['llmCalls']} LLM call(s), "
          f"{o['fallbacks']} scripted, budget {o['budgetLeft']}/"
          f"{o['callsPerHour']} left, cache hit {o['cacheHitRate']:.0%}")
    # ⚠ Three different sentences, because three different things are true
    # (issue #19). A local model is FREE -- zero is a measurement. A backend
    # whose rates could not be read is UNKNOWN -- printing $0.00000 there
    # would be a fabricated invoice, which is exactly what the acceptance
    # criterion forbids. Only a priced backend gets a number.
    backend = o.get("backend", "anthropic")
    where = f"{o['model']} on {backend}"
    if backend == "local":
      print(f"overseer cost          : no API cost -- {where}")
    elif not o.get("priced", True):
      print(f"overseer cost          : unknown -- {where} publishes no rates "
            "here, so the tokens above are the honest measure")
    else:
      print(f"overseer cost          : ${o['usd']:.5f}"
            f"  (${per_hour:.4f} per sim-hour, {where})")
    if o.get("escalationModel"):
      # What the ALLOWANCE bought (issue #37). Refusals are printed beside
      # the escalations because they are the more useful number: a robot
      # that keeps asking and keeps being told no is a budget too small or a
      # cadence too slow, and neither is visible from the count of the ones
      # that happened.
      money = o.get("allowance") or {}
      refused = o.get("escalationsRefused") or {}
      print(f"overseer escalations   : {o['escalations']} to "
            f"{o['escalationModel']} (${o.get('escalationUsd', 0.0):.5f}), "
            + (", ".join(f"{n} refused for {why}"
                         for why, n in sorted(refused.items())) or "none "
               "refused"))
      if money:
        print(f"weekly allowance       : ${money['spentUsd']:.4f} spent of "
              f"${money['weeklyUsd']:.2f}, ${money['leftUsd']:.4f} left"
              + (f", {money['unpriced']} call(s) at unknown rates"
                 if money.get("unpriced") else ""))
    if not o.get("constrained", True):
      # The menu is no longer enforced at the decoder -- see
      # Overseer.constrained. Worth a line: the fallback rate below is being
      # produced under a weaker guarantee than the default one.
      print("overseer decoding      : UNCONSTRAINED -- this endpoint refused "
            "the schema; answers are checked sim-side only")
    for err in o["errors"]:
      # A run that was scripted all along looks identical to a thoughtful one
      # from the outside unless this is printed.
      print(f"overseer fell back on  : {err}")
  if r.get("visitors"):
    v = r["visitors"]
    print(f"visitors               : {v['received']} message(s), "
          f"{v['delivered']} answered, {v['queued']} still waiting")
    if v["droppedInvalid"] or v["droppedFull"]:
      # Both are normal in small numbers and a story in large ones: garbage
      # is somebody probing, a full queue is the channel outrunning the robot.
      print(f"visitors dropped       : {v['droppedInvalid']} malformed, "
            f"{v['droppedFull']} overflowed the queue")
    for reply in r.get("replies", ()):
      print(f"visitor {reply['outcome']:<14s}: {reply['reply']}")
  print(f"sim / wall             : {r['sim_time']:.1f} s / {wall:.1f} s"
        f"  ({r['sim_time'] / wall:.2f}x real time)")
  if life.mode is not None and (life.paused_s or life.mode.mode != "llm"):
    # Said only when it is not the default: a run nobody touched should not
    # print a line about a switch nobody flipped.
    print(f"operator mode          : {life.mode.mode}"
          + (f", paused for {life.paused_s:.0f} s" if life.paused_s else ""))
    for note in life.mode.errors:
      print(f"mode file              : {note}")
  if pacer is not None:
    s = pacer.stats()
    print(f"pacing (target {s['rate']:.2f}x)  : drift {s['drift_s']:+.3f} s"
          f" at close, worst transient lag {s['max_lag_s']:.3f} s")
  print(f"frames sent / dropped  : {publisher.frames_sent}"
        f" / {publisher.frames_dropped}"
        f"  (connections: {publisher.connections})")
  if publisher.connections == 0 and publisher.last_error is not None:
    # A rejected token retries exactly like a down server; say which.
    print(f"never connected        : {publisher.last_error}")


if __name__ == "__main__":
  main()
