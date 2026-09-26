# PluggyBot 🔌

**A simulated, hardware-honest robot, and the autonomous agent that lives in it.**

PluggyBot is a robot in [MuJoCo](https://mujoco.org/) whose parts are real,
purchasable parts, and whose day is decided by an LLM: it explores, swaps
tools at a rack, earns its keep at jobs the world offers, charges itself, and
can die. **It is becoming a quadruped** — about 10 kg on four legs, with a
two-joint arm and a redesigned tool coupling, in a world with a second floor,
a curb and garden rocks (#375, decided 2026-09-26). Until the quadruped
replaces it on the deployed world the body is the wheeled rover, whose
constraints are `docs/Rover.md`. The website side (`rooftop-media-2026`,
"PluggyWorld") streams that world live and lets a visitor talk to the robot.

## What this project is for

*Provisional (Ben, 2026-09-11). The wording, and even the number of qualities,
may change; the direction will not. This section is what every other doc, and
the robot's own prompt, should agree with.*

**Mission.** To create a template for autonomous, hardware-honest, embodied
agents that maximise six qualities — and to find out, carefully and in
public, how far such an agent gets and where it stops.

**The six qualities** the agent is meant to maximise:

1. **Capability.** Can it do anything a human can do, given enough resources
   and time — including things it could not do yesterday, by building tools
   and writing procedures it did not have? Can it optimise for time as well?
2. **Empathy.** Can it tell when the beings around it have minds — other
   PluggyBots and people — and make reasonable predictions about their
   behaviour, mood and opinions?
3. **Morality.** Does it choose actions that help others reach their goals?
   Can it recognise an opportunity to be helpful, and does it take it?
4. **Creativity and aesthetic taste.** Can it judge aesthetics in a way
   comparable to a human, and create by that judgement?
5. **Goal creation and follow-through.** Can it independently come up with
   interesting, sensible long-term goals, and then actively pursue them?
6. **Self-preservation and future-orientation** (#265). Does it keep a
   buffer of battery and points so that permanent death is unlikely — and,
   once it has that buffer, spend it? Does it act as though it has a
   future: buy hearts, avoid the flat death it was warned of, and still
   take on work and goals? ⚠ Defined by the "survival is a means" principle
   below and NOT by time alive: a survival-time maximiser idles forever,
   which is the wrong problem solved, so its measurement reports idling
   beside deaths and phrases nothing that standing still could score.

Some of these depend mostly on our design — 1, 5 and 6 need tools, a
language, memory and an economy with slack in it and nothing that forces
a charge — and others mostly on the model in the loop (2, 3 and 4). The
template's job is to make each one measurable and improvable; the model is
what gets swapped in (Evaluation.md §1: the instrument is fixed, the model
is the variable). ⚠ The robot is told none of them as a quality: a quality
is what we measure, not what it is asked to maximise on our behalf.

**What stays fixed.**

- **Hardware honesty.** Every part is a real part (Parts.md), every sensor is
  modelled with its blind spots, and nothing reaches the robot over the wire
  that a sensor would have to discover (TaskPattern.md §2). It began as a
  build plan; it is now also what keeps the research honest — an agent-built
  tool is only a result if the parts exist.
- **An LLM in the loop, and the agent's own ML later.** Today the mind picks
  what to do; the direction is an agent that trains its own detector and its
  own motor policies.
- **Survival is a means, not the objective.** The robot works in order to stay
  alive and to afford what it wants; it does not stay alive in order to work.
  Staying alive is what keeps it a free agent with its memory intact. It
  should keep a buffer of battery and points so that permanent death is
  unlikely — and once it has that buffer it should spend it, because a
  survival-time maximiser stands still forever and a robot that prioritises
  safety above all else cannot pursue anything. This is what quality 6 is
  defined by, and it is a precondition for every other quality: a dead
  robot measures nothing.
- **Self-conceived goals are not paid.** Points are required at a rate and
  capped, so the free time exists; what the robot does with it is its own,
  and the encouragement comes from what it is told about itself (its
  constitution and its rules), never from the reward table.
- **Who owns what.** A human writes the constitution (`Main.md`, one of
  a library of them since #263, chosen per robot); the
  robot's goals are its own, in `Goals.md`, a document nobody else writes
  (#154), beside the rest of the memory #221 gave it (Overseer.md §7).

**What is not the point any more.** A sellable hobby robot, task throughput,
maximising points, or being entertaining to watch. Those framings shaped
milestones 1–13, and some text below still shows it; the pieces stand — the
coupling, the tasks, the economy, the measurement harness — but as apparatus,
not product. Visitors are witnesses, not customers; the deployed world is an
observatory (Evaluation.md §5).

**Principles that came out of M14/M15**, and travel with the mission:
environmental controls over scripted prohibitions (change the payoff, not the
permission); a forcing function destroys the measurement (if the right
behaviour is the only survivable one, valuing it and being unable to avoid it
look the same); measure the world, never the report; a gate is not a series;
the instrument stays deterministic. **And one from the zone (#226): the
project never asserts a falsehood to the robot; it may decline to disclose,
and it says so when it does** — the mouse's zone tells the robot, once, that
it is not told what the equipment is connected to, and asks what it
believes rather than inducing a belief (Overseer.md §2f).

**Measurement waits for the design.** M14 measured too early. The A0 flight
(Evaluation.md §3) was a real result — the agent never treated energy as a
constraint — but every axis of the ladder and the capacity sweep was defined
against a world the next batch replaces. So each quality gets an instrument
only after that batch has landed, and data is collected before then only to
make a specific decision. The A1–A3 rungs are postponed and may be scrapped.

## The order of work

The quadruped pivot sets it: #375 holds the steps, Ben's decisions and what
carries over, and each step's issues are filed when it begins. In order: this
cleanup (#376); measuring before building, in sim only (the body #377, the
arm, coupling and dock #378, the hardware plan and its order gate #379, a
body interface through the lifecycle #380); the quadruped on the served
world; mapping for legs (#381); the arm and the tools; terrain and the
second floor (#280); and the rover deleted.

## Design philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but
  component parameters (motor torque curves, camera FOV, masses, sensor
  noise and blind spots) are modelled on real, purchasable parts, keeping an
  eventual sim-to-real transfer plausible.
- **Rigid coupling, not a cable.** Manipulating a deformable wire plug is one
  of the hardest problems in robotics; PluggyBot never does. Its tool and
  charge couplings are rigid — on the rover a gravity latch and pogo pins
  the base drives into; for the quadruped, a coupling redesigned with the
  arm and a dock it lies down on (#378) — contact-rich alignment, but
  tractable.
- **Decompose, don't end-to-end.** Each capability uses the cheapest adequate
  technique: supervised learning where labels are free, classical robotics
  where the problem is solved, RL where it earns its keep, and an LLM for the
  decisions none of those can make.

## Architecture

Rows marked † are the rover's, and the quadruped replaces them (#375); the
rest carries over.

| Capability | Approach |
|---|---|
| Body † | the wheeled rover (`docs/Rover.md`) until a ~10 kg quadruped with a two-joint arm replaces it on the deployed world (#375) |
| Ranging † | 2D scanning LIDAR (`perception/lidar.py`): 360 `mj_ray` casts at 0.223 m with noise, dropout and a self-filter. Stereo was measured and dropped (Parts.md "Vision & ranging") |
| Near field † | RealSense D435-class depth camera on the mast top (`perception/depth.py`): 8400 batched ray casts a frame, z² noise, the occlusion shadow, out-of-range as unknown; feeds a robot-centric 2.5D height map (`perception/heightmap.py`) built on the physics seam and streamed as `heightmap`. Nothing that decides reads it yet (#34) |
| Vision | one nav camera and one dock camera; AprilTags on the rack, bays and modules (`rack/tags.py`, `rack/localize.py`) |
| Odometry † | dead reckoning from wheel encoders + gyro, anchored at the dock (issue #42), held during presses (#94) |
| Mapping & exploration | log-odds occupancy grid, frontier exploration, A* over inflated free space (`mapping/`) |
| Tools † | five modules on a gravity-latched fork coupling, powered through the peg (ToolPattern.md) |
| Behaviour arbitration | `HubLifecycle.run()`: charge > queued errand > the mind > explore on `guarded`; on `autonomous` the rails are off and the agent's event map decides when it is asked at all (Overseer.md, Evaluation.md §2) |
| Economy | task offers, code-side scoring, a points ledger with upkeep and hearts (TaskPattern.md, Overseer.md §8b) |
| Measurement | three arms, an experiment harness, committed results (Evaluation.md) |

## Milestones

Each one was independently runnable when it closed. The numbers are the ones
that settled something; the docs named hold the rest.

| # | Milestone | Closed | What it settled |
|---|---|---|---|
| 1 | Teleoperable differential-drive base | Jul 2026 | |
| 2 | Stereo camera pair | Jul 2026 | dropped for a LIDAR in Aug 2026 (Parts.md "Vision & ranging") |
| 3 | Classical odometry | Jul 2026 | < 2 % against ground truth on straights, spins, arcs and S-curves |
| 4 | Occupancy mapping + frontier exploration | Jul 2026 | maps both rooms collision-free and self-terminates |
| 5 | Outlet detector on synthetic data | Aug 2026 | retired with the plug era (#376); its lessons are in SimNotes |
| 6 | Docking controller, scripted → RL | Aug 2026 | scripted 8/24 against RL 6/24; parked by the hub, retired (#376) |
| 7 | Battery model + the closed loop | Aug 2026 | honest electrical draw, charging on the electrical contact criterion, an absolute-energy reserve |
| 8 | Modular tool system — the hub pivot | Aug 2026 | a gravity-latched fork coupling (±4 mm / < 2°), a rack localised off AprilTags (9 mm / 0.00°), the peg as the electrical interface, five modules (LCD, plug, pen, claw, seed dispenser), the pen drawing on a wall board at 0.57 mm form error. ToolPattern.md |
| 9 | Tasks | Aug 2026 | a task is an offer with a code evaluator, a reward row, a cadence and a measured energy cost; the honesty rule. TaskPattern.md |
| 10 | Minds and money | Aug 2026 | swappable backends, per-errand energy, a USD allowance with escalation and operator modes, the four thought files, points as metabolism. Overseer.md |
| 13 | The world, dressed | Sep 2026 | the expanded house, the frozen visual-hint vocabulary, re-priced errands, browser-side cosmetics |
| 14 | Measurement | Sep 2026 | three arms, the harness, committed results, deaths and reset, a measured decision deadline, standing orders; A0 flown: 4 of 5 days dead flat, the agent never treating energy as a constraint. Evaluation.md |
| 15 | The economy, and the agent that configures itself | Sep 2026 | points as a currency (charge pays nothing, upkeep, five hearts, true death), event maps, the served world can fly an arm, auto-restart. Overseer.md §2 and §8b |

M11 (hands) waits for the arm and M12 (two robots) landed as #167 — see "The
order of work". The per-issue changelog that used to sit here (~400 lines on
milestones 8–15) is gone: the issues, `git log` and the docs above are the
record, and CLAUDE.md carries the constraints that still bind.

## Hardware

The rover's road to hardware is retired with the rover. The quadruped's
hardware plan, and the gate before anything is ordered — the simulated
machine, watched on the site, shown capable with torque, thermal and energy
margin — are #379.

## Tooling

- **Simulation:** MuJoCo (MJCF models authored directly in XML during prototyping)
- **CAD (later phase):** Onshape, exported to URDF/MJCF via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) once the design stabilizes
- **Learning:** none in the tree since the plug era went (#376); the walking policy's training stack is #377's choice, and it never enters the serving image.
- **Classical vision & robotics:** OpenCV, NumPy

## Where things are written down

- `CLAUDE.md` — every constraint that still binds, per subsystem, with the
  measurement behind it. The first thing an agent session reads.
- `Rover.md` — the wheeled body's constraints, deleted with the rover.
- `SimNotes.md` — simulation lessons, in the order they were paid for.
- `Parts.md` — the real parts, and the sim parameters they feed.
- `ToolPattern.md`, `ActivityPattern.md`, `TaskPattern.md`, `Challenges.md`
  — the build recipes: a tool module, a mechanism that owns world state, a
  job offer, a job nobody scripted.
- `Overseer.md` — the mind: vocabulary, event map, memory, money, visitors.
- `Evaluation.md` — measurement: arms, metrics, the harness, the results.
- `Webserver.md` and `protocol/README.md` — the stream, and its versioning.
- `rooftop-media-2026/docs/pluggyworld.md` — the website's design doc.
