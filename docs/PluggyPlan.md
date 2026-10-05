# PluggyBot 🔌

**A simulated, hardware-honest robot, and the autonomous agent that lives in it.**

PluggyBot is a robot in [MuJoCo](https://mujoco.org/) whose parts are real,
purchasable parts, and whose day is decided by an LLM: it explores, swaps
tools at a rack, earns its keep at jobs the world offers, charges itself, and
can die. **It is a quadruped**: about 10 kg on four legs, with #378's
two-joint arm on its back, which takes tools off a rack beside its dock
(#405). The served pair walk the house (#387) and feed the mouse for pay
(#403). **What the project works on now is self-taught generality**: the
robot practising new skills in a simulation it builds itself (#465). The
wheeled rover that came before was deleted in #376; the tag `rover-final`
is the last commit that runs it. The website side (`rooftop-media-2026`,
"PluggyWorld") streams that world live and lets a visitor talk to the robot.

## What this project is for

*Provisional (Ben, 2026-09-11). The wording, and even the number of qualities,
may change; the direction will not. This section is what every other doc, and
the robot's own prompt, should agree with.*

**Mission.** To create a template for autonomous, hardware-honest, embodied
agents that maximise six qualities — and to find out, carefully and in
public, how far such an agent gets and where it stops.

**The six qualities** the agent is meant to maximise:

1. **Capability.** Can it acquire arbitrary new skills by its own effort?
   Measured as learning efficiency: how much experience it takes to reach a
   threshold on a task new to it, with real attempts, imagination compute,
   LLM tokens and human help kept apart (#465). The tools it builds and the
   procedures it writes are routes to a skill, not the definition of one.
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

Some of these depend mostly on our design and others mostly on the model in
the loop (2, 3 and 4). 1 needs a way to practise (a simulation the robot
builds of what is in front of it) and a record of what each skill cost (#465,
Evaluation.md §3); 5 and 6 need memory and an economy with slack in it and
nothing that forces a charge. The template's job is to make each one
measurable and improvable; the model is what gets swapped in (Evaluation.md
§1: the instrument is fixed, the model is the variable). ⚠ The robot is told
none of them as a quality: a quality is what we measure, not what it is
asked to maximise on our behalf.

**What stays fixed.**

- **Hardware honesty.** Every part is a real part (Parts.md), every sensor is
  modelled with its blind spots, and nothing reaches the robot over the wire
  that a sensor would have to discover (TaskPattern.md §2). It began as a
  build plan; it is now also what keeps the research honest — an agent-built
  tool is only a result if the parts exist.
- **An LLM in the loop, and the agent's own ML later.** Today the mind picks
  what to do; the direction is an agent that practises a skill in its own
  simulation and trains it there (#465, #281).
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
milestones 1–13; the pieces stand — the coupling, the tasks, the economy,
the measurement — but as apparatus, not product. Visitors are witnesses, not customers; the deployed world is an
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

**Measurement waits for the design.** M14 measured too early: A0 (Evaluation.md
§3) was a real result, but the ladder and the capacity sweep were defined
against a world that was then replaced. So a quality gets an instrument once
the design it measures has landed, and data is collected before then only to
make a specific decision; the A1–A3 rungs are postponed and may be scrapped.

## The order of work

#465 holds it: its "The next stretch" is the live order, with Ben's
decisions and what carries over, and this doc keeps no second copy. #375,
the quadruped pivot before it, closed on 2026-10-04 with its final status.

## Design philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but
  component parameters (motor torque curves, camera FOV, masses, sensor
  noise and blind spots) are modelled on real, purchasable parts, keeping an
  eventual sim-to-real transfer plausible.
- **Rigid coupling, not a cable.** Manipulating a deformable wire plug is one
  of the hardest problems in robotics; PluggyBot never does. Its tool and
  charge couplings are rigid — a gravity latch on a two-joint arm's fork, and
  a dock it lies down onto, sprung pins pressing up into its belly (#378) —
  contact-rich alignment, but tractable.
- **Decompose, don't end-to-end.** Each capability uses the cheapest adequate
  technique: supervised learning where labels are free, classical robotics
  where the problem is solved, RL where it earns its keep, and an LLM for the
  decisions none of those can make.

## Architecture

| Capability | Approach |
|---|---|
| Body interface | `body.py` (#380): everything the day loop and the procedure verbs may ask of a body, and the only way they reach one; the command a routine yields is the body's own; a `StubBody` carries the loop's bookkeeping in tests |
| Body | a ~10 kg quadruped (`legs/`, #377): twelve GIM8108-8 joints with their GDS68 drivers, a walking policy trained in `training/` and run as numpy on the physics thread (`legs/policy.py`), a rest by reflex and a get-up from a fall (`legs/posture.py`), and #378's two-joint arm on its back (`legs/arm.py`, #405); `legs/body.py` is its `Body` (#387) |
| Ranging | 2D scanning LIDAR (`perception/lidar.py`), RPLIDAR C1-class, 0.51 m up: 360 ray casts (one `mj_multiRay`) with noise, dropout and a self-filter. Stereo was measured and dropped (Parts.md "Vision & ranging") |
| Near field | RealSense D435-class depth camera on the nose (`perception/depth.py`): batched ray casts, z² noise, the occlusion shadow, out-of-range as unknown. The planner plans round what it sees under the LIDAR's plane (#387); it also feeds a robot-centric 2.5D height map (`perception/heightmap.py`), streamed as `heightmap`, that nothing deciding reads (#34) |
| Vision | the nose camera; AprilTags on the dock's board, the rack's bays and the places' signs (`rack/tags.py`, `legs/dock.py`, `legs/rack.py`, `legs/places.py`) |
| Odometry | legged odometry off the joints' CAN fields and an ICM-42688-P IMU read with their datasheets' noise (`legs/odometry.py`, #386), a zero-rate update at rest (#425), a fix off the dock's board (#378); every LIDAR scan matched against the robot's own map before it is fused, and the matched pose is the belief (`mapping/scan_match.py`, #386) |
| Mapping & exploration | log-odds occupancy grid and frontier exploration (`mapping/`); the planner plans through floor it has not seen too, at a price, and finds its doors by finding walls (`mapping/optimistic.py`, #381's walking stage); places, not coordinates: a job names a place, and the robot finds its sign (#419) |
| Tools | three modules on the rack beside the dock (`legs/rack.py`), taken and hung back by the arm's fork (`legs/swap.py`, #405), powered through the peg. The pen draws on the whiteboards lying in front of them, each found by its tags and its face by touch (`legs/draw.py`, `tools/drawing.py`, #406); the claw takes the cubes it finds by the D435's colour imager, lying (`legs/claw.py`, `tools/claw.py`, #407), and the LCD shows the census's count (`legs/survey.py`) |
| Behaviour arbitration | `HubLifecycle.run()`: with no mind, charge > queued errand > a claimed job > explore; with one the rails are off, the mind decides after the queue, and its event map decides when it is asked at all (Overseer.md, Evaluation.md §2) |
| Economy | task offers, code-side scoring, a points ledger with upkeep and hearts (TaskPattern.md, Overseer.md §8b) |
| Measurement | two arms, `scripted` and the one mind, and the six qualities read as shapes off the deployed world's rows; capability's per-task record is defined ahead of its rows (Evaluation.md) |

## Milestones

Milestones 1–15 (July to September 2026) were flown on the plug robot and
then the rover, both deleted (#376; `rover-final`): a teleoperated base,
odometry, mapping and exploration, the battery loop, the modular tool hub
and its rack, tasks, minds and money, the dressed world, measurement (A0:
4 of 5 days dead flat, the agent never treating energy as a constraint) and
an economy it can die in. M12, two robots, landed as #167, and M11, hands,
became the arm (#378, #405). The issues, `git log` and the docs are the
record; CLAUDE.md carries the constraints that still bind.

## Hardware

Nothing is ordered. The next body leans humanoid, and no body work starts
until demo 2 of #465 passes its bar; #379's gate before any order stands.
Parts.md holds the quadruped's bill of materials.

## Tooling

- **Simulation:** MuJoCo. The house and the body are MJCF written by Python
  generators (`home/world.py`, `legs/model.py`), and the served world is put
  together at load (`legs/world.py`).
- **CAD (later phase):** Onshape, exported to URDF/MJCF via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) once the design stabilizes
- **Learning:** `training/`, a uv project of its own (mjlab, #377): the
  walking policies and the get-up, exported to `.npz` and run as numpy; it
  never enters the serving image.
- **Classical vision & robotics:** OpenCV, NumPy

## Where things are written down

- `CLAUDE.md` — every constraint that still binds, per subsystem, with the
  measurement behind it. The first thing an agent session reads.
- `SimNotes.md` — simulation lessons, in the order they were paid for.
- `Parts.md` — the real parts, and the sim parameters they feed.
- `ToolPattern.md`, `ActivityPattern.md`, `TaskPattern.md`, `Challenges.md`
  — the build recipes: a tool module, a mechanism that owns world state, a
  job offer, a job nobody scripted.
- `Overseer.md` — the mind: vocabulary, event map, memory, money, visitors.
- `Evaluation.md` — measurement: arms, metrics, the observatory.
- `Observatory.md` — the deployed world's periods: what was running when.
- `Testing.md` — pinning a rule without paying for a mission.
- `Webserver.md` and `protocol/README.md` — the stream, and its versioning.
- `rooftop-media-2026/docs/pluggyworld.md` — the website's design doc.
