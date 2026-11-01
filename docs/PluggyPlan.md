# PluggyBot 🔌

**A simulated, hardware-honest robot, and the autonomous agent that lives in it.**

PluggyBot is a robot in [MuJoCo](https://mujoco.org/) whose parts are real,
purchasable parts, and whose day is decided by an LLM: it explores, earns
points at jobs the world offers, charges itself, and can die. **It is a
humanoid** (#507), our own body from real parts, on two legs with two
grippers; the served pair walk the house (#559's stage 4), charge by a
cable each plugs into itself, and take the jobs their hands do at the
storeroom's counter (a drawer, a cupboard's door, a chest's lid). **The
quadruped before them** (about 10 kg on four legs, #387), with #378's arm
taking tools off a rack, fed the mouse, played hide-and-seek, drew and
counted (#403, #404, #406, #407); #559's stage 5 deleted it (the tag
`quadruped-final` is the last commit that runs it), and its jobs come back
as tasks the hands learn (#560). **What the project works on now is
self-taught generality**: the robot acquiring new skills by its own effort
through five processes (reuse what it has, try and react, follow
instructions, copy, and imagine first). It chooses among them, and that
choice is measured (#487, #465). The wheeled rover that came before was
deleted in #376; the tag `rover-final` is the last commit that runs it. The
website side (`rooftop-media-2026`, "PluggyWorld") streams that world live
and lets a visitor talk to the robot.

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
   LLM tokens and human help kept apart (#465). The procedures it writes
   are routes to a skill, not the definition of one.
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
the loop (2, 3 and 4). 1 needs the five processes, a skill library that
keeps only what worked, and a record of what each attempt cost (#487; the
per-task record, Evaluation.md §3); 5 and 6 need memory and an economy with
slack in it and nothing that forces a charge. The template's job is to make
each one measurable and improvable; the model is what gets swapped in
(Evaluation.md §1: the instrument is fixed, the model is the variable). ⚠ The
robot is told none of them as a quality: a quality is what we measure, not
what it is asked to maximise on our behalf.

**What stays fixed.**

- **Hardware honesty.** Every part is a real part (Parts.md), every sensor is
  modelled with its blind spots, and nothing reaches the robot over the wire
  that a sensor would have to discover (TaskPattern.md §2). It began as a
  build plan; it is now also what keeps the research honest — a result
  counts only if the parts behind it exist.
- **An LLM in the loop, and the agent's own ML later.** Today the mind picks
  what to do; the direction is an agent that acquires skills by several
  processes and chooses among them, with its own simulation as one tool where
  trying is dear (#487). Training there on a reward it wrote comes last.
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
milestones 1–13; the pieces stand — the tasks, the economy, the
measurement — but as apparatus, not product. Visitors are witnesses, not customers; the deployed world is an
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
- **A cable, where only the plug's rigid body is handled.** Manipulating a
  deformable wire is one of the hardest problems in robotics, and PluggyBot
  never does it: the humanoid grips its charging plug's rigid body and
  pushes it into a funnel on its torso, where a magnetic catch seats it,
  while a tool balancer keeps the rope taut and draws the plug home into its
  cup (#559's stage 2) — contact-rich alignment, but tractable. In the sim
  the cable touches nothing: it is a line that pulls the plug home and stops
  it 0.6 m out (`activity/station.py`), and only the plug collides. (The
  quadruped's couplings were rigid: a gravity latch on its arm's fork, and a
  dock it lay down onto, #378.)
- **Decompose, don't end-to-end.** Each capability uses the cheapest adequate
  technique: supervised learning where labels are free, classical robotics
  where the problem is solved, RL where it earns its keep, and an LLM for the
  decisions none of those can make.

## Architecture

| Capability | Approach |
|---|---|
| Body interface | `body.py` (#380): everything the day loop and the procedure verbs may ask of a body, and the only way they reach one; the command a routine yields is the body's own; a `StubBody` carries the loop's bookkeeping in tests |
| Body | the humanoid (`humanoid/`, #507): our own body from real parts, RobStride joints with their drivers and two Dex1-1 grippers, its proportions Booster T2's; standing, walking and get-up policies trained in `training/` and run as numpy on the physics thread (`humanoid/policy.py`, `walk.py`, `getup.py`), its arms an IK reach on its model of itself (`humanoid/reach.py`, `arms.py`); `humanoid/body.py` is its `Body`, the pair served in the house since #559's stage 4 (`home_humanoid`). The quadruped before it (#387) was deleted in stage 5 (`quadruped-final`) |
| Ranging | 3D LIDAR (`perception/lidar.py`), a Livox Mid-360S upside down over the head, its rays from 1.47 m: 30 rings of 360 ray casts a frame (one `mj_multiRay`) with its sheet's noise, a blind zone and a self-filter. Its level ring is the scan the map is laid from; its points laid flat are measured, not yet served (#581). Stereo was measured and dropped (Parts.md "Vision & ranging") |
| Near field | RealSense D435-class depth camera on the head (`perception/depth.py`): batched ray casts, z² noise, the occlusion shadow, out-of-range as unknown. The planner plans round what it sees under the LIDAR's level ring (#387, `HumanoidMission._fold_low`); it also feeds a robot-centric 2.5D height map (`perception/heightmap.py`), streamed as `heightmap`, that nothing deciding reads (#34) |
| Vision | the head's cameras: the navigation camera for the places' signs (`humanoid/places.py`), the D435's colour imager for the tags its hands and its station work by (`Body.spot`, `perception/watch.py`, `humanoid/charge.py`); the tags are `perception/tags.py` |
| Odometry | legged odometry off its joints' drivers and an ICM-42688-P IMU read with their datasheets' noise, the planted foot told at 3 mm (`humanoid/odometry.py`, #386, #507), a fix off its station's tags, and a search by its LIDAR when the station is missing where the belief puts it (#476, #559); every LIDAR scan matched against the robot's own map before it is fused, and the matched pose is the belief (`mapping/scan_match.py`, #386) |
| Mapping & exploration | log-odds occupancy grid and frontier exploration (`mapping/`); the planner plans through floor it has not seen too, at a price, and finds its doors by finding walls (`mapping/optimistic.py`, #381's walking stage); places, not coordinates: a job names a place, and the robot finds its sign (#419) |
| Hands and charging | the hands' vocabulary (`docs/Hands.md`, #525): position and force moves closed on what a hand feels, its force read off the whole body (`humanoid/reach.py`, `arms.py`), and the feet's `step`; its jobs are the hands' at the storeroom's counter (#559's stage 3). It charges by a cable it plugs into itself at a station on the living room's wall (`activity/station.py`, `humanoid/charge.py`, #559's stage 2) |
| Behaviour arbitration | `HubLifecycle.run()`: with no mind, charge > queued errand > a claimed job > explore; with one the rails are off, the mind decides after the queue, and its event map decides when it is asked at all (Overseer.md, Evaluation.md §2) |
| Economy | task offers, code-side scoring, a points ledger with upkeep and hearts (TaskPattern.md, Overseer.md §8b) |
| Imagination | the robot's own world, one process of five (#487), postponed (#563): #466's -- a scene language compiled with the quadruped's body, its probe's record replayed there in a worker process -- is at `quadruped-final`; #466's mechanism, the drop-handle chest (`activity/chest.py`), is the humanoids' chest job |
| Measurement | two arms, `scripted` and the one mind, and the six qualities read as shapes off the deployed world's rows; capability's per-task record off a demo's attempts, each one try from a fresh set-out graded by the world (`evaluation/attempts.py`, `challenge/lid.py`, #493) (Evaluation.md) |

## Milestones

The milestones (July to September 2026) were flown on the plug robot and
then the rover, both deleted (#376; `rover-final`): a teleoperated base,
odometry, mapping and exploration, the battery loop, the modular tool hub
and its rack (the pen drew its square at 0.57 mm form error), tasks, minds
and money, the dressed world, measurement (A0:
4 of 5 days dead flat, the agent never treating energy as a constraint) and
an economy it can die in. M12, two robots, landed as #167, and M11, hands,
became the arm (#378, #405). The issues, `git log` and the docs are the
record; CLAUDE.md carries the constraints that still bind.

## Hardware

Nothing is ordered. The body is a humanoid, built first in simulation so
that each demo's body-specific parts are built once, for the body that
will use them (#507; Ben, 2026-10-13). It is drawn from real, purchasable
parts from its first draft (#487). #379's gate before any order stands, and
is read once the demos have results. Parts.md holds its bill of materials,
and `protocol/parts.json` is the site's parts page.

## Tooling

- **Simulation:** MuJoCo. The house and the body are MJCF written by Python
  generators (`home/world.py`, `humanoid/model.py`), and the served world is
  put together at load (`humanoid/world.py`).
- **CAD (later phase):** Onshape, exported to URDF/MJCF via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) once the design stabilizes
- **Learning:** `training/`, a uv project of its own (mjlab, #377): the
  humanoid's standing, walking and get-up policies, exported to `.npz` and
  run as numpy; it never enters the serving image.
- **Classical vision & robotics:** OpenCV, NumPy

## Where things are written down

- `CLAUDE.md` — every constraint that still binds, per subsystem, with the
  measurement behind it. The first thing an agent session reads.
- `SimNotes.md` — simulation lessons, in the order they were paid for.
- `Parts.md` — the real parts, and the sim parameters they feed.
- `ActivityPattern.md`, `TaskPattern.md`, `Challenges.md` — the build
  recipes: a mechanism that owns world state, a job offer, a job nobody
  scripted.
- `Overseer.md` — the mind: vocabulary, event map, memory, money, visitors.
- `Hands.md` — the humanoid's hands' vocabulary: its verbs, the force move,
  the step, the capability check.
- `Evaluation.md` — measurement: arms, metrics, the observatory.
- `Observatory.md` — the deployed world's periods: what was running when.
- `Testing.md` — pinning a rule without paying for a mission.
- `Webserver.md` and `protocol/README.md` — the stream, and its versioning.
- `rooftop-media-2026/docs/pluggyworld.md` — the website's design doc.
