# PluggyBot 🔌

**A simulated, hardware-honest robot, and the autonomous agent that lives in it.**

PluggyBot is a small wheeled robot in [MuJoCo](https://mujoco.org/) whose parts
are real, purchasable parts, and whose day is decided by an LLM: it explores,
swaps tools at a rack, earns its keep at jobs the world offers, charges itself,
and can die. The website side (`rooftop-media-2026`, "PluggyWorld") streams
that world live and lets a visitor talk to the robot.

## What this project is for

*Provisional (Ben, 2026-09-11). The wording, and even the number of qualities,
may change; the direction will not. This section is what every other doc, and
the robot's own prompt, should agree with.*

**Mission.** To create a template for autonomous, hardware-honest, embodied
agents that maximise five qualities — and to find out, carefully and in
public, how far such an agent gets and where it stops.

**The five qualities** the agent is meant to maximise:

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

Some of these depend mostly on our design — 1 and 5 need tools, a language,
memory and an economy with slack in it — and others mostly on the model in
the loop (2, 3 and 4). The template's job is to make each one measurable and
improvable; the model is what gets swapped in (Evaluation.md §1: the
instrument is fixed, the model is the variable).

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
  safety above all else cannot pursue anything.
- **Self-conceived goals are not paid.** Points are required at a rate and
  capped, so the free time exists; what the robot does with it is its own,
  and the encouragement comes from what it is told about itself (its
  constitution and its rules), never from the reward table.
- **Who owns what.** A human writes the constitution (`Main.md`, and for now
  the starting goals in `Goals.md`); the robot's goals are its own — today
  they live in `Knowledge_and_Opinions.md`, and a robot-owned goals file is
  the next step (#154).

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
the instrument stays deterministic.

**Measurement waits for the design.** M14 measured too early. The A0 flight
(Evaluation.md §3) was a real result — the agent never treated energy as a
constraint — but every axis of the ladder and the capacity sweep was defined
against a world the next batch replaces. So each quality gets an instrument
only after that batch has landed, and data is collected before then only to
make a specific decision. The A1–A3 rungs are postponed and may be scrapped.

## The next batch

The order, agreed 2026-09-11; not yet issues except where numbered.

1. Docs and prompts aligned to this section, and a per-doc slimming pass
   (#153); the constitution / robot-owned goals split (#154).
2. ✅ #120 as a written decision, with one predicate-graded challenge: criteria
   first, a passing run, a failing run — `Challenges.md`, the three-block
   tower (`challenge/stack.py`).
3. **Agent-written procedures.** Rung one is #58 — composable errands over
   the guarded primitives, with a test that only that vocabulary writes
   `data.ctrl`. Rung two is a small, total procedure language: conditionals
   and bounded loops over sensed scalars, arithmetic, a step budget and a
   sim-time budget, per-step verdicts, abort meaning stow. Not sandboxed
   Python; callable from event-map rows and standing orders.
4. **Novel tasks.** A curated challenge set with no scripted solution, chosen
   to need no new sensing, rewarded generously; one two-tool job resolving to
   one verdict.
5. **Agent-built tools, modules first.** A catalog of real parts as data, the
   generator in `rack/coupling.py` as the emitter, ToolPattern.md's envelope as
   the validator, an `MjSpec` recompile spike, one agent-built tool in the
   rig, then a fabrication cost model. Body redesign waits.
6. **Near-field 3D.** The sensor decision (#34), a robot-centric 2.5D height
   map, then floor-object challenges and the ramp.
7. A new baseline, with instruments derived from the five qualities (#155
   designs them now and flies nothing).

Deferred behind it: M11 (hands: tier-1 tagged objects) and M12 (two robots —
which quality 2 needs, so it is deferred, not dropped).

## Design philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but
  component parameters (motor torque curves, camera FOV, masses, sensor
  noise and blind spots) are modelled on real, purchasable parts, keeping an
  eventual sim-to-real transfer plausible.
- **Rigid coupling, not a cable.** Manipulating a deformable wire plug is one
  of the hardest problems in robotics; PluggyBot never does. The wall plug is
  a rigid module on the arm, and the hub's tool and charge couplings are a
  gravity latch and pogo pins the base drives into — contact-rich alignment,
  but tractable.
- **Decompose, don't end-to-end.** Each capability uses the cheapest adequate
  technique: supervised learning where labels are free, classical robotics
  where the problem is solved, RL where it earns its keep, and an LLM for the
  decisions none of those can make.

## Architecture

| Capability | Approach |
|---|---|
| Ranging | 2D scanning LIDAR (`perception/lidar.py`): 360 `mj_ray` casts at 0.223 m with noise, dropout and a self-filter. Stereo was measured and dropped (Parts.md "Vision & ranging") |
| Vision | one nav camera and one dock camera; AprilTags on the rack, bays and modules (`rack/tags.py`, `rack/localize.py`); a YOLO outlet detector on the plug-era path |
| Odometry | dead reckoning from wheel encoders + gyro, anchored at the dock (issue #42), held during presses (#94) |
| Mapping & exploration | log-odds occupancy grid, frontier exploration, A* over inflated free space (`mapping/`) |
| Tools | five modules on a gravity-latched fork coupling, powered through the peg (ToolPattern.md) |
| Behaviour arbitration | `HubLifecycle.run()`: charge > queued errand > the mind > explore on `guarded`; on `autonomous` the rails are off and the agent's event map decides when it is asked at all (Overseer.md, Evaluation.md §2) |
| Economy | task offers, code-side scoring, a points ledger with upkeep and hearts (TaskPattern.md, Overseer.md §8b) |
| Measurement | three arms, an experiment harness, committed results (Evaluation.md) |

## Milestones

Each one was independently runnable when it closed. The numbers are the ones
that settled something; the docs named hold the rest.

| # | Milestone | Closed | What it settled |
|---|---|---|---|
| 1 | Teleoperable differential-drive base | Jul 2026 | |
| 2 | Stereo camera pair | Jul 2026 | later dropped (Aug 2026): real SGBM on the sim's own pair gave disparity on 49.7 % of the scan row at 593 mm median error, against a 50 mm cell |
| 3 | Classical odometry | Jul 2026 | < 2 % against ground truth on straights, spins, arcs and S-curves |
| 4 | Occupancy mapping + frontier exploration | Jul 2026 | maps both rooms collision-free and self-terminates |
| 5 | Outlet detector on synthetic data | Aug 2026 | YOLO11n on 1200 domain-randomised renders; 3/3 on a room it never saw, no false positive on the decoy switch |
| 6 | Docking controller, scripted → RL | Aug 2026 | scripted 8/24, RL 6/24, failures complementary (union 13/24); four measured design findings. Parked when the hub superseded wall docking |
| 7 | Battery model + the closed loop | Aug 2026 | honest electrical draw, charging on the electrical contact criterion, an absolute-energy reserve; the plug era's "repo MVP" |
| 8 | Modular tool system — the hub pivot | Aug 2026 | a gravity-latched fork coupling (±4 mm / < 2°), a rack localised off AprilTags (9 mm / 0.00°), the peg as the electrical interface, five modules (LCD, plug, pen, claw, seed dispenser), the pen drawing on a wall board at 0.57 mm form error. ToolPattern.md |
| 9 | Tasks | Aug 2026 | a task is an offer with a code evaluator, a reward row, a cadence and a measured energy cost; the honesty rule. TaskPattern.md |
| 10 | Minds and money | Aug 2026 | swappable backends, per-errand energy, a USD allowance with escalation and operator modes, the four thought files, points as metabolism. Overseer.md |
| 13 | The world, dressed | Sep 2026 | the expanded house, the frozen visual-hint vocabulary, re-priced errands, browser-side cosmetics |
| 14 | Measurement | Sep 2026 | three arms, the harness, committed results, deaths and reset, a measured decision deadline, standing orders; A0 flown: 4 of 5 days dead flat, the agent never treating energy as a constraint. Evaluation.md |
| 15 | The economy, and the agent that configures itself | Sep 2026 | points as a currency (charge pays nothing, upkeep, five hearts, true death), event maps, the served world can fly an arm, auto-restart. Overseer.md §2 and §8b |

M11 (hands) and M12 (two robots) are deferred, not dropped — see "The next
batch". The per-issue changelog that used to sit here (~400 lines on
milestones 8–15) is gone: the issues, `git log` and the docs above are the
record, and CLAUDE.md carries the constraints that still bind.

## Road to hardware (open items, Aug 2026)

*Kept as the build plan. It is no longer the milestone that orders the
work — see "What this project is for" — but hardware-honesty is, and this
is where its open decisions live.*

The hardware MVP bar is **a physical robot swapping plug ↔ LCD at a real
hub**. What stands between here and ordering parts:

**Blocking**

1. ✅ **Compute budget on the Pi 5 — measured, and the hub pivot's bet pays
   off.** Desktop timings for the stages that TRANSFER to hardware (rendering
   is a sim artefact — a real robot is handed images by its cameras), scaled
   by a deliberately pessimistic 5× for a Cortex-A76, came to **~138 ms per
   perception cycle → 7.3 Hz**, against a loop that looks for a tag every
   0.3 s and drives at ≤ 0.25 m/s. So **the hub MVP needs no accelerator** —
   YOLO is only in the plug-anywhere path, exactly what the pivot predicted,
   and that is ~€150 of Hailo HAT not spent. The 5× is an estimate, not a
   measurement on silicon. Two things fell out: the occupancy grid update
   cost as much as the tag decode until it was vectorized to 1.3 ms per scan
   (issue #2, SimNotes), and dropping stereo (item 3) took the budget to
   ~78 ms.
2. ✅ **Third camera routing — closed by the LIDAR swap.** Nav camera and dock
   camera on the Pi's two CSI ports, no multiplexer, LIDAR on USB/UART. A
   *blocking* item resolved as a side effect of a decision taken for entirely
   different reasons.
3. 🟡 **Sensor-realism pass — the ranging half is done.** Stereo is gone,
   replaced by a 2D LIDAR plus one camera, and measuring it is what killed it
   (Parts.md "Vision & ranging", SimNotes); `ELECTRONICS_W` rose 6.0 → 8.5 W
   to pay for the unit's 2.5 W.
   **Still open on this item:** gyro bias/drift and encoder quantization
   (odometry is currently perfect-encoder), and camera realism for the tag
   path — rendered tag images are noise-free, perfectly focused and perfectly
   exposed, so the measured 4.5 m decode range will shrink under motion blur
   and real optics.
4. **The plotter's calibration has no hardware equivalent** *(found Aug 2026)*.
   `drawing.calibrate()` reads the pen tip from `site_xpos` — ground truth.
   A real robot has no pen-tip sensor, so the same two-point procedure needs
   either a physical calibration jig or the dock camera watching the pen
   against a fiducial. This is a *design* decision that changes what the pen
   module needs, so it should be settled before the module is printed.
5. **Mass re-budget** once the pack is chosen (~1.14 → ~1.54 kg invalidates
   every physics threshold derived from the current model). Do this LAST,
   after the other hardware choices settle.
6. ✅ **Veer with a tool aboard — re-measured, no re-tune needed.** Open-loop
   straight runs over 2.69 m: bare −9.7 mm, LCD −13.6 mm, and the heaviest
   module (the 182 g pen) −15.2 mm — **5.5 mm more than bare**, two orders
   off the measured 26 cm veer over 4 m that motivated the y=+0.06
   counterweight. Tool mass is bounded by the coupling and the tip-load
   budget, not by veer.

**Hub-specific (cheap, physical)**

7. **Print-tolerance trial**: print the fork + one V-tray, measure the real
   capture envelope by hand against the sim's ±4 mm / <2°. PLA is fine for
   this (stiffer and more dimensionally accurate than PETG; the 150 g load
   is nowhere near creep). PETG only matters if the rack lives somewhere hot.
8. **Pogo-pin geometry trial**: contacts that engage on the same nose-in
   motion, recessed, dead-by-default with a hub-side handshake.
9. ✅ **Lean-pad — built in sim** on `pluggybot_fork.xml`, and re-scoped by
   measurement: its job is not damping sway, it is letting the tool exert
   force *at all*. A peg-hung module gave out at **0.1 N**; with the pad, 2 N
   holds the pen tip inside **0.26 mm**, linearly. ⚠ The power contacts do
   NOT belong on it — its gravity preload caps at 0.56 N, under what a pogo
   pin needs. ToolPattern.md ("the force budget"), SimNotes.
10. ✅ **Module electrical interface — the peg IS the connector.** Split into
    two conductors around an insulated centre, the fork's left and right
    V-notch pairs become the two poles of a power-only coupling: no extra
    parts and no extra alignment, because the alignment is the gravity latch
    that was already there. `module_power_state` reports each pole separately,
    since a half-seated coupling is a real failure mode. There are no
    brown-outs while carrying — every interruption is a mating or release
    transition, worst measured **178 ms** under hard driving, which is what
    sizes the module's holding capacitor at ~200 ms (Parts.md, SimNotes).
    Demo: `scripts/module_power.py`.
11. ✅ **Drawing tool (pen carriage)** — built, and it draws. The module
    supplies the lateral DoF nothing else owns (a carriage on its own
    actuated slide joint along the peg axis, ±55 mm) and pairs with the lift
    to make an X-Y plotter against a vertical board, base parked so nothing
    in a drawing is integrated from wheel odometry. **0.57 mm form error** on
    the square. ToolPattern.md; demo `scripts/draw.py`.
12. ✅ **Claw module** — full pick-carry-place verified. A pendant straight
    down the peg axis, because the coupling takes ~0.45 N·m of pitch moment
    and reach costs `W × L`; the arm angles 55 mm forward so `claw_eye` — the
    first camera on a TOOL rather than the chassis — can see its grip point.
    ToolPattern.md; demo `scripts/pickup.py`. **Open:** nothing can yet
    *find* a floor object autonomously, so a grasp still runs from a
    memorised pose — the perception ladder, TaskPattern.md §3.

**Then**: the Parts.md open decisions (plug body diameter, specific 3S pack,
igus stroke quote, chassis material, motor brackets).

## Tooling

- **Simulation:** MuJoCo (MJCF models authored directly in XML during prototyping)
- **CAD (later phase):** Onshape, exported to URDF/MJCF via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) once the design stabilizes
- **Learning:** PyTorch, Gymnasium, Stable-Baselines3 (RL); Ultralytics/torchvision (detection)
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
- `Evaluation.md` — measurement: arms, metrics, the harness, the results.
- `Webserver.md` and `protocol/README.md` — the stream, and its versioning.
- `rooftop-media-2026/docs/pluggyworld.md` — the website's design doc.
