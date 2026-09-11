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
2. #120 as a written decision, with one predicate-graded challenge: criteria
   first, a passing run, a failing run.
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
1. ✅ **Compute budget on the Pi 5 — measured (Aug 2026), and the hub pivot's
   bet pays off.** Profiled by separating costs that TRANSFER to hardware from
   simulation artefacts (rendering is sim-only — a real robot is handed images
   by its cameras). Desktop timings, scaled by a deliberately pessimistic 5×
   for a Cortex-A76:

   | stage | here | Pi 5 est | transfers? |
   |---|---|---|---|
   | AprilTag decode 1280×720 | 7.2 ms | 36 ms | yes |
   | stereo SGBM 640×480 | 12.1 ms | 60 ms | yes — **not paid today** |
   | occupancy grid update | ~~8.3 ms~~ 1.3 ms | ~~42 ms~~ ~6 ms | yes |
   | tag render / scanner | 2.1 ms | — | no (sim artefact) |

   **~138 ms per perception cycle → 7.3 Hz**, against a loop that looks for a
   tag every 0.3 s and drives at ≤0.25 m/s. So **the hub MVP does not need an
   accelerator** — YOLO is only in the plug-anywhere path, which is exactly
   what the pivot predicted, and that is ~€150 of Hailo HAT not spent.
   Caveats worth keeping honest: the 5× penalty is an estimate, not a
   measurement on real silicon; SGBM is untuned; all four Pi cores are
   available, so pipelining has headroom. The surprise was the **occupancy
   grid update costing as much as the tag decode** — a per-ray Python loop,
   and the cheapest thing on this list to optimise. Optimised (Aug 2026,
   issue #2): numpy-vectorized to 1.3 ms/scan, 7.4× (SimNotes).
2. ✅ **Third camera routing — closed (Aug 2026) by the LIDAR swap.** Dropping
   stereo frees a CSI port: nav camera + dock camera on the Pi's two ports,
   no multiplexer, LIDAR on USB/UART. A *blocking* item resolved as a side
   effect of a decision taken for entirely different reasons.
3. 🟡 **Sensor-realism pass — the ranging half is DONE (Aug 2026): stereo is
   gone, replaced by a 2D LIDAR + one camera.** Measuring it is what killed
   it: real SGBM on the sim's own stereo pair produced disparity for 49.7 % of
   the mapper's scan row at 593 mm median error, against a 50 mm grid cell
   (see Parts.md "Vision & ranging" and SimNotes). `perception/lidar.py` casts
   360 rays via `mj_ray` with ±10 mm + 1 % noise, 2 % dropout and a real
   self-filter; the hub mission and the full battery lifecycle both still
   close on it, collision-free. Side effects: blocking item #2 below is now
   closed, the Pi budget drops to ~78 ms/cycle, and `ELECTRONICS_W` rose
   6.0 → 8.5 W for the unit's 2.5 W.
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
6. ✅ **Veer with a tool aboard — re-measured, no re-tune needed (Aug 2026).**
   The y=+0.06 counterweight was tuned against a measured 26 cm veer over 4 m
   and predates carrying anything. Open-loop straight runs: bare −9.7 mm,
   LCD −13.6 mm, **pen module (182 g) −15.2 mm** over 2.69 m. The heaviest
   module costs 5.5 mm more than the bare robot — two orders off what
   motivated the counterweight. Tool mass is bounded by the coupling and the
   tip-load budget, not by veer.

**Hub-specific (cheap, physical)**
5. **Print-tolerance trial**: print the fork + one V-tray, measure the real
   capture envelope by hand against the sim's ±4 mm / <2°. PLA is fine for
   this (stiffer and more dimensionally accurate than PETG; the 150 g load
   is nowhere near creep). PETG only matters if the rack lives somewhere hot.
6. **Pogo-pin geometry trial**: contacts that engage on the same nose-in
   motion, recessed, dead-by-default with a hub-side handshake.
7. ✅ **Lean-pad — built in sim (Aug 2026)**, on `pluggybot_fork.xml`.
   **Re-scoped by measurement first**: its job is not damping sway, it is
   letting the tool exert force *at all*. A peg-hung module gave out at
   **0.1 N** of tool force (restoring and disturbing arms are both the peg's
   22 mm) and ran away past 0.25 N, flopping 51° at 0.5 N; a marker wants
   0.5–2 N. With the pad, 2 N holds the pen tip inside **0.26 mm** and the
   response is linear. Shape was bounded by the *parked* envelope (~60 mm
   between chassis top and the scanner row), not by the lever — see SimNotes.
   ⚠ **The power contacts do NOT belong on it** (this list said they did): the
   pad's gravity preload caps at 0.56 N and is realistically ~0.1 N, under
   what a pogo pin needs. The **peg in its V-notches** already carries
   0.43–0.47 N per plate, free, self-wiping, and already the one metal part
   in the design — that is where the module electrical interface goes.
8. ✅ **Module electrical interface — built in sim (Aug 2026)**. The peg is
   the connector: split into two conductors around an insulated centre, the
   fork's left and right V-notch pairs become the two poles of the power-only
   coupling. No extra parts and no extra alignment, because the alignment is
   the gravity latch that was already there. `module_power_state` reports each
   pole separately (a half-seated coupling is a real failure mode); a coupled
   module draws 0.6 W from the battery, gated on the electrical criterion
   rather than on "are we carrying it". Measured over a full errand:
   **0 brown-outs while carrying**, all interruptions confined to the
   mating/release transitions, worst 50 ms at release — which is the number
   that sizes the module's holding capacitor. Demo:
   `scripts/module_power.py` (`--view` to watch it live).
9. 🚧 **Drawing tool (pen carriage)** — groundwork done, controller next. The
   rack now has a **third bay** (bays at 0.125 / −0.125 / 0.375; the bay↔tag
   pairing is by index via `coupling.bay_tag_id`, replacing a hardcoded
   two-bay check that would have steered every bay-C swap onto bay B's
   marker). `module_pen` hangs there: a standard module frame plus a rail and
   a **carriage on its own actuated slide joint along the peg axis**
   (±55 mm), carrying a pen. That axis is the point — the base owns x/yaw,
   the lift owns z, the arm owns reach, and *nothing* owns lateral, because a
   differential drive cannot translate sideways. The module supplies the
   missing DoF and pairs with the lift to make an X-Y plotter against a
   vertical board. Verified: hangs, is picked, conducts through the coupling,
   drives its carriage end-to-end without shaking off the fork, and its sweep
   clears the robot. **Mass 182 g** (vs LCD 130 g, plug 156 g) — over the
   150 g soft budget, inside the 300 g the latch was validated to.
   **It draws.** `tools/drawing.py` assembles an X-Y plotter from the module's
   carriage (horizontal), the robot's lift (vertical), and the arm's reach
   through a sprung quill (pen pressure); the base stays parked, so nothing
   in a drawing is integrated from wheel odometry. Calibration is measured
   two-point per axis, then re-zeroed with the pen pressed. Measured end to
   end — fetch the tool from bay C, carry it to a board, plot a figure:
   **form error 0.79 mm (square) / 1.84 mm (circle)**, 94-99 % inked,
   tool still electrically seated afterwards. Error is reported decomposed:
   a rigid offset (a calibration constant, still ~16 mm on the square) versus
   FORM (is it the right shape). Much of what was first blamed on module yaw
   under drag turned out to be MuJoCo's regularized-friction creep — the
   `noslip` solver pass took square form error 2.14 -> 0.79 mm and ink
   63 % -> 94 %. Demo: `scripts/draw.py` (`--view`, `--shape square`).
   Still to build: a stiffer yaw constraint (or a measured-while-sweeping
   fit) for that residual, and the drawing surface in `room_hub` so the errand
   runs in the real room rather than the bare world.
10. 🚧 **Claw module (the fourth tool)** — grasp and lift verified, carrying
   open. A pendant straight down the peg axis, because the coupling takes
   ~0.45 N·m of pitch moment and reach costs `W × L`: 800 g on the axis is
   fine, 400 g at 150 mm out unseats the module. The chassis — the obvious
   worry — was never close (800 g costs 2.4 N of wheel load; tipping needs
   ~5 kg). The rack grew a fourth bay for it (rail now 1.36 m). Verified:
   fetched from bay D, powered through the coupling, aimed to 1.9 mm, gripped
   and lifted a 60 g block **99.6 mm** off the floor, module still seated.
   Demo: `scripts/pickup.py`. The full pick-carry-place now works: gripped,
   lifted **122 mm**, carried through a turn with zero dropouts, set down,
   module still seated. Getting there needed MuJoCo's `noslip` pass — a
   gripped object otherwise creeps out of the jaws at ~8 mm/s regardless of
   clamp force (see SimNotes; it was degrading the pen module too).
   The arm angles **55 mm forward** so the tool's own camera can see its grip
   point — 0.03 N·m of the 0.45 N·m budget. **`claw_eye` is the first camera
   on a TOOL rather than the chassis**, and costs no CSI port because module
   data already crosses the coupling wirelessly. **Open:** nothing can yet
   *find* a floor object autonomously — the LIDAR plane is 223 mm up and the
   nav camera is blind to the floor inside 0.48 m, so the approach is still
   driven from a known object pose.

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
- `ToolPattern.md`, `ActivityPattern.md`, `TaskPattern.md` — the three build
  recipes: a tool module, a mechanism that owns world state, a job offer.
- `Overseer.md` — the mind: vocabulary, event map, memory, money, visitors.
- `Evaluation.md` — measurement: arms, metrics, the harness, the results.
- `Webserver.md` and `protocol/README.md` — the stream, and its versioning.
- `rooftop-media-2026/docs/pluggyworld.md` — the website's design doc.
