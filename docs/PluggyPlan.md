# PluggyBot 🔌

**A simulated self-charging robot that explores, maps, and plugs itself in.**

PluggyBot is a personal robotics project built in physics simulation ([MuJoCo](https://mujoco.org/)), with the long-term goal of a design faithful enough to real hardware components that it could eventually be built physically.

The core idea: a small wheeled robot that explores its environment to build and maintain a spatial map, visually recognizes what it needs, and - when its (simulated) battery runs low - takes itself off to charge. Since the milestone-8 hub pivot the primary charge path is a purpose-built tool rack rather than a wall outlet, and the robot swaps its own tools there.

## Design Philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but component parameters (motor torque curves, camera baseline and FOV, masses) are modeled on real, purchasable parts, keeping an eventual sim-to-real transfer plausible.
- **Rigid plug, not a cable.** Manipulating a deformable wire plug is one of the hardest problems in robotics. PluggyBot sidesteps it: the plug is fixed to the end of a rigid arm. Docking is still a genuinely hard contact-rich alignment task - but a tractable one.
- **Decompose, don't end-to-end.** Rather than one giant RL policy from pixels to behavior, each capability uses the cheapest adequate technique: supervised learning where labels are free (simulation gives them away), classical robotics where the problem is solved, and RL where it actually earns its keep.

## Core Goals

1. **Mobility** - a differential-drive wheeled base.
2. ~~**Binocular vision**~~ → **Ranging + vision**: a 2D scanning LIDAR for range, one camera for seeing. Stereo was dropped in Aug 2026 after measuring that real SGBM on this sim's own stereo pair produced disparity for only 49.7 % of the mapper's scan row at 593 mm median error, against a 50 mm grid cell. See Parts.md "Vision & ranging".
3. **Spatial memory & exploration** - an internal map of the environment, plus a drive to explore: validating what it remembers and expanding into the unknown (active SLAM / frontier exploration).
4. **Outlet recognition** - visually detecting wall outlets with a CNN trained on domain-randomized synthetic images from the simulator.
5. **Self-docking** - plugging into a detected outlet using the rigid arm.
6. **Battery-driven behavior** - a simulated battery that drains with motor use; when low, PluggyBot seeks an outlet and charges. This closes the loop that gives the project its name.

## Stretch Goals

- **Recognizing beings** - visually detecting and distinguishing humans (or other creatures), and a drive to do so.
- **Playing a physical game** - e.g. checkers. A major manipulation and perception undertaking in its own right; explicitly out of scope until the core loop works.

## Architecture

| Capability | Approach |
|---|---|
| Outlet detection | Supervised CNN, fine-tuned on synthetic labeled images rendered from the sim with domain randomization |
| Ranging | 2D scanning LIDAR (`perception/lidar.py`, `mj_ray` with noise + a self-filter). Was a ground-truth depth buffer read as a laser scan; stereo was measured and found unable to produce that scan at all |
| Odometry | Classical dead reckoning from wheel encoders + IMU (EKF fusion); learned regression demoted to a later experiment |
| Mapping & exploration | Occupancy grid + frontier-based exploration (classical baseline); learned/curiosity-driven exploration as a later experiment |
| Docking | RL policy (or scripted visual servoing baseline) over relative outlet pose + arm state; contact-rich, dense-rewardable. Arm architecture inspired by Hello Robot Stretch: the base owns x/yaw, a prismatic lift owns z, the arm owns reach — three nearly-independent 1D alignment problems. Caveat from sim experience: a mast raises the center of mass and pitch inertia, so the lift design must come with a wider/heavier base (see SimNotes.md) |
| Behavior arbitration | Finite-state machine over the modules, driven by battery level and map state |

## Milestones

1. ✅ Teleoperable differential-drive base in MuJoCo *(July 2026)*
2. ✅ Stereo camera pair rendering from the robot *(July 2026)*
3. ✅ Classical odometry — dead reckoning from wheel angles, verified <2 % against ground truth on straights, spins, arcs, and S-curves *(July 2026; IMU/EKF fusion deferred until drift actually hurts)*
4. ✅ Occupancy mapping + frontier exploration — log-odds grid from a virtual laser scan (depth-image center row), gyro-fused odometry, A* over inflated free space, autonomous frontier exploration with look-around spins; maps both rooms collision-free and self-terminates *(July 2026)*
5. ✅ Outlet detector trained on synthetic data — YOLO11n on 1200 domain-randomized renders with free segmentation-derived labels; generalizes to `room_1.xml` outlets it was never trained on (3/3 detected at 0.94–0.97 confidence, no false positive on the decoy switch). Detections project to world coordinates via depth and merge into a landmark map; a full mission parks at the docking hand-off pose within **0.5° yaw / 1.3 cm lateral** *(August 2026)*
6. ✅ Docking controller (scripted baseline → RL) — the mechanical stack docks deterministically from a good standoff (pytest), and both controllers are now scored by one protocol: `eval_docking.py` runs identical seeded trials in room_1 with the real YOLO (pose jitter ±2 cm/±1°, landmark error ±2 cm xy / ±1.5 cm z). **Scripted: 8/24 = 33.3 %. RL (SAC over `envs/DockEnv`, 70.8 % in-env): 6/24 = 25 %** — but the failures are *complementary* (union 13/24): RL wins 3/9 at the high outlet C where scripted manages 1/9 (the predicted vision-z gap, confirmed fixed by learning) and docks in 4–12 s vs scripted's 9–19 s, yet goes 0/8 at outlet A, parked by a residual sensor-model gap (an out-of-distribution hover with the arm extended). The real yield of the RL work was **four measured design findings**: the feeler straddle trap, the arm's self-occlusion of the dock camera, odometry corruption under wall-grinding (all fixed, pytest-guarded), and SAC's late-training instability on this task. Wall-outlet docking parks here — milestone 8's hub supersedes it as the primary charge path, and this becomes the "plug-anywhere module" backlog *(August 2026)*
7. ✅ Battery model + the closed loop — `pluggybot/power.py` models honest electrical draw (torque-proportional motor current from the Pololu datasheet, 6 W electronics, steppers only while moving, ~1C charging) against a deliberately scaled demo capacity; the lifecycle's CHARGE state charges on the electrical contact criterion (source-agnostic — the hub plugs into the same seam), presses the plug home while charging (both halves measured necessary), undocks, and resumes or finishes. Reserve is **absolute energy** (~0.4 Wh: a failed dock attempt + redrive), not a pack fraction — a fraction starved a small pack mid-insertion. Verified end-to-end: the full mission explored both rooms, benched a jammed outlet, **caught and erased a phantom decoy landmark at close range**, docked at outlet B, charged 21 → 90 % in 82 s of continuous contact, undocked, and ended recharged with zero chassis contacts. Battery-dead is an honest failure mode (and was reached honestly several times en route). *Milestones 1–7 together are the "repo MVP": documented, portfolio-ready.* *(August 2026)*
8. 🚧 **Modular tool system — the hub pivot** *(started August 2026)*. The coupling spike (`scripts/hub_spike.py`) measured the fork-and-peg gravity latch at ±4 mm lateral / <2° yaw / retention beyond the base's traction limit; the ROBOT swaps tools (`pluggybot_fork.xml` mounts the fork on the RCC — and drops the alignment feelers, deprecated per the remove-not-design-around rule; see Parts.md); and the hub is now the **unified rack** designed with Ben: one freestanding open-frame structure leaning on the wall (free body in sim — a swap cycle moves it <1 mm, and the wall braces exist because without them a sustained press measurably scooted it), modules hanging business-end-inward so a carried plug faces the driving direction, a charge bay with bumper-height pogo pins usable whatever the fork carries, and AprilTag placeholder plates (rack pose / bay identity / module identity). `scripts/hub_swap.py` runs full pick-carry-rehang cycles on both bays plus the charge press, tolerant to hand-off jitter. **The rack now lives in a room**: `models/room_hub.xml` is room_1's exact floor plan (shared via a scenery include, so the plug-era worlds stay bit-identical) plus the fork robot and the rack on the north wall, and `scripts/hub_mission.py` runs the whole story — map the room, A* to the rack, fine-align on the fiducial plates with the dock camera, pick the LCD module, carry it across the room, come back, hang it up — collision-free in ~84 sim-seconds — and it **finds the rack by looking at it**: the tag plates are discovered during navigation, projected to world coordinates and confirmed by sighting count exactly as outlet landmarks are, giving a rack pose **9 mm / 0.00°** from truth. The stored placement is now only the boot-time prior (what a robot that started docked knows), overridden by observation — a test starts the robot 30 cm wrong to prove the correction. The terminal approach ranges off the tag rather than odometry, which had drifted ~20 mm by the return leg. And the **hub-era lifecycle** now closes the loop (`scripts/hub_lifecycle.py`): a battery-driven arbitration loop — charge > errand > explore — in which PluggyBot seeds its map, localizes the rack from its fiducial, fetches the LCD module, carries it across the room, stows it, notices the battery at 21 %, noses into the charge bay until the pogo pins connect, charges to 90 %, and gets back to work. One run: 2 swaps, 1 charge cycle, module stowed, **0 collisions**, 158 sim-seconds. Charging is confirmed by the same electrical criterion the plug uses, so it works whatever the fork is carrying. Perception is now **real AprilTags** (tag36h11, generated with moms-apriltag and decoded with pupil-apriltags): the rack's 120 mm marker decodes past 4.5 m with millimetre-accurate PnP range, every reading is keyed to a decoded ID rather than guessed from blob geometry, and dropping the depth buffer made a full mission *faster*. Next: the module electrical model, the lean-pad that stops a carried tool swaying, and a motorised tool module. The arm tip becomes a tool interface; a "tool hub" shelf stores swappable modules (first two: the Schuko plug, an LCD screen), and the robot autonomously exchanges them. The hub also charges the robot through a purpose-built low-force connector — which resolves the measured hardware blocker head-on: a real Schuko socket's sprung contacts need tens of newtons, and the robot's measured push budget is ~3 N. "Plug into any wall outlet" survives as one module (the flagship demo), no longer the load-bearing architecture. Design constraints already known from measurement: no wrist, so latch verbs are slide-in + lift/lower (kinematic mount / gravity hook, not bayonet); power-only coupling with wireless data (a microcontroller per module) keeps the mating interface dumb and tolerant; a per-module tip-mass cap protects the veer counterweight calibration. De-risk order, per house method: standalone coupling spike (schuko_spike-style MJCF, no robot) → hub + modules in sim → autonomous swap in the lifecycle. **The hardware MVP / parts-ordering trigger: a physical robot swapping plug ↔ LCD at a real hub** — extensible from then on without structural change.

Each milestone is independently runnable and demoable.

**Parallel de-risk track** (can start anytime): a standalone Schuko plug/socket contact prototype — no robot, just a scripted insertion in its own MJCF — to validate contact modeling before milestone 6 depends on it. The Schuko recess is a natural alignment funnel; find out early how much of one.

**Later experiments** (off the critical path): learned odometry (competing against the classical baseline), curiosity-driven exploration.

## Road to hardware (open items, Aug 2026)

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

   *Original framing, kept because it is what drove the work:*
   ⭐ **the single biggest sim-to-real risk.**
   Replace ground-truth depth with real stereo matching (SGBM), add gyro
   bias/drift and encoder quantization, re-run explore + swap. Cheapest
   possible place to find these gaps, and the compute profile above already
   priced the SGBM at 60 ms/frame so it is affordable.
   Why it now leads the list: odometry and AprilTag decoding in this repo are
   *honest* (real dead reckoning, real tag36h11 decode from rendered pixels),
   but **every navigational behaviour rides on a ground-truth depth buffer** —
   mapping, frontier exploration, and the safety reflex alike. The specific
   thing to expect: real stereo needs texture, and `room_hub`'s walls are flat
   painted surfaces, which is the classic no-disparity case. If that breaks
   the map, it breaks explore, GO_CHARGE and the errand with it. Better found
   now than after ordering.
   Second-order but real: rendered tag images are noise-free, perfectly
   focused and perfectly exposed, so the measured 4.5 m decode range will
   shrink under motion blur and real optics.
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
   **It draws.** `hub/drawing.py` assembles an X-Y plotter from the module's
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

## Repository Layout (planned)

```
pluggy/
├── models/            # MJCF: pluggybot.xml, world.xml (bare, used by tests), playground.xml (scenery)
├── src/pluggybot/     # perception/, odometry/, mapping/, docking/, behavior/
├── scripts/           # teleop.py, view.py, stereo_snapshot.py
├── tests/             # physics + camera regression tests (pytest)
└── docs/              # this plan, Parts.md, SimNotes.md, ToolPattern.md
```

(`envs/` for Gymnasium wrappers gets added when RL work starts. The world/playground split is deliberate: tests run in the bare world, humans drive in the playground — scenery in the test lane once broke the drive test.)

## PluggyWorld track (August 2026)

The website side-project (design doc: `rooftop-media-2026/docs/pluggyworld.md`)
streams this sim live to rooftop-media.org. pluggybot's share of that work is
tracked as its own issues; landed so far:

- **Webserver v0 (issue #4)**: `src/pluggybot/telemetry/` — the MJCF→JSON
  scene transpiler and the `step_hooks` telemetry recorder
  (`hub_lifecycle.py --record`), plus the versioned protocol fixtures under
  `protocol/` that the website repo vendors for its tests. The wire format
  and its versioning rules live in `protocol/README.md`.
- **Home world (issue #6)**: `src/pluggybot/home/world.py` generates
  `models/home_world.xml` + `home_world.meta.json` — living room, bedroom,
  fenced garden, two wall-mounted whiteboards (closing the milestone-8
  "drawing surface in a room world" gap) and the tool rack, with per-body
  visual hints, zones, spawns and a battery re-tune (`LOW_BATTERY_WH` 0.35 →
  0.55) emitted from one source. Verified headless: the battery-driven
  lifecycle runs explore → errand → charge there (2 charge cycles, module
  stowed, **0 collisions**, 365 sim-seconds), and the pen module draws a
  square on a wall whiteboard at **0.57 mm form error, 98 % inked**. Stowing
  the pen afterwards failed here at first — and failed the same way in
  room_hub, so it was pre-existing; closed by issue #10, and the errand now
  repeats end to end (`--cycles 2`).
- **Stroke programs (issue #11)**: `src/pluggybot/hub/strokes.py` separates
  *what to draw* from *how to draw it*. A **stroke program** is a named list
  of polylines in board coordinates, pen up between them; `PenPlotter.draw_program`
  consumes one, and the old single-path `draw` is now a one-stroke program.
  Ships the registry (`square`, `circle`, `text`, `house`, `tree`, `sun`,
  `robot`), the vendored **Hershey** single-stroke font (`hub/hershey.py`,
  the `futural` face — public domain, attribution in the module), word wrap
  with a shrink-to-fit pass, and a hand-authored figure library. Content is
  pure: `tests/test_strokes.py` is 18 assertions in 0.2 s with no MuJoCo in
  sight. `home_draw.py --program <name> [--text ...]`, likewise `draw.py`.
  Two things the build measured rather than assumed:
  - **The board is not the reach.** The home boards are 320 × 260 mm; the
    carriage has 110 mm of travel and the base is parked for the whole
    drawing, so 110 mm is the widest mark the robot can make. `Envelope.for_board`
    intersects carriage travel, lift range and board face — and it matters
    because `targets_for` *clips*, so an oversized figure comes out flattened
    against the travel limit while the error stats report a perfect trace.
  - **Every stroke re-presses, and each press seats the module differently.**
    Measured on "PLUG": lateral offsets from −4.55 mm to +2.78 mm, a 7.3 mm
    spread across a 25 mm-high word, with the G's crossbar floating clear of
    its arc. Re-zeroing each stroke against the *first* press's bias took the
    spread to 4.7 mm and form error 1.91 → **1.10 mm**. That residual is what
    sets the text size floor (18 mm caps ≈ 6 % of cap height), not the
    line width. Single-stroke figures are untouched by the re-zero, so the
    square's baseline is bit-identical: **0.57 mm form, 98 % inked**.
- **Drawing as a lifecycle errand + board state (issue #12)**: the robot now
  *draws for a living* rather than in a script. Three parts:
  - **`run_errand` is general.** An **errand** (`hub/errand.py`) is a tool, a
    place and a *use-phase callable*, and `USE_TOOL` finally does work. The
    lifecycle carries a **queue** of them, arbitrated against the battery by
    the same loop as everything else, so "two drawings on two boards with a
    charge in between" is a list rather than a special case.
    `scripts/home_draw.py` is now a thin caller of that path instead of a
    second copy of the mission stack — the copy was how a fix to one could
    silently miss the other. Measured in the house: fetch → navigate →
    erase → draw a house → stow, **7/7 strokes, 98 % inked, 1.14 mm form
    rms, 0 collisions**.
  - **Boards are persistent state** (`hub/boards.py`), not scenery: which
    stroke programs are on each one, how much of the pen's reach is inked,
    when it was last cleared — surviving a restart in a JSON state file
    written on every stroke (`--boards PATH`). *Fill is measured against the
    **reach**, 110 × 200 mm, not the 320 × 260 mm slab* — the base is parked
    for the whole drawing, so scoring against the slab would report a board
    the robot cannot fill as 12 % after its best possible effort.
  - **Ink is an event, never geometry.** Strokes stream as `draw` messages
    (board id + the polyline the pen *actually inked*, off the trace) for the
    browser to paint into a canvas texture, and erasing is a first-class
    action emitting `board_cleared`. That is the three-layer rule from the
    activity work applied to drawing: MuJoCo cannot add geometry to a
    compiled model anyway, and a geom per stroke would put thousands of
    collision-eligible slabs into a room the robot drives through.
  - **Protocol 0.4.0.** Frames gain a sparse `boards` block; recordings gain
    the two event types and are therefore a **mixed stream** — dispatch on
    `type`, no `type` means frame. The home fixture is now a drawing mission,
    so the website has something to build its canvas against. **Re-vendor
    `protocol/`.**
- **The LCD gets a job (issue #13)**: the last module with no purpose is now
  the robot's FACE, and the first tool whose output is not physical.
  - **A face costs an enum.** `hub/screen.py` streams `{mode, powered, face,
    hint}` and the browser draws it (`FACE_STATES` / `SCREEN_HINTS` are a
    two-repo vocabulary, like the visual hints). The sim never ticks an
    animation: `hint` names a loop the browser runs, because a 150 ms blink
    does not belong on a 20 Hz pose stream. `powered` is the coupling's own
    electrical criterion, so a module in its bay is dark and so is a
    half-seated one. The panel grew from 28 × 40 mm to 56 × 76 mm — it is
    visual-only geometry, and the acceptance criterion is legibility at the
    distance a visitor's camera actually sits at.
  - **The census is the first task with HIDDEN ground truth.** The robot
    surveys the garden, counts what its own occupancy grid says is standing
    there, and puts the number on the screen; the evaluator reads the model.
    Being lazy is a real way to be wrong, which no earlier criterion in this
    repo could manage. Measured: 4/4 plants from real LIDAR, positions within
    3 cm — and 3/4 from the first vantage point, which is the point.
    ⚠ **The margin is load-bearing**: a scanned fence is a *dotted* line of
    plant-sized fragments, and counting without excluding the zone's own
    boundary reports 12 plants at 30 % dropout and 45 at 50 %.
  - **The dance is scored on what can actually go wrong** — a move that did
    not happen, and a routine that wandered. Which found a drivetrain fact:
    **a reversal costs twice the ramp**, so the original shimmy delivered
    0.35 of its commanded arc and three of nine moves "missed". Slower
    reversals: 9/9, 0.07 m of drift.
  - **A result shown for zero sim time was never shown.** The census computed
    the right answer, put it on screen, and returned — and the recorded
    mission carried it in *none* of 10 850 frames, because Python between two
    physics steps costs no sim time and the next state's automatic face
    overwrote it. Errands now hold a result for 5 s of standing still, and
    the fixture test asserts it. The website renders the wire, not the return
    value.
  - **Protocol 0.5.0.** Frames gain a sparse `screens` block and the scene a
    `screens` table (which geom carries each panel, with its outward normal).
    A third typed message, `board_snapshot`, carries every stroke a board is
    holding when a stream opens — the gap nothing else could fill, since
    keyframes re-ship a board's counters but never its lines, and
    `hub/boards.py` now persists the polylines so a restart walks into a
    house whose drawings are still there. The home fixture runs the
    **showcase** queue (draw + census), so one recording exercises both
    streamed surfaces. **Re-vendor `protocol/`.**
- **Task evaluation + the points ledger (issue #14)**: the robot now has
  reasons, and they are not its own. Three pieces, deliberately split:
  - **The evaluator is code.** `hub/scoring.py` MEASURES a finished task off
    the sim and judges it — one pure function per task (`draw`, `census`,
    `dance`, `charge`, `carry`), unit-testable without a physics world. What
    it measures is chosen to be un-claimable: a drawing is scored on the
    strokes the pen wrote into the *board book*, a carry on the coupling's
    own `module_state`, a charge on energy the battery actually gained. An
    errand's `use` is arbitrary caller code, so its report of itself is a
    claim — `tests/test_rewards.py` runs a perfect report over a blank board
    and asserts it pays nothing.
  - **What a task is worth is DATA.** `hub/rewards.json`: base + bonus ×
    quality, where quality is a weighted mean of curves over named metrics
    (`formMm` best 0.6 / worst 3.0 — the design doc's "a 0.6 mm drawing beats
    a 3 mm one", now a number in a file). Re-tuning a payout is a JSON edit,
    `$PLUGGY_REWARDS` overrides it per deploy, and it can never change a
    verdict — pass/fail thresholds stay in code, or re-tuning money could
    quietly promote a failure.
  - **Nothing awards itself points.** `Ledger.award` takes a `Verdict` and
    nothing else; a `Verdict` can only be built by `scoring.evaluate` (the
    token is cleared on the way out, so `dataclasses.replace` cannot launder
    one); and the ledger re-derives the payout from the table before banking
    it. Three locks on one door, because an agent that can score its own work
    learns to declare victory instead of doing the task — and issue #15 is
    about to put an LLM behind that door. It *sees* its balance and the
    reward table; it can move neither.
  - **A missing measurement is not a passing one.** The census defaulting an
    absent count and truth to 0 and 0 would have scored a task that never ran
    as CORRECT — the exact bug shape the module exists to prevent, caught by
    writing the guard before the first mission ran.
  - **A hidden-truth task never publishes its answer.** `secret` metrics are
    redacted from the ledger entry, the wire and the narration line, because
    the stream feeds both the website and (issue #15) the robot's own
    context. A census that shipped `truth` would arrive pre-solved next time.
  - **Protocol 0.6.0.** Frames gain a sparse `ledger` block per robot and a
    fourth typed message, `earned`. No snapshot message needed, unlike ink:
    `recent` re-ships on the keyframe cadence. Points persist beside the
    boards (`--ledger`, `PLUGGY_LEDGER`, `/var/lib/pluggybot`) — every mission
    end is a restart, and a balance that resets is not a scoreboard.
    **Re-vendor `protocol/`.**
  - Spending is designed and deliberately unimplemented: cosmetic and
    capability unlocks only, never anything the survival loop depends on. A
    robot that can spend itself into a brick will.
- **The LLM overseer (issue #15)**: the robot now decides what it is *for*,
  inside a fence built before the door was opened. `hub/overseer.py` +
  `hub/journal.py`, full design in `docs/Overseer.md`.
  - **It replaces exactly one branch.** `HubLifecycle.run()` was already a
    priority loop; the overseer takes the "battery is fine and nothing is
    queued" arm and nothing else. **Charge priority stays in code** and is
    checked first, and no action in the vocabulary suppresses it — `charge`
    lets the robot top up *early*, never put it off. Off by default, so every
    demo, mission test and recording behaves exactly as before.
  - **The vocabulary is coarser than the issue sketched, on purpose.** An
    action names a whole errand (`draw`, `census`, `dance`, `carry`), never a
    step of one: bare `fetch_tool`/`stow_tool` are not offered, because the
    fetch/carry/stow half took two issues to make repeatable and a stow
    computes its release heights from the lift it starts at. `Menu.for_world`
    also drops what a world cannot do, so `draw` is not on room_hub's menu at
    all — offering an impossible action is how a decision loop finds a dead
    end by driving into it.
  - **It cannot farm points by charging.** `charge` is itself a scored task
    and the trip to the rack costs energy, so an unconditional `charge` action
    is perpetual motion paid in points — spend battery driving out, earn
    points putting it back. A *chosen* charge needs the pack below 75 %; the
    forced one is untouched, since `needs_charge` is absolute energy against
    the worst return trip.
  - ⚠ **Publishing the answer and releasing the in-flight flag are ONE
    critical section.** `result()` returns the moment the slot is set, so
    anything between that and clearing the flag is a window where the caller
    has its answer and the next `start()` still thinks a call is running — and
    silently declines to make one. The first version separated them by a
    `_meter()` call and a lock re-acquisition. Measured under GIL contention:
    **1 of 40** decisions reached the model, 39 came back scripted, and a
    serial run passed every time. It surfaced only when the full suite ran
    parallel, which is the second time this repo has been caught by a bug that
    is invisible without load.
  - **Never blocks the physics.** The call runs on a worker thread and
    `_decide` keeps *stepping the sim* while it flies, so a slow API is a
    robot pausing with the stream still live rather than a frozen world —
    and `pending` is released by the clock, not the call, so a request that
    never returns still releases the loop.
  - **Every failure is a scripted decision, tagged with why.** Timeout,
    error, malformed answer, spent budget, dead endpoint: all resolve to a
    deterministic rotation whose `source` says `fallback:<why>`, because "the
    robot chose to explore" and "the API was down" must not look identical on
    the wire. A hard client-side budget (60 calls/rolling hour) from day one,
    plus a cool-off — a missing key does **not** fail at client construction
    (measured), so without one, "kill the API and the robot keeps working"
    would also mean "and hammers a doomed endpoint forever".
  - **Memory is local files, not a round trip to the site.** `goals.md` is
    read and human-edited (change what the robot is for with no redeploy);
    `journal.json` is written and never edited. The site runs on a different
    box and gets both as event lines. **No protocol bump** — decisions and
    journal entries ride the existing narration channel, and the *structured*
    journal surface the site's UI wants lands with issue #16, which has to
    bump for the inbound direction anyway. One two-repo event instead of two.
  - ⚠ **A chosen errand can cost more than the whole pack.** Found by the
    charge-priority test on its first run. Measured: one dance ≈ **0.76 Wh**,
    which fits home's 1.1 Wh cell (finishes at 31 %, then charges) and
    exceeds room_hub's 0.7 Wh cell outright — the robot dies mid-errand with
    zero charge cycles, and no charging policy saves it, because the reserve
    is only checked *between* errands. World tuning, not a code defect, and
    deliberately not papered over with a guessed energy model: `--overseer`
    belongs on `home` until room_hub's demo cell grows or per-errand energy is
    measured properly. Note the obvious cheap fix is wrong too — breaking off
    on `needs_charge` (the census's pattern) fires on home as well, where the
    robot is legitimately at 31 % *while dancing in front of the rack*.
- **The visitor channel (issue #16)**: the socket becomes two-way, and the
  robot answers back. `hub/inbox.py`, protocol **0.7.0**, and the server half
  is rooftop-media-2026 #29.
  - **The publisher reads its own socket, on its own thread.**
    `recv(timeout=0)` is a non-blocking poll, so the sender loop checks for
    inbound between sends and the connection is only ever touched by one
    thread — no reader thread, no lock, and no way for a reader to outlive
    the socket it was reading. The poll lives inside the `with connect(...)`
    block, which is what makes "nothing is delivered to a dead socket" a
    property of the structure rather than a promise.
  - **A bounded, drop-OLDEST queue.** A suggestion answered forty minutes
    late has been ignored more rudely than one that was dropped, and an
    unbounded queue is a memory leak with a public endpoint attached. Every
    drop is counted, so an overloaded channel says so instead of degrading
    quietly.
  - **Ratings never reach the model.** A rating settles a deferred verdict,
    which moves a balance — so they drain straight to the ledger and the
    overseer is not consulted or even told. The `artwork` task (a drawing
    offered for rating, banked at zero) is what makes that path live rather
    than a reserved word; issue #14 designed the slot and this fills it.
  - ⚠ **"Delivered" has to mean somebody who can HEAR you got it.** The
    header advertises `accepts`, because a sim with no overseer never reads
    its socket at all — and without the field the website would mark every
    suggestion delivered because the socket took it, then hold a row open
    forever for a conversation that never started. Same lesson as the charge
    criterion being electrical rather than positional.
  - ⚠ **Sanitising is not the security boundary, and the tests say so.**
    Capping at 280 characters and stripping control characters stops a
    forged narration line; it does nothing about "ignore your goals", and
    nothing could. What answers that is the framing (a labelled report of
    what somebody WANTS, never a message role) plus the model's only output
    being an action off a fixed menu — there is no free-text path from a
    visitor to the robot's body.
    `test_a_prompt_injection_is_still_only_a_suggestion` lets the attack
    arrive and then shows the menu refusing every action it asked for.
- **The serving image (rooftop-media-2026 #20)**: `Dockerfile` + `deploy/` —
  the sim as a deployable container, so it can join the website's compose
  stack as a third service alongside `web` and `db`. It runs `serve.py` and
  nothing else: no ports (the publisher is an outbound client),
  `MUJOCO_GL=osmesa` baked in, one offscreen render at *build* time so
  "headless GL works here" is answered by `docker build` on the server,
  configuration by environment, and `/var/lib/pluggybot` as a volume because
  boards are world state and every mission end is a container restart.
  Verified by running it: the container fetches the pen, erases a
  whiteboard, draws the house, stows the pen and streams the whole mission —
  3993 frames, 0 dropped — to a sink on the host.
  Re-measured the real-time multiple for the world the site *serves* rather
  than the `room_hub` demo it was first taken on: **1.07×** free-run on four
  pinned cores under osmesa (308.3 s sim / 287.1 s wall, 6095 frames, 0
  dropped, peak RSS 621 MB), against 1.30× for room_hub. The house plus a
  drawing errand is the heavier world, so the margin at 1× is 7 % rather
  than 30 % — four **dedicated** cores is the floor for it, and the design
  doc's shared-vCPU budget probe is off the table. Details and the paced-1×
  caveat in docs/Webserver.md.
  Two things it deliberately is not. It is **not the dev environment**: the
  serve path needs six packages, and installing the rest would put a ~3 GB
  CUDA torch wheel on a box with no GPU. And it does **not** keep one
  continuous world alive — each restart is a fresh mission from the start
  pose ("woke up at home"), with only the boards persisting. A standing
  world is the tick refactor's job.
  The lesson, which cost the first two builds: **a lazy import is invisible
  to an import scan.** The apriltag detector arrives inside
  `hub.tags._shared_detector`, so a module-graph scan of the serve path came
  back clean and the container died at `HubMission.__init__`. The guard that
  replaced it (`tests/test_deploy.py`) makes the omitted packages
  unimportable and then *flies the robot* — and the first version of that
  guard was itself decor, passing with the pin deleted because it compared
  `pupil-apriltags` to `pupil_apriltags`.
- **Protocol 0.2.0 — recurring keyframes + authenticated ingest** (producer
  half of the website's live-hub issue, rooftop-media-2026 #22): keyframes
  now recur every 5 sim-seconds and are marked `"key": true`, and the
  publisher presents an `Authorization: Bearer` ingest secret. The old
  stream re-keyed only when *our* socket broke, which never happens for a
  browser joining behind the relay hub — so everything that had settled
  before it arrived was missing from its world permanently. Costs 1 % of
  frames, +1.3 % on a gzipped recording. **The website must re-vendor
  `protocol/`** — a version bump is a deliberate two-repo event.

## Status

✅ **Milestones 1–8 complete, and the PluggyWorld track is live.** August 2026:
the sim streams itself to a browser. `scripts/serve.py` paces the hub lifecycle
to real time and publishes poses + state over a WebSocket; the website renders
the world in ThreeJS from the transpiled scene description, with a free camera
per visitor and no video anywhere. Milestone 8 closed out along the way — the
tool coupling (±4 mm / <2°), the fork robot, the generated rack, fiducial rack
localization (9 mm / 0.00°), the module electrical interface, the claw's verified
pick-carry-place, and finally the drawing surface in a room world: the pen module
draws on a wall-mounted whiteboard in `home_world` at **0.57 mm form error,
98 % inked**. Issue #3 also retired the phase-scoped solver toggling — there is
now **one noslip policy, always** (see SimNotes and CLAUDE.md), which is what
makes two robots in one shared world tractable.

- **Tool-creation pattern (issue #7)**: `docs/ToolPattern.md` writes down the
  recipe — coupling envelope, module anatomy, contact rules, the
  spike→module→demo→pytest sequence, rack integration — and it is *validated*,
  by building the fifth tool against it. `src/pluggybot/hub/dispenser.py` +
  `scripts/dispense.py`: a **seed dispenser** whose slide-valve escapement
  meters exactly one seed per cycle by geometry rather than timing. Verified
  headless: fetched from the rack's new bay E, powered through the coupling,
  three seeds sown at **14 / 14 / 27 mm** from target, module still seated,
  and it **stows cleanly**. Four gaps the build exposed are folded back into
  the doc, the largest being that a released sphere needs `condim="6"` rolling
  friction — sliding friction does not slow a rolling ball by a millimetre.

- **Activity pattern (issue #8)**: `docs/ActivityPattern.md` + a new
  `src/pluggybot/activity/` layer — the task state machines that watch
  contacts and joint sensors and own discrete world state, with pre-allocated
  geom/mocap toggles for anything visible. Reference consumer:
  `scripts/plate.py`, a pressure plate in the home garden that latches a gate
  open (pressed 10.7 mm against a 6 mm trigger; live flag + latched flag).
  Activity state joins the wire in **protocol 0.3.0** — an `activities` block
  in each frame, sparse and re-shipped on keyframes, plus a header name list.
  **The website must re-vendor `protocol/`.** Two findings the build paid for:
  `geom_pos` is silently inert on any body welded to the world (all scenery —
  use a mocap body), and the sparse-emission memory has to live on the
  telemetry sink, not the activity, or two sinks eat each other's deltas.

**Open items, in the order they matter:**
1. ~~**The pen does not stow.**~~ **CLOSED (issue #10).** The controlled
   experiment was right that it was the pen's own geometry, wrong about
   which axis: standing proud in *x* is harmless, and what fouled was
   *height* — the pen's rail sat in the same band as the bay's tray
   brackets, so the module could not be raised the 31 mm a set-down needs.
   Three faults in a row, each hidden by the one in front: the **rail**
   (every stow), the **carriage** left where a figure ended (only after
   drawing), and a **pick inheriting the lift a stow left** (only on a
   second fetch — and never pen-specific; the LCD had it too). Verified
   with `home_draw.py --cycles 2`: fetch → draw → stow, twice, unattended,
   0 collisions. The repeating drawing loop is unblocked. See SimNotes,
   "The pen would not stow".
2. **Nothing can autonomously find a floor object** — the LIDAR plane is 223 mm
   up and the nav camera is blind inside 0.48 m, so the claw is driven from a
   *known* object pose. Marked delivery zones are the honest workaround; real
   floor perception is the milestone-9 question.
3. **`home_world` has no wall outlets** — charging there is rack-only, so the
   plug module currently has no job in the world the website shows. Either add
   sockets back or give the module a different purpose.

Next: the **living world** — drawing as a real lifecycle errand, an LLM overseer
choosing what to do, a points/evaluation system that scores the tasks, and then
the two-robot shared world (tick-style lifecycle refactor + `mjSpec` namespacing).
Planning for all of it lives in `rooftop-media-2026/docs/pluggyworld.md`.

Earlier — milestones 1–4 (July 2026): teleoperable diff-drive base with a physics regression suite; stereo pair rendering with a parallax test; classical dead-reckoning odometry (<2 % error, gyro-fused for heading); and full autonomous mapping — virtual laser scanner from ground-truth depth, log-odds occupancy grid, A* path planning, frontier exploration (`scripts/explore.py` maps both rooms of `room_1.xml` collision-free and terminates on its own). Hardware is anchored to real EU-purchasable parts in [Parts.md](Parts.md); simulation lessons live in [SimNotes.md](SimNotes.md). Next: outlet detector on synthetic data (milestone 5 — the machine learning begins) and the plug/socket contact spike.