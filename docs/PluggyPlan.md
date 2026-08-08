# PluggyBot 🔌

**A simulated self-charging robot that explores, maps, and plugs itself in.**

PluggyBot is a personal robotics project built in physics simulation ([MuJoCo](https://mujoco.org/)), with the long-term goal of a design faithful enough to real hardware components that it could eventually be built physically.

The core idea: a small wheeled robot with stereo vision that explores its environment to build and maintain a spatial map, visually recognizes wall outlets, and - when its (simulated) battery runs low - navigates to an outlet and docks itself using a rigid, plug-tipped arm.

## Design Philosophy

- **Simulation-first, hardware-honest.** Everything runs in MuJoCo, but component parameters (motor torque curves, camera baseline and FOV, masses) are modeled on real, purchasable parts, keeping an eventual sim-to-real transfer plausible.
- **Rigid plug, not a cable.** Manipulating a deformable wire plug is one of the hardest problems in robotics. PluggyBot sidesteps it: the plug is fixed to the end of a rigid arm. Docking is still a genuinely hard contact-rich alignment task - but a tractable one.
- **Decompose, don't end-to-end.** Rather than one giant RL policy from pixels to behavior, each capability uses the cheapest adequate technique: supervised learning where labels are free (simulation gives them away), classical robotics where the problem is solved, and RL where it actually earns its keep.

## Core Goals

1. **Mobility** - a differential-drive wheeled base.
2. **Binocular vision** - two cameras providing stereo depth perception.
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
| Depth perception | Ground-truth sim depth first; swap in real stereo matching (e.g. OpenCV SGBM or a learned stereo net) later |
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
8. **Modular tool system — the hub pivot** *(proposed August 2026)*. The arm tip becomes a tool interface; a "tool hub" shelf stores swappable modules (first two: the Schuko plug, an LCD screen), and the robot autonomously exchanges them. The hub also charges the robot through a purpose-built low-force connector — which resolves the measured hardware blocker head-on: a real Schuko socket's sprung contacts need tens of newtons, and the robot's measured push budget is ~3 N. "Plug into any wall outlet" survives as one module (the flagship demo), no longer the load-bearing architecture. Design constraints already known from measurement: no wrist, so latch verbs are slide-in + lift/lower (kinematic mount / gravity hook, not bayonet); power-only coupling with wireless data (a microcontroller per module) keeps the mating interface dumb and tolerant; a per-module tip-mass cap protects the veer counterweight calibration. De-risk order, per house method: standalone coupling spike (schuko_spike-style MJCF, no robot) → hub + modules in sim → autonomous swap in the lifecycle. **The hardware MVP / parts-ordering trigger: a physical robot swapping plug ↔ LCD at a real hub** — extensible from then on without structural change.

Each milestone is independently runnable and demoable.

**Parallel de-risk track** (can start anytime): a standalone Schuko plug/socket contact prototype — no robot, just a scripted insertion in its own MJCF — to validate contact modeling before milestone 6 depends on it. The Schuko recess is a natural alignment funnel; find out early how much of one.

**Later experiments** (off the critical path): learned odometry (competing against the classical baseline), curiosity-driven exploration.

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
└── docs/              # this plan, Parts.md, SimNotes.md
```

(`envs/` for Gymnasium wrappers gets added when RL work starts. The world/playground split is deliberate: tests run in the bare world, humans drive in the playground — scenery in the test lane once broke the drive test.)

## Status

✅ **Milestones 1–7 complete — the repo MVP.** August 2026: the loop that names the
project runs end to end in `scripts/lifecycle.py`: PluggyBot explores, remembers
outlets, and when its (honestly modeled) battery runs low, drives to one, docks with
real contact physics, charges through the plug, and gets back to work. The
verification run had everything: a jam at the hard high outlet (benched), a phantom
decoy landmark (caught by close-range verification and erased), a successful dock
and 21 → 90 % charge at outlet B, and zero wall strikes. Milestone 6 closed with
both docking controllers scored by one seeded protocol (scripted 33.3 %, RL 25 %
end-to-end with complementary failure sets — see SimNotes for the four measured
design findings the RL work surfaced). Next: the **hub pivot** (milestone 8) — the
coupling spike first, then hub + tool modules in sim; the hardware MVP bar is a
physical plug ↔ LCD swap at a real hub.

Earlier — milestones 1–4 (July 2026): teleoperable diff-drive base with a physics regression suite; stereo pair rendering with a parallax test; classical dead-reckoning odometry (<2 % error, gyro-fused for heading); and full autonomous mapping — virtual laser scanner from ground-truth depth, log-odds occupancy grid, A* path planning, frontier exploration (`scripts/explore.py` maps both rooms of `room_1.xml` collision-free and terminates on its own). Hardware is anchored to real EU-purchasable parts in [Parts.md](Parts.md); simulation lessons live in [SimNotes.md](SimNotes.md). Next: outlet detector on synthetic data (milestone 5 — the machine learning begins) and the plug/socket contact spike.