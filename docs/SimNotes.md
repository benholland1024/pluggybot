# Simulation Notes

MuJoCo lessons from building PluggyBot, each one paid for. An entry says what
broke, why (the physics), what is true now (the constant or the rule) and what
pins it. The story of how a thing was found lives in git and the issues; a
measured number lives at its constant with its failure mode attached
(CLAUDE.md, "Working style"). Read this before touching `models/`, contact or
actuator parameters, the swap/coupling stack, or anything that trusts
odometry. The pattern docs say *what the constraint is*; this file says *how
we found out*, in the order it was paid for.

## Physics modeling rules

### Wheel joints need `armature`
A bare 50 g wheel has ~3×10⁻⁵ kg·m² of rotational inertia; a motor torque on
that overshoots any velocity target within one 2 ms timestep and the servo
chatters at the timestep frequency — the robot vibrates and bounces instead
of driving. `armature` is the motor rotor's inertia *reflected through the
gearbox*, which scales with gear-ratio²: rotor ~5×10⁻⁶ kg·m² × 50² →
**`armature="0.012"`** (`models/pluggybot*.xml`). The same chatter appears on
any velocity servo driving a light mass — the schuko rig's 90 g carrier did
it — and the cure there is a constant-force `<motor>` plus joint
`damping = F/v`, which `implicitfast` integrates implicitly and which is the
honest "10 N push, 2 cm/s free speed" semantics anyway.

### Wheel joints need `damping`, and it is physically real
Gearboxes eat torque: Pololu lists ~65 % efficiency for the 37D 50:1, so
~30–35 % is lost to friction — **`damping="0.05"`**. It is also load-bearing
for stability: the velocity servo regulates *joint* velocity, which includes
chassis pitch rate, a positive feedback that pumps energy into chassis-pitch
oscillation; joint damping dissipates it. Do not exceed honest magnitudes
(0.2 would mean a gearbox that eats the whole stall torque, and the robot
crawls).

### Wheel joints need `frictionloss` — the parking brake
A velocity servo commanded 0 resists *speed*, not force, so a parked base
walks under a sustained tool load: ~0.5 N of pen drag moved the chassis a few
mm per figure and starved the plotter's square of ink. The physical gearbox's
Coulomb friction is a parking brake the robot gets for free:
**`frictionloss="0.05"`** on the hub-era robot's wheel joints
(`pluggybot_fork.xml`; the plug robot in `pluggybot.xml` is frozen without it
for milestone 6–7 reproducibility). Its bill is a stiction deadband — see
"One always-on solver policy", below.

### Use `integrator="implicitfast"`
MuJoCo's default explicit Euler adds energy to velocity-dependent forces —
velocity actuators and joint damping, the exact ingredients of a wheeled
robot. Standard practice for actuated models (most Menagerie models use it).

### THE caster lesson: MuJoCo combines pair friction as the elementwise MAX
`friction="0.001"` on the caster never worked: at equal priority a contact's
friction is the element-wise **maximum** of the two geoms' values, and the
floor's default is 1.0. Our "frictionless" caster was a full-grip rubber ball
for weeks, silently exerting ~4 N of drag and ~0.4 N·m of yaw braking — the
hidden cause of the cruise pitch resonance, the tail-flip, a 12.6 %
straight-line odometry creep and a 2.7× in-place-turn overestimate. The
correct frictionless caster (Menagerie's Stretch):

```xml
<geom name="caster" type="sphere" size="0.02" ... condim="1" priority="1"/>
```

`condim="1"` is a normal-force-only contact (no friction dimensions exist at
all); `priority="1"` makes the caster's parameters win over the floor's
(condim also combines as max, and the floor's 3 would win). With it, the head
sits at mast height (z = 0.16, pitch 0.1°) and dead-reckoning agrees with
truth to ~0.2–1 % on straights, spins and arcs. The same rule has since
bitten the pen pads, the coupling peg (`PEG_FRICTION = 0.4` with
`priority="1"`) and the seeds' rolling friction: **know the pair-combination
rules** (friction → max, condim → max, solref/solimp → priority/solmix) before
trusting any per-geom contact attribute.

### Mass goes over the drive axle
A 120 g head cantilevered 16 cm ahead of the axle destabilised launches
(wheelie → riding the rear chassis corner); the identical mass over the axle
is benign. It is mass × position, not mass, and traction depends on it —
the Roomba/TurtleBot battery-placement rule.

### Motor sizing: torque-to-weight matters
A 30:1 / 1.4 N·m motor on a 1.1 kg robot demands ~40 N of thrust per wheel
against ~3 N of available traction — permanent wheelspin and enough reaction
torque to wheelie. Robots this size are traction-limited, not motor-limited;
when behaviour looks violent, check whether the actuator could physically
exist in that weight class.

## Test & world hygiene

- **`models/world.xml` is bare** (floor + light + robot) and physics tests run
  there; `playground.xml` / `room_1.xml` add scenery via `<include>`. Scenery
  once parked a box in the drive-test lane, and the veer re-measurement drove
  straight into a board the author had placed two hours earlier. Copying the
  floor into an including file doubles every wheel contact — a "repeated
  name" MJCF error means *delete* the duplicate, not rename it.
- **Every debugged failure becomes a pytest assertion, shown failing first.**
- **Relative-error metrics need denominators that cannot vanish.** Position
  error ÷ distance blows up on an in-place spin; heading error ÷ net rotation
  blows up on an S-curve (a 0.17° error read as "142 %"). Normalise by path
  travelled (distance rolled, rotation swept) or assert absolute error.
- **A module-scoped fixture holding mutated state makes tests
  order-dependent, and the file passes anyway.** A solver-mode set in
  `__init__` leaked into a later test's coupling pick; the dispenser's landing
  test only ever saw seeds an earlier test had dropped. Function scope is the
  honest price; run a new test in isolation before believing the file.
- **A repro that fails for its own reason cannot test a hypothesis.** A
  bare-world "draw then stow" harness with no lateral servo was already
  failing on a 17 mm standoff error, and read as a clean refutation of the
  carriage-position fault it was built to test. An A/B is meaningful only once
  the harness passes on one arm.
- **A contact census that does not exclude the floor measures gravity** — a
  cube resting on the ground is "in contact" on 86 % of steps.

## Conventions & gotchas

- MJCF `size` values are **half**-extents; `pos` is relative to the parent.
- Cameras look down their own **−z**, image-up is +y. Forward camera on a
  +x-facing body: `xyaxes="0 -1 0 0 0 1"`.
- Pitch from the freejoint quaternion `(w,x,y,z)`: `asin(2·(w·y − z·x))`,
  **positive = nose down**.
- A body with no joint is welded to its parent; `contype="0" conaffinity="0"`
  makes a geom visual-only.
- Velocity actuators: `ctrlrange` = ± no-load speed, `forcerange` = ± stall
  torque, both off the datasheet; `kv` is a tuning gain.
- **The standard frame correction**: dead reckoning tracks the **axle
  midpoint**; `qpos[:2]` tracks the **body origin**, 8 cm ahead. On curved
  paths they trace different circles (~0.1 m apart after a half-turn), and a
  spin orbits the origin around the axle. Compare truth at the axle:
  `(x − 0.08·cos ψ, y − 0.08·sin ψ)` (`tests/test_odometry.py` `axle_pos`).
- **`mj_geomDistance` does not measure box–box separation** — the box
  collider reports penetration, not distance, so pairs millimetres apart read
  `+0.0`. Sweep real contacts (set the pose, `mj_forward`, count `data.ncon`).
- **Adjacent-link interpenetration is silent.** Contact filtering treats a
  weld group as one body, so a carriage can sweep straight through its own
  head mount and the pipeline reports nothing. Clearance must be asserted from
  *geometry* — `tests/test_arm.py`'s AABB transit sweep, which has since
  caught the parked fork (9 mm into the chassis), a widened prong through the
  battery, and two 5 mm tube/carriage overlaps that contacts never would.
  Endpoint checks are not envelope checks; contact checks are not clearance
  checks. (A grandchild is not filtered either: the pen quill fought the
  module plate it was modelled inside, jamming the carriage at +21.9 mm while
  its joint reported perfect command-following.)
- **`geom_pos` is inert on anything welded to the world.** A static body's
  geom world poses are computed once when `MjData` is created and
  `mj_kinematics` never revisits them; the write lands, nothing anybody reads
  changes, no error. `rgba` and `size` work (no pose cache in between).
  Anything an activity must *move* is a **mocap body** (`activity/base.py`
  `MocapToggle`), whose pose is an input re-read every forward pass — and a
  mocap pose reaches `geom_xpos` only on the *next* forward pass, so a
  filmstrip grab comes one step later. Table in ActivityPattern.md §3.4.
- **`type="2d"` textures do not paint primitives** — boxes carry no texcoords,
  so the plate renders flat grey and nothing decodes, with no compiler error.
  `cube` mapping puts the image on every face, which is what a printed marker
  glued to a plate looks like.

## Exploration lessons (milestone 4)

- **The nearest-frontier deadlock.** A forward sensor cannot observe the cells
  beside the wheels, so the nearest frontier is always the sliver just outside
  the FOV — the robot "arrives" instantly and the frontier never dissolves
  (420 sim-seconds, zero movement). `MIN_FRONTIER_CELLS` (0.3 m) ignores them,
  and a 360° look-around runs at startup and whenever no distant frontier is
  reachable. Discovery rides along with every manoeuvre: line of sight is a
  property of where you are standing, and driving is what buys the view.
- **Collisions corrupt the map, not just the paint.** Grinding a wall slips
  the wheels → odometry counts phantom distance → the map frame slides → old
  walls repaint at new believed positions ("jail bar" artifacts). Prevention:
  inflation must exceed the chassis half-diagonal (0.15 m) plus margin **and
  track the outermost geometry** — `traversable_mask` inflates 7 cells
  (0.35 m) because the arm's tips sweep 0.27 m; waypoints ≤ 3 cells apart with
  ≤ 0.08 m arrival radii (sparse ones cut corners through the ring); and a
  scan reflex at `FRONT_STOP_RANGE` (0.25 m) for what planning misses.
- **The reflex must be armed in every manoeuvre, not just while driving.**
  Gated on `mode == "drive"`, look-around spins ran blind and passed 0.257 m
  from a box — 7 mm outside the threshold, luck not design; a refactor that
  shifted the trajectory by <1 mm/step then ground for 503 steps. Arming it
  everywhere cut that to 43, all during the escape. A forward ±20° camera
  reflex fundamentally could not protect a spin (the chassis corner sweeps
  arcs the camera never sees); the 360° LIDAR is what closed that (Parts.md,
  "Vision & ranging").
- **Do not re-arm a reflex that is already firing** — re-triggering backoff on
  every scan turns a bounded `BACKOFF_TIME` pulse into an open-ended reverse.
- **Termination is "no *reachable* frontiers".** Pockets and hairline gaps are
  blacklisted when A* fails; exploration ends after a look-around plus
  repeated pathless replans. A* plans to the reachable cell NEAREST an
  unreachable goal, never a blind greedy advance into unknown space.

## Plug-era lessons (milestones 5–6, parked)

Wall-outlet docking — the YOLO outlet detector, the plug arm with its
alignment feelers, the scripted DOCK controller and the RL environment — was
parked when the hub made purpose-built low-force docking the charge path
(PluggyPlan, milestone 6). `scripts/lifecycle.py`, `envs/dock_env.py` and
`docking/schuko.py` still run and their tests still pass; this section keeps
what transferred.

### Landmark and detector lessons (milestone 5)
- **"Where I saw it from" is not "which way it faces."** Deriving an outlet's
  normal from the mean sighting position recorded where the robot happened to
  drive: **31.2°** off the wall, the hand-off pose 33 cm sideways. Reading the
  normal off the occupancy grid (`landmarks.wall_normal`: sum unit vectors to
  nearby free cells; a wall blocks half the circle so the sum points out of
  it) gave 0.0°. Still the rack's facing source (`RackFinder`), with its
  free-standing-partition caveat — see "The charge press is an odometry pump".
- **Decompose an error before fixing it.** The 33° miss was first blamed on
  odometry drift, which measured **0.02°**; the estimator owned all of it.
- **The camera's FOV bounds usable height**: eye at z 0.18 m, fovy 41° → a
  target at 0.38 m is seen at 0.6 m and gone at 0.35 m. The same geometry is
  why the claw cannot see its grip point and why the dock camera must drop the
  lift to see the charge tag.
- **The generator's own val split cannot measure the detector** — it reported
  mAP50-95 0.9938 while calling a light switch an outlet. Every real defect
  was found by `scripts/eval_detector.py` (collision-free poses through
  `room_1.xml`, scored against segmentation truth) or by a distance sweep.
- **A handful of poses is a smoke test, not a measurement.** An 8-pose spot
  check read a decoy moving 0.61 → 0.93 as a regression; over 300 poses the
  ranking reversed. Compare recipes on hundreds of samples or not at all
  (`train_docking.py` `EVAL_EPISODES`).
- **Check what a generator change actually produced** before training on it:
  a distance-scaled aim jitter pushed the decoy out of shot in 22 % of the
  negatives it was meant to sharpen (decoy false positives 9 → 26).
- **Recall is recoverable; precision errors compound.** The robot sees each
  outlet 13+ times a run, so a miss is recovered next glance, but a
  *systematic* false positive on a fixed decoy accumulates in one place and
  graduated into a confirmed phantom the robot parked in front of (−364 cm
  from the nearest real outlet). Threshold 0.5 → 0.7 (26 of 40 decoy hits
  gone for 2 detections in 105), `MIN_SIGHTINGS = 3`, and a close-range
  re-verify before committing — `reject_target(forget=True)` erases phantoms
  only, because deleting a real outlet that merely jammed twice loses a
  charging spot for good. Tune a threshold against what the consumer does
  with the errors, not against F1.
- **Budget the tolerance across the whole chain.** `FACING_TOLERANCE` at 2°
  spent two-thirds of the ±3° docking budget on the settle alone; at 0.5° the
  pipeline parks at −0.49° / 1.3 cm / 60 cm out, and that 0.5° is what the hub
  coupling's < 2° yaw envelope still relies on.

### Docking lessons (milestone 6)
- **Walls have no holes**, so wall sockets are surface-mount (Aufputz)
  fixtures — recess fully proud of the wall, invisible collision bodies in
  the generated `models/schuko_sockets.xml`, aligned with the visual outlets
  and absent from segmentation.
- **A wall cannot brace you — one-way contacts don't pull.** Wall pads meant
  to take the insertion couple only push the robot *away*, the same direction
  as the reaction; with-brace tipped worse than without. Measured push
  capacity at outlet height: ~3 N forward, ~4 N backward. The mirror image
  arrived with the hub: a gravity hook doesn't push either (the lean-pad).
- **Asymmetric mass makes a diff-drive veer open-loop** — the arm at y=−0.05
  veered 26 cm over 4 m because unequal wheel loads slip unequally. The
  battery at y=+0.06 is tuned to null the *measured* veer, counterweight
  first and packaging second; re-measured with every module aboard, the
  heaviest adds 5.5 mm over 2.7 m and no re-tune is needed: tool mass is
  bounded by the coupling and the tip-load budget, not by veer.
- **Gravity is a docking axis.** The plug axis sags 7.8 mm under gravity
  (`DOCK_DROOP_COMP`; the fork line sags the same, `swap.DROOP_COMP`), and
  press-induced pitch sag grows with lift height. Feed-forward calibration
  constants, exactly the shape a real robot needs.
- **A camera above the plug loses the target exactly when it matters.** Below
  ~0.32 m the socket slides out of the frame bottom and the box centre biases
  up (+23 mm at 0.19 m) while the lateral centre stays honest (+1 mm).
  Measure each axis of a visual servo separately; they do not fail together.
- **The seat detector was harder than the seating.** Four positional verdicts
  (extension window, base release, odometry advance, floor contact by name)
  were each defeated by a real event. What survived is **the electrical
  criterion** — a pin ≥ 19 mm into the recess, reachable only through a hole
  — and it is literally the sensor the hardware has. It carried over verbatim
  to `rack_charge_contact`, to `module_power_state` and to every swap
  verdict since.
- **Scripted-baseline verdict: mechanics solved, vision-z is the gap.** From
  a perfect standoff the stack docks deterministically
  (`test_mechanical_dock_from_perfect_standoff`); end to end under 2 cm of
  standoff jitter, 4/12, failures dominated by plug-height error the
  box-centre servo cannot measure. That gap is what `envs/dock_env.py` was
  built to fill.

### Schuko contact spike (`docking/schuko.py`, `scripts/schuko_spike.py`)
A compliant carrier pushes the plug at a `PUSH_FORCE` 10 N cap; guarded by
`tests/test_schuko.py` and still one row of the noslip sweep.
- **Collision geoms must be convex**, so the recess is *composed*: a 12-box
  well wall, 12 tilted boxes as the entry funnel, floor slabs leaving two pin
  holes; capsule pin tips double as their own chamfer.
- **Capture is set by the entry chamfer, not the recess.** Measured
  chamfer → tolerance: 2 mm → ±3 mm, 4 mm → ±6 mm, 8 mm → ±18 mm. Real Schuko
  rims have a ~2 mm bevel, so the honest envelope is **±3 mm lateral / ±3 mm
  vertical / ±2° yaw**: capture ≈ body clearance (0.75 mm) + chamfer, and the
  deep well only guides *after* capture. Do not widen the chamfer to make a
  failing controller pass — that tunes the world, not the robot (a dished
  face plate on the physical outlet is a Parts.md decision, not a sim
  default).
- **Yaw is the tight constraint**, with a signature: every yaw/lateral jam
  stops ~19 mm short (= pin length) as the pins bottom beside their holes.
- Jams stall cleanly at the force cap (worst transient ~32 N) at
  `timestep 0.001` + `solref "0.005 1"`, which is the swap's floor too.

### The RL docking environment (`envs/dock_env.py`, milestone 6)
- **A synthetic sensor must be calibrated against the real one — including
  the robot's own body.** The projected "detector box" needed two measured
  corrections: the real YOLO boxes plate + housing at a 47 mm half-extent,
  not the 42 mm plate; and with the arm extended the tube top climbs the
  frame and shrinks real boxes ~25 % from below. A policy trained without
  the occlusion learned to approach arm-first, because in-env vision saw
  through its own arm.
- **Randomise with a mocap body, not recompilation** — the socket rides
  `mocap="true"` and reset writes `mocap_pos`; static geoms on it behave as
  kinematic fixtures.
- **The eval protocol is only trustworthy once it reproduces the baseline.**
  `eval_docking.py` makes the jitter protocol explicit constants and reran
  the scripted controller to the recorded 4/12 before scoring the policy —
  same trials, same real YOLO, same electrical criterion for both.
- **Odometry in the loop is part of the sensor model.** A policy trained on
  true-pose observations scored 100 % in-env and 1/24 in the room: it had
  learned to grind the wheels against the wall, free under ground truth and
  fatal under dead reckoning. The env runs the actual `DeadReckoner`;
  whatever estimator the robot runs, the training env runs the same one.
- **SAC on this task finds the skill early and destabilises late** (peak,
  then collapse with runaway episodes in the replay buffer, reproduced twice);
  `best.zip` is chosen by eval success rate, never recency.
- The scoreboard — scripted 8/24, RL 6/24, failures complementary — is in
  PluggyPlan's milestone table. The lesson: in-env success bought by details
  of a synthetic sensor is repaid at deployment, and only an eval that runs
  BOTH controllers on the SAME trials makes the repayment visible. Its own
  residual mismatch (an arm-out box extent the calibration did not cover)
  parked the policy in an out-of-distribution hover at one outlet.

## Rendering lessons (milestone 5)

### Segmentation rendering needs `offsamples="0"`
MuJoCo applies multisample antialiasing to the segmentation buffer too,
blending geom IDs at edges, so stray pixels carry IDs of geoms that are not
there — and a label box spans the min/max of its mask, so one pixel 200 px
away stretches it across the frame. Over 236 positive scenes **12.3 % grew a
second blob and 5.5 % came out grossly elongated**; training scored mAP50-95
0.940 *despite* ~3 % garbage labels, and the val images showed the model
drawing two tight, correct boxes where the truth was one absurd bar. The
model was right. `<visual><quality offsamples="0"/></visual>` drops stray
blobs to 0.0 %, and `make_labeled_sample` keeps only the largest connected
blob. **The renderer is not a measurement device by default** — the same
lesson returned for the robot's cameras in issue #110, below.

### Dataset regeneration must clean its output directory
The train/val split is a per-image random draw, so a file whose split changes
leaves its old copy alive in the other split: regenerating after the MSAA
fix left **195 stale pairs** from the buggy run in training, and every
"residual" corrupt label was a leftover. The generator deletes `images/` and
`labels/` first (pytest-guarded); the symptom is one basename in both splits.

## Hub coupling spike (milestone-8 prep)

Standalone fork-and-peg rig (`rack/coupling.py`, `scripts/hub_spike.py`): a
tool hangs by a long peg axle in two upward-open V-trays, the arm's fork takes
the peg *outboard* of the trays, gravity is the latch, and the only verbs are
slide and lift (no wrist). Guarded by `tests/test_hub_coupling.py`:

- **Measured envelope: ±4 mm lateral, −8/+6 mm vertical, < 2° yaw.** Yaw is
  the tight constraint again: picks survive 2°, returns do not, ±4° jams at
  50–120 N (`test_yaw_4deg_is_outside_the_envelope` pins the limitation — if
  it starts passing, re-measure and update here). Navigation's 0.5° settle is
  what makes v1 usable.
- **Retention beats traction**: held through 8 m/s² of shake, more than the
  wheels can transmit, carrying 300 g (2× the per-module budget). No spring,
  magnet or actuator.
- The fork runs 22 mm low and lifts `LIFT_STEP` 36 mm to latch: the V-plate
  tips must pass *below* the peg, and 8 mm less lift makes the peg exit by
  grinding up the tray flank at the full 10 N cap (clean is ~2 N). Three such
  geometry bugs were found by filmstrip + phase telemetry, none by reasoning.
- **Depth referencing by gentle press**: the approach drives until the fork
  bridge bottoms against the tool face under the force cap.

### Robot integration: the fork inherits the plug's lessons, item by item
- The bumper reaches the hub before the fork does (retracted vertex 25 mm
  behind the chassis front), so the arm extends 60 mm for hub work.
- RCC droop is a fork axis too: `DROOP_COMP` 8 mm, feed-forward in the lift
  preset.
- The parked fork interpenetrated the chassis (9 mm plus 3 mm of spring
  droop), caught by the geometric clearance sweep; `FORK_MOUNT_RAISE` absorbs
  it, and the lift presets import it rather than re-typing it — a constant
  that describes geometry has exactly one home.
End state: robot-driven swap cycles succeed with ±3 mm / 1° hand-off jitter.

### Hub-in-room navigation: three bugs the bare world could not show
Putting the rack in `room_hub.xml` and driving to it end to end:
- **A verdict written in the wrong frame.** `module_state` compared WORLD
  coordinates against RACK-LOCAL constants; with the rack at the origin they
  coincide, so the check "worked" for weeks and reported every correct
  placement in room_hub as a failure while the module landed 0.2 mm from the
  hang plane. The verdict transforms through the rack body's LIVE pose. **An
  identity transform is not a validated transform**: a check that only ever
  ran at the origin has never been tested.
- **A sign on the terminal travel.** The carried peg rides 7 mm *ahead* of the
  fork vertex (the wrist tilts nose-down under the tool), so the return must
  stop SHORT by that much; the code added it, overdriving 14 mm past the
  tray's ~16 mm mouth. Symmetric-looking compensations are where sign errors
  hide. These two are the costliest bugs in the repo, and both were invisible
  to every cheaper test — which is why the mission suite runs on any geometry
  change.
- **Fixed choreography does not survive navigation.** A* arrives to ~8 cm and
  the coupling captures ±11 mm, so travel is computed from the BELIEVED
  distance to the hang plane every time; and since `drive_to`'s arrival
  radius happily parks 7 cm off the bay line, where a 0.2 m creep's tag servo
  has 1–2 cm of authority, the fix is the oldest one in driving — **back up
  and take another run at it** (`refine_standoff`, 70 mm → 3 mm in one retry).

### Real AprilTags (`rack/tags.py`)
- **Marker size is a range decision.** A tag36h11 needs ~25–30 px to decode:
  at 1280×720 the rack's 120 mm marker decodes past **4.5 m**, the 30 mm bay
  and module markers to ~2.5 m — far more than the arm's-length reads they
  exist for. PnP translation is linear in the assumed size, so one decode
  serves every marker size.
- It got faster (PnP replaced the depth buffer: one render per look, not
  two) and it matches hardware, which has no depth camera on that path.
- **What it actually bought is identity.** Every reading is keyed to a decoded
  ID, so the servo is told *which* marker to steer on; the colour-plate
  stand-in it replaced once locked onto the charge bay's marker and dragged a
  module 22 cm toward the wrong bay. Two of the stand-in's lessons survive
  it: **a stale prior must lose to an observation** (the believed rack pose is
  a fallback; a confirmed tag overrides it, tested from a 30 cm-wrong prior),
  and **odometry cannot close a terminal approach** (~20 mm out by the return
  leg, twice the capture window — the terminal travel is ranged off a tag).

### The hub lifecycle: four failures, all of them about *driving*
- **The fork does not hold a tool sideways.** Nothing constrained the peg
  along its own axis; the first carried module walked off in a turn. Fixed
  in hardware: axial end-stops (`fork_stop_l/r`).
- **Tuck the arm before you drive.** After a successful stow the extended
  fork swept the module it had just hung off its trays. Stowed is the
  driving configuration; the fork deploys only once lined up on a bay.
- **Stop on the sensor, not on believed distance.** Pressing the rack slips
  the wheels and dead reckoning counts it as progress, so a charge press ran
  to timeout 15 s after connecting. It stops on `rack_charge_contact`.
- **Intended contact is not a collision.** The pogo-pin press read as 24 750
  steps of crashing to a counter written for wall-grinding; `press_steps`
  now sits beside `collision_steps`.
And an arbitration rule: charge > errand > explore, with an explore budget —
explore-first mapped, ran flat, charged and mapped again without ever
reaching the errand.

### Capture and release are not symmetric
A carried peg settles only ~9–18 mm above its hanging rest — BELOW the tray
flank tops — so a flat drive-in strikes the flank: `RETURN_CLEARANCE` lifts
clear, then sets down. And undoing the pick's lift exactly leaves the fork's V
still touching the peg (seated in both trays *and* resting on the fork, then
dragged out on the retreat): release needs `RELEASE_DROP` clearly below.
Reading the contact list settled in one run what three rounds of tuning had
not.

### Rack v2 (the unified rack): three measured calibrations
- **Approach overshoot is 4 mm, not 12** (`PICK_OVERSHOOT`): with no depth stop
  to press against, 12 mm wedged the peg between the fork V and the tray flank
  tips; 4 mm puts it low on the near flank where the lift centres it.
- The carried peg rides 7 mm ahead of the vertex (`CARRY_OFFSET`) — the droop
  story again.
- **"Leaning on the wall" must be geometry, not intention.** The free-body
  rack scooted 9.6 mm under a sustained charge press with its rear member
  6 cm proud of the wall; with actual braces the press costs 1.2 mm of
  take-up and a swap cycle moves it < 1 mm. Modelling the rack as a free body
  is what made this measurable.

### A gravity latch cannot push: the lean-pad, and where the power goes
Measured on a carried module before the drawing tool was designed:
- **The hang is repeatable**: after a net-zero manoeuvre the module returns to
  its resting lean within 0.00° / 0.02 mm and settles in 0.14 s, so a pen tip
  *can* be calibrated against the resting pose. In transit it swings 10.3°
  peak-to-peak through a turn — a sequencing constraint (pen clear of the
  board while driving), not a mechanical one.
- **The restoring arm and the disturbing arm are the same 22 mm**
  (`PEG_ABOVE_BODY`): a horizontal tool force buys a proportional, large
  rotation — 0.10 N → 5.3°, 0.25 N → 25° and running away, 0.50 N → 51°
  flopped onto the fork with the tip 38 mm back. A marker wants 0.5–2 N. So
  the lean-pad is not a sway damper, **it is the part that lets a tool exert
  force at all**, reacting the torque in compression at a ~29 mm lever below
  the peg instead of by gravity at 22 mm. With it: 0.1 N → 0.01°, 2 N →
  0.19° / 0.26 mm, linear at ~0.095°/N
  (`test_carried_module_can_exert_tool_force`, shown failing at 51° first).
- Three things bounded the pad's shape, and only one was the lever: the V
  self-centres so well that the seated module lands at the same fork-local x
  to 0.00 mm across 0–8 mm of overshoot, so the pad's target plane is a
  constant and the *lift* carries it into contact; the **parked envelope**
  (~60 mm between chassis top and the scanner row) set the depth, after a
  first version jacked the fork into the scan row; and the lever has a floor,
  because pad force is reacted at the peg and a peg pushed harder than its own
  weight (1.28 N) rides out of its V — tool force caps near 1.5 N by
  arithmetic and measured fine at 2 N. Check that first if the parked
  envelope ever tightens.
- **⚠ The power contacts do NOT belong on the lean-pad** (correcting an
  earlier Parts.md decision): a pad's preload is capped by the same weak
  22 mm geometry at 0.56 N absolute, ~0.1 N realistically. The peg already
  sits in four V-notch plates carrying **0.43–0.47 N each, 1.79 N total** of
  gravity preload, self-wiping on the seating slide, and is already the one
  metal part. So the peg IS the connector: split into two conductors around
  an insulated centre, the left and right V-pairs are the two poles
  (`peg_xml` / `module_power_state`, `scripts/module_power.py`).
- **Continuity is clean where it should be.** Over a room_hub errand the
  carry had 0 brown-outs; every interruption was a mate/release transition.
  Report brown-outs as **worst duration in the carry window**, not a
  percentage of steps (0.4 % as blips and 0.4 % as one outage are different
  hardware), and report the poles **separately** — a half-seated coupling
  (one pole on) is a real failure a boolean files under "off". With honest
  peg friction the worst outage under deliberately harsh driving is
  **178 ms**, so a module's holding capacitor is sized for ~200 ms (Parts.md).

### The drawing tool: five bugs, and only one was about drawing
`tools/drawing.py`, `scripts/draw.py`. The plotter's Y is the module's own
carriage, Z the robot's lift, pen pressure the arm's reach through a sprung
quill; the base is parked throughout. What it cost:
- **Teleporting the robot drops the tool** — a free body held only by gravity
  stays behind in mid-air (`dz/dlift` calibrated to exactly 0.000: the pen
  was on the floor). Carried tools mean the robot has to *drive*.
- **A stiff position servo handed a step is an impulse.** An 80 mm carriage
  setpoint jump threw the module out of the fork, both poles open; the same
  80 mm in 5 mm steps held. **Every position setpoint is ramped** —
  `control.slew` for the wheels, `PenPlotter.ramp` / `ClawTool.set_lift` /
  `jaws` for the rest — and the claw relearned it on three axes (a 184 mm
  lift step ejected the module; slammed jaws batted the block away).
- **`on_fork` is a position heuristic, not a seating check** (3–4 cm
  tolerances; it read True through that ejection). The electrical criterion
  is the seating check.
- **A P-controller that stops commanding at the target overshoots when the
  command is rate-limited** — `slew` was still unwinding, and the pen's
  square-up went in at −9.5° and came out at +7.5°. A 7.5° yaw error swings
  pen depth 14 mm across the 110 mm carriage sweep, most of the quill's
  travel, which is why one edge of the first square was blank. Settle and
  re-check (`control.square_up`).
- **The pen needs its own compliance, soft and long-travel.** The wrist has
  no RCC along the approach axis, so pressure would be arm error × the arm's
  1200 N/m servo. A 200 N/m quill engaged ~0.5 mm and lifted off the top of
  the figure as droop grew with lift; at **60 N/m** and ~10 mm nominal press
  the force holds 0.4–0.8 N across the figure (`PRESS_EXTRA`).
- **Report shape error, not tracking error**, and **decompose before you
  fix**: a rigid-translation fit (translation-only ICP against the command)
  splits *where it landed* from *what shape it is*. The first square measured
  10.35 mm absolute but 2.14 mm form under a 17.47 mm offset — displaced, not
  distorted — and the writeup had gone looking for a stiffness problem. An
  offset is one calibration constant; distortion is mechanics.
- **Calibrate under load, but know what it fixes.** Pressing shifts the pen's
  home 10 mm sideways and re-zeroing there took the circle from 9.2 to
  2.2 mm; it does *not* fix scale (gain 0.999 → 0.992). The "kinetic" residual
  measured after that was mostly the parked base rolling — see "One always-on
  solver policy".

## Sensor-realism pass: stereo could not produce the map's scan

The mapper was fed MuJoCo's ground-truth depth buffer through `scanner.py`,
which read ONE camera's depth row — a 2D laser scan wearing a camera's
clothes, with `right_eye` vestigial. OpenCV SGBM on the actual rendered pair,
in the best case stereo will ever get (parallel, coplanar, noise-free
cameras): mid-room facing a painted wall, disparity on **49.7 %** of the scan
row at **593 mm** median error against a 50 mm grid cell; flat painted walls
are the classic no-disparity case, and the one good pose was aimed at the
rack's AprilTags. The decision (2D LIDAR + one camera, `perception/lidar.py`)
and both tables are in Parts.md, "Vision & ranging". The lesson: **a sensor
that never fails cannot teach you which behaviours depend on it** — this gap
survived seven milestones because ground-truth depth always works, the same
trap the detector's val split set one layer up.

## The claw module (milestone 8): the fourth tool

Decided by measurement before anything was drawn (`tools/gripper.py`,
`scripts/pickup.py`; numbers at `coupling.py`'s claw constants):
- **The chassis was never the limit** (800 g at the peg drops wheel load
  15.0 → 12.6 N; static tipping needs ~5 kg). **The coupling is, and it is a
  moment limit**: the gravity latch takes ~**0.45 N·m** of pitch moment, so
  800 g hangs on the peg axis and 400 g unseats at 150 mm out. Reach is far
  more expensive than mass — hence a pendant straight down the peg axis.
- **The robot cannot see what it is picking up** (floor leaves the nav
  camera's view inside 0.48 m; the grip point is closer). The grasp is
  open-loop from a memorised pose, as the socket vanished from the dock
  camera. Finding floor objects is unsolved: the LIDAR plane sees nothing on
  the floor.
- **The structure carrying a gripper must not occupy the gripper's space** —
  the drop tube reached the block first and the jaws closed on air while a
  graze read as "both pads in contact". The pendant length is derived from
  the pad height, pytest-guarded, after breaking once when one moved alone.
- **Converge a height, do not correct it once**: the grip follows the lift at
  ~0.87:1 (droop and lean change as the arm descends), so one correction left
  the pads 11.6 mm high, holding the block by its top 12 mm.
- **Carry clearance is sized by SWING, not static height**: a 172 mm pendant
  swinging the 10° a carried module swings moves its tip ~30 mm
  (`CARRY_LIFT`).
- **55 mm of forward angle costs 7 % of the moment budget** (`CLAW_REACH`;
  60 g at 55 mm is 0.032 N·m) and lets the tool's own `claw_eye` — the first
  camera on a TOOL, free of a CSI port because module data already crosses
  the coupling wirelessly — see the grip point. What bounds the angle is the
  rack: modules hang business-end-inward with 80 mm to the wall. The camera's
  placement took three renders, not a calculation, and it aims through
  `_look_at()` rather than hand-typed cosines, which rot when a constant
  moves. The angle amplifies heading error at the grip by ~17 %; the aim
  assertion is per-axis because the jaws capture 62 mm laterally and only
  28 mm fore-aft, which a single radial tolerance hid.

### The grip that leaked: a solver artifact, and a bad inference
The block crept out of the jaws at ~8 mm/s during a carry. "Tripling the grip
force changed nothing, therefore not slipping" was the wrong inference —
**a negative result is not a diagnosis** — and Ben, watching the render, said
it was visibly creeping. Slip was identical to 0.01 mm across a 3× force
range (an 18× friction margin) and *slower lifts were worse*, which no real
friction failure does: the signature of MuJoCo's **regularized friction
drifting under sustained load**. `noslip_iterations` cures it (−99 mm →
+0.03 mm) — and broke the coupling: at 3 iterations the module landed on the
fork **unpowered**, because the peg seats by SLIDING into its V and the pass
suppresses sliding. That conflict was **a friction bug wearing a solver
costume**: the peg and V-notches were on MuJoCo's default μ = 1.0, whose 45°
friction angle is exactly the V's flank angle, so the peg sat on the sliding
threshold and depended on solver drift to seat. Steel on printed plastic is
μ ≈ 0.4 — the spike had always set it while the generated world silently
used 1.0, so the measured envelope was never the one in play. At the honest
`PEG_FRICTION = 0.4` with `priority="1"` (the caster lesson, third outing)
the coupling seats with noslip on or off, and continuity got honestly worse
(a lower-μ peg shifts more under hard driving: the 178 ms above). The
tempting move was to raise μ until it stuck — the schuko-chamfer mistake in
a new costume.

The claw then stopped needing any solver mode: a hard contact constraint on
the two jaw pads (`GRIP_SOLIMP`) removes the creep at its source (−21.7 mm →
−0.13 mm over a 100 mm lift, against +0.28 mm for the noslip cure) and is the
better model — a rigid pad on a rigid block should not squash. The plotter
kept the pass for a while as a cost trade; what it was actually curing is
the next section.

## Vectorizing the occupancy-grid scan update (issue #2)

The per-scan ray loop was the most expensive Python in the mission loop.
Rewritten as one (ray × sample) numpy batch with a single weighted
`bincount`: **9.4 ms → 1.3 ms per scan, 7.4×** on the recorded room_hub
LIDAR fixture (`tests/test_grid_vectorization.py` keeps the loop as its
reference and requires ≥ 5×; time the two sides INTERLEAVED — CLAUDE.md).
- **The old loop's semantics hid in a numpy footnote**: fancy-index `+=`
  counts a duplicated index ONCE, so a cell sampled twice by one ray got one
  vote while separate rays accumulated normally. A naive vectorization doubles
  most free evidence (47.8 % of cells wrong, up to 3.6 log-odds). The batch
  dedupes *consecutive* samples, legitimate because a straight ray never
  re-enters a cell.
- **"Identical" has a stated tolerance**: the same cells get the same votes
  (linspace matched term for term) but `k·L_FREE` once is not bit-equal to
  `L_FREE` k times — atol 1e-9 on values, exact equality on the thresholded
  image, byte-identical `map.png` end to end.
- Two micro-optimisations (~0.3 ms each): int32 indices viewed as uint32 make
  one compare cover both bounds, and one weighted `bincount` replaces
  count-then-scale.

## One always-on solver policy (issue #3): noslip loses, the wheels confess

The shared two-robot world cannot phase-scope solver settings per robot, so
`noslip_iterations` is one number for everyone. `scripts/noslip_spike.py`
measured every behaviour with a stake in it under each candidate. All pen
figures are the SQUARE (the creep-sensitive figure); "robot swap" is a
9-point hand-off jitter grid (±3 mm × ±1°), open-loop from
`place_at_standoff`:

| noslip | rig cycle | schuko | robot swap clean | pen ink | pen form | ms/step (room_hub) |
|---|---|---|---|---|---|---|
| 0 (was shipped) | 5/7 | 5/5 | 5/9 | 63 % | 1.94 mm | 0.210 |
| 1 | 5/7 | 5/5 | 3/9 | 72 % | 1.84 mm | 0.410 |
| 2 | — | — | 3/9 | 92 % | 0.60 mm | ≈0.41 |
| 3 | 6/7 | 5/5 | 1/9 | 93 % | 0.61 mm | 0.422 |
| **0 + wheel brake** | 5/7 | 5/5 | **5/9** | **99 %** | **0.60 mm** | **0.210** |

- **Always-on noslip is worse than useless at the system level.** The
  aligned all-clear still holds, but under ±3 mm of hand-off jitter every
  noslip ≥ 1 run half-seats the module (on the fork, not powered), and at 3
  two corner cells miss the tray by ~35 mm on the return. The bare rig said
  noslip 3 was its *best* row while the robot said 1/9 — **measure a solver
  policy on the full system, not the rig.** Cost is binary, not
  per-iteration: turning the pass on at all doubles the step.
- **The plotter never needed the solver, and mostly it was not solver drift.**
  The parked base **rolls**: a wheel velocity servo commanded 0 resists speed,
  not force, and ~0.5 N of pen drag walks the chassis. `frictionloss="0.05"`
  on the wheel joints — the gearbox's parking brake, the static sibling of
  the `damping="0.05"` that already models its 65 % efficiency — draws a
  better square than the 2× step-cost pass ever did, at zero extra cost.
- **A wrong fix taught the right lesson.** Hard `solimp` on the tire
  *contacts* (the claw-pad pattern) also fixed the pen — and two mission
  tests failed: mm-scale wheel slip is baked into the swap's feed-forward
  travel constants (`PICK_OVERSHOOT`'s comment says so), hard tires slip
  ~0.7 mm less, and the stow's LOWER phase showed the base rolling 9 mm
  forward with believed and true advancing *in lockstep* — rolling, not
  slipping, which is what finally named the mechanism. The brake fixes the
  rolling at its source; the tires and the travel constants stay exactly as
  calibrated. Anything touching wheel contact or joint friction must
  re-verify the bay-C pick and the mission stow.
- **The mission was rolling dice, and every physics tweak rerolled them.**
  Three configurations differing by under a millimetre of wheel behaviour
  failed the full mission three different ways — pre-existing single-attempt
  fragilities one lucky trajectory had been threading. So `swap_at_bay`
  VERIFIES its outcome (electrical seating for a pick, hung-in-bay for a
  return) and takes another run with a freshly ranged travel, and a route
  failure triggers a look-around and a re-plan instead of giving up. **A
  retried measurement is a new draw from the error distribution; a retried
  constant is the same error again.**
- **The verdict**: `noslip_iterations = 0`, always, everywhere; no code
  mutates solver options at runtime (`contact_physics` / `grasp_physics` are
  deprecated no-ops); creep under sustained load is fixed at its actual
  source, per part — `GRIP_SOLIMP` where a *contact* drifts, `frictionloss`
  where a *joint* rolls. `tests/test_noslip_policy.py` guards all of it,
  including a slow end-to-end square that must ink > 85 % with no solver help
  (63 % without the brake, shown failing first).
- **Open, measured en route**: the open-loop hand-off envelope is narrower and
  more asymmetric than "±3 mm / 1°" — −1° yaw picks fail at every policy (the
  fork under-reaches the peg) and the (−3 mm, +1°) return misses by ~34 mm.
  The grid bypasses the tag-servo refinement real missions run, so mission
  margins are better; the asymmetry is real and nobody had swept the corners
  before.

Meta-lesson, twice now: a solver mode that "fixes" a behaviour is a claim
about WHICH part is misbehaving, and it does not name the part — the pass
"fixed" grasping while the bug was the peg's friction coefficient, and "fixed"
drawing while the wheels were rolling. Decompose before you fix, and a
velocity servo is not a position hold.

### The brake's bill: static friction creates a control deadband
The claw's square-up took **55 s** of sim time to settle 0.23° on braked
wheels (9.5 s before). The servo torque is `kv·(target − actual)`, so a
commanded wheel speed under `frictionloss / kv` (0.1 rad/s) cannot move a
stopped wheel at all, and every P-turn controller shrinks its command toward
zero as the error shrinks — parking itself in the deadband. `control.
turn_command` floors every nonzero turn at `W_BREAKAWAY` (0.08 rad/s of yaw,
above the ~0.05 breakaway): commanding less than breakaway is
indistinguishable from commanding zero, so the floor costs nothing. **Every
honest piece of physics added to the model sends a bill to the controllers
tuned without it**, and it is paid in the controller, not by removing the
physics.

## The home world (issue #6)

The generated house + garden (`pluggybot.home.world`) taught three things:
- **Put the origin inside a room.** A bare `MjData` spawns the robot at the
  origin; a plan running from (0, 0) wedged the chassis in the south wall at
  every plain load (`test_robot_spawns_clear_of_the_geometry`).
- **Range off the bay's OWN marker, not the rack's.** The rack tag hangs at
  the rack's centre, outside the dock camera's view from the outer bays, so
  those bays silently fell back to odometry — whose +13…21 mm drift a pick
  forgives (the V self-centres on the lift) and a return does not.
  `TagSpotter.bay_range` reads the marker that sits on every bay's approach
  axis: commanded travel vs truth ≤ 2 mm.
- **A retry is a repeat of the attempt, not a different manoeuvre.**
  `put_back` computes every height from the lift it starts at, so a retry
  that inherited RELEASE height began 50 mm low and drove the peg into the
  flanks; `swap_at_bay` restores the entry lift first.
The pen still did not stow after a navigated errand, identically in both
worlds — pre-existing, and the next section.

## The pen would not stow (issue #10): two clearances, and a tool's stow pose

A set-down must satisfy two clearances **at once**, and nobody had written
the second down: a **floor** — the peg rides over the tray FLANKS, measured
**14.7 mm** above where it hangs — and a **ceiling** — whatever else the
module carries must clear the bracket feet and columns hanging from the rail
just under the trays. `put_back`'s raise has to land between them. A
bare-world sweep (module hung at its bay, lifted in 1 mm steps, contacts
counted):

| module | ceiling (first foul) | window above the 14.7 mm floor |
|---|---|---|
| LCD, plug, claw, seed | none below **80 mm** | wide open |
| **pen, rail above the pen line** | **16 mm** (rail vs bracket feet) | **~1 mm** |
| pen, rail below the pen line | 40 mm (same pair) | 15–39 mm |
| pen, rail below, carriage parked out | 28 mm (block vs feet) | 15–27 mm |

Three faults sat in a row, each hidden behind the one in front, and each was
found by making the previous fix and running again:

1. **The rail moved below the pen line** (`PEN_RAIL_DZ`). Its job — keeping a
   retracted quill off the shaft — is served on either side; what did NOT
   move is the pen line, so `PEN_BELOW_PEG`, the lift presets and the pen's
   moment arm are untouched (lowering the whole assembly would have spent
   25 % of the lean-pad's ceiling). The drawing came back unchanged.
2. **The carriage has a stow pose.** A figure leaves the carriage where its
   last stroke ended (+37 mm after a square); run out along the peg axis it
   swings into the bracket band and the ceiling drops to 27 mm against a
   needed 31 (`RETURN_CLEARANCE` 20 mm + the ~11 mm a carried peg rides above
   its rest). `PenPlotter.carry_config` centres it — the errand rule "leave
   the tool in its CARRY configuration". Found because a navigated
   pick-and-return *without* drawing passed in both worlds: the difference
   was the drawing, not the world or the navigation.
3. **A pick INHERITED the lift a stow left behind.** A stow ends at RELEASE
   height, `LIFT_STEP + RELEASE_DROP` = 50 mm below where a pick must enter,
   and nothing put it back — every mission before this fetched exactly once.
   The second fetch slid under the peg and came away with nothing, then
   reported `stow OK` for a tool still hanging in its bay. `swap_at_bay`
   commands `align_lift()` before every pick.

The trap in how it failed: the module jammed on the bracket feet, the wheels
slipped, dead reckoning counted the slip, and `_drive_until` returned
**"arrived"** with every number looking clean. Any drive that can end in
contact needs a contact-sensing stop (issue #94 built the general one).
**A capability is not demonstrated until it has been demonstrated twice in a
row** — `home_draw.py --cycles 2` before believing any change to the
swap/coupling stack, because the second cycle starts from the state the first
one left. Verified: three navigated fetch → stow cycles in room_hub, two full
fetch → draw → stow cycles in the home world.

## The seed dispenser (issue #7): the first tool built from the pattern doc

`docs/ToolPattern.md` was mined from the pen and the claw; the dispenser was
built *against* it, and stage 0 (measure the tool's demands against the
envelope before drawing it) came out entirely on paper and was right: the
magazine hangs on the peg axis (L = 0, no moment cost), a dispenser releases
and never presses (the lean-pad ceiling does not apply), and it inherits the
coupling envelope with no new mating interface. The escapement metered one
seed per cycle on the first run. What the doc lacked, folded back in:
- **Tolerance class is a stage-0 question.** A sown seed lands where it
  lands, so this controller skips the claw's back-up-and-retry refinement —
  matching control effort to tolerance class, and not touching travel
  constants that implicitly contain wheel slip.
- **A payload retained by geometry** (a capped tube over a shelf, exit
  blocked by the shuttle) needs no carry-swing analysis.
- **A dropped sphere rolls forever, and sliding friction does not touch it.**
  Released with 0.15 m/s of residual motion a seed travelled 586 mm at
  `condim=3` — identically at μ 0.7 and at μ 1.0 with priority, because a
  rolling ball is not sliding. Rolling resistance is a separate friction
  dimension MuJoCo will not solve below **`condim="6"`** (`SEED_FRICTION`
  "0.9 0.02 0.005": 14 mm, stopped). Placement 220 mm → 14–27 mm, bounded by
  the drop. The caster lesson on a third axis; table at the constant.
- Five new dynamic bodies moved the scene census and made both committed
  fixtures stale: adding a tool is a telemetry event even with no protocol
  bump.
A bare-world return sweep across all five modules — same script, same
choreography — hung four and left the pen on the fork with no navigation in
the picture, which is what exonerated navigation and localised issue #10 to
the module's own geometry. The pattern's "keep the rack-facing face flat,
below plate height" is measured, not a hunch.

### Pure pursuit orbits a nearby target (the pirouettes)
Sowing three seeds 200 mm apart cost 111 s and 3130° of turning.
**`drive_toward` cannot converge on a destination closer than about its own
overshoot — it flies a circle around it**: overshoot puts the target beside
the robot, `w` saturates, heading error pins near 85° and `v = V_MAX·cos(85°)`
keeps just enough speed alive to sustain a ~25 mm orbit until drift drops it
inside the arrival radius. ~900° of turning per 200 mm hop, and a stable
limit cycle, not a wobble. Only short hops are affected (the claw and the
plotter drive 0.45–1 m and measure 183° / 194°, near the geometric minimum),
which is why nothing caught it. Fix: a **terminal mode** (`slow_radius=`,
opt-in, path-followers untouched) with two properties, and it needs both — a
hard `TERMINAL_CONE` (±25°) outside which `v` is exactly zero (a soft taper
alone does not kill the orbit, because `v` in the orbit is already only
0.03 m/s) and a linear distance taper inside it. Terminal speed clears the
stiction deadband at any sane radius, so no breakaway treatment is needed.
`tests/test_navigation.py` integrates a kinematic unicycle rather than
MuJoCo (the cycle is a property of the control law: 939° kinematic vs ~900°
in sim) and pins the *defect* in the default law so the fix's premise cannot
rot. Second fault, demo geometry: sowing ACROSS a row makes every hop a
sideways translation, ~90° out and back per seed; `SeedDispenser.
row_heading()` sows along it, as a seed drill does. 111 s / 3130° → 56 s /
288°, placement 17 / 20 mm. (The cube next to the row was ruled innocent by
measurement — zero non-floor contacts — not by eye.)

## The activity layer (issue #8)

A pressure plate that latches a garden light (`activity/plate.py`; it was a
gate until issue #93, and the gate was this repo's one live mocap body — the
`geom_pos` trap under "Conventions" is its surviving lesson, table in
ActivityPattern.md §3.4). Three measured rules the pattern doc carries:
- **Hysteresis**: one wheel crossing flips a bare threshold **4** times, a
  hysteretic one (`PLATE_ON` 6 mm / `PLATE_OFF` 3 mm) **2** — one press, one
  release, the floor.
- **An analogue flag defeats sparseness**: `depressMm` over a 300-frame
  crossing is 38 deltas at 0.1 mm rounding, 15 at 1 mm. Quantise to what a
  consumer can act on or keep it off the wire.
- **Delta memory belongs to the SINK, not the activity.** `serve.py --record`
  runs a publisher and a recorder over one physics with independent
  `FrameBuilder`s; a memory on the Activity would have had them consume each
  other's deltas (`test_two_sinks_over_one_activity_set_stay_independent`,
  shown failing).

## Stroke programs (issue #11): a press is a measurement, and it moves

Splitting *what to draw* (`tools/strokes.py`) from *how* (`tools/drawing.py`)
was meant to be content work; two of three lessons were physics, invisible to
every earlier figure because a square and a circle are ONE closed,
mirror-symmetric stroke.
- **Every stroke re-presses, and each press seats the module differently.**
  `calibrate_loaded` re-zeros for the press *once*; an eleven-stroke word
  carries eleven offsets, which is distortion, not offset — "PLUG" at 25 mm
  caps measured −4.55…+2.78 mm per stroke. `draw_program` reads where the pen
  actually is after each press and corrects that stroke back to the FIRST
  press's bias, lifting to move (sliding a pressed pen draws the correction):
  spread 7.3 → 4.7 mm, form 1.91 → 1.10 mm. The first stroke is untouched, so
  single-stroke figures are bit-identical. The residual sets the **text size
  floor**: at 18 mm caps 1.1 mm is 6 % of a letter and it reads; at 12 mm it
  does not. Legibility sets the size, not line width.
- **The board is not the reach.** The slab is 320 mm wide and the carriage
  reaches 110 mm of it with the base parked (`Envelope.for_board`);
  `targets_for` CLIPS, so an oversized figure draws flattened against the
  travel limit while `track_rms` reports a perfect trace of the wrong
  commands.
- **A mirrored figure is invisible to every number.** +lat is the viewer's
  LEFT, so text advances toward −lat; flip it and bounds, ink length and form
  error are all identical. The sign convention gets its own unit test
  (`test_text_advances_away_from_the_viewers_left`) — and the same bug was
  already in `home_draw.py`'s overlay panel.

## The LCD's face and the census (issue #13): counting the fence

- **The census counts the FENCE unless told not to.** The zone rectangle IS
  the fence line; a solid fence is one huge component the span filter drops,
  but real LIDAR dropout makes it a *dotted* line and the dots are
  plant-sized. With 30 / 50 / 70 % of boundary returns dropped the counter
  found 12 / 45 / 57 "plants" among four, silently; with `census.MARGIN`
  (0.4 m) it found 4 every time, and ground truth excludes the same border.
- **A survey that finishes its route can finish the battery too.** The
  four-vantage route answered correctly and the mission ended dead holding
  it; the garden reads 100 % coverage from the first vantage, so the errand
  stops on coverage or `needs_charge` and reports the coverage it got.
  Coverage is what separates a surveyed answer from a lucky one.
- **A result shown for zero sim time was never shown.** Python between two
  physics steps costs no sim time, so `show_count()` then `return` was
  overwritten by the next state's automatic face before a single 20 Hz frame
  was built — the count appeared in **none of 10 850 frames** while the
  result dict was perfect. Errands hold their result for `PRESENT_S` and the
  fixture test asserts it on the wire. **Any state a viewer is meant to see
  has to outlive a frame interval, and only the recording proves it did.**
- **A reversal costs twice the ramp.** `control.slew` ramps at 30 rad/s², so a
  turn reversing an existing turn crosses the whole ±range with the first
  half still turning the wrong way: the dance's shimmy landed 0.35 of its arc
  back-to-back against 0.86 from rest, and the three "missed" moves were the
  three reversals (numbers at `errand.py`'s dance routine). A probe that
  pauses between moves cannot see it — the bug lives in the transition.

## The whiteboard question (issue #22): a grader that could not read

The plan was to read the board: render the answer's glyphs, compare to the
inked polylines. Measured with real Hershey digits, **a 6 and an 8 are closer
to each other (1.7 mm at 55 mm caps) than a correctly drawn 6 is to its own
ideal (1.1 mm)** at every size the robot can draw, and coverage is no better
(a drawn 6 covers 97 % of an 8 at 2 mm). A classifier would fail correct
drawings and pass wrong ones on exactly the pairs arithmetic produces. So
the grader is a split: **correctness** against the answer the mind committed
to at claim time; **fidelity** of the ink to *those* glyphs, as a check for
wrong WORK. The fidelity bar had to be measured too — 8.0 mm from synthetic
renderings passed the `robot` figure drawn instead of a "5" at 5.05 mm,
because a busy figure covers the glyph it stands in for and only the
ink → glyph direction notices — hence `ANSWER_MATCH_MM` 4.0 with an
ink-length ratio (a correct answer measures 0.8–1.2 mm; the robot figure uses
2.76× the ink). Table and bars at `economy/questions.py`;
`scripts/answer_spike.py` re-measures if the pen, board or `ANSWER_CAP`
moves. **Before building a recogniser, measure how far apart the things you
must tell apart are, in the units your machine's error is measured in.**

## The charge press is an odometry pump (issue #22's acceptance test)

The first mission to fetch the SAME tool twice with a charge in between
failed its second pick with the robot **828 mm** from where it believed it
was: `charge()` holds `CHARGE_PRESS` against the pins for the whole cycle,
the wheels turn, the chassis does not move, and `DeadReckoner` integrated
every slipping revolution. `_drive_until`'s docstring had said so for two
milestones — what was fixed then was the *stopping*; the *integration*
outlives the drive. Downstream it presents as hardware: `drive_to` "arrives"
a metre from the bay, the bay tag honestly ranges 1.25 m, the terminal creep
asks for 4× nominal and grinds into the rack for its full timeout.
- **A press is not travel.** `HubSwap.pinned` keeps the encoders counting and
  discards the position they imply; heading still comes off the gyro, which
  does not slip. Set for the press, cleared before the undock (real travel).
- **A plausibility guard can reject the truth.** `plausible_travel` refused
  a terminal creep far from the ~0.215 m a hand-off implies, halved the
  furniture-shoving, and the pick still failed — it was rejecting the *tag*,
  which was right, in favour of odometry, which was wrong. Kept as a damage
  limiter; when a guard fixes the symptom and not the outcome, the model of
  the fault is wrong. (The same shape recurs in tests: an assertion the
  design does not promise is a guard rejecting the truth, not rigour.)
- **More map can make an estimate worse.** Driving behind the rack to charge
  turns it into the free-standing partition `wall_normal` warns about:
  conditioning 0.824 → 0.077, direction 80° off. `RackFinder` KEEPS a facing
  that came off a well-conditioned look (`MIN_FACING_CONF`) — more sightings
  improve a position, not a facing whose input is the shape of mapped free
  space.
The workflow lesson that found it: **print the believed pose next to
`data` before theorising** — one line ended two rounds of plausible reasoning.

## The charge approach was blind (issue #32): the one dock with no eyes

On a hosting pack the robot works ~1000 sim-seconds between charges, and
some way into every long run `go_charge` crept to a believed standoff,
touched nothing, and ended the mission at 7 % with "mission complete". The
tool bays kept working in the same runs (their creep is steered and ranged
off the bay's own tag); the charge approach was dead reckoning end to end,
with the charge tag the rack generator had always emitted for it unread.
Why intermittent: on a 2-sim-hour run the frame drifted 0.002 → 0.054 →
**0.693 m** over three approaches and the third still docked within 8 mm,
because the rack belief (1983 sightings through the same drifted pose) had
drifted COHERENTLY and the errors cancelled. The blind dock survives exactly
as long as belief and frame drift together; whatever decoheres them lands the
standoff outside the envelope — measured (`scripts/charge_spike.py --blind`,
robot placed truly at standoff-plus-error while believing itself at the
standoff) at ~6 cm lateral, ~10° heading, and a 10° rack-yaw belief error,
which one badly conditioned free-space look delivers. `HubMission.
charge_approach` measures the standoff off the charge tag's own PnP pose,
creeps under tag servo, verifies electrically and takes a backed-off second
run; every row then passes to 20 cm and 20°. Two camera traps in `dock_eye`:
- **It rides the carriage, on the fork line** — `PLUG_LATERAL` (5 cm) right
  of the chassis centreline, so "centre the tag" aligns the FORK and is 5 cm
  wrong for aligning the CHASSIS, the edge of the whole blind envelope. The
  charge servo holds the tag at `-PLUG_LATERAL`. A servo law is a statement
  about which part of the robot must arrive.
- **It rides the LIFT.** From the align preset the charge tag at pin height
  is below the camera's view entirely (decodes from every pose at lift 0,
  none at align height); the approach commands `CHARGE_LOOK_LIFT` first.
And a failed dock is narrated `stranded`, never "mission complete".

## The tool drop was an approach error (issue #30): the bays get eyes too

Long unattended runs degenerated hours in: swaps flawless, then chronic pick
and return failures (31/33 in the back half of a 4-sim-hour run), reproduced
with zero overseer calls — physics, not policy. Measured
(`scripts/swap_spike.py --blind`, belief decohered by `across` after a clean
pick): 0–2 cm hangs; **4–8 cm puts the module on the FLOOR at the rack's
foot** (the peg misses the tray V's ±8 mm mouth and the retreat drags it off);
**−3° of heading alone drops it**; pick-side, 8–10 cm knocks it off its
bracket. Once on the floor the failure is permanent — every later pick finds
an empty bay, the approach lane is littered, and the grinding retries slip
the wheels and pump more drift. `refine_standoff` kills BELIEVED lateral
only, the tag servo has 1–2 cm of authority over the creep, and the tag RANGE
fixes depth but not the line; retention was never the problem (8 m/s² of
shake). `HubMission.bay_fix` — issue #32's measured standoff, one bay over,
`_measured_standoff` the shared core — measures off the bay's own tag inside
the retry loop, so a second attempt is a fresh measurement; every row of both
sweeps to 10 cm / ±10° then ends hung. For what measurement cannot promise
away, the `reset_tool` inbound kind puts a lost module back at
`model.qpos0`: admin-only, code-handled, never shown to the overseer, refused
while a module is electrically seated.

## Drift hygiene (issue #42): the dock is the anchor, and identity beats distance

Dead reckoning was never corrected, and every long-run failure family traced
to it: 0.69 m of frame drift, the blind dock, the 4 cm cliff, and — with both
measured approaches in — the pen lost at t=7372 when drift put the bay tag
outside the dock camera's field and the first look answered None. The system
survives *coherent* drift and dies of *decoherence*. One mechanism keeps the
frame coherent:
- **The dock is the re-anchor** (`HubMission.anchor_at_dock`, called by
  `charge()` when the pins conduct). With both pins pressed the axle sits
  0.2056–0.2059 m from the pin faces, ±1 mm across, 0.0° of heading, on every
  instrumented dock — the one pose the robot occupies to millimetres BY
  CONSTRUCTION, held for minutes every cycle, free. ⚠ **Snapped to the
  COMMISSIONED prior (`rack_prior`), not the believed rack — measured, not
  argued.** The first version anchored to the belief, and its four-sim-hour
  run logged, per dock:

  | t | drift before dock | after anchor | rack belief error |
  |---|---|---|---|
  | 2480 | 0.015 | 0.006 | 0.003 |
  | 6702 | 0.144 | 0.147 | 0.146 |
  | 13466 | 0.367 | 0.361 | 0.344 |

  `after_anchor` tracked the belief's own error exactly: belief follows the
  frame (that is what recency weighting is for), the frame drifts, the anchor
  snaps to the belief — a loop in which nothing references the world. The
  prior is "what a robot that booted docked knows": the map frame is DEFINED
  by the dock, as on hardware. The rack landmark is re-seeded there too, and
  drift is bounded to one shift's worth.
- **The rack merges by identity, not distance** (`RackFinder.look`). The
  store's 0.4 m gate is for anonymous outlets; after 0.5 m of decoherence a
  recovery spin's sightings spawned a SECOND rack landmark the stale one
  outvoted, so the spin that existed to fix the belief could not touch it.
- **The rack belief is recency-weighted** (`RACK_RECENCY` 0.25, an EMA past
  the first sightings). A mission-long average remembers the MEAN historical
  frame: a 2000-sighting belief moved by nothing when fresh looks arrived.
  Sized against what a spin delivers (~5 sightings), five looks move it ~76 %,
  enough for the bay tag to enter the dock camera's view. Outlets keep the
  pure average.
- **The bays got the charge bay's no-tag recovery** (spin, refresh, fresh run,
  look again) — and it only works WITH the two belief rules above:
  `test_the_recovery_finds_a_bay_the_first_look_lost` fails if any of the
  three is removed.
Also: **the erase rides the first stroke** (`mission/errand.py`). `book.clear`
on believed arrival let a drifted robot narrate "erased whiteboard_a", press
at empty air, and leave the ink's ground truth blank as if a wipe had
happened at a board nobody visited. Map evidence decay was DEFERRED on
measurement: the smear is drift's shadow, and it gets built only if a
post-anchor long run still shows planner refusals.

## The bay tag's yaw was a coin flip (issue #88): fit the rack, not the tag

`test_bay_fix_measures_the_standoff_the_bay_is_actually_at` went red when a
wall on the far side of the house moved — a change that altered the pick
trajectory by a fraction of a millimetre — and the only number that differed
was the bay tag's decoded yaw. A square planar target viewed near head-on has
two PnP solutions mirrored about its normal, nearly equal in reprojection
error, chosen by pixel noise (`Error, more than one new minima found.`); the
robot is square to the bay tag *by construction* at every standoff, so this
is the worst case, hit every time. `scripts/swap_spike.py --yaw` (true pose
nudged 2 mm at a time, belief untouched, one look per pose):

| nudge | bay tag yaw | standoff error (one tag) | error (fit) |
|---|---|---|---|
| +0 mm | +6.09° | 0.041 m / 5.2° | 0.003 m / 0.2° |
| +2 mm | +2.90° | 0.016 m / 2.0° | 0.005 m / 0.1° |
| +4 mm | −6.93° | 0.062 m / 7.8° | 0.008 m / 0.4° |
| +10 mm | −7.48° | 0.066 m / 8.4° | 0.004 m / 0.0° |
| +16 mm | −1.56° | 0.019 m / 2.5° | 0.005 m / 0.1° |
| +24 mm | +6.95° | 0.047 m / 6.1° | 0.006 m / 0.1° |

Bimodal, not noisy — about ±6–7° or near zero, never between — while the
tag's *translation* holds under a millimetre; at the 0.45 m reach, 7° is
5.5 cm of lateral error, the issue-30 cliff spent on a coin toss by the one
manoeuvre made measured specifically to be drift-immune. A median of N looks
cannot help (the render is deterministic, so N looks at one pose are one
look), nor a sign from the believed facing (the MAGNITUDE is wrong too, and
the belief is what this measurement exists to be independent of). What the
robot has is more tags: `localize.fit_rack_facing` fits the rack's known
layout (`coupling.RACK_TAG_FACES`, commissioning knowledge on `RackPose.
prior`'s footing) to the six or seven decoded translations by a 2D Kabsch
fit, so the facing comes off the BASELINE between tags — 0.25 m at the bay
pitch, most of a metre across the rail — instead of the foreshortening of a
30 mm square. `_measured_standoff` uses it with ≥ 2 rack tags, keeps the
bay's own tag for position, falls back to the single yaw with one
(`HubMission.fix_source`: `plane:N` / `yaw`). **0.0075 m / 0.7° worst** at
the bay; **0.0023 m / 0.3°** at the charge standoff from a two-tag fit,
against 0.008 m / 1.1° single-tag. Two things found on the way: **the layout
must be FACES, consistently** — mixing a plate centre in put a steady
+0.46° bias on the two-tag fit (atan(2 mm / 0.25 m), the plate's
half-thickness over the baseline; a differential error is a rotation, a
common one is swallowed by the translation); and bay E's tag, oblique at the
edge of the field, fits 11–15 mm off and the unweighted fit still holds
±0.4° because six others outvote it (weighting deferred until a pose needs
it; the rms is on `RackFacingFit`). The bars are 0.02 m / 2°, the sweep is
the regression test (single-tag arithmetic fails it at twelve of thirteen
poses), and the premise — one tag's yaw still flips — is pinned separately
and marked slow.

## A stalled drive is an odometry pump (issue #94): the bumper is the sensor

A robot in the garden driving 30 s at a point beyond a shut gate: true pose
pressed against it, believed pose **4.38 m** further on. `pinned` guards the
one press the code DECLARES; an ordinary navigation drive that stalls had no
guard but its timeout. Reproduced against the east fence
(`scripts/stall_spike.py --blind`):

| 30 s into the fence at | true travel | believed | pump |
|---|---|---|---|
| `drive_toward` (V_MAX 0.4) | 0.71 m | 4.99 m | **+4.28 m** |
| 0.30 m/s | 0.79 m | 6.76 m | **+5.98 m** |
| 0.12 m/s | 0.78 m | 2.05 m | +1.27 m |
| APPROACH_V 0.05 | 0.78 m | 0.95 m | +0.17 m |

The pressed wheels turn at 83 % of command; the robot does not move. And
`drive_to`'s own stagnation check watches the BELIEVED pose, which is
advancing at full speed: behind a knee-high box the lidar looks over, it
returned **True after 6.6 s with the robot 1.30 m short** — the
frame-relative lie again.
- **The motor is not the sensor, measured.** Pressed, the wheel servos sit at
  0.44 N·m against 0.35 cruising and a 1.23 N·m start transient, nowhere near
  the 2.06 N·m limit: a 1.8 kg robot's tyres slip long before its motors
  saturate. A cap on believed travel against the command cannot work either —
  the belief IS consistent with the command; the ground disagrees.
- **The bumper is.** `HubSwap.pressing` is `pinned`'s rule sensed rather than
  declared: a chassis contact on the side the wheels are turning toward is a
  press, and a press is not travel. Judged against the **encoders, not the
  command** (the new bumper reflex reverses the command the step after
  contact while the wheels spend the slew still rolling forward — waved
  through as travel until this was fixed, 1.27 m short); held
  `PRESS_RELEASE_S` (50 ms) past the last contact, because a cruise-speed
  press bounces — 777 gaps of 4–20 ms in 30 s pumped 1.21 m on their own;
  quiet when nothing is pressing (zero chassis contacts through a pick, a
  carry and a return). After it: −0.004 / −0.003 / +0.002 / +0.002 m down the
  same table.
- Downstream: `drive_to` gains a bumper reflex, the lidar reflex's twin for
  what the scan plane looks over — back off and replan — and its stagnation
  check now fires honestly (behind the box: False after 12.4 s, belief 0.07 m
  from truth). The charge creep's no-progress detector sees the press itself:
  a healthy dock presses **0.67–1.24 s** from first chassis contact to both
  pins conducting, which the swap's 0.4 s `STALL_TIME` cut short every time;
  `CHARGE_PRESS_STALL_S` (4 s) bounds a press that will never conduct.
- What it does not cover: the bumper spans z 0.06–0.12 m and the scan plane
  sits at 0.223 m, so an obstacle between them is met by the fork or the
  arm, and one under 0.06 m by the wheels or the caster. The fork is
  excluded on purpose (it touches pegs and trays as its job). Marking a
  bumped cell in the map was dropped on paper: the lidar's over-the-top rays
  scrub a low obstacle's cells free within a second. If a post-#94 long run
  still shows a pumped frame, those are where to look.

## The squaring-up loop had no floor (issue #108): a wedged robot never ends

The M14 baseline's first six unattended days included one that never ended:
`USE_TOOL: arrived` at t=2259, then nothing from the robot while offers
expired, the pack drained 57 % → 0 % and the sim kept stepping 700 s past
`max_sim_time`. The pen's `drive_to_board` ended with `while |heading error|
> 0.004 rad: step(turn_command(err))` — no timeout, no step cap, no stall
check — and the chassis had ridden up onto the board's mount (z 0.04 →
0.12 m), wheels half off the floor, yaw drifting a degree a second and never
inside tolerance. Three more copies existed (claw, dispenser, `HubMission.
face`). Nothing else caught it because every mission guard sits BETWEEN
errands on purpose (a tool on the fork has to be stowed), and an empty pack
does not stop the body: `Battery` clips at 0 Wh and the motors kept drawing
~30 W. One implementation, `control.square_up`, with a sim-time budget
(`FACE_BUDGET_S`, 30 s, ~3× the worst healthy face — 3.0 s from 5°, 11.4 s
from 179°) and an explicit `(error, squared)`; the pen's `drive_to_board`
returns False, the errand skips the drawing and stows the tool. **A bound is
not a recovery** — the robot on the mount is still on the mount; that is
issue #107's `reset_robot` and the `stuck` death. The premise pin,
`test_a_pen_that_cannot_square_up_gives_up_and_says_so`, raises after twice
the budget because a regression that hangs the suite is worse than one that
fails it.

## The world was not the same world twice (issue #110): MSAA is a random number generator

Evaluation.md §1 rests on "nothing in the world is random", and the first
committed `scripted` series broke it: five identical days gave three
trajectories, identical to the milliwatt-hour for five errands and then
parting inside a DRIVE — never inside a press, a stroke or a charge. A
drive's inputs are the lidar (`mj_ray` with seeded noise) and the tag camera
(an EGL render plus the AprilTag decoder). `scripts/determinism_spike.py`
flies the scripted day N times in separate processes, hashes state every
half second and every camera image, decode and scan, and reports what moved
first. **What moved**: three days state-identical for 1500 s disagreed on the
camera image on 2563 of 2613 looks with a tag in view, on the decode on
15–18 of them, and on the lidar on **0 of 10 784** scans. Rendering one
static scene ten times with no physics step gave ten different images — ±1
in 7–37 pixels, all at shadow edges — and the decode moves only on the
marginal look where a flickering pixel sits on a quad corner; one moved
decode is a moved rack belief that a later drive plans from. **The GPU's
multisample resolve of shadowed edges is not deterministic, and 0.6 % of
looks carry it into the robot's beliefs.** `offsamples="0"` makes every
render byte-identical at no cost to the detector (same tag set, centres and
translations at every range out to 2 m, nothing at 3), so the robot's
cameras render without it (`models/pluggybot*.xml`, the coupling spike's
XML) — the segmentation-label lesson through a different door. Ruled OUT:
the lidar, the two-thread decoder (one answer per image, every time), and
contention (per render, not per machine). ⚠ The committed `scripted` series
in `results/` predates the fix and is the record of the pre-fix spread;
`tests/test_render_determinism.py` pins the fix and its premise.

## Stacked blocks creep on default contact (issue #120): the grip that leaked, again

Three 26 mm cubes stacked with 2 and 4 mm of lean — a tower by any standard —
fell at 16.9 s on MuJoCo's default soft contact; 6 and 12 mm of lean fell at
2.6 s; only a perfect stack stood 60 s. The same regularised-friction drift
as the jaw pads, fixed the same way, at the source: the challenge's blocks
carry `GRIP_SOLIMP`, after which a tower stands for the 30 s measured iff its
centre of mass is over its support and falls inside 0.31 s otherwise. **What
is true now:** any free body meant to REST on another for longer than a few
seconds needs the hard contact, or the sim grades the solver (Challenges.md §4).

## A second robot perturbs the first at the last bit, and not before 98 s (issue #167)

Measured while M12's namespacing was built. With a second copy of the robot
attached (`MjSpec.attach`, prefix `r2_`) and left parked, the first robot's
spin and 15 s drive in `room_hub`, and a whole pen drawing in `hub_world`,
hashed **byte-identical** to the same runs alone — near or far, contacts on or
off. The full 1500 s scripted `home` day did not: the state diverged at
**t = 98.2 s**, mid-drawing, by **1.4 × 10⁻¹⁴** on the pen module's free
joint and 10⁻¹⁵ on the fork's compliant joints and the modules hanging on the
rack, with `ctrl` identical at that sample and no perception input differing
for the next 70 s. That is the constraint solver's rounding with an extra
island in the problem, not a code path (the code path was ruled out by flying
the refactored code alone against the baseline: IDENTICAL), and chaos does the
rest — 0.02 mm on that drawing's form error, a different day by 1500 s.
Enabling `mjENBL_SLEEP` to drop the parked body from the solve changes the
first robot's numbers on its own. **What is true now:** the parity instrument
proves a CODE change exact (fly the new code alone); a WORLD change — another
body, even one that touches nothing — is a different day at the 10⁻¹⁵ level
and its parity claim is "same code path, same decisions", read off `ctrl` and
the perception trace, never off the state hash. `determinism_spike.py
--second-robot X,Y` is the flight; `tests/test_two_robots.py` pins the short
identical cases.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list,
   `ncon`) — vibes don't bisect. Print the believed pose next to `data`
   before theorising.
2. **Render a filmstrip** (offscreen `Renderer`, 12 tiled frames) when
   numbers confuse — "standing on its tail" was invisible in scalars. Pick
   the camera by sweeping azimuth at the moment of contact, and aim a *free*
   camera at the working point; a tracking camera on the module body hid a
   16 mm seed in all eight azimuths.
3. Bisect one variable at a time, and assert that programmatic XML patches
   actually applied (`assert count == 2`) — a silent no-op once produced
   identical "before/after" results.
4. When symptom-fixes keep trading one failure for another, stop tuning and
   **measure the force balance directly**: `mj_contactForce` per contact,
   summed as torque about the COM, named the caster as the yaw-brake in one
   run after days of plausible theories.
5. **Consult reference models** (MuJoCo Menagerie — Stretch for diff-drive);
   the caster idiom was sitting in their XML all along.
6. **Decompose before you fix**, and a **discriminating experiment beats a
   hypothesis**: navigated-vs-bare and drawing-vs-not each localised a fault
   in one comparison; `--cycles 2` found the third by doing the thing twice.
