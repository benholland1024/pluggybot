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
any velocity servo driving a light mass — a spike rig's 90 g carrier did
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
  there; `room_hub.xml` adds scenery via `<include>`. Scenery
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
  *geometry* — an AABB transit sweep (`tests/test_hub_swap.py` sweeps the
  fork's), which has caught the parked fork 9 mm into the chassis, a widened
  prong through the battery, and two 5 mm tube/carriage overlaps that
  contacts never would.
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

## The plug era

The plug-anywhere robot (milestones 5–7: an outlet detector, a plug arm with
alignment feelers, scripted and RL docking controllers, a Schuko contact
spike) was parked by the hub pivot and deleted in #376; git history before
that issue holds it and its numbers. Its wall sockets survive as frozen
scenery in `models/schuko_sockets.xml`, which `room_hub.xml` includes.

## Hub coupling spike (milestone-8 prep)

Standalone fork-and-peg rig (`rack/coupling.py`, `scripts/hub_spike.py`): a
tool hangs by a long peg axle in two upward-open V-trays, the arm's fork takes
the peg *outboard* of the trays, gravity is the latch, and the only verbs are
slide and lift (no wrist). Guarded by `tests/test_hub_coupling.py`:

- **Measured envelope: ±4 mm lateral, −8/+6 mm vertical, < 2° yaw.** Yaw is
  the tight constraint: picks survive 2°, returns do not, ±4° jams at
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

### Robot integration
- The bumper reaches the hub before the fork does (retracted vertex 25 mm
  behind the chassis front), so the arm extends 60 mm for hub work.
- RCC droop is a fork axis: `DROOP_COMP` 8 mm, feed-forward in the lift
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

The mapper was fed MuJoCo's ground-truth depth buffer through a camera
scanner that read ONE camera's depth row — a 2D laser scan wearing a camera's
clothes, with `right_eye` vestigial. OpenCV SGBM on the actual rendered pair,
in the best case stereo will ever get (parallel, coplanar, noise-free
cameras): mid-room facing a painted wall, disparity on **49.7 %** of the scan
row at **593 mm** median error against a 50 mm grid cell; flat painted walls
are the classic no-disparity case, and the one good pose was aimed at the
rack's AprilTags. The decision (2D LIDAR + one camera, `perception/lidar.py`)
and both tables are in Parts.md, "Vision & ranging". The lesson: **a sensor
that never fails cannot teach you which behaviours depend on it** — this gap
survived seven milestones because ground-truth depth always works, the same
trap the outlet detector's val split set one layer up.

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
tempting move was to raise μ until it stuck — the widened-chamfer mistake in
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

| noslip | rig cycle | robot swap clean | pen ink | pen form | ms/step (room_hub) |
|---|---|---|---|---|---|
| 0 (was shipped) | 5/7 | 5/9 | 63 % | 1.94 mm | 0.210 |
| 1 | 5/7 | 3/9 | 72 % | 1.84 mm | 0.410 |
| 2 | — | 3/9 | 92 % | 0.60 mm | ≈0.41 |
| 3 | 6/7 | 1/9 | 93 % | 0.61 mm | 0.422 |
| **0 + wheel brake** | 5/7 | **5/9** | **99 %** | **0.60 mm** | **0.210** |

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
while a module is electrically seated. Since issue #347 the world does the
same by itself for a module lost for `LOST_TOOL_S` (300 s).

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
- **The rack merges by identity, not distance** (`RackFinder.look`). Behind
  a 0.4 m distance gate, 0.5 m of decoherence let a recovery spin's
  sightings spawn a SECOND rack landmark the stale one outvoted, so the spin
  that existed to fix the belief could not touch it.
- **The rack belief is recency-weighted** (`RACK_RECENCY` 0.25, an EMA past
  the first sightings). A mission-long average remembers the MEAN historical
  frame: a 2000-sighting belief moved by nothing when fresh looks arrived.
  Sized against what a spin delivers (~5 sightings), five looks move it ~76 %,
  enough for the bay tag to enter the dock camera's view.
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
XML). Ruled OUT:
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

## The other robot's wake walls you in (issue #167)

The first contested bay: two scripted robots sent for the LCD, and the one
that arrived first reported "no route" after 16 s with the module a metre
away. The trace showed its planner returning None from its OWN cell for
12 s: the other robot had driven past at 0.6 m, every scan painted its
body into the occupancy grid, the 7-cell inflation (0.35 m) grew each of
those cells into a 0.7 m disc, and the wake of discs covered the cell the
first robot stood on — `nearest_traversable`'s halo could not find a way out
of a ghost. A real fleet subtracts each robot's broadcast footprint from its
scans; so does this one now (`Lidar.exclude_robot`), and avoidance reads the
other robot's REPORTED pose in real time (`HubMission.others`, masked at plan
time; a stagnated drive with the other within 1.2 m waits 2 s and looks
again instead of giving up). Dropping it from the scan outright was half a
sensor too far: the same rays fed the 0.25 m front-stop reflex, which for
seven deployed days could not see the one thing in the world that moves —
382 encounters, nine `stuck` deaths, and no row anywhere saying two robots
had touched. `Lidar.scan_split` answers the two consumers separately
(issue #316), and contact is an `encounter` phase.

**What is true now:** another robot is never in the map, always in the
SENSORS, and in the mask as its reported pose -- or, lying on the floor,
as its body ("A robot lying down was avoided where it said it was",
below). Both sensors sort the same
casts by what they hit -- `Lidar.scan_split` (#316) and `DepthFrame.peers`
(#328) -- and neither hands the map a body that will have driven off by
the time anything reads it.

⚠ The two see very different things, and the lidar sees the less useful
one. Measured on the pair: the scan plane at 0.223 m crosses only the
peer's MAST (30 mm square; the chassis, fork, battery and even its own
lidar body all top out below 0.20 m), and the front stop's +/-0.35 rad
cone holds 3 rays of it at 0.9 m and NONE at all with the peer 0.25 m
across the bow -- while its 0.15 m of half-width is still wide enough to
clip. The depth camera carries 150-970 points of the same peer between
2.0 m and 0.4 m, and its answer is a corridor test on the robot's own
footprint rather than a cone. Driving past a peer 0.25 m off the bow with
NOTHING broadcast, closest approach was 0.225 m -- inside contact -- on
the lidar alone and 0.594 m with the depth channel. Below a ~0.4 m gap
the peer's near face falls inside `depth.MIN_Z` and the camera loses it,
which is why the stop fires at 0.60 m and the lidar's 0.25 m reflex stays
underneath as the floor.

⚠ The corridor is measured against WHAT IS LEFT OF THE DRIVE, not
against the camera: a robot with 0.1 m to go cannot reach a body 0.5 m
ahead. Holding for one anyway is not caution -- measured, a peer parked
0.50-0.56 m from the CHARGE standoff took that approach from 96 s and a
dock to 201 s and none, because arriving at a standoff turns the robot to
face the rack and sweeps a body it never travels into through the
corridor. A tool bay never showed it (the peer sits beside the approach
there, 0 holds at every legal distance); the charge bay did, and a charge
that does not happen is a `flat` death.

⚠ **Which half does what.** The BROADCAST (`HubMission.others`) is plan
time only: A* keeps 0.6 m clear of where the other robot SAYS it is, and a
stagnated drive waits when it says it is near. That is a fleet fact and
honest as one, and routing round a robot 3 m away is exactly what it is
good for -- but it drifts 0.24-0.55 m and it is not a perception. The
SENSORS decide in real time and need the other robot to say nothing. So a
wrong or missing broadcast now costs a detour rather than a collision,
which is the whole of issue #328. ⚠ The mask is an obstacle of
its own: a goal inside one cannot be reached at all, because the nearest
cell A* may plan to is (0.60 − d) away and a stagnated drive is only called
arrived inside 0.15 m — so a robot standing within 0.45 m of a bay standoff
makes that bay unpickable, measured 0/3 picks at 0.26 m against 3/3 at
0.56 m (issue #313, which spent a week looking like a pen fault, a planner
fault and a contention fault in turn). `HubMission.peer_on_the_goal` is
the arithmetic; since issue #346 the swap WAITS for the bay rather than
spending attempts on a point it cannot reach ("Two robots at one rack",
below). The pick lands at 50 s with the
other robot crossing its path, and the second robot's pick fails honestly
at the empty bay.

## Near-field 3D: the sensor before the map (issue #34)

The LIDAR looks along one plane at 0.223 m and the navigation camera is
blind inside 0.48 m, so "find a thing on the floor" had no sensor. The
decision (a RealSense D435 on the mast top; the runners-up and why each
lost) is Parts.md "near-field depth camera"; what the sim taught on the
way, each with what is true now:

- **A depth image is ray casts here, not a render.** `mj_multiRay` at
  ~0.6 µs a ray gives a 120 × 70 frame for ~5 ms with a geom id per pixel
  (the self-filter for free), byte-deterministic and off the GPU — the
  MSAA lesson (#110) never arises. ⚠ Its `cutoff` is a MAX DISTANCE and
  `0` tests nothing: the first timing table read "100 % hit" off arrays the
  call had never written. `perception/depth.py` passes `MAX_Z`.
- **The camera sat on its own housing's face and every ray hit the
  housing.** On the box face exactly, the model settles to a hair of pitch
  and the origin is inside the box. The lens stands 1.5 mm proud.
- **A camera at head height sees only deck.** Pitched down from the head
  (z 0.135), the top of the frame was the LIDAR body and everything steeper
  than −24° was the chassis top: nothing but the robot in the central
  column. Height buys the near field twice — the deck clears sooner and
  the datasheet's min-Z stops binding — so the unit went to the mast top
  (0.51 m), which is also over the axle (the cantilever lesson above).
- **The deck sets the near edge; the pitch sets the far one.** Measured
  30°–55°: the centre column's floor starts at 0.26–0.27 m ahead of the
  axle at every pitch ≥ 35°, and ends at 2.1 / 1.7 / 1.3 / 1.0 m at
  40 / 45 / 50 / 55°. Steeper only spends rows on the deck. 40°.
- **72 g at 0.51 m is affordable, and the numbers are recorded, not
  assumed** (`scripts/nearfield_spike.py` is the sensor; the launch probe
  was a scratch script): CoM +10.7 mm, step launch 3.9 → 6.4° at full
  throttle (never commanded — `MAX_WHEEL_ACCEL` ramps), cruise launch
  0.9 → 1.3°, braking and veer unchanged, the y-counterbalance still inside
  its 10 mm bar.
- **The stereo shadow is angular, so at sim resolution it is often
  sub-pixel.** The band a near edge hides from the right imager is
  `f·B·(1/z_near − 1/z_far)` pixels; a 30 cm post at 0.7 m against the floor
  at ITS OWN depth is 0.6 px and rightly casts nothing, against the floor
  1.6 m behind its top it is 2.2 px. A test that looked in the wrong row
  "found" no shadow. The real unit at 447 px focal length sees 1.7 px where
  the sim sees 0.24; same angle.
- **Standing still, the floor is sampled in rows ~3 cm apart at 1 m**, so a
  5 cm cube arrives as two stripes with an unmeasured cell between and a
  2 cm cube is never seen (`--find`). The map bridges ONE unmeasured cell
  in `things()` and otherwise integrates over motion; nothing invents
  samples.
- **A voxel map was measured, not assumed away.** Same frame, same window:
  the 2.5D map is 40 000 cells and 0.7 ms an update; voxels at 2 cm are
  250 000 cells and, with the free-space carving that lets a map forget a
  moved object, 111 ms as written (the 2D grid's per-sample efficiency
  would make it ~25 ms). The occupied-only voxel update is 0.2 ms — and
  cannot forget, which is the cost hiding in every "voxels are cheap" claim.
  **What is true now:** the height map is the representation; voxels are
  the answer if what is under an overhang ever matters, and nothing asks.
- **The loop builds the map before anything reads it** (stage 2). The
  sensor ticks the physics seam at 10 Hz where it is on (`near_field=`;
  served yes, tests no — a frame is ~7 ms, a minute per mission test),
  placed by dead reckoning like the grid; the draw is
  `power.DEPTH_CAMERA_W` while it runs and `energy.json` was re-measured
  with it on; the map streams as `heightmap` beside `grid`. ⚠ The
  lifecycle's own settling ticks the seam before a test gets to it: a
  direct `_near_field_step()` call inside the same period takes no frame,
  which read as "the throttle is broken" until the count was taken as a
  delta. **What is true now:** the observatory shows the floor; nothing
  that decides has read it.

## The second house and the loop (issue #215): the planner is the new cost

The home world grew from 26.5 × 12 m to 49 × 21 m -- a second house across
the street, a sidewalk band and a 3 m street loop round both, a fence round
the loop -- and the occupancy grid from 159,600 to 469,200 cells (5 cm).
Measured before the cap moved (`occupancy_grid.MAX_CELLS`, 250k → 750k):
the frontier mask stays linear (4.6 → 12.6 ms a pass), and the thing that
did not was **A\***: a plan across the whole loop went from 0.34 s to
1.26 s of pure-Python heap work, because the search area grew with the
world and nothing else in the mapping stack is written in Python per cell.
It is paid per replan, and a mission's plans are mostly short, so a day is
not four times slower; but it is the first cost in this repo that scales
with the WORLD rather than with what the robot does. **What is true now:**
`MAX_CELLS` carries the table; vectorising the planner is the lever if the
world grows again.

The reserve's worst point moved BY ROUTE, not by straight line: the
loop's east legs are the farthest from the rack as the crow flies and
among the nearest as the robot drives, because the rack is reached only
through the middle street's gate at y = 3.1. `tests/test_world_budget.py`
now routes every zone's centre and corners over a 0.25 m raster of the
compiled world (Dijkstra, walls read off the geometry that crosses the
LIDAR's beam) and asks which is farthest; a straight-line comparison would
have sized the reserve off the wrong corner by twenty metres. The
measured reserve is at `home.HOME_LOW_BATTERY_WH`.

The lab's props arrived as geometry with no behaviour behind them (the
cage's mouse a mocap body at rest, three plates lighting nothing, two
tagged mass cubes), by decision: a world change is one regime break, and
#226 and #227 add behaviour to a world that already holds what they need.
`dynamic_flags` counts a mocap body as dynamic so the mouse rides the wire
from day one.

**A bigger world moved the cameras' near plane, and the robot went blind
at the rack.** The first pricing flight on the new plan lost the pen at
its first stow, and `home_draw.py --cycles 2` -- the swap-stack check --
failed its second fetch where the old world passed both. Not the swap:
MuJoCo derives `statistic.extent` from the geometry's bounding box and
scales every camera's near clipping plane by it (`visual.map.znear`
x extent, 0.01 by default). The property's box gave 37.2 m, a 0.37 m near
plane; the loop's gave 70 m and 0.70 m -- and the dock camera at a bay
standoff, 0.34 m from the rack, clipped the whole rack out of its own
image (mean pixel brightness 94.6 -> 27.6; the lights were suspected first
and measured innocent -- 1 light or 16, the same 27.6). `bay_fix` answered
None every time, nothing raised, every pick and stow ran on the believed
standoff alone, and a blind stow put the pen on the floor. **What is true
now:** the generator writes `<statistic extent>` down (`home.CAMERA_
EXTENT_M`, the value the tag pipeline was proven at) instead of letting
the world's size choose it, and `test_the_dock_camera_decodes_a_bay_tag_
from_the_standoff` decodes a real tag through the real pipeline, with the
unpinned world kept beside it as the premise. The lesson generalises: any
world whose bounding box grows past the property's must pin its extent, or
its cameras lose whatever they look at from closer than a hundredth of it.
⚠ **A pin in the XML does not survive a runtime `mj_setConst`**, which
re-derives the extent all over again: the bench's set-out (#227) called it
on the live model, so a bench offer undid the pin for the rest of the
process, and the deployed pair's picks failed 13 of 15 with nothing raised
(issue #264; a run's first pick could still land off a belief fresh from
the start pose). `bench.set_unknown_mass` puts the statistic back;
anything else that calls `mj_setConst` on a live model must do the same.

## A goal out of sight is aimed at through the nearest wall (issue #215)

The first flown lap of the loop, as sixteen `drive_to` legs of up to 12 m,
drove thirteen and failed heading east along the north street. Not drift:
re-flown with belief logged against truth, the error stayed under 10 cm
and 0.3 degrees for 100 m, and the other robot was 20 m away. The goal,
12 m ahead, was still UNMAPPED, and `HubMission._plan_to` answers an
unmapped goal by aiming at the known-free cell nearest it by STRAIGHT
LINE. From the loop's north-west corner that cell is indoors, behind the
first house's north wall -- the hall the robot explored that morning --
and the only way there is back round the loop and in through the gate: a
463-waypoint plan that starts by driving AWAY from the goal, which the
drive's stagnation check (10 s without 2 cm of progress) rightly ends.
The old world could not show it: its street stopped at the property, so
no unmapped goal ever sat just outside a building the robot knew.

**What is true now:** the straight-line rule stands, deliberately, with
one word changed (issue #298): the nearest known-free cell OF THE ROBOT'S
OWN COMPONENT (`scipy.ndimage.label` over the traversable mask, 4-connected
like `astar`), never one it cannot reach. The same fallback is what lets a
bay or board approach reach a goal INSIDE a wall's inflation (the approach
plans to the nearest traversable cell and finishes on `drive_toward`), so
the fuller fix -- for an UNKNOWN goal, aim at the nearest cell that borders
unknown space, since driving there is what grows the map toward the goal;
for an INFLATED one, today's rule -- changes every drive whose goal is
unmapped and wants its own flights. A PROCEDURE's `drive_to` past the LIDAR's
reach or off the map walks `lifecycle.route_to`'s doorways first (issue
#353); its hops are the house's legs (up to 7.9 m, where the lab's way
meets the workshop's), and the first is as far as the robot stands from
it. Expect the same stall from an
`explore(zone)` decision aimed at a loop zone the robot has not seen: it
drives toward a wall, stops, and explores from there. That is the rover; the
quadruped plans through the unknown instead ("Walking into the unknown").

## ...and then the bed was standing in its way (#305)

The same board, the half #298 did not reach. With the planner fixed the
robot drives the whole way and stops 0.26 m from the use pose, and there
it bounces: the LIDAR's front-stop reflex reverses it 0.8 s whenever
something within `FRONT_STOP_RANGE` (0.25 m of the scanner) lies dead
ahead, and the bed's north-east corner sat 0.23 m from whiteboard_b's use
pose. Back off, come back, trip again, until the drive stagnates after
10 s and the errand stows a pen that never drew. MEASURED: 13 of 13 trips
on a flight from the hall spawn ranged the same world point, (0.42, 5.41)
-- the bed's corner -- while `_pressing` never fired once, so this is the
lidar reflex and not the bumper.

⚠ It depended on the APPROACH ANGLE, which is why the board had ink on it
and a history of failures at the same time: the corner has to swing
through the ±20° front cone. From the living-room spawn, whose straight
line to the board threads the bedroom doorway, the same errand drew
fine -- which is how #298's flown proof passed while the deployed pair,
coming from the rack and the hall, failed three times in a row.

Two things were tried and only one works. **Gating the reflex on forward
rolling** -- the bumper's own rule, "a pure spin is heading, which the
gyro owns" -- was implemented and flown: no change, because reaching a
pose 0.23 m from an obstacle means driving forward at it, which is
exactly what the reflex is for. **Moving the furniture** is the fix: the
bed is in the bedroom's north-west corner now, 1 cm off both walls, and
its corner is 0.62 m from the use pose. The identical flight arrives at
85.6 s and draws 9 of 9 strokes at 2.05 mm.

**What is true now:** a board's use pose must have `FRONT_STOP_RANGE +
LIDAR_ORIGIN[0]` = 0.35 m of clearance from any furniture, because the
robot squares up there and the scanner rides 0.10 m ahead of the axle.
`test_home_world.py::test_a_board_can_be_stood_in_front_of` is the bar
and fails at the old position with the number. The walls and the board
are exempt: the standoff is measured TO them and the robot arrives
facing them on purpose.

## whiteboard_b was aimed at through a one-cell island (issue #298)

The deployed pair paid nobody for a drawing on `whiteboard_b` in thirty
hours -- 25 correct answer claims, 0 paid -- and it was not the pair:
`home_draw.py --board whiteboard_b`, one robot, read `SWAP_PICK done` and
`USE_TOOL: never got there` in the same second. The board is in the
bedroom, seen from the start pose only through the 1 m divider doorway,
and what the LIDAR paints through a doorway is a wedge of free cells with
ragged edges. The use pose (0.45, 5.62) sits in the north wall's
inflation, so `_plan_to` aimed at the known-free cell nearest it by
straight line: a one-cell island at the wedge's edge, one cell nearer
than the wedge itself and joined to nothing. `astar` answered None and
the drive gave up in 0 s; every replan found the same island. On the
deployed world the map was empty at every hourly restart (until #345) and the mind
takes a job before it explores, so the bedroom was never mapped and the
board never reached. Aimed at the nearest cell of the robot's own
component instead, the same flight arrives at 80 s and draws 7 of 7
strokes at 1.74 mm form error. `test_navigation.py` pins the rule on a
synthetic grid: a corridor, an unconnected island nearer the goal, and a
plan that stays in the corridor.

## The tag camera spent 97 % of its render on shadows (rooftop-media-2026 #296)

The served pair stuttered: the page played 50 ms of motion and froze ~170
ms, over and over. Not the network. The deployed log's wall stamps against
its sim clock read **0.23× real time**, the sim container sat at 306 % of
four cores, and inside it the eight `llvmpipe` threads (Mesa's software
rasteriser behind `MUJOCO_GL=osmesa`) held ~80 % of all CPU ever spent
while the physics thread used 47 % of one core -- waiting on renders. The
site's clock advances at 1× and clamps to the newest frame, so a stream at
0.23× is a frame's worth of motion and then a wait for the next.

A wrong turn first, worth keeping: timing renders inside that container
found "fast" contexts at 2-35 ms beside slow ones at 550-1900 ms, and the
pattern looked like "a second GL context makes every render slow". It was
not: the fast ones were contexts whose shadow framebuffer had failed to
create after another context was closed, and MuJoCo then renders silently
without shadows. A lone renderer, toggling the scene flags, gave the real
number: **one 1280×720 frame of the home world costs 1113 ms with shadows
and 32 ms without** (reflections 32 → 28; the skybox nothing). The home
world has sixteen lights, every one `castshadow`, and MuJoCo's default
`shadowsize` is 4096 -- sixteen 16-megapixel depth passes per look, at
3-4 looks a sim-second per robot, on a software rasteriser. A shared
renderer (the fix for the wrong diagnosis) changed the multiple not at
all; the flag took the same pair from 0.20× to **0.54×** on the deploy
box, with the container down to one core: the physics thread, which is
where a pair's cost actually is (0.58× on the GPU dev box).

**What is true now:** `TagDetector` clears `mjRND_SHADOW` and
`mjRND_REFLECTION` on its scene once at construction, and `update_scene`
keeps the flags (`tests/test_render_context.py`). A tag decode thresholds
gray levels and a real camera sees no shadow pass, so nothing hardware
has is lost; the viewer and every filmstrip keep their shadows, since this
is the detector's scene alone. Two things ride with it: the rack finder is
rebound on a recompile like everything else (it rendered the old model
until now), and the site's clock follows the stream's measured pace, so a
world that cannot hold real time -- a pair, on one physics thread --
plays smoothly slow rather than stop-and-go, with the pace shown beside
"live". The entry below is where the remaining half went.

## The physics thread was four-fifths bookkeeping (rooftop-media-2026 #296, the profile)

With the tag camera's shadows off (the entry above) the served pair still
ran at 0.5×, on one core, so the thread itself was profiled: `py-spy
record` at 100 Hz over a 130-sim-second pair day inside the deployed
image (414 s of samples). Self time, by frame:

| share | frame | what it was |
|---|---|---|
| 26.3 % | `coupling.module_power_state` | the tool's electrical criterion |
| 12.8 % | `tick.step` | `mj_step` — the physics |
| 9.8 % | `scipy._binary_erosion` | the map's inflation, per replan |
| 9.6 % | `coupling.rack_charge_contact` | the charge pins' criterion |
| 8.9 % | `depth.frame` | the depth camera, 10 Hz per robot |
| 7.2 % | `swap._pressing` | the bumper |
| 5.7 % | `mission._after_step` | the collision count |
| 3.1 % | `render` | the tag camera (was ~80 %) |

Four of the top seven were Python loops over `data.contact[i]` — a pybind
struct per contact, 83 contacts at rest in the home world, 2 ms steps,
two robots — run EVERY step: ~330 000 struct constructions a sim-second,
49 % of the thread against 13 % in the physics. Each reader now answers
off `data.contact.geom`, the (ncon, 2) array view, in one expression, and
resolves a geom's id by name once per model (`coupling.geom_id`; the
string lookup was paid twice a step). The inflation was
`binary_dilation(occupied, iterations=7)` over the 49 × 21 m grid on
every 2-second replan, ~300 ms a call because scipy re-walks the array
seven times; the same taxicab ball is one chamfer distance transform
(`distance_transform_cdt`), two passes, and `tests/test_frontier.py` pins
the masks identical on random grids.

**What is true now:** the readers give the answers the loops gave
(`tests/test_contact_reads.py` runs both on a live world), a scripted
day hashes IDENTICAL before and after (`determinism_spike --compare`),
and the pair's physics thread is mostly physics. What is left and
deliberate: the depth camera's 8 400 rays at 10 Hz per robot (a sensor's
honest rate, Parts.md), the lidar, and the decode.

## A trip across the street drifts the reckoning a quarter of a metre, and the dock absorbs it (issue #226)

The mouse's acts are the first errands that end 25 m from the rack. Flown
from the rack after the world's explore, with belief logged against truth
at every leg: the error grew ~3 cm a leg on the way out -- 0.06 m at the
garden doorway, 0.24 m in the lab -- a heading bias of about half a
degree over 25 m of mostly straight driving, and the plate itself added
nothing measurable (0.19 -> 0.24 m across the run onto it and off). On the
way back it kept growing in the same direction, to 0.45 m in the living
room and 0.55 m at the pins. `go_charge` from the lab docked through it:
one `drive_to` to the standoff (121 s), the measured approach off the
charge tag, pins connected. A `drive_to` to the spawn pose first and
`go_charge` from there once answered "no route to the charge bay"
(twice, with a spin between) -- the same drift from a different heading,
and the one flight where the standoff was unplannable; not reproduced
after the spike went home by the dock instead.

**What is true now:** a far errand ends where its act is and the return is
`go_charge`'s (`cage_program`; `energy_spike.py` docks between cage rows),
because the dock is the anchor and one long approach through the drift
has measured better than a stop on the way. Nothing corrects the heading
mid-trip; a day of trips is what the observatory period reads for
(Observatory.md, "The mouse"). The plate is driven THROUGH, not parked
on (#287): parked on the believed centre, the press was the reckoning's
-- the same route flown twice drifted 0.10 m (pad pressed 11 mm, three
rising edges) and 0.41 m (a wheel on the pad's edge, 5.6 mm against the
6 mm trigger, nothing registered), and the deployed world's shock landed
on 2 of 11 jobs. A pass from 0.8 m south to 0.3 m north and back crosses
the pad for any longitudinal drift in (-0.6, +0.5) m, and the map, built
in the same drifted frame, keeps the chassis its inflation off the cage.
A wheel on a 400 mm pad still needs no controller of its own; it needs
not to be asked to stop on it.

## The lift is a scale (issue #227)

The bench asks for the mass of a cube nobody told the robot, and the issue
asked for the honest sensor: what a real lead screw reports is its load.
Probed before anything was designed round it: the claw on the fork, a
26 mm block in its jaws, lifted 50 mm and settled, and `data.actuator_
force[lift]` read while the block's `body_mass` was set to five values in
turn (with the inertia scaled and `mj_setConst` run on a SCRATCH MjData --
it writes `qpos0` into whatever data it is handed). The force moved by
`dm · g` to 1 mN at every step of 0.05..0.40 kg; the empty claw read
6.40 N and the jaws held 0.40 kg. So the position servo on a damped slide
with no `frictionloss` is an exact scale at rest, and everything a real
reading has that this does not is added on purpose in the sensor: a load
cell's noise (`axes.LOAD_NOISE_N`), deterministic per physics step so one
world replays and a `for` loop of `read` calls without a `wait` between
them reads one sample, as a sensor polled faster than it updates does.

What is true now: `read("lift.force")` is `actuator_force + noise`; the
tare is the fork's own weight and the robot has to subtract it or
calibrate against the known cube; a mass set at runtime goes on the model
AND the spec (the workshop's recompile rebuilds from the spec, and a
runtime-only mass reverted to the world file's placeholder in the probe).

## The tower had never been stacked, and the claw could always do it (issue #264)

Every passing tower test `stack.place()`d the blocks and tested the grader,
so nobody knew whether a 26 mm cube picked, carried and released onto
another would stand. Probed first with the claw's own routines from TRUE
poses (`drive_over`, `pick_up`, a lowering to one pitch above the base, the
jaws opened): 2.6 and 5.9 mm of lean, verdict ok with the hold. The physics
was never the problem. The next three weeks of the evening were the
language's reach to the claw, and each wall was a measurement:

- **A 20 mm tag is ~24 px wide from the dock eye at 0.8 m.** It decodes in
  a band around 0.65-1.0 m and patchily inside it (which lift sees it
  changes with the range by centimetres), and never inside ~0.65 m -- not
  the fork prongs, the frame's bottom edge: the eye looks level and the
  floor leaves the picture. A verb that needs a decode hunts lifts and backs
  off 0.25 m at a time out of the blind zone.
- **PnP's range to a tag that small is quantised by the pixel.** Half a
  pixel of 24 is 2 %, and it read as ±10-15 mm of range scatter between
  looks -- fitted as a 1.1 % scale first, which flipped sign on the next
  sample. The tag's centre PIXEL is good to a millimetre, so the position
  is the ray through it cut at the cube's known layer height
  (`HubMission.spot(at_height=)`): 2 mm. In the WORLD frame: a module on
  the fork pitches the chassis 0.1 deg, which is 7 mm of range at that
  geometry.
- **Off-axis, the same decode is 25 mm long.** A verb stages itself to look
  head-on from 0.55 m short of the cube before it approaches.
- **The grip offset off the belief bakes in the drift.** `ClawTool.
  calibrate` measures the grip site against the reckoner; after 1.2 m of
  driving that put a block 43 mm long. Off the BODY instead -- and at the
  DEPLOYED reach: the mission tucks the arm to drive, and an offset read
  tucked put the grip 60 mm long once the arm came out.
- **The held cube is not where the grip is, and it moves.** It hangs 8 mm
  behind and 18 mm below the grip point at the pick, then slips ~7 mm down
  and ~10 mm along the pads over a carry's turns. A release aimed at the
  grip point pressed it into the base and shoved the base 24 mm; a hang
  measured at the pick left a 10 mm aim error. `held_hang` is re-read on
  arrival, the along-track difference crept out, the release height set
  off it (`PLACE_GAP` 4 mm above the surface).
- **At carry height with the arm out the claw sits in the lidar's cone.**
  The module's body is 8 cm from the lidar; at `APPROACH_LIFT` with the arm
  in it crosses the scan plane inside the ±20 deg front-stop window, and
  `drive_to` backed the robot away from itself for as long as it was asked
  to drive. 36 mm higher -- the swap's own carry lift, where every mission
  leg already drives -- it clears. `tuck_routine`.
- **The sidestep the language can write moves the axle a centimetre.** A
  `face`, a `drive`, a `face` back: the caster swings and the axle lands
  ~1 cm off, and no sensor a procedure can read says where. That, with the
  decode band, is why the motor-level procedure topped out at two layers
  and the claw got `pick`/`place` at `fetch`/`stow`'s level.

Three navigation traps met on the way and routed around, not fixed: the
workshop's spawn point is its table; the garden doorway leg of the lab
route stalls from a cold start (the plan hugs the wall and the door post
trips the reflex -- the leg now ends 0.6 m short); the planner's own way
back from the street hugs the house wall past the garden light's pole and
stalls there (the bench solution returns through the open garden). The
auto-stow that "could not return from the workshop corner" was the cone
bug above: tucked, the errand's own stow brings the claw home from there
(bay error 0.3 mm); from the lab the solution drives the road in legs.

And one more, from ladder B's third day: a model wrote `fetch; pick(20);
place(21); pick(22); place(20); stow` and `pick(20)` failed at the rack --
the cube was 15 m away and a single `drive_to` across the house stalls in
the hall at 41 s (the unmapped-goal defect). So a cube out of view is
looked for where the house set it out: route legs to the zone, a stand on
the room's open side, one more look.

What is true now: `solutions.TOWER` -- those six lines, verbatim -- passes
`eval_stack_tower` from the rack (489 sim s, 2.7 Wh, 5.3 mm of lean);
`solutions.WEIGH` weighs the bench's cube to 2 %; the feed act lands.
`tests/test_solutions.py` pins each rule above in milliseconds and flies
the three behind `--endurance`.

## A robot on its side maps the sky (issue #339)

After a topple on the deployed pair (build `0f2faf5`), Rowan's live map
(the stream's `grid`) had its hall -- where it stands up -- painted solid
occupied with concentric floor-hit arcs, and a free fan running east through
the living room's walls and out past the fence, its outer edge an 8 m arc
about where it had fallen. Luca's map, off the same stream, was sane.

Three rules met there. The LIDAR casts along its real frame, and no return
is "free to max range", so on its side about half its rays see the sky and
go into the map as 8 m of free space at the believed, upright bearings
(reproduced: 5 s on its side painted 38 622 free cells onto an empty map,
straight through the walls). A dead robot keeps scanning, because its wait
steps through `_drive_routine` and every step runs `_after_step`. And a
stand-up warps and refills but keeps the map. **What is true now:** a scan
goes into the map only while `HubMission.level()`, within `MAP_TILT_RAD`
(1.5°) of level. Past 1.6° the scan plane meets the floor inside the 8 m
range (the LIDAR sits 0.223 m up); draw, census and dance tilt the chassis
0.66° at most and the charge creep's bumper contact spikes to 1.4-1.7° for
about 20 ms (one scan skipped per dock; the held press is under 0.1°),
measured, while crossing a 21 mm
plate pad tilts it 4.3-8.1° for about 2 s -- those scans are skipped, and
at that tilt they had been painting floor-hit arcs 1.6-3 m out on every
crossing. The rack finder's sightings and the near-field height map take
the same gate: on its side 1-2 m from the rack, a robot's camera had moved
the rack belief 0.1-1.4 m. The reflex still reads
every scan, and a deploy clears a map already damaged: the grid lives in
memory only.

⚠ **It was not the death loop it was found beside.** Rowan went `flat`
401-403 s into every life after that topple, pushing into `wall_west_1` at
~70 W, and the map looked like the reason. It was not: the same death loop
reproduces WITH the gate in place, and replaying Rowan's own map from its
stand-up pose explores the house without touching the wall. The loop was
`refine_standoff`'s unbounded drive back to a bay standoff (#339, the
budget PR). Trust the reproduction over the story that fits.

## A rack end is not symmetric for the robot (issue #277)

The built-tool rail is the first rack's frame emitter run again beside it,
and its posts were first placed as the rack's E-end is: 0.055 m past the
outer bays on both sides. Bays B and C of the rail then fetched and stowed
5/5 on the first flight in home and bay A failed twice at the pick,
"arrived" and unpowered, the module nudged 9.6 mm along its trays. A
contact log of the robot against both racks during the creep named it:
`caster` on `rack_built_foot_r` for 10 731 steps. The fork line rides
`PLUG_LATERAL` (0.05 m) to one side of the chassis, so the robot's
centreline -- and its caster, a 20 mm sphere 18 cm ahead of the axle on
that line -- arrives at `station - 0.05`, and a foot 0.055 to THAT side is
directly under it: the caster climbed the 20 mm foot, the fork rose with
it and its stop met the tray. Bay E has always worked because its post is
on the other side, 0.105 from the centreline, and the rail's bay C works
for the same reason. Nothing in the swap or the mission was wrong; the
rail's near end was.

What is true now: `BUILT_RACK_HALF_W` is not half the bays' span -- the
near post is 0.155 from bay A (the caster clears the foot by 0.07, E's
margin) and the far post E's 0.055 past bay C; a rail end on the fork's
side of a bay needs the extra 0.10, and a probe that places the robot with
`place_at_standoff` says nothing about a room world (it is rack-local
coordinates for the bare hub world, and the pen fails from it in home
too). Also learned on the way: stepping the physics with raw `mj_step`
between two of the lifecycle's routines moves the robot without the
reckoner seeing it -- 500 steps with the opening spin's wheel command
still in `ctrl` put 49° between belief and truth, painted a diagonal
ghost wall across the living room, and looked exactly like a seam bug.

## A loop with no budget outlived the robot (issue #339)

On the deployed pair (build `0f2faf5`) Rowan was knocked over mid-pick, and
every life after that ended `flat` 401-403 s after its stand-up, silent,
pushing into `wall_west_1` at ~70 W. `refine_standoff`'s drive back to a
bay standoff looped `while` it was more than 5 cm away -- no budget, no
stall check, no planner. On its side the robot could never arrive, so the
loop outlived the death; the timer stood it up at the far end of the house
(its rule waits only for a SEATED module, #311), and the same loop drove it
straight at the rack's standoff until the pack was flat, then resumed after
every stand-up. Found by walking the day routine's generator chain
(`gi_yieldfrom`) in a reproduction: `refine_standoff_routine` both while
dead and after the stand-up. The robot's map, corrupted by the same topple
(#340), looked like the reason and was not -- the loop reproduces with the
map fixed. **What is true now:** `REFINE_BUDGET_S` (10 s; healthy passes
measured 0.97-1.25 s). A robot stood up mid-errand still finishes the errand
it died in, in bounded passes: in the reproduction, about 70 s of it, and
91 % of the pack left. A budget that runs out is `refine_blocked`, and the
swap and the charge approach take no attempt from where it stopped: out of
line, a deployed fork or a creep is the knocked-off module the refine
exists to prevent (second review).

## Two robots at one rack (issue #346)

On build `42f4a11` the pair's rack was the biggest single cause of failed
jobs and of every `flat` death: 6 of 47 picks, 7 of 38 stows and 11 of 30
charge approaches were refused because the other robot stood on the
standoff, and a further 6 charges failed "no-tag". Three things combined:
a robot said it was clearing the rack and never checked it had left
(`_clear_rack_routine` threw the drive's answer away); a bay somebody stood
on was given up at once (#313's early return); and nothing moved a robot
from the rack between a swap or a charge and its next decision.

Two explanations were measured, not guessed:

- **"no-tag" is a look from BESIDE the standoff.** With the other robot at
  every tool bay's standoff and 0.5 m out in the charge lane, the charge
  tag decoded from the charge standoff every time: the other robot never
  hides it (at camera height the line is clear even at bay B, 0.200 m
  off). Held 0.3-0.6 m off the standoff and facing the rack, the tag
  decodes only within ±30° of the bay's axis (room_hub; ±30-60° in the
  home world). 60-90° along the rack face is out of the camera's 41°, and
  that is exactly where the other robot's 0.60 m planner disc leaves a
  robot when the other stands at the neighbouring bay. The live case
  (2026-09-25, `a803c83`, Rowan at t=2198) started its approach while Luca
  picked the LCD one bay over.
- **The 60 s "never got there" is the other robot standing at the
  neighbouring bay.** A robot at bay A's standoff after a pick, driving to
  the world's use pose with near-field on, as deployed:

  | the other robot | room_hub | home |
  |---|---|---|
  | far away | arrived, 13.2 s | arrived, 15.2 s |
  | at the charge standoff, or 0.7 m out in bay A's lane | arrived, 14.1 / 16.7 s | — |
  | at bay B's standoff (0.25 m off) | **60.0 s, 2.96 m short** | **60.0 s, 3.41 m short** |
  | at the new waiting spot | arrived, 13.2 s | arrived, 15.2 s |

  Each failure is one peer HOLD (#328's corridor) and then the stagnation
  wait (`OTHER_WAIT_S`) until the errand's 60 s ran out.

How long a swap holds a bay, off the believed pose within the 0.45 m of
`peer_on_the_goal`, one flight each world on the hosting pack: room_hub
carry pick 26.9 s and stow 34.2 s; home pen pick 30.6 s, and a stow run
straight into the next pick 55.6 s. A charge is the deployed figure: 278-542
s, median 462 s.

**What is true now:** after a swap, a charge or a failed pick, a robot with
somebody else in the world leaves the rack before it decides
(`_leave_rack_routine`), and checks that it got there: a clear that ended
inside `RACK_CLEAR_M` tries `CLEAR_SPOTS` other places and says in History
where it ended. A robot that stays within the rack's reach for
`RACK_LINGER_S` doing nothing there is logged `RACK: lingering`. A taken
bay, the charge bay included, is waited for (`HubMission.bay_wait` ->
`HubLifecycle._await_bay_routine`) from a spot beside and behind the
holder's lane, or from the start pose when the map is too thin there, for
3× the typical occupancy of what holds it (`SWAP_OCCUPANCY_S` 30,
`CHARGE_OCCUPANCY_S` 462; a charge holds the neighbouring tool bay too).
The charge approach waits again before each look, and faces the rack
before it looks. Only a pick's wait ends early, on the robot's own
interrupt or at the reserve; a return's and a charge's run to the bound. A return that failed is tried again, `STOW_RETRIES` times, before
anything but a charge. A failed charge approach logs `charge_trace`: per
look, fix or none, the belief's drift, the distance from the standoff, the
other robot's distance and what the camera's line to the tag meets first.

## A drawing that set off from the rack (issue #347)

Rowan's `pen_check` (`fetch("module_pen")`, in one version three `move`s,
then `draw(..., "whiteboard_b")`) reached `draw` six times on the deployed
pair and was knocked over all six times, the pen left on the floor; the pen
was eight of the nine tools reset by hand that week. The issue blamed the
pose: the lift at 0.15, the arm at 0.10 and the carriage at 0.03 when `draw`
drove off. Flown locally from the start pose, BOTH versions toppled the
same way -- 94 deg, 14 s into the draw, at (0.4, 2.3), the pen at
(0.46, 1.73), 3.7 m from its bay as live -- and three of the six live runs
had carried the pen in its carrying pose. The cause was the route. A
procedure's `draw` ran the drawing errand's use-phase, whose
`drive_to_board_routine` is a straight `drive_toward` at the board with no
planner, meant to settle from `use_at` after the native errand's A* carry
drive; from the rack it drove at whiteboard_b through the house. With the
carry drive first both versions draw at 98 % and stow, with the carrying
pose switched off as well. Measured on the way, in the pose itself: a claw
that let go of a cube at a 0.033 m lift and then drew its arm in came off
its seat, the module 114 mm down the fork and unpowered, and a stow drove
that to the rack; lifting first keeps it seated (the cube, between open
jaws, moved 0 mm either way). **What is true now:** `draw` takes the
planner to its board's `use_at` first and says "never reached" if it
cannot; every verb that moves the base puts the fork into its carrying pose
first (`steps.run_verb`), which is a rule of its own and was not what
toppled Rowan; that pose and the return's (`carry_configuration_routine`)
move the lift up before the arm comes in and down after it; and a module
lost for `LOST_TOOL_S` goes back to its bay by itself.

## A robot lying down was avoided where it said it was (issue #365)

On 2026-09-23 at 18:02 UTC Rowan drove into Luca, who had lain on its side
for three minutes after toppling mid-fetch at the rack. It bumped Luca for
about a minute (the encounter rows close from 0.51 m to 0.06 m) and fell
over too. The build was `e77704c`: the lidar saw the other robot (#316),
and the depth camera's peer channel (#328) had merged 13 minutes earlier
and was not deployed. Each guard, measured on room_hub with one robot
toppled at a fixed pose and the other sent past it:

- **The planner's disc followed the belief, and the belief had left the
  body.** The errand a robot falls in keeps commanding its wheels until it
  returns, and a wheel turning in the air is travel to the reckoner: up
  to 2.2 m in 10 s, by how it lies and which way the wheels are told to
  turn. On its side, 10 s of cruise put the reported pose 1.5-2.2 m off
  the body; face-down, cruising moved it not at all and reversing 1.3 m.
  A dead peer stayed in `others`; nothing filtered it. It was just in the
  wrong place.
- **A lying robot reaches further.** Its geoms' bounding circles reach
  0.51-0.52 m from the chassis origin (the mast lies along the floor),
  against 0.30 m upright. From the middle of its footprint they reach
  0.32-0.33 m, against a standing robot's 0.27 m armed.
- **The lidar sees it in most poses**: 1-27 returns off the body, but on
  its left side none beyond 0.7 m.
- **The depth camera's peer channel sees it well**: 120-590 points at
  0.5-1.2 m in every pose, and the corridor catches it inside 0.60 m.

Flown from (-1.6, 0) to (1.3, 0) past a robot lying at (-0.3, 0) or 0.25 m
off that line, its reckoner 1.8 m off its body; four falls x two offsets:

| guards | arrived | touched it |
|---|---|---|
| 09-23's (no peer channel), disc on the belief | 3/8 | 4/8, up to 1313 contact steps; one driver tilted 9°; the five that failed backed off 8-16 times |
| today's (peer channel), disc on the belief | 0/8 | 0/8 -- held ~0.6 m short until the drive gave up |
| disc on the body, no peer channel | 8/8 | 0/8 |
| disc on the body, peer channel holding for it | 6/8 | 0/8 -- face-down, both offsets, held 0.52 m short |
| disc on the body, not held for | 8/8 | 0/8 |

The face-down hold is geometry, not noise. The detour runs straight at the
body and turns at the disc's edge, and the corridor looks 0.60 m ahead, so
it sees the body just before the turn. A hold waits for the other robot to
move, and a robot on the floor will not move until it is stood up.

**What is true now:** a robot lying down is avoided where its body lies.
"Lying down" means the chassis past `TOPPLE_TILT_RAD`, from the moment it
falls, dead or not. `HubLifecycle.keep_clear` answers the middle of its
footprint with `DOWN_ROBOT_CELLS` (0.70 m), placed where the DRIVER's own
sensors would put it (`HubMission.as_seen`), so the driver's own drift
cancels. That matters: 0.37 m of floor is all there is between the detour
and the body, and the pair's drifts ran 0.24-0.55 m. A fallen robot cannot
report itself, and a real robot would see it as a lump in a depth image.
The depth camera does not hold for a robot lying down, and a stagnated
drive does not wait on one (`_other_in_the_way`); the lidar's front stop
and the bumper still see it. Because the hold no longer covers the moment
of a fall, a drive looks at who is lying down every `DOWN_CHECK_S` (0.1 s)
and replans at once when that changes. Measured without it, a robot
knocked flat 0.5 m ahead just after a replan was met only by the lidar's
0.25 m stop, the driver's axle 0.18 m from the body; with it, the replan
comes 0.04-0.1 s after the fall and the axle keeps 0.31 m or more. Once
stood up (a warp that resets the reckoner), it is avoided where it says it
is again, and the plan hears that at once too. A line about a robot on the
floor says it is "lying knocked over" (`posture`), never "standing". Not
changed: the mind is still shown the reported pose (`others_context`), and
a fallen robot's reckoner still counts its wheels.

## The quadruped body (issue #377)

The body was sized before anything was trained, on a scripted gait
(`legs/scripted.py`: stance feet push, tau = -J^T f, for the torso's height,
attitude and velocity; swing feet follow a Raibert placement under a
Cartesian PD). It is a measuring instrument, never the robot's gait, and it
can under-read a policy that stamps harder, so the tables state margins.
`scripts/quad_spike.py` flies every table below (`--torque`, `--sweep`,
`--thermal`, `--energy`, `--pupper`); Parts.md, "The quadruped body", has
the parts and the sources.

**The torque table** (the chosen body, 9.34 kg, a 43.2 V pack; "p99.5" is the
99.5th percentile of a joint's |torque| over every sample of all four legs,
because a touchdown spike lasts a step and says more about the footfall
than the load; the peak is 22 N·m, the continuous rating 6.71):

| activity | knee p99.5 | knee RMS | fastest joint, % of no-load |
|---|---|---|---|
| stand | 3.3 | 3.2 | 0 |
| stand, #378's arm at full reach | 4.1 | 4.1 | 0 |
| deep crouch (0.15 m) | 4.8 | 4.8 | 0 |
| walk (a trot at 0.3 m/s) | 7.1 | 4.7 | 43 |
| trot 1.0 m/s | 8.4 | 4.8 | 69 |
| trot 1.5 m/s | 10.2 | 5.2 | 90 |
| push up a 0.18 m riser | 5.1 | 3.8 | 7 |

The knee is always the worst joint. **Speed binds before torque**: a 1.5
m/s trot takes a joint to 90 % of its no-load speed on a nominal pack and to
100 % on an empty one (`--bus 36`), so the body's top speed is ~1.0–1.2 m/s.

**What the instrument got wrong first, each found by filming it:**
- *The attitude loop ran about the world's axes.* Roll and pitch are the
  HEADING's, and past 90° of heading the correction reverses: a turn on the
  spot flipped the body at 140°. The loop now runs in the heading frame.
- *A foot placed for the torso's velocity lands behind a turning hip.* The
  Raibert target uses the hip's own velocity (w × r).
- *Height and pitch referenced to the ground under the hips* jump a whole
  riser in one step as a hip crosses an edge, and the body flipped
  backwards; they follow the ground the FEET stand on, low-passed.
- *A scripted climb is chaotic*: 4 of 18 climbs across six riser heights,
  the failures landing a foot on an edge or trailing the hind legs. So the
  table measures a climb's LOAD by a repeatable push up a riser (the front
  feet on the step, the hind on the floor, the torso crouched and pitched,
  risen to the stand in 0.6 s); the policy's own climbs are its measurement.
- *A four-beat walk falls* without a body sway to keep the CoM inside three
  feet; "walk" in the tables is a trot at 0.3 m/s, as a learned policy's is.

**The sweep that chose it** (`--sweep`: knee belt 1.0/1.5 × legs 0.19/0.21/
0.23 m × the unpublished rotor inertia at both ends of its range): a direct
knee on 0.21 m links. A 1.5:1 belt buys torque the knee does not need (its
p99.5 is 54–55 % of the peak at 1:1 across the inertia range, and its RMS at
most 86 % of the continuous rating at the heaviest rotor) and costs the one
thing that binds, knee speed, while multiplying the knee's reflected inertia
by 2.25. The 0.19 m legs fell once at the heavy rotor; 0.23 m buys nothing.

**The belly.** With a 0.21 m leg and the knee at its −2.75 rad stop, a
folded leg holds the hips 0.10 m up, so a torso whose underside is 0.055 m
below the hips cannot rest on it: the legs carry the robot "lying down".
The battery hangs below the torso as a belly pack, 0.105 m under the hip
axis; the robot lies on it with its shanks flat and its drivers holding
nothing (`model.lie_qpos`), and its charge pads are flush with its
underside (#378, "The quadruped's dock"). The belly carries 67 of the
robot's 92 N on a bare floor: the rest is the legs' own weight (a hip's
two-motor stack is 0.8 kg) resting on their feet. Lying down and standing
up take 2.2 s each (`--energy`).

**Heat** (`--thermal`; the Mini Cheetah actuator's measured 1.23 K/W and
32 J/K, a 39 s time constant): a first-order winding driven by non-negative
heat never passes the highest steady state among the activities it runs
through, so each activity SUSTAINED bounds any day. The hottest winding is
a knee's: 66 °C on a 40 °C day at a sustained 1.5 m/s trot, 24 K under the
GDS68's 90 °C alarm; standing, 50 °C.

**Energy** (`--energy`; windings 1.5·R·I², shaft work counted only when
positive, no credit for regeneration):

| state | W | of which windings |
|---|---|---|
| lying (drivers powered, holding nothing) | 14.9 | 0 |
| standing | 53.9 | 38.9 |
| walking (0.3 m/s) | 116 | 85 |
| trotting 1.0 m/s | 180 | 110 |

The windings dominate everything below a trot: standing costs 39 W more
than lying, and a stand-up plus a lie-down cost 93 mWh together, so **any
wait longer than 8.6 s is cheaper lying down**. The mind takes 5–40 s a
decision (Overseer.md §6), so a body that stands through its waits spends
most of its idle power on holding itself up. The 194 Wh pack is ~13 h
lying, ~3.6 h standing, ~1.7 h walking.

**The rest posture, proposed** (#377 item 5; the decision is Ben's, at the
first quadruped deploy): **code lies the robot down, and the mind is not
asked.** A body left standing through its mind's silences bleeds 39 W, ~20 %
of the pack an hour of waiting, so if lying down were the agent's to find,
surviving would require discovering a posture, and valuing survival and
knowing the body's trick would look the same — the forcing function
PluggyPlan's principles rule out. The rover's parking brake is the
precedent: a body reflex that decides nothing the mind decides. The reflex
lies down after `T_REST` without a motion command (the break-even, ~9 s)
and stands up before the next one (2.2 s, 44 mWh), and it does NOT choose
what to do, refuse an act, or hide itself: the posture rides the wire as a
fact, and the stand-up is part of every errand's measured cost. Open: the
agent may also be given a way to hold a stand (to watch a door), as a power
on `autonomous` — that adds a choice without forcing one.

**The small body** (`--pupper`): a Pupper-v3-class body (its published
geometry and actuator, 3.0 kg) stands with a 55 % RMS margin and walks at
0.3 m/s. Carrying the suite (4.4 kg) it spends 92 % of its continuous
torque standing, 123 % holding the arm out, and cannot stand up from its
belly or push up the 0.12 m curb; its whole leg is shorter than the riser.

**The walking policy** (`training/`, mjlab 1.5.3 on MuJoCo 3.10; the first
one is `models/quadruped_policy.npz`, 98 M steps on 2048 envs in 66 min on
the GTX 1660 Super): trained on mjlab's 5 ms physics with its DC-motor
actuator, and FLOWN in ours (2 ms, `legs/policy.py`: the policy in numpy,
the driver's PD and envelope in MuJoCo's `dcmotor` since #385, which
re-flew these numbers within 0.01 m/s, 0.3 N·m and 3 % of the power) by
`quad_spike.py --policy`. It tracks 0.5 and 1.0 m/s at 0.54 and 1.07,
turns 0.78 of 0.8 rad/s, sidesteps 0.22 of 0.3, and never falls; its
knee's p99.5 is 7.9 N·m and its worst RMS 4.8 (the scripted trot's were
8.4 and 4.8 at 1.0 m/s), it draws 88 W at 0.5 m/s
and 109 W at 1.0 m/s against the scripted trot's 116 and 180, and it
tilts the torso under 0.4°. The exported file is checked against its ONNX
source before it is written (2e-6), and a flight hashes IDENTICAL in two
processes, at one BLAS thread or six (`--determinism`). The actor never
sees the base's linear velocity; what no datasheet gives is randomised
(rotor inertia, joint friction, a 0-20 ms command delay, effort limits,
mass and CoM for the arm).

**Getting up** (`models/quadruped_getup.npz`, `Pluggy-Quad-Getup`, a policy
of its own, trained on a rented GPU with `training/pod.sh`): each episode
starts lying on the belly pack or dropped from 0.35-0.55 m in a random
orientation, legs anywhere. It rises rather than springs since #389: "A
gentler get-up" below has what it takes and how it was trained.
⚠ mjlab's `upright` reward reads only the sideways tilt and scores a body
on its BACK as upright; the task pays for gravity's sign in the body frame.

**Stairs** (`Pluggy-Quad-Rough`, blind: the critic sees the terrain, the
actor does not; `-Perceptive`: the actor is also shown the terrain's height
under a 1.6 × 1.0 m grid at 0.1 m, on the robot the D435's height map
sampled there): "The stairs curriculum (issue #388)" below. ⚠ mjlab casts
that grid from the body's own height, so a ray a metre ahead on a flight
starts INSIDE a step above the body and reports the floor under it; the
grid is cast from 1 m above the body (`task.RaisedGridPatternCfg`), and
`legs/policy.py` casts the same grid in our physics (its layout is pinned).
⚠ MuJoCo Warp warns once a step for every box lying on a height field with
more than 50 contacts: 1.5 M such lines through Python to a pod's network
volume slowed a run threefold (`pod.sh` filters them).

**Posture** (`models/quadruped_posture.npz`, `Pluggy-Quad-Posture`: walking
with the torso's height offset, pitch and roll commanded; 2500 iterations on
the 4090): in our physics (`--posture`) it crouches to within 6 mm of 0.224
and 0.184 m, pitches ±11° and rolls 8.4 of 8.6°, walks crouched at 0.56
m/s, and walks as the flat policy does (1.06 m/s; turns 0.85 of 0.8 rad/s;
sidesteps 0.26 of 0.3) without falling. The tilt is tracked as gravity's
direction in the body frame, replacing the stock "stay level" reward.

**Legged odometry** (`legs/odometry.py`, `--odometry`: the policy walks a
21 m course of straights, an arc, a turn on the spot and a sidestep): 2.4-
4.0 % of distance over five noise seeds, heading within 2°. Two corrections
made it: the ball foot ROLLS, so its centre moves while its contact point
does not (without the correction, 6.6 % with perfect contact and no
noise); and a contact estimate read off current arrives ~31 ms late
(ODRI's measurement), so for a moment after each footfall only the LIFTING
pair is flagged planted: gated against the last estimate and tested for a
foot rising off the floor, the estimate holds through it (averaged in, 27 %,
and a gate alone ratcheted the estimate down to nothing).

**The served loop** (`--served`: the home world with its rover taken out
and two quadrupeds in, against two rovers) is #385's to measure and move
("The served sim's speed" below).

**The sensors on a moving torso** (`--rays`, quiet box): the policy tilts
the torso under 0.4°, which moves the 2D scan plane (0.49 m up, on a rear
mast clear of the stowed arm) ±5 cm at 8 m — IMU compensation removes it,
and the D435 on the nose carries the ground. A 2D LIDAR stays for the first
quadruped deploy; a 3D one (Livox Mid-360: 265 g, 6.5 W, €739) is #381's
question for stairs and a multi-floor map. What the sim would pay: the 2D
scan is ~1.2 ms since #385, one `mj_multiRay` of 4 000 rays 4.9 ms, of
10 000 12 ms, and a full Mid-360 frame of 20 000 rays 24 ms — a quarter of
real time per robot at 10 Hz.

**What is true now:** the body is `legs.model.CHOSEN` and
`models/quadruped.xml` is its generated MJCF; the numbers are a
datasheet's where one exists and a range where none does (Parts.md); the
scripted gait's rules above are pinned in `tests/test_legs.py` only as far
as "it trots"; the policy's arithmetic, its observation, its rate and the
odometry's two corrections are pinned there too; the tables are the
script's to re-fly.

## The stairs curriculum (issue #388)

**Why #377's stalled.** mjlab promotes a robot a level for ending a 20 s
episode 4 m from its tile's centre, and its commands turn and sidestep every
3-8 s, so a capable robot seldom got there: across 6000 iterations even the
FLAT column's level sat at 4.3 of 10, where the level changes nothing. And
the columns were named backwards — ⚠ mjlab's `pyramid_stairs_inv` spawns the
robot in a pit, so walking out CLIMBS; `pyramid_stairs` spawns it on top —
and the one that climbs had stalled at level 1.7, risers of ~0.08 m.

**What changed** (`training/quad_train/task.py`, `stairs.py`):
- on a stair tile, three commands in four walk straight out at 0.3-0.7 m/s,
  heading along the nearest of the pyramid's four flights (±14°). ⚠ A reset
  resamples the command BEFORE mjlab's `sim.forward()`, so the pose read
  there is the last episode's: the command is aimed on the next update;
- flights of eight (mjlab's 3 m platform and 1 m border left five), risers
  0.10-0.26 m across the ten levels, the house's 0.18 m between levels 4 and
  5; 60 % of the robots on stairs;
- `upright` against the terrain's fitted plane, as mjlab's Go1 recipe has
  it: a flight is 33°, level against gravity the hind legs cannot reach it,
  and the stock reward kept a quarter of its value for a torso parallel to
  the flight;
- mjlab's foot clearance also charged a foot swung HIGH, the lift a riser
  needs. `foot_nose_clearance` is one-sided: a moving foot's centre 7 cm over
  the highest terrain within half a tread (a ray starting inside a step reads
  zero), so a foot closing on a riser answers to the tread above it, and one
  stepping down to the tread it leaves until it is half a tread past the nose;
- a thigh or shank on a step costs 0.25, the torso or belly 1; the walking
  pose's tolerance is wider at the hip's flexion (0.5 rad) and the knee (0.8).

Resumed from #377's seeing policy (6000 iterations) for 4000 more on a rented
L40S (1 h 24 min), the climbing column went from 0.9 to ~4 of 10 (0.17 m; a
column's mean, which the graduates' reset to a random level holds down) and
the descending one to ~7; `models/quadruped_rough_seeing.npz` is the last
iteration, the best of the checkpoints flown.

**What it climbs** (`quad_spike.py --climb`, our physics, five trials a
case; a flight is ten risers on a 0.28 m tread, the house's floor to its
second floor; the robot runs the policy's mean action; "map" is the D435's
scan, below):

| flight of ten, up / down | #377 blind | #377 seeing | #388 seeing | #388 seeing, map |
|---|---|---|---|---|
| 0.10 m | 1/5 / 5/5 | 3/5 / 0/5 | 5/5 / 5/5 | 5/5 / 5/5 |
| 0.15 m | 0/5 / 3/5 | 0/5 / 0/5 | 5/5 / 5/5 | 5/5 / 5/5 |
| **0.18 m, the house's** | 0/5 / 3/5 | 0/5 / 0/5 | **5/5 / 5/5** | **5/5 / 5/5** |
| 0.20 m | 0/5 / 1/5 | 0/5 / 0/5 | 5/5 / 2/5 | 5/5 / 5/5 |
| 0.22 m (the UK's limit for a home) | 0/5 / 0/5 | 0/5 / 0/5 | 4/5 / 0/5 | 5/5 / 5/5 |

Ten trials each way on the house's flight: 10/10 up and 10/10 down, on either
scan. It still walks the flat (0.52 and 1.09 m/s for 0.5 and 1.0, turns 0.79
of 0.8 rad/s, sidesteps 0.27 of 0.3; 89 W at 0.5 m/s against #377's 90), and
its run hashes IDENTICAL in two processes (`--determinism`). #377's blind
policy stays the blind one: trained on this curriculum it could not climb the
pits' flights even at the lowest level (0.10 m), and it lost the flights it
had (`stairs-blind1`, stopped at 3684 iterations).

**The mean action hesitates at an edge.** Mid-run, a checkpoint walked to
the first riser and STOOD there, front feet at its base — while the same
weights with training's exploration noise (std 0.32) climbed 0.15 m flights
2-3 of 5: in training the noise breaks the stall, so the curriculum's level
is the noisy policy's. Four hundred iterations later the mean action climbed
them 5 of 5. The same happened at the top of a descent, later. A policy on
the ideal scan is also shown an input it never trained on: mjlab corrupts the
scan with ±10 cm of uniform noise.

**The scan the robot has** (`legs/scan.py`; `--climb --scan map|odometry`):
the D435 on the nose (`DepthCamera(mount="body")`: the torso is not level on
a flight, so its attitude is applied by the caller), its points laid in a
2 cm map (`HeightMap` with a world-height band) through the body's pose,
read at the scan's grid. Where the camera has seen, the map's scan is the
ideal to a median 0.3-0.9 mm; 5-18 % of the points are off by more than
5 cm, at step edges. It sees the floor from ~0.5 m ahead and ~0.45 m to
either side, so 17-45 % of the scan is UNSEEN on a flight — the outer rows,
the ground behind, the lower treads behind a nose going down — and each such
point reads the nearest cell the camera saw. ⚠ Read as the ground under the
feet instead, a flight's upper treads were the floor, 0.1-0.4 m wrong on up
to a fifth of the scan. On that scan the policy climbs AS WELL OR BETTER
(table): at the top of a flight the occluded treads, filled from the nose's
side, make the drop look gentler, and the mean action walks down where the
ideal scan's true drop made it hesitate (ten trials at 0.18 m, iteration
6950: down 0/10 on the ideal scan, 10/10 on the map's, 10/10 on the ideal
with the map's fill in the unseen points, 2/10 on the map with the ideal's
there; neither ±2-10 cm of noise nor a 2-4 cm max-pooling of the ideal scan
did it).
So the next policy should be TRAINED on the map's scan — its occlusion, its
fill and its noise — rather than lean on a difference it never saw.

**Legged odometry on the stairs** (for #381): climbing a flight of ten the
position drifts 1-5 % of distance and the height reads 9-22 cm LOW at the
top (1.0-2.2 m climbed). Much of that is a sink the height has on flat
ground too — 18 cm over 4.9 m walked at 0.4 m/s, 14 cm of it with perfectly
timed contact and no sensor noise, so the stance feet's kinematics, not the
31 ms contact lag. DESCENDING is where it fails: ⚠ a foot is taken for
lifting when its contact point rises faster than 0.25 m/s relative to the
BODY, and a flight walked down at 0.4 m/s lowers the body ~0.3 m/s, so every
planted foot is dropped and the estimate coasts: 2-4 % of distance on
0.10-0.15 m flights but 28-37 % on 0.18-0.20 m ones, and, walked down faster
on the map's scan, 9-86 % and up to 0.4 m of height (a crouch lowered at
0.97 m/s does the same; gating against the body's estimate instead fed back,
and a trot drifted 37 %). Laid through the legs' own pose, the map's scan is
off by a median 16-32 mm and 12-41 % of its points by more than 5 cm: the
robot still climbs every flight to 0.22 m, and descends the house's 2/5. The
stairs need the pose corrected against the map, not the legs alone.

**What the body pays on the house's flight:** the knee's p99.5 is 13.5 of
its 22 N·m (the scripted push up one riser needed 5.1; 0.22 m risers take
15.3), and a climb straightens a hind knee to its stop (−0.35 rad) for ~0.4 s
while the front knees fold to within 0.15-0.4 rad of theirs: the reach is
there, with no margin to spare in the hind leg — a knee driven into its stop
is #379's to cushion or to extend. The failures along the way were stalls at
the bottom (the knees never left the stand's range), not a leg that could not
reach.

**What is true now:** the house's 0.18 m flight is climbed and descended, and
so is a 0.22 m one on the map's scan: the limit was the curriculum, and the
body's size is not the answer (#388's decision point was not reached). The
stairs policy runs on the ideal scan in `legs/policy.py` by default and on
the D435's map with `scan=MapScan(...).scan`; nothing served runs it yet.

## The quadruped's dock (issue #378)

The quadruped charges the way it rests: it walks over a cradle, stops with
its belly above it and lies down onto it (`legs/dock.py`; every table here
is `scripts/dock_spike.py`'s; the parts are Parts.md, "The dock"). The
cradle is a 6 mm UHMW-PE plate with a bed 128 mm wide and two faces rising
30 mm at 55° to a mouth 170 mm wide; in the bed, two poles of two
spring-loaded pins press up into the belly's two flush pads. A board of four
60 mm tags on a post 0.62 m ahead of the seat is what the walk steers by
and what the robot measures itself off once it lies there: its nose camera
sees the upper pair standing to lie down and the lower pair lying.

**The capture envelope** (`--capture`; the robot placed standing where it
lies down from, off by the row, then lying down with #377's scripted
lower-and-fold; the verdict is the electrical criterion). ✓ charges:

| across | yaw 0° | 5° | 10° | 15° |
|---|---|---|---|---|
| 0–25 mm | ✓ | ✓ | ✓ | – |
| 30 mm | – | ✓ | ✓ | – |
| 35 mm | – | ✓ | – | – |
| 40, 50 mm | – | – | – | – |

Along the axis it charges from −60 to +90 mm (the pads are 160 mm long, the
pins 20 mm apart at the seat), and it lies within ±4 mm across and ±2° of
the axis whatever it came down with. The two premises, same rows: with the
faces at the belly case's own friction (`--sticky`, μ 1.0 — the pair's MAX
without `priority`) it fails from 10 mm square-on and everywhere from
20 mm; with no funnel at all (`--flat`) the pads alone forgive 10 mm, and
the belly lies wherever it landed. **A funnel is a funnel only while tan(face angle) beats the
friction with room to spare.** (Without walls a 15° twist still charges;
the funnel's walls prop a belly turned that far.)

**The approach** (`--approach`; standing at the standoff 1 m behind the
seat, TRULY off by the row while it believes it stands exactly there): it
looks, walks in steering by the board, stops, checks what it believes,
lies down, and backs out for another run if the pins do not conduct. The
starts: a grid (0–0.3 m across × 0–30°) and 40 drawn uniform in ±0.3 m
across, ±0.3 m along and ±30°; turned away from the board, it sweeps the
spot until the board decodes.

| steering by | docked | first try | a foot on the dock | lies across / yaw / along | median |
|---|---|---|---|---|---|
| the board, re-read every 0.25 s | 52 of 52 | 40 | 8 runs, at most 1.1 s | ±4 mm / ±1.6° / −16..+6 mm | 9.9 s, 182 mWh |
| one look a run, then its legs (`--blind`) | 52 of 52 | 39 | 27 runs, up to 1.4 s | ±4 mm / ±1.6° / −31..−5 mm | 10.1 s, 187 mWh |

Both dock every time: one good look from the standoff is enough, the funnel
forgiving what the legs drift over a metre. Re-reading the board is what
keeps the feet off the dock and the belly on the seat along its axis. The
retries are the starts both close and far off the axis: from where a front
foot could land on the dock the walk goes straight, more than 20 mm off the
axis there it stops, and the check backs it out for another run. Six of
the eight runs that touched the dock were those retries, every contact over
0.3 s among them; the other two grazed it (0.23 s, 0.02 s) from starts that
began close. (Before a fit had to span the board, a
turned start could decode one column of it, fit that at any heading as 0,
and carry the one look into the wrong place: blind docked 41 of 52.)

**Docked** (`--hold`, five dockings from up to 0.2 m and 20° off): the criterion
held on every step of a minute lying there, each pole pressing 2.3 N; the
board's lower pair gave the robot its pose in the dock's frame to 2.2–2.7
mm and within 0.17°; and it stood up and backed 0.6 m off the dock in 7 s
and 157 mWh, no foot touching it. A docking and an undocking together cost
about a third of a watt-hour against a 194 Wh pack; the charger's 216 W
fills that pack in under an hour, the CV tail aside.

**What shaped it, each measured before it was believed:**

- **The walking policy does not creep.** Commanded 0.15 m/s it walks
  5 mm/s, 0.2 → 0.11, 0.3 → 0.28; sideways 0.2 → 0.01; turning under
  0.2 rad/s it barely turns (all four committed policies; the posture
  policy's band is the narrowest) — a pose loop with a proportional gain
  stalled 11 cm short. WALKING, small corrections do track (yaw 0.05 →
  0.045 rad/s, sideways 0.1 → 0.06). So the walk-in walks THROUGH at
  0.3 m/s, pursuing a point on the dock's axis, and cuts the command
  `STOP_M` short: from 0.3 m/s the flat policy comes to rest 22 ± 5.5 mm on
  (eight stops in every phase of the gait; the posture policy ±11). ⚠ This
  is every fine positioning's problem on this body, not only the dock's.
- **The feet bound the dock's width, not the belly.** The nearest stance
  foot's centre to the body's centreline: 124 mm walking straight at
  0.3 m/s, 120 steering at 0.1 rad/s, 108 sidestepping at 0.15 m/s, 84–92
  turning at 0.5 rad/s while walking and 74 turning on the spot. A 95 mm
  mouth (±98 over its faces) put a foot on a face now and then; the mouth is
  ±85 (±88), and from where a front foot could land on it (17–20 mm ahead
  of its hip) the walk-in walks straight — no sidestep, at most 0.1 rad/s of
  steering, never a turn on the spot, and more than 20 mm off the axis it
  stops and the check backs it out.
- **Rigid contacts under a lying quadruped are statically indeterminate.**
  The first cradle had contact bars 4 mm proud, the robot's weight as the
  preload the issue proposed. But the belly shares the load with four limp
  legs (it carries 67 of the robot's 92 N on a bare floor; the rest is the
  legs' own weight on their feet), so which support carries what is set by
  sub-millimetre heights: a robot lying 6 mm off centre rolled 0.2° onto its
  right feet, all 57 N went through the left bar, and the right pad hung
  0.3 mm clear — seated in the funnel, not charging. The contacts are
  sprung pins (Mill-Max 0858, two a pole, 2.35 N a pole at their rated
  1.143 mm travel): the weight seats the belly, the springs are the contact.
- **Lying down moves the body back 34.8 mm onto the dock** (35.5–36.2 onto
  the bare floor): #377's fold swings the legs forward under the belly. The
  walk-in stops `LIE_SHIFT_M` ahead of the seat.
- **The board's baseline is the approach's precision.** The facing comes
  from where the tags are (Kabsch over their translations, the #88 rule),
  and the seat is 0.62 m behind the board, so a facing error swings the
  believed axis: with the tags ±70 mm apart the robot believed itself up to
  17 mm off the axis while it walked over the dock within 7 mm of it. At
  ±140 mm, and each look blended half into the last (`dock.blend`), the
  belief follows the truth to ~5 mm. A fit needs its tags to SPAN the
  board: a column's two tags are one point in the plane, and seen alone
  they fitted any heading as 0 with a perfect rms. And one whose tags sit
  more than 30 mm from the drawing is refused (real looks: 1.6 mm median,
  14 worst of 80), not blended in.

**Five MuJoCo traps the pins walked into**, each now a constant with its
number: (1) contact filtering between a parent and its child body does not
apply to a body welded to the world, so the pins collided with their own
plate — they meet the robot alone (`contype` bits); (2) soft contact scales
with the bodies' effective mass, and against a gram-scale sprung pin the
pad sank 1.2 mm into it at 0.12 N instead of pressing it down — the pins'
contact is `solref 0.004` (two steps) with `solimp 0.99 0.999`, pressing
2.32 N a pole of the part's 2.35; (3) a light pole then chattered against
that stiff contact, the criterion flipping 1286 times in 10 s of lying
there — a 0.1 kg armature (the plungers' inertia, a numerical stand-in)
holds it every step; (4) a 9 kg belly let go 30 mm up sank through the
6 mm plate onto the floor, so the plate's collision box reaches 50 mm under
the floor; (5) #377's `VirtualModel` read the body's inertia off a fresh
MjData's all-zero mass matrix when built before a forward pass, and its
lie-down moved the body 44.6 mm instead of 34.8 — it now refuses.

**The dock is the map's origin** (#381's plan: decided). Lying docked, the
robot reads its pose in the dock's frame off the board's lower pair to
2.2–2.7 mm and 0.17° (`--hold`). The seat alone would give it a few mm and
~2° — the funnel's clearance lets the belly lie up to 2° twisted — so the
anchor is the BOARD, read lying, not the seat. The dock is the one pose
the robot returns to by construction, it needs no survey, and one dock
serves the pair, so both robots' maps share its frame. `rack_prior`'s
commissioned pose retires for the quadruped: #387's `anchor_at_dock` snaps
the legs' estimate to what the board says. Where the dock stands in the
world is the sim's to know (the site's drawing, its own checks); the robot
needs only its frame.

**What is true now:** the dock is `legs.dock.DEFAULT` (`dock_xml`, any
world; `world_xml`, the spike's and the tests' standalone world, the dock
AFTER the robot, `body_xml(after=)`, because the standalone keyframes index
the robot's free joint from `qpos[0]` — `LegOdometry` finds its robot by
name); the criterion is `dock_charge_contact`, each pad on a pin of its own
pole; the walk-in is `walk_in_twist`, `fit_dock`,
`seen_from` and `blend`; nothing in the served sim loads any of it until
#387 builds `QuadBody.dock_routine` on them, and the lie-down it uses still
lives in `scripts/quad_spike.py`. `tests/test_dock.py` pins each rule above.

## The quadruped's arm, its coupling and the rack (issue #378)

The arm is two pitch joints on the torso's top front, both motors stacked
at the shoulder (`legs/arm.py`; every table here is `scripts/arm_spike.py`'s;
the parts are Parts.md, "The arm and its coupling"). The elbow's motor
drives the forearm's ABSOLUTE angle through a rod beside the upper arm, and
a second, passive parallelogram keeps the end plate at the torso's angle
wherever the arm is, as a palletising robot's does. The shoulder axis sits
0.15 m ahead of the torso's centre and 0.10 m over it; the links are 0.25 and
0.35 m; the arm weighs 1.17 kg without a tool (the body's placeholder
budgeted 0.9). A tool still hangs by the rover's plate and split peg, now
220 mm long; the fork at the arm's tip takes it at ±85 mm in V's steeper
than the rover's (60°, for the stairs, below), and the rack's trays at ±45.
The arm is the served body's since #405 ("The arm on the served body",
below); the rack and the tools on it are its stage B.

**The level-tool choice** (`--reach`). A gravity seat wants its V's level,
and with two pitch joints the plate's angle is their sum. The three options,
against the world's targets with the plate level:

| target | (a) a parallelogram | (c) the legs pitch the body |
|---|---|---|
| the claw on the floor, 0.40–0.50 m ahead | yes, ≤ 2.1 N·m | no |
| the claw on a 0.8 m tabletop | yes | no |
| the pen at a board's bottom row (0.19 m) | yes | no |
| its middle and top rows (0.30, 0.41 m) | yes | at −11° and −3° of pitch |
| a rack bay's peg (0.50 m) | yes | at −9° and a 7 cm crouch |
| carrying over the nose | yes | no |

(c) needs the forearm within the posture policy's ±11° of level, so it
reaches three of nine, each at a posture the tool's job then depends on.
(b), a third motor pitching the plate, reaches what (a) reaches and puts
its motor at the end of the lever: holding the claw and a 0.4 kg cube
straight out takes 4.86 N·m of the two motors under (a), 7.02 with a
GIM4310-10 at the wrist (+44 %, the wrist motor at 61 % of its own
continuous rating) and 8.01 with a GIM8108-8 there (+65 %). What (b) buys
is a plate held level against the body's pitch — 27–33° on a flight of
stairs, measured below — and a tool tilted on purpose. A hung tool hangs
plumb whatever the plate does, so the seat needs the first only coming
down the stairs, where the pitch lays a V's flank down (below), and a
steeper V does that for nothing. **The choice is (a).**

**The motors are the legs' GIM8108-8.** Holding, the arm's worst static
load at any target is 2.1 N·m, and flown it peaked at 2.0 N·m on the flat
and 4.2 coming down the stairs with its RMS under 1.3 on the flat — a fifth of the
motor's 6.76 N·m continuous rating. The GIM6010-8 would do the work, but it
weighs 388 g against 396 for 5.16 N·m continuous at 48 V: the same mass for
less torque, and one spare, one driver and one set of gains is worth it.

**The coupling's capture** (`--capture`: the robot placed standing at the
bay, off by the row, the arm aimed where the peg truly is; a pick, then the
tool hung back; ✓ both, P picked only):

| off by | the arm's fork | the rover's fork (`--rover`) | trays at ±35 (`--narrow`) |
|---|---|---|---|
| across −25 / −20 / −15 / −10 mm | P / ✓ / ✓ / ✓ | – / – / – / – | P / ✓ / P / P |
| across 5 / 10 / 15 / 20 / 25 / 30 mm | ✓ / ✓ / ✓ / ✓ / ✓ / – | ✓ / – / – / – / – / – | ✓ / ✓ / ✓ / ✓ / ✓ / ✓ |
| yaw 6 / 8° | ✓ / – | ✓ / ✓ | ✓ / ✓ |
| aim along −10..+5 / +10 / +15 mm | ✓ / ✓ / – | ✓ / P / P | ✓ / ✓ / – |
| aim height −10 / +10 mm | ✓ / ✓ | P / ✓ | ✓ / ✓ |

The rover's fork on the arm, with its own 22 mm drop and 36 mm lift, takes
nothing 10 mm off: its end-stops sit 4 mm past its peg's ends, and one lands
under the peg. The arm's fork takes the peg from −20 to +25 mm across (its
stops are 53° ramps a peg end slides down) and yaw to 6° (at 8° the far V
is 12 mm out, past its mouth). Swept at the line-up gate's corners (±15 mm
and ±4°) and out to ±18 mm, every pick and return held; ±20 is the edge,
where a pick at −2° failed and one at −4° held. A return turns on the room
the tool's plate has between the trays: at ±35 mm (9 mm of room) it set
its plate on a tray's corner from 10 and 15 mm off. The asymmetry is the
stance's yaw creep during the pick (below): the fork keeps moving one way.

**The approach** (`--approach`; standing 1 m behind the bay's working pose,
TRULY off by up to ±0.3 m across and along and ±30° while it believes it
stands exactly there: it looks, walks in steering by the rack's tags,
stops, waits, measures the bay, and takes the tool if lined up, else backs
out and tries again; then it hangs the tool back): **41 of 41 — a centred
start and 40 random ones — took the tool and hung it back, 24 on the first
try**, a median 18.8 s. Measured off the tags, the aim was within 0.9 mm
along and 1.5 mm in height every time; the robot stood −15..+6 mm across
and within 1.9° of square. A start turned 26° away sees none of the rack's
tags, and sweeps the spot until they fit, as the dock's approach does. The
walk-in's stops ride the gate's edge: the settle's creep leaves most of
them 13–15 mm across, and flown on an intermediate fork 3 of 41 starts
never lined up in three tries. The gate stays at 15 mm, 3 mm inside the
capture's reliable ±18.

**Carrying** (`--retention`: the tool carried high over the nose; the
criterion every step; "swing" the tool off plumb, "plate" the plate off
level):

| flight | open, longest | swing | plate | still seated |
|---|---|---|---|---|
| walk 0.3 m/s, trot 0.6 and 1.0 m/s, turn, sidestep, a hard start | ≤ 4 ms | ≤ 7° | ≤ 2.8° | yes |
| stop from 1.0 m/s | 2–6 ms | 22° | 3.9° | yes |

**The stairs** (`--retention --stairs`: the house's flight, ten 0.18 m
risers, on the seeing policy, from rest and facing along it, 20 starts a
row; a tool whose CoM sits on its peg's line, 30 mm ahead of it, and at the
envelope's 60; lost of the flights that arrived — the policy stalls on
about 1 in 10 by itself):

| flight | CoM ahead | this fork (60° V's) | as first built (`--first`: 45°) |
|---|---|---|---|
| up | 0 / 30 / 60 mm | none of 19 / 18 / 18; open ≤ 38 ms | none; open ≤ 122 ms |
| down | 0 mm | 1 of 18; open ≤ 60 ms | none of 18; ≤ 74 ms |
| down | 30 mm | none of 17; ≤ 64 ms | 1 of 18; ≤ 80 ms |
| down | 60 mm | **none of 18**; ≤ 88 ms | **13 of 17** |

The plate pitches 27° climbing and 33° coming down; the arm's motors peak
at 4.2 N·m, and the knees read p99.5 14.0 climbing and 9.2 descending.

**The V's are steeper than the rover's because of the stairs** (found by
`--view stairs`). The parallelogram holds the plate at the TORSO's angle,
and coming down the flight the torso pitches 33° nose-down: at 45° one
flank of each V then lies 12° off level, under the peg's friction angle
(21.8°), and only 5 mm tall. A step's jolt hops the peg, and it lands on
that flank and stays there half out — traced, 4.7 mm up it with the tool
leaning 28° on the pad — until a later jolt walks it off. A tool that leans
onto the pad starts every jolt from that side, which is why the lean
decides it. Flown at 60 mm ahead, 30 starts a row, one part swapped at a
time:

| V's | end-ramps | lost of the flights that arrived |
|---|---|---|
| 45° (as first built) | 45° | 22 of 27 |
| 45° | 53° | 7 of 27 |
| 60° | 45° | 0 of 28 |
| 60° (this fork) | 53° | 0 of 28 |

The V is the fix: at 60° the low flank lies at 27°, over the friction
angle, and a hopped peg slides home. The flanks are 31 mm long so the
mouth keeps the rover's 15.6 mm; on the rover's 22 mm flanks it was 11, and
the capture's corners went chaotic (a pick 15 mm across at 4° held or
failed with a 0.4 mm move of the ramps). The drop and lift follow, 33.5
and 56 mm, keeping the rover's 3.4 mm under a hanging peg and 14 mm over
the trays' corners. The steeper V grips a peg 41 % harder along its axis
(0.40 of the tool's weight against 0.28), so the end-ramps steepened to
53°: their slippery face pushes 0.59. Two things were tried and dropped:
V faces at μ 0.2 (a sideways jolt slid the peg along its axis and up a
ramp, and the tool was lost) and the short flanks. At 90 mm ahead, 27
descents held too.

**No lock, and the stairs are where that could change.** On the flat the
contact opens for at most 6 ms and up the flight for 38, under the 200 ms
holding capacitor the rover's peg was sized with. Coming down, the fork
falls faster than gravity at a step (traced: −15 m/s², then +24 as the leg
lands), the peg floats 8 mm and moves along its axis, and about 1 descent
in 60 it lands on an end-ramp and the tool goes: 1 of 58 with a tool on its
peg's line, none of 58 on the fork as first built, inside the noise. No V
holds a peg while the fork falls faster than g. The levers, cheapest first,
for step 4 to weigh against how often the robot carries a tool downstairs:
a slower descent, a carry pose nearer the torso's pitch axis (the fork's
0.57 m lever multiplies the pitch), and the positive one, a magnet in each
V (Parts.md). A fall throws the tool whatever holds it (below). What moves
is the swing, so a carried tool needs clear air 57° either way. The knees
with the real arm and a tool aboard read p99.5 7.7–8.9 N·m on the flat,
14.1 in a hard start and 14.0 climbing (the climb read 13.5 without the
arm, #388). Carrying, the robot weighs 9.60 kg
against the placeholder's 9.34, its CoM 21 mm forward and 18 mm up: inside
the committed policies' randomisation (the torso's mass ±15 %, its CoM
±5 cm along and ±3 cm up), which is why they walk it untrained.

**A fall throws the tool, and the arm folds** (`--fall`, `--getup`).
Pushed over while trotting, the robot throws the tool whatever holds it: a
gravity seat cannot hold upside down, and a lock would keep the tool on an
arm the robot is rolling across. Left out at its carry pose, the arm then
props the robot on its side. #377's get-up policy — trained, as #389's is,
with a placeholder arm that collides with nothing — could not roll it
there (3 of 6 pushes stayed down), so the arm folds as the torso passes
60°. #389's rolls it either way: 6 of 6 pushes, standing 1.2–2.0 s after
it takes over with the arm out and 1.3 s folded; from 20 random drops with
the arm folded it stands 20 of 20 (median 1.5 s, as with the
placeholder). The fold drives the elbow's motor to its 22 N·m peak
through the impact; #405 keeps it, and folds the arm before a rest too.

**What the arm hides** (`--sensors`). Stowed — the upper arm straight back
over the torso, the forearm 10° up from it — it hides nothing: no LIDAR ray
and no pixel of the nose or depth camera. Folded flat, the fork hid 13 %
of the nose camera's image; tilted 15°, the forearm blinds 9 LIDAR rays.
Carrying at the carry pose, only the upper arm crosses the scan plane: 5
rays of 360, and no camera pixels.

**The tool envelope** (`--envelope`, the constants in `legs/arm.py`, for the
workshop's validator in step 4): **0.40 kg; its CoM on its peg or up to
60 mm ahead of it (0.35 N·m); 0.20 m under its peg; 12 W.** A CoM ahead of
the peg leans the tool onto the lean-pad (6°, seated through a trot at
0.25, 0.40 and 0.60 kg, out to 90 mm ahead). A mass behind it, below the
peg, is where the pad and its post are: the table's −30 mm rows rest on
the post. 60 mm is the stairs' tested range; 90 held on the flat and down
27 flights too, where the 45° V's had flopped a 0.40 kg tool to 74° on the
flat.

**What shaped it, each measured before it was believed:**
- **A pad that pushes levers the tool while its peg is on the trays.** At
  the rover's 2.7 mm of intrusion the lean-pad met the plate's bottom edge
  as the fork lifted, before the V's took the peg, and levered the tool 15°
  up the V's flank. A pad fixed to the fork cannot reach the plate only
  after the seat (the geometry makes the contact at the seat or before), so
  it stands 3 mm behind the plate and takes a pressing tool's reaction
  after 4° of swing. It is a round bar: a square one's corner caught the
  plate's edge.
- **The IK must not wrap the shoulder.** The stow is at 180°, and a target
  close over the shoulder wants more; wrapped to −180, the joint limit
  swung the arm the other way through the robot.
- **A tool faces the robot, so its left conductor rides the arm's right
  V.** The V's are named for the pole they take.
- **The level linkage carries a tool's lean.** The elbow's motor holds the
  weight at the wrist; what the tool does ahead of the wrist goes to the
  torso through the parallelogram (virtual work: the equality acts on all
  three joints alike). The closed form first charged it to the elbow.
- **The lift must clear the trays' V corners** by more than an aim 10 mm
  low: the rover's 36 mm cleared them by 4 mm, and such a return knocked
  the tool off. 56 mm, from the 60° V's 33.5 mm drop, clears them by 14.
- **The ramps' face is slippery, and steeper than the V's friction.** With
  45° V's and ramps at the peg's own 0.4, the ramp's push was 0.30 of the
  tool's weight against the far V's grip of 0.28, and a pick from 15 mm off
  at 4° left the end on the ramp. Faced with acetal or PTFE (μ 0.15) and at
  53°, it pushes 0.59 against the 60° V's 0.40.
- **Swinging a tool up and back down moves the stance under the arm.** In
  the viewer's fetch the robot held the tool up at the carry pose for 3 s
  and brought it down to hang it back: the torso had shifted 25 mm, the aim
  taken before was stale, and the return dragged the tool off the trays.
  After any big arm move it stands and measures the bay again, as after a
  walk.
- **A loop of the stairs measures the policy, not the coupling.** Turned on
  the landing to within 8° of the flight, the seeing policy stalled at the
  top edge for its whole budget in about half the loops, and a turn there
  tipped it down the flight and threw the tool. Flights are flown from
  rest, facing along them; the viewer's scene crosses a hill instead of
  turning at an edge.
- **A carried tool is terrain to the stairs policy.** The height scan
  reads group-0 geometry as ground, and a tool over the nose read as a
  0.4 m obstacle: the robot filters what it carries as it filters itself.
- **A stopped robot keeps turning.** The flat policy holds no heading (its
  yaw-rate dead band never corrects a slow turn): after the walk-in's stop
  it turned 1.4° in 0.5 s and 2.7° by 6 s, 7 mm a second at the fork, and 3
  of 21 picks that began inside the capture ended outside it. The robot
  waits 3 s and measures the bay again (`rack.SETTLE_AFTER_WALK_S`).
- **A bay's tags must stay in the nose camera's view from the working
  pose.** Between the bays (±150 mm) they sat 25° off its axis, and 2.7° of
  yaw took one out of the frame; a pair a bay at ±75 mm sits at 20°.

**Which tools survive** (#375 step 4 rebuilds them on the longer peg):
- **the pen** — it keeps its sideways carriage; the arm gives it a board's
  full height from one stance (0.19–0.41 m from 0.43 m out) and the pressing
  force, reacted by the pad;
- **the claw** — the arm reaches the floor 0.40–0.50 m ahead with the
  rover's 154 mm pendant, so the pendant can shorten; holding the bench's
  0.4 kg cube it weighs 0.61 kg, carried seated as the envelope's 0.60 kg
  rows were, and held straight out at 39 % of the motors' rating;
- **the LCD and the seed dispenser** — unchanged but for the peg;
- **the plug does not** — no job uses it, and the robot charges by lying
  on its dock (#390).

**What is true now:** the arm is `legs.arm.ArmSpec()` (the choice),
`CHOSEN.arm` since #405, built onto the body by `model.body_xml`; the
controller is `ArmDriver` (the GDS68's PD plus the arm's own gravity off
its planar model, #405); the coupling is `ArmSpec().fork` (60° V's) with
`rack.tool_xml`'s peg, judged by `rack.tool_power`; the rack is
`rack.DEFAULT` (three bays at 0.30 m, pegs at 0.50 m, tags 29–34), found by
`rack.bay_aim` and walked into by `rack.walk_in_twist`; the stow, the carry
pose, the fold threshold and the tool envelope are `legs/arm.py`'s
constants. `tests/test_arm.py` pins each rule above. #405 built the arm
and the fold into the served body; its stage B brings the rack and the
tools into the served world and the carried-tool filter into the scans,
and step 4c the envelope into the workshop's validator.

## The served sim's speed (issue #385)

A pair on the deploy box ran below real time (0.96× over a carry, 0.80×
over a day, rooftop #296), and two quadrupeds cost more than two rovers:
their policies and their drivers' PD. Everything here was timed on the
deploy box itself (AMD EPYC 9645, Zen 5c, 3.7 GHz; the prod image, osmesa)
in throwaway containers beside the live world, because the dev machine was
never quiet.

**The LIDAR is one call.** `Lidar.scan_split` cast its 360 bearings with an
`mj_ray` each and drew the noise inside the loop: 3.55 ms a scan on the
box, 35 ms of every sim second per robot at 10 Hz — and most of it was the
Python around the rays (a scalar `np.clip` alone is microseconds), not the
rays. One `mj_multiRay` returns every distance and geom bit-identical to
the per-ray calls (108 000 rays at 300 poses in the home world, at any
cutoff); the masks are numpy; the noise is still drawn ray by ray in
bearing order from each of the two streams, because a batch draws the same
numbers in another order and no flown day would hash the same again. 1.18
ms a scan now: ~24 ms a sim second per robot. A scripted home day hashes
IDENTICAL before and after, alone and with a parked second robot (3 234
state samples, 11 262 scans, 4 063 tag images and decodes).

**The legs' drivers in C.** The driver's PD and the torque-speed envelope
ran in numpy on every physics step. MuJoCo's `dcmotor` in position mode
runs them in C, but it models a VOLTAGE: its controller makes
v = kp (target − q) − kd q̇, clamped at the bus, and the motor turns it
into K (v − K q̇) / R, clamped at the current limit. With K and R the
envelope's own line (K = bus / no-load speed, R = K · bus / saturation
torque — an equivalent DC pair, not the windings' Kt and R) the clamp at
the bus IS the envelope, and kp = Kp R / K, kd = Kd R / K − K make the
torque the PD: 1e-14 N·m apart at 24 000 random joint states, the peak
clip among them. kd comes out negative (−0.43), and that is right: an FOC
driver cancels the back-EMF that a bare voltage would add as damping.
What does differ is the integration: `implicitfast` integrates an
actuator's velocity term implicitly, so one step from identical states
differs by up to ~1 rad/s (1e-15 under Euler) — the closer model of a
driver whose loop runs at kHz, not the explicit 5 ms PD the policies
trained on. They walk on it within tolerance (numpy → dcmotor):

| | numpy PD | dcmotor |
|---|---|---|
| tracking at 0.5 / 1.0 m/s | 0.54 / 1.07 | 0.53 / 1.06 |
| knee p99.5 / worst RMS at 1.0 m/s, N·m | 7.9 / 4.8 | 7.8 / 4.7 |
| power at 0.5 / 1.0 m/s, W | 88.1 / 108.9 | 89.0 / 112.1 |
| posture, every commanded row | | within 0.1° and 1 mm |
| legged odometry, 5 seeds | 2.4–4.0 % | 3.3–4.0 % |
| seeing climbs | | the same, one more 0.10 m flight (3/3) |

and a flight hashes IDENTICAL in two processes. #388's stairs policy,
merged after these, re-flown on the dcmotor: every case at the house's
0.18 m riser 10/10 on either scan (the flight of ten up and down among
them), the knee's p99.5 13.6–13.9 N·m (#388's 13.5), and the flat as
before (0.51 / 1.08 m/s, 89 W at 0.5). #378's dock, whose approach walks
on the policy and lies down on the scripted routines' torque (the reason
for `legs/drivers.py`), re-flown on them: the capture envelope cell for
cell; the approach 52 of 52 docked (37 at the first try, #378's 40), feet
on the dock in 8 runs for at most 1.06 s, median 9.8 s and 182 mWh; held
docked, 100 % contact at 2.3 N a pole, the board's pose to 2.2–3.0 mm. On the box the quadruped
pair's physics and controllers went from 231–236 to 191–199 ms of wall per
sim second (`quad_spike.py --served`, A B A B): quadruped ÷ rover 1.30–1.40
→ 1.12–1.13, the rest the two policies.

**A served day, profiled.** `serve.py --pair --free-run` on the box, the
home world, a scripted day with the jobs, hunger and the near field on
(`--errand draw --errand2 carry`, 4 084 sim s) streaming to a local sink,
under `py-spy` at 100 Hz. The physics thread's time, by what it was doing:

| share | what |
|---|---|
| 28.9 % | `mj_step`, the physics |
| 25.6 % | the depth camera (8 400 rays a frame at 10 Hz per robot) and its height map |
| 11.4 % | the tag camera: osmesa renders and the decode, localising at the rack and the dock |
| 4.0 % | the charge pins' contact criterion, every step |
| 3.8 % | the tool's poles, every step (11 % before `touching` lost `np.isin`) |
| 3.5 % | the planner's replans (the inflation's distance transform) |
| 3.3 % | the activities (encounters, the cage, the plates) |
| 2.8 % | the LIDAR |
| 2.7 % | the occupancy grid's update |

Fixed, as exact rewrites (each day hashes IDENTICAL): `touching`'s two
`np.isin` a call, and `rack_charge_contact`'s, both now the few rows that
hold the geom, checked in Python; the pack's scalar `np.clip`. **Not
cheap, and deliberate:** the physics; the depth camera, whose 8 400 rays
at 10 Hz is the sensor as specified (Rover.md) — halving its rate or its
rays would halve its share, and that is a decision about the sensor, not
a speed-up; the tag camera's renders, whose resolution is what the decode
needs at the standoff, and which #378's dock replaces. The quadruped
brings one more when it climbs: the stairs policy's scan (#388). On the
ideal casts it is 187 `mj_ray`s from Python a policy step (parallel rays,
which one `mj_multiRay` cannot cast): 1.2 ms in the home world, ~60 ms a
sim second per robot. On the D435's map (`legs/scan.py`) it is the depth
frames and the map's update instead, not measured here. Either is paid
when the stairs are served, not by the first deploy's flat policy.

**The day, before and after** (box, the same day both ways, the narration
IDENTICAL line for line — the pair's peer returns included): staging
**0.97–0.98×** real time (4 084 s in 4 177–4 195 s of wall, two runs), the
scan batched 1.01×,
the tool's poles and the pack too 1.10×, the charge pins too **1.14×**
(3 578 s). That is a longer and calmer day than #296's 0.80× (a 259 s
carry day heavy with swaps at 1 ms steps); a day's multiple is a property
of what the day did.

**The server.** A quadruped pair costs the box ~20 ms a sim second more
than a rover pair (the two policies, above), so it would serve at ~1.1×
over this day. #377 had asked for at least 1.3× — a margin, never argued:
1.3× needs ~770 ms of wall a sim second, a physics thread ~1.2× the box's.
A candidate core was rented by the hour (Runpod's `cpu5c`, 4 vCPUs): an
**AMD EPYC 4564P** — the Ryzen 9 7950X's silicon on AM5, Zen 4 at up to
5.88 GHz — on a host shared with other tenants (load 25–30 of its 32
threads, so a pessimistic reading). The same code (all but the charge pins'
fix), the same day: **1.82×** real time (4 167 s in 2 293 s), and over the
span the two days share before they part (0.1–640 s; its Mesa is 25.1, the
image's 25.0, and a render that differs moves a decode) 1.75× against the
box's 1.07×: **1.64× the box**. Physics alone (`--served`) 1.5×; a scan
0.72 ms. A quadruped pair there, the same arithmetic: **~1.78×**. The core
is the answer, not the count: the physics is one thread.

**Decided (Ben, 2026-09-27): the served world stays on the current box**,
at about a fifth of the candidates' price. So the served world does NOT
have #377's 1.3×: a quadruped pair is expected at ~1.1× over a day, and the
known costs ahead take it under 1× — #386's scan matching runs on every
scan, and the stairs policy's scan is ~60 ms a sim second per robot. Below
1× nothing breaks: the world runs slower than the wall, the site plays the
stream at its measured pace (rooftop #298), and everything the robots are
scored on runs on sim time, as the rover pair's weeks at 0.5–0.8× showed.
The pair is re-measured on the box at #387, once it is served for real; a
faster box is the lever if the slow playback ever matters, and the EPYC
above is what one buys.

**What is true now:** the scan is one `mj_multiRay`, its noise drawn in
bearing order (`test_the_batched_scan_is_the_scan_it_replaced` pins every
ray against the per-ray scan); the leg drivers are position-mode
`dcmotor`s, commanded as a GDS68 is (`legs/drivers.py`): a policy's command
carries its own gains, a routine's torque rides a target with the damping
cancelled and steps as a torque motor does, and `limp()` holds nothing, so
the policy and the scripted routines share one body (the dock's approach
does); the default gains are `actuator.driver_gains`, one definition that
`training/` reads from `quadruped.json` (`tests/test_legs.py`). The served
pair's physics thread is now about a third physics, a quarter depth camera
and a tenth tag camera, and none of the three is cheap to cut. A pair is
one physics thread, so the lever left is the core, and the box keeps its
core for now (above).

## The map stays true under drift (issue #386)

The sim's odometry was kinder than hardware: exact wheel angles, a gyro with
no noise, and a quadruped whose tilt was read off the truth. Made honest
(`perception/imu.py`, `perception/encoders.py`; the numbers are Parts.md's
table), the rover's heading walked -7.3 deg in 260 s -- the ICM-42688-P's
offset after its boot calibration, 0.005 deg/s/degC over a 10 degC swing --
and its lab trip failed on the way: `no route over the floor mapped so far`,
0.2 m short of the lobby's door (staging, with the perfect gyro, arrived).
Legged odometry went from 3.3-4.0 % of distance on #377's course to
3.7-5.2 %. So every level scan is now matched against the robot's own map
before it is fused (`mapping/scan_match.py`), and the pose it matches to is
the belief.

**Why Gauss-Newton, and a search only when it fails.** Between two scans
0.1 s apart odometry is off by millimetres, well inside a fit's basin, and
a fit costs ~0.5 ms where a correlative search over a useful window costs
tens; the fit's own normal matrix says which directions the walls fix,
which the issue asks for by name. A search (brute force over ±0.6 m and
±6° in whole cells and degrees, then a fit from its best) runs only when a
fit disagrees with its map -- a pose already past the fit's reach -- and a
robot lost beyond that window has the dock, and #381's loop closure.

Every piece was found by flying a lab trip with each scan and the true pose
recorded, and replaying the matcher over the recording in seconds. A
replay of the matcher the flight flew reproduces the flight to the digit; a
replay of a variant shows what it would have believed along the same path
(a flight with it steers by that belief and parts from the recording), so
the numbers below the table are re-flown.

| piece | without it | with it |
|---|---|---|
| a SIGNED distance, zero on a wall's first cell, "inside" measured from the free side | a wall seen through ranging noise is a band 3 cells deep; unsigned, a point inside it pulled on nothing (a synthetic room recovered to 0.5 deg); measured from the band's back, the back was a second face, a point past the middle was pushed through the wall and the fit crept, unconverged after 8 iterations; zeroed on the cell's face, every wall read up to a cell near | recovered to ~1 cm and 0.05 deg, every offset to the same pose |
| returns short of 6.5 m only | a wall at the LIDAR's 8 m reach is seen SHORT -- its long draws are clipped to "no return", and the map clears it with the same rays: the street's far wall pulled the pose +32 mm a scan (from the true pose, against a true map) | 0.0 mm (sd 3.5); the trip's worst error 0.61 → 0.18 m |
| odometry's pose as a term of the fit | every scan put the map's quantisation into the pose: with PERFECT odometry the house read 5-12 cm and 0.8 deg off | 7.5 cm, under 1 deg |
| ...and a ROBUST one (Cauchy, 5 mm) | wheels spinning on the lab's feed plate pumped ~10 mm a scan for 12 s; a plain prior held on, the fit left its basin and turned the pose 12 deg to explain it, and the robot drove home 3.4 m out | the pump absorbed: 0.22 m at the plate, 5 cm by the day's end |
| degeneracy off the walls' SMOOTHED normals | 5 cm cells make a straight wall step, and a corridor's jags read 12 % of a real constraint; a threshold that held a corridor held the lab's weak direction during the pump too, where odometry was the thing that was wrong | a corridor 0.003, a room 0.34: held at 0.05 |
| a scan fused only once the robot moved (1 cm / 0.5 deg) or every 5 s | docked 388 s, 3 800 scans fused at a pose jittering by a millimetre walked the map and the pose together 0.15 m | 0.03 m |
| a fit the map disagrees with searched round (±0.6 m, ±6°), and refused if nothing agrees | a second plate stall left the pose 0.3-0.5 m out, past the field's 0.3 m reach: flown, it drove home that far off and fused a second house until the dock's anchor | an injected 0.45 m offset recovered in 5 s at the dock and in the house, in ~1 min leaving the lab; replayed without one it never fires, and flown it fired twice in a day of three trips |

What did NOT work, and why: gating on the share of inliers (refusing a scan
whose live points mostly disagreed) refused exactly the scans that would have
corrected the plate's pump, and the robot was lost anyway; a Gaussian-
smoothed distance gave the fit a valley with a floor above zero, and
Gauss-Newton overshot it into a two-cycle; weighting points by their ranging
noise held directions the walls fixed (800-970 of 2 284 scans) and the
trip's worst error rose 0.23 → 0.42-0.52 m; a reach of 0.6 m pulled the
pose onto the wrong walls during the plate's stall (0.22 → 0.53 m); a
field recomputed every scan made the pose chase its own freshest evidence
(0.45 → 0.60 m worst).

**The drift baseline, before and after, for both bodies**
(`scripts/drift_spike.py`). The rover flies a home day of three lab round
trips -- the mouse's feed in the lab, then a carry at the rack, `--trips
3` -- on each tree:

| rover, three lab trips | worst | median | worst heading | the lab |
|---|---|---|---|---|
| perfect sensors, no matcher (056a4a1, the sim before) | 0.60 m | 1.7 cm | 0.34° | reached each time; the first trip's plate pump left the belief 0.6 m out through all three, until the dock |
| honest sensors, no matcher (the before) | 5.41 m | 0.28 m | 70.7° | never reached: `no route over the floor mapped so far` 0.2 m short of the lobby's door, then lost |
| honest sensors, matched (the after) | 0.34 m | 1.6 cm | 1.08° | reached each time, 0.2-0.34 m out in the lab and back to 7 mm, 7 mm and 6 cm at the rack; the search fired twice |

The quadruped walks the rover's own route, three times out and back (145
m), on the flat policy steered by the TRUE pose -- what is measured is the
estimate, so its steering must not depend on it -- with two estimates off
the same IMU draws:

| quadruped, three lab trips | worst | median | at home | worst heading |
|---|---|---|---|---|
| legged odometry alone (the before) | 3.14 m | 1.13 m | 2.26 m | 12.9° |
| matched (the after) | 0.20 m | 0.11 m | 0.7 cm | 0.95° |

Neither estimate accumulates once matched: every trip reaches the lab
~0.2 m out (the frame its map was laid in on the way) and comes home to
about a centimetre (1.2, 0.6 and 0.7 cm).

**The lab door**, asked of the robot's own planner: can it plan from where
it believed the lobby was to where it believed the lab was? Open for the
matched rover after each of its trips, and for the sim before; the honest
rover without the matcher never reached the lab to have a door to ask
about. The quadruped's odometry map plans "open" as well -- because its
walls are gone: three rotated copies of the lab, and the lobby's east wall,
the one with the door in it, rubbed out by rays laid through later poses.
A map smeared by drift loses walls as well as doubling them, which is why
the spike saves each map's picture beside its trace. For scale: in a map
laid through the TRUE poses the door leaves 0.30 m a plan can use at the
wall (1 m, less 0.35 m of inflation each side), so the issue's estimate
holds -- ~0.3 m of smear closes it.

**Found on the way.** The quadruped attached into a world read its joint
ranges in DEGREES: the include form of `legs/model.body_xml` carries no
`<compiler>`, and MuJoCo's default unit is the degree -- the knee's
-2.75..-0.35 rad became -0.048..-0.006, and standing, the joint limits threw
the robot 0.4 m into the air. `quad_spike.py`'s pair world was built so,
which #385's quad-pair timing flew; both go through
`legs.model.attachable()` now. And the quadruped's LIDAR site sat at the
centre of its own puck: every ray hit the housing at 4 cm and the
self-filter dropped all 360 -- the site is 6 mm above the puck now, as the
rover's is (no mass moved).

**What is true now.** Every level scan is matched before it is fused and
the matched pose is the belief (`HubMission._match`); the rover's dock
anchor stays, and the two agree at the dock to centimetres. A match costs
0.67 ms on the dev machine (median 0.46, p99 7.2 -- the field's refresh, 5
ms every 20 fused scans or 1 m, and the rare search): 6.7 ms of a sim
second per robot at 10 Hz. Fusing only the scans after the robot moved
took the grid's updates on a recorded day of lab trips from every level
scan to 6 918 of 10 947, and the mapping as a whole from ~21 to ~20 ms a
sim second; a day that stands still less saves less. The matcher is
deterministic (einsum's own loops, no BLAS product) and its field is kept
state: a scripted home day hashes IDENTICAL in two processes (2 025 state
samples, `determinism_spike.py`), and saved at t = 419 s and carried on
in a new process it is IDENTICAL after the restore (1 186 samples,
`--resume-at 400`). Not done: a scan
tilted past `MAP_TILT_RAD` is neither matched nor fused, so a wheel pump
while tilted runs free until the robot is level and the search finds it;
a robot lost beyond the search's window has only the dock
(#381's loop closure); the first trip through new territory carries its
own drift into the map it lays (0.2-0.3 m by the lab), which later visits
match to but do not correct; and the rover's map gate still reads its tilt
off the pose, the one sensor left that is the sim's (it goes with the
rover).

## The first quadruped deploy (issue #387)

The body #377 sized, walking on its policy and getting up on its own, put
into the house the rover lived in (`legs/world.py`: the rover taken out by
what names it, the quadruped -- or the served pair -- and #378's dock put
in, the dock's board against the living room's south wall at x 3.5). It is
a `Body` (`legs/body.py`) over `QuadMission`, a `Navigator`: the map, the
planner, the drive and the peer rules the rover had, moved out of
`HubMission` into `navigator.py` whole -- a scripted home day of 1012 sim s
hashes IDENTICAL against the rover before the move.

**The rest reflex and the posture.** Beneath every command a posture
machine: `standing` walks what it is told; 8.6 s with no motion command
(`posture.T_REST_S`, #377's break-even) lies the body down (2.2 s, then a
second limp to settle); a motion command to a lying body stands it first
(2.2 s); a torso past 60° (the topple angle every body is judged down at)
or slumped under 0.20 m for half a second is a fall, and the get-up
policy drives until it is upright at its height for half a second. The
lie-down and the stand-up moved from `quad_spike.py` into `legs/posture.py`
and fly the spike's tables byte for byte.

**Getting up, and the `stuck` death.** Shoved over 48 times in six places
(`quad_spike.py --shove`: the open floor, beside the couch, against the
south wall, the hall, the kitchen counter, a doorway), #389's get-up stood
from all 37 falls -- median 2.0 s, p95 3.5 s, the slowest 4.6 s. #377's,
which this deploy flew first, left 4 of 35 wedged upside down and took
13.4 s once. So a body that rights itself is `stuck` after 20 s down
(`QuadBody.stuck_after_s`), not the rover's 2 s, and its death line says
"fell and could not get up in 20 s".

**The LIDAR's plane is over the furniture.** The quadruped's LIDAR sits on
a rear mast so its plane clears the stowed arm: 0.506 m up. The couch
(0.50 m) and the bed (0.40 m) are UNDER it, and a planner fed the LIDAR
alone routes through both; the rover's plane (0.223 m) saw them. The D435
on the nose does, so its points between 0.08 and 0.60 m over the floor are
a layer the planner plans round (`QuadMission._fold_low`,
`_planning_grid`); the explore's frontiers and the streamed map read the
same view (`QuadBody.grid`). It is the first thing on the robot that
decides off the depth camera, and it is NOT the height map: a layer of its
own, log-odds like the grid. Three rules made it usable, each after it
shut a door:

- only points within 1.8 m (σ 12 mm): at the part's 3 m range a floor
  pixel read 5 cm high often enough to leave single obstacle cells in the
  middle of doorways, and at the inflation's 0.35 m a side a 1.0 m door has
  0.30 m of plannable width -- one stray cell closes it;
- three frames of evidence before a cell is an obstacle, and a floor point
  clears one;
- never within three cells of what the LIDAR maps: walls are the LIDAR's
  (and matched, #386), and the camera's view of their bases through a
  drifting pose only thickened them.

**The planner and the drive, as the house found them.**

- *The front stop sat beyond the planner's clearance.* A centre 0.35 m from
  a wall puts the rear-mast LIDAR 0.50 m from it, and a 0.64 m stop fired
  on every waypoint the planner laid along one: a drive toward the
  bedroom's divider backed off every second for 135 s. It is 0.45 m
  (the nose 6 cm off), and it tests the CORRIDOR ahead (±0.12 m), not the
  rover's 0.35 rad cone, which at that range reached 0.15 m out -- a wall
  alongside at 0.16 m fired it every scan, and the robot backed 2.3 m into
  a corner.
- *The dead band made arcs.* The policy walks nothing under ~0.2 m/s
  (#378), so a small command is raised to 0.25 m/s -- and the rover's
  driving law crawls while it turns, so raised, the crawl swept 0.25 m
  arcs into the walls it was turning away from. A crawl with a turn to
  make is a pivot (`command_for`).
- *A detour read as a stall.* The rover's drive gives up after 10 s
  without getting closer to its goal in a straight line; from the
  bedroom's corner the only way to the hall walks 2.5 m round the divider
  first, and every such drive "stalled". A quadruped's progress is read
  along its route (`Navigator.PROGRESS_ALONG_ROUTE`); the rover's is not,
  and its day is unchanged.
- *A press reversed into the far wall.* The thighs scraped the kitchen
  counter's end walking past it; reversing replanned the same scrape until
  the hind knees met the north wall. A press on a flank steps SIDEWAYS
  away (`retreat_from`), one on the nose backs off, one on the hind knees
  steps forward. (The press itself had first fired on the floor: the
  house's floors are boxes named `*_floor_geom`.)
- *The explore gave up on reachable floor.* The rover's explorer A*s the
  20 nearest frontiers and blacklists each failure; the depth layer's
  small obstacles seal pockets of frontier near the robot, and three such
  rounds ended the explore with the kitchen and the garden unseen. The
  quadruped's drops frontiers off its own connected floor before any A*,
  and blacklists none (`Body.plan_frontier`).
- *A stand-in replanned every step.* A goal just off the plannable floor
  is driven to its nearest reachable cell; standing on it, the waypoints
  run out and the drive replanned every physics step until its stagnation
  cut -- 4 190 plans in 90 s of a pair, 15 ms each. A plan asked again
  within 0.5 s from the same spot is the last plan.

With those, a 260 s explore tours the living room, the bedroom, the hall,
the kitchen, the workshop and both gardens, and the belief ends within a
few centimetres of the truth (scan matching, #386).

**The planner's sizes.** The body's outline, measured off its geoms:
standing it reaches 0.38 m from the torso's centre (the hind knees, 0.36 m
behind it) and 0.20 m to either side; lying, 0.43 m; fallen, 0.63 m from
the middle of its footprint (worst of 20 random drops). The map's
inflation stays the rover's 7 cells (0.35 m square on, 0.25 m on a
diagonal): it covers the half-width, and the body walks forward through a
door. The disc round another quadruped is 14 cells (its outline lying plus
this one's half-width), 18 round a fallen one.

**The dock, from the loop.** The approach is #378's (`dock_routine`): turn
to the dock's heading at the standoff (the walk there arrives facing where
it came from, and the search turns reach only ±75°), find the board, walk
in by it, stop, check the line-up, lie down; the pads conduct or it backs
out and tries again. Lying there, the board anchors the reckoning (a few
mm); the charger puts 216 W in and the body draws 15 W lying, 201 W net.
A day that starts low docks first try and charges 35 → 90 % in 198 s on
the 20 Wh test pack.

**Energy** (`energy_spike.py --world home_quad`, 40 Wh pack): the explore
draws 84.7 W (6.11 Wh over 260 s) -- the legs' windings and shafts off the
drivers' own torques (`LegPack`) plus the electronics' 14.9 W; the charge
is 201.1 W net. No errand rows: the body has no arm. The served pack is
the real one, 194.4 Wh (twelve P45B cells).

**Speed.** Profiled on a pair, the first cut ran 0.56x real time on the
dev machine: the replanning storm above, and the legs' odometry at
0.78 ms a step -- most of it `np.cross` on 3-vectors, replaced by the same
arithmetic written out (bit-identical over 200 000 random pairs, a tenth of
the cost). With those and lookups in place of `np.isin` on the physics
seam, the same day at 0.96x on a quiet dev machine, IDENTICAL end state to
before. The deploy box's core measured 1.17x the dev machine's (#377).

**The reserve** (`energy_spike.py --world home_quad --reserve`): over the
rover's worst-case route home (the loop's south-west corner, the sidewalk,
the gate, the garden door; 44.6 m) the walk costs 2.64 Wh (59 mWh/m, twice
the rover's) and the dock 0.43 Wh; with one failed docking on top (another
approach, and the stand-up and lie-down it costs), `legs.world.RESERVE_WH`
is 3.6 Wh.

**A pair, twice alike -- and what a stand-up did to the other robot.**
`determinism_spike.py --pair` flies the served pair's day with a fall and a
death arranged (`pair.arrange_hazards`: the second robot knocked onto its
side at 40 s and its pack emptied at 120 s; a 30 s restart timer; an inbox
each, as `serve.py` builds them) and hashes the whole world. The first
flight hashed IDENTICAL twice and was wrong both times: the timer's
stand-up placed the second robot with `start_at`, which settled the body by
stepping the physics for a second on its own -- and for that second the
first robot, walking in the same loop, ran no policy and no odometry. It
went over, got up 0.44 m from where it believed it was, found no route to
the dock and died `stuck`. The quadruped's `start_at` steps nothing now:
the body settles under the day's next command. (The rover's still settles a
second, coasting its pair's other rover on its last wheel command; that is
milder, and changing it moves the rover's parity.) A stand-up also never
lands on another robot: a start pose with a body within 1.0 m is passed
over for the next commissioned one (`HubLifecycle.up_pose`). Flown again:
IDENTICAL over 924 samples to 461.6 s -- the second robot knocked over at
40 s and up in a second, dead `flat` at 120 s and waiting, stood up at 150
s and exploring on, lying down by reflex at 260 s once it had nothing to
do; the first docked at 170 s and charged 11 -> 90 % in 284 s, stood and
backed off. Nobody else fell. Restarted mid-day (`--resume-at 40
--battery-fraction 0.45`), a robot saved as it left for the dock (246 s)
and carried on in a new process is IDENTICAL to the day flown straight
through over 869 samples: the 115 s walk, the docking, 311 s lying on the
pins (3 -> 90 %), the stand and the back-off.

**Open: two robots meeting head-on hold for each other.** The re-recorded
pair fixture has them meet 0.9 m apart just inside the workshop doorway;
each holds for the other (the rover's #328 rule: a robot is held for, never
backed away from) for ~66 s, the rest reflex laying each down in turn, until
a drive's patience runs out and one goes round. It ends by itself; nothing
yet decides who yields.

**What is true now:** the served world is `home_quad_pair`
(`serve.py --pair --body quadruped`); the rules above are pinned in
`tests/test_quadruped.py`; the prompt says where a quadruped charges and
how it dies in its own words, and nothing about upkeep where there is none
(`overseer.mortal_rule`, `for_body`); the constitution's body paragraph is
swapped for the quadruped's (`constitution.for_body`, asserted).

## A gentler get-up (issue #389)

#377's get-up policy sprang. Paid for every step it stood, the sooner the
better, it stood from the belly in 0.2 s with a knee at the drivers' 22 N·m
peak and the torso rising at 1.4 m/s: hard on a gearbox, a hazard to a
hand (#379's safety section), and it throws #378's arm and a tool. Now
`Pluggy-Quad-Getup` (`training/quad_train/getup.py`) pays the STAND (the
height and the pose) only from 2 s after the release (`BUDGET_S`), so
standing sooner earns nothing, and it charges for getting up hard: each
joint's torque past the motor's continuous rating (6.71 N·m; squared, in
ratings) and harder past 60 % of the peak (13.2 N·m), the joints' speed and
acceleration, the torso rising faster than 0.25 m/s, and the action's rate.
The charges start at a tenth of their weight and reach it at iteration 900
(`GENTLE_STAGES`). Being the right way up pays from the release.

**The committed policy** (2000 iterations, 37 min on a rented RTX 4090
beside two other runs) against #377's and the scripted fold-and-push, in
our physics (`quad_spike.py --getup`, `--shove`; stood is the served
body's own test):

| | scripted | #377's | #389's |
|---|---|---|---|
| from the belly: stood in | 2.2 s | 0.2 s | 1.5 s |
| ... peak torque, abduction / flexion / knee, N·m | 2.2 / 0.9 / 5.9 | 3.0 / 13.4 / 22.0 | 5.3 / 8.6 / 10.1 |
| ... the torso's fastest rise, m/s | | 1.43 | 0.21 |
| ... energy to the stand, mWh | 44 | 23 | 31 |
| #377's 20 drops: stood | | 20 | 20 |
| ... first touch to stood, median (longest), s | | 0.3 (0.8) | 1.3 (2.4) |
| ... past 60 % of the peak after the landing | | 10 | 1 (14.2 N·m) |
| 100 fresh drops: stood | | 100 | 100 |
| ... past 60 %: in the air / the landing / after it | | 80 / 96 / 62 | 78 / 35 / 7 |
| the house, shoved 48 times: fell / up | | 37 / 32 | 37 / 37 |
| ... fall to standing, median (slowest), s | | 1.2 (28.0) | 2.0 (4.6) |
| ... peak after the landings, N·m | | 22.0 | 14.1 |

"The landing" is the first 0.3 s after anything touches the floor
(`LANDING_S`). #377's 87 mWh was the whole 6 s trial, 5.8 s of it standing
still; to the stand its spring cost 23 mWh, less than the scripted routine
because it was short, and the rise costs 31.

**Held back to 2 s, being right way up never learned to roll.** With
`belly_down` gated like the stand, three runs (the ramp; full charges from
the first iteration; three times the charge past 60 %) rose gently from
the belly by iteration 1000 (0.9–1.2 s at 10.5–11.5 N·m) and stood from
at most 16 of 20 drops, every failure on its back to the end. In the house
the ramped run got up from 9 of 37 shoves, and those traced lay on their
backs: the shove's spin flips the body, and one settled from its side onto
its back, the state its charges made cheapest.
#377's rolls were flips, upside down to upright in 0.2 s with up to three
joints briefly past 60 %, and a policy learning to stand while its charges
rose never found one. Paid from the release (on its side a quarter, on its
back nothing), the roll came from scratch by iteration 1350. A warm start
from #377's actor under the same rewards kept its roll too (20 of 20 at
each checkpoint flown) and was not needed. Full charges from the first
iteration also learned to stand (16 of 20 by iteration 1000, as the ramp
did), so the ramp is not what makes it stand.

**The landing is the policy's own aim.** At the worst moment of each of
#377's landings past 60 % (19 of #377's 20 drops) the PD's stiffness term
(where the policy aims the joint) was the larger, not its damping term
(the joint's speed): it sprang at the floor too. The committed policy's
landings pass 60 % in 35 of 100. What is left in the air is each drop's
start, the joints at random angles and the first targets swinging them into
place, which no fall on the robot starts from. After the landing, 7 of 100
pass it: five in its tail, 0.31–0.38 s after the touch, the damping term
the larger in four, and two rolling off the back (13.3 and 15.4 N·m).

**An arm left out no longer stops it** (#378's section, "A fall throws
the tool, and the arm folds").

## Walking into the unknown (issue #381, the walking stage)

A fresh quadruped -- its start pose in the living room, an empty map, one
look round -- sent once to each of the property's 24 zones
(`scripts/unknown_spike.py`: a zone's centre moved to the nearest floor
0.45 m from anything, each walk in its own process, 300 s a walk) reached
**9**, and every failure was the same one: `no route` at a stand-in pressed
against a wall. The planner planned over mapped floor only and aimed an
unmapped goal at the reachable cell nearest it in a straight line ("A goal
out of sight is aimed at through the nearest wall"), which from inside a
house is against the wall between the robot and the goal: the workshop's
wall for the kitchen, the living room's south wall for the south garden,
the garden fence 3 m from its gate for the street. The lab and the store
arrived only because every door on their way (the gate, garden_2's, the
lobby's, the lab's) sits on y = 3.1, the straight line -- in 103 and 118 s,
past the 60 s a procedure's `drive_to` and a decided `explore(zone)` gave a
walk: with the budgets the robot had, 4 of 24.

**Unknown is floor at a price** (`mapping/optimistic.py`, the quadruped's
`Navigator.OPTIMISTIC`). A cell nobody has seen costs `UNKNOWN_COST` a
metre against a mapped cell's 1; a known wall grown by the inflation is not
floor on either side of it (the far side of a wall is not a way through);
the plan runs as straight as the map allows, the LIDAR sees the walls it
crossed as the robot walks, and the next plan (every 2 s) goes round them.
A doorway is found by finding the walls either side of it. No route at all
is then a fact about the map: the walls the robot has seen enclose the
goal. The stand-in stays for a goal INSIDE a wall's inflation (a board's
use pose), and a goal on the floor but walled off is no plan. The same 24
walks after:

| zone | true route | before | after |
|---|---|---|---|
| kitchen | 13.3 m | no route, 2.4 m short | 38.6 s, 13.6 m walked |
| workshop | 11.9 m | 40.0 s, 14.1 m | 36.4 s, 12.0 m |
| hall | 5.6 m | 20.4 s, 6.3 m | 17.7 s, 5.6 m |
| living | 0.3 m | 3.6 s | 5.4 s |
| bedroom | 3.7 m | 10.2 s, 3.7 m | 10.7 s, 3.8 m |
| garden | 7.1 m | 67.0 s, 13.3 m | 20.8 s, 7.6 m |
| garden_south | 9.4 m | no route, 2.4 m short | 28.3 s, 10.1 m |
| sidewalk | 13.4 m | no route, 1.1 m short | 39.4 s, 14.2 m |
| street | 14.3 m | no route, 3.5 m short | 38.8 s, 15.1 m |
| sidewalk_2 | 16.5 m | 64.7 s, 21.0 m | 44.6 s, 17.2 m |
| garden_2 | 19.7 m | no route, 1.9 m short | 52.3 s, 20.3 m |
| lobby | 22.7 m | no route, 2.0 m short | 61.2 s, 23.3 m |
| lab | 25.0 m | 118.3 s, 32.6 m | 64.6 s, 25.6 m |
| store | 29.5 m | 103.3 s, 34.0 m | 78.3 s, 30.1 m |
| sidewalk_north | 24.9 m | out of time, 1.9 m short | 159.3 s, 60.6 m |
| sidewalk_2_north | 23.6 m | 118.9 s, 32.6 m | 65.8 s, 24.2 m |
| sidewalk_south | 31.1 m | no route, 4.3 m short | 163.7 s, 62.0 m |
| sidewalk_2_south | 28.3 m | no route, 12.5 m short | 75.6 s, 29.0 m |
| sidewalk_west | 42.7 m | no route, 1.1 m short | 243.8 s, 93.3 m |
| sidewalk_east | 36.2 m | no route, 1.2 m short | 165.1 s, 63.6 m |
| street_north | 17.2 m | no route, 4.8 m short | 54.5 s, 20.1 m |
| street_south | 23.3 m | no route, 8.1 m short | 84.2 s, 30.6 m |
| street_west | 43.6 m | no route, 3.5 m short | 246.2 s, 94.5 m |
| street_east | 37.1 m | no route, 3.3 m short | 165.8 s, 64.7 m |

**24 of 24**, 15 of them within 7 % of the true route and 16 within 20 %
(the "true route" is the same planner over the compiled world's own floor,
nothing left unknown; the living room's 0.3 m aside). The other seven all
end on the loop round the two houses, at 1.3-2.4 times the route, and it is
the right answer to the question the robot was asked: from the middle of a
house it has never seen, the cheapest way anywhere is
through the walls it has not seen yet. Sent to the west street it tried the
bedroom's north wall, then the kitchen's west wall -- the house has no door
on either side -- found both, and went out by the garden door and round the
north sidewalk. A first walk into the world is an exploration, and its map
is kept: sent back to its start and there again (`--again`), the second
walk to the west street took 117 s and 45.2 m (the route is 43.6), to
the south street 60 s and 23.2 m (23.3); the near ones were near their
routes both times. ⚠ One walk back did not arrive: from the north sidewalk
it shuttled east and west for its whole 300 s (108.7 m), because the house's
north wall is 4 cm thick, thinner than a cell, and the robot walking along
it carves it out of its own map with grazing rays -- the map had it free
from x = -3 to 6 m -- and the planner routes through where it was (#401: a
choice of sensor model, filed, not made here).

**Why this price.** A known detour is taken while it is under twice the
unknown shortcut (`UNKNOWN_COST` 2). It is a choice, not a fit: swept to 1.25 and 4
over six fresh walks it moved four not at all and two by 3 s, because a
fresh map offers no known detour to weigh -- every way out is unknown. What
it decides is a second walk's: the way it knows against a shortcut it has
not seen.

**The map still growing is progress** (`PROGRESS_MAP_GROWTH`,
`MAP_GROWTH_CELLS`): a walk into the unknown turns back when it meets a
wall, and a route that lengthened because the map showed a wall is not a
stall. The rover's progress rule (the straight line) and the quadruped's
(#387's route length) both read that as no progress. Walking into new floor
the map grows by 800-1 850 cells a second; standing still 20 s after
arriving, by -5 to +195 cells (a fused scan's edge at the LIDAR's reach):
`MAP_GROWTH_CELLS` is 400 in 10 s, 4x that.

**The search is scipy's compiled Dijkstra, on a 10 cm lattice built once.**
Measured on the home grid (469 200 cells) on a busy dev box, a fresh map
from the start pose: A* in pure Python with the unknown priced took 0.5-10 s
a far plan (it expands every cell cheaper than the goal, and the unknown's
price makes that most of them); scipy's Dijkstra over the 5 cm lattice
~260 ms, over a 10 cm one built per plan ~240 ms. What made it cheap:

- a lattice cell is 2 x 2 map cells and passes only if all four do
  (`BLOCK`): a gap keeps a path at any alignment while it is 3 map cells
  wide, and every door of the house is 6 once inflated;
- the CSR graph is built once per grid shape (~0.1-0.4 s, shared by a
  pair) with eight slots a node, and a plan writes its weights -- a step at
  a time into contiguous arrays and interleaved in one copy (6.5 ms, where
  writing each step into its stride-8 slots cost 15.2);
- an `inf` weight is no edge to scipy, so a wall is a weight, not a
  rebuilt graph;
- a goal on the floor is searched for no further than the straight line to
  it at the dearest cost and 2 m (`LIMIT_SLACK_M`), then without a limit if
  that missed: a 3 m hop settles 1.9 ms of search where the whole lattice
  takes ~20.

Alone (the busy box's best of nine) a plan is ~5 ms of search near and
10-20 far, on ~8 ms of costing the map (the inflation's distance transform
over the whole grid, the next lever); in flight the median plan was 35-65 ms.
Over the 24 walks planning was **30.7 ms a walked sim second (2.6 % of the
wall clock)** against the old planner's 18.7 (1.5 %), on the same busy box
-- dearer a plan, and fewer of them: no stand-in replanned in a storm.

**The patience is the robot's, and its interrupt reaches the walk.**
`drive_to(x, y, patience=S)`: 60 s unsaid, 600 at most, never past the
procedure's own budget; the longest walk above took 246 s. A decided
`explore(zone)` walks with `ZONE_PATIENCE_S` (300 s: all 24 fit). A walk
asks the robot's own hazard rows every second (`go_to_routine(stop=)`) and
ends where it stands on the answer -- `DRIVE_STOPPED`, never one of the four
ways a drive gives up -- and a procedure it stops is `stopped:
interrupted`, as between two verbs.

**Found on the way: no procedure on legs had walked a step.** Every verb
that moves takes the carrying pose first (#347), which asked the body for
its arm, and the quadruped has none: `drive_to`, `face` and `drive` raised
before they moved, on every run since the first quadruped deploy. The
observatory's two days on `5c6e6cf`: 6 565 procedures validated, 6 562
aborted, 3 ran; of the last 196 aborts read, 193 were one procedure run
again and again at one sim instant, the loop spinning with the world stopped
(#400: an action that takes no sim time can be fired again at once).

**What is true now.** The quadruped walks into the unknown and the rover
does not: its planner and its day are unchanged until #376's stage C -- a
scripted rover home day hashes IDENTICAL against staging (2 025 samples to
1 012 s). The quadruped pair's day hashes IDENTICAL in two processes (1 046
samples to 522.6 s), and a quadruped's day saved at 232.7 s, as it left its
explore for the dock, and carried on in a new process is IDENTICAL after
the restore (942 samples, `--resume-at 40`). `scripts/unknown_spike.py [--before]
[--again]` is the measurement; the rules are pinned in
`tests/test_unknown.py`. Not done: the map is still the
property's rectangle (#381's stage 4), loop closure (stage 3), and what a
walk into the unknown costs is not yet in any offer (stage 7).

## The arm on the served body (issue #405)

#378's arm is `CHOSEN`'s: `BodySpec.arm` is `ArmSpec()`, and the budget's
0.9 kg arm and 0.25 kg tool placeholders are 0, so the robot is 9.36 kg
without a tool (the placeholder body was 9.34) and fourteen drivers stand
by. #377's torque, thermal and energy tables were flown on the placeholder
and still are: `model.SIZING` is that premise, and the torque-driven
instruments (`quad_spike.py`'s tables, the scripted gait's tests) fly it,
their `ctrl` the legs' twelve. The policy flights fly the chosen body with
its arm held where it stands (`PolicyDriver.step`): unheld, it falls
across the nose camera within half a second -- the dock test's lying robot
lost the board that way. The fork's eleven 2 g geoms had been counted on
top of the plate's 80 g budget; the plate's own geom carries the rest now.

**The driver holds the arm on every step, off `qpos` alone.**
`QuadMission.arm` is an `ArmDriver` stepped in `_before_step` whatever the
posture: at the stow unless a program moved it. #378's driver took its
gravity feed-forward off MuJoCo's Jacobians -- three bodies, each a
Jacobian 122 columns wide in the house -- and cost 100 us a step, 43 of
them the Jacobians and 22 numpy's overhead on two-element arrays: a fifth
more work a step for the served pair, which runs near real time. And a
restart forwards the saved world at its instant, where a running world's
step reads the kinematics one step old, so a controller that reads them on
every step parts the two worlds at the first step back (the policy reads
them only at a decision, the fall check only against a threshold). The
feed-forward is the arm's own planar model now -- each link's mass and CoM
off the model, the joints' angles and the torso's attitude off `qpos` --
equal to the Jacobians to 1e-15 N·m at any pose, attitude and payload, and
the two joints' PD and envelope in plain floats: 12.5 us a step, measured
at load 18.

**Held, the stow barely moves** (the served body in the house; the arm's
draw is its two motors' windings and shafts):

| | off the stow, worst | motors' peak, shoulder / elbow | the arm's draw |
|---|---|---|---|
| standing | 0.02° | 0.8 / 0.5 N·m | 0.6 W |
| walking 0.3 m/s | 1.1° | 4.4 / 2.4 N·m | 0.8 W |
| trotting 0.8 m/s | 2.6° | 2.3 / 3.9 N·m | 2.6 W |
| turning 0.8 rad/s | 3.8° | 2.4 / 4.0 N·m | 4.3 W |
| lying down to rest | 0.1° | 1.0 / 0.7 N·m | 0.6 W |

**Getting up with the arm needed no training.** #389's policy, trained with
#377's placeholder that collided with nothing, gets up with the real arm on
its back as it did without it (`quad_spike.py --getup`, `--shove`):

| | the placeholder | the arm, held stowed |
|---|---|---|
| from the belly: stood in; peak a/f/k | 1.5 s; 5.3 / 8.6 / 10.1 N·m | 1.5 s; 5.3 / 8.6 / 10.1 N·m |
| #377's 20 drops: stood; median; worst after the landing | 20; 1.5 s; 14.2 N·m | 20; 1.4 s; 12.6 N·m |
| 100 drops: stood; past 60 % after the landing | 100; 5 | 100; 4 |
| the house, 48 shoves: fell / up; median (slowest) | 37 / 37; 2.0 s (4.6) | 36 / 36; 1.9 s (5.8) |

A fall folds the arm (below), so the get-up always starts from the stow,
and that is the pose training fixes now: `quad_train.robot.freeze_arm`
turns each arm body to the stow its driver holds and takes its joints out,
so a future run falls on the arm's real geometry while a policy's joints
stay the legs' twelve (a two-iteration run builds, its actor's observation
42 wide, as the committed get-up's).

**The fold stays, and folds before a rest too.** #389 left open whether
step 4 keeps #378's fold. A fall (the body's own test: 60°, or slumped) and
the rest reflex's lie-down both aim the arm at the stow and let go of any
payload: a gravity seat cannot hold a tool upside down, the get-up rolls a
folded arm as it was trained to, and a robot lying down should not lie on
an arm held out.

**A program moves the arm's joints.** `shoulder` and `elbow` are body axes
and sensors (`axes.BODY_AXES`; `world_facts` gives each body its own, so
the rover's world names neither and the quadruped's neither `lift` nor
`arm`), and `move` is a legs verb. A move stands a lying body first and
keeps the rest reflex off while it runs; the verbs that walk fold the arm
first (`steps.travel_pose`: the shoulder, then the elbow, so the forearm
comes in over the body rather than under it), and so do lying down to
rest and every walk of the body's own (`QuadMission._twist_routine`): a
pose outlives its procedure, and the loop's next walk dragged it on the
floor or carried it across the dock's board. A move the joint cannot
finish in time, or that a fall takes from it, is a failed step, and its
hold on the rest reflex (`QuadMission.working`) is never saved: kept, a
restart restored it with nothing left to clear it. A move is motion to the
reflex, which counts its 8.6 s from the move's end: counted from the last
walk, a wait after a move lay the body down and folded the pose it had
just set. The motors'
`ctrl` is a torque, so what an axis is held to is `Body.setpoint` -- the
rover's `ctrl`, the quadruped's driver's goal -- and a death's
`setpoints` read that.

**The fork is the body's front.** Folded, the fork reaches 0.348 m ahead of
the torso's centre, 0.14 m past the nose, and it cannot fold further back
without the forearm crossing the LIDAR's plane (the stow above). The front
stop, set at 0.45 m for the nose, let the fork meet a wall first: a walk
to a goal inside the planner's clearance ended in 346 bumper steps, every
one the arm, and a 105 s explore made 19. At 0.53 m
(`QuadMission.FRONT_STOP_RANGE`, from the LIDAR 0.15 m behind the centre)
the fork stops 3 cm short: 0 presses on the same walk and explore, the same
map, and #399's 24 zones still 24 of 24, the far ones within 4 % of their
time. It stays under the planner's clearance because a waypoint is dropped
0.08 m out (`navigator.WAYPOINT_REACHED_M`), and a goal at the clearance
stops inside `CLOSE_ENOUGH_M`.

**The served speed.** The arm adds physics -- 26 geoms, 6 degrees of
freedom, 4 tendons and 2 equality constraints a robot -- and a driver step
a robot; its motors' bill is in floats (`LegPack._arm_w`: numpy on two
elements cost 6.3 us a step, for the same bits). Side by side, the served
pair free-ran at 0.82-0.83x real time on staging and 0.80-0.81x with the
arm (`serve.py --pair --free-run`, both under the dev machine's same load):
about 2.5 %.

**Energy.** `energy_spike.py --world home_quad` with the arm aboard: the
explore 6.067 Wh over 251.4 s (86.9 W; 84.7 without the arm), the dock's
charge 199.5 W net (201.1); the worst-case return 2.699 Wh over 44.64 m
(60.5 mWh/m; 59.1), its dock leg 0.270 Wh against #387's 0.432, and with
the dearer carried the reserve is 3.7 Wh (`legs.world.RESERVE_WH`, 3.6).

**What is true now:** the served quadruped carries #378's arm stowed and
held, folds it on a fall, before resting and before walking, stops for
walls before its fork meets them, and a program on `autonomous` may move
its two joints; `tests/test_quad_arm.py` pins each rule. A
quadruped's day saved at 257.6 s and carried on in a new process is
IDENTICAL after the restore (330 samples, `determinism_spike.py --world
home_quad --resume-at 40`), and the pair's day with its arranged fall and
drain is IDENTICAL twice (1 050 samples to 524.8 s, `--pair`). It takes no
tool: the rack at its reach, fetch and stow, and the carried-tool filter in
the scans are this issue's stage B.

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
