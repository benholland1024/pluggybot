# Simulation Notes

MuJoCo lessons from building PluggyBot, each one paid for. An entry says what
broke, why (the physics), what is true now (the constant or the rule) and what
pins it. The story of how a thing was found lives in git and the issues; a
measured number lives at its constant with its failure mode attached
(CLAUDE.md, "Working style"). Read this before touching `models/`, contact or
actuator parameters, the hands, or anything that trusts odometry. The
pattern docs say *what the constraint is*; this file says *how we found
out*, in the order it was paid for.

Three bodies came before the humanoid: the plug-anywhere robot (milestones
5–7), the wheeled rover (milestones 8–15) and the quadruped (#375–#559).
The rover went in #376, and the tag `rover-final` holds it, its worlds and
the whole of every story it paid for. The quadruped went in #559's stage 5:
its notes -- its body, dock, arm and rack, its tools and its jobs -- and the
imagination's (#466) are at the tag `quadruped-final`, among them six the
observatory's periods cite by name: "A drawing that set off from the rack"
(#347), "A gentler get-up" (#389), "The rack at the arm's reach" and "The
fork is the body's front" (#405), "The feed on legs" (#403) and "Lying
still, the heading holds" (#425). What this file keeps of those eras is
what still binds the humanoid, any body, or the house it lives in, cut to
what is true now; the humanoid's own entries start at "The humanoid body".

## Physics modeling rules

### Use `integrator="implicitfast"`
MuJoCo's default explicit Euler adds energy to velocity-dependent forces --
velocity actuators and joint damping. Every world here integrates
`implicitfast` (`humanoid/model.py`, `home/world.py`), as most Menagerie
models do.

### A light mass on a stiff servo chatters
A 50 g wheel under a velocity servo overshot its target within one 2 ms step
and chattered at the timestep frequency: the robot bounced instead of
driving. What was missing is honest inertia: `armature`, the rotor's inertia
reflected through the gearbox, which grows with the gear ratio squared. On a
rig, a constant-force `<motor>` with joint `damping = F/v` does the same. A
humanoid joint carries its part's armature, the sheet's output-side inertia
(Parts.md).

### THE caster lesson: MuJoCo combines pair friction as the elementwise MAX
At equal `priority` a contact's friction is the element-wise **maximum** of
the two geoms' values (condim combines as max too; solref/solimp by priority,
then `solmix`), and the floor's default is 1.0. The rover's
`friction="0.001"` caster was a full-grip rubber ball for weeks, the hidden
cause of a pitch resonance and 12.6 % odometry creep, until Menagerie's
Stretch idiom, `condim="1" priority="1"`, went in. **What is true now:** a
per-geom friction does nothing unless its geom has priority: the humanoid's
soles and gripper pads set it (`humanoid.model.SOLE_FRICTION`,
`PAD_FRICTION`;
`test_humanoid.py::test_the_soles_and_the_pads_grip_with_their_own_friction`).
Know the pair-combination rules before trusting any per-geom contact
attribute.

### The grip that leaked: soft contact drifts under sustained load
A block crept out of the rover's claw at ~8 mm/s, identically across a 3×
range of grip force, and slower lifts were worse: no real friction failure
does that. It is MuJoCo's regularised friction drifting under sustained load.
**A negative result is not a diagnosis**: "tripling the grip changed nothing,
so it is not slipping" was the wrong inference, and Ben, watching the render,
saw it creep. `noslip_iterations` cured the grip and broke the coupling, and
that was a friction bug in a solver costume: the peg sat on MuJoCo's default
μ 1.0, whose 45° friction angle is the V's flank, so it seated only through
solver drift; at steel-on-plastic's 0.4 it seats with or without the pass.
The same drift toppled leaning block stacks that should have stood (#120: a
2 + 4 mm lean fell at 16.9 s). **What is true now:** creep is fixed at its
source with a hard contact: `home.props.GRIP_SOLIMP` on the house's blocks,
on which a tower stands iff its centre of mass is over its support, and the
humanoid's gripper pads (`humanoid.model.PAD_SOLREF`, `PAD_SOLIMP`: on
MuJoCo's soft default a cube crept out of the quadruped's claw, #407). Any
free body meant to rest on another for more than a few seconds needs it, or
the sim grades the solver (Challenges.md §4).

### One solver policy (issue #3): `noslip_iterations` is 0, always
Two robots share one world, so a solver setting is one number for everyone.
Measured on the whole rover, every noslip ≥ 1 half-seated a module under
±3 mm of hand-off jitter (on the fork, not powered) while the bare coupling
rig called noslip 3 its best row, and turning the pass on at all doubled the
step. **Measure a solver policy on the full system, not the rig.** Each thing
the pass "fixed" was a part misbehaving: the peg's friction (above), and a
pen that drew badly because the parked wheels ROLLED -- a velocity servo
commanded 0 resists speed, not force -- until the gearbox's parking brake
(`frictionloss`) went in. **A solver mode that fixes a behaviour is a claim
about which part misbehaves, and it does not name the part.** What is true
now: `noslip_iterations = 0` everywhere, no code mutates solver options at
runtime, and creep is fixed per part -- a hard contact where a contact
drifts, friction where a joint rolls. `tests/test_noslip_policy.py` guards
it.

### Every honest piece of physics sends a bill to its controllers
The rover's parking brake put a static-friction deadband under every
proportional turn: a command under breakaway cannot move a stopped wheel,
and a controller that shrinks its command with its error parks itself in the
band (a 0.23° square-up went from 9.5 s to 55 s). A walking policy has
the same shape: it walks nothing under its dead band. The bill is paid in
the controller, never by removing the physics: a nonzero command is floored
above the band (`control.turn_command`'s `W_BREAKAWAY`; a walk's crawl,
`humanoid.mission.command_for`, "The first quadruped deploy"). And when
behaviour looks violent, check that the actuator could exist in that
weight class: a 1.4 N·m motor on a 1.1 kg rover asked ~40 N of thrust of
~3 N of traction.

### A dropped sphere rolls forever
Sliding friction does not touch a rolling ball: a released seed rolled
586 mm identically at μ 0.7 and at μ 1.0 with priority. Rolling resistance is
a friction dimension of its own, solved only at `condim="6"` (then it
stopped in 14 mm).

### A position setpoint is ramped, never stepped
A stiff position servo handed a step delivers an impulse: an 80 mm carriage
jump threw a module off the rover's fork, both poles open, where the same
80 mm in 5 mm steps held, and slammed jaws batted a block away. What is true
now: `Body.ramp_routine` walks an actuator's setpoint to its target, and a
hand's joints move no faster than `reach.JOINT_RATE`.

## Test & world hygiene

- **A physics test world carries no scenery.** Scenery once parked a box in
  a drive-test lane, and a re-measurement drove into a board placed two hours
  earlier. Copying the floor into an including file doubles every contact: a
  "repeated name" MJCF error means *delete* the duplicate, not rename it.
- **Every debugged failure becomes a pytest assertion, shown failing first.**
- **A capability is not demonstrated until it has been demonstrated twice
  in a row**: the second cycle starts from the state the first one left
  (the rover's pen, #10: a pick inherited the lift a stow had left behind),
  so every attempt restores its own entry state.
- **A number can be blind to what matters.** A mirrored figure scores
  identically on bounds, ink length and form error (the rover's pen, #11),
  so a sign convention gets a test of its own. And before building a
  recogniser, measure how far apart the things it must tell apart are, in
  the units the machine's error is measured in: a drawn 6 and 8 were closer
  to each other (1.7 mm at 55 mm caps) than a well-drawn 6 to its own ideal
  (1.1 mm) (#22).
- **Relative-error metrics need denominators that cannot vanish.** Position
  error ÷ distance blows up on an in-place spin; heading error ÷ net rotation
  blows up on an S-curve (a 0.17° error read as "142 %"). Normalise by path
  travelled (distance walked, rotation swept) or assert absolute error.
- **A module-scoped fixture holding mutated state makes tests
  order-dependent, and the file passes anyway** (a solver mode set in one
  test leaked into a later test's pick). Function scope is the honest price;
  run a new test in isolation before believing the file.
- **A repro that fails for its own reason cannot test a hypothesis.** An A/B
  is meaningful only once the harness passes on one arm.
- **A contact census that does not exclude the floor measures gravity** -- a
  cube resting on the ground is "in contact" on 86 % of steps.
- **A lifecycle's world is stepped through its routines, never by a raw
  `mj_step` between them**: the body's estimate is not told. 500 steps with a
  spin's command still in `ctrl` put 49° between belief and truth and painted
  a ghost wall that looked exactly like a seam bug (#277); a stand-up that
  stepped on its own froze the other robot ("The first quadruped deploy").
- **Adding a dynamic body is a telemetry change** even with no protocol bump:
  it moves the scene census and stales the committed fixtures.
- **A robot's compiled pose is clear of the geometry**: a bare `MjData`
  starts every free body there, and a plan from the origin once wedged the
  rover in a wall on every plain load.

## Conventions & gotchas

- MJCF `size` values are **half**-extents; `pos` is relative to the parent.
- Cameras look down their own **−z**, image-up is +y. Forward camera on a
  +x-facing body: `xyaxes="0 -1 0 0 0 1"`.
- Pitch from the freejoint quaternion `(w,x,y,z)`: `asin(2·(w·y − z·x))`,
  **positive = nose down**.
- A body with no joint is welded to its parent; `contype="0" conaffinity="0"`
  makes a geom visual-only.
- **`mj_geomDistance` does not measure box–box separation** — the box
  collider reports penetration, not distance, so pairs millimetres apart read
  `+0.0`. Sweep real contacts (set the pose, `mj_forward`, count `data.ncon`).
- **Adjacent-link interpenetration is silent.** Contact filtering treats a
  parent and its child as one, so a part can sweep straight through its own
  mount and the pipeline reports nothing -- while a grandchild is not
  filtered, and fights (the rover's pen quill fought the plate it was
  modelled inside, jamming its carriage while its joint reported perfect
  command-following). Clearance is asserted from *geometry*, an AABB sweep of
  the moving part, which caught the rover's parked fork 9 mm into its
  chassis. Endpoint checks are not envelope checks; contact checks are not
  clearance checks.
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

- **The nearest-frontier deadlock.** A forward sensor cannot see beside the
  body, so the nearest frontier is always the sliver just outside its view:
  the robot "arrives" at once and the frontier never dissolves.
  `MIN_FRONTIER_CELLS` (0.3 m) ignores those, and a look-around runs when no
  distant frontier is reachable (`behavior/navigation.py`). Driving is what
  buys the view.
- **Collisions corrupt the map, not just the paint.** Grinding a wall slips
  the drive, the estimate counts phantom distance, the map frame slides and
  old walls repaint at new believed positions ("jail bars"). The inflation
  and the front stop track the OUTERMOST geometry, not the chassis.
- **The reflex is armed in every manoeuvre, not just while driving** -- a
  look-around spin ran blind 7 mm outside a box -- **and never re-armed while
  it is firing**, which turned a bounded `BACKOFF_TIME` reverse into an
  open-ended one (`Navigator`'s scan step).
- **Exploration ends on no *reachable* frontiers**: frontiers off the
  robot's own connected floor are dropped before any plan
  (`behavior.navigation.plan`, `own_component`), and it ends after a
  look-around plus repeated pathless replans (`STRIKES_TO_FINISH`).

## The swap stack's costliest bugs: a frame, a sign and a retry (milestone 8)

- **An identity transform is not a validated transform.** A verdict that
  compared WORLD coordinates against RACK-LOCAL constants "worked" for weeks
  with the rack at the origin, and in the first room world reported every
  correct placement as a failure. A verdict transforms through the judged
  body's LIVE pose; a check that only ever ran at the origin has never been
  tested.
- **Symmetric-looking compensations are where sign errors hide.** The return
  added a 7 mm offset it had to subtract and overdrove 14 mm past a 16 mm
  mouth. These two were the costliest bugs in the repo and invisible to every
  cheaper test, which is why a whole mission runs on any geometry change
  (CLAUDE.md, "Tests").
- **Fixed choreography does not survive navigation.** Travel is computed
  from a fresh measurement every time, and a stand the measurement puts
  outside the capture backs out and takes another run, as the humanoid's
  plug-in fetches a miss again a step aside (`humanoid/charge.py`). **A
  retried measurement is a new draw from the error distribution; a retried
  constant is the same error again.** And a manoeuvre VERIFIES its outcome
  on a physical criterion before it calls it done, never on its command:
  the rover's swap seated and conducting or hung, the humanoid's plug-in
  off its pack's charge current.
- **A tag buys identity, and the last centimetres are the tag's.** Every
  reading is keyed to a decoded id, so a servo steers on the marker it means
  (a colour-plate stand-in once locked onto the charge bay's marker and
  dragged a module 22 cm toward the wrong bay); a stale prior loses to an
  observation; and odometry cannot close a terminal approach, so it is
  ranged off a tag (`perception/tags.py`; the charging station's and the
  places' tags).
- **Stop on the sensor, not on believed distance.** A press slips its drive
  and the estimate counts the slip as travel (a charge press ran to timeout
  15 s after connecting; a drive stalled against a fence believed itself
  4.28 m further on, #94), and a stagnation check that watches the believed
  pose calls that progress. What senses a press is contact, never the command
  or the motor's current: the belief is consistent with the command, and the
  ground disagrees.
- **A plausibility guard can reject the truth.** A guard that refused a tag
  range far from what a hand-off implies halved the damage and kept the
  failure: it was rejecting the tag, which was right, in favour of odometry,
  which was wrong. When a guard fixes the symptom and not the outcome, the
  model of the fault is wrong -- and an assertion the design does not promise
  is the same guard in a test.
- **A result shown for zero sim time was never shown.** Python between two
  physics steps costs no sim time, so a count written and overwritten before
  the next frame appeared in none of 10 850 frames while the result dict was
  perfect. A state a viewer is meant to see must outlive a frame interval,
  and only the recording proves it did.

## Sensor-realism pass: a sensor that never fails

The mapper was once fed MuJoCo's ground-truth depth through one camera's
row. OpenCV's SGBM on the rover's actually rendered stereo pair, in the best
case stereo gets, found disparity on 49.7 % of the scan row at 593 mm median
error against a 50 mm cell: flat painted walls are the classic no-disparity
case. So the robots carry a 2D LIDAR (`perception/lidar.py`) and cameras,
and stereo went (Parts.md). **A sensor that never fails cannot teach you
which behaviours depend on it**: this gap survived seven milestones because
ground-truth depth always works. Every sensor that feeds an estimate is now
the part's ("The map stays true under drift").

## Vectorizing the occupancy-grid scan update (issue #2)

The per-scan ray loop was the most expensive Python in the mission loop; one
(ray × sample) numpy batch with a single weighted `bincount` took it from 9.4
to 1.3 ms a scan, 7.4× (`tests/test_grid_vectorization.py` keeps the loop as
its reference on twelve recorded scans and requires ≥ 5×; time the two sides
INTERLEAVED -- CLAUDE.md).
- **The old loop's semantics hid in a numpy footnote**: fancy-index `+=`
  counts a duplicated index ONCE, so a cell sampled twice by one ray got one
  vote while separate rays accumulated normally. A naive vectorization
  doubles most free evidence (47.8 % of cells wrong, up to 3.6 log-odds). The
  batch dedupes *consecutive* samples, legitimate because a straight ray
  never re-enters a cell.
- **"Identical" has a stated tolerance**: the same cells get the same votes,
  but `k·L_FREE` once is not bit-equal to `L_FREE` k times -- atol 1e-9 on
  values, exact equality on the thresholded image.

## Pure pursuit orbits a nearby target (issue #7)

`drive_toward` cannot converge on a destination closer than about its own
overshoot: the target ends up beside the robot, the turn saturates, and
`v = V_MAX·cos(~85°)` keeps just enough speed to fly a stable ~25 mm circle
round it -- ~900° of turning per 200 mm hop, a limit cycle, not a wobble. The
terminal mode (`slow_radius=`, which the navigator's arrivals use) needs both
of its properties: a hard `TERMINAL_CONE` outside which `v` is exactly zero,
and a linear distance taper inside it. `tests/test_navigation.py` integrates
a kinematic unicycle, because the cycle is a property of the control law, not
of MuJoCo, and pins the defect in the default law so the fix's premise cannot
rot.

## The activity layer (issue #8)

A pressure plate that latches a garden light (`activity/plate.py`; the
`geom_pos` trap under "Conventions" is the surviving lesson of the gate it
replaced, table in ActivityPattern.md §3.4). Three measured rules the pattern
doc carries:
- **Hysteresis**: one wheel crossing flips a bare threshold **4** times, a
  hysteretic one (`PLATE_ON` 6 mm / `PLATE_OFF` 3 mm) **2** — one press, one
  release, the floor.
- **An analogue flag defeats sparseness**: `depressMm` over a 300-frame
  crossing is 38 deltas at 0.1 mm rounding, 15 at 1 mm. Quantise to what a
  consumer can act on or keep it off the wire.
- **Delta memory belongs to the SINK, not the activity.** A publisher and a
  recorder over one physics each run their own `FrameBuilder`; a memory on
  the Activity would have had them consume each other's deltas
  (`test_activity.py::test_two_sinks_over_one_activity_set_stay_independent`,
  shown failing).

## Drift hygiene (issue #42): the charger is the anchor

Every long-run failure family of the rover traced to uncorrected dead
reckoning, and the system survived COHERENT drift (belief and map drifting
together) and died of DECOHERENCE. The dock is the one pose a robot occupies
to millimetres by construction, held for minutes every cycle, so it
re-anchors the estimate there -- ⚠ to the COMMISSIONED pose, never the
belief. Measured: anchored to the believed rack, the anchor tracked the
belief's own error, 0.003 → 0.146 → 0.344 m over four sim-hours, a loop in
which nothing references the world. **What is true now:** the map frame is
DEFINED by the charging fixture, as on hardware: `Body.anchor_at_dock`
fixes the belief off its commissioned pose (`rack_prior`), on the humanoid
the station's two tags as it plugs in (`HumanoidBody._fix_at`; "Lost on its
own map, and found by its fixtures"). Between charges every level scan is
matched against the map ("The map stays true under drift").

## The bay tag's yaw was a coin flip (issue #88): fit the layout, not the tag

A square planar tag viewed near head-on has two PnP solutions mirrored about
its normal, nearly equal in reprojection error and chosen by pixel noise: a
bay tag's decoded yaw read about ±6–7° or near zero, never between, while its
translation held under a millimetre -- 5.5 cm of lateral error at the
rover's 0.45 m reach. A median of looks cannot help (the render is
deterministic, so N looks at one pose are one look), nor a sign from the
belief (the magnitude is wrong too). **What is true now:** a facing comes off
several tags' TRANSLATIONS fitted to their known layout by a 2D Kabsch fit,
the baseline between them rather than one tag's foreshortening: the
charging station's two tags (`humanoid.charge.fit`) and a row of signs
(`mapping/places.py`; "Places, not coordinates"). ⚠ The layout is
FACES, consistently: one plate centre mixed in put a steady +0.46° bias on a
two-tag fit (half a plate's thickness over the baseline; a differential error
is a rotation, a common one is swallowed by the translation).

## The squaring-up loop had no floor (issues #108, #339): every terminal loop has a budget

The pen's square-up at a board ended in `while |heading error| > tol: turn`,
with no timeout; ridden up onto the board's mount, wheels half off the
floor, the rover never got inside tolerance, drained its pack to 0 % and ran
700 s past the day's budget. Nothing else caught it because every guard sits
BETWEEN errands, and an empty pack does not stop the body. Later a drive
back to a bay standoff looped `while` it was more than 5 cm away: toppled,
the robot could never arrive, the loop outlived the death, and every
stand-up drove it back into a wall until it was flat (found by walking the
day routine's generator chain, `gi_yieldfrom`). **What is true now:** every
terminal loop has a sim-time budget and an explicit answer
(`control.FACE_BUDGET_S`); a bound is not a recovery (that is `reset_robot`
and the `stuck` death); and a premise test that could hang raises after
twice the budget, because a regression that hangs the suite is worse than
one that fails it.

## The world was not the same world twice (issue #110): MSAA is a random number generator

Evaluation.md rests on "nothing in the world is random", and five identical
scripted days gave three trajectories, parting inside a DRIVE.
`scripts/determinism_spike.py` flies a day N times in separate processes,
hashes the state and every camera image, decode and scan, and reports what
moved first: the camera images, on 2563 of 2613 looks with a tag in view, and
the lidar on 0 of 10 784 scans. One static scene rendered ten times gave ten
images, ±1 in 7–37 pixels at shadow edges: **that GPU's multisample resolve
of shadowed edges is not deterministic** (a GTX 1660 SUPER), and 0.6 % of
looks carried it through a decode into the robot's beliefs. Mesa's Intel
driver (Meteor Lake) and llvmpipe resolve it identically every time — one
image of 64 each, MSAA on (#440) — so the deployed box (osmesa, llvmpipe)
never had the bug, and the backend does not say which kind a box has: EGL
is the GPU, whichever GPU that is. Re-measured on the GTX for #440, most
renders are one image now and a second comes within a median 4 (at most 33
in 400 runs), so 16 renders showed no variation about one run in 22.
**What is true now:** every camera renders with `offsamples="0"` (the
house, `humanoid/model.py`), byte-identical at no cost to the
detector; `tests/test_render_determinism.py` pins the fix, and its premise
per rasteriser (`MSAA_VARIES`, read off `GL_RENDERER`).

## A second robot perturbs the first at the last bit (issue #167)

With a second robot attached and parked, the first robot's short runs hashed
byte-identical to the same runs alone, and a full day did not: the state
parted at t = 98.2 s by 1.4 × 10⁻¹⁴ on a free joint, with `ctrl` identical
and no perception input differing for 70 s after. That is the constraint
solver's rounding with an extra island in the problem, not a code path, and
chaos does the rest; `mjENBL_SLEEP` changes the first robot's numbers on its
own. **What is true now:** the parity instrument
(`scripts/determinism_spike.py`) proves a CODE change exact by flying the
same world before and after; a WORLD change -- another body, even one that
touches nothing -- is a different day at the 10⁻¹⁵ level, and its parity
claim is "same code path, same decisions", read off `ctrl` and the
perception trace, never the state hash.

## The other robot's wake walls you in (issue #167)

Painted into the occupancy grid and inflated, a robot driving past left a
wake of discs that covered the cell the other robot stood on: "no route" from
its own cell, with its module a metre away. Dropping the other robot from the
scan outright was half a sensor too far: the same rays fed the front stop,
which for seven deployed days could not see the one thing in the world that
moves (382 encounters, nine `stuck` deaths).

**What is true now:** another robot is never in the map, always in the
SENSORS, and in the planner's mask as its reported pose -- or, lying on the
floor, as its body ("A robot lying down was avoided where it said it was").
`Lidar.scan_split` (#316) and `DepthFrame.peers` (#328) sort the same casts
by what they hit, and neither hands the map a body that will have moved off
by the time anything reads it. ⚠ **Which half does what**: the BROADCAST is
plan time only (the planner's `OTHER_ROBOT_CELLS` disc; a stagnated drive
waits `OTHER_WAIT_S` when the other says it is near), a fleet fact that
drifts with its reporter; the SENSORS decide in real time and need the other
robot to say nothing, so a wrong broadcast costs a detour, not a collision.
⚠ The depth camera's hold is measured against WHAT IS LEFT OF THE DRIVE, not
the camera's reach: holding for a body the robot would never travel into
cost the rover its dock, and a charge that does not happen is a `flat`
death. ⚠ The mask is an obstacle of its own: a goal inside another robot's
disc cannot be reached at all (`Navigator.peer_on_the_goal` is the
arithmetic), so a taken charge seat is waited for ("Two robots at one
rack").

## Near-field 3D: the sensor before the map (issue #34)

The LIDAR looks along one plane, so "find a thing on the floor" had no
sensor; the part is a RealSense D435 (Parts.md). What the sim taught, each
with what is true now:
- **A depth image is ray casts here, not a render.** `mj_multiRay` at ~0.6 µs
  a ray gives a 120 × 70 frame in ~5 ms with a geom id per pixel (the
  self-filter for free), byte-deterministic and off the GPU, so the MSAA
  lesson never arises. ⚠ Its `cutoff` is a MAX DISTANCE and `0` tests
  nothing: the first timing table read "100 % hit" off arrays the call had
  never written. `perception/depth.py` passes `MAX_Z`.
- **A sensor's origin sits outside its own housing.** On the housing's face
  exactly, the model settles a hair of pitch and every ray hits the housing
  (the D435's lens stands 1.5 mm proud); a LIDAR site at the centre of its
  puck repeated it ("The map stays true under drift").
- **The stereo shadow is angular, so at sim resolution it is often
  sub-pixel.** The band a near edge hides from the right imager is
  `f·B·(1/z_near − 1/z_far)` pixels: a 30 cm post at 0.7 m against the floor
  at its own depth is 0.6 px and rightly casts nothing. A test that looked in
  the wrong row "found" no shadow.
- **Standing still, the floor is sampled in rows ~3 cm apart at 1 m**, so a
  5 cm cube arrives as two stripes and a 2 cm one is never seen. The height
  map bridges ONE unmeasured cell in `things()` and otherwise integrates over
  motion; nothing invents samples.
- **A voxel map was measured, not assumed away.** Same frame, same window:
  the 2.5D map is 40 000 cells and 0.7 ms an update; 2 cm voxels are 250 000
  cells and, with the free-space carving that lets a map forget a moved
  object, 111 ms as written. The occupied-only voxel update is 0.2 ms -- and
  cannot forget, the cost hiding in every "voxels are cheap" claim. The height
  map is the representation; voxels are the answer if what is under an
  overhang ever matters.
- **The loop builds the map before anything reads it.** The sensor ticks the
  physics seam at 10 Hz where it is on (`near_field=`), placed like the grid
  and gated like it (`Body.level()`), and the map streams as `heightmap`
  beside `grid`. ⚠ The lifecycle's own settling ticks the seam before a test
  gets to it: a direct `_near_field_step()` call inside the same period takes
  no frame, which read as "the throttle is broken" until the count was taken
  as a delta. Nothing that decides reads the height map; the planner reads
  the D435 through a layer of its own ("The first quadruped deploy").

## The second house and the loop (issue #215): the planner is the new cost

The home world grew from 26.5 × 12 m to 49 × 21 m -- a second house across
the street, a sidewalk band and a street loop round both, a fence round the
loop -- and the grid from 159,600 to 469,200 cells. The frontier mask stayed
linear (4.6 → 12.6 ms a pass); pure-Python A* did not (0.34 → 1.26 s a plan
across the loop), the first cost here that scales with the WORLD rather than
with what the robot does. **What is true now:** both bodies' walks plan on
a compiled lattice ("Walking into the unknown"), and the cap was measured
again at up to four times the cells ("The map's size limit", #558).

The reserve's worst point is found BY ROUTE, not by straight line: the loop's
east legs are the farthest from home as the crow flies and among the nearest
as the robot goes, because the house is reached only through the middle
street's gate. `tests/test_world_budget.py` routes every zone over the
compiled world and asks which is farthest (a straight line would have sized
the reserve off the wrong corner by twenty metres); the reserve is
`humanoid.world.RESERVE_WH`. The lab's props arrived as geometry with no
behaviour behind them, by decision: a world change is one regime break, and
the issues that add behaviour find the world already holding what they need.

**A bigger world moved the cameras' near plane.** MuJoCo derives
`statistic.extent` from the geometry's bounding box and scales every
camera's near plane by it (`visual.map.znear` × extent). The loop took the
extent from 37.2 to 70 m and the near plane from 0.37 to 0.70 m, and the
rover's dock camera, 0.34 m from the rack at a bay, clipped the whole rack
out of its image: nothing raised, every pick ran on belief, and a blind stow
put the pen on the floor (the lights were suspected first and measured
innocent). **What is true now:** the generator writes `<statistic extent>`
down (`home.CAMERA_EXTENT_M`;
`test_home_world.py::test_the_camera_extent_is_written_down_not_derived`),
and the humanoid's world sets its own near plane (`humanoid.world.NEAR_M`,
0.10 m: at the house's 0.37 m it cut a hand near the head). Any world whose
bounding box grows must pin its extent, or its cameras lose whatever they
look at from closer than a hundredth of it. ⚠ **A pin in the XML does not
survive a runtime `mj_setConst`**, which re-derives the extent: a bench
offer undid it for the rest of the process, and the deployed pair's picks
failed 13 of 15 with nothing raised. Anything that calls `mj_setConst` on a
live model puts the statistic back and hands it a scratch MjData (it writes
`qpos0` into whatever data it is given), as the storeroom's set-out does
(`home/storeroom.py`).

## A goal out of sight is aimed at through the nearest wall (issues #215, #298)

The planner answered an UNMAPPED goal by aiming at the known-free cell
nearest it in a straight line. From the loop's north-west corner that cell
was indoors, behind the first house's north wall, and the only way there was
back round the loop: a plan that began by driving AWAY from the goal, which
the drive's stagnation check rightly ended. Near a doorway (#298) the nearest
cell was a one-cell island at the ragged edge of the wedge of free cells a
LIDAR paints through a door, joined to nothing: every replan found it, and a
board in the bedroom went unreached for thirty hours of the deployed pair.
**What is true now:** the humanoid plans through the unknown at a price
("Walking into the unknown"), and needs a stand-in only for a goal INSIDE a
wall's inflation. The mapped-floor planner the
rover drove by -- a stand-in on the robot's OWN component, `scipy.ndimage.label`
over the traversable mask, 4-connected like `astar` -- is still
`Navigator`'s default, pinned as the premise of
`test_unknown.py::test_a_goal_past_the_map_is_walked_to_by_the_door_it_has_seen`.

## The tag camera spent 97 % of its render on shadows (rooftop-media-2026 #296)

The served pair ran at 0.23× real time, and the site played 50 ms of motion
and froze ~170 ms, over and over: Mesa's software rasteriser behind
`MUJOCO_GL=osmesa` held ~80 % of all CPU, with the physics thread waiting on
renders. A lone renderer with the scene flags toggled gave the number: **one
1280×720 home frame costs 1113 ms with shadows and 32 ms without** --
sixteen `castshadow` lights at MuJoCo's default 4096 `shadowsize`, sixteen
16-megapixel depth passes a look. A wrong turn worth keeping: timing inside
the container found "fast" and "slow" GL contexts, and the fast ones were
contexts whose shadow framebuffer had failed to create; MuJoCo then renders
without shadows and says nothing. **What is true now:** `TagDetector` clears
`mjRND_SHADOW` and `mjRND_REFLECTION` on its own scene, and `update_scene`
keeps the flags (`tests/test_render_context.py`). A tag decode thresholds
gray levels and a real camera has no shadow pass, so nothing hardware has is
lost; the viewer and the filmstrips keep their shadows. The site's clock
follows the stream's measured pace, so a world that cannot hold real time
plays smoothly slow rather than stop-and-go.

## The physics thread was four-fifths bookkeeping (rooftop-media-2026 #296, the profile)

With the shadows off the served pair still ran at 0.5×, so the thread was
profiled (`py-spy record` at 100 Hz inside the deployed image). Four of the
top seven frames were Python loops over `data.contact[i]` -- a pybind struct
per contact, ~330 000 constructions a sim-second for two robots -- run every
step: 49 % of the thread against 13 % in `mj_step`. And the map's inflation
was `binary_dilation(iterations=7)` over the whole grid on every replan,
~300 ms a call. **What is true now:** contact is read as an ARRAY
(`contacts.contact_pairs`); `tests/test_contact_reads.py` shows the readers
give the loops' answers, and a new per-step check goes through them. The
inflation is one chamfer distance transform (`distance_transform_cdt`),
which `tests/test_frontier.py` pins identical to the iterated dilation.

## A robot on its side maps the sky (issue #339)

A toppled robot's map had its hall painted solid with floor-hit arcs and a
free fan running out through the living room's walls past the fence. The
LIDAR casts along its real frame, and no return is "free to max range", so on
its side half its rays see the sky and go into the map as 8 m of free space
at the believed, upright bearings (5 s on its side painted 38 622 free cells
onto an empty map, through the walls). A dead robot keeps scanning, and a
stand-up kept the map. **What is true now:** a reading goes into the map
only while `Body.level()` -- on the humanoid, its IMU's tilt within
`HumanoidMission.LEVEL_TILT` (2 deg) -- and the near-field height map takes the
same gate; the front stop still reads every scan. A stand-up drops the map
since #572, but a fall the robot gets up from keeps it, so the gate stays.
⚠ The corrupt map was not the death loop it was found beside: that loop
reproduced with the gate in place ("The squaring-up loop had no floor").
Trust the reproduction over the story that fits.

## Two robots at one rack (issue #346)

On the rover pair the rack was the biggest single cause of failed jobs and
of every `flat` death: a robot said it was clearing the rack and never
checked that it had left, a bay somebody stood on was given up at once, and
nothing moved a robot off the rack between a swap or a charge and its next
decision. Two explanations were measured, not guessed: a "no-tag" was a look
from BESIDE the standoff, where the other robot's planner disc had left the
robot, out of the tag's decodable cone; and a 60 s "never got there" was one
hold for the other robot at the neighbouring bay and then the stagnation
wait until the errand's time ran out.

**What is true now:** the rack the loop works round is the humanoid's
charging station (`body.rack_prior`, the station's frame). After an
interaction there, a robot with somebody else in the world leaves it before
it decides (`_leave_rack_routine`) and checks that it got there
(`RACK_CLEAR_M`, `CLEAR_SPOTS`; History says where it ended); one that stays
within its reach for `RACK_LINGER_S` doing nothing is logged `RACK:
lingering`. A taken charge seat is waited for
(`HubLifecycle._await_bay_routine`) for 3× the typical charge
(`CHARGE_OCCUPANCY_S`, measured on the quadruped pair; the humanoids' are
not yet), and a failed charge approach logs `charge_trace`.

## A robot lying down was avoided where it said it was (issue #365)

On 2026-09-23 Rowan drove into Luca, who had lain on its side for three
minutes after toppling, and fell over too. The errand a robot falls in keeps
commanding its body, and the rover's reckoner counted wheels turning in the
air as travel, so its reported pose left its body (1.5–2.2 m in 10 s on its
side) and a planner disc on the report guarded the wrong place; a robot
lying down also reaches further than one standing. Flown past a toppled
robot whose reckoner was 1.8 m off its body, a disc on the report touched it
in 4 of 8 passes; a disc on the body, not held for, touched it in none and
arrived all 8 times.

**What is true now:** a robot lying down (past `TOPPLE_TILT_RAD`, from the
moment it falls, dead or not) is avoided where its body lies:
`HubLifecycle.keep_clear` answers the middle of its footprint with
`DOWN_ROBOT_CELLS`, placed where the DRIVER's own sensors would put it
(`Body.as_seen`), so the driver's own drift cancels. The depth camera does
not hold for it and a stagnated drive does not wait on it
(`Navigator._other_in_the_way`): a hold waits for the other robot to move,
and a robot on the floor will not move until it is stood up; the lidar's
front stop still sees it. A drive looks at who is lying down
every `DOWN_CHECK_S` and replans at once when that changes (without it, a
robot knocked flat just after a replan was met only by the front stop). Once
stood up it is avoided where it says it is again. A line about a robot on the
floor says it is "lying knocked over" (`posture`), never "standing";
`tests/test_downed_peer.py` pins each rule.

## The stairs curriculum (issue #388)

The quadruped's curriculum, which the humanoid's walkers train on
(`training/humanoid_train/stairs.py`, `walk.py`). What it found, each still
binding:

- **mjlab's promotion starves a climber.** A robot is promoted a level for
  ending a 20 s episode 4 m from its tile's centre, and its commands turn
  and sidestep every 3-8 s, so a capable robot seldom got there: across
  6000 of the quadruped's iterations even the FLAT column's level sat at 4.3
  of 10, where the level changes nothing, and the climbing one at 1.7
  (risers of ~0.08 m). On a stair tile most commands now walk straight out
  along the nearest of the pyramid's four flights (`StairsCommand`). ⚠ A
  reset resamples the command BEFORE mjlab's `sim.forward()`, so the pose
  read there is the last episode's: the command is aimed on the next
  update.
- ⚠ **mjlab's columns are named backwards.** `pyramid_stairs_inv` spawns
  the robot in a pit, so walking out CLIMBS; `pyramid_stairs` spawns it on
  top.
- **A foot swung HIGH is the lift a riser needs.** mjlab's foot clearance
  charged it. `foot_nose_clearance` is one-sided: a moving foot's centre
  over the highest terrain within half a tread (a ray starting inside a
  step reads zero), so a foot closing on a riser answers to the tread above
  it, and one stepping down to the tread it leaves until it is half a tread
  past the nose.
- **The mean action hesitates at an edge.** Mid-run, a checkpoint walked to
  the first riser and STOOD there, while the same weights with training's
  exploration noise climbed: in training the noise breaks the stall, so the
  curriculum's level is the noisy policy's, and the robot runs the mean
  action. Four hundred iterations later it climbed.
- **A policy is shown the scan the robot has.** The quadruped's, trained on
  the ideal scan, climbed as well or better on the D435's map -- its
  occlusion, its fill (an unseen point reads the nearest cell seen) and its
  noise -- but only because the map's fill made a drop look gentler: down
  0/10 on the ideal scan, 10/10 on the map's, a difference it never trained
  on. What a walker that sees the ground is shown in flight is the map the
  robot lays ("One walker that sees the ground").

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

**The drivers in C.** A joint driver's PD and torque-speed envelope ran in
numpy on every physics step. MuJoCo's `dcmotor` in position mode runs them
in C, but it models a VOLTAGE: its controller makes
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
trained on. The quadruped's policies walked on it within tolerance --
tracking, torques, power, posture and odometry as before -- and a flight
hashed IDENTICAL in two processes.

**A served day, profiled.** On the rover pair, the world served then
(`serve.py --pair --free-run`, a scripted day with the jobs, hunger and the
near field on, under `py-spy` at 100 Hz), the physics thread was about a
third `mj_step`, a quarter the depth camera (8 400 rays a frame at 10 Hz
per robot) and its height map, a tenth the tag camera's renders and
decodes, and the rest a few per cent each. **Not cheap, and deliberate:**
the physics; the depth camera, whose rays and rate are the sensor as
specified (Parts.md) -- halving either would halve its share, and that is a
decision about the sensor, not a speed-up; and the tag camera's renders,
whose resolution is what the decode needs.

**The server.** #377 had asked for at least 1.3× real time — a margin,
never argued. A candidate core was rented by the hour (Runpod's `cpu5c`, 4
vCPUs): an **AMD EPYC 4564P** — the Ryzen 9 7950X's silicon on AM5, Zen 4
at up to 5.88 GHz — on a host shared with other tenants (a pessimistic
reading). Over the span a scripted rover day shared on both (its Mesa 25.1,
the image's 25.0, and a render that differs moves a decode) it ran **1.64×
the box**; physics alone 1.5×. The core is the answer, not the count: the
physics is one thread.

**Decided (Ben, 2026-09-27): the served world stays on the current box**,
at about a fifth of the candidates' price. Below 1× nothing breaks: the
world runs slower than the wall, the site plays the stream at its measured
pace (rooftop #298), and everything the robots are scored on runs on sim
time, as the rover pair's weeks at 0.5–0.8× showed. A faster box is the
lever if the slow playback ever matters, and the EPYC above is what one
buys.

**What is true now:** the scan is one `mj_multiRay`, its noise drawn in
bearing order (`test_the_batched_scan_is_the_scan_it_replaced` pins every
ray against the per-ray scan); the joint drivers are position-mode
`dcmotor`s, commanded as an MIT-mode driver is (`humanoid/drivers.py`,
`actuator.JointLimits.dcmotor`): a policy's command carries its own gains,
an arm's the default gains scaled by `ARM_GAIN_SCALE`, and `limp()` holds
nothing, so the policies and the arms share one body. (A routine's torque, the quadruped's lie-down and
stand-up, went with it, #559.) A pair is one physics thread, so the lever left is
the core, and the box keeps its core for now; the humanoid pair's speed on
it is "The humanoid pair served"'s.

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
robot lost beyond that window has its fixtures (#476), and #381's loop
closure.

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

**The drift baseline, before and after.** The rover flew a home day of
three lab round trips -- the mouse's feed in the lab, then a carry at the
rack -- on each tree:

| rover, three lab trips | worst | median | worst heading | the lab |
|---|---|---|---|---|
| perfect sensors, no matcher (056a4a1, the sim before) | 0.60 m | 1.7 cm | 0.34° | reached each time; the first trip's plate pump left the belief 0.6 m out through all three, until the dock |
| honest sensors, no matcher (the before) | 5.41 m | 0.28 m | 70.7° | never reached: `no route over the floor mapped so far` 0.2 m short of the lobby's door, then lost |
| honest sensors, matched (the after) | 0.34 m | 1.6 cm | 1.08° | reached each time, 0.2-0.34 m out in the lab and back to 7 mm, 7 mm and 6 cm at the rack; the search fired twice |

The quadruped, walking the same route three times out and back (145 m) on
its flat policy steered by the TRUE pose -- what is measured is the
estimate, so its steering must not depend on it -- was 3.14 m out at worst
on legged odometry alone and 0.20 m matched, and every trip came home to
about a centimetre: matched, neither estimate accumulates.

**The lab door**, asked of the robot's own planner: can it plan from where
it believed the lobby was to where it believed the lab was? Open for the
matched rover after each of its trips, and for the sim before; the honest
rover without the matcher never reached the lab to have a door to ask about.
The quadruped's odometry map plans "open" as well -- because its walls are
gone: three rotated copies of the lab, and the lobby's east wall, the one
with the door in it, rubbed out by rays laid through later poses. A map
smeared by drift loses walls as well as doubling them, so a map's picture is
read beside its trace. For scale: in a map laid through the TRUE poses the
door leaves 0.30 m a plan can use at the wall (1 m, less 0.35 m of inflation
each side), so the issue's estimate holds -- ~0.3 m of smear closes it.

**Found on the way.** A body attached into a world through MJCF's include
form carries no `<compiler>`, and MuJoCo's default angle unit is the
degree: the quadruped's knee range read as degrees and its joint limits
threw it 0.4 m into the air. A body attaches in radians
(`humanoid.model.attachable()`). And a LIDAR site at the centre of its own
puck had every ray hit the housing at 4 cm and the self-filter drop all
360: the site sits 6 mm above the puck (no mass moved).

**What is true now.** Every level scan is matched before it is fused and the
matched pose is the belief (`Navigator._match`); the fixture's anchor stays
beside it ("Drift hygiene"). A match costs 0.67 ms on the dev machine
(median 0.46, p99 7.2 -- the field's refresh, 5 ms every 20 fused scans or 1
m, and the rare search): 6.7 ms of a sim second per robot at 10 Hz. Fusing
only the scans after the robot moved took the grid's updates on a recorded
day of lab trips from every level scan to 6 918 of 10 947, and the mapping
as a whole from ~21 to ~20 ms a sim second; a day that stands still less
saves less. The matcher is deterministic (einsum's own loops, no BLAS
product) and its field is kept state: a scripted rover day hashed IDENTICAL
in two processes (2 025 state samples, `determinism_spike.py`), and saved at
t = 419 s and carried on in a new process it was IDENTICAL after the restore
(1 186 samples, `--resume-at 400`). Not done: a scan from a body that is not
`level()` is neither matched nor fused, so drift while it is not level runs
free until it is and the search finds it; a robot lost beyond the search's
window searches wider after a run of refusals (#422, "Lost after a long
explore, and found again"); and the first trip through new territory carries
its own drift into the map it lays (0.2-0.3 m by the lab, ~1 m round the
street loop), which later visits match to but do not correct (#381's loop
closure).

## The first quadruped deploy (issue #387)

The quadruped's deploy moved the map, the planner, the drive and the peer
rules out of the rover's mission into `navigator.py` whole (a scripted
rover day of 1012 sim s hashed IDENTICAL across the move), and the house
taught them the rules below. The humanoid keeps each (`HumanoidMission`:
the quadruped's rules with its own numbers).

**A LIDAR's plane over the furniture.** The quadruped's LIDAR rode 0.51 m
up, over the couch and the bed; the humanoid's rides 1.41 m up, over every
table and counter. A planner fed the LIDAR alone routes through them, so
the D435's points between a foot's clearance and that plane are a layer the
planner plans round (`_fold_low`, `_planning_grid`). It is the first thing
on the robot that decides off the depth camera, and it is NOT the height
map: a layer of its own, log-odds like the grid. Three rules made it
usable, each after it shut a door:

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

- *The dead band made arcs.* A walking policy walks nothing under its dead
  band, so a small command is raised out of it -- and the driving law
  (`drive_toward`) crawls while it turns, so raised, the crawl swept arcs
  into the walls it was turning away from. A crawl with a turn to make is a
  pivot (`command_for`).
- *A detour read as a stall.* The rover's drive gave up after 10 s
  without getting closer to its goal in a straight line (`Navigator`'s
  default still does); from the bedroom's corner the only way to the hall
  walks 2.5 m round the divider first, and every such drive "stalled". A
  walker's progress is read along its route
  (`Navigator.PROGRESS_ALONG_ROUTE`).
- *The explore gave up on reachable floor.* The rover's explorer A*ed the
  20 nearest frontiers and blacklisted each failure; the depth layer's
  small obstacles seal pockets of frontier near the robot, and three such
  rounds ended the explore with the kitchen and the garden unseen.
  Frontiers off the robot's own connected floor are dropped before any A*,
  and none is blacklisted (`Body.plan_frontier`).
- *A stand-in replanned every step.* A goal just off the plannable floor
  is driven to its nearest reachable cell; standing on it, the waypoints
  run out and the drive replanned every physics step until its stagnation
  cut -- 4 190 plans in 90 s of a pair, 15 ms each. A plan asked again
  within 0.5 s from the same spot is the last plan
  (`Navigator.REPLAN_HOLD_S`; the humanoid's drive did it too, #558).

**A stand-up steps nothing.** `determinism_spike.py --pair` flies the
served pair's day with a fall and a death arranged
(`pair.arrange_hazards`) and hashes the whole world. Its first flight
hashed IDENTICAL twice and was wrong both times: the timer's stand-up
placed the second robot with `start_at`, which settled the body by stepping
the physics for a second on its own -- and for that second the first robot,
walking in the same loop, ran no policy and no odometry. It went over, got
up 0.44 m from where it believed it was, found no route to the dock and
died `stuck`. A body's `start_at` steps nothing now: the body settles under
the day's next command. And a stand-up never lands on another robot: a
start pose with a body within 1.0 m is passed over for the next
commissioned one (`HubLifecycle.up_pose`, `START_CLEAR_M`).

**Open: two robots meeting head-on hold for each other** (#395). Each holds
for the other (the rover's #328 rule: a robot is held for, never backed
away from) until a drive's patience runs out and one goes round. It ends by
itself; nothing yet decides who yields.

## Walking into the unknown (issue #381, the walking stage)

A fresh quadruped -- its start pose in the living room, an empty map, one
look round -- sent once to each of the property's 24 zones (a zone's centre
moved to the nearest floor 0.45 m from anything, each walk in its own
process, 300 s a walk) reached **9**, and every failure was the same one:
`no route` at a stand-in pressed against a wall. The planner planned over
mapped floor only and aimed an unmapped goal at the reachable cell nearest
it in a straight line ("A goal out of sight is aimed at through the nearest
wall"), which from inside a house is against the wall between the robot and
the goal: the workshop's wall for the kitchen, the living room's south wall
for the south garden, the garden fence 3 m from its gate for the street. The
lab and the store arrived only because every door on their way (the gate,
garden_2's, the lobby's, the lab's) sits on y = 3.1, the straight line -- in
103 and 118 s, past the 60 s a procedure's `drive_to` and a decided
`explore(zone)` gave a walk: with the budgets the robot had, 4 of 24.

**Unknown is floor at a price** (`mapping/optimistic.py`,
`Navigator.OPTIMISTIC`, on in `HumanoidMission`). A cell nobody has seen
costs `UNKNOWN_COST` a metre against a mapped cell's 1; a known wall grown
by the inflation is not floor on either side of it (the far side of a wall
is not a way through); the plan runs as straight as the map allows, the
LIDAR sees the walls it crossed as the robot walks, and the next plan (every
2 s) goes round them. A doorway is found by finding the walls either side of
it. No route at all is then a fact about the map: the walls the robot has
seen enclose the goal. The stand-in stays for a goal INSIDE a wall's
inflation, and a goal on the floor but walled off is no plan. The same 24
walks after: **24 of 24**, 15 of them within 7 % of the true route and 16
within 20 % (the "true route" is the same planner over the compiled world's
own floor, nothing left unknown; the living room's 0.3 m aside). The other
seven all end on the loop round the two houses, at 1.3-2.4 times the route,
and it is the right answer to the question the robot was asked: from the
middle of a house it has never seen, the cheapest way anywhere is through
the walls it has not seen yet. Sent to the west street it tried the
bedroom's north wall, then the kitchen's west wall -- the house has no door
on either side -- found both, and went out by the garden door and round the
north sidewalk. A first walk into the world is an exploration, and its map
is kept: sent back to its start and there again, the second walk to the west
street took 117 s and 45.2 m (the route is 43.6), to the south street 60 s
and 23.2 m (23.3); the near ones were near their routes both times. ⚠ One
walk back did not arrive: from the north sidewalk it shuttled east and west
for its whole 300 s (108.7 m), because the house's north wall is 4 cm thick,
thinner than a cell, and the robot walking along it carves it out of its own
map with grazing rays -- the map had it free from x = -3 to 6 m -- and the
planner routes through where it was (#401: a choice of sensor model, filed,
not made here).

**Why this price.** A known detour is taken while it is under twice the
unknown shortcut (`UNKNOWN_COST` 2). It is a choice, not a fit: swept to 1.25 and 4
over six fresh walks it moved four not at all and two by 3 s, because a
fresh map offers no known detour to weigh -- every way out is unknown. What
it decides is a second walk's: the way it knows against a shortcut it has
not seen.

**The map still growing is progress** (`PROGRESS_MAP_GROWTH`,
`MAP_GROWTH_CELLS`): a walk into the unknown turns back when it meets a
wall, and a route that lengthened because the map showed a wall is not a
stall. The straight-line progress rule and #387's route length both read
that as no progress. Walking into new floor
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
- the CSR graph is built once per grid shape (under 10 ms, #558; shared by
  a pair) with eight slots a node, and a plan writes its weights -- a step at
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
`explore(zone)` walks with `ZONE_PATIENCE_S` (300 s: all 24 fit). Its goal
is the zone's middle, which can be furniture, where the walks above went to
the nearest clear floor; a walk that ends in its zone has got there (#454).
A walk asks the robot's own hazard rows every second
(`go_to_routine(stop=)`) and ends where it stands on the answer --
`DRIVE_STOPPED`, never one of the four ways a drive gives up -- and a
procedure it stops is `stopped: interrupted`, as between two verbs.

**What is true now.** The humanoid walks into the unknown (`OPTIMISTIC`, as
the quadruped did; the straight-line stand-in is `Navigator`'s default), and
the rules are pinned in `tests/test_unknown.py`. Not done: the map is still
the property's rectangle (#381's stage 4), loop closure (stage 3), and what
a walk into the unknown costs is not yet in any offer (stage 7).

## Places, not coordinates (issue #419)

A job is handed the building's address, a few metres off, and the task
area's written directions -- never where things are. The robot finds the
place by a tag on it, remembers where it saw it in its own map, and walks
the last of the way steered by the tag (`mapping/places.py`; the humanoid's
search, `humanoid/places.py`, is "The humanoids' first jobs").
`tests/test_places.py` pins each rule.

**A handed coordinate carries the frame's error.** #414 handed the lab
plates' world positions to the quadruped's feed job: after an explore the
belief reached the lab 0.43-0.45 m off, and the walk home pressed the shock
plate five times. A place found by sight is laid in the frame the map and
the belief are, so the drift is in all three and cancels.

**A sign, not the floor.** A 120 mm tag at the camera's height, facing the
room, decoded square-on past 5 m, 55 deg off its face to 5 m and 70 deg off
to 1.5 m, its position to a median 1.2 mm; the same tag flat on a pad
decoded in 2 poses of 77: the floor is seen at a graze. So a place carries
its sign on a wall or a post.

**One tag's rotation is a coin flip square-on.** Its reported yaw was up to
11.6 deg wrong square-on, and 20 deg off it once came out mirrored; from 30
deg off it was within 0.6 deg at every range -- but one look 27 deg off at
4.1 m read 38. So a place's facing comes off the row of signs fitted to its
drawing when two are known (#88), else a look at least 35 deg off and
within 3 m, else where it was seen from.

**The search follows the map, out from the address.** The first search took
its viewpoints nearest the robot, and walked out of the facility, round it
and into the street. Now (`next_viewpoint`): where the row's drawing puts
the sign once any of the row is read; else the address; else, by the WALK
from the address over its own map, a FRONTIER -- known floor meeting the
unknown -- or floor it has mapped and not yet looked over with no wall
between; else a lattice point in the unknown. A fresh quadruped found the
lab from 3 m errors in all eight directions (62-272 s, median 94), and from
the facility's own address, which lands in its storeroom a wall from the
lab, in 123-157 s.

**The walking look costs a render**, 14-20 ms on the dev GPU under EGL and
32 ms on the box under osmesa: a look every metre walked or 45 deg turned,
never oftener than 0.5 s, and none standing -- 16 ms a sim second at the
box's render, 1.6 % of real time per walking robot.

**Still open.** Loop closure (#381): a place's frame is the one the map was
laid in. The address is a fixed offset, not a GPS sensor.

## Lost after a long explore, and found again (issue #422)

**Where the drift grows.** A quadruped explored 480 sim s from its
commissioned start and walked home to dock, its belief against the truth
every 10 s, over eight seeds of its sensors' noise: the explores ended
0.10-1.42 m out, and 4 of 8 docked. In the house the belief stays within
0.14 m. It grows on the street loop and the long sidewalks: walls on one
or two sides, straight for 20 m, met at grazing angles (#401), every one
laid by the robot on the same walk. There a scan can only be matched
against what the robot laid seconds before, through the same drifting
pose, so the matcher holds the belief to its own recent map rather than to
the world, and 20-70 % of its matches keep a direction (the street's
length) odometry's.

**Why it failed to dock: the walk home.** In every failure the robot never
reached the standoff. Crossing from its garden (laid drifted) into the
living room (mapped true at the start), its scans matched "ok" with
0.56-0.73 of their points on walls, and a match need only put half its
points on walls to be laid in. Two seconds of those painted an offset copy
of the room; the field's next refresh put the copy in it, and every scan
after matched the copy and was laid in, erasing the room as first mapped
(the log-odds are clamped at ±5, so a wall is floor after 14 misses).
Another came home to a garden its map held two ways at once, and no one
pose fitted both.

**The fix, in three pieces.**

- *Near enough, the fixture decides* (`lifecycle.NEAR_STANDOFF_M`, 0.75
  m). A walk to the charge standoff that gives up within it goes on to the
  approach, which finds the charging fixture by its tags whatever the
  belief; with none in sight from there, the walk's own retries go on as
  before. The standoff was only ever how the robot got to the
  neighbourhood (`go_charge_routine`). Replayed from the three saves that
  failed that way, all three docked on the first approach.
- *An anchor re-lays the map round it* (`scan_match.ANCHORED_SCANS`, 30).
  A fixture's fix puts the belief in the world's frame, and the next 30
  scans laid in go in at that belief, unmatched, and overwrite a copy laid
  askew: without it, a robot backed off the dock was 0.62 m off a second
  later, matched back into the copy; with it, 0.00-0.01 m. Only a fixture's
  tags anchor: a seat with no decode was good to 2 deg, and 30 scans laid
  unmatched through that would lay the room askew themselves. Nor is the
  commissioned start an anchor (a look-around laid unmatched smears the
  gyro's scale error, up to 1.8 deg a turn, into the first map).
- *A robot its map keeps refusing searches wider* (`scan_match._relocate`).
  After ten scans running refused (any other verdict breaks the run), the
  search looks 2 m and 15 deg round the belief, on a lattice of 20 cm and
  1 deg with the walls widened to cover it, then cell by cell round the
  three best PLACES (0.5 m apart at any heading: one spot at three
  headings would hide a second place). A pose is taken only if it fixes
  all three directions with three quarters of the scan on walls, no
  second place within reach explains the scan as well (nine tenths of its
  inliers: a pose 2 m off with most of the scan past the map's edge was
  "found" with 216 where the truth had 359), and the next wide search,
  ten refusals on, finds the same correction. 72 ms on a house scan (the
  search it widens is 32 ms, on every refused scan); one in ten of a lost
  robot's scans. It is the issue's own ask, for a robot lost with its map
  whole: a quadruped set down 1.5 m from its belief in its living room
  found itself on the twentieth scan.

**What was tried and dropped: a scan laid in only where the map agrees.**
Fused only with three quarters of its points on walls, the living room
was not laid over, and two more seeds docked. But on a sidewalk whose thin
wall the map had eroded (#401), it refused the scans that would have laid
the wall again, the share kept falling, and the wide search, then taking
one search's answer, jumped 1.8 and then 3.4 m along the sidewalk: two
robots that had docked were lost mid-explore. The map's confidence cannot
tell the cases apart either: an eroded wall reads -5, as sure as the
living room's floor. It went, and a relocation must now be found twice.

**What is true now.** A walk to charge that gives up within 0.75 m of the
standoff goes on to the fixture, and back to its retries if the fixture is
not in sight; after a fixture's fix the next 30 scans are laid at its
belief; and a robot its map refuses ten scans running searches 2 m and 15
deg round its belief, taking a pose that no other place explains as well
and that it finds twice. Over the quadruped's eight seeds the explores were
unchanged, to the bit, and seven docked where four had.

**Not done.** The drift itself: the map laid round the street loop is
still laid through a belief that drifts a metre, later visits match it and
do not correct it, and where parts laid through different drifts meet, no
pose fits. Correcting the map behind the robot is #381's stage 3, a pose
graph, and this measurement says it is needed. A fix off a fixture away
from the charge came with #476, the next section.

## Lost on its own map, and found by its fixtures (issue #476)

**What the live pair showed.** Of the quadruped pair's 44 deaths from
2026-10-03, 19 had the belief more than 1 m from the truth, and the error
came back on maps laid from nothing. It moved two ways, neither of them a
run of refusals: a *slide*, 1.2-2.5 m between two saves with almost every
scan matched `ok`, and a *jump*: four relocations of five came where the
error grew most, 1.6-2.0 m each.

**Why: the map holds copies, and the matcher holds the belief to them.**
Put at the truth on each robot's own map and walked to the lab and back,
the belief was 1.3 m and 1.9 m out within 20 s: the map refused the
truth's scans, and the wide search (#422) relocated onto a copy. On an
empty map the same walk was 0.23 m out at worst. So a belief put right is
lost again while the map stays. The first copy is laid where the matcher
cannot hold a walk true -- the long walks outside, #422's residue, are the
likeliest.

**The fix: the commissioned fixtures.** A fixture is where it was
installed, so a look at it puts the robot in the world's frame: the
quadruped's dock board and rack, and the humanoid's charging station by its
two tags (`HumanoidBody._fix_at`).

- *Where a fix is taken*: only where a look was MEASURED to be good enough
  -- at the charge, and on a lost robot's looks. Never on a walk-in: the
  quadruped's one look there was 26 cm and 6.6 deg out, past the line that
  drops a map.
- *What it does*: past `FIX_TOL_*` (10 cm or 2 deg) it moves the belief and
  lays the map again round it; past the matcher's own search (`ASKEW_*`,
  0.6 m or 6 deg) the map goes with its places -- kept, a map laid while
  lost pulled a corrected belief back onto a copy within 20 s.
- *Lost*: no fixture in sight where the charge's walk ended (within
  `NEAR_STANDOFF_M` of the standoff, #422), with no other robot near it.
  The map goes, the robot looks for the fixture -- the humanoid steps in by
  its LIDAR, never its belief (`HumanoidBody.lost_routine`; "The humanoid's
  charging station") -- and tries again.

- *A stand-up* (#572, Ben's call): put at its start, the belief is right
  and the map laid round the wrong one would pull it back, so the map goes
  with its places there too (`HubLifecycle._forget_world`).

Flown off the box's saves, each quadruped's own state where it truly
stood, four lost robots of four found a fixture and docked, or took their
tool, 1-6 cm out. Every relocation, fix, loss and search, and a sample
every five minutes, is a `drift` row (protocol/README.md).

**Not done.** The first slip on a fresh map (a pose graph, #381's stage 3);
a robot lost with no fixture within its search's reach; the places a drop
forgets, found again from their addresses; and the wide search can still
relocate onto a copy a map keeps.

## A robot resting across the other's way (issue #415)

**What happened.** On the quadruped pair, a robot lying down by reflex to
rest is not down (#365), so the other's drive found its way cut by the
resting robot's disc and gave up, and the resting robot never moved -- it
stood only for a command of its own, and a finished robot never gets one.
A fixture day stranded a robot four times in four at one doorway.
`scripts/make_way_spike.py` lays both maps from the true floor, rests one
robot in a doorway of the house and walks the other through it.

**The rule (Ben, 2026-09-30, #415's option 1).** A drive whose way a
RESTING robot cuts -- a plan without that robot's disc finds one -- asks it
off the way (`Navigator._ask_way`, over the pair's `Body.ask_way`), with
the way it would walk; a stagnated drive asks one resting near it or its
goal. It waits for it as for a robot that will move, `MAKE_WAY_WAIT_S`
(30 s) from the first yes at most, and then gives up as before. The asked
body says yes only resting and free to: not charging, not mid-move, not
inside a walk of its own (#395's head-on hold between two walking robots is
that walk's, and still open), not dead. Then, beneath whatever routine
holds it, a STILL command is replaced by the step aside (the humanoid's,
standing, is `humanoid/way.py`): it walks the drive's law along one plan to
the nearest floor it has SEEN, reached round the asker, `aside_clear_m` off
every point of the way -- the way the asker sent is then open whatever it
plans. Any command that moves takes over at once; a fall ends it. It is
said ("MAKE WAY ...", "MADE WAY ..."), and remembered once over: the body
moved and the mind did not decide it. A restart's save waits it out.

**Measured** on the quadruped pair, seven scenes in the house: before, six
walks of seven gave up at once; after, all seven arrived, the resting
robot 0.82-0.87 m aside in 6.7-8.9 s, with no touch and no fall. The
humanoid pair's doors are "The humanoid pair served"'s.

**What is true now:** a robot resting across the other's way is asked to
make way and steps aside, beneath whatever it holds; a drive gives up at a
resting robot only once it said no or did not clear the way in 30 s
(`tests/test_make_way.py`). Where the two maps disagree, the next section.

## A robot resting where the other's map put it elsewhere (issue #455)

**What happened.** On the quadruped pair, walks gave up naming a robot
resting 1.1-3.6 m off while the `encounter` rows put the two bodies within
2 m, and presses stalled short of the feed plate with the other robot
lying beside it. One robot's map of the lab sat 2.4-2.9 m and 9.5 deg off
for hours.

**The mechanism.** A robot resting was kept clear of where it SAID it was:
its belief, in its own map's frame, so two maps a few metres apart put its
disc a few metres from its body. The walker's planner routed round a spot
the robot was not at and straight at its body, and the depth camera held
the walk there (a robot lying down is under the LIDAR's plane, and only the
camera sees it). The stalled drive then looked for a robot near it or its
goal, to wait on and ask, and by the belief there was none. Once an ask was
made, the way was sent in the asker's frame and laid in the other's, where
it ran 1.1 m clear of it: five times it stood up, stepped 0.0 m and lay
down again. And wherever the maps agreed, a hold at the disc's edge, while
the two ways round it swapped plan by plan and reset the stagnation clock,
lasted until the walk timed out: #365's trap, the one that took the hold
off a fallen robot.

**A goal it lies on** (#439). A goal under a resting robot's disc ended
every plan at a stand-in beside it, and only a stagnation asked; the
stand-in swapped side to side, and every new route was progress. A goal a
resting robot lies on is now a way it cuts, asked at the plan that finds it
there, within `PAST_M` of it.

**Two bodies on one spot** (#455's closing reading). Every geom of the
quadruped met nothing of the other robot (the walking policy's model has
no self-contact), so only the planner kept them apart, and the walks it
does not plan were blind: nine times on one build both lay on the dock's
seat at once, and inside each other's discs, deeper than the planner's
halo escape reaches, neither could plan a first step nor step aside when
asked ("no spot"), for up to 94 minutes.

Measured on the quadruped pair (`make_way_spike.py --near-field`, the maps
true and each laid 2.9 m off): before, the walks stalled or gave up in
every scene; after, every walk arrived, the resting robot 0.81-1.02 m
aside, with no touch and no fall.

**What is true now:**

- a robot down, dead, resting or stepping aside is kept clear of where its
  body is, placed as the other's own sensors would put it
  (`HubLifecycle.keep_clear`), and the depth camera's hold skips it
  (`_near_field_step`); walking, it is where it says it is, and held for;
- a walk whose goal a resting robot lies on (`peer_on_the_goal`) asks it at
  the plan that finds it there, within `PAST_M` of it, and waits while it
  steps off;
- the way it is asked off is told relative to its body (`make_way`:
  `Body.from_seen`, then `Body.as_seen`);
- a dead robot is never waited on, held for or asked;
- a robot further off than `PAST_M` (3 m) whose disc alone cuts a walk's
  way is planned past (`Navigator._plan_to`); nearer, it is the wall it was;
- a robot that cannot make way says why, once every `MAKE_WAY_WAIT_S`;
- THE PAIR COLLIDES (Ben, 2026-10-14): each robot meets every solid geom of
  the other and none of its own (`PAIR_BITS`, bits the world never sets; a
  robot alone is unchanged), and avoidance is what keeps them apart: the
  charge seat is held by a robot on its standoff, its way in or the seat
  itself (`Body.charge_seat`), and waited for out of the lane;
- a robot inside another's disc walks out of it over the floor, on its own
  side of any wall (`Navigator._escape_from`: chosen by distance alone, a
  plan began across a 4 cm wall), and so does its step aside;
- a robot restored inside the other is put at its first clear start before
  a step throws them apart (`HubLifecycle.unclip`), and its History says so.

`tests/test_resting_in_the_way.py`, `tests/test_two_robots.py` and
`tests/test_continuation.py` pin each rule.

## The probe from the robot's own senses (issue #466, stage 2)

The quadruped's probe opened #469's chest from its own senses (at
`quadruped-final`). What it learned of seeing a box binds the humanoid's
`measure` and its reference opener, which read the box through the same
code (`perception/box.py`, `Body.depth_cloud`).

**The D435 at its own resolution.** A box is measured off the part's 848 x
480 (the near field's 120 x 70 left the top's back 3 cm unsampled), and so
is its nearest depth: Intel's tuning guide puts min-Z at the focal length
(px) times the baseline over the 126 disparities the ASIC searches, 17 cm
at 848 x 480, where the near field's 0.28 is 1280 x 720's. At 0.28 a
bracket's tip 0.26 m from the lens read nothing, and the pin found off it
came 19 mm into the box.

**What depth says of the box** (`perception/box.py`, the map frame through
the belief, `tests/test_box.py`):
- the top is the highest level a large share of the points stand at;
- an edge seen from in front -- a top's back edge, where a hinge is -- is
  read off the mean of a window inside it, which rows laid 4-5 mm apart at
  a grazing angle and their noise move 0.3 mm on average (0.7 at 6 mm apart
  under 3 mm of noise);
- an edge whose face looks at the camera -- a bracket's tip -- is a low
  percentile: its face's points stand at the tip and never past it, and the
  window's mean read it 4 mm past;
- which way it faces: roughly the least rectangle the top fills, then
  square to its far edge laid slice by slice. Trimmed, the least
  rectangle's area is flat within a degree or two of its least, and alone
  it read a plain lid of 90 000 points 2.5 deg off.

## The humanoid body (issue #507)

The next body, sized before anything is trained: our own design from real
parts, drawn from Booster T2's published model (its proportions, its legs'
joint order and ranges, its masses as a check). Parts.md, "The humanoid
body", has the parts, why each joint is there and the bill;
`scripts/humanoid_spike.py` flies every table below, and `--view body` is
the stage's scene. Every number is MuJoCo's, an upper bound on hardware.

![The `body` scene: the humanoid held up on a gantry in the house, each
joint group through its range](figures/humanoid_body.png)

(`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --out
docs/figures/humanoid_body.png` draws it again.)

**The mass** (`--mass`, the T2's segments; ours off the compiled model):

| segment | ours kg | T2 kg |
|---|---|---|
| a leg, from its hip pitch link | 9.67 | 8.27 |
| an arm, from its shoulder pitch link, with its gripper | 4.04 | 4.51 |
| the head | 0.56 | 1.56 |
| the rest: trunk, pack, electronics, the hips' and shoulders' pitch motors | 14.63 | 14.93 |
| **whole** | **42.6** | **42.0** |

The legs are heavier because both ankle motors are the leg's own (the T2
drives its ankle through a parallel pair); the arms lighter by a joint; the
head by two.

**Held in our physics** (`--torque`): the robot set down on its soles in a
pose, its centre of mass over them, every joint held by a PD stiff enough
that it sags under 0.05 rad at its part's peak, the mean |torque| over the
last second of a 3 s hold, the worse side, against the part's rating HELD
STILL (below):

| joint (part) | stand | deep crouch | on one leg | an arm out level, 1 kg | both arms out, 2 kg each |
|---|---|---|---|---|---|
| hip pitch (RS04) | 1.0 | 0.9 | 8.7 | 8.0 | 20.5 (72 %) |
| hip roll (RS04) | 0.0 | 0.0 | **41.7 (146 %)** | 0.1 | 0.0 |
| knee (RS04) | 8.6 (30 %) | **32.0 (112 %)** | 10.9 | 3.0 | 4.0 |
| ankle pitch (RS04) | 3.7 | 10.2 | 18.3 (64 %) | 8.9 | 10.1 |
| shoulder pitch (RS03) | 1.0 | 1.1 | 0.1 | **14.2 (110 %)** | **20.0 (154 %)** |
| wrist pitch (RS00) | 0.2 | 0.2 | 0.0 | 2.2 | **3.7 (102 %)** |

Standing takes every joint under a third of its rating. The rows past it
are poses the body holds for a time, not for good, and the sheets' own
overload tables say how long (`--thermal`, `parts.held_for`): the deep
crouch over 130 s, one leg 118 s, an arm out with a load over 190 s.

**Estimates for walking and the stairs** (`--torque`, `ESTIMATES`): ⚠ NOT
OUR PHYSICS until stages 4 and 5 fly them (#507, rule 7). Published peak
loads per kg of body, the upper end of each range and a robot's own where
one was measured, scaled by mass alone (the gait lab's rule: children of
1.47 m sit inside the adult per-kg range at one speed):

| joint | walking ~1.25 m/s | stairs up, 0.16-0.22 m | stairs down |
|---|---|---|---|
| hip pitch | 55 (46 % of peak; human) | 39 (32 %; human) | 26 (21 %) |
| hip roll | 45 (37 %; G1) | 92 (76 %; Adam Lite) | no data |
| hip yaw | 32 (53 %; G1) | **51 (85 %; Adam Lite)** | no data |
| knee | 49 (41 %; G1) | 92 (77 %; Adam Lite) | 68 (57 %; human) |
| ankle pitch | 70 (59 %; G1) | 72 (60 %; human) | 64 (53 %; human) |
| ankle roll | 12 (20 %; G1) | no data | no data |

The G1's are read off Feng 2026's plots of the robot walking on hardware;
Adam Lite's off its own figure of 15 cm stairs climbed on hardware (60 kg;
arXiv 2509.12858). Adam's RMS over the stairs, scaled, holds the hip roll's
winding at 127 °C of 145 at 25 °C ambient (92 % of its turning rating) and
the knee's at 88: **the hip roll is where rule 7's re-read looks first.**
The ankle pitch is why it is an RS04: walking's push-off, 70 N·m, is 117 %
of an RS03's peak.

**Energy** (`--energy`): 69 W is drawn whatever the joints do, 41 W of it
the drivers' standby -- the RS03's "<= 40 mA", the only RS part that
publishes one, carried for all 24 -- and 15 the e-stop contactor's coil.
Standing draws 80 W: 9.7 h on the 778 Wh pack. Walking, ESTIMATED off the
T2's and the G1's published runtimes per kg, is 244-256 W: 3.0-3.2 h.

**The sensors** (`--sensors`, the stand): the D435, fixed 1.26 m up and
pitched 40°, sees the counter's height (0.94 m) from 0.12 to 1.64 m ahead
and the floor from 0.48 m; the floor at the feet is out of its view, so a
flight is seen on the way to it. The LIDAR's plane is 1.41 m up, over the
tables, the couch and the bed.

**What the build turned up, each pinned in `tests/test_humanoid.py` or at
its constant:**
- *A pose held still is a stalled motor.* RobStride rates a held torque
  below a turning one (one phase carries the current, "1.414 times" its
  heat): the RS04 is 35 N·m turning and 28.5 held, the RS03 20 and 13. An
  arm held out with 1 kg is 71 % of the RS03's turning rating and 110 % of
  its held one (`Motor.stall_torque`).
- *A statue cannot climb the house's stair.* All the weight on a knee bent
  for a 0.18 m riser, the body upright over its ankle, asks the ankle past
  the T2's 48° of dorsiflexion (-0.84 rad); leaned 0.35 rad forward, past
  its 18° of roll. A climb shares that instant with the trailing leg, which
  is a policy's to do (stage 5): the stairs rows stay estimates.
- *A weld added through `MjSpec` writes its own data.* Left at the default,
  the gantry's weld held a point a metre off the pelvis (the data's first
  three numbers read 0 1 0): the body sagged 42 mm swinging its legs and,
  turned 90°, fell to the floor. Written -- no offset, to a hook whose frame
  is the pelvis's own -- it holds at MuJoCo's own time constant
  (`humanoid_spike.held_world`).
- *A gripper's jaws are geared together; MuJoCo's equality is soft.* At its
  default the second jaw lagged the first 2.8 mm under a full squeeze, and
  the gap the gripper reports off its one encoder read 37 mm on a 40 mm
  block; at 0.004 s, 0.7 mm (`model.JAWS_SOLREF`).

## The humanoid stands, reaches and grasps (issue #507, stage 2)

The humanoid on its own feet: its legs a standing policy trained in mjlab
(`training/humanoid_train/stand.py`, `Pluggy-Humanoid-Stand`) and run in
numpy (`humanoid/policy.py`), its arms an inverse-kinematics reach
(`humanoid/arms.py`, `humanoid/reach.py`), its grippers judged by their own
reading. Everything that steers reads the robot's senses: the IMU part
through its attitude filter and every joint through its driver's reply
fields; a test spies on what the code reads off `MjData` and fails on any
of the simulator's truth (`test_the_robot_steers_off_its_senses_...`).
Every number is MuJoCo's, an upper bound on hardware.

![The `reach` scene: the left hand held at a counter's height, a high
shelf, a low shelf and the floor, through Body with its get-up
loaded](figures/humanoid_reach.png)

(`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --strip reach --out
docs/figures/humanoid_reach.png`; `stand` and `grasp` beside it.)

**The pipeline, first, on a body its maker walks (#507, rule 6).** mjlab's
stock G1 flat task, 3000 iterations on a rented RTX 5090 (22.5 min, about
218,000 steps a second), flown OUTSIDE mjlab by `humanoid_train/g1_check.py`
-- its ONNX read into arrays and run in numpy, MuJoCo's C engine at the 5 ms
it trained at and the served 2 ms -- walks: 0.5 and 1.0 m/s forward within
8 %, backward, sideways and turning within 15 %, and never falls. A failure
from here is our design's or our training's.

**How it stands.** The policy drives the legs' twelve joints and sees all
twenty-four encoders, so it knows where its arms' mass is. In training the
arms are SCRIPTED (`ScriptedArms`): sent somewhere new every 1-4 s, ramped
at 1.5 rad/s, their weight fed forward off the bias force, as the reach's
commands carry it. It is asked a POSTURE -- the pelvis's height off the
stand's (to -0.25 m) and the trunk's pitch (to 1.1 rad) -- ramped at
0.15 m/s and 0.6 rad/s, and disturbed by shoves to the trunk, pulls at the
hands and up to 2 kg in each. Six runs taught the charges:

| run | what changed | what it did in our physics |
|---|---|---|
| stand1 | full-size shoves (0.4 m/s) | episodes fell from 950 to 338 steps: shoves past the capture point need a step, and nothing paid for one |
| stand2 | shoves at 0.6 of that | stood, but its feet CREPT: 10 cm in an 8 s bow -- the charge on the soles' speed squared (`feet_still`) sees nothing in 0.5 cm/s |
| stand3 | + each sole's distance from its touchdown | a step renewed the anchor, so a step was free: bowing, it stepped its feet 40 cm |
| stand4 | the anchor is the EPISODE's stance (`feet_anchor`, -10/m) | feet 1.1 cm in 20 s of reaching; committed |
| stand5 | the anchor at -25/m | fell rather than step (its tilt lost 9 times a window to stand4's 3) |
| stand6 | stand4 with shoves at 0.4 | no steadier: from the front it fell at 20 N*s, where stand4 stepped |

**How it reaches.** A target is a grip pose in the STANCE frame -- on the
floor between the soles, x along their heading -- fixed in the world while
the feet stay put; the robot poses its MODEL OF ITSELF (its own drawing,
`hm.body_xml()` compiled alone) from its encoders and its IMU's tilt in that
frame, and solves the arm's six joints there by damped least squares. A
reach is PLANNED (the least-bent posture, on a statue's legs, from which the
target and its pre-grasp 8 cm short are in reach with 1.5 cm to spare, else
at full stretch), MOVED in joint space to the pre-grasp while the posture
comes, then APPROACHED along a straight line at 0.2 m/s and HELD, solved
again every 10 ms as the trunk sways; a command carries the arm's weight
off its model and the stiffness its caller asks. The posture is held by its
own loop: the policy is asked the plan's posture plus the integral of what
the robot's model says it misses by (it stood 2 cm under the height asked).

**Measured** (`humanoid_spike.py --shove`, `--envelope`, `--hold`, on a bare
floor, a fresh robot a trial):

| shoved | standing: feet held to / stood to, N*s | crouched 0.2 m |
|---|---|---|
| from the front | 10 / 10 (fell at 15) | 15 / 15 (fell at 20) |
| from behind | 15 / 15 | 20 / 20 |
| from its left | 15 / 15 | 20 / 20 |
| from its right | 15 / 15 | 15 / 20 |
| front-left, front-right | 15 / 15 | 20, 15 / 20 |
| back-left, back-right | 20, 15 / 20 | 20 / 25 |

The shove is a force at the trunk's centre of mass for 0.1 s; 10 N*s
changes the whole body's velocity by 0.23 m/s. Crouched it stands more
(its centre of mass lower). Training shoved at up to 0.24 m/s a direction,
about 10 N*s: a policy that steps to recover is #507's stage 5 (falls).

The left hand's reach, in the stance frame (m), flown and held 2 s
balancing: each cell the grip's true miss held, cm, with its grip (F
fingers forward, U tilted 45 deg up, D down); `--` planned but the robot's
own solve stopped it short (`reach.BLOCKED_M`); `.` no posture the policy
holds puts it in reach:

| z \\ x | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 |
|---|---|---|---|---|---|---|
| 1.55 | . | . | 0.4 U | . | . | . |
| 1.43 | . | . | 0.4 U | 3.0 U | . | . |
| 1.25 | . | . | 0.3 U | 0.5 U | -- | . |
| 1.10 | . | 0.6 U | 0.5 U | 0.7 F | -- | . |
| 0.94 | 0.3 U | 0.2 U | 0.8 F | 0.8 F | 0.3 F | -- |
| 0.80 | 0.4 F | 0.5 F | 0.6 F | -- | -- | -- |
| 0.65 | 0.3 F | 0.4 F | 0.4 F | 0.3 F | 1.7 F | 1.4 F |
| 0.49 | . | 0.4 F | 0.8 F | 0.3 F | 1.1 F | 1.1 F |
| 0.35 | -- | 0.3 F | 0.2 F | 0.3 F | 0.6 F | . |
| 0.20 | 0.2 D | . | . | . | . | . |
| 0.09 | 0.3 D | . | . | . | . | . |

The `--` cells at the edge are the reach's own word: planned with slack on
a statue's legs, the line in tracked one arm's branch until the wrist's
range ended it, and the hand stopped where its solve missed by 3.7-8 cm
(the feet moved 6 mm: the stance did not move).

The heights #507 names, 0.09, 0.49, 0.94 and 1.43 m, are all reached, the
`reach` scene's four held 0.1-0.9 cm off through `Body`, its get-up loaded
as served (#520): the floor only near the toes and at full stretch from
the deepest bow the policy holds (the hip folds to -2.34 rad there, its
thigh a centimetre off the torso); 1.43 m with the
fingers tilted up 45 deg. **A waist pitch joint, or 3 cm more arm, would
give the floor slack** -- the design question for rule 7's re-read.

**The humanoid's pull** (`--hold`): the drawer locked, its handle gripped --
held by the gripper's own reading, its jaws 22 mm apart on the 24 mm bar,
squeezing 111 N -- and the hand's goal drawn back a centimetre each half
second at the reach's full stiffness. The pull truly on the handle rose to
33.8 N at 2 cm and 37.4 N at 5 cm, the hand's reading 31.4 and 36.9 N;
between them it fell back twice as the grip slid 29 mm along the bar and
the feet up to 27 mm toward the counter. **About 35 N is what it pulls
standing**, the grip and the stance giving together; how near the elbow's
RS02 (17 N*m peak) a hard pull takes it is rule 7's re-read to make.

**The force at the hand** is read off the arm's six drivers' torques (their
current, counted and noisy) less what holding the arm still takes, through
the grip's Jacobian (`SelfModel.hand_force`): a 20 N push on a still hand
reads 20.2 N, 0.4 N across it; the `grasp` scene's drawer, drawn 14.5 cm
against its spring with a compliant arm, read 4.1 N against a true 3.8 N.
While the body works under a hard pull the reading scatters 7-15 N about
the truth: the estimate assumes the arm is still.

**What the build turned up, each pinned in `tests/test_humanoid.py` or at
its constant:**
- *A reach solved in one jump from a hanging arm falls into the wrong
  basin* (the shoulder thrown back, the elbow over): a hanging arm is
  nearly singular for a hand moving up. The plan solves from seven seed
  arms; the motion tracks from wherever the arm last was (`reach.SEEDS`).
- *A level grip over the shoulder runs out of wrist.* At 1.43 m the wrist's
  pitch (+-1.6 rad) held the gripper level only at full stretch; tilted 45
  deg up, it reaches with slack.
- *The posture shown is not the posture held.* A deep bow's body takes a
  second or two after its command; an approach begun on the command's
  arrival stopped short. It begins on the robot's own measure
  (`reach.POSTURE_TOL`), and once its trunk is still (`reach.TRUNK_STILL_RATE`,
  #520): the force read off the drivers assumes the body still, and the
  floor's approach begun as the bow stopped turning (0.59 rad/s) read
  492 N at the hand and stopped 6 cm short of it.
- *A hand that cannot get there stops* (`reach.BLOCKED_M`, `BLOCKED_N`,
  held `BLOCKED_S`): its own solve 3 cm out of reach, or the force it reads
  (its arm's currents then, the whole body's since #547) a 30 N push on its
  way in. Unstopped, a crouch left a chest's knob 10.7 cm out of reach and
  the arm drove the gripper into the counter's edge until it pushed the
  robot over backwards. The force read off the drivers assumes the body
  still, and a deep bow's folding read 81 N for an instant: the push must
  last 0.15 s.

**Compute** (#507's stage 2 line: ~$8 likely, $14 the stop): the G1 check
and six standing runs on two rented RTX 5090s (Vast.ai, $0.49/h), 2.83
GPU-hours, **$1.53** off Vast's own charges.

## The humanoid behind Body (issue #507, stage 3)

`HumanoidBody` (`humanoid/body.py`) answers every `Body` member, and
`body.body_for` chooses it by the robot's own joint names. Its posture,
senses and arms are stage 2's; a member it cannot serve says why
(`NotImplementedError`). `Body` gains a hand's members -- `reach_routine` (a grip pose
in the believed world frame, the posture the body's to choose),
`release_routine`, `grip_routine`, `gripped` (the gripper's own reading)
and `hand_force` (off the arm's currents; a hand at work, standing, off the
whole body's since #547) -- which the stub answers.

![The `demo` scene: #466's chest on a counter; the humanoid finds its knob
by its tag, reaches it and grips it, through Body](figures/humanoid_demo.png)

**The demo world** (`evaluation/counter.py`, a world of its own): #466's
drop-handle chest on a counter in the
storeroom -- 0.72 m since #493, 0.90 before -- seeded (`set_out(k)`), the
humanoid standing in front of it.
Every act is a `Body` member: the knob found by its tag in the D435's
colour imager (`spot`: the face seen most square-on, its middle half a cube
in), a reach with the fingers tilted down onto it, the gripper shut and the
grip judged by its own reading. Over twelve set-outs
(`humanoid_spike.py --demo-table`) at 0.72 m **all twelve held the knob**
-- held by the gripper's reading and the pads truly on it (our grading),
still holding after 2 s -- its reading agreeing with the truth every time;
at 0.90, ten, and the robot said NOT HELD of the other two, truly so. The
tag put the knob 0.5-0.6 cm from where it truly hung.

**What the chest taught:**
- *A drop handle's knob hangs nearly under its pin*, its drop bar and stem
  behind and above it (the world compiles it 0.57 rad from where it rests,
  and it swings there in the first second): jaws closing up and down
  brought a pad down onto them, swung the handle 13 deg and stalled 79 mm
  open on them -- HELD by the gripper's reading, the pads truly off the
  knob. The jaws close ACROSS it, fingers tilted 25 deg down
  (the scene tries 15, 35 and 45 after).
- *Level, the plan crouched, and the elbow met the counter*: the arm drove
  the gripper into the counter's edge until the robot fell backwards. Two
  answers: the reach stops a hand that cannot get there (stage 2's
  `BLOCKED_*`), and the fingers tilt down so the plan reaches from the
  stand, the elbow over the counter.
- *The near edge*: at 0.90 m, drawn 0.34 m ahead, the knob hung under 0.37
  m in two of ten set-outs, and at 0.99 m up no grip the arm holds folded
  the hand back that near; that set-out drew 0.37-0.43 m. At 0.72 m it
  draws 0.33-0.37 ahead, the lid's arc in reach (SimNotes, "The humanoid
  opens the chest"), and nearer the knob leaves the colour camera's view.
- *A headlight lights the robot's cameras too.* Drawn brighter, as the
  other scenes are, it moved where the knob's tag decoded by 3 cm: a scene
  that renders for the robot keeps the house's own light, and a filmstrip
  brightens its own tiles alone.

## The humanoid walks (issue #507, stage 4)

The humanoid walks the house through the `Navigator`, as the quadruped
did (`humanoid/mission.py`): its map off its head's LIDAR, the D435's
points under the LIDAR's plane a layer the planner plans round, its pose
off its legs and IMU (`humanoid/odometry.py`) matched to its map. Its legs
are the WALKING policy's (`models/humanoid_walk.npz`, `humanoid/walk.py`)
while it moves and the standing policy's while it reaches; they change
hands only at a standstill, its arms home (`to_walk_routine`,
`to_stand_routine`).

![The `walk` scene: from the living room through the hall into the kitchen
and back, steered by its own map](figures/humanoid_walk.png)

**The walk** (`humanoid_spike.py --view walk`): from the living room
through the hall into the kitchen, about 13 m, in 37 s, 9 cm from the goal
with its belief 17 cm off the truth; and back in 33 s, 2 cm from the goal
and its belief 2 cm off -- the scan matcher pulled the odometry's drift
back on the way. No falls.

**The house's walls were the rover's 1.2 m; its LIDAR spins at 1.41 m.**
It saw over every wall, and a map of its plane alone was empty: stage 5
raised the house's walls to a room's 2.4 m.

**The walking policy** (`Pluggy-Humanoid-Walk`, walk2, the served walker:
mjlab's G1 recipe on our body, its arms scripted more quietly than a reach,
its legs meeting each other). In our physics (`humanoid_spike.py --track`:
each command from standing, measured over 6 s after 3 s) it walks 0.51,
0.99 and 1.27 m/s asked 0.5, 1.0 and 1.3; backward 0.46 asked 0.5 (0.20
asked the Navigator's back-off, 0.3); sideways 0.31 asked 0.3; and turns
on the spot 1.01 rad/s asked 1.0.
- *A turn is the hips' yaw alone* -- there is no waist. With the hips'
  yaw held as tightly as the G1's (0.15 rad) and the turn's reward paying
  61 % for not turning at all, `walk1` asked 0.5 rad/s turned 0.08. Twice
  the yaw's tolerance and a sharper reward (`STD_WALKING`, `TURN_STD`):
  `walk2`. **A slow turn on the spot is still its weakest command**: 0.35
  and 0.28 rad/s asked 0.5 either way, and out of a walk it can stall a
  few seconds before it turns.
- *The legs meet each other* (`model.LEG_GEOMS`'s contact pairs): the
  robot's own geoms touch nothing of the robot's, and a gait could pass one
  leg through the other.

**Rough ground** (`Pluggy-Humanoid-Walk-Rough`; `humanoid_spike.py --rough`
flies the served and the rough walker over courses of what that task
trains on, each eight times at 0.6 m/s from a start drawn askew, graded by
us):

| walker | flat | 5 cm doorsteps | paving to 4 cm | 10 deg slope | blocks to 8 cm |
|---|---|---|---|---|---|
| walk2, served | 0 fell, 7.1 m | **8 fell**, 1.9 | 1 fell, 5.2 | 0 fell, **1.1** | **7 fell**, 3.2 |
| rough4, not served | 0 fell, 7.2 | 0 fell, 7.2 | 0 fell, 7.0 | 0 fell, 5.9 | 0 fell, 5.9 |

(falls of eight, and the mean distance made of 7.2 m asked.)

![The `rough` scene: the walker that sees the ground across two doorsteps,
paving, a 10 deg slope and blocks on its own map, steered off its own
odometry](figures/humanoid_rough.png)
- ⚠ *Box soles blow up on a height field.* mjlab's rough terrains are
  mostly height fields, and on them the box soles blew up in MuJoCo Warp:
  flown by the flat walker, about one world in seven on the rough patches
  within 20 s (root speeds to 1e5 m/s), 2-4 % on the slopes, none of 978
  on the boxes. The humanoid's rough1 ended in a NaN at its iteration 1273;
  rough2's critic was wrecked by them (an iteration's mean return of
  -3,244, the value loss from 0.13 to 9e6 and on to 9e16, the action noise
  from 0.36 to 2.6 in 450 iterations). The task's terrain is BOXES alone
  -- blocks, low stairs up and down, tilted plates -- and a world that
  blows up anyway is reset (`walk.blown_up`); flown on the boxes, none did.
- ⚠ *Blind, it gives up what risks a fall.* Under mjlab's terrain
  curriculum (rough4, 2000 iterations on from walk2; replaced since by the
  walker that sees the ground) it never falls on a course, and it
  walks up and over the slope the served walker stalls at -- but asked to
  walk backward or sideways it stands still, on flat ground too; it turns
  on the spot 0.07 and 0.23 rad/s asked 0.5 either way, and walking
  forward it drifts sideways at 0.15-0.25 m/s (`--track`). It cannot tell
  rough ground from flat, so it stood still wherever a command had risked
  a fall. The Navigator backs off and turns on the spot, so **it is not
  the served walker**. Every difficulty at once (rough3) narrowed it
  further: it stood still at 1.3 m/s too, and turned 0.08 asked 0.5.
  Sharper tracking rewards (0.2 m/s and 0.3 rad/s wide, rough5) paid
  nothing for a walk it could not track precisely, and at 3,850 iterations
  it stood still at 1.0 m/s too; it was stopped there.
- *The `rough` scene* (`humanoid_spike.py --view rough`): the walker that
  sees the ground across a doorstep twice, paving, the slope and blocks end
  to end, 19 m in 32 s (rough4 took 43). ⚠ Walked open-loop rough4 turned
  off the courses within 20 s; the scene steers back to their middle line
  off the robot's own leg odometry and IMU.
- *The served walker does not walk up a 10 deg ramp*: it climbs a few
  tens of centimetres, stops stepping and slides back to its foot (so did
  rough3). The house has none; its stairs are stage 5's. The walker that
  sees the ground walks up both ("One walker that sees the ground").

**Its odometry**: the heading the IMU's (within a degree over a walk), the
position moving as the planted foot does not -- the lower sole, levelled,
off the legs' encoders through its model of itself. On a bare floor, with
nothing for a scan to match (`humanoid_spike.py --odometry`, 8 s a
command), it reads 4-6 % short walking forward or back, 3 % on a walk with
a turn, and 7 % sideways once the planted foot is told at 3 mm (at 1 cm a
shuffle's swing foot read planted, and sideways 21 % short: "Stepping
toward the work"); the Navigator never walks it sideways.

**Rule 7, read again for walking** (`humanoid_spike.py --torque`, its
walking rows FLOWN by the policy, 10 s at each speed, the start left out):

| joint (part) | 0.5 m/s: peak (% peak), RMS (% rated), winding C | 1.0 m/s | 1.3 m/s |
|---|---|---|---|
| hip pitch (RS04) | 23.1 (19 %), 11.0 (31 %), 52 | 35.2 (29 %), 16.6 (47 %), 67 | 47.6 (40 %), 21.8 (62 %), 86 |
| hip roll (RS04) | 33.7 (28 %), 17.8 (51 %), 71 | 35.9 (30 %), 18.1 (52 %), 72 | 38.8 (32 %), 21.3 (61 %), 84 |
| hip yaw (RS03) | 13.5 (22 %), 5.4 (27 %), 49 | 14.2 (24 %), 6.2 (31 %), 52 | 16.7 (28 %), 7.0 (35 %), 55 |
| knee (RS04) | 37.2 (31 %), 13.2 (38 %), 57 | 57.7 (48 %), 17.8 (51 %), 71 | 52.0 (43 %), 19.3 (55 %), 77 |
| ankle pitch (RS04) | 44.6 (37 %), 15.6 (45 %), 64 | 64.4 (54 %), 24.2 (69 %), 98 | 71.6 (60 %), 27.2 (78 %), 113 |
| ankle roll (RS03) | 11.8 (20 %), 5.5 (27 %), 49 | 15.6 (26 %), 6.4 (32 %), 52 | 20.9 (35 %), 7.3 (37 %), 56 |

The winding is held steady by that RMS on a 40 C day, of the RS line's
145 C. **No joint passes its peak or its continuous rating walking: flat
walking sends nothing back to the design.** The ankle pitch is nearest
both, at the top speed, and the estimate had it (70 N*m, 59 % of the
peak; flown 71.6). The stairs' rows stay estimates until stage 5.

**Compute** (#507's stage 4 line: ~$11 likely, $16 the stop; stage 3
trained nothing): one rented RTX 5090 (Vast, $0.49 an hour) for 3.4 hours
-- walk1 and walk2 (23 min each), rough1 (a NaN at 19 min), rough2
(diverged, 12), rough3 (33), rough4 (27) and rough5 (stopped at 13), three
starts out of memory and the blow-up diagnostic -- **$1.65** off Vast's
own charges.

## The humanoid climbs, falls and gets up (issue #507, stage 5)

The served house changed for it, its senses learnt to see the ground ahead
and keep their height on a flight, and its loop was timed against the
quadruped pair's. Every number is MuJoCo's, an upper bound on hardware.

**The house is a room's height, and its stairs are steps** (`home/world.py`;
Observatory.md has the period). Every wall is 2.4 m, where the rover's were
1.2: the humanoid's LIDAR spins at 1.41 m and saw over every one (stage 4's
walk scene raised them in a world of its own; that world is gone). The
staircase, one solid 0.9 m box since #68 -- the rover's LIDAR rode at
0.223 m and could not have seen a riser -- is the flight the website drew
inside it: five 0.18 m risers on 0.30 m treads against the hall's west
wall, a landing 0.9 m up filling the rest of the block, and a half wall
0.9 m tall on its open edges (`stairs_geoms`). On the humanoid's head the
LIDAR is 2.31 m up on the landing, still under the walls.

**What it sees of the ground ahead** (`humanoid/scan.py`): the D435 on its
head laid into a map through its belief, read at the seeing walker's grid
(the terrain under 1.6 x 1.0 m at 0.1 m, from 0.5 m behind the pelvis to
1.1 m ahead), a point unseen reading the nearest cell seen (#388). Walked
to the flight's foot, the map's scan was the world's to a median 1.6 cm
(90th percentile 2.9 cm). What it took:
- ⚠ *The ground under a sole is only what it saw or stood on.* At boot the
  nearest seen cell to the feet was a wall's, 0.67 m up, and the height the
  odometry anchored there was too: `under` reads seen cells alone, and a
  planted sole writes its footprint into the map (`feel`) -- the ground it
  stands on is known before it has looked.
- ⚠ *A frame's height is registered against the map, cell against cell.*
  The legs alone ran the belief's height away a metre in a few seconds of
  flat walking, and the map with it. Each frame now overlaps ground already
  mapped, and the median height between it and the cells it lands on moves
  the belief (`_drift`). Point by point against the cells' maxima (the map
  keeps a cell's highest point) it read ~7 mm low every frame and ratcheted
  the belief up 9 cm in 7 s; each cell's highest new point against the
  stored one reads nothing.
- *Our ideal scan casts against the world's geoms alone*, as training did:
  cast at everything it hit the robot's own head.
- ⚠ *The camera's rays are cast unculled* (`depth.UNCULLED`, both the
  humanoid's cameras). MuJoCo's batched cast skips a geom whose bounding
  sphere lies past its cutoff, and a plane's sphere is its anchor point:
  cast to the camera's 3 m range, 2.75 m from where the bare floor is
  anchored the camera saw no floor at all, its map froze, its height went
  unregistered, and a seeing walker fell at 1.0-1.3 m/s on a flat floor
  (shown the true ground, never). Unculled a frame costs the same (3.9 ms
  against 4.2 on the bare floor; 10-14 ms either way in the house). Every
  bare-floor table of a seeing walker before it -- tracking, the rough
  courses, the bare flight coming down from a landing 3.1 m out -- was
  flown half blind, and the tables below are flown again.
- ⚠ *What it is shown is ground it could step on* (`HeadScan.ground`).
  Training's ground is floors, blocks and flights, all of a piece; the
  house's flight runs between the hall's wall and the deck's face, and
  shown them as they stand the walker fell over backwards a metre short of
  the first riser (between the same walls with them taken away, it walked
  on). Ground is what joins the ground under the walker through rises no
  taller than a riser (0.25 m), across what it has not seen at the nearest
  height it has, and is no wall's -- a cell where one frame's points ran
  taller than 0.3 m, a wall's face filling it from the floor up; a point
  that is not reads as the nearest that is. NOT BY HOW HIGH A POINT IS:
  half a metre to the side, a wall seen only at its foot read 0.64-0.75 m,
  as high as the treads a slope rule had to keep, and a counter's top
  1.1 m ahead is a flight's fifth tread. The wall rule is the one the
  odometry needs too: a sole set against a wall read the wall's top as the
  ground under it.

**Which foot is planted on a flight** (`humanoid/odometry.py`): on a stair
the lower foot is the TRAILING one, lifting off its tread while the leading
one carries the body; stage 4's rule (the lower foot) kept it planted until
it rose a riser past the other -- the height sinking a riser a step, the
pelvis moved back as the foot swung forward. Where the mapped ground under
the soles differs by more than 5 cm, the planted foot is the one nearer the
ground under it -- the height that carries 30 % of its footprint, never its
highest point (a toe at the next riser's edge read the tread above) -- and
on one floor the lower-foot rule stands (swapping on the map's centimetre
of noise read a walk 10 % short where the rule read it 2.5 %); the planted
sole is followed by its LOWEST corner, the one on the ground (its middle
travels 2-3 cm a step as it rolls heel to toe). ⚠ The height moves only as
the planted sole does, NEVER LAID OFF THE MAP: laid off the ground under a
landing sole, a footprint believed 6 cm up the flight read the next tread,
the height jumped a riser, the map was laid a riser high from there, and
the walker stumbled on the third tread; continuous, it stayed within 3 cm
to the landing. Coming down, two more: the ground under a sole is ground it
could step on, as the scan's is (from the landing the half wall beside the
flight shows its top alone, no face for the span rule to see, and a sole
beside it read the ground under it 1.4-1.6 m up); and a planted sole tipped
over a tread's edge is followed by the corner nearest the mapped ground
under it, where the ground under its corners differs by a riser (stepping
off the landing its toe hung 10 cm below the heel's tread, and followed by
its lowest corner the height rose 9 cm as the body stepped down 10). Three
other signals failed:
a load off the legs' torques (standing it read 215 N against a true 215;
walking, a swinging leg's own torques read up to 400 N), the soles' motion
against each other (a sole rising and the other sinking read the same: the
descent's case), and the IMU's velocity against each foot's (the height
drifted a metre in 8 s).

**Two walkers that see the ground were trained** (`training/README.md`):
the seeing walker over floors, blocks, plates and flights, and a stairs
specialist over flights alone at the house's tread. Carried on from walk2,
the seeing walker kept walk2's 2.5 cm shuffle and never left the easiest
row of risers; fresh with the easiest at 0.05 m, nor did it; from 0.02 m
both climbed, and both stopped climbing rows by iteration 4000 (up near
0.13 m on average, down near 0.14). The specialist climbed better then;
the seeing walker carried on with narrow tracking climbs better than
either and is the one walker that sees the ground now (the next section,
which has its tables and the descent's state). Each fix above is pinned in
`tests/test_humanoid_stairs.py`.

**Falling, and getting up** (`humanoid/getup.py`, `Pluggy-Humanoid-Getup`
and `-Getup-Head`; `humanoid_spike.py --fall`, `--view fall`, `--view
getup`). Without a get-up a fall is today's `stuck` death: a heart, and on
a served world a stand-up 300 s later. The get-up policy takes every joint,
arms too, once the IMU reads the trunk tipped 35 deg past the posture its
legs hold (below: a bow is no fall) or the pelvis has sunk under 0.40 m
over the sole it stands on (shoved hard from the front, the standing
policy folded into a squat 32 deg off upright, and the tilt alone never
fired), and hands back once it has stood still 0.5 s. Shoved 45 N*s each of
six ways on a bare floor, by our grading off the world:

| falling as | stood | the head struck at, m/s (the floor's push back, kN) | gears past their part's peak | the hardest into a hard stop |
|---|---|---|---|---|
| the standing policy | never | 3.8-5.0 (5.5-10.0) in four falls of six | 0-10 joints, to 3.9x (an elbow) | 530 N*m |
| limp | never | 1.2-5.3 (to 9.7) in five | 0-4 joints, to 4.0x | 232 N*m |
| the get-up policy | all six, 2.7-4.2 s | 0.3-3.3 (at most 0.3) in three; in three not at all | 1-8 joints, to 1.2x | 592 N*m |

From lying it stood in 3.0 s off its back, 2.1 off its front and 2.7 off
its side. The kilonewtons are MuJoCo's soft contact -- the shape of a
blow, not a number to bank -- and the torque through each gear counts what
its rotor's inertia takes as the joint is stopped short, the shock that
strips a gearbox. What it took:
- *A lift that fades*: the pelvis held up by half the body's weight, to
  nothing by iteration 2400 (HoST's way; ⚠ the event's body must be in its
  params, or the lift went on every body and the robots flew).
- ⚠ *The head charged on its own*: on the shared charge for any part's hit
  past two body weights, falling forward or sideways it let the head strike
  at 5.5-6.9 m/s, harder than the standing policy's own falls. Charged 20 a
  body weight past half a body weight (the rewards are scaled by the step,
  so a 5 kN hit costs a tenth of an episode's pay) and four episodes in ten
  a fall, 2000 more iterations brought the hardest head strike to 0.3 kN.
- ⚠ *Every start it must get up from*: with no side-lying start (dropped
  any way up was 0.2), it lost getting up off one side, which the policy
  before it had; 1000 iterations with sides among the starts brought it
  back.

In the house, shoved the same four ways at five places, it stood 17 times
in 20 within 15 s, the slowest 11.9 s, on the flight four in four. Down at
15 s: shoved sideways in the 3 m hall, against the wall it fell into; and
twice at the landing's edge, once to the foot of the flight, where it stood
at 30 s. So a fall into a wall or off the landing is a `stuck` death still,
and getting up among walls and flights is the training that would close it
(the get-up trained on open floor alone). The stuck budget is 20 s, as
the quadruped's was (`HumanoidBody.GETUP_BUDGET_S`). The mission waits out a
get-up before handing the legs to a walk (⚠ a fall mid-walk with the arms
home found them at rest and took the legs off the floor; one on the step
that brought them home went to the walker, #520) or back from one (⚠ a
fall while the walker settled went to the standing policy at the settle's
end, #520), and on standing again forgets its map of the ground and starts
its height again over the ground it stands on, as at boot (⚠ kept through
a fall down the flight, the height was 0.41 m off at the foot for good).

**A bow is no fall** (#520). The IMU is on the pelvis and the body has no
waist, so a bow the standing policy is asked tilts it by the whole bow, and
read off upright the watch took the reach's deepest postures for falls:
through `Body`, a low shelf (0.8 rad) and the floor (1.1 rad) each counted
one and the get-up took the joints -- and the reach said it had arrived,
because the stand-up's fresh reach hangs and `Reach.done` counts a hanging
hand (and a hand stopped short) as done. The tip is read off the posture
the legs hold now (`getup.tipped`, `Reach.held_pitch`: what the standing
policy is asked less the trim that ask carries, so exactly upright at the
stand -- and no trim where the deepest bow's ask is held at the policy's
range, or a stand's trim read the floor's held bow 2.3 deg tipped, not
1.5; the walker's, upright): a bow's trunk follows its posture within
2.5 deg going down and 1.5 coming up, and stage 5's 38 shoves from the
stand come out row for row as before. From each posture a reach takes,
shoved four ways at 10-45 N*s (`--fall`'s third table; on the standing
policy alone, graded by us, then as served; times from the shove):

| posture | falls | taken first by the tilt, the collapse | ahead of the strike, s | shoves it stood through called falls: now, off upright | stood, served |
|---|---|---|---|---|---|
| the stand | 11 of 16 | 8, 3 | 0.22-0.60 | 0 of 5, 0 | 11, 2.7-4.1 s |
| -0.20 m, 0.55 rad | 10 of 16 | 7, 3 | 0.05-0.40 | 0 of 6, 0 | 10, 2.7-4.0 s |
| -0.22 m, 0.80 rad | 10 of 16 | 7, 3 | 0.04-0.46 | 0 of 6, 6 | 10, 2.6-4.7 s |
| -0.25 m, 0.95 rad | 9 of 16 | 6, 3 | 0.04-0.45 | 0 of 7, 7 | 9, 2.6-3.9 s |
| -0.25 m, 1.10 rad | 9 of 16 | 6, 3 | 0.03-0.48 | 0 of 7, 7 | 9, 2.6-3.9 s |

The most a shove it stood through tipped it past its posture was 8.4 deg.
⚠ A forward fall from a bow folds at the knees, and THE COLLAPSE TAKES IT,
0.03-0.08 s before they strike: lying on its face is only 27 deg past the
deepest bow, so the tilt alone would never fire there. As served the head
struck in 22 of the 49 falls at 1.0-5.3 m/s (on the standing policy alone,
in 36 at 0.2-5.0): the get-up's training tips its falling starts over from
the stand, never from a bow. `reach_routine` says True only of a hand
holding at its goal now, never on a stop or a fall, which ends it at once;
a fall ends a ramp of the arm (`move`) too -- ramped on, it drove the
stand-up's fresh reach -- and the stand-up keeps the grippers as they were
asked, where a fresh set read a shut gripper as never shut.

**The sizing tables, re-read with the stairs and the get-up** (rule 7;
`humanoid_spike.py --torque`, each joint's peak against its part's peak
and its RMS against the part's continuous rating), flown by the walker
that sees the ground (the next section). Up the house's flight, 5.9 s on
it, the knee peaks at 118 N*m (98 % of the RS04's) with its RMS 110 % of
its rating -- a burst, which the part's overload table lets it hold for at
least 290 s (a winding's time constant is minutes; the stairs specialist,
climbing in 11 s, peaked at 108 and 89 %) -- and coming down, in 3.3 s, the
knee peaks at 116, the ankle's pitch at 92 and its roll at 53 (88 % of the
RS03's). The estimates the parts were chosen on put the knee and the hip's
roll at ~92 and the hip's turn at ~51: flown, the hip's roll is 52-63 and
its turn 24-37. Getting up keeps every leg part inside its ratings (peaks
to 73 %, RMS to 72 % over 2-3 s) and spends the arms whole: the upper
arm's twist and the elbow (RS02) at their 17 N*m peaks from every start,
and the shoulder's pitch and roll (RS03) at their 60 off its front and its
side. Nothing past a rating; the arms have no margin for a body heavier
than this one.

**Whether it can be served** (`humanoid_spike.py --speed`): the same walks
in the same house, each body through the Navigator with its own senses,
timed one process at a time; measured on the deploy box itself, pinned to a
core the live pair was not on (CPU 3; the live sim on CPU 1 kept its 77 %),
in the serving image:

| | real time, the walks | served |
|---|---|---|
| the quadruped pair | 1.67x | 1.40x (`serve.py --free-run`, `--pair --tasks --near-field`) |
| one humanoid | 3.70x | ~3.1x, estimated |
| the humanoid pair | 2.14x | ~1.75x, estimated |

The served column for the humanoid adds the lifecycle's own cost, read off
the quadruped pair (0.10 wall s a sim second for a pair, the near-field,
the jobs and the stream). **A humanoid pair cost less than the quadruped
pair**, twice the joints notwithstanding: its physics step was 15 % dearer
(0.31 ms against 0.27), but the quadruped ran its leg odometry every
physics step (500 Hz, the biggest of its Python costs) and cast more rays,
where the humanoid reckons at 100 Hz. One lever left: the humanoid's reach
reads both hands' forces at 100 Hz while it only walks. Seeing the ground
costs a fifth more: on the dev box one humanoid walking the same routes ran
at 2.7x real time blind and 2.3x with a walker that sees the ground,
interleaved twice (⚠ the map laid at the depth camera's 10 Hz, 1.8x: its
ray-cast frames are most of it, and at 5 Hz the frames still overlap at
1 m/s).

**What it cost**: RTX 5090s on Vast, $5.27 by Vast's charges -- $0.77 on
two hosts let go (one ran our tasks ten times slow, `pod.sh`'s DLPerf
filter since; one downloaded slowly), $2.37 on a Ryzen 9 9950X host (the
seeing walker beside the get-ups) and $2.12 on a Ryzen 5 9600X host (the
stairs walker, the last get-up, the gait variant) -- against the stage's
$16 likely. #507 to here: $8.45.

(`humanoid_spike.py --strip stairs`, `--strip fall` and `--strip getup`
draw `docs/figures/humanoid_stairs.png`, `humanoid_fall.png` and
`humanoid_getup.png`.)

## One walker that sees the ground (issue #507)

Ben's rule for serving the walker that sees the ground in walk2's place: it
tracks walk2's flat-floor commands as walk2 does, and beats it on rough
ground and the stairs. Before any run the rule was set as numbers
(`humanoid_spike.py --track`: each command walked three times from
standing, each started at another moment of the stance): on every command
the error in the asked direction at most walk2's plus 0.10, and what was
not asked -- sideways, forward or turning -- within 0.10.

**Narrow tracking, carried on from seeing7** (`Pluggy-Humanoid-Walk-Seeing-Fine`;
`training/README.md` has each round's flags). mjlab's tracking terms are
wide (0.5 m/s and rad/s): they pay 70 % for standing still asked to step
sideways at 0.3 m/s, and seeing7 stood still there. Narrow terms beside
them -- the planar velocity within 0.15 m/s, the turn within 0.25 rad/s,
then 0.10 -- and its command's curriculum gone: six rounds of 1500-2000
iterations, each fixing a row and costing another. Narrowing the turn's
width from 0.25 to 0.10 brought a turn asked 1.0 rad/s from 0.82 to 0.92;
raising the planar weight from 2 to 3 cured a 0.11 m/s curve at 1.3 m/s
(0.03) and, with the narrow turn, took that turn back to 0.65-0.68. Along
fine5a's run, of six checkpoints 100 iterations apart one passed every row,
one passed with two values on the bar, four missed one to three; the one
that passed (iteration 12500) is the walker, `models/humanoid_walk_seeing.npz`.
Chosen on the rule's walks, it was checked on three more a command, started
where none of the first three were:

| asked (forward, sideways m/s; turn rad/s) | walk2 | the bar, the asked part's error | the walker, the rule's walks | three more |
|---|---|---|---|---|
| 0.5, 1.0, 1.3 forward | 0.51, 0.99, 1.28 | 0.11, 0.11, 0.12 | 0.49, 1.02, 1.30 | 0.49, 1.02, 1.29 |
| 0.3, 0.5 back | 0.19, 0.46 | 0.21, 0.14 | 0.30, 0.62 | 0.29, 0.63 |
| 0.3 sideways | 0.31 | 0.11 | 0.27 | 0.31 |
| turns 0.5, -0.5, 1.0 | 0.21, -0.30, 1.01 | 0.39, 0.30, 0.11 | 0.40, -0.26, 0.92 | 0.40, -0.28, 0.78 |

What was not asked stayed within 0.09. The rule's table passes; the three
more fell once in 27, turning on the spot at 1 rad/s for six seconds.

**Every fall on a flat floor is its map's.** On the true ground
(`ideal_scan`) nothing fell in 54 walks, six a command -- and it backed off
at -0.67 m/s asked -0.5, past the bar's -0.64, where its own map's walks
were inside it (-0.62, -0.63): on that row the walker sits on the bar. On
its own map each of fine5a's checkpoints fell once or twice in 27 more
walks, at 1 m/s or in a long turn. Two mechanisms, both its soles' footprints (`HeadScan.feel`: a
planted sole's footprint goes into the map at the legs' reckoned height,
over the camera's cells):
- *A footprint a few centimetres off the camera's floor is a step there.* At
  1 m/s the planted-foot rules read it as a riser on a flat floor -- 105
  ticks of the stair rule, the tipped-sole rule moving its corner 155 times
  -- the height ran on, the map's scan came 0.23 m off the truth, and it
  fell; without the tipped-sole rule it walked on (but that rule keeps the
  height down a flight: 0.10 m off at its foot with it, 0.34 without).
- *A few centimetres it never saw in training.* Turning on the spot the
  map stayed within 2-4 cm of the truth for seven seconds and the walker
  fell at the eighth. Training's scan is noisy (0.1 m a cell, drawn afresh
  each step), never wrong the same way for seconds.

Without footprints it never fell on a flat floor -- and climbed none of 25
flights on the bare flight, up or down, every one a fall. Kept off the
camera's cells (the camera's floor winning), it climbed 6 of 25 up where it
had 25: up a flight a sole knows its tread better than a camera seeing it
end-on. So the footprints stay, and with them the falls, and **walk2 stays
the served walker**; this one replaces the rough task's walker and the
stairs' (git has rough4 and stairs1). Serving it is Ben's call on these
tables. What would close the falls is the map's, not the gait's: training
against a map wrong the ways this one is -- an offset, a plateau, a slope
that lasts -- or footprints that cannot disagree with the camera on one
floor. The back-off at 0.5 m/s, on the bar, is the gait's.

On rough ground (`--rough`, eight walks a course; falls, and the metres
made of 7.2 asked) it falls where walk2 cannot cross, as rough4 did, and
makes more ground than rough4 on the slope and the blocks:

| | flat | doorstep | paving | slope | blocks |
|---|---|---|---|---|---|
| walk2 | 0, 7.1 | 8, 1.9 | 1, 5.2 | 0, 1.1 | 7, 3.2 |
| rough4 | 0, 7.2 | 0, 7.2 | 0, 7.0 | 0, 5.9 | 0, 5.9 |
| the walker | 0, 7.2 | 0, 7.1 | 0, 7.1 | 1, 6.9 | 0, 7.0 |

On stairs (`--climb`): on the bare flight, five flights at each riser from
0.10 to 0.22 m, **25 of 25 up and 25 of 25 down on its own map** (and on
the ideal scan); in the house, the `stairs` scene from ten starts drawn
askew, **up 10 of 10** (no fall; 10 s, its belief 5 cm off) and **down 6
of 10**, with 7 falls its get-up took. The stairs specialist, flown again
with its camera seeing the floor: 14 and 22 of 25 on the bare flight, 7 and
7 of 10 in the house.

**Coming down is still the open half.** Every fall coming down the house's
flight but one came with the belief 0.7-1.9 m behind the body along the
hall: near the foot the walker runs at 0.8-1.3 m/s asked 0.4, and its
odometry counted 35-90 % of the way. By the simulator's contacts (our
diagnosis, never its senses): it follows a sole in the air 6-11 % of the
time, losing 0.4-0.6 m a descent and walking the belief back; no sole is
down 0-4 % of the time (0.1-0.3 m); a planted sole slides 0.1-0.3 m; and
once the scan matcher pulled the belief 1.1 m back -- the hall is a
corridor it cannot fix along. Its map, laid through that belief, shows
treads on the hall's floor, and it trips. Registering the depth frames
along the flight, not only in height, is the lever left.

**What it cost**: RTX 5090s on Vast, $3.70 by Vast's charges -- $0.58 on a
first host (fine1), $2.18 on an EPYC 9B14 host (the fine-tunes, the
checkpoints' export, every tracking table off the dev box) and $0.95 on a
Ryzen 7 7800X3D host (the other half of each round). #507 to here: $12.15.

(`humanoid_spike.py --view rough` and `--view stairs` fly it;
`--strip rough` and `--strip stairs` draw `docs/figures/humanoid_rough.png`
and `humanoid_stairs.png`.)

## The humanoid opens the chest (issue #493, stages 2 and 3)

The humanoid's reference (`evaluation/opener.py`) is the quadruped probe's
plan (#466's stage 2) on the humanoid's hands, built on `Body`'s members
alone: the knob off its tag, the box off the D435's depth at the part's 848
x 480 (`Body.depth_cloud`, `opener.guess`), the lid's arc planned about the
hinge it measured, the knob gripped and carried up the arc and back down
it, and let go. Every number is MuJoCo's, an upper bound on hardware.

![The reference on chest 0: set out, the lid past 57 deg, the call, and 5 s
on](figures/humanoid_opener.png)

(`MUJOCO_GL=egl uv run python scripts/attempts.py --body humanoid --strip 0
--out docs/figures/humanoid_opener.png`; `--view 0` watches it, `--chests
0-31 --jobs 4 --into DIR` flies the 32.)

**The counter is 0.72 m** (Ben, 2026-10-18; `evaluation.counter.COUNTER_H`).
At a kitchen's 0.90 the lid's 57 deg put the knob 0.52 m ahead and 1.21 m
up -- the arm's full stretch, however the hand and the handle turned
together -- and 62 deg was out of every set-out's reach. Nearer did not
help: 3 or 5 cm nearer the knob sat at the bottom of the colour camera's
view (19-61 deg below level), and with the robot bowing to look
(`Body.gaze_routine`) the reference opened 19 and 18 of 32 against 19
where it stands; a 0.80 m counter, 17.

**What made it work, each measured:**
- ⚠ *The drop handle's arm lies on the lid's front edge 57 deg up.* The
  handle rests 33 deg round its pin from the lid's own frame; held at its
  turn, the hand keeps the handle as it hangs while the lid turns under
  it, and at 57 deg of lid -- the grader's own line -- the arm meets the
  edge. Pulled on past it, the wrist gave 7-11 deg and the lid went on to
  74, or stopped short. So past 50 deg the hand turns as the lid does
  (`opener.FREE_RAD`): 8-12 of 32 became 19.
- *The robot leans back as it lifts.* The live pelvis stood 1.5-2 cm
  behind the statue the planner posed, and 2 cm more at the arc's top,
  where a path planned with no slack stopped at 57 deg. A path's points are
  reached 3.5 cm further ahead on a statue (`reach.PATH_MARGIN_M`), and
  planned for a hand that is out on the robot AS IT STANDS -- its model
  posed off its senses, its arm as its encoders read it -- with 2 cm on
  the points further from the shoulder (`LIVE_MARGIN_M`): from the statue,
  the path back down from the top solved 6 cm short of where the hand WAS.
- *One posture a path.* Changing posture partway (a crouch for the grip, a
  bow for the top) jerked the lid down at 1.7-2.3 rad/s and up to 81 deg:
  16 of 32 against 19. A reach given the path to come grips in the posture
  that follows it (`Body.reach_routine(path=)`), or, where that posture
  makes no approach, in any, and the hand holds still while the body takes
  the path's (`Reach._posture_held`): gripped level, a crouch had shuffled
  the feet 12 cm and no arc was followable from there.
- *A carry keeps the turn it was asked*, never the one it sagged to: the
  wrist's 0.1-0.15 rad of give at the top put the path back down out of
  every posture's reach. Its points are followed within 2 cm
  (`PATH_POS_TOL`): a solve trades position for the turn it holds.

**The reference opens 23 of 32** (95 % interval 0.55-0.84), every pass in
17.7-22.5 sim-s, the lid to 57.5-63.5 deg and down at 0.34-0.68 rad/s at
its fastest while open, no fall. The nine it does not are reach: in five
no grip's tilt follows the arc to 60 deg (`no tilt puts the lid's arc in
reach`), in three no arc follows from the grip the reach could make, in one
the hand ran out of reach a step up.

**The pull** that #493's practice flew -- a compliant straight move, its
turn held -- is rebuilt as the force move: "The force move (issue #525)".

**The arms' friction is calibrated once and kept**
(`humanoid/friction.py`, `models/humanoid_friction.json`;
`humanoid_spike.py --friction --write`): each arm joint swept in free air
at two speeds, the drivers' torque beyond the arm's weight fitted as dry
friction and viscous, the arm's weight where its encoders put it, and taken
off where a joint turns. The fit reads the sheets' 0.300 N*m at the
shoulder as 0.289-0.292, every joint within 4 %, and the viscous terms
within 0.09 of none; ⚠ swept at one speed, the two terms are one column,
and it read 1.78 dry and -4.97 viscous, right at 0.3 rad/s and nowhere
else.

What the robot's own mind made of these hands -- graded practice,
baseline (a′) and the flight over held-out chests -- is Evaluation.md
§4's.

## The force move (issue #525)

`pull` became a force move (docs/Hands.md §4): the command walks along the
way as the hand's FELT force allows, follows the hand across it, and its
turn gives to what it holds. Each rule was measured; every number is
MuJoCo's, an upper bound on hardware.

- *The force is closed on the reading, never on the arm's spring.* The
  compliant pull capped its force by how far the arm's model said it bent:
  on a drawer's stop it put 15 N for a 5 N limit and 27 for 10, where the
  hand's reading was within 0.5 N of the contact's true force. Closed on
  the reading (admittance), the stop came to 3.7, 8.9 and 18.3 N for 5, 10
  and 20.
- *Across the way the hand follows what it holds, no faster than twice the
  pull's speed* (`ACROSS_SPEED`). Unbounded, a lid let go of its weight
  followed it down; yielding only past a dead band of force, the hand
  lagged a drop handle's flip and ran out of reach on every lid (0 of 8).
  Holding its line (`give` 0) is for a set-down: giving, a cube set down
  slid 13 cm along the mat.
- *Its turn gives, but never about the jaws' line.* Held whole, the pads
  dragged a drop handle's knob sideways through its pin and the robot
  turned 22 deg on its feet; giving 10 % of the gap a command, it did not.
  Giving about every axis, a lid's weight bent the wrist about the jaws'
  line and the give followed: 70 deg down at 46-54 deg of lid, the elbow
  straight at its stop, the wrist pitched 80 deg, and the lid held there
  (4 chests of 8). Held about that line -- what a two-jaw grip holds turns
  about it between the pads -- the same single 27 cm pull took 7 of 8 past
  57 deg, and one pull toward the body set each down.
- *Going on, another posture from the arm as it is.* A replan that asked
  the posture it held from fresh seeds found another branch "reaching",
  and the live arm stood at its elbow's stop 2 cm short for 9 s. (A hand
  held takes none: "The force move against a held hand", #547.)
- *The grip's force is a verb's.* At the Dex1-1's 120 N a pad, the pads'
  torsional friction (condim 4, 0.01 m: up to 2.4 N*m) locked a door's
  lever in the jaws, and the hand gripping it held the door shut; at 15 N
  the tip turns between the jaws as the door swings.
- *A grip is judged on its squeeze's running mean* (`arms.SQUEEZE_S`, 0.1
  s, sampled every physics step). One reading carries the current sense's
  5.1 N of noise, and against half the force asked it read a 15 N grip as
  nothing held 7 % of the time and a 5 N one 31 %.
- *A grip lost ends the pull where the hand is* ("let go"). Ended at its
  distance alone, a knob slipped from the jaws would leave the hand
  pulling on through the air, its report that free motion's.

**The checks' props** (`activity/drawer.py`, `door.py`, `cube.py`; their
set-outs `evaluation/counter.py`, each placement's reason at its constant):
- *The cupboard's door is hung on its LEFT and its lever points away from
  the hinge.* Hung on its right, the door swung into the left arm that held
  its lever and stopped there, 36-45 deg open; its lever pointing toward
  the hinge, pressing it down rolled the left wrist toward its stop, and 6
  presses of 8 went 1-3 cm.
- *A trip latch and a kick.* One grip could not both hold the lever down
  and leave the door free (the jaws locked the tip), so the latch trips
  once the lever is pressed past 32 deg and stays tripped until the door
  has stood ajar and shut again, lever up; tripped, a spring kicks it 6 deg
  ajar, damped (undamped, 0.25 N*m swung a low-friction door 33-48 deg out
  of reach). ⚠ A door kicked free coasts: the hinge's friction is 0.02-0.15
  N*m, and in the check 2 doors of 8 stood past 50 deg after the press
  alone -- so a procedure looks before it grips again.
- *The cube is set out in sight and out of reach, 0.40-0.46 m ahead on a 6
  cm board, and the robot steps to it* ("Stepping toward the work").
  Standing put, its sight and its reach boxed a cube into 4 cm: nearer
  than 0.29 m the head camera did not see it (5 of 8), and from 0.34-0.38
  ahead, 12 cm up, the robot's model of itself refused the carry 6-10 cm
  ahead (6 of 8); a mat out to the left or nearer the body was out of
  sight even bowed 0.45 rad.
- ⚠ *Every set-out draws off a stream of its own* (`counter._drawn`, the
  stream (seed, k)). Off one seed with the cube's hidden draw, the
  furthest cube was always the heaviest and the leftmost the most
  slippery, so its weight could be read off where it stood.
- ⚠ *Our procedures were fitted to their own set-outs.* Redrawn off those
  streams, the drawer passed 3 of 8 where it had 7 and the cube 3 where it
  had 4 (the same code passes the old draws as before). On a drawer's way
  in to a knob 15 cm over the counter on a 2 cm stem, the jaws' pads met
  the cabinet's top or the counter, the reach blocked at 50-65 N; letting
  the arm find the grip's turn as a last try passed no more (4 of 8 on
  set-outs 8-15 either way).

## Stepping toward the work (issue #525)

`step(ahead, left, turn)` (docs/Hands.md §3) moves the feet: a turn on the
spot, one walk, both the walking policy's, then the standing policy's
settle (`HumanoidBody.step_routine`). Flown at the counter's set-out 0 and
read against the truth (`_true_stance`):

| asked | went | its report off by |
|---|---|---|
| 5 cm ahead | 9.6 cm | 0.4 cm |
| 12, 20, 30 cm ahead | 12.7, 23.5, 31.0 cm | 0.2, 0.8, 0.4 cm |
| 15 cm back | 11.6 cm | 0.1 cm |
| 15 cm left, 15 cm right | 14.4, 17.9 cm | 2.7, 0.8 cm |
| 20 ahead and 10 left | 20.8 and 7.1 cm | 1.1 cm |
| a turn of 17, 29, 57, 86 deg | 16.8, 27.0, 58.5, 85.1 deg | 0.2-0.3 deg; 2-6 cm (below) |

- *At the walker's slowest a short walk is a slide.* At 0.2 m/s
  (`mission.V_MIN`) the feet slid rather than lifted, and the odometry counted
  the first stride backward: asked 5 cm it went 16 and believed 5. Backing
  up at 0.3 it went 4 cm and believed it had gone FORWARD 1. A step walks
  at `STEP_V` 0.3 and backs up at `BACK_V` 0.4.
- *The stride carries on after the last command*, 0.2-0.27 s of its pace,
  so a walk stops once what it believes it has walked, plus its speed times
  `STEP_STOP_S` (0.22 s), reaches the step; the least a walk goes is a
  stride, about 10 cm.
- *A turn at 0.6 rad/s sprang back 6-8 deg* as the legs changed hands; at
  `TURN_W` 1.0 it lands within 2 deg of the turn asked, and its heading is
  read within 0.3.
- *Nothing is measured until the walker has the legs.* Out of a deep bow
  the pelvis came 16 cm forward over the feet as the arms came home, which
  a walk measured from before it counted as walked (ok, without a step),
  and the hand-over outlasted the 1 s a walk is given to get anywhere,
  which read as `blocked` on a bare floor.
- *The walker skews an oblique walk*: it raises a forward speed under
  `V_MIN` to it and leaves the sideways one, so a step at `STEP_V` more than
  48 deg off straight ahead walked 12 cm further ahead than asked. Such a
  step is walked as two, ahead then left (`body._step_walks`).
- ⚠ *A turn moves the feet a few centimetres nothing measures.* Its
  shuffle walks the stance 2-6 cm (turns of 17-86 deg), and the odometry is
  3-7 cm off over it: so the step takes a turn as on the spot (`turn` alone
  reports no travel), the nearer of the two.
- *Its odometry's swap.* A shuffle lifts its swing foot under a
  centimetre, so at a 1 cm threshold the odometry kept the sliding foot
  planted and read the pelvis going BACK as it slid forward: a short step
  lost 4.3 of its 13.6 cm. At 3 mm (`odometry.SWAP_M`) a forward step reads
  within a centimetre; `humanoid_spike.py --odometry` reads forward, back
  and turning as before, and sideways 7 % short where it read 21 %. ⚠ A
  short step to the LEFT still reads up to a fifth short (above).
- *Nearer than 0.29-0.31 m the head camera does not see a cube* on the
  counter's 6 cm board, even bowed 0.75 rad; from 0.3-0.5 m its spot is
  good to 0.4-0.8 cm. So a robot that steps in to a cube turns what it read
  before the step into the frame after it (`step.*`), as the check's
  procedure does, rather than look again.

**The cube in the hand.** The Dex1-1's fingers are 8 cm long and its grip
point is their middle (`humanoid/model.py`, `jaw_z`), so at the 60 deg tilt
that reaches a 26 mm cube on a board, the fingertips are 3.5 cm below the
grip point and meet the board while the grip is still 2-3 cm above the
cube's middle (1.8-2.8 cm, every set-out): the jaws pinch its top few
millimetres, and it tipped over or dropped as it was let go. The reach
still holds there -- its command arrived, and the fingertips press under
`BLOCKED_N` -- as do a lid's at set-out 1 (1.8 cm) and the door's press at
set-out 0 (2.3 cm), both of which open, so no tolerance on the hand's place
tells the cube's grip from theirs. Gripped shallower, 15-30 deg, so the
tips clear the board, the reach stops short on it (2 of 15).

## The way home (issue #539)

`withdraw` brings a hand out along its way in (the reach's 8 cm line),
then walks the arm home in joint space, and is ok once the arm is home.
Writing the Attempts page's guide (rooftop-media-2026 #441) found it wrong
twice, flown in the session world (reach to (0.25, -0.30, 1.30), lift 10
cm, withdraw):

- *It said home before the arm was.* It returned as the walk home began,
  at 4.47 s; the arm was home (each joint within 0.02 rad of its rest, by
  the sim) at 7.24 s, the next verb or the grader's hold under way
  meanwhile. It waits for `Reach.home` now -- the rest commanded and every
  joint read within `HOME_RAD` by its driver -- and returns at 7.56 s, the
  arm home at 7.57; `withdraw` is ok only then, else says why. Begun in a
  get-up, a release waits it out first (it spent its `RELEASE_S` on the arm
  the fall had let go of, and said "held"), and a fall while its posture
  comes back to the stand is "fell".
- *Straight home, a jaw came down on the counter's front edge* (0.28 m
  ahead, 0.72 m up) at chests 0, 36 and 63: it rested there 1.3 s at 0 and
  0.3 s at 63, and brushed it at 36; the left arm's mirror of the reach
  rested a jaw there 3.5 s. The arm was raised before its shoulder and out
  to its side, and one joint-space line home swept the hand down in front.

Every way home tried was scored on the robot's own drawing at the stand,
from the 125 starts the reach could plan of 540 asked: the pre-grasp for a
grip on a grid 0.25-0.55 m ahead, 0.45 m right to 0.30 m left and 0.8-1.4
m up, at three turns, the right arm, the stand's own posture (a bowed
reach's are not measured). Against a 0.72 m counter whose edge is 0.28 m
ahead (from the 30 starts over it), a 0.90 m one (26), and a shelf 10 cm
under the hand with its edge 8 cm behind it (30):

| the way home, from the line's end | into the 0.72 m | the 0.90 m | the shelf |
|---|---|---|---|
| straight in joint space (before) | 25, 4.7 cm deep | 23, 13.8 cm | 30, 26.7 cm |
| a line back over the toes at its height, then straight | 12 | 12 | 21 |
| the elbow folded first, then the shoulder, then the elbow | 30, 1.2 cm | 26 | 30 |
| lines back, across to its side and down, then straight | 0 | 0 | 12 |
| the arm rolled out as the rest come home, then lowered | 0 | 5 | 22 |
| **rolled out first, the rest home, then lowered (the side way)** | **0** | **0** | **9**, 8.5 cm |

- *Lines home through Cartesian space cannot be solved position-only.*
  From 8 of the 125 starts the solve along them jammed at a joint's limit
  (25 cm off its line) or jumped half a metre in a command; holding the
  grip's turn on them, it missed by over 2 cm from 72 of 77 starts on a
  coarser grid. A joint-space walk has neither failure.
- *Rolled out first, the arm cannot come down in front.* Held out at 1.3
  rad (`HOME_ROLL`) its elbow stays within 6 cm of its shoulder's height
  while its other joints come home, so its hand is no lower than a forearm
  under it; then only its roll turns, and lowered at its side the arm
  reaches 9 cm ahead of the soles' middle, behind a counter whose edge is
  further ahead. Rolled out WHILE the rest came home, the hand dipped
  before it was out.
- *It is a detour, so it is taken only where straight would dip*
  (`Reach.way_home`): judged on the same drawing as the robot stands,
  straight home unless some part of the arm goes more than `HOME_DIP_M`
  under the lowest of it now ahead of the robot's front -- where whatever
  the hand worked over is -- and the side way only where it dips less. 124
  of the 125 starts take it, a hand close in does not: 3.2 s home at
  `JOINT_RATE` (at most 4.2) where straight took 1.3.
- *The front is the toes, or the arm hanging at rest where that reaches
  further* (`front`): every way home ends there. Posed as a reach's statue
  the hanging grip is 0.8 cm past the toes, and judged from the toes alone
  straight home tied the side way there and took it down onto the counter.
- *An arm with none of it ahead of the front comes home straight*: it
  worked over nothing there. Judged as though all under it were solid, the
  capability check's drawer at set-out 3, pulled 17 cm out toward the
  robot, took the side way, whose roll-out swept a jaw into the drawer's
  face and pushed it 4 cm back in; straight, the hand drops 8 mm past the
  front, under the open drawer. ⚠ So from over its head the arm swings
  forward and down through whatever stands ahead: on the drawing, through a
  0.72 m counter 0.28 m ahead.

**Flown again** on the rule as it stands, against `9357d53`:

- The issue's procedure and four more, and the guide's examples at its
  chests, touch neither the counter nor the chest, and every `withdraw` is
  ok. The reference opener's verdicts are unchanged, 23 of the 32 public
  chests, each pass 3.0-4.1 s longer and chest 20's 9.1: its release waits
  for the arm. Its slowest pass is 31.6 s, where 22.5 s set the humanoid's
  limit (Evaluation.md §4).
- *The practice's failed holds came from it* (#493's and #535's records,
  1,181 tries). Of the 30 a touch during the hold failed, 21 never
  withdrew -- the jaws still on the lid at the call -- and 9 had. Flown
  again from their kept procedures, all 9 fail as recorded on the old way
  home, a jaw back on the lid's handle in the hold, and 5 pass on this one.
- *So did most of fixA's falls after its job* (#535's held-out tries,
  flown again from their kept procedures). Its 59 tries pass at the same
  21 and fall 15 times, not 36: all 24 that no longer fall had fallen after
  their last `withdraw` began, and the 3 new are its caught knob (below).
  Within 3 tries and without a fall it passes 16 of its 25 chests, not 6
  (Evaluation.md §4, "A fall was lost time"; since #544 a fall fails the
  try).
- *The door arms lost a pass each* (their 117 held-out tries). doorB falls
  in every try, its pull swinging the body; once it fell on its way home,
  `withdraw` says so now, and the procedure stopped there. doorA's steps
  back after its `withdraw`: begun while the arm was still on its way, the
  step had dragged the hand off the open door, and now the arm comes home
  first -- straight, as before -- and its jaw pushes the door from 63 to
  43 deg.
- The capability check (#525) is unchanged: the lid at 7 set-outs of 8,
  the drawer 3, the door 6 (set-outs 3 and 4 trading a pass: its
  procedure withdraws mid-job, and the reaches after begin from the arm
  home), the cube 3, no falls. Its cube's last `withdraw` now says "held"
  at 3 set-outs: the procedure steps in until the counter is 2-6 cm ahead
  of the soles' middle, where the arm's own rest runs into it, and the jaw
  comes down on its edge. The old walk home ended on the counter there
  too, or on the cube's mat, and said ok.

⚠ The side way assumes the space at the robot's side is clear, as
straight assumed the space under the hand was: its hand goes out to 0.74 m
from the body's middle, 1.0-1.1 m up. Nothing checks either against the
depth picture, which does not see beside the robot.

⚠ *A hand still caught after the line out is lifted by the side way.* In
3 of those 9 tries a jaw was left under a drop handle's knob, its grip
turned for the lid's arc: rolling out, the arm lifted the shut lid 36 deg
by it and let it fall, and the robot fell. Straight home it had pressed
the knob down (they fail either way). The force the arm's currents read
at the hand, held 0.15 s, tells a caught hand (60-105 N) from a free way
home (4-6 N), but not from a reach blocked on the counter whose press is
still on (31 N). Stopped there by the way in's own guard (`BLOCKED_N`),
on the line out and the walk home, 8 of the 9 passed, and a reach pressed
into the counter's top was held on it through the hold; on the walk home
alone, a short line out left the press on, and the capability check's lid
fell to 5 of 8. A guard that tells a press the motion relieves from a pull
it works against is not built.

## Holding the pull verb's 60 N at a hand (issue #544)

Ben decided (2026-10-25) that the standing policy be retrained to hold
what `pull` then allowed, 60 N at a hand, where it trained against 30 N a
component. Retrained as decided it holds no more, and four ways round it
did not either: what it would need is the force at its hand, and the
robot cannot read that.

**What a hand holds** (`humanoid_spike.py --pulls`): the left hand held by
the reach where the locked handle's bar is, 0.84 m up and 0.43 m ahead,
pulling each of 14 ways (named as the verb's way: "back" draws the hand
toward the robot) at 20-80 N, the force built at 90 N/s -- the verb's own
fastest build on that bar -- and held 3 s, twice. Staged with the truth: a
force at the hand, the policy's own measure. ⚠ The verb on the bar is not
the instrument: a `pull` takes the posture its way's END needs first, and
0.3 m down the body bowed 48 deg on a fixed bar and its grip rolled off;
on a short way it ends at its distance before the force builds, the body
giving. ⚠ And on a fixed bar the force move overshot its ask -- 104 N for
20 pulling up, as its body took postures with the hand on the bar -- until
#547 ("The force move against a held hand", below).
stand4, the served policy, stands to:

| pulling | back | forward | left | right | up | down | back-down | forward-down |
|---|---|---|---|---|---|---|---|---|
| stood to, N | 30 | 20 | 40 | 40 | 80 | 60 | 60 | 20 |
| fell at, N | 40 | 30 | 50 | 50 | -- | 70 | 70 | 30 |

(600 N summed over the 14 ways.) Its shoves, its envelope and `--hold`
(37.4 N, the grip slipping) are as stage 2 of #507 measured them.

**Retrained, four ways** (`Pluggy-Humanoid-Stand`, 4096 envs, 4000
iterations, 32-33 min on a rented 5090 each; the draw and its flags are in
git, bee276e to a8738f1, and the task trains stand4's again):

| run | what changed | steps an episode, its last 200 iterations | --pulls, summed |
|---|---|---|---|
| stand4 | (served: 30 N a component at both hands, pulls of 0.5-3 s) | 646 | 600 N |
| stand7 | a pull drawn from the ball of radius 60 N at one hand (`hand_pull`) | 235, the tilt lost | 590 N |
| stand8 | + its tipping moment capped at 60 N*m (`moment-nm`) | 246 | 580 N |
| stand9 | + held up to 10 s (`longest-s`) | 235 | 590 N: back 40, forward 0 |
| stand10 | + brought in over 0.5 s (`ramp-s`) | 256 | 540 N |
| stand11 | + a fall charged, about 4.4 s of pay (`fell`) | 280 | 600 N |

**Why none holds more.** In mjlab's own physics too, stand4 and stand7 fall
every time to 40 N held at the trunk and to 60 N at a hand hanging at rest:
no gap between the two physics. That hanging hand's 60 N needs the centre
of pressure 9 cm ahead of the centre of mass, inside the sole; traced, it
goes to the toes as the force builds and stays, the centre of mass drifts
forward with the pull and the body tips -- neither the hips nor a step are
used. A policy shown its trunk and its joints feels a pull only once the
body has moved, and by then the toes are spent; trained against frequent
pulls, it learns to stand back instead: 1.5-2 cm further back over its
feet (its centre of mass 1.4 cm), which pulls 40 N back and falls to 20
pushing, and puts a handle planned with 1.5 cm of slack out of the reach
-- stand7's grip came off the locked handle at the first centimetre,
stand8's never closed on it.

**Shown its hand's force, it holds -- where the force is true.** stand12
(`Pluggy-Humanoid-Stand-Felt`, git 34f0ed3: each hand's force in the
trunk's frame through 8 N of noise, beside stand10's flags; 413 steps an
episode) fell
to none of 40 or 60 N at a hand, any way, in mjlab's physics, where stand7
fell every time at 60; stand13, the fall charged too (862 steps), 0-14 %
at 60. But the robot's reading of
that force -- its arm's six drivers' torques through the arm's Jacobian
(`SelfModel.hand_force`) -- is about 20 N off with nothing applied at a
0.35 m reach (the still joints' friction, taken off only where a joint
turns), and along a stretched arm it sees little: pushed 40 N forward at
the bar, it read 8.5 N forward and 17 N up. Shown that, stand12 held no
more than the blind ones (550 N summed; up 30, where stand4 holds 80).
The served wiring was built, tested and reverted (git: 34101aa).

![The `pull` scene on stand4: the left hand held where a handle would be,
pulling at the verb's most, 20 N -- back, pushing away, left, right, up
and down, the force staged on the hand -- the body standing
throughout](figures/humanoid_pull.png)

(`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --strip pull --out
docs/figures/humanoid_pull.png`; `--view pull` in the viewer.)

**What is true now:** the served standing policy is still stand4, a hand
at a counter's height holds 20-40 N across the floor against it for 3 s,
and `pull` asks at most 20 N (`steps.PULL_MOST_N`, Ben, 2026-10-25), its
text telling the robot that pushing away holds around two thirds of that
-- a margin under the 20 N pushing held. Our own procedures do every hands
job with 4-20 N. Holding more wants a hand force the robot can read and a
policy can be shown: the whole body's reading (#547) reads it, but it is
the legs' torques, the policy's own commands, and stand12 shown it fell at
20 N ("A hand's force off the whole body", below); a sensor at the wrist
(Parts.md, "The humanoid body") stays the part to add.

## The force move against a held hand (issue #547)

`pull` makes its force by walking its command ahead of the hand, and on
#544's locked bar it put far more than its ask on what held it.
`humanoid_spike.py --bar` is the measure: the left hand shut on the welded
drawer's handle (0.84 m up, 0.43 m ahead) and `pull`ed 5 cm each of the 14
`PULL_WAYS` at 10 and 20 N, a fresh robot a trial, the force on the bar
the pads' (ours) beside what the hand read:

| on the locked bar | before | after |
|---|---|---|
| the hardest 0.1 s mean on the bar, asked 10 N | 35 N (down; forward 34, forward-down 33, up 30) | 24 N (back, up) |
| asked 20 N | 103 N (up, until the grip let go) | 36 N (up) |
| pulls that never ended in 6 s, of 14 | 7 at 10 N, 7 at 20 | none at 10, 3 at 20 (grips sliding at their force) |
| postures the body took with the hand on the bar | up to 7 | the pull's first; one way, at 20 N, 3 more |

**Why.** The move's postures were written for a hand that moves (#493,
#525): the one its way's end needs, taken first, and another wherever its
solve missed the command. Against a hand held, the command's lead IS the
miss. Pulling up 20 N, the command got 4 cm ahead, the solve missed, and
the body stood up 8 cm with the hand on the bar: 100 N on it for 1.5 s,
while the loop -- frozen while a posture comes -- read 290 N and did not
back off. Pushing forward or down, it bowed and crouched through five or
six postures, to a 54 deg bow, and never ended.

**Now** (`Reach._pulled`): a hand that is not going -- 1.5 cm along its
way in a second -- whose command is out of the arm's reach where the hand
itself is not, takes no other posture, and its command stays where the arm
last reached; pinned there, going nowhere along its way for `FAST_S`, the
pull ends held fast (out of reach pushing light, and backing off past its
force since: "The force loop"). While the body takes a posture, the
command still backs off past its force. A hand the arm cannot hold where
it is takes a posture as before -- a lid's knob gripped 15-18 mm past the
arm's reach (its reach stopped short) opened only from another, before the
lid had moved -- and so does a hand that goes. No lead tells the two
apart: moving lids took theirs with the command up to 17 mm ahead, and a
push on the bar six with it under 15.

⚠ Tried and dropped, each against the hands check flown beside it:
- *A creep* -- the command closed on at 2 cm/s while the hand did not go:
  it cut the back's overshoot by a quarter, but its slower start ended
  the cube's lift at set-out 0 a few millimetres lower, and the body fell.
- *The first posture deferred until the hand moved, and a held hand
  pinned wherever its command missed*: a door's lever, yielding to its
  spring too slowly to count as going, ended its press at 2-3 N, and lids
  whose knobs were gripped past the arm's reach were pinned before they
  could move -- the door fell from 6 of 8 to 4, the lid from 7 to 3.
- *Going as the hand's motion any way*: the sensed hand moved 4 mm in a
  tenth of a second and the grip slid 3-7 mm a second on the bar, so a
  held hand read as going and the postures came back. Along its way, a
  grip sliding as a push leans the body in still reads 10-12 mm a second;
  1 cm a second let the postures through, and so going is 1.5.
- *Held fast off its way alone*: a door swung by its hinge moves the hand
  across its way while its reading is high, and was let go of. Held fast
  at its force stays off the hand's place (`FAST_M`); only a pinned pull
  is read along its way.

**What was left: the arm cannot feel along itself.** A load along a nearly
straight arm goes through its structure, not its motors, and the arm's
reading missed most of it (the next section has the bench); the hand now
reads its force off the whole body, and what is left of the overshoot is
the loop's own (the next section). ⚠ And a grip that rolls
on the bar as the pull builds still takes a posture: in the `bar` scene's
pull up, from a second grip, the hand rolled 6.6 cm round the handle, its
turn left the command's, the arm read as unable to hold it where it was,
and the body stood up into the bar -- 91 N. Pinning a hand the arm reached
as its pull began kept that posture and cost the lid three set-outs of
eight, so it stays.

**Against the procedures** -- the move before and after, flown on one
machine (`hands_check.py --jobs 4`; #535's references, the opener on 32
chests and the door on its 32 held-out set-outs):

| flown on one machine | the move before | after |
|---|---|---|
| the hands check: lid, drawer, door, cube, 8 set-outs each | 7, 3, 6, 3 | 7, 3, 7, 3 |
| the opener (#535's reference, 32 chests) | 23 of 32 | 23 of 32, the same chests |
| the door on its 32 held-out set-outs | 19 of 32, 1 fall | 19 of 32 (two set-outs trading places), 2 falls |

The second fall is a press that began with the hand out of the arm's
reach, which the move leaves as it was: there the command now backs off
past its force while a posture comes, its grip slipped off the lever 6 cm
lower than the old move's did, and the way home from there fell.

(The `bar` scene's filmstrip, as the loop flies it now, is in "The force
loop".)

**What is true now:** against something that holds the hand, a pull takes
no posture but the first -- unless its grip rolls off the turn it was
given -- and ends held fast, or out of reach pushing under `FAST_SHARE` of
its ask; what it puts on what it holds is its loop's ("The force loop"),
and the grip slides on a bar at 20 N.

## A hand's force off the whole body (issue #547, stage 2)

A load along a nearly straight arm goes through its structure, not its
motors, so the arm's torques cannot read it; but whatever the world pushes
on the robot other than at its soles comes off the ground under them, and
standing, the legs' drivers read that. `SelfModel.felt` reads a hand's
force off the WHOLE body's: the robot's balance standing still -- its
weight against what every motor pushes with, G = Sᵀτ + ΣJᵀw -- solved in
least squares for each sole's wrench and the hand's force and twist about
its grip, over the trunk's six rows (no motor pushes there), each leg's
joints and the working arm's. Each row is weighed by what it cannot know:
a still joint's friction (the arms' as calibrated, the legs' their
drawing's 0.3-0.6 N·m), and on the trunk's rows the motion a balance taken as
still leaves out (2 N, 1 N·m). A joint whose encoder reads it on its stop
is left out, the hand's twist is taken as about 1 N·m, and a balance whose
rows will not agree reads nothing (`CLOSES`). The reach reads
it for a hand at work -- on its way in, holding, carried, pulling, stopped
or on its way out -- while the standing policy has the legs, which the
mission says every step (`Reach.standing`, off until it is said: the
spike's walkers step a reach too); otherwise, a hand moved joint by joint
or at rest included, and walking (a sole in the air), the arm's own
(`Reach.arm_force`). A read costs 0.16 ms, the arm's 0.07.

**The bench** (`humanoid_spike.py --feel`): 20 N staged at the grip, the
left hand held by the reach where #544's locked bar is (0.84 m up, 0.43 m
ahead), along each of the 14 ways; along the way and across it over the
hold's last second, and the worst along it from the build's start (0.25
s); still, and swaying -- the trunk's pitch ±0.1 rad at 0.4 Hz, the pelvis
4 cm down and up at 0.25 Hz:

| the hand's reading | still: along, N | across | worst | swaying: along | across | worst |
|---|---|---|---|---|---|---|
| its arm's own | 2.4-20.5 (forward 5.4, forward-down 2.4, back-up 9.1, up 12.3) | to 9.8 | to 19.8 | 1.5-20.1 | to 9.9 | to 25.0 |
| the whole body's | 19.7-20.2, every way | to 1.0 | 3.6-16.2 | 18.6-20.9 | to 2.1 | 3.7-15.8 |

What each piece is for, each measured on the same bench:
- *The legs alone* -- the soles' reaction less the weight -- read 7.7 N
  down with nothing at the hand: the still knees' and hips' friction holds
  part of the weight, and a vertical force has a poor lever on a nearly
  straight leg. The balance's moment rows and the arm's rows take it to
  0.3.
- *A joint on its stop*: pushed away and up, the arm straightened onto its
  elbow's stop, which held 3.1 N·m its motor never saw, and read in the
  hand read 11 N for 20.
- *The twist*: taken as a force alone, the hand read best on the bench
  (rms 2.0 N) and on the bar, but grips twist what they hold -- 0.7-1.7
  N·m on average on the hands check's, under 4 in 19 readings of 20 -- and
  read the jobs 1.3-3 times worse (below); left free, it read 180 N for an
  instant on the bar.
- *A balance that does not close*: dragged onto a sole's edge by
  `--hold`'s hardest pull (a sole 28 deg off flat, the grip slipping), the
  balance read the hand's pull 60 N the wrong way while the arm read the
  truth. What its rows miss by, weighed and shared over its spare rows, was
  0.6 at the median and 28 at the 99th percentile over the bench, the bar
  and the jobs, and 225-727 there; past 50 it reads nothing and the arm's
  stands (0.7 % of the jobs' readings, where the arm read truer).
- *The body's motion*: the worst from the build's start is it -- the
  body's centre of mass sped up by the push before the feet push back, up
  to 9 N pulling back. With nothing at the hand, a posture change's first
  second reads 6-8 N (the arm 1.5-2.4), 0.8-1.1 two seconds on. A
  dynamics estimate off the robot's own senses (the joints' speeds, the
  gyro, the planted soles) halved that, to 3-4.5 N, adding noise standing
  still (1.1-1.4), and is left out: while a posture comes, a pull's
  command only backs off and an approach has not begun.

**On the jobs** (`hands_check.py` flown as it stands, 32 flights, the
readings computed in shadow beside the truth -- every contact on the hand
from outside the robot; rms error in 3-D against the hand's true 0.1 s
mean force while it held something):

| the hand's reading | lid | drawer | door | cube | the bar, 10 / 20 N |
|---|---|---|---|---|---|
| its arm's own | 7.4 | 7.7 | 5.0 | 27.7 | 6.1 / 7.7 |
| the whole body's | 4.1 | 5.2 | 2.6 | 4.7 | 3.0 / 4.0 |
| ...a force alone, no twist | 6.0 | 6.9 | 4.4 | 14.1 | 2.3 / 2.9 |

⚠ Everything the world pushes on but the soles reads as the hands'. Staged
on the robot's model of itself, 20 N sideways on the resting forearm read
as 6.8 N at the working hand and 20 N on a thigh as 5.0 (the working arm's
own rows outvote most of it: 1 kg in the resting hand read 0.8); on the
cube's job, in the 4 % of its readings where some other part of the robot
touched the world, the whole body read 5.9 N rms off, the arm 4.1. ⚠ And
in free air a long pull's hardest reading rides higher: pulling up 0.3 m
at 4 N with nothing held, `pull.force` read 3.6 N off the whole body
where the arm's read 2.1 -- near a procedure's "it pressed" of 3 N (the
cube's set-down, 0.1 m, read 2.4 either way).

**On the locked bar** (`--bar`, the hardest 0.1 s mean on the bar), the
force move closed on it: at 10 N asked 23.7 → 21.8 N, at 20 N 36.1 →
33.5. What the hand read now is what the bar took (pulling up at 20 N:
32.1 read, 33.5 on the bar, where the arm read 21.6 of 36.1). ⚠ The rest
was the loop's: closed on the bar's true force (an oracle, ours), the move
put 17.0 N on it for 10 asked and 34.8 for 20 ("The force loop", next).

**A standing policy shown it falls.** stand12 (#544's
`Pluggy-Humanoid-Stand-Felt`, trained shown its hand's true force) shown
the arm's reading held 550 N summed over the 14 ways of `--pulls`; shown
the whole body's, it fell at 20 N on every way. That reading is the legs'
torques, which are the policy's own commands: what it is shown moves as it
moves. A felt policy would have to be trained with that reading in its
loop.

**Against the procedures** -- the old reading and the new, flown on one
machine (#535's references beside the hands check):

| flown on one machine | the arm's reading | the whole body's |
|---|---|---|
| the hands check: lid, drawer, door, cube, 8 set-outs each | 7, 3, 7, 3 | 7, 3, 7, 4 (the cube's set-out 6) |
| the opener (#535's reference, 32 chests) | 23 of 32 | 23 of 32, the same chests |
| the door on its 32 held-out set-outs | 19 of 32, 2 falls | 21 of 32, none lost, 3 falls |

The third fall is at a set-out both fail: the hand's way in pushed the
lever 30-66 N, which both readings read (22-62 N); the old stopped that
reach short, the new one's dipped under `BLOCKED_N` before `BLOCKED_S`
was out, so the hand went on, its jaws shut on nothing six times, and the
last way home fell.

![The `feel` scene: the left hand held out where the bar would be, 20 N
staged at its grip pulling back, then pushing away, away and down, and up,
what its arm read of each and what the whole body read](figures/humanoid_feel.png)

(`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --strip feel --out
docs/figures/humanoid_feel.png`; `--view feel` in the viewer.)

**What is true now:** a hand at work, standing, reads its force off the
whole body -- within a newton every way on the bench, the ways a straight
arm cannot feel included, and 2.6-5.2 N rms on the jobs -- and the force
move closes on it ("The force loop", next).

## The force loop (issue #547, stage 3)

Stage 2 left the move closed on a reading that sees along the arm, and a
locked bar still taking 1.7-2.2 times its ask -- 1.7 closed on the bar's
true force, so what was left was the loop's own. `humanoid_spike.py --bar`
is the measure, and `--bar --oracle` closes the move on the bar's true
force instead (`closed_on_the_bar`, ours): the loop's floor, what it does
with a perfect reading.

| on the locked bar: the hardest 0.1 s mean, the ways of 14 over a quarter more | the loop before | after |
|---|---|---|
| closed on the hand's reading, asked 10 N | 21.8 N (up); 8 | 13.9 (up); 1 |
| ...asked 20 N | 33.5 (up); 5 | 32.7 (up); 1 |
| closed on the bar's true force, asked 10 N | 17.0 (forward); 8 | 11.5 (back-right); none |
| ...asked 20 N | 34.8 (up); 4 | 23.7 (up); none |

Nothing fell, before or after, and the postures taken with the hand on the
bar were 6 and 9 where they were 6 and 10.

**Why it overshot**, each command traced against the force the pads put on
the bar:
- *It built faster than it read.* At 6 cm/s into the arm's spring (2.1
  kN/m pulling back, the arm's gains through its Jacobian) the force rose
  over 100 N/s, and the reading, smoothed over 0.1 s, ran about 0.1 s
  behind: it read under half its ask with the ask on the bar, and walked
  on.
- *A held hand could not back off.* Pinned where its arm last reached
  (stage 1), every command put its command back where it was, a back-off
  too, and froze its arm's joints: pulling up at 20 N the bar took 21-24 N
  for half a second while the loop asked to back off and the body leaned on
  the hand.
- *It backed off no faster than it walked in.*

**Now** (`Reach._pulled`): the loop reads the hand over 0.03 s
(`ADMIT_SMOOTH_S`) and closes on that reading carried 0.1 s ahead at the
rate it rises (`AHEAD_S`, the rate over `RISE_SMOOTH_S` 0.08 s): it walks
while the force a moment ahead is under half its ask, slows as it nears it,
and backs off past it at up to twice its speed (`BACK_SPEED`). A hand going
along its way is only slowed by where its force is going, and backs off past
its force as read, at its speed: the whole body's reading rides 4-5 N on a
hand moving in free air, and backed off on where that went, a 4 N pull up
0.3 m stopped after 1.1 cm (the old loop went the 30 cm; going, it does in
8.1 s, and the bar's numbers are unmoved: no held hand goes). ⚠ A hand that
meets a stop reads as going for up to `GO_S` (1.5 cm in a second), and backs
off there only on the force as read: no measure here meets a stop at speed.
A held hand
pinned where its arm last reached backs off a share of the way toward the
hand, its arm's joints as far toward those that reach it, none faster than
`JOINT_RATE` (that solve begins where a missed one left the arm, so at a
share of 1 a setpoint would be written across a gap). What the pull
reports (`pull.force`) and whether it is held fast stay on the 0.1 s mean
(`MEAN_S`): the cube's set-down asks whether it pressed by it.

⚠ Tried and dropped, each over the bar's 28 pulls on both readings:
- *less smoothing alone* (0.03 s): 14.4 and 30.1 N closed on the truth;
- *a cap on the force's rise* (3 or 5 times the ask a second): 14.4 and
  29.3-32.9 -- against a held hand the force rose with the command stood;
- *the look-ahead braking but never backing off*, the back-off on the
  force as read now or over 0.1 s: 14.0-15.8 and 26.1-29.6;
- *stage 1's rule kept while a posture comes* (stand, and back off over 0.1
  s at walking speed): 16.5 for 10 on the reading;
- *a softer arm* (`PULL_STIFFNESS` 0.4 and 0.3): pulling up at 20 N within
  a quarter on the reading (25.7, 23.5), but 14.6-15.2 for 10;
- *the arm's solve begun from its arm as sensed*: no gain (30.4 for 20);
  its redundancy projected out of the joints it commands: three times the
  ask;
- *the rise read over 0.03 s*: the same peaks, but it held 0.85 and 0.71 of
  its ask (below).

**Pulling up, the reading lags the bar.** On the robot's own reading the
bar still takes 13.9 N for 10 pulling up and 32.7 for 20, where closed on
the truth it takes 11.4 and 23.7. Pulled up, the grip turns on the bar
about its jaws' line (3 deg each way) and the arm's commanded
joints swing 0.3 rad from those it senses, about the nearly straight arm's
shoulder-yaw and wrist-roll redundancy; meanwhile the whole body's reading
runs 2-3 N under the truth at 10 N and up to 10 N under for 0.2 s at 20 N.
That is the reading's or the grip's to fix: a loop holds what it reads.

**It holds a little under its ask.** On a contact that jitters -- pulled
back, the bar's force swung 1-18 N from one command to the next -- a loop
that backs off faster than it walks in sits under its ask: the bar's mean
force from 1 s on, the median over its 14 ways, was 0.90 of 10 N where the
old loop's was 0.94, and 0.77 of 20 where it was 0.85 (the ways whose arm's
reach ends short of the ask included). 6 of the 28 pulls never ended in 6 s,
as 6 did, each a grip sliding 18-39 mm along the bar.

**A pull past the arm's reach** ended held fast: stage 1's pin took a
command the arm cannot reach, with the hand not going, as held. Now a pull
pinned past its arm's reach is held fast only pushing `FAST_SHARE` of its
ask, else it is out of reach. Pulled 0.35 m down at 4 N from the bar's
place with nothing held, it ends out of reach after 1.3 cm (the old said
held fast, after 1.1). From there, every way at 4 and 10 N for 0.2 m: the 8
of 14 that went the whole way still do, and those whose arm ran out stop
where they did, but for forward-left at 4 N (9 cm, where the old went 14),
saying out of reach where they said held fast. On the bar, the pushes the
arm's reach ends at 1-20 N end so too: 4 ways of 14 at 10 N, 6 at 20.

**Against the procedures** -- the loop before and after, flown on one
machine:

| flown on one machine | the loop before | after |
|---|---|---|
| the hands check: lid, drawer, door, cube, 8 set-outs each | 7, 3, 7, 4 | 6, 3, 7, 4 (the lid's set-out 1) |
| the opener (#535's reference, 32 chests) | 23 of 32 | 23 of 32, the same chests |
| the door on its 32 held-out set-outs | 21 of 32, 3 falls | 18 of 32, 5 falls, none gained |
| ...and on 32 more (seed 547) | 18 of 32, 4 falls | 17 of 32, 4 falls, one gained |
| `--hold`, a locked handle pulled harder until it gives | 37.4 N, the grip slipping | the same, step for step (it carries, never pulls) |

Where a verdict flipped, it flips again with either loop's constants nudged
2-6 %: the lid's set-out 1 -- opened to 75 deg, and its grip gone as the
lowering began -- passes at four nudges of four; of the door's three lost
on the first seed, the new loop passes 48491512 at four nudges of four and
645308331 at three, and the old loop, nudged, fails 48491512 and 980759840
at two of four. All seven of the door's falls before are falls still, on
the same set-outs; the two more are 48491512 and 645308331. Over its 64
set-outs the door's three pulls go as far as they did (medians within 3
mm) and peak 0.2-0.7 N lower: the overshoot gone. Flown with the held
back-off's `JOINT_RATE` bound and without it, every flight above landed the
same: none stepped a joint 10 mrad a command there (the door flight traced,
at most 2).

![The `bar` scene: the left hand shut on the locked handle and pulled at
`pull`'s most, 20 N, each from a fresh grip -- back (no end in 6 s, the bar
taking 22 N at most), up (32 N, the grip turning on the bar and the body
taking two postures) and forward (out of reach at 14 N); the fourth grip,
for down, shut on nothing](figures/humanoid_bar.png)

(`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --strip bar --out
docs/figures/humanoid_bar.png`; `--view bar` in the viewer.)

**What is true now:** the force move slows before its ask and backs off
past it at twice its speed; against a held hand, closed on the bar's true
force, it keeps within a quarter of its ask every way (at most 1.2 times),
and on the robot's own reading every way but up, where the reading lags a
turning grip (1.4-1.6 times); once built it holds 0.8-0.9 of its ask, and
pinned past its arm's reach pushing light, it says out of reach.

## A move that stops on a reading (issue #554, stage 1)

A procedure reacts between verbs, about twice a second, and a verb runs to
its own ends; #544's lid arms lost the lid inside one pull, past the drop
handle's stop. `reach`, `move_hand` and `pull` may now each stop on one
reading (docs/Hands.md §3's end): `steps._guarded_routine` drives the
body's routine as `yield from` would, reads the sensor once the routine has
begun its move and then once each `Sensor.every_s` -- a hand's place, force
and grip, and the clock, once a command of its arm's -- and on the first
reading past its number asks `Body.stop_hand`, which stops the move where
it is (`Reach.halt`): a carry or a pull through the stop its patience
already used (`Reach.stop`: held where the hand IS), a reach on its line in
as one stopped on its way (`blocked`), a reach walking out in joint space
at its joints' command and the body in the posture it had got to, out at
the grip they put it at. Until a move is under way, the stop answers no
and the guard reads again at its rate: a reach's first steps wait out a
get-up, and while the get-up policy has the legs nothing is stopped. A
reading's rate is rounded UP to physics steps, so a part is never read
oftener than it reads (33 ms is 17 steps of 2 ms, not 16). A hand's place
is read as its arm's loop last read it (`Body.hand_sensed`,
`Reach.sensed_grip`), never afresh: each fresh reading drew on the drivers' noise,
and a move whose condition never came true ended 2.4 mrad from the same
move with none; read so, the two are one flight, bit for bit.

**What its review found** (four of five from the humanoid flown): a reach
stopped as its arm swung out held its joints while the body went on into
the crouch its plan had asked, the hand riding 17 cm down 2 s on, and the
withdraw after, its line starting at the command the body had left,
stepped a joint 1.43 rad in one command (unstopped, a reach and its
withdraw step at most 0.078); holding the posture the body had got to, the
hand stays within 3 cm and the withdraw steps at most 0.034. During a
get-up the reach a fall let go of still said its hand moved, so a stop
there used up a condition the stand-up's fresh reach then ran without. A
pull stopped on its grip's word (`left.held`) was relabelled "let go". And
four figures said `time read 1.15 (above 1.15)` of a reading past 1.15; it
is said to as many as tell them apart.

**Flown, the humanoid on a bare floor** (`tests/test_guard.py`'s flight):
a carry up at 6 cm/s told to stop past 1.15 m read 1.151 and stopped there,
the grip truly at 1.150 and 1.150 a second on; a pull up at 10 N told to
stop past 1.2 read 1.201, truly 1.1998, and 1.2006 a second on. A
reading's lag is one command (10 ms, 0.6 mm at 6 cm/s) and the stop's
another.

**What a moving hand's force reads.** Carried 30 cm across in free air,
the right hand's whole-body force read 11-17 N AHEAD in its first half
second -- the balance takes the body as still, and the start of a move is
not -- then under 6 N ahead, 4.4 N across and 2.7 N up for the rest; a
carry up read 17 N ahead at its start, and a reach's approach in free air
10.2 N up as it began. A stop condition on a force from a move's start
reads that start: the prompt's example stops at 10 N across, over the 4.4
its own carry read that way.

**The scene** (`scripts/hands_check.py --view until K`): our lid procedure
with its lift stopped on the hand's height 10 cm up, held two seconds,
then lifted the rest of the way (`0.27 - read("pull.moved")`) and set down:
set-outs 0, 1 and 3 passed, the lid set down from 61-63 deg. Lifted the
whole 0.27 m again after the stop, it opened to 94 deg and stayed open.

![The `until` scene at set-out 1: the lid's lift stopped on its hand's
height, the lid held part open, then lifted on and set
down](figures/hands_until.png)

(`MUJOCO_GL=egl uv run python scripts/hands_check.py --strip until 1 --out
docs/figures/hands_until.png`; `--view until 1` in the viewer.)

**What is true now:** a hand's move stops within a command of its reading
crossing, and holds there; a reading a move changes is read at its part's
rate and no oftener, from the move's first command; a hand's force read
from a move's start reads that start.

## A tag watched while moving (issue #554, stage 2)

`watch(tag)` (docs/Hands.md §3) keeps the head's colour camera reading one
tag while a procedure goes on: `perception.watch.TagWatch` takes a picture
every 34 physics steps through `spot`'s own decode
(`perception.tags.TagDetector` at 1280 x 720), and each picture's reading is the robot's 50 steps (0.1 s)
later.

**Measured first: what a picture costs** (the counter world, this box):
the render 1.5 ms on EGL and 12 under osmesa, the decode 12.8 (pupil
AprilTag at full resolution, two threads), beside the humanoid's own 0.26 s
of wall a sim-second standing and 0.22 reaching. So 15 pictures a second
add 0.21 s a sim-second on EGL and 0.37 under osmesa: a flight near doubles
while a watch runs. The deploy box renders slower (a walking look is 32 ms
there, "Places, not coordinates"), so a person's session runs slower than real time
while a watch runs. 15 is the D435 colour imager's 15 fps mode, and half
what a real robot decodes: PhotonVision's docs give a Raspberry Pi 5 two
AprilTag streams at 1280 x 800, 30 fps. No Pi 5 latency is published; 0.1 s
is a picture's capture at 15 fps and a decode at the Pi 5's rate.

**Placed without moving the flight.** Each picture is placed through the
robot's model of itself as its arm's loop last posed it (`Reach._update`,
at most 10 ms old). Posed afresh, as `spot` does, every picture drew on the
encoders' shared noise, and a watch nothing read changed the flight.
Placed so, a flight whose pictures show the tag and one whose pictures show
nothing are one flight, bit for bit (`tests/test_watch.py`). On one picture
the two placements differ by 0.4 +- 0.9 mm. A picture taken before a
stand-up's fresh reach has posed its model shows nothing, as does every
picture while the get-up policy has the body.

**Standing still,** every picture showed the knob's tag at set-outs 0-3,
its place within 2-4 mm (one standard deviation) picture to picture -- and
drifting. ⚠ The humanoid turns in place after its set-out's settle: 49 mrad
in 6 s at the counter, slowing, its stance belief within 3 mrad of the
truth. The tag's place in the stance frame moved 15 mm with it, so a
`spot` read at the start is that stale by then. The watch follows it.

**What it sees in our lid procedure** (`hands_check.LID` watching from its
spot on): the knob's tag was in view through the reach and the grip, 98-100
% of the time at set-outs 0-2. Through the lift and the lowering at set-out
1 it was in view 62 % of the time: always under 10 deg of lid, 86 % at
10-20, about half at 20-50, never past 50. The camera is fixed on the
trunk, and the knob rises out of its view and behind the hand. At set-out
0 the jaws hid it for the whole of both pulls. So no watch sees the knob at
the drop handle's 57 deg stop, where #544's fresh arms lost the lid.

**What it flatters** (`attempts.WATCH_FLATTERS`): a render is sharp. The
knob's tag crossed the picture at a median 63-242 px/s in a reach and 108
in a lift, 270-440 at the 90th percentile, and up to 1,060 on a fast arm
move. On a tag 33 px across at 0.6 m (a cell 4 px), 270 px/s is a 2.7 px
smear in a 10 ms exposure, which a real camera's picture has and the
render does not.

**The scene** (`scripts/hands_check.py --view watch K`): our lid
procedure watching the knob, its lift stopped where the watch puts the knob
6 cm higher than as it was gripped, held 2 s, then lifted on and set down.
At set-out 1 it stopped 6.9 s in and passed, the tag out of view by 10 s.
At set-out 0 its own jaws hid the tag, the condition never read past, and
the lift went its whole way: passed too.

![The `watch` scene at set-out 1: the lid's lift stopped on where the watch
puts the knob, held, then lifted on out of the camera's view and set
down](figures/hands_watch.png)

(`MUJOCO_GL=egl uv run python scripts/hands_check.py --strip watch 1 --out
docs/figures/hands_watch.png`; `--view watch 1` in the viewer.)

**What is true now:** a watched tag's reading is the robot's 0.1 s after
its picture, 15 pictures a second, placed through the robot's own model of
itself without moving the flight; the fixed head camera loses the knob past
50 deg of lid; a render flatters a moving tag.

## The map's size limit (issue #558)

The world is to grow as one world (#557, #560), and a robot's map was
capped at 750,000 cells of 5 cm (`occupancy_grid.MAX_CELLS`), a cap #215
set for a planner since replaced: pure-Python A*, 1.26 s a plan across the
loop. The robots walk on the compiled lattice ("Walking into the
unknown"). `scripts/grid_spike.py` times what grows with the cells at
today's 469,200 and at 2x, 3x and 4x as many -- the bounds grown evenly
round the house by whole half metres, so no cell of today's map moves --
through the bodies' own code, each scale in a process of its own; the
plans and the explore's leg on #215's synthetic grid (known floor with
one wall across three quarters of it, corner to corner). Timed on the
deploy box by #385's method: the serving image in a throwaway container
pinned to a core the live pair was not on, one llvmpipe thread.

| cells | 469,200, today's | 937,200, 2x | 1,408,000, 3x | 1,876,800, 4x |
|---|---|---|---|---|
| a scan's update (10 Hz), ms | 1.2 | 1.4 | 2.3 | 2.6 |
| the camera's walls a frame, quadruped / humanoid, ms | 0.09 / 3.0 | 0.09 / 6.6 | 0.11 / 9.3 | 0.13 / 15.4 |
| a plan across the grid / a 3 m hop, ms | 42 / 9 | 74 / 19 | 144 / 36 | 177 / 42 |
| ... of a plan, the inflation's transform and the weights, ms | 7 | 14 | 23 | 32 |
| the explore's mask, frontiers and component, ms | 7 | 14 | 24 | 30 |
| ... its A* across the grid (pure Python), s | 0.80 | 1.66 | 2.64 | 3.79 |
| the matcher's field / a match, ms | 3.3 / 0.3 | 3.7 / 0.3 | 3.3 / 0.3 | 5.1 / 0.4 |
| the grid's image (physics thread) / its PNG (sender's), ms | 0.4 / 3.9 | 1.0 / 8.1 | 1.3 / 11.6 | 2.7 / 23.5 |
| the PNG of a world mapped as the house is, kB | 7.3 | 14.4 | 21.5 | 28.6 |
| `world.npz`, the pair's maps: raw / on disk, MB; the write, ms | 13.7 / 0.5; 32 | 24.9 / 1.1; 61 | 36.2 / 1.6; 82 | 47.4 / 2.1; 138 |
| a robot's map arrays / a plan's peak / the lattice, MB | 7 / 14 / 20 | 14 / 28 / 40 | 20 / 42 / 61 | 27 / 56 / 81 |
| the quadruped pair walking #507's routes, real time | 1.37x | 1.31x | 1.29x | 1.30x |
| ... its map work, ms a sim second a robot | 32 | 41 | 52 | 62 |
| the humanoid pair, the same walks | 1.30x | 1.32x | 0.85x | 1.06x |
| ... its plans a sim second a robot, and each | 11.2, 9 ms | 3.5, 19 | 9.1, 28 | 2.5, 43 |
| a fresh quadruped across the house (254-262 s to the west street): real time; a plan | 2.43x; 27 ms | 2.34x; 55 | 2.21x; 77 | 2.15x; 108 |

The PNG and the save run on the house as the quadruped's own senses laid
it from every metre of floor it could reach (88 % of today's grid known),
stacked k times along y so that no copy lies inside DEFLATE's 32 kB window
of another: a world k times the house, mapped as completely. The live
pair's save was 1.4 MB at today's size (#345): a day's maps are noisier
than one pass. The walks are the same routes and sim seconds at every
size, not the same trajectory: Dijkstra breaks a tie by a cell's index,
and an index moves with the grid's width. The dev box (an i5-9600KF) read
the same shape: its quadruped pair 1.21x to 1.15x, the explore's A* 0.69
to 2.98 s.

**What grows.** Each scan's update grows with the cells, not flat as the
cap's old note had it: its rays' work is flat, but its evidence is a
grid-sized `bincount`, added to every cell and clipped, where only the
cells its rays crossed change. Each plan pays the inflation's distance
transform and the lattice's weights over the whole grid however short it
is -- 7 ms of a 3 m hop today, 32 at 4x -- while its search stays inside
its limit (the straight line and 2 m). The explore's frontier leg
(`behavior.navigation.plan`) is the last planner in pure Python, A* over
the mapped floor to the nearest frontier: across the whole grid it took
0.80 s today and 3.79 s at 4x, and a far frontier late in an explore pays
it, holding both robots' physics while it does. The wire's image is built
on the physics thread once a sim second a robot (3 ms at 4x), its PNG on
the sender's. The save, every 60 sim s on the physics thread, is 12 bytes
a cell a robot raw (the map and the camera's layer), deflated at level 1.
A process holds 146 MB more at 4x (the quadruped pair's walks peaked at
724 MB today, 870 at 4x), against the live pair's 638 MiB of its 2 GiB
`mem_limit` as this ran.

**What does not.** The scan matcher works in a window (`WINDOW_M`, 7.5 m
either way): its field costs 3-5 ms a refresh at every size, a match
0.3-0.4 ms. The height map is a 4 m window. The lattice is built once a
process in under 30 ms at 4x (not the 0.1-0.4 s "Walking into the
unknown" had).

**The humanoid paid two of them oftener.** Its camera's walls (what the
LIDAR maps, grown 3 cells, which its layer is kept out of) were its whole
grid dilated every depth frame: 11 ms a walked sim second today, 47 at 4x,
where the quadruped's were the 4 m square its points can land in, under 2
at any size. And its walk replanned every physics step while its waypoints
were spent short of the goal: on #507's walk into the kitchen (the dev box)
114 and 64 of its plans came in the last 0.3 m, the quadruped's 5 and 2 --
it takes its last steps slowly. Over the pairs' walks it made 2.5 to 11
plans a sim second where the quadruped made 0.5, a number that moves with
the trajectory, so its pair's speed followed its replans more than the
grid: 1.30x at today's size with 1,738 plans, 0.85x at 3x with 1,323,
1.06x at 4x with 360. At 4x its worst rate would be about 490 ms of
planning a sim second a robot.

**What is true now:** `MAX_CELLS` is 1,876,800, four times the house's
grid and the largest size measured: there the quadruped pair walked at
0.95 of today's speed, a plan across the whole grid takes 177 ms, a robot
holds 27 MB of map and a process 81 MB of lattice. The humanoid's two
levers are the quadruped's now (#559, "The humanoid pair served"): its
walls kept to the camera's window, and a plan asked for again within half
a second from where the last was made held. The levers left, if the world must grow
past it: the update into the cells it touches; a plan's costs and weights
within its search's limit; the explore's leg on the lattice.

## The humanoid pair served (issue #559, stage 1)

The humanoids replaced the quadrupeds in the live world (#559, Ben's
decision), deployed at stage 4. Stage 1 serves them in today's house and measures it, without
deploying: `serve.py --pair --body humanoid` (`$PLUGGY_BODY=humanoid`)
streams `home_humanoid_pair`. Every number is the dev box's (an i5-9600KF,
six cores, a browser and an editor beside it) unless it says otherwise.

![The humanoid pair's day: Luca and Rowan exploring the house on their own
legs, each from above and behind every 40 s
(`two_robots.py --world home_humanoid --fast --strip --strip-every 40`)](figures/humanoid_pair_day.png)

**The world** (`humanoid/world.py`): the house built at load from the
generator's file, with the house's own fixtures -- the lab's plate signs,
the whiteboards' tags, the task areas' tags -- and none of the quadruped's
dock, rack or tools, so stage 1 offered no job (the hands' came at stage
3). The pair meets each other's bodies and never its own (`PAIR_BITS`), the
second in its livery. A world with no charger -- stage 1's, before its
cable's station (stage 2) -- ends a charge the robot chooses at once, said
in History (`NO_CHARGER`), and sends it to none, and an empty pack is the
`flat` death. On its real pack (778 Wh) that is hours away: its explore
draws 166.4 W (11.745 Wh over 254 s, `energy_spike.py --world
home_humanoid`), so a full pack explores for 4.7 h.

**What the loop took.** The lifecycle had never run a humanoid's day, and
its first flights met six things that were the quadruped's: the day began
by watching for the rack's tag (`Body.start_discovery`); the near field and
the peer channel read a depth camera the humanoid did not share
(`Body.depth_camera`: now the frames its walk lays its layer from, at its
camera's 5 Hz); each robot mapped the other (`Body.know_peer` did nothing:
painted into the map, a robot walked past walls in the one that passed it);
the day's summary asked after a tool; a restored pair indexed the
humanoid's geoms as an array, which were a set; and a mind's first
`other_needs` (#208) asked what was on the other's fork. Each answers
plainly now (`tests/test_humanoid_served.py`).

**#558's two levers.** Over 100 sim s of the pair exploring (cProfile on)
the walk made 3,115 plans, 23 ms each, 71 s of the run's 155 -- the pair at
0.65x real time. It replanned every physics step once its waypoints ran out
short of the goal, which the quadruped had stopped by holding a plan asked
for again within half a second and 5 cm (`REPLAN_HOLD_S`, the Navigator's
now -- but a plan the drive drops because the world moved,
another robot fallen, lain down or up again or a hold standing it still, is
made afresh); and its camera's walls were the whole grid dilated
each depth frame (3.6 s), where the quadruped's were the window its points
land in (the humanoid's now too). With both, the same run made 126 plans
and ran at 1.30x.

**The speed**, interleaved on the dev box (one llvmpipe thread):

| | real time |
|---|---|
| the humanoid pair served (`serve.py --pair --tasks --near-field --free-run`, the stream to a sink), 300 sim s of its day: explore, then stand by | 1.55x, 1.59x |
| the quadruped pair served the same, its day the jobs it took (820 sim s) | 1.12x |
| the humanoid pair on #507's walks (`humanoid_spike.py --speed`'s rows, 60 sim s), before the levers / with them | 1.42x, 1.42x / 1.57x, 1.56x |
| the quadruped pair on the same walks | 1.27x |

The served days are not the same work, the walks are: on them the humanoid
pair ran 1.23 times the quadruped pair's speed. **On the deploy box** (Ben
approved the run; 0880e6f in a throwaway container of the served image on
two idle cores beside the live pair, the same flags with `--pack hosting`,
interleaved): the humanoid pair served at **1.62x and 1.65x** over 300 sim
s, the quadruped pair at 1.16x and 1.21x over its 606 s of jobs. The
humanoid pair's process held 287 MiB, the quadruped pair's 734 (its
cameras rendered tags as it walked), against the sim's 2 GiB `mem_limit`.

**A restart.** The humanoid keeps what decides anything
(`HumanoidMission.kept_state`): its three policies' drivers -- their steps,
targets and last actions, and the gains they set in the MODEL, which a
restart compiles fresh -- its IMU's filter and its joints' and grippers'
noise, its reach, its reckoning with its planted foot, its map, the
camera's layer and the walls it is kept off, the scan matcher and its
clocks. Saved at the day loop's first pass and carried on in a new process,
its day was IDENTICAL through its whole explore (`determinism_spike.py
--world home_humanoid --resume-at 5`: 508 state samples to t = 260 s). ⚠ A
test saving on a step every controller's period divides read a fresh reach
as a kept one: 685 steps are the reach's 5 and the policies' 10 alike, so
the parity tests save on 687. Never saved with a hand out (`Body.working`):
a hand's move is a routine's. The walker that sees the ground (not served)
keeps no scan, and starts its ground anew as after a fall.

**The change of body.** Luca and Rowan keep who they are: a save of the
quadrupeds' house carries on in the humanoids' as a changed world does --
the clock, the deaths and the jobs, never the bodies or the maps -- and the
pack at the share it was saved at (`capacityWh`; a save from before it is
on the quadruped's served pack): a quadruped at 40 % of 194 Wh wakes a
humanoid at 40 % of 778. History says "the world restarted and could not
put me back where I was (I have a new body since then)". The share is the
change of body's alone: the same body on another pack keeps its watt-hours.
A job held there that this house cannot do -- each of the quadrupeds' --
is back on offer and said in History (`_resume_jobs`); in review the
mouse's crashed the restart, and a challenge's stayed held with nothing
here to grade it.

**What its head does not see.** The garden light's pole is 4 cm wide and
0.9 m tall, under the humanoid's LIDAR (1.41 m up; the quadruped's, at
0.51 m, mapped it). On the first minded day Rowan walked into it and stood
against it for the rest of the run, every walk after "stalled" where it
stood. Its depth camera saw the pole well -- 80 to 105 points a frame from
1 m -- but the pole fills part of its 5 cm cell and the floor round it the
rest, and each floor point voted the cell down: from 1 m off the layer
read -0.4 to -1.0 there, from 1.5 m it took 7 frames to mark it, and past
1.8 m the layer does not reach. A cell a frame's points hit is no floor in
that frame now (`HumanoidMission._fold_low`; the quadruped's rule as it
was): from 1 m and from 1.5 m the pole is in the layer at the third frame,
0.6 s, and #507's walk scene flies the same to the centimetre.

**Falls and deaths in the loop** (`determinism_spike.py --pair --world
home_humanoid --fall-at 40 --drain-at 120 --restart-after 30`, the pair's
day flown twice: IDENTICAL over 508 state samples, through both deaths and
both stand-ups). Laid on its side where it stood (`pair.arrange_hazards`)
Rowan rose part way, fell back with its head at a wall and lay there: the
`stuck` death 20 s on (`GETUP_BUDGET_S`), stood up by the world 30 s after
that. Emptied at t = 120 s, it died `flat` and was stood up on a full
pack. Laid down on open floor (1.6 m clear all round) it was up 5 s later
and walked on with its explore. The get-up trained on open floor alone
(#507): a fall against a wall is still a death.

**A minded day** (the deployed arm and model: `--arm autonomous --origin
unseeded`, `zai-org/GLM-5.3-Flash:cheapest`; 1,207 sim s in two runs, the
first ended by SIGTERM at 626 s as a deploy ends one, the second carried on
from its save at 1.34x and 1.63x real time). No death, no fall and no
charge asked for, the pack at 95 % at the end. Both minds wrote their event
maps first, split the rooms between them and told each other where they
were going; after the restart Luca said "the world restarted under me but I
picked up where I left off". Of their walks, 7 arrived and 1 ended in its
zone short of its middle, and 8 gave up with the other robot in the way:
from t = 720 s on the pair stood at the living room's garden door, each
walk of each giving up after 10 s 1.2 m from the other, the minds trying
one room after another through the same door. A robot in another's way is
asked to make way only while it RESTS (#415, `Navigator._ask_way`): a
quadruped lay down to rest by reflex, and a humanoid stood and never
rested, so two standing robots waited on each other (it rests standing
now, below). A mind also asked visitors where "the
charging dock" is (and on an earlier flight one wrote a procedure to push
on "turning if my bumper hits"): the prompt and the constitutions then
still described the quadruped's body and world (stage 3 rewrote them).

**Making way, standing** (`humanoid/way.py`). A humanoid rests STANDING
now -- no motion command for `REST_S` (the quadruped's reflex's 8.6 s),
no walk of its own, no hand out, on its feet -- and resting across the
other's way it is asked aside as #415's quadruped was, by the same rule and
walk to the same spot (`MakeWay`, each body saying its own in hooks) beneath
whatever routine holds it still, its legs handed to the walker where it
rests once its posture is the stand's (a bow left held is no stance to walk
from); a routine that takes the legs for the standing policy ends it. Two
things took it. Its front stop,
re-armed by every scan while it rests in a doorway, ended the step aside
before it began, at 3 doors of 6: a humanoid's backs it off and plans
again, as a walk does. And the asker's way past it (`_route_past`, "the
planner's own, unremembered") answered the plan the drive had just made
with that robot in the door, None, once the plan memo was the Navigator's
(#558's lever, above): nobody of either body was asked at all. A robot
resting in each of six doors of the house and the other walking through
(`make_way_spike.py --near-field`, true-floor maps):

| | arrived | the step aside |
|---|---|---|
| the humanoid pair, nobody asking (`--before`) | 1 of 7 | -- |
| the humanoid pair | 7 of 7 | 0.78-0.89 m in 2.8-5.7 s |
| the quadruped pair, the same doors | 7 of 7 | 0.82-1.02 m in 2.6-8.9 s |

(the seventh scene is the 3 m hall's middle, where nobody is asked.)

A minded day with it (949 sim s, the deployed arm and model, from nothing):
no walk gave up with the other robot in the way, where the day before it
8 had; the pair did not meet in a doorway, so nobody was asked.

**Never off its charger** (stage 4). The pair's recording found it: the
first robot plugged in at the station, the other came to charge and walked
to wait, its way cut the charging robot's disc and asked it aside, and it
stepped off with the plug in its port and pulled it out ("CHARGE: lost the
pins" 27 s into the charge, plugged in again 26 s later). A robot whose
plug a port catch holds (`HumanoidWay.plugged_in`, what its BMS reads) says
no, `charging`, as the quadruped on its dock said `docked`, and the other
waits: flown again, the first charged 13 % to 90 % in 160 s and the second
after it. It reads the plug, never the lifecycle's `Body.docked`, which is
set only while a charge holds and which a restart forgets with the plug
still in.

**On the lab's plates** (stage 4). The change of body flown on one volume
(a scripted quadruped pair's save, then the humanoid pair on it, as the
deploy runs) found that a humanoid walks onto them: Luca found the lab's
three plate signs exploring and then walked onto the plates for 30 s, the
feed, the toy and the shock plate (the mouse "on_its_side"). Kept off every
pad it knew, as the quadruped's planner was (`PlaceWalk.keep_out`), it
pressed none after the signs' finding -- but one standing on a pad as it
found the sign at the pad's far edge planned no walk off it (from the feed
pad, none 2.5 m south nor 3 m east, the sign's facing known by its tag or
only by where it was seen from). So its walks keep off no plate, and the
humanoids' cage has its shock plate switched off (`Cage(shock=False)`; Ben,
2026-10-31: no act, no count, no press, no flag on the wire): a walk across
the feed or the toy plate feeds or amuses the mouse, a `press` row, and
pays nothing.

**Its belief drifts on a long explore.** After a 240 s explore of the
whole house the humanoids' beliefs were 0.2 to 0.7 m off the truth (dev box,
scripted pair days, before and after every change here). Its belief is
fixed at its station as it charges, the station being commissioned (#476,
stage 2). A map laid through a
belief that far off lays the camera's points beside the LIDAR's walls of
another moment: in one minded day Luca's layer held 29 cells across the
living room's hall door, 12-17 cm off the wall's line, and three walks to
the kitchen gave up "no route". On the hand jobs' tour it ended 2.3-2.5 m
off across the street, and every job still passed (stage 3); the deployed
period reads it (Observatory.md).

**Seen by the site** (stage 4). `look` asks the site for a picture from the
head's camera (`nav_eye`, 1.3 m up, facing where it faces), and the site
draws the pair off `home_humanoid_pair`'s scene, the station's cable strung
between the station and its plug every frame (the scene's `cables`: a
tendon has no geometry). (The death seam reads a topple off the posture the
legs hold, `Body.held_pitch`, since stage 3: a reach's deepest bow is 63 deg
of pelvis tilt, past `TOPPLE_TILT_RAD`'s 60.)

## The humanoid's charging station (issue #559, stage 2)

**What is true now.** A humanoid charges standing, by a cable it plugs into
itself (Ben's call 1). Its station is a box on the living room's south wall
where the quadrupeds' dock lay (`humanoid.world.station_pose`), holding a
plug on a tool balancer's rope; the port is on its torso's front, 0.90 m up
and 7.5 cm right of its middle (Ben, 2026-10-30). Both are built and flown
from the robot's own senses: the station off its two tags, the port off its
model of itself, a mated connector off its pack's current (`Body.charging`).
The routine is ours, never one it learns (`humanoid/charge.py`); the
station is an activity (`activity/station.py`); the parts are in Parts.md.

**Where the port goes.** The call said "where a hand reaches and the
head camera sees". No point of the trunk is in any head camera's view, bowed
or not -- they ride the trunk, the nearest point 21-46 deg past each one's
edge -- so the port is where a hand reaches: an IK sweep of the robot's
model of itself found a straight carry into a forward port right of the
middle 17-26 cm long with the arm 26-57 mm clear of the trunk, standing.
(`mj_geomDistance` read 0 for two boxes 10 cm apart: box clearances are
taken off their surface points.)

**The parts.** Rosenberger's RoPD -- the e-bike charging connector:
magnetic, 60 V, 25 A, pulled off past its sheets' minimum 20 N, NOT
hot-pluggable (the charger runs off its signal pins) -- a panel plug in a
printed funnel on the robot and a cable jack at the station; a Mean Well
NPB-750-48 at 10 A, its CV set to 50.4 V; a Tecna 9300 tool balancer, its
least pull 2.0 N, drawing the plug's back home into a cup whenever nothing
holds it (Ben, 2026-10-30). No spring cable reel pulls lightly enough: the
smallest, 13-36 N, would pull the plug off the robot.

**Two catches are the activity**, each a weld taken where the plug is and
drawn to its seat (`PULL_IN_S`), off past its force: the PORT (20 N off its
face or across it) and the CUP (4 N straight out of it; its walls take a load
across it). What flying took, each at its constant:

- the plug's frame at its face and the welds at solref 0.004: at 0.02 a 50 g
  plug drooped 8 deg about its seat (MuJoCo's soft constraints are stiff in
  ACCELERATION, and a small inertia is little to hold);
- the cup a catch, never a socket: held by contact alone the plug rattled
  (0.1 m/s, 6.7 deg) and, after a breakaway, swung under the box and lodged
  inside it;
- the plug's swing damped (its cable's drag, 0.5 N s/m) and its spin apart
  (0.02): one value for the free joint's six axes, 0.5, made its spin grow
  four times a step until MuJoCo reset the world;
- its inertia its first 0.3 m of cable's: the bare jack's, pinched between
  the pads, threw its rotation off in a step;
- a catch takes only a plug NO GRIPPER PAD HOLDS: held stiffly off the
  middle, the port fought the hand past 20 N and let go again and again, and
  the cup, catching a plug the hand was still carrying, let go and never
  took it back (3 unplugs in 10 left it unparked; 11 of 11 after);
- the port re-arms only beyond the funnel's reach (re-armed inside it, the
  unplug's pull was caught again every step, 30 times, and the plug never
  came free).

**The plug-in.** The hand takes the plug's line 10 cm short of the parked
plug's grip -- jaws upright, the fingers across it to the robot's left and
swung 20 deg toward the plug's back about the jaws' line, which keeps the
pads pinching it from above and below -- slides its open jaws over the plug
along its axis (in from its side, the arm's swing read 35-49 N it was not
feeling and stopped short), grips it 50 mm behind its face, draws it out
against the balancer, turns it to the port as the body then stands (a
posture taken to reach moves the port) and pushes it to the body with 10
N, and LETS GO in the funnel. Straight across, the parked plug was in the
standing arm's reach only from up to 3 cm behind the seat (the wrist bent
past 90 deg), and past that the hand ended held back or out of reach of it;
swung, from 7 cm (the robot's model of itself, every seat of the check). The
swung grip is 45 mm behind the face: at the jack's back end, 40, the
gripper's housing met the belly with the plug 21 mm off the port, past the
catch's 15, and at 50 the pads' far edge came 4 mm off the station's boss.
Where the swung hand's way in -- along the fingers -- presses on the trunk
(3 cm in and 4-6 cm to the right of the seat), the hand fetches straight
across at the jack's back end (`charge.GRIPS`). Every squeeze on a plug a
catch holds begins at the gripper's least, 5 N, until both pads are on it:
shut at once, the first pad to meet a seated plug pushed it across the port
past the connector's 20 N and it broke away. And a squeeze holds the plug
only where the jaws stop at the jack's 24.5 mm (the gripper's encoder reads
it to a millimetre): the station's fit is 1-2 cm off along the plug (its
tags' range off the colour camera), and walked in from across the house
with the grip at 50 mm, 12 first pushes in 15 were made with a hand shut on
the boss, 40 mm across -- held, and pushing nothing. Shut wider than the
jack, the hand opens, draws back 2.5 cm along the plug and shuts again. A
look at the seat that sees no pair is taken again bowed 0.2 rad: nearer the
station than its seat the tags are under the colour camera's view, and on
the fit from before its last step, the step's odometry put the hand on the
boss from both grips. Pushed to
the body, plugs landed 12-33 mm across the port and 7-10 mm low: the hand
knows where it holds one to 1-2 cm (its encoders and backlash, and the plug
free along the Dex1-1's 80 mm pads), so the funnel round the port is 70 mm
across -- as wide as the housing, 40 mm off the plug's axis, clears -- and
takes a plug anywhere in its mouth (⚠ the funnel is the catch's, as the cup
is: drawn, never touched). A miss is fetched again from the cup and pushed a
step to one side. The walk in: from the standoff, 0.75 m off (past 1.07 m
the tags leave the colour camera's view, which looks no higher than 19 deg
under level), facing the cup (facing the station's own heading from 0.2 m
aside put the far tag out of view), it fits the station and steps to the
seat as it sees it, 0.35 m off; within 6 cm and 6 deg, but never more than
3 cm nearer the station (4.9 cm in, its hand swung into the box). Its walker
barely backs up -- 4 cm in 3 s asked 0.25 m/s -- so an undock turns away.

**The unplug.** The open jaws slide along the seated plug from 3 cm behind
its back, squeeze it at 5 N and then 80 (at 40 it slipped out; shut at 80
at once, 46 unplugs in 110 could not grip it, 45 of them having knocked it
off its catch), pull it straight off past the connector's 20 N, carry it to the cup's
mouth and let go there. One that fails still ends unplugged: the robot turns
away, the rope's stop pulls the plug off at 20 N and the balancer takes it
home.

![Lost and found by its station: a humanoid 2 m out, its belief set 0.7 m
nearer the wall than it is, walks to where it believes the standoff is
(1.45 m off the station's face, the tags above its colour camera), sees
nothing, steps in by its LIDAR, finds the tags, puts its belief right off
them, and walks in again and plugs itself in, a tile every 4 s
(`station_spike.py --strip lost`)](figures/humanoid_station_lost.png)

**Lost.** No tags in sight from the standoff the belief walked it to, the
robot is lost (#476's rule), and the station, commissioned as the dock was,
puts it right: facing where it believes the station is, it steps in until
its LIDAR reads the wall 0.8 m ahead -- never by its belief, which is what
is wrong; from viewpoints set on a belief 0.7 m off toward the wall, every
look was further from the station than 1.07 m -- and looks ahead, 34 and 63
deg either side (the camera sees 34.5 either side of its axis), turning to
a lone tag of the two where it sees one, then from a step 0.5 m along the
wall each way. A look that fits the station is a FIX (`HumanoidBody._fix_at`,
the same as `anchor_at_dock`'s): the belief moved, and past the matcher's own
search the map laid on the old one dropped, and the loop's walk to the
standoff goes again on it.

**Measured** (`scripts/station_spike.py`, the house, one humanoid):

| | |
|---|---|
| `--seats`: 140 seats over what the seat check accepts (3 cm in to 6 cm back, 6 cm across either way, 6 deg either way), each started there | plugged in 140 of 140 (the swung grip 133, straight across 7), most on the first push; unplugged and the plug home 140 of 140; no falls (5 after the plug was off, before the unplug let go at the cup's mouth). Straight across at the jack's back end, the same seats plugged in 110 and unplugged 64 |
| `--approach`: 15 starts round the standoff (head on, 0.2 m and 20 deg off each way, 12 drawn within 0.2 m along, 0.25 m across, 25 deg) | docked 15 of 15 in 18-41 s, each unplugged and the plug parked; no falls (11 of 15 before) |
| `--hold`: plugged in for 60 s | held all of it; 432 W in, 86.9 W drawn standing: 345.1 W net |
| `--fall`: shoved over backward plugged in | the plug broke away and was home in 2.0 s; the robot up again |
| the loop's own walk and plug-in (`energy_spike.py`) | 345.2 W net into the pack (energy.json) |
| the reserve (`energy_spike.py --reserve`) | 13.72 Wh over 44.62 m to the garden doorway, 307.6 mWh/m; carried 17.5 Wh |
| `--walk-in`: a fresh humanoid explores 120 or 240 s, walks on its own map to each of 8 places across the house (the bedroom, hall, kitchen, workshop, garden and south garden; the lobby and lab across the street), and the loop's own `go_charge` takes it to the station | **docked 15 of 16**, each in 46-131 s, 10 on the first push; its belief 0.04-0.51 m off where it walked to (1.07 m at the lobby after 120 s, which docked on its third push), 0.01-0.42 m once docked, before the charge's own fix; no falls (10 of 16 before) |
| `--lost`: a fresh humanoid 2 m out facing the station, its belief set 0.7 m wrong each way and 12 drawn 0.4-1 m off and up to 5 deg turned, the loop's own `go_charge` | **docked 16 of 16**: the 14 that saw no station searched and found it in 4-62 s, their fix putting the belief within 0.11 m; the two whose walks ended near enough docked as they were, their fix the charge's own |

**What is not yet.** One walk-in in 16 pushed four times and the port took
none: where the plug sits in the jaws still misses pushes now and then. The
reserve's own dock leg gave up on the explore's map and is carried by rate;
one station a world.

## The humanoids' first jobs (issue #559, stage 3)

**What is true now.** The served humanoids' first jobs are the hands' four
checks (Challenges.md §10, §11) set into the house: a counter 0.72 m high
and 0.6 deep against the east wall of the research center's storeroom --
the room the facility's address lands in, and the practice's and the
sessions' counter's room -- with the chest, a cabinet of drawers, a
cupboard and the cube on its board, 1.2 m apart, each square where the
middle of its check's set-outs had it from a robot's work spot
(`home.storeroom`). Over each, a 120 mm tag on the wall at 1.15 m (60-63),
one row.

**Finding the counter.** The humanoid finds a place by the quadruped's
search, unchanged (`humanoid/places.py`, `HumanoidPlaces`: where it should
be, the address, viewpoints round it), with a look of its own:
its head's NAVIGATION camera, pitched 20 deg for the walls' tags, each tag
in a decode placed through its odometry's model of itself -- posed every
10 ms off its encoders and IMU, so a look draws no noise a flight reads --
and its stance into its map. From the storeroom's door, 5.5 m off, it reads
all four of the row; from in front of the counter, 0.8 m, the one over it.
⚠ A FIND STOPS AT ITS TAG'S FIRST SIGHT, so the place's facing is only as
good as what was seen by then: the row's baseline once two of it were in
view, else the way the tag was seen from (`mapping.places`). One tag seen
through the storeroom's door from the lobby left a walk's `face` 17-20
deg off the wall.

**The walk there, then the check.** Each reference (`challenge/
solutions.py`) is two procedures in the robot's own vocabulary, because one
would pass the language's 1,500 characters. The first walks there: it finds
the tag beside the prop's in the row, then the prop's own; walks to 1.3 m
out of the wall's face and 0.2 m to its right (outside the planner's margin
round the counter, which the depth camera's low layer lays); faces the
wall; reads the handle's tag from there and turns ONCE, off that face's
normal, to a stance a little left of square (`STANCE`, each prop's own);
steps to 30 cm short of where the check's set-outs' middle put the handle;
and steps the last 30 cm off a fresh look. The second is the check's own
procedure (Hands.md), unchanged for the door and the chest. What flying
took, each at its constant:

- **The row's neighbour is found first.** One tag of the row seen through
  the storeroom's door from the lobby left a walk's `face` 17-20 deg off
  the wall (the find stops at that sight, above); two give the wall's way.
- **`step` lands a 0.3 m step.** From rest it walks within 1-3 cm of what
  it is asked, but after a turn a tenth short (0.39 m of 0.43), and a short
  step, 0.10-0.20 m, is a stride that overshoots by up to 3.6 cm: a door's
  lever stepped 3 cm too near was gripped on nothing five times. `spot`
  reads a handle's tag to 1-1.5 cm at every distance it decodes (0.47-0.95
  m).
- **The turn is read once, from far, off the face's normal.** `spot` of a
  26 mm cube from 0.85 m reads its FRONT face (the most square-on from
  there), whose edges, one of them upright, give no turn: read off the
  edges, the cube's procedure gripped at 42 deg. Read again from the
  stance, a face seen obliquely gave 43 deg. The edges are left to the
  check's own procedure, from close, where it reads the top.
- **One tag's facing left a robot 28 deg turned to the counter**, and its
  arm's way home dragged a jaw along the counter's top, turned the robot
  23 deg and toppled it, a second into the grade's hold (four cube jobs in
  six from the station, before the neighbour and the turn).
- **The stance is the checks' procedures'**, the set-outs they were written
  for. Squared to the prop, the arm's way home pushed the drawer 4 cm back
  in and the door opened only 44 deg; turned 5-10 deg left of square, the
  drawer, the door and the chest passed 8, 8 and 6 of 8. A stance lands
  4-5 deg either side of the one asked (the turn, the step's own, the
  face's read), so each prop has its own: 0.12-0.25 rad.
- **The drawer's procedure pulls 4 cm further and lets go of the knob
  before its arm goes home.** Over the tour, its way home pushed the drawer
  back to 14-15 cm in 5 of 8 (15 is the bar): pulled to 0.22 m, the open
  hand drawn 6 cm back off the knob first, every drawer stayed 20-21 cm
  out.

![A humanoid's drawer job from the station: across the street on a fresh
map, into the research center's storeroom, the counter's row found, the
turn to its stance and the steps in (a tile a minute on its way), then the
check's own procedure at the counter, reach, grip, pull and let go (a tile
every 4 s), and the grade
(`solve.py --feature drawer --from station --draw 3 --strip`)](figures/humanoid_job_drawer.png)

**Measured** (`scripts/solve.py --feature F --from tour --n 8`): each flight
a fresh humanoid on a fresh map, the job offered and its prop drawn, set
out, claimed, its reference run as the robot's procedures, `done`, and
graded on the seam. The tour starts flight k at the next of eight places
across the house (the station, the storeroom, the lobby and the lab across
the street, the living room, the hall, the kitchen, the garden), with the
prop at draw k. **The bar, written before they flew: 6 of 8.**

| job | passed | the flights |
|---|---|---|
| `open_drawer` | **8 of 8** (3 before its procedure's change) | 20-21 cm out, the call 39-281 s after the claim |
| `open_door` | **8 of 8** | 62-74 deg open, the call 57-301 s after the claim |
| `open_chest` | **7 of 8** | opened to 59-65 deg, the call 61-301 s after the claim; the garden's (draw 7) let fall at 1.7 rad/s as it was lowered |
| `place_cube` | 0 of 8, **not offered** | set down 2-7 cm from the mat's middle (2.5 allowed) |

No falls in the 32. The slowest pass took 313 s from the station, and each
job allows 16 minutes from the claim (three times it, rounded up). Priced
from the station with the place remembered (`energy.json`): the drawer
7.08 Wh, the door 5.72, the chest 5.98.

**What is not yet.** The cube: offered once a reference places it (its
check passed 3-4 of 8 at its own set-outs, the fingers against a 26 mm cube
on a board). From the station, every flight's belief ended 2.3-2.5 m off
the truth, the street crossed on a fresh map, and passed: the counter's
places are laid in the map its walks plan on. A lid let fall once in 8.

## Where the humanoids get lost (issue #572, stage 2)

**What the live pair showed.** Eighteen minutes after the humanoids went
live (#559's stage 4), Luca's belief was 2.05 m and 15 deg off the truth
and Rowan's 1.19 m and 16 deg; an hour in, 2.3 and 5.3 m. Neither knew:
past the matcher's search (0.6 m, 6 deg) a robot lays its map askew round
its belief, and its scans agree with it. Only the station puts them right,
from wherever its lost search finds it -- flown until now from a belief
0.4-1 m wrong ("The humanoid's charging station"). Two instruments take it
apart, one humanoid in the house, the served body, each flight a process
of its own under a memory cap.

**The drift, taken apart** (`scripts/humanoid_drift_spike.py`). A humanoid
flies the loop's own explore (the served 240 s), then the loop's walks
across the street to the storeroom, where the jobs' counter is, stands
there 60 s as a job would and walks back to the station's standoff, twice,
and is sent to charge. Its error is taken apart as it is made: each 10 ms
tick of its reckoning lays the legs' step through the heading it believed;
the same step laid through the TRUE heading, less the step it truly made,
is the legs' share; the two layings' difference is the believed heading's
share; every other move of the belief is the matcher's or the station's.
The heading has two shares only, the gyro's integration and the
corrections: the humanoid's heading is its IMU's alone
(`StanceOdometry.yaw`), so its legs put none there. The shares add up to
the error to 1e-13 m and 1e-11 deg. A flight's draws are its IMU's offset
and scale and its joints' noise: Luca's and Rowan's own, and other seeds;
`--learned` takes the gyro's offset off what its heading integrates, as a
zero-rate update that had learned it would (the quadruped's
`imu.Standstill`, #425: nothing under `humanoid/` has one).

| flight, from | gyro offset, deg/min | along the facility's outer walls | worst | home from the storeroom | the charge |
|---|---|---|---|---|---|
| Luca's draws, the living room | +1.06 | no | 0.66 m | 0.6, 0.6 m | docked; the station's fix put it within 1 cm |
| Rowan's, the hall | +2.98 | no | 0.57 m | 0.3, 0.4 m | docked |
| seed 5, the living room | -1.41 | no | 0.38 m | 0.2, 0.3 m | its plug-in missed: the port never took the plug |
| seed 7, the living room | -0.04 | no | 0.61 m | 0.2, 0.1 m | docked |
| seed 1, the living room, `--learned` | (-2.79) | no | 0.96 m | 0.1, 0.3 m | docked |
| seed 1 | -2.79 | round it, to the storeroom | 3.26 m | 3.2, 2.4 m | its walk gave up 4.6 m short |
| seed 2, the hall | +2.50 | round it, to the storeroom | 5.22 m | 3.8, 4.0 m | no station where it searched |
| seed 2, `--learned` | (+2.50) | its north side, already lost | 4.84 m | 2.6, 4.3 m | its walk gave up 4.9 m short |
| seed 10, the living room | +2.40 | round it, exploring | 8.49 m | 5.7, 5.7 m | its walk gave up 4.2 m short |
| seed 10, `--learned` | (+2.40) | round it, to the storeroom | 4.90 m | 4.9 m, 0.5 m and 12 deg | its walk found no route at all |
| seed 11, the hall | -2.69 | no | 1.65 m | 1.2, 1.5 m | no station where it searched |
| seed 11, `--learned` | (-2.69) | round it, exploring | 8.21 m | 3.5, 8.1 m | no station where it searched |
| Rowan's, stood 180 s in the street | +2.98 | round it, after | 6.24 m | 6.1 m | its walk gave up 3.4 m short |
| ...`--learned` | (+2.98) | its north side, exploring | 2.78 m | 2.5 m | no station where it searched |

**Indoors the belief holds; outdoors the head's LIDAR sees one wall or
none.** It spins at 1.41 m and every fence is 0.9 m, so outdoors it sees
the two buildings' walls alone -- and the middle street is 6-8 m from both,
at the edge of what the matcher reads (6.5 m):

| fourteen flights | returns a scan | matched, every direction fixed | matched, held along a wall | nothing to match | refused |
|---|---|---|---|---|---|
| the rooms | 318 | 97 % | 0 | 0 | 3 % |
| the gardens | 105 | 42 % | 30 % | 9 % | 19 % |
| the sidewalks and outer streets | 96 | 18 % | 57 % | 19 % | 5 % |
| the middle street | 30 | 0 | 25 % | 68 % | 7 % |

Indoors the belief stays within 0.1-0.3 m whatever the gyro does: Rowan's
offset, near the part's worst (3 deg a minute), put +39.8 deg into its
heading over 13 minutes and the matcher took -39.2 off. The legs add 0.2-2
cm a metre, which the matcher takes off as well. The gyro's share is its
offset times the time and its scale times the true turning, to 2 deg in
every flight.

**Lost is coming home past the matcher's reach.** The house's map, laid
true on the explore, refuses the scans of a robot that comes home further
off than the matcher's search (0.6 m, 6 deg) -- up to 1,800 a flight --
and the wide search (#422, 2 m and 15 deg) took back none of them for
good: not Rowan's `--learned`, which came into the living room 1.95 m off.
Nine of the fourteen were lost so. Four searched where they believed the
standoff was and found no station; five never searched, their walks to it
finding no route, 3-22 m short, in a map that put walls where they
believed they stood.

**Where: in and beyond the middle street.** In all nine, the error first
passed a metre within a minute of the middle street -- crossing it, on the
sidewalks either side of it, or going round the facility (a fall's few
seconds aside). Six of the nine went round the facility: walks to the
storeroom that passed its gate about a metre off and went round the block
looking for a way in (seeds 1 and 2, seed 10's `--learned`, Rowan's after
its stand), and explores that went round it (seed 10's, seed 11's
`--learned`); none of the five that held did. What it was made of there:

- *The matcher along one wall.* Along a sidewalk it holds the direction
  along the wall to odometry and turns the heading to the wall it laid
  seconds before, half a degree to two every five seconds: on seed 1's walk
  round, +21.6 deg against its gyro's -10; on seed 2's, +5.3 deg beside its
  gyro's +6.8; on seed 10's `--learned`, +9.9 deg with its gyro at +2. The
  facility's map was laid askew through it.
- *A heading carried across the street.* Nothing there fixes the position
  along the walk, and the 20 m between the garden's door and the lobby's
  turn each degree into 35 cm: Rowan's `--learned` left the storeroom 0.8
  m and 3.4 deg off and reached the living room 1.95 m off.
- *Standing where nothing is matched.* Rowan's draws stood 180 s in the
  middle street: its heading went from +0.4 to +9.2 deg, its offset times
  the time, past the matcher's 6; its next walk passed the gate 1 m off,
  went round the block and ended 6.2 m off. The same stand with its offset
  taken off held its heading to 0.1 deg. The live pair showed it too: on
  2026-10-31 Luca's heading turned 5.2 deg in 300 s standing at the
  storeroom's counter, no scan matched -- its offset, 1.06 deg a minute.

**The gyro's offset is one cause of three.** Taken off (`--learned`), the
gyro's share fell to 0-2 deg a flight, and one of the five flights it
changed held: seed 1's, whose walk then went in by the gate. Rowan's own
offset, near the part's worst, the matcher took off to the degree wherever
it walked.

**A fall.** Seed 11 fell in the garden on its explore, and its `--learned`
there too (the same draws, the same first minute). Its odometry followed
its feet through the get-up, and the belief jumped 1.7 m (2.1 m in the
`--learned`; the live Luca's 3.7 m on 2026-10-31, falling at the
drawer); and while it lay, its heading read -49 deg -- a body lying down
has no heading, its forward axis at the sky. Both jumps were inside the
wide search's 2 m: each map refused 19 scans and the search put the belief
back within 0.2 m. Luca's was not, and it stayed lost. Seed 11 came home
1.2-1.5 m off, to where it believed the standoff was: truly the south
garden, behind the station's wall. No station was there, and as it
searched, every scan refused, its heading ran on the gyro alone: -12 deg in
110 s of turning on the spot.

**The station from far off** (`scripts/station_spike.py --lost --far`). A
fresh humanoid, its belief set 1-5 m and 5-20 deg wrong, sent to charge by
the loop's own path: 2 m out facing the station (each of 1, 2, 3, 4 and 5 m
by 5, 12 and 20 deg, a bearing drawn), from the house's six rooms, and from
across the street (the lobby, the lab and the storeroom), four drawn each:

| from | docked | the rest |
|---|---|---|
| 2 m out | 3 of 15 | the three set 1 m wrong: one docked where its walk ended, two found it in 12 and 38 s; from 2 m and more, each searched 72-84 s and did not see it |
| the house's rooms | 1 of 24 | the kitchen's, its walk ending 0.47 m off; 23 searched and did not see it |
| across the street | 0 of 12 | 8 searched and did not see it; 4 never searched: their walks ran out of time twice (90 s each), 3-7 m short of where they believed the standoff was |

**Explored first, as the live pair's maps were laid** (`--explore 120`:
the belief set wrong, then 120 s of the loop's explore laying its map
through it, then the charge), one from each room and each place across the
street: docked 1 of 9. The explore grew the error before the charge began
in eight of the nine (to 2.1-6.5 m); the one found began its search 0.8 m
off, the seven not found 3.7-5.9 m off. Over both sweeps, 5 of 60 docked.

![The station searched for from 2 m off: a humanoid 2 m out, its belief
set 2 m and 12 deg wrong, walks round by the garden to where it believes
the standoff is -- truly outside the living room's south wall, the
station on the wall's other side -- and searches there 78 s, its belief
2.2 m and 12 deg off throughout, a tile every 8 s (`station_spike.py
--strip lost --lost-from station --lost-off
-1.606,-1.192,12`)](figures/humanoid_station_far.png)

**The search reaches about a metre.** By how far off the belief was where
it began: within 1.1 m, 3 of 3 found the station; from 1.5 m, 0 of 43 (4 at
1.5-2 m, 6 at 2-3, 18 at 3-5 and 15 further). The heading mattered less:
one found it 20 deg off. The search steps toward where it believes the
station is until its LIDAR reads the wall 0.8 m ahead, looks, and looks
again from a step 0.5 m either way along it: it searches the wall in front
of it, and from further off that is the wrong wall. Of the 43, 14 searched
from the south garden, outside the very wall the station hangs on (the
figure), 10 from elsewhere in the living room, 10 from the east garden, and
the rest from the bedroom and the sidewalks and streets.

**What is true now.** Indoors the humanoid's belief holds within 0.1-0.3
m: its LIDAR fixes every direction there, and the matcher takes off its
gyro's offset, whatever it is, and its legs' 0.2-2 cm a metre. Outdoors its
LIDAR sees the buildings' walls alone -- along the sidewalks one at a time,
the matcher holding the direction along it to odometry, and in the middle
street none -- and its belief comes home off by more than the matcher's
search there three ways: walking along the facility's outer walls, a
heading carried across the street, and standing still where nothing is
matched; and anywhere, a fall, whose get-up its odometry walks. Back in
the house, the map refuses it every scan and nothing puts it right but the
station, whose lost search finds it only from within about a metre; from
across the street the charge's walks run out of time first.

**Not done** (#572's stage 3, and its stage 4): a search that finds the
station from further off than a metre, and a charge's walk given the time
a walk across the street takes; walks to the facility that go in by its
gates, or a station there (#560's in the research center); the heading held
while the robot stands where nothing is matched (a zero-rate update, as
the quadruped's while it lies); its odometry held while a get-up has its
legs; and why the wide search took back none of them.

## The 3D LIDAR on the head (issue #581, stage 1)

**What is true now.** The humanoid's LIDAR is a Livox Mid-360S, upside down
on a bracket over its head (Parts.md, "The LIDAR"): its field is +7 down to
-52 deg, its rays leave 1.47 m up standing. Its LEVEL RING, the 360 bearings
at 0 deg, is the scan the map is laid from and matched by, as the RPLIDAR
C1's plane was, 5 cm higher: served, it behaves as the C1 did. Its points
LAID FLAT -- per bearing the nearest surface 0.15-2.2 m over the floor,
through the IMU's tilt (`Lidar.band`) -- see the house's 0.9 m fences, the
couch, the tables and the beds, and are a measurement's switch
(`HumanoidMission.band`) until stage 2 serves them. Flown with them, the 14
drift flights that #572 lost 9 of were 14 of 14 docked, as with the fences
raised to 1.8 m. A full frame costs 14-17 ms on the deploy box, and two
robots casting it at 10 Hz ran at about real time (0.97-1.15x).

![The 3D LIDAR on the head: a humanoid walks from the living room out by
the garden's gate, across the street and through the lobby to the
storeroom, its map laid from its points laid flat; red, each bearing's
nearest return 0.15-2.2 m over the floor -- the couch, the fences, the
walls' lower halves -- and blue, the level ring its map is laid from as
served, the buildings' walls alone; a tile every 15 s and one as it
arrives (`lidar_spike.py --strip`)](figures/humanoid_lidar_walk.png)

**The part.** Livox's Mid-360S (its sheet: 360 x 59 deg, 200,000 points a
second at 10 Hz, a range's sigma <= 2 cm at 10 m and <= 4 cm at 0.2 m, 0.15
deg, a 0.1 m blind zone, 265 g, 6.5 W; €772.95 at Funduino), Ben's choice
(call 2, 2026-11-01). Unitree's G1 carries a Mid-360 upside down in its
head. Against it:

| | Mid-360S | Unitree L2 | Mid-360L |
|---|---|---|---|
| field, deg | 360 x 59 | 360 x 96 | 360 x 44 |
| upside down, its steepest | -52 | -90 | -34 |
| points a second | 200,000 | 64,000 | 100,000 |
| a full round | 0.1 s | 0.18 s | 0.1 s |
| a range's sigma at 10 m | <= 2 cm | <= 2 cm | <= 3 cm |
| W, g | 6.5, 265 | 10, 230 | 4.2, 245 |
| € with VAT | 772.95 (Funduino) | about 437 ($419, Unitree) | none read |

The L2 spreads a third of the points over a field 1.6 times as tall, much
of it straight down into the body; the Mid-360L's field ends 34 deg down,
the floor 2.2 m out against the Mid-360S's 1.15. The bill buys the Mid-360S,
its cable (none in its box), its own 12 V converter (its 18 W start-up and
the contactor's coil are past one DDR-30L-12's 30 W) and its bracket: €16,322.28
with contingency, from €15,319.75. Always on, the body draws 74.9 W, 6.2
more (the unit through its converter, against the C1's 1.15), and weighs
43.01 kg, 0.36 more (the head 0.24, the converter 0.12).

**Its mount.** Upside down, its window under its base, so it hangs from a
plate: a 20 x 3 mm 6082 post up the head's back edge, turned ON TO the
sensor, holds it. Casting from the C1's height, 4.7 cm over the head's top,
the head hid every ray steeper than 31-35 deg ahead and aside -- the floor
out to 2 m. Its rays leave 10 cm over the head's top (`lidar_rise`), so the
steepest, -52 deg, clears the head ahead and to either side and meets the
floor 1.15 m out. What the robot's own body hides (`lidar_spike.py
--hidden`, each sector 20 deg either side):

| pose | of a frame | ahead | aside | behind |
|---|---|---|---|---|
| standing, arms hanging | 3.8 % | nothing | clear to -50 | the post's 3 bearings |
| a hand held at a counter, 0.94 m | 4.3 % | clear to -42 (its gripper) | to -50 | the post |
| at a high shelf, 1.43 m | 6.0 % | nothing | to -50 (its upper arm) | the post |
| at a low shelf, 0.49 m | 4.7 % | clear to -36 (its gripper) | to -50 | the post |
| at the floor | 3.6 % | clear to -44 (its gripper, its soles) | to -50 | the post |

The head's corners hide the rings under -45 deg between the four
directions; the shoulders and the arms hanging hide nothing. The rays are
cast from 5 mm under the unit, as the C1's were from 6 mm over its puck:
from inside the housing every ray hit the housing (#386). The unit touches
nothing in the sim, as the C1 did not.

**The model** (`perception/lidar.py`, `tests/test_lidar.py`). A frame is 30
rings 2 deg apart, -52 to +6, 360 bearings each, 10,800 rays in ONE
`mj_multiRay`: its manual has a 0.1 s frame covering the field as "a 32-line
mechanical LiDAR" does. Each return carries the sheet's range sigma (4 cm at
0.2 m to 2 cm at 10 m, a line between) and 0.15 deg on each angle; 2 % drop
out (the C1's model's, carried); one nearer than 0.1 m is no reading, as a
self-hit is none. Its own robot's returns are dropped, another robot's
split out for the front stop (#316), its noise its robot's own
(`imu.seed_for`; the C1's model drew both robots' from seed 0) and kept
across a restart. Laid flat, a bearing reads the nearest surface in the
band by the median of its returns within 3 sigma of the nearest: a dozen
rings or more strike a wall on each bearing, and the least of their noisy
ranges read every wall 1.7 sigma short, 6.4 cm in the test's room, a map
cell into it; the median reads 0.5 cm short
(`test_the_band_reads_a_wall_where_it_stands`).

**The policies on the new head.** Flown again on it beside today's staging
(`humanoid_spike.py --shove`, `--track`, `--fall`), none retrained:

- *shoves*: every one it stood through before it stands through now, and
  two a step harder (from its left standing, 20 N*s where 15 was; crouched
  from its right, its feet held to 20);
- *tracking*: every served command to within 0.01-0.03 of before;
- *falls*: every get-up on a bare floor stood in the same 2.7-4.4 s, and
  every fall from a bow was caught and stood; in the house's corners, two
  stood sooner and one (left, off the landing's edge) now stays down where
  two of the four already did.

The walking policy's turn asked 0.5 rad/s is two gaits, 0.22-0.27 or
0.37-0.43 rad/s, by the draw of its senses' noise, on either head (8 draws
a head and a way; asked 0.4 or 0.6 the heads turned alike): the served
draw now takes the slower, and its test's bar, 0.3, sat between them. It is
0.2 now, still far over the 0.08 a policy turned that held its hips' yaw as
tightly as the G1's.

**What a frame costs** (`lidar_spike.py --cost`, 80 places in the house):

| ms a frame | the dev box | the deploy box, two runs |
|---|---|---|
| the level ring, 360 rays | 0.66 | 0.54, 0.74 |
| the cloud, 10,800 rays | 16.1 | 13.9, 16.7 |
| the band, laid flat from it | 16.4 | 14.2, 17.8 |
| a D435 frame, 8,400 rays | 12.4 | 10.7, 10.9 |

Almost all of it is the cast (15.0 ms of the dev box's 16.1); reading each
bearing by its median (above) costs 0.5 ms a frame more than by its nearest
return (1.8 ms against 1.3 to lay a frame flat, the same places on the dev
box, after this table was measured). ⚠ It casts with no cutoff, as the C1's
did: MuJoCo culls a plane by its centre's distance, and cut off at 8 m every
ray that met the house's floor -- its centre 19 m off at the loop's far end
-- read nothing (`test_the_floor_is_ struck_however_far_its_middle`); the
floor is under the band and outside the ring, so no flight below moved, and
the cutoff saved no time. The box's frames vary by a fifth from one run to
the next, on an idle box: it is a shared VPS. The pair's real-time multiple
there (`humanoid_spike.py --speed --band`, one core, each run its own
container, the trees interleaved):

| the pair, x real time | runs |
|---|---|
| today's staging, the C1 | 1.70, 1.55, 1.61 |
| this head, its level ring (as served) | 1.52, 1.65, 1.64 |
| this head, the band at 10 Hz | 0.97, 1.15, 1.08 |
| ...from 15 rings, 4 deg apart (5,400 rays) | 1.42, 1.17 |

One robot alone: 2.76 (staging), 2.77 (the level ring), 2.01 (the band).
Served on the level ring, the pair runs as fast as before. With the band
at 10 Hz each robot casts 108,000 rays a second, 0.28 s of the deploy box's
every sim second for the pair, and it ran at about real time: serving it
needs fewer rays a second. Half the rings gave 1.17-1.42, and stage 2
casts 15 at 10 Hz (Ben's call 3, 2026-11-01).

**The probe on its model** (`lidar_spike.py --probe`: #572's static probe,
the robot standing at every outdoor point 0.5 m apart, from its true pose,
its own body in the scene): the share of places a scan does NOT pin the
heading.

| | C1 (#572) | level ring | band |
|---|---|---|---|
| middle street | 68 % | 66 % | 0 % |
| sidewalks beside it | 38 % | 36 % | 0 % |
| the loop | 26 % | 26 % | 0 % |
| block sidewalks | 9 % | 9 % | 0 % |
| gardens | 7 % | 7 % | 0 % |
| rooms | 0 % | 0 % | 2 % |

The level ring is the C1's plane, to the place; the band pins every
outdoor place's heading, as #572's bare sensor did, with the robot's own
body in the way of its rays (in the middle street 49 % of places in all
three directions and the rest held along the street).

**The drift batch** (`humanoid_drift_spike.py --band`): #572's 14 flights,
the same draws and walks, on the new head with its level ring and with its
band, beside #572's on the C1 and its re-fly with the fences at 1.8 m:

| | C1 | C1, fences 1.8 m | level ring | band |
|---|---|---|---|---|
| docked | 4 of 14 | 14 of 14 | 7 of 14 | 14 of 14 |
| lost | 9 | 0 | 6 | 0 |
| middle street: scans fixed in every direction | 0 % (82 % nothing to match) | 99 % | 0 % (84 %) | 95 % (5 % held) |
| gardens | 44 % | 100 % | 44 % | 100 % |
| sidewalks and outer streets | 17 % (59 % held) | 70 % (30 %) | 9 % (71 %) | 100 % |
| the rooms | 94 % | 100 % | 99 % | 100 % |

With the band no flight was lost: its belief was off by a median 0.11 m as
the explore ended (the level ring's 0.39, the C1's 0.42), every trip ended
home within 0.1 m, and the worst error anywhere, 1.66 m, was a fall's
get-up in the living room, put right by the wide search 5 s later. The
band was flown at each of its three versions -- binned by the bearing its
ray was cast on, then by its own bearing levelled, then read by its median
-- and docked 14, 13 and 14 (the one lost, a walk the planner found no
route for), off a median 0.29, 0.16 and 0.11 m after the explore. The level
ring lost 6 where the C1 lost 9: one is the other, the street blind to
both, a flight's luck on where it walked.

**The station's batches** (`station_spike.py`, staging, the level ring and
the band):

| | staging | level ring | band |
|---|---|---|---|
| `--approach`: docked and unplugged | 15 of 15 | 15 of 15 | 11 of 15 |
| `--lost`, belief 0.4-1 m wrong: docked | 16 of 16 | 15 of 16 | 15 of 16 |
| `--walk-in`, from 8 places, explored 120 and 240 s: docked | 16 of 16 | 14 of 16 | 16 of 16 |

Staging docked all 47; the level ring missed 3 (a search begun 0.95 m off,
where #572 found the search reaches about a metre and staging found it;
a walk from the lobby across the street out of time, #572's known way to
fail; and a seat it stepped back and forth round, 4-8 cm either side, in
strides of about 10 cm, until its steps ran out); the band missed 5, four
at the dock and a search begun 1.03 m off. Over the band's three versions
the dock missed 5, 2 and 4 of the 47, and the same 15 starts docked 13, 15
and 11: which start misses is a draw, the failure the same.

**With the band, the dock is where it is not yet right.** Each miss there
either pushed the plug into the funnel three or four times and the port
never took it, or never lined up on the seat. Flown again, one of the first
version's misses: its matcher moved the belief 264 times through the dock
against 77 on the level ring, 38 cm of moves against 19, at the station's
near returns (its box, its cup and plug, 0.3-0.5 m off); the seat it steps
to and the reach to the plug are both placed through that belief, and it
took a seat 3.4 cm off and missed from it every try. Made its own to the
LIDAR -- held or parked -- the plug's returns gone, the same start docked in
one push; owned only while the hand held it, it missed again: it is the
station's near returns under the matcher, not what the hand carries; read by
its median, the near returns' bias gone, it misses as often. Near, the part
is noisier than the C1's model was (its sheet's 4 cm at 0.2 m, against 1.4
cm at 0.4 m), and the level ring's one seat lost (above) is the same
failure, once.

**The energy table is not re-priced here; the reserve is.** On this head
the explore drew 173.8 W (12.35 Wh over 256 s; staging, the same day,
164.8; the table carries 166.4) and a charge put 339 W net into the pack
(the table 345). The hand jobs could not be priced: the drawer's and the
door's walks across the street ended off their counter -- the drawer's
procedure computed a 0.62 m sidestep past its verb's 0.5, the door's never
saw its tag -- and staging's drawer did the same the same day (its door
went in and passed), #572's drift, which the band ends; the cube's reference ended with the cube still
in its hand, 4 cm over its mat, and the chest's first step then refused to
carry it. Stage 2 re-prices the table once its walks localize. The reserve moved now, on the pieces its comment
sums (`humanoid.world.RESERVE_WH`): the dearer travel with 6.2 W over its
222 s, the dock's leg and the plug-in twice at the 93 W it now draws
standing, 17.9 Wh, carried as 18, from 17.5 (its own dock leg ran out of
time 10 m short, as the C1's had).

**Not done** (#581's stage 2): the band served, through the believed pose
and the IMU's tilt, so a bowed robot maps and matches; the D435's low
layer kept only for what the LIDAR cannot see by the feet, measured; every
reader of the grid checked against it; the matcher's reach read again; the
dock steady under the band; and the band cut to 15 rings at 10 Hz (Ben's
call 3), the drift batch flown again on it -- `MAP_EPOCH` bumped at its
deploy.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, joint rates, contact
   list, `ncon`) — vibes don't bisect. Print the believed pose next to `data`
   before theorising.
2. **Render a filmstrip** (offscreen `Renderer`, 12 tiled frames) when
   numbers confuse — "standing on its tail" was invisible in scalars. Pick
   the camera by sweeping azimuth at the moment of contact, and aim a *free*
   camera at the working point; a tracking camera on a module body hid a
   16 mm seed in all eight azimuths.
3. Bisect one variable at a time, and assert that programmatic XML patches
   actually applied (`assert count == 2`) — a silent no-op once produced
   identical "before/after" results.
4. When symptom-fixes keep trading one failure for another, stop tuning and
   **measure the force balance directly**: `mj_contactForce` per contact,
   summed as torque about the COM, named the rover's caster as the yaw-brake
   in one run after days of plausible theories.
5. **Consult reference models** (MuJoCo Menagerie; the caster idiom was
   sitting in Stretch's XML all along).
6. **Decompose before you fix**, and a **discriminating experiment beats a
   hypothesis**: navigated-vs-bare and drawing-vs-not each localised a fault
   in one comparison, and doing the thing twice found the third.
7. A loop that never ends is named by walking the day routine's generator
   chain (`gi_yieldfrom`) in a reproduction.
