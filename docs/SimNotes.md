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
(Parts.md). ⚠ A drive's line is a damper too: its back-EMF, K²/R. The
gripper's jaws were heavy enough for their PD's stiffness and not for their
line, and MuJoCo integrates a clipped drive's line explicitly, so a shut's
force flipped between its clips every step ("The gripper's drive on its
line").

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
replaced, table in ActivityPattern.md §3.4). Every plate is flush with the
floor since #684, read off its load ("A flight that never got up"); the
sprung pad below was every plate's before it. Three measured rules the
pattern doc carries:
- **Hysteresis**: one wheel crossing of the sprung pad flipped a bare
  threshold **4** times, a hysteretic one (6 mm on / 3 mm off) **2** — one
  press, one release, the floor; a humanoid walking across the flush plate,
  175 and 71 over 15 crossings (`LOAD_ON_N` 100 / `LOAD_OFF_N` 50).
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
floor, resting, or in this robot's LIDAR's sight, as its body ("A robot
lying down was avoided where it said it was", "A robot at work kept off
where its body stands").
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
it decides, and before its day ends there (#594: DONE 0.37 m off it, a
robot stood in the other's way in until asked aside), with
`_leave_rack_routine`, and
checks that it got there (`RACK_CLEAR_M`, `CLEAR_SPOTS`; History says where
it ended); one that stays within its reach for `RACK_LINGER_S` doing nothing
is logged `STATION: lingering`. A taken charge seat is waited for
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

**On the humanoid** (#580): the middle of its footprint is the box round its
geoms while its get-up holds its joints, and its soles' middle on its feet
(`HumanoidBody.footprint_centre`). The disc first went round its soles, so
its head and shoulders, up to 1.5 m away, were outside the disc, and the
other robot's plan ran through them. The get-up takes the joints 0.1-0.8 s
before the trunk passes `TOPPLE_TILT_RAD` (six shoves in the house), so the
disc is round the middle the whole time the robot is down. Lying after the
falls the get-up took, the body reached 0.69-0.92 m from that middle
(`humanoid_spike.py --fall`, its lying reach; lying straight, 0.80). Its 1.55 m
from soles to the LIDAR's roof is more than the 1.4 m the disc was sized
for, so `DOWN_ROBOT_CELLS` is 24 (1.20 m): that reach and the walker's
half-width. With the standing disc's 10 cm as well (26 cells), two walks of
four past a robot lying limp in the living room never got by, where at 24
all four did, none touching it.

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
±6° in whole cells and degrees, then a fit from its best) runs when a fit
disagrees with its map -- a pose already past the fit's reach -- or, since
#681, slid past `MAX_STEP_M` or only partly agreed with ("The storeroom
keeps an error", below: 11-37 ms each); a robot lost beyond that window has
its fixtures (#476), and #381's loop closure.

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
m, and the rare search): 6.7 ms of a sim second per robot at 10 Hz. Since
#681 a doubted or slid fit is searched too, ~25 ms a search, at most once in
10 matched scans (below). Fusing
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
explore, and found again"); a match the walls only partly agree with is
searched too (#681, "The storeroom keeps an error"); and the first trip
through new territory carries its own drift into the map it lays (0.2-0.3 m
by the lab, ~1 m round the street loop), which later visits match to but do
not correct (#381's loop closure).

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
moved and the mind did not decide it. A restart's save waits it out. A
stand-up or a forgotten map (a true death) drops it unsaid
(`HumanoidWay.drop_aside`, #580): left under way, a humanoid stood up walked
on to the spot it had been stepping to.

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
  (`_near_field_step`); standing or walking, it is where it says it is
  unless the other's LIDAR sees it (#675, "A robot at work kept off where
  its body stands"), and held for;
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

**How it stands.** The policy drives the legs' twelve joints and sees every
encoder -- the arms' and, since the waist (#560's stage (c3), "The waist
built"), the waist's -- so it knows where its arms' mass is. In training the
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
- *A turn is the hips' yaw alone* -- the waist (#560's stage (c3)) is held
  home while it walks. With the hips'
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
and `-Getup-Head`, and the served get-up's `-Getup-Wall3-Mirror` since #679,
below; `humanoid_spike.py --fall`, `--view fall`, `--view
getup`). Without a get-up a fall is today's `stuck` death: a heart, and on
a served world a stand-up 300 s later. The get-up policy takes every joint,
arms too, once the IMU reads the trunk tipped 35 deg past the posture its
legs hold (below: a bow is no fall) or the pelvis has sunk under 0.40 m
over the sole it stands on (shoved hard from the front, the standing
policy folded into a squat 32 deg off upright, and the tilt alone never
fired), and hands back once it has stood still 0.5 s. Shoved 45 N*s each of
six ways on a bare floor, by our grading off the world (the waist's body,
11-13; a get-up's peaks to its first stand, the waist held home as the
mission holds it):

| falling as | stood | the head struck at, m/s (the floor's push back, kN) | gears past their part's peak | the hardest into a hard stop |
|---|---|---|---|---|
| the standing policy | never | 5.0-5.4 (1.6-8.9) in four falls of six | 4-8 joints, to 4.3x (an elbow) | 577 N*m |
| limp | never | 0.4-6.0 (to 8.5) in all six | 0-4 joints, to 4.2x | 384 N*m |
| the get-up policy | all six, 2.9-3.7 s | 4.0-5.4 (to 5.9) in two; in four not at all | 1-8 joints, to 1.3x | 570 N*m |
| getup3, served until #679 | all six, 3.1-3.8 s | 0.1-6.0 (to 4.6) in three | 3-9 joints, to 1.7x | 416 N*m |

From lying it stood in 2.2 s off its back, 2.0 off its front and 2.9 off
its side (getup3 2.9, 2.3 and 2.9). The kilonewtons are MuJoCo's soft contact -- the shape of a
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

In the house, shoved the same four ways at five places and read on #684's
clock (watched 25 s), it lives through 18 of 20: it dies at the landing's
edge, shoved forward and back (getup3 17, down on the flight shoved left
too). A fall off the landing is a `stuck` death still, and so is one that
leaves it on its side facing a wall or the counter within 0.1 m of it
(below). The stuck budget is 20 s, as
the quadruped's was (`HumanoidBody.GETUP_BUDGET_S`). The mission waits out a
get-up before handing the legs to a walk (⚠ a fall mid-walk with the arms
home found them at rest and took the legs off the floor; one on the step
that brought them home went to the walker, #520) or back from one (⚠ a
fall while the walker settled went to the standing policy at the settle's
end, #520), and on standing again forgets its map of the ground and starts
its height again over the ground it stands on, as at boot (⚠ kept through
a fall down the flight, the height was 0.41 m off at the foot for good).

**Getting up beside a wall or the counter** (#679; `--fall`'s counter and
lying tables, `Pluggy-Humanoid-Getup-Wall`). The get-up trained on open
floor alone (getup3, served until #679) never stands a body lying on its
SIDE with its FRONT to a face. Laid along the storeroom's counter and its south wall with the
truth -- the joints at the stand's, its nearest part 0, 0.1 or 0.25 m off
the face -- it stood from every other way of lying (its side with its back
to the face in 2.8-3.5 s, its back in 4.0-10.2, its front in 2.4-4.8) and
from that one never in 25 s, at the 0.72 m counter and the 2.4 m wall alike
(0 of 18): it rolls onto its front to push up, the face stops the roll, and
it pushes again for as long as it lies there. That is how the counter's
falls of 11-08 lay: flown again on the build that fell (#646's thin grip
pulled the robot over, which #647 stopped; the get-up's code and file are
unchanged since), it stood within 20 s in 0 of the 14, one at 30.4 s. Shoved
45 N*s from each prop's work spot, it went down in 20 of 24 and stood within
20 s in 15; the five that did not stayed against the counter, three of
them shoved to their right. Fine-tuning getup3 on today's body with a box
beside it taught, each flown in the served physics but the last:
- ⚠ *A lean is no stand*: paid from 25 deg off upright, as on open floor,
  it stood leaning on the counter 17-20 deg off upright, and the served
  get-up, which hands back at 10, never did -- neither up nor past 60 deg,
  the get-up holding the legs for good. The stand is paid within 10 deg and
  touching nothing of the box.
- ⚠ *Keep the noise*: with no entropy bonus the action's noise fell from
  0.23 to 0.06, and a get-up that stood at the stand's height 3 deg off
  upright shook at 0.9 rad/s and was never handed back. The training physics
  cannot say so: there a standing robot turns on its soles at ~1 rad/s,
  getup3 too.
- ⚠ *A lazy mean*: on today's body getup3 pays 4-7 times the torque charges
  it paid on its own, and the fine-tune cut them by not trying: in the
  training physics its mean action lay still on its back beside the box in
  136 of 256 starts that getup3 stood from in 252. Staying down is charged
  (`down`).
- ⚠ *A one-sided get-up*: fine-tuned, it rose off its back by rolling to its
  left, and lay still beside a box there (in the training physics 10 % stood
  with the box at its left, 100 % at its right; getup3 96 and 100). Each
  sample and its mirror image trained together (`-Getup-Wall-Mirror`) kept
  it even: 100 and 100.

The first round's pick, getupwall10's iteration 8450 (from getup3; never
served), chosen on the training physics alone (`python -m humanoid_train.scan`,
ten starts), stood from the counter's falls by catching the falling body
against the counter rather than lying down along it (getup3 from none), and
from a side facing the face only 0.25 m off it; but shoved sideways in the
hall and left or front-left at the chest it SITS, or lies propped on the
wall -- never past 60 deg, the get-up holding the legs, which since #684 is
the `stuck` death at 20 s -- and its head struck 1 m/s or more harder than
getup3's in 16 rows. Later checkpoints stood more often from a side facing
the face and lost the back beside a box on its left again.

**The wall get-up's second and third rounds** (#679; `--fall`'s tables read
on #684's clock, `-Getup-Wall2-Mirror`, `-Getup-Wall3-Mirror`). Flown on past
its first stand as the mission flies it -- the legs back to the standing
policy, the watch on, a fall taken again -- until the `stuck` clock settles
(`StuckClock`; the tables had stopped at the first stand, or watched on with
nothing to take a second fall), getupwall10 regresses on SIX of the 151 falls
getup3 lives through, not four: lying on its back, head back,
against the counter, it stands in 5.1 s with its back to the counter, the
standing policy sways back onto it 2.7 s later, and it ends on hands and
knees. getup3 lives through 119. The three sitters hold one posture (the
left hip flexed 2.1-2.8 rad, the right turned 0.65-1.0) -- a fixed point of a
blind policy, as the prop on the wall and the hands and knees are; in the
hall it sat down with its trunk never past 32 deg. Started there (`posed`: where it
ended each fall, and the fall as the get-up was handed it,
`humanoid_spike.py --fall-ends`), the training physics' mean action learned
to rise from a sit in half the starts (0.04 to 0.50-0.54); a wall far in the
way a body tips taught nothing, the training's clean tip standing 32 of 32
against it. That round's pick stood from the hall's left and the counter's
two, but stood faster with its soles 0.53-0.65 m apart (getup3's 0.32), and
given that stance the standing policy tipped within 2 s in nine falls the
get-up then never stood from: 14 regressions. ⚠ A get-up trained never
handing back stands where the standing policy cannot keep it. Trained handing
back (`HandBackAction`), in the training physics:
- ⚠ *Its stillness is not the training's*: waiting for the trunk's turning
  under the served get-up's 0.3 rad/s, the action's noise never let it settle,
  and nothing was handed back.
- ⚠ *A get-up keeps the legs where it can*: with the stand paid whoever held
  it, both runs learned within 450 iterations to stand just outside the
  hand-back; paid only handed back, within 170; with the wait charged too,
  within 140 -- handed back, the get-up's own joint and torque charges read
  the standing policy's stepping against the training physics' ~1 rad/s turn
  on the soles (an episode's -18 against -0.5). Paid only handed back and
  charged only while it holds the legs, it handed back in 0.78-0.86 of every
  second it could through 1000 iterations.
The served get-up since #679, `models/humanoid_getup.npz`, is getupwall14's
iteration 9850 (that round's pick carried on handing back, chosen in
`-Getup-Wall3`'s physics before the bar was read). It loses none of the 151
falls getup3 lives through on the clock and gains 24 (lying along a face 57
of 63 to getup3's 39, the counter's shoves 20 of 20 to 15, the house's 18 of
20 to 17) -- all six of getupwall10's back, the hall shoved back standing in
2.9 s, its soles 0.35 m apart and 0.26 m 0.4 s later -- and stands from the
counter's falls inside the 20 s (2.8-14.8 s after the take). Lying on its
side facing the face, touching it or 0.1 m off, it still never stands (the
six of the 63 it does not). Its slowest of the 151: shoved left at the
chest, it sits, and rises 18.2 s after the take. Its head strikes in fewer
of the rows (58 to getup3's 68), softer at worst in every table but the
bow's: there 7.0 m/s, the rim of its cap, from the stand shoved forward at
30 N*s (getup3 1.3), and 10 falls from a bow 1 m/s or more harder. On 130
falls outside the bar (the shoves at 35 and 55 N*s, the bar's 45; laid 0.05
and 0.18 m off a face), never trained on, it lives through 110 to getup3's
93 and loses 3 of getup3's: shoved right at 35 N*s it sits in a posture its
posed starts never held, and one fall back down the flight. Read again on
the tree it is served on (over staging's 13ddef7, beside getup3, on one pod
of the CPU type the rounds were read on, 11-13), the 151 and the 130 came out
row for row as they had; the counter's 14, flown on that pod for both (the
rounds had flown them on the dev box), part from the dev box's flights by
the CPU (one, n1425-1, does not fall there), and it stands from the 13 that
fall (getup3 from none).

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
policy alone, graded by us, then as served; times from the shove; the
waist's body and its standing policy, waist16, 11-13):

| posture | falls | taken first by the tilt, the collapse | ahead of the strike, s | shoves it stood through called falls: now, off upright | stood, served |
|---|---|---|---|---|---|
| the stand | 10 of 16 | 7, 3 | 0.24-0.43 | 0 of 6, 0 | 10, 2.9-4.2 s |
| -0.20 m, 0.55 rad | 8 of 16 | 6, 2 | 0.05-0.42 | 0 of 8, 0 | 8, 2.8-3.8 s |
| -0.22 m, 0.80 rad | 8 of 16 | 6, 2 | 0.06-0.43 | 0 of 8, 8 | 8, 2.7-3.5 s |
| -0.25 m, 0.95 rad | 8 of 16 | 6, 2 | 0.06-0.45 | 0 of 8, 8 | 8, 2.6-4.9 s |
| -0.25 m, 1.10 rad | 8 of 16 | 6, 2 | 0.06-0.50 | 0 of 8, 8 | 8, 2.7-4.1 s |

The most a shove it stood through tipped it past its posture was 7.2 deg.
⚠ A forward fall from a bow folds at the knees, and THE COLLAPSE TAKES IT,
0.05-0.08 s before they strike: lying on its face is only 27 deg past the
deepest bow, so the tilt alone would never fire there. As served the head
struck in 18 of the 42 falls at 0.3-7.0 m/s (getup3 15, at 0.7-6.0; on the
standing policy alone, in 35 at 0.5-8.1): the get-up's training tips its
falling starts over from the stand, never from a bow. `reach_routine` says True only of a hand
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
its turn 24-37. Getting up off its back, its front and its side (the served
get-up, #679; 11-13) keeps every leg part under its peak -- the hip's pitch
to 118 N*m off its front, 98 % of the RS04's (getup3: 88, 73 %) -- with RMS
to 94 % of its rating over the 2-3 s (getup3 72 %); holds the waist home at
its RS03's whole 60 from every start, its RMS 116 % of its rating over the
2.2 s off its back (getup3 to 96 %); and spends the arms whole: the upper
arm's twist and the elbow (RS02) at their 17 N*m peaks from every start,
the shoulder's pitch (RS03) at its 60 off its front and its side and its
roll off its side. Past a rating only for a burst of seconds; the arms have
no margin for a body heavier than this one.

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
  where a path planned with no slack stopped at 57 deg. A path is planned
  for a hand that is out on the robot AS IT STANDS -- its model posed off its
  senses, its arm as its encoders read it -- and in another posture on a
  statue standing as it reads itself (#627; 3.5 cm ahead on a statue over
  the soles' middle before), with 2 cm on the points further from the
  shoulder (`LIVE_MARGIN_M`): from the statue, the path back down from the
  top solved 6 cm short of where the hand WAS.
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
  `STEP_STOP_S` (0.22 s), reaches the step (below: a stride moves the stance
  about 5 cm, and a short ask may take none).
- *A turn at 0.6 rad/s sprang back 6-8 deg* as the legs changed hands; at
  `TURN_W` 1.0 it lands within 2 deg of the turn asked, and its heading is
  read within 0.3.
- *Nothing is measured until the walker has the legs.* Out of a deep bow
  the pelvis came 16 cm forward over the feet as the arms came home, which
  a walk measured from before it counted as walked (ok, without a step),
  and the hand-over outlasted the 1 s a walk is given to get anywhere,
  which read as `blocked` on a bare floor.
- *A walk's first window is 2 s* (`STEP_START_S`, #560), then 1 cm in each
  second after. Handed the standing policy's stance after a reach -- 23 cm
  between the soles' middles, the walker's own 18 -- the walker loads its
  right foot first whichever way it is asked: a sidestep to the left lifts
  within 0.25 s, one to the right slides 0.6-1.1 s first, where the legs'
  reckoning cannot see (it read 0.6-3.9 cm of the 1.9-6.1 they went), and
  in a 1 s window 2 of 7 at the table's set-outs read `blocked`. The retry
  that went had begun from the walker's own stance, which the slide had
  left; waiting, or the walker held still first, changed neither. ⚠ Not a
  foot's lift, judged off the robot's model of itself: through the slide
  it read the soles 4-5 mm apart when they were under 2, and 2 when one
  had lifted 7. ⚠ The slide is in no report: such a step lands where it is
  asked, but its `step.dx/dy` read 3-4 cm short of 21.
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

**Short asks and small turns (issue #649).** A walk's stop rule watches
the pelvis (above), and the pelvis moves before the feet do: for the
first 0.3 s of a walk it leans over planted feet, a foot lifts after
that, and each landing moves the stance about 5 cm. Flown on the bare floor
from four stances (stood from the boot, after a 12 cm step, after a reach
withdrawn, after a 29 deg turn), each at three noise draws:

| asked | ahead | back | left | right |
|---|---|---|---|---|
| 3 cm | 0.0-3.9 cm | 0.0-0.1 | 0.7-2.5 | 0.0-5.8 |
| 4 cm | 2.1-6.3 | 0.0-0.1 | 0.7-3.2 | 0.4-7.8 |
| 5 cm | 6.1-7.2 | 0.1-5.1 | 0.9-7.8 | 1.7-9.2 |
| 6 cm | 6.4-10.2 | 2.7-8.9 | 2.8-11.8 | 4.5-11.8 |
| 8 cm | 9.8-13.5 | 5.3-10.9 | 3.5-15.8 | 6.2-14.8 |
| 10 cm | 12.1-14.6 | 6.6-11.0 | 9.7-16.3 | 6.3-15.4 |

Every one of those read `stepped`, though 3 and 4 cm back never moved, a
walk's first 0.3 s being all lean. The same flights' verdicts now:

- *A walk asked at least `STEP_LEAST_M` (1 cm) that its own legs reckon
  went under it reads false* (`too short`, "your step was too short to
  walk"). Of 312 flights asked 8 cm and under, none reads `stepped` having
  truly moved under 1 cm, and 73 read false, 3 and 4 cm back 12 of 12. ⚠ THE
  WORD IS THE LEGS': 7 of the 73 truly went 3 cm or more, five left sidesteps
  of 4-5 cm straight after a turn (they went 3-8 cm and the legs read them
  back 0.4-1.4) and two backward asks of 7-8 cm (they went 7-8 cm and read
  under 1), and a foot's lift does not tell them: their feet slid, lifting
  under a millimetre, as in the runs that truly went nowhere. Asked over 8
  cm, none of 264 read false. A walk's physics did not change: its stop rule
  is as it was, and only what the verb says of it is new. Under `STEP_SHORT_M` (6 cm) the text says a walk may take no stride
  at all, and it may stop a few centimetres off; "at least a stride, about 10 cm"
  was never true (one landing moves the stance about 5 cm).
- *A sidestep's `step.dy` counts short*, under what the legs can see (the
  walker's slide before its first lift): to the left 1.2-3.7 cm short over
  asks of 3-30 cm, to the right 0.6-2.4, ahead and back within a centimetre
  (`STEP_DY_SHORT_M`). ⚠ AND MUCH MORE LEFT AFTER A STEP AHEAD
  (`STEP_DY_LEFT_AFTER_M`): a step that goes ahead and then left (`_step_walks`'
  two walks, the second off the first's carry) read its left 1.3-19 cm
  short, 3-8 on average over asks of 3-10 cm ahead and 10-20 left, its
  true left going past the ask by 2-6; to the right 0.2-2.4. Stated, not
  corrected: it varied with the start and the ask.
- *A step's turn was stopped on the pelvis's heading, and the feet do not
  follow the pelvis's last degrees*: asked 3 deg, the stance turned 0.3;
  asked 7, 1.3 left and 2.2 right; asked 10, 5; asked 12 and 15 to the right,
  5.8 and 9.1 (6-13 over the starts), up to 7 short, and within 2 past 20
  (5 past 45 to the right).
  The turn is now closed on the STANCE's own heading (`_turn_routine`,
  `TURN_*`): measured from where the stance stood when the step began,
  (⚠ NOT after the hand-over: the stance moves up to a degree in it, and
  measured from there it landed a degree short), the walker's own turn first
  past `TURN_BIG_RAD` (6 deg) then short pulses of `TURN_W`, each read after
  a settle (0.5 s; 0.7 after the walker's own, whose last foot is landing:
  read at 0.3 a pulse moved the stance 7.8 deg where from rest it moves 2-3),
  until it is within `TURN_TOL_RAD` (0.75 deg), at most `TURN_PULSES` (8).
  It lands within a degree for 3 deg (2.0-3.3 from most stances, 1.6-3.8
  right after a big turn, the feet still recovering), within 1-2.6 elsewhere
  (12 flights an ask), and takes 2.5-4 s where a small turn took 1.4. A
  turn whose feet will not follow runs out its pulses and reads `turned
  short` (`TURN_MISS_RAD`); none of 144 flown did, every one ending within its
  0.75 deg. Its closing is over the step's patience (`TURN_CLOSE_S`).
- *What a turn pulse does to the feet*: at 1 rad/s, 0.08 s turns the
  stance 0.7-1.3 deg, 0.10 s 1-1.5 and 0.14 s 2-4 (more once the gait
  stirs), with no foot lifted; past 0.2 s a foot lifts and lands and the
  stance jumps 7-14 deg, so a pulse is never longer than 0.14 s
  (`_turn_pulse`). Its reckoned stance heading is the truth's to 0.2 deg.
- *The references and the station, flown beside the tree before*, one pod
  (EPYC 7B13, osmesa), each flight its own noise: the hands checks at
  set-outs 0-31, lid / drawer / door / cube / table 26 / 21 / 25 / 17 / 27 on
  both trees, the same flights in every number (their steps ask no turn); the
  served drawer, door and chest from the tour 29, 28 and 31 of 32 on both; the
  table from the tour 12 of 32 against 9; and the station 39 of 39 from
  round the standoff and 140 of 140 seats on both. Across their flights, the
  real turns' error (asked against the stance's truth): 0.8 deg at the median
  under 6 deg asked where the tree before read 1.5, 1.1 against 1.4 from 6 to
  20, 0.7 against 1.3 past 20. A `too short` walk happened once, in a table
  draw that failed on both trees.

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
  home), the cube 3, no falls. Its cube's last `withdraw` now said "held"
  at 3 set-outs, a jaw on the counter's edge (where, below). The old walk
  home ended on the counter there too, or on the cube's mat, and said ok.

⚠ The side way assumes the space at the robot's side is clear, as
straight assumed the space under the hand was: its hand goes out to 0.74 m
from the body's middle, 1.0-1.1 m up. Nothing checks either against the
depth picture, which does not see beside the robot.

⚠ *Lowered in the hanging arm's line, it met a counter the robot stood
turned to* (#560). The cube check steps in until its counter's face is 3-4
cm ahead of the toes, and the step turns the stance 3.5-4 deg on its own;
the side way's roll came in 9 cm ahead, out at 0.55 m to the side, where
the turned face was nearer, and its pads came down on the top's edge, the
roll in dragging them over it for 10 s at 30-48 N and turning the robot
20 deg. The roll now comes in with the shoulder pitched back
(`HOME_BACK`), then the arm forward to its rest: on its model that clears
a counter at the toes turned 10 deg to it.

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

**What a moving hand's force reads** is #631's, below: taken as still, as
it was here, a carry's start read 11-17 N ahead.

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
rate and no oftener, from the move's first command -- a hand's force from
0.1 s into it (#631, below).

## A moving hand's force, its motion taken in (issue #631)

From 11-03 Luca opened the storeroom's drawer with `reach("right", 0.290,
y, 0.85, ..., until="right.fx", above=3.0)`, reading each stop as its hand
meeting the drawer's face. Every stop came 0.18 s in, its hand by its
side, and it shut its jaws on air thirteen times. The whole body's balance
(#547, "A hand's force off the whole body") took the body as still, so the
torques that swing an arm out and sway the trunk read as a push.

**Measured** (`scripts/humanoid_spike.py --moving`: 18 reaches, carries and
pulls in free air, either hand, the humanoid alone on a bare floor, the
reading taken as still and as served):

| | taken as still | as served |
|---|---|---|
| a reach, after its first 0.1 s | 11-14 N (at 0.25 s, and where its way out bends, 1.46 s) | 6 or under; 9.9 to the floor in a deep bow |
| a carry or a pull, its first 0.1 s | 13-16 N | as still (below) |
| a carry or a pull, after | 3-9 N | as still |

**The physics first.** Given the simulator's own speeds and accelerations
(a probe's, never the robot's), the same balance read 1-4 N on Luca's
reach where it read 16: the reading is the body's motion and nothing else.
The arm's motion and the trunk's must be taken together; either alone read
worse than neither, since the trunk answers the arm. The legs' motion
moved one move's worst of 18 by 2 N, and is left out.

**What the robot can know of it** (`arms.Motion`, `SelfModel._held(motion)`):
the trunk's turn rate off its gyro and that rate's change over a control
period; its acceleration off its accelerometer, the specific force turned
by the tilt the model is posed with, gravity taken off; and each arm
joint's and the waist's acceleration, its driver's speeds differenced over
two physics steps (`MOTION_STEPS`), sampled where the reach steps
(`JointSense.speeds`, which draws no noise). The balance becomes `M a + C +
G = S^T tau + sum J^T w`, the arm's own reading too. The speeds' field
bounds it: a count is 0.01-0.024 rad/s, a few rad/s^2 differenced over one
step, and Luca's reach read 8 N over one step, 9.4 over a control period
(5 ms late), 5.5 over two. Smoothing the arm's set-points instead (a
bounded acceleration) took the reach's swing to 4 N and left its way in
and its hold at 6-7 while slowing the reach by half (4.0 s to 6.2); an IK
solved to 0.5 mm in place of 2 moved nothing.

**Only what the robot reads takes it in; the body's own loops never do.**
A hand's force is read twice (`Reach._feel`): as still, as before, for the
body's own loops (`Reach.force`: the reach's stop on a push, the ease
before a shut, the force move's loop), and with the motion in for what the
robot reads (`Reach.sensed_force`: `Body.hand_force`, a procedure's `read`,
its stop conditions, a verb's report) -- and for that only while the hand
goes its own way, swung out or home, on its line in or out, never while it
holds or is held (`reach.HELD_PHASES`: held at its goal, pulling,
stopped, carried). The two were measured on one pod (Vast, EPYC 7B13,
osmesa) against today's tree at the same seeds, every hands check at
set-outs 0-95:
- **Into everything**, the motion cost the checks 18 set-outs of 480 for 5
  won (a sign test's p 0.01): the cube placed 46 of 96 where 53, set down
  3-6 cm off its mat's middle, and the door opened 65 where 69. Flown with
  the true force at the hand beside it (doors 14 and 17, drawer 12), the
  reading with the motion was the truer overall, but in a pull its error
  across the pull's way rose from 0.8 to 1.1 N rms, and a pull's give
  follows that reading across its way: pressing slowly on what it holds,
  the hand's motion is smaller than its estimate's noise. A quieter
  estimate (the IMU over the control period, a speed's flicker of one count
  read as none) read further from the truth: its lag cost more than its
  noise saved.
- **Kept out of the held phases**, still 7 for 1 (p 0.07), and 3 drawers
  fell: the approaches stopped on a push (`BLOCKED_N`) at other moments, and
  what came after parted.
- **Read by nothing of the body's** (this): every flight that reads no
  force sensor is today's, bit for bit -- the checks' 24 set-outs that had
  parted, flown both ways on the dev box, identical in their grades, their
  verbs' ends and their times; and on the pod, every hands check at
  set-outs 0-95, all 480 identical to today's.
The reach's stop on a push is 30 N against a phantom of 9 at most, and
needs no motion taken out.

**A move's first instant.** A carry or a pull starts from where the hand
IS, its command a step off its hold, and read still it reads 13-16 N for
its first instants and 9 or under from 0.1 s on. So a stop on a force reads
it from `axes.FORCE_FROM_S` (0.1 s) into its move; a hand pressing already
as its move begins stops that much late.

**What the robot is told.** Its hands' force sensors and the stop
condition's rule say that a hand moving through the air reads up to
`axes.MOVING_N` (12 N) with nothing touched, and that a stop on a force
reads it from 0.1 s into the move. The prompt's example stops at 15 N
(`GUARD_EXAMPLE_N`), over that and under the 20 the standing body holds at
a hand; at 10, its number before, a stop on Luca's reach's force stopped
it 0.23 s in. History says what each stop condition read
(`lifecycle.stops_said`), as practice's report does.

**What is true now:** what the robot reads of a hand's force takes the
body's own motion in as its senses have it while the hand goes its own
way, and reads it as still while the hand holds or is held; the body's own
loops read it as still, as before; a stop on one reads it from 0.1 s into
its move; moving through free air with nothing touched it reads up to 12
N, 6 or under on most reaches (`tests/test_guard.py` flies Luca's reach to its goal stopped
on its force at 12). ⚠ What is left is the estimate's: the drivers'
replies at the physics rate, or a momentum observer, would take it nearer
the 1-4 N the simulator's own motion reads, and might let a held hand and
the body's loops take it.

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
clocks, and its models of itself as their loops last posed them (the
LIDAR's band is laid through the odometry's: Webserver.md, "Parity").
Saved at the day loop's first pass and carried on in a new process,
its day was IDENTICAL through its whole explore (`determinism_spike.py
--resume-at 1`: 509 state samples to t = 260 s, 2026-11-11). ⚠ A
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

**Asked aside as an arm comes home** (#668). An arm whose command is at
rest is no hand out, so a humanoid whose arm the counter held on its way
home could be asked aside, and on 11-08 Luca's `withdraw` waited 2,191 s,
asked aside 17 times, its mind never asked, until a fall ended it. The
walker took its legs and nothing gave them back; and the posture's own loop
(`Reach._trim`), reading the walker's posture, carried the standing goal off
a standing posture that moves only while its own policy steps -- 0.020 rad
forward over a walk of a second or more, the edge of the band `settled`
reads, and 0.008-0.017 over a short step -- so the release's wait on the
two meeting never ended. A release now takes the legs back for the
standing policy once its arm is home or held (ending a step aside still
under way, or one asked as the legs come back) and waits on its posture
`STAND_BACK_S` at most. On a bare floor, its arm read home 6 s late and
asked aside meanwhile, it returns 0.8 s after the arm, where before it
never returned;
at the chest as Luca worked it, the ask landing while the counter held its
arm, the live build's withdraw had not returned 365 s on, and with the
hand-back (the trim gated too, below) it returned 1.7 s after the ask.
Where nobody asks, nothing changes: the four offered jobs from the tour,
32 draws each, replay staging's flights exactly. The trim is left running
under the walker. Trimmed only while the standing policy steps, every
walk's hand-back stood truer (over the 3 s after, the trunk read 0.10-0.24
deg RMS off its plan, where the drifted goal left 0.40-0.55; five walks on
a bare floor), but our table's reference, tuned on the drifted goal, laid
20 of 32 from the tour where it lays 27, all seven draws that parted going
the one way; the counter's three jobs read as before (31, 26 and 31 of 32,
on one pod).

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
the body, plugs landed 12-33 mm across the port and 7-10 mm low -- most of
it where the hand was sent, through a belief that moved (#596) -- and the
hand knows where it holds one to 1-2 cm (its encoders and backlash, and the
plug free along the Dex1-1's 80 mm pads), so the funnel round the port is 70 mm
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

![The plug-in and the unplug: from its standoff the humanoid walks in, takes
the plug out of its cup, plugs it into the port on its torso (16 s),
charges, unplugs it (26.5 s), hands it back to the cup and turns away, a
tile every 1.5 s (`station_spike.py --strip plug`)](figures/humanoid_station.png)

![Lost and found by its station: a humanoid 2 m out, its belief set 0.7 m
nearer the wall than it is, walks to where it believes the standoff is
(1.45 m off the station's face, the tags above its colour camera), sees
nothing, looks round by its head's camera and sees the station, walks
nearer by what it saw, puts its belief right off a look there, and walks
in again and plugs itself in, a tile every 4 s (`station_spike.py --strip
lost`)](figures/humanoid_station_lost.png)

**Lost.** No tags in sight from the standoff the belief walked it to, the
robot is lost (#476's rule), and the station, commissioned as the dock was,
puts it right: a look that fits the station is a FIX (`HumanoidBody._fix_at`,
the same as `anchor_at_dock`'s): the belief moved, and past the matcher's own
search the map laid on the old one dropped, and the loop's walk to the
standoff goes again on it. How it looks for the station is #572's stage 3
("The station found from far off"): by its head's camera, never by its
belief.

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
| `--lost`: a fresh humanoid 2 m out facing the station, its belief set 0.7 m wrong each way and 12 drawn 0.4-1 m off and up to 5 deg turned, the loop's own `go_charge` | **docked 16 of 16**: the 14 that saw no station searched and found it in 4-62 s, their fix putting the belief within 0.11 m; the two whose walks ended near enough docked as they were, their fix the charge's own (the search as it was then, stepping in by its LIDAR; by its head's camera, #572's stage 3, 15 of 16) |

**What is not yet.** The reserve's own dock leg gave up on the explore's map
and is carried by rate; one station a world. (A walk-in that pushed four
times and the port took none was #596's: "A plug-in that missed every try".)

![Knocked over plugged in: shoved backward, the humanoid falls, the plug
breaks away at the port and swings on its rope, and its balancer draws it
home into the cup about 2 s later as the robot gets up, a tile every 0.5 s
(`station_spike.py --strip fall`)](figures/humanoid_station_fall.png)

### A plug through a restart (#654)

A charge used to hold its run open until the pack was full, so only a deploy
ever restarted a robot with its plug in, and that never worked: a catch is a
weld switched on where the plug is, its anchor put at the plug (`_take`), and
`Station.restore_kept` gave a new process's model the catch's pose but not
its anchor. Compiled, the port's anchor is 1 m off. So on the first step after
a restart the weld read 1 m out and 3,100 N, the catch let go, and the plug
was thrown 0.9 m in 20 ms. Kept, the same restart replays bit for bit
(`replay` true) and the catch reads a steady 2 N.
`test_a_plug_in_its_port_rides_a_restart` pins it.

Flown on the served pair (osmesa, `$PLUGGY_CHARGE_SCALE` 30 so a pack fills
in minutes), one robot's charge cut by the run's end, and each row a new
process from that save:

| | |
|---|---|
| the run's end | plugged in at 53 s; at 170 s `the run's time is up at 45%`, the plug in; the process ended at 182 s, as the other robot's walk did |
| restarted, `charge` | no walk; 50 to 90 % in 111 s, `CHARGE complete`, one unplug, the plug home in its cup, no falls |
| restarted, `explore` | the plug out by hand in 8 s, then the walk; the plug home, no falls |
| restarted, walked off with the plug in, as the loop did | the plug pulled out past the port's 20 N, 3 s into the walk; the plug home, no falls (one flight) |
| restarted at 95 %, the station cleared before a decision | the plug out by hand in 8 s, then the walk home; the plug home, no falls |
| restarted, a `spot` then a `face` | the plug in through the `spot`, out by hand in 8 s for the `face`; the plug home, no falls |

A bow while plugged in, down to its deepest (1.1 rad, 63 deg), leaves the plug in, no
catch broken and no fall: what a `spot` looking lower does.

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
| `place_cube` | 0 of 8, not offered then | set down 2-7 cm from the mat's middle (2.5 allowed); **offered since #560**, its tries reading again, 28 of 32 ("The cube from the tour") |

No falls in the 32. The slowest pass took 313 s from the station, and each
job allows 16 minutes from the claim (three times it, rounded up). Priced
from the station with the place remembered (`energy.json`; since #560 the
dearest of three, the mall's, and on the waist's body since 10 Nov): the
drawer 7.50 Wh, the door 9.04, the chest 8.54.

**What came later.** The cube is offered since its reference placed it 28
times in 32 over the tour, with a minutes and a pay of its own ("The cube
from the tour"); why it missed before: "Setting a table". From the station,
every flight's belief ended 2.3-2.5 m off the truth, the street crossed on
a fresh map, and passed: the counter's places are laid in the map its walks
plan on. A lid let fall once in 8.

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
zero-rate update that had learned it would (the quadruped's, #425: the
humanoid had none until #593, below).

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

**What was true then, on the C1.** Indoors the belief held within 0.1-0.3
m, the matcher taking off the gyro's offset and the legs' 0.2-2 cm a metre;
outdoors the LIDAR saw the buildings' walls alone, and the belief came home
past the matcher's search by walking along the facility's outer walls, a
heading carried across the street, standing where nothing was matched, or a
fall whose get-up its odometry walked; the lost search found the station
only from within about a metre. What is true now: #581 gave the LIDAR the
fences (29 of 30 drift flights docked, "The band served"), and #572's stage
3 gave the search its head's camera, held the odometry through a get-up and
the charge's walk its time ("The station found from far off"); #593 holds
the heading of a robot standing where nothing is matched ("A robot left
standing when its run ends, and a heading held at a standstill").

## The 3D LIDAR on the head (issue #581, stage 1)

**What is true now.** The humanoid's LIDAR is a Livox Mid-360S, upside down
(Parts.md, "The LIDAR"): its field is +7 down to -52 deg. This stage hung
it from a bracket over the head, its rays 1.47 m up standing; stage 3
built the head round it, its rays 1.41 m up ("The sensor in the head",
below). Its LEVEL RING, the 360 bearings at 0 deg, behaves as the RPLIDAR
C1's plane did, a few cm higher. Its points
LAID FLAT -- per bearing the nearest surface 0.15-2.2 m over the floor,
through the IMU's tilt (`Lidar.band`) -- see the house's 0.9 m fences, the
couch, the tables and the beds: flown with them, the 14 drift flights that
#572 lost 9 of were 14 of 14 docked, as with the fences raised to 1.8 m,
and since stage 2 the map is laid from them ("The band served", below). A
full frame of 30 rings costs 14-17 ms on the deploy box, and two robots
casting it at 10 Hz ran at about real time (0.97-1.15x).

![The 3D LIDAR on the head: a humanoid walks from the living room out by
the garden's gate, across the street and through the lobby to the
storeroom, its map laid from its points laid flat; red, each bearing's
nearest return 0.15-2.2 m over the floor -- the couch, the fences, the
walls' lower halves -- and blue, the level ring, the buildings' walls
alone; a tile every 15 s and one as it arrives (`lidar_spike.py
--strip`)](figures/humanoid_lidar_walk.png)

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

**Its mount, this stage's** (stage 3 built the head round it instead;
"The sensor in the head", below). The unit hung upside down from a plate
on a 20 x 3 mm 6082 post up the head's back edge, turned on to the sensor,
its rays 10 cm over the head's top so the steepest ray cleared the head
ahead and to either side. Cast from the C1's height, 4.7 cm over the head,
the head hid every ray steeper than 31-35 deg, the floor out to 2 m. It
touched nothing in the sim. Standing, the robot hid 3.8 % of a frame of 30
rings (4.8 % of 15, stage 2's): the post's 3 bearings behind, and the
head's corners under -45 deg.

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

**What came after**: stage 2 served the band ("The band served", below),
and stage 3 built the head round the unit ("The sensor in the head").

## The band served (issue #581, stage 2)

**What is true now.** The humanoid's map is laid from its LIDAR's points
laid flat (`Lidar.band`, `HumanoidMission._scan`), 15 rings at 10 Hz (Ben's
call 3), and they are matched and laid while the robot is on its feet in
the posture its legs hold, bowed too (`maps`): per bearing the nearest
surface between 0.15 m over the floor, rising 2 deg a metre out, and 2.2 m.
The matcher leaves out returns nearer than 1 m (`MATCH_MIN_M`), a step walks
by its legs' reckoning (`step_routine`), and `MAP_EPOCH` is 4. Flown so,
with the review's fixes (below), #572's 14 drift flights and 16 more
docked 29 of 30, as the code before the fixes did (the C1 lost 9 of the
14), and the station 80 of 85 starts, as before them (the stage's 47
trials docked 43; staging 47, stage 1's band 42).
Two robots ran at 1.40-1.41x real time on the deploy box, the level ring's
1.68-1.78.

![The band served: a humanoid walks from the living room out by the
garden's gate, across the street and through the lobby to the storeroom in
78 s, its map laid from its points laid flat; red, each bearing's nearest
return in the band -- the couch, the garden's and the street's fences, the
counter -- and blue, its level ring, the walls alone; a tile every 15 s
and one as it arrives (`lidar_spike.py --strip`)](figures/humanoid_band_walk.png)

**The band's floor rises with range** (`BAND_RISE`). Walking, the IMU's
tilt -- its complementary filter, pulled toward the accelerometer, which
reads the trunk's sway -- is off by a median 0.8 deg, 1.8 at its 95th
percentile, 2.1-2.7 at its 99th and 3.6 at worst, over 120 s of the loop's
explore and a walk home; bowing slowly it is off 0.6 deg at its 95th
percentile, 52 deg bowed. A floor point 6.9 m out, where the -12 deg ring
meets the floor, levelled 1.8 deg wrong reads 0.22 m up, in a band whose
floor is 0.15 m: the floor rises 2 deg a metre out, clearing the 99th
percentile everywhere in the LIDAR's 8 m, and a fence is still struck to
the 8 m edge (`test_the_bands_floor_rises_with_range`).

**A bowed robot maps and matches** (`HumanoidMission.maps`). Its points go
in through the IMU's tilt and its model of itself while its trunk is within
12 deg of the posture its legs hold (`getup.tipped`, the fall watch's own
reading of a tip): walking it tipped up to 6.6 deg off upright; a bow is
the posture, never a tip. Gated on level, as the C1's plane was, a bowed
robot neither mapped nor matched, and Luca's heading turned 5.2 deg in 300
s at a counter (#572); bowed at a counter now, its scans are matched
(`test_a_bowed_humanoid_matches_and_maps_its_lidars_points`). Never while
the get-up has its legs: no model of a body on its soles is a model of one
on its side. ⚠ A body laid down between two steps, as a test does it, is
read by the filter off its accelerometer alone -- the gyro saw no turn --
past 12 deg only 0.4 s later; a fall turns the body, and the gyro follows
it.

**What a bearing may free.** Two rules, each a defect the flights found:

- *Behind the bracket's post.* The post hides most of a bearing's rays
  there, and the one ray that peeked past it decided the bearing: its
  point, jittered by the part's angle noise, landing a bearing over, or a
  steep ray seeing the band empty for its first metre and meeting the
  floor, laid the bearing free to 8 m through the wall behind -- every
  bearing round as the robot turned, streaks through every wall of the
  room it looked round in (78 frames in 240 in a closed room, tilted as
  walking tilts it; stage 1's band did it too, with twice the rays a
  bearing to hide it). A bearing is free only off a ray that stayed in the
  band to the 8 m reach and brought nothing back: in a closed room, none
  (`test_a_closed_room_has_no_free_bearing_behind_the_fin`).
- *Nearer than its steepest ray comes down.* Upside down 1.41 m up, the
  steepest ray meets the band's floor 1 m out (2.8 m behind the robot, under
  the head's back since stage 3): nearer, every ray passes
  over whatever is lower than it, and a bearing laid free from the sensor
  out cleared a low thing mapped from afar as the robot walked up to it.
  Nearer than that (`Scan.free_from`, per bearing, through the IMU's tilt)
  a bearing clears no cell the map holds a wall in, and lays floor only
  where none was seen -- freeing nothing there at all, a robot amid cells
  it had never seen from afar, at every boot and every stand-up, found no
  frontier it could reach, and its explore ended at once
  (`test_a_bearing_clears_no_wall_nearer_than_its_steepest_ray_comes_down`).

Over a 150 s explore, 10 % of the bearings laid free (2,140 of 21,152) were
ones a noise-free cast from the true pose saw a surface on, at a median
6.6 m, three in four past 5.5 m: a fence far out, struck by one ring just
over the band's rising floor, drops in and out with the IMU's tilt, and
between, its bearing is laid free past it. The fences hold in the map
(struck more than a third of the time, L_OCC 0.85 against L_FREE 0.4), and
what is freed is the street beyond them.

**What the review found.** An independent read of the stage found five
ways the band laid floor it had never seen. Each was shown by a probe and
is now pinned by a test.

- *The band's floor rises to meet a ray.* Where a ray leaves the band was
  worked out for a floor that fell away. Standing, the -8 deg ring comes
  down through the band's floor 7.5 m out, but was taken to stay in it to
  12.5 m. So a ray that struck a wall under the band's floor freed its
  bearing through that wall, and `free_from` read 1.06 m where it is 1.00
  (`test_a_ray_the_rising_floor_meets_short_of_max_range_frees_nothing`).
- *A scan is laid from where the sensor is* (`Scan.origin`). The band's
  ranges, and the map's free rays, ran from the sensor's place on the robot
  standing upright, 0.735 m over the pelvis. Bowed 52 deg, the sensor is
  0.58 m ahead of that place, so every free bearing behind the robot ran
  0.6 m past what its ray had reached. In a room 7 m round that was 16
  bearings through the wall behind, by up to 0.5 m (once 2 m); even a 10
  deg lean put a free ray 0.4 m through a wall
  (`test_a_bowed_robots_scan_is_laid_from_where_its_sensor_is`).
- *Tilted past 30 deg, a scan clears no wall* (`CLEARS_TILT_RAD`). Bowed,
  the rings no longer fan down along every bearing: behind the robot they
  point up. Its steepest rings, turned past straight down, land on the
  bearings behind a few cm out. Taken as those bearings' floor, they let a
  low box 2 m behind, under every ray, be cleared through the return off
  the wall beyond: on 2 bearings of 9 at a 40 deg bow, 8 of 9 at 52. Up to
  35 deg it held on all 9. A bowed scan still lays walls, frees cells no
  wall was seen in, and is matched (`test_a_bowed_robot_clears_no_wall`).
- *Behind another robot nearer than the room, a bearing clears no wall.*
  The rings over the other robot's head read the wall beyond it. Laid out
  to that wall, they cleared a 0.5 m box the robot hid, from +5 to -5 in 3
  s (`test_nothing_low_behind_another_robot_is_cleared`).
- *A dropped return is no reading*, as on the level ring (`scan_split`).
  Taken as "nothing back", half the returns dropped in a closed room laid
  25 bearings free in 50 frames (`test_a_dropped_return_frees_nothing`).

**The dock.** With the band, stage 1's dock missed 2-5 of 47; two things
were the band's.

- *The matcher at the station's near returns* (`MATCH_MIN_M`). Within 1 m
  the band strikes the station's box, the cup and the plug with a ring or
  two, at the part's noisiest (4 cm at 0.2 m), where a wall is struck by
  ten; a box 0.4 m off covers some 40 bearings. Matched, they moved a
  dock's belief 40-67 cm in 16-19 s, up to 22 mm a move; left out, 26-41
  cm, and wide starts that missed 2 of 3 docked 3 of 3. What is that near
  is what the hands work on -- the station, the plug in the hand, a drawer,
  a door -- and it is still mapped, and the front stop still reads it.
- *A step walked by the matcher's word* (`HumanoidBody._by_legs`,
  `StanceOdometry.corrected`). A step stops once what it believes it has
  walked reaches its length, and the belief it read moved with the
  matcher's corrections: a sideways step of 8 cm ended 7 cm forward, 4.7 cm
  inside the seat, and the walker cannot back up so little, so the dock
  stepped there until its steps ran out, "not lined up". A step now walks,
  turns and reports by its legs' reckoning alone, the belief less what the
  matcher and the fixtures corrected it by since the step began
  (`test_a_step_walks_by_its_legs_never_by_a_correction_the_matcher_made_meanwhile`).

Two things were tried and taken out again, each a defect a flight showed:

- *The matcher held at the station*, the scans laid unmatched from the
  dock's first look, as an anchor lays them. After the first step toward
  the seat the tags are out of the colour camera's view, even bowed, and
  the dock carries the station from its first look through the belief:
  held on odometry alone, the belief drifted 4-7 cm over two steps, and a
  seat came out 7 cm wrong, the hand sent 70 mm off the plug.
- *The station held at the feet*, the plug-in's hand placed in the frame of
  its soles: pushing the plug home turns the body 3-9 deg on its feet, the
  IMU following, and the station fix after the plug-in, read through a
  station that had turned with the feet, put every flight's heading 3-9
  deg off where it had been within 3 -- once 8 deg, past the matcher's
  search, and the map was dropped.

The plug-in is staging's, through the belief. What is left at the dock is
the walker's: a step that lands 4-5 cm inside the seat (its tolerance is
3, a hand swung into the station's box from 4.9) cannot be backed off --
asked 4 cm back, five steps moved it 1-5 mm each -- and the dock ends "not
lined up", as the level ring's did once in stage 1.

**The matcher's reach, read again** (`MATCH_RANGE_M`). It was 6.5 m because
the C1 read a wall at its 8 m reach short. The Mid-360S's band reads a wall
square ahead within 2-3 mm of where it stands out to 7.9 m (60 frames a
range, 5-7.95 m: a mean of -1.5 to -3.9 mm, 17 mm a bearing), so the bias
no longer forced 6.5 m; it stayed while this house had a fence within 4 m
everywhere outdoors. **What is true now:** it is the map's 8 m, for the
grown world's park ("The grown world", #560).

**The depth camera's low layer, measured: kept.** Over a 150-240 s explore
and a walk across the street, the layer blocked 606-812 cells the band's
map held no wall in, every one laid at least 0.8 m from the robot and
nearly all 1.2 m or more -- none in the LIDAR's blind ring by the feet.
What they were, off the truth under each at the end: the bed's top (278
cells, 0.4 m) and the couch's (133, 0.5 m), two of a plant, four of bare
floor. The band outlines the bed and the couch from where it stands, one
return a bearing at the near edge; the camera sees their whole tops. The
layer stays: it fills in what the band outlines, it sees what is under the
band's floor -- the flight's first riser, 0.18 m -- and the camera's frames
are rendered anyway for the near-field map and the peer channel.

**Decor in the band.** The dressing rule kept non-colliding geoms out of
the level ring's plane; with the band served it keeps them out of the band
(`test_decor_stays_out_of_the_band_the_map_is_laid_from`, on the house as
served), but for a tag's size lying on a solid geom (the tags on their
walls) and the plants, which the robot is meant to see
(`protocol/hints.json`'s note), as the camera's layer planned round them.
The test plants three pieces of decor the rule must catch, and it names
exactly those three: as first written it could not fail, `mj_geomDistance`
answering its `distmax` when nothing is nearer, so asked at the rule's 1 cm
every geom inside a wall's bounding sphere lay on the wall -- and asked
within 2 m, a box 1.08 m from a fence read 0 m, so it is asked a little
past the rule and never far.

**The drift batch** (`humanoid_drift_spike.py`, #572's 14 flights):

| | C1 | level ring | band, 30 rings (stage 1) | served, 15 rings | 30 rings, served otherwise |
|---|---|---|---|---|---|
| docked | 4 of 14 | 7 | 14 | 14 | 13 |
| lost | 9 | 6 | 0 | 0 | 0 |
| belief off after the explore, median (max) | 0.42 m | 0.39 (1.67) | 0.11 (0.29) | 0.17 (0.42) | 0.13 (0.41) |
| at the storeroom | | 0.91 (5.33) | 0.13 (0.46) | 0.23 (0.67) | 0.14 (1.60) |
| back at the standoff | | 0.80 (6.41) | 0.05 (0.10) | 0.08 (0.29) | 0.05 (0.30) |
| middle street: scans fixed in every direction | 0 % | 0 % | 95 % | 100 % | |
| sidewalks and outer streets | 17 % | 9 % | 100 % | 83 % (17 % held along) | |

The 15 rings cost the belief 0.04 m after the explore and 0.09 m at the
storeroom against 30, the same flights otherwise (the matcher's floor
0.03 and nothing): the accuracy Ben's call 3 traded for the pair's speed.
Every flight's station fix put its belief within 0.01 m and 0.6 deg; no
scan in any flight was refused. The 30-ring flight not docked, and one of
the matcher's-floor variant's, never lined up on the seat (before steps
walked by the legs).

Flown again after the review's fixes (above), the 14 flights docked 13, off
a median 0.11 m after the explore, 0.25 at the storeroom and 0.05 home.
Rowan, on its second walk home, took a place 1 m off from one search in the
garden, mid-turn, and the map it laid there had no way to its station. On
16 seeds no batch had flown, the code before the fixes docked 15 and the
code after them 16, off the same medians (0.13, 0.31, and 0.10 against
0.08 home): one flight lost in 30 either way, a draw.

**The station** (`station_spike.py`, the same 47 trials):

| | staging | level ring (stage 1) | band, 30 rings (stage 1) | served |
|---|---|---|---|---|
| `--approach`: docked and unplugged | 15 of 15 | 15 | 11 | 15 |
| `--lost`, belief 0.4-1 m wrong: docked | 16 of 16 | 15 | 15 | 15 |
| `--walk-in`, from 8 places, explored 120 and 240 s: docked | 16 of 16 | 14 | 16 | 15 |

Both misses never lined up on the seat (above). After the review's fixes
the same 47 docked 43 (14, 13 and 16): three never lined up on the seat,
and a search begun 0.94 m off never saw the station. On 33 approaches and
52 lost starts, the code before the fixes docked 80 of 85, and the code
after them 80: a search begun 0.92-0.96 m off failed either way, at the
edge of the metre #572 found it reaches. The probe (`lidar_spike.py
--probe`) is the band's of stage 1: no outdoor place leaves the heading
unpinned, and in the rooms 4 % (2 % with 30 rings), the matcher's floor
leaving a corner a robot faces nothing to fix its heading by.

**What it costs.** On the deploy box, the served pair ran at 1.40 and
1.41x real time (`humanoid_spike.py --speed`'s pair, one core, the trees
interleaved), the level ring's at 1.68 and 1.78: stage 1 read 1.17-1.42
for 15 rings. The energy table is priced again on this head
(`energy_spike.py --world home_humanoid`): the explore 173.5 W (166.4 on
the C1's head), a charge 338.8 W net (345.2), and the four hand jobs, all
passed from the station -- the drawer 4.735 Wh, the door 5.167, the cube
4.865 (its first pass), the chest 5.254 -- the table carrying the C1's
head's dearer drawer, door and chest, as its rule says.

**Not done.** A far fence's intermittent ring (above); the walker's seat
(above); and #581's live reading, a day of the pair with neither robot
lost outdoors, which opens with the deploy (Observatory.md). The world's
height map as the store, which the grid becomes a projection of, is its
own issue. (#572's stage 3 read this one for the heading at rest: where
its scans are matched, bowed at a counter too, no zero-rate update is
needed, "The station found from far off"; where none are, #593 holds it,
below.)

## The sensor in the head (issue #581, stage 3)

**What is true now.** The Mid-360S is built into the head (Parts.md, "The
LIDAR"), the head one cylinder 136 mm across above and below its window:
- **The face**, under the window: its front a dark visor the cameras sit
  in, a dark dome over it inside the steepest ray's cone.
- **The window:** open all round, but where a 6 mm 6082 fin behind carries
  the cap, and where seven printed panels round the back half close it
  under -26 deg.
- **The cap**, over the window: the unit's base bolted under a plate in
  it, on four rubber-metal buffers, and a padded TPU rim 4 mm proud round
  its foot.
- **Collision:** every part collides, the unit too.
- **Height:** its rays leave 0.68 m over the hips, 1.41 m up standing, 4
  mm over the lowest from which its steepest ray, -52 deg, clears the
  navigation camera's front; the cap's top is 1.48 m up.
- **What it hides:** standing, 25.6 % of a frame, nearly all of it the
  back's steep rays: the floor within 2.6 m behind.

Flown against stage 3's first head and stage 1's post (`~/pluggybot-training/
issue-581/batches/hf1`, beside stage 3's batches), nothing measurably got
worse:
- **Drift:** over 60 flights a head, 58 docked on each, and the belief
  after the explore is tighter: a median 0.06 m off, against 0.12 on both.
- **Station:** 126 of 132 starts (the first head 127, the post 123).
- **Probe:** every outdoor place pinned; the rooms 2 % unpinned (4 %).
- **Hands:** 17 of 32 (the first head 17, the post 18).
- **Policies:** none retrained; the shoves are the first head's, cell for
  cell.
- **Falls:** the rim and the cap take a fall; as served, the unit grazed the
  floor once in 74 falls, at 1.4 m/s (below).

![The sensor in the head: at eye level from ahead, its left and behind; from
the site camera's 36 deg the same three ways; where the three steepest rings
meet the floor round it -- ahead only, the back's panels closing them behind
-- and from overhead; and the robot as the site's follow camera frames it
(`lidar_spike.py --strip head`)](figures/humanoid_lidar_head.png)

**Why.** Stage 1 hung the unit from a 20 x 3 mm post 10 cm over the head,
and Ben called it fragile and impractical (2026-11-01): a €773 sensor at
the highest point, on a strip that swayed at about 12 Hz, where a fall
lands first and a door frame or a shelf catches it, and in the sim it
touched nothing. This stage's first head (b13045b) built a crown round the
unit's base on a 6 x 15 mm rib, over a dome on the box face. Seeing it, Ben
found it still read as mounted above the head, not in it (2026-11-02):
the crown, narrower than the face, read as a puck on a stick. He chose to
keep the Mid-360 and reshape the head (`~/pluggybot-training/issue-581/
HEAD_BRIEF.md`), and picked this head of three he was shown.

**What fixed the head's shape.** Three things:
- *The cone.* Upside down, the field runs from +7 down to -52 deg. Below
  the window a part r off the axis must sit 1.28 r under the rays' origin,
  above it 0.12 r over it. So at the head's edge the window is 1.4 times
  the head's radius tall, and at eye level you see through it whatever the
  head's shape: an open window cannot be closed by styling. Every part
  keeps 2 mm out of the whole field (`model.FIELD_CLEAR`; the sim's rings
  step over most of it, so a test checks the shapes against the field).
- *The site's camera.* It looks down 36 deg (`VIEW_DIR` in the site's
  `camera.ts`), and from there a line through the window lands on the dark
  dome on its far side. So the visor, the dome, the rim and the panels'
  inner faces are one dark colour (the PETG's and the TPU's own black,
  unpainted: the livery is paint, and `robot.paint` leaves them), and from
  the site's angle the window reads as one dark band across one head.
- *The manual.* Nothing in the field, "including glass", and 10 mm of air
  round the unit.

**The back keeps the rings the band is laid from** (`model.BACK`). A back
closed over every ring would make the head read as one from behind, and
it cost localization: closed over +-55 deg, #572's probe (`lidar_spike.py
--probe`) left 3 % of the street loop's places and 7 % of the rooms' with
the heading unpinned (the head without it: 0 and 4), the gardens' fully
pinned share falling from 99 % to 84 %. The band's walls come off its
middle rings: from 1.41 m up, the rings from -8 to -24 deg meet a 2.4 m
wall anywhere from 0.5 to 9 m off inside the band, and a 0.9 m fence from 1
to 3 m. The steeper ones meet the floor within 2.6 m. So the panels close
only the rings under -26 deg, round the back half: the -24 deg ring passes
1 mm over their corners and the -28 deg one meets them (a test). The probe
pins every outdoor place with them as without them, and the rooms' 2 %
unpinned, against 4. What the robot gives up is the floor and anything low
within 2.6 m behind it and aside, where it barely walks: the Navigator never
walks it sideways, and its walker barely backs up (an undock turns away).
The panels and the fin are the only parts in the field. One fin carries the
cap, edge-on to the sensor: its shadow is 9 bearings, every ring.

**Lower.** Stage 3's first head held its rays at 0.712 m over the hips: the
navigation camera rode the dome, its front 0.613 m up, and the box face's
corners, 0.099 m off the axis, bound there. The face is a cylinder now, its
edge 0.068 m off the axis, and the navigation camera comes 3.5 cm down into
the face, over the depth camera: the rays come down 3.2 cm, to 4 mm over
where the navigation camera's front meets the -52 deg ray (at 0.67 it met
it ahead), 9 over the depth camera's corners. The depth camera is where it
rode. Lower costs the shoulders' motors aside, under the -48 deg ring: 0.5
% of a frame against stage 1's height.

**Its rays leave through its window.** The unit is a body of its own round
its rays' origin. Its manual puts the IMU chip 44.12 mm under the origin,
so the origin is about 47 mm over the base's face and 13 mm short of the
window's end. A cast skips the unit's own body (`Lidar._housing`, MuJoCo's
`bodyexclude`): cast from inside the housing, every ray met it
(`test_its_rays_leave_through_its_window_never_its_housing`).

**What it hides** (`lidar_spike.py --hidden --heights`), standing, arms
hanging, the head built round the unit at each height of its rays' origin
(the face's top, the visor, the dome, the back, the cap and the fin follow
it; the cameras stay):

| rays' origin over the hips | the cap's top, standing | of a frame | what hides more |
|---|---|---|---|
| 0.735 (stage 1's height) | 1.532 m | 25.1 % | the back's steep rays, the fin |
| 0.712 (stage 3's first head) | 1.509 m | 25.3 % | and the shoulders' motors at -52 aside |
| 0.69 | 1.487 m | 25.6 % | and at -48 |
| **0.68 (served)** | **1.477 m** | **25.6 %** | |
| 0.67 | 1.467 m | 26.0 % | and the navigation camera at -52 ahead |

Nothing within 60 deg of ahead is hidden, nor any ray of the band's rings
off the fin's bearings. In the reach scene's holds (`--hidden`) it hides
26.4 % with a hand at a counter, 29.2 % at a high shelf, 27.2 % at a low
shelf and 27.8 % at the floor. No camera sees any part of the head
(`test_the_cameras_see_nothing_of_the_head_round_the_lidar`).

**A fall onto the head is the rim's and the cap's.** The unit lies inside
the head's other parts by 18 mm at the least, straight up, where the cap's
top is over its base; and by 20 mm or more within 60 deg of level, the ways
a fall meets a head (`test_no_plane_meets_the_lidar_before_its_head`; the
first head's held 6 over its unit). On the bill a padded rim and rubber
buffers under the unit's base take a blow through rubber; the sim's unit
is rigid on the trunk.

**Where the head strikes, and how fast** (`humanoid_spike.py --fall`, the unit
a part of its own):
- **Hard falls**, the robot limp or on the standing policy alone, shoved
  back, aside or back-right: the cap strikes first, at 4.7-6.2 m/s, the
  floor pushing back 4-9 kN. In 4 of those 12 the unit then grazes the
  floor, at 1.1-2.8 m/s and under 50 N: MuJoCo's soft contact lets the cap
  sink 2-4 cm at those speeds, past the rim's 2.6 cm. On the first head the
  unit met the floor in 8 of the 12, at up to 4.9 m/s.
- **Falls the get-up takes**, as served: on a bare floor the head touched
  in 3 of the 6 ways, at up to 4.8 m/s (front, the face and the cap), the
  floor pushing back 0.2 kN at most; in the house's 20 the visor struck
  first at 3.8-4.2 m/s or the cap at up to 5.0. The unit touched nothing in
  any of them.
- **Falls from a bow:** the cap is first into the floor 36 times (1.3-6.0
  m/s), the rim 16 (0.3-4.8), the visor 9 (1.7-3.1) and the face once. The
  unit grazed the floor in 16 falls on the standing policy alone (0.3-2.7
  m/s), and in 1 of the 48 the get-up takes (1.4 m/s).
- **The get-up stood** from every bare-floor fall (2.8-4.2 s), from all 48
  bowed falls as served, and in the house from 15 of 20. Falling forward
  off the landing's edge it stays down now, as on the post, where the
  first head stood: CLAUDE.md's "off the landing ... still end stuck". The
  other four are the first head's.

**The batches** (one pod; the first head's and the post's rows are stage
3's, the sim deterministic: stage 3's staging run reproduced stage 2's
batches flight for flight):

| | the post | the first head | reshaped |
|---|---|---|---|
| drift, 60 flights: docked (lost) | 58 (1) | 58 (0) | 58 (1) |
| falls in the flights | 3 | 1 | 2 |
| belief off after the explore, median / 75th / max, m | 0.12 / 0.25 / 0.48 | 0.12 / 0.23 / 0.48 | 0.06 / 0.17 / 0.51 |
| at the storeroom | 0.32 / 0.47 / 0.79 | 0.23 / 0.40 / 0.73 | 0.21 / 0.41 / 2.96 |
| station: `--approach`, 48 starts | 46 | 48 | 48 |
| `--lost`, 68 | 61 | 63 | 62 |
| `--walk-in`, 16 | 16 | 16 | 16 |
| hands, 32 flights | 18 | 17 | 17 |
| probe: outdoor places / rooms unpinned | 0 / 4 % | 0 / 4 % | 0 / 2 % |

- **The drift:** after the explore the reshaped head's belief is tighter
  (Mann-Whitney p 0.02 against the post, 0.14 against the first head; the
  spreads one, p 0.2-0.3), and at the storeroom the three are one
  distribution (p 0.2-0.7). Its lost flight (seed 16 from the start) fell
  in the garden at 255 s and got up believing itself 3 m from where it
  stood, and never found its way back: a fall's get-up walking the
  odometry, which #591 (#572's stage 3) fixes. Its other fall, in the
  garden too, was fixed off the map 3.6 s after it stood.
- **The station's misses** are every head's: not lined up, and a walk that
  gave up 0.2 m short. The pair's recording found one more: from its own
  start the first robot's plug-in misses the port 3.5 cm aside, on all four
  tries, every time. The plug arrives off the port and meets the torso
  (where it sits along the jaws' pads is not measured), and the starting
  charge does not move that walk, so the fixture swaps which robot starts
  low.
- **The policies**, none retrained (`--shove`, `--track`): the shoves it
  stands through are the first head's, cell for cell (from its left,
  standing, 15 N*s where the post held 20). Every command tracks within
  0.03 of the first head's but the 0.5 rad/s turn, 0.21 against 0.28 (the
  post 0.23), inside the two gaits its noise draws (stage 1).
- **The hands:** the lid 6 of 8 (its bar met: set-out 5 passes now), the
  drawer 4, the door 5 (set-out 2 out of reach where it passed) and the
  cube 2, at the post's set-outs (0 fell). Flown again with every contact
  on the head logged, neither miss touched it.

**What it costs.** Nothing measurable (`humanoid_spike.py --speed`, the three
heads interleaved twice on the pod, a Xeon that runs the served pair at
0.7x real time where the deploy box runs it at 1.4): the pair at 0.77 and
0.71x on this head, 0.73 and 0.71 on the first and 0.69 and 0.72 on the
post; one humanoid at 1.40 and 1.38x, 1.47 and 1.34, and 1.29 and 1.39.
One head's two rounds differ by more than the heads do. The panels add
seven contact geoms a robot, and the arms' pairs with the trunk and the
head go from 66 to 120: a bounding-sphere check each a step.

**The bill.** The rib's allowance becomes the fin-and-plate's, two machined
parts, the same €66.67. The visor, the dome, the back and the cap are
printed with the face in PETG; the rim in TPU (`tpu_1kg`, €35), and the
unit sits on four of RS's M3 rubber-metal buffers (`rs_189_3249`, €10.03 a
pack of four). The total is €16,374.07, €51.79 more with the contingency.
The robot weighs 43.18 kg, 0.03 more than the first head.

**What is left.**
- **Other parts outside the body still touch nothing in the sim**, as
  before this stage: the D435 stands 2.5 cm proud of the visor and the
  charge port's funnel 1.5 cm proud of the torso; the motors' housings are
  visual too, though the limbs' capsules cover most of them. Each wants
  the same treatment as the LIDAR: collide, then fly the batch again.
- **The unit still grazes the floor in some falls**, the rim and the
  buffers being on the bill and not in the physics: MuJoCo's soft contact
  sinks the cap 2-4 cm in a hard fall, past the rim's 2.6 cm. A rim 2 cm
  proud of the head (a 4 cm margin, read as a brim) or a stiffer contact
  on the head's parts would end it; the second changes what the get-up
  meets.
- **The plug-in's grip along the jaws is not measured**: from the pair's
  own start the first robot misses the port 3.5 cm aside every time, the
  plug off its catch and against the torso.
- **#581's live reading**, a day of the pair with neither robot lost
  outdoors, opens with the deploy (Observatory.md).

## A robot left standing when its run ends, and a heading held at a standstill (issue #593)

**What the live pair showed.** On 34cd4e3 (2026-11-01 and 02) Rowan stood
in `DONE` four times for 20-55 minutes before a restart, no scan matched,
its heading turning by its gyro's offset (3 deg a minute, near the part's
worst) until it was 105-161 deg off. Each next run carried on from that
belief (#345): a wide search put it 2.6 m and 176 deg off, a room turned
round, and it died flat 40 m lost; later a false fit 7 m off, and a fall
5-20 m lost. Luca never stood in `DONE`: each time it was the robot still
charging.

**Why.** A run ends robot by robot: the day loop checks its budget between
errands, so a robot charging passes it inside the charge, and the other's
day ends first. `tick.run_many` held the finished robot `STILL`, running
its stepper's bookkeeping -- the odometry, integrating the gyro -- and
nothing its routines do between steps, where a scan is matched and laid
(`Navigator._after_step`).

**The finished robot's wait** (`run_many`'s `waits`, given by `run_pair`):
its body's own hold, as a robot waiting for work holds, until the other's
day ends; the loop still ends on that step, and the wait is closed there.
Flown, the served pair in the house, Rowan's day over at 6 s and Luca's
exploring, running low and charging past the run's budget to 478 s
(`humanoid_drift_spike.py --done`; Rowan's draws, its offset +2.98 deg a
minute), with each of #593's two fixes taken out and both:

| Rowan in `DONE`, 472 s | heading off at most | belief off at most | scans matched while it waited |
|---|---|---|---|
| held `STILL` by the loop, no hold (before: `--no-wait --no-hold`) | 23.4 deg, its offset times the time | 6 cm | none |
| held `STILL`, its heading held (`--no-wait`) | 1.7 deg | 6 cm | none |
| its body's hold and its heading's (served) | 0.10 deg | 1.2 cm | 4,666, ten a second |

**A heading held at a standstill** (`imu.HeadingHold` in the odometry,
#425's zero-rate update on the heading alone, the filter's tilt the
accelerometer's). Where nothing is matched -- every scan refused, a robot
lost -- the same offset turns a standing robot. One humanoid in the house,
its matcher off, its gyro's offset the part's worst (0.05 deg/s each axis),
stood three minutes after three scenes (`humanoid_drift_spike.py --still
boot|walk|turn`; `--hold-after S` for a hold begun elsewhere in the rest):

| standing still | the body's own turn as it settled | held from 30 s (served) | from 20 s | unheld, at 3 min |
|---|---|---|---|---|
| booted, the standing policy | 5.6 deg: 0.75 deg/s at once, 0.08 at 20 s, under 0.001 from 45 s | 1.60 deg | 1.40 | 9.0 |
| after a 2 m walk, the walker | 1.5 deg, 0.03 deg/s at 20 s | 1.49 | 1.07 | 9.0 |
| after a turn on the spot, the walker | 16.1 deg: 2 deg/s, 0.34 at 20 s, 0.07 at 40, under 0.01 from 60 | 2.63 | 3.56 | 8.9 |

Settled, the rate its gate reads -- its heading's, less the offset
learned, smoothed -- stayed under 0.09 deg/s (p99 0.05-0.06) on either
policy, inside the gate's 0.2, and the hold held throughout; it learned
the offset to 0.049-0.050 deg/s. What it cannot tell from rest is the
settle's slow real turn: what of it comes once the hold begins is taken
for stillness, and the offset turns the heading until then. Begun at the
rest's 8.6 s (as the quadruped's was) the worst was 3.6 deg, at 60 s 3.0.
Over the three, 30 s's worst (2.6) is second only to 40 s's (2.4), and it
keeps a commoner rest's -- after a walk, a boot -- at 1.5-1.6 deg, where
40 s let the offset turn it 2.0.

The drift spike counts what the hold takes off as a heading share of its
own, `hold`, so its shares add up to the error again: Rowan's draws,
explored 60 s and stood 90 s in the bedroom with the matcher on, the hold
took 2.8 deg off the gyro's 7.9, to a residual of 1e-13 deg.

The 30 s count from anything that moved the body, never the walk command
alone: a stand-up's warp, a get-up and a hand's work move it without one,
and it settles after each. Counted off the walk command, the hold began
the moment a stand-up warped the robot back to its start, and held it
through the settle a boot turns through: off by 2.36 deg where the boot's
was 1.60 (`--still standup`, standing 60 s and then put back at its
start).

**What is true now.** A robot whose day ends first holds as one waiting
for work does, its scans matched, until the other's ends. A humanoid
resting 30 s after anything moved it holds its heading while its IMU
agrees nothing turns it, and takes off its heading the offset it learned
there, walking too: standing where nothing is matched it turns off the
heading it stopped with by what its offset turned it in those 30 s and
what of its settle came after, 1.5-2.6 deg at the part's worst, where its
offset turned it 3 deg a minute unheld. Where its scans are matched, the
matcher holds its heading as before.

## The station found from far off (issue #572, stage 3)

**What is true now.** A humanoid that does not find its station where its
belief puts it -- none in sight from the standoff, or no walk there got
there -- looks for it by its head's camera and walks nearer by what it saw
(`HumanoidBody.lost_routine`): a look round where it stands, then
viewpoints outward from where it believes the station is, and once seen,
to a metre out of the station as that look placed it, where a look fits it
near enough to put the belief right (`_fix_at`). Its belief set 1-5 m and
5-20 deg wrong -- 2 m out, from the house's six rooms and across the street
-- it docked 121 of 123 (staging, from the same starts, 14); explored
first, laying its map askew through that belief, 23 of 27 (staging 1).
Begun within 5 m of the truth, its search found the station 109 times in
109. The walk to the station is given the time its length takes, a dock that
saw the station and never got in goes again, and a fall's get-up walks no
belief. Where its scans are matched its heading needs no hold at rest:
with the band, Rowan's draws stood 180 s in the middle street and their
heading moved 0.1 deg (9 on the C1); where none are, #593's hold keeps it.

![Found from behind its station's wall: a humanoid 2 m out, its belief set
2 m and 12 deg wrong (stage 2's figure, "Where the humanoids get lost"),
walks to where it believes the standoff is -- truly the south garden,
outside the station's own wall -- looks round and searches the garden's
viewpoints round where it believes the station is, comes in by the garden
door at about 155 s, sees the station, walks nearer by what it saw, puts
its belief right (1.7 m and 11 deg) and plugs itself in at 199 s, a tile
every 16 s (`station_spike.py --strip lost --lost-from station --lost-off
-1.606,-1.192,12`, every other tile and the
last)](figures/humanoid_station_far_found.png)

**What its cameras read of the station** (`station_spike.py --reach`: set
out of the station's face and off its axis, facing it, a decode of each
camera; the belief is the truth there, so what is measured is the look).
The colour imager the dock looks by reads the pair within 1 m and nothing
past 1.25 m -- the tags rise out of its view -- and it was all the search
had: from 1.5 m off and further it found the station 0 times in 43 (stage
2). The head's navigation camera, pitched for the walls' signs, reads both
60 mm tags square-on from 0.8 m to 4.5 m, one from 30 deg off the station's
face to 5 m (the parked plug hides the far tag from about 20 deg), and
from 50 deg only within 1.5 m:

| out of the face | tags the head's camera read, 0 / 15 / 30 / 50 deg off its axis | the pair fitted off them: its facing, its place |
|---|---|---|
| 0.8-1.0 m | 2 / 2 / 2 / 1-2 | within 0.7 deg, 7 mm |
| 1.25-1.5 m | 2 / 2 / 1 / 1 | within 2.4 deg, 6 mm |
| 2-2.5 m | 2 / 2 / 1 / 0 | 1.0-14.4 deg, 0-51 mm |
| 3-4.5 m | 0-2 / 0-2 / 1 / 0 | 1.1-40.2 deg, 4-47 mm |
| 5 m | 0 / 0 / 1 / 0 | |

(On #581's stage 3 head, its navigation camera 3.5 cm lower; on the first
head the same tags read, and the pair's facing came out within 1.9 deg to
1.5 m, 1.0-5.5 deg at 2-2.5 m and up to 28.6 deg at 3-4.5 m.)

A far look says where the station is to a few centimetres and which way
it faces to tens of degrees: the pair is 18 cm across, and a centimetre of
range between its two tags turns it 3 deg.

**The search** (`charge.LOST_*`, `NEAR_*`):

- *A look round where it stands*, a decode every 60 deg (the places'
  look-around, `humanoid.places`): lost within a room of it, the robot sees
  the station from there.
- *Then viewpoints outward from where it believes the standoff is*, over
  its own map, the nearest by the walk first, a look round at each, its
  walks stopping as soon as a decode shows a tag (`next_viewpoint`, the
  find's search; `HumanoidPlaces.see_station` notes every look of the head
  that shows one). An askew belief puts the station's believed place in
  another room or the garden, and the true one a walk or two away. ⚠
  ROUND THE STATION'S BELIEVED PLACE, NEVER ROUND THE ROBOT: lost 5 m away,
  a search round itself walks off from the station.
- *Seen, nearer by what it saw* (`_near_station_routine`): to `NEAR_M` (1
  m) out of the station as the newest look placed it (`charge.rough`: the
  pair fitted, or one tag where it is and facing as its own rotation says,
  either never more than 60 deg off the way it was seen from), facing it,
  and a look of the colour imager and of the head's camera. ⚠ NEVER A FIX
  OFF A FAR LOOK: put right 20 deg wrong, the robot lays its map askew
  again. A look that fits the station there -- the colour imager's, or the
  head camera's with both tags within 1.6 m (`near_fit`) -- is the fix.
- *Bounded*: `LOST_PATIENCE_S` (600 s), or the pack empty
  (`HubLifecycle._search_routine`); seen and never near after two rounds of
  four tries, it says so (`seen, never near`).

**Walks that never got there, and their time** (`HubLifecycle.
go_charge_routine`). Five of stage 2's lost flights never searched, their
walks to where they believed the standoff was finding no route through a
map laid askew: two walks that gave up anywhere are a robot lost too,
searched for once from where they ended, unless the other robot stopped
them. The walk itself is given 6 s a metre of the straight line, at least
the 90 s it had (`CHARGE_WALK_S_M`): home from the storeroom, 22 m in a
straight line, #581's drift flights walked 70-82 s and up to 117. And a
dock that saw the station and never got in goes again from the standoff,
once a charge: 6 of the 117 stations found in the first sweep were never
got in, the walker's step 4-5 cm inside the seat, which it cannot back off
(#581, "The band served").

**A fall** (`HumanoidMission._after_physics`; `humanoid_drift_spike.py
--falls`: shoved over six ways, 300 N for 0.3 s, after a 120 s explore). The
odometry stepped while the get-up policy had the legs, following a
tumbling body's feet: a fall walked the belief 1.7-2.1 m in stage 2's
flights and Luca's 3.7 m live, past the wide search's 2 m. It is held
through the get-up, and stood, its planted foot is read afresh -- the foot
planted before the fall counted the whole get-up's move at once. Luca's
draws in the hall and Rowan's in the bedroom, where their explores end on
#581's stage 3 head, staging's code and this stage's: 8 of 12 shoves put
the robot down on either (it stood through the rest), and every one stood
within 5 s:

| | staging | the odometry held |
|---|---|---|
| the body moved during the get-up | 0.10-1.35 m | the same |
| the belief moved during the get-up | 0.66-1.80 m | 0.00 |
| the belief off as it stood | 0.64-1.79 m | 0.04-1.38 |
| 5 s on, back within 0.11 m | 5 of 8 | 7 of 8 |

What a held belief is off by as it stands is what the get-up moved the
body and what the stagger before the fall left (the legs track a stagger
under a shove only to 0.1-1.0 m). The matcher's searches took most back
within 5 s. The held one they did not, Rowan's draws shoved from 240 deg,
the get-up carrying the body 1.35 m, stood 1.37 m off with its scans
sliding -- refused, never fused, so its map stays true -- and is left to
the station's look at its next charge; walked, staging left three 0.6-1.2
m off, two on scans the matcher held along a wall. (On the first head, in
the workshop and the garden, every one came back within 5 s.) Held, the
belief can be no further off than the fall carried the body; walked, it
went 3.7 m live. The three falls in this stage's far sweeps all docked.

**The heading at rest, read before and after** (stage 2 asked for it to be
read first). Rowan's draws stood 180 s in the middle street: on the C1 its
heading went from +0.4 to +9.2 deg -- its gyro's offset, nothing matched --
and its next walk passed the facility's gate a metre off and ended 6.2 m
off. With the band (staging 44f28e4) it went from -0.06 to +0.08 deg,
every scan matched, and both its flights docked; with the offset taken off
(`--learned`), 0.81 to 0.81. The band's matcher takes the offset off
wherever its scans are matched, bowed at a counter too: no zero-rate
update is needed there. Where nothing is matched -- every scan refused, a
robot lost -- #593 holds the heading at a standstill ("A robot left
standing when its run ends, and a heading held at a standstill").

**Measured** (`station_spike.py --lost --far`, flown on a Vast CPU pod, each
trial a process; staging is 15e37db -- #581's stage 3 head, its navigation
camera 3.5 cm lower, and #593's heading hold -- both with the workshop's
start moved off its table, where stage 2's put the robot inside it; the
times over the 123):

| | staging | this stage |
|---|---|---|
| belief 1-5 m and 5-20 deg wrong: 2 m out (15), the house's six rooms (72), across the street (36) | **14 of 123** docked; 4 never searched, their walks out of time | **121 of 123**; every station found was plugged into |
| ...explored 120 s first, its map laid through the wrong belief: the rooms (18), across the street (9) | **1 of 27** | **23 of 27** |
| found, where the search began within 5 m of the truth | 13 of 105 | **109 of 109** |
| ...where it began 5 m and more off | 0 of 37 | 32 of 38 |
| the search, when it found the station | median 40 s | median 95 s; 9 in 10 within 307 s |
| the whole charge, when it docked | median 91 s | median 177 s; 9 in 10 within 440 s |

On the first head (staging 44f28e4) the same sweeps read 118 of 123 and 24
of 27 (staging 14 and 2), every search begun within 5 m finding it.

The drift batch (`humanoid_drift_spike.py`, #581's stage 3's 60: #572's 14
flights and seeds 3-29 from either start) docked 60 of 60, none lost and
none fallen, its beliefs staging's to the centimetre until the charge
(staging docked 59, its miss a plug the port never took). On this head
before #593's hold (#581's stage 3 alone) one was lost: seed 16 from the
start fell in the garden at 255 s and stood believing itself 3 m off, the
get-up having walked its odometry. With #593's heading hold its explore
goes otherwise and it never falls, here or on staging, and its own commit
flown again on another pod's CPU parted from that flight within 100 s and
fell nowhere, so what the get-up's hold does for it is read off the falls
above. The station's
own batches, as #581's stage 3 flew them: from round the standoff 15 of 15
and 33 of 33, its belief 0.4-1 m wrong 16 of 16 and 51 of 52 (every search
found it, in 2-235 s; the miss a plug the port never took), from across
the house 16 of 16 -- 131 of 132, where staging docked 124 (four never
lined up, which the dock's retry takes, two walks that gave up 0.2 m
short, and two plugs the port never took).

**Not done.** What the search does not reach is a robot lost 5 m and more
off across the street: its walk home ends outside the house's south fence,
in the street or on its sidewalk, and 6 such searches in 38, begun 6.1-9.2 m
off, ran out of their 600 s there. Stage 4's station in the research
center would put a robot right before it crossed back, and is Ben's call on
this reading. The walker's seat (#581) is what is left of a charge from its
standoff.

## A plug-in that missed every try (issue #596)

**What the scene showed.** The pair recording's world with the first robot
sent to charge at 22 % (`two_robots.py --world home_humanoid --fast --pack
demo --near-field --errands none,none --battery 0.22,0.9`): on llvmpipe its
plug-in missed the port four times the same way, each letting the plug go
31-35 mm to the right of it, where the port takes it within 22, and the
robot died `stuck`; since #591 a dock that never got in goes again from the
standoff, and it plugged in on its second approach, at 137.7 s. On the GTX
the same scene plugged in at once.

**Where the hand's error came from** (`station_spike.py --anatomy`, the
issue's scene, the truth beside what the robot believes). Four things, each
a few millimetres to two centimetres, all in how the hand was sent:

- *The belief moves under a robot standing still.* From the look the
  plug-in went on to its squeeze, the feet turned 1.6-7.7 deg where they
  stood, their middle under 1 mm, while the belief moved its stance 5-25
  mm: the matcher's corrections at the station's near returns, and the
  legs' reckoning reading the turn as a walk (alone, 10-30 mm). Each of the
  hand's targets went through the believed world at its move's instant,
  and went with that move. Held in the frame of the feet instead, they went
  with the turn: 18 mm at the plug.
- *A look just after a step reads the trunk's tilt off.* For a second or
  two after a step the IMU's filter (`imu.TAU_S`, 1 s) read it 0.4-0.6 deg
  off, and through the 0.9 m from the soles up to the station's tags that
  put the station 5-12 mm off from its feet; three seconds standing, 0.05
  deg and a millimetre or two.
- *A retry was a retried constant*: every try went on the dock's last look.
- *The port, read once, moved 6 mm before the push*: a posture taken to
  carry the plug moves the trunk it is on.

In staging's first try the hand shut 59 mm behind the plug's face, on the
station's boss; drawn back off it through the moved belief again, it took
the plug 22 mm behind its face and 16 mm deep in the jaws, and the
gripper's housing met the belly with the plug 25 mm off the port.

**The plug-in now** (`humanoid/charge.py`). Each try looks at the station
afresh, the first after `LOOK_SETTLE_S` standing still, and carries what it
saw to each move's instant as the feet carry it (`_carried`): as far as the
belief has moved its stance -- standing, the feet stay where they stood --
and never turned with it, the body turning on its feet as the belief does.
Slid onto the plug, the hand goes there again as the station is then (its
feet turned meanwhile, 7.7 mm at the grip), or as near as its arm goes. The
port is read at each move. On the issue's scene the plug sat 2.7 mm off the
jaws' middle on llvmpipe and 0.1 mm on the GTX (staging, 15-21), the hand
1.8 and 2.4 mm off the port's line before the push, and each plugged in at
the first try.

**The push.** Pressed on the body, the arm's spring slides the hand 6-11 mm
to the right and 4-6 mm down as it pushes, the plug with it, well inside
the port's 22 mm; and the plug is let go where the push stops, pressed on
the body. Put back on its line after the push, or pushed again from as far
across as the first slid, it was let go 9-12 mm off the body, near the
port's 15, and of 140 seats' let-goes the port missed 20 and 26 where,
pressed on the body, it missed 1.

**Measured** (`station_spike.py`, staging and the plug-in now flown on one
Vast pod, an AMD EPYC 7B12, osmesa; every trial's row says where each try
let the plug go against the port, `letGo`):

| | staging | the plug-in now |
|---|---|---|
| `--seats`: 140 seats over what the seat check accepts | plugged in 138; 181 let-goes, 43 missed: 14 over 22 mm across, 26 over 15 mm out | 139; 140 let-goes, 1 missed (27 mm out) |
| `--approach --n 30`: 33 starts round the standoff (`--approach`'s 15 first) | docked 33; 45 let-goes | 33; 33 |
| `--lost --n 48`: 52 beliefs set 0.4-1 m wrong (`--lost`'s 16 first) | 51; 78 let-goes | 52; 52 |
| `--walk-in`: explored 120 s, walked to 8 places across the house | docked 8; 10 let-goes | 8; 8 |
| the issue's scene | four misses, then docked on its second approach (137.7 s) | the first try (45.9 s) |
| across the port at a let-go, a batch's median; its worst | -12 to -15 mm; 22-55 | -12 to -13; 15-17 |

The seat it now loses is the seat check's furthest corner -- 6 cm behind
the seat, 6 cm to its right, turned 6 deg right -- where the arm stretched
shut on the plug near its nose, the first push let it go 27 mm off the
body, and the retry's reach fell short; staging docked there, and lost two
of its own corners (3 cm in, 4-6 cm left, turned 6 deg left), pushing four
times. 3 cm in from the seat the plug is let go 10-13 mm off the body,
inside the port's 15 with the least room. `--walk-in`'s 240 s explore walks
nowhere, on staging too: from where that explore leaves the robot every
walk out stalls in 10 s, so its eight rows are one trial.

**What is true now.** A try's hand is placed off a look of its own, taken
standing still, carried as the feet carry it -- moved with the belief's
stance and never turned with it -- with the port read at each move; and it
lets go where its push stops: over the four batches' 233 starts it docked
232 with 233 pushes, where staging docked 230 with 314.

## The grown world (issue #560, stage 1)

**What is true now.** The house is a 2x2 grid inside one fence
(`home/world.py`): today's row north -- the house, the middle street, the
research center -- and south of it the PARK beside the house and the MALL
beside the research center, the middle street running on between the
columns, a pedestrian ALLEY (3 m) between the research center and the mall,
the house's south sidewalk widened to the same 3 m between the house and the
park, and the loop round all four. The second row is 16 m tall: the map is
51 x 42 m, 856,800 cells -- 1.83 times the one row's 469,200, 46 % of
#558's cap. The house gained a room for each robot (`room_west` and
`room_east`, off the living room's south wall, where the garden wrapped;
`home.ROOM_OF` names whose by root body, never a name); the research center
a study room (the south half of its front garden, off the lobby, a table and
two chairs: for studies with the robots as subjects, #557). The park is a
lawn behind a garden fence, a 1.5 m gate on its north and on its east
(across from the mall's front door), two trees and a sign in its middle. The
mall is a concourse down its west side, as the research center's lobby runs
down its own, with the restaurant (its dining room, four tables of two; the
expo line, a pass in the kitchen's wall with its counter in it and a staff
door beside it; the kitchen, its worktop and its dish shelf) and the store
(the shop, its checkout, a gondola and a wall shelf; behind a staff door the
loading zone, its rack and a door out to the sidewalk) side by side off it.
Every table, counter and shelf is scenery and stands empty: the dishware and
the stock are each task's own (stage 2), and every geom is one the LIDAR's
rays are cast against.

**Three charging stations**, one in each building, each on a south wall
facing north, as the house's: the research center's at its lobby's south
end, the mall's at its concourse's (`home.STATION_WALLS`,
`humanoid.world.stations`), each its own tag pair (64-65, 66-67) and its
own activity. Which one a charge goes to is the body's, once a charge
(`Body.choose_station`): the one its plug is in, else the nearest by a walk
over its OWN map (`Navigator.nearest_by_walk`: the planner's lattice
through the unknown at its price, one Dijkstra and three readings). From
the park's west sidewalk the house's is nearer as the crow flies (22.8 m)
and further by every way round the fenced park (49 m against the mall's
47); with nothing mapped, the house's wins there -- the robot's map, never
the world's. A lost robot fixes off WHICHEVER station it sees, its pair
saying which, and charges there; the loop walks to that station's standoff
after (`test_a_search_that_found_another_station_walks_to_that_one`). The
choice rides a restart. A station takes one robot, and the #346 wait is
per station.

**The reserve's worst place** is now the middle of the park's west
sidewalk, where every way round the fenced park is long: 47.3 m on the
test's raster to the mall's station, through the park gate to gate, and
49.4 to the house's (`HOME_WORST_RETURN_PATH`: 49.65 m of waypoints; the
one station's worst, the loop's south-west corner, was 49.64). MEASURED
(`energy_spike.py --reserve`, a pod): the walk 14.20 Wh in 301 s -- an upper
bound, its 54.8 m wandering past the waypoints' 42.4 -- and from the mall's
door the loop's own charge, down the concourse and plugged in at the mall's
station, 4.38 Wh: 18.57, carried with a plug-in tried again as 19.5
(`humanoid.world.RESERVE_WH`; 18 before).

**Localization** (`lidar_spike.py --probe`, the band, the share of places
0.5 m apart whose scan leaves the heading unpinned):

| group | n | matcher to 6.5 m | to 8.0 m |
|---|---|---|---|
| middle street, sidewalks, the loop, the alley | 4,032 | 0 % | 0 % |
| gardens | 312 | 0 % | 0 % |
| rooms (the robots' and the study room with the old) | 1,320 | 2 % | 2 % |
| the mall | 768 | 5 % | 5 % |
| the park | 1,408 | 10 % | 2 % |
| the park's middle (within 3 m of its centre) | 112 | 71 % | 26 % |

The park's middle is 8 m from its north and south fences and 11 m from its
east and west, past the matcher's 6.5 m (`MATCH_RANGE_M`, set on the C1's
recorded lab trip against a far wall's pull): only its two trunks and its
sign are in reach. Read at the map's own 8 m it pins 74 % of it; four more
trees round the middle pinned 37 %, and both ranges raised to 12 m 79 %,
for half as much again of each scan's cost. **So the matcher reads the
map's whole 8 m** (`MATCH_RANGE_M`; the band reads a wall within 3 mm out to
7.9 m, "The band served"). Flown on one pod -- the drift spike, eight
flights an arm, two of them standing 180 s in the park's middle -- the
belief's worst error had a median of 0.13 m at 8.0 against 0.15 at 6.5
(the worst 0.30 and 0.43), its error at the end 0.05 and 0.06, every flight
docked; in the park it stayed within 0.23 m (0.35 at 6.5). The same flight
flown twice on the pod was bit-identical.

**A wall at the reach, and the field's window.** The C1's far wall at its
8 m reach was seen short -- its long draws clipped to "no return" -- and
pulled the pose 32 mm a scan, which is why its cut was 6.5 m. The band's
is not: matched from the true pose at the 1,328 places whose scan has ten
returns or more 7.5-8 m out, the pull was 1.0 mm (p95 2.1) at 8.0 against
0.8 (1.9) at 6.5 -- its sigma is 2.4 cm there, and it reads a surface by
its returns' median. The field the fit reads (`WINDOW_M`) stayed 7.5 m
either way, short of the reach and a refresh's 1 m of travel, so a point
past it pulls on nothing. Grown to 9 m, the probe pinned the park's middle
better (19 % unpinned against 26) on its map laid from the true poses;
flown on the robot's own map -- the drift spike's same eight flights, on
a pod where staging's arm replayed bit for bit -- the worst error's median
went from 0.13 m to 0.21, five flights worse and one from 0.13 to 0.47 m
in the living room. So the window stays where it flew best.

**What it costs.** The served pair's walks (`humanoid_spike.py --speed`'s
pair, the same walks in the house as every reading since #507). On the
deploy box (3 Nov, the world paused by its schedule; #558's recipe, the
serving image a container a tree pinned to one idle core, the trees
interleaved), the grown world first read 0.94, 1.13 and 1.04x against
today's staging's 1.24-1.35x -- under the 1.15x floor Ben set. Profiled on
a pod it was 14 % slower, 59 % of that the LIDAR's cast. ⚠ EVERY GEOM IN THE
MODEL COSTS EVERY RAY of an `mj_multiRay`, reachable or not: the band cost
9.4 ms a frame on staging's 236 geoms and 12.6 on the grown world's 351,
and its cutoff saved nothing, again; geom groups a cast leaves out do drop
out. So a cast leaves out every static geom past its sensor's reach
(`perception.cull`, the LIDAR's and the depth camera's, 177-241 of the 248
at each place): the same readings bit for bit (`test_cull.py`), and the
box's walks went to **1.09, 1.29, 1.25, 1.25, 1.10 and 1.21x, median
1.23x**, staging's 1.24, 1.33, 1.29, 1.35, 1.30 and 1.31x (median 1.30x)
the same rounds: 7 % slower, where it was 19 %. A further lever, not built:
the visual-only floor and ground slabs and the tags (55 boxes, 1-3 mm over
what is behind them) inside a sensor's reach -- left out, a return moves
by their millimetres, so a flight is no longer the same flight.

**Flown at the new stations** (`station_spike.py --approach --n 30
--station`, the same 33 starts round each standoff, one pod): the house's
docked 32, the research center's 30 and the mall's 33 of 33, every one
unplugged and its plug parked, none fell. The house's one miss was a start
that missed at the research center too ("not lined up", the seat's steps);
the research center's other two, another not lined up and a grip that never
got round the plug. In the loop a dock that saw its station and never got in
goes again from the standoff, once a charge.

**The hand jobs are priced from the mall's station.** A job is taken from
any of the three, and the energy table carries the dearest: the mall's,
43 m by route from the storeroom's counter, where the house's is 29 and
the research center's 7. Left to choose, the spike measured every job
after the first from the research center's, the nearest once the robot
had been to the storeroom; `--station` pins one. Flown from each on the
waist's body (10 Nov, a process each on one pod; `energy.json`'s
`measuredOn` has every reading), the mall's was the dearest of every job:
the drawer 7.503 Wh (a try that stopped at its reach: a pass costs
more), the door 9.041 and the chest
8.543 (each passed), all three carried, and the cube 8.058 (placed),
under 3 Nov's 9.858 from the same station, a placement that failed after
four reaches, which is kept; from the house's 5.44, 6.53, 5.80 and 7.39,
from the research center's 2.17, 3.66, 2.78 and 5.98. Each of the four
mall flights came to the storeroom with its belief 0.41-0.67 m off (the
house's and the research center's about 5 cm): its find's first walk ran
out of its 90 s 8-11 m short, it looked round and found the row from the
building's address, and the drawer's reference then asked `step` for
0.79 m aside, past the 0.5 m it takes, and could not reach the handle.
The tests' pack (`DEMO_WH`) grew from 30 Wh to 33 on 3 Nov, so that
charged it still funds the dearest job and the reserve both (29.7 of
29.36), and the pair's fixture was flown again on it.

**Three stations sensed every step** (`Station.sense`, on the physics
seam) cost 32 us a step each, against a pair's 229 us `mj_step`; an armed
port beside a parked plug is no longer read -- it has nothing to take and
nothing to re-arm -- so a parked station costs 12 us, and the three what
one did. The drift spike's flights replayed bit for bit across it.

**Carried on from the deployed save** (the 11-02 backup, `serve.py` as
deployed with a fake model): the world changed, so each robot stood up at
its start with its pack (Luca 83 %, Rowan 56 %), its jobs still its own, its
map dropped, History saying why. In that flight Rowan charged at the house's
station from 56 %; Luca, choosing to charge at 82 %, found it taken, waited
its 1,386 s bound and gave up, though both new stations stood free: a charge
goes to the NEAREST station and waits for it there, #208's contention per
station (a decided charge that never began costs time, never a heart). So
the robot decides (Ben, 3 Nov): a decided `charge` with `station: "free"`
passes a taken station for the nearest one nobody holds, at most
`MAX_STATION_SWITCHES` a charge, and waiting stays the default -- an
event-map row's charge and a standing order's.

**Not done.** The tasks (stage 2). The second story (the mall's, #557's
first idea) waits for maps by floor; the park's rough patch for the walker
that sees the ground.

## Setting a table (issue #560, its first task)

**What is true now.** #560's first task, a place laid at the restaurant's
table, is built and graded offline (Challenges.md §12): the dishware
(`activity/dishware.py`: a plate pinched by its wall, a lidded cup taken
across its body, a fork by its thick handle, and a place card), its grader,
its set-out (`counter.table_set_out`) and our procedure (`hands_check.TABLE`).
**On the waist's body our procedure meets its bar** -- 32 of 32 places at
set-outs 0-31, from one stance ("The table's reference on the waist",
below); its job is offered ("The lay from the tour"). This section is the
body before the waist,
whose procedure (`hands_check.TABLE_STANCES`, a stance a piece) met no bar:
0 of the check's 8 set-outs and 6 of 32 held out on today's body, the cup
23, the plate 24 and the fork 6 of the 32 (after #560's stage (a), below),
and none on the waist's body. Every part of the place sat at an edge of what
that body reaches at a 0.72 m table, below; the levers that move those
edges, and what a waist yaw does to them, are stage (c2)'s ("The table's
put-downs, measured before the waist is built").

**Why the cube missed its mat** (#559's `place_cube`, 0 of 8; its check's
procedure flown at set-outs 0-7, the truth beside the belief at every
verb):

- *The grip scrubs the stance round.* Aimed at the cube's middle, the
  fingertips meet the board first and the reach holds pressing it (45, 42,
  17 and 14 N at four contacts); the jaws then drag the pads across it as
  they shut, and the feet swivel 4-6 deg in 0.2 s. The mat, read in the
  stance frame before the grip, is 3.7-5.4 cm off by the carry (set-outs 0,
  2 and 4, whose carries then ran out of reach). Eased to where it is
  before its jaws shut (`Reach.ease`, stage (a)), its command walked there,
  the shut turns the soles 0.5-1.7 deg.
- *A held cube's carry reads as a drop.* The carry's posture swings the
  hand +-5 cm at up to 1 m/s with the cube held fast, and the grader's 0.5
  m/s line, read at every step, failed two cubes set down 0.7 and 1.3 cm
  from the mat's middle (3 and 5). Both graders now read a drop off a free
  piece alone (Ben's call): re-flown, the check lays 4 of 8 (1, 3, 5 and
  6; the bar 6), the two carries 0.97 and 0.59 m/s held and 0.21 and 0.23
  free.
- *The way home caught the counter*: a jaw on its edge, 15 s, the robot
  turned 20 deg (1 and 6, after the cube was down) -- the side way's roll
  coming in at the robot's side, where the counter it stood turned to was
  nearer; it now comes in pitched back ("The way home"), and turns the robot
  0.2-1.8 deg. With both, the check lays 5 of 8 (2 and 4 gained, 6 lost;
  4 of 8 before them, on the same machine).

**What the table took, each measured** (a scratch world and the set-out,
the humanoid at the table's south place; flights in
`~/pluggybot-training/issue-560-table/`):

- *A walk carries nothing*: `step` refuses while a gripper holds
  (`"holding"`), so a piece is picked and placed from one stance, and the
  dishware waits on the table beside the place -- never on the expo
  counter or another table, as #560 first had it.
- *A solid table meets the thighs.* Crouching to reach, the robot pressed
  the dining table's box with both thighs at about 200 N from 13 cm out; the
  set-out's table is a top on legs. The served room's tables are boxes.
- *The way out swept a hanging hand into the table.* Standing 13 cm out,
  the hanging grip is 7 mm behind the top's edge, its top at the
  underside's height: straight in joint space, a reach's way out drove the
  left forearm and gripper into the table at 353-359 N. It is judged as the
  way home is now (`Reach.way_out`, stage (a)): from the hanging arm, round
  by its side, and the table is met with 0 N. Opening a hanging hand alone
  pushed 171 N (#615's scratch world).
- *Each hand works its own side, in bands.* Flown from 13 cm out at the
  legged table: fingers straight down at |y| 0.24-0.40 m (0.18-0.42 ahead),
  tilted 45-60 deg at |y| 0.24-0.48 (0.30-0.48 ahead), and a near strip at
  |y| 0.08-0.16 (0.18-0.30 ahead); neither hand the middle 0.08 m. One carry
  does not cross from a band to the strip (out of reach), and a reach holds
  only 2-3 cm above its grasp's height (fingers down at the cup: 0.82 m held,
  0.84 out of reach). So a place and its dishware need a stance a piece --
  the robot sidesteps, its hands empty -- and a stance's window is about 3
  cm wide: at a 0.21 m sidestep the plate's wall, read right, was past the
  band at 19 of 32 grips (the left gripper "closed on nothing"). A blocked
  step's retry walked about 5 cm further (its slide unseen), and 6 of 7
  such gripped the plate where 5 of 24 others did: at 0.26 m, the step now
  landing where it is asked (stage (a)), the plate is laid 21-22 of the
  public 32 and 24 of 32 held out (at 0.21, 4 and 4).
- *Flat tags* read to 2-4 mm from 0.37-0.53 m ahead, nothing nearer, and
  nothing 45 deg to the side after a 0.21 m step: the plate and the fork
  are read from the start, as offsets from the card.
- *The stance turns where it stands*: 2-3 deg in its first seconds after a
  set-out (4.9 deg in 15 s), and as an arm goes out, the reaching side's
  foot 7.5-8.6 deg a reach to the cup -- the same at half and a quarter of
  the arm's speed, so the standing policy's stance under an arm held out,
  not its swing. A place read before is stale; the card is read again
  before each reach, and each arm is rolled out and folded (`move`) before
  its piece is read: read first, the hand came down on the cup's rim 2-5
  cm off, and it was laid 5 of 32 where 21.
- *A sidestep after an arm's way home was "blocked"* (6 of 32 tries): the
  walker slid its first second, unseen (Stepping toward the work); a
  walk's first window is 2 s now (stage (a)), and none of 41 blocked.
- *The reach's approach starts where the posture left the hand*, 3-5 cm
  from its pre-grasp, and converges diagonally: tall things are clipped.
  Sent through its pre-grasp instead (`Reach`, tried and not kept), the cup
  was laid 5 of 8 either way.
- *Shape is not the plate's trouble*: a 4 cm wall and a 3 cm handle laid 8
  plates of 32 and no fork.

![The procedure before the waist's (`TABLE_STANCES`) at the table's set-out
4, the first place it laid on the body before the waist: the cup carried by
the right hand and set down by the card (5-10 s), that arm home the side way
(15 s), then a step aside and the plate, a step back and the fork, both by
the left hand -- the plate 1.5 cm, the cup 1.5 and the fork 2.1 from where
each goes (`scripts/hands_check.py --strip table 4` at e4be1d4, before the
waist)](figures/hands_table.png)

## A waist, measured before it is built (issue #560, stage (c))

**What is true now.** The yaw measured here is BUILT (stage (c3), "The
waist built", below), #620's candidate exactly; this section is what was
measured before it was. Three candidates were measured on the robot's model
of itself (`scripts/waist_spike.py`, drawn from the body before the waist,
`HumanoidSpec(waist_yaw=False)`). **A waist yaw, one RS03 at the pelvis's top (+1.73 kg), buys most of what
was measured**: each hand reaches past the middle of the table, every
pick-up our procedure makes is made standing tall, and the floor and the
high shelf come into reach in front of the shoulder. A waist pitch keeps the
legs at the stand, but it adds little reach and holds its RS04 near its
limit. **Neither widens the plate's or the fork's stance window with our
procedure's grip as it asks it**: its carry into a put-down near the body's
middle stays out of reach with any waist -- but asked the other way round,
the yaw opens both, and one stance lays the whole place (stage (c2), "The
table's put-downs, measured before the waist is built", below). Ben's call
(2026-11-05): the yaw alone, on an RS03, built once those put-downs were
measured with other grips, paths and hands. No policy holds a waist yet, so
every number here is a STATUE's: what the body could reach and hold still,
planned as the served reach plans. That is an upper bound on what a policy
will do, and it is compared with today's body planned the same way, never
with a flight.

**The candidates**, each today's generated drawing with a waist put in at
the pelvis's top. Its zero pose is today's body to a micrometre, and the
waist turns the head's sensors, the port and the arms and nothing below
(`tests/test_waist_spike.py`):

| candidate | its axis over the hip axes | range | part (stalled, N*m) | adds | rides on it |
|---|---|---|---|---|---|
| pitch | 0.18 m (the T2's 0.182), the pack below it | 32 deg back, 43 deg forward (the T2's) | RS04 (28.5) | 1.96 kg: the motor and its bracket | 14.4 kg, its centre 0.11 m over the axis |
| yaw | 0.06 m, the pelvis's top: the motor stood in the pelvis's middle, where the hips' pitch motors leave 108 mm | +-60 deg, all of it used by the reaches (the T2 turns +-138) | RS03 (13) | 1.73 kg: the motor, its bracket and a 5 mm turntable plate | 19.5 kg, its centre 0.18 m up the axis |
| both | the yaw below the pitch, the T2's and the G1's order: a bow goes where the shoulders face | | | 3.68 kg (43.18 -> 46.86) | |

The whole body's centre of mass rises 0.3 cm with the yaw, 1.0 with the
pitch and 1.3 with both. The yaw's motor fits the pelvis with room; the
pitch's does not quite: at 0.18 m the RS04's 120 mm housing takes the top
2.4 cm of the pack's 74 mm, so a pitch built would raise its axis or lay the
pack flatter.

**How a reach is measured**, as `reach.plan` plans one:
- A posture is the legs' (`reach.POSTURES`) and the waist's angles (its
  pitch in steps of 0.15 rad, its yaw in steps of 15 deg), tried least bent
  first on a statue balanced over the soles (`reach.static_legs`).
- The arm's six joints are solved for the goal and the pre-grasp 8 cm back,
  with 1.5 cm of slack; the other grip half a turn round only where the one
  asked plans nothing (`Reach._planned`).
- The statue is then balanced again with the arm out, as the policy moves
  its pelvis back for one. It must still reach, keep its centre of mass over
  its soles, keep the arm and the trunk out of the table's top (13 cm out,
  as the set-out stands) and the arm out of its own trunk and legs.
- A posture whose trunk folds into a thigh stands nowhere: at the hips'
  deepest bow a yaw of 15-30 deg put the torso's corner 1-2.5 cm into one.
- A window adds our procedure's carry (lifted 5 cm, carried straight, set
  down), followed from one posture, each point solved on from the last and
  checked for clearance on its way.

**The method, checked on today's body.** The planner reproduces
`reach.plan` cell for cell over #615's grid. With the carry, today's
windows land where #615's flights did: the plate's wall is picked up and
carried from a 3 cm stretch of stances, 0.29-0.31 m to the right (our
procedure steps 0.26, where a carry that changes posture once makes it,
below; laid 21-24 times of 32 there); the fork's stretch is 12 cm from where our
procedure stands for it (laid 6 times of 32); the cup's is 25 cm (laid 23
of 32); and no single stance lays the whole place. Without the carry the
statue found an 8-11 cm plate window and one stance for everything, which
the flights refute: a carry, not a reach, is where a place fails. The
statue's BANDS are wider than the flown ones: fingers down, |y| 0.24-0.56
with slack where #615 flew 0.24-0.40. The plan's slack does not cover what the
standing policy does to a posture (it stood 2 cm under the height asked,
and the stance turns 5-8 deg as an arm goes out). So a band below is a
statue's, beside today's statue.

**At the table** (the left hand at 0.77 m, where the pieces are gripped;
the right is its mirror). Reached clear over the grid 0.10-0.62 m ahead and
-0.20 to +0.60 across, in 4 cm cells:

| | fingers down | tilted 45 deg (our grip) | between the two hands' bands, \|y\| < 0.24 (down / tilted) | across the middle, y < 0 | the legs at the stand's (down / tilted) |
|---|---|---|---|---|---|
| today | 0.107 m^2 | 0.194 m^2 | 0 / 19 cells | none | 15 % / 46 % |
| pitch | 0.114 | 0.205 | 2 / 33 | none | 80 % / 65 % |
| yaw | 0.229 | 0.264 | 73 / 73 | 27 / 16 | 30 % / 45 % |
| both | 0.234 | 0.251 | 94 / 73 | 40 / 23 | 79 % / 75 % |

- *The yaw doubles the fingers-down band and fills the middle.* Fingers
  down, the left hand reaches clear to 0.20 m past the middle (the grid's
  edge) from 0.26 to 0.54 m ahead, 27 cells; tilted, past it more patchily
  and further out (0.34-0.62 m, 16 cells), its forearm dipping into the
  table's top nearer the edge. Today's reaches neither past the middle.
  Across at a single distance (`--across`) the run has gaps where the first
  plan fails once balanced again.
- *Every pick-up our procedure makes is made standing tall with a yaw*
  (`--alternatives`, every posture tried): the cup's, the plate's and the
  fork's from the start, and the table's middle near and far, the worst
  joint at 38-57 % of its part's stall rating. Today each needs a crouch or
  a bow, the knee at 81-84 % of an RS04's.
- *The put-downs nearest the body still crouch 8 cm*, whichever the waist:
  the plate's wall set down 8 cm left of the middle and the fork 16 cm,
  both 0.30 m ahead, are reached from one to four postures, the knee at
  76-82 %.
- *The pitch keeps the legs at the stand*: the waist bows where the hips
  and the knees did, and over the table's reaches the knee's median load
  falls from 22 N*m to 5. Its reach grows little: the far edge 4-8 cm, and
  never across the middle.

**The envelope** (the left hand in front of its shoulder, 0.2 m out; 77
cells from 0.09 to 1.55 m up and 0.15 to 0.75 m ahead; `--envelope`):
reached clear with slack, today 36, the pitch 49, the yaw 59 and both 56
(a candidate with more postures can count fewer: the planner keeps its
first plan, and more of them fail once balanced again). The floor (0.09 m)
today only at the toes and at full stretch; the pitch and the yaw reach it
0.15-0.45 m ahead, and both to 0.55. The high shelf (1.43 m), today
0.35-0.45 m ahead; the yaw 0.15-0.55. The floor is reached from the hips'
deepest bows, the hips and the knees at 87-101 % of their stall ratings
with any waist.

**Balance.** Every reach found stood balanced: the statue keeps its centre
of mass over the soles' middle fore and aft, as the policy moves its pelvis,
and across it stayed at least 10.4 cm inside the soles' outline.

**The part**, sized by a pose held as a stalled motor (`Motor.stall_torque`),
1 kg in the hand:
- *The yaw: an RS03.* It holds at most 8.1 N*m over the table's reaches
  (its median 0, its 90th percentile 2.4) and 2.4 over the envelope; 14
  with the pitch bowed over it. A 20 N `pull` sideways at a hand (the
  verb's most) puts up to 10.1 N*m on it: 78 % of the RS03's 13 N*m
  stalled, where an RS02's 6 would not hold it. Its winding takes 1 W at
  3 N*m and 10.5 at 10.
- *The pitch: an RS04, at its limit.* It holds a median 14-15 N*m over the
  table's reaches, 26-27 at the 90th percentile and 32-35 at the deepest
  (the envelope's), past the RS04's 28.5 stalled in 2-5 % of reaches (the
  worst held 130 s by its sheet); a vertical pull adds up to 10.2. Its
  winding takes 10 W at 13 N*m, 22 at 20 and 44 at 28.5, where each knee it
  relieves took 31 W at 24 N*m.
- *Each driver's standby* is the bill's 1.7 W.

**The stance window** (`--window`, `--reposture`): where the stance may
stand, moved across in 1 cm steps at the procedure's 13 cm from the edge,
and the hand pick a piece up with our procedure's grip (45 deg, the jaws
across) and carry it where it goes as our procedure does:

| | the cup | the plate | the fork | one stance for the whole place |
|---|---|---|---|---|
| today | 25 cm | 3 cm (0.29-0.31 m to the right) | 3 cm, 12 cm from where our procedure stands | none |
| pitch | 13 + 31 cm | 2 cm | 4 + 2 cm | none |
| yaw | 67 cm | 1-3 cm stretches | 3 + 3 cm | none |
| both | 33 + 34 cm | 1 cm | 3 + 1 cm | a few single cells, the waist turning in the arm's chain |

Picked up alone, or set down alone, the plate and the fork each open to
27-53 cm of stances with the yaw. What stays shut is the CARRY between
them, toward a put-down near the body's middle (the plate's wall set down
0.30 m ahead and 8 cm left of the middle, the fork 16 cm): the arm's solve
misses its points there by 1-4 cm. No way of carrying that a waist allows
changes it:
- the waist held in its posture, as a carry holds one today;
- the waist turning in the arm's chain as the hand goes;
- either at the reach's own tolerance or at the served carry's
  (`reach.REPLAN_M`, `REPLAN_ROT`);
- the posture changed on the way with the hand held still, as a served
  carry may change it (`--reposture`). From the start, the yaw's plate
  carry gets 10 of its 14 points (today's 3) and the fork's 6-8, and neither
  arrives.

Today's body, by the same search, carries the plate from our procedure's
0.26 m step with one change of posture (a bow, then standing), where the
flights laid it 21-24 times of 32; and the fork from its own stance not at
all, where they laid it 6 times. So with this grip, asked as our procedure
asks it, the place's limit is not the waist's to remove -- and the grip is
the lever: today the reach takes it half a turn round wherever it carries,
and the yaw lets it take it as asked, which never carries (stage (c2),
below).

**What building it would cost.**
- *The model and its generated files*: `humanoid/model.py` (the trunk cut at
  the waist, a joint list of 25 or 26, the keyframes, the actuators,
  `training_constants`), `parts.DRIVES`, the bill (an RS03 at €215.95 and an
  RS04 at €254.75, with their brackets and the turntable plate at #379's
  €33.33 a machined part: €316 for the yaw, €321 for the pitch, €637 for
  both before the contingency). Then `models/humanoid.{xml,json}`, Parts.md's
  bill and `protocol/parts.json`, `models/humanoid_friction.json` (the
  waist's rows), and the three humanoid scenes and both recordings, which the
  site re-vendors first.
- *What is perceived through the model of itself.* The tag cameras and the
  station's port are placed through it already (`mj_camlight` on the
  self-model, `charge`'s port site): they follow once `JointSense` reads the
  waist and `SelfModel.pose` sets it. The LIDAR's place and turn on the trunk
  (`HumanoidMission.LIDAR_ORIGIN`, `Lidar._site_in`) and the depth camera's
  mount (`depth._fixed_mount`, which refuses a joint on its way) are fixed at
  construction; each must be posed off the waist's encoders every frame,
  and the band's and the depth frame's levelling (`Lidar.band`, `_levelled`)
  must compose the waist's turn over the pelvis's tilt. The whole body's
  force reading (`SelfModel.felt`) gains the waist's rows, since the waist
  carries the arm's load to the legs. A tip or a topple is still read off
  the pelvis's IMU.
- *The policies.* The stand is retrained: the waist scripted as the arms
  are, with stage (a)'s yaw anchor on each sole. The walk and the get-up are
  flown on the new body first, the waist held stiff and unseen (a policy
  picks the joints it sees by name, `StandPolicy.joint_names`, once its
  check that they are all the body's joints is a subset's), and retrained
  only if they fail; the seeing walker (not served) likewise. At
  training/README.md's cost per million steps on a 5090, one try of the
  served three is 1.7 GPU-hours and $1.02 (the stand 31 min and $0.27, the
  walk 23 min and $0.19, the get-up 49 min and $0.56), and the seeing
  walker's 3.3 h and $2.14. #507 trained them all, every try, for $12.15.
- *Flown again*, each a CPU pod's batch (about $0.1-0.2, as stage (a)'s
  were): every hands check, the served jobs (`solve.py --from tour`), the
  station (`station_spike.py --approach`, `--seats`, `--lost`), the body's
  tables (`humanoid_spike.py --torque`, `--shove`, `--envelope`, `--pulls`,
  `--feel`, `--bar`, `--fall`), the LIDAR's (`lidar_spike.py --probe`,
  `--hidden`), the drift spike, the continuation's parity
  (`determinism_spike.py --resume-at`), and the pair's real-time multiple
  on the deploy box: the waist costs the bare robot 2.4-4.8 us a step
  (+7-11 % of its own step), about 0.5 % of the served pair's (an
  estimate; the box reads 1.23x against Ben's 1.15x floor).
- *The deployed pair's body changes*: a new Observatory period, and a
  restart over the changed geometry keeps the clock, the packs, the deaths
  and the jobs but drops the bodies and the maps (`continuation.fingerprint`).

**What a statue cannot say.** Whether the standing policy holds these
postures (today's statue bands are wider than the flown ones, above); whether
the stance still turns as an arm goes out (stage (a) measured the crouch's
share at 5.2 deg of 8.2, and a waist that keeps the legs at the stand may
take it away, which only a retrained stand can show); and how a yaw turned
for a reach lays the LIDAR's band (held at zero while walking, the band is
today's).

![Each candidate's statue at three of the table's reaches, as the planner
takes them clear of the table and of its own body: today's body crouches 16
cm for the cup and the plate's wall and reaches no posture for the table's
middle; the pitch bows 9 deg for both and 43 deg over the table for the
middle; the yaw stands tall, turned 15-30 deg (`scripts/waist_spike.py
--strip`)](figures/waist_table.png)

## The table's put-downs, measured before the waist is built (issue #560, stage (c2))

**What is true now.** On today's body the PLATE's carry opens with its grip
and the FORK's does not open: gripped 30 deg round its wall from straight
ahead rather than at its side, the plate is picked up and carried from 13 cm
of stances by every measure below -- from where the robot stands for the
cup, no step -- and flown, it is laid 27 of 32 held-out set-outs (23
before), 33 and 34 of the 34 times it was reached in two runs; no grip,
path, hand or stance opens the fork by every measure but a relay, which
loses the fork in flight (and the fork waiting nearer the table's edge,
where the robot could not read its tag). The best fork flown, gripped 60 deg
down, is laid 12 and 17 of 32 in two flights of it (6 before): 9 and 14
places of 32 (4 before), and 2 and 1 of the check's 8 (the bar 6). **The
waist yaw opens both, its grip asked the other way round**: the plate's and
the fork's windows open from 3 cm to 21 (35 tilted 60 deg), and one stance
lays the whole place -- 11-13 cm of stances, 21 tilted 60 deg, by both
measures run on it. #620 found no waist widened them because it asked our
procedure's grip, which as it is asked carries neither piece from anywhere:
today the reach takes it the other way round at most stances, and every
carry is from those, and the yaw puts it in reach as asked at most -- 38 of
the plate's 55 pick-ups -- where it is taken as asked, and never carries.
The yaw at +-90 widens several windows further, which the table does not
need. Every number is a statue's but the flights'.

**How it is measured** (`scripts/putdown_spike.py`, on the robot's model of
itself as #620's instrument measures a reach). A lever's WINDOW is the
stances, 1 cm apart across, from which the hand picks a piece up -- clear of
the table and of its own parts, still reached once balanced again -- and
carries it where it goes and sets it down, clear; the bar, written before
anything was measured, a window of 6 cm (twice a sidestep's +-3 cm) for
every piece. Three measures of the carry, because they disagree:
- *one posture* follows it all, the reach's postures, at the reach's own
  tolerance: #620's `--window`, reproduced cell by cell (`--check`: 48
  cells);
- *repose*, #620's `--reposture`: the posture changed on the way, the hand
  held still, wherever any posture reaches -- reproduced at 12 cells; run
  on today's body alone (on a waist it tries every posture at every point,
  36 s of a stance's 57);
- *served*, new: our procedure's verbs as the served reach makes them,
  every posture one of the carry's own (`reach.HOLD_POSTURES`: three bows
  at full height the reach's lack) -- each `pull` in the posture `path_for`
  plans and another, onward, where its arm misses; a straight `move_hand`
  in the one `posture_for` finds for its end and another where it misses,
  at most once a second; a turn or a `via` path in the one `path_for` plans
  for it all.

On our procedure today they read the plate 3, 8 and 7 cm, the fork 3, 7 and
5, and the cup 25, 30 and 26. Its plate step, 0.26 m, is at the served
window's edge (laid 24 of 32 at #618), its fork stance, 0.09, outside all
three (6 of 32).

**The levers** (62, `--list`), each on today's body, the yaw and the yaw at
+-90, and the widest window, cm, by one posture / repose / served (the bar
6; the relays a leg each):

| lever | plate today | fork today | plate, yaw | fork, yaw |
|---|---|---|---|---|
| our procedure: 45 deg, the jaws as it asks, a 5 cm lift, straight | 3 / 8 / 7 | 3 / 7 / 5 | 3 / - / 3 | 3 / - / 3 |
| the jaws as asked, never the other way round | 0 / 0 / 0 | 0 / 0 / 0 | 0 / - / 0 | 0 / - / 0 |
| the jaws asked the other way round | 3 / 11 / 7 | 3 / 11 / 5 | 21 / - / 21 | 21 / - / 25 |
| tilted 60 deg, the jaws the other way round | 1 / 8 / 9 | 1 / 7 / 8 | 35 / - / 37 | 35 / - / 38 |
| tilted 30 deg, the jaws the other way round | 3 / 11 / 9 | 3 / 13 / 9 | 14 / - / 11 | 14 / - / 20 |
| fingers straight down | 0 / 2 / 1 | 0 / 3 / 2 | 16-23 / - / 10 | 16-22 / - / 11 |
| the plate's wall gripped 30 deg round from ahead | 13 / 13 / 13 | | 14 / - / 10 | |
| the plate turned 45 or 60 deg about the upright on its way | 13-22 / - / 24-25 | | 36-44 / - / 22-37 | |
| the fork's handle 3 cm nearer its end | | 3 / 7 / 7 | | 3 / - / 5 |
| lifted 15 cm; through a point 5 cm out ahead | 3 / 5 / 6; 3 / - / 6 | 3 / 6 / 6; 3 / - / 8 | 1-3 / - / 1-3 | 1-3 / - / 1-3 |
| the right hand (the plate's right wall) | 11-22 / 11-22 / 6-13 | 0 / 0 / 0-1 | 19-35 / - / 10-16 | 1-11 / - / 9-13 |
| the stance 5 cm back, or nearer (knee room) | 2 / 7-8 / 6 | 1 / 7-10 / 6-8 | 5-7 / - / 2-5 | 5-7 / - / 2-3 |
| the stance turned 10-30 deg | 0-5 / 0-6 / 0-7 | 0-3 / 0-6 / 0-5 | 0-7 / - / 0-6 | 1-4 / - / 1-5 |
| the dishware waiting nearer the edge | 8 / 8 / 4 | 7 / 7 / 6 | 4 / - / 2 | 5 / - / 2 |
| a relay: set down halfway, a step, taken up again | 8+9 / 10+9 / 8+9 | 7+9 / 10+12 / 7+8 | 10+16 / - / 12+17 | 12+10 / - / 8+11 |

- *The jaws are the yaw's lever, and its cheapest.* A two-jaw grip is the
  same grip half a turn round its fingers (`Reach._planned` takes the other
  where the asked plans nothing; the jaws are mirrors about the grip).
  Today every pick-up of the plate and the fork in a window is planned the
  other way round; with the yaw, ours as asked is planned at most stances,
  and carries nowhere. Asked the other way round -- one sign of our
  procedure -- on the yaw the whole place, the cup by our procedure's grip,
  is laid from one stance, 13 cm of stances by one posture and 11 served;
  tilted 60 deg too, from 0.04-0.24 m to the left of where the robot starts
  by both: one step of about 0.14 m, and none between the pieces.
- *Where the plate is gripped is today's plate's lever*: its wall 30 deg
  round from straight ahead puts the hand out in front of its shoulder at
  both ends of the carry, where at its side the put-down is near the body's
  middle. Turned about the upright on its way (`move_hand`'s turn: the
  plate is round, and turns with the hand) it opens wider, 22-25 cm, and is
  not yet flown; nor is the right hand on the plate's right wall, 0.29-0.50
  m to the left (22 cm by one posture, 13 served).
- ⚠ *A tilt turned on the way* (picked up at 45 deg, set down at 60 or 90)
  reads wide on the statue and cannot be used: the turn is about the jaws'
  own line, which runs through the plate's middle, so a plate held firmly
  turns with the hand, and nothing pulls it level.
- *The fork waiting nearer the table's edge* opens its window too, 7 / 7 /
  6, but nowhere our procedure stands could find it: 0.19 m ahead and 0.38
  to the left of the start, 63 deg to the side, where the head camera reads
  a flat tag only 0.37-0.53 m ahead and within about 45 deg ("Setting a
  table"). Not flown. Tilted 30 deg, the jaws the other way round, the
  served carry reads the fork 9 cm, the one posture 3: not flown either.
- *A relay* clears the bar on the statue, piece by piece, and flown it loses
  the fork: after a step the robot knows where it set the fork down by its
  legs' reckoning, which misses a sidestep's slide (its second grip closed
  on nothing 5 times in 40), or by the fork's tag, read far to the side
  (unseen 8 times; laid 3 cm off 16).
- *The yaw at +-90* widens several windows past +-60's -- tilted 60, the
  plate's to 51 cm served (37) and the fork's to 47 by both (35 and 38);
  fingers down, the plate's to 45 by one posture (23) -- where +-60's
  already clear the bar several times over.

**Flown** on one machine (an EPYC 7J13, osmesa, as deployed: our procedure
flown again there repeated all 40 flights exactly), our procedure changed
by each lever alone, at the check's set-outs 0-7 and #618's 32 held-out
ones. The two that bought something are kept as scenes
(`hands_check.SCENES`; `scripts/hands_check.py --view table_rim30_t60 K`
watches one), flown again as kept -- the same procedure but for half a
second's stand before the plate, which moved 10 verdicts of 40: a flight's
outcome is that chaotic, so the two runs are both given:

| our procedure, changed | places at 0-7 (the bar 6) | places of 32 | cup / plate / fork of 32 |
|---|---|---|---|
| as it is | 0 | 4 | 23 / 23 / 6 |
| standing at the served windows' middles (0.29 and 0.19 m) | 0 | 3 | 23 / 12 / 4 |
| both pieces gripped 60 deg down, at their windows | 0 | 2 | 23 / 21 / 2 |
| the plate gripped 30 deg round, from where it stands (`table_rim30`) | 1, 1 | 2, 1 | 23 / 27 / 3, 2 |
| ...and the fork at 60 deg, from 6.5 cm to the right (`table_rim30_t60`) | 2, 1 | 9, 14 | 23 / 27 / 12, 17 |
| ...and the fork relayed | 0 | 0 | 23 / 27 / 0 |
| ...and relayed, its tag read again | 0 | 2 | 23 / 27 / 2 |

- ⚠ *The statue's windows sit a few cm off the flown ones*: stood at the
  served windows' middles, the plate's pick-up stopped short 16 times in 40,
  and our procedure's own 0.26 m step -- the window's edge on the statue --
  laid it 23 of 32. Compare statue to statue, and fly what is chosen.
- *The cup is a place's limit too*: laid 23 of 32 whatever the lever (it
  comes first, unchanged), its grip closing on nothing at 4 set-outs of 40,
  and knocked off the table before the plate is reached at 5 others -- the
  same set-outs whatever the lever.

![Our procedure changed by the two levers that bought something, at the
check's set-out 0, the one of the 8 it laid whole: the cup as ever (5-15
s), then the plate gripped 30 deg round its wall from where the robot
stands (25 s) and the fork at 60 deg after a step -- the plate 0.4 cm, the
cup 0.6 and the fork 2.0 from where each goes (`MUJOCO_GL=osmesa
scripts/hands_check.py --strip table_rim30_t60 0`; `--view` watches
it)](figures/hands_table_putdown.png)

**What it cannot say.** Whether the standing policy, retrained for the yaw,
holds these postures; how far the stance turns as an arm goes out on a body
whose trunk turns instead (#620's open half); and the yaw's windows flown.

## The waist built (issue #560, stage (c3))

**What is true now.** The body has a waist yaw: an RS03 stood in the
pelvis's middle turns everything above the pelvis's top -- the torso, the
pack, the head and its sensors, the port and the arms, 19.5 kg, its centre
0.18 m up the axis -- +-60 deg (`HumanoidSpec.waist_yaw`). It is #620's
`yaw` candidate exactly (`tests/test_waist_spike.py`: every body, geom and
joint, place, mass and inertia, the same), and at its zero the body before
it to 1e-15 m. 44.9 kg, 25 joints.

**The senses, off the parts.** The IMU rides the pelvis, the head's sensors
the waist, and nothing turns them by the simulator's angle:
- *A sensor's mount is posed through each joint on its way at its driver's
  reading* (`perception.mount`); a pose handed no reading is refused, never
  laid at zero. The LIDAR's band and level ring and the depth camera's
  frames take the waist's angle the odometry posed its model of itself with
  (`HumanoidMission.waist_read`, every 10 ms); the band levels its points
  through the IMU's tilt composed with that turn. The LIDAR rides the
  waist's axis, so the waist turns it where it is.
- *Every camera placed through the robot's model of itself follows*, since
  the model sets the waist off its reading: the tags' colour imager, the
  places' navigation camera, the station's port.
- *The whole body's force reading* gains the waist's row (it carries a
  hand's load to the legs), and the friction table its row: 0.29 N*m dry,
  0.06 viscous, against the RS03 sheet's 0.30 (`humanoid_spike.py
  --friction`).

**The reach with a waist.** A posture is the legs' and the waist's
(`REACH_POSTURES`, `CARRY_POSTURES`: #620's grid of 0, +-15 ... +-60 deg by
the legs' postures, least bent first by #620's cost), the waist driven as
an arm is and held in its posture through a carry (Ben's call). Three
things it took:
- *A statue a posture*, worked out once a drawing and kept
  (`SelfModel.statues`): 36 ms each, and a plan that found nothing, 1.3 s
  over today's 8 postures, would have been 11 s over 72 on the physics
  thread.
- *An exact bound prunes*: no arm takes its grip further from its shoulder
  than hanging straight, so a goal past that is skipped unsolved.
- *A turned trunk folded into a thigh stands nowhere* (`reach.stands`): at
  the deepest bow a turn of 15-30 deg puts the torso 1.1 cm into one, as
  #620 found; nothing in the drawing collides there.
Planned so, a left hand at a table's height 0.45 m ahead stands nearly tall
turned 30 deg where today's crouched and bowed, the floor 0.09 m up is
reached 0.15-0.55 m ahead from a turned bow (today only at the toes), and
the right hand reaches 0.4 m past the middle.

**The waist driven** (`arms.Waist`). Its gains are sized to what it turns:
0.46 kg*m^2 about its axis with the arms hanging, 1.14 with both out, where
the arms' scale (4x, 2x the policy's) rang at 4.2 and 2.7 Hz at a damping
ratio of 0.42 and 0.27; at 8x and 10x (kp 632, kd 50, inside the RS03's MIT
range, its manual's 4.1.2) it is 0.94-1.48. Its turn goes at 0.6 rad/s at
most and is set going and stopped at 1.5 rad/s^2 (`reach.turned`; 60 deg in
2.1 s), as the stand's training turns it -- for that alone: set going at
once, the soles turned no more (60 deg either way with both arms held
level: 1.85 deg to 1.81 on the stand that knew no waist). Walking and
getting up hold it home: neither policy knows it, the walker takes the legs
only once it reads home, and the get-up walks it home.

**The reach turns it only where no square posture reaches** (`reach.with_waist`):
every posture with the waist square first, in the legs' order, then the
turned ones by #620's cost. By #620's cost alone -- a turn of 20 deg as bent
as the legs' next posture -- the planner turned the trunk before it bowed
deeper, and the served jobs from the tour (32 draws each, one pod, the stand
that knew no waist) passed the door 23 times and the chest 14, where square
first they passed 28 and 25 (today's body: 31 and 20). A turn AWAY from the
reaching hand -- the right hand's target ahead of it with the trunk turned
left 30-45 deg -- solved the arm in a contorted branch (its shoulder rolled
-1.6 rad and turned 2 rad), and its swing out turned the whole stance 7-17
deg on every stand flown. Eight draws a job could not tell: the door's 5 of
8 there was 23 of 32, and a single draw passed on one machine and failed on
another.

**The stand retrained** (`Pluggy-Humanoid-Stand`, the runs in
`~/pluggybot-training`'s record): stand4's recipe on the new body, the waist
scripted as the arms are -- a new angle within +-60 deg every 1-4 s or home,
turned as the reach turns it, its weight fed forward -- and the policy shown
its encoder. Measured in our physics: in training, its episodes' length at
the end (stand4 702 steps); the soles' turn, beyond standing still, as the
cup's reach at the table's set-outs 0-7 is taken (the robot 9 cm back,
nothing touched: the arm walked out alone, and with the reach's posture);
and the soles standing still a minute on a bare floor from the stand.

| run | each sole charged for | episodes | the arm out | with its posture | still a minute |
|---|---|---|---|---|---|
| stand4 (no waist), as served | nothing of its heading | 702 | 4.6-4.8 deg | 8.2-8.4 | -5 to -7 deg, 2.9 cm, settled by 30 s |
| waist1-11 | (each of the below) | 220-450 | | | |
| waist12 | nothing of its heading | 613 | 0.3-1.8 | 1.5-1.8, but 12 at 3 of 8 | -6 to -7 deg, 1.7 cm |
| waist13 | its heading off the episode's, 2.5 a rad | 330 | | | |
| waist14 | its worst turn planted, once, 20 a rad | 748 | 0.5-0.6 | 1.2-1.6 | CREEPS: 7 cm, still going |
| waist16, served | its heading off the episode's, 0.5 a rad | 620 | 0.9-1.5 | 3.6-3.9 | -4 deg, 1.4 cm, settled by 30 s |
| waist17 | the same past 2 deg, 1.0 a rad | 594 | 1.4-2.3 | 0.9-2.5 | -2 to -3 deg, 1.5 cm |

(The turns are the worse sole's beyond what standing still turned it over
the same 6 s, as stage (a) measured stand4: `stance_turn.py` in the
stage's scratch.)

- ⚠ *At the shoves' full size every stand falls* (waist1-11, as stand1
  did): stand4 was trained at 0.6 of it, a flag then, the default now
  (`DISTURBANCE_SCALE`).
- ⚠ *A heading charged at 2.5 a radian off the episode's made falling
  cheaper than standing* (waist13, as stand5's -25/m anchor did): 0.5 is
  about as dear a turn as `feet_anchor`'s -10/m is a shift, 5 deg of it as
  0.4 cm.
- ⚠ *A sole's turn charged a planted phase at a time lets a re-planted sole
  off* (waist14, as stand3's anchor renewed at touchdown let a step go
  free): it turned its stance least of all under a reach and shuffled 7 cm
  in a minute standing still -- a creep its odometry cannot see (its belief
  ran 2 cm off in 12 s at the counter).
- *Forced through each of the waist's angles* on that reach, waist16's soles
  turned 2-6 deg beyond standing still, 10-12 at +45 and 9.6 at -60, with no
  fall; stand4's turned 7-15 (17 at +45).

So for Ben's pitch call: the stand retrained for the yaw turns its stance
a third to a fifth as much as an arm goes out -- 0.9-1.5 deg where stand4
turned it 4.6-4.8, and 3.6-3.9 with the reach's posture where stand4's
turned 8.2-8.4.

Served, flown again once the cube's causes were fixed (one pod, an EPYC
7Q83, which flies today's tree to the earlier pod's numbers exactly; today's
in brackets): the lid at 32 set-outs 21 (21), gripped then at the knob's
middle (22, 23 and 25 at three seeds 2 cm short: "The chest's grip"); the
drawer from the tour 7 of 8 (7) and the door 29 of 32 (31). The station as
before: docked from round the standoff 36 of 36, plugged in from 138 of 140
seats (139), 16 of 16 lost near and walked in, lost far 112 of 123 (116).
Standing, it takes a shove 5 N*s bigger from the front and the sides (15
and 20 where stand4 took 10 and 15). The drift's worst belief error 0.15 and
0.13 m (0.11 and 0.28), both docked; the LIDAR's probe the same on every
row; a restart's parity identical. A hand job draws 157-166 W from the
mall's station (156-161), inside the energy table's 10 %, so `energy.json`
is not priced again. The pull a hand holds (`--pulls`, flown before this
round's fixes) 590 N summed over its 14 ways, as stand4's, 20 at its
weakest, so `pull` still asks at most 20.

**Each set-out its own noise** (#560's seeds stage; Evaluation.md §5), on
one pod, a Xeon Gold 6138, osmesa, which flew #622's rows for the lid and
the cube 61 times of 64; in brackets the one realization every set-out
shared before, on the same pod:

| | of 8 (0-7) | of 32 |
|---|---|---|
| the lid's check, its grip 2 cm short | 8 (8) | 22 (22) |
| the drawer's check | 5 (5) | 22 (20) |
| the door's check | 5 (5) | 25 (22) |
| the cube's check | 4 (4) | 15 (15) |
| the table's place | 0 (0) | 0 (0): the plate laid 19 times, the cup 20, the fork never |
| ...at the 32 held out | | 0: the plate 9 (#622's realization 17), the cup 20, the fork never |
| the opener (`attempts.py`) | 6 (6) | 19 (21) |
| the served drawer from the tour | 8 | 31 |
| the served door from the tour | **5** (7) | 28 (30), one fall |
| the served chest from the tour, its grip 2 cm short | 8 (6) | 32 (25), no fall |

The door falls under the bar of 6 at its first 8 draws (Ben's call). The
station, each trial its own noise: 36 of 36 round the standoff at a push
each, 139 of 140 seats, 16 of 16 lost near and walked in, lost far 114 of
123 -- the house's 71 of 72, across the street 28 of 36 (#622's
realization: 36, 138, 16, 16, 112). Swings of two or three in 32 are what
the draws give.

**The served chest** passed less often on this body until its lid's grip
was aimed 2 cm short of the knob's middle: from the tour over 32 draws at
three paired seeds it passes 29, 31 and 29 (15, 19 and 20 before, today's
body 20, 28 and 24) and none of the 96 falls ("The chest's grip", below).

**The LIDAR in falls.** Over the house's 20 falls at three seeds of the
noise, the LIDAR itself touched a wall or an edge in 3, 2 and 2 on this body
-- by a wall shoved right at every seed, at 1.9-2.6 m/s -- and in none on
today's (nor today's with the waist's mass); welded, 3, 1 and 2; on stand4,
2 and 0. So neither the waist's turn nor its give: it is how waist16 falls
by a wall. From a trunk turned 45 or 60 deg either way on a bare floor
(four ways, 30 and 45 N*s) the head struck no harder than square (at most
5.8 m/s against 6.0) and the LIDAR no oftener (1 of 32 falls turned, 1 of 8
square): a turned trunk does not lead a fall with the head, and nothing
brings the waist home faster as one begins. In the falls from a bow the
LIDAR touched the floor less often than on today's body (8 falls on the
standing policy alone where today's touched in 16; once as served on each).

**The waist in `--torque`** (an RS03: 60 N*m peak, 20 turning, 13 held
still): it holds nothing in the statues (it only turns); walking, 3.4-9.6
N*m at its peak (6-16 %; RMS 0.9-1.9, its winding 40-41 C), and on the
house's flight 10.5 up and 17.7 down; getting up from lying, its whole 60
in each, as the shoulders' do (RMS 13-19 over the 2-3 s it takes). ⚠ The
get-up's rows held the waist on the drawing's default gains, an eighth of
its own, until `Falling` held it as the mission does: then on its face the
get-up never stood in 30 s, at three seeds; held as served it stands from
its back, its front and its side in 2.9, 2.3 and 2.9 s, as today's body
(2.9, 2.2, 2.9).

**The cube put on its mat fell on the waist's body (22 of 32 to 9) for two
reasons, neither the waist's** (next section): the one realization of the
robot's noise a whole check flies, which the waist re-drew, and our step,
which leaned on stand4's creep.

## The cube and the robot's noise (issue #560, stage (c3))

**What is true now.** Each set-out of a check draws noise of its own,
keyed by its job and its number (`attempts.noise_of`; Evaluation.md §5,
"Each set-out flies noise of its own"). Until then every set-out replayed
ONE realization of the robot's noise -- each sensor seeded by the robot's
name, so each flight drew the same values at the same instants -- and a
count carried that realization's luck in common: our procedure placed the
cube on today's body 22 times of 32 as served, and 17, 16 and 21 times
with its joints' noise seeded three other ways. The joint senses dealt one
stream over every joint in order, so the waist re-drew every other joint's
noise: today's body, its noise dealt the waist's way, placed the cube 9
times, the waist body's own count on #622's pod. The legs' and the arms'
drivers draw one block of a key's stream in their order, as before the
waist, and a part added since draws its own (`senses.SHARED`; a test reads
each shared joint with and without the waist): a body change keeps every
shared part's noise, and one set-out is the same draws on either body.

**Found by elimination**, each arm switching off one difference between the
bodies -- set-outs 2, 12 and 16 first, then all 32 (locally, osmesa, a
process a flight):

| arm | placed of 32 | its carry failed |
|---|---|---|
| today's body | 22 | 2 |
| today's, the waist's 1.73 kg welded on, no joint | 18 | 4 |
| today's code on the waist's robot, its waist welded | 18 | 5 |
| the waist's code on that robot (stand4, every posture square) | 8 | 16 |
| ...its waist free | 5 | 15 |

(Welded, or its encoder exact, the waist's body placed it at none of the
first three.) Today's code and the waist's, flying the one welded robot,
part only by code: flown with the joints' noise drawn as today's and the
waist uncommanded, they were the same bit for bit until 5.25 s, where the
reach's solve on the two models of itself parted by 10^-11. Every joint is
read and driven by name (each reads within its backlash of its own joint),
commanding the waist moved nothing, and the noise's dealing alone moved the
step's sideways error over the 32 -- the true move less the legs' reckoning
-- from 8.2 mm to 4.5: the realization.

**And our step leaned on stand4's creep.** Under today's realization the
waist's body still placed the cube 10 times. Over the step stand4 turns the
stance 2.5 deg clockwise -- 1.7 in the walk itself, taken over from its
stance, the rest settling -- and waist16 0.4; the cube, 0.16 m to the
robot's left before it, lay 0.177 to its left after it on today's body and
0.157 on the waist's, where the left hand's carry to the mat ran into its
shoulder's inner stop ("held back", "out of reach"). The cube's procedure
steps in and aside now (`step(x - 0.32, y - 0.18)`, `hands_check.PLACE`).
Over three seeds, paired (the same draws for every shared joint):

| seed | today's, stepped in | today's, stepped aside | the waist's, stepped in | the waist's, stepped aside |
|---|---|---|---|---|
| 0 | 22 | 17 | 10 | 16 |
| 1 | 17 | 11 | 11 | 18 |
| 2 | 16 | 14 | 8 | 23 |

Stepped aside, waist16 lands the cube 172.5 +- 2.5 mm to its left (+-10.5
stepped in); stand4 goes on creeping after it (+-13 mm, and three of its
flights fell). Each body places the cube best stepped its own way, and as
often: 19.0 stepped aside on the waist's, 18.3 stepped in on today's. Each
set-out its own noise, the waist's places it 15 times of 32 (#560's seeds
stage, "The waist built").

- *A turned plan's hand lands as near as a square one's*: of 37 flights
  whose plans turned the waist 8 placed the cube, of 59 that never turned it
  29, but the true grip lies 4.7-6.9 mm off the believed one at the median,
  square or turned, the believed one 26-32 mm short of the cube's middle as
  the fingertips meet the board: it is the hard tries that turn.
- *The plan's bound on an arm's reach* (`reach.arm_reach`), measured the
  first time a process asks for it, left the planner's model on straight
  legs for the solve that followed: once a hand a process, a check flight's
  first reach and a served robot's after each restart. Measured on a pose of
  its own now (a test).

## The chest's grip (issue #560, stage (c3))

**What is true now.** The lid's procedure (`hands_check.LID`, the second
half of the served chest's reference) grips the knob 2 cm short of its
middle, back level along its facing. The jaws' pads are 8 cm long, centred on
the grip, so their tips reach 4 cm past it, and as the lid rises -- the
handle kept at its turn in the world by the hand holding the knob -- the
lid's front edge swings up and back under them. On the chest's and the
gripper's own geometry (`test_hands_check.py`'s scene: the two pads and the
lid as boxes, by separating axes), a grip at the knob's middle meets the lid
at 54 deg, 11 deg sooner each centimetre deeper and 3 deg sooner each
centimetre higher; the fingers' tilt hardly matters (51-54 deg from 34 deg
up to 5 deg down).

**Aimed at the middle, the grips landed 0.9 cm deeper at the median**, on
either body (0.91 cm on the waist's, 0.79 on today's; 0-2 cm over 32 draws
from the tour), so the pads met the lid at 43 deg. The lift then pushed the
lid's edge through them: across the hand at a median 35 N at its peak, it
turned the stance 5 deg and carried the knob 5 cm off in the robot's frame
on the waist's body (3 deg and 3 cm on today's), and the arc's top left the
reach: 12 lifts of 32 at seed 0 out of reach, where the pull's onward
postures -- standing taller, then bowing, five in 4 s at worst -- took the
lid to 50-57 deg or let it go, and four robots fell (today's body: no lift
out of reach, five falls).

**Back level, never along the tag's normal.** The knob hangs 34
deg round its pin, so `spot` reads its top face, tilted toward the camera: a
step back along that face's normal is 0.56 of it back and 0.83 of it up the
knob. 2 cm along it moved the grips 1.1 cm back and 1.65 cm up, the pads
met the lid at 51 deg, and the chest passed 27, 30 and 27. Back level, at
seed 0 (the waist's body):

| aimed short of the middle | passed of 32 | fell | the grip past the middle, median (range) | pads met the lid |
|---|---|---|---|---|
| 0, as before | 15 | 4 | +0.91 cm (+0.1..+1.9) | in 28 |
| 1 cm | 28 | 1 | -0.04 | in 28 |
| 1.5 cm | 29 | 0 | -0.58 | in 21 |
| 2 cm | 29 | 0 | -1.04 (-2.1..+0.1) | in 7 |
| 2.5 cm | 29 | 0 | -1.54 | in 4 |
| 3 cm | 29 | 0 | -2.06 (-2.9..-0.9) | in none |

From 1.5 cm on every lift that began ended at its distance, at 61-63 deg,
and the three draws left are the approach's. 2 cm keeps both edges: read 0.9
cm far, the pads clear the lid to 66 deg, and read true, the knob's far
corner is 4 mm inside the pads' tips (2.5 cm short, 0.7 mm past them). The
test pins both at each of the six tilts the procedure tries, where the pads
clear the lid to 63.5 deg at the least. Gripped so, the lift pushes 9-10 N
across the hand at its peak, and the stance turns 0.1-0.4 deg.

**Measured on one pod** (an EPYC 7Q83, which flies #622's served flights bit
for bit), the committed tree's own instruments, seeds paired:

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| the served chest from the tour (`solve.py --from tour`, 32 draws), the waist's body | **29** | **31** | **29** | **29.7**, no fall |
| ...gripped at the middle, as before | 15 | 19 | 20 | 18.0, 15 falls |
| ...today's body, as before | 20 | 28 | 24 | 24.0, 9 falls |
| the lid's check at 32 set-outs, the waist's body | 22 | 23 | 25 | 23.3 |
| ...gripped at the middle, as before | 21 | 18 | 21 | 20.0 |
| ...today's body, as before | 21 | 23 | 20 | 21.3 |

Each draw its own noise (#560's seeds stage; a Xeon Gold 6138, where seed
0's realization passed the chest 25 and the lid's check 22): the chest 32
of 32 from the tour with no fall, the lid's check 22 of 32.

From the mall's station the chest cost 8.11 Wh over 175 s (passed), and
8.543 over 183 s on 10 Nov's tree, which `energy.json` carries.

**What it leaves is the approach's.** Its first turn comes off the knob's
tag's facing read square-on from 0.75 m, and its walk turns the stance too,
so on the waist's body the stance lands a median 3 deg short of the turn
asked, within 7-8 deg of it either way in 8 approaches of 10 (219 flown);
then the last step is blocked, or leaves the robot 0.30 m from the knob
turned right of square with its tag out of the head camera's view. 7 draws
of 96 on the waist's body, 13 on today's.

## The table's reference on the waist (issue #560)

**What is true now.** Our table procedure (`hands_check.TABLE`, 2,481
characters of the language's 3,000) lays the place from ONE STANCE, 0.14 m
to the left of the start, where #621 measured the waist's yaw reaching every
piece: the cup by the right hand, the plate and the fork by the left, their
jaws asked the other way round and tilted 60 deg. Since #627 every reach is
aimed at its grip, and it lays **29 of 32 places at set-outs 0-31** on one
pod (an AMD EPYC 7B12, osmesa; "The reach planned where the live trunk
stands", below). As this section found it, its reaches aimed past their
grips, on one pod (a Xeon E5-2697 v4, osmesa) it laid 32 of 32 there and 32 of 32 at
#618's held-out set-outs, and on two sets more that nobody tuned on 46 of 48
(public 80-127) and 30 of 32 (the practice arms' held out): 140 of 144, no
flight falling, every pass 64-81 s. The procedure before it (`TABLE_STANCES`,
a stance a piece) laid none at 0-31 or at #618's there. It was tuned on
public set-outs 32-79 (46 of 48, locally), never on those it was read at.
Laid, a piece rests a median 0.4-0.7 cm from its spot, the cup 2.7 cm at
worst (once over the line). Three of the four misses were the cup's reach
bowing into a pocket where no tag is in view (below). What it took, each
found by tracing a flight -- the truth beside what the reach holds, read off
stored values only (a hand's place, read, draws on the joints' noise):

- *The live trunk stood 2-3 cm behind the statue the planner reached from*
  (1.6-2.1 cm standing, 3.1-3.2 after the step or with an arm out), so each
  of the plate's and the fork's reaches was aimed 3 cm on along the wall or
  the handle and the hand pulled back to its grip. Since #627 the reach plans
  where the robot reads itself standing and each is aimed at its grip ("The
  reach planned where the live trunk stands", below).
- *A pull goes on to another posture where its arm misses*
  (`posture_for(onward=)`), as a carry did not while it re-planned on a
  statue ahead of the trunk (#627 re-plans both where the robot stands).
  Every precise move is a pull from where the hand senses itself (`go`, a def
  of its own), a correction under a pull's least 1 cm a `move_hand` --
  skipped, a 1 cm error was the cup's whole margin.
- *The cup is taken from above, 3 cm short of its middle.* Its approach began
  3-5 cm from its pre-grasp, where the posture left the hand, and came in on
  the diagonal: a pad landed on the lid and tipped the cup before the jaws
  shut. Reached first at 10 cm over its grip, then lowered, it is not clipped
  -- and at its middle the palm, 4 cm back up the fingers, cuts into the
  lid's back, 6 mm with the set-out at its middle (the open hand as its
  drawing compiles, `test_the_tables_grips_clear_what_they_must`). Gripped
  near its top the cup slid down the jaws on its carry.
- *Where the cup is, after the reach's posture: the plate's tag.* The
  stance turns about 1 deg as the reach takes its posture, 8 mm at the cup,
  0.45 m to the side, where the jaws clear its 70 mm by 1 cm a side; the
  cup's own tag is out of view from the stance and the card's out of view
  with the arm out, so the plate's tag, straight ahead, is read again and the
  cup put where it was from it. Turned by the change in the plate's tag's
  edges as well, the cup came out worse (+1.35 cm along the place against
  +0.68): the edges read the turn too coarsely for an offset 0.4 m long.
- *The planner's bow above the cup had pockets* (#627 found them in the
  arm's solve): scattered points 5 mm apart took 0.55 or 0.8 rad, where no
  tag is in the head camera's view, so the reach was aimed 1.5 cm on from the
  cup's grip, and where it still bows deeper it withdraws, reads the tag
  again and reaches 1.5 cm higher -- which rescued 2 of the 5 the pod met in
  144 flights; the other 3 were its misses. Since #627 the reach is aimed at
  the cup's grip; the second try stays.
- *The plate is gripped 1.5 cm inside its wall.* The fork waits beside it, and
  at the wall the open outer pad came within 5 mm of the fork's handle and
  brushed it onto its side; 1.5 cm in, it stays 2 cm off, and the jaws, which
  close on their middle, slide the plate over until its wall sits between
  them (its middle then 8.4 cm from the grip, measured).
- *The fork is squeezed at 25 N*, chosen while the gripper's drive
  chattered and batted so light a handle from 40 N up; since #628 every
  squeeze from 10 to 120 N shuts on it ("The gripper's drive on its line").
  A grip that closes on nothing tries once more, lighter.
- *Each piece is set down by where its jaws left it, the card read again
  once it is lifted.* The jaws centre what they hold, so the cup and the fork
  sit on the grip's line (to 1 mm) and the plate's middle 8.4 cm from it,
  whatever the hand's own error; the stance turns 0.7-1.9 deg between a read
  and its set-down, and read before the lift, the plate came down 2.0 cm and
  the fork 1.3 cm to the diner's left. What the carry's turn still leaves --
  the plate 1 cm to the diner's left, the fork 0.5 cm toward the card,
  measured over 24 set-outs -- is set down the other way, and flown over 48.
  A card read again over the spot, the piece held, bought nothing: the cup
  then fell over in 5 flights of 24.
- *An open hand lifts straight up, its jaws opened first* (`wait(0.4)`):
  carried back as well, its move re-postured and dragged the plate by its
  wall; lifted while still opening, a pad rolled the fork onto its side.

| on the pod (each set-out its own noise) | places | cup | plate | fork |
|---|---|---|---|---|
| our procedure, set-outs 0-31 (the bar: 24, no fall) | **32 of 32** | 32 | 32 | 32 |
| ...#618's 32 held out | 32 of 32 | 32 | 32 | 32 |
| ...public 80-127 | 46 of 48 | 46 | 47 | 47 |
| ...the practice arms' 32 held out | 30 of 32 | 30 | 30 | 30 |
| the procedure before it, set-outs 0-31 | 0 of 32 | 21 | 18 | 0 |
| ...#618's 32 held out | 0 of 32 | 19 | 11 | 0 |

The same pod flew our procedure again traced, and its 64 rows came out
identical to the instrument's. ⚠ A flight is chaotic across machines: locally
(another CPU) the deep-bow pockets took the cup's reach in a few of every 48.

![Our procedure at the table's set-out 0, from across the table: the cup
carried by the right hand from above (10-20 s), the plate by the left (30-40
s) and the fork after it, the waist turned as the reach takes it, all from
the one stance -- the plate 0.7 cm, the cup 0.8 and the fork 0.5 from where
each goes (`MUJOCO_GL=osmesa uv run python scripts/hands_check.py --strip
table 0`; `--view table K` watches one)](figures/hands_table_waist.png)

**What it cannot say.** Every number is one pod's CPU; the place is the
easy mode's, the dishware a few cm from where #615 set it out; and the
reference knows the set-out's geometry -- the stance, the plate's wall, the
cup's height -- as no mind is told.

## The reach planned where the live trunk stands (issue #627)

**What is true now.** A reach plans on the robot's model of itself as it
reads itself standing (`reach.Lean`, `Reach._lean`): the statue of each
posture it weighs is balanced with its centre of mass where its own model,
posed off its encoders and its IMU, puts it over its soles, shifted, turned
and rolled as its pelvis stands off that statue, the posture it holds posed
as it holds it, the other arm where it is commanded and the reaching arm
where its solve puts it (`reach.statue`, `plan`'s `_reached`). Once the
posture is held and the arm is out at its pre-grasp, it reads itself again
and takes the least bent posture that puts its goal in reach -- deeper or
less bent -- before the hand goes in (`Reach._look`, at most `LOOKS`, never
a posture it has left; the hand waits at its pre-grasp meanwhile). The
carry's and the pull's postures (`posture_for`, `path_for`) stand on the same
statue; `PATH_MARGIN_M` is gone, and a path's points further from the
shoulder are reached `LIVE_MARGIN_M` (2 cm) ahead on a statue as on the live
robot. The arm's solve keeps the converged arm nearest the arm as it is, or
hanging, and each of the goal and its slack point is solved from the other's
arm too.

**Measured** (`humanoid_spike.py --lean`: the living room, as `--friction`
stands; each true shoulder against each statue, cm ahead, stored values
only):

| | the statue before (over the soles' middle, arms hanging) | the statue as the robot reads itself | the robot's own model |
|---|---|---|---|
| the stand | -1.6; -2.0 | 0.0; 0.0 | 0.0; 0.0 |
| each bow, -0.08 to 1.1 rad | -1.5 to +0.9 | -0.1 to +0.1 | 0.0 to +0.1 |
| after step(0, 0.14) | -2.2; -2.5 | 0.0; 0.0 | 0.0; 0.0 |
| ...the other arm out of the way | -3.2; -3.1 | 0.0; 0.0 | 0.0; 0.0 |
| ...the hand at the table's plate | -5.9; -5.1 | 0.0; 0.0 | 0.0; 0.0 |

(left; right; sideways the read statue is within 0.1 cm too.) #626's 1.6-2.1
cm standing and 3.1-3.2 cm after a step or with an arm out are the first
column's.

- *The robot reads its own lean to 0.05-0.2 cm.* At every posture, step, arm
  and noise draw measured, its model of itself posed off its encoders and
  its IMU put its shoulders and its centre of mass where they truly were.
- *What the lean is.* The standing policy holds its centre of mass 1.2-1.3
  cm behind its soles' middle at the stand and almost none in a deep bow;
  the statue holds it over the middle. An arm held out moves the statue's
  pelvis back 0.7-2.2 cm for its weight, and the policy goes on: at the stand
  0.6-0.8 cm more of its weight back for each cm the arm's weight moves it
  forward, in bows a tenth to a third of that, by side.
- ⚠ *No table predicts it.* Over 24 random arms at each of two postures the
  policy's weight moved -2 to +5 cm, and the best fit to the arm's angles
  missed by 0.7-1.7 cm rms (to its weight alone, 1.4-1.7). An arm over the
  top -- the shoulder thrown back, rolled past 2.2 rad -- leant it 4.4 cm
  FORWARD where an arm in front, as heavy, leant it 0.8 back. So the plan
  reads the lean, and reads it again with the arm out.
- *The first plan, read before the arm goes out, is 1.5-2.4 cm hopeful* for
  an arm in front (the policy's own lean for the arm, which nothing but the
  arm out shows), and too cautious for one over the top: the look takes the
  posture each needs. Reaches from the stand on a bare floor (the plate's
  grip, the other arm hanging and held out of the way): before, blocked 3.2 and
  5.1 cm short; now held within 0.1 cm, after one look.

**The bow's pockets were the arm's solve.** At 10 cm over the table's cup,
points 5 mm apart took the 0.3 rad bow or the 0.55 and 0.8: none of seven
seeds converged there for the goal or the point its slack asks for -- each
missed by 0.4-0.5 rad of turn, its wrist on a stop -- where each solved from
the other's arm did (15 of the 18). And with no arm to be near, the solve
kept the FIRST seed that converged: the hanging arm's own, over the top,
where five others reached in front. Now the grid over the cup takes the 0.3
rad bow everywhere but one corner, where the 0.55 truly reaches and the 0.3
does not with its slack (`posture_map.py` of #626).

- ⚠ *The cup's asked grip is reached only over the top* (its arm in front
  misses by 0.44 rad, its wrist's roll at its stop). The first plan's
  caution bows it 0.55 in about half the set-outs, and the look, reading the
  arm's forward lean, brings it back to 0.3. Asked half a turn round, an arm
  in front reaches it standing, the waist turned -- but the pull down onto
  the cup then took another posture, and the turning trunk swung the giving
  hand 11 cm. Our reference keeps the grip as asked.
- *The table's reference aims at every grip.* The 3 cm past and the pull
  back went with the statue they worked round, and so did the cup's 1.5 cm
  on: it now took the cup's reach to the 0.55 bow's edge.

**The planner's cost**, on the physics thread: a statue balanced at a lean
read is the kept statue's legs moved by their own change with the weight's
place (`_slope`) and two Newton steps (`_posed_fast`): the least squares'
answer to 0.01 mm at 0.3 ms where it takes 12. A plan over the cup 0.18 s
(0.21 before), one past the arm's reach 0.06 s (0.02), a carry's re-plan
0.03-0.05 s (0.01).

**Flown on one pod** (Vast, an AMD EPYC 7B12, osmesa: the tree before,
07a1d2f, and this one at the same seeds, each set-out its own noise; of 32,
the first 8 in brackets):

| | before | planned where the robot stands |
|---|---|---|
| the lid | 22 (8) | 26 (8) |
| the drawer | 22 (5) | 21 (5) |
| the door | 25 (5) | 24 (7) |
| the cube placed | 15 (4) | 17 (3) |
| the table laid | 32 (8), its reaches aimed past their grips | 29 (8), aimed at them |
| the served drawer, from the tour | 30 (8) | 30 (8) |
| the served door, from the tour | 28 (7), a fall | 29 (8), a fall |
| the served chest, from the tour | 32 (8) | 32 (8) |
| the station, round the standoff; at its seats | 36 of 36; 139 of 140 | 36 of 36; 140 of 140 |
| the opener at its counter (`attempts.py`; before, 8ba89ce) | 19 (6) | 20 (7) |

Draw by draw the drawer lost 2 and won 1, the door lost 5 and won 4, the
cube lost 7 and won 9, the lid won 4, the opener lost 2 and won 3: a change
to any reach reshuffles which draws pass, and a sign test over the pairs
tells none of them apart. The served door's fall is draw 8's on both trees.
- ⚠ *The opener aimed at the knob's middle, and the reach now gets there.*
  Its reaches held their goal 40 times in 73 where they held it 29, but 13
  of its 40 grips did not hold the knob where 1 of 29 did: arrived short,
  its hand had gripped where the lid check aims on purpose, 2 cm short of
  the middle (the pads reach 4 cm past the grip, "The chest's grip"). It
  grips so now, back level along the box's facing (`opener.SHORT_M`, off
  the depth's fit: a square-on tag's normal swings +-7 deg): 22 of 32 (7
  of 0-7), losing none the tree before opened -- on the dev box, where both
  trees flew chest for chest as on the pod; back along the tag's normal, 23.
- *The look costs no time it does not use.* A posture the look would
  choose that is the one held lets the hand straight in, and paired by
  set-out the checks' passes took no longer: the median pass's time moved 0
  to -1.2 s a job, the slowest lid's 41 s to 30.
- *The table's three it lost.* Two forks let go as the carry turned the
  waist to -60 deg, the fork turning 55-64 deg in the jaws; a reach's
  posture is chosen for its grip alone, and the carry after it may need
  another. One cup (set-out 28) was knocked over as its pull down went on
  to a 0.95 rad bow: its arm over the top, whose lean (above) no plan
  foresees. With the cup's 1.5 cm on that #626's reference aimed, 19.

## Where `measure` looks (issue #630)

`measure(x, y)` reads the highest flat top among the D435's points within
`MEASURE_NEAR_M` (0.45 m) of (x, y), and its ends where those points end
(`perception.box`). It came with #493 at chests whose top the window held
whole; the served robots aim it at anything, and two of its answers were
where the look ended, not the top. Probed at the drawer's set-outs 0 and 3
from the reference's stance, on the live body:

- **The window's rim.** A top that runs on past the window ends, in its
  points, at the rim: from (0.9, 0) Luca's counter read `object.near`
  0.453-0.456 however it stepped, and it crept on that into the counter;
  at set-out 0 the cabinet on the counter read its near end 0.46 there.
  Of 96 ends read from 12 points at set-outs 0 and 3, each against the
  true edge of the top it was (the cabinet stands turned 5-9 deg), the
  points an end is read off (`box.edge_read`) stood, at their median,
  0.9-2.0 cm inside the rim at the 24 ends the window cut short, and 3.9 cm
  and more at the 28 whose edge it held whole. At the 42 whose edge it held
  only in part they stood 1.5-34 cm in, and where the camera saw that part
  the reading was its: the cabinet's turned left edge from (0.9, 0), a
  fifth of it held, read 0.358 where it lies at 0.357-0.361. So an end whose points stand within
  `MEASURE_RIM_M` (3 cm) of the rim, halfway between, is no reading, and
  its read says the top runs on past the window that way. That costs 8 of
  the 42 held in part, 4 of which read within 4 mm, and on a synthetic top
  an end 2 cm inside the rim put its points 2.5-3.1 cm in and one 3 cm
  inside 3.1-4.1: a true end within about 2 cm of the rim goes with the
  window's, and reads from nearer.
- **The floor.** The D435's noise moves a point along its ray by 0.0036 z²
  of its axial depth (`depth.NOISE_K`), which lifts the floor's own points:
  windows of bare floor 1.5-3.0 m off held 12 000-61 000 points, spread
  7-13 mm about a level within 7 mm of the true floor, the highest 3.2-5.3
  cm up and at most 8 over 5 cm. Counted from 1 cm, their tail's highest
  bin holding a quarter of its fullest stood 1.2-2.6 cm up, and every window
  read as a flat top there (Rowan's chest_diag read 0.0168 m). From 6 cm
  (`box.FLOOR_M`) none does, and a box's top 0.10 m up still reads.

What is true now: the verb's text says the window and the floor, an end
the window cut is left out of the verdict (`runsOn` names it) and of
`object.*`, and every `object.*` read with no number says why -- the
window, a look that saw no top, or no look yet.

**Where the look ended inside the window (#636).** Two answers were where
the camera stopped seeing, not where the top ended: the picture's edge --
from where the references work at the counter, its front (0.22-0.24 m ahead)
is under the picture's bottom edge at the counter's height, and its near end
read 0.26 wherever `measure` looked -- and a taller thing in front -- from
(0.95, 0.9) the counter's right end read 0.56-0.57 where the cabinet on it
hides its top. No count of pixels tells them from a true end: a near end the
picture cut stood 9-14 px from its border, the counter's true left end at
the picture's corner 14-16. What tells them apart is whether the camera saw
past the end, so `measure` now asks the picture beside the points
(`Body.depth_pictures`, `box.past_end`). Each of an end's points (at most
400) is walked outward, pixel by pixel, along where the end's direction
falls in the picture through the camera's own pinhole, and told by the first
thing past the top it walks over; the walks' most common word says what the
end is (`steps.MEASURE_WORDS`, a tie the one that claims less):
- **lower**: the top stopped there (its own face going down, the floor).
  It reads as before.
- **taller**: something stands there and hides what lies beyond. It reads as
  the end of the top's free surface (Ben, 2026-11-08), and
  `object.<end>_taller` says so.
- **cut**: the walk left the picture with the top still on its last pixels
  (`PAST_CUT_PX`, 3, unread pixels at most between them: a dropout's
  length). No reading, and it says so.
- **runs**: the walk's own top went on past where the end reads by more
  than `PAST_OVER_M` (3 cm), the end's points having stopped short of the
  top: 9 of 2,399 ends past the rim's check at set-outs of five checks, the
  counter going on beside the chest, read a few cm short, the world having
  the top going on along most of each. No reading, and it says so.

A pixel that reads nothing is walked past: the D435's shadow (the band of
nothing beside a near thing's image-left), a dropout, or the robot's own
body (a cloud is told nothing of whose body a pixel was, and another robot
reads as anything taller would), what stands behind it saying which. Unread
right to the picture's edge, past 3 px, the top stopped: a flat top running
on would have read on, and what is past it is out of the part's range or in
its shadow (the counter's left end at the picture's corner; a table's far
edge across a room). An end whose points stand at the part's farthest or
nearest reading (`PAST_REACH_M` of its 3 m, `PAST_NEAR_M` of its 0.17) is
the range's: no reading. A pixel's slack is the noise IN ITS HEIGHT, the
noise along its ray times the ray's rise (`DepthPicture.rises`): the ray's
whole noise, 2.4 cm at 1.5 m, passed a face 5 cm behind a far end for the
top running on. And a lone pixel past that slack is the noise, never a
surface (`_agreed`, the walk stepping a pixel of its longer axis at a time
so none is read twice: a surface is several pixels wide, and a walk of a
hundred pixels of the top meets a 3-sigma one in a quarter of walks).

Measured with `scripts/measure_spike.py` and a sweep of the same code
(`--poses all`): the drawer's set-outs 0 and 3 from the reference's stance,
96 ends: every near end the picture cut (0.261-0.264) is no reading now; the
28 the window holds whole read exactly as they did; the counter's left end
at the picture's corner reads 0.838 and 0.834 (the truth 0.836 and 0.830,
its walks 29-34 % cut); the cabinet end reads 0.561 and 0.572 and says so.
Over the five checks' set-outs 0 and 3 from eight viewpoints and 42 windows
(9,076 ends), a ray straight down 2 cm past each end read against the world:
of 4,365 reads, 4,241 stand where the top truly stops; 100 stand at a taller
thing and say so, and 22 beside one and do not (a chair 5 cm behind a table's
far edge: the camera sees down the gap first); 2 stand where the top truly
goes on, edges seen from a stance turned 25 deg, a diagonal edge whose
reading lies between its ends (and before #657 24 more, one dining table's
right edge and far edge, read through its top: below). 438 ends went unread
at the picture's edge, 4,257 at the window and 16 past where they read.

⚠ **A top reads high at range.** `box.top_height` takes the band round
the highest bin holding a quarter of the fullest, and `TOP_BAND_M`'s 5 mm
is narrower than the noise past a metre: on the synthetic floor, a box's
top 0.10 m up read 0.107 at 1.5 m and 0.116 at 2.6 m.

## A ray through a body MuJoCo's cull left out (issue #657)

**What is true now.** `mj_multiRay` returns what `mj_ray` returns on every
body welded to the world: each stands on an inertial frame unturned on the
middle of its box, its mass and inertia as compiled
(`perception.cull.static_inertials`) -- the house's nine written into
`models/home_world.xml` by its generator (`home.world`), what a world puts in
at load (the chest, the drawers' cabinet, the cupboard, the stations, the
signs) set on its spec (`fix_static_inertials`, in `home_spec` and
`counter.model_of`), and a prop's redraw copies only what moves
(`storeroom.redraw` copied the carcass's frame back too, at every offer).
`tests/test_cull.py` reads every such body in the house's file and each world
built at load, and casts the ray this was filed on and a fan at each dining
table both ways. A body that moves keeps its true inertia and this bug: from
rings of viewpoints, at worst 8 of a fork's 689 rays (1.2 %), a robot's
pelvis 6 of 3,696, the chest's lid 1 of 3,908 -- which is why the cage's own
sighting stays one `mj_ray` a ray.

**What it was.** `mj_multiRay` tests a ray against a body's bounding sphere
before it looks at the body's geoms, and MuJoCo 3.10 centres that sphere at
`xipos` plus the box's offset unturned (`mju_singleRay`:
`mju_add3(center, pos, d->xipos+3*b)`), though the box is kept in the
inertial frame. The compiler turns that frame to the body's principal axes,
so wherever a body's box is off its centre of mass and its frame is turned,
the sphere stands `|ximat c - c|` off the body: 0.353 m on a dining table on
legs (a 120 deg turn), 0.524 m on the stairs. `mj_ray` tests the box itself.
From rings of viewpoints round each body, the depth camera's pattern and the
LIDAR's, the rays landing on a body that `mj_multiRay` sent on through it:

| body | sphere off by | D435 rays through, of those on it | LIDAR |
|---|---|---|---|
| `dining_table_0` | 0.353 m | 3,098 of 38,260 (8.1 %) | 187 of 2,906 (6.4 %) |
| `lab_cage` | 0.088 m | 433 of 17,399 (2.5 %) | 33 of 1,164 (2.8 %) |
| the drawers' cabinet | 0.031 m | 54 of 9,726 (0.6 %) | 4 of 727 |
| the chest | 0.028 m | 21 of 4,268 (0.5 %) | 2 of 280 |
| `loading_rack` | 0.113 m | 25 of 36,510 | 5 of 2,818 |
| `kitchen_shelf`, `shop_shelf`, a station | 0.021-0.171 m | 4-18 of 13,500-72,000 | 0 |

The stairs' sphere, 2.7 m across, lost none. Every one of these is 0 now, and
`measure` over the same sweep (above) reads the dining table's right end
where it is: the floor seen past a table's edge is the table's edge again.
To a depth picture a table with a hole in it has ended there, and `measure`
could not tell: it read that edge up to 7 cm short.

**Why the frame and not the cast.** Welded to the world a body's inertia moves
nothing: a limp humanoid dropped onto a seated table, a seated shelf and the
cage flew the same 2,500 steps, bit for bit, with and without it. Unturned
alone would not do: a body yawed in the world turns a horizontal offset (the
shelves' centres of mass lie off their boxes' middles), so the frame stands
on the box's middle, where there is no offset to turn.

## The table set served (issue #560)

**What is true now.** The restaurant's dining tables are a top on legs and
table 0's south chair is away (`home.world`), the place's dishware is
compiled into the humanoids' house put away under the floor
(`home.restaurant`, `dishware.park`), and `set_table` is built -- an offer
sets the place out where its draw lays it, as the table check's set-out
does, so the practice's place and the job's are one (Challenges.md §12) --
and offered since its reference, a walk to the place and the table check's
procedure from its one stance (`challenge/solutions.py`), passed its bar
over the tour ("The lay from the tour", next). Put away, the dishware costs
the served pair nothing measurable: the same walks, alternated on one pod
(an AMD EPYC 7B12, one core a process), ran at 0.913-0.916x real time with
it and 0.902-0.905x on the tree before it. What flying the walk took:

**One tag gives no facing.** The place is found by the card's tag (93) alone.
A find stops at its tag's first sight, and from the house's station that was
4.3 m off, so its facing was where it was seen from: 51 deg off the card's
face, and stood 1.3 m out along it, `spot` could not decode the card at all.
One tag's own rotation is believed only from a look 40 deg and more off its
face (`mapping.places`: within 0.5 deg there), and the directions say the
card faces south: so the walk looks at it from 45 deg south-west of its face,
1.12 m out, reads its face's normal there, and turns once to square to it.
⚠ `spot.n*` is INTO the tag's face, away from the eye: the turn to face
along it is its own angle.

**What `step` does, measured** (the same probes, in the dining room, stood
still first):
- *A small turn was lost.* Asked 1.7, 2.9 and 5.7 deg, the stance turned
  0.0, 0.3 and 1.1, its turn stopped on the pelvis's heading with the feet
  not following its last 2-4 deg -- asked 34 deg, they turned 32.6 left and
  30.2 right. (Closed on the stance's heading since #649, "Stepping toward
  the work"; so the walk's one big turn was asked `WIDER` than read.)
- *A step aside to the right turns the robot 1-1.5 deg clockwise*, by its
  own reckoning, which agrees with the truth (two flights at one noise,
  apart at their last step, differed by exactly what it reckoned).
- *A step with a step to the right in it lands long*: 28 cm ahead and 5-10
  cm right landed 2.0-3.7 cm further ahead than asked, straight ahead
  0.4-0.8 cm, with a step to the left in it 0.1 cm long to 1.3 cm short.
- *A step forward under a stride is walked as one or not at all*
  (`_step_walks`: a forward speed in the walker's dead band is walked on its
  own, a landing moving the stance about 5 cm): stepping out a centimetre's
  error ahead walked ten, and from the exact start a step `(k - 0.4866, m +
  0.14)` laid 5 of 8.

So the walk's last 30 cm is straight ahead off two looks averaged; its turn
is asked 3.3 deg wider than read (flown at 3.3, 2 and 1.1 deg wide the
reference read 12 of 32 draws each, the stance landing 3.9, 2.6 and 1.75 deg
left of square by its reading of the card, and 3.6 at 3.3 on the tree before
#649's turn, so it is no compensation for a short turn). What it leaves
across the card's line and how it ends up turned, "The lay from the tour"
has.

## The lay from the tour (issue #560)

**What is true now.** Our table reference lays the place from anywhere in
the house: **24 of 32 draws over the tour** (`solve.py --feature table --from
tour --n 32`, one pod, an AMD EPYC 7282, osmesa, each draw its own noise; the
bar, written before it first flew), where the tree before it read 12 on the
same pod, and the table check at set-outs 0-31 reads 24 where it read 27 (its
bar 24). It is at the bar's edge: two sets of draws nothing was tuned on,
400-431 and 500-531, read 21 and 25, 46 of 64. A try in which the robot fell
is no pass: of the 32, 4 fell, two on the walk (22 and 55 s in, draws that
fell on the tree before as well) and two on the left arm's way home. Its
passes called 194-279 s after the claim, so the job gives 14 minutes, three
times the slowest rounded up, and pays 79 (Ben's 90 for 16 minutes, pro rata);
from the dearest station, the research center's, it cost 9.820 Wh over 199.2 s
(`energy.json`; from the house's 7.618, from the mall's 4.762).

| the 32 draws of the bar, how each ended | the tree before | now |
|---|---|---|
| **laid** | 12 | **24** |
| the cup's tag out of view at the lay's first look | 5 | 0 |
| an arm that did not get home | 3 | 0 |
| a grip closed on nothing | 2 | 0 |
| a step too short to walk | 1 | 0 |
| the cup 3-4 cm off its place, or 16 | 4 | 3 |
| the fork off its place, dropped or knocked over | 2 | 1 |
| fell at the table | 1 | 2 |
| fell on the walk | 2 | 2 |

![Our reference at draw 0 from the house's station: the walk there (0-120 s),
the oblique look and its steps in (146-156 s), the cup reached for (160-168
s), and the place laid (232 s) (`MUJOCO_GL=osmesa uv run python
scripts/solve.py --feature table --from station --draw 0
--strip`)](figures/solve_table_tour.png)

What flying took, each measured from the flights' own records (the rows and
probes are in `~/pluggybot-training/issue-560-table-lay/`):

**Where the lay can stand is a patch, and its old stance stood at the patch's
corner.** Flown from a stance set directly -- its first looks replaced by
what they would have read, so that only what comes after flies -- at set-outs
40-47, not turned:

| nearer the table than the check's start | 14 cm left of the card's line | 16 cm | 18 cm |
|---|---|---|---|
| 2 cm further back | 5 of 8 | 3 | 1 |
| at it | 2 | 8 | 2 |
| 1 cm nearer | -- | 8 | 8 |
| 2 cm nearer | 3 | 8 | 8 |
| 3 cm nearer | -- | 5 | 6 |

Further left or further back, the cup's reach bows the head past where the
plate's tag can be read; less far left, the fork's reach, past 19 cm from
the stance, takes a square crouch in place of a waist turned 30 deg and its
carry drops it. So the walk ends about 1 to 2 cm nearer than the check's
start (`NEARER`, less what the lay's step gives back) and the lay stands 15.5
cm left of the card's line (`TABLE_LAY_STEP`; the check, starting on the
line, asks 15). Turned 2 deg right to 4 left barely mattered there (7 or 8
of 8 at each degree, 2 cm nearer and 17 cm left), where at the old stance
turned 1 deg right it laid 3 of 8.

**A step aside walks in whole strides.** Asked from a robot that has just
walked, 9.5 to 16 cm went 13 to 16 cm, and 18 to 20 usually 21 to 23: what
it lands on barely depends on what was asked, so a stance is set by where
the step starts from. The walk's two steps across left the stance a mean 2 cm
left of the card's line, standard deviation 4, up to 14: so at its last two
looks it reads the card's across and steps out anything over 3 cm before it
goes in (from 10 cm left of the line the lay's first look could not see the
cup). Where the first look loses the cup all the same, the lay turns 10 deg
to its right, reads the cup and the plate there, and carries the pair
through its turn back.

**What the lay reads before its step, it carries through the step's own
turn.** The cup is out of view from the stance, so the cup's place from the
plate's tag and its spot from the card are read before the step; a step aside
turned the stance 0.5 to 2.5 deg, 0.8 cm at the cup a degree, and `step.turn`
says that turn to 0.2 deg against the truth (0.07 to 0.2 over 32 flights).

**The walk squares its heading by its own reckoning.** Its turn landed up to
3 deg over, and its steps aside added up to 2 more, so about one flight in
fifteen stood 5 to 8 deg left, where its pieces went wide. It counts each
step's turn against the one the oblique look asked (`a`), and past 1.5 deg
off facing (`FACING`, 1.5 deg left of square) turns the difference.

**The cup's reach is aimed inside the band its posture holds.** A reach takes
the least bent posture that reaches, and only one of them, height -0.16 and
pitch 0.3, carries through the cup's way in: its pre-grasp between `0.7 y -
0.72 x` = -0.597 and -0.577 (m, the stance frame). Nearer, the planner stands
it up and the way down then bows it; further, it bows 0.55 rad and the plate's
tag leaves the view. Aimed inside that band, a pull carrying the hand on over
the grip, a grip closed on nothing fell from 10 flights in 64 to 1 in 64.

**The left arm lifts 9 cm before it goes home.** The side way home sweeps
the hand past the table's edge, and from 2 to 4 cm nearer than the check's
start a jaw struck the top and the robot fell: over the same eight stances,
26 of 64 flights laid the place lifted 6 cm and 59 of 64 lifted 9. At 12 the
lift itself ran out of reach (18 of 53 lifts) or moved the feet (6), and 3
fell.

**The cup is set down 7 mm nearer the robot than the place's numbers say.**
Its own carry left it a mean 1.2 cm too far toward the card over the
development flights, which the 2.5 cm allowed cannot spare; centred, 2 of 59
cups measured on two development sets were out by more than that, where 9 of
60 were.

⚠ What the flights still lose: over the bar's 8 misses, the cup off its place
3 (4, 4 and 16 cm), a fall on the left arm's way home 2 and on the walk 2, the
fork off its place 1. Where the plate's tag is not seen after the cup's lift,
the set-down falls back on the grip's own arithmetic, and on the development
flights 6 of the 10 that did missed. The walk's two falls are the walker's.

## The gripper's drive on its line (issue #628)

**What is true now.** Each of the Dex1-1's jaws carries 3.6 kg of reflected
inertia (`parts.Gripper.jaw_armature`), the least that holds its drive's
line on the 2 ms step: a jaw's mechanical time constant on the line -- its
inertia over the line's back-EMF, K²/R = 1778 N·s/m -- is 2.05 ms, a step
and a little more. The drive keeps to its line. In free air every squeeze
from 5 to 120 N shuts in 0.33-0.38 s, the line's own speed at the 43.2 V
bus, never faster, and its force never flips between its clips; a light
handle lying 8 mm off the jaws' middle is shut on at 40-120 N as under 40
(`tests/test_humanoid.py`). And from the state each of our table
reference's grips began in, every squeeze from 10 to 120 N closes on its
piece as under 40 N, where the old drive let the plate's wall go from 50 N
and the fork's handle from 50 (held of set-outs 0-31 on one pod;
`MUJOCO_GL=osmesa uv run python scripts/humanoid_spike.py --grips 0-31
--jobs N`):

| piece | jaws | 10-35 N | 40 | 50 | 60 | 80 | 100 | 120 | shuts whose force flipped |
|---|---|---|---|---|---|---|---|---|---|
| cup | before | 31-32 of 32 | 31 | 30 | 29 | 28 | 29 | 30 | 349 of 384 |
| cup | built | 32 of 32 | 31 | 31 | 31 | 31 | 31 | 31 | 0 |
| plate's wall | before | 31 of 31 | 31 | 22 | 5 | 1 | 2 | 3 | 372 of 372 |
| plate's wall | built | 31 of 31 | 31 | 31 | 31 | 31 | 31 | 30 | 0 |
| fork's handle | before | 30-31 of 31 | 31 | 26 | 25 | 23 | 17 | 6 | 372 of 372 |
| fork's handle | built | 31 of 31 | 31 | 31 | 31 | 31 | 31 | 31 | 0 |

The built drive's misses are not its own: one cup, at set-out 28 -- where
our reference's own grip misses and tries again, its flight never reaching
the plate or the fork -- is gripped at its edge and squeezes out from 40 N;
and the plate's 8 mm wall at 120 N read not held at one -- shut on it the
jaws stall 10 mm apart, where `GRIP_KP`'s spring pushes under the half of
120 `held` then asked. The table is the rule of its day: since #646 (below)
the same command reads the wall held at 120 N at every set-out.
`humanoid_spike.py --view jaws` shows the fork at set-out
0 shut on again and again from one moment, at 40, 80 and 120 N, its jaws as
they were and as built.

**What it was.** At 0.2 kg a jaw -- stated "so the 2 ms step stays stable
at the stiffness a squeeze commands", against the PD and never the line --
the time constant was 0.13 ms. MuJoCo 3.10's `implicitfast` takes a
`dcmotor`'s velocity term into its implicit step only while the drive is
unclipped (`mjd_actuator_vel`): at its force clip the term is skipped, and
at its voltage clip it keeps the controller's own term, netting -300 N·s/m
where the line's is -1778. A shut -- the target stepped from open to shut
-- travels at both clips the whole way, so the line was integrated
explicitly and overshot within every step: the force flipped between +clip
and -clip, in free air too, and the jaws' speed swung between two values
whose mean nothing set (a cycle exists for a whole band of speeds). They
ran at up to 8.2x the line's top speed, shutting in 0.13 s at 120 N where
the motor needs 0.33, and wandered. Against the fork at the table, from the
one state its shut began in (set-out 0, on the dev box), the jaws stood 4-8
cm apart 1.5 s later at 40 to 80 N, a pad on the handle for 3-13 steps of
750 and never squeezing it (the table above has it at every set-out). ⚠ In
the first steps the jaws' coupling (`JAWS_SOLREF`, 4 ms) lets the motor
drive the first jaw alone, so it is ONE jaw's inertia that must hold the
line: at 1.0-2.0 kg a jaw the 120 N shut still ran past the line's speed
(1.2-2.3x), from 2.5 none did.

**Why the inertia, and not a ramped shut.** A target ramped at a jaw speed
keeps the drive off its clips while it travels, and the chatter off that
path -- and leaves the line explicit, and unstable, wherever the drive is
clipped with the jaws free to move (a grip lost under its squeeze, a jaw
pushed open). The line is the part's, and so is its inertia: no motor
spins its rotor up in 0.13 ms. The rotor behind the Dex1-1's 1:25
reduction is unpublished, and any rotor its size reflects more through
1:25 and a 7 mm pinion; 3.6 kg is the least that holds.

**What it costs, and what it does not.** A heavier jaw arrives at the same
speed with more momentum -- at a light squeeze, fifteen times the light
jaw's energy -- and the station's unplug squeezes a seated plug its catch
lets go of past 20 N. Flown on one pod beside the tree before it, each
flight its own noise: the station's approach docked, unplugged and parked
its plug 39 of 39 in both, and its seats 140 of 140 in both, every plug-in
at its first push, no fall (`station_spike.py --approach --n 36`,
`--seats`). The hands checks at set-outs 0-31 moved inside their own noise:
the lid 26, the drawer 21, the door 24 to 25 and the cube 17, against
25-26, 21, 24 and 16-17 for the tree before on a second draw of its noise;
the table, over eight draws a tree, 221 of 256 against 222, 13 set-outs
lost and 12 won (its falls, 12 against 6, Fisher's p 0.23; the two traced
each the carry's reach taking a deeper posture with the fork still in the
air and driving it into the table, a posture any flight may reach). The
served drawer, door and chest from the tour, 64 draws each, 60, 58 and 61
against 57, 56 and 60; the table from the tour (not offered), 44 of 128
against 52, 24 draws won and 32 lost -- half its losses (16) its walk
landing out of the lay's window before any grip, as 10 of its wins were the
other tree's. Not the chatter's: the squeeze's running mean
(`arms.SQUEEZE_S`) -- held, the drive's squeeze is steady under either
inertia (sd 0.000 N) and what the gripper reports spreads 5.3 N, its
current sense's stated 5.1 -- and the lever that could not turn at 120 N,
its pads' torsional friction (Hands.md).

**The practice arms' grips on nothing (#560, PR #642).** The two DeepSeek
arms' 70 tries flown again, each from its set-out, its noise and its
source, on the tree before this fix -- whose table stands where the served
one does since the arms flew (#643), and the tries came out alike, 98 grips
and 54 on nothing, where the arms had 100 and 56 -- and each grip landed
again from the state it began in under the drive as it was and as fixed: 13
were the stall -- the fixed drive held where the old closed on nothing,
every one the plate's wall at 100-120 N -- 41 were misses, no piece between
the jaws under either, and 43 held under both. One, at 120 N, held under
the old drive alone: its chattering jaws caught the fork by chance where
the clean shut met nothing. So of their failed grips at 40 N and up, a
quarter were the drive's and three quarters their own reaches'.

**What it cannot say.** The inertia is stated, not measured: the real
gripper's may be several times it, and its jaws then slower to start and
harder to stop -- the procedures' grips were never flown so. Every count
is one pod's CPU (an AMD EPYC 7702, osmesa).

![The fork at the table's set-out 0, from the moment our reference's left
hand shuts on it, shut on again at 40, 80 and 120 N (rows), the jaws as they
were before #628 (left) and as built (right): before, the force flipped
between its clips 108-749 times in 1.5 s and at 80 N batted the fork askew
and stood 39 mm apart; built, it flipped none and shut on the fork every
time (`MUJOCO_GL=egl uv run python scripts/humanoid_spike.py --strip jaws`;
`--view jaws` watches it)](figures/humanoid_jaws.png)

## A hard squeeze on something thin (issue #646)

**What is true now.** A grip's word judges the squeeze against what the
gripper's drive pushes at the gap it reads (`arms.pushes`, `Gripper.held`):
the force asked -- its torque limit -- or, short of it, its spring's,
`GRIP_KP` on the jaw its motor drives, `gap / 2` from shut, half of it a
pad. Its own command and its own encoder: `senses.GripperSense` reads the
driven jaw. Under 2 cm a hard squeeze's spring is the less -- at 120 N it
pushes 60 N a pad at 1 cm, 30 N at 5 mm -- and judged against half the
force asked, as before, jaws stalled on a 1 cm block at 120 N read "closed
on nothing it holds" (59.1 N, under 60), on a 6 mm block from 80 N. Shut on
blocks 6-15 mm across at 40-120 N, the squeeze reads within 1 N of what
`pushes` says and every grip reads held by the time its verb first reads it
(`body.GRIP_S`); `HELD_GAP_M` still reads 5 mm or less as closed
(`tests/test_humanoid.py`).

**Both sides of the comparison are running means** (`SQUEEZE_S`;
`Gripper.gap_mean`, kept across a restart). Judged at the gap as it read
that step, the bar fell with the jaws while the squeeze's mean lagged by
its 0.1 s, and a grip lost on something 26-35 mm across at 80-120 N read
held until the jaws read 5 mm: a block taken from between the jaws read
lost 6-8 ms later than judged against the force at 26 mm, 14 at 28, 22 at
30 and 40 at 35. At the gap's mean it reads lost when it did before at every
width from 18 mm up (4 ms later at 12 mm and 120 N, where the jaws reach 5
mm first). What that costs is a thin grip's first word: from open at 40-120
N, 342-502 ms after the shut where the gap as it read gave 342-388, still
before the verb first reads it at 600. A shut on nothing does not read held
as its jaws pass a thin gap: from open in free air, at every squeeze from 5
to 120 N, the squeeze's mean reaches at most a ninth of what the rule asks
while the jaws are under 2 cm apart, and shut from 1-3 cm apart, at the
squeezes where the rule asks less, at most a quarter (a test, every step
through `GRIP_S` and `GRIP_SETTLE_S`).

**Measured** on one pod (an Intel Xeon Platinum 8272L, osmesa), each flight
its own noise, the tree before and #646's flown side by side, and beside
every `held` reading a watch asking the other rule of the same reading:
- **Our table reference's grips**, each shut again from the state it began
  in, a real `Gripper` reading them (`humanoid_spike.py --grips 0-31`): the
  plate's 8 mm wall at 120 N reads held at 32 set-outs of 32, where the rule
  before read 28 -- at the other four its jaws stopped 9.7-10.0 mm apart,
  squeezing 56-57 N at the verb's first read, its pads on the wall. Below
  120 N, and the cup and the fork at every squeeze, the two rules read alike
  (the cup's edge grip at set-out 28 squeezes out from 50 N under either).
  Drawn 6 mm thick, the wall's jaws stop 7.2-10.1 mm apart and read held at
  all 32 at every squeeze, where the rule before read 28, 23, 17 and 0 of 32
  at 90, 100, 110 and 120 N.
- **The hands checks** at set-outs 0-31: 160 flights of 160 the same on both
  trees, verdict, reason and sim time, the watch seeing no disagreement; the
  lid 26, the drawer 21, the door 25, the cube 18 and the table 29 on both.
- **The served drawer, door and chest from the tour**, 32 draws each: 96
  flights of 96 the same, 27, 29 and 30 passed on both, no disagreement.
- **The station**: the approach docked, unplugged and parked 39 of 39 and
  the seats 140 of 140 on both trees, the same numbers, no
  disagreement in any of its 179 flights.
- **The table's practice arms' 70 tries** (#642) flown again from each
  set-out, noise and source: 3 flights changed, each where a grip on the
  plate's wall at 120 N, its jaws stalled 9.2-9.7 mm apart, read closed on
  nothing before and held now, the procedure going on to carry the plate --
  out of reach once, out of time twice, one of those ending in a fall. None
  passed on either tree (43 grips of 86 held, against 42 of 88). Six more
  read the rules apart only in the felt readings a practice shows its mind
  (the watch named who read each), "held" false before while the jaws held
  the plate; their flights the same.
- **The recordings the site replays** fly the same: the counter's byte for
  byte, the pair's but for one wall-clock stamp. **A restart** is the same
  flight (`determinism_spike.py --resume-at 300`: identical after the
  restore over 360 state samples).

**Why the reading, and not a harder drive.** A target past closed, by
`2 F / GRIP_KP`, puts the spring at its clip at every gap: on the bare hand
it squeezed 119 N on blocks 6-15 mm across at 120 N, and `held` would stay
half the force. But a hand shut on nothing then presses its own stop at its
whole squeeze for as long as it stays shut (reading 110 N at rest, where
today's reads none), every thin grip's physics moves (at 120 N a door's
lever could not turn in the jaws), and whether the Dex1-1 takes a target
past closed is unpublished. That is a change to what a grip does; this was
what it reads.

**What it cannot say.** The squeeze did not move: asked 120 N on the plate's
wall, the jaws squeeze what their spring pushes there, 58-74 N a pad across
the set-outs (`GRIP_KP`, #407's stiffness). And a shut on nothing at the
least squeeze still reads held for a moment, 0.07-0.10 s in as its jaws
speed up -- its drive pushes its whole 5 N there and the mean reaches half
of it -- as it did before #646, and long before its verb reads its word.

## What the colour camera can see (issue #660)

`spot` and `watch` read the D435's colour imager, pitched 40 deg down with the
depth camera (`HumanoidSpec.depth_pitch_deg`), its `fovy` 42 and its picture
1280 x 720: standing, 19 to 61 deg below level, and to either side of the
way it faces 34 deg at its top and 53 at its bottom, the picture being tilted
(flown on `waist16`: 18.7, 60.7, the stand holding the head 0.3 deg back). `look`,
`find`, a practice's first picture and the lab's cage view are the navigation
camera (`Body.head_camera`, `nav_eye`, 20 deg down, +1 to -40). Flown on
55858d3 in `home_humanoid`, osmesa, two noise draws, `spot`'s camera read:

| what | read | not |
|---|---|---|
| a 120 mm tag at 1.15 m on a bare wall (a room's; tags 73-75) | camera 10.6-12.6 cm off the wall, the toes 5 cm off | 8.6 cm (its top edge nearer than the cameras' 0.10 m near plane) and 15.6 cm on (its top edge above the picture) |
| the storeroom's wall row (60-63: the same size and height, over a 0.6 m deep counter) | never, anywhere the feet can go: its centre is above the picture past 0.31 m of camera, and the counter's front edge is 0.6 m out | |
| a 26 mm block's tag on the floor, the picture's middle | reliably out to 1.5-1.6 m of range, 0.9-1.1 m off along the floor | thinning past that: one read at 1.74 m, none beyond |
| a block on a prop: drawer knob 0.80 m, chest knob 0.81 m, door lever 0.87 m | 1.40, 1.35 and 1.15 m off the handle (range 1.38, 1.35, 1.12) | past those, where the tag's top edge leaves the top of the picture |

So the limit a 26 mm tag meets depends on where it stands: on the floor the
tag detector's own (about 1.5 m of range), at a counter's height the picture's top
first. A tag has to be whole inside the picture, and what stands d m ahead
of the camera is in it only under (1.26 - 0.34 d) m: standing, the camera is
4.5 cm ahead of the soles' middle, so at x in the stance frame, where the
robot reads every place, under (1.27 - 0.34 x) m. That is what `spot`'s text
says, its numbers off the model (`steps.SPOT_TOP_AT_FEET_M` and the rest) but
the small block's reach, flown (`SPOT_BLOCK_READS_M`); of a tag level with the
shoulders it says a small one's 0.3 m and that a larger one reads nearer
still: the 120 mm tags' window, 10-14 cm off the camera, is narrower than a
step lands. The instruments and their rows are
`~/pluggybot-training/issue-660/`.

## Two robots at the counter (issue #670)

**What happened.** At 11:48 on 11-08 Rowan's walk to the cupboard stopped at
the disc it keeps round Luca, who stood hung in its own withdraw (#668), and
its procedure then stepped onto its work spot anyway: `step` drove the walker
with no disc and no front stop. 2.8 s into its withdraw from the lever its
forearm hooked Luca's arm, both fell, and Rowan died (the investigation:
`~/pluggybot-training/counter-falls-1108/findings.md`).

**What a step keeps now** (`HumanoidBody._kept_off`): what a walk keeps.
- *The disc round the other robot's middle*, where `Body.others` puts it --
  its body (placed through this robot's own pose error, `as_seen`) lying,
  resting or dead, or standing or walking in this robot's LIDAR's sight
  (#675), else its reported pose -- 0.65 m, 1.2 m round one that has fallen. Judged where the stride's carry would take the
  robot were it stopped now (the step's own stop rule, `STEP_STOP_S` times its
  speed): the stride carries on 0.2-0.27 s of its pace, 6-8 cm. Standing
  inside the disc already, a step goes only away: one any of whose walks
  would head nearer, judged from where the walk before it leaves it, neither
  turns nor hands its legs to the walker -- refused at the walk instead, each
  such step took 1.3 s, a hand-over and a settle, and moved the feet.
- *The walks' front stop turned to the step's way*: a return off the other
  robot within 0.40 m along it and 0.27 m either side, judged at the carry
  as the disc is (stopped at 0.40 m, the carry's 6-8 cm would take a
  sidestep to 0.32-0.34 m of the body, its hands hanging 0.27 m out), from
  the scans the step has taken, its hand-over's included (none a restart
  could drop). Standing still the LIDAR takes no scan, so a body in the way
  stops the walk after its first command, its legs handed over and no
  stride taken. ⚠ ITS RETURNS ALONE (`Lidar.exclude_robot`'s split, #316, as
  the walks' depth camera holds for its peer channel's alone, #328): stood
  still at each prop's work spot
  (`storeroom.work_spot`, one scan each, square and turned the reference's
  `STANCE`), the props' own faces read 0.35-0.38 m ahead of the chest's and
  the drawer's, inside the stop, and 0.42-0.46 m ahead of the door's and the
  cube's. The counter itself, its edge 0.22-0.33 m ahead, is under the band's
  steepest ring there (-52 deg from 1.41 m meets the counter's top 0.54 m
  out). On every return, a step in to two props of four would stop short of
  where the work is done.

Either reads false, `blocked`, saying how (`steps.STEP_KEPT_OFF`). A robot
alone reads neither -- no other robot, no returns split off -- and steps as it
did, bit for bit: the cube check's set-outs 0 and 1, its steps among them,
flew row for row on both trees.

**As hardware-honest as the walks.** A robot standing or walking is kept off
at its body where this robot's LIDAR sees it, and at its reported pose, a
network fact, where it does not (#675: at the counter the two were often half
a metre apart, "A robot at work kept off where its body stands"); the front
stop reads the LIDAR, which sees the other robot. Its two stand-ins are the
walks': a robot placed at its body is placed where its body truly lies, seen
through the observer's own pose error (#365, #455), for spotting a
robot-shaped lump; and the LIDAR tells the other robot's returns by their
geoms (#316), for segmenting them.

**Flown: 11:48 from where the walk stopped** (`~/pluggybot-training/
issue-670-step-clear/fly1148.py`, osmesa; the investigation's `door.py --case
1148` began after the steps). Rowan starts where the log's steps say its walk
stopped -- its spot's stance less three steps of 0.10 ahead and two of 0.03
left -- and runs its own door_go4 from its `face(0)`, its spots, faces, steps
and reaches its own. Luca stands behind its right shoulder on the walker,
held still: where its 11:45 drift sample put it, and 6 and 12 cm east of
that, 0.54-0.57 m from Rowan's start (soles to soles); three noise draws
each, nine flights a tree:

| | touched | Rowan fell | Luca fell | Rowan's steps |
|---|---|---|---|---|
| staging (232a55d) | 9 of 9 | 4 | 4 | 3-4 taken, onto its work spot |
| this build | 0 | 0 | 0 | 6 toward Luca refused in each, none taken |

Staging's 11:48 is the live one: two steps in and one left, the lever's grip
out of reach, the press closing on nothing, and the arm's way home hooking
Luca's (its forearm on Luca's upper arm); at Luca's middle place, both fell,
3 of 3, as live -- the investigation's flights never felled Luca. On this
build the procedure, its steps refused, ended at its sidestep (`too short`),
which it called as a statement; it reached for nothing.

**Its reaches from there, with no step** (the tail alone, door.py's, from the
same start): the reach for the lever 0.69 m ahead, taken, swung out to the
side and its jaws met Luca's right thigh 0.7 s in, then the bow brought its
head into Luca's torso: touched in 6 of 9, Rowan fell in 6 (Luca once), and
in the other 3 the reach was out of reach. So keeping a step off another
robot keeps a robot from walking in, and nothing else: a procedure that reads
its refused step and reaches anyway reaches into it.

**At the disc's edge** (Luca 16-18 cm further south, 0.70 and 0.73 m from
Rowan's start, where a walk would have left them; three draws each): the steps
are taken, as they should be, and from the work spot the press's way home met
Luca's arm with its jaw or gripper in 6 of 6 -- at 0.70 m a brush, no fall; at
0.73 m it hooked Luca's upper arm and both fell, 3 of 3, 11:48 whole. Its
reaches alone from there touched 2 of 6, no fall. The arms met at 0.68 m on
11-08, just outside the disc: whether keeping a step off would have saved
11:48 comes down to where Luca stood within a few centimetres, and the way
home is what felled them either way.

**What it leaves open** (#670's own list): an arm on its way out or home
reaching into the other robot -- since #678 kept off it ("The arms keep off
the other robot"); a fall at one prop reaching the next, 1.2 m apart (a fall carries the
pelvis 0.8-0.9 m sideways); and the get-up against the counter (#669).

## A robot at work kept off where its body stands (issue #675)

**What happened.** Every walk and every step keeps off the other robot where
`HubLifecycle.keep_clear` puts it, and a robot standing or walking -- one
working with a hand out among them, which is not resting -- was kept off at
its REPORTED pose: its belief, in its own map. On 11-07 and 11-08, with the
pair within 2 m of each other, one robot's reported pose lay 0.49 m from its
body in the other's map at the median (0.75 m at the 75th percentile, 1.13 m
at the 90th; 62 paired drift samples), and on 11-08 both beliefs were
0.49-0.71 m off at every sample from 11:22 to 11:45. The disc keeps 10 cm
beyond the two bodies (0.27 + 0.27 + 0.10 = 0.65 m): half a metre off, it
kept a step off where the robot was not -- refused for nothing, its words
giving a distance that was not true -- and not where it was, where only the
front stops stood between them.

**What is true now** (`keep_clear`, `Body.seen_at`, `Body.moved_at`):
- A robot standing or walking that this robot's LIDAR sees is kept off at
  its body, placed through this robot's own pose error (`as_seen`), as one
  resting already was; out of sight, at its reported pose. Fallen, resting,
  dead, getting up or stepping aside: unchanged.
- *Seen* is the band's returns off that robot, named by its root
  (`Scan.peer_roots`: `Lidar.exclude_robot`'s split, #316, kept robot by
  robot), each scan stamping whom it named (`Navigator.seen_t`).
- A sighting holds `SEEN_HOLD_S` (1 s) of this robot's own scans, so a
  frame or a few that miss it -- its own fin, an arm, the part's dropouts --
  move no disc half a metre. MEASURED over the flights below: in the
  storeroom's open floor, 10,364 scans of 10,364 with the other robot
  within 2 m named it (36 flights); through the house (make_way's scenes),
  every scan that missed it within 4 m had a wall or a door's frame between
  the LIDAR and its chest (a ray, the truth's), but one, a single scan 3.1 m
  off. Longer, a robot gone behind a wall is still placed at its body for a
  second of its walk, 0.4 m at most.
- Its LIDAR idle, its newest scans' word stands: standing still with a hand
  at work the humanoid takes no scan (only its walks, turns, holds and the
  legs' hand-overs do), so a step after a reach is judged on the scans
  before it -- but only while the other robot has gone nowhere since it was
  seen (`Body.moved_at`: a walk command, a get-up or a warp, never a hand at
  work). Past the hold, an old sighting of one that walked off meanwhile
  would place it where it went, unseen.
- A plan hears of a robot coming into sight or leaving it within
  `DOWN_CHECK_S`, as of a fall (`KeepClear.seen`): its disc moves by the two
  beliefs' disagreement.
- A warp (a start, a stand-up) forgets whom the LIDAR saw; a restart keeps
  it, and when the other last went.

**As hardware-honest as the walks.** The stand-in is the walks' own (#365,
#455): a robot-shaped lump where its body is, placed through the observer's
own pose error -- now gated on the sensor that would see it, and a robot
standing is the easiest lump to see, upright in the band #670's front stop
already reads. The split is #316's stand-in for segmenting the other robot's
returns; retiring both with detection off the LIDAR's points, no geom labels,
is a later issue. `moved_at` is the other robot's own word, as its reported
pose is.

**Flown** (`~/pluggybot-training/issue-675-seen-at-body/`: `fly675.py`,
`batch.sh`, `read.py`; osmesa, three noise draws a scene, staging b4074eb
and this build). Each robot's map and belief are laid off the world
together (`make_way_spike.drift`), its matcher told the map it matched
against is gone (`ScanMatcher.forget`: against the map as it last computed
it, a search put a belief laid 0.55 m off straight back). Luca at work: its
right hand out ahead of it at 0.95 m and held.

*Steps* -- Rowan from where its walk to the cupboard stopped at 11:48 (#670's
`fly1148.py`), running its own door_go4 from its `face(0)`; "2" is the same
scene with both beliefs off, Rowan's 0.5 m and Luca's laid to the same
disagreement:

| scene | Luca from Rowan, body / word | staging | this build |
|---|---|---|---|
| A, A2: its body in the disc, its word out of it | 0.56 / 0.99 m | 2-4 steps taken in each flight; touched in 5 of 6, both fell in 3 | every step toward it refused, saying 0.56 m; no touch |
| B, B2: its body out of it, its word in it | 1.15 / 0.56 m | every step refused, saying 0.56-0.57 m | 4 steps taken in each; no touch |

*Walks* -- Rowan from the storeroom's west side (23.0, -3.0) to a goal at
the counter or across the room, the depth camera on, as served
(`batch2.sh`); no walk touched, and none fell:

| scene | staging | this build |
|---|---|---|
| C: its body 0.5 m from the goal, its word 1.0 m | gave up 3 of 3, held by the depth camera 0.70-0.90 m off its body, each saying it was 1.2-1.4 m off (its word) | gave up 3 of 3, 0.71-0.81 m off, each saying "the other robot standing 0.5 m from where it was going" |
| D: its body 1.1 m from the goal, its word 0.5 m | arrived 2 of 3, and gave up once for nothing, saying it stood 0.5 m from the goal | arrived 3 of 3 |
| E: its body on the way, its word 0.6 m off it | planned through its body, held by the camera 0.84 m off it, and gave up after 90 s 2.5 m short, 3 of 3 | planned round it and arrived in 10 s, 3 of 3, passing 0.79 m off |
| F: its body 0.9 m off the way, its word on it | walked round its word, arrived in 9.7-10.0 s, passing its body 1.5 m off | arrived in 8.2-8.3 s, passing 0.80 m off |

Without the camera (the counter set's own C and D, each also with both
beliefs off), staging's walk went on to stand 0.55-0.58 m from Luca's body
in 3 of 6, its disc on a word 1 m off, and this build's once, 0.55 m: a
walk whose goal lies just inside a disc ends its plan at a stand-in short
of it, and once that is reached steers at the goal itself a step at a time,
on both trees.

*The pair's other flights* -- make_way's scenes (`scripts/make_way_spike.py`,
the depth camera on, as served; this build's through `sightings.py`, which
also logs whom each LIDAR saw), sixteen a tree: #415's seven doorways, #455's
three with the robot in the doorway standing 8 s and then walking off, those
three with its map and word 2.9 m off the world, and three with the walker's
1.0 m off. Fifteen ended as on staging -- arrived, asked aside or not as
before, a step aside of 0.82-0.88 m, no bump, no fall -- seven identical in
every field, the rest seconds apart. The sixteenth, the kitchen's doorway
with the standing robot's map 2.9 m off, gave up "peer" 8.2 m short on both
trees: staging's words said the robot in its way was 2.5 m off, its word;
this build's, 0.7 m, its body. There the standing robot's own walk off kept
off the walker where its LIDAR saw it, and ended 15.7 m from where it had
stood (in the east garden), where staging's, keeping off a word 2.9 m off in
its own map, ended 3.7 m away. The pair's fixture day sharing the house's
station (`two_robots.py` as the fixture's recording is made) ended alike on
both trees: one charge each, the same meeting, no death. A robot alone is
unchanged: the cube check's set-outs 0 and 1 flew row for row.

**What it does not fix.** The arms (#670's open half): the disc is about
where bodies stand, and an arm's way out or home reaches 0.74 m to the side
-- #678's ("The arms keep off the other robot"). And the drift itself: both robots 0.5-0.7 m off in the storeroom for half an
hour hurts the jobs too, and is worth a reading of its own (#572's
instruments) of why the storeroom loses them.

## Letting go before the way home (issue #669)

**What happened.** At 14:25 on 11-08 Rowan's third grip shut its jaws on the
cupboard door's lever -- a 1 cm bar, at 120 N -- and read "closed on nothing
it holds" (#646's thin grip). Its procedure opened the hand only `if
read("right.held") > 0`, so it withdrew with the bar clamped: the hand came
out 1.3 cm, the side way home moved the trunk instead (0.27 m toward the
counter, 31 deg of tilt), and the robot went over into the counter and never
stood -- 9 of 9 noise draws on the live build (the investigation:
`~/pluggybot-training/counter-falls-1108/findings.md`). #647 fixed the
misread; a pull that lets go can still leave the jaws shut, and our own
references withdraw with their jaws shut after a grip that read false.

**What a withdraw does now** (`HumanoidBody.release_routine`): its gripper is
opened first, whatever it says it holds, the jaws given `GRIP_S` (0.6 s, as
`open_hand` gives them) while the robot stands still, before the hand starts
out; a fall in that time ends it the step it comes, the hand never sent. A
gripper already open (`Gripper.opened`: last told to open its whole stroke)
is left as it is, so a withdraw after an `open_hand` is the flight it was.
**So does a walk's hand-over** (`to_walk_routine`), for an arm that is out:
`step` refuses only a grip that reads held, and a procedure that ends with a
hand out walks after it, so the same jaws on the same lever came home
through every walk (#669's review). One routine lets go for both
(`HumanoidMission.let_go_routine`). A `step` asked while a gripper says it
holds something still refuses before that hand-over (`holding`), opening
nothing, and the body text says both: "Before a `step` or any other walk
moves your feet, a hand that is out comes home, its gripper opened as
`withdraw` opens it: a walk carries nothing. A `step` asked while either
gripper says it holds something moves nothing, and reads false." The
refusal went unsaid at first (#669's second review): asked through the
deployed prompt, GLM-5.3-Flash on baseten read a `step` with its jaws on a
knob that read held as letting go and walking 12 times of 12 on that text,
and as refused, the gripper still shut, 12 of 12 on this one; a `step`
after jaws that read nothing held, and a `drive_to` with a cube in hand,
read as letting go 12 of 12 on both (`~/pluggybot-training/day-shift-1111/
d1-release/quiz/`, $0.07). With nothing to open it
steps nothing, and no reference begins a walk with a hand out -- nor does any
of the 459 practice tries below, by their recorded verbs. Its text
says so with no condition the robot must work out. Worded "open a hand's
gripper if you have shut it", the deployed mind (GLM-5.3-Flash on baseten, 24
answers a text, each free to say "unknown") read a held cube as let go 24
times of 24, but jaws a `close_hand` that read false had shut as opened only
17 times: the other 7 were not sure that gripper counted as shut. So it
opens one that boots with its jaws together too, and "open a hand's gripper,
letting go of whatever it holds, then ..." read as meant 24 times of 24 on
both questions -- the old text 0 of 24 on both, and the verb's way home, two
questions more, 24 of 24 on every text.

**Flown** (one Vast pod, an AMD EPYC 7B12, osmesa; this build and the tree it
branched from, b4074eb; `~/pluggybot-training/issue-669/`):
- *14:25's door tail* (draw 177, the investigation's `door.py`), its last
  `open_hand` taken out so the withdraw begins with the jaws on the lever's
  bar: staging fell 9 of 9 noise draws, every one on that withdraw; this
  build 0 of 9, each withdraw home. As Rowan wrote it -- #647's grip reads
  held, and the procedure opens before it withdraws -- the same flight on
  both trees at those 9 draws and 12 stance jitters, none falling.
- *Our references from the tour* (the served drawer, door, chest and table,
  32 draws each, each its own noise): the same flight on both trees, row for
  row -- the drawer 31, the door 26, the chest 31 and the table 27 of 32: none
  withdraws with its jaws shut.
- *The hands check* (the lid, drawer, door and cube at set-outs 0-127, the
  table at 0-31, each its own noise): a flight whose every withdraw began
  with its gripper open is the same flight on both trees, and 60 of 544 were
  not -- each a withdraw after a grip that read false (the references'
  `else: withdraw`), whose jaws now open at the prop, the robot standing 0.6
  s longer, the flight going its own way from there. The lid passed 102 then
  103, the drawer 87 then 88, the door 95 and 95, the cube 73 then 70 (4 lost,
  1 gained, of its 26 that parted), the table 24 and 24. Falls: the drawer 3
  then 4, the door 2 and 2, the table 3 and 3, none at the lid or the cube --
  the new one the drawer's set-out 2 on its LAST withdraw, which began with
  its jaws open, as every withdraw before this did: a flight gone its own way.
  At the cube's four lost set-outs the jaws read shut on nothing as the
  withdraw began (a gap of 0.1 mm at most); as they opened, the cube moved
  0.4 mm at most at three, and 6 mm at the fourth (set-out 29, where it ended
  4 cm from the mat's middle, not 1.3) -- opening beside a free thing after a
  missed grip can nudge it. Staging's passes at the four were 1.3-2.3 cm from
  the middle, inside 2.5. A gripper shut on nothing holds nothing to let go
  and could be left shut; this build does not, as its text would then need a
  condition again.
- *Procedures the mind wrote* (#554's lid and door arms and #544's door B: 459
  recorded practice tries, each flown again from its set-out on each tree):
  448 the same flight, 11 parted, passes 60 and 60, falls 76 then 75, on the
  way home 8 then 7 -- the one no longer falling a door try whose withdraw
  brought home a hand it never opened.
- *The rule as shipped* (a gripper opened unless it is open, c35477c) flew
  the check's 160 flights at set-outs 0-31 as the first version did, which
  opened only a gripper told to shut, every row the same: no reference
  withdraws a hand it never opened.

**What it leaves open** (#669's follow-ups, each its own decision): what holds
an OPEN hand -- the counter's top under a jaw, a lid come down over the hand
(Luca's 15:17) -- still holds its arm, and the way home still knows nothing of
the world (Hands.md §7); a reach of one hand while the other is held takes a
posture planned for a free body; and the get-up against the counter (13 of 14
flown falls there never stood in 60 s).

## Carrying while walking, measured (#560's next task)

The next task carries the table's dishware from the restaurant kitchen's
shelf, and today a walk carries nothing: `step` refuses a held grip and the
walk's hand-over lets go (#669). `scripts/carry_spike.py` hands walk2 the legs
with an arm posed holding the plate or the cup anyway, on a bare floor -- the
piece put into the open jaws where our table reference's shut leaves it, the
wrist solved to hold it level -- and walks open-loop courses and goals through
the Navigator (2,192 flights, each its own noise and draw; the rows and the
plan are the night shift's, 2026-11-11, T7).

- **walk2 never trained with its arms at rest**: its scripted arms went
  anywhere within half their range, with 0-1 kg in a hand, and it sees every
  arm joint. But a wrist holding a piece level turns 0.6-1.5 rad (trained to
  0.7-0.8), and an elbow folded close 1.6-2.0 (to 1.15).
- **At a designed carry pose it never fell** (392 open-loop walks, 128 closed
  loop, 64 with 1-2 kg on one arm). It veers toward the carrying arm, 4-22 deg
  more than arms-home over 4 m, and walks 8-15 % less far; closed loop the
  Navigator steers it out (its reckoning holds), arriving within 0.24 m of
  where arms-home does.
- **The arm the reach leaves for the table's grips is over the top** (roll
  2.5-2.9 rad); walked so it fell in 16 of 37 -- and in none shown that arm at
  rest (`--blind-arms`): an arm input far outside its training topples walk2.
  Shown its arms at rest, a piece held close walks true, but a load held out
  goes unbalanced (both arms out fell 20 of 28): it balances off those inputs.
- **The robot's own sensors see what it holds**: its front stop backs it off
  its own plate, or its map leaves no route -- 7 of 10 holding poses never
  arrived; hidden from every cast (`--unseen`), all did.
- **The plate's wall pinch creeps with its weight**: at 40 N a 0.5 kg plate
  dropped in 8 of 20 routes of 13 m, at 80 N in none (the turn 7-8 deg, the
  same at 120 N) -- friction at the pinch, not the solver's drift, which
  ignores the force ("The grip that leaked"). The cup, held across its body,
  slips under 1.1 cm on a bare floor.
- **In the house** (`--house`: the shelf through the staff door to table 0's
  standoff, closed loop), every carry arrives with the held piece hidden from
  the casts (28 of 28, within 0.13 m), and 6 of 24 as served. But the cup
  held out slid down its jaws through the longer walk and fell at a pivot in
  2 of 8, touching nothing, sooner at 120 N than at 40: a creep that is not
  the squeeze's, not yet diagnosed.

**What is true now:** nothing served carries. Carrying needs a hand-over that
keeps a held grip and walks the arm to a carry pose first, a self-filter for
what a hand holds, the plate squeezed at 80 N, and the cup's creep found.
walk2 as it is carries
closed loop at a cost in time and a veer; a walker trained with carry poses
held and loads at the grip is the honest fix. `carry_spike.py --view
plate_close:route:1` watches a 0.47 kg plate held close tilt in the jaws from
35 s and fall at 47 s; `--strip` draws it.

## Carrying, served (#682's stage A)

Ben's calls of 11-12 on #682: one piece in one hand, held close (`close`);
the walk takes the carry pose itself and says so, no `carry` verb; the plate
squeezed at 80 N; a self-filter of about 0.12 m round a held grip; and a
walker retrained to carry (below, "A walker trained to carry"). This is the
walk that carries, on walk2 (the loop of 11-12/13, L4; rows and probes in
`~/pluggybot-training/loop-1112/l4-carry/`).

- **The hand-over carries a hand that holds** (`HumanoidMission.carries`,
  `_lift_routine`, `to_walk_routine`; `Reach.CARRY_POSE`, `carry_pose`): a
  hand out at a pose of its own whose gripper reads held is lifted 2 cm
  straight up at 20 N, the most the standing body holds; only if the lift
  went its distance is it carried -- out along its line, then its judged way
  (the way home's, ending there) to the forearm folded up and the hand at the
  hips, its gripper as it was -- and held there in joint space through every
  walk, a `step` too, until a reach or a let-go ends it. Anything else that is
  out lets go and comes home, as #669's does.
- **Lifted first, because a carry pulls on what it holds.** Carried with no
  lift, a drawer's handle held through a step came out 19 cm with the arm.
  A guard that let go where the arm's own force reading passed 30 N on its
  way (a free plate's move read 16.5 N at most) felled the robot at the first
  drawer flown: the arm, let go mid-move against the drawer's pull, threw the
  trunk. With the lift, a drawer's handle (2 draws) and a door's lever read
  held fast and were let go, no fall (`probe/held_handle.py`).
- **The wrist keeps the hand turned against the floor as it was, solved on
  the stand's statue.** Solved on the model as it stood -- bowed 0.3 rad for
  the table reference's plate grip -- the plate rode 15-17 deg tipped once the
  trunk stood up; on the stand's statue, 2.3 and 6.9 deg (the second began at
  3.0). The turn is the one the hand's command keeps, never its joints read
  again: a read draws on the drivers' noise, and the check a walk verb makes
  before it walks would move the flight.
- **The self-filter** (`HELD_DROP_M`, `_held_drop`, `Lidar.band`'s `drop`):
  each grip that reads held, placed off the odometry's model of itself and
  levelled along the pelvis's heading as a scan's points are; a room's return
  within 0.12 m is neither a point nor a ray that went on (a bearing the
  piece alone stopped reads nothing, never free), in the band and in the depth
  camera's low layer. Another robot's returns are kept: split out by the part,
  dropped they would blind the front stop to it.
- **Flown on walk2** (one pod, a Ryzen 9 5950X, osmesa, c3dcca4; each flight
  its own noise and draw; rows `batches/b4/`):
  - the reference's own grip and lift, then the served hand-over (raised, out
    along its line, to `close`): the plate at 80 N arrived 15 of 16 walks (4
    m straight, the 13 m route, the goals 4 m ahead and 2.5 m behind; one
    found no route), no fall, no drop, turned at most 13 deg in the jaws; the
    cup at 40 N 11 of 11, no fall, but it fell out of its jaws on the 13 m
    route twice of three (turned 180 deg): the cup's creep, T7's A7, still not
    diagnosed. 5 set-ups whose grip closed on nothing were not flown;
  - in the house (the kitchen's shelf, through the staff door, to table 0's
    standoff; closed loop), the arm posed as T7 posed it and held out there,
    the served hand-over taking it to `close`: arms home 4/4 in 39-46 s;
    plate close 4/4, plate low 3/4, plate out 3/4, cup out 3/4, cup tray 4/4
    (cup at 40 N: 2/4 and 3/4) -- 17 of 20 carries arrived, where T7's as
    served (no carry pose, no filter) arrived 6 of 24 -- in 43-108 s; no
    fall. ⚠ But the plate fell out of its jaws once in each of its three
    poses (3 of 12 walks at 80 N; at the instrument's own pose before the
    review's fixes, 5 of 8), turning in the pinch until it slid free: it
    touched nothing of the world first (every geom a held piece met was
    recorded; the floor alone, after it fell). Not the door frame's, then:
    the plate's own pinch over a walk of 90-100 s with many stops and turns.
    The cup never fell in the house (16 walks);
  - two pieces at once (`both_low`, outside Ben's one piece): 2 of 8 goals
    arrived and one fell standing (before the lift);
  - walk2 veers toward the carrying arm at `close`, as T7 measured: +24-27
    deg over 4 m with the plate (left), -10 to -15 with the cup, arms home
    +3.5-5.
  - the hands checks at set-outs 0-31 (lid, drawer, door, cube, table: 26,
    21, 24, 20 and 24 of 32), beside loop-staging on the same pod: 160 of 160
    verdicts the same, reason for reason -- no reference walks holding
    anything, and the self-filter moved none of them.
- **The table's reference with the plate at 80 N** (`hands_check.TABLE`, the
  served lay): 77 of 96 set-outs (0-95) against 82 at 40 N on the same pod,
  paired -- 5 lost, none won, every one the fork knocked from its jaws or off
  its place after the plate was laid; at 0-31, 22 against 24, its bar. The
  lay walks nothing, so it squeezes at 40 N still; 80 N is a carried plate's,
  stage C's reference's.

**What is true now:** a walk carries a hand that holds and has come away, at
one pose, its wrist level as it was, its piece hidden from its own sensors
within 0.12 m of the grip; everything else lets go as before. On walk2 it
veers toward the carrying arm, which the Navigator steers out and a `step`
does not. Two things stand before a carry is graded (stage C): the plate's
pinch let it slide free in 3 of 12 house walks even at 80 N, touching
nothing, and the cup's creep on the 13 m route.

## A walker trained to carry (#682's stage B)

`Pluggy-Humanoid-Walk-Carry` (`training/humanoid_train/walk.py`
`carry_env_cfg`, `CarryArms`; `robot.get_carry_spec`): walk2's flat task
fine-tuned from its checkpoint (`model_2999.pt`), each arm in segments of
4-15 s hanging (30 %), moving as walk2's arms did (20 %) or holding one pose
drawn from the carry box the whole segment (50 %; both arms carry a quarter
of the time), a 0-1 kg payload at each grip site drawn at every reset, the
actor seeing walk2's 24 joints in walk2's order (the exporter records them,
`export.seen_joints`).

- **⚠ Trained on the waist's joint, it learnt to spin** (carry1, 1250
  iterations on a 5090, $0.25, stopped): asked to stand still it turned 0.8
  rad/s by its heading, each world at +-1.9 rad/s by the root's velocity
  sampled after a step -- while the reward's own reading of the same root
  said it tracked its turn (the turn's term rose 0.05 -> 1.0, its yaw error
  1.2 -> 4.5). The pelvis's yaw rang against the trunk on the waist's spring
  at the policy's rate, and the policy phased the ringing to the instant the
  reward samples it. walk2 itself, flown in that task, scored 0.3 on the
  turn's reward with the waist's joint and 34.9 without it (its own measure
  of its turn 0.53 and 0.86 a step). The carry task's model is RIGID at the
  waist, as walk2's body was (`get_carry_spec`); served, the waist is held
  home while the robot walks. ⚠ Any walker retrained on today's model with
  the waist's joint free meets this.
- **carry2** (rigid waist): 2500 iterations from walk2 on a 5090 (1:08, an
  EPYC 7K62 host at 1.6 s an iteration), its turn's yaw error 0.43-0.45
  through training (walk2's own 0.46); `models/humanoid_walk_carry.npz`,
  beside walk2, served by nothing. Its checks, beside walk2 on one pod (a
  Ryzen 9 5950X, osmesa; each flight its own noise and draw; rows
  `batches/b3`, `b4`, `b5`):
  - **Ben's flat-floor rule passes** (`--track`, arms home): forward 0.51,
    0.99, 1.31; back -0.21, -0.47; sideways 0.28; turns 0.47, -0.42, 1.04 --
    every asked part's error inside the rule's bar (0.11, 0.11, 0.12; 0.21,
    0.14; 0.11; 0.39, 0.30, 0.11), what was not asked within 0.07. It turns
    asked 0.5 rad/s at 0.47 where walk2 turned 0.24.
  - **The house's route, carried by the served hand-over: 32 of 32 arrived,
    no fall, no drop** -- plate close, low and out in 42-56 s, the cup out and
    tray in 53-78 s (at 40 N and at 80), arms home in 37-43 -- where walk2,
    the same flights, arrived 26 of 32 in 43-108 s and let the plate fall
    free 3 times.
  - **The carry matrix** (T7's poses, the instrument's own hand-over): no
    fall at a designed pose (0 of 120); the plate's carries walk true -- over
    4 m, 0.1-0.3 deg off its own arms-home walk and 3-4 % shorter (walk2:
    +8-21 deg, 6-8 %) -- and the route's open-loop end 0.8-2.0 m from its
    arms-home end's (walk2: 5.4-12.7); but a carry in the RIGHT hand veers
    still, -14 to -20 deg over 4 m (cup close, cup out; walk2 -8 to -15), and
    its arms-home walk itself veers +9.6 deg over 4 m (walk2 +4.6). Closed
    loop, 39 of 40 goals arrived (walk2 35); the plate turned no more in its
    jaws (8-9 deg).
  - The reach's own arm, never walked by the served hand-over: the plate's
    walked with no fall (0 of 15), the cup's over the top fell every time (7
    of 7), as walk2's did -- outside the carry box, so untrained.
  - Rough ground as walk2 (`--rough`: the same falls on doorsteps and blocks,
    none on flat, paving and the slope).
  - Not flown: `--odometry`, `--torque`, `--handover`, the step's and the
    station's instruments, the drift and make-way spikes, the tour's
    references, the pair's determinism and the real-time multiple.
- **carry3**, 1500 iterations more from carry2 with half its carries drawn
  within 0.15 rad of `close` (`close_share` 0.5; a Xeon w7-3455 host, 25
  min): the right hand's veer went (cup close +3.4 deg, cup out -5.7) and the
  left's came back (plate close +6.6, plate low +7.2), its arms-home walk
  -7.4 deg over 4 m; 44 of 48 goals (carry2 47); the house 22 of 24, one
  plate let fall; ⚠ the plate's reach arm, which carry2 walked 15 of 15,
  toppled it 9 times; and on the flat-floor rule's back-off at 0.3 m/s one
  trial of three stood still (-0.14 m/s its mean, the bar's edge). Narrowed
  toward one pose, it lost what the box had given it: carry2 is the
  candidate, carry3 kept off the repository (`loop-1112/l4-carry/`).

## The place laid from the pass (#682's stage C, easy mode)

**What is true now.** Table 0's place can be laid from its pieces set out on
the restaurant's pass -- the expo counter in the dining room's east wall,
0.90 m up, between its pair of tags (68, 69) -- each piece carried 4-5 m to
the place by the walk's own hand-over (`set_table_pass`, NOT OFFERED and paid
nothing; `counter.pass_set_out`; Challenges.md §12). Its reference
(`challenge/solutions.py`, `PASS` over `LIBRARY`) was flown against the bar
written before it, 24 of 32 draws over the tour on walk2 and on the carry
walker (`scripts/solve.py --feature pass --from tour`; the reading is
below). What developing it took, each measured on the loop's pods (AMD EPYC
7B13, osmesa, each draw its own noise; rows in
`~/pluggybot-training/loop-1113/l4-shelf/`):

- **A carried piece hangs under the pass's top.** The carry pose holds the
  cup's grip at 0.89 m and the plate's at 0.865, 0.23-0.26 m ahead of the
  soles' middle (`CARRY_POSE`, its wrist solved to keep the grip's turn), so
  the cup's bottom rides at 0.83 m and the plate's at 0.81, their far edges
  0.32-0.35 m ahead: the walk's hand-over clears a 0.90 m counter only from
  about 0.37 m off its face. Standing (no bow) a hand takes a piece at the
  pass's height no further than about 0.45 m ahead, 0.25 m to its side
  (`reach.plan`; further in, or nearer the middle, it bows 0.55-0.8 rad over
  the counter). Set out 0.18-0.20 m in, the cup was gripped and the hand-over
  swept it into the counter's edge; the pieces lie 0.06-0.11 m in, and each
  is pulled back over the edge before the walk.
- **A grip's descent holds its line.** Re-targeted off the hand's sensed
  place (`go`), the cup's descent drifted 1.4-3.6 cm back and 11 of 16 grips
  closed on nothing; a pull straight down that gives across its way (`give`
  1) slid on the force reading's phantom 5-8 N -- 3.6 cm back in one flight,
  19 cm forward with the cup pushed in another. Straight down holding its
  line, the cup's grips landed 1.5-3.4 cm short of its middle. Gripped 4.9-5.5
  cm short it was lost on the pull back or the walk, 7 short closed on
  nothing.
- **The stance turns 5-10 deg as an arm goes out**, and a reach planned
  before it put the plate's grip 4-9 cm off its 8 mm wall: the plate and the
  fork are reached to a hover 6 cm over their grip, read again there,
  corrected at that height and pulled straight down (the plate laid 7 of 10
  where 4-5).
- **The plate's rim pinch (#682's open item 1).** The table check's plate
  grip -- fingers 60 deg down, the grip 5.3 cm over the surface -- puts the
  80 mm pads' lower tips only 0.7-0.9 cm over the 2.8 cm wall's top: an edge
  pinch, enough for the table's 20-30 cm move. Carried from the pass it fell
  in 37 of 42 legs (80 and 120 N, pulled back or not, on walk2; 6 of 9 on the
  carry walker), and posed at that pinch on the 13 m house route (`carry_spike.py
  --house --served`, 80 N) in 8 of 8. Gripped 1 cm lower, the pads over
  1.6-1.9 cm of the wall: 1 of 9 legs. ⚠ The hand's sensed place is 1.5-2 cm
  off across the jaws, and a carry moves the plate a further 2 cm forward in
  them: its offset in the hand is the flights' (2 cm ahead, 8.3 cm across),
  as the table check's is a constant.
- **The cup's creep (#682's open item 2) is where its grip is aimed.** Taken
  from above, the fingers 45 deg down, a cup touches each pad along the line
  of its side, which crosses the pad's face where the grip is aimed along
  the fingers' way. Aimed 2.5 cm short of its middle (the table check aims
  3: at the table's height the palm cuts the lid at the middle), the grips
  landed 2.1-3.5 cm short (a median 2.8-3.0), and the line met each pad only
  at its fingertip's corner -- a contact trace (dev draw 152) read one
  contact a pad, 35-40 mm from the pad's middle, its tip at 40, the cup
  pivoting about the jaws' line and creeping toward the tips about 0.2 mm/s
  -- and carried, it let go in 20 of 32 of the bar's flights on each walker
  at 25 N (at 40 N, 12 of 16 from the tour), the hand-over's way to the
  carry pose turning the hand about 29 deg midway. Aimed at its middle
  (landed 0.8 cm short to 0.4 past), the line crosses each pad's width
  1.5-2.5 cm off the tips (the trace's two contacts a pad, 4 cm apart), and
  of 64 cups carried from the tour (dev draws 200-231, 32 on each walker)
  none let go; 54 of the 64 grips held (a pad landed on the rim, or the cup
  was pushed aside, in the other 10). The cup posed in the grip on the 13 m
  house route held at 40 N (0 of 16): `carry_spike.py`'s pinch has its side
  up each pad to 2.7 cm off the tips, where the pass's grips reached 1.3 cm
  at the shut, and less as the cup crept.
- **A far find reads one tag's own facing 16 deg off**, and from the tour
  the robot faced the pass that far off, the cup out of the colour camera's
  picture; the wall's facing is taken off the line through its two tags. A
  face lands up to 16 deg off its 4 deg tolerance anyway: a piece the
  picture misses is looked for by a turn toward its side, then a step in
  (`near`), and the fork, whose tag is 19 mm, is taken from 0.20 m to the
  side, not 0.25 (at 32 deg it sat on the picture's edge).
- **The table's approach, carrying,** ended up to 0.2 m further back than
  it expects and once asked its last step 0.53 m (refused over 0.5): taken in
  two. A step aside the lay took after it, carrying the cup, fell once, and
  is gone; the approach's own steps across (0.3-0.4 m, one with its big turn)
  stay, and a cup has let go during them. The
  cup's put-down sits at the edge of the right hand's band both ways: from
  the table check's stance it took a crouch and a waist turn and fell, from 5
  cm further back a 0.55 rad bow that missed by 13 cm; the walk ends 1 cm
  further back (`PASS_BACK`). The walk back from the table to a standoff 1.2
  m off the pass found no route past the tables; it walks to 1.1.
- **The Navigator's footprint with a piece (#682's open item 3, A5)**, read
  off the truth from the pelvis -- where the map's inflation and the LIDAR's
  front stop are measured from -- over every carry (min / median / max): the
  cup reaches 0.34 / 0.38 / 0.40 m ahead and 0.29-0.32 m to its side, 0.44 /
  0.46 / 0.48 round; the fork 0.36 / 0.38 / 0.40 ahead, 0.47 / 0.48 / 0.50
  round; the plate 0.31 / 0.42 / 0.46 ahead, to 0.37 aside, 0.37 / 0.48 /
  0.50 round, tilted a median 24 deg in its pinch. Against the inflation's
  0.35 m and the front stop's 0.40 m ahead and 0.27 to each side: a carried
  piece pokes 0.10-0.15 m past the inflation on its side, and its outer edge
  past the front stop's half-width, so a walk that hugs a jamb on the
  carrying side would strike it. Not changed (the pass's route has no door).
- **The bar's readings: NOT MET.** Written before any flight (24 of 32 draws
  over the tour, a fall failing the try, on walk2 and on the carry walker,
  one pod, then 32 held-out draws each, sealed by the frozen commit), and
  read twice on one pod (an AMD EPYC 7B13, osmesa):
  - At b531491 (the cup aimed 2.5 cm short): walk2 0 of 32, the carry walker
    0; held out, 1 (laid in 421 s) and 0. The first piece not laid, a flight
    (walk2's bar and held-out draws, then the carry walker's): the cup let
    go while carried 20 and 13, 16 and 17; a put-down off its place, or its
    reach failing, 9 and 8, 6 and 9; a pick-up at the pass that failed 1 and
    6, 8 and 4; a walk that never got there 2 and 4, 0 and 0; a fall 0 and
    0, 2 and 2.
  - At 0804ecc (the cup gripped at its middle, each put-down moved by where
    dev flights set its piece down, the cup pulled back to 0.29 m): walk2 10
    of 32, the carry walker 0; held out (sealed by 0804ecc), 6 and 0; each
    laid in 413-516 s. Over walk2's 64 the cup was laid in 29 (11 before),
    the plate in 36 (32) and the fork in 37 (13), and a cup let go carried
    in 1 (34 before). Walk2's first piece not laid, the bar's draws and the
    held-out: a put-down off its place 13 and 16 -- the cup's, set down a
    median 1.5 cm right of its spot and up to 7 cm either side of it; a
    pick-up at the pass 4 and 8 (the plate came down as its grip shut, near
    the pass's edge, 14 times in the 64); a fall 3 and 0 (the fork's pull
    back from the pass felled the robot 3 times in each reading's bar
    draws); a cup let go by the walk's hand-over 2 and 0 (its pull back
    still ran out of reach in 4).
  - The carry walker laid no place in either: the plate's put-down reach
    failed in 23 and 21 of 32, then 24 and 23 (no posture, stopped short, a
    fall): its walk leaves the card a median 0.46 m ahead and 6 cm to its
    right, where walk2's leaves it 0.52 ahead and 5 cm to its left, and the
    reference's put-downs are walk2's stances.

## The arms keep off the other robot (issue #678)

**What happened.** #670 kept a step off the other robot as a walk is kept
off it, and an arm still knew nothing of it: at 11:48 on 11-08 Rowan's way
home from the cupboard's lever hooked Luca's arm, their middles 0.68 m
apart, and both fell; at #670's placements (0.54-0.73 m) the reach for the
lever swung out by its side and its jaw met Luca's thigh, or its bow brought
its head into Luca. #678's stage 1 (the night of 11-10; `~/pluggybot-training/
night-shift-1110/t4-arms/findings.md`) prototyped what follows, flew it, and
Ben chose it on 11-12 with four answers: about 1 m of side clearance at a
counter, the minds coordinating; a walk with an arm stuck out refused by the
verbs and waited out, bounded, by the loop's walks; the other robot as a
cylinder first; and only a robot the LIDAR sees, one scan at the move's start.

**What is true now** (`Reach._kept_off`, `HumanoidMission.look_round` and
`keep_out`, `HumanoidBody.reach_routine`, `release_routine` and `arm_kept`,
`steps.ARM_KEPT_OFF`):
- *What an arm keeps off.* Another robot as a cylinder round its soles'
  middle, 0.28 m round (the standing body, arms hanging: its grippers hang
  0.28 m out on the drawing) and 1.55 m tall (its head's top), every part of
  the arm kept 5 cm off it (`OTHER_BODY_M`, `OTHER_TOP_M`, `OTHER_CLEAR_M`):
  a capsule at five points of its axis with its radius, a box at its corners
  and its middle (`arm_points`, `room`), the arm walked in joint space as
  `dip` walks it. An arm begun nearer than the 5 cm only goes no nearer. Its
  place, never its heading: a robot standing square to the counter beside
  this one is narrower that way than round.
- *Whom.* Only a robot the LIDAR sees in ONE scan as the move begins
  (`look_round`): standing still, the humanoid's LIDAR takes no scan of its
  own (#670), so a reach, a withdraw, a walk verb and a walk's hand-over
  with an arm out each take one, recorded as any scan's sighting is
  (`seen_t`, `peer_returns`) and never into the map. A robot it saw is placed
  where `others` places one seen (#675: its body, through this robot's own
  pose error); one it did not is kept off nowhere -- never at its reported
  pose, which at the counter lay 0.49 m from its body at the median. One
  lying down or dead is left out (it lies under the arms' work, and the
  walks' wider disc keeps the feet off it), and so is one past `ARMS_NEAR_M`
  (1.5 m): on the drawing no part of an arm comes further than 0.91 m from
  the stance's middle in any posture a reach or a carry takes (300 random
  arms a side in each of the 99 postures with their waist's yaws, then the
  farthest pushed further by an optimiser from 12 starts in the 30 deepest),
  so past 1.24 m none comes within the body's 0.33 m. With no other robot
  about no look is taken and nothing is judged: a robot alone flies as it
  did, bit for bit.
- *Which ways.* A reach's way out, as the robot stands and in the posture
  the reach takes, and its way home from its pre-grasp, in that posture and
  stood up -- an arm never goes where it could not come back from; a
  withdraw's way home, as it stands and stood up, from where it truly begins
  (`Reach._home_from`): a hand at its goal comes out along its line first,
  `APPROACH_M` (8 cm), and its way home begins at that pre-grasp -- and the
  way judged clear is the way taken (`Reach.judged_home`, where the arm
  begins it within `HOME_RAD` of where it was judged from): judged from the
  grip, or judged afresh a few mm off, a way clear by a hair could cross the
  other robot unrefused. The first of today's way,
  straight, the side way and the way drawn in (`_tucked_way`: its elbow bent
  to 1.9 rad where it stands, its shoulder moved, its elbow out) that keeps
  clear and dips no deeper over the work than today's (#539's and #560's
  rule kept whole). At a counter there is none: today's way out there is the
  side way, which rolls the arm 75 deg out and meets a robot beside out to
  about 1.0 m, and every way that keeps the arm in dips into the counter's
  face (stage 1: 1,500 random ways of one and two via points found none at
  0.65 m). Elsewhere there can be: an arm raised high and out to its right,
  whose straight way home sweeps through a robot ahead and to its right,
  takes the side way 16 cm clear (`tests/test_arms_keep_off.py`).
- *Otherwise refused*, nothing moved, in words the robot reads
  (`steps.ARM_KEPT_OFF`): what did not move, why, where that robot stands
  from its own middle ("another robot, its middle 0.54 m from yours on your
  right"), and for a reach what the other hand's ways would do ("your left
  hand's ways there and home keep clear of it": `Reach.clear_way` judges
  both, and said of its way there alone, 3 of 12 asked could not say a left
  reach would go). A withdraw is judged before #669's let-go, so its
  gripper stays as it was, holding what it holds.
- *A walk with a hand out that could not come home.* A walk brings the arms
  home first (#669), so `step`, `drive_to`, `face` and `find` refuse to walk
  then (`Body.arm_kept`, one look) and say so; the loop's own walks -- a
  charge's among them -- are never refused: their hand-over waits, looking
  every `ARM_LOOK_S` (1 s), until a look finds a way home clear, `ARM_WAIT_S`
  (60 s) at most, then lets go and brings the arm home the way that comes
  least near that robot (at a counter, today's). ⚠ SO DO THE LOOP'S OWN
  HAND MOVES (`refuse` False: a charge's plug-in and unplug), as the verbs'
  (`refuse` True) never do: refused, the plug went unfetched, the charge
  failed, and a failed charge is a death (`HubLifecycle._strand`) -- a reach
  waits, then goes the way out that comes least near that robot; a release
  waits as the hand-over does. A robot whose arm waits is never asked aside
  (`arm_waiting`: standing still with no hand out, a reach's wait would read
  as resting, and walked off it planned off the stance it had left). Each
  wait is kept on the mission (`arm_waits`, the latest 100). One look an
  instant: a walk verb's and its hand-over's are one scan. ⚠ THE WAIT IS NO WALK'S: a drive counts it against
  neither its patience nor its stagnation clock (`Navigator.held_s`), nor a
  step against its own -- counted, a minute's wait in a walk's first
  command left a 40 s walk out of time before its first stride, and a
  longer one gave up "stalled" as the arm came home.
- *What it costs.* A look is one band scan. Judging a reach is about 0.17 s
  of planning on the dev box (way out and way back, four ways in two poses);
  a refused reach also plans the other hand for its words, a full search
  where that hand does not reach, as an out-of-reach reach's words do.

**The text.** The reach's and the withdraw's texts and YOUR BODY's walk
sentence each say it, without naming the LIDAR, which no text the robot
reads mentions: "Its way out, and its way home from there, keep clear of
the body of another robot standing near you: where no way does, your arm
does not move, and it reads false saying where that robot stands"; the
withdraw's the same of its way home, "your gripper stays as it is and your
hand where it is"; and YOUR BODY's "Nor does a `step` or any other walk
move while a hand is out whose every way home would meet the body of
another robot standing near you". #669's withdraw text still begins with
its let-go, the exception after it (the deployed mind misread a condition
there, #669); a quiz of these words on the deployed mind was not flown.

**Flown** (`~/pluggybot-training/loop-1112/l3-arms/`: stage 1's instruments
`fly/fly1148.py` and `fly/pairs.py`, and `fly/read_l3.py`; one Vast pod, an
AMD EPYC 7B13, osmesa; this build at fbaac57 and loop-staging 91d5025, which
it had merged, side by side, both before #684):
- *11:48's placements* (Rowan's door_go4 from where its walk stopped, its
  steps flown, or its tail alone; Luca behind its right shoulder on the
  walker, held still, 0.54-0.73 m off; three noise draws each, 30 flights a
  tree): loop-staging touched in 14 (233 contacts), Rowan fell in 9 and Luca
  in 4 -- stage 1's base, flight for flight; this build touched in none and
  nobody fell, 23 reaches refused, each for Luca's body 0.54-0.79 m on its
  right, each saying the left hand's ways keep clear. The steps #670
  refuses it refuses still, 54 on each tree.
- *Neighbouring props* (both robots running their check's own procedure at
  once, draws 0-3): at the counter's work spots, 1.16-1.24 m apart, the same
  flights on both trees, row for row, 12 of 12, with no refusal and the same
  props done (one fall, the cube's robot's, on both). Moved nearer -- 0.86-0.95
  m (each 15 cm toward the other) and 0.71-0.80 m (22.5 cm), six flights
  each -- loop-staging touched in 4, two of them falling; this build in
  none, with 40 refusals in 8 flights at 0.73-0.99 m, 15 of them in the door
  and cube's flights whose loop-staging twin never touched (the cube's robot
  steps aside first and still stands 0.96-0.99 m off), and one withdraw
  refused at 0.88 m, its arm left out. No prop was done at those spacings on
  either tree: the references need their stance within a few cm.
- *The hands checks* (the lid, the drawer, the door, the cube and the table
  at set-outs 0-31, a robot alone): 160 flights of 160 the same on both
  trees, row for row -- the lid 26, the drawer 21, the door 24, the cube 20,
  the table 24.
- *Again after the first review* (a139754 beside loop-staging e791777, both
  after #684): 11:48's the same -- this build no touch and no fall, the
  same 23 refusals; loop-staging touched in 14, Rowan fell in 9 and Luca in
  4, four of those flights ended by #684's `Stuck` (down 20 s where nothing
  dies) -- the work spots 12 of 12 and the hands checks 160 of 160 row for
  row.
- *The pair's fixture day* (`two_robots.py` as the pair's recording is
  made, before #684's plate; the first robot plugs in at the station while
  the second explores, then the second): one charge each on both trees, no
  death, both robots done. The two parted from 95.5 s: the first robot's
  plug-in looked as its reaches and withdraw began, and a look's scan draws
  on its LIDAR's noise, so its later scans read otherwise; the second's
  charge began at 378 s here and 357 s on loop-staging (on the reviewed
  build, 87fcf5e beside e791777, after #684: 361 s and 352 s, the same day
  otherwise).
- *A hand left out, then a loop walk* (`fly/strand.py`, on a7bb9c5 and
  loop-staging e791777: the nearer neighbours' drawer and door, each robot
  then walking 1 m back from the counter if a hand is still out, the other
  standing still or walking off 10 s after the procedures end; draws 0 and 1
  at both nearer spacings): one withdraw was refused, at 0.88 m, leaving
  Rowan's left hand out. Its walk's hand-over waited for it -- 60 s with Luca
  standing still, then the arm came home today's way, meeting nothing of
  Luca (whose body that way is narrower than the cylinder); 15 s with Luca
  walking off, the arm coming home at the first look that found its way
  clear. Both walks arrived, their drives counting none of the wait (10.5
  and 10.7 s of walking), and the reviewed build (87fcf5e) flew the set the
  same. On loop-staging the same flights touched three times each at 0.88
  m, and at 0.73 m both robots fell, in both draws.

**What it leaves open.** A carried hand (`move_hand`, `pull`) is not
judged, nor a posture a reach's look takes with its hand waiting at its
pre-grasp (#627: a waist turned toward the other robot carries the arm with
it), nor anything after a move begins: a robot walking up beside an arm
already out is met at the arm's next move, and its way home then is the
walk's to wait for. The cylinder is round: turned to the heading the
LIDAR sees, fewer refusals would come where the arms would have passed --
Ben's later step, if live refusals show it. Whether the minds act on the
words -- the other hand, a step away, a `tell` -- is the live world's to
show. And a hand stranded out (a withdraw refused, nothing walking) holds
the world's saves, as any hand out does (`Keeper.busy`).

## The cube from the tour (issue #560)

**What is true now.** Our cube reference places the cube from anywhere in
the house: **28 of 32 draws over the tour** (`solve.py --feature cube --from
tour --n 32`, one pod, an AMD EPYC 7B12, osmesa, each draw its own noise; the
bar, 24, written before it first flew), where the tree before it placed 23
on the same pod, and the place check at set-outs 0-31 reads 20 where it read
17. On fresh draws, 400-431, 23 of 32 (25 for round 1; draw 431's trace
showed the one-tag case the review fixed, so they are not untouched, but
not flattered either: 431 still failed). So `place_cube` is
offered ("The humanoids' first jobs"): 9 minutes from the claim, three
times its slowest pass from the dearest station, the mall's, its area
remembered (177 s and 180 s, the re-pricing's flight and this one), paying
51 (Ben's 90 for 16 minutes, pro rata), and priced at 9.858 Wh
(`energy.json`: flown on this reference from the mall's station it cost
8.124 Wh over 180 s, from the house's 5.955, from the research center's
6.208, and 3 Nov's dearer reading is kept). Three times its slowest pass
over the tour (159 s, on a fresh map; the tour has no mall) gave 8, and
left a claim at the mall about 300 s to write the procedure in, where one
answer may take 300 (Ben's call for 9, 2026-11-11); a claim at the mall on
a fresh map, its find before the walk, was not flown. #669's `withdraw`,
which opens its gripper first, changes none of its flights: every `withdraw` it makes follows an `open_hand` (the first
round's 32 flown on both trees, the same row for row).

| the bar's 32 draws, how each ended | the tree before | now |
|---|---|---|
| **placed** | 23 | **28** |
| shut on nothing or let go as it rose, and never held again | 5 | 1 |
| fell at the first reach | 2 | 2 (the same draws) |
| set down 2.5-3 cm off the mat's middle | 2 | 1 |

![Our reference at draw 0 from the house's station: the walk there (0-91
s), the steps to the counter and in (95-107 s), the cube gripped and lifted
(111 s), set down on its mat (115 s), the arm home (119-123 s), and the
grade (`MUJOCO_GL=osmesa uv run python scripts/solve.py --feature cube
--from station --draw 0 --strip`)](figures/solve_cube_tour.png)

The change is the place's half (`hands_check.PLACE`): the walk there is
the counter's (`solutions.APPROACH`) unchanged. What flying took, read off
64 development draws (200-231 and 300-331; the tree before placed 42 of
them, the reference now 58), every flight's verbs, the cube's and the mat's
true places at each and the true stance where each began (the rows are in
`~/pluggybot-training/night-shift-1110/t2-cube/`):

**A try that failed moved the feet, and every try after it aimed at the
first read.** A reach stopped short (its tips on the board, or out of reach
from how the robot stood) or a shut on nothing moved the stance 3-7 cm, up
to 12 at a second try at 75 deg, which took 12 s; the tries after it went
to where the cube had been in the frame before. Each try after one that
failed now reads the mat and the cube again and aims where the cube is, 1
cm on along the fingers (as the first try's read is, below), its carry
from the mat read in that frame; where only one of the two reads, the other
is moved as that one seems to have moved, the feet's move moving both alike
in the frame (read again only where both read, a fresh draw whose mat was
out of view aimed its next try where the cube had been and set it down 6 cm
off). Of the development draws' misses, 14 never
held the cube on the tree before and 2 do now; of the flights that tried
twice or more, none placed it at its second try there (15) and 10 of 17 do
now.

**The first try keeps the read the step carried.** The cube's first read,
from 0.4-0.45 m, is true to 0.2 cm; the step to the work overshoots, 2.6 +-
1.1 cm (asked 0.11 m, walked 0.137), and the legs' reckoning of it is 0.4
cm short, so the first reach is aimed 0.2-1.6 cm on from the cube's middle.
Read again just before it, the aim true to 0.5 cm, the reach found no
posture at 0.27 m, stopped short more, and twice turned the stance 14 deg:
13 placed of 22 where 17 had been. The first try also stays as it was; its
two falls (the bar's draws 3 and 9, and one development draw) are its own.

**A lift that left the cube behind is a try that failed.** A shut on the
cube's edge read held, and the cube stayed as the hand rose: the carry took
nothing to the mat. The hand now reads whether it holds after the lift
(`left.held`, once its running mean has had 0.1 s: a grip lost read held up
to 40 ms on, #646) and, holding nothing, lets go, goes home and tries
again.

**The tries go 60, 60, 46 and 75 deg.** Second at 75 deg, a try took 12 s
and moved the feet most; second at 60, again aimed, more placed (51 of 64
where 47).

**The carry aims 0.6 cm on and 0.5 cm to the right of the grip's offset.**
The cube's reads share the step's and the reach's error with the mat's, and
the offset carries it through, so the set-down's error is the cube's own
move as the jaws shut on it (along them, 0.3 cm to the left at the mean, up
to 2.5 either way) and the carry's: a mean 0.55 cm short and 0.5 cm to the
left, which took 9 placings of 62 past the 2.5 cm allowed. Aimed off by its
mean, the set-down's error is +0.2 +- 0.9 cm ahead and -0.05 +- 1.1 across.
Centring the offset between the jaws instead (the cube is centred there
after a shut, to 0.4 cm) was worse: the jaws' axis carried the step's
sideways reckoning, which the mat's read shares.

**Not taken: the step asked further.** The cube lands 0.290 +- 0.011 m
ahead where the step asks 0.32, and a first try at a cube nearer than 0.285
held 9 times in 20, further 37 in 44; asked 0.335, the first tries held
more, but more cubes were never held and three robots fell (49 placed of 64,
the reference without it 47); asked 0.345, 41.

| development draws (200-231, 300-331) | placed of 64 |
|---|---|
| the tree before | 42 |
| read again before every try, the first too | 13 of 22 (17) |
| ...after a try that failed, aimed 1 cm on (60, 75, 46, 61 deg) | 47 |
| ...aimed true | 50 |
| ...aimed 1 cm on, at 60, 60, 46 and 75 deg | 51 |
| ...and the carry aimed off by its bias (round 1) | 59 |
| ...and the step asked to 0.335 | 48 of 62 |
| round 1's, one tag read again carrying the other, `held` read 0.1 s on (**the reference**, after the review) | **58** |

⚠ What it still loses: the first reach falls on a few draws (2 of the bar's
32, 1 of 64 dev), the same draws on the tree before; and some cubes are
never held in four tries (2 of 32), each try stopped short, out of reach or
shut on nothing.


## A flight that never got up (the drift batch of 11-10)

One flight of the doubt rule's drift batch, seed 13 from the hall, never
finished on either tree: 57 and 72 minutes at full CPU on a pod (an EPYC
7773X), nothing on its stdout or stderr, where the rest flew whole in 22-42
minutes. It read as a C call that never returned, and was not one: a
flight's stdout redirected to a file is block-buffered (a whole flight's
report is 4 KB, under the 8 KB buffer, so every flight's stdout is empty
until it ends), and a SIGINT that reaches Python prints its traceback
within seconds, so the one sent must have reached a shell. Flown again on
another CPU and Python (an i5-9600KF on 3.14, the pod's 3.12) it hung the
same way, every flight of it at the same step: 117.7 s in, the explore's
walk north along the house's east wall crossed the garden's pressure plate
(a sprung pad 21 mm proud of the floor), went down on it, and fell forward
0.65 m over the garden lamp's pole (4 cm, which its maps held within 10 cm
of it), the pole between its legs. The get-up thrashed round the pole and
lay still against the house's wall, 120 deg from upright, for good. Two
flights tell the causes apart: with the pad's contacts off (under the
LIDAR's band and the depth camera's low layer, so nothing the robot senses
changes) the same walk passed the pole 0.42 m off and never fell; with the
pole's off from the step it fell, the get-up stood 2.6 s later. The plate
tripped the served walker, and the pole kept it down.

`HumanoidMission._getup_routine` holds the walk until the get-up stands,
with no budget of its own: served, a fall it cannot get up from is the
lifecycle's `stuck` death, whose stand-up closes the routine. The spike's
lifecycle is not mortal, and the explore's budget and the walk's timeout
are read only between drive steps. Flown under the served rules (mortal,
its 300 s restart, the explore inside `_until_stood_up_routine`), the same
fall is a `stuck` death 20.6 s on, and the stand-up 300 s later closes the
explore; the physics thread never stalls, and no save is taken while it
lies (`Keeper.busy`, 320 s). The death's clock ran only while the trunk lay
60 deg or more off its posture, so a get-up stuck short of that never died
`stuck`.

**What is true now** (#684, Ben's calls of 11-12):
- A get-up's death is timed off the get-up: a body that rights itself is
  `stuck` once its get-up has held its legs 20 s, whatever its tilt, a
  stand under 5 s between two get-ups one struggle on one clock (the drift
  spike's own guard's rule, kept). Of
  #679's 151 falls on the served get-up, one that stood dies now (laid on
  its front against the south wall, up at 21.5 s, its trunk past 60 deg for
  at most 19.7 s running).
- Where nothing dies the same clock ends the run (`lifecycle.Stuck`, a
  `MissionAborted`): every instrument's flight and every demo's day. 13h's
  own command line, on the raised plate, fell at 117.8 s and ended at
  137.8 s, its rows kept; the drift spike's own guard is retired for it.
- A batch's flights are watched (`pluggybot.batch`): their output as it is
  written, and every thread's stack on `kill -USR1` of a flight's python.
- The garden's plate is flush with the floor: the floor's own square, read
  off its load, its pad paint nothing touches (`floor_plate`;
  ActivityPattern.md §3.1). 13h on it crossed the plate at 115.9-117.5 s,
  where it had fallen, its pelvis at 0.725 m at the lowest, and five times
  more, the lamp lit at 115.4 s; it explored, walked both trips and docked
  with no fall. Of the served walker's straight walks, none fell on the
  flush plate (15 crossings, 9 begun at a standstill, the toes at its edge);
  on the raised pad, none fell either, but all 9 begun at its edge and 3 of
  the 15 stood on its edge for good (0.10-0.25 m); the lab's three, the
  same 21 mm sprung pad, were crossed 21 times at speed, and stalled all 9
  edge starts (0.07 m of a 0.3 m ask, 0.11 of 0.6).
- The lab's three are flush too (#684's call 3): from those 9 starts the
  walker walks what it walks on bare floor 1 m off them (0.18 m of 0.3,
  0.49 of 0.6), each start pressing its plate, and crossed at speed 19
  times it never fell. A load plate presses about once a footfall on it:
  1-3 presses a crossing (the cage counted 41 acts over 19 crossings, the
  sprung pads 27 over 21), each the robot's whose weight pressed it (its
  contacts over the square carrying the most of the load).

## The storeroom keeps an error, and a match the walls only partly agree with is searched (issue #681)

**What happened.** At the storeroom's counter the matcher could not take
back an east-west error over about 0.15 m. The room's north and south walls
stay on their walls whatever the offset along them and make half or more of
a scan's points, so the fit is never `inconsistent` and the bounded search
never runs; the points that would fix east-west -- the west wall 5 m behind,
the counter's front -- lie past the field's 0.3 m reach and pull on nothing,
or "fix" it on points that alias (the counter's front, its props and the
wall behind it are parallel lines, the front and the wall 0.60 m apart); and
the scan is laid in from the wrong pose, a copy of the room that every later
scan matches in full. Live (11-07/08) the store's error had a median of 0.16
m, a 90th percentile of 0.66 and a worst of 1.04 (the lobby next door 0.05
and 0.14), in jumps kept for hours: Luca's +0.57 m at 09:09 on 11-08 stood
2.6 hours. Its errors came from a hand's work dragging the body while no
scan is taken, from a map laid askew on the way from the mall, and from the
bounded search itself. A static probe at 40 poses round the counter, on a
map laid from the true poses, kept a belief +0.3 or +0.4 m off at all 40
(the night shift of 11-10, T5).

**A match the walls only partly agree with is doubted**
(`scan_match.DOUBT_*`): one that held a direction, or whose weakest
direction rests on under a tenth of its points (`DOUBT_WEAK_SHARE`), with
under 95 % of its points on walls, is searched as an inconsistent one is --
and a slid one too -- and a pose found there is taken at 90 % on walls with
every direction fixed, never one the rule would doubt again (a take whose
weakest direction rests on a few aliasing points, at 90-95 %, would be
searched off at the next scan). Found nothing, the fit is what it was: a
doubted scan is laid in as an `ok` one is, correction and all, and a slid
one stays refused (left out of the map, doubted scans starved new ground:
on the way to the mall's station 229 of ~720 never went in, the station's
fix kept the map laid askew under `ASKEW`, and no walk out of there found a
route).

**Never on a fit's share alone.** The first two rounds also doubted any fit
with under 80 % of its points on walls, and the review of 11-12 found that
half capturing a RIGHT belief: where a copy of the room is already in the
map, the copy's scans cleared some of the true walls' cells, a fit at the
truth reads 0.54-0.83 on walls, and the search takes the copy, which
explains more and is under `UNIQUE_M` from the truth (so no rival of it).
What tells the two apart is what fixes the fit. Toward the counter, an
error's fit is often "fixed" in every direction by the few points that
alias, its weakest direction resting on 0.05-0.19 of its points (median
0.075, the counter's 40 poses at 0.2-0.4 m off, a map laid true; 0.05-0.26
on the live maps); a right belief by a copy rests its weakest on the room's
own walls, 0.12-0.30 (Luca's live map of 11-09) and 0.17-0.30 (the tests'
copies). Nothing else read separated them: how much more the copy's place
explains (the truth 0.71-0.88 of the copy's count, an error 0.60-0.79 of the
truth's), how many of the found pose's points on walls facing the
correction sit on a wall at the kept pose too (0.44-0.66 against 0.14-0.64
toward the counter, where its parallel lines double every wall), or rays
passing a mapped wall. The same review's copies, read the same way (T5's
counter probe; a right belief at 40 true poses round the counter, and
beliefs 0.1-0.4 m off along x, one scan each):

| map | right belief moved over 0.15 m: base / share doubted / now | a belief off ending within 0.15 m (of 320): base / share doubted / now |
|---|---|---|
| laid from true poses | 0 / 0 / 0 of 40 | 97 / 319 / 313 |
| Luca's of 11-08 | 0 / 0 / 0 | 91 / 300 / 263 |
| Luca's of 11-09 (copies 0.25 m apart along the counter) | 0 / 35 / 0 | 101 / 10 / 100 |
| Rowan's of 11-09 | 0 / 0 / 0 | 82 / 319 / 301 |
| Rowan's of 11-08 (the storeroom laid 0.57 m off whole) | 40 / 40 / 40 | 1 / 0 / 1 |

On the tests' storeroom with a copy the base laid 0.3, -0.3 and 0.57 m off
along the counter (20 minutes from 24 work poses, the band's `free_from`),
a right belief standing 30 s at 12 work poses was moved onto the copy at
5, 0 and 0 of 12, the base's own (the share doubted 12, 11 and 12), and
a belief 0.3 or 0.4 m off on the room without a copy came back at
12 of 12 each (the base none). From the truth across the house
(7,840 places, a map laid true) 17 fits are doubted and the pose moves over
0.1 m at 2, the hall's stairs, the base's own. Flown with Luca's map of
11-09 as the robot's own from the start (the counter's flight: a look round
in the storeroom, the drawer's, the door's or the chest's job, then
standing, four steps toward the counter and standing), the share doubted
took the copy at the look and ended 0.21-0.24 m off; this round flew the
base's flight row for row, 0.12 m off at the end (where the base's own map
lies), every fit `ok`, every job passed.

**A slid fit's search is taken only within 8 cm and 1.5 deg of where the
fit slid** (`DOUBT_SLID_*`, its correction along the directions the walls
fix). On Luca's map of 11-09 a belief 0.1 m off slid 0.17 m toward one copy
of the counter and the search found the other, 0.25 m from it (13 poses of
40; the base refuses a slid fit, and kept the belief within 0.15 m). Where
the counter's errors slid back, the search's pose was within 0.053 m and
0.74 deg of the slide's.

**...and its search takes only a place no other explains as well**
(`DOUBT_RIVAL_SHARE`). As first written the doubt walked drift flight 19h
(seed 19 from the hall) 0.8 to 5.95 m off in 45 s: along `sidewalk_east`
and `sidewalk_2_north` -- the facility's wall and the loop's fence,
parallel, nothing across -- fits held along the walk with 89-95 % of their
points on walls were doubted, and the search took its best 25 times, each a
few points ahead (2-11 of ~150) and up to 0.6 m behind the robot (the map's
far end lags the scan's reach, so a pose behind puts a few more far points
on mapped wall). Every place along such a sidewalk scores 0.95-1.00 of the
best; at the counter the best place half a metre or more from the truth
scores 0.65-0.75 of it. So the doubted search refuses a best that a place
`UNIQUE_M` (0.5 m) or more off matches at 0.9 of its points on walls, each
place at its best turn. It holds only within the window: in a corridor
fenced in panels a metre apart, a belief 0.75 m off (the truth outside the
±0.6 m window) was taken onto the alias a metre from the truth at 20 noise
draws of 20, the base keeping 0.75; the served sidewalks repeat nothing.

**...and only a place that explains the scan much better** (`DOUBT_GAIN`:
1.2 times as many points on walls as the fit it would replace, never the
pose asked about: a fit corrected across a sidewalk would make a gain of
any place along it). Captured on a
drift flight, a weakly fixed fit on open ground -- the street, its fences
far -- was searched once a second, and the places it found put 1.00-1.11
times as many points on walls as the pose asked about: four takes in 5 s
moved the belief 10 cm the wrong way and 9 cm back, 0.09 to 0.12 m off in
all. The counter's take-backs gained 1.21-2.1 over the fits they replaced
(medians 1.34-1.74 by map, on T5's probe; the sidewalks' aliases of 19h,
1.01-1.07 over the pose asked about).

**...at most once in 10 scans the matcher is handed** (`DOUBT_EVERY`, a
second at the LIDAR's rate), afresh wherever the belief is put somewhere
new (a fix, a relocation; a stand-up forgets the map), never at
`fuse_next`, which the Navigator calls at every scan it does not map. A
search costs ~25 ms where a match costs 0.5, and
standing among something its map lacked, the review's robot was doubted at
600 scans of 600 in a minute; between searches a doubted fit is laid in as
it was and a slid one refused, as before #681. A minute standing, in ms of
matching a sim second (base / first two rounds / now): beside the review's
copy 3.7 / 173 / 3.7, beside a cabinet the map lacks 3.5 / 144 / 3.5, in a
corridor with a post it lacks, every fit held and doubted, 2.9 / 117 / 15.

**Flown** beside the base (loop-staging with #682), D4's batch again, every
flight a process, both trees on one pod (an EPYC 7C13), the code as it
stands:

| | this round | base |
|---|---|---|
| the counter: a belief moved +0.25/+0.4/-0.3 m after each of three jobs | 9 of 9 taken back at the first scan, 0.005-0.009 m | 0 of 9, 0.23-0.41 m kept |
| the counter's jobs from each station (9) | the base's, row for row: 9 passed (from the mall's station 0.27-0.30 m off in the storeroom) | 9 passed |
| drift flights (60 pairs) | the base's, row for row: 429 scans doubted in 28 flights, none taken | all docked, every walk there |
| walk-ins (16), lost (52) and approach (33) trials | the base's, row for row, all docked | |

On the same pod, without `DOUBT_GAIN` 11 drift flights parted from the
base, three worse at their worst by 2-27 mm and one better by 7 mm, the
takes on open ground moving a belief a few centimetres either way; with it
measured from the pose asked about, one (a take 0.09 m off the truth where
the base was 0.15). From the mall's station the takes without the gain put
the robot 0.01-0.03 m off in the storeroom where the base was 0.27-0.30 --
a leg that turns with the CPU on both trees (on two EPYC 9554s, before the
review's fixes, both went 0.9-1.0 m off alike and the chest failed): #681's
call 3, a fix under `ASKEW` keeping a map laid askew.

**What is true now.** `tests/test_scan_match.py` pins it on synthetic scans:
a storeroom whose side walls hold while its end walls disagree by 0.3 m is
taken back and laid in, at `DOUBT_FOUND_SHARE`'s bar; the same storeroom
with a copy laid 0.3 m along the counter, where a right belief's fit has
under 0.8 of its points on walls and the search finds the copy, keeps the
belief; a full fit is doubted where its weakest direction rests on under
`DOUBT_WEAK_SHARE` of its points and not over; a slid fit's search is taken
only within `DOUBT_SLID_M` of where it slid; a run of doubted scans is
searched once in `DOUBT_EVERY`, the count kept across a restart and
started afresh at a fix or a relocation, never at `fuse_next`; a take the
rule would doubt again is refused, and one that gains under `DOUBT_GAIN`
over the fit it would replace; in a
corridor fenced in panels a metre apart the search takes no place another
explains as well, and a fit it finds nothing for keeps its own correction;
a corridor with a post the map lacks, a tenth of the scan, is doubted
(`DOUBT_HELD_SHARE`); a straight corridor held along is laid in; and a slid
or doubted fit gets the doubt's search, an inconsistent one the base's. The
drift rows' `matched.ok` counts no doubted fit: across the periods, `ok`
before is `ok` + `doubted` + the doubt's `found` after (an inconsistent
fit's `found`, the base's, is rare: none in its 60 drift flights), and the
instruments count all three (`FIT_TAKEN`).

**Not done.** A drag past the search's 0.6 m (the live 1.1 m of 13:56): the
wide search or the station. The slow creep (0.2 m and 2 deg an hour with
every scan matched, Rowan charging and in `DONE`): not explained. A fix
under `ASKEW` keeps an askew map (the mall's leg). An error toward a wall
whose far side was never seen is not doubted: its points land on unseen
cells, no information, so the fit reads 0.93-1.00 on walls (the review's
dead end mapped from one side: 0.3 m toward its end kept in 44 cases of
45). A copy already in a kept map goes only with the map (a true death, a
stand-up, a `MAP_EPOCH` bump or a world whose geometry changed); it no
longer moves a right belief the base's fit kept, and from a belief 0.2 m
off on Luca's map of 11-09 one pose of 40 was searched onto the copy where
the base's fit ended 0.1 m off. The base's own search, from an
`inconsistent` fit, still takes the best of two places that explain a scan
alike, and the doubt's beyond its window.

## One robot's interrupt asked, the other robot held (issue #690)

**What the live pair showed.** On 1e8f1d48 (11-12, 20:30 UTC) Luca fell 63
s into a plain `drive_to` in the research center's lobby, and its get-up
stood it 120 deg off the truth: every scan after it was refused for the 26
minutes to the pause (944 matched from the restart to the fall, none after),
and it walked into the garden believing it stood at the storeroom's
counter. The log has why: Rowan's `battery below 22% -> ask` fired at
t=227566.3, its mid-errand interrupt was answered at t=227572.1 -- 5.8 sim
s, in 3.1 wall s -- and Luca's "you fell" came at t=227572.9, 0.8 s later.

**Not the IMU's filter.** The first hypothesis -- the attitude filter's
gravity correction (`imu.Attitude`, `TAU_S`) running on through a tumble,
when what the accelerometer reads is no longer gravity, a wrong tilt turned
into a heading -- was flown and is not it. 72 falls on a bare floor (8 ways,
45-100 N*s, standing and walking at 0.6 and 1.0 m/s) left the filter's
heading within 0.8 deg of the truth's. 201 flights of the served mission in
the house -- the lobby's walk shoved mid-stride 8 ways at 70-150 N*s, along
its west wall, standing where Luca stood up facing four ways, three other
IMUs' draws, a belief 0.42 m off with its map laid through it, as Luca's was
-- fell 129 times: the 101 get-ups that stood left the heading within 1.7
deg (p95 1.1), the filter's own change across a fall at most 1.35, though
the bodies turned 5-177 deg and the get-ups took 2.1-14.8 s, and 28 never
stood in 20 s (`Stuck`). The filter follows the gyro exactly but for a
two-step lag (it reads the sensor computed the step before), so a body at
rest before and after is tracked whole; its scale error (Luca's +0.39 %
about z) is a fraction of a fall's 5-12 rad; and its correction turns the
estimate about an axis level in its own world, so the only heading it adds
is the holonomy of the tilt errors it chases.

**The cause: one robot's blocking wait stepped the world without the
other.** `HubLifecycle._ask_interrupt` waits on the mind with
`self.body.run(self.body.hold_routine(THINK_SLICE_S))`: a blocking
`tick.run` on the asking robot's stepper, inside the pair's
`tick.run_many`. Through every step of the ask the other robot's stepper
never ran -- no policy step, so its legs held the last targets they were
sent; no step of its IMU's filter; no fall watch, no odometry; and no seam,
whose hooks on the first robot are the world's activities, the stream's
frames, the recorder and the real-time pacer (Rowan's 5.8 s passed in 3.1
wall s, unpaced, unstreamed). Flown as live (`humanoid_drift_spike.py
--asked walk`: the pair, Luca walking the lobby's walk, Rowan asked through
the real `interrupted()` by a mind that answers carry on after 5.8 s): Luca's
physics stepped none of the ask's 2,900 steps; it lay on the floor 1.4 s
into the ask while its filter's tilt read 1.5 deg; 0.78 s after the answer,
as the filter's correction caught up, its fall watch fired (live: 0.8); and
the get-up stood it 2.6 m and 40 deg off, kept. What it lost is the turn its
body made unseen: the correction rights the tilt by the shortest arc, and
whatever of the fall's turning that arc does not undo stays as a heading.
Whatever else the other robot was doing, the ask stopped its body as it
stood (below).

**What is true now** (`tick._Loop.run`): a blocking `run` begun inside
`run_many` is the loop's -- every robot applied in the loop's order, one
world step, every robot booked -- each other robot held through its wait
(`run_pair`'s: its body's own hold, its scans matched), one wait across the
ask's think slices, closed once its own routine drives it again, and its own
routine left where it was, so no second robot's blocking call can begin
inside the first's. What such a run raises that is not its routine's own --
a held robot's wait's, the step's -- is `run_many`'s to raise
(`tick._Raised`): inside the asker's stack, the procedure runner's `except
Exception` took the other robot's error for its own. `_ask_interrupt` is the
one such run in `src/`: every other blocking twin (the explore, a charge, a
decision, a walk) is called only by scripts and tests. Flown, the other
robot doing each thing as the second's interrupt is asked for 5.8 s
(Rowan's), 30 and 115 (`INTERRUPT_TIMEOUT_S`; the drift spike's `--asked`),
before (the base, on a pod) -> now (on the dev box):

| the other robot, as the ask flies | 5.8 s | 30 s | 115 s |
|---|---|---|---|
| walking the lobby's walk | fell, stood 2.6 m and 40 deg off -> held, walked on | fell, 2.6 m and 40 deg -> held, walked on | fell, 0.9 m and 39 deg -> held, walked on |
| standing, resting | leaned 7.7 deg -> held | fell -> held | fell -> held |
| a hand out at a counter's height | fell -> held, the hand kept there | fell -> held | fell -> held |
| getting up from a shove | its get-up stopped where it lay, then stood -> stood | stopped, and its 20 s clock ran on: `stuck` -> stood | `stuck` -> stood |
| plugged in at its station | leaned 16 deg -> held, charging | leaned 17 deg -> held, charging | leaned 23 deg, fell after, `stuck` -> held, charging |
| turning a `step`, 90 deg | leaned 4.7 deg -> held, turned | fell -> held, turned | fell -> held, turned |

Before, its physics stepped none of the ask's steps in any of the 18; now
every one, its trunk (but for the get-up's own fall) at most 1.2 deg off
upright through any ask, and each ended within 0.01 m and 0.2 deg of the
truth. Held, a robot stands where its legs' policy has it (the walker
marking time, the standing policy holding, a get-up getting up), its hand's
own loop holding its reach, its plug in, its senses on.

**The price.** While one robot's interrupt is asked, the other's errand
pauses, standing, for the whole ask -- up to 115 s of think and its slices.
A walk or a `step` counts none of the pause: the held robot is told how long
it was held (its stepper's `held`, into `Navigator.held_s`, as an arm's wait
for its way is, #678), and a walk goes on with its patience whole and a
fresh plan -- told nothing, asked 115 s, the lobby's walk ran out of its 60
s and gave up where it stood, and a `step` held 30 s read "out of time". A
procedure's budget and a job's minutes run on through the pause.
Before, the body went on without it. The whole fix is `_ask_interrupt` as a
routine, yielding its think slices as `_decide_routine` does, so the other
robot's routine runs on through the ask (not done: it moves `interrupted()`
and its three callers, a walk's among them).

**Nothing else moves.** On the same pod, a robot alone (600 s) and the
pair's day with a fall and a death arranged fly the base's trajectory bit
for bit (`determinism_spike.py --compare`: 1,201 and 528 state samples
identical, two runs each tree, and the pair's day again on the dev box);
the 201 flights of the falls above, the drift spike's walking falls in the
lobby and two drift flights (Luca's and Rowan's draws) are the base's row
for row; and the pair's recording, flown as its fixture is, is the base's
but for two wall-clock stamps.

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
8. A flight that seems hung is read from inside it: run it `-u` (redirected,
   its stdout says nothing until it ends), signal the Python process and
   never its shell, and give it `faulthandler.register(SIGUSR1,
   all_threads=True)`, which dumps every thread's stack with no ptrace; one
   that calls `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` lets `py-spy dump`
   attach where Yama allows only a parent (scope 1).
