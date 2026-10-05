# Simulation Notes

MuJoCo lessons from building PluggyBot, each one paid for. An entry says what
broke, why (the physics), what is true now (the constant or the rule) and what
pins it. The story of how a thing was found lives in git and the issues; a
measured number lives at its constant with its failure mode attached
(CLAUDE.md, "Working style"). Read this before touching `models/`, contact or
actuator parameters, the arm and its coupling, or anything that trusts
odometry. The pattern docs say *what the constraint is*; this file says *how
we found out*, in the order it was paid for.

Two bodies came before the quadruped: the plug-anywhere robot (milestones
5–7) and the wheeled rover (milestones 8–15). Both were deleted in #376, and
the tag `rover-final` holds the rover, its worlds and the whole of every
story it paid for. What this file keeps of the wheeled era is what still
binds any body, or the house the quadruped lives in, cut to what is true
now; the quadruped's own entries start at "The quadruped body".

## Physics modeling rules

### Use `integrator="implicitfast"`
MuJoCo's default explicit Euler adds energy to velocity-dependent forces --
velocity actuators and joint damping. Every world here integrates
`implicitfast` (`legs/model.py`, `home/world.py`), as most Menagerie models
do.

### A light mass on a stiff servo chatters
A 50 g wheel under a velocity servo overshot its target within one 2 ms step
and chattered at the timestep frequency: the robot bounced instead of
driving. What was missing is honest inertia: `armature`, the rotor's inertia
reflected through the gearbox, which grows with the gear ratio squared. On a
rig, a constant-force `<motor>` with joint `damping = F/v` does the same. The
dock's sprung pins carry an armature for this reason ("The quadruped's dock",
trap 3).

### THE caster lesson: MuJoCo combines pair friction as the elementwise MAX
At equal `priority` a contact's friction is the element-wise **maximum** of
the two geoms' values (condim combines as max too; solref/solimp by priority,
then `solmix`), and the floor's default is 1.0. The rover's
`friction="0.001"` caster was a full-grip rubber ball for weeks, the hidden
cause of a pitch resonance and 12.6 % odometry creep, until Menagerie's
Stretch idiom, `condim="1" priority="1"`, went in. **What is true now:** a
per-geom friction does nothing unless its geom has priority: the coupling's
peg (`coupling.PEG_FRICTION`), the fork's end-ramps (`legs.arm.RAMP_MU`) and
the quadruped's feet (`legs/model.py`;
`test_legs.py::test_a_foot_grips_with_its_own_friction_not_the_floors`) set
it, and the dock's `--sticky` premise is what the MAX costs ("The
quadruped's dock"). Know the pair-combination rules before trusting any
per-geom contact attribute.

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
source with a hard contact, `coupling.GRIP_SOLIMP` (the tower's blocks,
`challenge/stack.py`; the bench's cubes, `challenge/bench.py`), on which a
tower stands iff its centre of mass is over its support. Any free body meant
to rest on another for more than a few seconds needs it, or the sim grades
the solver (Challenges.md §4).

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
band (a 0.23° square-up went from 9.5 s to 55 s). The legs' policy has the
same shape: it walks nothing under ~0.2 m/s ("The quadruped's dock"). The
bill is paid in the controller, never by removing the physics: a nonzero
command is floored above the band (`control.turn_command`'s `W_BREAKAWAY`;
the quadruped's crawl raised to 0.25 m/s, "The first quadruped deploy"). And
when behaviour looks violent, check that the actuator could exist in that
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
now: `Body.ramp_routine` walks an actuator's setpoint to its target, and the
arm's joint targets move at most `legs.arm.ARM_SLEW`.

## Test & world hygiene

- **A physics test world carries no scenery.** Scenery once parked a box in
  a drive-test lane, and a re-measurement drove into a board placed two hours
  earlier. Copying the floor into an including file doubles every contact: a
  "repeated name" MJCF error means *delete* the duplicate, not rename it.
- **Every debugged failure becomes a pytest assertion, shown failing first.**
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
  and the front stop track the OUTERMOST geometry, not the chassis (the
  quadruped's front stop covers its fork: "The arm on the served body").
- **The reflex is armed in every manoeuvre, not just while driving** -- a
  look-around spin ran blind 7 mm outside a box -- **and never re-armed while
  it is firing**, which turned a bounded `BACKOFF_TIME` reverse into an
  open-ended one (`Navigator`'s scan step).
- **Exploration ends on no *reachable* frontiers**: frontiers off the
  robot's own connected floor are dropped before any plan
  (`behavior.navigation.plan`, `own_component`), and it ends after a
  look-around plus repeated pathless replans (`STRIKES_TO_FINISH`).

## The coupling: a gravity seat, and the peg is the connector (milestone 8)

A tool hangs by a long split peg in two upward-open V-trays, the fork takes
the peg outboard of the trays, gravity is the latch, and the verbs are slide
and lift (`rack/coupling.py`; the arm's fork is "The quadruped's arm, its
coupling and the rack"). What the rover's rig and robot measured that still
holds:
- **Retention beats traction.** The seat held through 8 m/s² of shake
  carrying 300 g, with no spring, magnet or actuator.
- **A gravity latch cannot push.** The restoring and the disturbing lever
  about the peg are the same 22 mm, so a tool's own force swings it (0.10 N
  → 5.3°, 0.50 N → 51°, flopped onto the fork). The lean-pad reacts that
  torque in compression, which is what lets a tool press at all.
- **The peg IS the connector.** It sits in four V-notch plates with ~0.45 N
  of gravity preload each, wiped clean as it seats; split into two
  conductors round an insulated centre, the left and right V-pairs are the
  two poles (`coupling.module_power_state`; on the arm, `legs.rack.tool_power`).
  A pad's preload, capped by the same weak geometry at ~0.1 N, could never
  carry power.
- **A brown-out is reported as the worst duration in the window, the poles
  separately**: 0.4 % of steps as blips and 0.4 % as one outage are
  different hardware, and a half-seated coupling (one pole on) is a failure
  a boolean files under "off". The worst outage under harsh driving was
  178 ms, so a tool's holding capacitor is sized for ~200 ms (Parts.md).
- **Near the fork is not seated.** A position heuristic read True through an
  ejection; the electrical criterion is the seating check.

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
  outside the capture backs out and takes another run, as the quadruped's
  walk-ins do (`legs/rack.py`, `legs/dock.py`). **A retried measurement is a
  new draw from the error distribution; a retried constant is the same error
  again.** And a swap VERIFIES its outcome -- seated and conducting, or
  hung -- before it calls it done.
- **A tag buys identity, and the last centimetres are the tag's.** Every
  reading is keyed to a decoded id, so a servo steers on the marker it means
  (a colour-plate stand-in once locked onto the charge bay's marker and
  dragged a module 22 cm toward the wrong bay); a stale prior loses to an
  observation; and odometry cannot close a terminal approach, so it is
  ranged off a tag (`rack/tags.py`; the dock's, the rack's and the plate
  signs' walk-ins).
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

## The pen would not stow (issue #10): two clearances, and a tool's carry configuration

A set-down must satisfy two clearances **at once**: a floor -- the peg rides
over the tray flanks, 14.7 mm above where it hangs -- and a ceiling --
whatever else the tool carries must clear what hangs under the rail. The
rover's pen had a window of ~1 mm, and three faults sat in a row, each hidden
behind the one in front: its rail above the pen line, its carriage parked
where the last stroke left it, and a pick that inherited the lift a stow had
left behind. The jammed module slipped the wheels, and the drive, counting
the slip, returned "arrived". **What is true now:** a tool is carried and
stowed in its CARRY configuration, never where the work left it
(`mission/errand.py`), and every attempt restores its own entry state. **A
capability is not demonstrated until it has been demonstrated twice in a
row**: the second cycle starts from the state the first one left.

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

## Stroke programs (issue #11): what a number cannot see

Splitting *what to draw* (`tools/strokes.py`) from *how* left two lessons for
whatever draws next:
- **A mirrored figure is invisible to every number.** +lat is the viewer's
  LEFT, so text advances toward −lat; flip it and bounds, ink length and form
  error are all identical. The sign convention has its own unit test
  (`test_strokes.py::test_text_advances_away_from_the_viewers_left`).
- **Report shape error, not tracking error, and decompose before you fix.** A
  rigid-translation fit against the command splits *where it landed* from
  *what shape it is*: the rover's first square measured 10.35 mm but was
  2.14 mm of form under a 17.47 mm offset -- displaced, not distorted, one
  calibration constant and not mechanics. And a figure clipped to the tool's
  reach (`Envelope.for_board`) draws flattened while its tracking reports a
  perfect trace of the wrong commands.

## The whiteboard question (issue #22): a grader that could not read

The plan was to read the board. Measured with real Hershey digits, **a 6 and
an 8 are closer to each other (1.7 mm at 55 mm caps) than a correctly drawn 6
is to its own ideal (1.1 mm)** at every size a robot draws, and coverage is
no better. So the grader is a split: **correctness** against the answer the
mind committed to at claim time, and **fidelity** of the ink to *those*
glyphs, a check for wrong WORK. The fidelity bar was measured too: a busy
figure covers the glyph it stands in for and only the ink → glyph direction
notices, hence `ANSWER_MATCH_MM` 4.0 with an ink-length ratio (table and bars
at `economy/questions.py`). **Before building a recogniser, measure how far
apart the things you must tell apart are, in the units your machine's error
is measured in.**

## Drift hygiene (issue #42): the dock is the anchor

Every long-run failure family of the rover traced to uncorrected dead
reckoning, and the system survived COHERENT drift (belief and map drifting
together) and died of DECOHERENCE. The dock is the one pose a robot occupies
to millimetres by construction, held for minutes every cycle, so it
re-anchors the estimate there -- ⚠ to the COMMISSIONED pose, never the
belief. Measured: anchored to the believed rack, the anchor tracked the
belief's own error, 0.003 → 0.146 → 0.344 m over four sim-hours, a loop in
which nothing references the world. **What is true now:** the map frame is
DEFINED by the dock, as on hardware; `Body.anchor_at_dock` snaps to
`rack_prior`, on legs the dock's commissioned pose read through its board
(`legs.body.QuadMission.anchor_at_dock`; "The quadruped's dock"). Between
docks every level scan is matched against the map ("The map stays true under
drift").

## The bay tag's yaw was a coin flip (issue #88): fit the layout, not the tag

A square planar tag viewed near head-on has two PnP solutions mirrored about
its normal, nearly equal in reprojection error and chosen by pixel noise: a
bay tag's decoded yaw read about ±6–7° or near zero, never between, while its
translation held under a millimetre -- 5.5 cm of lateral error at the
rover's 0.45 m reach. A median of looks cannot help (the render is
deterministic, so N looks at one pose are one look), nor a sign from the
belief (the magnitude is wrong too). **What is true now:** a facing comes off
several tags' TRANSLATIONS fitted to their known layout by a 2D Kabsch fit,
the baseline between them rather than one tag's foreshortening: the dock's
board (`legs.dock.fit_dock`), the rack (`legs.rack.fit_rack`) and a row of
plate signs (`mapping/places.py`; "Places, not coordinates"). ⚠ The layout is
FACES, consistently: one plate centre mixed in put a steady +0.46° bias on a
two-tag fit (half a plate's thickness over the baseline; a differential error
is a rotation, a common one is swallowed by the translation).

## The squaring-up loop had no floor (issues #108, #339): every terminal loop has a budget

The pen's square-up at a board ended in `while |heading error| > tol: turn`,
with no timeout; ridden up onto the board's mount, wheels half off the floor,
the rover never got inside tolerance, drained its pack to 0 % and ran 700 s
past the day's budget. Nothing else caught it because every guard sits
BETWEEN errands, and an empty pack does not stop the body. Later a drive back
to a bay standoff looped `while` it was more than 5 cm away: toppled, the
robot could never arrive, the loop outlived the death, and every stand-up
drove it back into a wall until it was flat (found by walking the day
routine's generator chain, `gi_yieldfrom`). **What is true now:** every
terminal loop has a sim-time budget and an explicit answer
(`control.FACE_BUDGET_S`, the quadruped's `QuadMission.FACE_BUDGET_S`); a
bound is not a recovery (that is `reset_robot` and the `stuck` death); and a
premise test that could hang raises after twice the budget, because a
regression that hangs the suite is worse than one that fails it.

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
house, `legs/model.py`, the coupling rig), byte-identical at no cost to the
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
arithmetic), so a taken bay is waited for ("Two robots at one rack").

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
  (the D435's lens stands 1.5 mm proud); the quadruped's LIDAR site repeated
  it at the centre of its puck ("The map stays true under drift").
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
  as a delta. Nothing that decides reads the height map; the quadruped's
  planner reads the D435 through a layer of its own ("The first quadruped
  deploy").

## The second house and the loop (issue #215): the planner is the new cost

The home world grew from 26.5 × 12 m to 49 × 21 m -- a second house across
the street, a sidewalk band and a street loop round both, a fence round the
loop -- and the grid from 159,600 to 469,200 cells. The frontier mask stayed
linear (4.6 → 12.6 ms a pass); pure-Python A* did not (0.34 → 1.26 s a plan
across the loop), the first cost here that scales with the WORLD rather than
with what the robot does. **What is true now:** `occupancy_grid.MAX_CELLS`
carries the table, and the quadruped's walks plan on a compiled lattice
("Walking into the unknown").

The reserve's worst point is found BY ROUTE, not by straight line: the loop's
east legs are the farthest from home as the crow flies and among the nearest
as the robot goes, because the house is reached only through the middle
street's gate. `tests/test_world_budget.py` routes every zone over the
compiled world and asks which is farthest (a straight line would have sized
the reserve off the wrong corner by twenty metres); the reserve is
`legs.world.RESERVE_WH`. The lab's props arrived as geometry with no
behaviour behind them, by decision: a world change is one regime break, and
the issues that add behaviour find the world already holding what they need.

**A bigger world moved the cameras' near plane.** MuJoCo derives
`statistic.extent` from the geometry's bounding box and scales every camera's
near plane by it (`visual.map.znear` × extent). The loop took the extent from
37.2 to 70 m and the near plane from 0.37 to 0.70 m, and the rover's dock
camera, 0.34 m from the rack at a bay, clipped the whole rack out of its
image: nothing raised, every pick ran on belief, and a blind stow put the pen
on the floor (the lights were suspected first and measured innocent). **What
is true now:** the generator writes `<statistic extent>` down
(`home.CAMERA_EXTENT_M`;
`test_home_world.py::test_the_camera_extent_is_written_down_not_derived`),
and a world with legs sets its own near plane (`legs.world.NEAR_M`, "The rack
at the arm's reach"). Any world whose bounding box grows must pin its extent,
or its cameras lose whatever they look at from closer than a hundredth of it.
⚠ **A pin in the XML does not survive a runtime `mj_setConst`**, which
re-derives the extent: a bench offer undid it for the rest of the process,
and the deployed pair's picks failed 13 of 15 with nothing raised.
`bench.set_unknown_mass` puts the statistic back, hands `mj_setConst` a
scratch MjData (it writes `qpos0` into whatever data it is given) and sets
the mass on the spec too, which the workshop's recompile rebuilds from;
anything else that calls `mj_setConst` on a live model must do the same.

## A goal out of sight is aimed at through the nearest wall (issues #215, #298)

The planner answered an UNMAPPED goal by aiming at the known-free cell
nearest it in a straight line. From the loop's north-west corner that cell
was indoors, behind the first house's north wall, and the only way there was
back round the loop: a plan that began by driving AWAY from the goal, which
the drive's stagnation check rightly ended. Near a doorway (#298) the nearest
cell was a one-cell island at the ragged edge of the wedge of free cells a
LIDAR paints through a door, joined to nothing: every replan found it, and a
board in the bedroom went unreached for thirty hours of the deployed pair.
**What is true now:** the quadruped plans through the unknown at a price
("Walking into the unknown"), and needs a stand-in only for a goal INSIDE a
wall's inflation, such as a board's use pose. The mapped-floor planner the
rover drove by -- a stand-in on the robot's OWN component, `scipy.ndimage.label`
over the traversable mask, 4-connected like `astar` -- is still
`Navigator`'s default, flown as `unknown_spike.py --before` and pinned as the
premise of
`test_unknown.py::test_a_goal_past_the_map_is_walked_to_by_the_door_it_has_seen`.

## ...and then the bed was standing in its way (#305)

The bed's corner sat 0.23 m from whiteboard_b's use pose, inside the front
stop's reach once the robot squared up, so the robot bounced off it until
the drive stagnated -- from some approach angles and not others, which is how
the board had ink on it and a history of failures at once. Gating the reflex
on forward motion was flown and changed nothing (reaching a pose 0.23 m from
an obstacle means driving at it); moving the furniture was the fix. **What
is true now:** a board's use pose keeps the front stop's reach clear of
furniture (the walls and the board are exempt: the robot arrives facing them
on purpose); the story is at `home.BED_HALF`.

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
(`coupling.contact_pairs` / `touching`, a geom's id resolved once per model by
`coupling.geom_id`); `tests/test_contact_reads.py` shows the readers give the
loops' answers, and a new per-step check goes through them. The inflation is
one chamfer distance transform (`distance_transform_cdt`), which
`tests/test_frontier.py` pins identical to the iterated dilation.

## A robot on its side maps the sky (issue #339)

A toppled robot's map had its hall painted solid with floor-hit arcs and a
free fan running out through the living room's walls past the fence. The
LIDAR casts along its real frame, and no return is "free to max range", so on
its side half its rays see the sky and go into the map as 8 m of free space
at the believed, upright bearings (5 s on its side painted 38 622 free cells
onto an empty map, through the walls). A dead robot keeps scanning, and a
stand-up keeps the map. **What is true now:** a reading goes into the map
only while `Body.level()` -- on the quadruped, standing with its IMU's tilt
within `QuadMission.LEVEL_TILT` -- and the near-field height map takes the
same gate; the front stop still reads every scan. ⚠ The corrupt map was not
the death loop it was found beside: that loop reproduced with the gate in
place ("The squaring-up loop had no floor"). Trust the reproduction over the
story that fits.

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

**What is true now:** after a swap, a charge or a failed pick, a robot with
somebody else in the world leaves the rack before it decides
(`_leave_rack_routine`) and checks that it got there (`RACK_CLEAR_M`,
`CLEAR_SPOTS`; History says where it ended); one that stays within the
rack's reach for `RACK_LINGER_S` doing nothing is logged `RACK: lingering`. A
taken bay, the charge bay included, is waited for
(`HubLifecycle._await_bay_routine`) for 3× the typical occupancy of what holds
it (`SWAP_OCCUPANCY_S`, `CHARGE_OCCUPANCY_S`: the rover's measured figures);
a failed return is tried again `STOW_RETRIES` times before anything but a
charge; and a failed charge approach logs `charge_trace`.

## A drawing that set off from the rack (issue #347)

A procedure's `draw` drove straight at its board from the rack with no
planner under it, and was knocked over six times of six on the deployed
pair; the carrying pose the issue blamed was not the cause. **What is true
now:** a verb that moves the body walks by the planner, and puts the tool in
the body's carrying pose first (`steps.run_verb`, `steps.travel_pose`) -- a
rule of its own; and a module lost for `LOST_TOOL_S` goes back to its bay by
itself.

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
front stop and the bumper still see it. A drive looks at who is lying down
every `DOWN_CHECK_S` and replans at once when that changes (without it, a
robot knocked flat just after a replan was met only by the front stop). Once
stood up it is avoided where it says it is again. A line about a robot on the
floor says it is "lying knocked over" (`posture`), never "standing";
`tests/test_downed_peer.py` pins each rule.

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
PluggyPlan's principles rule out. The rover's parking brake was the
precedent: a body reflex that decides nothing the mind decides. The reflex
lies down after `T_REST_S` without a motion command (the break-even, ~9 s)
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

**The served loop** is #385's to measure and move ("The served sim's
speed" below).

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
serves the pair, so both robots' maps share its frame. On legs
`rack_prior` IS the dock's commissioned pose, and #387's `anchor_at_dock`
snaps the legs' estimate to what the board says. Where the dock stands in the
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

**Which tools survive** (rebuilt on the longer peg, #405–#407):
- **the pen** — it keeps its sideways carriage; the arm gives it a board's
  full height from one stance (0.19–0.41 m from 0.43 m out) and the pressing
  force, reacted by the pad;
- **the claw** — the arm reaches the floor 0.40–0.50 m ahead with the
  rover's 154 mm pendant; built, it hangs its jaws 175 mm under the peg, so
  its crossbar clears its bay's tags ("The claw on legs"); holding the
  bench's 0.4 kg cube it weighs 0.61 kg, carried seated as the envelope's
  0.60 kg rows were, and held straight out at 39 % of the motors' rating;
- **the LCD** — unchanged but for the peg (the seed dispenser is retired:
  no job ever used it, #407);
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
and the fold into the served body, and the rack, the tools and the
carried-tool filter into the served world; #407 made the envelope the
workshop's validator ("The workshop on legs").

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
`np.isin` a call (and the rover's charge pins' criterion's), now the few
rows that hold the geom, checked in Python; the pack's scalar `np.clip`. **Not
cheap, and deliberate:** the physics; the depth camera, whose 8 400 rays
at 10 Hz is the sensor as specified (Parts.md) — halving its rate or its
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
(`scripts/drift_spike.py`, whose rover mode left with the rover). The rover
flew a home day of three lab round trips -- the mouse's feed in the lab,
then a carry at the rack -- on each tree:

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
rover's was (no mass moved).

**What is true now.** Every level scan is matched before it is fused and
the matched pose is the belief (`Navigator._match`); the dock's anchor
stays beside it ("Drift hygiene"). A match costs
0.67 ms on the dev machine (median 0.46, p99 7.2 -- the field's refresh, 5
ms every 20 fused scans or 1 m, and the rare search): 6.7 ms of a sim
second per robot at 10 Hz. Fusing only the scans after the robot moved
took the grid's updates on a recorded day of lab trips from every level
scan to 6 918 of 10 947, and the mapping as a whole from ~21 to ~20 ms a
sim second; a day that stands still less saves less. The matcher is
deterministic (einsum's own loops, no BLAS product) and its field is kept
state: a scripted rover day hashed IDENTICAL in two processes (2 025 state
samples, `determinism_spike.py`), and saved at t = 419 s and carried on
in a new process it was IDENTICAL after the restore (1 186 samples,
`--resume-at 400`). Not done: a scan
from a body that is not `level()` is neither matched nor fused, so drift
while it is not level runs free until it is and the search finds it (a
lying quadruped's heading is #425's, "Lying still, the heading holds"); a
robot lost beyond the search's window searches wider after a run of
refusals (#422, "Lost after a long explore, and found again"); and the
first trip through new territory carries its own drift into the map it
lays (0.2-0.3 m by the lab, ~1 m round the street loop), which later
visits match to but do not correct (#381's loop closure).

## The first quadruped deploy (issue #387)

The body #377 sized, walking on its policy and getting up on its own, put
into the house the rover lived in (`legs/world.py`: the quadruped -- or the
served pair -- and #378's dock put into `models/home_world.xml`, the dock's
board against the living room's south wall at x 3.5). It is a `Body`
(`legs/body.py`) over `QuadMission`, a `Navigator`: the map, the planner,
the drive and the peer rules the rover had, moved out of its mission into
`navigator.py` whole (a scripted rover day of 1012 sim s hashed IDENTICAL
across the move).

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
  (#378), so a small command is raised to 0.25 m/s -- and the driving law
  (`drive_toward`) crawls while it turns, so raised, the crawl swept 0.25 m
  arcs into the walls it was turning away from. A crawl with a turn to
  make is a pivot (`command_for`).
- *A detour read as a stall.* The rover's drive gave up after 10 s
  without getting closer to its goal in a straight line (`Navigator`'s
  default still does); from the bedroom's corner the only way to the hall
  walks 2.5 m round the divider first, and every such drive "stalled". A
  quadruped's progress is read along its route
  (`Navigator.PROGRESS_ALONG_ROUTE`).
- *A press reversed into the far wall.* The thighs scraped the kitchen
  counter's end walking past it; reversing replanned the same scrape until
  the hind knees met the north wall. A press on a flank steps SIDEWAYS
  away (`retreat_from`), one on the nose backs off, one on the hind knees
  steps forward. (The press itself had first fired on the floor: the
  house's floors are boxes named `*_floor_geom`.)
- *The explore gave up on reachable floor.* The rover's explorer A*ed the
  20 nearest frontiers and blacklisted each failure; the depth layer's
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
the body settles under the day's next command. A stand-up also never
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
(`overseer.mortal_rule`); the rules and the constitutions are written in
the quadruped's words since #427, which removed the swap.

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
`explore(zone)` walks with `ZONE_PATIENCE_S` (300 s: all 24 fit). Its goal
is the zone's middle, which can be furniture, where the spike's was the
nearest clear floor; a walk that ends in its zone has got there (#454). A
walk asks the robot's own hazard rows every second (`go_to_routine(stop=)`)
and ends where it stands on the answer -- `DRIVE_STOPPED`, never one of the
four ways a drive gives up -- and a procedure it stops is `stopped:
interrupted`, as between two verbs.

**Found on the way: no procedure on legs had walked a step.** Every verb
that moves takes the carrying pose first (#347), which asked the body for
its arm, and the quadruped has none: `drive_to`, `face` and `drive` raised
before they moved, on every run since the first quadruped deploy. The
observatory's two days on `5c6e6cf`: 6 565 procedures validated, 6 562
aborted, 3 ran; of the last 196 aborts read, 193 were one procedure run
again and again at one sim instant, the loop spinning with the world stopped
(#400: an action that takes no sim time can be fired again at once).

**What is true now.** The quadruped walks into the unknown (`OPTIMISTIC`;
the straight-line stand-in is `Navigator`'s default). The quadruped pair's
day hashes IDENTICAL in two processes (1 046
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
restart then forwarded the saved world at its instant, where a running
world's step reads the kinematics one step old, so a controller reading
them on every step parted the two worlds at the first step back. A restore
steps the last step again now (#420, Webserver.md "A restart is a
continuation"), and forwards fresh only where it cannot. The
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
and sensors (`axes.BODY_AXES`; `world_facts` gives each body its own), and
`move` is a legs verb. A move stands a lying body first and
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
driver's goal, never the torque -- and a death's `setpoints` read that.

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

## The rack at the arm's reach, and the swap (issue #405)

The quadruped's world has its own rack (`legs/world.py`): #378's rack on
the living room's south wall beside the dock (`RACK_X` 2.1; its board spans
x 1.6..2.6, the dock is at 3.5), its three bays holding the LCD (A, the east
one), the pen (B) and the claw (C). The LCD is a plate, a peg and its
screen, its mass on the plate (the envelope's "on its peg" case); the pen
and the claw are built ("Drawing on legs", "The claw on legs", below). The tools keep the rover's module names, and
the rack's parts its bay letters (`bay<letter>_tray_...`), so the rack view,
the lost-tool clock, a program's `fetch` and an admin's reset read them
unchanged (`coupling.bay_switches` reads the three bays). The lifecycle
names a bay by its `coupling.STATION_YS` entry; here it is that entry's
index (`swap.bay_of`).

The swap is #378's spike's approach as the served body's routines
(`legs/swap.py`, a mixin of `QuadMission`): the walk to a metre behind the
bay's working pose, a look, the walk in steering by the tags, the settle,
the measurement, the fork in under the peg, up and out, the tool to the
carry pose over the nose; hanging it back is the reverse, with a second
settle and look after the fork comes down from the carry pose. A program
on `autonomous` runs it as `fetch` and `stow` (`world_config`'s `swap`:
`SWAP_VERBS`); the boards' jobs take the pen (#406), the census the LCD and
the challenges the claw (#407).

**Four things the spike's world could not show:**

- **The near plane.** The house pins its extent at 37.2 m for its cameras
  (`home.CAMERA_EXTENT_M`), and MuJoCo's near plane is a hundredth of it:
  0.37 m. From a bay's working pose its tags are 0.31 m from the nose
  camera, so the robot found the rack from its approach a metre off and no
  bay once it stood at one -- every walk-in ended "no fit at the bay". A
  world with legs sets the near plane to 0.10 m (`legs.world.NEAR_M`: a
  Camera Module 3 focuses from 10 cm).
- **The settle turns one way.** The walk-in stopped within a few mm and
  about 1° of square, and in the 3 s settle the body stayed within 4 mm but
  turned counter-clockwise every time, +1.2..+2.7°: at the peg, 5-20 mm
  across, and 3 of 6 walk-ins failed the 15 mm gate (every try from the
  kitchen). The walk-in now stops aimed `rack.SETTLE_DRIFT` (1.9°)
  clockwise; the policy barely turns that finely, but the stop lands a few
  mm to the other side, and after the settle 12 of 12 walk-ins at bays A
  and C were inside the gate (across -6.9..+6.1 mm; before, -20.3..-5.5).
- **Arriving is not the verdict.** At the bottom of a put the tray takes
  the tool's weight an instant before the feed-forward lets go of it, and
  the arm stood 0.016 rad off its goal. Judged by the arm's arrival, a tool
  already hung counted as a failed put, the fork stayed under it, and the
  swap lifted it straight back off: every pen and claw stow failed (the
  LCD, lighter, scraped under the tolerance). A fork move fails only when
  it is out of reach; what happened is the world's (seated and
  conducting, or hung).
- **A tool compiled 0.3 mm up** (the spike's settle) read not hung until
  the world's first steps, so a first rack view would have said the tools
  were off their bays; compiled exactly at rest they read hung at load and
  move 0.008 mm in the first second.

**A restart's save waits out the part of a swap AT the bay**
(`Body.working`, read by `continuation.Keeper.busy` as a stand-up is): a
fork half under a peg is nothing a file holds. The walk there does not:
held off through a 45 s walk across the house, a stop outlasted Docker's
10 s grace. A stop at the bay waits too, so a process killed first comes
back from the save before the swap reached it.

**A carried tool is the body's own to its senses** (`QuadMission.carry`):
out of the LIDAR's map and the depth camera's cloud, as its own geoms are,
ignored by the press, and excluded from a stairs policy's height scan
(`PolicyDriver.scan_exclude`: it read as a 0.4 m obstacle over the nose,
#378). Walking and lying down to rest keep it at the carry pose -- folded,
the arm would swing it into its own back -- and a fall lets go of it, as
does a tool out of the fork's reach (`swap.NEAR_FORK_M`): knocked off by the
other robot's at the rack, or sent home by an admin, it had left the body
walking with its arm up, turning slowly and blind to the tool, until a fall
or its next swap. A fork move stops when the body goes down under it: the
fall folds the arm, and re-aimed along its line the fork was held out
through the get-up. A tool a pick leaves on the fork unseated is carried
too: left unclaimed, a
tool lying on the fork made every step of a walk a press, and a 1 m walk
backed off to 2.97 m from its start. ⚠ Only a CARRIED tool is the body's
own to the bumper: ignoring every tool against the fork, the pair's
carried tools met each other's forks at the rack and were knocked off (4
of 12 hung back), where the bump backs the two robots apart. Its power is
read every step for the module the lifecycle watches, so a tool away from
the fork reads no contact (`swap.NEAR_FORK_M`): 1.5 us a step, where the
lookups by name cost 20-32.

**Fetched and hung back, the served body in the house** (`arm_spike.py
--served`; a fresh map every flight, the tools in turn):

| from | fetched | hung back | fetch, median | stow, median |
|---|---|---|---|---|
| the dock's approach, jittered 0.1 m / 15° | 10 of 10 | 10 of 10 | 25.6 s | 38.4 s |
| across the house (kitchen, workshop, hall, bedroom, the gardens, the living room) | 10 of 10 | 10 of 10 | 45.4 s | 39.1 s |
| the served pair at once, each from its start, bays A and C (0.6 m apart) | 20 of 20 | 19 of 20 | 38.7 s (the slower) | 50.2 s (the slower) |

A stow takes longer than a fetch because a carried tool turns at
`W_CARRY` (below); flown before that cap, the stows took 30-31 s. Two
house flights first started from the rover's workshop spawn, which is
inside the workshop's table: set down on it the body fell and found no
route, and from a clear spot in the workshop both swaps landed.

**Two robots carrying at the rack can knock each other's tools off.** The
pair's one loss: walking back to their standoffs, the two met face to face
0.8 m apart, and one's carried LCD struck the other's carried claw -- a
contact neither bumper reads, since each tool is its own robot's. The
code before the review loses the same flight the same way. The bump is
what keeps a carried tool off the other robot's FORK: with every tool
against the fork ignored (a review fix, withdrawn), both robots' tools
were knocked off in 4 of 6 flights. Making a carried tool a limb to its
own robot's bumper as well landed that flight and lost three tools in two
others (17 of 20): both robots flinched at once, bumped and backed off in
turn until the tools fell -- withdrawn too, and settled by #418's wait
("Two robots at one rack, on legs", below). At the
carry pose the fork is the body's collidable front (0.418 m ahead of the
centre; the tools' plates 0.411, their faces visual only), past the front
stop's 0.38 m: a carrying body walked at a wall is answered by the bump.

**Carried through the house** (`arm_spike.py --served --carry`: each tool
in turn fetched, walked to the hall, the kitchen, back and out to the
garden by the east door, trotted at 0.8 m/s, turned, sidestepped and
stopped, then hung back; the coupling's criterion every step): **6 of 6
rode to the end and were hung back.** The coupling opened for up to 160 ms
a flight (its holding capacitor is 200), and every long opening was a
PIVOT at the drive's full 1.0 rad/s: the tool swings out of its V's. So a
body carrying turns at most `legs.body.W_CARRY`:

| the turn, carrying | the coupling's longest opening |
|---|---|
| 1.0 rad/s (the drive's full rate) | 160 ms (110-160 over six flights) |
| 0.6 rad/s | 56 ms |
| 0.45 rad/s (`W_CARRY`) | 14 ms; over six flights 4-14, and 82 in one |

The house has no stairs yet (#280): a flight carried is #378's table
(above), one descent in about 60 floating the peg at the first step.

**What is true now:** the served quadruped fetches, carries and stows the
three tools on its own rack from a program; `tests/test_quad_rack.py` pins
each rule above, and the whole swap flies behind `--endurance`. Not done:
a carried tool down the house's stairs (#280 builds
them); the other robot's carried tool is not filtered from this one's
senses (at the carry pose it is above the LIDAR's plane), and nothing
keeps two carrying robots' tools apart at the rack (1 of 20 knocked off at
bays A and C, 0.60 m apart: outside the 0.55 m #418's wait covers).

## The feed on legs (issue #403)

The paid feed on legs is #419's two verbs: `find` the feed plate's sign
round the lab's address, then `press` off it (`lifecycle.cage_program`).
Nothing the job carries is finer than the building, and every pad the robot
has seen is a wall to its planner, so no walk crosses one whatever the
map's drift -- the failure a handed coordinate had (#419's section, "A
handed coordinate carries the frame's error").

**One press a feed.** The pad stands 21 mm tall, under the depth camera's
floor line (30 mm), and a foot on it is not a bump (`_press_now` leaves the
feet out). `press` stands the front feet inside the pad, holds and backs
out: one rising edge, where the rover's pass through the pad was two or
more. `landed` is still a count, and "did it land" is `> 0`.

**Company is not a `care` act on legs** (`Menu.care_acts`): it is a spot
beside the cage no tag marks. The robot's own procedure may walk there.

**Measured** (`scripts/solve.py --feature mouse --pair`:
the pair as deployed, arm aboard, the paid job offered, claimed with a
prediction, found, pressed and graded by `eval_feed`, the other robot
standing in the hall):

| from | paid | sim s each | presses of another plate | belief off at the end |
|---|---|---|---|---|
| the dock (the first a fresh map's find, 207 s; walked back to it between) | 10 / 10 | 73-75 | 0 | 0.12 m |
| inside the lab (after one find from the hall, a fresh map) | 10 / 10 | ~21 | 0 | 0.21 m |

⚠ The pair's robots look round their starts in ONE loop
(`solve.start`): with #405's arm a quadruped that is not stepped lets
go of the arm's hold, and Luca stood 0.58 m from its belief while Rowan
looked round alone.

**After an explore the find walks to the floor it has not looked at.**
#419's search took any frontier within 7.5 m of the address before a point
it had not looked over. After an explore the lab's floor was mapped -- the
LIDAR's plane (0.51 m) is over the plates, the cage and the signs, and it
sees the room through its door -- so the only frontiers near the facility's
address (in its storeroom) were outside the building, near in a straight
line and a long walk by the way in. Flown, the robot stood in the lab's
doorway facing in, the feed sign 36 deg off its nose (the camera's
half-field is 33.5), and turned away to search round the outside for
600 s; three finds of three failed. And a look-around counted floor within
3.5 m looked over through walls. Now the robot's own floor comes first --
its frontiers and the floor it has mapped and not looked over IN SIGHT --
by the WALK from the address (`next_viewpoint`, `_search_map`: one
Dijkstra over the planner's lattice), then the lattice in the unknown.

**...and the press walks in on its axis.** The first fix's flights found a
press that crossed the next plate: a sign first seen from far off its face
knows its facing only by where it was seen from (up to 70 deg out), the
look at the standoff fitted the row, and the walk in -- steered straight at
the pad, no planner under it -- started from the old standoff, 15 deg off
the axis and over the shock pad (4 presses). A look that moves the
standoff more than `STANDOFF_MOVED_M` (0.25 m) now sends the press there
first, over the planner with every seen pad kept out.

**A failed press says which try failed, and why** (#439). Live, all 24
failed presses said "ran out of time" after 85 s. The first walk to the
standoff is handed all of `PRESS_PATIENCE_S` but `FINAL_S` (34.8 s), so a
walk that used it left the second try nothing, and that try's "out of
time" overwrote the walk's own cause. A press now reports the last try
that ran: the walk, in #350's words, or its sign not in view from in front
of the plate and how its look round ended. That reason leads the job's
verdict (`_program_failure`). Every try goes in the step's `trace`, the
log's alone: the walk's record, where the belief stood and its error
against the truth, and the time too short for the next.

**A second try is tried only where something changed, never the same
walk there again** (#439's part 3). The question was whether the patience
should let both tries run. On 8a61ada, the first build to log them, they
always had: every one of 16 failed presses ran two, and every second try
walked from where the first had stopped to the same standoff (within
14 mm), stopped within 32 mm of where the first had, and gave up as it
had, 0-19 s later. What stopped both was the other robot lying by the
standoff (#455, and the goal under it, below). So a walk there that gave
up, or a robot that stays across the way in, ends the press with that
try's words. So does a sign out of view, unless the look round moved the
standoff (a neighbour's sign fitting the row's facing), which is walked to
once more. A walk in that put no foot on the pad is walked in again from
the standoff, after a fresh look, as the dock, the rack, the claw and the
boards retry theirs. No live press has ever failed either way. Every
press, pressed or not, now leaves its tries
in the log, so the next week can say whether the second walk in is ever
used: a feed whose press took 67 s and stepped on the shock plate 22 times
on the way left no record of how.

Flown on it (`scripts/places_spike.py --find --n 8 --error 3 --again`;
explore-then-find flights of one robot from its start):

| | found | find, s | shock presses |
|---|---|---|---|
| #419's 8 addresses, a fresh map | 8 / 8, and 8 / 8 again | 62-225, median 148 (#419's search: 94); again 68-76 | 0 |
| after a 240 / 300 / 360 s explore | 3 / 3, and pressed | 198 / 200 / 279 | 0 |
| after a 480 s explore | 0 / 1 | ran out, 300 | 0 |

The price is the fresh map's median, 148 s against 94: from the south of
the facility the walk takes in more of the building's floor before the
lab's door. After 480 s of exploring the belief was 1.4 m off the truth and
the robot then failed to dock: no find works through that, and the drift
is the map's, not the search's (below). Found once, the place is
remembered, and a feed from the dock is 73-75 s and 1.77 Wh
(`economy/energy.json`), about 60 mWh a metre walked.

⚠ **After a long explore the armed robot's belief drifts past a metre**,
and a place search cannot work through that, nor the dock. One quadruped
from its start, exploring 480 sim s, belief against truth every minute
(`--world home_quad`, the same script on both builds):

| sim s | before #405's arm (`e5f66d5`) | with it |
|---|---|---|
| 240 | 0.04 m | 0.12 m |
| 360 | 0.26 m | 0.59 m |
| 420 | 0.45 m | 0.60 m |
| 480 | 0.21 m | 1.40 m |
| then to the dock | docked | failed to dock |

One flight a side, and their paths differ (the rack and the arm change the
world), so it is a measurement, not a cause (#422): the scans
were accepted throughout, so the error is in the frame the map was laid in.

## Places, not coordinates (issue #419)

A job used to be handed where things are. On legs it is handed the
building's address, a few metres off, and the task area's written
directions; the robot finds the place by a tag on it, remembers where it saw
it in its own map, and walks the last metre and a half steered by the tag.
`scripts/places_spike.py` flies it; `tests/test_places.py` pins each rule.

**A handed coordinate carries the frame's error.** #414 handed the plates'
world positions to the feed job: after an explore with the arm aboard the
belief reached the lab 0.43-0.45 m off, and the walk home from the feed's
approach point pressed the shock plate five times. A place found by sight
is laid in the frame the map and the belief are, so the drift is in all
three and cancels.

**A sign, not the floor.** The nose camera sits 0.367 m up, level, its
vertical field 41 deg, decoded at 1280x720 (`--tags`, `--flat`):

- a 120 mm tag at the camera's height, facing the room, decodes square-on
  past 5 m, 55 deg off its face to 5 m, 70 deg off to 1.5 m; positions to
  median 1.2 mm, worst 57 mm;
- the same tag flat on the pad, 320 mm across, decoded in 2 cells of 77
  (1.5 m, 40 and 55 deg): the floor is seen at a graze.

So each plate has a sign at its far edge (`cage.plate_signs_xml`): a post,
a white board and the tag at 0.37 m, its face 0.45 m past the pad's centre.
Pressing, the torso stands 0.10 m short of the pad's centre
(`PRESS_BACK_M`): the front feet 0.09 m inside the pad, the hind feet off
it, the folded fork 0.2 m short of the sign, the camera 0.33 m from the tag
and still reading it. The signs are `legs/world.py`'s, put in beside the
house's lab.

**One tag's rotation is a coin flip square-on.** Its reported yaw was up to
11.6 deg wrong square-on (2 of 10 looks past 5), and 20 deg off it once came
out mirrored (10.8 deg wrong); from 30 deg off it was within 0.6 deg at every
range -- but one look 27 deg off at 4.1 m read 38. So a place's facing
comes off the row of signs fitted to its drawing when two are known (their
positions to a few mm, a metre apart), else a look at least 35 deg off and
within 3 m, else where it was seen from. The press needs the pad within
0.3 m of its axis and the neighbours are a metre away, so none of the three
misleads it far.

**The search follows the map, out from the address.** The first search
took its viewpoints nearest the robot and walked out of the facility, round
it and into the street: 600 s, not found. Taken round the address it found
the plate from all eight directions of a 3 m error (62-460 s) -- but with
the facility's own address, which lands in its storeroom a wall from the
lab, it failed twice from the dock: the lattice's first nine viewpoints
were the storeroom's and its walls'. Now (`next_viewpoint`): where the
row's drawing puts the sign once any of the row is read; else the address;
else, by the WALK from the address over its own map, a FRONTIER -- known
floor meeting the unknown -- or floor it has mapped and not yet looked over
with no wall between (the lab's, seen through its door); else a lattice
point in the unknown. The walk and the floor beside the frontiers are
#403's: after an explore the frontiers near the address were outside the
building (SimNotes, "The feed on legs").

**Flown** with #419's search (frontiers first, by the straight line), a
fresh robot from the living room's start unless said
(`--find --n 8 --error 3 --again`; `--find --again --from X`):

| address | found and pressed | find, s | again, remembered, s | shock presses |
|---|---|---|---|---|
| 3 m off the facility's middle, 8 directions | 8 / 8 | 62-272, median 94 | 67-76 | 0 |
| the facility's own (in the storeroom), from the dock | 1 / 1 | 123 | 64 | 0 |
| ...from the living room | 1 / 1 | 124 | 68 | 0 |
| ...from the hall | 1 / 1 | 157 | 81 | 0 |

Every press stopped 22 mm short of its pose (the policy's run-on,
`rack.STOP_M`), within 7 mm across and 2.1 deg of the axis; the belief was
0.07-0.15 m off at the end, no robot fell, and the toy plate was never
pressed either. "Again" is mostly the walk back across the street: from
the remembered place there is no search.

**Keeping off.** Every pad the robot knows is a wall to the planner
(`PlaceWalk.keep_out`, the corners' circle, then the planner's 0.35 m
inflation): from south of the row to between it and the cage the plan goes
round the row's ends, where without it the plan walks over the feed plate.

**The walking look costs a render**, 14-20 ms on the dev GPU under EGL and
32 ms on the box under osmesa: at the dock's four looks a second the served
pair would spend a quarter of the box on them. A look every metre walked or
45 deg turned, and none standing: MEASURED on a walk across the street, 43
looks in 86 sim s, one every 2 s -- 16 ms a sim second at the box's render,
1.6 % of real time per walking robot.

**Still open.** Loop closure (#381): the lab's frame is the one the map was
laid in. The address is a fixed offset, not a GPS sensor.

## Lying still, the heading holds (issue #425)

**What the live pair showed.** Over the last 40 runs of `5736e23` the pair
docked 0 of 17 times, and at 7 of the 8 deaths the belief was 4-22 m from
the truth, six of them more than 100 deg off in heading. The cause is the
rest. A scan is matched only while the body is `level()`, and the
quadruped is level only standing: lying, its LIDAR is at 0.29 m, not the
0.51 m plane its map was laid at. So while it lay, nothing corrected the
heading, and the heading integrated the gyro's offset (`imu.GYRO_BIAS`,
up to 0.05 deg/s; Luca's is -0.0438, 2.6 deg a minute). Past 6 deg the
matcher cannot recover: its search reaches ±0.6 m and ±6 deg round the
belief and refuses anything else as `inconsistent`. A robot that lay 2-5
minutes stood up lost for good, and everything it mapped from then on was
laid askew. A charge is a rest too: the dock re-anchors the belief when
the pins first conduct, and a charge from low lasts half an hour or more.
At Luca's offset, a robot that docked got up about 80 deg off.

**The fix: a zero angular-rate update** (`imu.Standstill`, read by
`LegOdometry`). A body lying still is not turning, so what its gyro reads
there is its offset. The body says when it rests: its posture is `lying`,
its drivers limp (`LegOdometry.resting`). The gyro must agree: its
reading, less the offset learned so far, smoothed over 0.2 s, must read
under 0.2 deg/s. While both hold, the heading integrates nothing, and the
reading is learned as the offset (a running average over 30 s). Otherwise,
walking included, the odometry integrates the reading less that offset.
The gyro gate is for a push. MEASURED, once `lying` begins the body turns
under 0.001 deg/s, so what trips it is the other robot shoving a lying
one, and that turn is integrated. A turn slower than the gate (a slide, a
slow shove) is missed, as it is by any zero-rate update, and learned as
offset. So the learned offset is bounded at twice the part's worst
(`BIAS_MAX`): unbounded, a 0.12 deg/s slide for 2 minutes taught 0.17
deg/s, and every walk until the next rest turned 7 deg a minute; bounded,
3.

Flown on #422's probe: Luca's draws, a look-around, six minutes lying,
then stand and turn four times.

| | before (staging `008299b`) | after |
|---|---|---|
| heading after lying 30 s | -0.97 deg | -0.16 deg |
| ...after lying 360 s | -15.4 deg | -0.16 deg |
| position after lying 360 s | 21 mm | 4 mm |
| stood and turned four times | -16.0 deg; 102 of 102 scans `inconsistent` | within 0.08 deg; 104 of 104 `ok` |

The 0.16 deg is the lie-down's own 3 s, integrated before the rest began,
with nothing learned yet. The first rest learns the offset (-0.0437 deg/s
against the part's -0.0438, after 150 s), and from then on it is taken off
every reading, the lie-downs' included. Thirty minutes lying, as long as a
charge from low, ended where it began, at -0.16 deg; unheld, that is about
79 deg at Luca's offset. Every scan after it matched, 99 of 99. On the dock
itself (a 279 s charge from 15 %), the body turns under 0.001 deg/s once
lying on the pins, the update holds on every step, and the heading ended
0.001 deg from where the dock's anchor put it; unheld, 12 deg.

**Why not also match while lying**, the issue's other option: the
standstill is enough, and a lying match has its own risk. At the start
pose, a lying scan matched against the standing map (applied to nothing,
fused into nothing) agreed every time, 358 of 358 over the half hour. But that was one room. At
0.29 m the plane sees the bed and the couch the standing map does not
hold, and a scan accepted there would move a body that is not moving.

**The maps kept before it are askew.** The deploy that carries this drops
them: `continuation.MAP_EPOCH` 1. A save of another epoch keeps the clock,
the packs, the deaths and the jobs, as a changed geometry does, but not the
bodies, the beliefs, the maps or the places found in them (#419).

**A restart mid-rest, flown for the first time.** #387's resume check
saved before the day's first lie-down, and a body lying moves nothing its
belief could change, so it could not see a learned offset go missing. The
check that can uses one quadruped on the demo pack at 60 %. It explores
until 253 s, lies until the pack runs low, walks to the dock at 713 s and
charges. Saved at 603 s while lying, the first restore diverged 1 s into
the stand-up, whether or not the offset came back. The cause was not the
offset. The scripted gait (`VirtualModel`) reads the body's inertia off
the mass matrix where it is first built, and the restarted process built
it from folded legs; that inertia is kept now too. Restored, the day is
IDENTICAL after the save over 852 samples: the rest, the stand-up, the
walk, the 272 s charge and the back-off. With the learned offset stripped
from the same save, it parts at 718 s, as the walk begins.

**What is true now.** Lying, the quadruped's heading holds at whatever
the lie-down left, 0.16 deg with nothing learned and less after. The
learned offset and the gait's inertia ride a restart. A robot already
lost now searches wider, and a long explore drifts as far with the offset
learned as without it (#422, "Lost after a long explore, and found
again"). Not done: a slow turn while lying (none measured on the floor or
the dock).

## Lost after a long explore, and found again (issue #422)

**What was measured.** One quadruped in `home_quad` explores for 480 sim
s from its commissioned start, then walks home and docks
(`scripts/drift_spike.py --explore 480 --seed K`), the belief against the
truth every 10 s with the zone it truly stands in. A seed draws the IMU's
offset and scale and the encoders' noise; eight a build:

| build | error the explore ended with | median | over 1 m | docked |
|---|---|---|---|---|
| before the arm (`e5f66d5`) | 0.10-1.61 m | 0.45 m | 2 of 8 | |
| staging (`db669ab`: the arm, #425) | 0.10-1.42 m | 0.50 m | 3 of 8 | 4 of 8 |
| this change | the same explores, to the bit | | | 7 of 8 |

The issue's one flight a side (0.21 m before the arm, 1.40 m after) was
the luck of one path: the arm is not the cause.

**Where it grows.** In the house the belief stays within 0.14 m. It grows
on the street loop and the long sidewalks: walls on one or two sides,
straight for 20 m, met at grazing angles (#401), every one laid by the
robot on the same walk. There a scan can only be matched against what
the robot laid seconds before, through the same drifting pose, so the
matcher holds the belief to its own recent map rather than to the world,
and 20-70 % of its matches keep a direction (the street's length)
odometry's. Split per match, the matches take off nearly all the heading
the gyro adds (seed 0: odometry -20.3 deg over the explore, the matches
+19.8); what is left goes into the map as it is laid.

**What it is not.**

- *The arm.* The legs' own odometry, their velocity estimate integrated
  through the TRUE heading on drift_spike's lab walk (47.8 m, the truth
  steering), reads -8.6 % along the walk and -2.0 % across it before the
  arm, -7.6 % and -0.1 % with it.
- *The gyro's offset.* Robots that lay two minutes first, the offset
  learned to 97 % (#425), ended 0.82, 1.26 and 1.34 m out. With it
  learned, the matches themselves add up to 3.4 deg outside, aligning to
  walls laid askew on the same walk.
- ⚠ *The legs' speed, found on the way.* Those -8 % are at drift_spike's
  0.5 m/s. At the explore's 0.35-0.4 m/s the legs read -0.5 %, at 0.45
  -3.2 %; with perfect sensors at 0.5, +0.6 %. A walk past 0.4 m/s is not
  measured by its own odometry; nothing here walks that fast.

**Why it failed to dock: the walk home.** In every failure the robot
never reached the standoff, and the dock itself was never tried. Seed 0
came home 0.9 m out. Crossing from its garden (laid drifted) into the
living room (mapped true at the start), its scans matched "ok" with
0.56-0.73 of their points on walls, the garden behind agreeing and the room
ahead not, and a match need only put half its points on walls to be laid
in. Two seconds of those painted an offset copy of the room; the field's
next refresh put the copy in it, the share jumped to 0.91, and every scan
after matched the copy and was laid in, erasing the room as first mapped
(the log-odds are clamped at ±5, so a wall is floor after 14 misses). The
standoff then lay inside the copy's couch, the planner aimed at a
stand-in, and the robot pressed the real couch, believing itself 0.2 m
short, until the walk gave up. Seed 2 came home 1.2 m out to a garden
its map held two ways at once: the house's east wall and its door where it
had first laid them, and the garden's east fence 0.45 m off, laid again
askew as it came back along the street. No one pose fitted both: fits
started 0.1 m from the truth put a third of the points on walls. It
walked into the house's east wall 1.35 m north of the door it believed it
stood in front of.

**The fix, in three pieces.**

- *Near enough, the board decides* (`lifecycle.NEAR_STANDOFF_M`, 0.75
  m). A walk to the charge standoff that gives up within it, by its own
  record, goes on to the approach, which finds the dock's board and walks
  in by it whatever the belief; with no board in sight from there (a wall
  between, a belief further off than it says), the walk's own retries go
  on as before. The standoff was only ever how the robot got to the
  neighbourhood (`go_charge_routine`); a walk that gave up 0.1-0.3 m short
  of it ended the charge. Replayed from the three saves that failed that
  way, all three docked on the first approach, from 0.89, 0.53 and 0.61 m
  off. 0.75 m keeps the board within 2.4 m of a robot that believes
  itself there, where one look reads it to 6 cm (median).
- *An anchor re-lays the map round it* (`scan_match.ANCHORED_SCANS`, 30).
  Docked, the board puts the belief in the dock's frame to millimetres;
  the next 30 scans laid in go in at that belief, unmatched, and
  overwrite a copy laid askew. Without it, one of those robots backed off
  the dock and was 0.62 m off a second later, matched back into the copy;
  with it, 0.00-0.01 m, and each of the four then walked to the lobby
  across the street (0.19-0.42 m off on arrival) and docked again. Only
  the BOARD anchors: the seat, with no decode, is good to 2 deg, and 30
  scans laid unmatched through that would lay the room askew themselves.
  Nor is the commissioned start an anchor (a look-around laid unmatched
  smears the gyro's scale error, up to 1.8 deg a turn, into the first
  map), and a stand-up there closes a window a death on the dock left.
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
  ten refusals on, finds the same correction. 72 ms on a house
  scan (the search it widens is 32 ms, on every refused scan); one in ten
  of a lost robot's scans. It did not fire in these flights: no lost
  robot here had an intact map under it. It is the issue's own ask, for a
  robot lost with its map whole (#425's lying drift was one), and the
  served body set down 1.5 m from its belief in its living room finds
  itself on the twentieth scan.

**What was tried and dropped: a scan laid in only where the map agrees.**
Fused only with three quarters of its points on walls, the living room
was not laid over, and seeds 0 and 1 docked. But on a sidewalk whose thin
wall the map had eroded (#401), it refused the scans that would have laid
the wall again, the share kept falling, and the wide search, then taking
one search's answer, jumped 1.8 and then 3.4 m along the sidewalk: two
robots that had docked were lost mid-explore. The map's confidence cannot
tell the cases apart either: an eroded wall reads -5, as sure as the
living room's floor. It went, and a relocation must now be found twice.

**What is true now.** A walk home that gives up within 0.75 m of the
standoff goes on to the board, and back to its retries if the board is not
in sight; docked by the board, the next 30 scans are laid at the dock's
belief; and a robot its map refuses ten scans running searches 2 m and 15
deg round its belief, taking a pose that no other place explains as well
and that it finds twice. Over the eight seeds the explores are unchanged, to the
bit, and seven dock where four did. Rested first, three of three dock, as
on staging. From two other starts, the hall and the garden, three of four
dock on both builds; the fourth ended its explore 0.36 m out on the west
sidewalk and paced it for 270 s beside the house's 4 cm west wall, which
its map had eroded nearly end to end: #401's shuttle, a gap the planner
kept routing through.

**Not done.** The drift itself: the map laid round the street loop is
still laid through a belief that drifts a metre, later visits match it and
do not correct it, and where parts laid through different drifts meet,
no pose fits (seed 2's garden). Correcting the map behind the robot is
#381's stage 3, a
pose graph, and this measurement says it is needed. Relocating by tags
away from the dock was not needed to dock. One look at the dock's board
puts the robot within 1.2 cm and 0.4 deg (median) from inside 1.5 m, but
up to 0.8 m and 11 deg off from past 2.5 m (the robot stood at 160 poses
round the dock, nothing stepped: 95 fits), so a far look could only
propose a pose for the matcher; and the plate signs are positions the
robot estimated itself, the pose graph's landmarks.

## A robot resting across the other's way (issue #415)

**What happened.** #405's fixture day stranded the first robot four times
in four, at the same doorway: a robot lying down by reflex is not down
(#365), so the other's drive found its way cut by the resting robot's disc
and gave up ("the walk gave up 8.2 m short after 9 s", then `stuck` at
7 %), and the resting robot never moved -- it stands only for a command
of its own, and a finished robot never gets one. Its words were wrong
too: "Rowan standing 3.5 m from where it was going", of a robot lying in a
doorway 3.8 m off. On today's code the fixture day's own trajectories no
longer meet there (the four battery levels all dock), so the case is built
instead: `scripts/make_way_spike.py` lays both maps from the true floor,
lies one robot in a doorway of the house and walks the other through it.

**The rule (Ben, 2026-09-30, #415's option 1).** A drive whose way a
RESTING robot cuts -- a plan without that robot's disc finds one -- asks it
off the way (`Navigator._ask_way`, over the pair's `Body.ask_way`), with
the way it would walk; a stagnated drive asks one resting near it or its
goal. It waits for it as for a robot that will move, `MAKE_WAY_WAIT_S`
(30 s) from the first yes at most, and then gives up as before. The asked
body says yes only lying down to rest and free to: not on the dock, not
mid lie-down, stand-up or arm move, not inside a walk of its own (#395's
head-on hold between two standing robots is that walk's, and still open),
not dead. Then, beneath whatever routine holds it, a STILL command is
replaced by the step aside (`legs/way.py`): stood up by the reflex's own
rule, it walks the drive's law along one plan to the nearest floor it has
SEEN, reached round the asker, `aside_clear_m` (0.85 m: the asker's 0.70 m
disc and three cells) off every point of the way -- the way the asker sent
is then open whatever it plans. Any command that moves takes over at
once; a fall ends it. It is said ("MAKE WAY ...", "MADE WAY ..."), and
remembered once over: the body moved and the mind did not decide it. A
restart's save waits it out.

**Measured** (`make_way_spike.py`, seven scenes, the pair in the house):

| scene | before: the walk | after: the walk | the step aside |
|---|---|---|---|
| living <-> hall door | gave up, 0 s | arrived, 29.9 s | 0.83 m, 6.7 s |
| ...the other way | gave up, 0 s | arrived, 30.5 s | 0.84 m, 6.7 s |
| living <-> bedroom door | gave up, 0 s | arrived, 22.1 s | 0.87 m, 8.9 s |
| hall <-> kitchen door | gave up, 0 s | arrived, 34.7 s | 0.82 m, 6.7 s |
| hall <-> workshop door | gave up, 0 s | arrived, 36.8 s | 0.84 m, 7.3 s |
| living <-> garden door | gave up, 0 s | arrived, 29.8 s | 0.86 m, 7.4 s |
| the middle of the hall | arrived, 19.9 s | arrived, 19.9 s | not asked |

No touch, no fall, in any. With the resting robot's day already over (its
routine returned, as the fixture's had), all seven alike: the step runs
beneath the command the loop holds a finished robot with.

**What is true now:** a robot resting across the other's way is asked to
make way and steps aside, beneath whatever it holds; a drive gives up at a
resting robot only once it said no or did not clear the way in 30 s; and
a drive's words call a robot resting "lying down to rest", "in the way"
unless its disc covers the goal (`tests/test_make_way.py`). Where the two
maps disagree, the next section.

## A robot resting where the other's map put it elsewhere (issue #455)

**What happened.** In the last 16 hours of 8a61ada's log, thirteen walks
gave up naming a robot lying down to rest: five charges, four presses,
two procedure walks and two explores, and two more press tries whose
verdict said "stalled". One had an ask beside it, though make-way worked
all 11 times it was asked. The `encounter` rows say what the log could
not. In eleven of the thirteen the two bodies were within 2 m while the
words put them 1.1-3.6 m apart. So were they in every press walk that
stalled 0.7-0.9 m short of the feed plate's standoff, both robots' (14 of
the 18 failed feeds), and in no successful press. Luca's map of the lab
sat 2.4-2.9 m and 9.5 deg off for hours (each press trace's `belief off`),
and its map of the house 1.1-1.7 m off at its deaths.

**The mechanism.** A robot resting was kept clear of where it SAID it
was: its belief, in its own map's frame, so two maps a few metres apart
put its disc a few metres from its body. The walker's planner routed round
a spot the robot was not at and straight at its body. The depth camera
held the walk there, because a robot lying down is under the LIDAR's plane
and only the camera sees it. The stalled drive then looked for a robot
near it or its goal, to wait on and ask, and by the belief there was none.
The give-up named the robot 3.4 m off, or said "stalled" if the hold had
lapsed that instant. Once an ask was made (Rowan's charge at the dock,
Luca's belief 1.1 m off its body), the way was sent in Rowan's frame and
laid in Luca's, where it ran 1.1 m clear of Luca: five times it stood up,
stepped 0.0 m and lay down again. A robot backed out of its own press
lies 0.25 m from the plate's standoff (press pose 0.55 m from the sign,
back-out 1.0 m, standoff 1.8 m), where the other robot walks next. Two
more: Luca's `unminded` death left it lying 0.68 m from Rowan, whose
charge then called it "lying down to rest" (#441 asks no dead robot); and
three walks to the workshop gave up at once, 10-25 m from Rowan walking
the hall, the one way in, its disc across the way at the moment of a plan.

**And the hold, wherever the maps agree.** Flown with the camera on, #415's
own doorways lost three walks of six: the robot asked stepped aside, lay
down again 0.8 m on, and the walker's new plan ran past it close enough for
the camera to hold. A robot resting in the middle of the hall, with the
way round it open, did the same to the walk at two placements of six a few
centimetres apart: held at the disc's edge facing it, while the two ways
round it swapped plan by plan and reset the stagnation clock, until the
walk timed out. It is #365's trap, the one that took the hold off a
fallen robot, and a resting robot placed where its body lies is that case:
planned round, and asked off a way it cuts.

**Measured** (`make_way_spike.py --near-field`, the depth camera on as the
served world flies it; `--drift 0,2.9` lays the resting robot's map and
belief 2.9 m off the world together):

| scene | before | after |
|---|---|---|
| living <-> hall door | stalled 4.4 m short, 22 s, not asked | arrived 26.9 s; aside 0.84 m |
| living <-> bedroom door | gave up 91 s, "lying down to rest in the way, 4.0 m off"; asked, moved 0.04 m | arrived 22.9 s; aside 0.88 m |
| hall <-> kitchen door | stalled 4.4 m short, 23 s, not asked | arrived 37.9 s; aside 0.81 m |
| living <-> garden door | stalled 3.3 m short, 23 s, not asked | arrived 36.0 s; aside 1.02 m |

With the WALKER's map and belief laid off instead, as Luca's lab was
(`--drift-walker 1.2,-2.9`): before, all four stalled 3.1-4.4 m short in
18-26 s, and nobody was asked; after, all four arrived in 22-38 s, the
resting robot 0.81-1.02 m aside each time. A robot STANDING in a doorway
10.6-13.9 m off that walks off 8 s later (`--standing 8`, the walker from
the east garden): before, all three walks gave up at once, "12.2 m short
after 0 s (Rowan in the way, 10.6 m off)"; after, all three arrived, in
39-55 s. #415's seven scenes, maps true: before, 4 of 7 (the three
doorways above lost), and the hall's middle at 4 of 6 placements; after,
7 of 7, and 6 of 6. No touch and no fall in any. Without the camera the
walker steps over a lying robot instead, with no contact, which is how
#415's scenes were first flown.

**...and the goal it lies on** (#439's part 2). The `encounter` rows put
the other robot within 2 m at 95 of 96 failed presses over three builds
(4f1288f, a85772d, 8a61ada). The 96th was a walk given up at once by a
map 3.4 m off. With the other robot near, 50-82 % of presses failed; with
it apart, 0-2 %. `scripts/press_spike.py` flies that geometry in the lab,
the pair on true-floor maps with the camera on. One robot lies where its
own press backs out (0.25 m short of the feed plate's standoff), on the
standoff, or across the way in from the door; the other walks in from the
door, from the south or from the east and finds and presses the feed
plate:

| code | maps true | the resting robot's 2.9 m off | the walker's 2.9 m off |
|---|---|---|---|
| deployed (5ba8a0c) | 8 / 8 (the door's and the south's), 21-53 s | 0 / 5: both tries stalled after 17-19 s and 14 s, nobody asked | 0 / 5, the same |
| with the change above | 13 / 14, 21-91 s; one walk ran out at 85 s | 5 / 5, 26-56 s | 4 / 11: seven walks circled for 85-87 s, asked late or never |
| with this | 14 / 14, 21-28 s | 5 / 5, 22-44 s | 11 / 11, 22-29 s |

The deployed code's failures are the live ones, down to the seconds: on
8a61ada a failed press's stalled walks gave up after 14-19 s, and its
second after 14-15 s. The eight that #455's change still lost were all a
goal under a disc. The standoff sat inside the
resting robot's disc, so every plan ended at a stand-in beside it, and only
a stagnation asked. None came. Its waypoints spent at the stand-in, the
drive steered at the goal and planned again from the disc's far side, the
stand-in swapped side to side, and every new route was progress. Whether
the swap only delayed the ask (to about 30 s, the 91 s press) or held it
off for good turned on where the goal fell against the planner's lattice:
the walker's map laid off moved it, and so did a start 0.8 m nearer. A
goal a resting robot lies on is a way it cuts now, asked at the plan that
finds it there, as one whose disc cuts the route is -- within `PAST_M` of
it: from further off the walk goes on, and the plan that gets it there
asks, so no walker stands waiting on a robot across a house. Every press
that pressed, on all three codes, pressed on its first try; none touched
another plate, and nothing touched or fell.

**What is true now:**

- a robot lying still, fallen, resting or dead, is kept clear of where
  its body lies, placed as the other's own sensors would put it
  (`HubLifecycle.keep_clear`), and the depth camera's hold skips it
  (`_near_field_step`); standing, it is where it says it is, and held for;
- a walk whose goal a resting robot lies on (`peer_on_the_goal`) asks it
  at the plan that finds it there, within `PAST_M` of it, and waits while
  it steps off (#439);
- the way it is asked off is told relative to its body (`make_way`:
  `Body.from_seen`, then `Body.as_seen`);
- a dead robot is never waited on, held for or asked, and is "lying dead";
- a robot further off than `PAST_M` (3 m) whose disc alone cuts a walk's
  way is planned past (`Navigator._plan_to`); nearer, it is the wall it was;
- a press's ask carries its walk in (`drive_to_routine(beyond=)`), and
  the walk in is cleared before it is walked (`clear_way_routine`), or the
  try ends "in the way";
- a robot that cannot make way says why, once every `MAKE_WAY_WAIT_S`.

`tests/test_resting_in_the_way.py` pins each rule.

## Hide and seek on legs (issue #404)

**What the rover's game was.** The hider drove to a surveyed spot and
waited; the seeker counted 20 s and drove to four surveyed points
(`HIDE_AND_SEEK_SPOTS`, coordinates neither robot ever found); a referee
keyed on the rover's `lidar` site and `chassis` geom called a find within
1.0 m with one ray from the seeker's LIDAR to the hider's torso. #419's
places rule takes the spots away, and on the quadruped the referee's names
do not exist.

**Who picks the spot: code, from the hider's own map** (decided here). The
mind does not see its map -- it sees its places (the lab's plate signs) and
its pose -- so a spot it named would be a coordinate it never saw; and a
role is code the way a `find`'s search is (#419: the errand searches). The
mind still decides whether to play, and by claiming first or second, which
role. The roles are two verbs of a game's program alone (`hide`, `seek`;
`steps.GAME_VERBS`), never the procedure language's.

- **The hider** (`legs/game.py`, `hiding_spot`): floor its map has SEEN, a
  walk it can make in the head start (0.3 m/s), 0.6 m off anything in its
  way, 1.5 m off the dock and the rack, farther than a find from where the
  seeker SAYS it counts (its reported pose, a network fact), and out of
  the seeker's sight from there wherever the map allows; of those, the
  seeker's longest walk, and of walks as long, its own shortest -- and
  never where its body would cut the seeker off from the house
  (`_cuts_off`). The seeker's planner keeps 0.70 m clear of the other
  robot on top of 0.35 m off walls, and MEASURED, a hider in the middle of
  a corridor sealed it at any width up to 2.2 m (the house's sidewalks are
  1.5 m), so a spot is taken only where the seeker's floor, the hider's
  disc taken out of it, keeps all but 2 m^2 of what it had; in the seven
  scenes below the first choice always passed. Nor does the hider step
  aside for the seeker while the game is on (#415's make-way is refused):
  stood up and walked out of the way, it would be the network handing the
  seeker the find.
- **The seeker** (`seek_routine`): counts where it stands, then searches
  its own map outward from there -- first the floor out of its own sight
  from where it counted, where a hider hides, then the rest -- ring by ring
  of its walk, to the nearest viewpoint (a lattice 2 m apart) its sight
  has not covered; each walk ends the moment its viewpoint is covered.
  Sight is the map's, both ways: what stands up -- the LIDAR's walls and
  the depth camera's furniture, never a floor plate the planner keeps off
  (a known plate's keep-out hid up to 11.7 m^2 of open garden). ⚠ Its
  choice of where to look never reads where the hider is, though the
  network carries every robot's reported pose: a test runs the search with
  the hider's disc on its planner and without, and the viewpoints are the
  same, in the same order. The walk under it keeps clear of the hider's
  reported pose, as every walk keeps clear of the other robot. A walk that
  ends without a step (no route, the other robot in the way) is followed
  by a 1 s pause before the next pick -- back to back at one instant, 225
  of them froze both robots for 9.9 s of wall clock and gave up all but
  five viewpoints -- and a viewpoint the other robot kept it from is tried
  again 20 s later; one the map gives up on stays given up.
- **The referee's eye** is the seeker's LIDAR on its rear mast, 0.51 m up,
  and the hider is every geom of its body. Re-keyed naively it would have
  been blind: from the mast the eye looks down across the seeker's own
  stowed arm, and to a robot lying 0.7-1.0 m in front of it every one of 41
  rays met the seeker first (standing 1.4 m off, 29 of them, the rover's
  one ray at the torso among them). So a ray that meets the seeker is cast
  on past it (a ray cast from inside a geom meets where it leaves, so each
  geom costs two casts). 41 rays and those casts cost 0.15-0.35 ms a check,
  so they are cast only within reach of a find, at most every 0.1 s.
- **The claims.** A role's claim queues nothing; once both are held the
  pair's referee queues each robot its role (`pair.referee_games`), so the
  robot that took the first role is free until the other takes the last --
  and is not shown the offer meanwhile: shown it, a standing order or a
  map row took it again and again and was refused every time. The claim
  goes in its History instead, and a decline of a held role is refused.
- ⚠ **The game starts once BOTH roles' errands have begun**
  (`HideAndSeek.begin`). The first claimant is often still busy -- a feed,
  an explore, a charge -- when the other takes the last role, and with the
  clock started by the first errand to begin, the seeker searched for a
  hider still charging at the dock, and a hider that never hid could win.
  Each role waits for the start, and the hider then picks its spot from
  where the seeker counts. The wait is a role's budget too (600 s at most),
  and the energy row does not carry it.
- **Called off, nobody paid**: when the two have not both begun 600 s after
  the roles were taken, and the step a player dies (the pair's `watch`) --
  a dead hider was never found and a dead seeker never sought. A true
  death gives back a role its robot took in a game still on offer. One
  referee a world, game after game; the wire names a role's claimant and a
  game's winner, not the last claimant.

**Balance, measured** (`solve.py --feature hide_and_seek --swap`: seven
start pairs about the house, both ways round, both maps laid from the true
floor):

| the seeker's search | seeking | the find | found |
|---|---|---|---|
| viewpoints 1 m apart, nearest first | 120 s | 1.5 m | 0 of 14 |
| ...out of its own sight first | 120 s | 1.5 m | 1 of 5 (stopped there) |
| ...out of its own sight first | 120 s | 2.0 m | 0 of 6 (stopped there) |
| 2 m apart, out of its own sight first | 240 s | 1.5 m | 7 of 14 |
| ...as it is now: begun by both, no spot that cuts the seeker off, the search's pause | 240 s | 1.5 m | 6 of 14 |

At 1 m the seeker stopped and turned every 4 s (28 viewpoints in 120 s),
covering the rooms round where it counted while the hider sat a 7.7-13.7 m
walk off. As it is now, at 2 m and 240 s, the finds came at 82-249 s;
where the hider won, the seeker passed 1.7-4.9 m from it. Every hider
found a spot out of sight and reached it in 10.3-20.5 s (median 17 s), so
the head start stays 20 s. Each game paid its winner 25 and the other
nothing. The seeker's role is the dearer, 6.0-6.1 Wh whenever the seeking
ran out (the hider's 0.7-1.7), and that is its row in `energy.json`. A
wider find did not help (2.0 m at 120 s found none of six): the search's
pace, not the find, was what lost, so the find stays two body lengths.

**What is true now:** the pair plays hide and seek on `home_quad` when
the cadence offers it to them on `autonomous`; the hider hides where its
own map says it is hidden, the seeker searches its own map without being
told, and the referee sees a quadruped -- any part of it, past its own body
(`tests/test_hide_and_seek.py`).

## Drawing on legs (issue #406)

The rover drew with its base braked and its lift as one axis
(`rover-final`): the square at 0.57 mm of form error, 98 % inked. The
quadruped draws with the pen on its arm (`tools/drawing.py`'s plotter, the
body's half in `legs/draw.py`; `scripts/draw_spike.py` flies every table
here): the pen module's own carriage across the board, the arm's fork up
and down it and into it, the pen's sprung quill setting the pressure.

**The pen module, rebuilt on the 220 mm peg** (`legs.rack.pen_face`). Its
carriage is the bill's slide, the Actuonix L12-100, at its whole 100 mm
stroke (±50 mm, 22 N, 25 mm/s): the rover's sim gave its pen 110 mm, and
the bill bought this slide 10 mm short. The figures fit it: a house is 72
mm wide fitted to the board, a two-digit answer 81. ⚠ **The module
balances on its peg.** Built as the rover's -- the carriage standing off in
front of the plate, 56 g 26 mm ahead -- its CoM sat 8.4 mm ahead of the peg,
and on the legs' rack, which asks a hung tool to be within 2° of plumb, it
hung 16° off and read not hung a second after load. The rail runs under
the plate now, 2 mm behind the peg: it hangs 0.6° off, its tip 48 mm ahead
of the peg and 58 mm under it, 36 mm short of the rack's back board (at the
rover's pen length, 2.5). Fetched and hung back from the dock's approach,
jittered, with a walk between: 6 of 6.

**The stance is lying.** A pen pressed on a board's middle and held 30 s
(`--sway`, where the walk in stops; three starts each, jittered 1 cm and
1°):

| stance | the ink point wandered | the torso drifted | its yaw |
|---|---|---|---|
| lying (the rest posture, legs limp) | 0.11-0.43 mm across, 0.04 up | 0.24-0.27 mm | 0.03° |
| standing on the walking policy | 0.35-1.0 mm across, 0.27-0.29 up | 4.8-5.2 mm | 0.01-0.04° |
| crouched / held on the scripted gait | fell (crouched at 0.27 m) / spun round | -- | -- |

The policy never quite stops (the rack's `SETTLE_DRIFT`): standing, the
torso backed straight off the board, 5.0-5.2 mm along its normal and
0.09 across, which the sprung quill takes up, so the ink point wandered
far less than the torso (placed 3.5 cm nearer, a first table had the
torso turning 0.34° and the ink point 3.6 mm across). Over a whole figure
it shows: a square standing came out at 0.86 mm, lying at 0.24 (below).
The scripted gait (`legs.scripted`) is a sizing instrument with no stance
controller to hold an arm held out. Lying costs 39 W less than standing,
and from the floor -- the torso's centre `belly_depth` up, 0.62 m out of
the wall the board's tags are on, 0.60 out of its face -- the arm reaches
the board's whole height, pressed and lifted, the carriage at either end
(`tests/test_drawing.py`). So the body walks in, lies down, draws, and
stands up.

**The squares** (`draw_spike.py --n 6`: from 1.7 m out of whiteboard_a's
face, jittered 0.2 m and 15°, the board's tags in its memory; the stats
off the WORLD's trace of the tip):

| stance | drew | form error, RMS | shape | inked | travel ink |
|---|---|---|---|---|---|
| lying | 6 of 6 | 0.22-0.25 mm (median 0.24) | 0.26-0.31 mm | 97.9 % | 0 |
| standing (the premise, `--stance stand`) | 3 of 3 | 0.86-0.90 mm (median 0.86) | 0.87-0.95 mm | 97.4 % | 0 |
| the rover (`rover-final`) | -- | 0.57 mm | 0.60 | 98 % | -- |

A two-digit answer ("42", three strokes, each pressed on afresh): 3 of 3,
form 0.41-0.43 mm, 98 % inked. A house on whiteboard_a, end to end from the
dock's standoff -- the board found, the pen fetched, drawn, the pen hung
back -- inked 7 of 7 strokes at 0.53 mm and paid 30 points.

**Ladder A on the pair** (`solve.py --feature answer --pair --n 4`: the
served pair, Luca from the dock while Rowan stands in the hall; a question
offered, claimed with its right answer, the job's errand run and graded by
`eval_answer` off the board's ink): **4 of 4 paid**, the boards in turn,
the ink 0.4-0.5 mm from the glyphs (the bar is 4), 164-201 s a job.

**Calibration reads no ground truth** (the rover's `calibrate()` read its
tip off the sim; PluggyPlan's "Hardware honesty"). The plotter steers by
the arm's encoders through its own kinematics, the slide's position, the
quill's Hall sensor and the nose camera. The board's tags -- two 120 mm
tags on the wall either side of it, level with its middle, 0.53 m apart --
put its middle and its facing from a metre or more out: one look from
1.0 m read its middle 1.9 mm across, its height 1.7 mm and its facing
0.03°. Lying, the camera is under their view, so the face is found by
TOUCH: the pen walks in at 5 mm/s at four points round the figure's middle
until the quill reads 0.5 mm, and a plane through the four touches is the
board in the arm's coordinates -- residuals 0.01-0.10 mm, the body's yaw
off the board read to a tenth of a degree, its face to a millimetre. (The
pen's point is its shaft's rounded end, `rack.PEN_TIP_R` past the tip's
site: counted from the site, every face read 2 mm short.) Nothing needs scaling: the
carriage is a lead screw with its own position sense, and the height is the
arm's kinematics. `tests/test_drawing.py` flies the plotter against a
perfect arm with the sim's records of the tip answering nonsense, and every
command comes out the same; the rover's per-stroke re-zero read the tip, so
it is gone, and lying the strokes need none (a house's seven strokes 0.53).

**What shaped it, each measured before it was believed:**
- **The lift off and the press on are no travel.** Recorded as travel
  rows, the pen's own way off the board counted as ink laid travelling:
  51 % of a one-stroke figure, past the evaluator's 25 %, and every drawing
  would have failed "the pen did not lift". Only the move ACROSS is a
  travel row.
- **Face the board's middle, not its facing.** A board found by one tag
  faces where that tag was seen from, up to 6° out: squared to that from
  0.1 m off its axis, the other tag was past the camera's edge and the
  approach ended "lost" (1 of 3). Facing the middle, both tags fall within
  ±21° from the look point.
- **The walk in is what lines the body up.** From a look point 1.0 m out
  (0.42 m of walk) one of six lay down 5 cm off the board's axis and 5.5°
  askew, the figure 47 mm off its middle; a two-digit answer has 9.5 mm a
  side to spare in the carriage's stroke. From 1.4 m (0.8 m of walk, the
  rack's is a metre) and with the rack's line-up gate (2 cm, 4°, else back
  out and in again), six of six lay within 6 mm, and the figure goes on the
  board's middle as far as the carriage lets it.
- **The board before the pen.** A body carrying a tool turns at `W_CARRY`,
  and a fresh robot searching the house with the pen aboard did not find
  the bedroom's board in 300 s; empty-handed, with the language's whole
  600 s, it did in 506 s (eight viewpoints round the house's address, in the
  hall). After the day's 240 s explore both boards were in its memory, and
  confirming one took 0-17 s. So a job finds its board first, then fetches
  the pen.
- **A walk's first turn is no stagnation.** After a drawing the rack is
  behind the robot, and carrying the pen it turns at most `W_CARRY`: the
  half-turn took 9 s of the 10 a walk may go without progress along its
  route, and the stow after the energy spike's second drawing on the
  living room's board gave up "stalled" as it finished turning, every time
  that day was flown (the loop's own return then hung the pen back). The navigator now starts that clock once the walk has
  faced its route (`navigator.AIMED_RAD`), for 10 s at most: a body that
  cannot turn stalls 20 s after it set out, never at its patience.
- **A fall ends the drawing.** It folds the arm and throws the pen; the
  plotter stops aiming the arm the step the body falls -- its fall counter
  moves -- or holds the pen no more (`PenPlotter.fell`), or it would hold
  the fork out through the get-up. ⚠ Not its posture alone (the review):
  a fall on the walk in was up and lying again by the stance, and it
  probed the board with an empty fork, reading the quill of the pen on the
  floor. So a drawing lies down only where it can draw -- the pen aboard,
  its re-face not lost, in time, not asked to stop -- and the interrupt is
  asked before it lies down and before every stroke. The drawing AT the
  board -- the walk in, lying, the strokes -- is `working`, as a swap at
  its bay is: a restart's save waits it out, and the boards' heights ride
  a restart with the places.
- **A job is graded on the ink IT laid.** Only the first ink erases a
  board, so a program's errand reads the board before it runs, as the
  rover's native errand did (`scoring.board_before`): read without it (the
  cage's reading alone), a job that inked nothing was paid for the drawing
  already up -- "correct, 0.0 mm from the glyphs", on the stub.

**Energy** (`energy_spike.py --actions draw:whiteboard_a,...`, each from
the dock with the board remembered, at the dearest figure its kind is
offered with -- `lifecycle.DEAREST_FIGURE`):

| board | draw (a house) | artwork (the robot) | answer ("88") |
|---|---|---|---|
| the living room's, `whiteboard_a` | 2.518 Wh, 187 s | 2.613, 195 | 2.448, 168 |
| the bedroom's, `whiteboard_b` | 3.397, 222 | 3.188, 217 | 3.315, 204 |

48-58 W over the job, against the explore's 87: most of a job is spent
lying down. The kinds' own figures (`TaskKind.estimate_wh`) carry the
bedroom's, the dearer.

**What is true now:** the served quadruped draws on both whiteboards from
a job: `draw_figure`, `rate_artwork` and `whiteboard_answer` are offered on
`home_quad`, each a program over the step vocabulary (find, fetch, draw,
stow; `lifecycle.draw_program`), and `draw` is a legs verb a procedure may
call where the world has boards, a rack and places. `tests/test_drawing.py`
pins each rule above; the flown tables are `draw_spike.py`'s.

## The claw on legs (issue #407)

The claw is rebuilt on the 220 mm peg (`legs.rack.claw_face`): two jaws
closed by one servo, an FS90MG through a rack and pinion (two position
actuators commanded as one), on a second L12-100 slide across the module
-- the body cannot sidestep millimetres, so the module does. Hung, the
jaws sit `CLAW_JAW_DROP` (0.175 m) under the peg: at 0.150 m the crossbar
hid the bay's tags from the working pose, and no fetch of the claw fitted
its bay.

**The eye.** A cube is 26 mm, its tag 21 mm. The nose camera, level at
0.36 m, sees no floor nearer than 1.1 m standing and decodes such a tag no
further than 0.8 m: it never sees a cube it could reach. The D435's colour
imager (`color_eye`, pitched 30 deg down with the depth camera; 1280 x 720,
the offscreen buffer's size) decodes one 0.5-0.9 m ahead standing and
0.38-0.68 m lying. A cube carries its tag on every face, so one render
decodes its top and its sides; keyed by id, the last decode won, and lying
in front of a cube that was its top seen edge-on, 16 mm over the face the
robot faced. Each decode is classed by its normal and the face seen most
square-on is read, its centre pixel cut at the face's known height -- the
floor under the robot, through its legs or its belly, and a layer is a
cube's edge -- never PnP's range: lying 0.58 m off, both faces put a cube
within 0.1 mm. The cube in the jaws stands on no layer (cut at one it read
18 mm long), so it is PnP's, from close.

**The stance** is lying, as the pen's: standing, the walking policy never
quite stops. Lying, the arm puts the jaws on the floor from 0.35 to 0.82 m
ahead, and the imager sees a cube's whole tag from 0.40 to 0.68 -- the claw's
working band (`tools.claw.REACH_X`). The walk in is steered by the cube's
tag to where lying down leaves it mid-reach (`LIE_AT_M`, 0.54 m), the body
lies and looks again; out of reach it stands, backs out and walks in again,
three times.

**The grip.** MuJoCo's soft contact lets a held cube creep at about
(1 - d)/d * g/b. The pads are stiff (solref 0.006, solimp 0.99/0.999 with a
10 um width) and the jaw servo saturates at its stall force (kp 2000, 10 N):
at 600 N/m the jaws breathed and a shaken cube slipped 1.6 mm/s; now it
creeps 0.032, 0.09 and 0.19 mm/s at 60, 150 and 320 g. condim 4 and a
stiffer solimp (0.9995) made it worse. Held is judged by DISTANCE -- both
pads within 0.5 mm of one body that can move -- because under the moving
body the stiff pads' contacts came and went from step to step, and a cube
in the jaws read "nothing held". Setting one down, the claw first shows it
to the imager low in front of the target and measures where it hangs.

**The search.** Where the cube was last seen, else in front of its area's
tags: the workshop corner's pair (42-43) and the bench's (44-45), the props
set out 0.70 m in front of them. From 1.2-1.5 m out the row decodes whole.
The bench's tags, first seen from the street through two doorways, were
remembered 0.12-0.23 m off and 6 deg askew, and from the one look point on
their middle cube 24 decoded at 0.78 m while cube 23 stood 0.96 m off, past
the imager: so the search stops three times along the row, each stop
planned off the tags as remembered when it is walked to.

**What is true now:** the claw picks and places lying; towers stacked by
the body's routines stood 6 of 6 (lean 1.3-5.7 mm); ladder A flies TOWER
and WEIGH on the served pair from the dock (`scripts/solve.py --feature
tower|bench`, reported on #407). A claw hung back holding a cube is a cube
hung on the rack, so a stow sets it down first.

## The census on legs (issue #407)

The garden's plants are 0.30 m tall and the quadruped's LIDAR scans at
0.51 m, over them: they are the depth camera's low layer. The area is
found, not handed over -- the known floor the LIDAR's walls and fence
enclose round the garden's tag (46, on the east fence across from the
living room's doorway; on the house's own wall a robot inside never saw
it), opened by half a door (`census.DOOR_M`, 1.2 m) so the doorway closes
it.

Three counts were wrong before one was right. Counted off the planner's
layer, a stalk's cell seen from one side was taken back by the floor seen
round it from another: 2 of 4 at 91 % coverage. A survey keeps a layer of
its own that only adds, an object at three frames. Then five: the garden
light's 40 mm pole, which rays slipping past clear from the fused map,
stood in the low layer -- what a LIDAR return hits is taller than its
plane, so an object at one is not a plant. Then two: a walking torso's
pitch put returns 7.4-8.0 m off onto two of the plants (the level gate is
1.5 deg), so only a return within 3 m (`survey.LIDAR_TALL_M`) marks
anything.

**What is true now:** from the living room's doorway, 4 of 4 at 91 %
coverage in 361 sim s over 23 vantages; from the dock the whole job (the
tag found, the LCD fetched, the survey, the LCD hung back) took 586 s and
13.3 Wh, and paid. The count is on the LCD while the LCD is on the fork.

## The bench's scale (issue #407)

The rover weighed with its lift's load. The arm's drivers report torque
(the third field of the reply: 12 bits over +-22 N*m off the phase current,
with a stated 25 mA x 1.19 N*m/A of sense noise -- 0.03 N*m a reading;
`perception/encoders.py`). The weighing is a ratio: the empty claw, the
known cube and the unknown, each held at the pose a pick leaves the arm
in, ten readings each; the mass is the known one's times the ratio of what
each added, so the arm's lengths, its own weight and the claw's cancel. The
elbow read 250 g within 0.2 % and 60 g within 1.5 % (the shoulder 4.5 % and
2.7 %, carrying the body's sway); `bench.TOLERANCE` is 10 %.

**What is true now:** WEIGH recorded 0.181 kg for 0.18 on the served pair,
paid; a robot writing its own weighing reads `shoulder.torque` and
`elbow.torque`.

## Two robots at one rack, on legs (issue #418)

The rover's wait (#346) backed off to a spot. The quadruped holds where its
approach starts, a metre behind the bay, while the other robot's reported
pose is within the planner's disc (0.55 m standing) of THIS bay's working
pose -- Ben's decision: the next bay's is 0.30 m off, two over 0.60. Pair
tables (`arm_spike.py --served --pair --bays`): neighbouring bays 9 of 10
swaps, waits 2-16 s; two apart 9 of 10, no waits; one bay, one hanging back
while the other comes for it, 8 of 8, waits 4-9 s. One robot alone, 18 of
18. The two misses were stows that let go (the pen at B, -11.2 mm across
and -2.6 deg; the claw at C, +6.9 mm), each with the other robot working
at the same time: recorded, not explained.

The first one-bay table found what the wait does not cover: a robot left
standing after its stow lay down by reflex 0.36 m inside the bay's
approach, and the other's walk in -- steered by the bay's tags, never the
planner -- walked over it. The loop never leaves one there (#346 sends a
robot done at the rack away), so the table models the loop; the walk-in
still avoids no peer.

**What is true now:** `ToolSwap._bay_free_routine` asks the lifecycle's
wait with `hold`, and a swap whose wait gave up is `blocked`.

## The workshop on legs (issue #407)

The rover's built-tool rail went with it (#376). The quadruped's rack has
three bays and the three tools every offered job is written against hang in
them (#277's rule keeps them permanent), so a built tool gets a rail of its
own: three more bays along the same board, in line with the rack's
(`legs.rack.BUILT`, at 0.65, 0.95 and 1.25 m in the rack's frame, a pair of
tags a bay, 47–52). One board means one commissioned pose, one approach and
one look that fits every tag it sees (`fit_rack(seen, SPECS)`).

**The envelope is the arm's** (`workshop/validate.py`, every number read off
`legs/arm.py` and `legs/rack.py`; ToolPattern.md §2). The rover's latch
moment, bracket band and wall went with the rover. Each new rule was found
by a build that broke it:

- **A hung tool hangs plumb.** The rover's example scoop, its blade forward
  of the plate, hung 5° off plumb, and `on_bay` read it as not hung; the
  prompt's example hangs its blade straight down.
- **...and centred.** A centre of mass 70 mm to one side validated and
  tipped 47° off the trays. Measured in the rig, 30 mm hung, was taken and
  hung back with the fork 15 mm either way; 35 mm did not seat with the fork
  15 mm toward it; 45 mm (the trays' own ±45) did not hang: the rule is 25.
- **Nothing behind the plate** (the fork's prongs, bridge and lean-pad),
  **nothing in front of its bay's tags** — the plate's own slab included:
  a crossbar there, 76 mm wide, hid both tags from the working pose — and
  **nothing in reach of the rail** the trays hang from, hung or lifted by a
  pick: a thin handle up to 150 mm over the peg jammed on it at 15°.

**And the rig is the last gate** (`build.trial`), before a point moves: the
module hung, taken, worked and hung back on a bench rack, the fork on the
bay's middle and at the line-up gate either side, 0.25 s a try. A tool that
cannot hang is otherwise lost the moment it hangs, and the lost-tool clock
puts it back on the same bay every 300 s for ever. The names a module would
bring are checked against the world's first (`seam.names_taken`): a tool
called `claw_carriage` named a body of the claw's and was refused only once
its parts were bought, and one called `claw_pad` with a part `l` broke the
world's compile.

**At its stow.** The validator checks a tool hanging at its stow, so the
module compiles there: each moving part built at its axis's stow with the
joint's `ref` set to it, so `qpos0` is the stow — the rest a walk, a
hang-back and a reset return it to — and the seam sets each new servo to
hold there (a recompile starts a new actuator at 0). Built at 0, a flap
whose stow is −90° hung 14° off plumb and was lost from the moment it hung,
and a hinge a procedure had swung was hung back swung.

**Two bugs in the build path**, each pinned in `tests/test_workshop_build.py`:
a servo's range was written in degrees into a module compiled in radians, so
its 0–90° compiled as 0–90 rad and limited nothing; and the stand-in inertial
given a part with no mass of its own zeroed the masses of the parts riding on
it (only an empty load gets one now).

**The seam on the served pair.** A built module is attached after the
robots, so no robot's ids move; every holder of the old model re-resolves its
ids by name (`QuadMission.rebind`: the drivers, the policy, the arm, the
odometry, the places, the claw). The hang waits until every body is still: a
body lying down, standing up, working a bay or stepping aside is a generator
holding the world it began in, as a restart's save waits (`Keeper.busy`). A
recompile builds the model from the spec, so whatever the running world had
written into the old model goes: the garden lamp went dark for good until
its toggle's rebind applied its state again. Measured (`scripts/workshop.py
--example --served --bay A|B|C`, the pair standing): the recompile 6–10 ms
of wall time; no stale holder in either lifecycle; neither robot moved or
fell; the scoop hung plumb; the arm fetched it on its first walk-in
(3.5–5.3 mm across, within 1.5°) and hung it back.

**What is true now:** `home_quad` has the workshop (`world_config`'s
`built_bays` is 3), its prompt states the arm's envelope (`workshop_rule`),
and a bay letter past the rail's is refused without naming an original's
bay. A robot standing by is sent clear of the rail's bays and their
approaches as well as the dock (`rack_distance`, off each rail bay's
standoff: computed off the dock's frame, two rail bays' approaches were
"clear").

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
