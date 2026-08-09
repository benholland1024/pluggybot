# Simulation Notes

Hard-won MuJoCo lessons from building PluggyBot. Each entry exists because something broke without it. Keep this file growing — future-you (and the Onshape import) will thank present-you.

## Physics modeling rules

### Wheel joints need `armature`
A bare 50 g wheel has ~3×10⁻⁵ kg·m² of rotational inertia; a motor torque applied to that overshoots any velocity target within a single 2 ms timestep, and the servo then chatters at the timestep frequency — the robot "vibrates and bounces" instead of driving. `armature` models the motor rotor's inertia *reflected through the gearbox*, which scales with gear-ratio²:

> armature ≈ rotor inertia (~5×10⁻⁶ kg·m²) × ratio² → **0.005 at 30:1, 0.012 at 50:1**

### Wheel joints need `damping` (and it's physically real)
Gearboxes eat torque: Pololu lists ~65 % efficiency for the 37D 50:1, so ~30–35 % of torque is lost to friction. `damping="0.05"` on the wheel joints models this — and it is also *load-bearing for stability*: the velocity servo regulates **joint** velocity, which includes chassis pitch rate, creating positive feedback that pumps energy into chassis-pitch oscillation. Joint damping dissipates it. Don't exceed honest magnitudes (0.2+ would mean a gearbox that eats the entire stall torque — unphysical, and it makes the robot crawl).

### Use `integrator="implicitfast"`
MuJoCo's default explicit Euler integrator adds energy to velocity-dependent forces (velocity actuators, joint damping) — the exact ingredients of a wheeled robot. `<option integrator="implicitfast"/>` is standard practice for wheeled/actuated models (most Menagerie models use it).

### THE caster lesson: MuJoCo combines pair friction as the elementwise MAX
`friction="0.001"` on the caster never worked: when two geoms touch, the contact's friction is (at equal priority) the element-wise **maximum** of the two geoms' values — and the floor's default is 1.0. Our "frictionless" caster was a full-grip rubber ball for weeks, silently exerting ~4 N of drag and ~0.4 N·m of yaw braking. That single hidden brake caused, downstream: the cruise pitch resonance blamed on 90 mm wheels, the tail-flip, the 12.6 % straight-line odometry creep, the 2.7× in-place-turn overestimate ("turn walk"), and the apparent head-height ceiling. The correct frictionless caster (copied from Menagerie's Stretch):

```xml
<geom name="caster" type="sphere" size="0.02" ... condim="1" priority="1"/>
```

`condim="1"` = normal-force-only contact (no friction dimensions exist at all); `priority="1"` makes the caster's contact parameters win over the floor's (otherwise condim also combines as max and the floor's 3 wins). With this fixed, the earlier `solref` tire-softening became unnecessary and was removed, the head can sit at mast height (tested to z = 0.16, pitch 0.1°), and dead-reckoning odometry agrees with ground truth to ~0.2–1 % on straights, spins, and arcs.

Meta-lesson: **know the pair-combination rules** (friction → max, condim → max, solref/solimp → priority/solmix-weighted) before trusting any per-geom contact attribute.

### Mass goes over the drive axle
A 120 g head cantilevered 16 cm ahead of the axle destabilized launches (wheelie → riding the rear chassis corner at ~30°); the identical mass directly above the axle is benign. Established empirically by bisection: it's mass × position, not mass. Same principle as Roomba/TurtleBot battery placement — keep it. *Historical correction:* the follow-on "head height ceiling" (z = 0.135 unstable) was measured **before** the caster-friction bug was found; with the caster truly frictionless, head heights to at least z = 0.16 are stable at 0.1° pitch. The forward-cantilever result may also have been amplified by caster drag (it loaded the caster harder), but weight-over-drive-wheels remains sound design regardless — traction depends on it.

### Motor sizing: torque-to-weight matters
The original 30:1/1.4 N·m spec on a 1.1 kg robot demands ~40 N of thrust per wheel against ~3 N of available traction — permanent wheelspin at launch, and enough reaction torque to wheelie. Real robots this size are torque-limited by traction, not by motor. When behavior looks violent, check whether the actuator could physically exist in that weight class.

## Known artifacts (accepted for now)

- ~~In-place turns "walk"~~ — **resolved**: this was the caster friction bug (the caster pinned its end of the robot, shifting the turn center off the axle). With `condim="1" priority="1"` the turn center returns to the axle line and spin odometry matches truth to <1 %. The residual "drift" in the turn regression test is mostly geometry: the freejoint origin sits 8 cm ahead of the axle and orbits it during a spin.
- **Frame mismatch in odometry comparisons** — dead reckoning tracks the **axle midpoint**; `qpos[:2]` tracks the **body origin**, 8 cm ahead. On curved paths they trace different circles (~0.1 m apart after a half-turn). Compare truth at the axle: `(x − 0.08·cos ψ, y − 0.08·sin ψ)`.

## Test & world hygiene

- **`models/world.xml` is bare** (floor + light + robot): physics tests run here. **`models/playground.xml`** adds scenery via `<include file="world.xml"/>`: teleop and camera scripts run here. Scenery once parked a box in the drive-test lane; separately, copying the floor into the playground doubled every wheel contact (the include already provides it — a "repeated name" MJCF error means *delete* the duplicate, not rename it).
- Every debugged failure becomes a pytest assertion (rests level, drives straight, no wheelie, turns in place, stereo parallax exists). The suite has already caught two real regressions.
- **Relative-error metrics need denominators that can't vanish.** Two incidents: position-error ÷ distance blows up on an in-place spin (distance ≈ 0), and heading-error ÷ net-rotation blows up on an S-curve (segments cancel, net ≈ 0 — a 0.17° error read as "142 %"). Normalize by *path traveled* (distance rolled, rotation swept), or assert absolute error.
- Odometry comparisons: dead reckoning tracks the **axle midpoint**; `qpos` tracks the body origin 8 cm ahead. Transform truth to the axle before comparing (see tests/test_odometry.py `axle_pos`).

## Conventions & gotchas

- MJCF `size` values are **half**-extents; `pos` is relative to the parent body frame.
- Cameras look down their own **−z**, image-up is +y. Forward-looking camera on a +x-facing body: `xyaxes="0 -1 0 0 0 1"`.
- Pitch extraction from the freejoint quaternion `(w,x,y,z)`: `asin(2·(w·y − z·x))`, **positive = nose down** with x-forward/z-up.
- A body with no joint is welded to its parent; a `<geom>` with `contype="0" conaffinity="0"` is visual-only (wheel spokes, future pretty meshes).
- Velocity actuators: `ctrlrange` = ± no-load speed (rad/s), `forcerange` = ± stall torque (N·m), both straight off the motor datasheet; `kv` is a tuning gain, not a datasheet number.

## Exploration lessons (milestone 4)

- **The nearest-frontier deadlock.** A forward camera cannot observe the cells beside its own wheels, so the nearest frontier is always the unscanned sliver just outside the FOV — the robot "arrives" instantly, stops, and the frontier never dissolves. First verification run: 420 sim-seconds, zero movement. Fixes: ignore frontiers closer than ~0.3 m, and do a 360° look-around spin at startup and whenever no distant frontier is reachable.
- **Collisions corrupt the map, not just the paint job.** Grinding a wall slips the wheels → odometry counts phantom distance → the map frame slides → old walls repaint at new believed positions ("jail bar" artifacts, evidence outside the room). Prevention beats cure: obstacle inflation must exceed the chassis **half-diagonal** (0.15 m) plus margin — we use 5 cells / 0.25 m; sparse waypoints + generous arrival radii cut corners through the inflation ring (use ≤3-cell spacing, ≤0.08 m radius); and a scan-based reflex (stop + back off when anything is <0.25 m dead ahead) catches what planning misses.
- **The safety reflex must be armed in every maneuver, not just while driving.**
  `explore.py` gated it on `mode == "drive" and waypoints`, so look-around spins ran
  blind. Measured over a full run: its spins passed within **0.257 m** of the L-box —
  7 mm outside the 0.25 m threshold. Zero collisions was luck, not design. Refactoring
  into `lifecycle.py` shifted the trajectory by <1 mm/step (closed-loop driving among
  obstacles is chaotic; identical logic will not retrace an identical path), the final
  spin landed inside the margin, and the chassis ground for **503 steps**. Arming the
  reflex in all maneuvers cut that to 43, all of them *during the escape*. Caveat: a
  forward ±20° reflex fundamentally cannot protect a spin (the chassis corner sweeps
  through arcs the camera never sees) — driving it to zero needs 360° clearance
  memory, e.g. checking the robot's own cell against the inflated grid before spinning.
- **Don't re-arm a reflex that is already firing.** Re-triggering backoff on every scan
  while reversing turns a bounded 0.8 s pulse into an open-ended reverse into whatever
  is behind the robot.
- **Termination is "no *reachable* frontiers," not "no frontiers."** Unreachable slivers (pockets inside obstacles, hairline gaps) are blacklisted when A* fails; exploration ends after a look-around spin plus repeated pathless replans. Benchmark: both rooms of room_1.xml in ~80 sim-seconds, 0 chassis contacts.

## Landmark & docking-approach lessons (milestone 5→6)

### "Where I saw it from" is not "which way it faces"
The first standoff-pose estimate derived the outlet's outward normal from the mean
robot position across sightings. It sounds sound — the robot only ever sees an outlet
from the open side — but it records *where the robot happened to drive*, and a drive-by
biases it badly. Measured end to end: **31.2° off the true wall normal**, which put the
docking hand-off pose 33 cm sideways with the socket at the very edge of the camera's
66° horizontal FOV. The fix reads the normal off the occupancy grid instead
(`landmarks.wall_normal`): sum a unit vector toward every nearby known-free cell, and
since a wall blocks half the circle the sum points out of it — no line fitting, no
normal-direction ambiguity. **31.2° → 0.0°.** Seen-from survives only as the fallback
for outlets whose surroundings were never mapped.

Meta-lesson: **decompose an error before fixing it.** The 33° miss was first written off
as odometry drift. Splitting it into controller settle / odometry drift / direction
estimate showed drift was **0.02°** (the gyro fusion is excellent) and the estimator
owned essentially all of it. Guessing would have wasted the effort on the wrong subsystem.

### The camera's FOV, not the wall, bounds usable outlet height
With the eye fixed at z = 0.18 m and fovy 41°, the visible band is z ≤ 0.40 m at the
0.6 m standoff but only z ≤ 0.31 m at 0.35 m. Verified in `room_1.xml`: outlet C at
0.38 m is detected at 0.6 m (conf 0.93) and **disappears entirely at 0.35 m**. So the
approach can see a high outlet and then lose it exactly when it matters — the concrete
argument for the prismatic lift in milestone 6, and outlet C is deliberately left high
as the test case. Room walls are 1.20 m (was 0.30 m), which also puts them inside the
detector's trained wall-height range of 0.5–1.5 m.

### The generator's own val split cannot measure the detector
Training and validation images come from the same `outlet_scene.py` sampler, so the val
split only measures what the generator already thought to vary. It reported **mAP50-95
0.9938 while the detector was calling a light switch an outlet**. `scripts/eval_detector.py`
exists for this: it samples collision-free robot poses throughout `room_1.xml`, renders
what the camera would really see, and scores against segmentation ground truth. Every
real defect in this milestone was found there or by the distance sweep — none by mAP.

### A handful of poses is a smoke test, not a measurement
Three data recipes were compared on an 8-pose spot check, and a decoy false positive
moving 0.61 → 0.93 was read as a regression signal. At that sample size it is inside
run-to-run training variance. Re-run over 300 poses the ranking was unambiguous
(false positives per frame): original 0.140, +close-range 0.187, **+decoy-aimed negatives
0.127**, +distance-scaled aim jitter 0.213. Compare recipes on hundreds of samples or
don't compare them.

### One measured mistake, kept as a warning
Aiming negatives at decoys with a *distance-scaled* jitter was meant to teach the
detector about decoys cropped by the frame edge. It measured worse: the wide jitter
pushed the decoy out of shot in 22 % of negatives, diluting the hard negatives it was
meant to sharpen (decoy false positives tripled, 9 → 26). Reverted to a tight aim.
The lesson generalizes: **check what a generator change actually produced** — a contact
sheet or a pixel-coverage histogram — before spending 18 minutes training on it.

### Recall is recoverable; precision errors compound
Raising the detector threshold 0.5 → 0.7 costs 2 detections in 105 (recall 0.952 →
0.933) and removes 26 of 40 false positives that land on wall decoys. That trade is
right *for this system* because the two errors are not symmetric: the robot sees each
outlet 13+ times per run, so a miss is recovered on the next glance, but a *systematic*
false positive on a fixed decoy accumulates sightings in the same place and graduates
into a confirmed phantom landmark the robot will drive to. Tune the threshold against
what the downstream consumer does with the errors, not against an F1 score.

### The sighting threshold earns its keep
Drawing tentative landmarks on `map.png` in olive immediately exposed one: the detector
fires on `decoy_switch_w` (a light switch) about twice per run, at the right position
and height. `min_sightings=3` filters it, and it never reaches the confirmed list. Worth
remembering that the detector's precision on the val set (0.98) is not zero false
positives in the world — the confirmation count is what makes the landmark map clean.

### Budget the tolerance across the whole chain, not per-stage
`FACING_TOLERANCE` was 2° against the spike's ±3° docking budget — the settle criterion
alone spent two-thirds of the allowance before odometry or the docking controller got
any. At 0.5° the full pipeline now parks at **-0.49° yaw, 1.3 cm lateral, 60 cm out**.

## Rendering lessons (milestone 5)

### Segmentation rendering needs `offsamples="0"`
Free labels from segmentation rendering are only free if the buffer is honest. MuJoCo
applies multisample antialiasing to the segmentation image too, blending geom IDs at
object edges — so pixels appear carrying IDs of geoms that aren't there. Most land
harmlessly, but a label box spans the **min/max** of its mask, so one stray pixel 200 px
from the outlet stretches the box across half the frame. Measured over 236 positive
scenes: **12.3 % grew a second blob, 5.5 % came out grossly elongated** (a square 80 mm
Schuko plate labeled as a bar with w/h up to 7.7). The training run scored mAP50-95 0.940
*despite* ~3 % of its labels being garbage, and the val images showed the model drawing
two correct tight boxes where the truth was one absurd bar — the model was right.

```xml
<visual><quality offsamples="0"/></visual>
```

Drops stray blobs to 0.0 %. `make_labeled_sample` additionally keeps only the largest
connected blob, so a survivor gets dropped rather than silently poisoning a label.

Meta-lesson: **the renderer is not a measurement device by default.** Anything that
smooths pixels for human eyes — antialiasing, filtering, interpolation — corrupts a
buffer whose values are identifiers rather than colors.

### Dataset regeneration must clean its output directory
The train/val split is a random per-image draw, so a file whose split changes between
runs leaves its old copy alive in the other split — regenerating after the MSAA fix
still left **195 stale image/label pairs** from the buggy run mixed into training, and
the "residual" corrupt labels were all leftovers. The generator now deletes `images/`
and `labels/` before writing (guarded by a pytest). Symptom to remember: the same
basename appearing in both splits.

## Schuko contact spike (milestone-6 prep)

Standalone plug/socket rig (`docking/schuko.py`, `scripts/schuko_spike.py`) — no robot,
a compliant carrier pushes the plug with a 10 N force limit. Findings (guarded by
tests/test_schuko.py):

- **Collision geoms must be convex**, so the concave recess is *composed*: 12-box
  dodecagonal well wall, 12 tilted boxes as a 45° entry funnel, 5 floor slabs leaving
  two square pin holes, 4 boxes framing the face. Capsule pin tips double as their own
  entry chamfer.
- **Capture is set by the entry chamfer, not the recess.** First version used an 8 mm
  45° funnel and measured a flattering ±18 mm lateral / ±4° yaw. Ben's challenge —
  "real Schuko rims don't have that funnel" — was correct: with an honest 2 mm rim
  bevel the envelope is **±3 mm lateral / ±3 mm vertical / ±2° yaw**. Capture ≈ body
  clearance (0.75 mm) + chamfer; the deep recess only *guides after* capture. Don't
  widen the chamfer to make a failing controller pass — that tunes the world, not the
  robot. (Measured chamfer→tolerance: 2 mm→±3 mm, 4 mm→±6 mm, 8 mm→±18 mm; a dished
  face plate on the physical charging outlet is a legitimate *hardware* choice that
  buys margin honestly — a Parts.md decision, not a sim default.)
- **Yaw is the tight constraint** — with a diagnostic signature: every yaw/lateral jam
  stops ~19 mm short (= pin length): the pins bottom on the floor beside their holes
  before the shallow well can square the 40 mm body. Docking must get *facing* right.
- Jams are clean (stall at the force cap, no solver explosions); worst transient
  contact force ~32 N during an edge-of-envelope wedge, fine at `timestep 0.001` +
  `solref "0.005 1"`.
- **The velocity-servo chatter lesson generalizes.** A velocity actuator driving the
  90 g carrier chattered at the timestep frequency exactly like the bare wheels once
  did (kv needed for 10 N @ 2 cm/s is far too stiff for that mass). Fix: constant-force
  `<motor>` + heavy joint damping (`damping = F/v`) — implicitfast integrates damping
  implicitly, so it cannot chatter, and "10 N push, 2 cm/s free speed" is the honest
  robot semantics anyway.
- Caveats for milestone 6: carrier compliance was 150 N/m lateral / 1 N·m/rad angular
  (a guess at arm+base flex — revisit with the real arm), gravity off (the lift owns
  height), and tolerances scale with that compliance.

## Arm modelling lessons (milestone 6)

### A wall cannot brace you — one-way contacts don't pull
The Parts.md wall-brace idea (pads take the insertion couple so the base doesn't tip)
was **measured false**: a wall contact only pushes the robot *away*, the same direction
as the insertion reaction, so pads add to the overturning moment. With-brace performed
worse than without. The prongs survive as *alignment feelers* (two tips touching = yaw
mechanically squared across a 0.14 m base + depth reference), not as a brace. Measured
push capacity at outlet height (2.34 kg robot, ramped load): **forward docking holds
~3 N then slides; backward holds ~4 N then goes caster-light**. Against spike insertion
forces (0.7 N aligned / 6.1 N at 2 mm / 7.8 N at 2° yaw) the budget means: insertion
must stay ≲3 N → terminal alignment ≲1 mm, softer RCC, or active-drive strategies —
the docking controller's problem, now with numbers.

### MuJoCo silently ignores adjacent-link interpenetration
A body with no joint is *welded* into its parent, and contact filtering treats the weld
group as one body — so the carriage (child of the chassis weld group) can sweep its
prongs straight **through** the head mount and the contact pipeline reports nothing
(verified: overlapping AABBs, zero contacts). On hardware that's a collision; in sim
it's silence. Consequence: clearance between adjacent links must be asserted from
*geometry* — tests/test_arm.py sweeps the full envelope checking AABB disjointness, and
the check caught two further 5 mm interpenetrations (tube↔motor, carriage↔chassis) that
contacts never would. Endpoint checks are not envelope checks; contact checks are not
clearance checks.

### Asymmetric mass makes a diff-drive veer open-loop
The arm assembly (~0.8 kg at y=−0.05) made the robot veer **26 cm right over 4 m** of
straight driving — wheel velocity servos track equal speeds, but unequal wheel loads
slip unequally. Fixed by counterweight: the battery sits at y=+0.06, tuned to null the
*measured veer* (+4 mm over 4.2 m), which leaves ~7 mm of static CoM offset because
load-dependent slip is not linear in CoM. Position the battery in the model as a
counterweight first, packaging second.

### Inflation must track the outermost geometry, not the chassis
The feeler tips sweep 0.27 m from the axle — outside the 0.25 m obstacle inflation
calibrated for the bare chassis (0.15 m half-diagonal). First long explore run with the
arm: a prong clipped an obstacle mid-corner. `traversable_mask` default is now 7 cells
(0.35 m). Anything that grows the robot must grow the inflation.

### A repeated false positive becomes a confirmed phantom — demonstrated
On the first long armed run, the detector fired 6× on `decoy_blank_s` (a blank plate),
the landmark confirmed, GO_CHARGE selected it as nearest, and the robot parked squarely
in front of a blank plate. The dock report's ground-truth check exposed it (−364 cm
"lateral offset" vs the nearest real outlet). The threshold/eval work reduced random
false positives, but a *systematic* one at a fixed spot defeats sighting counts by
simply recurring. Defense: the DOCK state must re-verify the target with a close-range
look (dock_eye) before committing — planned for the docking controller.

## Docking controller lessons (milestone 6)

### Walls have no holes, so wall sockets must be surface-mounted
The spike's socket floated in space; placed in room_1, its 37 mm recess passed through
the wall slab, and the first docking probe measured the pins bottoming on the *wall*
one pin-length short. MuJoCo cannot cut a hole in a box, so the sockets are modelled
as German **Aufputz** (surface-mount) fixtures — recess fully proud of the wall, with
a visible housing box behind the visual plate. Real hardware, not a workaround.
Collision layer: invisible (alpha 0) generated bodies in `models/schuko_sockets.xml`
(regenerate: `uv run python -m pluggybot.docking.schuko`), aligned with the visual
outlets; unrendered geoms also stay out of segmentation, so detector GT is untouched.

### Gravity was off in the spike, and gravity is a docking axis
The plug axis sags **7.8 mm** under gravity (lift servo droop + RCC springs) — the
spike never saw it. Pressing the feelers adds a pitch sag that GROWS with lift height
(wheel-torque reaction, softer at high CoM). Both are feed-forward calibration
constants now (`DOCK_DROOP_COMP`, press-sag ~118 mm per m of lift), exactly the shape
of calibration a real robot needs.

### A camera above the plug loses the target exactly when it matters
The dock camera rides 0.06 m above the plug axis; below ~0.32 m range the socket
slides out of the frame bottom, the detector boxes the surviving sliver, and its
centre biases up — hand-eye calibration measured +2 mm of vertical bias at 0.32 m,
+23 mm at 0.19 m, while the LATERAL centre stayed honest to 0.19 m (+1 mm). Hence the
split-axis servo: lateral steers nearly to contact, vertical freezes at 0.32 m.
Measure each axis of a visual servo separately; they do not fail together.

### The seat detector was harder than the seating
Four verdict schemes failed in sequence, each defeated by a real event: extension
windows (base slides back, extension lies), base-release (a jammed plug also shoves
the base off), odometry advance (frame bug on one wall heading), floor-contact by
name (a pin bottoming on the floor's FRONT face reads the same as a pin in a hole).
What survived: **the electrical criterion** — pin contact ≥19 mm into the recess,
i.e. inside a pin channel, which is unreachable except through a hole. It is also
literally the sensor the physical robot will have (charging voltage), and the
milestone-7 battery hook.

### Scripted-baseline verdict: mechanics solved, vision-z is the gap
From a perfect standoff the mechanical stack docks deterministically (pytest:
`test_mechanical_dock_from_perfect_standoff`) — feelers square the yaw, the RCC +
funnel absorb ±4 mm, the charge criterion confirms sub-mm seats. End-to-end with the
visual servo under 2 cm standoff jitter: **4/12 (33 %)**, failures dominated by
plug-height error the box-centre servo cannot measure well. That is precisely the
gap the keypoint/PnP pose model (or RL) was predicted to fill — next.

## RL docking environment lessons (milestone 6)

### The feelers must clear the housing, not the plate
First cheat-policy run in the RL dock env: 0/12, every near-miss stalled ~20 mm
short at the socket mouth. Telemetry showed one feeler in contact 40 mm too
early: at the ±0.07 m straddle, >7 mm of lateral error put a prong tip on the
surface-mount housing's edge (collision half-width 0.055, standing 39 mm proud
of the wall) instead of the wall behind it — one prong 39 mm short of the other
wrecks both the yaw squaring and the depth reference. The ±0.07 figure had been
checked against the ±0.042 *visual* plate. Widened to ±0.085 (22 mm margin):
12/12. Then the widened left prong swept through the battery's corner at low
lift — caught by test_arm.py's geometric transit sweep (contacts silent as
always, same weld-filtering story) — so the prongs now ride 2 cm above the plug
axis on a standoff bracket. Both constraints are pytest-guarded.

### A synthetic sensor must be calibrated against the real one — including the robot's own body
The env trains without YOLO: the "detector box" is the socket projected through
the pinhole model and clipped to the frame (clipping is the measured mechanism
behind the real detector's close-range vertical bias, so the synthetic sensor
inherits that signature by geometry). Two calibrations still had to be
measured, not assumed:
  * **Effective extent.** The real YOLO boxes plate + housing at a steady
    **47 mm half-extent** (measured 0.6 → 0.22 m, aligned robot) — not the
    42 mm plate the first model projected.
  * **Self-occlusion.** With the arm extended, real boxes shrank ~25 % from
    below: the tube top rides ~0.048 m under the camera and its edge climbs the
    frame as the arm extends. The first policy trained on the un-occluded model
    learned to approach arm-first — in-env vision saw *through* its own arm, so
    nothing taught it that extending early blinds the camera. The occlusion
    line is now part of the sensor model, with the measured signature guarded
    by a pytest.
Meta-lesson: a simulated sensor is a claim about the real one; measure the
claim before spending training compute on it (the contact-sheet lesson again,
one level up).

### Randomize with a mocap body, not recompilation
The socket rides a `mocap="true"` body: reset repositions it continuously
(height 0.24–0.40, lateral ±0.30) by writing `mocap_pos` — no model recompile,
no discrete height grid. Static collision geoms on a mocap body behave as
kinematic fixtures; contacts work normally.

### The eval protocol is only trustworthy once it reproduces the baseline
`eval_docking.py` makes the "2 cm standoff jitter" protocol explicit constants
(pose jitter ±2 cm/±1°, landmark error ±2 cm xy / ±1.5 cm z, odometry drift
±5 mm/±0.3°) and reruns the scripted controller on seeded trials: **4/12 =
33.3 %**, matching the recorded baseline exactly (and 8/24 on the larger
sweep — the rate held). Same trials, same real YOLO, same electrical charge
criterion for both controllers — the RL number and the scripted number are
finally the same experiment.

### Odometry in the loop is part of the sensor model
The policy trained on true-pose observations scored 100 % in-env and **1/24**
in room_1. Telemetry: it had learned to grind the wheels against the wall at
0.1 m/s — free when observations come from ground truth, catastrophic when
they come from dead reckoning, because slipping wheels integrate phantom
distance and the believed target range collapsed to zero within seconds of
wall contact (the milestone-4 slip lesson, relearned inside an observation
vector). The env now runs the actual `DeadReckoner` class per physics step.
Corollary: whatever estimator the real robot runs, the training env must run
the same one, fed by the same failure-prone signals.

### SAC on this task finds the skill early and destabilizes late
Reproduced twice: success climbs to a peak (100 % in-env at 300k on the
lenient sensor model; 70.8 % at 140k on the honest one), then collapses to
0 % sweeps with runaway episodes polluting the replay buffer. Resuming the
best checkpoint at lr 1e-4 dipped (fresh empty buffer) and then diverged
again. Mitigations worth trying next time someone cares about this
controller: lower UTD, tau 0.002, TQC, or plain checkpoint-selection (which
is what shipped: `best.zip` is chosen by eval success rate, never by recency).

### The honest scoreboard, and what the RL work actually bought
End-to-end on identical trials: scripted **8/24**, RL **6/24** — from a
70.8 %-in-env policy, so a ~3× sim-to-room gap survived two rounds of
sensor-model honesty. But the failures are complementary (union 13/24): RL
took the high outlet C 3/9 where scripted took 1/9 — the vision-z gap it was
built to fix, fixed — and docks in 4–12 s vs 9–19 s. Its own systematic hole
is outlet A (0/8): telemetry shows a stable hover 0.35 m out, arm fully
extended, an observation state that cannot occur in-env — a residual
detection-channel mismatch (arm-out effective box extent ~41 mm vs the 47 mm
aligned calibration) parks the policy in an out-of-distribution attractor.
Meta-lesson: in-env success bought by details of a synthetic sensor is repaid
with interest at deployment, and only an eval that runs BOTH controllers on
the SAME trials makes the repayment visible. The wall-outlet RL effort parks
here by design — the hub pivot (PluggyPlan milestone 8) makes purpose-built
low-force docking the primary charge path, and everything this environment
taught (coupling spikes, sensor calibration, odometry-in-the-loop) transfers.

## Hub coupling spike (milestone-8 prep)

Standalone fork-and-peg rig (`hub/coupling.py`, `scripts/hub_spike.py`) — no
robot: a compliant carrier runs scripted pick-and-return cycles against a hub
shelf. The coupling is designed around the no-wrist constraint (latch verbs:
slide + lift): the tool hangs by a long peg axle in two upward-open V-trays;
the arm's fork grabs the peg *outboard* of the trays, so lateral capture is
set by peg overhang, not machined clearance. Gravity is the latch. Findings
(guarded by tests/test_hub_coupling.py):

- **Measured envelope: ±4 mm lateral, ≥ −8/+6 mm vertical, <2° yaw.** Yaw is
  the tight constraint — *again* (schuko found the same): picks survive 2°,
  returns don't, and ±4° jams at 50–120 N. Unlike the outlet, we own both
  sides of this interface, so the v2 levers are known: y-chamfered trays, a
  squaring press against the hub back wall (the feeler idea, reincarnated),
  softer yaw compliance. Navigation already delivers 0.5° settle, so v1 is
  usable as-is; the margin is just thinner than it should be.
- **Retention beats traction: held through 8 m/s² shakes** — more than the
  wheels can transmit — and carried a 300 g tool (2× the planned per-module
  mass cap). Gravity latching needs no spring, magnet, or actuator.
- Three geometry bugs found by filmstrip + phase telemetry, none by reasoning:
  the approach stroke stopped short of the peg; the V-plate tips rode above
  the peg's underside and rammed it horizontally (the whole fork now runs
  22 mm low and lifts 36 mm); and a lift 8 mm too short made the peg exit by
  grinding up the tray flank at the full 10 N push cap (clean is ~2 N).
- **Depth referencing by gentle press** carries over from docking: the
  approach just drives until the fork bridge bottoms against the tool face
  under the force cap — no range sensing needed at the hub.

### Robot integration: the fork inherits the plug's lessons, item by item
Mounting the fork on the real robot (`pluggybot_fork.xml`, `hub/swap.py`,
demo `scripts/hub_swap.py`) replayed three known plot beats in one afternoon:
- **The bumper reaches the hub before the fork does.** Retracted, the fork
  vertex sits 25 mm behind the chassis front — the chassis bottomed on the
  tray posts and the wheels slipped the last 5 cm of "odometry travel"
  (vertex stopped 43 mm short). The arm extends 60 mm for hub work — the
  same reach fix the plug needed.
- **RCC droop is a fork axis too**: the fork line sags ~7 mm under gravity
  (plug measured 7.8); compensated feed-forward in the lift preset.
- **The parked fork interpenetrated the chassis** (right prong, 9 mm at zero
  lift, plus 3 mm of spring droop) — caught by the geometric clearance sweep,
  silent to contacts, third time this class of bug has been caught this way.
  The fork now mounts 16 mm higher on the wrist; the lift preset absorbs it.
End state: full robot-driven swap cycles (pick, carry, re-hang) succeed with
±3 mm / 1° hand-off jitter — inside the spike's measured envelope, from the
real base, lift, odometry, and wrist.

### Hub-in-room navigation: three bugs the bare world could not show
Putting the rack in `room_hub.xml` (room_1's floor plan via a shared scenery
include, plus the fork robot) and driving to it end to end found three
failures that every bare-world test had passed:

- **A verdict written in the wrong frame.** `module_state` compared WORLD
  coordinates against RACK-LOCAL constants. In the bare world the rack sits
  at the origin unrotated, so the two coincide and the check looked correct
  for weeks; in room_hub (rack at (-0.9, 5.99), yaw -90°) it reported every
  correct placement as a failure. The mechanics had been working — the
  module was landing 0.2 mm from the hang plane while the test said "no".
  The verdict now transforms through the rack body's LIVE pose, which also
  survives the rack being a free body that can shift. Meta-lesson: an
  identity transform is not a validated transform. A check that only ever
  ran at the origin has never actually been tested.
- **A sign on the terminal travel.** The carried peg rides 7 mm *ahead* of
  the fork vertex, so the return must stop SHORT by that much; the code
  added it instead, overdriving 14 mm — past the tray V's ~16 mm mouth, so
  the peg came down outside the tray every time. Symmetric-looking
  compensations are exactly where sign errors hide.
- **Fixed choreography does not survive navigation.** The bare-world swap
  used fixed approach distances (a perfect standoff was assumed). Arriving
  by A* is good to ~8 cm and the coupling captures ±11 mm, so travel is now
  computed from the BELIEVED distance to the hang plane each time. Related:
  drive_to's 8 cm arrival radius happily parks 7 cm off the bay line — far
  outside capture, and the tag servo's authority over a 0.2 m creep is only
  1–2 cm. The fix is the oldest one in driving: **back up and take another
  run at it** (`refine_standoff`), which converges the P-controller
  laterally given a runway — 70 mm → 3 mm in one retry.

Two more that were merely ordinary: A* must plan to the reachable cell
NEAREST an unreachable goal (a blind greedy advance toward an unknown-space
goal ignored the very map it was building and ground into the floor-box's
reflex zone), and a demo waypoint that sits inside an obstacle produces
"mysterious" collisions — check the target before debugging the driver.

### Real AprilTags: what the swap cost and what it bought
Replacing the colour-plate stand-in with tag36h11 markers (`hub/tags.py`,
generated with moms-apriltag, decoded with pupil-apriltags):

- **`type="2d"` textures do not paint primitives.** A 2d texture is mapped
  through geom texcoords, which boxes do not carry: the plate rendered flat
  grey and nothing ever decoded, with no error from the compiler. `cube`
  mapping puts the image squarely on every face — which is also what a
  printed marker glued to a plate looks like.
- **Marker size is a range decision, not a detail.** A tag36h11 needs
  roughly 25–30 px to decode, so physical size sets detection range.
  Measured at 1280×720: the rack's 120 mm marker decodes past **4.5 m**
  with PnP range good to a few mm at working distances; the 30 mm bay and
  module markers decode to ~2.5 m, far more than the arm's-length reads
  they exist for.
- **One decode serves every marker size.** PnP translation is linear in the
  assumed tag size, so a single pass can be rescaled per id exactly —
  cheaper than a render per size.
- **It got FASTER.** PnP replaced the depth buffer, so each look is one
  render instead of two: a full mission dropped from ~40 s to ~30 s of wall
  time. And it matches the hardware, which has no depth camera on that path.
- **What it actually bought: identity.** Every reading is now keyed to a
  decoded ID, so the servo is told *which* marker to steer on. The
  stand-in's one unfixable weakness — guessing by size and image position,
  which once locked onto the charge bay's marker and dragged a module 22 cm
  toward the wrong bay — is no longer expressible.

### Finding the hub by looking at it: what the fiducial stand-in taught
Promoting the rack from "a pose the robot is told" to "a landmark the robot
sees" (`hub/localize.py`) cost five measured lessons, none of them about the
rack:

- **A white marker is not a detectable marker.** The tag plates started
  white; the room lights straight down, so vertical white surfaces render at
  ~178/255 and a `> 215` threshold saw *nothing* — while a lower one would
  have caught every pale wall. Recoloured orange, they still failed a
  channel-RATIO test, because MuJoCo's ambient term washes a (1.0, 0.45,
  0.05) material out to about (185, 115, 64). What works is **hue plus
  saturation**: scaling all channels (what lighting does) moves neither.
  This is the honest stand-in for the property real AprilTags get from being
  *patterns* — and a reminder that the marker is part of the perception
  system, not scenery.
- **Line of sight is a property of where you are standing.** The opening
  look-around saw nothing at all: from the demo's start pose the floor-box
  occludes the rack tag, the sight line passing just under its top edge.
  Discovery now rides along with every maneuver (the milestone-4 lesson
  again) — driving toward the rack is itself what buys the view. Measured
  result: **9 mm / 0.00°** from truth by the time it matters.
- **A stale prior must lose to an observation.** The believed rack pose is
  now a *fallback* (what a robot that booted docked knows); a confirmed tag
  overrides it, and a test starts the robot with a 30 cm-wrong prior to
  prove the correction happens.
- **Odometry cannot close a terminal approach.** Dead reckoning integrated
  over a mission was ~20 mm out by the return leg — twice the coupling's
  capture window. The terminal travel is now **ranged off the rack tag**
  through the dock camera, which does not drift; the odometry version
  survives only as the no-tag fallback.
- **"Nearest/lowest blob" is not identity.** With several fiducials in view
  the lateral servo locked onto the *charge bay's* tag and dragged the peg
  22 cm toward the wrong bay. Selection now drops the rack plate (much the
  largest, and deliberately offset from every bay) and takes whatever
  remains closest to the camera axis. This is the stand-in's weakest point
  and precisely what a real tag's ID removes.

### The hub lifecycle: four failures, all of them about *driving*, not docking
Wiring the hub into a battery-driven mission (`hub/lifecycle.py`) broke in
four ways, and not one of them was in the coupling that had been so
carefully measured:

- **The fork does not hold a tool sideways.** The V-notches constrain the
  peg vertically and fore-aft, but nothing constrains it along its own
  axis — the first carried module walked off during a turn and landed on
  the floor 2.5 m from the rack. Fixed in hardware, not control: end-stop
  posts just outboard of the peg ends (`fork_stop_l/r`). Carrying now
  survives three legs of hard maneuvering.
- **Tuck the arm before you drive.** After a *successful* stow, the robot
  drove along the rack to the charge bay with the fork still extended at
  module height and swept the module it had just put away off its trays.
  Stowed is the driving configuration; the fork deploys only once lined up
  on a bay.
- **Stop on the sensor, not on believed distance.** The charge approach
  drove by odometry, and pressing against the rack slips the wheels — dead
  reckoning counts the slip as progress, so neither the distance target nor
  the stall detector ever fired. It had been connected for 15 s and kept
  pushing until timeout, and the wasted press flattened the battery. Now it
  stops on `rack_charge_contact`, the same electrical criterion the plug
  uses. (Third time wheel slip has broken something that trusted odometry.)
- **Intended contact is not a collision.** The charge press put the pogo
  pins on the bumper, and the collision counter — written for wall-grinding
  in room_1 — read a perfect charge as 24 750 steps of crashing. Metrics
  need to know the difference between hitting something and touching it on
  purpose.

One arbitration lesson too: exploring must not outrank the job. With
explore first, the robot mapped, ran flat, charged, mapped again, and never
got to the errand it existed for. Priority is charge > errand > explore,
with an explore budget so background work is bounded.

### Capture and release are not symmetric
The return leg needed two asymmetries that "undo the pick" does not provide.
A carried peg settles only ~9–18 mm above its hanging rest, which is BELOW
the tray flanks' tops, so driving in flat strikes the flank (measured: the
module came down 29 mm low, still on the fork, and rode away again) — hence
`RETURN_CLEARANCE`, lift-clear-then-set-down, exactly as a person stows a
tool. And undoing the pick's lift exactly leaves the fork's V still touching
the peg: the contact list showed the peg correctly seated in *both* trays
*and* resting on the fork, 5 mm high, then dragged out on the way back.
Capture only needs the V to reach peg height; release needs it to go
clearly below (`RELEASE_DROP`). Reaching for the contact list rather than
guessing settled in one run what three rounds of tuning had not.

### Rack v2 (the unified rack): three measured calibrations
Rebuilding the hub as one freestanding rack (open frame, modules hanging
business-end-inward, charge bay, free body in sim) surfaced three numbers:
- **Approach overshoot is 4 mm, not 12.** With no depth stop to press
  against (the open rack removed the accidental one), the approach is pure
  odometry + V self-centering. 12 mm of overdrive wedged the peg between the
  fork V and tray V flank tips during the hand-off; the spike's own aligned
  figure (4 mm) puts the peg low on the near flank where the lift centres it.
- **The carried peg rides 7 mm ahead of the fork vertex** (the wrist tilts
  nose-down under the tool). The return compensates, or the drop misses the
  tray mouth by exactly that much. Feed-forward calibration again — the
  droop story in its fourth costume.
- **"Leaning on the wall" must be geometry, not intention.** The free-body
  rack scooted 9.6 mm under a sustained charge-bay press when its rearmost
  member stood 6 cm proud of the wall; with actual wall braces the same
  press costs 1.2 mm of take-up, and a full swap cycle moves the rack <1 mm.
  Modeling the rack as a free body is what made this measurable at all.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list, `ncon`) — vibes don't bisect.
2. **Render a filmstrip** (offscreen `Renderer`, tracking camera, 12 tiled frames) when numbers confuse — "it's standing on its tail" was invisible in scalars.
3. Bisect one variable at a time, and assert that programmatic XML patches actually applied (`assert count == 2`) — a silent no-op replacement once produced identical "before/after" results and nearly a wrong conclusion.
4. When symptom-fixes keep trading one failure for another, stop tuning and **measure the force balance directly**: `mj_contactForce` per contact, summed as torque contributions about the COM, named the caster as the yaw-brake in one run — after days of plausible-but-wrong theories (integrator energy, tire stiffness, wheel radius resonance, servo feedback).
5. **Consult reference models** (MuJoCo Menagerie — Stretch for diff-drive). The caster `condim="1" priority="1"` idiom was sitting in their XML all along; professionally-tuned models encode solved problems.
