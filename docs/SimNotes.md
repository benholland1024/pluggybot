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

### Motorised-module prep: a gravity latch cannot push
Before designing the lean-pad and the drawing tool, three things were measured
on a carried module (headless probes against `hub_world.xml`, LCD module on
the fork). They redirected the design, one of them away from a Parts.md
decision:

- **The hang is repeatable, and that is the good news.** Over a net-zero
  maneuver (out, back, turn, counter-turn), the module returns to its resting
  lean within **+0.00° / 0.02 mm** of where it started, and settles under
  0.5 mm in **0.14 s**. Three identical maneuvers landed identically. So a pen
  tip *can* be calibrated against the resting pose — the peg seats the same
  way every time. In transit it swings **10.3° peak-to-peak** during a turn
  (pen-tip proxy: ±10.9 mm vertical, 3.7 mm board-normal), which is why the
  pen must be clear of the board while driving, but that is a sequencing
  constraint, not a mechanical one.
- **A gravity latch cannot push — the levers are the same length.** The
  restoring torque is `m·g·d·sin θ` with d = `PEG_ABOVE_BODY` = 22 mm, and the
  moment arm of a *horizontal* pen force about the peg is the same 22 mm
  vertical offset (not the 60 mm horizontal reach — an arithmetic slip in the
  probe's first draft, caught because the measured angles did not match).
  Equal arms means any pen force buys a proportional, large rotation:

  | pen force | rotation | tip retreat |
  |---|---|---|
  | 0.10 N | −5.3° | 2.1 mm |
  | 0.25 N | −25.5° | 14.5 mm |
  | 0.50 N | −51.3° | 38.3 mm (flopped onto the fork) |

  And it *runs away* past ~0.25 N: rotating swings the tip further from the
  peg, growing the disturbing arm faster than `sin θ` grows the restoring one.
  A whiteboard marker wants 0.5–2 N. The latch gives out at 0.1 N. This is the
  mirror image of the milestone-6 finding that a wall cannot brace you: one-way
  contacts don't pull, and a gravity hook doesn't push. **The lean-pad is not a
  sway damper — it is the part that lets the tool exert force at all**, and it
  works by reacting the pen torque in compression at a ~50 mm lever instead of
  by gravity at 22 mm. Convenient sign: the pen's reaction rotates the module's
  lower back *toward* the robot, i.e. into the pad, so drawing load and preload
  push the same way.
- **⚠ The power contacts do NOT belong on the lean-pad** (correcting Parts.md).
  A lean-pad's preload is capped by the same weak geometry: `m·g·d / lever` =
  0.56 N absolute maximum for a 130 g module, realistically ~0.1 N at a few
  degrees of lean — well under what a pogo pin needs. Meanwhile the peg is
  *already* sitting in four V-notch plates carrying **0.43–0.47 N each,
  1.79 N total** of gravity preload (vertical components summing to the
  module's 1.276 N weight, as they should). So the electrical interface should
  be **the peg and the V-notches**: the preload is free, the seating slide
  wipes the contact clean, the peg is already the one metal part in the design
  (6 mm steel rod, Parts.md), and left-pair/right-pair gives exactly the two
  conductors a power-only coupling needs. `rack_charge_contact`'s "both pins
  touching" criterion maps over verbatim.

### The lean-pad: what it fixed, and what bounded its shape
Built on `pluggybot_fork.xml` as a rigid pad on the fork, bearing on the
module's lower back (guarded by `test_carried_module_can_exert_tool_force`,
shown failing at 51.3° first). Result, same probe as above:

| tool force | before | after |
|---|---|---|
| 0.10 N | −5.30° / 2.10 mm | −0.01° / 0.01 mm |
| 0.50 N | −51.32° / 38.27 mm | −0.06° / 0.07 mm |
| 2.00 N | −52.51° / 39.50 mm | −0.19° / 0.26 mm |

The response is also **linear now** (~0.095°/N) instead of running away, and
2 N still holds — past the ~1.5 N the peg-popping arithmetic predicted.

Three things decided the geometry, and only one of them was the lever:

- **The V self-centres so well that it erased the design problem I thought I
  had.** A fork-mounted rigid pad has to clear the module while the approach
  drives `PICK_OVERSHOOT` past the peg line, yet touch it after the lift.
  Measured across 0/2/4/6/8 mm of overshoot, the seated module lands at the
  same fork-local x **to 0.00 mm every time** — the seat has no memory of the
  approach. So the pad's target plane is a constant you can design against,
  and the conflict is resolved by *height*: the fork runs 22 mm low and lifts
  36 mm to latch, so a pad below the tray-hung module's bottom edge clears the
  approach entirely and the **lift** carries it into contact. Slide-in-then-
  lift, the coupling's own verb, doing one more job for free.
- **The parked envelope, not the lever, sets the depth.** Arm retracted, the
  fork tucks over the chassis with only ~60 mm between the chassis top (0.12)
  and the scanner's centre row (0.18). A first version reaching to z = −0.056
  put its bracket ON the chassis and jacked the whole fork *up* into the scan
  row — and this time contacts caught it (`chassis <-> lean_pad_arm`), because
  `tool_fork` is outside the `pluggybot`/`arm` exclude. Fix: route the bracket
  behind the module instead of under it, stop the pad at −0.036, and raise the
  fork mount 19 mm (`FORK_MOUNT_RAISE` 0.016 → 0.035, its second instalment).
- **The lever has a floor, set by the peg popping out of its own V.** Pad force
  is (tool torque / lever) and it is reacted at the peg, which sits in a 45°
  V. Push the peg harder than its own weight (1.28 N) and it rides up the
  flank and out. At the 29 mm lever that survived the envelope, tool force
  caps near 1.5 N by arithmetic and measured fine at 2 N. Moving the pad *up*
  shrinks that number fast — it is the constraint to check first if the parked
  envelope ever gets tighter.

One duplication bug fell out: `mission.py` had the lift preset re-typed as bare
`0.016`/`0.008` instead of importing `FORK_MOUNT_RAISE`/`DROOP_COMP`, so
raising the mount would have silently left it aiming at the old geometry. It
imports them now. A constant that describes geometry should have exactly one
home — the same lesson the rack-frame verdict taught, one level down.

### The module electrical interface: put the contacts where the force is
Built as a split peg — two conductors around an insulated centre, so the
fork's left and right V-notch pairs are the two poles (`hub/coupling.py`
`peg_xml` / `module_power_state`, demo `scripts/module_power.py`). Nothing
new was added to the coupling: the connector *is* the latch.

- **Continuity through a full errand is clean, and the failures are exactly
  where they should be.** Over the room_hub errand (navigate, pick, carry
  across the room, stow — 89.5 s), brown-outs by phase: pick 5 spans / worst
  14 ms, **carry 0 spans**, stow 5 spans / worst 50 ms. Every interruption is
  a mating or release transition, where one pole necessarily breaks before
  the other; while actually hauling the tool, the coupling never flickers.
  Under a deliberately harsher 1.2 rad/s spin (the pytest, not the mission)
  a pole opens for <1 % of steps, so hard spinning *can* flicker it.
- **Report brown-outs as durations in a window, not as a percentage over the
  run.** The first version printed "left 0.28 %, right 0.08 %" and that
  number is nearly meaningless twice over: 0.4 % as microsecond blips and
  0.4 % as one 200 ms outage are different hardware, and a run-wide figure is
  dominated by mate/release events that are not faults. The duration inside
  the carry window is what sizes a module's holding capacitor.
- **Reporting the poles separately is what makes the criterion useful.** A
  half-seated coupling — one conductor on, one off — is a real failure mode
  of any two-point latch (the feelers taught it: one prong 39 mm short
  wrecked everything while the other looked perfect), and a bare boolean
  files it under "off" with no way to tell a missing tool from a bad seat.
- A demo-authoring trap worth remembering: the bare-world version of the demo
  originally hauled the tool with 360° of spinning between `pick` and
  `put_back`, and the stow missed the trays entirely — `put_back`'s default
  travel assumes the pose `pick`'s retreat left it in, and that much turning
  drifts odometry well past the coupling's ±11 mm capture window. Same
  lesson as "fixed choreography does not survive navigation", met from the
  other side. The room errand computes its travel and stows fine.

### The drawing tool: five bugs, and only one was about drawing
Building the pen module's plot controller (`hub/drawing.py`, demo
`scripts/draw.py`). End state: a circle traced to **2.2 mm RMS shape error,
98 % inked**, with the tool still electrically seated afterwards. Getting
there cost five failures, each of which is a lesson this repo already had in
some other costume:

- **Teleporting the robot drops the tool.** `park_at_board` wrote the base's
  qpos to place it in front of the board — and the module, a free body held
  only by gravity in the fork's V, stayed behind in mid-air and fell. The
  giveaway was `dz/dlift` calibrating to exactly **0.000**: the pen was lying
  on the floor. Carried tools mean the robot has to *drive*.
- **A stiff position servo handed a step command is an impulse.** Raising the
  carriage gain to kp=2000 (a lead screw does not yield to pen drag) fixed a
  12 mm tracking error — and then an 80 mm setpoint jump threw the module
  clean out of the fork, both electrical poles opening as it yawed ~100°.
  Walking the same 80 mm in 5 mm steps held seated the whole way. The wheels
  have had `control.slew` since milestone 1; position setpoints need it too.
- **`on_fork` is a position heuristic, not a seating check.** It reported
  True throughout that ejection, because its tolerances are 3–4 cm. The
  electrical criterion caught it instantly. Written the same day the
  criterion was built, and still reached for the wrong one first.
- **Parent-child contact filtering is not grandparent filtering.** The pen
  carriage and shaft were modelled at the module plate's own x, i.e. buried
  inside it. MuJoCo excludes body↔parent contacts but the quill is a
  *grandchild*, so the shaft fought the plate: the carriage jammed at
  +21.9 mm and stopped tracking for most of a figure while its joint reported
  perfect command-following. The clearance sweep missed it because it checked
  the pen against the ROBOT and never against the module's own frame — the
  same "an envelope check is not the check you didn't write" trap as the arm.
- **A P-controller that stops commanding at the target overshoots when the
  command is rate-limited.** `_face` went in at −9.52° and came out at
  **+7.47°**, a 17° overshoot, because `slew` was still unwinding when the
  loop exited. Nobody noticed for a while because nothing before drawing
  cared about squareness — but the carriage sweeps 110 mm across the board,
  so a yaw error θ swings pen depth by 110·sin θ, and 7.5° is **14 mm**, most
  of the quill's entire travel. That is why one edge of the first square came
  out blank. Fix: settle and re-check, the same shape as refine_standoff.

Two things that are design, not bugs:
- **The pen needs its own compliance.** The wrist has RCC in y, z and both
  yaws but none along the approach axis, so pen pressure would be (arm
  position error) × the arm's 1200 N/m servo — 6 mm of overshoot is 7 N,
  against a peg that rides out of its V-notch past ~1.5 N. A sprung quill
  makes pressure a design constant instead. It must be **soft and
  long-travel**: a first attempt at 200 N/m engaged only ~0.5 mm and the pen
  lifted clean off the top of the figure (0 % ink there, 98 % at the bottom),
  because arm droop grows with lift height. At 60 N/m and ~10 mm nominal
  press the force holds 0.4–0.8 N across the whole figure.
- **Report shape error, not tracking error.** Same-instant error includes
  following lag, and a figure traced a fraction of a second behind schedule
  is still the right figure on the board — nobody looking at it can tell. On
  the good circle: 4.6 mm track, **2.2 mm shape**. Reporting only the first
  would condemn a drawing for being late.

### Decompose a drawing error before believing it
Ben, looking at `draw.png`: *"it drew a great square, it was just to the right
of where we wanted it."* He was right, and the metric was hiding it. Fitting a
rigid translation out of the trace (translation-only ICP against the commanded
polyline) splits the error into **where it landed** and **what shape it is**:

| figure | absolute | rigid offset | FORM (offset removed) |
|---|---|---|---|
| square | 10.35 mm | 17.47 mm (y −17.4) | **2.14 mm** |
| circle | 2.18 mm | 1.73 mm | **1.80 mm** |

So the plotter's *form* accuracy is ~2 mm for both figures — the square was
never distorted, it was displaced. That matters because the two have different
fixes and wildly different costs: an offset is one calibration constant, and
distortion is mechanics. The earlier writeup called the square "worse" and
went looking for a stiffness problem that was not there. Same lesson as the
33° standoff miss, which turned out to be 0.02° of odometry and all
estimator: **decompose before you fix.**

### Veer with a tool aboard: the counterweight still holds
Milestone 6's counterweight (battery at y=+0.06) was tuned against a measured
26 cm veer over 4 m, and it predates carrying anything. A module hangs on the
fork line (y=−0.05, ~0.28 m ahead of the axle), loading the same side the arm
does, so the calibration had to be re-checked against the heavier pen module.
Open-loop straight run, equal wheel-velocity commands, 2.69 m travelled:

| carrying | mass | veer | heading |
|---|---|---|---|
| nothing | 2.334 kg | −9.7 mm | −0.35° |
| LCD (130 g) | 2.464 kg | −13.6 mm | −0.49° |
| pen (182 g) | 2.516 kg | −15.2 mm | −0.55° |

The heaviest module costs **5.5 mm of extra veer over 2.7 m** — about 1.6× the
bare robot's, and two orders off the 260 mm that motivated the counterweight
in the first place. No re-tune needed; the tool-mass cap is bounded by the
coupling and the tip-load budget, not by veer.

(The first baseline run was invalid and worth recording: parked at y=0 it
drove straight into the drawing board — 0.32 m travelled, 52° of heading
change. Scenery in the test lane, the oldest hazard in this file, this time
scenery that I had added myself two hours earlier.)

Two measured facts left standing, characterised but not fixed:
- **Calibrate under load — but know what it fixes.** Pressing the pen shifts
  its home position **10 mm** sideways (module + wrist deflect), and
  re-zeroing there is what took the circle from 9.2 mm to 2.2 mm. It does
  *not* fix the scale: the quasi-static gain barely moves (0.999 → 0.992). I
  asserted it did, and the measurement said otherwise.
- **The real residual is kinetic** ⚠ *(partly superseded — see "The grip that
  leaked" below: much of this was MuJoCo's regularized-friction drift, and
  `noslip_iterations` halves the form error. Re-measure before trusting the
  numbers here.)* Sweeping with the pen down, the tip
  tracks the carriage at only **~0.81:1** against ~1:1 at rest — drag
  reverses with direction and yaws the module as it goes, so a calibration
  that settles before each reading cannot see it by construction. It shows up
  as the square's **17 mm rigid offset** against the circle's 1.7 mm — a
  square holds an extreme carriage offset for a whole edge, so the
  drag-induced displacement is one-sided and large. It does NOT distort the
  figure: form error is ~2.1 mm either way. Wants a stiffer yaw constraint on
  the module, or a calibration fit taken while sweeping rather than at rest.

## Sensor-realism pass: can stereo actually produce the map's scan?

The mapper has always been fed MuJoCo's ground-truth depth buffer. On hardware
that has to come from matching two eyes, so: OpenCV SGBM on the ACTUAL
rendered stereo pair from `room_hub`, centre row scored against the truth the
mapper gets today. 640×480, 60 mm baseline, f = 642 px.

Deliberately the **best case for stereo** — the two sim cameras are perfectly
parallel, coplanar and identical, with no rectification error, calibration
drift, exposure mismatch, noise or motion blur. This is an upper bound.

| pose | rays with any disparity | median error | p90 | worse than 100 mm |
|---|---|---|---|---|
| mid-room, facing the rack wall | **49.7 %** | **593 mm** | 1881 mm | 80 % |
| corner, facing a bare wall | 84.4 % | 218 mm | 1215 mm | 66 % |
| close to the rack | 84.5 % | 64 mm | 64 mm | 0 % |

And the geometry says the same thing without any images (σ_d = 1 px):

| true range | stereo σ_z | vs the 50 mm grid cell |
|---|---|---|
| 1 m | ±26 mm | half a cell |
| 2 m | ±104 mm | two cells |
| 5 m (the scanner's max range) | ±649 mm | **thirteen cells** |

So real stereo **cannot** produce the scan the occupancy grid is built from.
Mid-room, half the rays do not exist at all and the survivors are 0.6 m out at
the median — the mapper would paint walls a metre from where they are, which
is precisely the odometry-slip "jail bar" corruption class documented above,
arriving by a different route. The one good pose is the one aimed at the rack,
because the rack is close and carries AprilTags: **texture**. The rest of the
room is flat painted surfaces, the classic no-disparity case.

Two structural facts that came out of the same look:
- **`scanner.py` was never stereo.** It renders a depth image from ONE camera
  (`left_eye`) and takes the centre row — 320 rays in a horizontal band. The
  mapping has always been a 2D laser scan wearing a camera's clothes.
- **`right_eye` is vestigial.** Nothing algorithmic reads it: only `viz.py`'s
  screenshot panel, `stereo_snapshot.py`, and the parallax test in
  `test_cameras.py`. The project has been carrying a second camera in its
  parts list that no behaviour depends on.

Meta-lesson, and an uncomfortable one: this gap survived seven milestones
because ground-truth depth *always works*. A sensor that never fails cannot
teach you which of your behaviours depend on it. Milestone 5 learned this
about detector evaluation (`the generator's own val split cannot measure the
detector`); the same trap had been sitting under the mapper the whole time,
one layer lower and much better hidden.

## The claw module (milestone 8): the fourth tool

Design decided by measurement before anything was drawn, and the measurements
overturned the intuition that prompted them. The idea started as a claw angled
downward to reach under the chassis; the worry was mass and centre of mass.

- **The chassis was never the limit.** 800 g hung at the tool peg drops
  drive-wheel load only **15.0 → 12.6 N**; the robot's CoM sits ~68 mm ahead
  of the axle against a caster at 180 mm, so static forward tipping needs
  about **5 kg**. The obvious worry was the wrong one.
- **The COUPLING is the limit, and it is a moment limit.** Unseating tracks
  `W × L` almost exactly — 0.39 N·m holds, 0.59 N·m does not — so the gravity
  latch takes about **0.45 N·m** of pitch moment, full stop:

  | forward reach L | 200 g | 400 g | 800 g |
  |---|---|---|---|
  | 0 mm | ok | ok | **ok** |
  | 100 mm | ok | ok | unseats |
  | 150 mm | ok | unseats | — |

  Reach is far more expensive than mass. Hence a pendant straight down the
  peg's own axis (L = 0), not an angled arm. (Pre-measurement arithmetic said
  ~25 g at 150 mm; the truth is 200 g. Modelling the pad force as capped by
  the module's own weight was too crude — the four V-plates resist more.)
- **The robot cannot see what it is picking up.** Nav camera 180 mm up, 41°
  fovy → the floor leaves view inside **0.48 m**, while the grip point is
  244 mm ahead of the axle. The grasp is necessarily open-loop from a
  memorised pose, exactly as the socket vanishes from the dock camera at close
  range. Finding floor objects is unsolved: the LIDAR plane is 223 mm up and
  sees nothing on the floor at all.

### Five failures building it, and three were the same lesson
Verified end state (`scripts/pickup.py`): fetch the claw from bay D, power it
through the coupling, aim to **1.9 mm lateral / 3.0 mm forward**, grip, and
lift the block **99.6 mm** off the floor with the module still seated.

- **The structure carrying a gripper must not occupy the gripper's space.**
  The drop tube ran 28 mm INTO the grip zone: descending, it reached the block
  first, shoved it 17 mm forward, and the jaws closed on air — while a graze
  on the way past still read as "both pads in contact". Fixed, then broken
  *again* when the pad height changed and the pendant constant did not follow.
  It is derived from the pad height now, and pytest-guarded.
- **Ramp every position setpoint. Every one.** Three separate axes, all the
  same bug the pen carriage first taught: a 184 mm lift step threw the module
  clean off the fork (both poles open); slamming the jaws shut batted the
  block away before contact settled, running the closure to its full 27 mm
  stop on a 26 mm object that should have halted it at 18 mm. `control.slew`
  has existed for the wheels since milestone 1.
- **Converge a height, do not correct it once.** The grip follows the lift
  command at ~0.87:1 (droop and lean both change as the arm descends), so one
  correction left the pads 11.6 mm high — a grip that *looked* right, held the
  block by its top 12 mm, and lost it the moment the lift took the weight.
- **Carry clearance is sized by SWING, not by static height.** 46 mm of
  clearance was "clear of the floor" and was not: a 172 mm pendant swinging
  the 10° a carried module was measured to swing moves its tip ~30 mm.

### The grip that leaked: a solver artifact, and a bad inference
The claw gripped and lifted but lost the block during the carry. I called it
"not slipping", inferring that from *tripling the grip force changed nothing* —
which does not follow, and I never plotted the block's height in the jaws'
frame, the one measurement that settles it. **Ben watched the render and said
the block was visibly creeping down and out.** He was right.

Measured properly, block height relative to the grip during a lift: −9.2 →
−13.6 → −20.2 → −27.9 mm, then gone. A steady creep of roughly **8 mm/s**.

Two sweeps ruled out both obvious causes. Squeeze depth:

| jaw command | clamp force | slip |
|---|---|---|
| −19 mm | 0.65 N | −99 mm (lost) |
| −21 mm | 2.40 N | **−15.98 mm** |
| −23 mm | 4.00 N | **−15.98 mm** |
| −27 mm | 7.20 N | **−15.98 mm** |

Identical to 0.01 mm across a 3× force range, against a friction capacity of
10.8 N holding a 0.59 N block — an 18× margin. And lift speed:

| lift speed | slip |
|---|---|
| 0.050 m/s | −15.98 mm (kept) |
| 0.020 m/s | −99 mm (lost) |
| 0.003 m/s | −99 mm (lost) |

**Slower was worse**, which no real friction failure does: a fast lift simply
finished before the block crept out.

That is the signature of MuJoCo's **regularized friction**, which drifts under
sustained load. `noslip_iterations` is the engine's post-solve pass for exactly
this symptom: **3 iterations take the slip from −99 mm to +0.03 mm**, and 10 is
no better. The physical design was adequate all along, and the apparent "lift
ceiling" at 0.193 dissolved with it — it was never a ceiling, it was a clock
measuring how long the grip had been held.

**But it must be scoped to the hold, and not for the reason I first thought.**
It is expensive (2.7× step time, 0.35 → 0.97 ms), which was the original
argument — and then the full suite failed and gave the real one: **the pass
BREAKS the tool coupling.** Measured on a claw pick:

| | on fork | powered | bay error |
|---|---|---|---|
| noslip = 0 | True | **True** | 3.1 mm |
| noslip = 3 | True | **False** | 5.0 mm |

The module lands on the fork but never seats electrically. That is coherent
rather than mysterious: **the peg seats by SLIDING into its V-notch** — the
self-centring the entire gravity latch depends on — and this pass exists to
suppress sliding. Grasping wants no slip; the coupling wants slip. Opposite
requirements, and they must never be on at once.

So it lives in `ClawTool.grasp_physics()` / `PenPlotter.contact_physics()`,
enabled for the hold and cleared before any swap. Precedent — the codebase
already tunes solver fidelity per phase, dropping the timestep to
`SWAP_TIMESTEP` for mm-scale peg contacts and restoring it afterwards.

⚠ Setting it in `__init__` looked equivalent and was not: a module-scoped
pytest fixture carried the mutated model into a LATER test's coupling pick,
which is how the conflict surfaced at all. A solver mode is global state; turn
it on for a phase, not for an object's lifetime.

### …and the conflict was a friction bug wearing a solver costume
Ben asked whether toggling a solver mode per phase really represents physics.
It does not, and asking was the right instinct — the toggle was covering an
error. **The peg and V-notches were running on MuJoCo's DEFAULT friction of
1.0**, whose friction angle is 45° — *exactly* the V's flank angle. The peg
sat right on the sliding threshold and barely self-centred, so it depended on
the solver's tangential drift to seat at all, and suppressing that drift broke
it. Steel on printed plastic is μ≈0.4, and the coupling **spike has always set
0.4** while the generated hub world silently used 1.0: the measured ±4 mm
envelope was never the one in play.

| peg μ | noslip | on fork | powered | bay error |
|---|---|---|---|---|
| 1.0 | 0 | ✓ | ✓ | 3.1 mm |
| 1.0 | 3 | ✓ | **✗** | 5.0 mm |
| 0.4 (honest) | 0 | ✓ | ✓ | 4.2 mm |
| **0.4** | **3** | ✓ | **✓** | 3.5 mm |

With honest friction there is no conflict in either direction. `priority="1"`
on the peg is what makes it stick — MuJoCo combines pair friction as the
elementwise MAX, so a low friction without priority does nothing at all. The
caster lesson, third outing.

**And it moved a hardware number.** A lower-friction peg genuinely shifts more
in its V under hard driving, so electrical continuity got worse — honestly
worse. Re-measured over 11.7 s of deliberately harsh maneuvering: 20 dropout
spans, **worst 178 ms**, 289 ms total. The module's holding capacitor must
therefore cover ~**200 ms**, not the 50 ms the release transient suggested.
The regression test now judges the WORST OUTAGE rather than a percentage of
steps, which is the metric `module_power.py` had already argued for and which
the old test had not adopted.

**And it was not only the claw.** The pen module drags a tip across a board,
which is the same sustained tangential load. Final figures with the pass on:

| figure | | ink | rigid offset | FORM error |
|---|---|---|---|---|
| square | before | 63 % | 17.47 mm | 2.14 mm |
| square | **after** | **94 %** | 15.71 mm | **0.79 mm** |
| circle | before | 98 % | 1.73 mm | 1.80 mm |
| circle | after | 99 % | 1.30 mm | 1.84 mm |

The SQUARE improves dramatically and the circle barely moves — which is exactly
what a creep-under-sustained-load explanation predicts, because a square holds
an extreme carriage offset for whole edges while a circle is always moving. The
plotter's form accuracy is now **0.79 mm**. The rigid offset barely shifts, and
that is the informative half: the offset is a genuine calibration error, and
only shape fidelity was being eaten by the solver. The "kinetic gain loss"
recorded against the pen module was largely this, not module yaw.

Two meta-lessons, both uncomfortable:
- **A negative result is not a diagnosis.** "More force didn't help, therefore
  not friction" skipped the step where you look at the thing actually moving.
  The fix came from someone watching the video.
- **It nearly got tuned away instead.** The tempting move was to raise μ until
  it stuck, which would have buried a solver bug under a dishonest friction
  coefficient — the schuko-chamfer mistake in a new costume.

### The angled arm and the tool's own eye
Ben asked for the claw to angle down-forward after the coupling and to carry
its own camera, keeping the dock camera free for looking ahead. Both landed:

- **55 mm of forward angle costs 7 % of the coupling budget.** 60 g at 55 mm
  is 0.032 N·m against 0.45 N·m. Reach is only expensive when it is LONG —
  400 g at 150 mm was 0.59 N·m and unseated the module. What actually bounds
  the angle is the RACK, not the moment: modules hang business-end-inward, so
  the arm points at the wall when stowed, and there are exactly **80 mm**
  between a racked module's front face and the wall.
- **`claw_eye` is the first camera on a TOOL rather than the chassis**, and it
  costs no CSI port: module data already crosses the coupling wirelessly, so
  a tool camera is free where a fourth chassis camera would need a
  multiplexer. Only the claw pays for it.
- **Camera placement took three renders, not one calculation.** Mounted above
  the arm's root it saw mostly its own arm and clipped the block once lowered;
  moving 44 mm outboard helped; mounting it *partway down the arm* put the arm
  BEHIND it and halved the working distance to ~65 mm. Aim is computed by a
  `_look_at()` helper rather than hand-typed direction cosines — a camera
  bolted to a swappable tool aims at a point fixed by other constants, and
  hand-derived cosines rot silently when one of those constants moves.
- **The angle has a cost worth recording:** it puts the grip 285 mm ahead of
  the axle instead of 244, so any heading error is amplified 17 %. Measured
  aim went from 2.0/3.5 mm to 9.3/13.0 mm (lateral/fore-aft). Still inside
  capture, but the fore-aft margin is thin — the pads are only 28 mm deep.

⚠ A test threshold moved for a good reason and it is worth being explicit: the
aim assertion was a single 15 mm RADIAL tolerance, which hid that the jaws
capture a 62 mm lateral gap but only 28 mm fore-aft. It is per-axis now. The
grasp itself remains the real criterion.

### One solver policy, or two? (the toggle, revisited)
Ben pushed back on the per-phase `noslip` toggle: is that unavoidable, or is
there an implementation with one policy? Measuring it was worth it — the
answer is mostly yes, and the reasoning I had written down was already stale.

- **The conflict was gone and I had not noticed.** "The coupling needs noslip
  OFF" was measured at the peg's *wrong* μ=1.0. At the honest μ=0.4 the
  coupling picks, powers and returns correctly at noslip 0, 1 AND 3. So the
  toggle stopped being a correctness requirement the moment the friction bug
  was fixed; I left the old justification in CLAUDE.md, where a future session
  would have believed it.
- **The claw needs no solver mode at all.** A hard contact constraint on the
  two jaw pads (`GRIP_SOLIMP = 0.99 0.999 0.0001`) removes the creep at the
  source: slip over a 100 mm lift goes **−21.7 mm → −0.13 mm**, against
  +0.28 mm for the noslip cure. It is also the better model — a rigid printed
  pad on a rigid block should not squash — and it follows the house pattern of
  putting contact behaviour on the geoms that need it (`condim="1"
  priority="1"` on the caster; friction+priority on the peg).
- **The plotter still needs the pass, and I could not find its equivalent.**
  Hardening the pen tip changed nothing (square: 63 % inked either way) and
  hardening the peg-in-V barely moved it (64 %, form 1.69 mm), against 94 %
  inked / 0.79 mm form with the pass on. Whatever the pen loses is not
  concentrated in one contact pair. Recorded as an open question rather than
  papered over.
- **Cost is binary, not per-iteration**: 0.333 ms/step off, 0.928 at noslip=1,
  1.070 at noslip=3. Turning the pass on at all is the expense; extra
  iterations are nearly free.

Net: one policy for everything except drawing, and the drawing's use of it is
now a cost trade rather than a correctness hack — nothing breaks if it is left
enabled anywhere.

## Vectorizing the occupancy-grid scan update (issue #2)

The per-scan ray loop was the most expensive Python in the mission loop
(~9.4 ms per 10 Hz scan, flagged in the Pi-budget profile as the cheapest
thing to optimize). Rewritten as one (ray × sample) numpy batch — all rays'
sample distances in a 2D array, shorter rays masking off their tail, one
weighted `bincount` accumulating all the evidence. Measured on the recorded
room_hub LIDAR fixture: **9.4 ms → 1.3 ms per scan, 7.4×** (guarded by
`tests/test_grid_vectorization.py`, which keeps the old loop as its reference
implementation and requires ≥5×).

- **The old loop's semantics hid in a numpy footnote.** Its per-ray
  `grid[iys, ixs] += L_FREE` used fancy indexing, and fancy-index `+=` counts
  a duplicated index ONCE — so a cell sampled twice by the same ray (the res/2
  oversampling guarantees this) got one vote per ray, while separate rays
  accumulated normally. A naive vectorization that counts every sample doubles
  most free evidence: 47.8 % of cells wrong, up to 3.6 log-odds. The
  vectorized version reproduces the once-per-ray rule by deduping consecutive
  samples — legitimate because a straight ray never re-enters a cell, so
  repeats are always adjacent.
- **"Identical" has a stated tolerance: the cells are bit-identical, the
  values differ in ulps.** The batch picks exactly the same sample positions
  and truncations as the loop (matching `linspace` term-for-term, endpoint
  pinned), so the same cells get the same votes — but adding `k·L_FREE` once
  is not bit-equal to adding `L_FREE` k times, so values differ at the 1e-15
  level. The test asserts atol 1e-9 and exact equality of the thresholded
  `to_image()`. End-to-end the difference vanishes: `explore.py --headless`
  produces a byte-identical `map.png`, same 67.5 s termination, same 25 362
  known cells, same 54 blacklisted frontiers; the full hub lifecycle still
  closes (2 swaps, 1 charge, 0 chassis contacts).
- Two micro-optimizations worth remembering (each ~0.3 ms here): int32
  indices viewed as uint32 make one compare cover both bounds (a negative
  index wraps to a huge unsigned value, so `< cols` rejects it too), and a
  single weighted `bincount` replaces two count-then-scale passes over the
  grid.
- Pi-budget consequence: the 42 ms Pi-5 estimate for this stage drops to
  ~6 ms, and the bigger PluggyWorld home world (~9× the cells, longer rays)
  now scales through numpy instead of a Python loop.

## One always-on solver policy (issue #3): noslip loses, the wheels confess

PluggyWorld puts two robots in one world, so solver settings can no longer be
phase-scoped per robot — whatever `noslip_iterations` is, it is for everyone,
all the time. The plan of record was "one realistic noslip policy, on all the
time"; `scripts/noslip_spike.py` measured every behavior with a stake in it
(coupling rig, schuko rig, jittered robot-driven swap, grip hold, pen square,
step cost) under each candidate, and the plan of record was wrong in both
halves. All figures are the SQUARE (the creep-sensitive figure); "robot swap"
is a 9-point hand-off jitter grid (±3 mm × ±1°), open-loop from
`place_at_standoff`:

| noslip | rig cycle | schuko | robot swap clean | pen ink | pen form | ms/step (room_hub) |
|---|---|---|---|---|---|---|
| 0 (was shipped) | 5/7 | 5/5 | 5/9 | 63 % | 1.94 mm | 0.210 |
| 1 | 5/7 | 5/5 | 3/9 | 72 % | 1.84 mm | 0.410 |
| 2 | — | — | 3/9 | 92 % | 0.60 mm | ≈0.41 |
| 3 | 6/7 | 5/5 | 1/9 | 93 % | 0.61 mm | 0.422 |
| **0 + wheel brake** | 5/7 | 5/5 | **5/9** | **99 %** | **0.60 mm** | **0.210** |

- **Always-on noslip is worse than useless at the system level.** The earlier
  all-clear ("the coupling picks, powers and returns at noslip 0, 1 AND 3")
  was measured ALIGNED, and it still holds there — but under ±3 mm of lateral
  hand-off jitter every noslip ≥ 1 run produces the half-seat failure: module
  on the fork, **not electrically powered**. At 3 iterations two corner cells
  miss the tray by ~35 mm on the return. The peg seats by sliding, the pass
  exists to suppress sliding, and honest friction only rescued the aligned
  case. The bare rig said noslip 3 was *fine* (6/7, its best row) while the
  robot said 1/9 — measure a solver policy on the full system, not the rig.
- **The plotter never needed the solver — and mostly it was not solver drift
  at all.** The clue was in the old note all along: "whatever the pen loses
  is not concentrated in one contact pair." It is not in any contact — the
  parked base **rolls**. A wheel velocity servo commanded 0 resists *speed*,
  not force, so a sustained ~0.5 N of pen drag walks the whole robot
  sideways a few mm per figure, and the pen starves of ink. The robot has no
  parking brake — and the physical robot has one for free: gearbox Coulomb
  friction. `frictionloss="0.05"` on the wheel joints (the static sibling of
  the `damping="0.05"` that already models the 37D's 65 % efficiency) draws
  a **better** square than the 2× step-cost noslip pass ever did: 99 %
  inked / 0.60 mm form, at zero extra cost.
- **A wrong fix taught the right lesson.** The first cure attempted was
  hard `solimp` on the tire *contacts* (the claw-pad pattern). It fixed the
  pen (90 % inked) — and then two mission tests failed, and the trail led
  somewhere unexpected: mm-scale wheel slip is baked into the swap's
  feed-forward travel constants. Hard tires slip ~0.7 mm less over an
  approach, which pushed the front-heavy pen module's pick onto a knife
  edge; and during the stow's LOWER phase the base rolled 9 mm forward
  (wheels turning, believed and true advancing in lockstep — a genuinely
  *rolling* walk, which is how the real mechanism was finally caught),
  jamming the peg past the tray into the bracket crevice, where the retreat
  then spun the wheels against the trapped fork for 583 "collision" steps.
  The brake fixes the rolling at its source; the tires stay stock, and the
  travel constants stay exactly as calibrated. `PICK_OVERSHOOT`'s comment
  now warns that it implicitly contains wheel slip.
- **The mission was rolling dice, and every physics tweak rerolled them.**
  Across three configurations that differ by under a millimetre of wheel
  behavior, the full mission failed three DIFFERENT ways: the return jam
  above (hard tires), a pick whose tag-ranged terminal travel scattered
  −5 to +11 mm against truth on otherwise identical runs (vs a ~±11 mm
  capture window), and — the last one standing — the pick leg's `drive_to`
  returning "no-route" four seconds in, because the opening look-around's
  map coverage is pose-marginal and the planner ran out of known cells.
  None of these are new defects; they are pre-existing single-attempt
  fragilities that one lucky trajectory had been threading. The fix is the
  repo's own doctrine applied one level up, twice: `swap_at_bay` now
  VERIFIES its outcome (electrical seating for a pick, hung-in-bay for a
  return — the criteria that already existed) and takes another run at it
  with a freshly-ranged travel; and a route failure triggers a look-around
  spin and a re-plan (milestone 4's rule, verbatim) instead of giving up.
  A retried measurement is a new draw from the error distribution; a
  retried constant is the same error again — retry the measurement.
- **The verdict:** `noslip_iterations = 0`, always, everywhere; no code
  mutates solver options at runtime (`contact_physics`/`grasp_physics` are
  deprecated no-ops); creep under sustained load is fixed at its actual
  source, per-part — `GRIP_SOLIMP` where a *contact* drifts (jaw pads),
  `frictionloss` where a *joint* rolls (wheels). Guarded by
  `tests/test_noslip_policy.py`: every world loads at 0, the old toggles are
  inert, the wheels keep their brake, and a slow end-to-end square must ink
  >85 % with no solver help (it inks 63 % without the brake — shown failing
  first). The 2× step cost the always-on plan had budgeted for is simply
  not spent.
- **Discovered en route, recorded as open:** the open-loop hand-off envelope
  is narrower and more asymmetric than the "±3 mm / 1°" note suggests —
  −1° yaw picks fail at EVERY policy (the fork under-reaches the peg), and
  the (−3 mm, +1°) return misses the tray by ~34 mm at noslip 0 and 3 alike.
  This grid bypasses the tag-servo standoff refinement that real missions
  run, so mission-level margins are better than these; but the yaw asymmetry
  is real and pre-existing, and nobody had ever swept the corners before. A
  mission-level jitter sweep is the follow-up if the margin ever matters.

Meta-lesson, the second time in this file: a solver mode that "fixes" a
behavior is a claim about WHICH part is misbehaving, and it does not name the
part. The noslip pass fixed the drawing while the real motion was the wheels
rolling under the chassis — exactly as it once "fixed" grasping while the
real bug was the peg's friction coefficient. And the first replacement fix
repeated the mistake one level down: hardening the tire contact treated the
symptom's location, not its mechanism, and only the phase telemetry that
showed believed-and-true advancing *in lockstep* (rolling, not slipping)
named the true culprit. Decompose before you fix — and a velocity servo is
not a position hold.

### The brake's bill: static friction creates a control deadband
Ben watched `pickup.py --view` and saw the robot fetch the tool, turn toward
the block, and then *stall for most of a minute* before creeping in. Measured:
`_face` took **55 s** of sim time to settle 0.23° on the braked wheels, 9.5 s
before the brake. The mechanism is the flip side of the parking brake: the
wheel servo's torque is `kv·(target − actual)`, so a commanded wheel speed
under `frictionloss / kv` (= 0.1 rad/s here) cannot move a stopped wheel *at
all* — and every P-turn controller in the repo shrinks its command toward
zero as the error shrinks, parking itself squarely inside that deadband. The
fix is what real motor drivers ship as deadband compensation:
`control.turn_command` floors every nonzero turn command at `W_BREAKAWAY`
(0.08 rad/s of body yaw, above the ~0.05 breakaway) — commanding less than
breakaway is indistinguishable from commanding zero, so the floor costs
nothing. `_face` is back to ~10 s and the whole claw approach went 76 → 19.5 s
(the unbraked robot's figure is 17). Guarded by a unit test on the floor and
a timed end-to-end facing test. Lesson: every honest piece of physics added
to the model sends a bill to the controllers that were tuned without it —
stiction's bill is a deadband, and it must be paid in the controller, not by
removing the physics.

## The home world (issue #6), and a stow gap it exposed

The generated house+garden (`pluggybot.home.world` → `models/home_world.xml`
+ `home_world.meta.json`) is 12 × 8 m against room_hub's 8 × 8, with a
second room, a fenced garden, two wall-mounted whiteboards and the rack on
the living room's south wall. Three things it taught:

**1. Put the origin inside a room.** The first layout ran the house from
(0, 0) to (7, 8), which is tidy on paper and wrong in practice: a bare
`MjData` spawns the robot at the origin, so every plain model load greeted
us with the chassis wedged in the south wall — 78 contacts at settle.
Shifting the plan to (−2, −2)…(5, 6) put the origin in open living-room
floor. Guarded by `test_robot_spawns_clear_of_the_geometry`.

**2. Bay C and D were never rangeable off the rack tag.**
`_terminal_travel` preferred the RACK's marker, which hangs at the rack's
centre — visible from the two inner bays and *outside the dock camera's
view* from the outer ones. So bays C/D silently fell back to odometry,
whose measured +13…21 mm drift a PICK forgives (the V self-centres during
the lift) and a RETURN does not. Fix: range off the **bay's own** marker
first, which sits on the approach axis at every bay
(`TagSpotter.bay_range`). Measured at bay C: commanded travel vs truth went
from 13–21 mm out to **≤ 2 mm**.

**3. A failed `put_back` left the lift at RELEASE height, and the retry
inherited it.** `put_back` computes every height relative to the lift it
starts with, so attempt 2 began 50 mm low and drove the peg into the tray
flanks. `swap_at_bay` now restores the entry lift before a retry — the
retry is a repeat of the attempt, not a different maneuver.

**⚠ Open item: the PEN module does not stow after a navigated errand.**
With both fixes in, bay C's return still ends with the module on the fork
(peg ~13 mm short of the tray line, riding the fork down through the
release instead of transferring). The discriminating experiment:
**it fails identically in `room_hub`** — same numbers to the millimetre —
so this is *pre-existing*, not something the home world introduced. It was
simply never exercised: `draw.py` works in the bare `hub_world` where there
is no navigation, and `hub_lifecycle` stows at bay A. Deliberate overdrive
(the pick's `PICK_OVERSHOOT` remedy) does **not** help — 6 mm and 12 mm of
extra travel moved the final module position by 6 mm total, which says the
robot is *bottoming out*, not stopping short. The prime suspect is the pen
module's own geometry: unlike the LCD, it carries a rail/carriage/quill
assembly standing ~26 mm proud of its plate (`PEN_MOUNT_X`), which is the
face that goes toward the rack. Next step is a bare-world clearance sweep
of a pen return (no navigation), the same shape as the original coupling
spike. Until then: **fetching and drawing with the pen works end to end in
the home world; stowing it afterwards does not**, and `scripts/home_draw.py`
reports the stow honestly rather than claiming success.

## Debugging workflow that worked

1. Reproduce headlessly with printed telemetry (pose, wheel ω, contact list, `ncon`) — vibes don't bisect.
2. **Render a filmstrip** (offscreen `Renderer`, tracking camera, 12 tiled frames) when numbers confuse — "it's standing on its tail" was invisible in scalars.
3. Bisect one variable at a time, and assert that programmatic XML patches actually applied (`assert count == 2`) — a silent no-op replacement once produced identical "before/after" results and nearly a wrong conclusion.
4. When symptom-fixes keep trading one failure for another, stop tuning and **measure the force balance directly**: `mj_contactForce` per contact, summed as torque contributions about the COM, named the caster as the yaw-brake in one run — after days of plausible-but-wrong theories (integrator energy, tire stiffness, wheel radius resonance, servo feedback).
5. **Consult reference models** (MuJoCo Menagerie — Stretch for diff-drive). The caster `condim="1" priority="1"` idiom was sitting in their XML all along; professionally-tuned models encode solved problems.
