# The rover

The wheeled body — a differential-drive base on a caster, a mast, a lift and
a telescoping arm ending in a fork — is the deployed robot until the
quadruped replaces it (#375, step 2). **Read this before touching the
rover**: the drive and dead reckoning (`control.py`, `odometry/`,
`behavior/`), the swap, dock and bays (`rack/`, `mission/`), the tools on
the lift (`tools/`), its sensors' mounts, its energy numbers and its routes.
Each bullet is a constraint, its number and where its story lives (SimNotes,
mostly); what outlives the body is in CLAUDE.md. Stage C of #376 deletes this
file, after tagging the last commit that runs the rover `rover-final`.

## Drive, turn and square up

- **Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm
  ahead.** A routine yields one `(v, w)` drive command per physics step
  (CLAUDE.md, "Every manoeuvre is a ROUTINE").
- **A final approach uses `drive_toward(..., slow_radius=R)`; a path waypoint
  does not.** The default pure-pursuit law cannot converge on a destination
  closer than its own overshoot and ORBITS it (~900° of turning per 200 mm
  hop); terminal mode adds a hard ±25° cone (`v` exactly 0 outside it — a soft
  taper alone does not kill the orbit) plus a distance taper.
  `tests/test_navigation.py` pins both the fix and the defect.
- **A terminal loop has a budget, and squaring up is `control.square_up`**
  (issue #108): `FACE_BUDGET_S` = 30 s (~3× the worst healthy case) with an
  explicit `squared` answer. An empty pack does NOT stop the body (motors draw
  ~30 W at 0 Wh) and every mission guard is checked BETWEEN errands, so an
  unbounded loop drains the pack and runs past `max_sim_time`. A bound is not
  a recovery — that is #107's death. `refine_standoff`'s drive back in is
  `REFINE_BUDGET_S` 10 s (issue #339): unbounded, a robot knocked over
  mid-pick drove at the standoff through its death AND every stand-up after
  it, into a wall until flat — thirteen lives on the deployed pair. A give-up
  is `refine_blocked`, and the swap and the charge approach then take NO
  attempt from there (`blocked`): a fork deployed off the line pushes a module
  off its trays, which is what the refine is for.
- **Setpoints are ramped** (CLAUDE.md's rule): `control.slew` for wheels;
  `ClawTool.set_lift`/`jaws` and `PenPlotter.ramp` for the rest.

## The wheels and the solver

- **`noslip_iterations` is 0 everywhere (CLAUDE.md), so creep is fixed at its
  source, per part** (issue #3; sweep table in SimNotes): wheel-joint
  `frictionloss` where a JOINT rolls — the parking brake; the plotter's square
  went 63 → 99 % inked — and `coupling.GRIP_SOLIMP` where a CONTACT drifts
  (the jaw pads, −21.7 → −0.13 mm). The caster is `condim="1"` `priority="1"`:
  at equal priority its low friction did nothing (the MAX rule; SimNotes, "THE
  caster lesson"). ⚠ The swap's travel constants contain mm-scale wheel slip:
  anything touching wheel contact or joint friction must re-verify the bay-C
  pick and the mission stow. ⚠ The brake creates a stiction DEADBAND (commands
  under `frictionloss/kv`, 0.1 rad/s, move a stopped wheel not at all), so
  P-turn controllers go through `control.turn_command` (breakaway floor).
  `PenPlotter.contact_physics` / `ClawTool.grasp_physics` are deprecated
  no-ops; `tests/test_noslip_policy.py` guards all of it.

## Dead reckoning: presses, stalls, the anchor, a level chassis

- **A press is not travel** (`HubSwap.pinned`, `HubSwap.pressing`; issues #22,
  #94). Wheels held against something immovable pump imaginary travel into
  dead reckoning (828 mm from one charge press; 4.28 m in 30 s against the
  fence). `charge()` sets `pinned` for the press and clears it for the undock;
  an UNDECLARED press is caught by the bumper — a chassis contact on the side
  the wheels are turning toward (judged against the encoders, not the command)
  holds the reckoner, held 50 ms past the last contact because a cruise-speed
  press bounces. Motor torque does NOT separate a press from a cruise (0.44 vs
  0.35 N m). The charge creep stalls on `CHARGE_PRESS_STALL_S` (4 s), not the
  swap's 0.4 s. Two lessons ride with it: a plausibility guard can reject the
  truth (`mission.plausible_travel` is a damage limiter, not a fix), and more
  map can make an estimate worse (`RackFinder` KEEPS a well-conditioned facing
  because driving behind the rack turns it into a free-standing partition).
  `scripts/stall_spike.py`; SimNotes "A stalled drive is an odometry pump".
- **The dock is also the anchor** (issue #42): dead reckoning is corrected in
  exactly one place, `HubMission.anchor_at_dock`, when both pins conduct — a
  pose the robot occupies to millimetres by construction. ⚠ Snapped to the
  COMMISSIONED PRIOR (`rack_prior`), never the believed rack (anchored to the
  belief, the error tracked itself 0.003 → 0.344 m over four sim-hours). Two
  belief rules travel with it: the rack landmark merges BY DECODED IDENTITY,
  never by distance, and its position is recency-weighted (`RACK_RECENCY`).
  The bay recovery (`swap_at_bay`'s spin-refresh-retry) depends on both;
  `test_the_recovery_finds_a_bay_the_first_look_lost` pins all three. The
  pen's ERASE rides the first successful press, never arrival. Map evidence
  decay is DEFERRED on measurement.
- **A scan goes into the map only while the chassis is level**
  (`HubMission.level`, `MAP_TILT_RAD` 1.5°, issue #339). On its side the LIDAR
  sees the sky, and "free to max range" painted 8 m of free space through
  every wall in reach — a dead robot keeps scanning, and the map outlives a
  stand-up (SimNotes, "A robot on its side maps the sky"). Past 1.6° the scan
  plane meets the floor inside the 8 m range; errands peak at 0.66°, the
  charge creep's bumper contact 1.4–1.7° for ~20 ms (one scan skipped per
  dock), and a 21 mm plate pad crossing (4.3–8.1° for ~2 s) is skipped — it
  painted floor arcs. The rack finder and the height map take the same gate;
  the front-stop reflex still reads every scan.

## The dock and the bays

- **The dock is measured, not believed** (issue #32):
  `HubMission.charge_approach` measures the standoff off the charge tag's PnP
  pose, creeps under servo and verified-retries (`scripts/charge_spike.py
  --blind` reproduces the old rows, which die at ~6 cm lateral / ~10°
  heading). ⚠ `dock_eye` rides the FORK LINE — the charge servo holds the tag
  at `-PLUG_LATERAL`, not centred, because charging aligns the CHASSIS — and
  it rides the LIFT (the approach commands `CHARGE_LOOK_LIFT` before its first
  look). A failed dock is narrated `stranded`, never "mission complete", and
  is a `stuck` death.
- **...and so are the bays** (issue #30; `_measured_standoff` is the shared
  core): `HubMission.bay_fix` measures the bay standoff off the bay's own tag
  inside `swap_at_bay`'s retry loop (`scripts/swap_spike.py --blind` drops the
  module at 4–8 cm across or −3° of heading). ⚠ A measured standoff's FACING
  comes off the rack's tags TOGETHER (`localize.fit_rack_facing` over
  `coupling.RACK_TAG_FACES`, issue #88; `HubMission.fix_source` says which
  answered), never off one tag's PnP yaw — square-on, a single 30 mm tag's yaw
  is a coin flip between mirrored solutions while its translation holds to a
  millimetre; the fit holds 0.4°. The layout is FACES, consistently.

## The swap, the errand and the tools on the lift

- **A FAILED PICK ENDS THE ERRAND AT THE RACK** (issue #298): no drive to the
  use pose, no return of a module it never had, `error: never picked up
  <module>`, and a History line saying WHICH (`HubLifecycle.pick_failure`,
  shared with `fetch`: whose fork holds it — `lifecycle.carrying`, the others'
  public surface — a robot at the bay, no route, a miss and how off
  `PICK_WHY`, or nowhere; #264: `swap_at_bay_routine`'s answer was thrown away
  and "the pick missed" covered approaches that never got there). On the pair
  the old phantom trip parked the robot at the rack as the other came back to
  stow, and the stows, picks and "no route" failures cascaded from there
  (Rowan paid 3 of 27).
- **A use-phase leaves the tool in its CARRY configuration** (a stow computes
  release heights from the lift it starts at), and `run_errand` honours
  `drive_to`'s answer (a use-phase after a failed drive is skipped and the
  tool still goes home). An abort STOWS at a safe point — after the pick,
  after the carry drive, between STROKES (`PenPlotter.should_stop` — pen UP;
  mid-line is SimNotes' "The pen would not stow"), at a census vantage,
  between dance moves — and costs: 0.20 Wh on a room_hub carry aborted at the
  use pose, recorded as `abortCostWh`. A restart stows a module left on the
  fork first, the pen's carriage centred, or it jams on the bracket feet.
- **A drive carries the tool posed** (issue #347;
  `tests/test_tools_on_the_floor.py`). Every verb that moves the base is
  `Verb.drives` and goes through `steps.run_verb`, which puts the fork into
  its carrying pose first (`travel_pose`: a tool's axes to their compiled
  rest, the arm in, `MODULE_DRIVE_LIFT`; a claw holding a cube keeps it at
  `CARRY_LIFT`, arm out; an empty fork only draws its arm in). ⚠ UP BEFORE IN,
  IN BEFORE DOWN, there and in `carry_configuration_routine`: MEASURED, a claw
  drawn in at a 0.033 m lift came off its seat. A new driving verb sets the
  flag, and `procedure_rule` names the flagged verbs to the mind. ⚠ A
  procedure's `draw` takes the planner to `use_at` before the use-phase:
  `drive_to_board_routine` is a straight line with no planner, and from the
  rack it knocked Rowan over six times of six (SimNotes, "A drawing that set
  off from the rack").
- **What to draw is `tools/strokes.py`; how to draw it is `tools/drawing.py`,
  and the plotter never imports the content module** (issue #11). A figure is
  sized to `Envelope.for_board` (carriage ±55 mm ∩ lift ∩ face), not the slab
  — `targets_for` CLIPS, so an oversized figure draws flattened and reports a
  perfect trace. +lat is the viewer's LEFT: text advances toward −lat, and
  asymmetric figures are authored in the reading frame and flipped once. Each
  stroke re-presses and each press seats the module differently (−4.55 to
  +2.78 mm across one word), so `draw_program` re-zeros every stroke against
  the FIRST press's bias. A board's `fill` is measured against the pen's REACH
  (110 × 200 mm), not the slab.
- **The claw's verbs are `pick(tag)` / `place(tag)`, at `fetch`/`stow`'s
  level** (issue #264; Overseer.md §2b — the motor-level procedure topped out
  at two layers). What they took: `HubMission.spot(at_height=)` — the RANGE
  off the tag's centre pixel at the cube's known layer height, because PnP's
  range to a 24 px tag is quantised ±10–15 mm — `ClawTool.calibrate_from_body`
  at the deployed reach, `held_hang` re-read on arrival (a cube slips 7 mm
  down and 10 mm along the pads over a carry), and `tuck_routine` at
  `MODULE_DRIVE_LIFT` (at `APPROACH_LIFT` the claw sits in the lidar's
  front-stop cone). ⚠ The claw holds only what can MOVE (`ClawTool.held()`: a
  body with degrees of freedom) — lowered to 0.02 m both pads rest on the
  floor, and `pick` was refused "already holding floor". ⚠ `place`'s ok is
  measured off the world after the retreat (rests one pitch up, within half an
  edge), never off the release.
- **The lift is a scale** (issue #227; SimNotes "The lift is a scale"):
  `read("lift.force")` is `actuator_force[lift]` + `axes.LOAD_NOISE_N` (0.03
  N, deterministic per physics step and per robot — `axes.noise`, a crc32
  seed, never `hash()`); MEASURED to 1 mN (the lift is a position servo on a
  damped slide with no `frictionloss`). The tare is the fork's own weight and
  the prompt does not say so.
- **An idle robot leaves the rack** (issue #298): the loop's stand-by and a
  DECIDED `idle` first clear the rack (`RACK_CLEAR_M` 2.0 m of the rack prior,
  measured from the nearer of the rack's two bodies,
  `HubLifecycle.rack_distance`, back to its start) — measured: an idle second
  robot at the bay standoff failed the first robot's next pick 0.4 m away, and
  Rowan stood at the standoff for an hour.

## Two rovers in one world

The loop that steps both is CLAUDE.md's ("Two robots run from ONE physics
loop"). How each one drives around the other:

- ⚠ **THE SWAP'S FINE TIMESTEP IS THE MODEL'S**, so it is counted per model
  (`mission.fine_step_begin/end`, issue #264): the first robot out of its swap
  used to put cruise back under the other's terminal approach.
- ⚠ **MUTUAL AWARENESS is the reported pose, not the scan**: each mission's
  `others` (callables → the other's dead-reckoned x, y — a network fact) masks
  a disc of `OTHER_ROBOT_CELLS` (12 = 0.6 m) out of the traversable mask at
  plan time, a stagnated drive with another robot within `OTHER_NEAR_M` WAITS
  (`OTHER_WAIT_S`) instead of failing, and each sensor keeps the other robot
  OUT OF THE MAP AND IN THE DRIVE — the lidar (`Lidar.scan_split`, issue #316:
  one set of casts, the room's returns and the peer's, feeding the 0.25 m
  front stop) and the near-field depth camera (`DepthFrame.peers`, issue #328:
  the same frame sorted by what each ray hit, feeding
  `HubMission.watch_for_peers`, which records a sighting inside
  `PEER_STOP_AHEAD_M` 0.60 m and `PEER_STOP_HALF_M` 0.20 m of dead ahead, and
  `drive_to_routine` HOLDS for one — never a blind reverse, because the other
  robot is the one obstacle that moves). ⚠ **THE HOLD IS AGAINST THE TRAVEL
  LEFT** (`PEER_CLEARANCE_M` 0.30 = the front face 0.20 m ahead of the axle
  plus 0.10 m): a drive with 0.1 m to go cannot reach a body 0.5 m ahead, and
  holding for one is how a robot parked BESIDE the charge bay stopped the
  other charging at all — MEASURED, a peer 0.50–0.56 m from the charge
  standoff took the approach from 96 s and a dock to 201 s and none, because
  arriving turns the robot to face the rack and sweeps a body it never travels
  into through the corridor. ⚠ The seam only SEES (`peer_sighting`, fresh for
  `PEER_HOLD_S`); the drive decides. ⚠ MEASURED (#328): the scan plane at
  0.223 m crosses only the peer's MAST, so a peer 0.25 m across the bow put
  ZERO rays in the front cone and the closest approach was 0.225 m — inside
  contact — where the depth channel holds at 0.594 m; the camera carries
  150–970 points of a peer between 2.0 m and 0.4 m, losing its near face below
  that to `depth.MIN_Z`. The camera is OPT-IN (`near_field=`, ON in
  `serve.py`) and the lidar stop is the floor under it. Measured: painted into
  the grid and inflated, a robot driving past walled in the robot it passed,
  which planned None from its own cell for 12 s and gave up the bay; dropped
  from the scan outright, the 0.25 m front stop was blind to the only thing in
  the world that moves (382 `met` and nine `stuck` deaths in the week that
  found it).
- ⚠ **THE MASK SWALLOWS A GOAL INSIDE IT** (issue #313): the nearest cell A*
  may plan to is (0.6 − d) from the goal and a stagnated drive counts as
  arrived only inside `CLOSE_ENOUGH_M` (0.15), so a peer within **0.45 m** of
  a bay standoff makes that bay unreachable however many attempts are spent on
  it — MEASURED 0/3 picks with a robot at the neighbouring standoff (0.26 m)
  against 3/3 at 0.56 m. `HubMission.peer_on_the_goal` is that arithmetic and
  has one home.
- ⚠ **A ROBOT LYING DOWN IS AVOIDED WHERE IT LIES** (issue #365; SimNotes "A
  robot lying down was avoided where it said it was"): the errand a robot
  falls in keeps turning its wheels, MEASURED up to 2.2 m of reported pose off
  the body in 10 s, and on 2026-09-23 the other robot drove into it and fell
  too. While a chassis is past `TOPPLE_TILT_RAD` (dead or not),
  `HubLifecycle.keep_clear` answers a `KeepClear(down=True)` at the middle of
  its BODY (`footprint_centre`), `DOWN_ROBOT_CELLS` 14 = 0.7 m because the
  mast lies along the floor (0.55 m for `peer_on_the_goal`), placed through
  the DRIVER's own pose error (`as_seen`, so its drift cancels). It will not
  move, so the depth camera does NOT hold for it (a detour turning at its
  disc's edge was held there until the drive gave up) and a stagnated drive
  does not wait on it; and since nothing holds for a fall any more, a drive
  checks who is down every `DOWN_CHECK_S` (0.1 s) and replans at once on a
  change. The stand-up hands it back to the reported pose, and a narration or
  History line calls it "lying knocked over" (`posture`), never "standing".
- ⚠ **A TAKEN BAY IS WAITED FOR, AND DONE AT THE RACK MEANS GONE** (issue
  #346; SimNotes "Two robots at one rack"): the swap and both halves of the
  charge approach ask `HubMission.bay_wait`
  (`HubLifecycle._await_bay_routine`) and wait beside the holder's lane for
  `WAIT_OCCUPANCIES` (3, Ben's) × the MEASURED occupancy of what holds the bay
  — `SWAP_OCCUPANCY_S` 30, `CHARGE_OCCUPANCY_S` 462, and a charge holds its
  neighbouring tool bay (0.200 m) too — before `peer-at-bay`, whose line says
  who held it and for how long. Only a PICK's wait ends early (the robot's own
  interrupt, the reserve): a return's never does — abort means stow, and a
  procedure's `stow()` runs inside its errand — and a charge's never does
  (giving up on it is the death). The wait is keyed on `peer_on_the_goal`,
  never on having a name for the peer. With a peer in the world the loop
  leaves the rack before deciding or grading (`_leave_rack_routine`, not
  re-driven from where a clear already failed), `_clear_rack_routine` READS
  its drive and tries `CLEAR_SPOTS` more, `RACK: lingering` logs what still
  escapes, and a failed return is retried `STOW_RETRIES` times before anything
  but a charge. ⚠ Alone, none of it moves the robot: a single robot's day is
  unchanged. ⚠ Every "no-tag" charge measured was a look from BESIDE the
  standoff (±30° of the axis decodes, 60–90° cannot), never the other robot
  hiding the tag; `charge_trace` logs what each look saw.
- A robot-to-robot CONTACT is an `encounter` row (`touched` / `separated`,
  `ENCOUNTER_PHASES`), not only `collision_steps`, which is on no record and
  no wire. `scripts/two_robots.py [--view] --errands carry,carry` is the demo.

## The near-field camera on the mast

CLAUDE.md keeps the rules that outlive the body (no return is not a
reading; nothing that decides reads the map). The rover's half (issue #34;
Parts.md "near-field depth camera", SimNotes "Near-field 3D"):

- A D435-class unit as 8400 `mj_multiRay` casts a frame (~5–7 ms; ⚠ the call's
  `cutoff` is a max distance and `0` tests nothing), honest in `MIN_Z`/`MAX_Z`
  on AXIAL z, z² noise, the image-left occlusion shadow and the self-view.
  Points come out in the ROBOT frame through the nominal mount; the map takes
  the believed pose. ⚠ The mount is measured: the deck sets the near edge
  (0.26 m ahead of the axle at any pitch ≥ 35°), pitch sets the far one (40° →
  2.1 m); a camera at head height sees only deck; the lens stands 1.5 mm proud
  of its housing or every ray hits the housing.
- ⚠ The height map is `SIZE_M` 4 m at `CELL_M` 2 cm = 40 000 cells (fewer than
  the 2D grid), the LAST frame's highest point per cell, `Z_MAX` 0.5 m (a 2.5D
  map cannot say what is under an overhang; voxels are 25× the cells and
  40–200× the update, MEASURED in the spike).
- ⚠ IN THE LOOP IT IS OPT-IN (`HubLifecycle(near_field=)`, `run_demo`/
  `build_pair` likewise; `serve.py` ON by default, `$PLUGGY_NEAR_FIELD=0` off;
  the demo scripts `--near-field`, off): `_near_field_step` ticks the seam at
  `depth.PERIOD` and folds each frame in at the BELIEVED pose, a frame is ~7
  ms so a mission test that did not ask pays nothing, and
  `power.DEPTH_CAMERA_W` (2.0) is drawn ONLY while it runs.
  `economy/energy.json` is measured WITH it on (`energy_spike.py` builds its
  lifecycles so), the dearer case; that re-pricing raised home's reserve 0.90
  → 0.95 and grew room_hub's demo cell 0.7 → 1.0 Wh (its carry read 0.817 as a
  first errand after the explore and a 0.7 cell could no longer OFFER the
  job).

## Routes across the property

The surveyed routes below are slated for removal (#381: honest mapping and
SLAM instead of surveyed facts); until then they are what makes a long
drive arrive.

- ⚠ **`drive_to` an UNMAPPED goal aims at the known-free cell nearest it by
  straight line IN THE ROBOT'S OWN COMPONENT** (issue #298: the nearest cell
  anywhere was a one-cell island beside `whiteboard_b`, `astar` answered None
  in 0 s and the board paid nobody for 30 hours), which beside a house is
  still INDOORS: a 12 m leg along the north street set off on a 463-waypoint
  detour and stalled. That half is unfixed on purpose (the same fallback
  carries every bay and board approach into a wall's inflation); a flown test
  keeps its legs inside the LIDAR's 8 m (SimNotes, "A goal out of sight"), and
  a PROCEDURE's `drive_to` past that reach or off the map walks
  `lifecycle.route_to`'s doorways first (#353) — ⚠ the house's own legs must
  route to nothing, or every cage program and `solutions.WEIGH` changes route
  (`tests/test_house_route.py`).
- ⚠ **A STOW FROM OUT ALONG THE LAB'S ROUTE COMES HOME BY THAT ROUTE FIRST**,
  from the door the robot's ZONE is behind (`lifecycle.HOME_FROM`, never the
  nearest leg by straight line; `steps.home_legs_routine`; `stow()` and the
  stow after a procedure alike; the workshop's single drive home works): a
  weighing that failed in the lab left the claw on the fork, the swap's single
  drive home across 30 m of street failed twice, and the claw was lost at the
  garden door. `lab_route`'s first leg stops 0.6 m short of the garden doorway
  (the door post trips the reflex from a cold start), and a cube NOT IN VIEW
  is looked for where the house set it out (`steps.prop_stand` +
  `lifecycle.zone_route`, on `fetch`'s terms: a single `drive_to` across the
  house stalls in the hall at 41 s, so the workshop has route legs as the lab
  does).
- ⚠ **The cage program passes THROUGH the pad, never parks on it**
  (`lifecycle.cage_program`, issues #226, #287): the route in legs ≤ 6.7 m,
  `cage_route` drops the legs behind the robot, a pass from 0.8 m south to
  `PLATE_PASS_M` 0.3 m north and back — parked on the believed centre after
  ~0.24 m of trip drift the press was the reckoning's, and the shock landed on
  2 of 11 deployed jobs (SimNotes, "A trip across the street"). The errand
  ends IN THE LAB: the return is the reserve's, and `go_charge` docks from
  there through 0.55 m of drift, measured.

## Energy on wheels

The rules are CLAUDE.md's ("What an errand costs"); the rover's numbers
(`economy/energy.json`, measured by `scripts/energy_spike.py`):

- **Home's reserve is 2.05 Wh** since the loop (#215: 1.354 Wh of travel over
  44.6 m plus a 0.316 dock, plus one retry leg; it was 0.95 and dock-dominated
  when the far corner was 15 m away), and its demo cell 4.5 Wh
  (`energy_spike.py --reserve`); room_hub's 1.0 Wh cell (0.7 before #34) is
  zero-margin. `--pack hosting` is 8 Wh home, 6 Wh room_hub. The reserve's
  worst point is the loop's south-west corner BY ROUTE (45 m), not the
  straight-line farthest corner; `tests/test_world_budget.py` routes every
  zone over a raster of the compiled world to check it.
- **A cost key may name a TARGET** (`draw:whiteboard_b`): the far board costs
  0.24 Wh more, and one number for both kills the robot or prices the near
  board off the cell. ⚠ EVERY ROW IS FROM THE RACK: a drawing flown FIRST from
  the explore's end read 1.354 against 0.992 from the rack and is NOT carried
  — that is the price of where the explore ended, a first errand starts on a
  full pack, and carrying it charged the loop before every second drawing (a
  two-answer day went 181 → 519 s). `energy_spike.py --actions draw:<board>`
  flies ONE board, first; the far board's drive from the explore's end fails
  every time (a failure is not a cost). `--actions
  care:feed,care:toy,care:company,shock,feed` prices the lab's acts (each ends
  in the lab; the spike docks between them): a `care` act is ~1.1–1.3 Wh and
  ~110–130 s each way. The tower's estimate is measured off the first written
  procedure (2.7 Wh, #264).
- **`chargeW` is the SLOWEST press measured** (19.4 W; others read up to 39.6)
  because the spread is geometry — a cap sized off a good approach fires on a
  slow charge that is working. `$PLUGGY_CHARGE_SCALE` is TEST-ONLY (served
  default 1.0, pinned three ways in `tests/test_battery.py`).

## Scripts that fly the rover

| script | what it is for |
|---|---|
| `scripts/charge_spike.py`, `swap_spike.py`, `stall_spike.py`, `noslip_spike.py`, `hub_spike.py`, `answer_spike.py` | tolerance sweeps behind a constant; each `--blind` (or `--no-brake`) reproduces the before-fix rows so the premise cannot rot. What each guards: `charge_spike` the dock approach, `swap_spike` the bay fix, `stall_spike` the press (all above), `noslip_spike` the solver policy (above; SimNotes' sweep table), `hub_spike` the coupling envelope (SimNotes "Hub coupling spike", ToolPattern.md §2), `answer_spike` `ANSWER_MATCH_MM` (TaskPattern.md §4.1) |
| `scripts/nearfield_spike.py` | the near-field depth camera and height map (issue #34): `--mount` (pitch → self-view and floor band), `--cost` (frame ms per resolution and world, the height map's update, the voxel alternative), `--find` (smallest cube found standing still, by range); default a filmstrip. Re-run `--cost` after touching `perception/depth.py`, `heightmap.py` or the mount |
| `scripts/draw.py`, `pickup.py`, `dispense.py`, `lcd.py`, `plate.py`, `module_power.py`, `home_draw.py`, `hub_swap.py`, `hub_mission.py` | one tool or mechanism each: the pen (`--program square|text`), the claw, the seed dispenser, the LCD (`--errand census|dance`), the garden pressure plate (the reference ACTIVITY), the module's electrical interface, the home drawing errand (a THIN caller of `HubLifecycle.run_errand`; `--cycles 2` before believing any change to the swap stack), the bay swap, the milestone-8 story. `--record PATH` on draw/pickup renders 720p video |
| `scripts/two_robots.py` | two rovers from one loop (`--view`, `--errands carry,carry`, `--game`) |
