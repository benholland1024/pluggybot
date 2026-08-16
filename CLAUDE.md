# PluggyBot — notes for Claude

Simulated self-charging robot in MuJoCo. Before doing anything, read:
- `docs/PluggyPlan.md` — goals, milestone status, architecture
- `docs/SimNotes.md` — hard-won simulation lessons; read BEFORE touching `models/` or contact/actuator params
- `docs/Parts.md` — locked hardware decisions and the sim parameters they feed
- `docs/ToolPattern.md` — the recipe for adding a tool module (coupling
  envelope, module anatomy, contact rules, build sequence, rack integration);
  read BEFORE designing a new tool, and fold any gap it left back into it
- `docs/ActivityPattern.md` — the recipe for adding an ACTIVITY (task state
  machine): sensed criteria, hysteresis + latching, pre-allocated geom/mocap
  toggles, and how activity state reaches telemetry. Read BEFORE building a
  puzzle, mechanism or gardening step
- `docs/Overseer.md` — the LLM overseer (issue #15): the ONE branch of the
  arbitration loop an LLM may replace, the action vocabulary, the three things
  it structurally cannot do, the scripted fallback, the call budget, and the
  measured battery limit. Read BEFORE touching `hub/overseer.py`, the decision
  vocabulary, or anything that changes what the model is shown

## Working style

- Explain in prose, at the level of "a teammate catching up": name the
  concepts (running average, pinhole projection, convex decomposition) rather
  than assuming them, and say what a number means, not just what it is.
- Ben may still claim ML training runs, but Claude runs them by default now.
- Verify physics claims empirically (headless probes, filmstrip renders via
  offscreen Renderer) rather than by reasoning alone; it has won every time.
  When a result looks good, try to break it before believing it — the MSAA
  label bug, the stale-dataset contamination, and the spin-collision gap were
  all found this way, and two of them were hiding behind green metrics.
- Every debugged failure becomes a pytest assertion, and the assertion must be
  shown to fail without the fix — a regression test that cannot fail is décor.

## Commands

- **The suite runs in PARALLEL by default** (`addopts = "-n auto --dist
  worksteal"`, pytest-xdist). Nothing to remember and nothing to pass — the
  numbers below are what you get. `-n0` runs in-process, which is what you
  want for a single test, for `--pdb`, and any time you need readable live
  output; six workers for one test is a net loss.
  Scaling is ~1.8× on the full suite, not 6×, and that is a real ceiling
  rather than a tuning problem: the mission tests contend for memory/GPU
  bandwidth and each runs ~36 % slower under load, so `--dist worksteal`
  measured identical to `load`. `test_full_hub_lifecycle[home]` takes 157 s
  on its own and no worker count beats that.
  ⚠ **A wall-clock benchmark must time its two sides INTERLEAVED**, never as
  two blocks: suite load VARIES over time, so a mission test starting during
  a solid block of one implementation's reps penalises that half alone.
  Going parallel is what exposed this — `test_vectorized_update_is_5x_faster`
  read 4.9x against a 5.0 bar it clears at 6.8-7.1x quiet. Timing by
  `process_time` is NOT the fix (the contention is memory bandwidth, not
  preemption); interleaving is.
- Tests, while iterating: `MUJOCO_GL=egl uv run pytest -q -m "not slow"` —
  **430 of 439 tests in ~1:18**, against **~6:22** for everything (the
  figures before issues #13–#15 were 343/351 in 1:13 and 6:34, i.e. ninety
  more tests for no wall-clock). Nine whole-mission integration runs carry
  `@pytest.mark.slow` and are most of the serial clock on their own
  (`test_full_hub_lifecycle[home]` is 22 % of the suite by itself; the top
  two are 40 %) — but only ~3:20 of marginal wall-clock in parallel, since
  they run alongside everything else. That margin is why they stay.
  The ninth is `test_charge_priority_survives_an_overseer_that_never_charges`
  (issue #15, 153 s alone): the only way to prove an LLM cannot skip charging
  is to fly a mission with one that keeps trying.
- Tests, before calling any work done: the **FULL** suite,
  `MUJOCO_GL=egl uv run pytest -q`. Run it while iterating too — not just at
  the end — whenever the change touches something a whole mission exercises:
  `models/` or a world generator (`home.world`, `hub.coupling`) · contact or
  actuator params · `control.py` / `behavior/navigation.py` · the
  swap/coupling/mission stack · the telemetry frame format or `protocol/`
  fixtures. Those tests exist precisely because the two costliest bugs in
  this repo (a frame-relative verdict, a sign on the return travel) were
  invisible to every cheaper test, and a geometry change is exactly what
  breaks them. Skipping them on that kind of edit is how the next one
  reaches a commit. At 6:34 it does not have to be dead time either — start
  it in the background and write the commit message while it runs.
  ⚠ Their runtime is EMERGENT, not fixed: the mission runs until the battery
  loop completes, so a world change reshuffles the whole trajectory. Adding
  the garden pressure plate took the home lifecycle from 219.7 to 353.6
  sim-seconds (1 → 2 charge cycles) while costing essentially no physics. A
  slower suite is not by itself a regression.
- Lint: `uv run ruff check src/ scripts/ tests/`
- Demos: `scripts/teleop.py`, `scripts/map_teleop.py`, `scripts/explore.py [--headless]`
  (milestone-4 mapping demo — kept as the minimal repro; `lifecycle.py` is the
  full mission), `scripts/spot_outlets.py` (detector → landmarks),
  `scripts/lifecycle.py [--headless] [--battery-wh W]` (full mission loop:
  explore → dock → charge → resume; battery-driven since milestone 7 —
  `--explore-budget N` remains as a timer override, and DOCK is real
  physics — plug seats in the socket), `scripts/schuko_spike.py`
  (docking tolerance sweep), `scripts/hub_spike.py` (milestone-8 tool-coupling
  tolerance sweep; `--film` for a filmstrip), `scripts/noslip_spike.py`
  (issue-3 solver-policy sweep: coupling/schuko seat, jittered robot swap,
  grip creep, pen square, step cost under candidate `noslip_iterations`
  values; `--no-brake` reproduces the before-fix rows), `scripts/hub_swap.py` (robot
  swaps a module at the hub in `models/hub_world.xml`),
  `scripts/draw.py` (the drawing tool: fetch the pen module from bay C, carry
  it to a board, plot a figure; saves `draw.png` — filmstrip + commanded-vs-
  traced overlay + error stats. `--view` watches it live and skips the
  filmstrip, `--program square` for the more diagnostic figure, `--program
  text --text "HELLO"` for Hershey lettering),
  `scripts/pickup.py` (the claw module: fetch it from bay D, grip a block off
  the floor, carry it, set it down; saves `pickup.png`. `--view` watches live.
  Full pick-carry-place verified),
  `scripts/plate.py` (the reference ACTIVITY, issue #8: the robot drives onto
  a sprung pressure plate in the home world's garden and latches a gate open;
  saves `plate.png`. `--view` watches live. Note the gate is a MOCAP body —
  `geom_pos` mutation is silently inert on anything welded to the world, which
  is all scenery; see docs/ActivityPattern.md §3.4),
  `scripts/dispense.py` (the seed dispenser, the fifth tool and the first
  built against `docs/ToolPattern.md`: fetch it from bay E, drive a row and
  meter out exactly one seed per point; saves `dispense.png`. `--view`
  watches live. The escapement meters by GEOMETRY, not by timing — and note
  the seeds carry `condim="6"` rolling friction, without which a dropped
  sphere rolls ~590 mm and *sliding* friction does not slow it at all),
  `scripts/lcd.py` (the LCD module, issue #13 — the LAST module to get a job,
  and the first whose output is not physical: fetch it from bay A and either
  `--errand census` (survey the garden, count the plants the robot can
  actually SEE, put the number on the screen — scored against hidden ground
  truth read out of the model) or `--errand dance` (a fixed routine with an
  expression per move). Saves `lcd.png`: a filmstrip of the module plus the
  screen's whole state timeline. ⚠ The face is drawn in the BROWSER, so
  MuJoCo renders a dark panel whatever the robot is feeling — the timeline
  is the artifact, because it is exactly what goes on the wire),
  `scripts/module_power.py` (module electrical interface: runs the errand and
  saves `module_power.png` — filmstrip with the module painted live/dead plus
  a per-pole continuity timeline; `--bare` for the faster hub_world version),
  `scripts/hub_mission.py`
  (the milestone-8 story: navigate room_hub → fine-align → swap → return;
  `--view` opens the MuJoCo viewer and paces it to real time, `--fast` skips
  the pacing), `scripts/hub_lifecycle.py` (the hub-era battery-driven loop:
  explore → fetch a tool → use it → stow it → charge at the hub; `--view`,
  `--battery-wh W`, `--record out.jsonl.gz` writes a PluggyWorld telemetry
  recording — issue #4; `--errand {carry,draw,draw2,census,dance,showcase,
  none}` picks what the robot is FOR this run and `--boards PATH` keeps what
  it drew — issues #12 and #13. `showcase` is draw + census, the queue the
  site's fixture is recorded from, so ONE recording exercises both streamed
  surfaces — ink and a face),
  `scripts/serve.py --endpoint ws://host:port`
  (webserver v1, issue #5: the hub lifecycle headless, paced to real time,
  streaming protocol frames + grid PNGs + event lines over an outbound
  WebSocket — the sim never blocks on the socket; `--rate X`, `--free-run`
  measures the machine's real-time multiple; `--token` / `$PLUGGYWORLD_TOKEN`
  is the website's ingest secret; `--world {room_hub,home}` picks the world
  (issue #9 — the site serves `home`); `--errand`/`--boards` as above, so the
  site can watch a real drawing errand; docs/Webserver.md),
  `scripts/ws_sink.py` (dummy sink for serve.py: message counts + received
  frame-gap stats + keyframe spacing; `--token` makes it refuse an
  unauthenticated publisher, like the real ingest path),
  `scripts/overseer_probe.py` (issue #15: makes REAL Haiku 4.5 calls against a
  synthetic robot state and reports tokens, cost per sim-hour and the prompt-
  cache hit rate. `--tokens-only` counts the stable prefix and stops, billing
  no tokens — the number that matters, because Haiku 4.5 does not cache a
  prefix under 4096 tokens and the marker is silently inert below it. ⚠ Both
  modes need `$ANTHROPIC_API_KEY`: `count_tokens` is a free endpoint, not a
  local tokenizer, and there is no offline Claude token count worth trusting)
- **The LLM overseer** (`--overseer` on `serve.py` / `hub_lifecycle.py`, issue
  #15; `docs/Overseer.md`) replaces **exactly one branch** of
  `HubLifecycle.run()`: which errand, when the battery is fine and nothing is
  queued. It is OFF by default and the loop is unchanged without it.
  **CHARGE PRIORITY STAYS IN CODE** — `needs_charge` is checked before the
  overseer is reached and no action suppresses it, because an LLM that can
  decline to charge bricks the world overnight. Same rule from the other end:
  it *sees* the reward table and its balance and can move neither, and the
  census's ground truth is redacted out of its context. A *chosen* `charge`
  also needs the pack below `TOP_UP_BELOW` (75 %): charging is a scored task
  and the trip costs energy, so an unconditional one is perpetual motion paid
  in points. Every failure
  (timeout, error, malformed answer, spent budget) resolves to a scripted
  rotation tagged `fallback:<why>`, because "the robot chose to explore" and
  "the API was down" must not look the same on the wire. Memory is two files
  in `/var/lib/pluggybot`: `goals.md` is read and human-edited, `journal.json`
  is written and never edited.
  ⚠ `output_config.effort` is NOT supported on Haiku 4.5 (400); structured
  outputs are, and are what the decision uses.
  ⚠ **A chosen errand can cost more than the whole pack.** Measured: one dance
  is ~0.76 Wh, which fits home's 1.1 Wh cell (ends at 31 %) and exceeds
  room_hub's 0.7 Wh cell outright — the robot dies mid-errand with zero charge
  cycles, and no charging policy can save it, because the reserve is only
  checked BETWEEN errands. `--overseer` belongs on `home` until room_hub's
  demo cell grows or per-errand energy is actually modelled.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`, rooftop-media-2026 #20) runs `serve.py` and nothing else, and
  is deliberately NOT the dev environment: it installs the six packages in
  `deploy/requirements-serve.txt` (pinned to `uv.lock`) rather than
  uv-syncing a project whose torch is a ~3 GB CUDA wheel with no place on a
  GPU-less box. `MUJOCO_GL=osmesa` is baked in and the build renders one
  offscreen frame, so headless GL is a red build rather than a mission that
  dies ten minutes in. Configuration is environment (`PLUGGY_ENDPOINT`,
  `PLUGGY_WORLD`, `PLUGGY_ERRAND`, `PLUGGY_RATE`, `PLUGGY_BATTERY_WH`,
  `PLUGGY_MAX_SIM_TIME`, `PLUGGY_BOARDS`, `PLUGGY_LEDGER`; the secret stays
  `$PLUGGYWORLD_TOKEN`, never a flag — `ps` is public).
  ⚠ **A lazy import is the failure mode here**: the detector comes in
  inside `hub.tags._shared_detector`, so nothing an import scan can see —
  which is why `tests/test_deploy.py` blocks the omitted packages and then
  actually flies the robot. Adding a runtime dependency to the mission
  stack means adding it there too.
  This repo owns the IMAGE; the website repo owns the DEPLOYMENT — the
  `sim:` service lives in `rooftop-media-2026/compose.yaml` (built from
  `context: ../pluggybot`, behind a `sim` profile), and there is deliberately
  no second copy here to drift from it. `/var/lib/pluggybot` must be a
  volume: boards AND the points ledger are world state, and every mission end
  is a restart.
- PluggyWorld protocol fixtures (`protocol/`, issue #4) are GENERATED, and
  there is one scene AND one recording **per world** — a replayer picks its
  scene off the recording's `model` header, so a room_hub mission replayed
  against the home scene drives through walls rather than erroring. Scene
  JSON + tag textures: `uv run python -m pluggybot.telemetry.scene
  [models/home_world.xml]` (rerun after changing ANY geometry in that world —
  the fixture test fails when stale). Recordings: `MUJOCO_GL=egl uv run
  python scripts/hub_lifecycle.py [--world home --errand showcase] --record
  protocol/telemetry.{hub,home}_lifecycle.jsonl.gz` — the HOME one runs the
  SHOWCASE queue (issues #12 + #13): a drawing errand and then a census on
  the LCD, so the website has ONE recording carrying `draw` /
  `board_cleared` events AND a `screens` block that changes, which are the
  two surfaces it paints.
  Format + versioning rules in `protocol/README.md`; a `protocolVersion` bump
  is a deliberate two-repo event (the website repo vendors these fixtures).
- **A recording is a MIXED stream as of protocol 0.4.0**: `draw`,
  `board_cleared` and (0.6.0) `earned` lines ride between the frames. Dispatch
  on `type`; no
  `type` means frame. Ink is NEVER MuJoCo geometry — a stroke is a `draw`
  event carrying the polyline the pen actually inked, and the browser paints
  it into a canvas texture (the three-layer rule from ActivityPattern.md).
  Board state (`hub/boards.py`) is world state, not run state: it survives a
  restart in a JSON file written on every stroke, and its `fill` is measured
  against the pen's REACH (110 × 200 mm — carriage travel with the base
  parked) rather than the 320 × 260 mm slab.
- The HOME world is GENERATED (issue #6) — regenerate `models/home_world.xml`
  + `models/home_world.meta.json` with `uv run python -m pluggybot.home.world`
  after changing any layout constant in `src/pluggybot/home/world.py` (the
  committed pair is tested against the generator, so a stale file fails).
  Layout, visual hints, zones, spawns, board specs and the battery re-tune
  all come from that ONE module. Run it: `--world home` on
  `scripts/hub_lifecycle.py` (explore → errand → charge in the house), and
  `scripts/home_draw.py` (fetch the pen → erase the board → draw on a
  wall-mounted whiteboard → stow it; `--board whiteboard_b`, `--program
  circle`, `--view`, `--boards PATH`, `--no-erase`, and `--cycles N` to
  repeat the whole errand N times. Since issue #12 it is a THIN CALLER of
  `HubLifecycle.run_errand` rather than a second mission stack — add
  behaviour to `hub/errand.py`, never here). The pen's stow works as
  of issue #10 — three faults in a row, and the ONE that found the last two
  was running the errand twice: a second fetch starts from the state the
  first cycle left, which is a different test. Use `--cycles 2` before
  believing any change to the swap/coupling stack. SimNotes, "The pen would
  not stow".
- **Face states are a two-repo contract too** (issue #13).
  `telemetry.protocol.FACE_STATES` / `SCREEN_HINTS` / `SCREEN_MODES` are the
  vocabulary the `screens` block may use; the website draws a parametric
  face per name and falls back to `idle` for one it does not know, so ADDING
  a face is additive and renaming one breaks both repos. The sim never ticks
  an animation — `hint` names a LOOP the browser runs, because a 150 ms
  blink does not belong on a 20 Hz pose stream. And `powered` is the
  coupling's electrical criterion, never "am I carrying it": a module in its
  bay is dark, and so is a half-seated one on the fork.
- **Visual hints are a two-repo contract.** `telemetry.protocol.VISUAL_HINTS`
  is the vocabulary; the sidecar's `visualHints` may only use those strings,
  and `scene_dict` raises on anything else. Adding a hint is additive (the
  website falls back to raw primitives); renaming one is a breaking change in
  both repos. NEVER encode hints as geom colors — the robot's cameras render
  rgba, and colour-as-encoding couples the tag detector to the website's art.
- Hub worlds are GENERATED — regenerate `models/hub_world.xml` +
  `models/hub_rack.xml` with `uv run python -m pluggybot.hub.coupling` after
  changing any rack geometry. The rack has **five tool bays** (A–E) plus the
  charge bay; `HUB_STATION_YS` is APPENDED to, never reordered, because
  bay↔tag pairing is by index. A sixth tool needs the rail to grow again and
  a re-check against BOTH rooms the rack stands in (`docs/ToolPattern.md` §6). `models/room_1_scenery.xml` is the shared floor
  plan behind both `room_1.xml` (plug robot) and `room_hub.xml` (fork robot);
  edit scenery there, never in one room only.
- `--views` on teleop.py / map_teleop.py / lifecycle.py saves `views.png`
  (stereo pair + map + dock camera, issue #1) alongside `map.png`; ~15 ms/save
- **Demo video**: `--record PATH` (`.mp4`/`.gif`) on `scripts/draw.py` and
  `scripts/pickup.py` renders 720p footage offscreen via `viz.Recorder`, with
  `--record-fps` and `--record-speed` (sim seconds per played second).
  Frames are STREAMED to the encoder, never buffered — a 90 s demo at 30 fps
  is ~2700 frames, which is ~7 GB of 720p RGB if you hold them. Two rules the
  tests guard (`tests/test_viz.py`): recording must never step the sim (an
  end-of-clip "hold on the final pose" belongs AFTER the result dict, or it
  silently shifts the reported settle state), and the render size must sit on
  the 16-px macroblock grid or ffmpeg resamples the frames behind you.
  ⚠ The recorder's camera is NOT always the filmstrip's: for `draw.py` the
  filmstrip's az=150 sits BEHIND the board (a thin slab at x=1.30, drawn on
  its -x face), so the whole drawing phase renders as a grey rectangle. The
  video pans to az=60 during the drive. Pick angles by sweeping azimuth at the
  moment of contact, not by reasoning about the geometry.
- RL docking (milestone 6): train
  `MUJOCO_GL=egl uv run python scripts/train_docking.py` (SAC over
  `pluggybot.envs.DockEnv`; checkpoints under `runs/docking/`), score
  `MUJOCO_GL=egl uv run python scripts/eval_docking.py --trials 24`
  (scripted DOCK vs RL policy on identical room_1 trials with the real YOLO)
- Sockets: `models/schuko_sockets.xml` is GENERATED (invisible collision
  layer for room_1's outlets) — regenerate with
  `uv run python -m pluggybot.docking.schuko` after moving an outlet
- Dataset (deterministic; `datasets/` is gitignored, and the generator wipes it
  first — regenerating into a dirty dir once contaminated 195 labels):
  `MUJOCO_GL=egl uv run python scripts/generate_outlet_dataset.py --count 1200`
- Eval (the one that matters — the val split shares the training generator and
  scored 0.99 mAP while calling a light switch an outlet):
  `MUJOCO_GL=egl uv run python scripts/eval_detector.py --poses 1000`
- Train: `uv run yolo detect train data=datasets/outlets/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640`
  (torch is pinned to the cu128 index in `pyproject.toml`: the driver here is
  CUDA 12.8, and PyPI's default cu130 build silently falls back to CPU)

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- `models/world.xml` = bare world for physics tests; `playground.xml` /
  `room_1.xml` add scenery for humans and mapping. Never put scenery in the
  test world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy, ix]`.
- Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm ahead.
- **Position setpoints are always RAMPED, never written across a gap.** A stiff
  servo handed a step delivers it as an impulse: it has thrown a module off the
  fork and batted a gripped block out of the jaws. `control.slew` does this for
  wheels; `ClawTool.set_lift`/`jaws` and `PenPlotter.ramp` for the rest.
- **One solver policy: `noslip_iterations` is 0, always and everywhere**
  (issue #3; sweep table in SimNotes). Never phase-toggle solver modes:
  always-on noslip ≥ 1 half-seats the jittered coupling (on the fork but not
  electrically powered), and a runtime toggle is global state that leaks
  across fixtures — and, in the shared PluggyWorld model, across robots.
  Creep under sustained load is fixed at its actual source, per-part:
  `coupling.GRIP_SOLIMP` where a **contact** drifts (the claw's jaw pads,
  slip −21.7 mm → −0.13 mm), and wheel-joint `frictionloss` where a
  **joint** rolls — a velocity servo commanded 0 resists speed, not force,
  so without the gearbox's parking brake the parked base walks under tool
  loads (the plotter's square: 63 % → 99 % inked, no solver pass, no cost).
  ⚠ The swap's travel constants implicitly contain mm-scale wheel slip;
  anything touching wheel contact or joint friction must re-verify the
  bay-C pick and the mission stow. ⚠ The brake also creates a stiction
  DEADBAND: wheel-speed commands under `frictionloss/kv` (0.1 rad/s) move a
  stopped wheel not at all, so P-turn controllers must go through
  `control.turn_command` (breakaway floor) — a raw `gain × err` command
  crawls (55 s to settle 0.23°, measured).
- **A final approach uses `drive_toward(..., slow_radius=R)`; a path waypoint
  does not.** The default pure-pursuit law cannot converge on a destination
  closer than its own overshoot — it flies a stable ORBIT around it (measured:
  ~900° of turning per 200 mm hop, heading error pinned at 85° with `w`
  saturated). Terminal mode adds a hard ±25° cone (`v` exactly 0 outside it —
  a soft taper alone does NOT kill the orbit) plus a distance taper. Only
  short hops are affected, which is why the claw and plotter never showed it.
  `tests/test_navigation.py` guards both halves, and pins the defect in the
  default law so the fix's premise cannot rot. `PenPlotter.contact_physics` /
  `ClawTool.grasp_physics` are deprecated no-ops;
  `tests/test_noslip_policy.py` guards all of it.
- **An ERRAND is a tool, a place and a use-phase** (`hub/errand.py`, issue
  #12). `HubLifecycle` carries a QUEUE of them, arbitrated against the
  battery by the same loop as everything else, so a repeat is a list rather
  than a flag. Anything the robot does with a tool goes in a use-phase, and
  the fetch/carry/stow half stays the one implementation it took two issues
  to make repeatable. The rule a use-phase must respect: **leave the tool in
  its CARRY configuration** — a stow computes its release heights from the
  lift it starts at, and a tool axis parked where the last stroke left it
  fouls the bay's brackets. And **a result has to outlive a frame**: Python
  between two physics steps costs ZERO sim time, so a use-phase that sets the
  screen and returns has its answer overwritten by the next state's automatic
  face before a single 20 Hz frame is built (measured: the census's count
  appeared in none of 10 850 frames while the result dict was perfect). Hold
  it — `_drive(PRESENT_S, 0, 0)` — and check the RECORDING, not the return
  value.
- **A task is scored by CODE, and nothing awards itself points** (issue #14).
  Three files, and the split is the whole design: `hub/scoring.py` MEASURES a
  finished task off the sim and judges it (`EVALUATORS`, one per task, pure
  and unit-testable); `hub/rewards.json` is DATA saying what it pays (base +
  bonus × quality curves — a 0.6 mm drawing beats a 3 mm one, and re-tuning a
  payout is a JSON edit, `$PLUGGY_REWARDS` to override); `hub/ledger.py` banks
  it. A `Verdict` can only be built by `scoring.evaluate`, and `Ledger.award`
  re-derives the points from the table before accepting one. The overseer
  (issue #15) will SEE its balance and the reward table and be able to move
  neither — an agent that can score its own work learns to declare victory.
  Three rules that follow, all guarded in `tests/test_rewards.py`:
  - **Measure the world, not the report.** A drawing is scored on the strokes
    the pen wrote into the BOARD BOOK, a carry on `module_state`, a charge on
    the battery's own energy. An errand's `use` is arbitrary caller code, so
    what it says about itself is a claim.
  - **A missing measurement is not a passing one.** The census defaulting its
    absent count and truth to 0 and 0 would have scored a task that never ran
    as CORRECT — the shape of bug the whole module exists to prevent.
  - **A hidden-truth task never publishes its answer.** `secret` metrics are
    redacted from the ledger entry, the wire and the `reason` line, because
    the stream reaches both the site and the overseer's context.
- **What to draw is `hub/strokes.py`; how to draw it is `hub/drawing.py`, and
  the plotter never imports the content module** (issue #11). A *stroke
  program* is a named list of polylines in board coordinates, pen up between
  them. Three rules the build paid for:
  - **The board is not the reach.** A figure is sized to `Envelope.for_board`
    — carriage travel (±55 mm) ∩ lift range ∩ board face — not to the 320 ×
    260 mm slab. `targets_for` CLIPS, so an oversized figure draws flattened
    against the travel limit and reports a *perfect* trace, because the pen
    went exactly where it was told.
  - **+lat is the viewer's LEFT**, so text advances toward −lat. Figures that
    are not mirror-symmetric are authored in the reading frame and flipped
    once (`strokes._from_unit`, `strokes.text`). Both diagnostics are
    symmetric, so nothing before Hershey text could catch a mirrored figure —
    bounds, ink length and form error are all identical under the flip.
  - **Each stroke re-presses, and each press seats the module differently**
    (−4.55 to +2.78 mm across one word). `draw_program` re-zeros every stroke
    against the FIRST press's bias; the first stroke is untouched, which is
    why single-stroke figures are bit-identical to before and the square's
    0.57 mm baseline still holds.
- Contact params combine as the elementwise **MAX** unless `priority` is set —
  so a low `friction` without `priority="1"` does nothing. This has bitten the
  caster, the pen pads and the coupling peg.
