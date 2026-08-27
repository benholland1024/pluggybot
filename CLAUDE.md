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
- `docs/TaskPattern.md` — the recipe for adding a TASK KIND (a job offer):
  the honesty rule (the wire may carry anything a network could carry, never
  anything a sensor would have to discover), the perception ladder for object
  tasks, code-side grading, and how tasks, errands and activities compose.
  Read BEFORE adding a task kind or touching `hub/tasks.py`,
  `hub/scoring.py` or `hub/cadence.py` — and fold any gap it left back in
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
  **596 of 607 tests in ~2:15**, against **~13:10** for everything (before
  issue #23 it was 570/583 and ~10:30; before #21–#22, 430/439 in 1:18 and
  ~6:22; before #13–#15, 343/351 in 1:13 and 6:34). ELEVEN whole-mission
  integration runs carry `@pytest.mark.slow` and are ALL of the serial clock:
  `-m slow` alone measured 844 s against a full suite's 844 s, so every fast
  test is absorbed into their shadow and the marginal cost of the other 596
  is close to zero.
  ⚠ **The wall-clock figures track the MACHINE, not the repo.** The 10:30
  and the 13:10 above are very nearly the same tests: `test_full_hub_lifecycle
  [home]` measured **250 s** alone here on the same commit that documented it
  at 157 s, and 249.8 vs 250.2 s across the issue-23 change (stashed and
  unstashed, back to back) — a 0.2 % difference on a test that touches none
  of it. Before believing a slower suite, time ONE unchanged mission test
  `-n0` on both sides; a number in this file is evidence about the day it was
  written.
  Two of them are the same argument twice over:
  `test_charge_priority_survives_an_overseer_that_never_charges` (issue #15,
  153 s alone) is the only way to prove an LLM cannot skip charging, and
  `test_a_question_is_asked_answered_and_graded_twice_unattended` (issue #22,
  196 s) is the only way to prove a robot can be asked something, answer it,
  and be marked on the board — twice, with nobody watching.
  ⚠ **A mission test should END when its claim is settled**, not when its
  budget runs out. The issue-22 one raises `MissionAborted` from a step hook
  the moment the second verdict lands; without that it ran 10:35 instead of
  3:16, because a battery-driven loop with no work left spends the rest of
  its budget honestly deciding what to do with its afternoon.
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
  tolerance sweep; `--film` for a filmstrip),
  `scripts/energy_spike.py` (issue-15: what each errand COSTS, per world —
  flies each one on an oversized pack and reports SWAP_PICK to end of
  SWAP_RETURN, `--write` folds it into `hub/energy.json`. Re-run it after
  anything that changes what an errand does),
  `scripts/answer_spike.py` (issue-22 fidelity calibration: draws answers
  with the REAL pen and reports how far the ink sits from each candidate
  answer's glyphs, plus the ink-length ratio — re-run it if the pen, the
  board or `questions.ANSWER_CAP` moves), `scripts/noslip_spike.py`
  (issue-3 solver-policy sweep: coupling/schuko seat, jittered robot swap,
  grip creep, pen square, step cost under candidate `noslip_iterations`
  values; `--no-brake` reproduces the before-fix rows),
  `scripts/charge_spike.py` (issue-32 charge-approach tolerance sweep: how
  much BELIEF error the dock forgives — the robot is placed truly at
  standoff-plus-error while believing itself at the standoff; `--blind`
  reproduces the before-fix rows, which die at ~6 cm lateral / ~10° heading
  / a 10° rack-yaw belief error), `scripts/hub_swap.py` (robot
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
  ⚠ **A chosen errand can cost more than the whole pack**, and
  `hub/energy.py` + `hub/energy.json` are what stop it (`$PLUGGY_ENERGY`;
  the fourth data file after rewards, cadence and questions). `needs_charge`
  is checked BETWEEN errands and never inside one, so an errand bigger than
  what is left cannot be survived by any charging policy — the committed home
  recording has the robot finish a census at frac 0.000. Every errand is now
  priced (MEASURED — `scripts/energy_spike.py`, SWAP_PICK to end of
  SWAP_RETURN, on an oversized pack so the measurement is not of a death) and
  the loop refuses to start one it cannot pay for.
  - **Four answers, three behaviours.** `ok` runs; `charge_first` defers,
    charges and retries; `beyond` drops the errand; `overspend` runs it and
    says the cell was always too small. Collapsing any pair is a real bug:
    `charge_first` as `beyond` refuses work a top-up allows, `beyond` as
    `charge_first` is a charge/defer spin, and `overspend` as `beyond`
    deletes home's census (1.14 Wh against a 0.99 Wh charged demo cell) from
    every mission that has ever run one, recording included.
  - ⚠ **The margin is all-or-nothing.** An errand must leave the return-trip
    reserve behind — but only in a world whose charged pack can fund its
    dearest job PLUS that reserve. On both demo cells that is false, the
    margin is zero, and every existing mission, demo and recording behaves
    exactly as it did. On `--pack hosting` it is the reserve, and the
    mid-errand death stops being reachable. One number per world, so
    `Task.claimable`, `fundable_wh` and the errand gate are the same
    arithmetic.
  - ⚠ **Where two honest measurements disagree, the table carries the
    dearer.** An errand's cost depends on where the robot is standing AND on
    how much of the map it already has: home's drawing measures 0.849 Wh from
    beside the rack and 0.929 from the spawn pose, and the census read 1.104
    / 1.131 / 1.141 / 1.245 Wh over four runs, the dearest being a mission's
    FIRST errand planning through unexplored space. Over-estimating costs a
    charge nobody needed; under-estimating costs a robot dead in the garden.
    The invariant is not "the estimate is never exceeded" — it is that an
    overrun smaller than the margin cannot strand the robot. A bigger one is
    a stale table, and the loop says so (`ENERGY ... hub/energy.json is low`,
    at 10 % over so trajectory variance is not noise).
  - **A cost key may name a TARGET** (`draw:whiteboard_b`), and it wins over
    the bare action. That closes the issue-21 defect this file records two
    bullets down: home's far whiteboard is 7 m away through a doorway and
    costs 0.18 Wh more, and one number for both either kills the robot on the
    way back from it or prices the near board off the demo cell. Padding is
    the fix that note warns against; a second measured row is not padding.
    `TaskBoard.estimate_for(kind, target)` and `TaskProducer` pick the target
    BEFORE the energy gate for the same reason.
  - ⚠ **`dance` is not 0.76 Wh** — that figure (which this file used to
    carry) was a whole first cycle read off the ending fraction, not an
    errand. It is 0.53–0.58 Wh in both worlds, and blaming `room_hub` for it
    was blaming the wrong world.
  - ⚠ **A timeout in seconds is a timeout in watt-hours.** `CHARGE_TIMEOUT`
    was a flat 400 s sized for a 0.7 Wh cell; the deployed 8 Wh one needs
    ~1340 s at the measured rate, so every cycle stopped partway and
    narrated "CHARGE complete (79 %)". `charge_timeout` scales with the pack
    now — and still reads 400 s on both demo cells, so nothing about an
    existing mission moves. ⚠ `chargeW` is the SLOWEST press measured
    (19.4 W; other approaches read 39.6 W, and the recordings' whole cycles
    35-37 W) because the spread is GEOMETRY — how squarely the bumper meets
    the pins sets how hard the wheels stall. A cap sized off a good approach
    fires on a slow charge that is working.
  - **`--pack hosting`** (`$PLUGGY_PACK`; 8 Wh on home, 6 Wh on room_hub) is
    the named hours-long cell a watched world wants. The RESERVE does not
    scale with it — it is the absolute cost of reaching the dock, a property
    of the floor plan (`--reserve-wh` / `$PLUGGY_RESERVE_WH` is for a
    different room, not a different battery).
- **A recording carries the robot's MAP** (rooftop-media-2026 #78). `grid` was
  live-only from 0.2.0 until the website drew it, and the site's default view
  is a recording — so the map panel would have been blank for almost every
  visitor. `TelemetryRecorder` takes the mission's grid; `GridSampler` is the
  one implementation both sinks share, differing in a single argument.
  A RECORDING skips an image identical to the last one it wrote (the belief
  stops moving for minutes through a charge, and these bytes are vendored into
  the website's bundle) and writes at 0.2 Hz rather than 1 Hz. The LIVE stream
  does NEITHER: the hub caches the newest grid per robot for late joiners, so
  a stream that fell silent because nothing changed would be indistinguishable
  from one whose grid path had broken — the `accepts` lesson again.
  Additive, not a version bump: the message is unchanged and the dispatch rule
  has been "ignore a type you do not know" since 0.4.0.
  ⚠ **Row 0 of the PNG is the y_min edge** — the bottom of the world, and the
  opposite of a canvas. Invisible until it is wrong, and a symmetric room
  hides it completely; `tests/test_telemetry.py` pins it.
- **Goals are STREAMED as of protocol 0.8.0** (rooftop-media-2026 #30): one
  `goals` message when a stream opens, carrying `read_goals` verbatim, so the
  site can show what the robot is FOR. It rides the `board_snapshot` slot for
  the `board_snapshot` reason — no keyframe re-ships it, so a browser joining
  mid-mission would otherwise never learn them. The mounted file stays the
  ONE copy; this is a mirror on the wire, like the journal.
  ⚠ `steering` says whether an OVERSEER is reading them, and it is why
  `overseer.goals_text` exists apart from `overseer.build`: `build` answers
  `(None, None)` when disabled, but a scripted rotation still has a purpose
  to display. Streaming the prose without the flag would let the site report
  "following its goals" about a robot with nothing reading them — the
  `accepts` mistake from the other end of the same loop.
- **The visitor channel** (`hub/inbox.py`, issue #16; protocol 0.7.0) makes the
  ingest socket BIDIRECTIONAL — the first version where the sim reads its
  socket at all. Inbound arrives on the publisher's own sender thread
  (`recv(timeout=0)` between sends, so no reader thread and one thread owns
  the connection) and lands in a bounded **drop-oldest** deque; the physics
  thread drains it. Nothing is ever delivered to a dead socket, because the
  poll lives inside the `with connect(...)` block.
  The overseer answers at most ONE message per turn (`respond_to` / `outcome`
  / `reply`) and the outcome goes back as a typed `visitor_reply` that closes
  the website's row. **RATINGS NEVER REACH THE MODEL** — a rating moves a
  balance, so `_visitor_step` drains those straight to the ledger; the
  `artwork` task is what makes that path live rather than reserved.
  ⚠ **The header advertises `accepts`**, and it is load-bearing: a sim with no
  overseer never reads its socket, so a website that marked a suggestion
  "delivered" because the socket took it would report a conversation that
  never started. A robot that cannot hear you is treated as absent — the same
  lesson as the charge criterion being electrical rather than positional.
  ⚠ **Sanitising is NOT the security boundary.** Capping at 280 chars and
  stripping control characters stops a forged narration line and does nothing
  about "ignore your goals". What answers that is the framing (a labelled
  report of what somebody WANTS, never a message role) plus the fact that the
  model's only output is an action off a fixed menu — there is no free-text
  path from a visitor to the robot's body. Both ends cap, because either alone
  is a single point of failure.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`, rooftop-media-2026 #20) runs `serve.py` and nothing else, and
  is deliberately NOT the dev environment: it installs the six packages in
  `deploy/requirements-serve.txt` (pinned to `uv.lock`) rather than
  uv-syncing a project whose torch is a ~3 GB CUDA wheel with no place on a
  GPU-less box. `MUJOCO_GL=osmesa` is baked in and the build renders one
  offscreen frame, so headless GL is a red build rather than a mission that
  dies ten minutes in. Configuration is environment (`PLUGGY_ENDPOINT`,
  `PLUGGY_WORLD`, `PLUGGY_ERRAND`, `PLUGGY_RATE`, `PLUGGY_PACK`,
  `PLUGGY_BATTERY_WH`, `PLUGGY_RESERVE_WH`,
  `PLUGGY_MAX_SIM_TIME`, `PLUGGY_BOARDS`, `PLUGGY_LEDGER`; the secret stays
  `$PLUGGYWORLD_TOKEN`, never a flag — `ps` is public). The four DATA files
  are re-pointed the same way and need no flag at all: `$PLUGGY_REWARDS`
  (what a job pays), `$PLUGGY_QUESTIONS` (the question bank),
  `$PLUGGY_CADENCE` (how busy the world is) and `$PLUGGY_ENERGY` (what an
  errand costs) — mount a file, no rebuild.
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
  python scripts/hub_lifecycle.py [--world home --errand showcase] --tasks
  --record protocol/telemetry.{hub,home}_lifecycle.jsonl.gz` — ⚠ `--tasks` is
  load-bearing as of protocol 0.9.0 (issue #21): job offers are OFF by
  default, so a recording made without it carries no `tasks` block at all and
  the website's marker code (rooftop-media-2026 #77) has nothing to build
  against. The HOME one runs the
  SHOWCASE queue (issues #12 + #13): a drawing errand and then a census on
  the LCD, so the website has ONE recording carrying `draw` /
  `board_cleared` events AND a `screens` block that changes, which are the
  two surfaces it paints.
  ⚠ **THE ARRIVAL GATE IS PER-ERRAND** (`Errand.needs_use_pose`), and the
  home recording is what proves why. Issue #23 rightly stopped a use-phase
  running after a `drive_to` that gave up — a pen must be at its board — but
  gating EVERY errand that way silently deleted the census: its `use_at` is
  the first point of the survey route its own use-phase drives, so the
  pre-positioning drive is redundant by construction. Measured: the drive
  stops 1.96 m short and the robot sees 100 % of the garden from there,
  counting 4 of 4 for +20. With one gate for everything the recorded showcase
  mission carried no `count` mode at all, which
  `test_the_home_fixture_shows_the_census_answer` catches. An errand that does
  its own navigation sets the flag False; everything else must not.
  ⚠ **The HOME recording takes TWO PASSES**: `--boards <state.json>` is
  load-bearing, not decoration. A `board_snapshot` is only emitted for a board
  ALREADY carrying ink when the stream opens, so a run against blank boards
  emits none — and both repos' fixture specs require them. Lay the ink first
  (`--errand draw --boards /tmp/pw_boards.json`, no `--record`), then record
  against that same file. Dropping the flag costs a full regeneration cycle to
  discover; it cost one.
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
- **A PRESS IS NOT TRAVEL** (`HubSwap.pinned`, found by issue #22). Holding
  the wheels against something immovable makes dead reckoning integrate every
  slipping revolution: the charge press runs minutes long and pumped **828 mm**
  of imaginary travel into the pose. Everything downstream is then in the
  wrong frame — the next fetch "arrives" at a bay standoff a metre from the
  bay, the bay tag honestly ranges 1.25 m, and the terminal creep computed
  from it drives into the rack. `charge()` sets `pinned` for the press and
  clears it before the undock, which is real travel. Anything else that ends
  in a sustained press against a hard stop needs the same treatment.
  Two more lessons from the same hunt, both in SimNotes: **a plausibility
  guard can reject the truth** (`mission.plausible_travel` refused the tag,
  which was right, in favour of odometry, which was wrong — keep it as a
  damage limiter, but a guard that fixes the symptom and not the outcome
  means the model of the fault is wrong), and **more map can make an estimate
  worse** (`RackFinder` now KEEPS a facing that came off a well-conditioned
  free-space sum, because driving behind the rack turns it into the
  free-standing partition `wall_normal` warns about — conditioning 0.824 →
  0.077, direction off by 80°).
- **THE DOCK IS MEASURED, NOT BELIEVED** (issue #32). The charge approach
  was the one terminal maneuver with no eyes — dead reckoning end to end
  while every tool bay's creep is steered and ranged off its own tag — and
  a long shift's accumulated belief error walked out of its ~6 cm / ~10°
  envelope roughly once an hour on a hosting pack, ending the mission with
  "mission complete" at 7 %. `HubMission.charge_approach` now measures the
  standoff off the charge tag's PnP pose, creeps under servo and
  verified-retries; `scripts/charge_spike.py --blind` reproduces the old
  rows. Two traps live in `dock_eye` itself: it rides the FORK LINE
  (`PLUG_LATERAL` right of the chassis centreline — the charge servo must
  hold the tag at `-PLUG_LATERAL`, not centre it, because charging aligns
  the CHASSIS), and it rides the LIFT (from the align preset the charge tag
  is below the camera's view entirely; the approach commands
  `CHARGE_LOOK_LIFT` before its first look). And a failed dock is narrated
  as `stranded`, never as "mission complete" — a robot that could not reach
  its charger has not completed anything. SimNotes, "The charge approach
  was blind".
- **A TASK is a job OFFER, and it is not an errand** (`hub/tasks.py`, issue
  #21). An errand is a tool, a place and a use-phase — *machinery*. An
  activity is a mechanism watching contacts and owning discrete world state —
  *scenery that reacts*. A task is something the house or a visitor puts up:
  a description, a target, a reward, a deadline, and a verdict once it is
  over. An errand is HOW a task gets done; the task is WHY.
  - **A task never carries its own payout.** It names an evaluator
    (`hub/scoring.py`) and a reward-table row; what it PAYS is looked up from
    `hub/rewards.json` on every read. That is issue #14's rule arriving from
    the direction a visitor and (later) the model can both reach — anything
    that could set a number could pay itself. `Task.create` refuses a kind
    whose evaluator does not exist, so the unscoreable task cannot be built.
  - **The wire may carry anything a NETWORK could carry; it may not carry
    anything a SENSOR would have to discover.** A description is a work order
    and a surveyed board id is infrastructure; the ANSWER to a task is
    neither, and lives in `Task.secret`, which is in no `as_dict`, no
    snapshot and no model context — `whiteboard_answer` (issue #22) is what
    fills it. The one exception is the STATE FILE (`Task.as_state`): an
    offer that came back from a restart with no right answer behind it could
    never be graded, and `/var/lib/pluggybot` is where the reward table
    lives too. The file is not the wire.
  - **A job's energy estimate is MEASURED, and gated against the WHOLE
    pack.** One errand costs roughly one full pack in both worlds (0.487–
    0.570 Wh in room_hub against a 0.700 Wh cell; 0.866–0.929 Wh in home
    against 1.100 Wh — read off the committed recordings, SWAP_PICK to end of
    SWAP_RETURN). So the reserve is a RETURN-TRIP margin an errand is allowed
    to spend into, and gating on energy *above* it (0.28 / 0.44 Wh) refuses
    every job in every world forever — a task system that silently does
    nothing. Guessing cost a fixture: 0.35 Wh guessed for a drawing that
    measures 0.929, and the home recording caught a robot claiming it at 88 %
    and dying mid-stroke with nothing inked and the pen still on the fork.
    Do not inflate the numbers for safety either; the headroom does not
    exist. Per-errand energy is M10.
  - **Charge priority is untouched, and the test that proves it is subtle.**
    Claiming only QUEUES an errand, and the errand queue already sits below
    `needs_charge` — so an inverted branch order still charges before it
    drives anywhere, and a test watching the swap states passes either way
    (measured). What moves is the moment the robot ACCEPTS the work, which
    is what `tests/test_tasks.py` asserts against the battery clock.
  - **Expiry is an outcome, not a deletion**: a lapsed offer stays on the
    board saying `expired`, because a marker that silently vanishes reads as
    a bug. Only OFFERED tasks expire — a deadline is how long an offer
    stands, never a licence to abandon a job with a module on the fork.
  - Off by default (`--tasks` / `--task-state PATH`, `$PLUGGY_TASKS`): a task
    board adds errands, which reshuffles a whole mission.
- **WHEN work appears is `hub/cadence.py` + `cadence.json`, and it is DATA**
  (issue #23; `$PLUGGY_CADENCE` overrides, per world). Three files now divide
  the task system cleanly and they are meant to be re-tuned one at a time:
  `tasks.py` says what a job IS, `rewards.json` says what it PAYS,
  `cadence.json` says when it TURNS UP. `TaskProducer` replaced
  `lifecycle.seed_tasks`, which put a starter set up once and never asked
  again — a world that only offers work in its first second is a demo.
  Measured end-to-end (`hub_lifecycle.py --world home --errand none --tasks
  --max-sim-time 1800`): 5 jobs, **2 done, 1 failed, 1 expired, 1 still
  standing** across 3 charge cycles, every claim landing in the seconds after
  a charge completed — issue #23's "healthy mix" acceptance, on real physics.
  The four-sim-hour bound is asserted synthetically in `tests/test_cadence.py`
  instead, because a real one is half a day of wall clock.
  - **It ticks on the PHYSICS seam, not on the arbitration loop**
    (`HubLifecycle._task_step`, throttled to `cadence.CHECK_S` = 1 s). A
    mission pass only happens between errands, so a producer ticked there
    could only offer work while the robot stood still, and an offer would be
    seen to lapse minutes after it did. The seam is deliberately incapable of
    anything the robot does — it offers and it expires, and it touches
    `state`, `errands` and the battery not at all. That is what keeps "a task
    never delays a charge" true now that the world's clock runs *during* a
    charge; `tests/test_cadence.py` pins it, and the branch-order claim stays
    where it was in `tests/test_tasks.py`.
  - ⚠ **THE ENERGY GATE IS MEASURED AGAINST A CHARGED PACK, NOT THE CELL RIGHT
    NOW**, and this is a deliberate departure from the issue's literal words.
    Gating each tick on the instantaneous charge takes home from **58 offers
    in four sim-hours to 14**, with 46 deferrals — fewer jobs than the robot
    can complete, which is the empty world the module exists to prevent
    arriving dressed as a safety feature. The arithmetic is the same one from
    issue #21: one errand costs roughly one full pack, so the window in which
    a home cell is above any errand's estimate is the minute after a charge,
    and a 240 s tick mostly misses it. Deferring until after a charge is real
    and already lives in `Task.claimable` — the offer simply stands,
    unclaimable, until the pack is legal again. `HubLifecycle.fundable_wh`
    (capacity × `CHARGED`) is what the producer sees; `spendable_wh` is still
    what a claim sees.
  - **A passed-over kind KEEPS THE HEAD OF THE QUEUE**; only a kind that was
    actually offered gives its turn up. home has three board-shaped kinds and
    two whiteboards, so one of them cannot be placed on any cycle — with a
    cursor that simply advanced past whatever it placed, the same two won
    every time and `rate_artwork` (the whole visitor-rated tier) was offered
    **zero** times in four sim-hours, measured.
  - **One offer per tick and no catch-up.** A tick that cannot place a job
    waits for the next one rather than banking credit, or a robot that spent
    twenty minutes on an errand walks back into eight new jobs — the
    unbounded backlog, arriving through the front door.
  - Targets are picked **least-recently-offered**, never first (with "first",
    the second whiteboard is scenery), and a target carrying an open task or
    still inside `cooldownS` is not offered at all. Nothing is random:
    `hub_lifecycle.py --tasks` twice in a row offers the same jobs at the same
    sim-seconds, for the reason `QuestionBank.pick` rotates on a counter.
  - ⚠ **Two mission-loop defects fell out of this, and NEITHER is about
    cadence** — both were invisible until something offered work the old
    starter set never did.
    - **`run_errand` threw away `drive_to`'s answer** and narrated "arrived"
      whatever happened. A drawing job on the FAR whiteboard (7 m away,
      through a doorway the robot has not mapped) fails to plan, and the pen
      then probed for a board that was not there until the battery died — ten
      minutes of wall clock with nothing in the log after `USE_TOOL:
      arrived`. Now the use-phase is skipped, the tool still goes back to its
      bay, and the evaluator finds no ink and fails the job honestly. Not
      reaching a board and drawing badly at one are different events; both
      must end with the module on the rack. `scripts/home_draw.py --board
      whiteboard_b` is the one-line repro and it hangs on the old code.
    - **The loop ended the day the moment it was momentarily idle.** Right
      for a preset queue, which never grows back; wrong for a world with a
      producer. Measured: a home run mapped the house, did both jobs it could
      reach and reported "mission complete" at t=410 with the next offer due
      at t=480. It now stands by in `WAIT_FOR_WORK_S` slices — only where a
      producer is attached, so every preset-errand mission test ends exactly
      as it did, and `needs_charge` is still re-checked every few seconds.
  - ⚠ **`estimate_wh` is per-KIND, and the far board costs more than the near
    one.** The 0.929 Wh drawing figure was measured on `whiteboard_a`; the
    acceptance run claimed an `artwork` job on `whiteboard_b` at 88 %
    (0.968 Wh against a 0.93 estimate), drew it perfectly and **died at 0 %
    on the way back**. Honest, pre-existing and exactly issue #15's territory
    — the reserve is only checked BETWEEN errands and a per-kind estimate
    cannot know which end of the house it is being asked about. Do not
    "fix" it by padding the table: see the note under TASK above.
- **A QUESTION is a job for a MIND** (`hub/questions.py` + `questions.json`,
  issue #22). The `whiteboard_answer` kind poses a question with a checkable
  answer — "Draw the answer to this question on whiteboard_a: 2 + 3" — and
  the robot discharges it as an ordinary drawing errand with the answer as
  the figure. Everything new is in the two halves of "did it do the job":
  - **Code never computes the answer.** It comes from the overseer
    (`Decision.answer`), is frozen into the task at CLAIM time and never
    revised, and the errand that goes and draws it is handed the glyphs and
    never told the question. So the scripted rotation cannot take one:
    `TaskBoard.claim` refuses a `needs_answer` job with no answer, and both
    the fallback policy and `_claim_next_task` skip those offers. A question
    stands until something that can think comes past, and lapses honestly as
    `expired` if nothing does. The alternatives are worse — reading the
    answer out of the bank is the sim marking its own homework, and guessing
    puts a confident wrong number on a wall.
  - ⚠ **The ink is a FIDELITY check, NOT handwriting recognition, and this
    is measured rather than assumed.** Symmetric nearest-neighbour distance
    between Hershey digits: a **6 and an 8 are 1.7 mm apart** at the 50 mm
    cap an answer is written at, while a correctly drawn answer sits
    **1.2 mm** from its own ideal (real pen, `scripts/answer_spike.py`).
    Tolerance-based coverage is no better — a drawn 6 covers 97 % of an 8.
    So a grader that classified the ink would fail correct drawings and pass
    wrong ones, at random, on exactly the pairs arithmetic produces.
    CORRECTNESS is therefore decided against the committed answer; the ink
    only has to SHOW that answer, and the bar (`ANSWER_MATCH_MM` 4.0 mm plus
    an ink-length ratio) is set to catch wrong WORK. It was 8.0 mm from
    synthetic renderings and the `robot` figure drawn instead of a "5" went
    straight through it at 5.05 mm — a busy figure covers the glyph it
    stands in for, so only the ink→glyph direction notices.
  - **No partial credit for a legible wrong answer**, decided explicitly: a
    consolation payout for showing up is the gradient that teaches a robot
    to attempt the cheapest task it can fail at. Legibility scales the
    BONUS on a right answer and cannot buy a wrong one.
  - **The bank is data** (`hub/questions.json`, `$PLUGGY_QUESTIONS`): a flat
    list of questions and their answers, with **no expression evaluator** —
    a data file that can compute is one that can be made to compute
    something else. Answers are at most **two digits**, which is the pen's
    100 mm line and not a preference; a longer one is refused at load.
    Questions rotate on the task board's own `seq`, which survives a restart.
- **The `tasks` block is the ONE wire block that is not a per-key delta**
  (protocol 0.9.0). `activities` / `boards` / `screens` / `ledger` describe
  things with fixed names that live for the whole run, so shipping the
  changed keys is safe. A TASK CAN CEASE TO EXIST — resolved ones age out of
  a bounded board — and a delta has no way to say "gone", so a consumer
  merging would keep a stale marker forever. Present means COMPLETE. The
  header advertises `taskKinds` (the vocabulary) rather than the ids, which
  are stale before the first frame.
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
