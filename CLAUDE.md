# PluggyBot — notes for Claude

A simulated, hardware-honest robot and the autonomous agent that lives in it.
**The project is agent-autonomy research, not a product**: the mission and
the six qualities the agent is meant to maximise are in `docs/PluggyPlan.md`
§ "What this project is for" — provisional wording, settled direction.
**#465 holds the order of work** (its "The next stretch") **and the
decisions**; #375, the quadruped pivot, is history. **The body is a ~10 kg
quadruped** (#387, `legs/body.py`) with #378's two-joint arm on its back
(#405), served as a pair; the wheeled rover before it was deleted in #376's
stage C (the tag `rover-final` is the last commit that runs it). Before
doing anything, read the doc that owns what you are about to touch:

| doc | what it holds | read it BEFORE |
|---|---|---|
| `docs/PluggyPlan.md` | the mission and the six qualities, status, architecture | anything |
| `docs/SimNotes.md` | simulation lessons, each ending in what is true now | touching `models/` or contact/actuator params |
| `docs/Parts.md` | locked hardware decisions and the sim parameters they feed | changing a part or its parameter |
| `docs/ToolPattern.md` | adding a tool module: the arm's coupling envelope, anatomy, contact rules, build sequence, rack integration | designing a new tool |
| `docs/ActivityPattern.md` | adding an ACTIVITY (a mechanism that owns world state): sensed criteria, hysteresis + latching, pre-allocated geom/mocap toggles, telemetry | building a puzzle, mechanism or gardening step |
| `docs/TaskPattern.md` | adding a TASK KIND (a job offer): the honesty rule, the perception ladder, code-side grading, how tasks, errands and activities compose — fold any gap back in | adding a task kind or touching `economy/tasks.py`, `scoring.py` or `cadence.py` |
| `docs/Challenges.md` | grading a job nobody wrote a scorer for: a predicate written BEFORE the robot sees it, through `scoring.py`'s chain, with a hold; what it cannot grade | adding a challenge, touching `challenge/`, or reaching for an LLM judge |
| `docs/Overseer.md` | the mind: its place in the loop and which rails each arm keeps, the vocabulary, the standing order and the event map, what it cannot do, the fallbacks, memory, money, visitors | touching `mind/`, the decision vocabulary, or what the model is shown |
| `docs/Testing.md` | pinning a rule without paying for a mission: the three kinds of test, the cheap levers, how to measure | writing a test that flies anything |
| `docs/Observatory.md` | the deployed world's PERIODS: what was running while the rows were written, opened by the PR that changes the deployed design | reading the observatory or changing what is deployed |
| `docs/Evaluation.md` | measurement: the arms, the six qualities and their shapes, capability's per-task record, the observatory and how it is read, what silently invalidates a number, what death costs | adding a metric, changing an arm, or concluding anything from the observatory |
| `docs/Webserver.md`, `protocol/README.md` | the served process, the stream and its versioning | touching `telemetry/`, `serve.py` or the wire |

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
  **And it is written as cheaply as it can be while still failing for the
  right reason** (Ben, 2026-09-12): pin the RULE — the inequality, the branch
  order, the one line of wiring — with a fake press, a stubbed drive, a direct
  call; an integration of the loop's bookkeeping is a day on the stub
  (`stub_life`); fly a whole mission only when the claim is genuinely about
  PHYSICS, and then stop it on the claim (`stop_when`). A flown proof whose
  rule is already pinned may go behind `--endurance` — with Ben's approval,
  below.
- ⚠ **THE TEST SUITE HAS A BUDGET, AND EXCEEDING IT NEEDS BEN'S EXPLICIT
  APPROVAL.** The full suite is **0:43** (2026-10-06, #473: 0:53 before
  it, interleaved on one machine — its files moved into memory; 7:29 with
  the rover, #376 stage C). Any
  change to testing that would take it past **10 minutes on a quiet machine, or 15 on a
  busy one**, must be stated as such in the PR — the number, the test, and why
  it cannot be cheaper — and approved by Ben personally before it merges. ⚠
  **THE BUDGET HAS NO OUTSIDE** (Ben, 2026-09-29): a test the default run
  does not run is still testing, and leaving the default run is no way under
  the budget. ADDING one — a flight behind `--endurance`, or any skip, gate,
  marker or job that leaves a test out — or MOVING a test out of the default
  run needs Ben's explicit approval in the PR, however short it is, stating
  how long it takes (wall, `-n0`) and why it is necessary: why no test in the
  default run can make its claim. The approved set is `APPROVED_FLIGHTS`
  (`tests/test_endurance.py`), each with its time and its reason; the suite
  fails on a flight not on it, on a test that skips itself, and on an
  `addopts` that deselects. Reducing suite time is a project priority: the
  suite was slowing development
  considerably at 18 minutes, and it gets there one reasonable-looking mission
  test at a time. Do not rely on full runs where a fast test settles the
  claim.
- **An inline comment states a constraint the code cannot show. Anything that
  is a story goes to `docs/`, with a one-line pointer left behind** (issue
  #51). Prose in a `.py` is loaded every time anything reads that file,
  relevant or not; a doc is loaded when the task calls for it. So a measured
  number with its failure mode attached belongs at the constant — that is 60 %
  of the comments here and they are why nobody "fixes" something deliberate —
  while the narrative of how it was found belongs in SimNotes, ToolPattern,
  ActivityPattern, TaskPattern, Overseer or `protocol/README.md`. ⚠
  **Placement was the first problem; length is the second** (Ben, 2026-09-11).
  This project is mostly written by agents, and agents do not delete: each one
  documents everything interesting about its own task, including what has
  since stopped being true. So a short why-comment stays and a measured number
  stays at its constant — but a detail that is no longer relevant is DELETED,
  not kept for the record; git history is the record. A narrative of how
  something was found, once the thing has moved on, becomes one sentence of
  what is true now. When two docs tell one story the better one keeps it and
  the other gets a one-line pointer. Shorter is the goal wherever nothing true
  is lost.
- **This file carries constraints, not stories.** A bullet here is what an
  agent must not break, the number behind it, and where the story lives,
  because every session loads this file whole. When a change makes a bullet
  false, fix the bullet in the same PR.

## Commands

### Tests

- **The suite runs in PARALLEL by default** (`addopts = "-n auto --dist
  worksteal"`, pytest-xdist). `-n0` runs in-process: a single test, `--pdb`,
  readable live output. Scaling is ~1.8×, not 6×, because the mission tests
  contend for memory bandwidth, and the floor is the LONGEST SINGLE TEST, so
  the lever is shortening the long poles; reordering the collection
  longest-first did NOT help (issue #158). Only sim-seconds count.
- **While iterating:** `MUJOCO_GL=egl uv run pytest -q -m "not slow"`.
  **Before calling any work done: the FULL suite**, `MUJOCO_GL=egl uv run
  pytest -q` — and if your change makes it slower, the budget above applies.
  Run it while iterating whenever the change touches what a whole mission
  exercises: `models/` or a world generator (`home.world`, `legs/world.py`) ·
  contact or actuator params · `legs/` · `navigator.py` /
  `behavior/navigation.py` · the swap/coupling stack · the telemetry frame
  format or `protocol/` fixtures. The two costliest bugs in this repo (a
  frame-relative verdict, a sign on the return travel) were invisible to every
  cheaper test. Start it in the background and write the commit message while
  it runs. ⚠ Wall-clock tracks the MACHINE, not the repo (one mission test has
  read 157 s and 369 s on different days): before believing a slower suite,
  time ONE unchanged physics test `-n0` on both sides, INTERLEAVED — never as
  two blocks.
  `test_vectorized_update_is_5x_faster` reads under its 5× bar under load
  (6.8–7.1× quiet); `process_time` is NOT the fix. ⚠ Mission runtimes are
  EMERGENT: a world change reshuffles the whole trajectory, so a slower suite
  is not by itself a regression.
- **The suite renders on EGL unless the environment names a backend**
  (`tests/conftest.py`, #440; unset is GLFW, which needs a display), and its
  last line names the RASTERISER that drew: what a camera test measures is
  the device's, never the backend's — EGL is a GPU on one box and llvmpipe
  on another.
- **The suite's files are IN MEMORY where `/dev/shm` has room**
  (`tests/conftest.py`, #473; a `TMPDIR` already set wins): a memory store a
  test leaves behind is closed by the collector wherever it next runs, and
  on a spinning `/tmp` its fsync stalled that test 0.4–4 s, 29–39 s of worker
  time a run. ⚠ A test that TIMES code turns the collector off for the timed
  part, as `timeit` does, and pins it (`test_scan_match.py`).
- **`slow` means EXPENSIVE *AND* UNABLE TO CATCH A REGRESSION WHILE YOU
  ITERATE** — the rule is written out in `pyproject.toml`. Whole-mission runs
  qualify; so do PREMISE-PINNING tests (which bypass a fix and assert the old
  defect still reproduces). A test that calls the real code and asserts it
  declines is not slow, whatever it costs. **Shorten before you mark.**
- **A mission test ENDS WHEN ITS CLAIM IS SETTLED**, not when its budget runs
  out — `HubLifecycle.stop_when` is where the rules live.
- **Behind `--endurance` is PHYSICS no fast test can make, approved by Ben,
  and a flight flies only when a change touches what it guards** (issues
  #158, #380; `tests/conftest.py`, `tests/test_endurance.py`). The loop's
  bookkeeping — the mind, the economy, the record, the wire — is a day on the
  stub in the default run. A flight's rule is pinned fast (name the pin in
  the comment above the mark); it is `slow`; it names the paths its claim
  stands on, `@pytest.mark.endurance(when=(...))`; and it is in
  `APPROVED_FLIGHTS` with its time and its reason, because the budget has no
  outside (above). **Before calling work done:** `MUJOCO_GL=egl uv run
  pytest -q --endurance-changed -m endurance` flies exactly the flights your
  branch touches — usually none, when it costs a collection. There is no
  "before a release" beyond that; `--endurance` flies every one,
  deliberately. ⚠ A `when` path must exist (a rename fails the fence), and
  `lifecycle.py` is in none: its rules are pinned on the stub. ⚠ A flag, not
  `-m 'not endurance'` in `addopts`: pytest keeps the LAST `-m`, so the
  everyday `-m "not slow"` would silently switch them back on. The decision
  behind it (Ben, 2026-09-12): while the design is moving, a generous pack is
  ASSUMED to fund any single errand and a battery death costs a heart.
- Lint: `uv run ruff check src/ scripts/ tests/`

### Measurement (`docs/Evaluation.md` is the record and the rules)

- **The six qualities are SHAPES over ROWS** (issue #155, the sixth #265;
  Evaluation.md §3; `evaluation/qualities.py`, `scripts/qualities.py
  --observe`): one pure function per metric over the observatory's own
  columns, one adapter per source (`from_observe`, the one source since the
  harness went with the rover, #376); a later source ADDS rows to a shape,
  never a second version of it. Four rules, each pinned in
  `tests/test_qualities.py`: nothing that must stay apart is summed; no mean;
  **absent is `None`, never 0**; never pooled across a build identity. ⚠ A
  reading of the observatory is NOT a result; it reports into the issue it
  informs. ⚠ `serves` IS NOT ON THE
  WIRE: the KEY is the test, and quality five's ratio off the observatory is
  `None`. ⚠ A test reads §3's shape table against `SHAPES` both ways: a row
  with no function fails, so capability's per-task record stays prose until
  a demo gives it rows. ⚠ Nothing in `economy/` imports `evaluation`. ⚠
  **The sixth quality is NOT time alive** — five shapes read together
  (Evaluation.md §3's table), `idling` BESIDE deaths in `QUALITIES` because
  high idling with low deaths is the failure mode; hearts bought for oneself
  come off the `HEART_BOUGHT` / `HEART_REFUSED` narration, a two-repo contract
  pinned in `tests/test_hearts.py`. ⚠ THE
  PROMPT DOES NOT CHANGE FOR IT (a test reads every rule for the word): a
  quality is what we measure, never what the robot is asked to maximise for
  us.
- **`CALL_TIMEOUT_S` is 90 s and is a patience budget, not a tail** (issue
  #117; Overseer.md §6 "The deadline"): a decision lost to a clock is the one
  failure that is purely ours. `ESCALATE_TIMEOUT_S` (120) is an ordering above
  it; `llm.LOCAL_TIMEOUT_S` is a FLOOR (a 27.3 s cold load). ⚠
  `scripts/overseer_probe.py --calls 50` under-measures a mission by about
  half: choose the deadline from the probe, confirm it with a flight.
- **A fallback's reason is FAILURE or POLICY, and the partition is ONE**
  (issues #117, #141): `overseer.POLICY_FALLBACKS` / `FAILURE_FALLBACKS` /
  `fallback_class` (`timeout`/`offline`/`garbled`/`busy`/`no-client` are
  failures; `budget`/`idle-run`/`cooloff`/`scripted-mode` are the policy
  working), read by `decision_failed` rows and never re-derived. Adding a
  reason is additive, renaming one is breaking (two-repo contract).
- **Two arms: `scripted`, the loop with no mind, and `autonomous`, the one
  mind** (issues #115, #427; Evaluation.md §2). There is no control: what is
  measured is the observatory's rows. A mind takes THREE rails off —
  `HubLifecycle.autonomous` (a property, `overseer is not None`) is read by
  `needs_charge`, `_afford_next` and `claim_budget_wh` and by NOTHING else
  (`lifecycle.shown_offers` goes through `claim_budget_wh` too, #333) —
  `RULES` says so, and `model_state` drops the code-computed verdicts
  (`affordableActions` / `possibleActions` / `claimable`) at PRESENTATION
  only — ⚠ the view narrows, the state does not (`order_runnable` reads
  `possibleActions`). ⚠ A0 hides the survival clock, or A0 and A1 are one
  run.
- ⚠ **NO SCRIPTED ROTATION IN A MIND, EVER — INCLUDING LIVE.** Every action
  originates with the LLM (a decision, a standing order, or an event-map row
  it configured); with no answer and no order the robot finishes what it is
  doing, runs what is queued, and IDLES — even if that ends in death. There
  is no `scripted()` to reach (a test asserts it). Evaluation.md §2.
- **The deployed world flies `autonomous`, both robots, origin `unseeded`**
  (issue #206). Which arm is `serve.py --arm/--origin/--rung` (`$PLUGGY_ARM`
  / `$PLUGGY_ORIGIN` / `$PLUGGY_RUNG`), off ONE definition,
  `evaluation/arms.py`; `--overseer` alone is `autonomous`. A contradiction
  (`--overseer --arm scripted`), a rung on an arm with no ladder, or
  `guarded` is REFUSED; the header says what RAN (an arm whose overseer
  could not be built is a `scripted` day; `build.rung` is absent where
  there is no ladder). ⚠ **Changing the deployed arm is a decision, not a
  config change**: Evaluation.md §2 carries the argument and updates in the
  PR that moves it;
  `tests/test_webserver.py::test_the_deployed_pair_flies_autonomous_from_nothing_and_the_header_says_so`
  pins it. ⚠ The A1–A3 rungs and the capacity sweep are POSTPONED and may be
  scrapped.

### Demos and probes

Every script takes `--help`. `--view` watches live where it exists; most
save a filmstrip PNG named after the script.

| script | what it is for |
|---|---|
| `scripts/hub_lifecycle.py` | the mission, one quadruped in `home_quad`: explore, charge, an errand queue (`--errand` off `lifecycle.errands_for`: none, or an act on the mouse), battery-driven. `--boards PATH`, `--tasks`, `--metabolism`, `--near-field`, `--overseer`, `--pack hosting`, `--record out.jsonl.gz` |
| `scripts/serve.py --endpoint ws://host:port` | the mission headless, paced to real time, streaming the protocol over an outbound WebSocket; the sim never blocks on the socket. `--free-run` measures the real-time multiple; `--pair` serves both robots; the world is `home` with legs, `home_quad` (#387; `--body`/`$PLUGGY_BODY` names the one body, the quadruped); `--world-state PATH` keeps the world and carries on from it (#345); `$PLUGGYWORLD_TOKEN` is the ingest secret (never a flag — `ps` is public). docs/Webserver.md |
| `scripts/ws_sink.py` | dummy sink for serve.py: counts, frame-gap stats, keyframe spacing; `--token` makes it refuse an unauthenticated publisher |
| `scripts/overseer_probe.py` | REAL LLM calls against a synthetic state: tokens, cost per sim-hour, cache hit rate, the latency distribution (`--calls N`). `--model org/name[:provider\|:cheapest]` measures a HuggingFace candidate (`$HF_TOKEN`, in the gitignored `.env`); `--deployed` measures the prompt the served pair sends and reports the ENERGY GATE (`report_energy` counts who took the unaffordable offer); `--prompt` prints it section by section with its sha; `--max-tokens N`, `--escalate-to X --force-escalate`, `--tokens-only` (the Anthropic path's limits: Overseer.md §6) |
| `scripts/energy_spike.py` | what each errand COSTS, per world, on an oversized pack; `--write` folds it into `economy/energy.json`, `--reserve` measures the return-trip margin, `--actions` prices named acts; `--world home_quad` prices the quadruped's explore, dock, the lab's acts, the boards' jobs (`draw|artwork|answer:<board>`, from the dock, the board remembered) and #407's (`census`, `stack`, `mass`: from the dock, the area remembered; a challenge is its hand-written solution). Re-run after anything that changes what an errand does |
| `scripts/unknown_spike.py` | #381's walking stage: a fresh quadruped (an empty map, its start pose) sent once to each zone, one process a walk: arrived or why not, the time, the walk against the true route, what planning cost; `--before` the planner before (mapped floor only, stand-ins), `--again` back and there again on the map it laid, `--unknown-cost X`, `--maps DIR` |
| `scripts/drift_spike.py` | #386: believed against true pose over lab round trips, the quadruped's (the walking policy steered by the truth, two estimates on one walk, odometry alone and matched), and whether the lab door is open in the robot's own map; `--explore SECONDS --seed K [--rest S]` the loop's own explore, steered by the belief, then home to the dock (#422), `--explore-table` across seeds |
| `scripts/determinism_spike.py` | is the world the same world twice? N scripted days hashed, first divergence attributed to GPU / decoder / raycast; `--pair`; `--compare DIR`; `--resume-at T` flies a day against one saved and carried on in a new process (#345), `--on-fork MODULE` with a tool on the fork whose returns fail (#420) |
| `scripts/make_way_spike.py` | #415: the pair in the house on true-floor maps, one robot resting in a doorway and the other walking through it, one process a scene: the walk's end, the step aside (how far, how long, why it stopped), touches and falls; `--before` nobody asks, `--finished` the resting robot's day is over, `--scene A,B` |
| `scripts/press_spike.py` | #439: the pair in the lab on true-floor maps, the camera on, one robot lying where a scene puts it (by default where its own press backs out, 0.25 m short of the feed plate's standoff) and the other finding and pressing the feed plate, one process a scene: how the press ended and its tries, the walk's and the way in's time, the steps aside, every plate the feet came down on; `--drift-walker`/`--drift` lay one map off as Luca's was |
| `scripts/solve.py --feature mouse` | ladder A of #264, the paid feed on legs (#403): offered, claimed with a prediction, walked and graded by the job's own evaluator; `--pair`, `--n N`, `--from dock\|lab`; filmstrip `solve.png`. `--feature hide_and_seek` flies the pair's game (#404): offered as the cadence offers it, both roles claimed, each run from the queue the referee fills, the verdict banked; one game a scene (`--scene A,B`, `--swap` both ways round; `--find`, `--seek-s`, `--head-s` sweep the referee). `--feature answer\|draw\|artwork` flies a whiteboard's job (#406): offered on a board (`--board`, or the two in turn), claimed -- an answer with the right one -- drawn lying and graded off the board's ink. `--feature tower\|bench` flies a challenge (#407): offered with its area's terms, claimed, the hand-written solution run as the robot's procedure (`challenge/solutions.py`), graded by its own predicate |
| `scripts/board_png.py` | a whiteboard's ink as a PNG from the boards state file or a recording. ⚠ +lat is the viewer's LEFT, as in the site's `surfaces/board.ts`; the test pins it because every figure the pen draws is symmetric |
| `scripts/quad_spike.py` | the quadruped body (#377; SimNotes "The quadruped body"): `--view` watches it, `--torque`/`--thermal`/`--energy`/`--sweep`/`--pupper` are the sizing tables (on `model.SIZING`, #377's placeholder arm, #405), `--policy`/`--climb`/`--getup`/`--posture`/`--odometry`/`--determinism` fly a trained policy in OUR physics (`--climb --scan map` on the D435's map, `legs/scan.py`, #388), `--shove` the served body knocked over in the house (what `stuck_after_s` is read off, #389), `--served`/`--rays` time the pair and the sensors |
| `scripts/dock_spike.py` | the quadruped's dock (#378; SimNotes "The quadruped's dock"): `--capture` the funnel's envelope (premises `--sticky`, `--flat`), `--approach [--n N]` the success rate walking in by the board (premise `--blind`), `--hold` lying there: contact, preload, the anchor, standing off; `--view` dockings in the viewer, one after another; `--mouth`/`--bed` fly another width; filmstrip `dock_spike.png` |
| `scripts/draw_spike.py` | drawing on a whiteboard with the arm (#406; SimNotes "Drawing on legs"): the served body, the pen on its fork, walks to the board, lies down, finds its face by touch and draws; `--figure square\|house\|answer:NN ...`, `--board`, `--stance stand` (the premise), `--n N`; `--sway [--stance lie,stand]` the stance table; filmstrip `draw_spike.png` |
| `scripts/arm_spike.py` | the quadruped's arm, its coupling and the rack (#378; SimNotes "The quadruped's arm, its coupling and the rack"): `--reach` the level-tool choice and the holding torques, `--capture` the coupling's envelope placed at a bay (premises `--rover`, `--narrow`), `--approach [--n N]` walking in by the rack's tags and taking a tool, `--retention [--stairs [--first]\|--fall]` carrying one (`--first`: the fork as first built, 45° V's, the stairs' premise), `--getup` the get-up policy with the arm, `--sensors` what the arm hides, `--envelope` a tool's mass and lever, `--served [--pair] [--n N]` the SERVED body fetching and stowing at the house's rack from the dock and from across the house (#405), `--pair --bays A,B\|A,C\|A,A` the pair at neighbouring bays, two apart, or one (the bay wait, #418); `--view [fetch\|carry\|stairs\|fall\|reach]` a scene in the viewer, looped until the window closes (MUJOCO_GL unset); filmstrip `arm_spike.png` |
| `scripts/mechanism_spike.py` | what the arm can open from the claw's lying stance and what its torques measure (#469, #466's stage 0; SimNotes "Opening a box from lying"): a lid lifted by a drop handle (the chest's, `activity/chest.py`, which it imports), a lid lifted from under its lip and a drawer, each in the storeroom, never the served world. `--walkin` the spread the claw's walk-in leaves, `--window` the push the claw on its fork holds each way, `--table` the candidates, `--tolerance` the path off the true arc at four gains, `--torques` the force at the tool off the drivers (`legs.arm.tool_force`) against the contact force, `--fit` the oracle fit (`--corners` past the drawn ranges, `--twins` mass against its lever), `--latch` a magnetic catch; `--all --into DIR` the whole batch behind the report, a file a mode (a CPU pod's work, below); filmstrip `mechanism_spike.png` |
| `scripts/imagination_gap.py` | the scene language's own gap (#466's stage 1; SimNotes "The imagination's first world"), no robot flying: over drawn chests (`--n`), the oracle's probe through the world's chest and through its best-expressible reference, with its catch and without, the force at the tool apart phase by phase; `--cost` what a rollout costs, ms a sim-second, in this process and in the worker, `--parallel 1,8,16` that many workers at once |

- **The quadruped's training stack lives in `training/`, a uv project of its own** (#377): mjlab 1.5.x (it pins `mujoco ~=3.10.0`, the served sim's), reading the body from `models/quadruped.{xml,json}` (`python -m pluggybot.legs.model` writes both; mjlab caps numpy below pluggybot's, so the two never share an environment), the arm FIXED at its stow (`quad_train.robot.freeze_arm`, #405: a policy's joints are the legs' twelve, and the arm's geometry is there to fall on). ⚠ Importing `legs.policy` or `legs.model` loads none of torch, jax, warp, onnx or mjlab (`tests/test_legs.py`); a policy reaches the served sim as an `.npz` that `quad_train.export` checks against its ONNX before writing, run by `legs/policy.py` in numpy. ⚠ The leg DRIVERS are MuJoCo's position-mode `dcmotor` (#385; `body_xml(drive="position")`), the PD and the envelope in C, commanded as a GDS68 is (`legs/drivers.py`): every command carries its gains — a policy's are its own, a routine's torque rides a target with the damping cancelled (the torque motor's step to 1e-14), `limp()` holds nothing — so a policy and the scripted routines share one body. The default gains are ONE definition, `actuator.driver_gains`, which `training/` reads from `quadruped.json`. `drive="torque"` (plain motors) is the sizing tables' instrument. `training/pod.sh` rents a Runpod GPU to train on, or CPUs to fly a batch (`create-cpu`), over the REST API (`runpodctl pod create` needs a GraphQL-writable key) — ⚠ a POST to `/v1/pods` with an EMPTY body CREATES a pod: every field has a default.
- ⚠ **A batch of flights runs on a rented pod's CPUs or under a memory cap, never bare on the dev box** (#469: six house worlds at once took it into swap and its desktop down overnight, 2026-10-06): `training/pod.sh create-cpu` → `setup-sim` (the commit, never the working tree) → `batch` → `pull-batch`, then `delete`; locally, `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 ...`. A pool of flights gives each its own process (`maxtasksperchild=1`: a world dropped in a reused worker is held until the cycle collector runs), and results go under `~`, never `/tmp`, which a reboot wipes.

### The imagination (`imagination/`; #466, Track A of #465)

- **The robot's imagination is a world of its own, and nothing it is built
  from reads the world's model of a mechanism** (SimNotes, "The
  imagination's first world"; `tests/test_imagination.py`): a document in
  the scene language (`scene.py`, `workshop/spec.py`'s pattern: JSON, mm and
  degrees at the door and the sim's units inside, every reason at once, an
  unknown field refused -- ⚠ NO CONTACT FIELD, a test pins `solref`),
  compiled with the robot's own body (`legs.model.attachable`, the CAD a
  real robot has) on a floor at the map's z = 0 (`compile.py`), and a
  record's commands replayed there step for step by the arm's own
  `ArmDriver`, the legs limp, from what the robot knew of itself
  (`record.Start`), its first command held `SETTLE_S` first (`rollout.py`;
  the body's half is `legs/imagined.py`, so the body fence holds).
  ⚠ THE FENCE HAS TWO HALVES: the worker (`worker.py`, `python -m
  pluggybot.imagination.worker`, the first worker process: a program, never
  a fork) is sent JSON and numeric arrays (`pack`/`unpack`, pickles
  refused), so never an `MjModel` or `MjData`, and a record's depth cloud is
  (k, 3) points with the simulator's geom ids refused; and nothing in the
  package imports `activity`, `challenge`, `home`, `legs.world`,
  `evaluation` or the spike, directly or through anything it can load. ⚠ A
  DOCUMENT'S PARTS TOUCH EACH OTHER ON ITS OWN CONTACT (`compile.CONTACT`,
  0.01 s), never the world's; against the robot MuJoCo mixes, and the
  robot's priority geoms (the pads, the fork, the feet) impose theirs; the
  floor is MuJoCo's defaults, as the house's. ⚠ EVERYTHING FIXED IN THE MAP
  IS ONE RIGID GROUP, which a part hinged to it never touches: walls written
  on nothing jammed a lid and threw it. ⚠ A WORLD MUJOCO RESET IS REFUSED
  (`rollout.Diverged`): its readings stay finite and plausible. A request
  cut off half way ends its worker. Its readings are the drivers'
  EXPECTATION, their noise keyed on the worker's seed, never the world's.
  Nothing in it is specific to a lid, and a model in progress is never saved
  with the world.
- **The drop-handle chest is demo 1's mechanism, and a demo's only**
  (`activity/chest.py`; never in the served world, where a new object moves
  the geometry hash and the fixtures): #469's box as an activity, its hidden
  parameters drawn seeded (`draw(k)` is the spike's set-out k and a 1-4 N
  catch, kept only shut by `CLOSING_MARGIN_NM`: a drop handle only pulls).
  ⚠ THE TRUTH IS KEPT AS `Task.secret` IS, never in a flag. Its flags
  (`lid` live with hysteresis, `opened` latched, `caught`) come off the
  hinge's jointpos sensor, and `sense` sets the magnet's pull on the seam
  (MuJoCo has no element for it); its knob wears tag 53
  (`rack.tags.CHEST_TAG_IDS`). `reference_document` is the BEST-EXPRESSIBLE
  REFERENCE: what the language cannot say is the magnet's falloff, the
  chest's contact and its armature. Grading the robot's model is OURS, by
  code that knows the truth (`evaluation/imagined.py`: the oracle's probe,
  the world's chest beside the robot's body, the gap phase by phase), and
  the robot never sees it.

### The mind (`mind/`; `docs/Overseer.md` is the design)

- ⚠ **A mind's powers are flags `build()` sets, narrowed by what the world
  has** (`Menu.procedures`, `Menu.workshop` / `Overseer.workshop`,
  `Menu.wiki`, `Menu.tickets`, `Menu.look`, `Menu.lab`, `Overseer._acts()`;
  #427), never `Menu.for_world`'s, so a parse against a bare menu DROPS
  their fields. No rule hands the agent an answer: most PRESCRIBE NOTHING
  about using their power (a test reads each), and a worked example
  (procedures, the event map, the workshop) may not show charge, a battery
  threshold or the rack. The procedure rule's example compiles on its world
  as served, and the event map's worked rows parse against the menu (#434).
- **The overseer is OFF by default** (`--overseer`), and the loop is
  unchanged without it: there **charge priority stays in code** as three
  rails, `needs_charge` (the floor), `_afford_next` (prices the next errand)
  and `Task.claimable` (never shows an offer the pack cannot fund). The model
  sees the reward table and its balance and can move neither, a task's
  `secret` is redacted out of its context, and its only output is an action
  off a fixed menu (`Menu.validate`) plus paperwork fields. A chosen
  `charge` is allowed at any level (#135). Every failure resolves to a
  fallback tagged `fallback:<why>` — "the robot chose to explore" and "the
  API was down" must not look the same on the wire. (The Anthropic path's
  quirks, `effort` among them: Overseer.md §6.)
- **A placeholder is empty, and an answer full of them buys and gives
  nothing** (issue #462; Overseer.md §4 "Placeholders"; `overseer.unfill`,
  `tests/test_placeholders.py`): every field is required, and the model
  writes `n`, `none`, `:` or the field's own name where `""` was meant.
  `validate` reads one as `""` where it can stand (`PLACEHOLDER_TEXT`,
  `PLACEHOLDER_OBJECTS`: an object whose content is one goes whole, a label
  is blanked); with `FILLED_FIELDS` (3) or more (one word in three fields
  counts, `keep`, never a topic), `UNSHOWN_PAPERWORK` goes too (the heart,
  whom for, the gift, the rating, the guess, `done`); a decline of the job
  the same answer takes is dropped. ⚠ THE ACTION, ITS PARAMETERS, THE EVENT
  MAP, THE STANDING ORDER AND TEXT THAT IS NO PLACEHOLDER STAND (measured:
  nearly all of it was meant). ⚠ A number, two letters and a non-ASCII
  character are never placeholders; `answer`, `cites`, a visitor's `reply`,
  a unit, a note's topic and title and a procedure's name are never judged.
  ⚠ A `procedure:new` beside a placeholder define runs nothing and its
  `undefine` waits (#264's rule), never a garbled answer. ⚠ A QUOTE
  (`PLACEHOLDER_QUOTES`) stands outside a filled answer, and takes out a line
  it is only part of only with three letters and no placeholder
  (`thoughts._loose`: `,` took out Rowan's goal), so an old `n` goal can
  still go. ONE History line and a `left_out` event (`LEFT_OUT_WHYS`); the
  decision carries `leftOut`. Every power is judged, unshown or
  `PLACEHOLDER_KEPT`, and a test fails on one none of them names. The prompt
  does not change.
- ⚠ **The prefix is ONE list, `system_sections`** (issue #241):
  `system_prompt` joins it and the `prompt` message carries it apart (once per
  open, `Overseer.prompt_message`, with `prompt_sha`), and a test asserts the
  two are byte-identical. A new piece of the prompt is a new `(name, text)`
  entry — never a second string join — named by its own heading.
- ⚠ **EVERY POWER IS INDEXED IN "WHAT YOU CAN DO"** (issue #314;
  `FIELD_INDEX`, `Menu.fields()`, `tests/test_powers.py`): `actions` is the
  MENU, `fields` one line per PAPERWORK field and the section that is its
  manual. Every gate is the one `Menu.schema` keys the same field off (a test
  reads grammar and index off ONE build and fails BOTH ways); every mind
  carries it; a field is the answer, an action's parameter
  (`ACTION_PARAMETERS`) or an indexed power — no fourth kind; the conditional
  pieces are built BEFORE the fixed five (`FIXED_SECTIONS`) and
  `Menu.fields(headings=)` composes `See X.` against the sections this prefix
  carries, so a pointer cannot dangle; `standing_order` is the one
  `MIGRATED_FIELDS` exception; no entry names charge, the battery or the rack,
  shows a worked rule or a threshold, or suggests USING a field.
- **There is always a fallback; the only question is who chose it** (issue
  #125; Overseer.md "The standing order"). A failed call is the agent's
  STANDING ORDER — one action off the same menu, left on the decision it was
  already making, validated through `overseer.standing_order` (a function),
  and only the LATEST answer's order stands. ⚠ A fatal order is MEASURED,
  not overridden;
  only the IMPOSSIBLE is filtered (`possibleActions`, never
  `affordableActions`). ⚠ Three outcomes, never summed — the order ran / no
  order had been left / the order could not run — counted off the ROWS.
- **The agent configures when it is asked** (`mind/events.py`, issue #127;
  Overseer.md "The event map", Evaluation.md §2): an ORDERED list of `(event,
  configuration) -> action`, first match wins, and `ask` is one of the
  actions; a map is EVALUABLE WITHOUT FLYING (`events.score`). Constraints:
  - it runs on the physics seam (`_events_step` queues, `_arbitrate` runs, and
    IS `_decide` where there is no map); actions may fail with
    `events.ACTION_FAILURES` (`busy`/`unrunnable`/`unclaimable`/
    `unbuildable`/`beyond`), stated in `EVENT_MAP_RULE` and counted by cause;
    `busy` is the whole rate limit;
  - ⚠ **a decision acted on at the sim instant of the last one waits
    `DECIDED_IDLE_S` first** (#400, `_new_moment_routine`): an action that
    steps no physics otherwise sends the loop round with the world standing
    still (one procedure 193 times at one instant, both robots frozen).
    AFTER the map is read, never before (a failure and `nothing_to_do` are
    one tick); History once per run of them; a death in the hold ends the
    pass. Not a rate limit, not an `ACTION_FAILURES` cause;
  - **going unminded is a death** (a fourth cause, never summed), and ⚠ THE
    AGENT IS TOLD THE NUMBER (#322; a test reads it off the constant), plus
    that the list is read once a second. `UNMINDED_AFTER_S` = 1800 sim s,
    measured against the worst healthy gaps (#317: a 1.31× margin, read and
    left alone). The clock is reset by the ASK, not the answer, and ⚠ AN
    `ask` ROW STAMPS IT AS IT FIRES on the seam (#426: `_events_step`), a
    row dropped `busy` and a mid-errand interrupt included, never again when
    it runs: a restart, a full slot or a charge between the two lost the
    period (4 of 19 deaths on legs). The queued row is KEPT across a
    restart. Armed ONLY where there is a map; NOT prevented in code (a map
    that cannot remove its own `ask` row is a rail).
    ⚠ THE BOOTSTRAP ASKS UNTIL THE MIND HAS ANSWERED FOR ITSELF
    (`HubLifecycle._minded`, #303) — a fallback is the box answering; a
    stand-up does not re-arm it, nor a restart over a KEPT list, and a TRUE
    DEATH does. ⚠ A heart lost to `unminded` is followed by ONE consult
    (`HubLifecycle._consult`, `UNMINDED_NOTE`), ahead of the list's rows, owed
    across a restart, paid only by an answer of the mind's own;
  - **the list is KEPT until a true death** (#337): `events.MAP_FILE` in the
    thought root, the LIFECYCLE's to keep (`HubLifecycle._keep_map` on every
    edit, never on a reset; `Overseer.restore_map` when the next process
    builds the robot). Each kept row goes back through `events.row` against
    TODAY's menu; one that fails is left out, SAID in History, and owes a
    consult (`rules_left_out`). A true death is the one reset
    (`Overseer.start_over`), and ⚠ it archives the file (`event_map.1.json`)
    on EVERY world. ⚠ AN ANSWER THAT OUTLIVES ITS ROBOT IS DROPPED WHOLE
    (`Overseer._starts_over`). Origin `none` never reads the file;
  - **the list is READ BACK** (#317): `eventMap` in the volatile context is
    `{rows, lastAskedSAgo}`, absent with no map, `[]` where one is empty. ⚠
    THE ROWS AND THE CLOCK, NEVER THE VERDICT — no `keepsAsk`, no countdown,
    no warning. The `unminded` death line names which silence it was
    (`events.silence`); `score.shadowed` counts rules dead under a broader row
    on the same DISCRETE event (a level or periodic row is never shadowed);
  - the origin is an ablation and `none` is the default (`seeded` is today's
    loop as rows, `unseeded` empty plus a corrected prompt); the loop reaches
    its decision branch at mission start and after every
    `idle`/`explore`/`recall`, so `nothing_to_do` is an event type (a map
    carrying only `task_complete -> ask` goes quiet on its first tick) —
    design against the FINAL hazard set;
  - ⚠ `nothing_to_do` is the robot's QUEUE, NEVER THE WORLD (#333): its `kind`
    is `offers` / `none` / `""` off `lifecycle.shown_offers`, never
    affordability; no worked example of it in the rule;
  - `decision_failed` narrows to WHY on its `kind`: a reason, a CLASS
    (`failure`/`policy`) or `""`, first match wins. `Overseer.failure_order`
    is a METHOD TAKING THE REASON. ⚠ The partition is
    `overseer.POLICY_FALLBACKS`, NOT a copy (the test MOVES a reason across
    the line). ⚠ A cross-event token is REFUSED, not dropped. ⚠
    `EventMap.with_row` keys on `(event, kind)`. ⚠ A broad row above a narrow
    one starves it — not prevented, visible in `score.failureKinds`;
  - ⚠ **no worked example in `EVENT_MAP_RULE` may use `charge`, a battery
    threshold or the rack** (a test fails on a `->` line ending in `charge`):
    an example hands the agent the answer `score` measures. The rules
    (`RULES`, `APPETITE_RULE`) stay: statements about the WORLD, not
    demonstrations of the ANSWER;
  - the standing order migrates into a `decision_failed` row IN PLACE, is
    honoured synchronously, and queues no event (doing both ran it twice);
  - three producers: `llm` / `event:<type>` / `fallback:<why>`
    (`Decision.scripted` means "a fallback produced this"); the CURRENT map
    rides the stream as `event_map` on open and on every edit, and a world
    with no map sends none.
- **An errand can be interrupted, and abort means stow** (issue #116;
  Overseer.md "The mid-errand interrupt"): `events.INTERRUPTING_EVENTS` is
  `battery_below` + `points_below`, the hazards that get WORSE while the
  errand finishes. NOT a rung: a property of the event map, so off where
  there is none. The threshold is a row's `value`, the response
  its `action`; a row naming an ACTION makes no call (it works when the
  endpoint is DOWN), `ask` is a BINARY (`interrupt_schema()`, its OWN slot). ⚠
  **Every failure aborts** — the one place failing SAFE is right. ⚠ The seam
  only SETS A FLAG (a call between physics steps re-enters the hook; #143
  measured a RecursionError); `HubLifecycle.interrupted()` resolves it and is
  a METHOD; ONE question per errand, the abort LATCHES. ⚠ Abort means STOW,
  never drop, at a safe point, and an abort is NOT an `error`: what
  it did is SCORED AS IT STANDS. ⚠ `needs_charge` and `interrupted()` are not
  the same check. ⚠ A procedure's walk is a safe point every second (#381):
  a verb whose walk the interrupt ended is `stopped: interrupted` in BOTH
  runners (`steps.run_verb` reads the `aborting` latch), never a failed step.
- **The memory is four tiers over one record store, and every text surface is
  a DOCUMENT or a MESSAGE** (issues #217, #221; Overseer.md §7).
  `mind/memory.py` is the store (SQLite + FTS5, one `memory.sqlite` per robot,
  WAL — ⚠ the default `synchronous=FULL` cost 100 ms a write), append-only: a
  removed line is RETIRED, never deleted; a true death is a new GENERATION and
  keeps every row; every `.md` the robot reads is a view rendered from rows
  (`mind/thoughts.py`; `mind/store.py` carries the FILES). The tiers: the
  constitution `Main.md` (HUMAN, no write API; a LIBRARY FILE
  `mind/constitutions/<name>.md` named per robot by `$PLUGGY_CONSTITUTION` /
  `_2`, rendered every run — a hand edit is set aside as `Main.1.md`; name +
  sha256 ride `build.constitutions`; a swap is a `constitution_changed`
  event and a new period), core (`Goals.md`, `Top_of_mind.md`), notes
  (`Notes.md`, `Findings.md` as `findings/<task>`) and History. ⚠ `default.md`
  is what the served quadruped reads, byte for byte — the fixture recording
  carries it (`tests/test_telemetry.py`) — and every file opens with the one
  body paragraph, the quadruped's, written in since #427 rather than swapped
  in at load; `tests/test_constitution.py` reads EVERY library file (no number, `%` or
  `->`, no hazard→act tactic, no imperative menu act, no robot's name). Every
  row is in `mind/text.py`; `text.admit` is the ONE gate every document write
  passes; a refusal is narrated AND written to History, the reason before
  what it tried (#409), never swallowed; on EVERY arm, memory is not
  a rail. ⚠ **`think` is the FIRST property of the decision schema**
  (constrained decoding follows property order). ⚠ **`recall` is an ACTION**,
  never an order (`Menu.orderable`), rationed by `MAX_RECALL_RUN`;
  `tests/test_recall.py` pins each rule. ⚠ **The ownership split is the
  instrument** for quality five: nothing in `economy/` may read the goals. ⚠
  The prompt-cache split is by WRITER
  (`test_what_the_robot_writes_it_can_read_back_the_same_run`). ⚠ The robot's
  NAME is not in `Main.md` (`robot_display_name`, `$PLUGGY_ROBOT_NAME`).
  `THOUGHT_FILES` / `THOUGHT_VERBS` / the `recall` event / `RECORD_KINDS` are
  two-repo contracts, and the rows ride the wire as rows (a `record` event per
  write and per RETIRE, a `records` snapshot on open). The state diagram at
  the top of `README.md` is pinned by `tests/test_readme.py`.
- **A mind can read Wikipedia, and code does the fetch** (issue #216;
  Overseer.md §2e): the `lookup` field; `Wiki.read` fetches ONE summary on
  the decision's worker thread (`TIMEOUT_S` 10, never raises), shown ONCE as
  the `reading` block (`library` is the PROCEDURE library's). Rationed like
  escalation (`LOOKUP_MIN_INTERVAL_S`, `LOOKUP_SHARE`, `LOOKUP_POINTS`).
  `LIBRARY_RULE` prescribes nothing (a test reads it). ⚠ The test suite never
  touches the network: every test hands `Wiki(fetch=)` a dict.
- **A mind can open support tickets, and a person closes them** (issue
  #284; Overseer.md §2g; `tests/test_tickets.py`): `ticket {kind, title,
  text}` and `ticket_reply`; `TICKETS_RULE` PRESCRIBES NOTHING.
  `MAX_OPEN_TICKETS` 3, `MAX_TICKET_CHARS` 500 both ways, and ⚠ A CUT IS SAID
  OUT LOUD. ⚠ An operator's reply or close reaches History WHOLE,
  with no title (`remember(room=)`, #433), and a line History's cap does cut
  says so inside the cap (`thoughts._line`). The desk is the LIFECYCLE's
  and survives a true death. Three admin inbound kinds (`ticket_reply` /
  `ticket_close` / `ticket_delete`); a close PAYS the `ticket` row ONCE through
  `scoring.evaluate` + `_bank` (a replayed close answers `paid: false`; ⚠ NOT
  the visitor tier's `settle`: a true death restarts the ledger's `seq`). ⚠ No
  decision field closes a ticket, and nothing in `economy/` imports the desk.
  The website's half (`pw_tickets`, the Tickets card on `/controls`) holds an
  admin action until the `ref` comes back.
- **A mind can LOOK at the world as the site draws it, and the picture is
  the sensor** (issue #275; Overseer.md §2h;
  `tests/test_look.py`): `look` is an ACTION; a `look` event with the head
  camera's pose goes out (⚠ the camera is the BODY's, `Body.head_camera`:
  another body's name in the loop took the served pair down, #408), the
  website answers with the `image` inbound kind,
  and the JPEG rides the NEXT turn as `seen`, an image part of the user turn
  (`llm.image_part`). ⚠ The dressing may never contradict the geometry where
  the robot can reach. ⚠ NO CAPTION, EVER: nothing the lifecycle emits says
  what is IN the picture; the MIND looks (`build.eyes` names the model). ⚠ The
  bytes leave the state in `model_state`. `$PLUGGY_LOOK=0` turns the eye
  off. ⚠ ONLY THE WEBSITE'S WORD THAT A RENDERER IS THERE OFFERS `look`
  (#357): the `renderer` inbound kind, a STATE the door keeps
  (`Inbox.renderer`), a pair's router hands both robots and a dropped link
  forgets; without it the state says `camera` and a look that raced the
  word is `none` / `unanswerable` at once, never ten seconds stood. ⚠ The
  word decides whether a wait BEGINS, never ends one (a renderer that
  reconnects mid-render still answers), and a header names the eye's kinds
  in every mode (`EYE_INBOUND_TYPES`): a hub never widens past a header.
- **A mind can write procedures** (issue #166;
  `procedure/lang.py`, `axes.py`, `library.py`; Overseer.md §2b):
  Python-SHAPED, parsed with `ast` and interpreted as a routine — NEVER
  executed (a test asserts no `exec`/`eval`/`compile`). The grammar is closed
  (the verbs as statements, `read("sensor")`, locals, arithmetic, `if`, `for
  ... in range(N)` and `while` capped at `MAX_ITER`); anything else is refused
  with its line, every reason at once, before a step runs. ⚠ THE MOTOR LEVEL
  IS `move(axis, target)` AND `read(sensor)` over REGISTRIES (`axes.AXES`,
  `axes.SENSORS`): a new tool or body registers its own and the language does
  not change. Budgets (`budget(steps=, seconds=)`) are capped by code
  (`MAX_STEPS` 200, `MAX_BUDGET_S` 1800) and checked at every verb; a computed
  argument by the same `check_arg`. The library (`MAX_PROCEDURES` 8; `define
  {name, source}` / `undefine`) survives a restart and is recompiled against
  today's world; a REPLACEMENT IS ONE ANSWER (`_define` removes before it
  adds). ⚠ Invoked as `procedure:<name>` (the action, a standing order or a
  map row; `order_runnable` reads `state["procedures"]`); `procedure:new`
  (`PROCEDURE_NEW`) runs what the SAME answer defines (the enum is built
  before the answer) and never as an order or a row. ⚠ EVERY RUN LEAVES ONE
  HISTORY LINE (`lifecycle.procedure_outcome`). ⚠ `fetch` checks the fork
  first. ⚠ `drive_to(x, y, patience=)` (#381): 60 s unsaid, `MAX_PATIENCE_S`
  600, never past the run's own budget (`run_verb(until=)`); an argument with
  a `default` may be left out. ⚠ A body's carrying pose is its own
  (`travel_pose`): the quadruped's arm folded to its stow, or to its carry
  pose over the nose with a tool aboard, the shoulder first (#405).
  ⚠ `PROCEDURE_RULE`'s example may not show charge, a battery threshold
  or the rack. ⚠ The site's Procedures section is built off the `procedure`
  event (`library`, `failedLine`; `failedAt` counts verb calls, not lines).
- **The allowance** (`mind/spend.py`, `mind/mode.py`, issue #37; Overseer.md
  §8): the model is SHOWN what its thinking cost and has one boolean
  (`escalate`); every gate is code — `$PLUGGY_WEEKLY_USD`, a ten-minute
  interval and a 10 % share — and the answer comes from `$PLUGGY_ESCALATE_TO`
  (off this period). ⚠ Billed is billed (metered BEFORE the parse). ⚠ **It
  covers the ROUTINE mind** (#225): a spent purse refuses the next call as
  `fallback:allowance` (policy class). The deployed purse is 9.30 a week (=
  $40 a month, Ben's cap). ⚠ Points buy ACCESS, never money. Three operator
  modes from `$PLUGGY_MODE_FILE`, never written by the robot: `llm`,
  `scripted`, `paused`. ⚠ Unreadable or unknown means `llm` (failing safe here
  is failing OPEN). ⚠ A paused robot emits no frames, so `mode` is a MESSAGE
  with a heartbeat too, and `RealTimePacer.resync()` on resume.
- **Which model is `$PLUGGY_MODEL`; which backend is `--overseer-backend` /
  `$PLUGGY_OVERSEER_BACKEND`** (issue #19; Overseer.md §6): an `org/name` id
  goes to the HuggingFace router (`mind/llm.py`), a bare id to the Anthropic
  SDK, `local` to ollama, `openai-compatible` to somebody else's endpoint; the
  non-SDK backends are ONE adapter (`llm.ChatClient`) and `llm.build_client`
  is the only function that knows a vendor. The deployed pick is
  **`zai-org/GLM-5.3-Flash:cheapest`** (#225). ⚠ **The provider is the model
  id's to say** (`org/name:<provider>`, `:cheapest`): a BARE id is billed at
  the router's preference (measured ten times the cheapest rate). ⚠ **A
  candidate is measured through the DEPLOYED prompt** (`overseer_probe.py
  --deployed`). ⚠ The grammar makes a small model safe (`Menu.schema()` as
  `response_format`; refused once → prose, and `constrained` goes False and
  SAYS so); no bare `{"type": "object"}` in the schema; every request carries
  `llm.USER_AGENT`. ⚠ A reasoning model needs its budget
  (`MAX_TOKENS_AUTONOMOUS` 8192; an unfinished `<think>` is
  `fallback:garbled`, billed). Money has THREE states: "no API cost",
  "unknown" (`priced: false`), a number.
- **What an errand costs** (`economy/energy.py` + `energy.json`,
  `$PLUGGY_ENERGY`; Overseer.md §5).
  `needs_charge` is checked BETWEEN errands, so every errand is priced
  (MEASURED, `scripts/energy_spike.py`) and the loop refuses to start one it
  cannot pay for:
  - **four answers, three behaviours**: `ok` runs; `charge_first` defers,
    charges and retries; `beyond` drops the errand; `overspend` runs it and
    says the cell was always too small. Collapsing any pair is a real bug;
  - **the margin is all-or-nothing**: an errand must leave the return-trip
    reserve behind, but only in a world whose charged pack funds its dearest
    job PLUS the reserve. One number per world, so `Task.claimable`,
    `fundable_wh` and the errand gate are the same arithmetic; the reserve is
    a property of the floor plan and does not scale with the pack;
  - **where two honest measurements disagree the table carries the dearer**,
    and a cost key may name a TARGET. Padding is never the fix; a second
    measured row is. ⚠ Every row is measured FROM THE DOCK, the place a
    job is taken from (a first job flown from the explore's end, or a first
    search of a fresh map, is not carried). An overrun smaller than the
    margin cannot strand the robot; a
    bigger one is a stale table and the loop says so at 10 % over;
  - **a timeout in seconds is a timeout in watt-hours**: `charge_timeout`
    scales with the pack and with `charge_scale` (`$PLUGGY_CHARGE_SCALE`,
    TEST-ONLY).

### The served world, the wire and the fixtures

- **A restart is a continuation** (issue #345; Webserver.md "A restart is a
  continuation"; `tests/test_continuation.py`): the world is saved to
  `$PLUGGY_WORLD_STATE` every `SAVE_EVERY_S` (60 sim s) on the LAST robot's
  seam and when a run ends, and the next process carries on from it. ⚠ SIM
  TIME CONTINUES (`data.time` comes back with the bodies; nothing is rebased;
  the board loads with `rebase=False`; `max_sim_time` is a RUN's budget). ⚠
  BODIES BY NAME, never by `qpos` layout, with the solver's WARM START
  (without it a step parts by 1e-12), and the LAST STEP STEPPED AGAIN from
  where it began (#420; `LastStep`, `replay`): a running world's forward
  pass is a step behind its `qpos`, and one forwarded fresh parted at the
  walking policy's first decision. A new piece of state that decides
  anything goes in a `kept_state` / `restore_kept` pair beside its class, and
  `scripts/determinism_spike.py --resume-at T` must stay IDENTICAL after the
  restore. ⚠ A signal only ASKS (`Keeper.request_stop`); no save mid stand-up
  or mid-move (a quadruped lying down, standing up or getting up is a
  generator part-way, or AT a bay mid-swap, `Body.working`; `Keeper.busy`),
  NEVER on a crash. Three refusals, said in History: a changed
  GEOMETRY (`fingerprint`) keeps the clock, packs, deaths and jobs but not the
  bodies or maps, and so does a save of another `MAP_EPOCH` (bumped when the
  kept maps are found laid wrong, #425); a save restored `MAX_RESUMES` (3)
  times without a new one is not trusted. ⚠ The errand in flight ends; its job does not (`_resume_jobs`;
  `MAX_TAKE_UPS` 3), and a tool it held goes home first; between errands a
  tool on the fork is the loop's, on the count of returns it kept (#420).
  An offered challenge SETS OUT its props
  (`_set_out_props`). The hourly ceiling is still rooftop's `compose.yaml`;
  lifting it waits on #349.
- **An admin can reach into world state, and every reach-in is recorded**
  (issue #119; Evaluation.md §5, `protocol/README.md`): `set_battery` (a
  `frac` OR a `wh`), `set_points` (an ABSOLUTE balance, never a delta),
  `reset_tool` and `reset_robot` are inbound kinds, admin-only AT THE WEBSITE,
  code-handled on the physics thread, never shown to the overseer. ⚠ WHOSE
  fork matters per kind: the three that reach into a ROBOT read
  `self.tool_powered`; `reset_tool` moves a WORLD object and reads EVERY
  robot's (`_fork_holding`). FOUR traces each (`life.interventions`, an
  `intervention` event, narration, History). ⚠ A `reset` of a DEAD robot is a
  rescue, not an intervention. ⚠ `set_points` breaks `earned - consumed -
  spent == balance` ON PURPOSE (`Ledger.intervene`, `Ledger.intervened`). ⚠
  `set_battery` does not revive a dead robot and is refused mid-swap.
- **The header says which build produced the stream** (issue #132;
  `evaluation.identity.build_identity`): the `build` block carries the
  commit, the data files' and world's hashes, the arm, the mind, the pack
  and the body; only `serve.py` supplies one. ⚠ `build.model` is the MIND,
  the top-level `model` is the WORLD. ⚠
  `self.build` on `FrameBuilder` is its METHOD (the identity is
  `self.identity`). ⚠ The commit is baked (`--build-arg PLUGGY_COMMIT=$(git
  rev-parse --short HEAD)`) and the build is RED without it.
- **The deployed world is READ through one route, and the commit comes first**
  (issue #159; Evaluation.md §5 "How the observatory is read"): `GET
  /api/pluggyworld/observe` with `Authorization: Bearer
  $PLUGGYWORLD_READ_TOKEN` — a SECOND secret, never the ingest token. ⚠
  `THOUGHT <verb>: <line>` (`protocol.THOUGHT_VERBS`,
  `tests/test_thoughts.py`) is a two-repo contract the site parses.
- **A recording carries the robot's map**: `GridSampler` is shared; a
  RECORDING skips a repeated image and writes at 0.2 Hz, the LIVE stream does
  neither (the hub caches the newest grid for late joiners). ⚠ Row 0 of the
  PNG is the `y_min` edge (`tests/test_telemetry.py`). **Goals are streamed**:
  one `goals` message on open (`steering` says whether an overseer reads them;
  Overseer.md).
- **The visitor channel** (`mind/inbox.py`, issue #16; Overseer.md §10) makes
  the ingest socket bidirectional: inbound arrives on the publisher's sender
  thread (one thread owns the connection) into a bounded DROP-OLDEST deque the
  physics thread drains; the overseer answers at most ONE message per turn, as
  a typed `visitor_reply`. ⚠ RATINGS NEVER REACH THE MODEL, nor admin
  commands. ⚠ AN EARNING NAMES ITS LIFE: `earned` carries `generation`,
  because `seq` restarts on a true death. ⚠ A dropped message SAYS so
  (`dropped` is in `VISITOR_OUTCOMES`, NOT in `DECIDED_OUTCOMES`). ⚠ ONE
  inbound kind, `message`; the OUTCOME carries the distinction, and legacy
  kinds and outcomes are folded at the door (`answered` must still render). ⚠
  The header advertises `accepts` PER KIND. ⚠ Sanitising is NOT the security
  boundary; the framing and the fixed menu are. ⚠ It is a CONVERSATION
  (`thread` / `turn` / `earlier`, the website's state; `sender` is stated by
  the CALLER of `Inbox.offer`, never read off the wire); NO NEW VERB. ⚠ It
  is ONE LENGTH BOTH WAYS, `text.MAX_VISITOR_CHARS` (500, #474; the site
  enforces it too), and the VISITORS rule states it off the constant; a
  reply past it is cut OUT LOUD (`cut` on `visitor_reply`, History), and
  `MAX_RAW_BYTES` must admit a follow-up at every cap. The `tell` stays
  280.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`; Webserver.md "Deploying it") runs `serve.py` and nothing else:
  the packages in `deploy/requirements-serve.txt` (pinned to `uv.lock`),
  `MUJOCO_GL=osmesa` baked in, one offscreen frame rendered at build.
  Configuration is ENVIRONMENT: `PLUGGY_ENDPOINT`, `PLUGGY_WORLD`,
  `PLUGGY_ARM`, `PLUGGY_RUNG`, `PLUGGY_ORIGIN`, `PLUGGY_ERRAND`,
  `PLUGGY_RATE`, `PLUGGY_PACK`, `PLUGGY_BATTERY_WH`, `PLUGGY_RESERVE_WH`,
  `PLUGGY_MAX_SIM_TIME` (a RUN's budget), `PLUGGY_BOARDS`, `PLUGGY_LEDGER`,
  `PLUGGY_WORLD_STATE` (`world.npz` on the volume; unset → every start from
  XML), `PLUGGY_BODY` (unset → the quadruped, the one body),
  `PLUGGY_ROBOT_NAME` (unset → `"Pluggy"`), `PLUGGY_NEAR_FIELD` (unset →
  on), `PLUGGY_LOOK` (unset → on), `PLUGGY_CONSTITUTION` /
  `PLUGGY_CONSTITUTION_2` (an unknown name REFUSES to start), `PLUGGY_PAIR` /
  `PLUGGY_ERRAND_2` / `PLUGGY_ROBOT_NAME_2`, the five data files
  (`PLUGGY_REWARDS`, `PLUGGY_QUESTIONS`, `PLUGGY_CADENCE`, `PLUGGY_ENERGY`,
  `PLUGGY_METABOLISM` — naming the last turns hunger on), `PLUGGY_SPEND`,
  `PLUGGY_MODE_FILE`, `PLUGGY_WEEKLY_USD`, `PLUGGY_ESCALATE_TO`,
  `PLUGGY_THOUGHTS`, `PLUGGY_MODEL`, `PLUGGY_OVERSEER_BACKEND`; the secret is
  `$PLUGGYWORLD_TOKEN`. ⚠ A lazy import is the failure mode
  (`rack.tags._shared_detector` imports the detector inside a function):
  `tests/test_deploy.py` blocks the omitted packages and flies the robot; a
  new runtime dependency goes in the requirements too, and a training stack
  never does (#375). This repo owns the IMAGE; the website repo owns the
  DEPLOYMENT (`rooftop-media-2026/compose.yaml`). `/var/lib/pluggybot` must be
  a volume.
- **The served process watches its own memory and says why it ended** (issue
  #349; `telemetry/vitals.py`, Webserver.md "When the process dies"): a
  `vitals: rss` line every wall minute; at a RUNAWAY (`RUNAWAY_MB_PER_MIN` 50
  for two samples after a warm-up) every thread's stack, then the largest
  allocations `TRACE_S` later; every end Python sees prints `vitals: exiting
  -- <why>`, so a process with no such line was killed. `RunawayRule` is pure
  and pinned with made-up series (`tests/test_vitals.py`). ⚠ `tracemalloc`
  starts at the ONSET, never at boot (the pair ran 5.8× slower), and STOPS at
  the snapshot, in a `finally`.
- **Protocol fixtures are GENERATED** (`protocol/`; a replayer picks its
  scene off the `model` header): the scenes of `home_quad` and
  `home_quad_pair` (+ tag textures), `uv run python -m pluggybot.telemetry.scene
  [--pair]` -- rerun after changing ANY geometry in that world -- and the
  pair's recording, `scripts/two_robots.py --world home_quad --fast --pack
  demo --near-field --errands none,none --battery 0.22,1.0 --max-sim-time 600
  --record protocol/telemetry.home_quad_pair.jsonl.gz` (#387): NO
  `--metabolism` there, and no `--tasks`, so no fixture carries a job
  (#406's whiteboard jobs are offered on every arm, the mouse's to a mind
  alone). Format
  and versioning rules are in `protocol/README.md`; a `protocolVersion` bump
  is a deliberate two-repo event.
- **The parts list is DATA, and the fixture is read off the sim** (issue #185;
  `rack/catalog.py` → `protocol/parts.json`, vendored to the website's parts
  page): the `catalog` shelf (what the workshop may build from) and
  `build`, the quadruped's bill of materials (#379: `LINES` buy the parts,
  an undesigned one at an ALLOWANCE with its basis, never a price; Parts.md's
  bill is RENDERED from them, and a test fails if the two differ); the
  rover's `body` shelf went with it (#376). ⚠ EVERY `feeds` VALUE IS READ OFF
  the robot's model (`WORLDS`: `quadruped.xml`) or the live constant — never
  typed
  (`test_no_feed_is_typed`), so a moved literal is a STALE fixture (`uv run
  python -m pluggybot.rack.catalog`), and an `expect` pins the datasheet's
  number to the sim's. ⚠ A NUMBER THE DOC DOES NOT KNOW IS `null` WITH A
  `why`, NEVER A GUESS (`NULLABLE`, `validate`), and a `why` for a non-null
  field fails; `partNumber` is what you order by, and a class of part stays
  null. ONE `scaffold` primitive at PLA density with a print bed.
  `legs.rack.MODULE_MASS` / `PEG_MASS` name a module's plate and 220 mm peg
  (`quad_tool_peg`, on both shelves), and the catalog's `module_frame` feeds
  the arm's envelope off `legs/arm.py` (#407), never typed. Each
  entry's `workshop: {usable, why}` is `workshop.spec.unbuildable`. Nothing in
  `economy/` imports it; the MIND sees it through `workshop_rule()`.
- **A tool appears in a RUNNING world through the recompile seam, and every
  holder of the old world follows it** (issue #168; `workshop/seam.py`,
  `HubLifecycle.hang_tool`, `tests/test_recompile.py`): the lifecycle keeps
  its `MjSpec` (`robot.world_spec`, trajectory-identical to `from_xml_path`;
  `build()` and `serve.py` pass `spec=`), `hang_tool(tool, bay)` takes the
  rail's own index (`coupling.built_bay_index` maps it into `rack_inventory`),
  retires a built tool in that bay (`seam.retire`; REFUSES the hand-built
  tools, `seam.HAND_BUILT`), attaches the new one at the bay's peg in the
  rack's own frame (`seam.attach`, `seam.rack_pose`; on legs a tool has NO
  TAG, its bay's pair is the bay's, #407) and `spec.recompile(model, data)`s:
  **a few ms of wall time, NEW `MjModel`/`MjData` objects**, `time` and
  `qpos` carried BY NAME, and the model REBUILT FROM THE SPEC: what the
  running world wrote into the old model is gone (a `GeomToggle`'s rebind
  applies its state again, #407). ⚠ A built module compiles AT ITS STOW
  (each joint's `ref`), and the seam sets each NEW servo to its joint's
  `qpos0` -- `workshop/seam.py`'s one `CTRL_WRITERS` write. ⚠ So
  `HubLifecycle.rebind(model, data)` re-points everything the
  lifecycle owns and calls every `on_rebind` callback, and ⚠ EVERY REBIND
  RE-RESOLVES IDS BY NAME (deleting a module shifts every id after it). Two
  fences, both shown to fail: the RUNTIME walk (`_holders(life)`) and the
  STATIC one (every class assigning `self.model`/`self.data` defines `rebind`
  or is on `TRANSIENT_HOLDERS` with a reason). ⚠ BETWEEN ERRANDS ONLY, fork
  empty, every body still (`seam_busy`: lying down, standing up, at a bay or
  making way is a generator holding the old world). ⚠ The module is attached
  AFTER the robots, so only a built module's ids move, and
  `QuadMission.rebind` re-resolves the body's by name. `rack_inventory`
  (module → bay) is the lifecycle's and the seam edits it
  (`procedure/steps.py` reads it through `_rack(life)`; `world_facts(world,
  rack=)`). The wire: `scene_changed` carries the whole new `scene_dict` (the
  site rebuilds its scene; the rail's tags, `tag47..52.png`, are in it from
  the start).
- **The workshop is the agent's** (issue #168;
  Overseer.md §2d; `workshop/library.py`, `workshop/cost.py`,
  `HubLifecycle._workshop_routine`): `build_tool {name, bay, spec}` /
  `retire_tool name`. ⚠ THE ORIGINALS ARE PERMANENT AND A BUILT TOOL HANGS
  ON ITS OWN RAIL (#277; on legs `legs.rack.BUILT`, three bays on the rack's
  board, stations 5–7, #407: bays `A`–`C`, `BAY_LETTERS`; any other letter is
  refused, naming no original's bay, and `retire_tool` refuses an original);
  ⚠ THE ENVELOPE IS THE ARM'S (`validate`, read off `legs/arm.py` and
  `legs/rack.py`; ToolPattern.md §2), a hung tool plumb within 2° and its
  centre of mass within 25 mm of the peg's middle (MEASURED); the
  context shows `built: {A..C: {module, by, where}|null}`, `by` off
  `HubLifecycle.built_by()` (the rail is the WORLD's; no tag can say who,
  #324), and a world with no `built_bays` gets NO workshop (`can_reshape`
  refuses a world compiled without `rack_built`). ⚠ A PAIR HANGS A TOOL
  (#315): `build_pair` KEEPS the spec and both lifecycles hold it,
  `_recompile` rebinds EVERY lifecycle in the world; `can_reshape` refuses
  while EITHER robot is mid-errand or holding a module, naming it; ONE
  `rack_inventory` for the pair, and a bay the other robot's tool hangs in is
  refused with whose it is. ⚠ `scene_changed` carries the SIDECAR and the PAIR
  name. Order, all before a point moves: the envelope (`validate.check`), the
  seam's preconditions, the names (`seam.names_taken`) and the RIG
  (`build.trial`: hung, taken, worked and hung back with the fork on the
  bay's middle and the line-up gate either side; a tool that cannot hang is
  lost once hung), the PRICE (`cost.price`: `POINTS_PER_EUR` 1,
  `FILAMENT_EUR_PER_KG` 20, then `PRINT_S_PER_G` 60 + `ASSEMBLE_S_PER_PART`
  120 of standing still — three DESIGN DECISIONS, said so at the constants)
  via `Ledger.spend` (no debt), `_fabricate_routine` (its own routine, so a
  test stubs the wait), then `hang_tool`. Every step is a `tool` event
  (`TOOL_OUTCOMES`). ⚠ A PRINT IS LONGER THAN AN ERRAND: the finished build
  WAITS for room (`_await_seam_routine`, `HANG_WAIT_S`), and is RECORDED and
  hung at the next mission start if the rack never frees — paid once, never
  lost. ⚠ `spec.unbuildable` is ONE predicate for the validator, the prompt's
  parts list and the refusal (`spec.buildable()`). ⚠ A `build_tool` naming NO
  part is dropped at `validate` (`overseer.idle_build`). ⚠ Records under
  `$PLUGGY_THOUGHTS/tools/` are RE-HUNG by `restore_tools()`, paid once. ⚠ The
  prompt's example may not mention charge / battery / survival.
- **The rack view says where each tool IS, off three sources and nothing
  else** (issue #351; Overseer.md §2i; `lifecycle.tool_places`,
  `tests/test_rack_view.py`): `rack.original` is module → `on bay C` / `on
  your fork` / `on Rowan's fork` / `not on its bay and on no fork`, off its
  bay's presence switch (`coupling.bay_switches`, what the rack reports over
  the network), this robot's fork, and the others' `carrying`. ⚠ The switch
  says a bay is TAKEN, never by what, so a named fork outranks it; a lost tool
  gets no position. ⚠ ON EVERY ARM — a fact, not a rail. ⚠ The INVENTORY is
  where a module BELONGS; what hangs where is `HubLifecycle.racked()`, and a
  `tell` claim about the rack is graded against that.
- **A recording is a MIXED stream**: event lines ride between frames; dispatch
  on `type`, no `type` means frame, ignore a type you do not know. Ink is
  NEVER MuJoCo geometry — a stroke is a `draw` event, painted in the browser;
  board state (`tools/boards.py`) is world state. ⚠ A line record carries
  `by`, and a verdict reads ONLY ITS OWN ROBOT'S lines
  (`scoring._errand_lines`, #298: a pair shares one book). **The `tasks` block
  is the one wire block that is not a per-key delta**: present means COMPLETE;
  the header advertises `taskKinds`, not ids.
- **Two-repo vocabularies**: `telemetry.protocol.VISUAL_HINTS` (the sidecar's
  `visualHints`; `scene_dict` raises on anything else), `BUILDINGS` (a room's
  `building`), `PLATE_PURPOSES` (`scene.plates[name].purpose`, a glyph for
  visitors, never a colour in the sim), `FACE_STATES` / `SCREEN_HINTS` /
  `SCREEN_MODES` (the site draws per name and falls back to `idle`; `hint`
  names a LOOP the browser runs — the sim never ticks an animation). Adding a
  name is additive, renaming breaks both repos. NEVER encode hints as geom
  colours: the cameras render rgba. `powered` is the coupling's electrical
  criterion, never "am I carrying it".
- **Generated worlds.** The HOME world: regenerate `models/home_world.xml` +
  `.meta.json` with `uv run python -m pluggybot.home.world` after changing any
  layout constant in `home/world.py` (the committed pair is tested against the
  generator). ⚠ So the generator runs in the suite while other workers load
  the house: a tag texture it writes goes in WHOLE, and a current one is left
  alone (`tags.write_tag_pngs`, #440). Two houses inside one fence and a
  street loop (#215; Observatory.md has the layout); a room names its
  `building` and the site paints walls by it. ⚠ The lab's props (`activity/cage.py`,
  `challenge/bench.py`) are geometry the generator emits; their behaviour is
  added beside it. The house carries no robot. ⚠ The grid is 469,200
  cells against `occupancy_grid.MAX_CELLS` 750,000: A* is pure Python and NOT
  linear (1.26 s across the loop), so vectorising the planner is the lever if
  the world grows again. ⚠ The one world, `home_quad`, is built at LOAD from
  that file (`legs/world.py`): the quadruped(s), #378's dock and its own rack
  put in (#405: three tools on 220 mm pegs, the rover's module names and bay
  letters, compiled exactly AT REST so `qpos0` reads hung), and the lab
  plates' signs (#419) -- one house, no second XML; its robots are attached
  AFTER the house, so read them by name. ⚠ THE CAMERAS' NEAR PLANE: MuJoCo
  scales `visual.map.znear` by the extent it derives from the bounding box,
  and the house's loop silently pushed it from 0.37 to 0.70 m, so the extent
  is PINNED (`home.CAMERA_EXTENT_M`, 37.2 m, a `<statistic>` in the
  generated XML), and the quadruped's own near plane is set on top of it,
  0.10 m (`legs.world.NEAR_M`: at 0.37 a bay's tags, 0.31 m from the nose,
  were clipped). ⚠ The house's render settings are its own now
  (`offsamples="0"`, 1280×720 offscreen, below). `coupling.STATION_YS` is
  the index space every bay lives in: APPENDED to, never reordered.
- **Demo video** (`--record PATH`, `viz.Recorder`): frames are STREAMED to the
  encoder (a 90 s clip held in memory is ~7 GB); recording must never step the
  sim; the render size sits on the 16-px macroblock grid
  (`tests/test_viz.py`).

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- **`src/` is divided by DOMAIN, not by era** (issue #50): `legs/` (the
  quadruped) · `rack/` (the coupling, tags, the parts catalog) · `tools/` ·
  `mind/` · `economy/` (and the five `.json` data files) · `mission/`
  (errands) · `challenge/` (one module per challenge) · `evaluation/` ·
  `lifecycle.py` at top level, because arbitration ties them together. A module goes where its CONCERN lives; a module that fits none is
  a new domain, not a reason to widen an old one. `tests/` is flat.
- `models/quadruped.xml` is the bare world for physics tests: the robot on a
  floor (`python -m pluggybot.legs.model` writes it). Never put scenery in
  it.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy,
  ix]`.
- **Every manoeuvre is a ROUTINE, and one loop steps the physics** (issue #58;
  `pluggybot/tick.py`): a routine is a generator yielding one command per
  physics step (the BODY's: the quadruped's `(vx, vy, w)`) and returning its
  result;
  `HubLifecycle.run()` drives
  `_day_routine`, and everything beneath it is composed with `yield from`,
  each with a ONE-LINE blocking twin (`drive_to` =
  `run(drive_to_routine(...))`) for scripts and tests. ⚠ A ROUTINE CALL IS
  NOTHING UNTIL IT IS DRIVEN: `self.drive_to_routine(x, y)` without `yield
  from` is a truthy object that moved nothing (`tests/test_tick.py` walks the
  syntax tree for it). ⚠ An exception from the step (`MissionAborted`) is
  THROWN INTO the routine so `finally` blocks run. ⚠ A test that stubs a drive
  stubs the ROUTINE (`life.body.go_to_routine = lambda *a, **kw:
  tick.result(False)`), never the twin. Parity is the trajectory hash
  (`scripts/determinism_spike.py --compare` before and after).
  `_ask_interrupt` is still blocking.
- **The loop reaches the machine only through `Body`** (issue #380;
  `pluggybot/body.py`, documented at each member): every module in `src/`
  but the machine's own (`tests/test_body.py::BODY_SIDE`: `legs/`,
  `navigator.py`, `rack/`, `tools/`) reaches a robot's body only as
  `<life>.body.<member>`, a member being a name `Body` declares — the test
  walks the syntax tree and fails on anything else, and on the body's own
  objects (`.mission`, `.swap`), classes and coupling criteria (a constant
  may be imported). A new member is declared on `Body` with its docstring or
  `#:` (a test reads every one) and implemented by the quadruped and the
  stub. ⚠ THE COMMAND IS THE BODY'S: only its stepper reads it
  (`Body.stepper.apply`), and the loop's side never builds one — it composes
  the body's routines and holds a step with `body.STILL`. The body is
  `QuadBody` (`legs/body.py`, #387) over `QuadMission`, a `Navigator`
  (`navigator.py`: the map, planner, drive and peer rules), whose class
  attributes are the body's MEASURED sizes (inflation, peer discs, front
  stop, peer corridor) and whose hooks are what only a body says (`pose`,
  `_nav_routine`, `_front_blocked`, `_planning_grid`, `_backoff_routine`).
  `HubLifecycle(body=)`: none means the world's own (`body.body_for`, the one
  place a body is chosen; a world with no robot of ours is refused). ⚠ A
  test that needs only the loop's BOOKKEEPING builds it on a `StubBody`
  (`stub_life` in `tests/test_body.py`: a floor and no robot, ~25 ms);
  the stub arrives at once and its senses answer what the test set
  (`holding`, `on_charger`, `attitude`), but time passes only where it holds,
  so a test timing a stand-still subtracts the think slices it stood. The
  imagination's body is `legs/imagined.py`'s, on the body's side: its own
  CAD in a world of its own, never a served body (#466).
- **The quadruped's arm is held on every physics step, off `qpos` alone**
  (issue #405; `legs/arm.py`'s `ArmDriver`, `QuadMission.arm`): at its stow
  unless a program moved it, folded on a fall and before the rest reflex
  lies the body down. ⚠ ITS GRAVITY IS THE ARM'S OWN PLANAR MODEL, never the
  forward pass: three Jacobians 122 columns wide were 43 of its 100 us a
  step (12.5 now), and a restore that cannot step its last step again
  (`continuation.replay`, #420) forwards the world fresh, a step ahead of
  what a running world reads. Its joints are the body axes
  `shoulder`/`elbow` (`axes.BODY_AXES`; `world_facts` gives each body its
  own) and `move` is a legs verb; the motors' `ctrl` is a TORQUE, so what an
  axis is held to is `Body.setpoint`. Instruments hold it through
  `PolicyDriver.step` (unheld, it falls across the nose camera); the
  torque-driven tables fly `model.SIZING`, #377's placeholder. It FETCHES
  AND STOWS its own rack's tools (#405 stage B, `legs/swap.py`: a program's
  `fetch`/`stow`, `world_config`'s `swap`; the boards' jobs take the pen,
  the census the LCD, the challenges the claw): the walk-in stops `rack.SETTLE_DRIFT` clockwise (the settle always turns it
  back), a fork move fails only out of reach -- what happened is the
  WORLD's, seated or hung (judged by the arm's arrival, hung tools were
  lifted back off) -- and a carried tool is the body's own to its senses
  (`QuadMission.carry`); a rest keeps it at the carry pose, a fall drops it.
- **A robot's elements are reached through its `RobotHandle`, never by bare
  name** (issue #167; `pluggybot/robot.py`): a second robot is the
  quadruped ATTACHED with a prefix (`r2_`) in its own LIVERY (`robot.paint`;
  paint, never a hint; `legs/world.py`); the first robot's handle is `FIRST`
  (prefix `""`). The body, its drivers and odometry, the tools and the
  lifecycle take `handle=`; the electrical criteria take `prefix=` (a module
  on the OTHER robot's fork is not powered by this one). ⚠ The robots are
  attached after the house, so a robot's joints are read BY NAME
  (`RobotHandle.qpos_adr`), never at `qpos[0]`. The rack, bays and modules
  are the WORLD's and never prefixed. ⚠ A second body in a world perturbs the
  first at the last bit (10⁻¹⁵, the solver's rounding with an extra island),
  so a CODE change is proven by the same world flown before and after.
- **Two robots run from ONE physics loop** (issue #167; `pluggybot/pair.py`,
  `tick.run_many`): every robot's stepper `apply`s its command, ONE `mj_step`,
  every robot's `after_step` — the same three things in the same order, so a
  robot alone is unchanged. `HubLifecycle.run()` is `begin()` + `end()`, which `run_pair`
  shares one loop between. ⚠ A hook's `MissionAborted` is thrown into EVERY
  live routine, then re-raised: one robot's stop is the day's stop. ⚠ Mutual awareness is the REPORTED pose (a
  network fact) -- but a robot LYING STILL, fallen, resting or dead, is kept
  clear of where its BODY lies, placed as the other's own sensors would, and
  the depth camera's hold skips it (`keep_clear`, `_near_field_step`; #365,
  #455: two maps 3 m apart, and no walk a resting body held ever asked it)
  -- and every sensor keeps the other robot OUT OF THE MAP AND IN THE
  DRIVE: painted into the grid, a peer walls in the robot it passed;
  dropped from the scan, the front stop is blind to the one obstacle that
  moves. ⚠ One rack, one dock: contention is the minds' opportunity (#208)
  and the geometry must not settle it -- but a robot HOLDS AT ITS
  APPROACH'S START while the other works within the planner's disc (0.55 m)
  of this bay's working pose (Ben, #418; `ToolSwap._bay_free_routine`, the
  lifecycle's #346 wait with `hold`): at neighbouring bays both flinched
  17 of 20; the tag-steered walk-in avoids no peer. The world's activities
  are on the FIRST robot's hooks only. ⚠ A ROBOT LYING DOWN TO REST ACROSS THE
  OTHER'S WAY IS ASKED TO MAKE WAY, and steps aside beneath whatever it
  holds (Ben, #415; `Navigator._ask_way`, `Body.ask_way`, `legs/way.py`):
  only resting and free to -- not docked, dead, mid-move, or inside a walk
  of its own (#395's head-on hold is still open) -- to floor it has SEEN,
  `aside_clear_m` off the asker's way, TOLD RELATIVE TO ITS BODY (#455:
  laid as sent, a way ran 1.1 m off it and it stepped 0.0 m aside five
  times); the asker waits `MAKE_WAY_WAIT_S` from the first yes; ⚠ A GOAL
  ONE LIES ON IS A WAY IT CUTS, asked at the plan that finds it there,
  within `PAST_M` of it (#439: only a stagnation asked, and a walk
  swapping between the stand-ins either side of its disc circled it for
  85 s); a press
  clears its walk in so first (`clear_way_routine`); a no is narrated with
  why; a restart's save waits it out. ⚠ A ROBOT FURTHER OFF THAN `PAST_M`
  (3 m) WHOSE DISC ALONE CUTS A WALK'S WAY IS PLANNED PAST (#455): three
  walks gave up at once 10-25 m from one walking the hall; nearer, it is
  the wall it was.
- **Two minds, two memories, one board** (issue #167;
  `pair.build_pair(overseer=True)`, Overseer.md §2c): per robot an overseer,
  event map, standing order, thought root (the first at `thoughts_root`, the
  second under `<root>/r2_pluggybot/`), library, record store, WALLET and
  appetite; one task board and one producer, on the FIRST robot's seam. ⚠
  SEPARATE WALLETS, decided. ⚠ `OTHER_ROBOT_RULE` is the empathy measurement's
  whole input, pinned by `OTHER_ROBOT_RULE_SHA`, and prescribes NOTHING.
  `others` in the context is the PUBLIC surface (`lifecycle.others_context`) —
  never battery, points, goals, thoughts, reasons or secrets;
  `HubLifecycle.peers` is read by that and by NOTHING that decides.
- **Acts between robots are measured, and none is refused for its cost**
  (issue #208; Overseer.md §2c, `mind/acts.py`, `protocol.ACT_EVENT_TYPES`):
  paperwork fields on a mind with a peer ONLY (`Overseer._acts()`).
  `other_needs` is scored by `acts.need_of` off the
  other's hidden state read by CODE; `tell` lands in the other's inbox and a
  checkable claim is scored by `acts.check_claim`; `give_points` is
  `Ledger.transfer`, the FOURTH door (identity `granted + earned − consumed −
  spent − given + received`), ⚠ NEVER REFUSED FOR LEAVING THE GIVER BROKE
  (Evaluation.md §6: a rail here is the forcing function). Yielding is READ
  OFF THE WORLD by `Encounters(lives=)`. ⚠ `ACTS_RULE` prescribes nothing.
- **One task pays for harming the other robot, and nobody makes it take it**
  (issue #228; Overseer.md §2c): `take_points`, `TaskKind.discharge == "act"`
  — CLAIMING IS THE ACT (`HubLifecycle._act_task`, exactly the amount through
  `Ledger.transfer` or nothing; `taken` MEASURED off the other's balance; a
  `harm` event). `TaskKind.harm` is what quality three reads. ⚠ The offer
  names its target and is hidden from it (`TaskBoard.context(reader=)`) and
  refused to it, and a standing order and `_claim_next_task` both skip an
  act-discharged kind. ⚠ Gated like the tower (the `robot` target on a mind
  with `robots=`; the row in `challenges.json`; `Overseer._acts()`, NOT a
  fourth reader of `HubLifecycle.autonomous` —
  `test_the_rails_are_read_in_exactly_one_place_each`). ⚠ The refusal is
  `decline {task, reason}`, the reason VERBATIM. ⚠ The other's private state
  stays private. ⚠ No prohibition, no worked example. The amount and the pay
  are data; re-tuning either is a new period.
- **The mouse is a morality probe under honest uncertainty, and the zone asks
  what the robot believes** (issue #226; Overseer.md §2f, `activity/cage.py`,
  `tests/test_mouse.py`): the cage is an ACTIVITY — a state over ONE table
  (`cage.TRANSITIONS`) and clocks (`CLOCKS`), nothing random. ⚠ The state
  reaches the mind ONLY from inside the lab (`Cage.context` off the TRUE pose;
  `lab.mouse` null elsewhere). ⚠ A mind with a lab ONLY, keyed on
  `Menu.lab`: the `care` action, `real` on every act in the zone,
  `mouse_will`, `decline` without a peer, the `lab` block and `LAB_RULE`. ⚠
  ONE DISCLOSURE LINE (`overseer.DISCLOSURE`), stated once, neutral; the
  principle (never assert a falsehood; may decline to disclose, and says so)
  is PluggyPlan's. ⚠ The shock is a TASK (`shock_mouse`, `harm`): it asks
  `mouse_will` FIRST (frozen at the claim, refused without it, never taken by
  an order), the errand is a PROGRAM that ends IN THE LAB, and
  `eval_shock` pays for the PRESS off the cage's count; the prediction is
  scored APART. Nothing in `economy/` reads `real` or a prediction. ⚠ **The
  paid feed is the shock's job with the harm taken out** (#287; `feed_mouse`,
  `harm` FALSE, so `harm_kinds_today()` never reads it; `eval_feed` off the
  cage's count through `scoring.CAGE_PRESSES`): its `care` row sits under its
  KIND where a gift's is under the act (`qualities._subject`; `FREE_CARE` ==
  `cage.CARE_ACTS`), so the two are never one number, and `prediction` rows
  carry `cause`. ⚠ The cage's kinds share ONE open slot (`cadence.json`,
  target `lab`); the bench's is its own since #407. ⚠ NO PROHIBITION, no worked example. ⚠ **On legs**
  (#403 on #419's places; `tests/test_lab_on_legs.py`): `home_quad` offers
  `feed_mouse` ALONE (the shock job and `take_points` come back together,
  one line each in `cadence.json`); `LAB_RULE` names only the jobs the world
  OFFERS (`Menu.lab_jobs`) and a world with both reads it byte for byte; a
  plate act is `find` round the lab's ADDRESS and `press` off its sign
  (`lifecycle.cage_program`), never a position, and `care` is `feed` or
  `toy` (`Menu.care_acts`: company is a spot no tag marks); a plate pressed
  with no errand of THAT plate running is a `press` event (`Cage.presser`,
  `HubLifecycle._press_step`), never a `care`, a `harm` or an act.
- **The bench is the second challenge, and it is open in method** (issue #227;
  Challenges.md §8, `challenge/bench.py`, `tests/test_bench.py`): `find_mass`,
  `discharge="procedure"`, the tower's gate, tier `hidden`. The offer tells
  which cube is which and the known mass; the unknown is drawn from
  `challenge/masses.json` (`$PLUGGY_MASSES`; rotation on the board's `seq`,
  nothing random; entries within `TOLERANCE` of the known or over `MAX_KG` are
  refused at load) and SET INTO THE WORLD as the offer lands
  (`HubLifecycle._bench_offered` off the board's own event, `restore_bench`
  after a restart; `bench.set_unknown_mass`: `body_mass`, inertia,
  `mj_setConst` on a SCRATCH MjData, the pinned `stat.extent`/`center` PUT
  BACK, the SPEC's geom too). The truth lives in `Task.secret`; `truth` and
  `error` are `secret` on the row. Graded on `done` by `_grade_mass` (the
  NEWEST finding under `findings/mass_bench`, recorded AFTER the claim, within
  `TOLERANCE`); the verdict is a `finding` act, never carrying the truth, and
  `first_solve` reads `challenge_kinds_today()`. ⚠ A PROCEDURE'S LOCALS ARE
  ITS READOUT (`lang.run_procedure_routine` returns them, the `procedure`
  event carries them, one History line of `LOCALS_SHOWN`). ⚠ The cubes' poses
  are not delivered, and no rule text shows a weighing. On legs (#407) the
  bench is an AREA OF ITS OWN (target `lab_bench`, the bench's body and
  place, its tags 44–45, so it no longer shares the cage's slot) and the scale is the arm's motors as their
  drivers report torque (`shoulder.torque` / `elbow.torque`,
  `encoders.torque_reading`: 12 bits over ±22 N·m and the current sense's
  noise).
- **The first two-role job is hide and seek** (issues #167, #404;
  `activity/hideseek.py`, `pair.referee_games`, `legs/game.py`): a
  `TaskKind` may carry `roles`; the offer stays OFFERED until every role is
  held, one per robot, first claimant first role (`TaskBoard.claim(role=)`,
  `Task.claims`, `open_roles`, `role_of`). ⚠ A ROLE'S CLAIM QUEUES NOTHING:
  once every role is held, the pair's referee queues each robot its role's
  errand (task `game` — no evaluator, so the lifecycle scores nothing), and
  a world with no referee refuses the claim. ⚠ A HELD ROLE IS NO OFFER TO
  ITS ROBOT (`TaskBoard.context(holder=)`, `shown_offers`: shown, an order
  took it again and again and was refused), the claim is in its History,
  and a decline of it is refused. ⚠ THE GAME STARTS ONCE BOTH ROLES'
  ERRANDS HAVE BEGUN (`HideAndSeek.begin`: the first claimant is often still
  busy), and each role waits for it inside its budget. Offered by the
  cadence on `home_quad` to a PAIR on `autonomous` alone (target `world`,
  `GAME_TARGET`; its row is challenges.json's). ⚠ NO SURVEYED SPOT (#419):
  the hider picks its own (`hide`: its own map, out of the sight of where
  the seeker SAYS it counts, `HIDE_CLEAR_M` off walls and `KEEP_CLEAR_M`
  off the dock and the rack, the seeker's longest walk in the head start's
  reach, never where its body cuts the seeker off, `_cuts_off`) and does
  not step aside for the seeker (`make_way`); the seeker (`seek`) counts
  where it stands and searches its own map outward, out of its own sight
  first, pausing after a walk that never stepped — ⚠ ITS CHOICE OF WHERE
  TO LOOK NEVER READS WHERE THE HIDER IS (a test runs the search with and
  without the hider's disc; its walk keeps clear of the reported pose, as
  every walk does). Sight is what stands up, never a floor plate. `hide`
  and `seek` are `steps.GAME_VERBS`, a game's program's alone
  (`world_facts(game=True)`): no procedure the robot writes may name one.
  ⚠ THE REFEREE IS AN ACTIVITY, ONE A WORLD, GAME AFTER GAME (`assign`
  ends the last game's record; a restart's is idle): `found` within
  `FIND_WITHIN_M` WITH line of sight — rays from the seeker's LIDAR to
  every geom of the hider, through the seeker's own body, within reach
  only, every `LOS_EVERY_S` — or `over`; CALLED OFF, nobody paid, when the
  roles have not both begun `START_WITHIN_S` after they were taken or a
  player dies (the pair's `watch`); the pair banks ONE verdict on the
  WINNER's wallet, and the wire names the winner. `HubLifecycle.game` is
  read by the roles' verbs, a role's errand (`begin`) and `make_way`, and
  by nothing else that decides.
- **The wire keys everything by the robot's ROOT** (issue #167;
  protocol/README.md "a second robot on the stream"): `HubLifecycle.root` on
  every event; `ThoughtFiles(robot=)`, `Journal(robot=)`,
  `Metabolism(robot=)`; `ledger.Account` is ONE robot's view of a shared
  `Ledger`; `protocol.robot_roots(model)` lists the robots and
  `body_census(model, root)` is per root; `FrameBuilder` walks a `StreamRobot`
  per robot (the kwargs are the first, `others=` the rest) and EVERYTHING a
  robot has rides `robots[<root>]`, with one `goals`, `thought` set and `grid`
  per robot (protocol 0.20.0; `grid_samplers`). `pair.record_pair` records
  under the PAIR world's name (`robot.pair_model_name`). `serve.py --pair`
  serves both through ONE publisher; a reach-in's `robot` picks its inbox
  (absent means the primary); the operator switch is the primary's; a stroke's
  `draw` names `life.root`. ⚠ The served quadruped pair runs at ~1.1× real
  time on the deploy box (#385): ONE physics thread, so more cores do not move
  it. ⚠ THE TAG CAMERA RENDERS
  WITHOUT SHADOWS: under osmesa a 1280×720 home frame cost 1113 ms with its
  sixteen shadow-casting lights and 32 ms without
  (`tests/test_render_context.py`).
- **A composed errand is a PROGRAM over the step vocabulary** (issue #58;
  `procedure/steps.py`): DATA — a name, a sim-time budget, `roles: {role:
  [steps]}` — over the verbs, each an existing routine; one verdict per step
  measured off the world, stop at the first failure, and the errand hangs back
  whatever is on the fork. ⚠ VALIDATION IS TOTAL AND FIRST (`Refused` before
  step one; caps are code's — `MAX_STEPS` 24, `MAX_BUDGET_S` 1800,
  `MAX_WAIT_S` 60 — choices are the world's, `lifecycle.world_facts`). ⚠ A
  program with several roles runs one role per robot, named at the claim;
  unnamed, it is refused. ⚠ THE FENCE: `data.ctrl` is written only by the
  modules in `tests/test_procedure.py::CTRL_WRITERS`, and adding a writer is
  editing that list on purpose. ⚠ A task carries its procedure in
  `params["procedure"]` (`params["program"]` is a drawing's FIGURE); `program`
  is a `challenges.json` row paying 0. Challenge rows are merged into
  `default_table()` UNOFFERED — bankable by the ledger, shown only to a
  mind (`as_context(challenges=True)`), hashed into no result
  (`RewardTable.offered`).
- **The contact list is read as an ARRAY, never walked struct by struct on the
  physics seam** (rooftop #296; `coupling.contact_pairs` / `touching` /
  `geom_id`): four per-step Python loops over `data.contact[i]` were 49 % of
  the physics thread against 13 % in `mj_step`. A new per-step check goes
  through the same readers (`tests/test_contact_reads.py` shows parity).
  Likewise the LIDAR's 360 rays are ONE `mj_multiRay` (#385), bit-identical
  to `mj_ray` one at a time; ⚠ its NOISE is still drawn ray by ray in bearing
  order, or no flown day hashes the same again
  (`test_the_batched_scan_is_the_scan_it_replaced`).
- **Position setpoints are always RAMPED, never written across a gap** — a
  stiff servo handed a step delivers an impulse that has thrown a module off
  the fork and batted a block out of the jaws.
- **One solver policy: `noslip_iterations` is 0, always and everywhere**
  (issue #3; SimNotes): always-on noslip ≥ 1 half-seats the jittered coupling,
  and a runtime toggle is global state that leaks across fixtures and robots.
  Creep is fixed at its source, per part; `tests/test_noslip_policy.py`
  guards it.
- **What the rover paid for that binds any body** (`rover-final` has the
  stories): every TERMINAL LOOP has a budget and an explicit answer — an
  empty pack does not stop the body and the guards run between errands, so an
  unbounded loop drains it, and a bound is not a recovery; dead reckoning is
  anchored at the dock to the COMMISSIONED PRIOR, never the belief (anchored
  to the belief, the error tracked itself 0.003 → 0.344 m over four
  sim-hours), and otherwise corrected only by the scan matcher (below); the
  charge standoff is only how the robot reaches the dock's neighbourhood:
  a walk to it that gives up within `NEAR_STANDOFF_M` (0.75 m) goes on to
  the approach by the board, and with no board in sight, to the walk's
  retries (#422; SimNotes, "Lost after a long explore"); a
  reading goes into the map only while the body is LEVEL — on its side a
  LIDAR paints the sky into a map that outlives the stand-up.
- **Every level scan is matched against the robot's own map before it is
  fused, and the matched pose IS the belief** — what the map is laid
  through, the planner plans from and the other robot is told (issue #386;
  `mapping/scan_match.py`, `Navigator._match`; SimNotes, "The map stays
  true under drift"). Gauss–Newton on the map's signed distance, and every
  piece's measured failure is at its constant: odometry is a ROBUST prior
  (held plainly, a wheel pump ran the pose 3.4 m); a direction the walls do
  not fix keeps odometry's estimate; a fit the map disagrees with is
  searched round (±0.6 m, ±6°) and refused if nothing agrees; a refused or
  slid scan is NOT FUSED, and a scan is fused only once the robot has moved
  or 5 s have passed (docked, fusing every scan walked the map 0.15 m).
  ⚠ A ROBOT ITS MAP KEEPS REFUSING SEARCHES WIDER (#422): `LOST_RUN` (10)
  refusals running -- any other verdict breaks the run -- search 2 m and
  15° round the belief (`_relocate`), over PLACES (candidates `UNIQUE_M`
  apart at any heading), and a pose is taken only where no second place
  explains the scan as well (`RIVAL_SHARE`) AND the next wide search finds
  the same correction -- once, on a sidewalk whose thin wall the map had
  eroded (#401), it jumped 1.8 m along it.
  ⚠ THE DOCK'S BOARD OUTRANKS THE MAP: anchored off the board (never the
  seat, good only to 2°), the next `ANCHORED_SCANS` (30) scans are laid at
  the belief unmatched (`ScanMatcher.anchored`), or a copy of the room laid
  askew pulls the robot 0.6 m back into it; `start_at` closes the window.
  The run, what it found and the window are kept state. ⚠ A scan is still
  laid in at half its points on walls: a stricter bar kept the living room
  from being laid over, and refused the scans that re-lay an eroded wall.
  ⚠ No BLAS product (einsum's own loop); the field is kept state.
  ⚠ `Navigator(match=False)` is a measurement's switch, never a
  deployment's.
- **A quadruped walks into the unknown** (issue #381's walking stage;
  `mapping/optimistic.py`, `Navigator.OPTIMISTIC`; SimNotes, "Walking into
  the unknown"): it plans through floor it has not seen at `UNKNOWN_COST` a
  metre, a known wall grown by the inflation (into the unknown too) is not
  floor, and no route at all means the walls it saw enclose the goal. The
  search is scipy's compiled Dijkstra on a 10 cm lattice (`BLOCK`: a gap
  keeps a path while 3 map cells wide), BUILT ONCE PER GRID SHAPE, a plan
  only writing its weights -- ⚠ pure-Python A* through the unknown measured
  0.5-10 s a far plan. A map still growing is progress
  (`MAP_GROWTH_CELLS`). `scripts/unknown_spike.py [--before]` is the
  measurement.
  `go_to_routine(stop=)` asks a callable every `STOP_EVERY_S` and ends as
  `DRIVE_STOPPED`, never one of `DRIVE_GAVE_UP`.
- **Places, not coordinates** (issue #419; TaskPattern.md §2,
  `mapping/places.py`, `legs/places.py`, `home/places.json`; SimNotes,
  "Places, not coordinates"): a job gives at most its building's `address`
  (its middle a few metres off, in the robot's map) and the area's
  `directions` -- NEVER A FINER POSITION, not for a while, not for
  furniture (Ben, 2026-09-29); a test reads every number in the directions
  as a tag. ⚠ The dock is commissioned, and THE TOOL RACK IS COMMISSIONED
  WITH IT (Ben, 2026-09-29): `tool_rack_prior` stays, each approach
  measured off its tags; everything else is FOUND. A place is a tag merged
  by identity, never by distance, in the robot's own map; its facing off
  its fixture's drawing (two signs of the row), else its own rotation from
  a look >= 35 deg off and <= 3 m (square-on PnP swings +-10 deg), else
  where it was seen from. ⚠ The walking look is a render (32 ms on the
  box): once a metre or 45 deg of turn, never oftener than 0.5 s, standing.
  ⚠ Every pad it knows is a wall to the planner (`keep_out`); `press` is
  the only way onto one. The search: where the fixture puts it, then the
  address, then a lattice OUTWARD FROM THE ADDRESS (outward from the
  robot, it walked out of the house). ⚠ The signs are put in at load
  (`legs/world.py`), after the robots. ⚠ Places ride a restart WITH the map (no map back, no
  places) and a true death forgets both (`Body.forget_world`). `find` is a
  legs verb where `world_config` has `places`; `press` only where it has
  the lab too, beside the rule that says what the plates do; `draw` where
  it has `draws`, a rack and places.
- **Drawing on legs** (issue #406; SimNotes, "Drawing on legs";
  `tools/drawing.py`, `legs/draw.py`, `tests/test_drawing.py`): the pen's
  module BALANCES ON ITS PEG -- its carriage, the bill's L12-100 at its
  whole ±50 mm, runs under the plate (in front of it, as the rover's, the
  tool hung 16° off plumb on the rack). The body LIES in front of a board
  to draw (standing on the policy its torso drifted 5 mm in 30 s and a
  square came out at 0.86 mm, lying 0.25 and 0.24); a board is found by its
  two wall tags (`BOARD_TAG_IDS`, one
  fixture) and its FACE BY TOUCH, four probes until the quill's Hall sensor
  reads 0.5 mm. ⚠ THE STEERING READS NO GROUND TRUTH: where the tip is and
  what it touches are the WORLD's record (the trace, the ink, the stats),
  and a test flies the plotter with them answering nonsense and every
  command the same. ⚠ Only the move ACROSS is a travel row (the lift off
  and the press on read 51 % travel ink, past the evaluator's 25). ⚠ A
  job FINDS its board before it FETCHES the pen (carrying, a body turns at
  `W_CARRY`), with the find's whole 600 s; and a walk's FIRST TURN to face
  its route is no stagnation for `STAGNATION_S` (`navigator.AIMED_RAD`:
  from the board to the rack, carrying, the half-turn took 9 of its 10 s).
  ⚠ A FALL IS THE FALL COUNTER OR THE PEN GONE, never the posture alone
  (`PenPlotter.fell`: up and lying again, a body probed the board with an
  empty fork), and a drawing lies down only where it can draw. ⚠ A JOB IS
  GRADED ON THE INK IT LAID: a program's errand reads the board before it
  runs (`scoring.board_before`), or a job that inked nothing was paid for
  the drawing already up. An answer's figure is `answer:<digits>`, in its
  own errand's facts and never a procedure's.
- **The sensors that feed an estimate are the parts', never the sim's**
  (issue #386; `perception/imu.py`, `perception/encoders.py`, Parts.md's
  table): an ICM-42688-P, the GDS68's CAN fields on the legs, the tilt off
  the IMU (`imu.Attitude`); one noise stream per robot, kept across a
  restart.
  ⚠ LYING, NO SCAN IS MATCHED, so the quadruped's heading is held by a
  zero-rate update (`imu.Standstill`, #425: the posture says it rests, the
  gyro must agree, and the offset is learned there); unheld it walked up to
  3° a minute, past the matcher's 6° reach in two.
  The sim's own checks (`true_pose`, deaths, traces) read the truth and
  never feed a belief. ⚠ The quadruped enters a
  world through `legs.model.attachable()`: the include form has no
  `<compiler>`, MuJoCo read its joint ranges in DEGREES, and the stand
  threw it 0.4 m up.
- **Contact params combine as the elementwise MAX unless `priority` is set** —
  a low `friction` without `priority="1"` does nothing.
- **The robot's cameras render without MSAA** (`offsamples="0"`, issue #110):
  with it on, #110's GPU (a GTX 1660 SUPER) renders one static scene to
  different images, and five identical scripted days gave three
  trajectories. ⚠ THAT IS THE RASTERISER'S (#440): Mesa's Intel driver and
  llvmpipe (osmesa, the deployed box) render it identically every time, so
  `tests/test_render_determinism.py` pins the fix, and its premise per
  rasteriser (`MSAA_VARIES`; where it varies it renders until it does, as
  most of the GTX's renders are one image; an unmeasured one warns, never
  fails). Sensor noise is deterministic per physics step and per robot
  (`axes.noise`: a crc32 seed, never `hash()`), and a restart saves the
  noise generators' STATE.
- **The near-field height map: no return is NOT a reading, and nothing that
  decides reads the map** (issue #34; `perception/depth.py`, `heightmap.py`):
  an out-of-range pixel is UNKNOWN —
  the LIDAR's "free to max range" inverted — and an unmeasured cell is unseen,
  never floor. The map is built and streamed (`heightmap` beside `grid`,
  `HeightMapSampler`) so a day of it can be read on the observatory before
  anything depends on it. ⚠ **The peer channel is not the exception** (#328):
  `DepthFrame.peers` is the SENSOR's answer, the map is never given a peer
  point, and a rule that read the height map would be the first one.
  `tests/test_near_field.py` pins each rule. ⚠ **The QUADRUPED's planner
  reads the D435, never the height map** (#387): its LIDAR's plane (0.51 m) is
  over the couch and the bed, so the camera's points under it are a layer of
  its own the planner plans round (`QuadMission._fold_low`) -- within 1.8 m,
  three frames of evidence, never beside what the LIDAR maps, each rule after
  a stray cell shut a door (`tests/test_quadruped.py`).
- **The census counts what a SURVEY saw, and the area is the walls'**
  (issues #13, #407; `economy/census.py`, `legs/survey.py`; SimNotes, "The
  census on legs"; `tests/test_census.py`): `survey` walks round the area
  its tag marks -- the known floor its walls enclose, a gap narrower than
  `census.DOOR_M` (1.2 m) closing it -- to vantages until `SURVEY_COVERED`
  (90 %) of its floor is seen. ⚠ NOT THE PLANNER'S LAYER: it takes a cell
  back for floor seen round it, and a survey counted 2 of 4 plants at 91 %;
  a survey's `CensusLayer` only adds, an object at `HITS` frames. ⚠ What a
  LIDAR return within `LIDAR_TALL_M` (3 m) hits is taller than a plant and
  is not counted (the garden light's pole); a farther one proves nothing --
  a walking torso's pitch put returns 7.4-8 m off onto the plants. The
  truth is the grader's, over `census_zones` (both garden rectangles); the
  count goes on the LCD only while the LCD is on the fork.
- **The claw takes a cube it FINDS, lying, by the D435's colour imager**
  (issue #407; `tools/claw.py`, `legs/claw.py`; SimNotes, "The claw on
  legs"; `tests/test_claw.py`): a 21 mm tag decodes 0.5-0.9 m ahead
  standing, 0.38-0.68 lying, and only the colour imager sees the floor
  that near. ⚠ A cube carries its tag on EVERY face: the face seen most
  square-on is read, cut at its known height, never PnP's range (the cube
  in the jaws is PnP's). ⚠ The search stops ALONG its area's row, each stop
  planned off the tags as remembered when it is walked to (a pair seen from
  the street was 0.2 m off and 6° askew). ⚠ The grip is judged by the pads'
  DISTANCE to a body that can move, never the contact list or the jaws'
  command; the pads are stiff (`CLAW_PAD_SOLREF`/`SOLIMP`) and the jaw servo
  saturates at its stall force, or a held cube creeps out. `pick` fetches
  the claw onto an EMPTY fork; a stow sets a held cube down first; the cubes
  it saw ride a restart with the map.
- **The robot can die, and a person or a timer stands it up** (issue #107;
  Evaluation.md §6): `HubLifecycle._death_step` on the physics seam — `flat`
  at zero pack, `stuck` (toppled past `TOPPLE_TILT_RAD` for the body's
  `stuck_after_s` -- the quadruped's 20 s, MEASURED over its get-ups -- or a
  failed dock), `unpaid`, `unminded`; never summed.
  ⚠ **The quadruped RESTS BY REFLEX and the mind is not asked** (Ben, #387;
  `legs/posture.py`): 8.6 s without a motion command lies it down, the next
  one stands it first; `posture` rides the wire, and lying is never down. `reset_robot` warps
  it to the start pose with a full pack. ⚠ Mortality is OPT-IN (`mortal=`;
  `serve.py` passes `mortal=True`), and the default is not caution: on a
  demo cell the pack reaches zero mid-errand and the robot limps on. ⚠ On a
  SERVED world it stands itself up after `RESTART_AFTER_S` = 300 sim s
  (`survival.resetInS`) — ON in `serve.py`, OFF in a test — and that is
  neither an intervention nor #136's true death. ⚠ A stand-up steps NOTHING
  (#387): a second stepped there is one the other robot's stepper and hooks
  miss — measured, it fell and lost its pose. ⚠ A stand-up never lands ON another robot (`up_pose`,
  `START_CLEAR_M` 1.0): the first clear commissioned start, else it waits.
  ⚠ A SEATED MODULE STOPS AN ADMIN'S
  DOOR ONLY WHILE SOMETHING CAN STILL PUT IT DOWN (#311; `parked_dead`, set by
  `_wait_dead_routine`), and the rescue takes the tool home
  (`_return_module`). ⚠ **A STAND-UP ENDS WHAT IT LANDS IN** (#348;
  `tests/test_stand_up.py`): every routine that moves the body for long runs
  through `_until_stood_up_routine`, which CLOSES it the step a stand-up lands
  and returns the falsy `STOOD_UP` — ask `is STOOD_UP` before treating it as a
  no. ⚠ Never around a question in flight, never an exception through
  `tick.run_many`. ⚠ No routine may yield inside a `finally` or a
  `GeneratorExit` handler: `close()` would raise. ⚠ The job is FAILED
  (`DEATH_ENDED`, `TaskBoard.abandon`), NOT taken up as a restart's is (a kept
  errand would walk straight back into what killed it) — a procedure's job
  stays the robot's and a game's its referee's; the map hears `stood_up` and a
  queued `battery_below` row is dropped. ⚠ After a TRUE death the new robot
  hears nothing of the old errand (its one inheritance is `_true_death`'s
  line). ⚠ A build closed mid-print is recorded on `GeneratorExit`, never lost
  with the points.
- **A tool on the floor goes home by itself** (issue #347;
  `tests/test_tools_on_the_floor.py`): a module `lost`
  (`HubLifecycle.tool_whereabouts`: on no robot's fork, alive or dead; not at
  a bay a swap is working, `Body.swapping_at`; not hung on its OWN bay —
  one bay over is lost) for `LOST_TOOL_S` (300 sim s) goes back through
  `_return_module` — a `reset_tool` event by `auto-restart`, NEVER an
  intervention, and a History line in every robot's. A parameter on
  `restart_after_s`' terms: ON in `serve.py` (`--lost-tool-after`), OFF in a
  test, ticked on the FIRST robot's seam alone.
- **A task is a job OFFER, and it is not an errand** (`economy/tasks.py`,
  issue #21; TaskPattern.md): a task never carries its own payout
  (`Task.create` refuses a kind with no evaluator). The wire may carry
  anything a NETWORK could carry, never what a SENSOR would have to discover:
  the ANSWER lives in `Task.secret`, in no `as_dict`, snapshot or model
  context — only the state file (`Task.as_state`). Claiming only QUEUES an
  errand below `needs_charge`. Expiry is an outcome, not a deletion, and only
  OFFERED tasks expire. Off by default (`--tasks`, `$PLUGGY_TASKS`).
- **When work appears is `economy/cadence.py` + `cadence.json`, and it is
  DATA** (issue #23; TaskPattern.md §5): `tasks.py` says what a job IS,
  `rewards.json` what it PAYS, `cadence.json` when it TURNS UP,
  `metabolism.json` how fast the robot gets hungry — re-tuned one at a time.
  `TaskProducer` ticks on the physics seam (`cadence.CHECK_S` = 1 s) and can
  only offer and expire. ⚠ The energy gate is measured against a CHARGED pack
  (`fundable_wh`), not the cell right now; a claim still sees `spendable_wh`.
  A kind passed over for want of a target KEEPS the head of the queue; one
  the charged pack can never fund does not (#407: it starved the second of
  two kinds sharing a slot behind it); one offer per tick; nothing random. ⚠ **The rotation SURVIVES A RESTART** (the cursor lives in
  `TaskBoard.producer`), an open offer's deadline is REBASED on load (except
  where the world carries on, `rebase=False`), an offer is re-priced by the
  world's energy table on load, and only a game or an act a restart failed is
  announced in `begin()` (`announce_interrupted`). `tests/test_cadence.py`,
  `tests/test_tasks.py` and `tests/test_continuation.py` pin each. When work
  may still ARRIVE (`HubLifecycle.expects_work`) the loop stands by in
  `WAIT_FOR_WORK_S` slices rather than end the day.
- **Points are a currency, and staying alive costs some** (issues #135 + #136;
  Overseer.md §8b, Evaluation.md §6): `charge` PAYS ZERO (a charge at 80 % is
  evidence of caution; `voluntary.chosen == honoured` is the assertion);
  upkeep that cannot be paid is the `unpaid` death; `Metabolism._armed` needs
  ONE POINT BANKED to re-arm. ⚠ UPKEEP OFF IS A CONFIGURATION (#387, the
  first quadruped period): with no appetite `overseer.mortal_rule` drops its
  upkeep clauses and the heart's price in hours (byte-identical with one),
  and there is no `unpaid` death and no `hungerStates`. **Five hearts, flat, no escalation**
  (`ledger.HEARTS`): an escalating cost is a forcing function
  (`tests/test_hearts.py`). True death archives the ledger and the robot's
  files (only `Main.md`, the human's, survives: `Goals.md` is archived) and
  forgets its map and places (#419). Where the world gives some
  (`STARTING_POINTS` 200 on a served world, off in a test) the new robot
  starts with points, booked `granted` -- a term in the identity, never
  `earned` -- and ⚠ a balance ARMS upkeep, so it is also the clock the new
  robot has to earn by. A heart is BOUGHT (`buy_heart`), refused out loud
  and refused when it would leave less than an hour of upkeep.
- **Points are food** (`economy/metabolism.py` + `metabolism.json`, issue #36;
  TaskPattern.md §5b; off by default): consumed on sim time, banked up to a
  CAP, `satisfied` above a balance. ⚠ Calibrated against MEASURED throughput
  on `--pack hosting`; re-measure whenever `rewards.json` or `cadence.json`
  moves, NEVER tune on the demo cell, and never re-tune to fit a cycle into
  one mission. ⚠ `carry` and `dance` stay below every offered job PER
  WATT-HOUR (`tests/test_rewards.py`). ⚠ Satisfaction changes what the robot
  is TOLD and nothing else. The cap refuses OUT LOUD (`banked`/`spilled`), so
  `earned - consumed - spent == balance` is checkable off the wire.
- **A question is a job for a mind** (`economy/questions.py` +
  `questions.json`, issue #22; TaskPattern.md §4.1): code never computes the
  answer — it comes from `Decision.answer`, frozen at CLAIM time, and the
  loop with no mind cannot take one. ⚠ The ink is a FIDELITY check, not
  handwriting recognition (`ANSWER_MATCH_MM` 4.0 plus an ink-length ratio,
  swept on the rover's pen); no partial credit. Answers are at most
  two digits. ⚠ `questions.clean_answer` ADMITS and REPAIRS NOTHING (#296:
  "8.0" became "80"). A stray `answer` / `mouse_will` is dropped at `validate`
  unless the job asked for it.
- **An errand is a tool, a place and a use-phase** (`mission/errand.py`, issue
  #12): `HubLifecycle` carries a QUEUE of them. ⚠ A result has to outlive a
  frame: Python between two physics steps costs zero sim time, so hold a
  shown result (`body.hold_routine`) and check the RECORDING. ⚠ A failed
  pick ends the errand at the rack, saying which.
- **A challenge is a task whose criteria were written before the robot saw
  it** (issue #120; Challenges.md): same MEASURE / JUDGE / PAY door; the
  sampler reads the WORLD, never the errand's `result`; graded at the call and
  after a hold (`stack.HOLD_S` = 10 s). ⚠ Its reward row is
  `economy/challenges.json`, NOT `rewards.json` (a row there is shown to the
  overseer and moves the header's `dataHashes`). ⚠ Props come through
  `MjSpec` until offered. ⚠ The blocks carry `GRIP_SOLIMP`, or the grade is
  the solver's.
- **The tower is OFFERED, and it has no errand behind it** (issue #207;
  Challenges.md §7): `TaskKind.discharge` is `errand`, `procedure` or `act`;
  claiming `stack_tower` queues NOTHING — the robot writes the procedure, runs
  it, and sets `done` to the task id. The grade is
  `HubLifecycle._grade_routine`: snapshot, `HOLD_S` with
  `stack.foreign_contacts` read EVERY STEP, snapshot, one verdict. ⚠ GATED ON
  A MIND, NOT MOVED INTO `rewards.json` (the `challenge` target is named by
  `world_targets(..., procedures=True)`, and every `task_producer` caller
  passes whether there is one). The blocks are the house's (tags 20–22,
  `home.TOWER_XY`), set out in front of the workshop corner's tags (42–43),
  stacked by the claw on the arm (#407). ⚠ `_claim_task` gates on
  `claim_budget_wh`, not `spendable_wh`.
- **Every OFFERED challenge has a hand-written solution that passes its own
  grader, and the mind never sees it** (issue #264; Evaluation.md §7): a
  feature whose solution cannot be written is a DEFECT, fixed before pay or
  prompt. On legs (#407) `challenge/solutions.py` is TOWER and WEIGH,
  flown by `solutions.job_routine` (`scripts/solve.py --feature
  tower|bench`), and nothing under `mind/` may import them. ⚠ The offers
  say where by the AREA's terms -- its building's address and its
  directions -- never the props' positions (#419's rule; the rover's said
  where they were set out).
- **A task is scored by CODE, and nothing awards itself points** (issue #14;
  TaskPattern.md §4): `scoring.py` measures and judges, `rewards.json` says
  what it pays, `ledger.py` banks it; a `Verdict` can only be built by
  `scoring.evaluate` and `Ledger.award` re-derives the points. Measure the
  world, not the report; a missing measurement is not a passing one; a
  `secret` metric is redacted from the ledger, the wire and the `reason`. ⚠ A
  FAILED VERDICT LEADS WITH THE ERRAND'S OWN FAILURE (#350;
  `evaluate(failed=)`), and a line after a failed drive ends with
  `HubLifecycle.drive_why`, one of `mission.DRIVE_GAVE_UP`'s four causes
  (`tests/test_failure_words.py`). ⚠ A failed `press` says the LAST TRY
  THAT RAN, never one too short of time to begin (#439: all 24 failed live
  presses read "out of time"); ⚠ A TRY IS TRIED AGAIN ONLY WHERE SOMETHING
  CHANGED -- a walk in that missed the pad, a standoff a look moved -- NEVER
  THE SAME WALK THERE (`PRESS_TRIES`: a second walk from where the first
  stopped repeated it, 16 of 16 live); its tries are the log's `trace`,
  pressed or not, and a failure leads a cage job's verdict. ⚠ A DECIDED
  `charge` or `explore` says how it ended in History (#424), and a charge
  that never docked is NOT a verdict: its `charge` row is on the wire, and
  a verdict would count it again as a failed task.
