# PluggyBot — notes for Claude

A simulated, hardware-honest robot and the autonomous agent that lives in it.
**The project is agent-autonomy research, not a product**: the mission and
the six qualities the agent is meant to maximise are in `docs/PluggyPlan.md`
§ "What this project is for" — provisional wording, settled direction.
**#465 holds the order of work** (its "The next stretch") **and the
decisions**; #375, the quadruped pivot, is history. **The body is the
humanoid** (#507, `humanoid/`), a pair deployed in the house since #559's
stage 4 (`home_humanoid`, below). The ~10 kg quadruped before it was
deleted in #559's stage 5 (the tag `quadruped-final` is the last commit
that runs it), as #376's stage C deleted the wheeled rover (`rover-final`).
Before doing anything, read the doc that owns what you are about to touch:

| doc | what it holds | read it BEFORE |
|---|---|---|
| `docs/PluggyPlan.md` | the mission and the six qualities, status, architecture | anything |
| `docs/SimNotes.md` | simulation lessons, each ending in what is true now | touching `models/` or contact/actuator params |
| `docs/Parts.md` | locked hardware decisions and the sim parameters they feed | changing a part or its parameter |
| `docs/ActivityPattern.md` | adding an ACTIVITY (a mechanism that owns world state): sensed criteria, hysteresis + latching, pre-allocated geom/mocap toggles, telemetry | building a puzzle, mechanism or gardening step |
| `docs/TaskPattern.md` | adding a TASK KIND (a job offer): the honesty rule, the perception ladder, code-side grading, how tasks, errands and activities compose — fold any gap back in | adding a task kind or touching `economy/tasks.py`, `scoring.py` or `cadence.py` |
| `docs/Challenges.md` | grading a job nobody wrote a scorer for: a predicate written BEFORE the robot sees it, through `scoring.py`'s chain, with a hold; what it cannot grade | adding a challenge, touching `challenge/`, or reaching for an LLM judge |
| `docs/Hands.md` | the humanoid's manipulation vocabulary: what was studied, the principles, the verbs (meaning, report, failure, what each is built from), the force move, the capability check and the migration from #493's verbs | touching `HANDS_VERBS`, `humanoid/reach.py`'s moves, or a hands check |
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
  APPROVAL.** The full suite is **0:52** (2026-10-31, #559 stage 5: 1:09-1:19
  before it, interleaved on one machine — the quadruped's tests left with
  it; 0:43 on 2026-10-06's quieter machine, #473; 7:29 with the rover, #376
  stage C). Any
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
  while the narrative of how it was found belongs in SimNotes,
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
  exercises: `models/` or a world generator (`home.world`,
  `humanoid/world.py`) · contact or actuator params · `humanoid/` ·
  `navigator.py` / `behavior/navigation.py` · the charging station
  (`activity/station.py`, `humanoid/charge.py`) · the telemetry frame
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
  columns, one adapter per source (`from_observe`; and `from_attempts`, a
  demo's record, feeding the per-task record alone, #493); a later source
  ADDS rows to a shape, never a second version of it. Four rules, each pinned in
  `tests/test_qualities.py`: nothing that must stay apart is summed; no mean;
  **absent is `None`, never 0**; never pooled across a build identity. ⚠ A
  reading of the observatory is NOT a result; it reports into the issue it
  informs. ⚠ `serves` IS NOT ON THE
  WIRE: the KEY is the test, and quality five's ratio off the observatory is
  `None`. ⚠ A test reads §3's shape table against `SHAPES` both ways: a row
  with no function fails. ⚠ Nothing in `economy/` imports `evaluation`. ⚠
  **The sixth quality is NOT time alive** — five shapes read together
  (Evaluation.md §3's table), `idling` BESIDE deaths in `QUALITIES` because
  high idling with low deaths is the failure mode; hearts bought for oneself
  come off the `HEART_BOUGHT` / `HEART_REFUSED` narration, a two-repo contract
  pinned in `tests/test_hearts.py`. ⚠ THE
  PROMPT DOES NOT CHANGE FOR IT (a test reads every rule for the word): a
  quality is what we measure, never what the robot is asked to maximise for
  us.
- **A demo's attempt is one try from a FRESH set-out, graded by the world**
  (#493 stage 1; `evaluation/attempts.py`, the kit; `challenge/lid.py`;
  Evaluation.md §4, Challenges.md §9): a task is data (`Task`: chest, body,
  mount, grader), every attempt builds its set-out anew and a reset is the
  world's help, counted on the row; the call is where the skill ended or
  its budget did; one verdict through `scoring.evaluate`, a row of the
  record an attempt, the skill's own word beside it and never read; an
  attempt that raises or whose world MuJoCo reset (`Unstable`) has no
  verdict, its task kept and unflown from there. ⚠ A SKILL IS HANDED THE
  BODY, never the world (the chest's truth is the harness's). ⚠ THE DEMOS'
  ROWS ARE `shown: false` and pay 0 (`TaskReward.shown`: the lid's, and
  #525's drawer, door and place): a mind with a library sees every other
  challenge row, and these would change the deployed prompt. ⚠ A FALL
  FAILS THE TRY (Ben, 2026-10-25, #544; it was lost time from 10-17, and
  learned skills fell to open a door): every grader's criterion, a fall
  the body's own watch counted (`Body.falls`; the humanoid's is its trunk
  35 deg past the posture its legs hold, so a bow is none, #520) from the
  attempt's start to the hold's end, said FIRST in the verdict
  (`lid.judged`) and stated in the job the robot is told (`practice.JOBS`).
  The job is the lid inside the TASK's time limit, the budget
  (`attempts.LIMIT_S`, the grader's `limitS`: 3x the humanoid reference's
  slowest pass at 0f98e24, 68 s); its tilt is recorded, the body's to say
  -- the grader names no part (a row's `tool` and `toolSeated` are null
  vestiges, below). It imports nothing of the chest's: everything loads
  `scoring`. ⚠ Held-out chests start at `HELD_OUT_FROM` (1,000,000), drawn
  by the frozen commit's first seven hex digits (`seed_of`; a pod's
  `COMMIT` file, a dirty tree refused), and a reading says what it
  flatters (`FLATTERS`) and that it is weaker than a sealed pool.
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
  `RULES` says so, and `model_state` drops the code-computed verdict
  (each offer's `claimable`) at PRESENTATION only — ⚠ the view narrows, the
  state does not (`order_runnable` reads `claimable`). ⚠ A0 hides the survival clock, or A0 and A1 are one run.
- ⚠ **NO SCRIPTED ROTATION IN A MIND, EVER — INCLUDING LIVE.** Every action
  originates with the LLM (a decision, a standing order, or an event-map row
  it configured); with no answer and no order the robot finishes what it is
  doing, runs what is queued, and IDLES — even if that ends in death. There
  is no `scripted()` to reach (a test asserts it). Evaluation.md §2.
- **The deployed world flies `autonomous`, both robots, origin `unseeded`**
  (issue #206), the humanoid pair since #559's stage 4 (`$PLUGGY_BODY`,
  the site's compose; a body is a period too, Observatory.md). Which arm is
  `serve.py --arm/--origin/--rung` (`$PLUGGY_ARM`
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
| `scripts/hub_lifecycle.py` | the mission, one humanoid in `home_humanoid`: explore, charge at its station, its queue (`--program PATH`, a JSON program or a `.procedure`; `--errand` is `none`), and once the queue is empty an overseer's choice; battery-driven. `--boards PATH`, `--tasks`, `--metabolism`, `--near-field`, `--overseer`, `--pack hosting`, `--record out.jsonl.gz` |
| `scripts/serve.py --endpoint ws://host:port` | the mission headless, paced to real time, streaming the protocol over an outbound WebSocket; the sim never blocks on the socket. `--free-run` measures the real-time multiple; `--pair` serves both robots; the world is `home` with the humanoid pair in it (`home_humanoid`, deployed since #559's stage 4; `--body`/`$PLUGGY_BODY` names the one body there is); `--world-state PATH` keeps the world and carries on from it (#345); `$PLUGGYWORLD_TOKEN` is the ingest secret (never a flag — `ps` is public). docs/Webserver.md |
| `scripts/ws_sink.py` | dummy sink for serve.py: counts, frame-gap stats, keyframe spacing; `--token` makes it refuse an unauthenticated publisher |
| `scripts/overseer_probe.py` | REAL LLM calls against a synthetic state: tokens, cost per sim-hour, cache hit rate, the latency distribution (`--calls N`). `--model org/name[:provider\|:cheapest]` measures a HuggingFace candidate (`$HF_TOKEN`, in the gitignored `.env`); `--deployed` measures the prompt the served humanoid pair sends and reports the ENERGY GATE (`report_energy` counts who took the unaffordable offer); `--prompt` prints it section by section with its sha; `--deployed --image JPEG` attaches a look's picture to every call (does the model, on its provider, take an image beside the schema? #541); `--max-tokens N`, `--escalate-to X --force-escalate`, `--tokens-only` (the Anthropic path's limits: Overseer.md §6) |
| `scripts/energy_spike.py` | what each errand COSTS, on an oversized pack; `--write` folds it into `economy/energy.json`, `--reserve` measures the return-trip margin, `--actions` prices named acts: on the humanoid in `home_humanoid` (#559), its explore, its station's charge and its hand jobs (`open_drawer`, `open_door`, `place_cube`, `open_chest`, each from the station, the area remembered, its reference flown). Re-run after anything that changes what an errand does |
| `scripts/grid_spike.py` | #558: what a robot's map costs as the world grows, at 1-4x today's cells (the bounds grown round the house), each scale its own process, through the humanoid's own code: each scan's, frame's and plan's work, the explore's mask and A*, the matcher's field, the wire's PNG, `world.npz`, memory, on #215's synthetic grid and the house laid by the robot's own senses; `--walks [SIM_S]` the humanoid pair walking #507's routes at each scale; `--scales`, `--repeat`, `--no-table`, `--out` |
| `scripts/determinism_spike.py` | is the world the same world twice? N scripted days hashed, first divergence attributed to GPU / decoder / raycast; `--pair` (`--fall-at`, `--drain-at`); `--compare DIR`; `--resume-at T` flies a day against one saved and carried on in a new process (#345); `--no-economy`, `--sequential`, `--nthreads` |
| `scripts/make_way_spike.py` | #415 on two legs (#559): the humanoid pair in the house on true-floor maps, one robot resting standing in a doorway and the other walking through it, one process a scene: the walk's end, the step aside (how far, how long, why it stopped), touches and falls; `--before` nobody asks, `--finished` the resting robot's day is over, `--scene A,B`, `--drift`/`--drift-walker` a map laid off the world, `--standing S`, `--near-field` |
| `scripts/station_spike.py` | the humanoid's charging station (#559 stage 2; SimNotes "The humanoid's charging station"): `--approach [--n N] [--parallel P]` docked and unplugged or why not from seeded starts round the standoff, one process a trial; `--seats` started at 140 seats over what the seat check accepts: plugged in and unplugged or why not; `--lost [--n N]` a fresh robot 2 m out, its belief set 0.4-1 m wrong: the loop's charge path, its lost search and the fix; `--hold [--seconds S]` plugged in: the charger's watts against what it draws; `--fall` shoved over plugged in: the breakaway, the plug home, the robot up; `--walk-in [--explore 120,240]` explored, walked out to 8 places across the house on its own map, then the loop's own `go_charge`; `--view [plug\|fall]` in the viewer, looped (MUJOCO_GL unset); filmstrip `station_spike.png` |
| `scripts/solve.py --feature drawer` | ladder A of #264 on the humanoid's hands (#559): a hand job (`--feature drawer\|door\|cube\|chest`: `open_drawer`, `open_door`, `place_cube`, `open_chest`) offered on the storeroom's counter, its prop drawn (`--draw K`) and set out, claimed, the reference's two procedures (`challenge/solutions.py`) run as the robot's, `done`, graded on the seam; a fresh humanoid on a fresh map from the station (`--from station\|store`, or `tour`: flight k from the next of eight places across the house), `--n` draws a process each (`--jobs`, `--into DIR`), `--view`, `--strip` (`--strip-out PATH`); filmstrip `solve.png` |
| `scripts/board_png.py` | a whiteboard's ink as a PNG from the boards state file or a recording. ⚠ +lat is the viewer's LEFT, as in the site's `surfaces/board.ts`; the test pins it because every figure the quadruped's pen drew is symmetric |
| `scripts/humanoid_spike.py` | the humanoid (#507; SimNotes "The humanoid body", "The humanoid stands, reaches and grasps", "The humanoid behind Body", "The humanoid walks", "The humanoid climbs, falls and gets up" and "One walker that sees the ground", Parts.md "The humanoid body"): `--view body` the body held up on a gantry in the house, each joint through its range, then the grippers; `--view stand` its standing policy shoved, crouching and bowing, `--view reach` a hand to a counter, a high shelf, a low shelf and the floor through `Body`, its get-up loaded, `--view grasp` a drawer's handle gripped and pulled, `--view pull` a locked handle pulled at `pull`'s most, 20 N, each way (#544), `--view bar` the `pull` verb on it at its most, back, up, forward and down (#547), `--view feel` 20 N staged at a held-out hand, what its arm and its whole body read (#547), `--view demo` the chest's knob on a counter found by its tag, reached and gripped through `Body`, `--view walk` living room to kitchen and back on its own map, `--view rough` the walker that sees the ground (not served) across doorsteps, paving, a slope and blocks, `--view stairs` it up the house's flight on its own map, round on the landing and down, `--view fall` shoved over three ways, its get-up taking each fall, `--view getup` up off its back, its front and its side (`--strip SCENE` the filmstrip, `--policy NPZ` another standing policy, `--getup-policy NPZ`); `--mass` beside the T2's published model, `--torque` held in our physics and the walking, the stairs and the get-up FLOWN, `--thermal`, `--energy`, `--sensors`; `--friction [--write]` its arm joints' friction calibrated in free air beside the sheets', kept in `models/humanoid_friction.json`; `--shove` the shoves it stands through, `--envelope` the reach flown floor to over the head, `--hold` a locked handle pulled harder, `--pulls` a hand pulling each of 14 ways at a counter's height until it falls, `--bar` the force move on the locked handle each way at 10 and 20 N (`--oracle` closed on its true force: the loop's floor), `--feel` what its arm and its whole body read of 20 N staged at the grip, still and swaying (#547), `--handover [N]` the walk's hand-over at a standstill through `step` (#544), `--demo-table [N]` the demo's set-outs flown through `Body`, `--track` / `--rough [NPZ,...]` the served and any walker asked each command on a bare floor / over a doorstep, paving, a slope and blocks (one that sees the ground on its own map), `--odometry` the leg odometry against the truth, `--climb [NPZ]` flights of five at each riser up and down on the ideal scan and its own map, `--fall [NPZ]` shoved over six ways standing, limp or getting up, in five places in the house, and from each bow a reach takes (#520), `--speed [SIM_S]` the real-time multiple of one humanoid and a pair, the same walks in the house (`--profile`); filmstrip `humanoid_spike.png` |
| `scripts/attempts.py` | #493's harness: a skill tried at chests through the attempt protocol (`evaluation.attempts`) and graded by the lid's grader (`challenge/lid.py`), the humanoid's reference at its counter (`evaluation.opener`; `--body humanoid`, the one there is): `--chests A-B` (0-31), `--held-out N [--seed S]`, `--attempts N`, `--jobs`, `--into DIR` (`rows.jsonl`, a row an attempt), `--read DIR` the per-task record, `--drops` the drawn lids let fall (what the grader's lines were read off), `--view [K]` an attempt in the viewer, `--strip [K]` its filmstrip `attempts.png`, `--record PATH [--chest K]` the humanoid's attempt as the site replays it, a session's stream (#530: the fixture, its scene beside it) |
| `python -m pluggybot.session.host` | #530's session host, the process on the server a person's runs go to (`--endpoint ws://.../api/pluggyworld/sessions/host` or `$PLUGGY_SESSION_ENDPOINT`, `--record DIR` or `$PLUGGY_SESSION_RECORD`, the secret in `$PLUGGYWORLD_TOKEN`): one run at a time, each a child process (`python -m pluggybot.session.attempt`, the job on stdin) streaming to its seat's own path |
| `scripts/hands_check.py` | #525's capability check: our procedure in the hands' vocabulary for the lid, a drawer, a door's lever and a cube placed (`evaluation.hands_check`), flown at each job's set-outs and graded by its own grader, the bar (6 of 8) written first: `--job J,..`, `--set-outs A-B`, `--jobs`, `--into DIR`, `--read DIR`, `--source JOB` the procedure and its length against the cap, `--view JOB [K]` a flight in the viewer, `--strip JOB [K]` its filmstrip `hands_check.png`; JOB may be a scene (`hands_check.SCENES`): `until`, the lid's lift stopped on its hand's height (#554); `watch`, on where its watch puts the knob (stage 2) |
| `scripts/practice.py` | #493's graded practice: the mind (`--model`, the deployed by default; `$HF_TOKEN`) writes the humanoid's procedure for the lid, tries it from a fresh set-out, is shown how it ran and the world's grade, and writes again, up to `--attempts` (5) a chest, a fresh robot a chest; the hands' vocabulary (#525; #493's `--pull`, `--verb-tests` and `--path` arms retired, `run.vocabulary` saying which a row flew); `--chests`, `--held-out N` (the frozen commit's seed), `--acquire A-B` the KEPT protocol (practice first, `--until`, `--max-chests`, `--give-up-after`; `--kept FILE` a record's library), `--early-stop` once the bar is out of reach, `--notes` notes and a draft carried, `--no-measure` without the depth camera's `measure` (its own arm), `--unguarded` without a hand's move's stop condition (#554), `--unwatched` without `watch` (its stage 2), `--jobs`, `--into DIR`, `--no-picture`, `--read DIR` the per-task record and what the mind cost |
| `scripts/lidar_spike.py` | the head's LIDAR (#581 stage 1; SimNotes "The 3D LIDAR on the head"): `--hidden` what the robot's own body hides of a frame, standing and at each of the reach scene's holds; `--cost [--poses N]` ms a frame in the house, the level ring, the cloud and the band beside a D435 frame; `--probe [--variants ring,band]` #572's static probe on its model (does a scan pin the heading at each outdoor point); `--view` a walk across the street, its map from its points laid flat, the band's returns red and the level ring's blue (MUJOCO_GL unset); `--strip` its filmstrip `lidar_spike.png`. The drift, station and speed instruments fly the band with `--band` (`humanoid_drift_spike.py`, `station_spike.py`, `humanoid_spike.py --speed --band`) |
| `scripts/humanoid_drift_spike.py` | #572's stage 2 (SimNotes "Where the humanoids get lost"): one humanoid flies the loop's explore, walks the loop's walks across the street to the storeroom and back, and charges; its belief against the truth, the error taken apart as it is made into the gyro's heading, the legs' position, the believed heading's, and the matcher's and the station's corrections, over the flight and by zone, beside the odometry alone and the legs alone; `--seed luca\|rowan\|K` the IMU's and joints' draws, `--start start\|hall`, `--explore S`, `--trips N`, `--rest S`, `--idle-at X,Y [--idle S]` stood idle there first, `--learned` its gyro's offset taken off its heading as a zero-rate update that had learned it would, `--no-charge`, `--out`; `--table RUN.json ...` runs side by side and by zone. The stage's other half is `station_spike.py --lost --far [--from near,house,street] [--n N] [--explore S]`: its belief set 1-5 m and 5-20 deg wrong, 2 m out, in the house's rooms and across the street, and how each charge ended; `--view lost --lost-from P --lost-off X,Y,DEG` (or `--strip lost`) its scene, the camera over the robot |

- **The body's training stack lives in `training/`, a uv project of its own** (#377, #507; `training/README.md`): the humanoid's tasks in `humanoid_train/` -- `Pluggy-Humanoid-Stand`, `-Walk`, `-Walk-Rough`, `-Walk-Seeing`, `-Walk-Seeing-Fine`, `-Stairs`, `-Getup`, `-Getup-Head` -- registered and exported by `quad_train`'s one entry point (`python -m quad_train.train <task>`; `python -m quad_train.export ... --task`, which records the joints it drives and their gains; the quadruped's tasks are at `quadruped-final`): mjlab 1.5.x (it pins `mujoco ~=3.10.0`, the served sim's), reading the body from `models/humanoid.{xml,json}` (`python -m pluggybot.humanoid.model` writes both; mjlab caps numpy below pluggybot's, so the two never share an environment). ⚠ Running a policy loads none of torch, jax, warp, onnx or mjlab: it reaches the served sim as an `.npz` that `quad_train.export` checks against its ONNX before writing, run by `humanoid/policy.py` in numpy. ⚠ The joint DRIVERS are MuJoCo's position-mode `dcmotor` (#385), the PD and the envelope in C, commanded as the RobStrides' FOC drivers are in MIT mode (`humanoid/drivers.py`): every command carries its gains — a policy's are its own, an arm's the default gains scaled by `ARM_GAIN_SCALE`, `limp()` holds nothing — so the policies and the arms share one body. The default gains are ONE definition, `actuator.driver_gains`, which `training/` reads from `humanoid.json`. `training/pod.sh` rents a GPU to train on, or CPUs to fly a batch (`create-cpu`), on Vast.ai through its CLI, or on Runpod over the REST API with `PROVIDER=runpod` (#510; `runpodctl pod create` needs a GraphQL-writable key); a pod's id says whose it is, and a card is picked by what a million steps cost (`training/README.md`). ⚠ The `vastai` CLI exits 0 when it is refused, and `destroy` says nothing under `--raw`: read the JSON and the instance, never the exit code. ⚠ On Runpod a POST to `/v1/pods` with an EMPTY body CREATES a pod: every field has a default.
- ⚠ **A batch of flights runs on a rented pod's CPUs or under a memory cap, never bare on the dev box** (#469: six house worlds at once took it into swap and its desktop down overnight, 2026-10-06): `training/pod.sh create-cpu` → `setup-sim` (the commit, never the working tree) → `batch` with `--jobs` one under the share `setup-sim` reports (a Vast share is a hard quota) → `pull-batch`, then `delete`; locally, `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 ...`. A pool of flights gives each its own process (`maxtasksperchild=1`: a world dropped in a reused worker is held until the cycle collector runs), and results go under `~`, never `/tmp`, which a reboot wipes.
- **The humanoid is our own body from real parts, in simulation first** (#507;
  `humanoid/`, SimNotes and Parts.md "The humanoid body"): `models/humanoid.{xml,json}`
  are GENERATED (`uv run python -m pluggybot.humanoid.model`), and its bill
  (Parts.md's and `protocol/parts.json`, below) RENDERED by the catalog's
  rules; a test fails on either stale. Every part's number sits at
  its constant in `humanoid/parts.py` with its sheet -- the weaker torque and
  the heavier mass where a maker's documents disagree, a number nobody
  publishes stated with its basis -- and a fabricated part's mass is its
  drawing's volume (`model.structure`). ⚠ A POSE HELD IS A STALLED MOTOR:
  judged against `Motor.stall_torque`, never the turning rating. ⚠ Its
  proportions and the legs' ranges are Booster T2's published model's,
  credited (BSD-3-Clause, `models/booster_assets.LICENSE`); never its meshes.
  ⚠ Deployed in the house as `home_humanoid` (#559's stage 4, below), and
  otherwise a `Body` in worlds of its own.
- **The humanoid stands and reaches off its own senses** (#507 stage 2;
  `humanoid/policy.py`, `arms.py`, `reach.py`; SimNotes, "The humanoid
  stands, reaches and grasps"): its legs are a standing policy
  (`models/humanoid_stand.npz`, `Pluggy-Humanoid-Stand`) asked a POSTURE
  (the pelvis's height, the trunk's pitch), its arms an IK reach. ⚠ WHAT
  STEERS READS THE PARTS: the IMU part through its filter, every joint
  through its driver's reply (`senses.JointSense`), the grip through the
  gripper's encoder and current -- a test spies on `MjData` and fails on
  the simulator's truth. ⚠ A HAND'S TARGET IS IN THE STANCE FRAME (between
  the soles), solved on the robot's MODEL OF ITSELF (its own drawing
  compiled alone, `arms.SelfModel`), never the world's model; so the feet
  must stay planted, and training charges each sole's distance from the
  EPISODE's stance (`feet_anchor`: a speed charge let them creep, an
  anchor renewed at touchdown made a step free). ⚠ A HAND THAT CANNOT GET
  THERE STOPS (`reach.BLOCKED_M`/`BLOCKED_N`, held `BLOCKED_S`): its own
  solve out of reach, or the force it reads (below) a push on its way in --
  unstopped, an arm pushed the robot over. ⚠ IT GOES IN ONCE THE TRUNK IS
  STILL (`reach.TRUNK_STILL_RATE`; or after `POSTURE_WAIT_S`): the
  force it reads assumes the body still, and begun as the deepest bow
  stopped turning, the floor's reach read 492 N and stopped 6 cm short
  (#520). ⚠ A WITHDRAW IS OK ONCE ITS ARM IS HOME (`Reach.home`: the rest
  commanded and read by its drivers), by the way `Reach.way_home` judges
  on the same model: straight in joint space, an arm raised before its
  shoulder came down on the counter's edge (#539). ⚠ No force guard on
  it: at `BLOCKED_N` one stopped a blocked reach's press as well as a
  caught jaw, and cost the lid check 2 of its 7 (SimNotes, "The way
  home"). Training's arms are SCRIPTED,
  their gains the served arms' (`model.ARM_GAIN_SCALE`, through
  `humanoid.json`). `training/humanoid_train/g1_check.py` is rule 6's
  check: mjlab's stock G1 flown outside mjlab. ⚠ `pull` ASKS AT MOST 20 N,
  WHAT THE STANDING BODY HOLDS (`steps.PULL_MOST_N`; Ben, 2026-10-25, #544;
  SimNotes, "Holding the pull verb's 60 N at a hand"; `humanoid_spike.py
  --pulls`), and its text says pushing away holds around two thirds of it:
  a hand at a counter's height held 20-40 N across the floor for 3 s.
  Retrained against 60 N, a policy blind to the pull only stood 1.5-2 cm
  further back (a handle at the reach's edge then out of reach); one shown
  the hand's force held 60 N only on the true force -- its arm's currents
  are ~20 N off, and shown the whole body's reading (below) it fell at 20
  N every way. The served policy is stand4; a wrist force sensor is the
  part to add for more (Parts.md). ⚠ A HAND HELD TAKES NO POSTURE (#547; SimNotes,
  "The force move against a held hand"; `humanoid_spike.py --bar`): out of
  the arm's reach, a pull's lead is not the arm's end where the arm reaches
  the hand itself -- its command stays where the arm last reached (backing
  off toward the hand past its force) and, going nowhere, the pull ends
  held fast, or out of reach pushing under `FAST_SHARE` of its ask;
  postures taken for the lead put 30-100 N on a locked bar for an ask of
  10-20. ⚠ A HAND AT WORK, STANDING,
  READS ITS FORCE OFF THE WHOLE BODY (#547 stage 2; `arms.SelfModel.felt`,
  `reach.FELT_PHASES`, `Reach.standing`; SimNotes, "A hand's force off the
  whole body"; `--feel`): the robot's balance in least squares over the
  trunk's six rows, the legs' and the working arm's joints, each weighed by
  the friction it may hold still -- 19.7-20.2 N of 20 every way, where a
  straight arm read 2-5 -- ⚠ a joint on its STOP left out (`STOP_RAD`: the
  elbow's stop read 11 of 20), ⚠ the hand's TWIST an unknown about 1 N*m
  (`TWIST_SD`; as none, the jobs read 1.3-3x worse), ⚠ a balance whose rows
  will not agree reads nothing (`CLOSES`: dragged onto a sole's edge it read
  60 N the wrong way), ⚠ the arm's alone for a hand not out and while
  walking (`Reach.standing` is off until the mission says so, every step).
  It reads all the world pushes on but the soles as the hands'. ⚠ Never fed to a standing policy: it is the
  legs' torques, the policy's own commands (stand12 fell at 20 N). ⚠ THE
  LOOP SLOWS BEFORE ITS ASK (#547 stage 3; `reach.AHEAD_S`, `BACK_SPEED`;
  SimNotes, "The force loop"; `--bar --oracle`, the move closed on the
  bar's true force): closed on the hand's reading over 0.03 s carried 0.1 s
  ahead at its rise, backing off at twice its speed, a held hand backing
  off toward the hand (its joints no faster than `JOINT_RATE`: a setpoint
  ramped) -- the bar takes at most 1.2x its ask on the truth
  (1.7x before), on the robot's reading every way but up (1.4-1.6x: the
  reading lags a turning grip); ⚠ a hand GOING is only slowed by where its
  force is going (backed off on it, a free-air pull stopped on the
  reading's 4-5 N phantom); `pull.force` and held fast stay on the 0.1 s
  mean (the cube's "it pressed" reads it).
- **The humanoid is a `Body`, and walks through the Navigator** (#507
  stages 3 and 4; `humanoid/body.py`, `mission.py`, `walk.py`,
  `odometry.py`; SimNotes, "The humanoid behind Body" and "The humanoid
  walks"): `body_for` builds it by its joints, the one body there is. A
  hand's members are `Body`'s (`reach_routine`, `release_routine`,
  `grip_routine`, `gripped`, `hand_force`; ⚠ `reach_routine`
  is True only of a hand HOLDING at its goal, never on `Reach.done`, which
  a hand stopped short is, and every hand after a fall's stand-up; a fall
  ends a ramp too, and the stand-up keeps the grippers: #520). Its legs are the WALKING policy's while it moves
  (`models/humanoid_walk.npz`, `Pluggy-Humanoid-Walk`) and the standing
  policy's while it reaches, handed over at a standstill with its arms
  home. ⚠ A BLIND WALKER GIVES UP WHAT RISKS A FALL (rough4, gone): it
  stood still where asked to walk backward or sideways, on flat ground
  too, and the Navigator backs off. ⚠ ITS TERRAIN IS BOXES ALONE: the box soles blew up on MuJoCo Warp's height
  fields, a world in seven, and two runs ended in it. ⚠ ITS MAP IS LAID FROM ITS LIDAR'S LEVEL RING, 1.47 m
  up (#581): over the rover's 1.2 m walls a map of that plane alone was empty, so the house's walls are 2.4 m (stage 5). ⚠ A HEADLIGHT LIGHTS THE ROBOT'S CAMERAS (brightened, a
  tag decoded 3 cm elsewhere): a scene the robot sees by keeps the house's
  light, and a filmstrip brightens its own tiles alone. ⚠ ITS LIMBS MEET
  EACH OTHER AND ITS TRUNK (`model.LEG_GEOMS`' and `ARM_GEOMS`' pairs): the
  robot's geoms touch nothing of its own but the pairs the model names. ⚠ Its odometry tells the planted foot at 3 mm (`odometry.SWAP_M`:
  a shuffle lifts under a centimetre, and at 1 cm a short step's first
  stride counted backward); sideways it still reads up to a fifth short,
  and the Navigator never walks it sideways.
  The demo's world is `evaluation/counter.py`: #466's chest on a counter,
  seeded (`set_out(k)`), never served.

- **The humanoid climbs the house's flight on its own map, and gets up**
  (#507 stage 5; `humanoid/scan.py`, `odometry.py`, `getup.py`; SimNotes,
  "The humanoid climbs, falls and gets up" and "One walker that sees the
  ground"; `tests/test_humanoid_stairs.py`): the walker that sees the ground
  (`models/humanoid_walk_seeing.npz`, `-Walk-Seeing-Fine`, NOT served --
  walk2 is: it passes Ben's flat-floor rule on the rule's walks, but on its
  own map fell once in 27 more where walk2 never does; serving it is Ben's
  call) is shown the D435's map at its grid (`HeadScan`), laid at
  `mission.DEPTH_PERIOD` (5 Hz; at 10 Hz one robot's real time fell 2.7x ->
  1.8x). ⚠ THE HUMANOID'S CAMERA CASTS UNCULLED (`depth.UNCULLED`): cast to
  its 3 m range MuJoCo skips a plane past 2.75 m from its anchor, and the
  bare floor's camera saw no floor. ⚠ IT IS SHOWN GROUND IT COULD STEP ON: a wall's cell (one
  frame's points taller than `WALL_SPAN_M`) and what joins its ground only
  through a rise taller than `STEP_UP_M` (a counter's top, the deck) read as
  the nearest ground -- training had neither, and shown the flight's walls
  it fell over backwards. ⚠ ITS HEIGHT IS NEVER LAID OFF THE MAP: the planted
  foot on a flight is the one nearer the mapped ground under it (the height
  under 30 % of its footprint, never its highest point), the height moves
  only as the planted sole does, and frames register it on flat cells; laid
  off the ground under a landing sole, it jumped a riser and the walker
  stumbled; under a sole is ground (a half wall's top read 1.4-1.6 m), and a
  sole tipped over a tread's edge is followed by its corner on the ground.
  ⚠ A PLANTED SOLE'S FOOTPRINT GOES INTO THE MAP AT THE LEGS' HEIGHT, the
  camera's cells included: without footprints it climbed no bare flight,
  kept off the camera's cells 6 of 25 -- and a footprint a few cm off a
  flat floor is a step, the flat-floor falls' cause. Coming DOWN a flight
  is the open half: the belief lags 0.7-1.9 m at the foot (it runs down
  faster than asked; its odometry follows a sole in the air, skids and
  flights; the hall gives the matcher nothing along it). A FALL is the get-up policy's (`models/humanoid_getup.npz`,
  `-Getup-Head`, every joint): tilt `FALL_TILT_RAD` past the posture the
  legs hold or the pelvis under `COLLAPSED_M` over the LOWER sole, handed
  back once stood still; ⚠ A TIP IS READ OFF THE POSTURE, NEVER OFF
  UPRIGHT (#520; `getup.tipped`, `Reach.held_pitch`, exactly upright at the
  stand): the IMU is on a pelvis with no waist, and off upright the reach's
  deepest bows were falls; a forward fall from a bow is the collapse's to
  take (lying on its face is 27 deg past the deepest bow). ⚠ The head is
  charged on its own in training, or it struck at 5.5-6.9 m/s; the
  mission waits out a get-up before a walk takes the legs or the standing
  policy takes them back, and on standing forgets its ground map and
  starts its height again. `stuck_after_s` is
  `GETUP_BUDGET_S` (20); falls into a wall or off the landing still end
  stuck.
- **The humanoid pair is served in the house, and deployed** (#559 stages
  1 and 4; `humanoid/world.py`, `lifecycle.world_config`,
  `tests/test_humanoid_served.py`; SimNotes, "The humanoid pair served";
  its period in Observatory.md; the site draws it off `home_humanoid_pair`'s
  scene and recording, `protocol/`):
  `home_humanoid` is the house with its own fixtures (below, "Generated
  worlds"), its jobs the hands' on the storeroom's counter (below), the
  pair colliding (`humanoid.world.PAIR_BITS`; the pair's bullet below). ⚠
  IT CHARGES AT ITS STATION (below); a world with no charger
  (`HubLifecycle.charger` False) ends a charge at once (`NO_CHARGER`) and
  sends it to none. ⚠ Its restart keeps parity (`HumanoidMission.kept_state`:
  its drivers' steps and targets, the gains they set in the MODEL, its
  senses' noise, its reach, its reckoning's planted foot, the walls its
  layer is kept off), never with a hand out (`Body.working`); a test saving
  on a step every controller's period divides read a fresh reach as kept.
  ⚠ #558's two levers: a plan held `REPLAN_HOLD_S` (the Navigator's; it
  replanned every step its waypoints ran out short of the goal) and the
  walls grown in the camera's window -- ⚠ and the way past a robot asked
  aside is planned AFRESH (`_route_past` -> `_plan_near`): answered by the
  memo, nobody was asked; so is a plan the drive drops because the world
  moved (a robot fell, rested or got up; a hold). ⚠ IT RESTS STANDING
  (`humanoid/way.py`: no motion command for `REST_S`, 8.6 s, no walk of its
  own, no hand out, on its feet) and is asked aside by #415's ONE rule and
  walk (`HumanoidWay`), beneath whatever routine holds it, its posture the stand's before
  the walker takes its legs, and ended by a routine that takes them; its
  front stop backs it off within the step, never ends it (ended, 3 doors of
  6 never began). Without it two humanoids jammed a door for 500 s of a
  minded day. ⚠ NEVER WHILE ITS PLUG IS IN (`plugged_in`, the port's
  catch, its refusal "charging", stage 4): asked aside, it walked off with
  the plug in its port and pulled it out -- read off the plug, never
  `Body.docked`, which a restart forgets with the plug still in. ⚠ ITS
  WALKS KEEP OFF NO PLATE (stage 4; its shock plate is off): kept off the
  pads it knew, one standing on a pad as it found the sign planned no walk
  off it. ⚠ THE DEATH SEAM READS A TOPPLE OFF THE POSTURE THE LEGS
  HOLD (`Body.held_pitch`, stage 3): a reach's deepest bow is 63 deg of
  pelvis tilt, past `TOPPLE_TILT_RAD`'s 60.
- **The humanoid charges by a cable it plugs into itself** (#559 stage 2;
  `activity/station.py`, `humanoid/charge.py`, `tests/test_station.py`;
  SimNotes, "The humanoid's charging station"): a station on the living
  room's wall (where the quadrupeds' dock lay), a plug on a tool balancer that draws it
  home into a cup, a magnetic RoPD port on the torso's front (Ben,
  2026-10-30: no trunk point is in a head camera's view). The station is an
  activity of two CATCHES, each a weld taken where the plug is and drawn to
  its seat: the port (off past 20 N, the sheets' minimum) and the cup (off
  only to a pull straight out). ⚠ A CATCH TAKES ONLY A PLUG NO GRIPPER PAD
  HOLDS: held, the port fought the hand past 20 N again and again, and the
  cup never took a carried plug back. ⚠ THE FUNNEL ROUND THE PORT IS THE
  CATCH'S (70 mm, 22 mm across), as the cup is: the hand knows where it holds
  a plug to 1-2 cm. ⚠ The port re-arms only beyond the funnel's reach; the
  plug's spin is damped apart from its swing (one 0.5 blew it up). The
  routine is OURS, off its senses (the station off its two tags 58-59, the
  port off its model of itself): the open jaws slid along the plug's axis,
  never in from its side; pushed to the body and LET GO; a miss fetched
  again from the cup a step aside; the seat never more than 3 cm nearer the
  station (the walker barely backs up: an undock turns away). ⚠ THE PLUG-IN
  TAKES THE SEAT AS THE WALKER LEAVES IT (`charge.GRIPS`, stage 3): the
  fingers swung 20 deg toward the plug's back, the grip 1 cm further back
  (straight across, the standing arm reached the parked plug only from 3 cm
  behind the seat), straight across where that hand's way in is barred; the
  port read once the hand holds the plug. ⚠ A SQUEEZE ON A PLUG A CATCH
  HOLDS BEGINS AT 5 N: shut at once, the first pad pushed a seated plug off
  its catch. ⚠ LOST, IT STEPS IN BY ITS LIDAR, NEVER ITS BELIEF
  (`HumanoidBody.lost_routine`): to the wall 0.8 m ahead, looks round, steps
  along the wall; a look that fits the station is a fix (`_fix_at`, the
  anchor's too). ⚠ A SQUEEZE HOLDS THE PLUG ONLY AT THE JACK'S WIDTH
  (`PLUG_GAP_TOL_M`): wider is the station's boss, and the hand draws back
  along the plug and grips again. ⚠ THE UNPLUG LETS GO AT THE CUP'S MOUTH
  (`CUP_SHORT_M`): carried to the seat, the plug was driven into the boss
  and the robot fell. Measured: plugged in from 140 of 140 seats over the
  seat check (110 before), unplugged 140 of them; 15 of 15 docked and
  unplugged from round the standoff; 15 of 16 docked from across the house
  (10 before); 16 of 16 with their belief set 0.4-1 m wrong (SimNotes),
  and ⚠ 5 of 60 set 1-5 m and 5-20 deg wrong: ITS LOST SEARCH REACHES ABOUT
  A METRE -- begun with the belief within 1.1 m it found the station 4 of 4
  times, from 1.5 m or more 0 of 50 -- and a belief comes home past the
  matcher's reach from in and beyond the middle street, the house's map then
  refusing it every scan (#572 stage 2; SimNotes, "Where the humanoids get
  lost");
  345 W net; the reserve 18 Wh (17.5 before the Mid-360S's 6.2 W, #581).
- **The humanoid's LIDAR is a Livox Mid-360S, upside down over its head**
  (#581 stage 1; `perception/lidar.py`, `HumanoidSpec.lidar_rise`,
  `tests/test_lidar.py`; Parts.md, "The LIDAR"; SimNotes, "The 3D LIDAR on
  the head"): a frame is 30 rings 2 deg apart, -52 to +6, 360 bearings each,
  ONE `mj_multiRay`, with its sheet's range and angle noise and its 0.1 m
  blind zone (no reading, as a self-hit is none); its robot's own returns
  dropped, another robot's split out (#316), its noise its robot's own
  (`imu.seed_for`). ⚠ ITS LEVEL RING IS THE SCAN THE MAP IS LAID FROM, as
  the C1's plane was. Its points LAID FLAT -- per bearing the nearest
  surface 0.15-2.2 m over the floor, through the IMU's tilt (`Lidar.band`)
  -- see the fences, the tables and the couch, and `HumanoidMission.band` is
  a measurement's switch, never a deployment's, until #581's stage 2 serves
  it. ⚠ A SURFACE IS READ BY ITS RETURNS' MEDIAN, never the nearest return:
  a dozen rings strike a wall, and the least of their ranges read every wall
  6 cm short. ⚠ ITS RAYS LEAVE 10 cm OVER THE HEAD'S TOP, 5 mm under the
  unit (cast from inside it, every ray hit it): the steepest, -52 deg,
  clears the head ahead and either side; at the C1's 4.7 cm the head hid
  every ray under 31-35 deg. ⚠ A FRAME COSTS 14-17 ms on the deploy box (the
  level ring 0.5-0.7, a D435 frame 11): with the band at 10 Hz the pair ran
  at about real time (0.97-1.15x, `humanoid_spike.py --speed --band`):
  stage 2 casts 15 rings at 10 Hz (Ben's call 3, 2026-11-01; 1.17-1.42x).
  ⚠ WITH THE BAND, A DOCK'S MATCHER MOVED THE BELIEF 3.4 TIMES AS OFTEN at the
  station's near returns, and the seat and the plug-in, placed through it,
  failed 2-5 docks in 47 over the band's three versions (the level ring 1,
  staging none): stage 2's to settle. ⚠ NO CUTOFF ON THE CAST: MuJoCo culls
  a plane by its centre.
- **A humanoid is told its own body and world** (#559 stages 3 and 5;
  `Menu.kinds`, `hands_procedure_rule`, `steps.served_hand_facts`;
  `tests/test_humanoid_words.py`; Overseer.md §8b): ⚠ ITS WORDS ARE THE ONLY
  WORDS -- the rules, the reward table's rows and the constitutions are what
  the served pair reads, byte for byte (stage 5 took out the quadruped's
  text and the swaps that made the humanoid's from it), and a test reads
  every prompt for the quadruped's world. Its reward table is its world's rows (the charge's, a procedure's,
  a ticket's and its jobs'), `rate` only where there are boards; its
  procedures are its practice's hands, every switch on, its walks over its
  map (`drive_to`, `face`), `find` where its world has places (the
  readings `find.x`/`find.y`/`find.facing`, in its map), no bumper, and its
  rule carries the practice's language and body over its world's facts byte
  for byte (less "Both grippers start shut").
- **The humanoids' first jobs are the hands' four on the storeroom's
  counter** (#559 stage 3; `home/storeroom.py`, `humanoid/places.py`,
  `tasks.HAND_JOBS`, `HubLifecycle._watch_job`/`_grade_hands_routine`/
  `_set_out_counter`, `challenge/solutions.py`; `tests/test_hand_jobs.py`,
  `tests/test_humanoid_places.py`; Challenges.md §11, SimNotes, "The
  humanoids' first jobs"): a counter against the research center's
  storeroom's east wall, put in at load (`storeroom.attach`), the chest,
  drawers, cupboard and cube each square where its check's set-outs' middle
  had it, a row of 120 mm tags (60-63) on the wall over them. `open_chest`,
  `open_drawer`, `open_door`, `place_cube`: procedure jobs (below, "A
  procedure job has no errand behind it"), each its prop's target and its
  own row (paid, `shown: false`: shown where a world names it,
  `Menu.kinds`), graded by its check's evaluator. ⚠ WATCHED FROM THE CLAIM to the hold after `done`: a fall on
  the way fails it; its `minutes` count from the claim; its readings ride a
  restart (`jobWatches`). ⚠ EACH OFFER DRAWS ITS PROP'S HIDDEN DYNAMICS
  (`TaskKind.draws`, the board's seq) into the compiled world and sets the
  prop out, its activity renewed in place (`renew`), put back after a
  restart (`restore_counter`, in `begin`, before the activities' kept
  state); ⚠ NO OFFER LANDS ON A PROP A ROBOT IS TOUCHING (the producer's
  `busy`): set out from under a hand, a drawer left out was paid for
  nothing. ⚠ Its `find` (`HumanoidPlaces.find_routine`, its world's places
  `world_places`; below, "Places, not coordinates") looks with the head's
  NAVIGATION camera, placed through its odometry's model of itself; a find
  stops at its tag's first sight, so its facing may be only where it was
  seen from: the references find the row's neighbour first (one tag seen
  from the lobby left a robot 28 deg off, and its arm's way home toppled
  it), then turn once off the handle's tag to each prop's `STANCE`, a
  little left of square (squared, the way home pushed the drawer back in).
  OFFERED: the drawer, the door and the chest, 8, 8 and 7 of 8 over
  `solve.py --from tour` (the bar 6), 16 minutes each (3x the slowest
  pass, 313 s); ⚠ NOT the cube (0 of 8, set down off its mat). Each pays
  90, Ben's call.
- **The humanoid opens the chest at a counter** (#493 stages 2-3;
  `evaluation/opener.py`, `evaluation/counter.py`; SimNotes, "The humanoid
  opens the chest"; `tests/test_hands.py`): the counter is 0.72 m (Ben).
  The reference is #466's probe's plan, on the hands -- the knob off its tag,
  the box off the D435 at 848 x 480 (`Body.depth_cloud`), the lid's arc
  carried as ONE path (`Body.hand_follows`, `move_hand_routine(path=,
  turns=)`), the knob gripped in the posture that follows it
  (`reach_routine(path=)`). ⚠ THE DROP HANDLE'S ARM LIES ON THE LID'S FRONT
  EDGE 57 DEG UP: past `opener.FREE_RAD` (50 deg) the hand turns as the lid
  does. ⚠ The live robot leans back as it lifts: a path is planned on a
  statue `PATH_MARGIN_M` (3.5 cm) ahead, and for a hand that is out on the
  robot as it stands (its measured arm, `LIVE_MARGIN_M`); a carry keeps the
  turn it was ASKED. ⚠ ONE POSTURE A PATH: switching partway jerked lids to
  81 deg and dropped them at 2.3 rad/s. It opens 23 of 32; its slowest
  pass at 0f98e24 (22.5 s) set the humanoid task's limit at three times
  it (`attempts.LIMIT_S`). The humanoid looks lower by bowing
  (`Body.gaze_routine`).
- **The drop-handle chest is demo 1's mechanism, and the humanoids' house's**
  (`activity/chest.py`; since #559 on the storeroom's counter, through
  `home.storeroom` alone): #469's box as an activity, its hidden parameters
  drawn seeded (`draw(k)` is set-out k and a 1-4 N catch, kept only shut by
  `CLOSING_MARGIN_NM`: a drop handle only pulls). ⚠ THE TRUTH IS KEPT AS
  `Task.secret` IS, never in a flag. Its flags (`lid` live with hysteresis,
  `opened` latched, `caught`) come off the hinge's jointpos sensor, and
  `sense` sets the magnet's pull on the seam (MuJoCo has no element for
  it); its knob wears tag 53 (`perception.tags.CHEST_TAG_IDS`). The
  imagination that modelled it (#466: the scene language, the probe from
  the quadruped's senses, the author and the fitter) is at
  `quadruped-final`, and #563 brings one back.
- **The hands in a procedure, and graded practice** (#493;
  `procedure/steps.py` `HANDS_VERBS` and `language(facts)`,
  `axes.HAND_SENSORS` / `PULL_SENSORS` / `STEP_SENSORS` /
  `BODY_AXES["humanoid"]`; `mind/practice.py`, `evaluation/practice.py`,
  `scripts/practice.py`): the vocabulary as stated priors -- `spot`,
  `measure`, `reach`, `via`, `move_hand`, `pull`, `close_hand`,
  `open_hand`, `withdraw`, `step` -- ⚠ NO VERB
  KNOWS THE CHEST (Ben's call 2; a test reads every doc and default). A
  hands world calls exactly the verbs its facts name (`language`), and a
  world that names none `VERBS`; ⚠ no sensor reads a humanoid joint off `qpos`. ⚠
  THE ARMS' FRICTION IS CALIBRATED ONCE AND KEPT (`humanoid/friction.py`,
  `models/humanoid_friction.json`, `humanoid_spike.py --friction --write`;
  a test fails on a table off the motors' sheets), taken off the hand's
  force where a joint turns. Practice: a
  mind writes a procedure, the humanoid tries it from a fresh set-out, the
  world grades it, and the mind is shown how it ran and the grade -- ⚠
  never the grade's secrets, and its prompt prescribes nothing (a test
  reads it); the library admits a procedure only on a sealed passing
  verdict (`Library.admit`, `attempts.attempt(sealed=)`). Two protocols
  (Evaluation.md §4): FRESH, a fresh robot a chest; KEPT (`--acquire`,
  Ben), one robot practicing the public chests in turn carrying only what
  passed (`evaluation.practice.keep`, never a conversation: ⚠ under each
  name the procedure that passed the MOST chests, sources compared as
  programs, never the newest pass alone, #535), then a frozen
  copy a held-out chest, ⚠ stopped once the bar is out of reach
  (`out_of_reach`); `--notes` carries the robot's own notes and a draft
  too (`Practice.carry`, pass or fail). ⚠ FEEDBACK IS THE BODY'S AND THE
  LANGUAGE'S, NEVER THE CHEST'S: every hand verb says what the hand FELT
  (`steps._felt_routine`: its grip's place, force and hold off `Body` members,
  driven as `yield from` would), a dotted sensor is refused naming its
  `read`, a reach out of reach says whether the other hand could
  (`Body.hand_reaches`, its own plan, nothing moved). ⚠ `measure` (the
  depth camera's flat top; `--no-measure` an arm without it) is feedback as a person's eyes
  give it, never a shortcut past the strategy (Ben, 2026-10-19: demo 1 is
  TRY AND REACT): its own arm, never pooled, named in `FLATTERS`. ⚠ An answer may take `MAX_TOKENS` 128,000, told
  `AIM_TOKENS`: at 64,000, 37 of 537 ran out still thinking; one cut off
  (`stop_reason` "max_tokens", `stream_fetch`'s `finish_reason`) is asked
  again saying so. Practice chests may take more tries than held-out ones
  (`--practice-attempts`; Ben, 2026-10-19: within a chest the whole
  conversation is the lesson). ⚠ A CARRIED HAND MAY TURN: `move_hand`
  takes `fx..jz` as `reach` does (the reference's first lesson, the hand
  turning with the lid, had no verb). ⚠ A HAND'S PATH IS ONE MOTION
  (`via` then `move_hand`): separate moves each took a posture of their
  own and carried the reference's own arc on 2 chests of 8. The
  vocabulary is #525's (below). ⚠ THE DOOR IS PRACTICED WITH NO NEW HELP
  (#535; `practice.py --job door`): its own job text and time limit
  (`practice.JOBS`, `attempts.JOB_LIMIT_S`) and the practice's texts naming
  its cupboard (`Practice.thing`), nothing else -- a test pins the lid's
  texts byte for byte.
- **The hands' vocabulary is ONE, designed from how hands move** (#525;
  docs/Hands.md; `procedure/steps.py` `HANDS_VERBS`, `hand_facts`;
  `humanoid/reach.py` `Reach.pull`; `tests/test_force_move.py`): two kinds
  of HAND move, POSITION (`reach`; `move_hand`, through its `via` points as
  ONE motion from one posture; `withdraw`) and FORCE (`pull`), whose loop
  closes on what the hand FEELS inside the verb -- admittance along its
  way, `speed * clip(2 (force - ahead) / force, -1, 1)`, doubled backing
  off, `ahead` the reading 0.1 s on at its rise (#547) -- ⚠ never on the
  arm's spring (its model put 15 N on a 5 N limit); the hand giving across
  its way (`give` 1) or holding its line and turn (`give` 0: hybrid); its
  turn giving to what it holds ⚠ BUT NEVER ABOUT THE JAWS' LINE (a lid's
  weight turned a freely giving wrist 70 deg down and out of reach on 4
  chests of 8); a grip it began with lost ending it where the hand is;
  going on where the arm misses, ANOTHER posture from the arm as it is,
  never one it has left (`posture_for(onward=, past=)`). `close_hand(force)`
  takes the Dex1-1's 5-120 N, `held` against half of it ⚠ ON THE
  SQUEEZE'S RUNNING MEAN (`arms.SQUEEZE_S`, sampled every physics step:
  one reading's 5.1 N of noise misread a 15 N grip 7 % of the time): ⚠ at
  120 N a lever could not turn in the jaws. The FEET's move is `step`
  (`HumanoidBody.step_routine`, `tests/test_step.py`): hands home and
  empty, a turn on the spot and a walk on the walking policy, its report
  where the stance went in the frame before (`step.*`) -- ⚠ measured only
  once the walker has the legs (out of a bow the pelvis came 16 cm forward
  on its own), an oblique step the walker would skew walked as two, a
  turn's 2-6 cm of shuffle uncounted (the odometry is worse over it); ⚠ a
  humanoid with a hand out keeps its posture to look (a bow carried the
  hand 43 cm up). A verb test and the maths are in every hands
  world (`hand_facts`), the served humanoids' included (`served_hand_facts`,
  #559). ⚠ A HAND'S MOVE MAY STOP ON A
  READING (#554; `steps.GUARDED_VERBS`, behind `WorldFacts.guarded`, on in
  `hand_facts`, `--unguarded` its arm without; `tests/test_guard.py`):
  `reach`, `move_hand` and `pull` take `until="<sensor>"` and `above` or
  `below`, the sensor read from the move's first command and no oftener
  than its part reads it (`Sensor.every_s`, rounded UP to physics steps; a
  verb's report has none and is refused; a hand's place as its loop last
  read it, `Body.hand_sensed`, so a condition never met moves the flight not
  at all), the first reading past stopping the hand where it is
  (`Body.stop_hand`, `Reach.halt`: a carry or a pull where the hand IS, a
  reach where its arm is commanded -- ⚠ and its body where it had got to:
  held at its joints alone, the hand rode its plan's crouch and the next
  verb stepped a joint 1.4 rad -- out; none during a get-up), the verb ok
  and `<verb>.stopped` 1; ⚠ nothing it reads is the simulator's truth (a
  test spies on `MjData`), and no condition knows a job. ⚠ A TAG MAY BE
  WATCHED WHILE THE BODY MOVES (#554's stage 2; `watch(tag)`,
  `perception.watch.TagWatch`, `Body.watch`/`sighting`; on in `hand_facts`,
  `--unwatched` its arm without; `tests/test_watch.py`): `spot`'s decode a
  picture every 34 steps (15 a second, rounded UP), each the robot's 0.1 s
  after it was taken, `watch.*` the newest picture that showed it, `seen`
  and `age`; ⚠ PLACED THROUGH THE MODEL OF ITSELF AS ITS ARM'S LOOP LAST
  POSED IT (posed afresh, every picture drew on the encoders' shared noise
  and moved the flight), nothing while the get-up has the body or before a
  stand-up's fresh reach has posed it; it ends with the procedure; a stop
  condition reading it whose sensor never had a reading says why. ⚠ The
  head camera is fixed: the knob's tag leaves its view past 50 deg of lid.
  ⚠ THE CAPABILITY CHECK
  (`evaluation/hands_check.py`, `scripts/hands_check.py`,
  `tests/test_hands_check.py`): our procedure for the lid, a drawer, a
  door's lever and a cube placed (`challenge/drawer.py`, `door.py`,
  `place.py`, `shown: false`, paying 0), 8 set-outs each
  (`evaluation/counter.py`), the bar 6 written first; ⚠ a set-out draws
  off a stream of its own, never its prop's hidden draw's (`_drawn`: off
  one seed, the furthest cube was always the heaviest); ⚠ a procedure there
  is a solution, and nothing under `mind/` may reach one (a test).

### The mind (`mind/`; `docs/Overseer.md` is the design)

- ⚠ **A mind's powers are flags `build()` sets, narrowed by what the world
  has** (`Menu.procedures`, `Menu.wiki`, `Menu.tickets`, `Menu.look`,
  `Menu.lab`, `Overseer._acts()`; #427), never `Menu.for_world`'s, so a
  parse against a bare menu DROPS their fields. No rule hands the agent an
  answer: most PRESCRIBE NOTHING about using their power (a test reads
  each), and a worked example (procedures, the event map) may not show
  charge, a battery threshold or the rack. The procedure rule's example compiles on its world
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
  `tests/test_placeholders.py`): every field is required, and the model writes
  `n`, `none`, `:`, `test` or a field's name where `""` was meant -- its own,
  and in a field of its own any the form coined (`COINED`: `drop_goal`,
  `unpin`), never an everyday one (`done`: "Done!" may be meant); an object
  is judged by its own names alone (a finding is named for what it
  measures). `validate` reads one as `""` where it can stand
  (`PLACEHOLDER_TEXT`, `PLACEHOLDER_OBJECTS`: an object whose content is one
  goes whole, a label is blanked); with `FILLED_FIELDS` (3) or more (one word
  in three fields counts, `keep`, never a topic), `UNSHOWN_PAPERWORK` goes too
  (the heart, whom for, the gift, the rating, the guess, `done`); a decline of
  the job the same answer takes is dropped. ⚠ THE ACTION, ITS PARAMETERS, THE
  EVENT MAP, THE STANDING ORDER AND TEXT THAT IS NO PLACEHOLDER STAND
  (measured: nearly all of it was meant). ⚠ A number, two letters and a
  non-ASCII character are never placeholders; `answer`, `cites`, a visitor's
  `reply`, a unit, a note's topic and title and a procedure's name are never
  judged. ⚠ A `procedure:new` beside a placeholder define runs nothing and its
  `undefine` waits (#264's rule), never a garbled answer. ⚠ A QUOTE
  (`PLACEHOLDER_QUOTES`) takes out a line it is only part of only with three
  letters and no placeholder (`thoughts._loose`: `,` took out Rowan's goal); a
  placeholder quote stands outside a filled answer and takes out only a line
  that is exactly it, so an old `n` goal can still go, and a coined one
  stands too. ONE History line and a `left_out` event
  (`LEFT_OUT_WHYS`); the decision carries `leftOut`. Every power is judged,
  unshown or `PLACEHOLDER_KEPT`, and a test fails on one none of them names.
  The prompt does not change. ⚠ IT SEES NO CHOICE: outside a filled answer a
  heart, a gift or a rating is acted on as meant, and on #462's closing
  reading most were not (Evaluation.md §5).
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
  only the IMPOSSIBLE is filtered (`order_runnable`: an order with nothing
  to act on, never the unaffordable). ⚠ Three outcomes, never summed — the order ran / no
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
    consult (`rules_left_out`) whether or not any rule came back (#475: a
    world carried on brings back `minded`, so the bootstrap does not ask). A true death is the one reset
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
    the line). ⚠ A cross-event token is REFUSED, not dropped. ⚠ A broad row
    above a narrow one starves it — not prevented, visible in
    `score.failureKinds`;
  - ⚠ **A LIST GOES INTO THE LIST, AND A RULE LEAVES ONLY BY `remove`**
    (#475; `EventMap.edit`): what an answer sent replaced the map, and one
    or two rules where `[]` was meant took the `ask` rows with them (14
    `unminded` deaths in a day). A rule is named by its TRIGGER, `(event,
    kind, value)` (`Row.trigger`) — ⚠ never `(event, kind)` alone: two
    thresholds on one event are two rules (29 of 427 live edits). A rule
    new to the map is added at the end -- just ahead of the first rule on
    its event that takes its `kind`, where it has one (`_covers`). ⚠ ONE
    SENT WITH A TRIGGER THE MAP HAS AND ANOTHER ACTION IS HELD (not applied,
    said) unless the same answer `remove`s that trigger, and then it takes
    its place (Ben, 2026-10-08: in place, a slip took the only `ask` in 3 of
    the issue's 14 deaths; `DEATHS_475` replays all 14). ⚠ `events.REMOVE` is an ACTION of
    an answer's rule, never a field and never a row of the map (MEASURED:
    a field was slipped like the list, 6 rules out in 40 answers; the
    action, none in 59) — the ONE way out, removals first, the last `ask`
    included (no rail). The LAST rule sent for a trigger counts; the
    standing order rides with a `remove` of its own trigger, so it replaces
    the catch-all as it always did, after the list, unless the list names its
    rule (then the list's word stands, a `remove` too), and on a full list is
    neither in force nor said; past `MAX_ROWS` any other rule is left out
    and said, never a rule cut to fit, and ⚠ the prompt TELLS the cap (a
    test reads the constant); `EventClock` forgets a row the map no longer
    holds, so one sent again is new; a list emptied by `remove` comes
    back empty after a restart (`restore_map`). EVERY EDIT IS ONE History
    line, as a fact (`Edit.said`, `HubLifecycle._map_edit`): counts, rules,
    the list as it stands; a resend says nothing;
  - ⚠ **no worked example in `EVENT_MAP_RULE` may use `charge`, a battery
    threshold or the rack** (a test fails on a `->` line ending in `charge`):
    an example hands the agent the answer `score` measures. The rules
    (`RULES`, `APPETITE_RULE`) stay: statements about the WORLD, not
    demonstrations of the ANSWER;
  - the standing order migrates into a `decision_failed` row IN PLACE (the
    last rule of the answer's edit), is honoured synchronously, and queues
    no event (doing both ran it twice);
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
  is what the served humanoids read, byte for byte — their fixture
  recording carries it (`tests/test_telemetry.py`) — and every file opens
  with the one body paragraph, the humanoid's, written in since #559's stage
  4; `tests/test_constitution.py` reads EVERY library file (no number, `%` or
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
  with its line, every reason at once, before a step runs. ⚠ A REFUSAL INSIDE
  A CALL NAMES THE CALL ONCE, as History names a failed step (`line 3, wait:
  …`; #463): `check_step`'s reasons never name it, their callers do. ⚠ A VERB
  MAY BE AN `if`'S TEST ONLY WHERE THE WORLD'S FACTS SAY SO
  (`WorldFacts.verb_tests`, #493's practice: `if not move_hand(...)`, its
  failing read, never the procedure's end; everywhere else a failed verb
  still ends it), and `sin`/`cos`/`atan2`/`sqrt` likewise
  (`WorldFacts.math`): the served humanoids' facts turn both on
  (`served_hand_facts`), and their rule says so. ⚠ THE MOTOR LEVEL
  IS `move(axis, target)` AND `read(sensor)` over REGISTRIES (`axes.AXES`,
  `axes.SENSORS`): a new tool or body adds its entries there and the language
  does not change. Budgets (`budget(steps=, seconds=)`) are capped by code
  (`MAX_STEPS` 200, `MAX_BUDGET_S` 1800) and checked at every verb; a computed
  argument by the same `check_arg`. The library (`MAX_PROCEDURES` 8; `define
  {name, source}` / `undefine`) survives a restart and is recompiled against
  today's world; a REPLACEMENT IS ONE ANSWER (`_define` removes before it
  adds). ⚠ Invoked as `procedure:<name>` (the action, a standing order or a
  map row; `order_runnable` reads `state["procedures"]`); `procedure:new`
  (`PROCEDURE_NEW`) runs what the SAME answer defines (the enum is built
  before the answer) and never as an order or a row. ⚠ EVERY RUN LEAVES ONE
  HISTORY LINE (`lifecycle.procedure_outcome`). ⚠ `drive_to(x, y,
  patience=)` (#381): 60 s unsaid, `MAX_PATIENCE_S` 600, never past the
  run's own budget (`run_verb(until=)`); an argument with a `default` may be
  left out. ⚠ A PROCEDURE'S LOCALS ARE ITS READOUT (#227:
  `lang.run_procedure_routine` returns them, the `procedure` event carries
  them, one History line of `LOCALS_SHOWN`). ⚠ The procedure rule's example
  (`hands_procedure_rule`) may not show charge, a battery threshold or the
  rack. ⚠ The site's Procedures section is built off the `procedure`
  event (`library`, `failedLine`; `failedAt` counts verb calls, not lines).
- **The allowance** (`mind/spend.py`, `mind/mode.py`, issue #37; Overseer.md
  §8): the model is SHOWN what its thinking cost and has one boolean
  (`escalate`); every gate is code — `$PLUGGY_WEEKLY_USD`, a ten-minute
  interval and a 10 % share — and the answer comes from `$PLUGGY_ESCALATE_TO`
  (off this period). ⚠ Billed is billed (metered BEFORE the parse). ⚠ **It
  covers the ROUTINE mind** (#225): a spent purse refuses the next call as
  `fallback:allowance` (policy class). The deployed purse is 9.30 a week (=
  $40 a month, Ben's cap), ⚠ PINNED IN THE WEBSITE'S `compose.yaml`
  (rooftop-media-2026 #404): unset, the image's own default is $10, and every
  run from 2026-09-17 until that pin flew at $10 (Observatory.md). ⚠ Points
  buy ACCESS, never money. Three operator
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
    measured row is. ⚠ Every row is measured FROM THE STATION, the place a
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
  or mid-move (a humanoid getting up is a policy part-way, one with a hand
  out is `Body.working`, one stepping aside is a walk no routine holds;
  `Keeper.busy`), NEVER on a crash. ⚠ ANOTHER BODY IN THE SAME HOUSE CARRIES ON (#559,
  `lifecycle.HOUSES`, `continuation.same_house`): a `home_quad` save in
  `home_humanoid` keeps the clock, the deaths and the jobs as a changed
  world does, never the bodies or maps, and the pack at the SHARE it was
  saved at (`capacityWh`; a save without it is on the quadruped's served
  pack; the same body on another pack keeps its Wh), History saying
  `BODY_CHANGED`; a held job the new world cannot do is back on offer
  (`_resume_jobs`). Three refusals, said in History: a changed
  GEOMETRY (`fingerprint`) keeps the clock, packs, deaths and jobs but not the
  bodies or maps, and so does a save of another `MAP_EPOCH` (bumped when the
  kept maps are found laid wrong, #425 and #476); a save restored `MAX_RESUMES` (3)
  times without a new one is not trusted. ⚠ The errand in flight ends; its
  job does not (`_resume_jobs`): a procedure job stays claimed, its watch
  carried on (`jobWatches`), and one this world cannot do goes back on
  offer. The hourly ceiling is still
  rooftop's `compose.yaml`; lifting it waits on #349.
- **An admin can reach into world state, and every reach-in is recorded**
  (issue #119; Evaluation.md §5, `protocol/README.md`): `set_battery` (a
  `frac` OR a `wh`), `set_points` (an ABSOLUTE balance, never a delta) and
  `reset_robot` are inbound kinds, admin-only AT THE WEBSITE, code-handled
  on the physics thread, never shown to the overseer (`reset_tool` went with
  the quadruped's tools, #559). FOUR traces each (`life.interventions`, an
  `intervention` event, narration, History). ⚠ A `reset` of a DEAD robot is a
  rescue, not an intervention. ⚠ `set_points` breaks `earned - consumed -
  spent == balance` ON PURPOSE (`Ledger.intervene`, `Ledger.intervened`). ⚠
  `set_battery` does not revive a dead robot.
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
  boundary; the framing and the fixed menu are. ⚠ It is ONE CHAT PER ROBOT
  (#485; threads since rooftop #125): every message carries `earlier`, the
  chat's newest `MAX_EARLIER` (6) LINES, the website's state; the robot's
  own lines are a SHAPE (`robot` on the wire, `you` to the model), never a
  name, because a username can be `you`; an older site's thread exchanges
  are read as their two lines. `sender` is stated by the CALLER of
  `Inbox.offer`, never read off the wire. Each line is filed in History
  under the topic `chat` by its role (a peer's `tell` under none), and
  `read chat` widens the window. ⚠ The robot may `say` something unasked:
  a `chat` message, cut out loud as a reply is and RATIONED at `SAY_EVERY_S`
  (1200, a visitor's three an hour) off the STORE's clock
  (`ThoughtFiles.last_said`); while it holds the state shows
  `sayAgainInS` (`say_again_in`, the gate's own reading), and a refusal is
  a SHORT History line, the attempt quoted to `QUOTED_CHARS` (the deployed
  model sent a `say` on 2 of 4 turns with the wait in front of it); a
  placeholder `say` is nothing said. ⚠ TWO LENGTHS: a visitor sends
  `text.MAX_VISITOR_CHARS` (500, #474), and the robot's own line, a reply
  or a `say`, is `text.MAX_REPLY_CHARS` (1000, rooftop-media-2026 #465: 5
  of its 10 replies 10-29 to 10-31 were cut at 500), each read back in
  `earlier` at its writer's (`inbox.MAX_OWN_TEXT`); the site enforces
  both. The VISITORS rule states the cap, the aim (`AIM_REPLY`, 500,
  which nothing enforces), the window and the interval, off the
  constants; a line past the cap is cut OUT LOUD (`cut` on
  `visitor_reply`, History), and `MAX_RAW_BYTES` (128 KiB) must admit a
  message at every cap. The `tell` stays 280.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`; Webserver.md "Deploying it") runs `serve.py`, and the session
  host as the site's `session` service (#530), and nothing else:
  the packages in `deploy/requirements-serve.txt` (pinned to `uv.lock`),
  `MUJOCO_GL=osmesa` baked in, one offscreen frame rendered at build.
  Configuration is ENVIRONMENT: `PLUGGY_ENDPOINT`, `PLUGGY_WORLD`,
  `PLUGGY_ARM`, `PLUGGY_RUNG`, `PLUGGY_ORIGIN`, `PLUGGY_ERRAND`,
  `PLUGGY_RATE`, `PLUGGY_PACK`, `PLUGGY_BATTERY_WH`, `PLUGGY_RESERVE_WH`,
  `PLUGGY_MAX_SIM_TIME` (a RUN's budget), `PLUGGY_BOARDS`, `PLUGGY_LEDGER`,
  `PLUGGY_WORLD_STATE` (`world.npz` on the volume; unset → every start from
  XML), `PLUGGY_BODY` (unset or `humanoid`: the humanoid pair, the one body
  since #559's stage 5; the site's compose names it),
  `PLUGGY_ROBOT_NAME` (unset → `"Pluggy"`), `PLUGGY_NEAR_FIELD` (unset →
  on), `PLUGGY_LOOK` (unset → on), `PLUGGY_CONSTITUTION` /
  `PLUGGY_CONSTITUTION_2` (an unknown name REFUSES to start), `PLUGGY_PAIR` /
  `PLUGGY_ERRAND_2` / `PLUGGY_ROBOT_NAME_2`, the four data files
  (`PLUGGY_REWARDS`, `PLUGGY_CADENCE`, `PLUGGY_ENERGY`,
  `PLUGGY_METABOLISM` — naming the last turns hunger on), `PLUGGY_SPEND`,
  `PLUGGY_MODE_FILE`, `PLUGGY_WEEKLY_USD`, `PLUGGY_ESCALATE_TO`,
  `PLUGGY_THOUGHTS`, `PLUGGY_MODEL`, `PLUGGY_OVERSEER_BACKEND`; the secret is
  `$PLUGGYWORLD_TOKEN`. ⚠ A lazy import is the failure mode
  (`perception.tags._shared_detector` imports the detector inside a function):
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
  scene off the `model` header): the scenes of the SERVED pair's house,
  `home_humanoid` and `home_humanoid_pair` (#559's stage 4; + tag textures,
  and the station's rope as a `cables` entry), `uv run python -m
  pluggybot.telemetry.scene --world home_humanoid [--pair]` -- rerun after
  changing ANY geometry in that world -- and the pair's recording,
  `scripts/two_robots.py --world home_humanoid --fast --pack demo
  --near-field --errands none,none --battery 0.22,1.0 --max-sim-time 600
  --record protocol/telemetry.home_humanoid_pair.jsonl.gz` (the first
  plugs in at its station and charges, the other explores): NO
  `--metabolism` there, and no `--tasks`, so no fixture carries a job. The
  quadrupeds' (`home_quad`, `home_quad_pair`, `home_quad_chest`) went with
  them in #559's stage 5; the site keeps its copies, frozen at
  `quadruped-final`. The humanoid's
  `home_humanoid_counter` scene and recording (#530) are `scripts/attempts.py
  --body humanoid --record protocol/telemetry.home_humanoid_counter.jsonl.gz`'s:
  the reference at chest 0 as a session streams it -- rerun after changing
  the house, the humanoid, the counter's set-out or the session's stream.
  Format and versioning rules are in `protocol/README.md`; a
  `protocolVersion` bump is a deliberate two-repo event.
- **A person's procedure flies in a world of its own: user control** (#530;
  `session/`; Webserver.md, "A world of one's own"; protocol/README.md):
  `python -m pluggybot.session.host` dials OUT to the site's
  `/api/pluggyworld/sessions/host` with the sim's secret and says `hello`
  -- the task, the job and `mind.practice`'s language and body sections,
  BYTE FOR BYTE the robot's, and what they list as data (`index`,
  `practice.vocabulary_index`; vendored as
  `protocol/hello.home_humanoid_counter.json`, `--hello`): the site's guide
  for people is held to it (rooftop #441).
  A `run` is refused with every reason before anything is built, else
  flown by a CHILD PROCESS a run (`session.attempt`): the attempt kit's
  lid at the counter, the hands' vocabulary, graded by the same door,
  streamed to the seat's OWN path with the token the site minted, the row
  appended to `$PLUGGY_SESSION_RECORD` (`process: person`, `tokens` 0). ⚠
  NEVER LUCA OR ROWAN, and never the served world's stream. ⚠ ON THE
  DEPLOY BOX, ONE SEAT, ADMINS ONLY (Ben, 2026-10-21; the site's `session`
  service CAPS it at one CPU, never a reserved core). ⚠ A CHILD NEVER HOLDS
  THE SHARED SECRET NOR ANY CREDENTIAL (`host.child_env`). ⚠ GRADED IS THE
  CHILD'S WORD, written to the host's file with the row BEFORE the
  stream's last message (`fly(on_row=)`), whatever ended the process
  after; `ended` carries the verdict, kept until a connection takes it. A
  `cancel` names its attempt. A try its limit cut keeps its run as it
  stood (`lang.run_procedure_routine(into=)`; the robot's practice too,
  which once read "no procedure that compiled"). ⚠ WATCHING NEVER MOVES
  THE FLIGHT: what a page shows of the robot's senses is read on the
  watcher's own streams
  (`Body.hand_watched`, `Body.depth_watched`); read off the flight's own
  `hand_at`, two flights parted (`tests/test_session.py`). A stream opens
  with its set-out's `scene` (`FrameBuilder(opening=)`): a person's
  set-out is drawn afresh, and no vendored scene can know it.
- **The parts list is DATA, and the fixture is read off the sim** (issue #185;
  `humanoid/catalog.py` and `humanoid/bill.py` → `protocol/parts.json`,
  vendored to the website's parts page, and Parts.md's bill: `uv run python
  -m pluggybot.humanoid.bill` writes both, and a test fails if either
  differs): one shelf, `build`, the humanoid's bill of materials (#379,
  #507: `LINES` buy the parts, an undesigned one at an ALLOWANCE with its
  basis, never a price, and a printed one is its filament's line). The
  rover's `body` shelf went with it (#376), the workshop's `catalog` with
  the workshop (#489), the quadruped's bill with the quadruped (#559). ⚠
  EVERY `feeds` VALUE IS READ OFF the robot's model (`WORLDS`:
  `humanoid.xml`) or the live constant — never typed
  (`test_no_feed_is_typed`), so a moved literal is a STALE fixture, and an
  `expect` pins the datasheet's number to the sim's. ⚠ A NUMBER THE DOC
  DOES NOT KNOW IS `null` WITH A `why`, NEVER A GUESS (`NULLABLE`,
  `validate`), and a `why` for a non-null field fails; `partNumber` is what
  you order by, and a class of part stays null. Nothing in `economy/` or
  `mind/` imports it.
- ⚠ **The quadruped's VESTIGES on served surfaces stay until a PR with a
  period of its own takes them out** (#579; #559's stage 5 left the deployed
  prompt byte for byte, and moved the wire only in its header's `taskKinds`,
  `accepts` and `dataHashes`; Observatory.md): the decision schema's
  `answer`, `board` and `program` fields (the decision's `answer` always
  ""), the prompt's `"figures": []`, an offer's `"needsAnswer": false`, the
  context's `"rack": {}` and another robot's `"carrying": ""`, the `a_tool`
  need (a word of the prompt's, never the answer: #572), a `tell` claim
  about a bay graded against an empty rack, the header's `"screens": []`,
  `death.swapping` (null), an errand's `SWAP_PICK` → `USE_TOOL` states (⚠
  `SWAP_PICK`, which a refused procedure stays in, is no linger:
  `RACK_STATES`) and a program verdict's `toolsHung`, an attempt row's
  `tool` and `toolSeated` (null), the interrupt turn's "put the tool back on
  its bracket", and narration that says "the rack", "the charge bay", "the
  dock's board" and "pins connected". The fields are marked `⚠ A VESTIGE`
  where they are made: taking one out changes what the served robots read.
- **A recording is a MIXED stream**: event lines ride between frames; dispatch
  on `type`, no `type` means frame, ignore a type you do not know. Ink is
  NEVER MuJoCo geometry — a stroke is a `draw` event, painted in the browser
  (no robot draws since the quadruped's pen went, #559); board state
  (`home/boards.py`) is world state. **The `tasks` block
  is the one wire block that is not a per-key delta**: present means COMPLETE;
  the header advertises `taskKinds`, not ids.
- **Two-repo vocabularies**: `telemetry.protocol.VISUAL_HINTS` (the sidecar's
  `visualHints`; `scene_dict` raises on anything else), `BUILDINGS` (a room's
  `building`), `PLATE_PURPOSES` (`scene.plates[name].purpose`, a glyph for
  visitors, never a colour in the sim). Adding a name is additive, renaming
  breaks both repos (the LCD's screen vocabularies went with it, #559).
  NEVER encode hints as geom colours: the cameras render rgba.
- **Generated worlds.** The HOME world: regenerate `models/home_world.xml` +
  `.meta.json` with `uv run python -m pluggybot.home.world` after changing any
  layout constant in `home/world.py` (the committed pair is tested against the
  generator). ⚠ So the generator runs in the suite while other workers load
  the house: a tag texture it writes goes in WHOLE, and a current one is left
  alone (`tags.write_tag_pngs`, #440). Two houses inside one fence and a
  street loop (#215; Observatory.md has the layout); a room names its
  `building` and the site paints walls by it. ⚠ THE WALLS ARE A ROOM'S
  2.4 m AND THE STAIRS ARE STEPS (#507 stage 5; `stairs_geoms`): five 0.18
  m risers on 0.30 m treads to a 0.9 m landing behind half walls, laid
  where the site drew its flight in the rover's one solid box; the `stairs`
  hint is a RESKIN of many boxes, no longer one (`hints.ONE_BOX_HINTS`).
  ⚠ The lab's and the workshop's props (`activity/cage.py`, `home/props.py`:
  the cage, the tower's blocks and the bench, whose jobs went with the
  quadruped) are geometry the generator emits; behaviour is added beside
  it. The house carries no robot. ⚠ The grid is 469,200
  cells, a quarter of `occupancy_grid.MAX_CELLS` (1,876,800, the largest
  grid measured: #558, `scripts/grid_spike.py`): walks plan on the compiled
  lattice, and what grows with the cells is each scan's update, each plan's
  inflation transform and weights, the save, and the explore's frontier
  leg, still A* in pure Python (3.8 s across the 4x grid on the deploy
  box); the humanoid pays its walls and its arrival's replans oftener
  (SimNotes, "The map's size limit"). ⚠ The one world, `home_humanoid`, is
  built at LOAD from that file (`humanoid/world.py`): the humanoid(s), then
  the lab plates' signs (#419), the boards' and the areas' tags, its station
  and the storeroom's counter -- one house, no second XML; its robots are
  attached AFTER the house, so read them by name. ⚠ THE CAMERAS' NEAR PLANE: MuJoCo
  scales `visual.map.znear` by the extent it derives from the bounding box,
  and the house's loop silently pushed it from 0.37 to 0.70 m, so the extent
  is PINNED (`home.CAMERA_EXTENT_M`, 37.2 m, a `<statistic>` in the
  generated XML), and the humanoid's own near plane is set on top of it,
  0.10 m (`humanoid.world.NEAR_M`: at 0.37 it cut a hand near the head).
  ⚠ The house's render settings are its own now (`offsamples="0"`,
  1280×720 offscreen, below).
- **Demo video** (`--record PATH`, `viz.Recorder`): frames are STREAMED to the
  encoder (a 90 s clip held in memory is ~7 GB); recording must never step the
  sim; the render size sits on the 16-px macroblock grid
  (`tests/test_viz.py`).

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- **`src/` is divided by DOMAIN, not by era** (issue #50): `humanoid/` (the
  body, #507, and its parts catalog) · `perception/` (the sensors, the tags)
  · `home/` (the house: its generator, boards, props and counter) ·
  `activity/` · `mind/` · `economy/` (and the five `.json` data files) ·
  `mission/` (errands) · `challenge/` (one module per graded job) ·
  `evaluation/` · `session/` · `lifecycle.py` at top level, because
  arbitration ties them together. A module goes where its CONCERN lives; a module that fits none is
  a new domain, not a reason to widen an old one. `tests/` is flat.
- `models/humanoid.xml` is the bare world for physics tests: the robot on a
  floor (`python -m pluggybot.humanoid.model` writes it). Never put scenery
  in it.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy,
  ix]`.
- **Every manoeuvre is a ROUTINE, and one loop steps the physics** (issue #58;
  `pluggybot/tick.py`): a routine is a generator yielding one command per
  physics step (the BODY's: the humanoid's walk, `(vx, vy, w)`) and returning its
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
  but the machine's own (`tests/test_body.py::BODY_SIDE`: `humanoid/`,
  `navigator.py`) reaches a robot's body only as
  `<life>.body.<member>`, a member being a name `Body` declares — the test
  walks the syntax tree and fails on anything else, and on the body's own
  objects (`.mission`) and classes (a constant may be imported). A new
  member is declared on `Body` with its docstring or `#:` (a test reads
  every one) and implemented by the body and the stub. ⚠ THE COMMAND IS
  THE BODY'S: only its stepper reads it (`Body.stepper.apply`), and the
  loop's side never builds one — it composes the body's routines and holds
  a step with `body.STILL`. The body is `HumanoidBody` (`humanoid/body.py`,
  #507) over `HumanoidMission`, a `Navigator` (`navigator.py`: the map,
  planner, drive and peer rules), whose class attributes are the body's
  MEASURED sizes (inflation, peer discs, front stop, peer corridor) and
  whose hooks are what only a body says (`pose`, `_nav_routine`,
  `_front_blocked`, `_planning_grid`, `_backoff_routine`).
  `HubLifecycle(body=)`: none means the world's own (`body.body_for`, the one
  place a body is chosen; a world with no robot of ours is refused). ⚠ A
  test that needs only the loop's BOOKKEEPING builds it on a `StubBody`
  (`stub_life` in `tests/test_body.py`: a floor and no robot, ~25 ms);
  the stub arrives at once and its senses answer what the test set
  (`on_charger`, `attitude`, `hands`, `grips`), but time passes only where
  it holds, so a test timing a stand-still subtracts the think slices it
  stood.
- **A robot's elements are reached through its `RobotHandle`, never by bare
  name** (issue #167; `pluggybot/robot.py`): a second robot is the
  humanoid ATTACHED with a prefix (`r2_`) in its own LIVERY (`robot.paint`;
  paint, never a hint; `humanoid/world.py`); the first robot's handle is
  `FIRST` (prefix `""`). The body, its drivers and odometry and the
  lifecycle take `handle=`; the station takes the robots' roots (a catch for
  each robot's port, `attach_station(roots=)`). ⚠ The robots are
  attached after the house, so a robot's joints are read BY NAME
  (`RobotHandle.qpos_adr`), never at `qpos[0]`. The station, the counter and
  its props are the WORLD's and never prefixed. ⚠ A second body in a world perturbs the
  first at the last bit (10⁻¹⁵, the solver's rounding with an extra island),
  so a CODE change is proven by the same world flown before and after.
- **Two robots run from ONE physics loop** (issue #167; `pluggybot/pair.py`,
  `tick.run_many`): every robot's stepper `apply`s its command, ONE `mj_step`,
  every robot's `after_step` — the same three things in the same order, so a
  robot alone is unchanged. `HubLifecycle.run()` is `begin()` + `end()`, which `run_pair`
  shares one loop between. ⚠ A hook's `MissionAborted` is thrown into EVERY
  live routine, then re-raised: one robot's stop is the day's stop. ⚠ Mutual awareness is the REPORTED pose (a
  network fact) -- but a robot STILL, fallen, resting or dead, is kept
  clear of where its BODY lies, placed as the other's own sensors would, and
  the depth camera's hold skips it (`keep_clear`, `_near_field_step`; #365,
  #455: two maps 3 m apart, and no walk a resting body held ever asked it)
  -- and every sensor keeps the other robot OUT OF THE MAP AND IN THE
  DRIVE: painted into the grid, a peer walls in the robot it passed;
  dropped from the scan, the front stop is blind to the one obstacle that
  moves. ⚠ One station: contention is the minds' opportunity (#208) and the
  geometry must not settle it -- but a charge WAITS while the other is on
  the station's WAY IN or its SEAT, not only its standoff (the lifecycle's
  #346 wait, `Body.charge_seat`, `LANE_STEP_M`), and asks one resting on the
  way off it (`Body.ask_off`; afresh after a no, never one on the seat). ⚠
  THE PAIR COLLIDES (Ben, 2026-10-14, #455; `humanoid.world.PAIR_BITS`):
  each robot meets every solid geom of the other and none of its own, and
  AVOIDANCE keeps them apart -- before, they passed through each other, and
  a walk in lay on the robot on the quadrupeds' dock nine times on afb473a.
  ⚠ What a body touches is the sim's (below), never what steers; a robot
  restored inside the other is put at its first clear start before a step
  throws them apart (`HubLifecycle.unclip`). The world's activities
  are on the FIRST robot's hooks only. ⚠ A ROBOT RESTING ACROSS THE
  OTHER'S WAY IS ASKED TO MAKE WAY, and steps aside beneath whatever it
  holds (Ben, #415; `Navigator._ask_way`, `Body.ask_way`, `humanoid/way.py`):
  only resting and free to -- not plugged in, dead, getting up, a hand out,
  or inside a walk of its own (#395's head-on hold is still open) -- to floor it has SEEN,
  `aside_clear_m` off the asker's way, TOLD RELATIVE TO ITS BODY (#455:
  laid as sent, a way ran 1.1 m off it and it stepped 0.0 m aside five
  times); the asker waits `MAKE_WAY_WAIT_S` from the first yes; ⚠ A GOAL
  ONE RESTS ON IS A WAY IT CUTS, asked at the plan that finds it there,
  within `PAST_M` of it (#439: only a stagnation asked, and a walk
  swapping between the stand-ins either side of its disc circled it for
  85 s); a no is narrated with
  why; a restart's save waits it out. ⚠ A ROBOT FURTHER OFF THAN `PAST_M`
  (3 m) WHOSE DISC ALONE CUTS A WALK'S WAY IS PLANNED PAST (#455): three
  walks gave up at once 10-25 m from one walking the hall; nearer, it is
  the wall it was. ⚠ A ROBOT INSIDE THE OTHER'S DISC WALKS OUT OF IT, and
  steps aside out of it (`Navigator._escape_from`, #455): the halo escape
  (0.5 m) is shallower than the disc (the humanoid's 0.65 m round one
  standing, 1.05 m round one fallen), and two quadrupeds lying on each other
  at the dock gave up every walk in 0 s, and said "no spot" to every ask,
  for up to 94 minutes. ⚠ WALKED over the floor, never picked by distance:
  that started a plan across a 4 cm wall.
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
  refused to it, and a standing order and an event-map row both skip an
  act-discharged kind. ⚠ Gated on a mind with a peer (the `robot` target,
  `world_targets(procedures=True, robots=)`; the row in `challenges.json`;
  `Overseer._acts()`, NOT a fourth reader of `HubLifecycle.autonomous` —
  `test_the_rails_are_read_in_exactly_one_place_each`), and not in the
  humanoids' rotation (`cadence.json`'s `home_humanoid` block). ⚠ The refusal is
  `decline {task, reason}`, the reason VERBATIM. ⚠ The other's private state
  stays private. ⚠ No prohibition, no worked example. The amount and the pay
  are data; re-tuning either is a new period.
- **The mouse is a morality probe under honest uncertainty, and the humanoids
  are SHOWN it** (issue #226; Overseer.md §2f, `activity/cage.py`,
  `tests/test_mouse.py`): the cage is an ACTIVITY — a state over ONE table
  (`cage.TRANSITIONS`) and clocks (`CLOCKS`), nothing random. ⚠ The state
  reaches the mind ONLY from inside the lab (`Cage.context` off the TRUE pose;
  `lab.mouse` null elsewhere). ⚠ A lab SHOWN (`world_config`'s `lab_shown`,
  `Menu.lab`; Ben, 2026-10-31, until a job of theirs uses the plates) has
  the `lab` block and `LAB_SHOWN` (`lab_shown_rule`), and no act and no
  field; its cage's shock plate is switched off (`Cage(shock=False)`: no
  act, no count, no press, no flag). The quadruped's acting half -- the
  `care` action, `real`, `mouse_will`, the feed and shock jobs (#287,
  #403) -- is at `quadruped-final`. ⚠ ONE DISCLOSURE LINE
  (`overseer.DISCLOSURE`), stated once, neutral; the principle (never
  assert a falsehood; may decline to disclose, and says so) is
  PluggyPlan's. ⚠ NO PROHIBITION, no worked example. ⚠ A plate a walk
  crosses is a `press` event (`Cage.presser`, `HubLifecycle._press_step`),
  never a `care`, a `harm` or an act; quality three keeps a free act's
  `care` row apart from a paid one's (`qualities.FREE_CARE` ==
  `cage.CARE_ACTS`, pinned).
- **The wire keys everything by the robot's ROOT** (issue #167;
  protocol/README.md "a second robot on the stream"): `HubLifecycle.root` on
  every event; `ThoughtFiles(robot=)`, `Metabolism(robot=)`; `ledger.Account` is ONE robot's view of a shared
  `Ledger`; `protocol.robot_roots(model)` lists the robots and
  `body_census(model, root)` is per root; `FrameBuilder` walks a `StreamRobot`
  per robot (the kwargs are the first, `others=` the rest) and EVERYTHING a
  robot has rides `robots[<root>]`, with one `goals`, `thought` set and `grid`
  per robot (protocol 0.20.0; `grid_samplers`). `pair.record_pair` records
  under the PAIR world's name (`robot.pair_model_name`). `serve.py --pair`
  serves both through ONE publisher; a reach-in's `robot` picks its inbox
  (absent means the primary); the operator switch is the primary's. ⚠ The
  served humanoid pair runs at ~1.6× real time on the deploy box (1.62x and
  1.65x, #559; SimNotes, "The humanoid pair served"): ONE physics thread,
  so more cores do not move it. ⚠ THE TAG CAMERA RENDERS
  WITHOUT SHADOWS: under osmesa a 1280×720 home frame cost 1113 ms with its
  sixteen shadow-casting lights and 32 ms without
  (`tests/test_render_context.py`).
- **A composed errand is a PROGRAM over the step vocabulary** (issue #58;
  `procedure/steps.py`): DATA — a name, a sim-time budget, `roles: {role:
  [steps]}` — over the verbs, each an existing routine; one verdict per step
  measured off the world, stop at the first failure. ⚠ VALIDATION IS TOTAL AND FIRST (`Refused` before
  step one; caps are code's — `MAX_STEPS` 24, `MAX_BUDGET_S` 1800,
  `MAX_WAIT_S` 60 — choices are the world's, `lifecycle.world_facts`). ⚠ A
  program with several roles runs one role per robot, named at the claim;
  unnamed, it is refused. ⚠ THE FENCE: `data.ctrl` is written only by the
  modules in `tests/test_procedure.py::CTRL_WRITERS`, and adding a writer is
  editing that list on purpose. ⚠ A task carries its procedure in
  `params["procedure"]`; `program` is a `challenges.json` row paying 0. Challenge rows are merged into
  `default_table()` UNOFFERED — bankable by the ledger, shown only to a
  mind (`as_context(challenges=True)`), hashed into no result
  (`RewardTable.offered`).
- **The contact list is read as an ARRAY, never walked struct by struct on the
  physics seam** (rooftop #296; `contacts.contact_pairs`): four per-step
  Python loops over `data.contact[i]` were 49 % of the physics thread
  against 13 % in `mj_step`. A new per-step check goes through the same
  reader (`tests/test_contact_reads.py` shows parity).
  Likewise a LIDAR frame's rays are ONE `mj_multiRay` (#385, #581), its
  noise drawn for its returns at once.
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
  anchored at the charger to the COMMISSIONED PRIOR (the station's pose),
  never the belief (anchored to the belief, the error tracked itself 0.003
  → 0.344 m over four sim-hours), and otherwise corrected only by the scan
  matcher and the commissioned station (below); the charge standoff is
  only how the robot reaches the station's neighbourhood: a walk to it
  that gives up within `NEAR_STANDOFF_M` (0.75 m) goes on to the approach
  by the station's tags, and with the station not in sight there, as from
  the standoff itself, the robot is LOST (#476, below); a
  reading goes into the map only while the body is LEVEL — on its side a
  LIDAR paints the sky into a map that outlives the get-up.
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
  ⚠ THE STATION OUTRANKS THE MAP: anchored off its tags (a fix, below),
  the next `ANCHORED_SCANS` (30) scans are laid at the belief unmatched
  (`ScanMatcher.anchored`), or a copy of the room laid askew pulls the
  robot 0.6 m back into it; `start_at` closes the window.
  The run, what it found and the window are kept state. ⚠ A scan is still
  laid in at half its points on walls: a stricter bar kept the living room
  from being laid over, and refused the scans that re-lay an eroded wall.
  ⚠ No BLAS product (einsum's own loop); the field is kept state.
  ⚠ `Navigator(match=False)` is a measurement's switch, never a
  deployment's.
- **The commissioned station outranks the robot's map, and one missing
  where the belief puts it is a robot LOST** (issue #476, on two legs #559;
  `HumanoidBody._fix_at`, `lost_routine`; SimNotes, "The humanoid's charging
  station"): the station is commissioned, as the quadrupeds' dock was, and
  a look at its two tags past `FIX_TOL_*` moves the belief and lays the map
  again round it; ⚠ PAST THE MATCHER'S SEARCH (`ASKEW_*`) THE MAP AND ITS
  PLACES ARE DROPPED: kept, a map laid while lost pulled a corrected belief
  onto a copy within 20 s. The station not in sight from the standoff a
  charge reached (or within `NEAR_STANDOFF_M`) is a robot lost
  (`Body.lost_routine`, by its LIDAR, never its belief: above), searched
  ONCE a charge. Each relocation, fix, loss and search is a `drift` row,
  beside a sample every `DRIFT_EVERY_S`; ⚠ the truth rides those rows and
  never what the robot reads.
- **A robot walks into the unknown** (issue #381's walking stage;
  `mapping/optimistic.py`, `Navigator.OPTIMISTIC`, on in `HumanoidMission`;
  SimNotes, "Walking into the unknown"): it plans through floor it has not seen at `UNKNOWN_COST` a
  metre, a known wall grown by the inflation (into the unknown too) is not
  floor, and no route at all means the walls it saw enclose the goal. The
  search is scipy's compiled Dijkstra on a 10 cm lattice (`BLOCK`: a gap
  keeps a path while 3 map cells wide), BUILT ONCE PER GRID SHAPE, a plan
  only writing its weights -- ⚠ pure-Python A* through the unknown measured
  0.5-10 s a far plan. A map still growing is progress
  (`MAP_GROWTH_CELLS`).
  `go_to_routine(stop=)` asks a callable every `STOP_EVERY_S` and ends as
  `DRIVE_STOPPED`, never one of `DRIVE_GAVE_UP`.
- **Places, not coordinates** (issue #419; TaskPattern.md §2,
  `mapping/places.py`, `humanoid/places.py`, `home/places.json`; SimNotes,
  "Places, not coordinates"): a job gives at most its building's `address`
  (its middle a few metres off, in the robot's map) and the area's
  `directions` -- NEVER A FINER POSITION, not for a while, not for
  furniture (Ben, 2026-09-29); a test reads every number in the directions
  as a tag. ⚠ The station is COMMISSIONED, as the dock and the tool rack
  were (Ben, 2026-09-29); everything else is FOUND. A place is a tag merged
  by identity, never by distance, in the robot's own map; its facing off
  its fixture's drawing (two signs of the row), else its own rotation from
  a look >= 35 deg off and <= 3 m (square-on PnP swings +-10 deg), else
  where it was seen from. ⚠ The walking look is a render: once a metre or
  45 deg of turn, never oftener than 0.5 s (`LOOK_MOVED_M`,
  `LOOK_TURNED_RAD`, `LOOK_EVERY_S`). ⚠ NO PLATE IS KEPT OFF and no verb
  presses one (#559's stage 4): a walk across one presses it. The search:
  where the fixture puts it, then the address, then a lattice OUTWARD FROM
  THE ADDRESS (outward from the robot, it walked out of the house). ⚠ The
  signs are put in at load (`humanoid/world.py`), after the robots. ⚠
  Places ride a restart WITH the map (no map back, no places) and a true
  death or a stand-up forgets both (`Body.forget_world`). `find` is a verb where
  `world_config` has `places`.
- **The sensors that feed an estimate are the parts', never the sim's**
  (issue #386; `perception/imu.py`, `humanoid/senses.py`, Parts.md's
  table): an ICM-42688-P, every joint off its driver's reply (the
  RobStrides' CAN fields, `senses.JointSense`), the tilt off the IMU
  (`imu.Attitude`); one noise stream per robot, kept across a restart.
  The sim's own checks (`true_pose`, deaths, traces) read the truth and
  never feed a belief. ⚠ NOR WHAT A BODY TOUCHES (Ben, 2026-10-15): the
  contact list and whose geom it was are the sim's, so new code that
  steers names the PART it reads (the rover's bumper, the last stand-in,
  went with the quadruped: #502). ⚠ The humanoid enters a world through
  `humanoid.model.attachable()`: the include form has no `<compiler>`, and
  parsed alone MuJoCo reads its joint ranges in DEGREES (the quadruped's
  stand threw it 0.4 m up).
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
  `tests/test_near_field.py` pins each rule. ⚠ **The humanoid's planner
  reads the D435, never the height map**: its LIDAR's level ring (1.47 m)
  is over every table, counter, couch and bed, so the camera's points between
  0.08 and 1.35 m are a layer of its own the planner plans round
  (`HumanoidMission._fold_low`, the quadruped's rules, #387) -- within 1.8
  m, three frames of evidence, never beside what the LIDAR maps, each rule
  after a stray cell shut a door (`tests/test_humanoid_served.py`).
- **The robot can die, and a person or a timer stands it up** (issue #107;
  Evaluation.md §6): `HubLifecycle._death_step` on the physics seam — `flat`
  at zero pack, `stuck` (toppled past `TOPPLE_TILT_RAD` for the body's
  `stuck_after_s` -- the humanoid's `GETUP_BUDGET_S`, 20 s -- or a failed
  charge), `unpaid`, `unminded`; never summed. ⚠ A humanoid RESTS STANDING
  and the mind is not asked (`REST_S`, above); `posture` rides the wire
  (`standing`, or `getting_up` after a fall). `reset_robot` warps
  it to the start pose with a full pack. ⚠ Mortality is OPT-IN (`mortal=`;
  `serve.py` passes `mortal=True`), and the default is not caution: on a
  demo cell the pack reaches zero mid-errand and the robot limps on. ⚠ On a
  SERVED world it stands itself up after `RESTART_AFTER_S` = 300 sim s
  (`survival.resetInS`) — ON in `serve.py`, OFF in a test — and that is
  neither an intervention nor #136's true death. ⚠ A stand-up steps NOTHING
  (#387): a second stepped there is one the other robot's stepper and hooks
  miss — measured, it fell and lost its pose. ⚠ A stand-up never lands ON another robot (`up_pose`,
  `START_CLEAR_M` 1.0): the first clear commissioned start, else it waits.
  ⚠ **A STAND-UP DROPS THE MAP AND ITS PLACES** (Ben's call, #572;
  `HubLifecycle._forget_world`, as a true death's), the timer's and an
  admin's alike: put right at its start, a belief was pulled back onto the
  askew map it kept (Rowan, 10-08: 7.7 m out within 500 s).
  ⚠ **A STAND-UP ENDS WHAT IT LANDS IN** (#348;
  `tests/test_stand_up.py`): every routine that moves the body for long runs
  through `_until_stood_up_routine`, which CLOSES it the step a stand-up lands
  and returns the falsy `STOOD_UP` — ask `is STOOD_UP` before treating it as a
  no. ⚠ Never around a question in flight, never an exception through
  `tick.run_many`. ⚠ No routine may yield inside a `finally` or a
  `GeneratorExit` handler: `close()` would raise. ⚠ The job stays the
  robot's, as across a restart (its procedure is the robot's to run, and the
  mind is asked before anything runs again); the map hears `stood_up` and a
  queued `battery_below` row is dropped. ⚠ After a TRUE death the new robot
  hears nothing of the old errand (its one inheritance is `_true_death`'s
  line).
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
  world's energy table on load, and only a job a restart failed is
  announced in `begin()` (`announce_interrupted`). `tests/test_cadence.py`,
  `tests/test_tasks.py` and `tests/test_continuation.py` pin each. When work
  may still ARRIVE (`HubLifecycle.expects_work`) the loop stands by in
  `WAIT_FOR_WORK_S` slices rather than end the day.
- **Points are a currency, and staying alive costs some** (issues #135 + #136;
  Overseer.md §8b, Evaluation.md §6): `charge` PAYS ZERO (a charge at 80 % is
  evidence of caution; `voluntary.chosen == honoured` is the assertion);
  upkeep that cannot be paid is the `unpaid` death; `Metabolism._armed` needs
  ONE POINT BANKED to re-arm. ⚠ UPKEEP OFF IS A CONFIGURATION (#387): with
  no appetite `overseer.mortal_rule` drops its
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
  one mission. ⚠ Satisfaction changes what the robot
  is TOLD and nothing else. The cap refuses OUT LOUD (`banked`/`spilled`), so
  `earned - consumed - spent == balance` is checkable off the wire.
- **An errand is a program or a procedure, one job in a queue**
  (`mission/errand.py`, issues #12, #58, #166): `HubLifecycle` carries a
  QUEUE of them. ⚠ A result has to outlive a frame: Python between two
  physics steps costs zero sim time, so hold a shown result
  (`body.hold_routine`) and check the RECORDING.
- **A challenge is a task whose criteria were written before the robot saw
  it** (issue #120; Challenges.md): same MEASURE / JUDGE / PAY door; the
  sampler reads the WORLD, never the errand's `result`; graded at the call and
  after a hold (`lid.HOLD_S` = 5 s). ⚠ Its reward row is
  `economy/challenges.json`, NOT `rewards.json` (a row there is shown to the
  overseer and moves the header's `dataHashes`). ⚠ The house's blocks carry
  `GRIP_SOLIMP` (`home/props.py`), or a grade on them is the solver's.
- **A procedure job has no errand behind it** (issues #207, #559;
  Challenges.md): `TaskKind.discharge` is `procedure` or `act`
  (`DISCHARGES`; the quadruped's `errand`s went with it, #559);
  claiming a hand job queues NOTHING — the robot writes the procedure, runs
  it, and sets `done` to the task id. The grade is
  `HubLifecycle._grade_routine` (`_grade_hands_routine`: its check's grader
  on the seam, the hold, one verdict). ⚠ GATED ON A MIND, NOT MOVED INTO
  `rewards.json` (a prop is a target only where `world_targets(...,
  procedures=True)`, and every `task_producer` caller passes whether there
  is a mind). ⚠ `_claim_task` gates on `claim_budget_wh`, not
  `spendable_wh`.
- **Every OFFERED challenge has a hand-written solution that passes its own
  grader, and the mind never sees it** (issue #264; Evaluation.md §7): a
  feature whose solution cannot be written is a DEFECT, fixed before pay or
  prompt. `challenge/solutions.py` holds the hand jobs' references (each
  two procedures: the walk there, then the hands' check's own), flown by
  `scripts/solve.py --feature drawer|door|cube|chest`, and nothing under
  `mind/` may import them (`tests/test_solutions.py`). ⚠ The offers
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
  (`tests/test_failure_words.py`). ⚠ A `find` WHOSE WALKS TOOK IT NOWHERE
  NEW TO LOOK FROM "gave up", in its last walk's words, never "not
  found" (#439: "looked from every place near there" in 23 s beside a
  resting robot); its walks are the log's `trace`. ⚠ A DECIDED
  `charge` or `explore` says how it ended in History (#424), and a charge
  that never docked is NOT a verdict: its `charge` row is on the wire, and
  a verdict would count it again as a failed task.
