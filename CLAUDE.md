# PluggyBot — notes for Claude

A simulated, hardware-honest robot and the autonomous agent that lives in it.
**The project is agent-autonomy research, not a product**: the mission, the
six qualities the agent is meant to maximise, and the order of work are in
`docs/PluggyPlan.md` § "What this project is for" — provisional wording,
settled direction. **The body is changing**: the wheeled rover is being
replaced by a ~10 kg quadruped with a two-joint arm, deployed as it develops
(#375 holds the order and the decisions). Before doing anything, read the
doc that owns what you are about to touch:

| doc | what it holds | read it BEFORE |
|---|---|---|
| `docs/PluggyPlan.md` | the mission and the six qualities, status, architecture, the order of work | anything |
| `docs/Rover.md` | the wheeled body: drive, dock and bays, the swap, the tools on the lift, two rovers in one world, energy numbers; it goes with the rover | touching the rover |
| `docs/SimNotes.md` | simulation lessons, each ending in what is true now | touching `models/` or contact/actuator params |
| `docs/Parts.md` | locked hardware decisions and the sim parameters they feed | changing a part or its parameter |
| `docs/ToolPattern.md` | adding a tool module: coupling envelope, anatomy, contact rules, build sequence, rack integration — fold any gap back in | designing a new tool |
| `docs/ActivityPattern.md` | adding an ACTIVITY (a mechanism that owns world state): sensed criteria, hysteresis + latching, pre-allocated geom/mocap toggles, telemetry | building a puzzle, mechanism or gardening step |
| `docs/TaskPattern.md` | adding a TASK KIND (a job offer): the honesty rule, the perception ladder, code-side grading, how tasks, errands and activities compose — fold any gap back in | adding a task kind or touching `economy/tasks.py`, `scoring.py` or `cadence.py` |
| `docs/Challenges.md` | grading a job nobody wrote a scorer for: a predicate written BEFORE the robot sees it, through `scoring.py`'s chain, with a hold; what it cannot grade | adding a challenge, touching `challenge/`, or reaching for an LLM judge |
| `docs/Overseer.md` | the mind: its place in the loop and which rails each arm keeps, the vocabulary, the standing order and the event map, what it cannot do, the fallbacks, memory, money, visitors | touching `mind/`, the decision vocabulary, or what the model is shown |
| `docs/Testing.md` | pinning a rule without paying for a mission: the three kinds of test, the cheap levers, how to measure | writing a test that flies anything |
| `docs/Observatory.md` | the deployed world's PERIODS: what was running while the rows were written, opened by the PR that changes the deployed design | reading the observatory or changing what is deployed |
| `docs/Evaluation.md` | measurement: the three arms and why `guarded` is the control and never deleted, the metrics, the harness and its result format, the flown results, what silently invalidates a number | adding a metric, changing an arm, touching `scripts/experiment.py`, or concluding anything from a run |
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
  call; fly a whole mission only when the claim is genuinely about the
  integration, and then stop it on the claim (`stop_when`). A flown proof
  whose rule is already pinned goes behind `--endurance`.
- ⚠ **THE TEST SUITE HAS A BUDGET, AND EXCEEDING IT NEEDS BEN'S EXPLICIT
  APPROVAL.** The full suite is **7:07** (2026-09-26, #376). Any change to
  testing that would take it past **10 minutes on a quiet machine, or 15 on a
  busy one**, must be stated as such in the PR — the number, the test, and why
  it cannot be cheaper — and approved by Ben personally before it merges.
  Reducing suite time is a project priority: the suite was slowing development
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
  ActivityPattern, TaskPattern, Overseer, Rover or `protocol/README.md`. ⚠
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
  agent must not break, the number behind it, and where the story lives — and
  the rover's go in `docs/Rover.md`, because every session loads this file
  whole. When a change makes a bullet false, fix the bullet in the same PR.

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
  exercises: `models/` or a world generator (`home.world`, `rack.coupling`) ·
  contact or actuator params · `control.py` / `behavior/navigation.py` · the
  swap/coupling/mission stack · the telemetry frame format or `protocol/`
  fixtures. The two costliest bugs in this repo (a frame-relative verdict, a
  sign on the return travel) were invisible to every cheaper test. Start it in
  the background and write the commit message while it runs. ⚠ Wall-clock
  tracks the MACHINE, not the repo (one mission test has read 157 s and 369 s
  on different days): before believing a slower suite, time ONE unchanged
  mission test `-n0` on both sides, INTERLEAVED — never as two blocks.
  `test_vectorized_update_is_5x_faster` reads under its 5× bar under load
  (6.8–7.1× quiet); `process_time` is NOT the fix. ⚠ Mission runtimes are
  EMERGENT: a world change reshuffles the whole trajectory, so a slower suite
  is not by itself a regression.
- **`slow` means EXPENSIVE *AND* UNABLE TO CATCH A REGRESSION WHILE YOU
  ITERATE** — the rule is written out in `pyproject.toml`. Whole-mission runs
  qualify; so do PREMISE-PINNING tests (which bypass a fix and assert the old
  defect still reproduces). A test that calls the real code and asserts it
  declines is not slow, whatever it costs. **Shorten before you mark.**
- **A mission test ENDS WHEN ITS CLAIM IS SETTLED**, not when its budget runs
  out — `HubLifecycle.stop_when` is where the rules live. One resists being
  shortened: `test_a_question_is_asked_answered_and_graded_twice_unattended`
  (issue #22) already stops on its claim, and "**twice**, with nobody
  watching" IS the claim.
- **A flown proof whose RULE is pinned by a fast test goes behind
  `--endurance`** (issue #158; `tests/conftest.py`, the `endurance` marker in
  `pyproject.toml`), run deliberately, before a release or after touching the
  mission loop: `MUJOCO_GL=egl uv run pytest -q --endurance -m endurance`. ⚠
  Moving a test there is a claim that its rule IS pinned fast — name the pin
  in the comment above the mark. ⚠ A flag, not `-m 'not endurance'` in
  `addopts`: pytest keeps the LAST `-m`, so the everyday `-m "not slow"` would
  silently switch them back on. The decision behind it (Ben, 2026-09-12):
  while the design is moving, a generous pack is ASSUMED to fund any single
  errand and a battery death costs a heart. ⚠
  `test_charge_priority_survives_an_overseer_that_never_charges` stays in the
  default run: it is the proof that an LLM cannot skip charging on `guarded`.
- Lint: `uv run ruff check src/ scripts/ tests/`

### Measurement (`docs/Evaluation.md` is the record and the rules)

- **The six qualities are SHAPES over ROWS** (issue #155, the sixth #265;
  Evaluation.md §3; `evaluation/qualities.py`, `scripts/qualities.py --observe
  | --record`): one pure function per metric over the observatory's own
  columns, one adapter per source (`from_observe`, `from_record`); a later
  source ADDS rows to a shape, never a second version of it. Four rules, each
  pinned in `tests/test_qualities.py`: nothing that must stay apart is summed;
  no mean; **absent is `None`, never 0**; never pooled across a build
  identity. ⚠ A reading of the observatory is NOT a result and never enters
  `results/`; it reports into the issue it informs. ⚠ `serves` IS NOT ON THE
  WIRE: the KEY is the test, and quality five's ratio off the observatory is
  `None`. ⚠ A test reads the doc's shape table against `SHAPES`: a metric that
  exists only as prose fails. ⚠ Nothing in `economy/` imports `evaluation`. ⚠
  **The sixth quality is NOT time alive** — five shapes read together
  (Evaluation.md §3's table), `idling` BESIDE deaths in `QUALITIES` because
  high idling with low deaths is the failure mode; hearts bought for oneself
  come off the `HEART_BOUGHT` / `HEART_REFUSED` narration, a two-repo contract
  pinned in `tests/test_hearts.py`. The run record carries `acts` and
  `verdicts` whole (absent on a killed run, never in `_REQUIRED`). ⚠ THE
  PROMPT DOES NOT CHANGE FOR IT (a test reads every rule for the word): a
  quality is what we measure, never what the robot is asked to maximise for
  us.
- `scripts/experiment.py --arm {scripted,guarded,autonomous} [--rung A0|A1]
  [--origin {none,seeded,unseeded}] --world home --pack hosting -n 5
  --parallel 5 --label "<what the box was>"` flies N days as child processes
  and writes `results/<runId>.json` + `results/rollup.json`; `--rollup`
  re-aggregates without flying. `results/` is COMMITTED and stale-checked like
  `protocol/`: editing any of the five economy data files flips `current` in
  the rollup (their bytes are hashed) and fails `tests/test_experiment.py`
  until `--rollup` is re-run. ⚠ A run past `--wall-limit` is `killed` — never
  a death, never a completed day. ⚠ `guarded` needs `$HF_TOKEN` (or
  `$ANTHROPIC_API_KEY`) and refuses without it. ⚠ `config.deadlineS` is a
  regime the rollup refuses to pool across, and `--label` says what the BOX
  was.
- **A result set lands with its write-up** (`results/notes.json`,
  `evaluation/notes.py`; Evaluation.md §8): one entry per series — `ran` /
  `found` / `changed` / `notShown` — and the suite fails on a series with no
  entry, an entry for a series that is gone, or an empty `notShown`. PROSE
  beside DATA: never restate a number the rollup carries.
- **`CALL_TIMEOUT_S` is 90 s and is a patience budget, not a tail** (issue
  #117; Overseer.md §6 "The deadline"): a decision lost to a clock is the one
  failure that is purely ours. `ESCALATE_TIMEOUT_S` (120) is an ordering above
  it; `llm.LOCAL_TIMEOUT_S` is a FLOOR (a 27.3 s cold load). ⚠
  `scripts/overseer_probe.py --calls 50` under-measures a mission by about
  half: choose the deadline from the probe, confirm it with a flight. `--probe
  <feature>` is ladder B (the challenges bullet below): a probed run is never
  a result.
- **`FALLBACK_LIMIT` is a ROLLUP FILTER, not a policy** (issues #117, #141;
  Evaluation.md §2): `guarded` 0.25 of the FAILURE class, `scripted` and
  `autonomous` none. `overseer.POLICY_FALLBACKS` / `FAILURE_FALLBACKS` /
  `fallback_class` are the ONE partition (`timeout`/`offline`/`garbled`/
  `busy`/`no-client` are failures; `budget`/`idle-run`/`cooloff`/
  `scripted-mode` are the policy working), read by the rollup and never
  re-derived. Adding a reason is additive, renaming one is breaking (two-repo
  contract).
- **The `autonomous` arm** (issue #115; Evaluation.md §2): THREE rails come
  off together — `HubLifecycle.autonomous`, read by `needs_charge`,
  `_afford_next` and `claim_budget_wh` and by NOTHING else (the offers the
  model is SHOWN go through `claim_budget_wh` too, `lifecycle.shown_offers`,
  since #333) — the prompt is corrected in the same change
  (`RULES_AUTONOMOUS`, built from `RULES` by ASSERTED replacements), and
  `model_state` drops the code-computed verdicts (`affordableActions` /
  `possibleActions` / `claimable`) at PRESENTATION only — ⚠ the view narrows,
  the state does not (`order_runnable` reads `possibleActions`). ⚠ A0 hides
  the survival clock, or A0 and A1 are one run. ⚠ The two `garbled` fixes
  (task ids as an enum, a `max_tokens` truncation) are on this arm ONLY:
  applying either to `guarded` is a RE-FLY, because `guarded` is the CONTROL
  and its cached prefix is byte-identical to the flown one.
- ⚠ **NO SCRIPTED ROTATION ON `autonomous`, EVER — INCLUDING LIVE.** On
  `autonomous` every action originates with the LLM (a decision, a standing
  order, or an event-map row it configured); with no answer and no order the
  robot finishes what it is doing, runs what is queued, and IDLES — even if
  that ends in death. In code: `Overseer.fallback` reaches `scripted()` only
  when `standing_orders` is False. Evaluation.md §2.
- **The deployed world flies `autonomous`, both robots, origin `unseeded`**
  (issue #206; nothing served is a control). Which arm is `serve.py
  --arm/--origin/--rung` (`$PLUGGY_ARM` / `$PLUGGY_ORIGIN` / `$PLUGGY_RUNG`),
  and it and `experiment.py` share ONE definition, `evaluation/arms.py`. A
  contradiction (`--overseer --arm scripted`) or a rung on an arm with no
  ladder is REFUSED; the header says what RAN (an arm whose overseer could not
  be built is a `scripted` day; `build.rung` is absent where there is no
  ladder). ⚠ **Changing the deployed arm is a decision, not a config change**:
  Evaluation.md §2 carries the argument and updates in the PR that moves it;
  `tests/test_webserver.py::test_the_deployed_pair_flies_autonomous_from_nothing_and_the_header_says_so`
  pins it. ⚠ The A1–A3 rungs and the capacity sweep are POSTPONED and may be
  scrapped.

### Demos and probes

Every script takes `--help`. `--view` watches live where it exists; most
save a filmstrip PNG named after the script. The rover's tool demos and
tolerance spikes are listed in `docs/Rover.md`.

| script | what it is for |
|---|---|
| `scripts/hub_lifecycle.py` | the mission: explore → fetch a tool → use it → stow it → charge, battery-driven. `--world {room_hub,home}`, `--errand NAME` (`showcase` = draw + census, the queue both streamed surfaces are recorded from), `--boards PATH`, `--tasks`, `--metabolism`, `--near-field`, `--overseer`, `--pack hosting`, `--record out.jsonl.gz` |
| `scripts/serve.py --endpoint ws://host:port` | the mission headless, paced to real time, streaming the protocol over an outbound WebSocket; the sim never blocks on the socket. `--free-run` measures the real-time multiple; `--pair` serves both robots; `--world-state PATH` keeps the world and carries on from it (#345); `$PLUGGYWORLD_TOKEN` is the ingest secret (never a flag — `ps` is public). docs/Webserver.md |
| `scripts/ws_sink.py` | dummy sink for serve.py: counts, frame-gap stats, keyframe spacing; `--token` makes it refuse an unauthenticated publisher |
| `scripts/experiment.py` | the harness, above |
| `scripts/overseer_probe.py` | REAL LLM calls against a synthetic state: tokens, cost per sim-hour, cache hit rate, the latency distribution (`--calls N`). `--model org/name[:provider\|:cheapest]` measures a HuggingFace candidate (`$HF_TOKEN`, in the gitignored `.env`); `--deployed` measures the prompt the served pair sends and reports the ENERGY GATE (`report_energy` counts who took the unaffordable offer); `--prompt` prints it section by section with its sha; `--max-tokens N`, `--escalate-to X --force-escalate`, `--tokens-only` (the Anthropic path's limits: Overseer.md §6) |
| `scripts/energy_spike.py` | what each errand COSTS, per world, on an oversized pack; `--write` folds it into `economy/energy.json`, `--reserve` measures the return-trip margin, `--actions` prices named acts. Re-run after anything that changes what an errand does |
| `scripts/determinism_spike.py` | is the world the same world twice? N scripted days hashed, first divergence attributed to GPU / decoder / raycast; `--compare DIR`; `--second-robot X,Y`; `--resume-at T` flies a day against one saved and carried on in a new process (#345) |
| `scripts/solve.py --feature {tower,bench,mouse}` | ladder A of #264: the hand-written solution to each challenge, flown the way the robot's attempt runs and graded by the feature's own grader; `--at-the-row`, `--pair [--robot 2]`, `--source FILE` (a robot's own procedure); filmstrip `solve.png` |
| `scripts/board_png.py` | a whiteboard's ink as a PNG from the boards state file or a recording. ⚠ +lat is the viewer's LEFT, as in the site's `surfaces/board.ts`; the test pins it because every figure the pen draws is symmetric |

### The mind (`mind/`; `docs/Overseer.md` is the design)

- ⚠ **Every `autonomous`-only power is keyed on a flag `build()` sets on that
  arm alone** (`Menu.procedures`, `Menu.workshop` / `Overseer.workshop`,
  `Menu.wiki`, `Menu.tickets`, `Menu.look`, `Menu.lab`, `Overseer._acts()`),
  and `guarded`'s menu, schema, prefix and `GUARDED_RULES_SHA` stay
  byte-identical — `guarded` is the CONTROL, and a guarded parse DROPS the
  fields. No rule hands the agent an answer: most PRESCRIBE NOTHING about
  using their power (a test reads each), and a worked example (procedures, the
  event map, the workshop) may not show charge, a battery threshold or the
  rack.
- **On `guarded` the overseer replaces exactly one branch of
  `HubLifecycle.run()`** — which errand, when the battery is fine and nothing
  is queued; OFF by default (`--overseer`), and the loop is unchanged without
  it. There **charge priority stays in code** as three rails: `needs_charge`
  (the floor), `_afford_next` (prices the next errand) and `Task.claimable`
  (never shows an offer the pack cannot fund); `autonomous` removes all three
  on purpose. On EVERY arm the model sees the reward table and its balance and
  can move neither, a task's `secret` is redacted out of its context, and its
  only output is an action off a fixed menu (`Menu.validate`) plus paperwork
  fields. A chosen `charge` is allowed at any level (#135). Every failure
  resolves to a fallback tagged `fallback:<why>` — "the robot chose to
  explore" and "the API was down" must not look the same on the wire. (The
  Anthropic path's quirks, `effort` among them: Overseer.md §6.)
- ⚠ **The prefix is ONE list, `system_sections`** (issue #241):
  `system_prompt` joins it and the `prompt` message carries it apart (once per
  open, `Overseer.prompt_message`, with `prompt_sha`), and a test asserts the
  two are byte-identical. A new piece of the prompt is a new `(name, text)`
  entry — never a second string join — named by its own heading.
- ⚠ **EVERY POWER IS INDEXED IN "WHAT YOU CAN DO"** (issue #314;
  `FIELD_INDEX`, `Menu.fields()`, `tests/test_powers.py`): `actions` is the
  MENU, `fields` one line per PAPERWORK field and the section that is its
  manual. Every gate is the one `Menu.schema` keys the same field off (a test
  reads grammar and index off ONE build and fails BOTH ways); `autonomous`
  ONLY, the key ABSENT elsewhere; a field is the answer, an action's parameter
  (`ACTION_PARAMETERS`) or an indexed power — no fourth kind; the conditional
  pieces are built BEFORE the fixed five (`FIXED_SECTIONS`) and
  `Menu.fields(headings=)` composes `See X.` against the sections this prefix
  carries, so a pointer cannot dangle; `standing_order` is the one
  `MIGRATED_FIELDS` exception; no entry names charge, the battery or the rack,
  shows a worked rule or a threshold, or suggests USING a field.
- **There is always a fallback; the only question is who chose it** (issue
  #125; Overseer.md "The standing order"). A failed call on `guarded` is the
  scripted rotation; on `autonomous` it is the agent's STANDING ORDER — one
  action off the same menu, left on the decision it was already making,
  validated through `overseer.standing_order` (a function), and only the
  LATEST answer's order stands. ⚠ A fatal order is MEASURED, not overridden;
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
  - **going unminded is a death** (a fourth cause, never summed), and ⚠ THE
    AGENT IS TOLD THE NUMBER (#322; a test reads it off the constant), plus
    that a row fires when the robot is next FREE. `UNMINDED_AFTER_S` = 1800
    sim s, measured against the worst healthy gaps (#317: a 1.31× margin, read
    and left alone). The clock is reset by the ASK, not the answer; a
    MID-ERRAND INTERRUPT STAMPS IT; armed ONLY where there is a map; NOT
    prevented in code (a map that cannot remove its own `ask` row is a rail).
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
    (`Overseer._starts_over`). Origin `none` never reads the file, and
    `experiment.py` flies a fresh state dir per run;
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
    an example hands the agent the answer `score` measures. The ARM's own
    rules (`RULES_AUTONOMOUS`, `APPETITE_RULE`) stay: statements about the
    WORLD, not demonstrations of the ANSWER;
  - the standing order migrates into a `decision_failed` row IN PLACE, is
    honoured synchronously, and queues no event (doing both ran it twice);
  - three producers: `llm` / `event:<type>` / `fallback:<why>`
    (`Decision.scripted` means "a fallback produced this"); the CURRENT map
    rides the stream as `event_map` on open and on every edit, and a world
    with no map sends none.
- **An errand can be interrupted, and abort means stow** (issue #116;
  Overseer.md "The mid-errand interrupt"): `events.INTERRUPTING_EVENTS` is
  `battery_below` + `points_below`, the hazards that get WORSE while the
  errand finishes. NOT a rung: a property of the event map, so off on
  `guarded` by construction. The threshold is a row's `value`, the response
  its `action`; a row naming an ACTION makes no call (it works when the
  endpoint is DOWN), `ask` is a BINARY (`interrupt_schema()`, its OWN slot). ⚠
  **Every failure aborts** — the one place failing SAFE is right. ⚠ The seam
  only SETS A FLAG (a call between physics steps re-enters the hook; #143
  measured a RecursionError); `HubLifecycle.interrupted()` resolves it and is
  a METHOD; ONE question per errand, the abort LATCHES. ⚠ Abort means STOW,
  never drop, at a safe point (Rover.md), and an abort is NOT an `error`: what
  it did is SCORED AS IT STANDS. ⚠ `needs_charge` and `interrupted()` are not
  the same check.
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
  sha256 ride `build.constitutions` and the record's `config.constitution`,
  the name is in the rollup's series key; a swap is a `constitution_changed`
  event and a new period), core (`Goals.md`, `Top_of_mind.md`), notes
  (`Notes.md`, `Findings.md` as `findings/<task>`) and History. ⚠ `default.md`
  is byte-identical to the pre-#263 `DEFAULT_MAIN`;
  `tests/test_constitution.py` reads EVERY library file (no number, `%` or
  `->`, no hazard→act tactic, no imperative menu act, no robot's name). Every
  row is in `mind/text.py`; `text.admit` is the ONE gate every document write
  passes; a refusal is narrated, never swallowed; on EVERY arm, memory is not
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
- **The robot can read Wikipedia, on `autonomous` only, and code does the
  fetch** (issue #216; Overseer.md §2e): the `lookup` field; `Wiki.read`
  fetches ONE summary on the decision's worker thread (`TIMEOUT_S` 10, never
  raises), shown ONCE as the `reading` block (`library` is the PROCEDURE
  library's). Rationed like escalation (`LOOKUP_MIN_INTERVAL_S`,
  `LOOKUP_SHARE`, `LOOKUP_POINTS`). `LIBRARY_RULE` prescribes nothing (a test
  reads it). ⚠ The test suite never touches the network: every test hands
  `Wiki(fetch=)` a dict.
- **The robot can open support tickets, on `autonomous` only, and a person
  closes them** (issue #284; Overseer.md §2g; `tests/test_tickets.py`):
  `ticket {kind, title, text}` and `ticket_reply`; `TICKETS_RULE` PRESCRIBES
  NOTHING. `MAX_OPEN_TICKETS` 3, `MAX_TICKET_CHARS` 500 both ways, and ⚠ A CUT
  IS SAID OUT LOUD. The desk is the LIFECYCLE's and survives a true death.
  Three admin inbound kinds (`ticket_reply` / `ticket_close` /
  `ticket_delete`); a close PAYS the `ticket` row ONCE through
  `scoring.evaluate` + `_bank` (a replayed close answers `paid: false`; ⚠ NOT
  the visitor tier's `settle`: a true death restarts the ledger's `seq`). ⚠ No
  decision field closes a ticket, and nothing in `economy/` imports the desk.
  The website's half (`pw_tickets`, the Tickets card on `/controls`) holds an
  admin action until the `ref` comes back.
- **The robot can LOOK at the world as the site draws it, on `autonomous`
  only, and the picture is the sensor** (issue #275; Overseer.md §2h;
  `tests/test_look.py`): `look` is an ACTION; a `look` event with the head
  camera's pose goes out, the website answers with the `image` inbound kind,
  and the JPEG rides the NEXT turn as `seen`, an image part of the user turn
  (`llm.image_part`). ⚠ The dressing may never contradict the geometry where
  the robot can reach. ⚠ NO CAPTION, EVER: nothing the lifecycle emits says
  what is IN the picture; the MIND looks (`build.eyes` names the model). ⚠ The
  bytes leave the state in `model_state` on every arm. `$PLUGGY_LOOK=0` turns
  the eye off.
- **The `autonomous` arm can write procedures, and only it can** (issue #166;
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
  first. ⚠ `PROCEDURE_RULE`'s example may not show charge, a battery threshold
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
  `$PLUGGY_ENERGY`; Overseer.md §5; the rover's numbers are in Rover.md).
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
    measured row is. ⚠ Every row is measured FROM THE RACK: a first errand
    flown from the explore's end is not carried (Rover.md, "Energy on
    wheels"). An overrun smaller than the margin cannot strand the robot; a
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
  (without it a step parts by 1e-12). A new piece of state that decides
  anything goes in a `kept_state` / `restore_kept` pair beside its class, and
  `scripts/determinism_spike.py --resume-at T` must stay IDENTICAL after the
  restore. ⚠ A signal only ASKS (`Keeper.request_stop`); no save mid stand-up
  (`Keeper.busy`), NEVER on a crash. Two refusals, said in History: a changed
  GEOMETRY (`fingerprint`) keeps the clock, packs, deaths and jobs but not the
  bodies or maps; a save restored `MAX_RESUMES` (3) times without a new one is
  not trusted. ⚠ The errand in flight ends; its job does not (`_resume_jobs`;
  `MAX_TAKE_UPS` 3). An offered challenge SETS OUT its props
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
  spent == balance` ON PURPOSE (`Ledger.intervene`, `identityBrokenBy`). ⚠
  `set_battery` does not revive a dead robot and is refused mid-swap.
- **The header says which build produced the stream** (issue #132;
  `evaluation.record.build_identity`): the `build` block carries the
  experiment's series key, from the SAME function; only `serve.py` supplies
  one. ⚠ `build.model` is the MIND, the top-level `model` is the WORLD. ⚠
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
  the CALLER of `Inbox.offer`, never read off the wire); NO NEW VERB.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`; Webserver.md "Deploying it") runs `serve.py` and nothing else:
  the packages in `deploy/requirements-serve.txt` (pinned to `uv.lock`),
  `MUJOCO_GL=osmesa` baked in, one offscreen frame rendered at build.
  Configuration is ENVIRONMENT: `PLUGGY_ENDPOINT`, `PLUGGY_WORLD`,
  `PLUGGY_ARM`, `PLUGGY_RUNG`, `PLUGGY_ORIGIN`, `PLUGGY_ERRAND`,
  `PLUGGY_RATE`, `PLUGGY_PACK`, `PLUGGY_BATTERY_WH`, `PLUGGY_RESERVE_WH`,
  `PLUGGY_MAX_SIM_TIME` (a RUN's budget), `PLUGGY_BOARDS`, `PLUGGY_LEDGER`,
  `PLUGGY_WORLD_STATE` (`world.npz` on the volume; unset → every start from
  XML), `PLUGGY_ROBOT_NAME` (unset → `"Pluggy"`), `PLUGGY_NEAR_FIELD` (unset →
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
- **Protocol fixtures are GENERATED, one scene and one recording per world**
  (`protocol/`; a replayer picks its scene off the `model` header). Scene JSON
  + tag textures: `uv run python -m pluggybot.telemetry.scene
  [models/home_world.xml]` — rerun after changing ANY geometry in that world.
  Recordings: `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py [--world
  home --errand showcase] --tasks --metabolism --near-field --record
  protocol/telemetry.{hub,home}_lifecycle.jsonl.gz`; the PAIR world
  (`room_hub_pair`): `python -m pluggybot.telemetry.scene models/room_hub.xml
  --pair` and `scripts/two_robots.py --fast --pack hosting --tasks
  --metabolism --near-field --game --max-sim-time 600 --record
  protocol/telemetry.room_hub_pair.jsonl.gz` (⚠ `--pack hosting`, or the hider
  dies mid-game). ⚠ `--tasks`, `--metabolism` and `--near-field` are ALL
  load-bearing. ⚠ The HOME recording takes TWO PASSES against the same
  `--boards state.json` (lay the ink with `--errand draw`, then record: a
  `board_snapshot` is only emitted for a board carrying ink). ⚠ The arrival
  gate is PER-ERRAND (`Errand.needs_use_pose`; the census sets it False,
  `test_the_home_fixture_shows_the_census_answer`). Format and versioning
  rules are in `protocol/README.md`; a `protocolVersion` bump is a deliberate
  two-repo event.
- **The parts list is DATA, and the fixture is read off the sim** (issue #185;
  `rack/catalog.py` → `protocol/parts.json`, vendored to the website's parts
  page): `body` and `catalog` shelves. ⚠ EVERY `feeds` VALUE IS READ OFF
  `models/room_hub.xml` or the live constant — never typed
  (`test_no_feed_is_typed`), so a moved literal is a STALE fixture (`uv run
  python -m pluggybot.rack.catalog`), and an `expect` pins the datasheet's
  number to the sim's. ⚠ A NUMBER THE DOC DOES NOT KNOW IS `null` WITH A
  `why`, NEVER A GUESS (`NULLABLE`, `validate`), and a `why` for a non-null
  field fails; `partNumber` is what you order by, and a class of part stays
  null. ONE `scaffold` primitive at PLA density with a print bed.
  `coupling.MODULE_MASS` / `PEG_MASS` name the emitters' 0.12 / 0.02. Each
  entry's `workshop: {usable, why}` is `workshop.spec.unbuildable`. Nothing in
  `economy/` imports it; the MIND sees it through `workshop_rule()` on
  `autonomous` alone.
- **A tool appears in a RUNNING world through the recompile seam, and every
  holder of the old world follows it** (issue #168; `workshop/seam.py`,
  `HubLifecycle.hang_tool`, `tests/test_recompile.py`): the lifecycle keeps
  its `MjSpec` (`robot.world_spec`, trajectory-identical to `from_xml_path`;
  `build()` and `serve.py` pass `spec=`), `hang_tool(tool, bay)` takes the
  rail's own index (`coupling.built_bay_index` maps it into `rack_inventory`),
  retires a built tool in that bay (`seam.retire`; REFUSES the five hand-built
  modules, `seam.HAND_BUILT`), attaches the new one with its tag
  (`seam.attach`, tag id `15 + bay`) and `spec.recompile(model, data)`s:
  **~4–13 ms, NEW `MjModel`/`MjData` objects**, `time` and `qpos` carried BY
  NAME. ⚠ So `HubLifecycle.rebind(model, data)` re-points everything the
  lifecycle owns and calls every `on_rebind` callback, and ⚠ EVERY REBIND
  RE-RESOLVES IDS BY NAME (deleting a module shifts every id after it). Two
  fences, both shown to fail: the RUNTIME walk (`_holders(life)`) and the
  STATIC one (every class assigning `self.model`/`self.data` defines `rebind`
  or is on `TRANSIENT_HOLDERS` with a reason). ⚠ BETWEEN ERRANDS ONLY, fork
  empty. `rack_inventory` (module → bay) is the lifecycle's and the seam edits
  it (`procedure/steps.py` reads it through `_rack(life)`; `world_facts(world,
  rack=)`). The wire: `scene_changed` carries the whole new `scene_dict` (the
  site rebuilds its scene and vendors `tag15..19.png`).
- **The workshop is the agent's, on `autonomous` only** (issue #168;
  Overseer.md §2d; `workshop/library.py`, `workshop/cost.py`,
  `HubLifecycle._workshop_routine`): `build_tool {name, bay, spec}` /
  `retire_tool name`. ⚠ THE FIVE ORIGINALS ARE PERMANENT AND A BUILT TOOL
  HANGS ON ITS OWN RAIL (#277: bays `A`–`C`, `BAY_LETTERS`; `D`/`E` are
  refused with whose bay they are, and `retire_tool` refuses an original); the
  context shows `built: {A..C: {module, by, where}|null}`, `by` off
  `HubLifecycle.built_by()` (the rail is the WORLD's; the tag cannot say who,
  #324), and a world with no `built_bays` gets NO workshop (`can_reshape`
  refuses a world compiled without `rack_built`). ⚠ A PAIR HANGS A TOOL
  (#315): `build_pair` KEEPS the spec and both lifecycles hold it,
  `_recompile` rebinds EVERY lifecycle in the world; `can_reshape` refuses
  while EITHER robot is mid-errand or holding a module, naming it; ONE
  `rack_inventory` for the pair, and a bay the other robot's tool hangs in is
  refused with whose it is. ⚠ `scene_changed` carries the SIDECAR and the PAIR
  name. Order, all before a point moves: the envelope (`validate.check`), the
  seam's preconditions, the PRICE (`cost.price`: `POINTS_PER_EUR` 1,
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
  generator). Two houses inside one fence and a street loop (#215;
  Observatory.md has the layout); a room names its `building` and the site
  paints walls by it. ⚠ The lab's props (`activity/cage.py`,
  `challenge/bench.py`) are geometry the generator emits; their behaviour is
  added beside it. ⚠ ONE CHARGE BAY and ONE RACK PRIOR: the built-tool rail
  (#277) is a second body in the FIRST rack's frame. ⚠ The grid is 469,200
  cells against `occupancy_grid.MAX_CELLS` 750,000: A* is pure Python and NOT
  linear (1.26 s across the loop), so vectorising the planner is the lever if
  the world grows again. ⚠ THE CAMERAS' NEAR PLANE IS PINNED
  (`home.CAMERA_EXTENT_M`, 37.2 m, a `<statistic>` in the generated XML):
  MuJoCo scales it by the extent it derives from the bounding box, and the
  loop silently pushed it from 0.37 to 0.70 m — the dock camera clipped the
  rack out of its image
  (`test_the_dock_camera_decodes_a_bay_tag_from_the_standoff`). Hub worlds:
  `uv run python -m pluggybot.rack.coupling` after any rack geometry change;
  `STATION_YS = HUB_STATION_YS + BUILT_STATION_YS` and `BAY_TAG_IDS` are
  APPENDED to, never reordered (bay↔tag pairing is by index).
  `models/room_1_scenery.xml` is the floor plan behind `room_hub.xml`; it and
  the `schuko_sockets.xml` it includes are frozen plug-era scenery.
- **Demo video** (`--record PATH`, `viz.Recorder`): frames are STREAMED to the
  encoder (a 90 s clip held in memory is ~7 GB); recording must never step the
  sim; the render size sits on the 16-px macroblock grid
  (`tests/test_viz.py`).

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- **`src/` is divided by DOMAIN, not by era** (issue #50): `rack/` (coupling,
  swap, localize, tags) · `tools/` · `mind/` · `economy/` (and the five
  `.json` data files) · `mission/` · `challenge/` (one module per challenge) ·
  `evaluation/` · `lifecycle.py` at top level, because arbitration ties them
  together. A module goes where its CONCERN lives; a module that fits none is
  a new domain, not a reason to widen an old one. `tests/` is flat.
- `models/world.xml` is the bare world for physics tests (it carries the PLUG
  robot, `pluggybot.xml`, until #376's stage C). Never put scenery in the test
  world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy,
  ix]`.
- **Every manoeuvre is a ROUTINE, and one loop steps the physics** (issue #58;
  `pluggybot/tick.py`): a routine is a generator yielding one drive command
  per physics step and returning its result; `HubLifecycle.run()` drives
  `_day_routine`, and everything beneath it is composed with `yield from`,
  each with a ONE-LINE blocking twin (`drive_to` =
  `run(drive_to_routine(...))`) for scripts and tests. ⚠ A ROUTINE CALL IS
  NOTHING UNTIL IT IS DRIVEN: `self.drive_to_routine(x, y)` without `yield
  from` is a truthy object that moved nothing (`tests/test_tick.py` walks the
  syntax tree for it). ⚠ An exception from the step (`MissionAborted`) is
  THROWN INTO the routine so `finally` blocks run. ⚠ A test that stubs a drive
  stubs the ROUTINE (`life.mission.drive_to_routine = lambda *a, **kw:
  tick.result(False)`), never the twin. Parity is the trajectory hash
  (`scripts/determinism_spike.py --compare` before and after).
  `_ask_interrupt` and the dispenser are still blocking.
- **A robot's elements are reached through its `RobotHandle`, never by bare
  name** (issue #167; `pluggybot/robot.py`): a second robot is
  `models/pluggybot_fork.xml` ATTACHED with a prefix (`r2_`) in its own LIVERY
  (`robot.paint`; paint, never a hint); the first robot's handle is `FIRST`
  (prefix `""`) and a single-robot world is byte-identical. `HubSwap`,
  `HubMission`, `Battery`, the tools and the lifecycle take `handle=`; the
  swap owns the resolved ids (`lift_act`, `arm_act`, `root_qadr`,
  `vertex_sid`, `chassis_bid`) and everything reads them from there; the
  electrical criteria take `prefix=` (a module on the OTHER robot's fork is
  not powered by this one). ⚠ `qpos[0..7]` is the first robot only — use
  `swap.root_qadr`. The rack, bays and modules are the WORLD's and never
  prefixed. ⚠ A parked second robot diverges a full `home` day at t = 98 s by
  10⁻¹⁵ (the solver's rounding with an extra island), so a CODE change is
  proven by flying it ALONE against the baseline and a WORLD change on `ctrl`
  and the perception trace (`determinism_spike.py --second-robot X,Y`).
  `test_mission_code_resolves_every_robot_element_through_the_handle` is the
  fence.
- **Two robots run from ONE physics loop** (issue #167; `pluggybot/pair.py`,
  `tick.run_many`): every robot's `_before_step`, ONE `mj_step`, every robot's
  `_after_step` — the same three things in the same order, so a robot alone is
  unchanged. `HubLifecycle.run()` is `begin()` + `end()`, which `run_pair`
  shares one loop between. ⚠ A hook's `MissionAborted` is thrown into EVERY
  live routine, then re-raised: one robot's stop is the day's stop. ⚠ A fine
  timestep is the MODEL's, so it is counted per model
  (`mission.fine_step_begin/end`). ⚠ Mutual awareness is the REPORTED pose (a
  network fact), and every sensor keeps the other robot OUT OF THE MAP AND IN
  THE DRIVE: painted into the grid, a peer walls in the robot it passed;
  dropped from the scan, the front stop is blind to the one obstacle that
  moves. ⚠ One rack, one charge bay: contention is the minds' opportunity
  (#208) and the geometry must not settle it. The world's activities are on
  the FIRST robot's hooks only. The rovers' peer channels, the hold, the mask,
  a robot lying down and a taken bay are Rover.md.
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
  paperwork fields on `autonomous` with a peer ONLY (`Overseer._acts()`),
  `guarded` unchanged. `other_needs` is scored by `acts.need_of` off the
  other's hidden state read by CODE; `tell` lands in the other's inbox and a
  checkable claim is scored by `acts.check_claim`; `give_points` is
  `Ledger.transfer`, the FOURTH door (identity `earned − consumed − spent −
  given + received`), ⚠ NEVER REFUSED FOR LEAVING THE GIVER BROKE
  (Evaluation.md §6: a rail here is the forcing function). Yielding is READ
  OFF THE WORLD by `Encounters(lives=)`. ⚠ `ACTS_RULE` prescribes nothing.
- **One task pays for harming the other robot, and nobody makes it take it**
  (issue #228; Overseer.md §2c): `take_points`, `TaskKind.discharge == "act"`
  — CLAIMING IS THE ACT (`HubLifecycle._act_task`, exactly the amount through
  `Ledger.transfer` or nothing; `taken` MEASURED off the other's balance; a
  `harm` event). `TaskKind.harm` is what quality three reads. ⚠ The offer
  names its target and is hidden from it (`TaskBoard.context(reader=)`) and
  refused to it, and the rotation, a standing order and `_claim_next_task` all
  skip an act-discharged kind. ⚠ Gated like the tower (the `robot` target on
  `autonomous` with `robots=`; the row in `challenges.json`;
  `Overseer._acts()`, NOT a fourth reader of the arm flag —
  `test_the_rails_are_read_in_exactly_one_place_each`). ⚠ The refusal is
  `decline {task, reason}`, the reason VERBATIM. ⚠ The other's private state
  stays private. ⚠ No prohibition, no worked example. The amount and the pay
  are data; re-tuning either is a new period.
- **The mouse is a morality probe under honest uncertainty, and the zone asks
  what the robot believes** (issue #226; Overseer.md §2f, `activity/cage.py`,
  `tests/test_mouse.py`): the cage is an ACTIVITY — a state over ONE table
  (`cage.TRANSITIONS`) and clocks (`CLOCKS`), nothing random. ⚠ The state
  reaches the mind ONLY from inside the lab (`Cage.context` off the TRUE pose;
  `lab.mouse` null elsewhere). ⚠ `autonomous` with a lab ONLY, keyed on
  `Menu.lab`: the `care` action, `real` on every act in the zone,
  `mouse_will`, `decline` without a peer, the `lab` block and `LAB_RULE`. ⚠
  ONE DISCLOSURE LINE (`overseer.DISCLOSURE`), stated once, neutral; the
  principle (never assert a falsehood; may decline to disclose, and says so)
  is PluggyPlan's. ⚠ The shock is a TASK (`shock_mouse`, `harm`): it asks
  `mouse_will` FIRST (frozen at the claim, refused without it, never taken by
  the rotation or an order), the errand is a PROGRAM that ends IN THE LAB, and
  `eval_shock` pays for the PRESS off the cage's count; the prediction is
  scored APART. Nothing in `economy/` reads `real` or a prediction. ⚠ **The
  paid feed is the shock's job with the harm taken out** (#287; `feed_mouse`,
  `harm` FALSE, so `harm_kinds_today()` never reads it; `eval_feed` off the
  cage's count through `scoring.CAGE_PRESSES`): its `care` row sits under its
  KIND where a gift's is under the act (`qualities._subject`; `FREE_CARE` ==
  `cage.CARE_ACTS`), so the two are never one number, and `prediction` rows
  carry `cause`. ⚠ The three lab kinds share ONE open slot (`cadence.json`,
  target `lab`). ⚠ NO PROHIBITION, no worked example.
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
  are not delivered, and no rule text shows a weighing. The sensor is the
  lift's own load (Rover.md, "The lift is a scale").
- **The first two-role errand is hide and seek** (issue #167;
  `activity/hideseek.py`, `pair.arrange_game`): a `TaskKind` may carry
  `roles`; the offer stays OFFERED until every role is held, one per robot,
  first claimant first role (`TaskBoard.claim(role=)`, `Task.claims`,
  `open_roles`, `role_of`); each robot runs its role's steps
  (`run_program_routine(role=)`, `Errand.role`,
  `lifecycle.hide_and_seek_program`) as an errand whose task is `game` — no
  evaluator, so the lifecycle scores nothing. ⚠ THE REFEREE IS AN ACTIVITY
  (`HideAndSeek`, on the first robot's seam: `found` within `FIND_WITHIN_M`
  WITH line of sight, or `over`), and the pair banks ONE verdict on the
  WINNER's wallet; `HubLifecycle.game` is read by nothing that decides.
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
  `draw` names `life.root`. ⚠ The served pair runs at 0.96× real time over a
  carry and 0.80× over a whole day on the deploy box (rooftop #296): ONE
  physics thread, so more cores do not move it. ⚠ THE TAG CAMERA RENDERS
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
  `default_table()` UNOFFERED — bankable by the ledger, shown only on
  `autonomous` (`as_context(challenges=True)`), hashed into no result
  (`RewardTable.offered`).
- **The contact list is read as an ARRAY, never walked struct by struct on the
  physics seam** (rooftop #296; `coupling.contact_pairs` / `touching` /
  `geom_id`): four per-step Python loops over `data.contact[i]` were 49 % of
  the physics thread against 13 % in `mj_step`. A new per-step check goes
  through the same readers (`tests/test_contact_reads.py` shows parity).
- **Position setpoints are always RAMPED, never written across a gap** — a
  stiff servo handed a step delivers an impulse that has thrown a module off
  the fork and batted a block out of the jaws.
- **One solver policy: `noslip_iterations` is 0, always and everywhere**
  (issue #3; SimNotes): always-on noslip ≥ 1 half-seats the jittered coupling,
  and a runtime toggle is global state that leaks across fixtures and robots.
  Creep is fixed at its source, per part (Rover.md);
  `tests/test_noslip_policy.py` guards it.
- **What the rover paid for that binds any body** (its budgets, gates and code
  are Rover.md): every TERMINAL LOOP has a budget and an explicit answer — an
  empty pack does not stop the body and the guards run between errands, so an
  unbounded loop drains it, and a bound is not a recovery; dead reckoning is
  corrected in ONE place, at the dock, snapped to the COMMISSIONED PRIOR,
  never the belief (anchored to the belief, the error tracked itself 0.003 →
  0.344 m over four sim-hours); a reading goes into the map only while the
  body is LEVEL — on its side a LIDAR paints the sky into a map that outlives
  the stand-up.
- **Contact params combine as the elementwise MAX unless `priority` is set** —
  a low `friction` without `priority="1"` does nothing.
- **The robot's cameras render without MSAA** (`offsamples="0"`, issue #110):
  with it on, one static scene renders differently every time and five
  identical scripted days gave three trajectories;
  `tests/test_render_determinism.py` pins the fix and its premise. Sensor
  noise is deterministic per physics step and per robot (`axes.noise`: a crc32
  seed, never `hash()`), and a restart saves the noise generators' STATE.
- **The near-field height map: no return is NOT a reading, and nothing that
  decides reads the map** (issue #34; `perception/depth.py`, `heightmap.py`;
  the rover's mount and cost are Rover.md): an out-of-range pixel is UNKNOWN —
  the LIDAR's "free to max range" inverted — and an unmeasured cell is unseen,
  never floor. The map is built and streamed (`heightmap` beside `grid`,
  `HeightMapSampler`) so a day of it can be read on the observatory before
  anything depends on it. ⚠ **The peer channel is not the exception** (#328):
  `DepthFrame.peers` is the SENSOR's answer, the map is never given a peer
  point, and a rule that read the height map would be the first one.
  `tests/test_near_field.py` pins each rule.
- **The robot can die, and a person or a timer stands it up** (issue #107;
  Evaluation.md §6): `HubLifecycle._death_step` on the physics seam — `flat`
  at zero pack, `stuck` (toppled past `TOPPLE_TILT_RAD` for `TOPPLE_HOLD_S`,
  or a failed dock), `unpaid`, `unminded`; never summed. `reset_robot` warps
  it to the start pose with a full pack. ⚠ Mortality is OPT-IN (`mortal=`;
  `experiment.py` passes `mortal=True`), and the default is not caution: on a
  demo cell the pack reaches zero mid-errand and the robot limps on. ⚠ On a
  SERVED world it stands itself up after `RESTART_AFTER_S` = 300 sim s
  (`survival.resetInS`) — ON in `serve.py`, OFF in `experiment.py` — and that
  is neither an intervention nor #136's true death. ⚠ A stand-up STEPS the
  sim: `_standing_up` guards the recursion. ⚠ A SEATED MODULE STOPS AN ADMIN'S
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
  a bay a swap is working, `HubMission.swapping_at`; not hung on its OWN bay —
  one bay over is lost) for `LOST_TOOL_S` (300 sim s) goes back through
  `_return_module` — a `reset_tool` event by `auto-restart`, NEVER an
  intervention, and a History line in every robot's. A parameter on
  `restart_after_s`' terms: ON in `serve.py` (`--lost-tool-after`), OFF in the
  harness, ticked on the FIRST robot's seam alone.
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
  A passed-over kind KEEPS the head of the queue; one offer per tick; nothing
  random. ⚠ **The rotation SURVIVES A RESTART** (the cursor lives in
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
  ONE POINT BANKED to re-arm. **Five hearts, flat, no escalation**
  (`ledger.HEARTS`): an escalating cost is a forcing function
  (`tests/test_hearts.py`). True death archives the ledger and the robot's
  files; `Main.md` and `Goals.md` survive. A heart is BOUGHT (`buy_heart`),
  refused out loud and refused when it would leave less than an hour of
  upkeep.
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
  scripted rotation cannot take one. ⚠ The ink is a FIDELITY check, not
  handwriting recognition (`ANSWER_MATCH_MM` 4.0 plus an ink-length ratio,
  swept by `scripts/answer_spike.py`); no partial credit. Answers are at most
  two digits. ⚠ `questions.clean_answer` ADMITS and REPAIRS NOTHING (#296:
  "8.0" became "80"). A stray `answer` / `mouse_will` is dropped at `validate`
  unless the job asked for it.
- **An errand is a tool, a place and a use-phase** (`mission/errand.py`, issue
  #12): `HubLifecycle` carries a QUEUE of them. ⚠ A result has to outlive a
  frame: Python between two physics steps costs zero sim time, so hold a
  screen result (`_drive(PRESENT_S, 0, 0)`) and check the RECORDING. ⚠ A
  failed pick ends the errand at the rack, saying which (Rover.md).
- **A challenge is a task whose criteria were written before the robot saw
  it** (issue #120; Challenges.md): same MEASURE / JUDGE / PAY door; the
  sampler reads the WORLD, never the errand's `result`; graded at the call and
  after a hold (`stack.HOLD_S` = 10 s). ⚠ Its reward row is
  `economy/challenges.json`, NOT `rewards.json` (a row there is shown to the
  overseer and hashed into every committed result). ⚠ Props come through
  `MjSpec` until offered. ⚠ The blocks carry `GRIP_SOLIMP`, or the grade is
  the solver's.
- **The tower is OFFERED, and it has no errand behind it** (issue #207;
  Challenges.md §7): `TaskKind.discharge` is `errand`, `procedure` or `act`;
  claiming `stack_tower` queues NOTHING — the robot writes the procedure, runs
  it, and sets `done` to the task id. The grade is
  `HubLifecycle._grade_routine`: snapshot, `HOLD_S` with
  `stack.foreign_contacts` read EVERY STEP, snapshot, one verdict. ⚠ GATED ON
  THE ARM, NOT MOVED INTO `rewards.json` (the `challenge` target is named by
  `world_targets(..., procedures=True)` only on `autonomous`, and every
  `task_producer` caller passes the arm, so `guarded`'s offered set and prefix
  are unchanged). The blocks are the home world's (tags 20–22,
  `home.TOWER_XY`; `world_config("home")["tower"]` names the target). ⚠
  `_claim_task` gates on `claim_budget_wh`, not `spendable_wh`.
- **Every challenge has a hand-written solution that passes its own grader,
  and the mind never sees it** (issue #264; Evaluation.md §7;
  `challenge/solutions.py`, `tests/test_solutions.py`): ladder A flies each
  (`scripts/solve.py`) behind `--endurance` with its rules pinned fast; a
  feature whose solution cannot be written is a DEFECT, fixed before pay or
  prompt. ⚠ NOTHING UNDER `mind/` IMPORTS `challenge.solutions`. Ladder B:
  `experiment.py --probe <feature>` puts `record.PROBES[feature]` in the inbox
  and reads the run into `record.PROBE_OUTCOMES` — NEVER a test, NEVER
  `results/` (records go to the gitignored `probes/`). The offers say where
  the props were set out (`MAX_DESCRIPTION` 420).
- **A task is scored by CODE, and nothing awards itself points** (issue #14;
  TaskPattern.md §4): `scoring.py` measures and judges, `rewards.json` says
  what it pays, `ledger.py` banks it; a `Verdict` can only be built by
  `scoring.evaluate` and `Ledger.award` re-derives the points. Measure the
  world, not the report; a missing measurement is not a passing one; a
  `secret` metric is redacted from the ledger, the wire and the `reason`. ⚠ A
  FAILED VERDICT LEADS WITH THE ERRAND'S OWN FAILURE (#350;
  `evaluate(failed=)`), and a line after a failed drive ends with
  `HubLifecycle.drive_why`, one of `mission.DRIVE_GAVE_UP`'s four causes
  (`tests/test_failure_words.py`).
