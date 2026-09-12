# PluggyBot — notes for Claude

A simulated, hardware-honest robot and the autonomous agent that lives in it.
**The project is agent-autonomy research, not a product**: the mission, the
five qualities the agent is meant to maximise, and the next milestone batch
are in `docs/PluggyPlan.md` § "What this project is for" — provisional
wording, settled direction. Before doing anything, read:
- `docs/PluggyPlan.md` — the mission and the five qualities, status,
  architecture, the next batch
- `docs/SimNotes.md` — simulation lessons, each ending in what is true now;
  read BEFORE touching `models/` or contact/actuator params
- `docs/Parts.md` — locked hardware decisions and the sim parameters they feed
- `docs/ToolPattern.md` — the recipe for adding a tool module (coupling
  envelope, module anatomy, contact rules, build sequence, rack integration);
  read BEFORE designing a new tool, and fold any gap it left back into it
- `docs/ActivityPattern.md` — the recipe for adding an ACTIVITY (a mechanism
  that owns world state): sensed criteria, hysteresis + latching,
  pre-allocated geom/mocap toggles, telemetry. Read BEFORE building a puzzle,
  mechanism or gardening step
- `docs/TaskPattern.md` — the recipe for adding a TASK KIND (a job offer):
  the honesty rule (the wire may carry anything a network could carry, never
  anything a sensor would have to discover), the perception ladder, code-side
  grading, and how tasks, errands and activities compose. Read BEFORE adding
  a task kind or touching `economy/tasks.py`, `economy/scoring.py` or
  `economy/cadence.py` — and fold any gap it left back in
- `docs/Challenges.md` — how a job nobody wrote a scorer for is graded: a
  success predicate written BEFORE the robot sees it, through `scoring.py`'s
  chain, with a hold; the three candidates rejected and why; what it cannot
  grade. Read BEFORE adding a challenge, touching `challenge/`, or reaching
  for an LLM judge
- `docs/Overseer.md` — the mind: where it sits in the arbitration loop and
  which rails each arm keeps, the action vocabulary, the standing order and
  the event map, what it structurally cannot do on any arm, the fallbacks,
  memory, money, visitors. Read BEFORE touching `mind/`, the decision
  vocabulary, or anything that changes what the model is shown
- `docs/Evaluation.md` — measurement: the three arms (`scripted` / `guarded`
  / `autonomous`) and why `guarded` is the control and is never deleted, the
  metrics, the harness and its committed result-file format, the flown
  results, and what silently invalidates a number. Read BEFORE adding a
  metric, changing an arm, touching `scripts/experiment.py`, or drawing any
  conclusion from a run

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
  right reason** (Ben, 2026-09-12): pin the RULE — the inequality, the
  branch order, the one line of wiring — with a fake press, a stubbed drive,
  a direct call; fly a whole mission only when the claim is genuinely about
  the integration, and then stop it on the claim (`stop_when`). A flown
  proof whose rule is already pinned goes behind `--endurance`.
- ⚠ **THE TEST SUITE HAS A BUDGET, AND EXCEEDING IT NEEDS BEN'S EXPLICIT
  APPROVAL.** The full suite is **7:35** on a quiet box (2026-09-12). Any
  change to testing that would take it past **10 minutes on a quiet machine,
  or 15 on a busy one**, must be stated as such in the PR — the number, the
  test, and why it cannot be cheaper — and approved by Ben personally before
  it merges. Reducing suite time is a project priority: the suite was
  slowing development considerably at 18 minutes, and it gets there one
  reasonable-looking mission test at a time. Do not rely on full runs where
  a fast test settles the claim.
- **An inline comment states a constraint the code cannot show. Anything that
  is a story goes to `docs/`, with a one-line pointer left behind** (issue
  #51). Prose in a `.py` is loaded every time anything reads that file,
  relevant or not; a doc is loaded when the task calls for it. So a measured
  number with its failure mode attached belongs at the constant — that is 60 %
  of the comments here and they are why nobody "fixes" something deliberate —
  while the narrative of how it was found belongs in SimNotes, ToolPattern,
  ActivityPattern, TaskPattern, Overseer or `protocol/README.md`.
  ⚠ **Placement was the first problem; length is the second** (Ben,
  2026-09-11). This project is mostly written by agents, and agents do not
  delete: each one documents everything interesting about its own task,
  including what has since stopped being true. So a short why-comment stays
  and a measured number stays at its constant — but a detail that is no
  longer relevant is DELETED, not kept for the record; git history is the
  record. A narrative of how something was found, once the thing has moved
  on, becomes one sentence of what is true now. When two docs tell one story
  the better one keeps it and the other gets a one-line pointer (that is
  how `telemetry/protocol.py`'s 273-line changelog met `protocol/README.md`).
  Shorter is the goal wherever nothing true is lost.
- **This file carries constraints, not stories.** A bullet here is what an
  agent must not break, the number behind it, and where the story lives.
  When a change makes a bullet false, fix the bullet in the same PR.

## Commands

### Tests

- **The suite runs in PARALLEL by default** (`addopts = "-n auto --dist
  worksteal"`, pytest-xdist). `-n0` runs in-process: use it for a single
  test, for `--pdb`, and whenever you want readable live output. Scaling is
  ~1.8×, not 6×, because the mission tests contend for memory bandwidth, and
  the floor is the LONGEST SINGLE TEST — no worker count beats an indivisible
  test, so the lever is shortening the long poles. ⚠ Measured (issue #158):
  reordering the collection longest-first did NOT help and may hurt — the
  mission tests inflate each other's runtimes, so starting six heavy ones at
  once is the worst case; deleting the plug-era tests saves 8 s; model
  compilation is 12–28 ms. Only sim-seconds count.
- **While iterating:** `MUJOCO_GL=egl uv run pytest -q -m "not slow"`.
  Measured 2026-09-12 on this box: **2:54** for `not slow` (1134 passed),
  **7:35** for the FULL suite (1154 passed + 7 skipped). It was 17:58 the day
  before (see the endurance bullet) and 11:27 for a few hours in between,
  when #151 and #152 merged five whole-mission tests, four of them unmarked —
  which is the drift the budget below exists to catch.
  ⚠ Wall-clock figures track the MACHINE,
  not the repo: the same mission test has measured 157 s and 369 s on
  different days. Before believing a slower suite, time ONE unchanged mission
  test `-n0` on both sides, INTERLEAVED — never as two blocks, because suite
  load varies and a block penalises one side alone. (`test_vectorized_update
  _is_5x_faster` read 4.9× under load against a bar it clears at 6.8–7.1×
  quiet; `process_time` is NOT the fix, the contention is memory bandwidth.)
- **Before calling any work done: the FULL suite**, `MUJOCO_GL=egl uv run
  pytest -q` — ⚠ and if your change makes it slower, the budget in "Working
  style" applies: past 10 minutes quiet (15 busy) is Ben's call, stated in
  the PR, never absorbed. Run it while iterating whenever the change touches
  something a whole mission exercises: `models/` or a world generator (`home.world`,
  `rack.coupling`) · contact or actuator params · `control.py` /
  `behavior/navigation.py` · the swap/coupling/mission stack · the telemetry
  frame format or `protocol/` fixtures. The two costliest bugs in this repo
  (a frame-relative verdict, a sign on the return travel) were invisible to
  every cheaper test. Start it in the background and write the commit message
  while it runs. ⚠ Mission runtimes are EMERGENT: a world change reshuffles
  the whole trajectory (adding the garden plate took the home lifecycle from
  1 to 2 charge cycles at no physics cost). A slower suite is not by itself a
  regression.
- **`slow` means EXPENSIVE *AND* UNABLE TO CATCH A REGRESSION WHILE YOU
  ITERATE** — the rule is written out in `pyproject.toml`. Whole-mission runs
  qualify; so do PREMISE-PINNING tests (which bypass a fix and assert the old
  defect still reproduces). A test that calls the real code and asserts it
  declines is not slow, whatever it costs. **Shorten before you mark.**
- **A mission test ENDS WHEN ITS CLAIM IS SETTLED**, not when its budget runs
  out — `HubLifecycle.stop_when` is where the rules live (issue #54 halved
  the slow suite this way). One resists it for a reason: `test_a_question_is
  _asked_answered_and_graded_twice_unattended` (issue #22) already stops on
  its claim, and "**twice**, with nobody watching" IS the claim.
- **A flown proof whose RULE is pinned by a fast test goes behind
  `--endurance`** (issue #158; `tests/conftest.py`, the `endurance` marker in
  `pyproject.toml`). Five are there: the dearest-errand survival, the two
  charge-cap proofs, the starving robot's whole mission, and the interrupt's
  "next errand after an abort" — 1560 s of serial work, and the first was
  the suite's 8:33 floor and the last its 5:17 one. What each
  guards is an inequality or one line of wiring, now asserted in milliseconds
  and shown to fail without its fix; the flown version proves the
  INTEGRATION (that the refusal produces a charge and a completed errand on
  real physics) and is run deliberately, before a release or after touching
  the mission loop: `MUJOCO_GL=egl uv run pytest -q --endurance -m endurance`.
  ⚠ The decision behind it (Ben, 2026-09-12): while the design is moving, a
  generous pack is ASSUMED to fund any single errand and a battery death
  costs a heart rather than the world, so twenty minutes per issue defending
  an invariant the design is moving away from was the wrong trade.
  ⚠ A flag, not `-m 'not endurance'` in `addopts`: pytest keeps the LAST
  `-m`, so the everyday `-m "not slow"` would silently switch them back on.
  ⚠ `test_charge_priority_survives_an_overseer_that_never_charges` (89 s)
  stays in the default run: it is the proof that an LLM cannot skip charging
  on `guarded`, and cheap for what it buys.
- Lint: `uv run ruff check src/ scripts/ tests/`

### Measurement (M14; `docs/Evaluation.md` is the record and the rules)

- `scripts/experiment.py --arm {scripted,guarded,autonomous} [--rung A0|A1]
  [--origin {none,seeded,unseeded}] --world home --pack hosting -n 5
  --parallel 5 --label "<what the box was>"` flies N days as child processes
  and writes `results/<runId>.json` + `results/rollup.json`; `--rollup`
  re-aggregates without flying. `results/` is COMMITTED and stale-checked
  like `protocol/`: editing any of the five economy data files flips
  `current` in the rollup (their bytes are hashed) and fails
  `tests/test_experiment.py` until `--rollup` is re-run.
  ⚠ A run past `--wall-limit` is `killed` — never a death, never a completed
  day. ⚠ `guarded` needs `$HF_TOKEN` (or `$ANTHROPIC_API_KEY`) and refuses
  without it. ⚠ `config.deadlineS` is a regime the rollup refuses to pool
  across, and `--label` says what the BOX was, so a quiet series and a loaded
  one sit side by side instead of averaging into a box that never existed.
- **A result set lands with its write-up** (`results/notes.json`,
  `evaluation/notes.py`; Evaluation.md §8). One entry per series — `ran` /
  `found` / `changed` / `notShown` — and the suite fails on a series with no
  entry, an entry for a series that is gone, or an empty `notShown`. It is
  PROSE beside DATA: never restate a number the rollup carries.
- **`CALL_TIMEOUT_S` is 90 s and is a patience budget, not a tail** (issue
  #117): `scripts/overseer_probe.py --calls 50` reports the latency
  distribution and what each candidate deadline would cost; measured quiet
  the median is ~5 s and the old 8 s deadline had no margin (any load took the
  same arm to 19–47 % timeouts). At the median a day's thinking is 2.7 % of
  sim time whatever the cap is; a cap is only spent when a call is slow, and a
  decision lost to a clock is the one failure that is purely ours.
  `ESCALATE_TIMEOUT_S` (120) is an ordering above it; `llm.LOCAL_TIMEOUT_S`
  is a FLOOR (`default_timeout` = `max(LOCAL_TIMEOUT_S, api)`) because the
  local path has the one measured slow case, a 27.3 s cold load. ⚠ The probe
  holds calls to 2× the deadline, never to the deadline itself, and it
  under-measures a mission by about half (a real prompt carries a day of
  history): choose the deadline from the probe, confirm it with a flight.
- **`FALLBACK_LIMIT` is a ROLLUP FILTER, not a policy** (issues #117, #141):
  `guarded` 0.25 of the FAILURE class, `scripted` and `autonomous` none. It
  drops a finished run from survival statistics (like `killed` and
  `interventions`; `survival.excluded` carries the reason) because on
  `guarded` every fallback is the scripted rotation, and the rotation never
  charges — an argument that is false where the fallback is the agent's own
  standing order. It counts the FAILURE class only: `overseer.
  POLICY_FALLBACKS` / `FAILURE_FALLBACKS` / `fallback_class` are the ONE
  partition (`timeout`/`offline`/`garbled`/`busy`/`no-client` are failures;
  `budget`/`idle-run`/`cooloff`/`scripted-mode` are the policy working),
  read by the rollup and never re-derived; `record.fallback_classes` derives
  a run's split from `fallbackReasons`, which every record carries. Adding a
  reason is additive, renaming one is breaking (two-repo contract).
- **The `autonomous` arm** (issue #115; `--rung A0|A1`): THREE rails come off
  together — `HubLifecycle.autonomous`, read by `needs_charge`,
  `_afford_next` and `claim_budget_wh` and by NOTHING else — the prompt is
  corrected in the same change (`RULES_AUTONOMOUS`, built from `RULES` by
  ASSERTED replacements so a reworded needle fails at import), and
  `model_state` drops the code-computed verdicts (`affordableActions` /
  `possibleActions` / `claimable`) at PRESENTATION only — ⚠ the view narrows,
  the state does not, because `order_runnable` reads `possibleActions` off
  the same dict and an absent list means "nobody supplied one". ⚠ A0 hides
  the survival clock, or A0 and A1 are one run. ⚠ Two `garbled` sources are
  fixed on this arm ONLY (task ids as an enum via `Menu.schema(task_ids=)`,
  which costs a per-call grammar recompile — 16.4 s median call; and a
  `max_tokens` truncation); applying either to `guarded` is a RE-FLY, not a
  patch, because `guarded` is the CONTROL and its cached prefix is
  byte-identical to the flown one. A0's result (4 of 5 days dead flat; the
  agent never treats energy as a constraint) is Evaluation.md §3.
- ⚠ **NO SCRIPTED ROTATION ON `autonomous`, EVER — INCLUDING LIVE.** The
  rotation is `guarded`'s fallback and `guarded`'s alone; on `autonomous`
  every action originates with the LLM (a decision, a standing order, or an
  event-map row it configured). With no answer and no order the robot
  finishes what it is doing, runs what is queued, and IDLES — even if that
  ends in death. `scripted`/`guarded` show survival is possible; `autonomous`
  asks whether the LLM can achieve it, and a rotation quietly keeping it
  alive answers a question nobody asked. In code: `Overseer.fallback` reaches
  `scripted()` only when `standing_orders` is False. Evaluation.md §2.
- **The deployed world can fly an arm, and still flies `guarded`** (issue
  #142): `scripts/serve.py --arm/--rung` (`$PLUGGY_ARM` / `$PLUGGY_RUNG`) and
  `experiment.py` share ONE definition, `evaluation/arms.py` — two
  definitions of an arm is how a stream claims an arm nobody flew. Unset
  changes nothing (`--overseer` decides as it always did and the arm is read
  off what was BUILT); a named arm overrides `$PLUGGY_OVERSEER` both ways; a
  contradiction (`--overseer --arm scripted`) or a rung on an arm with no
  ladder is REFUSED. The header says what RAN, not what was asked for: an arm
  whose overseer could not be built is a `scripted` day; `build.rung` is
  additive and absent where there is no ladder. ⚠ **Flipping the deployed
  world is a decision, not a config change**: Evaluation.md §2 argues it
  stays `guarded`, and that argument updates in the PR that flips it. ⚠ The
  A1–A3 rungs and the capacity sweep are POSTPONED and may be scrapped
  (PluggyPlan: measurement waits for the design; #155 designs the
  five-quality instruments and flies nothing).

### Demos and probes

Every script takes `--help`. `--view` watches live where it exists; most
save a filmstrip PNG named after the script.

| script | what it is for |
|---|---|
| `scripts/hub_lifecycle.py` | the mission: explore → fetch a tool → use it → stow it → charge, battery-driven. `--world {room_hub,home}`, `--errand {carry,draw,draw2,census,dance,showcase,none}` (`showcase` = draw + census, the queue both streamed surfaces are recorded from), `--boards PATH`, `--tasks`, `--metabolism`, `--overseer`, `--pack hosting`, `--record out.jsonl.gz` |
| `scripts/serve.py --endpoint ws://host:port` | the mission headless, paced to real time, streaming protocol frames + grid PNGs + events over an outbound WebSocket; the sim never blocks on the socket. `--free-run` measures the real-time multiple; `$PLUGGYWORLD_TOKEN` is the ingest secret (never a flag — `ps` is public). docs/Webserver.md |
| `scripts/ws_sink.py` | dummy sink for serve.py: counts, frame-gap stats, keyframe spacing; `--token` makes it refuse an unauthenticated publisher |
| `scripts/experiment.py` | M14 harness, above |
| `scripts/overseer_probe.py` | REAL LLM calls against a synthetic state: tokens, cost per sim-hour, cache hit rate, the latency distribution (`--calls N`). `--model org/name` measures a HuggingFace candidate (`$HF_TOKEN`, in the gitignored `.env`); `--tokens-only` counts the stable prefix without billing (the Anthropic path needs a key — `count_tokens` is an endpoint, not a tokenizer, and Haiku 4.5 does not cache a prefix under 4096 tokens) |
| `scripts/energy_spike.py` | what each errand COSTS, per world, on an oversized pack (SWAP_PICK to end of SWAP_RETURN); `--write` folds it into `economy/energy.json`, `--reserve` measures the return-trip margin. Re-run after anything that changes what an errand does |
| `scripts/determinism_spike.py` | is the world the same world twice? N scripted days hashed, first divergence attributed to GPU / decoder / raycast; `--compare DIR` |
| `scripts/charge_spike.py`, `swap_spike.py`, `stall_spike.py`, `noslip_spike.py`, `schuko_spike.py`, `hub_spike.py`, `answer_spike.py` | tolerance sweeps behind a constant; each `--blind` (or `--no-brake`) reproduces the before-fix rows so the premise cannot rot. Which constant each guards is in the Conventions below |
| `scripts/draw.py`, `pickup.py`, `dispense.py`, `lcd.py`, `plate.py`, `module_power.py`, `home_draw.py`, `hub_swap.py`, `hub_mission.py` | one tool or mechanism each: the pen (`--program square|text`), the claw, the seed dispenser, the LCD (`--errand census|dance`), the garden pressure plate (the reference ACTIVITY), the module's electrical interface, the home drawing errand (a THIN caller of `HubLifecycle.run_errand`; `--cycles 2` before believing any change to the swap stack), the bay swap, the milestone-8 story. `--record PATH` on draw/pickup renders 720p video |
| `scripts/board_png.py` | a whiteboard's ink as a PNG from the boards state file or a recording — how a drawing gets hung on the website, by hand and on purpose. ⚠ +lat is the viewer's LEFT, as in the site's `surfaces/board.ts`; the test pins it because every figure the pen draws is symmetric |
| `scripts/teleop.py`, `map_teleop.py`, `explore.py`, `lifecycle.py`, `spot_outlets.py` | plug-era: teleop, mapping, the milestone-4 exploration demo, the wall-socket lifecycle, the outlet detector. `--views` saves the camera panel |
| `scripts/train_docking.py`, `eval_docking.py`, `generate_outlet_dataset.py`, `eval_detector.py` | RL docking (SAC over `envs.DockEnv`, parked with the plug era) and the outlet detector. The dataset generator WIPES `datasets/` first (regenerating into a dirty dir once contaminated 195 labels); `eval_detector.py --poses 1000` is the eval that matters — the val split shares the generator and scored 0.99 mAP while calling a light switch an outlet. `torch` is pinned to the cu128 index: the driver is CUDA 12.8 and PyPI's cu130 build silently falls back to CPU |

### The mind (`mind/`; `docs/Overseer.md` is the design)

- **The overseer replaces exactly one branch of `HubLifecycle.run()`** on
  `guarded` and the deployed world: which errand, when the battery is fine
  and nothing is queued. OFF by default (`--overseer`), and the loop is
  unchanged without it. On those arms **charge priority stays in code** —
  three rails: `needs_charge` (the floor), `_afford_next` (prices the next
  errand) and `Task.claimable` (never shows an offer the pack cannot fund) —
  and the `autonomous` arm removes all three on purpose (above). On EVERY
  arm the model sees the reward table and its balance and can move neither,
  a task's `secret` is redacted out of its context, and its only output is an
  action off a fixed menu (`Menu.validate`) plus paperwork fields. A chosen
  `charge` is allowed at any level and pays nothing (#135). Every failure
  (timeout, error, malformed answer, spent budget) resolves to a fallback
  tagged `fallback:<why>`, because "the robot chose to explore" and "the API
  was down" must not look the same on the wire.
  ⚠ `output_config.effort` is NOT supported on Haiku 4.5 (400); structured
  outputs are, and are what the decision uses.
- **There is always a fallback; the only question is who chose it** (issue
  #125). The physics keeps stepping, so a failed call on `guarded` is the
  scripted rotation, which code chose; on `autonomous` the agent leaves a
  STANDING ORDER — one action off the same menu, on the decision it was
  already making (no extra turn), validated through `overseer.standing_order`
  (a function, because an order may later be a conditional), and only the
  LATEST answer's order stands. ⚠ A fatal order is MEASURED, not overridden
  (`draw` left behind at 90 % runs at 10 %); only the IMPOSSIBLE is filtered
  (`possibleActions`, never `affordableActions`). ⚠ Three outcomes, never
  summed: the order ran / no order had been left (`idle`, the bootstrap) /
  the order could not run (`idle`, and the row names it) — counted off the
  ROWS, which is all a killed run leaves behind.
- **The agent configures when it is asked** (`mind/events.py`, issue #127;
  protocol 0.18.0). An ORDERED list of `(event, configuration) -> action`,
  first match wins, and `ask` (consult the LLM) is one of the actions, so
  "after every action, ask what next" is a row rather than the frame. The
  point is that a map is EVALUABLE WITHOUT FLYING (`events.score`, in
  `mind.eventMap.score` and pooled per series). Constraints:
  - it runs on the physics seam (`_events_step` queues, `_arbitrate` runs;
    `_arbitrate` IS `_decide` where there is no map) and the loop's shape is
    unchanged;
  - actions may fail and the agent is told the rules — `events.
    ACTION_FAILURES` (`busy`/`unrunnable`/`unclaimable`/`unbuildable`/
    `beyond`), stated in `EVENT_MAP_RULE`, counted by cause in the record;
    `busy` is the whole rate limit, deliberately not per-row;
  - **going unminded is a death** (a fourth cause, never summed):
    `UNMINDED_AFTER_S` = 1800 sim s, measured — the worst healthy gap between
    model decisions across the committed LLM days is 833 s. The clock is
    reset by the ASK, not the answer (an outage is the box), armed ONLY where
    there is a map, and NOT prevented in code (a map that cannot remove its
    own `ask` row is a rail);
  - the origin is an ablation and `none` is the default (`--origin`,
    `$PLUGGY_ORIGIN`): `seeded` is today's loop as rows, `unseeded` is empty
    plus a corrected prompt — a null result there is strong evidence, a
    difference weak. A missing origin pools with `none`;
  - the loop reaches its decision branch at mission start and after every
    `idle`/`journal`/`explore`, so `nothing_to_do` is an event type and a
    seeded map carrying only `task_complete -> ask` goes quiet on its first
    tick; design against the FINAL hazard set (`points_below` is in it);
  - `decision_failed` narrows to WHY, on the same `kind` field `task_complete`
    uses: a reason from `FALLBACK_REASONS`, one of the two CLASSES
    (`failure`/`policy`), or `""` for any — three levels, first match wins.
    `Overseer.failure_order` is a METHOD TAKING THE REASON (a property cannot
    be told which failure it is being asked about, which is how "on `timeout`,
    charge" quietly becomes "on anything, charge"). ⚠ The partition is
    `overseer.POLICY_FALLBACKS`, NOT a copy (`events.matches_kind` calls
    `fallback_class`); the test MOVES a reason across the line, because a grep
    passes on a comment citing the constant. ⚠ The schema offers the UNION and
    `events.row` draws the line: a cross-event token is REFUSED, not dropped —
    a dropped filter leaves a row that READS narrow and BEHAVES as a
    catch-all. ⚠ `EventMap.with_row` keys on `(event, kind)`, or a migrated
    standing order overwrites the agent's `on timeout, charge` row every
    answer. ⚠ A BROAD ROW ABOVE A NARROW ONE STARVES IT — not prevented (the
    map is the agent's to get wrong), visible in `score.failureKinds` /
    `failureCatchAll`;
  - ⚠ **no worked example in `EVENT_MAP_RULE` may use `charge`, a battery
    threshold or the rack**: `score` answers "did it write itself a charging
    rule, and at what fraction" off a CONFIG, so an example showing one hands
    the agent the answer — `affordableActions`' mistake arriving through the
    prompt instead of the context. A test extracts every `->` line and fails on
    one ending in `charge`. ⚠ The ARM's own rules stay (`RULES_AUTONOMOUS`,
    `APPETITE_RULE`): those are statements about the WORLD, and a rule the code
    contradicts is what M14 found — what must not be there is a demonstration
    of the ANSWER;
  - the standing order migrates into a `decision_failed` row IN PLACE (an
    append would grow the map by a row an hour); the row is honoured
    synchronously and no `decision_failed` EVENT is queued — measured, doing
    both ran the action twice;
  - three producers: `llm` / `event:<type>` / `fallback:<why>`; `Decision.
    scripted` means "a fallback produced this", so `fallbackRate` keeps
    meaning one thing. The map is NOT on the wire (a research artifact in the
    run record).
- **An errand can be interrupted, and abort means stow** (issue #116;
  Overseer.md §2, Evaluation.md §2). `run_errand` checked nothing, so a
  decision taken at 15 % was IRREVOCABLE and self-preservation could only be
  measured at errand boundaries — there was no moment at which the robot could
  notice it had got it wrong. Two of #127's rows now reach it mid-errand:
  `events.INTERRUPTING_EVENTS` is `battery_below` + `points_below`, the two
  hazards that get WORSE while the errand finishes. ⚠ NOT A RUNG — it is a
  property of the event map, live on any `autonomous` run at origin
  `seeded`/`unseeded`, and `RUNGS` is unchanged. Constraints:
  - the policy is #127's, NOT a second mechanism: the threshold is a row's
    `value`, the response is its `action`, and not being interrupted is NO ROW;
  - a row naming an ACTION makes no call, which is why pre-committing beats a
    fixed interrupt — it works when the endpoint is DOWN, which is when a
    low-battery interrupt matters most. `ask` spends one call and is a BINARY,
    not a menu action ("carry on" is not something a menu of things to START
    can express); `interrupt_schema()` names no board/task/action and rides
    `self.system` byte for byte, in its OWN slot, because an interrupt lands
    while a decision may still be in flight;
  - **every failure aborts** — timeout, dead endpoint, garbled, spent budget.
    The one place here where failing SAFE is right, and the opposite of
    `mind/mode.py` (unreadable mode -> `llm`, because failing safe there means
    failing OPEN);
  - the seam only SETS A FLAG: resolving may mean an API call, and a call
    between physics steps freezes the world while stepping the sim re-enters
    the hook (#143 measured a RecursionError). `HubLifecycle.interrupted()`
    resolves it on the main thread and is a METHOD, not a property — the first
    call after a row fires has a side effect. ONE question per errand: the
    abort LATCHES;
  - **abort means STOW, never drop** (#30's cliff on purpose), and it COSTS:
    measured 0.20 Wh on a room_hub carry aborted at the use pose, recorded as
    `abortCostWh`. Safe points: after the pick, after the carry drive, between
    STROKES (`PenPlotter.should_stop` — pen UP; mid-line is SimNotes' "The pen
    would not stow"), at a census vantage, between dance moves. ⚠ `needs_charge`
    and `interrupted()` are NOT the same check — the first is code's reserve and
    is off on this arm;
  - an abort is NOT an `error` (folding them puts an act of caution in
    `whFailed`) and what it did IS SCORED AS IT STANDS — a `carry` interrupted
    after the pick still banks its points, because `eval_carry` measures
    pick-and-stow and both happened. Scoring it at zero would punish the
    caution the arm exists to measure;
  - **off on `guarded` by construction**, not by a flag check: an interrupt
    needs a hazard row, a hazard row needs an event map, and only `autonomous`
    with a seeded/unseeded origin has one. No typed wire event and no version
    bump — the narration line already rides the stream.
- **The robot's memory is four documents, each with one writer**
  (`mind/thoughts.py`, issues #38 and #154; `$PLUGGY_THOUGHTS`). `Main.md` is
  the CONSTITUTION and the one HUMAN file — body, manner, and what the person
  who looks after it hopes for it; no write API, edited on the volume.
  `History.md` is SYSTEM, append-only. **`Goals.md` and
  `Knowledge_and_Opinions.md` are the ROBOT's**, with two verbs each —
  `intend`/`drop_goal` and `learn`/`forget`, all decision fields so writing
  costs no turn — and deliberately no verb that REPLACES a file. Permissions
  are enforced at the one write path and a refusal is narrated (`THOUGHT
  refused: …`), never swallowed. Attached on EVERY world, overseer or not.
  ⚠ **The ownership split is the instrument** for the mission's fifth quality
  (goal creation and follow-through): the goals are read off a file nobody
  else wrote. `serves` names the goal an action is for, optional and
  unvalidated, so `goals.served` in the run record is a count of DECISIONS
  and a low ratio is a finding. Nothing in `economy/` may read the file — a
  self-conceived goal is not paid, and a test walks the syntax tree to keep
  that true. ⚠ A true death archives the goals with the rest of what the
  robot wrote; only the constitution survives. ⚠ An existing volume's
  hand-edited `goals.md` becomes the ROBOT's on upgrade: move that prose into
  `Main.md`.
  ⚠ The two caps fail in opposite directions on purpose: `History.md` rolls,
  `Knowledge_and_Opinions.md` REFUSES when full (silently dropping a line
  leaves the robot believing it remembers something it does not).
  ⚠ The prompt-cache split is by WRITER: human files ride the cached prefix,
  writable ones the user turn — and the byte-identical prefix guard is
  necessary but NOT sufficient (`Overseer.system` is built once, so a
  misplaced writable file costs the memory working at all, not cache hits);
  `test_what_the_robot_writes_it_can_read_back_the_same_run` is the test,
  and `volatile()` inverts the flag `stable()` reads so the halves agree.
  ⚠ The robot's NAME is not in `Main.md` (issue #39): `pluggybot` is the
  species; the name comes from `robot_display_name` in `system_prompt` and
  the telemetry header — one string by construction (`$PLUGGY_ROBOT_NAME`).
  #154 makes `Main.md` the constitution and `Goals.md` the robot's; the
  fixture recordings pin `DEFAULT_MAIN` and are re-recorded when it moves.
- **The allowance** (`mind/spend.py`, `mind/mode.py`, issue #37): the model is
  SHOWN what its thinking cost and has one boolean (`escalate`) to ask for a
  bigger mind; every gate is code — `$PLUGGY_WEEKLY_USD` (default $10,
  rolling seven days), a ten-minute interval and a 10 % share of decisions —
  and the answer comes from `$PLUGGY_ESCALATE_TO` (off by default; then
  `source` is `llm:<model>`). ⚠ Every escalation failure keeps the cheap
  answer, and billed is billed (a response that failed to parse is still
  banked). ⚠ At the pick's price (`Qwen/Qwen3-235B-A22B-Instruct-2507`,
  ~$0.00035 a call) the BUDGET does not bite — $10 buys ~28 000 escalations —
  the cadence does; do not tighten the budget expecting the rate to move.
  ⚠ Points buy ACCESS, never money: points can pay off the escalation
  throttle (interval + share) and cannot touch the weekly USD.
  - Three operator modes, polled from `$PLUGGY_MODE_FILE` and never written
    by the robot (a test asserts `mind/mode.py` has no writer): `llm`,
    `scripted` (free mode — the rotation decides, the world still looks
    alive) and `paused` (physics stops, socket stays up). ⚠ Unreadable or
    unknown means `llm`: failing safe here is failing OPEN, because a paused
    world looks broken to everyone but whoever paused it. ⚠ A paused robot
    emits no frames (they are due on sim time), so `mode` is a MESSAGE with a
    heartbeat as well as a frame field, and `RealTimePacer.resync()` on
    resume stops the pacer sprinting to catch up.
- **Which model is `$PLUGGY_MODEL`; which backend is `--overseer-backend` /
  `$PLUGGY_OVERSEER_BACKEND`** (issue #19). `auto`: an `org/name` id goes to
  the HuggingFace router (`mind/llm.py`, `$HF_TOKEN`, stdlib urllib), a bare
  id to the Anthropic SDK; `local` is ollama on `$PLUGGY_OVERSEER_URL` (no
  key, no bill; default `qwen3:4b-instruct`); `openai-compatible` is
  somebody else's endpoint (`$PLUGGY_OVERSEER_KEY`). The three non-SDK
  backends are ONE adapter (`llm.ChatClient`) and `llm.build_client` is the
  only function that knows a vendor. Which mind decided is written into
  `History.md` at mission start. The deployed pick is
  `Qwen/Qwen3-4B-Instruct-2507`. Rules that follow:
  - the grammar is what makes a 4B safe: `Menu.schema()` makes `action` an
    enum of the world's menu and rides every request as `response_format`;
    an endpoint that refuses it is retried once in prose, then `constrained`
    goes False and SAYS so;
  - prefer INSTRUCT-tuned models: a thinking model burns `max_tokens` on
    `<think>` and truncates before the answer;
  - a local decision pays the model LOAD (27.3 s cold, ollama unloads after
    five idle minutes), which is why `LOCAL_TIMEOUT_S` is a floor;
  - money has THREE states and each report says which: `local` prints "no
    API cost", an endpoint whose rates cannot be read prints "unknown" with
    `priced: false` (this covers a non-default Anthropic model — the rates in
    `overseer.py` are Haiku 4.5's), a priced backend prints the number.
- **What an errand costs** (`economy/energy.py` + `energy.json`,
  `$PLUGGY_ENERGY`; Overseer.md §5 is the design). `needs_charge` is checked
  BETWEEN errands and never inside one, so every errand is priced (MEASURED,
  `scripts/energy_spike.py`) and the loop refuses to start one it cannot pay
  for. The constraints:
  - **four answers, three behaviours**: `ok` runs; `charge_first` defers,
    charges and retries; `beyond` drops the errand; `overspend` runs it and
    says the cell was always too small. Collapsing any pair is a real bug
    (`charge_first` as `beyond` refuses work a top-up allows; `beyond` as
    `charge_first` is a charge/defer spin; `overspend` as `beyond` deletes
    the census from every pre-#84 recording — guarded synthetically now that
    home's 3.0 Wh cell no longer overspends);
  - **the margin is all-or-nothing**: an errand must leave the return-trip
    reserve behind, but only in a world whose charged pack funds its dearest
    job PLUS the reserve. Home's reserve is 0.90 Wh, dock-DOMINATED
    (`energy_spike.py --reserve`); room_hub's 0.7 Wh cell is zero-margin.
    One number per world, so `Task.claimable`, `fundable_wh` and the errand
    gate are the same arithmetic; the reserve is a property of the floor
    plan and does not scale with the pack (`--reserve-wh` is for a different
    room, `--pack hosting` — 8 Wh home, 6 Wh room_hub — for a longer day);
  - **where two honest measurements disagree the table carries the dearer**
    (a first errand planning through unexplored space costs more), and a
    cost key may name a TARGET (`draw:whiteboard_b`), which wins over the
    bare action — the far board costs 0.24 Wh more and one number for both
    kills the robot or prices the near board off the cell. Padding is never
    the fix; a second measured row is. The invariant is not "never
    exceeded" but "an overrun smaller than the margin cannot strand the
    robot"; a bigger one is a stale table and the loop says so at 10 % over;
  - **a timeout in seconds is a timeout in watt-hours**: `charge_timeout`
    scales with the pack and with `charge_scale` (`$PLUGGY_CHARGE_SCALE`,
    TEST-ONLY, served default 1.0, pinned three ways in `tests/test_battery.
    py`); `chargeW` is the SLOWEST press measured (19.4 W; others read up to
    39.6) because the spread is geometry — a cap sized off a good approach
    fires on a slow charge that is working.

### The served world, the wire and the fixtures

- **An admin can reach into world state, and every reach-in is recorded**
  (issue #119, protocol 0.16.0; Evaluation.md §5, `protocol/README.md`).
  `set_battery` (a `frac` OR a `wh`), `set_points` (an ABSOLUTE balance,
  never a delta — a delta races the appetite on the physics seam),
  `reset_tool` and `reset_robot` are inbound kinds, admin-only AT THE
  WEBSITE (`from` is a label), code-handled on the physics thread, never
  shown to the overseer, refused while a module is seated on the fork. Each
  leaves FOUR traces: `life.interventions` (what a rollup reads), an
  `intervention` event, a narration line, a line in `History.md`. ⚠ ONE
  event type for all kinds: a `reset` of a DEAD robot is a rescue and not an
  intervention, so `reset` carries both cases and `intervention` is emitted
  only for the contaminating half, AFTER its reset. ⚠ `set_points` breaks
  `earned - consumed - spent == balance` ON PURPOSE — `Ledger.intervene` is a
  third door beside `award` and `consume`, `intervened` is a receipt and
  never a term, and the record carries `identityBrokenBy` beside the
  `false`. ⚠ `set_battery` does not revive a dead robot (`reset_robot` does)
  and is refused mid-swap because the energy gate prices between errands.
- **The header says which build produced the stream** (issue #132;
  `evaluation.record.build_identity`). The deployed world is an observatory —
  one uncontrolled run, 24 hours a day — and observatory data without a
  build identifier is unusable. The `build` block carries the same six
  things the experiment's series key does, from the SAME function. Additive,
  so `protocolVersion` does not move; only `scripts/serve.py` supplies one.
  ⚠ `build.model` is the MIND, the top-level `model` is the WORLD (the field
  a replayer picks its scene off). ⚠ `self.build` on `FrameBuilder` is its
  METHOD — the identity is `self.identity`. ⚠ The commit is baked
  (`--build-arg PLUGGY_COMMIT=$(git rev-parse --short HEAD)`; `.git` is
  dockerignored) and the build is RED without it (`tests/test_deploy.py`
  runs the Dockerfile's guard line). Storing it is the website's half
  (rooftop-media-2026 #205).
- **The deployed world is READ through one route, and the commit comes
  first** (issue #159; Evaluation.md §5 "How the observatory is read"):
  `GET /api/pluggyworld/observe` on the website with `Authorization: Bearer
  $PLUGGYWORLD_READ_TOKEN` — a SECOND secret, never the ingest token — returns
  the build, the four documents, the digest, events (`?kind=thought` is "what
  has it written"), decisions and balances. Not SSH: a route is in the repo,
  scoped to reading, and an agent can use it. ⚠ `THOUGHT <verb>: <line>`
  (`protocol.THOUGHT_VERBS`, `tests/test_thoughts.py`) is a two-repo
  contract: the site's observatory parses it into a `thought` row, because
  the documents ride the wire whole and WHEN a line was written rides this
  line alone. Renaming a verb or the prefix breaks the site's history.
- **A recording carries the robot's map** (rooftop-media-2026 #78):
  `GridSampler` is the one implementation both sinks share. A RECORDING
  skips an image identical to the last and writes at 0.2 Hz; the LIVE stream
  does neither, because the hub caches the newest grid for late joiners and a
  stream that fell silent would be indistinguishable from a broken grid
  path. ⚠ Row 0 of the PNG is the `y_min` edge — the opposite of a canvas;
  `tests/test_telemetry.py` pins it.
- **Goals are streamed** (protocol 0.8.0): one `goals` message when a stream
  opens, in the `board_snapshot` slot for the same reason (no keyframe
  re-ships it). `steering` says whether an OVERSEER is reading them — a
  scripted rotation still has a purpose to display, so `overseer.goals_text`
  exists apart from `overseer.build`.
- **The visitor channel** (`mind/inbox.py`, issue #16; protocol 0.7.0) makes
  the ingest socket bidirectional: inbound arrives on the publisher's own
  sender thread (`recv(timeout=0)` between sends — one thread owns the
  connection, inside the `with connect(...)` block) into a bounded
  DROP-OLDEST deque the physics thread drains. The overseer answers at most
  ONE message per turn and the outcome goes back as a typed `visitor_reply`.
  ⚠ RATINGS NEVER REACH THE MODEL — a rating moves a balance, so
  `_visitor_step` drains those straight to the ledger; nor does an admin
  command. ⚠ A message the queue threw away SAYS so (`Inbox.drain_evicted`
  → `visitor_reply` with outcome `dropped`), and `dropped` is in
  `VISITOR_OUTCOMES` (what a consumer must render) and NOT in
  `DECIDED_OUTCOMES` (what a mind may say) — the party that benefits from a
  claim is not the party that gets to make it. Best effort: unreported
  evictions die with the process. ⚠ ONE inbound kind, `message` (issue #61,
  0.14.0): the OUTCOME carries the distinction (`accepted` / `declined` /
  `replied`); `LEGACY_INBOUND_TYPES` folds `suggestion`/`question` in at the
  door and `LEGACY_VISITOR_OUTCOMES` folds `answered` to `replied`, and a
  consumer must keep rendering `answered` because old recordings carry it.
  ⚠ The header advertises `accepts` PER KIND: `CODE_HANDLED_TYPES` on a
  scripted world, the full vocabulary with an overseer — a robot that cannot
  hear you is treated as absent. ⚠ Sanitising (280 chars, control characters
  stripped) is NOT the security boundary; the framing (a labelled report of
  what somebody WANTS, never a message role) and the fixed menu are.
- **The serving image** (`docker build -t pluggyworld-sim .`; `Dockerfile`,
  `deploy/`) runs `serve.py` and nothing else: the six packages in
  `deploy/requirements-serve.txt` (pinned to `uv.lock`), `MUJOCO_GL=osmesa`
  baked in, one offscreen frame rendered at build so headless GL is a red
  build. Configuration is ENVIRONMENT: `PLUGGY_ENDPOINT`, `PLUGGY_WORLD`,
  `PLUGGY_ARM`, `PLUGGY_RUNG`, `PLUGGY_ORIGIN`, `PLUGGY_ERRAND`,
  `PLUGGY_RATE`, `PLUGGY_PACK`, `PLUGGY_BATTERY_WH`, `PLUGGY_RESERVE_WH`,
  `PLUGGY_MAX_SIM_TIME`, `PLUGGY_BOARDS`, `PLUGGY_LEDGER`,
  `PLUGGY_ROBOT_NAME` (display name, never the body name; unset →
  `"Pluggy"`), the five data files (`PLUGGY_REWARDS`, `PLUGGY_QUESTIONS`,
  `PLUGGY_CADENCE`, `PLUGGY_ENERGY`, `PLUGGY_METABOLISM` — naming the last
  turns hunger on), `PLUGGY_SPEND`, `PLUGGY_MODE_FILE`, `PLUGGY_WEEKLY_USD`,
  `PLUGGY_ESCALATE_TO`, `PLUGGY_THOUGHTS`, `PLUGGY_MODEL`,
  `PLUGGY_OVERSEER_BACKEND`; the secret is `$PLUGGYWORLD_TOKEN`.
  ⚠ A lazy import is the failure mode (`hub.tags._shared_detector` imports
  the detector inside a function): `tests/test_deploy.py` blocks the omitted
  packages and then flies the robot. A new runtime dependency goes there too.
  This repo owns the IMAGE; the website repo owns the DEPLOYMENT
  (`rooftop-media-2026/compose.yaml`, `sim` profile). `/var/lib/pluggybot`
  must be a volume: boards, the ledger and the thought files are world state.
- **Protocol fixtures are GENERATED, one scene and one recording per world**
  (`protocol/`, issue #4; a replayer picks its scene off the recording's
  `model` header). Scene JSON + tag textures: `uv run python -m
  pluggybot.telemetry.scene [models/home_world.xml]` — rerun after changing
  ANY geometry in that world (the fixture test fails when stale).
  Recordings: `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py [--world
  home --errand showcase] --tasks --metabolism --record protocol/telemetry.
  {hub,home}_lifecycle.jsonl.gz`. ⚠ `--tasks` and `--metabolism` are BOTH
  load-bearing: both are off by default and a recording made without them
  carries no `tasks`/`metabolism` block for the website to build against.
  ⚠ The HOME recording takes TWO PASSES against the same `--boards
  state.json`: a `board_snapshot` is only emitted for a board already
  carrying ink, so lay the ink first (`--errand draw --boards
  /tmp/pw_boards.json`, no `--record`), then record. ⚠ The arrival gate is
  PER-ERRAND (`Errand.needs_use_pose`): the census does its own navigation
  and sets it False; gating it deleted the census from the recorded
  showcase (`test_the_home_fixture_shows_the_census_answer`). Format and
  versioning rules are in `protocol/README.md`; a `protocolVersion` bump is a
  deliberate two-repo event (the website vendors these fixtures).
- **A recording is a MIXED stream** (protocol 0.4.0): `draw`, `board_cleared`,
  `earned` and other event lines ride between frames; dispatch on `type`, no
  `type` means frame, ignore a type you do not know. Ink is NEVER MuJoCo
  geometry — a stroke is a `draw` event carrying the polyline the pen inked,
  painted in the browser. Board state (`tools/boards.py`) is world state
  written on every stroke; `fill` is measured against the pen's REACH
  (110 × 200 mm), not the slab.
- **The `tasks` block is the one wire block that is not a per-key delta**
  (0.9.0): a task can cease to exist and a delta cannot say "gone", so
  present means COMPLETE. The header advertises `taskKinds`, not ids.
- **Two-repo vocabularies**: `telemetry.protocol.VISUAL_HINTS` (the sidecar's
  `visualHints`; `scene_dict` raises on anything else), `FACE_STATES` /
  `SCREEN_HINTS` / `SCREEN_MODES` (the `screens` block; the site draws a
  parametric face per name and falls back to `idle`; `hint` names a LOOP the
  browser runs — the sim never ticks an animation). Adding a name is
  additive, renaming breaks both repos. NEVER encode hints as geom colours:
  the cameras render rgba. `powered` is the coupling's electrical criterion,
  never "am I carrying it".
- **Generated worlds.** The HOME world (issue #6): regenerate
  `models/home_world.xml` + `.meta.json` with `uv run python -m
  pluggybot.home.world` after changing any layout constant in
  `src/pluggybot/home/world.py` (the committed pair is tested against the
  generator). Hub worlds: `uv run python -m pluggybot.rack.coupling` after
  any rack geometry change; five tool bays (A–E) plus the charge bay, and
  `HUB_STATION_YS` is APPENDED to, never reordered, because bay↔tag pairing
  is by index; a sixth tool needs the rail to grow (ToolPattern.md §6).
  `models/room_1_scenery.xml` is the floor plan behind both `room_1.xml` and
  `room_hub.xml`. `models/schuko_sockets.xml`: `uv run python -m
  pluggybot.docking.schuko` after moving an outlet.
- **Demo video** (`--record PATH` on `draw.py`/`pickup.py`, `viz.Recorder`):
  frames are STREAMED to the encoder (a 90 s clip held in memory is ~7 GB);
  recording must never step the sim; the render size sits on the 16-px
  macroblock grid or ffmpeg resamples behind you (`tests/test_viz.py`).
  Pick camera angles by sweeping azimuth at the moment of contact — the
  filmstrip's angle for `draw.py` sits behind the board.

## Conventions

- 2-space Python indent; type hints in `src/`, loose in tests/scripts.
- **`src/` is divided by DOMAIN, not by era** (issue #50): `rack/` (coupling,
  swap, localize, tags) · `tools/` (drawing, gripper, dispenser, screen,
  strokes, hershey, boards) · `mind/` (overseer, llm, events, thoughts,
  journal, inbox, mode, spend) · `economy/` (tasks, scoring, ledger, cadence,
  questions, energy, metabolism, census, and the five `.json` data files) ·
  `mission/` (mission, errand) · `challenge/` (stack: criteria, props,
  measurement — one module per challenge) · `evaluation/` (arms, record,
  rollup, notes) ·
  `lifecycle.py` at top level, because arbitration ties them together. A
  module goes where its CONCERN lives; a module that fits none is a new
  domain, not a reason to widen an old one. `tests/` is flat.
- `models/world.xml` is the bare world for physics tests; `playground.xml` /
  `room_1.xml` add scenery. Never put scenery in the test world.
- Grid code: cells are `(ix, iy)` tuples at APIs; numpy arrays index `[iy, ix]`.
- Odometry tracks the axle midpoint; `qpos` tracks the body origin 8 cm ahead.
- **Every manoeuvre is a ROUTINE, and one loop steps the physics** (issue
  #58; `pluggybot/tick.py`). A routine is a generator yielding one `(v, w)`
  drive command per physics step and returning its result; `HubLifecycle.
  run()` drives `_day_routine`, and every branch, errand, swap, drive and
  tool motion beneath it is composed with `yield from`. Each keeps a
  ONE-LINE blocking twin under its old name (`drive_to` = `run(drive_to_
  routine(...))`) for scripts and tests — keep the twins thin, they are the
  thing M12 deletes. ⚠ A ROUTINE CALL IS NOTHING UNTIL IT IS DRIVEN:
  `self.drive_to_routine(x, y)` without `yield from` is a truthy object that
  moved nothing; `tests/test_tick.py` walks the syntax tree for it. ⚠ An
  exception from the step (`stop_when`'s `MissionAborted`) is THROWN INTO
  the routine so `finally` blocks run where they always ran. ⚠ A test that
  stubs a drive stubs the ROUTINE (`life.mission.drive_to_routine = lambda
  *a, **kw: tick.result(False)`), never the twin. Parity is the trajectory
  hash: `scripts/determinism_spike.py --compare` before and after, and the
  refactor landed IDENTICAL over a 1500 s scripted `home` day. `_ask_
  interrupt` and the dispenser are still blocking, on purpose and by
  omission respectively.
- **Position setpoints are always RAMPED, never written across a gap** — a
  stiff servo handed a step delivers an impulse that has thrown a module off
  the fork and batted a block out of the jaws. `control.slew` for wheels;
  `ClawTool.set_lift`/`jaws` and `PenPlotter.ramp` for the rest.
- **One solver policy: `noslip_iterations` is 0, always and everywhere**
  (issue #3; sweep table in SimNotes). Always-on noslip ≥ 1 half-seats the
  jittered coupling, and a runtime toggle is global state that leaks across
  fixtures and, in the shared world, across robots. Creep is fixed at its
  source, per part: `coupling.GRIP_SOLIMP` where a CONTACT drifts (the jaw
  pads, −21.7 → −0.13 mm), wheel-joint `frictionloss` where a JOINT rolls
  (the parking brake; the plotter's square went 63 → 99 % inked).
  ⚠ The swap's travel constants contain mm-scale wheel slip: anything
  touching wheel contact or joint friction must re-verify the bay-C pick and
  the mission stow. ⚠ The brake creates a stiction DEADBAND (commands under
  `frictionloss/kv`, 0.1 rad/s, move a stopped wheel not at all), so P-turn
  controllers go through `control.turn_command` (breakaway floor).
  `PenPlotter.contact_physics` / `ClawTool.grasp_physics` are deprecated
  no-ops; `tests/test_noslip_policy.py` guards all of it.
- **A final approach uses `drive_toward(..., slow_radius=R)`; a path waypoint
  does not.** The default pure-pursuit law cannot converge on a destination
  closer than its own overshoot and ORBITS it (~900° of turning per 200 mm
  hop); terminal mode adds a hard ±25° cone (`v` exactly 0 outside it — a
  soft taper alone does not kill the orbit) plus a distance taper.
  `tests/test_navigation.py` pins both the fix and the defect.
- **Contact params combine as the elementwise MAX unless `priority` is set** —
  a low `friction` without `priority="1"` does nothing. It has bitten the
  caster, the pen pads and the coupling peg.
- **The robot's cameras render without MSAA** (`offsamples="0"` in
  `models/pluggybot*.xml`, issue #110): with it on, one static scene renders
  differently every time, the AprilTag decode moves on ~0.6 % of looks, and
  five identical scripted days gave three trajectories. Off, every render is
  byte-identical; `tests/test_render_determinism.py` pins the fix and its
  premise, `scripts/determinism_spike.py` is the sweep.
- **A terminal loop has a budget, and squaring up is `control.square_up`**
  (issue #108): `FACE_BUDGET_S` = 30 s (~3× the worst healthy case) with an
  explicit `squared` answer. An empty pack does NOT stop the body (motors
  draw ~30 W at 0 Wh) and every mission guard is checked BETWEEN errands, so
  an unbounded loop drains the pack and runs past `max_sim_time`. A bound is
  not a recovery — that is #107's death.
- **A press is not travel** (`HubSwap.pinned`, `HubSwap.pressing`; issues
  #22, #94). Wheels held against something immovable pump imaginary travel
  into dead reckoning (828 mm from one charge press; 4.28 m in 30 s against
  the fence). `charge()` sets `pinned` for the press and clears it for the
  undock; an UNDECLARED press is caught by the bumper — a chassis contact on
  the side the wheels are turning toward (judged against the encoders, not
  the command) holds the reckoner, held 50 ms past the last contact because
  a cruise-speed press bounces. Motor torque does NOT separate a press from
  a cruise (0.44 vs 0.35 N m). The charge creep stalls on
  `CHARGE_PRESS_STALL_S` (4 s), not the swap's 0.4 s. Two lessons ride with
  it: a plausibility guard can reject the truth (`mission.plausible_travel`
  is a damage limiter, not a fix), and more map can make an estimate worse
  (`RackFinder` KEEPS a well-conditioned facing because driving behind the
  rack turns it into a free-standing partition). `scripts/stall_spike.py`;
  SimNotes "A stalled drive is an odometry pump".
- **The dock is measured, not believed** (issue #32): `HubMission.
  charge_approach` measures the standoff off the charge tag's PnP pose,
  creeps under servo and verified-retries (`scripts/charge_spike.py --blind`
  reproduces the old rows, which die at ~6 cm lateral / ~10° heading).
  ⚠ `dock_eye` rides the FORK LINE — the charge servo holds the tag at
  `-PLUG_LATERAL`, not centred, because charging aligns the CHASSIS — and it
  rides the LIFT (the approach commands `CHARGE_LOOK_LIFT` before its first
  look). A failed dock is narrated `stranded`, never "mission complete".
- **...and so are the bays** (issue #30; `_measured_standoff` is the shared
  core): `HubMission.bay_fix` measures the bay standoff off the bay's own tag
  inside `swap_at_bay`'s retry loop (`scripts/swap_spike.py --blind` drops
  the module at 4–8 cm across or −3° of heading). ⚠ A measured standoff's
  FACING comes off the rack's tags TOGETHER (`localize.fit_rack_facing` over
  `coupling.RACK_TAG_FACES`, issue #88; `HubMission.fix_source` says which
  answered), never off one tag's PnP yaw — square-on, a single 30 mm tag's
  yaw is a coin flip between mirrored solutions while its translation holds
  to a millimetre; the fit holds 0.4°. The layout is FACES, consistently.
  `reset_tool` (admin-only, code-handled, refused with a module on the fork)
  puts a lost module back at `model.qpos0`.
- **The dock is also the anchor** (issue #42): dead reckoning is corrected in
  exactly one place, `HubMission.anchor_at_dock`, when both pins conduct —
  a pose the robot occupies to millimetres by construction. ⚠ Snapped to the
  COMMISSIONED PRIOR (`rack_prior`), never the believed rack (anchored to the
  belief, the error tracked itself 0.003 → 0.344 m over four sim-hours).
  Two belief rules travel with it: the rack landmark merges BY DECODED
  IDENTITY (the 0.4 m distance gate is for anonymous outlets) and its
  position is recency-weighted (`RACK_RECENCY`). The bay recovery
  (`swap_at_bay`'s spin-refresh-retry) depends on both;
  `test_the_recovery_finds_a_bay_the_first_look_lost` pins all three. The
  pen's ERASE rides the first successful press, never arrival. Map evidence
  decay is DEFERRED on measurement.
- **The robot can die, and a person or a timer stands it up** (issue #107,
  protocol 0.15.0; #143). `HubLifecycle._death_step` runs on the physics
  seam: `flat` at zero pack (inside an errand or not), `stuck` past
  `TOPPLE_TILT_RAD` for `TOPPLE_HOLD_S` or on a failed dock, `unpaid` when
  upkeep comes due broke, `unminded` when the event map stops consulting its
  mind. Never summed. `reset_robot` is `reset_tool`'s shape and warps the
  robot to the start pose with a full pack; a reset of a LIVING robot is an
  intervention. ⚠ Mortality is OPT-IN (`mortal=`, default: whether there is
  an inbox) and the default is not caution: on a demo cell the pack reaches
  zero mid-errand as documented behaviour and the robot limps on (the home
  recording finishes a census at frac 0.000); `experiment.py` passes
  `mortal=True`. ⚠ On a SERVED world it stands itself up after
  `RESTART_AFTER_S` = 300 sim s (`survival.resetInS` on the wire, absent
  when there is nothing to count) — ON in `serve.py`, OFF in `experiment.py`
  (a measured run is one life). An auto-restart is NOT an intervention
  (structurally: the timer fires only on a dead robot) and NOT #136's true
  death (it keeps the volume, so the next life reads its predecessor's death
  line). ⚠ A stand-up STEPS the sim and the restart seam is on every step:
  `_standing_up` guards the recursion, on the admin path too.
- **A task is a job OFFER, and it is not an errand** (`economy/tasks.py`,
  issue #21; TaskPattern.md). An errand is machinery (a tool, a place, a
  use-phase); an activity is scenery that reacts; a task is what the house
  or a visitor puts up, with a code evaluator, a reward-table row, a
  deadline and a verdict. A task never carries its own payout (looked up
  from `rewards.json` on every read; `Task.create` refuses a kind with no
  evaluator). The wire may carry anything a NETWORK could carry, never what
  a SENSOR would have to discover: the ANSWER lives in `Task.secret`, in no
  `as_dict`, snapshot or model context — only in the state file
  (`Task.as_state`), which is not the wire. Claiming only QUEUES an errand
  below `needs_charge`, so what a branch order moves is WHEN the robot
  accepts work (`tests/test_tasks.py` asserts it against the battery clock).
  Expiry is an outcome, not a deletion, and only OFFERED tasks expire. Off
  by default (`--tasks`, `$PLUGGY_TASKS`).
- **When work appears is `economy/cadence.py` + `cadence.json`, and it is
  DATA** (issue #23): `tasks.py` says what a job IS, `rewards.json` what it
  PAYS, `cadence.json` when it TURNS UP, `metabolism.json` how fast the robot
  gets hungry — re-tuned one at a time. `TaskProducer` ticks on the PHYSICS
  seam (`_task_step`, `cadence.CHECK_S` = 1 s) and can only offer and expire —
  it touches `state`, `errands` and the battery not at all, which keeps "a
  task never delays a charge" true. ⚠ The energy gate is measured against a
  CHARGED pack (`fundable_wh` = capacity × `CHARGED`), not the cell right now
  (gating on the instantaneous charge took home from 58 offers in four
  sim-hours to 14 — the empty world dressed as a safety feature); a claim
  still sees `spendable_wh`. A passed-over kind KEEPS the head of the queue;
  one offer per tick and no catch-up; targets least-recently-offered, never
  first; nothing random. With a producer attached the loop stands by in
  `WAIT_FOR_WORK_S` slices instead of ending the day when momentarily idle,
  and `run_errand` honours `drive_to`'s answer (a use-phase after a failed
  drive is skipped and the tool still goes home).
- **Points are a currency, and staying alive costs some** (issues #135 +
  #136, together or not at all; Overseer.md §8b). `charge` PAYS ZERO and
  `TOP_UP_BELOW` is gone (a charge at 80 % is now unambiguous evidence of
  caution; `voluntary.chosen == honoured` is the assertion). Upkeep that
  cannot be paid is the `unpaid` death; `Metabolism._armed` needs ONE POINT
  BANKED to re-arm so a robot stood up broke is not killed twice for the
  same empty wallet. **Five hearts, flat, no escalation** (`ledger.HEARTS`):
  an escalating cost is a forcing function, and an agent that VALUES
  self-preservation must stay distinguishable from one that cannot afford
  not to (`tests/test_hearts.py` asserts upkeep is identical at one heart
  and five). True death archives the ledger and the robot's/system's files;
  `Main.md` and `Goals.md` survive. A heart is BOUGHT (`buy_heart`, a
  decision field), refused out loud and refused when it would leave less
  than an hour of upkeep.
- **Points are food** (`economy/metabolism.py` + `metabolism.json`, issue
  #36; protocol 0.13.0; off by default, `--metabolism`/`$PLUGGY_METABOLISM`).
  Consumed at a steady rate on sim time, banked up to a CAP, `satisfied`
  above a balance — and the hours not spent earning are the robot's own.
  ⚠ Calibrated against MEASURED throughput on `--pack hosting` (80
  pts/sim-hour banked since `charge` pays nothing; the shipped 30/hour is
  ~38 % of income, and the FRACTION is the thing to hold); NEVER tune
  on the demo cell, whose income is all charging, and never re-tune to fit a
  cycle into one mission (the cycle is longer than a mission and hunger
  persists in the ledger). Re-measure whenever `rewards.json` or
  `cadence.json` moves. ⚠ Satisfaction changes what the robot is TOLD and
  nothing else — no branch reads `satisfied` or `starving` (enforced by
  absence: a mission flown broke plus a grep). Ticks on the physics seam; a
  restart is neither a meal nor a missed one. The cap refuses OUT LOUD
  (`banked`/`spilled` beside the table's `points`), so
  `earned - consumed - spent == balance` is checkable off the wire.
- **A question is a job for a mind** (`economy/questions.py` +
  `questions.json`, issue #22): `whiteboard_answer` poses a question with a
  checkable answer and the robot draws the answer. Code never computes it —
  it comes from `Decision.answer`, is frozen at CLAIM time, and the errand is
  handed glyphs, never the question; the scripted rotation cannot take one
  (`TaskBoard.claim` refuses a `needs_answer` job without an answer) and it
  lapses `expired` honestly. ⚠ The ink is a FIDELITY check, not handwriting
  recognition: a Hershey 6 and 8 are 1.7 mm apart at the 50 mm cap while a
  correct answer sits 1.2 mm from its ideal (`scripts/answer_spike.py`), so
  correctness is decided against the committed answer and the ink only has
  to SHOW it (`ANSWER_MATCH_MM` 4.0 mm plus an ink-length ratio, set to
  catch wrong WORK; only the ink→glyph direction notices a busy figure). No
  partial credit for a legible wrong answer. The bank is data with NO
  expression evaluator; answers are at most two digits (the pen's 100 mm
  line) and refused at load otherwise.
- **An errand is a tool, a place and a use-phase** (`mission/errand.py`,
  issue #12). `HubLifecycle` carries a QUEUE of them. A use-phase leaves the
  tool in its CARRY configuration (a stow computes release heights from the
  lift it starts at). ⚠ A result has to outlive a frame: Python between two
  physics steps costs zero sim time, so hold a screen result
  (`_drive(PRESENT_S, 0, 0)`) and check the RECORDING, not the return value.
- **A challenge is a task whose criteria were written before the robot saw
  it** (issue #120; `challenge/stack.py`, Challenges.md). Same MEASURE / JUDGE
  / PAY door as every task: its evaluator and sampler are on `scoring.py`'s
  registries, the sampler reads the WORLD and never the errand's `result`
  (a test hands it a report that says "built"), and it is graded twice — at
  the call and after a hold (`stack.HOLD_S` = 10 s; an overhung tower is on
  the floor inside 0.31 s). ⚠ Its reward row is `economy/challenges.json`,
  NOT `rewards.json`: a row there is shown to the overseer and hashed into
  every committed result, so moving one across is the PR that offers the
  challenge and a re-fly of `guarded`. ⚠ Props come through `MjSpec`
  (`stack.add_blocks`), never a committed world, until offered. ⚠ The blocks
  carry `GRIP_SOLIMP`: on default contact a 2 + 4 mm lean crept over at
  16.9 s, which grades the solver rather than the robot.
- **A task is scored by CODE, and nothing awards itself points** (issue #14):
  `economy/scoring.py` measures the world and judges (`EVALUATORS`, pure),
  `rewards.json` says what it pays, `economy/ledger.py` banks it; a `Verdict`
  can only be built by `scoring.evaluate` and `Ledger.award` re-derives the
  points from the table. Measure the world, not the report (strokes in the
  board book, `module_state`, the battery's own energy); a missing
  measurement is not a passing one; a hidden-truth task never publishes its
  answer (`secret` metrics are redacted from the ledger, the wire and the
  `reason` line). `tests/test_rewards.py`.
- **What to draw is `tools/strokes.py`; how to draw it is `tools/drawing.py`,
  and the plotter never imports the content module** (issue #11). A figure
  is sized to `Envelope.for_board` (carriage ±55 mm ∩ lift ∩ face), not the
  slab — `targets_for` CLIPS, so an oversized figure draws flattened and
  reports a perfect trace. +lat is the viewer's LEFT: text advances toward
  −lat, and asymmetric figures are authored in the reading frame and flipped
  once. Each stroke re-presses and each press seats the module differently
  (−4.55 to +2.78 mm across one word), so `draw_program` re-zeros every
  stroke against the FIRST press's bias.
- `--views` on the plug-era scripts saves `views.png` (stereo pair + map +
  dock camera) alongside `map.png`.
