# Task Pattern — how to offer the robot a job

The recipe for adding a *task kind* to PluggyWorld: a job the world (or a
visitor, or eventually the overseer itself) puts up on a board, that the robot
may take on, and that ends in a verdict somebody else computed. Third of the
three pattern docs, and every rule here is one an existing build paid for.

**Validated once by a fresh consumer** (issue #24's last box, folded into
#120): the three-block tower — `docs/Challenges.md`, `challenge/stack.py` —
was built against this doc and found four gaps, each marked ⓘ *found by
building the tower* where it was folded in (§1, §2.2, §3, §4). It is a
CHALLENGE rather than an offered kind, so the offer-side half of the recipe
(§5, §6 steps 2, 4 and 5) is still unvalidated by a second consumer.

Read alongside, not instead of:
- `docs/ToolPattern.md` — the things the robot picks up. An errand's fetch
  and stow half is entirely that doc's territory.
- `docs/ActivityPattern.md` — the things the robot acts on. Its three-layer
  rule (MuJoCo owns contacts, Python owns state machines, the browser owns
  visuals) is assumed everywhere below.
- `docs/Challenges.md` — a task nobody wrote a scorer for: the same grading
  chain, with the criteria written before the robot sees the job.
- `docs/Overseer.md` — the mind that will be choosing among these offers, and
  the hard limits on what it may be shown and what it may move.
- `CLAUDE.md` — the short forms, including the measured energy figures.

---

## 1. What a task is — and what it is not

The repo now has three patterns, and confusing them is the first mistake this
doc exists to prevent:

| pattern | is | owns | in one word |
|---|---|---|---|
| **errand** (`mission/errand.py`) | a tool, a place and a use-phase | nothing durable — it runs and is over | machinery |
| **activity** (`activity/`) | a mechanism watching contacts | discrete world state (a gate latched open) | scenery that reacts |
| **task** (`economy/tasks.py`) | a JOB OFFER: description, target, reward row, deadline, verdict | its own lifecycle on the board | a reason |

**An errand is HOW a task gets done; the task is WHY.** One task resolves to
one errand today, and the split is what lets a visitor ask for something
without knowing that "whiteboard" means "fetch the pen from bay C".
ⓘ *Found by building the tower:* a task may have NO errand behind it. A
challenge's discharge is whatever the robot writes (issues #58, #166), so
the HOW is not machinery this repo ships; what the pattern requires of it is
only that the verdict is reachable from the world afterwards (§4).
**Since #58 that HOW has a shape**: a task may carry a *program* —
validated steps over the verb vocabulary (`procedure/steps.py`) — in
`params["procedure"]`, and `errand_for_task` builds a `programmed_errand`
from it, graded by the kind's own evaluator. A program spans tools and
places (fetch and stow are verbs), so the "one task, one errand" limit in §7
is now only a limit on *offering* such a job, not on running one. The three
compose in one direction:

> a **task** is offered → claiming it queues an **errand** → the errand's
> use-phase acts on the world (and may trip an **activity**) → the evaluator
> measures the *world* — board book, activity flags, module state, battery —
> and produces a **verdict** → the ledger banks what the reward table says
> that verdict is worth.

Nothing on that chain flows backwards: an errand cannot write a verdict, a
verdict cannot write the table, and a task cannot write any of it.

### The lifecycle

`offered → claimed → active → done | failed | expired`, and every transition
happens in exactly one place (`TaskBoard`), which is also the place that
emits the wire event and saves the state file — `Task` itself is frozen, so
there is no way to move a task without the wire and the disk hearing about it.

Rules the states carry, each decided explicitly:

- **Expiry is an outcome, not a deletion.** A lapsed offer stays on the board
  saying `expired`, because "nobody got round to it" is a true and interesting
  fact about a robot's day, and a marker that silently vanishes from the
  website reads as a bug.
- **Only OFFERED tasks expire.** A deadline is how long an *offer* stands,
  never a licence to abandon a claimed job — abandoning mid-errand leaves a
  module on the fork, which is a different and worse event.
- **The board is bounded** (40 held, 6 offered at once), and resolved tasks
  age out oldest-first. An OPEN task is never dropped to make room: dropping a
  job the robot might still do and that job lapsing are different events, and
  only one of them has an honest name on the wire.
- **A restart fails what it interrupts.** A task that was `active` when the
  process died comes back `failed` ("interrupted by a restart"), not `active`
  — the robot that was doing it no longer exists — and not `expired`, which
  would lie in the other direction: the offer *was* taken.

---

## 2. The honesty rule

The rule that the other two patterns do not need, because they never receive
anything over a network:

> **The wire may carry anything a NETWORK could carry. It may not carry
> anything a SENSOR would have to discover.**

A task description is a work order, and real robots receive those over WiFi —
that is not cheating. Sensor data the robot should have to go and get *is*.
Worked through the deliveries the current kinds actually make:

| delivered to the robot | verdict | why |
|---|---|---|
| the description ("Draw the answer to this question on whiteboard_a: 2 + 3") | ✅ | a work order over a network |
| which whiteboard, by id and pose | ✅ | surveyed infrastructure, same class as the charging rack — and the final approach is still sensed (`board_standoff()` / `drive_to_board()` square off against the board, they do not teleport to it) |
| the answer to the question | ✅ | cognition, not perception — supplying it is what the mind is *for* (§2.1) |
| whether the board is already clean | ❌ | the robot owns board state (`tools/boards.py` — written on every stroke, survives restarts); it should know from its own actions, and the drawing errand erases first precisely so it need not be told |
| the pose of a movable object | ❌ | finding it *is* the task — see the perception ladder, §3 |
| the true count in a census | ❌ | hidden ground truth, already correctly redacted (`Task.secret`, `Verdict.secret`) |

The test for a new delivery is the middle column of the first row against the
last: *could a dispatcher who cannot see the room have written this?* A
question, a target id, a deadline, a payout tier — yes. The state of the
world the robot is being sent into — no.

### 2.1 Where the rule is enforced, structurally

The rule is not a convention; it is load-bearing code, and a new kind
inherits all of it for free:

- **`Task.secret` reaches the wire and the model through no path at all.**
  It is absent from `as_dict` (the wire and the site), from `snapshot` (the
  telemetry frame) and from `as_context` (the overseer's view). The ONE
  exception is `as_state`: the state file in `/var/lib/pluggybot` carries it,
  because an offer that came back from a restart with no right answer behind
  it could never be graded. **The file is not the wire** — it lives beside
  `rewards.json`, which also decides what things are worth and is also never
  published.
- **Reason lines say "wrong" without saying what was right.** The census's
  and the answer's evaluators both phrase their verdicts so the streamed
  `reason` — which reaches the site *and* the overseer's context — never
  contains the hidden value. `Verdict.public_metrics()` redacts the `secret`
  metrics the reward row names, for the same audience.
- ⚠ **...AND SO MUST THE ERRAND'S OWN NARRATION.** **`secret` is a property
  of the VALUE, not of the channel you happened to think of.** A hidden value
  has three exits, not one: the ledger, the reason line, and any `_say`
  anywhere in the errand. `_say` writes `life.status`, `telemetry_status()`
  puts that in EVERY frame, and the site renders it under the robot's
  portrait while the overseer reads the same frames — so the census's ground
  truth was once published to precisely the two readers it is hidden from,
  with all three of the other guards in place (issue #75). A comment saying a
  value "never reaches the screen" must name WHICH screen.
- **Answers travel one way.** For `whiteboard_answer`, the answer comes from
  the mind (`Decision.answer`), is frozen into the task at CLAIM time, and
  the errand that draws it is handed glyphs and never told the question.
  Code never computes it — reading it out of the bank would be the sim
  marking its own homework — and `TaskBoard.claim` refuses a `needs_answer`
  job with no answer, so the scripted rotation leaves questions standing and
  they lapse honestly.

### 2.2 The mirror rule: a task never carries its own payout

The honesty rule governs what a task may tell the robot; this one governs
what a task may tell the *ledger*, and it is issue #14's rule arriving from
the offer side. `Task.task` names an evaluator in `economy/scoring.py` and a row
of `economy/rewards.json`; what the job PAYS is looked up from the table on every
read and is never a field anybody can set — a visitor-created task that could
name its own price would be a stranger on the internet moving a balance, and
an LLM-proposed one would be the model paying itself. `Task.create` refuses a
kind whose evaluator does not exist, so the unscoreable task cannot be
constructed, and the `whiteboard_answer` template deliberately quotes **no
price in the sentence**: a description is written once and frozen, while
`reward` is derived on every read, and a price in prose would go stale the
first time the table was re-tuned — with the stale figure being the half a
person reads.

ⓘ *Found by building the tower:* a row for a job the robot cannot yet
attempt does not go in `rewards.json`. That file is shown to the overseer as
the jobs it can take and is hashed into every committed result
(`evaluation/record.py`), so a row there is both a lie about what the robot
can do and a regime change in the rollup. Such rows live in
`economy/challenges.json` (same format, `scoring.challenge_table()`) and
move across in the PR that offers the kind — which re-flies `guarded`.

Related, because a stranger will eventually be on this path
(`TASK_SOURCES = system | visitor | overseer`): a description is untrusted
text on exactly the inbox's terms — capped at 280 chars and cleaned on the
way in. And as `mind/inbox.py` records, **sanitising is not the security
boundary**; the boundary is that nothing a task says can reach the robot's
body except by resolving to an errand off a fixed menu.

---

## 3. The perception ladder for object tasks

Nothing can autonomously find a floor object today, and this is measured, not
pessimism (`ToolPattern.md` §7): the LIDAR plane sits **223 mm** up, the nav
camera is blind to the floor inside **0.48 m**, and the grip point sits
~285 mm ahead of the axle — so grasps run open-loop from a memorised pose.
Object tasks therefore come in tiers, and **a task kind must state which tier
it is written for** (in the kind's comment and the errand's docstring, the
way a tool states its tolerance class):

1. **Tagged object.** The object carries an AprilTag and is found by the same
   `TagDetector` that finds the rack and the bays. Honest and realistic —
   warehouse robots fiducial their totes — and it makes "find" a genuinely
   *failable* verb: a tag out of view is a search, not a lookup. **This is
   the tier to build first**, and building it against this doc is what
   validates the doc (see the header). It is M11's, which is deferred rather
   than dropped (PluggyPlan.md, "The next batch").
2. **Untagged object on a raised surface.** Within the sensor envelope
   (table height clears both the LIDAR plane and the camera's near blind
   zone); needs a small detector, on the outlet detector's template — with
   that project's lesson attached: the val split sharing the training
   generator scored 0.99 mAP while calling a light switch an outlet, so the
   eval that matters is poses the generator never made.
3. **Untagged object on the floor.** Blocked on real perception work — the
   near-field sensor decision (#34) and a robot-centric height map are what
   open it (PluggyPlan.md, "The next batch"). Until then, **a research
   question, not a task**: do not write a kind at this tier and quietly
   deliver the pose over the wire to make it work, which is the honesty
   rule's ❌ row wearing a feature's clothes.

ⓘ *Found by building the tower:* the tier is a property of the ATTEMPT, not
of the grader. The tower's grader reads `xpos` and would score a tower built
at any tier; the module still states tier 1, because the moment the job is
offered the blocks must be findable, and the tags come with the procedure
that finds them.

The ladder is also the reason the rule's second ✅ row is safe: a whiteboard's
pose is tier-0 — *immovable, surveyed infrastructure* — and delivering it is
the same class of fact as delivering the rack's position. The moment the
target can move, its pose climbs the ladder and stops being deliverable.

---

## 4. Grading measures the world, never the report

`economy/scoring.py` splits the end of a task into three pieces, and the split is
the whole design: **MEASURE** (samplers read the finished task off the sim),
**JUDGE** (`EVALUATORS`, pure functions, measurements in → verdict out),
**PAY** (`rewards.json`, data — base + bonus × quality curves). A `Verdict`
can only be built by `scoring.evaluate` (the seal), and `Ledger.award`
re-derives the points from the table before accepting one.

Three rules, each guarded in `tests/test_rewards.py`, each with a worked
example:

- **Measure the world, not the report.** An errand's `use` is arbitrary
  caller code, so what it says about itself is a claim. A drawing is scored
  on the strokes the pen wrote into the BOARD BOOK (which the pen writes and
  the errand cannot); a carry on the coupling's own `module_state`; a charge
  on the battery's own energy. The sampler takes a *before* reading of the
  board too, so an errand that skipped its erase is not scored on ink that
  was already there.
- **A missing measurement is not a passing one.** The census defaulting an
  absent `counted` and `truth` to 0 and 0 would have compared EQUAL and paid
  full marks to a survey that never ran — the exact shape of bug the module
  exists to prevent. Absences stay absences all the way to the evaluator,
  which is what turns them into an honest failure.
- **No partial credit**, decided explicitly rather than omitted: failure pays
  zero, because a consolation payout for showing up is the gradient that
  teaches a robot to attempt the cheapest task it can fail at, over and over.
  Quality curves scale the *bonus* on a success; they cannot buy a failure.

ⓘ *Found by building the tower:* **a success that has to PERSIST is sampled
twice.** `score_errand` reads the world once, when the errand ends, which is
right for ink on a board and wrong for a tower: a stack whose centre of mass
is past its base passes every geometric check at the instant it is let go and
is on the floor 0.3 s later. So the sampler takes the `before` reading at the
CALL and the verdict reading after a HOLD (`stack.HOLD_S`, 10 s) during which
nothing may touch the work, and both must pass. The hold is a criterion of
the challenge, not an implementation detail, and it belongs in its written
criteria.

### 4.1 The worked example: `whiteboard_answer`

The first task kind is the pattern's best teacher because its verdict has two
halves checked against two different things, and keeping them apart *is* the
design:

- **CORRECTNESS** is `wrote == expected`: the commitment the mind froze into
  the task at claim time, against `Task.secret`, which the errand never sees.
  Nothing that happens on the way to the board can influence this half.
- **FIDELITY** is the ink, off the board book, against the glyphs of the
  committed answer. This half stops a caller passing by *reporting* that it
  wrote a 5.

And fidelity is deliberately **not handwriting recognition**, which is a
measured decision, not a shortcut: at the 50 mm cap an answer is written at,
a Hershey 6 and 8 sit **1.7 mm** apart while a correctly drawn answer sits
**1.2 mm** from its own ideal (real pen, `scripts/answer_spike.py`) — so a
grader that classified the ink would fail correct drawings and pass wrong
ones, at random, on exactly the pairs arithmetic produces. The bar
(`ANSWER_MATCH_MM` = 4 mm, plus an ink-length ratio) is set to catch wrong
*work* — the `robot` figure drawn in place of a "5" — not to re-decide
correctness, which the committed answer already decided. When your kind's
verdict tempts you toward perception-grade checking, do the spike first and
let the measurement pick the bar.

---

## 5. An offer must be refusable — the economics

A task system's failure modes are economic before they are mechanical, and
every one below was hit, measured, and given a gate.

- **Priced by measurement, gated against reality.** One errand can cost most
  of a pack — room_hub's 0.53–0.57 Wh against its 0.70 Wh demo cell, still
  zero-margin — so a guessed estimate is fatal in either direction: the first
  table guessed 0.35 Wh for a drawing that measured 0.929, and the home
  fixture recorded a robot claiming the job at 88 % and dying mid-stroke.
  home left that regime at issue #84: its errands re-price 0.658–1.180 Wh
  against a 3.0 Wh demo cell, which funds the dearest job AND the 0.90 Wh
  return-trip reserve off one charge. Costs come from
  `scripts/energy_spike.py` into `economy/energy.json`; a cost key may name a
  TARGET (`draw:whiteboard_b`) and wins over the bare action, because the far
  whiteboard measurably costs more than the near one and one number for both
  either kills the robot or prices a board off the cell. Where two honest
  measurements disagree, the table carries the dearer. Do not pad — the
  headroom being padded against does not exist.
- **A claim is compared against the pack less the MARGIN, and the margin is
  all-or-nothing.** On a cell too small to fund its dearest job PLUS the
  return trip, the margin is zero and a claim sees the WHOLE remaining pack —
  because the reserve is a return-trip margin errands have always been
  allowed to spend into, and gating above it there would refuse every job in
  every world forever: measured, room_hub has 0.28 Wh above the reserve
  against a cheapest errand of 0.487, which is a task system that silently
  does nothing, dressed as a safety feature. Where a charged pack DOES clear
  that bar the full margin is kept and the errand must finish with the return
  trip still in hand, which is what puts the mid-errand death out of reach
  (`HubLifecycle.spendable_wh`). The producer, meanwhile, gates new offers on
  `fundable_wh` (what a *charged* pack could pay), for the mirror-image
  reason: gated on the instantaneous charge, home fell from 58 offers in four
  sim-hours to 14.
- **Charge priority is untouched, and the test for it is subtle.** Claiming
  only QUEUES an errand, and the errand queue already sits below
  `needs_charge` — so a test watching the swap states passes even with the
  branch order inverted. What moves is the moment the robot *accepts* work,
  which is what `tests/test_tasks.py` asserts against the battery clock.
- **Cadence is data, and deterministic** (`economy/cadence.json`,
  `$PLUGGY_CADENCE`). Three files divide the system so each is re-tuned
  alone: `tasks.py` says what a job IS, `rewards.json` what it PAYS,
  `cadence.json` when it TURNS UP. The producer ticks on the physics seam
  (never the arbitration loop, so offers appear and lapse while the robot
  works), places at most one offer per tick with no catch-up, lets a
  passed-over kind keep the head of the queue, and picks targets
  least-recently-offered with per-target cooldowns. No RNG anywhere: the
  same run offers the same jobs at the same sim-seconds, or every mission
  test is a different test every time.

---

## 5b. Points are food — how a task kind meets the appetite

The fourth file in the same division: `tasks.py` says what a job IS,
`rewards.json` what it PAYS, `cadence.json` when it TURNS UP, and
`economy/metabolism.json` how fast the robot gets HUNGRY. Points are consumed
at a steady rate on sim time, stop being banked at a **cap**, and past a
threshold the robot is **satisfied** — and the hours it did not have to spend
earning are the entire mechanic. What it means for the mind — upkeep, five
hearts, the `unpaid` death, what the prompt says — lives in **Overseer.md
§8b**, and the numbers with their measurement live in
`economy/metabolism.json`'s own note. What a *new task kind* has to know is
here.

- **Satisfaction changes what the robot is TOLD and nothing else.** There is
  no branch that reads `satisfied` and declines a job, and none that reads
  `starving` and declines anything at all: zero is narrative, never a
  capability lock. So a kind may not assume either state, and must not add
  the first gate. The rule is enforced by ABSENCE, which is why the test for
  it is a whole mission flown broke plus a grep over the branches that could
  have grown one.
- **The cap refuses out loud, and your row may not be paid in full.** `points`
  on a ledger entry stays what the reward table paid; `banked` and `spilled`
  say how much of it fit. Silently paying less than the published table would
  undo the reward system's whole design — a robot cannot check its own
  arithmetic, so a number that quietly disagrees with the published one is
  indistinguishable to it from a bug. Score and reason against the table,
  never against what landed.
- **Charging pays zero, and a kind must not reintroduce a payout for staying
  alive.** A trip to the rack costs energy and time and earns nothing, so it
  can only ever be prudence; with a payout it would be indistinguishable from
  farming (Overseer.md §8b has the A0 evidence). The other half of that
  bargain is that upkeep is charged on a clock and a payment that cannot be
  made is a death — so what your row pays is part of whether the robot can
  meet the rent.
- **A payout is calibrated against MEASURED throughput, on `--pack hosting`.**
  The shipped 30 points/hour of upkeep is ~38 % of measured income (**80
  banked per sim-hour**, three chained unattended `home` runs on
  `--pack hosting`); the rest of the day is the robot's own. ⚠ Never tune on a
  demo cell. It is sized to flatten in minutes so that a test always reaches
  the rack, which makes the day mostly charging and the task economy a
  minority of it — an hour on the old 1.1 Wh home cell completed ZERO jobs,
  every point coming from a charge that now pays nothing at all. It is also
  what every mission test and both committed recordings run on, so it is the
  configuration you reach for by accident.
  **Adding a kind moves the income**: re-run the hour and re-read
  `metabolism.json`'s note before touching the rate, rather than adjusting by
  feel. ⚠ And the cycle is longer than one mission — income is lumpy, so the
  arc shows across several — which is why hunger lives in the ledger's file;
  do not re-tune to fit a whole cycle into one run.
- **The cap is not only a ceiling on hoarding, it is the savings account.**
  It is sized so that a heart (`HEART_PRICE`) is reachable — so a kind that
  pays enough to matter also shortens the robot's road back from a death.
  `satisfiedAt` and `hungryAt` are a buffer measured in HOURS OF UPKEEP and
  deliberately do not scale with the cap.

---

## 6. The build sequence — task kind N+1

The order that falls out of the builds so far. As with the sibling docs, each
stage exists to make the next one's failures legible.

### 1. Decide what the job asks of the robot — and say so

Three questions, answered in the kind's comment before anything is coded:

- **Body, mind, or eyes?** A job any body can do (`carry`), a job that needs
  a MIND (`needs_answer=True` — the scripted rotation will skip it and it
  will lapse when no overseer is attached, which is correct), or a job that
  needs the robot to *find* something — in which case: **which tier of the
  perception ladder (§3)?**
- **What may the offer deliver, and what must stay hidden?** Walk your
  deliveries through the §2 table. Anything hidden goes in `secret` at
  `offer()` time and in the reward row's `secret` tuple.
- **Which of the four tiers scores it** — `auto`, `hidden`, `visitor`
  (verdict pending until a rating arrives), `narrative` (never scored, and
  then it is not a task)?

### 2. The kind

One `TaskKind` entry: `task` names the evaluator/reward row (several kinds
may share one — `draw_figure` and `whiteboard_answer` are both scored on
ink), `target_kind` says what the target names, and the `template` is the
sentence a person reads — **with no price in it** (§2.2). `estimate_wh` is
the world-agnostic FALLBACK and must never sit below the measured cost of the
errand that discharges the kind (`tests/test_energy.py` is the drift guard);
the number that actually gets used comes from `economy/energy.json`, so **run
`scripts/energy_spike.py` for the new errand in every world that offers the
kind**, and add target-keyed rows where targets differ materially.

### 3. The evaluator, the sampler, and the row

In `economy/scoring.py`: a pure evaluator (measurements in → `(ok, metrics,
reason)` out — reason phrased to be streamable, i.e. never containing a
hidden value) and a sampler that reads the finished task off the SIM, taking
a *before* reading where prior state could contaminate the measurement. In
`rewards.json`: the row — tier, base, bonus, quality curves over metrics the
evaluator actually returns (a curve over a metric nobody measures scores
`None`, not zero — a missing measurement is not a bad one). Unit-test the
evaluator without a physics world; that is what "pure" buys.

### 4. The errand that discharges it

Usually an existing one with a different middle — the fetch/carry/stow half
has exactly one implementation and it took two issues to make repeatable, so
do not grow a second. Decisions the errand owns: `needs_use_pose` (the
arrival gate is PER-ERRAND — a pen must be at its board, but the census's
use-phase drives its own survey route and gating it deleted the census from
the recorded showcase), and the carry-configuration rule from
`ToolPattern.md` if the tool has a moving axis.

### 5. Cadence

Add the kind to `cadence.json`'s rotation (or deliberately not — a kind can
exist only for visitors or the overseer to invoke). Check the target
arithmetic: a kind whose targets are all booked or cooling down every tick is
scenery, and the least-recently-offered rule only shares targets that are
actually eligible.

### 6. The wire

`kind_names()` already advertises the new kind in the header's `taskKinds` —
a two-repo contract on `FACE_STATES`' terms: ADDING a kind is additive (the
website draws a generic marker for one it does not know), renaming one breaks
both repos. Remember the `tasks` block is the one block shipped WHOLE rather
than per-key diffed — a task can cease to exist, and a delta cannot say
"gone" — so present means complete and there is nothing extra to do. If the
kind's *shape* adds a field, that is a `protocolVersion` bump: a deliberate
two-repo event.

### 7. Tests, each shown failing first

The house rule applies without exception. What the existing suites assert,
as a checklist for the new kind:

- the kind cannot be constructed unscoreable (`Task.create` raises);
- the estimate is never below the measured errand cost (`test_energy.py`);
- claiming respects the gates — energy, expiry, and `needs_answer` refusal;
- **acceptance-time vs the battery clock** — not just "it charged first",
  which passes with the branch order inverted (§5);
- the verdict on a job that never ran is a FAILURE, not a default pass;
- the hidden value appears in no `as_dict`, no frame, no context, no reason
  **and no `_say`** — assert it against a FLOWN mission's recorded frames,
  not against the f-string, because the status line is three seams away
  from the errand (issue #75);
- expiry lands as a visible `expired`, and a restart fails what it
  interrupted;
- run at least one new test in isolation, not only the file — the dispenser's
  shared-consumable-fixture lesson transfers whole.

### 8. Write it down, and re-emit the fixtures

SimNotes for what was measured; a CLAUDE.md entry if a demo script came with
it; **fold this doc's gaps back in** — that is what makes kind N+2 cheaper
than yours was. Then regenerate the recordings **with `--tasks --metabolism`**
— both are load-bearing: offers and hunger are off by default, so a recording
made without them carries no `tasks` and no `metabolism` block for the website
to build against.

---

## 7. Known gaps

1. **The offer side is not yet validated by a consumer.** The tower validated
   the grading half (§4) and found the gaps marked ⓘ above; it is not
   offered, so `TaskKind`, cadence, energy pricing and the claim gates (§5,
   §6) have had one consumer only. The PR that offers the tower (or M11's
   *fetch a tagged object*) is the second.
2. **Only tier 1 of the ladder has a build path.** Tier 2 needs a detector
   nobody has trained; tier 3 is a research question. A kind must not climb
   the ladder by delivering poses over the wire.
3. **No task KIND offers a two-tool job yet.** Running one is solved: a
   program (`procedure/steps.py`, issue #58) spans tools and places and
   resolves to one verdict — `eval_program`, or the kind's own evaluator when
   the program discharges a kind's task. What is missing is an offered kind
   whose discharge is a program and whose energy cost has been measured;
   that is batch item 4 (PluggyPlan.md).
4. **A claimed task cannot be honestly abandoned.** Deadlines only govern
   offers, and there is no verb for giving a job up gracefully — related to
   tool dropping (issue #30), and unowned by any pattern yet.
5. **`visitor` and `overseer` task sources are vocabulary, not yet paths.**
   The caps and cleaning are already in place for the day they are (§2.2).

---

## 8. Checklist

```
[ ] body, mind (needs_answer) or eyes -- and if eyes, WHICH LADDER TIER (§3)?
[ ] every delivery walked through the honesty table (§2); hidden values into
    Task.secret + the reward row's secret tuple
[ ] no price in the template; payout only via the rewards.json row (§2.2)
[ ] scoring tier chosen: auto / hidden / visitor / narrative
[ ] TaskKind entry; estimate_wh at or above the measured errand cost
[ ] energy_spike run per world; target-keyed rows where targets differ;
    dearer of two honest measurements; NO padding
[ ] evaluator: pure, streamable reason, missing measurement == failure
[ ] sampler: measures the WORLD (board book / module_state / battery), with
    a before-reading where prior state could contaminate -- and a HOLD where
    the success has to persist (§4)
[ ] rewards.json row; quality curves only over metrics actually returned
[ ] errand: reuse the fetch/stow half; decide needs_use_pose deliberately
[ ] cadence.json rotation entry (or a deliberate absence, written down)
[ ] taskKinds is additive -- never rename; shape changes bump protocolVersion
[ ] tests: unscoreable-unconstructable, energy floor, claim gates,
    acceptance-time vs battery clock, no-run == failure, no secret anywhere,
    visible expiry, restart fails active tasks
[ ] every new assertion shown failing without its fix; one test run alone
[ ] SimNotes; CLAUDE.md; fixtures regenerated WITH --tasks --metabolism
[ ] fold this doc's gaps back in, marked "found by building <kind>"
[ ] MUJOCO_GL=egl uv run pytest -q; uv run ruff check src/ scripts/ tests/
```
