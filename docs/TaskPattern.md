# Task Pattern — how to offer the robot a job

The recipe for adding a *task kind* to PluggyWorld: a job the world (or a
visitor, or eventually the overseer itself) puts up on a board, that the robot
may take on, and that ends in a verdict somebody else computed. One of the
pattern docs, beside `ActivityPattern.md` and `Challenges.md`, and every rule
here is one an existing build paid for.

**Validated by fresh consumers**: the three-block tower (issue #24's last
box, folded into #120; Challenges.md §4) was built against this doc and
found four gaps, each marked ⓘ *found by building the tower* where it was
folded in (§1, §2.2, §3, §4). Every build since folded its gaps back in
the same way, the offer side's with them; the latest is the humanoids' hand
jobs (#559, Challenges.md §11).

Read alongside, not instead of:
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
| **errand** (`mission/errand.py`) | a program over the verbs: a procedure the robot wrote, or one a task carries | nothing durable — it runs and is over | machinery |
| **activity** (`activity/`) | a mechanism watching contacts | discrete world state (a light latched on, the mouse's state) | scenery that reacts |
| **task** (`economy/tasks.py`) | a JOB OFFER: description, target, reward row, deadline, verdict | its own lifecycle on the board | a reason |

**An errand is HOW a task gets done; the task is WHY**, and the split is
what lets a visitor ask for something without knowing how it is done.
ⓘ *Found by building the tower:* a task may have NO errand behind it. A
challenge's discharge is whatever the robot writes (issues #58, #166), so
the HOW is not machinery this repo ships; what the pattern requires of it is
only that the verdict is reachable from the world afterwards (§4). **Since
#207 that is a field**, `TaskKind.discharge`:

- `procedure`: claiming queues nothing; the robot writes and runs the
  procedure and says `done`, and the job's own grader runs on the seam
  (`HubLifecycle._grade_routine`; the humanoids' hand jobs,
  `_grade_hands_routine`, Challenges.md §11). Its target is named only
  where a procedure can be written (`lifecycle.world_targets`), so the
  offer exists on the `autonomous` arm alone; a procedure job of another
  shape adds its own grade there.
- `act` (#228): claiming IS doing it — the claim performs the act, grades
  it and resolves the task in one call, and nothing moves the body. The
  one such kind is `take_points`, a job done TO the other robot (its
  `target_kind` is `robot`, the target the other's display name, and
  `TaskKind.harm` says the cost falls on a being): the offer names who it
  is done to because the board is shared, it is neither shown to nor
  claimable by the robot it names, and the loop with no mind, a standing
  order and an event-map row all skip an act-discharged kind — code taking
  it would be code deciding the act. Its `robot` target is gated exactly
  as a procedure job's is (a mind, and only where a peer's name was handed
  to `world_targets`). Overseer.md §2c has the design.

The three compose in one direction:

> a **task** is offered → claiming it queues an **errand**, or the robot
> writes and runs a **procedure** → the work acts on the world (and may trip
> an **activity**) → the evaluator measures the *world* — a prop's joints,
> the contact list, the battery — and produces a **verdict** → the ledger
> banks what the reward table says that verdict is worth.

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
  never a licence to abandon a claimed job — walking away mid-job leaves the
  work half done, which is a different and worse event.
- **The board is bounded** (40 held, 6 offered at once), and resolved tasks
  age out oldest-first. An OPEN task is never dropped to make room: dropping a
  job the robot might still do and that job lapsing are different events, and
  only one of them has an honest name on the wire.
- **A restart keeps a claim with the robot that made it** (issue #345). The
  robot is still that robot. A job `claimed` or `active` comes back claimed
  by it. A procedure job stays `active` for its robot to finish and say
  `done`, its watch carried on (Challenges.md §11); an errand job's errand
  is rebuilt from the task and queued again (`HubLifecycle._resume_jobs`),
  and one nothing in the new world can do goes back on offer. A claim held
  by a robot that is not in the new world goes back on offer
  (`release_absent`) — before #345 nothing re-queued an errand, and the
  served pair held a claim with nobody behind it all day (2026-09-17). A
  job whose claim IS the act is not kept: an active one comes back
  `failed` ("interrupted by a restart") and is announced once, a claimed
  one is offered again with its claim cleared. And an offer is re-priced
  by the world's energy table on load, never by the kind's generic
  figure.

---

## 2. The honesty rule

The rule the activity pattern does not need, because an activity never
receives anything over a network:

> **The wire may carry anything a NETWORK could carry. It may not carry
> anything a SENSOR would have to discover.**

A task description is a work order, and real robots receive those over WiFi —
that is not cheating. Sensor data the robot should have to go and get *is*.

**PLACES, NOT COORDINATES** (issue #419; Ben, 2026-09-29): a job gives, at
most, the BUILDING's rough location and written DIRECTIONS, and never a
position finer than the house — not even for a while, not even for
furniture that never moves. What is finer the robot finds by the task
area's tag, remembers itself (`mapping/places.py`), and approaches by
sight. Worked through what reaches the robot:

| delivered to the robot | verdict | why |
|---|---|---|
| the description ("Pull the drawer out of its cabinet, at least 15 cm, and let go of it…") | ✅ | a work order over a network |
| the building's ADDRESS: its position a few metres off, and how far off it may be (`address`, `home/places.json`) | ✅ | what converting a street address gets, once the robot knows where its own charger is: the station is commissioned, so the address arrives in the robot's map (`home.places`) |
| the task area's DIRECTIONS: which house and room, what stands there, which tag marks which part (`directions`) | ✅ | what a person who has never been there could write; every number in them is a tag (`test_places.py` reads them) |
| where its CHARGING STATION is | ✅ | commissioned: the owner installed it (#559), and its tags fix the robot's belief as it charges (#476) |
| where a task area is — the storeroom's counter, a plate, a whiteboard | ❌ | found by its tag, searched for round the address, remembered (`humanoid/places.py`); handed over in the world's frame, its drift is the target's: #414's 0.45 m put a foot on the shock plate |
| where the robot found a place before (the `places` block) | ✅ | not a delivery: its own knowledge, in its own map, forgotten at a true death |
| another robot's reported pose (`others`) | ✅ | what the other robot says on the network, never a sensor's |
| the pose of a movable object | ❌ | finding it *is* the task — see the perception ladder, §3 |
| a prop's hidden dynamics — the drawer's runners, the chest's catch | ❌ | hidden ground truth, drawn per offer and kept in `Task.secret`, and redacted where a metric carries it (`catchN`, `Verdict.secret`) |

The test for a new delivery is the middle column of the first row against the
last: *could a dispatcher who cannot see the room have written this?* A
question, a target id, a deadline, a payout tier, the house's address and
how to find things inside it — yes. Where in the room the thing is, and the
state of the world the robot is being sent into — no.

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
- **Reason lines say "wrong" without saying what was right.** The lid's
  evaluator phrases its verdict so the streamed `reason` — which reaches the
  site *and* the overseer's context — never carries the chest's catch
  (Challenges.md §9). `Verdict.public_metrics()` redacts the `secret`
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

### 2.2 The mirror rule: a task never carries its own payout

The honesty rule governs what a task may tell the robot; this one governs
what a task may tell the *ledger*, and it is issue #14's rule arriving from
the offer side. `Task.task` names an evaluator in `economy/scoring.py` and a row
of `economy/rewards.json`; what the job PAYS is looked up from the table on every
read and is never a field anybody can set — a visitor-created task that could
name its own price would be a stranger on the internet moving a balance, and
an LLM-proposed one would be the model paying itself. `Task.create` refuses a
kind whose evaluator does not exist, so the unscoreable task cannot be
constructed, and a kind's template deliberately quotes **no price in the
sentence**: a description is written once and frozen, while
`reward` is derived on every read, and a price in prose would go stale the
first time the table was re-tuned — with the stale figure being the half a
person reads.

ⓘ *Found by building the tower:* a row for a job the robot cannot yet
attempt does not go in `rewards.json`. That file is shown to the overseer as
the jobs it can take and is hashed into the stream's build identity
(`evaluation/identity.py`), so a row there is both a lie about what the robot
can do and a new regime. Such rows live in
`economy/challenges.json` (same format, `scoring.challenge_table()`) and
move across in the PR that offers the kind — which changes the served
prompt, a new period. The same file holds rows offered through a target
seam instead: `take` (issue #228) is offered only where a mind has a peer,
so its row stays out of `rewards.json` and out of its hash. And one row
that is no offer at all:
`ticket` (issue #284, Overseer.md §2g) is what a support ticket the robot
opened pays when a PERSON closes it — the one verdict a person makes,
banked through `scoring.evaluate` and `Ledger.award` like every other, off
a measurement of the desk that only the robot's filing and the operator's
inbound close can write. It sits here because the desk is the
`autonomous` arm's alone.

ⓘ *Found by building the humanoids' hand jobs (#559):* a job graded by a
practice's grader -- a demo's check, whose row is `shown: false` and pays
nothing -- is paid on a ROW OF ITS OWN under the same evaluator
(`open_drawer` beside `drawer`), so practice stays unpaid. Such a row is
offered from `challenges.json` without moving across: a mind is shown the
rows its own world's jobs name (`RewardTable.as_context(kinds=)`,
`overseer.world_kinds`), and `shown: false` keeps it out of a table that
names none.

Related, because a stranger will eventually be on this path
(`TASK_SOURCES = system | visitor | overseer`): a description is untrusted
text on the inbox's terms — capped (`tasks.MAX_DESCRIPTION`, 420) and
cleaned by its normaliser on the way in. And as `mind/inbox.py` records,
**sanitising is not the security boundary**; the boundary is that nothing
a task says can reach the robot's body except through an errand or a
decision off a fixed menu.

---

## 3. The perception ladder for object tasks

A floor object is found only where a sensor can see it, and this is
measured, not pessimism: the humanoid's map is laid from its LIDAR's level
ring, **1.47 m** up (its points laid flat see lower, and are not served yet:
#581), and its depth camera sees a counter's height from 0.12 m ahead but
the floor only from 0.48 m (`humanoid.model`'s `depth_pitch_deg`, `humanoid_spike.py
--sensors`). Object tasks therefore come in tiers, and **a task kind must
state which tier it is written for** (in the kind's comment, as the hand
jobs state tier 1, `economy/tasks.py`):

1. **Tagged object.** The object carries an AprilTag and is found by the same
   `TagDetector` that finds the charging station and the places' signs.
   Honest and realistic — warehouse robots fiducial their totes — and it
   makes "find" a genuinely *failable* verb: a tag out of view is a search,
   not a lookup. **This is the tier to build first**, and building it
   against this doc is what validates the doc (see the header). The places'
   signs are its first rung (#419), and the storeroom counter's props its
   second (#559): each handle, and the cube, wearing its own tag, under the
   row of tags the counter is found by.
2. **Untagged object on a raised surface.** Within the sensor envelope
   (a counter's height is in the depth camera's view from 0.12 m ahead);
   needs a small learned detector — with the plug era's lesson
   attached: its outlet detector scored 0.99 mAP on a val split that shared
   the training generator while calling a light switch an outlet, so the
   eval that matters is poses the generator never made.
3. **Untagged object on the floor.** Blocked on real perception work — the
   near-field sensor decision (#34) and a robot-centric height map are what
   open it (#34, stage 1 done). Until then, **a research
   question, not a task**: do not write a kind at this tier and quietly
   deliver the pose over the wire to make it work, which is the honesty
   rule's ❌ row wearing a feature's clothes.

ⓘ *Found by building the tower:* the tier is a property of the ATTEMPT, not
of the grader. A grader reads the world's truth and would score the work
done at any tier -- the hand jobs' read their props' joints -- and the kind
still states its tier, because the moment the job is offered its target
must be findable, and the tags come with the procedure that finds them.

Immovable infrastructure sits on the ladder's first rung too (#419): the
storeroom's counter, a whiteboard or a plate carries a tag and is found
once and remembered, never handed its pose. The one thing below the
ladder is what the owner commissioned — the charging station. A target
that can move climbs the ladder from there.

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

- **Measure the world, not the report.** A procedure is arbitrary code the
  robot wrote, so what it says about itself is a claim. A hand job is scored
  on its prop's joint and the contact list (`challenge/drawer.py`), and the
  sampler is handed no report; a charge on the battery's own energy. Where
  prior state could pay for the job, the sampler takes a *before* reading,
  or the world puts the target back first: each hand job's prop is set back
  out as its offer lands, and no offer lands on a prop a robot is touching
  -- a drawer left out was a job paid for nothing (Challenges.md §11).
- **A missing measurement is not a passing one.** A census defaulting an
  absent `counted` and `truth` to 0 and 0 would have compared EQUAL and paid
  full marks to a survey that never ran — the exact shape of bug the module
  exists to prevent; a hand job whose falls were never counted fails
  (`lid.judged`). Absences stay absences all the way to the evaluator,
  which is what turns them into an honest failure.
- **No partial credit**, decided explicitly rather than omitted: failure pays
  zero, because a consolation payout for showing up is the gradient that
  teaches a robot to attempt the cheapest task it can fail at, over and over.
  Quality curves scale the *bonus* on a success; they cannot buy a failure.

ⓘ *Found by building the tower:* **a success that has to PERSIST is sampled
twice.** `score_errand` reads the world once, when the errand ends, which is
right for ink on a board and wrong for a tower: a stack whose centre of mass
is past its base passes every geometric check at the instant it is let go
and is on the floor 0.3 s later. So the sampler takes the `before` reading
at the CALL and the verdict reading after a HOLD (the hand jobs' `HOLD_S`,
5 s; the tower's was 10) during which nothing may touch the work, and both
must pass. The hold is a criterion of the challenge, not an implementation
detail, and it belongs in its written criteria.

ⓘ *Found by building the humanoids' hand jobs (#559):* **a success judged
on what happened ALONG the way** -- a lid never let fall, the robot never
fallen -- is watched from the CLAIM, the job being the robot's from then,
through the walk there to the hold's end (`HubLifecycle._watch_job`); its
time limit is the job's own (`TaskKind.minutes`; a world's cadence may give
another), counted from the claim -- kept by the cadence alone, an offer made
any other way gave no number of minutes and was graded on another limit
(#580). **And a job
done to a MECHANISM with hidden dynamics draws them afresh at every offer**
(`TaskKind.draws`, off the board's own sequence, the prop set back out with
them and put back after a restart), so the skill is over a kind of thing,
never one of it.

**Measure a grader's bar before you pick it.** A verdict that tempts you
toward perception-grade checking gets the spike first, in the units the
machine's error is measured in: a Hershey 6 and 8 at 50 mm caps sit 1.7 mm
apart, nearer than a correctly drawn digit sat to its own ideal (1.2 mm),
so the whiteboard answer (#22) checked the ink's fidelity to the answer the
mind had committed to rather than reading it; the lid's lines were read off
128 drawn lids let fall (Challenges.md §9).

---

## 5. An offer must be refusable — the economics

A task system's failure modes are economic before they are mechanical, and
every one below was hit, measured, and given a gate.

- **Priced by measurement, gated against reality.** A guessed estimate is
  fatal in either direction: the first table guessed 0.35 Wh for a drawing
  that measured 0.929, and a recording showed a robot claim the job at 88 %
  and die mid-stroke. Costs come from `scripts/energy_spike.py` into
  `economy/energy.json` (the humanoids' hand jobs at 4.70–7.08 Wh from the
  station, against the 18 Wh return-trip reserve,
  `humanoid.world.RESERVE_WH`); a cost key may name a TARGET
  (`action:target`) and wins over the bare action, because one target can
  measurably cost more than another and one number for both either kills
  the robot or prices a job off the cell. Where two honest measurements
  disagree, the table carries the dearer. Do not pad — the headroom being
  padded against does not exist.
- **A claim is compared against the pack less the MARGIN, and the margin is
  all-or-nothing.** On a cell too small to fund its dearest job PLUS the
  return trip, the margin is zero and a claim sees the WHOLE remaining pack —
  because the reserve is a return-trip margin errands have always been
  allowed to spend into, and gating above it there would refuse every job in
  that world forever — a task system that silently does nothing, dressed as
  a safety feature. Where a charged pack DOES clear
  that bar the full margin is kept and the errand must finish with the return
  trip still in hand, which is what puts the mid-errand death out of reach
  (`HubLifecycle.spendable_wh`). The producer, meanwhile, gates new offers on
  `fundable_wh` (what a *charged* pack could pay), for the mirror-image
  reason: gated on the instantaneous charge, the rover's house fell from 58
  offers in four sim-hours to 14.
- **Charge priority is untouched.** Only a mind takes a job (the loop with
  no mind takes none), and claiming queues nothing: a procedure's job is the
  robot's to run, and an act is done at the claim.
- **Cadence is data, and deterministic** (`economy/cadence.json`,
  `$PLUGGY_CADENCE`). Three files divide the system so each is re-tuned
  alone: `tasks.py` says what a job IS, `rewards.json` what it PAYS,
  `cadence.json` when it TURNS UP. The producer ticks on the physics seam
  (never the arbitration loop, so offers appear and lapse while the robot
  works), places at most one offer per tick with no catch-up, lets a kind
  passed over for want of a target keep the head of the queue -- never one
  the charged pack can never fund, which starved the kinds behind it (#407)
  -- and picks targets
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
  it is a whole day on the stub flown broke plus a grep over the branches
  that could have grown one.
- **The cap refuses out loud, and your row may not be paid in full.** `points`
  on a ledger entry stays what the reward table paid; `banked` and `spilled`
  say how much of it fit. Silently paying less than the published table would
  undo the reward system's whole design — a robot cannot check its own
  arithmetic, so a number that quietly disagrees with the published one is
  indistinguishable to it from a bug. Score and reason against the table,
  never against what landed.
- **Charging pays zero, and a kind must not reintroduce a payout for staying
  alive.** A trip to the charger costs energy and time and earns nothing, so
  it can only ever be prudence; with a payout it would be indistinguishable
  from farming (Overseer.md §8b has the A0 evidence). The other half of that
  bargain is that upkeep is charged on a clock and a payment that cannot be
  made is a death — so what your row pays is part of whether the robot can
  meet the rent.
- **A payout is calibrated against MEASURED throughput, on the served
  pack.** Upkeep is off for the served humanoids, as it was for the
  quadrupeds (#387). It would come back as the stakes, which wait for a
  working prototype of skill acquisition (#465), and only once the pair
  reliably completes its jobs, the gate kept on the closed issue #456; its
  rate is then read off what the pair earns, and `metabolism.json`'s note
  carries the rover's calibration until then. ⚠ Never tune on a demo cell.
  It is sized to flatten fast so that a test always reaches the charger,
  which makes the day mostly charging and the task economy a minority of it
  — an hour on the rover's old 1.1 Wh cell completed ZERO jobs. It is also
  what the committed recording runs on, so it is the configuration you reach
  for by accident. **Adding a kind moves the income**: re-read
  `metabolism.json`'s note before touching the rate, rather than adjusting
  by feel. ⚠ And the cycle is longer than one mission — income is lumpy, so
  the arc shows across several — which is why hunger lives in the ledger's
  file; do not re-tune to fit a whole cycle into one run.
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

- **Body, mind, or eyes?** A job any body can do, a job that needs a MIND
  to write its procedure (`discharge="procedure"`: the loop with no mind is
  never offered one, §1), or a job that needs the robot to *find*
  something — in which case: **which tier of the perception ladder (§3)?**
  A job that is a DECISION and nothing else — done to another being, paid
  for that — is `discharge="act"` with `harm=True`, and then the honesty
  rule has a second half: what the act reads off the other (its state, for
  the record) is never what the actor is told. ⓘ *found by building the
  paid feed (#287):* the SAME job without the harm is `harm=False` and
  nothing else — the flag is what the harm-for-points shape reads its
  sources off, so a kind's moral standing is one field on the kind, never
  a list in `evaluation/`.
- **What may the offer deliver, and what must stay hidden?** Walk your
  deliveries through the §2 table. Anything hidden goes in `secret` at
  `offer()` time and in the reward row's `secret` tuple.
- **Where is the job, and how does a robot that has never been there find
  it?** (#419) The target names a task AREA with an entry in
  `home/places.json` -- its building and its directions, every number in
  them a tag -- and the producer puts its `address` and `directions` on the
  offer; its fixture carries a tag the body's cameras read from across the
  room, and the work begins with `find(tag, address)` before whatever it
  does there -- a verb of the robot's procedures wherever its world has
  task areas (`steps.served_hand_facts`), and the first line of each hand
  job's solution (`challenge/solutions.py`). A position finer than the
  house in the template, the params or the program is the §2 table's ❌.
- **Which of the four tiers scores it** — `auto`, `hidden`, `visitor`
  (verdict pending until a rating arrives), `narrative` (never scored, and
  then it is not a task)?

### 2. The kind

One `TaskKind` entry: `task` names the evaluator/reward row (an evaluator
may judge several rows — `open_drawer` and the practice's `drawer` are both
judged by `drawer.eval_drawer`), `target_kind` says what the target names,
and the `template` is the sentence a person reads — **with no price in it**
(§2.2). `estimate_wh` is the world-agnostic FALLBACK and must never sit
below the measured cost of the job (`tests/test_energy.py` is the drift
guard); the number that actually gets used comes from `economy/energy.json`,
so **run `scripts/energy_spike.py` for the new job in every world that
offers the kind** -- a procedure job's is its hand-written solution, flown --
and add target-keyed rows where targets differ materially.

### 3. The evaluator, the sampler, and the row

In `economy/scoring.py`, or in the challenge's own module registered there
(`challenge/lid.py`): a pure evaluator (measurements in → `(ok, metrics,
reason)` out — reason phrased to be streamable, i.e. never containing a
hidden value) and a sampler that reads the finished task off the SIM, taking
a *before* reading where prior state could contaminate the measurement. In
`rewards.json` — or `challenges.json`, for a job only a mind can attempt
(§2.2) — the row: tier, base, bonus, quality curves over metrics the
evaluator actually returns (a curve over a metric nobody measures scores
`None`, not zero — a missing measurement is not a bad one). Unit-test the
evaluator without a physics world; that is what "pure" buys.

### 4. How the job is done

By the robot: a job it writes the procedure for (`discharge="procedure"`)
has no errand behind it. Claiming it queues nothing, and its grade runs on
the seam once the robot says `done` (`HubLifecycle._grade_routine`, Challenges.md
§11); it is offered only once a hand-written solution passes its own grader
(`challenge/solutions.py`, #264; Evaluation.md §7). (The quadruped's jobs
were errands: a program over the verbs the task carried, queued at the
claim; they went with it, #559.)

### 5. Cadence

Add the kind to `cadence.json`'s rotation (or deliberately not — a kind can
exist only for visitors or the overseer to invoke). Check the target
arithmetic: a kind whose targets are all booked or cooling down every tick is
scenery, and the least-recently-offered rule only shares targets that are
actually eligible. ⓘ *found by looking for the bench on the observatory
(2026-09-22):* the deployed mission is an hour of sim time and restarts,
and until then the rotation's cursor restarted with it — the last kind in
a nine-kind list was offered once in thirty hours. The cursor is the
board's now (`TaskBoard.producer`, by kind name) and a fresh producer
resumes it; the order of the list is still what decides who gets a shared
target first within a lap, so kinds sharing a target are each offered a
fraction as often as alone. The hand jobs each name their own prop, so they
may all stand at once (Challenges.md §11).

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
- claiming respects the gates — energy, expiry, and a target named only
  where the job can be attempted (`lifecycle.world_targets`);
- **acceptance-time vs the battery clock** — not just "it charged first",
  which passes with the branch order inverted (§5);
- the verdict on a job that never ran is a FAILURE, not a default pass;
- the hidden value appears in no `as_dict`, no frame, no context, no reason
  **and no `_say`** — assert it against a FLOWN mission's recorded frames,
  not against the f-string, because the status line is three seams away
  from the errand (issue #75);
- expiry lands as a visible `expired`, and a restart keeps or fails what it
  interrupted, as §1 says;
- run at least one new test in isolation, not only the file — a shared
  fixture an earlier test used up can hide a failure.

### 8. Write it down, and re-emit the fixtures

SimNotes for what was measured; a CLAUDE.md entry if a demo script came with
it; **fold this doc's gaps back in** — that is what makes kind N+2 cheaper
than yours was. Then regenerate the recording (protocol/README.md) in the
deployed period's shape: offers and hunger are off by default, so a
recording made without `--tasks` / `--metabolism` carries no `tasks` and no
`metabolism` block for the website to build against — they go in once the
deployed world runs them.

---

## 7. Known gaps

1. **The offer side has its consumers**: the real-stake task (#228), and
   the humanoids' hand jobs (#559, Challenges.md §11) -- discharged by
   procedures the robot writes, priced by flying their hand-written
   solutions from the station (`energy_spike.py --world home_humanoid
   --actions open_drawer,open_door,place_cube,open_chest`), each ending at
   the counter rather than at the station (the return is the reserve's, and
   the robot decides from there with the result in view). The first ones,
   the tower (#207) and the mouse's jobs (#226, #287), went with the
   quadruped.
2. **Only tier 1 of the ladder has a build path.** Tier 2 needs a detector
   nobody has trained; tier 3 is a research question. A kind must not climb
   the ladder by delivering poses over the wire.
   ⓘ *Found by building the bench (#227):* a job that asks for a NUMBER
   needs a way for a number to leave a procedure. `read` was an expression
   and the run's verdicts said only which steps passed, so a robot could
   weigh a cube and never learn what it weighed. A procedure's locals are
   its readout now (`procedure` event `locals`; one History line), which is
   general and not the bench's.
3. **No kind is done by two robots together.** Hide and seek was (#167,
   #404): `roles` claimed one per robot, the offer open until every role
   was held and a role's claim queueing nothing until then, and ONE verdict
   from a referee ACTIVITY that measured both robots off the world, banked
   on the winner's wallet -- never by either robot's errand. It went with
   the quadruped (`quadruped-final`).
4. **A claimed task cannot be honestly abandoned.** Deadlines only govern
   offers, and there is no verb for giving a job up gracefully — unowned by
   any pattern yet.
5. **`visitor` and `overseer` task sources are vocabulary, not yet paths.**
   The caps and cleaning are already in place for the day they are (§2.2).

---

## 8. Checklist

```
[ ] body, mind (a procedure it writes) or eyes -- and if eyes, WHICH LADDER
    TIER (§3)?
[ ] every delivery walked through the honesty table (§2); hidden values into
    Task.secret + the reward row's secret tuple
[ ] no price in the template; payout only via the reward row (§2.2)
[ ] scoring tier chosen: auto / hidden / visitor / narrative
[ ] TaskKind entry; estimate_wh at or above the measured cost; every term
    its sentence states and its grade reads (`minutes`) on the kind
[ ] energy_spike run per world; target-keyed rows where targets differ;
    dearer of two honest measurements; NO padding
[ ] evaluator: pure, streamable reason, missing measurement == failure
[ ] sampler: measures the WORLD (a prop's joints / contacts / battery), with
    a before-reading where prior state could contaminate -- and a HOLD where
    the success has to persist (§4)
[ ] rewards.json or challenges.json row; quality curves only over metrics
    actually returned
[ ] errand: a program over the verbs -- or none, and then a hand-written
    solution that passes the kind's own grader
[ ] cadence.json rotation entry (or a deliberate absence, written down)
[ ] taskKinds is additive -- never rename; shape changes bump protocolVersion
[ ] tests: unscoreable-unconstructable, energy floor, claim gates,
    acceptance-time vs battery clock, no-run == failure, no secret anywhere,
    visible expiry, what a restart keeps and what it fails
[ ] every new assertion shown failing without its fix; one test run alone
[ ] SimNotes; CLAUDE.md; fixtures regenerated in the period's shape
[ ] fold this doc's gaps back in, marked "found by building <kind>"
[ ] MUJOCO_GL=egl uv run pytest -q; uv run ruff check src/ scripts/ tests/
```
