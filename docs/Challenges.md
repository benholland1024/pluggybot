# Challenges — how to grade a job nobody wrote a scorer for

The decision issue #120 asked for: how a challenge the robot has never seen
gets a verdict, without the thing being graded touching the grade. One of
the pattern docs, beside `TaskPattern.md`, and written first as a decision
with one worked example rather than a framework — the framework came with
the challenge set: demo 1's lid (#493, §9), the hands' three checks (§10)
and the humanoids' jobs in the house built on them (§11). The first two,
the 2026-09-11 batch's tower and bench, went with the quadruped (#559), as
did its pair's hiding spot; the tag `quadruped-final` keeps them.

Read alongside:
- `docs/TaskPattern.md` §4 — grading measures the world, never the report.
  Everything here inherits that; this doc adds what a *novel* job needs on top.
- `docs/PluggyPlan.md` — the six qualities; a challenge is an instrument for
  the first one (capability: can it acquire new skills by its own effort).
- `docs/Evaluation.md` §5 — what silently invalidates a number; two of its
  warnings decide against two of the candidates below.

---

## 1. The problem, stated once

Every task kind names an evaluator, and `Task.create` refuses a kind whose
evaluator does not exist. That rule is right — an agent that can score its own
work learns to declare victory — and it means **a challenge is blocked on
scoring, not on tools.** A robot that can build a ramp and cannot be told
whether the ramp worked has learned nothing, and neither have we.

So the question is not "what can the robot attempt" but "what can code grade
that was never scripted" — and the answer has to keep `scoring.py`'s founding
property: **the thing being graded cannot influence the grade.**

## 2. The candidates, against the three example challenges

The challenges from #45, as the shape of the thing: *find a hiding spot the
other robot has to search for*; *stack a tower of three blocks, then roll in a
circle around it*; *build a ramp that gets you onto the coffee table*.

| approach | hiding spot | tower + circle | ramp | keeps the property? |
|---|---|---|---|---|
| **Geometric predicate** off `xpos` / contacts | ✗ — the goal is a relation between two minds, not a pose | ✓ tower (a world state) · ✗ circle (a *trajectory*: needs sampling during the run, not after) | ✓ — "chassis on the table top, stationary"; the ramp itself is not graded | ✓ by construction: reads the world, knows nothing about method |
| **Visitor rating**, generalised | ✓ but slow, and a rating of a *hiding* spot is a rating of the audience's search | ✗ — a tower either stands or it does not; asking a person is a worse instrument than a ruler | ✗ same | ✓ for taste, ✗ as a measurement — Evaluation.md §5: a result reachable by a visitor measures the audience |
| **LLM-as-judge** | ✓ in principle, ✗ in practice: it would judge a *description* of the hiding | ✓ but strictly worse than the ruler it would be paraphrasing | ✓ same | ✗ **by default** — the judge is shown text, and the nearest text is the actor's own account. Made safe only by rules (§5), never by construction |
| **Pre-declared predicate**, written by a human with the challenge | ✓ with a second robot to seek (M12): "the seeker never sights the hider inside T seconds" | ✓ tower · circle only with a sampler on a step hook (§6; the lid's is the first, §9) | ✓ | ✓ — it is the geometric predicate with the one thing it lacked: written down before the robot saw the job |

Reading the table: the first and fourth rows are the same mechanism. A
geometric predicate is what a pre-declared criterion *compiles to*; what makes
it a challenge rather than a task kind is only that a person wrote the
criterion before the robot saw the challenge, and that it says nothing about
how. The visitor tier stays what it is — the instrument for taste, quality 4 —
and is not a grader. The LLM judge is not chosen, and §5 says the terms on
which it could be.

## 3. The decision

**A challenge is graded by a success predicate written before the robot sees
it, evaluated by code against the measured world through the existing
MEASURE / JUDGE / PAY chain.** Concretely, a challenge is:

1. **Criteria, first and in writing.** Numbered, in the challenge module's
   docstring and as the constants the evaluator reads (`challenge/lid.py`
   is the shape). Each criterion is a statement about the *world* — poses,
   contacts, time — and none is a statement about method. "Set down shut"
   is a criterion; "lifted by its knob" is not.
2. **An evaluator and a sampler on `scoring.py`'s registry**, the same
   door every task goes through: the sampler reads the sim, the evaluator is
   pure, `evaluate` seals the verdict, `Ledger.award` re-derives the points.
   The sampler is handed the errand's `result` because the seam hands it to
   every sampler, and **reads nothing from it** — a test hands it a report
   that says "opened" over a world that says "not" and asserts the failure.
3. **A hold.** Grading happens twice: at the moment the robot says it is done
   and again after a wait during which nothing may touch the work. The
   second reading is the verdict; the first is what stops "it was a tower
   until I let go" from being paid. Where a challenge's success is a state
   that has to persist, the hold is a criterion, not an implementation detail.
4. **Its own props.** A challenge adds what it needs to a world spec through
   `MjSpec` rather than editing a committed world, so no fixture is
   re-emitted and no mission trajectory reshuffles for a world that does not
   offer it: the counter's props come into a world of their own for practice
   (`evaluation/counter.py`) and into the humanoids' house at load when
   offered there (`home/storeroom.py`, attached by `humanoid/world.py`).
   ⚠ The one exception is deliberate (issue #215): the lab's props -- the
   bench and its two masses and the cage (`home/props.py`,
   `activity/cage.py`) -- landed in the generator with the second house,
   before any grader or activity behind them, because a world change is the
   largest regime break there is and it was to happen ONCE. The tower's
   blocks joined them when it was offered (#207), and all of them stay now
   that their jobs have gone: the house's geometry is what a continuation
   fingerprints.
5. **A reward row in `economy/challenges.json`, not `rewards.json`.** Same
   format, same loader (`scoring.challenge_table()`), deliberately a separate
   file: a row in `rewards.json` is shown to the overseer as a job it can take
   and is hashed into the header's `dataHashes`, so moving a row across is
   the PR that offers the challenge, and a new period. The jobs offered so
   far stayed in it instead: a procedure job's target is named only where a
   mind can write the procedure (`lifecycle.world_targets`), and a mind is
   shown the rows its world's jobs name (`RewardTable.as_context(kinds=)`,
   §11) -- offering one changes the served prompt all the same.

What this gives up is generality, and that is the trade the issue predicted:
a challenge is exactly as gradeable as its author's ability to write its
success as a predicate. §6 is the list of what that excludes.

## 4. The worked example: the three-block tower

The first challenge (`challenge/stack.py`, #120), graded at the call and
again `HOLD_S` = 10 s later: three blocks one tower three high, each
resting on the one below within half an edge, and free-standing -- every
block touching only the floor or another block. It went with the
quadruped (#559), and what it established holds for every grader since
(the lid's, §9, and the hands' checks, §10):

- **The criteria were written before the physics was run**, and the
  measurement knows nothing of method; a failing reason names what it saw
  (`touchedBy`: whatever foreign geom was holding the tower up).
- **The passing run is placed by a test's hand**: there is no robot in it,
  and the grader cannot tell -- that is the point.
- **One failing run per criterion, each failing for its own reason**
  (`tests/test_lid.py` does the same for the lid): *it fell* -- a tower
  leaning 9 mm a layer passed every geometric check at the call and was on
  the floor 0.3 s later, paid without the hold; *propped up* -- the rover's
  claw closed on the top block, geometry passing and contacts not; *merely
  reported* -- the sampler handed a report of success over the untouched
  world, zero points; *never measured, or measured early* -- an absent
  snapshot or a short hold fails, because a missing measurement is not a
  passing one.
- **A challenge's props are part of its criteria.** On MuJoCo's default
  soft contact a stack offset 2 and 4 mm -- a tower by any standard --
  crept over and fell at 16.9 s: `SimNotes.md`'s regularised-friction drift
  ("The grip that leaked"), fixed at its source with the hard `GRIP_SOLIMP`
  (`home/props.py`; the house's blocks still carry it). With it a tower
  whose centre of mass was over its support stood the 30 s measured, and
  one whose was not fell inside 0.31 s, which is what wood does. A prop
  that creeps grades the solver, not the robot.

## 5. If an LLM judge is ever used

Not now, and not for anything a ruler can measure. If a challenge's success
genuinely cannot be written as a predicate (§6) and is not a matter of taste
(the visitor tier), the terms on which a model may grade are:

- **a different model from the actor**, named in the record like `build.model`;
- **shown the world, never the actor's account of it** — rendered frames,
  poses, the board book; not the reason line, not `History.md`, not the
  errand's `result`. A judge reading the robot's narration is the sim marking
  its own homework with extra steps;
- **its verdict sealed the same way** — through `scoring.evaluate`, with the
  criteria it was asked to apply written down first and committed, so that a
  re-run against the same world is a re-grade and not a new opinion;
- and treated, in the write-up, as what it is: a measurement of the judge as
  much as of the robot, which Evaluation.md §5 already says of the audience.

## 6. What this mechanism cannot grade

The boundary of the next milestone, stated so it is designed against rather
than discovered:

1. **Behaviour.** A predicate over the finished world grades a *state*; "roll
   in a circle around it" is a *trajectory*, and `score_errand` samples once,
   at the end. Grading it needs a sampler with a step hook — the activity
   pattern's `sense()` on the scoring seam — accumulating bearing swept and
   radius held about the tower, with the tower still standing at the end.
   That is one seam and no new principle (the lid's grader reads its lid on
   one every step, §9), and the circle half of the example was never built.
2. **Relations between minds.** A hiding spot is gradeable ("the seeker
   never sights the hider inside T") only with a seeker: the quadrupeds
   played it, refereed off the world (#167, #404). What stays ungradeable
   is the *quality* of a hiding spot — item 3, method — and the hider's win
   rate over many games is the nearest honest proxy.
3. **Method, and elegance.** By design the grader does not know how the work
   got there, so it cannot pay more for a better way. A challenge that is
   *about* the method ("open it without touching the lid") has to state the
   method as a world criterion (nothing of the robot's touches the lid, off
   the contact list, as a hold's criteria read it), or it is not gradeable
   here.
4. **Taste.** A predicate cannot say a tower is beautiful. That is the visitor
   tier's job and stays there.
5. **Anything whose success its author cannot write down in advance.** This
   is the real limit: the mechanism grades exactly what a person could
   specify before the attempt. A challenge that only becomes clear once
   attempted — most of what "arbitrary" meant in #45 — is a research question
   until someone can write its predicate, and writing it is the work.

## 9. Open the chest and set its lid down (issue #493)

Demo 1's verdict: #466's drop-handle chest (`activity/chest.py`), its
dynamics hidden and drawn, opened and its lid set down. `challenge/lid.py`;
proof in `tests/test_lid.py`. It is the first challenge graded on a
TRAJECTORY as well as a state (§6's item 1): a sampler with a step hook,
`lid.Watch`, reads the lid every physics step of the attempt and the hold.

**The criteria**, written before any robot learns the job: (1) the lid's
true angle reached 57 deg (1.0 rad, #469's "open"); (2) while it was open,
above 5 deg, it never closed faster than 1.0 rad/s, through the attempt and
the hold: lowered, never let fall; (3) 5 s after the call it is under 5 deg,
whether the catch holds it again (`caught`) said beside and gating nothing;
(4) nothing but the chest touches the lid, its bracket, its handle or its
knob during the hold: let go; (5) the call came within the task's time
limit of the attempt's start, where the harness makes it at the latest --
the TASK's data (`attempts.LIMIT_S`, handed to the grader as `limitS`;
`TIME_LIMIT_S`, 300 s, where none is): the humanoid's three times its
reference's slowest pass, 68 s (#493); (6) the
body's own watch counted no fall (`Body.falls`) from the attempt's start
to the end of the hold, whether the fall was used or an accident (Ben,
2026-10-25, #544). From 2026-10-17 a fall was lost time against the limit
and judged by nothing, and learned skills leaned on it: a hard pull swung a
door open and the body over with it, every held-out try of one door arm
(#535). A failed verdict says the fall first, then what the world shows
(`lid.fell`), and the job tells the robot the rule. Said beside, and
judged by nothing: its worst tilt (`Body.orientation`). What a fall is,
is the body's to say, so the grader names none: the humanoid's watch is its
trunk 35 deg past the posture its legs hold, by its IMU, or its pelvis sunk
-- so a bow its reach asks is no fall (#520). The
numbers are judged as measured and only shown
rounded; the call is wherever the attempt ended (Evaluation.md §4); the
reason never carries the catch, whose number is secret on the row.

**What the lines were read off** (`scripts/attempts.py --drops`, the 128
drawn lids on a bare floor): let go from rest 1 deg over the shut line,
every lid crosses it at 1.03-1.31 rad/s; from 2 deg over, 1.45-1.85; from 57
deg, 2.9-8.3. The quadruped's probe lowered each at 0.49-0.57 at its fastest
while open (its 128 records, #479). So the line passes a lowering with a
margin and fails a lid let go a degree or more over the shut line. Every lid
let go at the shut line is caught inside 0.07 s and one let go from 57 deg
inside 2.0, so the hold is 5 s. ⚠ A DROP CAN LEAVE THE CHEST OPEN: let go at
the top of its sweep, 57 of the 128 came to rest 8.5 deg open, propped on
their own handle, which swung back under the lid and wedged its knob against
the front wall. Such a lid fails (2) and (3) both.

Re-graded under it, the quadruped's probe (#466) passed all 32 of its
public chests at its first attempt: the reference row (Evaluation.md §4).

**The departure: `shown: false`.** Its row in `challenges.json` pays 0,
because a demo's attempt is practice graded by the world (Ben, 2026-10-10),
and it is a row no mind is shown (with §10's three, `TaskReward.shown`): a
mind is shown the rows its world's jobs name, no world offers a practice
row, and a row in the table is the deployed prompt changed. The served job
on the storeroom's chest is paid on a row of its own (`open_chest`, §11). ⚠
The grader imports nothing of the chest's: everything loads `scoring`, and
the chest's module holds its truth. Its parts are read by their names, and
`caught` is the catch's pull on the hinge.

**What it cannot grade**, beyond §6: a lid's ELEGANCE (a slam stopped a
hair over the shut line passes, and so does a lowering at 0.9 rad/s); and
whether the robot KNEW it had done it, which is its own test, beside the
verdict in the record and never read by it.

## 10. The hands' capability checks: a drawer, a door, a cube (issue #525)

Three graders beside the lid's, for the hands' vocabulary's gate
(docs/Hands.md §5): `challenge/drawer.py`, `door.py` and `place.py`, each
over an activity of its own (`activity/drawer.py`, `door.py`, `cube.py`)
whose dynamics are hidden and drawn, set out on the humanoid's counter
(`evaluation/counter.py`) and attempted through #493's protocol
(`evaluation/attempts.py`, `GRADERS`). Each is the lid's shape: a `Watch`
every physics step, a snapshot at the call and after a `HOLD_S` hold,
judged unrounded and shown rounded, the time limit the task's. Proof in
`tests/test_hands_check.py`.

**The criteria**, written before any procedure flew them:
- *drawer*: (1) out at least 15 cm at the call and still so 5 s later --
  its runners hold it where it is left; (2) nothing but the cabinet touches
  the drawer, its front or its knob during the hold; (3) in time.
- *door*: (1) open at least 50 deg at the call and still so 5 s later --
  an arm reaches in past its edge; (2) nothing but the cupboard touches the
  door, its lever or the lever's tip during the hold; (3) in time. Its
  latch frees the door only once the lever is pressed past most of its
  travel (`activity.door`): the grader reads none of it.
- *place*: (1) 5 s after the call the cube rests on the mat -- its middle
  within 2.5 cm of the mat's, its bottom within 4 mm of the mat's top,
  flat on a face within 10 deg, any face (`place.resting_tilt`: read off
  one, a cube set down upside down "tipped 180 deg"); (2) it never came
  down faster than 0.5 m/s, in the attempt or the hold -- set down, not
  dropped; (3) nothing of the robot's touches it during the hold; (4) in
  time. Slid there, it passes as well as carried.

And each, the lid's (6): the body's own watch counted no fall from the
attempt's start to the end of the hold (`lid.judged`, #544).

Said beside and judging nothing: the most the drawer came out and the door
opened, how far the lever was pressed, the fastest each moved, the cube's
lift and the body's worst tilt.

**The departure is the lid's: `shown: false`, paying 0** -- a check is our
instrument, offered by no world, and a row a mind reads would change the
deployed prompt; the served jobs are paid on rows of their own (§11).

**What they cannot grade**, beyond §6: a door pushed open by the open hand
rather than pulled by its lever passes -- the door is the job -- and so
does a drawer pulled by its front.

## 11. The hands' jobs in the house (issue #559, stage 3)

The hands' four checks (§9's lid, §10's three) set into the served
humanoids' house: on a counter against the east wall of the research
center's storeroom (`home.storeroom`), the chest, a cabinet of drawers, a
cupboard and the cube with its mat, each square to the counter where the
middle of its practice set-outs had it, and over each a 120 mm tag in a row
on the wall. Each is a job of its own -- `open_chest`, `open_drawer`,
`open_door`, `place_cube` (`tasks.HAND_JOBS`) -- its target its prop, so
they may all stand at once, and each is discharged by a procedure the robot
writes (`TaskKind.discharge = "procedure"`, #207): claiming queues nothing;
the robot writes the procedure, runs it and sets `done` to the job's id,
and only a mind with a library is offered one (`lifecycle.world_targets`).
Three are offered (cadence.json); the cube's is not, below.

**The criteria are the checks'**, through the same evaluators, with two
departures, both about where an attempt starts:

- *from the CLAIM.* The watch the checks run from an attempt's start runs
  from the claim, through the walk there, to the end of the hold after
  `done` (`HubLifecycle._watch_job`, `_grade_hands_routine`): a fall
  anywhere in it fails the job (#544's rule), and so does a lid let fall or
  a cube dropped before the call. Its readings ride a restart with the
  world they were read off (`kept_state`'s `jobWatches`): a lid opened
  before one is still a lid that was opened.
- *in the job's MINUTES.* The time limit is the offer's own term
  (`minutes`, cadence.json), counted from the claim: 16, three times the
  slowest pass of the references over a tour of the house (313 s, from the
  station), rounded up.

**The offer says where by the area's terms** (#419): the facility's
address, and directions that name the storeroom, the counter, the row's
four tags and the handle's own -- never a position. **Each offer draws its
prop's hidden dynamics afresh** (`TaskKind.draws`: the board's own
sequence number, `Task.secret["draw"]`), set into the compiled world as the
offer lands with the prop set back out (`HubLifecycle._set_out_counter`,
`storeroom.redraw`, `storeroom.set_out`), its activity begun again over
the draw's truth (`renew`), and put back after a restart
(`restore_counter`): a skill over a kind of drawer, never one drawer. ⚠ No
offer lands on a prop a robot is touching (`_worked_props`, the
producer's `busy`): set out from under a hand, a door was clamped shut
round it, and a drawer left out was a job paid for nothing.

**Rows of their own.** Each job is paid on a row of its own in
challenges.json (`open_drawer` and the rest), graded by its check's
evaluator, while the check's row stays the practice's, unpaid. The served
rows are `shown: false` too, so a table that names no world's jobs never
carries them; a mind is shown the rows its own world's jobs name
(`RewardTable.as_context(kinds=)`, `overseer.world_kinds`), so the
humanoids see theirs.

**Each has its hand-written solution** (`challenge/solutions.py`, Evaluation.md
§7), two procedures in the robot's own vocabulary, because one would pass
the language's 1,500 characters: the walk there, then the check's own
procedure -- unchanged for the door and the chest; the drawer's pulls 4 cm
further and lets go of the knob before its arm goes home. A job is offered
once its reference passes 6 of 8 over a tour of the house, the bar written
before they flew (`solve.py --from tour`; SimNotes, "The humanoids' first
jobs"): the drawer 8, the door 8, the chest 7. Each job's energy is its
reference flown from the station (`energy.json`). The cube's reference set
it down 2-7 cm off its mat's middle all 8 times, so `place_cube` is
defined, graded and priced but not offered.
