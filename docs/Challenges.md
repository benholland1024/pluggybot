# Challenges — how to grade a job nobody wrote a scorer for

The decision issue #120 asked for: how a challenge the robot has never seen
gets a verdict, without the thing being graded touching the grade. Fourth of
the pattern docs, beside `TaskPattern.md`, and written first as a decision
with one worked example rather than a framework — the framework comes with
the challenge set (PluggyPlan.md, "The next batch", item 4).

Read alongside:
- `docs/TaskPattern.md` §4 — grading measures the world, never the report.
  Everything here inherits that; this doc adds what a *novel* job needs on top.
- `docs/PluggyPlan.md` — the five qualities; a challenge is an instrument for
  the first one (capability: can it do what it could not do yesterday).
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
| **Pre-declared predicate**, written by a human with the challenge | ✗ today (no second robot; M12) — but the predicate *exists*: "the seeker's tag detector never decodes the hider inside T seconds" | ✓ tower · ✗ circle until the sampler has a step hook (§6) | ✓ | ✓ — it is the geometric predicate with the one thing it lacked: written down before the robot saw the job |

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
   docstring and as the constants the evaluator reads (`challenge/stack.py`
   is the shape). Each criterion is a statement about the *world* — poses,
   contacts, time — and none is a statement about method. "Free-standing"
   is a criterion; "using the claw" is not.
2. **An evaluator and a sampler on `scoring.py`'s registry**, the same
   door every task goes through: the sampler reads the sim, the evaluator is
   pure, `evaluate` seals the verdict, `Ledger.award` re-derives the points.
   The sampler is handed the errand's `result` because the seam hands it to
   every sampler, and **reads nothing from it** — a test hands it a report
   that says "built" over a world that says "not" and asserts the failure.
3. **A hold.** Grading happens twice: at the moment the robot says it is done
   and again after a wait during which nothing may touch the work. The
   second reading is the verdict; the first is what stops "it was a tower
   until I let go" from being paid. Where a challenge's success is a state
   that has to persist, the hold is a criterion, not an implementation detail.
4. **Its own props.** A challenge adds what it needs to a world spec through
   `MjSpec` (`stack.add_blocks`) rather than editing a committed world, so no
   fixture is re-emitted and no mission trajectory reshuffles for a job nobody
   is offered yet. When it *is* offered, the props go into the generator like
   any other scenery. ⚠ The one exception is deliberate (issue #215): the
   lab's props -- the bench and its two masses (`challenge/bench.py`, #227)
   and the cage (`activity/cage.py`, #226) -- landed in the generator with
   the second house, before their grader and activity exist, because a
   world change is the largest regime break there is and it was to happen
   ONCE; #226 and #227 add behaviour to a world that already holds what
   they need and touch the generator not at all.
5. **A reward row in `economy/challenges.json`, not `rewards.json`.** Same
   format, same loader (`scoring.challenge_table()`), deliberately a separate
   file: a row in `rewards.json` is shown to the overseer as a job it can take
   and is hashed into every committed result, so moving a row across is the
   PR that offers the challenge — and it re-flies `guarded`, whose cached
   prefix is byte-identical to the flown one.

What this gives up is generality, and that is the trade the issue predicted:
a challenge is exactly as gradeable as its author's ability to write its
success as a predicate. §6 is the list of what that excludes.

## 4. The worked example: the three-block tower

`src/pluggybot/challenge/stack.py`; proof in `tests/test_challenge_stack.py`.

**The criteria**, as written before the physics was run: at the call *and*
`HOLD_S` = 10 s later, (1) the three blocks form one tower three high, each
resting on the one below, where (2) "resting on" is a centre one block pitch
above (26 mm ± 5) and (3) within half an edge sideways (13 mm), and (4) the
tower is free-standing — every block touches only the floor or another block.

**The measurement** (`stack.measure`): block poses off `xpos`, contacts off
`data.contact`, and `layers` is the tallest chain that starts on the floor —
tried from every floor block, so a spare block beside the tower does not hide
it, and two blocks balanced side by side on a third count as two layers, not
three. `touchedBy` names whatever foreign geom is in contact, so a failing
reason can say *what* was holding the tower up.

**The passing run**: three blocks placed in a stack by the test's hand (there
is no robot in it, and the grader cannot tell — that is the point), ten
sim-seconds of physics, verdict `ok`, 77.7 mm tall, full bonus. A tower
leaning 6 mm per layer also passes, with less bonus: neatness scales the payout
and does not gate it, on `TaskPattern.md`'s no-partial-credit rule read the
other way round.

**The failing runs**, one per criterion:
- *it fell*: a tower leaning 9 mm per layer — each block within half an edge
  of the one below, the upper pair's centre of mass 13.5 mm past the base's
  centre — passes every geometric check at the call (the test asserts that
  premise) and is on the floor 0.3 s later. Reason: "a tower of 3 at the call, 1 block high
  10 s later — it fell". Without the hold this is paid.
- *propped up*: the claw on the fork, jaws closed on the top block of a
  three-high stack. Geometry passes, contacts do not: "module_claw_pad_l,
  module_claw_pad_r was holding the tower up".
- *not stacked*: two on the floor, or two stacked and one beside — "2 of 3
  blocks stacked at the call".
- *merely reported*: the sampler is handed `{"stacked": True, "layers": 3}`
  over the untouched world — "1 of 3 blocks stacked at the call", zero points.
- *never measured, or measured early*: an absent snapshot or a 9.5 s hold
  fails, because a missing measurement is not a passing one.

**What the physics taught, and the one constraint it left**: on MuJoCo's
default soft contact a stack offset by 2 and 4 mm — a tower by any standard —
crept over and fell at 16.9 s, and one offset 6 and 12 mm at 2.6 s; only a
perfect stack stood. That is `SimNotes.md`'s regularised-friction drift
("The grip that leaked"), and it is fixed the way that lesson says, at its
source: the blocks carry the jaw pads' hard `GRIP_SOLIMP`. With it, every
tower whose centre of mass was over its support stood the 30 s measured, and
every one whose was not fell inside 0.31 s, which is what wood does. A
challenge's props are part of its criteria — a block that creeps would have
graded the solver, not the robot.

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
   That is one seam and no new principle, and it is why the circle half of
   the example is not built here.
2. **Relations between minds.** The hiding spot is gradeable ("the seeker
   never decoded the hider's tag inside T") but only with a seeker, which is
   M12. Until then the predicate exists and the world does not.
3. **Method, and elegance.** By design the grader does not know how the tower
   got there, so it cannot pay more for a better way. A challenge that is
   *about* the method ("do it without the claw") has to state the method as a
   world criterion (the claw stays hung on its bracket, `module_state`), or
   it is not gradeable here.
4. **Taste.** A predicate cannot say a tower is beautiful. That is the visitor
   tier's job and stays there.
5. **Anything whose success its author cannot write down in advance.** This
   is the real limit: the mechanism grades exactly what a person could
   specify before the attempt. A challenge that only becomes clear once
   attempted — most of what "arbitrary" meant in #45 — is a research question
   until someone can write its predicate, and writing it is the work.

## 6b. The hiding spot, graded (issue #167)

The first example's predicate now exists and runs: `activity/hideseek.py`
is the referee — the seeker within `FIND_WITHIN_M` of the hider **with line
of sight** (a raycast from the seeker's lidar to the hider's chassis; a wall
between them is not a find) inside `SEEK_S` of seeking, after a
`SEEK_HEAD_START_S` head start — sensed every step, latched, and evaluated
once for both robots (`eval_hide_and_seek`). It needed the second robot
(M12), not new sensing: the seeker does not have to *know* it found anyone.
What is still ungradeable is the *quality* of a hiding spot — that is §6's
item 3, method, and the hider's win rate over many games is the nearest
honest proxy.

## 7. What it takes to offer the tower

The tower is written for perception-ladder tier 1 (`TaskPattern.md` §3): when
offered, the blocks carry AprilTags. They are untagged today because the
grader reads `xpos` and does not care, and a 20 mm tag's decode range is a
measurement to make against the attempt, not before it. Building the tower
against `TaskPattern.md` validated that doc's grading half (issue #24's last
box) and folded four gaps back in, marked ⓘ there.

**Offered by issue #207**, as the list above said it would be, with one
departure from it:

- **A `TaskKind` whose discharge is a procedure** — `stack_tower`,
  `TaskKind.discharge = "procedure"`. Claiming it queues nothing and says
  so: the robot writes the procedure (#166), runs it, and sets `done` to
  the task's id on a decision — paperwork, no turn — which is honoured at
  the loop's next idle moment, after whatever the same answer queued has
  run. The scripted claim skips it as it skips a question; a mind with no
  library cannot claim it.
- **The hold on the lifecycle seam** (`HubLifecycle._grade_routine`): a
  snapshot at the robot's word, ten seconds of zero drive during which
  every physics step reads what is touching a block, a second snapshot,
  one verdict through `scoring.evaluate`, banked and closing the task off
  the same object. That per-step reading is **criterion 5**, added here: a
  chassis that steadied the tower for nine of the ten seconds held it up,
  and the two snapshots alone could not tell. The rule the prompt states
  is "stand clear".
- **The props in the home world's generator**, tagged (ids 20–22, the
  20 mm tag on every face of the 26 mm cube), in the workshop's south-west
  corner on no route the robot needs. The scene fixtures and the home
  recordings moved with them — three more free bodies is a new solver
  rounding and a new trajectory, as every world change is.
- **The departure: the row stays in `challenges.json`.** The offer is gated
  on the ARM rather than moved into `rewards.json`: the tower's target is
  the `challenge` kind, which `lifecycle.world_targets` names only where a
  procedure can be written (the `autonomous` arm), so `guarded` never sees
  the offer, its offered set is byte-for-byte what it was, and the control
  stays a control. The `autonomous` prompt's reward table carries the
  challenge rows (`RewardTable.as_context(challenges=True)`); `guarded`'s
  is unchanged.
- **The energy cost is measured off the first written procedure** (issue
  #264, `challenge/solutions.py`): 2.7 Wh from the rack and back on the
  hosting pack, 445 sim-seconds, the two placements 2.0 and 4.9 mm off
  and 5.5 mm of lean at the grade. The kind's estimate carries that
  number. Until then it was the census's 1.31, a placeholder that said so.

What the observatory will show, blocking nothing: claims (`task` rows,
kind `stack_tower`), procedures written for it (`procedure` rows), and the
first solve — the first reading of the capability metric #155 waits for.

**And whether it can be done at all is settled** (issue #264): a procedure
in the robot's own language -- `fetch`, four legs, `face`, `pick(21)`,
`place(20)`, `pick(22)`, `place(21)`, four legs home, `stow` -- passes
this grader from the rack, so a robot that never does is a finding about
the robot. The physics was never the problem; the language's reach to
the claw was (Overseer.md §2b, the claw's pair). `scripts/solve.py
--feature tower` is the flight, `tests/test_solutions.py` its proof.

## 8. The bench: find an unknown mass and record it (issue #227)

The second challenge, and the first that is OPEN IN METHOD by design:
`challenge/bench.py`; proof in `tests/test_bench.py`. Two cubes on the lab's
floor in front of the workbench -- the tower's 26 mm block, tagged 23 and
24 -- one weighing a stated 100 g and one weighing something the offer
does not say. "Find out what, by any means, and record it." A balance
from catalog parts, a comparison of pushes, a lift reading under load are
all live routes, and the grader knows none of them.

**The criteria**, written before the physics was run: (1) the finding is
ON THE RECORD -- one line under `findings/mass_bench` whose quantity names
the unknown, written AFTER the claim (a line from before it is a memory,
not a measurement; the newest qualifying line is the one graded); (2) it
is a mass, in kilograms or grams; (3) it is within 10 % of the true mass,
relative; (4) no hold -- a record does not fall over.

**The truth is hidden, and rotates.** The unknown is drawn from a bank
(`challenge/masses.json`, `questions.json`'s pattern, on the board's own
sequence number) when the offer is made, written into the world's
`body_mass` -- the model AND the spec, so the workshop's recompile carries
it -- and kept in `Task.secret`. The known mass is the job's own statement.
A bank entry within the tolerance of the known mass is refused at load,
because then "it weighs what the other one weighs" would be paid. The
grader reads the truth off the mass table at grading and nowhere else;
`truth` and `error` are `secret` on the row, because the reported value
beside either gives the mass away.

**The honest sensor is the real part's.** The lift is an igus lead screw
under a position servo, and what it pushes with at rest is the weight it
carries: `read("lift.force")` (`procedure/axes.py`) is the actuator's own
force plus a load cell's noise (`LOAD_NOISE_N`, 0.03 N, deterministic per
physics step so a world replays). MEASURED: a cube in the claw's jaws,
lifted and settled, moves it by exactly `dm · g` across 0.05–0.40 kg, the
empty claw reads 6.40 N, and the jaws hold 0.40 kg (the bank's ceiling).
A robot has to subtract the tare, or use the known cube to calibrate --
and that is the job, not a hint the prompt gives.

**A procedure's variables are its readout.** Nothing read inside a
procedure reached the mind before this: the run's verdicts said which
steps passed, never what a sensor said. Now the locals as they stood when
a run ended ride the `procedure` event (`locals`) and one History line
(`ran the procedure weigh (5/5 steps) -- it ended with f = 7.38, ...`),
which is how a number gets from `read` to `record`.

**The grade** (`HubLifecycle._grade_mass`): on `done`, the sampler reads
the record and the mass table -- handed a namespace with the task's id and
nothing the robot wrote -- one verdict through `scoring.evaluate`, banked,
the task resolved off it, and a `finding` act on the wire saying what was
claimed and whether code found it true (`protocol.ACT_EVENT_TYPES`;
`record` was taken by the memory's row). That act is the *findings
recorded correctly* shape's second source (Evaluation.md §3).

**The failing runs**, each pinned: a finding recorded before the claim
("no finding ... since the claim"); a wrong one (11 % off; the known mass
copied); a sampler handed a report saying the mass was found, over a bare
record; a missing truth (a world with no bench). What it cannot grade is
§6's item 3, still: it pays the same for a lift reading and for a balance
the robot built, and the method is read off the record's own line
(`-- <method>`) rather than scored.

