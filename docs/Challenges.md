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
   any other scenery.
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

## 7. What it takes to offer the tower

The tower is written for perception-ladder tier 1 (`TaskPattern.md` §3): when
offered, the blocks carry AprilTags. They are untagged today because the
grader reads `xpos` and does not care, and a 20 mm tag's decode range is a
measurement to make against the attempt, not before it. Building the tower
against `TaskPattern.md` validated that doc's grading half (issue #24's last
box) and folded four gaps back in, marked ⓘ there.

Not done here, and listed so the next PR knows its shape: a `TaskKind` whose
discharge is not an errand but a procedure the robot writes (PluggyPlan.md
batch item 3 — the tower is the first job with no `Errand` behind it); the
hold on the lifecycle seam, with the robot told to stand clear; the props in
the hub world's generator; the row moved into `rewards.json`, which re-flies
`guarded`; the energy cost measured (`scripts/energy_spike.py`), which cannot
happen until there is a procedure to measure.
