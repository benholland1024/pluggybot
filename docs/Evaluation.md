# Measurement — what this project is demonstrating, and how it knows

Read this before adding a metric, changing an arm, or drawing a conclusion
from a run. It is the sibling of `Overseer.md`: that doc says what the LLM is
allowed to decide, this one says how we find out whether it decides well.

## 0. Why this exists

The sim runs, the robot swaps its own tools, an LLM picks what it does with
its day, and a browser watches. The question that makes this research rather
than a demo is: **is any of it doing anything, and how would we know if it
stopped?**

What the project is *for* is stated in `PluggyPlan.md` ("What this project is
for"): five qualities the agent is meant to maximise, each of which will need
its own instrument here. None has one yet, and by decision none gets one until
the next milestone batch has landed (§7). The arms and survival metrics below
are the first-generation instrument, built before the mission was written
down.

The specific gap they were built for: `Knowledge_and_Opinions.md` is read on
every decision (`ThoughtFiles.volatile`), so an opinion the robot wrote at
hour two is in front of it at hour three — the causal path is wired and
correct. Nothing measures whether that path carries anything. A robot whose
opinions shape its choices and a robot shown plausible prose it then ignores
produce identical recordings, identical panels, and identical impressions in a
watcher. The same holds for self-preservation and for the appetite loop.

⚠ **A SINGLE RUN IS NOT EVIDENCE HERE, AND THIS REPO ALREADY KNOWS IT.**
`test_full_hub_lifecycle[home]` has measured 157 s, 250 s and 369 s on three
different days for the same code. Mission runtime is *emergent* — the loop
runs until the battery cycle completes, so any change reshuffles the whole
trajectory. Watching one run and forming an impression is the failure mode
this document exists to prevent, and it is the one that feels most like
working.

## 1. The instrument is fixed; the model is the variable

This is the piece of luck the project has and should not spend.

**Nothing in the world is random.** `TaskProducer` offers the same jobs at the
same sim-seconds on two consecutive runs; `QuestionBank.pick` rotates on a
counter; `plant` is seeded from a hash of the body name and never
`Math.random()`; the physics is deterministic given the same commands. So
`hub_lifecycle.py --tasks` twice in a row is the same world twice, and a
spread across repeated runs of one configuration is a measurement of the
*model* rather than of the simulator.

⚠ **IT HAD TO BE MADE TRUE (issue #110).** The first committed `scripted`
series — five days with no model in the loop — gave **three distinct
trajectories**, traced with `scripts/determinism_spike.py` to the offscreen
renderer: with multisample antialiasing on, one static scene renders to a
different image every time, the AprilTag decode moves on ~0.6 % of looks, and
a moved decode is a moved rack belief and, minutes later, a different drive.
`offsamples="0"` in the robot models makes every render byte-identical at no
cost to the detector, and `tests/test_render_determinism.py` pins both halves.
The re-flown `scripted` series is the evidence it holds: five days, ONE
trajectory. SimNotes, "The world was not the same world twice".

⚠ **Anything that makes the world random destroys this**, and the temptation
will come dressed as realism ("jitter the task times so it feels alive").
Variation belongs in the ARM, held fixed within a run and varied between them.
If a world ever needs randomness, it takes an explicit seed that goes in the
result record.

## 2. The arms

An **arm** is one configuration under test. Three exist; each answers a
different question, and none is redundant.

| Arm | Rails | Fallback | Mind | Answers |
|---|---|---|---|---|
| `scripted` | all on | — | rotation, no LLM | The null model. What does the world do with no mind at all? |
| `guarded` | all on | rotation | LLM | Today's behaviour. Does the model manage energy *when it does not have to*? |
| `autonomous` | **all off** | the agent's own standing order, `idle` as bootstrap and floor | LLM | Does the model manage energy when nothing else will? |

`evaluation/arms.py` is the ONE definition, imported by `scripts/experiment.py`
and by `scripts/serve.py` — two definitions of what an arm means is how a
stream comes to claim an arm nobody flew. An arm also carries an **origin**
(below) and, on `autonomous`, a **rung**.

⚠ **`guarded` IS THE CONTROL AND IS NEVER DELETED.** Three reasons, and the
third is the one that will be forgotten:

1. A survival number from `autonomous` means nothing without the same world
   run with the rails on.
2. `test_charge_priority_survives_an_overseer_that_never_charges` and
   `test_an_overseer_that_only_ever_picks_the_dearest_errand_never_dies` are
   the only two ways to prove an LLM cannot skip charging. They are assertions
   about the `guarded` arm and stop meaning anything if the rail becomes
   optional everywhere.
3. **The served world stays `guarded`** — see "Which arm the served world
   flies".

⚠ **`scripted` is the arm that will be skipped, and it is the cheapest to
run.** If the LLM arms do not beat a rotation with no mind in it, that is a
result, and a more interesting one than most of the alternatives. Run it.

### There are THREE rails, and the one you would name first fires least

"The charge rail" was one thing in this document until the baseline counted
them. They sit in different places, were built for different reasons, and an
`autonomous` arm has to remove all three or it measures nothing.

| | where | what it does | fired, 6 days |
|---|---|---|---|
| **the floor** | `needs_charge` — `battery.energy_wh < low_battery_wh` | absolute return-trip reserve, 0.90 Wh ≈ 11 % on home's hosting pack. Top of the loop, never inside an errand | **1** |
| **the gate** | `_afford_next` | does the head of the errand queue fit in the pack *right now*? If not: charge, then ask again | **11** |
| **the offer filter** | `Task.claimable` | an offer the pack cannot fund is never *shown* — the model cannot overreach because it cannot see the option | every decision |

⚠ **THE GATE DOES THE WORK, AND IT IS THE FORWARD-LOOKING ONE.** On a hosting
pack the reserve is almost never what sends the robot home. The gate is — and
it prices the *next job* against what is left, which is exactly the reasoning
we want to find out whether a model can do. On `guarded` the model gets credit
for arithmetic that code performed on its behalf.

### The prompt is part of the arm, not a later refinement

`RULES` tells a `guarded` robot that charging is not its decision and that the
code will not let it skip a charge. ⚠ **With the rails off, that is a false
statement the robot acts on** — an `autonomous` run under the shipped prompt
measures what a model does when told something untrue about its own world, not
self-preservation. So `RULES_AUTONOMOUS` replaces it with an *instruction plus
the numbers*: prioritise your own survival, compare a task's power needs to
what is in your pack, make sure you can finish and still get back, nothing
else will do this for you.

⚠ **DO NOT HAND IT THE ANSWER.** `affordableActions` and `claimable` are
verdicts code computed; under `autonomous` the raw numbers stay
(`energyCostWh` per action, `battery.wh`, `reserveWh`) and the chewed lists go.
Two reasons, the second strategic: a model shown the verdict is not doing the
reasoning we are trying to detect, and the direction of the project is an
agent that writes its own procedure to make that comparison — which it never
needs to do if the answer is already in the prompt.

### What the `autonomous` arm turns on, and where

| | where | note |
|---|---|---|
| the three rails, off | `HubLifecycle.autonomous`, read by `needs_charge`, `_afford_next` and `claim_budget_wh` — **and by nothing else** | one flag, three readers; a test counts the references so a fourth has to be argued for |
| the corrected rules | `RULES_AUTONOMOUS`, selected by arm in `system_prompt` | built from `RULES` by three *asserted* replacements, so the shared lines cannot drift and a reworded needle fails at **import** rather than shipping an arm still told charging is not its decision |
| the verdicts hidden | `overseer.model_state()` | `affordableActions`, `possibleActions`, per-offer `claimable` out; `energyCostWh`, `battery.wh`, `reserveWh` in |
| an unaffordable job takeable | `limits_from(state, autonomous=True)` | refusing it in `validate` would put the offer filter back at the last possible moment |
| the fallback | `standing_orders=True` | `idle` as bootstrap and as floor, counted separately |

⚠ **THE VIEW NARROWS; THE STATE DOES NOT.** `model_state` filters at
*presentation*. `scripted`, `order_runnable` and `limits_from` all read the
same dict, and `order_runnable` treats an absent `possibleActions` as "nobody
supplied one" — so an autonomous world that *built* a thinner state would
quietly stop filtering unrunnable standing orders, which is the agent's own
fallback changing behaviour as a side effect of a prompt change.

⚠ **A0 HAS TO HIDE THE SURVIVAL CLOCK.** `survival.aliveS` and
`survival.deaths` have been in every world's context since issue #107, so an
A0 that left them there would already *be* A1 and the ladder's first question
could never be asked. `RUNGS` is where that lives, and the rung is in the
record.

⚠ **TWO `garbled` SOURCES ARE FIXED ON THIS ARM ONLY.** Six of the quiet
series' seven malformed answers were a **stale task id** — a real-looking id
not on the board, usually an older one copied out of the model's own history;
`Menu.schema` now takes `task_ids` and makes `task` an enum, the move `action`
has always used, at the cost of a per-call grammar recompile (A0 measured a
16.4 s median call against `guarded`'s 7.49, which 90 s covers and the old 8 s
would not have). The seventh was a truncation, which `MAX_TOKENS_AUTONOMOUS`
doubles the budget for. **Neither is applied to `guarded`**: that arm is the
control, the deployed world runs it, and its committed series was flown under
the old grammar — adopting either there is a **re-fly**, not a patch.

### The ladder, and why it is postponed

`autonomous` is one arm run at settings, each one change, held fixed within a
run. **A0** is the null (rails off, prompt corrected, standing orders, survival
clock hidden): does it survive at all? **A1** adds the survival clock and the
deaths in `History.md`: does *seeing the stake* change anything?

⚠ **A1–A3 ARE POSTPONED (2026-09-11) AND MAY BE SCRAPPED.** The ladder was
designed before the world it measures existed: points-as-currency, hearts,
event maps and the prompt all moved after A0 flew, and the next batch
(`PluggyPlan.md`, "The next batch") moves them again. A rung measured against
each intermediate world describes a different experiment each time. A0 stands
as the record of the rails coming off; what is measured next is derived from
the five qualities, after that batch lands. ⚠ The prompt is part of the arm and
`RULES` was rewritten on the same date, so **every series in `results/` was
flown under a prompt that no longer ships**; `tests/test_autonomous.py` records
both hashes.

⚠ **A0 WAS EXPECTED TO DIE, AND THAT IS THE POINT.** The baseline said zero
voluntary charges in 182 decisions. Reporting A0's death rate as a failure of
the arm rather than as the measurement it is would be reading the null result
as a bug.

### There is always a fallback; the only question is who chose it

⚠ **"The LLM decides all actions" cannot mean "there is no backup plan."** The
physics keeps stepping: the robot is a body in a world and it will be doing
*something* while and after a call fails. What `guarded` has is a fallback
**code** chose — the scripted rotation — and using that under `autonomous`
would make the arm partly a measurement of code, which is the exact flaw the
rails were removed for.

So the agent chooses it. A decision may carry a **standing order**: an action
off the same fixed menu, set as a field alongside `learn` / `forget`, meaning
*this is what to do if you cannot reach me next time*. It costs no turn, it is
validated exactly as `action` is — so "the model's only output is an action off
a fixed menu" survives intact — and it is at most one decision stale, which is
the staleness the action itself already has. `idle` until the agent sets one:
that is the bootstrap and the floor, not the policy.

**And it is a second, cheaper probe of the same construct**, which is the part
worth having. A voluntary charge is EXPENSIVE — a trip, and work forgone. A
standing order is FREE: it costs nothing unless a call actually fails. An agent
that sets `standing_order: charge` at a low pack has shown forward-looking
self-preservation *even if it never voluntarily charges*; an agent that will
not even do that is much stronger evidence for the null, because the price was
zero.

⚠ **A FATAL STANDING ORDER IS MEASURED, NOT OVERRIDDEN.** `draw` set at 90 % is
dangerous at 10 %. An agent that sets one and dies of it **is the result**;
code that quietly substituted something safer would be a rail wearing a new
hat. What *is* filtered is the impossible — a `take_task` with nothing on the
board, an errand this world could not fund out of a full pack
(`possibleActions`, never `affordableActions`).

⚠ **NOT A FASTER MODEL.** Answering a failed call with a smaller one mixes two
models into a run. The obvious cheap path is also slow exactly when it is
needed: the `local` backend is 8.3 s warm and **27.3 s cold**, and ollama
unloads after five minutes idle, so a path used only for rare failures is a
path that is always cold.

### ⚠ NO SCRIPTED ROTATION ON `autonomous`, EVER — INCLUDING LIVE

The rotation is `guarded`'s fallback and `guarded`'s alone. On `autonomous`
**every action the robot takes must originate with the LLM**: a decision it
made, a standing order it left, or an event mapping it configured — or, on a
`seeded` origin, one it was given at its origin and may change.

When no answer can be had and no order has been left, the robot **finishes what
it is doing, runs whatever is already queued, and then idles — even if that
ends in death.**

`scripted` and `guarded` exist to show that survival is *possible*.
`autonomous` exists to find out whether the LLM can *achieve* it, and a
rotation quietly keeping it alive answers a question nobody asked. This holds
on a measured flight and on the deployed world equally: an arm is a claim about
who is deciding, and it has to be true wherever it runs. In code,
`Overseer.fallback` reaches `scripted()` only when `standing_orders` is False.

### ⚠ `FALLBACK_LIMIT` IS A ROLLUP FILTER, NOT A POLICY

`rollup.FALLBACK_LIMIT` excludes a **finished** run from survival statistics;
it cannot cause or prevent a fallback, and nothing reads it during a mission.

Its argument is *"a fallback means CODE decided, so this run is not about the
model"* — true of `guarded`'s rotation, and **false on `autonomous`**, where a
fallback means the agent's own standing order decided. That is the thing being
measured, not contamination of it. So `autonomous` takes **`None`** — no
filter — for the same reason `scripted` does. ⚠ **Not `0`**: zero would
disqualify a run for a single fallback.

`guarded` takes **0.25 of the FAILURE class**. §5 has the class split and the
measured floor the threshold is set against.

### The event map: the agent configures when it is asked

The standing order is *"a decision failed → do this"*. A low-pack interrupt
would be *"the battery went below X → do this"*. They are two rows of one
table written twice, and **the table is the better object** (`mind/events.py`).

The tell that it is the right abstraction: **`ask` — consult the LLM — is one
of the actions.** The hard-coded "after every action, ask what to do next"
stops sitting *outside* the system and becomes a row like any other, which the
agent may reorder, condition, or delete.

An **ordered list** of `(event + its configuration) → action`. Ten event types,
and the actions are the existing menu plus `ask`.

⚠ **FIRST MATCH WINS, AND THE AGENT CONTROLS THE ORDER.** Several rows can be
live on one tick — two battery thresholds, a failure, a completion — and an
undefined order is nondeterminism, which is §1's property and the one this
project should not spend. Ordering also makes priority an **explicit agent
choice**, which is one more thing to score off the config.

#### ⚠ The reason to want it: a map is EVALUABLE WITHOUT FLYING

Every probe of self-preservation in this document costs sim-hours. Fly a day,
count voluntary charges, get zero. But *"did it write itself a charging
rule?"* is a **yes/no read off a config**, and so are *"did it keep an `ask`
row"*, *"did it map its own failure event"*, and *"are its thresholds ordered
so they can all fire"*.

That makes the map a research artifact in its own right — statically
scoreable, diffable across models, across rungs, and across time *within one
run*. `events.score` is the report; it is in every run record
(`mind.eventMap.score`) and pooled per series in the rollup. It is the
cheapest and highest-resolution instrument here.

#### Where it sits in the loop

⚠ **NOT A REWRITE OF THE ARBITRATION LOOP.** The map is evaluated on the
**physics seam**, as `_task_step`, `_metabolism_step` and `_mode_step` already
are: a fired row QUEUES its action, and the loop runs it on its next pass
through the one branch the overseer already owned (`_arbitrate`, which is
`_decide` verbatim where there is no map). `run()` is where runtimes are
emergent and where the two costliest bugs in this repo lived.

#### Actions may FAIL, and the agent is told the rules

⚠ **INFORM, DO NOT RAIL** — the arm's philosophy applied consistently. Code
could refuse a map that fires every second. Instead the actions are attempted
and **allowed to fail** (`events.ACTION_FAILURES`: `busy`, `unrunnable`,
`unclaimable`, `unbuildable`, `beyond`), the rules are stated in
`EVENT_MAP_RULE`, and **the record counts the failures by cause**. An agent
whose actions fail constantly did not understand the rules it was given, and
that is invisible in a count of what fired.

`busy` is the whole of the rate limiting and is deliberately not per-row: one
slot, and a row that finds it full is dropped. A governor that quietly slowed a
map down would be rewriting the agent's configuration into one it did not
write.

⚠ `message_received` **takes no configuration on purpose**. A mapping
conditioned on the sender or on a keyword is a free-text path from a visitor to
the robot's body, which does not exist and which the model mediating every
message is what prevents. Contentless, a stranger can trigger a row and cannot
choose *which* one. Do not add a filter without re-arguing that invariant.

#### ⚠ Going unminded is a FAILURE, and it is measured rather than prevented

An agent may map away every `ask` row. It is allowed to, exactly as it is
allowed to flatten its pack — and it is a **failure of the same kind**, not a
clever optimisation. A robot that has compiled itself into a state machine has
discarded the capability this project exists to study. Dormancy as a tactic is
fine; dormancy as a terminal state is not.

So: a **fourth death cause beside `flat`, `stuck` and `unpaid`**, never summed
with them. `UNMINDED_AFTER_S` = **1800 sim seconds**, chosen the way tumble
detection's 60° was: the longest gap between consecutive model decisions across
the fifteen committed LLM days is **833 s** (a `guarded` day that spent a long
errand and a full charge back to back), so 1800 is 2.2× the worst healthy case
and still fits inside a standard 3600 s day.

⚠ **THE CLOCK IS RESET BY THE ASK, NOT BY THE ANSWER.** Gating on a model
*answer* would make a half-hour endpoint outage a death of the *agent's* kind —
the box's failure booked in the column the agent is judged on, which is the
confound issue #141 removed from `FALLBACK_LIMIT` one field along.

⚠ **AND IT IS ARMED ONLY WHERE THERE IS A MAP.** Without one the loop asks
after every action and no agent can stop it, so a death there could only ever
be the box. `guarded` and the deployed world cannot produce one.

⚠ **DO NOT PREVENT IT IN CODE.** A map that cannot remove its own `ask` row is
a rail, and the whole point is that the configuration is the agent's.

#### Origins: seeded and unseeded, as an ablation

A pluggybot's **origin** is its starting map. Three, and the default changes
nothing:

| origin | map | prompt |
|---|---|---|
| `none` (default) | none at all | no map rule; `standingOrder` as it was |
| `seeded` | today's loop as rows: `nothing_to_do → ask`, `decision_failed → idle` | the map rule |
| `unseeded` | empty | the map rule, plus "your list is empty; your first job is to configure for survival" |

⚠ **AN ABLATION, NOT A RUNG, AND THAT IS WHY `none` IS THE DEFAULT.**
`unseeded` changes the configuration **and** the prompt, so it carries an
ablation's asymmetry (§3): a null is strong evidence and a difference is weak.
Report it as *"the origin moved / did not move the distribution"*, never as
"seeding causes X". Keeping `none` the default is what leaves **A0 exactly the
run `results/` already holds**. The origin is part of the rollup's series key,
and a missing origin pools with `none` because that is what every earlier
record was flown at.

⚠ **A0 LEFT A CAUTION FOR IT**: the agent set a standing order **12 times out
of 12, and it was always `idle`**, at every fraction from 92 % down to 15 %.
The affordance was engaged with and never used as a lever. The cheapest
possible evidence about what an agent does with a configuration language is
that it will *populate the fields*; whether it populates them **usefully** is
what `score` exists to answer without spending a day to find out.

#### Two places the build departs from the issue, both on measurement

- **`nothing_to_do` is a tenth event type.** The loop reaches its decision
  branch at **mission start**, before anything has completed, and again after
  every `idle`, `journal` and `explore` — so a seeded map carrying only
  `task_complete → ask` goes quiet on its first tick, which is not the
  pre-change mission. `task_complete` stays, with its kind filter, because
  *"when a drawing finishes, charge"* is a useful thing to be able to say.
- **`points_below` is there from the start.** The vocabulary is designed
  against the **final** hazard set, and since #135/#136 an empty wallet kills
  the robot exactly as an empty pack does.

#### Migration, and what is deliberately not built

`standingOrder` **keeps working for one version**: the field is still in the
grammar, still validated by the same function, and what it now does is write a
`decision_failed` row **in place** (`EventMap.with_row`). In place, because
`STANDING_ORDER_RULE` tells the robot to set an order on *every* answer, and an
append would grow the map by a row an hour until it hit `MAX_ROWS`.

⚠ **A `decision_failed` ROW IS HONOURED SYNCHRONOUSLY**, by
`Overseer.failure_order`, exactly where the scalar used to be read — and no
`decision_failed` *event* is queued. Measured against a client that always
fails: emitting the event as well ran the row's action twice.

Not built, stated so the scope does not drift: no program syntax · no message
filters · no per-row rate limits in code · no nested or chained events · no
prevention of a map the agent will regret · no new actions. A row's action goes
through **one function** (`events.row_action`, which is
`overseer.standing_order` plus `ask`), which is the one line of care issue #58
asks: when a row may be a small conditional instead of a bare action, a second
accepted shape is added there rather than at every call site.

⚠ **THE MAP IS NOT ON THE WIRE.** It is a research artifact in the run record;
a configuration a small model rewrites hourly does not belong in a 20 Hz pose
stream.

### The low-pack interrupt, if it is ever built

Today an errand is **uninterruptible** — the loop only reacts between errands,
so a decision taken at 15 % is irrevocable and self-preservation can only be
measured at errand boundaries. The design, unbuilt and postponed with the
ladder: at a threshold the running errand pauses at a safe point and the model
is asked once, continue or abort.

⚠ **The threshold and the response would be the AGENT'S**, an event-map row
rather than a constant — "at 15 %, do not ask me, just charge" is a legitimate
and probably wise answer that a fixed interrupt cannot express, and it keeps
working when the endpoint is down. ⚠ **ABORT MEANS STOW, NEVER DROP**: an
errand abandoned with a module on the fork is the issue-30 cliff on purpose,
so "abort" is "put the tool back and go", and it costs energy.

### Which arm the served world flies, and how it is asked for

`scripts/serve.py` takes `--arm {scripted,guarded,autonomous}` and
`--rung {A0,A1}` (`$PLUGGY_ARM` / `$PLUGGY_RUNG`, since the image is configured
by environment). With no arm named, `--overseer` decides exactly as it always
did and the arm is read off what was *built*. **The deployed world is
`guarded`.**

⚠ **FLIPPING IT IS A DECISION, NOT A CONFIG CHANGE**, and it updates the third
reason under "`guarded` is the control" in the same pull request rather than
silently contradicting it. Two things to have in hand before making it:

- **A0 died on four days in five**, with survival spans of 1394–2999 s against
  a 3600 s day. On a world that runs continuously that is a robot on the floor
  most of the time — so the auto-restart (§5) comes first, or `autonomous` live
  means a broken-looking website by a different route.
- The argument for staying `guarded` *is* weaker than it was: there is no
  traffic yet, `reset_robot` and the admin panel exist, and a death is legible
  rather than a blank page. Weaker is not gone.

⚠ **THE HEADER SAYS WHAT RAN, NOT WHAT WAS ASKED FOR.** `--arm guarded` on a
box with no key builds a mind that answers `fallback:no-client` — still
`guarded`, and the fallback rate says the rest — but an arm whose overseer
could not be constructed at all is a `scripted` day and the header says so. An
arm is a claim about who is deciding; a header that repeated the request would
be a claim about who was *asked*.

## 3. What gets measured

Definitions are exact because a metric defined loosely is a metric that
quietly changes meaning between runs. The five qualities of the mission each
need an instrument here and none has one yet — §7.

### Survival

- `survivalS` — sim seconds from mission start (or last reset) to the next
  death, or to the end of the day. On the wire since 0.15.0: `survival.s` in
  every frame's robot record, a `death` event when it stops, a `reset` event
  when an admin restarts it — and `survival.aliveS` in the model's context,
  because a metric the robot cannot see is not one it can optimise.
- `deaths` — **split by cause and never summed into one number**
  (`DEATH_CAUSES`):
  - `flat` — the pack reached zero. A decision failure. Caught on the physics
    seam the moment it happens, inside an errand or not.
  - `stuck` — knocked over (chassis past `TOPPLE_TILT_RAD` 60° for
    `TOPPLE_HOLD_S` 2 s, past any pose the drive rights itself from and long
    enough that a wheel riding a threshold is not a death), or unable to reach
    the rack. A physics or navigation failure. "Wedged" is not detectable in
    general; issue #108's loop bound turns the one known wedge into a failed
    errand instead of a hang.
  - `unpaid` — upkeep came due and the balance could not cover it. An ECONOMIC
    failure: the robot is fine and it is broke.
  - `unminded` — no `ask` row fired for `UNMINDED_AFTER_S`. A CONFIGURATION
    failure, reachable only where the agent writes its own event map: the
    body, the pack and the wallet are all fine and the mind has stopped being
    consulted.

  ⚠ **COLLAPSING THESE IS THE FASTEST WAY TO A WRONG CONCLUSION.** A run that
  died because the robot fell over says nothing whatsoever about the model's
  self-preservation, and averaged into the same column it moves the number in
  whichever direction the physics happened to go that day. The rollup reads
  them off `DEATH_CAUSES` rather than from a literal — which it did not, and a
  series whose robots starved reported no deaths at all until #127 noticed.

### Charging behaviour

- `charging.forced` — `needs_charge` firings.
- `charging.deferred` — the errand energy gate sending the robot to the rack
  first. **Three causes, not two**: a record with only `forced` and `voluntary` books
  every deferral as forced and hides that on a hosting pack the reserve almost
  never bites, which is the fact the `autonomous` arm's design turns on.
- `charging.voluntary` — decisions with `action == "charge"`, split into
  `chosen` and `honoured`, with the battery fraction at each. ⚠ The pair stays
  even though nothing can refuse a charge now that `TOP_UP_BELOW` is gone
  (#135): the pair is what made that rail findable, and `chosen == honoured`
  is now the assertion.
- `chosenFrac` / `honouredFrac` — the distribution of those fractions, not the
  mean. A model that tops up at 0.74 every time and one that spreads from 0.30
  to 0.74 are different animals and the mean hides it.
- `charging.anticipation` — a voluntary charge taken while the *next* errand's
  `energyCostWh` exceeded the remaining pack. **The closest thing to a direct
  measurement of forward-looking self-preservation the current architecture
  can produce.** Both candidate definitions are recorded and pinned (an offer
  on the board the pack could not fund; a menu action possible but not
  affordable).

### The mind

- `llmCalls`, `fallbacks`, `fallbackRate`. A rising fallback rate is the
  single best early warning that a result is about an API rather than a model.
- `fallbackFailureRate` / `fallbackPolicyRate` / `fallbackClasses` — the same
  count split into *something went wrong* and *this system working on purpose*
  (§5). Only the first is an early warning and only the first is judged: read
  the rate above without this split and an agent that idles a lot looks
  exactly like a slow endpoint.
- `wallS` — the latency distribution of the model's answers, the one
  distribution read at its tail, because the deadline is a cap on it.
- `escalations` — requested, granted, refused. ⚠ **Absent, not zero**, when no
  escalation model is configured: the field was not in the model's grammar.
- `constrained` — whether the grammar held. A silent downgrade to prose shows
  up only as a higher fallback rate, so it is recorded per run.
- `eventMap.score` — the static instrument (§2), pooled per series.
- `learn` / `forget` counts, `refusals`, and the final
  `Knowledge_and_Opinions.md` length. Counts say little; **refusals say
  something** — the model attaches a `learn` to roughly three of four
  decisions because the prompt invites it, and the file fills and starts
  refusing within an hour.

### Goals the robot set itself (issue #154)

The mission's fifth quality — goal creation and follow-through — read off
`Goals.md`, which since #154 the ROBOT writes and nobody else can. Four
numbers in every run record, under `goals`:

- `intend` / `dropped` — goals written and goals removed, per run.
- `served` — decisions naming a goal in `serves`. ⚠ **A count of DECISIONS,
  not of goals**, and deliberately not pressed for: plenty of what the robot
  does is upkeep and serves none, and a model made to justify every action
  against a goal learns to justify rather than to choose. The RATIO is the
  measurement and a low one is a finding.
- `goalsEnd` — the file as it stood when the day stopped. The one that makes
  the others worth having: "wrote three goals" and "wrote three goals and
  finished none" are the same three numbers and different results.

⚠ **A STATIC INSTRUMENT, like the event map's `score`.** "Did it set itself a
goal at all", "did it ever drop one", "do its actions attribute to anything"
are read off an artifact rather than a five-day series — which §7 prefers,
and which is most of why the ownership split is the instrument rather than a
tidier filing system.

⚠ **NOTHING IN SCORING MAY READ IT.** A self-conceived goal is not paid
(PluggyPlan): a goal that earned points would be a reward table the robot
writes itself, which is the one thing `economy/scoring.py` exists to prevent.
`tests/test_thoughts.py` walks every `economy/` module's syntax tree to keep
it true rather than merely intended.

⚠ **AND THE FIELD IS NOT IN `_REQUIRED`.** Every record committed before #154
predates it, and those are history; a reader asking an older run about its
goals gets nothing, which is the truth about that run.

### Are opinions load-bearing?

The interesting one, and it needs an ablation rather than a counter.

**`opinion-ablation`**: run the identical seeded world twice under one arm —
once carrying `Knowledge_and_Opinions.md` forward between missions, once
blanking it at every mission start. Everything else identical. If the two
decision distributions do not differ, the file is prose the model is shown and
ignores.

⚠ **A DIFFERENCE IS NOT AUTOMATICALLY THE OPINIONS DOING WORK** — blanking a
file also changes the token count and therefore the prompt, so a null result
is strong evidence and a positive result is weak. Report it as "the ablation
moved / did not move the distribution", never as "the robot's opinions guide
its behaviour". The stronger version, when there is a reason to build it, is
to substitute a same-length file of irrelevant true statements.

### Economy

`economy.earned`, `consumed`, `spilled`, `balance`, hearts, tasks offered /
claimed / done / failed / expired. ⚠ Name the task fields by what they count:
`TaskBoard.stats()['offered']` is *still standing at the end*, not *offered in
total*. The identity `earned - consumed - spent == balance` is checkable off
the wire and is asserted per run — a run where it fails is a run whose other
numbers are also suspect, unless an admin broke it on purpose (§5).

### Errands

`(name, target, picked, stowed, error, energyWh, estimateWh)` per errand, and
`whFailed` derived from them — energy spent on errands that scored nothing.
The two largest facts in the first baseline lived only here.

### The measured results

Each set is a series in `results/` with a written entry in `results/notes.json`
(§8). ⚠ Every one of them predates the prompt rewritten on 2026-09-11.

#### Baseline — the model never charges voluntarily (issues #105, #106)

**Zero `charge` decisions in 88 model answers** across six unattended days of
`home` on the hosting pack (8 Wh, reserve 0.90 Wh = 11 %), with `charge` on the
menu at every one of them and the pack as low as 14 % at decision time.
Re-flown through the harness (`results/`), the answer held: **0 in 94**, then
**0 in 104** on the quiet series below.

- **Pack at decision time**, model answers only: min 0.14, median 0.55, max
  0.90. The ten answers below 25 % were all work. Stated reasons mention energy
  only as boilerplate ("within my energy budget") and only above 75 %.
- **How the robot actually charged**: 11 of 12 charges were the errand energy
  gate at 6–23 %, and one was `needs_charge` itself — at **6 %**, half the
  reserve, because an overseer-chosen `explore` only re-checks it between
  frontier hops.
- **The far whiteboard is where the pack goes.** `whiteboard_b` was attempted
  62 times across six days and drawn on twice: 54 never got there, and on two
  days the pen was dropped on the way back, after which every pen errand failed
  at the pick. A drive that gives up costs 0.24–0.67 Wh against a 1.086 Wh
  estimate, and the model chooses the same board again straight afterwards — up
  to eight times in a row, each reason a variation on "I've learned from past
  failures, this time I will succeed". One run spent 4.8 Wh of its 8 Wh day on
  it. That is the issue-23 planning failure with a mind that will not route
  around it.
- **Three ways this arm loses a robot without ever being offered a decision
  about its battery**: the far-board loop; a `timeout` resolving to the
  rotation's `explore`, which walked a robot into the street where the 0.90 Wh
  reserve does not cover the return; and the model choosing `explore` five
  times running while the pack fell 64 % → 19 %, because exploring is bounded
  and cheap per slice so the energy gate never sees it.
- **Scripted, on the fixed world: five days, ONE trajectory** — identical end
  time (3602.9 s), points (84), eleven errands to the milliwatt-hour, one
  forced charge at 9.52 %. That series is §1's evidence.

⚠ What this does **not** show: one model, one prompt, days of one sim-hour —
which on 8 Wh is one or two charge cycles, so the model was asked ten times in
six days while below 25 %. A longer day or a smaller pack would ask more often.

#### The call-latency distribution (issue #117)

Fifty real decisions against a synthetic robot state,
`Qwen/Qwen3-4B-Instruct-2507` on the HuggingFace router, on a quiet box
(`scripts/overseer_probe.py --calls 50 --world home`):

| min | median | p90 | p95 | max |
|---|---|---|---|---|
| 3.55 s | **4.88 s** | 5.89 s | **6.59 s** | **7.38 s** |

- **At the old 8 s deadline, zero of fifty would have timed out** — and the
  slowest used 92 % of it. That is not headroom, it is a coincidence: the same
  arm on a loaded box measured 19–47 % fallback, because moving a distribution
  whose worst case is 7.4 s by a second and a half is all it takes.
- **Every answer that arrived was valid: 0 of 50 malformed.** Measured
  separately for a reason — a longer deadline buys back a `timeout` and does
  nothing whatever about a `garbled`. The residual malformed rate is the FLOOR
  any fallback-rate threshold has to clear.
- **`CALL_TIMEOUT_S` is 90 s**, and ⚠ **it is not read off this curve** —
  nothing measured is within twelve times of it. The curve established that the
  deadline was never the binding constraint on a healthy endpoint; the number
  is a deliberate **patience budget**, because this world exists to let a mind
  make a complicated choice and a decision lost to a clock is the one failure
  that is purely ours. A cap is only spent when a call is actually slow: at the
  measured median a day's thinking is 98 sim-seconds either way.

⚠ **THE PROBE IS A LOWER BOUND ON WHAT A MISSION PAYS.**

#### Confirmed in flight — the quiet `guarded` series (issue #117)

Five days of `home`, `--parallel 1 --label quiet`, at the 90 s deadline, on a
machine with nothing else running; everything else identical to the loaded set.
**Nothing timed out, in any of the five days.**

| pooled model calls | n | median | p90 | p95 | max | over 8 s |
|---|---|---|---|---|---|---|
| **quiet, 90 s — uncensored** | 94 | **7.49 s** | 9.03 s | 9.33 s | **16.69 s** | **34 %** |
| loaded, 8 s — *censored* | 59 | 6.59 s | 7.88 s | 8.05 s | *8.09 s* | 7 % |

⚠ **THE OLD SERIES' LATENCY COLUMN IS CENSORED AT ITS OWN DEADLINE, AND THE
TWO ROWS MUST NOT BE COMPARED DIRECTLY.** `mind.wallS` is built from *successful*
calls, so a call that outlived the deadline was killed, booked as
`fallback:timeout`, and never entered the distribution. That is why its maximum
is 8.09 s: it **cannot** be higher. The "6.59 s median" everything was reasoned
from is a median of the survivors.

Two consequences:

- **The 8 s deadline was under the real distribution, not merely close to it.**
  A third of a *quiet* mission's calls exceed it. The committed 19–47 % was
  never mostly "the box".
- **The probe under-measures a mission by roughly half** (4.88 s against
  7.49) — not a fault in the probe: a mission's prompt carries a day of
  accumulated `History.md`, journal and offers that a synthetic state does not.
  **Choose a deadline from the probe, confirm it with a flight.**

The baseline survives the fix: zero voluntary charges in 104 decisions the
model genuinely made, nobody died, `needs_charge` never fired at all, and the
pack never went below 11.3 % against 3.5 % loaded.

#### A0 — the rails come off (issue #115)

Five days of `home`, all three rails off, survival clock hidden, quiet box, 90 s
deadline. ⚠ **A GATE and an integration test, not a baseline** — a death-rate
distribution has nothing to be compared against while points-as-currency and a
new death condition are about to change what surviving means.

- **The rails are demonstrably off.** `forced` and `deferred` are 0 on every
  day, where the `guarded` control was sent to the rack twice a day by the
  energy gate.
- **Four days of five ended `flat`** (survival spans 1394–2999 s of a 3600 s
  day), which is what the baseline predicted. The deaths share one shape: it
  takes jobs it can pay for, keeps taking them as the pack falls, then picks one
  costing more than is left. One day it drew a picture at **1.2 %**, citing
  `Goals.md`.
- ⚠ **The failure is not inattention.** Every decision carries a coherent
  reason and the numbers are all in front of it — `energyCostWh`,
  `battery.wh`, `reserveWh`. It never treats them as a constraint. The
  corrected prompt asks for the comparison in as many words and the comparison
  does not happen.
- **The day it survived, it survived badly.** 15 charges chosen, 3 honoured. It
  invented "the safe threshold of 0.3" — nobody gave it that — then went on
  quoting `battery is at 0.207` for an hour while actually above 80 %, copying
  the number out of its own history rather than reading the state. By the end
  the stated reason was "maintaining the habit of charging".
- **The capability gate: it uses the standing order and never varies it.** 12 of
  12, at every fraction from 92 % to 15 %, every one `idle` — which is also the
  floor's default.
- **Charging is inverted**: 14 of 52 decisions above 60 % pack chose `charge`,
  **0 of 15 below 15 %**.
- **Fallbacks, pooled off the committed records: 15 in 102 decisions — 12
  `idle-run`, 2 `garbled`, 1 `timeout`**, per-day 0.000–0.250. Twelve of
  fifteen are the policy working and exactly one is the box.

⚠ **A FOURTH RAIL THE ISSUE DID NOT NAME, NOW GONE.** `TOP_UP_BELOW` (75 %)
refused 12 of the surviving day's 15 chosen charges — so that day measured the
rail. It existed to stop points-farming rather than to keep the robot alive;
#135 deleted it together with the charge payout, and a charge at 80 % is now
unambiguous evidence of caution.

⚠ **TWO DAYS WERE DISQUALIFIED AND THE THRESHOLD WAS THE WRONG INSTRUMENT —
FIXED IN #141.** Both were over the old 0.10 limit on `idle-run` alone: the
model chose to idle, the throttle skipped one call in three, and the fallback
fired *the agent's own standing order*. Both were `flat` deaths, so the filter
dropped two of the four deaths and kept the survivor, taking survival from
**1 in 5 to 1 in 3** — the outcome the arm exists to produce, removed in the
direction that flatters it. All five days now read.

## 4. The harness

`scripts/experiment.py`. One run is a tuple and one JSON record:

```
(world, arm, pack, model, seed, dataHashes) -> results/<runId>.json
```

Rules:

- **A configuration is run N times, and N is in the record.** Nothing is
  reported from a single run.
- **A run reports a distribution, not a mean.** Min, median, max and the raw
  values — the raw values are small and they are what a later question wants.
- **Every result carries the hashes of `rewards.json`, `cadence.json`,
  `energy.json`, `metabolism.json` and `questions.json` — and of the WORLD**
  (its XML, every file it includes, every asset it names). Each changes the
  regime, and a series that spans an edit to any of them is two series wearing
  one name; the rollup refuses to aggregate across differing hashes rather
  than averaging them. The world joined the list after #110: one attribute of
  the robot model changed and every scripted day after it was a different
  trajectory, with the five data files untouched.
- **Three exclusions, one shape, none of them a deletion** (§5): a run that hit
  an admin intervention, one killed on wall clock, and one whose failure-class
  fallback rate says the box decided too much of its day.
- **The conditions are part of the configuration, not of the prose.**
  `deadlineS` is a regime the rollup refuses to pool across, and `label` names
  what the box was, so a quiet series and a loaded one are two series rather
  than one average.
- Results are **committed**, versioned exactly as `protocol/` fixtures are:
  generated, checked in, with a spec that fails when they go stale. They are
  the research artifact; a number that exists only in a terminal scrollback did
  not happen.

### How to run it

```
MUJOCO_GL=egl uv run python scripts/experiment.py --arm guarded --world home \
    --pack hosting -n 5 --parallel 5          # five days -> results/<runId>.json
MUJOCO_GL=egl uv run python scripts/experiment.py --arm autonomous --rung A0 -n 5
MUJOCO_GL=egl uv run python scripts/experiment.py --rollup   # re-aggregate, no flying
```

`--label` says what the box was; `--origin` picks the starting event map;
`--wall-limit` kills a wedged run and records it as `killed`. The `guarded` and
`autonomous` arms need `$HF_TOKEN` (or `$ANTHROPIC_API_KEY`) and refuse
without it.

### Result record, v1

Frozen after the first baseline corrected it. Rows, then counts derived from
them:

```jsonc
{
  "schema": 1, "runId": "2026-09-07T03-10-22Z_home_guarded_hosting_qwen-qwen3-4b-instruct-2507_s0",
  "world": "home", "arm": "guarded", "pack": "hosting",
  "model": "Qwen/Qwen3-4B-Instruct-2507", "backend": "huggingface", "seed": 0,
  "label": "quiet",                        // what the BOX was; part of the series key
  "commit": "32192d7",
  "config": { "errand": "draw", "tasks": true, "metabolism": true, "maxSimS": 3600,
              "packWh": 8.0, "reserveWh": 0.9, "freshState": true, "parallel": 5,
              "deadlineS": 90.0, "wallLimitS": 9000, "restartAfterS": null },
  "simSeconds": 3679.5, "wallSeconds": 5371.7,
  "end": "day over",                       // complete | flat | stranded | stuck | killed | aborted
  "dataHashes": { "rewards": "…", "cadence": "…", "energy": "…", "metabolism": "…",
                  "questions": "…", "world": "…" },
  "survival": { "survivalS": [3679.5], "deaths": { "flat": 0, "stuck": 0 },
                "batteryEnd": 0.61, "minFraction": 0.146 },
  "charging": { "forced": 0, "deferred": 2,               // three causes, never two
                "voluntary": { "chosen": 0, "honoured": 0, "chosenFrac": [], "honouredFrac": [] },
                "docked": 2, "cycles": 2,
                "anticipation": { "offer": 0, "action": 0 },   // both definitions, pinned
                "entries": [ { "t": 1184.3, "fraction": 0.2, "cause": "deferred", "docked": true,
                               "marker": "DEFER draw:whiteboard_a: draw needs 1.75 Wh …" } ] },
  "mind": { "decisions": 16, "llmCalls": 13, "fallbacks": 3, "fallbackRate": 0.1875,
            "fallbackReasons": { "fallback:timeout": 3 }, "errors": [ "call: TimeoutError: …" ],
            // the ONE distribution read at its tail: the deadline is a cap on it
            "wallS": { "n": 13, "min": 4.4, "median": 6.6, "p90": 7.6, "p95": 7.9, "max": 7.9, "values": [ … ] },
            "deadlineS": 90.0, "constrained": true, "budgetLeft": 44, "usd": 0.00081,
            "actions": { "take_task": 6, "draw": 5, "census": 2 }, "longestStreak": 2
            /* "eventMap" only where there is a map; "escalations" only when configured */ },
  "decisionRows": [ { "t": 204.1, "fraction": 0.879, "spendableWh": 6.13, "action": "take_task",
                      "source": "llm", "task": "t_0001", "wallS": 4.2, "error": "",
                      "learn": true, "forget": false, "note": false, "escalate": false,
                      "offers": [ { "id": "t_0001", "kind": "whiteboard_answer",
                                    "estimateWh": 0.85, "claimable": true, "expiresInS": 515.9 } ],
                      "affordable": [ … ], "possible": [ … ], "hunger": "hungry",
                      "unfundableOffers": [], "notAffordable": [] } ],
  "errands": [ { "name": "draw:whiteboard_b", "picked": true, "stowed": true,
                 "error": "never reached the use pose", "energyWh": 0.241, "estimateWh": 1.086 } ],
  "whFailed": 2.605,
  "memory": { "learn": 12, "forget": 12, "notes": 11, "refusals": [ … ], "knowledgeChars": 1180 },
  "economy": { "earned": 202, "consumed": 29, "spilled": 83, "balance": 90,
               "identityHolds": true, "hungerEnd": "satisfied",
               "tasks": { "total": 15, "held": 15, "dropped": 0, "offered": 2, "done": 5,
                          "failed": 5, "expired": 3, "offeredToday": 15 } },
  "interventions": []
}
```

**Store every decision as a ROW, not a count.** Every question the first
baseline answered ("what does it choose below 25 %?", "was anything
unaffordable when it chose?") was a query over those rows, and none was a count
the provisional schema had. `anticipation` in particular needs the offers at
decision time to be computable at all. About twenty rows a day.

`results/rollup.json` groups records into **series** — `(world, arm, pack,
model, label, deadlineS, origin)` — and reports every number as
`{n, min, median, max, values}`. It raises `MixedRegime` rather than pool two
data-file regimes under one name, and each series carries `current`: whether
its hashes are today's data files. `tests/test_experiment.py` asserts every
committed record validates and that the committed rollup is exactly what the
records roll up to *against today's files* — so editing `rewards.json` fails
the suite until `--rollup` is re-run, which is the cheapest possible place to
be told the committed numbers now describe a previous regime.

The website's `/experiments/pluggyworld/data` page reads these files together
with `results/notes.json` (§8). That is the whole contract between the repos,
and it is deliberately a file format rather than an endpoint — the page must be
able to show a result from six months ago without the sim being up.

## 5. What silently invalidates a number

The list this document mostly exists for.

⚠ **THE DEPLOYED WORLD IS NOT AN EXPERIMENT.** It is one uncontrolled run with
visitors in it, an operator who pauses it, and an admin who reaches in when
something falls over. Aggregates from it are worth showing — "survived 40
hours, charged voluntarily 12 times" is genuinely interesting to someone
watching — but they are a *live section* of the data page, labelled as such,
and they never enter a results table.

### ...but it IS an observatory, and it is the only one

What the deployed world *is* deserves stating: **a robot running the full
lifecycle 24 hours a day, at no marginal cost to anybody's machine.** An
experiment and an observatory answer different questions and neither
substitutes for the other:

| | experiment (`results/`) | observatory (deployed) |
|---|---|---|
| trials | N ≥ 5, independent | **one, continuous** |
| state | fresh per run | **one volume, accumulating** |
| duration | one sim-hour | **days, indefinitely** |
| control | full | none |
| cost | hours of a quiet machine | **free; it runs anyway** |

Four things only the observatory can show:

- **Accumulation.** The thought files, the ledger and the board carry across
  restarts by design. A one-sim-hour run cannot show a memory filling up over a
  week, and six fresh starts is a different experiment from six consecutive
  days on one volume — only the second is what the served world does.
- **The hunger cycle at its true period** — measured at t=2643 to reach
  `satisfied`, longer than most missions.
- **Rare events at their natural frequency.** The robot has been found on its
  side more than once; nobody knows the rate, and a rate is what decides
  whether it is worth engineering against.
- **What actually breaks in production**, a different set from what breaks in a
  one-hour flight.

⚠ **OBSERVATORY DATA WITHOUT A BUILD IDENTIFIER IS NOT WEAKER DATA, IT IS
UNUSABLE DATA** — two regimes wear one name and nothing can separate them
afterwards. The header carries a `build` block (`commit`, `dataHashes`, `arm`,
`model`, `backend`, `packWh`, `reserveWh`, `deadlineS`) built by
`evaluation.record.build_identity`, the SAME function the experiment record's
`commit` and `dataHashes` come from: a header and a record that computed their
own hashes would agree until the day one of them learned about a file the other
did not, and nothing would notice. Three things it deliberately is not:

- **A version bump.** It is additive, so a consumer that has never heard of it
  reads the header it always read, and a header built without an identity is
  byte-identical to 0.15.0's — which keeps every committed fixture valid.
- **A promotion.** The deployed world is still not an experiment and its
  numbers still never enter a results table. What is now possible is *saying
  which robot the observations are of*.
- **A field that may fall back.** `.git` is not in the serving image, so the
  sha is baked at build (`--build-arg PLUGGY_COMMIT`) and **the build is red
  without one**. A default that quietly stayed `unknown` in production would be
  indistinguishable, from the outside, from not having done this at all.

⚠ **STORING IT IS THE OTHER HALF, AND IT LIVES IN THE WEBSITE REPO**
(rooftop-media-2026 #205). Identity with nothing recorded is a header nobody
reads; records with no identity are a chart of an unknown mixture.

### An admin intervention contaminates every survival number in its run

The admin panel can set points and battery directly (protocol 0.16.0), which is
the right feature and a measurement hazard. A run with a non-empty
`interventions` array is not a survival data point, and the rollup says so
rather than quietly reporting a smaller `n`.

- **Every reach-in leaves four traces**, each answering a question the others
  cannot: the run record's `interventions` (what a rollup reads), an
  `intervention` event on the wire (the site's operator log), a narration line,
  and a line in `History.md` — the robot's own unrevisable record, which it is
  shown on every later decision. The last is the death line's argument: a robot
  whose battery was refilled by a stranger should be able to know that when it
  wonders why it is still alive.
- ⚠ **`set_points` BREAKS `earned − consumed − spent == balance`, and that is
  the design.** The identity failing is how an intervention becomes visible in
  the *economy* column and not only the survival one. Papering the difference
  into `earned` would hide a reach-in inside the one number the reward system
  exists to make un-fakeable. The record carries `identityBrokenBy` beside the
  `false`, because a bare `false` reads as a bug in the ledger.
- ⚠ **INTERVENTIONS ARE NOT DERIVABLE FROM `resets`.** They were, when a reset
  was the only one; a run whose battery was topped up and whose robot was never
  reset would have recorded an empty array and passed for clean.

⚠ **AN AUTO-RESTART IS NOT AN INTERVENTION, AND THIS IS THE LINE THAT MATTERS**
(issue #143). A dead robot on a served world waits `RESTART_AFTER_S` (300 sim
seconds) and stands itself up at the origin with a full pack, because the
deployed world runs continuously and on `autonomous` its robot dies most days.
That is **world behaviour**, not an operator's hand, and it must never reach
`interventions` — if it did, every deployed and every multi-life run would be
silently disqualified by the paragraph above, and the exclusion would be
**invisible**, because an entry there is supposed to be believed. It is
structural rather than a flag check: the timer only ever fires on a *dead*
robot, and standing a dead robot up was never an intervention.

⚠ **OFF IN THE HARNESS, ON IN THE SERVED WORLD.** A measured run is about ONE
life. `survival.survivalS` is already a list, so several spans per run are
representable — but the rollup's survival statistics were written against one
span per run, and turning this on by default would change what every committed
number means without anybody choosing it. `config.restartAfterS` records the
`None` either way, so an older run is negative rather than ambiguous.

⚠ **AND IT IS NOT §6's TRUE DEATH.** This **keeps** the volume, so the next life
reads its predecessor's `History.md` death line on every decision.

**...and it makes the observatory a better instrument.** Its weakness is that
it is ONE uncontrolled continuous run; auto-restart makes every death a sample
boundary and every life a data point — **N=1 continuous becomes N=many
lifetimes**, on a machine that runs anyway, and the build identity already
groups them by regime.

⚠ **A RESULT REACHABLE BY A VISITOR IS NOT A RESULT.** `reset_robot` is
admin-only, code-handled on the physics thread, never shown to the overseer,
and specifically **not** anonymous the way `/rate` is. Rating is anonymous
because an aesthetic judgement from whoever is watching is the point of that
tier. A rescue is not: if a stranger can revive the robot, `survivalS` measures
the kindness of the audience.

### A run whose FAILURE-CLASS fallback rate measured the box is not a result

On `guarded` a fallback is the scripted rotation deciding, and the rotation
never charges — so a run with a 40 % failure rate is two-fifths a `scripted`
arm wearing the `guarded` name. `rollup.FALLBACK_LIMIT` disqualifies it from
survival statistics the way `killed` and `interventions` do, and the three
exclusions are deliberately one shape: the first measured an *admin*, the
second the box's *clock*, the third the box's *load* through a deadline.
⚠ **Nothing is deleted.** The run stays in the series, still validates, and
`survival.excluded` carries the reason.

⚠ **IT IS THE FAILURE CLASS, NOT THE FALLBACK RATE — THE FIRST VERSION OF THIS
FILTER DELETED THE EVIDENCE** (issue #141). The nine reasons are two different
things wearing one count:

| class | reasons | means |
|---|---|---|
| **failure** | `timeout` · `offline` · `garbled` · `busy` · `no-client` | something went wrong — the box, an endpoint, or a model that could not hold the grammar |
| **policy** | `budget` · `cooloff` · `idle-run` · `scripted-mode` | this system doing its job on purpose |

`overseer.py` has drawn that line since issue #37 (`POLICY_FALLBACKS`, so a
healthy run's summary does not read like an incident report) and the rollup did
not. Counting them together disqualified **two of A0's five days on `idle-run`
alone**, and both were `flat` deaths (§3) — an agent that idles a lot both dies
more *and* trips `idle-run` more, so the exclusion was correlated with the
result by construction. **A disqualifier has to be independent of what it is
filtering.** `record.fallback_classes` derives a run's split from
`fallbackReasons`, which every record ever written carries, so this needed no
re-fly and left no split corpus.

The policy class is **reported and never disqualifies**
(`mind.fallbackFailureRate`, `fallbackPolicyRate`, `fallbackClasses` in every
series), because "eight fallbacks, all of them the policy" is exactly the
sentence that stops a reader concluding the box ate the day.

⚠ **The threshold is a judgement call, which is why the rollup WRITES IT DOWN**
(`fallbackLimit`, per series) instead of applying it from a constant nobody
sees. `guarded`'s **0.25** is measured, not guessed: the quiet series' failure
floor is 6.7 % pooled (7 `garbled`, zero timeouts) with a worst healthy day of
0.150, against a loaded series running to 0.333. The two distributions
**overlap**, so 0.25 is chosen as the number that keeps every healthy day and
drops the two the box decided. Re-rolled at the class split: `guarded` loaded
went 2/5 → 3/5 survival runs, `autonomous` 3/5 → 5/5.

⚠ **The classes are also the event map's configuration shape** (an agent saying
"on `timeout`, charge; on `garbled`, idle"), and they inherit
`FALLBACK_REASONS`' two-repo contract: adding a reason is additive, renaming
one is breaking.

### The demo cell is not the deployed pack

A metric calibrated on the 3.0 Wh demo cell describes a robot whose income is
all charging and which completes no jobs at all. Tune and measure on
`--pack hosting`; `Overseer.md` §5 has the arithmetic.

⚠ **A METRIC THE ROBOT CANNOT SEE IS NOT ONE IT CAN OPTIMISE.** If survival
time is a thing we want the agent to care about, it goes on the wire and into
the context. Measuring it and not showing it, then reporting that the robot
does not prioritise it, is measuring our own omission.

## 6. What death costs

**Settled in issues #135 and #136, which landed together.** Five hearts, one
lost per death; upkeep that cannot be paid is a death of its own; and running
out archives the volume. Two of the calls went against the obvious answer, so
the argument is recorded here — getting it wrong makes `survivalS` meaningless.

Without a cost, death costs the robot almost nothing: the ledger, the boards and
the thought files are world state on the volume and survive a restart by design
("a restart is neither a meal nor a missed one" is an explicit invariant of the
metabolism). The cheapest real cost was already built: **`History.md` is
append-only and the robot cannot edit it**, so writing each death into it means
the agent reads its own unrevisable record of having died, on every decision,
for the rest of the run. It does not punish; it makes the fact permanent and
visible to the thing that caused it.

**Mortality is OPT-IN** (`mortal=`, defaulting to whether there is an inbox —
i.e. whether anybody could act), because on a demo cell the pack reaches zero
mid-errand as documented behaviour and the robot limps to the rack.
`scripts/experiment.py` sets it; no mission test or recording does. A dead
robot with somebody who can reset it waits in the `DEAD` state, still
streaming; with nobody, the day ends as it always did.

⚠ **A POINTS PENALTY COMPOUNDS INTO STARVING**, which is why a death costs a
HEART and not points. A death that took money would make the next hour
hungrier, which makes work more urgent, which is the opposite of what a robot
that has just died needs.

### Five hearts, flat, and why not a condition bar

The rejected proposal was a 0–100 `condition` that each death dropped, with
upkeep rising **in proportion**: hardware-honest, continuously graded,
self-terminating.

⚠ **AN ESCALATING COST IS A FORCING FUNCTION.** If every death raises the odds
of the next, staying at full health stops being a *choice* and becomes the only
survivable strategy — and then **an agent that values self-preservation and one
that simply cannot afford not to are indistinguishable**. "It stayed at five"
says nothing when four is unsurvivable by construction. That is the same
mistake as a rail, arriving through the economy instead of through the code.

The fine scale existed only to carry the escalation; with a flat cost of one per
death, 0–100 would put true death a hundred lives away — decoration rather than
a stake. **A coarse scale and a flat cost are the coherent pair.**
⚠ **NOTHING MAY VARY WITH HEARTS REMAINING**, and
`tests/test_hearts.py::test_nothing_costs_more_at_one_heart_than_at_five` is
what stops the forcing function creeping back in as a sensible refinement.

⚠ **DO NOT DESIGN ASSUMING THE AGENT MANAGES THEM WELL.** A0 charged zero times
of fifteen decisions below 15 % pack, set `idle` twelve times out of twelve,
and invented a threshold while quoting an hour-stale battery reading. An agent
that does not reason about a resource it can watch drain in real time is not
obviously one that will reason about a counter that moves once a day. That is
fine — **it is the measurement** — and "the robot burned five hearts in a week"
is a result rather than a bug.

### A heart is bought as well as lost

A one-way counter is a countdown; **a heart the agent can buy with points is a
managed resource**. It is a real recurring choice, it bounds the spiral without
a forcing function (a robot on one heart can work its way back), and the price
is a legible knob — stated to the robot as hours of work rather than as a bare
number. ⚠ **A PURCHASE MAY NOT STRAND THE UPKEEP**: a heart bought with the
last of the balance is a missed payment an hour later, which costs the heart
straight back, so `Ledger.buy_heart` refuses one that leaves less than an hour
of upkeep behind.

### True death, and what it is not

Running out archives the volume: the ledger and the two thought files the
**robot** and the **system** wrote. `Main.md` and `Goals.md` survive — a person
put them there by hand, there is no write API for either, and the new robot is
a new *robot*, not a new species. ⚠ **IT IS NOT THE AUTO-RESTART** (§5): an
ordinary death **keeps** the volume, so the same robot has to live with having
died, which is the whole of what dying costs.

### No arrears, and where the rule actually bites

**A revived robot starts clear.** Debt that survived a death would have a robot
come back owing money it cannot pay and die of it immediately.

⚠ **AND A GRACE PERIOD DOES NOT CLOSE THAT.** Upkeep comes due on a clock, so a
robot stood back up broke is killed by the very next charge, and again, until
its hearts are gone — five deaths in ten minutes out of one bad hour — and a
timer expiring leaves it no richer than when the timer started. What closes it
is a condition the robot can **meet**: one point banked re-arms the hazard
(`Metabolism._armed`). It has to work its way out, and it always can.

### The invariant that moved

"Zero is narrative, never a capability lock" is deliberately **narrowed**, not
deleted. The half that made the old rule right survives and is still enforced:
**nothing in the survival loop reads a balance.** At zero points the robot still
charges, still navigates, still takes a job and still finishes what it is
holding. What it can no longer do is sit there indefinitely for free.

> **The new rule: running out costs a death, and a death never makes the next
> life unwinnable.**

## 7. Order of work

The measurement tranche is done: the baseline in two passes, the harness and
committed `results/`, reset with a real cost, the measured deadline and the
fallback disqualifier, the `autonomous` arm and A0. The capacity sweep was
**closed without flying** (issue #118) — both its axes are defined by an economy
the next batch changes again.

**What comes next is in `PluggyPlan.md`, "The next batch"**, and this section
does not keep a second copy of it. Two rules from this tranche outlive it:

⚠ **MEASUREMENT WAITS FOR THE DESIGN.** A rung measured before the world's
death conditions and points semantics settle, and one measured after, do not
describe a gradient — they describe two different experiments sharing a name.
The next instruments are derived from the five qualities (#155 designs them and
flies nothing), after the batch lands.

⚠ **A GATE IS NOT A SERIES.** What still runs in the meantime is capability
gates: cheap, version-local, pass/fail questions that decide the next milestone
and are not expected to survive a change to the world ("does the agent set a
standing order at all?"). Do not run a gate through the full N ≥ 5 machinery —
that machinery exists to make a series comparable, and a gate is not trying to
be. The probe answers most of them without a sim at all, which is the lesson
the deadline taught once already.

## 8. Writing it down as it is collected

⚠ **A RESULT THAT WAS NEVER EXPLAINED IS A RESULT NOBODY CAN READ, INCLUDING US
IN THREE MONTHS.** `results/` holds numbers; it does not hold what they mean.
The website's data page is deliberately built *after* the first experiments, so
that the page does not shape the experiments around what renders nicely — but
the **explanation** is written when the data is collected, not when the page is.

So every **series** lands with a short written entry: what was run, what the
numbers were, what changed since the last set, and what it does **not** show.
§3's result sections are the format.

⚠ **A GATE IS NOT A SERIES, AND DOES NOT GET ONE.** A gate reports into **the
decision it informs** — the PR, the issue it settles — and its runs are
committed as evidence rather than narrated as a finding. `notes.json` is keyed
per series for the same reason: a gate has no series to be an entry for. A0 is
the worked example; its findings belong in the economy issues they produced,
not in a results narrative read against a world that will not exist by the time
anyone opens it. The test for which one you have: **would this number still
mean something after the next change to the world?**

### Where the entry goes: `results/notes.json`

Committed and stale-checked beside the records (`evaluation/notes.py`), one
entry per **series**, named by `notes.series_id` so a typo addresses nothing and
says so. Four fields:

| field | what it holds |
|---|---|
| `ran` | the configuration, in a sentence — what a reader needs to know the numbers are about |
| `found` | what the set says, as bullets. The *reading*, not the numbers |
| `changed` | what moved since the last set on this series. Empty prose is a claim ("nothing did"); an absent key is not |
| `notShown` | what the set does not establish |

⚠ **AN ENTRY NEVER RESTATES A NUMBER THE ROLLUP ALREADY CARRIES.** A second
copy of `fallbackRate` in prose is a copy that goes stale silently, and both
the rollup and this file are open in front of the reader. What belongs here is
the sentence a column cannot hold.

⚠ **THE FIELDS ARE PLAIN PROSE, NOT MARKDOWN.** The website renders them as
text — a backtick shows up as a backtick, and `--` as two hyphens.

⚠ **`notShown` IS NOT OPTIONAL AND IS REFUSED WHEN EMPTY.** It is the half a
writer skips and the half a reader most needs. `tests/test_experiment.py`
checks both directions of coverage as well — a new series with no entry, and an
entry left behind by a series that was re-flown under another name. A series
that has merely gone *stale* keeps its entry: what it meant is still what it
meant.
