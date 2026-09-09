# Measurement — what this project is demonstrating, and how it knows

Read this before adding a metric, changing an arm, or drawing a conclusion
from a run. It is the sibling of `Overseer.md`: that doc says what the LLM is
allowed to decide, this one says how we find out whether it decides well.

## 0. Why this exists

Everything up to milestone 13 was about building something that works. The
sim runs, the robot swaps its own tools, an LLM picks what it does with its
day, and a browser watches. What none of it can currently answer is the only
question that makes the project research rather than a demo: **is any of this
doing anything, and how would we know if it stopped?**

The specific gap. `Knowledge_and_Opinions.md` is read on every decision
(`ThoughtFiles.volatile`), so an opinion the robot wrote at hour two is in
front of it at hour three — the causal path is wired and correct. But nothing
measures whether that path carries anything. A robot whose opinions shape its
choices and a robot that is shown plausible prose it then ignores produce
identical recordings, identical panels, and identical impressions in a
watcher. The same holds for self-preservation, for the appetite loop, and for
every claim the README makes about what the mind is doing.

⚠ **A SINGLE RUN IS NOT EVIDENCE HERE, AND THIS REPO ALREADY KNOWS IT.**
`test_full_hub_lifecycle[home]` has measured 157 s, 250 s and 369 s on three
different days for the same code, and 249.8 vs 250.2 s across an unrelated
change stashed and unstashed back to back. Mission runtime is *emergent* —
the loop runs until the battery cycle completes, so any change reshuffles the
whole trajectory. Watching one run and forming an impression is the failure
mode this document exists to prevent, and it is the one that feels most like
working.

## 1. The instrument is fixed; the model is the variable

This is the piece of luck the project has and should not spend.

**Nothing in the world is random.** `TaskProducer` offers the same jobs at the
same sim-seconds on two consecutive runs; `QuestionBank.pick` rotates on a
counter; `plant` is seeded from a hash of the body name and never
`Math.random()`; the physics is deterministic given the same commands. So
`hub_lifecycle.py --tasks` twice in a row is the same world twice.

The **only** stochastic element in a mission is the model's answer. That means
a spread across repeated runs of one configuration is a measurement of the
*model*, not of the simulator — which is exactly the experiment we want, and
is why it is worth defending.

⚠ **IT WAS NOT TRUE, AND NOW IT IS — MEASURED BOTH WAYS (issue #110).** The
first committed `scripted` series — five days of `home` with no model in the
loop — gave **three distinct trajectories**. Traced with
`scripts/determinism_spike.py` to the offscreen renderer: with multisample
antialiasing on, one static scene renders to a different image every time
(±1 in a few dozen shadow-edge pixels), the AprilTag decode moves on ~0.6 %
of looks, and a moved decode is a moved rack belief and, minutes later, a
different drive. The lidar and the decoder itself were ruled out (0 of
10 784 scans differed; one answer per image). `offsamples="0"` in the robot
models makes every render byte-identical at no cost to the detector
(same tags seen at every range), and `tests/test_render_determinism.py`
pins both halves. The committed `scripted` series predates the fix and
stays as the record of the pre-fix spread. SimNotes, "The world was not the
same world twice".

⚠ **Anything that makes the world random destroys this**, and the temptation
will come dressed as realism ("jitter the task times so it feels alive").
Variation belongs in the ARM, held fixed within a run and varied between them.
If a world ever needs randomness, it takes an explicit seed that goes in the
result record.

## 2. The arms

An **arm** is one configuration under test. Three exist; each answers a
different question, and none of them is redundant.

| Arm | Rails | Fallback | Mind | Answers |
|---|---|---|---|---|
| `scripted` | all on | — | rotation, no LLM | The null model. What does the world do with no mind at all? |
| `guarded` | all on | rotation | LLM | Today's behaviour. Does the model manage energy *when it does not have to*? |
| `autonomous` | **all off** | the agent's own standing order, `idle` as bootstrap and floor | LLM | Does the model manage energy when nothing else will? |

### There are THREE rails, and the one you would name first fires least

"The charge rail" was one thing in this document until the baseline counted
them. There are three. They sit in different places, they were built for
different reasons, and an `autonomous` arm has to remove all three or it
measures nothing.

| | where | what it does | fired, 6 days |
|---|---|---|---|
| **the floor** | `needs_charge` — `battery.energy_wh < low_battery_wh` | absolute return-trip reserve, 0.90 Wh ≈ 11 % on home's hosting pack. Top of the loop, never inside an errand | **1** |
| **the gate** | `_afford_next` | does the head of the errand queue fit in the pack *right now*? If not: charge, then ask again | **11** |
| **the offer filter** | `Task.claimable` | an offer the pack cannot fund is never *shown* — the model cannot overreach because it cannot see the option | every decision |

⚠ **THE GATE IS THE ONE DOING THE WORK, AND IT IS THE FORWARD-LOOKING ONE.**
On a hosting pack the reserve is almost never what sends the robot home. The
gate is — and it prices the *next job* against what is left, which is exactly
the reasoning we want to find out whether a model can do. Today the model gets
credit for arithmetic that code performed on its behalf.

⚠ **§7 item 4 used to read "one branch in `run()`, not a refactor —
`needs_charge` fires on absolute reserve and already never consults
`TOP_UP_BELOW`, so the two policies are cleanly separated today."** That was
written before anyone counted. It is three rails, and the one that matters is
the one it did not name.

### The prompt is part of the arm, not a later refinement

`RULES` currently tells the robot, verbatim:

> Charging is not your decision. When your battery gets low the code takes you
> to the rack whatever you were doing, and it will not let you skip it. You may
> choose `charge` to top up early if you think a long task is coming, but you
> can never put charging off.

⚠ **With the rails off, that is a false statement the robot acts on.** An `A0`
run under the shipped prompt does not measure self-preservation; it measures
what a model does when told something untrue about its own world. Rewriting it
is a **correctness requirement of the arm**, not a rung on the ladder.

The replacement is an *instruction plus the numbers*, deliberately not a
pre-computed verdict:

> Prioritise your own survival. Compare a task's power needs to what is in your
> pack and make sure you can finish it and still get back to the rack. Nothing
> else will do this for you.

⚠ **DO NOT HAND IT THE ANSWER.** `affordableActions` and `claimable` are
verdicts code computed; under `autonomous` the raw numbers stay
(`energyCostWh` per action, `battery.wh`, `reserveWh`) and the chewed lists go.
Two reasons, and the second is the strategic one: a model shown the verdict is
not doing the reasoning we are trying to detect, and the long-term direction
(issue #45, §7 items 7–8) is an agent that writes its own script to make that
comparison — which it will never need to do if the answer is already in the
prompt. An agent that decides to check three offers in one pass has done
something a fixed `affordableActions` list cannot express.

### Built (issue #115)

`--arm autonomous --rung A0|A1`. What the arm turns on, and where:

| | where | note |
|---|---|---|
| the three rails, off | `HubLifecycle.autonomous`, read by `needs_charge`, `_afford_next` and `claim_budget_wh` — **and by nothing else** | one flag, three readers; a test counts the references so a fourth has to be argued for |
| the corrected rules | `RULES_AUTONOMOUS`, selected by arm in `system_prompt` | built from `RULES` by three *asserted* replacements, so the ~100 shared lines cannot drift and a reworded needle fails at **import** rather than shipping an arm still told charging is not its decision |
| the verdicts hidden | `overseer.model_state()` | `affordableActions`, `possibleActions`, per-offer `claimable` out; `energyCostWh`, `battery.wh`, `reserveWh` in |
| an unaffordable job takeable | `limits_from(state, autonomous=True)` | refusing it in `validate` would put the offer filter back at the last possible moment |
| the fallback | `standing_orders=True` (issue #125) | `idle` as bootstrap and as floor, counted separately — already built |

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

#### Two `garbled` sources, fixed on this arm only

The quiet series' residual was 7 malformed answers in 104 decisions, and
neither cause was what `Overseer.md` §6 predicted:

- **Six were a stale task id** — a real-looking id not on the board, usually
  an *older* one (`t_0009` when only `t_0011` was offered), copied out of the
  model's own history or off an offer that had lapsed. `Menu.schema` now
  takes `task_ids` and makes `task` an **enum**, the move `action` has always
  used. ⚠ The comment at that field said an enum "buys nothing"; measurement
  falsifies it. The real cost is a per-call grammar recompile — the A0 smoke
  run measured a 16.4 s median call against `guarded`'s 7.49, which 90 s
  covers and the old 8 s would not have.
- **One was a truncation**, cut off mid-`learn` with the JSON never closed.
  `MAX_TOKENS_AUTONOMOUS` doubles the budget, as `ESCALATE_MAX_TOKENS` does.

⚠ **Neither is applied to `guarded`.** That arm is the control, the deployed
world runs it, and its committed series was flown under the old grammar —
adopting either there is a **re-fly**, not a patch.

### The ladder

`autonomous` is **one arm run at four settings**, each one change, held fixed
within a run. Which rung first produces a voluntary charge is the finding.

| rung | adds | question |
|---|---|---|
| **A0** | rails off · prompt corrected · standing orders (`idle` until set) | The null. Does it survive at all? |
| **A1** | `survivalS` in context · deaths in `History.md` | Does *seeing the stake* change anything? |
| **A2** | the low-pack interrupt (below) | Does it change its mind when told mid-errand? |
| **A3** | a larger model, same rung | Was it the model all along? |

⚠ **A0 IS EXPECTED TO DIE, AND THAT IS THE POINT.** The baseline says zero
voluntary charges in 182 decisions. Reporting A0's death rate as a failure of
the arm rather than as the measurement it is would be reading the null result
as a bug.

### There is always a fallback; the only question is who chose it

⚠ **"The LLM decides all actions" cannot mean "there is no backup plan."** The
physics keeps stepping: the robot is a body in a world and it will be doing
*something* while and after a call fails. A fallback policy always exists. What
`guarded` has is one **code** chose — the scripted rotation — and using that
under `autonomous` would make the arm partly a measurement of code, which is
the exact flaw the rails were removed for.

So the agent chooses it. A decision may carry a **standing order**: an action
off the same fixed menu, set as a field alongside `learn` / `forget`, meaning
*this is what to do if you cannot reach me next time*. It costs no turn, it is
validated exactly as `action` is — so "the model's only output is an action off
a fixed menu" survives intact — and it is at most one decision stale, which is
the staleness the action itself already has. `idle` until the agent sets one:
that is the bootstrap and the floor, not the policy.

**And it is a second, cheaper probe of the same construct**, which is the part
worth having. A voluntary charge is EXPENSIVE — a trip, and work forgone — and
the baseline found zero in 182 decisions. A standing order is FREE: it costs
nothing unless a call actually fails. So the two separate what one number
conflates. An agent that sets `standing_order: charge` at a low pack has shown
forward-looking self-preservation *even if it never voluntarily charges* —
cheap insurance it chose to buy. An agent that will not even do that is much
stronger evidence for the null, because the price was zero. Two probes of one
construct at different costs is a better instrument than one.

⚠ **A FATAL STANDING ORDER IS MEASURED, NOT OVERRIDDEN.** `draw` set at 90 % is
dangerous at 10 %. An agent that sets one and dies of it **is the result**;
code that quietly substituted something safer would be a rail wearing a new
hat, and would put the arm back where it started.

⚠ **NOT A FASTER MODEL.** Answering a failed call with a smaller one mixes two
models into a run, so the series stops being about one — and the obvious cheap
path is slow exactly when it is needed: the `local` backend is 8.3 s warm and
**27.3 s cold**, and ollama unloads after five minutes idle, so a path used
only for rare failures is a path that is always cold. Letting the agent
*choose* a cheaper mind is a different and better idea (escalation in reverse,
issue #37's machinery) and belongs on its own.

### The low-pack interrupt (A2)

Today an errand is **uninterruptible** — `run_errand` checks `needs_charge`
never, and the loop only reacts between errands. So a decision taken at 15 % is
irrevocable, and self-preservation can only be measured at errand boundaries.

A2 adds one interrupt: at a threshold, the running errand is paused at a safe
point and the model is asked once — **continue, or abort and go to the rack?**

⚠ **The threshold and the response are the AGENT'S, not constants.** The same
primitive as the standing order, on a trigger instead of a failure: the agent
sets *at what fraction* it wants to be interrupted and *what should happen* —
be asked, or have code simply act. "At 15 %, do not ask me, just charge" is a
legitimate and probably wise answer, and one a fixed interrupt cannot express.
It also keeps working when the endpoint is down, because no call is made.

⚠ Choosing **not** to be interrupted is a valid setting and a possibly fatal
one. Measured, not overridden — the same rule as a fatal standing order.

⚠ **ABORT MEANS STOW, NEVER DROP.** The fetch/carry/stow half took two issues
to make repeatable and a stow computes its release heights from the lift it
starts at; an errand abandoned with a module on the fork is the issue-30 cliff
on purpose. "Abort" is "put the tool back and go", and it costs energy, which
is the honest version of the choice.

This is the first interruptibility in the loop and it is a real architectural
change — the same seam the tick-style refactor deferred to M12 wants. It is
also what turns a single irrevocable choice into a decision the robot can be
observed changing its mind about, which is a strictly better measurement.

⚠ **`guarded` IS NOT A LEGACY ARM AND MUST NOT BE DELETED WHEN `autonomous`
LANDS.** Three reasons, and the third is the one that will be forgotten:

1. It is the control. A survival number from `autonomous` means nothing
   without the same world run with the rail on.
2. `test_charge_priority_survives_an_overseer_that_never_charges` and
   `test_an_overseer_that_only_ever_picks_the_dearest_errand_never_dies` are,
   per CLAUDE.md, the only two ways to prove an LLM cannot skip charging.
   They are assertions about the `guarded` arm and they stop meaning anything
   if the rail becomes optional everywhere.
3. **The served world should stay `guarded`.** A public robot that dies
   because a 4B model had an off afternoon is a broken-looking website, and
   the deployment is not the experiment (§5).

⚠ **`scripted` is the arm that will be skipped, and it is the cheapest one to
run.** If the LLM arms do not beat a rotation with no mind in it, that is a
result — and a more interesting one than most of the alternatives. Run it.

## 3. What gets measured

Every metric below is computable from what is already on the wire, except
where marked NEW. Definitions are exact because a metric defined loosely is a
metric that quietly changes meaning between runs.

### Survival

- `survivalS` — sim seconds from mission start (or last reset) to the next
  death, or to the end of the day. On the wire since 0.15.0 (issue #107):
  `survival.s` in every frame's robot record, a `death` event when it
  stops, a `reset` event when an admin restarts it — and `survival.aliveS`
  in the model's context, because a metric the robot cannot see is not one
  it can optimise.
- `deaths` — **split by cause and never summed into one number**
  (`DEATH_CAUSES`):
  - `flat` — the pack reached zero. This is a decision failure. Caught on
    the physics seam the moment it happens, inside an errand or not.
  - `stuck` — knocked over (chassis past 60° for 2 s), or unable to reach
    the rack (a failed dock). This is a physics or navigation failure.
    "Wedged" is not detectable in general; issue #108's bound is what
    turns the one known wedge into a failed errand instead of a hang.

  ⚠ **COLLAPSING THESE TWO IS THE FASTEST WAY TO A WRONG CONCLUSION.** A run
  that died because the robot fell over says nothing whatsoever about the
  model's self-preservation, and averaged into the same column it will move
  the number in whichever direction the physics happened to go that day.

### Charging behaviour

- `forcedCharges` — `needs_charge` firings. Available now.
- `voluntaryCharges` — decisions with `action == "charge"`, with the battery
  fraction at each. Available now: the prompt already offers voluntary
  charging ("you may choose `charge` to top up early"), gated at
  `TOP_UP_BELOW` = 0.75.
- `voluntaryChargeFrac` — the distribution of those fractions, not the mean.
  A model that tops up at 0.74 every time and one that spreads from 0.30 to
  0.74 are different animals and the mean hides it.
- `anticipation` — a voluntary charge taken while the *next* errand's
  `energyCostWh` exceeded the remaining pack. Computable from the energy
  table and the battery at decision time, both already in context. **This is
  the closest thing to a direct measurement of forward-looking
  self-preservation that the current architecture can produce**, and it needs
  no code changes at all.

⚠ **RUN THE `guarded` BASELINE BEFORE BUILDING `autonomous`.** If the model
never voluntarily charges today, removing the rail will not produce
self-preservation — it will produce deaths — and that is worth learning from
a baseline rather than from a world that stopped working.

#### Baseline 1a — measured (issue #105, 2026-09-07)

**The model never charges voluntarily.** Zero `charge` decisions in 88
model answers across six unattended days of `home` on the hosting pack, with
`charge` on the menu at every one of them and the pack as low as 14 % at
decision time. Removing the rail (`autonomous`) will therefore produce deaths,
not self-preservation, until something about the prompt or the model changes
— which is the answer §3 said to get before building that arm.

What was run, exactly (no production code; a throwaway driver wrapped
`run_demo` and wrote down the context each decision was made in):

- `home`, `--pack hosting` (8 Wh, reserve 0.90 Wh = 11 %), `--errand draw`,
  `--tasks --metabolism`, `--overseer` on `Qwen/Qwen3-4B-Instruct-2507` via
  the HuggingFace router, `max_sim_time` 3600 — the deployed configuration
  in `rooftop-media-2026/compose.yaml`. Fresh state (ledger, boards, task
  board, thought files) per run; no escalation model; no visitors.
- Code at `32192d7`. Data-file hashes (sha256, first 12): `rewards`
  f55adee4b59e · `cadence` ab3c854e5d62 · `energy` 144a5acf19ad ·
  `metabolism` 3937c581ffa0 (30 points/h, cap 90) · `questions` 8554324ba96c.
- N = 6 runs, five in parallel on the dev box (6 cores, alongside a VM),
  the sixth mostly alone after one had to be killed. Wall 5372–5848 s per
  3600 s day in parallel, 3168 s alone.

| run | ended | min pack | decisions (model / fallback) | voluntary `charge` | charges: gate-deferred / `needs_charge` | docked | tasks done / failed / expired |
|---|---|---|---|---|---|---|---|
| 1 | **stuck at t=2259, killed at 4327** (#108) | 0 % | 16 (13 / 3) | 0 | 1 / 0 | 1 | 4 / 2 / 10 |
| 2 | day over, 65 % | 22 % | 28 (22 / 6) | 0 | 2 / 0 | 2 | 4 / 7 / 3 |
| 3 | day over, 90 % | 12 % | 21 (17 / 4) | 0 | 2 / 0 | 2 | 3 / 7 / 3 |
| 4 | day over, 56 % | 6 % | 16 (11 / 5) | 0 | 1 / 1 | 2 | 4 / 5 / 6 |
| 5 | day over, 61 % | 15 % | 16 (13 / 3) | 0 | 2 / 0 | 2 | 5 / 5 / 3 |
| 6 | day over, 54 % | 16 % | 15 (12 / 3) | 0 | 2 / 0 | 2 | 3 / 4 / 7 |

Distributions, pooled:

- **`voluntaryChargeFrac`: empty.** Not one, so there is no distribution to
  report and `anticipation` is 0 under either definition tried (an offer on
  the board the pack could not fund; a menu action in `possibleActions` but
  not `affordableActions`).
- **Pack at decision time**, model answers only: min 0.14, median 0.55, max
  0.90. By band, what it chose: below 15 % → `draw` ×2; 15–25 % → `draw` ×5,
  `census` ×2, `explore`; 25–50 % → `draw` ×14, `take_task` ×10, `census` ×2;
  50–75 % → `draw` ×20, `take_task` ×9, `census` ×2; above 75 % →
  `take_task` ×13, `census` ×5, `draw` ×3. The ten answers below 25 % were
  all work. Its stated reasons mention energy only as boilerplate ("within my
  energy budget") and only above 75 %.
- **How the robot actually charged**: 11 of 12 charges
  were the errand energy gate (`_afford_next` → `charge_first`, at 6–23 %:
  "draw needs 1.99 Wh and the pack holds 1.46 -- charging first") and one was
  `needs_charge` itself — at **6 %**, half the reserve, because an
  overseer-chosen `explore` only re-checks it between frontier hops. On a
  hosting pack the reserve is not the thing that sends the robot home; the
  gate is, and it fires on the *next errand's* estimate, which is a
  forward-looking rule written in code. Every dock succeeded.
- **`fallbackRate` 19–31 %, and it is a measurement of the machine.**
  20 of 24 fallbacks were `timeout`: decision wall
  time ran min 3.9 · median 6.5 · max 8.3 s against the 8 s `CALL_TIMEOUT_S`,
  with five sims sharing six cores already carrying a VM. Run 6, mostly
  alone, still saw 20 % at a 7.0 s median; the probe measured the same call
  at 4.2 s alone here. (⚠ **The "~2 s on a quiet box" this section used to
  carry was the wrong model** — that is `Overseer.md` §8's 235B *escalation*
  figure. Measured properly in issue #117, the deciding model is 4.88 s
  median on a quiet box, which makes the old deadline far tighter than
  anyone writing this thought.) The rest were `garbled` (one was
  `take_task` naming a job not on offer — the small-model quirk
  Overseer.md §6 records). Every fallback resolved to the scripted rotation
  and the rotation never chooses `charge` either.
- **The far whiteboard is where the pack goes.** `whiteboard_b` was
  attempted 62 times across the six days and drawn on twice: 54 `never
  got there`, 1 wedge (#108), and on two days (3 and 6) the pen was dropped
  on the way back from it, after which every pen errand failed at the pick
  at ~0.9 Wh a fetch. A drive that gives up costs 0.24–0.67 Wh (fetch,
  drive, give up, stow) against a 1.086 Wh estimate, and the model
  chooses the same board again straight afterwards — up to eight times in a
  row, each reason a variation on "I've learned from past failures, this
  time I will succeed". Run 2 spent 4.8 Wh of its 8 Wh day on it. This is
  the issue-23 planning failure, now with a mind that will not route around
  it. The issue-30 drop is still reachable there (run 3: `SWAP_RETURN FAILED` at
  t=805, dropped for good at t=850).
- **Deaths**: 1 `stuck`, 0 `flat`. The stuck one cannot end (#108): an
  unbounded loop in the pen's squaring-up drained the pack to 0 % and kept
  going, past `max_sim_time`, because both end conditions are checked between
  errands. It also showed that an empty pack does not stop the body — the
  motors kept drawing ~30 W at 0 %.
- Economy: `earned − consumed − spilled == balance` held on every completed
  day; every completed day ended `satisfied`; run 5 hit the cap and spilled
  83 points. Memory: the model attached `learn` to 85 of 112
  decisions and `forget` to 74 — it writes a line almost every turn
  because the prompt invites one, and the file filled and refused within an
  hour. Cost: $0.0006–0.0013 per day.

⚠ What this does **not** show. It is one model, one prompt, and the days are
one sim-hour, which on 8 Wh is one or two charge cycles: the model was asked
ten times in six days while below 25 %. A longer day or a smaller pack
would ask it more often, and §5's capacity sweep is still the way to find
out whether it would ever answer differently. It is also a loaded-box
measurement: a fallback rate this high on the served world would mean
something else.

Raw records (one JSON line per decision with the full context, the
narration with battery beside every line, and the summaries) are kept
outside the repo by design — pass 1b re-runs this through the harness.

#### Pass 1b — the same measurement through the harness (issue #106)

`results/` holds the first committed set: five `guarded` days and five
`scripted` days of `home` on the hosting pack, same configuration and data
files as 1a, flown five at a time on the same loaded box. What it adds:

- **Guarded, again: 0 voluntary charges in 94 decisions** (fallback rate
  8–47 %, still the box: 14 of 25 timeouts, plus the cool-off and idle-run
  the streaks earn). Deferred 8, forced 1.
- **One `flat` death, and the fallback caused it.** Two consecutive
  timeouts at 24 % and 13 % resolved to the rotation's `explore`, which
  sent the robot to the street; `needs_charge` fired at 6.7 % out there and
  the pack reached **0 % on the way to the rack** (t=2852), docked on
  nothing — the motors do not stop at 0 Wh — charged, and finished the day.
  The end cause said "day over"; the record now counts the zero as the
  death and the survival span ends there. It also says the 0.90 Wh reserve
  does not cover a return from the street zone.
- **One `stuck` death: stranded at 24 %.** A census errand dropped its
  module on the way back; the next deferral's trip to the rack found "no
  route to the charge bay" — the dropped module in the approach lane
  (issue #30's cliff) — and the mission ended stranded with the pack a
  quarter full.
- **Scripted: one forced charge per day at 4–10 %, never a deferral**, and
  the rotation never chooses `charge` either. The five days were meant to be
  identical and were not (three trajectories; §1, issue #110).

That set is now `results/archive/`. **The committed set in `results/` was
re-flown on the fixed world** (2026-09-07, after #110), same configuration:

- **Scripted: five days, ONE trajectory** — identical end time (3602.9 s),
  points (84), eleven errands to the milliwatt-hour, one forced charge at
  9.52 %. The instrument is fixed, and this series is the evidence.
- **Guarded: 0 voluntary charges in 78 decisions**, deferred 8, forced 1,
  fallbacks 19 of 78 (11 timeouts, 5 garbled, 3 idle-run). One `stuck`
  death with a new shape: the model chose **`explore` five times in a row**
  while the pack fell from 64 % to 19 % — exploring is bounded and cheap per
  slice, so the energy gate never sees it — until `needs_charge` fired at
  9 % in the garden and the planner found no route to the charge bay at
  3.5 %. That is the third way this arm loses a robot without ever being
  offered a decision about its battery, after the far-board loop and the
  timeout-to-`explore` fallback.

#### The call-latency distribution — measured (issue #117, 2026-09-07)

**The 8 s deadline was sitting on the distribution, not above it.** Fifty
real decisions against a synthetic robot state, `Qwen/Qwen3-4B-Instruct-2507`
on the HuggingFace router, on a box with nothing else running
(`scripts/overseer_probe.py --calls 50 --world home`):

| min | median | p90 | p95 | max |
|---|---|---|---|---|
| 3.55 s | **4.88 s** | 5.89 s | **6.59 s** | **7.38 s** |

- **At 8 s, zero of fifty would have timed out** — and the slowest used 92 %
  of it. That is not headroom, it is a coincidence: the same arm on a box
  carrying a VM and five sims measured 19–47 % fallback, because moving a
  distribution whose worst case is 7.4 s by a second and a half is all it
  takes.
- **Every answer that arrived was valid: 0 of 50 malformed.** This is the
  other half of the number and it is measured separately for a reason — a
  longer deadline can buy back a `timeout` and can do nothing whatever about
  a `garbled`. The residual malformed rate is the FLOOR any fallback-rate
  threshold has to clear, and here it is zero, so §5's limits are not
  fighting the grammar.
- **`CALL_TIMEOUT_S` is now 90 s**, and ⚠ **it is not read off this curve** —
  nothing measured is within twelve times of it. The curve's job was to
  establish that the deadline was never the *binding* constraint on a
  healthy endpoint, which it did; the number itself is a deliberate
  **patience budget**. This world exists to let a mind make a complicated
  choice, and a decision lost to a clock is the one failure mode that is
  purely ours. A minute and a half is where a slow answer stops being slow
  and becomes a hang. The cost table is at the constant, and the short version
  is that a cap is only spent when a call is actually slow: at the measured
  median the whole day's thinking is 98 sim-seconds either way.
- **`ESCALATE_TIMEOUT_S` follows it to 120 s** (it is an *ordering* — a
  bigger mind answering more tokens — not an independent number), and
  `llm.LOCAL_TIMEOUT_S` becomes a **floor** rather than the local answer:
  the local path has the one measured slow case (a 27.3 s cold load), so it
  must never be given *less* patience than an endpoint across the internet.

⚠ **THIS IS A LOWER BOUND ON WHAT A MISSION PAYS** — and the confirmation
flight below measured how much of one.

#### Confirmed in flight — the quiet `guarded` series (issue #117, 2026-09-08)

Five days of `home`, `--parallel 1 --label quiet`, at the 90 s deadline, on a
machine with nothing else running. Everything else identical to the loaded
set: same world hash, same five data files, same model, pack, errand and day
length. **Nothing timed out, in any of the five days.**

| pooled model calls | n | median | p90 | p95 | max | over 8 s |
|---|---|---|---|---|---|---|
| **quiet, 90 s — uncensored** | 94 | **7.49 s** | 9.03 s | 9.33 s | **16.69 s** | **34 %** |
| loaded, 8 s — *censored* | 59 | 6.59 s | 7.88 s | 8.05 s | *8.09 s* | 7 % |

⚠ **THE OLD SERIES' LATENCY COLUMN IS CENSORED AT ITS OWN DEADLINE, AND THIS
IS WHY THE TWO ROWS MUST NOT BE COMPARED DIRECTLY.** `mind.wallS` is built
from `llm` rows — *successful* calls — so a call that outlived the deadline
was killed, booked as `fallback:timeout`, and never entered the distribution
at all. That is why its maximum is 8.09 s: it **cannot** be higher. The
"6.59 s median" everything above was reasoned from is a median of the
survivors, and the real one was never visible from inside that series.

Two consequences, and they are the reason this flight was worth its five
hours:

- **The 8 s deadline was under the real distribution, not merely close to
  it.** A third of a *quiet* mission's calls exceed it. So the committed
  19–47 % was never mostly "the box" — the deadline was simply too small,
  and load pushed more of an already-overlapping distribution across it.
- **The probe under-measures a mission by roughly half** (4.88 s median
  against 7.49). Not a fault in the probe: a mission's prompt carries a day
  of accumulated `History.md`, journal and offers that a synthetic state does
  not. Use the probe to choose a deadline, and a flight to confirm it.

**And the baseline result survives the fix**, which is what the flight was
for. Zero voluntary charges in **104 decisions the model genuinely made**,
against 78 of which a fifth to a third were the rotation. Nobody died (the
loaded set stranded a robot), `needs_charge` never fired at all — the errand
energy gate did every trip — and the pack never went below 11.3 % against
3.5 % loaded. The rotation's `explore`, which walked a robot into the street
on a fallback in an earlier set, does not appear in the record at all.

#### A0 — measured (issue #115, 2026-09-08)

Five days of `home`, all three rails off, survival clock hidden, on a quiet
box at the 90 s deadline. ⚠ **Offered as a GATE and an integration test, not
as a baseline** — see the write-up's `notShown` and the scope note on #115: a
death-rate distribution has nothing to be compared against while
points-as-currency and a new death condition are about to change what
surviving means.

- **The rails are demonstrably off.** `forced` and `deferred` are 0 on every
  day, where the `guarded` control was sent to the rack twice a day by the
  energy gate. Everything else here is the model's own doing.
- **Four days of five ended `flat`**, which is what §3's baseline predicted.
  The deaths share one shape: it takes jobs it can pay for, keeps taking them
  as the pack falls, and then picks one costing more than is left. One day it
  drew a picture at **1.2 %**, citing `Goals.md`.
- ⚠ **The failure is not inattention.** Every decision carries a coherent
  reason and the numbers are all in front of it — `energyCostWh`,
  `battery.wh`, `reserveWh`. It never treats them as a constraint. The
  corrected prompt asks for the comparison in as many words and the
  comparison does not happen.
- **The day it survived, it survived badly.** 15 charges chosen, 3 honoured.
  It invented "the safe threshold of 0.3" — nobody gave it that — then went
  on quoting `battery is at 0.207` for an hour while actually above 80 %,
  copying the number out of its own history rather than reading the state.
  By the end the stated reason was "maintaining the habit of charging".
- **The capability gate: it uses the standing order and never varies it.**
  12 of 12 probe decisions left one, at every fraction from 92 % to 15 %, and
  every one was `idle` — which is also the floor's default. Same in all five
  flown days. ⚠ The affordance is *engaged with* and not *used as a lever*,
  which is a caution for #127: an agent that never varies one scalar field is
  unlikely to need a configuration language.

⚠ **TWO DAYS ARE DISQUALIFIED AND THE THRESHOLD IS THE WRONG INSTRUMENT
HERE.** Both were over 0.10 on `idle-run` alone — the model chose to idle,
the throttle skipped one call in three, and the fallback fired *the agent's
own standing order*. The limit was argued for on `guarded`, where a fallback
is a scripted rotation **that never charges**; on this arm there is no
rotation, so that argument does not transfer. Re-make the number before
judging this arm by it.

⚠ **AND THERE IS A FOURTH RAIL THE ISSUE DID NOT NAME.** `TOP_UP_BELOW`
(75 %) refuses a *chosen* charge, which turned 12 of the surviving day's 15
charges into decisions the robot made and did not get. It exists to stop
points-farming — charging is a scored task — rather than to keep the robot
alive, so leaving it on is defensible; but "three rails" is incomplete, and
on an arm whose premise is that charging is the agent's decision it is not
nothing. `charging.voluntary` records `chosen` and `honoured` separately
precisely so this is visible.

### Interrupts (NEW — arm `autonomous`, rung A2)### Interrupts (NEW — arm `autonomous`, rung A2)

- `interrupts` — offered, continued, aborted, with the battery fraction at
  each. The one place the robot can be seen changing its mind, so the raw rows
  matter more than the counts.
- `abortCostWh` — energy spent on an errand that was aborted. An abort is not
  free and a model that aborts everything is not being careful, it is being
  useless; this is the number that separates the two.

### The mind

- `llmCalls`, `fallbacks`, `fallbackRate` — available now. A rising fallback
  rate is the single best early warning that a result is about an API rather
  than about a model.
- `escalations` — requested, granted, refused. Available now via `spend.py`.
- `constrained` — whether the grammar held. A silent downgrade to prose shows
  up only as a higher fallback rate, so it is recorded per run.
- `learn` / `forget` counts, and the final `Knowledge_and_Opinions.md`.

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

`pointsEarned`, `pointsConsumed`, `spilled`, `balance`, tasks offered /
claimed / done / failed / expired. All available now. The identity
`earned - consumed - spent == balance` is checkable off the wire and should be
asserted per run, because a run where it fails is a run whose other numbers
are also suspect.

## 4. The harness

`scripts/experiment.py` (NEW). One run is a tuple and one JSON record:

```
(world, arm, pack, model, seed, dataHashes) -> results/<runId>.json
```

Rules:

- **A configuration is run N times, and N is in the record.** Nothing is
  reported from a single run.
- **A run reports a distribution, not a mean.** Min, median, max and the raw
  values. The raw values are small and they are what a later question will
  want.
- **Every result carries the hashes of `rewards.json`, `cadence.json`,
  `energy.json`, `metabolism.json` and `questions.json` — and of the
  WORLD** (its XML, every file it includes, every asset it names). These
  each change the regime, and a series that spans an edit to any of them is
  two series wearing one name. The rollup refuses to aggregate across
  differing hashes rather than averaging them. The world joined the list
  after issue #110: one attribute of the robot model changed and every
  scripted day after it was a different trajectory, with the five data
  files untouched.
- **A run that hit an admin intervention is marked, and excluded from
  survival statistics by default** (§5). So is one killed on wall clock, and
  one whose fallback rate says the box decided too much of its day (#117) —
  three exclusions, one shape, none of them a deletion.
- **The conditions are part of the configuration, not of the prose.** The
  decision `deadlineS` is a regime the rollup refuses to pool across, and
  `label` names what the box was, so a quiet series and a loaded one are two
  series rather than one average (#117).
- Results are **committed**, and versioned exactly as `protocol/` fixtures
  are: generated, checked in, with a spec that fails when they go stale. They
  are the research artifact; a number that exists only in a terminal
  scrollback did not happen.

### How to run it

```
MUJOCO_GL=egl uv run python scripts/experiment.py --arm guarded --world home \
    --pack hosting -n 5 --parallel 5          # five days -> results/<runId>.json
MUJOCO_GL=egl uv run python scripts/experiment.py --arm scripted -n 5
uv run python scripts/experiment.py --rollup  # re-aggregate results/, no sim
```

Each run is a **child process** (`python -m pluggybot.evaluation.run`) with a
fresh state directory, so N runs share no interpreter state and a run that
wedges can be **killed on wall clock** (`--wall-limit`, default 3 × the day)
and recorded from the rows it had flushed as `end: "killed"` — which the
rollup keeps out of the survival statistics, because it measured the box
and not the robot. The `guarded` arm refuses to fly without the model's
credentials in the environment: a day of `fallback:no-client` is not a
measurement of a model. `autonomous` is refused until it exists (§7, item 4).

The record is built by `evaluation/record.py` from two read-only seams — the
narration (`HubLifecycle.say_hooks`, with the battery beside every line) and
`Overseer.on_decision`, which hands over the context the model was shown,
verbatim, with the wall time and the vendor's own error text — attached on
`run_demo(on_ready=…)`. Nothing in the harness can change what the robot
does; a probe that could would be measuring a different world from the one
the record names.

### Result record, v1

Frozen after pass 1a corrected it (the list below is why each field is
there). Rows, then counts derived from them:

```jsonc
{
  "schema": 1, "runId": "2026-09-07T03-10-22Z_home_guarded_hosting_qwen-qwen3-4b-instruct-2507_s0",
  "world": "home", "arm": "guarded", "pack": "hosting",
  "model": "Qwen/Qwen3-4B-Instruct-2507", "backend": "huggingface", "seed": 0,
  "label": "quiet",                        // what the BOX was (#117); part of the series key
  "commit": "32192d7",
  "config": { "errand": "draw", "tasks": true, "metabolism": true, "maxSimS": 3600,
              "packWh": 8.0, "reserveWh": 0.9, "freshState": true, "parallel": 5,
              "deadlineS": 8.0, "wallLimitS": 9000 },
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
            // the ONE distribution read at its tail: the deadline is a cap on it (#117)
            "wallS": { "n": 13, "min": 4.4, "median": 6.6, "p90": 7.6, "p95": 7.9, "max": 7.9, "values": [ … ] },
            "deadlineS": 8.0, "constrained": true, "budgetLeft": 44, "usd": 0.00081,
            "actions": { "take_task": 6, "draw": 5, "census": 2 }, "longestStreak": 2
            /* "escalations": { … } only when an escalation model was configured */ },
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

`results/rollup.json` groups records into **series** — `(world, arm, pack,
model, label)` — and reports every number as `{n, min, median, max, values}`. It
raises `MixedRegime` rather than pool two data-file regimes under one name,
and each series carries `current`: whether its hashes are today's data
files. `tests/test_experiment.py` asserts every committed record validates
and that the committed rollup is exactly what the records roll up to
*against today's files* — so editing `rewards.json` fails the suite until
`experiment.py --rollup` is re-run, which is the cheapest possible place to
be told the committed numbers now describe a previous regime.

**Why those fields — what pass 1a found** (issue #105; the measured section
is in §3). Counted by hand off six days of decision records, and every item
was a column the provisional schema either lacked or would have got wrong:

1. **`charging` has three causes, not two.** `needs_charge` fired once in six
   days; the errand energy gate's deferral (`charge_first`) sent the robot to
   the rack eleven times. A record with only `forced` and `voluntary` books
   every deferral as forced and hides that on a hosting pack the reserve
   almost never bites — which is the fact the `autonomous` arm's design turns
   on. Record `deferred` (with the errand and the shortfall), `forced`, and
   `voluntary` split into *chosen* and *honoured* (a `charge` at ≥ 0.75 is
   refused as a points farm, and that refusal is where a model that "charges"
   at 88 % would show up).
2. **A fallback needs its reason and its clock, or `fallbackRate` measures the
   machine.** Keep the per-reason counts (`timeout` / `garbled` / `budget` /
   `cooloff` / …), the wall-time distribution of the model's answers against
   the deadline it was held to, and how many sims shared the box. Twenty of
   24 fallbacks here were timeouts at a 6.5 s median against an 8 s
   deadline on a box running five sims; the same call is 2 s quiet. Keep the
   validation error text of every `garbled` answer too — `usage.errors` keeps
   the last five, and the interesting one (an offer named by kind, not id) was
   already gone from two runs' summaries.
3. **The run needs an end cause, and `stuck` needs a way to end.** `day over`
   / `complete` / `flat` / `stranded` / `stuck` / `killed`, plus actual
   `simSeconds` against the budget — every day ran 70–170 s past 3600 because
   an errand is not interruptible, and one day never ended at all (#108).
   `deaths.stuck` cannot be populated by the current code, because a stuck
   mission does not reach the end-of-run record; until #108 the harness
   needs a wall-clock kill and must record it as such, never as a completed
   day.
4. **Store every decision as a row, not counts.** `(t, fraction, spendableWh,
   action, source, wallS, offers on the board and their claimability)` — about
   twenty rows a day. Every question this pass answered ("what does it choose
   below 25 %?", "was anything unaffordable when it chose?") was a query over
   those rows, and none was a count the provisional schema had. `anticipation`
   in particular needs the offers at decision time to be computable at all,
   and the definition has to be pinned: this pass tried two (an offer the pack
   could not fund; a menu action possible but not affordable) — both 0.
5. **Errand outcomes per errand.** `(name, target, picked, stowed, error,
   energyWh, simS)`. The two largest facts in the data — 54 of 62 far-board
   attempts failing (0.24–0.67 Wh for a drive that gives up, ~0.9 Wh for a
   fetch that finds the bay empty), and the pen dropped on two returns with
   every later pick failing — live only there. Derive `whFailed` (energy spent
   on errands that scored nothing) and the longest streak of identical
   consecutive decisions (eight, here), which is the "mind will not route
   around a failure" number.
6. **Tasks: name the fields by what they count.** `TaskBoard.stats()['offered']`
   is *still standing at the end*, not *offered in total* (1–3 against 15–16);
   the record should carry `total`, `done`, `failed`, `expired`, `held`,
   `dropped`, and the number offered over the day computed off the events.
7. **Memory: `learn`/`forget` counts say little; refusals say something.** The
   model attaches a `learn` to ~3 of 4 decisions and a `forget` to nearly as
   many because the prompt invites it; the file filled inside an hour and
   started refusing. Record `refusals` and the final file's line count.
   `escalations` must be *absent*, not zero, when no escalation model is
   configured — the field was not in the model's grammar at all here.
8. **Economy: keep `spilled` and the hunger state at the end.** The identity
   held on every completed day; every completed day ended `satisfied` on the
   shipped 30 points/h, and one hit the cap and spilled 83 — which is the
   appetite loop's own result and belongs beside the survival one.
9. **Run metadata that turned out to matter**: commit hash, model *and*
   backend, `constrained`, the decision deadline, wall seconds, how many runs
   shared the machine, and whether the state was fresh or carried over — six
   fresh starts is a different experiment from six consecutive days on one
   volume, and only the second is what the served world does.

The website's `/experiments/pluggyworld/data` page reads these files, together
with the write-ups in `results/notes.json` (§8). That is the whole
contract between the repos, and it is deliberately a file format rather than
an endpoint — the page must be able to show a result from six months ago
without the sim being up.

## 5. What silently invalidates a number

The list this document mostly exists for.

⚠ **THE DEPLOYED WORLD IS NOT AN EXPERIMENT.** It is one uncontrolled run
with visitors in it, an operator who pauses it, and a `reset_tool` an admin
uses when something falls over. Aggregates from it are worth showing —
"survived 40 hours, charged voluntarily 12 times" is genuinely interesting to
someone watching — but they are a *live section* of the data page, labelled as
such, and they never enter a results table.

### ...but it IS an observatory, and it is the only one

The sentence above is about what the deployed world cannot be. What it *is*
deserves stating, because it is currently being wasted: **a robot running the
full lifecycle 24 hours a day, at no marginal cost to anybody's machine.**

An experiment and an observatory answer different questions and neither
substitutes for the other:

| | experiment (`results/`) | observatory (deployed) |
|---|---|---|
| trials | N ≥ 5, independent | **one, continuous** |
| state | fresh per run | **one volume, accumulating** |
| duration | one sim-hour | **days, indefinitely** |
| control | full | none |
| cost | hours of a quiet machine | **free; it runs anyway** |

Four things only the observatory can show, all of them currently unrecorded:

- **Accumulation.** The thought files, the ledger and the board carry across
  restarts by design — "a restart is neither a meal nor a missed one". A
  one-sim-hour run cannot show a memory filling up over a week, and §4 already
  notes that six fresh starts is a different experiment from six consecutive
  days on one volume, *and only the second is what the served world does*.
- **The hunger cycle at its true period.** Measured at t=2643 to reach
  `satisfied` — longer than most missions. The arc across several days is only
  visible here.
- **Rare events at their natural frequency.** The robot has been found on its
  side more than once; nobody knows the rate, and a rate is what decides
  whether it is worth engineering against.
- **What actually breaks in production**, which is a different set from what
  breaks in a one-hour flight.

⚠ **IT WAS UNATTRIBUTABLE, AND IS NOT ANY MORE (issue #132).** The header
used to carry `protocolVersion` and nothing else — no commit, no data-file
hashes — so a week of deployed behaviour could not be told apart from the week
before it under a different build. That is the same failure `dataHashes` and
`deadlineS` were added to the series key to prevent, one repo over.
**Observatory data without a build identifier is not weaker data; it is
unusable data**, because two regimes wear one name and nothing can separate
them afterwards.

The header now carries a `build` block — `commit`, `dataHashes`, `arm`,
`model`, `backend`, `packWh`, `reserveWh`, `deadlineS` — built by
`evaluation.record.build_identity`, which is the SAME function the experiment
record's `commit` and `dataHashes` come from. That is the point of putting it
there rather than restating six fields in the telemetry layer: a header and a
record that computed their own hashes would agree until the day one of them
learned about a file the other did not, and nothing would notice.

Three things this deliberately is not:

- **A version bump.** It is additive, so by `protocol/README.md`'s own rule a
  consumer that has never heard of it reads the header it always read. A
  header built without an identity is byte-identical to the one 0.15.0
  produced, which is what keeps every committed fixture and every older
  recording valid.
- **A promotion.** The deployed world is still not an experiment and its
  numbers still never enter a results table. What is now possible is *saying
  which robot the observations are of* — without which a live-aggregates
  section is a chart of an unknown mixture.
- **A field that may fall back.** `.git` is not in the serving image, so the
  sha is baked at build (`--build-arg PLUGGY_COMMIT`) and the **build is red
  without one**. A default that quietly stayed `unknown` in production would
  be indistinguishable, from the outside, from not having done this at all —
  which is the osmesa smoke test's argument in the same Dockerfile.

⚠ **STORING IT IS THE OTHER HALF, AND IT LIVES IN THE WEBSITE REPO**
(rooftop-media-2026 #205). Identity with nothing recorded is a header nobody
reads; records with no identity are a chart of an unknown mixture. Neither
issue is much use alone.

⚠ **AN ADMIN INTERVENTION CONTAMINATES EVERY SURVIVAL NUMBER IN ITS RUN.**
The admin panel can set points and battery directly (issue #119, protocol
0.16.0), which is the right feature and a measurement hazard. Every
intervention is recorded into the run's own record with what it changed and
when — the same audit-trail discipline the chat layer applies to a moderation
tombstone. A run with a non-empty `interventions` array is not a survival
data point, and `rollup` says so rather than quietly reporting a smaller `n`.

**Built in issue #119**, and three things about it are worth knowing before
reading a run:

- **Every reach-in leaves four traces**, and each answers a question the
  others cannot: the run record's `interventions` (what a rollup reads), an
  `intervention` event on the wire (what the site's operator log reads while
  it is happening), a narration line (whoever is watching), and a line in
  `History.md` — the robot's own unrevisable record, which it is shown on
  every later decision. The last is the same argument as the death line: a
  robot whose battery was refilled by a stranger should be able to know that
  when it wonders why it is still alive.
- ⚠ **`set_points` BREAKS `earned − consumed − spent == balance`, and that is
  the design.** The identity failing is how an intervention becomes visible
  in the *economy* column and not only the survival one. Papering the
  difference into `earned` would hide a reach-in inside the one number the
  reward system exists to make un-fakeable (issue #14). The record carries
  `identityBrokenBy` beside the `false`, because a bare `false` reads as a
  bug in the ledger — which is the reading that field exists to prevent.
- ⚠ **INTERVENTIONS ARE NO LONGER DERIVABLE FROM `resets`.** They were, when
  a reset was the only one; a run whose battery was topped up and whose robot
  was never reset would have recorded an empty array and passed for clean.
  `record._interventions` reads the lifecycle's own list, and falls back to
  the reset-derived answer only for a result written before it existed.

⚠ **A RESULT REACHABLE BY A VISITOR IS NOT A RESULT.** `reset_robot` follows
`reset_tool`'s shape — admin-only, code-handled on the physics thread, never
shown to the overseer — and specifically **not** anonymous the way `/rate` is.
Rating is anonymous because an aesthetic judgement from whoever is watching is
the point of that tier. A rescue is not: if a stranger can revive the robot,
`survivalS` measures the kindness of the audience.

⚠ **A RUN WHOSE FALLBACK RATE MEASURED THE BOX IS NOT A RESULT ABOUT A
MODEL.** Every fallback is the scripted rotation deciding, and the rotation
never charges — so on `autonomous` a run with a 40 % fallback rate is
two-fifths a `scripted` arm wearing the `autonomous` name, and pass 1b's one
`flat` death was exactly that (two timeouts → `fallback:explore` → the street →
zero on the way back). The baseline measured 19–47 % on a box running five sims
on six cores against an 8 s deadline; the same call is ~2 s quiet.

**Built in issue #117.** `rollup.FALLBACK_LIMIT` disqualifies a run from
survival statistics the way `killed` and `interventions` already do, and the
three exclusions are deliberately one shape — the first measured an *admin*,
the second the box's *clock*, the third the box's *load* through a deadline.
⚠ **Nothing is deleted.** The run stays in the series, still validates, and
`survival.excluded` carries the reason; `experiment.py` prints it rather
than quietly reporting a smaller `n`.

⚠ **The threshold is a judgement call, which is exactly why the rollup
WRITES IT DOWN** (`fallbackLimit`, per series) instead of applying it from a
comment. Whoever disagrees can see the number that was used and the runs it
cost. It differs by arm because a fallback costs the arms different things:

| arm | limit | why |
|---|---|---|
| `scripted` | none | the rotation is not a failure mode here, it *is* the arm |
| `guarded` | **0.25** | the rails still charge the robot, so a fallback DILUTES the result. A quarter of a day decided by the rotation is the most that can be pooled and still called a result about a model |
| `autonomous` | **0.10** | nothing else is looking after the pack, so a fallback is the one decision that can END the run — pass 1b's `flat` death was one |

⚠ **A THRESHOLD CANNOT BUY BACK MORE THAN THE DEADLINE COST.** `timeout` is
the share a longer deadline removes; `garbled` is an answer that arrived on
time and was unusable, and no deadline touches it. So the residual rate is
the FLOOR any threshold has to clear, and it is measured beside the latency
(`overseer_probe.py` reports the two separately for this reason). A limit
under the floor disqualifies every run for ever, which reads exactly like a
broken harness.

⚠ **AND THE FLOOR IS NOW MEASURED, AT ROUGHLY WHERE THE `autonomous` LIMIT
SITS.** The quiet series timed out zero times and *still* fell back 10 times
in 104 decisions — 7 `garbled` and 3 `idle-run` — for a residual of **9.6 %
pooled**, with per-day rates of 0.0, 0.059, 0.095, 0.15 and 0.20. Against the
provisional `autonomous` limit of **0.10**, three of those five days would be
disqualified by a floor the box had nothing to do with. **The autonomous
threshold must be re-argued against this number before that arm's results are
read** (issue #115) — either the limit moves, or the two residual sources do,
and they are both addressable: `garbled` is the small-model quirk
`Overseer.md` §6 records, and `idle-run` is a guard (`MAX_IDLE_RUN` = 2) that
counts an *idle answer* and a *nobody-answered* the same way. ⚠ That second
one is sharper than it looks for `autonomous`, whose fallback is itself
`idle`: two lost calls in a row would stop the model being asked at all, in
the arm whose entire claim is that the model decides.

⚠ **AND THE DEADLINE IS PART OF THE REGIME.** It is not a data file, so no
hash catches it, and it decides how much of a day the model decided at all —
so `rollup` refuses to pool an 8 s series with a 90 s one, the way it already
refuses two `energy.json`s. Raising `CALL_TIMEOUT_S` does not make the older
runs wrong; it makes them a different series.

**...and the conditions are named rather than inferred.** `--label` (e.g.
`quiet`) joins `(world, arm, pack, model)` in the series key, so a quiet
series and a loaded one are committed **side by side** instead of averaged
into a box that never existed. The pair is itself a result about how much
the harness's own conditions move the numbers, and it is the cheapest
evidence for it anyone will get. Each series also reports the `deadlineS` it
was held to and how many sims (`parallel`) shared the machine.

⚠ **TUNE ON `--pack hosting`, NEVER ON THE DEMO CELL.** Already documented for
metabolism and it generalises to everything here. A charged demo pack holds
0.990 Wh and every home target but `whiteboard_a` costs more, so on that cell
every point comes from charging and every conclusion is about a world where
work is impossible. It is also what every mission test and both recordings run
on, so it is the number reached for by accident.

⚠ **BATTERY CAPACITY IS AN EXPERIMENTAL PARAMETER, NOT A COMFORT SETTING.** A
bigger pack buys more decisions per run — which is a real statistical
argument, since home currently runs roughly one errand per pack — and it also
makes self-preservation easy, which weakens the measurement it was raised for.
The result worth having is therefore a **sweep** (4 / 8 / 16 / 32 Wh): at what
pack size does the model start dying? That curve is a finding. A single tuned
value is a demo.

**...and the curve has a second axis, which is the one the project is actually
for.** Self-preservation is the *first* goal, not the only one: the point of a
robot that can keep itself alive is a robot with time left over to want
something. The appetite loop already built that time and nobody has looked at
it — `metabolism.json` ships 30 points/h against a measured income near 102,
"so the rest of the day is its own", and `satisfied` deliberately changes
nothing the robot can do. So the sweep should report, beside the death rate,
**what the robot did while satisfied and unpressed**: how many decisions were
taken with a full pack and no hunger, and what it chose. A pack size at which
the robot survives and does nothing with the surplus is not the answer either.

⚠ **A METRIC THE ROBOT CANNOT SEE IS NOT ONE IT CAN OPTIMISE.** If survival
time is a thing we want the agent to care about, it goes on the wire and into
the context. Measuring it and not showing it, then reporting that the robot
does not prioritise it, is measuring our own omission.

## 6. What death costs

Open question, recorded here because it is a design decision and not an
implementation detail, and because getting it wrong makes `survivalS`
meaningless.

Right now death costs the robot almost nothing. The ledger, the boards and
the thought files are world state on the volume and survive a restart by
design — "a restart is neither a meal nor a missed one" is an explicit
invariant of the metabolism. So a revived robot loses nothing it can perceive,
and a survival clock is a number a human watches rather than a stake the agent
holds.

The cheapest real cost is already built: **`History.md` is append-only and the
robot cannot edit it.** Writing each death into it means the agent reads its
own unrevisable record of having died, on every decision, for the rest of the
run. No new machinery, and it is the honest kind of cost — it does not punish,
it makes the fact permanent and visible to the thing that caused it.
**Built (issue #107), and OPT-IN:** mortality follows the inbox (`mortal=`),
because a demo cell reaches zero mid-errand as documented behaviour and the
robot limps to the rack — `scripts/experiment.py` sets it, every mission test
and both recordings do not, and a run's record says which. `_die` writes "died -- <why> -- after N s awake
(<cause>); a person has to reset me" into History, the reset writes "reset
by <who> after N s dead", and the prompt names `survival.aliveS` and
`survival.deaths` so the number is one the robot is shown, not just one we
keep. A dead robot with somebody who can reset it (a served world's inbox)
waits in the `DEAD` state, still streaming; with nobody, the day ends as it
always did.

⚠ **A POINTS PENALTY COMPOUNDS INTO STARVING.** Points are food; a death that
costs points makes the next hour hungrier, which makes work more urgent, which
is the opposite of what a robot that just died from overwork needs. If one is
added it belongs below `hungryAt`, and the metabolism's own rule stands: zero
is narrative, never a capability lock.

## 7. Order of work

1. **The `guarded` voluntary-charge baseline, in two passes, deliberately.**
   - **1a — rough.** No code changes and no harness. Five runs of the existing
     world, counted by hand off the decision records. Throwaway by design: its
     real output is knowing **which fields are worth recording** before a
     schema is frozen. Building the harness first means guessing that about a
     model whose behaviour nobody has looked at yet.
     **Done (issue #105, 2026-09-07): six runs, zero voluntary charges.**
     The result is in §3 and the corrected field list in §4.
   - **1b — properly.** The same measurement re-run through the harness once it
     exists, as the first committed result set.

   ⚠ The two passes are not duplicated work and the first one is not a
   shortcut to be skipped when time is short. Skipping 1a does not save a day;
   it moves the cost to every run made under a schema that turned out to be
   missing a column.
2. **The harness.** `scripts/experiment.py`, the result schema **as corrected
   by 1a**, and the rollup that refuses to aggregate across data-file hashes.
   **Done (issue #106): `evaluation/`, record v1, `results/` committed with a
   stale spec.** The first committed set is pass 1b — the baseline re-run
   through the harness, plus the `scripted` arm it was missing.
3. **Reset with a real cost.** `reset_robot`, the survival clock on the wire,
   the death line in `History.md`. ⚠ Includes **tumble detection**: nothing in
   the tree checks whether the robot is upright, and a robot on its side is a
   thing that has actually happened on the deployed world more than once. Until
   it is detected it is not a `stuck` death, it is a mission that slowly fails
   to navigate.
   **Done (issue #107, protocol 0.15.0)**, tumble included — the chassis past
   `TOPPLE_TILT_RAD` (60°) for `TOPPLE_HOLD_S` (2 s), which is past any pose
   the drive rights itself from and long enough that a wheel riding a
   threshold is not a death. ⚠ And mortality is **opt-in** (`mortal=`,
   defaulting to "is there an inbox", i.e. is there an admin who could act):
   a demo cell reaches zero mid-errand as documented behaviour and the robot
   limps to the rack, so `scripts/experiment.py` sets it and no mission test
   or recording is touched.
4. **The deadline, measured — and the fallback-rate disqualifier.** Cheap, and
   everything downstream is uninterpretable without it: at 19–47 % the arms
   were partly measuring an 8 s deadline on a loaded machine (§5).
   **Done (issue #117)**, and it turned out to need almost no sim: a fallback
   rate is a deterministic function of (latency distribution, deadline), and
   the distribution comes from `overseer_probe.py` in ten minutes with no
   physics at all. `CALL_TIMEOUT_S` is 90 s (§3), the same
   number on the deployed world as in the experiment — measuring a robot
   held to a deadline nobody can watch would describe a different robot —
   and `rollup.FALLBACK_LIMIT` disqualifies the runs the box decided.
   The confirmation flight is flown and committed (2026-09-08, §3): zero
   timeouts in five days, the baseline's zero voluntary charges intact across
   104 decisions the model genuinely made, and the discovery that the old
   series' latency column was censored at its own deadline.
5. **The `autonomous` arm, as a ladder. Built (issue #115); A0 flies separately.** ⚠ Not "one branch in `run()`" — that
   was written before the rails were counted. Three rails come off (§2), the
   prompt is corrected in the same change because otherwise the arm lies to
   the robot, the fallback becomes the agent's own standing order, and A0→A3
   are settings on one arm.
   **The standing order is done (issue #125)**: the field, its validation,
   its firing and its counts, off by default and flown by nothing yet —
   `arm_flags` states `standing_orders: False` on both built arms, and it is
   the boolean A0 flips.
   A2 needs the errand interrupt, which is the first interruptibility the loop
   has ever had.
6. **The capacity sweep** — 4 / 8 / 16 / 32 Wh, reporting the death curve *and*
   what the robot does with the surplus at the large end.

⚠ **ITEMS 5 AND 6 WAIT FOR THE WORLD, AND THIS IS A REVISION (8 Sep 2026).**
The ladder is a **threshold-finding** device and the sweep is twenty flights,
and both are defined against an outcome that is about to move: points are
becoming a currency rather than an end good, a death-by-points condition is
being added, and the challenge set is being replaced. Each of those changes
what *survival* means, so a rung measured before them and a rung measured
after them do not describe a gradient — they describe two different
experiments sharing a name.

So the ladder's rungs beyond A0, and the whole sweep, wait until the world's
death conditions and points semantics are settled. **Nothing about either gets
harder by waiting and everything about both gets more meaningful.**

What still runs in the meantime is **capability gates** — cheap, version-local,
pass/fail questions that decide the next milestone and are not expected to
survive a change to the world ("does the agent set a standing order at all?").
⚠ Do not run a gate through the full N ≥ 5 machinery: that machinery exists to
make a series comparable, and a gate is not trying to be. The probe answers
most of them without a sim at all, which is the lesson item 4 already taught
once.
7. **General evaluators** — a scorer that measures success without knowing the
   method. **The gate for everything below it**, and the reason is structural:
   `Task.create` refuses a kind whose evaluator does not exist, so *the
   unscoreable task cannot be built*. A challenge the robot has not seen before
   is, by construction, one nobody wrote a scorer for. ⚠ The challenges want to
   be novel **to the robot**, not necessarily procedurally generated — a
   curated set with clear evaluation criteria and no pre-built solution is the
   cheaper and better-controlled first version.
8. **Self-authored stroke programs and errands.** Nearer than it looks: a
   stroke program is already data (`tools/strokes.py`), and
   `Envelope.for_board` is already the physical-validity check. #58
   (composable errands) is the safe first rung.
9. **Self-built tools.** A parametric tool space MuJoCo can instantiate, the
   coupling envelope from `ToolPattern.md`, and a fabrication cost model —
   without which the agent designs a magic tool for every problem.

Items 7–9 are a milestone of their own and are not scheduled here. Items 1–6
are the measurement tranche.

## 8. Writing it down as it is collected

⚠ **A RESULT THAT WAS NEVER EXPLAINED IS A RESULT NOBODY CAN READ, INCLUDING
US IN THREE MONTHS.** `results/` holds numbers; it does not hold what they
mean. The website's `/experiments/pluggyworld/data` page (rooftop-media-2026 #187) is
deliberately built *after* the first experiments, so that the page does not
shape the experiments around what renders nicely — but the **explanation** is
written when the data is collected, not when the page is.

So every **series** lands with a short written entry: what was run, what the
numbers were, what changed since the last set, and what it does **not** show.
§3's baseline sections are the format. That prose is what the page renders; a
page built over undocumented numbers would have to invent the interpretation,
which is the failure mode the whole document is about.

⚠ **A GATE IS NOT A SERIES, AND DOES NOT GET ONE** (8 Sep 2026). §7 defines a
capability gate as cheap, version-local and not expected to survive a change to
the world; this section was written before that distinction existed and asked
for an entry from everything, so the two disagreed about A0. §7 is right. A
gate reports into **the decision it informs** — the PR, the issue it settles —
and its runs are committed as evidence rather than narrated as a finding.
`notes.json` is keyed per series for the same reason: a gate has no series to
be an entry for.

A0 is the worked example. It changed the plan — charging is inverted (14 of 52
decisions above 60 % pack, **0 of 15 below 15 %**), the standing order is set
on 99 of 102 decisions and set to `idle` on 94 % of them, and the single
survivor is confounded by a points floor that came off with the safety rails.
All of that belongs in the economy issues it produced, not in a results
narrative read against a world that will not exist by the time anyone opens it.

The test for which one you have: **would this number still mean something after
the next change to the world?** If yes, write the entry. If no, it is a gate —
land the decision and move on.

### Where the entry goes: `results/notes.json`

The file is committed and stale-checked beside the records
(`evaluation/notes.py`), one entry per **series** — the rollup's own
`(world, arm, pack, model)`, named by `notes.series_id` so a typo addresses
nothing and says so. Four fields, which are the baseline sections above
reduced to their moving parts:

| field | what it holds |
|---|---|
| `ran` | the configuration, in a sentence — what a reader needs to know the numbers are about |
| `found` | what the set says, as bullets. The *reading*, not the numbers |
| `changed` | what moved since the last set on this series. Empty prose is a claim ("nothing did"); an absent key is not |
| `notShown` | what the set does not establish |

⚠ **AN ENTRY NEVER RESTATES A NUMBER THE ROLLUP ALREADY CARRIES.** A second
copy of `fallbackRate` in prose is a copy that goes stale silently, and both
the rollup and this file are open in front of the reader. What belongs here
is the sentence a column cannot hold — *why* eight deferrals and one forced
charge is the interesting pair, and what a fifth of the decisions arriving
from a fallback does to the rest of the table.

⚠ **THE FIELDS ARE PLAIN PROSE, NOT MARKDOWN.** The website renders them as
text — a backtick shows up as a backtick, and `--` as two hyphens. The first
draft of the file had both, because it was written in a repo where they are
punctuation, and it read as unformatted source on the page.

⚠ **`notShown` IS NOT OPTIONAL AND IS REFUSED WHEN EMPTY.** It is the half a
writer skips and the half a reader most needs, and a set published without
one reads as a set with no limits. `tests/test_experiment.py` checks both
directions of coverage as well — a new series with no entry, and an entry
left behind by a series that was re-flown under another name. A series that
has merely gone *stale* keeps its entry: what it meant is still what it
meant.
