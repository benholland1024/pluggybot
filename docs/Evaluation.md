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

⚠ **Anything that makes the world random destroys this**, and the temptation
will come dressed as realism ("jitter the task times so it feels alive").
Variation belongs in the ARM, held fixed within a run and varied between them.
If a world ever needs randomness, it takes an explicit seed that goes in the
result record.

## 2. The arms

An **arm** is one configuration under test. Three exist; each answers a
different question, and none of them is redundant.

| Arm | Charge rail | Mind | Answers |
|---|---|---|---|
| `scripted` | on | rotation, no LLM | The null model. What does the world do with no mind at all? |
| `guarded` | on | LLM | Today's behaviour. Does the model manage energy *when it does not have to*? |
| `autonomous` | off | LLM | Does the model manage energy when nothing else will? |

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
  reset. NEW: needs the reset event of issue "reset a dead or stuck robot".
- `deaths` — resets, **split by cause and never summed into one number**:
  - `flat` — the pack reached zero. This is a decision failure.
  - `stuck` — knocked over, wedged, or unable to reach the rack. This is a
    physics or navigation failure.

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
  `energy.json`, `metabolism.json` and `questions.json`.** These five files
  each change the regime, and a series that spans an edit to any of them is
  two series wearing one name. The rollup refuses to aggregate across
  differing hashes rather than averaging them.
- **A run that hit an admin intervention is marked, and excluded from
  survival statistics by default** (§5).
- Results are **committed**, and versioned exactly as `protocol/` fixtures
  are: generated, checked in, with a spec that fails when they go stale. They
  are the research artifact; a number that exists only in a terminal
  scrollback did not happen.

⚠ **THE SCHEMA BELOW IS PROVISIONAL AND WAS WRITTEN WITH NO DATA BEHIND IT.**
It is a guess at what a run is worth recording, made before anyone had looked
at a single decision record — which is the same error as building the data page
before there are results, one layer down. **Run the rough baseline (§7.1a)
first and let it correct these fields**, then freeze v1. A schema frozen ahead
of the data is one every later run is stuck with, and the cost of getting it
wrong is paid in re-runs, not in an edit.

Result record, provisional:

```json
{
  "schema": 1,
  "runId": "2026-09-06T12-00-00Z_home_autonomous_hosting_qwen3-4b_s0",
  "world": "home", "arm": "autonomous", "pack": "hosting",
  "model": "Qwen/Qwen3-4B-Instruct-2507", "seed": 0,
  "simSeconds": 3600.0, "wallSeconds": 512.3,
  "dataHashes": { "rewards": "…", "cadence": "…", "energy": "…",
                  "metabolism": "…", "questions": "…" },
  "survival": { "survivalS": [3600.0], "deaths": { "flat": 0, "stuck": 0 } },
  "charging": { "forced": 3, "voluntary": 1,
                "voluntaryFrac": [0.61], "anticipation": 1 },
  "mind": { "llmCalls": 48, "fallbacks": 2, "escalations": 1,
            "constrained": true },
  "memory": { "learn": 6, "forget": 1 },
  "economy": { "earned": 180, "consumed": 148, "spilled": 0, "balance": 32,
               "tasks": { "offered": 9, "claimed": 5, "done": 3,
                          "failed": 1, "expired": 2 } },
  "interventions": []
}
```

The website's `/pluggyworld/data` page reads these files. That is the whole
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

⚠ **AN ADMIN INTERVENTION CONTAMINATES EVERY SURVIVAL NUMBER IN ITS RUN.**
The admin panel will be able to set points and battery directly, which is the
right feature and a measurement hazard. Every intervention is recorded into
the run's own record with what it changed and when — the same audit-trail
discipline the chat layer already applies to a moderation tombstone. A run
with a non-empty `interventions` array is not a survival data point.

⚠ **A RESULT REACHABLE BY A VISITOR IS NOT A RESULT.** `reset_robot` follows
`reset_tool`'s shape — admin-only, code-handled on the physics thread, never
shown to the overseer — and specifically **not** anonymous the way `/rate` is.
Rating is anonymous because an aesthetic judgement from whoever is watching is
the point of that tier. A rescue is not: if a stranger can revive the robot,
`survivalS` measures the kindness of the audience.

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
The result worth having is therefore a **sweep**: at what pack size does the
model start dying? That curve is a finding. A single tuned value is a demo.

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
   - **1b — properly.** The same measurement re-run through the harness once it
     exists, as the first committed result set.

   ⚠ The two passes are not duplicated work and the first one is not a
   shortcut to be skipped when time is short. Skipping 1a does not save a day;
   it moves the cost to every run made under a schema that turned out to be
   missing a column.
2. **The harness.** `scripts/experiment.py`, the result schema **as corrected
   by 1a**, and the rollup that refuses to aggregate across data-file hashes.
3. **Reset with a real cost.** `reset_robot`, the survival clock on the wire,
   the death line in `History.md`.
4. **The `autonomous` arm.** One branch in `run()`, not a refactor —
   `needs_charge` fires on absolute reserve and already never consults
   `TOP_UP_BELOW`, so the two policies are cleanly separated today.
5. **The capacity sweep.**
6. **General evaluators** — a scorer that measures success without knowing the
   method. The gate for everything below it.
7. **Self-authored stroke programs and errands.** Nearer than it looks: a
   stroke program is already data (`tools/strokes.py`), and
   `Envelope.for_board` is already the physical-validity check.
8. **Self-built tools.** A parametric tool space MuJoCo can instantiate, the
   coupling envelope from `ToolPattern.md`, and a fabrication cost model —
   without which the agent designs a magic tool for every problem.

Items 6–8 are a milestone of their own and are not scheduled here. Items 1–3
are the tranche that turns everything above from world-building into
measurement, and item 1 can be run this week.
