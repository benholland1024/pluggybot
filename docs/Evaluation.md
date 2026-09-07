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
  at 4.2 s alone here and ~2 s on a quiet box. The rest were `garbled` (one was
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

**What pass 1a found the record needs** (issue #105; the measured section is
in §3). Counted by hand off six days of decision records, and every item is
a column the provisional schema either lacks or would have got wrong:

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
