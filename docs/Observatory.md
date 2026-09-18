# The observatory's periods — what was deployed when, and what each period is for

The deployed world is one uncontrolled run, 24 hours a day (Evaluation.md
§5). A reading off it is only as good as the record of WHAT was running
while the rows were written, so this file is that record: one entry per
period, opened by the PR that changes the deployed design and closed by
the reading that ends it. The readings themselves (#223) and the decisions
they lead to (#224, #225) are entries here too. A reading of the
observatory is NOT a result and never enters `results/`.

## Periods

### Memory phase 1 (#221) — opens when this PR is deployed

**What changed on the wire and in the mind.** The robot's memory became
four tiers over one record store (Overseer.md §7): `Top_of_mind.md`
(renamed from `Knowledge_and_Opinions.md`, `pin`/`unpin`), `Notes.md`
(`note`/`unnote`, index shown, body by `recall`), `Findings.md` as typed
topics, History kept whole in the store and tailed with ids; `think` first
in every answer; `recall` as an action with a chain of at most three; the
`journal` action retired. Protocol 0.21.0. The deployed volume started
BLANK: the pre-#221 files were archived beside the new store and nothing
was imported, so every row in this period was written through the new
verbs. Same model (`Qwen/Qwen3-4B-Instruct-2507`), same arm
(`autonomous`, origin `unseeded`), same pair.

**What the period is for.** Phase 2 (#222) refines the memory against
what this period shows, AFTER the model upgrade (#225) has had its own
period. What to read, in the observatory's own kinds:

- `thought` rows by verb: how much it pins, notes, and how often a write
  is `refused` (a full document, a quote that matched nothing).
- `recall` rows, `found` against `empty`: whether it looks things up at
  all, and whether it looks for what it never wrote down. `data.run` says
  how long its chains are; `data.read` against `data.find` which form it
  uses.
- A `recall` followed by a `pin` or a `note` in the same run (`cites`
  on the decision row): memory that was USED, the paper's "written but
  never read" diagnostic inverted.
- `journal` rows (the thinks): whether the scratch is reasoning or filler,
  and whether it fills `THINK_CHARS` every turn.
- Deaths whose History line named an earlier death: did remembering
  change the next decision at the same battery fraction?
- Tokens per decision against the measurement in Overseer.md §7.

**Not yet known.** The death rate on this memory; whether a 4B uses
`recall` unprompted; whether the notes index at 64 titles is ever reached.
