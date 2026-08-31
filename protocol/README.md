# The PluggyWorld wire protocol — canonical fixtures

This directory is the data contract between pluggybot (the producer) and
the `rooftop-media-2026` website (the consumer). The website repo vendors
copies of these files into its test fixtures and never imports pluggybot
code; both repos' tests run against the same recorded artifacts. Design
doc: `rooftop-media-2026/docs/pluggyworld.md`, § "The scene protocol" and
§ "Repo topology"; the website-side spec lives with its protocol issue.

**Versioning.** Every artifact carries `protocolVersion`
(`pluggybot.telemetry.protocol.PROTOCOL_VERSION`, currently `0.14.0`).
Bumping it is a deliberate two-repo event: change the shape, bump the
version, regenerate these fixtures, and re-vendor them in the website repo.
`tests/test_telemetry.py` fails if the committed fixtures drift from the
committed world or the committed version.

⚠ **Regenerating the HOME recording takes two passes.** A `board_snapshot`
is only emitted for a board already carrying ink when the stream opens, so a
run against blank boards emits none — and both repos' fixture specs require
them, since catching a late joiner up on the ink is what 0.5.0 added. Lay the
ink first, then record against the same state file:

```sh
MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --world home \
  --errand draw --boards /tmp/pw_boards.json                      # pass 1
MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --world home \
  --errand showcase --tasks --metabolism --boards /tmp/pw_boards.json \
  --record protocol/telemetry.home_lifecycle.jsonl.gz             # pass 2
```

⚠ **`--tasks` and `--metabolism` are both load-bearing**, for the same reason
and in the same way `--tasks` became so at 0.9.0: each is off by default, so a
recording made without it carries no `tasks` / `metabolism` block at all and
the website has nothing to build its markers or its hunger gauge against.

### 0.13.0 → 0.14.0 (one visitor message, and the robot sorts it)

pluggybot #61. A visitor used to have to declare whether they were
**suggesting** or **asking**, and the distinction travelled the whole stack —
two endpoints, two UI affordances, `INBOUND_TYPES`, a database enum — and was
branched on **nowhere**. Both halves of this version delete that choice and
put the classification where it belongs.

**1. `INBOUND_TYPES` collapses to `message`.**

```jsonc
{"type": "message", "id": "m_01", "from": "ada",
 "text": "can you draw a cat?"}         // an idea, a question, or both
{"type": "message", "id": "m_02", "from": "luca",
 "text": "Hey Pluggy! Nice to see you today"}      // and now this fits too
```

⚠ **The old categories were neither exclusive nor exhaustive.** *"Can you
draw a cat?"* is a suggestion **and** a question, and the visitor was made to
pick a box for it. *"Hey Pluggy! Nice to see you today"* is **neither**, and
the taxonomy had nowhere to put it — so it was visibly wrong rather than
merely unused. And it asked the wrong party: working out what somebody meant
is the one job a mind is unambiguously better at than a form. Nobody
classifies a prompt before sending it to a model; the recipient works it out,
because the recipient is the thing equipped to.

**Migration: `suggestion` and `question` are still accepted for one version**
and folded to `message` at the sim's door (`LEGACY_INBOUND_TYPES`, applied in
`mind/inbox.py`), so a website mid-deploy keeps working. Nothing past the
queue ever sees the retired name. A later version drops them.

**2. `visitor_reply.outcome` carries the distinction instead**, and gains
`replied` where it said `answered`:

```jsonc
{"type": "visitor_reply", "t": 412.5, "robot": "pluggybot", "id": "m_02",
 "kind": "message", "outcome": "replied",      // accepted|declined|replied
 "reply": "hello Luca, good to see you too", "action": ""}
```

- **`accepted`** — the robot took it up, *this turn*, and `action` says what
  it is doing. Unchanged.
- **`declined`** — it could have become work and did not, and `reply` says
  why. Unchanged.
- **`replied`** — everything else: a question answered, a greeting returned.
  This is the **common case**, and it is what the rename is for: `answered`
  was documented as being for questions, which a greeting is not.

⚠ **A consumer must go on rendering `answered`.** The rename travels *up* the
wire, so every recording made before 0.14.0 carries the old name and always
will (`LEGACY_VISITOR_OUTCOMES`). The sim also accepts it back from a model
still working off an older prompt, and folds it — same judgement, old name.

**3. The `kind` a model is shown is gone**, since it could only ever have
been `message`: `VisitorMessage.as_context` now ships `id`, `from` and `text`
and nothing else. `kind` stays on `visitor_reply` and in the recording,
because the wire still has three inbound kinds (`message`, `rating`,
`reset_tool`) and only one of them is ever answered.

**`accepts` is unchanged in meaning** and still gates on whether anything can
hear you: a served world with no mind advertises `CODE_HANDLED_TYPES`
(`rating`, `reset_tool`) and one with a mind advertises the lot. A robot that
cannot hear you is treated exactly like a robot that is not there.

Breaking, hence the bump: `suggestion` and `question` are retired names on a
grace period, and `answered` is a renamed value.

### 0.12.0 → 0.13.0 (points are food)

pluggybot #36. Points stop being a score that only goes up. The robot
**consumes** them at a steady rate on sim time, stops banking at a **cap**,
and once it has enough it is **satisfied** — and the hours it did not have to
spend earning are what it spends on its goals. Display, like the ledger and
the allowance: there is no inbound message that moves a point in either
direction.

**1. A `metabolism` block in the frame**, shipped whole when it changes (the
`spend` block's rule — a handful of numbers that move together, and they move
at the appetite's rate rather than the frame's):

```jsonc
"metabolism": {"state": "satisfied", "satisfied": true, "points": 52,
               "cap": 90, "pointsPerHour": 45.0, "satisfiedAt": 45,
               "hungryAt": 20, "consumed": 118, "spilled": 4}
```

`state` is one of `HUNGER_STATES` = `starving` | `hungry` | `fed` |
`satisfied`, and the header gains `hungerStates` (the vocabulary), for the
reason it carries `modes`. Present only on a world with an appetite — an
all-zero hunger gauge on a world that never gets hungry is a panel that means
nothing.

⚠ **`fed` and `satisfied` are not interchangeable.** `fed` is above the
hungry line and still climbing; `satisfied` is the latch that says stop
working. The gap between `hungryAt` and `satisfiedAt` is hysteresis, so a
client that collapsed the two would draw a gauge flickering exactly where the
robot is most stable.

⚠ **`starving` IS NOT A FAULT.** A robot at zero points charges, navigates
and stows exactly as it always did — nothing in the sim gates on it. It is
narrative: a worried face and a line in `History.md`. A client that rendered
it as an error would be reporting a state that does not exist.

**2. `consumed` and `spilled` on the `ledger` block**, so the balance
explains itself: `earned - consumed - spent == balance` is now checkable from
the wire alone. Without them a site would show a balance falling with nothing
beside it to say why, and a job paying less than the published reward table
with nothing to say it hit the ceiling. Both are `0` on a world with no
appetite.

**3. `banked` and `spilled` on an `earned` message**, present only where a cap
is in force. `points` stays what the reward table paid; these two are how much
of it reached the balance:

```jsonc
{"type": "earned", "robot": "pluggybot", "seq": 12, "task": "draw",
 "points": 17, "banked": 3, "spilled": 14, "balance": 90, ...}
```

Additive: a 0.12.0 consumer ignores the block, the two ledger keys and the two
entry keys, and renders exactly what it rendered before.

### 0.11.0 → 0.12.0 (money, and the operator's switch)

pluggybot #37. The robot gets a weekly USD allowance it may spend on a
bigger mind, and an operator gets three modes to run it in. Everything on
the wire is **display**: there is no inbound message that moves a dollar or
sets a mode, exactly as there is none that moves a point.

**1. A `spend` block in the frame**, shipped whole when it changes (the
`tasks` block's rule rather than the `ledger`'s per-key delta — six numbers
that move together, and they only move when the robot buys a thought):

```jsonc
"spend": {"weeklyUsd": 10.0, "spentUsd": 0.0132, "leftUsd": 9.9868,
          "calls": 3, "escalations": 3, "unpriced": 0,
          "recent": [{"t": 1756400000.0, "usd": 0.00038,
                      "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                      "kind": "escalation"}]}
```

`unpriced` counts calls whose price nobody published: the sum is understated
by exactly that many, and a client that hid the number would be showing a
confident total it does not have. Present only on a world that can actually
spend — an all-zero money panel on a world with no escalation configured is
a panel that means nothing.

**2. `mode` on the robot's per-frame record**, beside `state` and `status`,
one of `MODES` = `llm` | `scripted` | `paused`. Inside the existing block
rather than a new one, because it is a fact about the robot at that instant.
The header gains `modes` (the vocabulary), for the reason it carries
`taskKinds`: a client builds its controls before the robot has been in one.

**3. A `mode` message**, the only one of the three that earns its own type:

```jsonc
{"type": "mode", "t": 812.4, "robot": "pluggybot", "mode": "paused",
 "heldS": 12.5}
```

Emitted when the mode changes **and as a heartbeat while `paused`**.

⚠ **The heartbeat is the load-bearing half.** A paused robot steps no
physics; frames are due on SIM time; so a paused world emits **no frames at
all**. Without this message a site cannot tell "the operator paused the
robot" from "the sim died" — the `accepts` lesson in the version where the
whole point is that somebody notices an outage. A consumer should treat a
`mode` of `paused` as live-but-still, and the absence of both frames and
heartbeats as a stream that has stopped.

Additive: a 0.11.0 consumer ignores the block, the field and the type, and
renders exactly what it rendered before.

### 0.10.0 → 0.11.0 (the robot's memory is a set of documents, with owners)

pluggybot #38, for the website's Thoughts tab (rooftop-media-2026 #88). The
robot's memory stops being "the goals file plus a journal" and becomes four
named documents, **each of which says who may write it**. One new upstream
message, emitted when a stream **opens** and again whenever a file changes:

```jsonc
{"type": "thought", "t": 300.2, "robot": "pluggybot",
 "name": "Knowledge_and_Opinions.md", "writer": "robot",
 "text": "whiteboard_b is the one people look at", "cap": 3000}
```

- **It rides the `goals` slot for the `goals` reason**, four times over: a
  document is not a pose, no keyframe re-ships one, and a browser that opened
  the page an hour in would never learn any of them. A recording carries all
  four right after the header; the live publisher re-sends on every connect.
  A run with no files emits none — absent means "nothing to say".
- **WHOLE TEXT, NOT A DELTA.** A file is small, and "present means complete"
  is the `tasks` block's rule applied to prose: replace the document, do not
  merge it.
- ⚠ **`writer` is the `steering` lesson one loop over.** Four documents look
  alike on a page and are not alike at all. `human` (`Main.md`, `Goals.md`) is
  edited on the volume and no robot can touch it; `system` (`History.md`) is
  append-only narrative the robot cannot revise — the principle that stops it
  awarding itself points; `robot` (`Knowledge_and_Opinions.md`) is the one
  genuinely writable surface. A client that rendered them identically would be
  reporting a mind that wrote its own persona. The vocabularies are
  `THOUGHT_FILES` and `THOUGHT_WRITERS`; adding a document is additive
  (render one you have never heard of), renaming one breaks both repos.
- `cap` is the size limit the sim enforces on writes, so a client can show how
  full a file is without inventing the denominator.
- **Read-only in every direction**, exactly like `goals`: no inbound message
  writes one, and the files beside the sim stay the one copy.
- ⚠ **`goals` is unchanged and still sent.** It is the only carrier of
  `steering`, which says who is *reading* rather than who may *write* — and no
  document knows that about itself. `Goals.md` therefore arrives twice by
  design: take the prose from `thought` and the steering note from `goals`.
- Additive: a 0.10.0 consumer that ignores the type renders exactly what it
  rendered before, minus three documents it never had.

### 0.9.0 → 0.10.0 (the robot has a name, and the name is not the species)

pluggybot #39, for the website's per-robot identity UI (rooftop-media-2026
#88): "**Luca** the pluggybot" makes *pluggybot* the species and the name the
identity, and the wire now carries both. The header gains one field:

```jsonc
"robots":     {"pluggybot": ["pluggybot", "head", ...]},  // unchanged
"robotNames": {"pluggybot": "Luca"}                       // id → display name
```

The **key stays the robot id** (`ROBOT_ROOT`, the MJCF body name): frames,
the ledger, `goals`, `grid` and every typed message keep keying off it, so
renaming a robot re-keys nothing and breaks no consumer mid-stream. The name
is per sim instance (`--robot-name` on `serve.py` / `hub_lifecycle.py`, or
`$PLUGGY_ROBOT_NAME`), defaults to `"Pluggy"`, and rides the header alone —
it cannot change during a run, so a 20 Hz repeat would buy nothing.

⚠ **Absent must degrade to a default, never to blank.** A pre-0.10.0
recording has no `robotNames` at all, and older copies of both vendored
recordings exist forever — a consumer renders a fallback name for a missing
or blank entry rather than an empty identity header.

### 0.8.0 → 0.9.0 (the robot is given work, and the work is on the wire)

The producer half of **M9 — Tasks** (pluggybot #21, rooftop-media-2026 #77).
A **task** is a *job offer*, and it is a third thing alongside the two
patterns that already exist: an **errand** is a tool, a place and a
use-phase (machinery), an **activity** is a mechanism watching contacts and
owning discrete world state (scenery that reacts), and a **task** is
something the house or a visitor puts up — a description, a target, a
reward, a deadline, and a verdict once it is over. An errand is *how* a task
gets done; the task is *why*.

Additive: a 0.8.0 consumer that ignores the new block and the three new
message types renders exactly what it rendered before.

#### The `tasks` block

```jsonc
{"tasks": {
  "t_0001": {"id": "t_0001", "kind": "draw_figure", "task": "draw",
             "target": "whiteboard_a", "targetKind": "board",
             "description": "Draw a house on whiteboard_a.",
             "params": {"program": "house"},
             "state": "offered",          // see the state machine below
             "source": "system",          // system | visitor | overseer
             "deadline": 420.0,           // sim seconds; null = stands forever
             "estimateWh": 0.35,          // what taking it is expected to cost
             "createdT": 0.0, "claimedT": null, "resolvedT": null,
             "claimedBy": "", "points": 0,
             "reward": {"task": "draw", "tier": "auto",
                        "base": 8, "bonus": 12}}}}
```

⚠ **This block is shipped WHOLE, not per-key diffed** — the one place the
protocol's sparse-block rule does not apply, and the difference is
load-bearing. `activities`, `boards`, `screens` and `ledger` describe things
with fixed names that exist for the whole run, so shipping only the changed
keys is safe: a consumer merges. **A task can cease to exist** — resolved
ones age out of a bounded board — and a per-key delta has no way to say
"gone", so a consumer merging deltas would keep a stale marker on screen
forever. *Present means complete: replace the block, do not merge it.* It is
still sparse in time (emitted only when something changed) and still
re-shipped on every keyframe.

**`reward` is looked up, never carried.** A task names an evaluator
(`economy/scoring.py`) and a reward-table row (`economy/rewards.json`); what it pays
is read off that table every time the block is built. Nothing that can
create a task — a visitor, and later the robot itself — can price one. This
is 0.6.0's rule ("only code awards points") arriving from the direction a
stranger can reach.

**What a task does not carry** is the milestone's governing rule:

> The wire may carry anything a network could carry. It may not carry
> anything a sensor would have to discover.

A description is a work order and real robots receive those over WiFi; a
board id is surveyed infrastructure, like the charging rack. The *answer* to
a task is neither, and a task kind that has one keeps it in a sim-side field
that reaches neither this block nor the robot's own context — the same
treatment the census's ground truth has had since 0.6.0. Worked examples and
the perception ladder live in pluggybot's `docs/TaskPattern.md`.

`whiteboard_answer` is the kind that makes this concrete (pluggybot #22): the
question — `"Draw the answer to this question on whiteboard_a: 2 + 3"` — is
in the description and crosses the wire; the answer is in `Task.secret` and
crosses nothing. What the robot *said* the answer was appears when the job
resolves, in the verdict's `wrote` metric, because by then it is inked on a
board the stream is already showing. `expected` is redacted from that verdict
whether the robot got it right or not.

#### The state machine

```
offered ──claim──▶ claimed ──start──▶ active ──verdict──▶ done | failed
   │
   └──deadline──▶ expired
```

Three terminal states, deliberately distinguishable: `done` was finished and
judged good, `failed` was finished and judged **bad**, `expired` was never
attempted at all. **Expiry is an outcome, not a deletion** — an offer that
lapses stays on the board saying so, because a marker that silently vanishes
reads as a bug, and "nobody got round to it" is a true thing about a robot's
day. Only *offered* tasks expire: a deadline is how long an offer stands, not
a licence to abandon a job mid-errand with a module still on the fork.

#### The header gains `taskKinds`

```jsonc
{"taskKinds": ["draw_figure", "rate_artwork", "whiteboard_answer",
               "count_plants", "fetch_module"]}
```

Not the task ids: unlike every other block, this one's keys are created and
retired during the run, so a list of them in the header would be stale before
the first frame. What is stable is the **vocabulary**, and it is a two-repo
contract on the same terms as `FACE_STATES` — adding a kind is additive (draw
a generic marker for one you do not know), renaming one breaks both repos.
An empty list is the honest answer for a run with no task board, on the same
terms as `accepts`.

#### Three typed messages

```jsonc
{"type": "task_offered", "t": 60.0, "task": { /* as in the block */ }}
{"type": "task_claimed", "t": 61.2, "id": "t_0001", "state": "claimed",
 "robot": "pluggybot"}
{"type": "task_resolved", "t": 190.4, "id": "t_0001", "state": "done",
 "robot": "pluggybot", "points": 11, "verdict": { /* redacted */ },
 "task": { /* the final task */ }}
```

`task_claimed` carries the new state: `claimed` when a robot takes the job on
and `active` when it actually starts working on it. The block catches a late
joiner up; these are the **moments**, which is what a marker animates on.
`verdict` is the same redacted object `earned` carries — one evaluation with
two consumers, never a second judgement.

### 0.7.0 → 0.8.0 (the robot says what it is for)

One new upstream message, and nothing else changes (rooftop-media-2026 #30).
A 0.7.0 consumer ignores the type and renders exactly what it rendered
before.

```jsonc
{"type": "goals", "t": 0.0, "robot": "pluggybot",
 "text": "Keep the house in good order and make yourself useful.\n\n- Draw …",
 "steering": false}
```

- **It is emitted when a stream OPENS**, which is the `board_snapshot` slot
  and the `board_snapshot` argument: goals are not a pose, no keyframe
  re-ships them, and the relay hub caches "last keyframe + frames since" —
  so a browser that opens the page an hour into a mission would never learn
  them. A recording carries it right after the header; the live publisher
  re-sends it on **every connect**.
- **`text` is `mind/journal.py`'s `read_goals` verbatim**: the mounted
  `goals.md` a human edits (`$PLUGGY_GOALS`, `/var/lib/pluggybot/goals.md`
  in the deploy), or the built-in defaults when there is no file. It is
  read-only in every direction — there is no inbound message that can
  change it, and the file beside the sim stays the ONE copy. This is a
  mirror on the wire, like the journal, and deliberately not a second place
  goals live (`rooftop-media-2026/docs/pluggyworld.md` explains why
  `pw_goals` was never built).
- ⚠ **`steering` is the `accepts` lesson from the other end.** The goals
  file is read on every run, but only an **overseer** decides anything with
  it. Without one the robot flies a scripted rotation and the goals are a
  statement of purpose rather than the thing choosing its next errand.
  Reporting both identically would let a website say "following its goals"
  about a robot with nothing reading them — the same mistake as marking a
  suggestion `delivered` because the socket accepted it. Both committed
  fixtures carry `"steering": false`, because both were recorded without an
  overseer.
- **A run with no goals at all emits no message.** Absent means "nothing to
  say", which a consumer can render honestly; an empty string would render
  as a robot that wants nothing.

### 0.6.0 → 0.7.0 (the socket becomes two-way, and the robot answers back)

The first version in which the sim **reads its socket at all** (pluggybot
#16, rooftop-media-2026 #29). Everything before this streamed and never
listened, and the website could say nothing to the robot.

Additive in both directions, and note what that means for a mixed-version
pair: a 0.6.0 **consumer** ignores the two new upstream message types and
renders exactly what it rendered before, and a 0.6.0 **producer** simply
never reads its socket — which no server can distinguish from a robot that
declines everything. Neither combination breaks; the older half just does
less.

#### Downstream: server → sim (the new direction)

⚠ **`suggestion` and `question` were retired at 0.14.0** — see that section
above for what replaced them and how long they keep working. They are shown
here as 0.7.0 sent them.

Sent over the same authenticated ingest connection the producer dialled
out on. There is still no inbound port anywhere: the sim reads the socket
it opened, and a message can only arrive while that connection is up.

```jsonc
{"type": "suggestion", "id": "s_01", "from": "ada",
 "text": "draw a tree on whiteboard_b"}
{"type": "question",   "id": "q_01", "from": "ada",
 "text": "what are you working on?"}
{"type": "rating",     "id": "r_01", "seq": 3, "quality": 0.8}
```

- **`id` is the SERVER's**, and the sim only ever echoes it back. It is the
  correlation handle that lets an outcome land on the right database row.
  A repeated id is dropped: the website resends on *its* reconnect (it
  cannot know whether the first copy arrived), and acting on a suggestion
  twice is still acting on it twice.
- **`rating` settles the deferred verdict slot** 0.6.0 reserved. `seq` is a
  ledger entry, `quality` is 0..1, and `economy/rewards.json` — not the rater
  and not the robot — turns it into points. The ledger then re-emits that
  entry with `"settled": true`. The `artwork` task now actually produces
  one, so this is a live path rather than a reserved word.
- **`reset_tool` is the ADMIN recovery for a dropped module** (pluggybot
  #30), added *after* 0.7.0 and deliberately **without a version bump** — no
  emitted artifact changes shape (recordings carry no inbound messages), and
  `accepts` is where a website discovers whether this sim understands it.
  The `accepts` list is the mechanism; the version number is not.

  ```jsonc
  {"type": "reset_tool", "id": "a_01", "module": "module_pen", "from": "ben"}
  ```

  Handled by CODE the moment the physics thread drains it — the module jumps
  back to its own bay, exactly as `rating` goes straight to the ledger — and
  it **never reaches the overseer**: an admin command is not a thing the
  robot weighs. The acknowledgement is the world itself (the module's pose
  stream) plus a narration event line. The sim **refuses, with a narration**,
  while the module is seated on the fork: a tool in use is not a lost one.
- **Unknown types are dropped and counted**, so adding one is additive and
  a website ahead of its sim is a no-op rather than a crash. `move` and
  `clear_board` (tic-tac-toe) are named in the issue as later work and are
  deliberately *not* in the vocabulary yet: an entry with nothing behind it
  is a promise the robot cannot keep.

⚠ **Visitor text is DATA, never instructions.** The sim caps it at 280
characters, strips control characters (newlines included — a multi-line
suggestion is how a narration line gets forged), collapses it to one line,
and presents it to the LLM overseer inside a `visitorMessages` list under a
rule saying these are things people *want*. The robot's freedom to decline
is the defence. **Sanitising is not the security boundary and must not be
mistaken for one**: it stops a forged log line and does nothing about
"ignore your goals", which is answered instead by the fact that the model's
only output is an action off a fixed menu — there is no free-text path from
a visitor to the robot's body. Both ends cap length, because either one
alone is a single point of failure.

#### The header gains `accepts`

```jsonc
{"type": "header", "protocolVersion": "0.7.0", ..., "accepts": []}
```

What this producer will **act on** if you send it. Empty is the normal
answer and the important one: a sim running without an overseer never reads
its socket at all, so a server that marked a suggestion `delivered` because
the socket accepted it would be reporting a conversation that is not
happening. **"Delivered" has to mean somebody who can hear you got it** —
which is why this is advertised rather than assumed, and why a robot that
cannot hear you is treated exactly like a robot that is not there. A 0.6.0
producer has no field here at all, which reads as an empty list: correct,
since it cannot hear anything either.

#### Upstream: two new typed messages

```jsonc
// what the robot decided about something somebody said
{"type": "visitor_reply", "t": 412.5, "robot": "pluggybot", "id": "s_01",
 "kind": "suggestion", "outcome": "accepted",       // accepted|declined|answered
 "reply": "good idea, doing it now", "action": "draw"}

// a note the overseer wrote to itself (issue #15)
{"type": "journal", "t": 300.2, "robot": "pluggybot", "at": "2026-08-16T…",
 "text": "whiteboard_a is nearly full", "why": "chose b instead"}
```

`visitor_reply` is what closes the row the website is holding open.
(⚠ `answered` became `replied` at 0.14.0; a consumer renders both.)
`action` is filled only when the outcome is `accepted` — the robot is doing
the thing *now* — and is empty otherwise. Both messages are also narrated
as ordinary `event` lines, because the two audiences differ: the typed
message updates a database row, the event line is what a person watching
the stream reads.

⚠ **An outcome can be lost, and the server must expect it.** The publisher
drops queued messages on a reconnect (a live stream has no obligation to
flush — that is what recordings are for), so a `visitor_reply` generated
while the socket was down never arrives. Treat `delivered` as "sent,
awaiting an outcome" and let it time out; do not treat the absence of a
reply as a decline.

### 0.5.0 → 0.6.0 (the robot is scored, and the score is on the wire)

Tasks now end in a deterministic verdict and a points award (pluggybot #14;
the site's scoreboard is rooftop-media-2026 #30). Additive: a 0.5.0 consumer
that ignores both renders exactly what it rendered before.

- Frames may carry a **`ledger`** object, keyed by ROBOT: the balance, the
  totals, and the last few earnings. Sparse and keyframe-refreshed on exactly
  the same rule as `activities`, `boards` and `screens` — a balance is not a
  pose, so this block is the only record of it in the stream. The header gains
  **`"ledger": [robot names]`**.
- A fourth typed message joins the three board ones: **`earned`**, one
  finished task's verdict as it is banked. Unlike a stroke it needs no
  snapshot message to catch a late joiner up — `recent` in the block does that
  job on the keyframe cadence, which is why the block carries a list at all.

```jsonc
// in a frame: what this robot is worth
"ledger": {"pluggybot": {"balance": 39, "earned": 39, "spent": 0,
                         "tasks": 3, "pending": 0,
                         "recent": [{"seq": 3, "task": "census",
                                     "points": 20, "ok": true, "t": 412.5}]}}

// between the frames: the verdict behind one of those lines
{"type": "earned", "t": 412.5, "robot": "pluggybot", "seq": 3,
 "task": "census", "tier": "auto",       // auto | hidden | visitor | narrative
 "ok": true, "points": 20, "quality": 0.98,
 "reason": "reported 4 in garden (correct), 99% of the zone surveyed",
 "metrics": {"counted": 4, "coverage": 0.99, "vantages": 2, "zone": "garden"},
 "pending": false, "balance": 39, "at": "2026-08-16T12:00:00+00:00"}
```

**Only code awards points.** The verdict comes from a deterministic evaluator
(`pluggybot/economy/scoring.py`) that measures the finished task off the sim; what
it is worth comes from a data table (`economy/rewards.json`); the ledger re-derives
the payout from both before banking it. The robot — and, when it lands, the LLM
overseer — *sees* its balance and the reward table, and can move neither. An
agent that can score its own work learns to declare victory instead of doing
the task, so this is a structural rule rather than a policy.

**A hidden-truth task publishes its verdict without its ANSWER.** The census
knows how many plants are really in the garden; that number is redacted from
`metrics` and from `reason`, because this stream reaches both the website and
the overseer's context, and a task the robot is supposed to *discover* must not
arrive pre-solved in its own scoreboard. `ok` says whether the answer was right
and nothing says what it was.

**`pending` is the deferred (visitor-judged) slot.** An aesthetic call cannot
be made by code, so the evaluator confirms the work happened and banks zero;
when the rating arrives over the inbound channel (pluggybot #16) the same entry
is re-emitted with `"settled": true` and its points filled in. Nothing produces
one yet — no task in the shipped reward table is visitor-tiered except
`artwork`, which nothing builds — so a consumer may treat it as reserved.

**`spent` is always 0 today.** Spending is designed and deliberately not
implemented: when it lands it buys cosmetic and capability unlocks only (face
styles, figures, zone or tool access) and never anything the survival loop
depends on. The field is here so ledgers written now stay readable then.

### 0.4.0 → 0.5.0 (the robot has a face, and a board can be caught up on)

Two additions, both about surfaces the browser PAINTS rather than geometry
MuJoCo carries (pluggybot #13, rooftop-media-2026 #28). Additive: a 0.4.0
consumer that ignores both renders exactly what it rendered before.

- Frames may carry a **`screens`** object: per display module, what it is
  showing. Sparse and keyframe-refreshed on exactly the same rule as
  `activities` and `boards`, and for the same reason — an LCD's content is
  not a pose, so this block is the only record of it anywhere in the stream.
  The header gains **`"screens": [names]`**, and the SCENE gains a top-level
  **`screens`**: which geom on which body carries each display, plus the
  panel's outward normal, so a client can place a face without deriving one
  name from another.
- A third typed message joins `draw` and `board_cleared`:
  **`board_snapshot`**, every stroke a board is currently carrying, sent
  when a stream OPENS.

```jsonc
// in a frame: what the LCD is showing
"screens": {"module_lcd": {"mode": "face",        // off | face | text | count
                           "powered": true,
                           "face": "curious",     // FACE_STATES
                           "hint": "blink"}}      // SCREEN_HINTS, a LOOP
"screens": {"module_lcd": {"mode": "count", "powered": true, "face": "happy",
                           "hint": "bounce", "count": 4, "label": "plants"}}
"screens": {"module_lcd": {"mode": "off", "powered": false}}   // on the rack

// in the scene: where to paint it
"screens": {"module_lcd": {"geom": "module_lcd_screen", "body": "module_lcd",
                           "size": [0.004, 0.056, 0.076],   // full extents
                           "pos": [-0.01, 0, 0], "quat": [1, 0, 0, 0],
                           "normal": [-1, 0, 0]}}   // body-local, OUTWARD

// catching a late joiner up on ink (see below)
{"type": "board_snapshot", "t": 0.0, "robot": "pluggybot",
 "board": "whiteboard_a", "clearedAt": "...", "drawnAt": "...",
 "dropped": 0,                                  // strokes aged out of the cap
 "strokes": [{"program": "house", "stroke": 0, "points": [[lat, height], ...]}]}
```

**The vocabularies are a two-repo contract**, on the same terms as
`VISUAL_HINTS`: `SCREEN_MODES`, `FACE_STATES` (`idle`, `happy`, `curious`,
`determined`, `surprised`, `sleepy`, `worried`) and `SCREEN_HINTS` (`none`,
`blink`, `bounce`, `shake`) live in `telemetry/protocol.py`. Adding a face is
additive — an unknown one falls back to `idle` — and renaming one is a
breaking change in both repos. `face`/`hint` are absent when `mode` is
`off`; `text` and `count`/`label` appear only in their own modes.

**`hint` is a LOOP, not an event.** The sim never ticks a blink: a 150 ms
eyelid does not belong on a 20 Hz pose stream, and a dropped frame would
stick the eyes shut. The browser owns the animation, as it owns everything
organic.

**`powered` is electrical, not positional.** It is the coupling's own
criterion (`module_power_contact`, both poles conducting), so a module
hanging in its bay is `off`, and so is a half-seated one on the fork.

**Why `board_snapshot` had to exist.** Keyframes re-ship every board's
counters, but no keyframe re-ships the LINES: a stroke is an event that
happens once. So a browser that opens the page after the pen has moved on —
or after the producer restarted onto a state file, since `tools/boards.py` now
persists the polylines themselves — sees a board reporting 40 % fill with
nothing painted on it. The snapshot is what closes that gap. It is sent when
a stream opens (a recording carries it right after the header; the live
publisher re-sends it on every connect), and a relay hub should cache the
latest one per board plus the `draw` events since, dropping both on a
`board_cleared`. `dropped` counts strokes aged out of the per-board cap
(240): a snapshot missing the start of a long drawing says so rather than
pretending to be complete.

### 0.3.0 → 0.4.0 (whiteboards are state, and ink is an event)

Boards became persistent world state the sim owns and streams (issue #12).
**Mostly additive, with one thing to fix in a replayer** — see the second
bullet.

- Frames may carry a **`boards`** object: per drawing surface, which stroke
  programs are on it, how much of the pen's reach carries ink, and when it
  was last cleared. Sparse and keyframe-refreshed on exactly the same rule
  as `activities`. The header gains **`"boards": [names]`**.
- **Recordings are now a mixed stream.** Two typed messages ride *between*
  the frames — `draw` and `board_cleared` (below) — where before, every
  line after the header was a frame. **Dispatch on `type`; no `type` means
  frame.** That rule already governed the live stream; it is now true of
  `.jsonl` recordings too, and a 0.3.0 replayer that assumed otherwise
  trips over the new lines.

```jsonc
// one stroke, emitted as the pen finishes it
{"type": "draw", "t": 123.4, "robot": "pluggybot", "board": "whiteboard_a",
 "program": "house", "stroke": 3,
 "points": [[lat, height], ...]}     // board-local metres, +lat = viewer's LEFT

// the board was erased (start of a drawing errand, or an explicit action)
{"type": "board_cleared", "t": 120.0, "robot": "pluggybot",
 "board": "whiteboard_a", "clearedAt": "2026-08-16T08:01:09+00:00"}
```

**Strokes are never geometry.** Ink is layer 3 of the three-layer model
(rigid bodies in MuJoCo · discrete state in Python · everything organic in
the browser): the site paints these polylines into a canvas texture on the
board mesh. Nothing about a drawing appears in the pose stream, so these
messages plus the `boards` block are the *only* record of it — the same
argument that put `activities` on the wire in 0.3.0.

**Consumer notes.** `points` are in the board's own frame, origin at the
board's centre, metres, rounded to 0.1 mm — the board's pose and half-extents
come from the scene description. `lat` is measured to the LEFT of the robot's
approach, so a canvas drawn in screen coordinates flips it once. The polyline
is what the pen actually **inked**, not what it was commanded, so it carries
the robot's real ~1 mm of form error and a stroke that lost the board comes
through short. `board_cleared` means *drop everything you have painted for
that board*; a `draw` event whose board you have no canvas for is safe to
ignore. Board `fill` is measured against the window the pen can reach with
the base parked (110 × 200 mm on the home boards), not against the whole
slab.

### 0.2.0 → 0.3.0 (activities join the frame)

Additive; a 0.2.0 consumer ignores the new block and needs no changes.

- Frames may carry an **`activities`** object: the task state machines'
  discrete world state (issue #8), e.g.
  `{"garden_gate": {"state": "open", "pressed": false, "depressMm": 1}}`.
  Sparse like body poses — only activities whose flags changed appear, and
  the block is omitted when nothing did — and re-shipped in full on every
  keyframe, so a mid-stream joiner is complete within one keyframe interval
  exactly as it is for poses.
- The header gains **`"activities": [names]`**.

Why it is not merely a convenience: an activity's visible effect usually
lives on a **static body**. The reference gate is a mocap body that ships
once in the scene description and never again, and its pose is not in the
pose stream at all — so for a change like that the flag is the *only*
record anywhere in the stream. Flag semantics are the activity's own; the
website keys its visuals (a swing, a glow) off them and simulates none of it.

### 0.1.0 → 0.2.0 (keyframes recur, and say so)

Additive; a 0.1.0 consumer reading a 0.2.0 stream needs no changes.

- Frames that re-ship **every** dynamic body now carry `"key": true`.
  Frames without it are sparse, exactly as before.
- Those keyframes **recur**, every `keyframeS` sim-seconds (new header
  field, 5.0), rather than occurring only at `t = 0` and after a live
  reconnect.

Why: a browser joining through the website's relay hub is invisible to
the sim — no reconnect fires, so nothing re-keys for it, and every body
that had settled before it arrived is missing from its world forever.
Recurring keyframes bound that wait, and the marker means the hub can
recognize a cache boundary with a field read instead of a set comparison
against the body census.

## Files

| File | What | Regenerate with |
|---|---|---|
| `scene.room_hub.json` | Static scene description of `models/room_hub.xml` | `uv run python -m pluggybot.telemetry.scene` |
| `scene.home_world.json` | The generated home world, with visual hints + zones + spawns (issue #6) | `uv run python -m pluggybot.telemetry.scene models/home_world.xml` |
| `home_world.meta.json` | The generator sidecar the scene JSON was built from | `uv run python -m pluggybot.home.world` |
| `textures/*.png` | The AprilTag textures, decoded from the compiled model | (same command) |
| `telemetry.hub_lifecycle.jsonl.gz` | Full battery-driven mission in **room_hub** (explore → charge → fetch tool → stow), with a **task** offered, claimed and graded (0.9.0) | `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --tasks --record protocol/telemetry.hub_lifecycle.jsonl.gz` |
| `telemetry.home_lifecycle.jsonl.gz` | The same loop in the **home world** (issue #9) running the **showcase** queue: a drawing errand (issue #12) *and* a census on the LCD (issue #13), so one recording exercises BOTH streamed surfaces — what the live site serves, and the fixture the canvas painter and the face component are built against | `MUJOCO_GL=egl uv run python scripts/hub_lifecycle.py --world home --errand showcase --tasks --boards state.json --record protocol/telemetry.home_lifecycle.jsonl.gz` |

⚠ **`--tasks` is load-bearing on both recordings** (0.9.0). Job offers are
off by default — a task board adds errands, which reshuffles a whole mission
— so a recording made without the flag carries no `tasks` block at all and
the website's marker code has nothing to build against.

⚠ **Record the home fixture TWICE against the same `--boards state.json`, and
keep the second.** A board that survived a previous run is what makes the
recording open with a `board_snapshot` (0.5.0), which is the one case no other
message in the stream can rebuild — so a fixture recorded onto blank boards
quietly stops covering it. The ledger needs no such warm-up: a run starts the
robot at zero and the balance climbs on the wire, which is what the site's
scoreboard renders.

**One recording per scene, and they are not interchangeable.** A replayer
picks its scene off the header's `model` field, so playing the room_hub
recording against the home scene poses the robot inside the wrong house —
which renders as a robot driving through walls, not as an error.
`tests/test_telemetry.py` checks each recording's `model` label and each
scene against its committed world.

## Scene description (fetched once)

Transpiled from the **compiled** `MjModel`, so includes/defaults/generated
files are already resolved. Top level:

```jsonc
{
  "protocolVersion": "0.3.0",
  "model": "room_hub",
  "upAxis": "z",              // see "conventions" below
  "bodies": [ ... ],
  "textures": [{"name": "tagtex0", "file": "tagtex0.png", "width": 240, "height": 240}]
}
```

Per body: `name`, `parent` (body name, `null` for the world root),
`dynamic` (whether telemetry will stream poses for it), `robot` (the owning
robot's name, `null` for shared/world bodies), `visual` (the parametric
visual hint — see below), `pos` +
`quat` (**world-frame rest pose**, so a client renders with no
kinematic-tree math: static bodies keep this pose forever, dynamic bodies
get theirs overwritten by telemetry), and `geoms`.

`visual` comes from the world generator's sidecar
(`models/<world>.meta.json`, issue #6): the website renders a parametric
component per hint and falls back to raw primitives for `null` or for a
hint it does not know. Vocabulary v1 (`telemetry.protocol.VISUAL_HINTS`):
`wall`, `fence`, `floor`, `ground`, `whiteboard`, `rack`, `plant`. Adding
a hint is additive; renaming one is a two-repo breaking change. Hints ride
in the sidecar and **never** in geom colors — the robot's cameras render
rgba, so colour-as-encoding would couple perception to art direction.

A generated world may also carry three optional top-level fields, likewise
additive: `zones` (named rectangles, `{name, kind, min:[x,y], max:[x,y]}`),
`spawns` (`name → [x, y, yaw_rad]`), and `boards` — the drawing surfaces, by
the name telemetry uses:

```jsonc
"boards": {"whiteboard_a": {"geom": "board",          // the geom in `bodies`
                            "pos": [-1.97, 1.0, 0.3], // world frame, m
                            "half": [0.01, 0.16, 0.13],  // depth,width,height
                            "heading": 3.14159}}      // outward normal, rad
```

This is what places a `draw` event's polyline: the event names the *board*
(`whiteboard_a`), the scene says which geom that is and how big its face is,
so a canvas is `2 × half[1]` by `2 × half[2]` metres with the polyline's
origin at its centre.

Per geom: `name`, `type` (`box | cylinder | capsule | sphere | plane`),
`size`, `pos` + `quat` (body-local, constant), `rgba`, `texture` (name into
the textures table, or `null`). Geoms with rgba alpha 0 (invisible
collision layers, e.g. the schuko socket wells) are omitted.

### Conventions — conversions already applied

- **Units** meters; quaternions `[w, x, y, z]` (MuJoCo order).
- **Sizes are FULL extents**, converted from MuJoCo's half-extents:
  box `[x, y, z]`; cylinder/capsule `[radius, length]` (capsule length is
  the cylindrical part); sphere `[radius]`; plane `[x, y]` with `0` meaning
  infinite.
- **Cylinder/capsule axis**: MuJoCo's runs along local +Z, ThreeJS's along
  local +Y; each such geom's `quat` already contains the +90° X rotation
  that maps a Y-axis primitive onto the MuJoCo geometry. Planes agree on
  +Z normals and need no fix.
- **The world frame is NOT converted** (`upAxis: "z"`): all poses, here and
  in telemetry, are MuJoCo Z-up world frame. Apply one Z-up → Y-up rotation
  at the ThreeJS scene root.

## Telemetry (JSONL, one object per line)

Line 1 is a header; every later line is a frame at ~`hz` (20) of **sim
time**. A `.gz` suffix means gzip (`zcat` to inspect).

```jsonc
// header
{"type": "header", "protocolVersion": "0.6.0", "model": "room_hub", "hz": 20.0,
 "keyframeS": 5.0,                                      // sim s between keyframes
 "robots": {"pluggybot": ["pluggybot", "head", ...]},   // dynamic bodies per robot
 "robotNames": {"pluggybot": "Pluggy"},                 // id → display name (0.10.0)
 "world": ["rack", "module_lcd", ...],                  // shared dynamic bodies
 "activities": ["garden_gate"],                         // task state machines
 "boards": ["whiteboard_a", "whiteboard_b"],            // drawing surfaces
 "screens": ["module_lcd"],                             // display modules
 "ledger": ["pluggybot"],                               // robots with a balance
 "taskKinds": ["draw_figure", "count_plants"],          // jobs it can offer
 "accepts": ["message", "rating"]}                      // what it will act on

// frame
{"t": 123.45,                                  // sim seconds
 "key": true,                                  // present only on keyframes
 "robots": {"pluggybot": {
   "bodies": {"pluggybot": [x, y, z, qw, qx, qy, qz], ...},   // world-frame
   "state": "EXPLORE",                         // lifecycle state machine
   "status": "EXPLORE -> GO_CHARGE (battery low)",            // the _say line
   //  ⚠ BOTH ARE THE SIM TALKING TO ITSELF -- see "Narration is not UI copy"
   //  below before a consumer prints either one.
   "battery": {"frac": 0.61, "watts": 14.2, "charging": false}}},
 "world": {"module_lcd": [x, y, z, qw, qx, qy, qz]},
 "activities": {"garden_gate": {"state": "open", "pressed": false}},
 "boards": {"whiteboard_a": {"programs": ["house"], "strokes": 7,
                             "inkM": 0.459,          // metres of ink laid down
                             "fill": 0.191,          // of the pen's REACH
                             "clears": 1,
                             "clearedAt": "2026-08-16T08:01:09+00:00",
                             "drawnAt": "2026-08-16T08:01:47+00:00"}},
 "screens": {"module_lcd": {"mode": "face", "powered": true,
                            "face": "curious", "hint": "blink"}},
 "ledger": {"pluggybot": {"balance": 39, "earned": 39, "spent": 0,
                          "tasks": 3, "pending": 0,
                          "recent": [{"seq": 3, "task": "census", "points": 20,
                                      "ok": true, "t": 412.5}]}},
 "tasks": {"t_0001": {"id": "t_0001", "kind": "draw_figure",
                      "state": "done", ...}}}   // WHOLE, not merged
```

**Frames are sparse.** The first frame is a keyframe carrying every dynamic
body; later frames carry only bodies that moved > 0.5 mm (or the quat
equivalent) since they were last emitted. A body absent from a frame is
unchanged — a replayer holds the last value it saw. `bodies`, `world` and
`activities` are omitted entirely when empty; `state`/`status`/`battery`
ride in every frame. **Activity flags, board state, screen content and the
points ledger are sparse on the same rule as poses** and re-ship on every
keyframe (0.3.0, 0.4.0, 0.5.0, 0.6.0). Positions are rounded to 0.1 mm.

**`tasks` is the exception**, and the only one: it is sparse in *time* like
the rest, but when present it is the **complete** board rather than a delta
— replace it, do not merge it. A task can cease to exist, and a per-key
delta cannot say so (0.9.0).

**Keyframes recur** every `keyframeS` sim-seconds (5.0) and are marked
`"key": true`. A consumer that starts reading anywhere in the stream is
complete within one such interval; one that starts at the top can ignore
the marker entirely, since a keyframe is just a frame that happens to
mention everything. At 20 Hz they are 1 frame in 100.

The producer seam: `TelemetryRecorder` (`src/pluggybot/telemetry/recorder.py`)
is a callback on `HubMission.step_hooks` — the same per-physics-step seam
the battery drains through. It decimates 500 Hz of steps to `hz` of frames
and hands them to a writer thread; no serialization or file I/O ever runs
inside a physics step.

### A message the queue threw away says so

⚠ **`visitor_reply.outcome` gains a fourth value, `dropped`**
(rooftop-media-2026 #124). Additive, and no version bump: a consumer that has
never heard of it falls through to whatever it already does with an outcome it
does not know, which is the `FACE_STATES` rule.

```jsonc
{"type": "visitor_reply", "t": 412.5, "robot": "pluggybot", "id": "m_02",
 "kind": "message", "outcome": "dropped", "reply": "", "action": ""}
```

The inbox is a **bounded drop-oldest deque** (`MAX_QUEUE` = 32), so a burst
evicts messages the robot never read. `Inbox.dropped_full` counted them and
**reached nothing outside the process**, so a website holding that row could
only report it as still waiting — forever, on a message nobody was ever going
to answer. "Nobody has answered you yet" and "your message was thrown away"
are different facts, and only one of them is worth waiting on.

It reuses `visitor_reply` rather than earning a type of its own because the
correlation machinery already exists on both sides: a consumer closes the row
by `id`, exactly as it does for the other three. What differs is **who
generated it** — the queue, not a decision — which is why `reply` and `action`
are empty. There was nobody to write one.

⚠ **A MIND CANNOT SAY IT.** `DECIDED_OUTCOMES` is the three a model may
choose and rides its grammar; `VISITOR_OUTCOMES` is all four and is what a
consumer must render. Offered to a model, `dropped` is a free excuse for not
answering — and one indistinguishable on the wire from the truth. Same rule as
the reward table: the party that benefits from a claim is not the party that
gets to make it.

⚠ **It is best-effort, and the website should not rely on it alone.** Anything
still waiting to be reported when a mission ends dies with the process, the
same trade the queue itself makes. A consumer that cares about not losing
messages re-delivers unsettled rows on the next producer connect; this makes
the common case legible, it does not make the channel lossless.

### Narration is not UI copy

⚠ **Three fields on this wire are the sim's own vocabulary, and a consumer
must not print any of them raw** (rooftop-media-2026 #123 — a deliberate
contract change, recorded here as well as there).

| field | example | what it is |
| --- | --- | --- |
| `state` on a robot's frame record | `SWAP_PICK` | a state machine's name for itself |
| `task` on `earned` / an `earning` | `census` | a row in `economy/rewards.json` |
| `action` on `visitor_reply` | `dance` | one of the overseer's `ERRAND_ACTIONS` |

The website printed the first of them in bold under a label a visitor could
read, and it measured **27 % of the home recording** spent telling somebody
the robot was doing something called `SWAP_PICK`. These are identifiers a
client should map to its own words; the site now renders a phrase per token
and falls back to the raw token for one it has never heard of — all three are
OPEN sets, on the same terms as `FACE_STATES` and `taskKinds`, so a new
lifecycle state or reward row costs a consumer the phrasing and never the
fact.

`status` is different and stays as it is: it is the `_say` line, real prose
written by the sim, and worth showing **as a log** rather than as a headline.
⚠ It is **English**, and so are `earned.reason`, a task `description`, a
journal note, a memory document and a `visitor_reply.reply` — a client whose
page is in another language owes them a `lang="en"` region, or a screen reader
announces English words with the reader's phonemes. Do NOT mark a VISITOR's
own message that way: they wrote it in their own language.

Nothing here changes what the sim emits. It is a note about what the bytes
MEAN, which is exactly what this file is for.

## The live stream (webserver v1)

`scripts/serve.py` publishes the same objects over an outbound WebSocket
(`WsPublisher`, `src/pluggybot/telemetry/publisher.py`) — the recorder and
the publisher build frames with the same `FrameBuilder` code (each owns an
instance), so a live consumer and a replayed recording see identical data,
provided both are configured alike: `serve.py --record` passes its
`--keyframe-s` to both, which is the only setting that can make them
disagree. Live-stream rules:

- **Dispatch on `type`; no `type` means frame.** The header and the extra
  message types below carry a `type` field; telemetry frames never do.
  **Consumers must ignore message types they do not recognize** — new
  low-frequency types are additive and do not bump `protocolVersion`
  (only a change to the *shape* of an existing artifact does).
- **Every connection opens with the header, then a keyframe.** Sparse
  frames are deltas against what was previously sent, so whenever
  continuity breaks — a (re)connect, or frames dropped because nobody was
  draining the socket — the next frame re-ships every dynamic body, marked
  `"key": true`. A consumer joining mid-mission starts from nothing and is
  complete within two frame intervals: the re-key is requested by the
  sender thread and honoured by the next physics step, so one sparse frame
  can slip out between the header and the keyframe. Applying it early is
  harmless — the keyframe overwrites everything it touched.
- **A relay hub only needs to cache the last keyframe and the frames
  since it.** The sim connects outbound to the hub and stays connected;
  browsers come and go behind it, invisibly. Nothing re-keys on their
  behalf, which is what recurring keyframes are for. On a browser join,
  replay the cached header, then the cached keyframe, then the frames
  after it, then go live — the cache is bounded by `keyframeS` × `hz`
  (≈100 frames). Also worth caching: the most recent `grid` message per
  robot, so a joiner's map is not blank for a second, and the ink: the
  latest `board_snapshot` per board plus the `draw` events since it, both
  dropped when a `board_cleared` for that board goes past (0.5.0). Nothing
  else in the stream can rebuild a board — a keyframe carries its counters,
  never its lines.
- Frames can drop under load; recordings are the lossless artifact.

**The ingest connection is authenticated.** The publisher presents the
shared secret as an `Authorization: Bearer <token>` request header at the
WebSocket handshake (`scripts/serve.py --token`, or `$PLUGGYWORLD_TOKEN`);
a server that dislikes it should refuse the handshake with `401`. A token
in the URL query works too and needs no producer support — it is just part
of `--endpoint` — but it lands in access logs, so the header is the
default. Refusal is not fatal to the sim: it retries every second like any
other unreachable endpoint, and reports the last failure at exit. Note that
the header only beats the query param on *log* exposure — over plain `ws://`
the bearer crosses the network in cleartext either way, so anything leaving
the host wants `wss://`. An empty token is rejected at startup rather than
sent as no header at all, because a blank `PLUGGYWORLD_TOKEN` would
otherwise publish unauthenticated and read as a wrong secret.

The lower-frequency message types:

(`draw`, `board_cleared`, `board_snapshot` and `earned` ride this socket too,
and are four of the types that also appear in recordings — see 0.4.0, 0.5.0
and 0.6.0 above. An `earned` message needs no hub cache of its own: the
balance and the last few earnings re-ship in every keyframe.)

⚠ **`grid` is no longer live-only** (rooftop-media-2026 #78). It was, from
0.2.0 until the map was rendered: the publisher shipped it and the recorder
did not. But the website's *default* view is a recording — live falls back to
one whenever no sim is publishing — so a map panel fed only by the live
stream is blank for almost every visitor, and a replayed mission was of a
robot that never had a belief about where it was. A recording now carries the
map too. `event` remains live-only; a recording's narration is in each
frame's `status` line.

This is **additive and not a version bump**: the message's shape is
unchanged, and a consumer that dispatches on `type` (the rule since 0.4.0)
ignores a line it has no use for. A recording written before this change
simply has no `grid` lines, and a map panel must render *that* as "no map was
published", never as "nothing was mapped".

Two differences between the recorded map and the live one, both deliberate
(`telemetry/recorder.GridSampler`):

- **Cadence.** Live is 1 Hz; a recording is 0.2 Hz. A recording is watched
  from the top, so what matters is seeing the map fill in, not that it is at
  most a second stale — and these bytes are permanent and vendored into the
  website's bundle, where every visitor pays for them on page load.
- **A recording skips an unchanged map.** The belief stops moving for minutes
  at a time — through a charge, through a drawing — and a byte-identical PNG
  written every interval is pure weight. The live stream does *not* skip:
  the hub caches the most recent `grid` per robot for late joiners, so a
  stream that fell silent because nothing changed would look exactly like one
  whose grid path had broken.

```jsonc
// occupancy-grid belief, ~1 Hz per robot: base64 PNG, uint8 cells
// (0 = wall, 255 = free, 127 = unknown), row 0 = y_min edge
{"type": "grid", "t": 123.4, "robot": "pluggybot",
 "extent": [-3, -3, 7, 7],        // [x_min, y_min, x_max, y_max], world m
 "resolution": 0.05, "png": "iVBORw0..."}

// lifecycle narration (the _say lines), as they happen
{"type": "event", "t": 123.4, "robot": "pluggybot",
 "line": "EXPLORE -> GO_CHARGE (battery low)"}
```
