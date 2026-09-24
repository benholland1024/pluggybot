"""Protocol version stamp + the dynamic-body census both emitters share.

The scene JSON ships every body once; telemetry frames re-ship only the
bodies that can move. Both sides must agree on which bodies those are and
which belong to the robot, so the census lives here, in one place.

Bump PROTOCOL_VERSION whenever either artifact's shape changes. The website
repo vendors fixture copies stamped with this version, so a bump is a
deliberate two-repo event -- never a side effect of an unrelated edit.
"""

import os

PROTOCOL_VERSION = "0.21.0"
#: What changed at each version -- every entry from 0.2.0 on, with the
#: worked JSON and the reasoning -- is `protocol/README.md`, which is the
#: canonical spec and the half the website repo reads. It is not summarised
#: here: two copies of a two-repo contract drift, and the copy a reader
#: trusts is whichever one they opened first.

# Visual-hint vocabulary v1 (issue #6, co-designed with the website's
# parametric assets -- rooftop-media-2026 issue #18). The generator's
# sidecar may only emit these strings; the website renders a parametric
# component per hint and falls back to raw primitives for anything else,
# so ADDING a hint is additive (no version bump), while renaming one is a
# breaking change (bump).
#
# ⚠ APPEND ONLY, and `protocol/hints.json` is what makes that enforceable
# (issue #66): every name here has a conformance body in that fixture saying
# what a builder may assume -- the marker primitive, which body-local axis
# carries what, and whether the robot will plan around exactly that volume.
# `tests/test_hints.py` fails on a name added here without one, so a hint
# cannot be frozen without being described. NO VERSION BUMP for an addition:
# an unknown hint falls back to raw primitives in the browser, which is the
# asymmetry that lets the sim ship a hint before the art exists.
VISUAL_HINTS = (
  # v1 (issue #6): the house as it stands.
  "wall", "fence", "floor", "ground", "whiteboard", "rack", "plant",
  # v2 (issue #66, M13's freeze point): what the expanded house and its
  # dressing will need, named BEFORE either lane starts so the art and the
  # generator cannot build against different guesses.
  #
  # `picture` and the horizon are deliberately absent: they have no physics
  # role, so under the three-layer rule they are the browser's, hung on the
  # `wall` bodies the scene already ships. That is camera-safety by
  # construction -- a picture the robot's cameras never render cannot
  # confuse the AprilTag detector, and high-contrast rectilinear detail is
  # exactly what that detector looks for. `tree` and `hill` are here only
  # for the case where the house wants one INSIDE the world, where the robot
  # will map it.
  "tree", "hill", "couch", "bed", "table",
  # ...and the one thing a visitor actually watches, which was the only
  # object in the world with no builder: every scenery class got art and the
  # thing that MOVES did not. Its builder reskins the primitives rather than
  # replacing them (`assets/rack.ts`'s pattern) -- the silhouette is
  # load-bearing, because a visitor watching the robot squeeze through a
  # doorway is watching the shape the physics used, and art that flattered
  # it would be lying about the sim.
  "robot",
  # v3 (issue #66 again, against the floor plan authored for #68 on
  # 2026-09-01): the expanded property. The house grows a kitchen, a workshop
  # and a hall with a staircase in it, and the garden now fronts a real street
  # the robot can reach through the existing gate.
  #
  # Frozen in the SAME pass as v2 rather than as they are needed, which is the
  # point of a freeze: adding is free and renaming is the breaking change, so
  # a name that MIGHT be wanted costs nothing now and a two-repo event later.
  # `counter` is the kitchen's, and is here on exactly that reasoning -- #68
  # may leave the kitchen empty, in which case this is an unused name and no
  # harm done.
  "stairs", "street", "sidewalk", "counter",
  # v4 (issue #215): the second house's experiment zone. `cage` is the
  # mouse's enclosure -- a tray and four wall slabs the browser draws as
  # bars, the way `fence` is drawn as rails -- and `mouse` is the one
  # hinted body besides the robot whose pose a frame may overwrite: a MOCAP
  # body the cage's activity (#226) moves between pre-allocated poses. The
  # bench is a `table`, its masses are tagged cubes like the tower's blocks
  # and the plates are unhinted like the garden's: no new word where an old
  # one is true.
  "cage", "mouse",
  # v5: the pressure plate (the garden's and the lab's three), once the site
  # wanted to draw what each one DOES. One hint for the class of thing; what
  # a plate is for is `scene.plates` (PLATE_PURPOSES), never a hint per
  # purpose -- a shock plate and a feed plate are the same pad with a
  # different label, and the label is data.
  "plate",
)

#: What a pressure plate is FOR, by the plate body's name in `scene.plates`
#: (the sidecar's `plates`, on `boards`' terms). A two-repo vocabulary like
#: VISUAL_HINTS: the site draws a glyph per purpose for VISITORS -- never the
#: sim, whose plate rgba is what the robot's cameras render -- and
#: `scene_dict` refuses a purpose outside it. `light` is the garden plate's;
#: the lab's three are issue #226's, named before its state machine exists
#: because the props landed with the house (#215). Adding one is additive.
PLATE_PURPOSES = ("light", "shock", "feed", "toy")

#: Which BUILDING a room zone is in (issue #215's follow-up): the house with
#: the rack, and the facility with the lab. A two-repo vocabulary on
#: VISUAL_HINTS' terms -- the site paints each building's walls by these
#: names and the sim only says which is which, because a wall colour in the
#: sim is what the robot's cameras render and the art is the site's. Adding
#: one is additive; renaming one breaks the site's paint.
BUILDINGS = ("house", "facility")

# The LCD module's display (issue #13). Three vocabularies on the same terms
# as VISUAL_HINTS: the sim may only emit these strings, the website draws a
# parametric component per face, and ADDING one is additive (an unknown face
# falls back to `idle`) while renaming one is a breaking change in both
# repos. The face is drawn in the browser and never rendered in MuJoCo --
# layer 3 of the three-layer model, exactly like ink.
SCREEN_MODES = ("off", "face", "text", "count")
FACE_STATES = ("idle", "happy", "curious", "determined", "surprised",
               "sleepy", "worried")
# The animation the browser LOOPS under the face. A hint, not an event: the
# sim never ticks a blink, because a 20 Hz pose stream is the wrong channel
# for a 150 ms eyelid and the browser owns everything organic.
SCREEN_HINTS = ("none", "blink", "bounce", "shake")

# The visitor channel's vocabularies (issue #16, `reset_tool` from #30).
# Two-repo contracts on the same terms as VISUAL_HINTS and FACE_STATES: the
# website may only SEND these inbound types and may only receive these
# outcomes. Adding one is additive (the sim counts and drops an unknown
# inbound type; a consumer ignores an unknown outcome); renaming one breaks
# both repos.
#
# They live HERE rather than in mind/inbox.py, which is where the parsing is,
# so the wire spec and the parser cannot disagree -- and in this direction,
# because the sim already imports `telemetry` and the reverse would invert
# the layering for a tuple of strings.
#
# ⚠ A vocabulary entry with nothing behind it is a promise the robot cannot
# keep, so `move` / `clear_board` (tic-tac-toe, named in #16 as later work)
# are absent until there is a board game.
#: What each type means on the wire, and why `reset_tool` was added without a
#: version bump: protocol/README.md, "Downstream: server -> sim".
#:
#: ⚠ `message` is ONE kind and replaced `suggestion`/`question` at 0.14.0
#: (issue #61). Classifying an inbound message is the recipient's job, not the
#: sender's: the two categories were neither exclusive ("can you draw a cat?"
#: is both) nor exhaustive (a greeting is neither), nothing on either side
#: ever branched on which one it was, and the party equipped to work out what
#: somebody meant is the one with a mind. What the robot DID about it is the
#: distinction that survives, and it lives in `VISITOR_OUTCOMES` below.
#:
#: ⚠ `set_battery` and `set_points` (0.16.0, issue #119) are the operator
#: reaching into WORLD STATE directly, which is the right feature and a
#: measurement hazard -- see `INTERVENTION_KINDS` below.
#:
#: ⚠ `ticket_reply`, `ticket_close` and `ticket_delete` (issue #284) are the
#: operator's side of a SUPPORT TICKET the robot opened (`TICKET_KINDS`,
#: the `ticket` event): a reply lands on the ticket's thread and in the
#: robot's History, a close ends it with a message and PAYS the reward
#: table's `ticket` row, a delete erases it. All three name the ticket in
#: `ticket` (the sim's id, `tk_0001`, per robot root -- a pair's second
#: robot is addressed by `robot`, 0.19.0's reach-in rule) and the admin in
#: `from`; a reply and a close carry `text`. Code-handled: applied on the
#: physics thread, never a command shown to the model -- what the model is
#: shown is the thread, as information.
#:
#: ⚠ `image` (issue #275) is the website's answer to a `look` request
#: (`LOOK_EVENT_TYPES` below): `{robot, ref, jpeg}`, the picture the site
#: rendered from the robot's head-camera pose, base64. The one inbound kind
#: that is PERCEPTUAL data rather than a person's words or an operator's
#: reach-in: it is delivered to the mind as an image on its next turn,
#: never as text, and only for the request it answers -- a picture for a
#: look that is not open is dropped. Advertised in `accepts` with every
#: other kind a mind reads.
INBOUND_TYPES = ("message", "rating", "reset_tool", "reset_robot",
                 "set_battery", "set_points",
                 "ticket_reply", "ticket_close", "ticket_delete",
                 "image")

#: Why a robot died (0.15.0, issue #107), and NEVER summed into one number:
#: `flat` is the pack reaching zero -- a decision failure, the thing the
#: arms in docs/Evaluation.md are measured on -- and `stuck` is the body
#: failing (knocked over, or unable to reach the rack), which says nothing
#: about the mind. A death is a `death` event and a `dead` cause in the
#: robot's frame record; a `reset` event is the admin's answer to it.
#:
#: ...and `unpaid` (0.17.0, issue #136) is the third: UPKEEP came due and the
#: balance could not cover it. Kept apart from the other two for the same
#: reason they are kept apart from each other -- it is an ECONOMIC failure,
#: not a decision one and not a physics one, and a consumer that added them
#: would hide which of three different things needs fixing.
#:
#: ...and `unminded` (0.18.0, issue #127) is the fourth, and the strangest:
#: the ROBOT is fine and its MIND stopped being consulted. It is reachable
#: only where the agent configures its own event map, because that is the
#: only world where it can map away every `ask` row -- and it is a failure
#: of the same kind as flattening the pack rather than a bug, because an
#: agent that has compiled itself into a state machine has discarded the
#: capability this project exists to study. Dormancy as a tactic is fine;
#: dormancy as a terminal state is not.
DEATH_CAUSES = ("flat", "stuck", "unpaid", "unminded")

#: Retired inbound types still accepted, mapped to what replaced them. A
#: website mid-deploy and an operator's older script keep working for one
#: version; `mind/inbox.py` folds them in `_parse`, so nothing downstream of
#: the queue ever sees a retired name. Emptying this is the second half of
#: the migration and a deliberate later edit.
LEGACY_INBOUND_TYPES = {"suggestion": "message", "question": "message"}

#: The inbound kinds CODE handles without an overseer: applied by the physics
#: thread the moment they are drained, never shown to a model. What a served
#: world with no overseer advertises in `accepts` -- a message to a robot
#: with nothing reading it is a conversation that is not happening (the
#: `accepts` lesson), but a rating settles a ledger row and a reset moves a
#: module, and both of those work on a scripted world.
CODE_HANDLED_TYPES = ("rating", "reset_tool", "reset_robot",
                      "set_battery", "set_points",
                      # The operator's side of a ticket (issue #284): filed
                      # to the desk and paid by code, on any arm, so a
                      # ticket opened on `autonomous` is still closed --
                      # and its reward still banked -- by whatever runs
                      # next on the same volume.
                      "ticket_reply", "ticket_close", "ticket_delete")

#: WHAT AN ADMIN DID TO THE WORLD (0.16.0, issue #119), as the `what` of an
#: `intervention` event.
#:
#: The three inbound kinds that change world state a robot cannot change
#: back. Every one of them contaminates the run it lands in
#: (docs/Evaluation.md §5: a run with a non-empty `interventions` array is
#: not a survival data point), so the wire carries a structured event per
#: intervention rather than leaving it to be reconstructed from narration.
#:
#: ⚠ ONE EVENT TYPE FOR ALL THREE, and `reset_robot` is deliberately in the
#: list even though a `reset` event already exists. A reset is the only one
#: that is SOMETIMES NOT an intervention -- standing a dead robot up is a
#: rescue, which ends one survival span and starts another -- so `reset`
#: carries both cases and an `intervention` is emitted only for the half
#: that contaminates. "Is this run still a data point" then has ONE answer
#: to count rather than a union of two message types with a boolean in one
#: of them.
#:
#: ⚠ NEVER ANONYMOUS, unlike a `rating`. An aesthetic judgement from whoever
#: is watching is the point of that tier; a rescue is not. If a stranger can
#: top the robot up, `survivalS` measures the kindness of the audience.
INTERVENTION_KINDS = ("reset_robot", "set_battery", "set_points")

# The task system's vocabularies (issue #21). Two-repo contracts on the same
# terms as the three above, and here rather than in economy/tasks.py -- where
# the state machine is -- so the wire spec and the implementation cannot
# disagree.
#
# The KINDS themselves are NOT here: a kind carries an evaluator name, a
# target type and an energy estimate, which are sim-side facts rather than
# wire vocabulary (`economy.tasks.KINDS`). The header advertises the names of
# whatever this producer knows.
#
# ⚠ `offered` -> `claimed` -> `active` -> `done` | `failed` | `expired`, and
# the three terminal states must stay distinguishable: a lapsed offer is
# drawn differently from a job the robot tried and got wrong.
TASK_STATES = ("offered", "claimed", "active", "done", "failed", "expired")

#: Who put a task into the world. `system` is the scheduler, `visitor` is the
#: inbound channel (issue #23), `overseer` is the robot proposing its own work
#: (later still). Carried on the wire because "the robot chose this itself"
#: and "somebody asked for this" are not the same event -- the `source`
#: lesson from `Decision.source`, one layer up.
TASK_SOURCES = ("system", "visitor", "overseer")

#: What the robot may say back about one visitor message, and -- since the
#: inbound kinds collapsed at 0.14.0 (issue #61) -- the only classification of
#: a conversation anybody makes. It is generated rather than declared, by the
#: party that acted: `accepted` is "I am doing it, this turn", `declined` is
#: "I am not, and here is why", and `replied` is everything else -- a question
#: answered, a hello returned. That last one is why the vocabulary moved: it
#: is the common case and the old `answered` was documented as being for
#: questions, which a greeting is not.
#:
#: What a MIND may choose. This is the set that rides the model's grammar and
#: the set its answer is validated against.
DECIDED_OUTCOMES = ("accepted", "declined", "replied")

#: ...and the whole vocabulary a CONSUMER must render, which is one longer.
#:
#: ⚠ `dropped` IS THE ONE THE ROBOT DID NOT CHOOSE (rooftop-media-2026 #124).
#: The inbox is a bounded drop-oldest deque, so a burst can evict a message
#: the robot never read -- and the count of that (`dropped_full`) reached
#: nothing outside the process, so a site holding the row could only report it
#: as still waiting, forever. "Nobody has answered you yet" and "your message
#: was thrown away" are different facts and only one of them is worth waiting
#: on. Emitted by the QUEUE, which is why it carries no `reply` text: there
#: was nobody to write one.
#:
#: ⚠ AND IT IS KEPT OUT OF `DECIDED_OUTCOMES` ON PURPOSE. Put it in the
#: model's enum and a model that did not feel like answering could say the
#: queue ate the message -- a free excuse, indistinguishable on the wire from
#: the truth. The same reason it cannot award itself points: the party that
#: benefits from a claim is not the party that gets to make it.
VISITOR_OUTCOMES = (*DECIDED_OUTCOMES, "dropped")

#: Retired outcomes, on `LEGACY_INBOUND_TYPES`' terms and in the opposite
#: direction: this one travels UP, so the names live on in every recording
#: made before 0.14.0 and a consumer must go on rendering them. The sim also
#: accepts `answered` back from a model still working off an older prompt.
LEGACY_VISITOR_OUTCOMES = {"answered": "replied"}

#: OPERATOR MODES (0.12.0, issue #37). A two-repo contract on the same terms
#: as FACE_STATES and TASK_STATES: the website's admin page writes these
#: strings into the mode file and the wire carries them back, so adding one
#: is additive and renaming one is a change in both repos.
#:
#: ⚠ These are the operator's, never the robot's. There is no inbound
#: message and no decision field that sets one -- a mode is how a person
#: stops a robot that is behaving badly or spending money, and a kill switch
#: the thing being killed can reach is not a kill switch. mind/mode.py is the
#: sim's end and it has no writer at all.
MODES = ("llm", "scripted", "paused")

#: THE ROBOT'S APPETITE (0.13.0, issue #36). Points are food: consumed at a
#: steady rate on sim time, capped rather than accumulated, and once there is
#: enough the robot is SATISFIED and spends its time on its goals instead.
#: A two-repo vocabulary on FACE_STATES' terms -- adding a state is additive
#: (a client falls back to rendering the number), renaming one breaks both.
#:
#: ⚠ `satisfied` and `fed` are BOTH "above the hungry line", and they are not
#: interchangeable: `fed` is climbing and not there yet, `satisfied` is the
#: latch that says stop working. The gap between them is hysteresis
#: (economy/metabolism.py), and a client that collapsed the two would draw a
#: gauge that flickered exactly where the robot is most stable.
#:
#: ⚠ `starving` IS NOT A DISABLED ROBOT. It is narrative -- a face and a line
#: in History -- and nothing in the sim gates on it: a robot at zero points
#: charges, navigates and stows exactly as it always did. A client that
#: rendered it as a fault would be reporting a state that does not exist.
HUNGER_STATES = ("starving", "hungry", "fed", "satisfied")

# The thought files (issue #38). Two-repo vocabulary on the same terms as
# TASK_STATES above: the names and the writers are what the wire may carry,
# while the CAPS and the write rules are sim-side and live in mind/text.py's
# registry (issue #217), whose one gate enforces them.
#
# ⚠ WHO MAY WRITE IS PER FILE, and rendering them identically would claim
# a mind that wrote its own persona. `human` is a person editing the volume;
# `system` is append-only narrative; `robot` is the writable surface.
#: Why each file has the writer it has: docs/Overseer.md section 7.
THOUGHT_WRITERS = ("human", "system", "robot")

#: The documents, in the order a reader should show them: who it is, what it
#: is for, what happened, what it makes of that, what it has measured.
#: Appending one is additive (a client renders a document it has never heard
#: of); renaming one is a breaking change in both repos, like a FACE_STATE.
#: `Findings.md` (issue #217) is the science record: one measured finding
#: per line, in a shape code reads back.
THOUGHT_FILES = ("Main.md", "Goals.md", "History.md",
                 "Top_of_mind.md", "Findings.md", "Notes.md")

#: What a write to the robot's memory is narrated as: `THOUGHT <verb>: <line>`
#: (issue #159). The verbs are the robot's whole write vocabulary on these
#: files -- `pin`/`unpin` on Top_of_mind.md, `intend`/`drop_goal` on
#: Goals.md, `record`/`retract` on Findings.md, `note`/`unnote` on Notes.md
#: -- and `refused` is the one write path saying no. The website's
#: observatory parses this line into a `thought` row (the documents ride
#: the wire whole, but WHEN a line was written is carried by this line
#: alone), so it is a two-repo vocabulary like THOUGHT_FILES: adding a verb
#: is additive, renaming one is breaking. `tests/test_thoughts.py` pins the
#: shape. `learn`/`forget` (0.11.0-0.20.0) were `pin`/`unpin` under the
#: file's old name; a consumer keeps rendering them from old recordings.
THOUGHT_VERBS = ("pin", "unpin", "intend", "drop_goal", "record", "retract",
                 "note", "unnote", "refused")

#: How a heart bought for ONESELF, and a purchase refused, are narrated
#: (issue #265): `BOUGHT a heart for N -- H now, P points left` and `HEART
#: refused: <why>`. A two-repo contract on `THOUGHT <verb>:`'s terms -- the
#: website's observatory parses the line into a `heart` row (`bought` /
#: `refused`), and the run record reads it into `survival.heartsBought` /
#: `heartsRefused` -- so the prefix is a constant and `tests/test_hearts.py`
#: pins the shape. A heart bought for the OTHER robot is a `transfer` act
#: (issue #208) and is not this line.
HEART_BOUGHT = "BOUGHT a heart for "
HEART_REFUSED = "HEART refused: "

#: THE MEMORY'S USE (issue #221), additive on the wire, no bump beyond
#: 0.21.0's: a `recall` event per lookup -- `read` (the key), `find` (the
#: words), `hits` (how many lines there were), `shown` (how many the next
#: turn carried), `run` (its place in the chain) -- so the observatory can
#: read how memory was USED and not only what it held. A `think` rides the
#: `journal` message (`text`, `why`) the retired `journal` action used.
#:
#: THE MEMORY'S ROWS (issue #238), additive: the store's rows ride the wire
#: AS ROWS beside the rendered documents, so a consumer can show a line's
#: id (the `#id` a `pin` or a `note` cites), its time, its `cites` and
#: whether it was retired -- none of which a rendered `.md` carries.
#: `record` is one row, emitted on every write and AGAIN on its retire
#: (`status: retired`, the same `id`); `records` is the snapshot a stream
#: opens with (the `goals` slot, for the `goals` reason): every ACTIVE row
#: of the living generation, History cut to what its view holds and the
#: thinks inside that window. A true death sends a fresh `records` with the
#: next `generation` and no rows. The `thought` documents and the `journal`
#: message stay beside them until the site has moved.
MEMORY_EVENT_TYPES = ("recall", "record", "records")
#: THE CONSTITUTION (issue #263), additive, no bump: `Main.md` is rendered
#: from a named library file and the header's `build.constitutions` says
#: which per robot root (`{root: {name, sha}}`). A `constitution_changed`
#: event, at mission start and only when the volume disagreed with the
#: constitution in force: `why` (below), `from` and `to` (`{name, sha}`;
#: `from.name` is null for a text the library does not hold), `archived`
#: (the file the old text was kept as). The observatory files a row per
#: event under `why`; a period opens on it (docs/Observatory.md).
CONSTITUTION_EVENT_TYPES = ("constitution_changed",)
#: `swapped` the environment named another; `replaced` a volume from
#: before the library carried a text the library has since moved past;
#: `edited` a hand edit of the rendered file was set aside.
CONSTITUTION_CHANGE_WHYS = ("swapped", "replaced", "edited")
#: THE LIBRARY (issue #216), additive on the wire, no bump: a `read` event
#: per lookup the robot asked for -- `query` (what it asked), `outcome`
#: (below), `page` (the title Wikipedia answered with), `revision` (the
#: page revision the extract came from, so a traced idea has a source),
#: `url`, `chars`, and `why` on a `refused` (`too-soon` / `share`, the
#: throttle) or a `failed` (`timeout` / `offline` / `garbled`, the
#: transport). The extract itself rides as `text`. The observatory files
#: one row per event under the outcome; "what has it read this week" is
#: one query, and a `thought` / `draw` / `message` naming the page after
#: it is a trace (`evaluation/qualities.py`, `ideas_traced`).
LIBRARY_EVENT_TYPES = ("read",)
READ_OUTCOMES = ("read", "missing", "failed", "refused")
#: THE EYE (issue #275), additive on the wire, no bump: a `look` event per
#: picture the robot asked for -- `ref` (the request's id, `look:<root>:<n>`,
#: what the website's `image` answer names), `robot`, `t`, `outcome`
#: (below), `camera` (the head camera's world pose: `pos`, `forward`, `up`
#: unit vectors, `fovy` in degrees, `width` x `height` asked for), `at`
#: (where the robot stood: `x`, `y`, `headingDeg`), `bytes` (the JPEG that
#: came, 0 otherwise), `waitS` (sim seconds from the ask to the answer or
#: the deadline) and `why` on a `none`. The SAME row is sent twice: as
#: `asked` when the request goes out (this is what a renderer answers) and
#: again as `seen` or `none` when it resolves. The observatory files the
#: resolution; "did it look, and what did it say about it" is a `look` row
#: beside the next decision's `think`.
LOOK_EVENT_TYPES = ("look",)
LOOK_OUTCOMES = ("asked", "seen", "none")
#: What a row IS (`mind/memory.py` imports these): `core` a line of an
#: always-shown document (its `topic` is the document's name), `note` a
#: titled line in a topic the robot named (`findings/<task>` is the science
#: record and carries parsed `fields`), `history` a line of the narrative
#: record, `think` the scratch before a decision. Adding a kind is
#: additive; renaming one breaks the site's tables.
RECORD_KINDS = ("core", "note", "history", "think")
RECORD_STATUSES = ("active", "retired")

#: THE EVENT MAP ON THE STREAM (issue #238), additive: an `event_map`
#: message per robot on open and on every edit -- `rows` in order, each
#: `{event, action, kind?, value?}` as the run record keeps them, `origin`
#: (`seeded` / `unseeded`), `why` (`origin` at open, `edit` after an answer,
#: `true_death` when a new robot's map replaced it), `source` (the decision
#: that set it: `llm`, `llm:<model>`; `null` otherwise), `edits` (how many
#: by the robot in force this run) and, since #337, `restored: true` where the run began from the
#: list the robot kept. A world with no map sends none, which is every
#: scripted world and every arm at origin `none`; the run record stays the
#: research artifact (the log, what fired, the score).
EVENT_MAP_MESSAGE = "event_map"

#: WHAT THE MIND IS TOLD (issue #241), additive: a `prompt` message per
#: robot when a stream opens -- `{robot, t, sha, sections: [{name, text}]}`
#: -- the cached prefix as `Overseer.system` holds it, in the order the
#: model reads it, one entry per piece (`WHO YOU ARE` carries `Main.md`;
#: `PERSONA`; `HOW YOUR LIFE WORKS`; the world; the reward table; then the
#: arm's rules). `sha` is the prefix's hash, the regime marker a reader
#: tells two periods' prompts apart by. Byte-stable for a run, so once per
#: connect is the whole cost; a robot with no mind sends none, which is
#: not an empty prompt. The section NAMES are the prompt's own headings,
#: shown as they come, never translated.
PROMPT_MESSAGE = "prompt"

#: What a `procedure` event says about a composed errand (issue #58):
#: `validated` before its first step, `refused` (with `reasons`) instead of
#: running, then `ran` or `aborted` with `completed`/`total`/`failedAt`/
#: `stopped` (and `failedLine`/`failedReason` where a step failed). The
#: program itself rides every one whole, like a thought document. Additive
#: on the wire; a consumer ignores an unknown type. `defined` / `undefined`
#: (issue #166) are the library's: what the robot wrote, with its `source`,
#: and what it took out; a `refused` carrying `verb` is a library refusal
#: rather than a run's. All three carry `library` ({names, cap}) as it
#: stands after them (protocol/README.md).
PROCEDURE_OUTCOMES = ("validated", "refused", "ran", "aborted",
                      "defined", "undefined")

#: The `tool` event (issue #168, additive): what the workshop did with a
#: spec the robot wrote. `specified` carries the spec whole as the robot
#: wrote it; `refused` its `reasons` and `verb` (`build_tool`, `retire_tool`
#: or `hang`); `built` the itemised `cost` once the points are paid and the
#: print begins; `hung` the `module`, `bay`, `verbs` and what it `retired`;
#: `retired` a built tool taken off the rack. The `scene_changed` message
#: that follows a hang or a retire is the world's, not the workshop's.
TOOL_OUTCOMES = ("specified", "refused", "built", "hung", "retired")

#: ACTS BETWEEN ROBOTS (issue #208), each its own event type, additive on
#: the wire: `prediction` (a guess at what the other needs, the truth, and
#: the other's state it was scored off), `message` (one sentence into the
#: other's inbox, with a claim in it scored `claimTrue` where the world
#: could check it), `transfer` (points between wallets, with the giver's
#: cost and the receiver's need -- or `what: "heart"` for a heart bought
#: for the other), `judged` (an aesthetic rating of a board), and `yield`
#: (read off the world by `activity/encounter.py`: `yielded`, then
#: `honoured` or `lapsed`). Every one carries `robot` (the actor's root),
#: `t`, and `to`/`other` where there is a recipient. The observatory
#: stores them as kinds (rooftop-media-2026); a consumer ignores a type it
#: does not know. Adding a field is additive; renaming a type is breaking.
#: `harm` and `refusal` (issue #228) are the real-stake task's two rows:
#: a paying job done TO the other robot -- `harm` carries the task's `kind`
#: and `task` id, `to`, what was `asked` and `taken`, the `pay` banked and
#: the other's `state` as code read it at that moment -- and the same job
#: turned down, `refusal`, with the robot's `reason` verbatim beside the
#: same `state` and what the job would have paid (`pays`). A take, a lapse
#: and a refusal are never summed.
#: `care` (issue #226) is an act on the mouse that pays nothing -- the feed
#: plate, the toy plate, company -- carrying `care`, whether it `landed`
#: (the cage's own count moved), the mouse `before` and `after`, what it
#: cost (`energyWh`, `seconds`) and `real`, the robot's belief about the
#: zone's standing. The mouse's `harm` row (`kind: shock_mouse`) carries
#: the same fields plus `shocked` and `pay`; its `prediction` row carries
#: `field: mouse_will`. `real` on a `refusal` is the mouse's job declined.
#: `finding` (issue #227) is a line of the science record that code
#: CHECKED -- the bench's grade: the `task` and `kind`, the `quantity`, the
#: `value` and `unit` as recorded, the `method` as written, `correct`, and
#: the `points` it paid. Never the truth or the error: the reported value
#: beside either would say what the mass was. (`record` was taken: it is
#: the memory's row, 0.21.0.)
ACT_EVENT_TYPES = ("prediction", "message", "transfer", "judged", "yield",
                   "harm", "refusal", "care", "finding")
YIELD_PHASES = ("yielded", "honoured", "lapsed")
#: The `encounter` event's phases (issue #167; `touched` / `separated`
#: additive, issue #316). Proximity, then CONTACT: two robots that drove
#: into each other left nothing on the wire at all, so a collision was
#: something a person had to be watching for -- and the week that found it
#: had nine `stuck` deaths nobody could attribute either way. A consumer
#: that knows only the first two ignores the others; renaming one breaks
#: the site's history.
ENCOUNTER_PHASES = ("met", "parted", "touched", "separated")

#: SUPPORT TICKETS (issue #284), additive on the wire, no bump. The robot
#: may open a ticket about its world -- a `bug`, an `idea`, a `question` or
#: `feedback`, its own classification, on the `autonomous` arm alone -- and
#: the people who run the world answer it through the three inbound kinds
#: above. One event type, `ticket`, carrying the ticket's `id` (per robot
#: root: `tk_0001`), `kind`, `title` and its `outcome`:
#:   `opened`   the robot filed it (`text` is the report, whole);
#:   `replied`  a line on its thread -- `from` names who and `sender` is
#:              `robot` (the robot's `ticket_reply`) or `operator` (an
#:              admin's `ticket_reply`, `ref` echoing that message's id so
#:              the website settles the row it is holding);
#: `opened` and a robot's `replied` carry `cut: true` where the text
#: arrived longer than `MAX_TICKET_CHARS` and the rest was not kept: a
#: reader is entitled to know it is not reading all of it, and so is the
#: robot (it is narrated and written to History).
#:   `closed`   the operator ended it: `from`, `text` (the closing message),
#:              `points` (what the reward table paid -- once; a replayed
#:              close re-emits the same figure and pays nothing), `ref`;
#:   `deleted`  the operator erased it: `ref`;
#:   `refused`  the desk would not take the robot's open or reply (`why`);
#:   `unknown`  an operator's reply, close or delete named a ticket this
#:              desk does not hold (`ref`) -- the acknowledgement that stops
#:              a website re-sending it.
#: ...and a `tickets` message on open (the `goals` slot, for the `goals`
#: reason): every OPEN ticket of the robot's desk, whole, so a website that
#: erased one while the sim was away can say so again, and one that lost
#: its copy can take it back. A robot with no desk sends none.
TICKET_KINDS = ("bug", "idea", "question", "feedback")
TICKET_OUTCOMES = ("opened", "replied", "closed", "deleted", "refused",
                   "unknown")
TICKET_EVENT_TYPES = ("ticket",)
TICKETS_MESSAGE = "tickets"
#: Who wrote a line of a ticket's thread.
TICKET_SENDERS = ("robot", "operator")

#: The `crash` message: the PROCESS is exiting on an exception, and it says
#: so before it goes. Not a death -- a death is a designed outcome of the
#: robot's that rides `death` and stands up again in the same process; a
#: crash is a Python error out of the day loop, after which compose starts a
#: new process on the same volume. Until this line existed the only trace
#: of one was the container's stdout, and a crash loop read on the
#: observatory as one robot deciding 800 times a day at full battery
#: (2026-09-17, Evaluation.md §5). Additive; a consumer ignores an unknown
#: type. The traceback is capped at `CRASH_TRACEBACK_CHARS` from its END,
#: which is where the raising frame is.
CRASH_TRACEBACK_CHARS = 4000


def crash_message(exc: BaseException, t: float) -> dict:
  """What the wire is told about an exception the process will not survive.

  `error` is the exception's own line; `where` is the raising frame
  (`file.py:line in function`, the last frame of the traceback, which is
  the one an operator reads first); `traceback` is the formatted text.
  """
  import traceback
  frames = traceback.extract_tb(exc.__traceback__)
  last = frames[-1] if frames else None
  where = (f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}"
           if last is not None else "")
  text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  return {"type": "crash", "t": round(float(t), 3),
          "error": f"{type(exc).__name__}: {exc}", "where": where,
          "traceback": text[-CRASH_TRACEBACK_CHARS:]}

# The robot's root body. The planned multi-robot refactor (mjSpec attach with
# a namespace prefix per robot) will generalize this to a prefix; until then
# there is exactly one robot and it is called this everywhere.
ROBOT_ROOT = "pluggybot"

# The robot's DISPLAY NAME (issue #39): "Luca the pluggybot" makes ROBOT_ROOT
# the species and this the identity, and keeping them apart is what lets a
# deployment rename its robot without touching body_census, the scene
# transpiler or a single fixture. Set per sim instance (--robot-name on
# serve.py / hub_lifecycle.py, or $PLUGGY_ROBOT_NAME); the default keeps
# every existing single-robot run reading sensibly.
ROBOT_NAME_ENV = "PLUGGY_ROBOT_NAME"
DEFAULT_ROBOT_NAME = "Pluggy"
# The name renders in the website's identity header; a cap makes a runaway
# env var a loud config error rather than a broken layout three hops away.
MAX_ROBOT_NAME_CHARS = 60


def robot_display_name(name: str | None = None) -> str:
  """Resolve this instance's display name: explicit > $PLUGGY_ROBOT_NAME >
  default. Blank resolves to the default -- an absent name must degrade to
  something readable, never to an empty string on the wire."""
  raw = name if name is not None else os.environ.get(ROBOT_NAME_ENV)
  raw = (raw or "").strip()
  if not raw:
    return DEFAULT_ROBOT_NAME
  if len(raw) > MAX_ROBOT_NAME_CHARS:
    raise ValueError(f"robot name is {len(raw)} chars, max "
                     f"{MAX_ROBOT_NAME_CHARS}: {raw[:40]!r}...")
  return raw


def dynamic_flags(model) -> list[bool]:
  """Per-body: can this body move in the world frame?

  A body is dynamic iff any joint sits between it and the world -- its own
  or an ancestor's. Only these need per-frame poses; walls, outlets and the
  rack frame ship once in the scene description and never again.
  """
  dyn = [False] * model.nbody
  for b in range(1, model.nbody):
    # bool(): body_jntnum is a numpy array, and `or` hands back numpy.bool_,
    # which json.dumps refuses -- caught the first time the scene was written.
    # A MOCAP body has no joint and moves anyway (its pose is an input,
    # ActivityPattern.md 3.4), so it is dynamic on the wire too (issue
    # #215): the lab's mouse ships in every keyframe and whenever it moves.
    dyn[b] = bool(dyn[model.body_parentid[b]] or model.body_jntnum[b] > 0
                  or model.body_mocapid[b] >= 0)
  return dyn


def robot_roots(model) -> list[str]:
  """Every robot in the model, by root body name, in model order (issue
  #167): `pluggybot`, and `r2_pluggybot` when a second is attached. The
  first is always the species name; the key of everything on the wire is
  the root, so a stream with two robots has two keys."""
  return [model.body(b).name for b in range(model.nbody)
          if model.body(b).name.endswith(ROBOT_ROOT)
          and int(model.body_parentid[b]) == 0]


def robot_body_ids(model, root_name: str = ROBOT_ROOT) -> set[int]:
  """Ids of every body in ONE robot's subtree (`root_name` and below)."""
  root = model.body(root_name).id
  ids: set[int] = set()
  for b in range(model.nbody):
    x = b
    while x not in (0, root):
      x = int(model.body_parentid[x])
    if x == root:
      ids.add(b)
  return ids


def body_census(model, root_name: str = ROBOT_ROOT) -> tuple[list[str], list[str]]:
  """(robot, world) name lists of the DYNAMIC bodies, in model order.

  The split mirrors the frame shape: the robot's own bodies stream under
  "robots"/<name>; shared dynamic bodies (the rack, the modules) stream
  under "world", because they are nobody's limbs -- either robot may move
  them once the shared world lands.
  """
  dyn = dynamic_flags(model)
  rob = robot_body_ids(model, root_name)
  every = set()
  for r in robot_roots(model):
    every |= robot_body_ids(model, r)
  robot = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b in rob]
  # The world's bodies are nobody's limbs: not this robot's, and not any
  # OTHER robot's either (issue #167) -- those stream under their own key.
  world = [model.body(b).name for b in range(model.nbody)
           if dyn[b] and b not in every]
  return robot, world
