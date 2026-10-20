# Hands — the humanoid's manipulation vocabulary

The verbs a mind writes a procedure in to work with the humanoid's hands
(issue #525): the verb set, what each says and how it fails, the language
around it, the capability check that gates it, and the migration from
#493's practice verbs. Every demo after #493 (#465) and #487's skill
library stand on this vocabulary.

Read alongside:
- `docs/SimNotes.md`, "The force move (issue #525)" -- how each number
  here was measured. This doc says what the constraint is.
- `docs/Challenges.md` §10 -- the three new checks' graders.
- `docs/Evaluation.md` §4 -- practice first, and the arms the vocabulary
  replaces.
- `procedure/steps.py` (`HANDS_VERBS`, `hand_facts`) and
  `humanoid/reach.py` (`Reach.pull`): the code. The verbs' own text is what
  a mind is shown; this doc does not repeat it.

---

## 1. What was studied

**Robot motion primitives.** Industrial controllers move a tool by joint
(MoveJ), along a line (MoveL), an arc (MoveC) or a path of points blended
into one motion (MoveP). Contact work adds three kinds of force control: a
GUARDED move (a position move that stops on a force), a FORCE MODE (UR's:
the tool pushes with a set force along chosen axes and is position
controlled along the others), and HYBRID position/force control (Raibert
and Craig 1981), which splits the task frame into the directions a contact
constrains and the ones it leaves free -- Mason's (1981) natural and
artificial constraints. Impedance control (Hogan 1985) makes the arm a
spring; admittance closes a loop on a measured force and moves the
command. Kröger, Finkemeyer and Wahl's manipulation primitives are a hybrid
move plus a stop condition, the unit a program is built from. Opening doors
and drawers without a model of them (Karayiannidis et al. 2013; Jain and
Kemp's equilibrium point control) comes down to letting the mechanism's
constraint steer a compliant hand and reading the way it went.

**Human manipulation.** Cutkosky (1989) and Feix et al.'s GRASP (2016) sort
grasps by power against precision and by which surfaces of the hand
oppose; a two-jaw gripper has one, the pad-against-pad pinch. Bullock, Ma
and Dollar (2013) sort manipulation by contact, whether it is prehensile,
whether the hand moves, and whether anything moves within the hand -- a
light grip that lets a held object turn between the pads is "within hand,
motion at contact". Non-prehensile work (Mason; Lynch: push, slide, pivot,
tilt) is what a hand does without grasping. Two hands together (Guiard
1987; Krebs and Asfour 2022) usually divide into a hand that holds a frame
and a hand that works in it.

**This body.** Two Unitree Dex1-1 two-jaw grippers (90 mm stroke, 5-120 N a
pad, adjustable: `humanoid/parts.py`), each on a six-joint arm whose grip
reaches 0.58 m from a shoulder 1.15 m up; the hand's force read off its
arm's motor currents through the robot's model of itself (within 0.5 N of
the truth held still: SimNotes); the grip's own word -- its jaws' gap and
squeeze; the head camera's tags (a face's middle, which way it looks, which
way its edges run); the D435's depth (`measure`); and a posture -- the
pelvis's height and the trunk's pitch -- that the standing policy holds.

## 2. The principles

1. **Two kinds of move, and only two.** A POSITION move puts the hand where
   it is told (`reach`, `move_hand`, `withdraw`). A FORCE move pushes with a
   force along a way and lets what the hand holds take it (`pull`): the one
   thing no composition of position moves can do, because its loop closes
   on what the hand feels at the physics rate, inside the verb. A
   procedure's loop runs between verbs (100 rounds, 200 verb calls at
   most); a hinge's steer needs every step.
2. **Every report is a sensor, every failure readable.** What a verb did
   rides `read(...)`, and `if verb(...)` reads its failure without ending
   the procedure.
3. **No verb knows a job** (Ben's call 2 on #493). No verb is a lid, a
   drawer, a door or a swing; each is generic, for any object.
4. **A verb earns its place by doing what no composition can.** A swing is
   a path and the maths; a re-aimed pull is two pulls and a `read`. Those
   are skills a robot writes for itself (#487, demo 3).
5. **The posture is the body's.** Every hand verb chooses the posture its
   whole motion is followed from; a procedure never names one.

## 3. The verbs

| verb | what it does | its report | how it fails | built from |
|---|---|---|---|---|
| `spot(tag)` | stands still and looks for a tag -- as it stands, then bowing lower (`SPOT_GAZES`) until it sees it -- and stands up again | `spot.x/y/z` (the face's middle), `spot.nx..nz` (into the thing it is on), `spot.ex..ez` (along an edge), `spot.range` | not in view from where it stands | the head camera's tag detector (`rack.tags`), `Body.gaze_routine` |
| `measure(x, y)` | the depth camera's look at what stands at (x, y) | `object.top/near/far/left/right` | no flat top seen | the D435's cloud (`perception/box.py`) |
| `reach(hand, x, y, z, fingers, jaws, stiffness)` | the grip to a point, turned as asked or as the arm finds; out to a point short of it, then in along the fingers | `hand.x/y/z`, `hand.fx..fz` | out of reach (and whether the other hand could), turned as asked or not; stopped short | the arm's IK on the robot's model of itself, a planned posture |
| `via(hand, x, y, z, turn)` | adds a point to the hand's next move; nothing moves | -- | a path holds 40 points | -- |
| `move_hand(hand, x, y, z, speed, stiffness, turn)` | a hand that is out carried along a line, or through its `via` points, as ONE motion from one posture | as `reach` | out of reach (planned or partway), held back | `Reach.carry`, `path_for` |
| `pull(hand, dx, dy, dz, force, distance, speed, give)` | THE FORCE MOVE (§4) | `pull.moved`, `pull.force` (the hardest along its way, as felt), `pull.dx..dz` (the way the hand was going at its end) | ok whenever it pulled; why it stopped (distance, held fast, let go, out of reach, feet moved) is the record's | `Reach.pull` |
| `close_hand(hand, force)` | shuts the gripper with at most `force` a pad, 5-120 N | `hand.held`, `hand.gap` | closed on nothing it holds | the Dex1-1's torque limit, its encoder and current |
| `open_hand(hand)`, `withdraw(hand)` | open the gripper; bring the hand back along its way in, and the arm home | -- | -- | -- |

Beside them stay `wait` and `move(axis, target)` -- the motor level, one arm
joint walked to a setpoint -- and the sensors `time` and each hand's place,
force, `held` and `gap`.

**The language** is the served robots' (`procedure/lang.py`), with two
switches on wherever there are hands (`steps.hand_facts`): a verb may be an
`if`'s test (`WorldFacts.verb_tests`), and an expression may call `sin`,
`cos`, `atan2` and `sqrt` (`WorldFacts.math`). Its caps are unchanged:
1,500 characters, nesting 4, `for` over a literal count of at most 100,
`while` stopped at 100 rounds, 200 verb calls, the budget's seconds.

## 4. The force move

`pull(hand, dx, dy, dz, force, distance, speed, give)`, in
`Reach._pulled` every command:

- **Along its way, the force is CLOSED on what the hand feels**
  (admittance): the command walks at `speed` while the hand pushes under
  half `force` along the way, slows to a stand as it reaches it, and backs
  off past it -- `speed * clip(2 (force - along) / force, -1, 1)`, the
  reading smoothed over `ADMIT_SMOOTH_S` -- and leads the hand by
  `LEAD_MAX_M` at most. ⚠ Never the arm's spring as a force cap: its model
  put 15 N on a drawer's stop for a 5 N limit; the hand's reading was
  within 0.5 N.
- **Across it, GIVING (`give` 1), the hand goes where what it holds pushes
  it**: the command follows the hand across the way, no faster than
  `ACROSS_SPEED` times its speed -- hybrid force and position, the hinge
  or the runners steering. ⚠ Unbounded, a lid let go of its weight fell;
  past a dead band only, the hand lagged a drop handle's flip.
- **HOLDING (`give` 0) it keeps its line and its turn**: a set-down.
  ⚠ Giving, a cube set down slid 13 cm along what it touched.
- **Its turn gives to what it holds** (`GIVE_RATE` of the gap a command),
  **but never about the jaws' own line**: what a two-jaw grip holds turns
  about that line between the pads (a light grip lets it), and a lid's
  weight bent the wrist about it, the give followed 70 deg down and the arm
  ran out of reach 46-54 deg up the lid on 4 chests of 8. Held there, 7 of
  8 opened past 57. ⚠ Held about every axis, the pads dragged a drop
  handle's knob sideways through its pin and turned the robot 22 deg on
  its feet.
- **Its posture** is the one from which the hand reaches the way's end,
  taken before it moves; going on where the arm misses its command, it
  takes ANOTHER posture from the arm as it is (`posture_for(onward=)`):
  ⚠ an arm stood at its elbow's stop 2 cm short for 9 s while a fresh seed
  of the same posture "reached".
- **It stops** at `distance` along its way ("distance"); held fast -- the
  hand moved under `FAST_M` in `FAST_S` pushing at least `FAST_SHARE` of
  its force; its grip lost ("let go"); out of reach; its feet moved; out
  of time. Then it holds where the hand is.
- **Its report** is how far it went, the hardest it pushed along its way
  as the hand felt it, and the way the hand was going over its last
  `TRAIL_M` -- what the mechanism allowed, which a procedure re-aims by.

A drawer is one pull straight out; a lid one pull up and a little back,
the hinge steering the hand round the arc, and one down toward the body; a
door one pull toward the body and one along the way the hand went. None of
them needs the mechanism's geometry, and none was handed it.

**The grip's force** is the other half of compliance: `close_hand(hand,
force)` shuts with 5-120 N a pad (the Dex1-1's range), and `held` reads the
squeeze against half the force asked. ⚠ At 120 N the pads' torsional
friction (2.4 N*m) locked a door's lever in the jaws; at 15 N its tip
turns between them as the door swings.

## 5. The capability check

#493's gate pattern: before any mind flies the vocabulary, a procedure WE
wrote in it, for each generic job, flown at the job's set-outs and graded
by the job's own grader, the bar written first -- **a job passes at 6 of
its 8 set-outs** (`evaluation/hands_check.py`, `BAR`, `SET_OUTS`).

| job | grader | the set-out (`evaluation/counter.py`) | what its procedure knows |
|---|---|---|---|
| lid | `challenge/lid.py`: past 57 deg, set down, let go | #493's chests on the 0.72 m counter | the knob's cube from its tag; lift up and a little back, lower toward the body |
| drawer | `challenge/drawer.py`: 15 cm out, let go | a cabinet on the counter, its knob where a chest's is | pull straight out |
| door | `challenge/door.py`: 50 deg open, let go | a cupboard hinged on its left, its lever's tip 0.40-0.44 ahead | press the lever, let go, look: a door already open is done; else grip it lightly and pull toward the body, then on the hand's way |
| place | `challenge/place.py`: on the mat within 2.5 cm, upright, set down, let go | a cube 0.29-0.33 ahead on a 6 cm board, the mat 6-8 cm further | grip it from above-front; lift, carry, set down holding the line; a carry refused is set back and gripped again |

The procedures and what each knows that a mind would have to find out are
in `hands_check.py`; `scripts/hands_check.py --source JOB` prints one.

**The result** (`scripts/hands_check.py --jobs 3 --into DIR`, the commit
in the PR): RESULTS

**The set-outs were moved for the body, never for the procedure**: each
move is at its constant with the measurement behind it. The door hung on
its right swung into the left arm holding its lever; its lever's tip 0.33
ahead was out of reach once the door stood ajar; the cube's carry 6-10 cm
ahead was out of every posture from 0.34-0.38 ahead and 12 cm up.

**What it cannot show.** The check proves the vocabulary EXPRESSES each
job, not that a mind finds the expression; it is one body, one hand at a
time, on a counter at 0.72 m.

## 6. The migration

**#493's practice verbs:**

| #493 | #525 | |
|---|---|---|
| `spot` | kept | bows lower until it sees; reports the edges' way |
| `measure` | kept | an arm may still go without it (`--no-measure`) |
| `reach` | kept | |
| `move_hand` | kept, MERGED with `follow` | goes through the `via` points first |
| `via` | kept | |
| `follow` | DROPPED | `move_hand` carries a path |
| `pull` | REBUILT as the force move | `stiffness` and the turn dropped (the arm's gains are the move's; its turn gives); `give` added; `force` is a force it pushes with, never a condition it stops on |
| `close_hand` | kept, `force` added | |
| `open_hand`, `withdraw` | kept | |

The practice's arms (`--pull`, `--verb-tests`, `--path`) were the
vocabulary's stages; it is one vocabulary now, and a row says which it
flew (`run.vocabulary`), so #493's records stay readable and are never
pooled with these.

**The served robots' language is unchanged.** The quadruped's procedures
are `VERBS`, and `WorldFacts`' switches are off there (`verb_tests`,
`math`): its procedure rule says "no other calls"
(`overseer.PROCEDURE_HEAD`). Turning either on is a change to the deployed
prompt, so it is its own Observatory period (Observatory.md), decided
apart (Evaluation.md §2); nothing here asks for it, and the hands' verbs
belong to a body the served world does not have. The hands' verbs are a table of their own
(`HANDS_VERBS`) that the language reads only where a world's facts name
them, so a served robot's kept procedure naming a local `pull` still
compiles.

**The practice prompt** (`mind/practice.py`) is built off the facts: the
verbs' text, the sensors, the maths and the verb test are each shown
because the world has them, and none of its rules prescribes a job (a test
reads them). Stage 3 of #493 flies again on this vocabulary when Ben says.

## 7. What it does not do

- **Two hands at once.** Every verb is one hand's; a bimanual frame-and-work
  split waits for a job that needs it.
- **Motion within the hand** beyond a light grip's turn between the pads.
- **Contact without a grip** other than pushing with a pull (a negative
  way): no slide or pivot verb, nor yet a job needing one.
- **Its own feet.** The stance is where the robot was put; reach is the
  envelope from there (`humanoid_spike.py --envelope`).
