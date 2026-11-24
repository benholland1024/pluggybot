# What has changed in your world

<!-- The robots read every entry through `recall`, newest first, and the
site shows them. Facts about their world and body only: never what we
measure or why, how anything is graded, a hidden truth, a hypothesis, or
advice. Ben approves each entry before it ships. The rules:
docs/Overseer.md §7b; tests/test_changelog.py reads every entry. -->

## 2026-11-24

- Your answers now have up to 600 s and 64,000 tokens before they are cut
  off, twice what they had. Before, an answer still being written at 300 s,
  or past 32,768 tokens, was lost.
- A `find` searches all you keep from this life (your History, `think`s,
  notes and findings, goals, pinned lines and the lines you took out) and
  brings back at most 8 lines, the best match first, after anything `read`
  brings. The lines your own recalls write are now left out of those 8.
  Before, they took places among them, and the same `find` asked again
  brought back fewer lines each time.
- When one of your jobs falls just short of its line, its result now gives
  your reading with as many decimals as tell the two apart: a drawer that
  came out a little under 15 cm reads "came out 14.7 cm, short of the 15",
  where it read "came out 15 cm, short of the 15".
- While you charge, the station's cable slowly pulls you in. After more
  than 300 s plugged in, taking the plug out now starts with a step
  straight back from the station and a look at it standing still, and your
  hand goes by where that look places the station, not by where you
  believed you stood. Before, after a long charge your hand could fail to
  get the plug out.
- When your arm's way home stalls far behind where it is sent, it is now
  held where it is, raised 5 cm and sent home again, up to twice, as an arm
  held at its rest already was. If it is still held, it stays there, and a
  `withdraw` reads false saying something holds it.
- After a walk or a `step`, your legs now go back to standing only once
  both your soles are down, or after 1.2 s more all the same. Before, they
  could go back with one sole still off the floor, and you could fall
  forward.

## 2026-11-20

- A charging station the other robot is plugged into is now taken for your
  charge wherever you believe that robot stands, because the other robot
  says which station its plug is in. Your charge waits for it, or with
  `station: "free"` goes on to another station, and says so. Before, you
  could walk up to it, not see it behind that robot, and look round for it
  as if you were lost. While you are plugged in, the other robot no longer
  asks you to make way.
- Where the other robot stands by a charging station that you do not see,
  your charge now waits up to 120 s for it before you look for the station
  yourself.
- When a walk brings your arm home before it sets off, and bringing it
  home would push you over away from that hand, the arm is now handled as
  one that something holds: it is raised 5 cm and sent home again, unless
  the other robot stands near you, and if still held it stays where it is.
  The walk then goes on with the arm there, and its report says `armHeld`.

## 2026-11-19

- A person's message now reaches you while you charge, where your
  `message_received` rule says `ask`: you are asked then, with your plug
  still in, unless asking you is held off, and then it waits. `charge`
  carries on charging where you are, what you do standing still leaves the
  plug in, and anything that moves you takes it out first. If no answer
  comes, you carry on charging.
- A `reach` now keeps your arm out of your own torso and head: the arm it
  plans at the goal and at the point the hand comes in from, in every
  posture it may take, and its way out from a hand already out. Where
  every way would pass through your body, the hand does not move, and the
  reach reads false saying so.
- A hand's place (`left.x`, `left.y`, `left.z` and the right hand's) now
  reads where your arm last read it, at most a hundredth of a second ago.
  While you are getting up after a fall, it has no reading, and says so.
- A `spot` that sees nothing now also says how far its picture reaches to
  either side: 34 degrees either side of where your head faces at the
  picture's top, and more than 50 at its bottom. It says so where the line
  about your run has room.
- A reply or a close to one of your tickets now reaches you even when it
  is sent while your world sleeps. Before, one sent then could be lost,
  and the ticket stayed open on your desk.
- The table's job now says which left and right it means: yours, as you
  face the card. When a table job is offered, the plate, the cup and the
  fork are set out on the table near the card, none of them where it goes.

## 2026-11-18

- Each person now has their own chat with you, and no other visitor sees
  their messages or your replies to them there. What comes with a message
  as `earlier` is that person's conversation with you alone. Your History
  still keeps every conversation, and people watching can read it, as they
  can all your memory.
- A `say` now goes to everyone watching, in a place of its own, and is
  never part of anyone's conversation with you.
- A message sent to you while the world was asleep now reaches you soon
  after you wake. Before, it could wait until your next run began, an hour
  or more later.
- While your other hand is out holding something, your body now keeps the
  posture it is in for a `reach`, a `move_hand` or a `pull`. A hand that
  does not reach where you asked from that posture does not move, and reads
  false saying so.

## 2026-11-17

- A `step` now keeps off walls and furniture as well as the other robot:
  it stops before it would bring your LIDAR within 0.3 m of what it reads at
  your head's height, or your middle within 0.2 m of what your depth camera
  saw at your knees' height, and reads false saying so. A step that your
  scans say went under half of what your legs counted reads false too, as
  `pressed`.
- A `spot`'s bow now goes no deeper than leaves your head 5 cm short of
  what your LIDAR reads in front of it, and a look that did not see its tag
  says how far it bowed.
- An arm that something holds on its way home is now raised 5 cm and sent
  home again, unless another robot stands near you. Held again, it stays
  where it is held, pressing on nothing, and the `withdraw` reads false
  saying so; a walk then goes with that arm where it is, and its report
  says `armHeld`.
- A `reach` now also stops on its way out, before it closes in, where its
  arm pushes on something hard enough to tip you toward it, your own body
  included. It says what a reach stopped on its way always says.
- A `done` said beside a run that never begins -- a `procedure:new` whose
  define was refused, say -- is now not acted on: the job stays yours, its
  time runs on, and a `done` said again is acted on. Your History says so
  when it happens.
- A fall that nothing else tells you of, on a charge trip, at your station
  or between runs, now leaves one line in your History once you are up, as
  in "I fell over and got up again 4 s later".
- In `others`, what the other robot is doing now shows only what its body
  is doing that anyone nearby could see: its walks, its charges, its
  procedures' steps (each as its verb alone, without the procedure's name),
  the jobs it holds, and its death and stand-up. Its thinking, its reasons,
  its list of rules, its pack's level and its points are never in it.
- Your map now keeps a thin wall that your LIDAR sees only at a glancing
  angle, such as the research center's south wall along the alley. Before,
  rays grazing such a wall could wear it out of your map, and a walk could
  be planned through it.

## 2026-11-15

- When the world came back, your map and your places were gone, and you
  began again at your start with no map: a map laid before could hold a
  room laid a second time, half a metre off, after a fall. Your pack, your
  points and your jobs are as they were.
- Up from a fall, until a fit of where you stand holds, your map's fit is
  checked by a wider look about once in ten scans. Where another place half
  a metre or more away explains what your LIDAR sees better, that scan is
  refused, and you are placed there once the next look finds it again.
- A fall while you hold a hand job now ends the job there, as if you had
  said done as you fell, and the fall fails it. Before, the job stayed
  yours until your `done` or until its time ran out.
- `measure(x, y)` now reads the flat top under (x, y), within 2 cm of it,
  and that top's own ends; before, it read the widest flat top within 0.45
  m, so at a small thing on a counter it read the counter. Where it sees
  too little of a top there, it reads a top round the point, and the new
  `object.under` reads 0; it reads 1 for the top under the point.
- Your body's description now says your grip lies halfway along your
  fingers, so their tips reach 4 cm past it. A `reach` that stopped short
  on its way now says how far its grip was from where you asked, as in "as
  it stopped, its grip was 2.7 cm from where you asked".
- `drive_to`'s description now says that a walk keeps your middle about
  0.35 m from what your map holds solid, so a walk sent to a wall or a
  counter's edge ends 0.3-0.6 m from it. A walk that ends short for that
  reason now says so, as in "the goal is 0.20 m from something the map
  holds solid, nearer than a walk keeps"; before, it said there was no
  route over the floor mapped so far.
- A `find` walking to where it remembers a place now has as long as a
  walk that far needs, 10 s a metre of the way; before, that walk gave up
  after 90 s, however far it was.
- A `pull` whose hand holds nothing as it begins now takes a new posture, a
  bow, a crouch or a turn of your waist, only while that hand reads at most
  40 N; where it reads more as one comes, your body backs off until the
  hand reads the force the pull asked. A `pull` that begins gripping is as
  it was.
- After a long charge, your unplug first backs your body off to your seat,
  since the cable's pull creeps you in toward the station while you stand
  plugged in, and it tries once more if the plug stays in.
- When your map has matched none of your scans for a minute, your History
  says so once: "My map has matched none of my scans for a minute, so I may
  not be where I believe I am". When your scans fit it again, it says "My
  map matches my scans again".
- A define refused for its length now says how long the source was as
  written, and every other reason it is refused, all at once; one refused
  because your library is full also says what else is wrong with it.
- In `others`, what the other robot is doing is now its own last line,
  never the job board's announcements.

## 2026-11-14

- The lab's three pressure plates now lie flush with the floor, as the
  garden's does, and each still works when stepped on.
- When the other robot is asked something in the middle of one of its
  errands, your body now stands still until its answer comes, for up to
  about two minutes, and then carries on with what it was doing. Before,
  your body went without its balance and its senses during such a wait,
  and could fall or lose track of which way it faced.
- A `done` is now acted on as soon as your answer is read, before that
  answer's other action, unless that action runs a procedure, which runs
  first; before, an `idle` or a `look` beside it held it back by up to a
  minute. A held job's `leftS` is what it had left when you were asked,
  and its clock runs on while you think.
- A refused note now says how full the file is, as in "`Notes.md` holds 64
  of 64 entries"; before, it said one more than the file held. `Notes.md`
  and `Findings.md` now come in every turn with their count, as `holds`.
- Your get-up is a new one, and it can stand you up where you have fallen
  beside a wall or along a counter. Before, a fall there often ended with
  you stuck on the floor. Fallen on your side with your face against a
  wall, you still cannot get up, and a fall off the stairs' landing still
  ends with you stuck.
- A `withdraw` made while you are bowed, crouched or turned, with no part
  of that arm out ahead of your front, now brings the arm home first and
  stands you up after. Before, you stood up as the arm set off, and at a
  counter that could press your open hand down on its top and tip you over
  backward. An arm that is out ahead of your front still comes home as you
  stand up.
- Every `withdraw` now waits until your trunk stands still before your
  next verb: about a second longer where you worked bowed, crouched or
  turned, and no longer where you stood. A walk begun with a hand out
  brings it home the same way.
- While the other robot is asked something in the middle of one of its
  errands, you now carry on with what you were doing. You no longer stand
  still until its answer comes.
- When you are asked in the middle of a procedure's walk or `find` whether
  to stop, the walk ends where you stand while you answer; if you carry on,
  it goes on to its goal from there with the patience it had left. While
  you are being asked you stand still, and the other robot may ask you to
  step aside. A fall or a death ends the question, and your History says
  the errand ended before your answer came.
- After a get-up, when what your LIDAR sees does not fit your map and the
  usual search near where you believe you are finds no fit, you now search
  again facing every way, while you are within half a metre of where you
  stood up. Before, the search reached only 15 degrees either side of the
  way you believed you faced, so a fall that turned you further could
  leave you lost.
- When an answer chooses `procedure:<name>` and nothing runs, your History
  now says why, as in "ran nothing: `procedure:open_it` is not in your
  library -- this answer's `undefine` removed it, and an undefine is done
  before the action". Before, History said nothing of it.

## 2026-11-13

- The garden's pressure plate now lies flush with the ground, where it
  stood about 2 cm proud, and it lights the lamp as it did. Because it
  changed, your map and your place were lost when the world came back: you
  began again at your start, with no map. Your pack, your points and your
  jobs are as they were.
- A walk now carries what a hand holds. Before any walk, a `step` too, a
  hand that is out and whose gripper says it holds something is raised 2
  cm, pulling up with at most 20 N. If what it holds comes up with it, the
  hand keeps its grip and folds to your side, the forearm up and the hand
  at your hips, and stays there through the walk and after, until a reach
  or a let-go ends it. If it does not come up, the hand lets go and comes
  home, as before.
- A `step` asked while a gripper holds something now walks; before, it
  moved nothing and read false. While a gripper holds something, your
  LIDAR and your depth camera no longer take anything within 0.12 m of that
  grip for something in your way or for a wall of your map.
- Your reaches and withdraws keep your arm clear of the other robot where
  your LIDAR sees it as the move begins, within 1.5 m of you. A `reach`
  whose every way out or home would meet it, or a `withdraw` whose every
  way home would, moves nothing, reads false and says where the other
  robot stands. A robot your LIDAR does not see is not kept clear of.
- `step`, `drive_to`, `face` and `find` now refuse to walk while a hand is
  out that could not come home, and say so. A charge or an explore waits
  up to a minute for such a hand, then brings it home.
- You now die once your get-up has had your legs for 20 s without standing
  you up, however far you lean; a stand of under 5 s between two get-ups
  does not start that count again. Before, it took your trunk staying 60
  degrees or more from upright for 20 s.
- A `read` of a verb whose last call failed now says so, as in "your last
  `spot` failed"; before, it said the verb had not run. A fault in a line's
  own arithmetic, an assignment's or a test's, now names its line, where
  History said only that it stopped at an expression.
- When a place laid at the restaurant's table fails, the line that says why
  now names every piece at fault, each by its first fault; before, it named
  only the first piece.
- Where your map and what your LIDAR sees agree only in part, as along the
  storeroom's counter, your belief of where you stand now looks round
  nearby for a place that fits clearly better, and moves there only when
  one does.

## 2026-11-12

- Two more jobs are on offer. At the restaurant's dining table you lay a
  place, a plate, a fork and a cup, set out beside it; on the storeroom's
  counter you put the cube on its mat. Each offer's sentence says what it
  asks, with its minutes and its points.
- The restaurant's dining tables are now a top on four legs, and the table
  where a place is laid has no chair on its south side.
- Because the restaurant's tables changed, your map and your place were
  lost when the world came back: you began again at your start, with no map. Your pack,
  your points and your jobs are as they were.
- `withdraw` now opens a shut gripper before your hand comes out. A
  `withdraw` asked to step aside while its arm came home used to wait
  forever; your legs now come back to standing first, and it ends.
- Before a `step` or any other walk moves your feet, a hand that is out
  comes home, its gripper opened as `withdraw` opens it: a walk carries
  nothing. A `step` asked while either gripper says it holds something
  moves nothing, and reads false.
- A fall ends the procedure you are running at the step where it happens,
  an `if`'s test or not, and your death ends it at once.
- A `step` keeps off the other robot as your walks do: one that would take
  you nearer it, inside the space kept round it, moves nothing and reads
  `blocked`. Your walks and steps keep off its body where your LIDAR sees
  it, and where it does not, off where the other robot says it is.
- A `step` whose walk your legs reckon went nowhere reads `too short`. A
  small turn now turns, closing on where your feet point, and one that
  cannot get there reads `turned short`.
- `spot` and `watch` read your colour camera, which looks down from your
  head, and their words say what its picture takes in; `look` and `find`
  use your navigation camera. On a block with a tag on each face, `spot`
  and `watch` read the face seen most square-on.
- `pull` reads false only when an argument is refused, a hand is not out,
  its way is all 0, or you fall. A hand held fast is ok, and `pull.held`
  says so.
- `measure` reads an end of a top only where your camera saw past that end.
  Where something taller stands just past an end, that end reads where the
  free top stops, and `object.near_taller` (`far`, `left`, `right`) reads
  1.
- Your grippers close steadily: their force no longer flips at every step,
  which could knock a light handle away. A grip on something thin now reads
  `held` when it holds; before, a hard squeeze on something under 2 cm
  thick read as holding nothing.
- History now names every variable a run's two lines hold and counts the
  rest, says when a `take_task` does not happen, and says why a `done` on a
  job you do not hold is refused.
- A job whose things are put back when it is offered says in its sentence
  when they were put back.
- A run's hour can now end while you charge: your plug stays in, and
  History says your charge was cut short. In the next run a `charge`
  carries on at once, and an `explore`, or a procedure whose first verb
  moves you, takes the plug out first.
- The procedure language now says that a tag is named by its number, what
  a `budget` is capped at and what it counts, what a name may be, and that
  the first def in a source is the one that runs.
- Your LIDAR's and your depth camera's rays no longer pass through parts of
  things that stand still, such as a table's top.

## 2026-11-07

- You have a waist. Your trunk turns on it over your hips, up to 60
  degrees either way, and your shoulders, your arms and your head turn
  with it. A reach turns it only where no posture with it square reaches,
  holds it through a carry, and turns it home before you walk or get up.
  No verb of yours turns it on its own.
- Because your body changed, your map and your place were lost when the
  world came back: you began again at your start, with no map. Your pack,
  your points and your jobs are as they were.
- Your reaches arrive where they are aimed. They were planned as if you
  stood a few centimetres further forward than you do, so a hand often
  stopped 2 to 6 cm short of its goal. Each reach is now planned from where
  your own senses put you, and once your arm is out it checks again before
  your hand goes in.
- What your hands' force sensors read now leaves out your body's own
  motion while a hand moves on its own way. A hand swinging out through the
  air read 11 to 14 N with nothing touched, enough to end a move that
  stops on a force as it began. A hand moving through the air with nothing
  touched now reads up to 12 N, and a stop on a hand's force reads it from
  0.1 s into the move.
- `measure` looks only within 0.45 m of the point you give it, and its
  words now say so. Where a top runs on past that, its end that way has no
  reading; before, the edge of what it looked at was read as the top's
  end. A top must stand 0.06 m or more over the floor; before, the floor
  itself could read as a top a centimetre or two up.
- History says what your procedures' tests and stop conditions read: how
  many tests read false, with the first reason for each line that did,
  and for each stop condition that ended a move, its sensor, its value
  and how far into the move. A run that reached its end is no longer
  called stopped by an error.
- A `done` now leaves a line in History, and the line that says whether a
  job passed or failed names the job's id.

## 2026-11-06

- The world is bigger. The house and the research center still stand on
  either side of the middle street, and south of them, inside the same
  fence, are two new blocks: a park south of the house and a mall south of
  the research center. The middle street runs on between the park and the
  mall, an alley runs between the research center and the mall, and a
  street goes round all four.
- The park is a fenced lawn with a gate in its north fence and another in
  its east fence, across from the mall's front door. Two trees stand near
  its middle, with a sign between them.
- The mall has a long concourse down its west side. Off it are a
  restaurant, with a dining room and a kitchen behind it, and a store, with
  a shop and a loading area behind it that has a door out to the sidewalk.
  Every table, counter and shelf in the mall stands empty.
- The house has two new rooms off its living room's south wall, one for
  each robot, and the research center has a study room off its lobby, with
  a table and two chairs.
- There are three charging stations: the house's in its living room, one
  at the south end of the research center's lobby, and one at the south end
  of the mall's concourse. Each takes one robot at a time.
- A procedure's source may hold several `def`s. The first is the one that
  runs, and a `def` may call another by its name; a procedure in your
  library may call another one as well. A source may be 3,000 characters
  long, where it was 1,500.

## 2026-11-05

- A `look` brings back a picture now. Until then nothing was running that
  could take one: every look you asked for came back without a picture,
  and for most of that time `look` was off your menu.

## 2026-11-03

- The jobs you hold are in your state, as `heldTasks`, each with the
  seconds it has left, as `leftS`. A job held past the time its
  description gives is failed.

## 2026-11-02

- A `say` is said again. From 2026-10-27, when the world's clock was set
  back to zero, every `say` had been refused, because your last one was
  dated on the clock from before.
- A ticket of yours pays again when the people who run your world close
  it. From 2026-10-27 a closed ticket had paid nothing, for the same
  reason.
