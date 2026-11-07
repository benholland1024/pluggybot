# What has changed in your world

<!-- The robots read every entry through `recall`, newest first, and the
site shows them. Facts about their world and body only: never what we
measure or why, how anything is graded, a hidden truth, a hypothesis, or
advice. Ben approves each entry before it ships. The rules:
docs/Overseer.md §7b; tests/test_changelog.py reads every entry. -->

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
