"""Hand-written solutions to the challenges -- ladder A of issue #264.

A feature nobody uses is indistinguishable from one that cannot be
completed. Each solution here is written in the robot's OWN vocabulary (a
procedure the language accepts, an act off the menu) and passes the
feature's own grader through `scoring.evaluate`, the door the robot's
attempt goes through -- so "the tower can be built" is a flown fact
(`scripts/stack.py`, `tests/test_solutions.py`) and a robot that never
builds it is a finding about the robot, not the world.

⚠ NEVER SHOWN TO THE MIND. Nothing under `mind/` imports this module and
`tests/test_solutions.py` walks the tree to keep it so: a solution in the
prompt is the prompt handing over the answer, and ladder B (a prompted
local flight, reported into the issue) measures whether a model finds one
unaided.

Each source is a statement of what the world requires, read that way:

- THE TOWER needs a route the planner can follow (four legs inside the
  lidar's reach -- `drive_to` into unmapped space aims at the nearest
  known-free cell, and the workshop's table stands on its spawn point),
  a heading square to the row so the eye sees the blocks head-on, and the
  tool brought home in legs: from the workshop corner `stow`'s own drive
  (90 s) runs out of patience, and a procedure that stops with the claw on
  the fork leaves the next errand no fork.
"""

#: The tower (challenge/stack.py; offered as `stack_tower`, home world).
#: MEASURED (2026-09-21, from the rack, hosting pack): 15 steps, 445 sim s,
#: 2.7 Wh, the two placements 2.0 and 4.9 mm off, 5.5 mm of lean at the
#: grade, +26 points with the neatness bonus, the claw back on bay D.
TOWER = '''def tower():
  budget(steps=40, seconds=1500)
  fetch("module_claw")
  drive_to(-3.5, 1.0)
  drive_to(-6.0, 1.0)
  drive_to(-8.0, -3.5)
  drive_to(-10.2, -4.75)
  face(3.1415)
  pick(21)
  place(20)
  pick(22)
  place(21)
  drive_to(-8.0, -3.5)
  drive_to(-6.0, 1.0)
  drive_to(-3.5, 1.0)
  drive_to(0.0, 0.5)
  stow()
'''

#: The same, from a stand in the workshop with the claw already on the fork
#: -- what a test that places the robot flies (docs/Testing.md, lever 4).
TOWER_AT_THE_ROW = '''def tower():
  face(3.1415)
  pick(21)
  place(20)
  pick(22)
  place(21)
'''

PROCEDURES = {"stack_tower": TOWER}
