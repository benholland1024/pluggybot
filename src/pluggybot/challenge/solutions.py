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

- THE TOWER is six verbs once `pick` can go to a cube it cannot see: the
  route to the workshop in legs inside the lidar's reach (`drive_to` into
  unmapped space aims at the nearest known-free cell -- a single leg from
  the rack stalled in the hall at 41 s -- and the workshop's table stands
  on its spawn point) and a stand on the room's open side, square to the
  row, are `lifecycle.zone_route` and `steps.prop_stand`'s, on `fetch`'s
  terms. The errand around it hangs the claw back from the workshop
  corner (MEASURED, once `place` ended in the driving configuration --
  before that the front-stop reflex stalled the return).
- THE BENCH needs a tare --
  the lift's force with the empty claw at the height the cube will be
  read at -- a few readings averaged against `axes.LOAD_NOISE_N`, the cube
  set back down, the arm tucked (`move("arm", 0)`) before the drive home
  -- out, the claw's body sits in the lidar's front-stop cone and the
  reflex backs the robot away from itself -- and the road home driven in
  legs (the lab is 30 m of street away; the tower's auto-stow was measured
  from the workshop, not from there). The finding itself is the
  mind's to write (`record`, off the locals History shows it); the script
  and the test write it from `mass` the way a mind would.
- THE MOUSE has its program in code already (`lifecycle.cage_program`, the
  `care` action's errand); its solution is that errand flown, and what it
  needed was the route's first leg moved off the garden doorway.
"""

#: The tower (challenge/stack.py; offered as `stack_tower`, home world) --
#: the six lines a model wrote on ladder B's third day, verbatim. `pick`
#: goes to where the house set a cube out when it cannot see it (its
#: `prop_stand`), and the errand around the procedure hangs the claw back.
#: MEASURED (2026-09-21, from the rack, hosting pack): 489 sim s, 2.7 Wh,
#: placements 2.6 and 4.7 mm off, 5.3 mm of lean at the grade (6 of the
#: neatness bonus's 10), the claw back on bay D to 0.7 mm. On the pair
#: (#353, Rowan standing by on the workshop route's first leg): 482 sim s,
#: 5.2 mm of lean.
TOWER = '''def tower():
  budget(steps=80, seconds=420)
  fetch("module_claw")
  pick(20)
  place(21)
  pick(22)
  place(20)
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

#: The bench (challenge/bench.py; offered as `find_mass`, home world).
#: The tare is read at the rack (the lift carries the same weight
#: anywhere), `pick(24)` drives the lab route itself, and the return goes
#: through the open garden (7.0, 1.2): the planner's own way back hugs the
#: house wall past the garden light's pole and the front-stop reflex
#: stalled it there. MEASURED (2026-09-21, from the rack, hosting pack,
#: the bank's first draw): 25 steps, 387 sim s, 2.8 Wh, `mass` 0.149 kg
#: within the grade's 10 %, the claw back on bay D. ⚠ On the pair (#353)
#: the first leg home, `drive_to(22.0, 3.0)` -- the lab door's line --
#: stops 0.26 m short ("no route") every time: the grade passes off the
#: recorded mass and the errand's stow brings the claw home.
WEIGH = '''def weigh():
  budget(steps=60, seconds=1800)
  fetch("module_claw")
  move("lift", 0.16)
  wait(2)
  tare = 0
  for i in range(4):
    wait(0.5)
    tare = tare + read("lift.force") / 4
  pick(24)
  wait(2)
  f = 0
  for i in range(4):
    wait(0.5)
    f = f + read("lift.force") / 4
  mass = (f - tare) / 9.81
  move("lift", 0.033)
  release()
  move("lift", 0.164)
  move("arm", 0)
  drive(-0.1, 0, 3)
  drive_to(22.0, 3.0)
  drive_to(19.0, 3.1)
  drive_to(16.0, 3.1)
  drive_to(10.0, 3.1)
  drive_to(7.0, 1.2)
  drive_to(4.4, 0.7)
  stow()
'''

#: The procedure that discharges each challenge kind, by kind. The mouse's
#: acts are errands, not procedures (`lifecycle.cage_errand`).
PROCEDURES = {"stack_tower": TOWER, "find_mass": WEIGH}
