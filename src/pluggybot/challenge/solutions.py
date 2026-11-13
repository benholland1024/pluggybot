"""Hand-written solutions to the offered jobs -- ladder A of issue #264, on
the humanoid's hands (#559's stage 3).

A feature nobody uses is indistinguishable from one that cannot be
completed. Each solution here is written in the robot's OWN vocabulary (a
procedure the language accepts) from what the offer tells it -- the area's
address and its directions -- and passes the job's own grader through
`scoring.evaluate`, the door the robot's attempt goes through: so "the
drawer can be opened" is a flown fact (`scripts/solve.py --feature
drawer`) and a robot that never opens it is a finding about the
robot, not the world. The quadrupeds' (the tower and the bench) are at
`quadruped-final`.

⚠ NEVER SHOWN TO THE MIND. Nothing under `mind/` imports this module and
`tests/test_solutions.py` walks the tree to keep it so: a solution in the
prompt is the prompt handing over the answer.
"""

#: THE HUMANOIDS' HAND JOBS (#559, stage 3; SimNotes, "The humanoids' first
#: jobs", has what flying took): each TWO procedures, written when one would
#: have passed the language's cap (1,500 then, `lang.MAX_SOURCE_CHARS`). The
#: walk there finds the row's NEIGHBOUR first (a find stops at its tag's
#: first sight, and one tag alone left the walk's `face` 17-20 deg off the
#: wall), stands 1.3 m out of the wall's face (nearer, the planner's margin
#: round the counter takes the goal), turns ONCE off the handle's tag read
#: from there to `STANCE`, steps to 30 cm short, then looks again and steps
#: the last 30 cm (a step after a turn falls a tenth short; a short one
#: overshoots by up to 3.6 cm). Then the check's own procedure
#: (`evaluation.hands_check.JOBS`; the drawer's is `DRAWER_JOB`).
APPROACH = '''def to_%(prop)s():
  budget(steps=16, seconds=900)
  find(%(beside)d, %(ax)g, %(ay)g, 600)
  find(%(row)d, %(ax)g, %(ay)g, 600)
  f = read("find.facing")
  drive_to(read("find.x") + 1.3 * cos(f) - 0.2 * sin(f), read("find.y") + 1.3 * sin(f) + 0.2 * cos(f), 180)
  face(f + 3.14159 - 6.28318 * (f > 0))
  spot(%(handle)d)
  a = atan2(read("spot.ny"), read("spot.nx"))
  while a > 0.79:
    a = a - 1.571
  while a < -0.79:
    a = a + 1.571
  t = a + %(stance)g
  x = read("spot.x") * cos(t) + read("spot.y") * sin(t)
  y = read("spot.y") * cos(t) - read("spot.x") * sin(t)
  if not step(x - %(ahead)g - 0.3, y - %(left)g, t):
    pass
  spot(%(handle)d)
  if not step(read("spot.x") - %(ahead)g, read("spot.y") - %(left)g):
    pass
'''


#: How far LEFT of square to its prop the robot stands at each, rad: where
#: its check's procedure passed (squared, its arm's way home pushed the
#: drawer back in and the door opened only 44 deg), with room for the 4-5
#: deg a stance lands either side of the one asked.
STANCE = {"drawer": 0.25, "door": 0.2, "chest": 0.12, "cube": 0.15}


#: The drawer's job: its check's procedure (`hands_check.DRAWER`) pulling to
#: 0.22 m, not 0.18 (its runners stop at 0.24), and drawing the open hand
#: back off the knob along the pull before its way home, which from the
#: knob pushed a drawer pulled 17 cm back under the 15 it must be, 5 times
#: in 8 over the tour.
DRAWER_JOB = '''def open_drawer():
  budget(steps=40, seconds=60)
  spot(54)
  x = read("spot.x") + 0.013 * read("spot.nx")
  y = read("spot.y") + 0.013 * read("spot.ny")
  z = read("spot.z") + 0.013 * read("spot.nz")
  a = atan2(read("spot.ey"), read("spot.ex"))
  while a > 0.79:
    a = a - 1.571
  while a < -0.79:
    a = a + 1.571
  fx = cos(a)
  fy = sin(a)
  g = 0
  t = 0.26
  for i in range(4):
    if g == 0:
      open_hand("left")
      if reach("left", x, y, z, fx * cos(t), fy * cos(t), -sin(t), -fy, fx, 0):
        g = 2
    if g == 2:
      if close_hand("left"):
        g = 1
      else:
        g = 0
        withdraw("left")
    t = t - 0.26
  if g == 1:
    pull("left", -fx, -fy, 0, 15, 0.22)
  open_hand("left")
  if not move_hand("left", read("left.x") - 0.06 * fx, read("left.y") - 0.06 * fy, read("left.z") + 0.02):
    pass
  withdraw("left")
'''


def _hands_job(check: str, prop: str) -> tuple[str, str]:
  """A hand job's reference: the walk to its prop (`APPROACH`), searching
  round the address its offer gives (`home.places.area`), then its check's
  procedure (`evaluation.hands_check.JOBS`) -- the drawer's `DRAWER_JOB`."""
  from pluggybot.evaluation import hands_check
  from pluggybot.home import places as addresses
  from pluggybot.home import storeroom
  from pluggybot.perception import tags
  handle = {"chest": tags.CHEST_TAG_IDS, "drawer": tags.DRAWER_TAG_IDS,
            "door": tags.DOOR_TAG_IDS, "cube": tags.CUBE_TAG_IDS}[prop][0]
  ahead, left = storeroom.HANDLE_FROM_SPOT[prop]
  i = storeroom.PROPS.index(prop)
  beside = storeroom.PROPS[i + 1 if i + 1 < len(storeroom.PROPS) else i - 1]
  address = addresses.area(prop)["address"]
  return (APPROACH % {"prop": prop, "row": storeroom.TAGS[prop],
                      "beside": storeroom.TAGS[beside], "handle": handle,
                      "ax": address["x"], "ay": address["y"],
                      "ahead": ahead, "left": left, "stance": STANCE[prop]},
          DRAWER_JOB if prop == "drawer" else hands_check.JOBS[check])


DRAWER = _hands_job("drawer", "drawer")
DOOR = _hands_job("door", "door")
CUBE = _hands_job("place", "cube")
CHEST = _hands_job("lid", "chest")


#: THE RESTAURANT'S TABLE (#560): the walk to the place, then the table
#: check's procedure from its one stance (`evaluation.hands_check.TABLE`,
#: served as `TABLE_LAY_STEP` says). What flying took is SimNotes', "The
#: table set served" and "The lay from the tour":
#:  - one tag gives no facing, so the card's own rotation is read OBLIQUELY
#:    (`mapping.places`: 40 deg and more off a tag, within 0.5 deg): from 45
#:    deg south-west of its face -- the card faces south, the directions say
#:    -- `back` m out on each axis, the stance's line 0.3 m short once
#:    stepped across, and the turn to face along it asked `WIDER` wider;
#:  - `spot.n*` is INTO the card's face, so that turn is its own angle;
#:  - the step across is halved (a step goes 0.5 m at most), and what it
#:    leaves across the card's line, 14 cm at worst, is stepped out off the
#:    two looks the last step needs, past 3 cm: stood 10 cm left of the line
#:    the lay's first look loses the cup, and the lay's step aside, whole
#:    strides, walks 13-16 cm for anything asked of 9.5-16;
#:  - the last 30 cm is STRAIGHT ahead off those two looks (with a step to
#:    its right in it, it landed 2-4 cm long), `NEARER` nearer the table
#:    than the check's start;
#:  - a walk's turns are counted as they went (`step.turn`, to 0.2 deg) and
#:    one more squares it to `FACING` past the card's normal where it is 1.5
#:    deg off.
TABLE_APPROACH = '''def to_table():
  budget(steps=16, seconds=900)
  find(93, %(ax)g, %(ay)g, 600)
  cx = read("find.x")
  cy = read("find.y")
  drive_to(cx - %(back)g, cy - %(back)g, 180)
  face(0.785)
  spot(93)
  a = atan2(read("spot.ny"), read("spot.nx"))
  y = read("spot.y") * cos(a) - read("spot.x") * sin(a)
  if not step(0, y / 2, a + %(wider)g):
    pass
  h = read("step.turn")
  if not step(0, y / 2):
    pass
  h = h + read("step.turn")
  spot(93)
  x = read("spot.x")
  y = read("spot.y")
  spot(93)
  x = (x + read("spot.x")) / 2
  y = (y + read("spot.y")) / 2
  if y * y > 0.0009:
    if not step(0, y):
      pass
    h = h + read("step.turn")
  if not step(x - %(ahead)g + %(nearer)g, 0):
    pass
  h = a + %(facing)g - h - read("step.turn")
  if h * h > 0.0007:
    if not step(0, 0, h):
      pass
'''
#: How much wider than read the walk's turn is asked, rad: turned 1 deg
#: right of the card the check's procedure laid 1 of 8, its fork's carry
#: running out of its posture and swinging the waist to -45 deg, turned 3.5
#: left 6 of 8. It reads no better at 1.1 or 2 deg (SimNotes, "The table set
#: served").
WIDER = 0.058
#: How much nearer the table than the check's start the walk's last step goes,
#: m -- a little of it (the lay's first step aside gives back ~1.4 cm) keeps
#: the stance in the lay's window: 0-3 cm nearer than the check's start
#: (SimNotes, "The lay from the tour"; further back the cup's reach bows out
#: of view of the plate's tag, nearer the arm's way home strikes the table).
NEARER = 0.025
#: Where the walk's last turn wants it to face, rad left of square, by its own
#: reckoning of the turns its steps made: more than 1.5 deg off it (a 41 deg
#: turn lands up to 3 deg over, and a step aside adds up to 2 more; 5 deg left,
#: the lay's put-downs go wide) one more turn puts it back.
FACING = 0.026

#: THE LAY, SERVED: the table check's procedure (`hands_check.TABLE`) with its
#: first step aside taken from where the card reads across the robot (`m`):
#: 15.5 cm left of the card's line, whatever the walk left across it, where
#: the check, starting on the line, asks 15 (SimNotes, "The lay from the
#: tour": a step walks in whole strides, so where it lands depends more on
#: where it starts than on what it is asked).
TABLE_LAY_STEP = ("  a = 0.15\n", "  a = m + 0.155\n")


def _table_job() -> tuple[str, str]:
  """The table job's reference: the walk to its place (`TABLE_APPROACH`),
  searching round the address its offer gives, then the table check's own
  procedure (`hands_check.TABLE`)."""
  from pluggybot.activity import dishware as dw
  from pluggybot.evaluation import counter as ct
  from pluggybot.evaluation import hands_check
  from pluggybot.home import places as addresses
  from pluggybot.home import restaurant
  address = addresses.area(restaurant.TABLE)["address"]
  # the soles' middle from the card's tag where the set-out starts the robot,
  # and the look from 0.3 m further out
  ahead = ct.CARD_IN + ct.TABLE_GAP - dw.CARD[1] - dw.FILM
  assert hands_check.TABLE.count(TABLE_LAY_STEP[0]) == 1
  return (TABLE_APPROACH % {"ax": address["x"], "ay": address["y"], "ahead": round(ahead, 4),
                            "back": round(ahead + 0.3, 4), "wider": WIDER, "nearer": NEARER,
                            "facing": FACING},
          hands_check.TABLE.replace(*TABLE_LAY_STEP))


TABLE = _table_job()


#: THE PLACE LAID FROM THE PASS (#682's stage C, easy mode; Ben, 11-12): the
#: table's place from its pieces set out on the restaurant's expo counter
#: (`counter.pass_set_out`), a piece a trip, each carried by the walk's own
#: hand-over (`HumanoidMission.carries`). Three runs -- the cup, the plate,
#: the fork -- over the procedures they call (`LIBRARY`); what flying took is
#: SimNotes', "The place laid from the pass":
#:  - the pass is found by its pair of tags (68, 69) -- the wall's facing
#:    off the line through the two, as a far find reads one tag's own facing
#:    16 deg off -- and each piece walked up to from 1.1 m off the wall, `s`
#:    to the right of its middle (from 1.2 the walk back from the table found
#:    no route past the tables), so that it lands where its hand takes it
#:    STANDING, about 0.45 m ahead and 0.25 m to that hand's side (deeper,
#:    the reach bows over the counter);
#:  - a piece is pulled back over the counter's edge before the walk: the
#:    carry pose holds it under the pass's top;
#:  - where a tag is not in the colour camera's picture, a turn toward its
#:    side and then a step in (`near`): a face lands up to 16 deg off;
#:  - the cup gripped from above 1.5 cm under its lid and squeezed at 25 N
#:    (at the table check's 40 N it let go while carried in 17 of 27 legs,
#:    at 25 N in 1 of 9), the plate's wall 1 cm lower than the table check
#:    grips it (there the pads' tips catch 0.7-0.9 cm of the wall, and carried
#:    it fell in 37 of 42 legs), the plate and the fork from a hover 6 cm over
#:    their grip, read again there and pulled straight down: the stance turns
#:    5-10 deg as an arm goes out;
#:  - the walk to the place is the table's (`TABLE_APPROACH`), ending
#:    `PASS_BACK` further back, its last step over 0.45 m taken in two
#:    (carrying, a walk ended 0.2 m further back); nothing steps aside
#:    carrying (a cup carried fell on one);
#:  - each piece set down from a hover over its spot, its offset in the hand
#:    the grip's own -- the plate's and the fork's measured from the flights,
#:    as the table check's is (the hand's sensed place is 1.5-2 cm off).
PASS_TO = '''def to_pass(s):
  face(0)
  find(69, %(ax)g, %(ay)g, 600)
  bx = read("find.x")
  by = read("find.y")
  find(68, %(ax)g, %(ay)g, 600)
  cx = read("find.x")
  cy = read("find.y")
  f = atan2(cy - by, cx - bx) + 1.5708
  mx = (bx + cx) / 2 + 1.1 * cos(f) + s * sin(f)
  my = (by + cy) / 2 + 1.1 * sin(f) - s * cos(f)
  drive_to(mx, my, 180)
  face(f + 3.14159 - 6.28318 * (f > 0))
  if not step(0.20, 0):
    pass
'''
PASS_NEAR = '''def near(t, a, b):
  if not spot(t):
    if not step(0, 0, b):
      pass
    if not spot(t):
      if not step(0.12, 0):
        pass
      spot(t)
  x = read("spot.x") - a
  y = read("spot.y") - b
  if y * y < 0.0009:
    y = 0
  if not step(x, y):
    pass
'''
PASS_GO = '''def go(h, x, y, z, f, g):
  dx = x - read(h + ".x")
  dy = y - read(h + ".y")
  dz = z - read(h + ".z")
  d = sqrt(dx * dx + dy * dy + dz * dz)
  if d > 0.01:
    pull(h, dx, dy, dz, f, d, give=g)
  elif d > 0.002:
    if not move_hand(h, x, y, z):
      pass
'''
#: ...the table's walk to the place, as the pass's trips call it: its last
#: step ahead taken in two where it is over 0.45 m (a step goes 0.5 at most).
#: It ends this much further back than the table check's start, m: the cup's
#: put-down is at the edge of the right hand's band both ways -- from where
#: the check stands it took a crouch and a waist turn and fell, from 5 cm
#: further back a bow of 0.55 rad that missed it by 13 cm.
PASS_BACK = 0.01
PASS_TO_PLACE = (("def to_table():\n  budget(steps=16, seconds=900)\n", "def to_place():\n"),
                 ("  if not step(x - %(ahead)g + %(nearer)g, 0):\n    pass\n",
                  "  x = x - %(ahead)g - %(back_pass)g\n  if x > 0.45:\n"
                  "    if not step(x / 2, 0):\n      pass\n    h = h + read(\"step.turn\")\n"
                  "    x = x / 2\n  if not step(x, 0):\n    pass\n"))
PASS_CUP = '''def lay_cup():
  budget(steps=120, seconds=1200)
  to_pass(-0.10)
  near(91, 0.47, -0.25)
  move("right_shoulder_roll", -0.9)
  move("right_elbow", 1.4)
  open_hand("right")
  spot(91)
  x = read("spot.x") - 0.025
  y = read("spot.y")
  cz = read("spot.z") - 0.015
  if not reach("right", x, y, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0):
    withdraw("right")
    reach("right", x, y, cz + 0.115, 0.707, 0, -0.707, 0, 1, 0)
  pull("right", 0, 0, -1, 12, read("right.z") - cz, give=0)
  if not close_hand("right", 25):
    open_hand("right")
    close_hand("right", 25)
  ox = x + 0.025 - read("right.x")
  oy = y - read("right.y")
  oz = read("right.z") - cz + 0.075
  pull("right", 0, 0, 1, 20, 0.05, give=0)
  go("right", 0.26, read("right.y"), read("right.z"), 20, 0)
  to_place()
  spot(93)
  tz = read("spot.z") - 0.055 + oz + 0.10
  reach("right", read("spot.x") - 0.085 - ox, read("spot.y") - 0.24 - oy, tz, 0.707, 0, -0.707, 0, 1, 0)
  pull("right", 0, 0, -1, 6, 0.15, give=0)
  open_hand("right")
  wait(0.4)
  pull("right", 0, 0, 1, 20, 0.08, give=0)
  withdraw("right")
'''
PASS_PLATE = '''def lay_plate():
  budget(steps=120, seconds=1200)
  to_pass(-0.18)
  near(90, 0.45, 0.184)
  move("left_shoulder_roll", 0.9)
  move("left_elbow", 1.4)
  open_hand("left")
  spot(90)
  px = read("spot.x")
  py = read("spot.y")
  gz = read("spot.z") + 0.0384
  reach("left", px, py + 0.066, gz + 0.06, 0.5, 0, -0.866, 0, -1, 0)
  if spot(90):
    go("left", read("spot.x"), read("spot.y") + 0.066, gz + 0.06, 20, 0)
  pull("left", 0, 0, -1, 12, read("left.z") - gz, give=0)
  if not close_hand("left", 80):
    open_hand("left")
    close_hand("left", 80)
  oz = read("left.z") - gz + 0.0434
  pull("left", 0, 0, 1, 20, 0.05, give=0)
  go("left", 0.22, read("left.y"), read("left.z"), 20, 0)
  to_place()
  spot(93)
  tz = read("spot.z") - 0.055 + oz + 0.05
  reach("left", read("spot.x") - 0.21, read("spot.y") + 0.073, tz, 0.5, 0, -0.866, 0, -1, 0)
  pull("left", 0, 0, -1, 6, 0.11, give=0)
  open_hand("left")
  wait(0.4)
  pull("left", 0, 0, 1, 20, 0.09, give=0)
  withdraw("left")
'''
PASS_FORK = '''def lay_fork():
  budget(steps=120, seconds=1200)
  to_pass(0.10)
  near(92, 0.44, 0.20)
  spot(92)
  px = read("spot.x")
  py = read("spot.y")
  gz = read("spot.z") + 0.0264
  a = atan2(read("spot.ey"), read("spot.ex"))
  while a > 0.79:
    a = a - 1.571
  while a < -0.79:
    a = a + 1.571
  move("left_shoulder_roll", 0.9)
  move("left_elbow", 1.4)
  open_hand("left")
  if spot(92):
    px = read("spot.x")
    py = read("spot.y")
  c = cos(a)
  e = sin(a)
  reach("left", px, py, gz + 0.06, 0.5 * c, 0.5 * e, -0.866, e, -c, 0)
  if spot(92):
    go("left", read("spot.x"), read("spot.y"), gz + 0.06, 20, 0)
  pull("left", 0, 0, -1, 12, read("left.z") - gz, give=0)
  if not close_hand("left", 25):
    open_hand("left")
    close_hand("left", 25)
  oz = read("left.z") - gz + 0.0484
  pull("left", 0, 0, 1, 20, 0.05, give=0)
  go("left", 0.21, read("left.y"), read("left.z"), 20, 0)
  to_place()
  spot(93)
  tz = read("spot.z") - 0.055 + oz + 0.05
  reach("left", read("spot.x") - 0.207, read("spot.y") + 0.16, tz, 0.5, 0, -0.866, 0, -1, 0)
  pull("left", 0, 0, -1, 6, 0.11, give=0)
  open_hand("left")
  wait(0.4)
  pull("left", 0, 0, 1, 20, 0.09, give=0)
  withdraw("left")
'''


def _pass_job() -> tuple[tuple[str, ...], tuple[str, ...]]:
  """The place laid from the pass's reference: the procedures its runs call
  (the walks to the pass and to the place, `near`, `go`) and its
  three runs, a piece each, the address its offer gives in its finds."""
  from pluggybot.activity import dishware as dw
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import places as addresses
  from pluggybot.home import restaurant
  address = addresses.area(restaurant.TABLE)["address"]
  ahead = ct.CARD_IN + ct.TABLE_GAP - dw.CARD[1] - dw.FILM
  place = TABLE_APPROACH
  for old, new in PASS_TO_PLACE:
    assert place.count(old) == 1, old
    place = place.replace(old, new)
  terms = {"ax": address["x"], "ay": address["y"], "ahead": round(ahead, 4),
           "back": round(ahead + 0.3, 4), "wider": WIDER, "nearer": NEARER, "facing": FACING,
           "back_pass": PASS_BACK}
  return ((PASS_TO % terms, place % terms, PASS_NEAR, PASS_GO),
          (PASS_CUP, PASS_PLATE, PASS_FORK))


PASS_LIBRARY, PASS = _pass_job()

#: The procedures that discharge each job, by kind: each run in turn, a run
#: that does not end ok the last.
PROCEDURES = {"open_drawer": DRAWER, "open_door": DOOR, "place_cube": CUBE,
              "open_chest": CHEST, "set_table": TABLE, "set_table_pass": PASS}
#: ...and the procedures they call, by kind: put in the library first, run
#: only by a call (#560's defs that call defs).
LIBRARY: dict = {"set_table_pass": PASS_LIBRARY}


#: The jobs by their place: the counter's props and the restaurant's table
#: (`world_config`'s `jobs`) -- the place laid from the pass is the table's.
TARGETS = {"open_drawer": "drawer", "open_door": "door", "place_cube": "cube",
           "open_chest": "chest", "set_table": "dining_table_0",
           "set_table_pass": "dining_table_0"}


def _def_name(source: str) -> str:
  """A source's first def's name: the procedure it is defined under."""
  return source.split("def ", 1)[1].split("(", 1)[0].strip()


def job_routine(life, kind: str, world: str):
  """A job as the robot does it: offered as the cadence offers it
  (`TaskProducer._build`: the area's terms, the prop's hidden dynamics
  drawn and set out as the offer lands), claimed, the hand-written solution
  defined in a library of its own and run as the errand a mind's
  `procedure:<name>` runs, `done`, and the grade on the seam -- the job's
  own predicate through `scoring.evaluate`. What `scripts/solve.py` flies
  and `scripts/energy_spike.py` prices."""
  from pluggybot.lifecycle import errand_from, task_producer, world_facts
  from pluggybot.mind import overseer as ov
  from pluggybot.procedure import lang
  from pluggybot.procedure import library as lib
  maker = task_producer(life.tasks, world, procedures=True)
  params, secret = maker._build(kind, TARGETS[kind])
  task = life.tasks.offer(kind, TARGETS[kind], params=params, secret=secret, ttl=3000.0,
                          t=float(life.data.time))
  assert task is not None and life._claim_task(task.id), "the claim was refused"
  facts = world_facts(world)
  library = lib.Library(facts)
  # ...the procedures the runs call, defined first and never run themselves
  for source in LIBRARY.get(kind, ()):
    library.define(_def_name(source), source)
  sources = PROCEDURES[kind] if isinstance(PROCEDURES[kind], tuple) else (PROCEDURES[kind],)
  runs = []
  for source in sources:
    name = _def_name(source)
    library.define(name, source)
    errand = errand_from(ov.Decision(action=f"procedure:{name}"), library=library)
    result = yield from life.run_errand_routine(errand)
    run = result.get("procedure") or {}
    runs.append(run)
    if not run.get("ok"):
      break
  life._done(ov.Decision(action="idle", reason="", done=task.id))
  yield from life._grade_routine()
  done = life.tasks.get(task.id)
  verdict = dict(done.verdict or {}) if done is not None else {}
  return {"errand": result, "procedure": {k: run.get(k) for k in ("ok", "failedAt", "seconds")},
          "steps": [(st.get("verb"), st.get("ok"), st.get("why"), st.get("reason"))
                    for r in runs for st in r.get("steps", [])],
          "locals": run.get("locals"), "secret": dict(secret),
          "grade": {"ok": bool(verdict.get("ok")), "reason": verdict.get("reason", ""),
                    "points": verdict.get("points", 0)}}
