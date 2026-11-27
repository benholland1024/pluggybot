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
#: the fork -- over the procedures they call (`LIBRARY`). Each choice is
#: measured (SimNotes, "The place laid from the pass"):
#:  - the pass found by its two tags, the wall's facing off the line through
#:    them (a far find reads one tag's own facing 16 deg off); each piece
#:    walked up to from 1.1 m off the wall, `s` to the LEFT of its middle, so
#:    that its hand takes it STANDING (deeper, the reach bows over the
#:    counter); a tag out of the colour camera's picture, a turn of `b` rad
#:    toward its side, then a step in (`near`);
#:  - the cup gripped at its MIDDLE (aimed short, its side met each pad only
#:    at the fingertip's corner and it crept out of the jaws), brought down to
#:    it from 10 cm over by `move_hand` (by `pull` it drifted across the jaws
#:    onto the lid's rim; reached for nearer, the arm took another
#:    configuration and its pull-back ran out of reach: `test_solutions`),
#:    not read again at its hover (its tag is on its lid, under the hand) --
#:    but a descent held back on the lid, which every such pick lost (the
#:    hover 0.8 cm or more off its middle), is tried again from the stance,
#:    looked at afresh (`test_solutions`);
#:    the plate's wall 1 cm lower than the table check grips it (the pads'
#:    tips over 0.7-0.9 cm of the wall, it slipped out), the plate and the
#:    fork from a hover read again there and pulled straight down (the
#:    stance turns 5-10 deg as an arm goes out);
#:  - each piece brought back over the counter's edge before the walk (the
#:    carry pose holds it under the pass's top): the cup's hand to 0.29 m
#:    (further, its arm runs out of reach, and the hand-over lets go of what
#:    its lift could not raise); the plate, taken from 0.50 m, to
#:    `PASS_PLATE_BACK` by `move_hand` (nearer the pass the hand-over struck
#:    it, and a `pull` that long took a 31 deg bow over the counter and fell:
#:    `test_solutions`), in two moves, halfway and then to no nearer the
#:    body's middle than `PASS_PLATE_LEFT`; the fork's by one pull to
#:    `PASS_FORK_BACK` (by `move_hand` it slipped in the jaws at the
#:    hand-over; `test_solutions`);
#:  - the walk to the place the table's (`TABLE_APPROACH`), `PASS_BACK`
#:    further back, its last step over 0.45 m in two, its last turn the
#:    facing its run wants (the cup's `PASS_CUP_FACING`); nothing steps aside
#:    carrying; its walk to the standoff going on where it gave up (a carried
#:    piece seen walls the robot in 0.1-0.4 m short) -- given up far off, the
#:    card's spot there reads nothing and ends the run (each of 20 given up
#:    4.4-5.3 m short);
#:  - each piece set down from a hover over its spot, its offset in the hand
#:    the grip's own, each put-down moved by where flights on walk2 set its
#:    piece down (the hand's sensed place is 1.5-2 cm off) -- the cup's in the
#:    card's own frame, turned by the card's angle in the stance; the plate's
#:    taken a step toward its spot first where its hand would go nearer the
#:    body's middle than `PASS_PLATE_LEAST`, and eased to the least squeeze
#:    before it is let go (tilted in the jaws, let go, it fell the last
#:    centimetres: `test_solutions`).
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
#: Where the walk to the place's last turn leaves the cup's trip facing, rad
#: left of square (the table's `FACING` for the plate's and the fork's): the
#: cup goes 24 cm right of the card, and a robot facing square reached for it
#: crouched as far as it stood left of the card's line -- a crouching reach
#: tilts a cup in the jaws (taken uncrouched 4 of 139 tilted past 10 deg,
#: crouched 16 cm 11 of 13). Turned 4 deg right, none of 19 tilted and its
#: rests' spread was 0.8 / 1.2 cm (SimNotes, "The place laid from the pass").
PASS_CUP_FACING = -0.07
#: Where the cup's hand is put down from the card's foot, in the card's own
#: frame (m toward the robot, m to its right), the cup's offset in the hand
#: taken off: its spot, 0.07 out and 0.24 right, moved by where dev flights
#: from that stance set it down. ⚠ ON #751'S CURE: aimed at 0.066, 0.209 --
#: tuned on the seeing walker, whose carried turn ended short and settled
#: the cup 2 cm under its pull down -- the cure's cups came down a median
#: 1.5 cm toward the card and 3.2 cm left of their spot (W4's dev draws
#: 3600-3663; SimNotes, "The place laid from the pass").
PASS_CUP_AT = (0.081, 0.24)
#: Where the plate's hand is put down from the card's tag, in the stance
#: frame (m short of the tag ahead, m left of it), and the fork's: moved by
#: where dev flights set each piece down -- the fork's on the cure, which set
#: it 2.0 cm toward the card and 1.7 cm right from 0.18, 0.146; the plate's
#: came down within 2.5 cm in 59 of 61 there, and is not moved.
PASS_PLATE_AT = (0.20, 0.073)
PASS_FORK_AT = (0.20, 0.163)
#: Where the fork's pull-back off the pass ends, m ahead, in one pull: one
#: 0.24 m or longer (a grip 0.45 m out or more) took a 31 deg bow and a 20 cm
#: crouch for its way, stood up out of it within 0.3 s as its arm missed,
#: and fell in 6 of 21, a shorter one in none of 79. On to 0.21 m in a second
#: pull, the gripped fork's pull took a bow of its own about once in 300,
#: and the body bore on it until its feet moved and it fell -- where the
#: walk's hand-over, judged by where its hand comes down (`reach.
#: CARRY_THIN_M`), clears the pass's top as high from here as from 0.21
#: (SimNotes, "The place laid from the pass").
PASS_FORK_BACK = 0.33
#: Where the plate's pull-back off the pass ends, m ahead: from 0.50 m (the
#: pass's face 0.39 ahead) to here by `move_hand` (nearer the pass the
#: hand-over struck it; by `pull` that far it bowed and fell).
PASS_PLATE_BACK = 0.23
#: Where the plate's `near` puts its tag, m left of the stance, and the
#: plate's grip from its tag across: the grip meant 0.25 m out.
PASS_PLATE_AIM, PASS_PLATE_ACROSS = 0.184, 0.066
#: How far left of the stance the plate's pull-back off the pass ends at the
#: least, m -- where the grip is meant to be: a grip the hand read 0.20 m
#: out or nearer the body's middle (`near` lands it 0.12-0.32 m out, half of
#: them under 0.25) was brought straight back to 0.23 m in a posture with
#: the waist turned 30-60 deg, and held back, tipped the plate in its jaws
#: or fell (SimNotes, "The place laid from the pass"). The pull-back goes
#: halfway first, then here: the path here is reached in a crouch at most,
#: while the first, from a grip read 0.08-0.13 m out, still bowed with the
#: waist turned about once in 12 (and never fell).
PASS_PLATE_LEFT = PASS_PLATE_AIM + PASS_PLATE_ACROSS
#: How far left of the stance's middle the plate's put-down goes at the
#: least, m, the card looked at after the walk to the place: nearer, a step
#: puts the card ahead and it is looked at again. Under 0.045 m its reach
#: was out of reach, or took a 0.95 rad bow with the waist turned 0.5-0.8
#: rad and dropped the plate or fell, 7 of 21; over it no such bow in 297
#: (the walk's last step forward leaves the card 2-5 cm across; SimNotes,
#: "The place laid from the pass").
PASS_PLATE_LEAST = 0.05
#: How far right of the stance's middle the cup's put-down goes at the most,
#: m, the card looked at after the walk to the place, and where a step puts
#: it from further: further right than about 0.20 m the cup's reach took no
#: waist turn and its pull down crouched, and the feet swivelled 3-7 deg
#: under the posture's change and carried the cup 3-5 cm off its place --
#: 25 of 44 put-downs within 2.5 cm past 0.19 m, where nearer 0.11-0.19 m all
#: 75 were (SimNotes, "The place laid from the pass").
PASS_CUP_RIGHT, PASS_CUP_MID = 0.20, 0.15
#: How far left of the stance's middle the fork's put-down goes at the most,
#: m, and where a step puts it from further: past about 0.18 m its pull down
#: crouched and swivelled the feet as the cup's did -- 25 of 37 put-downs
#: within 2.5 cm there, 54 of 57 at 0.11-0.17 m.
PASS_FORK_LEFT, PASS_FORK_MID = 0.18, 0.14
#: How far the cup set down may lie from where the card, looked at again with
#: the cup down and still held, puts its place before it is lifted 1.5 cm,
#: moved there and set down again, m: the reach and the pull down turned the
#: stance under the hand 1-2.5 deg (a stance stepped sideways before them, 2.5
#: deg), each degree 0.6 cm at the cup's place (SimNotes, "The place laid from
#: the pass").
PASS_CUP_AGAIN = 0.01
PASS_TO_PLACE = (("def to_table():\n  budget(steps=16, seconds=900)\n", "def to_place(f):\n"),
                 ("  h = a + %(facing)g - h - read(\"step.turn\")\n", "  h = a + f - h - read(\"step.turn\")\n"),
                 ("  drive_to(cx - %(back)g, cy - %(back)g, 180)\n",
                  "  if not drive_to(cx - %(back)g, cy - %(back)g, 180):\n    pass\n"),
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
  x = read("spot.x")
  y = read("spot.y")
  cz = read("spot.z") - 0.015
  if not reach("right", x, y, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0):
    withdraw("right")
    reach("right", x, y, cz + 0.115, 0.707, 0, -0.707, 0, 1, 0)
  if not move_hand("right", x, y, cz):
    if not move_hand("right", x, y, cz + 0.10):
      pass
    withdraw("right")
    move("right_shoulder_roll", -0.9)
    move("right_elbow", 1.4)
    open_hand("right")
    spot(91)
    x = read("spot.x")
    y = read("spot.y")
    cz = read("spot.z") - 0.015
    reach("right", x, y, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0)
    if not move_hand("right", x, y, cz):
      pass
  if not close_hand("right", 25):
    open_hand("right")
    close_hand("right", 25)
  ox = x - read("right.x")
  oy = y - read("right.y")
  oz = read("right.z") - cz + 0.075
  pull("right", 0, 0, 1, 20, 0.05, give=0)
  go("right", 0.29, read("right.y"), read("right.z"), 20, 0)
  to_place(%(cup_facing)g)
  spot(93)
  a = atan2(read("spot.ny"), read("spot.nx"))
  tx = read("spot.x") - %(cup_a)g * cos(a) + %(cup_b)g * sin(a) - ox
  ty = read("spot.y") - %(cup_a)g * sin(a) - %(cup_b)g * cos(a) - oy
  if ty < -%(cup_right)g:
    if not step(0, ty + %(cup_mid)g):
      pass
    spot(93)
    a = atan2(read("spot.ny"), read("spot.nx"))
    tx = read("spot.x") - %(cup_a)g * cos(a) + %(cup_b)g * sin(a) - ox
    ty = read("spot.y") - %(cup_a)g * sin(a) - %(cup_b)g * cos(a) - oy
  tz = read("spot.z") - 0.055 + oz + 0.10
  reach("right", tx, ty, tz, 0.707, 0, -0.707, 0, 1, 0)
  pull("right", 0, 0, -1, 6, 0.15, give=0)
  if spot(93):
    a = atan2(read("spot.ny"), read("spot.nx"))
    dx = read("spot.x") - %(cup_a)g * cos(a) + %(cup_b)g * sin(a) - ox - tx
    dy = read("spot.y") - %(cup_a)g * sin(a) - %(cup_b)g * cos(a) - oy - ty
    if dx * dx + dy * dy > %(cup_again)g * %(cup_again)g:
      pull("right", 0, 0, 1, 20, 0.015, give=0)
      if not move_hand("right", read("right.x") + dx, read("right.y") + dy, read("right.z")):
        pass
      pull("right", 0, 0, -1, 6, 0.05, give=0)
  open_hand("right")
  wait(0.4)
  pull("right", 0, 0, 1, 20, 0.08, give=0)
  withdraw("right")
'''
PASS_PLATE = '''def lay_plate():
  budget(steps=120, seconds=1200)
  to_pass(-0.18)
  near(90, 0.50, %(plate_aim)g)
  move("left_shoulder_roll", 0.9)
  move("left_elbow", 1.4)
  open_hand("left")
  spot(90)
  px = read("spot.x")
  py = read("spot.y")
  gz = read("spot.z") + 0.0384
  reach("left", px, py + %(plate_across)g, gz + 0.06, 0.5, 0, -0.866, 0, -1, 0)
  if spot(90):
    go("left", read("spot.x"), read("spot.y") + %(plate_across)g, gz + 0.06, 20, 0)
  pull("left", 0, 0, -1, 12, read("left.z") - gz, give=0)
  if not close_hand("left", 80):
    open_hand("left")
    close_hand("left", 80)
  oz = read("left.z") - gz + 0.0434
  pull("left", 0, 0, 1, 20, 0.05, give=0)
  if not move_hand("left", (read("left.x") + %(plate_back)g) / 2, read("left.y"), read("left.z")):
    pass
  y = read("left.y")
  if y < %(plate_left)g:
    y = %(plate_left)g
  if not move_hand("left", %(plate_back)g, y, read("left.z")):
    pass
  to_place(%(facing)g)
  spot(93)
  if read("spot.y") + %(plate_at_b)s < %(plate_least)g:
    if not step(0, read("spot.y")):
      pass
    spot(93)
  tz = read("spot.z") - 0.055 + oz + 0.05
  reach("left", read("spot.x") - %(plate_at_a)s, read("spot.y") + %(plate_at_b)s, tz, 0.5, 0, -0.866, 0, -1, 0)
  pull("left", 0, 0, -1, 6, 0.11, give=0)
  if not close_hand("left", 5):
    pass
  wait(0.5)
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
  go("left", %(fork_back)g, read("left.y"), read("left.z"), 20, 0)
  to_place(%(facing)g)
  spot(93)
  if read("spot.y") + %(fork_at_b)s > %(fork_left)g:
    if not step(0, read("spot.y") + %(fork_at_b)s - %(fork_mid)g):
      pass
    spot(93)
  tz = read("spot.z") - 0.055 + oz + 0.05
  reach("left", read("spot.x") - %(fork_at_a)s, read("spot.y") + %(fork_at_b)s, tz, 0.5, 0, -0.866, 0, -1, 0)
  pull("left", 0, 0, -1, 6, 0.11, give=0)
  open_hand("left")
  wait(0.4)
  pull("left", 0, 0, 1, 20, 0.09, give=0)
  withdraw("left")
'''


def _aim(v: float) -> str:
  """A put-down's aim as its source writes it: two decimals where they say
  it exactly, as the reference was first typed, else every digit."""
  two = f"{v:.2f}"
  return two if float(two) == v else repr(float(v))


def _pass_terms() -> dict:
  """The place laid from the pass's terms: what its sources format in -- the
  address its offer gives, the place's distances off the set-out's, and the
  constants above."""
  from pluggybot.activity import dishware as dw
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import places as addresses
  from pluggybot.home import restaurant
  address = addresses.area(restaurant.TABLE)["address"]
  ahead = ct.CARD_IN + ct.TABLE_GAP - dw.CARD[1] - dw.FILM
  return {"ax": address["x"], "ay": address["y"], "ahead": round(ahead, 4),
          "back": round(ahead + 0.3, 4), "wider": WIDER, "nearer": NEARER, "facing": FACING,
          "back_pass": PASS_BACK, "cup_facing": PASS_CUP_FACING,
          "cup_a": PASS_CUP_AT[0], "cup_b": PASS_CUP_AT[1], "fork_back": PASS_FORK_BACK,
          "plate_back": PASS_PLATE_BACK, "plate_left": PASS_PLATE_LEFT,
          "plate_aim": PASS_PLATE_AIM, "plate_across": PASS_PLATE_ACROSS,
          "plate_least": PASS_PLATE_LEAST, "cup_right": PASS_CUP_RIGHT, "cup_mid": PASS_CUP_MID,
          "fork_left": PASS_FORK_LEFT, "fork_mid": PASS_FORK_MID, "cup_again": PASS_CUP_AGAIN,
          "plate_at_a": _aim(PASS_PLATE_AT[0]), "plate_at_b": _aim(PASS_PLATE_AT[1]),
          "fork_at_a": _aim(PASS_FORK_AT[0]), "fork_at_b": _aim(PASS_FORK_AT[1])}


def _pass_job() -> tuple[tuple[str, ...], tuple[str, ...]]:
  """The place laid from the pass's reference: the procedures its runs call
  (the walks to the pass and to the place, `near`, `go`) and its
  three runs, a piece each, the address its offer gives in its finds."""
  place = TABLE_APPROACH
  for old, new in PASS_TO_PLACE:
    assert place.count(old) == 1, old
    place = place.replace(old, new)
  terms = _pass_terms()
  return ((PASS_TO % terms, place % terms, PASS_NEAR, PASS_GO),
          (PASS_CUP % terms, PASS_PLATE % terms, PASS_FORK % terms))


PASS_LIBRARY, PASS = _pass_job()


#: THE PLACE LAID FROM THE KITCHEN'S SHELF (#682's stage C, the full task;
#: Ben, 11-26): the pass's three runs over the procedures they call, each
#: trip begun at the kitchen's shelf instead (`counter.shelf_set_out`) -- the
#: kitchen found off the pass's two tags, across the wall from the pass and
#: through the staff door beside it, the walk's planner finding the way -- and
#: each piece laid as the pass lays it. What the shelf changes, each measured
#: (SimNotes, "The place laid from the kitchen's shelf"):
#:  - the walk stands `SHELF_ACROSS` behind the pass's tags, where `spot`
#:    reads each piece at the first look (on the shelf's 0.96 m board the
#:    cup's lid reads only within about 0.6 m), its last leg straight in, and
#:    `SHELF_S` to the left, so `near` steps little or nothing across;
#:  - `near` stands `SHELF_NEAR_BACK` further off each piece than the pass's;
#:  - the left hand's pre-pose rolled out to `SHELF_LEFT_ROLL` before its
#:    elbow bends to `SHELF_LEFT_ELBOW`: the pass's 0.9 and 1.4 hold the
#:    gripper on the board's front edge;
#:  - the cup is read before the right hand's pre-pose, and again after it
#:    only where it is seen (stood 0.43 m from the cup, the pre-posed gripper
#:    came between the camera and the cup's lid and the look failed); its
#:    retries bring the hand straight back to `SHELF_CUP_BACK` before they
#:    withdraw it (from the hover over the cup, the way home rose into the
#:    board above and was held there);
#:  - the cup's and the fork's look after `near` that misses turns
#:    `SHELF_LOOK_TURN` toward the piece and looks again;
#:  - each thin piece brought back clear of the board and lowered to
#:    `SHELF_STOW_Z` before the walk, the fork by a `pull` that holds its
#:    line and turn, its hand at least `SHELF_FORK_LEFT` left of the middle;
#:  - each piece put down only while its hand still holds it (`SHELF_HELD`).
SHELF_TO = '''def to_shelf(s):
  face(0)
  find(69, %(ax)g, %(ay)g, 600)
  bx = read("find.x")
  by = read("find.y")
  find(68, %(ax)g, %(ay)g, 600)
  cx = read("find.x")
  cy = read("find.y")
  f = atan2(cy - by, cx - bx) + 1.5708
  mx = (bx + cx) / 2 - %(first)g * cos(f) + s * sin(f)
  my = (by + cy) / 2 - %(first)g * sin(f) - s * cos(f)
  if not drive_to(mx, my, 300):
    pass
  if not drive_to(mx - %(last)g * cos(f), my - %(last)g * sin(f), 60):
    pass
  face(f + 3.14159 - 6.28318 * (f > 0))
'''
#: How far behind the pass's tags the walk to the shelf stands, m, across the
#: wall from them: the shelf's front is 2.65 m behind the wall's face, and a
#: walk that stops its pelvis 2.0 m behind left the pieces 0.67-0.72 m from
#: the soles' middle, past what a look reads (every first look failed). It
#: stops first `SHELF_FIRST` behind and walks the rest straight in: come from
#: the staff door, the walk ended turned 40 deg off the shelf, and the `face`
#: after it stopped 5-13 deg short, the cup outside the picture's side.
SHELF_ACROSS, SHELF_FIRST = 2.2, 1.4
#: The left hand's pre-pose before its reach, rad: the shoulder rolled out
#: first, then the elbow bent -- on the robot's model of itself no part of
#: the arm meets a board's front edge 0.33-0.36 m ahead of the pelvis on the
#: way, and the gripper ends 1.03-1.17 m up, over the 0.96 m board and under
#: the one above. The pass's 0.9 and 1.4 hold the gripper 0.40 m ahead at
#: 0.93-0.97 m: two forks of four stopped short there, caught on the edge.
SHELF_LEFT_ROLL, SHELF_LEFT_ELBOW = 1.2, 1.9
#: Where a cup's retry brings its hand back to before it withdraws it, m
#: ahead: in front of the board above, which the way home from the hover over
#: the cup rose into (13-39 N) and was held under.
SHELF_CUP_BACK = 0.25
#: How much further from each piece than the pass's `near` puts it the
#: shelf's does, m: `near`'s step goes past what it asks (a median 1.2 cm for
#: the cup, 3.6 for the plate, 1.0 for the fork, at most 5.6), and stood that
#: near, the shelf board's edge met the left pre-pose (125 N) and the cup's
#: lid left the picture's side.
SHELF_NEAR_BACK = 0.03
#: Where the cup's or the fork's look after `near` misses, the turn toward it
#: before it looks again, rad -- the cup's to the right, the fork's to the
#: left: `near`'s step went 2x across where asked and left the cup 35 deg off
#: the facing, at the picture's side.
SHELF_LOOK_TURN = 0.15
#: How far each walk to the shelf stands to the left of the pass's middle, m:
#: where the trip's piece -- its row 0.5 m left of the pass's middle plus its
#: place in it (`counter.SHELF_SOURCES`) -- stands where `near` puts it, so
#: `near` steps nothing across (an oblique step went twice across what it
#: asked): the cup 0.25 right, the plate 0.184 left -- but the fork 5 cm
#: short of its 0.20 left, `near` asking that across: stood square on it, its
#: step straight ahead turned the stance 5 deg and one fork of three was laid,
#: where stood 5 cm short six of seven were.
SHELF_S = {"cup": 0.40, "plate": 0.32, "fork": 0.60}
#: Where a thin piece is brought back to before the walk, m ahead -- the
#: plate's grip at its near wall, the fork's at its handle's tag -- and the
#: height its hand is then lowered to, m: the walk's hand-over takes a thin
#: grip's side way, which rose a median 0.21 m and swung 0.06-0.16 m forward
#: before it came down, so from over the board the plate's far rim came down
#: on the board's edge (18 of 58) and the fork struck the board above (46 of
#: 52) and fell out of the jaws 8 times. Brought back clear of the board's
#: face and lowered near the carry pose, its way there is short. The fork is
#: lowered by a `pull` that holds its line and turn: by `move_hand` two forks
#: of eight turned in the jaws (11 to 65 deg) and fell out on the way down.
SHELF_PLATE_BACK, SHELF_FORK_BACK, SHELF_STOW_Z = 0.20, 0.22, 0.92
#: The least the fork's hand stands left of the middle from its pull off the
#: shelf on, m (the pass's plate's own least): nearer the middle, the pull,
#: the move back and the stow took a 0.55-0.95 rad bow and the waist up to 60
#: deg in 10 of 49 trips -- the head on the work board in 7, one fall -- and
#: from 0.23 m or more none of 7.
SHELF_FORK_LEFT = 0.25
#: Each shelf run's put-down only while its hand still holds the piece, the
#: run ended otherwise: after the step at the table let the plate go, the
#: put-down's reach from the hanging arm bowed the robot 1.10 rad and it fell.
SHELF_HELD = '''  if read("%s.held") < 0.5:
    return
'''
#: What the shelf's runs change in the pass's, each (old, new) once: the
#: trip's walk, `near`'s distance, the left pre-pose, the cup's and the
#: fork's looks, the cup's retries, each thin piece's stow before the walk,
#: and each put-down's check that its piece is still held.
SHELF_FROM_PASS = {
  "cup": (("  to_pass(-0.10)\n", "  to_shelf(%(cup_s)g)\n"),
          ("  to_place(%(cup_facing)g)\n", "  to_place(%(cup_facing)g)\n" + SHELF_HELD % "right"),
          ('''  near(91, 0.47, -0.25)
  move("right_shoulder_roll", -0.9)
  move("right_elbow", 1.4)
  open_hand("right")
  spot(91)
  x = read("spot.x")
  y = read("spot.y")
  cz = read("spot.z") - 0.015
''', '''  near(91, %(cup_near)g, -0.25)
  if not spot(91):
    if not step(0, 0, -%(look_turn)g):
      pass
    spot(91)
  x = read("spot.x")
  y = read("spot.y")
  cz = read("spot.z") - 0.015
  move("right_shoulder_roll", -0.9)
  move("right_elbow", 1.4)
  open_hand("right")
  if spot(91):
    x = read("spot.x")
    y = read("spot.y")
    cz = read("spot.z") - 0.015
'''),
          ('''    withdraw("right")
    move("right_shoulder_roll", -0.9)
    move("right_elbow", 1.4)
    open_hand("right")
    spot(91)
    x = read("spot.x")
    y = read("spot.y")
    cz = read("spot.z") - 0.015
''', '''    withdraw("right")
    spot(91)
    x = read("spot.x")
    y = read("spot.y")
    cz = read("spot.z") - 0.015
    move("right_shoulder_roll", -0.9)
    move("right_elbow", 1.4)
    open_hand("right")
    if spot(91):
      x = read("spot.x")
      y = read("spot.y")
      cz = read("spot.z") - 0.015
'''),
          ('''  if not reach("right", x, y, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0):
    withdraw("right")
''', '''  if not reach("right", x, y, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0):
    if not move_hand("right", %(cup_back)g, read("right.y"), read("right.z")):
      pass
    withdraw("right")
'''),
          ('''    if not move_hand("right", x, y, cz + 0.10):
      pass
    withdraw("right")
''', '''    if not move_hand("right", x, y, cz + 0.10):
      pass
    if not move_hand("right", %(cup_back)g, read("right.y"), read("right.z")):
      pass
    withdraw("right")
''')),
  "plate": (("  to_pass(-0.18)\n", "  to_shelf(%(plate_s)g)\n"),
            ("  near(90, 0.50, %(plate_aim)g)\n", "  near(90, %(plate_near)g, %(plate_aim)g)\n"),
            ('  move("left_shoulder_roll", 0.9)\n  move("left_elbow", 1.4)\n',
             '  move("left_shoulder_roll", %(left_roll)g)\n  move("left_elbow", %(left_elbow)g)\n'),
            ("  to_place(%(facing)g)\n", "  to_place(%(facing)g)\n" + SHELF_HELD % "left"),
            ('(read("left.x") + %(plate_back)g) / 2', '(read("left.x") + %(shelf_plate_back)g) / 2'),
            ('  if not move_hand("left", %(plate_back)g, y, read("left.z")):\n    pass\n',
             '  if not move_hand("left", %(shelf_plate_back)g, y, read("left.z")):\n    pass\n'
             '  if not move_hand("left", read("left.x"), read("left.y"), %(stow_z)g):\n    pass\n')),
  "fork": (("  to_pass(0.10)\n", "  to_shelf(%(fork_s)g)\n"),
           ("  near(92, 0.44, 0.20)\n  spot(92)\n",
            "  near(92, %(fork_near)g, 0.20)\n"
            "  if not spot(92):\n    if not step(0, 0, %(look_turn)g):\n      pass\n    spot(92)\n"),
           ("  to_place(%(facing)g)\n", "  to_place(%(facing)g)\n" + SHELF_HELD % "left"),
           ('  move("left_shoulder_roll", 0.9)\n  move("left_elbow", 1.4)\n',
            '  move("left_shoulder_roll", %(left_roll)g)\n  move("left_elbow", %(left_elbow)g)\n'),
           ('  go("left", %(fork_back)g, read("left.y"), read("left.z"), 20, 0)\n',
            '  y = read("left.y")\n  if y < %(shelf_fork_left)g:\n    y = %(shelf_fork_left)g\n'
            '  go("left", %(fork_back)g, y, read("left.z"), 20, 0)\n'
            '  if not move_hand("left", %(shelf_fork_back)g, read("left.y"), read("left.z")):\n    pass\n'
            '  if read("left.z") > %(stow_above)g:\n'
            '    pull("left", 0, 0, -1, 20, read("left.z") - %(stow_z)g, give=0)\n')),
}


def _shelf_job() -> tuple[tuple[str, ...], tuple[str, ...]]:
  """The place laid from the kitchen's shelf's reference: the walk to the
  shelf, the pass's walk to the place, `near` and `go`, and the pass's three
  runs changed where `SHELF_FROM_PASS` says, the address its offer gives in
  its finds."""
  from pluggybot.home import places as addresses
  from pluggybot.home import restaurant
  address = addresses.area(restaurant.TABLE)["address"]
  runs = []
  for piece, source in (("cup", PASS_CUP), ("plate", PASS_PLATE), ("fork", PASS_FORK)):
    for old, new in SHELF_FROM_PASS[piece]:
      assert source.count(old) == 1, (piece, old)
      source = source.replace(old, new)
    runs.append(source)
  # ...each run's terms the pass's, and the shelf's
  terms = _pass_terms()
  terms.update({"left_roll": SHELF_LEFT_ROLL, "left_elbow": SHELF_LEFT_ELBOW,
                "cup_back": SHELF_CUP_BACK, "look_turn": SHELF_LOOK_TURN,
                "cup_s": SHELF_S["cup"], "plate_s": SHELF_S["plate"], "fork_s": SHELF_S["fork"],
                "cup_near": round(0.47 + SHELF_NEAR_BACK, 4),
                "plate_near": round(0.50 + SHELF_NEAR_BACK, 4),
                "fork_near": round(0.44 + SHELF_NEAR_BACK, 4),
                "shelf_plate_back": SHELF_PLATE_BACK, "shelf_fork_back": SHELF_FORK_BACK,
                "shelf_fork_left": SHELF_FORK_LEFT,
                "stow_z": SHELF_STOW_Z, "stow_above": round(SHELF_STOW_Z + 0.01, 4)})
  _, place, near, go = PASS_LIBRARY
  return ((SHELF_TO % {"ax": address["x"], "ay": address["y"], "first": SHELF_FIRST,
                       "last": round(SHELF_ACROSS - SHELF_FIRST, 4)},
           place, near, go),
          tuple(source % terms for source in runs))


SHELF_LIBRARY, SHELF = _shelf_job()

#: The procedures that discharge each job, by kind: each run in turn, a run
#: that does not end ok the last.
PROCEDURES = {"open_drawer": DRAWER, "open_door": DOOR, "place_cube": CUBE,
              "open_chest": CHEST, "set_table": TABLE, "set_table_pass": PASS,
              "set_table_shelf": SHELF}
#: ...and the procedures they call, by kind: put in the library first, run
#: only by a call (#560's defs that call defs).
LIBRARY: dict = {"set_table_pass": PASS_LIBRARY, "set_table_shelf": SHELF_LIBRARY}


#: The jobs by their place: the counter's props and the restaurant's table
#: (`world_config`'s `jobs`) -- the place laid from the pass, and from the
#: kitchen's shelf, is the table's.
TARGETS = {"open_drawer": "drawer", "open_door": "door", "place_cube": "cube",
           "open_chest": "chest", "set_table": "dining_table_0",
           "set_table_pass": "dining_table_0", "set_table_shelf": "dining_table_0"}


def _def_name(source: str) -> str:
  """A source's first def's name: the procedure it is defined under -- one
  with none refused here, never a StopIteration inside `job_routine`."""
  import ast
  name = next((n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)), None)
  if name is None:
    raise ValueError(f"a reference source with no def: {source[:60]!r}")
  return name


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
