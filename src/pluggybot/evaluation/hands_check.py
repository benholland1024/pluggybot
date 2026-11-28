"""The hands' capability check (issue #525; docs/Hands.md, "The capability
check"): #493's gate pattern for the hands' vocabulary. Before any mind
writes in the vocabulary, a procedure WE wrote in it, for each of a few
generic jobs, is flown at that job's set-outs and graded by the job's own
grader -- the bar written before it flew:

  lid     the chest's lid opened past 57 deg and set down (challenge/lid.py)
  drawer  a drawer pulled 15 cm out by its knob (challenge/drawer.py)
  door    a cupboard's door opened 50 deg by its lever (challenge/door.py)
  place   a cube picked up and set down on a mat (challenge/place.py)

Each is the language as a mind would write it (`procedure.lang`,
`steps.hand_facts`): its caps, the hands' verbs and sensors, the maths --
and no verb that knows the job (Ben's call 2 on #493). What each knows that
a mind would have to find out for itself is said above it.

⚠ OURS, NEVER THE ROBOT'S: a procedure here is a solution. Nothing under
`mind/` may import this module (a test reads the imports), and no prompt
shows one.
"""

from __future__ import annotations

from pluggybot.evaluation import attempts as at
from pluggybot.evaluation import practice as ep
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st

#: THE LID: the knob's middle off its tag (half the 26 mm cube in from the
#: face read), its facing off the tag's edges; gripped from the front 2 cm
#: SHORT of that middle, back level along its facing -- ⚠ the pads reach 4
#: cm past the grip and the rising lid's front edge swings up under them: at
#: the middle, where the grips landed 0.9 cm deeper, they met it at 43 deg
#: and the lift ran out of reach; 2.5 cm short, a knob read true sticks out
#: past the tips (SimNotes, "The chest's grip") -- the fingers tilted up
#: first (at the top of its arc, the hand that kept its turn reaches about
#: level) and the jaws along the hinge's line, at the first tilt that holds;
#: lifted up and a little back by a FORCE MOVE that lets the hinge steer the
#: hand along the arc; lowered toward the body and down, again by force --
#: the lid set down by its hinge, not by a plan -- and let go only once the
#: hand is back down. ⚠ ITS FIRST REACH THAT STOPPED SHORT, ITS HAND OUT
#: (`left.x`: on its way out or in), IS TRIED ONCE MORE AT THE SAME TILT
#: WITH THE GRIP 5 DEG TOWARD THE ROBOT'S MIDDLE before the tilts go on, the
#: last of them (44 deg) then untried (#708, as the door's: a first way out
#: stopped on the counter's edge sent lids to the next tilt, which let them
#: fall or never opened them); never after a reach refused with nothing
#: moved (so turned after three refusals, a grip found a plan whose lift let
#: the lid fall); at 10 deg its pads met the rising lid at 62.
LID = '''def open_lid():
  budget(steps=40, seconds=66)
  spot(53)
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
  x = x - 0.02 * fx
  y = y - 0.02 * fy
  g = 0
  t = -0.26
  r = 0.0
  n = 0
  for i in range(6):
    s = 0.0
    if g == 0:
      open_hand("left")
      if reach("left", x, y, z, cos(a - r) * cos(t), sin(a - r) * cos(t), -sin(t), -sin(a - r), cos(a - r), 0):
        g = 2
      else:
        s = read("left.x")
        withdraw("left")
    if g == 2:
      if close_hand("left"):
        g = 1
      else:
        g = 0
        withdraw("left")
    if n == 0 and g == 0 and s > 0.2:
      n = 1
      r = 0.09
    else:
      r = 0.0
      if t < -0.26:
        t = -0.52 - t
      else:
        t = -0.69 - t
  if g == 1:
    z0 = read("left.z")
    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27)
    pull("left", -fx, -fy, -1, 8, 0.5)
    for j in range(2):
      if read("left.z") > z0 + 0.03:
        pull("left", -0.3 * fx, -0.3 * fy, -1, 8, 0.3)
  open_hand("left")
  withdraw("left")
'''

#: THE DRAWER: its knob's middle and facing off its tag, as the lid's;
#: gripped from the front, the fingers tilted down first, the jaws across;
#: pulled straight out toward the body by force, its runners steering it.
DRAWER = '''def open_drawer():
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
    pull("left", -fx, -fy, 0, 15, 0.18)
  open_hand("left")
  withdraw("left")
'''

#: THE DOOR: the lever's tip off its tag; gripped from the front, the jaws
#: closing on its top and bottom; pressed down by force (the pivot steers
#: the hand round) until the lever stops, which frees the latch; let go,
#: the open hand backed straight out; then LOOKED AT AGAIN -- a door whose
#: lever moved 10 cm or out of sight is open, and gripping again would push
#: it shut -- and otherwise gripped again LIGHTLY, so its tip turns between
#: the jaws as the door swings, and pulled toward the body by force, the
#: hinge steering the hand along the door's arc, then on the way the hand
#: was last going. ⚠ A REACH THAT FAILS -- STOPPED SHORT, OR REFUSED (none
#: of the check's first reaches was, over set-outs 0-127) -- TRIES ITS
#: FINGERS 10 DEG TOWARD THE ROBOT'S MIDDLE, UNTILTED, BEFORE IT TILTS THEM
#: DOWN 15 DEG A TRY (#708; SimNotes, "The door's reach the stop sends
#: back"; its `g = 5` a state of its own: the language nests 4 deep): at 12 of
#: set-outs 0-191 the grip asked plans the deepest crouch with the waist
#: turned 30 deg, whose way out drags the open jaw over the counter's edge
#: into the door, and a way out stopped there (`reach.OUT_TIP_NM`) went to
#: the tilt, whose press and move back bowed the robot onto the counter;
#: turned toward the middle, it plans a shallower crouch, square, at all 12.
DOOR = '''def open_door():
  budget(steps=60, seconds=66)
  ph = 0
  g = 3
  t = 0.0
  r = 0.0
  for i in range(12):
    if g == 3:
      g = 4
      if spot(55):
        g = 7
    if g == 7:
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
      if ph == 0:
        x0 = x
        y0 = y
      if sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) > 0.1:
        g = 4
    if g == 0:
      open_hand("left")
      if reach("left", x, y, z, cos(a - r) * cos(t), sin(a - r) * cos(t), -sin(t), 0, 0, 1):
        g = 2
      else:
        withdraw("left")
        g = 5
    if g == 5:
      g = 0
      if r == 0 and t == 0:
        r = 0.17
      else:
        r = 0.0
        t = t + 0.26
    if g == 2:
      if close_hand("left", 120 - 105 * ph):
        g = 1
      else:
        g = 0
        withdraw("left")
    if g == 1:
      if ph == 0:
        pull("left", 0, 0, -1, 10, 0.09)
      else:
        pull("left", -fx, -fy, 0, 12, 0.14)
        pull("left", read("pull.dx"), read("pull.dy"), 0, 12, 0.08)
      open_hand("left")
      if not move_hand("left", read("left.x") - 0.08, read("left.y"), read("left.z") + 0.02):
        pass
      withdraw("left")
      ph = ph + 1
      g = 3 + (ph > 1)
      t = 0.0
      r = 0.0
'''

#: THE CUBE, out of reach where the robot stands: the mat's middle and the
#: cube's off their tags, the cube's turn off its edges; a STEP in and aside,
#: until the cube is 0.32 m ahead and 0.18 to the left -- nearer, the head
#: camera no longer sees it standing, so what it read is turned into the
#: frame it stands in after the step (`step.*`; stepped in alone, the waist's
#: stand left the cube where the carry ran into the shoulder's inner stop:
#: SimNotes, "The cube and the robot's noise"). Gripped steeply from
#: above-front, the jaws on two of its faces; lifted straight up and set down
#: by force moves that hold their line (`give` 0: a cube set down giving slid
#: along what it touched), carried over the mat by where the grip left it in
#: the hand; and let go only once its set-down has pressed on the mat. What
#: the tour took (#560; SimNotes, "The cube from the tour"):
#:  - ⚠ A TRY THAT FAILED READS THE MAT AND THE CUBE AGAIN, and the next aims
#:    where the cube now is, 1 cm on along the fingers: a reach stopped short
#:    or a shut on nothing moved the feet 3-7 cm, and every try after it
#:    aimed at the first read. Where only one tag reads, the other is moved
#:    as it seems to have moved: the feet's move moves both alike. The first
#:    try aims at the read the step carried, as before (read again first,
#:    more first reaches stopped short, found no posture or turned the stance
#:    14 deg);
#:  - ⚠ A LIFT THAT LEFT THE CUBE BEHIND IS A TRY THAT FAILED (`left.held`,
#:    read once its running mean has had 0.1 s: a grip lost read held up to
#:    40 ms on, #646): a shut on its edge read held and let go as it rose;
#:  - the tries' tilts are 60, 60, 46 and 75 deg: at 75 a reach took 12 s and
#:    moved the feet most;
#:  - the carry aims 0.6 cm on and 0.5 cm to the right of where the grip's
#:    offset puts it: the jaws push the cube along them as they shut, and it
#:    was set down a mean 0.55 cm short and 0.5 to the left over 62 flights;
#:  - 150 s of budget lets a fourth try end (the check grades its own 68 s).
PLACE = '''def place_cube():
  budget(steps=60, seconds=150)
  spot(57)
  mx = read("spot.x")
  my = read("spot.y")
  spot(56)
  x = read("spot.x") + 0.013 * read("spot.nx")
  y = read("spot.y") + 0.013 * read("spot.ny")
  z = read("spot.z") + 0.013 * read("spot.nz")
  a = atan2(read("spot.ey"), read("spot.ex"))
  if not step(x - 0.32, y - 0.18):
    pass
  c = cos(read("step.turn"))
  s = sin(read("step.turn"))
  u = x - read("step.dx")
  v = y - read("step.dy")
  x = c * u + s * v
  y = c * v - s * u
  u = mx - read("step.dx")
  v = my - read("step.dy")
  mx = c * u + s * v
  my = c * v - s * u
  a = a - read("step.turn")
  b = 0
  p = 0
  q = 0
  g = 0
  t = 1.05
  for i in range(4):
    k = (g == 0) * (i > 0)
    m = 0
    n = 0
    if k > 0:
      if spot(57):
        m = 1
        u = read("spot.x")
        v = read("spot.y")
    if k > 0:
      if spot(56):
        n = 1
        p = read("spot.x") + 0.013 * read("spot.nx")
        q = read("spot.y") + 0.013 * read("spot.ny")
        z = read("spot.z") + 0.013 * read("spot.nz")
        a = atan2(read("spot.ey"), read("spot.ex"))
    if m + n > 0:
      e = x
      h = y
      x = n * p + (1 - n) * (x + u - mx)
      y = n * q + (1 - n) * (y + v - my)
      mx = m * u + (1 - m) * (mx + p - e)
      my = m * v + (1 - m) * (my + q - h)
      b = 0.01
    if g == 0:
      while a > 0.79:
        a = a - 1.571
      while a < -0.79:
        a = a + 1.571
      open_hand("left")
      if reach("left", x + b * cos(a), y + b * sin(a), z + 0.003, cos(a) * cos(t), sin(a) * cos(t), -sin(t), -sin(a), cos(a), 0):
        g = 2
      elif not withdraw("left"):
        pass
    w = 0
    if g == 2:
      g = 0
      w = 1
      if close_hand("left"):
        ox = x - read("left.x")
        oy = y - read("left.y")
        pull("left", 0, 0, 1, 20, 0.03, give=0)
        wait(0.1)
        g = read("left.held") > 0.5
        w = 1 - g
    if w > 0:
      open_hand("left")
      if not withdraw("left"):
        pass
    t = 1.05 - 0.25 * (i > 0) + 0.51 * (i > 1)
  if g == 1:
    move_hand("left", mx - ox + 0.006, my - oy - 0.005, read("left.z"))
    for j in range(2):
      if g == 1:
        pull("left", 0, 0, -1, 4, 0.1, give=0)
        g = 1 + (read("pull.force") > 3)
  open_hand("left")
  withdraw("left")
'''

#: THE TABLE'S PLACE (#560) from ONE STANCE left of the start, where the
#: waist's yaw reaches every piece (#621; 0.15 m from the check's start,
#: `solutions.TABLE_LAY_STEP` off the walk). Each choice is deliberate and
#: measured (SimNotes, "The table's reference on the waist" and "The lay from
#: the tour"): the cup's tag read before the step (out of view from the
#: stance; turned to if the first look loses it), what was read carried
#: through the step's own turn, the plate's tag read again from the stance;
#: the cup from above, gripped 3 cm short of its middle (at the middle the
#: palm cuts the lid) and placed off the plate's tag read with the hand out;
#: its reach aimed inside the 2 cm band its shallow posture holds (`e`,
#: across the arm's way out: nearer, the planner stands it up and the descent
#: bows it; further, it bows the plate's tag out of view), a pull carrying the
#: hand on over the grip, a reach still cut short tried again 1.5 cm higher;
#: the plate's and the fork's jaws the other way round at 60 deg (as asked
#: before, the yaw takes the grip that never carries), each reach aimed at
#: its grip (#627: the reach plans where the robot stands), the plate
#: gripped 1.5 cm inside its wall (at it, a pad brushed the fork over), the
#: fork at 25 N (chosen before #628, when the drive's chatter batted it from
#: 40), the plate at 40 -- ⚠ a CARRIED plate is squeezed at 80 (#682, Ben's
#: call 3: at 40 a 0.5 kg plate dropped in 8 of 20 walked carries of 13 m), but
#: this lay walks nothing, and at 80 it laid 77 of 96 set-outs where 40 laid
#: 82, paired on one pod, the 5 lost all the fork's, knocked out after the
#: plate (SimNotes, "Carrying, served"); every precise move a pull (`go`: a carry takes only a posture its
#: statue reaches); each piece set down by where its jaws left it, the card
#: read again once it is up, the plate 1 cm to the right and the fork 0.5 cm
#: nearer the robot against what the carry's turn leaves; an open hand
#: lifted straight up, its jaws opened first, 9 cm before the left arm's way
#: home (at 6, a hand nearer the table struck its edge going home, and at 12
#: the lift itself ran out of reach and moved the feet).
TABLE = '''def lay_place():
  budget(steps=190, seconds=290)
  spot(93)
  k = read("spot.x")
  m = read("spot.y")
  a = 0.15
  spot(90)
  px = read("spot.x")
  py = read("spot.y")
  s = k - 0.077 - px
  w = m - 0.243 - py
  t = 0
  if not spot(91):
    if not step(0, 0, -0.17):
      pass
    t = read("step.turn")
    spot(90)
    px = read("spot.x")
    py = read("spot.y")
    spot(91)
  u = read("spot.x") - px
  v = read("spot.y") - py
  cz = read("spot.z") - 0.03
  if not step(0, a, -t):
    pass
  r = read("step.turn")
  c = cos(r)
  e = sin(r)
  q = c * u + e * v
  v = c * v - e * u
  u = q - 0.03
  c = cos(r + t)
  e = sin(r + t)
  q = c * s + e * w
  w = c * w - e * s
  s = q - 0.03
  move("right_shoulder_roll", -0.9)
  move("right_elbow", 1.4)
  open_hand("right")
  spot(90)
  x = read("spot.x") + u
  y = read("spot.y") + v
  e = 0.7 * y - 0.72 * x
  f = 0
  if e > -0.577:
    f = e + 0.577
  if e < -0.597:
    f = e + 0.597
  g = 0
  if reach("right", x + 0.72 * f, y - 0.7 * f, cz + 0.10, 0.707, 0, -0.707, 0, 1, 0):
    if spot(90):
      x = read("spot.x") + u
      y = read("spot.y") + v
      g = 1
  if g == 0:
    withdraw("right")
    reach("right", x + 0.72 * f + 0.005, y - 0.7 * f, cz + 0.115, 0.707, 0, -0.707, 0, 1, 0)
    if spot(90):
      x = read("spot.x") + u
      y = read("spot.y") + v
  go("right", x, y, cz + 0.10, 20, 0)
  go("right", x, y, cz, 12, 1)
  if not close_hand("right", 40):
    open_hand("right")
    close_hand("right", 30)
  ox = x - read("right.x")
  pull("right", 0, 0, 1, 20, 0.05, give=0)
  if spot(90):
    x = read("spot.x") + s
    y = read("spot.y") + w
  else:
    x = x + s - u
    y = y + w - v
  go("right", x - ox, y, read("right.z"), 20, 0)
  pull("right", 0, 0, -1, 6, 0.09, give=0)
  open_hand("right")
  wait(0.4)
  pull("right", 0, 0, 1, 20, 0.08, give=0)
  withdraw("right")
  for i in range(2):
    move("left_shoulder_roll", 0.9)
    move("left_elbow", 1.4)
    open_hand("left")
    spot(93)
    k = read("spot.x")
    m = read("spot.y")
    spot(90 + 2 * i)
    px = read("spot.x")
    py = read("spot.y")
    gy = py + 0.066 * (1 - i)
    gz = 0.7734 - 0.005 * i
    reach("left", px, gy, gz, 0.5, 0, -0.866, 0, -1, 0)
    if not close_hand("left", 40 - 15 * i):
      open_hand("left")
      close_hand("left", 30 - 15 * i)
    ox = px - read("left.x")
    oy = -0.084 * (1 - i)
    pull("left", 0, 0, 1, 20, 0.05, give=0)
    if spot(93):
      k = read("spot.x")
      m = read("spot.y")
    go("left", k - 0.19 - 0.005 * i - ox, m + 0.16 * i - 0.01 * (1 - i) - oy, read("left.z"), 20, 0)
    pull("left", 0, 0, -1, 6, 0.09, give=0)
    open_hand("left")
    wait(0.4)
    pull("left", 0, 0, 1, 20, 0.09, give=0)
    withdraw("left")


def go(h, x, y, z, f, g):
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

#: ...the reference before the waist's (#615, #618): a stance a piece --
#: the cup where the robot stands, a step right for the plate and back for
#: the fork -- and every grip 45 deg as asked; on the waist's body it laid no
#: place in 32 (SimNotes, "Setting a table"). Kept as the base of #621's
#: flown levers, the scenes below.
TABLE_STANCES = '''def lay_place():
  budget(steps=120, seconds=400)
  move("right_shoulder_roll", -0.9)
  move("right_elbow", 1.4)
  open_hand("right")
  spot(93)
  u = read("spot.x") - 0.07
  v = read("spot.y") - 0.24
  spot(91)
  reach("right", read("spot.x"), read("spot.y"), read("spot.z") - 0.03, 0.707, 0, -0.707, 0, 1, 0)
  close_hand("right", 40)
  pull("right", 0, 0, 1, 20, 0.05, give=0)
  move_hand("right", u, v, read("right.z"))
  pull("right", 0, 0, -1, 6, 0.09, give=0)
  open_hand("right")
  move_hand("right", read("right.x") - 0.04, read("right.y"), read("right.z") + 0.07)
  withdraw("right")
  spot(93)
  k = read("spot.x")
  m = read("spot.y")
  spot(90)
  px = read("spot.x") - k
  py = read("spot.y") - m + 0.081
  spot(92)
  fx = read("spot.x") - k
  fy = read("spot.y") - m
  for i in range(2):
    step(0, 0.43 * i - 0.26)
    move("left_shoulder_roll", 0.9)
    move("left_elbow", 1.4)
    open_hand("left")
    spot(93)
    k = read("spot.x")
    m = read("spot.y")
    reach("left", k + px + (fx - px) * i, m + py + (fy - py) * i, 0.767 - 0.005 * i, 0.707, 0, -0.707, 0, 1, 0)
    close_hand("left", 40)
    pull("left", 0, 0, 1, 20, 0.05, give=0)
    move_hand("left", k - 0.19, m + 0.081 + 0.079 * i, read("left.z"))
    pull("left", 0, 0, -1, 6, 0.09, give=0)
    open_hand("left")
    move_hand("left", read("left.x") - 0.04, read("left.y"), read("left.z") + 0.07)
    withdraw("left")
'''

#: THE SHELF STOCKED (#736) from the cart beside its slots, the left hand,
#: from where the check starts (`counter.stock_set_out`): the frame, then
#: the candle (`one`), each read off its tag and taken level from 10 cm above
#: it, carried by sidesteps to its label, set down 8 cm behind the board's
#: edge from 17 cm in front of it, let go, backed out 15 cm and only then
#: withdrawn (never from over an edge, #736's item 9). Written with the fewest
#: spaces its words need (the 3,000-character cap). Each choice measured
#: (SimNotes, "Stocking the shop's shelf"): ⚠ a sidestep's count reads 1-4
#: cm short of the feet, so a trip in steps over-travels: the carried one is
#: aimed to bring its label in 0.06 m left of the robot's middle (a carried
#: product hides a label 0.25 m or more to its left, and a step left carrying
#: often reads blocked), so `near` brings it to 0.23 mostly by steps right; ⚠ a
#: look that misses steps the way its tag most likely went -- a frame or a
#: candle at the cart to its left: uncarried, a step right to its mark there
#: goes 0.1-1.3 times its ask (a candle's 10 cm, 15-17), and every candle its
#: look missed stood 0.27-0.80 m left, where a search right lost it; ⚠ at a label
#: `near` steps no step back, and ahead only past 5 cm, half of it (carried,
#: back steps of 3-5 cm read 'too short' and fell; one ahead of 4 cm went 8);
#: at the cart 0.6 of a step ahead (one went 1.4 times its ask and lost the
#: candle under the picture); ⚠ squared off the label's face past 2 deg before
#: each set-down and the trip back (a metre's sidestep 9 deg off square took
#: the robot out of the labels' view); ⚠ the candle gripped 1.5 cm above its
#: middle (held at its centre of mass it turned over in the swing to the carry
#: pose); ⚠ a pick held back on a product's top tried again from a fresh look
#: (the frame is 75 mm in the jaws' 90); ⚠ a set-down reach stopped short
#: tried again 8 cm further back and 3 cm higher; ⚠ a press that read under 3
#: N never landed: the hand goes 5 cm lower before the let-go; ⚠ on the upper
#: board the hand is aimed 2.5 cm left of its label, on the lower 1 cm right
#: (where each board's placements landed); ⚠ a frame gripped never turned
#: outward, and one read nearer than 0.20 m left 0.1 rad inward (round 5):
#: turned to an outward face the arm's reach was refused, or its descent came
#: down on the frame's top and the close knocked it over; ⚠ on the upper board
#: the robot stands 3.5 cm further left of its label, the hand's aim off the
#: label unchanged (round 6): aimed past about 0.26 m left there, the set-down's
#: square grip planned only half a turn round, which turned the held product
#: over; ⚠ a lower label is read off its twin above it, the set-down 0.40 m
#: under it (round 6): from nearer than 0.48 m a lower label's tag is past the
#: colour picture's bottom, and the carried trip's sidesteps creep 2-8 cm in; ⚠
#: at the cart the candle's step is taken sideways alone while its sideways
#: error is past near()'s tolerance (round 10): taken ahead and sideways at once
#: after the trip back, the cart's front-right leg at the knees' height stopped
#: it, and the pick was out of reach; ⚠ a frame whose label reads nearer than
#: 0.45 m once the robot stands at it steps back 8 cm before its set-down (round
#: 13): the carried trip's sidesteps bring the robot in, and from there the set-
#: down's bow stopped short, its fallback was planned standing with the waist
#: turned and the frame turned over, or the trip back was blocked by the cart's
#: front-right leg (a carried step asked under about 6 cm moves nothing); ⚠ the
#: lower candle is aimed 1 cm further right (round 13): the set-down's postures
#: swivel the feet, and its placements landed a mean 1 cm left; ⚠ an upper
#: candle whose label reads further than 0.50 m is set down 2 cm less deep
#: (round 17): from there the press at full stretch ran out of reach with a jaw
#: on the board and slid the hand back toward the board's edge, and the candle
#: let go there fell; ⚠ an upper candle whose label reads nearer than 0.45 m
#: steps back 8 cm before its set-down too (round 18): set down from nearer, the
#: reach took its deepest posture and put the head on the board; ⚠ the frame is
#: aimed 1 cm further right (round 20): its placements landed a mean 0.8-0.9 cm
#: left of their labels, and every one off its label landed left of it; ⚠ on the
#: upper board the set-down reach's pre-pose is at most 0.35 m ahead (round
#: 22b): further out the stand reaches it, and the carry in and the press then
#: each take another posture with the product at the board -- set down so on the
#: dev rows, 41 of 85 upper products ended on their label, and 295 of 299 whose
#: reach went at once to the deep bow, which reaches the set-down point and the
#: press as well; the retry of a reach stopped short is unchanged, 8 cm behind
#: the unclamped pre-pose and 3 cm higher; ⚠ on the lower board a set-down is
#: aimed no nearer the robot's middle than 0.19 m left (round 23): planned
#: nearer, on the rows since round 9 a candle's first reach was refused 36 of 98
#: times in checks and 12 of 96 in tours, and a frame's at one check's draw on
#: every arm -- never one of 3,578 planned at 0.19 or more -- and in a tour the
#: fallback was refused too, 10 times of 12, the product still in the hand
#: (SimNotes, "Stocking the shop's shelf"). `%(candle)d` and `%(frame)d` are the
#: job's labels (`source`).
SHELF = '''def stock_shelf():
  budget(steps=200,seconds=290)
  one(%(frame)d,0)
  one(%(candle)d,1)


def one(lab,i):
  t=77-i
  k=lab-79-3*(lab>81);tl=lab+3*(lab<82)
  g=0
  for n in range(2):
    if g==0:
      open_hand("left")
      near(t,0.34*i,0.21+0.015*i,0.05*(1-i)+0.08*i)
      j=1-i;a=j*atan2(read("spot.ny"),read("spot.nx"));e=cos(a);s=sin(a)
      mx=read("spot.x")+0.01*e*j;my=read("spot.y")+0.01*s*j
      if j*(a+0.1*(my<0.2))>0:c=-0.1*(my<0.2);e=cos(c);s=sin(c)
      gz=read("spot.z")-0.035*i;bz=read("spot.z")-0.0575-0.0425*i
      h=0.02+0.01*j
      if not reach("left",mx-h*e,my-h*s,gz+0.1,e,s,0,-s,e,0):e=1;s=0;reach("left",mx-h,my,gz+0.1,1,0,0,0,1,0)
      m=0
      if not move_hand("left",mx-h*e,my-h*s,gz):m=1
      if m*n==1:pull("left",0,0,1,20,0.08,give=0);reach("left",mx-h*e,my-h*s,gz,e,s,0,-s,e,0);m=0
      if not close_hand("left",40):m=2
      if m==0:g=1
      if g==0:open_hand("left");pull("left",-2,0,1,20,0.25,give=0);withdraw("left")
  u=mx-read("left.x");v=my-read("left.y")
  q=u*e+v*s;r=v*e-u*s;oz=read("left.z")-bz
  pull("left",0,0,1,20,0.03,give=0)
  tz=0.035*(lab>81)
  side(my-0.87+0.36*i-0.25*k+tz)
  near(tl,0.5,0.23-tz,0.08)
  if ((i<1)+i*(lab>81))*(read("spot.x")<0.45):
    if not step(-0.08,0):pass
    near(tl,0.5,0.23-tz,0.08)
  a=atan2(read("spot.ny"),read("spot.nx"))
  if a*a>0.0012:
    if not step(0,0,a):pass
    near(tl,0.5,0.23-tz,0.08)
  x=read("spot.x");ly=read("spot.y");z=read("spot.z")-0.4*(lab<82)
  for j in range(3):
    if ly<0.19-tz or ly>0.27-tz:
      if not step(0,ly-0.23+tz+0.03*(2*(ly>0.27-tz)-1)):pass
      x=x-read("step.dx");ly=ly-read("step.dy")
      if spot(tl):x=read("spot.x");ly=read("spot.y")
  x=x+0.08-0.02*i*(lab>81)*(x>0.5)-q;z=z+0.075+oz;w=ly-r-0.01-0.01*(i*(lab<82)+1-i)+0.035*(lab>81);w=w+(0.19-w)*(w<0.19)*(lab<82)
  if not reach("left",x-0.17-(x-0.52)*(x>0.52)*(lab>81),w,z,1,0,0,0,1,0):reach("left",x-0.25,w,z+0.03,1,0,0,0,1,0)
  if not move_hand("left",x,w,z):
    if x-read("left.x")>0.01:pull("left",1,0,0,10,x-read("left.x"),give=0)
  pull("left",0,0,-1,6,0.1,give=0)
  if read("pull.force")<3:
    if not move_hand("left",read("left.x"),read("left.y"),read("left.z")-0.05):pass
  open_hand("left");wait(0.3)
  if not move_hand("left",read("left.x")-0.15,read("left.y"),read("left.z")+0.01):pull("left",-1,0,0,10,0.15,give=0)
  withdraw("left")
  if i==0:
    a=0
    if spot(tl):a=atan2(read("spot.ny"),read("spot.nx"))
    if a*a>0.0012:
      if not step(0,0,a):pass
    side(0.225+0.25*k+ly)


def side(d):
  for j in range(6):
    if d*d>0.0036:
      g=d
      if d*d>0.2:g=0.45*d/sqrt(d*d)
      if not step(0,g):pass
      d=d-read("step.dy")


def near(t,w,yt,o):
  for j in range(3):
    d=o;b=0;q=1
    if spot(t):x=read("spot.x");d=read("spot.y")-yt;b=(x-w)*(w>0)*(1-(w>0.4)*(0.5+0.5*(x<w+0.05))-0.4*(w<0.4)*(x>w));q=0
    if q+d*d+b*b<=0.0012:return
    if d*d+b*b>0.0012:
      if not step(b*(1-(w<0.4)*(d*d>0.0012)),d):pass
  spot(t)
'''

#: The jobs, each its grader (`attempts.GRADERS`) and its procedure.
JOBS = {"lid": LID, "drawer": DRAWER, "door": DOOR, "place": PLACE, "table": TABLE,
        "shelf": SHELF}


def source(job: str, k: int = 0) -> str:
  """`job`'s procedure as flown at its set-out `k` -- a scene's
  (`SCENES`) -- the shelf's with the labels its set-out names (the job's
  word, as an offer states it)."""
  if job in SCENES:
    return SCENES[job][1]
  if job == "shelf":
    from pluggybot.evaluation import counter as ct
    return SHELF % ct.stock_set_out(k).slots
  return JOBS[job]
#: A SCENE to watch, never a job the check grades (#554): the lid's
#: procedure with its lift STOPPED ON ITS HAND'S HEIGHT 10 cm up -- the lid
#: held part open for two seconds -- then lifted the rest of the way and set
#: down as the lid's is (`scripts/hands_check.py --view until`): its job's
#: set-out and grader, and its procedure. `watch` (stage 2): the same, the
#: knob's tag WATCHED from the spot on, and its lift stopped where the watch
#: puts the knob 6 cm higher than as it was gripped -- where its own jaws
#: hide the tag the reading never passes, and the lift goes its whole way.
SCENES = {"until": ("lid", LID.replace(
  '    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27)\n',
  '    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27, until="left.z", above=z0 + 0.1)\n'
  '    wait(2)\n'
  '    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27 - read("pull.moved"))\n')),
  "watch": ("lid", LID.replace('  spot(53)\n', '  spot(53)\n  watch(53)\n').replace(
    '    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27)\n',
    '    k = read("watch.z")\n'
    '    pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27, until="watch.z", above=k + 0.06)\n'
    '    if read("pull.stopped") > 0.5:\n'
    '      wait(2)\n'
    '      pull("left", 0.35 * fx, 0.35 * fy, 1, 20, 0.27 - read("pull.moved"))\n'))}
#: ...and the table's, flown in #560's stage (c2) (SimNotes, "The table's
#: put-downs, measured before the waist is built"): `table_rim30`, the
#: plate's wall gripped 30 deg round from straight ahead -- its far left, the
#: fingers leaning ahead and to the right, the jaws along the wall's radius
#: the way the planner took them -- and laid from where the robot stands, the
#: fork as the job's from 9 cm to the right; `table_rim30_t60`, the same, the
#: fork gripped 60 deg down, its jaws the other way round, from 6.5 cm.
_TABLE_RIM30 = (('  px = read("spot.x") - k\n', '  px = read("spot.x") - k + 0.0701\n'),
                ('  py = read("spot.y") - m + 0.081\n', '  py = read("spot.y") - m + 0.0405\n'),
                ('k - 0.19, m + 0.081 + 0.079 * i,', 'k - 0.1199 - 0.0701 * i, m + 0.0405 + 0.1195 * i,'))
_TABLE_GRIP = '0.767 - 0.005 * i, 0.707, 0, -0.707, 0, 1, 0)'
_TABLE_STEP = '    step(0, 0.43 * i - 0.26)\n'


def _table_scene(grip: str, step: str) -> str:
  out = TABLE_STANCES
  for old, new in (*_TABLE_RIM30, (_TABLE_GRIP, grip), (_TABLE_STEP, step)):
    out = out.replace(old, new)
  return out


SCENES.update({
  "table_rim30": ("table", _table_scene(
    "0.767 - 0.005 * i, 0.707 - 0.3536 * (1 - i), -0.6124 * (1 - i), -0.707, "
    "-0.866 * (1 - i), 1 - 1.5 * (1 - i), 0)", "    step(0, -0.09 * i)\n")),
  "table_rim30_t60": ("table", _table_scene(
    "0.767 + 0.0014 * i, 0.3536 + 0.1464 * i, -0.6124 + 0.6124 * i, -0.7071 - 0.1589 * i, "
    "-0.866 + 0.866 * i, -0.5 - 0.5 * i, 0)", "    step(0, -0.065 * i)\n"))})
#: The set-outs each job is flown at, and the bar, written before the check
#: flew: a job passes when its procedure passes at least `BAR` of them
#: (#493's gate: 6 of 8).
SET_OUTS, BAR = tuple(range(8)), 6


def facts() -> st.WorldFacts:
  """What the check's procedures are validated against: the hands'
  vocabulary as a mind is given it (`steps.hand_facts`)."""
  return st.hand_facts()


def compiled(job: str, k: int = 0) -> lang.Procedure:
  """`job`'s procedure at its set-out `k` -- or a scene's (`SCENES`) --
  compiled against `facts()` (raises `Refused` with every reason)."""
  return lang.compile_procedure(source(job, k), facts())


def fly(job: str, k: int, info: dict | None = None, noise: str | None = None) -> dict:
  """`job`'s procedure flown once at its set-out `k` and graded: the
  attempt's row (`attempts.attempt`), the run's verbs beside it (`ran`:
  each verb, whether it was ok and why it stopped). The set-out draws its
  own noise, or `noise`'s (`attempts.builder`)."""
  f = facts()
  out: dict = {}
  row = at.attempt(at.Task(chest=k, body="humanoid", mount="counter", grader=job),
                   ep.procedure_skill(compiled(job, k), f, out), 1,
                   info={"process": "hand-written", "skill": job, **(info or {})},
                   build=at.builder(noise))
  ran = out.get("ran") or {}
  row["ran"] = {"ok": ran.get("ok"), "stopped": ran.get("stopped"),
                "steps": [{k_: s.get(k_) for k_ in ("verb", "ok", "why", "reason", "moved",
                                                    "force")
                           if s.get(k_) is not None} for s in ran.get("steps", [])]}
  return row


def tally(rows: list[dict]) -> dict:
  """Per job: the check's set-outs flown, those passed -- each once, a
  set-out flown twice passing only where both did -- and whether it met
  `BAR` (over `SET_OUTS` alone; a job flown at fewer is `None`, never a
  fail)."""
  out: dict = {}
  for job in JOBS:
    ok: dict[int, bool] = {}
    for r in rows:
      k = r.get("task", {}).get("chest")
      if r.get("task", {}).get("grader") == job and "verdict" in r and k in SET_OUTS:
        ok[k] = ok.get(k, True) and bool(r["verdict"]["ok"])
    flown, passed = sorted(ok), sorted(k for k, v in ok.items() if v)
    met = (len(passed) >= BAR) if set(SET_OUTS) <= set(flown) else None
    out[job] = {"flown": flown, "passed": passed, "met": met}
  return out
