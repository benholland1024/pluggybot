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
#: face read), its facing off the tag's edges; gripped from the front, the
#: fingers tilted up first (at the top of its arc, the hand that kept its
#: turn reaches about level) and the jaws along the hinge's line, at the
#: first tilt that holds; lifted up and a little back by a FORCE MOVE that
#: lets the hinge steer the hand along the arc; lowered toward the body and
#: down, again by force -- the lid set down by its hinge, not by a plan --
#: and let go only once the hand is back down.
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
  g = 0
  t = -0.26
  for i in range(6):
    if g == 0:
      open_hand("left")
      if reach("left", x, y, z, fx * cos(t), fy * cos(t), -sin(t), -fy, fx, 0):
        g = 2
      else:
        withdraw("left")
    if g == 2:
      if close_hand("left"):
        g = 1
      else:
        g = 0
        withdraw("left")
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
#: was last going.
DOOR = '''def open_door():
  budget(steps=60, seconds=66)
  ph = 0
  g = 3
  t = 0.0
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
      if reach("left", x, y, z, fx * cos(t), fy * cos(t), -sin(t), 0, 0, 1):
        g = 2
      else:
        withdraw("left")
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
'''

#: THE CUBE, out of reach where the robot stands: the mat's middle and the
#: cube's off their tags, the cube's turn off its edges; a STEP in, until
#: the cube is 0.32 m ahead -- nearer, the head camera no longer sees it, so
#: what it read is turned into the frame it stands in after the step
#: (`step.*`); gripped steeply from above-front, the jaws on two of its
#: faces, at the first tilt that reaches and holds; lifted straight up and
#: set down by force moves that hold their line (`give` 0: a cube set down
#: giving slid along what it touched), carried over the mat by where the
#: grip left it in the hand; and let go only once its set-down has pressed
#: on the mat.
PLACE = '''def place_cube():
  budget(steps=60, seconds=66)
  spot(57)
  mx = read("spot.x")
  my = read("spot.y")
  spot(56)
  x = read("spot.x") + 0.013 * read("spot.nx")
  y = read("spot.y") + 0.013 * read("spot.ny")
  z = read("spot.z") + 0.013 * read("spot.nz")
  a = atan2(read("spot.ey"), read("spot.ex"))
  step(x - 0.32)
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
  while a > 0.79:
    a = a - 1.571
  while a < -0.79:
    a = a + 1.571
  g = 0
  t = 1.05
  for i in range(4):
    if g == 0:
      open_hand("left")
      if reach("left", x, y, z + 0.003, cos(a) * cos(t), sin(a) * cos(t), -sin(t), -sin(a), cos(a), 0):
        g = 2
      else:
        withdraw("left")
    if g == 2:
      g = 0
      if close_hand("left"):
        g = 1
      else:
        withdraw("left")
    t = t + 0.26 - 0.77 * (i == 1)
  if g == 1:
    ox = x - read("left.x")
    oy = y - read("left.y")
    pull("left", 0, 0, 1, 20, 0.03, give=0)
    move_hand("left", mx - ox, my - oy, read("left.z"))
    for j in range(2):
      if g == 1:
        pull("left", 0, 0, -1, 4, 0.1, give=0)
        g = 1 + (read("pull.force") > 3)
  open_hand("left")
  withdraw("left")
'''

#: The jobs, each its grader (`attempts.GRADERS`) and its procedure.
JOBS = {"lid": LID, "drawer": DRAWER, "door": DOOR, "place": PLACE}
#: The set-outs each job is flown at, and the bar, written before the check
#: flew: a job passes when its procedure passes at least `BAR` of them
#: (#493's gate: 6 of 8).
SET_OUTS, BAR = tuple(range(8)), 6


def facts() -> st.WorldFacts:
  """What the check's procedures are validated against: the hands'
  vocabulary as a mind is given it (`steps.hand_facts`)."""
  return st.hand_facts()


def compiled(job: str) -> lang.Procedure:
  """`job`'s procedure, compiled against `facts()` (raises `Refused` with
  every reason)."""
  return lang.compile_procedure(JOBS[job], facts())


def fly(job: str, k: int, info: dict | None = None) -> dict:
  """`job`'s procedure flown once at its set-out `k` and graded: the
  attempt's row (`attempts.attempt`), the run's verbs beside it (`ran`:
  each verb, whether it was ok and why it stopped)."""
  f = facts()
  out: dict = {}
  row = at.attempt(at.Task(chest=k, body="humanoid", mount="counter", grader=job),
                   ep.procedure_skill(compiled(job), f, out), 1,
                   info={"process": "hand-written", "skill": job, **(info or {})})
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
