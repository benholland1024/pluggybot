"""Hand-written solutions to the challenges -- ladder A of issue #264, on
legs (#407).

A feature nobody uses is indistinguishable from one that cannot be
completed. Each solution here is written in the robot's OWN vocabulary (a
procedure the language accepts) from what the offer tells it -- the area's
address and its directions -- and passes the feature's own grader through
`scoring.evaluate`, the door the robot's attempt goes through: so "the
tower can be built" is a flown fact (`scripts/solve.py --feature tower`)
and a robot that never builds it is a finding about the robot, not the
world. The rover's are at `rover-final`.

⚠ NEVER SHOWN TO THE MIND. Nothing under `mind/` imports this module and
`tests/test_solutions.py` walks the tree to keep it so: a solution in the
prompt is the prompt handing over the answer, and ladder B (a prompted
local flight, reported into the issue) measures whether a model finds one
unaided.

Each source is a statement of what the world requires, read that way:

- THE TOWER finds its area's tags first, empty-handed (a body carrying a
  tool turns at `legs.body.W_CARRY`, and a search with the pen aboard ran
  out of time, #406) and with the language's whole patience (the workshop's
  corner is the house's far end, and from a fresh map 300 s did not reach
  it), then takes the claw and stacks: each `pick` and
  `place` finds its cube by the cube's own tag in front of the area's
  tags, and the claw is hung back -- walking away from the tower, which the
  grade asks it to leave alone.
- THE BENCH is a balance, on the arm: the elbow motor's torque as its
  driver reports it (`read("elbow.torque")`) with the empty claw, the known
  cube and the unknown, each held at the same pose -- where `pick` leaves
  it -- and each the mean of ten readings against the current sense's
  noise. The mass is the known one's times the ratio of what each added,
  so the arm's lengths, its own weight and the claw's cancel. The finding
  itself is the mind's to write (`record`, off the locals History shows
  it); the script and the test write it from `mass` the way a mind would.
"""

#: The tower (challenge/stack.py; offered as `stack_tower`): the workshop
#: corner's tags found round the house's address, the claw, two picks and
#: two places, the claw hung back.
TOWER = '''def tower():
  budget(steps=40, seconds=1800)
  find(42, -5.3, 2.3, 600)
  fetch("module_claw")
  pick(21)
  place(20)
  pick(22)
  place(21)
  stow()
'''

#: The bench (challenge/bench.py; offered as `find_mass`): the bench's tags
#: found round the facility's address, the claw, a tare, the known cube and
#: the unknown weighed at the pose a pick leaves the arm in, each set back
#: down, the claw hung back.
WEIGH = '''def weigh():
  budget(steps=80, seconds=1800)
  find(44, 25.2, -2.4, 600)
  fetch("module_claw")
  wait(2)
  tare = 0
  for i in range(10):
    wait(0.2)
    tare = tare + read("elbow.torque") / 10
  pick(23)
  wait(2)
  known = 0
  for i in range(10):
    wait(0.2)
    known = known + read("elbow.torque") / 10
  put()
  pick(24)
  wait(2)
  unknown = 0
  for i in range(10):
    wait(0.2)
    unknown = unknown + read("elbow.torque") / 10
  put()
  mass = 0.1 * (unknown - tare) / (known - tare)
  stow()
'''

#: THE HUMANOIDS' HAND JOBS (#559, stage 3; SimNotes, "The humanoids' first
#: jobs", has what flying took): each TWO procedures, one would pass the
#: language's cap (`lang.MAX_SOURCE_CHARS`). The walk there finds the row's
#: NEIGHBOUR first (a find stops at its tag's first sight, and one tag alone
#: left the walk's `face` 17-20 deg off the wall), stands 1.3 m out of the
#: wall's face (nearer, the planner's margin round the counter takes the
#: goal), turns ONCE off the handle's tag read from there to `STANCE`, steps
#: to 30 cm short, then looks again and steps the last 30 cm (a step after a
#: turn falls a tenth short; a short one overshoots by up to 3.6 cm). Then
#: the check's own procedure (`evaluation.hands_check.JOBS`; the drawer's is
#: `DRAWER_JOB`).
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
  step(x - %(ahead)g - 0.3, y - %(left)g, t)
  spot(%(handle)d)
  step(read("spot.x") - %(ahead)g, read("spot.y") - %(left)g)
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
  from pluggybot.rack import tags
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

#: The procedure that discharges each challenge kind, by kind.
PROCEDURES = {"stack_tower": TOWER, "find_mass": WEIGH, "open_drawer": DRAWER,
              "open_door": DOOR, "place_cube": CUBE, "open_chest": CHEST}


#: The challenges by their place (`world_config`'s `tower` and `bench`, and
#: its counter's props).
TARGETS = {"stack_tower": "workshop", "find_mass": "lab_bench", "open_drawer": "drawer",
           "open_door": "door", "place_cube": "cube", "open_chest": "chest"}


def job_routine(life, kind: str, world: str):
  """A challenge as a robot on legs does it: offered as the cadence offers
  it (`TaskProducer._build`: the area's terms, the bench's unknown drawn
  and set out as the offer lands), claimed, the hand-written solution
  defined in a library of its own and run as the errand a mind's
  `procedure:<name>` runs, the bench's finding recorded off its `mass` as a
  mind writes one off the locals History shows it, `done`, and the grade
  on the seam -- the challenge's own predicate through `scoring.evaluate`.
  What `scripts/solve.py` flies and `scripts/energy_spike.py` prices."""
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
  sources = PROCEDURES[kind] if isinstance(PROCEDURES[kind], tuple) else (PROCEDURES[kind],)
  runs = []
  for source in sources:
    name = lang.compile_procedure(source, facts).name
    library.define(name, source)
    errand = errand_from(ov.Decision(action=f"procedure:{name}"), world, library=library)
    result = yield from life.run_errand_routine(errand)
    run = result.get("procedure") or {}
    runs.append(run)
    if not run.get("ok"):
      break
  if kind == "find_mass":
    mass = (run.get("locals") or {}).get("mass")
    if mass is not None:
      life._reconsider(ov.Decision(action="idle", record={
        "quantity": "unknown mass", "value": round(float(mass), 4), "unit": "kg",
        "method": "the elbow motor's torque, against the known cube", "topic": "mass_bench"}))
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
