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

#: THE HUMANOIDS' HAND JOBS (#559, stage 3), on the storeroom's counter
#: (`home.storeroom`), each TWO procedures -- one would pass the language's
#: cap (`lang.MAX_SOURCE_CHARS`), as it would for a robot. The WALK THERE:
#: the tag beside the prop's in the row found round the facility's address,
#: then the prop's own, so the row's two give the wall's way (`mapping.
#: places.Fixture`) -- a find stops at its tag's first sight, and the
#: cube's alone, through the storeroom's door from the lobby, put the walk's
#: `face` 17 deg off; a walk to stand 1.3 m out of the wall's face and 0.2 m
#: to its right -- the counter's depth and a little more: nearer, the
#: planner's margin round the counter takes the goal -- and a turn to face
#: the wall; SQUARED to the prop by the face of the handle's tag it reads
#: from there -- its front, whose normal is level (one tag's way, not the
#: row's, left a robot 28 deg turned to the counter, and its arm's way home
#: dragged a jaw along it and toppled it); then two steps putting the
#: handle where the check's set-outs' middle had it (`storeroom.
#: HANDLE_FROM_SPOT`), the first 12 cm short of it -- a step after a turn
#: falls a tenth short (`step` from rest walks to within 3 cm; after `face`,
#: 0.39 m of 0.43) -- the rest off a fresh look, a stride or more. Then THE
#: JOB, the capability check's own procedure, unchanged (`evaluation.
#: hands_check.JOBS`).
APPROACH = '''def to_%(prop)s():
  budget(steps=16, seconds=900)
  find(%(beside)d, 25.2, -2.4, 600)
  find(%(row)d, 25.2, -2.4, 600)
  f = read("find.facing")
  drive_to(read("find.x") + 1.3 * cos(f) - 0.2 * sin(f), read("find.y") + 1.3 * sin(f) + 0.2 * cos(f), 180)
  face(f + 3.14159 - 6.28318 * (f > 0))
  spot(%(handle)d)
  a = atan2(read("spot.ny"), read("spot.nx"))
  while a > 0.79:
    a = a - 1.571
  while a < -0.79:
    a = a + 1.571
  step(0, 0, a)
  spot(%(handle)d)
  step(read("spot.x") - %(ahead)g - 0.12, read("spot.y") - %(left)g)
  spot(%(handle)d)
  step(read("spot.x") - %(ahead)g, read("spot.y") - %(left)g)
'''


def _hands_job(check: str, prop: str) -> tuple[str, str]:
  """A hand job's reference: the walk to its prop (`APPROACH`), then its
  check's procedure (`evaluation.hands_check.JOBS`)."""
  from pluggybot.evaluation import hands_check
  from pluggybot.home import storeroom
  from pluggybot.rack import tags
  handle = {"chest": tags.CHEST_TAG_IDS, "drawer": tags.DRAWER_TAG_IDS,
            "door": tags.DOOR_TAG_IDS, "cube": tags.CUBE_TAG_IDS}[prop][0]
  ahead, left = storeroom.HANDLE_FROM_SPOT[prop]
  i = storeroom.PROPS.index(prop)
  beside = storeroom.PROPS[i + 1 if i + 1 < len(storeroom.PROPS) else i - 1]
  return (APPROACH % {"prop": prop, "row": storeroom.TAGS[prop],
                      "beside": storeroom.TAGS[beside], "handle": handle,
                      "ahead": ahead, "left": left}, hands_check.JOBS[check])


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
