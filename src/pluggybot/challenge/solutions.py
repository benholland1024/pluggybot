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

#: The procedure that discharges each challenge kind, by kind.
PROCEDURES = {"stack_tower": TOWER, "find_mass": WEIGH}


#: The challenges by their place (`world_config`'s `tower` and `bench`).
TARGETS = {"stack_tower": "workshop", "find_mass": "lab_bench"}


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
  library = lib.Library(world_facts(world, rack=life.rack_inventory))
  name = lang.parse(PROCEDURES[kind]).name
  library.define(name, PROCEDURES[kind])
  errand = errand_from(ov.Decision(action=f"procedure:{name}"), world, library=library)
  result = yield from life.run_errand_routine(errand)
  run = result.get("procedure") or {}
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
                    for st in run.get("steps", [])],
          "locals": run.get("locals"), "secret": dict(secret),
          "grade": {"ok": bool(verdict.get("ok")), "reason": verdict.get("reason", ""),
                    "points": verdict.get("points", 0)}}
