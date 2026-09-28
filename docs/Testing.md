# Testing — how to pin a rule without paying for a mission

The constraints are in `CLAUDE.md` ("Working style", "Tests"); this is the
recipe behind them, written after the suite went 18 min → 7 min in two
passes (issues #54, #158, and the 2026-09-13 pass in #182). Read it before
writing a test that flies anything.

## 1. Decide what the test is FOR before choosing how to fly it

Every debugged failure becomes an assertion, and there are exactly three
kinds:

- **A rule** — an inequality, a branch order, one line of wiring. Pin it
  with a direct call, a fake press, a stubbed drive. Milliseconds. This is
  the test that must exist; the other two are optional.
- **An integration** — "the refusal produces a charge and then the errand".
  If it is the loop's BOOKKEEPING — the mind, the economy, the record, the
  wire, what the body is sent to do and in what order — it is a day on the
  stub, in the default run (§2.1). If it is PHYSICS — a dock, a walk, a
  claw, a body in a world — fly it, stop it on the claim (`stop_when`), and
  with its rule pinned put it behind `--endurance`, naming what it guards
  (§4).
- **A premise** — the old defect still reproduces with the fix bypassed
  (`--blind`, `--no-brake`). `slow`; it cannot catch a regression while you
  iterate, it only stops the premise rotting.

A flown test whose rule is NOT pinned elsewhere is not an endurance
candidate, whatever it costs: `test_a_question_is_asked_answered_and_
graded_twice_unattended` (247 s) stays in the default run because the charge
between two fetches of the same tool is a path no fast test has, and it is
the path that found `test_rack_belief`'s defect.

## 2. The cheap levers, in the order to reach for them

Sim-seconds are the only cost that matters; model compilation is 12–28 ms
and wall clock tracks the machine. So:

1. **Build it on a stub body.** A claim about the loop's BOOKKEEPING — the
   mind, the economy, the record, the wire — needs no rover:
   `stub_life(world, **kw)` (`tests/test_body.py`) builds the lifecycle on a
   `StubBody` (issue #380), a floor and no robot, ~25 ms against ~350 ms. Its
   manoeuvres arrive at once, its senses answer what the test set
   (`holding`, `on_charger`, `attitude`), time passes only where it holds, and
   it outlives the rover. Two robots are two stubs on ONE world
   (`StubBody(model, data, handle=SECOND)`) ticked by `run_pair`. A claim
   about the body or the world's geometry (a camera's pose, a module on the
   floor, the built rail) stays on the rover.
   ⚠ A stub manoeuvre takes NO sim time, so a step hook never sees one:
   wrap the body's routines and read what the loop commanded, with what
   (`test_a_starving_robot_still_charges_navigates_and_stows`). And a gate
   that refuses forever SPINS at one instant instead of failing — the clock
   that would end the day never moves — so end it from the spy with
   `MissionAborted` and fail on that. A stub waiting sim-hours may run a
   coarser `<option timestep>`: nothing it steps is physics.
   ⚠ A LATE ANSWER IS SIM TIME on the stub: the loop holds think-slices
   while a call is out, so every 100 ms the answering thread waits is ~40
   sim-seconds. Stop on the claim — a count, a result, the point where it
   is decided either way — and give the budget room: a budget near the
   claim is a race the loaded suite loses (the seeded day did, 2026-09-28),
   and on the stub the long path a regression takes can run away rather
   than fail.
2. **Stub the routine, not the twin.** Every manoeuvre is a generator
   (`pluggybot/tick.py`); a test stubs it with `tick.result(value)` —
   `life.body.go_to_routine = lambda *a, **kw: tick.result(True)` — and
   records the call if the claim is "it drove home". On the rover,
   `life.body.mission.drive_to_routine` stubs its own drives too (inside a
   swap or a dock approach). Stubbing the blocking twin does nothing (the
   loop calls the routine), and the fence in `tests/test_tick.py` catches an
   undriven call.
3. **Stub the opening look.** `_day_routine` begins with
   `body.look_around_routine()`, the rover's ~7 s spin that seeds the map and
   says nothing about any branch below it: `life.body.look_around_routine =
   lambda *a, **kw: tick.result(None)` (`life.body.mission._spin_routine` for
   the rover's own retries too).
4. **Shrink the slice.** A loop that idles in `WAIT_FOR_WORK_S` (5 s) slices
   costs 5 s per iteration however small the budget: `monkeypatch.setattr(lc,
   "WAIT_FOR_WORK_S", 0.2)`. The slice length is never the claim.
5. **Place the belief, not the body.** Where the robot THINKS it is is its
   estimate (the rover's reckoner, `life.body.mission.swap.reckoner.x/.y`; a
   stub's `x`/`y`); where it IS is `qpos` at
   `handle.qpos_adr(model)` followed by `mj_forward`. Set whichever the branch
   reads and skip the drive that would have got it there.
6. **Ask the world's data, not a flight.** An `Activity` is sensed off
   `data` — write both robots' poses in and call `sense()` (the encounter and
   referee tests). An evaluator reads a dict of measurements — hand it one.
7. **Let a vendored fixture be the integration proof.** `protocol/*.jsonl.gz`
   are real flown missions, read in a second; a test over one of them proves
   the wire shape for free, and a recorder driven by hand — its events
   through the pair's own doors, `step_hook` for a frame — proves the wiring
   with nothing stepped.
8. **Stop on the claim.** `HubLifecycle.stop_when` / `MissionAborted` from a
   step hook the moment the assertion is decidable; make the predicate the
   SUCCESS condition so a regression still runs the long path and fails on
   the same line.
9. **One flight, both halves.** Two tests that fly the same mission with the
   same row and differ only in the client are one test with the stronger
   client (the two interrupt flights, #182).

## 3. Measure before believing a number

- `MUJOCO_GL=egl uv run pytest -q --durations=40` under `-n auto` ranks the
  suite but inflates every figure ~25 % (memory-bandwidth contention);
  `-n0 --durations=1` on one test is the real cost.
- Before believing a slower SUITE, time one unchanged test `-n0` on both
  trees, INTERLEAVED (A B A B). Identical per-test times mean the delta is
  xdist scheduling, not the repo — the 2026-09-13 bump ran 10:24 vs 9:39 on
  two trees whose common test timed 2.7 s on both.
- The floor is the longest single test; no worker count beats it. Reordering
  the collection does not help (#158).

## 4. What to write above the mark

A `slow` mark says why the test cannot be shortened past what it does. An
`endurance` flight is also `slow`; the comment above it names the fast test
that pins its rule, and `when=` names the paths its physics stands on — a
directory, a file or a file prefix. That is when it flies:
`--endurance-changed` selects the flights a change touches, and
`tests/test_endurance.py` fails on a path that has moved. Leave out what is
pinned on the stub (the loop's side, `lifecycle.py`), or every change flies
it. A test with neither mark and a mission in it is the drift the budget
exists to catch.
