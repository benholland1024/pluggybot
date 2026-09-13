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
- **An integration** — "the refusal produces a charge and a completed errand
  on real physics". Fly it, stop it on the claim (`stop_when`), and if the
  rule is already pinned put it behind `--endurance` with a comment naming
  the pin above the mark.
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

1. **Stub the routine, not the twin.** Every manoeuvre is a generator
   (`pluggybot/tick.py`); a test stubs it with `tick.result(value)` —
   `life.mission.drive_to_routine = lambda *a, **kw: tick.result(True)` —
   and records the call if the claim is "it drove home". Stubbing the
   blocking twin does nothing (the loop calls the routine), and the fence in
   `tests/test_tick.py` catches an undriven call.
2. **Stub the opening spin.** `_day_routine` begins with
   `mission._spin_routine()`, ~7 s of real physics that seeds the map and
   says nothing about any branch below it: `life.mission._spin_routine =
   lambda *a, **kw: tick.result(None)`.
3. **Shrink the slice.** A loop that idles in `WAIT_FOR_WORK_S` (5 s) slices
   costs 5 s per iteration however small the budget: `monkeypatch.setattr(lc,
   "WAIT_FOR_WORK_S", 0.2)`. The slice length is never the claim.
4. **Place the belief, not the body.** Where the robot THINKS it is is the
   reckoner (`life.mission.swap.reckoner.x/.y`); where it IS is `qpos` at
   `handle.qpos_adr(model)` followed by `mj_forward`. Set whichever the branch
   reads and skip the drive that would have got it there.
5. **Ask the world's data, not a flight.** An `Activity` is sensed off
   `data` — write both robots' poses in and call `sense()` (the encounter and
   referee tests). An evaluator reads a dict of measurements — hand it one.
6. **Let a vendored fixture be the integration proof.** `protocol/*.jsonl.gz`
   are real flown missions, read in a second; a test over one of them proves
   the wire shape for free. That is what let the live pair recording
   (226 s) move behind `--endurance`.
7. **Stop on the claim.** `HubLifecycle.stop_when` / `MissionAborted` from a
   step hook the moment the assertion is decidable; make the predicate the
   SUCCESS condition so a regression still runs the long path and fails on
   the same line.
8. **One flight, both halves.** Two tests that fly the same mission with the
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
`endurance` mark names the fast test that pins its rule. A test with neither
and a mission in it is the drift the budget exists to catch.
