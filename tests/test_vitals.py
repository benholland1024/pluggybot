"""The served process's memory watchdog (issue #349): the trigger pinned with
made-up series, the dumps with a fake memory reader, the crash stack in a
child process. Nothing here flies a robot."""

import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from pluggybot.telemetry import vitals
from pluggybot.telemetry.vitals import RunawayRule, Watchdog

ROOT = Path(__file__).resolve().parents[1]


def _feed(rule, series, minute=60.0):
  """One sample a minute; the indices at which the rule saw an onset."""
  return [i for i, mib in enumerate(series) if rule.sample(i * minute, mib)]


def _cumulative(start, steps):
  out = [start]
  for d in steps:
    out.append(out[-1] + d)
  return out


# The deployed process of 2026-09-25 (3518953), docker stats once a minute:
# 212 MiB one second in, 788 a minute later, then the busiest resting minutes
# on record (+23 and +20 back to back) among ordinary ones.
STARTUP = [212.6, 788.5]
RESTING = [0.5, 0.5, 3.7, 1.0, 4.0, 1.9, 22.9, 19.6, 0.3, 0.1, 2.7, 2.7] * 5


def _day(steps, startup=STARTUP):
  return startup[:-1] + _cumulative(startup[-1], steps)


def test_the_resting_rate_and_the_startup_never_fire():
  assert _feed(RunawayRule(), _day(RESTING)) == []
  # ...and a slower box, whose build and carry-on straddle two samples: two
  # fast minutes in a row, which is what the warm-up is for.
  assert _feed(RunawayRule(), _day(RESTING, [212.6, 500.0, 788.5])) == []


def test_the_threshold_keeps_its_margin_on_both_sides():
  """Why 50 and not the issue's example 30: the busiest resting minutes read
  +23 and +20 back to back, which 30 clears by only 1.3x, and a rule just
  under them fires on a process at rest (shown). 50 is ~2x from the resting
  pair and ~2x from the measured runaway; moving it is a decision."""
  assert vitals.RUNAWAY_MB_PER_MIN > 22.9 * 2
  assert 115.0 > vitals.RUNAWAY_MB_PER_MIN * 2
  assert _feed(RunawayRule(mb_per_min=19.0), _day(RESTING)) != []


def test_a_runaway_fires_once_at_its_second_minute():
  """The a803c83 deaths: ~115 MiB a minute for eight minutes to the cap."""
  series = _day(RESTING[:10] + [115.0] * 8)
  first_fast = len(STARTUP) + 10             # the sample closing minute one
  assert _feed(RunawayRule(), series) == [first_fast + 1]


def test_one_fast_minute_is_not_a_runaway():
  assert _feed(RunawayRule(), _day(RESTING[:10] + [300.0] + RESTING[:10])) == []


def test_an_episode_ends_only_after_it_has_been_calm_as_long_as_it_ran():
  one_calm = [115.0] * 3 + [2.0] + [115.0] * 3
  assert len(_feed(RunawayRule(), _day(RESTING[:6] + one_calm))) == 1, \
    "a dip of one minute is the same runaway"
  two_calm = [115.0] * 3 + [2.0, 2.0] + [115.0] * 3
  assert len(_feed(RunawayRule(), _day(RESTING[:6] + two_calm))) == 2


def test_the_rate_is_per_minute_of_wall_clock_not_per_sample():
  """A sample that came late (the GIL held by a long render) is not a spike."""
  rule = RunawayRule(warmup=0)
  rule.sample(0.0, 800.0)
  rule.sample(180.0, 900.0)                  # 100 MiB over three minutes
  assert abs(rule.rate - 100.0 / 3) < 1e-9


def _runaway_dog(out, series):
  clock = iter(i * 60.0 for i in range(10_000))
  values = iter(series)
  last = [series[-1]]

  def read():
    last[0] = next(values, last[0])          # flat once the series runs out
    return last[0]
  return Watchdog(out=out, read_rss=read, clock=lambda: next(clock))


def test_the_onset_dumps_the_stacks_and_the_report_names_the_allocation(tmp_path):
  was_tracing = tracemalloc.is_tracing()
  series = _cumulative(800.0, [2.0] * 4 + [115.0] * 6)
  out = open(tmp_path / "log.txt", "w")
  dog = _runaway_dog(out, series)
  dog.sim_time = lambda: 1234.5
  held = []
  try:
    onsets = [i for i in range(len(series)) if dog.tick()]
    assert onsets == [6] and tracemalloc.is_tracing()
    held.append(bytearray(4 << 20))          # what the runaway allocates
    dog.report()
  finally:
    out.close()
  text = (tmp_path / "log.txt").read_text()
  assert text.count("vitals: RUNAWAY") == 1
  assert text.count("(most recent call first)") >= 2, "faulthandler, twice"
  assert text.count("allocated since the onset is still held") == 1
  assert "test_vitals.py" in text and "bytearray(4 << 20)" in text, \
    "the report names where the growth was allocated"
  assert text.count("vitals: rss") == len(series)
  assert "t=1234.5 s" in text
  assert tracemalloc.is_tracing() == was_tracing, "tracing is the episode's only"


def test_the_loop_reports_once_per_episode(tmp_path):
  """The thread itself: an onset, the report `trace_s` later, and nothing
  more while the same runaway goes on."""
  series = _cumulative(800.0, [2.0] * 4 + [115.0] * 12)
  out = open(tmp_path / "log.txt", "w")
  dog = _runaway_dog(out, series)
  dog.sample_s = dog.trace_s = 0.001
  dog.start()
  deadline = time.monotonic() + 30.0
  while dog.ticks < len(series) + 3 and time.monotonic() < deadline:
    time.sleep(0.01)
  dog.close("the test ended")
  out.close()
  text = (tmp_path / "log.txt").read_text()
  assert dog.ticks >= len(series) + 3
  assert text.count("vitals: RUNAWAY") == 1
  assert text.count("allocated since the onset is still held") == 1
  assert text.count("every thread's stack, 0 s on") == 1
  assert "the watchdog failed" not in text


def test_the_exit_line_says_why(tmp_path):
  out = open(tmp_path / "log.txt", "w")
  dog = Watchdog(out=out, sample_s=3600.0, read_rss=lambda: 812.0).start()
  dog.close(vitals.why_of(KeyError("fill")))
  out.close()
  text = (tmp_path / "log.txt").read_text()
  assert text.startswith("vitals: rss 812 MiB"), "a first sample at start"
  assert "vitals: exiting -- KeyError: 'fill'; rss 812 MiB" in text
  assert not dog._thread.is_alive()
  assert vitals.why_of(None) == "the run ended"
  assert vitals.why_of(SystemExit(2)) == "exit 2"
  assert vitals.why_of(KeyboardInterrupt()) == "KeyboardInterrupt"


def test_a_segfault_prints_every_threads_stack():
  """A crash in MuJoCo or osmesa ends the process with nothing said unless
  faulthandler is on -- which `Watchdog.start` turns on."""
  code = ("import ctypes\n"
          "from pluggybot.telemetry.vitals import Watchdog\n"
          "Watchdog(sample_s=3600.0).start()\n"
          "ctypes.string_at(0)\n")
  env = {k: v for k, v in os.environ.items() if k != "PYTHONFAULTHANDLER"}
  env["PYTHONPATH"] = str(ROOT / "src")
  proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, env=env, timeout=60)
  assert proc.returncode < 0
  assert "Fatal Python error: Segmentation fault" in proc.stderr
  assert "most recent call first" in proc.stderr
