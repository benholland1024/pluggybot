"""The served process's memory watchdog (issue #349): the trigger pinned with
made-up series, the dumps with a fake memory reader, the crash stack in a
child process. Nothing here flies a robot."""

import io
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from pluggybot.telemetry import vitals
from pluggybot.telemetry.vitals import RunawayRule, Watchdog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _tracing_is_left_as_found():
  """A test that fails between an onset and its report would leave
  tracemalloc on for every later test in the worker, ~6x slower."""
  was = tracemalloc.is_tracing()
  yield
  if tracemalloc.is_tracing() and not was:
    tracemalloc.stop()


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
  # ...and a slower box, whose build and carry-on are spread over three
  # minutes -- which a warm-up of two would fire on (shown).
  slow = _day(RESTING, [212.6, 400.0, 600.0, 788.5])
  assert _feed(RunawayRule(), slow) == []
  assert _feed(RunawayRule(warmup=2), slow) != []


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


def test_an_episode_ends_after_two_calm_minutes_and_not_one():
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
    for i in range(len(series)):
      if dog.tick():
        break
      assert tracemalloc.is_tracing() == was_tracing, "never before the onset"
    assert i == 6 and tracemalloc.is_tracing()
    for _ in range(i + 1, len(series)):
      assert not dog.tick(), "once per episode"
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


def test_tracing_is_off_before_the_report_is_built(tmp_path, monkeypatch):
  """Traced, the statistics trace themselves: ~20x the cost (MEASURED, 3 s
  against 0.2 s per million traces), with the pair ~6x slower throughout."""
  seen = []
  real = tracemalloc.Snapshot.statistics
  monkeypatch.setattr(tracemalloc.Snapshot, "statistics", lambda self, *a, **kw:
                      seen.append(tracemalloc.is_tracing()) or real(self, *a, **kw))
  out = open(tmp_path / "log.txt", "w")
  dog = _runaway_dog(out, _cumulative(800.0, [2.0] * 4 + [115.0] * 3))
  while not dog.tick():
    pass
  dog.report()
  out.close()
  assert seen == [False]


def test_a_report_that_fails_still_stops_the_tracing(tmp_path, monkeypatch):
  """Left on, the pair would run ~6x slower until the process ended."""
  def broken(self, *a, **kw):
    raise RuntimeError("the report broke")
  monkeypatch.setattr(tracemalloc.Snapshot, "statistics", broken)
  out = open(tmp_path / "log.txt", "w")
  dog = _runaway_dog(out, _cumulative(800.0, [2.0] * 4 + [115.0] * 3))
  while not dog.tick():
    pass
  assert tracemalloc.is_tracing()
  try:
    dog.report()
  except RuntimeError:
    pass
  out.close()
  assert not tracemalloc.is_tracing() and not dog._tracing


def test_a_stream_without_a_descriptor_sends_the_stacks_to_stderr(capfd):
  """faulthandler writes to a file descriptor; without one the onset used
  to fail after latching the episode, and no report ever came."""
  out = io.StringIO()
  dog = _runaway_dog(out, _cumulative(800.0, [2.0] * 4 + [115.0] * 3))
  while not dog.tick():
    pass
  assert tracemalloc.is_tracing()
  dog.report()
  text = out.getvalue()
  assert text.count("the stacks went to stderr") == 2
  assert "allocated since the onset is still held" in text
  assert "most recent call first" in capfd.readouterr().err


def test_closing_inside_a_trace_window_reports_before_the_exit_line(tmp_path):
  """A crash while the runaway is being traced is the moment the report is
  for, and the exit line comes after it."""
  out = open(tmp_path / "log.txt", "w")
  dog = _runaway_dog(out, _cumulative(800.0, [2.0] * 4 + [115.0] * 3))
  while not dog.tick():
    pass
  dog.close(vitals.why_of(MemoryError()))
  out.close()
  text = (tmp_path / "log.txt").read_text()
  assert text.count("allocated since the onset is still held") == 1
  assert text.rstrip().splitlines()[-1].startswith(
    "vitals: exiting -- MemoryError; rss")
  assert not tracemalloc.is_tracing()


def test_the_exit_line_says_why(tmp_path):
  out = open(tmp_path / "log.txt", "w")
  was_tracing = tracemalloc.is_tracing()
  dog = Watchdog(out=out, sample_s=3600.0, read_rss=lambda: 812.0).start()
  deadline = time.monotonic() + 10.0
  while dog.ticks < 1 and time.monotonic() < deadline:
    time.sleep(0.005)
  assert tracemalloc.is_tracing() == was_tracing, \
    "tracing starts at an onset, never at boot: from boot the pair ran 5.8x slower"
  dog.close(vitals.why_of(KeyError("fill")))
  out.close()
  text = (tmp_path / "log.txt").read_text()
  assert text.startswith("vitals: rss 812 MiB"), "a first sample at start"
  assert "vitals: exiting -- KeyError: 'fill'; rss 812 MiB" in text
  assert not dog._thread.is_alive()
  assert vitals.why_of(None) == "the run ended"
  assert vitals.why_of(SystemExit(2)) == "exit 2"
  assert vitals.why_of(KeyboardInterrupt()) == "KeyboardInterrupt"
  assert vitals.why_of(None, "SIGTERM") == "stopped by SIGTERM"


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
