"""The served process's memory watchdog (issue #349): the trigger pinned with
made-up series, the dumps with a fake memory reader, the crash stack in a
child process. Nothing here flies a robot."""

import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

from pluggybot.telemetry import vitals
from pluggybot.telemetry.vitals import RunawayRule, Watchdog

ROOT = Path(__file__).resolve().parents[1]


def _feed(rule, series, minute=60.0):
  """One sample a minute; the rule's answer for each."""
  return [rule.sample(i * minute, mib) for i, mib in enumerate(series)]


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


def test_the_resting_rate_and_the_startup_never_fire():
  rule = RunawayRule()
  answers = _feed(rule, STARTUP[:1] + _cumulative(STARTUP[1], RESTING))
  assert set(answers) == {None}
  # ...and a slower box, whose build and carry-on straddle two samples: two
  # fast minutes in a row, which is what the warm-up is for.
  rule = RunawayRule()
  answers = _feed(rule, [212.6, 500.0] + _cumulative(STARTUP[1], RESTING))
  assert set(answers) == {None}


def test_the_threshold_keeps_its_margin_on_both_sides():
  """Why 50 and not the issue's example 30: the busiest resting minutes read
  +23 and +20 back to back, which 30 clears by only 1.3x, and a rule just
  under them fires on a process at rest (shown). 50 is ~2x from the resting
  pair and ~2x from the measured runaway; moving it is a decision."""
  assert vitals.RUNAWAY_MB_PER_MIN > 22.9 * 2
  assert 115.0 > vitals.RUNAWAY_MB_PER_MIN * 2
  low = RunawayRule(mb_per_min=19.0)
  assert "onset" in _feed(low, STARTUP[:1] + _cumulative(STARTUP[1], RESTING))


def test_a_runaway_fires_once_and_reports_a_minute_later():
  """The a803c83 deaths: ~115 MiB a minute for eight minutes to the cap."""
  rule = RunawayRule()
  steps = RESTING[:10] + [115.0] * 8
  answers = _feed(rule, STARTUP[:1] + _cumulative(STARTUP[1], steps))
  fired = [(i, a) for i, a in enumerate(answers) if a]
  first_fast = 1 + 10 + 1                   # the sample that closes minute one
  assert fired == [(first_fast + 1, "onset"), (first_fast + 2, "report")]


def test_one_fast_minute_is_not_a_runaway():
  rule = RunawayRule()
  steps = RESTING[:10] + [300.0] + RESTING[:10]
  assert set(_feed(rule, STARTUP[:1] + _cumulative(STARTUP[1], steps))) == {None}


def test_an_episode_ends_only_after_it_has_been_calm_as_long_as_it_ran():
  rule = RunawayRule()
  one_calm = [115.0] * 3 + [2.0] + [115.0] * 3
  answers = _feed(rule, STARTUP[:1] + _cumulative(STARTUP[1], RESTING[:6] + one_calm))
  assert answers.count("onset") == 1, "a dip of one minute is the same runaway"
  rule = RunawayRule()
  two_calm = [115.0] * 3 + [2.0, 2.0] + [115.0] * 3
  answers = _feed(rule, STARTUP[:1] + _cumulative(STARTUP[1], RESTING[:6] + two_calm))
  assert answers.count("onset") == 2 and answers.count("report") == 2


def test_the_rate_is_per_minute_of_wall_clock_not_per_sample():
  """A sample that came late (the GIL held by a long render) is not a spike."""
  rule = RunawayRule(warmup=0)
  assert rule.sample(0.0, 800.0) is None
  rule.sample(180.0, 900.0)                  # 100 MiB over three minutes
  assert abs(rule.rate - 100.0 / 3) < 1e-9


_LEAK: list = []


def _leaky_reader(series):
  """A memory reader that also allocates, so the report has a site to name."""
  values = iter(series)

  def read():
    _LEAK.append(bytearray(256 * 1024))
    return next(values)
  return read


def test_the_watchdog_dumps_the_stacks_and_the_allocations_once_per_episode(tmp_path):
  was_tracing = tracemalloc.is_tracing()
  series = _cumulative(800.0, [2.0] * 4 + [115.0] * 6)
  clock = iter(i * 60.0 for i in range(100))
  out = open(tmp_path / "log.txt", "w")
  dog = Watchdog(out=out, read_rss=_leaky_reader(series),
                 clock=lambda: next(clock))
  dog.sim_time = lambda: 1234.5
  try:
    answers = [dog.tick() for _ in series]
  finally:
    out.close()
    _LEAK.clear()
  text = (tmp_path / "log.txt").read_text()
  assert answers.count("onset") == 1 and answers.count("report") == 1
  assert text.count("vitals: RUNAWAY") == 1
  # faulthandler's own header, once at the onset and once a minute on
  assert text.count("(most recent call first)") >= 2
  assert text.count("every thread's stack, a minute on") == 1
  assert text.count("allocated since the runaway began") == 1
  assert "test_vitals.py" in text and "bytearray(256 * 1024)" in text, \
    "the report names where the growth was allocated"
  assert text.count("vitals: rss") == len(series)
  assert "t=1234.5 s" in text
  assert tracemalloc.is_tracing() == was_tracing, "tracing is the episode's only"


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


def test_a_segfault_prints_every_threads_stack(tmp_path):
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
