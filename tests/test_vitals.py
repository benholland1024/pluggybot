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
# The deployed process of 2026-11-12 19:17 UTC, an hourly restart into the
# grown world (#560): 173 MiB at its start and 1.3 GB four minutes later, its
# alarm set off at 19:21 with 4.6 MiB new at the onset. Made up between the
# two ends the log gave.
GROWN_START = [173.0, 470.0, 760.0, 1050.0, 1331.0]


def _day(steps, startup=STARTUP):
  return startup[:-1] + _cumulative(startup[-1], steps)


def _utc(hms):
  h, m, s = (int(x) for x in hms.split(":"))
  return 3600.0 * h + 60.0 * m + s


def _watched(tmp_path, rows):
  """The rows at which a Watchdog fires, fed `rows` of (UTC time, MiB, the
  sim clock or None before the world is built) as the deployed one read
  them: the clock is the server's `sim_time`, set once the world exists."""
  now = [rows[0]]
  out = open(tmp_path / "log.txt", "w")

  def read():
    now[0] = row = next(at)
    dog.sim_time = None if row[2] is None else (lambda: row[2])
    return row[1]
  at = iter(rows)
  dog = Watchdog(out=out, read_rss=read, clock=lambda: _utc(now[0][0]))
  try:
    return [i for i in range(len(rows)) if dog.tick()]
  finally:
    out.close()


def _ruled(rule, rows, clock=True):
  """The rows at which `rule` fires, the sim clock passed or left out."""
  return [i for i, (hms, mib, t) in enumerate(rows)
          if rule.sample(_utc(hms), mib, t if clock else None)]


def _at(rows, hms):
  return next(i for i, row in enumerate(rows) if row[0] == hms)


def _minutes(start, n, mib, t):
  """n made-up samples a minute apart from `start`, at one reading."""
  s = _utc(start)
  return [(f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d}",
           mib, t) for x in (s + 60.0 * i for i in range(n))]


# Three processes started INTO a pause, as the deployed log sampled them: the
# world built and paused at its first step (its renderers not yet built), then
# at the wake everything a fresh process builds in its first minutes.
# 2026-11-12, from its first line (`k5-alarm/live/vitals_1112_1113.txt`):
WAKE_1112 = ([("16:27:32", 174, None)]
             + _minutes("16:28:32", 32, 281, 216306.5)
             + [("17:00:32", 850, 216317.5), ("17:01:32", 904, 216367.6),
                ("17:02:52", 907, 216420.6), ("17:03:52", 1168, 216476.1),
                ("17:04:52", 1163, 216535.3), ("17:05:52", 1367, 216594.3),
                ("17:06:52", 1378, 216639.5), ("17:07:52", 1388, 216694.3),
                ("17:08:52", 1391, 216758.7), ("17:09:52", 1391, 216819.0),
                ("17:10:52", 1394, 216875.6)])
# 2026-11-14, woken from the 43-hour pause; the export begins at 15:59:58, so
# its first sample (no clock) and the 434 minutes before 15:59:58 at the
# export's first reading (+0.0) are made up (it was up 508 min at 17:12:48):
WAKE_1114 = ([("08:44:58", 174, None)]
             + _minutes("08:45:58", 435, 312, 228942.6)
             + [("16:00:58", 947, 228968.3), ("16:01:58", 1175, 229016.9),
                ("16:03:18", 1502, 229060.5), ("16:04:18", 1506, 229113.6),
                ("16:05:18", 1508, 229158.9), ("16:06:18", 1511, 229207.7),
                ("16:07:18", 1515, 229255.2), ("16:08:18", 1513, 229309.6),
                ("16:09:18", 1535, 229350.3), ("16:10:18", 1519, 229395.0)])
# 2026-11-15, woken after 10 h 43 min (`live/vitals_1115_start.txt`); its log
# begins at 08:40:04 reading +0.0, so its first two samples are made up:
WAKE_1115 = ([("08:38:04", 174, None)]
             + _minutes("08:39:04", 5, 318, 246646.9)
             + [("08:44:04", 334, 246664.0), ("08:45:04", 873, 246721.4),
                ("08:46:04", 1131, 246764.5)])
# 2026-11-14 17:12:50, a process begun with both robots charging; one robot's
# first walk built its two cameras' renderers a minute apart (17:32-17:33),
# then flat. Every line as sampled, the clock left out (the world ran).
FIRST_WALK = [
  ("17:12:50", 173), ("17:13:50", 853), ("17:14:50", 862), ("17:15:50", 872),
  ("17:16:50", 879), ("17:17:50", 883), ("17:18:50", 882), ("17:19:50", 882),
  ("17:20:50", 884), ("17:21:50", 886), ("17:22:50", 958), ("17:23:50", 958),
  ("17:24:50", 963), ("17:25:50", 970), ("17:26:50", 976), ("17:27:50", 977),
  ("17:28:50", 977), ("17:29:50", 977), ("17:30:50", 977), ("17:31:50", 979),
  ("17:32:50", 1185), ("17:33:50", 1424), ("17:35:10", 1439),
  ("17:36:10", 1440), ("17:37:10", 1440), ("17:38:10", 1478),
  ("17:39:10", 1469), ("17:40:10", 1485), ("17:41:10", 1484),
  ("17:42:10", 1490), ("17:43:10", 1490), ("17:44:10", 1490),
  ("17:45:10", 1483)]


def test_a_wake_into_a_world_that_never_ran_is_its_warm_up(tmp_path):
  """Three false alarms (11-12 17:01, 11-14 16:01, 11-15 08:46 UTC): a
  process started into a pause had built only its world, and built the rest
  at the wake -- each robot's camera renderers, ~210 MiB apiece -- which the
  rule read as a runaway, its warm-up spent on the pause's samples. A sample
  whose clock stood still, its memory calm, is a pause's, and the warm-up
  does not count it. A runaway once the woken world has warmed up still
  fires."""
  for rows in (WAKE_1112, WAKE_1114, WAKE_1115):
    assert _watched(tmp_path, rows) == [], rows[-1][0]
  # The old rule -- two minutes, the clock unread -- fired on each at the
  # live alarm's own sample (shown); the clock alone keeps all three quiet,
  # and three minutes alone do not keep the 43-hour pause's wake quiet.
  for rows, alarm in ((WAKE_1112, "17:01:32"), (WAKE_1114, "16:01:58"),
                      (WAKE_1115, "08:46:04")):
    assert _ruled(RunawayRule(minutes=2), rows, clock=False) == [_at(rows, alarm)]
    assert _ruled(RunawayRule(minutes=2), rows) == [], alarm
  assert _ruled(RunawayRule(), WAKE_1114, clock=False) == [
    _at(WAKE_1114, "16:03:18")]
  t = WAKE_1112[-1][2]
  runaway = [(f"17:{11 + i}:52", 1394 + 115.0 * (i + 1), t + 50.0 * (i + 1))
             for i in range(6)]
  assert _watched(tmp_path, WAKE_1112 + runaway) == [len(WAKE_1112) + 2]


def test_a_robots_first_walk_after_a_charge_is_a_step_not_a_runaway():
  """11-14 17:33 UTC: +207 and +239 MiB a minute as a robot that began the
  process charging walked off its station, then flat; the trace held 4.0
  MiB, because the step is MuJoCo's and osmesa's, not Python's. Two minutes
  over the rate are a step; a runaway is the third."""
  rule = RunawayRule()
  assert [i for i, (hms, mib) in enumerate(FIRST_WALK)
          if rule.sample(_utc(hms), mib)] == []
  rule = RunawayRule(minutes=2)
  assert [i for i, (hms, mib) in enumerate(FIRST_WALK)
          if rule.sample(_utc(hms), mib)] == [21], "two read it as a runaway"


def test_a_pause_mid_run_keeps_its_warm_up():
  """A world paused once it has run keeps what it built (11-12 21:00 to
  11-13 10:36 UTC: 1450 MiB, +0.0 MiB/min, every minute for 13.6 hours),
  so its wake is no warm-up: a runaway begun at the wake is reported at its
  third minute, as anywhere in a running world."""
  rule = RunawayRule()
  ran = [(60.0 * i, 1400.0 + i, 50.0 * i) for i in range(10)]
  paused = [(60.0 * i, 1409.0, ran[-1][2]) for i in range(10, 30)]
  woken = [(60.0 * i, 1409.0 + 115.0 * (i - 29), ran[-1][2] + 50.0 * (i - 29))
           for i in range(30, 36)]
  assert [i for i, row in enumerate(ran + paused + woken)
          if rule.sample(*row)] == [32]


def test_a_runaway_that_stops_the_world_is_judged_in_its_warm_up_too():
  """A runaway that stops the world stands its clock still, as a pause
  does; its memory does not, so the warm-up counts its samples, and one from
  boot is reported seven minutes in, as with the clock running."""
  boot = [(0.0, 174.0, None)] + [(60.0 * i, 174.0 + 115.0 * i, 30.0)
                                 for i in range(1, 9)]
  rule = RunawayRule()
  assert [i for i, row in enumerate(boot) if rule.sample(*row)] == [7]


def test_the_resting_rate_and_the_startup_never_fire():
  assert _feed(RunawayRule(), _day(RESTING)) == []
  # ...and a slower box, whose build and carry-on are spread over three
  # minutes -- which a warm-up of one would fire on (shown).
  slow = _day(RESTING, [212.6, 400.0, 600.0, 788.5])
  assert _feed(RunawayRule(), slow) == []
  assert _feed(RunawayRule(warmup=1), slow) != []


def test_the_grown_worlds_four_minute_start_never_fires_and_a_runaway_after_it_does():
  """11-12 19:21 UTC: a fresh process loading the grown world took four
  minutes, which a warm-up of three read as a runaway at two minutes over
  the rate (shown); a runaway once it has loaded is still caught at its
  third minute."""
  grown = _day(RESTING, GROWN_START)
  assert _feed(RunawayRule(), grown) == []
  assert _feed(RunawayRule(warmup=3, minutes=2), grown) != []
  late = _day(RESTING[:2] + [115.0] * 8, GROWN_START)
  first_fast = len(GROWN_START) + 2          # the sample closing minute one
  assert _feed(RunawayRule(), late) == [first_fast + 2]
  # ...and the window Webserver.md states: a load of six minutes passes, one
  # of seven does not -- a runaway from boot is reported seven minutes in
  ramp = [173.0 + 232.0 * i for i in range(8)]
  assert _feed(RunawayRule(), _day(RESTING, ramp[:7])) == []
  assert _feed(RunawayRule(), _day(RESTING, ramp)) == [7]


def test_the_threshold_keeps_its_margin_on_both_sides():
  """Why 50 and not the issue's example 30: the busiest resting minutes read
  +23 and +20 back to back, which 30 clears by only 1.3x, and a two-minute
  rule just under them fires on a process at rest (shown); three minutes
  in a row only widen it. 50 is ~2x from the resting pair and ~2x from the
  measured runaway; moving it is a decision."""
  assert vitals.RUNAWAY_MB_PER_MIN > 22.9 * 2
  assert 115.0 > vitals.RUNAWAY_MB_PER_MIN * 2
  assert _feed(RunawayRule(mb_per_min=19.0, minutes=2), _day(RESTING)) != []
  assert _feed(RunawayRule(mb_per_min=19.0), _day(RESTING)) == []


def test_a_runaway_fires_once_at_its_third_minute():
  """The a803c83 deaths: ~115 MiB a minute for eight minutes to the cap."""
  series = _day(RESTING[:10] + [115.0] * 8)
  first_fast = len(STARTUP) + 10             # the sample closing minute one
  assert _feed(RunawayRule(), series) == [first_fast + 2]


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
  sim = iter(1234.5 + 45.0 * i for i in range(100))
  dog.sim_time = lambda: next(sim)
  held = []
  try:
    for i in range(len(series)):
      if dog.tick():
        break
      assert tracemalloc.is_tracing() == was_tracing, "never before the onset"
    assert i == 7 and tracemalloc.is_tracing()
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
  before_close = (tmp_path / "log.txt").read_text()
  dog.close("the test ended")
  out.close()
  text = (tmp_path / "log.txt").read_text()
  assert dog.ticks >= len(series) + 3
  assert "allocated since the onset is still held" in before_close, \
    "the THREAD reports, trace_s after the onset -- not close() at the end"
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
