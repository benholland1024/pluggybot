"""The served process's memory, watched from inside it (issue #349).

The deployed pair stopped mid-mission several times a day without a word:
its memory ran away to the container's 2 GiB cap and the kernel killed it,
and a process says nothing about its own SIGKILL. So the watchdog speaks
BEFORE the kill: one line a minute of resident memory, and when that climbs
like a runaway, every thread's stack and -- TRACE_S later -- what was
allocated in between, once per episode. docs/Webserver.md, "When the
process dies", is the story.

`RunawayRule` is the trigger and is pure (one sample in, onset or not out),
so it is pinned with a made-up series; `Watchdog` reads the memory, prints,
and does the dumps, on a daemon thread.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import tracemalloc

#: Wall seconds between samples, and between the lines they print.
SAMPLE_S = 60.0
#: A RUNAWAY: resident memory growing faster than this, in MiB a minute, for
#: RUNAWAY_MINUTES samples in a row. MEASURED on the deployed pair (docker
#: stats, once a minute): both processes that died on 2026-09-24 (a803c83)
#: grew 115-120 MiB a minute for 7-9 minutes first; at rest a process
#: averages 2.7 MiB a minute, and its busiest resting minutes read +23 and
#: +20 back to back (3518953, 2026-09-25), which a 30 would clear by only
#: 1.3x. 50 sits ~2x from each.
RUNAWAY_MB_PER_MIN = 50.0
RUNAWAY_MINUTES = 2
#: Samples after start that judge nothing: the world is built and carried on
#: from then (212 -> 788 MiB in the deployed process's first minute), and a
#: slower box straddles two samples doing it.
WARMUP_MINUTES = 3
#: Wall seconds of tracemalloc between the onset and the allocation report.
#: It starts AT the onset, never at boot: MEASURED, tracing from boot ran the
#: pair at ~0.2x real time against ~0.9x, and a runaway still growing is
#: growing in what it traces. Short, because the report is ~3 s of Python per
#: million live traces, taken from the physics thread's share of the GIL
#: (filtering the snapshot cost 11 s more per million, so it is not
#: filtered): 20 s of the measured runaway is at most ~38 MiB.
TRACE_S = 20.0
#: Frames kept per traced allocation, and how many of the largest are shown.
TRACE_FRAMES = 8
TOP_ALLOCATIONS = 12

_MIB = float(1 << 20)


def rss_mib() -> float | None:
  """This process's resident memory in MiB, or None where /proc is not."""
  try:
    with open("/proc/self/statm") as f:
      pages = int(f.read().split()[1])
  except (OSError, ValueError, IndexError):
    return None
  return pages * os.sysconf("SC_PAGE_SIZE") / _MIB


class RunawayRule:
  """When to look, from one resident-memory sample a minute.

  `sample` is True at the ONSET: the growth has run over the rate for
  `minutes` samples in a row. ONCE PER EPISODE: another onset needs
  `minutes` samples under the rate first, or one long runaway would dump the
  stacks every minute until it was killed.
  """

  def __init__(self, mb_per_min: float = RUNAWAY_MB_PER_MIN,
               minutes: int = RUNAWAY_MINUTES,
               warmup: int = WARMUP_MINUTES) -> None:
    self.mb_per_min, self.minutes, self.warmup = mb_per_min, minutes, warmup
    self.samples = 0
    self.last: tuple[float, float] | None = None
    self.rate: float | None = None
    self.above = self.below = 0
    self.episode = False

  def sample(self, wall_s: float, mib: float | None) -> bool:
    self.samples += 1
    self.rate = None
    if mib is None:
      return False
    if self.last is not None and wall_s > self.last[0]:
      self.rate = (mib - self.last[1]) * 60.0 / (wall_s - self.last[0])
    self.last = (wall_s, mib)
    if self.rate is None or self.samples <= self.warmup:
      return False
    if self.rate > self.mb_per_min:
      self.above, self.below = self.above + 1, 0
    else:
      self.above, self.below = 0, self.below + 1
    if self.episode:
      if self.below >= self.minutes:
        self.episode = False
      return False
    if self.above >= self.minutes:
      self.episode = True
      return True
    return False


class Watchdog:
  """Samples the memory on a daemon thread and dumps at a runaway's onset.

  `start()` also turns `faulthandler` on for a fatal signal (a segfault in
  MuJoCo or osmesa prints every thread's stack) where nothing has yet;
  `close(why)` prints the exit line. `sim_time` is a callable the server sets
  once the world exists, read for the line and never trusted to answer.
  """

  def __init__(self, *, out=None, sample_s: float = SAMPLE_S,
               trace_s: float = TRACE_S, rule: RunawayRule | None = None,
               read_rss=rss_mib, clock=time.monotonic) -> None:
    self.out = out
    self.sample_s, self.trace_s = sample_s, trace_s
    self.rule = rule if rule is not None else RunawayRule()
    self.read_rss = read_rss
    self.clock = clock
    self.sim_time = None
    self.peak: float | None = None
    self.ticks = 0
    self._t0 = clock()
    self._tracing = False
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> "Watchdog":
    if not faulthandler.is_enabled() and sys.__stderr__ is not None:
      faulthandler.enable(file=sys.__stderr__, all_threads=True)
    self._thread = threading.Thread(target=self._run, name="vitals",
                                    daemon=True)
    self._thread.start()
    return self

  def close(self, why: str) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=5.0)
    if self._tracing:
      tracemalloc.stop()
      self._tracing = False
    mib = self.read_rss()
    self._say(f"vitals: exiting -- {why}; rss {_mb(mib)}, peak "
              f"{_mb(_max(self.peak, mib))}, up "
              f"{(self.clock() - self._t0) / 60.0:.0f} min")

  def _run(self) -> None:
    while True:
      try:
        if self.tick() and not self._stop.wait(self.trace_s):
          self.report()
      except Exception as e:                # noqa: BLE001 -- said, not swallowed
        self._say(f"vitals: the watchdog failed this sample -- "
                  f"{type(e).__name__}: {e}")
      if self._stop.wait(self.sample_s):
        return

  def tick(self) -> bool:
    """One sample and its line; at an onset, the stacks and the tracing."""
    mib = self.read_rss()
    onset = self.rule.sample(self.clock(), mib)
    self.ticks += 1
    self.peak = _max(self.peak, mib)
    rate = self.rule.rate
    self._say(f"vitals: rss {_mb(mib)}"
              + (f" ({rate:+.1f} MiB/min)" if rate is not None else "")
              + f", peak {_mb(self.peak)}" + self._clock_note())
    if onset:
      self._say(f"vitals: RUNAWAY -- resident memory grew over "
                f"{self.rule.mb_per_min:.0f} MiB/min for {self.rule.minutes} "
                f"minutes (now {rate:+.0f}); every thread's stack follows, and "
                f"in {self.trace_s:.0f} s what was allocated in between")
      faulthandler.dump_traceback(file=self._stream(), all_threads=True)
      if not tracemalloc.is_tracing():
        tracemalloc.start(TRACE_FRAMES)
        self._tracing = True
    return onset

  def report(self) -> None:
    """The largest allocations since the onset, and the stacks again."""
    if tracemalloc.is_tracing():
      stats = tracemalloc.take_snapshot().statistics("traceback")
      held = sum(s.size for s in stats)
      self._say(f"vitals: {held / _MIB:.1f} MiB allocated since the onset is "
                f"still held; the {TOP_ALLOCATIONS} largest, newest frame "
                "first:")
      for s in stats[:TOP_ALLOCATIONS]:
        self._say(f"  {s.size / _MIB:.1f} MiB in {s.count} block(s)")
        for line in s.traceback.format(most_recent_first=True):
          self._say(f"    {line.strip()}")
    else:
      self._say("vitals: tracemalloc was stopped before the report")
    self._say(f"vitals: every thread's stack, {self.trace_s:.0f} s on:")
    faulthandler.dump_traceback(file=self._stream(), all_threads=True)
    if self._tracing:
      tracemalloc.stop()
      self._tracing = False

  def _clock_note(self) -> str:
    try:
      return f", t={float(self.sim_time()):.1f} s" if self.sim_time else ""
    except Exception:                       # noqa: BLE001 -- a line, not a check
      return ""

  def _stream(self):
    return self.out if self.out is not None else sys.stdout

  def _say(self, line: str) -> None:
    # One write, newline included: `print` writes the two apart, and this
    # thread's line would land inside a narration line between them.
    stream = self._stream()
    stream.write(line + "\n")
    stream.flush()


def why_of(exc: BaseException | None) -> str:
  """What `close` says ended the process."""
  if exc is None:
    return "the run ended"
  if isinstance(exc, SystemExit):
    return f"exit {exc.code}"
  return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _max(a: float | None, b: float | None) -> float | None:
  return b if a is None else a if b is None else max(a, b)


def _mb(mib: float | None) -> str:
  return "unknown" if mib is None else f"{mib:.0f} MiB"
