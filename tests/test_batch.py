"""A batch of flights a person can see into (#684; `pluggybot.batch`):
every flight's output as it is written and its stacks on SIGUSR1, so a
flight gone quiet says where it is -- the drift batch's 13h read as a C
call for an hour, its stdout empty under a file's block buffer.

Pinned in processes of their own, the signal sent for real: a runner that
called `watched` and a flight it started each print every thread's stack
on SIGUSR1 and go on, the flight's output unbuffered; a python of ours
outside a watched batch still dies of SIGUSR1, as any python does. And
every script that starts a flight in a process of its own is a watched
runner, and a pod's batch watches its own.
"""

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from pluggybot.batch import STACKS_ENV

ROOT = Path(__file__).resolve().parents[1]

#: A flight: SIGUSR1 to itself, then says whether it is still here and
#: whether its stdout is unbuffered.
FLIGHT = ("import os, signal, sys, pluggybot; os.kill(os.getpid(), signal.SIGUSR1); "
          "print('flew on', type(sys.stdout.buffer).__name__)")
#: A runner: `watched` -- its `sys.stderr` one with no file descriptor, as
#: under pytest's capture -- SIGUSR1 to itself, then the flight.
RUNNER = f"""
import io, os, signal, subprocess, sys
from pluggybot.batch import watched
sys.stderr = io.StringIO()
watched()
os.kill(os.getpid(), signal.SIGUSR1)
flight = subprocess.run([sys.executable, "-c", {FLIGHT!r}], capture_output=True, text=True)
print("ran on")
print(flight.returncode, flight.stdout.strip())
print("flight stacks:", "most recent call first" in flight.stderr)
"""


def _python(code: str, **env) -> subprocess.CompletedProcess:
  clean = {k: v for k, v in os.environ.items() if k not in (STACKS_ENV, "PYTHONUNBUFFERED")}
  return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        env={**clean, **env}, timeout=60)


def test_a_watched_runner_and_the_flights_it_starts_say_their_stacks_on_sigusr1_and_go_on():
  r = _python(RUNNER)
  assert r.returncode == 0, r.stderr[-800:]
  out = r.stdout.split("\n")
  assert out[0] == "ran on" and "most recent call first" in r.stderr, "the runner's stacks"
  assert out[1] == "0 flew on FileIO", "the flight lived, its stdout unbuffered"
  assert out[2] == "flight stacks: True"


def test_outside_a_watched_batch_sigusr1_is_what_it_always_was():
  r = _python(FLIGHT)
  assert r.returncode == -signal.SIGUSR1 and "flew on" not in r.stdout


#: How a script starts a flight in a process of its own.
FORKS = re.compile(r"\bPool\(|\brun_pool\(|\[sys\.executable,\s*__file__")


def test_every_script_that_starts_flights_is_a_watched_runner_and_so_is_a_pods_batch():
  forkers = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if FORKS.search(p.read_text()))
  assert len(forkers) >= 10, forkers
  unwatched = [name for name in forkers
               if not re.search(r"^\s+watched\(\)$", (ROOT / "scripts" / name).read_text(), re.M)]
  assert unwatched == [], f"start flights, unwatched: {unwatched}"
  batch = (ROOT / "training" / "pod.sh").read_text().split("\n  batch)")[1].split(";;")[0]
  assert "PYTHONUNBUFFERED=1" in batch and f"{STACKS_ENV}=1" in batch
