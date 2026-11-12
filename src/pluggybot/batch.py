"""A batch of flights a person can see into (#684): every flight's output as
it is written, and every thread's stack on SIGUSR1.

A drift batch's flight ran 57 and 72 minutes on a pod with nothing on its
stdout or stderr, and read as a C call that never returned: stdout to a
file is block-buffered, and a flight's whole report sat under the buffer
until it ended. SimNotes, "A flight that never got up", is the story.

A batch runner calls `watched` first. The processes it starts inherit its
environment, so each prints as it goes (`PYTHONUNBUFFERED`) and, once it
imports this package (`pluggybot/__init__.py`), prints every thread's
stack to its stderr on `kill -USR1 PID`: faulthandler needs no ptrace,
which a pod's container has none of. A pod's batch sets both itself
(`training/pod.sh batch`).
"""

import faulthandler
import os
import signal
import sys

#: In a process's environment, every process of ours started under it
#: prints every thread's stack on SIGUSR1 (`stacks_on_sigusr1`).
STACKS_ENV = "PLUGGY_STACKS_ON_SIGUSR1"


def stacks_on_sigusr1() -> None:
  """Every thread's stack to stderr on SIGUSR1, the process going on: a
  flight gone quiet says where it is (`kill -USR1` its python, never the
  shell that waits on it). ⚠ To the process's own stderr (faulthandler
  writes to a file descriptor, and a replaced `sys.stderr` -- pytest's --
  may have none); a process with none goes without."""
  try:
    faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True)
  except (AttributeError, OSError, ValueError):
    pass


def watched() -> None:
  """This process and every flight it starts: their output as it is
  written, and their stacks on SIGUSR1 -- this one's from now, theirs by
  their environment."""
  os.environ["PYTHONUNBUFFERED"] = "1"
  os.environ[STACKS_ENV] = "1"
  for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
      reconfigure(line_buffering=True)
  stacks_on_sigusr1()
