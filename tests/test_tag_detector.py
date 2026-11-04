"""The one AprilTag detector a process holds (`perception.tags`), and its end:
pupil-apriltags' `Detector.__del__` frees its tag families before the
detector, whose destroy then reads them, so a process that made one could
segfault as it exited (#560). Pinned in a child process with glibc told to
overwrite what it frees, where that use-after-free crashes every time."""

import os
import subprocess
import sys

#: What every flight does: the shared detector made and used, then exit.
SCRIPT = ("import numpy as np\n"
          "from pluggybot.perception import tags\n"
          "tags._shared_detector().detect(np.zeros((480, 640), dtype=np.uint8))\n")


def test_a_process_that_made_the_detector_exits_cleanly_whatever_its_freed_memory_holds():
  # glibc's own switches: no per-thread cache, and every freed block filled
  # with 0xa5 -- what another thread's allocation left there at a real exit
  env = {**os.environ, "GLIBC_TUNABLES": "glibc.malloc.tcache_count=0", "MALLOC_PERTURB_": "165"}
  done = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True,
                        text=True, timeout=120)
  assert done.returncode == 0, done.stderr[-2000:]
