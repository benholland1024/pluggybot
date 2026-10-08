import os
import subprocess
import tempfile

import pytest

# ---- rendering: one backend, and the run names the rasteriser ---------------
# BEFORE ANYTHING IMPORTS MUJOCO, which reads MUJOCO_GL once, at import (issue
# #440). Unset is GLFW, which needs a display, so the default is the backend
# CLAUDE.md runs the suite on; one already in the environment wins (osmesa
# renders as the deployed box does). ⚠ What a camera test measures is the
# RASTERISER's, not the backend's -- EGL is a GPU on one box and llvmpipe on
# another -- so the run ends by naming it (`test_render_determinism.py`
# records it).
os.environ.setdefault("MUJOCO_GL", "egl")
RASTERISERS: set[str] = set()


def pytest_runtest_logreport(report):
  RASTERISERS.update(value for key, value in report.user_properties if key == "rasteriser")


def pytest_terminal_summary(terminalreporter):
  drew = ", ".join(sorted(RASTERISERS)) or "not read in this run"
  terminalreporter.write_line(f"MUJOCO_GL={os.environ['MUJOCO_GL']}, rasteriser {drew}")


# ---- temporary files: in memory where there is room -------------------------
# A memory store a test leaves behind is closed by the cyclic collector
# WHEREVER it next runs, and its WAL checkpoint's fsync, on a spinning /tmp,
# stalled whatever test was running 0.4-4 s (#473: 29-39 s of worker time a
# run, and two timing tests failed on it); on tmpfs an fsync is free, and the
# suite ran 53 s -> 43 s. A run writes ~11 MB, and pytest keeps three. Set
# before pytest names its base temp (lazily) and before xdist starts the
# workers, who inherit it. A TMPDIR already set wins, and a /dev/shm with less
# than `SHM_FREE_GB` free (a container's is 64 MB) is left alone.
SHM = "/dev/shm"
SHM_FREE_GB = 1.0


def memory_temp(environ, shm: str = SHM, free_gb: float = SHM_FREE_GB) -> str | None:
  """Where the suite's temporary files go: a directory of this user's in
  `shm`, where nothing has said otherwise and it has `free_gb` free, else
  None (the platform's default)."""
  if environ.get("TMPDIR") or not os.path.isdir(shm) or not os.access(shm, os.W_OK):
    return None
  free = os.statvfs(shm)
  if free.f_bavail * free.f_frsize < free_gb * 2**30:
    return None
  return os.path.join(shm, f"pluggybot-tests-{os.getuid()}")


if (_memory := memory_temp(os.environ)) is not None:
  os.makedirs(_memory, mode=0o700, exist_ok=True)
  os.environ["TMPDIR"] = _memory
  tempfile.tempdir = None           # read again: capture has opened its files already


# ---- endurance: the flights, flown when a change needs one ------------------
# A flight goes behind `--endurance` only when what it proves is PHYSICS no
# fast test can make (docs/Testing.md §1): the mind, the economy, the record
# and the wire fly on the stub in the default run. Each one names what it
# guards -- `@pytest.mark.endurance(when=(...))`, repo paths as prefixes --
# and `--endurance-changed` flies exactly the ones a change touches: the diff
# against the merge base with origin/staging, uncommitted and untracked
# included. `--endurance` flies every one. ⚠ A flight is test time like any
# other: each is approved by Ben with its time and its reason
# (`APPROVED_FLIGHTS`, tests/test_endurance.py, which fences the marks).
#
# ⚠ OPT-IN, NOT A MARKER EXPRESSION (issue #158). `addopts = "-m 'not
# endurance'"` does not compose: pytest keeps the LAST `-m`, so the everyday
# `-m "not slow"` would silently switch these back ON. A flag plus a skip
# applied at collection cannot be defeated by another `-m`.
ENDURANCE_OPT = "--endurance"
CHANGED_OPT = "--endurance-changed"
BASE_OPT = "--endurance-base"


def pytest_addoption(parser):
  parser.addoption(ENDURANCE_OPT, action="store_true", default=False,
                   help="fly every endurance proof (minutes each)")
  parser.addoption(CHANGED_OPT, action="store_true", default=False,
                   help="fly the endurance proofs whose `when` paths this "
                        "branch touches")
  parser.addoption(BASE_OPT, default="origin/staging",
                   help=f"what {CHANGED_OPT} diffs against")


def changed_paths(root, base: str) -> set[str]:
  """Every path the working tree differs in from its merge base with `base`:
  committed, staged, unstaged and untracked, and BOTH ends of a rename (a
  file moved out of a flight's directory is a change to it)."""
  def git(*args) -> list[str]:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if out.returncode:
      raise pytest.UsageError(f"{CHANGED_OPT}: git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.splitlines()
  since = git("merge-base", base, "HEAD")[0]
  return set(git("diff", "--name-only", "--no-renames", since)) | set(
    git("ls-files", "--others", "--exclude-standard"))


def guarded(when, changed) -> list[str]:
  """The changed paths a flight's `when` covers; each entry is a prefix."""
  return sorted(path for path in changed if any(path.startswith(w) for w in when))


def pytest_collection_modifyitems(config, items):
  if config.getoption(ENDURANCE_OPT):
    return
  changed = (changed_paths(config.rootpath, config.getoption(BASE_OPT))
             if config.getoption(CHANGED_OPT) else None)
  for item in items:
    mark = item.get_closest_marker("endurance")
    if mark is None:
      continue
    if changed is None:
      item.add_marker(pytest.mark.skip(
        reason=f"endurance proof: {CHANGED_OPT} flies it when a change needs it"))
    elif not guarded(mark.kwargs.get("when", ()), changed):
      item.add_marker(pytest.mark.skip(
        reason="endurance proof: this change touches nothing it guards"))
