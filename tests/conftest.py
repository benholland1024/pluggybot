import subprocess

import pytest


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
