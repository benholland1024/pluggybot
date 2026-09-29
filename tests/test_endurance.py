"""The endurance flights, who approved them and when they fly
(tests/conftest.py): each is on Ben's approved list with its time and its
reason, each names what it guards, and `--endurance-changed` flies exactly
the ones a change touches. Nothing here flies; the fences read the source,
the selector's two rules are pinned on a direct call and a scratch
repository."""

import ast
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

import conftest

TESTS = Path(__file__).parent
ROOT = TESTS.parent

#: ⚠ EVERY TEST OUTSIDE THE DEFAULT RUN, EACH APPROVED BY BEN (CLAUDE.md: the
#: test budget has no outside): name -> (the wall time it takes, why no test
#: in the default run can make its claim, the PR he approved it in). A flight
#: not listed here fails the fence below, so adding one -- or moving a test
#: out of the default run -- is adding it here and asking him in its PR.
APPROVED_FLIGHTS = {
  "test_a_low_quadruped_walks_to_its_dock_and_charges_on_it": (
    "~17 s: 16.5-18.1 s -n0, measured 2026-09-29", "the served robot walks "
    "the house to its dock and charges on it: the docking composed, on the "
    "loop's own charge path", "#413"),
  "test_a_fresh_quadruped_walks_to_the_kitchen_by_the_hall": (
    "~24 s: 23.2-23.8 s -n0, measured 2026-09-29", "the planner, the scan "
    "matcher and the walking policy, into the unknown", "#413"),
  "test_the_tower_is_stacked_by_the_claw_from_the_rack_and_graded": (
    "~170 s", "ladder A: the tower can be solved (Evaluation.md §7)", "#413"),
  "test_the_unknown_mass_is_weighed_on_the_lift_and_the_finding_graded": (
    "~150 s", "ladder A: the bench can be solved", "#413"),
  "test_a_feed_act_reaches_the_cage_and_the_mouse_eats": (
    "~45 s", "ladder A: the mouse's feed act lands", "#413"),
  "test_the_paid_feed_is_done_on_legs_by_the_pair": (
    "~110 s: 108 s -n0 on a busy box, measured 2026-09-28", "ladder A on legs "
    "(Evaluation.md §7): the one job the quadrupeds are offered, walked by "
    "the served pair across the street to the lab and graded by `eval_feed`",
    "#414"),
  "test_the_served_quadruped_fetches_a_tool_and_hangs_it_back": (
    "~30 s: 30.0 s -n0, measured 2026-09-29", "the served robot's swap "
    "composed: the walk to the bay, its tags, the walk-in and settle, the "
    "fork under the peg, the carry and the put, on the real body", "#417"),
}


def _is_mark(node, name="endurance") -> bool:
  return (isinstance(node, ast.Attribute) and node.attr == name
          and isinstance(node.value, ast.Attribute) and node.value.attr == "mark")


def _paths(node, consts: dict) -> list:
  """A `when` read off the syntax tree: strings, tuples of them, and a
  module-level tuple by name, starred or not."""
  if isinstance(node, ast.Constant):
    return [node.value]
  if isinstance(node, ast.Tuple):
    return [p for e in node.elts for p in _paths(e, consts)]
  if isinstance(node, ast.Starred):
    return _paths(node.value, consts)
  if isinstance(node, ast.Name) and node.id in consts:
    return _paths(consts[node.id], consts)
  raise AssertionError(f"a `when` the fence cannot read: {ast.unparse(node)}")


def _flights() -> dict:
  """Every flight's name -> (its `when`, whether it is also `slow`). A mark
  that is not a decorator CALL -- bare, or a `pytest.param` mark -- is
  refused, because it could name nothing."""
  found = {}
  for path in sorted(TESTS.glob("test_*.py")):
    text = path.read_text()
    if "mark.endurance" not in text:
      continue
    tree = ast.parse(text, filename=str(path))
    consts = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name)}
    marks = [n for n in ast.walk(tree) if _is_mark(n)]
    called = 0
    for node in ast.walk(tree):
      if not isinstance(node, ast.FunctionDef):
        continue
      for dec in node.decorator_list:
        if isinstance(dec, ast.Call) and _is_mark(dec.func):
          called += 1
          when = next((k.value for k in dec.keywords if k.arg == "when"), None)
          found[node.name] = (_paths(when, consts) if when is not None else [],
                              any(_is_mark(d, "slow") for d in node.decorator_list))
    assert called == len(marks), \
        f"{path.name}: an endurance mark that is not `@pytest.mark.endurance(when=...)`"
  return found


def test_every_flight_names_what_it_guards_and_each_path_is_in_the_tree():
  """A flight that names nothing is never flown by `--endurance-changed`, and
  a path that has moved disarms it silently -- so a rename fails here, where
  the flight is still a line away. `endurance` is a subset of `slow`
  (pyproject.toml)."""
  tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
  flights = _flights()
  assert flights, "no flight found: the fence is reading nothing"
  for name, (when, slow) in flights.items():
    assert when and all(isinstance(w, str) for w in when), f"{name} names no `when`"
    for w in when:
      assert any(t.startswith(w) for t in tracked), f"{name}: no tracked path is {w!r}"
    assert slow, f"{name} is not marked slow"


def test_nothing_leaves_the_default_run_without_bens_approval():
  """⚠ THE BUDGET HAS NO OUTSIDE (CLAUDE.md). A flight is the one way a test
  leaves the default run, and each is in `APPROVED_FLIGHTS` with its time
  and its reason: a new one fails here until it is listed, and a listed one
  that is gone fails too, so the list stays the set. No test skips itself,
  and `addopts` deselects nothing -- the other two ways out."""
  flights = set(_flights())
  listed = set(APPROVED_FLIGHTS)
  assert not flights - listed, (
    f"outside the default run without Ben's approval: {sorted(flights - listed)} "
    "-- list each in APPROVED_FLIGHTS with its wall time and why no test in the "
    "default run can make its claim, and ask for his approval in the PR")
  assert not listed - flights, f"approved, but no longer flights: {sorted(listed - flights)}"
  for name, (took, why, pr) in APPROVED_FLIGHTS.items():
    assert took and why and pr.startswith("#"), f"{name}: its time, its reason, its PR"
  for path in sorted(TESTS.glob("test_*.py")):
    text = path.read_text()
    if path.name == Path(__file__).name or not any(
        form in text for form in ("pytest.skip", "mark.skip", "importorskip")):
      continue
    for node in ast.walk(ast.parse(text, filename=str(path))):
      if (isinstance(node, ast.Attribute)
          and node.attr in ("skip", "skipif", "importorskip")
          and ast.unparse(node.value) in ("pytest", "pytest.mark")):
        raise AssertionError(f"{path.name}:{node.lineno}: `{ast.unparse(node)}` takes a "
                             "test out of the default run -- that needs Ben's approval")
  options = tomllib.loads((ROOT / "pyproject.toml").read_text())
  addopts = options["tool"]["pytest"]["ini_options"]["addopts"].split()
  assert not {"-m", "-k", "--deselect", "--ignore", "--ignore-glob"} & set(addopts), \
      f"addopts {addopts} takes tests out of the default run -- that needs Ben's approval"


def test_a_flight_is_flown_only_when_a_change_touches_what_it_guards():
  when = ("src/pluggybot/legs/", "models/quadruped")
  assert conftest.guarded(when, {"src/pluggybot/legs/dock.py", "README.md"}) == [
    "src/pluggybot/legs/dock.py"]
  assert conftest.guarded(when, {"models/quadruped_getup.npz"}), "a prefix, not a name"
  assert not conftest.guarded(when, {"src/pluggybot/lifecycle.py", "tests/test_legs.py"})
  assert not conftest.guarded((), {"src/pluggybot/legs/dock.py"}), "names nothing, flies never"


def test_the_change_is_the_branch_and_the_working_tree(tmp_path):
  """Against the MERGE BASE, so what landed on the base since is not this
  change's; the working tree counts -- committed, staged, unstaged and
  untracked -- because the flights are asked for while the work is live; and
  a rename is both its ends, or a file moved out of `legs/` flies nothing."""
  env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
         "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}

  def git(*args):
    subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True,
                   capture_output=True)

  def write(name, text="x"):
    (tmp_path / name).write_text(text)

  git("init", "-q", "-b", "base")
  for name in ("kept.py", "edited.py", "staged.py", "moved.py"):
    write(name, f"# {name}\n" * 20)
  git("add", ".")
  git("commit", "-q", "-m", "base")
  git("checkout", "-q", "-b", "work")
  write("committed.py")
  git("add", "committed.py")
  git("mv", "moved.py", "renamed.py")
  git("commit", "-q", "-m", "work")
  git("checkout", "-q", "base")
  write("landed_on_base.py")
  git("add", "landed_on_base.py")
  git("commit", "-q", "-m", "moved on")
  git("checkout", "-q", "work")
  write("staged.py", "y")
  git("add", "staged.py")
  write("edited.py", "y")
  write("untracked.py")
  assert conftest.changed_paths(tmp_path, "base") == {
    "committed.py", "moved.py", "renamed.py", "staged.py", "edited.py",
    "untracked.py"}
  with pytest.raises(pytest.UsageError):
    conftest.changed_paths(tmp_path, "no-such-branch")
