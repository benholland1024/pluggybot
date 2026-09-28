"""The endurance flights and when they fly (tests/conftest.py): each names
what it guards, and `--endurance-changed` flies exactly the ones a change
touches. Nothing here flies; the fence reads the marks, the selector's two
rules are pinned on a direct call and a scratch repository."""

import ast
import os
import subprocess
from pathlib import Path

import pytest

import conftest

TESTS = Path(__file__).parent
ROOT = TESTS.parent


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
