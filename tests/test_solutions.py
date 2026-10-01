"""Ladder A of issue #264: what a hand-written solution stands on, kept
while the challenges wait for legs (#407) -- and the harness that flies
the one feature legs have, the mouse's feed (`scripts/solve.py`).

The rules, each pinned without a mission (docs/Testing.md):

  1. every tool a run took off the rack is one it must hang back
     (`scoring.fetched_tools`, the procedure's `toolsHung`);
  2. nothing under `mind/` imports a challenge's solution;
  3. an offer says where the house set its props out, and a bare one
     carries no clause;
  4. the prompt says what a bare `return` does, and the lab's rule names
     the way there;
  5. the pair harness hooks its board as the constructor does.
"""

import ast
from pathlib import Path

from pluggybot.economy import scoring

SRC = Path(__file__).parent.parent / "src" / "pluggybot"


def test_every_tool_a_run_took_off_the_rack_is_one_it_must_hang_back():
  """`toolsHung` (the procedure's grade) is over every tool the run took
  off the rack -- each `fetch` that landed -- and never one a failed fetch
  left hanging."""
  run = {"steps": [{"verb": "fetch", "tool": "module_pen", "ok": True},
                   {"verb": "fetch", "tool": "module_lcd", "ok": False},
                   {"verb": "fetch", "tool": "module_claw", "ok": True},
                   {"verb": "wait", "ok": True}]}
  assert scoring.fetched_tools(run) == ["module_pen", "module_claw"]


def test_the_mind_never_imports_the_solutions():
  for path in (SRC / "mind").rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      names = ([a.name for a in node.names] if isinstance(node, ast.Import)
               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
      assert not any("solutions" in n for n in names), path
  for path in (SRC / "mind").rglob("*.py"):
    assert "solutions" not in path.read_text(), path


def test_the_offers_say_where_the_house_set_the_props_out(tmp_path):
  """Every ladder-B day that reached a challenge wrote a blind scout first:
  the offer said "in the lab" and nothing more. The house placed the
  cubes, so where they START is a work-order fact (TaskPattern §2), built
  by the producer off `world_config`; a bare offer (a test's) carries no
  clause and no stray placeholder."""
  from pluggybot.economy.tasks import TaskBoard
  from pluggybot.lifecycle import QUAD_HOME, task_producer
  board = TaskBoard(path=str(tmp_path / "t.json"))
  producer = task_producer(board, QUAD_HOME, procedures=True)
  params, secret = producer._build("find_mass", "lab")
  bench = board.offer("find_mass", "lab", params=params, secret=secret, t=1.0).description
  assert "(26.68, 2.00) (tag 24)" in bench and "(26.68, 1.00) (tag 23)" in bench
  assert bench.endswith("write the procedure.")
  bare = board.offer("stack_tower", "workshop", t=2.0).description
  assert "{" not in bare and "  " not in bare and "None" not in bare
  # ...and the cubes' LIVE poses stay out of the context (issue #227's
  # rule): the offer says where they were set out, the room says nothing
  assert "mass" not in str(producer.facts.get("lab", {}))


def test_the_prompt_says_what_a_bare_return_does_and_the_lab_rule_names_the_route():
  from pluggybot.mind import overseer as ov
  assert "`return` (alone" in ov.procedure_rule()
  assert "route" in ov.lab_rule("lab")


# ---- issue #353: the pair harness a robot's own procedure is replayed in ------


def test_the_pair_harness_hooks_its_board_as_the_constructor_does(tmp_path):
  """Review of #353: `solve.build_pair_lives` hands the flying robot a task
  board after it is built, so `HubLifecycle.__init__` never hooked it -- an
  offer set out no props and no bench mass, and a flight on a challenge
  would be graded against a world the offer never changed."""
  import sys
  sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
  import solve as demo
  lives, life = demo.build_pair_lives(2, str(tmp_path))
  try:
    assert life._bench_offered in life.tasks.on_event
  finally:
    for each in lives:
      each.body.close()
