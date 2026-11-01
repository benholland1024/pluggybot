"""Ladder A of issue #264: what a hand-written solution stands on -- the
humanoids' hand jobs' (#559: `challenge/solutions.py`, flown by
`scripts/solve.py --feature drawer|door|cube|chest`).

The rules, each pinned without a mission (docs/Testing.md):

  1. nothing under `mind/` imports a challenge's solution;
  2. an offer says where by its area's terms -- the building's address and
     the area's directions -- never a position, and a bare one carries no
     stray clause;
  3. the prompt says what a bare `return` does.
"""

import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "pluggybot"


def test_the_mind_never_imports_the_solutions():
  for path in (SRC / "mind").rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
      names = ([a.name for a in node.names] if isinstance(node, ast.Import)
               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
      assert not any("solutions" in n for n in names), path
  for path in (SRC / "mind").rglob("*.py"):
    assert "solutions" not in path.read_text(), path


def test_the_offers_say_where_by_their_areas_terms_and_never_a_position(tmp_path):
  """Every ladder-B day that reached a challenge wrote a blind scout first:
  the offer said "in the lab" and nothing more. The rover's offers gave the
  props' start positions; a job gives at most its building's address and
  the area's directions (#419): each hand job's offer carries its own
  area's terms, its description no position, and a bare offer no stray
  placeholder."""
  import re

  from pluggybot.economy.tasks import KINDS, TaskBoard
  from pluggybot.home import places as addresses
  from pluggybot.lifecycle import HUMANOID_HOME, task_producer
  board = TaskBoard(path=str(tmp_path / "t.json"))
  producer = task_producer(board, HUMANOID_HOME, procedures=True)
  offered = list(producer.cadence.kinds)
  assert offered
  for kind in offered:
    area = KINDS[kind].target_kind
    params, secret = producer._build(kind, area)
    job = board.offer(kind, area, params=params, secret=secret, t=1.0)
    shown = job.as_context(1.0)
    assert shown["directions"] == addresses.area(area)["directions"], kind
    assert shown["address"] == addresses.area(area)["address"], kind
    assert not re.search(r"-?\d+\.\d+\s*,\s*-?\d+\.\d+", job.description), (
      kind, job.description)
    assert "write the procedure" in job.description, kind
  bare = board.offer(offered[0], KINDS[offered[0]].target_kind, t=2.0).description
  assert "{" not in bare and "  " not in bare and "None" not in bare
  # ...and the props' LIVE poses stay out of the context (issue #227's
  # rule): the producer is handed the places and nothing else
  assert set(producer.facts) == {"places"}


def test_the_prompt_says_what_a_bare_return_does():
  from pluggybot.mind import overseer as ov
  rule = ov.hands_procedure_rule()
  assert "`return` (alone)" in rule and "your variables are your readout" in rule
